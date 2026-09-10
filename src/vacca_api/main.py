"""VACCA Vision API — FastAPI microservice for cow detection.

Endpoints:
    GET  /health          — service health + model info
    POST /detect          — cow detection with bounding boxes
    POST /bcs             — discrete BCS category 1..5

The test UI at /ui is for PROTOTYPE VALIDATION ONLY.
Remove the /ui route and static/ directory before connecting to production backend.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from time import perf_counter
from typing import AsyncIterator, NoReturn

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from .detection import DEFAULT_MODEL, get_detector
from .schemas import (
    BCSReadinessResponse,
    BCSResponse,
    DetectResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
)
from .metrics import PrototypeMetrics
from .upload_validation import (
    UploadTooLargeError,
    UploadValidationError,
    read_validated_upload,
)

# --- Configuration ---
STATIC_DIR = Path(__file__).resolve().parent / "static"
MODEL_IDENTIFIER = DEFAULT_MODEL.name
logger = logging.getLogger(__name__)
_ERROR_RESPONSE = {"model": ErrorResponse}
DETECT_GATE_NAME = "detect-inference-capacity"
BCS_GATE_NAME = "bcs-inference-capacity"
INFERENCE_GATE_CAPACITY = 1
INFERENCE_GATE_ACQUIRE_TIMEOUT_SECONDS = 0.1
INFERENCE_CAPACITY_BUSY_DETAIL = "Inference capacity is busy; retry shortly"


class InferenceCapacityBusyError(RuntimeError):
    """Raised when one capability's bounded inference capacity is busy."""


class InferenceCapacityGate:
    """Own a bounded, deterministic application-level inference capacity."""

    def __init__(
        self,
        name: str = DETECT_GATE_NAME,
        capacity: int = INFERENCE_GATE_CAPACITY,
        acquisition_timeout: float = INFERENCE_GATE_ACQUIRE_TIMEOUT_SECONDS,
    ) -> None:
        if capacity < 1 or acquisition_timeout <= 0:
            raise ValueError("inference capacity and timeout must be positive")
        self.name = name
        self.capacity = capacity
        self.acquisition_timeout = acquisition_timeout
        self._semaphore = asyncio.BoundedSemaphore(capacity)

    async def acquire(self) -> None:
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(), timeout=self.acquisition_timeout
            )
        except TimeoutError:
            raise InferenceCapacityBusyError(self.name) from None

    def release(self) -> None:
        self._semaphore.release()


def _raise_upload_http_exception(exc: UploadValidationError) -> NoReturn:
    """Log and map a validated-upload failure to its public HTTP contract."""

    if isinstance(exc, UploadTooLargeError):
        logger.warning("Rejected image upload: %s", exc.event)
        raise HTTPException(status_code=413, detail=str(exc)) from None
    logger.info("Rejected image upload: %s", exc.event)
    raise HTTPException(status_code=400, detail=str(exc)) from None

def _startup(target_app: FastAPI | None = None) -> None:
    # Pre-load the model so first request is fast
    try:
        detector = get_detector()
    except Exception as exc:
        logger.error("Model startup failed: %s", type(exc).__name__)
        raise
    if target_app is None:
        target_app = app
    target_app.state.detector = detector
    logger.info("Model loaded: %s", MODEL_IDENTIFIER)


@asynccontextmanager
async def lifespan(target_app: FastAPI) -> AsyncIterator[None]:
    _startup(target_app)
    yield


# --- App ---
app = FastAPI(
    title="VACCA Vision API",
    description="Cow detection microservice — VACCA Fase 1",
    version="0.1.0",
    lifespan=lifespan,
)
app.state.detect_inference_gate = InferenceCapacityGate(DETECT_GATE_NAME)
app.state.bcs_inference_gate = InferenceCapacityGate(BCS_GATE_NAME)
app.state.metrics = PrototypeMetrics()


# --- Health ---
@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    detector = request.app.state.detector
    return HealthResponse(
        model_loaded=True,
        model_path=MODEL_IDENTIFIER,
        gpu_available=detector.gpu_available,
    )


# --- Detect ---
@app.post(
    "/detect",
    response_model=DetectResponse,
    responses={
        400: _ERROR_RESPONSE,
        413: _ERROR_RESPONSE,
        500: _ERROR_RESPONSE,
        503: {
            **_ERROR_RESPONSE,
            "description": "Inference capacity is busy",
        },
    },
)
async def detect(request: Request, file: UploadFile = File(...)) -> DetectResponse:
    """Receive an image and return cow detections with bounding boxes."""
    request_started = perf_counter()
    metrics = request.app.state.metrics
    metrics.request("detect")
    try:
        image_bytes = await read_validated_upload(file)
    except UploadValidationError as exc:
        metrics.client_rejection("detect", (perf_counter() - request_started) * 1000)
        _raise_upload_http_exception(exc)

    gate = request.app.state.detect_inference_gate
    try:
        await gate.acquire()
    except InferenceCapacityBusyError:
        metrics.busy("detect", (perf_counter() - request_started) * 1000)
        raise HTTPException(status_code=503, detail=INFERENCE_CAPACITY_BUSY_DETAIL) from None
    started = perf_counter()
    try:
        try:
            detector = request.app.state.detector
            detections, img_w, img_h, elapsed_ms = await run_in_threadpool(
                detector.detect,
                image_bytes,
            )
        except Exception as exc:
            metrics.inference_failure(
                "detect",
                (perf_counter() - started) * 1000,
                (perf_counter() - request_started) * 1000,
            )
            logger.error("Detection request failed: %s", type(exc).__name__)
            raise HTTPException(
                status_code=500,
                detail="Detection failed — check server logs",
            ) from None
        metrics.inference_success(
            "detect",
            (perf_counter() - started) * 1000,
            (perf_counter() - request_started) * 1000,
        )
    finally:
        gate.release()

    return DetectResponse(
        cow_detected=len(detections) > 0,
        detection_count=len(detections),
        detections=detections,
        image_width=img_w,
        image_height=img_h,
        inference_time_ms=round(elapsed_ms, 2),
    )


# --- BCS ---
@app.post(
    "/bcs",
    response_model=BCSResponse,
    responses={
        400: _ERROR_RESPONSE,
        413: _ERROR_RESPONSE,
        500: _ERROR_RESPONSE,
        503: {
            **_ERROR_RESPONSE,
            "description": "Inference capacity is busy or BCS capability is unavailable",
        },
    },
)
async def bcs(file: UploadFile = File(...)) -> BCSResponse:
    """Estimate and expose one discrete BCS category from 1 through 5."""
    request_started = perf_counter()
    metrics = app.state.metrics
    metrics.request("bcs")
    try:
        image_bytes = await read_validated_upload(file)
    except UploadValidationError as exc:
        metrics.client_rejection("bcs", (perf_counter() - request_started) * 1000)
        _raise_upload_http_exception(exc)

    gate = app.state.bcs_inference_gate
    try:
        await gate.acquire()
    except InferenceCapacityBusyError:
        metrics.busy("bcs", (perf_counter() - request_started) * 1000)
        raise HTTPException(status_code=503, detail=INFERENCE_CAPACITY_BUSY_DETAIL) from None
    try:
        try:
            runtime = await run_in_threadpool(get_bcs_runtime)
            service = await run_in_threadpool(runtime.get_service)
        except Exception as exc:
            metrics.server_runtime_failure("bcs", (perf_counter() - request_started) * 1000)
            logger.error("BCS runtime unavailable: %s", type(exc).__name__)
            raise HTTPException(status_code=503, detail="BCS capability is unavailable") from None

        started = perf_counter()
        try:
            result = await run_in_threadpool(service.infer, image_bytes)
        except Exception as exc:
            if _is_bcs_input_error(exc):
                metrics.client_rejection("bcs", (perf_counter() - request_started) * 1000)
                logger.info("BCS image input rejected: %s", type(exc).__name__)
                raise HTTPException(status_code=400, detail="BCS image input is invalid") from None
            metrics.inference_failure(
                "bcs",
                (perf_counter() - started) * 1000,
                (perf_counter() - request_started) * 1000,
            )
            logger.error("BCS inference failed: %s", type(exc).__name__)
            raise HTTPException(status_code=500, detail="BCS inference failed") from None
        inference_time_ms = (perf_counter() - started) * 1000
        metrics.inference_success(
            "bcs", inference_time_ms, (perf_counter() - request_started) * 1000
        )
    finally:
        gate.release()

    model_status = _runtime_model_status(runtime)
    package_id = getattr(runtime, "package_id", None)
    if model_status == "experimental_not_approved":
        message = "Experimental BCS category 1..5 computed successfully; not approved for production."
    else:
        message = "BCS category 1..5 computed successfully."
    return BCSResponse(
        status="ok",
        message=message,
        cow_detected=None,
        bcs_category=result.bcs_category,
        model_status=model_status,
        package_id=package_id,
        inference_time_ms=round(inference_time_ms, 2),
    )


def get_bcs_runtime():
    """Resolve the optional BCS runtime only when a BCS operation asks for it."""
    from .bcs_runtime import get_bcs_runtime as resolve_runtime

    return resolve_runtime()


def _is_bcs_input_error(error: Exception) -> bool:
    try:
        from vacca_bcs.serving import BCSInferenceInputError
    except ImportError:
        return False
    return isinstance(error, BCSInferenceInputError)


def _runtime_model_status(runtime: object) -> str | None:
    raw_status = getattr(runtime, "model_status", None)
    if hasattr(raw_status, "value"):
        raw_status = raw_status.value
    return raw_status if raw_status in {"none", "experimental_not_approved", "external_unclassified"} else None


@app.get(
    "/ready/bcs",
    response_model=BCSReadinessResponse,
    responses={503: {"model": BCSReadinessResponse}},
)
def bcs_readiness() -> BCSReadinessResponse | JSONResponse:
    """Report BCS capability state without triggering lazy loading."""
    runtime = get_bcs_runtime()
    raw_status = runtime.status
    status = raw_status.value if hasattr(raw_status, "value") else raw_status
    messages = {
        "unconfigured": "BCS capability is not configured.",
        "not_loaded": "BCS capability is configured but not loaded.",
        "not_installed": "BCS private serving package is not installed.",
        "ready": "BCS capability is ready.",
        "unavailable": "BCS capability is unavailable.",
    }
    message = messages.get(status, "BCS capability is unavailable.")
    failure = getattr(runtime, "failure", None)
    if status == "unconfigured" and getattr(failure, "category", None) == "disabled":
        message = "BCS capability is disabled by emergency configuration."
    elif status in {"not_installed", "unavailable"} and failure is not None:
        message = failure.reason
    model_status = _runtime_model_status(runtime)
    package_id = getattr(runtime, "package_id", None)
    response = BCSReadinessResponse(
        status=status if status in messages else "unavailable",
        message=message,
        model_status=model_status,
        package_id=package_id,
    )
    if status != "ready":
        return JSONResponse(status_code=503, content=response.model_dump(exclude_none=True))
    return response


@app.get("/metrics", response_model=MetricsResponse, include_in_schema=False)
def metrics() -> MetricsResponse:
    """Expose path-free prototype counters without acquiring inference capacity."""
    return MetricsResponse.model_validate(app.state.metrics.snapshot())


# ============================================================
# PROTOTYPE UI — for validation only. Remove before production.
# ============================================================

@app.get("/ui", response_class=HTMLResponse)
def prototype_ui() -> HTMLResponse:
    """Serve the prototype validation UI."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        return HTMLResponse("<h1>UI not found</h1>", status_code=404)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))
