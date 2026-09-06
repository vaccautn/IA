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
)
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
INFERENCE_GATE_NAME = "shared-inference-capacity"
INFERENCE_GATE_CAPACITY = 1
INFERENCE_GATE_ACQUIRE_TIMEOUT_SECONDS = 0.1
INFERENCE_CAPACITY_BUSY_DETAIL = "Inference capacity is busy; retry shortly"


class InferenceCapacityBusyError(RuntimeError):
    """Raised when shared inference capacity cannot be acquired promptly."""


class InferenceCapacityGate:
    """Own a bounded, deterministic application-level inference capacity."""

    def __init__(
        self,
        capacity: int = INFERENCE_GATE_CAPACITY,
        acquisition_timeout: float = INFERENCE_GATE_ACQUIRE_TIMEOUT_SECONDS,
    ) -> None:
        if capacity < 1 or acquisition_timeout <= 0:
            raise ValueError("inference capacity and timeout must be positive")
        self.name = INFERENCE_GATE_NAME
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
app.state.inference_capacity_gate = InferenceCapacityGate()


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
    try:
        image_bytes = await read_validated_upload(file)
    except UploadValidationError as exc:
        _raise_upload_http_exception(exc)

    gate = request.app.state.inference_capacity_gate
    try:
        await gate.acquire()
    except InferenceCapacityBusyError:
        raise HTTPException(status_code=503, detail=INFERENCE_CAPACITY_BUSY_DETAIL) from None
    try:
        try:
            detector = request.app.state.detector
            detections, img_w, img_h, elapsed_ms = await run_in_threadpool(
                detector.detect,
                image_bytes,
            )
        except Exception as exc:
            logger.error("Detection request failed: %s", type(exc).__name__)
            raise HTTPException(
                status_code=500,
                detail="Detection failed — check server logs",
            ) from None
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
    try:
        image_bytes = await read_validated_upload(file)
    except UploadValidationError as exc:
        _raise_upload_http_exception(exc)

    gate = app.state.inference_capacity_gate
    try:
        await gate.acquire()
    except InferenceCapacityBusyError:
        raise HTTPException(status_code=503, detail=INFERENCE_CAPACITY_BUSY_DETAIL) from None
    try:
        try:
            runtime = await run_in_threadpool(get_bcs_runtime)
            service = await run_in_threadpool(runtime.get_service)
        except Exception as exc:
            logger.error("BCS runtime unavailable: %s", type(exc).__name__)
            raise HTTPException(status_code=503, detail="BCS capability is unavailable") from None

        try:
            result = await run_in_threadpool(service.infer, image_bytes)
        except Exception as exc:
            logger.error("BCS inference failed: %s", type(exc).__name__)
            if _is_bcs_input_error(exc):
                raise HTTPException(status_code=400, detail="BCS image input is invalid") from None
            raise HTTPException(status_code=500, detail="BCS inference failed") from None
    finally:
        gate.release()

    return BCSResponse(
        status="ok",
        message="BCS category 1..5 computed successfully.",
        cow_detected=None,
        bcs_category=result.bcs_category,
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
        "ready": "BCS capability is ready.",
        "unavailable": "BCS capability is unavailable.",
    }
    message = messages.get(status, "BCS capability is unavailable.")
    response = BCSReadinessResponse(
        status=status if status in messages else "unavailable", message=message
    )
    if status != "ready":
        return JSONResponse(status_code=503, content=response.model_dump())
    return response


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
