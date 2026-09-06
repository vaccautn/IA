"""VACCA Vision API — FastAPI microservice for cow detection.

Endpoints:
    GET  /health          — service health + model info
    POST /detect          — cow detection with bounding boxes
    POST /bcs             — BCS scoring (placeholder for Phase 2)

The test UI at /ui is for PROTOTYPE VALIDATION ONLY.
Remove the /ui route and static/ directory before connecting to production backend.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import AsyncIterator, NoReturn

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse

from .detection import DEFAULT_MODEL, get_detector
from .schemas import BCSResponse, DetectResponse, HealthResponse
from .upload_validation import (
    UploadTooLargeError,
    UploadValidationError,
    read_validated_upload,
)

# --- Configuration ---
STATIC_DIR = Path(__file__).resolve().parent / "static"
MODEL_IDENTIFIER = DEFAULT_MODEL.name
logger = logging.getLogger(__name__)


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
@app.post("/detect", response_model=DetectResponse)
async def detect(request: Request, file: UploadFile = File(...)) -> DetectResponse:
    """Receive an image and return cow detections with bounding boxes."""
    try:
        image_bytes = await read_validated_upload(file)
    except UploadValidationError as exc:
        _raise_upload_http_exception(exc)

    try:
        detector = request.app.state.detector
        detections, img_w, img_h, elapsed_ms = detector.detect(image_bytes)
    except Exception as exc:
        logger.error("Detection request failed: %s", type(exc).__name__)
        raise HTTPException(
            status_code=500,
            detail="Detection failed — check server logs",
        ) from None

    return DetectResponse(
        cow_detected=len(detections) > 0,
        detection_count=len(detections),
        detections=detections,
        image_width=img_w,
        image_height=img_h,
        inference_time_ms=round(elapsed_ms, 2),
    )


# --- BCS (placeholder) ---
@app.post("/bcs", response_model=BCSResponse)
async def bcs(request: Request, file: UploadFile = File(...)) -> BCSResponse:
    """Placeholder for Body Condition Score estimation (Fase 2)."""
    # For now, just run detection to confirm a cow is present
    try:
        image_bytes = await read_validated_upload(file)
    except UploadValidationError as exc:
        _raise_upload_http_exception(exc)

    try:
        detector = request.app.state.detector
        detections, _, _, _ = detector.detect(image_bytes)
        cow_detected = len(detections) > 0
    except Exception as exc:
        logger.error("BCS placeholder detection failed: %s", type(exc).__name__)
        cow_detected = None

    return BCSResponse(
        cow_detected=cow_detected,
    )


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
