"""VACCA Vision API — FastAPI microservice for cow detection.

Endpoints:
    GET  /health          — service health + model info
    POST /detect          — cow detection with bounding boxes
    POST /bcs             — discrete BCS category 1..5

The test UI at /ui is for PROTOTYPE VALIDATION ONLY.
Remove the /ui route and static/ directory before connecting to production backend.
"""

from __future__ import annotations

import traceback
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from .detection import get_detector
from .schemas import BCSReadinessResponse, BCSResponse, DetectResponse, HealthResponse
from .upload_validation import read_uploaded_image

# --- Configuration ---
ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
MODEL_PATH = ROOT / "outputs" / "training" / "combined-v2-finetune" / "weights" / "best.pt"

# --- App ---
app = FastAPI(
    title="VACCA Vision API",
    description="Cow detection microservice — VACCA Fase 1",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # prototyping — restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Startup ---
@app.on_event("startup")
def _startup() -> None:
    # Pre-load the model so first request is fast
    get_detector(model_path=MODEL_PATH)
    print(f"[vacca-api] Model loaded from {MODEL_PATH}")


# --- Health ---
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    detector = get_detector()
    return HealthResponse(
        model_loaded=True,
        model_path=detector.model_path,
        gpu_available=detector.gpu_available,
    )


# --- Detect ---
@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...)) -> DetectResponse:
    """Receive an image and return cow detections with bounding boxes."""
    image_bytes = await read_uploaded_image(file)

    try:
        detector = get_detector()
        detections, img_w, img_h, elapsed_ms = detector.detect(image_bytes)
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Detection failed — check server logs")

    return DetectResponse(
        cow_detected=len(detections) > 0,
        detection_count=len(detections),
        detections=detections,
        image_width=img_w,
        image_height=img_h,
        inference_time_ms=round(elapsed_ms, 2),
    )


# --- BCS ---
@app.post("/bcs", response_model=BCSResponse)
async def bcs(file: UploadFile = File(...)) -> BCSResponse:
    """Estimate and expose one discrete BCS category from 1 through 5."""
    image_bytes = await read_uploaded_image(file)
    try:
        service = get_bcs_runtime().get_service()
    except Exception:
        raise HTTPException(status_code=503, detail="BCS capability is unavailable") from None

    try:
        result = service.infer(image_bytes)
    except Exception as exc:
        if _is_bcs_input_error(exc):
            raise HTTPException(status_code=400, detail="BCS image input is invalid") from None
        raise HTTPException(status_code=500, detail="BCS inference failed") from None

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
