"""VACCA Vision API — FastAPI microservice for cow detection.

Endpoints:
    GET  /health          — service health + model info
    POST /detect          — cow detection with bounding boxes
    POST /bcs             — BCS scoring (placeholder for Phase 2)

The test UI at /ui is for PROTOTYPE VALIDATION ONLY.
Remove the /ui route and static/ directory before connecting to production backend.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .detection import get_detector
from .schemas import BCSResponse, DetectResponse, HealthResponse

# --- Configuration ---
ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
MODEL_PATH = ROOT / "models" / "deploy" / "vacca-yolo26n-v1.pt"

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
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, etc.)")

    try:
        image_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read uploaded file")

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

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


# --- BCS (placeholder) ---
@app.post("/bcs", response_model=BCSResponse)
async def bcs(file: UploadFile = File(...)) -> BCSResponse:
    """Placeholder for Body Condition Score estimation (Fase 2)."""
    # For now, just run detection to confirm a cow is present
    try:
        image_bytes = await file.read()
        detector = get_detector()
        detections, _, _, _ = detector.detect(image_bytes)
        cow_detected = len(detections) > 0
    except Exception:
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
