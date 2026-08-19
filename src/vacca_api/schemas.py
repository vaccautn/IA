"""Pydantic schemas for VACCA API request/response."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Normalized bounding box (0.0–1.0 relative to image dimensions)."""

    x_center: float = Field(..., ge=0.0, le=1.0, description="Center X (normalized)")
    y_center: float = Field(..., ge=0.0, le=1.0, description="Center Y (normalized)")
    width: float = Field(..., ge=0.0, le=1.0, description="Width (normalized)")
    height: float = Field(..., ge=0.0, le=1.0, description="Height (normalized)")


class Detection(BaseModel):
    """Single cow detection result."""

    class_name: str = Field(default="cow", description="Detected class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    bbox: BoundingBox

    # Pixel coordinates (computed server-side from image dimensions)
    x1: Optional[int] = Field(default=None, description="Left pixel")
    y1: Optional[int] = Field(default=None, description="Top pixel")
    x2: Optional[int] = Field(default=None, description="Right pixel")
    y2: Optional[int] = Field(default=None, description="Bottom pixel")


class DetectResponse(BaseModel):
    """Response from POST /detect"""

    cow_detected: bool = Field(..., description="True if at least one cow was detected")
    detection_count: int = Field(..., ge=0, description="Number of cows detected")
    detections: List[Detection] = Field(default_factory=list)
    image_width: int = Field(..., description="Input image width in pixels")
    image_height: int = Field(..., description="Input image height in pixels")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")


class BCSResponse(BaseModel):
    """Response from POST /bcs — placeholder for future BCS scoring."""

    status: str = Field(default="not_implemented")
    message: str = Field(
        default="BCS scoring endpoint is not yet implemented. "
        "Currently only cow detection is available at POST /detect."
    )
    cow_detected: Optional[bool] = None
    bcs_score: Optional[float] = None


class HealthResponse(BaseModel):
    """GET /health"""

    status: str = "ok"
    model_loaded: bool
    model_path: str
    gpu_available: bool
