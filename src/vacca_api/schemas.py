"""Pydantic schemas for VACCA API request/response."""

from __future__ import annotations

from typing import Annotated, List, Literal, Optional

from pydantic import BaseModel, Field


BCSScore = Annotated[int, Field(strict=True, ge=1, le=5)]


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
    """Response from POST /bcs."""

    status: str = Field(default="ok")
    message: str = Field(
        default="BCS score computed successfully."
    )
    cow_detected: Optional[bool] = None
    # Retained for response compatibility; failed requests use HTTP errors.
    bcs_score: Optional[BCSScore] = None


class BCSReadinessResponse(BaseModel):
    """Capability-specific BCS readiness response."""

    status: Literal["unconfigured", "not_loaded", "ready", "unavailable"]
    message: str


class HealthResponse(BaseModel):
    """GET /health"""

    status: str = "ok"
    model_loaded: bool
    model_path: str
    gpu_available: bool
