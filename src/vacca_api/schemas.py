"""Pydantic schemas for VACCA API request/response."""

from __future__ import annotations

from typing import Annotated, List, Literal, Optional

from pydantic import BaseModel, Field

from vacca_bcs.constants import SCORE_MAX, SCORE_MIN


BCSCategory = Annotated[int, Field(strict=True, ge=SCORE_MIN, le=SCORE_MAX)]
BCSModelStatus = Literal["none", "experimental_not_approved", "external_unclassified"]


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
    """Response from POST /bcs with a discrete BCS category."""

    status: str = Field(default="ok")
    message: str = Field(
        default="BCS category computed successfully."
    )
    cow_detected: Optional[bool] = None
    model_status: Optional[BCSModelStatus] = Field(
        default=None,
        description="BCS model classification; experimental status is not production approval.",
    )
    package_id: Optional[str] = Field(default=None, description="Bundled serving package identifier.")
    inference_time_ms: Optional[float] = Field(
        default=None, ge=0.0, description="Measured BCS service-call latency in milliseconds."
    )
    # Failed requests use HTTP errors; successful responses contain category 1..5.
    bcs_category: BCSCategory


class BCSReadinessResponse(BaseModel):
    """Capability-specific BCS readiness response."""

    status: Literal["unconfigured", "not_loaded", "not_installed", "ready", "unavailable"]
    message: str
    model_status: Optional[BCSModelStatus] = None
    package_id: Optional[str] = None

    def model_dump(self, *args, **kwargs):
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(*args, **kwargs)


class ErrorResponse(BaseModel):
    """Standard error body returned by an operation endpoint."""

    detail: str


class HealthResponse(BaseModel):
    """GET /health"""

    status: str = "ok"
    model_loaded: bool
    model_path: str
    gpu_available: bool


class CapabilityMetrics(BaseModel):
    requests: int = Field(..., ge=0)
    client_rejections: int = Field(..., ge=0)
    busy_rejections: int = Field(..., ge=0)
    server_runtime_failures: int = Field(..., ge=0)
    inference_attempts: int = Field(..., ge=0)
    inference_successes: int = Field(..., ge=0)
    inference_failures: int = Field(..., ge=0)
    request_wall_time_ms_total: float = Field(..., ge=0.0)
    last_request_wall_time_ms: Optional[float] = Field(default=None, ge=0.0)
    non_inference_wall_time_ms_total: float = Field(..., ge=0.0)
    last_non_inference_wall_time_ms: Optional[float] = Field(default=None, ge=0.0)
    eligible_operational_requests: int = Field(..., ge=0)
    service_impacting_failures: int = Field(..., ge=0)
    service_impacting_failure_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    service_impacting_failure_rate_review_thresholds: list[float]
    successful_inference_ms: float = Field(..., ge=0.0)
    failed_inference_ms: float = Field(..., ge=0.0)
    last_successful_inference_ms: Optional[float] = Field(default=None, ge=0.0)
    last_failed_inference_ms: Optional[float] = Field(default=None, ge=0.0)


class MetricsResponse(BaseModel):
    measurement_window_started_at_utc: str
    detect: CapabilityMetrics
    bcs: CapabilityMetrics
