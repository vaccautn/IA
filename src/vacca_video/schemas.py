"""Local video response DTOs, without imposing new artifact/request limits.

Partial artifacts may omit fields and retain extension fields. Response routes
exclude unset defaults so reading those artifacts never invents JSON keys.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError


class ArtifactResponse(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)


class VideoSourceResponse(ArtifactResponse):
    name: str | None = None
    width: int | None = None
    height: int | None = None
    fps: int | float | None = None
    declared_frames: int | None = None


class VideoModelResponse(ArtifactResponse):
    name: str | None = None
    sha256: str | None = None
    confidence: int | float | None = None


class SamplingResponse(ArtifactResponse):
    sample_fps: int | float | None = None
    max_frames: int | None = None


class CandidateResponse(ArtifactResponse):
    artifact: Literal["candidate.png"]
    encoding: Literal["PNG"]
    frame_index: int = Field(ge=0)
    sample_ordinal: int = Field(ge=0)
    timestamp_ms: int | float = Field(ge=0, allow_inf_nan=False)
    timestamp_source: str
    detector_confidence: int | float = Field(ge=0, le=1, allow_inf_nan=False)
    target_index: Literal[0]
    original_bbox: dict[str, JsonValue]
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    detector_model: VideoModelResponse


def reject_candidate(summary):
    """Discard rejected optional metadata, without loosening its DTO or paths."""
    return dict(summary, candidate=None, candidate_rejection_reason="invalid_candidate",
                bcs=dict(status="invalid_input", reason="invalid_candidate"))


def sanitize_optional_candidate(summary):
    """Protect status serialization even when fallback publication failed."""
    if summary.get("candidate") is not None:
        try:
            CandidateResponse.model_validate(summary["candidate"])
        except ValidationError:
            return reject_candidate(summary)
    return summary  # Preserve valid/old summaries, including unset and extra fields.


class VideoBCSResponse(ArtifactResponse):
    status: Literal["not_evaluated", "completed", "busy", "unavailable", "invalid_input", "failed"]
    reason: str | None = None
    bcs_category: int | None = Field(default=None, ge=1, le=5)
    model_status: str | None = None
    package_id: str | None = None
    inference_time_ms: int | float | None = None
    lineage: dict[str, JsonValue] | None = None


class AnalysisSummaryResponse(ArtifactResponse):
    selection_policy: str | None = None
    candidate: CandidateResponse | None = None
    bcs: VideoBCSResponse | None = None
    schema_version: str | None = None
    processing_id: str | None = None
    started_at_utc: str | None = None
    status: str | None = None
    source: VideoSourceResponse | None = None
    model: VideoModelResponse | None = None
    config: SamplingResponse | None = None
    model_load_seconds: int | float | None = None
    python: str | None = None
    platform: str | None = None
    frames_read: int | None = None
    frames_analyzed: int | None = None
    frames_with_cows: int | None = None
    max_cows_in_frame: int | None = None
    inference_ms_total: int | float | None = None
    timestamp_fallback_frames: int | None = None
    last_timestamp_ms: int | float | None = None
    stop_reason: str | None = None
    error: str | None = None
    processing_wall_seconds: int | float | None = None
    processing_fps: int | float | None = None
    mean_inference_ms: int | float | None = None
    preview_frames: int | None = None
    preview_limit: int | None = None
    preview_truncated: bool | None = None
    annotated_video: str | None = None
    annotated_video_fps: int | float | None = None
    count_semantics: str | None = None


class JobResponse(BaseModel):
    id: str
    name: str
    status: str
    created_at: str
    error: str | None
    frames_analyzed: int
    summary: AnalysisSummaryResponse | None
    schema_version: str


class FrameResponse(ArtifactResponse):
    schema_version: str | None = None
    processing_id: str | None = None
    model_version: str | None = None
    frame_index: int | None = None
    timestamp_ms: int | float | None = None
    timestamp_source: str | None = None
    cow_detected: bool | None = None
    detection_count: int | None = None
    detections: list[dict[str, JsonValue]] | None = None
    image_width: int | None = None
    image_height: int | None = None
    inference_time_ms: int | float | None = None
    image_url: str


class FramePageResponse(BaseModel):
    items: list[FrameResponse]
    total: int


class VideoErrorResponse(BaseModel):
    detail: str
