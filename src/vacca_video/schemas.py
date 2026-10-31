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
    highlighted_frame: "HighlightedFrameResponse | None" = None
    summary_error: str | None = None
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


class HighlightedFrameResponse(FrameResponse):
    sample_index: int = Field(ge=0)
    mean_confidence: int | float | None = Field(ge=0, le=1, allow_inf_nan=False)


class FramePageResponse(BaseModel):
    items: list[FrameResponse]
    total: int


class VideoErrorResponse(BaseModel):
    detail: str


class TrackResponse(ArtifactResponse):
    """One temporal trajectory inside a single video, never a bovine identity."""

    track_id: int = Field(ge=0)
    status: Literal["candidate", "review"]
    warnings: list[str]
    observations: int | None = None
    first_frame: int | None = None
    last_frame: int | None = None
    first_timestamp_ms: int | float | None = None
    last_timestamp_ms: int | float | None = None
    segments: list[dict[str, JsonValue]] | None = None
    gaps: list[dict[str, JsonValue]] | None = None
    possible_continuation_of: int | None = None
    possible_continued_by: int | None = None
    best_frame_index: int | None = None
    best_timestamp_ms: int | float | None = None
    best_confidence: int | float | None = None
    best_crop_contamination: int | float | None = None
    clip_url: str | None = None
    clip_frames: int | None = Field(default=None, ge=0)
    clip_media_type: Literal["video/webm", "video/mp4"] | None = None
    best_crop_url: str
    best_original_url: str
    segmentation: dict[str, JsonValue] | None = None
    best_masked_url: str | None = None
    best_mask_url: str | None = None


class TrackListResponse(BaseModel):
    schema_version: Literal["vacca-video-tracks-v1"]
    status: Literal["pending", "completed", "failed", "disabled", "unavailable"]
    reason: str | None = None
    identity_semantics: str | None = None
    tracker: str | None = None
    model_sha256: str | None = None
    fps: int | float | None = None
    width: int | None = None
    height: int | None = None
    frames_read: int | None = None
    frames_sampled: int | None = None
    stop_reason: str | None = None
    config: dict[str, JsonValue] | None = None
    warnings: list[str] | None = None
    notes: list[str] | None = None
    tracks: list[TrackResponse]


class PassageMetadata(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore", allow_inf_nan=False)


PassageBCSReason = Literal["no_candidate_captured", "invalid_original", "image_rejected",
    "capacity_busy", "capability_unavailable", "inference_failed", "evaluation_timeout", "invalid_outcome"]


class PassageBCSResult(PassageMetadata):
    """Public allowlist projected only after full sidecar/binding validation."""

    status: Literal["evaluated", "unevaluable"]
    bcs_category: int | None = Field(ge=1, le=5)
    reason: PassageBCSReason | None
    input_representation: Literal["original_only"] = "original_only"
    model_status: Literal["experimental_not_approved"] | None = None
    package_id: str | None = None
    inference_time_ms: float | None = Field(default=None, ge=0)
    lineage: dict[str, JsonValue] | None = None
    input_sha256: str | None = None


class PassageCandidateResponse(PassageMetadata):
    status: Literal["available", "not_available"]
    reason: Literal["no_candidate_captured"] | None = None
    frame_index: int | None = Field(default=None, ge=0)
    timestamp_ms: float | None = Field(default=None, ge=0)
    bcs_eligibility: Literal["not_assessed"] = Field(description="Clinical eligibility is not assessed.")
    bcs_status: Literal["not_evaluated"] = Field(deprecated=True,
        description="Legacy tracking-producer snapshot, not current inference state. Use passage.bcs.")
    best_original_url: str | None = None
    best_crop_url: str | None = None
    best_mask_url: str | None = None
    best_masked_url: str | None = None


class PassageResponse(PassageMetadata):
    """Chronological provisional interval, not a unique animal or clinical result."""
    sequence: int = Field(gt=0, le=2147483647)
    track_id: int = Field(ge=0)
    status: Literal["passage_evidence_candidate"]
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    start_timestamp_ms: float = Field(ge=0)
    end_timestamp_ms: float = Field(ge=0)
    end_exclusive: bool
    transition_method: Literal["initial_confidence", "concurrent_confidence", "missing_incumbent_foreground"]
    candidate: PassageCandidateResponse
    warnings: list[str]
    bcs: PassageBCSResult | None = Field(default=None,
        description="Authoritative experimental result; null when not performed or unavailable.")


class UnresolvedPassageResponse(PassageMetadata):
    track_id: int = Field(ge=0)
    first_frame: int = Field(ge=0)
    last_frame: int = Field(ge=0)
    first_timestamp_ms: float = Field(ge=0)
    last_timestamp_ms: float = Field(ge=0)
    possible_continuation_of: int | None = Field(ge=0)
    possible_continued_by: int | None = Field(ge=0)
    status: Literal["unresolved_passage_attribution"]


class PassageListResponse(PassageMetadata):
    schema_version: Literal["vacca-video-passages-v1"] = "vacca-video-passages-v1"
    status: Literal["pending", "completed", "failed", "disabled", "unavailable"]
    reason: str | None = None
    bcs_status: Literal["completed", "not_evaluated", "unavailable"] = Field(default="not_evaluated",
        description="Completed means the validated pass finished, including any unevaluable entries; not clinical approval.")
    bcs_reason: Literal["passage_bcs_unreadable"] | None = None
    evidence_status: Literal["provisional_evidence_not_unique_animal_identity"] | None = None
    identity_semantics: Literal["temporal_track_id_within_video"] = "temporal_track_id_within_video"
    calibration: Literal["uncalibrated_confidence_proxy_not_visibility_or_bcs"] | None = None
    passages: list[PassageResponse] = Field(default_factory=list)
    unresolved_count: int = 0
    unresolved_tracks: list[UnresolvedPassageResponse] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
