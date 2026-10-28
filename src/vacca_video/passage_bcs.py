"""Non-HTTP passage validation and digest-bound experimental BCS sidecars."""

import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from vacca_vision import validate_image_bytes

from .artifacts import (PASSAGE_BCS_FILE, passage_artifact, read_tracking_bytes,
                        read_tracking_json)
from .schemas import PassageResponse, UnresolvedPassageResponse

IMAGE_LIMIT = 10 * 1024 * 1024
SCHEMA = "vacca-video-passage-bcs-v1"
REASONS = {"no_candidate_captured", "invalid_original", "image_rejected", "capacity_busy",
           "capability_unavailable", "inference_failed", "evaluation_timeout", "invalid_outcome"}
LINEAGE_FIELDS = {"checkpoint_schema_version", "domain_id", "snapshot_schema", "snapshot_identity",
                  "dataset_manifest_digest", "run_id", "source_schema", "source_identity_scheme"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def passage_entries(summary):
    """Validate identity/chronology before any image inference; never infer empty on corruption."""
    evidence = summary["passage_evidence"]
    if (type(evidence) is not dict
            or evidence.get("status") != "provisional_evidence_not_unique_animal_identity"
            or type(evidence.get("policy")) is not dict
            or evidence["policy"].get("calibration") != "uncalibrated_confidence_proxy_not_visibility_or_bcs"
            or type(evidence.get("intervals")) is not list
            or type(evidence.get("unresolved_tracks")) is not list):
        raise ValueError("Invalid passage evidence")
    pairs = sorted(((PassageResponse.model_validate(raw), raw) for raw in evidence["intervals"]),
                   key=lambda pair: pair[0].sequence)
    previous, tracks = None, set()
    for index, (entry, raw) in enumerate(pairs, 1):
        if (entry.sequence != index or entry.track_id in tracks
                or entry.end_frame < entry.start_frame or entry.end_timestamp_ms < entry.start_timestamp_ms
                or entry.end_exclusive != (index < len(pairs))
                or (previous is not None and (entry.start_frame <= previous.start_frame
                    or entry.start_timestamp_ms <= previous.start_timestamp_ms
                    or entry.start_frame != previous.end_frame
                    or entry.start_timestamp_ms != previous.end_timestamp_ms))):
            raise ValueError("Invalid passage chronology")
        candidate = entry.candidate
        if candidate.status == "available":
            if (type(raw["candidate"].get("track_id")) is not int
                    or raw["candidate"]["track_id"] != entry.track_id
                    or candidate.frame_index is None or candidate.timestamp_ms is None
                    or not entry.start_frame <= candidate.frame_index <= entry.end_frame
                    or not entry.start_timestamp_ms <= candidate.timestamp_ms <= entry.end_timestamp_ms
                    or (entry.end_exclusive and (candidate.frame_index == entry.end_frame
                                                or candidate.timestamp_ms == entry.end_timestamp_ms))):
                raise ValueError("Invalid candidate provenance")
        elif (candidate.reason != "no_candidate_captured" or candidate.frame_index is not None
              or candidate.timestamp_ms is not None):
            raise ValueError("Invalid candidate absence")
        previous = entry
        tracks.add(entry.track_id)
    for raw in evidence["unresolved_tracks"]:
        item = UnresolvedPassageResponse.model_validate(raw)
        if (item.track_id in tracks or item.last_frame < item.first_frame
                or item.last_timestamp_ms < item.first_timestamp_ms):
            raise ValueError("Invalid unresolved track")
        tracks.add(item.track_id)
    return pairs


def binding(entry):
    return {key: getattr(entry, key) for key in (
        "sequence", "track_id", "start_frame", "end_frame", "start_timestamp_ms",
        "end_timestamp_ms", "end_exclusive")} | dict(
            frame_index=entry.candidate.frame_index, timestamp_ms=entry.candidate.timestamp_ms)


def original_bytes(root, job_id, entry, raw):
    if raw["candidate"].get("original") != f"passages/{entry.sequence}/best-original.jpg":
        raise ValueError("Invalid original reference")
    with passage_artifact(root, job_id, entry.sequence, "best-original").open("rb") as stream:
        image = stream.read(IMAGE_LIMIT + 1)
    if len(image) > IMAGE_LIMIT:
        raise ValueError("Original is too large")
    return image


def validate_original(image):
    validate_image_bytes(image)
    if not image.startswith(b"\xff\xd8\xff"):
        raise ValueError("Original encoding must be JPEG")


def safe_token(value):
    return type(value) is str and re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,199}", value) is not None


def validate_lineage(value):
    if value is None:
        return value
    if (type(value) is not dict or set(value) != LINEAGE_FIELDS | {
            "observed_classes", "missing_classes", "source_mapping"}
            or any(not safe_token(value[key]) for key in LINEAGE_FIELDS)):
        raise ValueError("Invalid lineage")
    for key in ("observed_classes", "missing_classes"):
        if (type(value[key]) is not list or len(value[key]) > 5
                or any(type(c) is not int or c not in range(1, 6) for c in value[key])):
            raise ValueError("Invalid lineage classes")
    mapping = value["source_mapping"]
    if (type(mapping) is not dict or len(mapping) > 5
            or any(type(k) is not str or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", k)
                   or type(v) is not int or v not in range(1, 6) for k, v in mapping.items())):
        raise ValueError("Invalid lineage mapping")
    return value


class PassageBCSRow(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)
    sequence: int = Field(gt=0)
    track_id: int = Field(ge=0)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    start_timestamp_ms: float = Field(ge=0)
    end_timestamp_ms: float = Field(ge=0)
    end_exclusive: bool
    frame_index: int | None = Field(ge=0)
    timestamp_ms: float | None = Field(ge=0)
    input_sha256: str | None
    status: Literal["evaluated", "unevaluable"]
    reason: str | None
    bcs_category: int | None = Field(ge=1, le=5)
    model_status: Literal["experimental_not_approved"] | None = None
    package_id: str | None = None
    inference_time_ms: float | None = Field(default=None, ge=0)
    lineage: dict | None = None

    @model_validator(mode="after")
    def valid_outcome(self):
        if self.input_sha256 is not None and not re.fullmatch(r"[a-f0-9]{64}", self.input_sha256):
            raise ValueError("Invalid image digest")
        if self.status == "evaluated":
            if (self.bcs_category is None or self.reason is not None or self.input_sha256 is None
                    or self.frame_index is None or self.timestamp_ms is None
                    or self.model_status != "experimental_not_approved"
                    or self.inference_time_ms is None
                    or (self.package_id is not None and not safe_token(self.package_id))):
                raise ValueError("Invalid evaluated outcome")
            validate_lineage(self.lineage)
        elif (self.reason not in REASONS or self.bcs_category is not None
              or any(v is not None for v in (self.model_status, self.package_id,
                                             self.inference_time_ms, self.lineage))):
            raise ValueError("Invalid unevaluable outcome")
        return self


class PassageBCSSidecar(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    schema_version: Literal["vacca-video-passage-bcs-v1"] = SCHEMA
    representation: Literal["original_only"] = "original_only"
    summary_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    passages: list[PassageBCSRow]


def read_passage_bcs(root, job_id):
    """Validate the full join, including current accepted image bytes, before API projection."""
    summary_bytes = read_tracking_bytes(root, job_id, "summary.json")
    pairs = passage_entries(json.loads(summary_bytes))
    sidecar = PassageBCSSidecar.model_validate(read_tracking_json(root, job_id, PASSAGE_BCS_FILE))
    if sidecar.summary_sha256 != digest(summary_bytes) or len(sidecar.passages) != len(pairs):
        raise ValueError("Stale passage BCS summary")
    for row, (entry, raw) in zip(sidecar.passages, pairs):
        if any(row.model_dump()[key] != value for key, value in binding(entry).items()):
            raise ValueError("Stale passage BCS binding")
        if row.status == "evaluated":
            image = original_bytes(root, job_id, entry, raw)
            validate_original(image)
            if row.input_sha256 != digest(image):
                raise ValueError("Stale passage BCS image")
    return sidecar.model_dump()
