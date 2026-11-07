"""Offline blank-label review packs; no scoring, splitting, or animal identity."""

import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

from vacca_bcs.path_safety import safe_path
from .artifacts import TRACKING_JSON_LIMIT
from .bcs_export import _sha256
from .schemas import PassageResponse

COLUMNS = ("image_id", "group_id", "source_video_sha256", "run_metadata_sha256",
    "sequence", "track_id", "start_frame", "end_frame", "start_timestamp_ms", "end_timestamp_ms",
    "end_exclusive", "frame_index", "timestamp_ms", "bbox", "candidate_status", "candidate_reason",
    "warnings", "fragment_status", "segmentation_status", "representations",
    "original", "crop", "mask", "masked", "bcs_category", "rump_visible", "notes")


def _checked(path, *, directory=False, missing=False):
    path = Path(path).absolute()
    return safe_path(path, base=path.anchor, approved_roots=(Path(path.anchor),),
        allow_missing_final=missing, require_dir=directory, require_file=not (directory or missing))


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate summary key")
        result[key] = value
    return result


def _codes(value):
    if (type(value) is not list or len(value) > 100 or any(type(code) is not str
            or not re.fullmatch(r"[a-z][a-z0-9_]{0,127}", code) for code in value)):
        raise ValueError("Invalid warning codes")
    return ";".join(value)


def _entries(summary):
    evidence = summary.get("passage_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("Passage evidence required; legacy track-only summaries are unsupported")
    if (evidence.get("status") != "provisional_evidence_not_unique_animal_identity"
            or type(evidence.get("policy")) is not dict
            or evidence["policy"].get("calibration") != "uncalibrated_confidence_proxy_not_visibility_or_bcs"
            or type(evidence.get("intervals")) is not list
            or type(evidence.get("unresolved_tracks")) is not list):
        raise ValueError("Invalid passage evidence state")
    _codes(evidence.get("warnings"))
    previous, tracks = None, set()
    for index, raw in enumerate(evidence["intervals"], 1):
        entry = PassageResponse.model_validate(raw)
        # The web gate is job-scoped; enforce its chronology without importing the API.
        if (entry.sequence != index or entry.track_id in tracks
                or entry.end_frame < entry.start_frame or entry.end_timestamp_ms < entry.start_timestamp_ms
                or entry.end_exclusive != (index < len(evidence["intervals"]))
                or (entry.end_exclusive and (entry.start_frame == entry.end_frame
                                            or entry.start_timestamp_ms == entry.end_timestamp_ms))
                or (previous and (entry.start_frame != previous.end_frame
                    or entry.start_timestamp_ms != previous.end_timestamp_ms
                    or entry.start_frame <= previous.start_frame
                    or entry.start_timestamp_ms <= previous.start_timestamp_ms))):
            raise ValueError("Inconsistent passage chronology")
        previous = entry
        tracks.add(entry.track_id)
        yield entry, raw


def export_passage_labels(run, output, *, source_video=None):
    """Publish a new directory atomically after strict metadata/path validation.

    A missing source is allowed only with a recorded full SHA256, never a basename.
    Only passage originals are inputs; derivative references are never accessed.
    """
    run = _checked(run, directory=True)
    output = _checked(output, missing=True)
    _checked(output.parent, directory=True)
    if output.exists():
        raise ValueError("Output already exists")
    if output.is_relative_to(run) or run.is_relative_to(output):
        raise ValueError("Input/output overlap is not allowed")
    with _checked(run / "summary.json").open("rb") as stream:
        data = stream.read(TRACKING_JSON_LIMIT + 1)
    if len(data) > TRACKING_JSON_LIMIT:
        raise ValueError("Summary is too large")
    summary = json.loads(data, object_pairs_hook=_pairs)
    if type(summary) is not dict:
        raise ValueError("Invalid summary object")
    json.dumps(summary, allow_nan=False)  # Reject NaN/Infinity, including overflowed JSON numbers.
    identities = [summary[key] for key in ("input_sha256", "video_sha256", "source_video_sha256") if key in summary]
    if any(type(value) is not str or not re.fullmatch("[0-9a-f]{64}", value) for value in identities):
        raise ValueError("Invalid source SHA256 metadata")
    if len(set(identities)) > 1:
        raise ValueError("Conflicting source SHA256 metadata")
    recorded = identities[0] if identities else None
    digest = _sha256(_checked(source_video)) if source_video is not None else recorded
    if not digest:
        raise ValueError("Explicit source file or source SHA256 metadata required")
    if recorded and digest != recorded:
        raise ValueError("Source SHA256 does not match summary")
    if source_video is not None and _checked(source_video).is_relative_to(output):
        raise ValueError("Source/output overlap is not allowed")
    metadata_digest = hashlib.sha256(data).hexdigest()
    rows, copies = [], []
    for entry, raw in _entries(summary):
        candidate = raw["candidate"]
        warning_codes = _codes(raw["warnings"])
        image_id = f"{digest}-p{entry.sequence}"
        row = dict.fromkeys(COLUMNS, "")
        row.update(image_id=image_id, group_id=f"video-{digest}", source_video_sha256=digest,
            run_metadata_sha256=metadata_digest, sequence=entry.sequence, track_id=entry.track_id,
            start_frame=entry.start_frame, end_frame=entry.end_frame,
            start_timestamp_ms=entry.start_timestamp_ms, end_timestamp_ms=entry.end_timestamp_ms,
            end_exclusive=entry.end_exclusive, candidate_status=entry.candidate.status,
            warnings=warning_codes, fragment_status="unresolved" if "fragment" in warning_codes else "not_asserted")
        if entry.candidate.status == "not_available":
            if candidate.get("reason") != "no_candidate_captured":
                raise ValueError("Invalid candidate absence reason")
            row["candidate_reason"] = "no_candidate_captured"
        else:
            frame, timestamp = entry.candidate.frame_index, entry.candidate.timestamp_ms
            bbox = candidate.get("bbox")
            if (type(candidate.get("track_id")) is not int or candidate["track_id"] != entry.track_id
                    or frame is None or timestamp is None
                    or not entry.start_frame <= frame <= entry.end_frame
                    or not entry.start_timestamp_ms <= timestamp <= entry.end_timestamp_ms
                    or (entry.end_exclusive and (frame == entry.end_frame or timestamp == entry.end_timestamp_ms))
                    or type(bbox) is not list or len(bbox) != 4
                    or any(type(value) is not int or not 0 <= value <= 2147483647 for value in bbox)
                    or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
                raise ValueError("Invalid candidate provenance or bbox")
            segmentation = candidate.get("segmentation")
            if segmentation is not None and (type(segmentation) is not dict or segmentation.get("status") not in (
                    "ok", "not_requested", "rejected", "failed", "skipped")):
                raise ValueError("Invalid candidate segmentation")
            row.update(frame_index=frame, timestamp_ms=timestamp, bbox=json.dumps(bbox),
                       segmentation_status=segmentation["status"] if segmentation is not None else "")
            reference = candidate.get("original")
            if reference != f"passages/{entry.sequence}/best-original.jpg":
                raise ValueError("Invalid passage artifact reference")
            source = _checked(run / reference)
            row["original"] = f"images/{image_id}-best-original.jpg"
            row["representations"] = "full_original"
            copies.append((source, row["original"]))
        rows.append(row)
    result = dict(schema_version="vacca-bcs-passage-labels-v2", source_video_sha256=digest,
        input_representation="full_original_frame_only",
        source_identity="verified_file" if source_video is not None else "recorded_sha256",
        run_metadata_sha256=metadata_digest, rows=len(rows), assets=len(copies),
        group_semantics="same_source_video_not_cow_identity", scoring="none",
        evidence_status=summary["passage_evidence"]["status"], passages=rows)
    staging = Path(tempfile.mkdtemp(prefix=".passage-labels-", dir=output.parent))
    try:
        (staging / "images").mkdir()
        for source, relative in copies:
            shutil.copyfile(_checked(source), staging / relative)
        with (staging / "labels.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        (staging / "export.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        _checked(output, missing=True)
        if output.exists():
            raise ValueError("Output already exists")
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return result
