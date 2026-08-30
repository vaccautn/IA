"""Transactional snapshots; empty train/val class directories are retained."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Protocol

from PIL import Image

from vacca_vision.image_validation import (
    SUPPORTED_FORMATS,
    ImageValidationConfig,
    _validate_dimensions,
)

from .source_client import BCSEvidencePayload
from .source_plan import SourceExclusion, SourceProvenance
from .source_split_plan import (
    IntegerSplitAssignment,
    IntegerSplitCounts,
    IntegerSplitPlan,
)

SNAPSHOT_SCHEMA_VERSION = "bcs-integer-snapshot-v1"
_IMAGE_CONFIG = ImageValidationConfig()


class IntegerSnapshotError(Exception):
    pass


class IntegerSnapshotInputError(IntegerSnapshotError):
    pass


class IntegerSnapshotMaterializationError(IntegerSnapshotError):
    pass


class IntegerSnapshotImageError(IntegerSnapshotMaterializationError):
    pass


class IntegerSnapshotOutputError(IntegerSnapshotError):
    pass


class IntegerEvidenceMaterializer(Protocol):
    def materialize(self, evidence_id: int) -> BCSEvidencePayload: ...


@dataclass(frozen=True, slots=True)
class IntegerSnapshotRecord:
    split: str
    bcs_score: int
    evidence_id: int
    relative_path: str
    sha256: str
    provenance: tuple[SourceProvenance, ...]


@dataclass(frozen=True, slots=True)
class IntegerDatasetSnapshot:
    output_root: Path
    records: tuple[IntegerSnapshotRecord, ...]
    exclusions: tuple[SourceExclusion, ...]
    counts: IntegerSplitCounts
    manifest_json: str


def _validate_image(payload: bytes) -> str:
    if not payload or len(payload) > _IMAGE_CONFIG.max_size_bytes:
        raise IntegerSnapshotImageError("materialized evidence is not a valid image")
    try:
        with Image.open(BytesIO(payload)) as image:
            image_format = image.format
            if image_format not in SUPPORTED_FORMATS:
                raise IntegerSnapshotImageError(
                    "materialized evidence is not a JPEG or PNG"
                )
            _validate_dimensions(*image.size, _IMAGE_CONFIG)
            image.verify()
        with Image.open(BytesIO(payload)) as image:
            image.load()
    except IntegerSnapshotImageError:
        raise
    except Exception:
        raise IntegerSnapshotImageError(
            "materialized evidence is not a valid image"
        ) from None
    return ".jpg" if image_format == "JPEG" else ".png"


def _materialize(
    materializer: Callable[[int], BCSEvidencePayload] | IntegerEvidenceMaterializer,
    evidence_id: int,
) -> BCSEvidencePayload:
    try:
        payload = (
            materializer(evidence_id)
            if callable(materializer)
            else materializer.materialize(evidence_id)
        )
    except IntegerSnapshotError:
        raise
    except Exception:
        raise IntegerSnapshotMaterializationError(
            "failed to materialize evidence"
        ) from None
    if type(payload) is not BCSEvidencePayload or payload.evidence_id != evidence_id:
        raise IntegerSnapshotMaterializationError(
            "materialized evidence identity mismatch"
        )
    if hashlib.sha256(payload.payload).hexdigest() != payload.sha256:
        raise IntegerSnapshotMaterializationError(
            "materialized evidence digest mismatch"
        )
    return payload


def _write_file(path: Path, content: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _manifest(
    plan: IntegerSplitPlan, records: tuple[IntegerSnapshotRecord, ...]
) -> str:
    counts = {"train": list(plan.counts.train), "val": list(plan.counts.val)}
    payload = {
        "counts": counts,
        "exclusions": [
            asdict(item)
            for item in sorted(
                plan.exclusions,
                key=lambda item: (item.evidence_id, item.evaluation_id, item.reason),
            )
        ],
        "manifest_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "records": [asdict(item) for item in records],
        "split_plan": {
            "counts": counts,
            "identity_digest": plan.identity.digest,
            "seed": plan.config.seed,
            "val_ratio": plan.identity.canonical_val_ratio,
        },
    }
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def _check_assignment(assignment: IntegerSplitAssignment) -> None:
    expected = f"{assignment.split}/{assignment.bcs_score}/{assignment.evidence_id}"
    if (
        assignment.split not in {"train", "val"}
        or not 1 <= assignment.bcs_score <= 5
        or assignment.relative_path_stem != expected
    ):
        raise IntegerSnapshotInputError("split plan contains an invalid assignment")


def build_integer_snapshot(
    plan: IntegerSplitPlan,
    output_root: Path,
    materializer: Callable[[int], BCSEvidencePayload] | IntegerEvidenceMaterializer,
) -> IntegerDatasetSnapshot:
    """Materialize and atomically publish one immutable integer snapshot."""
    if not isinstance(plan, IntegerSplitPlan):
        raise IntegerSnapshotInputError("input must be an integer split plan")
    output_root = Path(output_root)
    if output_root.exists():
        raise IntegerSnapshotOutputError("output root already exists")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent)
    )
    records: list[IntegerSnapshotRecord] = []
    seen: set[int] = set()
    try:
        for split in ("train", "val"):
            for score in range(1, 6):
                (staging / split / str(score)).mkdir(parents=True, exist_ok=True)
        for assignment in plan.assignments:
            _check_assignment(assignment)
            if assignment.evidence_id in seen:
                raise IntegerSnapshotInputError(
                    "split plan contains duplicate evidence IDs"
                )
            seen.add(assignment.evidence_id)
            payload = _materialize(materializer, assignment.evidence_id)
            extension = _validate_image(payload.payload)
            relative_path = f"{assignment.relative_path_stem}{extension}"
            destination = staging / Path(relative_path)
            _write_file(destination, payload.payload)
            records.append(
                IntegerSnapshotRecord(
                    assignment.split,
                    assignment.bcs_score,
                    assignment.evidence_id,
                    relative_path,
                    payload.sha256,
                    assignment.provenance,
                )
            )
        records_tuple = tuple(records)
        manifest_json = _manifest(plan, records_tuple)
        _write_file(staging / "manifest.json", manifest_json.encode("utf-8"))
        os.rename(staging, output_root)
        return IntegerDatasetSnapshot(
            output_root,
            records_tuple,
            plan.exclusions,
            plan.counts,
            manifest_json,
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
