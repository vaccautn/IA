"""Transactional snapshots; Windows directory fsync is unsupported, external mutation is out of contract."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
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
    validate_integer_split_plan,
)
from .constants import (
    BCS_CLASS_SCORES,
    BCS_DOMAIN_ID,
    CLASS_NAMES,
    NUM_CLASSES,
    NUM_THRESHOLDS,
    SCORE_BASE,
    SCORE_MAX,
    SCORE_MIN,
    SCORE_STEP,
    SPLITS,
)
from .source_client import SCHEMA_VERSION as SOURCE_SCHEMA_VERSION

SNAPSHOT_SCHEMA_VERSION = "bcs-integer-snapshot-v2"
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_KEYS = frozenset(
    {
        "manifest_schema_version",
        "domain_id",
        "class_values",
        "class_mapping",
        "score_min",
        "score_max",
        "score_base",
        "score_step",
        "num_classes",
        "num_thresholds",
        "source_schema",
        "counts",
        "split_plan",
        "records",
        "exclusions",
    }
)
_SPLIT_KEYS = frozenset(
    {
        "identity_digest",
        "seed",
        "canonical_val_ratio",
        "candidate_evidence_ids",
        "excluded_evidence_ids",
        "counts",
    }
)
_COUNT_KEYS = frozenset(SPLITS)
_RECORD_KEYS = frozenset(
    {"split", "bcs_score", "evidence_id", "relative_path", "sha256", "provenance"}
)
_PROVENANCE_KEYS = frozenset({"evidence_id", "evaluation_id"})
_EXCLUSION_KEYS = frozenset({"evaluation_id", "evidence_id", "bcs_score", "reason"})
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


class IntegerSnapshotLockError(IntegerSnapshotError):
    pass


class IntegerSnapshotLockCleanupError(IntegerSnapshotError):
    pass


class IntegerSnapshotCleanupError(IntegerSnapshotError):
    pass


class IntegerSnapshotDurabilityError(IntegerSnapshotError):
    pass


class IntegerSnapshotPublicationError(IntegerSnapshotError):
    pass


class IntegerSnapshotManifestError(IntegerSnapshotError, ValueError):
    """Base class for sanitized snapshot manifest failures."""


class IntegerSnapshotManifestSchemaError(IntegerSnapshotManifestError):
    """Raised when a manifest does not use the exact v2 schema."""


class IntegerSnapshotLegacyManifestError(IntegerSnapshotManifestSchemaError):
    """Raised for the retired integer-v1 and fractional manifest families."""


class IntegerSnapshotManifestValidationError(IntegerSnapshotManifestError):
    """Raised when a v2 manifest violates a contract or consistency rule."""


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
    try:
        with path.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        raise IntegerSnapshotDurabilityError(
            "snapshot file durability failed"
        ) from None


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        raise IntegerSnapshotDurabilityError(
            "snapshot directory durability failed"
        ) from None


def _lock_path(output_root: Path) -> Path:
    return output_root.parent / f".{output_root.name}.lock"


def _acquire_lock(output_root: Path) -> Path:
    lock = _lock_path(output_root)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(descriptor)
    except FileExistsError:
        raise IntegerSnapshotLockError("output reservation already exists") from None
    except OSError:
        raise IntegerSnapshotLockError("could not reserve output") from None
    return lock


def _release_lock(lock: Path) -> None:
    try:
        os.unlink(lock)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise IntegerSnapshotLockCleanupError(
            f"failed to release output reservation {lock.name}"
        ) from exc


def _cleanup_after_failure(
    staging: Path | None, lock: Path, original: BaseException
) -> None:
    failures: list[tuple[str, BaseException]] = []
    if staging is not None and os.path.lexists(staging):
        try:
            shutil.rmtree(staging)
        except BaseException as exc:
            failures.append(("staging", exc))
    try:
        _release_lock(lock)
    except BaseException as exc:
        failures.append(("lock", exc))
    if not failures:
        return
    kind, cause = failures[0]
    error_type = (
        IntegerSnapshotCleanupError
        if kind == "staging"
        else IntegerSnapshotLockCleanupError
    )
    error = error_type(
        f"failed to clean {kind} for {staging.name if kind == 'staging' and staging else lock.name}"
    )
    error.add_note(f"original failure: {type(original).__name__}")
    for failure_kind, failure in failures:
        error.add_note(f"{failure_kind} cleanup failure: {type(failure).__name__}")
    raise error from cause


def _manifest(
    plan: IntegerSplitPlan, records: tuple[IntegerSnapshotRecord, ...]
) -> str:
    counts = {"train": list(plan.counts.train), "val": list(plan.counts.val)}
    record_ids = sorted(record.evidence_id for record in records)
    excluded_ids = sorted(item.evidence_id for item in plan.exclusions)
    payload = {
        "class_mapping": {
            class_name: index for index, class_name in enumerate(CLASS_NAMES)
        },
        "class_values": list(BCS_CLASS_SCORES),
        "counts": counts,
        "domain_id": BCS_DOMAIN_ID,
        "exclusions": [
            asdict(item)
            for item in sorted(
                plan.exclusions,
                key=lambda item: (item.evidence_id, item.evaluation_id, item.reason),
            )
        ],
        "manifest_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "num_classes": NUM_CLASSES,
        "num_thresholds": NUM_THRESHOLDS,
        "records": [asdict(item) for item in records],
        "score_base": SCORE_BASE,
        "score_max": SCORE_MAX,
        "score_min": SCORE_MIN,
        "score_step": SCORE_STEP,
        "source_schema": SOURCE_SCHEMA_VERSION,
        "split_plan": {
            "candidate_evidence_ids": record_ids,
            "counts": counts,
            "excluded_evidence_ids": excluded_ids,
            "identity_digest": plan.identity.digest,
            "seed": plan.config.seed,
            "canonical_val_ratio": plan.identity.canonical_val_ratio,
        },
    }
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def _manifest_failure(message: str, *, schema: bool = False) -> IntegerSnapshotManifestError:
    error_type = IntegerSnapshotManifestSchemaError if schema else IntegerSnapshotManifestValidationError
    return error_type(message)


def _require_keys(value: object, expected: frozenset[str]) -> dict[str, object]:
    if type(value) is not dict or frozenset(value) != expected:
        raise _manifest_failure("snapshot manifest has invalid fields")
    return value


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _valid_digest(value: object) -> bool:
    return type(value) is str and _DIGEST_RE.fullmatch(value) is not None


def _valid_canonical_ratio(value: object) -> bool:
    if type(value) is not str or not value or value.strip() != value:
        return False
    try:
        ratio = Decimal(value)
    except (InvalidOperation, ValueError):
        return False
    if not ratio.is_finite() or not Decimal(0) <= ratio < Decimal(1):
        return False
    canonical = "0" if ratio == 0 else format(ratio.normalize(), "f")
    return value == canonical


def _validate_counts(value: object) -> dict[str, list[int]]:
    counts = _require_keys(value, _COUNT_KEYS)
    normalized: dict[str, list[int]] = {}
    for split in SPLITS:
        values = counts[split]
        if (
            type(values) is not list
            or len(values) != NUM_CLASSES
            or any(type(item) is not int or item < 0 for item in values)
        ):
            raise _manifest_failure("snapshot manifest has invalid counts")
        normalized[split] = list(values)
    return normalized


def _validate_relative_path(value: object, score: int, evidence_id: int) -> bool:
    if type(value) is not str or "\\" in value:
        return False
    expected = rf"(?:train|val)/{score}/{evidence_id}\.(?:jpg|png)"
    return re.fullmatch(expected, value) is not None


def validate_integer_snapshot_manifest(payload: object) -> dict[str, object]:
    """Validate a v2 manifest without touching its image files or source storage."""
    if type(payload) is not dict:
        raise _manifest_failure("snapshot manifest must be an object", schema=True)
    version = payload.get("manifest_schema_version")
    class_values = payload.get("class_values")
    if version == "bcs-integer-snapshot-v1" or (
        type(class_values) is list
        and any(type(value) is float or (type(value) is str and "." in value) for value in class_values)
    ):
        raise IntegerSnapshotLegacyManifestError("legacy snapshot manifest is unsupported")
    manifest = _require_keys(payload, _MANIFEST_KEYS)
    if manifest["manifest_schema_version"] != SNAPSHOT_SCHEMA_VERSION:
        raise _manifest_failure("unsupported snapshot manifest schema", schema=True)
    if manifest["domain_id"] != BCS_DOMAIN_ID or manifest["source_schema"] != SOURCE_SCHEMA_VERSION:
        raise _manifest_failure("snapshot manifest lineage does not match the domain")
    expected_mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    if (
        type(manifest["class_values"]) is not list
        or any(type(value) is not int for value in manifest["class_values"])
        or manifest["class_values"] != list(BCS_CLASS_SCORES)
        or type(manifest["class_mapping"]) is not dict
        or frozenset(manifest["class_mapping"]) != frozenset(expected_mapping)
        or any(type(value) is not int for value in manifest["class_mapping"].values())
        or manifest["class_mapping"] != expected_mapping
    ):
        raise _manifest_failure("snapshot manifest classes do not match the domain")
    for field, expected in (
        ("score_min", SCORE_MIN),
        ("score_max", SCORE_MAX),
        ("score_base", SCORE_BASE),
        ("score_step", SCORE_STEP),
        ("num_classes", NUM_CLASSES),
        ("num_thresholds", NUM_THRESHOLDS),
    ):
        if type(manifest[field]) is not int or manifest[field] != expected:
            raise _manifest_failure("snapshot manifest scale does not match the domain")

    counts = _validate_counts(manifest["counts"])
    split_plan = _require_keys(manifest["split_plan"], _SPLIT_KEYS)
    if (
        not _valid_digest(split_plan["identity_digest"])
        or type(split_plan["seed"]) is not int
        or not _valid_canonical_ratio(split_plan["canonical_val_ratio"])
    ):
        raise _manifest_failure("snapshot manifest split identity is invalid")
    split_counts = _validate_counts(split_plan["counts"])
    if split_counts != counts:
        raise _manifest_failure("snapshot manifest split counts are inconsistent")

    records = manifest["records"]
    exclusions = manifest["exclusions"]
    if type(records) is not list or type(exclusions) is not list:
        raise _manifest_failure("snapshot manifest lineage lists are invalid")
    record_ids: list[int] = []
    paths: set[str] = set()
    provenance_ids: set[int] = set()
    actual_counts = {split: [0] * NUM_CLASSES for split in SPLITS}
    record_sort_keys: list[tuple[int, int, int]] = []
    for record in records:
        item = _require_keys(record, _RECORD_KEYS)
        split = item["split"]
        score = item["bcs_score"]
        evidence_id = item["evidence_id"]
        if (
            type(split) is not str
            or split not in SPLITS
            or type(score) is not int
            or score not in BCS_CLASS_SCORES
            or not _positive_int(evidence_id)
            or not _valid_digest(item["sha256"])
            or not _validate_relative_path(item["relative_path"], score, evidence_id)
        ):
            raise _manifest_failure("snapshot manifest record is invalid")
        path_key = item["relative_path"].casefold()
        if evidence_id in record_ids or path_key in paths:
            raise _manifest_failure("snapshot manifest records are not unique")
        record_ids.append(evidence_id)
        paths.add(path_key)
        actual_counts[split][score - SCORE_MIN] += 1
        record_sort_keys.append((score, 0 if split == "train" else 1, evidence_id))

        provenance = item["provenance"]
        if type(provenance) is not list or not provenance:
            raise _manifest_failure("snapshot manifest provenance is invalid")
        provenance_keys: list[tuple[int, int]] = []
        for entry in provenance:
            provenance_item = _require_keys(entry, _PROVENANCE_KEYS)
            provenance_evidence = provenance_item["evidence_id"]
            evaluation_id = provenance_item["evaluation_id"]
            if not _positive_int(provenance_evidence) or not _positive_int(evaluation_id):
                raise _manifest_failure("snapshot manifest provenance is invalid")
            provenance_keys.append((provenance_evidence, evaluation_id))
            if provenance_evidence in provenance_ids:
                raise _manifest_failure("snapshot manifest provenance is not unique")
            provenance_ids.add(provenance_evidence)
        if provenance_keys != sorted(provenance_keys) or provenance_keys[0][0] != evidence_id:
            raise _manifest_failure("snapshot manifest provenance is inconsistent")

    if record_sort_keys != sorted(record_sort_keys):
        raise _manifest_failure("snapshot manifest records are not deterministic")
    if actual_counts != counts:
        raise _manifest_failure("snapshot manifest counts do not match records")

    exclusion_ids: list[int] = []
    exclusion_sort_keys: list[tuple[int, int, str]] = []
    for exclusion in exclusions:
        item = _require_keys(exclusion, _EXCLUSION_KEYS)
        if (
            not _positive_int(item["evaluation_id"])
            or not _positive_int(item["evidence_id"])
            or type(item["bcs_score"]) is not int
            or item["bcs_score"] not in BCS_CLASS_SCORES
            or type(item["reason"]) is not str
            or not item["reason"]
            or item["reason"].strip() != item["reason"]
        ):
            raise _manifest_failure("snapshot manifest exclusion is invalid")
        key = (item["evidence_id"], item["evaluation_id"], item["reason"])
        if item["evidence_id"] in exclusion_ids or item["evidence_id"] in record_ids:
            raise _manifest_failure("snapshot manifest evidence IDs are not unique")
        exclusion_ids.append(item["evidence_id"])
        exclusion_sort_keys.append(key)
    if exclusion_sort_keys != sorted(exclusion_sort_keys):
        raise _manifest_failure("snapshot manifest exclusions are not deterministic")

    candidates = split_plan["candidate_evidence_ids"]
    excluded = split_plan["excluded_evidence_ids"]
    if (
        type(candidates) is not list
        or any(not _positive_int(item) for item in candidates)
        or candidates != sorted(set(candidates))
        or candidates != sorted(record_ids)
        or type(excluded) is not list
        or any(not _positive_int(item) for item in excluded)
        or excluded != sorted(set(excluded))
        or excluded != sorted(exclusion_ids)
    ):
        raise _manifest_failure("snapshot manifest split identity is inconsistent")
    return payload


def load_integer_snapshot_manifest(path: Path | str | object) -> dict[str, object]:
    """Load and strictly validate a v2 manifest, or validate an in-memory object."""
    if isinstance(path, dict):
        return validate_integer_snapshot_manifest(path)
    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError),
        )
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError):
        raise IntegerSnapshotManifestError("could not read snapshot manifest") from None
    return validate_integer_snapshot_manifest(payload)


def _check_assignment(assignment: IntegerSplitAssignment) -> None:
    expected = f"{assignment.split}/{assignment.bcs_score}/{assignment.evidence_id}"
    if (
        assignment.split not in {"train", "val"}
        or not 1 <= assignment.bcs_score <= 5
        or type(assignment.evidence_id) is not int
        or assignment.evidence_id <= 0
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
    validate_integer_split_plan(plan)
    validate_integer_split_plan(plan)
    output_root = Path(output_root)
    if os.path.lexists(output_root):
        raise IntegerSnapshotOutputError("output root already exists")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    lock = _acquire_lock(output_root)
    staging: Path | None = None
    records: list[IntegerSnapshotRecord] = []
    seen: set[int] = set()
    try:
        if os.path.lexists(output_root):
            raise IntegerSnapshotOutputError("output root already exists")
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{output_root.name}.staging-", dir=output_root.parent
            )
        )
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
        for split in ("train", "val"):
            for score in range(1, 6):
                _fsync_directory(staging / split / str(score))
        _fsync_directory(staging)
        if os.path.lexists(output_root):
            raise IntegerSnapshotOutputError("output root already exists")
        try:
            os.rename(staging, output_root)
        except OSError:
            raise IntegerSnapshotPublicationError(
                "snapshot publication failed"
            ) from None
        _fsync_directory(output_root)
        _fsync_directory(output_root.parent)
        _release_lock(lock)
        return IntegerDatasetSnapshot(
            output_root,
            records_tuple,
            plan.exclusions,
            plan.counts,
            manifest_json,
        )
    except BaseException as original:
        _cleanup_after_failure(staging, lock, original)
        raise
