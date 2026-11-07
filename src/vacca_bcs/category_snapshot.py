"""Transactional, immutable BCS category snapshots."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
import stat
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from io import BytesIO

from PIL import Image

from vacca_vision.image_validation import SUPPORTED_EXTENSIONS, SUPPORTED_FORMATS, ImageValidationConfig, _validate_dimensions

from .constants import BCS_CLASS_SCORES, BCS_DOMAIN_ID, CLASS_NAMES, MANIFEST_FILENAME, NUM_CLASSES, NUM_THRESHOLDS, SCORE_BASE, SCORE_MAX, SCORE_MIN, SCORE_STEP, SPLITS
from .local_source import LOCAL_BCS_MAPPING, LocalSourceMaterialized
from .source_plan import LocalSourceProvenance, SourceExclusion
from .category_split_plan import CategorySplitAssignment, CategorySplitCounts, CategorySplitPlan, split_identity_digest, validate_category_split_plan
from .path_safety import SafePathError, safe_path

SNAPSHOT_SCHEMA_VERSION = "bcs-category-snapshot-v1"
PUBLICATION_MARKER_NAME = ".publication-pending"
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_KEYS = frozenset({
    "manifest_schema_version", "domain_id", "class_values", "class_mapping", "score_min", "score_max", "score_base", "score_step", "num_classes", "num_thresholds", "source_schema", "identity_scheme", "mapping", "observed_classes", "counts", "split_plan", "records", "exclusions", "isolation",
})


class CategorySnapshotError(Exception):
    pass


class CategorySnapshotInputError(CategorySnapshotError):
    pass


class CategorySnapshotMaterializationError(CategorySnapshotError):
    pass


class CategorySnapshotImageError(CategorySnapshotMaterializationError):
    pass


class CategorySnapshotOutputError(CategorySnapshotError):
    pass


class CategorySnapshotLockError(CategorySnapshotError):
    pass


class CategorySnapshotLockCleanupError(CategorySnapshotError):
    pass


class CategorySnapshotCleanupError(CategorySnapshotError):
    pass


class CategorySnapshotDurabilityError(CategorySnapshotError):
    pass


class CategorySnapshotPublicationError(CategorySnapshotError):
    pass


class CategorySnapshotManifestError(CategorySnapshotError, ValueError):
    pass


class CategorySnapshotManifestSchemaError(CategorySnapshotManifestError):
    pass


class CategorySnapshotLegacyManifestError(CategorySnapshotManifestSchemaError):
    pass


class CategorySnapshotManifestValidationError(CategorySnapshotManifestError):
    pass


@dataclass(frozen=True, slots=True)
class CategorySnapshotRecord:
    split: str
    bcs_category: int
    record_id: str
    relative_path: str
    sha256: str
    capture_group: str
    provenance: tuple[LocalSourceProvenance, ...]

@dataclass(frozen=True, slots=True)
class CategoryDatasetSnapshot:
    output_root: Path
    records: tuple[CategorySnapshotRecord, ...]
    exclusions: tuple[SourceExclusion, ...]
    counts: CategorySplitCounts
    manifest_json: str


def _manifest_failure(message: str, *, schema=False) -> CategorySnapshotManifestError:
    return (CategorySnapshotManifestSchemaError if schema else CategorySnapshotManifestValidationError)(message)


def _require(value: object, expected: frozenset[str]) -> dict[str, object]:
    if type(value) is not dict or frozenset(value) != expected:
        raise _manifest_failure("snapshot manifest has invalid fields")
    return value


def _valid_digest(value: object) -> bool:
    return type(value) is str and _DIGEST_RE.fullmatch(value) is not None


def _validate_image(payload: bytes) -> str:
    config = ImageValidationConfig()
    if not payload or len(payload) > config.max_size_bytes:
        raise CategorySnapshotImageError("materialized source is not a valid image")
    try:
        with Image.open(BytesIO(payload)) as image:
            image_format = image.format
            if image_format not in SUPPORTED_FORMATS:
                raise CategorySnapshotImageError("materialized source is not a JPEG or PNG")
            _validate_dimensions(*image.size, config)
            image.verify()
        with Image.open(BytesIO(payload)) as image:
            image.load()
    except CategorySnapshotImageError:
        raise
    except Exception:
        raise CategorySnapshotImageError("materialized source is not a valid image") from None
    return ".jpg" if image_format == "JPEG" else ".png"


def _materialize(materializer: object, assignment: CategorySplitAssignment) -> tuple[bytes, str]:
    try:
        payload = materializer(assignment.record_id) if callable(materializer) else materializer.materialize(assignment.record_id)
    except CategorySnapshotError:
        raise
    except Exception:
        raise CategorySnapshotMaterializationError("failed to materialize local source") from None
    if type(payload) is not LocalSourceMaterialized or payload.record_id != assignment.record_id:
        raise CategorySnapshotMaterializationError("materialized source identity mismatch")
    if sha256(payload.payload).hexdigest() != payload.sha256 or payload.sha256 != assignment.sha256:
        raise CategorySnapshotMaterializationError("materialized source digest mismatch")
    return payload.payload, payload.sha256


def _write(path: Path, content: bytes) -> None:
    try:
        with path.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        raise CategorySnapshotDurabilityError("snapshot file durability failed") from None


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
        raise CategorySnapshotDurabilityError("snapshot directory durability failed") from None


def _release(lock: Path) -> None:
    try:
        lock.unlink()
    except FileNotFoundError:
        return
    except OSError:
        raise CategorySnapshotLockCleanupError("failed to release output reservation") from None


def _manifest(plan: CategorySplitPlan, records: tuple[CategorySnapshotRecord, ...]) -> str:
    counts = {split: list(getattr(plan.counts, split)) for split in SPLITS}
    groups = sorted({record.capture_group for record in records})
    digests = sorted({record.sha256 for record in records})
    payload = {
        "manifest_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "domain_id": BCS_DOMAIN_ID,
        "class_values": list(BCS_CLASS_SCORES),
        "class_mapping": {name: index for index, name in enumerate(CLASS_NAMES)},
        "score_min": SCORE_MIN, "score_max": SCORE_MAX, "score_base": SCORE_BASE, "score_step": SCORE_STEP,
        "num_classes": NUM_CLASSES, "num_thresholds": NUM_THRESHOLDS,
        "source_schema": plan.identity.source_schema,
        "identity_scheme": plan.identity.identity_scheme,
        "mapping": dict(plan.identity.mapping_lineage),
        "observed_classes": list(plan.identity.observed_classes),
        "counts": counts,
        "split_plan": {
            "identity_digest": plan.identity.digest,
            "seed": plan.config.seed,
            "canonical_val_ratio": plan.identity.canonical_val_ratio,
            "canonical_test_ratio": plan.identity.canonical_test_ratio,
            "candidate_record_ids": sorted(record.record_id for record in records),
            "excluded_record_ids": sorted(item.record_id for item in plan.exclusions),
            "counts": counts,
            "capture_group_count": len(groups),
            "digest_count": len(digests),
        },
        "isolation": {"capture_group_count": len(groups), "digest_count": len(digests), "overlap": []},
        "records": [
            {"split": record.split, "bcs_category": record.bcs_category, "record_id": record.record_id, "relative_path": record.relative_path, "sha256": record.sha256, "capture_group": record.capture_group, "provenance": [asdict(item) for item in record.provenance]}
            for record in records
        ],
        "exclusions": [
            asdict(item)
            for item in sorted(
                plan.exclusions,
                key=lambda item: (item.sha256, item.source_label, item.relative_path, item.record_id),
            )
        ],
    }
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def _counts(value: object) -> dict[str, list[int]]:
    data = _require(value, frozenset(SPLITS))
    result = {}
    for split in SPLITS:
        values = data[split]
        if type(values) is not list or len(values) != NUM_CLASSES or any(type(item) is not int or item < 0 for item in values):
            raise _manifest_failure("snapshot manifest has invalid counts")
        result[split] = list(values)
    return result


def validate_category_snapshot_manifest(payload: object) -> dict[str, object]:
    if type(payload) is not dict:
        raise _manifest_failure("snapshot manifest must be an object", schema=True)
    if payload.get("manifest_schema_version") in {"bcs-integer-snapshot-v1", "bcs-integer-snapshot-v2"}:
        raise CategorySnapshotLegacyManifestError("legacy snapshot manifest is unsupported")
    manifest = _require(payload, _MANIFEST_KEYS)
    if manifest["manifest_schema_version"] != SNAPSHOT_SCHEMA_VERSION or manifest["domain_id"] != BCS_DOMAIN_ID or manifest["source_schema"] != "bcs-local-category-source-v1" or manifest["identity_scheme"] != "local-path-sha256-v1" or manifest["mapping"] != dict(LOCAL_BCS_MAPPING.entries) or manifest["class_values"] != list(BCS_CLASS_SCORES) or manifest["class_mapping"] != {name: index for index, name in enumerate(CLASS_NAMES)}:
        raise _manifest_failure("snapshot lineage is invalid")
    if type(manifest["observed_classes"]) is not list or manifest["observed_classes"] != list(BCS_CLASS_SCORES):
        raise _manifest_failure("snapshot must declare all five source categories")
    for field, expected in (("score_min", SCORE_MIN), ("score_max", SCORE_MAX), ("score_base", SCORE_BASE), ("score_step", SCORE_STEP), ("num_classes", NUM_CLASSES), ("num_thresholds", NUM_THRESHOLDS)):
        if type(manifest[field]) is not int or manifest[field] != expected:
            raise _manifest_failure("snapshot manifest scale does not match the domain")
    counts = _counts(manifest["counts"])
    split = _require(manifest["split_plan"], frozenset({"identity_digest", "seed", "canonical_val_ratio", "canonical_test_ratio", "candidate_record_ids", "excluded_record_ids", "counts", "capture_group_count", "digest_count"}))
    if not _valid_digest(split["identity_digest"]) or type(split["seed"]) is not int or type(split["canonical_val_ratio"]) is not str or type(split["canonical_test_ratio"]) is not str or split["counts"] != counts:
        raise _manifest_failure("snapshot split identity is invalid")
    isolation = _require(manifest["isolation"], frozenset({"capture_group_count", "digest_count", "overlap"}))
    if type(isolation["capture_group_count"]) is not int or type(isolation["digest_count"]) is not int or isolation["overlap"] != [] or split["capture_group_count"] != isolation["capture_group_count"] or split["digest_count"] != isolation["digest_count"]:
        raise _manifest_failure("snapshot isolation evidence is invalid")
    exclusions = manifest["exclusions"]
    if type(exclusions) is not list:
        raise _manifest_failure("snapshot exclusions are invalid")
    exclusion_ids = set()
    exclusion_sort_keys = []
    exclusion_digests = {}
    for raw in exclusions:
        item = _require(raw, frozenset({"record_id", "relative_path", "source_label", "bcs_category", "sha256", "reason"}))
        if (not _valid_digest(item["record_id"]) or type(item["relative_path"]) is not str or type(item["source_label"]) is not str or item["source_label"] not in dict(LOCAL_BCS_MAPPING.entries) or type(item["bcs_category"]) is not int or item["bcs_category"] != dict(LOCAL_BCS_MAPPING.entries).get(item["source_label"]) or not _valid_digest(item["sha256"]) or item["reason"] != "cross_category_identical_digest" or item["record_id"] in exclusion_ids):
            raise _manifest_failure("snapshot exclusion is invalid")
        if item["record_id"] != sha256(f"bcs-local-category-source-v1\0{item['relative_path']}".encode()).hexdigest() or item["relative_path"].split("/", 1)[0] != item["source_label"]:
            raise _manifest_failure("snapshot exclusion lineage is invalid")
        exclusion_ids.add(item["record_id"])
        exclusion_sort_keys.append((item["sha256"], item["source_label"], item["relative_path"], item["record_id"]))
        exclusion_digests.setdefault(item["sha256"], set()).add(item["bcs_category"])
    if exclusion_sort_keys != sorted(exclusion_sort_keys) or any(len(categories) < 2 for categories in exclusion_digests.values()):
        raise _manifest_failure("snapshot exclusions are not deterministic or do not prove a category conflict")
    records = manifest["records"]
    if type(records) is not list:
        raise _manifest_failure("snapshot records are invalid")
    ids, paths, groups, digests = set(), set(), {}, {}
    actual = {split_name: [0] * NUM_CLASSES for split_name in SPLITS}
    sort_keys = []
    for raw in records:
        item = _require(raw, frozenset({"split", "bcs_category", "record_id", "relative_path", "sha256", "capture_group", "provenance"}))
        split_name, category, record_id, relative, digest, group = (item[key] for key in ("split", "bcs_category", "record_id", "relative_path", "sha256", "capture_group"))
        if split_name not in SPLITS or type(category) is not int or category not in BCS_CLASS_SCORES or not _valid_digest(record_id) or not _valid_digest(digest) or type(group) is not str or not group or type(relative) is not str or re.fullmatch(rf"(?:{'|'.join(SPLITS)})/{category}/{record_id}\.(?:jpg|png)", relative) is None:
            raise _manifest_failure("snapshot record is invalid")
        if record_id in ids or record_id in exclusion_ids or relative.casefold() in paths:
            raise _manifest_failure("snapshot records are not unique")
        provenance = item["provenance"]
        if type(provenance) is not list or len(provenance) != 1:
            raise _manifest_failure("snapshot provenance is invalid")
        prov = _require(provenance[0], frozenset({"relative_path", "source_label"}))
        source_path, source_label = prov["relative_path"], prov["source_label"]
        if type(source_path) is not str or type(source_label) is not str or source_label not in dict(LOCAL_BCS_MAPPING.entries) or category != dict(LOCAL_BCS_MAPPING.entries).get(source_label) or source_path != _source_path(source_path) or source_path.split("/", 1)[0] != source_label or record_id != sha256(f"bcs-local-category-source-v1\0{source_path}".encode()).hexdigest():
            raise _manifest_failure("snapshot provenance is invalid")
        ids.add(record_id)
        paths.add(relative.casefold())
        actual[split_name][category - 1] += 1
        sort_keys.append((category, SPLITS.index(split_name), record_id))
        if group in groups and groups[group] != split_name:
            raise _manifest_failure("capture groups overlap snapshot partitions")
        groups[group] = split_name
        if digest in digests and digests[digest] != split_name:
            raise _manifest_failure("digests overlap snapshot partitions")
        digests[digest] = split_name
    if sort_keys != sorted(sort_keys) or actual != counts or len(groups) != isolation["capture_group_count"] or len(digests) != isolation["digest_count"]:
        raise _manifest_failure("snapshot records are not deterministic or counts do not match")
    if any(digest in digests for digest in exclusion_digests):
        raise _manifest_failure("snapshot quarantined digests were materialized")
    candidate_ids = split["candidate_record_ids"]
    excluded_ids_list = split["excluded_record_ids"]
    if type(candidate_ids) is not list or candidate_ids != sorted(ids) or type(excluded_ids_list) is not list or excluded_ids_list != sorted(exclusion_ids):
        raise _manifest_failure("snapshot split identity is inconsistent")
    if split["identity_digest"] != _split_identity_from_manifest(manifest, counts):
        raise _manifest_failure("snapshot split identity does not match declared assignments")
    return payload


def _source_path(value: str) -> str:
    pure = PurePosixPath(value)
    return value if not pure.is_absolute() and "\\" not in value and all(part not in {"", ".", ".."} for part in pure.parts) and value == "/".join(pure.parts) and len(pure.parts) == 2 and PurePosixPath(value).suffix.casefold() in SUPPORTED_EXTENSIONS else ""


def _split_identity_from_manifest(manifest: dict[str, object], counts: dict[str, list[int]]) -> str:
    split = manifest["split_plan"]
    assert isinstance(split, dict)
    for key in ("canonical_val_ratio", "canonical_test_ratio"):
        value = split[key]
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise _manifest_failure("snapshot split ratios are not canonical") from None
        canonical = "0" if numeric == 0 else format(numeric, ".15g")
        if not math.isfinite(numeric) or not 0 <= numeric < 1 or value != canonical:
            raise _manifest_failure("snapshot split ratios are not canonical")
    if float(split["canonical_val_ratio"]) + float(split["canonical_test_ratio"]) >= 1:
        raise _manifest_failure("snapshot split ratios leave no training data")
    assignments = sorted(
        (
            [
                item["split"], item["bcs_category"], item["record_id"],
                item["capture_group"], item["sha256"],
            ]
            for item in manifest["records"]
        ),
        key=lambda item: item[2],
    )
    exclusions = [
        (
            item["record_id"], item["relative_path"], item["bcs_category"],
            item["sha256"], item["reason"],
        )
        for item in sorted(
            manifest["exclusions"],
            key=lambda item: (
                item["sha256"], item["source_label"],
                item["relative_path"], item["record_id"],
            ),
        )
    ]
    return split_identity_digest(
        source_schema=manifest["source_schema"],
        identity_scheme=manifest["identity_scheme"],
        mapping_lineage=manifest["mapping"],
        observed_classes=manifest["observed_classes"],
        seed=split["seed"],
        canonical_val_ratio=split["canonical_val_ratio"],
        canonical_test_ratio=split["canonical_test_ratio"],
        assignments=assignments,
        counts=counts,
        exclusions=exclusions,
    )


def load_category_snapshot_manifest(path: Path | str | object) -> dict[str, object]:
    if isinstance(path, dict):
        return validate_category_snapshot_manifest(path)
    manifest_path = Path(path)
    if os.path.lexists(manifest_path.parent / PUBLICATION_MARKER_NAME):
        raise CategorySnapshotManifestError("snapshot publication is not durable")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"), parse_constant=lambda _: (_ for _ in ()).throw(ValueError))
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError):
        raise CategorySnapshotManifestError("could not read snapshot manifest") from None
    return validate_category_snapshot_manifest(payload)


def build_category_snapshot(
    plan: CategorySplitPlan,
    output_root: Path,
    materializer: object,
    *,
    approved_roots: tuple[Path, ...],
) -> CategoryDatasetSnapshot:
    if not isinstance(plan, CategorySplitPlan):
        raise CategorySnapshotInputError("input must be a category split plan")
    try:
        validate_category_split_plan(plan)
    except Exception as error:
        if isinstance(error, CategorySnapshotError):
            raise
        raise CategorySnapshotInputError(str(error)) from None
    if plan.identity.source_schema != "bcs-local-category-source-v1" or plan.identity.identity_scheme != "local-path-sha256-v1" or plan.identity.mapping_lineage != LOCAL_BCS_MAPPING.entries or plan.identity.observed_classes != BCS_CLASS_SCORES:
        raise CategorySnapshotInputError("category split plan source lineage is invalid")
    try:
        output_root = safe_path(
            output_root,
            base=Path(output_root).parent,
            approved_roots=approved_roots,
        )
    except SafePathError as error:
        raise CategorySnapshotOutputError(str(error)) from None
    if os.path.lexists(output_root):
        raise CategorySnapshotOutputError("output root already exists")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    lock = output_root.parent / f".{output_root.name}.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(descriptor)
    except FileExistsError:
        raise CategorySnapshotLockError("output reservation already exists") from None
    except OSError:
        raise CategorySnapshotLockError("could not reserve output") from None
    staging = None
    records = []
    try:
        staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent))
        for split in SPLITS:
            for category in BCS_CLASS_SCORES:
                (staging / split / str(category)).mkdir(parents=True)
        for assignment in plan.assignments:
            payload, digest = _materialize(materializer, assignment)
            extension = _validate_image(payload)
            relative = f"{assignment.relative_path_stem}{extension}"
            _write(staging / relative, payload)
            records.append(CategorySnapshotRecord(
                split=assignment.split,
                bcs_category=assignment.bcs_category,
                record_id=assignment.record_id,
                relative_path=relative,
                sha256=digest,
                capture_group=assignment.capture_group,
                provenance=assignment.provenance,
            ))
        records_tuple = tuple(sorted(records, key=lambda item: (item.bcs_category, SPLITS.index(item.split), item.record_id)))
        manifest = _manifest(plan, records_tuple)
        _write(staging / MANIFEST_FILENAME, manifest.encode())
        _write(staging / PUBLICATION_MARKER_NAME, b"snapshot publication pending\n")
        for split in SPLITS:
            for category in BCS_CLASS_SCORES:
                _fsync_directory(staging / split / str(category))
        _fsync_directory(staging)
        if os.path.lexists(output_root):
            raise CategorySnapshotOutputError("output root already exists")
        try:
            # The rename is the publication commit point. It is never rolled back.
            os.rename(staging, output_root)
        except OSError:
            raise CategorySnapshotPublicationError("snapshot publication failed") from None
        staging = None
        _fsync_directory(output_root.parent)
        try:
            (output_root / PUBLICATION_MARKER_NAME).unlink()
        except OSError:
            raise CategorySnapshotDurabilityError("snapshot publication marker could not be cleared") from None
        _release(lock)
        return CategoryDatasetSnapshot(output_root, records_tuple, plan.exclusions, plan.counts, manifest)
    except BaseException as original:
        cleanup_error: CategorySnapshotCleanupError | None = None
        if staging is not None and os.path.lexists(staging):
            try:
                shutil.rmtree(staging)
            except OSError:
                cleanup_error = CategorySnapshotCleanupError("failed to clean snapshot staging")
        try:
            _release(lock)
        except CategorySnapshotLockCleanupError:
            raise
        if cleanup_error is not None:
            raise cleanup_error from original
        raise


def finalize_category_snapshot_publication(
    output_root: Path,
    *,
    approved_roots: tuple[Path, ...],
) -> None:
    """Retry post-rename durability finalization without deleting the snapshot."""
    try:
        output_root = safe_path(
            output_root,
            base=Path(output_root).parent,
            approved_roots=approved_roots,
            allow_missing_final=False,
            require_dir=True,
        )
    except SafePathError as error:
        raise CategorySnapshotDurabilityError(str(error)) from None
    marker = output_root / PUBLICATION_MARKER_NAME
    try:
        marker_stat = os.lstat(marker)
    except OSError:
        raise CategorySnapshotDurabilityError("snapshot publication marker is unavailable") from None
    if not stat.S_ISREG(marker_stat.st_mode):
        raise CategorySnapshotDurabilityError("snapshot publication marker is unsafe")
    try:
        manifest_path = safe_path(
            output_root / MANIFEST_FILENAME,
            base=output_root,
            approved_roots=approved_roots,
            allow_missing_final=False,
            require_file=True,
        )
        validate_category_snapshot_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError, CategorySnapshotError):
        raise CategorySnapshotDurabilityError("snapshot publication contents are invalid") from None
    _fsync_directory(output_root.parent)
    try:
        marker.unlink()
    except OSError:
        raise CategorySnapshotDurabilityError("snapshot publication marker could not be cleared") from None
