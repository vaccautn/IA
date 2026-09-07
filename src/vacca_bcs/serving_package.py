"""Strict loading and integrity validation for the private BCS serving package."""

from __future__ import annotations

from collections.abc import Mapping
from io import BytesIO
import hashlib
import hmac
import json
import os
from pathlib import Path
import stat
from dataclasses import dataclass
from typing import Any
from zipfile import ZIP_STORED

import torch

from .constants import NUM_THRESHOLDS
from .model import BCSOrdinalModel
from .serving import BCSLineageMetadata, LoadedBCSModel
from .path_safety import SafePathError, safe_path

MAX_MANIFEST_BYTES = 128 * 1024
MAX_ARTIFACT_BYTES = 100 * 1024 * 1024
MAX_DOCUMENT_BYTES = 256 * 1024
# Sidecars contain one lowercase digest, two spaces, one filename, and a newline.
MAX_SIDECAR_BYTES = 256
# Allows ZIP central-directory metadata while bounding expansion above member limits.
MAX_ARCHIVE_OVERHEAD_BYTES = 64 * 1024
# Kept as defense in depth; the producer uses ZIP_STORED and the installer rejects other methods.
MAX_ARCHIVE_COMPRESSION_RATIO = 100
ARCHIVE_COMPRESSION = ZIP_STORED


@dataclass(frozen=True, slots=True)
class ServingPackagePolicy:
    """Immutable identity and validation policy for one serving-package instance."""

    schema: str
    package_id: str
    capability: str
    artifact_path: str
    artifact_format: str
    artifact_size_bytes: int
    artifact_sha256: str
    model_card_path: str
    model_card_sha256: str
    third_party_notices_path: str
    third_party_notices_sha256: str
    source_checkpoint_sha256: str
    source_role: str
    source_epoch: int
    source_best_epoch: int
    architecture: str
    imgsz: int
    classes: tuple[int, ...]
    thresholds: int
    checkpoint_schema_version: str
    domain_id: str
    snapshot_schema: str
    source_schema: str
    source_identity_scheme: str
    source_mapping: tuple[tuple[str, int], ...]
    observed_classes: tuple[int, ...]
    missing_classes: tuple[int, ...]
    approval_status: str
    serving_status: str
    validation_status: str
    failed_check_ids: tuple[str, ...]
    activation: str
    report_path: str
    artifact_license_status: str
    code_license: str
    model_artifact_license: str
    dataset_license: str
    pretrained_weights_permission: str
    private_distribution_acceptance: str
    archive_filename: str
    archive_directory: str
    archive_size_bytes: int
    archive_sha256: str


PACKAGE_POLICY = ServingPackagePolicy(
    schema="vacca-bcs-serving-package-v1",
    package_id="bcs-category-coral-2026-09-04",
    capability="bcs_ordinal_category_1_5",
    artifact_path="model_state.pt",
    artifact_format="torch_state_dict",
    artifact_size_bytes=44784009,
    artifact_sha256="41f0e9a644e25f4be6facba7f7c2dcbf2988c5332293970d8d521e67a0d44e60",
    model_card_path="MODEL_CARD.md",
    model_card_sha256="84cf063c80056eadad8c5436a7524a514259b343442e4a4fe26336c36c81b2ea",
    third_party_notices_path="THIRD_PARTY_NOTICES.md",
    third_party_notices_sha256="e0e9fc21178fdb995b3acee29157e009e0f886801d65e7be22fecc87f80cbdf9",
    source_checkpoint_sha256="592f8ce762b8a2bf68b722c8d4de4cb21f8ae48fdf42ea1979869e8d8105e38c",
    source_role="best",
    source_epoch=30,
    source_best_epoch=30,
    architecture="resnet18_coral",
    imgsz=224,
    classes=(1, 2, 3, 4, 5),
    thresholds=4,
    checkpoint_schema_version="bcs-category-coral-checkpoint-v1",
    domain_id="bcs-category-1-5-v1",
    snapshot_schema="bcs-category-snapshot-v1",
    source_schema="bcs-local-category-source-v1",
    source_identity_scheme="local-path-sha256-v1",
    source_mapping=(("3.25", 1), ("3.5", 2), ("3.75", 3), ("4.0", 4), ("4.25", 5)),
    observed_classes=(1, 2, 3, 4, 5),
    missing_classes=(),
    approval_status="experimental",
    serving_status="not_approved",
    validation_status="failed",
    failed_check_ids=(
        "macro_f1",
        "balanced_accuracy",
        "class_f1",
        "class_within_one",
        "class_error_ge_2",
        "ordinal_mae",
    ),
    activation="private_team_package_after_install",
    report_path="reports/bcs-category-baseline-2026-09-04.md",
    artifact_license_status="experimental_private_team_prototype_distribution_internal_only",
    code_license="AGPL-3.0-only",
    model_artifact_license="derived_research_prototype_private_team_distribution; component permissions are not asserted",
    dataset_license="CC BY 4.0 applies to attributed Science Data Bank V3 source material",
    pretrained_weights_permission="undetermined; private team distributor must determine permission for intended use",
    private_distribution_acceptance="maintainer_accepted_internal_private_team_prototype_distribution_2026-09-06",
    archive_filename="vacca-bcs-category-coral-2026-09-04-experimental.zip",
    archive_directory="bcs-category-coral-2026-09-04",
    archive_size_bytes=44796082,
    archive_sha256="735c1a0c6bb18721ee435508d8242adece089c8a295a6c9d10c82189fbb78184",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PACKAGE_ROOT = REPO_ROOT / "models" / "catalog" / PACKAGE_POLICY.package_id
CATALOG_MANIFEST_PATH = CATALOG_PACKAGE_ROOT / "manifest.json"
PRIVATE_PACKAGE_ROOT = REPO_ROOT / "models" / "private" / PACKAGE_POLICY.package_id
PRIVATE_MANIFEST_PATH = PRIVATE_PACKAGE_ROOT / "manifest.json"
PRIVATE_ARTIFACT_PATH = PRIVATE_PACKAGE_ROOT / PACKAGE_POLICY.artifact_path
PRIVATE_ARCHIVE_ROOT = REPO_ROOT / "artifacts" / "private"
PRIVATE_ARCHIVE_PACKAGE_ROOT = PRIVATE_ARCHIVE_ROOT / PACKAGE_POLICY.archive_directory
PRIVATE_ARCHIVE_PATH = PRIVATE_ARCHIVE_PACKAGE_ROOT / PACKAGE_POLICY.archive_filename
PRIVATE_ARCHIVE_SIDECAR_PATH = Path(f"{PRIVATE_ARCHIVE_PATH}.sha256")
PRIVATE_ROOT = REPO_ROOT / "models" / "private"
PRIVATE_PACKAGE_ID = PACKAGE_POLICY.package_id
PRIVATE_ARTIFACT_RELATIVE_PATH = PACKAGE_POLICY.artifact_path
PACKAGE_SCHEMA = PACKAGE_POLICY.schema
RECOVERY_BACKUP_PREFIX = f".{PACKAGE_POLICY.package_id}.backup-"
LEGACY_RECOVERY_BACKUP_NAME = f".{PACKAGE_POLICY.package_id}.backup"
DISABLED_RECOVERY_PREFIX = f".{PACKAGE_POLICY.package_id}.disabled-"
PACKAGE_MEMBERS = (
    "manifest.json",
    PACKAGE_POLICY.artifact_path,
    PACKAGE_POLICY.model_card_path,
    PACKAGE_POLICY.third_party_notices_path,
)
PACKAGE_MEMBER_SET = frozenset(PACKAGE_MEMBERS)
PACKAGE_MEMBER_LIMITS = {
    "manifest.json": MAX_MANIFEST_BYTES,
    PACKAGE_POLICY.artifact_path: MAX_ARTIFACT_BYTES,
    PACKAGE_POLICY.model_card_path: MAX_DOCUMENT_BYTES,
    PACKAGE_POLICY.third_party_notices_path: MAX_DOCUMENT_BYTES,
}
MAX_ARCHIVE_UNCOMPRESSED_BYTES = sum(PACKAGE_MEMBER_LIMITS.values()) + MAX_ARCHIVE_OVERHEAD_BYTES

_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "package_id",
        "capability",
        "artifact",
        "documentation",
        "artifact_license",
        "source_checkpoint",
        "model",
        "lineage",
        "approval",
        "activation",
        "report_path",
    }
)
_ARTIFACT_KEYS = frozenset({"path", "format", "size_bytes", "sha256"})
_DOCUMENTATION_KEYS = frozenset({"model_card", "third_party_notices"})
_DOCUMENT_ENTRY_KEYS = frozenset({"path", "sha256"})
_LICENSE_KEYS = frozenset(
    {
        "status",
        "code_license",
        "model_artifact_license",
        "dataset_license",
        "pretrained_weights_permission",
        "private_distribution_acceptance",
    }
)
_SOURCE_KEYS = frozenset({"role", "sha256", "epoch", "best_epoch", "selection_identity"})
_MODEL_KEYS = frozenset({"architecture", "imgsz", "classes", "thresholds"})
_LINEAGE_KEYS = frozenset(
    {
        "checkpoint_schema_version",
        "domain_id",
        "snapshot_schema",
        "snapshot_identity",
        "dataset_manifest_digest",
        "run_id",
        "config_sha256",
        "source_schema",
        "source_identity_scheme",
        "source_mapping",
        "observed_classes",
        "missing_classes",
    }
)
_APPROVAL_KEYS = frozenset({"status", "serving_status", "validation_status", "failed_check_ids"})
_FAILED_CHECK_IDS = PACKAGE_POLICY.failed_check_ids
_MODEL_CARD_REQUIRED_PHRASES = (
    "experimental",
    "no aprobado",
    "ResNet18",
    "224",
    "41f0e9a644e25f4be6facba7f7c2dcbf2988c5332293970d8d521e67a0d44e60",
)
_NOTICES_REQUIRED_PHRASES = (
    "Science Data Bank",
    "10.57760/sciencedb.16704",
    "CC BY 4.0",
    "Huang Xiao Ping",
    "PyTorch",
    "torchvision",
    "IMAGENET1K_V1",
    "permiso",
)


def read_publication_documents(
    model_card_path: Path, third_party_notices_path: Path
) -> dict[str, bytes]:
    """Read required publication documents and enforce attribution/provenance phrases."""
    documents = {
        "model_card": (Path(model_card_path), _MODEL_CARD_REQUIRED_PHRASES),
        "third_party_notices": (Path(third_party_notices_path), _NOTICES_REQUIRED_PHRASES),
    }
    result: dict[str, bytes] = {}
    for name, (path, phrases) in documents.items():
        raw = _read_bounded_file(path, MAX_DOCUMENT_BYTES)
        if not raw:
            raise ServingPackageError("serving package documentation is empty")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise ServingPackageError("serving package documentation is not UTF-8") from None
        if any(phrase.casefold() not in text.casefold() for phrase in phrases):
            raise ServingPackageError("serving package documentation is incomplete")
        result[name] = raw
    return result


class ServingPackageError(RuntimeError):
    """Raised when the immutable private package is absent or invalid."""


def _digest(value: object, length: int = 64) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_bounded_file(path: Path, maximum: int) -> bytes:
    try:
        before = os.lstat(path)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ServingPackageError("serving package file is unsafe")
        descriptor = os.open(
            str(path), os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
    except FileNotFoundError:
        raise ServingPackageError("serving package file is unavailable") from None
    except ServingPackageError:
        raise
    except OSError:
        raise ServingPackageError("serving package file cannot be opened safely") from None
    try:
        opened = os.fstat(descriptor)
        if stat.S_ISLNK(opened.st_mode) or not stat.S_ISREG(opened.st_mode):
            raise ServingPackageError("serving package file is unsafe")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1 << 20, maximum - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise ServingPackageError("serving package file exceeds its limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (after.st_dev, after.st_ino):
            raise ServingPackageError("serving package file changed during access")
        return b"".join(chunks)
    except ServingPackageError:
        raise
    except OSError:
        raise ServingPackageError("serving package file cannot be read safely") from None
    finally:
        os.close(descriptor)


def _safe_package_file(path: Path, package_root: Path) -> Path:
    try:
        return safe_path(
            path,
            base=REPO_ROOT,
            approved_roots=(package_root,),
            allow_missing_final=False,
            require_file=True,
        )
    except SafePathError:
        raise ServingPackageError("serving package path is unsafe") from None


def _safe_private_tree_path(
    path: Path,
    *,
    allow_missing_final: bool,
    require_dir: bool = False,
) -> Path:
    try:
        safe_root = safe_path(
            PRIVATE_ROOT,
            base=REPO_ROOT,
            approved_roots=(REPO_ROOT,),
            allow_missing_final=False,
            require_dir=True,
        )
        return safe_path(
            path,
            base=REPO_ROOT,
            approved_roots=(safe_root,),
            allow_missing_final=allow_missing_final,
            require_dir=require_dir,
        )
    except SafePathError:
        raise ServingPackageError("private package path is unsafe") from None


def validate_private_destination(
    path: Path = PRIVATE_PACKAGE_ROOT,
    *,
    allow_missing_final: bool = True,
    require_dir: bool = False,
) -> Path:
    """Validate the one allowed package path under this repository's private root."""
    expected = Path(os.path.normcase(os.path.abspath(PRIVATE_PACKAGE_ROOT)))
    candidate = Path(os.path.normcase(os.path.abspath(path)))
    if candidate != expected:
        raise ServingPackageError("private package destination is not the expected package path")
    return _safe_private_tree_path(
        Path(path), allow_missing_final=allow_missing_final, require_dir=require_dir
    )


def validate_private_tree_path(
    path: Path,
    *,
    allow_missing_final: bool = True,
    require_dir: bool = False,
) -> Path:
    """Validate any path component below the exact repository private root."""
    return _safe_private_tree_path(
        Path(path), allow_missing_final=allow_missing_final, require_dir=require_dir
    )


def _safe_relative_artifact(raw: object) -> str:
    if (
        type(raw) is not str
        or not raw
        or "\\" in raw
        or raw.startswith("/")
        or raw.startswith("~")
        or Path(raw).is_absolute()
        or any(part in {"", ".", ".."} for part in raw.split("/"))
        or Path(raw).as_posix() != raw
    ):
        raise ServingPackageError("serving package artifact path is invalid")
    return raw


def _strict_mapping(value: object, keys: frozenset[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ServingPackageError(f"serving package {name} schema is invalid")
    return value


def _validate_manifest_identity(manifest: dict[str, Any], policy: ServingPackagePolicy) -> None:
    if (
        manifest["schema"] != policy.schema
        or manifest["package_id"] != policy.package_id
        or manifest["capability"] != policy.capability
    ):
        raise ServingPackageError("serving package manifest identity is invalid")


def _validate_artifact(manifest: dict[str, Any], policy: ServingPackagePolicy) -> None:
    artifact = _strict_mapping(manifest["artifact"], _ARTIFACT_KEYS, "artifact")
    if artifact["path"] != policy.artifact_path or artifact["format"] != policy.artifact_format:
        raise ServingPackageError("serving package artifact declaration is invalid")
    if type(artifact["size_bytes"]) is not int or artifact["size_bytes"] != policy.artifact_size_bytes:
        raise ServingPackageError("serving package artifact size is invalid")
    if not _digest(artifact["sha256"]) or artifact["sha256"] != policy.artifact_sha256:
        raise ServingPackageError("serving package artifact digest is invalid")


def _validate_documentation(manifest: dict[str, Any], policy: ServingPackagePolicy) -> None:
    documentation = _strict_mapping(manifest["documentation"], _DOCUMENTATION_KEYS, "documentation")
    expected = {
        "model_card": (policy.model_card_path, policy.model_card_sha256),
        "third_party_notices": (policy.third_party_notices_path, policy.third_party_notices_sha256),
    }
    for name, (path, digest) in expected.items():
        entry = _strict_mapping(documentation[name], _DOCUMENT_ENTRY_KEYS, f"documentation {name}")
        if entry["path"] != path or not _digest(entry["sha256"] ) or entry["sha256"] != digest:
            raise ServingPackageError("serving package documentation declaration is invalid")


def _validate_license(manifest: dict[str, Any], policy: ServingPackagePolicy) -> None:
    license_data = _strict_mapping(manifest["artifact_license"], _LICENSE_KEYS, "artifact license")
    expected = {
        "status": policy.artifact_license_status,
        "code_license": policy.code_license,
        "model_artifact_license": policy.model_artifact_license,
        "dataset_license": policy.dataset_license,
        "pretrained_weights_permission": policy.pretrained_weights_permission,
        "private_distribution_acceptance": policy.private_distribution_acceptance,
    }
    if license_data != expected:
        raise ServingPackageError("serving package license declaration is invalid")


def _validate_source(manifest: dict[str, Any], policy: ServingPackagePolicy) -> None:
    source = _strict_mapping(manifest["source_checkpoint"], _SOURCE_KEYS, "source checkpoint")
    if (
        source["role"] != policy.source_role
        or source["sha256"] != policy.source_checkpoint_sha256
        or type(source["epoch"]) is not int
        or source["epoch"] != policy.source_epoch
        or type(source["best_epoch"]) is not int
        or source["best_epoch"] != policy.source_best_epoch
        or not _digest(source["selection_identity"])
    ):
        raise ServingPackageError("serving package source checkpoint declaration is invalid")


def _validate_model(manifest: dict[str, Any], policy: ServingPackagePolicy) -> None:
    model = _strict_mapping(manifest["model"], _MODEL_KEYS, "model")
    if (
        model["architecture"] != policy.architecture
        or type(model["imgsz"]) is not int
        or model["imgsz"] != policy.imgsz
        or type(model["classes"]) is not list
        or any(type(value) is not int for value in model["classes"])
        or model["classes"] != list(policy.classes)
        or type(model["thresholds"]) is not int
        or model["thresholds"] != policy.thresholds
        or policy.thresholds != NUM_THRESHOLDS
    ):
        raise ServingPackageError("serving package model declaration is invalid")


def _validate_lineage(manifest: dict[str, Any], policy: ServingPackagePolicy) -> None:
    lineage = _strict_mapping(manifest["lineage"], _LINEAGE_KEYS, "lineage")
    for field in ("snapshot_identity", "dataset_manifest_digest", "config_sha256"):
        if not _digest(lineage[field]):
            raise ServingPackageError("serving package lineage digest is invalid")
    if not _digest(lineage["run_id"], 32):
        raise ServingPackageError("serving package lineage run id is invalid")
    if (
        lineage["checkpoint_schema_version"] != policy.checkpoint_schema_version
        or lineage["domain_id"] != policy.domain_id
        or lineage["snapshot_schema"] != policy.snapshot_schema
        or lineage["source_schema"] != policy.source_schema
        or lineage["source_identity_scheme"] != policy.source_identity_scheme
        or type(lineage["source_mapping"]) is not dict
        or {key: lineage["source_mapping"][key] for key in sorted(lineage["source_mapping"])}
        != dict(policy.source_mapping)
        or type(lineage["observed_classes"]) is not list
        or any(type(value) is not int for value in lineage["observed_classes"])
        or tuple(lineage["observed_classes"]) != policy.observed_classes
        or type(lineage["missing_classes"]) is not list
        or any(type(value) is not int for value in lineage["missing_classes"])
        or tuple(lineage["missing_classes"]) != policy.missing_classes
    ):
        raise ServingPackageError("serving package lineage is invalid")


def _validate_approval(manifest: dict[str, Any], policy: ServingPackagePolicy) -> None:
    approval = _strict_mapping(manifest["approval"], _APPROVAL_KEYS, "approval")
    if (
        approval["status"] != policy.approval_status
        or approval["serving_status"] != policy.serving_status
        or approval["validation_status"] != policy.validation_status
        or type(approval["failed_check_ids"]) is not list
        or any(type(value) is not str for value in approval["failed_check_ids"])
        or tuple(approval["failed_check_ids"]) != policy.failed_check_ids
    ):
        raise ServingPackageError("serving package approval declaration is invalid")


def validate_manifest(value: object, policy: ServingPackagePolicy = PACKAGE_POLICY) -> dict[str, Any]:
    """Validate one exact manifest against its immutable package-instance policy."""
    manifest = _strict_mapping(value, _MANIFEST_KEYS, "manifest")
    _validate_manifest_identity(manifest, policy)
    _validate_artifact(manifest, policy)
    _validate_documentation(manifest, policy)
    _validate_license(manifest, policy)
    _validate_source(manifest, policy)
    _validate_model(manifest, policy)
    _validate_lineage(manifest, policy)
    _validate_approval(manifest, policy)
    if manifest["activation"] != policy.activation or manifest["report_path"] != policy.report_path:
        raise ServingPackageError("serving package activation declaration is invalid")
    return manifest


def _validate_manifest(value: object) -> dict[str, Any]:
    """Backward-compatible internal alias used by focused validation tests."""
    return validate_manifest(value, PACKAGE_POLICY)


def _validate_state_dict(raw: object, model: BCSOrdinalModel) -> dict[str, torch.Tensor]:
    if not isinstance(raw, Mapping) or not raw:
        raise ServingPackageError("serving package artifact is not a state dictionary")
    if any(type(key) is not str or not isinstance(value, torch.Tensor) for key, value in raw.items()):
        raise ServingPackageError("serving package state dictionary contains invalid values")
    expected = model.state_dict()
    if set(raw) != set(expected):
        raise ServingPackageError("serving package state dictionary architecture is invalid")
    for name, tensor in raw.items():
        if tensor.dtype != expected[name].dtype or tuple(tensor.shape) != tuple(expected[name].shape):
            raise ServingPackageError("serving package state dictionary tensor is invalid")
    return dict(raw)


def build_manifest(
    source: Mapping[str, Any],
    artifact_size: int,
    artifact_sha256: str,
    documentation: Mapping[str, str],
    policy: ServingPackagePolicy = PACKAGE_POLICY,
) -> dict[str, Any]:
    """Build the manifest consumed by the loader from measured artifact data."""
    if (
        type(artifact_size) is not int
        or artifact_size != policy.artifact_size_bytes
        or artifact_sha256 != policy.artifact_sha256
        or not _digest(artifact_sha256)
    ):
        raise ServingPackageError("serving package artifact measurement is invalid")
    source_mapping = source.get("source_mapping")
    canonical_mapping = (
        {key: source_mapping[key] for key in sorted(source_mapping)}
        if isinstance(source_mapping, dict)
        else None
    )
    expected_mapping = dict(policy.source_mapping)
    if (
        source.get("checkpoint_schema_version") != policy.checkpoint_schema_version
        or source.get("domain_id") != policy.domain_id
        or source.get("snapshot_schema") != policy.snapshot_schema
        or source.get("source_schema") != policy.source_schema
        or source.get("source_identity_scheme") != policy.source_identity_scheme
        or not isinstance(source_mapping, dict)
        or canonical_mapping != expected_mapping
        or source.get("config", {}).get("imgsz") != policy.imgsz
        or source.get("num_thresholds") != policy.thresholds
        or source.get("epoch") != policy.source_epoch
        or source.get("best_epoch") != policy.source_best_epoch
        or source.get("run_id") is None
        or type(documentation) is not dict
        or set(documentation) != {"model_card", "third_party_notices"}
        or documentation["model_card"] != policy.model_card_sha256
        or documentation["third_party_notices"] != policy.third_party_notices_sha256
    ):
        raise ServingPackageError("source metadata does not match serving package policy")
    lineage = {
        key: source[key]
        for key in (
            "checkpoint_schema_version",
            "domain_id",
            "snapshot_schema",
            "snapshot_identity",
            "dataset_manifest_digest",
            "run_id",
            "config_sha256",
            "source_schema",
            "source_identity_scheme",
            "source_mapping",
            "observed_classes",
            "missing_classes",
        )
    }
    lineage["source_mapping"] = canonical_mapping
    return {
        "schema": policy.schema,
        "package_id": policy.package_id,
        "capability": policy.capability,
        "artifact": {
            "path": policy.artifact_path,
            "format": policy.artifact_format,
            "size_bytes": artifact_size,
            "sha256": artifact_sha256,
        },
        "source_checkpoint": {
            "role": policy.source_role,
            "sha256": policy.source_checkpoint_sha256,
            "epoch": policy.source_epoch,
            "best_epoch": policy.source_best_epoch,
            "selection_identity": source["selection_identity"],
        },
        "model": {
            "architecture": policy.architecture,
            "imgsz": policy.imgsz,
            "classes": list(policy.classes),
            "thresholds": policy.thresholds,
        },
        "lineage": lineage,
        "documentation": {
            "model_card": {"path": policy.model_card_path, "sha256": documentation["model_card"]},
            "third_party_notices": {
                "path": policy.third_party_notices_path,
                "sha256": documentation["third_party_notices"],
            },
        },
        "artifact_license": {
            "status": policy.artifact_license_status,
            "code_license": policy.code_license,
            "model_artifact_license": policy.model_artifact_license,
            "dataset_license": policy.dataset_license,
            "pretrained_weights_permission": policy.pretrained_weights_permission,
            "private_distribution_acceptance": policy.private_distribution_acceptance,
        },
        "approval": {
            "status": policy.approval_status,
            "serving_status": policy.serving_status,
            "validation_status": policy.validation_status,
            "failed_check_ids": list(policy.failed_check_ids),
        },
        "activation": policy.activation,
        "report_path": policy.report_path,
    }


def _validate_artifact_bytes(
    manifest: dict[str, Any], raw: bytes, policy: ServingPackagePolicy = PACKAGE_POLICY
) -> None:
    artifact = manifest["artifact"]
    if (
        artifact["size_bytes"] != policy.artifact_size_bytes
        or artifact["sha256"] != policy.artifact_sha256
    ):
        raise ServingPackageError("serving package artifact is not trusted by its policy")
    if len(raw) != policy.artifact_size_bytes:
        raise ServingPackageError("serving package artifact size does not match its manifest")
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), policy.artifact_sha256):
        raise ServingPackageError("serving package artifact digest does not match its manifest")


def _validate_documentation_files(
    manifest: dict[str, Any], package_root: Path, policy: ServingPackagePolicy
) -> None:
    documentation = manifest["documentation"]
    required = {
        "model_card": _MODEL_CARD_REQUIRED_PHRASES,
        "third_party_notices": _NOTICES_REQUIRED_PHRASES,
    }
    for name, phrases in required.items():
        entry = documentation[name]
        path = _safe_package_file(package_root / _safe_relative_artifact(entry["path"]), package_root)
        raw = _read_bounded_file(path, MAX_DOCUMENT_BYTES)
        if not raw:
            raise ServingPackageError("serving package documentation is empty")
        if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), entry["sha256"]):
            raise ServingPackageError("serving package documentation digest does not match its manifest")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise ServingPackageError("serving package documentation is not UTF-8") from None
        if any(phrase.casefold() not in text.casefold() for phrase in phrases):
            raise ServingPackageError("serving package documentation is incomplete")


def _read_manifest(path: Path, package_root: Path, policy: ServingPackagePolicy) -> dict[str, Any]:
    manifest_raw = _read_bounded_file(_safe_package_file(path, package_root), MAX_MANIFEST_BYTES)
    try:
        value = json.loads(manifest_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ServingPackageError("serving package manifest is invalid UTF-8 JSON") from None
    return validate_manifest(value, policy)


def validate_package_directory(
    package_root: Path,
    policy: ServingPackagePolicy = PACKAGE_POLICY,
) -> dict[str, Any]:
    """Validate a complete staged package, including its artifact bytes."""
    try:
        safe_root = safe_path(
            package_root,
            base=REPO_ROOT,
            approved_roots=(package_root,),
            allow_missing_final=False,
            require_dir=True,
        )
    except SafePathError:
        raise ServingPackageError("serving package root is unsafe") from None
    _validate_package_members(safe_root, policy)
    manifest = _read_manifest(safe_root / "manifest.json", safe_root, policy)
    artifact_path = _safe_package_file(safe_root / _safe_relative_artifact(manifest["artifact"]["path"]), safe_root)
    _validate_artifact_bytes(
        manifest, _read_bounded_file(artifact_path, MAX_ARTIFACT_BYTES), policy
    )
    _validate_documentation_files(manifest, safe_root, policy)
    return manifest


def _validate_package_members(
    package_root: Path, policy: ServingPackagePolicy = PACKAGE_POLICY
) -> None:
    expected = {
        "manifest.json",
        policy.artifact_path,
        policy.model_card_path,
        policy.third_party_notices_path,
    }
    actual: set[str] = set()
    for entry in package_root.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise ServingPackageError("serving package contains an unsafe or unexpected member")
        actual.add(entry.name)
    if actual != expected:
        raise ServingPackageError("serving package contains unexpected or missing members")


def validate_catalog_manifest() -> dict[str, Any]:
    """Validate the Git-tracked catalog instance without requiring private weights."""
    expected = {
        "manifest.json",
        "archive.json",
        PACKAGE_POLICY.model_card_path,
        PACKAGE_POLICY.third_party_notices_path,
    }
    try:
        actual = {entry.name for entry in CATALOG_PACKAGE_ROOT.iterdir()}
    except OSError:
        raise ServingPackageError("serving package catalog is unavailable") from None
    if actual != expected:
        raise ServingPackageError("serving package catalog contains unexpected members")
    try:
        archive_catalog = json.loads(
            _read_bounded_file(CATALOG_PACKAGE_ROOT / "archive.json", MAX_MANIFEST_BYTES).decode("utf-8")
        )
    except (ServingPackageError, UnicodeDecodeError, json.JSONDecodeError):
        raise ServingPackageError("serving package archive catalog is invalid") from None
    if archive_catalog != {
        "archive_filename": PACKAGE_POLICY.archive_filename,
        "archive_directory": PACKAGE_POLICY.archive_directory,
        "archive_sha256": PACKAGE_POLICY.archive_sha256,
        "archive_size_bytes": PACKAGE_POLICY.archive_size_bytes,
        "package_id": PACKAGE_POLICY.package_id,
        "schema": "vacca-bcs-private-archive-v1",
    }:
        raise ServingPackageError("serving package archive catalog is not trusted")
    manifest = _read_manifest(CATALOG_MANIFEST_PATH, CATALOG_PACKAGE_ROOT, PACKAGE_POLICY)
    _validate_documentation_files(manifest, CATALOG_PACKAGE_ROOT, PACKAGE_POLICY)
    return manifest


def validate_installed_package() -> dict[str, Any]:
    """Read and validate the private installed package without loading a model."""
    safe_root = validate_private_destination(PRIVATE_PACKAGE_ROOT, allow_missing_final=False, require_dir=True)
    _validate_package_members(safe_root, PACKAGE_POLICY)
    manifest = _read_manifest(PRIVATE_MANIFEST_PATH, safe_root, PACKAGE_POLICY)
    artifact_path = _safe_package_file(
        safe_root / _safe_relative_artifact(manifest["artifact"]["path"]), safe_root
    )
    _validate_artifact_bytes(
        manifest, _read_bounded_file(artifact_path, MAX_ARTIFACT_BYTES), PACKAGE_POLICY
    )
    _validate_documentation_files(manifest, safe_root, PACKAGE_POLICY)
    return manifest


def find_recoverable_backups() -> tuple[Path, ...]:
    """List retained, valid package backups without exposing paths through the API."""
    if not PRIVATE_ROOT.exists() or PRIVATE_ROOT.is_symlink():
        return ()
    try:
        private_root = validate_private_tree_path(
            PRIVATE_ROOT, allow_missing_final=False, require_dir=True
        )
    except ServingPackageError:
        return ()
    found: list[Path] = []
    for candidate in private_root.iterdir():
        if (
            candidate.name != LEGACY_RECOVERY_BACKUP_NAME
            and not candidate.name.startswith(RECOVERY_BACKUP_PREFIX)
            and not candidate.name.startswith(DISABLED_RECOVERY_PREFIX)
        ):
            continue
        try:
            validate_private_tree_path(candidate, allow_missing_final=False, require_dir=True)
            validate_package_directory(candidate, PACKAGE_POLICY)
        except (OSError, ServingPackageError):
            continue
        found.append(candidate)
    return tuple(sorted(found, key=lambda path: path.name))


def _load_package_model(
    package_root: Path,
    _checkpoint_path: str | os.PathLike[str] | None = None,
    device: str | torch.device = "cpu",
    *,
    expected_sha256: str | None = None,
) -> LoadedBCSModel:
    """Load an installed package from the exact bytes declared by its policy."""
    try:
        manifest = validate_package_directory(package_root, PACKAGE_POLICY)
        if _checkpoint_path is not None or expected_sha256 is not None:
            raise ServingPackageError("installed package loader received external parameters")
        artifact = manifest["artifact"]
        artifact_path = _safe_package_file(
            package_root / _safe_relative_artifact(artifact["path"]), package_root
        )
        raw = _read_bounded_file(artifact_path, MAX_ARTIFACT_BYTES)
        _validate_artifact_bytes(manifest, raw, PACKAGE_POLICY)
        state = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
        model = BCSOrdinalModel(pretrained=False)
        state_dict = _validate_state_dict(state, model)
        model.load_state_dict(state_dict, strict=True)
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise ServingPackageError("requested CUDA device is unavailable")
        model.to(resolved).eval()
    except ServingPackageError:
        raise
    except Exception:
        raise ServingPackageError("serving package could not be loaded safely") from None
    lineage_data = manifest["lineage"]
    lineage = BCSLineageMetadata(
        checkpoint_schema_version=lineage_data["checkpoint_schema_version"],
        domain_id=lineage_data["domain_id"],
        snapshot_schema=lineage_data["snapshot_schema"],
        snapshot_identity=lineage_data["snapshot_identity"],
        dataset_manifest_digest=lineage_data["dataset_manifest_digest"],
        run_id=lineage_data["run_id"],
        source_schema=lineage_data["source_schema"],
        source_identity_scheme=lineage_data["source_identity_scheme"],
        source_mapping=tuple(sorted(lineage_data["source_mapping"].items())),
        observed_classes=tuple(lineage_data["observed_classes"]),
        missing_classes=tuple(lineage_data["missing_classes"]),
    )
    return LoadedBCSModel(
        model=model,
        imgsz=manifest["model"]["imgsz"],
        device=resolved,
        lineage=lineage,
        checkpoint_sha256=artifact["sha256"],
        checkpoint=None,
    )


def load_installed_bcs_model(
    _checkpoint_path: str | os.PathLike[str] | None = None,
    device: str | torch.device = "cpu",
    *,
    expected_sha256: str | None = None,
) -> LoadedBCSModel:
    """Load the verified package installed under the ignored private destination."""
    return _load_package_model(
        PRIVATE_PACKAGE_ROOT,
        _checkpoint_path,
        device,
        expected_sha256=expected_sha256,
    )


def load_package_model(
    package_root: Path,
    device: str | torch.device = "cpu",
) -> LoadedBCSModel:
    """Validate and load a staged package using the production loader."""
    return _load_package_model(package_root, device=device)
