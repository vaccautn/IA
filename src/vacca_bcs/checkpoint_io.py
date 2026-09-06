"""Bounded, same-bytes checkpoint loading for BCS artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import hmac
import json
import os
import stat
from pathlib import Path
from typing import Any

import torch

from .path_safety import SafePathError, safe_path

MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024
CHECKPOINT_SET_SCHEMA = "vacca-bcs-checkpoint-set-v1"


class CheckpointByteError(ValueError):
    """Raised when checkpoint bytes cannot be trusted or deserialized."""


class CheckpointByteUnavailableError(FileNotFoundError):
    """Raised when the approved checkpoint path is absent."""


@dataclass(frozen=True, slots=True)
class CheckpointBytes:
    path: Path
    raw: bytes
    sha256: str
    payload: dict[str, Any]


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_digest(value: object, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def load_checkpoint_bytes(
    path: str | os.PathLike[str],
    *,
    approved_roots: tuple[Path, ...],
    expected_sha256: str | None = None,
    require_checkpoint_set: bool = True,
) -> CheckpointBytes:
    """Validate, read once, hash, optionally trust-check, then deserialize bytes."""
    if expected_sha256 is not None and not _valid_sha256(expected_sha256):
        raise CheckpointByteError("checkpoint digest is invalid")
    safe = _safe_checkpoint_path(path, approved_roots=approved_roots)
    pointer = _checkpoint_set_reference(safe, approved_roots=approved_roots)
    if pointer is None and safe.name in {"best.pt", "last.pt"} and require_checkpoint_set:
        raise CheckpointByteError("authoritative checkpoint-set descriptor is required")
    if pointer is not None:
        generation = safe.parent / pointer["generation"]
        _generation_safe, raw = _read_checkpoint_file(
            generation, approved_roots=approved_roots
        )
        generation_digest = hashlib.sha256(raw).hexdigest()
        if generation_digest != pointer["sha256"]:
            raise CheckpointByteError("checkpoint generation digest does not match its descriptor")
        actual_sha256 = generation_digest
    else:
        safe, raw = _read_checkpoint_file(path, approved_roots=approved_roots)
        actual_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and not hmac.compare_digest(actual_sha256, expected_sha256):
        raise CheckpointByteError("checkpoint digest does not match the trusted digest")
    try:
        payload = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as error:
        raise CheckpointByteError(f"checkpoint could not be deserialized: {error}") from error
    if not isinstance(payload, dict):
        raise CheckpointByteError("checkpoint payload is not a dictionary")
    return CheckpointBytes(safe, raw, actual_sha256, payload)


def read_checkpoint_digest(
    path: str | os.PathLike[str], *, approved_roots: tuple[Path, ...],
    require_checkpoint_set: bool = True,
) -> str:
    """Return the digest committed by a checkpoint descriptor or file contents."""
    safe = _safe_checkpoint_path(path, approved_roots=approved_roots)
    pointer = _checkpoint_set_reference(safe, approved_roots=approved_roots)
    if pointer is None and safe.name in {"best.pt", "last.pt"} and require_checkpoint_set:
        raise CheckpointByteError("authoritative checkpoint-set descriptor is required")
    if pointer is None:
        _safe, raw = _read_checkpoint_file(path, approved_roots=approved_roots)
        return hashlib.sha256(raw).hexdigest()
    _generation_safe, generation_raw = _read_checkpoint_file(
        safe.parent / pointer["generation"], approved_roots=approved_roots
    )
    digest = hashlib.sha256(generation_raw).hexdigest()
    if digest != pointer["sha256"]:
        raise CheckpointByteError("checkpoint generation digest does not match its descriptor")
    return digest


def read_checkpoint_set(
    path: str | os.PathLike[str], *, approved_roots: tuple[Path, ...]
) -> dict[str, Any]:
    """Read and verify the authoritative best/last checkpoint-set descriptor."""
    safe, raw = _read_checkpoint_file(path, approved_roots=approved_roots)
    value = _parse_checkpoint_set(raw)
    for role in ("best", "last"):
        reference = value[role]
        try:
            _generation_safe, generation_raw = _read_checkpoint_file(
                safe.parent / reference["filename"], approved_roots=approved_roots
            )
        except CheckpointByteUnavailableError as error:
            raise CheckpointByteError(
                f"checkpoint-set {role} generation is unavailable"
            ) from error
        digest = hashlib.sha256(generation_raw).hexdigest()
        if digest != reference["sha256"]:
            raise CheckpointByteError(
                f"checkpoint-set {role} generation digest does not match its descriptor"
            )
    return value


def _parse_checkpoint_set(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise CheckpointByteError("checkpoint-set descriptor is invalid") from None
    if not isinstance(value, dict):
        raise CheckpointByteError("checkpoint-set descriptor is invalid")
    required = {
        "schema", "lineage_schema_version", "committed_epoch", "best", "last", "run_id",
        "domain_id", "source_schema", "snapshot_schema", "snapshot_identity",
        "dataset_manifest_digest", "config_sha256", "observed_classes",
        "missing_classes", "source_identity_scheme", "source_mapping",
        "best_epoch", "selection_identity", "best_validation",
    }
    if set(value) != required or value.get("schema") != CHECKPOINT_SET_SCHEMA:
        raise CheckpointByteError("checkpoint-set descriptor is invalid")
    if (
        type(value["committed_epoch"]) is not int
        or value["committed_epoch"] < 1
        or type(value["best_epoch"]) is not int
        or value["best_epoch"] < 1
        or not isinstance(value["selection_identity"], str)
        or not isinstance(value["lineage_schema_version"], str)
        or not isinstance(value["best_validation"], dict)
        or not _valid_digest(value["run_id"], 32)
        or not isinstance(value["domain_id"], str)
        or not isinstance(value["source_schema"], str)
        or not isinstance(value["snapshot_schema"], str)
        or not _valid_sha256(value["snapshot_identity"])
        or not _valid_sha256(value["dataset_manifest_digest"])
        or not _valid_sha256(value["config_sha256"])
        or not isinstance(value["observed_classes"], list)
        or not isinstance(value["missing_classes"], list)
        or not isinstance(value["source_identity_scheme"], str)
        or not isinstance(value["source_mapping"], dict)
    ):
        raise CheckpointByteError("checkpoint-set metadata is invalid")
    for role in ("best", "last"):
        reference = value[role]
        if (
            not isinstance(reference, dict)
            or set(reference) != {"filename", "sha256"}
            or not isinstance(reference["filename"], str)
            or not _valid_sha256(reference["sha256"])
            or Path(reference["filename"]).as_posix()
            != f"generations/{reference['sha256']}.pt"
        ):
            raise CheckpointByteError("checkpoint-set generation reference is invalid")
    return value


def _checkpoint_set_reference(
    safe: Path, *, approved_roots: tuple[Path, ...]
) -> dict[str, str] | None:
    if safe.name not in {"best.pt", "last.pt"}:
        return None
    set_path = safe.parent / "checkpoint_set.json"
    current_error: CheckpointByteError | None = None
    try:
        checkpoint_set = read_checkpoint_set(set_path, approved_roots=approved_roots)
    except CheckpointByteUnavailableError:
        checkpoint_set = None
    except CheckpointByteError as error:
        current_error = error
        checkpoint_set = None
    if checkpoint_set is None:
        recovery_path = safe.parent / "checkpoint_set.recovery.json"
        try:
            checkpoint_set = read_checkpoint_set(
                recovery_path, approved_roots=approved_roots
            )
        except CheckpointByteUnavailableError:
            if current_error is not None:
                raise current_error
            return None
        except CheckpointByteError:
            if current_error is not None:
                raise current_error
            raise
    return {
        "generation": checkpoint_set[safe.stem]["filename"],
        "sha256": checkpoint_set[safe.stem]["sha256"],
    }


def _read_checkpoint_file(
    path: str | os.PathLike[str], *, approved_roots: tuple[Path, ...]
) -> tuple[Path, bytes]:
    try:
        safe = safe_path(
            path,
            base=Path(__file__).resolve().parents[2],
            approved_roots=approved_roots,
            allow_missing_final=True,
        )
    except SafePathError:
        raise CheckpointByteError("checkpoint path is unsafe") from None
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        before = os.lstat(safe)
        if stat.S_ISLNK(before.st_mode):
            raise CheckpointByteError("checkpoint path is unsafe")
        descriptor = os.open(str(safe), flags)
    except FileNotFoundError:
        raise CheckpointByteUnavailableError("checkpoint is unavailable") from None
    except CheckpointByteError:
        raise
    except OSError:
        raise CheckpointByteError("checkpoint cannot be opened safely") from None
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise CheckpointByteError("checkpoint changed during access")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1 << 20, MAX_CHECKPOINT_BYTES - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_CHECKPOINT_BYTES:
                raise CheckpointByteError("checkpoint exceeds the maximum size")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino):
            raise CheckpointByteError("checkpoint changed during access")
    except CheckpointByteError:
        raise
    except OSError:
        raise CheckpointByteError("checkpoint cannot be read safely") from None
    finally:
        os.close(descriptor)
    return safe, b"".join(chunks)


def _safe_checkpoint_path(
    path: str | os.PathLike[str], *, approved_roots: tuple[Path, ...]
) -> Path:
    try:
        return safe_path(
            path,
            base=Path(__file__).resolve().parents[2],
            approved_roots=approved_roots,
            allow_missing_final=True,
        )
    except SafePathError:
        raise CheckpointByteError("checkpoint path is unsafe") from None
