"""Verify and atomically install the private BCS serving ZIP bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import time
from zipfile import BadZipFile, ZipFile

from vacca_bcs.serving_package import (
    ARCHIVE_COMPRESSION,
    MAX_ARCHIVE_COMPRESSION_RATIO,
    MAX_ARCHIVE_UNCOMPRESSED_BYTES,
    MAX_SIDECAR_BYTES,
    PACKAGE_MEMBER_LIMITS,
    PACKAGE_MEMBER_SET,
    PACKAGE_POLICY,
    REPO_ROOT,
    PRIVATE_ROOT,
    PRIVATE_PACKAGE_ROOT,
    ServingPackageError,
    load_package_model,
    validate_private_destination,
    validate_private_tree_path,
    validate_package_directory,
)
from vacca_bcs.path_safety import SafePathError, safe_path

_ARCHIVE_MAX_BYTES = MAX_ARCHIVE_UNCOMPRESSED_BYTES
_MEMBERS = PACKAGE_MEMBER_SET
_SIDECAR_PATTERN = re.compile(r"^([0-9a-f]{64})  ([^\r\n]+)\n$")
_BACKUP_PREFIX = f".{PACKAGE_POLICY.package_id}.backup-"
_LEGACY_BACKUP_NAME = f".{PACKAGE_POLICY.package_id}.backup"
_QUARANTINE_PREFIX = f".{PACKAGE_POLICY.package_id}.quarantine-"
_DISABLED_PREFIX = f".{PACKAGE_POLICY.package_id}.disabled-"


def _regular_file(path: Path) -> None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        raise ServingPackageError("private archive is unavailable") from None
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ServingPackageError("private archive is unsafe")


def _read_sidecar(archive_path: Path) -> str:
    sidecar = Path(f"{archive_path}.sha256")
    _regular_file(sidecar)
    if sidecar.stat().st_size > MAX_SIDECAR_BYTES:
        raise ServingPackageError("private archive sidecar is too large")
    try:
        value = sidecar.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError):
        raise ServingPackageError("private archive sidecar is invalid") from None
    match = _SIDECAR_PATTERN.fullmatch(value)
    if match is None or match.group(2) != archive_path.name:
        raise ServingPackageError("private archive sidecar does not match the archive")
    return match.group(1)


def _member_limit(name: str) -> int:
    return PACKAGE_MEMBER_LIMITS[name]


def _read_archive(archive_path: Path) -> dict[str, bytes]:
    _regular_file(archive_path)
    if PACKAGE_POLICY.archive_size_bytes <= 0 or not PACKAGE_POLICY.archive_sha256.strip("0"):
        raise ServingPackageError("expected private archive digest is not configured")
    if archive_path.stat().st_size != PACKAGE_POLICY.archive_size_bytes:
        raise ServingPackageError("private archive size is not trusted")
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if digest != PACKAGE_POLICY.archive_sha256:
        raise ServingPackageError("private archive digest is not trusted")
    if _read_sidecar(archive_path) != digest:
        raise ServingPackageError("private archive sidecar digest is not trusted")

    try:
        with ZipFile(archive_path) as archive:
            if archive.comment:
                raise ServingPackageError("private archive comment is not allowed")
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or set(names) != _MEMBERS:
                raise ServingPackageError("private archive members are invalid")
            total = 0
            result: dict[str, bytes] = {}
            for info in infos:
                name = info.filename
                if (
                    not name
                    or "\\" in name
                    or name.startswith(("/", "~"))
                    or "\x00" in name
                    or any(part in {"", ".", ".."} for part in name.split("/"))
                    or info.is_dir()
                    or info.extra
                    or info.comment
                    or info.flag_bits & 0x1
                    or info.compress_type != ARCHIVE_COMPRESSION
                    or stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF)
                ):
                    raise ServingPackageError("private archive contains an unsafe member")
                limit = _member_limit(name)
                if info.file_size > limit or info.compress_size <= 0:
                    raise ServingPackageError("private archive member exceeds its limit")
                if info.file_size / info.compress_size > MAX_ARCHIVE_COMPRESSION_RATIO:
                    raise ServingPackageError("private archive member compression ratio is excessive")
                total += info.file_size
                if total > _ARCHIVE_MAX_BYTES:
                    raise ServingPackageError("private archive is too large when expanded")
                with archive.open(info, "r") as handle:
                    payload = handle.read(limit + 1)
                if len(payload) != info.file_size:
                    raise ServingPackageError("private archive member size changed while reading")
                result[name] = payload
            return result
    except BadZipFile:
        raise ServingPackageError("private archive is not a valid ZIP") from None


def _ensure_private_root() -> Path:
    parent = PRIVATE_ROOT.parent
    try:
        safe_path(parent, base=REPO_ROOT, approved_roots=(REPO_ROOT,), allow_missing_final=False, require_dir=True)
    except SafePathError:
        raise ServingPackageError("private package parent is unsafe") from None
    if not PRIVATE_ROOT.exists():
        PRIVATE_ROOT.mkdir()
    return validate_private_tree_path(PRIVATE_ROOT, allow_missing_final=False, require_dir=True)


def _entry_exists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _new_private_entry(private_root: Path, prefix: str) -> Path:
    """Return a collision-free direct child name without touching the filesystem."""
    timestamp = time.time_ns()
    pid = os.getpid()
    for counter in range(1000):
        candidate = private_root / f"{prefix}{timestamp:x}-{pid:x}-{counter:x}"
        if not _entry_exists(candidate):
            validate_private_tree_path(candidate, allow_missing_final=True)
            return candidate
    raise ServingPackageError("could not allocate a private recovery path")


def _safe_remove_tree(path: Path) -> None:
    safe_path = validate_private_tree_path(path, allow_missing_final=False, require_dir=True)
    shutil.rmtree(safe_path)


def _preserve_staging(staging: Path, private_root: Path) -> Path | None:
    if not _entry_exists(staging):
        return None
    quarantine = _new_private_entry(private_root, _QUARANTINE_PREFIX)
    try:
        os.rename(staging, quarantine)
    except OSError:
        return staging
    return quarantine


def install(archive_path: Path, *, repair: bool = False) -> str:
    """Verify, stage, and publish while retaining every recoverable package copy."""
    private_root = _ensure_private_root()
    destination = validate_private_destination(PRIVATE_PACKAGE_ROOT, allow_missing_final=True)
    members = _read_archive(Path(archive_path))
    staging = Path(tempfile.mkdtemp(prefix=f".{PACKAGE_POLICY.package_id}.install-", dir=private_root))
    validate_private_tree_path(staging, allow_missing_final=False, require_dir=True)
    backup: Path | None = None
    publish_attempted = False
    transaction_started = False
    try:
        for name, payload in members.items():
            (staging / name).write_bytes(payload)
        validate_package_directory(staging, PACKAGE_POLICY)
        load_package_model(staging, device="cpu")

        destination_exists = _entry_exists(destination)
        if destination_exists:
            try:
                validate_private_destination(destination, allow_missing_final=False, require_dir=True)
                validate_package_directory(destination, PACKAGE_POLICY)
            except ServingPackageError as existing_error:
                if not repair:
                    raise FileExistsError(
                        f"mismatched private package already exists: {destination}; use --repair"
                    ) from existing_error
                # A corrupt package is still retained as the recoverable backup.
                validate_private_destination(destination, allow_missing_final=False)
            else:
                # A valid identical package is always an immutable no-op, even with --repair.
                return PACKAGE_POLICY.package_id

        if destination_exists:
            backup = _new_private_entry(private_root, _BACKUP_PREFIX)
            os.rename(destination, backup)
            transaction_started = True

        publish_attempted = True
        try:
            os.rename(staging, destination)
        except Exception as publish_error:
            quarantine = _preserve_staging(staging, private_root)
            staging = None
            if backup is None:
                raise publish_error
            try:
                os.rename(backup, destination)
            except Exception as rollback_error:
                raise ServingPackageError(
                    "private package publish and rollback both failed; "
                    f"backup preserved at {backup}; replacement preserved at {quarantine or staging}. "
                    f"Run --recover-backup \"{backup}\" after stopping the API."
                ) from rollback_error
            backup = None
            raise publish_error

        staging = None
        try:
            validate_private_destination(destination, allow_missing_final=False, require_dir=True)
            validate_package_directory(destination, PACKAGE_POLICY)
            load_package_model(destination, device="cpu")
        except Exception as publish_validation_error:
            quarantine = _new_private_entry(private_root, _QUARANTINE_PREFIX)
            try:
                os.rename(destination, quarantine)
            except Exception as quarantine_error:
                quarantine = destination
                if backup is not None:
                    raise ServingPackageError(
                        "published package validation failed and the replacement could not be quarantined; "
                        f"backup preserved at {backup}; destination preserved at {quarantine}"
                    ) from quarantine_error
                raise ServingPackageError(
                    f"published package validation failed; destination preserved at {quarantine}"
                ) from publish_validation_error
            if backup is None:
                raise ServingPackageError(
                    f"published package validation failed; replacement quarantined at {quarantine}"
                ) from publish_validation_error
            try:
                os.rename(backup, destination)
            except Exception as rollback_error:
                raise ServingPackageError(
                    "published package validation and rollback both failed; "
                    f"backup preserved at {backup}; replacement quarantined at {quarantine}. "
                    f"Run --recover-backup \"{backup}\" after stopping the API."
                ) from rollback_error
            backup = None
            raise ServingPackageError(
                f"published package validation failed; replacement quarantined at {quarantine}; "
                "previous package restored"
            ) from publish_validation_error

        return PACKAGE_POLICY.package_id
    finally:
        if staging is not None and _entry_exists(staging):
            if publish_attempted or transaction_started:
                _preserve_staging(staging, private_root)
            else:
                _safe_remove_tree(staging)


def _is_recovery_backup_name(name: str) -> bool:
    return (
        name == _LEGACY_BACKUP_NAME
        or name.startswith(_BACKUP_PREFIX)
        or name.startswith(_DISABLED_PREFIX)
    )


def find_recoverable_backups() -> tuple[Path, ...]:
    """List safe, retained backups without exposing them through the API."""
    if not _entry_exists(PRIVATE_ROOT) or PRIVATE_ROOT.is_symlink():
        return ()
    private_root = validate_private_tree_path(PRIVATE_ROOT, allow_missing_final=False, require_dir=True)
    found: list[Path] = []
    for candidate in private_root.iterdir():
        if not _is_recovery_backup_name(candidate.name):
            continue
        try:
            validate_private_tree_path(candidate, allow_missing_final=False, require_dir=True)
            validate_package_directory(candidate, PACKAGE_POLICY)
        except (OSError, ServingPackageError):
            continue
        found.append(candidate)
    return tuple(sorted(found, key=lambda path: path.name))


def recover_backup(backup_path: Path) -> str:
    """Restore a retained backup without deleting an existing destination."""
    private_root = _ensure_private_root()
    candidate = Path(os.path.normcase(os.path.abspath(backup_path)))
    expected_root = Path(os.path.normcase(os.path.abspath(private_root)))
    if candidate.parent != expected_root or not _is_recovery_backup_name(candidate.name):
        raise ServingPackageError("recovery backup path is not a retained private backup")
    validate_private_tree_path(candidate, allow_missing_final=False, require_dir=True)
    validate_package_directory(candidate, PACKAGE_POLICY)
    load_package_model(candidate, device="cpu")

    destination = validate_private_destination(PRIVATE_PACKAGE_ROOT, allow_missing_final=True)
    if _entry_exists(destination):
        try:
            validate_private_destination(destination, allow_missing_final=False, require_dir=True)
            validate_package_directory(destination, PACKAGE_POLICY)
        except ServingPackageError:
            validate_private_destination(destination, allow_missing_final=False)
            quarantine = _new_private_entry(private_root, _QUARANTINE_PREFIX)
            try:
                os.rename(destination, quarantine)
            except Exception as quarantine_error:
                raise ServingPackageError(
                    f"invalid destination could not be quarantined; backup preserved at {candidate}; "
                    f"destination preserved at {destination}"
                ) from quarantine_error
        else:
            raise ServingPackageError(
                "recovery refused because the fixed private destination is already valid; no bytes changed"
            )
    try:
        os.rename(candidate, destination)
    except Exception as restore_error:
        raise ServingPackageError(
            f"recovery failed; backup preserved at {candidate}; destination remains unavailable at {destination}"
        ) from restore_error
    return PACKAGE_POLICY.package_id


def uninstall() -> Path | None:
    """Disable the fixed package by retaining it under a validated quarantine name."""
    expected = Path(os.path.normcase(os.path.abspath(PRIVATE_PACKAGE_ROOT)))
    try:
        safe_path(
            PRIVATE_ROOT,
            base=REPO_ROOT,
            approved_roots=(REPO_ROOT,),
            allow_missing_final=True,
        )
    except SafePathError:
        raise ServingPackageError("private package root is unsafe") from None
    if not PRIVATE_ROOT.exists() and not PRIVATE_ROOT.is_symlink():
        return None
    destination = validate_private_destination(PRIVATE_PACKAGE_ROOT, allow_missing_final=True)
    if not _entry_exists(destination):
        return None
    validate_private_destination(destination, allow_missing_final=False, require_dir=True)
    if Path(os.path.normcase(os.path.abspath(destination))) != expected:
        raise ServingPackageError("refusing to remove an unexpected private package path")
    validate_package_directory(destination, PACKAGE_POLICY)
    load_package_model(destination, device="cpu")
    disabled = _new_private_entry(PRIVATE_ROOT, _DISABLED_PREFIX)
    os.rename(destination, disabled)
    return disabled


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--archive", type=Path, help="Validate and install a private ZIP after acceptance and authorized local receipt.")
    action.add_argument("--uninstall", action="store_true", help="Disable the validated package by retaining a reversible backup.")
    action.add_argument("--recover-backup", type=Path, help="Restore a retained backup without deleting a valid destination.")
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Repair an invalid destination only after staging, loading, and validating the replacement; retain rollback bytes.",
    )
    args = parser.parse_args()
    if args.uninstall:
        if args.repair:
            parser.error("--repair cannot be combined with --uninstall")
        disabled = uninstall()
        print(
            json.dumps(
                {
                    "package_id": PACKAGE_POLICY.package_id,
                    "disabled_backup": str(disabled) if disabled else None,
                }
            )
        )
    elif args.recover_backup is not None:
        if args.repair:
            parser.error("--repair cannot be combined with --recover-backup")
        print(json.dumps({"package_id": recover_backup(args.recover_backup)}))
    else:
        print(json.dumps({"package_id": install(args.archive, repair=args.repair)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
