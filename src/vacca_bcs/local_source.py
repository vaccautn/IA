"""Safe discovery and bounded materialization of the local BCS source."""
from __future__ import annotations

import hashlib
import os
import re
import stat
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from vacca_vision.image_validation import SUPPORTED_EXTENSIONS

from .constants import BCS_CLASS_SCORES, NUM_CLASSES
from .path_safety import SafePathError, safe_path

LOCAL_SOURCE_SCHEMA = "bcs-local-category-source-v1"
DEFAULT_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_GS_YM_RE = re.compile(r"^(GS|YM)_([0-9]+)_([0-9]+)$")
_SIDE_RE = re.compile(r"^(L|R)-i([0-9]+)$")


class LocalSourceError(Exception):
    """Base class for sanitized local-source failures."""


class LocalSourceConfigurationError(LocalSourceError):
    pass


class LocalSourceScanError(LocalSourceError):
    pass


class LocalSourceCollisionError(LocalSourceScanError):
    pass


class LocalSourceMaterializationError(LocalSourceError):
    pass


@dataclass(frozen=True, slots=True)
class LocalSourceMapping:
    entries: tuple[tuple[str, int], ...] | Mapping[str, int]

    def __post_init__(self) -> None:
        try:
            entries = tuple(self.entries.items()) if isinstance(self.entries, Mapping) else tuple(self.entries)
            valid = all(
                    type(label) is str and bool(label) and type(category) is int and category in BCS_CLASS_SCORES
                for label, category in entries
            )
        except (TypeError, ValueError):
            entries, valid = (), False
        if not valid or len({label for label, _ in entries}) != len(entries):
            raise LocalSourceConfigurationError("local source mapping is malformed")
        object.__setattr__(self, "entries", tuple(sorted(entries)))

    @property
    def by_label(self) -> dict[str, int]:
        return dict(self.entries)


LOCAL_BCS_MAPPING = LocalSourceMapping(
    (("3.25", 1), ("3.5", 2), ("3.75", 3), ("4.0", 4), ("4.25", 5))
)


@dataclass(frozen=True, slots=True)
class LocalSourceRecord:
    record_id: str
    source_label: str
    bcs_category: int
    relative_path: str
    sha256: str
    capture_group: str = ""
    member_id: str = ""


@dataclass(frozen=True, slots=True)
class LocalSourceExclusion:
    record_id: str
    relative_path: str
    source_label: str
    bcs_category: int
    sha256: str
    reason: str

@dataclass(frozen=True, slots=True)
class LocalSourceScan:
    root: Path
    records: tuple[LocalSourceRecord, ...]
    counts: tuple[int, ...]
    observed_classes: tuple[int, ...]
    mapping_lineage: tuple[tuple[str, int], ...]
    exclusions: tuple[LocalSourceExclusion, ...] = ()

    def __post_init__(self) -> None:
        for field in ("records", "counts", "observed_classes", "mapping_lineage"):
            object.__setattr__(self, field, tuple(getattr(self, field)))


@dataclass(frozen=True, slots=True)
class LocalSourceMaterialized:
    record_id: str
    relative_path: str
    payload: bytes
    sha256: str
    size_bytes: int


def _relative(*parts: str) -> str:
    return unicodedata.normalize("NFC", "/".join(parts))


def _root_path(root: str | Path, *, approved_roots: tuple[Path, ...]) -> Path:
    try:
        return safe_path(
            root,
            base=Path.cwd(),
            approved_roots=approved_roots,
            allow_missing_final=False,
            require_dir=True,
        )
    except SafePathError:
        raise LocalSourceScanError("local source root is unavailable") from None


def _inside(root: Path, path: Path, relative: str, error=LocalSourceScanError) -> None:
    try:
        path.resolve(strict=False).relative_to(root)
    except (OSError, RuntimeError, ValueError):
        raise error(f"local source path escapes root: {relative}") from None


def _regular(path: Path, relative: str, error=LocalSourceScanError) -> os.stat_result:
    try:
        info = os.lstat(path)
    except OSError:
        raise error(f"local source file is unavailable: {relative}") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise error(f"local source file is not a regular file: {relative}")
    return info


def _sha256_file(path: Path) -> str:
    try:
        before = _regular(path, "file")
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise LocalSourceScanError("local source file changed during access")
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(stream.fileno())
    except LocalSourceError:
        raise
    except OSError:
        raise LocalSourceScanError("local source file cannot be read") from None
    if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
        raise LocalSourceScanError("local source file changed during access")
    return digest.hexdigest()


def _record_id(relative_path: str) -> str:
    return hashlib.sha256(f"{LOCAL_SOURCE_SCHEMA}\0{relative_path}".encode()).hexdigest()


def _mapping(value: LocalSourceMapping | Mapping[str, int]) -> LocalSourceMapping:
    try:
        result = value if isinstance(value, LocalSourceMapping) else LocalSourceMapping(value)
    except (TypeError, ValueError):
        raise LocalSourceConfigurationError("local source mapping is malformed") from None
    if result != LOCAL_BCS_MAPPING:
        raise LocalSourceConfigurationError("local source mapping is not the exact BCS category mapping")
    return result


def _capture_identity(name: str) -> tuple[str, str]:
    stem = unicodedata.normalize("NFC", Path(name).stem)
    match = _GS_YM_RE.fullmatch(stem)
    if match:
        prefix, series, view = match.groups()
        return f"{prefix}|{series}", f"{prefix}|{series}|{view}"
    match = _SIDE_RE.fullmatch(stem)
    if match:
        side, index = match.groups()
        return f"LR|{index}", f"{side}|{index}"
    raise LocalSourceScanError(f"local source filename has no unambiguous capture identity: {name}")


def scan_local_source(
    root: str | Path,
    mapping: LocalSourceMapping | Mapping[str, int] = LOCAL_BCS_MAPPING,
    *,
    approved_roots: tuple[Path, ...],
    digest_fn: Callable[[Path], str] = _sha256_file,
    record_id_fn: Callable[[str], str] = _record_id,
) -> LocalSourceScan:
    """Scan mapped folders and derive fail-closed capture groups from filenames."""
    source_root, source_mapping = _root_path(root, approved_roots=approved_roots), _mapping(mapping)
    labels, paths, images, sidecars = source_mapping.by_label, {}, [], []
    try:
        class_entries = tuple(os.scandir(source_root))
    except OSError:
        raise LocalSourceScanError("local source root cannot be read") from None
    for class_entry in class_entries:
        class_path, class_relative = source_root / class_entry.name, _relative(class_entry.name)
        _inside(source_root, class_path, class_relative)
        if class_entry.is_symlink() or not class_entry.is_dir(follow_symlinks=False) or class_entry.name not in labels:
            raise LocalSourceScanError(f"unexpected local source folder: {class_relative}")
        try:
            entries = tuple(os.scandir(class_path))
        except OSError:
            raise LocalSourceScanError(f"local source folder cannot be read: {class_relative}") from None
        for entry in entries:
            relative, path = _relative(class_entry.name, entry.name), class_path / entry.name
            _inside(source_root, path, relative)
            if entry.is_symlink() or entry.is_dir(follow_symlinks=False):
                raise LocalSourceScanError(f"nested or linked local source path: {relative}")
            _regular(path, relative)
            if relative.casefold() in paths:
                raise LocalSourceCollisionError(f"local source path collision: {relative}")
            paths[relative.casefold()] = path
            extension, stem = Path(entry.name).suffix.casefold(), Path(entry.name).stem.casefold()
            if extension in SUPPORTED_EXTENSIONS:
                capture_group, member_id = _capture_identity(entry.name)
                images.append((class_entry.name, path, capture_group, member_id))
            elif extension == ".xml":
                sidecars.append((f"{class_entry.name.casefold()}\0{stem}", relative))
            else:
                raise LocalSourceScanError(f"unsupported local source file: {relative}")
    image_stems = {f"{label.casefold()}\0{Path(path.name).stem.casefold()}" for label, path, _, _ in images}
    for stem, relative in sidecars:
        if stem not in image_stems:
            raise LocalSourceScanError(f"unpaired XML metadata: {relative}")

    records, ids, digests, members = [], set(), {}, set()
    for label, path, capture_group, member_id in sorted(images, key=lambda item: _relative(item[0], item[1].name)):
        relative = _relative(label, path.name)
        member_key = member_id.casefold()
        if member_key in members:
            raise LocalSourceCollisionError("duplicate local capture-group member identity")
        members.add(member_key)
        try:
            digest = digest_fn(path)
        except LocalSourceError:
            raise
        except Exception:
            raise LocalSourceScanError("local source digest failed") from None
        if type(digest) is not str or not _DIGEST_RE.fullmatch(digest):
            raise LocalSourceScanError("local source digest is invalid")
        digests.setdefault(digest, []).append((relative, label, labels[label]))
        identifier = record_id_fn(relative)
        if type(identifier) is not str or not _DIGEST_RE.fullmatch(identifier):
            raise LocalSourceScanError("local source record ID is invalid")
        if identifier in ids:
            raise LocalSourceCollisionError("local source record ID collision")
        ids.add(identifier)
        records.append(LocalSourceRecord(identifier, label, labels[label], relative, digest, capture_group, member_id))
    records.sort(key=lambda item: item.relative_path)
    conflicting_digests = {
        digest
        for digest, values in digests.items()
        if len({category for _, _, category in values}) > 1
    }
    exclusions = tuple(
        LocalSourceExclusion(
            record.record_id,
            record.relative_path,
            record.source_label,
            record.bcs_category,
            record.sha256,
            "cross_category_identical_digest",
        )
        for record in records
        if record.sha256 in conflicting_digests
    )
    records = [record for record in records if record.sha256 not in conflicting_digests]
    counts = [0] * NUM_CLASSES
    for record in records:
        counts[record.bcs_category - 1] += 1
    return LocalSourceScan(
        source_root,
        tuple(records),
        tuple(counts),
        tuple(i + 1 for i, count in enumerate(counts) if count),
        source_mapping.entries,
        exclusions,
    )


def materialize_local_record(
    scan: LocalSourceScan,
    record: LocalSourceRecord,
    *,
    max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
) -> LocalSourceMaterialized:
    """Read one bounded scanned file and verify its immutable scan digest."""
    if type(max_bytes) is not int or max_bytes <= 0:
        raise LocalSourceConfigurationError("maximum local source bytes must be positive")
    if record not in scan.records or record.record_id != _record_id(record.relative_path):
        raise LocalSourceMaterializationError("local source record does not belong to scan")
    pure = PurePosixPath(record.relative_path)
    if pure.is_absolute() or "\\" in record.relative_path or any(part in {"", ".", ".."} for part in pure.parts) or _relative(*pure.parts) != record.relative_path:
        raise LocalSourceMaterializationError("local source relative path is invalid")
    try:
        root_info = os.lstat(scan.root)
        if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
            raise LocalSourceMaterializationError("local source root changed")
        current = scan.root
        for part in pure.parts[:-1]:
            current /= part
            info = os.lstat(current)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise LocalSourceMaterializationError("local source path changed")
    except LocalSourceMaterializationError:
        raise
    except (OSError, RuntimeError):
        raise LocalSourceMaterializationError("local source path changed") from None
    path = scan.root.joinpath(*pure.parts)
    _inside(scan.root, path, record.relative_path, LocalSourceMaterializationError)
    before = _regular(path, record.relative_path, LocalSourceMaterializationError)
    try:
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise LocalSourceMaterializationError("local source file changed during access")
            payload, after = stream.read(max_bytes + 1), os.fstat(stream.fileno())
    except LocalSourceMaterializationError:
        raise
    except (OSError, ValueError):
        raise LocalSourceMaterializationError("local source file cannot be read") from None
    if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
        raise LocalSourceMaterializationError("local source file changed during access")
    if not payload:
        raise LocalSourceMaterializationError("local source file is empty")
    if len(payload) > max_bytes:
        raise LocalSourceMaterializationError("local source file exceeds maximum size")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != record.sha256:
        raise LocalSourceMaterializationError("local source file does not match scan")
    return LocalSourceMaterialized(record.record_id, record.relative_path, payload, digest, len(payload))


class LocalSourceMaterializer:
    """Resolve one scanned local record by its stable identity."""

    def __init__(self, scan: LocalSourceScan, *, max_bytes: int = DEFAULT_MAX_IMAGE_BYTES) -> None:
        if type(max_bytes) is not int or max_bytes <= 0:
            raise LocalSourceMaterializationError("maximum local source bytes must be positive")
        self._scan, self._max_bytes = scan, max_bytes

    def materialize(self, record_id: str) -> LocalSourceMaterialized:
        record = next((item for item in self._scan.records if item.record_id == record_id), None)
        if record is None:
            raise LocalSourceMaterializationError("local source record is not in scan")
        return materialize_local_record(self._scan, record, max_bytes=self._max_bytes)
