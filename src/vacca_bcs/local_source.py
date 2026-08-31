"""Safe discovery and one-record materialization for the local BCS source."""
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
LOCAL_SOURCE_SCHEMA = "bcs-local-folder-v1"
DEFAULT_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
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
            valid = all(type(label) is str and label and type(score) is int for label, score in entries)
        except (TypeError, ValueError):
            entries, valid = (), False
        if not valid or len({label for label, _ in entries}) != len(entries):
            raise LocalSourceConfigurationError("local source mapping is malformed")
        object.__setattr__(self, "entries", tuple(sorted(entries)))
    @property
    def by_label(self) -> dict[str, int]: return dict(self.entries)
LOCAL_BCS_MAPPING = LocalSourceMapping(
    (("3.25", 3), ("3.5", 3), ("3.75", 4), ("4.0", 4), ("4.25", 4))
)
@dataclass(frozen=True, slots=True)
class LocalSourceRecord:
    record_id: str
    source_label: str
    bcs_score: int
    relative_path: str
    sha256: str
@dataclass(frozen=True, slots=True)
class LocalSourceScan:
    root: Path
    records: tuple[LocalSourceRecord, ...]
    counts: tuple[int, int, int, int, int]
    observed_classes: tuple[int, ...]
    mapping_lineage: tuple[tuple[str, int], ...]
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
def _root_path(root: str | Path) -> Path:
    candidate = Path(root).expanduser()
    try:
        info, resolved = os.lstat(candidate), candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        raise LocalSourceScanError("local source root is unavailable") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise LocalSourceScanError("local source root must be a real directory")
    return resolved
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
        raise LocalSourceConfigurationError("local source mapping is not the exact BCS mapping")
    return result
def scan_local_source(root: str | Path, mapping: LocalSourceMapping | Mapping[str, int] = LOCAL_BCS_MAPPING, *, digest_fn: Callable[[Path], str] = _sha256_file, record_id_fn: Callable[[str], str] = _record_id) -> LocalSourceScan:
    """Scan direct mapped folders; XML is accepted only as paired metadata."""
    source_root, source_mapping = _root_path(root), _mapping(mapping)
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
            raise LocalSourceScanError(
                f"local source folder cannot be read: {class_relative}"
            ) from None
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
                images.append((class_entry.name, path))
            elif extension == ".xml":
                sidecars.append((f"{class_entry.name.casefold()}\0{stem}", relative))
            else:
                raise LocalSourceScanError(f"unsupported local source file: {relative}")
    image_stems = {f"{label.casefold()}\0{Path(path.name).stem.casefold()}" for label, path in images}
    for stem, relative in sidecars:
        if stem not in image_stems:
            raise LocalSourceScanError(f"unpaired XML metadata: {relative}")
    records, ids, digests = [], set(), {}
    for label, path in sorted(images, key=lambda item: _relative(item[0], item[1].name)):
        relative = _relative(label, path.name)
        try:
            digest = digest_fn(path)
        except LocalSourceError:
            raise
        except Exception:
            raise LocalSourceScanError("local source digest failed") from None
        if type(digest) is not str or not _DIGEST_RE.fullmatch(digest):
            raise LocalSourceScanError("local source digest is invalid")
        if digest in digests:
            raise LocalSourceCollisionError(f"duplicate local source content: {relative}")
        digests[digest] = relative
        identifier = record_id_fn(relative)
        if type(identifier) is not str or not _DIGEST_RE.fullmatch(identifier):
            raise LocalSourceScanError("local source record ID is invalid")
        if identifier in ids:
            raise LocalSourceCollisionError(
                f"local source record ID collision: {relative}"
            )
        ids.add(identifier)
        records.append(LocalSourceRecord(identifier, label, labels[label], relative, digest))
    records.sort(key=lambda item: item.relative_path)
    counts = [0] * 5
    for record in records:
        counts[record.bcs_score - 1] += 1
    return LocalSourceScan(source_root, tuple(records), tuple(counts), tuple(i + 1 for i, count in enumerate(counts) if count), source_mapping.entries)
def materialize_local_record(scan: LocalSourceScan, record: LocalSourceRecord, *, max_bytes: int = DEFAULT_MAX_IMAGE_BYTES) -> LocalSourceMaterialized:
    """Read one bounded scanned file without decoding or backend identity."""
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
