"""Source/output topology and conservative filesystem safety validation."""
from __future__ import annotations

import stat
import shutil
from dataclasses import dataclass
from pathlib import Path

from .constants import BACKUP_SUFFIX


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DatasetTopology:
    source_dir: Path
    output_dir: Path


def canonical_path(path: Path) -> Path:
    return path.absolute().resolve(strict=False)


def path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink() or _is_reparse_point(path)


def backup_path(output_dir: Path) -> Path:
    return output_dir.with_name(output_dir.name + BACKUP_SUFFIX)


def failed_path(output_dir: Path) -> Path:
    return output_dir.with_name(f".{output_dir.name}.failed-install")


def _same_or_nested(path: Path, ancestor: Path) -> bool:
    return path == ancestor or ancestor in path.parents


def _paths_overlap(first: Path, second: Path) -> bool:
    return _same_or_nested(first, second) or _same_or_nested(second, first)


def _is_reparse_point(path: Path) -> bool:
    try:
        if path.is_junction():
            return True
    except (AttributeError, OSError):
        pass
    try:
        attributes = path.lstat().st_file_attributes
    except (AttributeError, OSError):
        return False
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _unsafe_path_component(path: Path) -> Path | None:
    current = Path(path.anchor) if path.anchor else Path()
    parts = path.parts[1:] if path.anchor else path.parts
    for part in parts:
        current /= part
        if current.is_symlink() or _is_reparse_point(current):
            return current
    return None


def _unsafe_tree_entry(root: Path) -> Path | None:
    if not root.is_dir() or root.is_symlink() or _is_reparse_point(root):
        return root
    pending = [root]
    while pending:
        directory = pending.pop()
        for candidate in directory.iterdir():
            if candidate.is_symlink() or _is_reparse_point(candidate):
                return candidate
            if candidate.is_dir():
                pending.append(candidate)
    return None


def validate_derived_path(topology: DatasetTopology, candidate: Path) -> Path:
    """Validate a concrete staging, backup, or quarantine path before touching it."""
    raw = Path(candidate).absolute()
    unsafe = _unsafe_path_component(raw)
    if unsafe is not None:
        raise ValueError(f"derived dataset path contains a symlink or reparse point: {unsafe}")
    resolved = canonical_path(raw)
    if _paths_overlap(resolved, topology.source_dir):
        raise ValueError(f"derived dataset path overlaps the BCS source: {resolved}")
    if _paths_overlap(resolved, topology.output_dir):
        raise ValueError(f"derived dataset path overlaps the live dataset: {resolved}")
    repository = canonical_path(REPOSITORY_ROOT)
    if resolved == repository or _same_or_nested(repository, resolved):
        raise ValueError(f"derived dataset path overlaps the repository: {resolved}")
    if resolved.exists() and _unsafe_tree_entry(resolved) is not None:
        raise ValueError(f"derived dataset path contains an unsafe tree: {resolved}")
    return resolved


def safe_remove_tree(topology: DatasetTopology, candidate: Path) -> None:
    """Remove only an already-validated non-live derived path."""
    resolved = validate_derived_path(topology, candidate)
    if resolved.is_dir() and not resolved.is_symlink():
        shutil.rmtree(resolved)
    elif path_exists(resolved):
        resolved.unlink()


def validate_topology(bcs_dir: Path, out_dir: Path) -> DatasetTopology:
    """Resolve roots and reject unsafe or overlapping source/output topologies."""
    raw_source = Path(bcs_dir).absolute()
    raw_output = Path(out_dir).absolute()
    unsafe_source = _unsafe_path_component(raw_source)
    if unsafe_source is not None:
        raise ValueError(f"BCS source path contains a symlink or reparse point: {unsafe_source}")
    unsafe_output = _unsafe_path_component(raw_output)
    if unsafe_output is not None:
        raise ValueError(
            f"generated dataset path contains a symlink or reparse point: {unsafe_output}"
        )

    source = canonical_path(Path(bcs_dir))
    output = canonical_path(Path(out_dir))
    repository = canonical_path(REPOSITORY_ROOT)
    if source == repository:
        raise ValueError("BCS source directory must not be the repository root")
    if output == repository or _same_or_nested(repository, output):
        raise ValueError(
            f"generated dataset path must not be the repository or its ancestor: {output}"
        )
    if _paths_overlap(source, output):
        raise ValueError(f"BCS source and generated dataset paths overlap: {source} and {output}")
    if output.exists() and not output.is_dir():
        raise ValueError(f"generated dataset path is not a directory: {output}")
    if output.is_symlink() or _is_reparse_point(output):
        raise ValueError(f"generated dataset path is unsafe: {output}")
    unsafe_existing = _unsafe_tree_entry(output) if output.exists() else None
    if unsafe_existing is not None:
        raise ValueError(
            f"existing generated dataset contains a symlink or reparse point: {unsafe_existing}"
        )
    unsafe_source = _unsafe_tree_entry(source) if source.is_dir() else None
    if unsafe_source is not None:
        raise ValueError(f"BCS source tree contains a symlink or reparse point: {unsafe_source}")
    return DatasetTopology(source, output)
