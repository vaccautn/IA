"""Lexical path checks for BCS filesystem entry points."""
from __future__ import annotations

import os
import stat
from collections.abc import Iterable
from pathlib import Path


class SafePathError(ValueError):
    """Raised when a path crosses an unsafe filesystem boundary."""


def safe_path(
    raw: str | os.PathLike[str],
    *,
    base: Path,
    approved_roots: Iterable[Path],
    allow_missing_final: bool = True,
    require_file: bool = False,
    require_dir: bool = False,
) -> Path:
    """Check lexical components before resolving and enforce approved roots."""
    value = Path(raw).expanduser()
    candidate = value if value.is_absolute() else Path(base) / value
    candidate = Path(os.path.normpath(str(candidate)))
    _reject_symlink_components(candidate, allow_missing_final=allow_missing_final)

    try:
        resolved = candidate.resolve(strict=not allow_missing_final)
    except (OSError, RuntimeError):
        raise SafePathError("path cannot be resolved safely") from None

    roots = []
    for root in approved_roots:
        approved = Path(root).expanduser()
        _reject_symlink_components(approved, allow_missing_final=True)
        try:
            roots.append(approved.resolve(strict=False))
        except (OSError, RuntimeError):
            raise SafePathError("approved filesystem root cannot be resolved") from None
    if not roots:
        raise SafePathError("approved filesystem roots are required")
    if roots and not any(_is_relative_to(resolved, root) for root in roots):
        raise SafePathError("path is outside the approved filesystem roots")
    if require_file and (not resolved.is_file() or resolved.is_symlink()):
        raise SafePathError("path must be a regular file")
    if require_dir and (not resolved.is_dir() or resolved.is_symlink()):
        raise SafePathError("path must be a real directory")
    return resolved


def _reject_symlink_components(candidate: Path, *, allow_missing_final: bool) -> None:
    current = Path(candidate.anchor)
    parts = candidate.parts[1:] if candidate.anchor else candidate.parts
    for index, part in enumerate(parts):
        if part in {"", "."}:
            continue
        current /= part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            if allow_missing_final:
                return
            raise SafePathError("path component does not exist") from None
        except OSError:
            raise SafePathError("path component cannot be inspected") from None
        if stat.S_ISLNK(info.st_mode):
            raise SafePathError("path must not traverse a symlink")
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise SafePathError("path has a non-directory ancestor")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
