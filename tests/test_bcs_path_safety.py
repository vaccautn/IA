from __future__ import annotations

from pathlib import Path

import pytest

from vacca_bcs.path_safety import SafePathError, safe_path


def test_safe_path_canonicalizes_only_after_boundary_checks(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "nested" / "output"
    assert safe_path(target, base=root, approved_roots=(root,)) == target.resolve()


def test_safe_path_rejects_symlinked_ancestors_and_dangling_final(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    ancestor = root / "ancestor"
    dangling = root / "dangling"
    try:
        ancestor.symlink_to(outside, target_is_directory=True)
        dangling.symlink_to(outside / "missing", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable")
    with pytest.raises(SafePathError, match="symlink"):
        safe_path(ancestor / "output", base=root, approved_roots=(root,))
    with pytest.raises(SafePathError, match="symlink"):
        safe_path(dangling, base=root, approved_roots=(root,))


def test_safe_path_rejects_resolved_paths_outside_approved_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(SafePathError, match="approved"):
        safe_path(outside / "output", base=root, approved_roots=(root,))
