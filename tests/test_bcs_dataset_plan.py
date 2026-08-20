from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from PIL import Image

from vacca_bcs.constants import CLASS_NAMES, IMAGE_EXTENSIONS
from vacca_bcs.dataset_build_plan import create_build_plan


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    formats = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".bmp": "BMP", ".webp": "WEBP"}
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), color).save(path, format=formats[path.suffix.lower()])


def _make_source(root: Path, per_class: int = 6) -> Path:
    suffixes = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".PNG"]
    source = root / "bcs"
    for class_index, class_name in enumerate(CLASS_NAMES):
        for index in range(per_class):
            _write_image(source / class_name / f"img-{index}{suffixes[index % len(suffixes)]}", (class_index * 20, index * 20, 100))
    return source


def test_plan_preserves_val_ratio_zero_and_split_boundaries(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    plan = create_build_plan(source, tmp_path / "out", max_per_class=1, val_ratio=0.0)
    assert all(selection.train == 1 and selection.val == 0 for selection in plan.selections)
    with pytest.raises(ValueError, match="both train and validation"):
        create_build_plan(_make_source(tmp_path / "invalid"), tmp_path / "invalid-out", max_per_class=1, val_ratio=0.1)


def test_plan_supports_all_case_insensitive_extensions_and_all_candidate_validation(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    plan = create_build_plan(source, tmp_path / "out", max_per_class=6, val_ratio=0.5)
    assert IMAGE_EXTENSIONS == {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    assert len(plan.planned_files) == len(CLASS_NAMES) * 6

    limited = create_build_plan(source, tmp_path / "limited-out", max_per_class=1, seed=1, val_ratio=0.0)
    selected = {item.source for item in limited.planned_files}
    candidate = next(path for path in (source / CLASS_NAMES[0]).iterdir() if path not in selected)
    candidate.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="invalid or undecodable image"):
        create_build_plan(source, tmp_path / "limited-out", max_per_class=1, seed=1, val_ratio=0.0)


def test_destination_collision_is_rejected_before_snapshot(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "bcs"
    monkeypatch.setattr("vacca_bcs.dataset_build_plan.CLASS_NAMES", ["a", "A"])
    _write_image(source / "a" / "same.jpg", (1, 2, 3))
    with pytest.raises(ValueError, match="collision"):
        create_build_plan(source, tmp_path / "out", max_per_class=1, val_ratio=0.0)


@pytest.mark.parametrize("max_per_class", [0, -1])
def test_plan_rejects_non_positive_limits(tmp_path: Path, max_per_class: int) -> None:
    source = _make_source(tmp_path)
    with pytest.raises(ValueError, match="max-per-class"):
        create_build_plan(source, tmp_path / "out", max_per_class=max_per_class)


@pytest.mark.parametrize("val_ratio", [-0.01, 1.0])
def test_plan_rejects_ratio_outside_bounds(tmp_path: Path, val_ratio: float) -> None:
    source = _make_source(tmp_path)
    with pytest.raises(ValueError, match="val-ratio"):
        create_build_plan(source, tmp_path / "out", val_ratio=val_ratio)


def test_plan_rejects_absent_source_and_missing_or_empty_classes(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="dataset directory"):
        create_build_plan(tmp_path / "missing", tmp_path / "out")
    source = _make_source(tmp_path / "classes")
    (source / CLASS_NAMES[-1]).rename(source / "missing-class")
    with pytest.raises(FileNotFoundError, match="class directory"):
        create_build_plan(source, tmp_path / "out")
    source = _make_source(tmp_path / "empty")
    for path in (source / CLASS_NAMES[0]).iterdir():
        path.unlink()
    with pytest.raises(ValueError, match="no supported images"):
        create_build_plan(source, tmp_path / "out")


def test_fresh_plans_are_deterministically_equal(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    first = create_build_plan(source, tmp_path / "out", max_per_class=4, seed=19, val_ratio=0.25)
    second = create_build_plan(source, tmp_path / "out", max_per_class=4, seed=19, val_ratio=0.25)
    assert first.planned_files == second.planned_files
    assert first.selections == second.selections


def test_change_summary_tracks_added_updated_unchanged_and_stale(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "out"
    plan = create_build_plan(source, output, max_per_class=2, seed=3, val_ratio=0.0)
    for item in plan.planned_files:
        destination = plan.topology.output_dir / item.destination_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.source, destination)
    unchanged = create_build_plan(source, output, max_per_class=2, seed=3, val_ratio=0.0)
    assert unchanged.change_summary.totals.added == 0
    assert unchanged.change_summary.totals.unchanged == unchanged.change_summary.totals.selected

    selected = unchanged.planned_files[0]
    _write_image(source / selected.source_relative, (255, 0, 0))
    stale = output / "train" / CLASS_NAMES[0] / "nested" / "stale.txt"
    stale.parent.mkdir()
    stale.write_text("stale", encoding="utf-8")
    changed = create_build_plan(source, output, max_per_class=2, seed=3, val_ratio=0.0)
    assert changed.change_summary.totals.updated == 1
    assert changed.change_summary.totals.stale == 1
