from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from vacca_bcs.constants import CLASS_NAMES
from vacca_bcs.dataset_build_plan import create_build_plan
from vacca_bcs.dataset_snapshot import create_staged_snapshot


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    formats = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".bmp": "BMP", ".webp": "WEBP"}
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), color).save(path, format=formats[path.suffix.lower()])


def _make_source(root: Path, per_class: int = 6) -> Path:
    suffixes = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".PNG"]
    source = root / "bcs"
    for class_index, class_name in enumerate(CLASS_NAMES):
        for index in range(per_class):
            _write_image(
                source / class_name / f"img-{index}{suffixes[index % len(suffixes)]}",
                (class_index * 20, index * 20, 100),
            )
    return source


def test_manifest_is_deterministic_relative_and_hashes_installed_snapshot(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "out"
    plan = create_build_plan(source, output, max_per_class=4, seed=7, val_ratio=0.25)
    first = create_staged_snapshot(plan, tmp_path / "stage-a")
    second = create_staged_snapshot(plan, tmp_path / "stage-b")
    first_bytes = (first.staging_dir / "manifest.json").read_bytes()
    assert first_bytes == (second.staging_dir / "manifest.json").read_bytes()
    manifest = json.loads(first_bytes)
    assert manifest["manifest_schema_version"] == 1
    assert manifest["builder_inputs"] == {"max_per_class": 4, "seed": 7, "val_ratio": 0.25}
    assert manifest["class_values"] == list(CLASS_NAMES)
    assert manifest["class_mapping"] == {name: index for index, name in enumerate(CLASS_NAMES)}
    entries = manifest["selected_files"]
    assert [entry["destination"] for entry in entries] == sorted(
        (entry["destination"] for entry in entries), key=str.casefold
    )
    assert all("/" in entry["source"] and "/" in entry["destination"] for entry in entries)
    assert all(not Path(entry["source"]).is_absolute() for entry in entries)
    assert "timestamp" not in first_bytes.decode().lower()
    for entry in entries:
        staged = first.staging_dir / entry["destination"]
        assert entry["sha256"] == hashlib.sha256(staged.read_bytes()).hexdigest()


def test_staged_bytes_are_revalidated_after_source_mutation(tmp_path: Path, monkeypatch) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "out"
    plan = create_build_plan(source, output, max_per_class=3, seed=1, val_ratio=0.5)
    target = plan.planned_files[0].source
    real_copy2 = shutil.copy2
    mutated = False

    def mutate_before_copy(source_path, destination):
        nonlocal mutated
        if Path(source_path) == target and not mutated:
            target.write_bytes(b"corrupt after source preflight")
            mutated = True
        return real_copy2(source_path, destination)

    monkeypatch.setattr(shutil, "copy2", mutate_before_copy)
    with pytest.raises(ValueError, match="invalid or undecodable image"):
        create_staged_snapshot(plan, tmp_path / "stage")
    assert mutated
    assert not (tmp_path / "stage").exists()


def test_snapshot_copy_and_manifest_failures_clean_staging(tmp_path: Path, monkeypatch) -> None:
    source = _make_source(tmp_path)
    plan = create_build_plan(source, tmp_path / "out", max_per_class=3, val_ratio=0.5)

    monkeypatch.setattr(shutil, "copy2", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("copy failure")))
    with pytest.raises(OSError, match="copy failure"):
        create_staged_snapshot(plan, tmp_path / "copy-stage")
    assert not (tmp_path / "copy-stage").exists()

    monkeypatch.undo()
    import vacca_bcs.dataset_snapshot as snapshot_module
    monkeypatch.setattr(snapshot_module, "_write_manifest", lambda *args: (_ for _ in ()).throw(OSError("manifest failure")))
    with pytest.raises(OSError, match="manifest failure"):
        create_staged_snapshot(plan, tmp_path / "manifest-stage")
    assert not (tmp_path / "manifest-stage").exists()


def test_builder_import_boundary_excludes_training_modules() -> None:
    probe = """
import sys
import vacca_bcs.dataset_build_plan
import vacca_bcs.dataset_snapshot
from vacca_bcs.constants import CLASS_NAMES
assert CLASS_NAMES == ("1", "2", "3", "4", "5")
for name in ("torch", "torchvision", "vacca_bcs.model", "vacca_bcs.dataset"):
    assert name not in sys.modules, name
"""
    result = subprocess.run([".venv\\Scripts\\python", "-c", probe], cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
