from __future__ import annotations

import hashlib
import json
import shutil
import warnings
from pathlib import Path

import pytest
from PIL import Image

import vacca_bcs.dataset_transaction as transaction
import vacca_bcs.dataset_recovery as recovery
from vacca_bcs.constants import CLASS_NAMES
from vacca_bcs.dataset_topology import backup_path, failed_path
from vacca_bcs.dataset_transaction import (
    DatasetInstallError,
    DatasetRecoveryRequiredError,
    DatasetRollbackError,
    build_dataset,
)


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    formats = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".bmp": "BMP", ".webp": "WEBP"}
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), color).save(path, format=formats[path.suffix.lower()])


def _make_source(root: Path) -> Path:
    suffixes = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".PNG"]
    source = root / "bcs"
    for class_index, class_name in enumerate(CLASS_NAMES):
        for index, suffix in enumerate(suffixes):
            _write_image(source / class_name / f"img-{index}{suffix}", (class_index * 20, index * 20, 100))
    return source


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_changed_same_name_source_updates_live_copy_and_manifest_digest(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "out"
    build_dataset(source, output, max_per_class=3, seed=1, val_ratio=0.5)
    entry = json.loads((output / "manifest.json").read_bytes())["selected_files"][0]
    source_file = source / entry["source"]
    _write_image(source_file, (255, 0, 0))
    build_dataset(source, output, max_per_class=3, seed=1, val_ratio=0.5)
    destination = output / entry["destination"]
    assert destination.read_bytes() == source_file.read_bytes()
    manifest = json.loads((output / "manifest.json").read_bytes())
    updated = next(item for item in manifest["selected_files"] if item["destination"] == entry["destination"])
    assert updated["sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()


def test_stale_generated_artifacts_disappear_on_replacement(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "out"
    build_dataset(source, output, max_per_class=3, seed=1, val_ratio=0.5)
    stale = output / "train" / CLASS_NAMES[0] / "nested" / "stale.txt"
    stale.parent.mkdir()
    stale.write_text("stale", encoding="utf-8")
    build_dataset(source, output, max_per_class=3, seed=2, val_ratio=0.5)
    assert not stale.exists()


def test_successful_replacement_keeps_previous_backup_exact(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "out"
    build_dataset(source, output, max_per_class=3, seed=1, val_ratio=0.5)
    before = _tree_bytes(output)
    build_dataset(source, output, max_per_class=3, seed=2, val_ratio=0.5)
    assert _tree_bytes(backup_path(output)) == before
    assert json.loads((output / "manifest.json").read_bytes())["builder_inputs"]["seed"] == 2


def test_stale_backup_cleanup_failure_preserves_live_and_cleans_stage(tmp_path: Path, monkeypatch) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "out"
    build_dataset(source, output, max_per_class=3, seed=1, val_ratio=0.5)
    before = _tree_bytes(output)
    backup = backup_path(output)
    shutil.copytree(output, backup)
    backup_before = _tree_bytes(backup)
    real_remove = recovery.safe_remove_tree

    def fail_stale_cleanup(_topology, path: Path) -> None:
        if path == backup:
            raise OSError("stale backup cleanup failure")
        real_remove(recovery_topology, path)

    recovery_topology = transaction.validate_topology(source, output)
    monkeypatch.setattr(recovery, "safe_remove_tree", fail_stale_cleanup)
    with pytest.raises(DatasetInstallError, match="stale dataset recovery backup"):
        build_dataset(source, output, max_per_class=3, seed=2, val_ratio=0.5)
    assert _tree_bytes(output) == before
    assert _tree_bytes(backup) == backup_before
    assert list(tmp_path.glob(".out.staging-*")) == []


def test_active_recovery_precedes_planning_and_retains_only_backup(tmp_path: Path, monkeypatch) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "out"
    build_dataset(source, output, max_per_class=3, seed=1, val_ratio=0.5)
    backup = backup_path(output)
    shutil.copytree(output, backup)
    shutil.rmtree(output)
    before_source = _tree_bytes(source)
    before_backup = _tree_bytes(backup)

    def fail_plan(*args, **kwargs):
        raise KeyboardInterrupt("planning should not run")

    monkeypatch.setattr(transaction, "create_build_plan", fail_plan)
    with pytest.raises(DatasetRecoveryRequiredError):
        build_dataset(source, output, max_per_class=3, seed=2, val_ratio=0.5)
    assert not output.exists()
    assert _tree_bytes(source) == before_source and _tree_bytes(backup) == before_backup
    assert list(tmp_path.glob(".out.staging-*")) == []


def test_install_failure_without_prior_live_leaves_output_absent(tmp_path: Path, monkeypatch) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "out"
    before_source = _tree_bytes(source)
    real_replace = transaction.os.replace

    def fail_install(source_path, destination):
        if Path(source_path).name.startswith(".out.staging-") and Path(destination) == output:
            raise OSError("install failure")
        real_replace(source_path, destination)

    monkeypatch.setattr(transaction.os, "replace", fail_install)
    with pytest.raises(DatasetInstallError, match="installation failed"):
        build_dataset(source, output, max_per_class=3, seed=2, val_ratio=0.5)
    assert not output.exists() and _tree_bytes(source) == before_source
    assert list(tmp_path.glob(".out.staging-*")) == []


def test_install_and_rollback_failure_retains_exact_backup(tmp_path: Path, monkeypatch) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "out"
    build_dataset(source, output, max_per_class=3, seed=1, val_ratio=0.5)
    before = _tree_bytes(output)
    before_source = _tree_bytes(source)
    backup = backup_path(output)
    real_replace = transaction.os.replace

    def fail_both(source_path, destination):
        if Path(source_path).name.startswith(".out.staging-") and Path(destination) == output:
            raise OSError("install failure")
        if Path(source_path) == backup and Path(destination) == output:
            raise OSError("rollback failure")
        real_replace(source_path, destination)

    monkeypatch.setattr(transaction.os, "replace", fail_both)
    with pytest.raises(DatasetRollbackError) as failure:
        build_dataset(source, output, max_per_class=3, seed=2, val_ratio=0.5)
    assert str(backup) in str(failure.value)
    assert _tree_bytes(backup) == before and _tree_bytes(source) == before_source
    assert not output.exists()
    assert list(tmp_path.glob(".out.staging-*")) == []


def test_post_install_interrupt_recovers_prior_live_and_no_prior_live(tmp_path: Path, monkeypatch) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "out"
    build_dataset(source, output, max_per_class=3, seed=1, val_ratio=0.5)
    before = _tree_bytes(output)
    real_replace = transaction.os.replace

    def interrupt_after_move(source_path, destination):
        if Path(source_path).name.startswith(".out.staging-") and Path(destination) == output:
            real_replace(source_path, destination)
            raise KeyboardInterrupt("post-install interruption")
        real_replace(source_path, destination)

    monkeypatch.setattr(transaction.os, "replace", interrupt_after_move)
    with pytest.raises(KeyboardInterrupt, match="post-install interruption"):
        build_dataset(source, output, max_per_class=3, seed=2, val_ratio=0.5)
    assert _tree_bytes(output) == before
    assert not backup_path(output).exists() and not failed_path(output).exists()

    shutil.rmtree(output)
    with pytest.raises(KeyboardInterrupt, match="post-install interruption"):
        build_dataset(source, output, max_per_class=3, seed=2, val_ratio=0.5)
    assert not output.exists() and not failed_path(output).exists()
    assert list(tmp_path.glob(".out.staging-*")) == []


def test_warning_filters_do_not_change_successful_publication(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "out"
    build_dataset(source, output, max_per_class=3, seed=1, val_ratio=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build_dataset(source, output, max_per_class=3, seed=2, val_ratio=0.5)
    assert backup_path(output).is_dir()
