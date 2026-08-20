from __future__ import annotations

from pathlib import Path

import pytest

import vacca_bcs.dataset_recovery as recovery
from vacca_bcs.dataset_recovery import (
    DatasetInstallError,
    DatasetRecoveryRequiredError,
    DatasetRollbackError,
    RecoveryState,
    assert_retry_allowed,
    prepare_swap,
    rollback_failed_publication,
)
from vacca_bcs.dataset_topology import backup_path, failed_path, validate_derived_path, validate_topology


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    (source / "source.bin").write_bytes(b"source")
    output = tmp_path / "out"
    return source, output


def test_active_recovery_refuses_retry_and_preserves_backup(tmp_path: Path) -> None:
    source, output = _roots(tmp_path)
    backup = backup_path(output)
    backup.mkdir()
    (backup / "generation.bin").write_bytes(b"generation")
    before = _tree_bytes(backup)
    topology = validate_topology(source, output)
    with pytest.raises(DatasetRecoveryRequiredError) as failure:
        assert_retry_allowed(topology)
    assert str(output) in str(failure.value) and str(backup) in str(failure.value)
    assert _tree_bytes(backup) == before and not output.exists()


def test_source_aliases_are_rejected_before_derived_cleanup(tmp_path: Path) -> None:
    for role in ("backup", "staging"):
        scenario = tmp_path / role
        output = scenario / "out"
        output.mkdir(parents=True)
        (output / "live.bin").write_bytes(b"live")
        candidate = backup_path(output) if role == "backup" else scenario / ".out.staging-probe"
        source = candidate
        source.mkdir()
        (source / "source.bin").write_bytes(b"source")
        topology = validate_topology(source, output)
        source_before = _tree_bytes(source)
        live_before = _tree_bytes(output)
        with pytest.raises(ValueError, match="overlaps the BCS source"):
            validate_derived_path(topology, candidate)
        assert _tree_bytes(source) == source_before
        assert _tree_bytes(output) == live_before


def test_quarantine_alias_keeps_source_and_backup_exact(tmp_path: Path) -> None:
    source = tmp_path / ".out.failed-install"
    source.mkdir()
    (source / "source.bin").write_bytes(b"source")
    output = tmp_path / "out"
    output.mkdir()
    (output / "candidate.bin").write_bytes(b"candidate")
    backup = backup_path(output)
    backup.mkdir()
    (backup / "generation.bin").write_bytes(b"generation")
    topology = validate_topology(source, output)
    state = RecoveryState(topology, tmp_path / ".out.staging-test", backup, True)
    source_before = _tree_bytes(source)
    backup_before = _tree_bytes(backup)
    output_before = _tree_bytes(output)
    with pytest.raises(DatasetRollbackError, match="rollback failed"):
        rollback_failed_publication(state, KeyboardInterrupt("interrupted"))
    assert _tree_bytes(source) == source_before
    assert _tree_bytes(backup) == backup_before
    assert _tree_bytes(output) == output_before


def test_prepare_swap_removes_stale_backup_and_retains_live_generation(tmp_path: Path) -> None:
    source, output = _roots(tmp_path)
    output.mkdir()
    (output / "generation.bin").write_bytes(b"live")
    backup = backup_path(output)
    backup.mkdir()
    (backup / "stale.bin").write_bytes(b"stale")
    staging = tmp_path / ".out.staging-test"
    staging.mkdir()
    topology = validate_topology(source, output)
    state = prepare_swap(topology, staging)
    assert state.live_moved and not output.exists()
    assert _tree_bytes(backup) == {"generation.bin": b"live"}


def test_stale_backup_cleanup_failure_preserves_live(tmp_path: Path, monkeypatch) -> None:
    source, output = _roots(tmp_path)
    output.mkdir()
    (output / "generation.bin").write_bytes(b"live")
    backup = backup_path(output)
    backup.mkdir()
    (backup / "stale.bin").write_bytes(b"stale")
    topology = validate_topology(source, output)
    staging = tmp_path / ".out.staging-test"
    staging.mkdir()
    real_remove = recovery.safe_remove_tree

    def fail_remove(_topology, candidate):
        if candidate == backup:
            raise OSError("stale cleanup failure")
        real_remove(_topology, candidate)

    monkeypatch.setattr(recovery, "safe_remove_tree", fail_remove)
    with pytest.raises(DatasetInstallError, match="stale dataset recovery backup"):
        prepare_swap(topology, staging)
    assert _tree_bytes(output) == {"generation.bin": b"live"}
    assert _tree_bytes(backup) == {"stale.bin": b"stale"}


def test_backup_move_failure_preserves_live(tmp_path: Path, monkeypatch) -> None:
    source, output = _roots(tmp_path)
    output.mkdir()
    (output / "generation.bin").write_bytes(b"live")
    backup = backup_path(output)
    topology = validate_topology(source, output)
    staging = tmp_path / ".out.staging-test"
    staging.mkdir()
    real_replace = recovery.os.replace

    def fail_move(source_path, destination):
        if Path(source_path) == output and Path(destination) == backup:
            raise OSError("backup move failure")
        real_replace(source_path, destination)

    monkeypatch.setattr(recovery.os, "replace", fail_move)
    with pytest.raises(DatasetInstallError, match="recovery path"):
        prepare_swap(topology, staging)
    assert _tree_bytes(output) == {"generation.bin": b"live"}
    assert not backup.exists()


def test_interruption_after_live_backup_restores_and_propagates(tmp_path: Path, monkeypatch) -> None:
    source, output = _roots(tmp_path)
    output.mkdir()
    (output / "generation.bin").write_bytes(b"live")
    before = _tree_bytes(output)
    backup = backup_path(output)
    topology = validate_topology(source, output)
    staging = tmp_path / ".out.staging-test"
    staging.mkdir()
    real_replace = recovery.os.replace

    def interrupt_after_backup(source_path, destination):
        if Path(source_path) == output and Path(destination) == backup:
            real_replace(source_path, destination)
            raise KeyboardInterrupt("after backup")
        real_replace(source_path, destination)

    monkeypatch.setattr(recovery.os, "replace", interrupt_after_backup)
    with pytest.raises(KeyboardInterrupt, match="after backup"):
        prepare_swap(topology, staging)
    assert _tree_bytes(output) == before
    assert not backup.exists()
