"""Recovery state machine for publishing a generated BCS dataset."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .dataset_topology import (
    DatasetTopology,
    backup_path,
    failed_path,
    path_exists,
    safe_remove_tree,
    validate_derived_path,
)


class DatasetInstallError(RuntimeError):
    """The staged dataset could not be installed at the canonical path."""


class DatasetRecoveryRequiredError(DatasetInstallError):
    """A complete previous generation requires manual recovery before retry."""


class DatasetRollbackError(DatasetInstallError):
    """Installation and restoration both failed; the backup needs manual recovery."""


@dataclass(frozen=True)
class RecoveryState:
    topology: DatasetTopology
    staging_dir: Path
    backup_dir: Path
    live_moved: bool


def assert_retry_allowed(topology: DatasetTopology) -> None:
    """Reject active recovery before planning, staging, or backup deletion."""
    recovery = validate_derived_path(topology, backup_path(topology.output_dir))
    if path_exists(recovery) and not path_exists(topology.output_dir):
        raise DatasetRecoveryRequiredError(
            "manual dataset recovery required: canonical dataset path is absent "
            f"({topology.output_dir}) while the complete recovery backup exists at {recovery}"
        )


def prepare_swap(topology: DatasetTopology, staging_dir: Path) -> RecoveryState:
    """Remove stale recovery, then move the current live generation to backup."""
    assert_retry_allowed(topology)
    staging = validate_derived_path(topology, staging_dir)
    recovery = validate_derived_path(topology, backup_path(topology.output_dir))
    state = RecoveryState(topology, staging, recovery, False)
    if path_exists(recovery):
        try:
            safe_remove_tree(topology, recovery)
        except BaseException as stale_cleanup_exc:
            if isinstance(stale_cleanup_exc, Exception):
                raise DatasetInstallError(
                    f"could not remove stale dataset recovery backup {recovery}"
                ) from stale_cleanup_exc
            raise

    if path_exists(topology.output_dir):
        try:
            os.replace(topology.output_dir, recovery)
            state = RecoveryState(topology, staging, recovery, True)
        except BaseException as move_exc:
            moved = path_exists(recovery) and not path_exists(topology.output_dir)
            if moved:
                state = RecoveryState(topology, staging, recovery, True)
                rollback_failed_publication(state, move_exc)
                if isinstance(move_exc, Exception):
                    raise DatasetInstallError(
                        "moving the existing dataset was interrupted; the previous dataset was restored"
                    ) from move_exc
                raise
            if isinstance(move_exc, Exception):
                raise DatasetInstallError(
                    f"could not move the existing dataset to recovery path {recovery}"
                ) from move_exc
            raise
    return state


def rollback_failed_publication(state: RecoveryState, original: BaseException) -> None:
    """Quarantine a new canonical tree, restore backup, and clean quarantine."""
    quarantine: Path | None = None
    try:
        if path_exists(state.topology.output_dir):
            quarantine = validate_derived_path(state.topology, failed_path(state.topology.output_dir))
            if path_exists(quarantine):
                raise DatasetRollbackError(f"failed-install quarantine already exists: {quarantine}")
            os.replace(state.topology.output_dir, quarantine)
        if state.live_moved:
            os.replace(state.backup_dir, state.topology.output_dir)
        if quarantine is not None and path_exists(quarantine):
            safe_remove_tree(state.topology, quarantine)
    except BaseException as rollback_exc:
        detail = (
            f"the complete backup is retained at {state.backup_dir}"
            if state.live_moved
            else "there was no previous live dataset to recover"
        )
        raise DatasetRollbackError(
            "staged dataset installation failed and rollback failed; "
            f"{detail}, and the canonical dataset path is not guaranteed to be restored; "
            f"rollback error: {rollback_exc}"
        ) from original
