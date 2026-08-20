"""Orchestrate planning, snapshot creation, and live BCS publication."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .constants import DEFAULT_MAX_PER_CLASS, DEFAULT_SEED, DEFAULT_VAL_RATIO
from .dataset_build_plan import create_build_plan
from .dataset_recovery import (
    DatasetInstallError,
    DatasetRecoveryRequiredError,
    DatasetRollbackError,
    assert_retry_allowed,
    prepare_swap,
    rollback_failed_publication,
)
from .dataset_snapshot import DatasetSnapshot, create_staged_snapshot
from .dataset_topology import path_exists, safe_remove_tree, validate_topology


@dataclass(frozen=True)
class PublicationResult:
    output_dir: Path
    backup_dir: Path | None
    previous_generation_retained: bool


def _publish_snapshot(snapshot: DatasetSnapshot) -> PublicationResult:
    state = prepare_swap(snapshot.plan.topology, snapshot.staging_dir)
    try:
        os.replace(snapshot.staging_dir, snapshot.plan.topology.output_dir)
    except BaseException as install_exc:
        if not state.live_moved and not path_exists(snapshot.plan.topology.output_dir):
            if isinstance(install_exc, Exception):
                raise DatasetInstallError(
                    f"staged dataset installation failed: {install_exc}"
                ) from install_exc
            raise
        rollback_failed_publication(state, install_exc)
        if isinstance(install_exc, Exception):
            raise DatasetInstallError(
                "staged dataset installation failed; the previous dataset was restored"
            ) from install_exc
        raise
    return PublicationResult(
        output_dir=snapshot.plan.topology.output_dir,
        backup_dir=state.backup_dir if state.live_moved else None,
        previous_generation_retained=state.live_moved,
    )


def build_dataset(
    bcs_dir: Path,
    out_dir: Path,
    *,
    max_per_class: int = DEFAULT_MAX_PER_CLASS,
    seed: int = DEFAULT_SEED,
    val_ratio: float = DEFAULT_VAL_RATIO,
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """Plan, snapshot, and publish the ordinal BCS dataset."""
    topology = validate_topology(Path(bcs_dir), Path(out_dir))
    assert_retry_allowed(topology)
    plan = create_build_plan(
        topology.source_dir,
        topology.output_dir,
        max_per_class=max_per_class,
        seed=seed,
        val_ratio=val_ratio,
    )
    topology.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir: Path | None = Path(
        tempfile.mkdtemp(
            prefix=f".{topology.output_dir.name}.staging-",
            dir=str(topology.output_dir.parent),
        )
    )
    try:
        snapshot: DatasetSnapshot = create_staged_snapshot(plan, staging_dir)
        _publish_snapshot(snapshot)
        staging_dir = None
    finally:
        if staging_dir is not None and path_exists(staging_dir):
            safe_remove_tree(topology, staging_dir)

    per_class = snapshot.summary.as_per_class_dict()
    totals = snapshot.summary.as_totals_dict()
    return per_class, totals
