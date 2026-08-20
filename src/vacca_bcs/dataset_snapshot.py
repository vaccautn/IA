"""Create complete, validated staged snapshots for the ordinal BCS dataset."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .constants import CHUNK_SIZE, CLASS_NAMES, MANIFEST_FILENAME, MANIFEST_SCHEMA_VERSION, SPLITS
from .dataset_build_plan import DatasetBuildPlan, validate_image
from .dataset_change_summary import ChangeSummary
from .dataset_topology import safe_remove_tree, validate_derived_path


@dataclass(frozen=True)
class DatasetManifest:
    payload: dict[str, object]


@dataclass(frozen=True)
class DatasetSnapshot:
    plan: DatasetBuildPlan
    staging_dir: Path
    summary: ChangeSummary
    manifest: DatasetManifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def build_manifest(
    plan: DatasetBuildPlan,
    summary: ChangeSummary,
    staged_digests: dict[str, str],
) -> DatasetManifest:
    payload: dict[str, object] = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "builder_inputs": {
            "max_per_class": plan.max_per_class,
            "seed": plan.seed,
            "val_ratio": plan.val_ratio,
        },
        "class_values": list(CLASS_NAMES),
        "class_mapping": {class_name: index for index, class_name in enumerate(CLASS_NAMES)},
        "selected_files": [
            {
                "source": item.source_relative,
                "sha256": staged_digests[item.destination_relative],
                "split": item.split,
                "destination": item.destination_relative,
            }
            for item in sorted(plan.planned_files, key=lambda entry: entry.destination_relative.casefold())
        ],
        "counts": {
            "train": {class_name: summary.counts_for(class_name).train for class_name in CLASS_NAMES},
            "val": {class_name: summary.counts_for(class_name).val for class_name in CLASS_NAMES},
        },
    }
    return DatasetManifest(payload)


def create_staged_snapshot(plan: DatasetBuildPlan, staging_dir: Path) -> DatasetSnapshot:
    """Copy, validate, hash, and manifest a complete dataset without publishing it."""
    staging_dir = Path(staging_dir)
    validate_derived_path(plan.topology, staging_dir)
    if staging_dir.exists() and any(staging_dir.iterdir()):
        raise ValueError(f"staging directory must be empty: {staging_dir}")
    staging_dir.mkdir(parents=True, exist_ok=True)
    try:
        for split in SPLITS:
            for class_name in CLASS_NAMES:
                (staging_dir / split / class_name).mkdir(parents=True, exist_ok=True)

        staged_by_class = {selection.class_name: 0 for selection in plan.selections}
        staged_digests: dict[str, str] = {}
        for item in plan.planned_files:
            destination = staging_dir / item.destination_relative
            shutil.copy2(item.source, destination)
            staged_by_class[item.class_name] += 1
            validate_image(destination)
            staged_digests[item.destination_relative] = _sha256(destination)

        manifest = build_manifest(
            plan,
            plan.change_summary,
            staged_digests,
        )
        _write_manifest(staging_dir / MANIFEST_FILENAME, manifest.payload)
        return DatasetSnapshot(
            plan=plan,
            staging_dir=staging_dir,
            summary=plan.change_summary.with_staged(staged_by_class),
            manifest=manifest,
        )
    except BaseException:
        if staging_dir.exists():
            safe_remove_tree(plan.topology, staging_dir)
        raise
