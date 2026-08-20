"""Deterministic, conservative planning for the ordinal BCS dataset builder."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .constants import (
    CHUNK_SIZE,
    CLASS_NAMES,
    DEFAULT_MAX_PER_CLASS,
    DEFAULT_SEED,
    DEFAULT_VAL_RATIO,
    IMAGE_EXTENSIONS,
    SPLITS,
)
from .dataset_change_summary import ChangeCounts, ChangeSummary
from .dataset_topology import DatasetTopology, validate_topology


@dataclass(frozen=True)
class PlannedFile:
    source: Path
    source_relative: str
    split: str
    class_name: str
    destination_relative: str


@dataclass(frozen=True)
class ClassSelection:
    class_name: str
    selected: int
    train: int
    val: int


@dataclass(frozen=True)
class DatasetBuildPlan:
    topology: DatasetTopology
    planned_files: tuple[PlannedFile, ...]
    selections: tuple[ClassSelection, ...]
    change_summary: ChangeSummary
    max_per_class: int
    seed: int
    val_ratio: float


def validate_image(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
    except Exception as exc:
        raise ValueError(f"invalid or undecodable image: {path}") from exc


def _files_match(source: Path, destination: Path) -> bool:
    if source.stat().st_size != destination.stat().st_size:
        return False
    with source.open("rb") as source_handle, destination.open("rb") as destination_handle:
        while True:
            source_chunk = source_handle.read(CHUNK_SIZE)
            destination_chunk = destination_handle.read(CHUNK_SIZE)
            if source_chunk != destination_chunk:
                return False
            if not source_chunk:
                return True


def _existing_changes(
    output_dir: Path,
    planned: tuple[PlannedFile, ...],
    selections: tuple[ClassSelection, ...],
) -> ChangeSummary:
    logical = {
        selection.class_name: {"added": 0, "updated": 0, "unchanged": 0, "stale": 0}
        for selection in selections
    }
    desired_by_folder = {
        output_dir / split / class_name: set()
        for split in SPLITS
        for class_name in CLASS_NAMES
    }
    for item in planned:
        destination = output_dir / item.destination_relative
        desired_by_folder[destination.parent].add(destination.name)
        if not destination.is_file():
            logical[item.class_name]["added"] += 1
        elif _files_match(item.source, destination):
            logical[item.class_name]["unchanged"] += 1
        else:
            logical[item.class_name]["updated"] += 1
    for folder, desired_names in desired_by_folder.items():
        if not folder.is_dir():
            continue
        for existing in folder.rglob("*"):
            if existing.is_file() and existing.relative_to(folder).as_posix() not in desired_names:
                logical[folder.name]["stale"] += 1
    return ChangeSummary(
        tuple(
            (
                selection.class_name,
                ChangeCounts(
                    selected=selection.selected,
                    train=selection.train,
                    val=selection.val,
                    staged=0,
                    **logical[selection.class_name],
                ),
            )
            for selection in selections
        )
    )


def create_build_plan(
    bcs_dir: Path,
    out_dir: Path,
    *,
    max_per_class: int = DEFAULT_MAX_PER_CLASS,
    seed: int = DEFAULT_SEED,
    val_ratio: float = DEFAULT_VAL_RATIO,
) -> DatasetBuildPlan:
    """Validate all candidates and produce a deterministic, immutable plan."""
    topology = validate_topology(bcs_dir, out_dir)
    if max_per_class <= 0:
        raise ValueError("--max-per-class must be positive")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0, 1)")
    if not topology.source_dir.is_dir():
        raise FileNotFoundError(f"BCS dataset directory not found: {topology.source_dir}")

    rng = random.Random(seed)
    planned: list[PlannedFile] = []
    selections: list[ClassSelection] = []
    destination_sources: dict[str, Path] = {}
    for class_name in CLASS_NAMES:
        source_dir = topology.source_dir / class_name
        if not source_dir.is_dir():
            raise FileNotFoundError(f"BCS class directory not found: {source_dir}")
        images = sorted(
            path
            for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise ValueError(f"BCS class directory contains no supported images: {source_dir}")
        for image in images:
            validate_image(image)

        rng.shuffle(images)
        selected = images[:max_per_class]
        if val_ratio == 0.0:
            n_val = 0
        else:
            n_val = max(1, int(len(selected) * val_ratio))
            if len(selected) < 2 or n_val >= len(selected):
                raise ValueError(
                    f"class {class_name} cannot provide both train and validation images "
                    f"with {len(selected)} selected images and --val-ratio {val_ratio}"
                )
        split_paths = {"train": selected[n_val:], "val": selected[:n_val]}
        selections.append(
            ClassSelection(class_name, len(selected), len(split_paths["train"]), len(split_paths["val"]))
        )
        for split, paths in split_paths.items():
            for source in paths:
                destination_relative = (Path(split) / class_name / source.name).as_posix()
                collision_key = destination_relative.casefold()
                if collision_key in destination_sources:
                    raise ValueError(
                        "destination filename collision: "
                        f"{destination_sources[collision_key]} and {source} map to {destination_relative}"
                    )
                destination_sources[collision_key] = source
                planned.append(
                    PlannedFile(
                        source=source,
                        source_relative=source.relative_to(topology.source_dir).as_posix(),
                        split=split,
                        class_name=class_name,
                        destination_relative=destination_relative,
                    )
                )
    planned_tuple = tuple(planned)
    selections_tuple = tuple(selections)
    return DatasetBuildPlan(
        topology=topology,
        planned_files=planned_tuple,
        selections=selections_tuple,
        change_summary=_existing_changes(topology.output_dir, planned_tuple, selections_tuple),
        max_per_class=max_per_class,
        seed=seed,
        val_ratio=val_ratio,
    )
