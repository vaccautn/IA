from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from vacca_bcs.constants import CLASS_NAMES
from vacca_bcs.dataset_topology import backup_path, failed_path, validate_derived_path, validate_topology


REPO_ROOT = Path(__file__).resolve().parents[1]


def _source(root: Path) -> Path:
    source = root / "bcs"
    for class_name in CLASS_NAMES:
        class_dir = source / class_name
        class_dir.mkdir(parents=True)
        Image.new("RGB", (2, 2), (1, 2, 3)).save(class_dir / "image.jpg")
    return source


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_topology_rejects_overlap_and_unsafe_output_file(tmp_path: Path) -> None:
    source = _source(tmp_path)
    for output in (source, source / "generated", source.parent, REPO_ROOT):
        with pytest.raises(ValueError):
            validate_topology(source, output)
    output_file = tmp_path / "output-file"
    output_file.write_bytes(b"keep")
    with pytest.raises(ValueError, match="not a directory"):
        validate_topology(source, output_file)
    assert output_file.read_bytes() == b"keep"


def test_source_junction_is_rejected_without_touching_target(tmp_path: Path) -> None:
    source = _source(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    junction = source / CLASS_NAMES[0]
    shutil.rmtree(junction)
    result = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("directory junctions are unavailable")
    with pytest.raises(ValueError, match="symlink or reparse point"):
        validate_topology(source, tmp_path / "out")
    assert target.is_dir()


def test_derived_backup_staging_and_quarantine_aliases_are_rejected(tmp_path: Path) -> None:
    for role in ("backup", "staging", "quarantine"):
        scenario = tmp_path / role
        output = scenario / "out"
        output.mkdir(parents=True)
        (output / "live.bin").write_bytes(b"live")
        candidate = {
            "backup": backup_path(output),
            "staging": scenario / ".out.staging-probe",
            "quarantine": failed_path(output),
        }[role]
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
