from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

import scripts.build_bcs_category as category_cli
from scripts.build_bcs_category import DEFAULT_OUTPUT, build_parser, main


def source(root: Path) -> Path:
    for category, label in enumerate(("3.25", "3.5", "3.75", "4.0", "4.25"), 1):
        folder = root / label
        folder.mkdir(parents=True)
        stream = io.BytesIO()
        Image.new("RGB", (2, 2), (category * 20, 30, 40)).save(stream, format="JPEG")
        (folder / f"GS_{category}_1.jpg").write_bytes(stream.getvalue())
    return root


def test_parser_has_only_local_category_controls():
    options = {action.dest for action in build_parser()._actions}
    assert {"local_root", "output", "seed", "max_image_bytes"} <= options
    assert "source" not in options
    assert "base_url" not in options


def test_local_cli_builds_safe_summary_and_exact_mapping(tmp_path):
    root = tmp_path
    rendered = io.StringIO()
    assert main(["--local-root", str(source(root / "data/source")), "--output", str(root / "data/snapshot")], root=root, stdout=rendered) == 0
    summary = json.loads(rendered.getvalue())
    assert summary["snapshot_schema"] == "bcs-category-snapshot-v1"
    assert summary["mapping"] == {"3.25": 1, "3.5": 2, "3.75": 3, "4.0": 4, "4.25": 5}
    assert summary["observed_classes"] == [1, 2, 3, 4, 5]


def test_cli_reports_typed_sanitized_failures(tmp_path, monkeypatch):
    error = io.StringIO()
    assert main(["--max-image-bytes", "not-an-integer"], stderr=error) == 1
    assert "invalid command line" in error.getvalue()
    assert "Traceback" not in error.getvalue()

    root = tmp_path
    source_root = source(root / "data/unexpected-source")
    monkeypatch.setattr(
        category_cli,
        "scan_local_source",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("private details")),
    )
    error = io.StringIO()
    assert main(["--local-root", str(source_root), "--output", str(root / "data/unexpected")], root=root, stderr=error) == 1
    assert error.getvalue().startswith("ERROR [unexpected]: RuntimeError; correlation_id=")
    assert "private details" not in error.getvalue()
    assert "Traceback" not in error.getvalue()

    monkeypatch.undo()
    root = source(root / "data/source")
    (root / "unexpected").mkdir()
    error = io.StringIO()
    assert main(["--local-root", str(root), "--output", str(tmp_path / "data/snapshot")], root=tmp_path, stderr=error) == 1
    assert error.getvalue().startswith("ERROR [local-source]:")
    assert "Traceback" not in error.getvalue()


def test_cli_summary_counts_quarantine_reasons(tmp_path):
    root = source(tmp_path / "data/source")
    duplicate = (root / "3.25" / "GS_1_1.jpg").read_bytes()
    unique_stream = io.BytesIO()
    Image.new("RGB", (3, 3), (200, 100, 20)).save(unique_stream, format="JPEG")
    unique_325 = unique_stream.getvalue()
    unique_stream = io.BytesIO()
    Image.new("RGB", (3, 3), (100, 200, 20)).save(unique_stream, format="JPEG")
    unique_35 = unique_stream.getvalue()
    (root / "3.25" / "GS_1_2.jpg").write_bytes(duplicate)
    (root / "3.5" / "GS_2_2.jpg").write_bytes(duplicate)
    (root / "3.25" / "GS_1_3.jpg").write_bytes(unique_325)
    (root / "3.5" / "GS_2_3.jpg").write_bytes(unique_35)
    rendered = io.StringIO()
    assert main(["--local-root", str(root), "--output", str(tmp_path / "data/snapshot")], root=tmp_path, stdout=rendered) == 0
    summary = json.loads(rendered.getvalue())
    assert summary["excluded"] == 3
    assert summary["exclusion_reason_counts"] == {"cross_category_identical_digest": 3}
    assert "manifest.json" in summary["exclusion_inspection"]


def test_default_output_is_new_category_root(tmp_path):
    assert DEFAULT_OUTPUT == "data/bcs-category-v1"


def test_builder_rejects_output_outside_repository_data_root(tmp_path):
    root = source(tmp_path / "data/source")
    error = io.StringIO()
    assert main(
        ["--local-root", str(root), "--output", str(tmp_path / "outside")],
        root=tmp_path,
        stderr=error,
    ) == 1
    assert "approved" in error.getvalue()


def test_builder_rejects_symlinked_output_and_ancestor(tmp_path):
    root = source(tmp_path / "data/source")
    outside = tmp_path / "outside"
    outside.mkdir()
    output = tmp_path / "data/output-link"
    try:
        output.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable")
    error = io.StringIO()
    assert main(["--local-root", str(root), "--output", str(output)], root=tmp_path, stderr=error) == 1
    assert "symlink" in error.getvalue()

    ancestor = tmp_path / "data/ancestor-link"
    ancestor.symlink_to(tmp_path, target_is_directory=True)
    error = io.StringIO()
    assert main(["--local-root", str(root), "--output", str(ancestor / "snapshot")], root=tmp_path, stderr=error) == 1
    assert "symlink" in error.getvalue()
