import csv
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vacca_video.passage_bcs_export import export_passage_labels  # noqa: E402


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "tracking"
    root.mkdir()
    intervals = []
    for sequence, track in enumerate((9, 2, 7), 1):
        prefix = f"passages/{sequence}"
        directory = root / prefix
        directory.mkdir(parents=True)
        for name in ("best-original.jpg", "best-crop.png", "best-mask.png", "best-masked.png"):
            (directory / name).write_bytes(f"{sequence}-{name}".encode())
        intervals.append(dict(sequence=sequence, track_id=track, status="passage_evidence_candidate",
            start_frame=(sequence - 1) * 10, end_frame=sequence * 10,
            start_timestamp_ms=(sequence - 1) * 100., end_timestamp_ms=sequence * 100.,
            end_exclusive=sequence < 3, transition_method="initial_confidence", warnings=[],
            candidate=dict(status="available", track_id=track, frame_index=sequence * 10 - 1,
                timestamp_ms=sequence * 100. - 10, bbox=[1, 2, 8, 9],
                original=f"{prefix}/best-original.jpg", best_crop=f"{prefix}/best-crop.png",
                bcs_eligibility="not_assessed", bcs_status="not_evaluated",
                segmentation=dict(status="not_requested"))))
    summary = dict(passage_evidence=dict(status="provisional_evidence_not_unique_animal_identity",
        policy=dict(calibration="uncalibrated_confidence_proxy_not_visibility_or_bcs"),
        intervals=intervals, unresolved_tracks=[], warnings=[]), tracks=[])
    video = tmp_path / "source.mp4"
    video.write_bytes(b"synthetic source, not decoded")
    return root, summary, video, tmp_path / "pack"


def export(run, source=True):
    root, summary, video, output = run
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return export_passage_labels(root, output, source_video=video if source else None)


def rows(output):
    with (output / "labels.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_rows_assets_and_provenance(run):
    root, summary, video, output = run
    items = summary["passage_evidence"]["intervals"]
    items[0]["candidate"]["segmentation"] = dict(status="ok",
        best_mask="passages/1/best-mask.png", best_masked="passages/1/best-masked.png")
    items[1]["candidate"] = dict(status="not_available", reason="no_candidate_captured",
        bcs_eligibility="not_assessed", bcs_status="not_evaluated")
    result = export(run)
    sheet = rows(output)
    assert [row["track_id"] for row in sheet] == ["9", "2", "7"]
    assert len({row["group_id"] for row in sheet}) == 1
    assert result["source_video_sha256"] == hashlib.sha256(video.read_bytes()).hexdigest()
    assert result["run_metadata_sha256"] == hashlib.sha256((root / "summary.json").read_bytes()).hexdigest()
    assert all(row[key] == "" for row in sheet for key in ("bcs_category", "rump_visible", "notes"))
    assert sheet[1]["candidate_reason"] == "no_candidate_captured"
    assert all(sheet[1][key] == "" for key in ("original", "crop", "mask", "masked"))
    for number, row in enumerate(sheet, 1):
        for key, name in (("original", "best-original.jpg"), ("crop", "best-crop.png"),
                          ("mask", "best-mask.png"), ("masked", "best-masked.png")):
            if row[key]:
                path = output / row[key]
                assert path.resolve().is_relative_to(output.resolve())
                assert path.read_bytes() == (root / f"passages/{number}/{name}").read_bytes()
    assert sheet[0]["representations"] == "full_original;rectangular_crop;binary_mask;masked_image"
    assert not sheet[2]["masked"]


@pytest.mark.parametrize("case", ["legacy", "state", "sequence", "duplicate", "order", "boundary",
    "nan", "type", "candidate_track", "bbox", "warning", "traversal", "backslash", "crossjob",
    "track_asset", "optional_escape", "missing_artifact", "missing_source", "hash"])
def test_invalid_input_never_publishes(run, case):
    root, summary, video, output = run
    evidence = summary["passage_evidence"]
    item = evidence["intervals"][0]
    candidate = item["candidate"]
    if case == "legacy":
        del summary["passage_evidence"]
    elif case == "state":
        evidence["status"] = "failed"
    elif case == "sequence":
        item["sequence"] = True
    elif case == "duplicate":
        evidence["intervals"][1]["track_id"] = 9
    elif case == "order":
        evidence["intervals"].reverse()
    elif case == "boundary":
        item["end_frame"] = 11
    elif case == "nan":
        candidate["timestamp_ms"] = float("inf")
    elif case == "type":
        candidate["segmentation"] = []
    elif case == "candidate_track":
        candidate["track_id"] = 100
    elif case == "bbox":
        candidate["bbox"] = [1, 2, 0, 3]
    elif case == "warning":
        item["warnings"] = ["=HYPERLINK(1)"]
    elif case in ("traversal", "backslash", "crossjob", "track_asset"):
        candidate["original"] = {"traversal": "../best-original.jpg", "backslash": "passages\\1\\best-original.jpg",
            "crossjob": "/other/result/passages/1/best-original.jpg", "track_asset": "tracks/9/best-original.jpg"}[case]
    elif case == "optional_escape":
        candidate["segmentation"] = dict(status="ok", best_mask="../mask.png")
    elif case == "missing_artifact":
        (root / "passages/1/best-crop.png").unlink()
    elif case == "missing_source":
        video.unlink()
    elif case == "hash":
        summary["input_sha256"] = "a" * 64
    with pytest.raises((ValueError, OSError)):
        export(run)
    assert not output.exists()
    assert not list(output.parent.glob(".passage-labels-*"))


def test_identity_required_but_digest_metadata_suffices(run):
    with pytest.raises(ValueError, match="source"):
        export(run, source=False)
    run[1]["input_sha256"] = "a" * 64
    assert export(run, source=False)["source_video_sha256"] == "a" * 64


def test_existing_overlap_and_copy_failure(run, monkeypatch):
    root, summary, video, output = run
    output.mkdir()
    marker = output / "keep"
    marker.write_bytes(b"unchanged")
    with pytest.raises(ValueError):
        export(run)
    assert marker.read_bytes() == b"unchanged"
    marker.unlink()
    output.rmdir()
    with pytest.raises(ValueError):
        export_passage_labels(root, root / "nested", source_video=video)
    def fail(*args):
        raise OSError("copy failed")
    monkeypatch.setattr("vacca_video.passage_bcs_export.shutil.copyfile", fail)
    with pytest.raises(OSError, match="copy failed"):
        export(run)
    assert not output.exists() and not list(output.parent.glob(".passage-labels-*"))


@pytest.mark.parametrize("location", ["root", "ancestor", "file", "output"])
def test_link_refusal(run, location):
    root, summary, video, output = run
    export(run)  # Save a valid summary, then remove only this test's pack.
    import shutil
    shutil.rmtree(output)
    target = {"root": root, "ancestor": root / "passages/1",
              "file": root / "passages/1/best-original.jpg", "output": output}[location]
    saved = target.with_name(target.name + "-real")
    if target.exists():
        target.rename(saved)
    else:
        saved.mkdir()
    try:
        target.symlink_to(saved, target_is_directory=saved.is_dir())
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError):
        export_passage_labels(root, output, source_video=video)


@pytest.mark.parametrize("payload", ['{"tracks":[],"tracks":[]}', '[]', '{"x":1e999}', ' ' * (4 * 1024 * 1024 + 1)],
                         ids=["duplicate_keys", "wrong_type", "nonfinite", "oversize"])
def test_bounded_strict_json(run, payload):
    root, _, video, output = run
    (root / "summary.json").write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        export_passage_labels(root, output, source_video=video)
    assert not output.exists()


@pytest.mark.parametrize("location", ["root", "ancestor", "output_parent"])
def test_junction_refusal(run, location):
    import os
    import subprocess
    if os.name != "nt":
        pytest.skip("Windows junction test")
    root, _, video, output = run
    export(run)
    target = {"root": root, "ancestor": root / "passages", "output_parent": output}[location]
    saved = target.with_name(target.name + "-real")
    target.rename(saved)
    completed = subprocess.run(["cmd", "/c", "mklink", "/J", str(target), str(saved)], capture_output=True)
    if completed.returncode:
        pytest.skip("junction creation unavailable")
    try:
        with pytest.raises(ValueError):
            export_passage_labels(root, output / "new" if location == "output_parent" else output,
                                  source_video=video)
    finally:
        target.rmdir()
