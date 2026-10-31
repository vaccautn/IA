from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vacca_video.bcs_export import export_crops, materialize_dataset  # noqa: E402


def _run(root: Path, name: str, video: Path, tracks: list[dict]) -> Path:
    run = root / name
    for track in tracks:
        track_dir = run / "tracks" / str(track["track_id"])
        track_dir.mkdir(parents=True)
        (track_dir / "best-crop.png").write_bytes(b"rect-%d" % track["track_id"])
        if (track.get("segmentation") or {}).get("status") == "ok":
            (track_dir / "best-masked.png").write_bytes(b"masked-%d" % track["track_id"])
    summary = {"input": str(video), "tracker": "bytetrack.yaml", "tracks": tracks}
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run


def _track(track_id, status="candidate", masked=False, continuation_of=None, warnings=()):
    track = {"track_id": track_id, "status": status, "warnings": list(warnings),
             "best_frame_index": 10 * track_id, "best_timestamp_ms": 416.6666,
             "best_crop": f"tracks/{track_id}/best-crop.png",
             "possible_continuation_of": continuation_of,
             "segmentation": {"status": "not_requested"}}
    if masked:
        track["segmentation"] = {"status": "ok", "best_masked": f"tracks/{track_id}/best-masked.png"}
    return track


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "manga.mov"
    path.write_bytes(b"video bytes")
    return path


def _rows(output: Path) -> list[dict]:
    with (output / "labels.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_export_prefers_silhouette_and_skips_review_tracks(tmp_path, video):
    run = _run(tmp_path, "run", video, [_track(1, masked=True), _track(2), _track(3, "review")])
    result = export_crops([run], tmp_path / "out")
    rows = _rows(tmp_path / "out")
    assert [(row["track_id"], row["image_kind"]) for row in rows] == [("1", "masked"), ("2", "rect")]
    assert all(row["bcs_category"] == "" for row in rows), "Categories are never guessed"
    assert (tmp_path / "out/images" / rows[0]["image"]).read_bytes() == b"masked-1"
    assert result["images"] == 2 and result["masked"] == 1


def test_linked_fragments_share_one_split_group(tmp_path, video):
    run = _run(tmp_path, "run", video, [_track(9), _track(16, "review", continuation_of=9)])
    export_crops([run], tmp_path / "out", include_review=True)
    rows = _rows(tmp_path / "out")
    assert rows[0]["group"] == rows[1]["group"]
    assert rows[1]["track_status"] == "review"


def test_same_video_from_two_trackers_is_rejected(tmp_path, video):
    first = _run(tmp_path, "bytetrack", video, [_track(1)])
    second = _run(tmp_path, "botsort", video, [_track(1)])
    with pytest.raises(ValueError, match="un solo tracker"):
        export_crops([first, second], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_materialize_builds_category_folders_and_groups(tmp_path, video):
    run = _run(tmp_path, "run", video, [_track(1, masked=True), _track(2), _track(3)])
    export_crops([run], tmp_path / "out")
    labels = tmp_path / "out/labels.csv"
    rows = _rows(tmp_path / "out")
    rows[0]["bcs_category"], rows[1]["bcs_category"] = "3", " 4 "
    with labels.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = materialize_dataset(labels, tmp_path / "dataset")
    assert result["labelled"] == 2 and result["skipped"] == 1
    assert result["counts"] == {"1": 0, "2": 0, "3": 1, "4": 1, "5": 0}
    assert sorted(path.name for path in (tmp_path / "dataset").iterdir()) == ["1", "2", "3", "4", "5", "groups.json"]
    assert (tmp_path / "dataset/3" / rows[0]["image"]).read_bytes() == b"masked-1"
    groups = json.loads((tmp_path / "dataset/groups.json").read_text(encoding="utf-8"))
    assert set(groups) == {rows[0]["image"], rows[1]["image"]}


@pytest.mark.parametrize("value", ["6", "3.5", "gorda"])
def test_materialize_rejects_invalid_categories_before_copying(tmp_path, video, value):
    run = _run(tmp_path, "run", video, [_track(1)])
    export_crops([run], tmp_path / "out")
    labels = tmp_path / "out/labels.csv"
    rows = _rows(tmp_path / "out")
    rows[0]["bcs_category"] = value
    with labels.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="inválidas"):
        materialize_dataset(labels, tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()


def test_materialize_rejects_paths_outside_images(tmp_path, video):
    run = _run(tmp_path, "run", video, [_track(1)])
    export_crops([run], tmp_path / "out")
    labels = tmp_path / "out/labels.csv"
    rows = _rows(tmp_path / "out")
    rows[0].update(image="../labels.csv", bcs_category="2")
    with labels.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Imagen inválida"):
        materialize_dataset(labels, tmp_path / "dataset")


def test_folder_dataset_can_read_the_materialized_layout(tmp_path, video):
    pytest.importorskip("torchvision")
    from PIL import Image

    from vacca_bcs.dataset import BCSFolderDataset

    run = tmp_path / "run"
    track_dir = run / "tracks/1"
    track_dir.mkdir(parents=True)
    Image.new("RGB", (32, 32), "white").save(track_dir / "best-crop.png")
    summary = {"input": str(video), "tracker": "bytetrack.yaml", "tracks": [_track(1)]}
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    export_crops([run], tmp_path / "out")
    labels = tmp_path / "out/labels.csv"
    rows = _rows(tmp_path / "out")
    rows[0]["bcs_category"] = "5"
    with labels.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    materialize_dataset(labels, tmp_path / "dataset")
    dataset = BCSFolderDataset(tmp_path / "dataset")
    assert dataset.class_counts["5"] == 1
