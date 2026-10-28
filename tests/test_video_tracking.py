from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vacca_video.tracking import (  # noqa: E402
    clamp_box,
    crop_contamination,
    crop_with_padding,
    frame_quality,
    iou,
    track_video,
)

cv2 = pytest.importorskip("cv2")

WIDTH, HEIGHT, FPS = 200, 120, 24.0


def test_clamp_box_stays_inside_image():
    assert clamp_box((-4, 3, 110, 80), 100, 60) == (0, 3, 100, 60)


def test_iou_is_zero_for_separate_boxes_and_one_for_same_box():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1


def test_quality_penalizes_overlap():
    image = np.full((100, 100, 3), 127, dtype=np.uint8)
    clean = frame_quality(image, (20, 20, 80, 80), 0.8, 0.0)
    contaminated = frame_quality(image, (20, 20, 80, 80), 0.8, 0.8)
    assert clean > contaminated


def test_crop_is_fixed_size_and_preserves_canvas_shape():
    image = np.zeros((60, 100, 3), dtype=np.uint8)
    crop = crop_with_padding(image, (10, 10, 50, 50), output_size=(320, 320))
    assert crop.shape == (320, 320, 3)


def test_contamination_sees_neighbour_in_padding_that_iou_misses():
    mine, neighbour = (10, 10, 110, 90), (105, 10, 190, 90)
    assert iou(mine, neighbour) < 0.05
    assert crop_contamination(mine, [neighbour], WIDTH, HEIGHT) > 0.05


def test_contamination_counts_small_box_inside_large_crop():
    large, small = (0, 0, 200, 120), (80, 40, 120, 80)
    assert iou(large, small) < 0.1
    assert crop_contamination(large, [small], WIDTH, HEIGHT) == pytest.approx(1600 / 24000)


def test_contamination_uses_union_not_sum():
    box = (0, 0, 100, 100)
    others = [(0, 0, 50, 100), (0, 0, 50, 100)]
    assert crop_contamination(box, others, 100, 100, margin=0) == pytest.approx(0.5)


class _Array:
    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _Boxes:
    def __init__(self, items):
        self.xyxy = _Array([box for _, box, _ in items])
        self.conf = _Array([conf for _, _, conf in items])
        self.cls = _Array([0] * len(items))
        ids = [track_id for track_id, _, _ in items]
        self.id = None if any(track_id is None for track_id in ids) else _Array(ids)

    def __len__(self):
        return len(self.xyxy.values)


class _Result:
    names = {0: "cow"}

    def __init__(self, items):
        self.boxes = _Boxes(items) if items else None


class FakeModel:
    """Replays scripted (track_id, box, confidence) detections per call."""

    def __init__(self, script):
        self.script = script
        self.calls = 0

    def track(self, image, **kwargs):
        items = self.script(self.calls)
        self.calls += 1
        return [_Result(items)]


def _write_video(path: Path, frames: int) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    rng = np.random.default_rng(0)
    for _ in range(frames):
        writer.write(rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def media(tmp_path):
    model = tmp_path / "model.pt"
    model.write_bytes(b"fake weights")
    return tmp_path, model


def _run(media, frames, script, **kwargs):
    root, model_path = media
    video = _write_video(root / "input.mp4", frames)
    output = root / "run"
    summary = track_video(video, output, model_path, output_size=(64, 64), model=FakeModel(script), **kwargs)
    return summary, output


def _clip_frames(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    count = 0
    while capture.read()[0]:
        count += 1
    capture.release()
    return count


def test_long_gap_splits_segment_and_keeps_clip_timeline(media):
    def script(call):
        return [(1, (20, 20, 80, 90), 0.9)] if call < 5 or call >= 20 else []

    summary, output = _run(media, 25, script)
    track = summary["tracks"][0]
    assert [(s["first_frame"], s["last_frame"]) for s in track["segments"]] == [(0, 4), (20, 24)]
    assert track["gaps"] == [{"after_frame": 4, "before_frame": 20, "missing_samples": 15,
                              "between_observations_ms": 666.667, "splits_segment": True}]
    assert track["clip_frames"] == 25 and track["clip_placeholder_frames"] == 15
    assert _clip_frames(output / track["clip"]) == 25
    assert track["status"] == "review"
    assert "long_gap_review_required" in track["warnings"]


def test_short_gap_is_recorded_without_splitting(media):
    def script(call):
        return [] if call in (3, 4) else [(1, (20, 20, 80, 90), 0.9)]

    summary, _ = _run(media, 10, script)
    track = summary["tracks"][0]
    assert len(track["segments"]) == 1 and track["segments"][0]["observations"] == 8
    assert track["gaps"][0]["missing_samples"] == 2
    assert not track["gaps"][0]["splits_segment"]
    assert track["clip_frames"] == 10
    assert track["status"] == "candidate"


def test_neighbour_inside_padding_marks_review_even_with_low_iou(media):
    def script(call):
        return [(1, (10, 10, 110, 90), 0.9), (2, (105, 10, 190, 90), 0.9)]

    summary, _ = _run(media, 6, script)
    first = summary["tracks"][0]
    assert first["best_iou"] < 0.05
    assert first["best_crop_contamination"] > 0.05
    assert "other_detection_in_crop_at_best_frame" in first["warnings"]


def test_identical_boxes_count_as_contamination(media):
    def script(call):
        return [(1, (20, 20, 80, 90), 0.9), (2, (20, 20, 80, 90), 0.8)]

    summary, _ = _run(media, 6, script)
    # The 60x70 box covers the crop except its 10% padding (72x84).
    for track in summary["tracks"]:
        assert track["best_crop_contamination"] == pytest.approx(4200 / 6048, abs=1e-6)
        assert track["status"] == "review"


def test_untracked_detections_still_count_as_contamination(media):
    def script(call):
        if call == 0:
            return [(None, (20, 20, 80, 90), 0.9), (None, (60, 20, 120, 90), 0.9)]
        return [(1, (20, 20, 80, 90), 0.9)]

    summary, output = _run(media, 6, script)
    assert [track["track_id"] for track in summary["tracks"]] == [1]
    lines = (output / "observations.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["frame_index"] == 1


def test_short_track_and_border_touch_are_reviewed(media):
    def script(call):
        return [(3, (0, 10, 60, 100), 0.9)] if call < 2 else []

    summary, _ = _run(media, 6, script)
    track = summary["tracks"][0]
    assert {"short_track_review_required", "border_touch_at_best_frame"} <= set(track["warnings"])


def test_frames_sampled_counts_analysed_frames_not_detections(media):
    summary, _ = _run(media, 24, lambda call: [], sample_fps=6)
    assert summary["frames_read"] == 24
    assert summary["frames_sampled"] == 6
    assert summary["clip_fps"] == 6
    assert summary["tracks"] == []


def test_summary_is_auditable(media):
    summary, output = _run(media, 6, lambda call: [(1, (20, 20, 80, 90), 0.9)])
    assert summary["model_sha256"] == hashlib.sha256(b"fake weights").hexdigest()
    assert summary["config"]["max_gap_seconds"] == 0.5
    track = summary["tracks"][0]
    assert "best_frame" not in track
    assert track["best_bbox"] == [20, 20, 80, 90]
    for key in ("clip", "best_original", "best_crop"):
        assert (output / track[key]).is_file()


def test_webm_clip_is_written_for_browsers(media):
    summary, output = _run(media, 6, lambda call: [(1, (20, 20, 80, 90), 0.9)], clip_format="webm")
    track = summary["tracks"][0]
    assert track["clip"] == "tracks/1/clip.webm"
    assert (output / track["clip"]).read_bytes()[:4] == b"\x1a\x45\xdf\xa3"  # EBML/WebM header
    assert _clip_frames(output / track["clip"]) == 6


def test_max_seconds_stops_reading_and_reports_reason(media):
    summary, _ = _run(media, 24, lambda call: [], max_seconds=0.5)
    assert summary["frames_read"] == 12
    assert summary["stop_reason"] == "time_limit"
    full, _ = _run_again(media, 3)
    assert full["stop_reason"] == "end_of_video"


def test_max_frames_bounds_dense_tracking_cost(media):
    summary, _ = _run(media, 10, lambda call: [], max_frames=4)
    assert summary["frames_read"] == 4
    assert summary["frames_sampled"] == 4
    assert summary["stop_reason"] == "frame_limit"


def _run_again(media, frames):
    root, model_path = media
    video = _write_video(root / "again.mp4", frames)
    output = root / "again"
    return track_video(video, output, model_path, output_size=(64, 64),
                       model=FakeModel(lambda call: [])), output


def test_invalid_clip_format_and_max_seconds_are_rejected(media):
    root, model_path = media
    video = _write_video(root / "input.mp4", 2)
    for kwargs in ({"clip_format": "avi"}, {"max_seconds": 0}):
        with pytest.raises(ValueError):
            track_video(video, root / "run", model_path, model=FakeModel(lambda call: []), **kwargs)
    assert not (root / "run").exists()


def test_missing_lap_fails_instead_of_auto_installing(media, monkeypatch):
    import vacca_video.tracking as tracking

    root, model_path = media
    video = _write_video(root / "input.mp4", 2)
    monkeypatch.setattr(tracking, "find_spec", lambda name: None)
    with pytest.raises(ImportError, match="lap"):
        track_video(video, root / "run", model_path)
    assert not (root / "run").exists()


def test_handoff_in_same_place_links_fragments_without_merging(media):
    # Track 1 is lost after frame 4 and re-born as 2 two frames later, slightly moved.
    def script(call):
        if call < 5:
            return [(1, (20, 20, 80, 90), 0.9)]
        if call >= 7:
            return [(2, (26, 24, 86, 94), 0.9)]
        return []

    summary, _ = _run(media, 12, script)
    first, second = summary["tracks"]
    assert (first["possible_continued_by"], second["possible_continuation_of"]) == (2, 1)
    assert "possible_fragment_review_required" in second["warnings"]
    assert "possible_fragment_review_required" not in first["warnings"]
    assert first["status"] == "candidate" and second["status"] == "review"


@pytest.mark.parametrize("later_box,frames_apart", [
    ((120, 20, 180, 90), 2),   # a different place in the chute
    ((26, 24, 86, 94), 15),    # same place, but after more than max_gap_seconds
])
def test_distant_or_late_tracks_are_not_linked(media, later_box, frames_apart):
    def script(call):
        if call < 5:
            return [(1, (20, 20, 80, 90), 0.9)]
        if call >= 4 + frames_apart:
            return [(2, later_box, 0.9)]
        return []

    summary, _ = _run(media, 4 + frames_apart + 6, script)
    assert all(track["possible_continuation_of"] is None for track in summary["tracks"])
    assert all(track["possible_continued_by"] is None for track in summary["tracks"])


def test_each_track_links_at_most_once_to_the_best_overlap(media):
    def script(call):
        if call < 5:
            return [(1, (20, 20, 80, 90), 0.9)]
        if call >= 6:
            return [(2, (22, 20, 82, 90), 0.9), (3, (30, 26, 90, 96), 0.9)]
        return []

    summary, _ = _run(media, 10, script)
    links = {track["track_id"]: track["possible_continuation_of"] for track in summary["tracks"]}
    assert links == {1: None, 2: 1, 3: None}


def test_distant_animal_is_flagged_as_too_small(media):
    # 15x10 px is under 1% of the 200x120 frame.
    summary, _ = _run(media, 6, lambda call: [(1, (90, 50, 105, 60), 0.9)])
    assert "small_at_best_frame" in summary["tracks"][0]["warnings"]
    big, _ = _run_again_with(media, lambda call: [(1, (20, 20, 80, 90), 0.9)])
    assert "small_at_best_frame" not in big["tracks"][0]["warnings"]


def _run_again_with(media, script):
    root, model_path = media
    video = _write_video(root / "again.mp4", 6)
    return track_video(video, root / "again", model_path, output_size=(64, 64),
                       model=FakeModel(script)), root / "again"


def _box_mask(box, shape=(HEIGHT, WIDTH)):
    mask = np.zeros(shape, dtype=bool)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = True
    return mask


def test_assess_mask_accepts_animal_and_rejects_background_grabs():
    from vacca_video.tracking import assess_mask

    box = (20, 20, 80, 90)
    good = assess_mask(_box_mask((30, 30, 70, 85)), box, [])
    assert good["status"] == "ok" and 0.15 < good["mask_fill"] < 0.95
    assert assess_mask(np.zeros((HEIGHT, WIDTH), bool), box, [])["reason"] == "empty_mask"
    assert assess_mask(_box_mask((40, 40, 45, 45)), box, [])["reason"] == "mask_too_small"
    assert assess_mask(_box_mask(box), box, [])["reason"] == "mask_fills_box"
    assert assess_mask(_box_mask((30, 30, 200, 120)), box, [])["reason"] == "mask_leaves_box"
    overlap = assess_mask(_box_mask((30, 30, 70, 85)), box, [(50, 30, 120, 85)])
    assert overlap["status"] == "ok" and overlap["mask_other_overlap"] == pytest.approx(0.5)


def test_accepted_mask_writes_silhouette_and_aligned_mask(media):
    calls = []

    def segmenter(image, bbox):
        calls.append(bbox)
        return _box_mask((30, 30, 70, 85))

    segmenter.metadata = {"model": "fake_sam.pt", "model_sha256": "abc", "prompt": "box+point@0.4"}
    summary, output = _run(media, 6, lambda call: [(1, (20, 20, 80, 90), 0.9)], segmenter=segmenter)
    track = summary["tracks"][0]
    assert calls == [(20, 20, 80, 90)]
    assert track["segmentation"]["status"] == "ok"
    masked = cv2.imread(str(output / track["segmentation"]["best_masked"]))
    mask = cv2.imread(str(output / track["segmentation"]["best_mask"]), cv2.IMREAD_GRAYSCALE)
    assert masked.shape == mask.shape[:2] + (3,) == (64, 64, 3)
    assert (masked[mask == 0] == 28).all(), "Background outside the mask is flat"
    assert summary["segmentation"]["model"] == "fake_sam.pt"
    assert summary["segmentation"]["limit"] == 40


def test_rejected_or_failed_masks_keep_rectangular_crop(media):
    def rejected(image, bbox):
        return _box_mask(bbox)

    def broken(image, bbox):
        raise RuntimeError("fixture")

    for segmenter, expected in ((rejected, "rejected"), (broken, "failed")):
        root, model_path = media
        video = _write_video(root / f"{expected}.mp4", 4)
        summary = track_video(video, root / expected, model_path, output_size=(64, 64),
                              model=FakeModel(lambda call: [(1, (20, 20, 80, 90), 0.9)]),
                              segmenter=segmenter)
        track = summary["tracks"][0]
        assert track["segmentation"]["status"] == expected
        assert not (root / expected / "tracks/1/best-masked.png").exists()
        assert (root / expected / track["best_crop"]).is_file()


def test_segment_limit_prefers_longest_tracks(media):
    def script(call):
        return [(1, (10, 10, 60, 60), 0.9)] + ([(2, (120, 10, 180, 60), 0.9)] if call < 2 else [])

    summary, _ = _run(media, 6, script, segmenter=lambda image, bbox: _box_mask((15, 15, 50, 55)),
                      segment_limit=1)
    states = {track["track_id"]: track["segmentation"]["status"] for track in summary["tracks"]}
    assert states == {1: "ok", 2: "skipped"}


def test_without_segmenter_nothing_is_segmented(media):
    summary, _ = _run(media, 4, lambda call: [(1, (20, 20, 80, 90), 0.9)])
    assert summary["segmentation"] is None
    assert summary["tracks"][0]["segmentation"] == {"status": "not_requested"}


def test_missing_segmentation_weights_are_never_downloaded(tmp_path):
    from vacca_video.tracking import load_sam_segmenter

    with pytest.raises(ValueError, match="no se descarga"):
        load_sam_segmenter(tmp_path / "mobile_sam.pt")
    assert not (tmp_path / "mobile_sam.pt").exists()


def test_existing_output_is_never_overwritten(media):
    root, model_path = media
    video = _write_video(root / "input.mp4", 3)
    output = root / "run"
    output.mkdir()
    with pytest.raises(ValueError, match="ya existe"):
        track_video(video, output, model_path, model=FakeModel(lambda call: []))


def test_unreadable_video_does_not_leave_output_folder(media):
    root, model_path = media
    video = root / "broken.mp4"
    video.write_bytes(b"not a video")
    output = root / "run"
    with pytest.raises(ValueError):
        track_video(video, output, model_path, model=FakeModel(lambda call: []))
    assert not output.exists()
