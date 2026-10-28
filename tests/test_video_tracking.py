from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vacca_video.tracking import (  # noqa: E402
    PassageEvidence,
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


def _passages(samples, count=3, margin=0.1):
    evidence = PassageEvidence(count, margin)
    for index, scores in enumerate(samples):
        evidence.observe([{"track_id": key, "confidence": score} for key, score in scores],
                         index * 4, index * 250.0)
    return evidence


def test_passage_short_crossover_and_eof_pending_do_not_close_incumbent():
    evidence = _passages([[(80, 0.9)], [(80, 0.5), (2, 0.8)], [(80, 0.5), (2, 0.8)]])
    assert [item["track_id"] for item in evidence.intervals] == [80]
    assert evidence.pending == {"track_id": 2, "count": 2, "start_frame": 4,
                                "start_timestamp_ms": 250.0}


@pytest.mark.parametrize("reset", [[], [(2, 0.9)], [(80, 0.8)],
                                  [(80, 0.8), (2, 0.85)], [(80, 0.8), (2, 0.8)],
                                  [(80, 0.4), (2, 0.9), (3, 0.9)]])
def test_passage_missing_or_nondominant_sample_resets_confirmation(reset):
    evidence = _passages([[(80, 0.9)], [(80, 0.4), (2, 0.9)], reset,
                          [(80, 0.4), (2, 0.9)], [(80, 0.4), (2, 0.9)]])
    assert len(evidence.intervals) == 1
    assert evidence.pending["count"] == 2
    assert evidence.pending["start_frame"] == 12


def test_passage_changed_challenger_resets_streak():
    evidence = _passages([[(80, 0.9)], [(80, 0.4), (2, 0.9)],
                          [(80, 0.4), (3, 0.9)], [(80, 0.4), (2, 0.9)]], count=2)
    assert len(evidence.intervals) == 1
    assert evidence.pending["track_id"] == 2 and evidence.pending["count"] == 1


def test_unique_strongest_challenger_margin_is_compared_to_incumbent():
    evidence = _passages([[(80, 0.9)]] +
                         [[(80, 0.4), (2, 0.9), (3, 0.85)]] * 3)
    assert [item["track_id"] for item in evidence.intervals] == [80, 2]


@pytest.mark.parametrize("ids", [(90, 2), (2, 90)])
def test_passage_sustained_current_confidence_not_historical_max(ids):
    incumbent, challenger = ids
    evidence = _passages([[(incumbent, 0.99)]] +
                         [[(challenger, 0.8), (incumbent, 0.5)]] * 3)
    first, second = evidence.intervals
    assert [first["track_id"], second["track_id"]] == list(ids)
    assert first["end_frame"] == second["start_frame"] == 12
    assert first["end_timestamp_ms"] == second["start_timestamp_ms"] == 750.0
    assert first["end_exclusive"] is True
    assert second["support"] == {"track_id": challenger, "count": 3, "start_frame": 4,
                                  "start_timestamp_ms": 250.0, "confirmation_frame": 12,
                                  "confirmation_timestamp_ms": 750.0}


@pytest.mark.parametrize("scores", [[(5, 0.9), (1, 0.9)], [(1, 0.9), (5, 0.9)],
                                    [(1, 0.85), (5, 0.9)]])
def test_passage_ambiguous_initial_competitors_stay_unknown(scores):
    evidence = _passages([scores] * 5)
    assert evidence.intervals == [] and evidence.pending is None
    assert "ambiguous_dominance_unresolved" in evidence.warnings


def test_passage_finalized_id_cannot_reenter_on_confidence_fluctuations():
    evidence = _passages([[(80, 0.9)]] + [[(80, 0.4), (2, 0.9)]] * 3 +
                         [[(80, 0.99), (2, 0.4)]] * 5)
    assert [item["track_id"] for item in evidence.intervals] == [80, 2]
    assert "finalized_track_reentry_unresolved" in evidence.warnings


def test_passage_id_return_after_absence_is_unresolved_not_new_identity():
    evidence = _passages([[(5, 0.9)], [], [(5, 0.9)]])
    assert len(evidence.intervals) == 1
    assert "possible_track_id_reuse_unresolved" in evidence.warnings


@pytest.mark.parametrize("sample,warning", [
    ([(5, 0.4), (2, float("nan"))], "invalid_confidence_unresolved"),
    ([(5, 0.4), (2, 1.1)], "invalid_confidence_unresolved"),
    ([(5, 0.4), (2, 0.9), (2, 0.95)], "duplicate_track_id_unresolved"),
])
def test_invalid_or_duplicate_observations_cannot_confirm_passage(sample, warning):
    evidence = _passages([[(5, 0.9)], [(5, 0.4), (2, 0.9)], sample,
                          [(5, 0.4), (2, 0.9)]], count=2)
    assert len(evidence.intervals) == 1 and evidence.pending["count"] == 1
    assert warning in evidence.warnings
    assert _passages([sample]).intervals == []


@pytest.mark.parametrize("kwargs", [
    {"passage_confirmation_count": value} for value in (0, -1, 1.5, True)
] + [{"passage_confidence_margin": value} for value in (-0.1, float("nan"), float("inf"), True)])
def test_invalid_passage_parameters_fail_before_output(media, kwargs):
    root, model = media
    with pytest.raises(ValueError, match="passage_"):
        track_video(root / "unused.mp4", root / "run", model,
                    model=FakeModel(lambda call: []), **kwargs)
    assert not (root / "run").exists()


def test_zero_margin_still_requires_unique_strict_dominance():
    evidence = _passages([[(4, 0.5), (1, 0.5)], [(4, 0.6), (1, 0.5)],
                          [(4, 0.5), (1, 0.6)]], count=1, margin=0)
    assert [item["track_id"] for item in evidence.intervals] == [4, 1]


def test_passage_summary_integrates_sampled_frames_policy_and_inactive_eof(media):
    samples = [[(90, 0.99)], [(90, 0.5), (2, 0.9)],
               [(90, 0.5), (2, 0.9)], [], []]
    summary, output = _run(media, 20,
                           lambda call: [(key, (20, 20, 80, 90), score)
                                         for key, score in samples[call]],
                           sample_fps=6, passage_confirmation_count=2,
                           passage_confidence_margin=0.2)
    evidence = summary["passage_evidence"]
    assert json.loads((output / "summary.json").read_text())["passage_evidence"] == evidence
    assert evidence["policy"]["confirmation_count"] == 2
    assert evidence["policy"]["confidence_margin"] == 0.2
    assert "uncalibrated" in evidence["policy"]["calibration"]
    assert evidence["status"] == "provisional_evidence_not_unique_animal_identity"
    first, last = evidence["intervals"]
    assert (first["start_frame"], first["end_frame"], last["start_frame"], last["end_frame"]) == (0, 8, 8, 16)
    assert last["end_timestamp_ms"] == pytest.approx(16000 / FPS)
    assert last["end_exclusive"] is False
    assert last["support"]["start_frame"] == 4
    for interval in (first, last):
        candidate = interval["candidate"]
        assert candidate["status"] == "available"
        assert candidate["track_id"] == interval["track_id"]
        assert (output / candidate["original"]).is_file()
    assert first["candidate"]["frame_index"] < 8
    assert last["candidate"]["frame_index"] == 8
    assert "missing_incumbent_attribution_unresolved" in evidence["warnings"]
    assert [track["track_id"] for track in summary["tracks"]] == [2, 90]
    assert all(track["segmentation"]["status"] == "not_requested" for track in summary["tracks"])


def test_passage_initial_incumbent_preserved_and_fragment_not_claimed_as_new_cow(media):
    summary, _ = _run(media, 8, lambda call: [
        (90 if call < 3 else 2, (20, 20, 80, 90), 0.9)])
    evidence = summary["passage_evidence"]
    assert len(evidence["intervals"]) == 1
    candidate = evidence["intervals"][0]
    assert candidate["track_id"] == 90 and candidate["end_frame"] == 7
    assert candidate["warnings"] == ["possible_fragment_identity_unresolved"]
    unresolved, = evidence["unresolved_tracks"]
    assert unresolved["track_id"] == 2 and unresolved["possible_continuation_of"] == 90
    assert unresolved["status"] == "unresolved_passage_attribution"


def test_ambiguous_observed_tracks_are_retained_without_identity_or_mask_filter(media):
    summary, _ = _run(media, 3, lambda call: [(7, (20, 20, 80, 90), 0.9),
                                             (1, (100, 20, 180, 90), 0.9)])
    evidence = summary["passage_evidence"]
    assert evidence["intervals"] == []
    assert {item["track_id"] for item in evidence["unresolved_tracks"]} == {7, 1}
    assert all(item["first_frame"] == 0 and item["last_frame"] == 2
               for item in evidence["unresolved_tracks"])


@pytest.mark.parametrize("score_type", [float, np.float32], ids=["float64", "float32"])
@pytest.mark.parametrize("high,low", [(0.9, 0.8), (0.8, 0.7)])
def test_nominal_margin_initial_selection_and_sustained_handoff(media, score_type, high, low):
    high, low = float(score_type(high)), float(score_type(low))
    summary, _ = _run(media, 4, lambda call: [
        (90, (20, 20, 80, 90), high if call == 0 else low),
        (2, (100, 20, 180, 90), low if call == 0 else high)])
    evidence = summary["passage_evidence"]
    assert evidence["policy"]["confidence_margin"] == 0.1
    assert evidence["policy"]["numerical_margin_epsilon"] == 2**-24
    first, second = evidence["intervals"]
    assert (first["track_id"], first["start_frame"], first["end_frame"]) == (90, 0, 3)
    assert (second["track_id"], second["start_frame"]) == (2, 3)
    assert second["support"]["count"] == 3
    assert second["support"]["start_frame"] == 1


@pytest.mark.parametrize("score_type", [float, np.float32], ids=["float64", "float32"])
@pytest.mark.parametrize("high,low", [(0.9, 0.8), (0.8, 0.7)])
def test_nominal_margin_handoff_from_already_selected_incumbent(score_type, high, low):
    high, low = float(score_type(high)), float(score_type(low))
    evidence = _passages([[(90, 0.99)]] + [[(90, low), (2, high)]] * 3)
    assert [item["track_id"] for item in evidence.intervals] == [90, 2]
    assert evidence.intervals[1]["support"]["count"] == 3


@pytest.mark.parametrize("score_type", [float, np.float32], ids=["float64", "float32"])
@pytest.mark.parametrize("high", [0.899999, 0.8, 0.800000001],
                         ids=["below_margin_outside_tolerance", "exact_tie", "near_tie"])
def test_margin_tolerance_does_not_admit_subthreshold_or_tied_scores(score_type, high):
    high, low = float(score_type(high)), float(score_type(0.8))
    sample = [(2, high), (90, low)]
    assert _passages([sample] * 4).intervals == []
    evidence = _passages([[(90, 0.99)]] + [sample] * 4)
    assert [item["track_id"] for item in evidence.intervals] == [90]
    assert evidence.pending is None


@pytest.fixture
def passage_images(monkeypatch):
    """Distinct, JPEG-stable source pixels and controlled quality per sample."""
    import vacca_video.tracking as tracking

    original = FakeModel.track

    def track(self, image, **kwargs):
        image[:] = 20 + self.calls * 20
        return original(self, image, **kwargs)

    monkeypatch.setattr(FakeModel, "track", track)

    def quality(scores):
        monkeypatch.setattr(tracking, "frame_quality",
                            lambda image, *args: scores[(int(image[0, 0, 0]) - 20) // 20])

    return quality


def _assert_passage_image(output, interval, frame):
    candidate = interval["candidate"]
    assert candidate["status"] == "available"
    assert candidate["frame_index"] == frame
    assert candidate["timestamp_ms"] == pytest.approx(frame * 1000 / FPS)
    assert candidate["track_id"] == interval["track_id"]
    assert candidate["bcs_eligibility"] == "not_assessed"
    assert candidate["bcs_status"] == "not_evaluated"
    assert candidate["original"] == f"passages/{interval['sequence']}/best-original.jpg"
    image = cv2.imread(str(output / candidate["original"]))
    assert image.shape == (HEIGHT, WIDTH, 3)
    assert np.abs(image.astype(int) - (20 + frame * 20)).max() <= 1
    assert interval["start_frame"] <= frame <= interval["end_frame"]
    if interval["end_exclusive"]:
        assert frame < interval["end_frame"]
    assert not {"best_crop", "best_mask", "segmentation"} & candidate.keys()
    return candidate


def test_passage_candidate_never_inherits_global_best_and_keeps_earliest_tie(media, passage_images):
    passage_images([9, 5, 4, 3, 3, 2])
    summary, output = _run(media, 6, lambda call: [
        (90, (20, 20, 80, 90), 0.95 if call == 0 else 0.5),
        (2, (100, 20, 180, 90), 0.7 if call == 0 else 0.9)],
        segmenter=lambda image, bbox: _box_mask((110, 30, 170, 85)))
    first, second = summary["passage_evidence"]["intervals"]
    assert [first["sequence"], second["sequence"]] == [1, 2]
    _assert_passage_image(output, first, 0)
    candidate = _assert_passage_image(output, second, 3)
    track = next(item for item in summary["tracks"] if item["track_id"] == 2)
    assert track["best_frame_index"] == 0
    assert track["segmentation"]["status"] == "ok"
    assert candidate["confidence"] == 0.9
    assert candidate["bbox"] == [100, 20, 180, 90]
    assert candidate["quality"]["score"] == 3
    assert candidate["quality"]["method"] == "frame_quality"
    assert candidate["quality"]["crop_contamination"] == 0
    assert candidate["quality"]["crop_margin"] == 0.1
    assert candidate["original"] != track["best_original"]
    assert json.loads((output / "summary.json").read_text())["passage_evidence"] == summary["passage_evidence"]


def test_passage_confirmation_frame_not_available_to_old_interval(media, passage_images):
    passage_images([1, 2, 3, 10])
    summary, output = _run(media, 4, lambda call: [
        (90, (20, 20, 80, 90), 0.95 if call == 0 else 0.5),
        (2, (100, 20, 180, 90), 0.7 if call == 0 else 0.9)])
    first, final = summary["passage_evidence"]["intervals"]
    assert first["end_frame"] == final["start_frame"] == 3
    _assert_passage_image(output, first, 2)
    _assert_passage_image(output, final, 3)
    assert final["end_frame"] == 3 and final["end_exclusive"] is False


def test_passage_missing_samples_and_reset_do_not_capture_other_track(media, passage_images):
    passage_images([1, 2, 99, 3, 4, 99])
    samples = [[(90, 0.95)], [(90, 0.5), (2, 0.9)], [(2, 0.9)],
               [(90, 0.5), (2, 0.9)], [(90, 0.5), (2, 0.9)], [(2, 0.9)]]
    summary, output = _run(media, 6, lambda call: [
        (key, (20, 20, 80, 90), score) for key, score in samples[call]])
    interval, = summary["passage_evidence"]["intervals"]
    _assert_passage_image(output, interval, 4)
    assert interval["end_frame"] == 5
    assert summary["passage_evidence"]["unresolved_tracks"][0]["track_id"] == 2
    assert {item["track_id"] for item in summary["tracks"]} == {90, 2}


@pytest.mark.parametrize("ambiguous", [False, True])
def test_no_selected_passage_creates_no_candidate_artifacts(media, ambiguous):
    summary, output = _run(media, 3, lambda call: [
        (1, (20, 20, 80, 90), 0.9), (2, (100, 20, 180, 90), 0.9)] if ambiguous else [])
    assert summary["passage_evidence"]["intervals"] == []
    assert len(summary["passage_evidence"]["unresolved_tracks"]) == (2 if ambiguous else 0)
    assert not (output / "passages").exists()


def test_passage_without_capture_is_explicit_not_a_bcs_result():
    from vacca_video.tracking import TrackStats

    evidence = _passages([[(1, 0.9)]])
    interval, = evidence.summary({1: TrackStats(1)})["intervals"]
    assert interval["candidate"] == {"status": "not_available", "reason": "no_candidate_captured",
                                     "bcs_eligibility": "not_assessed", "bcs_status": "not_evaluated"}


@pytest.mark.parametrize("failure", ["false", "raise", "replace"])
@pytest.mark.parametrize("failed_frame", [0, 1], ids=["initial", "replacement"])
def test_passage_write_failure_releases_resources_without_publishing(
        media, monkeypatch, passage_images, failure, failed_frame):
    passage_images([1, 2])
    imwrite, capture_factory, writer_factory = cv2.imwrite, cv2.VideoCapture, cv2.VideoWriter
    released = []

    class Resource:
        def __init__(self, resource, name):
            self.resource, self.name = resource, name

        def __getattr__(self, name):
            return getattr(self.resource, name)

        def release(self):
            released.append(self.name)
            self.resource.release()

    # Patch after input creation, so only tracking resources are counted.
    root, model_path = media
    video = _write_video(root / "input.mp4", 2)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a: Resource(capture_factory(*a), "capture"))
    monkeypatch.setattr(cv2, "VideoWriter", lambda *a: Resource(writer_factory(*a), "writer"))

    def write(path, image):
        if "passages" in Path(path).parts and int(image[0, 0, 0]) == 20 + failed_frame * 20:
            if failure == "raise":
                raise OSError("fixture candidate write")
            if failure == "false":
                Path(path).write_bytes(b"partial image")
                return False
        return imwrite(path, image)

    monkeypatch.setattr(cv2, "imwrite", write)
    if failure == "replace":
        original_replace = Path.replace
        replacements = []

        def replace(source, target):
            replacements.append(target)
            if len(replacements) > failed_frame:
                raise OSError("fixture candidate replace")
            return original_replace(source, target)

        monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(OSError):
        track_video(video, root / "run", model_path, output_size=(64, 64),
                    model=FakeModel(lambda call: [(1, (20, 20, 80, 90), 0.9)]))
    assert sorted(released) == ["capture", "writer"]
    assert not (root / "run/summary.json").exists()
    images = list((root / "run/passages").rglob("*.jpg"))
    if failed_frame == 0:
        assert images == []
    else:
        assert images == [root / "run/passages/1/best-original.jpg"]
        assert (cv2.imread(str(images[0])) == 20).all(), "Failed replacement preserves prior image"
