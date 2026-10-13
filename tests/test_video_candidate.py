"""Candidate selection uses fake detections and original decoded pixels only."""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vacca_video.candidate import CandidateSelector
from vacca_video.output import PresentationSink
from vacca_video.pipeline import Frame, SamplingConfig, process_frames


def detection(confidence=0.8, count=1, label="cow"):
    cow = dict(class_name=label, confidence=confidence,
               bbox=dict(x_center=0.5, y_center=0.5, width=0.5, height=0.5),
               x1=10, y1=8, x2=30, y2=24)
    return dict(cow_detected=count > 0, detection_count=count,
                detections=[cow.copy() for _ in range(count)],
                image_width=65, image_height=49, inference_time_ms=1.0)


@pytest.mark.parametrize("counts", [[0], [2, 3], [0, 2, 0]])
def test_no_eligible_candidate(counts):
    selector = CandidateSelector()
    for index, count in enumerate(counts):
        selector.consider(Frame(index, index * 200, None), detection(count=count), index)
    assert selector.metadata is None
    assert selector.image is None


@pytest.mark.parametrize("change", [dict(detection_count=2), dict(cow_detected=False),
                                   dict(detections=[dict(class_name="horse", confidence=0.99)]),
                                   dict(detections=[dict(class_name="cow", confidence=float("nan"))])])
def test_inconsistent_detection_is_not_eligible(change):
    selector = CandidateSelector()
    selector.consider(Frame(0, 0, None), detection() | change, 0)
    assert selector.metadata is None


def test_later_higher_and_first_tie_keep_independent_full_frame():
    selector = CandidateSelector()
    image = np.full((49, 65, 3), 17, np.uint8)
    for index, confidence in enumerate([0.7, 0.9, 0.9, 0.8]):
        selector.consider(Frame(index * 4, index * 200, image, "fps_fallback"),
                          detection(confidence), index)
    image[:] = 200
    assert selector.metadata["frame_index"] == 4
    assert selector.metadata["sample_ordinal"] == 1
    assert selector.metadata["timestamp_source"] == "fps_fallback"
    assert selector.metadata["detector_confidence"] == 0.9
    assert selector.metadata["original_bbox"]["x1"] == 10
    assert np.all(selector.image == 17)


def test_final_best_after_preview_limit_and_unsampled_frame_ignored(tmp_path):
    summary = dict(processing_id="fake", model=dict(name="fake", sha256="hash", confidence=0.25))
    image = np.full((49, 65, 3), (3, 37, 119), np.uint8)
    calls = iter([detection(0.6), detection(0.7), detection(0.95)])
    detector = SimpleNamespace(detect=lambda pixels: next(calls))
    frames = [Frame(i, t, image) for i, t in enumerate([0, 10, 200, 400])]
    output = tmp_path / "result"
    with PresentationSink(output, dict(fps=10, width=65, height=49), 5, 1, False) as sink:
        process_frames(frames, detector, sink, SamplingConfig(5, 3), summary)
        sink.finalize(summary)
    candidate = summary["candidate"]
    assert candidate["frame_index"] == 3
    assert candidate["sample_ordinal"] == 2
    assert candidate["timestamp_ms"] == 400
    assert candidate["detector_model"] == summary["model"]
    assert candidate["encoding"] == "PNG"
    assert (candidate["image_width"], candidate["image_height"]) == (65, 49)
    assert candidate["artifact"] == "candidate.png"
    assert np.array_equal(cv2.imread(str(output / "candidate.png")), image)
    assert not np.array_equal(cv2.imread(str(output / "frames/000000.jpg")), image)
    assert summary["bcs"] == dict(status="not_evaluated", reason="web_api_only")
    assert json.loads((output / "summary.json").read_text())["candidate"] == candidate


def test_no_eligible_report_is_explicit_and_has_no_image(tmp_path):
    summary = dict(processing_id="fake", model=dict(name="fake"))
    image = np.zeros((49, 65, 3), np.uint8)
    with PresentationSink(tmp_path / "result", dict(fps=5, width=65, height=49), 5, 1, False) as sink:
        process_frames([Frame(0, 0, image)], SimpleNamespace(detect=lambda _: detection(count=2)),
                       sink, SamplingConfig(), summary)
        sink.finalize(summary)
    assert summary["candidate"] is None
    assert summary["bcs"]["reason"] == "no_eligible_frame"
    assert not (tmp_path / "result/candidate.png").exists()


def test_best_can_be_after_the_120_frame_cli_preview(tmp_path):
    image = np.full((49, 65, 3), 23, np.uint8)
    summary = dict(processing_id="fake", model=dict(name="fake"))
    results = iter([detection(0.6)] * 120 + [detection(0.99)])
    with PresentationSink(tmp_path / "result", dict(fps=5, width=65, height=49), 5, 120, False) as sink:
        process_frames((Frame(i * 3, i * 200, image) for i in range(121)),
                       SimpleNamespace(detect=lambda _: next(results)), sink,
                       SamplingConfig(5, None), summary)
        sink.finalize(summary)
    assert summary["stop_reason"] == "end_of_source"
    assert summary["candidate"]["sample_ordinal"] == 120
    assert summary["candidate"]["frame_index"] == 360
    assert summary["preview_frames"] == 120
    assert (tmp_path / "result/candidate.png").is_file()


def test_only_eligible_single_cow_is_selected_even_if_multiple_have_higher_confidence():
    image = np.zeros((49, 65, 3), np.uint8)
    selector = CandidateSelector()
    for i, result in enumerate([detection(count=0), detection(0.6), detection(0.99, count=2)]):
        selector.consider(Frame(i, i * 200, image), result, i)
    assert selector.metadata["frame_index"] == 1
    assert selector.metadata["detector_confidence"] == 0.6


def test_candidate_write_failure_does_not_mask_existing_video_error(tmp_path, monkeypatch):
    summary = dict(status="failed", error="original detector failure", stop_reason="error",
                   frames_analyzed=1, model=dict(name="fake"))
    image = np.zeros((49, 65, 3), np.uint8)
    def write(*args):
        raise cv2.error("candidate encoding failure")
    with PresentationSink(tmp_path / "result", dict(fps=5, width=65, height=49), 5, 1, False) as sink:
        sink.candidate.consider(Frame(0, 0, image), detection(), 0)
        monkeypatch.setattr(cv2, "imwrite", write)
        sink.finalize(summary)
    assert summary["status"] == "failed"
    assert summary["error"] == "original detector failure"
    assert summary["stop_reason"] == "error"
    assert summary["candidate"] is None
    assert summary["bcs"] == dict(status="not_evaluated", reason="video_not_completed")


def test_required_preview_write_error_is_still_a_video_failure(tmp_path, monkeypatch):
    summary = dict(processing_id="fake", model=dict(name="fake"))
    image = np.zeros((49, 65, 3), np.uint8)
    original_write = cv2.imwrite
    monkeypatch.setattr(cv2, "imwrite", lambda path, pixels:
                        original_write(path, pixels) if Path(path).name == "candidate.png" else False)
    with PresentationSink(tmp_path / "result", dict(fps=5, width=65, height=49), 5, 1, False) as sink:
        with pytest.raises(OSError, match="frame de presentación"):
            process_frames([Frame(0, 0, image)], SimpleNamespace(detect=lambda _: detection()),
                           sink, SamplingConfig(), summary)
        sink.finalize(summary)
    assert summary["status"] == "failed"
    assert "frame de presentación" in summary["error"]
