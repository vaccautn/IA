from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vacca_video.pipeline import Frame, SamplingConfig, process_frames  # noqa: E402


class Detector:
    def __init__(self, count=1):
        self.calls = 0
        self.count = count

    def detect(self, image):
        self.calls += 1
        return {
            "detection_count": self.count,
            "detections": [{}] * self.count,
            "inference_time_ms": 2.0,
        }


class Sink:
    def __init__(self):
        self.results = []

    def write(self, frame, result):
        self.results.append(result)


class VideoPipelineTests(unittest.TestCase):
    def setUp(self):
        self.summary = {"processing_id": "test", "model": {"name": "fake"}}
        self.sink = Sink()
        self.detector = Detector()

    def run_frames(self, times, config=None):
        frames = (Frame(i, t, None) for i, t in enumerate(times))
        process_frames(
            frames, self.detector, self.sink, config or SamplingConfig(), self.summary
        )

    def test_sampling_uses_timestamps_not_frame_modulo(self):
        self.run_frames([0, 80, 250, 310, 410, 700], SamplingConfig(5, None))
        self.assertEqual([r["frame_index"] for r in self.sink.results], [0, 2, 4, 5])
        self.assertEqual(
            [r["timestamp_ms"] for r in self.sink.results], [0, 250, 410, 700]
        )
        self.assertEqual(self.detector.calls, 4)

    def test_high_requested_fps_does_not_duplicate_source_frames(self):
        self.run_frames([0, 100, 200], SamplingConfig(120, None))
        self.assertEqual(len(self.sink.results), 3)

    def test_frame_limit_stops_consuming_source_and_reports_partial_scope(self):
        def frames():
            yield Frame(0, 0, None)
            yield Frame(1, 200, None)
            self.fail("Source was read beyond frame limit")

        process_frames(
            frames(), self.detector, self.sink, SamplingConfig(5, 2), self.summary
        )
        self.assertEqual(self.summary["stop_reason"], "frame_limit")
        self.assertEqual(self.summary["frames_read"], 2)

    def test_zero_detections_is_success(self):
        self.detector.count = 0
        self.run_frames([0, 200])
        self.assertEqual(self.summary["status"], "completed")
        self.assertEqual(self.summary["frames_with_cows"], 0)

    def test_multiple_cows_are_kept_and_not_summed_as_unique_animals(self):
        self.detector.count = 3
        self.run_frames([0, 200, 400])
        self.assertEqual(self.summary["max_cows_in_frame"], 3)
        self.assertEqual(self.summary["frames_with_cows"], 3)
        self.assertNotIn("unique_animals", self.summary)
        self.assertEqual(self.sink.results[0]["model_version"], "fake")
        self.assertEqual(self.sink.results[0]["processing_id"], "test")

    def test_empty_video_fails(self):
        with self.assertRaisesRegex(ValueError, "frames decodificables"):
            self.run_frames([])
        self.assertEqual(self.summary["status"], "failed")

    def test_bad_timestamp_does_not_produce_success(self):
        for times in ([0, 0], [0, -1], [float("nan")], [0, float("inf")]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                self.run_frames(times)
            self.assertEqual(self.summary["status"], "failed")

    def test_detector_failure_keeps_partial_statistics(self):
        def frames():
            yield Frame(0, 0, None)
            raise RuntimeError("decoder stopped")

        with self.assertRaises(RuntimeError):
            process_frames(
                frames(), self.detector, self.sink, SamplingConfig(), self.summary
            )
        self.assertEqual(self.summary["frames_analyzed"], 1)
        self.assertEqual(self.summary["status"], "failed")

    def test_interrupt_is_not_completion(self):
        def frames():
            yield Frame(0, 0, None)
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            process_frames(
                frames(), self.detector, self.sink, SamplingConfig(), self.summary
            )
        self.assertEqual(self.summary["status"], "cancelled")
        self.assertEqual(self.summary["frames_analyzed"], 1)

    def test_sink_failure_does_not_count_unwritten_result(self):
        class BrokenSink:
            def write(self, frame, result):
                raise OSError("disk full")

        with self.assertRaises(OSError):
            process_frames(
                [Frame(0, 0, None)],
                self.detector,
                BrokenSink(),
                SamplingConfig(),
                self.summary,
            )
        self.assertEqual(self.summary["frames_analyzed"], 0)
        self.assertEqual(self.summary["status"], "failed")

    def test_bad_config_rejected_before_processing(self):
        for fps in (0, -1, 121, float("nan"), float("inf")):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                SamplingConfig(fps)
        for limit in (0, -1, 1.5, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                SamplingConfig(max_frames=limit)


if __name__ == "__main__":
    unittest.main()
