from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.process_video import main  # noqa: E402
from vacca_video.output import PresentationSink  # noqa: E402
from vacca_video.pipeline import SamplingConfig, process_frames  # noqa: E402
from vacca_video.source import OpenCVFileSource  # noqa: E402


class FakeCapture:
    def __init__(self, timestamps, declared=None):
        self.timestamps = timestamps
        self.declared = len(timestamps) if declared is None else declared
        self.position = -1
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        self.position += 1
        if self.position >= len(self.timestamps):
            return False, None
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def get(self, field):
        return {
            cv2.CAP_PROP_FPS: 10,
            cv2.CAP_PROP_FRAME_WIDTH: 64,
            cv2.CAP_PROP_FRAME_HEIGHT: 48,
            cv2.CAP_PROP_FRAME_COUNT: self.declared,
            cv2.CAP_PROP_POS_MSEC: self.timestamps[
                min(max(0, self.position), len(self.timestamps) - 1)
            ],
        }[field]

    def release(self):
        self.released = True


class FakeDetector:
    metadata = {"name": "test-detector", "confidence": 0.25, "sha256": "test"}

    def detect(self, image):
        return {
            "cow_detected": False,
            "detection_count": 0,
            "detections": [],
            "image_width": image.shape[1],
            "image_height": image.shape[0],
            "inference_time_ms": 1.0,
        }


class VideoIOTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.input = self.root / "input.mp4"
        self.input.write_bytes(b"mocked decoder")

    def test_repeated_decoder_timestamps_fall_back_monotonically(self):
        capture = FakeCapture([100, 100, float("nan"), 400])
        with patch("cv2.VideoCapture", return_value=capture):
            with OpenCVFileSource(self.input) as source:
                frames = list(source)
        self.assertEqual([f.timestamp_ms for f in frames], [0, 100, 200, 300])
        self.assertEqual(
            [f.timestamp_source for f in frames],
            ["decoder", "fps_fallback", "fps_fallback", "decoder"],
        )
        self.assertTrue(capture.released)

    def test_early_decoder_failure_is_not_end_of_file(self):
        capture = FakeCapture([0, 100], declared=100)
        with patch("cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(ValueError, "Lectura interrumpida"):
                with OpenCVFileSource(self.input) as source:
                    list(source)
        self.assertTrue(capture.released)

    def test_source_releases_on_invalid_metadata(self):
        capture = FakeCapture([0])
        with (
            patch("cv2.VideoCapture", return_value=capture),
            patch.object(capture, "get", return_value=0),
        ):
            with self.assertRaises(ValueError), OpenCVFileSource(self.input):
                pass
        self.assertTrue(capture.released)

    def test_output_is_incremental_and_preview_is_bounded(self):
        output = self.root / "results"
        capture = FakeCapture([0, 100, 200, 300])
        summary = {"processing_id": "test", "model": FakeDetector.metadata}
        with patch("cv2.VideoCapture", return_value=capture):
            with OpenCVFileSource(self.input) as source:
                summary["source"] = source.metadata
                with PresentationSink(output, source.metadata, 10, 2, False) as sink:
                    process_frames(
                        source, FakeDetector(), sink, SamplingConfig(10, None), summary
                    )
                    sink.finalize(summary)
        records = [
            json.loads(line)
            for line in (output / "detections.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(records), 4)
        self.assertEqual(len(list((output / "frames").glob("*.jpg"))), 2)
        self.assertTrue(summary["preview_truncated"])
        self.assertTrue((output / "frames.js").read_text().endswith("];\n"))
        self.assertIn(
            '<html lang="es">', (output / "report.html").read_text(encoding="utf-8")
        )
        self.assertIsNone(summary["annotated_video"])

    def test_existing_output_is_never_overwritten(self):
        output = self.root / "keep"
        output.mkdir()
        marker = output / "data.txt"
        marker.write_text("existing")
        source = {"fps": 10, "width": 64, "height": 48}
        with self.assertRaises(FileExistsError), PresentationSink(output, source, 5):
            pass
        self.assertEqual(marker.read_text(), "existing")

    def test_cli_failure_publishes_partial_report_and_releases_source(self):
        capture = FakeCapture([0, 100, 200], declared=100)
        output = self.root / "partial"
        with (
            patch("cv2.VideoCapture", return_value=capture),
            patch(
                "vacca_video.detector.ImageDetectorAdapter", return_value=FakeDetector()
            ),
        ):
            code = main([str(self.input), "--output", str(output), "--no-video"])
        self.assertEqual(code, 1)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertGreater(summary["frames_analyzed"], 0)
        self.assertTrue(capture.released)
        self.assertTrue((output / "report.html").exists())

    def test_real_codec_round_trip_and_cli(self):
        video = self.root / "real.avi"
        writer = cv2.VideoWriter(
            str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48)
        )
        self.assertTrue(
            writer.isOpened(), "MJPG is required by the locked OpenCV test environment"
        )
        try:
            for i in range(10):
                writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
        finally:
            writer.release()
        output = self.root / "roundtrip"
        with patch(
            "vacca_video.detector.ImageDetectorAdapter", return_value=FakeDetector()
        ):
            code = main(
                [
                    str(video),
                    "--output",
                    str(output),
                    "--sample-fps",
                    "5",
                    "--max-frames",
                    "0",
                ]
            )
        self.assertEqual(code, 0)
        metadata = json.loads((output / "summary.json").read_text())
        self.assertEqual(metadata["frames_analyzed"], 5)
        self.assertEqual(metadata["stop_reason"], "end_of_source")
        capture = cv2.VideoCapture(str(output / "annotated.mp4"))
        try:
            decoded = 0
            while capture.read()[0]:
                decoded += 1
        finally:
            capture.release()
        self.assertEqual(decoded, 5)


if __name__ == "__main__":
    unittest.main()
