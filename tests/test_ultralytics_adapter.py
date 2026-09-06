from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vacca_vision import (  # noqa: E402
    AptitudePipeline,
    ClassificationConfig,
    ModelIdentity,
    Status,
    UltralyticsAdapterError,
    UltralyticsDependencyError,
    UltralyticsDetector,
    validate_image,
)


IDENTITY = ModelIdentity(name="bovine-yolo", version="experiment-1")


class FakeTensor:
    def __init__(self, values: object) -> None:
        self._values = values

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> FakeTensor:
        return self

    def tolist(self) -> object:
        return self._values


class FakeBoxes:
    def __init__(
        self,
        xyxy: list[list[float]],
        confidence: list[float],
        class_ids: list[float],
    ) -> None:
        self.xyxy = FakeTensor(xyxy)
        self.conf = FakeTensor(confidence)
        self.cls = FakeTensor(class_ids)


class FakeResult:
    def __init__(
        self,
        boxes: FakeBoxes | None,
        *,
        names: object = None,
        orig_shape: object = (100, 200),
        speed: object = None,
    ) -> None:
        self.boxes = boxes
        self.names = names
        self.orig_shape = orig_shape
        self.speed = speed if speed is not None else {"inference": 12.5}


class FakeModel:
    def __init__(self, results: object, *, names: object = None) -> None:
        self._results = results
        self.names = names
        self.calls: list[tuple[object, str, dict[str, object]]] = []
        self.received_pixels: list[tuple[int, int, int]] = []

    def predict(self, *, source: object, device: str, **options: object) -> object:
        self.calls.append((source, device, options))
        if hasattr(source, "getpixel"):
            self.received_pixels.append(source.getpixel((0, 0)))
        if isinstance(self._results, Exception):
            raise self._results
        return self._results


def clock() -> object:
    values = iter((1.0, 1.02, 1.05))
    return lambda: next(values)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class UltralyticsDetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._image_directory = tempfile.TemporaryDirectory()
        cls.image_path = Path(cls._image_directory.name) / "validated-image.png"
        Image.new("RGB", (200, 100), color=(10, 20, 30)).save(
            cls.image_path,
            format="PNG",
        )
        cls.source = validate_image(cls.image_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._image_directory.cleanup()

    def detector(self, model: FakeModel) -> UltralyticsDetector:
        return UltralyticsDetector(
            weights_path="local-weights.pt",
            device="cpu",
            model_identity=IDENTITY,
            model=model,
            clock=clock(),
        )

    def test_maps_one_box_and_all_metadata(self) -> None:
        model = FakeModel(
            [
                FakeResult(
                    FakeBoxes([[10.0, 20.0, 150.0, 90.0]], [0.875], [2.0]),
                    names={2: "bovine"},
                )
            ]
        )

        batch = self.detector(model).detect(self.source)

        self.assertEqual(len(model.calls), 1)
        self.assertFalse(isinstance(model.calls[0][0], (str, Path)))
        self.assertEqual(model.calls[0][1], "cpu")
        self.assertEqual(
            model.calls[0][2],
            {"conf": 0.25, "imgsz": 640, "verbose": False},
        )
        with self.assertRaises(ValueError):
            model.calls[0][0].getpixel((0, 0))
        self.assertEqual(batch.image, self.source.metadata)
        self.assertEqual(batch.model, IDENTITY)
        self.assertEqual(batch.timing.inference_ms, 12.5)
        self.assertAlmostEqual(batch.timing.total_ms, 50.0)
        self.assertGreater(batch.timing.total_ms, batch.timing.inference_ms)
        self.assertEqual(len(batch.detections), 1)
        detected = batch.detections[0]
        self.assertEqual(detected.class_id, 2)
        self.assertEqual(detected.class_name, "bovine")
        self.assertEqual(detected.confidence, 0.875)
        self.assertEqual(detected.bbox.to_dict(), {
            "x_min": 10.0,
            "y_min": 20.0,
            "x_max": 150.0,
            "y_max": 90.0,
        })

    def test_maps_multiple_boxes_using_model_name_fallback(self) -> None:
        model = FakeModel(
            [
                FakeResult(
                    FakeBoxes(
                        [[1.0, 2.0, 30.0, 40.0], [50.0, 10.0, 190.0, 95.0]],
                        [0.9, 0.8],
                        [0.0, 1.0],
                    ),
                    names=None,
                )
            ],
            names=["bovine", "horse"],
        )

        batch = self.detector(model).detect(self.source)

        self.assertEqual(
            [(item.class_id, item.class_name) for item in batch.detections],
            [(0, "bovine"), (1, "horse")],
        )

    def test_maps_empty_boxes_to_empty_detection_batch(self) -> None:
        model = FakeModel(
            [FakeResult(FakeBoxes([], [], []), names={0: "bovine"})]
        )

        batch = self.detector(model).detect(self.source)

        self.assertEqual(batch.detections, ())

    def test_matching_digest_allows_lazy_factory_once(self) -> None:
        model = FakeModel(
            [FakeResult(FakeBoxes([], [], []), names={0: "bovine"})]
        )
        factory_calls: list[str] = []

        def factory(weights_path: str) -> FakeModel:
            factory_calls.append(weights_path)
            return model

        times = iter((1.0, 1.005, 1.01, 2.0, 2.005, 2.01))
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "local-weights.pt"
            weights.write_bytes(b"controlled-test-weights")
            detector = UltralyticsDetector(
                weights_path=weights,
                expected_weights_sha256=file_sha256(weights),
                device="cpu",
                model_identity=IDENTITY,
                model_factory=factory,
                clock=lambda: next(times),
            )

            detector.detect(self.source)
            detector.detect(self.source)

        self.assertEqual(len(factory_calls), 1)
        snapshot_path = Path(factory_calls[0])
        self.assertNotEqual(snapshot_path, weights)
        self.assertEqual(snapshot_path.suffix, weights.suffix)
        self.assertFalse(snapshot_path.exists())
        self.assertEqual(len(model.calls), 2)

    def test_rejects_invalid_expected_digest_before_loading(self) -> None:
        for digest in (None, "not-a-sha256", "g" * 64):
            with self.subTest(digest=digest):
                with self.assertRaisesRegex(
                    ValueError,
                    "expected_weights_sha256 must contain exactly "
                    "64 hexadecimal characters",
                ):
                    UltralyticsDetector(
                        weights_path="weights.pt",
                        expected_weights_sha256=digest,
                        device="cpu",
                        model_identity=IDENTITY,
                        model_factory=lambda path: object(),
                    )

    def test_refuses_missing_local_weights_without_calling_factory(self) -> None:
        factory_calls: list[str] = []
        detector = UltralyticsDetector(
            weights_path="missing-local-weights.pt",
            expected_weights_sha256="0" * 64,
            device="cpu",
            model_identity=IDENTITY,
            model_factory=lambda path: factory_calls.append(path),
        )

        with self.assertRaisesRegex(
            UltralyticsAdapterError,
            "Local weights file does not exist",
        ):
            detector.detect(self.source)
        self.assertEqual(factory_calls, [])

    def test_rejects_digest_mismatch_without_calling_factory(self) -> None:
        factory_calls: list[str] = []
        snapshots_before = set(
            Path(tempfile.gettempdir()).glob("vacca-weights-*")
        )
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "weights.pt"
            weights.write_bytes(b"controlled-test-weights")
            detector = UltralyticsDetector(
                weights_path=weights,
                expected_weights_sha256="0" * 64,
                device="cpu",
                model_identity=IDENTITY,
                model_factory=lambda path: factory_calls.append(path),
            )

            with self.assertRaisesRegex(
                UltralyticsAdapterError,
                "Local weights SHA-256 does not match the expected digest",
            ):
                detector.detect(self.source)

        self.assertEqual(factory_calls, [])
        self.assertEqual(
            set(Path(tempfile.gettempdir()).glob("vacca-weights-*")),
            snapshots_before,
        )

    def test_wraps_model_factory_failure_without_internal_details(self) -> None:
        def failing_factory(weights_path: str) -> object:
            raise RuntimeError(f"secret initialization detail for {weights_path}")

        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "weights.pt"
            weights.write_bytes(b"controlled-test-weights")
            detector = UltralyticsDetector(
                weights_path=weights,
                expected_weights_sha256=file_sha256(weights),
                device="cpu",
                model_identity=IDENTITY,
                model_factory=failing_factory,
            )

            with self.assertRaisesRegex(
                UltralyticsAdapterError,
                "Ultralytics model initialization failed",
            ) as context:
                detector.detect(self.source)

        self.assertNotIn("secret", str(context.exception))

    def test_wraps_model_failure_without_exposing_raw_message(self) -> None:
        model = FakeModel(RuntimeError("secret framework traceback detail"))

        with self.assertRaisesRegex(
            UltralyticsAdapterError,
            "Ultralytics inference failed",
        ) as context:
            self.detector(model).detect(self.source)

        self.assertNotIn("secret", str(context.exception))

    def test_uses_measured_predict_time_when_framework_timing_is_missing(self) -> None:
        model = FakeModel(
            [FakeResult(FakeBoxes([], [], []), names={}, speed={})]
        )

        batch = self.detector(model).detect(self.source)

        self.assertAlmostEqual(batch.timing.inference_ms, 20.0)
        self.assertAlmostEqual(batch.timing.total_ms, 50.0)

    def test_rejects_malformed_results_with_stable_public_error(self) -> None:
        malformed_results = (
            [],
            [FakeResult(FakeBoxes([[1.0, 2.0, 3.0, 4.0]], [], [0.0]), names={0: "bovine"})],
            [FakeResult(FakeBoxes([[1.0, 2.0, 3.0]], [0.9], [0.0]), names={0: "bovine"})],
            [FakeResult(FakeBoxes([[1.0, 2.0, 3.0, 4.0]], [0.9], [0.5]), names={0: "bovine"})],
            [FakeResult(FakeBoxes([[1.0, 2.0, 3.0, 4.0]], [0.9], [3.0]), names={0: "bovine"})],
            [FakeResult(FakeBoxes([], [], []), names={}, orig_shape=(100,))],
        )

        for results in malformed_results:
            with self.subTest(results=results):
                with self.assertRaisesRegex(
                    UltralyticsAdapterError,
                    "Ultralytics returned malformed detection results",
                ):
                    self.detector(FakeModel(results)).detect(self.source)

    def test_rejects_result_dimensions_that_differ_from_validated_image(self) -> None:
        model = FakeModel(
            [
                FakeResult(
                    FakeBoxes([], [], []),
                    names={},
                    orig_shape=(99, 200),
                )
            ]
        )

        with self.assertRaisesRegex(
            UltralyticsAdapterError,
            "Ultralytics returned malformed detection results",
        ):
            self.detector(model).detect(self.source)

    def test_adapter_cow_detection_is_accepted_without_renaming(self) -> None:
        model = FakeModel(
            [
                FakeResult(
                    FakeBoxes([[10.0, 10.0, 190.0, 90.0]], [0.9], [0.0]),
                    names={0: " cow "},
                )
            ]
        )
        detector = self.detector(model)
        pipeline = AptitudePipeline(
            detector,
            ClassificationConfig(framing_enabled=False),
        )

        result = pipeline.classify(self.source)

        self.assertEqual(result.status, Status.ACCEPTED)
        self.assertEqual(result.detections[0].class_name, " cow ")
        self.assertEqual(result.to_dict()["detections"][0]["class_name"], " cow ")

    def test_reports_useful_lazy_dependency_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "weights.pt"
            weights.touch()
            detector = UltralyticsDetector(
                weights_path=weights,
                expected_weights_sha256=file_sha256(weights),
                device="cpu",
                model_identity=IDENTITY,
            )

            with patch(
                "vacca_vision.ultralytics_adapter.importlib.import_module",
                side_effect=ImportError,
            ):
                with self.assertRaisesRegex(
                    UltralyticsDependencyError,
                    r"Install the optional YOLO adapter with: pip install -e \.\[yolo\]",
                ):
                    detector.detect(self.source)

    def test_factory_loads_verified_snapshot_after_original_is_replaced(self) -> None:
        loaded_bytes: list[bytes] = []
        snapshot_paths: list[Path] = []
        model = FakeModel(
            [FakeResult(FakeBoxes([], [], []), names={0: "bovine"})]
        )
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "weights.pt"
            verified_bytes = b"verified-weight-bytes"
            weights.write_bytes(verified_bytes)

            def factory(snapshot_path: str) -> FakeModel:
                snapshot = Path(snapshot_path)
                snapshot_paths.append(snapshot)
                weights.write_bytes(b"replaced-original")
                loaded_bytes.append(snapshot.read_bytes())
                return model

            detector = UltralyticsDetector(
                weights_path=weights,
                expected_weights_sha256=file_sha256(weights),
                device="cpu",
                model_identity=IDENTITY,
                model_factory=factory,
                clock=clock(),
            )
            detector.detect(self.source)

        self.assertEqual(loaded_bytes, [verified_bytes])
        self.assertNotEqual(snapshot_paths[0], weights)
        self.assertFalse(snapshot_paths[0].exists())

    def test_snapshot_is_cleaned_when_factory_fails(self) -> None:
        snapshot_paths: list[Path] = []
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "weights.pt"
            weights.write_bytes(b"verified-weight-bytes")

            def factory(snapshot_path: str) -> object:
                snapshot = Path(snapshot_path)
                snapshot_paths.append(snapshot)
                self.assertTrue(snapshot.exists())
                raise RuntimeError("internal failure")

            detector = UltralyticsDetector(
                weights_path=weights,
                expected_weights_sha256=file_sha256(weights),
                device="cpu",
                model_identity=IDENTITY,
                model_factory=factory,
            )
            with self.assertRaises(UltralyticsAdapterError):
                detector.detect(self.source)

        self.assertEqual(len(snapshot_paths), 1)
        self.assertFalse(snapshot_paths[0].exists())

    def test_inference_uses_validated_pixels_after_original_image_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.png"
            Image.new("RGB", (200, 100), color=(1, 2, 3)).save(path, "PNG")
            source = validate_image(path)
            Image.new("RGB", (200, 100), color=(200, 201, 202)).save(path, "PNG")
            model = FakeModel(
                [FakeResult(FakeBoxes([], [], []), names={0: "bovine"})]
            )

            self.detector(model).detect(source)

        self.assertEqual(model.received_pixels, [(1, 2, 3)])


if __name__ == "__main__":
    unittest.main()
