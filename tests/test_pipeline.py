from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vacca_vision import (  # noqa: E402
    AptitudePipeline,
    BoundingBox,
    ClassificationConfig,
    ClassificationResult,
    Detection,
    DetectionBatch,
    ImageMetadata,
    ModelIdentity,
    QualityFlags,
    Reason,
    Status,
    Timing,
)


IMAGE = ImageMetadata(width=100, height=100)
MODEL = ModelIdentity(name="fake-detector", version="test-1")
TIMING = Timing(inference_ms=4.5, total_ms=5.0)


class FakeDetector:
    def __init__(
        self,
        detections: tuple[Detection, ...],
        image: ImageMetadata = IMAGE,
    ) -> None:
        self._detections = detections
        self._image = image
        self.received_source: str | None = None

    def detect(self, source: str) -> DetectionBatch:
        self.received_source = source
        return DetectionBatch(
            image=self._image,
            detections=self._detections,
            model=MODEL,
            timing=TIMING,
        )


def detection(
    *,
    class_id: int = 0,
    class_name: str = "bovine",
    confidence: float = 0.9,
    bbox: BoundingBox | None = None,
) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox or BoundingBox(10, 10, 90, 90),
    )


class AptitudePipelineTests(unittest.TestCase):
    def classify(
        self,
        *detections: Detection,
        config: ClassificationConfig | None = None,
    ) -> ClassificationResult:
        detector = FakeDetector(tuple(detections))
        result = AptitudePipeline(detector, config).classify("image-reference")
        self.assertEqual(detector.received_source, "image-reference")
        return result

    def test_accepts_one_eligible_bovine(self) -> None:
        result = self.classify(detection(class_name=" BOVINE "))

        self.assertEqual(result.status, Status.ACCEPTED)
        self.assertIsNone(result.reason)
        self.assertEqual(result.animal_count, 1)
        self.assertEqual(result.quality, QualityFlags(True, True, True))

    def test_rejects_when_no_bovine_is_detected_without_quality_checks(self) -> None:
        result = self.classify()

        self.assertEqual(result.status, Status.REJECTED)
        self.assertEqual(result.reason, Reason.NO_BOVINE_DETECTED)
        self.assertEqual(result.animal_count, 0)
        self.assertEqual(result.quality, QualityFlags(None, None, None))

    def test_rejects_when_all_bovines_are_below_confidence(self) -> None:
        result = self.classify(detection(confidence=0.49))

        self.assertEqual(result.reason, Reason.LOW_CONFIDENCE)
        self.assertEqual(result.quality, QualityFlags(False, None, None))

    def test_accepts_confidence_exactly_at_threshold(self) -> None:
        result = self.classify(
            detection(confidence=0.5),
            config=ClassificationConfig(min_confidence=0.5),
        )

        self.assertEqual(result.status, Status.ACCEPTED)

    def test_uses_only_confidence_qualified_bovines_after_mixed_predictions(self) -> None:
        result = self.classify(
            detection(class_id=10, confidence=0.49),
            detection(class_id=20, confidence=0.5),
            config=ClassificationConfig(min_confidence=0.5),
        )

        self.assertEqual(result.status, Status.ACCEPTED)
        self.assertEqual(result.animal_count, 1)
        self.assertEqual([item.class_id for item in result.detections], [20])

    def test_rejects_multiple_confident_bovines_before_geometry_rules(self) -> None:
        result = self.classify(
            detection(bbox=BoundingBox(0, 0, 10, 10)),
            detection(bbox=BoundingBox(90, 90, 100, 100)),
        )

        self.assertEqual(result.reason, Reason.MULTIPLE_BOVINES_DETECTED)
        self.assertEqual(result.animal_count, 2)
        self.assertEqual(result.quality, QualityFlags(True, None, None))

    def test_rejects_a_bovine_that_is_too_small(self) -> None:
        result = self.classify(
            detection(bbox=BoundingBox(10, 10, 30, 30)),
            config=ClassificationConfig(min_relative_area=0.1),
        )

        self.assertEqual(result.reason, Reason.BOVINE_TOO_SMALL)
        self.assertEqual(result.quality, QualityFlags(True, False, None))

    def test_accepts_area_exactly_at_threshold(self) -> None:
        result = self.classify(
            detection(bbox=BoundingBox(10, 0, 20, 100)),
            config=ClassificationConfig(
                min_relative_area=0.1,
                framing_enabled=False,
            ),
        )

        self.assertEqual(result.status, Status.ACCEPTED)
        self.assertEqual(result.quality, QualityFlags(True, True, None))

    def test_size_rejection_precedes_framing_rejection(self) -> None:
        result = self.classify(
            detection(bbox=BoundingBox(0, 0, 20, 20)),
            config=ClassificationConfig(
                min_relative_area=0.1,
                border_margin_ratio=0.05,
            ),
        )

        self.assertEqual(result.reason, Reason.BOVINE_TOO_SMALL)
        self.assertEqual(result.quality, QualityFlags(True, False, None))

    def test_rejects_a_bovine_inside_each_configured_border_margin(self) -> None:
        edge_boxes = {
            "left": BoundingBox(4, 10, 80, 90),
            "top": BoundingBox(10, 4, 90, 80),
            "right": BoundingBox(20, 10, 96, 90),
            "bottom": BoundingBox(10, 20, 90, 96),
        }

        for edge, bbox in edge_boxes.items():
            with self.subTest(edge=edge):
                result = self.classify(
                    detection(bbox=bbox),
                    config=ClassificationConfig(border_margin_ratio=0.05),
                )
                self.assertEqual(result.reason, Reason.INSUFFICIENT_FRAMING)
                self.assertEqual(result.quality, QualityFlags(True, True, False))

    def test_accepts_bbox_exactly_on_configured_margin(self) -> None:
        result = self.classify(
            detection(bbox=BoundingBox(5, 5, 95, 95)),
            config=ClassificationConfig(border_margin_ratio=0.05),
        )

        self.assertEqual(result.status, Status.ACCEPTED)
        self.assertEqual(result.quality, QualityFlags(True, True, True))

    def test_skips_framing_flag_when_validation_is_disabled(self) -> None:
        result = self.classify(
            detection(bbox=BoundingBox(0, 0, 100, 100)),
            config=ClassificationConfig(framing_enabled=False),
        )

        self.assertEqual(result.status, Status.ACCEPTED)
        self.assertEqual(result.quality, QualityFlags(True, True, None))

    def test_ignores_non_bovine_classes(self) -> None:
        result = self.classify(detection(class_id=7, class_name="horse"))

        self.assertEqual(result.reason, Reason.NO_BOVINE_DETECTED)
        self.assertEqual(result.animal_count, 0)
        self.assertEqual(result.detections, ())

    def test_serializes_complete_accepted_result_to_dict_and_json(self) -> None:
        result = self.classify(
            detection(
                class_id=3,
                confidence=0.75,
                bbox=BoundingBox(10, 20, 90, 80),
            )
        )

        expected = {
            "status": "ACCEPTED",
            "reason": None,
            "animal_count": 1,
            "model": {"name": "fake-detector", "version": "test-1"},
            "image": {"width": 100, "height": 100},
            "detections": [
                {
                    "class_id": 3,
                    "class_name": "bovine",
                    "confidence": 0.75,
                    "bbox": {
                        "x_min": 10,
                        "y_min": 20,
                        "x_max": 90,
                        "y_max": 80,
                    },
                    "relative_area": 0.48,
                }
            ],
            "quality": {
                "confidence_ok": True,
                "size_ok": True,
                "framing_ok": True,
            },
            "timing": {"inference_ms": 4.5, "total_ms": 5.0},
        }

        self.assertEqual(result.to_dict(), expected)
        self.assertEqual(json.loads(result.to_json()), expected)

    def test_serializes_complete_rejected_result(self) -> None:
        result = self.classify(detection(class_id=4, confidence=0.25))

        expected = {
            "status": "REJECTED",
            "reason": "LOW_CONFIDENCE",
            "animal_count": 0,
            "model": {"name": "fake-detector", "version": "test-1"},
            "image": {"width": 100, "height": 100},
            "detections": [
                {
                    "class_id": 4,
                    "class_name": "bovine",
                    "confidence": 0.25,
                    "bbox": {
                        "x_min": 10,
                        "y_min": 10,
                        "x_max": 90,
                        "y_max": 90,
                    },
                    "relative_area": 0.64,
                }
            ],
            "quality": {
                "confidence_ok": False,
                "size_ok": None,
                "framing_ok": None,
            },
            "timing": {"inference_ms": 4.5, "total_ms": 5.0},
        }

        self.assertEqual(result.to_dict(), expected)
        self.assertEqual(json.loads(result.to_json()), expected)

    def test_repeated_classification_has_deterministic_output(self) -> None:
        detector = FakeDetector((detection(class_id=5),))
        pipeline = AptitudePipeline(detector)

        first = pipeline.classify("same-image").to_json()
        second = pipeline.classify("same-image").to_json()

        self.assertEqual(first, second)


class ContractValidationTests(unittest.TestCase):
    def test_rejects_invalid_bounding_box_geometry(self) -> None:
        invalid_coordinates = (
            (10, 0, 10, 20),
            (20, 0, 10, 20),
            (0, 10, 20, 10),
            (0, 20, 20, 10),
            (-1, 0, 20, 20),
        )

        for coordinates in invalid_coordinates:
            with self.subTest(coordinates=coordinates):
                with self.assertRaises(ValueError):
                    BoundingBox(*coordinates)

    def test_rejects_bounding_boxes_outside_image_dimensions(self) -> None:
        outside_boxes = (
            BoundingBox(0, 0, 101, 100),
            BoundingBox(0, 0, 100, 101),
        )

        for bbox in outside_boxes:
            with self.subTest(bbox=bbox):
                with self.assertRaisesRegex(
                    ValueError,
                    "Detection bounding box at index 0 exceeds image dimensions",
                ):
                    FakeDetector((detection(bbox=bbox),)).detect("image-reference")

    def test_accepts_bbox_maximums_equal_to_image_dimensions(self) -> None:
        batch = FakeDetector(
            (detection(bbox=BoundingBox(0, 0, 100, 100)),)
        ).detect("image-reference")

        self.assertEqual(batch.detections[0].bbox.area, IMAGE.area)

    def test_rejects_invalid_class_ids(self) -> None:
        for class_id in (-1, True):
            with self.subTest(class_id=class_id):
                with self.assertRaises((TypeError, ValueError)):
                    detection(class_id=class_id)

    def test_rejects_bool_for_all_numeric_configuration_fields(self) -> None:
        for field_name in (
            "min_confidence",
            "min_relative_area",
            "border_margin_ratio",
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(TypeError, f"{field_name} must be a number"):
                    ClassificationConfig(**{field_name: True})

    def test_rejects_invalid_numeric_configuration_values(self) -> None:
        invalid_values = {
            "min_confidence": (-0.01, 1.01, math.nan),
            "min_relative_area": (-0.01, 1.01, math.inf),
            "border_margin_ratio": (-0.01, 0.5, math.nan),
        }

        for field_name, values in invalid_values.items():
            for value in values:
                with self.subTest(field_name=field_name, value=value):
                    with self.assertRaises(ValueError):
                        ClassificationConfig(**{field_name: value})

    def test_rejects_non_boolean_framing_toggle(self) -> None:
        with self.assertRaisesRegex(TypeError, "framing_enabled must be a boolean"):
            ClassificationConfig(framing_enabled=1)


if __name__ == "__main__":
    unittest.main()
