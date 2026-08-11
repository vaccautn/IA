from __future__ import annotations

import importlib
import hashlib
import hmac
import math
import os
import stat
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

from .contracts import (
    BoundingBox,
    Detection,
    DetectionBatch,
    ImageMetadata,
    ModelIdentity,
    Timing,
)
from .image_validation import ValidatedImage


class UltralyticsDependencyError(RuntimeError):
    """Raised when the optional Ultralytics package is unavailable."""


class UltralyticsAdapterError(RuntimeError):
    """A stable public error for model initialization, inference, or output mapping."""


ModelFactory = Callable[[str], object]
Clock = Callable[[], float]


class UltralyticsDetector:
    """Adapt an injected or lazily loaded Ultralytics YOLO model to domain contracts."""

    def __init__(
        self,
        *,
        weights_path: str | Path,
        expected_weights_sha256: str | None = None,
        device: str,
        model_identity: ModelIdentity,
        model: object | None = None,
        model_factory: ModelFactory | None = None,
        prediction_confidence: float = 0.25,
        input_size: int = 640,
        clock: Clock = perf_counter,
    ) -> None:
        if model is not None and model_factory is not None:
            raise ValueError("Provide either model or model_factory, not both")
        if not str(weights_path).strip():
            raise ValueError("weights_path cannot be empty")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device cannot be empty")
        if (
            isinstance(prediction_confidence, bool)
            or not isinstance(prediction_confidence, (int, float))
            or not math.isfinite(prediction_confidence)
            or not 0 <= prediction_confidence <= 1
        ):
            raise ValueError("prediction_confidence must be between 0 and 1")
        if isinstance(input_size, bool) or not isinstance(input_size, int):
            raise TypeError("input_size must be an integer")
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if model is None:
            if (
                not isinstance(expected_weights_sha256, str)
                or len(expected_weights_sha256) != 64
                or any(
                    character not in "0123456789abcdefABCDEF"
                    for character in expected_weights_sha256
                )
            ):
                raise ValueError(
                    "expected_weights_sha256 must contain exactly "
                    "64 hexadecimal characters"
                )
        self._weights_path = Path(weights_path)
        self._expected_weights_sha256 = (
            expected_weights_sha256.casefold()
            if expected_weights_sha256 is not None
            else None
        )
        self._device = device
        self._model_identity = model_identity
        self._model = model
        self._model_factory = model_factory
        self._prediction_confidence = float(prediction_confidence)
        self._input_size = input_size
        self._clock = clock

    def detect(self, source: ValidatedImage) -> DetectionBatch:
        if not isinstance(source, ValidatedImage):
            raise TypeError("source must be a ValidatedImage")
        model = self._get_model()
        with source._inference_image() as inference_image:
            started_at = self._clock()
            try:
                results = model.predict(
                    source=inference_image,
                    device=self._device,
                    conf=self._prediction_confidence,
                    imgsz=self._input_size,
                    verbose=False,
                )
                predict_finished_at = self._clock()
            except Exception:
                raise UltralyticsAdapterError("Ultralytics inference failed") from None
        measured_inference_ms = (predict_finished_at - started_at) * 1000

        try:
            result = self._single_result(results)
            image = self._image_metadata(result)
            if image != source.metadata:
                raise ValueError("Result dimensions differ from validated image")
            detections = self._detections(result, model)
            inference_ms = self._inference_ms(result, measured_inference_ms)
            validated_batch = DetectionBatch(
                image=image,
                detections=detections,
                model=self._model_identity,
                timing=Timing(inference_ms=inference_ms, total_ms=inference_ms),
            )
            total_ms = max(
                (self._clock() - started_at) * 1000,
                inference_ms,
            )
            return DetectionBatch(
                image=validated_batch.image,
                detections=validated_batch.detections,
                model=validated_batch.model,
                timing=Timing(inference_ms=inference_ms, total_ms=total_ms),
            )
        except UltralyticsAdapterError:
            raise
        except Exception:
            raise UltralyticsAdapterError(
                "Ultralytics returned malformed detection results"
            ) from None

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        with self._verified_weights_snapshot() as snapshot_path:
            factory = self._model_factory
            if factory is None:
                try:
                    module = importlib.import_module("ultralytics")
                    factory = module.YOLO
                except (ImportError, AttributeError):
                    raise UltralyticsDependencyError(
                        "Ultralytics is not installed. Install the optional YOLO "
                        "adapter with: pip install -e .[yolo]"
                    ) from None

            try:
                self._model = factory(str(snapshot_path))
            except Exception:
                raise UltralyticsAdapterError(
                    "Ultralytics model initialization failed"
                ) from None
        return self._model

    @contextmanager
    def _verified_weights_snapshot(self) -> Iterator[Path]:
        if not self._weights_path.is_file():
            raise UltralyticsAdapterError(
                f"Local weights file does not exist: {self._weights_path}"
            )
        suffix = self._weights_path.suffix or ".weights"
        with tempfile.TemporaryDirectory(prefix="vacca-weights-") as directory:
            snapshot_path = Path(directory) / f"verified{suffix}"
            digest = hashlib.sha256()
            try:
                with (
                    self._weights_path.open("rb") as weights_file,
                    snapshot_path.open("xb") as snapshot_file,
                ):
                    if not stat.S_ISREG(os.fstat(weights_file.fileno()).st_mode):
                        raise OSError("Weights source is not a regular file")
                    snapshot_path.chmod(0o600)
                    for chunk in iter(
                        lambda: weights_file.read(1024 * 1024),
                        b"",
                    ):
                        digest.update(chunk)
                        snapshot_file.write(chunk)
            except OSError:
                raise UltralyticsAdapterError(
                    "Local weights file cannot be snapshotted"
                ) from None
            if not hmac.compare_digest(
                digest.hexdigest(),
                self._expected_weights_sha256 or "",
            ):
                raise UltralyticsAdapterError(
                    "Local weights SHA-256 does not match the expected digest"
                )
            yield snapshot_path

    @staticmethod
    def _single_result(results: object) -> Any:
        if isinstance(results, (str, bytes)):
            raise ValueError("Results must be a sequence")
        values = list(results)
        if len(values) != 1:
            raise ValueError("Expected exactly one result")
        return values[0]

    @staticmethod
    def _image_metadata(result: Any) -> ImageMetadata:
        shape = _as_sequence(result.orig_shape)
        if len(shape) < 2:
            raise ValueError("orig_shape must contain height and width")
        height = _strict_positive_int(shape[0])
        width = _strict_positive_int(shape[1])
        return ImageMetadata(width=width, height=height)

    @staticmethod
    def _detections(result: Any, model: Any) -> tuple[Detection, ...]:
        boxes = result.boxes
        if boxes is None:
            return ()
        coordinates = _as_sequence(boxes.xyxy)
        confidences = _as_sequence(boxes.conf)
        class_ids = _as_sequence(boxes.cls)
        if not (len(coordinates) == len(confidences) == len(class_ids)):
            raise ValueError("Box fields have different lengths")

        names = getattr(result, "names", None)
        if names is None:
            names = getattr(model, "names", None)

        detections: list[Detection] = []
        for coordinate_row, confidence, raw_class_id in zip(
            coordinates,
            confidences,
            class_ids,
            strict=True,
        ):
            row = _as_sequence(coordinate_row)
            if len(row) != 4:
                raise ValueError("xyxy row must contain four coordinates")
            class_id = _class_id(raw_class_id)
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=_class_name(names, class_id),
                    confidence=_finite_float(confidence),
                    bbox=BoundingBox(*(_finite_float(value) for value in row)),
                )
            )
        return tuple(detections)

    @staticmethod
    def _inference_ms(result: Any, measured_inference_ms: float) -> float:
        speed = getattr(result, "speed", None)
        if isinstance(speed, Mapping) and "inference" in speed:
            try:
                inference_ms = _finite_float(speed["inference"])
            except ValueError:
                pass
            else:
                if inference_ms >= 0:
                    return inference_ms
        return measured_inference_ms


def _as_sequence(value: object) -> Sequence[Any]:
    converted: Any = value
    for method_name in ("detach", "cpu", "numpy", "tolist"):
        method = getattr(converted, method_name, None)
        if callable(method):
            converted = method()
    if isinstance(converted, (str, bytes)) or not isinstance(converted, Sequence):
        raise ValueError("Expected an array-like sequence")
    return converted


def _finite_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a numeric value")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("Expected a finite numeric value")
    return converted


def _strict_positive_int(value: object) -> int:
    converted = _finite_float(value)
    if not converted.is_integer() or converted <= 0:
        raise ValueError("Expected a positive integer")
    return int(converted)


def _class_id(value: object) -> int:
    converted = _finite_float(value)
    if not converted.is_integer() or converted < 0:
        raise ValueError("Class id must be a non-negative integer")
    return int(converted)


def _class_name(names: object, class_id: int) -> str:
    if isinstance(names, Mapping):
        name = names.get(class_id, names.get(str(class_id)))
    elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        name = names[class_id] if class_id < len(names) else None
    else:
        name = None
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"No class name exists for class id {class_id}")
    return name
