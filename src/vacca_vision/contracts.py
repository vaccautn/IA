from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any


def _require_finite_number(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """A bounding box in absolute pixel coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        for field_name in ("x_min", "y_min", "x_max", "y_max"):
            _require_finite_number(getattr(self, field_name), field_name)
        if self.x_min < 0 or self.y_min < 0:
            raise ValueError("Bounding box minimum coordinates cannot be negative")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("Bounding box maximums must exceed their minimums")

    @property
    def area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def to_dict(self) -> dict[str, float]:
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
        }


@dataclass(frozen=True, slots=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox

    def __post_init__(self) -> None:
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, int):
            raise TypeError("class_id must be an integer")
        if self.class_id < 0:
            raise ValueError("class_id cannot be negative")
        if not self.class_name.strip():
            raise ValueError("class_name cannot be empty")
        _require_finite_number(self.confidence, "confidence")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")

    def to_dict(self, image: ImageMetadata) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox.to_dict(),
            "relative_area": self.bbox.area / image.area,
        }


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    width: int
    height: int

    def __post_init__(self) -> None:
        if isinstance(self.width, bool) or not isinstance(self.width, int):
            raise TypeError("width must be an integer")
        if isinstance(self.height, bool) or not isinstance(self.height, int):
            raise TypeError("height must be an integer")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image dimensions must be positive")

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> dict[str, int]:
        return {"width": self.width, "height": self.height}


class Status(str, Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"


class Reason(str, Enum):
    NO_BOVINE_DETECTED = "NO_BOVINE_DETECTED"
    MULTIPLE_BOVINES_DETECTED = "MULTIPLE_BOVINES_DETECTED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    BOVINE_TOO_SMALL = "BOVINE_TOO_SMALL"
    INSUFFICIENT_FRAMING = "INSUFFICIENT_FRAMING"
    INVALID_FILE = "INVALID_FILE"
    PROCESSING_ERROR = "PROCESSING_ERROR"


@dataclass(frozen=True, slots=True)
class QualityFlags:
    confidence_ok: bool | None
    size_ok: bool | None
    framing_ok: bool | None

    def to_dict(self) -> dict[str, bool | None]:
        return {
            "confidence_ok": self.confidence_ok,
            "size_ok": self.size_ok,
            "framing_ok": self.framing_ok,
        }


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    name: str
    version: str

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.version.strip():
            raise ValueError("Model name and version cannot be empty")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version}


@dataclass(frozen=True, slots=True)
class Timing:
    inference_ms: float
    total_ms: float

    def __post_init__(self) -> None:
        _require_finite_number(self.inference_ms, "inference_ms")
        _require_finite_number(self.total_ms, "total_ms")
        if self.inference_ms < 0 or self.total_ms < 0:
            raise ValueError("Timing values cannot be negative")
        if self.total_ms < self.inference_ms:
            raise ValueError("total_ms cannot be lower than inference_ms")

    def to_dict(self) -> dict[str, float]:
        return {
            "inference_ms": self.inference_ms,
            "total_ms": self.total_ms,
        }


@dataclass(frozen=True, slots=True)
class DetectionBatch:
    image: ImageMetadata
    detections: tuple[Detection, ...]
    model: ModelIdentity
    timing: Timing

    def __post_init__(self) -> None:
        object.__setattr__(self, "detections", tuple(self.detections))
        for index, detection in enumerate(self.detections):
            if (
                detection.bbox.x_max > self.image.width
                or detection.bbox.y_max > self.image.height
            ):
                raise ValueError(
                    f"Detection bounding box at index {index} exceeds image dimensions"
                )


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    status: Status
    reason: Reason | None
    animal_count: int
    image: ImageMetadata
    detections: tuple[Detection, ...]
    quality: QualityFlags
    model: ModelIdentity
    timing: Timing

    def __post_init__(self) -> None:
        object.__setattr__(self, "detections", tuple(self.detections))
        if self.animal_count < 0:
            raise ValueError("animal_count cannot be negative")
        if self.status is Status.ACCEPTED and self.reason is not None:
            raise ValueError("An accepted result cannot have a rejection reason")
        if self.status is Status.REJECTED and self.reason is None:
            raise ValueError("A rejected result must have a reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reason": self.reason.value if self.reason else None,
            "animal_count": self.animal_count,
            "model": self.model.to_dict(),
            "image": self.image.to_dict(),
            "detections": [item.to_dict(self.image) for item in self.detections],
            "quality": self.quality.to_dict(),
            "timing": self.timing.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)
