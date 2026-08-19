from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Generic, TypeVar

from .contracts import (
    ClassificationResult,
    Detection,
    DetectionBatch,
    QualityFlags,
    Reason,
    Status,
)
from .detector import Detector
from .rules import bovine_detections, has_sufficient_framing, relative_area


PipelineInput = TypeVar("PipelineInput")


@dataclass(frozen=True, slots=True)
class ClassificationConfig:
    min_confidence: float = 0.5
    min_relative_area: float = 0.1
    border_margin_ratio: float = 0.02
    framing_enabled: bool = True

    def __post_init__(self) -> None:
        _require_ratio(self.min_confidence, "min_confidence")
        _require_ratio(self.min_relative_area, "min_relative_area")
        _require_number(self.border_margin_ratio, "border_margin_ratio")
        if not 0 <= self.border_margin_ratio < 0.5:
            raise ValueError("border_margin_ratio must be between 0 and 0.5")
        if not isinstance(self.framing_enabled, bool):
            raise TypeError("framing_enabled must be a boolean")


def _require_number(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")


def _require_ratio(value: float, field_name: str) -> None:
    _require_number(value, field_name)
    if not 0 <= value <= 1:
        raise ValueError(f"{field_name} must be between 0 and 1")


class AptitudePipeline(Generic[PipelineInput]):
    """Classify image aptitude with deterministic, first-match precedence."""

    def __init__(
        self,
        detector: Detector[PipelineInput],
        config: ClassificationConfig | None = None,
    ) -> None:
        self._detector = detector
        self._config = config or ClassificationConfig()

    def classify(self, source: PipelineInput) -> ClassificationResult:
        batch = self._detector.detect(source)
        bovines = bovine_detections(batch.detections)

        if not bovines:
            return self._result(
                batch=batch,
                detections=(),
                animal_count=0,
                reason=Reason.NO_BOVINE_DETECTED,
                quality=QualityFlags(None, None, None),
            )

        eligible = tuple(
            item
            for item in bovines
            if item.confidence >= self._config.min_confidence
        )
        if not eligible:
            return self._result(
                batch=batch,
                detections=bovines,
                animal_count=0,
                reason=Reason.LOW_CONFIDENCE,
                quality=QualityFlags(False, None, None),
            )

        if len(eligible) >= 2:
            return self._result(
                batch=batch,
                detections=eligible,
                animal_count=len(eligible),
                reason=Reason.MULTIPLE_BOVINES_DETECTED,
                quality=QualityFlags(True, None, None),
            )

        selected = eligible[0]
        area_ratio = relative_area(selected.bbox, batch.image)
        if area_ratio < self._config.min_relative_area:
            return self._result(
                batch=batch,
                detections=eligible,
                animal_count=1,
                reason=Reason.BOVINE_TOO_SMALL,
                quality=QualityFlags(True, False, None),
            )

        if self._config.framing_enabled:
            framing_ok = has_sufficient_framing(
                selected.bbox,
                batch.image,
                self._config.border_margin_ratio,
            )
            if not framing_ok:
                return self._result(
                    batch=batch,
                    detections=eligible,
                    animal_count=1,
                    reason=Reason.INSUFFICIENT_FRAMING,
                    quality=QualityFlags(True, True, False),
                )
        else:
            framing_ok = None

        return self._result(
            batch=batch,
            detections=eligible,
            animal_count=1,
            reason=None,
            quality=QualityFlags(True, True, framing_ok),
        )

    @staticmethod
    def _result(
        *,
        batch: DetectionBatch,
        detections: tuple[Detection, ...],
        animal_count: int,
        reason: Reason | None,
        quality: QualityFlags,
    ) -> ClassificationResult:
        return ClassificationResult(
            status=Status.REJECTED if reason else Status.ACCEPTED,
            reason=reason,
            animal_count=animal_count,
            image=batch.image,
            detections=detections,
            quality=quality,
            model=batch.model,
            timing=batch.timing,
        )
