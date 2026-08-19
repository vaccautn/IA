from __future__ import annotations

from typing import Protocol, TypeVar

from .contracts import DetectionBatch


DetectorInput = TypeVar("DetectorInput", contravariant=True)


class Detector(Protocol[DetectorInput]):
    """Framework-neutral boundary implemented by future model adapters."""

    def detect(self, source: DetectorInput) -> DetectionBatch:
        """Return normalized detections and execution metadata for a source."""
        ...
