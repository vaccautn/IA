"""Reuse the image detector without modifying its API or loading BCS."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path


class ImageDetectorAdapter:
    def __init__(self, model_path: Path, confidence: float = 0.25) -> None:
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("La confianza debe estar entre 0 y 1")
        model_path = Path(model_path)
        if not model_path.is_file():
            raise ValueError(
                "No se encontró el modelo local; no se descarga automáticamente"
            )
        from vacca_api.detection import VACCADetector

        with model_path.open("rb") as model_file:
            digest = hashlib.file_digest(model_file, "sha256").hexdigest()
        self.metadata = {
            "name": model_path.name,
            "sha256": digest,
            "confidence": confidence,
        }
        self.detector = VACCADetector(model_path=model_path, conf=confidence)

    def detect(self, image) -> dict:
        import cv2

        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise ValueError("No se pudo codificar el frame para el detector")
        detections, width, height, elapsed = self.detector.detect(encoded.tobytes())
        return {
            "cow_detected": bool(detections),
            "detection_count": len(detections),
            "detections": [item.model_dump() for item in detections],
            "image_width": width,
            "image_height": height,
            "inference_time_ms": round(elapsed, 3),
        }
