"""YOLO inference engine for VACCA cow detection.

Loads the fine-tuned model once at startup and exposes a simple
detect(image_bytes) -> list[Detection] interface.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from .schemas import BoundingBox, Detection

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models" / "deploy" / "vacca-yolo26n-v1.pt"


class VACCADetector:
    """Thin wrapper around Ultralytics YOLO for cow detection."""

    def __init__(self, model_path: str | Path | None = None, conf: float = 0.25) -> None:
        from ultralytics import YOLO

        self._model_path = Path(model_path or DEFAULT_MODEL)
        self._conf = conf
        self._model = YOLO(str(self._model_path))

    @property
    def model_path(self) -> str:
        return str(self._model_path)

    @property
    def gpu_available(self) -> bool:
        import torch
        return torch.cuda.is_available()

    def detect(self, image_bytes: bytes) -> tuple[List[Detection], int, int, float]:
        """Run cow detection on an image.

        Args:
            image_bytes: Raw image bytes (JPEG/PNG).

        Returns:
            Tuple of (detections, image_width, image_height, inference_time_ms).
        """
        image = Image.open(io.BytesIO(image_bytes))
        img_w, img_h = image.size

        # Convert to RGB if needed (e.g. RGBA PNG)
        if image.mode != "RGB":
            image = image.convert("RGB")

        t0 = time.perf_counter()
        results = self._model(image, conf=self._conf, verbose=False)
        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000.0

        detections: list[Detection] = []
        if results and len(results) > 0:
            r = results[0]
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf_val = float(box.conf[0].item())
                    xywh = box.xywh[0].tolist()  # [x_center, y_center, w, h] in pixels

                    # Convert pixel coords to normalized [0,1]
                    x_c = xywh[0] / img_w
                    y_c = xywh[1] / img_h
                    w = xywh[2] / img_w
                    h = xywh[3] / img_h

                    # Pixel bbox for convenience
                    x1 = int(xywh[0] - xywh[2] / 2)
                    y1 = int(xywh[1] - xywh[3] / 2)
                    x2 = int(xywh[0] + xywh[2] / 2)
                    y2 = int(xywh[1] + xywh[3] / 2)

                    detections.append(Detection(
                        class_name="cow",
                        confidence=round(conf_val, 4),
                        bbox=BoundingBox(x_center=x_c, y_center=y_c, width=w, height=h),
                        x1=x1, y1=y1, x2=x2, y2=y2,
                    ))

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)

        return detections, img_w, img_h, inference_ms


# Singleton — loaded once at import time
_detector: Optional[VACCADetector] = None


def get_detector(model_path: str | Path | None = None, conf: float = 0.25) -> VACCADetector:
    """Get or create the singleton detector instance."""
    global _detector
    if _detector is None:
        _detector = VACCADetector(model_path=model_path, conf=conf)
    return _detector
