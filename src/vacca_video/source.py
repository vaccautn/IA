"""OpenCV file input. No network/camera side effects on import."""

from __future__ import annotations

import math
from pathlib import Path

from .pipeline import Frame


class OpenCVFileSource:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.capture = None
        self.metadata: dict = {}

    def __enter__(self):
        import cv2

        if not self.path.is_file():
            raise ValueError("El video de entrada no existe o no es un archivo")
        self.capture = cv2.VideoCapture(str(self.path))
        try:
            if not self.capture.isOpened():
                raise ValueError("No se pudo abrir el video; comprobar formato/codec")
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if not all(math.isfinite(v) and v > 0 for v in (fps, width, height)):
                raise ValueError("El video no declara FPS y dimensiones válidos")
            count = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
            self.metadata = {
                "name": self.path.name,
                "fps": fps,
                "width": int(width),
                "height": int(height),
                "declared_frames": int(count)
                if math.isfinite(count) and count > 0
                else None,
            }
            return self
        except BaseException:
            self.capture.release()
            raise

    def __exit__(self, *args) -> None:
        if self.capture is not None:
            self.capture.release()

    def __iter__(self):
        import cv2

        if self.capture is None:
            raise RuntimeError("La fuente debe abrirse con un contexto")
        index, previous, origin = 0, -1.0, None
        while True:
            ok, image = self.capture.read()
            if not ok:
                expected = self.metadata["declared_frames"]
                # Frame counts can be approximate; catch substantial early stops.
                if expected and index + max(2, math.ceil(expected * 0.01)) < expected:
                    raise ValueError(
                        f"Lectura interrumpida: {index} de {expected} frames declarados"
                    )
                break
            raw = self.capture.get(cv2.CAP_PROP_POS_MSEC)
            if origin is None and math.isfinite(raw) and raw >= 0:
                origin = raw
            timestamp = raw - (origin or 0)
            clock = "decoder"
            if not math.isfinite(timestamp) or timestamp < 0 or timestamp <= previous:
                timestamp = max(
                    index * 1000 / self.metadata["fps"],
                    previous + 1000 / self.metadata["fps"],
                )
                clock = "fps_fallback"
            height, width = image.shape[:2]
            if (width, height) != (self.metadata["width"], self.metadata["height"]):
                raise ValueError("Las dimensiones cambiaron durante la decodificación")
            yield Frame(index, timestamp, image, clock)
            previous = timestamp
            index += 1
