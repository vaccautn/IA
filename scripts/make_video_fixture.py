"""Generate an explicitly synthetic video from the repository's public fixture."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Crear un clip sintético; no es video de campo"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/fixtures/synthetic-cow.mp4"
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("La salida ya existe; usar otro nombre")
    import cv2
    import numpy as np

    image = cv2.imread(str(ROOT / "fixtures/cow_female_black_white.jpg"))
    if image is None:
        parser.error("No se pudo leer la imagen de prueba del repositorio")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), 10, (960, 640)
    )
    if not writer.isOpened():
        writer.release()
        parser.error("Codec MP4 no disponible")
    try:
        for i in range(80):
            canvas = np.full((640, 960, 3), (28, 38, 32), dtype=np.uint8)
            if 10 <= i < 50:
                scale = min(900 / image.shape[1], 530 / image.shape[0])
                resized = cv2.resize(
                    image, (int(image.shape[1] * scale), int(image.shape[0] * scale))
                )
                left = 20 + i % 20
                top = 70
                canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = (
                    resized
                )
            elif i >= 50:
                scale = min(440 / image.shape[1], 510 / image.shape[0])
                resized = cv2.resize(
                    image, (int(image.shape[1] * scale), int(image.shape[0] * scale))
                )
                for left in (15, 500):
                    canvas[
                        120 : 120 + resized.shape[0], left : left + resized.shape[1]
                    ] = resized
            cv2.putText(
                canvas,
                "SYNTHETIC TEST | still photo, not field footage",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (220, 245, 225),
                1,
            )
            writer.write(canvas)
    finally:
        writer.release()
    print(f"Clip sintético (8 s): {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
