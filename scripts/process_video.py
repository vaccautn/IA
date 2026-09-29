"""Process a local video with VACCA; produce a portable presentation and JSONL."""

from __future__ import annotations

import argparse
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vacca_video.pipeline import SamplingConfig, process_frames  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="VACCA: detección en video grabado")
    result.add_argument("input", type=Path, help="Video local; no webcam ni URL")
    result.add_argument("--output", type=Path, help="Carpeta nueva para los resultados")
    result.add_argument("--sample-fps", type=float, default=5.0)
    result.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Frames analizados; 0 = todo el video",
    )
    result.add_argument("--confidence", type=float, default=0.25)
    result.add_argument(
        "--preview-limit",
        type=int,
        default=120,
        help="Imágenes del HTML, entre 1 y 1000",
    )
    result.add_argument(
        "--no-video", action="store_true", help="Generar solo HTML/JSONL"
    )
    result.add_argument(
        "--model", type=Path, default=ROOT / "models/deploy/vacca-yolo26n-v1.pt"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config = SamplingConfig(
            args.sample_fps, None if args.max_frames == 0 else args.max_frames
        )
        if not 1 <= args.preview_limit <= 1000:
            raise ValueError("preview-limit debe estar entre 1 y 1000")
        if not args.input.is_file():
            raise ValueError("El video de entrada no existe")
        run_id = uuid4().hex
        output = args.output or ROOT / "outputs/video" / run_id
        if output.exists():
            raise ValueError(
                "La carpeta de salida ya existe; elegir otra para no sobrescribir"
            )
        # Create parents before importing model libraries: their fallback may
        # otherwise write settings into the checkout or the global user profile.
        for variable, folder in (
            ("YOLO_CONFIG_DIR", "ultralytics"),
            ("MPLCONFIGDIR", "matplotlib"),
        ):
            if variable not in os.environ:
                cache = ROOT / "outputs/runtime" / folder
                cache.mkdir(parents=True, exist_ok=True)
                os.environ[variable] = str(cache)
        os.environ.setdefault("YOLO_OFFLINE", "true")
        from vacca_video.detector import ImageDetectorAdapter
        from vacca_video.output import PresentationSink
        from vacca_video.source import OpenCVFileSource

        with OpenCVFileSource(args.input) as source:
            print("Cargando detector local (BCS no se utiliza)...", flush=True)
            started = perf_counter()
            detector = ImageDetectorAdapter(args.model, args.confidence)
            summary = {
                "schema_version": "vacca-video-summary-v1",
                "processing_id": run_id,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "source": source.metadata,
                "model": detector.metadata,
                "model_load_seconds": round(perf_counter() - started, 3),
                "python": platform.python_version(),
                "platform": platform.system(),
            }

            def progress(stats):
                n = stats["frames_analyzed"]
                if n == 1 or n % 25 == 0:
                    print(
                        f"Frames analizados: {n}; tiempo de video: {stats['last_timestamp_ms'] / 1000:.1f}s",
                        flush=True,
                    )

            with PresentationSink(
                output,
                source.metadata,
                config.sample_fps,
                args.preview_limit,
                not args.no_video,
            ) as sink:
                try:
                    process_frames(source, detector, sink, config, summary, progress)
                finally:
                    sink.finalize(summary)
                    print(
                        f"Presentación: {(output / 'report.html').resolve()}",
                        flush=True,
                    )
            print(
                f"Completado: {summary['frames_analyzed']} frames; {summary['processing_fps']} FPS de procesamiento."
            )
        return 0
    except KeyboardInterrupt:
        print("Cancelado por el usuario.", file=sys.stderr)
        return 130
    except ImportError as exc:
        print(
            f"Faltan dependencias: {exc}. Consultar docs/video-prototype.md.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"No se pudo completar el video: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
