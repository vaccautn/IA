"""Exporta un clip y la mejor foto por trayectoria temporal de vaca."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vacca_video.tracking import (  # noqa: E402
    CLIP_CODECS,
    DEFAULT_MAX_GAP_SECONDS,
    DEFAULT_SEGMENT_LIMIT,
    load_sam_segmenter,
    track_video,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="VACCA: tracking y recorte individual por vaca")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracker", choices=("bytetrack.yaml", "botsort.yaml"), default="bytetrack.yaml")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--sample-fps", type=float, default=0.0, help="0 procesa todos los frames")
    parser.add_argument("--max-gap-seconds", type=float, default=DEFAULT_MAX_GAP_SECONDS,
                        help="ausencia máxima antes de abrir un segmento nuevo y pedir revisión")
    parser.add_argument("--clip-format", choices=tuple(CLIP_CODECS), default="mp4",
                        help="webm (VP8) se reproduce en el navegador; mp4 usa mp4v")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="deja de leer el video al alcanzar este tiempo")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="deja de leer el video tras esta cantidad de frames")
    parser.add_argument("--segment-model", type=Path, default=None,
                        help="pesos MobileSAM/SAM locales para la silueta del mejor frame")
    parser.add_argument("--segment-limit", type=int, default=DEFAULT_SEGMENT_LIMIT,
                        help="máximo de trayectorias segmentadas, las más largas primero")
    parser.add_argument("--model", type=Path, default=ROOT / "models/deploy/vacca-yolo26n-v1.pt")
    args = parser.parse_args(argv)
    if args.sample_fps < 0:
        parser.error("--sample-fps no puede ser negativo")
    if args.segment_limit < 0:
        parser.error("--segment-limit no puede ser negativo")
    try:
        if args.output.exists():
            raise ValueError(f"La carpeta de salida ya existe: {args.output}")
        segmenter = load_sam_segmenter(args.segment_model) if args.segment_model else None
        summary = track_video(args.input, args.output, args.model, args.tracker,
                              args.confidence, args.sample_fps or None,
                              max_gap_seconds=args.max_gap_seconds, clip_format=args.clip_format,
                              max_seconds=args.max_seconds, max_frames=args.max_frames,
                              segmenter=segmenter, segment_limit=args.segment_limit)
    except ImportError as exc:
        print(f"Faltan dependencias: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"No se pudo procesar el video: {exc}", file=sys.stderr)
        return 1
    review = sum(track["status"] == "review" for track in summary["tracks"])
    print(f"Completado: {len(summary['tracks'])} trayectorias ({review} para revisión); salida: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
