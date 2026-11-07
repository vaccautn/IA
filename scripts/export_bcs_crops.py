"""Exporta recortes por vaca para etiquetar BCS y arma el dataset por categoría."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vacca_video.bcs_export import export_crops, materialize_dataset  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="VACCA: recortes de trayectorias para etiquetar BCS")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="copiar recortes y crear labels.csv vacío")
    export.add_argument("runs", type=Path, nargs="+", help="carpetas de salida de track_video.py")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--include-review", action="store_true",
                        help="incluir también trayectorias marcadas para revisión")
    dataset = commands.add_parser("materialize", help="copiar filas etiquetadas a <salida>/<1..5>/")
    dataset.add_argument("labels", type=Path, help="labels.csv completado")
    dataset.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            result = export_crops(args.runs, args.output, args.include_review)
        else:
            result = materialize_dataset(args.labels, args.output)
    except (OSError, ValueError, KeyError) as exc:
        print(f"No se pudo completar: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
