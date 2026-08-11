"""Fine-tune YOLO sobre un dataset de detección de bovinos.

Uso base:
    .venv/Scripts/python scripts/train.py --config configs/training_navid.yaml

Con GPU:
    .venv/Scripts/python scripts/train.py --config configs/training_navid.yaml --device cuda:0

Reanudar desde un checkpoint:
    .venv/Scripts/python scripts/train.py --config configs/training_navid.yaml --resume outputs/training/navid-finetune/weights/last.pt

El script asume que el dataset ya fue descomprimido y que el
data.yaml del dataset apunta correctamente a train/valid/test.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]


def _resolve_data_yaml(raw: str) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    return (ROOT / candidate).resolve()


def _validate_config(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("Training config must be a YAML dictionary")

    required: list[str] = ["data", "model", "training", "output"]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    data_path = _resolve_data_yaml(config["data"])
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset data.yaml not found: {data_path}")
    config["_data_path"] = data_path

    pretrained = config["model"].get("pretrained")
    if pretrained:
        pretrained_path = (ROOT / pretrained).resolve()
        if pretrained_path.is_file():
            config["_pretrained_path"] = pretrained_path
        else:
            print(
                f"[WARN] Pretrained weights not found at {pretrained_path}. "
                f"Ultralytics will download {pretrained!r} automatically."
            )
            config["_pretrained_path"] = str(pretrained)

    training = config.get("training", {})
    if "device" in training and training["device"] not in ("cpu",):
        import torch

        if training["device"] == "auto" and not torch.cuda.is_available():
            training["device"] = "cpu"
            print("[WARN] GPU no detectada, usando CPU.")

    return config


def _build_run_info(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "started_at_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "dataset": str(config["_data_path"]),
        "pretrained": str(config.get("_pretrained_path", config["model"]["pretrained"])),
        "config_file": str(config.get("_config_path", "")),
        "training": config.get("training", {}),
    }


def train(config: dict[str, Any]) -> Path:
    from ultralytics import YOLO

    training = config["training"]
    output = config["output"]

    model_path = config.get("_pretrained_path", config["model"]["pretrained"])
    model = YOLO(str(model_path))

    results = model.train(
        data=str(config["_data_path"]),
        epochs=training.get("epochs", 50),
        batch=training.get("batch", 16),
        imgsz=training.get("imgsz", 640),
        device=training.get("device", "cpu"),
        workers=training.get("workers", 2),
        optimizer=training.get("optimizer", "auto"),
        lr0=training.get("lr0", 0.001),
        lrf=training.get("lrf", 0.0001),
        momentum=training.get("momentum", 0.937),
        weight_decay=training.get("weight_decay", 0.0005),
        warmup_epochs=training.get("warmup_epochs", 3),
        warmup_momentum=training.get("warmup_momentum", 0.8),
        warmup_bias_lr=training.get("warmup_bias_lr", 0.1),
        cos_lr=training.get("cos_lr", True),
        hsv_h=training.get("hsv_h", 0.015),
        hsv_s=training.get("hsv_s", 0.7),
        hsv_v=training.get("hsv_v", 0.4),
        degrees=training.get("degrees", 10.0),
        translate=training.get("translate", 0.1),
        scale=training.get("scale", 0.5),
        shear=training.get("shear", 2.0),
        perspective=training.get("perspective", 0.0),
        flipud=training.get("flipud", 0.0),
        fliplr=training.get("fliplr", 0.5),
        mosaic=training.get("mosaic", 0.3),
        mixup=training.get("mixup", 0.1),
        patience=training.get("patience", 15),
        project=output.get("project", "outputs/training"),
        name=output.get("name", "navid-finetune"),
        exist_ok=output.get("exist_ok", True),
        save=output.get("save", True),
        save_period=output.get("save_period", 10),
        plots=True,
        verbose=True,
    )

    weights_dir = Path(results.save_dir) / "weights"
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"

    if best.is_file():
        export_path = Path(results.save_dir) / "exported"
        export_path.mkdir(exist_ok=True)
        exported = model.export(format="torchscript", imgsz=640)
        if isinstance(exported, str):
            print(f"[OK] Modelo exportado: {exported}")
        else:
            print(f"[OK] Modelo exportado a: {export_path}")

    return Path(results.save_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO para detección de bovinos (VACCA Fase 1)"
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "training_navid.yaml"),
        help="Ruta al archivo de configuración de entrenamiento",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Sobrescribe el dispositivo (cpu, cuda:0, auto, etc.)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Reanuda entrenamiento desde un checkpoint .pt",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Sobrescribe la cantidad de épocas",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config_path = (ROOT / args.config).resolve()
    if not config_path.is_file():
        print(f"[ERROR] Config file not found: {config_path}", file=sys.stderr)
        return 1

    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    try:
        config = _validate_config(config, config_path)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if args.device:
        config["training"]["device"] = args.device
    if args.epochs:
        config["training"]["epochs"] = args.epochs

    config["_config_path"] = str(config_path)

    run_info = _build_run_info(config)
    print(json.dumps(run_info, indent=2, default=str))

    if args.resume:
        from ultralytics import YOLO  # type: ignore[attr-defined]

        model = YOLO(args.resume)
        model.train(resume=True)
        return 0

    try:
        save_dir = train(config)
    except ImportError:
        print(
            "[ERROR] Ultralytics no está instalado. Ejecutá: "
            "pip install -e .[yolo]",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"[ERROR] Training failed: {exc}", file=sys.stderr)
        return 1

    best_path = save_dir / "weights" / "best.pt"
    print(f"\n[LISTO] Entrenamiento completado.")
    print(f"  Directorio: {save_dir}")
    if best_path.is_file():
        print(f"  Mejores pesos: {best_path}")

    # Guardar metadata de la corrida
    run_info["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    run_info["output_dir"] = str(save_dir)
    (save_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2, default=str), encoding="utf-8"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
