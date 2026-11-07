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
import re
import sys
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
_DATASET_KEYS = frozenset(
    {
        "path", "train", "val", "validation", "test", "minival", "nc", "names",
        "channels", "roboflow", "license", "url", "version", "description",
    }
)
_ROBOFLOW_KEYS = frozenset({"workspace", "project", "version", "license", "url"})
_REMOTE_PATH_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def _resolve_data_yaml(raw: str) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    return (ROOT / candidate).resolve()


def _local_dataset_path(raw: object, *, base: Path, field: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Dataset {field} must be a non-empty local path")
    if _REMOTE_PATH_RE.match(raw) or raw.startswith(("\\\\", "//")):
        raise ValueError(f"Dataset {field} must be local; remote paths are unsupported")
    candidate = Path(raw)
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def _dataset_root(descriptor: dict[str, Any], data_path: Path) -> Path:
    raw_root = descriptor.get("path")
    if raw_root is None or raw_root == "":
        root = data_path.parent.resolve()
    else:
        if not isinstance(raw_root, str):
            raise ValueError("Dataset path must be a local path string")
        root = _local_dataset_path(raw_root, base=Path.cwd(), field="path")
    if not root.is_dir():
        raise FileNotFoundError("Dataset root directory does not exist locally")
    return root


def _validate_image_manifest(manifest: Path, *, split: str) -> None:
    try:
        contents = manifest.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise ValueError(f"Dataset {split} image manifest is not readable UTF-8") from None
    entries = [line for line in contents.strip().splitlines() if line]
    if not entries:
        raise ValueError(f"Dataset {split} image manifest is empty")
    for entry in entries:
        if _REMOTE_PATH_RE.match(entry) or entry.startswith(("\\\\", "//")):
            raise ValueError(f"Dataset {split} manifest contains a non-local path")
        image_path = Path(entry)
        if not image_path.is_absolute() and entry.startswith("./"):
            image_path = manifest.parent / entry[2:]
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Dataset {split} manifest references a missing image")


def _validate_split(value: object, *, split: str, root: Path) -> None:
    entries = value if type(value) is list else [value]
    if not entries:
        raise ValueError(f"Dataset {split} must name at least one local directory or manifest")
    for entry in entries:
        path = _local_dataset_path(entry, base=root, field=split)
        if path.suffix.casefold() == ".txt":
            if not path.is_file():
                raise FileNotFoundError(f"Dataset {split} manifest does not exist")
            _validate_image_manifest(path, split=split)
        elif not path.is_dir():
            raise FileNotFoundError(f"Dataset {split} directory does not exist locally")


def _dataset_names_count(value: object) -> int:
    if type(value) is list:
        if not value or any(not isinstance(name, str) or not name.strip() for name in value):
            raise ValueError("Dataset names must be a non-empty list of strings")
        return len(value)
    if type(value) is dict and value:
        indices: list[int] = []
        for key, name in value.items():
            if type(key) is int and key >= 0:
                indices.append(key)
            elif isinstance(key, str) and key.isdecimal():
                indices.append(int(key))
            else:
                raise ValueError("Dataset names keys must be non-negative class indices")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Dataset names values must be non-empty strings")
        if sorted(indices) != list(range(len(indices))):
            raise ValueError("Dataset names class indices must be contiguous from zero")
        return len(indices)
    raise ValueError("Dataset names must be a non-empty list or class-index mapping")


def _validate_dataset_descriptor(data_path: Path) -> Path:
    try:
        descriptor = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        raise ValueError("Dataset data.yaml must be readable safe YAML") from None
    if type(descriptor) is not dict:
        raise ValueError("Dataset data.yaml must contain a YAML dictionary")
    if any(isinstance(key, str) and key.casefold() == "download" for key in descriptor):
        raise ValueError("Dataset download hooks are not supported; prepare data locally")
    unknown = {key for key in descriptor if not isinstance(key, str) or key not in _DATASET_KEYS}
    if unknown:
        raise ValueError(f"Unsupported dataset descriptor keys: {', '.join(sorted(map(str, unknown)))}")

    roboflow = descriptor.get("roboflow")
    if roboflow is not None:
        if type(roboflow) is not dict:
            raise ValueError("Dataset roboflow metadata must be a plain mapping")
        for key, value in roboflow.items():
            if not isinstance(key, str) or key.casefold() == "download":
                raise ValueError("Nested dataset hooks are not supported")
            if key not in _ROBOFLOW_KEYS or type(value) not in (str, int, float):
                raise ValueError("Unsupported nested dataset metadata")

    root = _dataset_root(descriptor, data_path)
    if "train" not in descriptor:
        raise ValueError("Dataset train split is required")
    if "val" in descriptor and "validation" in descriptor:
        raise ValueError("Specify only one of dataset val or validation")
    if "val" not in descriptor and "validation" not in descriptor:
        raise ValueError("Dataset val split is required")
    _validate_split(descriptor["train"], split="train", root=root)
    val = descriptor["val"] if "val" in descriptor else descriptor["validation"]
    _validate_split(val, split="val", root=root)
    for split in ("test", "minival"):
        if split in descriptor and descriptor[split] is not None:
            _validate_split(descriptor[split], split=split, root=root)

    names = descriptor.get("names")
    class_count = _dataset_names_count(names) if names is not None else None
    nc = None
    if "nc" in descriptor:
        nc = descriptor["nc"]
        if type(nc) is not int or nc < 1:
            raise ValueError("Dataset nc must be a positive integer")
    if class_count is None and nc is None:
        raise ValueError("Dataset names or nc must define the class count")
    if nc is not None and class_count is not None and nc != class_count:
        raise ValueError("Dataset nc does not match the number of names")
    channels = descriptor.get("channels")
    if channels is not None and (type(channels) is not int or channels < 1):
        raise ValueError("Dataset channels must be a positive integer")
    for key in ("license", "url", "version", "description"):
        if key in descriptor and not isinstance(descriptor[key], str):
            raise ValueError(f"Dataset {key} metadata must be a string")
    return root


def _require_detection_task(task: object, *, context: str) -> None:
    if task != "detect":
        raise ValueError(
            f"Only YOLO detection training is supported; {context} task must be 'detect'"
        )


def _dataset_validated_trainer():
    from ultralytics.models.yolo.detect import DetectionTrainer

    class DatasetValidatedDetectionTrainer(DetectionTrainer):
        def get_dataset(self):
            _require_detection_task(
                getattr(self.args, "task", None), context="effective trainer"
            )
            effective_data = self.args.data
            if not isinstance(effective_data, (str, Path)):
                raise ValueError("Effective dataset must be a local data.yaml path")
            _validate_dataset_descriptor(Path(effective_data))
            return super().get_dataset()

    return DatasetValidatedDetectionTrainer


def _validate_config(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("Training config must be a YAML dictionary")

    required: list[str] = ["data", "model", "training", "output"]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    for key in ("model", "training", "output"):
        if type(config[key]) is not dict:
            raise ValueError(f"Training config {key} must be a YAML dictionary")

    if not isinstance(config["data"], str) or not config["data"].strip():
        raise ValueError("Training config data must be a local descriptor path")
    data_path = _resolve_data_yaml(config["data"])
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset data.yaml not found: {data_path}")
    _validate_dataset_descriptor(data_path)
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
    data_path = config.get("_data_path")
    if not isinstance(data_path, Path):
        raise ValueError("A validated local dataset descriptor path is required")
    _validate_dataset_descriptor(data_path)
    from ultralytics import YOLO

    training = config["training"]
    output = config["output"]

    model_path = config.get("_pretrained_path", config["model"]["pretrained"])
    model = YOLO(str(model_path))
    _require_detection_task(getattr(model, "task", None), context="loaded model")
    trainer_class = _dataset_validated_trainer()

    results = model.train(
        data=str(config["_data_path"]),
        trainer=trainer_class,
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
        try:
            from ultralytics import YOLO  # type: ignore[attr-defined]

            model = YOLO(args.resume)
            _require_detection_task(getattr(model, "task", None), context="resume checkpoint")
            trainer_class = _dataset_validated_trainer()
            model.train(resume=True, trainer=trainer_class)
        except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
            print(f"[ERROR] Training failed: {exc}", file=sys.stderr)
            return 1
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
    print("\n[LISTO] Entrenamiento completado.")
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
