"""Train and validate the VACCA ordinal BCS ResNet18 model."""
from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path
from typing import Any

_CUBLAS_WORKSPACE_CONFIG = "CUBLAS_WORKSPACE_CONFIG"
_ACCEPTED_CUBLAS_WORKSPACE_CONFIGS = {":4096:8", ":16:8"}

def _configure_cublas_determinism() -> None:
    configured = os.environ.get(_CUBLAS_WORKSPACE_CONFIG)
    if configured is None:
        os.environ[_CUBLAS_WORKSPACE_CONFIG] = ":4096:8"
    elif configured not in _ACCEPTED_CUBLAS_WORKSPACE_CONFIGS:
        accepted = ", ".join(sorted(_ACCEPTED_CUBLAS_WORKSPACE_CONFIGS))
        raise RuntimeError(
            f"Invalid {_CUBLAS_WORKSPACE_CONFIG}={configured!r}; "
            f"unset it or set one of: {accepted}"
        )


_configure_cublas_determinism()

import torch
import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def _resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (ROOT / path).resolve()
def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Training config must be a YAML dictionary")
    required = {
        "data_dir",
        "output",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "optimizer",
        "patience",
        "num_workers",
        "imgsz",
        "device",
        "seed",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")
    schedule = str(config.get("lr_schedule", "cosine")).lower()
    if schedule != "cosine":
        raise ValueError(
            f"Unsupported lr_schedule: {config.get('lr_schedule')!r}. "
            "Only 'cosine' is supported."
        )
    config["lr_schedule"] = schedule
    config["_config_path"] = str(config_path.resolve())
    _validate_training_config(config)
    return config


def _validate_training_config(config: dict[str, Any]) -> None:
    """Reject invalid training values before any output directory is touched."""
    warmup_epochs = config.get("warmup_epochs", 2)
    integer_minimums = {
        "epochs": 1,
        "batch_size": 1,
        "patience": 1,
        "warmup_epochs": 0,
        "num_workers": 0,
        "imgsz": 1,
        "seed": 0,
    }
    for key, minimum in integer_minimums.items():
        value = warmup_epochs if key == "warmup_epochs" else config.get(key)
        if type(value) is not int or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")

    if warmup_epochs > config["epochs"]:
        raise ValueError("warmup_epochs cannot exceed epochs")

    for key, minimum in (("lr", 0.0), ("weight_decay", 0.0)):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a finite number >= {minimum}")
        numeric_value = float(value)
        if not math.isfinite(numeric_value) or numeric_value < minimum:
            raise ValueError(f"{key} must be a finite number >= {minimum}")
    if config["lr"] == 0:
        raise ValueError("lr must be a finite number > 0")
    if str(config["optimizer"]).lower() != "adamw":
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def set_seed(seed: int) -> None:
    """Seed supported generators and require deterministic Torch algorithms."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
