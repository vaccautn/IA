"""Train and validate the VACCA ordinal BCS ResNet18 model."""
from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path
from typing import Any
import hashlib
import json
import platform

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
import torchvision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vacca_bcs.constants import (
    CLASS_NAMES,
    IMAGE_EXTENSIONS,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    SCORE_STEP,
    SPLITS,
)

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
def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Training provenance contains a non-JSON value: {exc}") from exc


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_identity(device: torch.device) -> dict[str, Any]:
    """Return stable runtime facts that can affect deterministic resume behavior."""
    identity: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "cuda_runtime": None,
        "cudnn": None,
        "cuda_device_count": 0,
        "gpu_names": [],
    }
    if device.type != "cuda":
        return identity

    device_count = torch.cuda.device_count()
    identity.update(
        {
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_device_count": device_count,
            "gpu_names": [str(torch.cuda.get_device_name(index)) for index in range(device_count)],
        }
    )
    return identity


def _manifest_relative_path(raw: Any, *, field: str, manifest_path: Path) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"Dataset manifest {field} must be a non-empty relative path")
    path = Path(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(
            f"Dataset manifest {field} has an unsafe path {raw!r}: "
            f"{manifest_path}"
        )
    return path


def _dataset_manifest_provenance(data_dir: Path) -> dict[str, Any]:
    manifest_path = data_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Cannot establish dataset provenance: manifest not found at "
            f"{manifest_path}. Rebuild the dataset before starting or resuming training."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read dataset manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Dataset manifest must be a JSON object: {manifest_path}")
    schema_version = manifest.get("manifest_schema_version")
    if type(schema_version) is not int or schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported dataset manifest schema {schema_version!r} at {manifest_path}; "
            f"expected {MANIFEST_SCHEMA_VERSION}."
        )
    if manifest.get("class_values") != list(CLASS_NAMES):
        raise ValueError(f"Dataset manifest class_values do not match BCS classes: {manifest_path}")
    if manifest.get("class_mapping") != {
        class_name: index for index, class_name in enumerate(CLASS_NAMES)
    }:
        raise ValueError(f"Dataset manifest class_mapping does not match BCS classes: {manifest_path}")
    selected_files = manifest.get("selected_files")
    if not isinstance(selected_files, list):
        raise ValueError(f"Dataset manifest selected_files must be a list: {manifest_path}")

    data_root = data_dir.resolve()
    actual_files: dict[str, Path] = {}
    actual_counts = {split: {name: 0 for name in CLASS_NAMES} for split in SPLITS}
    for split in SPLITS:
        for class_name in CLASS_NAMES:
            class_dir = data_root / split / class_name
            if not class_dir.is_dir():
                raise ValueError(f"Dataset class directory is missing: {class_dir}")
            for path in sorted(class_dir.iterdir()):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                try:
                    path.resolve().relative_to(data_root)
                except ValueError:
                    raise ValueError(
                        f"Live dataset file escapes dataset root: {path}"
                    ) from None
                relative = path.relative_to(data_root).as_posix()
                key = relative.casefold()
                if key in actual_files:
                    raise ValueError(f"Dataset contains duplicate live path: {relative}")
                actual_files[key] = path
                actual_counts[split][class_name] += 1

    declared_files: dict[str, tuple[Path, str]] = {}
    for index, entry in enumerate(selected_files):
        if not isinstance(entry, dict):
            raise ValueError(f"Dataset manifest selected_files[{index}] must be an object")
        try:
            destination = _manifest_relative_path(
                entry["destination"], field=f"selected_files[{index}].destination", manifest_path=manifest_path
            )
            _manifest_relative_path(
                entry["source"], field=f"selected_files[{index}].source", manifest_path=manifest_path
            )
            split = entry["split"]
            declared_hash = entry["sha256"]
        except KeyError as exc:
            raise ValueError(
                f"Dataset manifest selected_files[{index}] is missing {exc.args[0]!r}"
            ) from None
        if not isinstance(split, str) or split not in SPLITS:
            raise ValueError(f"Dataset manifest selected_files[{index}] has invalid split {split!r}")
        parts = destination.parts
        if len(parts) != 3 or parts[0] != split or parts[1] not in CLASS_NAMES:
            raise ValueError(
                f"Dataset manifest selected_files[{index}] destination {destination.as_posix()!r} "
                "does not identify a valid split/class file"
            )
        if destination.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(
                f"Dataset manifest selected_files[{index}] destination is not an image: "
                f"{destination.as_posix()}"
            )
        if (
            not isinstance(declared_hash, str)
            or len(declared_hash) != 64
            or declared_hash.lower() != declared_hash
            or any(character not in "0123456789abcdef" for character in declared_hash)
        ):
            raise ValueError(
                f"Dataset manifest selected_files[{index}] has invalid sha256 for "
                f"{destination.as_posix()}"
            )
        key = destination.as_posix().casefold()
        if key in declared_files:
            raise ValueError(f"Dataset manifest has duplicate destination: {destination.as_posix()}")
        declared_files[key] = (destination, declared_hash)

    missing = sorted(set(actual_files).difference(declared_files))
    added = sorted(set(declared_files).difference(actual_files))
    if missing or added:
        details = []
        if missing:
            details.append(f"missing manifest entries for live files {missing}")
        if added:
            details.append(f"manifest entries do not exist in the live dataset {added}")
        raise ValueError("Dataset manifest/live dataset membership mismatch: " + "; ".join(details))

    live_entries: list[dict[str, str]] = []
    for key, path in sorted(actual_files.items()):
        destination, declared_hash = declared_files[key]
        actual_hash = _sha256_file(path)
        if actual_hash != declared_hash:
            raise ValueError(
                f"Dataset file hash mismatch for {destination.as_posix()}: "
                f"manifest={declared_hash}, live={actual_hash}"
            )
        live_entries.append({"destination": destination.as_posix(), "sha256": actual_hash})

    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise ValueError(f"Dataset manifest counts must be an object: {manifest_path}")
    for split in SPLITS:
        declared_counts = counts.get(split)
        if not isinstance(declared_counts, dict):
            raise ValueError(f"Dataset manifest counts.{split} must be an object")
        for class_name in CLASS_NAMES:
            value = declared_counts.get(class_name)
            if type(value) is not int or value != actual_counts[split][class_name]:
                raise ValueError(
                    f"Dataset manifest count mismatch for {split}/{class_name}: "
                    f"manifest={value!r}, live={actual_counts[split][class_name]}"
                )
    return {
        "path": str(manifest_path.resolve()),
        "schema_version": schema_version,
        "sha256": _sha256_text(_canonical_json(manifest)),
        "live_sha256": _sha256_text(_canonical_json(live_entries)),
    }


def _build_provenance(
    config: dict[str, Any], *, data_dir: Path, output_dir: Path, device: torch.device
) -> dict[str, Any]:
    config_for_hash = {
        key: value for key, value in config.items() if not key.startswith("_")
    }
    config_for_hash["data_dir"] = str(data_dir.resolve())
    config_for_hash["output"] = str(output_dir.resolve())
    runtime = _runtime_identity(device)
    return {
        "config_sha256": _sha256_text(_canonical_json(config_for_hash)),
        "data_dir": str(data_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "dataset_manifest": _dataset_manifest_provenance(data_dir),
        "device": str(device),
        "cuda_device_count": runtime["cuda_device_count"],
        "runtime": runtime,
        "classes": list(CLASS_NAMES),
    }


def _validate_provenance(
    checkpoint: dict[str, Any], expected: dict[str, Any], path: Path
) -> None:
    saved = checkpoint.get("provenance")
    if not isinstance(saved, dict):
        raise ValueError(f"Checkpoint {path} has no valid run provenance; it cannot be resumed safely")
    for key in expected:
        if saved.get(key) != expected[key]:
            raise ValueError(
                f"Resume provenance mismatch for {key}: checkpoint has {saved.get(key)!r}, "
                f"active run has {expected[key]!r}. Use the matching config and dataset manifest."
            )
