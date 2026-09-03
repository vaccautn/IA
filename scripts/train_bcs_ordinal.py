"""Train and validate the VACCA ordinal BCS ResNet18 model."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TextIO


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

import torch  # noqa: E402
import torchvision  # noqa: E402
import yaml  # type: ignore[import-untyped]  # noqa: E402
from torch import Tensor  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vacca_bcs.constants import (  # noqa: E402
    BCS_CLASS_SCORES,
    BCS_DOMAIN_ID,
    CLASS_NAMES,
    MANIFEST_FILENAME,
    NUM_CLASSES,
    NUM_THRESHOLDS,
    SCORE_BASE,
    SCORE_MAX,
    SCORE_MIN,
    SCORE_STEP,
    SPLITS,
)
from vacca_bcs.dataset import BCSFolderDataset  # noqa: E402
from vacca_bcs.integer_snapshot import load_integer_snapshot_manifest  # noqa: E402
from vacca_bcs.model import BCSOrdinalModel, coral_loss, predict  # noqa: E402

CHECKPOINT_SCHEMA_VERSION = "bcs-ordinal-integer-checkpoint-v1"
RESULTS_LINEAGE_FILENAME = "results_lineage.json"
RESULTS_LINEAGE_SCHEMA_VERSION = "bcs-ordinal-integer-results-v1"
_DEFAULT_RUN_ID = "0" * 32
_DEFAULT_SNAPSHOT_ID = "0" * 64
TORCH_SEED_MAX = 2**64 - 1
DEFAULT_PROGRESS_EVERY_BATCHES = 50
_NON_MODEL_CONFIG_KEYS = frozenset(
    {"val_num_workers", "val_seed", "progress_every_batches"}
)

RESULTS_FIELDNAMES = [
    "epoch",
    "lr",
    "train_loss",
    "val_exact_acc",
    "val_pm1_acc",
    "val_mae",
    "val_recall",
]

RESUMABLE_CHECKPOINT_FIELDS = {
    "checkpoint_schema_version",
    "domain_id",
    "classes",
    "class_mapping",
    "score_min",
    "score_max",
    "score_base",
    "score_step",
    "num_classes",
    "num_thresholds",
    "snapshot_schema",
    "snapshot_identity",
    "dataset_manifest_digest",
    "run_id",
    "model_state_dict",
    "optimizer_state_dict",
    "epoch",
    "best_mae",
    "epochs_without_improvement",
    "config",
    "classes",
    "provenance",
    "rng_state",
    "observed_classes",
    "missing_classes",
    "allow_partial_class_coverage",
    "source_identity_scheme",
    "source_mapping",
}


def _resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Training config must be a YAML dictionary")
    if "data_root" in config and "data_dir" in config and config["data_root"] != config["data_dir"]:
        raise ValueError("data_root and data_dir must identify the same snapshot")
    if "output_dir" in config and "output" in config and config["output_dir"] != config["output"]:
        raise ValueError("output_dir and output must identify the same run output")
    if "data_root" not in config and "data_dir" in config:
        config["data_root"] = config["data_dir"]
    if "output_dir" not in config and "output" in config:
        config["output_dir"] = config["output"]
    config["data_dir"] = config.get("data_root")
    config["output"] = config.get("output_dir")
    config.setdefault("allow_partial_class_coverage", False)
    required = {
        "data_root",
        "output_dir",
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
    if "data_dir" not in config and "data_root" in config:
        config["data_dir"] = config["data_root"]
    if "output" not in config and "output_dir" in config:
        config["output"] = config["output_dir"]
    config.setdefault("allow_partial_class_coverage", False)
    config.setdefault("val_num_workers", 2)
    config.setdefault("progress_every_batches", DEFAULT_PROGRESS_EVERY_BATCHES)
    if type(config.get("allow_partial_class_coverage")) is not bool:
        raise ValueError("allow_partial_class_coverage must be a strict boolean")
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

    if config["seed"] > TORCH_SEED_MAX:
        raise ValueError(f"seed must be <= {TORCH_SEED_MAX} for Torch")
    if "val_seed" not in config:
        config["val_seed"] = config["seed"] + 1

    for key, minimum in (
        ("val_num_workers", 0),
        ("val_seed", 0),
        ("progress_every_batches", 1),
    ):
        value = config[key]
        if type(value) is not int or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if config["val_seed"] > TORCH_SEED_MAX:
        raise ValueError(f"val_seed must be <= {TORCH_SEED_MAX} for Torch")

    if config["num_workers"] != 0:
        raise ValueError("num_workers must be 0 for reproducible training")

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


def _coverage_from_manifest(manifest: dict[str, Any]) -> dict[str, list[int]]:
    counts = manifest["counts"]
    observed = [
        score
        for score in BCS_CLASS_SCORES
        if counts["train"][score - SCORE_MIN] or counts["val"][score - SCORE_MIN]
    ]
    return {
        "observed_classes": observed,
        "missing_classes": [score for score in BCS_CLASS_SCORES if score not in observed],
    }


def _validate_class_coverage(
    manifest: dict[str, Any], *, allow_partial_class_coverage: bool
) -> dict[str, list[int]]:
    if type(allow_partial_class_coverage) is not bool:
        raise ValueError("allow_partial_class_coverage must be a strict boolean")
    coverage = _coverage_from_manifest(manifest)
    train = manifest["counts"]["train"]
    val = manifest["counts"]["val"]
    train_observed = [score for score in BCS_CLASS_SCORES if train[score - SCORE_MIN]]
    if not any(train):
        raise ValueError("snapshot training split is empty")
    if not any(val):
        raise ValueError("snapshot validation split is empty")
    if len(train_observed) < 2:
        raise ValueError("snapshot needs at least two training classes")
    if coverage["missing_classes"] and not allow_partial_class_coverage:
        raise ValueError("partial class coverage is disabled")
    return coverage


def _dataset_manifest_provenance(
    data_dir: Path, *, allow_partial_class_coverage: bool = False
) -> dict[str, Any]:
    manifest_path = data_dir / MANIFEST_FILENAME
    if not data_dir.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError("integer snapshot dataset or manifest is missing")
    manifest = load_integer_snapshot_manifest(manifest_path)
    coverage = _validate_class_coverage(
        manifest, allow_partial_class_coverage=allow_partial_class_coverage
    )
    data_root = data_dir.resolve()
    expected_dirs = set(SPLITS) | {
        f"{split}/{name}" for split in SPLITS for name in CLASS_NAMES
    }
    actual_dirs: set[str] = set()
    actual_files: dict[str, Path] = {}
    for path in data_root.rglob("*"):
        relative = path.relative_to(data_root).as_posix()
        if path.is_symlink():
            raise ValueError("integer snapshot contains an unsafe filesystem entry")
        if path.is_dir():
            actual_dirs.add(relative)
        elif path.is_file() and relative != MANIFEST_FILENAME:
            key = relative.casefold()
            if key in actual_files:
                raise ValueError("integer snapshot contains duplicate live paths")
            actual_files[key] = path
    if actual_dirs != expected_dirs:
        raise ValueError("integer snapshot root structure is inconsistent")

    declared_files = {
        record["relative_path"].casefold(): (record["relative_path"], record["sha256"])
        for record in manifest["records"]
    }
    if set(actual_files) != set(declared_files):
        raise ValueError("integer snapshot manifest/live dataset membership mismatch")

    live_entries: list[dict[str, str]] = []
    for key, path in sorted(actual_files.items()):
        destination, declared_hash = declared_files[key]
        actual_hash = _sha256_file(path)
        if actual_hash != declared_hash:
            raise ValueError("integer snapshot file digest mismatch")
        live_entries.append({"destination": destination, "sha256": actual_hash})
    return {
        "schema_version": manifest["manifest_schema_version"],
        "domain_id": manifest["domain_id"],
        "source_schema": manifest["source_schema"],
        "identity_scheme": manifest.get("identity_scheme"),
        "mapping": manifest.get("mapping"),
        **coverage,
        "allow_partial_class_coverage": allow_partial_class_coverage,
        "split_identity": manifest["split_plan"]["identity_digest"],
        "sha256": _sha256_text(_canonical_json(manifest)),
        "live_sha256": _sha256_text(_canonical_json(live_entries)),
    }


def _build_provenance(
    config: dict[str, Any], *, data_dir: Path, output_dir: Path, device: torch.device,
    run_id: str = _DEFAULT_RUN_ID,
) -> dict[str, Any]:
    config_for_hash = {
        key: value
        for key, value in config.items()
        if not key.startswith("_") and key not in _NON_MODEL_CONFIG_KEYS
    }
    config_for_hash.setdefault("allow_partial_class_coverage", False)
    config_for_hash["data_dir"] = str(data_dir.resolve())
    config_for_hash["output"] = str(output_dir.resolve())
    runtime = _runtime_identity(device)
    return {
        "config_sha256": _sha256_text(_canonical_json(config_for_hash)),
        "run_id": run_id,
        "domain_id": BCS_DOMAIN_ID,
        "data_dir": str(data_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "dataset_manifest": _dataset_manifest_provenance(
            data_dir,
            allow_partial_class_coverage=config.get("allow_partial_class_coverage", False),
        ),
        "device": str(device),
        "cuda_device_count": runtime["cuda_device_count"],
        "runtime": runtime,
        "classes": list(CLASS_NAMES),
        "class_values": list(BCS_CLASS_SCORES),
    }


def _checkpoint_lineage(provenance: dict[str, Any], run_id: str) -> dict[str, Any]:
    manifest = provenance.get("dataset_manifest", {})
    coverage = manifest.get("observed_classes", [1, 2, 3, 4, 5])
    missing = manifest.get("missing_classes", [])
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "domain_id": BCS_DOMAIN_ID,
        "classes": list(CLASS_NAMES),
        "class_mapping": {name: index for index, name in enumerate(CLASS_NAMES)},
        "score_min": SCORE_MIN,
        "score_max": SCORE_MAX,
        "score_base": SCORE_BASE,
        "score_step": SCORE_STEP,
        "num_classes": NUM_CLASSES,
        "num_thresholds": NUM_THRESHOLDS,
        "snapshot_schema": manifest.get("schema_version", "bcs-integer-snapshot-v2"),
        "snapshot_identity": manifest.get("split_identity", _DEFAULT_SNAPSHOT_ID),
        "dataset_manifest_digest": manifest.get("sha256", _DEFAULT_SNAPSHOT_ID),
        "run_id": run_id,
        "observed_classes": coverage,
        "missing_classes": missing,
        "allow_partial_class_coverage": manifest.get("allow_partial_class_coverage", False),
        "source_identity_scheme": manifest.get("identity_scheme"),
        "source_mapping": manifest.get("mapping"),
    }


def _results_lineage(provenance: dict[str, Any], run_id: str) -> dict[str, Any]:
    checkpoint = _checkpoint_lineage(provenance, run_id)
    return {
        "lineage_schema_version": RESULTS_LINEAGE_SCHEMA_VERSION,
        "run_id": checkpoint["run_id"],
        "domain_id": checkpoint["domain_id"],
        "snapshot_schema": checkpoint["snapshot_schema"],
        "snapshot_identity": checkpoint["snapshot_identity"],
        "dataset_manifest_digest": checkpoint["dataset_manifest_digest"],
        "observed_classes": checkpoint["observed_classes"],
        "missing_classes": checkpoint["missing_classes"],
        "allow_partial_class_coverage": checkpoint["allow_partial_class_coverage"],
        "source_identity_scheme": checkpoint["source_identity_scheme"],
        "source_mapping": checkpoint["source_mapping"],
    }


def _valid_hex(value: object, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_checkpoint_lineage(
    checkpoint: dict[str, Any], *, path: Path, expected: dict[str, Any] | None = None
) -> None:
    if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Checkpoint {path} has an unsupported integer schema")
    if checkpoint.get("domain_id") != BCS_DOMAIN_ID:
        raise ValueError(f"Checkpoint {path} has an invalid integer domain")
    if (
        type(checkpoint.get("classes")) is not list
        or any(type(value) is not str for value in checkpoint["classes"])
        or checkpoint["classes"] != list(CLASS_NAMES)
    ):
        raise ValueError(f"Checkpoint {path} classes must be a list of integer classes")
    if (
        type(checkpoint.get("class_mapping")) is not dict
        or any(type(value) is not int for value in checkpoint["class_mapping"].values())
        or checkpoint["class_mapping"]
        != {name: index for index, name in enumerate(CLASS_NAMES)}
    ):
        raise ValueError(f"Checkpoint {path} class mapping is invalid")
    if any(
        type(checkpoint.get(field)) is not int or checkpoint[field] != value
        for field, value in (
            ("score_min", SCORE_MIN),
            ("score_max", SCORE_MAX),
            ("score_base", SCORE_BASE),
            ("score_step", SCORE_STEP),
            ("num_classes", NUM_CLASSES),
            ("num_thresholds", NUM_THRESHOLDS),
        )
    ):
        raise ValueError(f"Checkpoint {path} scale is invalid")
    if (
        checkpoint.get("snapshot_schema") != "bcs-integer-snapshot-v2"
        or not _valid_hex(checkpoint.get("snapshot_identity"), 64)
        or not _valid_hex(checkpoint.get("dataset_manifest_digest"), 64)
        or not _valid_hex(checkpoint.get("run_id"), 32)
    ):
        raise ValueError(f"Checkpoint {path} snapshot or run lineage is invalid")
    observed = checkpoint.get("observed_classes")
    missing = checkpoint.get("missing_classes")
    if (
        type(observed) is not list
        or any(type(value) is not int for value in observed)
        or observed != sorted(set(observed))
        or any(value not in BCS_CLASS_SCORES for value in observed)
        or type(missing) is not list
        or any(type(value) is not int for value in missing)
        or missing != sorted(set(missing))
        or missing != [score for score in BCS_CLASS_SCORES if score not in observed]
        or type(checkpoint.get("allow_partial_class_coverage")) is not bool
        or (
            bool(missing)
            and not checkpoint["allow_partial_class_coverage"]
        )
        or type(checkpoint.get("source_identity_scheme")) not in (str, type(None))
        or type(checkpoint.get("source_mapping")) not in (dict, type(None))
    ):
        raise ValueError(f"Checkpoint {path} coverage metadata is invalid")
    if expected is not None:
        expected_lineage = _checkpoint_lineage(expected, expected["run_id"])
        if any(checkpoint.get(field) != value for field, value in expected_lineage.items()):
            raise ValueError(f"Checkpoint {path} has an invalid integer lineage field")
    return


def _validate_lineage_file(path: Path, expected: dict[str, Any], label: str) -> None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError(f"{label} lineage is unreadable") from None
    if (
        type(value) is not dict
        or (label == "results" and set(value) != set(expected))
        or any(value.get(key) != expected[key] for key in expected)
    ):
        raise ValueError(f"{label} lineage does not match the active run lineage")


def _validate_existing_run_lineage(output_dir: Path, expected: dict[str, Any]) -> None:
    run_info = output_dir / "run_info.json"
    if run_info.is_file():
        _validate_lineage_file(run_info, expected, "run")
    results_lineage = output_dir / RESULTS_LINEAGE_FILENAME
    if not results_lineage.is_file():
        raise ValueError("results lineage is missing")
    _validate_lineage_file(results_lineage, expected, "results")


def _validate_provenance(
    checkpoint: dict[str, Any], expected: dict[str, Any], path: Path
) -> None:
    saved = checkpoint.get("provenance")
    if not isinstance(saved, dict):
        raise ValueError(f"Checkpoint {path} has no valid run provenance; it cannot be resumed safely")
    for key in expected:
        if saved.get(key) != expected[key]:
            raise ValueError(
                f"Resume provenance mismatch for {key}; use the matching config and dataset manifest."
            )


def _capture_rng_state() -> dict[str, Any]:
    cuda_states: list[Tensor] | None = None
    cuda_device_count = 0
    if torch.cuda.is_available():
        cuda_device_count = torch.cuda.device_count()
        cuda_states = [state.cpu() for state in torch.cuda.get_rng_state_all()]
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "cuda": cuda_states,
        "cuda_device_count": cuda_device_count,
    }


def _restore_rng_state(checkpoint: dict[str, Any]) -> None:
    state = checkpoint.get("rng_state")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint has no valid RNG state; it cannot be resumed safely")
    try:
        random.setstate(state["python"])
        torch.set_rng_state(state["torch_cpu"])
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"Checkpoint CPU RNG state is invalid: {exc}") from exc

    saved_cuda = state.get("cuda")
    saved_count = state.get("cuda_device_count", 0)
    if type(saved_count) is not int or saved_count < 0:
        raise ValueError("Checkpoint CUDA RNG device count is invalid")
    current_available = torch.cuda.is_available()
    current_count = torch.cuda.device_count() if current_available else 0
    if saved_cuda is None:
        if saved_count != 0:
            raise ValueError("Checkpoint CUDA RNG metadata is inconsistent")
        return
    if not current_available:
        raise ValueError(
            "Checkpoint contains CUDA RNG state, but CUDA is unavailable in this environment; "
            "resume on CPU is unsafe."
        )
    if not isinstance(saved_cuda, list) or len(saved_cuda) != current_count:
        raise ValueError(
            f"Checkpoint CUDA RNG state covers {saved_count} device(s), but this environment "
            f"has {current_count}; resume requires the same CUDA device count."
        )
    for index, value in enumerate(saved_cuda):
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
            raise ValueError(f"Checkpoint CUDA RNG entry {index} is not a usable byte tensor")
    if saved_count != current_count:
        raise ValueError(
            f"Checkpoint CUDA RNG state covers {saved_count} device(s), but this environment "
            f"has {current_count}; resume requires the same CUDA device count."
        )
    try:
        torch.cuda.set_rng_state_all([value.cpu() for value in saved_cuda])
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"Checkpoint CUDA RNG state is invalid: {exc}") from exc


def _fsync_directory(path: Path) -> None:
    """Best-effort directory durability; Windows may not support directory fsync."""
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        os.close(descriptor)


def _flush_and_fsync(handle: Any) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Flush a staged checkpoint, then replace the destination atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(raw_temp_path)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            torch.save(payload, handle)
            _flush_and_fsync(handle)
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        if descriptor != -1:
            os.close(descriptor)
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise


def _existing_run_artifacts(output_dir: Path) -> list[Path]:
    """List run artifacts whose presence marks the output dir as an existing run."""
    candidates = [
        output_dir / "results.csv",
        output_dir / "run_info.json",
        output_dir / RESULTS_LINEAGE_FILENAME,
    ]
    weights_dir = output_dir / "weights"
    if weights_dir.is_dir():
        candidates.extend(sorted(weights_dir.glob("*.pt")))
    return [path for path in candidates if path.is_file()]


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    descriptor, raw_temp_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(raw_temp_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
            _flush_and_fsync(handle)
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        if descriptor != -1:
            os.close(descriptor)
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise


@dataclass(frozen=True)
class RunInfoContext:
    """Inputs shared by normal and terminal-finalization run metadata writes."""

    config: dict[str, Any]
    data_dir: Path
    output_dir: Path
    device: torch.device
    provenance: dict[str, Any]
    run_id: str
    train_dataset: BCSFolderDataset
    val_dataset: BCSFolderDataset
    started_at: str
    started_time: float
    resume: Path | None
    start_epoch: int
    terminal_finalization: bool = False


def _build_run_info(context: RunInfoContext) -> dict[str, Any]:
    """Build serialized run metadata from one cohesive execution context."""
    run_info: dict[str, Any] = {
        **_results_lineage(context.provenance, context.run_id),
        "started_at_utc": context.started_at,
        "finished_at_utc": _utc_now(),
        "config_file": str(context.config["_config_path"]),
        "dataset": str(context.data_dir),
        "domain_id": BCS_DOMAIN_ID,
        "class_values": list(BCS_CLASS_SCORES),
        "class_counts": {
            "train": context.train_dataset.class_counts,
            "val": context.val_dataset.class_counts,
        },
        "device": str(context.device),
        "output_dir": str(context.output_dir),
        "provenance": context.provenance,
        "coverage": {
            key: context.provenance["dataset_manifest"][key]
            for key in (
                "observed_classes",
                "missing_classes",
                "allow_partial_class_coverage",
            )
        },
        "wall_time_seconds": time.perf_counter() - context.started_time,
    }
    if context.resume is not None:
        run_info["resumed_from"] = str(context.resume)
        run_info["resumed_at_epoch"] = context.start_epoch
    if context.terminal_finalization:
        run_info["finalized_from_terminal_checkpoint"] = True
    return run_info


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    """Protect a fresh run from clobbering artifacts of a previous run."""
    existing = _existing_run_artifacts(output_dir)
    if existing and not overwrite:
        raise FileExistsError(
            "Output directory already contains run artifacts: "
            + ", ".join(str(path) for path in existing)
            + ". Use --overwrite to replace them, or --resume "
            "<output>/weights/last.pt to continue the existing run."
        )
    for path in existing:
        path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def _open_results_csv(
    results_path: Path, *, append: bool
) -> tuple[TextIO, csv.DictWriter[str]]:
    """Open results.csv, appending without a duplicate header when resuming."""
    if append and results_path.is_file() and results_path.stat().st_size > 0:
        handle = results_path.open("a", newline="", encoding="utf-8")
        return handle, csv.DictWriter(handle, fieldnames=RESULTS_FIELDNAMES)
    handle = results_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=RESULTS_FIELDNAMES)
    writer.writeheader()
    return handle, writer


def _atomic_write_results_prefix(path: Path, rows: list[list[str]]) -> None:
    descriptor, raw_temp_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(raw_temp_path)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            descriptor = -1
            writer = csv.writer(handle)
            writer.writerow(RESULTS_FIELDNAMES)
            writer.writerows(rows)
            _flush_and_fsync(handle)
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        if descriptor != -1:
            os.close(descriptor)
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise


def _validate_result_epoch(
    row: list[str], *, line_number: int, expected_epoch: int
) -> None:
    if not row:
        return
    try:
        epoch_value = int(row[0])
    except ValueError:
        raise ValueError(
            f"results.csv line {line_number} has a non-integer epoch: {row[0]!r}"
        ) from None
    if epoch_value != expected_epoch:
        raise ValueError(
            f"results.csv line {line_number} breaks the contiguous epoch"
            f" sequence: expected epoch {expected_epoch}, found {epoch_value}"
            " (gap, duplicate, or missing row)"
        )


def _validate_results_row(row: list[str], *, line_number: int, expected_epoch: int) -> None:
    if len(row) not in (len(RESULTS_FIELDNAMES) - 1, len(RESULTS_FIELDNAMES)):
        raise ValueError(
            f"results.csv line {line_number} is malformed: expected"
            f" {len(RESULTS_FIELDNAMES)} columns, got {len(row)}"
        )
    _validate_result_epoch(row, line_number=line_number, expected_epoch=expected_epoch)
    metric_domains = {
        "lr": (0.0, None),
        "train_loss": (0.0, None),
        "val_exact_acc": (0.0, 1.0),
        "val_pm1_acc": (0.0, 1.0),
        "val_mae": (0.0, SCORE_STEP * (len(CLASS_NAMES) - 1)),
    }
    for column, (lower, upper) in metric_domains.items():
        try:
            value = float(row[RESULTS_FIELDNAMES.index(column)])
        except (ValueError, OverflowError):
            raise ValueError(
                f"results.csv line {line_number} has a non-numeric {column}:"
                f" {row[RESULTS_FIELDNAMES.index(column)]!r}"
            ) from None
        if not math.isfinite(value) or value < lower or (
            upper is not None and value > upper
        ):
            domain = f"[{lower}, {upper}]" if upper is not None else f">= {lower}"
            raise ValueError(
                f"results.csv line {line_number} has an invalid {column} {value!r};"
                f" expected a finite value {domain}"
            )
    if len(row) == len(RESULTS_FIELDNAMES):
        try:
            recalls = json.loads(row[RESULTS_FIELDNAMES.index("val_recall")])
        except (TypeError, ValueError, json.JSONDecodeError):
            raise ValueError(f"results.csv line {line_number} has invalid val_recall") from None
        if (
            type(recalls) is not dict
            or set(recalls) != set(CLASS_NAMES)
            or any(
                value is not None
                and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or not 0 <= float(value) <= 1
                )
                for value in recalls.values()
            )
        ):
            raise ValueError(f"results.csv line {line_number} has invalid val_recall")


def _reconcile_results_csv(results_path: Path, *, checkpoint_epoch: int) -> int:
    """Reconcile an existing results.csv with a resumed checkpoint epoch.

    The CSV must carry the exact RESULTS_FIELDNAMES header and a contiguous
    committed prefix 1..checkpoint_epoch. Only one complete or partial final
    row for checkpoint_epoch + 1 is disposable as the CSV/checkpoint crash
    window. Older stale checkpoints and malformed committed rows raise.
    Returns the number of epoch rows kept (always checkpoint_epoch).
    """
    if not results_path.is_file():
        raise FileNotFoundError(
            "Cannot resume without the matching results history:"
            f" {results_path}"
        )
    with results_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(
            f"results.csv is empty; expected epochs 1..{checkpoint_epoch}:"
            f" {results_path}"
        )
    if rows[0] != RESULTS_FIELDNAMES:
        raise ValueError(
            f"results.csv has an unexpected schema; expected header"
            f" {RESULTS_FIELDNAMES}: {results_path}"
        )

    data_rows = rows[1:]
    for line_number, row in enumerate(data_rows[:checkpoint_epoch], start=2):
        _validate_result_epoch(row, line_number=line_number, expected_epoch=line_number - 1)
    if len(data_rows) < checkpoint_epoch:
        raise ValueError(
            f"results.csv history ends at epoch {len(data_rows)}, before the"
            f" checkpoint epoch {checkpoint_epoch}; missing rows cannot be"
            " recovered"
        )
    committed_rows = data_rows[:checkpoint_epoch]
    for line_number, row in enumerate(committed_rows, start=2):
        _validate_results_row(row, line_number=line_number, expected_epoch=line_number - 1)

    suffix = data_rows[checkpoint_epoch:]
    if not suffix:
        return checkpoint_epoch
    if len(suffix) != 1:
        raise ValueError(
            f"results.csv has {len(suffix)} rows beyond checkpoint epoch "
            f"{checkpoint_epoch}; only one crash-window row is recoverable"
        )
    extra_row = suffix[0]
    if len(extra_row) > len(RESULTS_FIELDNAMES):
        raise ValueError(
            f"results.csv line {checkpoint_epoch + 2} is malformed: expected"
            f" {len(RESULTS_FIELDNAMES)} columns, got {len(extra_row)}"
        )
    if len(extra_row) in (len(RESULTS_FIELDNAMES) - 1, len(RESULTS_FIELDNAMES)):
        try:
            extra_epoch = int(extra_row[0])
        except ValueError:
            raise ValueError(
                f"results.csv has an unrecoverable complete suffix beyond checkpoint "
                f"epoch {checkpoint_epoch}: non-integer epoch {extra_row[0]!r}"
            ) from None
        if extra_epoch != checkpoint_epoch + 1:
            raise ValueError(
                f"results.csv has an unrecoverable suffix beyond checkpoint epoch "
                f"{checkpoint_epoch}; expected disposable epoch {checkpoint_epoch + 1}"
            )
        _validate_results_row(
            extra_row,
            line_number=checkpoint_epoch + 2,
            expected_epoch=checkpoint_epoch + 1,
        )
    _atomic_write_results_prefix(results_path, committed_rows)
    if len(data_rows) > checkpoint_epoch:
        print(
            f"[INFO] results.csv had {len(data_rows)} epochs; truncated to epoch "
            f"{checkpoint_epoch} to match the checkpoint.",
            flush=True,
        )
    return checkpoint_epoch


def _build_last_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    best_mae: float,
    epochs_without_improvement: int,
    config: dict[str, Any],
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the resumable checkpoint saved as weights/last.pt every epoch."""
    active_provenance = provenance or {}
    run_id = active_provenance.get("run_id", _DEFAULT_RUN_ID)
    payload = {
        **_checkpoint_lineage(active_provenance, run_id),
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_mae": float(best_mae),
        "epochs_without_improvement": int(epochs_without_improvement),
        "config": config,
        "classes": list(CLASS_NAMES),
        "provenance": active_provenance,
        "rng_state": _capture_rng_state(),
    }
    return payload


def _validate_checkpoint_metadata(
    checkpoint: dict[str, Any], *, path: Path, total_epochs: int
) -> None:
    for field in ("model_state_dict", "optimizer_state_dict", "config", "provenance", "rng_state"):
        if not isinstance(checkpoint[field], dict):
            raise ValueError(f"Checkpoint {path} field {field!r} must be an object")
    _validate_checkpoint_lineage(checkpoint, path=path)
    epoch = checkpoint["epoch"]
    if type(epoch) is not int or epoch < 1:
        raise ValueError(f"Checkpoint {path} epoch must be a non-negative integer >= 1")
    if epoch > total_epochs:
        raise ValueError(
            f"Checkpoint epoch {epoch} exceeds configured total of {total_epochs} epochs"
        )
    best_mae = checkpoint["best_mae"]
    if isinstance(best_mae, bool) or not isinstance(best_mae, (int, float)):
        raise ValueError(f"Checkpoint {path} best_mae must be a finite number")
    best_mae = float(best_mae)
    if not math.isfinite(best_mae) or best_mae < 0 or best_mae > SCORE_STEP * (len(CLASS_NAMES) - 1):
        raise ValueError(f"Checkpoint {path} best_mae is outside the valid BCS metric domain")
    without_improvement = checkpoint["epochs_without_improvement"]
    if type(without_improvement) is not int or without_improvement < 0:
        raise ValueError(
            f"Checkpoint {path} epochs_without_improvement must be a non-negative integer"
        )
    if without_improvement > epoch:
        raise ValueError(
            f"Checkpoint {path} epochs_without_improvement cannot exceed completed epoch {epoch}"
        )
    rng_state = checkpoint["rng_state"]
    cuda_count = rng_state.get("cuda_device_count")
    if type(cuda_count) is not int or cuda_count < 0:
        raise ValueError(f"Checkpoint {path} CUDA RNG device count is invalid")
    cuda_state = rng_state.get("cuda")
    if cuda_count == 0:
        if cuda_state is not None:
            raise ValueError(
                f"Checkpoint {path} CUDA RNG state must be absent when device count is zero"
            )
    elif not isinstance(cuda_state, list) or len(cuda_state) != cuda_count:
        raise ValueError(f"Checkpoint {path} CUDA RNG state does not match its device count")
    if isinstance(cuda_state, list):
        for index, value in enumerate(cuda_state):
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.uint8
                or value.ndim != 1
            ):
                raise ValueError(f"Checkpoint {path} CUDA RNG entry {index} is not a usable byte tensor")


def _load_resume_checkpoint(
    path: Path,
    *,
    expected_classes: list[str],
    total_epochs: int,
    expected_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and validate a resumable checkpoint (weights/last.pt)."""
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Resume checkpoint could not be loaded safely: {path}: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Resume checkpoint is not a state dictionary: {path}")
    missing = sorted(RESUMABLE_CHECKPOINT_FIELDS.difference(checkpoint))
    if missing:
        hint = ""
        if "optimizer_state_dict" in missing:
            hint = (
                " Best-only checkpoints such as weights/best.pt are not"
                " resumable; pass weights/last.pt instead."
            )
        raise ValueError(
            f"Checkpoint {path} is missing required fields:"
            f" {', '.join(missing)}.{hint}"
        )
    if set(checkpoint).difference(RESUMABLE_CHECKPOINT_FIELDS):
        raise ValueError(f"Checkpoint {path} has unexpected fields")
    _validate_checkpoint_metadata(checkpoint, path=path, total_epochs=total_epochs)
    classes = list(checkpoint["classes"])
    if classes != list(expected_classes):
        raise ValueError(
            f"Checkpoint classes {classes} do not match the expected BCS"
            f" classes {list(expected_classes)}"
        )
    if expected_provenance is not None:
        _validate_checkpoint_lineage(checkpoint, path=path, expected=expected_provenance)
        _validate_provenance(checkpoint, expected_provenance, path)
    completed_epochs = int(checkpoint["epoch"])
    if completed_epochs < 1:
        raise ValueError(f"Checkpoint epoch must be >= 1, got {completed_epochs}")
    if completed_epochs == total_epochs:
        return checkpoint
    if completed_epochs > total_epochs:
        raise ValueError(
            f"Checkpoint already completed {completed_epochs} epochs, which"
            f" exceeds the configured total of {total_epochs} epochs"
        )
    return checkpoint


def _restore_model_state(model: torch.nn.Module, checkpoint: dict[str, Any]) -> None:
    """Restore model weights, failing clearly on architecture mismatch."""
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except (KeyError, TypeError, RuntimeError) as exc:
        raise ValueError(
            "Checkpoint model state does not match the BCS ordinal model"
            f" architecture: {exc}"
        ) from exc


def _restore_optimizer_state(
    optimizer: torch.optim.Optimizer,
    checkpoint: dict[str, Any],
    *,
    device: torch.device | None = None,
) -> None:
    """Restore optimizer state, moving tensors to the training device."""
    try:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"Checkpoint optimizer state is incompatible: {exc}") from exc
    if device is not None:
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)


def _lr_factor(epoch: int, epochs: int, warmup_epochs: int) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    cosine_epoch = epoch - warmup_epochs
    cosine_total = max(1, epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * cosine_epoch / cosine_total))


def _set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _format_duration(seconds: float) -> str:
    return f"{seconds:.1f}s"


def _worker_init_fn(worker_id: int) -> None:
    """Seed worker-local Python randomness without touching the parent process."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


def _build_data_loader(
    dataset: BCSFolderDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    generator: torch.Generator | None = None,
) -> DataLoader[tuple[Tensor, int]]:
    """Build a loader without passing multiprocessing-only options to worker=0."""
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if generator is not None:
        kwargs["generator"] = generator
    if num_workers > 0:
        kwargs.update(
            {
                "prefetch_factor": 2,
                "persistent_workers": False,
                "worker_init_fn": _worker_init_fn,
            }
        )
    return DataLoader(dataset, **kwargs)


def _validation_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _train_epoch(
    model: BCSOrdinalModel,
    loader: DataLoader[tuple[Tensor, int]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epoch: int | None = None,
    total_epochs: int | None = None,
    progress_every_batches: int = DEFAULT_PROGRESS_EVERY_BATCHES,
) -> float:
    model.train()
    loss_total = torch.zeros((), dtype=torch.float64, device=device)
    samples = 0
    total_batches = len(loader)
    started = time.perf_counter()
    for batch_number, (images, levels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        levels = levels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = coral_loss(model(images), levels)
        loss.backward()
        optimizer.step()
        loss_total += loss.detach().to(torch.float64)
        samples += levels.shape[0]
        if epoch is not None and total_epochs is not None and (
            batch_number % progress_every_batches == 0 or batch_number == total_batches
        ):
            elapsed = time.perf_counter() - started
            eta = elapsed * max(0, total_batches - batch_number) / batch_number
            print(
                f"[TRAIN {epoch}/{total_epochs} {batch_number}/{total_batches}] "
                f"loss={loss_total.item() / max(1, samples):.8f} "
                f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}",
                flush=True,
            )
    return float(loss_total.item()) / max(1, samples)


@torch.inference_mode()
def _validate(
    model: BCSOrdinalModel,
    loader: DataLoader[tuple[Tensor, int]],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    exact = torch.zeros((), dtype=torch.int64, device=device)
    plus_minus_one = torch.zeros((), dtype=torch.int64, device=device)
    absolute_index_error = torch.zeros((), dtype=torch.int64, device=device)
    total = torch.zeros((), dtype=torch.int64, device=device)
    class_total = torch.zeros(len(CLASS_NAMES), dtype=torch.int64, device=device)
    class_correct = torch.zeros(len(CLASS_NAMES), dtype=torch.int64, device=device)

    for images, levels in loader:
        images = images.to(device, non_blocking=True)
        levels = levels.to(device, non_blocking=True)
        pred_idx, _ = predict(model(images))
        errors = (pred_idx - levels).abs()
        exact += (errors == 0).sum()
        plus_minus_one += (errors <= 1).sum()
        absolute_index_error += errors.sum()
        total += levels.numel()
        for class_idx in range(len(CLASS_NAMES)):
            mask = levels == class_idx
            class_total[class_idx] += mask.sum()
            class_correct[class_idx] += ((pred_idx == class_idx) & mask).sum()

    summary = torch.cat(
        (
            exact.reshape(1),
            plus_minus_one.reshape(1),
            absolute_index_error.reshape(1),
            total.reshape(1),
            class_total,
            class_correct,
        )
    ).cpu().tolist()
    exact_value, pm1_value, absolute_error_value, total_value = summary[:4]
    class_total_values = summary[4 : 4 + len(CLASS_NAMES)]
    class_correct_values = summary[4 + len(CLASS_NAMES) :]

    recalls = {
        class_name: (
            class_correct_values[index] / class_total_values[index]
            if class_total_values[index]
            else None
        )
        for index, class_name in enumerate(CLASS_NAMES)
    }
    return {
        "exact_acc": exact_value / max(1, total_value),
        "pm1_acc": pm1_value / max(1, total_value),
        "mae": SCORE_STEP * absolute_error_value / max(1, total_value),
        "recall": recalls,
        "total": total_value,
    }


@dataclass
class TrainingContext:
    config: dict[str, Any]
    total_epochs: int
    device: torch.device
    data_dir: Path
    output_dir: Path
    resume: Path | None
    checkpoint: dict[str, Any] | None
    run_id: str
    provenance: dict[str, Any]
    start_epoch: int
    best_mae: float
    epochs_without_improvement: int
    weights_dir: Path
    results_path: Path
    run_info_context: RunInfoContext
    train_loader: DataLoader[tuple[Tensor, int]] | None
    val_loader: DataLoader[tuple[Tensor, int]] | None
    model: torch.nn.Module | None
    optimizer: torch.optim.Optimizer | None
    epoch_range: range


@dataclass(frozen=True)
class TrainingResult:
    final_metrics: dict[str, Any]
    train_loss: float
    best_mae: float
    epochs_without_improvement: int


def _initialize_training(
    config: dict[str, Any], *, resume: Path | None, overwrite: bool
) -> TrainingContext:
    """Validate inputs and assemble the resumable training context."""
    _validate_training_config(config)
    total_epochs = int(config["epochs"])
    device = resolve_device(str(config["device"]))
    data_dir = _resolve_path(config["data_dir"])
    output_dir = _resolve_path(config["output"])
    if device.type == "cuda":
        print(f"[DEVICE] {device} {torch.cuda.get_device_name(device)}", flush=True)
    else:
        print(f"[DEVICE] {device}", flush=True)

    checkpoint: dict[str, Any] | None = None
    if resume is None and _existing_run_artifacts(output_dir) and not overwrite:
        _prepare_output_dir(output_dir, overwrite=False)
    if resume is not None:
        checkpoint = _load_resume_checkpoint(
            resume, expected_classes=list(CLASS_NAMES), total_epochs=total_epochs
        )
        run_id = checkpoint["run_id"]
    else:
        run_id = uuid.uuid4().hex
    provenance = _build_provenance(
        config, data_dir=data_dir, output_dir=output_dir, device=device, run_id=run_id
    )
    if resume is not None:
        _validate_checkpoint_lineage(checkpoint, path=resume, expected=provenance)
        _validate_provenance(checkpoint, provenance, resume)
        _validate_existing_run_lineage(output_dir, _results_lineage(provenance, run_id))
    if resume is None:
        if _existing_run_artifacts(output_dir):
            _prepare_output_dir(output_dir, overwrite=True)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(config["seed"]))
    start_epoch = 0
    best_mae = float("inf")
    epochs_without_improvement = 0
    if resume is not None:
        _restore_rng_state(checkpoint)
        start_epoch = int(checkpoint["epoch"])
        best_mae = float(checkpoint["best_mae"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        print(
            f"[INFO] Resuming from {resume}: epoch {start_epoch} completed; "
            f"best MAE={best_mae:.4f}. Target total: {total_epochs} epochs.",
            flush=True,
        )

    weights_dir = output_dir / "weights"
    train_dataset = BCSFolderDataset(
        data_dir / "train", train=True, imgsz=int(config["imgsz"])
    )
    val_dataset = BCSFolderDataset(
        data_dir / "val", train=False, imgsz=int(config["imgsz"])
    )
    started_at = _utc_now()
    started_time = time.perf_counter()
    run_info_context = RunInfoContext(
        config=config,
        data_dir=data_dir,
        output_dir=output_dir,
        device=device,
        provenance=provenance,
        run_id=run_id,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        started_at=started_at,
        started_time=started_time,
        resume=resume,
        start_epoch=start_epoch,
    )
    results_path = output_dir / "results.csv"
    if checkpoint is not None:
        _reconcile_results_csv(results_path, checkpoint_epoch=start_epoch)
    if checkpoint is not None and start_epoch == total_epochs:
        return TrainingContext(
            config, total_epochs, device, data_dir, output_dir, resume, checkpoint,
            run_id, provenance, start_epoch, best_mae, epochs_without_improvement,
            weights_dir, results_path, run_info_context, None, None, None, None, range(0)
        )

    train_loader = _build_data_loader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = _build_data_loader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["val_num_workers"]),
        pin_memory=device.type == "cuda",
        generator=_validation_generator(int(config["val_seed"])),
    )
    model = BCSOrdinalModel(pretrained=checkpoint is None).to(device)
    optimizer_name = str(config["optimizer"]).lower()
    if optimizer_name != "adamw":
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    if checkpoint is not None:
        _restore_model_state(model, checkpoint)
        _restore_optimizer_state(optimizer, checkpoint, device=device)
        _restore_rng_state(checkpoint)
    if resume is None:
        _prepare_output_dir(output_dir, overwrite=overwrite)
        _atomic_write_json(
            _results_lineage(provenance, run_id), output_dir / RESULTS_LINEAGE_FILENAME
        )
    weights_dir.mkdir(parents=True, exist_ok=True)
    epoch_range = range(start_epoch, total_epochs)
    if checkpoint is not None and epochs_without_improvement >= int(config["patience"]):
        print(
            f"[INFO] Early stopping already reached at epoch {start_epoch}; "
            "no additional epoch was run.",
            flush=True,
        )
        epoch_range = range(0)
    return TrainingContext(
        config, total_epochs, device, data_dir, output_dir, resume, checkpoint,
        run_id, provenance, start_epoch, best_mae, epochs_without_improvement,
        weights_dir, results_path, run_info_context, train_loader, val_loader,
        model, optimizer, epoch_range
    )


def _save_best_checkpoint(
    context: TrainingContext, *, epoch: int, best_mae: float
) -> None:
    assert context.model is not None
    state_dict = {
        key: value.detach().cpu() for key, value in context.model.state_dict().items()
    }
    _atomic_torch_save(
        {
            **_checkpoint_lineage(context.provenance, context.run_id),
            "model_state_dict": state_dict,
            "config": context.config,
            "classes": list(CLASS_NAMES),
            "provenance": context.provenance,
            "epoch": epoch,
            "val_mae": best_mae,
        },
        context.weights_dir / "best.pt",
    )


def _execute_training(context: TrainingContext) -> TrainingResult:
    """Run epochs and persist each committed history/checkpoint unit."""
    assert context.model is not None
    assert context.optimizer is not None
    assert context.train_loader is not None
    assert context.val_loader is not None
    final_metrics: dict[str, Any] = {}
    train_loss = 0.0
    best_mae = context.best_mae
    epochs_without_improvement = context.epochs_without_improvement
    csv_file, writer = _open_results_csv(
        context.results_path, append=context.checkpoint is not None
    )
    with csv_file:
        for epoch in context.epoch_range:
            current_lr = float(context.config["lr"]) * _lr_factor(
                epoch,
                context.total_epochs,
                int(context.config.get("warmup_epochs", 2)),
            )
            _set_learning_rate(context.optimizer, current_lr)
            epoch_number = epoch + 1
            print(
                f"[EPOCH {epoch_number}/{context.total_epochs}] training: "
                f"{len(context.train_loader)} batches",
                flush=True,
            )
            train_loss = _train_epoch(
                context.model,
                context.train_loader,
                context.optimizer,
                context.device,
                epoch=epoch_number,
                total_epochs=context.total_epochs,
                progress_every_batches=int(context.config["progress_every_batches"]),
            )
            print(
                f"[EPOCH {epoch_number}/{context.total_epochs}] validating: "
                f"{len(context.val_loader)} batches",
                flush=True,
            )
            metrics = _validate(context.model, context.val_loader, context.device)
            final_metrics = metrics
            writer.writerow(
                {
                    "epoch": epoch_number,
                    "lr": f"{current_lr:.10g}",
                    "train_loss": f"{train_loss:.8f}",
                    "val_exact_acc": f"{metrics['exact_acc']:.8f}",
                    "val_pm1_acc": f"{metrics['pm1_acc']:.8f}",
                    "val_mae": f"{metrics['mae']:.8f}",
                    "val_recall": _canonical_json(metrics["recall"]),
                }
            )
            _flush_and_fsync(csv_file)
            elapsed = time.perf_counter() - context.run_info_context.started_time
            epochs_done = epoch - context.start_epoch + 1
            remaining_epochs = max(0, context.total_epochs - epoch_number)
            eta = elapsed * remaining_epochs / max(1, epochs_done)
            print(
                f"[EPOCH {epoch_number}/{context.total_epochs}] complete: "
                f"loss={train_loss:.8f} exact={metrics['exact_acc']:.8f} "
                f"pm1={metrics['pm1_acc']:.8f} MAE={metrics['mae']:.8f} "
                f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}",
                flush=True,
            )
            print(
                "  Recall: "
                + ", ".join(
                    f"{class_name}={recall if recall is None else format(recall, '.3f')}"
                    for class_name, recall in metrics["recall"].items()
                ),
                flush=True,
            )
            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                epochs_without_improvement = 0
                _save_best_checkpoint(context, epoch=epoch_number, best_mae=best_mae)
            else:
                epochs_without_improvement += 1
            _atomic_torch_save(
                _build_last_checkpoint(
                    context.model,
                    context.optimizer,
                    epoch=epoch_number,
                    best_mae=best_mae,
                    epochs_without_improvement=epochs_without_improvement,
                    config=context.config,
                    provenance=context.provenance,
                ),
                context.weights_dir / "last.pt",
            )
            if epochs_without_improvement >= int(context.config["patience"]):
                print(f"[INFO] Early stopping at epoch {epoch_number}.", flush=True)
                break
    return TrainingResult(
        final_metrics, train_loss, best_mae, epochs_without_improvement
    )


def _finalize_training(context: TrainingContext, result: TrainingResult) -> dict[str, Any]:
    run_info = _build_run_info(context.run_info_context)
    _atomic_write_json(run_info, context.output_dir / "run_info.json")
    print(
        f"\n[DONE] BCS training completed in {run_info['wall_time_seconds']:.1f}s.",
        flush=True,
    )
    print(f"  Directory: {context.output_dir}", flush=True)
    print(f"  Best checkpoint: {context.weights_dir / 'best.pt'}", flush=True)
    print(f"  Resumable checkpoint: {context.weights_dir / 'last.pt'}", flush=True)
    if result.final_metrics:
        print(
            f"  Final: loss={result.train_loss:.5f}, "
            f"exact={result.final_metrics['exact_acc']:.4f}, "
            f"pm1={result.final_metrics['pm1_acc']:.4f}, "
            f"MAE={result.final_metrics['mae']:.4f}",
            flush=True,
        )
    return {
        "run_info": run_info,
        "final_metrics": result.final_metrics,
        "best_mae": result.best_mae,
        "output_dir": context.output_dir,
    }


def train(
    config: dict[str, Any],
    *,
    resume: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    context = _initialize_training(config, resume=resume, overwrite=overwrite)
    if context.checkpoint is not None and context.start_epoch == context.total_epochs:
        run_info = _build_run_info(
            replace(context.run_info_context, terminal_finalization=True)
        )
        _atomic_write_json(run_info, context.output_dir / "run_info.json")
        return {
            "run_info": run_info,
            "final_metrics": {},
            "best_mae": context.best_mae,
            "output_dir": context.output_dir,
        }
    result = _execute_training(context)
    return _finalize_training(context, result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the VACCA ordinal BCS model")
    parser.add_argument("--config", required=True, help="YAML training configuration")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument(
        "--device",
        default=None,
        help="Training device: auto, cpu, or cuda[:index] (config default)",
    )
    parser.add_argument("--data-dir", default=None, help="Override the snapshot directory")
    parser.add_argument("--output", default=None, help="Override the run output directory")
    parser.add_argument(
        "--progress-every-batches",
        type=int,
        default=None,
        help="Override the configured training progress cadence in batches",
    )
    guard = parser.add_mutually_exclusive_group()
    guard.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing run artifacts in the output directory and start fresh",
    )
    guard.add_argument(
        "--resume",
        default=None,
        help="Path to a resumable weights/last.pt checkpoint to continue training",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = _resolve_path(args.config)
    if not config_path.is_file():
        print(f"[ERROR] Config file not found: {config_path}", file=sys.stderr, flush=True)
        return 1
    try:
        config = load_config(config_path)
        overrides = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": args.device,
            "data_dir": args.data_dir,
            "output": args.output,
            "progress_every_batches": args.progress_every_batches,
        }
        for key, value in overrides.items():
            if value is not None:
                config[key] = value
        resume_path = _resolve_path(args.resume) if args.resume else None
        train(config, resume=resume_path, overwrite=args.overwrite)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError, yaml.YAMLError) as exc:
        print(f"[ERROR] Training failed: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
