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

import torch
import torchvision
import yaml  # type: ignore[import-untyped]
from torch import Tensor
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vacca_bcs.constants import (  # noqa: E402
    CLASS_NAMES,
    IMAGE_EXTENSIONS,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    SCORE_STEP,
    SPLITS,
)
from vacca_bcs.dataset import BCSFolderDataset  # noqa: E402
from vacca_bcs.model import BCSOrdinalModel, coral_loss, predict  # noqa: E402

RESULTS_FIELDNAMES = [
    "epoch",
    "lr",
    "train_loss",
    "val_exact_acc",
    "val_pm1_acc",
    "val_mae",
]

RESUMABLE_CHECKPOINT_FIELDS = {
    "model_state_dict",
    "optimizer_state_dict",
    "epoch",
    "best_mae",
    "epochs_without_improvement",
    "config",
    "classes",
    "provenance",
    "rng_state",
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
    candidates = [output_dir / "results.csv", output_dir / "run_info.json"]
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
            handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
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
        "started_at_utc": context.started_at,
        "finished_at_utc": _utc_now(),
        "config_file": str(context.config["_config_path"]),
        "dataset": str(context.data_dir),
        "class_counts": {
            "train": context.train_dataset.class_counts,
            "val": context.val_dataset.class_counts,
        },
        "device": str(context.device),
        "output_dir": str(context.output_dir),
        "provenance": context.provenance,
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
    if len(row) != len(RESULTS_FIELDNAMES):
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
    if len(extra_row) == len(RESULTS_FIELDNAMES):
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
            f"[INFO] results.csv tenía {len(data_rows)} épocas; truncado a la"
            f" época {checkpoint_epoch} para coincidir con el checkpoint."
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
    return {
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_mae": float(best_mae),
        "epochs_without_improvement": int(epochs_without_improvement),
        "config": config,
        "classes": list(CLASS_NAMES),
        "provenance": provenance or {},
        "rng_state": _capture_rng_state(),
    }


def _validate_checkpoint_metadata(
    checkpoint: dict[str, Any], *, path: Path, total_epochs: int
) -> None:
    for field in ("model_state_dict", "optimizer_state_dict", "config", "provenance", "rng_state"):
        if not isinstance(checkpoint[field], dict):
            raise ValueError(f"Checkpoint {path} field {field!r} must be an object")
    classes = checkpoint["classes"]
    if not isinstance(classes, list) or not all(isinstance(value, str) for value in classes):
        raise ValueError(f"Checkpoint {path} classes must be a list of strings")
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
    _validate_checkpoint_metadata(checkpoint, path=path, total_epochs=total_epochs)
    classes = list(checkpoint["classes"])
    if classes != list(expected_classes):
        raise ValueError(
            f"Checkpoint classes {classes} do not match the expected BCS"
            f" classes {list(expected_classes)}"
        )
    if expected_provenance is not None:
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


def _train_epoch(
    model: BCSOrdinalModel,
    loader: DataLoader[tuple[Tensor, int]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    loss_total = 0.0
    samples = 0
    for images, levels in loader:
        images = images.to(device, non_blocking=True)
        levels = levels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = coral_loss(model(images), levels)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
        samples += levels.shape[0]
    return loss_total / max(1, samples)


@torch.no_grad()
def _validate(
    model: BCSOrdinalModel,
    loader: DataLoader[tuple[Tensor, int]],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    exact = 0
    plus_minus_one = 0
    absolute_index_error = 0
    total = 0
    class_total = [0] * len(CLASS_NAMES)
    class_correct = [0] * len(CLASS_NAMES)

    for images, levels in loader:
        images = images.to(device, non_blocking=True)
        levels = levels.to(device, non_blocking=True)
        pred_idx, _ = predict(model(images))
        errors = (pred_idx - levels).abs()
        exact += int((errors == 0).sum().item())
        plus_minus_one += int((errors <= 1).sum().item())
        absolute_index_error += int(errors.sum().item())
        total += levels.shape[0]
        for class_idx in range(len(CLASS_NAMES)):
            mask = levels == class_idx
            class_total[class_idx] += int(mask.sum().item())
            class_correct[class_idx] += int(((pred_idx == class_idx) & mask).sum().item())

    recalls = {
        class_name: (
            class_correct[index] / class_total[index]
            if class_total[index]
            else 0.0
        )
        for index, class_name in enumerate(CLASS_NAMES)
    }
    return {
        "exact_acc": exact / max(1, total),
        "pm1_acc": plus_minus_one / max(1, total),
        "mae": SCORE_STEP * absolute_index_error / max(1, total),
        "recall": recalls,
        "total": total,
    }


def train(
    config: dict[str, Any],
    *,
    resume: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    _validate_training_config(config)
    total_epochs = int(config["epochs"])
    set_seed(int(config["seed"]))
    device = resolve_device(str(config["device"]))
    data_dir = _resolve_path(config["data_dir"])
    output_dir = _resolve_path(config["output"])

    start_epoch = 0
    best_mae = float("inf")
    epochs_without_improvement = 0
    checkpoint: dict[str, Any] | None = None

    if resume is None:
        if _existing_run_artifacts(output_dir) and not overwrite:
            _prepare_output_dir(output_dir, overwrite=False)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    provenance = _build_provenance(
        config, data_dir=data_dir, output_dir=output_dir, device=device
    )
    if resume is not None:
        checkpoint = _load_resume_checkpoint(
            resume,
            expected_classes=list(CLASS_NAMES),
            total_epochs=total_epochs,
            expected_provenance=provenance,
        )
        _restore_rng_state(checkpoint)
        start_epoch = int(checkpoint["epoch"])
        best_mae = float(checkpoint["best_mae"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        print(
            f"[INFO] Reanudando desde {resume}: época {start_epoch} completada, "
            f"mejor MAE={best_mae:.4f} BCS. Objetivo total: {total_epochs} épocas."
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
        if start_epoch == total_epochs:
            run_info = _build_run_info(
                replace(run_info_context, terminal_finalization=True)
            )
            _atomic_write_json(run_info, output_dir / "run_info.json")
            return {
                "run_info": run_info,
                "final_metrics": {},
                "best_mae": best_mae,
                "output_dir": output_dir,
            }
    loader_kwargs = {
        "batch_size": int(config["batch_size"]),
        "num_workers": int(config["num_workers"]),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

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
    weights_dir.mkdir(parents=True, exist_ok=True)
    epoch_range = range(start_epoch, total_epochs)
    if checkpoint is not None and epochs_without_improvement >= int(config["patience"]):
        print(
            f"[INFO] Early stopping already reached at epoch {start_epoch}; "
            "no additional epoch was run."
        )
        epoch_range = range(0)

    csv_file, writer = _open_results_csv(results_path, append=checkpoint is not None)
    final_metrics: dict[str, Any] = {}
    train_loss = 0.0
    with csv_file:
        for epoch in epoch_range:
            current_lr = float(config["lr"]) * _lr_factor(
                epoch,
                total_epochs,
                int(config.get("warmup_epochs", 2)),
            )
            _set_learning_rate(optimizer, current_lr)
            train_loss = _train_epoch(model, train_loader, optimizer, device)
            metrics = _validate(model, val_loader, device)
            final_metrics = metrics
            writer.writerow(
                {
                    "epoch": epoch + 1,
                    "lr": f"{current_lr:.10g}",
                    "train_loss": f"{train_loss:.8f}",
                    "val_exact_acc": f"{metrics['exact_acc']:.8f}",
                    "val_pm1_acc": f"{metrics['pm1_acc']:.8f}",
                    "val_mae": f"{metrics['mae']:.8f}",
                }
            )
            _flush_and_fsync(csv_file)
            print(
                f"[ÉPOCA {epoch + 1}/{total_epochs}] "
                f"loss={train_loss:.5f} "
                f"exact={metrics['exact_acc']:.4f} "
                f"±1={metrics['pm1_acc']:.4f} "
                f"MAE={metrics['mae']:.4f} BCS"
            )
            print(
                "  Recall: "
                + ", ".join(
                    f"{class_name}={recall:.3f}"
                    for class_name, recall in metrics["recall"].items()
                )
            )

            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                epochs_without_improvement = 0
                state_dict = {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                }
                _atomic_torch_save(
                    {
                        "model_state_dict": state_dict,
                        "config": config,
                        "classes": list(CLASS_NAMES),
                        "provenance": provenance,
                        "epoch": epoch + 1,
                        "val_mae": best_mae,
                    },
                    weights_dir / "best.pt",
                )
            else:
                epochs_without_improvement += 1

            _atomic_torch_save(
                _build_last_checkpoint(
                    model,
                    optimizer,
                    epoch=epoch + 1,
                    best_mae=best_mae,
                    epochs_without_improvement=epochs_without_improvement,
                    config=config,
                    provenance=provenance,
                ),
                weights_dir / "last.pt",
            )

            if epochs_without_improvement >= int(config["patience"]):
                print(f"[INFO] Early stopping en época {epoch + 1}.")
                break

    run_info = _build_run_info(run_info_context)
    _atomic_write_json(run_info, output_dir / "run_info.json")
    print(f"\n[LISTO] Entrenamiento BCS completado en {run_info['wall_time_seconds']:.1f}s.")
    print(f"  Directorio: {output_dir}")
    print(f"  Mejor checkpoint: {weights_dir / 'best.pt'}")
    print(f"  Checkpoint reanudable: {weights_dir / 'last.pt'}")
    if final_metrics:
        print(
            f"  Final: loss={train_loss:.5f}, "
            f"exact={final_metrics['exact_acc']:.4f}, "
            f"±1={final_metrics['pm1_acc']:.4f}, "
            f"MAE={final_metrics['mae']:.4f} BCS"
        )
    return {
        "run_info": run_info,
        "final_metrics": final_metrics,
        "best_mae": best_mae,
        "output_dir": output_dir,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the VACCA ordinal BCS model")
    parser.add_argument("--config", required=True, help="YAML training configuration")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output", default=None)
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
        print(f"[ERROR] Config file not found: {config_path}", file=sys.stderr)
        return 1
    try:
        config = load_config(config_path)
        overrides = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": args.device,
            "data_dir": args.data_dir,
            "output": args.output,
        }
        for key, value in overrides.items():
            if value is not None:
                config[key] = value
        resume_path = _resolve_path(args.resume) if args.resume else None
        train(config, resume=resume_path, overwrite=args.overwrite)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError, yaml.YAMLError) as exc:
        print(f"[ERROR] Training failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
