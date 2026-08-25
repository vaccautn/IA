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
import csv
import datetime as dt
import tempfile
import time
from dataclasses import dataclass, replace
from typing import TextIO
from torch import Tensor

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
from vacca_bcs.dataset import BCSFolderDataset

RESULTS_FIELDNAMES = [
    "epoch",
    "lr",
    "train_loss",
    "val_exact_acc",
    "val_pm1_acc",
    "val_mae",
]
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
