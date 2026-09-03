"""Safely prepare and run the bounded overnight integer BCS workflow."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import yaml

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT_RELATIVE = Path("data/bcs/dataset")
LOCAL_SNAPSHOT_RELATIVE = Path("data/bcs-local-integer-v1")
LOCAL_OUTPUT_RELATIVE = Path("outputs/bcs-ordinal-local-integer-v1")
BACKEND_SNAPSHOT_RELATIVE = Path("data/bcs-integer-v1")
BACKEND_OUTPUT_RELATIVE = Path("outputs/bcs-ordinal-integer-v1")
CONFIG_RELATIVE = Path("configs/training_bcs_ordinal.yaml")
MANIFEST_NAME = "manifest.json"
LAST_CHECKPOINT = Path("weights/last.pt")
BEST_CHECKPOINT = Path("weights/best.pt")
RESULTS_LINEAGE = "results_lineage.json"
BASELINE_MAE = 0.0356609410
BASELINE_EXACT = 0.96433906
BASELINE_PM1 = 1.0


class OvernightError(RuntimeError):
    """Raised for safe, operator-facing orchestration failures."""


@dataclass(frozen=True)
class Plan:
    root: Path
    config: Path
    source: str
    local_root: Path | None
    snapshot: Path
    output: Path
    checkpoint: Path
    logs: Path
    skip_build: bool
    resume: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight and run the integer BCS snapshot/training workflow. "
            "Training output is tee'd to logs/<run-id>/train.log."
        )
    )
    parser.add_argument("--source", choices=("local", "backend"), default="local")
    parser.add_argument("--local-root", default=str(LOCAL_ROOT_RELATIVE))
    parser.add_argument("--config", default=str(CONFIG_RELATIVE))
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the source-specific weights/last.pt checkpoint",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--logs-root",
        default="logs/bcs-overnight",
        help="Parent directory for per-run build.log and train.log files",
    )
    return parser


def _path(raw: str | Path, root: Path) -> Path:
    candidate = Path(raw)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _environment(environment: Mapping[str, str] | None) -> dict[str, str]:
    return dict(os.environ if environment is None else environment)


def _load_config(path: Path) -> dict[str, object]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        raise OvernightError("training config could not be read") from None
    if not isinstance(value, dict):
        raise OvernightError("training config must be a YAML mapping")
    return value


def _require_path(path: Path, label: str) -> None:
    if not path.is_file():
        raise OvernightError(f"{label} is missing")


def preflight(
    args: argparse.Namespace,
    *,
    root: Path = ROOT,
    environment: Mapping[str, str] | None = None,
) -> Plan:
    root = Path(root).resolve()
    if not root.is_dir():
        raise OvernightError("repository root is missing")
    config = _path(args.config, root)
    _require_path(config, "training config")
    _require_path(root / "scripts" / "build_bcs_integer.py", "snapshot builder")
    _require_path(root / "scripts" / "train_bcs_ordinal.py", "ordinal trainer")
    values = _load_config(config)
    local_snapshot = (root / LOCAL_SNAPSHOT_RELATIVE).resolve()
    local_output = (root / LOCAL_OUTPUT_RELATIVE).resolve()
    for key, expected in (("data_root", local_snapshot), ("output_dir", local_output)):
        raw = values.get(key)
        if not isinstance(raw, str) or _path(raw, root) != expected:
            raise OvernightError(f"training config {key} disagrees with the canonical root")
    if args.source == "backend" and _path(args.local_root, root) != root / LOCAL_ROOT_RELATIVE:
        raise OvernightError("--local-root is only valid for local source")
    if args.source == "local":
        local_root = _path(args.local_root, root)
        if not local_root.is_dir():
            raise OvernightError("local source root is missing")
        snapshot, output = local_snapshot, local_output
    else:
        local_root = None
        snapshot = (root / BACKEND_SNAPSHOT_RELATIVE).resolve()
        output = (root / BACKEND_OUTPUT_RELATIVE).resolve()

    if args.resume and not args.skip_build:
        raise OvernightError("--resume requires --skip-build")
    if not args.skip_build and snapshot.exists():
        raise OvernightError("canonical snapshot already exists; use --skip-build")
    if args.skip_build and (
        not snapshot.is_dir() or not (snapshot / MANIFEST_NAME).is_file()
    ):
        raise OvernightError("--skip-build requires an existing snapshot manifest")
    if args.resume:
        checkpoint = output / LAST_CHECKPOINT
        if not output.is_dir() or not checkpoint.is_file():
            raise OvernightError("--resume requires outputs/.../weights/last.pt")
        if not (output / RESULTS_LINEAGE).is_file():
            raise OvernightError("--resume requires results_lineage.json")
    elif output.exists():
        raise OvernightError("canonical training output already exists; fresh runs refuse stale artifacts")
    if not args.skip_build and args.source == "backend":
        env = _environment(environment)
        if not env.get("VACCA_BACKEND_URL", "").strip():
            raise OvernightError("VACCA_BACKEND_URL is required for the snapshot build")
        if not env.get("VACCA_BACKEND_TOKEN", "").strip():
            raise OvernightError("VACCA_BACKEND_TOKEN is required for the snapshot build")
    return Plan(
        root=root,
        config=config,
        source=args.source,
        local_root=local_root,
        snapshot=snapshot,
        output=output,
        checkpoint=output / LAST_CHECKPOINT,
        logs=_path(args.logs_root, root),
        skip_build=args.skip_build,
        resume=args.resume,
    )


def _redact(text: str, environment: Mapping[str, str]) -> str:
    token = environment.get("VACCA_BACKEND_TOKEN", "")
    return text.replace(token, "[REDACTED]") if token else text


def _trainer_environment(environment: Mapping[str, str]) -> dict[str, str]:
    child_environment = dict(environment)
    child_environment.pop("VACCA_BACKEND_URL", None)
    child_environment.pop("VACCA_BACKEND_TOKEN", None)
    return child_environment


def _close_stdout(process: subprocess.Popen[str]) -> None:
    stdout = process.stdout
    if stdout is not None:
        try:
            stdout.close()
        except BaseException:
            pass


def _stop_process(process: subprocess.Popen[str]) -> None:
    """Terminate and reap the immediate child after a streaming failure."""
    try:
        if process.poll() is None:
            process.terminate()
    except BaseException:
        pass
    try:
        process.wait(timeout=5)
    except BaseException:
        pass


def _validate_best_checkpoint(
    best_path: Path, output_dir: Path, snapshot: Path, config_path: Path
) -> dict[str, object]:
    """Validate the selected checkpoint, lineage, history, and regression floor."""
    try:
        import torch

        checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    except Exception:
        raise OvernightError("best checkpoint could not be loaded safely") from None
    if not isinstance(checkpoint, dict):
        raise OvernightError("best checkpoint has an invalid schema")

    try:
        from scripts.train_bcs_ordinal import (
            RESULTS_FIELDNAMES,
            BCSFolderDataset,
            _build_data_loader,
            _NON_MODEL_CONFIG_KEYS,
            _canonical_json,
            _dataset_manifest_provenance,
            _sha256_text,
            _results_lineage,
            _validate,
            _validate_checkpoint_lineage,
            _validate_results_row,
            load_config,
        )
        from vacca_bcs.serving import load_bcs_model

        _validate_checkpoint_lineage(checkpoint, path=best_path)
        loaded = load_bcs_model(best_path, device="cpu")
        provenance = checkpoint["provenance"]
        if not isinstance(provenance, dict):
            raise ValueError("checkpoint provenance is invalid")
        live_manifest = _dataset_manifest_provenance(
            snapshot,
            allow_partial_class_coverage=bool(
                checkpoint["allow_partial_class_coverage"]
            ),
        )
        if provenance.get("dataset_manifest") != live_manifest:
            raise ValueError("checkpoint dataset lineage does not match the live snapshot")
        if Path(provenance.get("data_dir", "")).resolve() != snapshot.resolve():
            raise ValueError("checkpoint data lineage does not match the active snapshot")
        if Path(provenance.get("output_dir", "")).resolve() != output_dir.resolve():
            raise ValueError("checkpoint output lineage does not match the active output")
        checkpoint_config = checkpoint.get("config")
        if (
            not isinstance(checkpoint_config, dict)
            or checkpoint_config.get("_config_path") != str(config_path.resolve())
        ):
            raise ValueError("checkpoint config lineage does not match the active config")
        active_config = load_config(config_path)
        if snapshot != (config_path.parents[1] / "data/bcs-local-integer-v1").resolve():
            active_config["data_dir"] = str(snapshot)
            active_config["output"] = str(output_dir)
        config_for_hash = {
            key: value
            for key, value in active_config.items()
            if not key.startswith("_") and key not in _NON_MODEL_CONFIG_KEYS
        }
        config_for_hash.setdefault("allow_partial_class_coverage", False)
        config_for_hash["data_dir"] = str(snapshot.resolve())
        config_for_hash["output"] = str(output_dir.resolve())
        if provenance.get("config_sha256") != _sha256_text(
            _canonical_json(config_for_hash)
        ):
            raise ValueError("checkpoint provenance does not match the active run")
        results_lineage_path = output_dir / RESULTS_LINEAGE
        results_lineage = json.loads(results_lineage_path.read_text(encoding="utf-8"))
        expected_lineage = _results_lineage(provenance, checkpoint["run_id"])
        if results_lineage != expected_lineage:
            raise ValueError("checkpoint lineage does not match results lineage")

        results_path = output_dir / "results.csv"
        with results_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        if not rows or rows[0] != RESULTS_FIELDNAMES:
            raise ValueError("results.csv has an invalid schema")
        data_rows = rows[1:]
        for line_number, row in enumerate(data_rows, start=2):
            _validate_results_row(row, line_number=line_number, expected_epoch=line_number - 1)
        if not data_rows:
            raise ValueError("results.csv has no completed epochs")
        if not config_path.is_file():
            raise ValueError("active training config is missing")
        best_row = min(data_rows, key=lambda row: float(row[5]))
        final_row = data_rows[-1]
        if int(checkpoint.get("epoch", 0)) != int(best_row[0]):
            raise ValueError("best checkpoint epoch does not match selected results row")
        if format(float(checkpoint.get("val_mae", float("nan"))), ".8f") != format(
            float(best_row[5]), ".8f"
        ):
            raise ValueError("best checkpoint metric does not match selected results row")
        validation_dataset = BCSFolderDataset(
            snapshot / "val",
            train=False,
            imgsz=int(checkpoint["config"]["imgsz"]),
        )
        validation_loader = _build_data_loader(
            validation_dataset,
            batch_size=int(checkpoint["config"].get("batch_size", 64)),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            generator=torch.Generator().manual_seed(0),
        )
        verified_metrics = _validate(loaded.model, validation_loader, torch.device("cpu"))
        if any(
            format(float(verified_metrics[field]), ".8f") != format(float(best_row[column]), ".8f")
            for field, column in (
                ("exact_acc", 3),
                ("pm1_acc", 4),
                ("mae", 5),
            )
        ):
            raise ValueError("best checkpoint predictions do not match results lineage")
        best_metrics = {
            "epoch": int(best_row[0]),
            "mae": float(best_row[5]),
            "exact_acc": float(best_row[3]),
            "pm1_acc": float(best_row[4]),
        }
        if (
            best_metrics["mae"] > BASELINE_MAE
            or best_metrics["exact_acc"] < BASELINE_EXACT
            or best_metrics["pm1_acc"] < BASELINE_PM1
        ):
            raise ValueError(
                "selected best checkpoint is below the BCS regression floor "
                "(MAE <= 0.0356609410, exact >= 0.96433906, pm1 >= 1.0)"
            )
        return {"best": best_metrics, "final": {"epoch": int(final_row[0])}}
    except OvernightError:
        raise
    except Exception as error:
        raise OvernightError(f"best checkpoint validation failed: {error}") from None
def _run_subprocess(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    log_path: Path,
) -> int:
    process: subprocess.Popen[str] | None = None
    with log_path.open("w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(
                list(command),
                cwd=str(cwd),
                env=dict(environment),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                safe = _redact(line, environment)
                log.write(safe)
                log.flush()
                print(safe, end="", flush=True)
            return process.wait()
        except BaseException:
            if process is not None:
                _close_stdout(process)
                _stop_process(process)
            raise
        finally:
            if process is not None:
                _close_stdout(process)


def _sanitize_log(path: Path, environment: Mapping[str, str]) -> None:
    try:
        contents = path.read_text(encoding="utf-8")
        path.write_text(_redact(contents, environment), encoding="utf-8")
    except (OSError, UnicodeError):
        return


def _step(
    name: str,
    command: Sequence[str],
    *,
    plan: Plan,
    environment: Mapping[str, str],
    runner: Callable[..., int],
    log_path: Path,
    redaction_environment: Mapping[str, str] | None = None,
) -> None:
    redaction_environment = redaction_environment or environment
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch()
    print(
        f"[overnight] {name}: "
        + " ".join(_redact(str(part), redaction_environment) for part in command),
        flush=True,
    )
    try:
        status = runner(command, cwd=plan.root, environment=environment, log_path=log_path)
    except Exception:
        _sanitize_log(log_path, redaction_environment)
        raise OvernightError(f"{name} could not start; see {log_path}") from None
    _sanitize_log(log_path, redaction_environment)
    if status:
        raise OvernightError(f"{name} failed with exit code {status}; see {log_path}")


def run(
    args: argparse.Namespace,
    *,
    root: Path = ROOT,
    environment: Mapping[str, str] | None = None,
    runner: Callable[..., int] = _run_subprocess,
    best_validator: Callable[..., dict[str, object]] = _validate_best_checkpoint,
) -> Plan:
    env = _environment(environment)
    plan = preflight(args, root=root, environment=env)
    if args.preflight_only:
        print(
            f"READY: preflight passed for {plan.root}; no subprocess was started.",
            flush=True,
        )
        return plan
    run_dir = plan.logs / uuid.uuid4().hex
    build_log = run_dir / "build.log"
    train_log = run_dir / "train.log"
    if not plan.skip_build:
        command = [
            sys.executable,
            str(plan.root / "scripts" / "build_bcs_integer.py"),
            "--source",
            plan.source,
        ]
        if plan.source == "local":
            command.extend(("--local-root", str(plan.local_root)))
        command.extend(("--output", str(plan.snapshot)))
        _step(
            "snapshot build",
            command,
            plan=plan,
            environment=env,
            runner=runner,
            log_path=build_log,
            redaction_environment=env,
        )
    command = [
        sys.executable,
        "-u",
        str(plan.root / "scripts" / "train_bcs_ordinal.py"),
        "--config",
        str(plan.config),
    ]
    if plan.source == "backend":
        command.extend(("--data-dir", str(plan.snapshot), "--output", str(plan.output)))
    if plan.resume:
        command.extend(("--resume", str(plan.checkpoint)))
    _step(
        "ordinal training",
        command,
        plan=plan,
        environment=_trainer_environment(env),
        runner=runner,
        log_path=train_log,
        redaction_environment=env,
    )
    best = plan.output / BEST_CHECKPOINT
    best_validation = best_validator(best, plan.output, plan.snapshot, plan.config)
    best_metrics = best_validation["best"]
    assert isinstance(best_metrics, dict)
    print(
        f"[overnight] Validated best checkpoint: epoch={best_metrics['epoch']} "
        f"MAE={best_metrics['mae']:.8f} exact={best_metrics['exact_acc']:.8f} "
        f"pm1={best_metrics['pm1_acc']:.8f}",
        flush=True,
    )
    print("[overnight] Complete. API was not started.", flush=True)
    print(f"Morning: set VACCA_BCS_CHECKPOINT to {best}", flush=True)
    print("Morning: start the API, then verify /ready/bcs and /bcs.", flush=True)
    return plan


def main(
    argv: Sequence[str] | None = None,
    *,
    root: Path = ROOT,
    environment: Mapping[str, str] | None = None,
    runner: Callable[..., int] = _run_subprocess,
    stderr: TextIO | None = None,
    best_validator: Callable[..., dict[str, object]] = _validate_best_checkpoint,
) -> int:
    try:
        run(
            build_parser().parse_args(argv),
            root=root,
            environment=environment,
            runner=runner,
            best_validator=best_validator,
        )
    except OvernightError as error:
        print(
            f"ERROR: {_redact(str(error), _environment(environment))}",
            file=stderr or sys.stderr,
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
