"""Bounded local BCS category build/train orchestration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import uuid
from typing import TextIO

import torch

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT_RELATIVE = Path("data/bcs/dataset")
SNAPSHOT_RELATIVE = Path("data/bcs-category-v1")
OUTPUT_RELATIVE = Path("outputs/bcs-category-coral-v1")
CONFIG_RELATIVE = Path("configs/training_bcs_category.yaml")
DATA_ROOT = ROOT / "data"
OUTPUT_ROOT = ROOT / "outputs"
CONFIG_ROOT = ROOT / "configs"
LOG_ROOT = ROOT / "logs"
MANIFEST_NAME = "manifest.json"
LAST_CHECKPOINT = Path("weights/last.pt")
BEST_CHECKPOINT = Path("weights/best.pt")
CHECKPOINT_SET = Path("weights/checkpoint_set.json")
RESULTS_LINEAGE = "results_lineage.json"

sys.path.insert(0, str(ROOT / "src"))

from vacca_bcs.path_safety import SafePathError, safe_path  # noqa: E402


class OvernightError(RuntimeError):
    pass


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise OvernightError("invalid command line")


@dataclass(frozen=True)
class Plan:
    root: Path
    config: Path
    local_root: Path
    snapshot: Path
    output: Path
    checkpoint: Path
    logs: Path
    skip_build: bool
    resume: bool


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(description="Build and train the local BCS category 1..5 workflow")
    parser.add_argument("--local-root", default=str(LOCAL_ROOT_RELATIVE))
    parser.add_argument("--config", default=str(CONFIG_RELATIVE))
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--logs-root", default="logs/bcs-overnight")
    return parser


def _path(raw: str | Path, root: Path, *, approved_roots: tuple[Path, ...]) -> Path:
    return safe_path(raw, base=root, approved_roots=approved_roots, allow_missing_final=True)


def _load_config(path: Path) -> dict[str, object]:
    try:
        import yaml
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        raise OvernightError("training config could not be read") from None
    if not isinstance(value, dict):
        raise OvernightError("training config must be a mapping")
    return value


def preflight(
    args: argparse.Namespace,
    *,
    root: Path = ROOT,
    approved_roots: tuple[Path, ...],
    environment: Mapping[str, str] | None = None,
) -> Plan:
    try:
        root = safe_path(
            root,
            base=ROOT,
            approved_roots=approved_roots,
            allow_missing_final=False,
            require_dir=True,
        )
        data_root = DATA_ROOT if root == ROOT else root / "data"
        output_root = OUTPUT_ROOT if root == ROOT else root / "outputs"
        config_root = CONFIG_ROOT if root == ROOT else root / "configs"
        log_root = LOG_ROOT if root == ROOT else root / "logs"
        config = _path(args.config, root, approved_roots=(config_root,))
        local_root = safe_path(
            args.local_root,
            base=root,
            approved_roots=(data_root,),
            allow_missing_final=False,
            require_dir=True,
        )
        snapshot = _path(SNAPSHOT_RELATIVE, root, approved_roots=(data_root,))
        output = _path(OUTPUT_RELATIVE, root, approved_roots=(output_root,))
        logs = _path(args.logs_root, root, approved_roots=(log_root,))
    except SafePathError as error:
        raise OvernightError(f"unsafe workflow path: {error}") from None
    if not root.is_dir() or not config.is_file() or not local_root.is_dir():
        raise OvernightError("repository, training config, or local source root is missing")
    if not (root / "scripts" / "build_bcs_category.py").is_file() or not (root / "scripts" / "train_bcs_ordinal.py").is_file():
        raise OvernightError("BCS category workflow script is missing")
    values = _load_config(config)
    if not isinstance(values.get("data_root"), str) or _path(values["data_root"], root, approved_roots=(data_root,)) != snapshot:
        raise OvernightError("training config data_root disagrees with the canonical category snapshot")
    if not isinstance(values.get("output_dir"), str) or _path(values["output_dir"], root, approved_roots=(output_root,)) != output:
        raise OvernightError("training config output_dir disagrees with the canonical category output")
    if args.resume and not args.skip_build:
        raise OvernightError("--resume requires --skip-build")
    if not args.skip_build and snapshot.exists():
        raise OvernightError("canonical category snapshot already exists; use --skip-build")
    if args.skip_build and (not snapshot.is_dir() or not (snapshot / MANIFEST_NAME).is_file()):
        raise OvernightError("--skip-build requires an existing category snapshot manifest")
    if args.resume and (
        not (output / CHECKPOINT_SET).is_file()
        or not (output / RESULTS_LINEAGE).is_file()
    ):
        raise OvernightError("--resume requires the checkpoint set descriptor and results lineage")
    if not args.resume and output.exists():
        raise OvernightError("canonical category output already exists; use --resume")
    return Plan(root, config, local_root, snapshot, output, output / LAST_CHECKPOINT, logs, args.skip_build, args.resume)


def _close_stdout(process: subprocess.Popen[str]) -> None:
    if process.stdout is not None:
        try:
            process.stdout.close()
        except BaseException:
            pass


def _stop_process(process: subprocess.Popen[str]) -> None:
    try:
        if process.poll() is None:
            try:
                process.terminate()
            except BaseException:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            finally:
                process.wait(timeout=5)
    except BaseException:
        pass


def _run_subprocess(command: Sequence[str], *, cwd: Path, environment: Mapping[str, str], log_path: Path) -> int:
    process = None
    with log_path.open("w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(list(command), cwd=str(cwd), env=dict(environment), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return process.wait()
        except BaseException:
            if process is not None:
                _close_stdout(process)
                _stop_process(process)
            raise
        finally:
            if process is not None:
                _close_stdout(process)


def _step(name: str, command: Sequence[str], *, plan: Plan, environment: Mapping[str, str], runner: Callable[..., int], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[category] {name}: " + " ".join(map(str, command)), flush=True)
    try:
        status = runner(command, cwd=plan.root, environment=environment, log_path=log_path)
    except Exception:
        raise OvernightError(f"{name} could not start; see {log_path}") from None
    if status:
        raise OvernightError(f"{name} failed with exit code {status}; see {log_path}")


def _validate_best_checkpoint(
    best_path: Path,
    output_dir: Path,
    snapshot: Path,
    config_path: Path,
    approved_data_roots: tuple[Path, ...],
    approved_config_roots: tuple[Path, ...],
    approved_output_roots: tuple[Path, ...],
) -> dict[str, object]:
    try:
        output_dir = safe_path(
            output_dir,
            base=ROOT,
            approved_roots=approved_output_roots,
            allow_missing_final=False,
            require_dir=True,
        )
        snapshot = safe_path(
            snapshot,
            base=ROOT,
            approved_roots=approved_data_roots,
            allow_missing_final=False,
            require_dir=True,
        )
        config_path = safe_path(
            config_path,
            base=ROOT,
            approved_roots=approved_config_roots,
            allow_missing_final=False,
            require_file=True,
        )
        from vacca_bcs.category_snapshot import load_category_snapshot_manifest
        from vacca_bcs.checkpoint_io import load_checkpoint_bytes
        from vacca_bcs.serving import _build_loaded_bcs_model
        from scripts.train_bcs_ordinal import (
            RESULTS_FIELDNAMES,
            RESUMABLE_CHECKPOINT_FIELDS,
            _canonical_json,
            _config_sha256,
            _dataset_manifest_provenance,
            _evaluate_provisional_gates,
            _load_authoritative_checkpoint_set,
            _read_checkpoint_digest,
            _validate_checkpoint_set,
            _validate_checkpoint_metadata,
            _validate_metrics_payload,
            _validate_results_row,
            load_config,
            _results_lineage,
        )

        manifest = load_category_snapshot_manifest(snapshot / "manifest.json")
        manifest_digest = hashlib.sha256(_canonical_json(manifest).encode()).hexdigest()
        config = load_config(config_path)
        if (
            Path(str(config.get("data_root"))).resolve() != snapshot.resolve()
            or Path(str(config.get("output_dir"))).resolve() != output_dir.resolve()
        ):
            raise ValueError("active training config does not use canonical category roots")
        checkpoint_set = _load_authoritative_checkpoint_set(
            output_dir / "weights", approved_output_roots=approved_output_roots
        )
        best_path = safe_path(
            best_path,
            base=output_dir.parent,
            approved_roots=approved_output_roots,
            allow_missing_final=True,
        )
        checkpoint_bytes = load_checkpoint_bytes(
            best_path,
            approved_roots=approved_output_roots,
        )
        loaded = _build_loaded_bcs_model(checkpoint_bytes, torch.device("cpu"))
        checkpoint = loaded.checkpoint
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint payload is invalid")
        checkpoint_sha256 = loaded.checkpoint_sha256
        if not isinstance(checkpoint_sha256, str):
            raise ValueError("checkpoint digest is unavailable")
        last_path = safe_path(
            output_dir / "weights" / "last.pt",
            base=output_dir.parent,
            approved_roots=approved_output_roots,
            allow_missing_final=True,
        )
        last_sha256 = _read_checkpoint_digest(
            last_path, approved_output_roots=approved_output_roots
        )
        last_bytes = load_checkpoint_bytes(
            last_path,
            approved_roots=approved_output_roots,
            expected_sha256=last_sha256,
        )
        last_checkpoint = last_bytes.payload
        if (
            checkpoint_set["best"]["sha256"] != checkpoint_sha256
            or checkpoint_set["last"]["sha256"] != last_sha256
        ):
            raise ValueError("checkpoint-set descriptor does not match selected generations")
        missing_last_fields = sorted(RESUMABLE_CHECKPOINT_FIELDS.difference(last_checkpoint))
        if missing_last_fields:
            raise ValueError("last checkpoint is not resumable")
        _validate_checkpoint_metadata(
            last_checkpoint,
            path=last_path,
            total_epochs=int(config["epochs"]),
        )
        _validate_checkpoint_set(
            last_checkpoint,
            checkpoint,
            last_path=last_path,
            best_path=best_path,
        )
        expected_identity = manifest["split_plan"]["identity_digest"]
        expected_provenance = checkpoint["provenance"]
        live_dataset = _dataset_manifest_provenance(snapshot)
        if expected_provenance.get("dataset_manifest") != live_dataset:
            raise ValueError("live snapshot traversal does not match checkpoint provenance")
        if (
            checkpoint["snapshot_identity"] != expected_identity
            or checkpoint["dataset_manifest_digest"] != manifest_digest
            or checkpoint["source_schema"] != manifest["source_schema"]
            or checkpoint["config_sha256"] != expected_provenance.get("config_sha256")
            or checkpoint["config"] != config
            or expected_provenance["config_sha256"] != _config_sha256(
                config, data_dir=snapshot, output_dir=output_dir
            )
            or loaded.lineage.snapshot_identity != expected_identity
            or loaded.lineage.dataset_manifest_digest != manifest_digest
        ):
            raise ValueError("checkpoint does not match the canonical snapshot or active config")

        results_lineage_path = safe_path(
            output_dir / RESULTS_LINEAGE,
            base=output_dir.parent,
            approved_roots=approved_output_roots,
            allow_missing_final=False,
            require_file=True,
        )
        run_info_path = safe_path(
            output_dir / "run_info.json",
            base=output_dir.parent,
            approved_roots=approved_output_roots,
            allow_missing_final=False,
            require_file=True,
        )
        results_path = safe_path(
            output_dir / "results.csv",
            base=output_dir.parent,
            approved_roots=approved_output_roots,
            allow_missing_final=False,
            require_file=True,
        )
        results_lineage = json.loads(results_lineage_path.read_text(encoding="utf-8"))
        if results_lineage != _results_lineage(expected_provenance, checkpoint["run_id"]):
            raise ValueError("results lineage does not match the selected checkpoint")
        run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
        if (
            run_info.get("run_id") != checkpoint["run_id"]
            or run_info.get("config_sha256") != checkpoint["config_sha256"]
            or run_info.get("snapshot_identity") != checkpoint["snapshot_identity"]
            or run_info.get("dataset_manifest_digest") != checkpoint["dataset_manifest_digest"]
            or run_info.get("provenance") != expected_provenance
            or run_info.get("candidate_status") != "candidate_pending_handoff"
        ):
            raise ValueError("run info does not match the selected checkpoint")
        best = run_info.get("best_checkpoint")
        last = run_info.get("last_checkpoint")
        test_metrics = run_info.get("test_metrics")
        if (
            not isinstance(best, dict)
            or best.get("path") != str(best_path)
            or best.get("sha256") != checkpoint_sha256
            or best.get("run_id") != checkpoint["run_id"]
            or best.get("best_epoch") != checkpoint.get("best_epoch")
            or best.get("selection_identity") != checkpoint.get("selection_identity")
            or best.get("validation") != checkpoint.get("best_validation")
            or not isinstance(last, dict)
            or last.get("path") != str(last_path)
            or last.get("sha256") != last_sha256
            or last.get("run_id") != last_checkpoint.get("run_id")
            or checkpoint.get("best_results_row") is None
            or not isinstance(test_metrics, dict)
            or test_metrics.get("evaluated_checkpoint") != str(best_path)
            or test_metrics.get("checkpoint_sha256") != best["sha256"]
            or test_metrics.get("run_id") != checkpoint["run_id"]
            or test_metrics.get("config_sha256") != checkpoint["config_sha256"]
            or test_metrics.get("snapshot_identity") != checkpoint["snapshot_identity"]
            or test_metrics.get("dataset_manifest_digest") != checkpoint["dataset_manifest_digest"]
            or test_metrics.get("best_epoch") != checkpoint.get("best_epoch")
            or test_metrics.get("selection_identity") != checkpoint.get("selection_identity")
        ):
            raise ValueError("run info best/test selection lineage is invalid")
        _validate_metrics_payload(test_metrics)

        with results_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != RESULTS_FIELDNAMES:
                raise ValueError("results.csv schema is invalid")
            rows = list(reader)
        best_epoch = checkpoint.get("best_epoch")
        if type(best_epoch) is not int or best_epoch < 1 or best_epoch > len(rows):
            raise ValueError("results.csv does not contain the selected best epoch")
        for index, row in enumerate(rows, start=1):
            _validate_results_row(
                [row[field] for field in RESULTS_FIELDNAMES],
                line_number=index + 1,
                expected_epoch=index,
            )
        best_row = rows[best_epoch - 1]
        if checkpoint["best_results_row"] != best_row:
            raise ValueError("results.csv selected row does not match checkpoint")
        best_validation = checkpoint["best_validation"]
        for column, metric in (
            ("val_exact_acc", "exact_acc"),
            ("val_within_one", "within_one"),
            ("val_ordinal_mae", "ordinal_mae"),
            ("val_error_ge_2", "error_ge_2"),
            ("val_macro_f1", "macro_f1"),
            ("val_balanced_accuracy", "balanced_accuracy"),
        ):
            if not math.isclose(float(best_row[column]), float(best_validation[metric]), rel_tol=0, abs_tol=1e-8):
                raise ValueError("results.csv best validation row does not match checkpoint")
        for column, metric in (("val_precision", "precision"), ("val_recall", "recall"), ("val_f1", "f1")):
            if json.loads(best_row[column]) != best_validation[metric]:
                raise ValueError("results.csv best validation row does not match checkpoint")
        gates = _evaluate_provisional_gates(test_metrics, config)
        if run_info.get("provisional_acceptance") != gates or not gates["passed"]:
            raise ValueError("selected best checkpoint does not pass provisional test gates")
        return {
            "checkpoint": str(best_path),
            "checkpoint_sha256": checkpoint_sha256,
            "category_contract": "1..5",
            "candidate_status": "provisional_candidate_not_production",
            "best_epoch": best_epoch,
            "selection_identity": checkpoint["selection_identity"],
            "test_metrics": test_metrics,
            "validation_gate": "provisional engineering gates passed; clinical validation required",
        }
    except Exception as error:
        raise OvernightError(f"best checkpoint validation failed: {error}") from None


def run(args: argparse.Namespace, *, root: Path = ROOT, environment: Mapping[str, str] | None = None, runner: Callable[..., int] = _run_subprocess, best_validator: Callable[..., dict[str, object]] = _validate_best_checkpoint) -> Plan:
    env = dict(os.environ if environment is None else environment)
    plan = preflight(args, root=root, approved_roots=(root,), environment=env)
    if args.preflight_only:
        print(f"READY: category preflight passed for {plan.root}; no subprocess was started.", flush=True)
        return plan
    run_dir = plan.logs / uuid.uuid4().hex
    if not plan.skip_build:
        _step("category snapshot build", [sys.executable, str(plan.root / "scripts/build_bcs_category.py"), "--local-root", str(plan.local_root), "--output", str(plan.snapshot)], plan=plan, environment=env, runner=runner, log_path=run_dir / "build.log")
    command = [sys.executable, "-u", str(plan.root / "scripts/train_bcs_ordinal.py"), "--config", str(plan.config)]
    if plan.resume:
        command.extend(("--resume", str(plan.checkpoint)))
    _step("category CORAL training", command, plan=plan, environment=env, runner=runner, log_path=run_dir / "train.log")
    evidence = best_validator(
        plan.output / BEST_CHECKPOINT,
        plan.output,
        plan.snapshot,
        plan.config,
        (plan.root / "data",),
        (plan.root / "configs",),
        (plan.root / "outputs",),
    )
    print(f"[category] Validated checkpoint: {evidence}", flush=True)
    if isinstance(evidence.get("checkpoint_sha256"), str):
        print(
            f"[category] Checkpoint SHA-256: {evidence['checkpoint_sha256']}",
            flush=True,
        )
    print("[category] Complete. API was not started.", flush=True)
    return plan


def main(argv: Sequence[str] | None = None, *, root: Path = ROOT, environment: Mapping[str, str] | None = None, runner: Callable[..., int] = _run_subprocess, stderr: TextIO | None = None, best_validator: Callable[..., dict[str, object]] = _validate_best_checkpoint) -> int:
    try:
        run(build_parser().parse_args(argv), root=root, environment=environment, runner=runner, best_validator=best_validator)
    except OvernightError as error:
        print(f"ERROR: {error}", file=stderr or sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
