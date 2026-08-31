"""Safely prepare and run the bounded overnight integer BCS workflow."""
from __future__ import annotations

import argparse
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
SNAPSHOT_RELATIVE = Path("data/bcs-integer-v1")
OUTPUT_RELATIVE = Path("outputs/bcs-ordinal-integer-v1")
CONFIG_RELATIVE = Path("configs/training_bcs_ordinal.yaml")
MANIFEST_NAME = "manifest.json"
LAST_CHECKPOINT = Path("weights/last.pt")
BEST_CHECKPOINT = Path("weights/best.pt")
RESULTS_LINEAGE = "results_lineage.json"


class OvernightError(RuntimeError):
    """Raised for safe, operator-facing orchestration failures."""


@dataclass(frozen=True)
class Plan:
    root: Path
    config: Path
    snapshot: Path
    output: Path
    checkpoint: Path
    logs: Path
    skip_build: bool
    resume: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight and run the integer BCS snapshot/training workflow"
    )
    parser.add_argument("--config", default=str(CONFIG_RELATIVE))
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--logs-root", default="logs/bcs-overnight")
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
    snapshot = (root / SNAPSHOT_RELATIVE).resolve()
    output = (root / OUTPUT_RELATIVE).resolve()
    for key, expected in (("data_dir", snapshot), ("output", output)):
        raw = values.get(key)
        if not isinstance(raw, str) or _path(raw, root) != expected:
            raise OvernightError(f"training config {key} disagrees with the canonical root")

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
    if not args.skip_build:
        env = _environment(environment)
        if not env.get("VACCA_BACKEND_URL", "").strip():
            raise OvernightError("VACCA_BACKEND_URL is required for the snapshot build")
        if not env.get("VACCA_BACKEND_TOKEN", "").strip():
            raise OvernightError("VACCA_BACKEND_TOKEN is required for the snapshot build")
    return Plan(
        root=root,
        config=config,
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


def _run_subprocess(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    log_path: Path,
) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=dict(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            safe = _redact(line, environment)
            log.write(safe)
            log.flush()
            print(safe, end="")
        return process.wait()


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
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch()
    print(f"[overnight] {name}: " + " ".join(_redact(str(part), environment) for part in command))
    try:
        status = runner(command, cwd=plan.root, environment=environment, log_path=log_path)
    except Exception:
        _sanitize_log(log_path, environment)
        raise OvernightError(f"{name} could not start; see {log_path}") from None
    _sanitize_log(log_path, environment)
    if status:
        raise OvernightError(f"{name} failed with exit code {status}; see {log_path}")


def run(
    args: argparse.Namespace,
    *,
    root: Path = ROOT,
    environment: Mapping[str, str] | None = None,
    runner: Callable[..., int] = _run_subprocess,
) -> Plan:
    env = _environment(environment)
    plan = preflight(args, root=root, environment=env)
    if args.preflight_only:
        print(f"READY: preflight passed for {plan.root}; no subprocess was started.")
        return plan
    run_dir = plan.logs / uuid.uuid4().hex
    build_log = run_dir / "build.log"
    train_log = run_dir / "train.log"
    if not plan.skip_build:
        _step(
            "snapshot build",
            [sys.executable, str(plan.root / "scripts" / "build_bcs_integer.py"), "--output", str(plan.snapshot)],
            plan=plan,
            environment=env,
            runner=runner,
            log_path=build_log,
        )
    command = [
        sys.executable,
        str(plan.root / "scripts" / "train_bcs_ordinal.py"),
        "--config",
        str(plan.config),
    ]
    if plan.resume:
        command.extend(("--resume", str(plan.checkpoint)))
    _step("ordinal training", command, plan=plan, environment=env, runner=runner, log_path=train_log)
    best = plan.output / BEST_CHECKPOINT
    if not best.is_file():
        raise OvernightError(f"training succeeded but lineage-compatible best checkpoint is missing: {best}")
    print("[overnight] Complete. API was not started.")
    print(f"Morning: set VACCA_BCS_CHECKPOINT to {best}")
    print("Morning: start the API, then verify /ready/bcs and /bcs.")
    return plan


def main(
    argv: Sequence[str] | None = None,
    *,
    root: Path = ROOT,
    environment: Mapping[str, str] | None = None,
    runner: Callable[..., int] = _run_subprocess,
    stderr: TextIO | None = None,
) -> int:
    try:
        run(build_parser().parse_args(argv), root=root, environment=environment, runner=runner)
    except OvernightError as error:
        print(f"ERROR: {_redact(str(error), _environment(environment))}", file=stderr or sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
