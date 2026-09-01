"""Build an integer BCS snapshot from the authenticated source export."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TextIO

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = "data/bcs-integer-v1"
DEFAULT_LOCAL_ROOT = "data/bcs/dataset"
DEFAULT_LOCAL_OUTPUT = "data/bcs-local-integer-v1"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_SOURCE_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_IMAGE_BYTES = 16 * 1024 * 1024

sys.path.insert(0, str(ROOT / "src"))

from vacca_bcs.integer_snapshot import (  # noqa: E402
    IntegerSnapshotError,
    build_integer_snapshot,
)
from vacca_bcs.local_source import (  # noqa: E402
    LOCAL_BCS_MAPPING,
    LocalSourceError,
    LocalSourceMaterializer,
    scan_local_source,
)
from vacca_bcs.source_client import (  # noqa: E402
    BCSEvidenceMaterializer,
    BCSSourceClient,
    BCSSourceClientError,
)
from vacca_bcs.source_plan import (  # noqa: E402
    SourcePlanError,
    normalize_backend_source_export,
    normalize_local_source_scan,
)
from vacca_bcs.source_split_plan import (  # noqa: E402
    IntegerSplitPlanError,
    create_integer_split_plan,
)
class IntegerBuildCLIError(ValueError):
    """Raised for safe, operator-facing CLI failures."""


_TYPED_ERROR_CATEGORIES = (
    (LocalSourceError, "local-source"),
    (BCSSourceClientError, "source-client"),
    (SourcePlanError, "source-plan"),
    (IntegerSplitPlanError, "split-plan"),
    (IntegerSnapshotError, "snapshot"),
)


def _report_failure(error: Exception, stream: TextIO) -> None:
    for error_type, category in _TYPED_ERROR_CATEGORIES:
        if isinstance(error, error_type):
            print(f"ERROR [{category}]: {error}", file=stream)
            return
    correlation_id = uuid.uuid4().hex[:12]
    print(
        "ERROR [unexpected]: "
        f"{type(error).__name__}; correlation_id={correlation_id}; "
        "inspect the local operator log for details",
        file=stream,
    )


def _finite_ratio(value: str) -> float:
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("val-ratio must be a finite number in [0, 1)") from None
    if not math.isfinite(ratio) or not 0 <= ratio < 1:
        raise argparse.ArgumentTypeError("val-ratio must be a finite number in [0, 1)")
    return ratio


def _positive_float(value: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("timeout must be a finite number greater than zero") from None
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("timeout must be a finite number greater than zero")
    return result


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("byte limits must be positive integers") from None
    if result <= 0:
        raise argparse.ArgumentTypeError("byte limits must be positive integers")
    return result


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise IntegerBuildCLIError("invalid command line")


class _TrackedAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        explicit = set(getattr(namespace, "_explicit_options", ()))
        explicit.add(self.dest)
        setattr(namespace, "_explicit_options", explicit)


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(description="Build a transactional integer BCS snapshot")
    parser.set_defaults(_explicit_options=set())
    parser.add_argument("--source", choices=("local", "backend"), default="local")
    parser.add_argument("--local-root", default=DEFAULT_LOCAL_ROOT, action=_TrackedAction)
    parser.add_argument("--base-url", default=None, action=_TrackedAction, help="Backend URL; defaults to VACCA_BACKEND_URL")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=_finite_ratio, default=0.2)
    parser.add_argument("--timeout", type=_positive_float, default=DEFAULT_TIMEOUT, action=_TrackedAction)
    parser.add_argument("--max-source-bytes", type=_positive_int, default=DEFAULT_MAX_SOURCE_BYTES, action=_TrackedAction)
    parser.add_argument("--max-image-bytes", type=_positive_int, default=DEFAULT_MAX_IMAGE_BYTES)
    return parser


def _backend_inputs(args: argparse.Namespace) -> tuple[str, str]:
    base_url = args.base_url or os.environ.get("VACCA_BACKEND_URL")
    token = os.environ.get("VACCA_BACKEND_TOKEN")
    if not isinstance(base_url, str) or not base_url.strip():
        raise IntegerBuildCLIError("backend URL is required via --base-url or VACCA_BACKEND_URL")
    if not isinstance(token, str) or not token.strip():
        raise IntegerBuildCLIError("VACCA_BACKEND_TOKEN is required")
    return base_url, token


def _output_path(args: argparse.Namespace) -> Path:
    default = DEFAULT_LOCAL_OUTPUT if args.source == "local" else DEFAULT_OUTPUT
    output = Path(args.output or default)
    return output if output.is_absolute() else (ROOT / output).resolve()


def _validate_mode(args: argparse.Namespace) -> None:
    explicit = getattr(args, "_explicit_options", set())
    if args.source == "local":
        incompatible = {"base_url", "timeout", "max_source_bytes"}.intersection(explicit)
        if incompatible:
            raise IntegerBuildCLIError("backend options are not valid for local source")
    elif "local_root" in explicit:
        raise IntegerBuildCLIError("--local-root is only valid for local source")


def run(
    args: argparse.Namespace,
    *,
    source_client_factory: Callable[..., Any] = BCSSourceClient,
    materializer_factory: Callable[..., Any] = BCSEvidenceMaterializer,
    snapshot_builder: Callable[..., Any] = build_integer_snapshot,
) -> Any:
    _validate_mode(args)
    output = _output_path(args)
    if args.source == "local":
        scan = scan_local_source(args.local_root, LOCAL_BCS_MAPPING)
        if not scan.records:
            raise IntegerBuildCLIError("local source contains no image records")
        source_plan = normalize_local_source_scan(scan)
        split_plan = create_integer_split_plan(
            source_plan, seed=args.seed, val_ratio=args.val_ratio
        )
        return snapshot_builder(
            split_plan,
            output,
            LocalSourceMaterializer(scan, max_bytes=args.max_image_bytes),
        )
    base_url, token = _backend_inputs(args)
    with source_client_factory(
        base_url=base_url,
        bearer_token=token,
        timeout=args.timeout,
        max_response_bytes=args.max_source_bytes,
    ) as client:
        source_plan = normalize_backend_source_export(client.fetch())
        split_plan = create_integer_split_plan(
            source_plan, seed=args.seed, val_ratio=args.val_ratio
        )
        with materializer_factory(
            backend_base_url=base_url,
            bearer_token=token,
            timeout=args.timeout,
            max_image_bytes=args.max_image_bytes,
        ) as materializer:
            return snapshot_builder(split_plan, output, materializer)


def _summary(snapshot: Any) -> dict[str, object]:
    manifest = json.loads(snapshot.manifest_json)
    observed = manifest.get("observed_classes")
    if observed is None:
        counts = manifest["counts"]
        observed = [
            score
            for score in range(1, 6)
            if counts["train"][score - 1] or counts["val"][score - 1]
        ]
    mapping = manifest.get("mapping")
    return {
        "source_schema": manifest["source_schema"],
        "snapshot_path": str(snapshot.output_root),
        "included": len(snapshot.records),
        "excluded": len(snapshot.exclusions),
        "counts": manifest["counts"],
        "mapping": mapping,
        "observed_classes": observed,
        "missing_classes": [score for score in range(1, 6) if score not in observed],
        "plan_identity": manifest["split_plan"]["identity_digest"],
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    source_client_factory: Callable[..., Any] = BCSSourceClient,
    materializer_factory: Callable[..., Any] = BCSEvidenceMaterializer,
    snapshot_builder: Callable[..., Any] = build_integer_snapshot,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    try:
        snapshot = run(
            build_parser().parse_args(argv),
            source_client_factory=source_client_factory,
            materializer_factory=materializer_factory,
            snapshot_builder=snapshot_builder,
        )
    except IntegerBuildCLIError as error:
        print(f"ERROR: {error}", file=stderr or sys.stderr)
        return 1
    except Exception as error:
        _report_failure(error, stderr or sys.stderr)
        return 1
    print(json.dumps(_summary(snapshot), indent=2, sort_keys=True), file=stdout or sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
