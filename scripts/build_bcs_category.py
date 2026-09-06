"""Build the immutable local BCS category 1..5 snapshot."""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_ROOT = "data/bcs/dataset"
DEFAULT_OUTPUT = "data/bcs-category-v1"
DEFAULT_MAX_IMAGE_BYTES = 16 * 1024 * 1024
DATA_ROOT = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))

from vacca_bcs.category_snapshot import CategorySnapshotError, build_category_snapshot  # noqa: E402
from vacca_bcs.category_split_plan import CategorySplitPlanError, create_category_split_plan  # noqa: E402
from vacca_bcs.local_source import LOCAL_BCS_MAPPING, LocalSourceError, LocalSourceMaterializer, scan_local_source  # noqa: E402
from vacca_bcs.path_safety import SafePathError, safe_path  # noqa: E402
from vacca_bcs.source_plan import SourcePlanError, normalize_local_source_scan  # noqa: E402


class CategoryBuildCLIError(ValueError):
    pass


_TYPED_ERROR_CATEGORIES = ((LocalSourceError, "local-source"), (SourcePlanError, "source-plan"), (CategorySplitPlanError, "split-plan"), (CategorySnapshotError, "snapshot"))


def _report_failure(error: Exception, stream: TextIO) -> None:
    for error_type, category in _TYPED_ERROR_CATEGORIES:
        if isinstance(error, error_type):
            print(f"ERROR [{category}]: {error}", file=stream)
            return
    print(f"ERROR [unexpected]: {type(error).__name__}; correlation_id={uuid.uuid4().hex[:12]}", file=stream)


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise CategoryBuildCLIError("invalid command line")


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("byte limit must be a positive integer") from None
    if result <= 0:
        raise argparse.ArgumentTypeError("byte limit must be a positive integer")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(description="Build a transactional BCS category 1..5 snapshot")
    parser.add_argument("--local-root", default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-image-bytes", type=_positive_int, default=DEFAULT_MAX_IMAGE_BYTES)
    return parser


def run(args: argparse.Namespace, *, root: Path = ROOT) -> Any:
    data_root = DATA_ROOT if root == ROOT else root / "data"
    try:
        output = safe_path(
            args.output,
            base=root,
            approved_roots=(data_root,),
            allow_missing_final=True,
        )
        local_root = safe_path(
            args.local_root,
            base=root,
            approved_roots=(data_root,),
            allow_missing_final=False,
            require_dir=True,
        )
    except SafePathError as error:
        raise CategoryBuildCLIError(str(error)) from None
    scan = scan_local_source(local_root, LOCAL_BCS_MAPPING, approved_roots=(data_root,))
    if not scan.records or scan.observed_classes != (1, 2, 3, 4, 5):
        raise CategoryBuildCLIError("local source must contain all five BCS categories")
    plan = normalize_local_source_scan(scan)
    split = create_category_split_plan(plan, seed=args.seed)
    return build_category_snapshot(
        split,
        output,
        LocalSourceMaterializer(scan, max_bytes=args.max_image_bytes),
        approved_roots=(data_root,),
    )


def _summary(snapshot: Any) -> dict[str, object]:
    manifest = json.loads(snapshot.manifest_json)
    reason_counts: dict[str, int] = {}
    for exclusion in manifest["exclusions"]:
        reason_counts[exclusion["reason"]] = reason_counts.get(exclusion["reason"], 0) + 1
    return {
        "source_schema": manifest["source_schema"],
        "snapshot_schema": manifest["manifest_schema_version"],
        "snapshot_path": str(snapshot.output_root),
        "included": len(snapshot.records),
        "excluded": len(manifest["exclusions"]),
        "exclusion_reason_counts": reason_counts,
        "exclusion_inspection": "Inspect manifest.json exclusions by reason, source path, category, and digest before handoff.",
        "counts": manifest["counts"],
        "mapping": manifest["mapping"],
        "observed_classes": manifest["observed_classes"],
        "isolation": manifest["isolation"],
        "plan_identity": manifest["split_plan"]["identity_digest"],
    }


def main(argv: Sequence[str] | None = None, *, root: Path = ROOT, stdout: TextIO | None = None, stderr: TextIO | None = None) -> int:
    try:
        snapshot = run(build_parser().parse_args(argv), root=root)
    except CategoryBuildCLIError as error:
        print(f"ERROR: {error}", file=stderr or sys.stderr)
        return 1
    except Exception as error:
        _report_failure(error, stderr or sys.stderr)
        return 1
    print(json.dumps(_summary(snapshot), indent=2, sort_keys=True), file=stdout or sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
