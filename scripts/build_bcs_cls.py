"""CLI adapter for the ordinal BCS dataset builder."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vacca_bcs.constants import (  # noqa: E402
    DEFAULT_MAX_PER_CLASS,
    DEFAULT_SEED,
    DEFAULT_VAL_RATIO,
)
from vacca_bcs.dataset_transaction import (  # noqa: E402
    DatasetInstallError,
    DatasetRecoveryRequiredError,
    DatasetRollbackError,
    build_dataset,
)


def _resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (ROOT / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the ordinal BCS folder dataset")
    parser.add_argument("--max-per-class", type=int, default=DEFAULT_MAX_PER_CLASS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--out-dir", default="data/bcs-cls")
    parser.add_argument("--bcs-dir", default="data/bcs/dataset")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        per_class, totals = build_dataset(
            _resolve_path(args.bcs_dir),
            _resolve_path(args.out_dir),
            max_per_class=args.max_per_class,
            seed=args.seed,
            val_ratio=args.val_ratio,
        )
    except (DatasetInstallError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    for class_name, counts in per_class.items():
        print(
            f"  {class_name}: {counts['selected']} selected "
            f"({counts['train']} train, {counts['val']} val), "
            f"{counts['staged']} staged, {counts['added']} added, "
            f"{counts['updated']} updated, {counts['unchanged']} unchanged, "
            f"{counts['stale']} stale"
        )
    print("\n[READY] Ordinal BCS dataset created.")
    print(f"  Total selected: {totals['selected']}")
    print(f"  Train: {totals['train']}")
    print(f"  Val: {totals['val']}")
    print(
        f"  Staged: {totals['staged']}; added: {totals['added']}; "
        f"updated: {totals['updated']}; unchanged: {totals['unchanged']}; "
        f"stale: {totals['stale']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
