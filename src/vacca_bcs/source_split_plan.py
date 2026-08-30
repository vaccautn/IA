"""Pure deterministic train/validation planning for integer BCS records.

For a non-zero validation ratio, the validation count is ``floor(n * ratio)``
clamped to ``[1, n - 1]`` for classes with at least two records. A singleton
always stays in training, and a zero ratio assigns every record to training.
"""

from __future__ import annotations

import math
import random
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from numbers import Real

from .source_plan import SourceCandidate, SourceExclusion, SourcePlan, SourceProvenance


class IntegerSplitPlanError(Exception):
    """Base error for integer source split planning."""


class IntegerSplitConfigError(IntegerSplitPlanError):
    """Raised when the split configuration is not deterministic and valid."""


class IntegerSplitInputError(IntegerSplitPlanError):
    """Raised when the normalized source plan violates the integer contract."""


@dataclass(frozen=True, slots=True)
class IntegerSplitConfig:
    seed: int
    val_ratio: float
    canonical_val_ratio: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_seed(self.seed)
        ratio, canonical_ratio = _validate_ratio(self.val_ratio)
        object.__setattr__(self, "val_ratio", ratio)
        object.__setattr__(self, "canonical_val_ratio", canonical_ratio)


@dataclass(frozen=True, slots=True)
class IntegerSplitCounts:
    train: tuple[int, int, int, int, int]
    val: tuple[int, int, int, int, int]

    def __post_init__(self) -> None:
        train = tuple(self.train)
        val = tuple(self.val)
        if len(train) != 5 or len(val) != 5:
            raise ValueError("integer split counts must contain classes 1..5")
        object.__setattr__(self, "train", train)
        object.__setattr__(self, "val", val)


@dataclass(frozen=True, slots=True)
class IntegerSplitAssignment:
    split: str
    bcs_score: int
    evidence_id: int
    provenance: tuple[SourceProvenance, ...]
    storage_key: str
    relative_path_stem: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", tuple(self.provenance))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(split={self.split!r}, "
            f"bcs_score={self.bcs_score!r}, evidence_id={self.evidence_id!r}, "
            f"provenance={self.provenance!r}, "
            f"relative_path_stem={self.relative_path_stem!r})"
        )


@dataclass(frozen=True, slots=True)
class IntegerSplitPlanIdentity:
    seed: int
    val_ratio: float
    canonical_val_ratio: str
    candidate_evidence_ids: tuple[int, ...]
    excluded_evidence_ids: tuple[int, ...]
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_evidence_ids", tuple(self.candidate_evidence_ids)
        )
        object.__setattr__(
            self, "excluded_evidence_ids", tuple(self.excluded_evidence_ids)
        )


@dataclass(frozen=True, slots=True)
class IntegerSplitPlan:
    assignments: tuple[IntegerSplitAssignment, ...]
    exclusions: tuple[SourceExclusion, ...]
    counts: IntegerSplitCounts
    config: IntegerSplitConfig
    identity: IntegerSplitPlanIdentity

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignments", tuple(self.assignments))
        object.__setattr__(self, "exclusions", tuple(self.exclusions))


BCSIntegerSourcePlan = SourcePlan


def _validate_seed(seed: int) -> None:
    if type(seed) is not int:
        raise IntegerSplitConfigError("seed must be an integer, not bool")


def _validate_ratio(val_ratio: Real) -> tuple[float, str]:
    if isinstance(val_ratio, bool) or not isinstance(val_ratio, Real):
        raise IntegerSplitConfigError("val_ratio must be a finite real number")
    ratio = float(val_ratio)
    if not math.isfinite(ratio) or not 0.0 <= ratio < 1.0:
        raise IntegerSplitConfigError("val_ratio must be finite and in [0, 1)")
    try:
        decimal_ratio = Decimal(str(val_ratio))
    except InvalidOperation:
        decimal_ratio = Decimal(str(ratio))
    if decimal_ratio == 0:
        decimal_ratio = Decimal(0)
    return ratio, format(decimal_ratio.normalize(), "f")


def _validate_input(source_plan: BCSIntegerSourcePlan) -> tuple[SourceCandidate, ...]:
    if not isinstance(source_plan, SourcePlan):
        raise IntegerSplitInputError("input must be a normalized SourcePlan")
    candidates = tuple(source_plan.candidates)
    evidence_ids = [candidate.evidence_id for candidate in candidates]
    if len(set(evidence_ids)) != len(evidence_ids):
        raise IntegerSplitInputError(
            "normalized source plan has duplicate evidence IDs"
        )
    exclusion_ids = [item.evidence_id for item in source_plan.exclusions]
    duplicate_exclusions = tuple(
        sorted(item for item, count in Counter(exclusion_ids).items() if count > 1)
    )
    if duplicate_exclusions:
        raise IntegerSplitInputError(
            f"duplicate exclusion evidence IDs: {duplicate_exclusions}"
        )
    overlap = tuple(sorted(set(evidence_ids).intersection(exclusion_ids)))
    if overlap:
        raise IntegerSplitInputError(
            f"evidence IDs appear in both candidates and exclusions: {overlap}"
        )
    for candidate in candidates:
        if type(candidate.bcs_score) is not int or not 1 <= candidate.bcs_score <= 5:
            raise IntegerSplitInputError(
                "source plan labels must be integer scores in 1..5"
            )
        if type(candidate.evidence_id) is not int:
            raise IntegerSplitInputError("source plan evidence IDs must be integers")
    return candidates


def _validation_count(size: int, canonical_ratio: str) -> int:
    ratio = Decimal(canonical_ratio)
    if ratio == 0.0 or size < 2:
        return 0
    floor_count = int((Decimal(size) * ratio).to_integral_value(rounding=ROUND_FLOOR))
    return min(size - 1, max(1, floor_count))


def _assignment(
    candidate: SourceCandidate,
    split: str,
) -> IntegerSplitAssignment:
    return IntegerSplitAssignment(
        split=split,
        bcs_score=candidate.bcs_score,
        evidence_id=candidate.evidence_id,
        provenance=candidate.provenance,
        storage_key=candidate.storage_key,
        relative_path_stem=f"{split}/{candidate.bcs_score}/{candidate.evidence_id}",
    )


def _identity_digest(
    config: IntegerSplitConfig,
    candidates: tuple[SourceCandidate, ...],
    exclusions: tuple[SourceExclusion, ...],
    assignments: tuple[IntegerSplitAssignment, ...],
    counts: IntegerSplitCounts,
) -> str:
    payload = {
        "assignments": [
            (
                item.split,
                item.bcs_score,
                item.evidence_id,
                tuple((p.evidence_id, p.evaluation_id) for p in item.provenance),
                item.storage_key,
                item.relative_path_stem,
            )
            for item in assignments
        ],
        "candidates": [
            (
                item.evaluation_id,
                item.session_id,
                item.animal_id,
                item.evidence_id,
                item.storage_key,
                item.bcs_score,
                tuple((p.evidence_id, p.evaluation_id) for p in item.provenance),
            )
            for item in sorted(
                candidates, key=lambda item: (item.bcs_score, item.evidence_id)
            )
        ],
        "config": (config.seed, config.canonical_val_ratio),
        "counts": {"train": counts.train, "val": counts.val},
        "exclusions": sorted(
            (
                item.evidence_id,
                item.evaluation_id,
                item.bcs_score,
                item.reason,
            )
            for item in exclusions
        ),
    }
    serialized = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def create_integer_split_plan(
    source_plan: BCSIntegerSourcePlan,
    *,
    seed: int,
    val_ratio: Real,
) -> IntegerSplitPlan:
    """Create a deterministic, stratified, immutable split plan."""
    config = IntegerSplitConfig(seed=seed, val_ratio=val_ratio)
    candidates = _validate_input(source_plan)
    by_score = {score: [] for score in range(1, 6)}
    for candidate in sorted(
        candidates, key=lambda item: (item.bcs_score, item.evidence_id)
    ):
        by_score[candidate.bcs_score].append(candidate)

    assignments: list[IntegerSplitAssignment] = []
    train_counts = [0] * 5
    val_counts = [0] * 5
    for score in range(1, 6):
        class_candidates = by_score[score]
        random.Random(config.seed + score).shuffle(class_candidates)
        val_count = _validation_count(len(class_candidates), config.canonical_val_ratio)
        for index, candidate in enumerate(class_candidates):
            split = "val" if index < val_count else "train"
            assignments.append(_assignment(candidate, split))
            counts = val_counts if split == "val" else train_counts
            counts[score - 1] += 1

    ordered_assignments = tuple(
        sorted(
            assignments,
            key=lambda item: (
                item.bcs_score,
                0 if item.split == "train" else 1,
                item.evidence_id,
            ),
        )
    )
    counts = IntegerSplitCounts(tuple(train_counts), tuple(val_counts))
    excluded_ids = tuple(sorted(item.evidence_id for item in source_plan.exclusions))
    identity = IntegerSplitPlanIdentity(
        seed=config.seed,
        val_ratio=config.val_ratio,
        canonical_val_ratio=config.canonical_val_ratio,
        candidate_evidence_ids=tuple(
            item.evidence_id
            for item in sorted(
                candidates, key=lambda item: (item.bcs_score, item.evidence_id)
            )
        ),
        excluded_evidence_ids=excluded_ids,
        digest=_identity_digest(
            config,
            candidates,
            source_plan.exclusions,
            ordered_assignments,
            counts,
        ),
    )
    return IntegerSplitPlan(
        assignments=ordered_assignments,
        exclusions=source_plan.exclusions,
        counts=counts,
        config=config,
        identity=identity,
    )


plan_integer_splits = create_integer_split_plan
