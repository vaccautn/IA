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

from .source_plan import (
    BackendSourceProvenance,
    LocalSourceProvenance,
    SourceCandidate,
    SourceExclusion,
    SourcePlan,
    SourceRecord,
    SourceProvenance,
)


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
    evidence_id: int | None
    provenance: tuple[SourceProvenance | BackendSourceProvenance | LocalSourceProvenance, ...]
    storage_key: str | None
    relative_path_stem: str
    record_id: str = ""

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
    candidate_record_ids: tuple[str, ...] = ()
    source_schema: str = "bcs-source-v1"
    identity_scheme: str = "backend-evidence-v1"
    mapping_lineage: tuple[tuple[str, int], ...] = ()
    observed_classes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_evidence_ids", tuple(self.candidate_evidence_ids)
        )
        object.__setattr__(
            self, "excluded_evidence_ids", tuple(self.excluded_evidence_ids)
        )
        object.__setattr__(self, "candidate_record_ids", tuple(self.candidate_record_ids))
        object.__setattr__(self, "mapping_lineage", tuple(self.mapping_lineage))
        object.__setattr__(self, "observed_classes", tuple(self.observed_classes))


@dataclass(frozen=True, slots=True)
class IntegerSplitPlan:
    assignments: tuple[IntegerSplitAssignment, ...]
    exclusions: tuple[SourceExclusion, ...]
    counts: IntegerSplitCounts
    config: IntegerSplitConfig
    identity: IntegerSplitPlanIdentity
    source_candidates: tuple[SourceCandidate | SourceRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignments", tuple(self.assignments))
        object.__setattr__(self, "exclusions", tuple(self.exclusions))
        object.__setattr__(self, "source_candidates", tuple(self.source_candidates))


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


def _candidate_id(candidate: SourceCandidate | SourceRecord) -> str:
    if isinstance(candidate, SourceRecord):
        return candidate.record_id
    return f"backend/evidence/{candidate.evidence_id}"


def _candidate_score(candidate: SourceCandidate | SourceRecord) -> int:
    return candidate.bcs_score


def _candidate_sort_key(candidate: SourceCandidate | SourceRecord) -> tuple[int, object]:
    return (
        candidate.bcs_score,
        candidate.evidence_id
        if isinstance(candidate, SourceCandidate)
        else candidate.record_id,
    )


def _candidate_provenance(candidate: SourceCandidate | SourceRecord) -> tuple[object, ...]:
    return tuple(candidate.provenance)


def _validate_input(
    source_plan: BCSIntegerSourcePlan,
) -> tuple[SourceCandidate | SourceRecord, ...]:
    if not isinstance(source_plan, SourcePlan):
        raise IntegerSplitInputError("input must be a normalized SourcePlan")
    candidates = tuple(source_plan.candidates)
    if source_plan.source_schema not in {"bcs-source-v1", "bcs-local-folder-v1"}:
        raise IntegerSplitInputError("source plan schema is not supported")
    record_ids = [_candidate_id(candidate) for candidate in candidates]
    if any(not isinstance(record_id, str) or not record_id for record_id in record_ids):
        raise IntegerSplitInputError("source records must have stable record IDs")
    if len(set(record_ids)) != len(record_ids):
        raise IntegerSplitInputError(
            "normalized source plan has duplicate evidence IDs"
            if all(isinstance(candidate, SourceCandidate) for candidate in candidates)
            else "normalized source plan has duplicate record IDs"
        )
    exclusion_ids = [item.evidence_id for item in source_plan.exclusions]
    duplicate_exclusions = tuple(
        sorted(item for item, count in Counter(exclusion_ids).items() if count > 1)
    )
    if duplicate_exclusions:
        raise IntegerSplitInputError(
            f"duplicate exclusion evidence IDs: {duplicate_exclusions}"
        )
    candidate_evidence_ids = {
        candidate.evidence_id
        for candidate in candidates
        if isinstance(candidate, SourceCandidate)
    }
    overlap = tuple(sorted(candidate_evidence_ids.intersection(exclusion_ids)))
    if overlap:
        raise IntegerSplitInputError(
            f"evidence IDs appear in both candidates and exclusions: {overlap}"
        )
    for candidate in candidates:
        if type(candidate.bcs_score) is not int or not 1 <= candidate.bcs_score <= 5:
            raise IntegerSplitInputError(
                "source plan labels must be integer scores in 1..5"
            )
        if isinstance(candidate, SourceRecord) and candidate.source_schema != source_plan.source_schema:
            raise IntegerSplitInputError("source record schema does not match source plan")
        if isinstance(candidate, SourceCandidate) and type(candidate.evidence_id) is not int:
            raise IntegerSplitInputError("source plan evidence IDs must be integers")
    if source_plan.source_schema == "bcs-local-folder-v1" and any(
        not isinstance(candidate, SourceRecord) for candidate in candidates
    ):
        raise IntegerSplitInputError("local source plan contains backend records")
    return candidates


def _validation_count(size: int, canonical_ratio: str) -> int:
    ratio = Decimal(canonical_ratio)
    if ratio == 0.0 or size < 2:
        return 0
    floor_count = int((Decimal(size) * ratio).to_integral_value(rounding=ROUND_FLOOR))
    return min(size - 1, max(1, floor_count))


def _assignment(
    candidate: SourceCandidate | SourceRecord,
    split: str,
) -> IntegerSplitAssignment:
    record_id = _candidate_id(candidate)
    backend = isinstance(candidate, SourceCandidate)
    evidence_id = candidate.evidence_id if backend else None
    storage_key = candidate.storage_key if backend else None
    return IntegerSplitAssignment(
        split=split,
        bcs_score=_candidate_score(candidate),
        evidence_id=evidence_id,
        provenance=_candidate_provenance(candidate),
        storage_key=storage_key,
        relative_path_stem=f"{split}/{candidate.bcs_score}/{candidate.evidence_id}"
        if backend
        else f"{split}/{candidate.bcs_score}/{record_id}",
        record_id=record_id,
    )


def _identity_digest(
    config: IntegerSplitConfig,
    candidates: tuple[SourceCandidate | SourceRecord, ...],
    exclusions: tuple[SourceExclusion, ...],
    assignments: tuple[IntegerSplitAssignment, ...],
    counts: IntegerSplitCounts,
    source_plan: SourcePlan,
) -> str:
    def provenance_value(item: object) -> tuple[object, ...]:
        if isinstance(item, SourceProvenance):
            return (item.evidence_id, item.evaluation_id)
        if isinstance(item, BackendSourceProvenance):
            return (item.evidence_id, item.evaluation_id, item.session_id, item.animal_id)
        if isinstance(item, LocalSourceProvenance):
            return (item.relative_path, item.source_label)
        raise IntegerSplitInputError("source provenance type is unsupported")

    payload = {
        "source": (
            source_plan.source_schema,
            source_plan.identity_scheme,
            source_plan.mapping_lineage,
            source_plan.observed_classes,
        ),
        "assignments": [
            (
                item.split,
                item.bcs_score,
                item.record_id,
                item.evidence_id,
                tuple(provenance_value(p) for p in item.provenance),
                item.storage_key,
                item.relative_path_stem,
            )
            for item in assignments
        ],
        "candidates": [
            (
                _candidate_id(item),
                getattr(item, "evaluation_id", None),
                getattr(item, "session_id", None),
                getattr(item, "animal_id", None),
                getattr(item, "materializer_key", None),
                item.bcs_score,
                tuple(provenance_value(p) for p in item.provenance),
            )
            for item in sorted(
                candidates, key=_candidate_sort_key
            )
        ],
        "config": (config.seed, config.canonical_val_ratio),
        "counts": {"train": counts.train, "val": counts.val},
        "exclusions": sorted(
            (
                getattr(item, "evidence_id", None),
                getattr(item, "evaluation_id", None),
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
    for candidate in sorted(candidates, key=_candidate_sort_key):
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
                item.evidence_id if item.evidence_id is not None else item.record_id,
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
            for item in sorted(candidates, key=_candidate_sort_key)
            if isinstance(item, SourceCandidate)
        ),
        excluded_evidence_ids=excluded_ids,
        digest=_identity_digest(
            config,
            candidates,
            source_plan.exclusions,
            ordered_assignments,
            counts,
            source_plan,
        ),
        candidate_record_ids=tuple(
            _candidate_id(item) for item in sorted(candidates, key=_candidate_sort_key)
        ),
        source_schema=source_plan.source_schema,
        identity_scheme=source_plan.identity_scheme,
        mapping_lineage=source_plan.mapping_lineage,
        observed_classes=source_plan.observed_classes,
    )
    return IntegerSplitPlan(
        assignments=ordered_assignments,
        exclusions=source_plan.exclusions,
        counts=counts,
        config=config,
        identity=identity,
        source_candidates=tuple(
            sorted(candidates, key=lambda item: (item.bcs_score, _candidate_id(item)))
        ),
    )


plan_integer_splits = create_integer_split_plan


def validate_integer_split_plan(plan: IntegerSplitPlan) -> None:
    """Recompute and validate every contract-relevant part of a split plan."""
    if not isinstance(plan, IntegerSplitPlan):
        raise IntegerSplitInputError("input must be an integer split plan")
    source = SourcePlan(
        plan.source_candidates,
        plan.exclusions,
        (0,) * 5,
        source_schema=plan.identity.source_schema,
        identity_scheme=plan.identity.identity_scheme,
        mapping_lineage=plan.identity.mapping_lineage,
        observed_classes=plan.identity.observed_classes,
    )
    expected = create_integer_split_plan(
        source, seed=plan.config.seed, val_ratio=plan.config.val_ratio
    )
    if (
        plan.assignments != expected.assignments
        or plan.exclusions != expected.exclusions
        or plan.counts != expected.counts
        or plan.identity != expected.identity
    ):
        raise IntegerSplitInputError("integer split plan integrity validation failed")
