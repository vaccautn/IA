"""Deterministic capture-group-aware 80/10/10 BCS category planning."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Real

from .constants import BCS_CLASS_SCORES, NUM_CLASSES, SPLITS
from .source_plan import LocalSourceProvenance, SourceExclusion, SourcePlan, SourceRecord

SPLIT_OVERSHOOT_WEIGHT = 2
"""Stable business rule: overshooting a per-class split target costs twice as much."""


class CategorySplitPlanError(Exception):
    """Base error for category split planning."""


class CategorySplitConfigError(CategorySplitPlanError):
    pass


class CategorySplitInputError(CategorySplitPlanError):
    pass


@dataclass(frozen=True, slots=True)
class CategorySplitConfig:
    seed: int
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    canonical_val_ratio: str = field(init=False)
    canonical_test_ratio: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise CategorySplitConfigError("seed must be an integer, not bool")
        values = []
        for name, value in (("val_ratio", self.val_ratio), ("test_ratio", self.test_ratio)):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                raise CategorySplitConfigError(f"{name} must be a finite number")
            if not 0 <= float(value) < 1:
                raise CategorySplitConfigError(f"{name} must be in [0, 1)")
            values.append(float(value))
        if sum(values) >= 1:
            raise CategorySplitConfigError("validation and test ratios must leave training data")
        object.__setattr__(self, "val_ratio", values[0])
        object.__setattr__(self, "test_ratio", values[1])
        object.__setattr__(self, "canonical_val_ratio", _canonical(values[0]))
        object.__setattr__(self, "canonical_test_ratio", _canonical(values[1]))


@dataclass(frozen=True, slots=True)
class CategorySplitCounts:
    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in SPLITS:
            values = tuple(getattr(self, name))
            if len(values) != NUM_CLASSES or any(type(value) is not int or value < 0 for value in values):
                raise ValueError(f"category split counts must contain {NUM_CLASSES} non-negative classes")
            object.__setattr__(self, name, values)


@dataclass(frozen=True, slots=True)
class CategorySplitAssignment:
    split: str
    bcs_category: int
    record_id: str
    capture_group: str
    sha256: str
    provenance: tuple[LocalSourceProvenance, ...]
    relative_path_stem: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", tuple(self.provenance))


@dataclass(frozen=True, slots=True)
class CategorySplitPlanIdentity:
    seed: int
    val_ratio: float
    test_ratio: float
    canonical_val_ratio: str
    canonical_test_ratio: str
    candidate_record_ids: tuple[str, ...]
    excluded_record_ids: tuple[str, ...]
    digest: str
    source_schema: str
    identity_scheme: str
    mapping_lineage: tuple[tuple[str, int], ...]
    observed_classes: tuple[int, ...]

    def __post_init__(self) -> None:
        for field_name in ("candidate_record_ids", "excluded_record_ids", "mapping_lineage", "observed_classes"):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))


@dataclass(frozen=True, slots=True)
class CategorySplitPlan:
    assignments: tuple[CategorySplitAssignment, ...]
    exclusions: tuple[SourceExclusion, ...]
    counts: CategorySplitCounts
    config: CategorySplitConfig
    identity: CategorySplitPlanIdentity
    source_candidates: tuple[SourceRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignments", tuple(self.assignments))
        object.__setattr__(self, "exclusions", tuple(self.exclusions))
        object.__setattr__(self, "source_candidates", tuple(self.source_candidates))


def _canonical(value: float) -> str:
    return "0" if value == 0 else format(value, ".15g")


def split_identity_payload(
    *,
    source_schema: str,
    identity_scheme: str,
    mapping_lineage: object,
    observed_classes: object,
    seed: int,
    canonical_val_ratio: str,
    canonical_test_ratio: str,
    assignments: list[tuple[object, ...]],
    counts: dict[str, tuple[int, ...] | list[int]],
    exclusions: list[tuple[object, ...]],
) -> dict[str, object]:
    """Return the one canonical payload used to identify a category split."""
    return {
        "source": (
            source_schema,
            identity_scheme,
            tuple(sorted(dict(mapping_lineage).items())),
            tuple(observed_classes),
        ),
        "config": (seed, canonical_val_ratio, canonical_test_ratio),
        "assignments": assignments,
        "counts": {name: tuple(counts[name]) for name in SPLITS},
        "exclusions": exclusions,
    }


def split_identity_digest(**kwargs: object) -> str:
    """Hash the canonical split identity payload."""
    return hashlib.sha256(
        json.dumps(split_identity_payload(**kwargs), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_input(source: SourcePlan) -> tuple[SourceRecord, ...]:
    if not isinstance(source, SourcePlan) or source.source_schema != "bcs-local-category-source-v1":
        raise CategorySplitInputError("input must be a normalized local category source plan")
    records = tuple(source.candidates)
    if any(not isinstance(record, SourceRecord) for record in records):
        raise CategorySplitInputError("source plan contains an invalid record")
    if len({record.record_id for record in records}) != len(records):
        raise CategorySplitInputError("source plan contains duplicate record identities")
    if len({record.member_id if hasattr(record, "member_id") else record.provenance[0].relative_path for record in records}) != len(records):
        raise CategorySplitInputError("source plan contains duplicate capture members")
    by_digest: dict[str, int] = {}
    for record in records:
        if record.source_schema != source.source_schema or record.bcs_category not in BCS_CLASS_SCORES:
            raise CategorySplitInputError("source plan lineage or category is invalid")
        previous = by_digest.get(record.sha256)
        if previous is not None and previous != record.bcs_category:
            raise CategorySplitInputError("identical digest appears across BCS categories")
        by_digest[record.sha256] = record.bcs_category
    exclusions = tuple(source.exclusions)
    if any(not isinstance(item, SourceExclusion) or item.reason != "cross_category_identical_digest" for item in exclusions):
        raise CategorySplitInputError("source exclusions are invalid")
    if len({item.record_id for item in exclusions}) != len(exclusions) or {item.record_id for item in exclusions} & {item.record_id for item in records}:
        raise CategorySplitInputError("source candidates and exclusions overlap")
    return tuple(sorted(records, key=lambda record: record.record_id))


def _components(records: tuple[SourceRecord, ...]) -> tuple[tuple[SourceRecord, ...], ...]:
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    first_group: dict[str, int] = {}
    first_digest: dict[str, int] = {}
    for index, record in enumerate(records):
        if record.capture_group in first_group:
            union(index, first_group[record.capture_group])
        else:
            first_group[record.capture_group] = index
        if record.sha256 in first_digest:
            union(index, first_digest[record.sha256])
        else:
            first_digest[record.sha256] = index
    groups: dict[int, list[SourceRecord]] = {}
    for index, record in enumerate(records):
        groups.setdefault(find(index), []).append(record)
    return tuple(sorted((tuple(sorted(items, key=lambda record: record.record_id)) for items in groups.values()), key=lambda items: items[0].record_id))


def _assignment(record: SourceRecord, split: str) -> CategorySplitAssignment:
    return CategorySplitAssignment(
        split,
        record.bcs_category,
        record.record_id,
        record.capture_group,
        record.sha256,
        record.provenance,
        f"{split}/{record.bcs_category}/{record.record_id}",
    )


def create_category_split_plan(
    source: SourcePlan,
    *,
    seed: int,
    val_ratio: Real = 0.1,
    test_ratio: Real = 0.1,
) -> CategorySplitPlan:
    config = CategorySplitConfig(seed, val_ratio, test_ratio)
    records = _validate_input(source)
    components = list(_components(records))
    components.sort(key=lambda items: hashlib.sha256(f"{config.seed}\0{items[0].record_id}".encode()).hexdigest())
    total = [0] * NUM_CLASSES
    for record in records:
        total[record.bcs_category - 1] += 1
    ratios = (1 - config.val_ratio - config.test_ratio, config.val_ratio, config.test_ratio)
    targets = [[total[index] * ratio for index in range(NUM_CLASSES)] for ratio in ratios]
    current = [[0] * NUM_CLASSES for _ in SPLITS]
    selected: dict[str, str] = {}
    for component in components:
        vector = [0] * NUM_CLASSES
        for record in component:
            vector[record.bcs_category - 1] += 1
        scores: list[tuple[float, int, str]] = []
        for split_index, split in enumerate(SPLITS):
            candidate = [row[:] for row in current]
            for index, value in enumerate(vector):
                candidate[split_index][index] += value
            cost = sum(
                abs(candidate[row][index] - targets[row][index])
                + SPLIT_OVERSHOOT_WEIGHT * max(0, candidate[row][index] - targets[row][index])
                for row in range(len(SPLITS))
                for index in range(NUM_CLASSES)
            )
            scores.append((cost, split_index, split))
        _, split_index, split = min(scores)
        for index, value in enumerate(vector):
            current[split_index][index] += value
        for record in component:
            selected[record.record_id] = split

    assignments = tuple(_assignment(record, selected[record.record_id]) for record in records)
    counts = CategorySplitCounts(
        tuple(current[0]), tuple(current[1]), tuple(current[2])
    )
    assignment_rows = [
            (item.split, item.bcs_category, item.record_id, item.capture_group, item.sha256)
            for item in assignments
        ]
    exclusions = [
            (item.record_id, item.relative_path, item.bcs_category, item.sha256, item.reason)
            for item in sorted(
                source.exclusions,
                key=lambda item: (item.sha256, item.source_label, item.relative_path, item.record_id),
            )
        ]
    digest = split_identity_digest(
        source_schema=source.source_schema,
        identity_scheme=source.identity_scheme,
        mapping_lineage=source.mapping_lineage,
        observed_classes=source.observed_classes,
        seed=config.seed,
        canonical_val_ratio=config.canonical_val_ratio,
        canonical_test_ratio=config.canonical_test_ratio,
        assignments=assignment_rows,
        counts={split: getattr(counts, split) for split in SPLITS},
        exclusions=exclusions,
    )
    identity = CategorySplitPlanIdentity(
        seed=config.seed,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        canonical_val_ratio=config.canonical_val_ratio,
        canonical_test_ratio=config.canonical_test_ratio,
        candidate_record_ids=tuple(record.record_id for record in records),
        excluded_record_ids=tuple(item.record_id for item in source.exclusions),
        digest=digest,
        source_schema=source.source_schema,
        identity_scheme=source.identity_scheme,
        mapping_lineage=source.mapping_lineage,
        observed_classes=source.observed_classes,
    )
    return CategorySplitPlan(
        assignments=tuple(_assignment(record, selected[record.record_id]) for record in records),
        exclusions=source.exclusions,
        counts=counts,
        config=config,
        identity=identity,
        source_candidates=records,
    )


def validate_category_split_plan(plan: CategorySplitPlan) -> None:
    if not isinstance(plan, CategorySplitPlan):
        raise CategorySplitInputError("input must be a category split plan")
    expected = create_category_split_plan(
        SourcePlan(
            plan.source_candidates,
            plan.exclusions,
             (0,) * NUM_CLASSES,
            plan.identity.source_schema,
            plan.identity.identity_scheme,
            plan.identity.mapping_lineage,
            plan.identity.observed_classes,
        ),
        seed=plan.config.seed,
        val_ratio=plan.config.val_ratio,
        test_ratio=plan.config.test_ratio,
    )
    if plan != expected:
        raise CategorySplitInputError("category split plan integrity validation failed")
