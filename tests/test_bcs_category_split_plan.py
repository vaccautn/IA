from __future__ import annotations

from dataclasses import replace
import math

import pytest

from vacca_bcs.category_split_plan import (
    CategorySplitConfigError,
    CategorySplitInputError,
    create_category_split_plan,
    split_identity_digest,
    validate_category_split_plan,
)
from vacca_bcs.local_source import scan_local_source as _scan_local_source
from vacca_bcs.source_plan import normalize_local_source_scan


def scan_local_source(root, *args, **kwargs):
    return _scan_local_source(root, *args, approved_roots=(root.parent,), **kwargs)


def plan(tmp_path, *, count=10):
    for category, label in enumerate(("3.25", "3.5", "3.75", "4.0", "4.25"), 1):
        folder = tmp_path / label
        folder.mkdir(parents=True)
        for index in range(count):
            (folder / f"GS_{category * 100 + index}_1.jpg").write_bytes(bytes([category, index]))
    return normalize_local_source_scan(scan_local_source(tmp_path))


def test_group_atomicity_and_80_10_10_boundaries(tmp_path):
    source = plan(tmp_path)
    split = create_category_split_plan(source, seed=7)
    assert split.counts.train == (8, 8, 8, 8, 8)
    assert split.counts.val == (1, 1, 1, 1, 1)
    assert split.counts.test == (1, 1, 1, 1, 1)
    validate_category_split_plan(split)
    by_group = {}
    for assignment in split.assignments:
        by_group.setdefault(assignment.capture_group, set()).add(assignment.split)
    assert all(len(value) == 1 for value in by_group.values())


def test_same_digest_groups_union_transitively(tmp_path):
    source = plan(tmp_path, count=4)
    records = list(source.candidates)
    same_category = [index for index, record in enumerate(records) if record.bcs_category == 1][:3]
    digest = records[same_category[0]].sha256
    for index in same_category[1:]:
        records[index] = replace(records[index], sha256=digest)
    # The first three records are same-category and must remain one component.
    changed = replace(source, candidates=tuple(records))
    split = create_category_split_plan(changed, seed=1)
    assert {item.split for item in split.assignments if item.bcs_category == 1 and item.sha256 == digest}.__len__() == 1


def test_order_independence_and_identity_changes(tmp_path):
    source = plan(tmp_path)
    first = create_category_split_plan(source, seed=4)
    second = create_category_split_plan(replace(source, candidates=tuple(reversed(source.candidates))), seed=4)
    assert first == second
    assert first.identity == second.identity
    different_seed = create_category_split_plan(source, seed=5)
    assert first.identity.digest != different_seed.identity.digest
    assert first.assignments != different_seed.assignments
    changed = list(first.assignments)
    changed[0] = replace(changed[0], capture_group="changed-group")
    assert first.identity.digest != split_identity_digest(
        source_schema=first.identity.source_schema,
        identity_scheme=first.identity.identity_scheme,
        mapping_lineage=first.identity.mapping_lineage,
        observed_classes=first.identity.observed_classes,
        seed=first.identity.seed,
        canonical_val_ratio=first.identity.canonical_val_ratio,
        canonical_test_ratio=first.identity.canonical_test_ratio,
        assignments=[
            (item.split, item.bcs_category, item.record_id, item.capture_group, item.sha256)
            for item in changed
        ],
        counts={name: getattr(first.counts, name) for name in ("train", "val", "test")},
        exclusions=[],
    )
    with pytest.raises(CategorySplitInputError):
        validate_category_split_plan(replace(first, assignments=tuple(changed)))


def test_zero_and_tiny_class_boundaries_are_deterministic(tmp_path):
    source = plan(tmp_path, count=1)
    zero = create_category_split_plan(source, seed=7, val_ratio=0, test_ratio=0)
    assert zero.counts.train == (1,) * 5
    assert zero.counts.val == (0,) * 5
    assert zero.counts.test == (0,) * 5
    other = create_category_split_plan(source, seed=8, val_ratio=0, test_ratio=0)
    assert other.assignments == zero.assignments
    assert other.identity.digest != zero.identity.digest


@pytest.mark.parametrize("value", [True, 1.0])
def test_invalid_seed_is_rejected(tmp_path, value):
    with pytest.raises(CategorySplitConfigError):
        create_category_split_plan(plan(tmp_path), seed=value)


@pytest.mark.parametrize("field,value", [("val_ratio", True), ("test_ratio", "0.1"), ("val_ratio", math.nan), ("test_ratio", math.inf), ("val_ratio", -0.1), ("test_ratio", 1.0)])
def test_split_ratios_are_strictly_typed_and_bounded(tmp_path, field, value):
    ratios = {"val_ratio": 0.1, "test_ratio": 0.1}
    ratios[field] = value
    with pytest.raises(CategorySplitConfigError):
        create_category_split_plan(plan(tmp_path), seed=7, **ratios)


def test_split_ratios_must_leave_training_data(tmp_path):
    with pytest.raises(CategorySplitConfigError):
        create_category_split_plan(plan(tmp_path), seed=7, val_ratio=0.5, test_ratio=0.5)


def test_exclusions_are_kept_out_of_assignments(tmp_path):
    source = plan(tmp_path)
    excluded = source.candidates[-1]
    from vacca_bcs.source_plan import SourceExclusion
    exclusion = SourceExclusion(excluded.record_id, excluded.materializer_key, "4.25", 5, excluded.sha256, "cross_category_identical_digest")
    changed = replace(source, candidates=source.candidates[:-1], exclusions=(exclusion,))
    split = create_category_split_plan(changed, seed=3)
    assert excluded.record_id not in {item.record_id for item in split.assignments}
    assert split.exclusions == (exclusion,)
