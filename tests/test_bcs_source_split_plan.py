from __future__ import annotations

import math
from dataclasses import replace

import pytest

from vacca_bcs.source_plan import (
    SourceCandidate,
    SourceExclusion,
    SourcePlan,
    SourceProvenance,
)
from vacca_bcs.source_split_plan import (
    IntegerSplitConfigError,
    IntegerSplitInputError,
    create_integer_split_plan,
)


def candidate(evidence_id: int, score: int) -> SourceCandidate:
    evaluation_id = evidence_id + 100
    return SourceCandidate(
        evaluation_id=evaluation_id,
        session_id=evidence_id + 200,
        animal_id=evidence_id + 300,
        evidence_id=evidence_id,
        storage_key=f"untrusted-key-{evidence_id}",
        bcs_score=score,
        provenance=(
            SourceProvenance(evidence_id=evidence_id, evaluation_id=evaluation_id),
        ),
    )


def plan(
    *candidates: SourceCandidate,
    exclusions: tuple[SourceExclusion, ...] = (),
) -> SourcePlan:
    return SourcePlan(
        candidates=tuple(candidates), exclusions=exclusions, class_counts=(0,) * 5
    )


def digest(source: SourcePlan, seed: int = 7, ratio: float = 0.5) -> str:
    return create_integer_split_plan(source, seed=seed, val_ratio=ratio).identity.digest


def test_ratio_zero_and_stratified_boundary_counts_keep_small_classes_in_train():
    source = plan(
        candidate(1, 1),
        candidate(2, 2),
        candidate(3, 2),
        *(candidate(evidence_id, 3) for evidence_id in range(10, 14)),
    )

    split = create_integer_split_plan(source, seed=7, val_ratio=0.5)
    assert split.counts.train == (1, 1, 2, 0, 0)
    assert split.counts.val == (0, 1, 2, 0, 0)
    assert {item.bcs_score for item in split.assignments} == {1, 2, 3}
    assert all(
        item.split == "train"
        for item in create_integer_split_plan(source, seed=7, val_ratio=0).assignments
    )


def test_decimal_floor_boundaries_and_clamps_are_exact():
    hundred = plan(*(candidate(evidence_id, 1) for evidence_id in range(100)))
    assert (
        create_integer_split_plan(hundred, seed=7, val_ratio=0.29).counts.val[0] == 29
    )

    three = plan(*(candidate(evidence_id, 2) for evidence_id in range(3)))
    assert create_integer_split_plan(three, seed=7, val_ratio=0.01).counts.val[1] == 1
    two = plan(candidate(1, 3), candidate(2, 3))
    assert create_integer_split_plan(two, seed=7, val_ratio=0.99).counts.val[2] == 1


@pytest.mark.parametrize("ratio", [True, math.nan, math.inf, -0.01, 1.0, "0.5"])
def test_ratio_must_be_a_finite_non_bool_value_in_inclusive_lower_bound(ratio):
    with pytest.raises(IntegerSplitConfigError):
        create_integer_split_plan(plan(candidate(1, 1)), seed=1, val_ratio=ratio)


def test_seed_must_be_an_integer_but_zero_is_valid():
    with pytest.raises(IntegerSplitConfigError):
        create_integer_split_plan(plan(candidate(1, 1)), seed=True, val_ratio=0)
    assert create_integer_split_plan(plan(), seed=0, val_ratio=0).assignments == ()


def test_same_seed_is_order_independent_and_identity_is_stable():
    source = plan(*(candidate(evidence_id, 4) for evidence_id in range(1, 9)))

    first = create_integer_split_plan(source, seed=12, val_ratio=0.5)
    reversed_source = plan(*reversed(source.candidates))
    second = create_integer_split_plan(reversed_source, seed=12, val_ratio=0.5)

    assert first == second
    assert first.identity == second.identity


def test_different_seeds_change_eligible_assignments():
    source = plan(*(candidate(evidence_id, 5) for evidence_id in range(1, 11)))

    first = create_integer_split_plan(source, seed=1, val_ratio=0.5)
    second = create_integer_split_plan(source, seed=2, val_ratio=0.5)

    first_val = {item.evidence_id for item in first.assignments if item.split == "val"}
    second_val = {
        item.evidence_id for item in second.assignments if item.split == "val"
    }
    assert first_val != second_val


def test_assignments_are_complete_disjoint_preserve_exclusions_and_use_safe_stems():
    source = SourcePlan(
        candidates=(candidate(8, 1), candidate(2, 1), candidate(5, 2)),
        exclusions=(SourceExclusion(99, 900, 3, "empty_storage_key"),),
        class_counts=(2, 1, 0, 0, 0),
    )

    split = create_integer_split_plan(source, seed=4, val_ratio=0.5)
    assignments = split.assignments
    assert {item.evidence_id for item in assignments} == {2, 5, 8}
    assert {
        (item.evidence_id, item.bcs_score, item.storage_key, item.provenance)
        for item in assignments
    } == {
        (
            evidence_id,
            score,
            f"untrusted-key-{evidence_id}",
            (SourceProvenance(evidence_id, evidence_id + 100),),
        )
        for evidence_id, score in ((2, 1), (8, 1), (5, 2))
    }
    assert (
        len(
            {
                item.evidence_id for item in assignments if item.split == "train"
            }.intersection(
                item.evidence_id for item in assignments if item.split == "val"
            )
        )
        == 0
    )
    assert split.exclusions == source.exclusions
    assert split.identity.candidate_evidence_ids == (2, 8, 5)
    assert split.identity.excluded_evidence_ids == (900,)
    assert split.config.seed == 4
    assert split.config.val_ratio == 0.5
    assert [item.relative_path_stem for item in assignments] == [
        f"{item.split}/{item.bcs_score}/{item.evidence_id}" for item in assignments
    ]
    assert all("untrusted-key" not in item.relative_path_stem for item in assignments)


def test_provenance_and_outputs_are_immutable_and_fractional_labels_are_rejected():
    source = plan(candidate(1, 1))
    split = create_integer_split_plan(source, seed=1, val_ratio=0)

    assert isinstance(split.assignments, tuple)
    with pytest.raises((AttributeError, TypeError)):
        split.assignments = ()
    with pytest.raises((AttributeError, TypeError)):
        split.assignments[0].split = "val"

    with pytest.raises(IntegerSplitInputError):
        create_integer_split_plan(plan(candidate(1, 3.5)), seed=1, val_ratio=0)


def test_duplicate_and_overlapping_exclusion_ids_are_typed_and_order_independent():
    duplicate = (
        SourceExclusion(10, 9, 1, "empty_storage_key"),
        SourceExclusion(11, 9, 1, "whitespace_storage_key"),
    )
    overlap = (SourceExclusion(10, 1, 1, "empty_storage_key"),)

    def error_text(source):
        with pytest.raises(IntegerSplitInputError) as failure:
            create_integer_split_plan(source, seed=1, val_ratio=0)
        return type(failure.value), str(failure.value)

    first = error_text(plan(candidate(1, 1), exclusions=duplicate))
    second = error_text(plan(candidate(1, 1), exclusions=tuple(reversed(duplicate))))
    assert first == second

    assert error_text(plan(candidate(1, 1), exclusions=overlap))[1] == (
        "evidence IDs appear in both candidates and exclusions: (1,)"
    )


def test_identity_digest_changes_for_contract_relevant_inputs_and_hides_storage_keys():
    base_candidate = candidate(2, 2)
    base_exclusion = SourceExclusion(90, 900, 4, "empty_storage_key")
    base = plan(base_candidate, candidate(5, 3), exclusions=(base_exclusion,))
    base_digest = digest(base)

    candidate_changes = (
        replace(base_candidate, bcs_score=1),
        replace(base_candidate, evidence_id=3),
        replace(base_candidate, session_id=999),
        replace(base_candidate, animal_id=999),
        replace(base_candidate, evaluation_id=999),
        replace(base_candidate, storage_key="changed-key"),
        replace(base_candidate, provenance=(SourceProvenance(2, 999),)),
    )
    assert all(
        digest(plan(changed, candidate(5, 3), exclusions=(base_exclusion,)))
        != base_digest
        for changed in candidate_changes
    )

    exclusion_changes = (
        replace(base_exclusion, evidence_id=91),
        replace(base_exclusion, bcs_score=5),
        replace(base_exclusion, reason="whitespace_storage_key"),
    )
    assert all(
        digest(plan(base_candidate, candidate(5, 3), exclusions=(changed,)))
        != base_digest
        for changed in exclusion_changes
    )

    assert digest(base, seed=8) != base_digest
    assert digest(base, ratio=0.25) != base_digest
    changed_plan = create_integer_split_plan(
        plan(replace(base_candidate, storage_key="changed-key")), seed=7, val_ratio=0
    )
    assert "changed-key" not in repr(changed_plan)
