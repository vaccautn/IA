from __future__ import annotations

import pytest

from vacca_bcs.source_client import (
    BCSSourceEvidence,
    BCSSourceEvaluationRow,
    BCSSourceExport,
)
from vacca_bcs.source_plan import (
    SourcePlanConflictError,
    SourcePlanIntegrityError,
    SourceProvenance,
    normalize_source_export,
)


def row(evaluation_id, score, *evidence):
    return BCSSourceEvaluationRow(
        evaluation_id,
        evaluation_id + 100,
        evaluation_id + 200,
        score,
        tuple(BCSSourceEvidence(*item) for item in evidence),
    )


def source(*rows):
    return BCSSourceExport("bcs-source-v1", tuple(rows))


def test_plan_flattens_values_counts_classes_and_excludes_empty_keys():
    plan = normalize_source_export(
        source(
            row(10, 1, (4, "key-a")),
            row(20, 3, (8, "key-b"), (9, "")),
        )
    )

    assert [(item.bcs_score, item.evidence_id) for item in plan.candidates] == [
        (1, 4),
        (3, 8),
    ]
    assert plan.candidates[0].storage_key == "key-a"
    assert plan.candidates[1].storage_key == "key-b"
    assert plan.class_counts == (1, 0, 1, 0, 0)
    assert [(item.evidence_id, item.reason) for item in plan.exclusions] == [
        (9, "empty_storage_key")
    ]


def test_empty_export_and_empty_evidence_are_valid_plans():
    assert normalize_source_export(source()).class_counts == (0, 0, 0, 0, 0)
    assert normalize_source_export(source(row(1, 5))).candidates == ()


def test_whitespace_only_keys_are_excluded_but_trimmed_identity_fails_closed():
    plan = normalize_source_export(source(row(1, 2, (1, ""), (2, "  "), (3, "\t"))))
    assert {item.reason for item in plan.exclusions} == {
        "empty_storage_key",
        "whitespace_storage_key",
    }

    with pytest.raises(SourcePlanIntegrityError) as failure:
        normalize_source_export(source(row(1, 2, (1, "secret-storage-key "))))
    assert "secret-storage-key" not in str(failure.value)
    assert "evidence_id=1" in str(failure.value)


def test_same_key_and_label_collapses_with_sorted_full_provenance():
    first = row(20, 4, (2, "same-key"))
    second = row(3, 4, (9, "same-key"))

    plan = normalize_source_export((first, second))

    candidate = plan.candidates[0]
    assert (candidate.evidence_id, candidate.evaluation_id) == (2, 20)
    assert candidate.provenance == (
        SourceProvenance(evidence_id=2, evaluation_id=20),
        SourceProvenance(evidence_id=9, evaluation_id=3),
    )
    with pytest.raises((AttributeError, TypeError)):
        candidate.provenance = ()
    assert plan.class_counts == (0, 0, 0, 1, 0)


def test_multiple_conflicts_report_same_stable_details_in_any_input_order():
    conflicts = (
        row(30, 3, (1, "first-secret-key")),
        row(31, 4, (7, "first-secret-key")),
        row(20, 3, (2, "second-secret-key")),
        row(21, 4, (8, "second-secret-key")),
    )

    def conflict_details(rows):
        with pytest.raises(SourcePlanConflictError) as failure:
            normalize_source_export(rows)
        error = failure.value
        return type(error), error.evidence_ids, error.evaluation_ids, str(error)

    first = conflict_details(conflicts)
    reversed_input = conflict_details(tuple(reversed(conflicts)))

    assert first == reversed_input
    assert first[1:] == (
        (1, 7),
        (30, 31),
        "conflicting labels for evidence_ids=(1, 7), evaluation_ids=(30, 31)",
    )

    assert "secret-key" not in first[3]


def test_manual_duplicate_ids_are_rejected_deterministically():
    duplicate_evaluations = (row(1, 2, (1, "a")), row(1, 2, (2, "b")))
    duplicate_evidence = (row(1, 2, (1, "a")), row(2, 2, (1, "b")))

    for rows in (duplicate_evaluations, duplicate_evidence):
        with pytest.raises(SourcePlanIntegrityError):
            normalize_source_export(rows)


def test_output_is_independent_of_input_order_and_immutable_repr_hides_key():
    rows = (row(10, 2, (8, "key")), row(3, 2, (2, "key")))
    first = normalize_source_export(rows)
    second = normalize_source_export(tuple(reversed(rows)))

    assert first == second
    assert "key" not in repr(first.candidates[0])
    with pytest.raises((AttributeError, TypeError)):
        first.candidates = ()
