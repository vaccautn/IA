from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from vacca_bcs.local_source import LOCAL_BCS_MAPPING, scan_local_source
from vacca_bcs.source_client import (
    BCSSourceEvidence,
    BCSSourceEvaluationRow,
    BCSSourceExport,
)
from vacca_bcs.source_plan import normalize_backend_source_export, normalize_local_source_scan
from vacca_bcs.source_split_plan import IntegerSplitInputError, create_integer_split_plan


def write_file(root, name, payload=b"tiny"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def backend_export():
    return BCSSourceExport(
        "bcs-source-v1",
        (
            BCSSourceEvaluationRow(2, 7, 8, 3, (BCSSourceEvidence(11, "safe"),)),
        ),
    )


def test_backend_adapter_exposes_canonical_identity_without_changing_legacy_fields():
    plan = normalize_backend_source_export(backend_export())
    candidate = plan.candidates[0]
    assert candidate.source_schema == "bcs-source-v1"
    assert candidate.materializer_key == 11
    assert candidate.record_id == hashlib.sha256(b"bcs-source-v1\0evidence/11").hexdigest()
    assert candidate.provenance[0].evidence_id == 11
    assert candidate.provenance[0].evaluation_id == 2
    assert normalize_backend_source_export(tuple(reversed(backend_export().rows))) == plan


def test_backend_adapter_preserves_full_provenance_when_storage_key_collapses():
    source = BCSSourceExport(
        "bcs-source-v1",
        (
            BCSSourceEvaluationRow(2, 7, 8, 3, (BCSSourceEvidence(11, "same"),)),
            BCSSourceEvaluationRow(3, 9, 10, 3, (BCSSourceEvidence(12, "same"),)),
        ),
    )
    provenance = normalize_backend_source_export(source).candidates[0].provenance
    assert [(item.evidence_id, item.evaluation_id, item.session_id, item.animal_id) for item in provenance] == [
        (11, 2, 7, 8),
        (12, 3, 9, 10),
    ]


def test_local_adapter_preserves_scan_identity_and_never_invents_backend_ids(tmp_path):
    write_file(tmp_path, "3.25/a.jpg", b"a")
    write_file(tmp_path, "4.0/b.png", b"b")
    scan = scan_local_source(tmp_path)
    plan = normalize_local_source_scan(scan)
    assert plan.source_schema == "bcs-local-folder-v1"
    assert plan.mapping_lineage == LOCAL_BCS_MAPPING.entries
    assert plan.observed_classes == (3, 4)
    assert [(item.record_id, item.bcs_score) for item in plan.candidates] == [
        (record.record_id, record.bcs_score) for record in scan.records
    ]
    assert [item.materializer_key for item in plan.candidates] == [
        record.relative_path for record in scan.records
    ]
    assert all(not hasattr(item, "evidence_id") for item in plan.candidates)
    assert all(scan.root.as_posix() not in repr(item) for item in plan.candidates)


def test_local_plan_and_split_are_order_independent_and_use_record_ids(tmp_path):
    write_file(tmp_path, "3.25/b.jpg", b"b")
    write_file(tmp_path, "3.25/a.jpg", b"a")
    scan = scan_local_source(tmp_path)
    first = create_integer_split_plan(
        normalize_local_source_scan(scan), seed=4, val_ratio=0.5
    )
    reversed_scan = replace(scan, records=tuple(reversed(scan.records)))
    second = create_integer_split_plan(
        normalize_local_source_scan(reversed_scan), seed=4, val_ratio=0.5
    )
    assert first == second
    assert [item.record_id for item in first.assignments] == [
        record.record_id for record in scan.records
    ]
    assert [item.relative_path_stem for item in first.assignments] == [
        f"{item.split}/3/{item.record_id}" for item in first.assignments
    ]
    assert first.identity.candidate_record_ids == tuple(
        sorted(record.record_id for record in scan.records)
    )


def test_identity_changes_for_source_specific_lineage_and_unknown_schema_fails_closed(
    tmp_path,
):
    write_file(tmp_path, "3.25/a.jpg")
    scan = scan_local_source(tmp_path)
    first = normalize_local_source_scan(scan)
    changed = normalize_local_source_scan(
        replace(scan, mapping_lineage=(("3.25", 3),))
    )
    assert create_integer_split_plan(first, seed=1, val_ratio=0).identity.digest != (
        create_integer_split_plan(changed, seed=1, val_ratio=0).identity.digest
    )
    with pytest.raises(IntegerSplitInputError):
        create_integer_split_plan(replace(first, source_schema="unknown"), seed=1, val_ratio=0)
