from __future__ import annotations

from dataclasses import replace

import pytest

from vacca_bcs.category_split_plan import CategorySplitInputError, create_category_split_plan
from vacca_bcs.local_source import scan_local_source as _scan_local_source
from vacca_bcs.source_plan import normalize_local_source_scan


def scan_local_source(root, *args, **kwargs):
    return _scan_local_source(root, *args, approved_roots=(root.parent,), **kwargs)


def write_file(root, name, payload=b"tiny"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def local_scan(tmp_path):
    for category, label in enumerate(("3.25", "3.5", "3.75", "4.0", "4.25"), 1):
        write_file(tmp_path, f"{label}/GS_{category}_1.jpg", bytes([category]))
    return scan_local_source(tmp_path)


def test_local_adapter_preserves_capture_identity_and_never_invents_animal_ids(tmp_path):
    scan = local_scan(tmp_path)
    plan = normalize_local_source_scan(scan)
    assert plan.source_schema == "bcs-local-category-source-v1"
    assert plan.mapping_lineage == scan.mapping_lineage
    assert plan.observed_classes == (1, 2, 3, 4, 5)
    assert [(item.record_id, item.bcs_category) for item in plan.candidates] == [(record.record_id, record.bcs_category) for record in scan.records]
    assert all(item.capture_group for item in plan.candidates)
    assert all("animal" not in repr(item).lower() for item in plan.candidates)


def test_local_plan_and_grouped_split_are_order_independent(tmp_path):
    normalized = normalize_local_source_scan(local_scan(tmp_path))
    first = create_category_split_plan(normalized, seed=4)
    second = create_category_split_plan(replace(normalized, candidates=tuple(reversed(normalized.candidates))), seed=4)
    assert first == second
    assert first.identity.candidate_record_ids == tuple(sorted(item.record_id for item in normalized.candidates))


def test_unknown_schema_fails_closed(tmp_path):
    normalized = normalize_local_source_scan(local_scan(tmp_path))
    with pytest.raises(CategorySplitInputError):
        create_category_split_plan(replace(normalized, source_schema="unknown"), seed=1)
