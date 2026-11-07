from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from vacca_bcs.local_source import (
    LOCAL_BCS_MAPPING,
    LocalSourceCollisionError,
    LocalSourceMaterializationError,
    LocalSourceRecord,
    LocalSourceScanError,
    LocalSourceMapping,
    materialize_local_record,
    scan_local_source as _scan_local_source,
)


def scan_local_source(root, *args, **kwargs):
    return _scan_local_source(root, *args, approved_roots=(root.parent,), **kwargs)


def make_file(root, folder, name, payload=b"tiny"):
    path = root / folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_mapping_counts_order_and_path_identity_are_stable(tmp_path):
    make_file(tmp_path, "4.25", "GS_4_1.PNG", b"z")
    make_file(tmp_path, "3.5", "GS_2_1.jpg", b"b")
    make_file(tmp_path, "3.25", "GS_1_1.jpeg", b"a")
    make_file(tmp_path, "4.0", "GS_3_1.JPG", b"c")

    first = scan_local_source(tmp_path, LocalSourceMapping(tuple(reversed(LOCAL_BCS_MAPPING.entries))))
    second = scan_local_source(tmp_path)
    assert first == second
    assert [item.relative_path for item in first.records] == [
        "3.25/GS_1_1.jpeg", "3.5/GS_2_1.jpg", "4.0/GS_3_1.JPG", "4.25/GS_4_1.PNG"
    ]
    assert first.counts == (1, 1, 0, 1, 1)
    assert first.observed_classes == (1, 2, 4, 5)
    assert first.mapping_lineage == LOCAL_BCS_MAPPING.entries
    assert first.records[0].record_id == hashlib.sha256(
        b"bcs-local-category-source-v1\0" + b"3.25/GS_1_1.jpeg"
    ).hexdigest()
    with pytest.raises((AttributeError, TypeError)):
        first.records = ()


def test_xml_is_only_accepted_as_a_paired_unused_sidecar(tmp_path):
    make_file(tmp_path, "3.25", "GS_1_1.jpg", b"not-an-image")
    make_file(tmp_path, "3.25", "GS_1_1.xml", b"<label>1</label>")
    assert scan_local_source(tmp_path).records[0].bcs_category == 1

    make_file(tmp_path, "3.25", "orphan.xml", b"<label>5</label>")
    with pytest.raises(LocalSourceScanError, match="unpaired XML"):
        scan_local_source(tmp_path)


@pytest.mark.parametrize("kind", ["unknown", "nested", "unsupported"])
def test_unexpected_layout_is_rejected(tmp_path, kind):
    if kind == "unknown":
        (tmp_path / "4.5").mkdir()
    elif kind == "nested":
        make_file(tmp_path, "3.25/nested", "GS_1_1.jpg")
    else:
        make_file(tmp_path, "3.25", "GS_1_1.txt")
    with pytest.raises(LocalSourceScanError):
        scan_local_source(tmp_path)


def test_casefold_nfc_and_cross_class_digest_collisions_are_rejected(tmp_path):
    make_file(tmp_path, "3.25", "GS_1_1.jpg", b"one")
    make_file(tmp_path, "3.25", "GS_1_1.JPG", b"two")
    if (tmp_path / "3.25" / "GS_1_1.jpg").read_bytes() == b"one":
        with pytest.raises(LocalSourceCollisionError):
            scan_local_source(tmp_path)

    other = tmp_path / "nfc"
    make_file(other, "3.25", "GS_1_2.jpg", b"one")
    make_file(other, "3.25", "GS_1_2.JPG", b"two")
    if (other / "3.25" / "GS_1_2.jpg").read_bytes() == b"one":
        with pytest.raises(LocalSourceCollisionError):
            scan_local_source(other)

    digest_root = tmp_path / "digest"
    make_file(digest_root, "3.25", "GS_1_1.jpg")
    make_file(digest_root, "3.25", "GS_1_2.jpg")
    same_class = scan_local_source(digest_root, digest_fn=lambda path: "0" * 64)
    assert len(same_class.records) == 2

    cross_class = tmp_path / "cross-class-digest"
    make_file(cross_class, "3.25", "GS_1_1.jpg")
    make_file(cross_class, "4.0", "GS_2_1.jpg")
    quarantined = scan_local_source(cross_class, digest_fn=lambda path: "0" * 64)
    assert quarantined.records == ()
    assert len(quarantined.exclusions) == 2

    id_root = tmp_path / "record-id"
    make_file(id_root, "3.25", "GS_1_1.jpg")
    make_file(id_root, "3.25", "GS_1_2.jpg", b"different")
    with pytest.raises(LocalSourceCollisionError):
        scan_local_source(id_root, record_id_fn=lambda path: "0" * 64)


def test_symlink_root_class_and_file_are_rejected(tmp_path):
    target = tmp_path / "target"
    make_file(target, "3.25", "GS_1_1.jpg")
    try:
        (tmp_path / "root-link").symlink_to(target, target_is_directory=True)
        class_root = tmp_path / "class-link-root"
        class_root.mkdir()
        (class_root / "3.25").symlink_to(target / "3.25", target_is_directory=True)
        (target / "3.25" / "file-link.jpg").symlink_to(target / "3.25" / "a.jpg")
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(LocalSourceScanError):
        scan_local_source(tmp_path / "root-link")
    with pytest.raises(LocalSourceScanError):
        scan_local_source(class_root)
    with pytest.raises(LocalSourceScanError):
        scan_local_source(target)


def test_materializer_is_bounded_hash_checked_and_sanitized(tmp_path):
    path = make_file(tmp_path, "3.25", "GS_1_1.jpg", b"payload")
    scan = scan_local_source(tmp_path)
    materialized = materialize_local_record(scan, scan.records[0], max_bytes=20)
    assert materialized.sha256 == hashlib.sha256(b"payload").hexdigest()
    assert materialized.payload == b"payload"
    assert materialized.size_bytes == 7

    path.write_bytes(b"replacement")
    with pytest.raises(LocalSourceMaterializationError) as failure:
        materialize_local_record(scan, scan.records[0])
    assert str(tmp_path) not in str(failure.value)
    assert b"replacement" not in str(failure.value).encode()

    empty_root = tmp_path / "empty-root"
    make_file(empty_root, "3.25", "GS_1_1.jpg", b"")
    empty_scan = scan_local_source(empty_root)
    with pytest.raises(LocalSourceMaterializationError, match="empty"):
        materialize_local_record(empty_scan, empty_scan.records[0])
    (empty_root / "3.25" / "GS_1_1.jpg").unlink()
    (empty_root / "3.25").rmdir()
    empty_root.rmdir()

    fresh_scan = scan_local_source(tmp_path)
    with pytest.raises(LocalSourceMaterializationError, match="maximum"):
        materialize_local_record(fresh_scan, fresh_scan.records[0], max_bytes=2)


def test_materializer_rejects_traversal_and_forged_records(tmp_path):
    make_file(tmp_path, "3.25", "GS_1_1.jpg")
    scan = scan_local_source(tmp_path)
    forged = replace(scan.records[0], relative_path="../secret.jpg")
    with pytest.raises(LocalSourceMaterializationError):
        materialize_local_record(scan, forged)
    assert isinstance(scan.records, tuple)
    assert isinstance(scan.records[0], LocalSourceRecord)
