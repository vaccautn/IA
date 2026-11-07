from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image

import vacca_bcs.category_snapshot as snapshot_module
from vacca_bcs.category_snapshot import (
    CategorySnapshotCleanupError,
    CategorySnapshotDurabilityError,
    CategorySnapshotImageError,
    CategorySnapshotInputError,
    CategorySnapshotLockCleanupError,
    CategorySnapshotLockError,
    CategorySnapshotMaterializationError,
    CategorySnapshotOutputError,
    CategorySnapshotPublicationError,
    PUBLICATION_MARKER_NAME,
    build_category_snapshot as _build_category_snapshot,
    finalize_category_snapshot_publication,
    load_category_snapshot_manifest,
)
from vacca_bcs.category_split_plan import create_category_split_plan
from vacca_bcs.local_source import LocalSourceMaterialized, LocalSourceMaterializer, scan_local_source as _scan_local_source
from vacca_bcs.source_plan import normalize_local_source_scan


def build_category_snapshot(plan, output_root, materializer):
    return _build_category_snapshot(
        plan,
        output_root,
        materializer,
        approved_roots=(output_root.parent,),
    )


def scan_local_source(root, *args, **kwargs):
    return _scan_local_source(root, *args, approved_roots=(root.parent,), **kwargs)


def image_bytes(format_name="JPEG", color=(20, 40, 60)):
    stream = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(stream, format=format_name)
    return stream.getvalue()


def split_plan(tmp_path: Path):
    source = tmp_path / "source"
    for category, label in enumerate(("3.25", "3.5", "3.75", "4.0", "4.25"), 1):
        folder = source / label
        folder.mkdir(parents=True)
        (folder / f"GS_{category}_1.jpg").write_bytes(image_bytes(color=(category * 20, 40, 60)))
    scan = scan_local_source(source)
    return create_category_split_plan(normalize_local_source_scan(scan), seed=9), scan


def build(tmp_path: Path, name="snapshot"):
    plan, scan = split_plan(tmp_path)
    return build_category_snapshot(plan, tmp_path / name, LocalSourceMaterializer(scan)), scan


def test_manifest_round_trip_has_categories_test_and_exclusions(tmp_path):
    snapshot, _ = build(tmp_path)
    manifest = json.loads(snapshot.manifest_json)
    assert manifest["manifest_schema_version"] == "bcs-category-snapshot-v1"
    assert manifest["domain_id"] == "bcs-category-1-5-v1"
    assert manifest["source_schema"] == "bcs-local-category-source-v1"
    assert manifest["mapping"] == {"3.25": 1, "3.5": 2, "3.75": 3, "4.0": 4, "4.25": 5}
    assert set(manifest["counts"]) == {"train", "val", "test"}
    assert manifest["exclusions"] == []
    assert load_category_snapshot_manifest(snapshot.output_root / "manifest.json") == manifest


def test_manifest_rejects_legacy_or_tampered_lineage(tmp_path):
    snapshot, _ = build(tmp_path)
    manifest = json.loads(snapshot.manifest_json)
    for changed in (
        {**manifest, "manifest_schema_version": "bcs-integer-snapshot-v2"},
        {**manifest, "domain_id": "other"},
        {**manifest, "mapping": {"3.25": 3}},
        {**manifest, "observed_classes": [1, 2]},
    ):
        with pytest.raises(ValueError):
            load_category_snapshot_manifest(changed)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda manifest: manifest["records"][0].update(capture_group="moved-group"),
        lambda manifest: manifest["records"][0].update(sha256="0" * 64),
        lambda manifest: manifest["records"][0]["provenance"][0].update(source_label="4.0"),
    ],
)
def test_manifest_rejects_coherent_assignment_or_mapping_tampering(tmp_path, mutation):
    snapshot, _ = build(tmp_path)
    manifest = json.loads(snapshot.manifest_json)
    mutation(manifest)
    with pytest.raises(ValueError):
        load_category_snapshot_manifest(manifest)


def test_jpeg_png_layout_is_safe_and_deterministic(tmp_path):
    first, _ = build(tmp_path / "first")
    second, _ = build(tmp_path / "second")
    assert first.manifest_json == second.manifest_json
    assert all(path.suffix == ".jpg" for path in first.output_root.rglob("*.jpg"))
    assert sorted(path.relative_to(first.output_root).as_posix() for path in first.output_root.rglob("*.jpg")) == [f"train/{category}/{record.record_id}.jpg" for category, record in zip(range(1, 6), first.records)]


def test_invalid_payload_is_typed_and_transactional(tmp_path):
    plan, scan = split_plan(tmp_path)
    def broken(record_id):
        record = next(item for item in scan.records if item.record_id == record_id)
        return LocalSourceMaterialized(record_id, record.relative_path, b"not image", hashlib.sha256(b"not image").hexdigest(), 9)
    with pytest.raises(CategorySnapshotMaterializationError):
        build_category_snapshot(plan, tmp_path / "bad", broken)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("payload", [b"not image", image_bytes()[:10]])
def test_unsupported_or_truncated_materialized_images_fail_closed(tmp_path, payload):
    original_plan, scan = split_plan(tmp_path)
    target_id = original_plan.assignments[0].record_id
    payload_digest = hashlib.sha256(payload).hexdigest()
    source_plan = normalize_local_source_scan(scan)
    candidates = tuple(
        replace(record, sha256=payload_digest)
        if record.record_id == target_id else record
        for record in source_plan.candidates
    )
    plan = create_category_split_plan(replace(source_plan, candidates=candidates), seed=9)

    def broken(record_id):
        record = next(item for item in scan.records if item.record_id == record_id)
        if record_id == target_id:
            return LocalSourceMaterialized(record_id, record.relative_path, payload, payload_digest, len(payload))
        return LocalSourceMaterializer(scan).materialize(record_id)

    with pytest.raises(CategorySnapshotImageError):
        build_category_snapshot(plan, tmp_path / "invalid-image", broken)
    assert not (tmp_path / "invalid-image").exists()


def test_materializer_identity_and_digest_mismatch_are_rejected(tmp_path):
    plan, _ = split_plan(tmp_path)
    with pytest.raises(CategorySnapshotMaterializationError):
        build_category_snapshot(plan, tmp_path / "mismatch", lambda _: None)
    assert not (tmp_path / "mismatch").exists()


def test_existing_output_and_lock_are_preserved(tmp_path):
    plan, scan = split_plan(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "sentinel").write_bytes(b"keep")
    with pytest.raises(CategorySnapshotOutputError):
        build_category_snapshot(plan, existing, LocalSourceMaterializer(scan))
    assert (existing / "sentinel").read_bytes() == b"keep"
    lock = tmp_path / ".locked.lock"
    lock.write_bytes(b"held")
    with pytest.raises(CategorySnapshotLockError):
        build_category_snapshot(plan, tmp_path / "locked", LocalSourceMaterializer(scan))


def test_dangling_symlink_and_final_target_race_are_never_overwritten(tmp_path):
    plan, scan = split_plan(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    dangling = tmp_path / "dangling"
    try:
        dangling.symlink_to(outside / "missing", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable")
    with pytest.raises(CategorySnapshotOutputError):
        build_category_snapshot(plan, dangling, LocalSourceMaterializer(scan))

    raced = tmp_path / "raced"

    def publish_race(staging, target):
        target.mkdir()
        (target / "sentinel").write_bytes(b"preserve")
        raise OSError("target appeared during publication")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(snapshot_module.os, "rename", publish_race)
    try:
        with pytest.raises(CategorySnapshotPublicationError):
            build_category_snapshot(plan, raced, LocalSourceMaterializer(scan))
    finally:
        monkeypatch.undo()
    assert (raced / "sentinel").read_bytes() == b"preserve"
    assert not (tmp_path / ".raced.lock").exists()


def test_malformed_plan_fails_before_materialization_and_success_releases_lock(tmp_path):
    plan, scan = split_plan(tmp_path)
    malformed = replace(plan, assignments=())
    calls = []

    def materialize(record_id):
        calls.append(record_id)
        return LocalSourceMaterializer(scan).materialize(record_id)

    with pytest.raises(CategorySnapshotInputError):
        build_category_snapshot(malformed, tmp_path / "malformed", materialize)
    assert calls == []
    output = tmp_path / "success"
    build_category_snapshot(plan, output, LocalSourceMaterializer(scan))
    assert not (tmp_path / ".success.lock").exists()


def test_durability_and_publication_failures_leave_no_new_root(tmp_path, monkeypatch):
    plan, scan = split_plan(tmp_path)
    monkeypatch.setattr(snapshot_module.os, "fsync", lambda _: (_ for _ in ()).throw(OSError("fsync")))
    with pytest.raises(CategorySnapshotDurabilityError):
        build_category_snapshot(plan, tmp_path / "fsync", LocalSourceMaterializer(scan))
    monkeypatch.undo()
    monkeypatch.setattr(snapshot_module.os, "rename", lambda *_: (_ for _ in ()).throw(OSError("rename")))
    with pytest.raises(CategorySnapshotPublicationError):
        build_category_snapshot(plan, tmp_path / "rename", LocalSourceMaterializer(scan))


def test_post_publication_durability_failure_is_fail_closed_and_retryable(tmp_path, monkeypatch):
    plan, scan = split_plan(tmp_path)
    output = tmp_path / "post-fsync"
    real_fsync_directory = snapshot_module._fsync_directory

    def fail_after_publish(path: Path) -> None:
        if path == output.parent:
            raise CategorySnapshotDurabilityError("post-publication fsync failed")
        real_fsync_directory(path)

    monkeypatch.setattr(snapshot_module, "_fsync_directory", fail_after_publish)
    with pytest.raises(CategorySnapshotDurabilityError):
        build_category_snapshot(plan, output, LocalSourceMaterializer(scan))
    assert (output / "manifest.json").is_file()
    assert (output / PUBLICATION_MARKER_NAME).is_file()
    with pytest.raises(ValueError, match="not durable"):
        load_category_snapshot_manifest(output / "manifest.json")
    assert not (tmp_path / ".post-fsync.lock").exists()

    monkeypatch.undo()
    finalize_category_snapshot_publication(output, approved_roots=(tmp_path,))
    assert not (output / PUBLICATION_MARKER_NAME).exists()
    assert load_category_snapshot_manifest(output / "manifest.json")


def test_cleanup_and_lock_cleanup_failures_are_visible(tmp_path, monkeypatch):
    plan, scan = split_plan(tmp_path)
    monkeypatch.setattr(snapshot_module.shutil, "rmtree", lambda *_: (_ for _ in ()).throw(OSError("cleanup")))
    with pytest.raises(CategorySnapshotCleanupError):
        build_category_snapshot(plan, tmp_path / "cleanup", lambda _: (_ for _ in ()).throw(RuntimeError("private")))
    assert not (tmp_path / ".cleanup.lock").exists()
    monkeypatch.undo()
    real_unlink = Path.unlink
    monkeypatch.setattr(snapshot_module.Path, "unlink", lambda self: (_ for _ in ()).throw(OSError("lock")))
    with pytest.raises(CategorySnapshotLockCleanupError):
        build_category_snapshot(plan, tmp_path / "lock", LocalSourceMaterializer(scan))
    monkeypatch.setattr(snapshot_module.Path, "unlink", real_unlink)
