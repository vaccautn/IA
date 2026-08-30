from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from vacca_bcs.source_client import BCSEvidencePayload
from vacca_bcs.source_plan import (
    SourceCandidate,
    SourceExclusion,
    SourcePlan,
    SourceProvenance,
)
from vacca_bcs.source_split_plan import (
    IntegerSplitInputError,
    create_integer_split_plan,
    validate_integer_split_plan,
)
from vacca_bcs.integer_snapshot import (
    IntegerSnapshotCleanupError,
    IntegerSnapshotDurabilityError,
    IntegerSnapshotImageError,
    IntegerSnapshotLockCleanupError,
    IntegerSnapshotLockError,
    IntegerSnapshotMaterializationError,
    IntegerSnapshotOutputError,
    IntegerSnapshotPublicationError,
    build_integer_snapshot,
)


def candidate(evidence_id: int, score: int, key: str = "safe") -> SourceCandidate:
    evaluation_id = evidence_id + 100
    return SourceCandidate(
        evaluation_id=evaluation_id,
        session_id=evidence_id + 200,
        animal_id=evidence_id + 300,
        evidence_id=evidence_id,
        storage_key=f"{key}-{evidence_id}",
        bcs_score=score,
        provenance=(SourceProvenance(evidence_id, evaluation_id),),
    )


def split_plan(*candidates: SourceCandidate, exclusions=()):
    source = SourcePlan(tuple(candidates), tuple(exclusions), (0,) * 5)
    return create_integer_split_plan(source, seed=9, val_ratio=0)


def image_bytes(format_name: str) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (2, 2), (20, 40, 60)).save(stream, format=format_name)
    return stream.getvalue()


def materializer(payloads: dict[int, bytes]):
    def resolve(evidence_id: int) -> BCSEvidencePayload:
        payload = payloads[evidence_id]
        return BCSEvidencePayload(
            evidence_id, payload, hashlib.sha256(payload).hexdigest()
        )

    return resolve


def test_builds_jpeg_png_layout_manifest_counts_and_safe_provenance(tmp_path: Path):
    jpeg = image_bytes("JPEG")
    png = image_bytes("PNG")
    plan = split_plan(candidate(8, 1, "../../unsafe"), candidate(2, 2))

    snapshot = build_integer_snapshot(
        plan, tmp_path / "snapshot", materializer({8: jpeg, 2: png})
    )
    manifest = json.loads(snapshot.manifest_json)

    assert sorted(
        path.relative_to(snapshot.output_root).as_posix()
        for path in snapshot.output_root.rglob("*.jpg")
    ) == ["train/1/8.jpg"]
    assert sorted(
        path.relative_to(snapshot.output_root).as_posix()
        for path in snapshot.output_root.rglob("*.png")
    ) == ["train/2/2.png"]
    assert manifest["manifest_schema_version"] == "bcs-integer-snapshot-v1"
    assert manifest["counts"] == {"train": [1, 1, 0, 0, 0], "val": [0, 0, 0, 0, 0]}
    assert manifest["exclusions"] == []
    assert [record["relative_path"] for record in manifest["records"]] == [
        "train/1/8.jpg",
        "train/2/2.png",
    ]
    assert manifest["records"][0]["provenance"] == [
        {"evidence_id": 8, "evaluation_id": 108}
    ]
    assert "unsafe" not in snapshot.manifest_json
    assert "unsafe" not in repr(snapshot)


def test_manifest_and_bytes_are_deterministic_across_output_roots(tmp_path: Path):
    payloads = {1: image_bytes("JPEG"), 2: image_bytes("PNG")}
    plan = split_plan(candidate(1, 1), candidate(2, 1))
    first = build_integer_snapshot(plan, tmp_path / "one", materializer(payloads))
    second = build_integer_snapshot(plan, tmp_path / "two", materializer(payloads))

    assert first.manifest_json == second.manifest_json


def test_empty_plan_preserves_exclusions_and_creates_empty_class_dirs(tmp_path: Path):
    exclusion = SourceExclusion(9, 90, 3, "empty_storage_key")
    snapshot = build_integer_snapshot(
        split_plan(exclusions=(exclusion,)), tmp_path / "empty", lambda _: None
    )
    manifest = json.loads(snapshot.manifest_json)

    assert snapshot.records == ()
    assert manifest["records"] == []
    assert manifest["exclusions"][0]["reason"] == "empty_storage_key"
    assert all(
        (tmp_path / "empty" / split / str(score)).is_dir()
        for split in ("train", "val")
        for score in range(1, 6)
    )


@pytest.mark.parametrize(
    "payload", [b"not-an-image", image_bytes("BMP"), image_bytes("JPEG")[:10]]
)
def test_invalid_or_unsupported_payloads_are_typed_sanitized_and_transactional(
    tmp_path: Path, payload: bytes
):
    plan = split_plan(candidate(7, 4, "secret-key"))
    with pytest.raises(IntegerSnapshotImageError) as failure:
        build_integer_snapshot(plan, tmp_path / "bad", materializer({7: payload}))
    assert "secret-key" not in str(failure.value)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    "payload",
    [
        BCSEvidencePayload(8, b"x", ""),
        BCSEvidencePayload(7, b"x", "0" * 64),
    ],
)
def test_materializer_identity_and_digest_mismatches_are_typed(
    tmp_path: Path, payload: BCSEvidencePayload
):
    with pytest.raises(IntegerSnapshotMaterializationError):
        build_integer_snapshot(
            split_plan(candidate(7, 1)), tmp_path / "mismatch", lambda _: payload
        )
    assert not (tmp_path / "mismatch").exists()


def test_existing_output_is_refused_and_midway_failure_cleans_staging(tmp_path: Path):
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "sentinel"
    sentinel.write_bytes(b"keep")
    with pytest.raises(IntegerSnapshotOutputError):
        build_integer_snapshot(
            split_plan(candidate(1, 1)),
            existing,
            materializer({1: image_bytes("JPEG")}),
        )
    assert sentinel.read_bytes() == b"keep"

    jpeg = image_bytes("JPEG")
    sha256 = hashlib.sha256(jpeg).hexdigest()

    def fails_midway(evidence_id: int):
        if evidence_id == 2:
            raise RuntimeError("backend failure")
        return BCSEvidencePayload(evidence_id, jpeg, sha256)

    with pytest.raises(IntegerSnapshotMaterializationError):
        build_integer_snapshot(
            split_plan(candidate(1, 1), candidate(2, 1)),
            tmp_path / "partial",
            fails_midway,
        )
    assert not (tmp_path / "partial").exists()
    assert not list(tmp_path.glob(".partial.staging-*"))


def test_malformed_plan_fails_validation_before_materialization(tmp_path: Path):
    source = split_plan(candidate(1, 1))
    calls = 0

    def materialize(evidence_id: int):
        nonlocal calls
        calls += 1
        return BCSEvidencePayload(evidence_id, image_bytes("JPEG"), "wrong")

    malformed = (
        replace(source, counts=replace(source.counts, train=(0, 0, 0, 0, 0))),
        replace(source, identity=replace(source.identity, digest="0" * 64)),
    )
    for invalid in malformed:
        with pytest.raises(IntegerSplitInputError):
            validate_integer_split_plan(invalid)
        with pytest.raises(IntegerSplitInputError):
            build_integer_snapshot(invalid, tmp_path / f"bad-{calls}", materialize)
    assert calls == 0


def test_dangling_final_symlink_is_refused_when_supported(tmp_path: Path):
    target = tmp_path / "dangling"
    try:
        os.symlink(tmp_path / "missing", target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are unavailable")
    with pytest.raises(IntegerSnapshotOutputError):
        build_integer_snapshot(
            split_plan(candidate(1, 1)),
            target,
            materializer({1: image_bytes("JPEG")}),
        )


def test_sibling_lock_prevents_concurrent_publication_and_is_released(tmp_path: Path):
    lock = tmp_path / ".locked.lock"
    lock.write_bytes(b"held")
    with pytest.raises(IntegerSnapshotLockError):
        build_integer_snapshot(
            split_plan(candidate(1, 1)),
            tmp_path / "locked",
            materializer({1: image_bytes("JPEG")}),
        )
    lock.unlink()
    build_integer_snapshot(
        split_plan(candidate(1, 1)),
        tmp_path / "locked",
        materializer({1: image_bytes("JPEG")}),
    )
    assert not lock.exists()


def test_publication_rechecks_final_target_after_staging(tmp_path: Path, monkeypatch):
    import vacca_bcs.integer_snapshot as snapshot_module

    target = tmp_path / "raced"
    real_lexists = snapshot_module.os.path.lexists
    checks = 0

    def raced_lexists(path):
        nonlocal checks
        if Path(path) == target:
            checks += 1
            if checks >= 3:
                return True
        return real_lexists(path)

    monkeypatch.setattr(snapshot_module.os.path, "lexists", raced_lexists)
    with pytest.raises(IntegerSnapshotOutputError):
        build_integer_snapshot(
            split_plan(candidate(1, 1)), target, materializer({1: image_bytes("JPEG")})
        )
    assert not list(tmp_path.glob(".raced.lock"))


def test_durability_and_rename_failures_leave_no_published_root(
    tmp_path: Path, monkeypatch
):
    import vacca_bcs.integer_snapshot as snapshot_module

    monkeypatch.setattr(
        snapshot_module.os,
        "fsync",
        lambda _: (_ for _ in ()).throw(OSError("fsync")),
    )
    with pytest.raises(IntegerSnapshotDurabilityError):
        build_integer_snapshot(
            split_plan(candidate(1, 1)),
            tmp_path / "fsync",
            materializer({1: image_bytes("JPEG")}),
        )
    assert not (tmp_path / "fsync").exists()
    monkeypatch.undo()

    monkeypatch.setattr(
        snapshot_module.os,
        "rename",
        lambda *args: (_ for _ in ()).throw(OSError("rename")),
    )
    with pytest.raises(IntegerSnapshotPublicationError):
        build_integer_snapshot(
            split_plan(candidate(1, 1)),
            tmp_path / "rename",
            materializer({1: image_bytes("JPEG")}),
        )
    assert not (tmp_path / "rename").exists()


def test_cleanup_failure_is_typed_visible_and_chains_original_failure(
    tmp_path: Path, monkeypatch
):
    import vacca_bcs.integer_snapshot as snapshot_module

    real_remove = shutil.rmtree
    monkeypatch.setattr(
        snapshot_module.shutil,
        "rmtree",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("cleanup")),
    )
    with pytest.raises(IntegerSnapshotCleanupError) as failure:
        build_integer_snapshot(
            split_plan(candidate(1, 1, "secret-key")),
            tmp_path / "cleanup",
            lambda _: (_ for _ in ()).throw(RuntimeError("payload")),
        )
    assert ".cleanup.staging-" in str(failure.value)
    assert any("original failure" in note for note in failure.value.__notes__)
    assert "secret-key" not in str(failure.value)
    monkeypatch.undo()
    for staging in tmp_path.glob(".cleanup.staging-*"):
        real_remove(staging)


def test_lock_cleanup_failure_is_typed_after_publication(tmp_path: Path, monkeypatch):
    import vacca_bcs.integer_snapshot as snapshot_module

    real_unlink = snapshot_module.os.unlink
    monkeypatch.setattr(
        snapshot_module.os,
        "unlink",
        lambda path: (_ for _ in ()).throw(OSError("lock cleanup")),
    )
    with pytest.raises(IntegerSnapshotLockCleanupError):
        build_integer_snapshot(
            split_plan(candidate(1, 1)),
            tmp_path / "lock-cleanup",
            materializer({1: image_bytes("JPEG")}),
        )
    assert (tmp_path / "lock-cleanup").exists()
    monkeypatch.undo()
    real_unlink(tmp_path / ".lock-cleanup.lock")
    shutil.rmtree(tmp_path / "lock-cleanup")
