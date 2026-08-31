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

from vacca_bcs.source_client import (
    BCSEvidencePayload,
    BCSSourceEvidence,
    BCSSourceEvaluationRow,
    BCSSourceExport,
)
from vacca_bcs.local_source import LocalSourceMaterializer, scan_local_source
from vacca_bcs.source_plan import (
    SourceCandidate,
    SourceExclusion,
    SourcePlan,
    SourceProvenance,
    normalize_backend_source_export,
    normalize_local_source_scan,
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
    IntegerSnapshotLegacyManifestError,
    IntegerSnapshotManifestValidationError,
    build_integer_snapshot,
    load_integer_snapshot_manifest,
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


def local_split_plan(tmp_path: Path):
    write = tmp_path / "source"
    (write / "3.25").mkdir(parents=True)
    (write / "4.0").mkdir()
    (write / "3.25" / "cow.jpg").write_bytes(image_bytes("JPEG"))
    (write / "4.0" / "cow.png").write_bytes(image_bytes("PNG"))
    scan = scan_local_source(write)
    return create_integer_split_plan(
        normalize_local_source_scan(scan), seed=9, val_ratio=0
    ), scan


def test_builds_and_loads_local_lineage_without_backend_fields(tmp_path: Path):
    plan, scan = local_split_plan(tmp_path)
    snapshot = build_integer_snapshot(
        plan, tmp_path / "snapshot", LocalSourceMaterializer(scan)
    )
    manifest = json.loads(snapshot.manifest_json)
    assert manifest["source_schema"] == "bcs-local-folder-v1"
    assert manifest["identity_scheme"] == "local-path-sha256-v1"
    assert manifest["mapping"] == {
        "3.25": 3, "3.5": 3, "3.75": 4, "4.0": 4, "4.25": 4
    }
    assert manifest["observed_classes"] == [3, 4]
    assert str(tmp_path) not in snapshot.manifest_json
    assert all("evidence_id" not in item for item in manifest["records"])
    assert all("evaluation_id" not in item for item in manifest["records"])
    assert [item["relative_path"] for item in manifest["records"]] == [
        f"train/{item.bcs_score}/{item.record_id}{Path(item.relative_path).suffix}"
        for item in snapshot.records
    ]
    assert load_integer_snapshot_manifest(snapshot.output_root / "manifest.json") == manifest


def test_builds_canonical_backend_lineage_with_legacy_manifest_shape(tmp_path: Path):
    plan = create_integer_split_plan(
        normalize_backend_source_export(
            BCSSourceExport(
                "bcs-source-v1",
                (BCSSourceEvaluationRow(2, 7, 8, 3, (BCSSourceEvidence(11, "safe"),)),),
            )
        ),
        seed=9,
        val_ratio=0,
    )
    snapshot = build_integer_snapshot(
        plan, tmp_path / "backend", materializer({11: image_bytes("JPEG")})
    )
    manifest = json.loads(snapshot.manifest_json)
    assert manifest["source_schema"] == "bcs-source-v1"
    assert manifest["records"][0]["evidence_id"] == 11
    assert "record_id" not in manifest["records"][0]
    assert load_integer_snapshot_manifest(snapshot.output_root / "manifest.json") == manifest


def test_local_manifest_is_deterministic_and_rejects_variant_tampering(tmp_path: Path):
    plan, scan = local_split_plan(tmp_path)
    first = build_integer_snapshot(
        plan, tmp_path / "one", LocalSourceMaterializer(scan)
    )
    second = build_integer_snapshot(
        plan, tmp_path / "two", LocalSourceMaterializer(scan)
    )
    assert first.manifest_json == second.manifest_json
    manifest = json.loads(first.manifest_json)
    for tamper in (
        lambda value: value.update(mapping={"3.25": 3}),
        lambda value: value.update(observed_classes=[3]),
        lambda value: value.update(source_schema="bcs-source-v1"),
        lambda value: value.pop("identity_scheme"),
        lambda value: value["records"][0].update(evidence_id=1),
        lambda value: value["records"][0]["provenance"][0].update(source_label="5"),
        lambda value: value["records"][0].update(relative_path="../escape.jpg"),
        lambda value: value["split_plan"].update(candidate_record_ids=[]),
        lambda value: value.update(unexpected=None),
    ):
        changed = json.loads(json.dumps(manifest))
        tamper(changed)
        with pytest.raises(IntegerSnapshotManifestValidationError):
            load_integer_snapshot_manifest(changed)


def test_local_materializer_failure_cleans_transaction_and_leaks_no_root(tmp_path: Path):
    plan, scan = local_split_plan(tmp_path)
    (scan.root / "3.25" / "cow.jpg").write_bytes(b"replacement")
    with pytest.raises(IntegerSnapshotMaterializationError) as failure:
        build_integer_snapshot(plan, tmp_path / "local-failure", LocalSourceMaterializer(scan))
    assert str(scan.root) not in str(failure.value)
    assert not (tmp_path / "local-failure").exists()
    assert not list(tmp_path.glob(".local-failure.staging-*"))


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
    assert manifest["manifest_schema_version"] == "bcs-integer-snapshot-v2"
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


def _manifest_for_validation(tmp_path: Path) -> tuple[Path, dict]:
    payload = image_bytes("JPEG")
    snapshot = build_integer_snapshot(
        split_plan(
            candidate(8, 1),
            exclusions=(SourceExclusion(109, 9, 3, "empty_storage_key"),),
        ),
        tmp_path / "snapshot",
        materializer({8: payload}),
    )
    return snapshot.output_root / "manifest.json", json.loads(snapshot.manifest_json)

def test_v2_manifest_round_trips_complete_lineage_without_storage_keys(tmp_path: Path):
    path, manifest = _manifest_for_validation(tmp_path)

    assert tuple(manifest[key] for key in ("domain_id", "class_values", "source_schema")) == (
        "bcs-integer-1-5",
        [1, 2, 3, 4, 5],
        "bcs-source-v1",
    )
    assert manifest["class_mapping"] == {str(score): index for index, score in enumerate(range(1, 6))}
    assert [manifest[key] for key in ("score_min", "score_max", "score_base", "score_step", "num_classes", "num_thresholds")] == [1, 5, 1, 1, 5, 4]
    assert manifest["split_plan"]["candidate_evidence_ids"] == [8]
    assert manifest["split_plan"]["excluded_evidence_ids"] == [9]
    assert manifest["split_plan"]["canonical_val_ratio"] == "0"
    assert load_integer_snapshot_manifest(path) == manifest
    assert "safe-8" not in path.read_text(encoding="utf-8")

@pytest.mark.parametrize(
    "field,value",
    [
        ("domain_id", "other-domain"),
        ("class_values", [1, 2, 3, 4, 6]),
        ("class_mapping", {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5}),
        ("score_min", 0),
        ("num_classes", 4),
        ("num_thresholds", 5),
        ("source_schema", "bcs-source-v2"),
    ],
)
def test_manifest_rejects_tampered_domain_scale_and_source_lineage(
    tmp_path: Path, field: str, value: object
):
    _, manifest = _manifest_for_validation(tmp_path)
    manifest[field] = value

    with pytest.raises(IntegerSnapshotManifestValidationError):
        load_integer_snapshot_manifest(manifest)

@pytest.mark.parametrize("legacy_classes", (["1", "2", "3", "4", "5"], [3.25, 3.5, 3.75, 4.0, 4.25]))
def test_manifest_rejects_integer_v1_and_fractional_legacy_manifests(
    tmp_path: Path, legacy_classes: list[object]
):
    _, manifest = _manifest_for_validation(tmp_path)
    manifest["manifest_schema_version"] = "bcs-integer-snapshot-v1"
    manifest["class_values"] = legacy_classes

    with pytest.raises(IntegerSnapshotLegacyManifestError) as failure:
        load_integer_snapshot_manifest(manifest)
    assert "safe-8" not in str(failure.value)

@pytest.mark.parametrize(
    "tamper",
    [
        lambda m: m["records"][0].update(relative_path="../1/8.jpg"),
        lambda m: m["records"][0].update(relative_path="train/2/8.jpg"),
        lambda m: m["records"][0].update(sha256="x" * 64),
        lambda m: m["records"][0].update(evidence_id=9),
        lambda m: m["records"][0]["provenance"][0].update(evaluation_id=0),
        lambda m: m["counts"]["train"].__setitem__(0, 2),
        lambda m: m.update(num_classes=True),
        lambda m: m["class_values"].__setitem__(0, True),
        lambda m: m["records"][0].update(bcs_score=True),
        lambda m: m["records"][0].update(evidence_id=True),
        lambda m: m["counts"]["train"].__setitem__(0, False),
        lambda m: m["records"][0]["provenance"][0].update(evidence_id=True),
        lambda m: m["split_plan"].update(identity_digest="0" * 63),
        lambda m: m["split_plan"].update(seed=True),
        lambda m: m["split_plan"].update(canonical_val_ratio="0.0"),
        lambda m: m["split_plan"].update(candidate_evidence_ids=[999]),
        lambda m: m["split_plan"].update(excluded_evidence_ids=[]),
        lambda m: m["split_plan"].update(counts={"train": [0] * 5, "val": [0] * 5}),
        lambda m: m.update(unexpected=None),
        lambda m: m.pop("records"),
    ],
)
def test_manifest_rejects_path_count_digest_provenance_and_bool_edges(tmp_path: Path, tamper):
    _, manifest = _manifest_for_validation(tmp_path)
    tamper(manifest)

    with pytest.raises(IntegerSnapshotManifestValidationError):
        load_integer_snapshot_manifest(manifest)

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
