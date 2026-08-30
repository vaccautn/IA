from __future__ import annotations

import hashlib
import io
import json
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
from vacca_bcs.source_split_plan import create_integer_split_plan
from vacca_bcs.integer_snapshot import (
    IntegerSnapshotImageError,
    IntegerSnapshotMaterializationError,
    IntegerSnapshotOutputError,
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
