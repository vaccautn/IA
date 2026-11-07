from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path

import pytest
import torch

import vacca_bcs.checkpoint_io as checkpoint_io


def _set_descriptor(best_digest: str, last_digest: str) -> dict[str, object]:
    return {
        "schema": checkpoint_io.CHECKPOINT_SET_SCHEMA,
        "lineage_schema_version": "bcs-category-coral-results-v1",
        "committed_epoch": 1,
        "best": {"filename": f"generations/{best_digest}.pt", "sha256": best_digest},
        "last": {"filename": f"generations/{last_digest}.pt", "sha256": last_digest},
        "run_id": "a" * 32,
        "domain_id": "bcs-category-1-5-v1",
        "source_schema": "bcs-local-source-v1",
        "snapshot_schema": "bcs-category-snapshot-v1",
        "snapshot_identity": "b" * 64,
        "dataset_manifest_digest": "c" * 64,
        "config_sha256": "d" * 64,
        "observed_classes": [1, 2, 3, 4, 5],
        "missing_classes": [],
        "source_identity_scheme": "local-path-sha256-v1",
        "source_mapping": {"3.25": 1, "3.5": 2, "3.75": 3, "4.0": 4, "4.25": 5},
        "best_epoch": 1,
        "selection_identity": "e" * 64,
        "best_validation": {},
    }


def test_checkpoint_loader_deserializes_the_bytes_that_it_validated(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "checkpoint.pt"
    original = BytesIO()
    torch.save({"marker": "original"}, original)
    raw = original.getvalue()
    path.write_bytes(raw)
    expected = hashlib.sha256(raw).hexdigest()
    real_load = checkpoint_io.torch.load

    def load_and_swap(source, **kwargs):
        assert isinstance(source, BytesIO)
        source.seek(0)
        validated_bytes = source.read()
        path.write_bytes(b"replacement-after-validation")
        return real_load(BytesIO(validated_bytes), **kwargs)

    monkeypatch.setattr(checkpoint_io.torch, "load", load_and_swap)
    loaded = checkpoint_io.load_checkpoint_bytes(
        path,
        approved_roots=(tmp_path,),
        expected_sha256=expected,
    )

    assert loaded.raw == raw
    assert loaded.sha256 == expected
    assert loaded.payload == {"marker": "original"}


def test_checkpoint_loader_resolves_authoritative_checkpoint_set(
    tmp_path: Path,
) -> None:
    generation_dir = tmp_path / "generations"
    generation_dir.mkdir()
    serialized = BytesIO()
    torch.save({"marker": "generation"}, serialized)
    raw = serialized.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    generation = generation_dir / f"{digest}.pt"
    generation.write_bytes(raw)
    pointer = tmp_path / "best.pt"
    pointer.write_bytes(raw)
    (tmp_path / "last.pt").write_bytes(raw)
    descriptor = _set_descriptor(digest, digest)
    (tmp_path / "checkpoint_set.json").write_text(
        json.dumps(descriptor), encoding="utf-8"
    )

    loaded = checkpoint_io.load_checkpoint_bytes(
        pointer, approved_roots=(tmp_path,), expected_sha256=digest
    )

    assert loaded.path == pointer
    assert loaded.payload == {"marker": "generation"}
    assert loaded.sha256 == digest


def test_checkpoint_loader_does_not_require_mutable_alias_files(tmp_path: Path) -> None:
    generation_dir = tmp_path / "generations"
    generation_dir.mkdir()
    serialized = BytesIO()
    torch.save({"marker": "generation"}, serialized)
    raw = serialized.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    generation = generation_dir / f"{digest}.pt"
    generation.write_bytes(raw)
    descriptor = _set_descriptor(digest, digest)
    (tmp_path / "checkpoint_set.json").write_text(json.dumps(descriptor), encoding="utf-8")

    loaded = checkpoint_io.load_checkpoint_bytes(
        tmp_path / "best.pt", approved_roots=(tmp_path,), expected_sha256=digest
    )

    assert loaded.payload == {"marker": "generation"}
    assert loaded.sha256 == digest


@pytest.mark.parametrize(
    "descriptor_mutation",
    [
        lambda descriptor: "not-json",
        lambda descriptor: {**descriptor, "schema": "wrong"},
        lambda descriptor: {
            **descriptor,
            "best": {"filename": "../outside.pt", "sha256": "0" * 64},
        },
        lambda descriptor: {
            **descriptor,
            "best": {"filename": "generations/not-a-digest.pt", "sha256": "not-a-digest"},
        },
    ],
)
def test_checkpoint_loader_rejects_malformed_checkpoint_set(
    tmp_path: Path, descriptor_mutation
) -> None:
    path = tmp_path / "best.pt"
    serialized = BytesIO()
    torch.save({"marker": "alias"}, serialized)
    raw = serialized.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    (tmp_path / "generations").mkdir()
    (tmp_path / "generations" / f"{digest}.pt").write_bytes(raw)
    (tmp_path / "best.pt").write_bytes(raw)
    (tmp_path / "last.pt").write_bytes(raw)
    descriptor = descriptor_mutation(_set_descriptor(digest, digest))
    (tmp_path / "checkpoint_set.json").write_text(
        descriptor if isinstance(descriptor, str) else json.dumps(descriptor),
        encoding="utf-8",
    )

    with pytest.raises(checkpoint_io.CheckpointByteError, match="descriptor|reference"):
        checkpoint_io.load_checkpoint_bytes(path, approved_roots=(tmp_path,))


def test_checkpoint_loader_rejects_missing_generation_and_corrupt_generation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "best.pt"
    serialized = BytesIO()
    torch.save({"marker": "generation"}, serialized)
    raw = serialized.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    path.write_bytes(raw)
    (tmp_path / "last.pt").write_bytes(raw)
    descriptor = _set_descriptor(digest, digest)
    (tmp_path / "checkpoint_set.json").write_text(json.dumps(descriptor), encoding="utf-8")

    with pytest.raises(checkpoint_io.CheckpointByteError, match="unavailable"):
        checkpoint_io.load_checkpoint_bytes(path, approved_roots=(tmp_path,))

    generation = tmp_path / "generations" / f"{digest}.pt"
    generation.parent.mkdir()
    generation.write_bytes(b"corrupt-generation")
    with pytest.raises(checkpoint_io.CheckpointByteError, match="generation digest"):
        checkpoint_io.load_checkpoint_bytes(path, approved_roots=(tmp_path,))
