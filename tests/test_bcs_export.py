from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pytest
import torch

from scripts.export_bcs_serving_package import _compare_states, _project_state
from vacca_bcs.serving_package import ServingPackageError

PACKAGE_DOCS = Path("models/catalog/bcs-category-coral-2026-09-04")


def test_export_projection_keeps_only_tensor_state_and_exact_values() -> None:
    source = OrderedDict(weight=torch.arange(4, dtype=torch.float32), counter=torch.tensor(2))
    projection = _project_state(source)
    _compare_states(source, projection)
    assert set(projection) == {"weight", "counter"}
    assert all(isinstance(value, torch.Tensor) for value in projection.values())


def test_export_projection_rejects_metadata_and_state_mismatch() -> None:
    with pytest.raises(ValueError):
        _project_state({"optimizer": {}, "weight": torch.ones(1)})
    with pytest.raises(ValueError):
        _compare_states({"weight": torch.ones(1)}, {"weight": torch.zeros(1)})


def test_export_rejects_existing_immutable_destination_before_source_work(tmp_path: Path) -> None:
    from scripts import export_bcs_serving_package as exporter

    target = tmp_path / exporter.PACKAGE_POLICY.archive_directory
    target.mkdir()
    sentinel = target / "sentinel"
    sentinel.write_bytes(b"unchanged")
    with pytest.raises(FileExistsError):
        exporter.export(
            output_root=tmp_path,
            model_card_path=PACKAGE_DOCS / "MODEL_CARD.md",
            third_party_notices_path=PACKAGE_DOCS / "THIRD_PARTY_NOTICES.md",
        )
    assert sentinel.read_bytes() == b"unchanged"


@pytest.mark.private_source_model
def test_export_cleans_staging_when_validation_fails(
    tmp_path: Path, monkeypatch, private_source_checkpoint: Path
) -> None:
    from scripts import export_bcs_serving_package as exporter

    def fail_validation(*args, **kwargs):
        raise RuntimeError("synthetic staging validation failure")

    monkeypatch.setattr(exporter, "validate_package_directory", fail_validation)
    with pytest.raises(RuntimeError, match="synthetic staging validation failure"):
        exporter.export(
            output_root=tmp_path,
            source_path=private_source_checkpoint,
            model_card_path=PACKAGE_DOCS / "MODEL_CARD.md",
            third_party_notices_path=PACKAGE_DOCS / "THIRD_PARTY_NOTICES.md",
        )
    assert not (tmp_path / "bcs-category-coral-2026-09-04").exists()
    assert list(tmp_path.glob(".bcs-category-coral-2026-09-04.build-*")) == []
    assert list(tmp_path.glob(".bcs-category-coral-2026-09-04.delivery-*")) == []


def test_export_rejects_missing_publication_document_before_destination(tmp_path: Path) -> None:
    from scripts import export_bcs_serving_package as exporter

    with pytest.raises(ServingPackageError):
        exporter.export(
            output_root=tmp_path,
            model_card_path=tmp_path / "missing-card.md",
            third_party_notices_path=PACKAGE_DOCS / "THIRD_PARTY_NOTICES.md",
        )
    assert not (tmp_path / "bcs-category-coral-2026-09-04").exists()
    assert list(tmp_path.glob(".bcs-category-coral-2026-09-04.build-*")) == []
    assert list(tmp_path.glob(".bcs-category-coral-2026-09-04.delivery-*")) == []


@pytest.mark.private_source_model
def test_export_pair_is_atomic_when_sidecar_write_fails_and_retry_works(
    tmp_path: Path, monkeypatch, private_source_checkpoint: Path
) -> None:
    from scripts import export_bcs_serving_package as exporter

    original = exporter._write_staged

    def fail_sidecar(path, payload):
        if path.name.endswith(".sha256"):
            raise OSError("sidecar boundary failure")
        return original(path, payload)

    monkeypatch.setattr(exporter, "_write_staged", fail_sidecar)
    with pytest.raises(OSError, match="sidecar boundary failure"):
        exporter.export(output_root=tmp_path, source_path=private_source_checkpoint, model_card_path=PACKAGE_DOCS / "MODEL_CARD.md", third_party_notices_path=PACKAGE_DOCS / "THIRD_PARTY_NOTICES.md")
    final_dir = tmp_path / exporter.PACKAGE_POLICY.archive_directory
    assert not final_dir.exists()
    assert list(tmp_path.glob(".*.build-*")) == []
    assert list(tmp_path.glob(".*.delivery-*")) == []

    monkeypatch.setattr(exporter, "_write_staged", original)
    assert exporter.export(output_root=tmp_path, source_path=private_source_checkpoint, model_card_path=PACKAGE_DOCS / "MODEL_CARD.md", third_party_notices_path=PACKAGE_DOCS / "THIRD_PARTY_NOTICES.md")
    assert final_dir.is_dir()


@pytest.mark.private_source_model
def test_export_pair_is_atomic_when_final_directory_rename_fails(
    tmp_path: Path, monkeypatch, private_source_checkpoint: Path
) -> None:
    from scripts import export_bcs_serving_package as exporter

    original_rename = exporter.os.rename

    def fail_final_rename(source, destination):
        if Path(destination).name == exporter.PACKAGE_POLICY.archive_directory:
            raise OSError("final directory boundary failure")
        return original_rename(source, destination)

    monkeypatch.setattr(exporter.os, "rename", fail_final_rename)
    with pytest.raises(OSError, match="final directory boundary failure"):
        exporter.export(output_root=tmp_path, source_path=private_source_checkpoint, model_card_path=PACKAGE_DOCS / "MODEL_CARD.md", third_party_notices_path=PACKAGE_DOCS / "THIRD_PARTY_NOTICES.md")
    assert not (tmp_path / exporter.PACKAGE_POLICY.archive_directory).exists()
    assert list(tmp_path.glob(".*.delivery-*")) == []
