from __future__ import annotations

import json
from pathlib import Path
from io import BytesIO
import hashlib
import shutil

import pytest
import torch
from vacca_bcs import serving_package as package
from vacca_bcs.model import BCSOrdinalModel


def _manifest() -> dict:
    return json.loads(package.CATALOG_MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def private_package_integration_snapshot(request, tmp_path_factory):
    if request.node.get_closest_marker("private_model_installed") is None:
        yield
        return
    fixed_destination = package.PRIVATE_PACKAGE_ROOT
    snapshot = tmp_path_factory.mktemp("private-package-snapshot") / "installed-package"
    assert fixed_destination.is_dir(), "producer must install the external ZIP before integration tests"
    shutil.copytree(fixed_destination, snapshot)
    try:
        yield
    finally:
        if fixed_destination.exists():
            shutil.rmtree(fixed_destination)
        shutil.copytree(snapshot, fixed_destination)


def test_manifest_has_exact_nested_schema() -> None:
    manifest = _manifest()
    assert set(manifest) == package._MANIFEST_KEYS
    for name, keys in (
        ("artifact", package._ARTIFACT_KEYS),
        ("source_checkpoint", package._SOURCE_KEYS),
        ("model", package._MODEL_KEYS),
        ("lineage", package._LINEAGE_KEYS),
        ("approval", package._APPROVAL_KEYS),
    ):
        assert set(manifest[name]) == keys
    assert package.validate_catalog_manifest() == manifest


@pytest.mark.private_model
@pytest.mark.private_model_installed
def test_preflight_validates_bytes_and_documents_without_model_construction(monkeypatch) -> None:
    monkeypatch.setattr(
        package,
        "BCSOrdinalModel",
        lambda **kwargs: pytest.fail("preflight must not construct a model"),
    )
    assert package.validate_installed_package()["package_id"] == package.PRIVATE_PACKAGE_ID


@pytest.mark.private_model
@pytest.mark.private_model_installed
def test_shared_manifest_builder_output_is_consumable_and_instance_is_immutable() -> None:
    current = _manifest()
    source = {
        **current["lineage"],
        "epoch": current["source_checkpoint"]["epoch"],
        "best_epoch": current["source_checkpoint"]["best_epoch"],
        "selection_identity": current["source_checkpoint"]["selection_identity"],
        "config": {"imgsz": current["model"]["imgsz"]},
        "num_thresholds": current["model"]["thresholds"],
    }
    artifact = package.PRIVATE_ARTIFACT_PATH.read_bytes()
    produced = package.build_manifest(
        source,
        len(artifact),
        hashlib.sha256(artifact).hexdigest(),
        {
            "model_card": package.PACKAGE_POLICY.model_card_sha256,
            "third_party_notices": package.PACKAGE_POLICY.third_party_notices_sha256,
        },
    )
    assert package.validate_manifest(produced) == current
    reordered = json.loads(json.dumps(produced))
    reordered["lineage"]["source_mapping"] = dict(
        reversed(list(reordered["lineage"]["source_mapping"].items()))
    )
    assert package.validate_manifest(reordered) == reordered
    for mutation in (
        lambda manifest: manifest.update(package_id="different-instance"),
        lambda manifest: manifest["source_checkpoint"].update(epoch=29),
        lambda manifest: manifest["approval"].update(status="approved"),
    ):
        changed = json.loads(json.dumps(produced))
        mutation(changed)
        with pytest.raises(package.ServingPackageError):
            package.validate_manifest(changed)


@pytest.mark.parametrize(
    "change",
    [
        lambda value: value["artifact"].update(unknown=True),
        lambda value: value["artifact"].update(path="../model_state.pt"),
        lambda value: value["artifact"].update(path="models\\model_state.pt"),
        lambda value: value["artifact"].update(sha256="A" * 64),
        lambda value: value["approval"].update(status="approved"),
        lambda value: value["lineage"].update(run_id="not-a-run"),
        lambda value: value.update(local_path="C:\\Users\\author\\model.pt"),
    ],
)
def test_manifest_tampering_types_and_paths_are_rejected(change) -> None:
    manifest = _manifest()
    change(manifest)
    with pytest.raises(package.ServingPackageError):
        package._validate_manifest(manifest)


def test_manifest_rejects_wrong_types_and_non_lowercase_digests() -> None:
    manifest = _manifest()
    manifest["artifact"]["size_bytes"] = True
    with pytest.raises(package.ServingPackageError):
        package._validate_manifest(manifest)
    manifest = _manifest()
    manifest["artifact"]["sha256"] = "A" * 64
    with pytest.raises(package.ServingPackageError):
        package._validate_manifest(manifest)


@pytest.mark.private_model
@pytest.mark.private_model_installed
def test_missing_and_tampered_artifact_are_rejected(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    artifact_path = tmp_path / "model_state.pt"
    manifest = _manifest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    artifact_path.write_bytes(package.PRIVATE_ARTIFACT_PATH.read_bytes())
    source_root = package.PRIVATE_PACKAGE_ROOT
    (tmp_path / "MODEL_CARD.md").write_bytes((source_root / "MODEL_CARD.md").read_bytes())
    (tmp_path / "THIRD_PARTY_NOTICES.md").write_bytes(
        (source_root / "THIRD_PARTY_NOTICES.md").read_bytes()
    )
    assert package.validate_package_directory(tmp_path)["package_id"] == manifest["package_id"]
    artifact_path.write_bytes(b"tampered")
    with pytest.raises(package.ServingPackageError):
        package.load_package_model(tmp_path)
    artifact_path.unlink()
    with pytest.raises(package.ServingPackageError):
        package.load_package_model(tmp_path)


@pytest.mark.private_model
@pytest.mark.private_model_installed
def test_artifact_and_manifest_co_tampering_is_rejected_by_policy(
    monkeypatch, tmp_path: Path
) -> None:
    state = torch.load(
        BytesIO(package.PRIVATE_ARTIFACT_PATH.read_bytes()), map_location="cpu", weights_only=True
    )
    zeroed = {name: value.clone().zero_() for name, value in state.items()}
    buffer = BytesIO()
    torch.save(zeroed, buffer)
    zeroed_bytes = buffer.getvalue()
    manifest = _manifest()
    manifest["artifact"]["size_bytes"] = len(zeroed_bytes)
    manifest["artifact"]["sha256"] = hashlib.sha256(zeroed_bytes).hexdigest()
    root = tmp_path / package.PRIVATE_PACKAGE_ID
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "model_state.pt").write_bytes(zeroed_bytes)
    (root / "MODEL_CARD.md").write_bytes(package.PRIVATE_PACKAGE_ROOT.joinpath("MODEL_CARD.md").read_bytes())
    (root / "THIRD_PARTY_NOTICES.md").write_bytes(
        package.PRIVATE_PACKAGE_ROOT.joinpath("THIRD_PARTY_NOTICES.md").read_bytes()
    )
    monkeypatch.setattr(package, "PRIVATE_PACKAGE_ROOT", root)
    monkeypatch.setattr(package, "PRIVATE_MANIFEST_PATH", root / "manifest.json")
    monkeypatch.setattr(package, "PRIVATE_ARTIFACT_PATH", root / "model_state.pt")
    with pytest.raises(package.ServingPackageError):
        package.validate_installed_package()


@pytest.mark.private_model
@pytest.mark.private_model_installed
def test_symlinked_package_artifact_is_rejected(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    artifact_path = tmp_path / "model_state.pt"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    (tmp_path / "MODEL_CARD.md").write_bytes(package.PRIVATE_PACKAGE_ROOT.joinpath("MODEL_CARD.md").read_bytes())
    (tmp_path / "THIRD_PARTY_NOTICES.md").write_bytes(
        package.PRIVATE_PACKAGE_ROOT.joinpath("THIRD_PARTY_NOTICES.md").read_bytes()
    )
    try:
        artifact_path.symlink_to(package.PRIVATE_ARTIFACT_PATH)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this Windows checkout")
    with pytest.raises(package.ServingPackageError):
        package.load_package_model(tmp_path)


def test_raw_state_dict_only_and_strict_architecture_validation() -> None:
    model = BCSOrdinalModel(pretrained=False)
    state = model.state_dict()
    assert package._validate_state_dict(state, model)
    with pytest.raises(package.ServingPackageError):
        package._validate_state_dict({**state, "optimizer": {}}, model)
    incomplete = dict(state)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(package.ServingPackageError):
        package._validate_state_dict(incomplete, model)


@pytest.mark.private_model
@pytest.mark.private_model_installed
def test_real_installed_package_integrity_and_metadata() -> None:
    manifest = package.validate_installed_package()
    loaded = package.load_installed_bcs_model()
    assert manifest["approval"]["status"] == "experimental"
    assert manifest["approval"]["serving_status"] == "not_approved"
    assert manifest["approval"]["validation_status"] == "failed"
    assert loaded.lineage.run_id == manifest["lineage"]["run_id"]
    assert loaded.checkpoint is None
    assert loaded.model.training is False
