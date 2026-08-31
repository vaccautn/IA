from __future__ import annotations

import copy
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import vacca_bcs.serving as serving  # noqa: E402
from vacca_bcs.constants import (  # noqa: E402
    BCS_DOMAIN_ID,
    CLASS_NAMES,
    NUM_CLASSES,
    NUM_THRESHOLDS,
    SCORE_BASE,
    SCORE_MAX,
    SCORE_MIN,
    SCORE_STEP,
)
from vacca_bcs.model import BCSOrdinalModel  # noqa: E402


@pytest.fixture
def checkpoint(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    model = BCSOrdinalModel(pretrained=False)
    payload: dict[str, object] = {
        "checkpoint_schema_version": serving.CHECKPOINT_SCHEMA_VERSION,
        "domain_id": BCS_DOMAIN_ID,
        "classes": list(CLASS_NAMES),
        "class_mapping": {name: index for index, name in enumerate(CLASS_NAMES)},
        "score_min": SCORE_MIN,
        "score_max": SCORE_MAX,
        "score_base": SCORE_BASE,
        "score_step": SCORE_STEP,
        "num_classes": NUM_CLASSES,
        "num_thresholds": NUM_THRESHOLDS,
        "snapshot_schema": serving.SNAPSHOT_SCHEMA_VERSION,
        "snapshot_identity": "a" * 64,
        "dataset_manifest_digest": "b" * 64,
        "run_id": "c" * 32,
        "config": {"imgsz": 32},
        "model_state_dict": model.state_dict(),
    }
    path = tmp_path / "best.pt"
    torch.save(payload, path)
    return path, payload


def test_loads_cpu_checkpoint_and_returns_immutable_metadata(checkpoint) -> None:
    path, _ = checkpoint
    loaded = serving.load_bcs_model(path)

    assert loaded.imgsz == 32
    assert loaded.device == torch.device("cpu")
    assert not loaded.model.training
    assert loaded.lineage.domain_id == BCS_DOMAIN_ID
    assert loaded.lineage.snapshot_identity == "a" * 64
    with pytest.raises(FrozenInstanceError):
        loaded.lineage.run_id = "d" * 32
def test_uses_safe_load_and_pretrained_false(checkpoint, monkeypatch) -> None:
    path, _ = checkpoint
    calls: dict[str, object] = {}
    original_load = serving.torch.load
    original_model = serving.BCSOrdinalModel

    def safe_load(*args, **kwargs):
        calls["kwargs"] = kwargs
        return original_load(*args, **kwargs)

    class SpyModel(original_model):
        def __init__(self, *args, **kwargs):
            calls["model_kwargs"] = kwargs
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(serving.torch, "load", safe_load)
    monkeypatch.setattr(serving, "BCSOrdinalModel", SpyModel)
    serving.load_bcs_model(path)

    assert calls["kwargs"] == {"map_location": "cpu", "weights_only": True}
    assert calls["model_kwargs"] == {"pretrained": False}
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checkpoint_schema_version", "wrong"),
        ("domain_id", "wrong"),
        ("classes", ["1", "2"]),
        ("class_mapping", {name: 0 for name in CLASS_NAMES}),
        ("score_step", True),
        ("snapshot_schema", "bcs-integer-snapshot-v1"),
        ("snapshot_identity", "not-a-digest"),
        ("dataset_manifest_digest", "d" * 63),
        ("run_id", "e" * 31),
        ("config", {"imgsz": True}),
        ("config", {"imgsz": 0}),
    ],
)
def test_rejects_tampered_contract(checkpoint, field, value) -> None:
    path, payload = checkpoint
    tampered = copy.deepcopy(payload)
    tampered[field] = value
    torch.save(tampered, path)
    with pytest.raises(serving.BCSCheckpointLoadError):
        serving.load_bcs_model(path)
@pytest.mark.parametrize("field", ["model_state_dict", "config", "domain_id"])
def test_rejects_missing_or_unexpected_fields(checkpoint, field: str) -> None:
    path, payload = checkpoint
    missing = copy.deepcopy(payload)
    del missing[field]
    torch.save(missing, path)
    with pytest.raises(serving.BCSCheckpointLoadError):
        serving.load_bcs_model(path)

    extra = copy.deepcopy(payload)
    extra["secret"] = "must not be accepted"
    torch.save(extra, path)
    with pytest.raises(serving.BCSCheckpointLoadError) as failure:
        serving.load_bcs_model(path)
    assert "must not be accepted" not in str(failure.value)


def test_rejects_corrupt_file_directory_and_architecture(checkpoint, tmp_path: Path) -> None:
    path, payload = checkpoint
    path.write_bytes(b"checkpoint contains private material")
    with pytest.raises(serving.BCSCheckpointLoadError) as failure:
        serving.load_bcs_model(path)
    assert "private material" not in str(failure.value)

    directory = tmp_path / "checkpoint-dir"
    directory.mkdir()
    with pytest.raises(serving.BCSCheckpointLoadError):
        serving.load_bcs_model(directory)

    broken = copy.deepcopy(payload)
    state = broken["model_state_dict"]
    assert isinstance(state, dict)
    del state[next(iter(state))]
    torch.save(broken, path)
    with pytest.raises(serving.BCSCheckpointLoadError):
        serving.load_bcs_model(path)


def test_missing_and_unavailable_cuda_are_typed(checkpoint, monkeypatch, tmp_path: Path) -> None:
    path, _ = checkpoint
    with pytest.raises(serving.BCSCheckpointUnavailableError):
        serving.load_bcs_model(tmp_path / "missing.pt")

    monkeypatch.setattr(serving.torch.cuda, "is_available", lambda: False)
    with pytest.raises(serving.BCSCheckpointUnavailableError):
        serving.load_bcs_model(path, device="cuda:0")


def test_rejects_symlink_when_supported(checkpoint, tmp_path: Path) -> None:
    path, _ = checkpoint
    link = tmp_path / "link.pt"
    try:
        link.symlink_to(path)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable")
    with pytest.raises(serving.BCSCheckpointLoadError):
        serving.load_bcs_model(link)
