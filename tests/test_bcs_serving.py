from __future__ import annotations

import copy
from io import BytesIO
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageOps

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
        "observed_classes": [3, 4],
        "missing_classes": [1, 2, 5],
        "allow_partial_class_coverage": True,
        "source_identity_scheme": "local-path-sha256-v1",
        "source_mapping": {
            "3.25": 3, "3.5": 3, "3.75": 4, "4.0": 4, "4.25": 4
        },
        "config": {"imgsz": 32, "allow_partial_class_coverage": True},
        "provenance": {
            "run_id": "c" * 32,
            "domain_id": BCS_DOMAIN_ID,
            "dataset_manifest": {
                "source_schema": "bcs-local-folder-v1",
                "identity_scheme": "local-path-sha256-v1",
                "mapping": {
                    "3.25": 3, "3.5": 3, "3.75": 4, "4.0": 4, "4.25": 4
                },
                "observed_classes": [3, 4],
                "missing_classes": [1, 2, 5],
                "allow_partial_class_coverage": True,
                "split_identity": "a" * 64,
                "sha256": "b" * 64,
            }
        },
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
    assert loaded.lineage.source_schema == "bcs-local-folder-v1"
    assert loaded.lineage.observed_classes == (3, 4)
    assert loaded.lineage.missing_classes == (1, 2, 5)
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
@pytest.mark.parametrize(
    "field",
    [
        "model_state_dict", "config", "domain_id", "observed_classes",
        "missing_classes", "allow_partial_class_coverage", "source_identity_scheme",
        "source_mapping", "provenance",
    ],
)
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


def test_accepts_full_backend_coverage(checkpoint) -> None:
    path, payload = checkpoint
    backend = copy.deepcopy(payload)
    backend.update(
        observed_classes=[1, 2, 3, 4, 5],
        missing_classes=[],
        allow_partial_class_coverage=False,
        source_identity_scheme=None,
        source_mapping=None,
        config={"imgsz": 32, "allow_partial_class_coverage": False},
        provenance={
            "run_id": "c" * 32,
            "domain_id": BCS_DOMAIN_ID,
            "dataset_manifest": {
                "source_schema": "bcs-source-v1",
                "identity_scheme": None,
                "mapping": None,
                "observed_classes": [1, 2, 3, 4, 5],
                "missing_classes": [],
                "allow_partial_class_coverage": False,
                "split_identity": "a" * 64,
                "sha256": "b" * 64,
            }
        },
    )
    torch.save(backend, path)
    loaded = serving.load_bcs_model(path)
    assert loaded.lineage.source_schema == "bcs-source-v1"
    assert loaded.lineage.source_identity_scheme is None
    assert loaded.lineage.source_mapping is None
    assert loaded.lineage.observed_classes == (1, 2, 3, 4, 5)
    assert loaded.lineage.missing_classes == ()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("observed_classes", [4, 3]),
        ("observed_classes", [3, 3, 4]),
        ("observed_classes", [True, 4]),
        ("observed_classes", [3]),
        ("missing_classes", [1, 2]),
        ("allow_partial_class_coverage", False),
        ("source_identity_scheme", "backend-evidence-v1"),
        ("source_mapping", {"3.25": 3}),
    ],
)
def test_rejects_tampered_coverage_before_model_construction(
    checkpoint, monkeypatch, field: str, value: object
) -> None:
    path, payload = checkpoint
    tampered = copy.deepcopy(payload)
    tampered[field] = value
    torch.save(tampered, path)
    calls: list[bool] = []

    def forbidden_model(*args, **kwargs):
        calls.append(True)
        raise AssertionError("model construction must wait for lineage validation")

    monkeypatch.setattr(serving, "BCSOrdinalModel", forbidden_model)
    with pytest.raises(serving.BCSCheckpointLoadError):
        serving.load_bcs_model(path)
    assert calls == []


def test_rejects_tampered_source_manifest_before_model_construction(checkpoint, monkeypatch) -> None:
    path, payload = checkpoint
    tampered = copy.deepcopy(payload)
    tampered["provenance"]["dataset_manifest"]["source_schema"] = "bcs-source-v1"
    torch.save(tampered, path)
    calls: list[bool] = []

    def forbidden_model(*args, **kwargs):
        calls.append(True)
        raise AssertionError("model construction must wait for source validation")

    monkeypatch.setattr(serving, "BCSOrdinalModel", forbidden_model)
    with pytest.raises(serving.BCSCheckpointLoadError):
        serving.load_bcs_model(path)
    assert calls == []


class _InferenceModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor, failure: bool = False) -> None:
        super().__init__()
        self.logits = logits
        self.failure = failure
        self.seen: torch.Tensor | None = None

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.failure:
            raise RuntimeError("private model failure")
        assert not self.training
        assert torch.is_inference_mode_enabled()
        self.seen = image.detach().clone()
        return self.logits.to(image.device)


def _loaded_model(model: torch.nn.Module, imgsz: int = 8) -> serving.LoadedBCSModel:
    lineage = serving.BCSLineageMetadata(
        serving.CHECKPOINT_SCHEMA_VERSION,
        BCS_DOMAIN_ID,
        serving.SNAPSHOT_SCHEMA_VERSION,
        "a" * 64,
        "b" * 64,
        "c" * 32,
        "bcs-local-folder-v1",
        "local-path-sha256-v1",
        (("3.25", 3), ("3.5", 3), ("3.75", 4), ("4.0", 4), ("4.25", 4)),
        (3, 4),
        (1, 2, 5),
        True,
    )
    return serving.LoadedBCSModel(model, imgsz, torch.device("cpu"), lineage)


def _image_bytes(image_format: str = "JPEG", size: tuple[int, int] = (2, 1), exif=None) -> bytes:
    image = Image.new("RGB", size, (20, 80, 140))
    output = BytesIO()
    options = {"format": image_format}
    if exif is not None:
        options["exif"] = exif
    image.save(output, **options)
    return output.getvalue()


def test_infers_continuous_ordinal_score_without_rounding() -> None:
    model = _InferenceModel(torch.tensor([[20.0, 0.0, 0.0, 0.0]]))
    result = serving.BCSInferenceService(_loaded_model(model)).infer(_image_bytes())

    assert result.continuous_score == pytest.approx(3.5)
    assert result.score == result.continuous_score
    assert result.continuous_score != 3
    assert model.seen is not None and model.seen.shape == (1, 3, 8, 8)


def test_preprocessing_is_validation_path_exif_transposed_and_rgb(monkeypatch) -> None:
    calls: list[tuple[int, bool]] = []
    original_build = serving.build_transforms

    def build(imgsz: int, *, train: bool):
        calls.append((imgsz, train))
        return original_build(imgsz, train=train)

    monkeypatch.setattr(serving, "build_transforms", build)
    exif = Image.Exif()
    exif[274] = 6
    payload = _image_bytes(exif=exif)
    model = _InferenceModel(torch.zeros(1, 4))
    service = serving.BCSInferenceService(_loaded_model(model))
    service.infer(payload)

    with Image.open(BytesIO(payload)) as decoded:
        expected = original_build(8, train=False)(
            ImageOps.exif_transpose(decoded).convert("RGB")
        ).unsqueeze(0)
    assert calls == [(8, False)]
    assert model.seen is not None and torch.equal(model.seen, expected)


@pytest.mark.parametrize(
    "logits",
    [torch.zeros(1, 3), torch.full((1, 4), float("nan"))],
)
def test_rejects_invalid_logits(logits: torch.Tensor) -> None:
    model = _InferenceModel(logits)
    with pytest.raises(serving.BCSInferenceExecutionError):
        serving.infer_bcs(_loaded_model(model), _image_bytes())


def test_model_failure_is_typed_and_sanitized() -> None:
    model = _InferenceModel(torch.zeros(1, 4), failure=True)
    with pytest.raises(serving.BCSInferenceExecutionError) as failure:
        serving.infer_bcs(_loaded_model(model), _image_bytes())
    assert "private model failure" not in str(failure.value)


@pytest.mark.parametrize("payload", [b"", b"not an image", _image_bytes("GIF")])
def test_rejects_empty_corrupt_and_unsupported_images(payload: bytes) -> None:
    service = serving.BCSInferenceService(_loaded_model(_InferenceModel(torch.zeros(1, 4))))
    with pytest.raises(serving.BCSInferenceInputError):
        service.infer(payload)


def test_rejects_bomb_and_unsafe_dimensions(monkeypatch) -> None:
    service = serving.BCSInferenceService(_loaded_model(_InferenceModel(torch.zeros(1, 4))))

    def bomb(*args, **kwargs):
        raise Image.DecompressionBombError("private image details")

    monkeypatch.setattr(serving.Image, "open", bomb)
    with pytest.raises(serving.BCSInferenceInputError) as failure:
        service.infer(_image_bytes())
    assert "private image details" not in str(failure.value)

    monkeypatch.undo()
    monkeypatch.setattr(
        serving, "_IMAGE_CONFIG", serving.ImageValidationConfig(max_width=1)
    )
    with pytest.raises(serving.BCSInferenceInputError):
        service.infer(_image_bytes(size=(2, 1)))


def test_repeated_inference_is_deterministic_and_device_bound() -> None:
    model = _InferenceModel(torch.tensor([[0.5, -0.5, 1.0, -1.0]]))
    service = serving.BCSInferenceService(_loaded_model(model, imgsz=16))
    first = service.infer(_image_bytes())
    second = service.infer(_image_bytes())

    assert first == second
    assert model.seen is not None and model.seen.device == torch.device("cpu")
