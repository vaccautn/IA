from __future__ import annotations

from pathlib import Path
import sys
import pytest
import torch
from PIL import Image
import os
import subprocess
import yaml
import hashlib
import json
import random

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from vacca_bcs.constants import CLASS_NAMES  # noqa: E402
from vacca_bcs.dataset import BCSFolderDataset, Letterbox, build_transforms  # noqa: E402
from vacca_bcs.model import (  # noqa: E402
    BCSOrdinalModel,
    CORALHead,
    coral_loss,
    encode_levels,
    predict,
)
import scripts.train_bcs_ordinal as trainer  # noqa: E402
from scripts.train_bcs_ordinal import load_config, set_seed  # noqa: E402
from scripts.train_bcs_ordinal import _build_provenance, _runtime_identity  # noqa: E402
from scripts.train_bcs_ordinal import (  # noqa: E402
    _atomic_torch_save,
    _atomic_write_json,
    _capture_rng_state,
    _prepare_output_dir,
    _restore_rng_state,
)

def test_coral_level_encoding_for_all_classes() -> None:
    levels = torch.arange(5)
    expected = torch.tensor(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(encode_levels(levels), expected)


def test_predict_maps_passed_thresholds_to_class_and_score() -> None:
    logits = torch.tensor(
        [
            [-10.0, -10.0, -10.0, -10.0],
            [10.0, -10.0, -10.0, -10.0],
            [10.0, 10.0, -10.0, -10.0],
            [10.0, 10.0, 10.0, -10.0],
            [10.0, 10.0, 10.0, 10.0],
        ]
    )
    class_idx, scores = predict(logits)
    assert torch.equal(class_idx, torch.arange(5))
    assert torch.equal(scores, torch.tensor([3.25, 3.5, 3.75, 4.0, 4.25]))


def test_coral_loss_is_finite_and_lower_for_correct_levels() -> None:
    levels = torch.tensor([0, 1, 2, 3, 4])
    correct_logits = torch.tensor(
        [
            [-8.0, -8.0, -8.0, -8.0],
            [8.0, -8.0, -8.0, -8.0],
            [8.0, 8.0, -8.0, -8.0],
            [8.0, 8.0, 8.0, -8.0],
            [8.0, 8.0, 8.0, 8.0],
        ]
    )
    wrong_logits = -correct_logits
    correct_loss = coral_loss(correct_logits, levels)
    wrong_loss = coral_loss(wrong_logits, levels)
    assert torch.isfinite(correct_loss)
    assert correct_loss < wrong_loss


def test_coral_head_produces_ordered_logits() -> None:
    torch.manual_seed(0)
    head = CORALHead(in_features=4, num_classes=len(CLASS_NAMES))
    biases = head.ordered_biases()
    assert torch.all(biases[1:] > biases[:-1])

    features = torch.randn(8, 4)
    logits = head(features)
    assert logits.shape == (8, len(CLASS_NAMES) - 1)
    # Increasing thresholds must yield strictly decreasing logits per row.
    assert torch.all(logits[:, 1:] < logits[:, :-1])

    class_idx, scores = predict(logits)
    assert torch.all((class_idx >= 0) & (class_idx < len(CLASS_NAMES)))
    assert torch.all((scores >= 3.25) & (scores <= 4.25))

def test_real_dataset_transforms_and_model_training_step(tmp_path: Path) -> None:
    torch.manual_seed(11)
    image = Image.new("RGB", (12, 20), (32, 96, 160))
    letterboxed = Letterbox(32)(image)
    assert letterboxed.size == (32, 32)

    validation_transform = build_transforms(32, train=False)
    training_transform = build_transforms(32, train=True)
    assert validation_transform(image).shape == (3, 32, 32)
    assert training_transform(image).shape == (3, 32, 32)

    dataset_root = tmp_path / "dataset"
    for class_name in CLASS_NAMES:
        class_dir = dataset_root / class_name
        class_dir.mkdir(parents=True)
    image.save(dataset_root / CLASS_NAMES[0] / "low.jpg")
    image.save(dataset_root / CLASS_NAMES[-1] / "high.jpg")

    dataset = BCSFolderDataset(dataset_root, train=False, imgsz=32)
    samples = [dataset[index] for index in range(len(dataset))]
    images = torch.stack([sample[0] for sample in samples])
    levels = torch.tensor([sample[1] for sample in samples])
    assert levels.tolist() == [0, len(CLASS_NAMES) - 1]
    assert images.shape == (2, 3, 32, 32)
    assert torch.isfinite(images).all()
    assert dataset.class_counts == {
        CLASS_NAMES[0]: 1,
        CLASS_NAMES[1]: 0,
        CLASS_NAMES[2]: 0,
        CLASS_NAMES[3]: 0,
        CLASS_NAMES[4]: 1,
    }

    model = BCSOrdinalModel(pretrained=False)
    with torch.no_grad():
        model.head.weight.zero_()
    model.eval()
    logits = model(images)
    predicted_levels, scores = model.predict(images)
    assert logits.shape == (2, len(CLASS_NAMES) - 1)
    assert predicted_levels.shape == (2,)
    assert scores.shape == (2,)
    assert torch.isfinite(logits).all()
    assert torch.is_floating_point(scores)
    assert torch.equal(scores, torch.full((2,), 3.25))

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer.zero_grad(set_to_none=True)
    loss = coral_loss(model(images), levels)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    optimizer.step()
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, model.parameters())
    )
    assert optimizer.state
    assert all(
        torch.isfinite(value).all()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
def _write_config(tmp_path: Path, name: str = "config.yaml", **overrides: object) -> Path:
    config = {
        "data_dir": "data/bcs-cls",
        "output": str(tmp_path / "out"),
        "epochs": 2,
        "batch_size": 2,
        "lr": 0.001,
        "weight_decay": 0.0,
        "optimizer": "AdamW",
        "patience": 2,
        "num_workers": 0,
        "imgsz": 32,
        "device": "cpu",
        "seed": 0,
    }
    config.update(overrides)
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_load_config_accepts_cosine_or_default_schedule(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path, "a.yaml", lr_schedule="cosine"))
    assert config["lr_schedule"] == "cosine"
    default = load_config(_write_config(tmp_path, "b.yaml"))
    assert default["lr_schedule"] == "cosine"


def test_load_config_rejects_unsupported_schedule(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lr_schedule"):
        load_config(_write_config(tmp_path, lr_schedule="step"))


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("epochs", 0, "epochs"),
        ("batch_size", 0, "batch_size"),
        ("patience", 0, "patience"),
        ("warmup_epochs", -1, "warmup_epochs"),
        ("lr", 0, "lr"),
        ("weight_decay", -0.1, "weight_decay"),
        ("num_workers", -1, "num_workers"),
        ("imgsz", 0, "imgsz"),
        ("seed", -1, "seed"),
    ],
)
def test_load_config_rejects_invalid_training_boundaries(
    tmp_path: Path, key: str, value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(_write_config(tmp_path, **{key: value}))


def test_load_config_rejects_warmup_longer_than_training(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="warmup_epochs"):
        load_config(_write_config(tmp_path, epochs=2, warmup_epochs=3))

def _seed_run_artifacts(output_dir: Path) -> None:
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True)
    (output_dir / "results.csv").write_text("epoch\n1\n", encoding="utf-8")
    (weights_dir / "best.pt").write_bytes(b"fake")


def test_fresh_run_refuses_to_clobber_existing_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    _seed_run_artifacts(output_dir)
    with pytest.raises(FileExistsError, match="--overwrite"):
        _prepare_output_dir(output_dir, overwrite=False)
    assert (output_dir / "results.csv").is_file()
    assert (output_dir / "weights" / "best.pt").is_file()


def test_overwrite_removes_existing_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    _seed_run_artifacts(output_dir)
    _prepare_output_dir(output_dir, overwrite=True)
    assert not (output_dir / "results.csv").exists()
    assert not (output_dir / "weights" / "best.pt").exists()


def test_prepare_output_dir_allows_empty_fresh_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "new-run"
    _prepare_output_dir(output_dir, overwrite=False)
    assert output_dir.is_dir()


def test_atomic_checkpoint_failure_preserves_prior_checkpoint(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "weights" / "last.pt"
    path.parent.mkdir()
    path.write_bytes(b"prior-valid-checkpoint")

    def fail_save(*args, **kwargs):
        raise OSError("simulated checkpoint write failure")

    monkeypatch.setattr(trainer.torch, "save", fail_save)
    with pytest.raises(OSError, match="simulated"):
        _atomic_torch_save({"epoch": 2}, path)
    assert path.read_bytes() == b"prior-valid-checkpoint"
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))


def test_atomic_checkpoint_replace_failure_preserves_prior_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "weights" / "last.pt"
    path.parent.mkdir()
    path.write_bytes(b"prior-valid-checkpoint")

    def fail_replace(*args, **kwargs):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(trainer.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace"):
        _atomic_torch_save({"epoch": 2}, path)
    assert path.read_bytes() == b"prior-valid-checkpoint"
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))


def test_atomic_checkpoint_fsync_failure_preserves_prior_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "weights" / "last.pt"
    path.parent.mkdir()
    path.write_bytes(b"prior-valid-checkpoint")

    def fail_fsync(descriptor):
        raise OSError("checkpoint fsync failure")

    monkeypatch.setattr(trainer.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="checkpoint fsync failure"):
        _atomic_torch_save({"epoch": 2}, path)
    assert path.read_bytes() == b"prior-valid-checkpoint"
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))


def test_atomic_checkpoint_cleanup_failure_preserves_fsync_error(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "weights" / "last.pt"
    path.parent.mkdir()
    path.write_bytes(b"prior-valid-checkpoint")

    def fail_fsync(descriptor):
        raise OSError("checkpoint fsync failure")

    def fail_unlink(self, missing_ok=False):
        raise OSError("checkpoint cleanup failure")

    monkeypatch.setattr(trainer.os, "fsync", fail_fsync)
    monkeypatch.setattr(Path, "unlink", fail_unlink)
    with pytest.raises(OSError, match="checkpoint fsync failure"):
        _atomic_torch_save({"epoch": 2}, path)
    assert path.read_bytes() == b"prior-valid-checkpoint"


def test_run_info_fsync_failure_preserves_prior_metadata(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "run_info.json"
    path.write_bytes(b"prior-run-info")

    def fail_fsync(descriptor):
        raise OSError("run-info fsync failure")

    monkeypatch.setattr(trainer.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="run-info fsync failure"):
        _atomic_write_json({"complete": True}, path)
    assert path.read_bytes() == b"prior-run-info"
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))


def test_run_info_cleanup_failure_preserves_fsync_error(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "run_info.json"
    path.write_bytes(b"prior-run-info")

    def fail_fsync(descriptor):
        raise OSError("run-info fsync failure")

    def fail_unlink(self, missing_ok=False):
        raise OSError("run-info cleanup failure")

    monkeypatch.setattr(trainer.os, "fsync", fail_fsync)
    monkeypatch.setattr(Path, "unlink", fail_unlink)
    with pytest.raises(OSError, match="run-info fsync failure"):
        _atomic_write_json({"complete": True}, path)
    assert path.read_bytes() == b"prior-run-info"

def test_resume_restores_python_and_torch_cpu_rng_state() -> None:
    random.seed(123)
    torch.manual_seed(123)
    state = _capture_rng_state()
    expected_python = random.random()
    expected_torch = torch.rand(4)

    random.random()
    torch.rand(4)
    _restore_rng_state({"rng_state": state})

    assert random.random() == expected_python
    assert torch.equal(torch.rand(4), expected_torch)


def test_resume_rejects_cuda_rng_device_count_mismatch(monkeypatch) -> None:
    state = _capture_rng_state()
    state["cuda"] = [torch.zeros(4, dtype=torch.uint8)]
    state["cuda_device_count"] = 1
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(trainer.torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="device count"):
        _restore_rng_state({"rng_state": state})

def _tiny_training_config(tmp_path: Path, output: Path, *, patience: int = 10) -> dict[str, object]:
    data_dir = tmp_path / "dataset"
    selected_files = []
    for split in ("train", "val"):
        for class_name in CLASS_NAMES:
            class_dir = data_dir / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            path = class_dir / "fixture.jpg"
            content = f"{split}/{class_name}".encode("utf-8")
            path.write_bytes(content)
            selected_files.append(
                {
                    "source": f"fixture/{split}/{class_name}.jpg",
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "split": split,
                    "destination": f"{split}/{class_name}/fixture.jpg",
                }
            )
    (data_dir / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_schema_version": 1,
                "class_values": list(CLASS_NAMES),
                "class_mapping": {name: index for index, name in enumerate(CLASS_NAMES)},
                "selected_files": selected_files,
                "counts": {
                    split: {name: 1 for name in CLASS_NAMES} for split in ("train", "val")
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "data_dir": str(data_dir),
        "output": str(output),
        "epochs": 3,
        "batch_size": 1,
        "lr": 0.01,
        "weight_decay": 0.0,
        "optimizer": "AdamW",
        "lr_schedule": "cosine",
        "warmup_epochs": 0,
        "patience": patience,
        "num_workers": 0,
        "imgsz": 8,
        "device": "cpu",
        "seed": 7,
        "_config_path": str(tmp_path / "config.yaml"),
    }


def test_cpu_runtime_identity_is_canonical_and_does_not_query_cuda(monkeypatch) -> None:
    def fail_cuda_query(*args, **kwargs):
        raise AssertionError("CPU runtime identity must not query CUDA")

    monkeypatch.setattr(trainer.torch.cuda, "device_count", fail_cuda_query)
    monkeypatch.setattr(trainer.torch.cuda, "get_device_name", fail_cuda_query)
    identity = _runtime_identity(torch.device("cpu"))

    assert identity["python"] == f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    assert identity["torch"] == str(torch.__version__)
    assert identity["torchvision"] == str(trainer.torchvision.__version__)
    assert identity["cuda_runtime"] is None
    assert identity["cudnn"] is None
    assert identity["cuda_device_count"] == 0
    assert identity["gpu_names"] == []


def test_cuda_runtime_identity_records_version_and_gpu_facts(monkeypatch) -> None:
    monkeypatch.setattr(trainer.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        trainer.torch.cuda,
        "get_device_name",
        lambda index: f"controlled-gpu-{index}",
    )
    monkeypatch.setattr(trainer.torch.backends.cudnn, "version", lambda: 9010)
    identity = _runtime_identity(torch.device("cuda:0"))

    assert identity["cuda_runtime"] == torch.version.cuda
    assert identity["cudnn"] == 9010
    assert identity["cuda_device_count"] == 2
    assert identity["gpu_names"] == ["controlled-gpu-0", "controlled-gpu-1"]

@pytest.mark.parametrize("mutation", ["added", "missing", "escaping"])
def test_dataset_provenance_rejects_inconsistent_live_membership(
    tmp_path: Path, mutation: str
) -> None:
    config = _tiny_training_config(tmp_path, tmp_path / "out")
    data_dir = Path(config["data_dir"])
    manifest_path = data_dir / "manifest.json"
    if mutation == "added":
        (data_dir / "train" / CLASS_NAMES[0] / "added.jpg").write_bytes(b"added")
    elif mutation == "missing":
        (data_dir / "train" / CLASS_NAMES[0] / "fixture.jpg").unlink()
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["selected_files"][0]["destination"] = "../escape.jpg"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="membership|unsafe path"):
        _build_provenance(
            config,
            data_dir=data_dir,
            output_dir=Path(config["output"]),
            device=torch.device("cpu"),
        )

def test_set_seed_requires_deterministic_torch_algorithms(monkeypatch) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(trainer.torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(
        trainer.torch, "use_deterministic_algorithms", lambda enabled: calls.append(enabled)
    )

    set_seed(7)

    assert calls == [True]
    assert trainer.torch.backends.cudnn.deterministic is True
    assert trainer.torch.backends.cudnn.benchmark is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_deterministic_matrix_operation_subprocess() -> None:
    script = """
import os
import scripts.train_bcs_ordinal
import torch

assert os.environ["CUBLAS_WORKSPACE_CONFIG"] in {":4096:8", ":16:8"}
left = torch.randn((32, 32), device="cuda")
right = torch.randn((32, 32), device="cuda")
left @ right
torch.cuda.synchronize()
print("cuda-matmul-ok")
"""
    environment = os.environ.copy()
    environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "cuda-matmul-ok" in completed.stdout


@pytest.mark.parametrize("workspace", [":4096:8", ":16:8"])
def test_cublas_workspace_config_preserves_accepted_inherited_value(workspace: str) -> None:
    script = """
import os
import scripts.train_bcs_ordinal
assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == %r
print("cublas-value-ok")
""" % workspace
    environment = os.environ.copy()
    environment["CUBLAS_WORKSPACE_CONFIG"] = workspace
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "cublas-value-ok" in completed.stdout


def test_cublas_workspace_config_rejects_invalid_inherited_value() -> None:
    environment = os.environ.copy()
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":invalid"
    completed = subprocess.run(
        [sys.executable, "-c", "import scripts.train_bcs_ordinal"],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "Invalid CUBLAS_WORKSPACE_CONFIG" in completed.stderr
