from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from scripts.train_bcs_ordinal import (  # noqa: E402
    RESULTS_FIELDNAMES,
    RESUMABLE_CHECKPOINT_FIELDS,
    _build_last_checkpoint,
    _build_provenance,
    _runtime_identity,
    _capture_rng_state,
    _atomic_write_json,
    _atomic_torch_save,
    _load_resume_checkpoint,
    _open_results_csv,
    _prepare_output_dir,
    _reconcile_results_csv,
    _restore_model_state,
    _restore_optimizer_state,
    _restore_rng_state,
    set_seed,
    load_config,
)
from scripts.train_bcs_ordinal import main as train_main  # noqa: E402
import scripts.train_bcs_ordinal as trainer  # noqa: E402
from vacca_bcs.constants import (  # noqa: E402
    BCS_CLASS_SCORES,
    BCS_DOMAIN_ID,
    CLASS_NAMES,
    NUM_CLASSES,
    NUM_THRESHOLDS,
    SCORE_BASE,
    SCORE_MAX,
    SCORE_MIN,
    SCORE_STEP,
)
from vacca_bcs.dataset import BCSFolderDataset, Letterbox, build_transforms  # noqa: E402
from vacca_bcs.model import (  # noqa: E402
    BCSOrdinalModel,
    CORALHead,
    coral_loss,
    encode_levels,
    predict,
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


def test_integer_domain_constants_reject_fractional_class_names() -> None:
    assert BCS_DOMAIN_ID == "bcs-integer-1-5"
    assert BCS_CLASS_SCORES == (1, 2, 3, 4, 5)
    assert CLASS_NAMES == ("1", "2", "3", "4", "5")
    assert (SCORE_MIN, SCORE_MAX, SCORE_BASE, SCORE_STEP) == (1, 5, 1, 1)
    assert (NUM_CLASSES, NUM_THRESHOLDS) == (5, 4)
    assert not {"3.25", "3.5", "3.75", "4.0", "4.25"}.intersection(CLASS_NAMES)


def test_dataset_rejects_fractional_class_folders(tmp_path: Path) -> None:
    for class_name in ("3.25", "3.5", "3.75", "4.0", "4.25"):
        (tmp_path / class_name).mkdir()
    with pytest.raises(FileNotFoundError, match="Class directory"):
        BCSFolderDataset(tmp_path)


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
    assert torch.equal(scores, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))


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
    assert torch.all((scores >= 1.0) & (scores <= 5.0))


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
    assert torch.equal(scores, torch.full((2,), 1.0))

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
        "data_dir": "data/bcs-integer-v1",
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


def test_main_fails_clearly_when_output_exists(tmp_path: Path, capsys) -> None:
    output_dir = tmp_path / "run"
    _seed_run_artifacts(output_dir)
    config_path = _write_config(tmp_path, output=str(output_dir))
    exit_code = train_main(["--config", str(config_path)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "--overwrite" in captured.err
    assert "--resume" in captured.err


def test_last_checkpoint_roundtrip_restores_training_state(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    checkpoint = _build_last_checkpoint(
        model,
        optimizer,
        epoch=2,
        best_mae=0.5,
        epochs_without_improvement=1,
        config={"epochs": 4},
    )
    assert RESUMABLE_CHECKPOINT_FIELDS.issubset(checkpoint)

    path = tmp_path / "last.pt"
    torch.save(checkpoint, path)
    loaded = _load_resume_checkpoint(
        path, expected_classes=list(CLASS_NAMES), total_epochs=4
    )
    assert loaded["epoch"] == 2
    assert loaded["best_mae"] == 0.5
    assert loaded["epochs_without_improvement"] == 1

    new_model = torch.nn.Linear(3, 2)
    _restore_model_state(new_model, loaded)
    for name, value in new_model.state_dict().items():
        assert torch.equal(value, checkpoint["model_state_dict"][name])

    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=0.99)
    _restore_optimizer_state(new_optimizer, loaded)
    assert new_optimizer.param_groups[0]["lr"] == 0.01


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


def test_resume_rejects_malformed_cuda_rng_entry(tmp_path: Path, monkeypatch) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    checkpoint = _build_last_checkpoint(
        model,
        optimizer,
        epoch=1,
        best_mae=0.5,
        epochs_without_improvement=0,
        config={"epochs": 4},
    )
    checkpoint["rng_state"]["cuda"] = ["not-a-tensor"]
    checkpoint["rng_state"]["cuda_device_count"] = 1
    path = tmp_path / "last.pt"
    torch.save(checkpoint, path)
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(trainer.torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="CUDA RNG entry 0"):
        _load_resume_checkpoint(path, expected_classes=list(CLASS_NAMES), total_epochs=4)


@pytest.mark.parametrize(
    ("cuda_state", "cuda_count"),
    [(None, 1), ([torch.zeros(4, dtype=torch.uint8)], 2)],
)
def test_resume_rejects_inconsistent_cuda_rng_metadata(
    tmp_path: Path, cuda_state, cuda_count: int
) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    checkpoint = _build_last_checkpoint(
        model,
        optimizer,
        epoch=1,
        best_mae=0.5,
        epochs_without_improvement=0,
        config={"epochs": 4},
    )
    checkpoint["rng_state"]["cuda"] = cuda_state
    checkpoint["rng_state"]["cuda_device_count"] = cuda_count
    path = tmp_path / "last.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="CUDA RNG state"):
        _load_resume_checkpoint(path, expected_classes=list(CLASS_NAMES), total_epochs=4)


def test_resume_accepts_mocked_valid_cuda_rng_metadata(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    checkpoint = _build_last_checkpoint(
        model,
        optimizer,
        epoch=1,
        best_mae=0.5,
        epochs_without_improvement=0,
        config={"epochs": 4},
    )
    checkpoint["rng_state"]["cuda"] = [torch.zeros(4, dtype=torch.uint8)]
    checkpoint["rng_state"]["cuda_device_count"] = 1
    path = tmp_path / "last.pt"
    torch.save(checkpoint, path)
    loaded = _load_resume_checkpoint(path, expected_classes=list(CLASS_NAMES), total_epochs=4)
    assert loaded["rng_state"]["cuda_device_count"] == 1


def test_resume_accepts_valid_cpu_only_rng_metadata(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    checkpoint = _build_last_checkpoint(
        model,
        optimizer,
        epoch=1,
        best_mae=0.5,
        epochs_without_improvement=0,
        config={"epochs": 4},
    )
    checkpoint["rng_state"]["cuda"] = None
    checkpoint["rng_state"]["cuda_device_count"] = 0
    path = tmp_path / "last.pt"
    torch.save(checkpoint, path)
    loaded = _load_resume_checkpoint(path, expected_classes=list(CLASS_NAMES), total_epochs=4)
    assert loaded["rng_state"]["cuda"] is None


def test_resume_rejects_best_only_checkpoint(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)
    path = tmp_path / "best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"epochs": 4},
            "classes": list(CLASS_NAMES),
            "epoch": 1,
            "val_mae": 0.5,
        },
        path,
    )
    with pytest.raises(ValueError, match="not resumable"):
        _load_resume_checkpoint(path, expected_classes=list(CLASS_NAMES), total_epochs=4)


def test_resume_rejects_mismatched_classes(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    checkpoint = _build_last_checkpoint(
        model, optimizer, epoch=1, best_mae=0.5,
        epochs_without_improvement=0, config={"epochs": 4},
    )
    checkpoint["classes"] = ["a", "b"]
    path = tmp_path / "last.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="classes"):
        _load_resume_checkpoint(path, expected_classes=list(CLASS_NAMES), total_epochs=4)


def test_resume_rejects_already_completed_total_epochs(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    checkpoint = _build_last_checkpoint(
        model, optimizer, epoch=5, best_mae=0.5,
        epochs_without_improvement=0, config={"epochs": 4},
    )
    path = tmp_path / "last.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="exceeds"):
        _load_resume_checkpoint(path, expected_classes=list(CLASS_NAMES), total_epochs=4)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("best_mae", float("nan"), "best_mae"),
        ("epochs_without_improvement", -1, "non-negative"),
        ("classes", "invalid", "classes must be a list"),
        ("epoch", True, "epoch must be"),
    ],
)
def test_resume_rejects_malformed_checkpoint_metadata(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    checkpoint = _build_last_checkpoint(
        model, optimizer, epoch=1, best_mae=0.5,
        epochs_without_improvement=0, config={"epochs": 4},
    )
    checkpoint[field] = value
    path = tmp_path / "last.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match=message):
        _load_resume_checkpoint(path, expected_classes=list(CLASS_NAMES), total_epochs=4)


def test_restore_model_state_rejects_shape_mismatch(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    checkpoint = _build_last_checkpoint(
        model, optimizer, epoch=1, best_mae=0.5,
        epochs_without_improvement=0, config={"epochs": 4},
    )
    incompatible = torch.nn.Linear(5, 2)
    with pytest.raises(ValueError, match="does not match"):
        _restore_model_state(incompatible, checkpoint)


def _results_row(epoch: int) -> dict[str, str]:
    return {
        "epoch": str(epoch),
        "lr": "0.001",
        "train_loss": "1.0",
        "val_exact_acc": "0.5",
        "val_pm1_acc": "0.9",
        "val_mae": "0.25",
    }


def test_results_csv_appends_without_duplicate_header(tmp_path: Path) -> None:
    results_path = tmp_path / "results.csv"
    handle, writer = _open_results_csv(results_path, append=False)
    writer.writerow(_results_row(1))
    handle.close()

    handle, writer = _open_results_csv(results_path, append=True)
    writer.writerow(_results_row(2))
    handle.close()

    with results_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == RESULTS_FIELDNAMES
    assert [row[0] for row in rows[1:]] == ["1", "2"]


def test_results_csv_append_on_missing_file_writes_header(tmp_path: Path) -> None:
    results_path = tmp_path / "results.csv"
    handle, writer = _open_results_csv(results_path, append=True)
    writer.writerow(_results_row(1))
    handle.close()

    with results_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == RESULTS_FIELDNAMES
    assert rows[1][0] == "1"


def _write_results_history(path: Path, epochs: list[int]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(RESULTS_FIELDNAMES)
        for epoch in epochs:
            writer.writerow([epoch, "0.001", "1.0", "0.5", "0.9", "0.25"])


def _read_epoch_sequence(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == RESULTS_FIELDNAMES
    return [row[0] for row in rows[1:]]


def test_reconcile_truncates_one_row_ahead_history(tmp_path: Path) -> None:
    # Interruption after the CSV flush but before the last.pt save.
    results_path = tmp_path / "results.csv"
    _write_results_history(results_path, [1, 2, 3])
    kept = _reconcile_results_csv(results_path, checkpoint_epoch=2)
    assert kept == 2
    assert _read_epoch_sequence(results_path) == ["1", "2"]
    assert not (tmp_path / "results.csv.tmp").exists()


def test_reconcile_discards_partial_one_row_crash_suffix(tmp_path: Path) -> None:
    results_path = tmp_path / "results.csv"
    results_path.write_text(
        ",".join(RESULTS_FIELDNAMES)
        + "\n1,0.001,1.0,0.5,0.9,0.25\n2,0.001,",
        encoding="utf-8",
    )
    assert _reconcile_results_csv(results_path, checkpoint_epoch=1) == 1
    assert _read_epoch_sequence(results_path) == ["1"]
    assert not list(tmp_path.glob(".results.csv.*.tmp"))


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("lr", "-0.001"),
        ("train_loss", "nan"),
        ("val_exact_acc", "1.1"),
        ("val_pm1_acc", "-0.1"),
        ("val_mae", "4.1"),
    ],
)
def test_reconcile_rejects_invalid_complete_expected_suffix(
    tmp_path: Path, column: str, value: str
) -> None:
    row = _results_row(2)
    row[column] = value
    results_path = tmp_path / "results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_FIELDNAMES)
        writer.writeheader()
        writer.writerow(_results_row(1))
        writer.writerow(row)
    with pytest.raises(ValueError, match=column):
        _reconcile_results_csv(results_path, checkpoint_epoch=1)


@pytest.mark.parametrize("torn_suffix", ["\n", "2,", "12"])
def test_reconcile_discards_torn_suffix_without_complete_epoch(
    tmp_path: Path, torn_suffix: str
) -> None:
    results_path = tmp_path / "results.csv"
    results_path.write_text(
        ",".join(RESULTS_FIELDNAMES)
        + "\n1,0.001,1.0,0.5,0.9,0.25\n"
        + torn_suffix,
        encoding="utf-8",
    )
    assert _reconcile_results_csv(results_path, checkpoint_epoch=1) == 1
    assert _read_epoch_sequence(results_path) == ["1"]


def test_reconcile_rejects_complete_wrong_epoch_suffix(tmp_path: Path) -> None:
    results_path = tmp_path / "results.csv"
    _write_results_history(results_path, [1, 12])
    with pytest.raises(ValueError, match="unrecoverable suffix"):
        _reconcile_results_csv(results_path, checkpoint_epoch=1)


def test_reconcile_rejects_complete_non_integer_epoch_suffix(tmp_path: Path) -> None:
    results_path = tmp_path / "results.csv"
    results_path.write_text(
        ",".join(RESULTS_FIELDNAMES)
        + "\n1,0.001,1.0,0.5,0.9,0.25\nabc,0.001,1.0,0.5,0.9,0.25\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unrecoverable complete suffix"):
        _reconcile_results_csv(results_path, checkpoint_epoch=1)


def test_reconcile_rejects_overfull_suffix(tmp_path: Path) -> None:
    results_path = tmp_path / "results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(RESULTS_FIELDNAMES)
        writer.writerow(list(_results_row(1).values()))
        writer.writerow([*list(_results_row(2).values()), "extra"])
    with pytest.raises(ValueError, match="malformed"):
        _reconcile_results_csv(results_path, checkpoint_epoch=1)


def test_reconcile_rejects_two_rows_beyond_checkpoint(tmp_path: Path) -> None:
    results_path = tmp_path / "results.csv"
    _write_results_history(results_path, [1, 2, 3])
    with pytest.raises(ValueError, match="only one crash-window row"):
        _reconcile_results_csv(results_path, checkpoint_epoch=1)


def test_reconcile_replace_failure_preserves_prior_csv(tmp_path: Path, monkeypatch) -> None:
    results_path = tmp_path / "results.csv"
    _write_results_history(results_path, [1, 2])
    before = results_path.read_bytes()

    def fail_replace(*args, **kwargs):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(trainer.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace"):
        _reconcile_results_csv(results_path, checkpoint_epoch=1)
    assert results_path.read_bytes() == before
    assert not list(tmp_path.glob(".results.csv.*.tmp"))


def test_reconcile_fsync_failure_preserves_prior_csv(tmp_path: Path, monkeypatch) -> None:
    results_path = tmp_path / "results.csv"
    _write_results_history(results_path, [1, 2])
    before = results_path.read_bytes()

    def fail_fsync(descriptor):
        raise OSError("csv fsync failure")

    monkeypatch.setattr(trainer.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="csv fsync failure"):
        _reconcile_results_csv(results_path, checkpoint_epoch=1)
    assert results_path.read_bytes() == before
    assert not list(tmp_path.glob(".results.csv.*.tmp"))


def test_reconcile_cleanup_failure_preserves_fsync_error(
    tmp_path: Path, monkeypatch
) -> None:
    results_path = tmp_path / "results.csv"
    _write_results_history(results_path, [1, 2])
    before = results_path.read_bytes()

    def fail_fsync(descriptor):
        raise OSError("csv fsync failure")

    def fail_unlink(self, missing_ok=False):
        raise OSError("csv cleanup failure")

    monkeypatch.setattr(trainer.os, "fsync", fail_fsync)
    monkeypatch.setattr(Path, "unlink", fail_unlink)
    with pytest.raises(OSError, match="csv fsync failure"):
        _reconcile_results_csv(results_path, checkpoint_epoch=1)
    assert results_path.read_bytes() == before


def test_reconcile_accepts_exact_history_without_rewrite(tmp_path: Path) -> None:
    results_path = tmp_path / "results.csv"
    _write_results_history(results_path, [1, 2, 3])
    before = results_path.read_bytes()
    kept = _reconcile_results_csv(results_path, checkpoint_epoch=3)
    assert kept == 3
    assert results_path.read_bytes() == before


def test_reconcile_rejects_missing_results_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="results history"):
        _reconcile_results_csv(tmp_path / "results.csv", checkpoint_epoch=2)


def test_reconcile_rejects_empty_results_file(tmp_path: Path) -> None:
    results_path = tmp_path / "results.csv"
    results_path.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        _reconcile_results_csv(results_path, checkpoint_epoch=1)


def test_reconcile_rejects_wrong_schema(tmp_path: Path) -> None:
    results_path = tmp_path / "results.csv"
    results_path.write_text("epoch,loss\n1,1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        _reconcile_results_csv(results_path, checkpoint_epoch=1)


def test_reconcile_rejects_malformed_row(tmp_path: Path) -> None:
    results_path = tmp_path / "results.csv"
    results_path.write_text(
        ",".join(RESULTS_FIELDNAMES) + "\n1,0.001,1.0,0.5,0.9\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="malformed"):
        _reconcile_results_csv(results_path, checkpoint_epoch=1)


def test_reconcile_rejects_non_integer_epoch(tmp_path: Path) -> None:
    results_path = tmp_path / "results.csv"
    results_path.write_text(
        ",".join(RESULTS_FIELDNAMES) + "\nabc,0.001,1.0,0.5,0.9,0.25\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-integer"):
        _reconcile_results_csv(results_path, checkpoint_epoch=1)


def test_reconcile_rejects_gapped_history(tmp_path: Path) -> None:
    results_path = tmp_path / "results.csv"
    _write_results_history(results_path, [1, 2, 4])
    with pytest.raises(ValueError, match="contiguous"):
        _reconcile_results_csv(results_path, checkpoint_epoch=4)


def test_reconcile_rejects_duplicate_epoch(tmp_path: Path) -> None:
    results_path = tmp_path / "results.csv"
    _write_results_history(results_path, [1, 2, 2])
    with pytest.raises(ValueError, match="unrecoverable suffix"):
        _reconcile_results_csv(results_path, checkpoint_epoch=2)


def test_reconcile_rejects_history_ending_before_checkpoint(tmp_path: Path) -> None:
    results_path = tmp_path / "results.csv"
    _write_results_history(results_path, [1, 2])
    with pytest.raises(ValueError, match="ends at epoch 2"):
        _reconcile_results_csv(results_path, checkpoint_epoch=3)


@pytest.mark.parametrize(
    ("column", "value"),
    [("train_loss", "nan"), ("val_exact_acc", "1.1"), ("val_mae", "-0.01")],
)
def test_reconcile_rejects_invalid_or_impossible_metrics(
    tmp_path: Path, column: str, value: str
) -> None:
    row = _results_row(1)
    row[column] = value
    results_path = tmp_path / "results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_FIELDNAMES)
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError, match=column):
        _reconcile_results_csv(results_path, checkpoint_epoch=1)


def test_validate_computes_mae_with_canonical_integer_score_step() -> None:
    import scripts.train_bcs_ordinal as trainer

    class _FixedLogitsModel(torch.nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[-8.0, -8.0, -8.0, -8.0], [8.0, 8.0, 8.0, 8.0]])

    # Two samples, both off by 4 class indices: canonical integer MAE is 4.
    loader = [(torch.zeros(2, 3, 8, 8), torch.tensor([4, 0]))]
    metrics = trainer._validate(_FixedLogitsModel(), loader, torch.device("cpu"))
    assert metrics["mae"] == pytest.approx(4.0)


def _tiny_training_config(tmp_path: Path, output: Path, *, patience: int = 10) -> dict[str, object]:
    data_dir = tmp_path / "dataset"
    records = []
    for split in ("train", "val"):
        for class_index, class_name in enumerate(CLASS_NAMES):
            class_dir = data_dir / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            evidence_id = (0 if split == "train" else NUM_CLASSES) + class_index + 1
            path = class_dir / f"{evidence_id}.jpg"
            content = f"{split}/{class_name}".encode("utf-8")
            path.write_bytes(content)
            records.append(
                {
                    "split": split,
                    "bcs_score": class_index + 1,
                    "evidence_id": evidence_id,
                    "relative_path": f"{split}/{class_name}/{evidence_id}.jpg",
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "provenance": [
                        {"evidence_id": evidence_id, "evaluation_id": evidence_id + 100}
                    ],
                }
            )
    records.sort(key=lambda item: (item["bcs_score"], item["split"] == "val", item["evidence_id"]))
    counts = {split: [1] * NUM_CLASSES for split in ("train", "val")}
    (data_dir / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_schema_version": "bcs-integer-snapshot-v2",
                "domain_id": BCS_DOMAIN_ID,
                "class_values": list(BCS_CLASS_SCORES),
                "class_mapping": {name: index for index, name in enumerate(CLASS_NAMES)},
                "score_min": SCORE_MIN,
                "score_max": SCORE_MAX,
                "score_base": SCORE_BASE,
                "score_step": SCORE_STEP,
                "num_classes": NUM_CLASSES,
                "num_thresholds": NUM_THRESHOLDS,
                "source_schema": "bcs-source-v1",
                "counts": counts,
                "split_plan": {
                    "identity_digest": "0" * 64,
                    "seed": 7,
                    "canonical_val_ratio": "0",
                    "candidate_evidence_ids": list(range(1, 11)),
                    "excluded_evidence_ids": [],
                    "counts": counts,
                },
                "records": records,
                "exclusions": [],
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


def test_resume_roundtrip_rejects_runtime_identity_drift(tmp_path: Path) -> None:
    config = _tiny_training_config(tmp_path, tmp_path / "run")
    provenance = _build_provenance(
        config,
        data_dir=Path(config["data_dir"]),
        output_dir=Path(config["output"]),
        device=torch.device("cpu"),
    )
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    checkpoint = _build_last_checkpoint(
        model,
        optimizer,
        epoch=1,
        best_mae=0.5,
        epochs_without_improvement=0,
        config=config,
        provenance=provenance,
    )
    path = tmp_path / "last.pt"
    torch.save(checkpoint, path)

    loaded = _load_resume_checkpoint(
        path,
        expected_classes=list(CLASS_NAMES),
        total_epochs=3,
        expected_provenance=provenance,
    )
    assert loaded["provenance"]["runtime"] == provenance["runtime"]

    drifted = json.loads(json.dumps(provenance))
    drifted["runtime"]["torch"] = "different-torch-runtime"
    with pytest.raises(ValueError, match="provenance mismatch for runtime"):
        _load_resume_checkpoint(
            path,
            expected_classes=list(CLASS_NAMES),
            total_epochs=3,
            expected_provenance=drifted,
        )


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
        (data_dir / "train" / CLASS_NAMES[0] / "1.jpg").unlink()
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["records"][0]["relative_path"] = "../escape.jpg"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest|membership|unsafe path"):
        _build_provenance(
            config,
            data_dir=data_dir,
            output_dir=Path(config["output"]),
            device=torch.device("cpu"),
        )


def test_integer_v2_manifest_is_accepted_and_old_lineages_are_rejected(tmp_path: Path) -> None:
    config = _tiny_training_config(tmp_path, tmp_path / "out")
    manifest_path = Path(config["data_dir"]) / "manifest.json"
    assert _build_provenance(
        config, data_dir=Path(config["data_dir"]), output_dir=Path(config["output"]), device=torch.device("cpu")
    )["classes"] == list(CLASS_NAMES)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manifest_schema_version"] = "bcs-integer-snapshot-v1"
    manifest["records"][0]["storage_key"] = "private-storage-key"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="legacy|schema") as failure:
        _build_provenance(
            config, data_dir=Path(config["data_dir"]), output_dir=Path(config["output"]), device=torch.device("cpu")
        )
    assert "private-storage-key" not in str(failure.value)
    manifest["manifest_schema_version"] = "bcs-integer-snapshot-v2"
    manifest["class_values"] = [3.25, 3.5, 3.75, 4.0, 4.25]
    manifest["records"][0].pop("storage_key")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="legacy|schema"):
        _build_provenance(
            config, data_dir=Path(config["data_dir"]), output_dir=Path(config["output"]), device=torch.device("cpu")
        )


@pytest.mark.parametrize("mutation", [
    lambda manifest: manifest["records"][0].update(relative_path="val/1/1.jpg"),
    lambda manifest: manifest["records"][0].update(bcs_score=5),
    lambda manifest: manifest.update(class_mapping={name: 0 for name in CLASS_NAMES}),
])
def test_integer_v2_manifest_rejects_record_and_class_tampering(tmp_path: Path, mutation) -> None:
    config = _tiny_training_config(tmp_path, tmp_path / "out")
    path = Path(config["data_dir"]) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutation(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        _build_provenance(config, data_dir=Path(config["data_dir"]), output_dir=Path(config["output"]), device=torch.device("cpu"))


def test_default_integer_training_roots_are_canonical(tmp_path: Path) -> None:
    config = yaml.safe_load((REPO_ROOT / "configs" / "training_bcs_ordinal.yaml").read_text())
    assert config["data_dir"] == "data/bcs-integer-v1"
    assert config["output"] == "outputs/bcs-ordinal-integer-v1"


@pytest.mark.parametrize("mutation", ["missing", "unexpected"])
def test_integer_snapshot_rejects_root_structure_tampering(tmp_path: Path, mutation: str) -> None:
    config = _tiny_training_config(tmp_path, tmp_path / "out")
    root = Path(config["data_dir"])
    if mutation == "missing":
        (root / "val" / CLASS_NAMES[-1] / "10.jpg").unlink()
        (root / "val" / CLASS_NAMES[-1]).rmdir()
    else:
        (root / "unexpected").mkdir()
    with pytest.raises(ValueError, match="structure|membership"):
        _build_provenance(config, data_dir=root, output_dir=Path(config["output"]), device=torch.device("cpu"))


def _seed_prior_training_artifacts(output: Path) -> dict[Path, bytes]:
    artifacts = {
        output / "results.csv": b"prior-results",
        output / "run_info.json": b"prior-run-info",
        output / "weights" / "best.pt": b"prior-best",
        output / "weights" / "last.pt": b"prior-last",
    }
    for path, content in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return artifacts


def _assert_artifacts_unchanged(artifacts: dict[Path, bytes]) -> None:
    assert {path: path.read_bytes() for path in artifacts} == artifacts


def test_overwrite_invalid_manifest_preserves_prior_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "run"
    config = _tiny_training_config(tmp_path, output)
    artifacts = _seed_prior_training_artifacts(output)
    manifest_path = Path(config["data_dir"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest|membership"):
        trainer.train(config, overwrite=True)
    _assert_artifacts_unchanged(artifacts)


def test_overwrite_unsupported_optimizer_preserves_prior_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "run"
    config = _tiny_training_config(tmp_path, output)
    config["optimizer"] = "SGD"
    artifacts = _seed_prior_training_artifacts(output)
    _install_tiny_training_fakes(monkeypatch, [])

    with pytest.raises(ValueError, match="Unsupported optimizer"):
        trainer.train(config, overwrite=True)
    _assert_artifacts_unchanged(artifacts)


class _TinyDataset:
    class_counts = {name: 1 for name in CLASS_NAMES}

    def __init__(self, root, **kwargs) -> None:
        self.root = root

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        return torch.zeros(1), 0


class _TinyModel(torch.nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


def _install_tiny_training_fakes(monkeypatch, calls: list[float], interrupt_at: int | None = None) -> None:
    monkeypatch.setattr(trainer, "BCSFolderDataset", _TinyDataset)
    monkeypatch.setattr(trainer, "BCSOrdinalModel", _TinyModel)
    call_count = 0

    def fake_train(model, loader, optimizer, device) -> float:
        nonlocal call_count
        call_count += 1
        if interrupt_at == call_count:
            raise KeyboardInterrupt()
        target = random.random() + float(torch.rand(1).item())
        target_tensor = torch.tensor(target)
        optimizer.zero_grad(set_to_none=True)
        (model.weight - target_tensor).pow(2).sum().backward()
        optimizer.step()
        calls.append(target)
        return float((model.weight.detach() - target_tensor).abs().item())

    def fake_validate(model, loader, device) -> dict[str, object]:
        return {
            "exact_acc": 0.5,
            "pm1_acc": 1.0,
            "mae": 0.25,
            "recall": {name: 0.0 for name in CLASS_NAMES},
            "total": 1,
        }

    monkeypatch.setattr(trainer, "_train_epoch", fake_train)
    monkeypatch.setattr(trainer, "_validate", fake_validate)


def test_public_resume_rejects_changed_dataset_manifest(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "run"
    config = _tiny_training_config(tmp_path, output)
    _install_tiny_training_fakes(monkeypatch, [], interrupt_at=2)
    with pytest.raises(KeyboardInterrupt):
        trainer.train(config, overwrite=True)
    config["lr"] = 0.02
    with pytest.raises(ValueError, match="provenance mismatch"):
        trainer.train(config, resume=output / "weights" / "last.pt")
    config["lr"] = 0.01
    live_file = Path(config["data_dir"]) / "train" / CLASS_NAMES[0] / "1.jpg"
    live_file.write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest|hash mismatch"):
        trainer.train(config, resume=output / "weights" / "last.pt")


def test_interrupted_and_resumed_workflow_matches_uninterrupted_run(
    tmp_path: Path, monkeypatch
) -> None:
    interrupted_output = tmp_path / "interrupted"
    interrupted_config = _tiny_training_config(tmp_path, interrupted_output)
    resumed_calls: list[float] = []
    _install_tiny_training_fakes(monkeypatch, resumed_calls, interrupt_at=2)
    with pytest.raises(KeyboardInterrupt):
        trainer.train(interrupted_config, overwrite=True)

    baseline_output = tmp_path / "baseline"
    baseline_config = _tiny_training_config(tmp_path, baseline_output)
    baseline_calls: list[float] = []
    _install_tiny_training_fakes(monkeypatch, baseline_calls)
    trainer.train(baseline_config, overwrite=True)
    assert "provenance" in torch.load(
        baseline_output / "weights" / "best.pt", map_location="cpu", weights_only=True
    )
    run_info = json.loads((baseline_output / "run_info.json").read_text())
    assert run_info["domain_id"] == BCS_DOMAIN_ID
    assert run_info["class_values"] == list(BCS_CLASS_SCORES)

    resumed_calls.clear()
    _install_tiny_training_fakes(monkeypatch, resumed_calls)
    trainer.train(
        interrupted_config,
        resume=interrupted_output / "weights" / "last.pt",
    )

    assert resumed_calls == pytest.approx(baseline_calls[1:])
    assert (interrupted_output / "results.csv").read_bytes() == (
        baseline_output / "results.csv"
    ).read_bytes()


def test_terminal_checkpoint_resume_finalizes_run_info_without_overwrite(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "run"
    config = _tiny_training_config(tmp_path, output)
    config["epochs"] = 1
    _install_tiny_training_fakes(monkeypatch, [])
    real_atomic_save = trainer._atomic_torch_save

    def interrupt_after_terminal_save(payload, path):
        real_atomic_save(payload, path)
        if path.name == "last.pt":
            raise KeyboardInterrupt()

    monkeypatch.setattr(trainer, "_atomic_torch_save", interrupt_after_terminal_save)
    with pytest.raises(KeyboardInterrupt):
        trainer.train(config, overwrite=True)
    assert not (output / "run_info.json").exists()

    result = trainer.train(config, resume=output / "weights" / "last.pt")

    assert result["run_info"]["finalized_from_terminal_checkpoint"] is True
    assert (output / "run_info.json").is_file()


def test_resume_stops_at_restored_patience_boundary(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "run"
    config = _tiny_training_config(tmp_path, output, patience=1)
    calls: list[float] = []
    _install_tiny_training_fakes(monkeypatch, calls)
    data_dir = Path(config["data_dir"])
    device = torch.device("cpu")
    provenance = _build_provenance(config, data_dir=data_dir, output_dir=output, device=device)
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    checkpoint = _build_last_checkpoint(
        model,
        optimizer,
        epoch=1,
        best_mae=0.25,
        epochs_without_improvement=1,
        config=config,
        provenance=provenance,
    )
    _atomic_torch_save(checkpoint, output / "weights" / "last.pt")
    _write_results_history(output / "results.csv", [1])

    result = trainer.train(config, resume=output / "weights" / "last.pt")

    assert calls == []
    assert result["final_metrics"] == {}


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
