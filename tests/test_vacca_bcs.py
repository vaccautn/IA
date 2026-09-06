from __future__ import annotations

import csv
import hashlib
import io
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
    CHECKPOINT_SCHEMA_VERSION,
    RESULTS_FIELDNAMES,
    RESULTS_LINEAGE_FILENAME,
    RESUMABLE_CHECKPOINT_FIELDS,
    _build_last_checkpoint,
    _build_provenance,
    _checkpoint_lineage,
    _coverage_from_manifest,
    _validate_class_coverage,
    _runtime_identity,
    _capture_rng_state,
    _atomic_write_json,
    _atomic_torch_save,
    _load_resume_checkpoint as _load_resume_checkpoint_impl,
    _open_results_csv,
    _prepare_output_dir,
    _reconcile_results_csv,
    _restore_model_state,
    _restore_optimizer_state,
    _restore_rng_state,
    set_seed,
    load_config,
    TORCH_SEED_MAX,
)
from scripts.train_bcs_ordinal import main as train_main  # noqa: E402
import scripts.train_bcs_ordinal as trainer  # noqa: E402
from scripts.run_bcs_overnight import _validate_best_checkpoint as _validate_best_checkpoint_impl  # noqa: E402
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


def _load_resume_checkpoint(path, **kwargs):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    generations = path.parent / "generations"
    generations.mkdir(exist_ok=True)
    (generations / f"{digest}.pt").write_bytes(raw)
    for alias in (path.parent / "best.pt", path.parent / "last.pt"):
        if not alias.exists():
            alias.write_bytes(raw)
    descriptor_path = path.parent / "checkpoint_set.json"
    if not descriptor_path.exists():
        descriptor_path.write_text(
            json.dumps(
                {
                    "schema": "vacca-bcs-checkpoint-set-v1",
                    "lineage_schema_version": "bcs-category-coral-results-v1",
                    "committed_epoch": 1,
                    "best": {"filename": f"generations/{digest}.pt", "sha256": digest},
                    "last": {"filename": f"generations/{digest}.pt", "sha256": digest},
                    "run_id": "0" * 32,
                    "domain_id": BCS_DOMAIN_ID,
                    "source_schema": "bcs-local-category-source-v1",
                    "snapshot_schema": "bcs-category-snapshot-v1",
                    "snapshot_identity": "0" * 64,
                    "dataset_manifest_digest": "0" * 64,
                    "config_sha256": "0" * 64,
                    "observed_classes": [1, 2, 3, 4, 5],
                    "missing_classes": [],
                    "source_identity_scheme": "local-path-sha256-v1",
                    "source_mapping": {"3.25": 1, "3.5": 2, "3.75": 3, "4.0": 4, "4.25": 5},
                    "best_epoch": 1,
                    "selection_identity": "0" * 64,
                    "best_validation": {},
                }
            ),
            encoding="utf-8",
        )
    kwargs.setdefault("require_checkpoint_set", True)
    return _load_resume_checkpoint_impl(path, approved_output_roots=(path.parent,), **kwargs)


_train_impl = trainer.train


def _train_for_test(config, *args, **kwargs):
    kwargs.setdefault(
        "approved_data_roots",
        (Path(config["data_root"]).parent,),
    )
    kwargs.setdefault("approved_output_roots", (Path(config["output_dir"]).parent,))
    return _train_impl(config, *args, **kwargs)


trainer.train = _train_for_test


def _validate_best_checkpoint(best_path, output_dir, snapshot, config_path):
    return _validate_best_checkpoint_impl(
        best_path,
        output_dir,
        snapshot,
        config_path,
        (snapshot.parent,),
        (config_path.parent,),
        (output_dir.parent,),
    )
from vacca_bcs.category_split_plan import split_identity_digest  # noqa: E402
from vacca_bcs.metrics import (  # noqa: E402
    METRICS_TOLERANCE,
    assert_metrics_match_confusion,
    derive_category_metrics,
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


def test_category_domain_constants_reject_fractional_class_names() -> None:
    assert BCS_DOMAIN_ID == "bcs-category-1-5-v1"
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


def test_predict_maps_passed_thresholds_to_class_and_category() -> None:
    logits = torch.tensor(
        [
            [-10.0, -10.0, -10.0, -10.0],
            [10.0, -10.0, -10.0, -10.0],
            [10.0, 10.0, -10.0, -10.0],
            [10.0, 10.0, 10.0, -10.0],
            [10.0, 10.0, 10.0, 10.0],
        ]
    )
    class_idx, categories = predict(logits)
    assert torch.equal(class_idx, torch.arange(5))
    assert torch.equal(categories, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))


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

    class_idx, categories = predict(logits)
    assert torch.all((class_idx >= 0) & (class_idx < len(CLASS_NAMES)))
    assert torch.all((categories >= 1.0) & (categories <= 5.0))


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
    predicted_levels, categories = model.predict(images)
    assert logits.shape == (2, len(CLASS_NAMES) - 1)
    assert predicted_levels.shape == (2,)
    assert categories.shape == (2,)
    assert torch.isfinite(logits).all()
    assert torch.is_floating_point(categories)
    assert torch.equal(categories, torch.full((2,), 1.0))

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
        "data_root": "data/bcs-category-v1",
        "output_dir": str(tmp_path / "out"),
        "epochs": 2,
        "batch_size": 2,
        "lr": 0.001,
        "weight_decay": 0.0,
        "optimizer": "AdamW",
        "patience": 2,
        "num_workers": 0,
        "val_num_workers": 0,
        "val_seed": 1,
        "progress_every_batches": 50,
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


def test_trainer_entry_rejects_symlinked_config_path(tmp_path: Path, capsys) -> None:
    config = _write_config(tmp_path)
    link = tmp_path / "config-link.yaml"
    try:
        link.symlink_to(config)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable")
    assert train_main(["--config", str(link)]) == 1
    assert "unsafe config path" in capsys.readouterr().err


def test_trainer_entry_rejects_symlinked_output_ancestor(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "output-link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable")
    config_values = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_values["output_dir"] = str(link / "run")
    with pytest.raises(ValueError, match="symlink"):
        trainer.train(config_values)


@pytest.mark.parametrize("bad_output", ["data/bcs-category-v1", "data/bcs/dataset"])
def test_trainer_cli_rejects_data_paths_for_output_before_creation(
    tmp_path: Path, bad_output: str
) -> None:
    (tmp_path / "configs").mkdir()
    config_path = _write_config(
        tmp_path / "configs",
        data_root=str(tmp_path / "data"),
        output_dir=str(tmp_path / "outputs" / "run"),
    )
    bad_path = tmp_path / bad_output
    assert train_main(
        ["--config", str(config_path), "--output-dir", str(bad_path)],
        root=tmp_path,
    ) == 1
    assert not bad_path.exists()


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


@pytest.mark.parametrize(
    ("key", "value", "valid"),
    [
        ("seed", TORCH_SEED_MAX, True),
        ("seed", TORCH_SEED_MAX + 1, False),
        ("val_seed", TORCH_SEED_MAX, True),
        ("val_seed", TORCH_SEED_MAX + 1, False),
    ],
)
def test_torch_seed_bounds_are_validated_before_output_creation(
    tmp_path: Path, key: str, value: int, valid: bool
) -> None:
    values = {"seed": 42}
    values[key] = value
    path = _write_config(tmp_path, **values)
    output = tmp_path / "out"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    if valid:
        loaded = load_config(path)
        assert loaded[key] == value
    else:
        with pytest.raises(ValueError, match="seed|val_seed"):
            load_config(path)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_derived_validation_seed_is_rejected_before_output_creation(tmp_path: Path) -> None:
    path = _write_config(tmp_path, seed=TORCH_SEED_MAX)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config.pop("val_seed")
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "out"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="val_seed"):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        trainer.train(config)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_training_workers_are_fixed_at_zero_and_validation_workers_are_configurable(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="num_workers must be 0"):
        load_config(_write_config(tmp_path, num_workers=1))
    config = load_config(_write_config(tmp_path, val_num_workers=2, val_seed=99))
    assert config["val_num_workers"] == 2
    assert config["val_seed"] == 99


def test_data_loader_omits_multiprocessing_kwargs_for_worker_zero(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    class _Loader:
        def __init__(self, dataset, **kwargs) -> None:
            captured.append(kwargs)

    monkeypatch.setattr(trainer, "DataLoader", _Loader)
    trainer._build_data_loader(
        object(),
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    trainer._build_data_loader(
        object(),
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        generator=trainer._validation_generator(99),
    )
    assert "prefetch_factor" not in captured[0]
    assert "persistent_workers" not in captured[0]
    assert "worker_init_fn" not in captured[0]
    assert captured[1]["prefetch_factor"] == 2
    assert captured[1]["persistent_workers"] is False
    assert captured[1]["worker_init_fn"] is trainer._worker_init_fn


def test_validation_generator_and_worker_seed_are_independent(monkeypatch) -> None:
    torch.manual_seed(123)
    before = torch.get_rng_state()
    first = trainer._validation_generator(99)
    second = trainer._validation_generator(99)
    after = torch.get_rng_state()
    assert torch.equal(before, after)
    assert torch.equal(torch.rand(4, generator=first), torch.rand(4, generator=second))

    seeds: list[int] = []
    monkeypatch.setattr(trainer.random, "seed", lambda seed: seeds.append(seed))
    monkeypatch.setattr(trainer.torch, "initial_seed", lambda: 2**32 + 17)
    trainer._worker_init_fn(0)
    assert seeds == [17]


def test_validation_workers_preserve_order_and_parent_training_rng() -> None:
    dataset = [(torch.tensor([float(index)]), index) for index in range(5)]
    set_seed(1234)
    before = torch.get_rng_state()
    loader = trainer._build_data_loader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        generator=trainer._validation_generator(99),
    )
    observed = [(images[:, 0].tolist(), levels.tolist()) for images, levels in loader]
    after = torch.get_rng_state()
    assert torch.equal(before, after)
    assert observed == [
        ([0.0, 1.0], [0, 1]),
        ([2.0, 3.0], [2, 3]),
        ([4.0], [4]),
    ]


def test_coverage_policy_is_unconditional_and_unknown_legacy_keys_are_rejected(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    assert config["data_root"] == "data/bcs-category-v1"
    assert config["output_dir"] == str(tmp_path / "out")
    with pytest.raises(ValueError, match="Unsupported training config keys"):
        load_config(_write_config(tmp_path, legacy_coverage_switch=False))


def test_class_coverage_is_derived_from_manifest_counts_and_requires_every_class() -> None:
    manifest = {
        "counts": {
            "train": [0, 0, 2, 3, 0],
            "val": [0, 0, 1, 1, 0],
            "test": [0, 0, 0, 0, 0],
        }
    }
    coverage = _coverage_from_manifest(manifest)
    assert coverage == {
        "observed_classes": [3, 4],
        "missing_classes": [1, 2, 5],
    }
    with pytest.raises(ValueError, match="snapshot train split"):
        _validate_class_coverage(manifest)


def test_class_coverage_rejects_empty_category_cells() -> None:
    for counts, message in (
        (
            {
                "train": [0, 0, 2, 0, 0],
                "val": [0, 0, 1, 0, 0],
                "test": [0, 0, 1, 0, 0],
            },
            "snapshot train split",
        ),
        (
            {
                "train": [0, 0, 2, 2, 0],
                "val": [0, 0, 1, 1, 0],
                "test": [0, 0, 1, 1, 0],
            },
            "snapshot train split",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            _validate_class_coverage({"counts": counts})


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


@pytest.mark.parametrize("target", ["output", "weights"])
def test_overwrite_rejects_symlinked_output_paths(tmp_path: Path, target: str) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    try:
        if target == "output":
            output_dir.rmdir()
            output_dir.symlink_to(outside, target_is_directory=True)
        else:
            weights = output_dir / "weights"
            weights.symlink_to(outside, target_is_directory=True)
            (outside / "best.pt").write_bytes(b"must survive")
        with pytest.raises(ValueError, match="symlink"):
            _prepare_output_dir(output_dir, overwrite=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable")
    if target == "weights":
        assert (outside / "best.pt").read_bytes() == b"must survive"


def test_prepare_output_dir_allows_empty_fresh_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "new-run"
    _prepare_output_dir(output_dir, overwrite=False)
    assert output_dir.is_dir()


def test_main_fails_clearly_when_output_exists(tmp_path: Path, capsys) -> None:
    (tmp_path / "configs").mkdir()
    output_dir = tmp_path / "outputs" / "run"
    _seed_run_artifacts(output_dir)
    config_path = _write_config(
        tmp_path / "configs",
        data_root=str(tmp_path / "data"),
        output_dir=str(output_dir),
    )
    exit_code = train_main(["--config", str(config_path)], root=tmp_path)
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
    assert checkpoint["checkpoint_schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert checkpoint["domain_id"] == BCS_DOMAIN_ID

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


def test_atomic_checkpoint_symlink_is_rejected_without_touching_external_target(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external.pt"
    external.write_bytes(b"external-checkpoint")
    path = tmp_path / "last.pt"
    try:
        path.symlink_to(external)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable")
    with pytest.raises(ValueError, match="symlink"):
        _atomic_torch_save({"epoch": 2}, path)
    assert external.read_bytes() == b"external-checkpoint"


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
        "val_within_one": "0.9",
        "val_ordinal_mae": "0.25",
        "val_error_ge_2": "0.1",
        "val_macro_f1": "0.5",
        "val_balanced_accuracy": "0.5",
        "val_support": json.dumps([1, 1, 1, 1, 1]),
        "val_precision": json.dumps({name: 0.5 for name in CLASS_NAMES}, sort_keys=True),
        "val_recall": json.dumps({name: 0.5 for name in CLASS_NAMES}, sort_keys=True),
        "val_f1": json.dumps({name: 0.5 for name in CLASS_NAMES}, sort_keys=True),
        "val_confusion_matrix": json.dumps([[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]),
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


def test_results_csv_symlink_is_rejected_without_touching_external_target(tmp_path: Path) -> None:
    external = tmp_path / "external.csv"
    external.write_text("external-content\n", encoding="utf-8")
    linked = tmp_path / "results.csv"
    try:
        linked.symlink_to(external)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable")
    with pytest.raises(ValueError, match="symlink"):
        _open_results_csv(linked, append=True)
    assert external.read_text(encoding="utf-8") == "external-content\n"


def _write_results_history(path: Path, epochs: list[int]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(RESULTS_FIELDNAMES)
        for epoch in epochs:
            writer.writerow([_results_row(epoch)[field] for field in RESULTS_FIELDNAMES])


def _results_line(epoch: int) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([_results_row(epoch)[field] for field in RESULTS_FIELDNAMES])
    return output.getvalue().rstrip("\r\n")


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
        + "\n"
        + _results_line(1)
        + "\n2,0.001,",
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
        ("val_within_one", "-0.1"),
        ("val_ordinal_mae", "4.1"),
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
        + "\n"
        + _results_line(1)
        + "\n"
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
        + "\n"
        + _results_line(1)
        + "\n"
        + _results_line(1).replace("1,", "abc,", 1)
        + "\n",
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
    [("train_loss", "nan"), ("val_exact_acc", "1.1"), ("val_ordinal_mae", "-0.01")],
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


def test_validate_reports_null_recall_for_unobserved_validation_classes() -> None:
    class _FixedLogitsModel(torch.nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[-8.0, -8.0, -8.0, -8.0]])

    metrics = trainer._validate(
        _FixedLogitsModel(),
        [(torch.zeros(1, 3, 8, 8), torch.tensor([0]))],
        torch.device("cpu"),
    )
    assert metrics["recall"]["1"] == pytest.approx(1.0)
    assert metrics["recall"]["2"] is None
    assert metrics["recall"]["5"] is None


def test_provisional_test_gates_are_explicit_engineering_only() -> None:
    metrics = {
        "macro_f1": 0.75,
        "balanced_accuracy": 0.75,
        "ordinal_mae": 0.35,
        "f1": {name: 0.70 for name in CLASS_NAMES},
        "within_one_by_class": {name: 0.95 for name in CLASS_NAMES},
        "error_ge_2_by_class": {name: 0.05 for name in CLASS_NAMES},
    }
    config = {"provisional_acceptance_gates": dict(trainer.PROVISIONAL_ACCEPTANCE_GATES)}
    result = trainer._evaluate_provisional_gates(metrics, config)
    assert result["kind"] == "provisional_engineering_only"
    assert result["passed"] is True
    metrics["ordinal_mae"] = 0.350001
    assert trainer._evaluate_provisional_gates(metrics, config)["passed"] is False


def test_metrics_are_derived_from_a_cyclic_all_wrong_confusion_matrix() -> None:
    matrix = [
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0],
    ]
    metrics = derive_category_metrics(matrix)
    assert metrics["support"] == [1] * NUM_CLASSES
    assert metrics["total"] == NUM_CLASSES
    assert metrics["exact_acc"] == 0.0
    assert metrics["macro_f1"] == 0.0
    assert metrics["f1"] == {name: None for name in CLASS_NAMES}
    assert_metrics_match_confusion(metrics)


@pytest.mark.parametrize("mutation", [
    lambda metrics: metrics.update(support=[2, 1, 1, 1, 1]),
    lambda metrics: metrics.update(total=6),
    lambda metrics: metrics.update(macro_f1=1.0),
    lambda metrics: metrics["f1"].update({CLASS_NAMES[0]: 1.0}),
    lambda metrics: metrics["within_one_by_class"].update({CLASS_NAMES[0]: 0.0}),
])
def test_metrics_evidence_rejects_support_aggregate_and_per_class_tampering(mutation) -> None:
    matrix = [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [1, 0, 0, 0, 0]]
    metrics = derive_category_metrics(matrix)
    mutation(metrics)
    with pytest.raises(ValueError, match="does not match"):
        assert_metrics_match_confusion(metrics)


def test_metrics_evidence_uses_documented_tight_float_tolerance() -> None:
    metrics = derive_category_metrics([[1, 0, 0, 0, 0]] * NUM_CLASSES)
    metrics["macro_f1"] += METRICS_TOLERANCE
    assert_metrics_match_confusion(metrics)
    metrics["macro_f1"] += METRICS_TOLERANCE
    with pytest.raises(ValueError, match="macro_f1"):
        assert_metrics_match_confusion(metrics)


@pytest.mark.parametrize(
    ("gate", "value"),
    [
        ("macro_f1_min", 0),
        ("balanced_accuracy_min", 1.000001),
        ("class_f1_min", float("nan")),
        ("class_within_one_min", -0.001),
        ("class_error_ge_2_max", -0.001),
        ("class_error_ge_2_max", 1),
        ("ordinal_mae_max", -0.001),
        ("ordinal_mae_max", SCORE_STEP * (NUM_CLASSES - 1)),
        ("ordinal_mae_max", float("inf")),
    ],
)
def test_provisional_gate_domains_reject_invalid_or_vacuous_boundaries(
    tmp_path: Path, gate: str, value: object
) -> None:
    gates = dict(trainer.PROVISIONAL_ACCEPTANCE_GATES)
    gates[gate] = value
    with pytest.raises(ValueError, match="provisional_acceptance_gates"):
        load_config(_write_config(tmp_path, provisional_acceptance_gates=gates))


@pytest.mark.parametrize("failed_check", [
    "macro_f1", "balanced_accuracy", "every_class_f1", "every_class_within_one",
    "every_class_error_ge_2", "ordinal_mae",
])
def test_each_provisional_gate_failure_direction_is_reported(failed_check: str) -> None:
    metrics = {
        "macro_f1": 0.75,
        "balanced_accuracy": 0.75,
        "ordinal_mae": 0.35,
        "f1": {name: 0.70 for name in CLASS_NAMES},
        "within_one_by_class": {name: 0.95 for name in CLASS_NAMES},
        "error_ge_2_by_class": {name: 0.05 for name in CLASS_NAMES},
    }
    if failed_check == "macro_f1":
        metrics["macro_f1"] = 0.749
    elif failed_check == "balanced_accuracy":
        metrics["balanced_accuracy"] = 0.749
    elif failed_check == "every_class_f1":
        metrics["f1"][CLASS_NAMES[0]] = 0.699
    elif failed_check == "every_class_within_one":
        metrics["within_one_by_class"][CLASS_NAMES[0]] = 0.949
    elif failed_check == "every_class_error_ge_2":
        metrics["error_ge_2_by_class"][CLASS_NAMES[0]] = 0.051
    else:
        metrics["ordinal_mae"] = 0.351
    config = {"provisional_acceptance_gates": dict(trainer.PROVISIONAL_ACCEPTANCE_GATES)}
    result = trainer._evaluate_provisional_gates(metrics, config)
    assert result["passed"] is False
    assert result["checks"][failed_check] is False


def test_optimized_validation_matches_reference_with_partial_final_batch() -> None:
    class _FixedLogitsModel(torch.nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            logits = torch.tensor(
                [
                    [-8.0, -8.0, -8.0, -8.0],
                    [8.0, 8.0, -8.0, -8.0],
                    [8.0, 8.0, 8.0, 8.0],
                ]
            )
            return logits[images[:, 0, 0, 0].long()]

    loader = [
        (torch.tensor([[[[0.0]]], [[[1.0]]]]), torch.tensor([0, 2])),
        (torch.tensor([[[[2.0]]]]), torch.tensor([3])),
    ]
    model = _FixedLogitsModel()
    optimized = trainer._validate(model, loader, torch.device("cpu"))

    exact = 0
    pm1 = 0
    absolute_error = 0
    total = 0
    class_total = [0] * len(CLASS_NAMES)
    class_correct = [0] * len(CLASS_NAMES)
    with torch.no_grad():
        for images, levels in loader:
            pred_idx, _ = predict(model(images))
            errors = (pred_idx - levels).abs()
            exact += int((errors == 0).sum().item())
            pm1 += int((errors <= 1).sum().item())
            absolute_error += int(errors.sum().item())
            total += levels.shape[0]
            for class_idx in range(len(CLASS_NAMES)):
                mask = levels == class_idx
                class_total[class_idx] += int(mask.sum().item())
                class_correct[class_idx] += int(((pred_idx == class_idx) & mask).sum().item())
    reference = {
        "exact_acc": exact / total,
        "mae": absolute_error / total,
        "within_one": pm1 / total,
        "ordinal_mae": absolute_error / total,
        "error_ge_2": 0.0,
        "macro_f1": 1.0,
        "balanced_accuracy": 2 / 3,
        "within_one_by_class": {"1": 1.0, "2": None, "3": 1.0, "4": 1.0, "5": None},
        "error_ge_2_by_class": {"1": 0.0, "2": None, "3": 0.0, "4": 0.0, "5": None},
        "support": [1, 0, 1, 1, 0],
        "precision": {"1": 1.0, "2": None, "3": 1.0, "4": None, "5": 0.0},
        "recall": {
            name: (
                class_correct[index] / class_total[index]
                if class_total[index]
                else None
            )
            for index, name in enumerate(CLASS_NAMES)
        },
        "f1": {"1": 1.0, "2": None, "3": 1.0, "4": None, "5": None},
        "confusion_matrix": [
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ],
        "total": total,
    }
    assert optimized == reference


def test_training_updates_and_loss_match_reference_with_seeded_batches(capsys) -> None:
    batches = [
        (
            torch.tensor([[float((index % 7) + 1)]]),
            torch.tensor([index % 4]),
        )
        for index in range(512)
    ]

    def run_optimized() -> tuple[dict[str, torch.Tensor], dict, float]:
        set_seed(17)
        model = torch.nn.Linear(1, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        loss = trainer._train_epoch(model, batches, optimizer, torch.device("cpu"))
        return model.state_dict(), optimizer.state_dict(), loss

    def run_reference() -> tuple[dict[str, torch.Tensor], dict, float]:
        set_seed(17)
        model = torch.nn.Linear(1, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        loss_total = 0.0
        samples = 0
        for images, levels in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = coral_loss(model(images), levels)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
            samples += levels.shape[0]
        return model.state_dict(), optimizer.state_dict(), loss_total / samples

    optimized = run_optimized()
    reference = run_reference()
    assert f"{optimized[2]:.8f}" == f"{reference[2]:.8f}"
    assert all(
        torch.equal(optimized[0][key], reference[0][key])
        for key in optimized[0]
    )
    assert optimized[1]["param_groups"] == reference[1]["param_groups"]
    assert optimized[1]["state"].keys() == reference[1]["state"].keys()
    for parameter_id, state in optimized[1]["state"].items():
        for key, value in state.items():
            reference_value = reference[1]["state"][parameter_id][key]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, reference_value)
            else:
                assert value == reference_value
    assert capsys.readouterr().out == ""


def test_progress_uses_cadence_final_batch_and_ascii_flush(capsys) -> None:
    set_seed(18)
    model = torch.nn.Linear(1, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    batches = [
        (torch.ones(1, 1), torch.tensor([0])),
        (torch.ones(1, 1), torch.tensor([1])),
        (torch.ones(1, 1), torch.tensor([2])),
    ]
    trainer._train_epoch(
        model,
        batches,
        optimizer,
        torch.device("cpu"),
        epoch=11,
        total_epochs=30,
        progress_every_batches=2,
    )
    lines = capsys.readouterr().out.splitlines()
    assert [line.split("]", 1)[0] + "]" for line in lines] == [
        "[TRAIN 11/30 2/3]",
        "[TRAIN 11/30 3/3]",
    ]
    assert all(line.isascii() for line in lines)
    assert all("loss=" in line and "elapsed=" in line and "eta=" in line for line in lines)


def test_training_progress_prints_are_flushed(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **kwargs: calls.append(kwargs),
    )
    model = torch.nn.Linear(1, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    trainer._train_epoch(
        model,
        [(torch.ones(1, 1), torch.tensor([0]))],
        optimizer,
        torch.device("cpu"),
        epoch=1,
        total_epochs=1,
        progress_every_batches=50,
    )
    assert calls and all(call.get("flush") is True for call in calls)


def test_results_csv_validates_deterministic_partial_recall_json() -> None:
    row = _results_row(1)
    row["val_recall"] = json.dumps(
        {"1": 1.0, "2": None, "3": 0.5, "4": None, "5": None}, sort_keys=True
    )
    trainer._validate_results_row(
        [row[field] for field in RESULTS_FIELDNAMES], line_number=2, expected_epoch=1
    )


def _tiny_training_config(tmp_path: Path, output: Path, *, patience: int = 10) -> dict[str, object]:
    data_dir = tmp_path / "dataset"
    records = []
    source_labels = ("3.25", "3.5", "3.75", "4.0", "4.25")
    split_order = {"train": 0, "val": 1, "test": 2}
    for split in ("train", "val", "test"):
        for class_index, class_name in enumerate(CLASS_NAMES):
            class_dir = data_dir / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            source_path = f"{source_labels[class_index]}/GS_{split_order[split] + 1}_{class_index + 1}.jpg"
            record_id = hashlib.sha256(
                f"bcs-local-category-source-v1\0{source_path}".encode()
            ).hexdigest()
            path = class_dir / f"{record_id}.jpg"
            buffer = io.BytesIO()
            Image.new("RGB", (8, 8), (class_index * 30 + 10, split_order[split] * 50 + 20, 80)).save(
                buffer, format="JPEG"
            )
            content = buffer.getvalue()
            path.write_bytes(content)
            records.append(
                {
                    "split": split,
                    "bcs_category": class_index + 1,
                    "record_id": record_id,
                    "relative_path": f"{split}/{class_name}/{record_id}.jpg",
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "capture_group": f"{split}|{class_index + 1}",
                    "provenance": [{"relative_path": source_path, "source_label": source_labels[class_index]}],
                }
            )
    records.sort(key=lambda item: (item["bcs_category"], split_order[item["split"]], item["record_id"]))
    counts = {split: [1] * NUM_CLASSES for split in ("train", "val", "test")}
    identity_digest = split_identity_digest(
        source_schema="bcs-local-category-source-v1",
        identity_scheme="local-path-sha256-v1",
        mapping_lineage=tuple(zip(source_labels, BCS_CLASS_SCORES)),
        observed_classes=BCS_CLASS_SCORES,
        seed=7,
        canonical_val_ratio="0",
        canonical_test_ratio="0",
        assignments=[
            (item["split"], item["bcs_category"], item["record_id"], item["capture_group"], item["sha256"])
            for item in sorted(records, key=lambda item: item["record_id"])
        ],
        counts={split: tuple(counts[split]) for split in ("train", "val", "test")},
        exclusions=[],
    )
    (data_dir / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_schema_version": "bcs-category-snapshot-v1",
                "domain_id": BCS_DOMAIN_ID,
                "class_values": list(BCS_CLASS_SCORES),
                "class_mapping": {name: index for index, name in enumerate(CLASS_NAMES)},
                "score_min": SCORE_MIN,
                "score_max": SCORE_MAX,
                "score_base": SCORE_BASE,
                "score_step": SCORE_STEP,
                "num_classes": NUM_CLASSES,
                "num_thresholds": NUM_THRESHOLDS,
                "source_schema": "bcs-local-category-source-v1",
                "identity_scheme": "local-path-sha256-v1",
                "mapping": dict(zip(source_labels, BCS_CLASS_SCORES)),
                "observed_classes": list(BCS_CLASS_SCORES),
                "counts": counts,
                "split_plan": {
                    "identity_digest": identity_digest,
                    "seed": 7,
                    "canonical_val_ratio": "0",
                    "canonical_test_ratio": "0",
                    "candidate_record_ids": sorted(record["record_id"] for record in records),
                    "excluded_record_ids": [],
                    "counts": counts,
                    "capture_group_count": len(records),
                    "digest_count": len(records),
                },
                "isolation": {"capture_group_count": len(records), "digest_count": len(records), "overlap": []},
                "records": records,
                "exclusions": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "data_root": str(data_dir),
        "output_dir": str(output),
        "epochs": 3,
        "batch_size": 1,
        "lr": 0.01,
        "weight_decay": 0.0,
        "optimizer": "AdamW",
        "lr_schedule": "cosine",
        "warmup_epochs": 0,
        "patience": patience,
        "num_workers": 0,
        "val_num_workers": 2,
        "val_seed": 8,
        "progress_every_batches": 50,
        "imgsz": 8,
        "device": "cpu",
        "seed": 7,
        "provisional_acceptance_gates": dict(trainer.PROVISIONAL_ACCEPTANCE_GATES),
        "_config_path": str(tmp_path / "config.yaml"),
    }


def _valid_best_artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    output = tmp_path / "run"
    config = _tiny_training_config(tmp_path, output)
    config_path = Path(config["_config_path"])
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    data_dir = Path(config["data_root"])
    provenance = _build_provenance(
        config,
        data_dir=data_dir,
        output_dir=output,
        device=torch.device("cpu"),
        run_id="a" * 32,
    )
    model = BCSOrdinalModel(pretrained=False)
    best_path = output / "weights" / "best.pt"
    best_validation = {
        "exact_acc": 1.0,
        "within_one": 1.0,
        "ordinal_mae": 0.0,
        "error_ge_2": 0.0,
        "macro_f1": 1.0,
        "balanced_accuracy": 1.0,
        "f1": {name: 1.0 for name in CLASS_NAMES},
        "within_one_by_class": {name: 1.0 for name in CLASS_NAMES},
        "error_ge_2_by_class": {name: 0.0 for name in CLASS_NAMES},
        "precision": {name: 1.0 for name in CLASS_NAMES},
        "recall": {name: 1.0 for name in CLASS_NAMES},
    }
    validation_metrics = {
        **best_validation,
        "mae": best_validation["ordinal_mae"],
        "support": [1] * NUM_CLASSES,
        "confusion_matrix": [[1 if row == col else 0 for col in range(NUM_CLASSES)] for row in range(NUM_CLASSES)],
        "total": NUM_CLASSES,
    }
    selection_identity = trainer._selection_identity(provenance, 1, best_validation)
    _atomic_torch_save(
        {
            **_checkpoint_lineage(provenance, provenance["run_id"]),
            "model_state_dict": model.state_dict(),
            "config": config,
            "classes": list(CLASS_NAMES),
            "provenance": provenance,
            "epoch": 1,
            "val_ordinal_mae": 0.0,
            "best_epoch": 1,
            "best_validation": best_validation,
            "selection_identity": selection_identity,
            "best_results_row": trainer._results_row(1, lr=0.01, train_loss=1.0, metrics=validation_metrics),
        },
        best_path,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    last_path = output / "weights" / "last.pt"
    _atomic_torch_save(
        _build_last_checkpoint(
            model,
            optimizer,
            epoch=1,
            best_mae=0.0,
            epochs_without_improvement=0,
            config=config,
            provenance=provenance,
            best_epoch=1,
            selection_identity=selection_identity,
            best_validation=best_validation,
        ),
        last_path,
    )
    trainer._write_checkpoint_set(
        output / "weights",
        best_digest=hashlib.sha256(best_path.read_bytes()).hexdigest(),
        last_digest=hashlib.sha256(last_path.read_bytes()).hexdigest(),
        provenance=provenance,
        approved_output_roots=(tmp_path,),
    )
    _atomic_write_json(
        trainer._results_lineage(provenance, provenance["run_id"]),
        output / RESULTS_LINEAGE_FILENAME,
    )
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(RESULTS_FIELDNAMES)
        writer.writerow(
            [trainer._results_row(1, lr=0.01, train_loss=1.0, metrics=validation_metrics)[field] for field in RESULTS_FIELDNAMES]
        )
    best_checkpoint = {
        "path": str(best_path.resolve()),
        "sha256": hashlib.sha256(best_path.read_bytes()).hexdigest(),
        "run_id": provenance["run_id"],
        "best_epoch": 1,
        "selection_identity": selection_identity,
        "validation": best_validation,
    }
    last_checkpoint = {
        "path": str(last_path.resolve()),
        "sha256": hashlib.sha256(last_path.read_bytes()).hexdigest(),
        "run_id": provenance["run_id"],
    }
    test_metrics = {
        "evaluated_checkpoint": str(best_path.resolve()),
        "checkpoint_sha256": best_checkpoint["sha256"],
        "run_id": provenance["run_id"],
        "best_epoch": 1,
        "selection_identity": selection_identity,
        "config_sha256": provenance["config_sha256"],
        "snapshot_identity": provenance["dataset_manifest"]["split_identity"],
        "dataset_manifest_digest": provenance["dataset_manifest"]["sha256"],
        **validation_metrics,
    }
    run_info = {
        "run_id": provenance["run_id"],
        "config_sha256": provenance["config_sha256"],
        "snapshot_identity": provenance["dataset_manifest"]["split_identity"],
        "dataset_manifest_digest": provenance["dataset_manifest"]["sha256"],
        "provenance": provenance,
        "candidate_status": "candidate_pending_handoff",
        "best_checkpoint": best_checkpoint,
        "last_checkpoint": last_checkpoint,
        "test_metrics": test_metrics,
        "provisional_acceptance": trainer._evaluate_provisional_gates(test_metrics, config),
    }
    _atomic_write_json(run_info, output / "run_info.json")
    return best_path, output, data_dir, config_path


def test_overnight_validates_best_checkpoint_and_regression_direction(
    tmp_path: Path, monkeypatch
) -> None:
    best, output, data_dir, config_path = _valid_best_artifacts(tmp_path)
    monkeypatch.setattr(
        trainer,
        "_validate",
        lambda *args: {
            "exact_acc": 0.96433906,
            "mae": 0.03566094,
        },
    )
    result = _validate_best_checkpoint(best, output, data_dir, config_path)
    assert result["checkpoint"] == str(best)
    assert result["category_contract"] == "1..5"


def test_overnight_rejects_substituted_or_unbound_last_checkpoint(
    tmp_path: Path,
) -> None:
    best, output, data_dir, config_path = _valid_best_artifacts(tmp_path)
    last_path = output / "weights" / "last.pt"
    last_path.write_bytes(best.read_bytes())
    with pytest.raises(RuntimeError, match="last|checkpoint|digest"):
        _validate_best_checkpoint(best, output, data_dir, config_path)

    best, output, data_dir, config_path = _valid_best_artifacts(tmp_path / "missing")
    run_info_path = output / "run_info.json"
    run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
    del run_info["last_checkpoint"]
    run_info_path.write_text(json.dumps(run_info), encoding="utf-8")
    with pytest.raises(RuntimeError, match="last|checkpoint"):
        _validate_best_checkpoint(best, output, data_dir, config_path)


def test_overnight_rejects_alias_only_checkpoint_set(tmp_path: Path) -> None:
    best, output, data_dir, config_path = _valid_best_artifacts(tmp_path)
    (output / "weights" / trainer.CHECKPOINT_SET_FILENAME).unlink()

    with pytest.raises(RuntimeError, match="authoritative|checkpoint set"):
        _validate_best_checkpoint(best, output, data_dir, config_path)


def test_overnight_rejects_foreign_same_schema_or_stale_results(tmp_path: Path) -> None:
    best, output, data_dir, config_path = _valid_best_artifacts(tmp_path)
    checkpoint = torch.load(best, map_location="cpu", weights_only=True)
    checkpoint["snapshot_identity"] = "d" * 64
    checkpoint["provenance"]["dataset_manifest"]["split_identity"] = "d" * 64
    torch.save(checkpoint, best)
    with pytest.raises(RuntimeError, match="handoff|checkpoint|lineage"):
        _validate_best_checkpoint(best, output, data_dir, config_path)

    best, output, data_dir, config_path = _valid_best_artifacts(tmp_path / "results")
    lineage_path = output / RESULTS_LINEAGE_FILENAME
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage["run_id"] = "d" * 32
    lineage_path.write_text(json.dumps(lineage), encoding="utf-8")
    with pytest.raises(RuntimeError, match="handoff|lineage"):
        _validate_best_checkpoint(best, output, data_dir, config_path)

    best, output, data_dir, config_path = _valid_best_artifacts(tmp_path / "live")
    live_image = next((data_dir / "train" / CLASS_NAMES[0]).glob("*.jpg"))
    live_image.write_bytes(live_image.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="live snapshot|digest"):
        _validate_best_checkpoint(best, output, data_dir, config_path)

    best, output, data_dir, config_path = _valid_best_artifacts(tmp_path / "metric")
    results_path = output / "results.csv"
    with results_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["val_macro_f1"] = "0.99"
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(RuntimeError, match="selected row|validation row"):
        _validate_best_checkpoint(best, output, data_dir, config_path)


@pytest.mark.parametrize("mutation", ["corrupt", "foreign"])
def test_overnight_rejects_corrupt_or_foreign_best_checkpoint(
    tmp_path: Path, mutation: str
) -> None:
    best, output, data_dir, config_path = _valid_best_artifacts(tmp_path)
    if mutation == "corrupt":
        best.write_bytes(b"corrupt")
    else:
        checkpoint = torch.load(best, map_location="cpu", weights_only=True)
        checkpoint["run_id"] = "b" * 32
        torch.save(checkpoint, best)
    with pytest.raises(RuntimeError, match="best checkpoint"):
        _validate_best_checkpoint(best, output, data_dir, config_path)


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
        data_dir=Path(config["data_root"]),
        output_dir=Path(config["output_dir"]),
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
    data_dir = Path(config["data_root"])
    manifest_path = data_dir / "manifest.json"
    if mutation == "added":
        (data_dir / "train" / CLASS_NAMES[0] / "added.jpg").write_bytes(b"added")
    elif mutation == "missing":
        next((data_dir / "train" / CLASS_NAMES[0]).iterdir()).unlink()
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["records"][0]["relative_path"] = "../escape.jpg"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest|membership|unsafe path|record is invalid"):
        _build_provenance(
            config,
            data_dir=data_dir,
            output_dir=Path(config["output_dir"]),
            device=torch.device("cpu"),
        )


def test_category_manifest_is_accepted_and_old_lineages_are_rejected(tmp_path: Path) -> None:
    config = _tiny_training_config(tmp_path, tmp_path / "out")
    manifest_path = Path(config["data_root"]) / "manifest.json"
    assert _build_provenance(
        config, data_dir=Path(config["data_root"]), output_dir=Path(config["output_dir"]), device=torch.device("cpu")
    )["classes"] == list(CLASS_NAMES)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manifest_schema_version"] = "bcs-integer-snapshot-v1"
    manifest["records"][0]["storage_key"] = "private-storage-key"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="legacy|schema") as failure:
        _build_provenance(
            config, data_dir=Path(config["data_root"]), output_dir=Path(config["output_dir"]), device=torch.device("cpu")
        )
    assert "private-storage-key" not in str(failure.value)
    manifest["manifest_schema_version"] = "bcs-category-snapshot-v1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="fields|legacy|schema"):
        _build_provenance(
            config, data_dir=Path(config["data_root"]), output_dir=Path(config["output_dir"]), device=torch.device("cpu")
        )


@pytest.mark.parametrize("mutation", [
    lambda manifest: manifest["records"][0].update(relative_path="val/1/1.jpg"),
    lambda manifest: manifest["records"][0].update(bcs_category=5),
    lambda manifest: manifest.update(class_mapping={name: 0 for name in CLASS_NAMES}),
])
def test_category_manifest_rejects_record_and_class_tampering(tmp_path: Path, mutation) -> None:
    config = _tiny_training_config(tmp_path, tmp_path / "out")
    path = Path(config["data_root"]) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutation(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        _build_provenance(config, data_dir=Path(config["data_root"]), output_dir=Path(config["output_dir"]), device=torch.device("cpu"))


def test_default_category_training_roots_are_canonical(tmp_path: Path) -> None:
    config = yaml.safe_load((REPO_ROOT / "configs" / "training_bcs_category.yaml").read_text())
    assert config["data_root"] == "data/bcs-category-v1"
    assert config["output_dir"] == "outputs/bcs-category-coral-v1"


@pytest.mark.parametrize("mutation", ["missing", "unexpected"])
def test_category_snapshot_rejects_root_structure_tampering(tmp_path: Path, mutation: str) -> None:
    config = _tiny_training_config(tmp_path, tmp_path / "out")
    root = Path(config["data_root"])
    if mutation == "missing":
        next((root / "val" / CLASS_NAMES[-1]).iterdir()).unlink()
        (root / "val" / CLASS_NAMES[-1]).rmdir()
    else:
        (root / "unexpected").mkdir()
    with pytest.raises(ValueError, match="structure|membership"):
        _build_provenance(config, data_dir=root, output_dir=Path(config["output_dir"]), device=torch.device("cpu"))


def _lineage_checkpoint(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    config = _tiny_training_config(tmp_path, tmp_path / "out")
    provenance = _build_provenance(
        config, data_dir=Path(config["data_root"]), output_dir=Path(config["output_dir"]),
        device=torch.device("cpu"), run_id="a" * 32,
    )
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    checkpoint = _build_last_checkpoint(
        model, optimizer, epoch=1, best_mae=0.5, epochs_without_improvement=0,
        config=config, provenance=provenance,
    )
    path = tmp_path / "last.pt"
    torch.save(checkpoint, path)
    return path, provenance


@pytest.mark.parametrize("field,value", [
    ("checkpoint_schema_version", "old-checkpoint"), ("domain_id", "fractional"),
    ("classes", ["fractional"]),
    ("score_step", True), ("snapshot_schema", "bcs-category-snapshot-v0"),
    ("snapshot_identity", "b" * 64), ("dataset_manifest_digest", "c" * 64),
    ("run_id", "d" * 32),
])
def test_resume_rejects_checkpoint_lineage_tampering_before_model_use(
    tmp_path: Path, field: str, value: object
) -> None:
    path, provenance = _lineage_checkpoint(tmp_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint[field] = value
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        _load_resume_checkpoint(
            path, expected_classes=list(CLASS_NAMES), total_epochs=3,
            expected_provenance=provenance,
        )


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
    manifest_path = Path(config["data_root"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest|membership|counts"):
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


def test_overwrite_model_initialization_failure_preserves_prior_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "run"
    config = _tiny_training_config(tmp_path, output)
    artifacts = _seed_prior_training_artifacts(output)

    def fail_model(*args, **kwargs):
        raise RuntimeError("simulated model initialization failure")

    monkeypatch.setattr(trainer, "BCSOrdinalModel", fail_model)
    with pytest.raises(RuntimeError, match="simulated model initialization failure"):
        trainer.train(config, overwrite=True)
    _assert_artifacts_unchanged(artifacts)


def test_overwrite_optimizer_initialization_failure_preserves_prior_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "run"
    config = _tiny_training_config(tmp_path, output)
    artifacts = _seed_prior_training_artifacts(output)
    monkeypatch.setattr(trainer, "BCSOrdinalModel", _TinyModel)

    def fail_optimizer(*args, **kwargs):
        raise RuntimeError("simulated optimizer initialization failure")

    monkeypatch.setattr(trainer.torch.optim, "AdamW", fail_optimizer)
    with pytest.raises(RuntimeError, match="simulated optimizer initialization failure"):
        trainer.train(config, overwrite=True)
    _assert_artifacts_unchanged(artifacts)


def test_atomic_checkpoint_uses_content_addressed_generation_without_sidecar(
    tmp_path: Path,
) -> None:
    path = tmp_path / "weights" / "last.pt"
    _atomic_torch_save({"epoch": 2}, path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    generation = path.parent / "generations" / f"{digest}.pt"

    assert generation.read_bytes() == raw
    assert path.stat().st_ino == generation.stat().st_ino
    assert not path.with_name(path.name + ".sha256").exists()


def test_atomic_checkpoint_set_replacement_failure_preserves_prior_descriptor_and_generation(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "weights" / "last.pt"
    _atomic_torch_save({"epoch": 1}, path)
    descriptor = path.parent / "checkpoint_set.json"
    prior_descriptor = b'{"committed_epoch":1}\n'
    descriptor.write_bytes(prior_descriptor)
    prior_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    prior_generation = path.parent / "generations" / f"{prior_digest}.pt"
    prior_generation_bytes = prior_generation.read_bytes()
    real_replace = trainer.os.replace

    def fail_descriptor_replace(source, destination):
        if Path(destination) == descriptor:
            raise OSError("simulated checkpoint-set replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(trainer.os, "replace", fail_descriptor_replace)
    with pytest.raises(OSError, match="checkpoint-set replace"):
        trainer._atomic_write_json({"committed_epoch": 2}, descriptor)

    assert descriptor.read_bytes() == prior_descriptor
    assert prior_generation.read_bytes() == prior_generation_bytes
    assert not list(path.parent.glob(f".{descriptor.name}.*.tmp"))


def test_checkpoint_generation_cleanup_keeps_referenced_and_removes_orphans(
    tmp_path: Path,
) -> None:
    weights = tmp_path / "weights"
    best = weights / "best.pt"
    last = weights / "last.pt"
    _atomic_torch_save({"role": "best"}, best)
    _atomic_torch_save({"role": "last"}, last)
    referenced = {
        hashlib.sha256(best.read_bytes()).hexdigest(),
        hashlib.sha256(last.read_bytes()).hexdigest(),
    }
    orphan_raw = b"orphan-generation"
    orphan_digest = hashlib.sha256(orphan_raw).hexdigest()
    orphan = weights / "generations" / f"{orphan_digest}.pt"
    orphan.write_bytes(orphan_raw)

    trainer._gc_checkpoint_generations(weights, approved_output_roots=(tmp_path,))

    assert orphan.exists() is False
    assert all(
        (weights / "generations" / f"{digest}.pt").exists() for digest in referenced
    )


@pytest.mark.parametrize("role", ["best", "last"])
def test_new_checkpoint_generation_failure_preserves_authoritative_set(
    tmp_path: Path, monkeypatch, role: str
) -> None:
    _best, output, _data_dir, _config_path = _valid_best_artifacts(tmp_path)
    weights = output / "weights"
    descriptor = weights / trainer.CHECKPOINT_SET_FILENAME
    prior_descriptor = descriptor.read_bytes()
    prior_set = json.loads(prior_descriptor)
    prior_generations = {
        weights / reference["filename"]
        for reference in (prior_set["best"], prior_set["last"])
    }
    payload = torch.load(
        weights / f"{role}.pt", map_location="cpu", weights_only=True
    )
    payload["epoch"] = int(payload.get("epoch", 1)) + 1

    def fail_save(*args, **kwargs):
        raise OSError(f"simulated {role} generation failure")

    monkeypatch.setattr(trainer.torch, "save", fail_save)
    with pytest.raises(OSError, match=f"{role} generation failure"):
        _atomic_torch_save(payload, weights / f"{role}.pt")

    assert descriptor.read_bytes() == prior_descriptor
    assert all(path.exists() for path in prior_generations)


def test_checkpoint_set_transaction_failures_preserve_or_commit_coherent_sets(
    tmp_path: Path, monkeypatch
) -> None:
    best, output, _data_dir, _config_path = _valid_best_artifacts(tmp_path)
    weights = output / "weights"
    descriptor_path = weights / trainer.CHECKPOINT_SET_FILENAME
    prior_descriptor = descriptor_path.read_bytes()
    prior_set = json.loads(prior_descriptor)
    prior_generations = {
        weights / reference["filename"]
        for reference in (prior_set["best"], prior_set["last"])
    }
    run_info = json.loads((output / "run_info.json").read_text(encoding="utf-8"))
    provenance = run_info["provenance"]
    last_path = weights / "last.pt"
    last = torch.load(last_path, map_location="cpu", weights_only=True)
    last["epoch"] = 2
    new_last_digest = _atomic_torch_save(last, last_path)

    real_write_json = trainer._atomic_write_json

    def fail_set_descriptor(payload, path):
        if path == descriptor_path:
            raise OSError("simulated checkpoint-set descriptor failure")
        return real_write_json(payload, path)

    monkeypatch.setattr(trainer, "_atomic_write_json", fail_set_descriptor)
    with pytest.raises(OSError, match="checkpoint-set descriptor"):
        trainer._write_checkpoint_set(
            weights,
            best_digest=prior_set["best"]["sha256"],
            last_digest=new_last_digest,
            provenance=provenance,
            approved_output_roots=(tmp_path,),
        )
    assert descriptor_path.read_bytes() == prior_descriptor
    assert all(path.exists() for path in prior_generations)
    assert (weights / trainer.CHECKPOINT_SET_RECOVERY_FILENAME).is_file()

    monkeypatch.setattr(trainer, "_atomic_write_json", real_write_json)
    committed = trainer._write_checkpoint_set(
        weights,
        best_digest=prior_set["best"]["sha256"],
        last_digest=new_last_digest,
        provenance=provenance,
        approved_output_roots=(tmp_path,),
    )
    assert committed["committed_epoch"] == 2
    assert committed["last"]["sha256"] == new_last_digest

    real_gc = trainer._gc_checkpoint_generations

    def fail_gc(*args, **kwargs):
        raise OSError("simulated checkpoint generation cleanup failure")

    monkeypatch.setattr(trainer, "_gc_checkpoint_generations", fail_gc)
    with pytest.raises(OSError, match="cleanup failure"):
        trainer._complete_checkpoint_set_commit(
            weights, approved_output_roots=(tmp_path,)
        )
    assert descriptor_path.read_bytes() != prior_descriptor
    assert (weights / trainer.CHECKPOINT_SET_RECOVERY_FILENAME).is_file()
    assert all(path.exists() for path in prior_generations)
    monkeypatch.setattr(trainer, "_gc_checkpoint_generations", real_gc)


def test_checkpoint_alias_update_failure_after_generation_rename_preserves_set(
    tmp_path: Path, monkeypatch
) -> None:
    _best, output, _data_dir, _config_path = _valid_best_artifacts(tmp_path)
    weights = output / "weights"
    descriptor = weights / trainer.CHECKPOINT_SET_FILENAME
    prior_descriptor = descriptor.read_bytes()
    prior_set = json.loads(prior_descriptor)
    payload = torch.load(weights / "last.pt", map_location="cpu", weights_only=True)
    payload["epoch"] = 2
    real_replace = trainer.os.replace
    generation_replaced = False

    def fail_alias_update(source, destination):
        nonlocal generation_replaced
        destination = Path(destination)
        if destination.parent.name == "generations":
            generation_replaced = True
            return real_replace(source, destination)
        if generation_replaced and destination.name == "last.pt":
            raise OSError("simulated alias update failure")
        return real_replace(source, destination)

    monkeypatch.setattr(trainer.os, "replace", fail_alias_update)
    with pytest.raises(OSError, match="alias update"):
        _atomic_torch_save(payload, weights / "last.pt")

    assert descriptor.read_bytes() == prior_descriptor
    assert prior_set["last"]["sha256"] in {
        path.stem for path in (weights / "generations").glob("*.pt")
    }


def test_checkpoint_set_recovery_publish_failure_preserves_current_set(
    tmp_path: Path, monkeypatch
) -> None:
    _best, output, _data_dir, _config_path = _valid_best_artifacts(tmp_path)
    weights = output / "weights"
    descriptor = weights / trainer.CHECKPOINT_SET_FILENAME
    prior_descriptor = descriptor.read_bytes()
    prior_set = json.loads(prior_descriptor)
    payload = torch.load(weights / "last.pt", map_location="cpu", weights_only=True)
    payload["epoch"] = 2
    last_digest = _atomic_torch_save(payload, weights / "last.pt")
    provenance = json.loads((output / "run_info.json").read_text())["provenance"]
    real_write_json = trainer._atomic_write_json

    def fail_recovery(payload, path):
        if path.name == trainer.CHECKPOINT_SET_RECOVERY_FILENAME:
            raise OSError("simulated recovery descriptor failure")
        return real_write_json(payload, path)

    monkeypatch.setattr(trainer, "_atomic_write_json", fail_recovery)
    with pytest.raises(OSError, match="recovery descriptor"):
        trainer._write_checkpoint_set(
            weights,
            best_digest=prior_set["best"]["sha256"],
            last_digest=last_digest,
            provenance=provenance,
            approved_output_roots=(tmp_path,),
        )
    assert descriptor.read_bytes() == prior_descriptor
    assert not (weights / trainer.CHECKPOINT_SET_RECOVERY_FILENAME).exists()


def test_checkpoint_set_recovery_removal_failure_keeps_both_descriptors_and_generations(
    tmp_path: Path, monkeypatch
) -> None:
    _best, output, _data_dir, _config_path = _valid_best_artifacts(tmp_path)
    weights = output / "weights"
    descriptor = weights / trainer.CHECKPOINT_SET_FILENAME
    prior_set = json.loads(descriptor.read_text())
    payload = torch.load(weights / "last.pt", map_location="cpu", weights_only=True)
    payload["epoch"] = 2
    last_digest = _atomic_torch_save(payload, weights / "last.pt")
    provenance = json.loads((output / "run_info.json").read_text())["provenance"]
    trainer._write_checkpoint_set(
        weights,
        best_digest=prior_set["best"]["sha256"],
        last_digest=last_digest,
        provenance=provenance,
        approved_output_roots=(tmp_path,),
    )
    recovery = weights / trainer.CHECKPOINT_SET_RECOVERY_FILENAME
    real_unlink = Path.unlink

    def fail_recovery_removal(self, *args, **kwargs):
        if self == recovery:
            raise OSError("simulated recovery descriptor removal failure")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_recovery_removal)
    with pytest.raises(OSError, match="recovery descriptor removal"):
        trainer._complete_checkpoint_set_commit(
            weights, approved_output_roots=(tmp_path,)
        )
    assert descriptor.is_file()
    assert recovery.is_file()
    current = json.loads(descriptor.read_text())
    backup = json.loads(recovery.read_text())
    for value in (current, backup):
        for role in ("best", "last"):
            assert (weights / value[role]["filename"]).is_file()


def test_checkpoint_gc_deletion_failure_keeps_recovery_and_referenced_generations(
    tmp_path: Path, monkeypatch
) -> None:
    _best, output, _data_dir, _config_path = _valid_best_artifacts(tmp_path)
    weights = output / "weights"
    descriptor = weights / trainer.CHECKPOINT_SET_FILENAME
    prior_set = json.loads(descriptor.read_text())
    payload = torch.load(weights / "last.pt", map_location="cpu", weights_only=True)
    payload["epoch"] = 2
    last_digest = _atomic_torch_save(payload, weights / "last.pt")
    provenance = json.loads((output / "run_info.json").read_text())["provenance"]
    trainer._write_checkpoint_set(
        weights,
        best_digest=prior_set["best"]["sha256"],
        last_digest=last_digest,
        provenance=provenance,
        approved_output_roots=(tmp_path,),
    )
    orphan_raw = b"gc-failure-orphan"
    orphan = weights / "generations" / f"{hashlib.sha256(orphan_raw).hexdigest()}.pt"
    orphan.write_bytes(orphan_raw)
    recovery = weights / trainer.CHECKPOINT_SET_RECOVERY_FILENAME
    real_unlink = Path.unlink

    def fail_orphan_deletion(self, *args, **kwargs):
        if self == orphan:
            raise OSError("simulated generation deletion failure")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_orphan_deletion)
    with pytest.raises(OSError, match="generation deletion"):
        trainer._complete_checkpoint_set_commit(
            weights, approved_output_roots=(tmp_path,)
        )
    assert descriptor.is_file()
    assert recovery.is_file()
    current = json.loads(descriptor.read_text())
    backup = json.loads(recovery.read_text())
    for value in (current, backup):
        for role in ("best", "last"):
            assert (weights / value[role]["filename"]).is_file()
    assert orphan.is_file()


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
    import vacca_bcs.serving as serving

    monkeypatch.setattr(trainer, "BCSFolderDataset", _TinyDataset)
    monkeypatch.setattr(trainer, "BCSOrdinalModel", _TinyModel)

    def fake_load_bcs_model(path, device="cpu", **kwargs):
        loaded_bytes = trainer.load_checkpoint_bytes(
            path,
            approved_roots=(Path(path).parent,),
            expected_sha256=kwargs.get("expected_sha256"),
        )
        checkpoint = loaded_bytes.payload
        lineage = serving.BCSLineageMetadata(
            checkpoint_schema_version=checkpoint["checkpoint_schema_version"],
            domain_id=checkpoint["domain_id"],
            snapshot_schema=checkpoint["snapshot_schema"],
            snapshot_identity=checkpoint["snapshot_identity"],
            dataset_manifest_digest=checkpoint["dataset_manifest_digest"],
            run_id=checkpoint["run_id"],
            source_schema=checkpoint["source_schema"],
            source_identity_scheme=checkpoint["source_identity_scheme"],
            source_mapping=tuple(sorted(checkpoint["source_mapping"].items())),
            observed_classes=tuple(checkpoint["observed_classes"]),
            missing_classes=tuple(checkpoint["missing_classes"]),
        )
        restored = _TinyModel()
        restored.load_state_dict(checkpoint["model_state_dict"])
        return serving.LoadedBCSModel(
            restored,
            8,
            torch.device(device),
            lineage,
            loaded_bytes.sha256,
            checkpoint,
        )

    monkeypatch.setattr(serving, "load_bcs_model", fake_load_bcs_model)
    call_count = 0

    def fake_train(model, loader, optimizer, device, **kwargs) -> float:
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
            "within_one": 1.0,
            "ordinal_mae": 0.25,
            "error_ge_2": 0.0,
            "macro_f1": 0.5,
            "balanced_accuracy": 0.5,
            "mae": 0.25,
            "support": [1] * NUM_CLASSES,
            "precision": {name: 0.5 for name in CLASS_NAMES},
            "recall": {name: 0.0 for name in CLASS_NAMES},
            "within_one_by_class": {name: 1.0 for name in CLASS_NAMES},
            "error_ge_2_by_class": {name: 0.0 for name in CLASS_NAMES},
            "f1": {name: 0.0 for name in CLASS_NAMES},
            "confusion_matrix": [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)],
            "total": 1,
        }

    monkeypatch.setattr(trainer, "_train_epoch", fake_train)
    monkeypatch.setattr(trainer, "_validate", fake_validate)


def test_test_evaluates_the_reloaded_selected_best_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "run"
    config = _tiny_training_config(tmp_path, output)
    _install_tiny_training_fakes(monkeypatch, [])

    def distinguishable_train(model, loader, optimizer, device, *, epoch=None, **kwargs):
        with torch.no_grad():
            model.weight.fill_(1.0 if epoch == 1 else -1.0)
        return 0.1

    def model_dependent_validate(model, loader, device):
        good = model.weight.item() > 0
        value = 0.0 if good else 1.0
        return {
            "exact_acc": 1.0 if good else 0.0,
            "within_one": 1.0,
            "mae": value,
            "ordinal_mae": value,
            "error_ge_2": 0.0 if good else 1.0,
            "macro_f1": 1.0 if good else 0.0,
            "balanced_accuracy": 1.0 if good else 0.0,
            "within_one_by_class": {name: 1.0 for name in CLASS_NAMES},
            "error_ge_2_by_class": {name: 0.0 for name in CLASS_NAMES},
            "support": [1] * NUM_CLASSES,
            "precision": {name: 1.0 if good else 0.0 for name in CLASS_NAMES},
            "recall": {name: 1.0 if good else 0.0 for name in CLASS_NAMES},
            "f1": {name: 1.0 if good else 0.0 for name in CLASS_NAMES},
            "confusion_matrix": [[1 if row == col else 0 for col in range(NUM_CLASSES)] for row in range(NUM_CLASSES)],
            "total": NUM_CLASSES,
            "model_weight": model.weight.item(),
        }

    monkeypatch.setattr(trainer, "_train_epoch", distinguishable_train)
    monkeypatch.setattr(trainer, "_validate", model_dependent_validate)
    result = trainer.train(config, overwrite=True)

    assert result["test_metrics"]["model_weight"] == pytest.approx(1.0)
    selected = torch.load(output / "weights" / "best.pt", map_location="cpu", weights_only=True)
    assert selected["model_state_dict"]["weight"].item() == pytest.approx(1.0)


def test_public_resume_rejects_changed_dataset_manifest(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "run"
    config = _tiny_training_config(tmp_path, output)
    _install_tiny_training_fakes(monkeypatch, [], interrupt_at=2)
    with pytest.raises(KeyboardInterrupt):
        trainer.train(config, overwrite=True)
    config["lr"] = 0.02
    with pytest.raises(ValueError, match="category lineage|provenance mismatch"):
        trainer.train(config, resume=output / "weights" / "last.pt")
    config["lr"] = 0.01
    live_file = next((Path(config["data_root"]) / "train" / CLASS_NAMES[0]).iterdir())
    live_file.write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest|hash mismatch"):
        trainer.train(config, resume=output / "weights" / "last.pt")


def test_public_resume_rejects_alias_only_checkpoint(tmp_path: Path) -> None:
    _best, output, _data_dir, config_path = _valid_best_artifacts(tmp_path)
    config = load_config(config_path)
    (output / "weights" / trainer.CHECKPOINT_SET_FILENAME).unlink()

    with pytest.raises(ValueError, match="[Aa]uthoritative checkpoint set"):
        trainer.train(config, resume=output / "weights" / "last.pt")


def test_resume_rejects_stale_run_info_and_legacy_results_lineage(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "run"
    config = _tiny_training_config(tmp_path, output)
    _install_tiny_training_fakes(monkeypatch, [])
    trainer.train(config, overwrite=True)
    resume = output / "weights" / "last.pt"
    run_info_path = output / "run_info.json"
    run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
    run_info["run_id"] = "e" * 32
    run_info_path.write_text(json.dumps(run_info), encoding="utf-8")
    with pytest.raises(ValueError, match="run lineage"):
        trainer.train(config, resume=resume)
    run_info_path.unlink()
    (output / RESULTS_LINEAGE_FILENAME).unlink()
    with pytest.raises(ValueError, match="results lineage"):
        trainer.train(config, resume=resume)


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
    checkpoint = torch.load(
        baseline_output / "weights" / "last.pt", map_location="cpu", weights_only=True
    )
    lineage = json.loads(
        (baseline_output / RESULTS_LINEAGE_FILENAME).read_text(encoding="utf-8")
    )
    assert checkpoint["run_id"] == run_info["run_id"] == lineage["run_id"]

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


@pytest.mark.parametrize("setting", ["val_num_workers", "val_seed", "progress_every_batches"])
def test_resume_excluded_runtime_settings_preserve_training_state(
    tmp_path: Path, monkeypatch, setting: str
) -> None:
    interrupted_output = tmp_path / "interrupted"
    interrupted_config = _tiny_training_config(tmp_path, interrupted_output)
    interrupted_config.update(
        {"val_num_workers": 0, "val_seed": 11, "progress_every_batches": 2}
    )
    interrupted_calls: list[float] = []
    _install_tiny_training_fakes(monkeypatch, interrupted_calls, interrupt_at=2)
    with pytest.raises(KeyboardInterrupt):
        trainer.train(interrupted_config, overwrite=True)

    baseline_output = tmp_path / "baseline"
    baseline_config = _tiny_training_config(tmp_path, baseline_output)
    baseline_config.update(
        {"val_num_workers": 0, "val_seed": 11, "progress_every_batches": 2}
    )
    baseline_calls: list[float] = []
    _install_tiny_training_fakes(monkeypatch, baseline_calls)
    trainer.train(baseline_config, overwrite=True)

    changed_value = {
        "val_num_workers": 2,
        "val_seed": 23,
        "progress_every_batches": 7,
    }[setting]
    interrupted_config[setting] = changed_value
    resumed_calls: list[float] = []
    _install_tiny_training_fakes(monkeypatch, resumed_calls)
    trainer.train(
        interrupted_config,
        resume=interrupted_output / "weights" / "last.pt",
    )

    interrupted_checkpoint = torch.load(
        interrupted_output / "weights" / "last.pt", map_location="cpu", weights_only=True
    )
    baseline_checkpoint = torch.load(
        baseline_output / "weights" / "last.pt", map_location="cpu", weights_only=True
    )
    assert all(
        torch.equal(interrupted_checkpoint["model_state_dict"][key], baseline_checkpoint["model_state_dict"][key])
        for key in baseline_checkpoint["model_state_dict"]
    )
    assert interrupted_checkpoint["optimizer_state_dict"]["param_groups"] == baseline_checkpoint["optimizer_state_dict"]["param_groups"]
    for parameter_id, state in interrupted_checkpoint["optimizer_state_dict"]["state"].items():
        for key, value in state.items():
            expected = baseline_checkpoint["optimizer_state_dict"]["state"][parameter_id][key]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, expected)
            else:
                assert value == expected
    assert interrupted_checkpoint["rng_state"]["python"] == baseline_checkpoint["rng_state"]["python"]
    assert torch.equal(
        interrupted_checkpoint["rng_state"]["torch_cpu"],
        baseline_checkpoint["rng_state"]["torch_cpu"],
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

    def interrupt_before_checkpoint_gc(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        trainer, "_complete_checkpoint_set_commit", interrupt_before_checkpoint_gc
    )
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
    data_dir = Path(config["data_root"])
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
    best_validation = {
        "exact_acc": 0.5,
        "within_one": 1.0,
        "ordinal_mae": 0.25,
        "error_ge_2": 0.0,
        "macro_f1": 0.5,
        "balanced_accuracy": 0.5,
        "precision": {name: 0.0 for name in CLASS_NAMES},
        "recall": {name: 0.0 for name in CLASS_NAMES},
        "f1": {name: 0.0 for name in CLASS_NAMES},
        "within_one_by_class": {name: 1.0 for name in CLASS_NAMES},
        "error_ge_2_by_class": {name: 0.0 for name in CLASS_NAMES},
    }
    checkpoint["best_epoch"] = 1
    checkpoint["best_validation"] = best_validation
    checkpoint["selection_identity"] = trainer._selection_identity(
        provenance, 1, best_validation
    )
    _atomic_torch_save(checkpoint, output / "weights" / "last.pt")
    best_checkpoint = dict(checkpoint)
    best_checkpoint["val_ordinal_mae"] = 0.25
    _atomic_torch_save(best_checkpoint, output / "weights" / "best.pt")
    trainer._write_checkpoint_set(
        output / "weights",
        best_digest=hashlib.sha256((output / "weights" / "best.pt").read_bytes()).hexdigest(),
        last_digest=hashlib.sha256((output / "weights" / "last.pt").read_bytes()).hexdigest(),
        provenance=provenance,
        approved_output_roots=(tmp_path,),
    )
    _write_results_history(output / "results.csv", [1])
    _atomic_write_json(
        trainer._results_lineage(provenance, provenance["run_id"]),
        output / RESULTS_LINEAGE_FILENAME,
    )

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
