from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import train as yolo_training


VALID_DESCRIPTOR = (
    "path: data/fixture\n"
    "train: train/images\n"
    "val: val/images\n"
    "test: test/images\n"
    "nc: 1\n"
    "names: [cow]\n"
)


def write_training_project(tmp_path, descriptor=VALID_DESCRIPTOR):
    (tmp_path / "configs").mkdir()
    dataset_root = tmp_path / "data" / "fixture"
    for split in ("train", "val", "test"):
        (dataset_root / split / "images").mkdir(parents=True)
    data_path = dataset_root / "data.yaml"
    data_path.write_text(descriptor, encoding="utf-8")
    config_path = tmp_path / "configs" / "training.yaml"
    config_path.write_text(
        "data: data/fixture/data.yaml\n"
        "model:\n  pretrained: ''\n"
        "training:\n  device: cpu\n"
        "output: {}\n",
        encoding="utf-8",
    )
    return config_path, data_path, dataset_root


def install_fake_ultralytics(monkeypatch, tmp_path, *, model_task="detect"):
    save_dir = tmp_path / "outputs" / "fake-run"
    dataset_loader_calls = []

    class FakeDetectionTrainer:
        def __init__(self):
            self.args = SimpleNamespace(data=None, task="detect")

        def get_dataset(self):
            dataset_loader_calls.append(self.args.data)
            return self.args.data

    def fake_train(**kwargs):
        trainer_class = kwargs.get("trainer", FakeDetectionTrainer)
        trainer = trainer_class()
        trainer.args.data = kwargs["data"]
        trainer.get_dataset()
        save_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(save_dir=save_dir)

    model = SimpleNamespace(
        train=Mock(side_effect=fake_train),
        export=Mock(),
        dataset_loader_calls=dataset_loader_calls,
        dataset_trainer_class=FakeDetectionTrainer,
        task=model_task,
    )
    yolo_factory = Mock(return_value=model)
    module = ModuleType("ultralytics")
    module.__path__ = []
    module.YOLO = yolo_factory
    models = ModuleType("ultralytics.models")
    models.__path__ = []
    yolo = ModuleType("ultralytics.models.yolo")
    yolo.__path__ = []
    detect = ModuleType("ultralytics.models.yolo.detect")
    detect.DetectionTrainer = FakeDetectionTrainer
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    monkeypatch.setitem(sys.modules, "ultralytics.models", models)
    monkeypatch.setitem(sys.modules, "ultralytics.models.yolo", yolo)
    monkeypatch.setitem(sys.modules, "ultralytics.models.yolo.detect", detect)
    return yolo_factory, model


@pytest.mark.parametrize(
    "descriptor_suffix,resume",
    [
        ('download: "python -c print_fixture"\n', False),
        ('roboflow:\n  workspace: fixture\n  download: "python -c print_fixture"\n', False),
        ('download: "python -c print_fixture"\n', True),
    ],
)
def test_dataset_download_hooks_are_rejected_before_yolo(
    tmp_path, monkeypatch, descriptor_suffix, resume
):
    descriptor = VALID_DESCRIPTOR + descriptor_suffix
    config_path, data_path, _ = write_training_project(tmp_path, descriptor)
    monkeypatch.setattr(yolo_training, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    yolo_factory, model = install_fake_ultralytics(monkeypatch, tmp_path)

    args = ["--config", str(config_path)]
    if resume:
        args.extend(("--resume", str(tmp_path / "fake-resume.pt")))
    status = yolo_training.main(args)

    assert status == 1
    yolo_factory.assert_not_called()
    model.train.assert_not_called()


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_missing_split_is_rejected_before_yolo(tmp_path, monkeypatch, split):
    descriptor = VALID_DESCRIPTOR.replace(f"{split}: {split}/images", f"{split}: missing/{split}")
    config_path, _, _ = write_training_project(tmp_path, descriptor)
    monkeypatch.setattr(yolo_training, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    yolo_factory, model = install_fake_ultralytics(monkeypatch, tmp_path)

    status = yolo_training.main(["--config", str(config_path)])

    assert status == 1
    yolo_factory.assert_not_called()
    model.train.assert_not_called()


@pytest.mark.parametrize("invalid_descriptor", ["remote", "zip", "split_type", "class_type", "root_type"])
def test_nonlocal_or_malformed_dataset_fields_fail_closed(
    tmp_path, monkeypatch, invalid_descriptor
):
    config_path, data_path, dataset_root = write_training_project(tmp_path)
    descriptor = VALID_DESCRIPTOR
    if invalid_descriptor == "remote":
        descriptor = descriptor.replace("train: train/images", "train: https://example.invalid/train")
    elif invalid_descriptor == "zip":
        (dataset_root / "dataset.zip").write_bytes(b"synthetic archive")
        descriptor = descriptor.replace("train: train/images", "train: dataset.zip")
    elif invalid_descriptor == "split_type":
        descriptor = descriptor.replace("train: train/images", "train: {path: train/images}")
    elif invalid_descriptor == "class_type":
        descriptor = descriptor.replace("nc: 1", "nc: true")
    else:
        descriptor = "- train\n- val\n"

    data_path.write_text(descriptor, encoding="utf-8")
    monkeypatch.setattr(yolo_training, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    yolo_factory, model = install_fake_ultralytics(monkeypatch, tmp_path)

    assert yolo_training.main(["--config", str(config_path)]) == 1
    yolo_factory.assert_not_called()
    model.train.assert_not_called()


def test_null_nc_is_rejected_before_yolo(tmp_path, monkeypatch):
    config_path, data_path, _ = write_training_project(
        tmp_path, VALID_DESCRIPTOR.replace("nc: 1", "nc: null")
    )
    monkeypatch.setattr(yolo_training, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    yolo_factory, model = install_fake_ultralytics(monkeypatch, tmp_path)

    assert yolo_training.main(["--config", str(config_path)]) == 1
    yolo_factory.assert_not_called()
    model.train.assert_not_called()


@pytest.mark.parametrize("record", [" ./images/cow.jpg", ".\\images\\cow.jpg"])
def test_manifest_validation_matches_ultralytics_record_resolution(
    tmp_path, monkeypatch, record
):
    config_path, data_path, dataset_root = write_training_project(tmp_path)
    (dataset_root / "val" / "images" / "cow.jpg").write_bytes(b"fixture image")
    (dataset_root / "val" / "images" / "keep.jpg").write_bytes(b"other fixture")
    manifest_records = (
        f"./images/keep.jpg\n{record}\n"
        if record.startswith(" ")
        else f"{record}\n"
    )
    (dataset_root / "val" / "val.txt").write_text(manifest_records, encoding="utf-8")
    data_path.write_text(
        "path: data/fixture\n"
        "train: train/images\n"
        "val: val/val.txt\n"
        "test: test/images\n"
        "nc: 1\n"
        "names: [cow]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(yolo_training, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    yolo_factory, model = install_fake_ultralytics(monkeypatch, tmp_path)

    assert yolo_training.main(["--config", str(config_path)]) == 1
    yolo_factory.assert_not_called()
    model.train.assert_not_called()


def test_explicit_relative_dataset_root_does_not_fallback_to_yaml_parent(
    tmp_path, monkeypatch
):
    config_path, data_path, dataset_root = write_training_project(tmp_path)
    yaml_relative_root = dataset_root / "descriptor-relative-root"
    for split in ("train", "val", "test"):
        (yaml_relative_root / split / "images").mkdir(parents=True)
    data_path.write_text(
        "path: descriptor-relative-root\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "nc: 1\n"
        "names: [cow]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(yolo_training, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    yolo_factory, model = install_fake_ultralytics(monkeypatch, tmp_path)

    assert yolo_training.main(["--config", str(config_path)]) == 1
    yolo_factory.assert_not_called()
    model.train.assert_not_called()


def test_unknown_dataset_hook_and_unsafe_yaml_tag_are_rejected_before_yolo(
    tmp_path, monkeypatch
):
    descriptor = VALID_DESCRIPTOR + "callbacks: [fixture]\n"
    config_path, data_path, _ = write_training_project(tmp_path, descriptor)
    monkeypatch.setattr(yolo_training, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    yolo_factory, model = install_fake_ultralytics(monkeypatch, tmp_path)

    status = yolo_training.main(["--config", str(config_path)])

    assert status == 1
    yolo_factory.assert_not_called()
    model.train.assert_not_called()

    data_path.write_text(VALID_DESCRIPTOR + "callbacks: !unsafe fixture\n", encoding="utf-8")
    assert yolo_training.main(["--config", str(config_path)]) == 1
    yolo_factory.assert_not_called()
    model.train.assert_not_called()


def test_local_descriptor_accepts_split_lists_manifests_and_safe_metadata(
    tmp_path, monkeypatch
):
    config_path, data_path, dataset_root = write_training_project(tmp_path)
    (dataset_root / "train-extra" / "images").mkdir(parents=True)
    (dataset_root / "val" / "val.txt").write_text(
        "./images/relative.jpg\n", encoding="utf-8"
    )
    (dataset_root / "val" / "images" / "relative.jpg").write_bytes(b"fixture")
    monkeypatch.setattr(yolo_training, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    data_path.write_text(
        "path: data/fixture\n"
        "train: [train/images, train-extra/images]\n"
        "val: val/val.txt\n"
        "test: test/images\n"
        "nc: 1\n"
        "names: {0: cow}\n"
        "channels: 3\n"
        "roboflow: {workspace: fixture, project: fixture, version: 1, license: CC BY 4.0}\n",
        encoding="utf-8",
    )

    config_text = yolo_training.yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validated = yolo_training._validate_config(config_text, config_path)

    assert validated["_data_path"] == data_path
    assert yolo_training._validate_dataset_descriptor(data_path) == dataset_root.resolve()


def test_local_descriptor_without_root_uses_yaml_parent_and_validation_alias(
    tmp_path, monkeypatch
):
    config_path, data_path, dataset_root = write_training_project(tmp_path)
    monkeypatch.setattr(yolo_training, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path / "configs")
    data_path.write_text(
        "train: train/images\n"
        "validation: val/images\n"
        "names: [cow]\n",
        encoding="utf-8",
    )

    assert yolo_training._validate_dataset_descriptor(data_path) == dataset_root.resolve()


def test_valid_local_descriptor_accepts_split_lists_manifest_and_metadata(
    tmp_path, monkeypatch
):
    config_path, data_path, dataset_root = write_training_project(tmp_path)
    (dataset_root / "train-extra" / "images").mkdir(parents=True)
    (dataset_root / "test-extra" / "images").mkdir(parents=True)
    (dataset_root / "val" / "images" / "relative.jpg").write_bytes(b"fixture image")
    shared_image = tmp_path / "shared" / "absolute.jpg"
    shared_image.parent.mkdir()
    shared_image.write_bytes(b"another fixture image")
    (dataset_root / "val" / "val.txt").write_text(
        f"./images/relative.jpg\n{shared_image.as_posix()}\n", encoding="utf-8"
    )
    monkeypatch.setattr(yolo_training, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    data_path.write_text(
        "path: data/fixture\n"
        "train: [train/images, train-extra/images]\n"
        "val: val/val.txt\n"
        "test: [test/images, test-extra/images]\n"
        "nc: 1\n"
        "names: {0: cow}\n"
        "channels: 3\n"
        "roboflow: {workspace: fixture, project: fixture, version: 1, license: CC BY 4.0}\n",
        encoding="utf-8",
    )
    config = yolo_training.yaml.safe_load(config_path.read_text(encoding="utf-8"))

    validated = yolo_training._validate_config(config, config_path)

    assert validated["_data_path"] == data_path
    assert yolo_training._validate_dataset_descriptor(data_path) == dataset_root.resolve()


def test_valid_local_descriptor_reaches_fake_trainer_without_path_rewrite(
    tmp_path, monkeypatch
):
    config_path, data_path, _ = write_training_project(tmp_path)
    monkeypatch.setattr(yolo_training, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    yolo_factory, model = install_fake_ultralytics(monkeypatch, tmp_path)

    status = yolo_training.main(["--config", str(config_path)])

    assert status == 0
    yolo_factory.assert_called_once_with("")
    model.train.assert_called_once()
    assert model.train.call_args.kwargs["data"] == str(data_path)


@pytest.mark.parametrize("restored_descriptor_is_invalid", [True, False])
def test_resume_checkpoint_data_is_validated_before_dataset_loader(
    tmp_path, monkeypatch, restored_descriptor_is_invalid
):
    config_path, _, _ = write_training_project(tmp_path)
    restored_root = tmp_path / "restored-dataset"
    (restored_root / "train" / "images").mkdir(parents=True)
    (restored_root / "val" / "images").mkdir(parents=True)
    restored_data_path = tmp_path / "restored-data.yaml"
    restored_yaml = (
        f"path: {restored_root.as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "nc: 1\n"
        "names: [cow]\n"
    )
    if restored_descriptor_is_invalid:
        restored_yaml += 'download: "python -c print_fixture"\n'
    restored_data_path.write_text(restored_yaml, encoding="utf-8")

    monkeypatch.setattr(yolo_training, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    yolo_factory, model = install_fake_ultralytics(monkeypatch, tmp_path)

    def resume_with_checkpoint_metadata(**kwargs):
        assert kwargs["resume"] is True
        trainer_class = kwargs.get("trainer", model.dataset_trainer_class)
        trainer = trainer_class()
        trainer.args.data = str(restored_data_path)
        return trainer.get_dataset()

    model.train.side_effect = resume_with_checkpoint_metadata
    resume_path = tmp_path / "trusted-fake-last.pt"
    status = yolo_training.main(
        ["--config", str(config_path), "--resume", str(resume_path)]
    )

    yolo_factory.assert_called_once_with(str(resume_path))
    model.train.assert_called_once()
    assert model.train.call_args.kwargs["resume"] is True
    if restored_descriptor_is_invalid:
        assert status == 1
        assert model.dataset_loader_calls == []
    else:
        assert status == 0
        assert model.dataset_loader_calls == [str(restored_data_path)]


@pytest.mark.parametrize("model_task", ["segment", "classify", "pose", None])
def test_non_detection_model_task_is_rejected_before_detection_trainer(
    tmp_path, monkeypatch, model_task
):
    config_path, _, _ = write_training_project(tmp_path)
    monkeypatch.setattr(yolo_training, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    yolo_factory, model = install_fake_ultralytics(
        monkeypatch, tmp_path, model_task=model_task
    )

    assert yolo_training.main(["--config", str(config_path)]) == 1
    yolo_factory.assert_called_once_with("")
    model.train.assert_not_called()
    assert model.dataset_loader_calls == []


@pytest.mark.parametrize("restored_task", ["segment", "classify", "pose", None])
def test_resume_rejects_non_detection_restored_task_before_loader(
    tmp_path, monkeypatch, restored_task
):
    config_path, _, _ = write_training_project(tmp_path)
    restored_root = tmp_path / "restored-dataset"
    (restored_root / "train" / "images").mkdir(parents=True)
    (restored_root / "val" / "images").mkdir(parents=True)
    restored_data_path = tmp_path / "restored-data.yaml"
    restored_data_path.write_text(
        f"path: {restored_root.as_posix()}\n"
        "train: train/images\nval: val/images\nnc: 1\nnames: [cow]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(yolo_training, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    yolo_factory, model = install_fake_ultralytics(monkeypatch, tmp_path)

    def resume_with_task(**kwargs):
        assert kwargs["resume"] is True
        trainer = kwargs["trainer"]()
        trainer.args.data = str(restored_data_path)
        trainer.args.task = restored_task
        return trainer.get_dataset()

    model.train.side_effect = resume_with_task
    resume_path = tmp_path / "trusted-fake-last.pt"

    assert yolo_training.main(
        ["--config", str(config_path), "--resume", str(resume_path)]
    ) == 1
    yolo_factory.assert_called_once_with(str(resume_path))
    model.train.assert_called_once()
    assert model.dataset_loader_calls == []


def test_direct_train_entrypoint_revalidates_before_yolo_construction(tmp_path, monkeypatch):
    _, data_path, _ = write_training_project(
        tmp_path,
        VALID_DESCRIPTOR + 'download: "python -c print_fixture"\n',
    )
    yolo_factory, model = install_fake_ultralytics(monkeypatch, tmp_path)
    config = {
        "_data_path": data_path,
        "_pretrained_path": "fixture-weights",
        "model": {"pretrained": "fixture-weights"},
        "training": {},
        "output": {},
    }

    with pytest.raises(ValueError, match="download hooks"):
        yolo_training.train(config)

    yolo_factory.assert_not_called()
    model.train.assert_not_called()
