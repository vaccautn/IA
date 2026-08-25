from __future__ import annotations

from pathlib import Path
import sys
import pytest
import torch
from PIL import Image

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
