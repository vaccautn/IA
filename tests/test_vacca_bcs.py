from __future__ import annotations

from pathlib import Path
import sys
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from vacca_bcs.constants import CLASS_NAMES  # noqa: E402
from vacca_bcs.model import (  # noqa: E402
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
