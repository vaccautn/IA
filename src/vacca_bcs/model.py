"""CORAL ordinal ResNet model for Body Condition Score prediction."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models import resnet18

from .constants import NUM_CLASSES, NUM_THRESHOLDS, SCORE_MIN, SCORE_STEP


def encode_levels(levels: Tensor, num_classes: int = NUM_CLASSES) -> Tensor:
    """Encode class indices into CORAL cumulative binary targets."""
    if levels.ndim != 1:
        raise ValueError("levels must be a one-dimensional tensor")
    if levels.numel() and (levels.min() < 0 or levels.max() >= num_classes):
        raise ValueError("levels contain an invalid class index")
    thresholds = torch.arange(num_classes - 1, device=levels.device)
    return (levels.to(torch.long).unsqueeze(1) > thresholds).to(torch.float32)


def coral_loss(logits: Tensor, levels: Tensor) -> Tensor:
    """Return summed BCE loss over the CORAL binary ranking tasks."""
    if logits.ndim != 2 or logits.shape[1] != NUM_THRESHOLDS:
        raise ValueError(f"logits must have shape (batch, {NUM_THRESHOLDS})")
    targets = encode_levels(levels.to(logits.device))
    return nn.functional.binary_cross_entropy_with_logits(
        logits, targets.to(dtype=logits.dtype), reduction="sum"
    )


def predict(logits: Tensor) -> tuple[Tensor, Tensor]:
    """Convert CORAL logits to class indices and BCS categories."""
    if logits.ndim != 2 or logits.shape[1] != NUM_THRESHOLDS:
        raise ValueError(f"logits must have shape (batch, {NUM_THRESHOLDS})")
    class_idx = (torch.sigmoid(logits) > 0.5).sum(dim=1).to(torch.long)
    categories = SCORE_MIN + SCORE_STEP * class_idx.to(dtype=logits.dtype)
    return class_idx, categories


class CORALHead(nn.Module):
    """One shared feature weight and monotonically ordered thresholds."""

    def __init__(self, in_features: int, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, in_features))
        self.bias_1 = nn.Parameter(torch.zeros(1))
        self.raw_deltas = nn.Parameter(torch.zeros(max(0, num_classes - 2)))
        nn.init.normal_(self.weight, mean=0.0, std=0.01)

    def ordered_biases(self) -> Tensor:
        """Return b_1 ... b_K-1 with strictly increasing thresholds."""
        increments = nn.functional.softplus(self.raw_deltas)
        if not increments.numel():
            return self.bias_1
        following = self.bias_1 + torch.cumsum(increments, dim=0)
        return torch.cat((self.bias_1, following))

    @property
    def biases(self) -> Tensor:
        """Expose the current ordered thresholds for inspection."""
        return self.ordered_biases()

    def forward(self, features: Tensor) -> Tensor:
        # Increasing thresholds become decreasing logits for y_k = 1[c > k].
        shared_score = features @ self.weight.t()
        return shared_score - self.ordered_biases().unsqueeze(0)


class BCSOrdinalModel(nn.Module):
    """ImageNet-pretrained ResNet18 with a rank-consistent CORAL head."""

    def __init__(self, *, pretrained: bool = True) -> None:
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = CORALHead(in_features, NUM_CLASSES)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.backbone(x))

    def predict(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Predict from images, or directly from logits shaped (B, 4)."""
        logits = x if x.ndim == 2 and x.shape[1] == NUM_THRESHOLDS else self(x)
        return predict(logits)
