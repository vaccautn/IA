"""Folder-based dataset and image transforms for ordinal BCS training."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from .constants import CLASS_NAMES, IMAGE_EXTENSIONS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Letterbox:
    """Resize a PIL image into a square and pad unused pixels with gray 114."""

    def __init__(self, size: int = 224) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self.size = int(size)

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")

        scale = min(self.size / width, self.size / height)
        resized_size = (
            max(1, round(width * scale)),
            max(1, round(height * scale)),
        )
        resized = image.resize(resized_size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (self.size, self.size), (114, 114, 114))
        left = (self.size - resized.width) // 2
        top = (self.size - resized.height) // 2
        canvas.paste(resized, (left, top))
        return canvas


def build_transforms(imgsz: int, train: bool) -> transforms.Compose:
    """Build the training or deterministic validation transform pipeline."""
    operations: list[object] = [Letterbox(imgsz)]
    if train:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(operations)


class BCSFolderDataset(Dataset[tuple[Tensor, int]]):
    """Load BCS images from ``root/<class-name>/*.jpg`` folders."""

    def __init__(
        self,
        root: str | Path,
        *,
        train: bool = False,
        imgsz: int = 224,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.root = Path(root)
        self.classes = list(CLASS_NAMES)
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.transform = transform or build_transforms(imgsz, train)
        self.samples: list[tuple[Path, int]] = []

        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")

        for class_name in self.classes:
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Class directory not found: {class_dir}")
            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, self.class_to_idx[class_name]))

        if not self.samples:
            raise ValueError(f"No images found under {self.root}")

    @property
    def class_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in self.classes}
        for _, class_idx in self.samples:
            counts[self.classes[class_idx]] += 1
        return counts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image_path, class_idx = self.samples[index]
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        return self.transform(image), class_idx
