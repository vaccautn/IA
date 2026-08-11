from __future__ import annotations

from .contracts import BoundingBox, Detection, ImageMetadata


SUPPORTED_BOVINE_CLASS_NAMES = frozenset({"bovine", "cow"})


def bovine_detections(detections: tuple[Detection, ...]) -> tuple[Detection, ...]:
    return tuple(
        item
        for item in detections
        if item.class_name.strip().casefold() in SUPPORTED_BOVINE_CLASS_NAMES
    )


def relative_area(bbox: BoundingBox, image: ImageMetadata) -> float:
    return bbox.area / image.area


def has_sufficient_framing(
    bbox: BoundingBox,
    image: ImageMetadata,
    margin_ratio: float,
) -> bool:
    horizontal_margin = image.width * margin_ratio
    vertical_margin = image.height * margin_ratio
    return (
        bbox.x_min >= horizontal_margin
        and bbox.y_min >= vertical_margin
        and bbox.x_max <= image.width - horizontal_margin
        and bbox.y_max <= image.height - vertical_margin
    )
