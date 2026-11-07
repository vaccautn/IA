"""Tracking de vacas y exportación de artefactos por trayectoria.

La identidad producida por este módulo es temporal y sólo es válida dentro de
un video. No se intenta convertir un ID de tracker en una identidad bovina.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "vacca-video-tracking-v1"
MIN_OBSERVATIONS = 5
CONTAMINATION_REVIEW_THRESHOLD = 0.05
# A best box under 1% of the frame is a distant animal: too few pixels for review.
SMALL_BOX_FRACTION = 0.01
# Handoffs measured on the manga video: same animal re-born after occlusion had
# IoU 0.38 / centre distance 0.17 diagonals; a different animal had 0.17 / 0.27.
FRAGMENT_MIN_IOU = 0.3
FRAGMENT_MAX_CENTRE_DISTANCE = 0.25
DEFAULT_MAX_GAP_SECONDS = 0.5
DEFAULT_CROP_MARGIN = 0.10
# Numerical tolerance only, not policy hysteresis: two float32 scores in [0, 1]
# each round by at most 2**-25, so their difference can err by at most 2**-24.
# Applied only to the margin threshold; strict positive dominance/ties stay exact.
PASSAGE_MARGIN_EPSILON = 2**-24
# Best-frame segmentation (box + one point prompt). On the manga video a usable
# cow mask covered 31-60% of its box and stayed inside it; outside that range
# the mask usually grabbed ground or fence, so the rectangular crop is kept.
SEGMENT_MIN_FILL = 0.15
SEGMENT_MAX_FILL = 0.95
SEGMENT_MAX_OUTSIDE = 0.10
SEGMENT_POINT_HEIGHT = 0.4
DEFAULT_SEGMENT_LIMIT = 40
# mp4v is the archival default; browsers only play the VP8 WebM variant.
CLIP_CODECS = {"mp4": "mp4v", "webm": "VP80"}


@dataclass
class TrackStats:
    track_id: int
    observations: int = 0
    first_frame: int | None = None
    last_frame: int | None = None
    first_timestamp_ms: float | None = None
    last_timestamp_ms: float | None = None
    last_sample: int | None = None
    segments: list[dict] = field(default_factory=list)
    gaps: list[dict] = field(default_factory=list)
    clip_frames: int = 0
    placeholder_frames: int = 0
    best_score: float = -1.0
    best_frame: object | None = None
    best_frame_index: int | None = None
    best_bbox: tuple[int, int, int, int] | None = None
    best_timestamp_ms: float | None = None
    best_confidence: float | None = None
    best_contamination: float = 0.0
    best_iou: float = 0.0
    best_others: list = field(default_factory=list)
    max_width: int = 0
    max_height: int = 0
    first_bbox: tuple[int, int, int, int] | None = None
    last_bbox: tuple[int, int, int, int] | None = None
    possible_continuation_of: int | None = None
    possible_continued_by: int | None = None


def clamp_box(box: Any, width: int, height: int) -> tuple[int, int, int, int]:
    if width < 2 or height < 2:
        raise ValueError("La imagen debe tener al menos 2x2 píxeles")
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def iou(a: Any, b: Any) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if not inter:
        return 0.0
    area_a = max(1, ax2 - ax1) * max(1, ay2 - ay1)
    area_b = max(1, bx2 - bx1) * max(1, by2 - by1)
    return inter / (area_a + area_b - inter)


def padded_region(bbox: Any, width: int, height: int,
                  margin: float = DEFAULT_CROP_MARGIN) -> tuple[int, int, int, int]:
    """Region that the exported crop actually covers, padding included."""
    if not 0 <= margin <= 1:
        raise ValueError("margin debe estar entre 0 y 1")
    x1, y1, x2, y2 = clamp_box(bbox, width, height)
    pad_x, pad_y = round((x2 - x1) * margin), round((y2 - y1) * margin)
    return max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)


def crop_contamination(bbox: Any, others: list, width: int, height: int,
                       margin: float = DEFAULT_CROP_MARGIN) -> float:
    """Fraction of the padded crop covered by the union of other detections.

    Unlike IoU, a small neighbour fully inside a large crop is not diluted by
    the size of the neighbour. Animals the detector missed cannot be measured.
    """
    import numpy as np

    rx1, ry1, rx2, ry2 = padded_region(bbox, width, height, margin)
    mask = np.zeros((ry2 - ry1, rx2 - rx1), dtype=bool)
    for other in others:
        ox1, oy1, ox2, oy2 = clamp_box(other, width, height)
        ix1, iy1 = max(rx1, ox1), max(ry1, oy1)
        ix2, iy2 = min(rx2, ox2), min(ry2, oy2)
        if ix2 > ix1 and iy2 > iy1:
            mask[iy1 - ry1:iy2 - ry1, ix1 - rx1:ix2 - rx1] = True
    return float(mask.mean())


def touches_border(bbox: Any, width: int, height: int) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 == 0 or y1 == 0 or x2 == width or y2 == height


def _validate_output_size(output_size) -> tuple[int, int]:
    if len(output_size) != 2 or any(int(value) < 2 for value in output_size):
        raise ValueError("output_size debe contener dos dimensiones positivas")
    return int(output_size[0]), int(output_size[1])


def crop_with_padding(image, bbox, output_size=(320, 320), margin=DEFAULT_CROP_MARGIN,
                      *, binary_mask=False):
    """Shared geometry; label maps use nearest sampling and zero letterboxing."""
    import cv2
    import numpy as np

    target_w, target_h = _validate_output_size(output_size)
    height, width = image.shape[:2]
    x1, y1, x2, y2 = padded_region(bbox, width, height, margin)
    crop = image[y1:y2, x1:x2]
    scale = min(target_w / crop.shape[1], target_h / crop.shape[0])
    if binary_mask:
        crop = (crop != 0).astype(np.uint8) * 255
    resized = cv2.resize(crop, (max(1, min(target_w, round(crop.shape[1] * scale))),
                                max(1, min(target_h, round(crop.shape[0] * scale)))),
                         interpolation=cv2.INTER_NEAREST if binary_mask else cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w) + image.shape[2:],
                     0 if binary_mask else 28, dtype=np.uint8)
    top = (target_h - resized.shape[0]) // 2
    left = (target_w - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas


def gap_placeholder(output_size=(320, 320)):
    """Labelled empty frame that keeps clip time aligned with the source video."""
    import cv2
    import numpy as np

    target_w, target_h = _validate_output_size(output_size)
    canvas = np.full((target_h, target_w, 3), 28, dtype=np.uint8)
    text = "sin deteccion"
    scale = max(0.3, min(target_w, target_h) / 400)
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    origin = (max(0, (target_w - text_w) // 2), (target_h + text_h) // 2)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (150, 150, 150), 1, cv2.LINE_AA)
    return canvas


def frame_quality(image, bbox, confidence: float, contamination: float) -> float:
    """Score sharpness and visibility without allowing confidence to dominate."""
    import cv2

    height, width = image.shape[:2]
    x1, y1, x2, y2 = clamp_box(bbox, width, height)
    region = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    sharpness = min(1.0, math.log1p(float(cv2.Laplacian(gray, cv2.CV_64F).var())) / 10.0)
    area = ((x2 - x1) * (y2 - y1)) / (width * height)
    size_score = min(1.0, math.sqrt(max(0.0, area) / 0.35))
    border_penalty = 0.15 if touches_border((x1, y1, x2, y2), width, height) else 0.0
    return (0.42 * sharpness + 0.28 * size_score + 0.20 * confidence
            + 0.10 * (1.0 - contamination) - border_penalty)


def track_warnings(item: TrackStats, width: int, height: int) -> list[str]:
    warnings = []
    if item.observations < MIN_OBSERVATIONS:
        warnings.append("short_track_review_required")
    if len(item.segments) > 1:
        warnings.append("long_gap_review_required")
    if item.possible_continuation_of is not None:
        warnings.append("possible_fragment_review_required")
    if item.best_contamination > CONTAMINATION_REVIEW_THRESHOLD:
        warnings.append("other_detection_in_crop_at_best_frame")
    if item.best_bbox is not None and touches_border(item.best_bbox, width, height):
        warnings.append("border_touch_at_best_frame")
    if item.best_bbox is not None:
        x1, y1, x2, y2 = item.best_bbox
        if (x2 - x1) * (y2 - y1) < SMALL_BOX_FRACTION * width * height:
            warnings.append("small_at_best_frame")
    return warnings


def assess_mask(mask, bbox, others) -> dict:
    """Accept a best-frame mask only when it plausibly is the boxed animal."""
    import numpy as np

    x1, y1, x2, y2 = bbox
    area = int(mask.sum())
    if area == 0:
        return {"status": "rejected", "reason": "empty_mask"}
    inside = int(mask[y1:y2, x1:x2].sum())
    fill = inside / ((x2 - x1) * (y2 - y1))
    outside = 1.0 - inside / area
    covered = np.zeros_like(mask)
    for ox1, oy1, ox2, oy2 in others:
        covered[oy1:oy2, ox1:ox2] = True
    result = {"mask_fill": round(fill, 6), "mask_outside_box": round(outside, 6),
              # Informational: overlapping boxes also share legitimate pixels.
              "mask_other_overlap": round(int((mask & covered).sum()) / area, 6)}
    if fill < SEGMENT_MIN_FILL:
        return dict(result, status="rejected", reason="mask_too_small")
    if fill > SEGMENT_MAX_FILL:
        return dict(result, status="rejected", reason="mask_fills_box")
    if outside > SEGMENT_MAX_OUTSIDE:
        return dict(result, status="rejected", reason="mask_leaves_box")
    return dict(result, status="ok")


def masked_image(image, mask, background: int = 28):
    """Keep only the animal's pixels; everything else becomes a flat background."""
    result = image.copy()
    result[~mask] = background
    return result


def _write_image(target, image):
    """Publish an image atomically; persistence errors must abort the run."""
    import cv2

    temporary = target.with_name(f".{target.stem}.tmp{target.suffix}")
    try:
        if not cv2.imwrite(str(temporary), image):
            raise OSError("Could not write tracking artifact")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _segmentation_artifacts(output, relative, image, bbox, others, crop,
                            output_size, margin, segmenter):
    """Optional model failure is recoverable; artifact storage failure is not."""
    import numpy as np

    if segmenter is None:
        return {"status": "not_requested"}
    try:
        mask = np.asarray(segmenter(image, bbox))
        if mask.ndim != 2 or mask.shape != image.shape[:2]:
            return {"status": "rejected", "reason": "invalid_mask_shape"}
        if mask.dtype.kind not in "buif" or not np.isfinite(mask).all():
            return {"status": "rejected", "reason": "invalid_mask_values"}
        # Binary labels only: bool, numeric 0/1, or numeric 0/255. The local
        # adapter returns bool; probabilities and mixed encodings are unsupported.
        zero = mask == 0
        if not (np.all(zero | (mask == 1)) or np.all(zero | (mask == 255))):
            return {"status": "rejected", "reason": "invalid_mask_values"}
        mask = mask.astype(bool)
        segmentation = assess_mask(mask, bbox, others)
    except Exception as exc:
        return {"status": "failed", "reason": type(exc).__name__}
    if segmentation["status"] == "ok":
        mask_crop = crop_with_padding(mask, bbox, output_size, margin, binary_mask=True)
        # Mask AFTER RGB resizing: retain antialiasing but no background bleed.
        masked = masked_image(crop, mask_crop != 0)
        _write_image(output / relative / "best-masked.png", masked)
        _write_image(output / relative / "best-mask.png", mask_crop)
        segmentation.update(best_masked=(relative / "best-masked.png").as_posix(),
                            best_mask=(relative / "best-mask.png").as_posix())
    return segmentation


def load_sam_segmenter(model_path):
    """Box+point MobileSAM/SAM prompt from local weights; never downloads."""
    import cv2
    import numpy as np

    model_path = Path(model_path)
    if not model_path.is_file():
        raise ValueError("Falta el modelo de segmentación local; no se descarga automáticamente")
    from ultralytics import SAM

    sam = SAM(str(model_path))

    def segment(image, bbox):
        x1, y1, x2, y2 = bbox
        point = [(x1 + x2) // 2, int(y1 + SEGMENT_POINT_HEIGHT * (y2 - y1))]
        result = sam(image, bboxes=[list(bbox)], points=[point], labels=[1], verbose=False, device="cpu")[0]
        if result.masks is None or len(result.masks.data) == 0:
            return np.zeros(image.shape[:2], dtype=bool)
        mask = result.masks.data[0].cpu().numpy().astype(np.uint8)
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask.astype(bool)

    segment.metadata = {"model": model_path.name, "model_sha256": _file_sha256(model_path),
                        "prompt": f"box+point@{SEGMENT_POINT_HEIGHT:g}"}
    return segment


def link_fragments(stats: dict[int, TrackStats], max_gap_ms: float) -> None:
    """Mark tracks that likely continue an earlier one after a short loss.

    Tracks are never merged: a link only asks for review, because two animals
    can also hand over the same position in a chute.
    """
    pairs = []
    for later in stats.values():
        for earlier in stats.values():
            if earlier is later or earlier.last_frame >= later.first_frame:
                continue
            if later.first_timestamp_ms - earlier.last_timestamp_ms > max_gap_ms:
                continue
            a, b = earlier.last_bbox, later.first_bbox
            overlap = iou(a, b)
            diagonal = math.hypot(a[2] - a[0], a[3] - a[1])
            distance = math.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2)
            if overlap >= FRAGMENT_MIN_IOU and distance <= FRAGMENT_MAX_CENTRE_DISTANCE * diagonal:
                pairs.append((overlap, earlier, later))
    for _, earlier, later in sorted(pairs, key=lambda pair: -pair[0]):
        if earlier.possible_continued_by is None and later.possible_continuation_of is None:
            earlier.possible_continued_by = later.track_id
            later.possible_continuation_of = earlier.track_id


def _result_detections(result, image) -> tuple[list[dict], list[tuple[int, int, int, int]]]:
    """Return tracked cows plus every cow box, tracked or not, for contamination."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return [], []
    names = result.names
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
    tracked, all_boxes = [], []
    for index, box in enumerate(xyxy):
        label = names.get(int(classes[index]), str(classes[index])) if isinstance(names, dict) else str(classes[index])
        if label.lower() != "cow":
            continue
        bbox = clamp_box(box, image.shape[1], image.shape[0])
        all_boxes.append(bbox)
        if ids is None:
            continue
        tracked.append({"track_id": int(ids[index]), "bbox": bbox, "box_index": len(all_boxes) - 1,
                        "confidence": float(confs[index]), "class_name": label})
    return tracked, all_boxes


def _file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@dataclass
class PassageEvidence:
    """Confidence/foreground candidates, never unique animal identities.

    Defaults are uncalibrated heuristics. Intervals are half-open at the
    confirmation sample; the last interval ends inclusively at the last sample.
    Pending state and warning categories are bounded, not a per-frame log.
    """

    confirmation_count: int = 3
    confidence_margin: float = 0.1
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS
    recovery_area_ratio: float = 1.5
    recovery_bottom_fraction: float = 0.10
    intervals: list[dict] = field(default_factory=list)
    retired: set[int] = field(default_factory=set)
    warnings: set[str] = field(default_factory=set)
    pending: dict | None = None
    last: tuple | None = None
    active_missing: bool = False
    reference: dict | None = None

    def __post_init__(self):
        if type(self.confirmation_count) is not int or self.confirmation_count < 1:
            raise ValueError("passage_confirmation_count must be a positive integer")
        if (isinstance(self.confidence_margin, bool)
                or not isinstance(self.confidence_margin, (int, float))
                or not math.isfinite(self.confidence_margin) or self.confidence_margin < 0):
            raise ValueError("passage_confidence_margin must be finite and nonnegative")
        for name, value, lower, upper in (
            ("max_gap_seconds", self.max_gap_seconds, 0, math.inf),
            ("recovery_area_ratio", self.recovery_area_ratio, 1, math.inf),
            ("recovery_bottom_fraction", self.recovery_bottom_fraction, 0, 1),
        ):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not lower < value <= upper):
                raise ValueError(f"passage_{name} must be finite and in ({lower}, {upper}]")

    @staticmethod
    def _geometry(detection, frame_size):
        """Reject malformed geometry, never clamp it into recovery evidence."""
        try:
            if len(frame_size) != 2 or len(detection.get("bbox")) != 4:
                return None
            values = (*frame_size, *detection["bbox"])
            if any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in values):
                return None
            width, height = frame_size
            x1, y1, x2, y2 = detection["bbox"]
            if width < 2 or height < 2 or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                return None
            return {"bbox": [x1, y1, x2, y2], "frame_size": list(frame_size)}
        except TypeError:
            return None

    def observe(self, detections, frame, timestamp_ms, *, frame_size=None):
        # Callers supply strictly increasing source frames/times, not wall time.
        self.last = (frame, timestamp_ms)
        scores = {}
        seen = set()
        invalid = False
        for detection in detections:
            track_id, score = detection["track_id"], detection["confidence"]
            if track_id in seen:
                self.warnings.add("duplicate_track_id_unresolved")
                invalid = True
            seen.add(track_id)
            if math.isfinite(score) and 0 <= score <= 1:
                scores[track_id] = score
            else:
                self.warnings.add("invalid_confidence_unresolved")
                invalid = True
        active = self.intervals[-1]["track_id"] if self.intervals else None
        if invalid:
            self.pending = None
            return
        by_id = {d["track_id"]: d for d in detections}
        missing = active is not None and active not in scores
        if active is not None:
            if missing:
                self.active_missing = True
                self.warnings.add("missing_incumbent_attribution_unresolved")
            else:
                if self.active_missing:
                    self.warnings.add("possible_track_id_reuse_unresolved")
                    self.active_missing = False
                    self.pending = None
                geometry = self._geometry(by_id[active], frame_size)
                if geometry is not None:
                    self.reference = dict(geometry, track_id=active, frame=frame,
                                          timestamp_ms=timestamp_ms)
        method = "missing_incumbent_foreground" if missing else "concurrent_confidence"
        if self.pending and self.pending.get("method", "concurrent_confidence") != method:
            self.pending = None
        if missing:
            if (self.reference is None or self.reference["track_id"] != active
                    or self._geometry(self.reference, frame_size) is None
                    or self.reference["frame_size"] != list(frame_size)
                    or not 0 < timestamp_ms - self.reference["timestamp_ms"] <= self.max_gap_seconds * 1000):
                self.warnings.add("missing_incumbent_reference_unavailable_or_stale")
                self.pending = None
                return
        ranked = sorted(scores, key=scores.get, reverse=True)
        winner = ranked[0] if ranked else None
        if len(ranked) > 1:
            delta = scores[winner] - scores[ranked[1]]
            if delta <= 0 or ((active is None or missing) and delta < self.confidence_margin
                              and not math.isclose(delta, self.confidence_margin,
                                                   rel_tol=0, abs_tol=PASSAGE_MARGIN_EPSILON)):
                self.warnings.add("ambiguous_dominance_unresolved")
                winner = None
        if winner is None or winner == active or winner in self.retired:
            if winner in self.retired:
                self.warnings.add("finalized_track_reentry_unresolved")
            self.pending = None
            return
        # The unique strongest challenger must also beat this sample's incumbent.
        if active is not None and not missing:
            delta = scores[winner] - scores[active]
            if delta <= 0 or (delta < self.confidence_margin
                              and not math.isclose(delta, self.confidence_margin,
                                                   rel_tol=0, abs_tol=PASSAGE_MARGIN_EPSILON)):
                self.pending = None
                return
        recovery = None
        if missing:
            geometry = self._geometry(by_id[winner], frame_size)
            if geometry is not None:
                a, b = self.reference["bbox"], geometry["bbox"]
                area_ratio = ((b[2] - b[0]) * (b[3] - b[1])) / ((a[2] - a[0]) * (a[3] - a[1]))
                bottom_fraction = (b[3] - a[3]) / frame_size[1]
                if area_ratio >= self.recovery_area_ratio and bottom_fraction >= self.recovery_bottom_fraction:
                    recovery = dict(geometry, area_ratio=area_ratio, bottom_fraction=bottom_fraction)
            if recovery is None:
                self.pending = None
                return
        if active is None:
            support = {"track_id": winner, "count": 1, "start_frame": frame,
                       "start_timestamp_ms": timestamp_ms}
        else:
            if self.pending is None or self.pending["track_id"] != winner:
                self.pending = {"track_id": winner, "count": 0, "start_frame": frame,
                                "start_timestamp_ms": timestamp_ms}
                if missing:
                    self.pending["method"] = method
            self.pending["count"] += 1
            if self.pending["count"] < self.confirmation_count:
                return
            support = self.pending.copy()
            if missing:
                support.update(reference=self.reference.copy(), geometry=recovery,
                               elapsed_since_reference_ms=timestamp_ms - self.reference["timestamp_ms"])
            self.intervals[-1].update(end_frame=frame, end_timestamp_ms=timestamp_ms,
                                      end_exclusive=True)
            self.retired.add(active)
        self.intervals.append({"track_id": winner, "status": "passage_evidence_candidate",
                               "transition_method": method if active is not None else "initial_confidence",
                               "sequence": len(self.intervals) + 1,
                               "candidate": {"status": "not_available", "reason": "no_candidate_captured",
                                             "bcs_eligibility": "not_assessed", "bcs_status": "not_evaluated"},
                               "start_frame": frame, "start_timestamp_ms": timestamp_ms,
                               "support": dict(support, confirmation_frame=frame,
                                               confirmation_timestamp_ms=timestamp_ms),
                               "warnings": []})
        self.pending = None
        self.active_missing = False
        geometry = self._geometry(by_id[winner], frame_size)
        self.reference = (dict(geometry, track_id=winner, frame=frame, timestamp_ms=timestamp_ms)
                          if geometry is not None else None)

    def capture_candidate(self, output, image, detection, score, contamination, overlap_iou,
                          crop_margin, others=()):
        """Persist only the active interval's observed source, with earliest ties.

        Called after observe on the same sample, so a confirmation frame belongs
        only to the new interval. No full-resolution passage images are retained.
        The original is a full frame, not an isolated animal or visibility proof.
        """
        if not self.intervals or self.last is None:
            return
        interval = self.intervals[-1]
        if detection["track_id"] != interval["track_id"]:
            return
        candidate = interval["candidate"]
        if not math.isfinite(score) or not 0 <= detection["confidence"] <= 1:
            return
        if candidate["status"] == "available" and score <= candidate["quality"]["score"]:
            return
        relative = Path("passages") / str(interval["sequence"]) / "best-original.jpg"
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_image(target, image)
        # Publish provenance only after the image has been successfully replaced.
        frame, timestamp_ms = self.last
        height, width = image.shape[:2]
        interval["candidate"] = {
            "status": "available", "original": relative.as_posix(),
            "frame_index": frame, "timestamp_ms": timestamp_ms,
            "track_id": detection["track_id"], "confidence": detection["confidence"],
            "bbox": list(detection["bbox"]),
            "quality": {"method": "frame_quality", "score": score,
                        "crop_contamination": contamination, "crop_margin": crop_margin,
                        "iou": overlap_iou, "touches_border": touches_border(detection["bbox"], width, height),
                        "frame_size": [width, height], "other_bboxes": [list(box) for box in others]},
            "bcs_eligibility": "not_assessed", "bcs_status": "not_evaluated"}

    def finalize_candidates(self, output, output_size, crop_margin, segmenter=None):
        """Decode each settled candidate sequentially, never the track-global best.

        Passage segmentation has no track segment_limit. All artifacts come from
        the persisted JPEG (including its compression), not a retained raw frame.
        Segmentation is neither anatomical visibility nor unique identity proof.
        """
        import cv2

        for interval in self.intervals:
            candidate = interval["candidate"]
            if candidate["status"] != "available":
                continue
            image = cv2.imread(str(output / candidate["original"]))
            if image is None or list(image.shape[1::-1]) != candidate["quality"]["frame_size"]:
                raise OSError("Could not decode saved passage candidate")
            relative = Path(candidate["original"]).parent
            bbox = tuple(candidate["bbox"])
            crop = crop_with_padding(image, bbox, output_size, crop_margin)
            _write_image(output / relative / "best-crop.png", crop)
            segmentation = _segmentation_artifacts(
                output, relative, image, bbox, candidate["quality"]["other_bboxes"],
                crop, output_size, crop_margin, segmenter)
            candidate.update(best_crop=(relative / "best-crop.png").as_posix(),
                             artifact_source="decoded_saved_original_jpeg", segmentation=segmentation)

    def summary(self, stats):
        if self.intervals and self.last is not None:
            self.intervals[-1].update(end_frame=self.last[0], end_timestamp_ms=self.last[1],
                                      end_exclusive=False)
        for interval in self.intervals:
            item = stats[interval["track_id"]]
            if item.possible_continuation_of is not None or item.possible_continued_by is not None:
                interval["warnings"] = ["possible_fragment_identity_unresolved"]
                self.warnings.add("possible_fragment_identity_unresolved")
        selected = {interval["track_id"] for interval in self.intervals}
        unresolved = [{"track_id": item.track_id, "first_frame": item.first_frame,
                       "last_frame": item.last_frame, "first_timestamp_ms": item.first_timestamp_ms,
                       "last_timestamp_ms": item.last_timestamp_ms,
                       "possible_continuation_of": item.possible_continuation_of,
                       "possible_continued_by": item.possible_continued_by,
                       "status": "unresolved_passage_attribution"}
                      for item in stats.values() if item.track_id not in selected]
        return {"status": "provisional_evidence_not_unique_animal_identity",
                "policy": {"confirmation_count": self.confirmation_count,
                           "confidence_margin": self.confidence_margin,
                           "max_gap_seconds": self.max_gap_seconds,
                           "recovery_area_ratio": self.recovery_area_ratio,
                           "recovery_bottom_fraction": self.recovery_bottom_fraction,
                           "recovery_limitations": "uncalibrated_larger_lower_foreground_not_identity; "
                                                   "fragment_like_or_stale_geometry_stays_unresolved",
                           "observation_order": "strictly_increasing_source_frames_and_timestamps",
                           "numerical_margin_epsilon": PASSAGE_MARGIN_EPSILON,
                           "calibration": "uncalibrated_confidence_proxy_not_visibility_or_bcs",
                           "boundary": "confirmation_sample_half_open_final_end_inclusive",
                           "candidate_selection": "interval_local_frame_quality_max_earliest_tie",
                           "candidate_limitations": "full_frame_not_isolated_animal; "
                                                    "contamination_measures_detected_boxes_only; "
                                                    "no_anatomical_visibility_guarantee",
                           "initial_selection": "single_observation_unique_margin_dominance"},
                "intervals": self.intervals, "pending": self.pending,
                "unresolved_tracks": unresolved, "warnings": sorted(self.warnings)}


def track_video(input_path: Path, output: Path, model_path: Path, tracker: str = "bytetrack.yaml",
                confidence: float = 0.25, sample_fps: float | None = None,
                output_size=(320, 320), *, max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
                crop_margin: float = DEFAULT_CROP_MARGIN, clip_format: str = "mp4",
                max_seconds: float | None = None, max_frames: int | None = None, model=None,
                segmenter=None, segment_limit: int = DEFAULT_SEGMENT_LIMIT,
                passage_confirmation_count: int = 3, passage_confidence_margin: float = 0.1,
                passage_recovery_area_ratio: float = 1.5,
                passage_recovery_bottom_fraction: float = 0.10) -> dict:
    """Run Ultralytics tracking and write one directory per temporal track.

    ``model`` accepts any object exposing Ultralytics' ``track`` API; when it is
    None the local weights at ``model_path`` are loaded, never downloaded, and
    a missing ``lap`` fails instead of letting Ultralytics install it.
    ``segmenter(image, bbox) -> bool mask`` optionally segments each track's
    best frame (longest tracks first, up to ``segment_limit``).
    Each available passage candidate is also cropped and optionally segmented
    from its own saved original, without the legacy track segmentation limit.
    Passage parameters are uncalibrated confidence heuristics, not anatomical
    visibility, BCS, or unique-animal identity. Effective policy is in the summary.
    """
    import cv2

    passages = PassageEvidence(passage_confirmation_count, passage_confidence_margin,
                               max_gap_seconds, passage_recovery_area_ratio,
                               passage_recovery_bottom_fraction)
    input_path, output, model_path = Path(input_path), Path(output), Path(model_path)
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("confidence debe estar entre 0 y 1")
    if sample_fps is not None and (not math.isfinite(sample_fps) or not 0 < sample_fps <= 120):
        raise ValueError("sample_fps debe estar entre 0 y 120")
    if not math.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds debe ser positivo")
    if not 0 <= crop_margin <= 1:
        raise ValueError("crop_margin debe estar entre 0 y 1")
    if clip_format not in CLIP_CODECS:
        raise ValueError(f"clip_format debe ser uno de: {', '.join(CLIP_CODECS)}")
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds debe ser positivo")
    if max_frames is not None and (type(max_frames) is not int or max_frames < 1):
        raise ValueError("max_frames debe ser un entero positivo")
    output_size = _validate_output_size(output_size)
    if model is None and find_spec("lap") is None:
        raise ImportError("Falta lap==0.5.13 para ByteTrack/BoT-SORT; no se instala automáticamente")
    if output.exists():
        raise ValueError(f"La carpeta de salida ya existe: {output}")
    if not input_path.is_file() or not model_path.is_file():
        raise ValueError("Falta el video de entrada o el modelo local")
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("No se pudo abrir el video")
    run_warnings = []
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(fps) or fps <= 0:
        fps = 24.0
        run_warnings.append("fps_unavailable_assumed_24")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    clip_fps = min(sample_fps, fps) if sample_fps else fps
    # Count includes the first missing sample: all confirmations must fit since
    # the last valid incumbent, not just since the challenger started winning.
    if passage_confirmation_count / clip_fps > max_gap_seconds:
        passages.warnings.add("recovery_confirmation_exceeds_gap_at_sample_rate")
    max_gap_ms = max_gap_seconds * 1000.0
    stats: dict[int, TrackStats] = {}
    writers = {}
    placeholder = gap_placeholder(output_size)
    clip_name = f"clip.{clip_format}"
    stop_reason = "end_of_video"
    frame_index = 0
    sampled_frames = 0
    next_sample_ms = 0.0
    try:
        if model is None:
            from ultralytics import YOLO

            model = YOLO(str(model_path))
        model_sha256 = _file_sha256(model_path)
        output.mkdir(parents=True)
        tracks_dir = output / "tracks"
        tracks_dir.mkdir()
        with (output / "observations.jsonl").open("w", encoding="utf-8") as observations:
            while True:
                if max_frames is not None and frame_index >= max_frames:
                    stop_reason = "frame_limit"
                    break
                ok, image = capture.read()
                if not ok:
                    break
                timestamp_ms = frame_index * 1000.0 / fps
                if max_seconds is not None and timestamp_ms >= max_seconds * 1000.0:
                    stop_reason = "time_limit"
                    break
                if sample_fps and timestamp_ms + 1e-6 < next_sample_ms:
                    frame_index += 1
                    continue
                if sample_fps:
                    next_sample_ms = (math.floor(timestamp_ms / (1000.0 / sample_fps)) + 1) * (1000.0 / sample_fps)
                sample_index = sampled_frames
                sampled_frames += 1
                results = model.track(image, persist=True, tracker=tracker, conf=confidence,
                                      verbose=False, device="cpu")
                detections, all_boxes = _result_detections(results[0], image)
                if len(all_boxes) > len(detections):
                    passages.warnings.add("untracked_detection_attribution_unresolved")
                passages.observe(detections, frame_index, timestamp_ms,
                                 frame_size=(image.shape[1], image.shape[0]))
                for detection in detections:
                    track_id, bbox = detection["track_id"], detection["bbox"]
                    others = [box for index, box in enumerate(all_boxes) if index != detection["box_index"]]
                    contamination = crop_contamination(bbox, others, width, height, crop_margin)
                    overlap_iou = max((iou(bbox, other) for other in others), default=0.0)
                    score = frame_quality(image, bbox, detection["confidence"], contamination)
                    item = stats.setdefault(track_id, TrackStats(track_id))
                    track_dir = tracks_dir / str(track_id)
                    if track_id not in writers:
                        track_dir.mkdir()
                        writer = cv2.VideoWriter(str(track_dir / clip_name),
                                                 cv2.VideoWriter_fourcc(*CLIP_CODECS[clip_format]),
                                                 clip_fps, output_size)
                        if not writer.isOpened():
                            raise ValueError("No se pudo abrir el writer de clips")
                        writers[track_id] = writer
                    writer = writers[track_id]
                    missing = 0 if item.last_sample is None else sample_index - item.last_sample - 1
                    if missing > 0:
                        between_ms = timestamp_ms - item.last_timestamp_ms
                        splits = between_ms > max_gap_ms
                        item.gaps.append({"after_frame": item.last_frame, "before_frame": frame_index,
                                          "missing_samples": missing,
                                          "between_observations_ms": round(between_ms, 3),
                                          "splits_segment": splits})
                        for _ in range(missing):
                            writer.write(placeholder)
                        item.placeholder_frames += missing
                        item.clip_frames += missing
                        if splits:
                            item.segments.append({})
                    if not item.segments:
                        item.segments.append({})
                    segment = item.segments[-1]
                    if not segment:
                        segment.update(first_frame=frame_index, first_timestamp_ms=round(timestamp_ms, 3),
                                       observations=0)
                    segment.update(last_frame=frame_index, last_timestamp_ms=round(timestamp_ms, 3),
                                   observations=segment["observations"] + 1)
                    item.observations += 1
                    item.last_sample = sample_index
                    if item.first_frame is None:
                        item.first_frame, item.first_timestamp_ms = frame_index, timestamp_ms
                        item.first_bbox = bbox
                    item.last_frame, item.last_timestamp_ms = frame_index, timestamp_ms
                    item.last_bbox = bbox
                    item.max_width = max(item.max_width, bbox[2] - bbox[0])
                    item.max_height = max(item.max_height, bbox[3] - bbox[1])
                    if score > item.best_score:
                        item.best_score = score
                        item.best_frame = image.copy()
                        item.best_frame_index = frame_index
                        item.best_bbox = bbox
                        item.best_timestamp_ms = timestamp_ms
                        item.best_confidence = detection["confidence"]
                        item.best_contamination = contamination
                        item.best_iou = overlap_iou
                        item.best_others = others
                    # Duplicate IDs have no unique box attribution on this sample.
                    if sum(other["track_id"] == track_id for other in detections) == 1:
                        passages.capture_candidate(output, image, detection, score, contamination,
                                                   overlap_iou, crop_margin, others)
                    writer.write(crop_with_padding(image, bbox, output_size, crop_margin))
                    item.clip_frames += 1
                    observations.write(json.dumps({
                        "track_id": track_id, "frame_index": frame_index, "sample_index": sample_index,
                        "timestamp_ms": round(timestamp_ms, 3), "bbox": list(bbox),
                        "confidence": round(detection["confidence"], 6),
                        "crop_contamination": round(contamination, 6), "iou": round(overlap_iou, 6),
                    }, ensure_ascii=True) + "\n")
                frame_index += 1
    finally:
        capture.release()
        for writer in writers.values():
            writer.release()
    link_fragments(stats, max_gap_ms)
    passages.finalize_candidates(output, output_size, crop_margin, segmenter)
    segmented = set(sorted(stats, key=lambda key: -stats[key].observations)[:segment_limit]) if segmenter else set()
    track_summaries = []
    for track_id, item in sorted(stats.items()):
        track_dir = tracks_dir / str(track_id)
        original_name, crop_name = "best-original.jpg", "best-crop.png"
        _write_image(track_dir / original_name, item.best_frame)
        crop = crop_with_padding(item.best_frame, item.best_bbox, output_size, crop_margin)
        _write_image(track_dir / crop_name, crop)
        warnings = track_warnings(item, width, height)
        segmentation = {"status": "not_requested"}
        if track_id in segmented:
            segmentation = _segmentation_artifacts(
                output, Path("tracks") / str(track_id), item.best_frame, item.best_bbox,
                item.best_others, crop, output_size, crop_margin, segmenter)
        elif segmenter:
            segmentation = {"status": "skipped", "reason": "segment_limit"}
        track_summaries.append({"track_id": track_id, "observations": item.observations,
            "first_frame": item.first_frame, "last_frame": item.last_frame,
            "first_timestamp_ms": item.first_timestamp_ms, "last_timestamp_ms": item.last_timestamp_ms,
            "segments": item.segments, "gaps": item.gaps,
            "possible_continuation_of": item.possible_continuation_of,
            "possible_continued_by": item.possible_continued_by,
            "best_frame_index": item.best_frame_index, "best_timestamp_ms": item.best_timestamp_ms,
            "best_bbox": list(item.best_bbox), "best_confidence": item.best_confidence,
            "best_score": round(item.best_score, 6),
            "best_crop_contamination": round(item.best_contamination, 6),
            "best_iou": round(item.best_iou, 6),
            "clip": f"tracks/{track_id}/{clip_name}", "clip_frames": item.clip_frames,
            "clip_placeholder_frames": item.placeholder_frames,
            "clip_duration_ms": round(item.clip_frames * 1000.0 / clip_fps, 3),
            "best_original": f"tracks/{track_id}/{original_name}",
            "best_crop": f"tracks/{track_id}/{crop_name}",
            "segmentation": segmentation,
            "status": "review" if warnings else "candidate",
            "warnings": warnings})
    summary = {"schema_version": SCHEMA_VERSION, "input": str(input_path),
               "model": str(model_path), "model_sha256": model_sha256, "tracker": tracker,
               "config": {"confidence": confidence, "sample_fps": sample_fps,
                          "max_gap_seconds": max_gap_seconds, "crop_margin": crop_margin,
                          "output_size": list(output_size), "clip_format": clip_format,
                          "max_seconds": max_seconds, "max_frames": max_frames,
                          "contamination_review_threshold": CONTAMINATION_REVIEW_THRESHOLD,
                          "min_observations": MIN_OBSERVATIONS},
               "fps": fps, "clip_fps": clip_fps, "width": width, "height": height,
               "frames_read": frame_index, "frames_sampled": sampled_frames, "stop_reason": stop_reason,
               "tracks": track_summaries, "identity_semantics": "temporal_track_id_within_video",
               "passage_evidence": passages.summary(stats),
               "warnings": run_warnings,
               "segmentation": dict(getattr(segmenter, "metadata", {}), limit=segment_limit,
                                    min_fill=SEGMENT_MIN_FILL, max_fill=SEGMENT_MAX_FILL,
                                    max_outside=SEGMENT_MAX_OUTSIDE) if segmenter else None,
               "notes": ["No se reconstruyen partes ocultas.",
                         "La silueta (best-masked.png) sólo se segmenta en el mejor frame y no "
                         "reconstruye lo que tapa otro animal.",
                         "Los clips conservan la línea de tiempo muestreada: cada muestra sin detección "
                         "entre dos observaciones se escribe como un lienzo gris rotulado.",
                         "Un hueco mayor a max_gap_seconds abre un segmento nuevo y marca el track para revisión.",
                         "La contaminación sólo mide otras detecciones; una vaca no detectada dentro "
                         "del recorte no se mide.",
                         "El recorte es rectangular: no separa píxeles de otra vaca dentro de la caja."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return summary
