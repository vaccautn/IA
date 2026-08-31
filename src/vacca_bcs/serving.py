"""Safe, capability-scoped loading of integer BCS serving checkpoints."""

from __future__ import annotations

import math
import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps

from vacca_vision.image_validation import ImageValidationConfig, _validate_dimensions

from .constants import (
    BCS_DOMAIN_ID,
    CLASS_NAMES,
    NUM_CLASSES,
    NUM_THRESHOLDS,
    SCORE_BASE,
    SCORE_MAX,
    SCORE_MIN,
    SCORE_STEP,
)
from .dataset import build_transforms
from .model import BCSOrdinalModel

CHECKPOINT_SCHEMA_VERSION = "bcs-ordinal-integer-checkpoint-v1"
SNAPSHOT_SCHEMA_VERSION = "bcs-integer-snapshot-v2"
_REQUIRED_FIELDS = frozenset(
    {
        "checkpoint_schema_version",
        "domain_id",
        "classes",
        "class_mapping",
        "score_min",
        "score_max",
        "score_base",
        "score_step",
        "num_classes",
        "num_thresholds",
        "snapshot_schema",
        "snapshot_identity",
        "dataset_manifest_digest",
        "run_id",
        "config",
        "model_state_dict",
    }
)
_OPTIONAL_TRAINING_FIELDS = frozenset({"epoch", "val_mae", "provenance"})
_IMAGE_CONFIG = ImageValidationConfig()


class BCSCheckpointError(Exception):
    pass


class BCSCheckpointUnavailableError(BCSCheckpointError):
    pass


class BCSCheckpointLoadError(BCSCheckpointError):
    """Raised when a checkpoint cannot be safely loaded or validated."""


BCSServingError = BCSCheckpointError
BCSServingUnavailableError = BCSCheckpointUnavailableError
BCSServingLoadError = BCSCheckpointLoadError


class BCSInferenceError(Exception):
    """Base class for sanitized BCS inference failures."""


class BCSInferenceInputError(BCSInferenceError):
    """Raised when uploaded image bytes are not a safe supported image."""


class BCSInferenceExecutionError(BCSInferenceError):
    """Raised when model inference cannot produce a valid ordinal result."""


@dataclass(frozen=True, slots=True)
class BCSLineageMetadata:
    """Safe immutable lineage values exposed to the serving boundary."""

    checkpoint_schema_version: str
    domain_id: str
    snapshot_schema: str
    snapshot_identity: str
    dataset_manifest_digest: str
    run_id: str


@dataclass(frozen=True, slots=True)
class LoadedBCSModel:
    """An evaluated model and immutable serving metadata."""

    model: BCSOrdinalModel
    imgsz: int
    device: torch.device
    lineage: BCSLineageMetadata


@dataclass(frozen=True, slots=True)
class BCSInferenceResult:
    """Immutable continuous BCS output and safe model lineage."""

    continuous_score: float
    lineage: BCSLineageMetadata

    @property
    def score(self) -> float:
        return self.continuous_score


class BCSInferenceService:
    """Run deterministic full-image inference for one loaded BCS model.

    The service owns no global state. Validation transforms are deterministic and
    each request allocates its own tensor, so concurrent calls only read the model.
    """

    def __init__(self, loaded: LoadedBCSModel) -> None:
        if not isinstance(loaded, LoadedBCSModel):
            raise BCSInferenceInputError("loaded BCS model is invalid")
        self._loaded = loaded
        try:
            self._transform = build_transforms(loaded.imgsz, train=False)
        except Exception:
            raise BCSInferenceInputError("BCS image preprocessing is unavailable") from None

    def infer(self, image_bytes: bytes | bytearray | memoryview) -> BCSInferenceResult:
        tensor = self._prepare(image_bytes)
        try:
            self._loaded.model.eval()
            with torch.inference_mode():
                logits = self._loaded.model(tensor.to(self._loaded.device))
        except Exception:
            raise BCSInferenceExecutionError("BCS inference failed") from None
        if (
            not isinstance(logits, torch.Tensor)
            or tuple(logits.shape) != (1, NUM_THRESHOLDS)
            or not bool(torch.isfinite(logits).all())
        ):
            raise BCSInferenceExecutionError("BCS model returned invalid logits")
        try:
            score = float(SCORE_BASE + torch.sigmoid(logits).sum().item())
        except Exception:
            raise BCSInferenceExecutionError("BCS model returned an invalid score") from None
        if not math.isfinite(score) or not SCORE_BASE <= score <= SCORE_MAX:
            raise BCSInferenceExecutionError("BCS model returned an out-of-range score")
        return BCSInferenceResult(score, self._loaded.lineage)

    def _prepare(self, image_bytes: bytes | bytearray | memoryview) -> torch.Tensor:
        if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
            raise BCSInferenceInputError("BCS image input must be bytes")
        try:
            if not image_bytes or len(image_bytes) > _IMAGE_CONFIG.max_size_bytes:
                raise BCSInferenceInputError("BCS image input is empty or too large")
            payload = bytes(image_bytes)
        except BCSInferenceInputError:
            raise
        except (TypeError, ValueError):
            raise BCSInferenceInputError("BCS image input is invalid") from None
        if not payload or len(payload) > _IMAGE_CONFIG.max_size_bytes:
            raise BCSInferenceInputError("BCS image input is empty or too large")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(payload)) as probe:
                    if probe.format not in {"JPEG", "PNG"}:
                        raise BCSInferenceInputError("BCS image format is unsupported")
                    _validate_dimensions(*probe.size, _IMAGE_CONFIG)
                    probe.verify()
                with Image.open(BytesIO(payload)) as decoded:
                    decoded.load()
                    transposed = ImageOps.exif_transpose(decoded)
                    try:
                        with transposed.convert("RGB") as rgb:
                            return self._transform(rgb).unsqueeze(0)
                    finally:
                        if transposed is not decoded:
                            transposed.close()
        except BCSInferenceInputError:
            raise
        except Exception:
            raise BCSInferenceInputError("BCS image cannot be decoded safely") from None


def infer_bcs(
    loaded: LoadedBCSModel, image_bytes: bytes | bytearray | memoryview
) -> BCSInferenceResult:
    """Infer one continuous integer-domain BCS score from a full image."""
    return BCSInferenceService(loaded).infer(image_bytes)


def _valid_digest(value: object, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolve_checkpoint_path(raw_path: str | os.PathLike[str]) -> Path:
    try:
        path = Path(raw_path)
        if path.is_symlink():
            raise BCSCheckpointLoadError("BCS checkpoint path must not be a symlink")
        if not path.exists():
            raise BCSCheckpointUnavailableError("BCS checkpoint is unavailable")
        if not path.is_file():
            raise BCSCheckpointLoadError("BCS checkpoint must be a regular file")
    except BCSCheckpointError:
        raise
    except (OSError, TypeError, ValueError):
        raise BCSCheckpointLoadError("BCS checkpoint path is invalid") from None
    return path


def _resolve_device(raw_device: str | torch.device) -> torch.device:
    try:
        device = torch.device(raw_device)
    except (RuntimeError, TypeError, ValueError):
        raise BCSCheckpointLoadError("BCS serving device is invalid") from None
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise BCSCheckpointUnavailableError("requested CUDA device is unavailable")
        count = torch.cuda.device_count()
        if count <= 0 or device.index is not None and not 0 <= device.index < count:
            raise BCSCheckpointUnavailableError("requested CUDA device is unavailable")
    return device


def _validate_checkpoint(checkpoint: object) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise BCSCheckpointLoadError("BCS checkpoint has an invalid schema")
    fields = set(checkpoint)
    if not _REQUIRED_FIELDS.issubset(fields):
        raise BCSCheckpointLoadError("BCS checkpoint is missing required fields")
    if fields.difference(_REQUIRED_FIELDS | _OPTIONAL_TRAINING_FIELDS):
        raise BCSCheckpointLoadError("BCS checkpoint has unexpected fields")

    if checkpoint["checkpoint_schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise BCSCheckpointLoadError("BCS checkpoint schema is unsupported")
    if checkpoint["domain_id"] != BCS_DOMAIN_ID:
        raise BCSCheckpointLoadError("BCS checkpoint domain is invalid")

    classes = checkpoint["classes"]
    mapping = checkpoint["class_mapping"]
    expected_mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    if (
        type(classes) is not list
        or any(type(value) is not str for value in classes)
        or classes != list(CLASS_NAMES)
        or type(mapping) is not dict
        or mapping != expected_mapping
        or any(type(value) is not int for value in mapping.values())
    ):
        raise BCSCheckpointLoadError("BCS checkpoint classes are invalid")

    expected_scale = {
        "score_min": SCORE_MIN,
        "score_max": SCORE_MAX,
        "score_base": SCORE_BASE,
        "score_step": SCORE_STEP,
        "num_classes": NUM_CLASSES,
        "num_thresholds": NUM_THRESHOLDS,
    }
    if any(
        type(checkpoint[field]) is not int or checkpoint[field] != expected
        for field, expected in expected_scale.items()
    ):
        raise BCSCheckpointLoadError("BCS checkpoint scale is invalid")

    if (
        checkpoint["snapshot_schema"] != SNAPSHOT_SCHEMA_VERSION
        or not _valid_digest(checkpoint["snapshot_identity"], 64)
        or not _valid_digest(checkpoint["dataset_manifest_digest"], 64)
        or not _valid_digest(checkpoint["run_id"], 32)
    ):
        raise BCSCheckpointLoadError("BCS checkpoint lineage is invalid")

    config = checkpoint["config"]
    if not isinstance(config, dict) or type(config.get("imgsz")) is not int or config["imgsz"] <= 0:
        raise BCSCheckpointLoadError("BCS checkpoint image size is invalid")

    state_dict = checkpoint["model_state_dict"]
    if (
        not isinstance(state_dict, Mapping)
        or not state_dict
        or any(type(key) is not str for key in state_dict)
        or any(not isinstance(value, torch.Tensor) for value in state_dict.values())
    ):
        raise BCSCheckpointLoadError("BCS checkpoint model state is invalid")
    return checkpoint


def load_bcs_model(
    checkpoint_path: str | os.PathLike[str], device: str | torch.device = "cpu"
) -> LoadedBCSModel:
    """Load one validated integer checkpoint without unsafe deserialization."""
    path = _resolve_checkpoint_path(checkpoint_path)
    resolved_device = _resolve_device(device)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        raise BCSCheckpointLoadError("BCS checkpoint could not be loaded safely") from None
    checkpoint = _validate_checkpoint(checkpoint)

    try:
        model = BCSOrdinalModel(pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(resolved_device)
        model.eval()
    except Exception:
        raise BCSCheckpointLoadError("BCS checkpoint model architecture is invalid") from None

    lineage = BCSLineageMetadata(
        checkpoint["checkpoint_schema_version"],
        checkpoint["domain_id"],
        checkpoint["snapshot_schema"],
        checkpoint["snapshot_identity"],
        checkpoint["dataset_manifest_digest"],
        checkpoint["run_id"],
    )
    return LoadedBCSModel(model, checkpoint["config"]["imgsz"], resolved_device, lineage)
