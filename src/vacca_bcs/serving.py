"""Safe loading and hard-category serving for BCS CORAL checkpoints."""
from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
from typing import Any
import warnings

import torch
from PIL import Image, ImageOps

from vacca_vision.image_validation import ImageValidationConfig, _validate_dimensions

from .category_snapshot import SNAPSHOT_SCHEMA_VERSION
from .checkpoint_io import (
    CheckpointByteError,
    CheckpointBytes,
    CheckpointByteUnavailableError,
    load_checkpoint_bytes,
)
from .constants import (
    BCS_DOMAIN_ID,
    BCS_CLASS_SCORES,
    CHECKPOINT_SCHEMA_VERSION,
    CLASS_NAMES,
    NUM_CLASSES,
    NUM_THRESHOLDS,
    SCORE_BASE,
    SCORE_MAX,
    SCORE_MIN,
    SCORE_STEP,
)
from .dataset import build_transforms
from .local_source import LOCAL_BCS_MAPPING, LOCAL_SOURCE_SCHEMA
from .model import BCSOrdinalModel
from .path_safety import SafePathError

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_ROOT = REPO_ROOT / "outputs"
_REQUIRED_FIELDS = frozenset({"checkpoint_schema_version", "domain_id", "source_schema", "classes", "class_mapping", "score_min", "score_max", "score_base", "score_step", "num_classes", "num_thresholds", "snapshot_schema", "snapshot_identity", "dataset_manifest_digest", "run_id", "config_sha256", "observed_classes", "missing_classes", "source_identity_scheme", "source_mapping", "config", "provenance", "model_state_dict"})
_OPTIONAL_FIELDS = frozenset({"epoch", "val_ordinal_mae", "best_epoch", "best_validation", "selection_identity", "best_results_row"})
_IMAGE_CONFIG = ImageValidationConfig()


class BCSCheckpointError(Exception):
    pass


class BCSCheckpointUnavailableError(BCSCheckpointError):
    pass


class BCSCheckpointLoadError(BCSCheckpointError):
    pass


class BCSInferenceError(Exception):
    pass


class BCSInferenceInputError(BCSInferenceError):
    pass


class BCSInferenceExecutionError(BCSInferenceError):
    pass


@dataclass(frozen=True, slots=True)
class BCSLineageMetadata:
    checkpoint_schema_version: str
    domain_id: str
    snapshot_schema: str
    snapshot_identity: str
    dataset_manifest_digest: str
    run_id: str
    source_schema: str
    source_identity_scheme: str
    source_mapping: tuple[tuple[str, int], ...]
    observed_classes: tuple[int, ...]
    missing_classes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LoadedBCSModel:
    model: BCSOrdinalModel
    imgsz: int
    device: torch.device
    lineage: BCSLineageMetadata
    checkpoint_sha256: str | None = None
    checkpoint: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class BCSInferenceResult:
    bcs_category: int
    lineage: BCSLineageMetadata

class BCSInferenceService:
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
        if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (1, NUM_THRESHOLDS) or not bool(torch.isfinite(logits).all()):
            raise BCSInferenceExecutionError("BCS model returned invalid logits")
        try:
            category = int((torch.sigmoid(logits) > 0.5).sum().item()) + 1
        except Exception:
            raise BCSInferenceExecutionError("BCS model returned an invalid category") from None
        if category not in BCS_CLASS_SCORES:
            raise BCSInferenceExecutionError("BCS model returned an out-of-range category")
        return BCSInferenceResult(category, self._loaded.lineage)

    def _prepare(self, image_bytes: bytes | bytearray | memoryview) -> torch.Tensor:
        if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
            raise BCSInferenceInputError("BCS image input must be bytes")
        try:
            payload = bytes(image_bytes)
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


def infer_bcs(loaded: LoadedBCSModel, image_bytes: bytes | bytearray | memoryview) -> BCSInferenceResult:
    return BCSInferenceService(loaded).infer(image_bytes)


def _valid_digest(value: object, length: int) -> bool:
    return type(value) is str and len(value) == length and value == value.lower() and all(character in "0123456789abcdef" for character in value)


def _validate_checkpoint(checkpoint: object) -> dict[str, Any]:
    if not isinstance(checkpoint, dict) or not _REQUIRED_FIELDS.issubset(checkpoint) or set(checkpoint).difference(_REQUIRED_FIELDS | _OPTIONAL_FIELDS):
        raise BCSCheckpointLoadError("BCS checkpoint schema is invalid")
    if checkpoint["checkpoint_schema_version"] != CHECKPOINT_SCHEMA_VERSION or checkpoint["domain_id"] != BCS_DOMAIN_ID or checkpoint["snapshot_schema"] != SNAPSHOT_SCHEMA_VERSION:
        raise BCSCheckpointLoadError("BCS checkpoint lineage is unsupported")
    if checkpoint["source_schema"] != LOCAL_SOURCE_SCHEMA:
        raise BCSCheckpointLoadError("BCS checkpoint source schema is invalid")
    if checkpoint["classes"] != list(CLASS_NAMES) or checkpoint["class_mapping"] != {name: index for index, name in enumerate(CLASS_NAMES)}:
        raise BCSCheckpointLoadError("BCS checkpoint classes are invalid")
    if any(type(checkpoint[field]) is not int or checkpoint[field] != expected for field, expected in (("score_min", SCORE_MIN), ("score_max", SCORE_MAX), ("score_base", SCORE_BASE), ("score_step", SCORE_STEP), ("num_classes", NUM_CLASSES), ("num_thresholds", NUM_THRESHOLDS))):
        raise BCSCheckpointLoadError("BCS checkpoint scale is invalid")
    if not _valid_digest(checkpoint["snapshot_identity"], 64) or not _valid_digest(checkpoint["dataset_manifest_digest"], 64) or not _valid_digest(checkpoint["config_sha256"], 64) or not _valid_digest(checkpoint["run_id"], 32):
        raise BCSCheckpointLoadError("BCS checkpoint lineage is invalid")
    if checkpoint["observed_classes"] != list(BCS_CLASS_SCORES) or checkpoint["missing_classes"] != []:
        raise BCSCheckpointLoadError("BCS checkpoint must cover all five categories")
    if checkpoint["source_identity_scheme"] != "local-path-sha256-v1" or checkpoint["source_mapping"] != dict(LOCAL_BCS_MAPPING.entries):
        raise BCSCheckpointLoadError("BCS checkpoint source lineage is invalid")
    config = checkpoint["config"]
    if not isinstance(config, dict) or type(config.get("imgsz")) is not int or config["imgsz"] <= 0:
        raise BCSCheckpointLoadError("BCS checkpoint configuration is invalid")
    provenance = checkpoint["provenance"]
    manifest = provenance.get("dataset_manifest") if isinstance(provenance, dict) else None
    if not isinstance(manifest, dict) or provenance.get("run_id") != checkpoint["run_id"] or provenance.get("config_sha256") != checkpoint["config_sha256"] or provenance.get("domain_id") != BCS_DOMAIN_ID or provenance.get("source_schema") != checkpoint["source_schema"] or provenance.get("identity_scheme") != checkpoint["source_identity_scheme"] or provenance.get("mapping") != checkpoint["source_mapping"] or manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION or manifest.get("source_schema") != checkpoint["source_schema"] or manifest.get("identity_scheme") != checkpoint["source_identity_scheme"] or manifest.get("mapping") != checkpoint["source_mapping"] or manifest.get("observed_classes") != list(BCS_CLASS_SCORES) or manifest.get("missing_classes") != [] or manifest.get("split_identity") != checkpoint["snapshot_identity"] or manifest.get("sha256") != checkpoint["dataset_manifest_digest"]:
        raise BCSCheckpointLoadError("BCS checkpoint source lineage is invalid")
    state = checkpoint["model_state_dict"]
    if not isinstance(state, Mapping) or not state or any(type(key) is not str or not isinstance(value, torch.Tensor) for key, value in state.items()):
        raise BCSCheckpointLoadError("BCS checkpoint model state is invalid")
    return checkpoint


def load_bcs_model(
    checkpoint_path: str | os.PathLike[str],
    device: str | torch.device = "cpu",
    *,
    expected_sha256: str | None = None,
    approved_roots: Iterable[Path] | None = None,
) -> LoadedBCSModel:
    try:
        if not _valid_digest(expected_sha256, 64):
            raise BCSCheckpointLoadError("BCS checkpoint digest is required and invalid")
        loaded = load_checkpoint_bytes(
            checkpoint_path,
            approved_roots=(CHECKPOINT_ROOT,) if approved_roots is None else approved_roots,
            expected_sha256=expected_sha256,
            require_checkpoint_set=True,
        )
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise BCSCheckpointUnavailableError("requested CUDA device is unavailable")
        return _build_loaded_bcs_model(loaded, resolved)
    except CheckpointByteUnavailableError:
        raise BCSCheckpointUnavailableError("BCS checkpoint is unavailable") from None
    except CheckpointByteError as error:
        raise BCSCheckpointLoadError(str(error)) from None
    except BCSCheckpointError:
        raise
    except SafePathError:
        raise BCSCheckpointLoadError("BCS checkpoint path is unsafe") from None
    except Exception:
        raise BCSCheckpointLoadError("BCS checkpoint could not be loaded safely") from None


def _build_loaded_bcs_model(loaded: CheckpointBytes, device: torch.device) -> LoadedBCSModel:
    try:
        checkpoint = _validate_checkpoint(loaded.payload)
        model = BCSOrdinalModel(pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device).eval()
    except BCSCheckpointError:
        raise
    except Exception:
        raise BCSCheckpointLoadError("BCS checkpoint model architecture is invalid") from None
    lineage = BCSLineageMetadata(
        checkpoint_schema_version=checkpoint["checkpoint_schema_version"],
        domain_id=checkpoint["domain_id"],
        snapshot_schema=checkpoint["snapshot_schema"],
        snapshot_identity=checkpoint["snapshot_identity"],
        dataset_manifest_digest=checkpoint["dataset_manifest_digest"],
        run_id=checkpoint["run_id"],
        source_schema=LOCAL_SOURCE_SCHEMA,
        source_identity_scheme=checkpoint["source_identity_scheme"],
        source_mapping=tuple(sorted(checkpoint["source_mapping"].items())),
        observed_classes=tuple(BCS_CLASS_SCORES),
        missing_classes=(),
    )
    return LoadedBCSModel(
        model,
        checkpoint["config"]["imgsz"],
        device,
        lineage,
        loaded.sha256,
        checkpoint,
    )
