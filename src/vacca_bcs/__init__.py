"""BCS category components for VACCA.

The training dataset and model exports are loaded lazily so lightweight
constants and builder imports do not import Torch or Torchvision.
"""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .constants import CLASS_NAMES

if TYPE_CHECKING:
    from .dataset import BCSFolderDataset, Letterbox
    from .model import (
        BCSOrdinalModel,
        coral_loss,
        encode_levels,
        predict,
    )
    from .serving import (
        BCSInferenceError,
        BCSInferenceExecutionError,
        BCSInferenceInputError,
        BCSInferenceResult,
        BCSInferenceService,
        BCSLineageMetadata,
        LoadedBCSModel,
        infer_bcs,
        load_bcs_model,
    )


__all__ = [
    "BCSFolderDataset",
    "BCSOrdinalModel",
    "CLASS_NAMES",
    "Letterbox",
    "coral_loss",
    "encode_levels",
    "predict",
    "BCSLineageMetadata",
    "BCSInferenceError",
    "BCSInferenceExecutionError",
    "BCSInferenceInputError",
    "BCSInferenceResult",
    "BCSInferenceService",
    "LoadedBCSModel",
    "infer_bcs",
    "load_bcs_model",
]

_LAZY_EXPORTS = {
    "BCSFolderDataset": (".dataset", "BCSFolderDataset"),
    "Letterbox": (".dataset", "Letterbox"),
    "BCSOrdinalModel": (".model", "BCSOrdinalModel"),
    "coral_loss": (".model", "coral_loss"),
    "encode_levels": (".model", "encode_levels"),
    "predict": (".model", "predict"),
    "BCSLineageMetadata": (".serving", "BCSLineageMetadata"),
    "BCSInferenceError": (".serving", "BCSInferenceError"),
    "BCSInferenceExecutionError": (".serving", "BCSInferenceExecutionError"),
    "BCSInferenceInputError": (".serving", "BCSInferenceInputError"),
    "BCSInferenceResult": (".serving", "BCSInferenceResult"),
    "BCSInferenceService": (".serving", "BCSInferenceService"),
    "LoadedBCSModel": (".serving", "LoadedBCSModel"),
    "infer_bcs": (".serving", "infer_bcs"),
    "load_bcs_model": (".serving", "load_bcs_model"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, export_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = getattr(module, export_name)
    globals()[name] = value
    return value
