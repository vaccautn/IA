"""Ordinal Body Condition Score components for VACCA.

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
        CORALModel,
        coral_loss,
        encode_levels,
        predict,
    )
    from .serving import (
        BCSLineageMetadata,
        LoadedBCSModel,
        load_bcs_model,
    )


__all__ = [
    "BCSFolderDataset",
    "BCSOrdinalModel",
    "CLASS_NAMES",
    "CORALModel",
    "Letterbox",
    "coral_loss",
    "encode_levels",
    "predict",
    "BCSLineageMetadata",
    "LoadedBCSModel",
    "load_bcs_model",
]

_LAZY_EXPORTS = {
    "BCSFolderDataset": (".dataset", "BCSFolderDataset"),
    "Letterbox": (".dataset", "Letterbox"),
    "BCSOrdinalModel": (".model", "BCSOrdinalModel"),
    "CORALModel": (".model", "CORALModel"),
    "coral_loss": (".model", "coral_loss"),
    "encode_levels": (".model", "encode_levels"),
    "predict": (".model", "predict"),
    "BCSLineageMetadata": (".serving", "BCSLineageMetadata"),
    "LoadedBCSModel": (".serving", "LoadedBCSModel"),
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
