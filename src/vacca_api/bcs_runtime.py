"""Lazy, isolated runtime state for the optional BCS capability."""

from __future__ import annotations

import os
import string
import threading
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from vacca_bcs.serving import (
    BCSCheckpointLoadError,
    BCSCheckpointUnavailableError,
    BCSInferenceService,
    CHECKPOINT_ROOT,
    LoadedBCSModel,
    load_bcs_model,
)
from vacca_bcs.path_safety import SafePathError, safe_path
from vacca_bcs.serving_package import (
    PRIVATE_PACKAGE_ID,
    PRIVATE_PACKAGE_ROOT,
    ServingPackageError,
    find_recoverable_backups,
    load_installed_bcs_model,
    validate_installed_package,
)

_CHECKPOINT_ENV = "VACCA_BCS_CHECKPOINT"
_CHECKPOINT_SHA256_ENV = "VACCA_BCS_CHECKPOINT_SHA256"
_DEVICE_ENV = "VACCA_BCS_DEVICE"
_DISABLED_ENV = "VACCA_BCS_DISABLED"
logger = logging.getLogger(__name__)


class BCSRuntimeStatus(str, Enum):
    UNCONFIGURED = "unconfigured"
    NOT_LOADED = "not_loaded"
    NOT_INSTALLED = "not_installed"
    READY = "ready"
    UNAVAILABLE = "unavailable"


class BCSModelStatus(str, Enum):
    NONE = "none"
    EXPERIMENTAL_NOT_APPROVED = "experimental_not_approved"
    EXTERNAL_UNCLASSIFIED = "external_unclassified"


class BCSRuntimeUnavailableError(RuntimeError):
    """Raised when the optional BCS capability cannot serve a request."""


@dataclass(frozen=True, slots=True)
class BCSRuntimeFailure:
    """Sanitized failure category and reason safe for status inspection."""

    category: str
    reason: str


Loader = Callable[..., LoadedBCSModel]
ServiceFactory = Callable[[LoadedBCSModel], Any]


def _recovery_hint() -> str:
    try:
        available = bool(find_recoverable_backups())
    except (OSError, ServingPackageError):
        available = False
    if not available:
        return ""
    return "; a retained recovery backup is available; run --recover-backup <backup-name> after stopping the API"


class BCSRuntime:
    """Own one lazily initialized BCS service and its cached outcome."""

    def __init__(
        self,
        environment: Mapping[str, str | None] | None = None,
        *,
        loader: Loader | None = None,
        service_factory: ServiceFactory | None = None,
        checkpoint_root: Path | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._custom_loader = loader is not None
        self._loader = load_bcs_model if loader is None else loader
        self._service_factory = (
            BCSInferenceService if service_factory is None else service_factory
        )
        self._checkpoint_root = CHECKPOINT_ROOT if checkpoint_root is None else Path(checkpoint_root)
        self._service: Any = None
        self._failure: BCSRuntimeFailure | None = None
        self._checkpoint: str | None = None
        self._checkpoint_sha256: str | None = None
        self._device: Any = "cpu"
        self._model_status: BCSModelStatus = BCSModelStatus.NONE
        self._package_id: str | None = None
        self._status = BCSRuntimeStatus.UNCONFIGURED
        self._configure(environment if environment is not None else os.environ)

    @property
    def status(self) -> BCSRuntimeStatus:
        with self._lock:
            return self._status

    @property
    def failure(self) -> BCSRuntimeFailure | None:
        with self._lock:
            return self._failure

    @property
    def model_status(self) -> BCSModelStatus:
        with self._lock:
            return self._model_status

    @property
    def package_id(self) -> str | None:
        with self._lock:
            return self._package_id

    def get_service(self) -> Any:
        """Load once on demand; cache both success and failure under one lock."""
        with self._lock:
            if self._status is BCSRuntimeStatus.READY:
                return self._service
            if self._status is BCSRuntimeStatus.UNCONFIGURED:
                raise BCSRuntimeUnavailableError("BCS capability is unconfigured")
            if self._status in {
                BCSRuntimeStatus.NOT_INSTALLED,
                BCSRuntimeStatus.UNAVAILABLE,
            }:
                raise BCSRuntimeUnavailableError("BCS capability is unavailable")

            try:
                loaded = self._loader(
                    self._checkpoint,
                    device=self._device,
                    expected_sha256=self._checkpoint_sha256,
                )
            except BCSCheckpointUnavailableError as exc:
                self._cache_failure(
                    "checkpoint_unavailable",
                    "BCS checkpoint is unavailable",
                    type(exc).__name__,
                )
                raise BCSRuntimeUnavailableError("BCS capability is unavailable") from None
            except BCSCheckpointLoadError as exc:
                self._cache_failure(
                    "checkpoint_load",
                    "BCS checkpoint could not be loaded",
                    type(exc).__name__,
                )
                raise BCSRuntimeUnavailableError("BCS capability is unavailable") from None
            except ServingPackageError as exc:
                self._cache_failure(
                    "package_load",
                    "BCS serving package could not be loaded",
                    type(exc).__name__,
                )
                raise BCSRuntimeUnavailableError("BCS capability is unavailable") from None
            except Exception as exc:
                self._cache_failure(
                    "checkpoint_load",
                    "BCS checkpoint could not be loaded",
                    type(exc).__name__,
                )
                raise BCSRuntimeUnavailableError("BCS capability is unavailable") from None

            try:
                self._service = self._service_factory(loaded)
            except Exception as exc:
                self._cache_failure(
                    "service_factory",
                    "BCS service could not be initialized",
                    type(exc).__name__,
                )
                raise BCSRuntimeUnavailableError("BCS capability is unavailable") from None
            self._status = BCSRuntimeStatus.READY
            return self._service

    def reset(self, environment: Mapping[str, str | None] | None = None) -> None:
        """Explicitly discard the cached outcome and read a new configuration."""
        with self._lock:
            self._service = None
            self._failure = None
            self._checkpoint = None
            self._checkpoint_sha256 = None
            self._device = "cpu"
            self._model_status = BCSModelStatus.NONE
            self._package_id = None
            if not self._custom_loader:
                self._loader = load_bcs_model
            self._status = BCSRuntimeStatus.UNCONFIGURED
            self._configure(environment if environment is not None else os.environ)

    def _configure(self, environment: Mapping[str, str | None]) -> None:
        try:
            checkpoint = environment.get(_CHECKPOINT_ENV)
            checkpoint_sha256 = environment.get(_CHECKPOINT_SHA256_ENV)
            device = environment.get(_DEVICE_ENV, "cpu")
            disabled = environment.get(_DISABLED_ENV)
        except Exception as exc:
            self._cache_failure(
                "configuration",
                "BCS environment configuration is invalid",
                type(exc).__name__,
            )
            return
        if disabled is not None:
            if disabled != "1":
                self._cache_failure("configuration", "BCS disable configuration is invalid")
                return
            self._failure = BCSRuntimeFailure(
                "disabled", "BCS capability is disabled by emergency configuration"
            )
            self._status = BCSRuntimeStatus.UNCONFIGURED
            return
        has_checkpoint = checkpoint is not None
        has_digest = checkpoint_sha256 is not None
        if not has_checkpoint and not has_digest:
            self._model_status = BCSModelStatus.EXPERIMENTAL_NOT_APPROVED
            self._package_id = PRIVATE_PACKAGE_ID
            self._device = device
            if not self._custom_loader:
                if not PRIVATE_PACKAGE_ROOT.is_dir() or PRIVATE_PACKAGE_ROOT.is_symlink():
                    self._cache_failure(
                        "package_not_installed",
                        f"BCS serving package is not installed{_recovery_hint()}",
                        status=BCSRuntimeStatus.NOT_INSTALLED,
                    )
                    return
                try:
                    validate_installed_package()
                except ServingPackageError as exc:
                    self._cache_failure(
                        "package_unavailable",
                        f"BCS serving package is unavailable{_recovery_hint()}",
                        type(exc).__name__,
                    )
                    return
            self._status = BCSRuntimeStatus.NOT_LOADED
            self._loader = self._loader if self._custom_loader else load_installed_bcs_model
            return
        if not has_checkpoint or not has_digest:
            self._cache_failure("configuration", "BCS environment configuration is invalid")
            return
        if (
            not isinstance(checkpoint, str)
            or not isinstance(checkpoint_sha256, str)
            or len(checkpoint_sha256) != 64
            or checkpoint_sha256 != checkpoint_sha256.lower()
            or any(character not in string.hexdigits[:16] for character in checkpoint_sha256)
        ):
            self._cache_failure("configuration", "BCS environment configuration is invalid")
            return
        self._checkpoint = checkpoint
        self._checkpoint_sha256 = checkpoint_sha256
        try:
            self._checkpoint = str(
                safe_path(
                    checkpoint,
                    base=self._checkpoint_root.parent,
                    approved_roots=(self._checkpoint_root,),
                    allow_missing_final=False,
                    require_file=True,
                )
            )
        except SafePathError as exc:
            self._cache_failure(
                "configuration",
                "BCS checkpoint path is invalid",
                type(exc).__name__,
            )
            return
        self._device = device
        self._model_status = BCSModelStatus.EXTERNAL_UNCLASSIFIED
        self._package_id = None
        self._status = BCSRuntimeStatus.NOT_LOADED

    def _cache_failure(
        self,
        category: str,
        reason: str,
        exception_type: str = "BCSConfigurationError",
        *,
        status: BCSRuntimeStatus = BCSRuntimeStatus.UNAVAILABLE,
    ) -> None:
        self._failure = BCSRuntimeFailure(category, reason)
        self._status = status
        logger.error("BCS %s failure: %s", category, exception_type)


_global_runtime: BCSRuntime | None = None
_global_lock = threading.Lock()


def get_bcs_runtime() -> BCSRuntime:
    """Return the process-owned runtime without triggering BCS loading."""
    global _global_runtime
    with _global_lock:
        if _global_runtime is None:
            _global_runtime = BCSRuntime()
        return _global_runtime


def reset_bcs_runtime(environment: Mapping[str, str | None] | None = None) -> None:
    """Explicitly replace the process-owned runtime, primarily for tests."""
    global _global_runtime
    with _global_lock:
        _global_runtime = BCSRuntime(environment)
