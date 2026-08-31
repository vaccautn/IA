"""Lazy, isolated runtime state for the optional BCS capability."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from vacca_bcs.serving import (
    BCSCheckpointLoadError,
    BCSCheckpointUnavailableError,
    BCSInferenceService,
    LoadedBCSModel,
    load_bcs_model,
)

_CHECKPOINT_ENV = "VACCA_BCS_CHECKPOINT"
_DEVICE_ENV = "VACCA_BCS_DEVICE"


class BCSRuntimeStatus(str, Enum):
    UNCONFIGURED = "unconfigured"
    NOT_LOADED = "not_loaded"
    READY = "ready"
    UNAVAILABLE = "unavailable"


class BCSRuntimeUnavailableError(RuntimeError):
    """Raised when the optional BCS capability cannot serve a request."""


@dataclass(frozen=True, slots=True)
class BCSRuntimeFailure:
    """Sanitized failure category and reason safe for status inspection."""

    category: str
    reason: str


Loader = Callable[..., LoadedBCSModel]
ServiceFactory = Callable[[LoadedBCSModel], Any]


class BCSRuntime:
    """Own one lazily initialized BCS service and its cached outcome."""

    def __init__(
        self,
        environment: Mapping[str, str | None] | None = None,
        *,
        loader: Loader | None = None,
        service_factory: ServiceFactory | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._loader = load_bcs_model if loader is None else loader
        self._service_factory = (
            BCSInferenceService if service_factory is None else service_factory
        )
        self._service: Any = None
        self._failure: BCSRuntimeFailure | None = None
        self._checkpoint: str | None = None
        self._device: Any = "cpu"
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

    def get_service(self) -> Any:
        """Load once on demand; cache both success and failure under one lock."""
        with self._lock:
            if self._status is BCSRuntimeStatus.READY:
                return self._service
            if self._status is BCSRuntimeStatus.UNCONFIGURED:
                raise BCSRuntimeUnavailableError("BCS capability is unconfigured")
            if self._status is BCSRuntimeStatus.UNAVAILABLE:
                raise BCSRuntimeUnavailableError("BCS capability is unavailable")

            try:
                loaded = self._loader(self._checkpoint, device=self._device)
            except BCSCheckpointUnavailableError:
                self._cache_failure("checkpoint_unavailable", "BCS checkpoint is unavailable")
                raise BCSRuntimeUnavailableError("BCS capability is unavailable") from None
            except BCSCheckpointLoadError:
                self._cache_failure("checkpoint_load", "BCS checkpoint could not be loaded")
                raise BCSRuntimeUnavailableError("BCS capability is unavailable") from None
            except Exception:
                self._cache_failure("checkpoint_load", "BCS checkpoint could not be loaded")
                raise BCSRuntimeUnavailableError("BCS capability is unavailable") from None

            try:
                self._service = self._service_factory(loaded)
            except Exception:
                self._cache_failure("service_factory", "BCS service could not be initialized")
                raise BCSRuntimeUnavailableError("BCS capability is unavailable") from None
            self._status = BCSRuntimeStatus.READY
            return self._service

    def reset(self, environment: Mapping[str, str | None] | None = None) -> None:
        """Explicitly discard the cached outcome and read a new configuration."""
        with self._lock:
            self._service = None
            self._failure = None
            self._checkpoint = None
            self._device = "cpu"
            self._status = BCSRuntimeStatus.UNCONFIGURED
            self._configure(environment if environment is not None else os.environ)

    def _configure(self, environment: Mapping[str, str | None]) -> None:
        try:
            checkpoint = environment.get(_CHECKPOINT_ENV)
            device = environment.get(_DEVICE_ENV, "cpu")
        except Exception:
            self._cache_failure("configuration", "BCS environment configuration is invalid")
            return
        if checkpoint is None or (isinstance(checkpoint, str) and not checkpoint.strip()):
            return
        if not isinstance(checkpoint, str):
            self._cache_failure("configuration", "BCS environment configuration is invalid")
            return
        self._checkpoint = checkpoint
        self._device = device
        self._status = BCSRuntimeStatus.NOT_LOADED

    def _cache_failure(self, category: str, reason: str) -> None:
        self._failure = BCSRuntimeFailure(category, reason)
        self._status = BCSRuntimeStatus.UNAVAILABLE


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
