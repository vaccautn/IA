from __future__ import annotations

import threading
import time

import pytest

from vacca_api.bcs_runtime import (
    BCSRuntime,
    BCSRuntimeStatus,
    BCSRuntimeUnavailableError,
)


def _service(value: str = "service") -> object:
    return value


def test_missing_or_blank_checkpoint_is_unconfigured_without_loading() -> None:
    calls: list[tuple[object, object]] = []
    for value in (None, "", "   "):
        runtime = BCSRuntime({"VACCA_BCS_CHECKPOINT": value}, loader=calls.append)
        assert runtime.status == BCSRuntimeStatus.UNCONFIGURED
        with pytest.raises(BCSRuntimeUnavailableError):
            runtime.get_service()
    assert calls == []


def test_configured_runtime_is_not_loaded_until_explicit_get_and_passes_device() -> None:
    calls: list[tuple[object, object]] = []
    runtime = BCSRuntime(
        {"VACCA_BCS_CHECKPOINT": "/private/model.pt", "VACCA_BCS_DEVICE": "cuda:7"},
        loader=lambda path, *, device: calls.append((path, device)) or "loaded",
        service_factory=lambda loaded: ("service", loaded),
    )
    assert runtime.status == "not_loaded"
    assert calls == []
    assert runtime.get_service() == ("service", "loaded")
    assert runtime.status == "ready"
    assert calls == [("/private/model.pt", "cuda:7")]


def test_missing_device_defaults_to_cpu() -> None:
    calls: list[tuple[object, object]] = []
    runtime = BCSRuntime(
        {"VACCA_BCS_CHECKPOINT": "model.pt"},
        loader=lambda path, *, device: calls.append((path, device)) or "loaded",
        service_factory=lambda loaded: loaded,
    )
    assert runtime.get_service() == "loaded"
    assert calls == [("model.pt", "cpu")]


def test_success_is_cached_and_status_does_not_load() -> None:
    calls: list[int] = []
    runtime = BCSRuntime(
        {"VACCA_BCS_CHECKPOINT": "model.pt"},
        loader=lambda path, *, device: calls.append(1) or object(),
        service_factory=lambda loaded: _service(),
    )
    assert runtime.status == "not_loaded"
    assert runtime.status == "not_loaded"
    assert runtime.get_service() == runtime.get_service() == "service"
    assert calls == [1]


def test_failure_is_cached_as_unavailable_and_sanitized() -> None:
    calls: list[int] = []

    def fail(path, *, device):
        calls.append(1)
        raise RuntimeError("secret checkpoint path and payload")

    runtime = BCSRuntime({"VACCA_BCS_CHECKPOINT": "/secret/model.pt"}, loader=fail)
    for _ in range(2):
        with pytest.raises(BCSRuntimeUnavailableError) as failure:
            runtime.get_service()
        assert "secret" not in str(failure.value)
    assert runtime.status == "unavailable"
    assert runtime.failure is not None
    assert "secret" not in repr(runtime.failure)
    assert calls == [1]


def test_concurrent_get_service_loads_exactly_once() -> None:
    calls = 0
    lock = threading.Lock()

    def loader(path, *, device):
        nonlocal calls
        with lock:
            calls += 1
        time.sleep(0.02)
        return "loaded"

    runtime = BCSRuntime(
        {"VACCA_BCS_CHECKPOINT": "model.pt"},
        loader=loader,
        service_factory=lambda loaded: _service(loaded),
    )
    results: list[object] = []
    threads = [threading.Thread(target=lambda: results.append(runtime.get_service())) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == ["loaded"] * 8
    assert calls == 1


def test_reset_reads_new_configuration_and_allows_one_new_load() -> None:
    calls: list[str] = []

    def loader(path, *, device):
        calls.append(path)
        return path

    runtime = BCSRuntime(
        {"VACCA_BCS_CHECKPOINT": "old.pt"},
        loader=loader,
        service_factory=lambda loaded: loaded,
    )
    assert runtime.get_service() == "old.pt"
    runtime.reset({"VACCA_BCS_CHECKPOINT": "new.pt", "VACCA_BCS_DEVICE": "cpu"})
    assert runtime.status == "not_loaded"
    assert runtime.get_service() == "new.pt"
    assert calls == ["old.pt", "new.pt"]


def test_runtime_module_has_no_detection_dependency() -> None:
    import vacca_api.bcs_runtime as runtime_module

    assert not hasattr(runtime_module, "get_detector")
