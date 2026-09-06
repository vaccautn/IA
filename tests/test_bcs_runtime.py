from __future__ import annotations

import threading
import hashlib
from pathlib import Path

import pytest

from vacca_api.bcs_runtime import (
    BCSRuntime,
    BCSRuntimeStatus,
    BCSRuntimeUnavailableError,
)


def _service(value: str = "service") -> object:
    return value


_DIGEST = "a" * 64


def _configured(tmp_path: Path, name: str = "model.pt") -> tuple[dict[str, str], Path]:
    path = tmp_path / name
    path.write_bytes(b"trusted checkpoint bytes")
    return {
        "VACCA_BCS_CHECKPOINT": str(path),
        "VACCA_BCS_CHECKPOINT_SHA256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }, path


def test_missing_or_blank_checkpoint_is_unconfigured_without_loading() -> None:
    calls: list[tuple[object, object]] = []
    for value in (None, "", "   "):
        runtime = BCSRuntime({"VACCA_BCS_CHECKPOINT": value}, loader=calls.append)
        assert runtime.status == BCSRuntimeStatus.UNCONFIGURED
        with pytest.raises(BCSRuntimeUnavailableError):
            runtime.get_service()
    assert calls == []


def test_configured_runtime_is_not_loaded_until_explicit_get_and_passes_device(tmp_path: Path) -> None:
    calls: list[tuple[object, object]] = []
    environment, path = _configured(tmp_path)
    environment["VACCA_BCS_DEVICE"] = "cuda:7"
    runtime = BCSRuntime(
        environment,
        loader=lambda path, *, device, expected_sha256: calls.append(
            (path, device, expected_sha256)
        ) or "loaded",
        service_factory=lambda loaded: ("service", loaded),
        checkpoint_root=tmp_path,
    )
    assert runtime.status == "not_loaded"
    assert calls == []
    assert runtime.get_service() == ("service", "loaded")
    assert runtime.status == "ready"
    assert calls == [(str(path), "cuda:7", environment["VACCA_BCS_CHECKPOINT_SHA256"])]


def test_missing_device_defaults_to_cpu(tmp_path: Path) -> None:
    calls: list[tuple[object, object]] = []
    environment, path = _configured(tmp_path)
    runtime = BCSRuntime(
        environment,
        loader=lambda path, *, device, expected_sha256: calls.append(
            (path, device, expected_sha256)
        ) or "loaded",
        service_factory=lambda loaded: loaded,
        checkpoint_root=tmp_path,
    )
    assert runtime.get_service() == "loaded"
    assert calls == [(str(path), "cpu", environment["VACCA_BCS_CHECKPOINT_SHA256"])]


def test_success_is_cached_and_status_does_not_load(tmp_path: Path) -> None:
    calls: list[int] = []
    environment, _ = _configured(tmp_path)
    runtime = BCSRuntime(
        environment,
        loader=lambda path, *, device, expected_sha256: calls.append(1) or object(),
        service_factory=lambda loaded: _service(),
        checkpoint_root=tmp_path,
    )
    assert runtime.status == "not_loaded"
    assert runtime.status == "not_loaded"
    assert runtime.get_service() == runtime.get_service() == "service"
    assert calls == [1]


def test_failure_is_cached_as_unavailable_and_sanitized(tmp_path: Path) -> None:
    calls: list[int] = []
    environment, _ = _configured(tmp_path)

    def fail(path, *, device, expected_sha256):
        calls.append(1)
        raise RuntimeError("secret checkpoint path and payload")

    runtime = BCSRuntime(
        environment,
        loader=fail,
        checkpoint_root=tmp_path,
    )
    for _ in range(2):
        with pytest.raises(BCSRuntimeUnavailableError) as failure:
            runtime.get_service()
        assert "secret" not in str(failure.value)
    assert runtime.status == "unavailable"
    assert runtime.failure is not None
    assert "secret" not in repr(runtime.failure)
    assert calls == [1]


def test_load_failure_logs_safe_event_and_exception_type(tmp_path: Path, caplog) -> None:
    environment, checkpoint = _configured(tmp_path)

    def fail(path, *, device, expected_sha256):
        raise RuntimeError(f"secret checkpoint {checkpoint}")

    runtime = BCSRuntime(environment, loader=fail, checkpoint_root=tmp_path)
    with caplog.at_level("ERROR", logger="vacca_api.bcs_runtime"):
        with pytest.raises(BCSRuntimeUnavailableError):
            runtime.get_service()

    assert any("BCS checkpoint_load failure: RuntimeError" in message for message in caplog.messages)
    assert all("secret" not in message for message in caplog.messages)
    assert all(str(checkpoint) not in message for message in caplog.messages)


def test_configuration_failure_logs_safe_event_and_exception_type(tmp_path: Path, caplog) -> None:
    checkpoint = tmp_path / "outside.pt"
    environment = {
        "VACCA_BCS_CHECKPOINT": str(checkpoint),
        "VACCA_BCS_CHECKPOINT_SHA256": _DIGEST,
    }
    with caplog.at_level("ERROR", logger="vacca_api.bcs_runtime"):
        runtime = BCSRuntime(environment, loader=lambda *args, **kwargs: pytest.fail())

    assert runtime.status == BCSRuntimeStatus.UNAVAILABLE
    assert any("BCS configuration failure: SafePathError" in message for message in caplog.messages)
    assert all(str(checkpoint) not in message for message in caplog.messages)


def test_concurrent_get_service_loads_exactly_once(tmp_path: Path) -> None:
    calls = 0
    lock = threading.Lock()
    start_barrier = threading.Barrier(8)
    loader_entered = threading.Event()
    release_loader = threading.Event()
    environment, _ = _configured(tmp_path)

    def loader(path, *, device, expected_sha256):
        nonlocal calls
        with lock:
            calls += 1
        loader_entered.set()
        if not release_loader.wait(timeout=2):
            raise AssertionError("loader was not released")
        return "loaded"

    runtime = BCSRuntime(
        environment,
        loader=loader,
        service_factory=lambda loaded: _service(loaded),
        checkpoint_root=tmp_path,
    )
    results: list[object] = []
    failures: list[BaseException] = []

    def worker() -> None:
        try:
            start_barrier.wait(timeout=2)
            results.append(runtime.get_service())
        except BaseException as exc:
            failures.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    try:
        for thread in threads:
            thread.start()
        assert loader_entered.wait(timeout=2)
    finally:
        release_loader.set()
        for thread in threads:
            thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert failures == []
    assert results == ["loaded"] * 8
    assert calls == 1


def test_reset_reads_new_configuration_and_allows_one_new_load(tmp_path: Path) -> None:
    calls: list[str] = []
    old_environment, old_path = _configured(tmp_path, "old.pt")
    new_environment, new_path = _configured(tmp_path, "new.pt")

    def loader(path, *, device, expected_sha256):
        calls.append(path)
        return path

    runtime = BCSRuntime(
        old_environment,
        loader=loader,
        service_factory=lambda loaded: loaded,
        checkpoint_root=tmp_path,
    )
    assert runtime.get_service() == str(old_path)
    runtime.reset({**new_environment, "VACCA_BCS_DEVICE": "cpu"})
    assert runtime.status == "not_loaded"
    assert runtime.get_service() == str(new_path)
    assert calls == [str(old_path), str(new_path)]


def test_missing_or_malformed_checkpoint_digest_is_unavailable_without_loading() -> None:
    calls: list[object] = []
    for digest in (None, "", "A" * 64, "not-a-digest"):
        runtime = BCSRuntime(
            {"VACCA_BCS_CHECKPOINT": "model.pt", "VACCA_BCS_CHECKPOINT_SHA256": digest},
            loader=lambda *args, **kwargs: calls.append(True),
        )
        assert runtime.status == BCSRuntimeStatus.UNAVAILABLE
        with pytest.raises(BCSRuntimeUnavailableError):
            runtime.get_service()
    assert calls == []


def test_external_checkpoint_path_is_unavailable_without_loading() -> None:
    calls: list[object] = []
    runtime = BCSRuntime(
        {
            "VACCA_BCS_CHECKPOINT": "/private/model.pt",
            "VACCA_BCS_CHECKPOINT_SHA256": _DIGEST,
        },
        loader=lambda *args, **kwargs: calls.append(True),
    )
    assert runtime.status == BCSRuntimeStatus.UNAVAILABLE
    with pytest.raises(BCSRuntimeUnavailableError):
        runtime.get_service()
    assert calls == []


def test_runtime_module_has_no_detection_dependency() -> None:
    import vacca_api.bcs_runtime as runtime_module

    assert not hasattr(runtime_module, "get_detector")
