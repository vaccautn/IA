"""Fake parent-runtime integration and cancellation capacity characterization."""

import asyncio
import threading
import json
from io import BytesIO
from types import SimpleNamespace

import pytest

from vacca_api import main
from vacca_api.metrics import PrototypeMetrics
from PIL import Image
from vacca_video.artifacts import BCS_WRITE_ERROR, job_artifact
from vacca_video.bcs import VideoBCSCoordinator
from vacca_video.candidate import POLICY
from vacca_video.schemas import AnalysisSummaryResponse
from vacca_video.store import Store
from vacca_video.worker import Worker


def candidate_job(tmp_path, candidate=True):
    store = Store(tmp_path)
    store.add("fake", "fake.mp4")
    directory = tmp_path / "fake/result"
    directory.mkdir(parents=True)
    image = BytesIO()
    Image.new("RGB", (65, 49), (3, 37, 119)).save(image, format="PNG")
    (directory / "candidate.png").write_bytes(image.getvalue())
    selected = dict(artifact="candidate.png", encoding="PNG", frame_index=83, sample_ordinal=2,
                    timestamp_ms=16600, timestamp_source="decoder", detector_confidence=0.9,
                    target_index=0, original_bbox=dict(x1=10, y1=8, x2=30, y2=24),
                    image_width=65, image_height=49,
                    detector_model=dict(name="fake", sha256="fake-hash", confidence=0.25))
    summary = dict(status="completed", stop_reason="frame_limit", candidate=selected if candidate else None,
                   selection_policy=POLICY, bcs=dict(status="not_evaluated", reason="web_api_only"),
                   custom=None)
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (directory / "summary.js").write_text("old fixture", encoding="utf-8")
    (directory / "detections.jsonl").write_text('{"frame_index":83}\n', encoding="utf-8")
    return store, directory, image.getvalue()


@pytest.mark.parametrize("failure, expected", [(None, "completed"), ("busy", "busy"),
    ("unavailable", "unavailable"), ("invalid", "invalid_input"), ("failure", "failed")])
def test_worker_parent_result_optional_failures_preserve_video(tmp_path, monkeypatch, failure, expected):
    store, directory, payload = candidate_job(tmp_path)
    calls = []
    def infer(image):
        calls.append(image)
        if failure == "invalid":
            from vacca_bcs.serving import BCSInferenceInputError
            raise BCSInferenceInputError("private path must not leak")
        if failure == "failure":
            raise RuntimeError("private checkpoint path must not leak")
        return SimpleNamespace(bcs_category=4)
    def runtime():
        if failure == "unavailable":
            raise ImportError("optional package absent")
        return SimpleNamespace(get_service=lambda: SimpleNamespace(infer=infer),
                               model_status="experimental_not_approved", package_id="fake-package")
    monkeypatch.setattr(main, "get_bcs_runtime", runtime)
    monkeypatch.setattr(main.app.state, "metrics", PrototypeMetrics())
    async def run():
        gate = main.InferenceCapacityGate("bcs", acquisition_timeout=0.02)
        monkeypatch.setattr(main.app.state, "bcs_inference_gate", gate)
        if failure == "busy":
            await gate.acquire()
        coordinator = VideoBCSCoordinator(asyncio.get_running_loop(), main.evaluate_bcs)
        worker = Worker(store)
        worker.finish_video = coordinator.finish
        class Process:
            returncode = 0
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def poll(self): return 0
        monkeypatch.setattr("vacca_video.worker.subprocess.Popen", lambda *a, **kw: Process())
        update = store.update
        def record(job_id, status, error=None):
            update(job_id, status, error)
            worker.stop.set()
        monkeypatch.setattr(store, "update", record)
        try:
            await asyncio.wait_for(asyncio.to_thread(worker.loop), 3)
        finally:
            await coordinator.close()
            if failure == "busy":
                gate.release()
        assert store.get("fake")["status"] == "completed"
        assert store.get("fake")["error"] is None
    asyncio.run(run())
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["bcs"]["status"] == expected
    assert summary["candidate"]["frame_index"] == 83
    assert summary["custom"] is None
    assert (directory / "detections.jsonl").read_text() == '{"frame_index":83}\n'
    assert (directory / "summary.js").read_text() == "window.VACCA_SUMMARY = " + json.dumps(summary, ensure_ascii=True, allow_nan=False) + ";\n"
    assert calls == ([] if failure in {"busy", "unavailable"} else [payload])
    assert "private" not in json.dumps(summary["bcs"])
    assert main.app.state.metrics.snapshot()["bcs"]["client_rejections"] == 0
    if expected != "completed":
        assert "bcs_category" not in summary["bcs"]


@pytest.mark.parametrize("kind", ["none", "legacy", "failed", "cancelled", "bad_stop", "invalid_path", "wrong_size", "invalid_bytes", "wrong_encoding", "too_wide", "too_large", "symlink"])
def test_ineligible_old_or_invalid_candidates_never_infer(tmp_path, kind, monkeypatch):
    store, directory, payload = candidate_job(tmp_path)
    summary = json.loads((directory / "summary.json").read_text())
    if kind == "none":
        summary["candidate"] = None
    elif kind == "legacy":
        summary.pop("candidate")
    elif kind in {"failed", "cancelled"}:
        summary["status"] = kind
    elif kind == "bad_stop":
        summary["stop_reason"] = "error"
    elif kind == "invalid_path":
        summary["candidate"]["artifact"] = "../../secret.png"
    elif kind == "wrong_size":
        summary["candidate"]["image_width"] = 64
    elif kind == "invalid_bytes":
        (directory / "candidate.png").write_bytes(b"not an image")
    elif kind == "wrong_encoding":
        Image.new("RGB", (65, 49)).save(directory / "candidate.png", format="JPEG")
    elif kind == "too_wide":
        Image.new("RGB", (12001, 1)).save(directory / "candidate.png", format="PNG")
    elif kind == "too_large":
        (directory / "candidate.png").write_bytes(b"x" * (10 * 1024 * 1024 + 1))
    elif kind == "symlink":
        candidate = directory / "candidate.png"
        path_type = type(candidate)
        original = path_type.is_symlink
        monkeypatch.setattr(path_type, "is_symlink", lambda path: path == candidate or original(path))
    (directory / "summary.json").write_text(json.dumps(summary))
    async def never(*a, **kw): pytest.fail("Invalid candidate must not request BCS")
    async def run():
        coordinator = VideoBCSCoordinator(asyncio.get_running_loop(), never)
        assert await asyncio.to_thread(coordinator.finish, store.directory, "fake", threading.Event()) is None
        await coordinator.close()
    asyncio.run(run())
    result = json.loads((directory / "summary.json").read_text())
    if kind == "legacy":
        assert result == summary
    else:
        assert result["bcs"]["status"] in {"not_evaluated", "invalid_input"}


@pytest.mark.parametrize("replace_number", [1, 2])
def test_metadata_write_failure_is_not_claimed_as_saved(tmp_path, monkeypatch, replace_number):
    store, directory, _ = candidate_job(tmp_path)
    before = (directory / "summary.json").read_bytes()
    from vacca_video import artifacts
    replace = artifacts.os.replace
    attempts = 0
    def fail(*a, **kw):
        nonlocal attempts
        attempts += 1
        if attempts == replace_number:
            raise OSError("disk full")
        return replace(*a, **kw)
    monkeypatch.setattr("vacca_video.artifacts.os.replace", fail)
    async def evaluate(*a, **kw):
        return SimpleNamespace(bcs_category=4, model_status="experimental_not_approved", package_id="fake", inference_time_ms=1)
    async def run():
        coordinator = VideoBCSCoordinator(asyncio.get_running_loop(), evaluate)
        assert await asyncio.to_thread(coordinator.finish, store.directory, "fake", threading.Event()) == BCS_WRITE_ERROR
        await coordinator.close()
    asyncio.run(run())
    assert (directory / "summary.json").read_bytes() == before
    assert (directory / "summary.js").read_text() == "old fixture"
    assert not list(directory.glob(".bcs-*"))


def test_optional_summary_contract_preserves_old_null_and_extra():
    for value in ({"custom": None}, {"candidate": None, "bcs": None, "custom": "extension"}):
        assert AnalysisSummaryResponse.model_validate(value).model_dump(exclude_unset=True) == value


@pytest.mark.parametrize("job_id,name", [("../fake", "candidate.png"), ("fake", "../secret"), ("fake", "/absolute")])
def test_fixed_artifact_rejects_traversal(tmp_path, job_id, name):
    with pytest.raises(ValueError):
        job_artifact(tmp_path, job_id, name)


def test_shutdown_is_bounded_and_retains_native_gate(tmp_path, monkeypatch):
    store, directory, _ = candidate_job(tmp_path)
    entered, release = threading.Event(), threading.Event()
    def infer(payload):
        entered.set()
        assert release.wait(4)
        return SimpleNamespace(bcs_category=3)
    monkeypatch.setattr(main, "get_bcs_runtime", lambda: SimpleNamespace(get_service=lambda: SimpleNamespace(infer=infer)))
    async def run():
        gate = main.InferenceCapacityGate("bcs", acquisition_timeout=0.02)
        monkeypatch.setattr(main.app.state, "bcs_inference_gate", gate)
        coordinator = VideoBCSCoordinator(asyncio.get_running_loop(), main.evaluate_bcs)
        waiter = asyncio.create_task(asyncio.to_thread(coordinator.finish, store.directory, "fake", threading.Event()))
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            await asyncio.wait_for(coordinator.close(), 0.5)
            with pytest.raises(RuntimeError, match="interrumpió"):
                await asyncio.wait_for(waiter, 0.5)
            assert not coordinator.tasks
            with pytest.raises(main.InferenceCapacityBusyError):
                await gate.acquire()
            assert json.loads((directory / "summary.json").read_text())["bcs"]["status"] == "not_evaluated"
        finally:
            release.set()
        for _ in range(100):
            if not gate._work_threads:
                break
            await asyncio.sleep(0.01)
        assert not gate._work_threads
        await gate.acquire()
        gate.release()
    asyncio.run(run())


def test_principal_shutdown_does_not_join_worker_on_its_event_loop(tmp_path, monkeypatch):
    from vacca_video import service
    store, directory, _ = candidate_job(tmp_path)
    entered, release = threading.Event(), threading.Event()
    def infer(payload):
        entered.set()
        assert release.wait(4)
        return SimpleNamespace(bcs_category=3)
    monkeypatch.setattr(service, "check_runtime", lambda: None)
    monkeypatch.setattr(main, "get_detector", lambda: SimpleNamespace(gpu_available=False))
    monkeypatch.setattr(main, "get_bcs_runtime", lambda: SimpleNamespace(get_service=lambda: SimpleNamespace(infer=infer)))
    monkeypatch.setattr(main, "VideoResources", lambda: service.VideoResources(tmp_path))
    class Process:
        returncode = 0
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def poll(self): return 0
    monkeypatch.setattr("vacca_video.worker.subprocess.Popen", lambda *a, **kw: Process())
    async def run():
        gate = main.InferenceCapacityGate("bcs", acquisition_timeout=0.02)
        monkeypatch.setattr(main.app.state, "bcs_inference_gate", gate)
        lifecycle = main.app.router.lifespan_context(main.app)
        await lifecycle.__aenter__()
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            await asyncio.wait_for(lifecycle.__aexit__(None, None, None), 0.8)
            assert not main.app.state.video.worker.thread.is_alive()
            assert store.get("fake")["status"] == "failed"
            with pytest.raises(main.InferenceCapacityBusyError):
                await gate.acquire()
        finally:
            release.set()
        for _ in range(100):
            if not gate._work_threads:
                break
            await asyncio.sleep(0.01)
        assert not gate._work_threads
    asyncio.run(run())


def test_evaluation_timeout_is_not_published_as_late_success(tmp_path, monkeypatch):
    store, directory, _ = candidate_job(tmp_path)
    entered, release = threading.Event(), threading.Event()
    def infer(payload):
        entered.set()
        assert release.wait(3)
        return SimpleNamespace(bcs_category=3)
    monkeypatch.setattr(main, "get_bcs_runtime", lambda: SimpleNamespace(get_service=lambda: SimpleNamespace(infer=infer)))
    async def run():
        gate = main.InferenceCapacityGate("bcs", acquisition_timeout=0.01)
        monkeypatch.setattr(main.app.state, "bcs_inference_gate", gate)
        coordinator = VideoBCSCoordinator(asyncio.get_running_loop(), main.evaluate_bcs, max_wait=0.1)
        try:
            assert await asyncio.to_thread(coordinator.finish, store.directory, "fake", threading.Event()) is None
            assert entered.is_set()
            with pytest.raises(main.InferenceCapacityBusyError):
                await gate.acquire()
        finally:
            release.set()
        for _ in range(100):
            if not gate._work_threads:
                break
            await asyncio.sleep(0.01)
        assert not gate._work_threads
        await coordinator.close()
    asyncio.run(run())
    assert json.loads((directory / "summary.json").read_text())["bcs"] == dict(status="not_evaluated", reason="evaluation_timeout")


def test_result_lineage_is_preserved_without_checkpoint_paths(monkeypatch):
    lineage = SimpleNamespace(checkpoint_schema_version="fake-v1", domain_id="fake-domain",
        snapshot_schema="snapshot-v1", snapshot_identity="fake-identity", dataset_manifest_digest="fake-digest",
        run_id="fake-run", source_schema="source-v1", source_identity_scheme="identity-v1",
        observed_classes=(1, 2, 3, 4, 5), missing_classes=(), source_mapping=(("3.25", 1), ("4.25", 5)),
        checkpoint_path="private/not-public")
    monkeypatch.setattr(main, "get_bcs_runtime", lambda: SimpleNamespace(
        get_service=lambda: SimpleNamespace(infer=lambda _: SimpleNamespace(bcs_category=2, lineage=lineage))))
    monkeypatch.setattr(main.app.state, "bcs_inference_gate", main.InferenceCapacityGate("bcs"))
    result = asyncio.run(main.evaluate_bcs(b"fake", internal=True))
    assert result.lineage["source_mapping"] == {"3.25": 1, "4.25": 5}
    assert result.lineage["observed_classes"] == [1, 2, 3, 4, 5]
    assert "checkpoint_path" not in result.lineage
    assert "private" not in json.dumps(result.lineage)


def test_shared_parent_bcs_evaluates_exact_bytes_once(monkeypatch):
    calls = []
    runtime = SimpleNamespace(model_status="experimental_not_approved", package_id="fake-package",
                              get_service=lambda: SimpleNamespace(infer=lambda payload:
                                  calls.append(payload) or SimpleNamespace(bcs_category=4)))
    monkeypatch.setattr(main, "get_bcs_runtime", lambda: runtime)
    monkeypatch.setattr(main.app.state, "bcs_inference_gate", main.InferenceCapacityGate("bcs"))
    monkeypatch.setattr(main.app.state, "metrics", PrototypeMetrics())
    result = asyncio.run(main.evaluate_bcs(b"original full candidate", internal=True))
    assert calls == [b"original full candidate"]
    assert result.bcs_category == 4
    assert result.model_status == "experimental_not_approved"
    assert result.package_id == "fake-package"


def test_cancelled_bcs_waiter_keeps_gate_until_native_work_finishes(monkeypatch):
    async def run():
        entered, release = threading.Event(), threading.Event()
        def infer(payload):
            entered.set()
            assert release.wait(3)
            return SimpleNamespace(bcs_category=3)
        monkeypatch.setattr(main, "get_bcs_runtime", lambda: SimpleNamespace(
            get_service=lambda: SimpleNamespace(infer=infer)))
        gate = main.InferenceCapacityGate("bcs", acquisition_timeout=0.02)
        monkeypatch.setattr(main.app.state, "bcs_inference_gate", gate)
        task = asyncio.create_task(main.evaluate_bcs(b"fake", internal=True))
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            with pytest.raises(main.InferenceCapacityBusyError):
                await gate.acquire()
        finally:
            release.set()
        for _ in range(100):
            if not gate._work_threads:
                break
            await asyncio.sleep(0.01)
        assert not gate._work_threads
        await gate.acquire()
        gate.release()
    asyncio.run(run())
