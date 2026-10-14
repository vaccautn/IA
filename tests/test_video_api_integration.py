"""Principal API + video integration, using fake runtimes and no inference."""

import asyncio
import json
import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import run_api, run_video_web
from scripts.smoke_test_api import _BOUNDARY, _multipart_body
from vacca_api import main
from vacca_api.metrics import PrototypeMetrics
from vacca_video import service, webapp
from vacca_video.store import Store
from test_video_web import TestClient, submit


class FakeWorker:
    instances = []

    def __init__(self, store):
        self.store = store
        self.ready = threading.Event()
        self.stop = SimpleNamespace(set=Mock())
        self.thread = self
        self.start = Mock()
        self.join = Mock()
        self.instances.append(self)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(FakeWorker, "instances", [])
    monkeypatch.setattr(service, "Worker", FakeWorker)
    monkeypatch.setattr(service, "check_runtime", lambda: None)
    monkeypatch.setattr(main, "get_detector", lambda: SimpleNamespace(
        gpu_available=False, detect=lambda payload: ([], 8, 6, 1.25)
    ))
    bcs_service = SimpleNamespace(infer=lambda payload: SimpleNamespace(bcs_category=3))
    monkeypatch.setattr(main, "get_bcs_runtime", lambda: SimpleNamespace(
        status="ready", get_service=lambda: bcs_service, model_status=None
    ))
    monkeypatch.setattr(main.app.state, "metrics", PrototypeMetrics())
    monkeypatch.setattr(main.app.state, "detect_inference_gate", main.InferenceCapacityGate())
    monkeypatch.setattr(main.app.state, "bcs_inference_gate", main.InferenceCapacityGate())
    monkeypatch.setattr(webapp.shutil, "disk_usage", lambda path: SimpleNamespace(free=1024**3))
    return Path(service.os.environ["VACCA_VIDEO_DATA"])


@pytest.fixture
def client(runtime):
    with asyncio.Runner() as runner:
        lifecycle = main.app.router.lifespan_context(main.app)
        runner.run(lifecycle.__aenter__())
        try:
            yield TestClient(main.app, runner=runner)
        finally:
            runner.run(lifecycle.__aexit__(None, None, None))


def image_upload():
    from io import BytesIO
    from PIL import Image

    image = BytesIO()
    Image.new("RGB", (8, 6)).save(image, format="JPEG")
    return _multipart_body("image.jpg", image.getvalue(), "image/jpeg")


IMAGE_HEADERS = {"content-type": f"multipart/form-data; boundary={_BOUNDARY.decode()}"}


def test_principal_routes_and_video_contract_coexist(client):
    for path in ("/health", "/ui", "/docs", "/openapi.json", "/metrics", "/ready/bcs"):
        assert client.get(path).status_code == 200
    assert client.get("/").status_code == 404
    assert client.get("/videos").status_code == 307
    schema = client.get("/openapi.json").json()
    assert {"/detect", "/bcs", "/api/v1/analyses"} <= schema["paths"].keys()
    for endpoint in ("/detect", "/bcs"):
        response = client.post(endpoint, content=image_upload(), headers=IMAGE_HEADERS)
        assert response.status_code == 200
        if endpoint == "/detect":
            assert response.json()["detection_count"] == 0
        else:
            assert response.json()["bcs_category"] == 3
    response = submit(client)
    assert response.status_code == 202
    job = response.json()
    assert job["error"] is None and job["summary"] is None
    assert job["schema_version"] == "vacca-video-job-v1"
    assert client.get(f"/api/v1/analyses/{job['id']}").json() == job
    assert client.get("/api/v1/analyses/missing").json() == {"detail": "Análisis no encontrado"}


def test_primary_lifespan_recovery_persistence_and_finally_shutdown(runtime):
    store = Store(runtime)
    store.add("interrupted", "one.mp4")
    store.add("pending", "two.mp4")
    store.claim()
    with asyncio.Runner() as runner:
        async def run():
            with pytest.raises(RuntimeError, match="fixture shutdown"):
                async with main.app.router.lifespan_context(main.app):
                    assert main.app.state.video.available
                    assert Store(runtime).get("interrupted")["status"] == "failed"
                    assert Store(runtime).get("pending")["status"] == "pending"
                    response = await TestClient(main.app).arequest("GET", "/api/v1/analyses")
                    assert len(response.json()) == 2
                    raise RuntimeError("fixture shutdown")
        runner.run(run())
    worker = FakeWorker.instances[0]
    worker.start.assert_called_once()
    worker.stop.set.assert_called_once()
    worker.join.assert_called_once_with(timeout=10)
    assert not main.app.state.video.available
    assert Store(runtime).get("pending")["status"] == "pending"
    with asyncio.Runner() as runner:
        async def restart():
            async with main.app.router.lifespan_context(main.app):
                response = await TestClient(main.app).arequest("GET", "/api/v1/analyses")
                assert {job["id"] for job in response.json()} == {"interrupted", "pending"}
                assert main.app.state.video.store.get("pending")["status"] == "pending"
        runner.run(restart())
    assert len(FakeWorker.instances) == 2
    FakeWorker.instances[1].start.assert_called_once()
    FakeWorker.instances[1].join.assert_called_once_with(timeout=10)


@pytest.mark.parametrize("package", ["cv2", "ultralytics"])
def test_video_runtime_probe_identifies_missing_optional_packages(monkeypatch, package):
    monkeypatch.setattr(service, "find_spec", lambda name: None if name == package else object())
    with pytest.raises(ImportError, match=f"Missing video runtime: {package}"):
        service.check_runtime()


def test_failed_recovery_commit_cannot_claim_jobs_and_stops_started_thread(runtime, monkeypatch):
    from contextlib import contextmanager
    from vacca_video.worker import Worker

    resources = service.VideoResources(runtime, worker_factory=Worker)
    resources.prepare()
    resources.store.add("interrupted", "one.mp4")
    resources.store.add("pending", "two.mp4")
    resources.store.claim()
    original_connect = resources.store.connect
    claim = Mock(side_effect=AssertionError("Startup must not claim jobs before commit"))
    monkeypatch.setattr(resources.store, "claim", claim)

    @contextmanager
    def failing_commit():
        with original_connect() as conn:
            yield conn
            raise OSError("fixture recovery commit failed")

    monkeypatch.setattr(resources.store, "connect", failing_commit)
    monkeypatch.setattr(main, "VideoResources", lambda: resources)
    with asyncio.Runner() as runner:
        async def run():
            async with main.app.router.lifespan_context(main.app):
                assert not resources.available
                assert (await TestClient(main.app).arequest("GET", "/health")).status_code == 200
        runner.run(run())
    claim.assert_not_called()
    assert not resources.worker.thread.is_alive()
    assert Store(runtime).get("interrupted")["status"] == "processing"
    assert Store(runtime).get("pending")["status"] == "pending"


@pytest.mark.parametrize("failure", ["runtime", "storage", "worker"])
def test_video_startup_failure_is_503_without_breaking_legacy(runtime, monkeypatch, failure):
    if failure == "storage":
        runtime.write_text("existing unrelated file", encoding="utf-8")
    else:
        store = Store(runtime)
        store.add("interrupted", "one.mp4")
        store.claim()
        if failure == "runtime":
            def unavailable():
                raise ImportError("optional video dependency missing")
            monkeypatch.setattr(service, "check_runtime", unavailable)
        else:
            class FailingWorker(FakeWorker):
                def __init__(self, store):
                    super().__init__(store)
                    self.start.side_effect = OSError("thread unavailable")
            monkeypatch.setattr(service, "Worker", FailingWorker)
    with asyncio.Runner() as runner:
        async def run():
            async with main.app.router.lifespan_context(main.app):
                client = TestClient(main.app)
                response = await client.arequest("GET", "/api/v1/analyses")
                assert response.status_code == 503
                assert response.json() == {"detail": service.UNAVAILABLE_DETAIL}
                assert (await client.arequest("GET", "/health")).status_code == 200
                for path in ("/detect", "/bcs"):
                    assert (await client.arequest("POST", path, image_upload(), IMAGE_HEADERS)).status_code == 200
        runner.run(run())
    if failure != "storage":
        assert Store(runtime).get("interrupted")["status"] == "processing"
    else:
        assert runtime.read_text(encoding="utf-8") == "existing unrelated file"


def test_missing_build_is_only_a_video_ui_failure(client, tmp_path, monkeypatch):
    mount = next(route for route in main.app.routes if isinstance(getattr(route, "app", None), webapp.VideoUI))
    monkeypatch.setattr(mount.app, "directory", tmp_path / "missing-build")
    assert client.get("/videos/").status_code == 503
    assert client.get("/api/v1/analyses").status_code == 200
    assert client.get("/ui").status_code == 200
    assert client.get("/health").status_code == 200


def test_video_static_assets_use_prefix_without_overriding_legacy(client, tmp_path, monkeypatch):
    # A tiny build fixture keeps Python tests independent of installed npm packages.
    directory = tmp_path / "dist"
    (directory / "assets").mkdir(parents=True)
    (directory / "fonts").mkdir()
    index = (webapp.ROOT / "web/index.html").read_text(encoding="utf-8")
    (directory / "index.html").write_text(index, encoding="utf-8")
    (directory / "assets/main.js").write_text("fixture", encoding="utf-8")
    (directory / "assets/main.css").write_text("fixture", encoding="utf-8")
    (directory / "fonts/Manrope-VariableFont_wght.woff2").write_bytes(b"font fixture")
    (directory / "favicon.svg").write_text("fixture", encoding="utf-8")
    (directory / "vacca_logo.png").write_bytes(b"logo fixture")
    mount = next(route for route in main.app.routes if isinstance(getattr(route, "app", None), webapp.VideoUI))
    monkeypatch.setattr(mount, "app", webapp.VideoUI(directory))
    assert client.get("/videos/").content == (directory / "index.html").read_bytes()
    for asset in ("assets/main.js", "assets/main.css", "favicon.svg", "vacca_logo.png", "fonts/Manrope-VariableFont_wght.woff2"):
        assert client.get(f"/videos/{asset}").status_code == 200
    assert 'src="/videos/assets/main.js"' in index
    assert 'href="/videos/assets/main.css"' in index
    assert client.get("/assets/main.js").status_code == 404


@pytest.mark.parametrize("path", ["/api/v1/analyses", "/videos/", "/api/v1/analyses/missing/results"])
def test_remote_peer_cannot_spoof_local_headers_but_legacy_routes_stay_available(client, path):
    remote = TestClient(main.app, client_host="192.0.2.10", runner=client.runner)
    assert remote.get(path, headers={"host": "localhost"}).status_code == 403
    assert remote.get(path, headers={"host": "127.0.0.1", "x-forwarded-for": "127.0.0.1"}).status_code == 403
    headers = {"host": "public.example", "origin": "http://elsewhere.example"}
    for legacy in ("/health", "/ui", "/docs", "/metrics", "/ready/bcs"):
        assert remote.get(legacy, headers=headers).status_code == 200
    for legacy in ("/detect", "/bcs"):
        assert remote.post(legacy, content=image_upload(), headers=dict(IMAGE_HEADERS, **headers)).status_code == 200


@pytest.mark.parametrize("peer,host", [
    ("127.0.0.1", "localhost"), ("127.0.0.2", "127.0.0.2:8001"),
    ("::1", "[::1]:8001"), ("::ffff:127.0.0.1", "localhost"),
])
def test_loopback_clients_and_same_origin_upload(client, peer, host):
    local = TestClient(main.app, client_host=peer, runner=client.runner)
    assert local.get("/api/v1/analyses", headers={"host": host}).status_code == 200
    assert local.post("/api/v1/analyses?filename=cow.mp4", content=b"0000ftypisom0000",
                      headers={"host": host, "origin": f"http://{host}"}).status_code == 202


def test_video_host_origin_and_proxy_rejection_are_scoped(client):
    assert client.get("/api/v1/analyses", headers={"host": "public.example"}).status_code == 400
    assert client.post("/api/v1/analyses?filename=cow.mp4", headers={"origin": "http://evil.example"}).status_code == 403
    assert client.get("/videos/", headers={"x-forwarded-for": "127.0.0.1"}).status_code == 403
    assert client.get("/api/v1/analyses", headers={"host": "testserver"}).status_code == 400
    unknown = TestClient(main.app, client_host="testclient", runner=client.runner)
    assert unknown.get("/api/v1/analyses").status_code == 403


def test_asgi_root_path_cannot_bypass_video_local_security(client):
    remote = TestClient(main.app, client_host="192.0.2.10", runner=client.runner, root_path="/prefix")
    assert remote.get("/prefix/api/v1/analyses").status_code == 403
    assert remote.get("/prefix/videos/").status_code == 403
    assert remote.get("/prefix/health").status_code == 200
    local = TestClient(main.app, runner=client.runner, root_path="/prefix")
    assert local.get("/prefix/api/v1/analyses", headers={"origin": "http://localhost"}).status_code == 200


def test_video_upload_lock_and_processing_job_do_not_gate_image_or_bcs(client):
    async def run():
        video = main.app.state.video
        video.store.add("processing", "existing.mp4")
        video.store.update("processing", "processing")
        await video.upload_lock.acquire()
        upload = asyncio.create_task(client.arequest(
            "POST", "/api/v1/analyses?filename=cow.mp4", b"0000ftypisom0000"
        ))
        try:
            await asyncio.sleep(0)
            assert not upload.done()
            for path in ("/detect", "/bcs"):
                response = await asyncio.wait_for(client.arequest("POST", path, image_upload(), IMAGE_HEADERS), 3)
                assert response.status_code == 200
            assert (await client.arequest("GET", "/health")).status_code == 200
            assert not upload.done()
        finally:
            video.upload_lock.release()
        assert (await asyncio.wait_for(upload, 3)).status_code == 202
    client.runner.run(run())


def test_worker_claim_fault_preserves_history_and_rejects_racing_upload(runtime, monkeypatch):
    from vacca_video.worker import Worker

    resources = service.VideoResources(runtime, worker_factory=Worker)
    resources.prepare()
    store = resources.store
    store.add("history", "older.mp4")
    store.update("history", "completed")
    history_result = store.directory / "history/result"
    history_result.mkdir(parents=True)
    (history_result / "detections.jsonl").write_bytes(b'{"frame_index":0}\n')

    claim_entered = threading.Event()
    release_claim = threading.Event()
    upload_waiting = threading.Event()

    def fail_claim():
        claim_entered.set()
        if not release_claim.wait(3):
            raise TimeoutError("claim fault barrier was not released")
        raise OSError("private database path")

    monkeypatch.setattr(store, "claim", fail_claim)
    monkeypatch.setattr(main, "VideoResources", lambda: resources)

    async def streamed_upload():
        messages = []
        part = 0

        async def receive():
            nonlocal part
            if part == 0:
                part += 1
                return {
                    "type": "http.request",
                    "body": b"\x00\x00\x00\x18ftypisom",
                    "more_body": True,
                }
            if part == 1:
                part += 1
                upload_waiting.set()
                assert await asyncio.to_thread(release_claim.wait, 2)
                for _ in range(100):
                    if not resources.available:
                        break
                    await asyncio.sleep(0.01)
                assert not resources.available
                return {"type": "http.request", "body": b"00000000", "more_body": False}
            await asyncio.Event().wait()

        async def send(message):
            messages.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/v1/analyses",
            "raw_path": b"/api/v1/analyses",
            "query_string": b"filename=racing.mp4",
            "root_path": "",
            "server": ("localhost", 80),
            "client": ("127.0.0.1", 1),
            "headers": [(b"host", b"localhost")],
        }
        await main.app(scope, receive, send)
        status = next(message["status"] for message in messages if message["type"] == "http.response.start")
        body = b"".join(message.get("body", b"") for message in messages)
        return status, json.loads(body)

    async def exercise():
        async with main.app.router.lifespan_context(main.app):
            assert await asyncio.to_thread(claim_entered.wait, 2)
            request = TestClient(main.app)
            upload = asyncio.create_task(streamed_upload())
            assert await asyncio.to_thread(upload_waiting.wait, 2)
            release_claim.set()
            status, payload = await asyncio.wait_for(upload, 3)
            assert status == 503
            assert payload == {"detail": service.UNAVAILABLE_DETAIL}

            history = await request.arequest("GET", "/api/v1/analyses")
            assert history.status_code == 200
            assert [job["id"] for job in history.json()] == ["history"]
            result = await request.arequest("GET", "/api/v1/analyses/history/results")
            assert result.status_code == 200
            assert result.content == b'{"frame_index":0}\n'
            assert (await request.arequest("GET", "/health")).status_code == 200
            for path in ("/detect", "/bcs"):
                response = await request.arequest("POST", path, image_upload(), IMAGE_HEADERS)
                assert response.status_code == 200

            files = list(store.directory.glob("*/input.mp4"))
            assert files == []
            assert [job["id"] for job in store.list()] == ["history"]

    with asyncio.Runner() as runner:
        runner.run(exercise())


def test_unhandled_worker_loop_exception_faults_resources_without_retry(runtime, monkeypatch, caplog):
    from vacca_video.worker import Worker

    resources = service.VideoResources(runtime, worker_factory=Worker)
    resources.prepare()
    store = resources.store
    store.add("history", "older.mp4")
    store.update("history", "completed")
    result_dir = store.directory / "history/result"
    result_dir.mkdir(parents=True)
    (result_dir / "detections.jsonl").write_bytes(b'{"frame_index":0}\n')

    claim_returned = threading.Event()
    claims = []

    def claim_malformed_legacy_id():
        claims.append("claim")
        claim_returned.set()
        return b"legacy-id"

    monkeypatch.setattr(store, "claim", claim_malformed_legacy_id)
    monkeypatch.setattr(main, "VideoResources", lambda: resources)
    process_start = Mock(side_effect=AssertionError("faulted claim must not start processing"))
    monkeypatch.setattr(webapp.subprocess, "Popen", process_start)

    async def exercise():
        async with main.app.router.lifespan_context(main.app):
            assert await asyncio.to_thread(claim_returned.wait, 2)
            for _ in range(100):
                if not resources.worker.thread.is_alive():
                    break
                await asyncio.sleep(0.01)

            assert not resources.worker.thread.is_alive()
            assert resources.worker.health == "faulted"
            assert not resources.available
            assert resources.worker_faulted
            assert claims == ["claim"]
            process_start.assert_not_called()

            client = TestClient(main.app)
            history = await client.arequest("GET", "/api/v1/analyses")
            assert history.status_code == 200
            assert [job["id"] for job in history.json()] == ["history"]
            artifact = await client.arequest("GET", "/api/v1/analyses/history/results")
            assert artifact.status_code == 200
            assert artifact.content == b'{"frame_index":0}\n'
            rejected = await client.arequest(
                "POST", "/api/v1/analyses?filename=new.mp4", b"\x00\x00\x00\x18ftypisom"
            )
            assert rejected.status_code == 503
            assert [job["id"] for job in store.list()] == ["history"]

    with caplog.at_level(logging.ERROR, logger="vacca_video.worker"):
        with asyncio.Runner() as runner:
            runner.run(exercise())
    assert any("TypeError" in record.message for record in caplog.records)
    assert "legacy-id" not in caplog.text


def test_completed_video_artifacts_preserve_paths_and_nullable_fields(client):
    job_id = submit(client).json()["id"]
    store = main.app.state.video.store
    path = store.directory / job_id / "result"
    (path / "frames").mkdir(parents=True)
    frame = {"frame_index": 0, "timestamp_ms": 0, "detection_count": 0, "extra": None}
    (path / "detections.jsonl").write_text(json.dumps(frame) + "\n", encoding="utf-8")
    (path / "summary.json").write_text('{"schema_version":"vacca-video-summary-v1","custom":null}', encoding="utf-8")
    (path / "frames/000000.jpg").write_bytes(b"fixture")
    store.update(job_id, "completed")
    data = client.get(f"/api/v1/analyses/{job_id}/frames").json()
    assert data["items"][0] == dict(frame, image_url=f"/api/v1/analyses/{job_id}/frames/0/image")
    assert client.get(data["items"][0]["image_url"]).content == b"fixture"
    assert client.get(f"/api/v1/analyses/{job_id}/results").status_code == 200
    assert client.get(f"/api/v1/analyses/{job_id}").json()["summary"]["custom"] is None


def test_candidate_image_is_fixed_job_scoped_and_local_only(client, monkeypatch):
    from vacca_video.artifacts import job_artifact
    job_id = submit(client).json()["id"]
    store = main.app.state.video.store
    directory = store.directory / job_id / "result"
    directory.mkdir()
    (directory / "candidate.png").write_bytes(b"original candidate fixture")
    endpoint = f"/api/v1/analyses/{job_id}/candidate/image"
    assert client.get(endpoint).content == b"original candidate fixture"
    assert client.get(endpoint + "?artifact=../../secret.png").content == b"original candidate fixture"
    assert client.get("/api/v1/analyses/missing/candidate/image").status_code == 404
    remote = TestClient(main.app, client_host="192.0.2.10", runner=client.runner)
    assert remote.get(endpoint).status_code == 403
    assert client.get(endpoint, headers={"origin": "http://evil.example"}).status_code == 403
    candidate = job_artifact(store.directory, job_id, "candidate.png")
    path_type = type(candidate)
    original = path_type.is_symlink
    monkeypatch.setattr(path_type, "is_symlink", lambda path: path == candidate or original(path))
    assert client.get(endpoint).status_code == 404


def test_failed_video_or_unsaved_bcs_never_claims_completed_bcs(client):
    from vacca_video.artifacts import BCS_WRITE_ERROR
    job_id = submit(client).json()["id"]
    store = main.app.state.video.store
    directory = store.directory / job_id / "result"
    directory.mkdir()
    (directory / "summary.json").write_text(json.dumps({
        "bcs": {"status": "completed", "bcs_category": 4}, "custom": None
    }), encoding="utf-8")
    store.update(job_id, "failed", "Interrupción de prueba")
    outcome = client.get(f"/api/v1/analyses/{job_id}").json()["summary"]["bcs"]
    assert outcome == dict(status="not_evaluated", reason="video_not_completed")
    store.update(job_id, "completed", BCS_WRITE_ERROR)
    outcome = client.get(f"/api/v1/analyses/{job_id}").json()["summary"]["bcs"]
    assert outcome == dict(status="failed", reason="metadata_write_failed")


@pytest.mark.parametrize("invalid", ["artifact", "missing_fields", "confidence"])
@pytest.mark.parametrize("publication_fails", [False, True])
def test_invalid_candidate_cannot_break_detail_history_or_download(client, monkeypatch, invalid, publication_fails):
    from test_video_bcs import candidate_job
    from vacca_video.artifacts import BCS_WRITE_ERROR
    from vacca_video.bcs import VideoBCSCoordinator
    store = main.app.state.video.store
    _, directory, _ = candidate_job(store.directory)
    summary = json.loads((directory / "summary.json").read_text())
    if invalid == "artifact":
        summary["candidate"]["artifact"] = "../../secret.png"
    elif invalid == "missing_fields":
        summary["candidate"] = {"artifact": "candidate.png"}
    else:
        summary["candidate"]["detector_confidence"] = 2.0
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    store.update("fake", "completed")
    valid_id = submit(client).json()["id"]

    # Status reads must be safe even before the worker can publish a fallback.
    response = client.get("/api/v1/analyses/fake")
    assert response.status_code == 200
    assert response.json()["summary"]["candidate"] is None
    assert response.json()["summary"]["bcs"] == dict(status="invalid_input", reason="invalid_candidate")
    assert response.json()["summary"]["custom"] is None
    assert client.get("/api/v1/analyses").status_code == 200

    if publication_fails:
        def fail(*args):
            raise OSError("fixture publication failure")
        monkeypatch.setattr("vacca_video.bcs.publish_summary", fail)
    async def never(*args, **kwargs):
        pytest.fail("Rejected candidate must never reach BCS")
    async def finish():
        coordinator = VideoBCSCoordinator(asyncio.get_running_loop(), never)
        try:
            return await asyncio.to_thread(coordinator.finish, store.directory, "fake", threading.Event())
        finally:
            await coordinator.close()
    error = client.runner.run(finish())
    assert error == (BCS_WRITE_ERROR if publication_fails else None)
    store.update("fake", "completed", error)
    response = client.get("/api/v1/analyses/fake")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["summary"]["candidate"] is None
    assert data["summary"]["candidate_rejection_reason"] == "invalid_candidate"
    assert data["summary"]["bcs"] == dict(
        status="failed" if publication_fails else "invalid_input",
        reason="metadata_write_failed" if publication_fails else "invalid_candidate",
    )
    history = client.get("/api/v1/analyses")
    assert history.status_code == 200
    assert {item["id"] for item in history.json()} == {"fake", valid_id}
    assert next(item for item in history.json() if item["id"] == "fake") == data
    assert client.get("/api/v1/analyses/fake/results").content == (directory / "detections.jsonl").read_bytes()
    persisted = json.loads((directory / "summary.json").read_text())
    assert persisted["candidate"] == (summary["candidate"] if publication_fails else None)


@pytest.mark.parametrize("failure", ["false", "encode", "write"])
def test_candidate_persistence_failure_keeps_cli_worker_and_api_completed(client, monkeypatch, failure):
    import cv2
    from scripts.process_video import main as process_video
    from test_video_io import FakeCapture
    from test_video_candidate import detection
    from vacca_video.bcs import VideoBCSCoordinator
    from vacca_video.worker import Worker
    job_id = submit(client).json()["id"]
    store = main.app.state.video.store
    original_write = cv2.imwrite
    def write(path, image):
        if Path(path).name != "candidate.png":
            return original_write(path, image)
        Path(path).write_bytes(b"partial candidate")
        if failure == "encode":
            raise cv2.error("fixture PNG encoding failure")
        if failure == "write":
            raise OSError("fixture candidate write failure")
        return False
    monkeypatch.setattr(cv2, "imwrite", write)
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: FakeCapture([0, 200]))
    fake = SimpleNamespace(metadata=dict(name="fake", confidence=0.25, sha256="fake-hash"),
                           detect=lambda image: detection() | dict(image_width=64, image_height=48))
    monkeypatch.setattr("vacca_video.detector.ImageDetectorAdapter", lambda *args: fake)
    exit_codes = []
    worker = Worker(store)
    class Process:
        def __enter__(self):
            self.returncode = process_video([
                str(store.directory / job_id / "input.mp4"), "--output",
                str(store.directory / job_id / "result"), "--no-video",
            ])
            exit_codes.append(self.returncode)
            return self
        def __exit__(self, *args): pass
        def poll(self): return self.returncode
    monkeypatch.setattr("vacca_video.worker.subprocess.Popen", lambda *args, **kwargs: Process())
    update = store.update
    def finished(identifier, status, error=None):
        update(identifier, status, error)
        worker.stop.set()
    monkeypatch.setattr(store, "update", finished)
    async def never(*args, **kwargs):
        pytest.fail("An unsaved candidate must never reach BCS")
    async def run():
        coordinator = VideoBCSCoordinator(asyncio.get_running_loop(), never)
        worker.finish_video = coordinator.finish
        try:
            await asyncio.wait_for(asyncio.to_thread(worker.loop), 3)
        finally:
            await coordinator.close()
    client.runner.run(run())
    assert exit_codes == [0]
    response = client.get(f"/api/v1/analyses/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == data["summary"]["status"] == "completed"
    assert data["error"] is None
    assert data["summary"]["candidate"] is None
    assert data["summary"]["bcs"] == dict(status="not_evaluated", reason="candidate_write_failed")
    assert data["frames_analyzed"] == 2
    assert not (store.directory / job_id / "result/candidate.png").exists()
    assert client.get("/api/v1/analyses").status_code == 200
    assert client.get(f"/api/v1/analyses/{job_id}/results").status_code == 200


def test_candidate_status_boundary_preserves_valid_extensions(client):
    from test_video_bcs import candidate_job
    store = main.app.state.video.store
    _, directory, _ = candidate_job(store.directory)
    summary = json.loads((directory / "summary.json").read_text())
    summary["candidate"]["custom_candidate"] = None
    summary["candidate"]["detector_model"]["custom_model"] = "extension"
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    store.update("fake", "completed")
    detail = client.get("/api/v1/analyses/fake")
    assert detail.status_code == 200
    assert detail.json()["summary"] == summary
    assert client.get("/api/v1/analyses").json()[0] == detail.json()


def test_compatibility_launcher_delegates_to_principal(monkeypatch):
    launch = Mock()
    monkeypatch.setattr(run_api, "main", launch)
    run_video_web.main(["--open", "--port", "8001"])
    launch.assert_called_once_with(["--open", "--port", "8001"])


@pytest.mark.parametrize("host,browser_host", [
    ("127.0.0.1", "127.0.0.1"), ("0.0.0.0", "127.0.0.1"), ("::", "[::1]"),
])
def test_primary_open_preserves_existing_flags_and_opens_videos(monkeypatch, host, browser_host):
    from vacca_api import browser

    launch = Mock()
    monkeypatch.setattr(browser, "run_with_browser", launch)
    run_api.main(["--open", "--host", host, "--port", "8123", "--reload"])
    launch.assert_called_once_with(host=host, port=8123, reload=True,
                                   url=f"http://{browser_host}:8123/videos/")
