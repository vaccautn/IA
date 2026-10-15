"""Public video router contract and persistence checks, without model loading."""

import json
import asyncio
import logging
import threading
from urllib.parse import urlencode, urlsplit

import pytest

from vacca_video import webapp


class TestClient:
    """Exercise the ASGI boundary using only the already locked dependencies."""

    __test__ = False

    def __init__(self, app, client_host="127.0.0.1", runner=None, root_path=""):
        self.app = app
        self.client_host = client_host
        self.runner = runner
        self.root_path = root_path

    def request(self, method, url, content=b"", headers=None, params=None):
        run = self.runner.run if self.runner else asyncio.run
        return run(self.arequest(method, url, content, headers, params))

    async def arequest(self, method, url, content=b"", headers=None, params=None):
        if params:
            url += "?" + urlencode(params)
        parts = urlsplit(url)
        messages = []

        async def run():
            delivered = False

            async def receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": content, "more_body": False}
                await asyncio.Event().wait()

            async def send(message):
                messages.append(message)

            scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.4"},
                "http_version": "1.1",
                "method": method,
                "scheme": "http",
                "path": parts.path,
                "raw_path": parts.path.encode(),
                "query_string": parts.query.encode(),
                "root_path": self.root_path,
                "server": ("localhost", 80),
                "client": (self.client_host, 1),
                "headers": [
                    (k.lower().encode(), v.encode())
                    for k, v in dict({"host": "localhost"}, **(headers or {})).items()
                ],
            }
            await self.app(scope, receive, send)

        await run()
        body = b"".join(item.get("body", b"") for item in messages)
        return type(
            "Response",
            (),
            {
                "status_code": messages[0]["status"],
                "content": body,
                "json": lambda self: json.loads(body),
            },
        )()

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(
        webapp.shutil, "disk_usage", lambda path: type("Disk", (), {"free": 1024**3})()
    )
    yield TestClient(webapp.create_app(tmp_path, start_worker=False))


def submit(client, name="vacas.mp4"):
    return client.post(
        "/api/v1/analyses",
        params={"filename": name},
        content=b"\x00\x00\x00\x18ftypisom00000000",
    )


def test_upload_is_persistent_and_path_is_generated(client):
    response = submit(client, "../vacas.mp4")
    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "pending"
    assert job["schema_version"] == "vacca-video-job-v1"
    assert (client.app.state.store.directory / job["id"] / "input.mp4").is_file()
    reopened = webapp.Store(client.app.state.store.directory)
    assert reopened.get(job["id"])["name"] == "../vacas.mp4"
    assert client.get("/api/v1/analyses").json()[0]["id"] == job["id"]


def test_invalid_and_large_uploads_leave_no_job_or_file(client, monkeypatch):
    assert (
        client.post(
            "/api/v1/analyses?filename=bad.mp4", content=b"not video"
        ).status_code
        == 415
    )
    assert submit(client, "wrong.avi").status_code == 415
    monkeypatch.setattr(webapp, "MAX_UPLOAD", 10)
    assert submit(client).status_code == 413
    assert client.get("/api/v1/analyses").json() == []
    assert list(client.app.state.store.directory.glob("*/input.mp4")) == []


def test_queue_is_bounded_and_disk_checked(client, monkeypatch):
    for _ in range(3):
        assert submit(client).status_code == 202
    assert submit(client).status_code == 429
    for job in client.app.state.store.list():
        client.app.state.store.update(job["id"], "completed")
    monkeypatch.setattr(
        webapp.shutil, "disk_usage", lambda path: type("Disk", (), {"free": 1})()
    )
    assert submit(client).status_code == 507


def test_recovery_only_fails_interrupted_jobs(client):
    first = submit(client).json()["id"]
    second = submit(client).json()["id"]
    store = client.app.state.store
    assert store.claim() == first
    store.recover()
    assert store.get(first)["status"] == "failed"
    assert store.get(second)["status"] == "pending"
    assert store.claim() == second
    assert store.claim() is None


def test_frames_and_artifacts_are_scoped_and_paginated(client):
    job_id = submit(client).json()["id"]
    path = client.app.state.store.directory / job_id / "result"
    (path / "frames").mkdir(parents=True)
    (path / "detections.jsonl").write_text(
        "\n".join(
            json.dumps(
                {"frame_index": i, "timestamp_ms": i * 200, "detection_count": 0}
            )
            for i in range(3)
        )
        + '\n{"incomplete":',
        encoding="utf-8",
    )
    (path / "frames/000001.jpg").write_bytes(b"jpeg fixture")
    data = client.get(f"/api/v1/analyses/{job_id}/frames?offset=1&limit=1").json()
    assert data["total"] == 3
    assert data["items"][0]["timestamp_ms"] == 200
    assert client.get(data["items"][0]["image_url"]).content == b"jpeg fixture"
    assert client.get(f"/api/v1/analyses/{job_id}/frames/-1/image").status_code == 404
    assert client.get(f"/api/v1/analyses/{job_id}/frames?limit=1000").status_code == 422
    assert client.get(f"/api/v1/analyses/{job_id}/results").status_code == 409
    client.app.state.store.update(job_id, "failed", "interrupted")
    assert client.get(f"/api/v1/analyses/{job_id}/results").status_code == 200
    assert client.get("/api/v1/analyses/missing/frames").status_code == 404


def test_cross_origin_upload_and_invalid_host_rejected(client):
    assert (
        client.post(
            "/api/v1/analyses?filename=cow.mp4",
            headers={"origin": "https://outside.example"},
            content=b"",
        ).status_code
        == 403
    )
    assert (
        client.get("/api/v1/analyses", headers={"host": "outside.example"}).status_code
        == 400
    )


def test_worker_error_is_persisted(client, monkeypatch):
    job_id = submit(client).json()["id"]
    worker = webapp.Worker(client.app.state.store)

    def fail(*args, **kwargs):
        worker.stop.set()
        raise OSError("No se pudo iniciar el proceso")

    monkeypatch.setattr(webapp.subprocess, "Popen", fail)
    worker.loop()
    assert client.app.state.store.get(job_id)["status"] == "failed"


@pytest.mark.parametrize("fault", ["claim", "update"])
def test_worker_storage_fault_is_terminal_and_does_not_retry(client, monkeypatch, caplog, fault):
    job_id = submit(client).json()["id"]
    store = client.app.state.store
    worker = webapp.Worker(store)
    faults = []
    fault_notified = threading.Event()
    claim_entered = threading.Event()
    release_claim = threading.Event()

    def notify_fault():
        faults.append("faulted")
        fault_notified.set()

    worker.on_fault = notify_fault
    claims = 0
    updates = []

    if fault == "claim":
        def failed_claim():
            nonlocal claims
            claims += 1
            claim_entered.set()
            if not release_claim.wait(2):
                raise TimeoutError("claim fault barrier was not released")
            raise OSError("private database path")

        monkeypatch.setattr(store, "claim", failed_claim)
    else:
        claim = store.claim

        def counted_claim():
            nonlocal claims
            claims += 1
            claim_entered.set()
            if not release_claim.wait(2):
                raise TimeoutError("update fault barrier was not released")
            return claim()

        def failed_update(identifier, status, error=None):
            updates.append(status)
            raise OSError("private database path")

        def failed_process(*args, **kwargs):
            raise RuntimeError("synthetic subprocess failure")

        monkeypatch.setattr(store, "claim", counted_claim)
        monkeypatch.setattr(store, "update", failed_update)
        monkeypatch.setattr(webapp.subprocess, "Popen", failed_process)

    worker.ready.clear()
    with caplog.at_level(logging.ERROR, logger="vacca_video.worker"):
        worker.thread.start()
        assert worker.health == "starting"
        worker.ready.set()
        assert claim_entered.wait(2)
        assert worker.health == "healthy"
        release_claim.set()
        assert fault_notified.wait(2)
        worker.thread.join(2)

    assert worker.health == "faulted"
    assert not worker.thread.is_alive()
    assert claims == 1
    assert faults == ["faulted"]
    assert "private database path" not in caplog.text
    if fault == "claim":
        assert store.get(job_id)["status"] == "pending"
    else:
        assert updates == ["failed"]
        assert store.get(job_id)["status"] == "processing"


def test_ordinary_job_failure_is_persisted_and_worker_continues(client, monkeypatch):
    failed_id = submit(client).json()["id"]
    completed_id = submit(client).json()["id"]
    store = client.app.state.store
    worker = webapp.Worker(store)
    original_update = store.update
    updates = []
    starts = 0
    faults = []

    def record_update(job_id, status, error=None):
        updates.append((job_id, status))
        original_update(job_id, status, error)
        if status == "failed":
            assert worker.health == "healthy"
        if job_id == completed_id:
            worker.stop.set()

    class CompletedProcess:
        returncode = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def poll(self):
            return 0

    def process(*args, **kwargs):
        nonlocal starts
        starts += 1
        if starts == 1:
            raise RuntimeError("synthetic process failure")
        return CompletedProcess()

    monkeypatch.setattr(store, "update", record_update)
    monkeypatch.setattr(webapp.subprocess, "Popen", process)
    worker.on_fault = lambda: faults.append("faulted")

    worker.loop()

    assert updates == [(failed_id, "failed"), (completed_id, "completed")]
    assert starts == 2
    assert store.get(failed_id)["status"] == "failed"
    assert store.get(completed_id)["status"] == "completed"
    assert worker.health == "stopped"
    assert faults == []


def test_job_json_contract_and_nullable_summary(client):
    job = submit(client).json()
    assert set(job) == {
        "id", "name", "status", "created_at", "error", "frames_analyzed",
        "summary", "schema_version",
    }
    assert job["error"] is None and job["summary"] is None
    assert job["frames_analyzed"] == 0
    assert client.get(f"/api/v1/analyses/{job['id']}").json() == job
    assert client.get("/api/v1/analyses/missing").json() == {
        "detail": "Análisis no encontrado"
    }
    path = client.app.state.store.directory / job["id"] / "result"
    path.mkdir()
    summary = {"status": "failed", "stop_reason": "error", "custom": None}
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert client.get(f"/api/v1/analyses/{job['id']}").json()["summary"] is None
    client.app.state.store.update(job["id"], "failed", "fixture")
    assert client.get(f"/api/v1/analyses/{job['id']}").json()["summary"] == summary


@pytest.mark.parametrize("outcome", ["completed", "nonzero", "timeout", "shutdown", "kill"])
def test_worker_lifecycle_without_real_subprocess(client, monkeypatch, outcome):
    job_id = submit(client).json()["id"]
    worker = webapp.Worker(client.app.state.store)
    calls = []

    class Stop:
        stopped = False
        waits = 0

        def is_set(self):
            return self.stopped

        def set(self):
            self.stopped = True

        def wait(self, timeout):
            self.waits += 1
            if outcome == "shutdown" or self.waits > 1200:
                self.set()
            return self.stopped

    worker.stop = Stop()

    class Process:
        returncode = 1 if outcome == "nonzero" else 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            worker.stop.set()

        def poll(self):
            return self.returncode if outcome in ("completed", "nonzero") else None

        def terminate(self):
            calls.append("terminate")

        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            if outcome == "kill" and timeout is not None:
                raise webapp.subprocess.TimeoutExpired("fixture", timeout)

        def kill(self):
            calls.append("kill")

    def spawn(command, **kwargs):
        assert command[-1] == "--no-video"
        assert command[command.index("--max-frames") + 1] == "300"
        assert command[command.index("--preview-limit") + 1] == "300"
        assert kwargs["cwd"] == webapp.ROOT
        return Process()

    monkeypatch.setattr(webapp.subprocess, "Popen", spawn)
    worker.loop()
    job = client.app.state.store.get(job_id)
    assert job["status"] == ("completed" if outcome == "completed" else "failed")
    if outcome in ("timeout", "shutdown", "kill"):
        assert job["error"] == "Procesamiento interrumpido o límite de 10 minutos alcanzado."
        assert calls[:2] == ["terminate", ("wait", 5)]
        assert worker.stop.waits == (1 if outcome == "shutdown" else 1200)
    if outcome == "kill":
        assert calls[2:] == ["kill", ("wait", None)]


def test_store_claim_is_atomic(client):
    from concurrent.futures import ThreadPoolExecutor

    job_id = submit(client).json()["id"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        claimed = list(pool.map(lambda _: client.app.state.store.claim(), range(2)))
    assert claimed.count(job_id) == 1
    assert claimed.count(None) == 1


def test_video_policy_matches_ui_and_cli():
    import re

    from scripts.process_video import parser
    from vacca_video.pipeline import SamplingConfig
    from vacca_video import policies

    policy = (webapp.ROOT / "web/src/policy.js").read_text(encoding="utf-8")
    upload = re.search(r"MAX_UPLOAD_BYTES = (\d+) \* 1024 \* 1024", policy)
    assert int(upload[1]) * 1024 * 1024 == policies.WEB_MAX_UPLOAD_BYTES
    assert f"SAMPLE_FPS = {policies.DEFAULT_SAMPLE_FPS:g};" in policy
    assert f"MAX_FRAMES = {policies.DEFAULT_MAX_FRAMES};" in policy
    args = parser().parse_args(["fixture.mp4"])
    assert args.sample_fps == SamplingConfig().sample_fps
    assert args.max_frames == SamplingConfig().max_frames
    assert args.preview_limit == policies.CLI_PREVIEW_LIMIT == 120


def test_full_frame_and_summary_contract_preserves_artifact_fields(client):
    job_id = submit(client).json()["id"]
    path = client.app.state.store.directory / job_id / "result"
    path.mkdir()
    frame = {
        "schema_version": "vacca-video-frame-v1", "processing_id": "processor-id",
        "model_version": "fixture.pt", "frame_index": 0, "timestamp_ms": 0,
        "timestamp_source": "decoder", "cow_detected": True, "detection_count": 1,
        "detections": [{"class_name": "cow", "confidence": 0.9,
                        "bbox": {"x_center": 0.5, "y_center": 0.5,
                                 "width": 0.3, "height": 0.4},
                        "x1": None, "y1": 3, "x2": 8, "y2": 10}],
        "image_width": 64, "image_height": 48, "inference_time_ms": 1.25,
        "extension": {"nullable": None},
    }
    summary = {
        "schema_version": "vacca-video-summary-v1", "processing_id": "processor-id",
        "status": "completed", "source": {"name": "input.mp4", "fps": 5.0,
            "width": 64, "height": 48, "declared_frames": None},
        "model": {"name": "fixture.pt", "confidence": 0.25, "sha256": "fixture"},
        "config": {"sample_fps": 5.0, "max_frames": 300}, "frames_analyzed": 1,
        "last_timestamp_ms": 0, "stop_reason": "end_of_source",
        "annotated_video": None, "annotated_video_fps": None, "extension": [1, None],
    }
    (path / "detections.jsonl").write_text(json.dumps(frame) + "\n", encoding="utf-8")
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    client.app.state.store.update(job_id, "completed")
    expected = dict(frame, image_url=f"/api/v1/analyses/{job_id}/frames/0/image")
    assert client.get(f"/api/v1/analyses/{job_id}/frames").json() == {
        "items": [expected], "total": 1,
    }
    assert client.get(f"/api/v1/analyses/{job_id}").json()["summary"] == summary
    schema = client.get("/openapi.json").json()
    assert "JobResponse" in schema["components"]["schemas"]
    assert "FramePageResponse" in schema["components"]["schemas"]


@pytest.mark.parametrize(
    "summary_payload",
    ["{", "[]", '"not an object"', '{"status": 5}'],
    ids=["malformed-json", "list", "scalar", "invalid-field-type"],
)
def test_corrupt_job_summary_does_not_hide_other_history(client, summary_payload):
    store = client.app.state.store
    corrupt_id = submit(client, "corrupt.mp4").json()["id"]
    valid_id = submit(client, "valid.mp4").json()["id"]
    store.update(corrupt_id, "failed", "original worker error")
    store.update(valid_id, "completed")
    corrupt_path = store.directory / corrupt_id / "result"
    valid_path = store.directory / valid_id / "result"
    corrupt_path.mkdir()
    valid_path.mkdir()
    (corrupt_path / "summary.json").write_text(summary_payload, encoding="utf-8")
    valid_summary = {"status": "completed", "frames_analyzed": 3, "extension": None}
    (valid_path / "summary.json").write_text(json.dumps(valid_summary), encoding="utf-8")

    response = client.get("/api/v1/analyses")

    assert response.status_code == 200
    jobs = {job["id"]: job for job in response.json()}
    assert set(jobs) == {corrupt_id, valid_id}
    assert jobs[valid_id]["summary"] == valid_summary
    assert "summary_error" not in jobs[valid_id]
    assert jobs[corrupt_id]["summary"] is None
    assert jobs[corrupt_id]["error"] == "original worker error"
    assert jobs[corrupt_id]["summary_error"] == "No se pudo leer el resumen guardado; las detecciones y el historial siguen disponibles."
    detail = client.get(f"/api/v1/analyses/{corrupt_id}")
    assert detail.status_code == 200
    assert detail.json()["summary"] is None
    assert detail.json()["summary_error"] == jobs[corrupt_id]["summary_error"]


def test_missing_terminal_summary_is_reported_without_inventing_bcs_outcome(client):
    job_id = submit(client).json()["id"]
    client.app.state.store.update(job_id, "completed")

    response = client.get(f"/api/v1/analyses/{job_id}")

    assert response.status_code == 200
    job = response.json()
    assert job["summary"] is None
    assert job["summary_error"] == "No se pudo leer el resumen guardado; las detecciones y el historial siguen disponibles."
    assert "bcs" not in job


def _nested_array(depth):
    value = "leaf"
    for _ in range(depth):
        value = [value]
    return value


@pytest.mark.parametrize(
    "extension",
    [_nested_array(256), _nested_array(300), "\ud800"],
    ids=["near-serializer-depth", "deep-serializer-depth", "unpaired-surrogate"],
)
def test_response_serialization_errors_are_isolated_per_job(client, extension):
    store = client.app.state.store
    corrupt_id = submit(client, "deep-extension.mp4").json()["id"]
    valid_id = submit(client, "valid-neighbor.mp4").json()["id"]
    store.update(corrupt_id, "completed")
    store.update(valid_id, "completed")
    corrupt_path = store.directory / corrupt_id / "result"
    valid_path = store.directory / valid_id / "result"
    corrupt_path.mkdir()
    valid_path.mkdir()
    (corrupt_path / "summary.json").write_text(
        json.dumps({"status": "completed", "extension": extension}), encoding="utf-8"
    )
    valid_summary = {"status": "completed", "custom": None}
    (valid_path / "summary.json").write_text(json.dumps(valid_summary), encoding="utf-8")
    (corrupt_path / "detections.jsonl").write_text('{"frame_index":0}\n', encoding="utf-8")

    history = client.get("/api/v1/analyses")

    assert history.status_code == 200
    jobs = {job["id"]: job for job in history.json()}
    assert jobs[valid_id]["summary"] == valid_summary
    assert "summary_error" not in jobs[valid_id]
    assert jobs[corrupt_id]["summary"] is None
    assert jobs[corrupt_id]["summary_error"] == webapp.SUMMARY_UNAVAILABLE_DETAIL
    assert jobs[corrupt_id]["frames_analyzed"] == 1
    detail = client.get(f"/api/v1/analyses/{corrupt_id}")
    assert detail.status_code == 200
    assert detail.json()["summary"] is None
    assert detail.json()["summary_error"] == webapp.SUMMARY_UNAVAILABLE_DETAIL


def test_normal_summary_serialization_preserves_extensions_and_nullable_fields(client):
    job_id = submit(client, "serializable.mp4").json()["id"]
    client.app.state.store.update(job_id, "completed")
    result = client.app.state.store.directory / job_id / "result"
    result.mkdir()
    summary = {"status": "completed", "extension": _nested_array(8), "nullable": None}
    (result / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    response = client.get(f"/api/v1/analyses/{job_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"] == summary
    assert "summary_error" not in payload


def test_lifespan_recovers_pending_and_stops_worker(tmp_path, monkeypatch):
    calls = []

    class FakeWorker:
        def __init__(self, store):
            import threading

            self.ready = threading.Event()
            self.stop = type("Stop", (), {"set": lambda _: calls.append("stop")})()
            self.thread = self
            self.store = store

        def start(self):
            calls.append("start")

        def join(self, timeout):
            calls.append(("join", timeout))

    monkeypatch.setattr(webapp, "Worker", FakeWorker)
    app = webapp.create_app(tmp_path)
    app.state.store.add("interrupted", "one.mp4")
    app.state.store.add("pending", "two.mp4")
    app.state.store.claim()

    async def run():
        async with app.router.lifespan_context(app):
            assert app.state.store.get("interrupted")["status"] == "failed"
            assert app.state.store.get("pending")["status"] == "pending"

    asyncio.run(run())
    assert calls == ["start", "stop", ("join", 10)]
