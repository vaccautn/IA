"""Public video router contract and persistence checks, without model loading."""

import json
import asyncio
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
