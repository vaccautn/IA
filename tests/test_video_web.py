"""Public standalone API contract and persistence checks, without model loading."""

import json
import asyncio
import io
import zipfile
from urllib.parse import urlencode, urlsplit

import pytest

from vacca_video import webapp


class TestClient:
    """Exercise the ASGI boundary using only the already locked dependencies."""

    __test__ = False

    def __init__(self, app):
        self.app = app

    def request(self, method, url, content=b"", headers=None, params=None):
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
                "root_path": "",
                "server": ("testserver", 80),
                "client": ("127.0.0.1", 1),
                "headers": [
                    (k.lower().encode(), v.encode())
                    for k, v in dict({"host": "testserver"}, **(headers or {})).items()
                ],
            }
            await self.app(scope, receive, send)

        asyncio.run(run())
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


def test_highlight_prefers_count_then_confidence_and_keeps_first_tie():
    frames = [
        {"frame_index": 0, "timestamp_ms": 0, "detection_count": 1,
         "detections": [{"confidence": 0.99}]},
        {"frame_index": 5, "timestamp_ms": 200, "detection_count": 2,
         "detections": [{"confidence": 0.6}, {"confidence": 0.8}]},
        {"frame_index": 10, "timestamp_ms": 400, "detection_count": 2,
         "detections": [{"confidence": 0.8}, {"confidence": 0.9}]},
        {"frame_index": 15, "timestamp_ms": 600, "detection_count": 2,
         "detections": [{"confidence": 0.8}, {"confidence": 0.9}]},
    ]
    best = webapp.highlighted_frame(frames, "job")
    assert best["frame_index"] == 10
    assert best["sample_index"] == 2
    assert best["mean_confidence"] == pytest.approx(0.85)
    assert best["image_url"].endswith("/frames/2/image")
    assert webapp.highlighted_frame([], "job") is None
    empty = webapp.highlighted_frame([
        {"frame_index": 0, "timestamp_ms": 0, "detection_count": 0,
         "detections": []}], "job")
    assert empty["mean_confidence"] is None


def test_completed_job_includes_highlight_and_downloadable_image(client):
    job_id = submit(client).json()["id"]
    store = client.app.state.store
    path = store.directory / job_id / "result"
    (path / "frames").mkdir(parents=True)
    frame = {"frame_index": 12, "timestamp_ms": 400, "detection_count": 1,
             "detections": [{"confidence": 0.9}]}
    (path / "detections.jsonl").write_text(json.dumps(frame), encoding="utf-8")
    (path / "frames/000000.jpg").write_bytes(b"jpeg fixture")
    assert client.get(f"/api/v1/analyses/{job_id}").json()["highlighted_frame"] is None
    store.update(job_id, "completed")
    best = client.get(f"/api/v1/analyses/{job_id}").json()["highlighted_frame"]
    assert best["frame_index"] == 12
    assert best["mean_confidence"] == 0.9
    assert client.get(best["image_url"] + "?download=true").content == b"jpeg fixture"


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_download_frames_zip_contains_only_job_images(client, status):
    job_id = submit(client).json()["id"]
    store = client.app.state.store
    frames = store.directory / job_id / "result/frames"
    frames.mkdir(parents=True)
    (frames / "000000.jpg").write_bytes(b"first jpeg")
    (frames / "000002.jpg").write_bytes(b"third jpeg")
    (frames / "private.txt").write_text("not a frame")
    (frames / "000300.jpg").write_bytes(b"outside limit")
    url = f"/api/v1/analyses/{job_id}/frames.zip"
    assert client.get(url).status_code == 409
    store.update(job_id, status)
    response = client.get(url)
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.namelist() == ["000000.jpg", "000002.jpg"]
        assert archive.read("000000.jpg") == b"first jpeg"
        assert archive.read("000002.jpg") == b"third jpeg"
    assert client.get("/api/v1/analyses/missing/frames.zip").status_code == 404


def test_download_frames_zip_rejects_empty_results(client):
    job_id = submit(client).json()["id"]
    client.app.state.store.update(job_id, "completed")
    assert client.get(f"/api/v1/analyses/{job_id}/frames.zip").status_code == 409
