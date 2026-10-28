"""Track gallery routes and the optional tracking step, without loading models."""

import json
import os
import threading

import pytest

from test_video_web import TestClient, submit
from vacca_video import tracking_job, webapp


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(
        webapp.shutil, "disk_usage", lambda path: type("Disk", (), {"free": 1024**3})()
    )
    yield TestClient(webapp.create_app(tmp_path, start_worker=False))


def tracked_job(client, status="completed", tracks=None, clip="clip.webm"):
    store = client.app.state.store
    job_id = submit(client).json()["id"]
    result = store.directory / job_id / "result"
    tracking = result / "tracking"
    (tracking / "tracks" / "3").mkdir(parents=True)
    for name in (clip, "best-crop.png", "best-original.jpg"):
        (tracking / "tracks" / "3" / name).write_bytes(name.encode())
    summary = {
        "schema_version": "vacca-video-tracking-v1", "input": "C:/private/input.mp4",
        "model": "C:/private/model.pt", "model_sha256": "abc", "tracker": "bytetrack.yaml",
        "fps": 24.0, "width": 464, "height": 688, "frames_read": 145, "frames_sampled": 145,
        "stop_reason": "end_of_video", "identity_semantics": "temporal_track_id_within_video",
        "warnings": [], "notes": ["nota"], "config": {"max_gap_seconds": 0.5},
        "tracks": tracks if tracks is not None else [{
            "track_id": 3, "observations": 13, "first_frame": 104, "last_frame": 143,
            "first_timestamp_ms": 4333.3, "last_timestamp_ms": 5958.3,
            "segments": [{"first_frame": 104}, {"first_frame": 130}], "gaps": [],
            "best_frame_index": 121, "best_bbox": [1, 2, 3, 4], "best_confidence": 0.73,
            "best_crop_contamination": 0.0, "custom": "kept",
            "clip": f"tracks/3/{clip}", "best_crop": "tracks/3/best-crop.png",
            "best_original": "tracks/3/best-original.jpg",
            "status": "review", "warnings": ["long_gap_review_required"],
        }],
    }
    (tracking / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    tracking_job.write_status(result, status)
    store.update(job_id, "completed")
    return job_id, result


def test_completed_tracks_expose_urls_not_disk_paths(client):
    job_id, _ = tracked_job(client)
    data = client.get(f"/api/v1/analyses/{job_id}/tracks").json()
    assert data["status"] == "completed"
    assert data["schema_version"] == "vacca-video-tracks-v1"
    assert data["identity_semantics"] == "temporal_track_id_within_video"
    assert "input" not in data and "model" not in data
    track = data["tracks"][0]
    prefix = f"/api/v1/analyses/{job_id}/tracks/3"
    assert track["clip_url"] == f"{prefix}/clip" and track["clip_media_type"] == "video/webm"
    assert track["best_crop_url"] == f"{prefix}/best-crop"
    assert track["best_original_url"] == f"{prefix}/best-original"
    assert not {"clip", "best_crop", "best_original"} & set(track)
    assert track["custom"] == "kept" and track["status"] == "review"


@pytest.mark.parametrize("suffix,body,media", [
    ("clip", b"clip.webm", "video/webm"),
    ("best-crop", b"best-crop.png", "image/png"),
    ("best-original", b"best-original.jpg", "image/jpeg"),
])
def test_track_files_are_served_with_their_media_type(client, suffix, body, media):
    job_id, _ = tracked_job(client)
    response = client.get(f"/api/v1/analyses/{job_id}/tracks/3/{suffix}")
    assert response.status_code == 200
    assert response.content == body
    assert response.headers["content-type"].startswith(media)
    assert "content-disposition" not in response.headers


def test_archival_mp4_clip_is_still_served(client):
    job_id, _ = tracked_job(client, clip="clip.mp4")
    data = client.get(f"/api/v1/analyses/{job_id}/tracks").json()
    assert data["tracks"][0]["clip_media_type"] == "video/mp4"
    response = client.get(f"/api/v1/analyses/{job_id}/tracks/3/clip")
    assert response.headers["content-type"].startswith("video/mp4")


@pytest.mark.parametrize("path", ["tracks/4/clip", "tracks/-1/clip", "tracks/x/clip"])
def test_unknown_or_invalid_tracks_are_not_found(client, path):
    job_id, _ = tracked_job(client)
    assert client.get(f"/api/v1/analyses/{job_id}/{path}").status_code in (404, 422)


def test_failed_tracking_keeps_partial_files_private(client):
    job_id, _ = tracked_job(client, status="failed")
    data = client.get(f"/api/v1/analyses/{job_id}/tracks").json()
    assert data == {"schema_version": "vacca-video-tracks-v1", "tracks": [],
                    "status": "failed", "reason": "unknown"}
    assert client.get(f"/api/v1/analyses/{job_id}/tracks/3/clip").status_code == 404


def test_quicktime_mov_uploads_are_accepted_like_mp4(client):
    for name in ("celular.MOV", "manga.mov"):
        response = client.post("/api/v1/analyses", params={"filename": name},
                               content=b"\x00\x00\x00\x14ftypqt  \x00\x00\x00\x00qt  ",
                               headers={"content-type": "video/quicktime"})
        assert response.status_code == 202
        job = response.json()
        assert job["name"] == name
        assert (client.app.state.store.directory / job["id"] / "input.mp4").is_file()
    rejected = client.post("/api/v1/analyses", params={"filename": "clip.avi"},
                           content=b"\x00\x00\x00\x14ftypqt  \x00\x00\x00\x00qt  ")
    assert rejected.status_code == 415
    assert rejected.json()["detail"] == "Elegí un video MP4 o MOV."


def test_accepted_silhouette_is_exposed_without_disk_paths(client):
    job_id, result = tracked_job(client)
    track_dir = result / "tracking" / "tracks" / "3"
    (track_dir / "best-masked.png").write_bytes(b"masked")
    (track_dir / "best-mask.png").write_bytes(b"mask")
    summary_path = result / "tracking" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["tracks"][0]["segmentation"] = {
        "status": "ok", "mask_fill": 0.4, "best_masked": "tracks/3/best-masked.png",
        "best_mask": "tracks/3/best-mask.png"}
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    track = client.get(f"/api/v1/analyses/{job_id}/tracks").json()["tracks"][0]
    prefix = f"/api/v1/analyses/{job_id}/tracks/3"
    assert track["best_masked_url"] == f"{prefix}/best-masked"
    assert track["best_mask_url"] == f"{prefix}/best-mask"
    assert track["segmentation"] == {"status": "ok", "mask_fill": 0.4}
    for suffix, body in (("best-masked", b"masked"), ("best-mask", b"mask")):
        response = client.get(f"{prefix}/{suffix}")
        assert response.content == body
        assert response.headers["content-type"].startswith("image/png")


def test_rejected_silhouette_has_no_urls(client):
    job_id, result = tracked_job(client)
    summary_path = result / "tracking" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["tracks"][0]["segmentation"] = {"status": "rejected", "reason": "mask_leaves_box"}
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    track = client.get(f"/api/v1/analyses/{job_id}/tracks").json()["tracks"][0]
    assert track["best_masked_url"] is None and track["best_mask_url"] is None
    assert client.get(f"/api/v1/analyses/{job_id}/tracks/3/best-masked").status_code == 404


def test_tracking_command_uses_local_segmentation_weights_only_when_present(tmp_path, monkeypatch):
    directory = _job_directory(tmp_path)
    monkeypatch.setattr(tracking_job, "SEGMENTATION_MODEL", tmp_path / "missing.pt")
    assert "--segment-model" not in tracking_job.tracking_command(directory)
    weights = tmp_path / "mobile_sam.pt"
    weights.write_bytes(b"weights")
    monkeypatch.setattr(tracking_job, "SEGMENTATION_MODEL", weights)
    command = tracking_job.tracking_command(directory)
    assert command[command.index("--segment-model") + 1] == str(weights)


def test_track_links_are_exposed(client):
    job_id, result = tracked_job(client)
    summary_path = result / "tracking" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["tracks"][0].update(possible_continuation_of=1, possible_continued_by=None)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    track = client.get(f"/api/v1/analyses/{job_id}/tracks").json()["tracks"][0]
    assert (track["possible_continuation_of"], track["possible_continued_by"]) == (1, None)


def test_known_failure_reason_is_reported(client):
    job_id, result = tracked_job(client)
    tracking_job.write_status(result, "failed", "timeout")
    assert client.get(f"/api/v1/analyses/{job_id}/tracks").json()["reason"] == "timeout"


def test_jobs_without_tracking_and_pending_jobs_are_explicit(client):
    pending = submit(client).json()["id"]
    assert client.get(f"/api/v1/analyses/{pending}/tracks").json()["status"] == "pending"
    assert client.get(f"/api/v1/analyses/{pending}/tracks/0/clip").status_code == 404
    store = client.app.state.store
    store.update(pending, "completed")
    data = client.get(f"/api/v1/analyses/{pending}/tracks").json()
    assert (data["status"], data["reason"]) == ("unavailable", "not_tracked")
    failed = submit(client).json()["id"]
    store.update(failed, "failed", "fixture")
    assert client.get(f"/api/v1/analyses/{failed}/tracks").json()["reason"] == "video_not_completed"
    assert client.get("/api/v1/analyses/missing/tracks").status_code == 404


def test_running_tracking_is_pending_and_hides_partial_files(client):
    job_id, result = tracked_job(client)
    tracking_job.write_status(result, "running")
    assert client.get(f"/api/v1/analyses/{job_id}/tracks").json()["status"] == "pending"
    assert client.get(f"/api/v1/analyses/{job_id}/tracks/3/clip").status_code == 404


def test_startup_recovery_settles_interrupted_tracking(client):
    running, running_result = tracked_job(client)
    done, done_result = tracked_job(client)
    tracking_job.write_status(running_result, "running")
    root = client.app.state.store.directory
    assert tracking_job.recover_interrupted(root) == 1
    assert _status(running_result.parent) == {"status": "failed", "reason": "interrupted"}
    assert _status(done_result.parent) == {"status": "completed"}
    assert client.get(f"/api/v1/analyses/{running}/tracks").json()["reason"] == "interrupted"


def test_resources_recover_tracking_only_when_enabled(tmp_path, monkeypatch):
    from vacca_video.service import VideoResources

    calls = []
    monkeypatch.setattr(tracking_job, "recover_interrupted", lambda root: calls.append(root))
    monkeypatch.setattr("vacca_video.service.check_runtime", lambda: None)
    for hook in (None, tracking_job.run_tracking):
        resources = VideoResources(tmp_path / str(len(calls)), worker_factory=webapp.Worker)
        resources.track_video = hook
        resources.start()
        assert resources.worker.track_video is hook
        resources.close()
    assert len(calls) == 1


def test_disabled_tracking_is_not_a_failure(client):
    job_id, result = tracked_job(client)
    tracking_job.write_status(result, "disabled", "disabled_by_configuration")
    data = client.get(f"/api/v1/analyses/{job_id}/tracks").json()
    assert (data["status"], data["reason"]) == ("disabled", "disabled_by_configuration")


@pytest.mark.parametrize("tracks", [
    [{"track_id": -1}],
    [{"track_id": "3", "status": "review", "warnings": []}],
    [{"track_id": 3, "status": "invented", "warnings": []}],
    "not a list",
])
def test_corrupt_tracking_summary_is_isolated(client, tracks):
    job_id, _ = tracked_job(client, tracks=tracks)
    data = client.get(f"/api/v1/analyses/{job_id}/tracks").json()
    assert (data["status"], data["reason"], data["tracks"]) == ("failed", "summary_unreadable", [])
    job = client.get(f"/api/v1/analyses/{job_id}").json()
    assert job["status"] == "completed"


def test_unreadable_summary_file_is_isolated(client):
    job_id, result = tracked_job(client)
    (result / "tracking" / "summary.json").write_text("{", encoding="utf-8")
    assert client.get(f"/api/v1/analyses/{job_id}/tracks").json()["reason"] == "summary_unreadable"


def test_linked_tracking_directory_is_rejected(client, tmp_path):
    job_id, result = tracked_job(client)
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    (result / "tracking").rename(outside)
    try:
        os.symlink(outside, result / "tracking", target_is_directory=True)
    except OSError:
        pytest.skip("Symbolic links are not available")
    assert client.get(f"/api/v1/analyses/{job_id}/tracks").json()["reason"] == "summary_unreadable"
    assert client.get(f"/api/v1/analyses/{job_id}/tracks/3/clip").status_code == 404


def test_job_contract_is_unchanged_by_tracks(client):
    import shutil

    job_id, result = tracked_job(client)
    with_tracks = client.get(f"/api/v1/analyses/{job_id}").json()
    shutil.rmtree(result / "tracking")
    (result / tracking_job.TRACKING_STATUS).unlink()
    assert client.get(f"/api/v1/analyses/{job_id}").json() == with_tracks
    assert not any("track" in key for key in with_tracks)


def test_tracks_routes_are_local_only(client):
    job_id, _ = tracked_job(client)
    remote = TestClient(client.app, client_host="10.0.0.5")
    assert remote.get(f"/api/v1/analyses/{job_id}/tracks").status_code == 403
    assert remote.get(f"/api/v1/analyses/{job_id}/tracks/3/clip").status_code == 403


class _Process:
    def __init__(self, returncode=0, finished=True, on_enter=None):
        self.returncode = returncode
        self.finished = finished
        self.on_enter = on_enter
        self.calls = []

    def __enter__(self):
        if self.on_enter:
            self.on_enter()
        return self

    def __exit__(self, *args):
        return None

    def poll(self):
        return self.returncode if self.finished else None

    def terminate(self):
        self.calls.append("terminate")

    def wait(self, timeout=None):
        self.calls.append(("wait", timeout))

    def kill(self):
        self.calls.append("kill")


def _job_directory(tmp_path):
    directory = tmp_path / "job"
    (directory / "result").mkdir(parents=True)
    (directory / "input.mp4").write_bytes(b"video")
    return directory


def _status(directory):
    return json.loads((directory / "result" / tracking_job.TRACKING_STATUS).read_text(encoding="utf-8"))


def test_tracking_command_is_dense_webm_and_time_bounded(tmp_path, monkeypatch):
    directory = _job_directory(tmp_path)
    seen = {}

    def spawn(command, **kwargs):
        seen.update(command=command, cwd=kwargs["cwd"])
        return _Process(on_enter=lambda: (directory / "result/tracking").mkdir()
                        or (directory / "result/tracking/summary.json").write_text("{}"))

    monkeypatch.setattr(tracking_job.subprocess, "Popen", spawn)
    assert tracking_job.run_tracking(directory, threading.Event()) == "completed"
    command = seen["command"]
    assert command[command.index("--sample-fps") + 1] == "0"
    assert command[command.index("--clip-format") + 1] == "webm"
    assert command[command.index("--max-seconds") + 1] == "60.0"
    assert command[command.index("--max-frames") + 1] == "1200"
    assert command[command.index("--output") + 1] == str(directory / "result" / "tracking")
    assert seen["cwd"] == tracking_job.ROOT
    assert _status(directory) == {"status": "completed"}


@pytest.mark.parametrize("process,reason", [
    (_Process(returncode=1), "process_failed"),
    (_Process(returncode=0), "summary_missing"),
])
def test_tracking_failures_are_recorded(tmp_path, monkeypatch, process, reason):
    directory = _job_directory(tmp_path)
    monkeypatch.setattr(tracking_job.subprocess, "Popen", lambda *args, **kwargs: process)
    assert tracking_job.run_tracking(directory, threading.Event()) == "failed"
    assert _status(directory) == {"status": "failed", "reason": reason}


def test_tracking_spawn_error_never_raises(tmp_path, monkeypatch):
    directory = _job_directory(tmp_path)

    def broken(*args, **kwargs):
        raise OSError("fixture")

    monkeypatch.setattr(tracking_job.subprocess, "Popen", broken)
    assert tracking_job.run_tracking(directory, threading.Event()) == "failed"
    assert _status(directory)["reason"] == "process_failed"


@pytest.mark.parametrize("shutdown,reason", [(False, "timeout"), (True, "interrupted")])
def test_tracking_timeout_and_shutdown_terminate_the_process(tmp_path, monkeypatch, shutdown, reason):
    directory = _job_directory(tmp_path)
    process = _Process(finished=False)
    monkeypatch.setattr(tracking_job.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(tracking_job, "TRACKING_TIMEOUT_SECONDS", 1)

    class Stop:
        stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, timeout):
            self.stopped = shutdown
            return self.stopped

    assert tracking_job.run_tracking(directory, Stop()) == "failed"
    assert process.calls[:2] == ["terminate", ("wait", 5)]
    assert _status(directory)["reason"] == reason


def test_tracking_can_be_disabled(tmp_path, monkeypatch):
    directory = _job_directory(tmp_path)
    monkeypatch.setenv(tracking_job.TRACKING_ENV, "0")
    monkeypatch.setattr(tracking_job.subprocess, "Popen", lambda *a, **k: pytest.fail("must not spawn"))
    assert tracking_job.run_tracking(directory, threading.Event()) == "disabled"
    assert _status(directory) == {"status": "disabled", "reason": "disabled_by_configuration"}


def _worker_run(client, monkeypatch, hook, detection_returncode=0):
    job_id = submit(client).json()["id"]
    store = client.app.state.store
    worker = webapp.Worker(store)
    worker.track_video = hook
    update = store.update

    def finished(identifier, status, error=None):
        update(identifier, status, error)
        worker.stop.set()

    monkeypatch.setattr(store, "update", finished)
    monkeypatch.setattr(webapp.subprocess, "Popen",
                        lambda *args, **kwargs: _Process(returncode=detection_returncode))
    worker.loop()
    return store.get(job_id)


def test_worker_publishes_detections_before_tracking(client, monkeypatch):
    calls = []
    store = client.app.state.store

    def hook(directory, stop):
        calls.append((directory.name, store.get(directory.name)["status"]))

    job = _worker_run(client, monkeypatch, hook)
    assert job["status"] == "completed" and job["error"] is None
    assert calls == [(job["id"], "completed")]


def test_tracking_hook_errors_never_fail_the_job(client, monkeypatch):
    def broken(directory, stop):
        raise RuntimeError("fixture")

    job = _worker_run(client, monkeypatch, broken)
    assert job["status"] == "completed" and job["error"] is None


def test_failed_detection_skips_tracking(client, monkeypatch):
    calls = []
    job = _worker_run(client, monkeypatch, lambda directory, stop: calls.append(1), detection_returncode=1)
    assert job["status"] == "failed"
    assert calls == []


def test_openapi_describes_track_routes():
    from vacca_api import main

    schema = main.app.openapi()
    listing = schema["paths"]["/api/v1/analyses/{job_id}/tracks"]["get"]
    assert listing["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/TrackListResponse"
    }
    for suffix, media in (("clip", {"video/webm", "video/mp4"}), ("best-crop", {"image/png"}),
                          ("best-original", {"image/jpeg"})):
        operation = schema["paths"][f"/api/v1/analyses/{{job_id}}/tracks/{{track_id}}/{suffix}"]["get"]
        assert set(operation["responses"]["200"]["content"]) == media
        for code in ("404", "422", "503"):
            assert set(operation["responses"][code]["content"]) == {"application/json"}
