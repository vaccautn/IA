"""Persisted provisional passage API, using synthetic metadata and bytes only."""

import json
import os
import threading

import pytest

from test_video_tracks_api import client, tracked_job
from test_video_web import TestClient, submit
from vacca_video import tracking_job


@pytest.mark.parametrize("metadata", ["missing", "running", "completed"])
def test_worker_keeps_written_tracking_private_until_pipeline_settles(client, monkeypatch, metadata):
    from test_video_tracks_api import _Process
    from vacca_video.worker import Worker

    job_id, result, _ = passage_job(client)
    summary = json.loads((result / "tracking/summary.json").read_text(encoding="utf-8"))
    summary["tracks"] = [{"track_id": 3, "status": "review", "warnings": [],
                           "clip": "tracks/3/clip.webm"}]
    (result / "tracking/summary.json").write_text(json.dumps(summary), encoding="utf-8")
    store = client.app.state.store
    store.update(job_id, "pending")
    if metadata == "missing":
        (result / tracking_job.TRACKING_STATUS).unlink()
    else:
        tracking_job.write_status(result, metadata)
    worker = Worker(store)
    entered, release = threading.Event(), threading.Event()

    def hook(directory, stop):
        entered.set()
        assert release.wait(5), "test did not release tracking hook"
        tracking_job.write_status(result, "completed")

    worker.track_video = hook
    update = store.update

    def finish(identifier, status, error=None):
        update(identifier, status, error)
        worker.stop.set()

    monkeypatch.setattr(store, "update", finish)
    monkeypatch.setattr("vacca_video.worker.subprocess.Popen", lambda *a, **k: _Process())
    worker.thread.start()
    try:
        assert entered.wait(5)
        assert store.get(job_id)["status"] == "processing"
        assert submit(client).status_code == 202
        assert submit(client).status_code == 202
        assert submit(client).status_code == 429
        for route in ("tracks", "passages"):
            assert client.get(f"/api/v1/analyses/{job_id}/{route}").json()["status"] == "pending"
        for route in ("tracks/3/clip", "passages/1/best-original"):
            assert client.get(f"/api/v1/analyses/{job_id}/{route}").status_code == 404
    finally:
        release.set()
        worker.stop.set()
        worker.thread.join(5)
    assert not worker.thread.is_alive()
    assert store.get(job_id)["status"] == "completed"
    assert submit(client).status_code == 202
    for route in ("tracks", "passages"):
        assert client.get(f"/api/v1/analyses/{job_id}/{route}").json()["status"] == "completed"
    for route in ("tracks/3/clip", "passages/1/best-original"):
        assert client.get(f"/api/v1/analyses/{job_id}/{route}").status_code == 200


def test_restart_recovery_hides_partial_pipeline_artifacts(client):
    job_id, result, _ = passage_job(client)
    store = client.app.state.store
    store.update(job_id, "processing")
    tracking_job.write_status(result, "running")
    store.recover()
    assert tracking_job.recover_interrupted(store.directory) == 1
    assert store.get(job_id)["status"] == "failed"
    assert json.loads((result / tracking_job.TRACKING_STATUS).read_text()) == {
        "status": "failed", "reason": "interrupted"}
    assert (result / "tracking/passages/1/best-original.jpg").is_file()
    for route in ("tracks", "passages"):
        assert client.get(f"/api/v1/analyses/{job_id}/{route}").json()["reason"] == "video_not_completed"
    for route in ("tracks/3/clip", "passages/1/best-original"):
        assert client.get(f"/api/v1/analyses/{job_id}/{route}").status_code == 404


def passage_job(client):
    job_id, result = tracked_job(client)
    intervals = []
    for sequence, track_id in enumerate((9, 2, 8, 1), 1):
        prefix = f"passages/{sequence}"
        candidate = dict(status="available", original=f"{prefix}/best-original.jpg",
                         best_crop=f"{prefix}/best-crop.png", frame_index=sequence * 10,
                         timestamp_ms=sequence * 100., track_id=track_id,
                         bcs_eligibility="not_assessed", bcs_status="not_evaluated",
                         segmentation=dict(status="ok", best_mask=f"{prefix}/best-mask.png",
                                           best_masked=f"{prefix}/best-masked.png"))
        directory = result / "tracking" / prefix
        directory.mkdir(parents=True)
        for name in ("best-original.jpg", "best-crop.png", "best-mask.png", "best-masked.png"):
            (directory / name).write_bytes(name.encode())
        intervals.append(dict(sequence=sequence, track_id=track_id,
                              status="passage_evidence_candidate", start_frame=sequence * 10,
                              end_frame=(sequence + 1) * 10, start_timestamp_ms=sequence * 100.,
                              end_timestamp_ms=(sequence + 1) * 100., end_exclusive=sequence < 4,
                              transition_method="initial_confidence" if sequence == 1 else "concurrent_confidence",
                              support={"count": 1}, warnings=[], candidate=candidate))
    evidence = dict(status="provisional_evidence_not_unique_animal_identity",
                    policy={"calibration": "uncalibrated_confidence_proxy_not_visibility_or_bcs"},
                    intervals=intervals, warnings=[], unresolved_tracks=[dict(
                        track_id=7, first_frame=1, last_frame=9, first_timestamp_ms=10.,
                        last_timestamp_ms=90., possible_continuation_of=9, possible_continued_by=None,
                        status="unresolved_passage_attribution")])
    save(result, evidence)
    return job_id, result, evidence


def save(result, evidence):
    (result / "tracking/summary.json").write_text(json.dumps({"passage_evidence": evidence}), encoding="utf-8")


def listing(client, job_id):
    response = client.get(f"/api/v1/analyses/{job_id}/passages")
    assert response.status_code == 200
    return response.json()


def test_order_semantics_and_allowlisted_metadata(client):
    job_id, result, evidence = passage_job(client)
    evidence["intervals"].reverse()
    evidence["intervals"][0]["support"] = {"private": "C:/secret"}
    evidence["warnings"] = ["C:/private/error", "possible_fragment_identity_unresolved"]
    save(result, evidence)
    data = listing(client, job_id)
    assert data["status"] == "completed"
    assert data["evidence_status"] == "provisional_evidence_not_unique_animal_identity"
    assert data["identity_semantics"] == "temporal_track_id_within_video"
    assert data["calibration"] == "uncalibrated_confidence_proxy_not_visibility_or_bcs"
    assert [p["track_id"] for p in data["passages"]] == [9, 2, 8, 1]
    assert [p["sequence"] for p in data["passages"]] == [1, 2, 3, 4]
    assert [p["end_exclusive"] for p in data["passages"]] == [True, True, True, False]
    assert data["unresolved_count"] == 1 and data["unresolved_tracks"][0]["track_id"] == 7
    candidate = data["passages"][0]["candidate"]
    assert candidate["status"] == "available" and candidate["bcs_status"] == "not_evaluated"
    assert candidate["bcs_eligibility"] == "not_assessed"
    assert candidate["best_original_url"] == f"/api/v1/analyses/{job_id}/passages/1/best-original"
    assert not any(value in json.dumps(data) for value in ("C:/", "best-original.jpg", "support", "bcs_category", "score"))


@pytest.mark.parametrize("kind,media", [("best-original", "image/jpeg"), ("best-crop", "image/png"),
                                      ("best-mask", "image/png"), ("best-masked", "image/png")])
def test_authorized_media(client, kind, media):
    job_id, _, _ = passage_job(client)
    response = client.get(f"/api/v1/analyses/{job_id}/passages/1/{kind}")
    assert response.status_code == 200 and response.headers["content-type"] == media
    assert response.content == (kind + (".jpg" if kind == "best-original" else ".png")).encode()
    assert "content-disposition" not in response.headers
    partial = client.get(f"/api/v1/analyses/{job_id}/passages/1/{kind}", headers={"range": "bytes=0-3"})
    assert partial.status_code == 206 and partial.content == response.content[:4]


@pytest.mark.parametrize("mode", ["candidate_less", "missing_crop", "rejected_mask", "missing_mask"])
def test_optional_artifacts_and_candidateless_intervals(client, mode):
    job_id, result, evidence = passage_job(client)
    candidate = evidence["intervals"][0]["candidate"]
    kind = "best-original" if mode == "candidate_less" else "best-crop" if mode == "missing_crop" else "best-mask"
    if mode == "candidate_less":
        evidence["intervals"][0]["candidate"] = dict(status="not_available", reason="no_candidate_captured",
            bcs_eligibility="not_assessed", bcs_status="not_evaluated")
    elif mode == "rejected_mask":
        candidate["segmentation"] = {"status": "rejected", "reason": "private exception"}
    else:
        (result / "tracking/passages/1" / (kind + ".png")).unlink()
    save(result, evidence)
    data = listing(client, job_id)
    assert len(data["passages"]) == 4
    assert data["passages"][0]["candidate"][kind.replace("-", "_") + "_url"] is None
    assert client.get(f"/api/v1/analyses/{job_id}/passages/1/{kind}").status_code == 404


@pytest.mark.parametrize("state,expected", [("running", "pending"), ("failed", "failed"),
                                           ("disabled", "disabled"), ("absent", "unavailable")])
def test_tracking_gates(client, state, expected):
    job_id, result, _ = passage_job(client)
    if state == "absent":
        (result / tracking_job.TRACKING_STATUS).unlink()
    else:
        tracking_job.write_status(result, state)
    assert listing(client, job_id)["status"] == expected
    assert client.get(f"/api/v1/analyses/{job_id}/passages/1/best-original").status_code == 404


@pytest.mark.parametrize("defect", ["legacy", "json", "empty", "duplicate", "zero", "string", "nan",
    "reverse_time", "bad_type", "missing_candidate", "absolute", "escape", "other_sequence", "bcs",
    "missing_crop_ref", "bad_unresolved", "bad_status", "oversize", "nested", "bool_sequence",
    "infinity", "sparse_sequence", "duplicate_track", "candidate_track", "boundary", "missing_mask_ref"])
def test_invalid_or_legacy_evidence_is_controlled(client, defect):
    job_id, result, evidence = passage_job(client)
    first = evidence["intervals"][0]
    if defect == "duplicate": evidence["intervals"][1]["sequence"] = 1
    elif defect == "zero": first["sequence"] = 0
    elif defect == "string": first["sequence"] = "1"
    elif defect == "nan": first["start_timestamp_ms"] = float("nan")
    elif defect == "reverse_time": evidence["intervals"][1]["start_timestamp_ms"] = 1.
    elif defect == "bad_type": first["candidate"] = []
    elif defect == "missing_candidate": first.pop("candidate")
    elif defect in ("absolute", "escape", "other_sequence"):
        first["candidate"]["original"] = {"absolute": "C:/secret.jpg", "escape": "../secret.jpg",
                                          "other_sequence": "passages/2/best-original.jpg"}[defect]
    elif defect == "bcs": first["candidate"]["bcs_status"] = "completed"
    elif defect == "missing_crop_ref": first["candidate"].pop("best_crop")
    elif defect == "bad_unresolved": evidence["unresolved_tracks"] = [None]
    elif defect == "bad_status": evidence["status"] = "clinical"
    elif defect == "bool_sequence": first["sequence"] = True
    elif defect == "infinity": first["end_timestamp_ms"] = float("inf")
    elif defect == "sparse_sequence": evidence["intervals"].pop(1)
    elif defect == "duplicate_track": evidence["intervals"][1]["track_id"] = first["track_id"]
    elif defect == "candidate_track": first["candidate"]["track_id"] = 2
    elif defect == "boundary": first["end_exclusive"] = False
    elif defect == "missing_mask_ref": first["candidate"]["segmentation"].pop("best_mask")
    elif defect == "empty": evidence["intervals"] = []; evidence["unresolved_tracks"] = []
    save(result, evidence)
    path = result / "tracking/summary.json"
    if defect == "legacy": path.write_text('{"tracks": []}')
    elif defect == "json": path.write_text("{")
    elif defect == "oversize": path.write_text(" " * (4 * 1024 * 1024 + 1))
    elif defect == "nested": path.write_text("[" * 2000)
    data = listing(client, job_id)
    assert data["status"] == ("completed" if defect == "empty" else "unavailable")
    assert data["passages"] == []
    assert client.get(f"/api/v1/analyses/{job_id}/passages/1/best-original").status_code == 404


@pytest.mark.parametrize("sequence", ["0", "-1", "x", "9999999999999999999999999", "5", "1%2F..", "1?sequence=2"])
def test_fixed_sequence_and_job_isolation(client, sequence):
    job_id, _, _ = passage_job(client)
    response = client.get(f"/api/v1/analyses/{job_id}/passages/{sequence}/best-original")
    assert response.status_code in (404, 422)
    other = submit(client).json()["id"]
    assert listing(client, other)["status"] == "pending"
    assert client.get(f"/api/v1/analyses/{other}/passages/1/best-original").status_code == 404
    assert client.get("/api/v1/analyses/missing/passages").status_code == 404
    assert client.get(f"/api/v1/analyses/{job_id}/passages/1/summary.json").status_code in (404, 422)


@pytest.mark.parametrize("part", ["best-original.jpg", "directory"])
def test_linked_artifacts_are_not_served(client, tmp_path, part):
    job_id, result, _ = passage_job(client)
    directory = result / "tracking/passages/1"
    path = directory if part == "directory" else directory / part
    outside = tmp_path / ("outside" if part == "directory" else "outside.jpg")
    path.rename(outside)
    try:
        os.symlink(outside, path, target_is_directory=part == "directory")
    except OSError:
        pytest.skip("Symbolic links are not available")
    assert client.get(f"/api/v1/analyses/{job_id}/passages/1/best-original").status_code == 404
    assert listing(client, job_id)["passages"][0]["candidate"]["best_original_url"] is None


@pytest.mark.parametrize("state", ["pending", "processing", "failed"])
def test_incomplete_video_hides_completed_tracking(client, state):
    job_id, _, _ = passage_job(client)
    client.app.state.store.update(job_id, state)
    assert listing(client, job_id)["status"] == ("unavailable" if state == "failed" else "pending")
    assert client.get(f"/api/v1/analyses/{job_id}/passages/1/best-original").status_code == 404


@pytest.mark.parametrize("recorded", ["{", '{"status": "failed", "reason": []}', '[]'])
def test_corrupt_tracking_status_is_controlled(client, recorded):
    job_id, result, _ = passage_job(client)
    (result / tracking_job.TRACKING_STATUS).write_text(recorded)
    assert listing(client, job_id)["status"] == "failed"
    assert client.get(f"/api/v1/analyses/{job_id}/passages/1/best-original").status_code == 404


def test_junction_detection_without_windows_privileges(client, monkeypatch):
    from pathlib import Path

    job_id, result, _ = passage_job(client)
    junction = result / "tracking/passages"
    monkeypatch.setattr(Path, "is_junction", lambda path: path == junction)
    assert client.get(f"/api/v1/analyses/{job_id}/passages/1/best-original").status_code == 404
    assert listing(client, job_id)["passages"][0]["candidate"]["best_original_url"] is None


def test_passages_openapi_and_local_only(client):
    from vacca_api import main

    schema = main.app.openapi()
    operation = schema["paths"]["/api/v1/analyses/{job_id}/passages"]["get"]
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/PassageListResponse"}
    media = schema["paths"]["/api/v1/analyses/{job_id}/passages/{sequence}/{kind}"]["get"]
    assert set(media["responses"]["200"]["content"]) == {"image/jpeg", "image/png"}
    assert set(media["responses"]["404"]["content"]) == {"application/json"}
    job_id, _, _ = passage_job(client)
    remote = TestClient(client.app, client_host="10.0.0.5")
    assert remote.get(f"/api/v1/analyses/{job_id}/passages").status_code == 403
