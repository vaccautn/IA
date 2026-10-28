"""Original-only passage inference on the owning loop, without real models."""

import asyncio
import json
import threading
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image
from fastapi import HTTPException

from vacca_video.bcs import VideoBCSCoordinator


def passage_job(tmp_path):
    directory = tmp_path / "fake/result/tracking"
    directory.mkdir(parents=True)
    rows, images = [], []
    for seq, track in enumerate((9, 2, 8), 1):
        stream = BytesIO()
        Image.new("RGB", (65, 49), (seq * 60, 20, 30)).save(stream, format="JPEG")
        images.append(stream.getvalue())
        target = directory / f"passages/{seq}"
        target.mkdir(parents=True)
        (target / "best-original.jpg").write_bytes(images[-1])
        (target / "best-crop.png").write_bytes(b"never infer crops")
        rows.append(dict(sequence=seq, track_id=track, status="passage_evidence_candidate",
            start_frame=seq * 10, end_frame=(seq + 1) * 10,
            start_timestamp_ms=seq * 100., end_timestamp_ms=(seq + 1) * 100.,
            end_exclusive=seq < 3, transition_method="initial_confidence", warnings=[],
            candidate=dict(status="available", track_id=track, frame_index=seq * 10,
                timestamp_ms=seq * 100., original=f"passages/{seq}/best-original.jpg",
                bcs_eligibility="not_assessed", bcs_status="not_evaluated")))
    summary = dict(passage_evidence=dict(status="provisional_evidence_not_unique_animal_identity",
        policy=dict(calibration="uncalibrated_confidence_proxy_not_visibility_or_bcs"),
        intervals=rows, unresolved_tracks=[], warnings=[]))
    (directory / "summary.json").write_text(json.dumps(summary))
    return directory, summary, images


def result(category=3):
    return SimpleNamespace(bcs_category=category, model_status="experimental_not_approved",
                           package_id="fake-package", inference_time_ms=1.25, lineage=None)


def finish(tmp_path, evaluate, *, stop=None, wait=1):
    async def run():
        coordinator = VideoBCSCoordinator(asyncio.get_running_loop(), evaluate, max_wait=wait)
        try:
            return await asyncio.to_thread(coordinator.finish_passages, tmp_path, "fake", stop or threading.Event())
        finally:
            await coordinator.close()
    return asyncio.run(run())


def test_originals_order_and_binding(tmp_path):
    from vacca_video.passage_bcs import read_passage_bcs
    directory, _, images = passage_job(tmp_path)
    before = (directory / "summary.json").read_bytes()
    calls = []
    async def evaluate(image, *, internal):
        assert internal
        calls.append(image)
        return result(len(calls))
    finish(tmp_path, evaluate)
    data = read_passage_bcs(tmp_path, "fake")
    assert calls == images
    assert [row["bcs_category"] for row in data["passages"]] == [1, 2, 3]
    assert [row["track_id"] for row in data["passages"]] == [9, 2, 8]
    assert (directory / "summary.json").read_bytes() == before
    (directory / "passages/1/best-original.jpg").write_bytes(images[1])
    with pytest.raises(ValueError):
        read_passage_bcs(tmp_path, "fake")


@pytest.mark.parametrize("failure,reason", [(503, "capability_unavailable"),
    (400, "image_rejected"), (500, "inference_failed"), ("busy", "capacity_busy"),
    ("timeout", "evaluation_timeout"), ("exception", "inference_failed"),
    ("category", "invalid_outcome"), ("lineage", "invalid_outcome"),
    ("nan", "invalid_outcome"), ("missing_fields", "invalid_outcome")])
def test_per_image_failures_continue(tmp_path, failure, reason):
    directory, _, _ = passage_job(tmp_path)
    calls = []
    async def evaluate(image, **kwargs):
        calls.append(image)
        if len(calls) == 2:
            if type(failure) is int:
                raise HTTPException(failure, "private path")
            if failure == "busy":
                raise HTTPException(503, "Inference capacity is busy; retry shortly")
            if failure == "timeout":
                await asyncio.sleep(1)
            if failure == "exception":
                raise RuntimeError("private path")
            if failure == "category":
                return result(True)
            if failure == "lineage":
                value = result(); value.lineage = {"run_id": "C:/private"}; return value
            if failure == "nan":
                value = result(); value.inference_time_ms = float("nan"); return value
            if failure == "missing_fields": return SimpleNamespace()
        return result()
    finish(tmp_path, evaluate, wait=.05)
    rows = json.loads((directory / "passage-bcs.json").read_bytes())["passages"]
    assert [row["status"] for row in rows] == ["evaluated", "unevaluable", "evaluated"]
    assert rows[1]["reason"] == reason and rows[1]["bcs_category"] is None
    assert "private" not in json.dumps(rows)


@pytest.mark.parametrize("defect", ["absent", "missing", "unsafe", "link", "dimension", "bytes"])
def test_invalid_original_never_falls_back(tmp_path, monkeypatch, defect):
    directory, summary, images = passage_job(tmp_path)
    row = summary["passage_evidence"]["intervals"][0]
    path = directory / "passages/1/best-original.jpg"
    if defect == "absent":
        row["candidate"] = dict(status="not_available", reason="no_candidate_captured",
            bcs_eligibility="not_assessed", bcs_status="not_evaluated")
    elif defect == "missing": path.unlink()
    elif defect == "unsafe": row["candidate"]["original"] = "../secret.jpg"
    elif defect == "link":
        original = type(path).is_symlink
        monkeypatch.setattr(type(path), "is_symlink", lambda p: p == path or original(p))
    elif defect == "dimension": Image.new("RGB", (12001, 1)).save(path)
    else: path.write_bytes(b"invalid")
    (directory / "summary.json").write_text(json.dumps(summary))
    calls = []
    async def evaluate(image, **kwargs): calls.append(image); return result()
    finish(tmp_path, evaluate)
    assert calls == images[1:]
    rows = json.loads((directory / "passage-bcs.json").read_bytes())["passages"]
    assert rows[0]["status"] == "unevaluable" and rows[0]["bcs_category"] is None


@pytest.mark.parametrize("defect", ["sequence", "track", "boundary", "evidence", "candidate"])
def test_invalid_global_evidence_fails_before_inference(tmp_path, defect):
    directory, summary, _ = passage_job(tmp_path)
    evidence = summary["passage_evidence"]
    if defect == "sequence": evidence["intervals"][1]["sequence"] = 1
    elif defect == "track": evidence["intervals"][0]["candidate"]["track_id"] = 2
    elif defect == "boundary": evidence["intervals"][0]["end_exclusive"] = False
    elif defect == "evidence": evidence["status"] = "clinical"
    else: evidence["intervals"][0]["candidate"]["frame_index"] = 999
    (directory / "summary.json").write_text(json.dumps(summary))
    async def never(*args, **kwargs): pytest.fail("Malformed evidence must not infer")
    with pytest.raises(RuntimeError): finish(tmp_path, never)
    assert not (directory / "passage-bcs.json").exists()


@pytest.mark.parametrize("mode", ["before", "during", "write"])
def test_interrupt_or_storage_failure_never_publishes(tmp_path, monkeypatch, mode):
    directory, _, _ = passage_job(tmp_path)
    stop = threading.Event()
    if mode == "before": stop.set()
    if mode == "write":
        def fail(*args): raise OSError("disk full")
        monkeypatch.setattr("vacca_video.artifacts.os.replace", fail)
    async def evaluate(*args, **kwargs):
        if mode == "during": stop.set()
        return result()
    with pytest.raises(RuntimeError): finish(tmp_path, evaluate, stop=stop)
    assert not (directory / "passage-bcs.json").exists()
    assert not list(directory.glob(".passage-bcs-*"))


@pytest.mark.parametrize("mode", ["completed", "failed", "disabled", "no_tracking", "no_hook", "write_error"])
def test_worker_phase_order_and_final_persistence(tmp_path, monkeypatch, mode):
    from vacca_video.store import Store
    from vacca_video.worker import Worker
    from vacca_video.tracking_job import write_status
    from vacca_video.webapp import _tracking_state
    from test_video_tracks_api import _Process

    directory, _, _ = passage_job(tmp_path)
    store = Store(tmp_path)
    store.add("fake", "fake.mp4")
    worker, calls = Worker(store), []
    monkeypatch.setattr("vacca_video.worker.subprocess.Popen", lambda *a, **kw: _Process())
    def legacy(*args): calls.append("legacy")
    def tracking(job_directory, stop):
        calls.append("tracking")
        write_status(job_directory / "result", mode if mode in {"failed", "disabled"} else "completed")
    worker.finish_video = legacy
    worker.track_video = None if mode == "no_tracking" else tracking
    if mode == "write_error":
        def fail(*args): raise OSError("disk full")
        monkeypatch.setattr("vacca_video.artifacts.os.replace", fail)
        # Tracking status publication is not part of this injected sidecar failure.
        def tracked(job_directory, stop):
            calls.append("tracking")
            (job_directory / "result/tracking-status.json").write_text('{"status":"completed"}')
        worker.track_video = tracked
    update = store.update
    def settle(identifier, status, error=None):
        calls.append("settled")
        if mode == "completed":
            assert (directory / "passage-bcs.json").is_file()
        update(identifier, status, error)
        worker.stop.set()
    monkeypatch.setattr(store, "update", settle)
    async def evaluate(image, **kwargs):
        calls.append("bcs")
        job = store.get("fake")
        assert job["status"] == "processing"
        assert _tracking_state(tmp_path, job)["status"] == "pending"
        assert not (directory / "passage-bcs.json").exists()
        return result()
    async def run():
        coordinator = VideoBCSCoordinator(asyncio.get_running_loop(), evaluate)
        worker.finish_passages = None if mode == "no_hook" else coordinator.finish_passages
        try:
            await asyncio.wait_for(asyncio.to_thread(worker.loop), 3)
        finally:
            await coordinator.close()
    asyncio.run(run())
    assert calls[0] == "legacy" and calls[-1] == "settled"
    expected = mode in {"completed", "write_error"}
    assert calls.count("bcs") == (3 if expected else 0)
    if expected: assert calls[1] == "tracking"
    assert store.get("fake")["status"] == ("failed" if mode == "write_error" else "completed")
    assert (directory / "passage-bcs.json").exists() == (mode == "completed")


def test_empty_sidecar_and_summary_digest_validation(tmp_path):
    from vacca_video.passage_bcs import read_passage_bcs
    directory, summary, _ = passage_job(tmp_path)
    summary["passage_evidence"]["intervals"] = []
    (directory / "summary.json").write_text(json.dumps(summary))
    async def never(*args, **kwargs): pytest.fail("Zero passages must not infer")
    finish(tmp_path, never)
    assert read_passage_bcs(tmp_path, "fake")["passages"] == []
    with (directory / "summary.json").open("ab") as stream: stream.write(b" ")
    with pytest.raises(ValueError, match="Stale"):
        read_passage_bcs(tmp_path, "fake")


@pytest.mark.parametrize("field,value", [("sequence", 2), ("frame_index", 11),
    ("timestamp_ms", 101.), ("input_sha256", "0" * 64), ("bcs_category", True)])
def test_sidecar_cannot_join_tampered_bindings(tmp_path, field, value):
    from vacca_video.passage_bcs import read_passage_bcs
    directory, _, _ = passage_job(tmp_path)
    async def evaluate(*args, **kwargs): return result()
    finish(tmp_path, evaluate)
    path = directory / "passage-bcs.json"
    sidecar = json.loads(path.read_bytes())
    sidecar["passages"][0][field] = value
    path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError): read_passage_bcs(tmp_path, "fake")


def test_sidecar_bound_applies_to_reads_and_publication(tmp_path):
    from vacca_video.artifacts import PASSAGE_BCS_LIMIT, publish_passage_bcs, read_tracking_json
    directory, _, _ = passage_job(tmp_path)
    with pytest.raises(ValueError):
        publish_passage_bcs(tmp_path, "fake", {"oversize": "x" * PASSAGE_BCS_LIMIT}, lambda: None)
    assert not (directory / "passage-bcs.json").exists()
    (directory / "passage-bcs.json").write_bytes(b" " * (PASSAGE_BCS_LIMIT + 1))
    with pytest.raises(ValueError): read_tracking_json(tmp_path, "fake", "passage-bcs.json")


def test_close_interrupts_pending_passage_without_sidecar(tmp_path):
    directory, _, _ = passage_job(tmp_path)
    async def run():
        entered = asyncio.Event()
        async def evaluate(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()
        coordinator = VideoBCSCoordinator(asyncio.get_running_loop(), evaluate)
        waiter = asyncio.create_task(asyncio.to_thread(
            coordinator.finish_passages, tmp_path, "fake", threading.Event()))
        await asyncio.wait_for(entered.wait(), 1)
        await coordinator.close()
        with pytest.raises(RuntimeError, match="interrumpió"):
            await asyncio.wait_for(waiter, 1)
        assert not coordinator.tasks
    asyncio.run(run())
    assert not (directory / "passage-bcs.json").exists()


@pytest.mark.parametrize("mode", ["success", "busy", "timeout"])
def test_passages_use_native_parent_evaluator_and_retain_capacity(tmp_path, monkeypatch, mode):
    from vacca_api import main
    from vacca_api.metrics import PrototypeMetrics
    from vacca_video.passage_bcs import read_passage_bcs

    _, _, images = passage_job(tmp_path)
    calls, release = [], threading.Event()
    lineage = SimpleNamespace(checkpoint_schema_version="fake-v1", domain_id="fake-domain",
        snapshot_schema="snapshot-v1", snapshot_identity="fake-identity", dataset_manifest_digest="fake-digest",
        run_id="fake-run", source_schema="source-v1", source_identity_scheme="identity-v1",
        observed_classes=(1, 2, 3, 4, 5), missing_classes=(), source_mapping=(("3.25", 1),),
        checkpoint_path="C:/private/checkpoint")
    def infer(image):
        calls.append(image)
        if mode == "timeout":
            assert release.wait(5)
        return SimpleNamespace(bcs_category=4, lineage=lineage)
    service = SimpleNamespace(infer=infer)
    runtime = SimpleNamespace(model_status="experimental_not_approved", package_id="fake-package",
                              get_service=lambda: service)
    monkeypatch.setattr(main, "get_bcs_runtime", lambda: runtime)
    monkeypatch.setattr(main.app.state, "metrics", PrototypeMetrics())
    async def run():
        gate = main.InferenceCapacityGate("bcs", acquisition_timeout=.01)
        monkeypatch.setattr(main.app.state, "bcs_inference_gate", gate)
        coordinator = VideoBCSCoordinator(asyncio.get_running_loop(), main.evaluate_bcs, max_wait=.15)
        if mode == "busy":
            await gate.acquire()
        try:
            await asyncio.to_thread(coordinator.finish_passages, tmp_path, "fake", threading.Event())
            if mode == "timeout":
                with pytest.raises(main.InferenceCapacityBusyError):
                    await gate.acquire()
        finally:
            release.set()
            if mode == "busy":
                gate.release()
            await coordinator.close()
        for _ in range(100):
            if not gate._work_threads:
                break
            await asyncio.sleep(.01)
        assert not gate._work_threads
        await gate.acquire()
        gate.release()
    asyncio.run(run())
    rows = read_passage_bcs(tmp_path, "fake")["passages"]
    if mode == "success":
        assert calls == images  # Full original JPEG bytes, never the crop sentinel.
        assert [r["bcs_category"] for r in rows] == [4, 4, 4]
        assert rows[0]["lineage"]["source_mapping"] == {"3.25": 1}
        assert "private" not in json.dumps(rows)
    else:
        assert len(calls) == (1 if mode == "timeout" else 0)
        assert [r["reason"] for r in rows] == (["evaluation_timeout", "capacity_busy", "capacity_busy"]
                                               if mode == "timeout" else ["capacity_busy"] * 3)
        assert all(r["bcs_category"] is None for r in rows)
