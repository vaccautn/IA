"""Fake parent-runtime integration and cancellation capacity characterization."""

import asyncio
import threading
import json
from types import SimpleNamespace

import pytest

from vacca_api import main
from vacca_api.metrics import PrototypeMetrics


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
