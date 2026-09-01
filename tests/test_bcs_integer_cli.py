from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from vacca_bcs.source_client import (
    BCSEvidencePayload,
    BCSSourceEvaluationRow,
    BCSSourceEvidence,
    BCSSourceExport,
    SCHEMA_VERSION,
)
from vacca_bcs.integer_snapshot import IntegerSnapshotMaterializationError
from vacca_bcs.local_source import LOCAL_BCS_MAPPING
from vacca_bcs.local_source import LocalSourceScanError
from vacca_bcs.source_plan import SourcePlanIntegrityError
from vacca_bcs.source_split_plan import IntegerSplitInputError
from scripts import build_bcs_integer

from scripts.build_bcs_integer import DEFAULT_LOCAL_OUTPUT, ROOT, build_parser, main, run


def _source_export() -> BCSSourceExport:
    rows = tuple(
        BCSSourceEvaluationRow(
            evaluation_id=100 + score,
            session_id=200 + score,
            animal_id=300 + score,
            valor_cc=score,
            evidence=(BCSSourceEvidence(score, f"private-{score}"),),
        )
        for score in (1, 2)
    )
    return BCSSourceExport(SCHEMA_VERSION, rows)


def _image() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (2, 2), (20, 40, 60)).save(stream, format="JPEG")
    return stream.getvalue()


class _FakeClient:
    instances: list[_FakeClient] = []

    def __init__(self, export: BCSSourceExport | None = None, **kwargs):
        self.export = export or _source_export()
        self.kwargs = kwargs
        self.closed = False
        self.__class__.instances.append(self)

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *_args) -> None:
        self.closed = True

    def fetch(self) -> BCSSourceExport:
        return self.export


class _FakeMaterializer:
    instances: list[_FakeMaterializer] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.__class__.instances.append(self)

    def __enter__(self) -> _FakeMaterializer:
        return self

    def __exit__(self, *_args) -> None:
        self.closed = True

    def materialize(self, evidence_id: int) -> BCSEvidencePayload:
        payload = _image()
        return BCSEvidencePayload(evidence_id, payload, hashlib.sha256(payload).hexdigest())


def _args(tmp_path: Path, *extra: str):
    return build_parser().parse_args(
        ["--source", "backend", "--base-url", "https://backend.test", "--output", str(tmp_path / "snapshot"), *extra]
    )


def _local_root(tmp_path: Path) -> Path:
    root = tmp_path / "local-source"
    for label, format_name in (("3.25", "JPEG"), ("4.0", "PNG")):
        folder = root / label
        folder.mkdir(parents=True)
        stream = io.BytesIO()
        Image.new("RGB", (2, 2), (20, 40, 60)).save(stream, format=format_name)
        (folder / f"cow.{format_name.lower() if format_name == 'PNG' else 'jpg'}").write_bytes(stream.getvalue())
    return root


def test_run_orchestrates_validated_integer_snapshot_and_closes_dependencies(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("VACCA_BACKEND_TOKEN", "secret-token")
    _FakeClient.instances.clear()
    _FakeMaterializer.instances.clear()

    snapshot = run(
        _args(tmp_path, "--seed", "17", "--val-ratio", "0.25", "--timeout", "3.5"),
        source_client_factory=_FakeClient,
        materializer_factory=_FakeMaterializer,
    )
    manifest = json.loads(snapshot.manifest_json)

    assert manifest["manifest_schema_version"] == "bcs-integer-snapshot-v2"
    assert snapshot.output_root == tmp_path / "snapshot"
    assert _FakeClient.instances[0].closed
    assert _FakeMaterializer.instances[0].closed
    assert _FakeClient.instances[0].kwargs == {
        "base_url": "https://backend.test",
        "bearer_token": "secret-token",
        "timeout": 3.5,
        "max_response_bytes": 64 * 1024 * 1024,
    }
    assert _FakeMaterializer.instances[0].kwargs["max_image_bytes"] == 16 * 1024 * 1024


def test_main_prints_only_safe_summary_and_reads_token_from_environment(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("VACCA_BACKEND_TOKEN", "super-secret-token")
    output = io.StringIO()

    assert main(
        ["--source", "backend", "--base-url", "https://backend.test", "--output", str(tmp_path / "snapshot")],
        source_client_factory=_FakeClient,
        materializer_factory=_FakeMaterializer,
        stdout=output,
    ) == 0
    rendered = output.getvalue()
    assert "super-secret-token" not in rendered
    assert "private-" not in rendered
    assert "https://backend.test" not in rendered
    summary = json.loads(rendered)
    assert {"snapshot_path", "included", "excluded", "counts", "plan_identity", "source_schema"} <= set(summary)


def test_main_rejects_missing_token_and_bool_like_numeric_arguments(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("VACCA_BACKEND_TOKEN", raising=False)
    for option, value in (("--seed", "true"), ("--val-ratio", "true"), ("--timeout", "nan")):
        error = io.StringIO()
        assert main(
            ["--source", "backend", "--base-url", "https://backend.test", "--output", str(tmp_path / value), option, value],
            stderr=error,
        ) == 1
        assert "super-secret" not in error.getvalue()
    error = io.StringIO()
    assert main(["--source", "backend", "--base-url", "https://backend.test"], stderr=error) == 1
    assert "token" in error.getvalue().lower()
    assert "https://backend.test" not in error.getvalue()


def test_main_sanitizes_client_materializer_snapshot_and_existing_root_failures(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("VACCA_BACKEND_TOKEN", "private-token")

    class FailingClient(_FakeClient):
        def fetch(self):
            raise RuntimeError("storage-key-and-signed-url")

    error = io.StringIO()
    assert main(
        ["--source", "backend", "--base-url", "https://backend.test"],
        source_client_factory=FailingClient,
        materializer_factory=_FakeMaterializer,
        stderr=error,
    ) == 1
    assert "storage-key-and-signed-url" not in error.getvalue()
    assert FailingClient.instances[-1].closed

    class FailingMaterializer(_FakeMaterializer):
        def materialize(self, evidence_id: int):
            raise RuntimeError("secret-storage-key")

    error = io.StringIO()
    assert main(
        ["--source", "backend", "--base-url", "https://backend.test", "--output", str(tmp_path / "failed")],
        source_client_factory=_FakeClient,
        materializer_factory=FailingMaterializer,
        stderr=error,
    ) == 1
    assert "secret-storage-key" not in error.getvalue()
    assert _FakeClient.instances[-1].closed
    assert FailingMaterializer.instances[-1].closed

    def fail_snapshot(*_args):
        raise RuntimeError("storage-key-from-builder")

    error = io.StringIO()
    assert main(
        ["--source", "backend", "--base-url", "https://backend.test", "--output", str(tmp_path / "snapshot-fail")],
        source_client_factory=_FakeClient,
        materializer_factory=_FakeMaterializer,
        snapshot_builder=fail_snapshot,
        stderr=error,
    ) == 1
    assert "storage-key-from-builder" not in error.getvalue()
    assert _FakeClient.instances[-1].closed
    assert _FakeMaterializer.instances[-1].closed

    existing = tmp_path / "existing"
    existing.mkdir()
    error = io.StringIO()
    assert main(
        ["--source", "backend", "--base-url", "https://backend.test", "--output", str(existing)],
        source_client_factory=_FakeClient,
        materializer_factory=_FakeMaterializer,
        stderr=error,
    ) == 1
    assert "private-token" not in error.getvalue()


@pytest.mark.parametrize(
    ("failure", "category"),
    (
        (LocalSourceScanError("safe scan detail"), "local-source"),
        (SourcePlanIntegrityError("safe plan detail"), "source-plan"),
        (IntegerSplitInputError("safe split detail"), "split-plan"),
        (IntegerSnapshotMaterializationError("safe snapshot detail"), "snapshot"),
    ),
)
def test_main_reports_typed_failures_with_safe_categories(failure, category, monkeypatch):
    monkeypatch.setattr(build_bcs_integer, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(failure))
    error = io.StringIO()

    assert main([], stderr=error) == 1
    assert f"ERROR [{category}]:" in error.getvalue()
    assert str(failure) in error.getvalue()


def test_main_reports_unexpected_failure_with_class_and_correlation_guidance(monkeypatch):
    monkeypatch.setattr(
        build_bcs_integer,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("storage-key-and-token")
        ),
    )
    error = io.StringIO()

    assert main([], stderr=error) == 1
    rendered = error.getvalue()
    assert "ERROR [unexpected]: RuntimeError" in rendered
    assert "correlation_id=" in rendered
    assert "storage-key-and-token" not in rendered


def test_parser_exposes_deterministic_operator_controls_without_token_argument() -> None:
    options = {action.dest for action in build_parser()._actions}
    assert {"source", "local_root", "base_url", "output", "seed", "val_ratio", "timeout", "max_source_bytes", "max_image_bytes"} <= options
    assert "token" not in options


def test_local_mode_uses_fixed_mapping_without_backend_environment_and_safe_summary(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("VACCA_BACKEND_URL", raising=False)
    monkeypatch.delenv("VACCA_BACKEND_TOKEN", raising=False)
    root = _local_root(tmp_path)
    output = tmp_path / "local-output"
    rendered = io.StringIO()
    assert main(
        ["--local-root", str(root), "--output", str(output)], stdout=rendered
    ) == 0
    summary = json.loads(rendered.getvalue())
    assert summary["source_schema"] == "bcs-local-folder-v1"
    assert summary["mapping"] == dict(LOCAL_BCS_MAPPING.entries)
    assert summary["observed_classes"] == [3, 4]
    assert summary["missing_classes"] == [1, 2, 5]
    assert str(root) not in rendered.getvalue()
    assert output.exists()


def test_source_defaults_outputs_and_incompatible_options_fail_closed(tmp_path: Path):
    args = build_parser().parse_args([])
    assert args.source == "local"
    assert args.local_root == "data/bcs/dataset"
    assert args.output is None
    for argv in (
        ["--base-url", "https://backend.test"],
        ["--source", "backend", "--local-root", str(tmp_path)],
    ):
        error = io.StringIO()
        assert main(argv, stderr=error) == 1
        assert "backend.test" not in error.getvalue()


def test_local_missing_root_fails_before_output_creation(tmp_path: Path):
    output = tmp_path / "not-created"
    error = io.StringIO()
    assert main(
        ["--local-root", str(tmp_path / "missing"), "--output", str(output)],
        stderr=error,
    ) == 1
    assert not output.exists()

    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["--local-root", str(empty), "--output", str(output)], stderr=error) == 1
    assert not output.exists()


def test_local_omitted_output_uses_local_default_without_publishing(tmp_path: Path):
    root = _local_root(tmp_path)
    seen = {}

    def capture(plan, output, materializer):
        seen["output"] = output
        seen["schema"] = plan.identity.source_schema
        return output

    result = run(
        build_parser().parse_args(["--local-root", str(root)]),
        snapshot_builder=capture,
    )
    assert result == ROOT / DEFAULT_LOCAL_OUTPUT
    assert seen == {"output": ROOT / DEFAULT_LOCAL_OUTPUT, "schema": "bcs-local-folder-v1"}
