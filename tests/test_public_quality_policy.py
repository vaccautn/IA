from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_public_workflow_has_no_private_download_or_test_access() -> None:
    workflow = (ROOT / ".github" / "workflows" / "public-quality.yml").read_text(encoding="utf-8")
    assert "--private-model-archive" not in workflow
    assert "VACCA_BCS_PRIVATE_ARCHIVE" not in workflow
    assert "google" not in workflow.casefold()
    assert "secrets." not in workflow


def test_readme_contains_exact_private_command_shape() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "pytest tests -m private_model --private-model-archive $privateArchive" in readme


def test_private_distribution_acceptance_is_internal_only_and_not_public_git() -> None:
    policy = (ROOT / "src" / "vacca_bcs" / "serving_package.py").read_text(encoding="utf-8")
    catalog_root = ROOT / "models" / "catalog" / "bcs-category-coral-2026-09-04"
    manifest = (catalog_root / "manifest.json").read_text(encoding="utf-8")
    model_card = (catalog_root / "MODEL_CARD.md").read_text(encoding="utf-8")
    notices = (catalog_root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    accepted = "maintainer_accepted_internal_private_team_prototype_distribution_2026-09-06"
    assert accepted in policy and accepted in manifest
    assert "undetermined" in policy and "IMAGENET1K_V1" in notices
    assert "no deben entrar en Git público" in model_card + notices
    assert "no constituye asesoramiento legal" in model_card + notices
    assert "no aprobado" in model_card and "producción" in model_card


def test_metrics_schema_and_documentation_use_service_impacting_rate() -> None:
    schema = (ROOT / "src" / "vacca_api" / "schemas.py").read_text(encoding="utf-8")
    metrics = (ROOT / "src" / "vacca_api" / "metrics.py").read_text(encoding="utf-8")
    api_doc = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    required = (
        "eligible_operational_requests",
        "service_impacting_failures",
        "service_impacting_failure_rate",
        "service_impacting_failure_rate_review_thresholds",
        "non_inference_wall_time_ms_total",
        "measurement_window_started_at_utc",
    )
    assert all(field in schema and field in metrics and field in api_doc for field in required)
    assert "server_runtime_failure_review_thresholds" not in schema + metrics + api_doc
    assert "server_runtime_failures + inference_failures" in api_doc
    assert "requests - client_rejections - busy_rejections" in api_doc


def test_private_marker_without_readable_archive_fails_instead_of_skipping(tmp_path: Path) -> None:
    missing = tmp_path / "missing-private-model.zip"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests",
            "-m",
            "private_model",
            "--private-model-archive",
            str(missing),
            "-q",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
    )
    assert result.returncode != 0
    assert "requires --private-model-archive PATH" in result.stdout + result.stderr


def test_marker_negative_and_absent_expressions_never_require_private_archive() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    for expression in ("not private_model", None):
        arguments = [sys.executable, "-m", "pytest", "tests/test_bcs_install.py"]
        if expression is not None:
            arguments.extend(["-m", expression])
        arguments.append("-q")
        result = subprocess.run(
            arguments,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            env=environment,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "requires --private-model-archive PATH" not in result.stdout + result.stderr


@pytest.mark.private_command_regression
def test_exact_documented_private_command_executes_private_tests(pytestconfig) -> None:
    archive = pytestconfig.getoption("--private-model-archive") or os.environ.get(
        "VACCA_BCS_PRIVATE_ARCHIVE"
    )
    if not archive or not Path(archive).is_file():
        pytest.skip("valid external private archive is required for this command regression")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests",
            "-m",
            "private_model",
            "--private-model-archive",
            str(archive),
            "-q",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        env=environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert re.search(r"\b\d+ passed\b", result.stdout)
    assert "0 passed" not in result.stdout
    assert "deselected" in result.stdout or "skipped" in result.stdout


@pytest.mark.private_command_regression
def test_conjunctive_private_command_executes_installer_tests(pytestconfig) -> None:
    archive = pytestconfig.getoption("--private-model-archive") or os.environ.get(
        "VACCA_BCS_PRIVATE_ARCHIVE"
    )
    if not archive or not Path(archive).is_file():
        pytest.skip("valid external private archive is required for this command regression")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_bcs_install.py",
            "-m",
            "private_model and private_model_installed",
            "--private-model-archive",
            str(archive),
            "--private-model-install",
            "-q",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        env=environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    match = re.search(r"\b(\d+) passed\b", result.stdout)
    assert match and int(match.group(1)) > 0, result.stdout + result.stderr


@pytest.mark.private_command_regression
def test_or_private_command_executes_consumer_tests(pytestconfig) -> None:
    archive = pytestconfig.getoption("--private-model-archive") or os.environ.get(
        "VACCA_BCS_PRIVATE_ARCHIVE"
    )
    if not archive or not Path(archive).is_file():
        pytest.skip("valid external private archive is required for this command regression")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_bcs_install.py",
            "-m",
            "private_model or private_source_model",
            "--private-model-archive",
            str(archive),
            "--private-model-install",
            "-q",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        env=environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    match = re.search(r"\b(\d+) passed\b", result.stdout)
    assert match and int(match.group(1)) > 0, result.stdout + result.stderr
