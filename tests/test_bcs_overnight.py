from __future__ import annotations

import io
from pathlib import Path

import pytest

from scripts.run_bcs_overnight import (
    BACKEND_OUTPUT_RELATIVE,
    BACKEND_SNAPSHOT_RELATIVE,
    LOCAL_OUTPUT_RELATIVE,
    LOCAL_ROOT_RELATIVE,
    LOCAL_SNAPSHOT_RELATIVE,
    main,
)


def _repo(tmp_path: Path, *, source: str = "local", ready: bool = False) -> Path:
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    for path in ("build_bcs_integer.py", "train_bcs_ordinal.py"):
        (tmp_path / "scripts" / path).touch()
    (tmp_path / "configs/training_bcs_ordinal.yaml").write_text(
        "data_root: data/bcs-local-integer-v1\noutput_dir: outputs/bcs-ordinal-local-integer-v1\n",
        encoding="utf-8",
    )
    if source == "local":
        (tmp_path / LOCAL_ROOT_RELATIVE).mkdir(parents=True)
        snapshot = tmp_path / LOCAL_SNAPSHOT_RELATIVE
        output = tmp_path / LOCAL_OUTPUT_RELATIVE
    else:
        snapshot = tmp_path / BACKEND_SNAPSHOT_RELATIVE
        output = tmp_path / BACKEND_OUTPUT_RELATIVE
    if ready:
        (snapshot / "manifest.json").parent.mkdir(parents=True)
        (snapshot / "manifest.json").touch()
        (output / "weights").mkdir(parents=True)
        (output / "weights/last.pt").touch()
        (output / "results_lineage.json").touch()
    return tmp_path


def _runner(calls, *, fail: str | None = None, make_best: bool = True):
    def fake(command, *, cwd, environment, log_path):
        calls.append((list(command), cwd, log_path))
        log_path.write_text(environment.get("VACCA_BACKEND_TOKEN", ""), encoding="utf-8")
        if fail and command[1].endswith(fail):
            return 2
        if make_best and command[1].endswith("train_bcs_ordinal.py"):
            local = "--data-dir" not in command
            output = cwd / (LOCAL_OUTPUT_RELATIVE if local else BACKEND_OUTPUT_RELATIVE)
            (output / "weights/best.pt").parent.mkdir(parents=True, exist_ok=True)
            (output / "weights/best.pt").touch()
        return 0

    return fake


def test_local_preflight_help_and_missing_root_are_safe(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    calls = []
    assert main(["--preflight-only"], root=root, environment={}, runner=_runner(calls)) == 0
    assert not calls
    with pytest.raises(SystemExit):
        main(["--help"], root=root)
    error = io.StringIO()
    assert main(["--local-root", str(tmp_path / "missing"), "--preflight-only"], root=root, environment={}, stderr=error) == 1
    assert "local source root" in error.getvalue() and not calls


def test_local_sequence_uses_distinct_exact_args_and_redacts_token(tmp_path: Path, capsys) -> None:
    root = _repo(tmp_path)
    calls = []
    env = {"VACCA_BACKEND_TOKEN": "secret"}
    assert main([], root=root, environment=env, runner=_runner(calls)) == 0
    build, train = calls
    assert build[0][2:] == ["--source", "local", "--local-root", str(root / LOCAL_ROOT_RELATIVE), "--output", str(root / LOCAL_SNAPSHOT_RELATIVE)]
    assert train[0][2:] == ["--config", str(root / "configs/training_bcs_ordinal.yaml")]
    assert "secret" not in build[2].read_text(encoding="utf-8")
    assert "secret" not in capsys.readouterr().out


def test_backend_requires_env_and_keeps_backend_roots(tmp_path: Path) -> None:
    root = _repo(tmp_path, source="backend")
    calls = []
    error = io.StringIO()
    assert main(["--source", "backend"], root=root, environment={"VACCA_BACKEND_URL": "x"}, runner=_runner(calls), stderr=error) == 1
    assert not calls and "VACCA_BACKEND_TOKEN" in error.getvalue()
    env = {"VACCA_BACKEND_URL": "https://backend.test", "VACCA_BACKEND_TOKEN": "secret"}
    assert main(["--source", "backend"], root=root, environment=env, runner=_runner(calls)) == 0
    assert calls[0][0][2:] == ["--source", "backend", "--output", str(root / BACKEND_SNAPSHOT_RELATIVE)]
    assert calls[1][0][2:] == ["--config", str(root / "configs/training_bcs_ordinal.yaml"), "--data-dir", str(root / BACKEND_SNAPSHOT_RELATIVE), "--output", str(root / BACKEND_OUTPUT_RELATIVE)]
    error = io.StringIO()
    assert main(["--source", "backend", "--local-root", "other"], root=root, environment=env, stderr=error) == 1
    assert not calls[2:] and "local-root" in error.getvalue()


@pytest.mark.parametrize("source", ["local", "backend"])
def test_skip_build_resume_uses_source_specific_paths(tmp_path: Path, source: str) -> None:
    root = _repo(tmp_path, source=source, ready=True)
    calls = []
    assert main(["--source", source, "--skip-build", "--resume"], root=root, environment={}, runner=_runner(calls)) == 0
    expected = (LOCAL_OUTPUT_RELATIVE if source == "local" else BACKEND_OUTPUT_RELATIVE) / "weights/last.pt"
    assert calls[0][0][-2:] == ["--resume", str(root / expected)]


def test_failures_stop_and_missing_best_is_nonzero(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    calls = []
    assert main([], root=root, environment={}, runner=_runner(calls, fail="build_bcs_integer.py"), stderr=io.StringIO()) == 1
    assert len(calls) == 1
    root = _repo(tmp_path / "train")
    calls = []
    assert main([], root=root, environment={}, runner=_runner(calls, fail="train_bcs_ordinal.py"), stderr=io.StringIO()) == 1
    assert len(calls) == 2
    root = _repo(tmp_path / "best")
    assert main([], root=root, environment={}, runner=_runner([], make_best=False), stderr=io.StringIO()) == 1
