from __future__ import annotations

import io
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from scripts.run_bcs_overnight import (
    BACKEND_OUTPUT_RELATIVE,
    BACKEND_SNAPSHOT_RELATIVE,
    LOCAL_OUTPUT_RELATIVE,
    LOCAL_ROOT_RELATIVE,
    LOCAL_SNAPSHOT_RELATIVE,
    _run_subprocess,
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


def _runner(
    calls, *, fail: str | None = None, make_best: bool = True, environments=None
):
    def fake(command, *, cwd, environment, log_path):
        calls.append((list(command), cwd, log_path))
        if environments is not None:
            environments.append((list(command), dict(environment)))
        log_path.write_text(environment.get("VACCA_BACKEND_TOKEN", ""), encoding="utf-8")
        script = next(part for part in command if part.endswith(".py"))
        if fail and script.endswith(fail):
            return 2
        if make_best and script.endswith("train_bcs_ordinal.py"):
            local = "--data-dir" not in command
            output = cwd / (LOCAL_OUTPUT_RELATIVE if local else BACKEND_OUTPUT_RELATIVE)
            (output / "weights/best.pt").parent.mkdir(parents=True, exist_ok=True)
            (output / "weights/best.pt").touch()
        return 0

    return fake


def _accept_best(*args) -> dict[str, object]:
    return {
        "best": {"epoch": 1, "mae": 0.0, "exact_acc": 1.0, "pm1_acc": 1.0},
        "final": {"epoch": 1},
    }


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
    environments = []
    env = {"VACCA_BACKEND_TOKEN": "secret"}
    assert main([], root=root, environment=env, runner=_runner(calls, environments=environments), best_validator=_accept_best) == 0
    build, train = calls
    assert build[0][2:] == ["--source", "local", "--local-root", str(root / LOCAL_ROOT_RELATIVE), "--output", str(root / LOCAL_SNAPSHOT_RELATIVE)]
    assert train[0][1:3] == ["-u", str(root / "scripts/train_bcs_ordinal.py")]
    assert train[0][3:] == ["--config", str(root / "configs/training_bcs_ordinal.yaml")]
    assert environments[0][1]["VACCA_BACKEND_TOKEN"] == "secret"
    assert "VACCA_BACKEND_TOKEN" not in environments[1][1]
    assert "VACCA_BACKEND_URL" not in environments[1][1]
    assert "secret" not in build[2].read_text(encoding="utf-8")
    assert "secret" not in capsys.readouterr().out


def test_backend_requires_env_and_keeps_backend_roots(tmp_path: Path) -> None:
    root = _repo(tmp_path, source="backend")
    calls = []
    environments = []
    error = io.StringIO()
    assert main(["--source", "backend"], root=root, environment={"VACCA_BACKEND_URL": "x"}, runner=_runner(calls), stderr=error) == 1
    assert not calls and "VACCA_BACKEND_TOKEN" in error.getvalue()
    env = {"VACCA_BACKEND_URL": "https://backend.test", "VACCA_BACKEND_TOKEN": "secret"}
    assert main(["--source", "backend"], root=root, environment=env, runner=_runner(calls, environments=environments), best_validator=_accept_best) == 0
    assert calls[0][0][2:] == ["--source", "backend", "--output", str(root / BACKEND_SNAPSHOT_RELATIVE)]
    assert calls[1][0][1:3] == ["-u", str(root / "scripts/train_bcs_ordinal.py")]
    assert calls[1][0][3:] == ["--config", str(root / "configs/training_bcs_ordinal.yaml"), "--data-dir", str(root / BACKEND_SNAPSHOT_RELATIVE), "--output", str(root / BACKEND_OUTPUT_RELATIVE)]
    assert environments[0][1]["VACCA_BACKEND_URL"] == "https://backend.test"
    assert environments[0][1]["VACCA_BACKEND_TOKEN"] == "secret"
    assert "VACCA_BACKEND_URL" not in environments[1][1]
    assert "VACCA_BACKEND_TOKEN" not in environments[1][1]
    error = io.StringIO()
    assert main(["--source", "backend", "--local-root", "other"], root=root, environment=env, stderr=error) == 1
    assert not calls[2:] and "local-root" in error.getvalue()


@pytest.mark.parametrize("source", ["local", "backend"])
def test_skip_build_resume_uses_source_specific_paths(tmp_path: Path, source: str) -> None:
    root = _repo(tmp_path, source=source, ready=True)
    calls = []
    assert main(["--source", source, "--skip-build", "--resume"], root=root, environment={}, runner=_runner(calls), best_validator=_accept_best) == 0
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


def test_real_unbuffered_child_progress_reaches_tee_log_before_exit(
    tmp_path: Path, capsys
) -> None:
    alive = tmp_path / "child-alive"
    log_path = tmp_path / "train.log"
    code = (
        "import pathlib, sys, time; "
        "alive = pathlib.Path(sys.argv[1]); alive.write_text('1'); "
        "print('[TRAIN 1/1 1/1] loss=1.0 elapsed=0.1s eta=0.0s', flush=True); "
        "time.sleep(1.0); alive.unlink()"
    )
    status: list[int] = []
    thread = threading.Thread(
        target=lambda: status.append(
            _run_subprocess(
                [sys.executable, "-u", "-c", code, str(alive)],
                cwd=tmp_path,
                environment={},
                log_path=log_path,
            )
        )
    )
    thread.start()
    deadline = time.monotonic() + 5.0
    observed_while_alive = False
    terminal_output = ""
    while time.monotonic() < deadline and thread.is_alive():
        terminal_output += capsys.readouterr().out
        log_output = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
        if (
            alive.is_file()
            and "[TRAIN 1/1 1/1]" in log_output
            and "[TRAIN 1/1 1/1]" in terminal_output
        ):
            observed_while_alive = True
            break
        time.sleep(0.01)
    thread.join(timeout=5.0)
    terminal_output += capsys.readouterr().out
    assert observed_while_alive
    assert status == [0]


def _sleeping_child_command(pid_path: Path) -> list[str]:
    code = (
        "import os, pathlib, sys, time; "
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
        "print('child-ready', flush=True); time.sleep(30)"
    )
    return [sys.executable, "-u", "-c", code, str(pid_path)]


def _assert_process_is_gone(pid: int) -> None:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.05)
    pytest.fail(f"child process {pid} is still running")


def test_stream_failure_terminates_and_reaps_real_child(tmp_path: Path, monkeypatch) -> None:
    pid_path = tmp_path / "child.pid"
    log_path = tmp_path / "train.log"

    class _FailingLog:
        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def write(self, value: str) -> None:
            raise OSError("simulated log write failure")

        def flush(self) -> None:
            return None

    real_open = Path.open

    def open_log(self, *args, **kwargs):
        if self == log_path:
            return _FailingLog()
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_log)
    with pytest.raises(OSError, match="log write failure"):
        _run_subprocess(
            _sleeping_child_command(pid_path),
            cwd=tmp_path,
            environment={},
            log_path=log_path,
        )
    pid = int(real_open(pid_path, encoding="utf-8").read())
    _assert_process_is_gone(pid)


def test_keyboard_interrupt_terminates_and_reaps_real_child(tmp_path: Path, monkeypatch) -> None:
    pid_path = tmp_path / "child.pid"
    log_path = tmp_path / "train.log"

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr("builtins.print", interrupt)
    with pytest.raises(KeyboardInterrupt):
        _run_subprocess(
            _sleeping_child_command(pid_path),
            cwd=tmp_path,
            environment={},
            log_path=log_path,
        )
    pid = int(pid_path.read_text(encoding="utf-8"))
    _assert_process_is_gone(pid)
