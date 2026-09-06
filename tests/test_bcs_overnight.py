from __future__ import annotations

import io
import builtins
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from scripts.run_bcs_overnight import (
    CONFIG_RELATIVE,
    LOCAL_ROOT_RELATIVE,
    OUTPUT_RELATIVE,
    SNAPSHOT_RELATIVE,
    _run_subprocess,
    main,
)


def _repo(tmp_path: Path, ready=False) -> Path:
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    (tmp_path / "scripts/build_bcs_category.py").touch()
    (tmp_path / "scripts/train_bcs_ordinal.py").touch()
    (tmp_path / CONFIG_RELATIVE).write_text(
        "data_root: data/bcs-category-v1\noutput_dir: outputs/bcs-category-coral-v1\n",
        encoding="utf-8",
    )
    (tmp_path / LOCAL_ROOT_RELATIVE).mkdir(parents=True)
    if ready:
        (tmp_path / SNAPSHOT_RELATIVE).mkdir(parents=True)
        (tmp_path / SNAPSHOT_RELATIVE / "manifest.json").touch()
        (tmp_path / OUTPUT_RELATIVE / "weights").mkdir(parents=True)
        (tmp_path / OUTPUT_RELATIVE / "weights/last.pt").touch()
        (tmp_path / OUTPUT_RELATIVE / "weights/checkpoint_set.json").touch()
        (tmp_path / OUTPUT_RELATIVE / "results_lineage.json").touch()
    return tmp_path


def _runner(calls, *, fail=None, make_best=True):
    def fake(command, *, cwd, environment, log_path):
        calls.append((list(command), cwd, log_path, dict(environment)))
        log_path.touch()
        script = next(part for part in command if part.endswith(".py"))
        if fail and script.endswith(fail):
            return 2
        if make_best and script.endswith("train_bcs_ordinal.py"):
            (cwd / OUTPUT_RELATIVE / "weights").mkdir(parents=True, exist_ok=True)
            (cwd / OUTPUT_RELATIVE / "weights/best.pt").touch()
        return 0
    return fake


def _accept_best(*args):
    return {"checkpoint": "best.pt", "category_contract": "1..5"}


def test_local_preflight_and_sequence_use_new_paths(tmp_path, capsys):
    root = _repo(tmp_path)
    calls = []
    assert main(["--preflight-only"], root=root, environment={}, runner=_runner(calls)) == 0
    assert calls == []
    assert main([], root=root, environment={}, runner=_runner(calls), best_validator=_accept_best) == 0
    build, train = calls
    assert build[0][2:] == ["--local-root", str(root / LOCAL_ROOT_RELATIVE), "--output", str(root / SNAPSHOT_RELATIVE)]
    assert train[0][1:3] == ["-u", str(root / "scripts/train_bcs_ordinal.py")]
    assert train[0][3:] == ["--config", str(root / CONFIG_RELATIVE)]
    assert "SECRET" not in build[2].read_text(encoding="utf-8")
    assert "SECRET" not in capsys.readouterr().out


def test_skip_build_resume_requires_new_snapshot_and_last_checkpoint(tmp_path):
    root = _repo(tmp_path, ready=True)
    calls = []
    assert main(["--skip-build", "--resume"], root=root, runner=_runner(calls), best_validator=_accept_best) == 0
    assert calls[0][0][-2:] == ["--resume", str(root / OUTPUT_RELATIVE / "weights/last.pt")]


def test_overnight_preflight_rejects_symlinked_workflow_paths(tmp_path):
    root = _repo(tmp_path)
    link = root / "local-link"
    try:
        link.symlink_to(root / LOCAL_ROOT_RELATIVE, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable")
    error = io.StringIO()
    assert main(["--local-root", str(link)], root=root, stderr=error) == 1
    assert "symlink" in error.getvalue()


def test_overnight_preflight_rejects_crossed_config_and_data_capabilities(tmp_path):
    root = _repo(tmp_path)
    error = io.StringIO()
    assert main(
        ["--config", str(root / "data" / "config.yaml")],
        root=root,
        stderr=error,
    ) == 1
    assert "approved" in error.getvalue() or "unsafe" in error.getvalue()
    assert not (root / "data" / "config.yaml").exists()


def test_failures_stop_and_missing_best_are_nonzero(tmp_path):
    root = _repo(tmp_path)
    calls = []
    assert main([], root=root, runner=_runner(calls, fail="build_bcs_category.py"), stderr=io.StringIO()) == 1
    assert len(calls) == 1
    root = _repo(tmp_path / "train")
    calls = []
    assert main([], root=root, runner=_runner(calls, fail="train_bcs_ordinal.py"), stderr=io.StringIO()) == 1
    assert len(calls) == 2
    root = _repo(tmp_path / "best")
    assert main([], root=root, runner=_runner([], make_best=False), stderr=io.StringIO()) == 1


def test_real_unbuffered_child_progress_reaches_tee_log_before_exit(tmp_path, capsys):
    alive = tmp_path / "child-alive"
    log_path = tmp_path / "train.log"
    code = "import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); p.write_text('1'); print('[TRAIN 1/1 1/1]',flush=True); time.sleep(1); p.unlink()"
    status = []
    thread = threading.Thread(target=lambda: status.append(_run_subprocess([sys.executable, "-u", "-c", code, str(alive)], cwd=tmp_path, environment={}, log_path=log_path)))
    thread.start()
    deadline = time.monotonic() + 5
    observed = False
    while time.monotonic() < deadline and thread.is_alive():
        output = capsys.readouterr().out
        if alive.is_file() and "[TRAIN 1/1 1/1]" in log_path.read_text(encoding="utf-8") and "[TRAIN 1/1 1/1]" in output:
            observed = True
            break
        time.sleep(0.01)
    thread.join(timeout=5)
    assert observed and status == [0]


def _pid_is_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, OSError):
        return True
    return False


def _child_command(pid_path: Path) -> list[str]:
    code = (
        "import os,pathlib,time; "
        f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); "
        "print('CHILD_READY', flush=True); time.sleep(30)"
    )
    return [sys.executable, "-u", "-c", code]


def test_real_log_write_failure_terminates_and_reaps_immediate_child(tmp_path, monkeypatch):
    log_path = tmp_path / "train.log"
    pid_path = tmp_path / "child.pid"

    class FailingLog:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def write(self, _value):
            raise OSError("simulated log write failure")

        def flush(self):
            pass

    real_open = Path.open

    def open_path(path, *args, **kwargs):
        if path == log_path:
            return FailingLog()
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_path)
    with pytest.raises(OSError, match="log write"):
        _run_subprocess(_child_command(pid_path), cwd=tmp_path, environment={}, log_path=log_path)
    pid = int(pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not _pid_is_gone(pid):
        time.sleep(0.01)
    assert _pid_is_gone(pid)


def test_real_keyboard_interrupt_terminates_and_reaps_immediate_child(tmp_path, monkeypatch):
    log_path = tmp_path / "train.log"
    pid_path = tmp_path / "child.pid"
    real_print = builtins.print

    def interrupting_print(*args, **kwargs):
        if args and args[0] == "CHILD_READY\n":
            raise KeyboardInterrupt()
        return real_print(*args, **kwargs)

    monkeypatch.setattr(builtins, "print", interrupting_print)
    with pytest.raises(KeyboardInterrupt):
        _run_subprocess(_child_command(pid_path), cwd=tmp_path, environment={}, log_path=log_path)
    pid = int(pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not _pid_is_gone(pid):
        time.sleep(0.01)
    assert _pid_is_gone(pid)
