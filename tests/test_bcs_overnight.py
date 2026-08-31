from __future__ import annotations
import io
from pathlib import Path
from scripts.run_bcs_overnight import main


def _repo(tmp_path: Path, *, output: bool = False, snapshot: bool = False) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "configs").mkdir()
    for path in ("scripts/build_bcs_integer.py", "scripts/train_bcs_ordinal.py"):
        (tmp_path / path).touch()
    (tmp_path / "configs/training_bcs_ordinal.yaml").write_text(
        "data_dir: data/bcs-integer-v1\noutput: outputs/bcs-ordinal-integer-v1\n",
        encoding="utf-8",
    )
    if snapshot:
        (tmp_path / "data/bcs-integer-v1").mkdir(parents=True)
        (tmp_path / "data/bcs-integer-v1/manifest.json").touch()
    if output:
        (tmp_path / "outputs/bcs-ordinal-integer-v1/weights").mkdir(parents=True)
        for path in ("results_lineage.json", "weights/last.pt"):
            (tmp_path / "outputs/bcs-ordinal-integer-v1" / path).touch()
    return tmp_path


def _runner(calls, *, make_best=True, failure=None):
    def fake(command, *, cwd, environment, log_path):
        calls.append((list(command), cwd, dict(environment), log_path))
        log_path.write_text("safe progress\n" + environment.get("VACCA_BACKEND_TOKEN", ""), encoding="utf-8")
        if failure and command[1].endswith(failure):
            return 2
        if make_best and command[1].endswith("train_bcs_ordinal.py"):
            best = cwd / "outputs/bcs-ordinal-integer-v1/weights/best.pt"
            best.parent.mkdir(parents=True, exist_ok=True)
            best.touch()
        return 0
    return fake


def test_missing_env_and_preflight_only_do_not_run(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    calls = []
    cases = (({}, "VACCA_BACKEND_URL"), ({"VACCA_BACKEND_URL": "x"}, "VACCA_BACKEND_TOKEN"))
    for environment, missing in cases:
        error = io.StringIO()
        assert main(["--preflight-only"], root=root, environment=environment, runner=_runner(calls), stderr=error) == 1
        assert not calls and missing in error.getvalue()
    assert main(["--preflight-only"], root=root, environment={"VACCA_BACKEND_URL": "x", "VACCA_BACKEND_TOKEN": "t"}, runner=_runner(calls)) == 0
    assert not calls


def test_fresh_sequence_skip_build_and_logs(tmp_path: Path) -> None:
    root = _repo(tmp_path, snapshot=True)
    calls = []
    environment = {"VACCA_BACKEND_URL": "x", "VACCA_BACKEND_TOKEN": "secret"}
    assert main(["--skip-build"], root=root, environment=environment, runner=_runner(calls)) == 0
    assert len(calls) == 1 and calls[0][0][1].endswith("train_bcs_ordinal.py")
    assert calls[0][3].is_file() and "secret" not in calls[0][3].read_text(encoding="utf-8")
    root = _repo(tmp_path / "fresh")
    calls = []
    assert main([], root=root, environment=environment, runner=_runner(calls)) == 0
    assert [Path(call[0][1]).name for call in calls] == ["build_bcs_integer.py", "train_bcs_ordinal.py"]


def test_resume_uses_supported_args_and_requires_lineage(tmp_path: Path) -> None:
    root = _repo(tmp_path, snapshot=True, output=True)
    calls = []
    assert main(["--skip-build", "--resume"], root=root, environment={}, runner=_runner(calls)) == 0
    command = calls[0][0]
    assert command[-2:] == ["--resume", str(root / "outputs/bcs-ordinal-integer-v1/weights/last.pt")]
    (root / "outputs/bcs-ordinal-integer-v1/results_lineage.json").unlink()
    assert main(["--skip-build", "--resume"], root=root, environment={}, runner=_runner(calls), stderr=io.StringIO()) == 1


def test_failures_stop_or_require_best_checkpoint(tmp_path: Path) -> None:
    env = {"VACCA_BACKEND_URL": "x", "VACCA_BACKEND_TOKEN": "secret"}
    root = _repo(tmp_path / "build")
    calls = []
    assert main([], root=root, environment=env, runner=_runner(calls, failure="build_bcs_integer.py"), stderr=io.StringIO()) == 1
    assert len(calls) == 1
    root = _repo(tmp_path / "train")
    calls = []
    assert main([], root=root, environment=env, runner=_runner(calls, failure="train_bcs_ordinal.py"), stderr=io.StringIO()) == 1
    assert len(calls) == 2
    root = _repo(tmp_path / "best")
    assert main([], root=root, environment=env, runner=_runner([], make_best=False), stderr=io.StringIO()) == 1
    root = _repo(tmp_path / "stale", snapshot=True, output=True)
    error = io.StringIO()
    assert main([], root=root, environment=env, stderr=error) == 1
    assert "snapshot" in error.getvalue()
    assert main(["--skip-build"], root=root, environment={}, stderr=io.StringIO()) == 1
