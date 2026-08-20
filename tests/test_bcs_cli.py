from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import scripts.build_bcs_cls as cli
from vacca_bcs.dataset_transaction import DatasetInstallError


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_parser_defaults_and_options() -> None:
    defaults = cli.build_parser().parse_args([])
    assert defaults.max_per_class == 6000
    assert defaults.seed == 42
    assert defaults.val_ratio == 0.2
    options = cli.build_parser().parse_args(
        ["--max-per-class", "3", "--seed", "7", "--val-ratio", "0", "--out-dir", "out"]
    )
    assert (options.max_per_class, options.seed, options.val_ratio, options.out_dir) == (3, 7, 0.0, "out")


def test_help_exits_successfully(capsys) -> None:
    with pytest.raises(SystemExit) as failure:
        cli.build_parser().parse_args(["--help"])
    assert failure.value.code == 0
    assert "Build the ordinal BCS folder dataset" in capsys.readouterr().out


def test_main_success_formats_result_at_cli_boundary(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr(
        cli,
        "build_dataset",
        lambda *args, **kwargs: (
            {"3.25": {"selected": 1, "train": 1, "val": 0, "staged": 1, "added": 1, "updated": 0, "unchanged": 0, "stale": 0}},
            {"selected": 1, "train": 1, "val": 0, "staged": 1, "added": 1, "updated": 0, "unchanged": 0, "stale": 0},
        ),
    )
    assert cli.main(["--bcs-dir", str(tmp_path / "source"), "--out-dir", str(tmp_path / "out")]) == 0
    output = capsys.readouterr().out
    assert "[READY]" in output and "Total selected: 1" in output


def test_main_returns_failure_for_install_errors(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "build_dataset", lambda *args, **kwargs: (_ for _ in ()).throw(DatasetInstallError("failed")))
    assert cli.main([]) == 1
    assert "[ERROR] failed" in capsys.readouterr().out


def test_cli_import_boundary_excludes_training_modules() -> None:
    probe = """
import sys
import scripts.build_bcs_cls
from vacca_bcs.constants import CLASS_NAMES
assert CLASS_NAMES == ["3.25", "3.5", "3.75", "4.0", "4.25"]
for name in ("torch", "torchvision", "vacca_bcs.model", "vacca_bcs.dataset"):
    assert name not in sys.modules, name
"""
    result = subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
