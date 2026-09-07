from __future__ import annotations

import ast
import os
from pathlib import Path
import random

import pytest
from _pytest.mark import Expression, MarkMatcher


def pytest_addoption(parser) -> None:
    group = parser.getgroup("private BCS model")
    group.addoption(
        "--private-model-archive",
        action="store",
        default=None,
        help="External private BCS ZIP used only by explicitly selected private tests.",
    )
    group.addoption(
        "--private-model-install",
        action="store_true",
        default=False,
        help="Allow explicitly selected tests to inspect the fixed private install destination.",
    )
    group.addoption(
        "--private-source-checkpoint",
        action="store",
        default=None,
        help="Ignored source checkpoint required only by explicitly selected producer tests.",
    )


def _marker_is_positive_in_expression(markexpr: str, marker_name: str) -> bool:
    try:
        tree = ast.parse(markexpr, mode="eval")
    except SyntaxError:
        return False

    def contains_positive_marker(node: ast.AST, negated: bool = False) -> bool:
        if isinstance(node, ast.Name) and node.id == marker_name:
            return not negated
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return contains_positive_marker(node.operand, not negated)
        return any(contains_positive_marker(child, negated) for child in ast.iter_child_nodes(node))

    return contains_positive_marker(tree)


def _marker_expression_selects_item(config, item, marker_name: str) -> bool:
    markexpr = config.option.markexpr or ""
    if not markexpr or not _marker_is_positive_in_expression(markexpr, marker_name):
        return False
    expression = Expression.compile(markexpr)
    return expression.evaluate(MarkMatcher.from_markers(item.iter_markers()))


def _option_or_environment(config, option: str, environment: str) -> str | None:
    return config.getoption(option) or os.environ.get(environment)


@pytest.fixture(scope="session")
def private_model_archive(pytestconfig) -> Path:
    raw_archive = _option_or_environment(
        pytestconfig, "--private-model-archive", "VACCA_BCS_PRIVATE_ARCHIVE"
    )
    if not raw_archive or not Path(raw_archive).is_file():
        raise pytest.UsageError(
            "-m private_model requires --private-model-archive PATH or "
            "VACCA_BCS_PRIVATE_ARCHIVE pointing to a readable external ZIP"
        )
    archive = Path(raw_archive)
    if not Path(f"{archive}.sha256").is_file():
        raise pytest.UsageError("private archive requires its adjacent .sha256 sidecar")
    return archive


@pytest.fixture(scope="session")
def private_source_checkpoint(pytestconfig) -> Path:
    raw_checkpoint = _option_or_environment(
        pytestconfig, "--private-source-checkpoint", "VACCA_BCS_PRIVATE_CHECKPOINT"
    )
    if not raw_checkpoint or not Path(raw_checkpoint).is_file():
        raise pytest.UsageError(
            "-m private_source_model requires --private-source-checkpoint PATH "
            "pointing to a readable ignored source checkpoint"
        )
    return Path(raw_checkpoint)


def pytest_collection_modifyitems(config, items) -> None:
    for item in items:
        if "private_model" in item.keywords:
            if not _marker_expression_selects_item(config, item, "private_model"):
                item.add_marker(
                    pytest.mark.skip(
                        reason="private consumer tests are opt-in; select -m private_model"
                    )
                )
            elif "private_model_installed" in item.keywords and not config.getoption(
                "--private-model-install"
            ):
                item.add_marker(
                    pytest.mark.skip(
                        reason="test requires --private-model-install after external installation"
                    )
                )
        if "private_source_model" in item.keywords and not _marker_expression_selects_item(
            config, item, "private_source_model"
        ):
            item.add_marker(
                pytest.mark.skip(
                    reason="private producer tests are opt-in; select -m private_source_model"
                )
            )
    if os.environ.get("VACCA_RANDOMIZE_TEST_ORDER") == "1":
        random.Random(20260907).shuffle(items)
