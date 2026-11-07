from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest


ROOT = Path(__file__).resolve().parents[1]
_NODE_TEST_DECLARATION_PATTERN = re.compile(
    r"\b(?:test|it|describe|suite)\s*\.\s*(?:only|skip|todo)\s*\("
)
_NODE_TEST_CALL_PATTERN = re.compile(r"\b(?:test|it|describe|suite)\s*\(")
_NODE_TEST_OPTION_PROPERTY_PATTERN = re.compile(
    r"""^\s*
        (?P<key>only|skip|todo|__JS_QUOTED_OPTION_(?:ONLY|SKIP|TODO)__)
        \s*:\s*(?P<value>.*?)\s*$
    """,
    re.DOTALL | re.VERBOSE,
)
_JS_DECIMAL_NUMBER_PATTERN = re.compile(
    r"[+-]?(?:\d(?:_?\d)*(?:\.(?:\d(?:_?\d)*)?)?|\.(?:\d(?:_?\d)*))"
    r"(?:[eE][+-]?\d(?:_?\d)*)?\Z"
)


def _mask_javascript_comments_and_literals(source: str) -> str:
    pattern = re.compile(
        r"//[^\r\n]*|/\*.*?\*/|'(?:\\.|[^'\\\r\n])*'|\"(?:\\.|[^\"\\\r\n])*\"|`(?:\\.|[^`\\])*`",
        re.DOTALL,
    )

    def blank(match: re.Match[str]) -> str:
        text = match.group()
        if text.startswith(("'", '"', "`")):
            value = text[1:-1]
            if value in {"only", "skip", "todo"}:
                marker = f"__JS_QUOTED_OPTION_{value.upper()}__"
            elif not value:
                marker = "__JS_EMPTY_STRING__"
            else:
                marker = "__JS_STRING_LITERAL__"
            return marker + "".join(
                "\n" if char == "\n" else " " for char in text
            )
        return "".join("\n" if character == "\n" else " " for character in text)

    return pattern.sub(blank, source)


def _matching_close(character: str) -> str:
    return {"{": "}", "[": "]", "(": ")"}[character]


def _first_options_object(source: str, opening_paren: int) -> int | None:
    """Find a literal second argument without parsing the test callback body."""
    stack = [")"]
    for index in range(opening_paren + 1, len(source)):
        character = source[index]
        if character in "{[(":
            stack.append(_matching_close(character))
        elif character in "}])":
            if not stack or character != stack[-1]:
                continue
            if len(stack) == 1:
                return None
            stack.pop()
        elif character == "," and len(stack) == 1:
            index += 1
            while index < len(source) and source[index].isspace():
                index += 1
            return index if index < len(source) and source[index] == "{" else None
    return None


def _top_level_object_members(source: str, opening_brace: int):
    """Yield direct properties while balancing nested static literal values."""
    stack = ["}"]
    member_start = opening_brace + 1
    for index in range(member_start, len(source)):
        character = source[index]
        if character in "{[(":
            stack.append(_matching_close(character))
        elif character in "}])":
            if not stack or character != stack[-1]:
                continue
            if len(stack) == 1:
                yield member_start, index
                return
            stack.pop()
        elif character == "," and len(stack) == 1:
            yield member_start, index
            member_start = index + 1


def _node_test_option_is_active(value: str) -> bool:
    value = value.strip()
    if value in {"false", "null", "undefined", "__JS_EMPTY_STRING__"}:
        return False
    if _JS_DECIMAL_NUMBER_PATTERN.fullmatch(value):
        return float(value.replace("_", "")) != 0.0
    # Dynamic expressions are conservatively treated as active; this is not
    # an attempt to evaluate arbitrary JavaScript.
    return True


def _find_forbidden_node_test_declarations(root: Path) -> list[str]:
    violations: list[str] = []
    root = Path(root)
    public_roots = [root / "tests", root / "web"]
    search_roots = [path for path in public_roots if path.is_dir()] or [root]
    excluded_directories = {
        ".git", ".venv", "venv", "node_modules", "dist", "build", "bundles",
        "generated", "outputs", "output", "fixtures", "__fixtures__",
    }
    # The public runners execute *.test.js under tests/ and *.test.mjs under
    # web/. Restrict discovery to those suites, not bundled or fixture sources.
    node_test_files = {
        path
        for search_root in search_roots
        for pattern in ("*.test.js", "*.test.mjs")
        for path in search_root.rglob(pattern)
        if not excluded_directories.intersection(path.relative_to(search_root).parts[:-1])
    }
    for path in sorted(node_test_files):
        source = _mask_javascript_comments_and_literals(path.read_text(encoding="utf-8"))
        path_violations = []
        for match in _NODE_TEST_DECLARATION_PATTERN.finditer(source):
            line = source.count("\n", 0, match.start()) + 1
            path_violations.append(
                f"{path}:{line}: focused, skipped, or TODO Node test declaration"
            )
        for call in _NODE_TEST_CALL_PATTERN.finditer(source):
            options_start = _first_options_object(source, call.end() - 1)
            if options_start is None:
                continue
            for member_start, member_end in _top_level_object_members(
                source, options_start
            ):
                member = source[member_start:member_end]
                option = _NODE_TEST_OPTION_PROPERTY_PATTERN.match(member)
                if option is None or not _node_test_option_is_active(
                    option.group("value")
                ):
                    continue
                key_start = member_start + option.start("key")
                line = source.count("\n", 0, key_start) + 1
                path_violations.append(
                    f"{path}:{line}: focused, skipped, or TODO Node test option"
                )
        path_violations.sort(key=lambda item: (int(item.rsplit(":", 2)[1]), item))
        violations.extend(path_violations)
    return violations


def _normalized_node_test_violations(root: Path, violations: list[str]):
    normalized = set()
    detail_kinds = {
        "focused, skipped, or TODO Node test declaration": "declaration",
        "focused, skipped, or TODO Node test option": "option",
    }
    for violation in violations:
        path_text, line_text, detail = violation.rsplit(":", 2)
        kind = detail_kinds.get(detail.strip(), f"unexpected:{detail.strip()}")
        normalized.add(
            (Path(path_text).relative_to(root).as_posix(), int(line_text), kind)
        )
    return normalized


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
    assert "busy_rejections + server_runtime_failures + inference_failures" in api_doc
    assert "requests - client_rejections" in api_doc


def test_node_test_policy_guard_self_verifies_code_and_ignores_text(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.test.js"
    fixture_source = dedent(
        """\
        // test.only('comment') and test.skip('comment')
        const text = "describe.only('text')";
        test.only('focused', () => {});
        it.skip('skipped', () => {});
        describe('skipped', { skip: true }, () => {});
        test.todo('unfinished declaration', () => {});
        test('unfinished option', { todo: 'not implemented' }, () => {});
        test('conditional skip', { skip: process.env.SKIP_TESTS }, () => {});
        test('ordinary falsy options', { only: false, skip: false, todo: false }, () => {});
        test('ordinary nullish options', { only: null, skip: undefined, todo: false }, () => {});
        test('ordinary empty options', { skip: '', todo: '' }, () => {});
        test('ordinary nested metadata', { metadata: { 'skip': true } }, () => {});
        """
    )
    fixture.write_text(fixture_source, encoding="utf-8")

    violations = _find_forbidden_node_test_declarations(tmp_path)

    assert _normalized_node_test_violations(tmp_path, violations) == {
        ("fixture.test.js", 3, "declaration"),
        ("fixture.test.js", 4, "declaration"),
        ("fixture.test.js", 5, "option"),
        ("fixture.test.js", 6, "declaration"),
        ("fixture.test.js", 7, "option"),
        ("fixture.test.js", 8, "option"),
    }


def test_node_test_discovery_covers_web_mjs_and_excludes_non_test_trees(
    tmp_path: Path,
) -> None:
    tests = tmp_path / "tests"
    web = tmp_path / "web"
    tests.mkdir()
    web.mkdir()
    (tests / "api.test.js").write_text(
        dedent(
            """\
        // test.only('comment')
        const text = "test.skip('string')";
        test.todo('unfinished declaration', () => {});
        test('ordinary nested metadata', { metadata: { 'skip': true } }, () => {});
        test('ordinary falsy options', { only: false, skip: false, todo: false }, () => {});
        test('ordinary nullish options', { only: null, skip: undefined, todo: false }, () => {});
        test('ordinary empty options', { skip: '', todo: '' }, () => {});
        test('root option after nested metadata', { metadata: { owner: 'public' }, skip: true }, () => {});
        test('root option before nested metadata', { only: true, metadata: { owner: 'public' } }, () => {});
        test('root option after nested arrays', { metadata: [{ owner: { id: 1 } }], todo: 'unfinished' }, () => {});
        test('quoted skip key', { 'skip': true }, () => {});
        test('quoted todo key', { "todo": 'pending' }, () => {});
        test('quoted only key', { 'only': true }, () => {});
        test('fractional skip value', { skip: 0.5 }, () => {});
        test('negative skip value', { skip: -1 }, () => {});
        test('exponent skip value', { skip: 1e-3 }, () => {});
        test('truthy numeric string', { skip: '0' }, () => {});
        test('truthy false-looking string', { todo: 'false' }, () => {});
        test('ordinary falsey numbers', { skip: 0, todo: -0, only: 0.0 }, () => {});
        test('ordinary zero exponent', { skip: 0e2 }, () => {});
        """
        ),
        encoding="utf-8",
    )
    (web / "ui.test.mjs").write_text(
        dedent(
            """\
        describe.only('focused browser suite', (t) => {
          t.test('skipped child', { skip: 'not ready' }, () => {});
          it('unfinished child', { todo: true }, () => {});
          it('ordinary child', () => {});
        });
        """
        ),
        encoding="utf-8",
    )
    ignored_sources = (
        web / "node_modules/third-party/third-party.test.js",
        web / "dist/bundle.test.mjs",
        tests / "fixtures/sample.test.js",
        web / "generated/generated-output.test.js",
    )
    for path in ignored_sources:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test.skip('generated or fixture source', () => {});", encoding="utf-8")

    violations = _find_forbidden_node_test_declarations(tmp_path)

    expected = {
        ("tests/api.test.js", 3, "declaration"),
        ("tests/api.test.js", 8, "option"),
        ("tests/api.test.js", 9, "option"),
        ("tests/api.test.js", 10, "option"),
        ("tests/api.test.js", 11, "option"),
        ("tests/api.test.js", 12, "option"),
        ("tests/api.test.js", 13, "option"),
        ("tests/api.test.js", 14, "option"),
        ("tests/api.test.js", 15, "option"),
        ("tests/api.test.js", 16, "option"),
        ("tests/api.test.js", 17, "option"),
        ("tests/api.test.js", 18, "option"),
        ("web/ui.test.mjs", 1, "declaration"),
        ("web/ui.test.mjs", 2, "option"),
        ("web/ui.test.mjs", 3, "option"),
    }
    assert _normalized_node_test_violations(tmp_path, violations) == expected


def test_public_node_tests_do_not_use_focused_skipped_or_todo_declarations() -> None:
    violations = _find_forbidden_node_test_declarations(ROOT)
    assert not violations, "\n".join(violations)


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
