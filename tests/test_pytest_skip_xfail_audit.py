"""AST audit for pytest skip/xfail/subprocess-skip sentinels.

This file checks **static, mechanically detectable** properties — every
`pytest.mark.skip`/`skipif`/`xfail` and every subprocess-case `_skip_case(...)`
sentinel must carry a non-empty reason, and tests must not silently swallow
subprocess skips.

The complementary **human-review** rule for test quality — "what's the
independent oracle for this assertion?" — lives at
`tests/REVIEWER_ORACLE_LINT.md`. Oracle quality is not statically detectable
in general (a re-export `is`-identity check vs. an analytic comparison can
look identical at the AST level), so it is enforced by reviewer checklist
during code review of new `test_*_jax_*.py` files. See AI-3 in
`.artifacts/jax-test-audit-2026-05-13/TEST_QUALITY_TODOS.md`.
"""

from __future__ import annotations

import ast
from pathlib import Path


_TEST_ROOT = Path(__file__).resolve().parent
_SUBPROCESS_CASES_DIR = _TEST_ROOT / "subprocess"
_SKIP_SENTINEL_CALL_NAME = "_skip_case"
_SUBPROCESS_CASE_NAME_PREFIX = "_run_"
_SUBPROCESS_CASE_NAME_SUFFIX = "_case"
_VALUE_DISPATCH_CASE_FUNCTIONS = frozenset(
    {
        "_run_legacy_curve_objective_value_case",
        "_run_legacy_curve_objective_gradient_case",
    }
)
_SKIP_XFAIL_MARKERS = {"skip", "skipif", "xfail"}
_SKIP_XFAIL_RUNTIME_CALLS = {"skip", "xfail"}
_MARKER_REASON_POSITION = {
    "skip": 0,
    "skipif": 1,
    "xfail": 1,
}


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        if parent is None:
            return None
        return f"{parent}.{node.attr}"
    return None


def _is_non_empty_message(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return bool(node.value.strip())
    if isinstance(node, ast.Constant):
        return False
    return True


def _keyword_message(call: ast.Call) -> bool:
    return any(
        keyword.arg == "reason" and _is_non_empty_message(keyword.value)
        for keyword in call.keywords
    )


def _has_message(call: ast.Call, *, positional_index: int) -> bool:
    if _keyword_message(call):
        return True
    return len(call.args) > positional_index and _is_non_empty_message(
        call.args[positional_index]
    )


def _keyword_bool(call: ast.Call, name: str) -> bool:
    for keyword in call.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            return keyword.value.value is True
    return False


def _iter_test_python_files() -> list[Path]:
    return sorted(path for path in _TEST_ROOT.rglob("*.py") if path.is_file())


def _audit_tree(tree: ast.AST, relative_path: Path) -> list[str]:
    failures: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call):
                    continue
                marker_call_name = _dotted_name(decorator)
                if marker_call_name in {
                    f"pytest.mark.{name}" for name in _SKIP_XFAIL_MARKERS
                }:
                    failures.append(
                        f"{relative_path}:{decorator.lineno}: "
                        f"{marker_call_name} needs a non-empty reason"
                    )
            continue

        if not isinstance(node, ast.Call):
            continue
        call_name = _dotted_name(node.func)
        if call_name in {f"pytest.{name}" for name in _SKIP_XFAIL_RUNTIME_CALLS}:
            if not _has_message(node, positional_index=0):
                failures.append(
                    f"{relative_path}:{node.lineno}: "
                    f"{call_name} needs a non-empty reason"
                )
            continue
        if call_name not in {f"pytest.mark.{name}" for name in _SKIP_XFAIL_MARKERS}:
            continue
        marker_name = call_name.rsplit(".", maxsplit=1)[-1]
        if not _has_message(
            node,
            positional_index=_MARKER_REASON_POSITION[marker_name],
        ):
            failures.append(
                f"{relative_path}:{node.lineno}: {call_name} needs a non-empty reason"
            )
        if marker_name == "xfail" and not _keyword_bool(node, "strict"):
            failures.append(
                f"{relative_path}:{node.lineno}: pytest.mark.xfail needs strict=True"
            )
    return failures


def _audit_source(source: str) -> list[str]:
    return _audit_tree(ast.parse(source), Path("tests/example.py"))


def test_pytest_skip_xfail_audit_rejects_implicit_or_empty_reasons():
    failures = _audit_source(
        """
import pytest

@pytest.mark.skipif(True)
def test_skipif_without_reason():
    pass

@pytest.mark.xfail(True, strict=True)
def test_xfail_without_reason():
    pass

@pytest.mark.skip(reason="")
def test_skip_empty_reason():
    pass

@pytest.mark.xfail(reason="tracked issue")
def test_xfail_without_strict():
    pass

def test_runtime_skip_empty_reason():
    pytest.skip("")
"""
    )

    assert failures == [
        "tests/example.py:4: pytest.mark.skipif needs a non-empty reason",
        "tests/example.py:8: pytest.mark.xfail needs a non-empty reason",
        "tests/example.py:12: pytest.mark.skip needs a non-empty reason",
        "tests/example.py:16: pytest.mark.xfail needs strict=True",
        "tests/example.py:21: pytest.skip needs a non-empty reason",
    ]


def test_pytest_skip_xfail_audit_accepts_explicit_reasons():
    assert (
        _audit_source(
            """
import pytest

@pytest.mark.skip("tracked issue")
def test_skip_with_reason():
    pass

@pytest.mark.skipif(True, reason="tracked issue")
def test_skipif_with_reason():
    pass

@pytest.mark.xfail(True, reason="tracked issue", strict=True)
def test_xfail_with_reason_and_strict():
    pass

def test_runtime_skip_with_reason():
    pytest.skip("tracked issue")
"""
        )
        == []
    )


def test_pytest_skip_xfail_contracts_are_explicit_and_ci_audited():
    failures: list[str] = []
    for path in _iter_test_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        failures.extend(_audit_tree(tree, path.relative_to(_TEST_ROOT.parent)))

    assert failures == []


def _iter_subprocess_case_files() -> list[Path]:
    if not _SUBPROCESS_CASES_DIR.is_dir():
        return []
    return sorted(
        path for path in _SUBPROCESS_CASES_DIR.glob("*_cases.py") if path.is_file()
    )


def _is_skip_sentinel_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == _SKIP_SENTINEL_CALL_NAME
    )


def _branch_calls_skip_sentinel(statements: list[ast.stmt]) -> bool:
    return any(
        isinstance(statement, ast.Expr) and _is_skip_sentinel_call(statement.value)
        for statement in statements
    )


def _audit_subprocess_case_function(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    relative_path: Path,
) -> list[str]:
    failures: list[str] = []
    if func.name in _VALUE_DISPATCH_CASE_FUNCTIONS:
        return failures
    for sub in ast.walk(func):
        if not isinstance(sub, ast.If):
            continue
        body = sub.body
        if not body or not isinstance(body[-1], ast.Return):
            continue
        if body[-1].value is not None or _branch_calls_skip_sentinel(body[:-1]):
            continue
        failures.append(
            f"{relative_path}:{sub.lineno}: silent return in {func.name} — "
            f"must call {_SKIP_SENTINEL_CALL_NAME}(reason) before returning"
        )
        if body[:-1]:
            failures.append(
                f"{relative_path}:{body[-1].lineno}: silent return in {func.name} — "
                "logging or other side effects are not a skip sentinel"
            )
    return failures


def _audit_subprocess_case_tree(tree: ast.AST, relative_path: Path) -> list[str]:
    failures: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not (
            node.name.startswith(_SUBPROCESS_CASE_NAME_PREFIX)
            and node.name.endswith(_SUBPROCESS_CASE_NAME_SUFFIX)
        ):
            continue
        failures.extend(_audit_subprocess_case_function(node, relative_path))
    return failures


def _audit_subprocess_case_source(source: str) -> list[str]:
    return _audit_subprocess_case_tree(
        ast.parse(source),
        Path("tests/subprocess/example_cases.py"),
    )


def test_subprocess_case_audit_rejects_silent_returns():
    failures = _audit_subprocess_case_source(
        """
def _run_silent_case():
    if not _configure_backend():
        return
    do_work()


def _run_silent_gpu_case():
    gpu = _configure_strict_gpu_fast_backend()
    if gpu is None:
        return
    do_work()


def _run_logged_silent_case():
    if not _configure_backend():
        print("backend unavailable")
        return
    do_work()
"""
    )

    assert failures == [
        "tests/subprocess/example_cases.py:3: silent return in _run_silent_case — "
        "must call _skip_case(reason) before returning",
        "tests/subprocess/example_cases.py:10: silent return in _run_silent_gpu_case — "
        "must call _skip_case(reason) before returning",
        "tests/subprocess/example_cases.py:16: silent return in _run_logged_silent_case — "
        "must call _skip_case(reason) before returning",
        "tests/subprocess/example_cases.py:18: silent return in _run_logged_silent_case — "
        "logging or other side effects are not a skip sentinel",
    ]


def test_subprocess_case_audit_accepts_skip_sentinel_emissions():
    assert (
        _audit_subprocess_case_source(
            """
def _run_explicit_skip_case():
    if not _configure_backend():
        _skip_case("backend unavailable")
        return
    do_work()


def _run_explicit_skip_gpu_case():
    gpu = _configure_strict_gpu_fast_backend()
    if gpu is None:
        _skip_case("no gpu device detected")
        return
    do_work()
"""
        )
        == []
    )


def test_subprocess_case_audit_covers_all_repo_subprocess_case_files():
    failures: list[str] = []
    case_files = _iter_subprocess_case_files()
    assert case_files, (
        f"audit must find at least one subprocess case file under {_SUBPROCESS_CASES_DIR}"
    )
    for path in case_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        failures.extend(
            _audit_subprocess_case_tree(tree, path.relative_to(_TEST_ROOT.parent))
        )

    assert failures == []
