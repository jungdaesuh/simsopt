from __future__ import annotations

import ast
from pathlib import Path


_TEST_ROOT = Path(__file__).resolve().parent
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
    return (
        len(call.args) > positional_index
        and _is_non_empty_message(call.args[positional_index])
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
                f"{relative_path}:{node.lineno}: "
                f"{call_name} needs a non-empty reason"
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
