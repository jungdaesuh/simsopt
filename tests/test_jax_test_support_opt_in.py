"""Pin which test modules opt into the JAX test runtime of ``jax_test_support``.

The root conftest applies no JAX runtime, so upstream's native tests run as on
master. A JAX-dependent module takes the XLA pins, x64 and the per-test backend
guard from its first import instead; these static checks keep that opt-in
complete (no JAX module without it) and exclusive (no native module with it).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# Parsing upstream's test sources reports their invalid escape sequences.
pytestmark = pytest.mark.filterwarnings(
    "ignore:invalid escape sequence:DeprecationWarning"
)

_TESTS_ROOT = Path(__file__).resolve().parent
_JAX_RUNTIME_PACKAGES = frozenset({"jax", "simsopt_jax", "simsopt_jax_adapters"})
_SUPPORT_FIXTURES = frozenset({"parity_lane", "assert_two_rank_replay_matches"})
_GUARD_IMPORT = "fixture_jax_runtime_guard"
_GUARD_FIXTURE = "jax_runtime_guard"


def _test_modules() -> dict[str, ast.Module]:
    return {
        path.relative_to(_TESTS_ROOT).as_posix(): ast.parse(path.read_text())
        for path in sorted(_TESTS_ROOT.rglob("test_*.py"))
        if "test_files" not in path.relative_to(_TESTS_ROOT).parts
    }


def _imported_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        return [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
    return []


def _requested_fixtures(node: ast.AST) -> set[str]:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return set()
    args = node.args
    return {arg.arg for arg in args.posonlyargs + args.args + args.kwonlyargs}


def _is_jax_dependent(tree: ast.Module) -> bool:
    """Imports JAX, the port or ``examples.jax`` anywhere, or requests a support fixture."""
    return any(
        any(
            name.partition(".")[0] in _JAX_RUNTIME_PACKAGES
            or name == "examples.jax"
            or name.startswith("examples.jax.")
            for name in _imported_names(node)
        )
        or _requested_fixtures(node) & _SUPPORT_FIXTURES
        for node in ast.walk(tree)
    )


def _first_import_takes_guard(tree: ast.Module) -> bool:
    for stmt in tree.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
            continue
        return (
            isinstance(stmt, ast.ImportFrom)
            and stmt.module == "jax_test_support"
            and any(alias.name == _GUARD_IMPORT for alias in stmt.names)
        )
    return False


def _is_autouse_fixture(node: ast.AST) -> bool:
    return isinstance(node, ast.FunctionDef) and any(
        isinstance(decorator, ast.Call)
        and any(
            keyword.arg == "autouse"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in decorator.keywords
        )
        for decorator in node.decorator_list
    )


def test_jax_dependent_modules_and_only_they_take_the_runtime_first():
    mismatched = sorted(
        relpath
        for relpath, tree in _test_modules().items()
        if _first_import_takes_guard(tree) != _is_jax_dependent(tree)
    )
    assert not mismatched, (
        f"first import must be `from jax_test_support import {_GUARD_IMPORT}` "
        f"exactly in the JAX-dependent test modules: {mismatched}"
    )


def test_module_autouse_fixtures_run_inside_the_guard():
    """pytest orders a module's autouse fixtures by name, not by import order."""
    unguarded = sorted(
        f"{relpath}::{node.name}"
        for relpath, tree in _test_modules().items()
        if _first_import_takes_guard(tree)
        for node in tree.body
        if _is_autouse_fixture(node) and _GUARD_FIXTURE not in _requested_fixtures(node)
    )
    assert not unguarded, f"autouse fixtures must request {_GUARD_FIXTURE}: {unguarded}"
