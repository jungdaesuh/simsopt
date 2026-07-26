"""Ratchet: forbid redefining thin host-materialization wrappers outside SSOT."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

# Function bodies that still call device_get by design. Prefer host_boundary
# imports; only extend this set when a helper has distinct semantics that
# cannot share the SSOT implementation.
_ALLOWED_LOCAL_HOST_HELPER_DEFS = frozenset(
    {
        # SSOT D2H materializer.
        "src/simsopt_jax/runtime/host_boundary.py::host_array",
        # Tracer-aware validation (returns None under tracing; not pure host_array).
        "src/simsopt_jax/core/pm_optimization.py::_host_scalar_for_validation",
    }
)

_HOST_HELPER_NAMES = frozenset(
    {
        "host_array",
        "host_scalar",
        "host_float",
        "host_float64",
        "host_int",
        "host_bool",
        "host_tree",
        "_host_array",
        "_host_scalar",
        "_host_float",
        "_host_float64",
        "_host_int",
        "_host_pytree",
        "_jax_trace_host_array",
        "_host_scalar_for_validation",
    }
)


def _iter_python_files():
    for path in SRC_ROOT.rglob("*.py"):
        relative = path.relative_to(REPO_ROOT).as_posix()
        if "/__pycache__/" in f"/{relative}/":
            continue
        yield path, relative


def _function_uses_device_get(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and child.attr == "device_get":
            return True
        if isinstance(child, ast.Name) and child.id == "device_get":
            return True
    return False


def _local_host_helpers_reimplementing_device_get():
    found: list[str] = []
    for path, relative in _iter_python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name not in _HOST_HELPER_NAMES:
                continue
            if not _function_uses_device_get(node):
                continue
            key = f"{relative}::{node.name}"
            found.append(key)
    return sorted(found)


def test_no_unapproved_local_host_array_device_get_wrappers():
    actual = set(_local_host_helpers_reimplementing_device_get())
    unexpected = sorted(actual - _ALLOWED_LOCAL_HOST_HELPER_DEFS)
    stale = sorted(_ALLOWED_LOCAL_HOST_HELPER_DEFS - actual)
    message_parts = []
    if unexpected:
        message_parts.append(
            "Unexpected local host materialization helpers that reimplement "
            "device_get (import host_boundary instead):\n  " + "\n  ".join(unexpected)
        )
    if stale:
        message_parts.append(
            "Stale allowlist entries (remove from test allowlist):\n  "
            + "\n  ".join(stale)
        )
    assert not message_parts, "\n\n".join(message_parts)


def test_host_boundary_exports_ready_variants():
    from simsopt_jax.runtime import host_boundary

    assert callable(host_boundary.host_array)
    assert callable(host_boundary.host_array_after_ready)
    assert callable(host_boundary.host_float_after_ready)
    assert callable(host_boundary.host_tree)


def test_minimize_runtime_reexports_ready_host_helpers():
    """Solver packaging keeps ready semantics via host_boundary aliases."""
    source = (REPO_ROOT / "src/simsopt_jax/solve/minimize_runtime.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    aliases = {
        alias.name: alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "simsopt_jax.runtime.host_boundary"
        for alias in node.names
    }
    assert aliases.get("host_array_after_ready") == "host_array"
    assert aliases.get("host_float_after_ready") == "host_float"
