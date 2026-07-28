"""Source-owned contract for the VMEC-free Boozer single-stage pair."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
NATIVE = (
    ROOT
    / "examples"
    / "3_Advanced"
    / "single_stage_boozer_vacuum_optimization.py"
)
JAX = (
    ROOT
    / "examples"
    / "jax"
    / "3_Advanced"
    / "single_stage_boozer_vacuum_optimization.py"
)


def _module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def test_native_and_jax_vacuum_single_stage_have_distinct_exact_paths() -> None:
    assert NATIVE.is_file()
    assert JAX.is_file()


def test_jax_vacuum_single_stage_uses_decomposed_public_jax_kernels() -> None:
    module = _module(JAX)
    imported_names = {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    imported_modules = {
        node.module
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert "make_traceable_objective_session" in imported_names
    assert "minimize_lbfgs_host_core" in imported_names
    assert not any("vmec" in name.lower() for name in imported_modules)
    assert not any("scipy" in name.lower() for name in imported_modules)


def test_both_single_stage_examples_report_implicit_physics_state() -> None:
    required = {
        "inner_solver_success",
        "iota",
        "volume",
        "non_qs_ratio",
        "boozer_residual",
        "gradient",
        "solution",
    }
    for path in (NATIVE, JAX):
        module = _module(path)
        observable_keys = {
            key.value
            for node in ast.walk(module)
            if isinstance(node, ast.Dict)
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        assert required <= observable_keys
