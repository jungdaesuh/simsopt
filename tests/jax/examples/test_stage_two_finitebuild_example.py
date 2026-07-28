"""Source-owned contracts for the finite-build Stage-II JAX mirror."""

from __future__ import annotations

import ast
from pathlib import Path


EXAMPLE = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "jax"
    / "3_Advanced"
    / "stage_two_optimization_finitebuild.py"
)


def test_finitebuild_is_an_exact_name_public_jax_example() -> None:
    module = ast.parse(EXAMPLE.read_text(encoding="utf-8"))
    imported_names = {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    called_names = {
        node.func.id
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    observable_keys = {
        key.value
        for node in ast.walk(module)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }

    assert "make_finite_build_stage_two_objective" in imported_names
    assert "finite_build_stage_two_diagnostics" in imported_names
    assert "serial_solve_jax" in imported_names
    assert "minimize" not in called_names
    assert {
        "squared_flux",
        "length_penalty",
        "distance_penalty",
        "minimum_clearance",
        "coil_lengths",
        "gradient",
        "solver_status",
    } <= observable_keys
