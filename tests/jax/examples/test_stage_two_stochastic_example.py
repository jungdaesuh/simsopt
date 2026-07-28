"""Source-owned contracts for the stochastic Stage-II JAX mirror."""

from __future__ import annotations

import ast
from pathlib import Path


EXAMPLE = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "jax"
    / "2_Intermediate"
    / "stage_two_optimization_stochastic.py"
)


def test_stochastic_stage_two_is_an_exact_name_public_jax_example() -> None:
    module = ast.parse(EXAMPLE.read_text(encoding="utf-8"))
    imported_names = {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    observable_keys = {
        key.value
        for node in ast.walk(module)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    called_names = {
        node.func.id
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "materialize_stochastic_coil_perturbations" in imported_names
    assert "make_stochastic_stage_two_objective" in imported_names
    assert "serial_solve_jax" in imported_names
    assert "minimize" not in called_names
    assert {
        "sample_metadata",
        "out_of_sample_metadata",
        "training_flux_objective",
        "nominal_flux_objective",
        "out_of_sample_objective",
        "solver_status",
    } <= observable_keys
