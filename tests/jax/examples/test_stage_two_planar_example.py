"""Source-owned contracts for the planar Stage-II JAX mirror."""

from __future__ import annotations

import ast
from pathlib import Path

EXAMPLE = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "jax"
    / "2_Intermediate"
    / "stage_two_optimization_planar_coils.py"
)


def test_planar_stage_two_publishes_physical_geometry_coordinates() -> None:
    module = ast.parse(EXAMPLE.read_text(encoding="utf-8"))
    observable_keys = {
        key.value
        for node in ast.walk(module)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }

    assert "canonical_geometry" in observable_keys


def test_planar_stage_two_reuses_one_parametric_device_workflow() -> None:
    source = EXAMPLE.read_text(encoding="utf-8")

    assert "solve_standard_stage_two" in source
    assert source.count("jax.device_get(") == 1
    assert "TraceableScalarProblem" not in source
    assert "make_stage_two_objective" not in source
