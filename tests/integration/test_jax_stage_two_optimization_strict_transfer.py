"""Device-boundary contracts for the standard Stage-II mirror."""

from __future__ import annotations

import ast
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = REPOSITORY_ROOT / "examples/jax/2_Intermediate/stage_two_optimization.py"
WORKFLOW_PATH = REPOSITORY_ROOT / "src/simsopt_jax/examples/stage_two_standard.py"


def test_standard_stage_two_uses_public_workflow_and_one_host_publication() -> None:
    source = EXAMPLE_PATH.read_text()

    assert "solve_standard_stage_two" in source
    assert "make_stage_two_objective" not in source
    assert "serial_solve_jax" not in source
    assert source.count("jax.device_get(") == 1
    assert "first_length_weight_device = jax.device_put(" in source
    assert "second_length_weight_device = jax.device_put(" in source
    assert "first_length_weight=first_length_weight_device" in source
    assert "second_length_weight=second_length_weight_device" in source


def test_stage_two_lbfgs_history_is_bounded_independently_of_iterations() -> None:
    module = ast.parse(WORKFLOW_PATH.read_text(encoding="utf-8"))
    history_sizes = {
        node.targets[0].id: node.value.value
        for node in module.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, int)
        and node.targets[0].id == "_STAGE_TWO_LBFGS_HISTORY_SIZE"
    }
    lbfgs_history_arguments = [
        keyword.value.id
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "serial_solve_jax"
        for keyword in node.keywords
        if keyword.arg == "maxcor" and isinstance(keyword.value, ast.Name)
    ]

    assert history_sizes == {"_STAGE_TWO_LBFGS_HISTORY_SIZE": 10}
    assert lbfgs_history_arguments == [
        "_STAGE_TWO_LBFGS_HISTORY_SIZE",
        "_STAGE_TWO_LBFGS_HISTORY_SIZE",
    ]
