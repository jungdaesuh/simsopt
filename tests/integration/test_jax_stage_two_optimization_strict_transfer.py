"""Device-boundary contracts for the standard Stage-II mirror."""

from __future__ import annotations

import ast
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = REPOSITORY_ROOT / "examples/jax/2_Intermediate/stage_two_optimization.py"
WORKFLOW_PATH = REPOSITORY_ROOT / "src/simsopt_jax/examples/stage_two_standard.py"
ROUTE_PATH = REPOSITORY_ROOT / "src/simsopt_jax/examples/scalar_stage.py"
NATIVE_PATH = REPOSITORY_ROOT / "examples/2_Intermediate/stage_two_optimization.py"


def _module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _history_bounded_calls(module: ast.Module, argument: str) -> list[str]:
    """Every call bounding ``maxcor`` with the named variable, by callee name."""
    return [
        (node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id)
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
        for keyword in node.keywords
        if keyword.arg == "maxcor"
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == argument
    ]


def _history_size(module: ast.Module) -> dict[str, int]:
    return {
        node.targets[0].id: node.value.value
        for node in module.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, int)
        and node.targets[0].id == "_STAGE_TWO_LBFGS_HISTORY_SIZE"
    }


def _native_maxcor() -> set[int]:
    """Every ``maxcor`` the native twin hands ``scipy.optimize.minimize``."""
    return {
        value.value
        for node in ast.walk(_module(NATIVE_PATH))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "minimize"
        for keyword in node.keywords
        if keyword.arg == "options" and isinstance(keyword.value, ast.Dict)
        for key, value in zip(keyword.value.keys, keyword.value.values)
        if isinstance(key, ast.Constant)
        and key.value == "maxcor"
        and isinstance(value, ast.Constant)
    }


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


def test_stage_two_lbfgs_history_matches_the_native_twins_maxcor() -> None:
    """One history size, equal to what the native script's two stages use.

    The workflow used to carry 10 while the native twin ran at 300, which made
    the mirror's curvature model a different model from the one it is timed and
    compared against.  The value is pinned to the native call sites rather than
    to a literal, so a native script that changes its ``maxcor`` fails here
    instead of silently reopening the gap.
    """
    native_maxcor = _native_maxcor()

    assert native_maxcor == {300}, (
        f"{NATIVE_PATH.name} no longer runs both stages at one maxcor: "
        f"{sorted(native_maxcor)}"
    )
    assert _history_size(_module(WORKFLOW_PATH)) == {
        "_STAGE_TWO_LBFGS_HISTORY_SIZE": native_maxcor.pop()
    }


def test_stage_two_lbfgs_history_is_bounded_independently_of_iterations() -> None:
    """The history size is a constant, and every stage on every driver gets it.

    This is the property the contract exists to protect: compiled state must not
    scale with the iteration budget.  It used to be checked by counting two
    ``serial_solve_jax`` calls; the workflow now hands the constant to one
    ``solve_scalar_stage`` call per stage, and that shared route is where the
    two driver branches live, so the check has two halves -- the constant is
    what the workflow bounds the history with, and the route's own ``maxcor``
    reaches both of its branches.  ``max_steps`` may bound neither.
    """
    workflow = _module(WORKFLOW_PATH)
    route = _module(ROUTE_PATH)
    stage_calls = [
        node
        for node in ast.walk(workflow)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_solve_stage"
    ]

    assert _history_bounded_calls(workflow, "_STAGE_TWO_LBFGS_HISTORY_SIZE") == [
        "solve_scalar_stage"
    ], (
        "the workflow must bound the L-BFGS history with the shared constant on "
        "the one route both drivers take; found "
        f"{_history_bounded_calls(workflow, '_STAGE_TWO_LBFGS_HISTORY_SIZE')}"
    )
    assert sorted(_history_bounded_calls(route, "maxcor")) == [
        "native_matched",
        "serial_solve_jax",
    ], (
        "both driver branches of solve_scalar_stage must bound the L-BFGS "
        "history with the maxcor it was handed; found "
        f"{sorted(_history_bounded_calls(route, 'maxcor'))}"
    )
    assert len(stage_calls) == 2, (
        "solve_standard_stage_two must run both length-weight stages through "
        f"_solve_stage; found {len(stage_calls)} calls"
    )
    assert not _history_bounded_calls(workflow, "max_steps"), (
        "the L-BFGS history must not be derived from the iteration budget"
    )
    assert not _history_bounded_calls(route, "max_steps"), (
        "the L-BFGS history must not be derived from the iteration budget"
    )
