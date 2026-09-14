"""Source-owned contracts for the force-optimized Stage-II JAX mirror."""

from __future__ import annotations

import ast
from pathlib import Path

from simsopt_jax.examples import STANDARD_STAGE_TWO_OPTIMIZER_OBSERVABLES


EXAMPLE = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "jax"
    / "3_Advanced"
    / "coil_forces.py"
)


def test_coil_forces_is_an_exact_name_public_jax_example() -> None:
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

    source = EXAMPLE.read_text(encoding="utf-8")

    assert "make_force_stage_two_objective" in imported_names
    assert "force_stage_two_diagnostics" in imported_names
    # ``Driver.SIMSOPT_BFGS`` keeps no curvature history and bounds its own line
    # search, so the mirror drives it directly; both L-BFGS-B routes belong to
    # the shared owner.
    assert "serial_solve_jax" in imported_names
    # Native drives its two stages with SciPy L-BFGS-B.  How the device
    # objective reaches that routine at native's policy is
    # ``simsopt_jax.examples.scalar_stage.solve_scalar_stage``'s knowledge, and
    # is pinned against the native scripts in
    # ``tests/jax/examples/test_stage_two_standard_policy.py``; what this mirror
    # owns is which constants it hands over.
    assert "solve_scalar_stage" in imported_names
    assert "solve_scalar_stage" in called_names
    assert "maxcor=NATIVE_HISTORY_SIZE" in source
    assert "tol=NATIVE_TOLERANCE" in source
    assert {
        "initial_parameters",
        "start_parameters",
        "start_objective",
        "start_gradient",
        "taylor_errors",
        "force_objective",
        "maximum_force",
        "vacuum_energy",
        "final_gradient",
    } <= observable_keys


def test_coil_forces_publishes_the_shared_two_stage_solver_spelling() -> None:
    """One reader reads every two-stage mirror's solver outcome the same way."""
    source = EXAMPLE.read_text(encoding="utf-8")
    module = ast.parse(source)
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
    shared = set(STANDARD_STAGE_TWO_OPTIMIZER_OBSERVABLES)

    assert "two_stage_optimizer_observables" in called_names
    assert "**two_stage_optimizer_observables(" in source
    assert {
        "solver_driver",
        "solver_success",
        "solver_stage_success",
        "solver_status",
        "solver_message",
        "solver_iterations",
        "solver_evaluations",
        "solver_gradient_evaluations",
    } <= shared
    assert not {name for name in observable_keys if name.startswith("solver_")}, (
        "no solver_ name may be re-typed in this mirror: the shared producer "
        "owns every one of them, and a locally spelled key is how the two "
        "mirrors drifted into publishing one name with two types"
    )
