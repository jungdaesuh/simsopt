"""Source-owned contracts for the stochastic Stage-II JAX mirror."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from simsopt_jax.examples.stochastic_stage_two import (
    STOCHASTIC_STAGE_TWO_OPTIMIZER_OBSERVABLES,
    stochastic_stage_two_optimizer_observables,
)
from simsopt_jax.solve.contracts import OptimizerResult
from simsopt_jax.solve.driver import Driver
from simsopt_jax.solve.scipy.contracts import ScipyLBFGSBOptions


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
    # The solve routes through the shared helper, never through a bare
    # ``minimize`` or a second copy of the driver branch in this script.
    assert "solve_stochastic_stage_two" in imported_names
    assert "stochastic_stage_two_optimizer_observables" in imported_names
    assert "minimize" not in called_names
    assert "serial_solve_jax" not in called_names
    assert {
        "sample_metadata",
        "out_of_sample_metadata",
        "training_flux_objective",
        "nominal_flux_objective",
        "out_of_sample_objective",
    } <= observable_keys


def test_stochastic_stage_two_runs_the_native_optimizer_and_publishes_once() -> None:
    """The live route is SciPy L-BFGS-B at the native example's policy.

    The history size comes from the configuration SSOT rather than a literal in
    this script, and it must not track ``max_steps`` — a history that grew with
    the budget would be a different policy at every scale.
    """
    source = EXAMPLE.read_text(encoding="utf-8")

    assert "driver=Driver.SCIPY_LBFGSB" in source
    assert "maxcor=configuration.lbfgs_history_size" in source
    assert "tol=configuration.rtol" in source
    assert "maxcor=min(max_steps, 400)" not in source
    assert "scalar_example_driver" not in source
    assert source.count("jax.device_get(") == 1


def test_stochastic_mirror_publishes_the_solver_verdict_it_does_not_collapse() -> None:
    """Every optimizer observable the mirror owes a reader reaches the artifact.

    ``status: "ok"`` is this example's *scientific* gate; the convergence
    verdict is ``solver_success``/``solver_status``, published beside it.
    """
    source = EXAMPLE.read_text(encoding="utf-8")

    assert "**stochastic_stage_two_optimizer_observables(result)" in source
    published = stochastic_stage_two_optimizer_observables(
        OptimizerResult(
            x=np.zeros(1),
            fun=0.0,
            jac=None,
            nit=20,
            nfev=25,
            njev=25,
            status=1,
            success=False,
            message="STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT",
            driver=Driver.SCIPY_LBFGSB,
            options_used=ScipyLBFGSBOptions.native_matched(
                maxiter=20, maxcor=400, tol=1.0e-15
            ),
            wallclock_s=0.5,
        )
    )

    assert set(published) == set(STOCHASTIC_STAGE_TWO_OPTIMIZER_OBSERVABLES)
    assert published["solver_success"] is False
    assert published["solver_driver"] == "scipy_lbfgsb"
    assert published["solver_options"]["ftol"] == 1.0e-15
    assert published["solver_options"]["gtol"] == 1.0e-15
    assert published["solver_options"]["maxfun"] == 15000
    assert published["solver_options"]["maxcor"] == 400
