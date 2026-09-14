"""What the standard Stage-II mirror says about its optimizer, per stage.

The finding these tests answer is that a mirror ``status`` of ``ok`` can mean
"the objective improved" while both stages returned ``success=False``, so a
reader who translates ``ok`` into convergence is reading something the run never
claimed.  The fix is not a different word for ``status``; it is publishing the
optimizer's own verdict per stage, which is what is pinned here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from simsopt_jax.examples.stage_two_standard import (
    STANDARD_STAGE_TWO_OPTIMIZER_OBSERVABLES,
    StandardStageTwoDeviceResult,
    StandardStageTwoState,
    standard_stage_two_optimizer_observables,
)
from simsopt_jax.solve.contracts import OptimizerResult
from simsopt_jax.solve.driver import Driver
from simsopt_jax.solve.scipy.contracts import ScipyLBFGSBOptions

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MIRROR = (
    REPOSITORY_ROOT
    / "examples"
    / "jax"
    / "2_Intermediate"
    / "stage_two_optimization.py"
)


def _state() -> StandardStageTwoState:
    scalar = jnp.asarray(1.0, dtype=jnp.float64)
    return StandardStageTwoState(
        parameters=jnp.zeros(2, dtype=jnp.float64),
        objective=scalar,
        objective_gradient=jnp.zeros(2, dtype=jnp.float64),
        squared_flux=scalar,
        geometric_penalty=scalar,
        maximum_normal_field=scalar,
        total_curve_length=scalar,
    )


def _optimizer(*, success: bool, status: int, message: str, nit: int) -> OptimizerResult:
    return OptimizerResult(
        x=np.zeros(2),
        fun=1.0,
        jac=np.zeros(2),
        nit=nit,
        nfev=nit * 2,
        njev=nit * 2,
        status=status,
        success=success,
        message=message,
        driver=Driver.SCIPY_LBFGSB,
        options_used=ScipyLBFGSBOptions(maxiter=400, maxcor=300, ftol=1e-15, gtol=1e-15),
        wallclock_s=0.5 * nit,
    )


def _result(*, first_success: bool, second_success: bool) -> StandardStageTwoDeviceResult:
    state = _state()
    return StandardStageTwoDeviceResult(
        initial=state,
        first=state,
        final=state,
        taylor_errors=jnp.zeros(5, dtype=jnp.float64),
        first_optimizer=_optimizer(
            success=first_success,
            status=0 if first_success else 1,
            message="CONVERGENCE" if first_success else "STOP: TOTAL NO. OF "
            "ITERATIONS REACHED LIMIT",
            nit=400,
        ),
        second_optimizer=_optimizer(
            success=second_success,
            status=0 if second_success else 1,
            message="CONVERGENCE" if second_success else "STOP: TOTAL NO. OF "
            "ITERATIONS REACHED LIMIT",
            nit=200,
        ),
        two_stage_minimize_seconds=310.0,
        whole_call_seconds=340.0,
        execution_device="cuda:0",
    )


def test_each_stage_reports_its_own_optimizer_verdict() -> None:
    """A converged stage and a capped stage must not collapse into one word."""
    published = standard_stage_two_optimizer_observables(
        _result(first_success=True, second_success=False)
    )

    assert published["solver_stage_success"] == (True, False)
    assert published["solver_status"] == (0, 1)
    assert published["solver_iterations"] == (400, 200)
    assert published["solver_evaluations"] == (800, 400)
    assert published["solver_gradient_evaluations"] == (800, 400)
    assert published["solver_message"][0] == "CONVERGENCE"
    assert "ITERATIONS REACHED LIMIT" in published["solver_message"][1]
    assert published["solver_success"] is False, (
        "a two-stage run is only a success when both stages succeeded; collapsing "
        "a mixed pair to True is exactly the claim this observable must not make"
    )


def test_no_optimizer_observable_is_reduced_to_a_verdict_word() -> None:
    """Nothing here may be published as ``ok``/``failed`` or any other label."""
    published = standard_stage_two_optimizer_observables(
        _result(first_success=False, second_success=False)
    )

    flattened = [
        value
        for name, value in published.items()
        if name != "solver_message"
        for value in (value if isinstance(value, tuple) else (value,))
    ]

    assert "ok" not in flattened
    assert "failed" not in flattened
    assert "converged" not in flattened


def test_published_names_are_the_declared_names() -> None:
    """The name tuple a driver republishes from is the helper's actual output."""
    published = standard_stage_two_optimizer_observables(
        _result(first_success=True, second_success=True)
    )

    assert tuple(published) == STANDARD_STAGE_TWO_OPTIMIZER_OBSERVABLES


def test_timings_separate_the_minimize_region_from_the_whole_call() -> None:
    """Per-stage minimize clocks, their region, and the whole call are distinct."""
    published = standard_stage_two_optimizer_observables(
        _result(first_success=True, second_success=True)
    )

    assert published["first_stage_minimize_seconds"] == 200.0
    assert published["second_stage_minimize_seconds"] == 100.0
    assert published["two_stage_minimize_seconds"] == 310.0
    assert published["standard_solve_seconds"] == 340.0
    assert (
        published["two_stage_minimize_seconds"] < published["standard_solve_seconds"]
    ), "the whole call must contain the minimize region, never equal it by accident"


def test_options_published_per_stage_are_the_ones_the_stage_ran_under() -> None:
    """``solver_options`` is ``OptimizerResult.options_used``, not a declaration."""
    published = standard_stage_two_optimizer_observables(
        _result(first_success=True, second_success=True)
    )
    first, second = published["solver_options"]

    assert first == second
    assert first["type"] == "ScipyLBFGSBOptions"
    assert first["maxcor"] == 300
    assert first["ftol"] == 1e-15
    assert first["gtol"] == 1e-15
    assert first["maxfun"] == ScipyLBFGSBOptions().maxfun


def test_execution_device_is_attested_from_the_endpoint_array() -> None:
    published = standard_stage_two_optimizer_observables(
        _result(first_success=True, second_success=True)
    )

    assert published["execution_device"] == "cuda:0"


def test_the_mirror_publishes_the_per_stage_optimizer_observables() -> None:
    """The shipped mirror hands its result to the helper rather than re-typing it."""
    source = MIRROR.read_text(encoding="utf-8")
    module = ast.parse(source)
    called = {
        node.func.id
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "standard_stage_two_optimizer_observables" in called
    assert source.count("jax.device_get(") == 1
