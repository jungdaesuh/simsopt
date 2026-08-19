"""Artifact-free coverage of the optimizer policy each fused lane constructs.

The F3 campaign charter hashes ``{method, maxiter, maxfun, gtol, ftol, maxcor,
maxls}`` per leg and voids a leg whose policy differs from the archived one, so
what each binder puts in that record is a contract, not an implementation
detail.  These tests read the constructed options record — the same object the
charter's policy-identity sha covers — and assert its field values; the
``minimize`` seam is intercepted only to reach that record, never asserted on
for having been called.

Nothing here solves a real problem: a three-DOF quadratic is enough, because
the policy is chosen before any physics is seen.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt_jax.examples import fused_lane
from simsopt_jax.examples.fused_lane import (
    PreparedFusedLaneSolve,
    prepare_fused_lane_solve,
    solve_fused_lane,
)
from simsopt_jax.examples.single_stage_flat675 import (
    FLAT675_LBFGS_HISTORY,
    FLAT675_LBFGS_MAXLS,
    solve_single_stage_flat675,
)
from simsopt_jax.examples.stage_two_finitebuild import (
    FINITE_BUILD_LBFGS_HISTORY,
    solve_finite_build_stage_two,
)
from simsopt_jax.solve.driver import Driver
from simsopt_jax.solve.simsopt.contracts import (
    SimsoptBFGSOptions,
    SimsoptLBFGSBOptions,
)

# The archived genuine-675 lane record's L-BFGS-B policy block.
ARCHIVED_MAXCOR = 300
ARCHIVED_MAXLS = 8
STEP_BUDGET = 2


@pytest.fixture
def prepared() -> PreparedFusedLaneSolve:
    return prepare_fused_lane_solve(
        objective_fn=lambda x: jnp.sum(x * x),
        diagnostics_fn=lambda x: x,
        initial_parameters=jax.device_put(np.ones(3, dtype=np.float64)),
        objective_scale=jax.device_put(np.asarray(1.0, dtype=np.float64)),
    )


@pytest.fixture
def constructed_options(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Capture the options record each solve hands to the dispatcher."""
    recorded: list[Any] = []
    real_minimize = fused_lane.minimize

    def recording_minimize(
        value_and_grad_fn: Any, x0: Any, *, driver: Any, options: Any
    ) -> Any:
        recorded.append(options)
        return real_minimize(value_and_grad_fn, x0, driver=driver, options=options)

    monkeypatch.setattr(fused_lane, "minimize", recording_minimize)
    return recorded


def test_flat675_binder_constructs_the_archived_policy(
    prepared: PreparedFusedLaneSolve,
    constructed_options: list[Any],
) -> None:
    """The flat-675 lane must run the archived history and line-search cap."""
    solve_single_stage_flat675(
        prepared,
        driver=Driver.SIMSOPT_LBFGSB,
        max_steps=STEP_BUDGET,
        rtol=1.0e-15,
        atol=1.0e-12,
    )

    (options,) = constructed_options
    assert isinstance(options, SimsoptLBFGSBOptions)
    assert options.maxcor == FLAT675_LBFGS_HISTORY == ARCHIVED_MAXCOR
    assert options.maxls == FLAT675_LBFGS_MAXLS == ARCHIVED_MAXLS
    assert options.maxiter == STEP_BUDGET
    # The charter states this cap cannot bind: at maxls=8 an iteration costs at
    # most nine evaluations, and 20 per iteration is well above nine.
    assert options.maxfun == STEP_BUDGET * 20


def test_finite_build_binder_keeps_the_default_line_search(
    prepared: PreparedFusedLaneSolve,
    constructed_options: list[Any],
) -> None:
    """The certified finite-build lane must not inherit the flat-675 pin."""
    solve_finite_build_stage_two(
        prepared,
        driver=Driver.SIMSOPT_LBFGSB,
        max_steps=STEP_BUDGET,
        rtol=1.0e-15,
        atol=1.0e-12,
    )

    (options,) = constructed_options
    assert options.maxcor == FINITE_BUILD_LBFGS_HISTORY == 10
    assert options.maxls == SimsoptLBFGSBOptions().maxls == 20


def test_unset_line_search_arguments_take_their_optimizer_defaults(
    prepared: PreparedFusedLaneSolve,
    constructed_options: list[Any],
) -> None:
    """Naming neither knob reproduces the behavior from before they existed."""
    solve_fused_lane(
        prepared,
        driver=Driver.SIMSOPT_LBFGSB,
        max_steps=STEP_BUDGET,
        rtol=1.0e-15,
        atol=1.0e-12,
        lbfgs_history=10,
    )
    solve_fused_lane(
        prepared,
        driver=Driver.SIMSOPT_BFGS,
        max_steps=STEP_BUDGET,
        rtol=1.0e-15,
        atol=1.0e-12,
        lbfgs_history=10,
    )

    lbfgsb_options, bfgs_options = constructed_options
    assert lbfgsb_options.maxls == SimsoptLBFGSBOptions().maxls
    assert (
        bfgs_options.line_search_max_steps == SimsoptBFGSOptions().line_search_max_steps
    )


def test_named_line_search_caps_reach_their_optimizer(
    prepared: PreparedFusedLaneSolve,
    constructed_options: list[Any],
) -> None:
    solve_fused_lane(
        prepared,
        driver=Driver.SIMSOPT_LBFGSB,
        max_steps=STEP_BUDGET,
        rtol=1.0e-15,
        atol=1.0e-12,
        lbfgs_history=10,
        lbfgs_line_search_max_steps=3,
    )
    solve_fused_lane(
        prepared,
        driver=Driver.SIMSOPT_BFGS,
        max_steps=STEP_BUDGET,
        rtol=1.0e-15,
        atol=1.0e-12,
        lbfgs_history=10,
        line_search_max_steps=7,
    )

    lbfgsb_options, bfgs_options = constructed_options
    assert lbfgsb_options.maxls == 3
    assert bfgs_options.line_search_max_steps == 7


@pytest.mark.parametrize(
    ("driver", "argument", "message"),
    [
        pytest.param(
            Driver.SIMSOPT_LBFGSB,
            "line_search_max_steps",
            "configured through lbfgs_line_search_max_steps",
            id="bfgs-knob-under-lbfgsb",
        ),
        pytest.param(
            Driver.SIMSOPT_BFGS,
            "lbfgs_line_search_max_steps",
            "configured through line_search_max_steps",
            id="lbfgsb-knob-under-bfgs",
        ),
    ],
)
def test_each_line_search_knob_refuses_the_other_driver(
    prepared: PreparedFusedLaneSolve,
    driver: Driver,
    argument: str,
    message: str,
) -> None:
    """The two knobs name different optimizers and must not be interchanged."""
    with pytest.raises(TypeError, match=message):
        solve_fused_lane(
            prepared,
            driver=driver,
            max_steps=STEP_BUDGET,
            rtol=1.0e-15,
            atol=1.0e-12,
            lbfgs_history=10,
            **{argument: 5},
        )
