"""Production flat-675 single-stage optimization workflow.

The internal workflow every flat-675 JAX caller routes through: the adapter
layer builds the physics (objective and diagnostics closures) from a frozen
input bundle and hands them here; the fused lane in
:mod:`simsopt_jax.examples.fused_lane` owns the traceable program construction
and the solve, and this module owns the campaign's frozen optimizer selection.

The flat formulation has no nested Boozer solve, so the whole objective is one
device program and the solve is the fused on-device L-BFGS-B lane end to end.
That is also why the workflow drops the source campaign's host-side rejection
and anchor protocol: those existed to keep a SciPy driver consistent across
proposals whose inner solve could fail on the host, and no such boundary
survives here.

The L-BFGS history is read off the archived genuine-675 campaign whose
certificate this workflow reproduces — the ``maxcor`` of the L-BFGS-B policy
published in its lane record.  It is deliberately not a configuration knob:
``solve_single_stage_flat675`` accepts no history argument.
"""

from __future__ import annotations

from simsopt_jax.examples.fused_lane import (
    PreparedFusedLaneSolve,
    prepare_fused_lane_solve,
    solve_fused_lane,
)
from simsopt_jax.solve.contracts import OptimizerResult
from simsopt_jax.solve.driver import Driver

# Frozen campaign selection, transcribed from the archived lane record's
# ``policy.maxcor``.  Not a solve parameter.
FLAT675_LBFGS_HISTORY = 300

# The flat-675 lane prepares exactly the shared fused lane; the name is kept
# so callers spell the workflow the way its tests and examples do.
PreparedSingleStageFlat675 = PreparedFusedLaneSolve
prepare_single_stage_flat675 = prepare_fused_lane_solve


def solve_single_stage_flat675(
    prepared: PreparedFusedLaneSolve,
    *,
    driver: Driver,
    max_steps: int,
    rtol: float,
    atol: float,
) -> OptimizerResult:
    """Solve the prepared flat-675 problem at the frozen L-BFGS history."""
    return solve_fused_lane(
        prepared,
        driver=driver,
        max_steps=max_steps,
        rtol=rtol,
        atol=atol,
        lbfgs_history=FLAT675_LBFGS_HISTORY,
    )


__all__ = [
    "FLAT675_LBFGS_HISTORY",
    "PreparedSingleStageFlat675",
    "prepare_single_stage_flat675",
    "solve_single_stage_flat675",
]
