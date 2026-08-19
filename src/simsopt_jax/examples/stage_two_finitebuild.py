"""Production finite-build Stage-II coil optimization workflow.

The internal workflow both finite-build JAX callers route through: the
Advanced example and the parity case build their physics (objective and
diagnostics closures) from the adapter layer and hand them here; the fused
lane in :mod:`simsopt_jax.examples.fused_lane` owns the traceable program
construction and the solve, and this module owns the campaign's frozen
optimizer selection.

The L-BFGS history is the frozen selection of the successor speed campaign
(``docs/jax_gpu_finitebuild_native_speed_successor_plan.md``): history 10
was the only history whose crossing endpoint passed the frozen endpoint
contract, selected over histories 20 and 40 on qualifying warm solve time.
It is deliberately not a configuration knob — ``solve_finite_build_stage_two``
accepts no history argument.
"""

from __future__ import annotations

from simsopt_jax.examples.fused_lane import (
    PreparedFusedLaneSolve,
    prepare_fused_lane_solve,
    solve_fused_lane,
)
from simsopt_jax.solve.contracts import OptimizerResult
from simsopt_jax.solve.driver import Driver

# Frozen campaign selection, single source of truth for both lanes: the
# parity case's native twin reads it so the matched workflows share one
# history.  Not a solve parameter.
FINITE_BUILD_LBFGS_HISTORY = 10

# The finite-build lane prepares exactly the shared fused lane; the name is
# kept because the Advanced example, the parity case, and the strict-transfer
# test spell the workflow this way.
PreparedFiniteBuildStageTwo = PreparedFusedLaneSolve
prepare_finite_build_stage_two = prepare_fused_lane_solve


def solve_finite_build_stage_two(
    prepared: PreparedFusedLaneSolve,
    *,
    driver: Driver,
    max_steps: int,
    rtol: float,
    atol: float,
    line_search_max_steps: int | None = None,
) -> OptimizerResult:
    """Solve the prepared finite-build problem at the frozen L-BFGS history."""
    return solve_fused_lane(
        prepared,
        driver=driver,
        max_steps=max_steps,
        rtol=rtol,
        atol=atol,
        lbfgs_history=FINITE_BUILD_LBFGS_HISTORY,
        line_search_max_steps=line_search_max_steps,
    )


__all__ = [
    "FINITE_BUILD_LBFGS_HISTORY",
    "PreparedFiniteBuildStageTwo",
    "prepare_finite_build_stage_two",
    "solve_finite_build_stage_two",
]
