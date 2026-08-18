"""Strict-transfer coverage for the production finite-build Stage-II workflow.

Inputs are staged on the device first; the prepared workflow is then solved
inside ``jax.transfer_guard("disallow")`` with ``host_transfer_audit()`` open,
and the endpoint is materialized only after the guarded call has returned.

``final_result`` is the positive control.  The typed optimizer's terminal
read-back is expected and must be observed, otherwise a renamed phase label
would quietly turn the assertions beside it into statements about a ledger
that saw nothing.  Against that control, zero ``advance`` and zero
``callback`` records say the fused device route ran: the stepwise route
labels its per-accepted-step host observation ``callback`` when an observer
is attached and ``advance`` when one is not, so neither a stepwise fallback
nor a callback-configured lane can reach these counts.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import jax
import numpy as np
import pytest
from simsopt.field import (
    Coil,
    Current,
    apply_symmetries_to_currents,
    apply_symmetries_to_curves,
)
from simsopt.geo import (
    CurveLength,
    SurfaceRZFourier,
    create_equally_spaced_curves,
    create_multifilament_grid,
)
from simsopt_jax.core import compute_filament_offsets
from simsopt_jax.examples.stage_two_finitebuild import (
    PreparedFiniteBuildStageTwo,
    prepare_finite_build_stage_two,
    solve_finite_build_stage_two,
)
from simsopt_jax.geo.optimizers.private._common import (
    private_optimizer_runtime_is_supported,
)
from simsopt_jax.runtime.host_boundary import host_transfer_audit
from simsopt_jax.solve.contracts import OptimizerResult
from simsopt_jax.solve.driver import Driver
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives import (
    FiniteBuildStageTwoConfig,
    finite_build_stage_two_diagnostics,
    make_finite_build_stage_two_objective,
)
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

TEST_DATA = Path(__file__).resolve().parents[2] / "tests" / "test_files"
# The Advanced example's bounded-scale construction.
NUM_BASE_CURVES = 4
NUM_FILAMENTS_N = 2
NUM_FILAMENTS_B = 3
GAP_SIZE_N = 0.02
GAP_SIZE_B = 0.04
ROTATION_ORDER = 1
SOLVE_OBJECTIVE_SCALE = 1.0e-4
SHORT_BUDGET = 3
LONG_BUDGET = 6

pytestmark = [
    pytest.mark.private_optimizer_runtime,
    pytest.mark.skipif(
        not private_optimizer_runtime_is_supported(jax.__version__),
        reason="The fused L-BFGS-B lane is validated on the pinned JAX runtime.",
    ),
]


@pytest.fixture(scope="module")
def prepared() -> PreparedFiniteBuildStageTwo:
    """Stage the bounded finite-build problem and prepare it once.

    Construction reads host geometry and places it on the device, so it must
    happen before any strict guard is entered.  The record is shared across
    the tests below because ``solve_finite_build_stage_two`` restarts from
    ``initial_parameters`` on every call.
    """
    surface = SurfaceRZFourier.from_vmec_input(
        TEST_DATA / "input.LandremanPaul2021_QA",
        range="half period",
        nphi=4,
        ntheta=4,
    )
    base_curves = create_equally_spaced_curves(
        NUM_BASE_CURVES,
        surface.nfp,
        stellsym=True,
        R0=1.0,
        R1=0.7,
        order=2,
        numquadpoints=8,
        use_jax_curve=False,
    )
    filament_count = NUM_FILAMENTS_N * NUM_FILAMENTS_B
    base_currents = []
    for index in range(NUM_BASE_CURVES):
        current = Current(1.0)
        if index == 0:
            current.fix_all()
        base_currents.append(current * (1.0e5 / filament_count))
    base_filaments = list(
        itertools.chain.from_iterable(
            create_multifilament_grid(
                curve,
                NUM_FILAMENTS_N,
                NUM_FILAMENTS_B,
                GAP_SIZE_N,
                GAP_SIZE_B,
                rotation_order=ROTATION_ORDER,
            )
            for curve in base_curves
        )
    )
    filament_currents = list(
        itertools.chain.from_iterable(
            [current] * filament_count for current in base_currents
        )
    )
    coils = [
        Coil(curve, current)
        for curve, current in zip(
            apply_symmetries_to_curves(base_filaments, surface.nfp, True),
            apply_symmetries_to_currents(filament_currents, surface.nfp, True),
            strict=True,
        )
    ]
    field = BiotSavartJAX(coils)
    flux_spec = SquaredFluxJAX(surface, field).fixed_surface_flux_spec()
    config = FiniteBuildStageTwoConfig(
        num_base_curves=NUM_BASE_CURVES,
        filament_offsets=compute_filament_offsets(
            numfilaments_n=NUM_FILAMENTS_N,
            numfilaments_b=NUM_FILAMENTS_B,
            gapsize_n=GAP_SIZE_N,
            gapsize_b=GAP_SIZE_B,
        ),
        symmetry_copies=surface.nfp * 2,
        length_targets=tuple(float(CurveLength(curve).J()) for curve in base_curves),
        length_weight=1.0e-2,
        curve_curve_minimum_distance=0.1,
        curve_curve_weight=10.0,
    )
    return prepare_finite_build_stage_two(
        objective_fn=make_finite_build_stage_two_objective(field, flux_spec, config),
        diagnostics_fn=finite_build_stage_two_diagnostics(field, flux_spec, config),
        initial_parameters=jax.device_put(np.asarray(field.x, dtype=np.float64)),
        objective_scale=jax.device_put(
            np.asarray(SOLVE_OBJECTIVE_SCALE, dtype=np.float64)
        ),
    )


def _audited_solve(
    prepared: PreparedFiniteBuildStageTwo,
    *,
    max_steps: int,
) -> tuple[OptimizerResult, dict[str, int]]:
    """Solve under the strict guard and return the result with its ledger."""
    with host_transfer_audit() as audit, jax.transfer_guard("disallow"):
        result = solve_finite_build_stage_two(
            prepared,
            driver=Driver.SIMSOPT_LBFGSB,
            max_steps=max_steps,
            rtol=1.0e-15,
            atol=1.0e-12,
        )
    return result, {entry.phase: entry.calls for entry in audit.summary()}


def test_finite_build_workflow_observes_no_step_on_the_host(
    prepared: PreparedFiniteBuildStageTwo,
) -> None:
    solve_result, transfer_calls = _audited_solve(prepared, max_steps=SHORT_BUDGET)
    solution = np.asarray(jax.device_get(prepared.problem.x), dtype=np.float64)

    assert transfer_calls.get("final_result", 0) > 0, (
        "the transfer audit recorded no endpoint read-back, so it observed "
        f"nothing to disprove: {transfer_calls}"
    )
    assert transfer_calls.get("advance", 0) == 0, (
        "the finite-build workflow round-trips accepted steps through the "
        "host; the fused run mode did not reach the optimizer lane"
    )
    assert transfer_calls.get("callback", 0) == 0, (
        "an accepted-step observer was configured, which selects the "
        "callback-capable stepwise route instead of the fused one"
    )
    assert solve_result.nit == SHORT_BUDGET
    assert np.all(np.isfinite(solution))


def test_finite_build_workflow_host_transfers_do_not_grow_with_the_step_budget(
    prepared: PreparedFiniteBuildStageTwo,
) -> None:
    """The affirmative statement: the boundary cost is O(1) in accepted steps.

    Doubling the budget doubles the accepted steps.  If any host boundary were
    crossed per step, at least one phase's call count would move with it; a
    ledger that is identical under both budgets can only be describing
    transfers that happen once per solve.
    """
    short_result, short_calls = _audited_solve(prepared, max_steps=SHORT_BUDGET)
    long_result, long_calls = _audited_solve(prepared, max_steps=LONG_BUDGET)

    assert long_result.nit > short_result.nit
    assert long_calls == short_calls, (
        "host transfers scaled with the step budget: "
        f"{short_result.nit} steps gave {short_calls}, "
        f"{long_result.nit} steps gave {long_calls}"
    )


def test_finite_build_workflow_records_no_unclassified_host_transfers(
    prepared: PreparedFiniteBuildStageTwo,
) -> None:
    """Every device-to-host read the workflow makes must name its phase."""
    _solve_result, transfer_calls = _audited_solve(prepared, max_steps=SHORT_BUDGET)

    assert transfer_calls.get("unclassified", 0) == 0, (
        "the workflow materialized values outside every named transfer "
        f"phase: {transfer_calls}"
    )
