"""Strict-transfer coverage for the flat-675 single-stage workflow.

The flat formulation exists so that a single-stage solve has no nested Boozer
solver and therefore no per-step host boundary.  This file is what holds that
claim: the prepared workflow is solved inside ``jax.transfer_guard("disallow")``
with ``host_transfer_audit()`` open, and the endpoint is materialized only
after the guarded call has returned.

``final_result`` is the positive control.  The typed optimizer's terminal
read-back is expected and must be observed, otherwise a renamed phase label
would quietly turn the assertions beside it into statements about a ledger
that saw nothing.  Against that control, zero ``advance`` and zero
``callback`` records say the fused device route ran: the stepwise route labels
its per-accepted-step host observation ``callback`` when an observer is
attached and ``advance`` when one is not, so neither a stepwise fallback nor a
callback-configured lane can reach these counts.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt_jax.examples.single_stage_flat675 import (
    PreparedSingleStageFlat675,
    prepare_single_stage_flat675,
    solve_single_stage_flat675,
)
from simsopt_jax.geo.optimizers.private._common import (
    private_optimizer_runtime_is_supported,
)
from simsopt_jax.runtime.host_boundary import host_transfer_audit
from simsopt_jax.solve.contracts import OptimizerResult
from simsopt_jax.solve.driver import Driver
from simsopt_jax_adapters.geo.flat675 import (
    FLAT675_OBJECTIVE_TERM_KEYS,
    FLAT675_OUTER_DOF_COUNT,
    bind_flat675_programs,
    load_flat675_bundle,
)

ARTIFACT_ROOT = Path.home() / "simsopt_mixed_artifacts"
BUNDLE_ROOT = ARTIFACT_ROOT / "genuine675-r3-input-1c23f6c5-20260721-r1"
# The certified objective is unscaled; the parametric scale exists so callers
# can republish without retracing, not to reshape this problem.
SOLVE_OBJECTIVE_SCALE = 1.0
STEP_BUDGET = 3
# The archived native lane's final certificate after the same three-step
# fixed budget from the same start candidate, driven by host SciPy L-BFGS-B
# under the rejection/anchor protocol this fused workflow drops.
ARCHIVED_THREE_STEP_OBJECTIVE = 1.8133486877704437

pytestmark = [
    pytest.mark.private_optimizer_runtime,
    pytest.mark.skipif(
        not private_optimizer_runtime_is_supported(jax.__version__),
        reason="The fused L-BFGS-B lane is validated on the pinned JAX runtime.",
    ),
    pytest.mark.skipif(
        not BUNDLE_ROOT.is_dir(),
        reason="the frozen genuine-675 input bundle is host-local",
    ),
]


@pytest.fixture(scope="module")
def prepared() -> PreparedSingleStageFlat675:
    """Read the frozen bundle and prepare the workflow once.

    Construction reads host JSON and places it on the device, so it must
    happen before any strict guard is entered.  The record is shared across
    the tests below because ``solve_single_stage_flat675`` restarts from
    ``initial_parameters`` on every call.
    """
    bundle = load_flat675_bundle(BUNDLE_ROOT)
    programs = bind_flat675_programs(
        material=bundle.material,
        objective_policy=bundle.objective_policy,
        boozer_policy=bundle.boozer_policy,
    )
    return prepare_single_stage_flat675(
        objective_fn=programs.objective_fn,
        diagnostics_fn=programs.diagnostics_fn,
        initial_parameters=jax.device_put(bundle.start_candidate.outer_vector()),
        objective_scale=jax.device_put(
            np.asarray(SOLVE_OBJECTIVE_SCALE, dtype=np.float64)
        ),
    )


def _audited_solve(
    prepared: PreparedSingleStageFlat675,
    *,
    max_steps: int,
) -> tuple[OptimizerResult, dict[str, int]]:
    """Solve under the strict guard and return the result with its ledger."""
    with host_transfer_audit() as audit, jax.transfer_guard("disallow"):
        result = solve_single_stage_flat675(
            prepared,
            driver=Driver.SIMSOPT_LBFGSB,
            max_steps=max_steps,
            rtol=1.0e-15,
            atol=1.0e-12,
        )
    return result, {entry.phase: entry.calls for entry in audit.summary()}


def test_flat675_workflow_prepares_the_full_675_coordinate_problem(
    prepared: PreparedSingleStageFlat675,
) -> None:
    """The prepared problem carries every outer coordinate, not a reduction."""
    assert prepared.initial_parameters.shape == (FLAT675_OUTER_DOF_COUNT,)
    assert prepared.initial_parameters.dtype == jnp.dtype(jnp.float64)


def test_flat675_workflow_observes_no_step_on_the_host(
    prepared: PreparedSingleStageFlat675,
) -> None:
    solve_result, transfer_calls = _audited_solve(prepared, max_steps=STEP_BUDGET)
    solution = np.asarray(jax.device_get(prepared.problem.x), dtype=np.float64)

    assert transfer_calls.get("final_result", 0) > 0, (
        "the transfer audit recorded no endpoint read-back, so it observed "
        f"nothing to disprove: {transfer_calls}"
    )
    assert transfer_calls.get("advance", 0) == 0, (
        "the flat-675 workflow round-trips accepted steps through the host; "
        "the fused run mode did not reach the optimizer lane"
    )
    assert transfer_calls.get("callback", 0) == 0, (
        "an accepted-step observer was configured, which selects the "
        "callback-capable stepwise route instead of the fused one"
    )
    assert transfer_calls.get("unclassified", 0) == 0, (
        "the workflow materialized values outside every named transfer "
        f"phase: {transfer_calls}"
    )
    assert solve_result.nit == STEP_BUDGET
    assert solution.shape == (FLAT675_OUTER_DOF_COUNT,)
    assert np.all(np.isfinite(solution))


def test_flat675_workflow_lands_on_the_archived_three_step_endpoint(
    prepared: PreparedSingleStageFlat675,
) -> None:
    """The fused device solve walks the archived lane's three-step trajectory.

    The archived lane spent the same budget from the same start candidate on
    host SciPy L-BFGS-B under a rejection/anchor protocol this workflow drops.
    Landing on its objective says the dropped protocol never fired over these
    three steps and that the fused lane is the same optimizer on the same
    problem — a claim no per-point objective comparison can make.
    """
    solve_result, _transfer_calls = _audited_solve(prepared, max_steps=STEP_BUDGET)

    endpoint_objective = float(solve_result.fun)
    relative_error = (
        abs(endpoint_objective - ARCHIVED_THREE_STEP_OBJECTIVE)
        / ARCHIVED_THREE_STEP_OBJECTIVE
    )
    assert relative_error < 1.0e-9, (
        "the fused three-step endpoint left the archived native trajectory: "
        f"got {endpoint_objective!r}, archive recorded "
        f"{ARCHIVED_THREE_STEP_OBJECTIVE!r} (relative error {relative_error:.3e})"
    )


def test_flat675_workflow_diagnostics_report_every_certified_term(
    prepared: PreparedSingleStageFlat675,
) -> None:
    """The prepared diagnostics program publishes the eight weighted terms."""
    terms = np.asarray(
        jax.device_get(prepared.diagnostics(prepared.initial_parameters)),
        dtype=np.float64,
    )

    assert terms.shape == (len(FLAT675_OBJECTIVE_TERM_KEYS),)
    assert np.all(np.isfinite(terms))
