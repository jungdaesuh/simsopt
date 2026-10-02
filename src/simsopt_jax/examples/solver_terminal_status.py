"""Terminal status of the example solvers whose provider reports no status.

The greedy upstream solvers mirrored by these examples return arrays and print
their stop line; they never report a convergence status, so every consumer has
to DERIVE the stop reason from the returned data, and must derive it the same
way. This module is that single derivation for the GSCO family
(``simsoptpp::GSCO``), shared by the executable examples
(``examples/jax/2_Intermediate/wireframe_gsco_modular.py``,
``wireframe_gsco_sector_saddle.py``,
``examples/jax/3_Advanced/wireframe_gsco_multistep.py``). It lives in the
installed package because the examples are executed as standalone
scripts, with no repository root on ``sys.path``
(``tests/integration/test_jax_external_solver_free_examples.py``).

GSCO has three stop conditions (``src/simsoptpp/wireframe_optimization.cpp``
lines 264-279 **of this branch**, the binary the lanes link): no eligible loop,
the best loop would undo the previous one, or the last allowed iteration. The
first two are the solver deciding it is done -- ``converged``; the third is the
declared budget -- ``budget_exhausted``. A run that adopted no loop at all moved
nothing and is ``failed``, whichever of the first two conditions ended it. No
scientific predicate ("final objective below initial") takes part: upstream's
own run is not required to satisfy one.

Pure in its own computation: every function here is a total function of its
arguments, with no I/O, no globals and no JAX call. Importing the module is not
JAX-free: it takes the ``ExecutionScale`` alias from
``simsopt_jax.examples.execution``, which imports ``simsopt_jax.backend.runtime``
and ``simsopt_jax.solve.driver`` and therefore JAX. That costs the three GSCO
examples nothing -- they import JAX anyway -- and it keeps the scale vocabulary
in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from simsopt_jax.examples.execution import ExecutionScale

#: ``stop_undone_loop`` with the undo accepted; upstream prints
#: "Stopping iterations: minimum objective reached".
GSCO_MINIMUM_OBJECTIVE_REACHED: Final = "gsco_minimum_objective_reached"
#: An early stop whose last candidate loop was rejected. Upstream prints either
#: "no eligible loops" or "minimum objective reached" here, and the returned
#: tuple (x, loop_count and the six history arrays) does not separate them;
#: both are the solver stopping itself, so both are ``converged``.
GSCO_STOPPED_WITHOUT_ACCEPTING_A_LOOP: Final = "gsco_stopped_without_accepting_a_loop"
#: ``stop_last_iter``: the declared ``max_iter`` budget ran out.
GSCO_MAXIMUM_ITERATION_REACHED: Final = "gsco_maximum_iteration_reached"
#: An early stop that adopted no loop at all: the returned currents are the
#: ones the caller passed in, so the solve is degenerate, not converged.
GSCO_NO_ACCEPTED_UPDATE: Final = "gsco_stopped_without_accepting_any_update"
#: One staged GSCO call of the multistep workflow reached its allocated budget.
GSCO_MULTISTEP_STAGE_BUDGET_REACHED: Final = "gsco_multistep_stage_budget_reached"
#: Every staged call stopped itself and the final adjustment ran.
GSCO_MULTISTEP_STABLE_CURRENTS: Final = "stable_current_final_adjustment_complete"
#: The staged loop ended with a usable endpoint that no stage improved.
GSCO_MULTISTEP_NO_ACCEPTED_UPDATE: Final = (
    "gsco_multistep_first_stage_accepted_no_update"
)

#: Scales whose GSCO iteration cap is the campaign's reduced one rather than
#: the source script's. Upstream GSCO accepts an explicit ``max_iter`` cap
#: (``wireframe_optimization.cpp:281`` ``stop_last_iter``) and the official
#: ``9e027eac3`` runs stop earlier on their own rule -- 1848 of 2000 accepted
#: updates for the modular example, 1698 of 2000 for the sector-saddle one
#: (accepted iterations against the allocated iteration budget) -- so only the
#: reduced cap is ever reached, and only there is a budget stop an expected
#: outcome rather than a failure to mirror the source.
GSCO_REDUCED_BUDGET_SCALES: Final[tuple[ExecutionScale, ...]] = ("bounded",)


@dataclass(frozen=True)
class LaneTerminalLabel:
    """One lane's normalized category, raw provider reason and success flag."""

    normalized_status: str
    raw_status: str
    success: bool


def _accepted_undo_pair(
    loop_history: NDArray[np.int64],
    current_history: NDArray[np.float64],
) -> bool:
    """True when the last two recorded loops are the same loop in opposite signs.

    ``record_iter(0, 0.0, 0, ...)`` seeds the history before the greedy loop
    (``wireframe_optimization.cpp:165-171``), so accepted updates are rows
    ``1..hist_ind``. ``opt_ind_prev`` is always the previous iteration's index
    -- the two early stops break immediately -- so upstream's
    ``(opt_ind + nLoops % (twoNLoops)) == opt_ind_prev`` test with the undo
    accepted is "row ``n`` repeats row ``n-1``'s loop with the opposite
    current"; upstream's own recorded stages end in such pairs. The caller must
    therefore record EVERY iteration: a sampled history cannot show the pair,
    and the label then reports the coarser "stopped without accepting a loop"
    (both are ``converged``).
    """

    if loop_history.shape[0] < 3:
        return False
    return bool(
        loop_history[-1] == loop_history[-2]
        and current_history[-1] * current_history[-2] < 0.0
    )


def gsco_terminal_label(
    *,
    accepted_updates: int,
    max_iterations: int,
    loop_history: NDArray[np.int64],
    current_history: NDArray[np.float64],
    endpoint_usable: bool,
) -> LaneTerminalLabel:
    """Label one GSCO solve from the arrays the solver returned.

    ``accepted_updates`` is ``iter_history[-1]``, the number of loops the
    solver actually adopted; it is a measured count, not the configured cap.
    """

    if accepted_updates < 1:
        # Degenerate: the solver stopped before adopting a single loop, so the
        # returned currents are the ones it was given. "The solver stopped
        # itself" is not evidence that it did anything (completion policy: a
        # demoted gate still requires a non-degenerate result).
        return LaneTerminalLabel(
            normalized_status="failed",
            raw_status=GSCO_NO_ACCEPTED_UPDATE,
            success=False,
        )
    if _accepted_undo_pair(loop_history, current_history):
        raw_status = GSCO_MINIMUM_OBJECTIVE_REACHED
        budget_stop = False
    elif accepted_updates >= max_iterations:
        raw_status = GSCO_MAXIMUM_ITERATION_REACHED
        budget_stop = True
    else:
        raw_status = GSCO_STOPPED_WITHOUT_ACCEPTING_A_LOOP
        budget_stop = False
    if not endpoint_usable:
        return LaneTerminalLabel(
            normalized_status="failed",
            raw_status=raw_status,
            success=False,
        )
    if budget_stop:
        return LaneTerminalLabel(
            normalized_status="budget_exhausted",
            raw_status=raw_status,
            success=False,
        )
    return LaneTerminalLabel(
        normalized_status="converged",
        raw_status=raw_status,
        success=True,
    )


def gsco_multistep_terminal_label(
    *,
    stage_iterations: NDArray[np.int64],
    stage_budget: NDArray[np.int64],
    final_adjustment_run: bool,
    endpoint_usable: bool,
    limit_raw_status: str,
) -> LaneTerminalLabel:
    """Label the staged GSCO workflow from the per-stage counts it publishes.

    Each stage is one ``gsco_wireframe`` call, so the same rule as
    ``gsco_terminal_label`` applies stage by stage: a stage whose accepted
    updates reached its allocated ``max_iter`` is upstream's ``stop_last_iter``
    -- a budget stop -- and a lane that contains one is never ``converged``.
    A first stage that adopted no loop moved nothing at all. The staged loop
    that never reached its final adjustment is reported with the caller's
    ``limit_raw_status``, which names which limit ended it.
    """

    iterations = np.asarray(stage_iterations, dtype=np.int64).reshape(-1)
    budget = np.asarray(stage_budget, dtype=np.int64).reshape(-1)
    if iterations.shape != budget.shape:
        raise ValueError("every recorded stage publishes one iteration budget")
    if iterations.size < 1:
        raise ValueError("the staged workflow recorded no stage")
    if not final_adjustment_run or not endpoint_usable:
        return LaneTerminalLabel(
            normalized_status="failed",
            raw_status=(
                limit_raw_status
                if not final_adjustment_run
                else "failed_endpoint_after_final_adjustment"
            ),
            success=False,
        )
    if int(iterations[0]) < 1:
        return LaneTerminalLabel(
            normalized_status="failed",
            raw_status=GSCO_MULTISTEP_NO_ACCEPTED_UPDATE,
            success=False,
        )
    if bool(np.any(iterations >= budget)):
        return LaneTerminalLabel(
            normalized_status="budget_exhausted",
            raw_status=GSCO_MULTISTEP_STAGE_BUDGET_REACHED,
            success=False,
        )
    return LaneTerminalLabel(
        normalized_status="converged",
        raw_status=GSCO_MULTISTEP_STABLE_CURRENTS,
        success=True,
    )


def gsco_example_status(
    label: LaneTerminalLabel,
    scale: ExecutionScale,
    *,
    budget_admitted_scales: tuple[ExecutionScale, ...] = (),
) -> Literal["ok", "failed"]:
    """Map one GSCO label onto the two-valued status an example publishes.

    ``ExampleResult.status`` says whether the example executed its workflow and
    has no third value for "ran out of the budget it was given". A solver that
    stopped itself is ``ok``; a solver that hit ITS SOURCE's cap did not mirror
    the source and is ``failed``; a solver that hit a cap the campaign reduced
    ran exactly the work that reduced scale asked of it, and the caller admits
    that by naming the scale in ``budget_admitted_scales`` -- an opt-in per
    workflow, empty by default. The distinction stays visible in the published ``terminal_status``
    and ``terminal_reason``, and ``solver_success`` keeps the strict meaning
    (converged only).
    """

    if label.success:
        return "ok"
    if (
        label.normalized_status == "budget_exhausted"
        and scale in budget_admitted_scales
    ):
        return "ok"
    return "failed"
