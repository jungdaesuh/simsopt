"""VMEC-free single-stage optimization by the projected full-space route.

This is the same physics as ``3_Advanced/single_stage_boozer_vacuum_optimization``
-- the NCSX coil set, an implicit Boozer surface at fixed volume, and the
non-quasisymmetry / Boozer-residual / iota / major-radius / length objective --
solved by a different formulation.

The mirror example nests the two problems: an outer optimizer moves coil degrees
of freedom, and every outer evaluation runs an inner Newton solve to put the
surface back on the Boozer manifold.  This script COUPLES them.  Coil dofs,
surface dofs, the rotational transform and the net poloidal current are one
state vector, the Boozer residual and the volume label are exact equality
CONSTRAINTS on that vector, and the route walks the constraint manifold
directly:

* a certified Gram factorization projects the objective gradient into the
  tangent space and supplies the least-squares multipliers,
* a truncated Newton-CG solve on the projected LAGRANGIAN curvature
  ``P (grad2 Phi + sum_i lambda_i grad2 c_i) P`` gives the step -- the
  constraint-curvature term is exactly what an objective-only model omits,
* an Armijo line search on the retracted objective decides how far to go, and
* an exact retraction restores feasibility at every accepted iterate, so the
  raw equality residuals never leave 1e-10.

There is no trust region and no penalty: the constraints are enforced, not
priced.  The loop runs on the host, one jitted kernel per phase, so nothing in
the compiled program depends on the iteration budget and a two-step run compiles
the same executables a seven-hundred-step run does.  The endpoint evaluations
this script publishes run through a jitted kernel for the same reason the loop
does: the host boundary is crossed deliberately and in one place, which is what
lets the strict transfer-guard lane execute this lesson.

REACHING THE TARGET IS A DRAW, NOT A CERTAINTY
----------------------------------------------
The route's step is proposed from a projector whose constraint rows may be
several iterations old.  Where that carry goes stale the line search can refuse
every trial it is allowed -- the retraction hits its eight-correction cap and
the step scale falls through the 1e-6 floor -- and the run stops at
``LINE_SEARCH_COLLAPSE`` short of the objective target.  That outcome is a
property of the DRAW, not of the configuration: measured on an A100, one of
three certified-budget attempts collapsed (at iteration 293, objective
8.96e-8), the other two latched; on an RTX 5090 two of two latched.  Roughly
one attempt in five, and the two certified GPUs disagree about which.

So a collapse is not reported as success.  This script mirrors the
certification root's pre-registered protocol at example scale: up to
``ATTEMPT_BUDGET`` attempts, a collapse draw is retried by the next attempt,
and the run publishes ``protocol_verdict`` --

* ``ok`` -- an attempt stopped soundly and did not collapse; exit code 0.
* ``retry_exhausted`` -- every attempt was a collapse and none of them carried
  a second defect; exit code 1.
* ``unsound_terminal_state`` -- an attempt ended in a state that is a defect
  rather than a draw (a nonfinite state, a non-descent direction, an infeasible
  start), or left the manifold, or let the objective rise.  Such a stop is NOT
  retried: it is reproducible, and a retry would only hide it.  Exit code 1.

The retries are a GPU protocol.  A CPU run of this script is deterministic, so
three CPU attempts reproduce one another exactly and the budget costs nothing
unless the first attempt collapses -- in which case the honest report is that
it collapsed three times.

``--smoke`` runs the SAME problem for two iterations; it is a bounded diagnostic
lane, and its ``ok`` status means the route stayed feasible and descended, never
that it converged.  The route's configuration below is the one certified in
``docs/single_stage_jax_gpu_projected_route_certification_plan.md``; the
certification chain owns the frozen copy and
``tests/jax/examples/test_single_stage_boozer_vacuum_projected_route_example.py``
pins this one to it so the two cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import jax
import numpy as np
from simsopt_jax.backend.dtypes import explicit_device_array
from simsopt_jax.examples import ExampleResult, ExecutionScale, run_example
from simsopt_jax.geo.optimizers.projected_lbfgs import (
    ProjectedLbfgsOptions,
    ProjectedLbfgsRun,
    ProjectedLbfgsStatus,
    build_projected_lbfgs_kernels,
    run_projected_lbfgs,
)
from simsopt_jax.objectives.single_stage_fullspace import (
    evaluate_fullspace,
    flatten_fullspace_constraints,
)
from simsopt_jax.runtime.host_boundary import host_array, host_float, host_tree
from simsopt_jax.solve.fullspace import (
    fullspace_optimizer_coordinates,
    fullspace_physical_coordinates,
    fullspace_scaling_from_bootstrap,
)
from simsopt_jax_adapters.geo.single_stage_fullspace import (
    build_single_stage_fullspace_bootstrap,
)

EXAMPLE_ID = "projected-route-single-stage-boozer-vacuum-optimization"

# The budget the certified route is configured with.  It is smaller than the
# native BFGS reference's because the coupled formulation reaches the reference's
# endpoint objective in roughly four hundred iterations, not a thousand.
PROJECTED_NATIVE_ITERATIONS = 700

# The certification root pre-registers three attempts; this lesson mirrors that
# number rather than inventing one, so a reader who follows the plan document
# finds the same protocol here.
ATTEMPT_BUDGET = 3

ROUTE_OPTIONS = ProjectedLbfgsOptions(
    maximum_iterations=PROJECTED_NATIVE_ITERATIONS,
    feasibility_tolerance=1.0e-10,
    objective_target=4.4822246533126125e-08,
    lagrangian_newton=True,
    frozen_projector_line_search=True,
    newton_tangent_fraction_threshold=0.25,
    projector_refresh_period=4,
)

# Terminal states a SOUND attempt can carry.  ``LINE_SEARCH_COLLAPSE`` is
# deliberately absent: it is the draw the attempt budget exists to redraw, and
# a script that called it sound would report a run that missed its target as a
# run that met it.
_SOUND_TERMINAL_STATES = (
    ProjectedLbfgsStatus.ITERATION_LIMIT,
    ProjectedLbfgsStatus.OBJECTIVE_TARGET_REACHED,
)

VERDICT_OK = "ok"
VERDICT_RETRY_EXHAUSTED = "retry_exhausted"
VERDICT_UNSOUND = "unsound_terminal_state"


class Attempt(NamedTuple):
    """One draw of the route and every fact the protocol judges it on.

    The two measurements soundness is built from are carried rather than
    recomputed downstream, so what the run publishes and what the protocol
    decided on are one set of numbers.  ``collapse_draw`` is the narrower of
    the two dispositions the protocol acts on: a collapse whose OTHER
    measurements held, and therefore the only outcome a redraw may replace.
    """

    run: ProjectedLbfgsRun
    status: ProjectedLbfgsStatus
    monotone_descent: bool
    worst_feasibility_inf: float
    sound: bool
    collapse_draw: bool


def selected_kernel_names(joint_value_constraints, options: ProjectedLbfgsOptions):
    """Name the executables this configuration selects, compiling none of them.

    ``build_projected_lbfgs_kernels`` is the same construction the loop drives,
    so what this reports is what a run compiles rather than a re-spelled list
    that can drift away from it.
    """

    kernels = build_projected_lbfgs_kernels(joint_value_constraints, options=options)
    return tuple(
        name
        for name in kernels._fields
        if name != "frozen_projector_line_search"
        and getattr(kernels, name) is not None
    )


def judge_attempt(run: ProjectedLbfgsRun, options: ProjectedLbfgsOptions) -> Attempt:
    """Classify one attempt: its terminal state, and whether it was sound.

    Soundness is the whole of what this lesson claims about a bounded run --
    the route stopped for a reason that is not a defect, every recorded iterate
    stayed on the constraint manifold, and the objective never went up.  The
    last two measurements are read once here and then decide BOTH dispositions:
    a collapse that also left the manifold or let the objective rise carries a
    defect, so it is not a draw and the budget may not spend an attempt hiding
    it behind a later success.
    """

    status = ProjectedLbfgsStatus(int(run.status))
    objectives = [record.objective for record in run.iterations]
    worst_feasibility = max(
        (record.feasibility_inf for record in run.iterations),
        default=run.feasibility_inf,
    )
    monotone = all(
        later <= earlier
        for earlier, later in zip(objectives, objectives[1:], strict=False)
    )
    feasible = worst_feasibility <= options.feasibility_tolerance
    return Attempt(
        run=run,
        status=status,
        monotone_descent=monotone,
        worst_feasibility_inf=float(worst_feasibility),
        sound=status in _SOUND_TERMINAL_STATES and feasible and monotone,
        collapse_draw=(
            status is ProjectedLbfgsStatus.LINE_SEARCH_COLLAPSE
            and feasible
            and monotone
        ),
    )


def protocol_verdict(attempts: list[Attempt]) -> str:
    """The protocol's disposition over the attempts that were actually spent.

    ``retry_exhausted`` is reserved for a budget spent entirely on draws: one
    attempt carrying a defect makes the whole spend an unsound stop, whichever
    terminal state it wore.
    """

    if attempts[-1].sound:
        return VERDICT_OK
    if all(attempt.collapse_draw for attempt in attempts):
        return VERDICT_RETRY_EXHAUSTED
    return VERDICT_UNSOUND


def draw_attempts(
    draw: Callable[[], ProjectedLbfgsRun], options: ProjectedLbfgsOptions
) -> list[Attempt]:
    """Spend draws until one is not a collapse draw, or the budget is gone.

    A collapse whose other measurements held is the stochastic outcome the
    budget exists for, so it -- and only it -- is redrawn.  Every other
    outcome is reproducible, so a redraw would spend the budget to reach the
    same place and report the third copy of one defect as if it were three.
    """

    attempts: list[Attempt] = []
    while len(attempts) < ATTEMPT_BUDGET:
        attempts.append(judge_attempt(draw(), options))
        if not attempts[-1].collapse_draw:
            break
    return attempts


def published_status(verdict: str) -> str:
    """Map the protocol's three verdicts onto the example contract's two.

    The example contract publishes ``ok`` or ``failed`` and the process exit
    code follows it, so both ways of missing the target -- an exhausted retry
    budget and an unsound stop -- publish ``failed`` and exit nonzero, with
    ``protocol_verdict`` naming which one it was.
    """

    return "ok" if verdict == VERDICT_OK else "failed"


def solve(
    _output_directory: Path, max_steps: int, scale: ExecutionScale
) -> ExampleResult:
    bootstrap = build_single_stage_fullspace_bootstrap()
    # The problem and the scaling are carried to the host ONCE, at explicit
    # boundaries, and every jitted kernel below closes over those host copies.
    # A kernel that closes over DEVICE arrays has to read them back to embed
    # them as constants, which is a device-to-host transfer at TRACE time --
    # invisible on a CPU backend, refused on the GPU-strict lane.  Nothing is
    # recomputed: these are the bootstrap's own values, copied.
    problem = host_tree(bootstrap.problem)
    scaling = host_tree(
        jax.jit(lambda state: fullspace_scaling_from_bootstrap(state, problem))(
            bootstrap.z0
        )
    )
    anchor = host_array(bootstrap.z0, dtype=np.float64)

    def raw_joint(coordinates: jax.Array) -> tuple[jax.Array, jax.Array]:
        """The weighted objective and the RAW equality residuals at one point.

        Raw, not scaled: the optimizer's own feasibility tolerance is then the
        physical 1e-10 on the Boozer residual and the volume label rather than a
        proxy for it.
        """

        physical = fullspace_physical_coordinates(coordinates, scaling)
        evaluation = evaluate_fullspace(physical, problem)
        return (
            evaluation.weighted_total,
            flatten_fullspace_constraints(evaluation.constraints),
        )

    @jax.jit
    def endpoint(coordinates: jax.Array):
        """The physical state and the full evaluation at one joint coordinate."""

        physical = fullspace_physical_coordinates(coordinates, scaling)
        return physical, evaluate_fullspace(physical, problem)

    options = replace(ROUTE_OPTIONS, maximum_iterations=max_steps)
    start = explicit_device_array(
        fullspace_optimizer_coordinates(anchor, scaling), dtype=np.float64
    )
    kernels = selected_kernel_names(raw_joint, options)
    initial_objective = host_float(endpoint(start)[1].weighted_total)

    # The bootstrap, the scaling and the compiled kernels are shared across
    # attempts; only the solve is redrawn.
    attempts = draw_attempts(
        lambda: run_projected_lbfgs(raw_joint, start, options=options), options
    )
    verdict = protocol_verdict(attempts)

    reported = attempts[-1]
    run = reported.run
    terminal_physical, terminal = endpoint(run.coordinates)
    state = host_array(terminal_physical, dtype=np.float64)

    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "scale": scale,
            "route": "projected-lagrangian-newton-cg",
            "selected_kernels": kernels,
            "joint_dof_count": int(start.shape[0]),
            "equality_count": int(problem.exact_mask_indices.shape[0] + 1),
            "maximum_iterations": options.maximum_iterations,
            "attempt_budget": ATTEMPT_BUDGET,
            "attempts_spent": len(attempts),
            "attempt_terminal_states": tuple(
                attempt.status.name for attempt in attempts
            ),
            "protocol_verdict": verdict,
            "initial_objective": initial_objective,
            "final_objective": run.objective,
            "objective_target": options.objective_target,
            "iterations_run": len(run.iterations),
            "terminal_status": reported.status.name,
            "monotone_descent": reported.monotone_descent,
            "feasibility_inf": run.feasibility_inf,
            "maximum_feasibility_inf": reported.worst_feasibility_inf,
            "feasibility_tolerance": options.feasibility_tolerance,
            "projected_gradient_inf": run.projected_gradient_inf,
            "projector_materializations": run.projector_materializations,
            "engine_compile_seconds": run.compile_seconds,
            "engine_solve_seconds": run.solve_seconds,
            "iota": host_float(terminal.observables.iota),
            "G": host_float(terminal.observables.G),
            "volume": host_float(terminal.observables.volume),
            "major_radius": host_float(terminal.observables.major_radius),
            "total_length": host_float(terminal.observables.total_length),
            "non_qs_ratio": host_float(terminal.observables.non_qs_ratio),
            "boozer_residual": host_float(terminal.observables.boozer_residual_scalar),
            "boozer_residual_rms": host_float(terminal.observables.boozer_residual_rms),
            "solution": tuple(float(value) for value in state),
        },
        status=published_status(verdict),
    )


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-projected-single-stage-",
        bounded_steps=2,
        native_default_steps=PROJECTED_NATIVE_ITERATIONS,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
