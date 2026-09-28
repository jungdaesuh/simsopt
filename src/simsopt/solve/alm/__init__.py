"""Augmented Lagrangian (ALM) solver for ``min f(x)`` s.t. ``g_i(x) <= 0``.

:func:`minimize_alm` calls ``evaluate_problem(x, multipliers, penalty) -> dict``.
The dict must contain ``total`` and ``grad`` (the augmented Lagrangian L and its
gradient), ``constraint_values`` (signed g, feasible when <= 0),
``feasibility_values`` (per-row violation), ``dual_update_values`` (the g the
multiplier update uses), and ``constraint_grads`` (one gradient per row).
``augmented_inequality_objective(f, grad_f, g, grad_g, multipliers, penalty)``
returns all of them, so an evaluator may return its output unchanged. The
typed forms are :class:`ALMEvaluation` (the dict), :class:`ALMEvaluator` (the
callable) and :class:`CachedALMEvaluator` (what :func:`cached_alm_evaluator`
returns, with ``cache_clear()``).

Optional keys: ``base_value`` / ``base_total`` / ``physics_total`` (objective
without penalty terms, used to rank incumbents), ``base_grad`` / ``metric_grad``
(its gradient, for stationarity), ``constraint_scales``,
``constraint_activity_tolerances``, ``nonfinite_evaluation`` (the point is
unusable), and ``search_step_success=False`` (reject the trial step).
:class:`ALMEvaluation` declares these and the other keys the solver owns (the
builder's summaries, raw and normalized rows); an application types its own
keys by subclassing it. Any other key is kept as a diagnostic. The solver copies the mappings, lists and
tuples in an evaluation, so none may contain itself (a ``ValueError`` names
the path; a shared subtree is fine); other objects pass by reference.

Hybrid signals: an evaluator that smooths g returns all four of
``hard_signed_constraint_values``, ``hard_violation_values``,
``surrogate_signed_constraint_values``, ``hard_dual_update_values``, or none.
L uses the surrogate g, the dual update uses the hard g, and a disagreement
blocks ``success``. A missing member raises ``KeyError``. What a step does
under a disagreement is in :mod:`simsopt.solve.alm.hybrid`, which no evaluator
without the quartet reaches.

Stateful evaluators (e.g. warm-started inner solves) pass both
``snapshot_accepted_state_fn() -> state`` and ``restore_incumbent_state_fn(state)``;
the solver snapshots accepted iterates and restores one when it returns a
best-feasible incumbent or resumes. The result is a frozen :class:`ALMResult`:
x, objective, signed g, max violation, multipliers, penalty, success,
termination reason and message, stationarity norms, iteration counts, the
restore flag and reason, the final evaluation and the last inner result.

A toy problem, min ||x||^2 s.t. 1 - x0 <= 0 (solution x = (1, 0))::

    import numpy as np
    from simsopt.solve.alm import (ALMPhysics, ALMSettings, cached_alm_evaluator,
                                   minimize_alm)

    def physics(x):
        return ALMPhysics(
            base_value=float(x @ x),
            base_grad=2.0 * x,
            constraint_values=np.array([1.0 - x[0]]),
            constraint_grads=(np.array([-1.0, 0.0]),),
        )

    result = minimize_alm(np.array([3.0, 2.0]), ["x0_at_least_one"],
                          cached_alm_evaluator(physics), ALMSettings(),
                          {"maxiter": 200})
    print(result.termination_reason, result.x.round(4) + 0.0)  # converged [1. 0.]

For simsopt objectives, :func:`evaluate_alm_problem` builds that dict from a
base Optimizable and signed constraint rows (see :mod:`.problem`).

Physics reuse: the solver evaluates the same x again with new multipliers or a
new penalty (the outer iterate, then L-BFGS-B's first call; after a dual or
penalty update). An evaluator that returns an :class:`ALMPhysics` (f, grad f,
g, grad g and x-only ``extras``) for x can be wrapped as
``cached_alm_evaluator(physics_at_x)``, which recomputes only the augmented
terms at a revisited x, bit for bit (:func:`alm_problem_physics` is such an
evaluator). Call its ``cache_clear()`` whenever anything besides x that the
physics reads changes (e.g. a smoothing width). A stateful evaluator, whose
second evaluation at the same x can differ (warm-started inner solves), must
not use it.

Outer-step events: ``on_outer_step(event)`` runs once per continuation step,
when its decision is final and before any terminal result or best-feasible
restore. :class:`ALMOuterStepEvent` holds owned, read-only copies (arrays
not writable, mappings read-only views); its field types
:class:`ALMIterateMeasurement`, :class:`ALMInnerSolveOutcome`,
:class:`ALMDualUpdate`, :class:`ALMLoopState`, :class:`ALMFeasibleIncumbent`,
:class:`ALMConstraintRoutingState` and :class:`ALMConstraintSignalState` are
exported too. Events cannot be pickled or deep-copied; store the fields you
need. It holds:

- ``outer_iteration``, ``continuation_iteration``, ``constraint_names``.
- The decision: ``action`` (e.g. ``dual_update``, ``penalty_increase``,
  ``subproblem_continue``, ``converged``, ``penalty_cap_reached``),
  ``outer_termination`` (``max_outer`` / ``process_budget_exhausted`` or
  None), ``subproblem_limit_reason``, ``signal_mismatch_repair``,
  ``feasible_stall_count``.
- The iterate: ``start_x`` and ``start`` (the outer iterate measured before
  the inner solve), ``inner`` (None when the start already met the KKT gate;
  else the accepted ``x`` and ``measured`` iterate, iteration and attempt
  counts, optimizer status, last box and options, stall flags, progress,
  feasibility delta and the ALGENCAN sufficient-decrease measure and
  reference), and the property ``measured``. A measurement carries the
  evaluation, the multipliers and penalty it used, g, feasibility, routing
  state, stationarity norms and the tolerances it was judged with.
- Transitions: ``dual_update`` (projected multipliers, cap binding),
  ``penalty_update`` (the re-evaluation at a raised penalty) and
  ``dual_update_penalty_reason``.
- ``after``: the loop state after the decision (x, multipliers, penalty,
  tolerances, trust radius, best-feasible incumbent without its inner result,
  counters, cap flags and the sufficient-decrease carrier), the content of a
  checkpoint. Accepted states inside it are the caller's own objects.

History: the opt-in ``simsopt.solve.alm.history.ALMHistoryRecorder`` builds
one entry per event (``recorder.record`` as ``on_outer_step``, then
``recorder.history()``, owned copies); ``ALMHistoryRecorder.from_settings``
keeps ``settings.history_max_entries`` of them.

Outer boundaries: ``on_outer_boundary(boundary)`` runs after each completed
outer iteration and once when the run returns. :class:`ALMOuterBoundary` holds
the completed outer count and action, the termination reason (None while the
run can continue), the constraint names and blocks, the caller's accepted state
at that x (``snapshot_accepted_state_fn()``, called only when a boundary is
built), and ``state``, an :class:`ALMLoopState`. A resumable boundary restarts
the run as ``minimize_alm(resume_from=boundary)`` with the same x0, names and
blocks. The opt-in ``simsopt.solve.alm.checkpoint`` module turns boundaries
into ``ALMTransitionSnapshot`` checkpoints and back
(``alm_checkpointing(inner_options, resume_state, completed_outer_callback)``
returns the three ``minimize_alm`` arguments).

Continuation policy: the loop measures, solves and executes; an
``ALMContinuationPolicy`` (:mod:`simsopt.solve.alm.policy`) decides each step
(inner-solve options, stalled-trial retries, convergence, dual update,
penalty raise or hold, another subproblem, stop) from a read-only view of the
measurements (:mod:`simsopt.solve.alm.continuation`).
``minimize_alm(continuation_policy=...)`` defaults to
``DefaultContinuationPolicy``. The loop executes a policy's convergence
decision as given, so the success guarantees hold for
``DefaultContinuationPolicy`` and for policies that keep its vetoes:
``converged`` means an approximate KKT point at the shifted multipliers
``max(0, multipliers + penalty * g)`` (violations within ``feasibility_tol``,
augmented-gradient norm within ``stationarity_tol``, complementarity gap
``sum(shift * max(0, -g))`` within ``feasibility_tol * max(1, |f|)``), with no
hybrid disagreement and no binding multiplier cap.
"""

from __future__ import annotations

from .continuation import ALMInnerSolveOutcome, ALMIterateMeasurement
from .boundary import ALMOuterBoundary
from .control import ALMResult, minimize_alm
from .core import (
    ALMConstraintRoutingState,
    ALMConstraintSignalState,
    ALMSettings,
    augmented_inequality_objective,
    normalize_alm_constraints,
)
from .evaluation import ALMEvaluation, ALMEvaluator
from .events import (
    ALMDualUpdate,
    ALMFeasibleIncumbent,
    ALMLoopState,
    ALMOuterStepEvent,
)
from .problem import (
    ALMPhysics,
    CachedALMEvaluator,
    alm_problem_physics,
    cached_alm_evaluator,
    evaluate_alm_problem,
    signed_lower_bound,
    signed_upper_bound,
)
from .taylor import run_directional_taylor_test

__all__ = [
    "ALMConstraintRoutingState",
    "ALMConstraintSignalState",
    "ALMDualUpdate",
    "ALMEvaluation",
    "ALMEvaluator",
    "ALMFeasibleIncumbent",
    "ALMInnerSolveOutcome",
    "ALMIterateMeasurement",
    "ALMLoopState",
    "ALMOuterBoundary",
    "ALMOuterStepEvent",
    "ALMPhysics",
    "ALMResult",
    "ALMSettings",
    "CachedALMEvaluator",
    "alm_problem_physics",
    "augmented_inequality_objective",
    "cached_alm_evaluator",
    "evaluate_alm_problem",
    "minimize_alm",
    "normalize_alm_constraints",
    "run_directional_taylor_test",
    "signed_lower_bound",
    "signed_upper_bound",
]
