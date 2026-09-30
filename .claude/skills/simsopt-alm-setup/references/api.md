# API

The names the skill uses, where they live, and the contracts the generated
files follow. Full docstrings: `python -c "import simsopt_alm as a; help(a)"`
and the help of each module named below.

## Names

| Name | Module | Use |
|---|---|---|
| `minimize_alm` | `simsopt_alm` | The solver: `minimize_alm(x0, constraint_names, evaluate_problem, settings, inner_options, **optional)` returns an `ALMResult`. |
| `ALMSettings` | `simsopt_alm` | Frozen, validated solver settings ([settings.md](settings.md)). |
| `ALMResult` | `simsopt_alm` | Frozen result; fields below. |
| `ALMPhysics` | `simsopt_alm` | f, grad f, g, grad g (and x-only `extras`) at one x; `.evaluation(multipliers, penalty)` builds the evaluator dict. |
| `cached_alm_evaluator` | `simsopt_alm` | Wraps `physics(x) -> ALMPhysics` into an evaluator that reuses the physics at a revisited x. Stateless physics only. |
| `CachedALMEvaluator` | `simsopt_alm` | The type `cached_alm_evaluator` returns (has `cache_clear()`). |
| `alm_problem_physics` | `simsopt_alm` | `alm_problem_physics(dofs, base_objective, inequalities) -> ALMPhysics` for a simsopt Optimizable and a list of rows. |
| `evaluate_alm_problem` | `simsopt_alm` | The uncached evaluator form of `alm_problem_physics`. |
| `signed_upper_bound` | `simsopt_alm` | Row `objective.J() - bound` (with its gradient over the base objective's free dofs). |
| `signed_lower_bound` | `simsopt_alm` | Row `bound - objective.J()`. |
| `augmented_inequality_objective` | `simsopt_alm` | Builds the evaluator dict from f, grad f, g, grad g, multipliers, penalty (what `ALMPhysics.evaluation` calls). |
| `run_directional_taylor_test` | `simsopt_alm` | Central differences of an evaluator's `total` along a direction, with the claimed directional derivative (`gradient_check.py` sweeps its steps per quantity). Its `status` is `passed`, `failed` (a ratio above the threshold) or `unavailable` (nonfinite values or no ratio checked); two or more decreasing positive steps. |
| `ALMEvaluation` | `simsopt_alm` | TypedDict of the evaluator dict (required and optional keys). |
| `ALMEvaluator` | `simsopt_alm` | Protocol `(x, multipliers, penalty) -> ALMEvaluation`. |
| `ALMOuterStepEvent` | `simsopt_alm` | What `on_outer_step` receives once per continuation step. |
| `ALMOuterBoundary` | `simsopt_alm` | What `on_outer_boundary` receives after each outer iteration; `resume_from` takes a non-final one. |
| `ALMHistoryRecorder` | `simsopt_alm.history` | Opt-in history: pass `recorder.record` as `on_outer_step`, read `recorder.history()`. |
| `alm_checkpointing` | `simsopt_alm.checkpoint` | `alm_checkpointing(inner_options, resume_state, completed_outer_callback)` returns the `inner_options`, `resume_from` and `on_outer_boundary` arguments of one checkpointed run. |
| `ALMCheckpointing` | `simsopt_alm.checkpoint` | The frozen triple `alm_checkpointing` returns. |
| `ALMTransitionSnapshot` | `simsopt_alm.checkpoint` | Immutable checkpoint (no Jacobians): `x`, `total_inner_iterations`, `completed_outer_iterations`, `resume_eligible`, `accepted_state`, ... |
| `DefaultContinuationPolicy` | `simsopt_alm.policy` | The default `continuation_policy`; its vetoes give the success guarantees. |
| `ALMContinuationPolicy` | `simsopt_alm.policy` | Protocol of a custom policy (advanced; keep the default's convergence vetoes). |
| `ALMProcessBudgetExhausted` | `simsopt_alm.control` | Raise it from `accepted_callback` to stop with `process_budget_exhausted` (your own budget, e.g. wall clock). |
| `smooth_min_curve_curve_signed_constraint` | `simsopt_alm.signed_constraints` | Row `minimum_distance - min dist(curve_i, curve_j)`, smooth; returns `(signed, grad, hard_signed)`. |
| `smooth_min_curve_surface_signed_constraint` | `simsopt_alm.signed_constraints` | Row `minimum_distance - min dist(curve_i, surface)`, smooth. |
| `smooth_max_curvature_signed_constraint` | `simsopt_alm.signed_constraints` | Row `max(kappa) - threshold` for one curve, smooth. |

## Rows

A row is feasible when `g <= 0`. The solver has no equality rows: write
`q = target` as the pair `target - q <= 0` and `q - target <= 0` (both rows
are then active), or better a band `target - w <= q <= target + w`.

For a simsopt Optimizable `base_objective` (f), a row is a callable
`row(base_objective) -> (signed_value, grad, ...)` evaluated at the dofs just
set, whose `grad` spans `base_objective`'s free dofs; trailing items are
ignored by `alm_problem_physics`. Rows:

- `functools.partial(signed_upper_bound, objective, bound)`: `objective <= bound`.
- `functools.partial(signed_lower_bound, objective, bound)`: `objective >= bound`.
- A `simsopt_alm.signed_constraints` kernel with its leading arguments bound,
  e.g. `partial(smooth_min_curve_curve_signed_constraint, curves, 0.1, 0.005)`
  (curves, minimum distance in m, smoothing temperature in m). The third
  item is the hard value: the extremum over the sampled quadrature points of
  the curves (and surface). The smooth value is never looser than it, so
  smooth-feasible implies feasible over those samples, not for the
  continuous coils: they can come closer or bend more between samples, so
  check clearance and curvature at a higher resolution (e.g. re-evaluate the
  rows on copies of the curves with more quadrature points) before accepting
  a physical bound. The temperature is in the constrained quantity's units
  and must lie in [1e-100, 1e100], and every sample coordinate must be
  finite with |x| <= 1e100 (`ValueError` otherwise, naming the bound and the
  value, also for a row with no pair; 0 is rejected: use the third item for
  the hard value). Accuracy: a distance row's value has absolute error at
  most a small multiple of machine epsilon times its scale, max(bound, the
  largest sampled distance, T log N), and its gradient at most a small
  multiple of epsilon times scale / T; a value much smaller than its scale
  is a difference of nearly equal numbers and is not resolved. The curvature
  row is the soft maximum of simsopt's `curve.kappa()`, so its accuracy is
  simsopt's; a nonfinite curvature or curvature derivative raises
  `ValueError` naming the curve.

Divide every row by a positive scale in its units (its bound, or a typical
size) so all rows are O(1): the penalty is shared, and the tolerances are
absolute. The templates use `scaled_row(row, scale, base_objective)`.

## Evaluators

`minimize_alm` calls `evaluate_problem(x, multipliers, penalty) -> dict` and
re-evaluates the same x with new multipliers or a new penalty. The dict needs
`total`, `grad`, `constraint_values`, `feasibility_values`,
`dual_update_values`, `constraint_grads`; `ALMPhysics.evaluation` and
`augmented_inequality_objective` return all of them, plus `base_value` and
`base_grad` (f without penalty terms, used to rank incumbents and for the KKT
residual). An optional `stationarity_norm` that is not `||grad||` is the
evaluator's own stationarity measure, used as given (so it must account for
`base_bounds`); otherwise the solver measures `||grad||` without the
components pointing out of the box at an active bound. The solver checks every dict where it enters (outer iterate and
each inner trial): a required key absent or None raises `KeyError`
(`constraint_grads` included, even with every row inactive); `grad` and each
of the one-per-row `constraint_grads` must have x's shape, and
`constraint_values` one entry per row (`ValueError`), and the optional
`constraint_activity_tolerances` (activity bands) nonnegative (`ValueError`).
Bands route rows and feed diagnostics only; the convergence test's
complementarity gap uses each row's actual slack
([termination.md](termination.md)). The optional
`search_step_success` (False rejects the trial step; absent means accepted)
must be a `bool` or `numpy.bool_`: 0, None or any other type raises
`ValueError`. A non-finite value, or `nonfinite_evaluation=True`, at a trial
point rejects the trial (the line search backtracks); at an outer iterate
either raises `ValueError`.

- **Stateless physics** (depends on x alone): `cached_alm_evaluator(physics)`.
  Call its `cache_clear()` whenever anything else the physics reads changes
  (a threshold, a smoothing temperature).
- **Stateful physics** (warm-started inner solves: Boozer Newton, a VMEC
  restart, any solve seeded by the previous one): no cache (a second solve
  at the same x can differ). Evaluate fresh each call, warm-start only from
  solutions the solver accepted, and pass `snapshot_accepted_state_fn() ->
  state` together with `restore_incumbent_state_fn(state)` (both or neither),
  plus `inner_callback(x)` (L-BFGS-B moved to x) and `accepted_callback(x)`
  (the solver accepted the subproblem's x) to track them. The Boozer template
  (`BoozerSingleStageProblem.solve`, `accept_inner_iterate`,
  `accept_outer_iterate`, `snapshot_accepted`, `restore_incumbent`) is the
  worked pattern.

## Hybrid quartet

For smoothed rows whose exact ("hard") value should decide feasibility and
drive the multiplier update, return the four keys
`hard_signed_constraint_values`, `hard_violation_values`,
`surrogate_signed_constraint_values`, `hard_dual_update_values`, all or none
(a missing member raises `KeyError`). The augmented Lagrangian uses the
surrogate (smooth) values; a disagreement between the channels blocks
`success` and can end in `signal_mismatch_*` reasons. At an active boundary
a disagreement means a row with a live surrogate shift whose surrogate
value is more than the feasibility gate away from its hard value; identical
channels never disagree, so a hybrid run can converge with rows active. As `ALMPhysics` extras
(the Stage-2 template's `HYBRID_QUARTET = True` path):

```python
import numpy as np
from simsopt_alm import ALMPhysics


def hybrid_physics(base_value, base_grad, surrogate, hard, constraint_grads):
    """ALMPhysics whose L uses ``surrogate`` and whose feasibility and
    multiplier update use ``hard`` (both arrays of scaled row values)."""
    hard_violation = np.maximum(hard, 0.0)
    return ALMPhysics(
        base_value=base_value,
        base_grad=base_grad,
        constraint_values=surrogate,
        constraint_grads=constraint_grads,
        extras={
            "dual_update_values": hard,
            "feasibility_values": hard_violation,
            "max_feasibility_violation": float(np.max(hard_violation)),
            "hard_signed_constraint_values": hard,
            "hard_violation_values": hard_violation,
            "surrogate_signed_constraint_values": surrogate,
            "hard_dual_update_values": hard,
        },
    )
```

## minimize_alm arguments

`minimize_alm(x0, constraint_names, evaluate_problem, settings, inner_options,
...)`; the keywords:

- `inner_callback(x)`, `accepted_callback(x)`: per L-BFGS-B iterate / per
  accepted subproblem result.
- `outer_state_callback(outer_iteration, multipliers, penalty)`: at the start
  of each outer iteration (e.g. to refresh something the physics reads, then
  `cache_clear()`).
- `snapshot_accepted_state_fn`, `restore_incumbent_state_fn`: stateful physics.
- `initial_multipliers` (nonnegative, one per row), `initial_penalty`: warm
  start from an earlier result; not with `resume_from`.
- `constraint_blocks`: one group label per row (reports only).
- `base_bounds`: box bounds on x, a sequence of `(lower, upper)` pairs or an
  object with `lb`/`ub`. x0 is projected onto the box (as L-BFGS-B does); a
  `resume_from` whose x lies outside it raises `ValueError` (it came from
  other bounds).
- `resume_from`, `on_outer_boundary`: checkpoint/resume (use
  `alm_checkpointing`); resuming needs the same x0 (the checkpoint's x), names
  and blocks.
- `on_outer_step`: per-step events (use `ALMHistoryRecorder.record`).
- `continuation_policy`: default `DefaultContinuationPolicy()`.

## ALMResult fields

`x` (the best hard-feasible iterate when `restored_best_feasible`, else the
last), `success`, `termination_reason` ([termination.md](termination.md)), `message`,
`objective` (f without penalty terms), `constraint_names`,
`constraint_values` (signed g in name order), `max_violation`, `multipliers`,
`penalty`, `stationarity_norm`, `kkt_stationarity_norm` (None when
unavailable, e.g. no active row), `nit` (all L-BFGS-B iterations), `outer_iterations`,
`restored_best_feasible`, `restored_best_feasible_reason`, `evaluation`
(read-only copy of the final evaluator dict), `inner_result`.

## Problem-module contract

The generated `alm_problem.py` (from a template) defines
`build_problem(smoke: bool = False)`, returning an object with the members
below. `run_alm.py`, `gradient_check.py` and `sign_check.py` import it as
`from alm_problem import build_problem`, so they find it through the script's
own directory (`run_alm.py`) or `PYTHONPATH=<problem dir>` (the checks). Each
template is standalone: it repeats the few helpers it needs.

| Member | Type | Read by |
|---|---|---|
| `name` | `str` | all |
| `x0` | `np.ndarray`, the free dofs at the start | runner, both checks |
| `constraint_names` | `tuple` of unique `str`, one per row | all |
| `physics(x)` | `ALMPhysics` at x (sets the problem's objects to x) | both checks |
| `evaluator` | the `evaluate_problem` for `minimize_alm` | runner |
| `settings` | `ALMSettings` | runner |
| `inner_options` | L-BFGS-B options with `maxiter` | runner |
| `solver_callbacks()` | `dict` of extra `minimize_alm` keywords (`{}` when stateless) | runner |
| `sign_probes()` | tuple of `SignProbe(label, x, violated, satisfied)` | `sign_check.py` |
| `shared_source_rows` | tuple of row names whose probe expectation reads the row's own source | `sign_check.py` |
| `finish(result)` | JSON-serializable `dict`; sets the objects to `result.x` and writes outputs | runner |

`run_alm.py` itself exposes `run(problem, history=None, checkpoints=None,
resume=None) -> dict`, the summary its command line prints; an existing
script calls it directly ([existing-script.md](existing-script.md)).

`SignProbe.violated` / `.satisfied` name rows known to be violated (g > 0) or
satisfied (g <= 0) at `x` from the physics, not from the row code: a bound
you can check by hand (generic), or a measurement compared with the bound
(Stage-2 and Boozer templates, via each row's `measure`). A measurement that
reads the row's own source (the Boozer template's iota and major radius, the
values `Iotas` and `MajorRadius` return) checks only the row's sign and
bound; those rows are listed in `shared_source_rows`, and `sign_check.py`
says so.
