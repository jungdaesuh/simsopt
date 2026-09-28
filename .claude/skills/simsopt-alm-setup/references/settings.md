# Settings

## ALMSettings

`ALMSettings` is a frozen dataclass; construction validates every field
(`ValueError` names the field). The table lists every field and its default
(the drift test checks both against the package). Tolerances are absolute, in
the units of your rows and of f: divide every row by its bound or typical size
and f by a reference value (the templates do) so that they read as relative
tolerances.

| Field | Default | Meaning | Change it when |
|---|---|---|---|
| `max_outer_iterations` | 10 | Outer iterations; each ends with a multiplier update, a penalty raise or a penalty hold. | A `max_outer_*` reason: the run was still progressing. At a fixed penalty each outer iteration shrinks the violation by a roughly constant factor. |
| `max_subproblem_continuations` | 20 | Extra re-solves of one subproblem within an outer iteration (the loop runs n + 1) before the penalty is raised. | `max_outer_after_subproblem_limit_penalty_increase`, or every outer iteration ends in `max_outer`. |
| `penalty_init` | 1.0 | Initial penalty, shared by every row. | Rows stay violated while f improves (raise to 10 to 100); the first subproblem overshoots into infeasibility of a stateful solve (lower). |
| `penalty_scale` | 10.0 | Factor of each penalty raise (> 1). | Raises make the iterate jump and a warm-started inner solve fails (use 2 to 5). |
| `penalty_max` | 1e8 | Largest penalty; a raise past it ends the run (`penalty_cap_reached`). None removes the cap. | Rarely: a run that needs a larger penalty usually has badly scaled or conflicting rows. |
| `feasibility_tol` | 1e-6 | Largest row violation that counts as feasible, in the rows' units. Also sets the complementarity-gap tolerance, relative to the objective: Σ λ⁺_i max(0, -g_i) <= `feasibility_tol` x max(1, abs(f)) (the gap is in f's units, since the multipliers scale with f). | Scaled rows: 1e-4 is a violation of 0.01% of a bound; anything below the accuracy of the row's physics is unreachable. |
| `stationarity_tol` | 1e-6 | Largest augmented-gradient norm that counts as stationary, in f's units per dof. | Set it to what f's gradient accuracy allows (a noisy or discretized f cannot reach 1e-6); scale f first. |
| `trust_radius_init` | None | Initial trust radius of each inner L-BFGS-B attempt: a box of half-width radius x max(1, abs(x_i)) around the iterate. None: no box. | A stateful inner solve (Newton warm start) fails on long trial steps: a box keeps the trials near the accepted point. |
| `trust_radius_min` | 1e-4 | Smallest radius; a rejected attempt at it keeps the start iterate. | Rarely. |
| `trust_radius_shrink` | 0.5 | Radius factor after a rejected attempt, in (0, 1). | Rarely. |
| `trust_radius_grow` | 1.5 | Radius factor after an accepted step that used at least half the radius (> 1). | Rarely. |
| `max_inner_attempts` | 4 | L-BFGS-B attempts per subproblem; a rejected attempt retries with a shrunk radius (only with a trust radius). | Rarely; with a trust radius, more attempts tolerate more rejected trials. |
| `relaxed_feasibility_gate_cap` | 0.01 | Cap on the loose early feasibility gate that classifies active rows: max(`feasibility_tol`, min(scheduled tolerance, cap)). | Rarely. |
| `multiplier_max` | 1e6 | Largest multiplier; a clamped update blocks convergence until a later update is not clamped. | Multipliers pin at the cap with O(1) rows: a row is nearly infeasible or conflicting; fix the problem first. |
| `history_max_entries` | 512 | Entries `ALMHistoryRecorder.from_settings` keeps (None: all); `minimize_alm` itself keeps no history. | Long runs whose full history you want. |
| `continue_on_signal_mismatch` | False | Hybrid quartet only: a stalled mismatch with a live surrogate shift keeps re-solving the subproblem instead of raising the penalty. | `signal_mismatch_*` reasons with a smooth surrogate you trust. |
| `penalty_sufficient_decrease_tau` | 0.5 | ALGENCAN's tau: an infeasible step holds the penalty when the infeasibility measure fell to <= tau x its previous value. | Rarely; smaller raises the penalty sooner. |

The loop also schedules looser tolerances early: at penalty rho the
multiplier-update test uses max(`feasibility_tol`, rho^-0.1) and
max(`stationarity_tol`, 1/rho), each divided by `penalty_scale` after every
multiplier update. So with the default `penalty_init=1` the first subproblems
stop after very few L-BFGS-B iterations; convergence needs several outer
iterations.

## Inner options

The fifth `minimize_alm` argument is the options dict of scipy's L-BFGS-B:

- `maxiter`: the L-BFGS-B iteration budget of the **whole** `minimize_alm`
  call, over all subproblems, checked at every outer iteration (the
  continuation steps of one outer iteration share what was left at its start).
  When it runs out the run stops and `termination_reason` is the last step's
  action (see [termination.md](termination.md)). A resumed run gets the budget minus the
  checkpoint's `total_inner_iterations` (`run_alm.py --resume` does this).
- `gtol`: raised to at least min(1e-4, 0.1 x the scheduled stationarity
  tolerance), so each subproblem stops near the current target.
- `maxls`: line-search steps, default 20. `maxcor`, `ftol`, `maxfun`: passed
  through (validated positive).

## Presets in the templates

The templates carry their presets as code, each non-default field with a
one-line reason; `smoke=True` shrinks the resolution, `max_outer_iterations`
and `maxiter` so that `run_alm.py --smoke` checks the pipeline in seconds to
minutes. A smoke run's termination reason is usually an outer-limit or
budget reason; that is expected. The production values are starting points:
tune them from the termination reason, not in advance.
