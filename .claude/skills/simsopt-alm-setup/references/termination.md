# Termination reasons

`ALMResult.termination_reason` says why `minimize_alm` returned. This table
lists every reason the package can return (the drift test in
`packages/simsopt-alm/tests/test_alm_setup_skill.py` checks the list against
the source in both directions). `success` is `yes` only for the two
converged reasons.

Read these together with `result.restored_best_feasible`: on any failure the
solver returns the best hard-feasible iterate it saw instead of the last one
when the last one is hard-infeasible or has a worse objective
(`result.restored_best_feasible_reason` says which). A restored `result.x` is
feasible and usable; set your objects to it before saving (`finish` in the
templates does).

## Converged

| Reason | Success | Meaning | Action |
|---|---|---|---|
| `converged` | yes | An approximate KKT point at the returned point (not global optimality, not descent from the start; see [pitfalls.md](pitfalls.md) on offsets), at the stated tolerances, with the shifted multipliers λ⁺ = max(0, λ + ρg): max violation <= `feasibility_tol` (solver and hard channels); augmented-gradient norm <= `stationarity_tol` (with `base_bounds`, components on an active bound that point out of the box do not count); and the complementarity gap Σ_i λ⁺_i max(0, -g_i) <= `feasibility_tol`, absolute in f's units (nonnegative per-row terms, so rows cannot cancel; unchanged when a row is rescaled, g -> Mg, λ -> λ/M, ρ -> ρ/M²; activity bands play no part). No hybrid signal mismatch, no binding multiplier cap. Limit: a nonconvex row steep enough that λ⁺_i x slack stays under the tolerance while λ⁺_i ∇g_i cancels ∇f can still pass; no finite tolerance excludes it. | Accept. Check the physics at `result.x` (the runner's `finish` summary). |
| `constraints_inactive_converged` | yes | Hybrid quartet only: every hard row is strictly inactive (no surrogate activity, zero shift) and the same approximate-KKT test holds. | Accept. The constraints did not bind; check whether the thresholds are the ones you meant. |

## Stopped early

| Reason | Success | Meaning | Action |
|---|---|---|---|
| `plateau_stall` | no | Two consecutive hard-feasible subproblems made no meaningful progress while the multiplier-update test stayed unmet. This includes a feasible, stationary point whose complementarity gap Σ λ⁺_i max(0, -g_i) exceeds `feasibility_tol`: a multiplier on a row with slack, even a large multiplier on a row only slightly inside its bound, blocks success until dual updates shrink it. | The iterate is feasible: usable, but not certified optimal. Find the gap's rows: multiply each row's shifted multiplier max(0, `result.multipliers` + ρ_i x `result.constraint_values`), ρ_i its penalty (`result.penalty`, or that row's entry when the penalty is per row), by its slack max(0, -`result.constraint_values`). For a row clearly inside its bound, rerun from `result.x` with `initial_multipliers` zero on it and `initial_penalty=result.penalty`. For a large multiplier on a nearly active row, rerun from `result.x` with `initial_multipliers=result.multipliers` and `initial_penalty=result.penalty` and a larger `max_outer_iterations`, so further dual updates shrink it (or resume from the last checkpoint). If f is far from O(1), scale it (divide it by a positive scale: |f(x0)|, or a chosen positive reference scale when f(x0) = 0, as the templates do; a negative divisor would turn minimization into maximization): the gap tolerance is absolute in f's units. Widening an activity band does not help. Otherwise, for a tighter optimum, run `gradient_check.py` (a wrong or noisy gradient stalls L-BFGS-B), loosen `stationarity_tol` to what f's accuracy allows, or raise `inner_options["maxiter"]`. |
| `constraints_inactive_stall` | no | Hybrid quartet only: the hard rows are inactive but stationarity stopped improving. | Feasible: usually usable. Same remedies as `plateau_stall`. |
| `signal_mismatch_stall` | no | Hybrid quartet only: hard-feasible while the smooth (surrogate) rows read active, repeated without corrective progress and with a zero surrogate shift. At an active boundary a mismatch needs real disagreement: a row with a live surrogate shift whose surrogate value is more than the feasibility gate away from its hard value (identical channels never mismatch). | Lower the smoothing temperature (surrogate closer to the hard value), or set `continue_on_signal_mismatch=True`, or drop the quartet (smooth rows only). |
| `process_budget_exhausted` | no | Your `accepted_callback` raised `ALMProcessBudgetExhausted` (a budget you enforce, e.g. wall clock). | Resume from the last checkpoint: `run_alm.py --resume <dir>/outer_NNN.pkl`. |
| `penalty_cap_reached` | no | The next penalty raise would exceed `penalty_max`; no better feasible iterate to restore. | Usually an infeasible or badly scaled problem: run `sign_check.py`, check the scale warnings, relax thresholds that may conflict. Raise `penalty_max` only when the rows are O(1). |
| `penalty_cap_reached_restored_best_feasible` | no | As `penalty_cap_reached`, and `result.x` is the restored best hard-feasible iterate. | `result.x` is feasible and usable; for a better objective treat it as `penalty_cap_reached`. |

## Outer-iteration limit

`max_outer_iterations` ran out; the suffix names what the last outer iteration
did. To continue rather than restart, raise `max_outer_iterations` and resume
from the last non-final checkpoint (`run_alm.py --resume <dir>/outer_NNN.pkl`),
or start a new `minimize_alm` at `result.x` with
`initial_multipliers=result.multipliers` and `initial_penalty=result.penalty`.

| Reason | Success | Meaning | Action |
|---|---|---|---|
| `max_outer` | no | The last outer iteration ended on a subproblem continuation (or a hybrid subproblem-limit penalty raise). | Raise `max_outer_iterations`; if every outer ends this way, raise `max_subproblem_continuations` or `inner_options["maxiter"]`. |
| `max_outer_after_dual_update` | no | The last outer iteration updated the multipliers: the method was still converging normally. | Raise `max_outer_iterations` (or continue as above). |
| `max_outer_after_sufficient_decrease_hold` | no | The last outer iteration held the penalty because infeasibility was shrinking fast enough. | Raise `max_outer_iterations`; the violation is decreasing. |
| `max_outer_after_infeasible_stall` | no | The last inner solve stalled while infeasible (no move, no feasibility gain), forcing a penalty raise. | Run `gradient_check.py` and `sign_check.py`; a far infeasible start may need a larger `penalty_init`. |
| `max_outer_after_penalty_increase` | no | The last outer iteration raised the penalty: infeasibility did not shrink enough. | Check for conflicting constraints (relax a threshold), row scales, then raise `max_outer_iterations` or `penalty_init`. |
| `max_outer_after_subproblem_limit_penalty_increase` | no | Hard-feasible, but the subproblem hit `max_subproblem_continuations` without meeting the multiplier-update test. | Raise `inner_options["maxiter"]` or `max_subproblem_continuations`, or loosen `stationarity_tol`. |
| `max_outer_after_signal_mismatch_penalty_increase` | no | Hybrid quartet only: a stalled signal mismatch raised the penalty on the last outer iteration. | As `signal_mismatch_stall`, then raise `max_outer_iterations`. |
| `max_outer_restored_best_feasible` | no | The outer limit or the inner budget ran out, and the final iterate was hard-infeasible or worse than the best feasible one, which `result.x` now is. | `result.x` is feasible and usable; raise `max_outer_iterations` or `inner_options["maxiter"]` for a better one. |

## Inner budget spent

`inner_options["maxiter"]` is the L-BFGS-B iteration budget of the whole
`minimize_alm` call. When it is used up the run stops before the next outer
iteration and the reason is the **action name of the last step** (unless a
better feasible iterate is restored: then `max_outer_restored_best_feasible`).

| Reason | Success | Meaning | Action |
|---|---|---|---|
| `dual_update` | no | Budget spent; the last step updated the multipliers. | Raise `inner_options["maxiter"]`, or resume from the last checkpoint with a fresh budget. |
| `penalty_increase` | no | Budget spent; the last step raised the penalty (infeasibility not shrinking enough). | As `dual_update`; also check scales and conflicting constraints. |
| `sufficient_decrease_hold` | no | Budget spent; the last step held the penalty (infeasibility shrinking). | As `dual_update`. |
| `infeasible_stall_penalty_increase` | no | Budget spent; the last inner solve stalled while infeasible. | As `dual_update`; run `gradient_check.py`. |
| `subproblem_limit_penalty_increase` | no | Budget spent; the last subproblem hit `max_subproblem_continuations`. | As `dual_update`. |
| `signal_mismatch_penalty_increase` | no | Hybrid quartet only; budget spent after a stalled signal mismatch. | As `dual_update`; see `signal_mismatch_stall`. |
| `signal_mismatch_subproblem_limit_penalty_increase` | no | Hybrid quartet only; budget spent after a mismatch subproblem hit its limit. | As `dual_update`. |
| `subproblem_continue` | no | Budget spent right after a subproblem continuation (only a custom continuation policy ends an outer iteration this way). | As `dual_update`. |

## Never returned

| Reason | Success | Meaning | Action |
|---|---|---|---|
| `terminated` | no | The initial value of the loop's exhausted-termination carrier; every step overwrites it, so `minimize_alm` does not return it. | Report it as a bug with the run's history (`run_alm.py --history`). |
