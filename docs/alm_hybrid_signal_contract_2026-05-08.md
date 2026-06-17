# ALM Hybrid Signal Contract

Date: 2026-05-08

Branch: `surrogate-confinement-v2`

Scope: This document pins the documented contract for the augmented Lagrangian (ALM) inner-objective vs. dual-update signal split currently used by the single-stage stellarator search in this repository, names the convergence theory we forfeit, enumerates the engineering safeguards, and forbids future refactors that re-route either signal without a fresh derivation. The contract concerns the ALM mode only (it does not affect weighted-sum or surrogate-only modes).

## Statement of the hybrid contract

The inner-solve augmented Lagrangian uses **surrogate** (smoothed) signed constraint values; multiplier (dual) updates use **hard** (true) signed values. Both signals are produced by stage-2 evaluation, carried side-by-side through the ALM control flow, and routed at well-known choke points.

Choke-point citations (paths relative to repo root):

- Inner objective is fed the surrogate signal:
  - `examples/single_stage_optimization/banana_opt/stage2_objectives.py:2831-2838` — `augmented_inequality_objective(...)` is called with `normalized_surrogate_signed_constraint_values` as the constraint argument that drives the augmented penalty term.
  - `examples/single_stage_optimization/banana_opt/stage2_objectives.py:2839-2861` — both signal channels (`hard_signed_constraint_values`, `hard_violation_values`, `surrogate_signed_constraint_values`, `hard_dual_update_values`, plus their raw counterparts) are stored on the evaluation dict so downstream ALM control flow can pick the correct one without re-evaluating geometry.
- Dual update consumes the hard signal:
  - `examples/single_stage_optimization/alm_utils.py:2264-2325` — `_extract_stage2_constraint_signal_state` selects `hard_dual_update_values` as `preferred_dual_update_values` when explicit stage-2 signals are present, and surfaces the signal arrays via `ALMConstraintSignalState`. The activity masks live on `ALMConstraintRoutingState`, built by `_constraint_routing_state` below.
  - `examples/single_stage_optimization/alm_utils.py:3466-3481` — `_handle_alm_dual_update_transition` projects new multipliers using `routing_state.signal_state.preferred_dual_update_values` (the hard channel).

## Why we accept the trade-off

Stellarator geometric constraints (coil-coil distance, coil-surface distance, max curvature, poloidal extent, etc.) admit smoothed surrogates that have stable gradients suitable for L-BFGS-B inner solves. The hard signal is non-smooth at boundaries — the "true" constraint values come from `max(...)`, hinge, or distance-to-set forms that are non-differentiable at activation, and they generate gradients that are zero outside the active set and discontinuous on the boundary. Feeding hard signals to L-BFGS-B inside the inner loop produces step rejection cascades, line-search failures, and stalled subproblems.

The dual update needs the hard signal to converge on the actual feasible region, not the smoothed approximation. Updating multipliers on the surrogate would push the dual estimate toward feasibility of the smoothed problem, which may differ from the true feasible region by the surrogate margin; the converged design would be smoothed-feasible but possibly hard-infeasible. The hard channel is therefore the only theoretically sound choice for the dual update step.

## Convergence guarantees forfeited

Classical ALM convergence theory (Bertsekas, *Constrained Optimization and Lagrange Multiplier Methods*, 1982; Conn-Gould-Toint, *LANCELOT: A Fortran Package for Large-Scale Nonlinear Optimization*, 1992) requires the inner objective and the dual update to share the same constraint signal. Specifically:

- Bertsekas (Theorem 2.1 and the subsequent rate-of-convergence theorems for the multiplier method) requires the inner subproblem to minimize the augmented Lagrangian on the same constraint values that drive `λ_{k+1} = λ_k + ρ_k c(x_k)`.
- Conn-Gould-Toint (LANCELOT convergence theorems, sections 4.4 and 6.2) make the same assumption for the bound-constrained augmented Lagrangian framework that L-BFGS-B is used inside.

The hybrid forfeits both the rate-of-convergence guarantees and the strict KKT-at-limit theorem. We do not have a theorem that says the iterates converge to a KKT point of the *hard* problem. We have instead an engineering equilibrium argument: when surrogate and hard agree on activity and sign, the standard theorems apply; when they disagree, the algorithm is operating outside its theory, and the engineering safeguards below are the only correctness rail.

## Dual-update firing rule

The dual update fires when the accepted inner subproblem is inside the scheduled feasibility window and stationary under the update metric. It does not require final hard feasibility and does not require surrogate/hard signal agreement. The update remains `mu_next = max(0, mu + rho * g_hard(x))`; the non-negative projection is the per-constraint guard that relaxes satisfied inequalities and engages violated inequalities. Gating the update on final hard feasibility would prevent multiplier growth on recoverable infeasible starts and would reduce ALM to a pure quadratic-penalty loop.

`signal_mismatch_active` is a success-labeling guard, not a dual-update guard. When hard and surrogate activity disagree, the run must not be labeled converged; the surrogate active-set KKT residual still defines update stationarity, and the hard-channel dual update may still engage violated hard constraints so later inner subproblems can move out of a surrogate-feasible but hard-infeasible basin.

## Engineering safeguards

Two safeguards prevent the hybrid from silently accepting a smoothed-feasible-but-hard-infeasible design as "converged":

1. **Mismatch detection**.
   - `examples/single_stage_optimization/alm_utils.py:2358-2418` — `_constraint_routing_state` builds both `hard_activity_mask` and `surrogate_activity_mask` and sets `signal_mismatch_active = True` when the masks disagree on any constraint (active-mask disagreement) or when the hard side reports feasible while the surrogate positive shift is live (boundary disagreement at `examples/single_stage_optimization/alm_utils.py:2398-2404`).
   - The mismatch flag is computed at every `_constraint_routing_state` consumer that needs routing diagnostics; current call sites are at `examples/single_stage_optimization/alm_utils.py:387, 2898, 3105, 3390, 4334, 4497`. History/result builders that persist mismatch use those routing states, while the inner callback uses the same state for early-stop gating.

2. **Converged-gate guard**.
   - **Cap-binding parity guard.** Each converged arm additionally requires `not run_state.last_cap_binding_active` so a clamped dual update cannot park the iterate at small max-violation/stationarity and be labeled `converged`.
   - `examples/single_stage_optimization/alm_utils.py:4663-4672` — the post-inner converged branch requires both `not signal_mismatch_active` and `not run_state.last_cap_binding_active` *in addition to* the standard feasibility-and-stationarity tolerance check. A run that reaches small max-violation and small stationarity norm under sustained mismatch will not be labeled `converged`. The constraints-inactive path keeps the mismatch guard in the candidate predicate (`examples/single_stage_optimization/alm_utils.py:4655-4661`) and the cap-active guard in its converged arm (`examples/single_stage_optimization/alm_utils.py:4699-4702`). The same false-success guards appear at the skipped-inner shortcut at `examples/single_stage_optimization/alm_utils.py:4358-4370`.
   - Effect: false-success labeling is structurally blocked. The run can still terminate with a non-success label (`max_outer`, `signal_mismatch_stall`, `signal_mismatch_penalty_increase` cycles followed by `max_outer`, etc.), but `result.success` cannot be `True` while the mismatch is active or the multiplier cap is binding.

## Residual risk class

The remaining risk is **failure-labeling chatter** under sustained mismatch: the run terminates without success, but the specific termination reason and history action sequence depend on whether the mismatch fires the `signal_mismatch_stall` arm, the `signal_mismatch_penalty_increase` arm, or simply lets the outer iteration cap exhaust into `max_outer`. The deterministic-termination property test (see Verification below) pins bounded termination under sustained mismatch.

We accept this residual risk because:
- The output is always a non-success result; downstream consumers gate on `result.success`.
- The label set is bounded and inspectable in `result.history`.
- Multiplier-cap-binding diagnostics, mismatch flags, and shift-zero flags are all surfaced for postmortem.

## What this contract forbids

This contract is the single source of truth for the surrogate-vs-hard signal split in ALM mode. The following refactors are forbidden without first re-deriving the relevant theorem:

- **Routing hard signals into the inner objective.** Any change that passes `hard_signed_constraint_values` (or the raw equivalents at `examples/single_stage_optimization/banana_opt/stage2_objectives.py:2853-2860`) to `augmented_inequality_objective` requires a fresh smoothness analysis showing that L-BFGS-B convergence still holds, including line-search behavior at activation boundaries.

- **Routing surrogate signals into the dual update.** Any change that passes `surrogate_signed_constraint_values` (or any non-hard channel) into `_project_nonnegative_multipliers_with_diagnostics` at `alm_utils.py:3466-3481` requires a fresh dual-convergence analysis showing that the multiplier sequence converges to a KKT point of the *hard* problem, not the smoothed problem.

- **Removing the `signal_mismatch_active` guard from the converged gate.** The `not signal_mismatch_active` clauses in the converged branch (`examples/single_stage_optimization/alm_utils.py:4663-4672`) and the constraints-inactive candidate (`examples/single_stage_optimization/alm_utils.py:4655-4661`), plus the same guard at the skipped-inner shortcut (`examples/single_stage_optimization/alm_utils.py:4358-4370`), are load-bearing for the false-success block. They must remain coupled to the converged labels. The cap-binding parity guard `not run_state.last_cap_binding_active` is paired with each of those mismatch guards and is similarly load-bearing.

- **Removing the `hard_dual_update_values` field from stage-2 evaluation output.** `_extract_stage2_constraint_signal_state` raises `KeyError` when this field is missing (`alm_utils.py:2264-2285`); that strict-error behavior is part of the contract surface and must not be loosened.

## Verification

The deterministic-termination property test pins the contract:

- File: `tests/geo/test_alm_utils.py`
- Class: `MinimizeAlmTests`
- Test: `test_alm_terminates_deterministically_under_sustained_signal_mismatch`

The test drives `minimize_alm` with an evaluator whose hard and surrogate signed values disagree on activity for every iteration, runs the same fixture twice with identical inputs, and asserts:

- The run terminates within the iteration cap (no infinite loop under sustained mismatch).
- `result.success is False` (mismatch blocks success labeling).
- `result.termination_reason` is the same stable string across both runs.
- The history action sequence is identical across both runs (no chatter between continuation arms).

The test relies on the deterministic behavior of `scipy.optimize.minimize` (L-BFGS-B) under fixed inputs and on the patched-`minimize` test harness already used elsewhere in the file. No randomness is introduced.
