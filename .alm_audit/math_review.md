# ALM Mathematical-Soundness Review

Scope: math correctness only. Physics, JAX/dtype, test coverage, and style
are out of scope and assigned to other agents.
File audited: `examples/single_stage_optimization/alm_utils.py` (4637 LOC)
Caller integration spot-checked:
`examples/single_stage_optimization/banana_opt/stage2_objectives.py`.

## Executive summary

The standard inequality augmented-Lagrangian formulas (objective, gradient,
positive-shift dual update, multiplier projection, penalty schedule, KKT-style
outer convergence test) are implemented correctly under the project's
"feasible iff `c_i ≤ 0`" convention. Constraint normalization (`c_norm = c/s`,
`grad_norm = grad/s`, `λ_raw = λ_norm / s`) is mutually consistent and the
augmented Lagrangian is invariant to that scaling.

There are **two non-blocking but real mathematical concerns** worth recording:

1. **HIGH** — Surrogate-vs-hard split between the inner Lagrangian gradient
   and the dual update breaks classical ALM convergence theory whenever the
   surrogate constraint differs materially from the hard constraint. The code
   detects mismatch and refuses to declare convergence, which is a defensible
   engineering safeguard, but the standard Bertsekas / Conn-Gould-Toint
   convergence guarantees do not transfer.
2. **MEDIUM** — Hitting the multiplier cap (`settings.multiplier_max=1e6`
   default) is recorded as a diagnostic but does not gate convergence or
   trigger an automatic safeguard. A capped multiplier silently corresponds
   to a different dual feasible set and the reported KKT residual may be
   biased.

The diagnostic `_kkt_stationarity_norm` (LOW, see below) is essentially a
restatement of the augmented-Lagrangian gradient norm and not an independent
KKT residual; the convergence test relies on `||∇L_A||` directly so this is
diagnostic-only, but the field name oversells the quantity.

Everything else (penalty schedule, inner-tolerance schedule, signed
constraint semantics, dual-update transition, scale-invariance) checks out.

---

## Findings

### F1 — HIGH: Surrogate/hard hybrid violates classical ALM derivation

- **File**: `alm_utils.py` L374-414 (`augmented_inequality_objective`),
  L1880-1938 (`_extract_stage2_constraint_signal_state`),
  L2937-2967 (`_handle_alm_dual_update_transition`).
  Caller: `banana_opt/stage2_objectives.py` L1893-1972.

- **Math claim under audit**: the inner solver minimizes
  `L_A(x, λ, μ) = f(x) + (1/(2μ)) Σ_i { [max(0, λ_i + μ c_i^surrogate(x))]^2
  − λ_i^2 }` (using the surrogate constraint values and gradients), but the
  outer dual update uses
  `λ_{k+1} = max(0, λ_k + μ_k c_i^hard(x_k))` (using the hard constraint
  values via `preferred_dual_update_values = hard_dual_update_values`).

- **Bug class**: theoretical mismatch. Bertsekas (Constrained Optimization
  and Lagrange Multipliers, 2nd ed., §4.2) and Nocedal-Wright
  (Numerical Optimization, 2nd ed., §17.3-17.4) derive the dual update
  `λ_{k+1} = max(0, λ_k + μ_k c_i(x_k))` as the unique gradient of the dual
  function corresponding to the *same* `c_i` used in the augmented
  Lagrangian. When the inner Lagrangian uses `c_surrogate` and the dual update
  uses `c_hard`, none of the local convergence theorems
  (e.g., Bertsekas Prop. 4.2.3 on multiplier convergence rate, or
  Conn-Gould-Toint LANCELOT global convergence Thm. 4.5) apply. The fixed
  point of the outer iteration solves the KKT system of the *hard* problem
  only if `∇c_surrogate(x*) = ∇c_hard(x*)` and `c_surrogate(x*) =
  c_hard(x*)`; otherwise the iteration may converge to a non-KKT point of
  the hard problem.

- **Correct formula** (per Bertsekas, eq. 4.66 inequality form):
  if the augmented Lagrangian is built from `c_inner`, the dual update must
  also use `c_inner`. The "use surrogate for gradient, hard for update"
  pattern is a heuristic, not a method-of-multipliers iteration.

- **Mitigation already in code**: `_constraint_routing_state` (L1980-2038)
  computes `signal_mismatch_active` from disagreement between hard and
  surrogate activity masks (or surrogate active under hard-feasible state),
  and the "converged" arm at L4091-4096 explicitly excludes the case
  `signal_mismatch_active == True`. The code therefore *refuses to certify
  convergence* on a mismatched configuration. That is the correct safety
  rail, but it means runs that bottom out under signal mismatch will
  terminate with a non-converged status (e.g., `signal_mismatch_stall`)
  rather than meeting Bertsekas-style guarantees.

- **Verdict**: design is intentional and the safeguard is real, but anyone
  reading the code should be aware that the local convergence rate
  (linear with rate `O(1/μ)` per Bertsekas Prop. 4.2.3) is **not**
  established. Document this in the algorithm description.

---

### F2 — MEDIUM: Multiplier cap hit does not gate convergence

- **File**: `alm_utils.py` L1556-1575
  (`_project_nonnegative_multipliers_with_diagnostics`),
  L4091-4096 (post-inner converged arm),
  L4244-4252 (cap-binding diagnostic update).

- **Math claim**: `λ_{k+1} = min(cap, max(0, λ_k + μ_k c_k))`.

- **Bug**: when the cap binds (i.e., the unclipped update would be larger),
  the algorithm silently caps. The cap-binding flag is recorded
  (`run_state.cap_binding_detected`, `cap_binding_indices`) but is **not**
  consulted by the convergence test. Consequently, a run where the true
  optimal multiplier exceeds the cap will report `converged=True` with
  `stationarity_norm ≤ stationarity_tol` whenever the inner solve happens
  to balance `∇f + Σ cap · ∇c_active = 0` to within tolerance. That
  point is not necessarily a KKT point of the original problem (the true
  λ would be larger).

- **Correct production behavior** (per Conn-Gould-Toint LANCELOT Algorithm
  4.1): either (a) treat cap-binding as a forced penalty increase
  (the multiplier update is rejected and only μ rises), or (b) emit a
  termination with reason `multiplier_cap_binding` so the user can decide
  whether to raise the cap. Bertsekas' "safeguarded multiplier" rule
  (eq. 4.69) resets `λ` to a small non-negative number when it grows past
  threshold, then continues. Either is well-grounded.

- **Severity rationale**: with the default `multiplier_max=1e6`, hitting
  the cap requires combinations of `μ · c` that are extreme; in the
  observed banana-coil problems this is unlikely. But the math gap is
  real and should at minimum gate the "converged" success label.

---

### F3 — MEDIUM: Inner-tolerance schedule uses α=1, not Bertsekas' α<1

- **File**: `alm_utils.py` L1605-1606 (`_penalty_schedule_tolerance`),
  L2531-2538 (`_apply_alm_penalty_increase`),
  L2954-2964 (`_handle_alm_dual_update_transition`).

- **Math claim**: tolerances follow `ε_k = max(ε_floor, 1/μ_k)` after a
  penalty increase, and `ε_{k+1} = max(ε_floor, ε_k / penalty_scale)`
  after a dual update.

- **Bug class**: deviation from Conn-Gould-Toint LANCELOT. The classical
  schedule is `ε_k^feas = η_k = η_∞ μ_k^{-α_η}` with `α_η ≈ 0.9`, and
  `ε_k^stat = ω_k = ω_∞ μ_k^{-α_ω}` with `α_ω ≈ 1`. Using `α = 1` for
  feasibility (as here) is more aggressive than the canonical recipe and
  in principle lets the dual update fire at "barely feasible" points
  early on, which slows multiplier convergence rate from the proved
  `O(1/μ)` toward sub-linear. The penalty cap (`1e8` default) bounds the
  damage in practice.

- **Correct schedule (per Conn-Gould-Toint, Trust-Region Methods, 2000,
  Algorithm 14.4.2)**:
  - `η_k = max(η_floor, η_0 / μ_k^{α_η})`, `α_η ∈ (0, 1]` typically `0.9`
  - `ω_k = max(ω_floor, ω_0 / μ_k^{α_ω})`, `α_ω ∈ (0, 1]` typically `1.0`

  The current code is the `α_η = α_ω = 1` corner. This is not wrong,
  it is just the most aggressive end of the design space. Document and
  optionally expose `α` as a setting.

- **Severity**: MEDIUM because the floors (`settings.feasibility_tol=1e-6`,
  `settings.stationarity_tol=1e-6`) cap the schedule so it cannot drive
  to zero faster than the user wants. In practice fine; theoretically
  off-recipe.

---

### F4 — LOW: `_kkt_stationarity_norm` is essentially a restatement of `||∇L_A||`

- **File**: `alm_utils.py` L2041-2073 (`_kkt_stationarity_norm`),
  L2076-2116 (`_stationarity_metrics`).

- **Math claim made by the field name**: residual of the standard KKT
  stationarity condition `∇f + Σ_active λ_i ∇c_i = 0` with `λ_i ≥ 0`
  evaluated by an NNLS active-set lifting.

- **Actual computation**: the function is called with
  `total_grad = ∇L_A = ∇f + Σ_i positive_shift_i · ∇c_i`, then NNLS
  solves for `μ ≥ 0` minimizing `||∇L_A + Σ_active μ_i ∇c_i||`. The
  effective multiplier returned is therefore
  `λ_effective = positive_shift + μ_NNLS ≥ positive_shift ≥ 0`, so the
  search is restricted to multipliers *at least as large as the current
  positive-shift augmenting term*. At converged ALM, `∇L_A → 0` and
  NNLS picks `μ = 0`, so the residual collapses to `||∇L_A||`. The
  diagnostic adds nothing beyond `stationarity_norm` at the convergence
  boundary.

- **Correct KKT residual**: pass `total_grad = ∇f` (the *base* objective
  gradient, before adding any augmenting term), then NNLS over the active
  constraint gradients gives the true KKT residual `min_{λ ≥ 0} ||∇f
  + A_active λ||`. The code already has `metric_grad` and `base_grad`
  in the evaluation dict; `metric_grad` is the augmented gradient and
  `base_grad` is the bare `∇f`. The function should consume `base_grad`,
  not `metric_grad`.

- **Severity**: LOW because the **convergence test** uses
  `stationarity_norm = ||∇L_A||` directly (L2094-2099, used at L4093),
  and that test is correct. The misnamed diagnostic is reported in
  history but not a primary stop condition. Should still be fixed
  because the field is consumed downstream as if it were a real KKT
  residual.

---

### F5 — LOW: Penalty-increase trigger is "inner-solve outcome" not "feasibility ratio"

- **File**: `alm_utils.py` L4228 (dual-update arm) and L4267-4346 (penalty
  fallback).

- **Math note (not a bug, just a deviation)**: the canonical Bertsekas
  rule for choosing dual-update vs penalty-increase is the *feasibility
  ratio test*: dual-update if
  `||c_+(x_{k+1})||_∞ ≤ τ ||c_+(x_k)||_∞` (typically `τ = 0.25`),
  else penalty-increase. The code instead branches on:
  - **dual update**: `hard_feasible_for_update AND stationarity_norm ≤ update_stationarity_tol`
  - **else penalty increase or continuation**.

  This is the LANCELOT-style "branch on inner-solve completion plus
  feasibility-under-gate", which is also a valid choice (Conn-Gould-Toint
  Trust-Region Methods, eq. 14.4.5-14.4.6). Both rules drive the same
  convergence theory under mild conditions; just flagging that the
  branch is not the textbook Bertsekas ratio test.

- **Severity**: LOW (deviation, not bug).

---

### F6 — LOW: Plateau-stall break can shadow a real KKT point

- **File**: `alm_utils.py` L4267-4318 (subproblem-limit and plateau-stall
  arms), `_PLATEAU_STALL_LIMIT = 2` (L308).

- **Math note**: after two consecutive feasible-but-no-meaningful-progress
  outer updates, the run terminates with `plateau_stall`. "Meaningful
  progress" (`_made_meaningful_inner_progress`, L1317-1359) is satisfied
  by *any* of: positional move, objective drop, stationarity drop, or
  feasibility drop, each at `1e-6` relative tolerance. This is a sensible
  budget control, but in principle a run can stop at a feasible point
  with `stationarity_norm > settings.stationarity_tol` if the inner
  solver lands on a flat region that satisfies feasibility but not the
  outer stationarity gate. The result will report `converged=False`,
  `restored_best_feasible=True` if a feasible incumbent exists.

- **Severity**: LOW. The feature behaves as advertised (it is a
  termination guard, not a convergence claim). The user-facing
  `success/converged` flags correctly reflect the situation.

---

## What I checked and confirmed correct

The following items were verified line-by-line and I found no bug:

1. **Augmented-Lagrangian value formula** (L386-413, L528-538). The
   identity `(1/(2μ)) [max(0, λ + μc)]^2 − λ^2/(2μ)
   = (1/(2μ)) (s − λ)(s + λ)` with `s = max(0, λ + μc)` is implemented
   correctly. Algebraically equivalent to the standard inequality
   augmented Lagrangian.

2. **Augmented-Lagrangian gradient** (L394-400). Coefficient is
   `positive_shift = max(0, λ + μc)`, multiplied into `∇c` and added
   to `∇f`. Standard form.

3. **Dual update projection** (L1543-1575). `λ_{k+1} =
   min(cap, max(0, λ_k + μ_k c_k))`. The non-negativity projection is
   correct for inequality `c ≤ 0`; in the alternative project convention
   (`c ≥ 0`) the same formula still gives the correct sign because
   `dual_update_values` is signed accordingly.

4. **Penalty schedule monotonicity** (L1609-1625). `μ_{k+1} = β μ_k`
   capped at `penalty_max`. CLI validation enforces `β > 1`. ✓

5. **Penalty-update tolerance recomputation** (L2531-2538). After a
   penalty increase, both feasibility and stationarity tolerances are
   monotonically tightened (`min(prev, ...)`).

6. **Dual-update tolerance tightening** (L2954-2964). Geometric
   tightening by `÷ penalty_scale` floored at `settings.feasibility_tol`
   / `settings.stationarity_tol`. Standard Bertsekas-style.

7. **Outer convergence test** (L4091-4096). Tests **both**
   `max_feasibility_violation ≤ feasibility_tol` AND
   `stationarity_norm ≤ stationarity_tol`, i.e. primal feasibility AND
   stationarity of the augmented Lagrangian (which equals the
   Lagrangian at the limit). Pure objective stalling does NOT trigger
   this branch.

8. **Constraint normalization invariance**
   (`normalize_alm_constraints`, L417-471, plus
   `alm_raw_dual_estimates`, L618-626).
   With `c_norm = c/s`, `∇c_norm = ∇c/s`, and stage-2 multipliers
   propagated as `λ_raw = λ_norm / s`, the augmented Lagrangian
   `f + (μ/(2)) c_norm^2 + λ_norm c_norm = f + (μ/(2 s^2)) c^2 +
   (λ_norm/s) c` agrees with the unscaled formulation under
   `μ_eff = μ/s^2`, `λ_eff = λ_norm/s`. Internally the optimizer works
   in normalized space, and reported "raw" duals undo the scaling. No
   sign flips, no dropped factors.

9. **Box (trust-radius) bounds** (L1490-1502). Symmetric box
   `[x_i − r·max(1,|x_i|), x_i + r·max(1,|x_i|)]`. Relative trust
   radius. Sane.

10. **Inner-solve early-stop on KKT** (L223-227 in
    `_ALMInnerAttemptEvaluator.callback`). Stops as soon as the
    *inner* tolerances `(effective_feasibility_tol,
    update_stationarity_tol)` are met. The outer loop's stricter
    `(settings.feasibility_tol, settings.stationarity_tol)` test then
    decides whether to declare global convergence or continue.

11. **Block-penalty math** (`_penalty_values`, L1578-1588). Per-constraint
    penalty array `μ_i` is a valid generalization: the augmented
    Lagrangian becomes
    `f + Σ_i (1/(2μ_i)) [max(0, λ_i + μ_i c_i)]^2 − λ_i^2/(2μ_i)`,
    which decomposes additively per constraint. KKT structure is
    preserved per block.

12. **Signed constraint semantics fix** (commit `bfd4b5195`).
    The `_raw_signed_constraint_values` helper (introduced in that
    commit) now reports the *signed* values rather than the violation
    values for diagnostics. No impact on the dual update math (which
    uses signed values throughout); diagnostic correctness only.

---

## What I did **not** verify

- Numerical conditioning of the L-BFGS-B inner solve as `μ` grows
  toward `penalty_max=1e8`. Theoretically the augmented Hessian
  becomes increasingly singular along constraint normals. Out of math
  scope but worth a numerics audit.
- Behavior of the `feasibility_gate` (effective tolerance) widening
  rules in `_effective_feasibility_gate` (`relaxed_feasibility_gate_cap
  = 1e-2` default). It widens the inner-solve feasibility tolerance up
  to a cap; I scanned but did not exhaustively trace its interaction
  with the dual-update gate.
- The exact contract of `gradient_value_kinds` /
  `dual_update_value_kinds` mismatch detection
  (`_multiplier_interpretation`, L798-806).
- Adaptive smoothing path (Phase 5 of the normalization plan) — no math
  changes claimed there beyond the gradient consumed by
  `augmented_inequality_objective`.
- All physics constraints (banana current, iota, coil clearance) — this
  is a separate agent's scope.
