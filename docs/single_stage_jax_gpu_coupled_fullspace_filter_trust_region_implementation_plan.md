# Coupled Fullspace Filter/Trust-Region GPU Implementation Plan

**Status:** `AUTHORIZED / NOT_IMPLEMENTED`
**Route:** `CFS-FTR1`
**Role:** sole SSOT for the next single-stage convergence route

## Objective

Implement and validate a device-resident, FP64 JAX equality-constrained
filter/trust-region SQP route that exploits GPU batching and dense linear
algebra while preserving the same physics and mathematical problem as the
native single-stage solve.

Done means either:

1. a provenance-valid RTX 5090 endpoint satisfies every frozen physics,
   feasibility, derivative, and KKT certificate, followed by warm RTX 5090 and
   A100 timing against the `287.30421751597896 s` engineering threshold; or
2. the smallest authorized canary fails and the route is closed
   `BOUNDED_NEGATIVE / NON_PROMOTING` with immutable evidence.

No partial trajectory, projected timing, utilization number, or uncertified
endpoint is a speed result.

## Parent evidence and reason for the route

The preceding `CFS-SQP1` Revision 3 route is closed bounded-negative at:

`/home/jungdaesuh/campaigns/cfs-sqp1-r3-dirty-snapshot-20260809`

Its ten-step gate receipt SHA-256 is
`90d7e416ced18978b56061ee1f7e56cc1fb7a9bf4cdbe1e0ea620342fa49557d`.
It maintained scaled feasibility at `3.302115321685254e-11` and reduced the
objective, but raw KKT stationarity rose from `0.005108879270420846` to
`0.030689422261180984`. The accepted normal correction changed the primal step
while the multiplier update remained tied to the uncorrected step. Therefore
this route must control primal and dual progress at the accepted state; further
line-search SQP tuning is prohibited.

## Frozen scientific contract

The route must preserve without reinterpretation:

- 716 FP64 physical/optimizer variables and 255 equality constraints;
- the exact Boozer residual, volume equality, objective ledger, weights, fixed
  DOFs, variable order, scaling, field conventions, and root branch;
- exact JAX JVP/VJP derivatives and objective-first joint VJP row orientation;
- raw and scaled constraint conventions and multiplier mapping;
- pairwise/exact-tail geometry reductions and the existing endpoint audit;
- endpoint thresholds: scaled feasibility `<= 1e-10`, raw KKT stationarity
  `<= 1e-7`, objective `<= 4.4822247e-08`, and all retained KKT/Schur solve
  residuals `<= 1e-10`.

Different optimizer trajectories are allowed. Changed physics, reduced-result
precision, relaxed tolerances, padded geometry, host callbacks, pseudoinverses,
or hidden fallback to a prior route are not allowed.

## Frozen algorithm

At accepted state `(u_k, lambda_k, B_k)` form the exact objective gradient
`g_k`, constraint vector `c_k`, and Jacobian `A_k` once.

1. Factor `A_k A_k^T` by Cholesky. A nonfinite or rank-invalid factor fails
   closed.
2. Form the minimum-norm Newton normal step
   `n_N = -A_k^T (A_k A_k^T)^-1 c_k`. Restrict it to the normal radius
   `0.8 * Delta_k` with the normal dogleg between the steepest-descent Cauchy
   step and `n_N`; exact `c_k = 0` gives `n_k = 0`.
3. With `P z = z - A_k^T (A_k A_k^T)^-1 A_k z`, solve the tangential model
   using fixed-trip projected Steihaug CG:
   `min_t (g_k + B_k n_k)^T t + 0.5 t^T B_k t`, subject to `A_k t = 0`
   and `||n_k+t||_2 <= Delta_k`. Negative curvature and boundary crossing take
   the positive-root boundary step. The trip count is `min(716, 64)` with
   masked `lax.fori_loop` control.
4. For `s_k = n_k + t_k`, retain
   `pred_f = -(g_k^T s_k + 0.5 s_k^T B_k s_k)` and
   `pred_theta = ||c_k||_2 - ||c_k + A_k s_k||_2`. Nonfinite or nonpositive
   applicable prediction rejects the candidate.
5. Use a fixed-capacity filter of `(theta, objective, active)` arrays with at
   most one insertion per accepted iteration. A trial is filter-acceptable
   against every active entry when either
   `theta_trial <= (1-gamma_theta)*theta_i` or
   `f_trial <= f_i-gamma_f*theta_i`, with both gammas `1e-4`. Objective-type
   steps require `pred_f >= 1e-4*theta_k^2` and
   `ared_f/pred_f >= eta_1`; otherwise feasibility steps require
   `pred_theta > 0` and `ared_theta/pred_theta >= eta_1`, where `eta_1=0.1`.
6. At the accepted coordinates, rebuild `(g_{k+1}, A_{k+1})` and project the
   multiplier by solving
   `(A_{k+1} A_{k+1}^T) lambda_{k+1} = -A_{k+1} g_{k+1}`.
   The relative projection residual must be `<= 1e-10`.
7. Update Powell-damped BFGS with the actual accepted displacement. Form both
   secant gradients with the same accepted-state projected multiplier.
8. Update the radius: shrink by `0.25` after rejection or ratio below `0.25`,
   retain it for ratios in `[0.25, 0.75]`, and expand by `2` up to `8.0` when
   ratio exceeds `0.75` and the boundary is active. Initial radius is `1.0`;
   minimum radius is `2^-20`.

All arrays, counters, filter state, radius state, candidate selection, factor
state, and convergence state remain device-resident and fixed-shape.

## Implementation boundaries

Add, do not retune, the failed route:

- `src/simsopt_jax/geo/optimizers/filter_trust_region_sqp.py`
- `src/simsopt_jax/solve/fullspace_filter_trust_region.py`
- additive `CFS-FTR1` route policy and exhaustive dispatch
- route-specific runner, receipt, validator, endpoint-certificate, and tests

Reuse certified derivative materialization, KKT solve, BFGS, scaling, physics
evaluation, snapshot, runtime, and endpoint-audit primitives. Do not duplicate
physics formulas or alter `CFS-SQP1`, legacy coupled routes, or the default
native/JAX routes.

## Required telemetry

For every accepted iteration retain fixed-shape arrays for objective,
feasibility, raw/scaled stationarity, normal/tangential/combined step norms,
radius, selected radius index, predicted reduction, actual reduction, ratio,
filter decision, multiplier-projection residual, KKT/Schur residuals, BFGS
reset, and evaluation counts. Retain causal failure counters for factor,
nonfinite, projection, model, filter, radius, and budget rejection.

## Gates

### Gate 1: CPU mathematical canaries

Pass synthetic linear and nonlinear equality problems proving normal/tangential
orientation, trust-radius truncation, filter decisions, accepted-state dual
projection, nonsymmetric residual orientation, finite failure behavior, fixed
history shapes, and exact evaluation accounting. Preserve all prior route and
receipt tests.

### Gate 2: one RTX 5090 ten-step canary

Run exactly one canonical-bootstrap ten-step canary. It must satisfy:

- finite FP64 state and exact 716/255 ledger;
- final objective strictly below bootstrap;
- final scaled feasibility `<= max(initial feasibility, 1e-10)`;
- final independently recomputed raw KKT stationarity strictly below bootstrap;
- multiplier-projection residual and KKT/Schur residuals `<= 1e-10`;
- valid model/filter decisions and joint evaluations `<= 1200`;
- one initial H2D, zero hot H2D/D2H, one final D2H;
- peak GPU memory below `0.8` of physical memory; and
- `10 * synchronized_ten_step_seconds < 287.30421751597896 s`.

Any failed condition closes `CFS-FTR1` bounded-negative. Do not tune or replay
the gate under the same SSOT.

### Gate 3: RTX endpoint

Only after Gate 2 passes, run the cold solve to convergence and require the
complete endpoint/KKT certificate, cross-evaluator parity, fixed-state and
inactive-hardware evidence, exact iteration/evaluation accounting, and zero
hot transfers. Failure closes the route; no timing follows.

### Gate 4: conditional timing

Only after Gate 3 passes, run the frozen warm RTX 5090 samples and then the
same source/runtime/physics route on A100. Compare provenance-compatible
receipts against native and issue exactly one of `WIN`, `LOSS`, or
`CLOSED_BOUNDED_NEGATIVE`. A100 behavior is portability evidence, not a
substitute for the RTX endpoint gate.

## Completion checklist

- [ ] Plan, budget, contract, and prior-evidence digests frozen.
- [ ] Additive optimizer and fullspace adapter implemented.
- [ ] Route/runner/receipt/validator/certificate paths implemented.
- [ ] CPU mathematical, compatibility, and tamper tests pass.
- [ ] Exactly one immutable RTX ten-step gate recorded.
- [ ] If and only if it passes, cold endpoint certificate recorded.
- [ ] If and only if the endpoint passes, RTX and A100 warm timing recorded.
- [ ] Results document and final disposition committed without modifying prior
      campaign artifacts.
