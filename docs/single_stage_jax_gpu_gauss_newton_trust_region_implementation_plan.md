# Single-stage JAX GPU Gauss--Newton trust-region convergence canary

Status: **closed bounded-negative; endpoint campaign not authorized**  
Route: `CFS-GNTR1`  
Date: 2026-08-10

## Objective

Test whether a device-resident, feasibility-restored Gauss--Newton trust-region
state machine makes sustained KKT progress on the unchanged single-stage
fullspace problem. This route intentionally changes the optimizer trajectory,
but preserves the physics, scalar objective, 716 physical degrees of freedom,
255 exact equalities, FP64 derivative authority, optimizer scaling, bootstrap,
dual convention, and endpoint certificate.

The native `287.30421751597896 s` record is an engineering timing threshold,
not a converged endpoint: it exhausted 1,000 BFGS iterations with final
gradient infinity norm `2.8661774598730246e-05`, above the `1e-7` stationarity
target. This canary therefore gates on mathematical convergence progress, not
trajectory parity with native.

## Evidence and hypothesis

The closed routes establish:

- augmented Lagrangian was fast and device resident but did not converge;
- SQP nonlinear restoration held feasibility below `1e-10`, but identity/BFGS
  curvature did not reduce KKT stationarity;
- filter trust region accepted objective improvement while drifting off the
  nonlinear equality manifold;
- the unregularized exact-Hessian KKT step was destructive;
- projected exact-HVP and Gauss--Newton one-step canaries were finite,
  feasible, and safe, but both hit the same first projected-gradient trust
  boundary before curvature could rotate the direction;
- the sealed `CFS-GN1` step had actual/model ratio about `0.983`, so the
  DESC-style rule would accept it and expand the next radius from `2^-10` to
  `2^-9`.

The hypothesis is that accepted-state relinearization plus adaptive radius
growth will let the PSD Gauss--Newton model influence later directions while
trial-state nonlinear correction preserves feasibility.

## Frozen problem and curvature contract

Reuse the canonical `CFS-GN1` objective residual without changing its block
order or formulas:

```text
R(z) = concat(R_non_qs[1600], R_boozer[507], R_iota, R_major, R_length)
Phi(z) = 0.5 * R(z)^T R(z)
z(u) = z0 + S*u
B_GN(v) = J_R(u)^T J_R(u) v.
```

`evaluate_fullspace(...).weighted_total` remains the scalar authority. The
residual is curvature-only and must retain the existing value defect `<=1e-12`
and gradient defect `<=1e-10`. It differentiates through the magnetic field,
surface metric, non-QS average and denominator, full 507-component Boozer
objective residual, and the existing one-sided length term. Gauss--Newton is
an approximate PSD optimizer model; it is not called an exact Lagrangian
Hessian.

At each current state, retain one matrix-free residual linearization:

```text
R0, Jv = jax.linearize(R_u, u)
JT = jax.linear_transpose(Jv, u)
B_GN(v) = JT(Jv(v))[0].
```

Do not materialize the `2110 x 716` Jacobian or `J_R^T J_R`.

## Frozen algorithm

Run from the canonical feasible bootstrap with:

```text
maximum accepted steps = 8
maximum total attempts = 12
initial trust radius   = 2^-10
minimum trust radius   = 2^-20
maximum trust radius   = 2^-4
maximum Steihaug HVPs  = 32 per attempt
projected tolerance    = 1e-10
linear/tangency tolerance = 1e-10
corrected feasibility tolerance = 1e-10
estimated forward-error limit = 1e-7
maximum correction/step ratio = 1e-3
maximum corrected radius excess = 1e-6
```

One device-side attempt is:

1. Evaluate the authoritative objective and scaled equalities at the current
   accepted state; construct the certified multiplier projection and tangent
   projector.
2. Build the retained residual linearization and solve the matrix-free
   projected Gauss--Newton model with Steihaug CG:

   ```text
   minimize_t  r_d^T t + 0.5*t^T B_GN*t
   subject to  A*t=0, ||t||2 <= Delta.
   ```

3. Apply exactly one certified minimum-norm nonlinear correction `delta` using
   the trial-state equality Jacobian. With `s=t+delta`, require
   `||delta|| <= 1e-3*min(||t||, Delta)` and
   `||s|| <= (1+1e-6)*Delta`. A violation is a retryable rejected trial and
   forces radius contraction. Let `g_phi=J_R^T R`, the certified scalar-objective
   gradient. Compute corrected-step prediction from
   `pred=-(g_phi^T s + 0.5*s^T B_GN(s))`, including one paired `B_GN(s)` action;
   never use the Lagrangian/KKT gradient `r_d` for this non-tangent prediction
   and never pair endpoint evidence with a tangent-only prediction.
4. Evaluate the corrected candidate. Define actual reduction from the
   unchanged authoritative scalar objective and predicted reduction from the
   unshifted Gauss--Newton model evaluated on the applied corrected step `s`.
5. A candidate is accepted only when it is finite, has positive predicted and
   actual reduction, corrected scaled feasibility `<=1e-10`, satisfies the
   correction/total-step bounds, and all trial correction/projection
   certificates pass. Rejected coordinates and multipliers remain unchanged.
6. With `rho = actual_reduction / predicted_reduction`, update the radius using
   DESC's frozen semantics:

   ```text
   rho < 0.25 or nonfinite: Delta_next = 0.25 * ||s||
   rho > 0.75:              Delta_next = max(2 * ||s||, Delta)
   otherwise:               Delta_next = Delta
   ```

   Clamp only to the frozen minimum and maximum. Positive-reduction accepted
   steps may still shrink the next radius. An otherwise valid objective-rejected
   trial uses the ratio rule. A nonfinite trial, failed trial correction or
   projection certificate, failed corrected-feasibility gate, or failed
   correction/total-step bound is retryable and always sets
   `Delta_next=0.25*Delta`; it cannot retain or expand the radius. Current-state
   objective/residual, multiplier, Gram/projector, or GN certificate failure is
   fatal `CANARY_NOT_USABLE`, not retryable.
7. Rebuild objective, equality, multiplier, projector, and residual
   linearization only at the next attempt. No host-controlled line search,
   callback, or scalar decision is allowed.

The whole bounded solve uses fixed-shape JAX state and device-side control
flow. Expensive inactive HVP work must be guarded with `lax.cond`. History
stores scalars and status codes, not residual vectors, Jacobians, or AD tapes.

## Design alternatives

Selected: a new additive matrix-free GN trust-region optimizer composed from
the existing certified Gram, projection, Steihaug, and nonlinear-correction
primitives. This isolates the route and leaves all sealed AL, SQP, FTR,
projected-HVP, and GN1 behavior unchanged.

Rejected: retrofit `CFS-FTR1`. Its state machine is coupled to dense BFGS,
filter history, reset policy, and a private dense Steihaug implementation.
Changing it would amplify risk and reinterpret a closed route.

Rejected: copy DESC's dense Jacobian/QR path or Python outer loop. That would
materialize a large Jacobian and reintroduce host synchronization. Only its
accept/reject, radius-update, and accepted-state relinearization semantics are
transferred.

Rejected: another fixed-radius one-step comparison. `CFS-GN1` already proved
that the first boundary direction is non-discriminating.

## Fail-closed canary gate

The run is numerically usable only if:

- the frozen RTX 5090, source snapshot, bootstrap, dimensions `716/255/2110`,
  JAX x64 mode, and FP64 arrays match the receipt contract;
- current-state residual value/gradient, Gram/projector, multiplier, and GN
  certificates pass before every attempt; a current-state failure is fatal;
- deterministic GN bilinear-symmetry relative defect is `<=1e-10`, normalized
  curvature on the frozen deterministic probe is `>=-1e-10`, and normalized
  curvature on every actual terminal Steihaug search direction is
  `>=-1e-10`; exact zero is valid, but materially negative curvature is fatal
  rather than a null-curvature success;
- retryable trial correction/projection, corrected-feasibility, finiteness, and
  correction/total-step failures leave the accepted state unchanged and force
  `0.25` radius contraction; they do not make the receipt numerically unusable
  unless the attempt budget is exhausted;
- every accepted state is finite and has corrected scaled feasibility
  `<=1e-10`;
- every accepted step has positive authoritative objective reduction;
- Steihaug never exhausts its cap, and every termination satisfies its
  interior, boundary, or null-curvature certificate;
- the source manifest is identical before and after execution;
- the timed executable records zero hot H2D/D2H transfers and peak process GPU
  memory below `0.8`.

For mechanism telemetry use the Euclidean metric in frozen optimizer
coordinates. Let `gP=P(r_d)`, `d0=-gP`, and let `t` be the pre-correction
tangent step. For finite nonzero `d0` and `t`, define the sign-invariant
relative orthogonal component

```text
rotation = sqrt(max(0, ||t||^2 - (t^T d0)^2/||d0||^2)) / ||t||.
```

Zero or nonfinite norms make the mechanism predicate false. The curvature
mechanism is exercised only if at least one accepted step after the first uses
at least two GN HVPs and has `rotation>=1e-3`. Otherwise close
`MECHANISM_NOT_EXERCISED`; do not call the optimizer negative.

The route is selected for a separately frozen endpoint campaign only if:

- it reaches all 8 accepted steps within 12 attempts;
- the curvature mechanism is exercised;
- final raw physical KKT stationarity is `<=0.9` times the bootstrap value;
- final authoritative objective does not exceed the bootstrap objective; and
- the least-squares slope of device-resident scaled stationarity over
  accepted-step indices for the last three accepted states, divided by the
  bootstrap scaled stationarity, is `<=-5e-3` per accepted step.

Otherwise close `NOT_SELECTED_BY_BOUNDED_CONVERGENCE_CANARY` or
`CANARY_NOT_USABLE` as appropriate. Timing is diagnostic only. No A100 run,
native speed comparison, or speed claim is authorized by this canary.

## Implementation ownership

- New generic optimizer module and focused mathematical/JIT tests.
- New fullspace adapter reusing the canonical residual, scaling, bootstrap,
  and independent raw endpoint diagnostics.
- New standalone RTX runner and semantic receipt validator with immutable
  artifact sealing.
- This document remains the route SSOT; executed values and disposition are
  appended only after the sealed run.

## Done

- Unit tests cover acceptance/rejection, exact radius transitions, unchanged
  state on rejection, retained-linearization GN HVP, nonlinear correction,
  scalar-objective corrected-step prediction with nonzero correction and
  multiplier, all certificate failures, mechanism telemetry, fixed history,
  and whole-loop JIT compilation.
- Fullspace tests prove unchanged residual reconstruction, optimizer scaling,
  constraint order, dual mapping, and independent raw endpoint KKT authority.
- A production-shape lower/compile preflight uses the exact frozen graph and
  performs no adaptive tuning or canary execution.
- Exactly one supervised RTX 5090 run emits a sealed terminal receipt even on
  compile/resource failure.
- The receipt records per-attempt and per-accepted-state objective,
  feasibility, scaled stationarity, radius, ratio, step norms, HVPs,
  direction-rotation telemetry, and certificates. Raw physical KKT is
  independently audited at the bootstrap and final endpoint only; it is not
  recomputed inside the timed loop. The receipt also records transfers,
  memory, source identity, and the final gate.
- If selected, the next SSOT is a full endpoint campaign to raw KKT `<=1e-7`
  and feasibility `<=1e-10`; only a certified endpoint then authorizes warm
  timing on RTX 5090 and Landau A100.

## Revision 1 infrastructure failure -- 2026-08-10

The first supervised invocation is sealed at
`artifacts/cfs-gntr1-20260810T1114Z`. Result SHA-256 is
`837d5dce3bc42289cdb189bd1268d5ae7f5030f32f9ddb4d4c54238a64f27d41` and
source-manifest SHA-256 is
`e7daef5ba620b31fb861623304d23fe1d1d384f5d36e005bace01e1a09551735`.
The directory is mode `0555` and its two files are mode `0444`.

Its terminal status is `CANARY_NOT_USABLE`. The supervisor invoked the worker
as a file path, so Python placed `benchmarks/` rather than the repository root
on `sys.path`; the worker failed importing the `benchmarks` package before JAX
backend discovery, lowering, compilation, optimization, or memory sampling.
The receipt records `backend=unobserved`, `bounded_solve_attempted=null`, and
zero memory samples. Direct invocation reproduces the same import failure,
while the separately executed production-shape module preflight compiled the
exact `716/255/2110` graph successfully on the RTX 5090 in
`252.6156296179979 s`.

Revision 2 changes only the supervisor worker launch from a script path to
Python module execution (`python -m benchmarks... --worker`) and adds a
regression proving the child can import from the repository root. No numerical
policy, physics, bootstrap, optimizer, gate, or GPU setting changes. Because
Revision 1 never reached the numerical canary, exactly one repaired supervised
execution is authorized after the runner fix and focused validation. The
Revision 1 artifact remains immutable and non-promoting.

## Executed outcome -- 2026-08-10

The repaired provenance-bound RTX 5090 result is sealed at
`artifacts/cfs-gntr1-20260810T1116Z`. Result SHA-256 is
`c7be75af2b0aeb93c48b900e0749f23ffdbcb402929394b08a479e2f24ed1255` and
the 1,944-entry source-manifest SHA-256 is
`9aeddd8e6cab4d2c998463b59ca294db8dcf32e736ff2534fb3db8a7b7bfca83`.
The directory is mode `0555`, both files are mode `0444`, and the receipt binds
the pre-execution Revision 2 plan SHA-256
`c2b3c88024e4d86045d220195ccb2f5bcff2ed2696f62b4090637e65298652f7`.

Execution completed all 8 accepted steps in 11 attempts. The device-resident
solve took `12.552584356 s`; cold lower/compile took `242.478039986 s` and
independent endpoint finalization took `30.864744952 s`. Peak process GPU
memory was `24,826 MiB`, fraction `0.7613702579200785`, with zero hot H2D/D2H
transfers. All runtime, FP64, dimensions, source pre/post identity, residual,
Gram, projection, correction, Steihaug, final-state, and resource certificates
pass.

The curvature mechanism was exercised: accepted step 2 used two HVPs with
rotation `0.9891674199005259`, and accepted step 7 used two HVPs with rotation
`0.5036619528610806`. Feasibility remained certified; maximum accepted scaled
feasibility was `7.187080792281232e-11` and final scaled feasibility was
`2.0229725207445842e-13`. The physical objective decreased by about `21.53%`,
from `8.444212891013073e-05` to `6.626072375931041e-05`.

The route nevertheless fails its convergence-selection gate. Raw physical KKT
stationarity worsened from `0.030894326526207546` to
`0.07480578988973594`, a factor of about `2.42134` rather than the required
`<=0.9` factor. The normalized last-three scaled-stationarity slope was
`+3.2491456366068223`, not `<=-5e-3`; the final accepted step produced the
sharp stationarity regression.

The immutable receipt embeds `CANARY_NOT_USABLE` with reason
`ACCEPTED_STATE_LEDGER`. Post-run audit proved this was a receipt-authority bug:
the serializer mixed independent endpoint bootstrap feasibility
`1.2260439519664147e-15` with timed-loop feasibility
`1.3375024930542707e-15`, and the validator required exact equality. The
`1.1145854108785606e-16` difference is below the frozen `1e-15` ledger
tolerance; all eight post-bootstrap accepted states match exactly. The runner
and validator now use one scaled bootstrap authority and a `1e-15` absolute
tolerance. Re-adjudicating the immutable receipt with the corrected validator
produces:

```text
gate_status = PASS
failure_reasons = []
terminal_status = NOT_SELECTED_BY_BOUNDED_CONVERGENCE_CANARY
selected_for_endpoint_campaign = false
```

The sealed artifact is not rewritten; its embedded historical label remains
visible. The authoritative scientific closure is
`NUMERICALLY_USABLE_BUT_NOT_SELECTED`: the route preserved physics and
feasibility, exercised GPU-parallel GN curvature, and reduced objective, but it
made KKT stationarity substantially worse. No endpoint campaign, A100 run,
warm timing, or speed claim is authorized. No more tuning or replay of
`CFS-GNTR1` is allowed.
