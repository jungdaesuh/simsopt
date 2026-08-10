# GPU-Native SQP/Primal-Dual Single-Stage Implementation Plan — Revision 3

**Status:** Active convergence-debugging SSOT
**Route:** `CFS-SQP1`
**Supersedes for the next canary only:** Revision 2 execution instructions
**Parent SSOT:** `docs/single_stage_jax_gpu_sqp_primal_dual_implementation_plan.md`

## Decision and scope

Revision 2's ten-step canary failed on convergence, not transfers or the KKT
linear solve. The accepted nonlinear trials reduced the scalar merit while
growing the actual equality residual from approximately `7.8e-16` to
`1.11e-8`, and raw KKT stationarity grew from `5.11e-3` to `6.22e-3`.
Revision 3 addresses that globalization defect in the generic dense SQP core.
It does not retune the physics ledger, scaling, FP64 authority, DOFs,
constraints, KKT certificate, or performance boundary.

Only the ten-step bootstrap canary is authorized in this revision. Cold
endpoint and warm timing runs are prohibited until the revised ten-step gate
passes. If it fails again, this route is closed for this tranche and work
pivots to the coupled fullspace formulation.

## Globalization correction

The device line search retains the Revision 2 exact L1 merit and Armijo
derivative. For each finite nonlinear trial whose actual scaled constraint
infinity norm exceeds

```text
max(current_scaled_constraint_infinity_norm, scaled_feasibility_tolerance),
```

it performs one minimum-norm normal restoration using the current constraint
Jacobian:

```text
S = A A^T
S w = c(u + alpha p)
r = -A^T w
u_restored = u + alpha p + r.
```

The `S` Cholesky factor is built once per SQP iteration and stays on device.
The corrected point is independently reevaluated. It is accepted only when
all of the following hold:

1. the corrected objective and constraints are finite;
2. the actual corrected feasibility is no larger than the cap above; and
3. the corrected actual L1 merit satisfies the unchanged Armijo inequality.

The original trial and any correction reevaluation both count against the
existing joint-evaluation budget. A nonfinite factor, correction, or corrected
trial is rejected; no pseudoinverse or silent fallback is allowed. Identity
retry uses the same restoration rule. The existing zero-hot-transfer boundary
and one initial/final transfer contract remain unchanged.

## Required per-iteration diagnostic telemetry

The fixed-shape device history records, for every accepted iteration:

- accepted step length;
- actual L1 merit and selected penalty;
- scaled constraint infinity norm;
- stationarity and certified KKT residual;
- infinity norm of the multiplier update;
- whether Powell BFGS reset occurred; and
- whether normal restoration was applied.

The legacy receipt `history` object remains byte-compatible. Revision 3's
ten-step diagnostic record may expose the additional telemetry separately;
it must not be used to weaken the existing receipt validator or promotion
criteria.

## Gate sequence

1. Run the focused CPU regression suites for dense SQP, fullspace SQP, route
   contracts, and certificates.
2. Reproduce one bootstrap ten-step run and publish the per-iteration
   telemetry, including all six requested measures and restoration counts.
3. Create the immutable Revision 3 plan digest and bind the revised ten-step
   raw result/receipt to it.
4. Run exactly one RTX 5090 bootstrap ten-step gate. No derivative, one-step,
   cold, warm, A100, or speed campaign is authorized before this gate.
5. The gate must retain all prior finite-state, rank, KKT, objective,
   feasibility, raw-stationarity, memory, transfer, and timing checks. In
   particular, objective must decrease, final feasibility must be at most
   `1e-10` or lower than bootstrap feasibility, raw KKT stationarity must
   decrease, and projected time must remain below the existing engineering
   threshold.
6. If the revised ten-step gate passes, separately authorize the cold endpoint
   certificate and only then conditional warm timing. If it fails, record the
   exact failure and stop this SQP route; pivot to
   `docs/single_stage_jax_gpu_coupled_fullspace_implementation_plan.md`.

## Explicit prohibitions

- Do not run cold or warm solves before a passing revised ten-step artifact.
- Do not claim GPU speedup from a partial, projected, or non-certified solve.
- Do not change the native objective ledger, physical equations, state order,
  constraints, derivative orientation, FP64 precision, or endpoint gates.
- Do not alter or reinterpret the immutable Revision 1/2 artifacts.

## Done criteria for Revision 3

Revision 3 is complete when the focused tests, telemetry reproduction, and one
revised ten-step gate are recorded with source/runtime/device identity and
exact digests, followed by either the conditional next gate after success or a
fail-closed pivot record after failure. No speed verdict is produced by this
revision alone.
