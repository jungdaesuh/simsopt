# Single-stage JAX GPU Gauss--Newton curvature canary

Status: **closed bounded-negative; no multi-step authorization**  
Route: `CFS-GN1`  
Date: 2026-08-10

## Purpose

Test whether a DESC-like least-squares curvature model supplies a materially
better feasible direction than identity curvature for the unchanged
single-stage fullspace problem. This route keeps the same physics, objective,
716 variables, 255 exact equalities, FP64 derivatives, scaling, bootstrap, and
raw endpoint KKT audit. It changes only the optimizer curvature model.

This is a one-step causal and resource canary. It is not a convergence or
speed-to-solution claim.

## Frozen residual contract

The authoritative scalar remains `evaluate_fullspace(...).weighted_total`.
The following residual is used only for Gauss--Newton curvature, in fixed block
order:

```text
R(z) = concat(R_non_qs.ravel(), R_boozer, R_iota, R_major, R_length)
shape = 1600 + 507 + 1 + 1 + 1 = 2110

dS = ||x_phi cross x_theta||
b = ||B||
b_QS = sum_axis(b*dS) / sum_axis(dS)
Q = sum(dS*b_QS^2)
R_non_qs = sqrt(2*w_non_qs*dS/Q) * (b-b_QS)
R_boozer = sqrt(w_residual/507) * r_full
R_iota = sqrt(w_iota) * (iota-iota_target)
R_major = sqrt(w_major) * (Rmajor-Rmajor_target)
R_length = sqrt(w_length) * max(total_length-length_target, 0)
```

`r_full` is the complete 13-by-13-by-3 Boozer objective residual, flattened in
`[phi, theta, xyz]` order. The 254-component mask remains equality-only.
Differentiation includes the magnetic field, geometry, `dS`, `b_QS`, and `Q`;
none is frozen. The one-sided length term uses the existing `jnp.maximum`
exactly. At the bootstrap kink this is a deterministic JAX generalized
derivative, not a unique classical Hessian.

The residual is valid only if all weights are finite and nonnegative, `Q` is
finite and positive, every residual entry is finite, and both identities pass:

```text
abs(0.5*R^T R - Phi) / max(1, abs(0.5*R^T R), abs(Phi)) <= 1e-12
||J_R^T R - grad Phi||inf / max(1, ||J_R^T R||inf, ||grad Phi||inf) <= 1e-10.
```

## Frozen Gauss--Newton graph

Use optimizer coordinates `z=z0+S*u` and define `R_u(u)=R(z0+S*u)`.
Linearize once at the bootstrap:

```text
R0, Jv = jax.linearize(R_u, u0)
JT = jax.linear_transpose(Jv, u0)
B_GN(v) = JT(Jv(v))[0].
```

No dense 2110-by-716 Jacobian is materialized. `B_GN` must be finite,
bilinear-symmetric to `1e-10`, and positive semidefinite within FP64-scaled
roundoff on deterministic probes.

Gauss--Newton omits residual-weighted objective second derivatives and all
multiplier-weighted equality curvature. It is therefore an approximate PSD
optimizer model, not exact Lagrangian curvature. The objective and all endpoint
certificates remain authoritative and unchanged.

## Shared one-step experiment

Identity and Gauss--Newton use the same certified multiplier projection,
tangent projector, projected-Steihaug state machine, nonlinear correction,
trust radius, and endpoint audit. Both solve

```text
minimize_t  r_d^T t + 0.5*t^T B*t
subject to  A*t=0, ||t||2 <= Delta
```

with `Delta=2^-10`, at most 32 iterations, projected-residual tolerance
`1e-10`, linear/tangency/feasibility tolerance `1e-10`, forward-error limit
`1e-7`, and full step `alpha=1`. No radius, tolerance, or acceptance replay is
allowed after observing the result.

## Design alternatives

Selected: an additive objective-residual module plus a generic callable-HVP
wrapper around the existing projected-Steihaug core. This keeps the residual
representation out of every existing `FullSpaceEvaluation` PyTree and leaves
the sealed exact-HVP route semantically unchanged.

Rejected: adding the 2110-vector to `FullSpaceEvaluation`. That would change
every existing evaluator output/HLO and retain Gauss--Newton-only state in
unrelated routes. Also rejected: five scalar residuals, which reconstruct the
value but create a rank-five, noncanonical curvature model.

## Fail-closed gate

The canary is usable only when:

- backend is exactly one frozen RTX 5090 GPU, FP64, dimensions `716/255`;
- the residual value/gradient certificate and GN symmetry/PSD checks pass;
- the GN curvature observed on the terminal Steihaug search direction is
  normalized-curvature `>=-1e-10` (exact zero/null curvature remains valid);
- source manifest is identical before and after execution;
- all Gram/correction/projection residuals are `<=1e-10` with estimated
  forward error `<1e-7`;
- both steps satisfy tangency and radius gates, have positive predicted
  reduction, and terminate validly rather than exhausting the iteration cap;
- corrected scaled feasibility is `<=1e-10`;
- raw physical objective and KKT stationarity are finite;
- zero hot H2D/D2H transfers and peak GPU memory is below `0.8`.

The hypothesis is supported only if both variants are usable, the
Gauss--Newton objective does not exceed the bootstrap objective, and the
Gauss--Newton raw physical KKT stationarity is at most half both the bootstrap
and identity endpoint values. Otherwise the result is
`NOT_SUPPORTED_BY_ONE_STEP_CANARY` or `CANARY_NOT_USABLE`.

Only `SUPPORTED_BY_ONE_STEP_CANARY` authorizes a separately frozen bounded
multi-step convergence experiment.

## Done

- Residual tests cover fixed block order, termwise scalar reconstruction,
  physical and optimizer-coordinate gradient identity, state-dependent non-QS
  normalization, full Boozer residual, and the length kink.
- Generic candidate-HVP tests preserve the existing exact-HVP wrapper and prove
  candidate validity fails closed.
- The fullspace adapter lowers and compiles, and independently recomputes raw
  endpoint objective/KKT diagnostics.
- One fresh RTX 5090 artifact records provenance, resource use, transfers,
  reconstruction and curvature certificates, endpoint values, and the frozen
  terminal gate; it is sealed read-only and recorded here.

## Executed outcome — 2026-08-10

The single provenance-bound RTX 5090 result is sealed at
`artifacts/cfs-gn1-20260810T0937Z`. The result SHA-256 is
`621198a8d83662324441a2ce478cfa720d3ebe15514843d4bfe8da2e2db81eab`;
the 1,941-entry source-manifest SHA-256 is
`8b837a19e8630e3585d508c28b2fb8e18d03090971bf56e5909aed732af5299e`.
The directory is mode `0555` and both files are mode `0444`. Strict JSON,
source-manifest identity, transfer, memory, numerical, and terminal gates pass.

The terminal status is **NOT_SUPPORTED_BY_ONE_STEP_CANARY**. Both variants
were usable, finite, feasible, and condition-certified, but Gauss--Newton did
not meet the frozen twofold raw-KKT reduction:

| Quantity | Bootstrap | Identity endpoint | Gauss--Newton endpoint |
|---|---:|---:|---:|
| Raw physical KKT stationarity | `3.0894326526207615e-02` | `3.051565685045248e-02` | `3.0348667202548613e-02` |
| Physical objective | `8.444212891013073e-05` | `8.3997191868032e-05` | `8.381261411456691e-05` |
| Scaled feasibility | `1.3375024930542707e-15` | `3.120839150459965e-15` | `1.1591688273137013e-14` |
| Model-step norm | `0` | `6.79119204955272e-04` | `9.765625e-04` |

The canonical residual contract passed with value defect
`1.3552527156068805e-20` and gradient defect
`1.734723475976807e-18`. The matrix-free GN action had bilinear symmetry defect
`1.4625283242488353e-12`, normalized probe curvature
`1.1571113970940355e-01`, and actual terminal normalized curvature
`4.152044093661152e-02`. The GN step used one HVP, encountered no negative
curvature, and reached the frozen trust boundary.

Cold compilation took `145.309710097 s`; synchronized execution took
`2.499218881 s`. Peak process GPU memory was `24,826 MiB`, or
`0.7613702579200785` of the RTX 5090. The timed executable recorded zero hot
H2D and D2H transfers. These are diagnostic timings, not a speed-to-solution
claim.

Gauss--Newton improved raw KKT by only about 1.8% from the bootstrap and about
0.55% relative to identity. Its recorded endpoint scalars agree with the sealed
projected exact-HVP receipt to about `1e-12` relative in raw KKT and `3e-14` in
objective. Both methods terminate on the first common projected-gradient
direction at the same trust boundary, so the one-step agreement is consistent
with boundary truncation before curvature can rotate the search direction.
This does not establish behavior at another radius or over multiple steps. No
multi-step execution or parameter replay is authorized. `CFS-GN1` is closed as
a bounded-negative one-step route.
