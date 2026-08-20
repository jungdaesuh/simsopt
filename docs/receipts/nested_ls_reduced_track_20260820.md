# Reduced nested-LS JAX track (2026-08-20)

**Status: new track opened. Not an F3 speed claim. Not a B3/B37 nested
timing claim.**

F3 remains the sealed flat-675 GPU vs flat-native L-BFGS-B win (B37
7.70×). This receipt starts the separate reduced nested-LS architecture:
exact two-column QR for `(ι, G)`, Newton on the 661 surface DOFs of
`Φ̂(c, s) = Φ(c, s, y*(c, s))`.

## Gate 0 — contract (locked)

| Bar | What it is | Use |
| --- | --- | --- |
| **Physics** `reconstruct_newton` | `constraint_weight=1`, free `G`, `weight_inv_modB=True`, `stab=1e-4`, `tol=1e-13` on `‖∇J_LS‖₂`, Volume label, `maxiter=10` | Branch, rollback, stationarity |
| **Timing** `banana_run_code` | BFGS then Newton, `stab=0`, `newton_tol=1e-11`, `newton_maxiter=40` | Gate 5 B3 process-wall only, after physics match |

Constants: `src/simsopt_jax_adapters/geo/nested_ls_contract.py`.
Banana and reconstruct are **not** the same operator. Do not time a
reconstruct Newton against banana `run_code`, and do not inherit 7.70×.

## What this commit implements

- `nested_ls_reduced.py`: `r(s, y) = A y − b` by `jacfwd` in `y`, QR
  `y*` (`solve_flat675_y_qr`, rank gate), `Φ̂`, exact `∇_s Φ̂` and
  `H_ss v`, reduced Newton on `s` with the reconstruct persist predicate.
- 7×7 NCSX tests: affinity, `y*` minimizes `Φ` at frozen `s`, FD gradient
  and HVP, reduced Newton vs native on-manifold and after a surface step.
  `tests/geo/test_nested_ls_reduced.py`: 7 passed in 102.13 s.
- Gate 1 **archived start, 255×64, produced** via
  `nested_ls_reduced_scale.py` (bundle Biot-Savart JSON + surface spec).
  `tests/geo/test_nested_ls_reduced_scale.py`: 2 passed in 49.99 s
  (`JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu`), including a native C++ LS
  Newton rejudge. Grid is locked at 255×64 / 661 DOFs. Reduced `y*`
  matches the reconstruct QR certificate `(ι, G) = (0.1500517839808274,
  2.010619295609829)`; both C++ and reduced Newton are no-ops
  (`iter=0`, `‖∇_s Φ̂‖₂ ≤ 1e-13`, coils frozen). Dense Hessian
  materialization is
  **off** at this scale: an earlier attempt with it on reached ~66 GB
  RSS and was aborted. Assembling `H_ss` at 661 is Gate 3.

## Explicitly not produced

- One reduced Newton step or a ten-step walk at F3 B37 (AD-through-QR
  GMRES). A `maxiter=1` probe was killed after 10 min; isolated HVP is
  the bounded path.
- C++ rejudge of a JAX Newton endpoint.
- Gate 1 flat-native B37 (blocked until the F3 one-step gate passes).
- Gate 2 mixed `H_sc`, predictor identity, implicit adjoint vs native.
- Gate 3 dense-vs-Krylov / Schur `H_ss` operator at 661.
- Gate 4 custom implicit VJP.
- Gate 5 B3 and Gate 6 B37 nested timing.
- In-graph fused outer L-BFGS-B. F3 is unchanged.

## Physics equivalence (this track)

Required: same LS residual, grid, weights, Volume label, free-`G`; same
stationary branch under a native C++ reconstruct Newton rejudge.
Trajectories may differ (JAX Armijo vs C++ full step). Gauss–Newton is
not the certified reduced Hessian.

**Produced:** 7×7 NCSX; archived-start 255×64 no-op; F3 GPU B37
**bounded** gate (`test_f3_gpu_b37_bounded_hvp_and_native_reference`,
1 passed in 184.27 s):

- Independent dual-lane load of `pair2-l1` `endpoint_candidate`.
- JAX `y*` from a zero probe: `ι=0.15164961478467412`,
  `G=2.010619298254682` (QR, not the 8-ULP lane inner state).
- `‖∇_s Φ̂‖₂ = 0.01609557303688543` (finite, off-manifold).
- One reduced HVP on the unit gradient: finite, **8.007 s**,
  `‖Hv‖₂=833.169`, current RSS 1.93 GiB after, peak RSS 30.8 GiB
  during the JAX AD graph (`ru_maxrss` / `VmRSS` on this process).
- Frozen-coil C++ reconstruct Newton: success, `iter=10`, 156.82 s,
  `Δι=-0.010792505231594696`, `‖Δs‖_∞=0.00503530466753932`,
  `ι→0.14085710955307942`, `G→2.0106193053897154` (matches the
  reconstruct diagnostic). Coils frozen.

**Not produced:** one JAX Newton step, full reduced walk, C++ rejudge
of a JAX endpoint, flat-native B37.

## Next

Replace autodiff-through-QR HVPs with the exact two-variable Schur
operator `Ĥ_ss v = Φ_ss v − Φ_sy Φ_yy⁻¹ Φ_ys v` (`Φ_yy` is 2×2).
Then one reduced Newton step plus C++ rejudge of that JAX state.
Repeat at flat-native B37 only after that passes. Do not reopen F3.
Do not inherit 7.70×. Trajectories need not match; require the same
native-rejudged branch.
