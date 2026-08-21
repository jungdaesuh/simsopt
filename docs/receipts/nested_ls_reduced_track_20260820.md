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

- A ten-step reduced walk at F3 B37.
- Gate 1 flat-native B37 (blocked until the F3 one-step gate passes).
- Gate 2 mixed `H_sc`, predictor identity, implicit adjoint vs native.
- Gate 4 custom implicit VJP.
- Gate 5 B3 and Gate 6 B37 nested timing.
- In-graph fused outer L-BFGS-B. F3 is unchanged.

## Physics equivalence (this track)

Required: same LS residual, grid, weights, Volume label, free-`G`; same
stationary branch under a native C++ reconstruct Newton rejudge.
Trajectories may differ (JAX Armijo vs C++ full step). Gauss–Newton is
not the certified reduced Hessian.

**Produced:** 7×7 NCSX; archived-start 255×64 no-op; F3 GPU B37
**bounded** feasibility probe. Frozen snapshot:
`docs/receipts/evidence/nested_ls_reduced_gate1_f3_b37_bounded_20260820.json`
(schema v2). Pytest does not write that file. Regenerable driver:
`evaluate_f3_b37_bounded_probe`. Publication wording:

> The bounded F3 B37 feasibility probe is complete: QR elimination, an
> off-manifold reduced gradient, a finite synchronized HVP, and the
> native reconstruction reference were produced. AD-through-QR GMRES
> did not produce a Newton step within the attempted bound. No JAX
> nested-LS walk, endpoint parity, or speed claim exists yet.

Recorded 2026-08-20 CPU run
(`JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv-qn-cpu/bin/python`, host
`jungdaesuh-playstation`, `cpu:0`; "F3 GPU B37" is the input lineage,
not the HVP hardware):

- Independent dual-lane load of `pair2-l1` `endpoint_candidate`.
- JAX `y*` from a zero probe: `ι=0.15164961478467412`,
  `G=2.010619298254682`. Versus fused lane inner state
  `(0.1516496147846736, 2.010619298254679)` this is 19 ULP in `ι`
  and 7 ULP in `G`, not an 8-ULP slogan.
- `‖∇_s Φ̂‖₂ = 0.01609557303688543` (finite, off-manifold).
- One AD-through-QR HVP on the unit gradient: finite, **8.007 s**,
  `‖Hv‖₂=833.169`. After HVP, `VmRSS` 1,970,732 KiB = **1.879 GiB**
  (current process RSS, not an HVP-local or GPU peak). `ru_maxrss`
  32,321,968 KiB = 30.825 GiB is the process-lifetime peak at that
  capture.
- Frozen-coil C++ reconstruct Newton: success, `iter=10`, 156.82 s,
  `Δι=-0.010792505231594696`, `‖Δs‖_∞=0.00503530466753932`,
  `ι→0.14085710955307942`, `G→2.0106193053897154` (matches the
  reconstruct diagnostic). Coils frozen.
- AD-through-QR GMRES (`maxiter=1`) did not produce a Newton step
  within the attempted bound. The ~10 min kill is session narrative
  and is not independently auditable (no durable command, timeout
  exit/signal, timestamps, or raw log).

**Not produced by the bounded snapshot:** JAX Newton step, full reduced
walk, C++ rejudge of a JAX endpoint, nested speed claim, F3 inheritance.

**Schur operator and one capped step (this commit):** `Ĥ_ss v = Φ_ss v −
Φ_sy Φ_yy⁻¹ Φ_ys v` with only the 2×2 `y=(ι, G)` block solved
explicitly. Packed HVPs differentiate `Φ(s, y)` and do not go through
QR. Full HVP vectors matched AD-through-QR at 7×7 NCSX and at the F3
B37 unit-gradient direction (`derivative_heavy` second-derivative
tol). Then one host GMRES Newton step at F3 B37, cap `restart=8`,
`maxiter=1`:

- Factor 2.51 s, GMRES 4.61 s / 9 matvecs, wall 10.26 s.
- `gmres` info=1 (rtol 1e-10 not met). Linear residual
  `‖(Ĥ+stab I)dx − g‖₂ = 0.00379`. This is not a fully solved Newton
  step.
- Step accepted, `α=1`. `‖g‖₂` 0.01610 → 0.00380. `ι`
  0.15164961478467412 → 0.15288054398364126. Coils frozen.
- C++ reconstruct rejudge of that JAX point: success, `iter=10`,
  161.46 s, `ι=0.1408571095660965`, `G=2.0106193053897154` (same
  reconstruct branch as the original-point native reference).
- Runtime: JAX CPU `.venv-qn-cpu`, `cpu:0`, host
  `jungdaesuh-playstation`. After the step, `VmRSS` 1,780,452 KiB =
  1.698 GiB; `ru_maxrss` 18,094,940 KiB = 17.257 GiB process-lifetime
  peak.
- Snapshot:
  `docs/receipts/evidence/nested_ls_reduced_gate1_f3_b37_schur_one_step_20260820.json`.
  Pytest does not write it.

**Still not produced:** a Newton-quality linear solve, a full reduced
walk, endpoint parity of the JAX one-step itself, nested speed, F3
inheritance, flat-native B37.

## Next

Raise the Schur Krylov budget only enough to drive the linear residual
below the Newton tolerance, rejudge that better step in C++, then
repeat at flat-native B37. Do not attempt a full walk or timing
campaign before that. Do not reopen F3. Do not inherit 7.70×.
Trajectories need not match; require the same native-rejudged branch.
