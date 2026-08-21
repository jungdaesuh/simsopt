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

**Still not produced:** a Newton-quality linear solve, a device-resident
GPU Krylov, a full reduced walk, nested speed, F3 inheritance,
flat-native B37.

Publication wording for this packet:

> The Schur operator and one accepted inexact CPU correction are
> validated. Independent C++ rejudging confirms the reconstruct branch.
> Scientific feasibility passes; receipt provenance and GPU-performance
> qualification remain open.

Independent CPU replay of `063b4fe83`: Schur vs AD-through-QR
`rel_l2=8.29e-16`, `max_abs=1.42e-13`; C++ rejudge
`||∇J_LS||_2=2.24e-14`; 661-DOF surface vs reconstruct endpoint
`||Δs||_∞=4.16e-12`. Unpreconditioned GMRES restart 8/16/32 residuals
`3.79e-3 / 1.77e-3 / 8.35e-4`. Do not raise restart toward `1e-13`.
`rtol=1e-10` at `||g||_2=0.0161` requests about `1.61e-12`, not
`1e-13`. Host SciPy GMRES uses `jax.device_get` per matvec.

**Device GMRES (live path, this commit):**
`run_reduced_nested_ls_schur_newton` no longer uses SciPy
`LinearOperator` or per-matvec `device_get`. The linear solve is
`_run_operator_gmres` (`solve_method="incremental"`, `atol=0`) on
`v ↦ Ĥ_ss v + stab v`. The certificate is the explicit residual
`‖(Ĥ+stab I)δs − g‖₂` and forcing `η = residual / ‖g‖₂`. Default
`η = 0.24` (the observed SciPy-cap ratio). Eisenstat–Walker is not
implemented. JAX `info` is the 0/−1 NaN placeholder, not SciPy's
iteration count. Frozen F3 JSON above remains the SciPy CPU packet.

**Fourier-block `M` canary (live path, this commit):**
physical ``(m, n)`` TensorFourier blocks of the exact ``Ĥ_ss+stab I``
Schur operator, used as left GMRES ``M``. Default Newton remains
unpreconditioned. 7×7 NCSX is the canary, not F3.

**Chunked dense ``Ĥ_ss`` A/B (live path, this commit):**
memory-capped chunked materialization of ``Ĥ_ss+stab I`` via the
linear-solve SSOT. A is dense LU; B is the dense inverse as left GMRES
``M``. Default Newton stays GMRES. 7×7 is the canary.

**Steps 4–8 (live path, this commit):** GPU F3 B37 one Schur
correction, flat-native B37, ten-step frozen-coil walk, runtime coil
DOFs + implicit adjoint, B3 banana ``run_code``, and B37 nested
timing after B3. Importing ``simsopt_jax`` without
``SIMSOPT_BACKEND_MODE=jax_gpu_fast`` pins JAX to CPU even when
``jax.devices()`` would have been CUDA. Frozen-coil Newton stays
captured-coil; coil sensitivities use
``nested_ls_runtime_coil_closures``. B37 nested timing is JAX
reconstruct Schur walk vs native banana ``run_code`` — different
operators, not F3 7.70×.

Live GPU one-step (``jax_gpu_fast``, ``cuda:0``, 2026-08-21):
``step_accepted=True``, ``η=0.23566``, JAX 19.3 s, C++ rejudge
``ι=0.14085710956609662``, ``‖∇J‖₂=2.24e-14``, surface vs reconstruct
``4.16e-12``. Ten-step walk: 10 accepted steps, ``‖g‖₂`` 0.01610 →
0.000692, JAX 106 s, rejudge still reconstruct
(``ι=0.14085710964186415``, surface inf ``2.84e-11``). Flat-native
pair2-l2 reconstruct ``ι=0.14085710955509628``. B3 7×7 banana
``run_code`` ``physics_matched=True``. B37 timing after B3: JAX walk
144.1 s / 10 iter / ``success=False`` (not ``1e-13``); native banana
263.1 s / Newton iter 1 / ``success=True`` / ``ι=0.1408571095830707``.
Do not publish a speed ratio. The timing payload sets
``comparable_operators=False``. Runtime coil closures use the binary
``(x, coil_set_spec)`` kernels.

## Next

Physics gates 4–8 of the amended order are produced. Remaining nested
work is not F3: tighter inexact Newton (forcing / ``M``) so a ten-step
walk can reach the reconstruct ``1e-13`` bar, and a same-operator banana
timing at 255×64 only if that inner exists. Do not reopen F3. Do not
inherit 7.70×.

## Validation of the SciPy CPU packet (`063b4fe83` / `96a1e5856`)

- `ruff check` + `ruff format --check` on the nested-LS modules: pass
- `git diff --check`: pass
- targeted `pyright` with `.venv-qn-cpu` on
  `nested_ls_contract.py`, `nested_ls_reduced.py`,
  `nested_ls_reduced_scale.py`: 0 errors
- `tests/geo/test_nested_ls_reduced.py`: 10 passed in 164.71 s
- always-on JSON/provenance tests: 4 passed in 1.11 s
- scale start + F3 bounded + F3 one-step: 4 passed in 569.26 s

## Validation of the device-GMRES amendment (CPU `.venv-qn-cpu`)

- `ruff check` + `ruff format --check` on the nested-LS modules: pass
- `git diff --check`: pass
- targeted `pyright --pythonpath .venv-qn-cpu/bin/python` on
  `nested_ls_contract.py`, `nested_ls_reduced.py`,
  `nested_ls_reduced_scale.py`: 0 errors
- `tests/geo/test_nested_ls_reduced.py` identity, forcing-η, and
  SciPy-import ratchet: 3 passed in 54.11 s
- always-on JSON/provenance tests (`-m "not slow"`): 4 passed
- Slow F3 live `test_f3_b37_one_schur_newton_step_and_cpp_rejudge`
  was not rerun (GPU F3 correction is a later gate)

## Validation of the Fourier-block `M` canary (CPU `.venv-qn-cpu`)

- `ruff check` + `ruff format --check` on nested-LS + `linear_solve.py`: pass
- targeted `pyright --pythonpath .venv-qn-cpu/bin/python`: 0 errors
- `test_fourier_block_m_canary_on_exact_schur` + forcing-η: 2 passed in 35.26 s
- identity, SciPy-import ratchet, JSON (`-m "not slow"`): 6 passed in 39.53 s
- Default Newton remains unpreconditioned. Slow F3 live not rerun.

## Validation of steps 4–8 (this commit)

- `ruff check` + `ruff format --check` on nested-LS modules + tests: pass
- targeted `pyright --pythonpath .venv-qn-cpu/bin/python` on
  `nested_ls_contract.py`, `nested_ls_reduced.py`,
  `nested_ls_reduced_scale.py`: 0 errors
- `tests/geo/test_nested_ls_reduced.py` plus timing-refusal and
  no-write scale tests: 21 passed in 610.19 s
  (`JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv-qn-cpu`)
- GPU one-step and ten-step walk: live `jax_gpu_fast` drivers, not
  pytest (CPU collection skips GPU nodes). C++ reconstruct branch held.
- B37 nested timing driver ran only after B3 `physics_matched=True`.
