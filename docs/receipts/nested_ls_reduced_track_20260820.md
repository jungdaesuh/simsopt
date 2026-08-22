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
`η_max = 0.24` (the observed SciPy-cap ratio). Eisenstat–Walker
Choice 2 now selects η_k. JAX `info` is the 0/−1 NaN placeholder, not SciPy's
iteration count. Frozen F3 JSON above remains the SciPy CPU packet.

**Fourier-block `M` canary (live path, this commit):**
physical ``(m, n)`` TensorFourier blocks of the exact ``Ĥ_ss+stab I``
Schur operator, used as left GMRES ``M``. Default Newton remains
unpreconditioned. 7×7 NCSX is the canary, not F3.

**Chunked dense ``Ĥ_ss`` A/B (live path, this commit):**
memory-capped chunked materialization of ``Ĥ_ss+stab I`` via the
linear-solve SSOT. A is dense LU; B is the dense inverse as left GMRES
``M``. Default Newton stays GMRES. 7×7 is the canary.

**Steps 4–8 (live path at `f50642424`):** GPU F3 B37 one Schur
correction, flat-native B37, ten-step frozen-coil walk, runtime coil
DOFs + implicit adjoint, B3 banana ``run_code``, and B37 nested
timing after B3. Importing ``simsopt_jax`` without
``SIMSOPT_BACKEND_MODE=jax_gpu_fast`` pins JAX to CPU even when
``jax.devices()`` would have been CUDA. Frozen-coil Newton stays
captured-coil; coil sensitivities use
``nested_ls_runtime_coil_closures``. B37 nested timing is JAX
reconstruct Schur walk vs native banana ``run_code`` — different
operators, not F3 7.70×. Those steps are **feasibility / branch
canaries**, not a closed nested GPU solve: the `f50642424` walk ended
at ``‖g‖₂ = 6.92×10⁻⁴`` with ``success=False``, Armijo accepted
``η > η_requested``, the adjoint was a regularized dense 7×7 canary,
and Gate 8 timed different operators.

**Forcing certificate (this commit):** Armijo runs only when the
independent unpreconditioned ``η = ‖(Ĥ+stab I)δs − g‖₂ / ‖g‖₂`` is
at most the Eisenstat–Walker Choice 2 request (``η_max = 0.24``).
Misses retry by doubling GMRES ``maxiter`` up to
``NESTED_LS_SCHUR_GMRES_MAXITER_CAP`` (not by raising ``restart``)
and tightening JAX ``tol`` so incremental GMRES cannot stop early on
its internal residual estimate. Default adjoint is unregularized
(``stab=0``) matrix-free GMRES; dense LU with ``stab=1e-4`` remains
the regularized 7×7 canary. Exact Fourier-block ``M`` at 661 still
costs one HVP per live DOF and stays opt-in. Pytest still does not
write evidence JSON; ``write_strict_json`` is the authored-snapshot
entry. Do not publish a nested/banana ratio. Do not inherit 7.70×.

GPU one-step artifact (``d2bf1fd24``, ``jax_gpu_fast``, ``cuda:0``,
2026-08-21):
``docs/receipts/evidence/nested_ls_reduced_gpu_one_step_20260821.json``.
``step_accepted=True``, ``η=0.23566 ≤ 0.24``, GMRES ``maxiter=1``,
JAX 16.6 s, ``‖g‖₂`` 0.01610 → 0.00380, ``success=False``. C++
rejudge reconstruct ``ι=0.14085710956609662``, ``‖∇J‖₂=2.24e-14``,
surface inf vs reconstruct ``4.16e-12``. Schur vs AD-QR ``rel_l2 =
8.62e-16``. Not a ten-step walk and not a timing claim.

GPU walk attempt (``a03b51987``, ``jax_gpu_fast``, ``cuda:0``,
2026-08-21): fail-closed under the JAX 1e-13 / η / C++ no-op
contract. Diagnostic artifact
``docs/receipts/evidence/nested_ls_reduced_gpu_walk_20260821.incomplete.json``
is **not** claim-grade (the claim path is
``nested_ls_reduced_gpu_walk_20260821.json``). Step 1 accepted
``η=0.23566 ≤ 0.24`` at ``maxiter=1`` (one restart cycle of
``restart=8``). Step 2 Eisenstat–Walker requested ``η≈0.05006``;
unpreconditioned GMRES at ``maxiter=8`` (eight restart cycles)
achieved ``η=0.15141`` and the forcing gate refused Armijo.
``success=False``, ``iters=1``, JAX ``‖g‖₂=0.00380``. C++ rejudge
took 10 iterations (``Δι=-0.012023``, ``ΔG=7.08e-9``,
``Δs_inf=0.005023``) — reconstruct basin, not endpoint no-op.
Walk wall 51.2 s is diagnostic only. Not a nested speed claim and
not F3 7.70×. ``gmres_matvecs=0`` is unavailable JAX telemetry, not
zero Krylov work.

GPU step-2 forcing probe (``45c0643ce`` dirty tree, ``jax_gpu_fast``,
``cuda:0``, 2026-08-21):
``docs/receipts/evidence/nested_ls_reduced_gpu_step2_forcing_20260821.json``.
After the accepted first step, Choice 2 requested ``η_k=0.05006``.
Unpreconditioned GMRES ``restart=8`` from ``x0=0``: ``maxiter=8``
``η=0.193``, ``16`` ``η=0.147``, ``32`` ``η=0.101``. Production-style
doubling to ``maxiter_cap=32`` reached ``η=0.0648`` (still above
``η_k``). Fourier-block Jacobi ``M`` factored in 9.3 s and made the
unpreconditioned certificate worse (``η=0.470`` at ``maxiter=1`` and
``8``). A same-session follow-up with doubling ``maxiter_cap=64``
met the gate: ``η=0.03898 ≤ 0.05006`` in 88.3 s, ``used=64``. Default
``NESTED_LS_SCHUR_GMRES_MAXITER_CAP`` is therefore 64. Fourier ``M``
stays opt-in. Not a walk, not 1e-13, and not a timing claim.

GPU walk with ``maxiter_cap=64`` (``98449f6ec``, ``jax_gpu_fast``,
``cuda:0``, 2026-08-21): fail-closed. Diagnostic
``docs/receipts/evidence/nested_ls_reduced_gpu_walk_20260821.cap64.incomplete.json``
is **not** claim-grade. Step 1 accepted ``η=0.23566 ≤ 0.24`` at
``maxiter=1``. Step 2 accepted ``η=0.03898 ≤ 0.05006`` at
``maxiter=64``. Step 3 accepted ``η=0.21181 ≤ 0.24`` at
``maxiter=16`` (Choice 2 returned ``η_max`` because step 2 barely
reduced ``‖g‖₂``: 0.00380 → 0.00367). Step 4 refused:
``η=0.12039 > 0.04072`` at ``maxiter=64``. JAX ``success=False``,
``iters=3``, ``‖g‖₂=7.81×10⁻⁴``. C++ rejudge ``iter=9``
(``Δι=-0.014688``, ``Δs_inf=0.004808``) — basin, not no-op. Walk
wall 248 s is diagnostic only. Not a nested speed claim and not F3
7.70×.

The cap-64 incomplete JSON at recording commit ``0f95d618c`` had
SHA-256 ``1210d35beda7b54721d4846468f4bb73f111c5c41b8241bb0d117dda1b8c9287``
and incorrectly pointed ``execution_log`` at the cap-8 log. The
pointer is now
``docs/receipts/evidence/nested_ls_reduced_gpu_walk_20260821.cap64.log``.
Step 4 **requested** ``η=0.04071795165373735`` and **achieved**
``η=0.1203881060498997``; those are distinct.

GPU step-4 forcing probe, vector freeze (``a99eb84a1`` dirty tree,
``jax_gpu_fast``, ``cuda:0``, 2026-08-21):
``docs/receipts/evidence/nested_ls_reduced_gpu_step4_forcing_20260821.json``.
Persisted 661 DOFs, clobbered, reloaded; SHA of the loaded vector is
``75c3765e…`` (reload match). Historical SHA ``286e3dab…`` remains
archive only. Live Choice 2 **requested** ``η=0.04071795132091288``
(distinct from cap-64 **achieved** ``η=0.12038811``). Unpreconditioned
``restart=8``, ``M=None``, residual decreased at every cap:

- cap 64: ``η=0.120388``, resid ``9.40e-5``, 86.7 s
- cap 128: ``η=0.078691``, resid ``6.15e-5``, 86.6 s
- cap 256: ``η=0.042179``, resid ``3.29e-5``, 164 s
- cap 512: ``η=0.038648 ≤ 0.040718``, resid ``3.02e-5``, 350 s

Default ``NESTED_LS_SCHUR_GMRES_MAXITER_CAP`` is therefore 512.
Fourier-block ``M`` was not used. Not a ten-step walk and not a
timing claim.

GPU walk with ``maxiter_cap=512`` (``42ad7e11c``, ``jax_gpu_fast``,
``cuda:0``, 2026-08-21): fail-closed. Diagnostic
``docs/receipts/evidence/nested_ls_reduced_gpu_walk_20260821.cap512.incomplete.json``.
Step 4 accepted ``η=0.02842 ≤ 0.04072`` at ``maxiter=512`` (the
frozen-vector probe held). Step 5 accepted ``η=0.173 ≤ 0.24`` and
cut ``‖g‖₂`` to ``1.45×10⁻⁴``. Step 6 refused ``η=0.09404 > 0.02703``
at ``maxiter=512``. JAX ``success=False``, ``iters=5``. C++ rejudge
``iter=8`` (``Δι=-0.00755``, ``Δs_inf=0.00275``) — basin, not no-op.
Walk wall 1147 s is diagnostic only. Surface DOFs are persisted on the
JSON. Not a nested speed claim and not F3 7.70×.

The step-4 forcing JSON producer recorded ``git_dirty=True``; it is a
numerical replay certificate, not independently clean-source
promotion evidence. The clean cap-512 walk at ``42ad7e11c``
corroborates step-4 acceptance (``η=0.02842 ≤ 0.04072``). Walk logs
that printed ``forcing_ok True`` while the last step was rejected were
scoring accepted steps only; last-step η vs that step's η_k is now
the certificate (``last_step_meets_forcing``).

GPU step-6 forcing probe (``b61b7899f`` dirty tree, ``jax_gpu_fast``,
``cuda:0``, 2026-08-21):
``docs/receipts/evidence/nested_ls_reduced_gpu_step6_forcing_20260821.json``.
Loaded 661-vector SHA ``a0493560…effe``, clobber/reload match.
Bound ``ι=0.1484103489869863``, ``G=2.0106193052280394``,
``‖g‖₂=1.449305895×10⁻⁴``. Requested ``η=0.027034810094191494``
(distinct from cap-512 achieved ``0.09404256``). ``restart=8``,
``M=None``. Residual fell; 2048 skipped on the 1200 s wall:

- cap 512: ``η=0.094043``, resid ``1.36×10⁻⁵``, 486 s
- cap 1024: ``η=0.031795``, resid ``4.61×10⁻⁶``, 629 s
- cap 2048: not started (``2×629 s > 1200 s``)

Did not meet ``η≤0.02703``. Residual still falling (ratio 0.34), so
this is not stagnation and not a declaration that unpreconditioned
GMRES is insufficient. Physics-feasibility probe only; not nested
performance. Claim-grade walk JSON remains absent.

Receipt qualification (same standard as step-4): the step-6 forcing
JSON recorded ``git_dirty=True``. Its producer hash for
``nested_ls_reduced_scale.py`` is ``1f4d66ac…``, which is not in git
history (HEAD of that file is ``02b3a71f…`` at the later protocol
fix). Solver modules ``nested_ls_reduced.py`` /
``nested_ls_contract.py`` matched HEAD at the time. The cap-512 η
reproduced the clean walk at ``42ad7e11c`` to ~1e-11 relative — that
is reordered-reduction-level agreement, not bitwise. The load-bearing
cap-1024 “residual still falling” row exists only under uncommitted
producer bytes. It is a numerical replay certificate, not
promotion-grade evidence.

The 2048 skip used a ``2.0 × previous_seconds`` predictor on a
double-pay schedule (``maxiter=1024``, ``cap=2048`` ⇒ 1024+2048
cycles). The protocol now starts that leg at the cap
(``maxiter=2048``, ``cap=2048``) and predicts
``previous_seconds × 2048 / doubling_budget``. Do not remint the
step-6 forcing receipt this turn: with the start-at-cap predictor the
2048 row would run (~840 s ≤ 1200 s). Do not brute-force 2048.

``unpreconditioned_gmres_insufficient`` previously fired on any
``eta_unmet``. It now fires only on stagnation, or on ``eta_unmet``
when the material residual-ratio test fails. Budget exhaustion while
the residual is still falling is not insufficiency.

GPU step-6 solver-architecture canary (dirty tree, ``jax_gpu_fast``,
``cuda:0``, 2026-08-21):
``docs/receipts/evidence/nested_ls_reduced_gpu_step6_architecture_20260821.json``.
Same frozen SHA ``a0493560…effe``, requested ``η_k=0.0270348``,
live-matvec unpreconditioned η. Not a walk, not cap-2048, not a
timing claim. ``git_dirty=True`` (protocol + canary sources
uncommitted); diagnostic, not promotion. Assemble ``~25 s`` is
**cold / compile-inclusive** until a walk measures warm.

- Dense LU of chunked 661×661 Ĥ_ss+stab I (3,495,368 bytes; 661 HVP
  columns): live ``η=3.64×10⁻¹³``; assemble 24.6 s cold
  compile-inclusive + LU 0.066 s. Materialization residual
  ``η=1.51×10⁻¹³``. Operator factor 5.29 s. Complete first direct
  step ≈ ``5.29+24.59+0.066 ≈ 29.95 s`` cold. Meets ``η≤0.02703``
  by ten orders.
- Option B, dense inverse as left ``M``, GMRES restart=8 maxiter=1:
  live ``η=1.40×10⁻¹³`` in 16.6 s, 4 apps — **excludes** the shared
  24.6 s assembly **and** the inversion. The 16.6 s is almost
  certainly XLA compile of the preconditioned loop, not a cheaper
  standalone solve. Option B is strictly dominated by dense LU; it
  only falsifies “preconditioned Krylov can’t do it”; not a
  production candidate. Do not productionize explicit ``H⁻¹``; reuse
  LU factors if Shamanskii is ever chartered.
- Full GMRES on the dense matvec (restart=661, one cycle): live
  ``η=4.42×10⁻¹¹`` in 4.0 s, 664 apps — also excludes shared
  assembly.
- Equal-HVP live sweep (~130–146 apps): restart 8/32/64/128 →
  ``η`` 0.270/0.229/0.202/0.192. None meet 0.027. Incremental vs
  batched η agree to ~1e-14 relative.
- Spectrum: symmetry defect ``9.1×10⁻¹⁶``, 0 negative, 0 complex,
  ``λ∈[9.52×10⁻³, 2.01×10³]``, ``κ≈2.11×10⁵``.
  ``λ_min=9.52×10⁻³ ≫ stab=10⁻⁴``, so ``H_ss`` is PD on its own at
  this iterate, not stab-manufactured.

Closed Krylov argument (same canary, SPD ``κ=2.11×10⁵`` ⇒
``√κ≈459``). Chebyshev: unpreconditioned Krylov needs
``k≈(√κ/2)·ln(2/0.027)≈990`` matvecs to reach ``η=0.027``, but
exact termination at ``n=661`` arrives first. Minimum matrix-free
budget equals the assembly budget (661 HVPs), so matrix-free cannot
beat assemble-and-factor at this dimension at any restart or
``solve_method``. The measured sweep (``η≈0.19–0.27`` at ~130 HVPs)
sits on that bound. Therefore unpreconditioned Krylov is
insufficient at any budget below the cost of the exact solve.
Cap-2048 is the wrong lever. This upgrades the earlier “not
declared insufficient” (budget-exhaustion while residual falling)
into a principled insufficiency of the *unpreconditioned restart-8
lane*, which is a different flag from
``unpreconditioned_gmres_insufficient`` on a single probe row.

Clean remint of the architecture canary (``0cf4b1359``,
``git_dirty=False``, ``jax_gpu_fast``, ``cuda:0``): same file path.
Chunk batch width recorded as ``8`` (env unset). Live LU η
``3.79×10⁻¹³``. Option B rows now carry ``shared_dense_assembly``,
``excludes_assembly_seconds``, and ``excludes_inversion_seconds``.
Do not remint the step-6 forcing receipt.

GPU opt-in dense-LU walk canary (``4ee97459d``, ``git_dirty=False``,
``jax_gpu_fast``, ``cuda:0``, chunk batch 8):
``docs/receipts/evidence/nested_ls_reduced_gpu_walk_20260821.dense_lu.json``.
``linear_solver="dense_lu"`` argument only; GMRES remains the code
default. 8 accepted steps, JAX ``‖g‖₂=2.40×10⁻¹⁴``, C++ rejudge
``iter=0`` (Δι=0, ΔG=0, Δs_inf=0), coils frozen. Live η ~10⁻¹⁴ at
every step. Diagnostic walk wall 153 s is **not** a nested speed
claim. Per-step linear seconds: 28.5 s cold then ~16.1–17.1 s warm;
Schur factor 4.53 s then ~0.17 s. ``ten_step_walk=true`` means
``maxiter=10``; eight steps were accepted. Linear-solve wall is
still stored as ``gmres_seconds`` (future schema:
``linear_solve_seconds``). Claim-grade GMRES walk JSON
``nested_ls_reduced_gpu_walk_20260821.json`` remains absent.

GPU unregularized IFT adjoint canary at that walk endpoint
(``evaluate_f3_b37_endpoint_adjoint_probe``, opt-in
``max_dense_linearization_bytes=None``, ``stab=0``,
``linear_solver="dense_lu"``; 1 MiB adjoint cap and Newton
``gmres`` default unchanged). Unregularized Ĥ_ss is SPD:
``λ∈[6.54×10⁻³, 1.999×10³]``, ``n_negative=0``, ``κ≈3.06×10⁵``.
Stabilized spectrum from the same matrix plus ``10⁻⁴ I`` (no second
assembly) has ``λ_min=6.64×10⁻³``; ``‖λ_0−λ_stab‖₂=1.70``, so the
Newton factor is not the IFT factor. Live adjoint η ``2.10×10⁻¹²``.
Coil-tangent scan picked index 1 (``‖Ĥ_sc v‖₂=56.5``). VJP matched
``−λᵀ Ĥ_sc v``. Unregularized dense-LU FD (ε=10⁻⁶, control 0 Newton
steps, perturbed 2 steps) matched predicted ``ds/dc`` to
relative ℓ₂ ``8.01×10⁻⁵``. Coils frozen. Diagnostic only; not B3,
not a nested speed claim, not F3 7.70×. Remint after protocol
commit ``eaf4cef4f``: ``git_dirty=False``,
``docs/receipts/evidence/nested_ls_reduced_gpu_endpoint_adjoint_20260821.json``,
live η ``2.21×10⁻¹²``, FD relative ℓ₂ ``8.01×10⁻⁵``.

Native banana ``run_code`` at F3 B37 start
(``evaluate_f3_b37_chunk_banana_probe``): success, coils frozen,
171 s **BFGS+Newton wall**. ``banana_iter=1`` is the **Newton
polish** nit that ``run_code`` returns after BFGS; it is not “native
needed one iteration of anything.” This receipt has **no
``OMP_NUM_THREADS`` pin, sweep, or interleaved repeats**, so 171 s
is an anecdote, not a Gate-6 bar. The dense-LU reconstruct walk
(``55c4ec46f``) is closed physics: 8 exact-Newton steps, JAX
``‖g‖₂=2.40×10⁻¹⁴``, C++ rejudge ``iter=0``, 153 s diagnostic.
Banana and walk land on the same ι to ``~6×10⁻¹²``. That is
scaffolding for a later fair-bar contract, **not** a 1.12× speed
claim: different algorithms, no OMP law, no interleaved process
timing, ``comparable_operators=false``.

Cold single-shot chunk assemble at the LU endpoint: 8:23.1 s,
16:17.8 s, 32:24.4 s, 64:18.0 s; LU 0.060 s. Matrices agree to
``≲5×10⁻¹⁶``. The walk's **warm** per-step assemble at width 8 was
~16 s, faster than every cold-sweep number, so this sweep is
first-touch/XLA, not the production regime. Keep production batch
8 because the cold sweep is **insufficient repeated/interleaved
evidence**, not because “width is not monotonic.” Re-decide only
on a warm repeated sweep. Lagging the **LU** (0.06 s) saves
nothing. Shamanskii **lagged assembly** is now the Gate-6
**numerator** lever, not an optional extra after a timing claim.

F3-B37 Volume **directional-gradient** canary (not a B3 outer
optimizer): cotangent ``∇_s Volume``, unregularized dense LU IFT,
one coil-direction FD. Live η ``1.15×10⁻¹⁴``, FD relative
``4.32×10⁻⁶``. Receipt flag ``moving_coil_b3_outer`` is **false**;
``outer_optimizer_loop`` is false. One direction certifies a
projection; more directions/step-halving belong before an outer
coil optimizer.

OMP-pinned interleaved banana sweep
(``evaluate_f3_b37_banana_omp_sweep``, threads ``{4,8,16,32}`` × 2):
all pinned, coils frozen, BFGS ~682 iters then Newton 1. Fastest
``OMP=16`` ~116–119 s; ``OMP=32`` slower (~145–149 s). Unpinned 171 s
was a weak native, not the bar. At ``OMP=16`` the split is BFGS
≈91–93 s (682 steps) + Newton polish ≈25–26 s. JAX's **warm**
Newton step (~16.7 s assemble+factor) **beats** native polish
(~25.5 s). The walk loses on (a) cold step-1 XLA premium
(~16 s) and (b) six extra Newton steps doing work native covers
with cheap BFGS. Mimicking BFGS-then-polish in JAX loses: envelope
grad ~0.4 s × 682 ≈ 273 s. Fewer, heavier, batched-assembly
Newton steps is the GPU shape. ``OMP=32`` forked ``bfgs_iter``
682→683 with endpoint still ~10⁻¹² — threading-fork class, and
margin under a 10⁻¹¹ Gate-6 ι/G tolerance. Gap fill ``{20,24}``
on this 32-core box (diagnostic, dirty at ``39e620eec``):
``OMP=20`` 128.2/126.2 s, ``OMP=24`` 135.4/134.5 s, both slower
than ``OMP=16`` and faster than ``OMP=32``. Native best-of-sampled
remains ``OMP=16`` at 116.02 s after ``{20,24}``. Min-bracket fill
``{12,14}`` (diagnostic, dirty at ``a378d49a9``): ``OMP=12``
133.3/118.3 s inner, ``OMP=14`` 119.7/117.7 s inner. Neither
beats 16. Repeat-0 of 12 was a slow first-touch outlier; the
warm pair sits just above 16. Process wall is consistently
~9–10 s above inner (import+setup). ``OMP_PROC_BIND`` /
``OMP_PLACES`` recorded unset. Sweep set
``{4,8,12,14,16,20,24,32}`` is now frozen: the inner minimum is
16, bracketed by 14 (117.7 s) and 20 (126.2 s). ``bfgs_iter``
stayed 682 at 12/14/20/24 (the 682→683 fork is the OMP=32
datum). Still **not** a speed claim. The 153.06 vs 116.02 =
0.76× row is **inner solver vs inner solver**: JAX
``walk_seconds`` is the Newton loop; native ``seconds`` is
BFGS+Newton. Parent process wall is now stored as
``process_wall_seconds``; the OMP=16 receipt does not have it.

Warm in-process chunk repeats (warmup discarded): mean assemble
8:17.0 s, 16:16.6 s, 32:17.6 s, 64:18.0 s. Matches the walk's ~16 s
warm width-8. Keep production batch 8; 16 is not a proven win.

### Receipt metadata rule

Diagnostic receipts may receive **metadata-only** amendments
(``claim_boundary`` labels, publication sentence) via a **named
commit** that does not touch payload numbers. Claim-grade receipts
are **reminted**, never edited in place. Gate-6 runs are
claim-bearing by construction: **clean tree from the first byte**.

### Gate-6 contract (pre-declared, pre-lever)

Before Shamanskii or compile-cache land, so levers cannot move the
goalposts. The published 0.76× row is **not** process-wall vs
process-wall; it is inner-solver vs inner-solver
(``NESTED_LS_GATE6_PRE_LEVER_CLOCK="inner_solver"``).

- Same F3 B37 start SHA; coils bitwise frozen.
- Endpoint ι/G within ``10⁻¹¹`` vs native banana (today ~10⁻¹²).
- JAX stationarity ``‖g‖₂ ≤ 10⁻¹³``.
- C++ rejudge of every JAX claim endpoint is a no-op (``‖Δs‖∞``,
  Δι, ΔG; today ``native_rejudge_iter=0``). Record
  ``‖s_JAX − s_native‖∞``; do not require the two solvers'
  surfaces to match (different algorithms).
- Each side its own solver to its own success criterion (native
  banana ``run_code`` vs JAX reconstruct dense-LU Newton).
- Native = banana at **best-of-swept OMP**. Sweep set **frozen**
  at ``{4,8,12,14,16,20,24,32}``. Inner minimum is OMP=16
  (116.02 s), bracketed by 14 (117.7 s) and 20 (126.2 s).
- Record **both** clocks on both lanes: ``process_wall_seconds``
  (parent ``perf_counter`` around a fresh child: import + setup +
  solver) and ``inner_solver_seconds`` (banana BFGS+Newton, or JAX
  Newton walk). Claim comparison is process wall vs process wall.
- Interleave **complete native/JAX processes**, not merely native
  OMP configs. Aggregation: report every repeat; claim numerator
  and denominator are the **minimum** of ≥3 interleaved process
  walls per lane (also report the median).
- Publish cold-cache and persistent-cache JAX rows separately. The
  ~137 s cache-only figure is a **model**, not a measurement.
- ``OMP_PROC_BIND`` / ``OMP_PLACES`` are recorded. Diagnostic fills
  leave them unset to match the existing OMP=16 bar. Claim runs
  use that same unset bind/places, or remint native under an
  explicit pin.
- Clean tree from the first byte.
- **Pre-lever baseline row (inner solver):** JAX walk 153.06 s vs
  native OMP=16 116.02 s = **0.76×** (JAX ~1.32× slower). Not a
  claim. Process-wall before/after will be a new pair of numbers
  on the claim receipts.

Assembly reuse is required for a plausible win; Shamanskii is the
selected hypothesis: ``δ = H_stale⁻¹ g`` (direct LU apply; JAX
GMRES+``M``+``x0=0`` can return zeros), one live HVP, live
unpreconditioned η vs Eisenstat–Walker, fail-closed reassemble on
miss. Opt-in ``linear_solver="shamanskii"``; default remains
``gmres``. Step records store ``assembled``, ``shamanskii_reused``,
``shamanskii_reassembled``, and the pre-reassembly
``shamanskii_attempt_eta``. Bundle compile-cache attribution
(cache-only / lag-only / both). Cache-only is a measurement, not
the ~137 s model. One avoided 16–17 s assembly is insufficient —
at least two net avoided assemblies.

Do **not** chase BFGS-then-polish in JAX (envelope grad ~0.4 s ×
682 ≈ 273 s).

Product ``linear_solver="dense_lu"`` must **not** reuse the adjoint
1 MiB cap. Inner Newton currently has ``max_dense_linearization_bytes=None``;
661 stored Ĥ is **3,495,368 bytes** (3.50 MB) and would be refused by
the adjoint default. After E2E the product cap is **memory-derived**
(stored ``n²×8`` plus the transient chunk peak against a
device-memory fraction), not a performance threshold: the
spectrum/termination argument says the crossover never swings back
to matrix-free below memory limits.

### Shamanskii / cache attribution (matched-cache rows)

Committed log
``docs/receipts/evidence/nested_ls_reduced_gpu_shamanskii_attr_20260822.log``.
JSON not minted (40 min timeout). ``success`` was not in the log
line. Every lane:

| lane | cache | role | inner walk (s) | reused | reassembled |
|---|---|---|---|---|---|
| cache_only | on | prime | 117.89 | [] | [] |
| cache_only | on | measure | 110.78 | [] | [] |
| cache_only | on | measure | 112.91 | [] | [] |
| lag_only | off | measure | 137.18 | [8] | [2–7] |
| lag_only | off | measure | 133.84 | [8] | [2–7] |
| both | on | prime | 110.91 | [8] | [2–7] |
| both | on | measure | **97.16** | [8] | [2–7] |

Matched-cache inner walks: Shamanskii **equal-or-faster** than
dense_lu (97.16 and 110.91 vs 110.78–117.89). The 97.16 s both
row is a **single warm measure**, uncertified: no JSON, no
``success`` bit in the log, n=1. Provisional vs native OMP=16
116.02 s is 1.19× inner-solver — **not a claim**. Lag-only
(cache off) vs cache-only (cache on) is not a matched pair; that
comparison produced the false “Shamanskii is slower” verdict.

Reuse is only step 8; steps 2–7 reassemble. Process wall is
~300–340 s because the child still includes native reconstruct
and rejudge. JAX process-wall floor (import+init+cache load) is
unmeasured.

### Report-authoring rule

Every comparative verdict cites **matched-state** rows from the
committed artifact. Every lane in the log appears in the table.
Every native number carries its OMP pin inline. Do not author a
verdict before an independent read of the artifact.

## Next

Protocol is implemented (refinement Shamanskii, split-lane
attribution with ``REPEATS=3``, child ``_T0`` clocks, Gate-6
driver). Numbers below still require a clean-tree remint.

1. Remint Shamanskii/cache attribution JSON from a clean tree:
   ``--lane`` one process, ≥3 repeats, both included, ``success``
   in the log line. Certifies or corrects 97.16 s.
2. Measure the JAX process-wall floor (import+init+cache load)
   via ``--lane floor`` (cache-on and cache-off).
3. **Clean-tree** Gate-6 claim run against the frozen contract
   (``benchmarks/nested_ls_gate6_claim.py``). Both sides use parent
   subprocess wait; JAX subtracts native reconstruct and rejudge.
4. A100 rung on Landau after Gate-6, with its own OMP contract.
5. New eight-term outer optimizer charter: B3, then B37,
   after multi-direction step-halved FD. Optimistix/Lineax
   belong there, not as a mid-campaign linear-algebra swap.

**NO-GO:** cap-2048, adjoint-1-MiB as inner LU budget, mimicking
native BFGS in JAX, nested speed claim from 153/171, 153/116, or
the uncertified 97.16 s both-lane row, unmatched-cache Shamanskii
verdicts, F3 7.70×, remint of the dirty step-6 forcing JSON,
claim-grade GMRES walk JSON
``nested_ls_reduced_gpu_walk_20260821.json``.

Frozen-coil reconstruct physics at 661 is closed for opt-in
``linear_solver="dense_lu"``. Unregularized IFT at that endpoint is
closed. Volume directional FD is a one-direction canary. Native
timing bar is the OMP sweep, not the unpinned 171 s anecdote.

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
