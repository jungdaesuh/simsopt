# `stage_two_minimal` DESC-style reformulation — DIAGNOSTIC, NOT CERTIFYING

> **Status: DIAGNOSTIC. This document certifies nothing and promotes nothing.**
> No `src/`, `tests/`, or `examples/` file was read-modified for it; the entire
> harness lives outside the repository at
> `~/simsopt-campaigns/stage-two-minimal-coupled-20260816/`. The device-assignment
> record (`docs/jax_example_device_assignment.md`) is **not** amended by this
> receipt and no scoreboard row moves on its evidence.

- Date: 2026-08-16 · Box: this machine — RTX 5090 (32607 MiB), 64 logical CPUs.
  Campaign-start device state: `artifacts/gpu_start.txt`.
- Artifact root (all paths below are relative to it): `/home/jungdaesuh/simsopt-campaigns/stage-two-minimal-coupled-20260816/`
- **Provenance limitation (harness gap, disclosed).** The leg records do **not**
  carry a repository commit, branch, or interpreter/JAX version. Each leg binds
  only its own runtime: wallclock UTC, loadavg, `OMP_NUM_THREADS`,
  `JAX_PLATFORMS`, `CUDA_VISIBLE_DEVICES`, `SIMSOPT_BACKEND_MODE`,
  `SIMSOPT_PRECISION`, `JAX_ENABLE_X64`, transfer guards, `XLA_FLAGS`, the
  observed jax platform/device, and the full `nvidia-smi` compute-app list.
  Repository state (`HEAD` `4c2b368fa`, branch `pr/jax-port-squashed`, only this
  receipt untracked), Python 3.11.15 and jax 0.10.0 CUDA were confirmed **only at
  write time**, and are recorded as such in `artifacts/provenance_posthoc.txt`,
  which is stamped POST-HOC and is explicitly **not** leg-time binding. No claim
  here rests on the repository having been in a particular state while a leg ran.
  Closing this gap (stamp commit + versions into every leg record) is a
  prerequisite for any future certifying use of this harness.
- `MPI4PY_RC_INITIALIZE=false` on every leg.
- Every **headline** numeral in this receipt — every figure in the §3, §4, §5 and
  §7 tables and every derived ratio — is re-extracted from its source file into
  `artifacts/receipt_numbers.json` (**71** labelled entries, each carrying its
  source path). Incidental figures quoted inline in prose (configuration
  constants, dimensions restated from §1, and file/leg counts) are not all
  separately bound; where a prose numeral is not in that file it is stated with
  its source path in the sentence itself.

---

## 0. Premise correction — read this before the results

The campaign was chartered as *"a coupled (no-nested-solve) reformulation of the
repo's nested-Boozer stage-two problem"*, scoped to `stage_two_minimal` on the
strength of the handoff ledger item *"`stage_two_minimal` as the one clean
remaining DESC-style reformulation candidate"*.

**`stage_two_minimal` has no nested solve and no Boozer residual to un-nest.**
It is the mirror of `examples/1_Simple/stage_two_optimization_minimal.py`: a
fixed-surface, coil-only Stage-II problem. The plasma boundary is loaded once
from `tests/test_files/input.LandremanPaul2021_QA` and never varies; there are
no surface DOFs in the optimizer vector, no inner Newton solve, and no Boozer
constraint. Evidence:

- `src/simsopt_jax/examples/stage_two_minimal.py` takes `surface_gamma` /
  `surface_normal` as fixed device constants and a `FixedSurfaceFluxSpec`.
- `examples/jax/parity/cases/native_stage_two_optimization_minimal.py`
  constructs the surface with `SurfaceRZFourier.from_vmec_input(...)` and never
  frees a surface DOF; the optimizer vector is `(flux + w*QuadraticPenalty).x`.

Therefore the brief's step 2 — *"surface DOFs promoted into the outer problem,
Boozer residual imposed as constraints/penalty"* — is **inapplicable by
construction**, and the brief's step-3 requirement of *"Boozer residual norm at
the same tolerance the nested inner solve enforces"* has no referent. The
surviving coupled machinery in-tree (`src/simsopt_jax/objectives/single_stage_fullspace.py`,
`FROZEN_LAYOUT` = 716 coordinates, Boozer as `FullSpaceConstraints.boozer`) belongs
to the single-stage boozer-vacuum problem, not to this case.

**What the reformulation actually is for this case**, and what was measured:
`stage_two_minimal` is already an exact sum of squares, so the DESC-shaped move
available is replacing the shipped *scalar quasi-Newton* loop (sequential, one
value+grad per step) with a *batched-Jacobian Gauss–Newton/Levenberg–Marquardt*
step whose batch dimension is the DOF count and which factorizes once per outer
iteration and reuses that factorization across the whole damping ladder. That is
the same recipe (batch the derivative dimension, factor-once-reuse, one tiny jit
surface) applied to the one case where it is algebraically exact.

*Charter note.* An earlier draft of this receipt attributed that reformulation
to a "2026-08-14 design note" proposing `TraceableLeastSquaresProblem` +
`SIMSOPT_LM_QR` at residual dimension 1025. **No such note exists** — the
2026-08-14 documents concern dispatch-routing defaults — and the citation has
been removed. What actually chartered this work is the campaign brief for
`stage-two-minimal-coupled-20260816` plus the handoff ledger item quoted above;
the `SIMSOPT_LM_QR` lane is measured in §7.1 because it is the repository's own
least-squares driver and because `docs/receipts/lm_qr_gpu_probe.md` (2026-08-16)
names a production-residual probe as its open deciding experiment — not because
any prior note prescribed it for this case.

---

## 1. The reformulation is an exact restatement (not an approximation)

`SquaredFlux` with the `"quadratic flux"` definition is
`0.5/(nphi*ntheta) * Σ (B·n̂ − B_T)² |n|` and the shipped length term is
`0.5 w max(L − L_T, 0)²` — both already squares. The residual vector is
`r = [ residual_BdotN(...) (1024 entries) ; √w · max(L − 18, 0) ]`, built only
from installed-package entry points
(`simsopt_jax.core.objectives_flux.fixed_surface_flux_residual_from_B`,
`simsopt_jax.objectives.stage_two.stage_two_coil_geometry`).

Source: `artifacts/probe_structure_cpu.json`, scale `native_default`
(nphi=ntheta=32, curve order 5, 100 quadrature points, 4 base curves).

| quantity | value |
|---|---|
| `dim_x` (coil DOFs: 4×33 curve + 3 free currents) | **135** |
| `dim_r` (1024 flux rows + 1 length row) | **1025** |
| Jacobian shape / bytes (fp64 dense) | 1025×135 / **1 107 000 B** |
| value identity `0.5‖r‖² − J` | **0.0** (exact) |
| gradient identity `max|Jᵀr − ∇J| / max|∇J|` | **3.831e-16** |
| JAX vs native C++ objective, relative | **2.098e-16** |
| JAX vs native C++ gradient, relative ∞-norm | **1.197e-15** |

The reformulated problem is the same problem, to round-off.

## 2. Structural facts the verdict rests on

Source: `artifacts/probe_structure_cpu.json`, `artifacts/probe_kernel_gpu_cold.json`,
`artifacts/probe_kernel_cpu_omp32.json`, `artifacts/native_eval_cost.json`.

**Jacobian conditioning — the reformulation inherits a rank-deficient Jacobian.**

| quantity | value |
|---|---|
| σ_max | 9.417e-01 |
| σ_min | 7.033e-17 |
| cond(J) | **1.339e+16** |
| numerical rank at σ_max·1e-12 | **111 of 135** |
| numerical rank at σ_max·1e-10 | **100 of 135** |

At least 24 of the 135 columns are null directions of the residual map (coil
re-parameterization freedom that leaves the field on the surface unchanged).
Undamped Gauss–Newton is singular here; Levenberg–Marquardt damping is not an
optimization nicety on this problem, it is a requirement.

**Per-iteration kernel costs (medians).**

| kernel | time | source |
|---|---|---|
| native C++ value+grad, `OMP_NUM_THREADS=8` | **2.073 ms** | `artifacts/native_eval_cost.json` |
| native C++ value+grad, `OMP=16` | 2.312 ms | `artifacts/native_eval_cost.json` |
| native C++ value+grad, `OMP=32` | 2.360 ms | `artifacts/probe_kernel_cpu_omp32.json` |
| JAX value+grad, **GPU** | **0.858 ms** | `artifacts/probe_kernel_gpu_cold.json` |
| JAX value+grad, CPU `OMP=32` | 18.434 ms | `artifacts/probe_kernel_cpu_omp32.json` |
| **`jacfwd` full 1025×135 Jacobian, GPU** | **28.518 ms** | `artifacts/probe_kernel_gpu_cold.json` |
| `jacfwd` full Jacobian, CPU `OMP=32` | 858.3 ms | `artifacts/probe_kernel_cpu_omp32.json` |
| thin SVD 1025×135, GPU | 9.556 ms | `artifacts/probe_kernel_gpu_cold.json` |

Derived (both inputs named in `artifacts/receipt_numbers.json`):

- GPU value+grad is **2.415×** faster than the best native value+grad. *The device wins the kernel.*
- One GPU Jacobian costs **33.23×** one GPU value+grad — the 135 forward tangents amortize almost not at all.
- One GPU Jacobian costs **13.76×** one best-native value+grad. **This is the whole verdict in one number.**

## 3. The native bar was chosen by measurement, not assumed

The mandate is to beat *properly-configured* native. Measured at `maxiter=600`,
two interleaved repeats each (`harness/sweep_omp.sh` runs one full cycle over all
12 configurations before starting the second), `legs/omp/*.json`
(`artifacts/sweep_omp.log`). **Solve seconds below are listed ascending, not in
chronological repeat order** — see the load-ramp note after the table:

| native config | solve s (2 repeats, sorted) |
|---|---|
| OMP=2, maxcor=20 | 2.808, 2.858 |
| OMP=4, maxcor=20 | 2.176, 2.211 |
| **OMP=8, maxcor=20** | **1.861, 1.982** |
| **OMP=16, maxcor=20** | **1.577, 1.974** |
| OMP=32, maxcor=20 | 1.980, 4.706 |
| OMP=48, maxcor=20 | 17.095, 20.486 |
| OMP=8, maxcor=300 | 3.472, 3.745 |
| OMP=32, maxcor=300 | 13.618, 19.253 |
| OMP=48, maxcor=300 | 34.249, 37.257 |

**Load ramp — why the two repeats are not interchangeable.** The sweep is
interleaved, but the load rose through repeat 1, peaked at the repeat boundary,
and fell back through repeat 2, so the two repeats sit in different load
regimes: repeat 1 ran at 1-minute loadavg **24.96–54.01**, repeat 2 at **53.91–75.54**
(`artifacts/receipt_numbers.json :: omp_sweep_load_ramp`). Configurations whose
two samples straddle that ramp therefore show large spreads that are an artifact
of *when* they ran rather than of the configuration — `omp32_mc20` is the clear
case (**r1 = 4.706 s at load 30.37, r2 = 1.980 s at load 54.96**, i.e. the
*slower* sample is the *less* loaded one, so the ramp is not a simple
contention effect and the pair should not be averaged). This is exactly why the
verdict in §4 rests on the separate 3-repeat final matrix and not on this sweep.

Ratio convention for the two statements below: **all-pairs**, i.e. every
OMP=32/48 sample divided by every OMP=8/16 `maxcor=20` sample (4 reference
samples), source `artifacts/receipt_numbers.json :: omp_slowdown_vs_omp8_16_mc20`.
The repeat-paired convention (r1÷r1, r2÷r2) is given alongside for contrast.

Two findings that bear on every prior native bar in this program:

1. **`OMP_NUM_THREADS=48` is catastrophic on this problem** — **8.63–12.99×**
   slower than OMP=8/16 all-pairs (9.19–10.84× repeat-paired) — and OMP=32 is
   **1.00–2.98×** slower all-pairs (1.06–2.38× repeat-paired). The C++ Biot–Savart kernel
   stops scaling at 8 threads here (2.073 ms at OMP=8 vs 2.360 ms at OMP=32), so
   everything above that is pure oversubscription overhead. The program's
   standing "OMP=32/48 for fair native" rule is *not* fair for a 135-DOF,
   1024-quadrature-point case.
2. **The shipped example's `maxcor: 300` is a misconfiguration for `n=135`.**
   With more correction pairs than dimensions, scipy's own L-BFGS-B linear
   algebra dominates: the as-shipped native run
   (`maxiter=300, maxcor=300, OMP=32`, `legs/pilot_native_omp32.json`) spends
   **5.896 s** to reach `J = 6.5747e-07` at `‖∇J‖_∞ = 1.189e-04` — which
   **fails the case's own 1e-4 success gate**. `OMP=16, maxcor=20` reaches that
   same objective in ≈1.4 s (bracketed by 1.212 s at `J ≤ 7e-07` and 1.682 s at
   `J ≤ 6e-07`, table below) — a **≈4×** speedup available in the native lane
   alone, from one option value.

The properly-configured native bar used below is therefore **OMP=8 or 16 with
`maxcor=20`**, with the mandated OMP=32/48 rows carried alongside.

Median solve-seconds to first reach `objective ≤ J*`, 3 interleaved repeats,
from the per-iteration traces (`artifacts/verdict_final.json ::
native_table_median_solve_s`):

| native cfg | 2.0e-6 | 1.5e-6 | 1.0e-6 | 8e-7 | 7e-7 | 6e-7 | 5e-7 | 4.5e-7 | 4.3e-7 | repeat spread @4.5e-7 |
|---|---|---|---|---|---|---|---|---|---|---|
| omp8_mc20 | 0.190 | 0.275 | 0.682 | 1.104 | 1.537 | 2.038 | 3.135 | 4.189 | 6.838 | 1.04× |
| **omp16_mc20** | 0.117 | 0.247 | 0.532 | 0.875 | 1.212 | 1.682 | 2.870 | 4.081 | 6.538 | 1.11× |
| omp8_mc300 | 0.180 | 0.245 | 0.462 | 0.751 | 1.079 | 1.726 | 3.424 | 6.142 | 9.751 | 1.36× |
| omp32_mc20 | 0.203 | 0.330 | 0.853 | 1.318 | 1.716 | 2.403 | 3.880 | 5.221 | 8.326 | 1.05× |
| omp32_mc300 | 0.184 | 0.580 | 2.763 | 4.537 | 5.526 | 8.632 | 16.064 | 25.685 | 39.483 | 1.16× |
| omp48_mc20 | 0.771 | 1.402 | 6.358 | 10.408 | 14.043 | 20.071 | 33.170 | — | — | — |

**Budget footnote on the `omp48_mc20` row.** Those legs ran **`maxiter=1200`**, not 2400 like every other row
(`artifacts/receipt_numbers.json :: omp48_mc20_budget`), because at ~36 s each
they were the most expensive legs in the matrix. Its "—" cells therefore mean
**the iteration budget ended** (per-repeat endpoints `J` = 4.851e-07 /
4.795e-07 / 4.766e-07; the deepest, 4.766e-07, is the informative bound), not
that the configuration cannot reach those rungs. Read that row as a lower bound on its own cost; it does
not affect the best-native column, which `omp48_mc20` never wins at any rung.

## 4. Verdict — GPU reformulation vs properly-configured native

Fixed-iteration comparison is invalid here (the lanes reach different answers at
equal iteration counts), so the verdict is stated as **wall-clock seconds to
first reach `objective ≤ J*`**. Native time-to-target comes from an exact
per-iteration callback trace inside one `maxiter=2400` run; the device-resident
LM cannot be probed mid-flight without destroying the property under test, so it
is swept over 13 budgets (10…600) and its time-to-target is the first budget
that reaches the rung — an **upper** bound on its true crossing time.

3 interleaved repeats, 57 legs, round-robin over the full lane list per repeat.
Sources: `legs/final/*.json`, reduced by `harness/verdict.py` into
`artifacts/verdict_final.json`.

**The "LM wall s" column is the warm-cache median, and that choice is charitable
to the GPU.** Eleven of the thirteen repeat-1 GPU legs paid a cold XLA compile —
median **5.530 s**, max 5.857 s over the 13 repeat-1 legs — against **1.367 s**
median for repeats 2–3 once the persistent cache was populated; the two
exceptions, `it80` (1.344 s) and `it300` (1.353 s), hit the cache warm even on
repeat 1. The cold/warm split moves process wall by up to **2.098×** across
repeats of the same budget
(`artifacts/receipt_numbers.json :: gpu_compile_cold_vs_warm`). Taking the median
of three therefore reports the GPU lane at its warm-cache best. A cold-start user
sees the wall-clock column at **0.11–0.32×** over the eight rungs with a cold
observation; the deepest rung (`4.3e-07`, budget `it300`) is the exception — its
repeat-1 leg was warm, so its cold-start figure equals the table's 0.50×. The
verdict does not change; at eight of nine rungs the reformulation loses by more,
not less, than the table shows.

| `J*` | best native cfg | native solve s | LM budget | LM solve s | **speedup (solve)** | native wall s | LM wall s | **speedup (wall)** |
|---|---|---|---|---|---|---|---|---|
| 2.0e-06 | omp16_mc20 | 0.117 | 10 | 0.466 | **0.25×** | 0.906 | 3.856 | 0.23× |
| 1.5e-06 | omp8_mc300 | 0.245 | 10 | 0.466 | **0.53×** | 1.030 | 3.856 | 0.27× |
| 1.0e-06 | omp8_mc300 | 0.462 | 10 | 0.466 | **0.99×** | 1.247 | 3.856 | 0.32× |
| 8.0e-07 | omp8_mc300 | 0.751 | 10 | 0.466 | **1.61×** | 1.536 | 3.856 | 0.40× |
| 7.0e-07 | omp8_mc300 | 1.079 | 60 | 2.328 | **0.46×** | 1.864 | 5.810 | 0.32× |
| 6.0e-07 | omp16_mc20 | 1.682 | 100 | 3.796 | **0.44×** | 2.470 | 7.485 | 0.33× |
| 5.0e-07 | omp16_mc20 | 2.870 | 150 | 5.649 | **0.51×** | 3.659 | 9.106 | 0.40× |
| 4.5e-07 | omp16_mc20 | 4.081 | 200 | 7.481 | **0.55×** | 4.870 | 10.888 | 0.45× |
| 4.3e-07 | omp16_mc20 | 6.538 | 300 | 11.072 | **0.59×** | 7.326 | 14.520 | 0.50× |

**`CLOSED_BOUNDED_NEGATIVE` at the properly-configured boundary.** The GPU
reformulation leads on the solve boundary at exactly one rung (`J* = 8.0e-07`,
1.61×, and that rung is flattered by the LM budget grid: its `it10` leg
overshoots to 7.734e-07), ties at 1.0e-06, and loses 0.25–0.59× everywhere
else. It never leads on process wall (0.23–0.50×), because ~1.3 s of warm
compile plus ~1.2 s of host setup sit in front of every GPU run.

**Against the mandated-but-suboptimal native configurations it "wins" — and that
win is an artifact of native misconfiguration, not of the reformulation.** Same
`artifacts/verdict_final.json`, median solve-s to `J* = 5.0e-07`:

| native configuration | solve s | GPU LM (`it150`) | ratio |
|---|---|---|---|
| omp16_mc20 (properly configured) | 2.870 | 5.649 | **0.51×** |
| omp32_mc20 (mandated) | 3.880 | 5.649 | 0.69× |
| omp32_mc300 (mandated + shipped maxcor) | 16.064 | 5.649 | **2.84×** |
| omp48_mc20 (mandated) | 33.170 | 5.649 | **5.87×** |

Any headline drawn from the bottom two rows would be false. It is recorded here
so it cannot be re-derived later as a positive result. The bottom row is the
weakest of the four in a second way as well: `omp48_mc20` is the **only**
configuration in the matrix whose repeats drift materially (**1.209×**, §8), so
its 33.170 s is the least stable number on this page and the "5.87×" built on it
should be treated as indicative, not measured to three digits.

**Same-formulation GPU control** (`legs/pilot/gpu_bfgs_control_r1.json`): the
device-resident `SIMSOPT_BFGS` driver over the identical scalar objective runs
300 iterations in **15.862 s** (52.87 ms/iteration) to `J = 8.4392e-07`. Since
one GPU value+grad is 0.858 ms, ~98% of that lane's per-iteration wall is
optimizer overhead, not physics. The device therefore does **not** win this case
by simply moving the shipped algorithm onto it either.

## 5. Physics equivalence

Bitwise parity is impossible by construction. Equivalence is established on the
converged physics, using the `QualityBand` pattern of
`examples/jax/parity/{arbiter,contracts}.py` (case-declared ceiling on an
endpoint observable, plus a recorded derivation). **Declared deviation:** that
comparator admits `budget_exhausted` lanes only when they share one matched
*iteration* budget; an LM outer iteration and an L-BFGS-B iteration are not the
same unit of work, so the matched boundary used here is *wall-clock at a declared
objective rung*. This substitution is deliberate and is why this receipt is
diagnostic rather than certifying.

**Declared band.** `final:objective ≤ 4.30e-07`.
*Derivation:* the deepest measured endpoints of the two lanes are
`4.184576e-07` (native `omp32_mc300_it2400_r3`) and `4.165418e-07`
(GPU LM `it600`, identical across all three repeats); the as-shipped native
endpoint `6.5747e-07` does **not** clear it, so the band is a real floor rather
than a rubber stamp.

**Hard gates**, taken from the case's own native success predicate in
`examples/jax/parity/cases/native_stage_two_optimization_minimal.py`. That
predicate is a conjunction of **four** clauses; the two quantitative ones are
tabulated below — `‖∇J‖_∞ ≤ 1e-4` and
`total_curve_length ≤ 1.1 × length_target` — and the two remaining clauses,
`np.isfinite(final:objective).all()` and `final:objective < initial:objective`,
are **also satisfied by both compared iterates** (objectives 4.19e-07 and
4.19e-07 against an initial 3.307e-02, `artifacts/probe_structure_cpu.json`).
Note that `18.0` is the `length_target` **configuration value**
(`NATIVE_DEFAULT["length_target"]`, mirrored from the case's
`_scale_configuration`), not a literal in the predicate; the gate is
`1.1 × length_target = 19.8`.

Compared pair, source `artifacts/equivalence_matched.json` — recomputed from
scratch through the **native C++** objective for both iterates, so the comparison
is not mediated by either lane's own evaluator. This is **a closely-matched
pair, not the closest**: over the 702 native × GPU-LM endpoint pairs in the final
matrix it ranks **4th** at 1.685e-04 relative (0.017%); the closest is
**1.342e-04**, `native_omp32_mc300_it2400_r1` vs `gpulm_eigh_it400`
(`artifacts/receipt_numbers.json :: equivalence_pair_ranking`). The pair below
was chosen before that ranking was computed; the two candidates differ by 25% in
match tightness on a quantity already agreeing to 1.7e-04, so no conclusion below
turns on the choice:

| observable | native `omp8_mc300_it2400_r3` | GPU LM `eigh it400` | agreement |
|---|---|---|---|
| objective | 4.188396e-07 | 4.187690e-07 | **1.685e-04** rel |
| squared flux | 4.188381e-07 | 4.187682e-07 | 1.668e-04 rel |
| max &#124;B·n&#124; | 1.021025e-03 | 1.025071e-03 | 3.96e-03 rel |
| RMS &#124;B·n&#124; over the 32×32 grid | 3.423489e-04 | 3.429348e-04 | 1.71e-03 rel |
| RMS of the pointwise B·n **difference** | — | — | 1.2276e-05 = **3.586%** of the field RMS |
| max pointwise B·n difference | — | — | 4.932e-05 (vs 1.02e-03 peak field) |
| total curve length | 18.0000017 | 18.0000013 | 4.5e-07 m |
| per-curve lengths, max difference | — | — | 5.373e-04 m (1.2e-04 rel) |
| `‖∇J‖_∞` | 4.732e-06 | 3.162e-06 | both **pass** the 1e-4 gate |
| length gate 19.8 | pass | pass | |

**Verdict on equivalence: PASS at the field level, with one honest caveat.**
Both iterates clear the declared band and both hard gates; the objective, the
squared flux, the peak normal field and the realized coil lengths agree to
≤ 4e-3 relative. But the **realized coil geometry does not** match: max pointwise
curve distance **0.1544 m**, mean **0.0447 m** over 16 coils × 100 quadrature
points (minor radius 0.5 m). Two visibly different coil sets produce the same
field on the surface to 3.6% RMS. That is the direct physical signature of the
24-dimensional Jacobian null space in §2 — the minimum is a flat manifold, not a
point — and it means *"physics-equivalent"* for this case must be asserted on the
field and on the objective, never on coil shapes.

**Basin confirmation** (`legs/basin/*.json`): the two lanes are not in different
basins. The warm-start sources are named precisely, because they are *not* the
converged endpoints tabulated above — the basin tests were run early, before
those endpoints existed:

- `B2_lm_from_native` starts from `legs/pilot_native_omp32.json`, the
  **as-shipped misconfigured native pilot** (`maxiter=300, maxcor=300`,
  `J = 6.5747e-07`, gradient gate failed) — not from the converged native answer.
  It lands at `4.3129e-07`.
- `B3_native_from_lm` starts from `legs/pilot/gpu_lm_reformulation_r1.json`, the
  **`gtol=1e-4` trap leg** of §7.2 (`J = 1.5431e-06`, length penalty inactive) —
  not from the converged LM answer. It lands at `4.5133e-07`.
- `B4`/`B5`, LM from cold with `lam0 = 1e0` and `lam0 = 1e-12`, land at
  `4.2840e-07` and `4.2872e-07`.

Because the two warm starts are the *worst* iterates either lane produced and
both still descend into the same valley as the cold-start runs, the basin
conclusion is if anything strengthened by the imprecise starting points: all five
paths converge to the same flat valley.

## 6. Structural explanation, tied to the two-regime law

- **Batch width of the reformulated Jacobian is `dim_x = 135`.** That is the
  quantity DESC makes wide; here it is fixed by the physics of a 4-base-coil,
  order-5 problem and cannot be grown. The GPU shows it: one 135-tangent
  `jacfwd` costs **33.23×** a single value+grad — essentially linear in tangent
  count, i.e. the device was already saturated at one tangent and the batch
  dimension buys nothing.
- **The reduction dimension is 1025** (1024 surface quadrature points), against
  GSCO's 1024-row win at far larger per-step work. The per-step work volume here
  is one Biot–Savart pass over 1024 × 16 coils × 100 quadrature points ≈ 1.64 M
  interactions — small enough that the launch floor dominates.
- **Factor-once-reuse works, and is not the bottleneck.** The damping ladder
  consumes 3.01 residual evaluations per outer iteration (903 evaluations over
  300 iterations, `legs/factor/eigh_it300.json`) against **one** factorization.
  Swapping the thin SVD of `J` for an `eigh` of the 135×135 Gram matrix saves
  **8.11 ms per iteration** (`legs/factor/svd_it300.json` vs `eigh_it300.json`:
  13.535 s → 11.103 s over 300 iterations). The two factorizations trace the same
  path: **identical evaluation counts** (903/903 residual, 300/300 Jacobian at
  `it300`; 244/244 and 80/80 at `it80`) and objectives agreeing to **2.6e-13**
  relative at `it300` and **1.9e-14** at `it80`
  (`artifacts/receipt_numbers.json :: factor_svd_vs_eigh_agreement`). They are
  *not* bit-identical — squaring the spectrum through the Gram matrix costs those
  last few ulps, as it must. Even after that improvement the factorization is
  ~4% of the 37.01 ms iteration; **77% is the Jacobian**.
- **The arithmetic bill is the verdict, and it closes to two significant
  figures.** Reaching `J ≈ 4.29e-07` takes the reformulation **300** outer
  iterations at **37.010 ms** each (`legs/factor/eigh_it300.json`) and the native
  lane **2400** iterations at **2.878 ms** each
  (`legs/final/native_omp16_mc20_it2400_r1.json`, 3631 function evaluations).
  The reformulation needs **8.0× fewer iterations** but each costs **12.86×**
  more wall — predicting 8.0 / 12.86 = **0.62×**, against the **0.59×** measured
  at the `4.3e-07` rung (the residual gap is that native crosses that rung at
  ~2270 iterations, not 2400). Nothing unexplained is left in the deficit. The
  Gauss–Newton convergence advantage is real but too small, because the
  rank-deficient κ(J)=1.3e16 Jacobian forces LM to crawl along the small singular
  directions exactly as L-BFGS-B does.
- **Consistent with the standing law**: narrow/sequential → CPU wins. This case
  is narrow (135) on *both* axes that matter — DOF count and per-step work —
  and it loses, which is the law's prediction, not an exception to it.

## 7. Side findings (each with its own artifact)

1. **The repository's own `SIMSOPT_LM_QR` route is not competitive on this
   residual.** `legs/shipped_lm/*.json`: both `Driver.SIMSOPT_LM_QR` and
   `Driver.SIMSOPT_LM_GMRES` stop after **10 iterations** on an `xtol`
   termination at `J = 3.5898e-06` with `‖∇J‖_∞ = 2.102e-04` — an objective
   **8.6× worse** than either lane above, and a gradient that **fails** the
   case's 1e-4 gate.

   | lane | **solve s** | **process wall s** |
   |---|---|---|
   | GPU LM_QR | 11.660 | 15.667 |
   | CPU LM_QR | 16.914 | 22.358 |
   | GPU LM_GMRES | 22.712 | 25.802 |
   | CPU LM_GMRES | 29.466 | 34.992 |

   (`artifacts/receipt_numbers.json :: shipped_lm_lanes`,
   `:: shipped_lm_process_wall_s`. An earlier draft labelled the solve column
   "Wall"; the two are ~4–6 s apart per lane and the derived ratios below are all
   solve-based.) That is **1.166 s per iteration** for a 1025×135 solve on GPU,
   ~31× the campaign LM's 37.01 ms; GPU beats CPU **1.45×** on the QR lane. The
   four lanes' objectives agree to **1.11e-14 absolute**, which on a 3.59e-06
   quantity is **3.09e-09 relative** (`:: shipped_lm_objective_spread`) — close,
   but not the near-machine-precision cross-platform identity the bare absolute
   figure suggests. This is a data point for the open LM_QR Phase-3 item — though
   not the experiment itself: Phase 3 names the production **Boozer** residual at
   300–600 columns, and this is a stage-two **flux** residual at 135 columns.
   What it shows is that at this width the pivoted-QR lane's problem is
   not only the kernel, it is also an `xtol` stop far from stationarity.
2. **A `gtol = 1e-4` stopping gate is a trap on this problem.** The first LM
   pilot (`legs/pilot/gpu_lm_reformulation_r1.json`) converged in **5 iterations
   / 0.323 s** to `‖∇J‖_∞ = 9.47e-06` — comfortably inside the case's own gate —
   but at `J = 1.5431e-06` with `total_curve_length = 17.535`, i.e. at a
   stationary point where the one-sided length penalty is *inactive* and
   contributes neither value nor Jacobian row. Reported against the shipped
   5.896 s native leg that reads as an **18× speedup**; it is not one. Tightening
   to `gtol = 1e-10` moved the same solver to `J = 4.2872e-07` in 13.787 s
   (`legs/basin/B1_lm_tight.json`). Any future stage-two LM lane must not use the
   case's success gate as its stopping gate.
3. **GPU LM endpoints are exactly reproducible across repeats.** Objectives at
   budgets 300 / 400 / 600 are identical to the last bit across all three
   interleaved repeats (`4.2871804164384375e-07`, `4.187690393164326e-07`,
   `4.1654184532969955e-07`).

## 8. Box discipline, contention, and discarded work

- **`boxstate.py` was rewritten, not copied.** The prior campaigns'
  `_is_baseline` classified GPU processes by substring-matching `"code"`, which
  matched every process launched from a path containing `/code/` — including this
  repo's own `.venv-qn-gpu` children (the incident disclosed in both 2026-08-16
  receipts). This campaign's version uses an **exact pid allowlist** captured at
  campaign start (`BASELINE_PIDS`, 6 desktop pids from `artifacts/gpu_start.txt`)
  and additionally excludes the leg's own pid. Every leg records
  `nvidia-smi --query-compute-apps` at entry and exit.
- **The box was not quiet, and this is disclosed rather than hidden.** An
  unrelated python job ran throughout. Quoting the only process record captured
  (`artifacts/gpu_contention_sample.txt`), the command line is
  `python -u - 6800 60` at **1651 %CPU = 16.5 cores**; that record carries **no
  cwd**, so this receipt does not name a script or project for it — an earlier
  draft asserted `bash scripts/run_trial.sh` under
  `~/code/fusion/fusion_equilibrium_challenge`, which the artifacts do not
  support, and the "10–16 cores" range had no source at all (16.5 is the single
  observation).
  Its device footprint differs between campaign phases and the two must not be
  merged: **530 MiB** during the pilots (`artifacts/probe_structure_cpu.json`,
  `legs/pilot_native_omp32.json`, pid 1913751) and **798 MiB** during the final
  matrix — 798 MiB in **101 of the 106** foreign-process observations across the
  114 box captures of the 57 final legs, median **798 MiB**
  (`artifacts/receipt_numbers.json :: foreign_job_final_matrix_mib`,
  `:: foreign_job_pilot_era_mib`). **53 of 57** final-matrix legs saw it as a GPU
  contender; peak 1-minute load average 68.37 (`artifacts/verdict_final.json`).
  Sampled GPU utilization while it ran was 3–4% at 76–78 W
  (`artifacts/gpu_contention_sample.txt`), so it held a context rather than
  computing — the GPU legs are believed clean, the native legs are the ones at
  risk. Mitigation: full round-robin interleaving and 3 repeats.
- **Measured native drift across the 3 interleaved repeats was 1.01–1.21×**,
  over all six configurations rather than the three quoted in an earlier draft
  (`artifacts/receipt_numbers.json :: native_repeat_drift_all_configs`; solve
  seconds listed **ascending, not chronologically**):
  `omp16_mc20` 6.899/6.906/6.974 (**1.011×**), `omp8_mc300` 15.859/15.949/16.155
  (**1.019×**), `omp32_mc20` 8.580/8.831/8.832 (**1.029×**), `omp32_mc300`
  69.568/70.440/72.378 (**1.040×**), `omp8_mc20` 6.841/6.992/7.180 (**1.049×**),
  and `omp48_mc20` 36.238/37.643/43.818 (**1.209×**). The conclusion is
  unchanged — all six sit well inside the program's recorded 53% batched-drift
  ceiling — but the outlier matters: `omp48_mc20` is the row that produces the
  "5.87× win" in §4, so that particular comparison rests on the least stable
  native number on this page.
- **Why the §3 OMP sweep is not the verdict instrument.** `harness/sweep_omp.sh`
  *is* interleaved — one full cycle over all 12 configurations before the second
  repeat begins, confirmed in `artifacts/sweep_omp.log` — so an earlier draft's
  claim that it was "un-interleaved" was simply wrong. Its 2.4× spread on
  `omp32_mc20` comes from the **load ramp** documented in §3 (repeat 1 at loadavg
  24.96–54.01, repeat 2 at 53.91–75.54), which interleaving spreads across
  configurations but cannot remove. The 3-repeat final matrix, run later at more
  stable load, is what the verdict rests on.
- **One leg was discarded and is retained as evidence**, not deleted:
  `artifacts/DISCARDED_pilot_jaxlm_gpu_fp32_cpu_env_bug.json`. zsh does not
  word-split unquoted parameters, so an env string passed as `env $GPUENV …`
  collapsed into a single assignment; the leg silently ran fp32-on-CPU. All
  subsequent legs are launched from Python with an explicitly constructed
  environment dict (`harness/run_matrix.py::CPU_ENV`/`GPU_ENV`), and every leg
  record carries its observed platform and `JAX_ENABLE_X64`.

## 9. Artifact index

Root: `/home/jungdaesuh/simsopt-campaigns/stage-two-minimal-coupled-20260816/`
(**436** files, 40 MB — `artifacts/receipt_numbers.json`, its generator
`harness/receipt_numbers.py`, and `artifacts/provenance_posthoc.txt` were
written after the measurement legs, which is
why an earlier draft counted 433).

| path | contents |
|---|---|
| `harness/problem.py` | shared native + JAX problem construction at `native_default` |
| `harness/gn_solver.py` | device-resident LM, single `lax.while_loop`, `svd`/`eigh` factor-once-reuse |
| `harness/run_leg.py` | one timed leg; lanes `native`, `jax-bfgs`, `jax-lbfgsb`, `jax-lm`, `jax-lm-qr`, `jax-lm-gmres` |
| `harness/boxstate.py` | exact-pid-allowlist box capture (replaces the substring-matching version) |
| `harness/{run_matrix,sweep_quality,final_matrix}.py`, `harness/sweep_omp.sh`, `harness/basin_test.sh` | interleaved launchers |
| `harness/{analyze,verdict,equivalence,receipt_numbers}.py` | reductions |
| `artifacts/probe_structure_cpu.json` | dimensions, exactness identities, singular spectrum |
| `artifacts/probe_kernel_{cpu_omp32,gpu_cold}.json` | per-kernel medians |
| `artifacts/native_eval_cost.json` | native value+grad at OMP=8/16 |
| `legs/omp/` (48 files), `artifacts/sweep_omp.log` | native OMP × maxcor sweep |
| `legs/sweep1/` (74 files), `artifacts/{sweep1.log,analysis_sweep1.json}` | first budget sweep |
| `legs/final/` (116 files), `artifacts/final_matrix.log` | **decisive interleaved matrix, 57 legs, 3 repeats** |
| `artifacts/verdict_final.json` | time-to-quality tables, drift, contention |
| `legs/basin/` (5 files) | basin / warm-start / damping-initialization tests |
| `legs/factor/` (8 files) | SVD vs eigh factorization comparison |
| `legs/shipped_lm/` (8 files) | `SIMSOPT_LM_QR` / `SIMSOPT_LM_GMRES` on GPU and CPU |
| `legs/pilot*/` (6 files) | pilots incl. the same-formulation GPU BFGS control |
| `artifacts/equivalence_matched.json` | physics-equivalence comparison |
| `artifacts/receipt_numbers.json` | **every headline numeral in this receipt with its source path** (71 entries) |
| `artifacts/provenance_posthoc.txt` | POST-HOC repo/interpreter state; **not** leg-time binding (see header) |
| `artifacts/gpu_start.txt`, `artifacts/gpu_contention_sample.txt` | box state |
| `artifacts/DISCARDED_*.json` | the retained invalid leg |

## 10. Recommended next action

**Close `stage_two_minimal` as a DESC-style reformulation candidate**
(`CLOSED_BOUNDED_NEGATIVE`: 0.25–0.59× on solve, 0.23–0.50× on process wall vs
properly-configured native at every quality rung but one) and, in the same
motion, **re-derive the program's native-configuration rule from the OMP sweep in
§3** — the standing "OMP=32/48 for fair native" convention was up to **13×** off
the measured optimum on this case, and any past bar set with it against a
narrow-DOF, small-quadrature example understates native by up to an order of
magnitude. That correction is worth more than any further work on this case.
The concrete first step: re-time the `native_default` GSCO-siblings and LM_QR
bars at `OMP_NUM_THREADS ∈ {4, 8, 16}` and check whether either bounded-negative
verdict was ever measured against a properly-threaded native lane.
