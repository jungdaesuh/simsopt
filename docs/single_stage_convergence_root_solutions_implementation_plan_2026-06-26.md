# Single-Stage Convergence: Root-Cause → Root-Solution Implementation Plan

> Created: 2026-06-26 · Source: GPD orchestrated root-cause analysis (3 `gpd-debugger`/`gpd-verifier`
> agents + runtime evidence, conflicts reconciled). Repo HEAD at analysis: `4704d5671`.
> Review-fix update: live source was rechecked at `2c71021b1`; source-backed corrections below
> supersede stale analysis details where the two differ.
> External-docs check: official JAX docs confirm that `jax_enable_x64` controls 64-bit defaults and
> availability (`jax-ml/jax` `docs/default_dtypes.md`), and that benchmark/runtime claims must account
> for device transfer, JIT compilation, and `block_until_ready()` timing (`docs/benchmarking.md`).
> Status: PLAN (no fixes applied yet).

## Purpose

Convert the verified root-cause → root-solution synthesis for the **single-stage banana
optimization's non-convergence** into an executable, reviewable plan. The triggering symptom was a
production run that ran but did not converge (SciPy L-BFGS-B `ABNORMAL_TERMINATION_IN_LNSRCH`). The
analysis established the failure is a **finite line-search stall** (not a NaN, not a core-JAX bug)
concentrated in the single-stage *outer-optimization driver*, plus a separate GPU *cold-compile*
blocker. This file is the SSOT for fixing those.

## Goals

- [ ] A real **converged** single-stage result on GPU: J monotonically decreasing, SciPy
      `status == 0`, ≥1 accepted L-BFGS step, `||grad||` decreasing, iota → target.
- [ ] The production single-stage gradient (operator/lstsq adjoint) **FD-certified at production
      scale** (mpol 8–10, κ≈3.9e5).
- [ ] Full-space lanes (`scipy-jax-decomposed` / `-fullgraph`) converge from feasible-but-marginal
      and infeasible-at-seed configurations (no `ABNORMAL` from the constraint cliff).
- [ ] Robustness: outer `ftol` matched to the objective noise floor; no rank-deficiency NaNs.
- [ ] The `newton_polish` host-materialization contract re-audited; stale contradiction wording retired
      if the current focused tests already coexist.

## Non-Goals

- Re-porting or "fixing" the **core JAX kernels** (`simsoptpp` C++ + `simsopt_jax*` BiotSavart/Boozer
  /adjoint) — verified machine-precision-faithful and broadly used (30 files); out of scope.
- Switching optimizer **library** (Optax / on-device / IPOPT) — the issue is not SciPy; the same
  SciPy L-BFGS-B converges on the reduced lane. A library swap would change the symptom, not the cause.
- Global multi-modality / local-minima search — separate concern, handled by continuation + multistart.
- fp32-on-GPU single-stage adjoint — physically unobtainable (κ·eps_f32), tracked elsewhere.

## Current Context (confirmed facts)

- **Empirical decisive datum** (rejected production run diagnostics, `prod_ss_fixval` rung-mpol10):
  `OPTIMIZER_FUN_FINITE=true`, `OPTIMIZER_JAC_FINITE=true`, `OPTIMIZER_INVALID_STATE=false`,
  `status=2` (`ABNORMAL_TERMINATION_IN_LNSRCH`), `nfev=21`. → L-BFGS-B evaluated 21 **finite**
  (value, grad) points and could not find an acceptable step. **Refutes the NaN hypothesis**; it is a
  genuine finite line-search failure.
- **Scope is single-stage-driver, not core JAX**: `success_filter` / `_traceable_rejected_objective_value`
  / `baseline_coil_gradient` appear only in the single-stage example + its adapter rejection path; the
  **Stage-2 JAX lane does not use any of it** (grep-confirmed). Core kernels are sound and broadly used.
- **The reduced lane already converges on CPU** (`scipy-jax`, 153 iters → iota ≈ target; the `rc=1`
  was a dimension-mismatch parity fail, not a convergence fail — `HANDOFF-ss-11-51-matrix.md`).
- **The per-eval GPU kernel is correct at high mpol** (m18/m36 `dJ` finite, C++-matching forward iota
  to 16 digits — `HANDOFF-mpol-homotopy-ladder-push.md`).
- **The c=64 dimension floor binds on the production operator/lstsq adjoint** and clears κ≈3.9e5 with
  ~10³–10⁴× margin (empirical κ-oracle, Agent C). NaN risk from conditioning alone is low.
- The full-space `scipy-jax-fullgraph` lane reaches the **same** `ABNORMAL`/status 2 (it is not a
  decomposed-only issue; both are full-space SciPy L-BFGS-B).

> File:line anchors below are from the analysis at HEAD `4704d5671`; the example/adapter files are
> large and line numbers drift — **re-confirm with grep at implementation time** (anchors given as
> symbol + approximate line).

## Rationale

- **Smooth barrier is the true root fix for the `ABNORMAL` (1a).** The rejected-step plateau has
  *exactly zero* slope, so no `ftol` change can rescue it — only restoring a real gradient can. This
  also moves back toward original simsopt's smooth-penalty design (which the banana driver deviated
  from by layering a hard gate on top of an already-smooth objective).
- **Certify before trusting (3).** The production operator/lstsq adjoint has never been FD-validated
  crossing the inner solve; "the optimizer ran and J dropped" is not proof the gradient is correct.
- **Reduced lane = fastest real converged GPU result.** It already converges on CPU and its GPU
  per-eval kernel is correct; the only blocker is a single slow cold compile, which is avoidable at
  reduced mpol.

## Assumptions (must hold; explicitly marked)

- **[A1 — UNVERIFIED for this seed]** The production-seed `ABNORMAL` is the same finite-stall
  mechanism (1a cliff and/or 1b noise) as the documented infeasible-seed cases. The 21 finite evals
  support this, but the per-trial line-search value/gradient trace for *this* seed has not been
  captured. Closing this is Phase 0.
- **[A2]** Objective J-value noise δJ ≈ 1e-10 worst-case (from homotopy floor data); per-seed value
  needs calibration before fixing `ftol` precisely.
- **[A3]** The smooth-barrier change can be made in the adapter rejection path + example filter
  without altering the core kernel numerics (the kernels are not touched).

## Implementation Plan

### Phase 0 — Confirm the mechanism for the production seed (closes A1)
1. Capture the per-trial line-search trace on the failing seed.
   - [ ] Re-run the rejected single-stage config with `--record-objective-evaluation-trace`
         (and without `--compact-objective-evaluation-trace` if replay-grade vectors are needed) →
         inspect `${OUT_DIR_ITER}/outer_optimizer_progress.json` objective-evaluation events for the
         21 outer trial `(x, f, g)` tuples. `record_scipy_callback_trace` / `result.scipy_callback_trace`
         is Boozer/adapter metadata, not the outer single-stage line-search trace.
   - [ ] Classify: all-finite + tiny-decrease/flat-plateau → 1a/1b confirmed (expected); any NaN →
         re-open the adjoint-NaN path (3). (Pre-registered prediction: finite + plateau.)
2. Calibrate δJ for the seed (closes A2).
   - [ ] Two inner solves at `newton_tol=1e-11` vs `1e-13`, hold coils fixed, diff J → measured δJ.

### Phase 1 — FD-certify the production gradient (Root Cause 3)
1. Add a production-scale finite-difference gate for the **fused operator/lstsq** single-stage gradient.
   - [ ] New test (e.g. `tests/integration/test_single_stage_production_gradient_fd.py`): build the
         real fixture at mpol 8, converge inner Boozer, take K=3–5 random unit coil-DOF directions,
         central differences vs adjoint directional derivative, **re-solving inner Boozer per probe**.
   - [ ] Use eps in **[6e-4, 5e-3]** (truncation-limited window given δJ≈1e-11, |J|≈0.18; below ~6e-4
         it is noise-limited — the existing `(4e-4,2e-4,1e-4)` ladder is INVALID at production noise).
         Recommended ladder `(3e-3, 1.5e-3, 7.5e-4)`; require Taylor-rate decrease + abs-tol ≈ 1e-6.
   - [ ] Confirm the test exercises the operator/lstsq path (`linear_solve_factors=None`,
         `surface_objectives_traceable.py` ~607-615 → `optimizer.py` `_solve_dense_square_operator_least_squares_system_with_status` ~4792), NOT the eager dense-PLU sibling the toy tests use.

### Phase 2 — Get a real converged GPU result via the reduced lane (ranked #1)
1. Run `--optimizer-backend scipy-jax` (reduced) on GPU at **mpol ≤ 6** (compiles feasibly).
   - [ ] Persistent compile cache on a **network volume** (`JAX_COMPILATION_CACHE_DIR=/workspace/...`,
         not `/tmp`); confirm warm-cache hit on a second run.
   - [ ] Record the convergence table (see Validation). Gate acceptance on the Phase-1 FD cert.
   - [ ] (No code change expected for this step — config/run only.)

### Phase 3 — Smooth the hardware penalty (Root Cause 1a) — TRUE ROOT FIX
1. Convert the hard hardware feasibility gate to a constraint-residual contract.
   - [ ] Add a single-stage hardware constraint evaluator beside `success_filter`
         (`single_stage_banana_example.py` ~7728-7800) that returns the four positive-when-violating
         residuals:
         `cc_dist - curve_curve_min_dist`, `cs_dist - curve_surface_min_dist`,
         `ss_dist - surface_vessel_min_dist`, and `max_curvature - curvature_threshold`.
         Keep the existing boolean predicate as a derived feasibility value, or update every
         `success_filter` callsite/cache/test in one coherent contract change. Do not silently make a
         bool-typed `success_filter` return an array.
   - [ ] Keep self-intersection as a separate hard predicate unless a differentiable surrogate is
         explicitly designed; do not fold it into the smooth hardware barrier by accident.
2. Replace the flat plateau + frozen gradient with a smooth exterior barrier.
   - [ ] `_traceable_rejected_objective_value` (`surface_objectives.py` ~1038-1048): drop the
         `stop_gradient` on the candidate; return `objective + Σ_k w_k·max(0, violation_k)²` (C¹, real
         gradient on the infeasible side). The residual sign matters: distance constraints are
         lower bounds, curvature is an upper bound, so a uniform `metric - threshold` margin would
         penalize safe distance slack and miss unsafe distance violations.
   - [ ] Consumer (`surface_objectives_traceable.py` ~718-739): feed margins into the smooth penalty
         instead of the boolean `lax.cond`.
   - [ ] Return the **true point-dependent gradient** on rejection, not the frozen baseline:
         `surface_objectives_traceable.py` ~3640-3641 and the decomposed fallback
         `single_stage_banana_example.py` ~9714-9722.
   - [ ] Update the existing rejection-path tests that currently pin baseline-gradient behavior
         (e.g. `tests/geo/test_surface_objectives_jax.py` and
         `tests/integration/test_single_stage_jax_cpu_reference.py`) to assert the new
         point-dependent hardware-barrier gradient.
3. Preserve genuine inner-solve-failure handling.
   - [ ] Keep a (smooth, informative-gradient) penalty for true Boozer-solve failure (distinct from
         hardware-margin violation); do NOT reintroduce a flat plateau.

### Phase 4 — Robustness hardening (Root Causes 1b + 3 residual)
1. Match outer `ftol` to the noise floor (1b).
   - [ ] Floor `ftol_by_mpol` (`single_stage_banana_example.py` ~14328-14340) at ≥ 1e-8 for high mpol
         (`ftol ≥ ~100·δJ`), or clamp at the resolution site (~15629-15633). Optionally tighten inner
         `newton_tol` 1e-11→1e-13 (knob already plumbed: `--target-lane-boozer-newton-tol`) to lower δJ.
2. Guard the rank-deficiency NaN (3 residual).
   - [ ] First add a regression/probe proving the current `_dense_matrix_solve_numerically_safe`
         misses a near-singular float64 production iterate. The live code already applies a
         dtype-specific condition screen for float64 and deliberately keeps the forward-error bound
         fp32-only because the float64 forward-error bound can false-reject large production solves
         (`optimizer.py` ~4554-4607). Do **not** extend the forward-error gate to float64 without a
         new repro. If a repro exists, choose the smallest fail-closed fix (for example a tiny relative
         `newton_stab` floor or a dimension-aware condition criterion) and re-run the Phase-1 FD cert.

### Phase 5 — Production-scale GPU compile (Root Cause 2, mpol10)
1. Measure before narrowing.
   - [ ] Run `benchmarks/compile_breadth_probe.py` at mpol 6/8/10 under `JAX_LOG_COMPILES=1` +
         `XLA_FLAGS=--xla_dump_to=...` → identify the dominant sub-kernel (K1 forward vs K2 adjoint).
2. Narrow the dominant kernel (only what the probe implicates).
   - [ ] Promote the decomposed host-split (removes the outer `lax.cond`,
         `surface_objectives_traceable.py` ~1417); and/or make the dense-adjoint gate breadth-aware
         (`optimizer.py` `_dense_square_operator_materialization_allowed` ~4746) so surface-sized
         adjoints default to operator-GMRES without an env var.
   - [ ] Add the fused `_value_and_grad_for` to `CALLBACK_FREE_TARGETS` (persistent-cache coverage),
         confirm via `benchmarks/check_cached_kernel_callback_compatibility.py`.

### Phase 6 — Re-audit the `newton_polish` host-materialization contract (Root Cause 4)
1. Verify whether the historical contradiction still exists.
   - [ ] Run the focused contract slice:
         `/opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest tests/geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_newton_polish_host_control_uses_host_dense_materialization tests/geo/test_optimizer_jax_item19.py::test_item19_host_dense_hessian_reuses_chunked_device_materializer tests/geo/test_optimizer_jax_item19.py::test_item19_host_dense_hessian_agrees_with_device_materializer -q`
   - [ ] Current review result at `2c71021b1`: `1 failed, 2 passed`. The failing node is
         `test_newton_polish_host_control_uses_host_dense_materialization`; it monkeypatches
         `_materialize_dense_hessian` to prove host control uses an independent host materializer, but
         `_materialize_dense_hessian_host` currently delegates back into `_materialize_dense_hessian`.
         The two item19 tests pass, so the live contradiction is specifically about whether
         `allow_host_control=True` is allowed to reuse the chunked device materializer.
   - [ ] If it fails, then make the product decision explicitly: keep a true host materializer for
         `allow_host_control`, or route host control through the chunked device materializer. Update
         both tests and comments to assert that single contract.

## Validation Plan

- [ ] **Convergence table** (the primary acceptance artifact), from the run progress JSON:
      | outer iter | J | ‖grad‖∞ | accepted? | status |
      Pass = J monotone ↓, ≥1 accepted step, `status==0`, `‖grad‖` ↓ to `gtol`.
- [ ] **A/B control**: same seed pre-fix (`status=2, nfev=21, 0 accepted`) vs post-fix (`status=0`).
- [ ] **Phase-3 proof**: on a constraint-marginal seed, the accepted-step gradient is
      point-dependent (differs from the frozen `baseline_coil_gradient`).
- [ ] **FD cert** (Phase 1) passes at mpol 8 with the [6e-4, 5e-3] eps window.
- [ ] **Regression — core untouched**: Stage-2 JAX lane + BiotSavart/Boozer parity tests stay green
      (the smooth-penalty edit touches the adapter rejection path; prove no Stage-2/kernel impact).
- [ ] **Compile probe** (Phase 5) emits a committed results JSON before any narrowing PR.
- [ ] Full JAX-port test suite green under the repo interpreter
      (`/opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest`, `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`).

## Risks and Mitigations

- Risk: Smooth-barrier weights `w_k` distort the physical optimum near the constraint boundary.
  Mitigation: short weight sweep; compare converged config + constraint activity to the CPU-converged
  reference; require constraints satisfied at the optimum.
- Risk: The 1a edit lives in the **core adapter** (`surface_objectives*.py`) shared by the traceable
  machinery, so it could perturb Stage-2 / other traceable objectives.
  Mitigation: gate behind the single-stage rejection path; run the Stage-2 + parity regression
  (Validation) before/after; the path is grep-confirmed single-stage-only today — keep it that way.
- Risk: Tightening inner `newton_tol` (1b) raises per-eval cost.
  Mitigation: prefer raising `ftol` first (cheap); tighten inner tol only if mpol18-level precision is
  needed; note the bfgs override is hard-capped at 1e-8 (`single_stage_banana_example.py` ~11170/11221).
- Risk: `newton_stab > 0` perturbs the adjoint solution.
  Mitigation: use a tiny *relative* floor only as a NaN guard; verify the FD cert still passes with it on.
- Risk: Phase-5 narrowing rewrites the non-dominant kernel (wasted effort / regression).
  Mitigation: measurement-first — do not narrow without the `compile_breadth_probe.py` result.

## Completion Criteria

- [ ] Converged GPU single-stage result with `status==0`, FD-certified gradient (Phases 1+2).
- [ ] A full-space lane (`scipy-jax-decomposed`) converges from a marginal/infeasible seed after the
      smooth-barrier fix (Phase 3).
- [ ] `ftol`/noise budget + rank-deficiency guard landed and validated (Phase 4).
- [ ] `newton_polish` host-materialization contract audited; focused tests agree and suite is green
      (Phase 6).
- [ ] No regression in Stage-2 / kernel parity tests; core JAX untouched.

## Open Questions

- **[Conditional decision]** Phase 6 only needs a product decision if the focused contract tests fail.
  If they pass, retire the contradiction as stale analysis and keep the current true host Hessian path.
- **[SSOT]** Should the smooth penalty live in the example driver or be promoted into the adapter as
  the canonical constraint handling? (Affects whether other future single-stage drivers inherit it.)
- **[Data]** Per-seed δJ (Phase 0.2) and the mpol10 dominant-kernel breadth (Phase 5.1) — both require
  one short runtime measurement each before the dependent fix is finalized.
- Is a converged result wanted at **production mpol10** (needs Phase 5) or is mpol≤6 sufficient for
  the immediate milestone (Phase 2 only)?
