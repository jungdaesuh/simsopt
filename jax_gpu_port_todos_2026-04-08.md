# simsopt-jax GPU Port — Validation TODO List

**Audit date:** 2026-04-08
**Branch:** `gpu-purity-stage2-20260405`
**Source:** Deep-dive validation via 7 parallel audit agents (JIT/vmap/lax, device residency, autodiff/IFT, GPU/sharding, private optimizer, test coverage, docs) + direct verification.
**Scope:** 60 actionable items — 3 ship blockers · 8 correctness/defensive · 11 transfer-guard · 15 performance · 12 test coverage · 11 docs/cleanup.

> Note: Agent 1's "HIGH severity Python-loop recompilation" finding was **rebutted** by [jax-ml/jax#16611](https://github.com/jax-ml/jax/issues/16611) — JAX maintainer Jake VanderPlas confirmed unrolled Python for-loops are ~16× faster than `lax.scan` for small trip counts because XLA fuses across iterations. The existing `_grouped_field` and `_accumulate_grouped_field` designs are correct and are **not** on this list.

---

## Tier 0 — Ship blockers (must fix before any "production GPU mode" claim)

- [x] **1.** ~~Fix `test_pure_objective_matches_optimizable_value` at `tests/integration/test_single_stage_jax_cpu_reference.py:5358-5375`. M5 `f(coil_dofs)` diverges from `JF_jax.J()` beyond `rtol=1e-10`.~~ **DONE** — test passes at rtol=1e-10 (now at line 5818).
- [x] **2.** ~~Fix `test_boozersurface_jax.py:753` — "DID NOT RAISE RuntimeError" strict-mode enforcement gap.~~ **DONE** — strict JAX-backend rejection covers the remaining reference methods (`adam` / `bfgs` / `lbfgs`), and removed `bfgs-hybrid` now has its own absent-from-surface regression.
- [x] **3.** ~~Fix `test_lbfgs_reduces_objective` at `tests/geo/test_boozersurface_jax.py:745-752`.~~ **DONE** — sign fix `q = -jnp.conj(state.g_k)` at `_lbfgs.py:60`; test passes (now at line 938).

---

## Tier 1 — Correctness / defensive bugs (real but low-probability)

- [x] **4.** **[MED]** ~~`optimizer_jax_private/_lbfgs.py:200` — add finite-guard on `rho_k`.~~ **DONE** — `valid_curvature` guard at lines 245-253 checks `jnp.isfinite(rho_k)` & `(rho_k_inv > curvature_tol)`; gates history update via `update_curvature = valid_curvature & (~stalled_step)`.
- [x] **5.** **[MED]** ~~`_lbfgs.py:201` — clamp `gamma`.~~ **DONE** — `jnp.clip(gamma, step_eps, gamma_max)` at lines 214-218 with machine-precision bounds from `_lbfgs_step_tolerances()`.
- [x] **6.** **[MED]** ~~`_lbfgs.py` body_fun — port BFGS stalled-step check.~~ **DONE** — multi-condition stalled-step check at lines 237-242 (`s_k_norm <= step_tol`, `function_change <= objective_tol`, `gradient_change <= gradient_tol`).
- [x] **7.** **[LOW]** ~~`_bfgs.py:122-130` — add Powell damping or skip-if-negative-curvature to BFGS Hessian update. Strong Wolfe should prevent this in normal operation, but defensive hardening matters for edge cases.~~ **DONE** — dense BFGS now skips the Hessian update on non-finite or non-positive curvature using the same `curvature_tol`-style guard pattern as L-BFGS, leaving `state.H_k` unchanged on bad curvature.
- [x] **8.** **[LOW]** ~~`optimizer_jax.py:2161-2166` — hybrid scipy→on-device continuation has no fallback when scipy produces non-finite state. Currently returns `success=False` silently; should log or retry with tighter scipy tolerance.~~ **DONE** — non-continuable SciPy prefixes now emit `logger.warning(...)` with `success/nit/fun/grad_inf` and forward one last `progress_callback(...)` snapshot before returning failure.
- [x] **9.** **[LOW]** ~~`surfaceobjectives_jax.py:271-275` — add runtime signature-check for VJP callback at `run_code` result construction. Wrong signatures currently only surface deep in the gradient pass.~~ **DONE** — `BoozerSurfaceJAX` now validates result-dict VJP hook arity via `_require_boozer_vjp_callback_signature()` / `_prepare_result_callback()` at result construction time; covered by `test_run_code_rejects_bad_group_vjp_signature`.
- [x] **10.** **[LOW]** ~~`boozersurface_jax.py:733-756` — `_build_ls_group_vjp_callback` closes over `booz_surf` state. Add a solver-generation counter and assert freshness when VJP is invoked so stale reuse is detected.~~ **DONE** — solver-generation freshness guard lives in `_guard_solver_callback_freshness()` and grouped-LS stale reuse is covered by `test_ls_group_vjp_detects_stale_reuse_after_resolve`.
- [x] **11.** **[LOW]** ~~`surfaceobjectives_jax.py:678-694` (`_compute_dJ_ds`) — `_ensure_solved` at line 418 checks `res["success"]` but not the actual residual norm. Log final `‖grad‖` / residual norm alongside the success flag.~~ **DONE** — `_ensure_solved()` now logs cached solve quality with `success`, `grad_inf`, and `residual_inf`, distinguishing exact-path residuals from true gradient norms.

---

## Tier 2 — Transfer-guard hardening (path to `disallow` baseline)

### Real runtime transfers (MB/s-class, must fix before strict mode)

- [x] **12.** **[HIGH]** ~~`field/force.py` — `jnp.asarray(gammas_targets/gammadashs_targets/currents_*)` per-call conversions.~~ **DONE** — refactored to `_as_jax_float64()` throughout; `_prepare_target_source_inputs_pure()` and `_CoilStateGroupCache` handle all conversions.
- [x] **13.** **[HIGH]** ~~`field/force.py` — `jnp.asarray(opt.full_x)` / `jnp.asarray(coil.full_x)` per-call conversions.~~ **DONE** — now uses `_as_jax_float64()` at lines 329, 618.
- [x] **14.** **[MED]** ~~`field/force.py:520,525` — `jnp.asarray(symmetry.rotmat/scale)` per-coil conversions.~~ **DONE** — `_apply_coil_state_symmetry()` uses `_as_jax_float64()` at construction time (lines 528, 533).
- [x] **15.** **[MED]** ~~Re-scope the `geo/optimizer_jax.py` SciPy-boundary cleanup.~~ **DONE** — split the optimizer contract into explicit reference-vs-target lanes, removed public `hybrid` / `bfgs-hybrid` routing, moved SciPy host adapters into `geo/optimizer_jax_reference.py`, updated Boozer / Stage 2 / single-stage call sites to use lane-specific wrappers, and added regression coverage that `backend="jax"` flows cannot enter the SciPy adapter path.
- [x] **16.** **[MED]** ~~`jax_core/objectives_flux.py:86` — remove fallback `jnp.asarray(surface.gamma()), jnp.asarray(surface.normal())` when no `surface_spec()` is available. Enforce spec with a clear error.~~ **DONE** — `SquaredFluxJAX` now requires `surface_spec()` and raises a clear contract error instead of falling back to `surface.gamma()/normal()`.
- [x] **17.** **[LOW-MED]** ~~`geo/curveperturbed.py:195-196` — `jnp.asarray(self.sample[0/1])` in `__init__`. Replace with `_explicit_device_array()` for `disallow` compliance.~~ **DONE** — `CurvePerturbed` now materializes sampled perturbations with `_explicit_device_array(..., dtype=np.float64)`.
- [x] **18.** **[LOW-MED]** ~~`jax_core/curve_geometry.py:48-53` — `_as_explicit_float64` numpy fallback path. Tighten input contract to specs only.~~ **DONE** — `_as_explicit_float64()` no longer accepts raw host NumPy inputs without a runtime/spec reference; intended host entry points use the explicit referenced conversion path.

### Cosmetic under `disallow` (zero runtime cost, trace-time only, but currently flagged)

- [x] **19.** **[LOW]** ~~`jax_core/biotsavart.py:46, 87-88, 387` — `_float64_scalar(_MU0_OVER_4PI)` is constant-folded at trace time but trips `transfer_guard=disallow`. Replace with `_device_scalars.device_one(reference) * 1e-7` idiom from `_device_scalars.py`.~~ **DONE** — `_float64_scalar(reference, value)` now uses `_device_scalars.device_one(reference) * value`, eliminating the trace-time host transfer path.
- [x] **20.** **[LOW]** ~~`jax_core/biotsavart.py:91-102` — `_as_int32_scalar`, `_index_range`, `_zero_scalar` use `jax.device_put(np.asarray(...))` at trace time. Zero runtime cost but flagged by `disallow`. Switch to pure `jnp.arange`/`jnp.zeros` inside the traced scope.~~ **DONE** — these helpers now stay inside JAX creation ops (`jnp.asarray(..., dtype=jnp.int32)`, `jnp.arange(...)`, `jnp.zeros((), ...)`) and no longer trip `disallow`.
- [x] **21.** **[LOW]** ~~`geo/surfaceobjectives_jax.py:107` — `_explicit_index_array` uses `jax.device_put` at trace time.~~ **DONE** — intentional one-time at spec construction; acceptable under current contract.
- [x] **22.** **[LOW]** ~~`jax_core/specs.py:463-616` — 20+ `_as_float64_array` calls at spec construction.~~ **DONE** — all calls live in immutable spec factory functions (`make_coil_group_spec`, `make_curve_xyzfourier_spec`, etc.), not hot loops.

---

## Tier 3 — Performance opportunities

### Quick wins

- [x] **23.** **[PERF 2-3%]** ~~`jax_core/biotsavart.py:387` — add `precision=lax.Precision.HIGHEST` to `jnp.einsum("c,cj->j", ...)`. Matches private optimizer convention at `_common.py:23-24`.~~ **DONE** — the final Biot-Savart current contraction now uses `jnp.einsum(..., precision=lax.Precision.HIGHEST)` in the hot path.
- [ ] **24.** **[PERF 5-10% VRAM?]** Re-scope Biot-Savart buffer donation to a real `jax.jit` boundary instead of the internal `fori_loop` carries (`biotsavart.py`). JAX `donate_argnums` applies at `jit` / `pjit` / `pmap` call boundaries, not directly to `_coil_chunk_reduce`, `_quadrature_block_integral`, or `_point_chunk_reduce`. Find an outer compiled entry point with same-shape input/output pytrees, add donation there, and keep the item only if peak-memory profiling shows a real win. **(PARTIAL — added `benchmarks/biotsavart_donation_probe.py` plus `tests/test_biotsavart_donation_probe.py` to measure an outer-`jax.jit(donate_argnums=(0,))` wrapper on disposable `points` buffers while keeping the public API contract unchanged; local CPU probe matched baseline numerically, but CUDA VRAM benefit is still unverified.)**
- [ ] **25.** **[EXPERIMENT / PERF variable]** `jax_core/_math_utils.py:107-121` — evaluate replacing `explicit_rsqrt` custom JVP with `jax.lax.rsqrt` + default JVP. `lax.rsqrt` is the direct primitive, but this should remain blocked on the explicit parity gate: require CPU/GPU objective and gradient parity across the Biot-Savart operating range before any swap. If parity fails, delete the item instead of weakening the contract.
- [ ] **26.** **[EXPERIMENT / PERF 10-20%?]** `surfaceobjectives_jax.py:268` — investigate whether LS-only Boozer adjoint / warm-start solves can skip iterative refinement behind a measured heuristic. Do **not** gate blindly on dense `cond(...)` unless profiling shows the estimator is cheaper than the refinement it suppresses. Keep iterative refinement as the default fallback for exact mode and for any inconclusive LS case. **Current state:** project docs/tests still justify `iterative_refinement=True` as the stable default for dense Boozer PLU solves.

### Medium effort

- [ ] **27.** **[EXPERIMENT / PERF 40-60%?]** Re-scope LS grouped-VJP optimization in `boozersurface_jax.py`. The naive proposal to precompute `B_shared = grouped_biot_savart_B_from_spec(...)` once and reuse it across group runners is **not valid**: the grouped callback is differentiated through the surface-point geometry, so freezing `B(points)` would drop `dB/dX` terms, and routing through the full grouped-VJP helper also breaks the streaming-memory contract used by the grouped-adjoint probes. Any future optimization must preserve per-group streaming behavior and point-derivative correctness. **(PARTIAL — review confirmed the original item was wrong as written; added a regression guard in `tests/geo/test_boozersurface_jax.py` that `vjp_groups` must not route through `_boozer_ls_coil_vjp`.)**
- [ ] **28.** **[PERF 15-25%]** `boozersurface_jax.py:413-423` — `_surface_geometry_from_dofs` computes gamma/xphi/xtheta separately. Fuse into a single JAX primitive for memory locality. Called thousands of times per outer solve.
- [ ] **29.** **[PERF 40%]** `optimizer_jax.py:1360-1363, 1375-1377` — `_materialize_dense_hessian` does full-column HVPs. For LS Hessians (J^T J, SPD), compute only the upper triangle and mirror.
- [x] **30.** **[PERF 25-40%]** ~~`surfaceobjectives_jax.py:663-675` — BoozerResidualJAX/IotasJAX/NonQuasiSymmetricRatioJAX all solve `(PLU)ᵀ adj_i = rhs_i` with the same PLU. Batch via `jax.vmap(solve_triangular)`. Check JAX 0.9.3+ for native batched triangular solve.~~ **DONE** — added `compute_standard_surface_objective_gradients(...)` in `surfaceobjectives_jax.py`, which batches the standard LS wrapper trio through one shared `jax.vmap(_solve_boozer_adjoint)` pass while preserving the public `dJ()` contract; covered by matrix-RHS solve parity and reduced-real wrapper-gradient integration tests.
- [ ] **31.** **[PERF]** `_lbfgs.py:28-34` — replace `_shift_history` slice+concatenate (~200k element copies/step at `maxcor=200, d=1000`) with a ring-buffer + head-pointer. Requires rewriting two-loop recursion indexing.
- [ ] **32.** **[PERF small]** `_line_search.py:283` — dead re-evaluation path: BFGS/L-BFGS always pass `state.f_k`, but line search re-evaluates `restricted_func_and_grad(zero)` when `old_fval=None`. Remove.
- [ ] **33.** **[PERF]** Line-search bracketing and zoom don't share intermediate evals. Cache bracketing-phase `(α, φ, φ')` samples for zoom reuse. Saves 10-30 evals/iteration in worst cases.
- [ ] **34.** **[PERF compile 10-15%]** Default `SIMSOPT_JAX_COMPILATION_CACHE_DIR` to `~/.cache/simsopt-jax-xla/` on first run instead of requiring manual setup. Gate already plumbed at `runtime.py:1380-1381`. **(PARTIAL — env var defined and wired, but `_default_compilation_cache_dir` returns None; no auto-default)**

### Larger structural

- [ ] **35.** **[PERF 2-4× on 4+ GPUs]** Extend `jax_core/sharding.py` to support multi-GPU collective reductions inside the Biot-Savart kernel. Currently only "replicated coils / sharded points" — no cross-device reduction primitive.
- [x] **36.** **[PERF / OOM]** ~~`optimizer_jax.py:1744,1831` — exact Newton Jacobian OOM.~~ **DONE** — Newton iterations are matrix-free (JVPs via GMRES); dense materialization only at final iterate with `max_dense_jacobian_bytes` policy cap. Documented in docstring at line 1762.
- [ ] **37.** **[PERF]** `runtime.py:127-131` — GPU reproducibility settings (`gpu_reduction_order_max_ulp`, `gpu_reproducibility_seed`, etc.) are policy metadata only, not applied to kernel execution. Either wire into kernels or document as "contract probe only". **(PARTIAL — lines 299-301 document these as "reporting/acceptance metadata…do not force deterministic GPU behavior"; not wired to kernels)**

---

## Tier 4 — Test coverage

### Tier 1 (required for 1.0 release)

- [x] **38.** ~~Add `test_boozer_residual_jax_gpu` — full LS solve with `BoozerResidualJAX` / `IotasJAX` / `NonQuasiSymmetricRatioJAX` on CUDA with `transfer_guard=disallow`.~~ **DONE** — `TestRealFixtureGpuM5Parity::test_real_fixture_gpu_wrapper_values_and_gradients_match_cpu_reference` exercises the reduced real LS solve and M5 wrapper value/gradient parity on CUDA with `SIMSOPT_JAX_TRANSFER_GUARD=disallow`, alongside `test_real_fixture_gpu_solver_stays_ondevice_under_disallow`; validated on Runpod RTX 4090 on 2026-04-13.

### Tier 2 (required for production use)

- [x] **39.** ~~Add `TestRunCodeLSParityProductionScale` — `nphi=16, ntheta=8, ncoils=4` fixture (current `TestRunCodeLSParity` uses `nphi=5, ntheta=5, ncoils=2` = 25 points, 1-2 orders below production).~~ **DONE** — `TestRunCodeLSParity::test_ls_solve_parity_production_scale` covers the larger CPU-vs-JAX LS fixture, and `test_ls_solve_parity_production_scale_gpu_under_disallow` adds the strict CUDA `transfer_guard=disallow` lane with on-device solver-state assertions; validated on Runpod RTX 4090 on 2026-04-13.
- [x] **40.** ~~Add full single-stage **outer-loop convergence** test on GPU (not just init-parity at `benchmarks/single_stage_init_parity.py`). Verify IFT adjoint decreases the objective over ≥10 outer iterations on CUDA.~~ **DONE** — `tests/integration/test_single_stage_physics_parity.py::TestSingleStageOuterLoopGpuProof` now validates the real CUDA proof path under `SIMSOPT_BACKEND_MODE=jax_gpu_parity`, `SIMSOPT_BACKEND_STRICT=1`, and `SIMSOPT_JAX_TRANSFER_GUARD=disallow`; passed on Runpod RTX 4090 on 2026-04-13. Local proof path: `benchmarks/validation_ladder_contract.py`, `benchmarks/single_stage_outer_loop_probe.py`, `tests/integration/test_single_stage_physics_parity.py::TestSingleStageOuterLoopGpuProof`, `.github/workflows/jax_smoke.yml`, `.github/workflows/jax_gpu_parity.yml`.
- [x] **41.** ~~Add XLA recompilation-count smoke test. Track compilation counter across an optimizer loop; fail if compiles > expected per iteration. Protects against shape-dependent recompile regressions.~~ **DONE** — the existing subprocess compile-reuse harness now covers both the real target-lane value/grad path and the real Stage 2 target outer-loop path via `tests/test_jax_import_smoke.py::test_target_lbfgs_ondevice_reuses_compiled_solver_across_identical_value_and_grad_calls` and `tests/test_jax_import_smoke.py::test_stage2_target_outer_loop_reuses_compiled_solver_across_identical_calls`, backed by `tests/subprocess/jax_runtime_cases.py` cases `target-compile-count` and `stage2-target-compile-count`; validated locally on 2026-04-13 with `JAX_ENABLE_COMPILATION_CACHE=0`.
- [x] **42.** ~~Add CI test that runs `SIMSOPT_JAX_TRANSFER_GUARD=disallow` against the **full** suite on GPU (not just the e2e smoke at `jax_smoke.yml:257-318`).~~ **DONE** — `.github/workflows/jax_gpu_parity.yml` now includes `gpu-full-suite-disallow`, which runs `python -m pytest tests` on a self-hosted CUDA runner under `SIMSOPT_BACKEND_MODE=jax_gpu_parity`, `SIMSOPT_BACKEND_STRICT=1`, `SIMSOPT_JAX_TRANSFER_GUARD=disallow`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, and live-output settings (`PYTHONUNBUFFERED=1`, `--capture=tee-sys`, `log_cli`). The workflow contract is guarded by `tests/test_benchmark_helpers.py::test_gpu_parity_workflow_adds_full_suite_disallow_lane`; validated locally on 2026-04-13.
- [x] **43.** ~~Add direct GPU unit test for `BoozerSurfaceJAX` inner solver (LS and exact paths).~~ **DONE** — `test_run_code_traceable_exact_executes_inner_solve_on_gpu` (line 2953) + `test_run_code_traceable_lm_ondevice_executes_inner_solve_on_gpu` (line 3116) in `test_boozersurface_jax.py`; run in CI with `transfer_guard=disallow`.

### Tier 3 (nice-to-have)

- [ ] **44.** Multi-GPU collective operation test (if/when #35 is implemented).
- [ ] **45.** Tolerance ratchet regression test — verify CI contract `gpu_reduction_order_max_ulp` and `gpu_reduction_order_rel_tol` cannot loosen without explicit override. **(PARTIAL — contract tests exist in `test_benchmark_helpers.py` for ratchet tightening, ULP tracking, and payload state; but framed as unit tests, not CI regression gates)**
- [ ] **46.** Transfer guard fuzz test — systematically inject host scalars into kernel entry points and assert rejection under `disallow`. Scope this to the real single-stage target-lane entry points and immutable runtime-bundle boundaries, matching the official JAX transfer-guard semantics for implicit host↔device movement.
- [x] **47.** ~~Unit tests for private-optimizer edge cases: `y_k·s_k ≈ 0`, `‖y_k‖² ≈ 0`, stalled step, curvature-sign flip.~~ **DONE** — `test_minimize_lbfgs_private_rejects_degenerate_curvature_update`, `test_minimize_lbfgs_private_rejects_stalled_nonconverged_step`, `test_minimize_lbfgs_private_clamps_gamma_on_large_curvature_ratio` in `test_boozersurface_jax_private.py`; CI-validated.
- [x] **48.** ~~scipy optimizer lane on GPU parity workflow.~~ **DONE** — re-scoped by contract: `backend="jax"` no longer supports the SciPy optimizer lane at high-level entrypoints, so GPU parity and target workflows exercise the on-device optimizer path only, while the SciPy lane remains native CPU/reference-only.
- [ ] **49.** **[LOW]** Relax FD tolerances (or add skip+reason) for the two known FD-sensitive failures:
      - `tests/geo/test_boozer_derivatives_jax.py::TestComposedWeightInvModB::test_gradient_weighted_fd` (1/|B| near poles)
      - `tests/geo/test_boozer_derivatives_jax.py::TestBoozerResidualCoilVJP::test_coil_vjp_geometry_fd[gammas]`
      Either loosen `fd_tol` from 1e-4 or mark `@pytest.mark.skip(reason="...")`. Leave this last; it is cleanup around known FD sensitivity, not a blocker for the real CUDA correctness/runtime closure.

#### Single-stage closure order (2026-04-13)

1. Close **`#46`** next, against the same real entry points used by `#40`, so transfer-guard hardening is tied to the production lane rather than toy kernels.
2. Close **`#49`** last.

#### Single-stage algorithm follow-up outside this GPU-port block

The main remaining single-stage work after the GPU-port proof is donor/seed/search policy in `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`, not more proof scaffolding. The working direction is donor-class-aware continuation plus restoration-style shrink/retry for invalid geometry rather than a generic preserve-first rule. Useful references: Nocedal and Wright, *Numerical Optimization*; JAXopt `LBFGS` docs/release notes; and the Wächter-Biegler / IPOPT restoration-phase literature and output docs.

#### Stage 2 closure order (2026-04-13)

- [ ] Treat Stage 2 as **algorithm-first**, not **port-first**. The Stage 2 outer optimizer is already routed through the lane-specific JAX/reference substrate at `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py:1185-1259`, `tests/integration/test_stage2_jax.py:4627-4794` already guards the target-lane routing contract, and the real CUDA parity backlog for reduced-real plus production-scale LS coverage is already closed by **`#38/#39`**.
- [ ] Fix the **legacy-lane closeout** before adding more Stage 2 proof scaffolding. On **April 13, 2026**, the representative legacy lanes `014417_iota15` and `002084_iota20` stayed hardware-feasible but kept re-solving the same basin instead of closing decisively. The current Stage 2 objective in `banana_coil_solver.py:1917-1924` is still a fixed weighted penalty objective, so the next experiment should move toward proper augmented-Lagrangian semantics rather than more penalty-cap tuning.
- [ ] Keep the ALM follow-up narrow and evidence-driven:
  - [ ] Add explicit outer-loop observability for hard-feasibility, surrogate residual, projected stationarity, multiplier norm, `rho`, and inner-solver status.
  - [ ] Add a "feasible but not closed" detector keyed to persistent hard-feasibility plus stalled stationarity decrease.
  - [ ] Once that detector fires, switch from generic replay to a **feasible-closeout** mode that tightens first-order stationarity on the current feasible manifold.
  - [ ] Upgrade the outer loop toward proper ALM semantics with multiplier updates and projection for inequality constraints; increase `rho` only when violation reduction stalls. The Algencan defaults `tau=0.5` and `gamma=10` are reasonable initial settings, not immutable constants.
  - [ ] Prefer Newton / reduced-KKT closeout and trust-region closeout directions when the feasible-closeout path needs a second stage, closer to **ALGENCAN-NEWTON** / **ALGENCAN-OTR** than to simply increasing `rho`.
  - [ ] Avoid the unvalidated scalar warm-start heuristic `-grad_f / grad_c`; if dual warm-start is needed, use previous dual state or an active-set least-squares / KKT estimate.
- [x] **`#41`** is closed: the existing compile-count harness now covers both the real target-lane value/grad path and the real Stage 2 target outer-loop path, so no second mechanism is needed.
- [x] **`#42`** is closed: `.github/workflows/jax_gpu_parity.yml` now contains the `gpu-full-suite-disallow` lane that runs the full `tests/` suite on a self-hosted CUDA runner under `SIMSOPT_JAX_TRANSFER_GUARD=disallow`, with `XLA_PYTHON_CLIENT_PREALLOCATE=false` and live-output settings. Keep any future exemptions on the public JAX transfer-guard controls rather than `jax._src...` internals.
- [ ] If shared speed work is needed to make Stage 2 experimentation cheaper, prioritize it in this order:
  - [ ] **`#34`** first: `src/simsopt/backend/runtime.py:434-436` still declines to set a default compilation-cache directory, which directly hurts repeated real-lane Stage 2 iteration. Follow the official JAX persistent-cache setup, including setting the cache dir before first compile and using `jax_persistent_cache_min_compile_time_secs=0` when caching all real-lane compiles is desired.
  - [ ] **`#31`** second: `_lbfgs.py` still shifts history with slice+concatenate at `src/simsopt/geo/optimizer_jax_private/_lbfgs.py:105-123`, so a ring buffer is the clearest recurring hot-path cleanup.
  - [ ] **`#32`** third: line-search cleanup is still useful and low risk because the active BFGS/L-BFGS callers already pass `old_fval` and `gfk`; the relevant path is `src/simsopt/geo/optimizer_jax_private/_line_search.py:340-418`.
  - [ ] **`#24`** after that: keep buffer donation conditional on measured CUDA VRAM benefit. JAX donation only applies at true `jit` boundaries, and the repo already has the right probe scaffold in `benchmarks/biotsavart_donation_probe.py`.
  - [ ] **`#33`** stays low priority until profiling proves it matters on the real Stage 2 lane.
- [ ] De-prioritize, but do **not** close, the more speculative perf items until profiling or HLO evidence says otherwise:
  - [ ] **`#24`** is not a ship blocker and may turn out to be noise, but keep it open until CUDA memory profiling says the public outer-JIT donation probe is worthless.
  - [ ] **`#28`** should stay open until HLO or profiler evidence shows the three geometry calls are already fully deduplicated.
  - [ ] **`#33`** should stay open until measured line-search traces show the bracketing/zoom overlap is negligible on real Stage 2 workloads.

#### Stage 2 reference shelf

- Official JAX docs:
  - Persistent compilation cache: <https://docs.jax.dev/en/latest/persistent_compilation_cache.html>
  - Config options (`jax_log_compiles`, `jax_explain_cache_misses`): <https://docs.jax.dev/en/latest/config_options.html>
  - Transfer guard: <https://docs.jax.dev/en/latest/transfer_guard.html>
  - Buffer donation: <https://docs.jax.dev/en/latest/buffer_donation.html>
  - Device memory profiling: <https://docs.jax.dev/en/latest/device_memory_profiling.html>
  - OpenXLA HLO dumps / compile debugging: <https://openxla.org/xla/hlo_dumps>
- Open-source algorithm references:
  - NLopt AUGLAG notes and references: <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/>
  - ALGENCAN family codes (`ALGENCAN-NEWTON`, `ALGENCAN-OTR`): <https://www.ime.usp.br/~egbirgin/tango/codes.php>
  - SciPy strong-Wolfe line search notes: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.line_search.html>
- Literature:
  - Gil et al. (2025), *Augmented Lagrangian methods produce cutting-edge magnetic coils for stellarator fusion reactors*: <https://arxiv.org/abs/2507.12681>
  - Birgin and Martinez, augmented-Lagrangian survey: <https://www.ime.usp.br/~egbirgin/publications/bmsurveyal.pdf>
  - Nocedal and Wright, *Numerical Optimization*.

---

## Tier 5 — Documentation & cleanup

- [x] **50.** ~~Update `docs/using_jax_backend.md` with transfer-guard operational guidance.~~ **DONE** — `docs/using_jax_backend.md` documents `transfer_guard="log"` as default (lines 58, 67), strict mode comparison (lines 77-88), and three-tier validation pattern (lines 285-328).
- [x] **51.** ~~Document the spec-caching contract in `CLAUDE.md`.~~ **DONE** — CLAUDE.md lines 161-163 document the traceable runtime bundle cache contract and JIT closure strategy.
- [ ] **52.** Document `XLA_PYTHON_CLIENT_PREALLOCATE=false` requirement in deployment scripts (currently only mentioned in `docs/source/jax_gpu_setup.rst:301-304`).
- [ ] **53.** `optimizer_jax_private/*.py` — add inline algorithm references (e.g., "Nocedal & Wright, *Numerical Optimization*, Algorithm 7.4" for L-BFGS two-loop).
- [ ] **54.** `biotsavart.py:468-470` — add explicit `in_axes=(0,)` to the outer `jax.vmap` (currently relies on default). Minor but clarifying.
- [ ] **55.** `biotsavart.py:224-227, 282-283, 314` — document padding overhead budget and the `chunk_size` tuning trade-off (when is 2× overhead acceptable vs when should chunk_size be raised).
- [ ] **56.** `biotsavart.py:208-222`, `surface_rzfourier.py:260-279` — profile the two-chunk fast path special-case against the padded `fori_loop` path. If <5% improvement, delete for simplicity.
- [ ] **57.** Document the `biot_savart_d2B_by_dXdX` Hessian kernel memory cost (3×N tensor per point) at `biotsavart.py:550-553`. Consider opt-in flag if usage audit shows it's rarely called.
- [ ] **58.** `curve_geometry.py:486-572` — `segment_segment_distance_pure` uses 5 levels of nested `lax.cond`. Correct but hard to review. Add a comment diagram or split.
- [ ] **59.** Document the PLU ill-conditioning finding in a code comment near `_solve_boozer_adjoint` at `surfaceobjectives_jax.py:265-268` — explain why iterative refinement is on by default and why CPU/JAX direct parity is impossible on the exact path. **(PARTIAL — docstring at lines 266-272 explains iterative refinement rationale, but does NOT mention CPU/JAX parity impossibility on exact path)**
- [ ] **60.** Track upstream PR status for the simsopt merge (gate 5 of the ship gates — "Upstream PRs to simsopt: NOT STARTED" per `project_gpu_ship_gates.md`).

---

## Progress tracking

**Last audit:** 2026-04-10

**Total:** 60 items — **29 done, 10 partial, 21 open**

| Tier | Items | Done | Partial | Open |
|------|-------|------|---------|------|
| 0 — Ship blockers | 3 | **3** | 0 | 0 |
| 1 — Correctness/defensive | 8 | **8** (4-11) | 0 | 0 |
| 2 — Transfer-guard | 11 | **10** (12-14,16-22) | **1** (15) | 0 |
| 3 — Performance | 15 | **3** (23,30,36) | **5** (24,26,27,34,37) | **7** (25,28,29,31-33,35) |
| 4 — Test coverage | 12 | **3** (43,47,48) | **3** (40,42,45) | **6** (38,39,41,44,46,49) |
| 5 — Docs/cleanup | 11 | **2** (50,51) | **1** (59) | **8** (52-58,60) |

**Estimated remaining effort:**
- Tier 0: **CLEARED**
- Tier 1: **CLEARED**
- Tier 2: **NEARLY CLEARED** (item 15 remains partial; SciPy oracle lane still crosses a host NumPy boundary)
- Tier 3: ~2-3 weeks (quick wins in ~2 days, rest incremental)
- Tier 4: ~1 week (GPU test infrastructure)
- Tier 5: ~2-3 days

**None of these items invalidate the port.** The validation concluded the JAX port is correctly built on JAX idioms and production-grade for Stage-2 outer optimization on single-GPU (L4 evidence: 254/255 tests, 238 MB VRAM, bitwise reproducible; V100: 33× speedup). This list represents the punchlist between "research-usable on L4" and "production-ready strict-cuda on A100 with `disallow` baseline".
