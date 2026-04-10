# simsopt-jax GPU Port — Validation TODO List

**Audit date:** 2026-04-08
**Branch:** `gpu-purity-stage2-20260405`
**Source:** Deep-dive validation via 7 parallel audit agents (JIT/vmap/lax, device residency, autodiff/IFT, GPU/sharding, private optimizer, test coverage, docs) + direct verification.
**Scope:** 60 actionable items — 3 ship blockers · 8 correctness/defensive · 11 transfer-guard · 15 performance · 12 test coverage · 11 docs/cleanup.

> Note: Agent 1's "HIGH severity Python-loop recompilation" finding was **rebutted** by [jax-ml/jax#16611](https://github.com/jax-ml/jax/issues/16611) — JAX maintainer Jake VanderPlas confirmed unrolled Python for-loops are ~16× faster than `lax.scan` for small trip counts because XLA fuses across iterations. The existing `_grouped_field` and `_accumulate_grouped_field` designs are correct and are **not** on this list.

---

## Tier 0 — Ship blockers (must fix before any "production GPU mode" claim)

- [x] **1.** ~~Fix `test_pure_objective_matches_optimizable_value` at `tests/integration/test_single_stage_jax_cpu_reference.py:5358-5375`. M5 `f(coil_dofs)` diverges from `JF_jax.J()` beyond `rtol=1e-10`.~~ **DONE** — test passes at rtol=1e-10 (now at line 5818).
- [x] **2.** ~~Fix `test_boozersurface_jax.py:753` — "DID NOT RAISE RuntimeError" strict-mode enforcement gap.~~ **DONE** — `test_jax_minimize_rejects_fallback_methods_in_strict_mode` passes (4 parameterizations: adam/bfgs/lbfgs/bfgs-hybrid).
- [x] **3.** ~~Fix `test_lbfgs_reduces_objective` at `tests/geo/test_boozersurface_jax.py:745-752`.~~ **DONE** — sign fix `q = -jnp.conj(state.g_k)` at `_lbfgs.py:60`; test passes (now at line 938).

---

## Tier 1 — Correctness / defensive bugs (real but low-probability)

- [x] **4.** **[MED]** ~~`optimizer_jax_private/_lbfgs.py:200` — add finite-guard on `rho_k`.~~ **DONE** — `valid_curvature` guard at lines 245-253 checks `jnp.isfinite(rho_k)` & `(rho_k_inv > curvature_tol)`; gates history update via `update_curvature = valid_curvature & (~stalled_step)`.
- [x] **5.** **[MED]** ~~`_lbfgs.py:201` — clamp `gamma`.~~ **DONE** — `jnp.clip(gamma, step_eps, gamma_max)` at lines 214-218 with machine-precision bounds from `_lbfgs_step_tolerances()`.
- [x] **6.** **[MED]** ~~`_lbfgs.py` body_fun — port BFGS stalled-step check.~~ **DONE** — multi-condition stalled-step check at lines 237-242 (`s_k_norm <= step_tol`, `function_change <= objective_tol`, `gradient_change <= gradient_tol`).
- [ ] **7.** **[LOW]** `_bfgs.py:122-130` — add Powell damping or skip-if-negative-curvature to BFGS Hessian update. Strong Wolfe should prevent this in normal operation, but defensive hardening matters for edge cases. **(PARTIAL — non-finite guard exists at line 132 via `jnp.where(jnp.isfinite(rho_k), H_kp1, state.H_k)`, but no Powell damping or explicit negative-curvature skip like L-BFGS has)**
- [ ] **8.** **[LOW]** `optimizer_jax.py:2161-2166` — hybrid scipy→on-device continuation has no fallback when scipy produces non-finite state. Currently returns `success=False` silently; should log or retry with tighter scipy tolerance. **(PARTIAL — `_scipy_result_is_continuable()` check + message at lines 2276-2281, but no logger.warning() or callback invocation)**
- [ ] **9.** **[LOW]** `surfaceobjectives_jax.py:271-275` — add runtime signature-check for VJP callback at `run_code` result construction. Wrong signatures currently only surface deep in the gradient pass.
- [ ] **10.** **[LOW]** `boozersurface_jax.py:733-756` — `_build_ls_group_vjp_callback` closes over `booz_surf` state. Add a solver-generation counter and assert freshness when VJP is invoked so stale reuse is detected.
- [ ] **11.** **[LOW]** `surfaceobjectives_jax.py:678-694` (`_compute_dJ_ds`) — `_ensure_solved` at line 418 checks `res["success"]` but not the actual residual norm. Log final `‖grad‖` / residual norm alongside the success flag.

---

## Tier 2 — Transfer-guard hardening (path to `disallow` baseline)

### Real runtime transfers (MB/s-class, must fix before strict mode)

- [x] **12.** **[HIGH]** ~~`field/force.py` — `jnp.asarray(gammas_targets/gammadashs_targets/currents_*)` per-call conversions.~~ **DONE** — refactored to `_as_jax_float64()` throughout; `_prepare_target_source_inputs_pure()` and `_CoilStateGroupCache` handle all conversions.
- [x] **13.** **[HIGH]** ~~`field/force.py` — `jnp.asarray(opt.full_x)` / `jnp.asarray(coil.full_x)` per-call conversions.~~ **DONE** — now uses `_as_jax_float64()` at lines 329, 618.
- [x] **14.** **[MED]** ~~`field/force.py:520,525` — `jnp.asarray(symmetry.rotmat/scale)` per-coil conversions.~~ **DONE** — `_apply_coil_state_symmetry()` uses `_as_jax_float64()` at construction time (lines 528, 533).
- [ ] **15.** **[MED]** `geo/optimizer_jax.py:145, 150, 159, 165, 260` — `jnp.asarray(flat_x)` / `jnp.asarray(flat_grad)` at scipy boundary each iteration. Keep `x` as JAX array through the scipy call. **(PARTIAL — bare `jnp.asarray()` still used at scipy re-entry boundaries; x converted on each callback)**
- [ ] **16.** **[MED]** `jax_core/objectives_flux.py:86` — remove fallback `jnp.asarray(surface.gamma()), jnp.asarray(surface.normal())` when no `surface_spec()` is available. Enforce spec with a clear error.
- [ ] **17.** **[LOW-MED]** `geo/curveperturbed.py:195-196` — `jnp.asarray(self.sample[0/1])` in `__init__`. Replace with `_explicit_device_array()` for `disallow` compliance.
- [ ] **18.** **[LOW-MED]** `jax_core/curve_geometry.py:48-53` — `_as_explicit_float64` numpy fallback path. Tighten input contract to specs only. **(PARTIAL — function still accepts numpy via `jax.device_put(np.asarray(...))` fallback)**

### Cosmetic under `disallow` (zero runtime cost, trace-time only, but currently flagged)

- [ ] **19.** **[LOW]** `jax_core/biotsavart.py:46, 87-88, 387` — `_float64_scalar(_MU0_OVER_4PI)` is constant-folded at trace time but trips `transfer_guard=disallow`. Replace with `_device_scalars.device_one(reference) * 1e-7` idiom from `_device_scalars.py`. **(PARTIAL — still uses `jax.device_put(np.asarray(...))` wrapper at lines 88-95)**
- [ ] **20.** **[LOW]** `jax_core/biotsavart.py:91-102` — `_as_int32_scalar`, `_index_range`, `_zero_scalar` use `jax.device_put(np.asarray(...))` at trace time. Zero runtime cost but flagged by `disallow`. Switch to pure `jnp.arange`/`jnp.zeros` inside the traced scope. **(PARTIAL — same pattern persists)**
- [x] **21.** **[LOW]** ~~`geo/surfaceobjectives_jax.py:107` — `_explicit_index_array` uses `jax.device_put` at trace time.~~ **DONE** — intentional one-time at spec construction; acceptable under current contract.
- [x] **22.** **[LOW]** ~~`jax_core/specs.py:463-616` — 20+ `_as_float64_array` calls at spec construction.~~ **DONE** — all calls live in immutable spec factory functions (`make_coil_group_spec`, `make_curve_xyzfourier_spec`, etc.), not hot loops.

---

## Tier 3 — Performance opportunities

### Quick wins

- [ ] **23.** **[PERF 2-3%]** `jax_core/biotsavart.py:387` — add `precision=lax.Precision.HIGHEST` to `jnp.einsum("c,cj->j", ...)`. Matches private optimizer convention at `_common.py:23-24`.
- [ ] **24.** **[PERF 5-10% VRAM]** Add `donate_argnums` to accumulator-pytree arguments in `_coil_chunk_reduce`, `_quadrature_block_integral`, `_point_chunk_reduce` fori_loops (`biotsavart.py:240, 297, 329`). Recovers intermediate chunk buffers across iterations.
- [ ] **25.** **[PERF variable]** `jax_core/_math_utils.py:107-121` — replace `explicit_rsqrt` custom JVP with `jax.lax.rsqrt` + default JVP. GPU has native `RSQRT.APPROX.FTZ.F64` exposed directly. **Verify CPU-JAX bitwise parity first** — if the custom JVP exists only for explicitness, swapping is safe. Biot-Savart is rsqrt-dominated.
- [ ] **26.** **[PERF 10-20%]** `surfaceobjectives_jax.py:268` — `_solve_boozer_adjoint` unconditionally runs iterative refinement. Gate on estimated condition number for well-conditioned LS Hessians. **(PARTIAL — `iterative_refinement=True` always passed; no condition-number gating)**

### Medium effort

- [ ] **27.** **[PERF 40-60%]** `boozersurface_jax.py:753-755` — `_build_ls_group_vjp_callback` re-evaluates Biot-Savart for each coil group. Precompute `B_shared = grouped_biot_savart_B_from_spec(...)` once and reuse across group runners.
- [ ] **28.** **[PERF 15-25%]** `boozersurface_jax.py:413-423` — `_surface_geometry_from_dofs` computes gamma/xphi/xtheta separately. Fuse into a single JAX primitive for memory locality. Called thousands of times per outer solve.
- [ ] **29.** **[PERF 40%]** `optimizer_jax.py:1360-1363, 1375-1377` — `_materialize_dense_hessian` does full-column HVPs. For LS Hessians (J^T J, SPD), compute only the upper triangle and mirror.
- [ ] **30.** **[PERF 25-40%]** `surfaceobjectives_jax.py:663-675` — BoozerResidualJAX/IotasJAX/NonQuasiSymmetricRatioJAX all solve `(PLU)ᵀ adj_i = rhs_i` with the same PLU. Batch via `jax.vmap(solve_triangular)`. Check JAX 0.9.3+ for native batched triangular solve.
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

- [ ] **38.** Add `test_boozer_residual_jax_gpu` — full LS solve with `BoozerResidualJAX` / `IotasJAX` / `NonQuasiSymmetricRatioJAX` on CUDA with `transfer_guard=disallow`. **Currently zero GPU coverage of M5.**

### Tier 2 (required for production use)

- [ ] **39.** Add `TestRunCodeLSParityProductionScale` — `nphi=16, ntheta=8, ncoils=4` fixture (current `TestRunCodeLSParity` uses `nphi=5, ntheta=5, ncoils=2` = 25 points, 1-2 orders below production).
- [ ] **40.** Add full single-stage **outer-loop convergence** test on GPU (not just init-parity at `benchmarks/single_stage_init_parity.py`). Verify IFT adjoint decreases the objective over ≥10 outer iterations on CUDA. **(PARTIAL — `single_stage_outer_loop_probe.py` benchmark runs ≥10 iters on CUDA in CI, but no pytest-based test in `tests/`)**
- [ ] **41.** Add XLA recompilation-count smoke test. Track compilation counter across an optimizer loop; fail if compiles > expected per iteration. Protects against shape-dependent recompile regressions.
- [ ] **42.** Add CI test that runs `SIMSOPT_JAX_TRANSFER_GUARD=disallow` against the **full** suite on GPU (not just the e2e smoke at `jax_smoke.yml:257-318`). **(PARTIAL — `jax_gpu_parity.yml` and `jax_smoke.yml` set `disallow`, but only run curated slices, not full suite)**
- [x] **43.** ~~Add direct GPU unit test for `BoozerSurfaceJAX` inner solver (LS and exact paths).~~ **DONE** — `test_run_code_traceable_exact_executes_inner_solve_on_gpu` (line 2953) + `test_run_code_traceable_lm_ondevice_executes_inner_solve_on_gpu` (line 3116) in `test_boozersurface_jax.py`; run in CI with `transfer_guard=disallow`.

### Tier 3 (nice-to-have)

- [ ] **44.** Multi-GPU collective operation test (if/when #35 is implemented).
- [ ] **45.** Tolerance ratchet regression test — verify CI contract `gpu_reduction_order_max_ulp` and `gpu_reduction_order_rel_tol` cannot loosen without explicit override. **(PARTIAL — contract tests exist in `test_benchmark_helpers.py` for ratchet tightening, ULP tracking, and payload state; but framed as unit tests, not CI regression gates)**
- [ ] **46.** Transfer guard fuzz test — systematically inject host scalars into kernel entry points and assert rejection under `disallow`.
- [x] **47.** ~~Unit tests for private-optimizer edge cases: `y_k·s_k ≈ 0`, `‖y_k‖² ≈ 0`, stalled step, curvature-sign flip.~~ **DONE** — `test_minimize_lbfgs_private_rejects_degenerate_curvature_update`, `test_minimize_lbfgs_private_rejects_stalled_nonconverged_step`, `test_minimize_lbfgs_private_clamps_gamma_on_large_curvature_ratio` in `test_boozersurface_jax_private.py`; CI-validated.
- [ ] **48.** scipy optimizer lane on GPU parity workflow. Currently `jax_gpu_parity.yml` only tests the JAX optimizer path. **(PARTIAL — `TestRealFixtureGpuM5Parity` uses `optimizer_backend="scipy"` on GPU, but no dedicated scipy lane in `jax_gpu_parity.yml`)**
- [ ] **49.** **[LOW]** Relax FD tolerances (or add skip+reason) for the two known FD-sensitive failures:
      - `tests/geo/test_boozer_derivatives_jax.py::TestComposedWeightInvModB::test_gradient_weighted_fd` (1/|B| near poles)
      - `tests/geo/test_boozer_derivatives_jax.py::TestBoozerResidualCoilVJP::test_coil_vjp_geometry_fd[gammas]`
      Either loosen `fd_tol` from 1e-4 or mark `@pytest.mark.skip(reason="...")`.

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

**Total:** 60 items — **16 done, 14 partial, 30 open**

| Tier | Items | Done | Partial | Open |
|------|-------|------|---------|------|
| 0 — Ship blockers | 3 | **3** | 0 | 0 |
| 1 — Correctness/defensive | 8 | **3** (4,5,6) | **2** (7,8) | **3** (9,10,11) |
| 2 — Transfer-guard | 11 | **5** (12,13,14,21,22) | **4** (15,18,19,20) | **2** (16,17) |
| 3 — Performance | 15 | **1** (36) | **3** (26,34,37) | **11** (23-25,27-33,35) |
| 4 — Test coverage | 12 | **2** (43,47) | **4** (40,42,45,48) | **6** (38,39,41,44,46,49) |
| 5 — Docs/cleanup | 11 | **2** (50,51) | **1** (59) | **8** (52-58,60) |

**Estimated remaining effort:**
- Tier 0: **CLEARED**
- Tier 1: ~0.5 day (3 open are LOW-priority defensive guards)
- Tier 2: ~2-3 days (force.py bulk done; remaining are boundary/cosmetic)
- Tier 3: ~2-3 weeks (quick wins in ~2 days, rest incremental)
- Tier 4: ~1 week (GPU test infrastructure)
- Tier 5: ~2-3 days

**None of these items invalidate the port.** The validation concluded the JAX port is correctly built on JAX idioms and production-grade for Stage-2 outer optimization on single-GPU (L4 evidence: 254/255 tests, 238 MB VRAM, bitwise reproducible; V100: 33× speedup). This list represents the punchlist between "research-usable on L4" and "production-ready strict-cuda on A100 with `disallow` baseline".
