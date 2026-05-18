# Big-File Convention Review: `boozersurface_jax.py` + Optimizer Lane

**Date:** 2026-05-16
**Branch:** `gpu-purity-stage2-20260405`
**Runtime:** JAX 0.10.0 / jaxlib 0.10.0 / Python 3.11 / NumPy 2.x
**Worktree:** `/Users/suhjungdae/code/columbia/simsopt-jax`

**Files under review (line counts):**

| File | LOC |
|---|---|
| `src/simsopt/geo/boozersurface_jax.py` | 6104 |
| `src/simsopt/geo/optimizer_jax.py` | 3832 |
| `src/simsopt/geo/optimizer_jax_reference.py` | 536 |
| `src/simsopt/geo/optimizer_jax_private/__init__.py` | 38 |
| `src/simsopt/geo/optimizer_jax_private/_bfgs.py` | 281 |
| `src/simsopt/geo/optimizer_jax_private/_common.py` | 313 |
| `src/simsopt/geo/optimizer_jax_private/_lbfgs.py` | 469 |
| `src/simsopt/geo/optimizer_jax_private/_lbfgsb_scipy.py` | 3737 |
| `src/simsopt/geo/optimizer_jax_private/_line_search.py` | 745 |
| `src/simsopt/geo/optimizer_jax_private/_result_converters.py` | 201 |
| `src/simsopt/geo/optimizer_jax_private/_types.py` | 143 |
| **TOTAL** | **16399** |

---

## Executive Summary — Top 10 Findings (highest severity first)

### 1. [HIGH — correctness] L-BFGS-B RESTART task path is unimplemented
The on-device L-BFGS-B port omits SciPy's `task = RESTART` flag and associated memory refresh in 4 of the 5 SciPy upstream failure modes (`cauchy` info!=0, `formk` info!=0, `cmprlb`/`subsm` info!=0, `lnsrlb` failure with `col != 0`). Only `formt` info!=0 resets the memory state, and even that never writes `task = RESTART, task_msg = NO_MSG`. Effect: when the SciPy reference would refresh L-BFGS memory and continue, the JAX port re-enters the failed line-search state with `task = FG, task_msg = FG_LNSRCH`. The outer `lax.while_loop` then re-evaluates the objective at effectively the same `x` until `maxfun`/`maxiter` halts (livelock). This is the headline finding from `.artifacts/lbfgsb_parity_review_2026-05-16/00_SYNTHESIS.md` and is independently visible in `_lbfgsb_scipy.py` (`grep -n "task = RESTART"` returns nothing).
**File:** `src/simsopt/geo/optimizer_jax_private/_lbfgsb_scipy.py` — `_lbfgsb_setulb_subspace_line_search` (~line 777), `_lbfgsb_setulb_fg_start_line_search` (~line 425), `_lbfgsb_setulb_line_search_continue` (~line 1016).
**Fix:** Interpose `info != 0` gates after `lbfgsb_formk`, `lbfgsb_cmprlb`, `lbfgsb_subsm`, `lbfgsb_cauchy`; add the `col != 0` branch on `lnsrlb` failure; write `task = RESTART, task_msg = NO_MSG` in all five paths.

### 2. [HIGH — runtime SSOT divergence in exact lane] `solve_residual_equation_exactly_newton` materializes a dense LU factorization on success
At `boozersurface_jax.py:5757-5762`, the exact-Newton public solve path calls `jax.scipy.linalg.lu(J)` whenever `jacobian_available` is true, and stores the triple under `res["PLU"]`. CLAUDE.md states: "the `production_operator` exact lane never falls back to dense factors at runtime. … the exact normalizer in `_normalize_solver_options` continues to drop `optimizer_backend` from the user-visible exact path." But the dense `(P, L, U)` is materialized for the result dict even on the production path — the dispatcher correctly never *consumes* it for a linear solve (the runtime adjoint goes through `_jacobian_linear_operator` in `_build_runtime_linear_solve_callbacks`, see `boozersurface_jax.py:3664-3707`), so the runtime contract holds, but the LU computation is wall-clock work that the contract calls "diagnostic/reporting only". The corresponding `dense_linear_solve_factors_available = plu is not None` is set in `boozersurface_jax.py:5836`, which a downstream consumer reading the public `res` may use to make a (false) choice about a dense adjoint path.
**Recommend:** gate the LU on a debug flag or `verbose=True`; clarify the public schema field `dense_linear_solve_factors_available` to mean "diagnostic factor present, not runtime-load-bearing".

### 3. [HIGH — perf / trace size] `_lbfgsb_ddot` is a Python-unrolled per-element loop with `lax.cond`
`_lbfgsb_scipy.py:340-351` uses a Python-level `for i in range(int(x.shape[0]))` loop, each iteration issuing a `jax.lax.cond` that adds `x[i] * y[i]` to `total` when nonzero. This faithfully reproduces SciPy's BLAS skip-zero accumulation order, but the trace cost is O(n) HLO ops per call site. It is called 5+ times per L-BFGS-B iteration on n-length vectors and 2*n² times inside `lbfgsb_matupd` for an m×n history. For n=500, that's ~500 HLO operations × ~5–10 call sites × hot loop, producing very large compiled programs and high first-jit compile times. **Justified by parity intent, but should be documented as the dominant trace-time cost and a candidate for a "parity vs. fast" lane toggle.** Replacement: `jnp.sum(jnp.where(x * y != 0.0, x * y, 0.0))` reproduces the skip-zero semantics with O(1) HLO emit cost.
**File:** `src/simsopt/geo/optimizer_jax_private/_lbfgsb_scipy.py:340-351`.

### 4. [HIGH — perf] `_lbfgsb_initial_state_kernel` and `_lbfgsb_mainlb_kernel` re-jit per call
`_lbfgs.py:93-130` builds fresh `jax.jit(closure)` on every call without routing through `_cached_private_solver`. Existing `lru_cache` on `_make_traceable_levenberg_marquardt_runner`, `_make_traceable_newton_polish_runner`, and `_make_traceable_exact_newton_runner` (all in `optimizer_jax.py`) caches at the `lru_cache(maxsize=128)` level, but the L-BFGS-B initial-state and main-loop kernels do not. CPU users pay JIT compile latency per call; GPU users pay compile + device-binary build before each launch.
**Recommend:** route through `_cached_private_solver` keyed on `(n, m, maxls, ftol, gtol)`.
**File:** `src/simsopt/geo/optimizer_jax_private/_lbfgs.py:93-130`.

### 5. [MEDIUM — naming clarity] LM lane is damped Gauss–Newton, not trust-region LM
The matrix-free `_lm_iteration` (`optimizer_jax.py:1533-1665`) tracks a `delta` field labelled "trust-region radius" but never enforces a step-norm constraint. The `delta_after_step` update at `_lm_delta_after_step` (1510-1530) only feeds the `xtol` MINPACK info-code check (`xtol_met = delta <= xtol * x_norm` at line 1478) and never gates the step accept/reject loop. The step is solved by `(J^T J + λI) step = J^T r` GMRES (without the trust-region projection MINPACK does). This is correctly disclosed in the module docstring at `optimizer_jax.py:15-49`, but the variable name `delta` and the absence of a clear "this is just a convergence-proxy" comment in `_lm_iteration` mislead a reader.
**File:** `src/simsopt/geo/optimizer_jax.py:1437-1530, 1533-1665`.

### 6. [MEDIUM — observability gap] `accepted_step_callback` cannot abort the L-BFGS-B optimizer
SciPy halts the optimizer when the user-provided callback returns a truthy value (the `STOP_CALLB` task code). The on-device port routes the host observer through `jax.debug.callback` (line `_lbfgsb_scipy.py:1648-1661`), which has no return path back into the running `lax.while_loop`. Effect: the target lane silently runs to completion when SciPy would have stopped early. The corresponding `STOP_CALLB` constant exists at `_lbfgsb_scipy.py:34` but is never written by any path in this port. Public-API divergence; low operational impact in current production.
**File:** `src/simsopt/geo/optimizer_jax_private/_lbfgsb_scipy.py:1648-1661`.

### 7. [MEDIUM — asymmetric adjoint contract LS vs exact] `compute_G_from_currents` cotangent
On the exact lane (`solve_residual_equation_exactly_newton` at line 5638), when `G is None` the code calls `float(compute_G_from_currents(self.coil_currents))` at line 5684. The JAX VJP hook in `_boozer_exact_coil_vjp` at line 2187 only differentiates through `iota` and `G` explicitly; the `G_from_currents` cotangent is added in `_add_G_current_cotangent` (line 2693) only on the LS lane. The exact lane's `_boozer_exact_coil_vjp` (line 2187) takes `G` as an explicit arg of the VJP-traced function and never adds the `dG_dcurrents` cotangent. The asymmetry is correct in practice because the None-G case is rejected upstream by `_guard_none_G_coil_gradient_callback` on the exact lane (`freshness_guard=False` but the None-G guard fires before the VJP runs — see line 5796-5803 and 735-741), but this contract is implicit and worth a documenting comment.
**File:** `src/simsopt/geo/boozersurface_jax.py:2187-2245, 2693-2703, 5796-5815`.

### 8. [MEDIUM — performance] LS `linear_solve_factors` builds device copies twice
At `boozersurface_jax.py:3475-3518`, the `dense-plu-shared` callback path issues `lu_device = jnp.asarray(lu_piv[0], dtype=x.dtype)`, `piv_device = jnp.asarray(lu_piv[1], dtype=jnp.int32)`, AND constructs `linear_solve_factors=tuple(jnp.asarray(factor, ...) for factor in self.res["PLU"])` at line 3516-3518. The PLU triple is already on-device from `_optimizer_jax._plu_from_lu_piv(lu_piv)` (`boozersurface_jax.py:5297`), so the second `jnp.asarray` is a no-op for matched dtypes but materializes new device buffers if dtypes don't match. Combined with the same pattern in the SciPy `dense-plu` path (line 3526-3578) where `np.asarray(...)` actually moves data host-side then back, the net round-trip cost is non-trivial.
**File:** `src/simsopt/geo/boozersurface_jax.py:3475-3578`.

### 9. [LOW — code organization] `boozersurface_jax.py` is 6104 lines; should be split
14 result-dict schema definitions (lines 267-436), result builders for 5 solve modes, and grouping snapshot machinery (`_BoozerLSGroupedVJPSnapshot` and 14 helpers around it) all live in a single module. Suggested split: core class, residuals, adjoint snapshots, options/schemas. No correctness issue; just review/maintenance pain.

### 10. [LOW — test oracle hygiene] `optimizer_jax_reference.py` is not an independent oracle for LM
The reference module is a thin host-NumPy boundary that wraps `scipy.optimize.minimize(method="BFGS"|"L-BFGS-B")` for the reference/CPU lane (`optimizer_jax_reference.py:106-205`) and *also* dispatches to `_optimizer.levenberg_marquardt(...)` (the JAX LM in `optimizer_jax.py`) for the reference LS lane (`optimizer_jax_reference.py:381-449`). So `reference_least_squares()` is the same JAX kernel as `target_least_squares()` — they are not each other's parity oracle for LM. The module docstring at line 4-49 of `optimizer_jax.py` correctly says "`'lm'` (reference, host-driven) and `'lm-ondevice'` (target, trace-safe) are each other's byte-equality oracle, not MINPACK", but the file name `optimizer_jax_reference.py` is misleading.

---

## A. Per-Module Findings

### A.1 `boozersurface_jax.py` (6104 LOC)

#### A.1.a — `BoozerSurfaceJAX` Optimizable contract

| Check | Result | Notes |
|---|---|---|
| `super().__init__(depends_on=[biotsavart])` | PASS | Matches CPU at `boozersurface.py:325`. |
| `recompute_bell(self, parent=None)` sets `need_to_run_code = True` | PASS | `boozersurface_jax.py:3768-3770`. |
| `local_dof_names` / `local_full_x` defaults inherited from `Optimizable` | PASS | No DOFs of its own; correct. |
| `set_dofs` defaults inherited | PASS | `BoozerSurfaceJAX` is a derived quantity, not a primal DOF carrier. |
| Surface `set_dofs` is mirrored through `_set_surface_dofs(dofs_jax)` | PASS | Line 3869-3872; mirrors `_surface_dofs` → live `surface.set_dofs(_host_numpy(...))`. |

**No issues.** The Optimizable contract is observed.

#### A.1.b — `_ensure_solved` semantics (per CLAUDE.md code-review history)

The `_ensure_solved_value_state` helper at `surfaceobjectives_jax.py:2061-2076` correctly checks:
1. `if booz_surf.need_to_run_code` → re-run if `res` exists else raise.
2. `if booz_surf.res is None or not booz_surf.res["primal_success"]` → raise.

This matches the 2026-03-18 code-review fix called out in CLAUDE.md ("must check `res is not None` AND `res['success']`" — implemented via `primal_success` which is the correctness-tracked success boolean).

`_ensure_solved` (line 2055-2058) is a thin alias that calls `_resolved_boozer_adjoint_runtime_state` which goes through `_ensure_solved_value_state` first, so the same chain protects both value-path and adjoint-path callers.

**No issues.**

#### A.1.c — Result-dict schema completeness vs. CLAUDE.md

Exact-lane required fields (per CLAUDE.md "Exact Boozer scaling-limit contract"):
- `linear_solve_backend` PASS (line 5836 = "operator"; line 5747 = "operator").
- `dense_linear_solve_factors_available` PASS (line 5836; line 5745).
- `failure_category` PASS (via `exact_reporting = _exact_newton_reporting_fields(result)` at line 5710 spreading `**exact_reporting` into res at 5746 and 5839).
- `failure_stage` PASS.
- `jacobian_materialized` PASS.
- `dense_jacobian_shape` PASS.
- `dense_jacobian_bytes` PASS.
- `max_dense_jacobian_bytes` PASS.

LS lane required fields:
- `weight_inv_modB` PASS (line 5386, 5246, 5155).
- `linear_solve_backend` PASS (line 5379 via `_ls_linear_solve_backend`).
- `dense_linear_solve_factors_available` PASS (line 5384).
- `hessian_materialized`, `dense_hessian_shape`, `dense_hessian_bytes`, `max_dense_hessian_bytes` PASS (lines 5388-5391).

All required keys are present on success and failure result paths. The unified failure-path constructor at line 5229-5277 mirrors the success-path schema exactly. Non-trivial discipline win in a 6104-line file.

**No issues.**

#### A.1.d — `_normalize_solver_options` exact-lane strip

CLAUDE.md cites `boozersurface_jax.py:3122` and `:3185-3186` for the strip. Current file:
- Line 3132 (`_normalize_solver_options` definition).
- Line 3195-3196: `if boozer_type == "exact": normalized_options.pop("optimizer_backend", None)`.

The strip is correctly applied. The CLAUDE.md line numbers are slightly stale (file edits drift); the contract is preserved.

**No issues.**

#### A.1.e — `production_operator` exact-lane runtime fallback (no dense factors)

Verified at `boozersurface_jax.py:3664-3707`: `linearization_kind == "exact_jacobian"` branch in `_build_runtime_linear_solve_callbacks` builds `_jacobian_linear_operator(residual_fn, x)` and exposes only operator matvec + `_solve_jacobian_system[_with_status]` callbacks — none of which ever materialize a dense LU. The `EXACT_FACTORIZATION_BACKEND = "operator-gmres"` constant (line 165) is set in every exact-lane result dict.

The only dense `J` materialized on the exact path is the public diagnostic LU at lines 5757-5762 (Finding 2) which is exposed via `res["PLU"]` but never read by the runtime adjoint solver.

**Verdict:** runtime contract holds. The diagnostic dense LU is the `cpp_compatible_probe` reporting artifact.

#### A.1.f — `(P, L, U)` lifetime contract (LS load-bearing vs. exact metadata)

LS runtime callbacks use the PLU/LU_PIV factors as load-bearing:
- `dense-plu-shared` branch (`boozersurface_jax.py:3456-3519`): consumes `lu_piv` for both forward and adjoint solves via `_lu_solve_dense_hessian`. The public `linear_solve_factors` triple is derived from the same `(lu, piv)` via `_plu_from_lu_piv` so the bytes are bit-identical. PASS.
- `dense-plu` scipy branch (`boozersurface_jax.py:3521-3578`): consumes `H_host = P @ L @ U` and `scipy.linalg.solve_triangular` for the forward solve and `solve_triangular(U.T)`/`solve_triangular(L.T)` for the transpose. PASS. (Intentionally host-resident for byte parity with the C++ oracle.)

Exact runtime callbacks consume only the operator: `_jacobian_linear_operator(residual_fn, x)` provides matvec / transpose_matvec; `_solve_jacobian_system_with_status` runs operator-GMRES (no PLU). The exact-path result dict's `PLU` field is computed at line 5759 only for reporting and never plumbed into the adjoint runtime state. PASS.

**Verdict:** Contract holds.

#### A.1.g — Adjoint VJP signature `(lm, booz_surf, iota, G)`

- `_boozer_exact_coil_vjp(lm, booz_surf, iota, G)` PASS (line 2187).
- `_boozer_ls_coil_vjp(lm, booz_surf, iota, G, weight_inv_modB=True)` PASS (line 2321).
- `_boozer_exact_coil_vjp_groups(lm, booz_surf, iota, G)` PASS (line 2248).
- `_boozer_ls_coil_vjp_groups(lm, booz_surf, iota, G, weight_inv_modB=True)` PASS (line 2361).

`_require_boozer_vjp_callback_signature` (line 680-691) enforces the signature using `inspect.signature(callback).bind(object(), object(), object(), object())`. Strong enforcement. PASS.

The kwarg `weight_inv_modB=True` on the LS path is bound via `partial(_boozer_ls_coil_vjp, weight_inv_modB=weight_inv_modB)` in `minimize_boozer_penalty_constraints_newton` at line 5338, so consumers see a positional 4-arg callable.

**No issues.**

#### A.1.h — Traceable runtime bundle cache key

`_TraceableRuntimeCacheKey` (`surfaceobjectives_jax.py:4138-4148`) is a frozen dataclass with 8 fields:
- `solve_state_token` (int, from `booz_jax._traceable_solve_state_token`).
- `coil_dof_state_token` (int, from `bs_jax._coil_dof_state_token`).
- `coil_layout_signature` (structural signature object).
- `optimize_G` (bool).
- `predictor_kind` (str).
- `objective_contract_signature` (object — value-hashed).
- `option_signature` (object — value-hashed).
- `success_filter_signature` (either `("structural", sig)` or `("callable", _TraceableCallableSignature(filter))`).

`_traceable_success_filter_signature` (`surfaceobjectives_jax.py:4127-4135`) chooses the structural path when `success_filter._traceable_runtime_cache_signature` is set, and falls back to `_TraceableCallableSignature(success_filter)` for ad-hoc callables. The `_TraceableCallableSignature.__eq__` (line 4117-4121) uses `self.callback is other.callback` (identity, not equality) — so the cache shares for the *same* callable object reference and not for two equal-by-value lambdas. `object.__hash__(self.callback)` (line 4123-4124) is identity-based for normal Python objects.

This is the **correct cache contract** as described in CLAUDE.md. `id(callable)` is not used; instead Python's default `is` semantics on object identity is used.

The `solve_state_token` advances on every `_advance_solver_generation` (`boozersurface_jax.py:718-723`) and the `coil_dof_state_token` advances on every BiotSavart DOF write (`biotsavart_jax_backend.py:460, 499, 1023, 1050`). Tokens compare correctly.

`_advance_solver_generation` also nulls `booz_surf._traceable_runtime_entry_cache` (line 722), forcing the consumer to rebuild after each solve.

**Verdict:** Cache contract is correctly enforced.

#### A.1.i — `_ls_factor_once_dispatch_eligible` budget gate

At `boozersurface_jax.py:2981-3013`, the gate is implemented exactly as described in CLAUDE.md:
```
return n * n * itemsize <= int(max_dense_jacobian_bytes)
```
The "shared dispatch" path is engaged at line 5288-5298 (Newton-polish accept branch) and line 5310-5313 (`_ls_shared_lu_piv_dispatch`).

`_traceable_solve_plu_linearization` at `surfaceobjectives_jax.py:3167-3220` (cited by CLAUDE.md) uses `jsp_linalg.lu_solve((lu, piv), rhs, trans=1 if transpose else 0)` for the adjoint path, which guarantees bit-identical factor consumption for forward and adjoint.

The wording in CLAUDE.md "stored under `lax.stop_gradient`" refers to the IFT custom_vjp `f_fwd` saving `jax.tree_util.tree_map(lax.stop_gradient, result["linear_solve_factors"])` at `surfaceobjectives_jax.py:4466`.

**No issues.**

#### A.1.j — JIT closure strategy on LS path (`_make_penalty_objective_with`)

The closure captures `coil_set_spec` at construction time (line 3940-3999). The `hostify_inputs=True` path (the SciPy lane) does `_hostify_tree(resolved_coil_set_spec)` which moves the spec to host NumPy. The cache key (line 4444-4460) is keyed on `_traceable_surface_signature()` (value-hashed for arrays) and `_runtime_cache_tree_signature(coil_set_spec)` (uses `blake2b` of `array.tobytes()` — see line 799-812). Two callers with the same content build the same key.

**No issues.** Careful cache invalidation.

#### A.1.k — `int()`/`bool()` boundary conversions (per CLAUDE.md JAX-scalar contract)

Spot check:
- `"iter": int(_host_scalar(result["nit"], dtype=np.int64))` at lines 5142, 5364. PASS.
- `"success": bool(_host_scalar(result["success"]))` at lines 5144, 5365, 5366, 5367. PASS.
- `bool(_host_all_finite(...))` at line 4369. PASS.
- `int(_host_scalar(result["nit"], dtype=np.int64))` at lines 5821, 5364 (newton, exact). PASS.

**No issues.** Discipline is consistent across all `self.res` writes.

#### A.1.l — Stage callback (`_emit_stage_callback`)

`_emit_stage_callback` (line 3795-3802) and `_solver_diagnostics_payload` (line 3804-3821) provide a structured progress observability seam. The diagnostics payload moves data host-side (via `_host_scalar(result["fun"])`) only when a callback is registered. Stage gates emit `before_boozer_lbfgs`, `after_boozer_lbfgs`, `before_boozer_newton`, `after_boozer_newton`. Payloads are computed lazily.

`_make_solver_progress_callback` (line 3823-3838) emits at iterations ≤5 and every 25 iterations afterward.

**No issues.**

#### A.1.m — `_boozer_penalty_residual_vector` mathematical convention

The residual is divided by `sqrt(3 * nphi * ntheta)` at line 1888. This matches the CPU `boozer_penalty_constraints(..., scalarize=False)` convention (Boozer residual scaled by 1/sqrt(N) so the 1/2 ||r||² norm becomes 1/(2N) * sum_i r_i²). Label residuals `rl, rz` (line 1902-1903) are NOT divided by sqrt(N) — they're standalone scalar constraint residuals. Consistent with CPU behavior.

**No issues.**

#### A.1.n — Tracer hygiene in failure paths

`_traceable_plu_or_dummy` (line 2795-2822) and `_traceable_lu_piv_or_dummy` (line 2825-2855) handle the failed-solve case via a `lax.cond` that emits NaN factors when `finite` is False. Both helpers correctly distinguish concrete-finite (host bool) from traced-finite (`jax.core.Tracer`) cases.

**No issues.** Correct discipline for "compute factors when we know it'll succeed, else dummy" in a JAX-traceable context.

#### A.1.o — `run_code_traceable` schema

`run_code_traceable` (line 4561-4818) returns the `traceable` / `traceable_ls` / `traceable_exact` schemas defined at lines 416-435. Required keys are uniformly present; forbidden keys (`PLU`, `vjp`, etc.) are absent on the traceable lane (the LS path writes `plu` lowercase and `lu_piv`).

`_BOOZER_TRACEABLE_FORBIDDEN_RESULT_KEYS` (line 253-265) correctly forbids `PLU` (uppercase) and `LU_PIV` (uppercase). Lowercase `plu` and `lu_piv` survive — these are the traceable schema companions.

**No issues.**

#### A.1.p — `_boozer_exact_residual` stellsym branch selection (compile-time vs. traced)

`_select_exact_residual_fn(stellsym_surface)` (line 1972-1981) returns the stellsym or non-stellsym implementation at Python call time. This is a compile-time specialization choice (the two residual functions have *different* output shapes — stellsym omits the axis-z constraint, line 2068-2072). Tracing both branches under a `lax.cond` would be wrong because the cond branches must have the same output shape.

**No issues.** Excellent disciplined factoring.

#### A.1.q — `_normalize_solver_options` known-key whitelisting

`_ALLOWED_OPTIONS_LS` (line 3065-3073) unions ~15 frozensets to define the legal LS option keys. The whitelist is comprehensive but a future option addition has to thread through 4 places. Suggest a single SSOT dict-of-frozensets that the union is computed from.

**No correctness issue.**

#### A.1.r — Closure cache LRU sizes

The `lru_cache(maxsize=128)` on `_make_traceable_levenberg_marquardt_runner` (line 1216), `_make_traceable_newton_polish_runner` (line 2940), `_make_traceable_exact_newton_runner` (line 3204) bounds cache growth. For a long-running outer optimization with many different `(residual_fn, maxiter, tol)` triples this is bounded; for a single-`(maxiter, tol)` workflow the cache holds the same compiled kernel for the lifetime of the process. Reasonable.

**No issues.**

### A.2 `optimizer_jax.py` (3832 LOC)

#### A.2.a — `VALID_OPTIMIZER_BACKENDS`

Line 144: `VALID_OPTIMIZER_BACKENDS = frozenset({"scipy", "ondevice"})`. Matches CLAUDE.md. PASS.

`VALID_OUTER_OPTIMIZER_BACKENDS = frozenset({"scipy", "ondevice", "scipy-jax", "scipy-jax-fullgraph"})` (line 145-147) is the *outer* loop superset that also admits the SciPy-control lanes.

`OPTIMIZER_BACKEND_ROLE` (line 148-153) gives each backend a role label.

`TARGET_X64_REQUIRED_OPTIMIZER_BACKENDS` (line 154-156) requires float64 for all on-device-target backends.

`VALID_LEAST_SQUARES_ALGORITHMS = frozenset({"quasi-newton", "lm"})` (line 157).

**No issues.**

#### A.2.b — `_finalize_optimizer_result` and pytree adapter

The `_OptimizerPytreeAdapter` (line 224-285) handles non-flat pytree `x0` by flattening, running the optimizer on the flat vector, and unflattening at the boundary. `solver_cache_key()` (line 280-285) gives a stable cache key.

**No issues.**

#### A.2.c — LM `_lm_iteration` and `_matrix_free_lm_info`

The MINPACK-style `info` code generation at `_matrix_free_lm_info` (line 1451-1507) emits codes 1, 2, 3, 5, 6, 7 based on `ftol`/`xtol`/`maxiter`/`epsmch` gates. Codes 4 and 8 (which require pivoted-QR scaled gradient norm) are correctly NOT emitted.

The `legacy_success = grad_norm_next <= gradient_tol` (line 1645) preserves the pre-MINPACK gradient-norm gate, OR'd with `info_success = (info_next == 1) | (info_next == 2) | (info_next == 3)` (line 1646-1664). Either gate triggers success. Correct backward-compat shape.

Damping update: `damping_after_accept` uses 0.5× on ratio > 0.75 and 2× on ratio < 0.25, matching the docstring at line 35-39.

**Predicted reduction verification:** at lines 1566-1572,
```
predicted_reduction = 0.5 * (λ * <step, step> + <step, grad>)
```
Given step solves `(J^T J + λI) step = J^T r = grad`, rearranging gives `J^T J step = grad - λ step`. So `step^T J^T J step = step^T grad - λ step^T step`. Predicted reduction (model objective decrease):
```
m(0) - m(step) = grad^T step - 0.5 * step^T J^T J step
              = grad^T step - 0.5 * (step^T grad - λ step^T step)
              = 0.5 * grad^T step + 0.5 * λ step^T step
```
Matches the code. **Algorithm is correct.**

**No issues.**

#### A.2.d — `newton_polish` host-driven loop

Lines 2796-2937:
1. Linear solve via GMRES with fixed `linear_tol = min(1e-10, max(float(tol) * 0.1, 1e-14))` (line 2838).
2. Optional dense `H_step` solve when `dense_newton_steps=True`.
3. Iterative refinement when residual > tol.
4. Backtracking value-grad step.
5. Materialize dense Hessian at the final iterate only when `materialize_hessian=True`.

The host loop reads `float(norm) > tol` as the while condition (line 2837), so `norm` and `tol` are host-converted on every iteration. Acceptable for the reference lane.

The traceable variant `_make_traceable_newton_polish_runner` (line 2940-3070) uses `lax.while_loop` with `state["norm"] > tol_value` evaluated in the trace, AND Eisenstat-Walker `linear_tol = _eisenstat_walker_choice2_tolerance(...)` at line 2985-2989 — γ=0.9, α=2.

**No issues.** Asymmetry is by design (host loop = reference; trace loop = target with EW).

#### A.2.e — `newton_exact` host-driven loop and `newton_exact_traceable`

Same pattern as `newton_polish`:
- Host `newton_exact` (line 3104-3201) uses fixed `linear_tol`.
- Traceable `_make_traceable_exact_newton_runner` (line 3204-3353) uses Eisenstat-Walker forcing.

Both correctly avoid materializing the Jacobian inside the loop (matrix-free GMRES against JVP). The host loop's iterative refinement (line 3144-3156) and the traceable loop's `add_correction` (line 3247-3274) both gate the correction on `correction_finite`.

`exact_newton_linear_residual_rel` and `exact_refinement_correction_rel` are reported at the final state for solve-quality auditing.

**No issues.**

#### A.2.f — `_run_operator_gmres` transfer-guard relaxation

`_run_operator_gmres` (line 2288-2307) wraps the `gmres` call in `jax.transfer_guard("allow")`. JAX library issue: `gmres` lowers a few scalar literals through host-to-device conversions even when operands are fully device-resident. The relaxation is scoped (only `gmres` is inside the `with`) so the surrounding adjoint/forward code stays strict-transfer clean.

Similar pattern in `_hager_higham_inverse_1_norm_estimate` at lines 2468-2474: `lax.fori_loop` is wrapped in `transfer_guard("allow")` because the Python integer bounds lower through host-to-device conversions.

Both relaxations are confined and justified.

**No issues.**

#### A.2.g — `_eisenstat_walker_choice2_tolerance` correctness

Lines 2397-2431: implements Eisenstat & Walker (1996) eq. (2.6) with γ=0.9, α=2 (`ratio * ratio`). The `strict_cap` (line 2411-2417) bounds the EW η from above by the fixed `min(1e-10, max(tol*0.1, 1e-14))` ceiling. The `eta_min = 1e-12` and `eta_max = 0.5` floors and caps prevent EW from going below 1e-12 or above 0.5.

Published EW algorithm faithfully implemented.

**No issues.**

#### A.2.h — `_factor_dense_hessian` LAPACK/cuSOLVER routing

Line 1992-2016: the `optimizer_backend == "scipy"` branch routes through host LAPACK `dgetrf` via `scipy.linalg.lu_factor` to preserve "CPU pivot tie-breaks" for the C++ oracle byte parity. Other backends use `jsp_linalg.lu_factor` on the device. SciPy and JAX both use 0-indexed packed pivot semantics, so the returned `(lu, piv)` is a drop-in to `jax.scipy.linalg.lu_solve`.

**No issues.**

#### A.2.i — `_plu_from_lu_piv` permutation reconstruction

Lines 2033-2067: `@jax.jit`-wrapped permutation matrix reconstruction. The body uses `lax.fori_loop` to apply the pivot swaps to a permutation index vector, then constructs `P` via `.at[perm, columns].set(1.0)`. This produces the standard `P` such that `P L U = A`.

The `@jax.jit` wrapper hoists the static `jnp.eye(n)` and `jnp.zeros((n, n))` constructors inside the trace, avoiding a host roundtrip per call.

**No issues.**

#### A.2.j — `_solve_hessian_least_squares_system_with_status` Moore-Penrose

Lines 2702-2738: handles gauge-null Hessian directions on the LS lane by solving `H^T H y = H^T b` via operator-only GMRES. The docstring at 2710-2717 documents this design.

**No issues.** Correct approach for an inconsistent square system on a rank-deficient Hessian.

#### A.2.k — Adam optimizer

`adam_optimize` (lines 1067-1136) and `adam_optimize_traceable` (lines 1139-1205): host-driven and trace-safe variants with bias-corrected first/second-moment estimates. Standard Adam algorithm with `step_size`, `β1`, `β2`, `eps` defaults at `_adam_defaults`.

`_adam_iteration` (line 1011-1064) correctly applies the candidate step and falls back to the previous state when the candidate is non-finite. Standard "guard the step" Adam variant.

**No issues.**

#### A.2.l — `_load_private_pkg` and `__getattr__` lazy loading

Lines 536-561: private-package lazy load via `__getattr__`. The `_PRIVATE_LAZY_NAMES` frozenset (line 522-533) defines which private symbols are exposed. If the package is missing, attempts to access these raise AttributeError with a helpful message.

The mechanism is necessary because `_common.py` imports `PRIVATE_OPTIMIZER_JAX_VERSION` and `_x64_enabled` from the public module, so the public module must finish loading before the private package can be loaded. Correct circular-import workaround.

**No issues.**

### A.3 `optimizer_jax_reference.py` (536 LOC)

#### A.3.a — `reference_least_squares` is NOT a SciPy LM oracle (cross-ref Finding 10)

Line 414: `result = _optimizer.levenberg_marquardt(residual_fn, x0, ...)`. The reference LS path uses the same JAX LM kernel as the target LS path. They are tolerance-equivalent (per the docstring at `optimizer_jax.py:41-44`) but NOT byte-equivalent oracles.

#### A.3.b — `_strip_internal_options` for SciPy dispatch

Line 36-50: strips simsopt-internal option keys before passing to SciPy's `minimize`. Correct discipline.

#### A.3.c — `target_scipy_minimize_value_and_grad`

Line 208-261: implements the `scipy-jax` and `scipy-jax-fullgraph` lane — SciPy L-BFGS-B host control with JAX target-lane value/grad evaluations. Does NOT route through `_require_native_cpu_reference_backend_for_scipy_adapter` (the reference-lane guards), so it stays callable from the target/JAX backend mode.

**No issues.**

### A.4 `optimizer_jax_private/_bfgs.py` (281 LOC)

#### A.4.a — Dense BFGS Hessian update (NOT a ring-buffer history)

`_minimize_bfgs_private` (line 45-281): full dense O(d²) Hessian approximation `H_k`, not a limited-memory history. Classic BFGS:

```
ρ = 1 / (y^T s)
H_{k+1} = (I - ρ s y^T) H_k (I - ρ y s^T) + ρ s s^T
```

Implemented at lines 144-150:
```python
sy_k = s_k[:, np.newaxis] * y_k[np.newaxis, :]
w = identity - rho_k * sy_k
H_kp1 = _einsum("ij,jk,lk", w, state.H_k, w) + rho_k * s_k[:, np.newaxis] * s_k[np.newaxis, :]
```

The `_einsum` contraction `ij,jk,lk → il` evaluates as `Σ_j Σ_k W[i,j] * H[j,k] * W[l,k] = (W @ H @ W^T)[i,l]`. With `W = I - ρ s y^T`, `W^T = I - ρ y s^T`. So this gives `(I - ρ s y^T) H (I - ρ y s^T) + ρ s s^T` — matches the textbook.

**Algorithm is correct.** This is the inverse Hessian approximation `H_k`, consistent with the SciPy `BFGS` method.

#### A.4.b — Curvature check

`_bfgs_curvature_terms` (line 29-42): checks `rho_k_inv = y^T s > step_eps * ||s|| * ||y||`. When `valid_curvature` is False, `H_kp1 = state.H_k` (line 151) — skip the update.

**No issues.** Standard practice.

#### A.4.c — Strong Wolfe check

Lines 155-161: `strong_wolfe` requires both Armijo and curvature with c1=1e-4, c2=0.9. OR'd with `line_search_results.failed | stalled_step` (line 224) so the step is rejected if any fires.

**No issues.**

#### A.4.d — Cache caching via `_cached_private_solver`

Lines 264-281: `can_cache_solver = adapter is None and callback is None and progress_callback is None`. The compiled solver is cached on `fun` keyed on `("bfgs", norm, line_search_maxiter, gtol, maxiter)`. Caching is correctly disabled when a callback is registered.

**No issues.**

### A.5 `optimizer_jax_private/_lbfgs.py` (469 LOC)

#### A.5.a — L-BFGS is implemented via the SciPy L-BFGS-B port, NOT a two-loop recursion

The on-device L-BFGS is NOT a two-loop recursion. It is a faithful port of the SciPy L-BFGS-B Fortran routine via `_lbfgsb_scipy.py`. The "search direction" is computed via the compact-form L-BFGS-B subproblem (Byrd–Lu–Nocedal–Zhu 1995), which uses `bmv`, `cauchy`, `cmprlb`, `subsm`, `formt`, `formk`, `lnsrlb` kernels.

So the review questions about "two-loop recursion alpha[i] = ρ[i] * s[i]·q" and "wrap order" do not apply — this is a different L-BFGS implementation than the classical Nocedal two-loop recursion. The choice mirrors SciPy's underlying Fortran, which gives byte-for-byte parity with SciPy at the cost of more complex state. See `_lbfgsb_scipy.py` module docstring line 1.

**Naming tension:** CLAUDE.md says "the private methods live in `optimizer_jax_private/` and intentionally mirror the JAX 0.9.2 optimizer semantics" — but the L-BFGS implementation actually mirrors SciPy 1.17.1, not JAX 0.9.2's two-loop recursion. The BFGS (dense) variant DOES mirror JAX 0.9.2. So:
- **BFGS** (private) ≈ JAX 0.9.2.
- **L-BFGS** (private) ≈ SciPy 1.17.1 Fortran L-BFGS-B.

CLAUDE.md's "BFGS device residency" rule should be updated to reflect this dual-source design.

#### A.5.b — Caching of kernel

`_cached_lbfgs_value_and_grad_kernel` (line 51-90): only caches when the user-provided value-and-grad is marked `_CACHEABLE_VALUE_AND_GRAD_ATTR` and when an adapter (pytree boundary) either is None or exposes a `_STRUCTURED_SOLVER_CACHE_TOKEN_ATTR`. Guarantees the cached compiled kernel is keyed on the stable parts of the contract.

#### A.5.c — `_lbfgsb_initial_state_kernel` and `_lbfgsb_mainlb_kernel` not cached (Finding 4)

Each call to `_minimize_lbfgs_private_impl` (line 328-401) builds *new* `_lbfgsb_initial_state_kernel(...)` (line 374-379) and `_lbfgsb_mainlb_kernel(...)` (line 393-398) closures, each of which does `jax.jit(build)` (line 104) or `jax.jit(run)` (line 130). These re-jit on every call.

#### A.5.d — Trace budget gate

`_check_lbfgsb_trace_budget` (line 143-167) gates the optional `optimizer_state_trace` against `max_optimizer_state_trace_bytes` (default 64 MB). Bytes estimate: `iterations × (2 × d + 5) × 8`. Protects against runaway memory consumption.

**No issues.**

### A.6 `optimizer_jax_private/_lbfgsb_scipy.py` (3737 LOC)

This is the largest single file in the optimizer hierarchy. Literal port of the SciPy 1.17.1 `__lbfgsb.c` plus supporting kernels.

#### A.6.a — `lbfgsb_mainlb` outer while_loop

Line 1607-1645: outer `jax.lax.while_loop` with `continue_condition = task[0] < CONVERGENCE`. Body invokes `lbfgsb_setulb`, conditionally evaluates `value_and_grad` when `task == FG`, and conditionally emits the `accepted_step_callback` when `task == NEW_X`. Then applies `_lbfgsb_stop_after_new_x_limits` to gate on `maxiter`/`maxfun`.

#### A.6.b — `lbfgsb_setulb` re-entry state machine

Line 1534-1570: 5 entry points (START, RESTART, NEW_X, FG_LNSRCH, FG_START). The reentry dispatch (line 1516-1531) branches on `task_is_restart` vs. `task_is_new_x` vs. `task_is_fg_lnsrch`. The `task_is_restart` case calls `_lbfgsb_setulb_fg_start_line_search` (line 1518) — but **the input `task[0] == RESTART` is never reachable in the current port** (Finding 1), so this branch is dead code in practice.

The two-iteration `while_loop` at lines 1561-1569 with `continue_condition = restart & (count < 2)` lets `setulb` re-enter itself once on a RESTART — but again, since RESTART is never signalled, this is dead code.

#### A.6.c — `lbfgsb_matupd` ring-buffer-like update

Line 2204-2280: SciPy's `matupd` updates the m×n S and Y matrices in Fortran ring-buffer order:
- `next_col = min(iupdat, m)` (line 2235).
- `next_itail = (head + iupdat - 1) % m` when `iupdat <= m`, else `(itail + 1) % m` (line 2236).
- `next_head = head` when `iupdat <= m`, else `(head + 1) % m` (line 2237).

Standard SciPy/Fortran convention: `head` is the oldest correction index, `itail` is where the next correction is written; once `iupdat > m`, head advances mod m. PASS.

`ws[next_itail, :] = d` and `wy[next_itail, :] = r` install the new s_k, y_k. PASS.

The intricate `sy`/`ss` rollover at lines 2244-2255 shifts the lower-triangular `(s,y)` and `(s,s)` inner-product tables by one row/column when the buffer rolls over (mirroring SciPy's `idx` arithmetic). PASS.

`theta = rr / dr` (line 2240): θ is the scaling factor for the initial Hessian approximation. PASS. Matches SciPy.

The final loop (line 2257-2265) populates the new column of `sy` and `ss` with `d^T wy[pointr]` and `ws[pointr]^T d` inner products. PASS. The `(j < next_col - 1)` mask correctly handles the partial-buffer case before the ring is full.

**The matupd implementation is correct vs. SciPy.**

#### A.6.d — `lbfgsb_bmv` block matrix-vector product

Line 2283-2371: implements `p = M^{-1} v` where `M` is the BFGS compact-form middle matrix. The algorithm follows SciPy: two triangular solves via `jsp_linalg.solve_triangular(solve_matrix, rhs, trans=1, lower=False)` and `trans=0`. The `active_matrix` mask (line 2311-2312) selectively replaces inactive rows/columns with the identity so the triangular solve is well-defined when `col < m`.

The factor info code (line 2321-2322) checks for singular triangular diagonal (`diagonal == 0.0`). If `info != 0`, the result is returned with the unmodified `p` and `info != 0` so the upstream caller can branch. **But the upstream caller (`_lbfgsb_setulb_subspace_line_search`) does NOT branch on this info code today (Finding 1).**

#### A.6.e — `lbfgsb_formk` Schur complement

Line 2419-2803: forms the `wn` matrix used in `subsm`. Extremely intricate Fortran-faithful port. The Python-level unrolling means trace cost of O(m × n × hot_paths_per_call). For m=10 and n=500, ~5000 trace ops per `formk` invocation. Acceptable.

The Cholesky factorization is computed via `jnp.linalg.cholesky` with `finite` gating on result. Like `bmv`, the info code path is NOT consumed by the caller (Finding 1).

#### A.6.f — `lbfgsb_dcsrch` and `lbfgsb_dcstep` (Moré-Thuente line search)

Line 1968 and 1770: faithful port of the Moré-Thuente line search algorithm (cubic + secant interpolation with safeguarding). Structure matches: bracketing interval setup, dcstep for trial alpha, gate on Armijo + curvature with sufficient interval reduction.

#### A.6.g — `_lbfgsb_lnsrlb` line search outer driver

Line 3416-end-of-file: drives the Moré-Thuente line search. Handles the non-descent direction case (line 3585: `non_descent = first_function_value & (next_gd >= 0.0)` → return with `info=-4`). Handles the `iback >= maxls` exhaustion.

**Critical:** when `lnsrlb` returns with failure and `col != 0`, the SciPy upstream writes `task = RESTART; col = 0; head = 0; theta = 1.0; iupdat = 0; updatd = false` and re-enters the main loop with fresh memory. **The JAX port does NOT do this** (Finding 1).

#### A.6.h — `lbfgsb_inverse_hessian_history`

Line 1664-1678: exposes the current `s_history` (m×n) and `y_history` (m×n) buffers, plus `n_corrs = min(iupdat, m)`. Consumed by `_private_lbfgs_result_to_optimize_result` (`_result_converters.py:170-193`) to build a SciPy `LbfgsInvHessProduct` for the `hess_inv` attribute, which matches SciPy's `OptimizeResult.hess_inv` shape.

**No issues.**

### A.7 `optimizer_jax_private/_line_search.py` (745 LOC)

This file is **only used by `_bfgs.py`** (not by `_lbfgs.py`, which uses the embedded `dcsrch`/`dcstep` line search inside `_lbfgsb_scipy.py`).

#### A.7.a — Strong-Wolfe line search structure

`_line_search_from_restricted_func_and_grad` (line 395-672) implements the textbook strong-Wolfe line search with cubic/quadratic interpolation (`_cubicmin`/`_quadmin` from `_common.py`). Bracketing/zoom contract follows Nocedal & Wright Section 3.5 (cited in module docstring at line 5).

#### A.7.b — Best-finite-sample fallback

Lines 603-637: if the search fails but `best_a` is an acceptable α (improves over phi_0), accept it. Mirrors JAX 0.9.2 upstream.

#### A.7.c — Alpha clamping at float32 precision

Lines 656-661: clamps to ≥1e-8 in absolute value for float32 dtype. For float64 (the default in this stack), no-op.

#### A.7.d — Debug instrumentation

Lines 31-72: `SIMSOPT_LBFGS_DEBUG` env var enables runtime line-search prints via `_emit_debug_callback` (unordered). The docstring at line 49-55 documents that debug prints may interleave due to unordered=True (a forced choice to avoid strict transfer-guard tripping on the JAX 0.9.2 `bool[0]` host token that ordered=True introduces). Justified compromise.

**No issues.**

### A.8 `optimizer_jax_private/_common.py` (313 LOC)

#### A.8.a — `_scalar_value_and_grad` shape contract

Line 189-202: wraps a scalar objective into a `(value, grad)` callable. Uses `jax.vjp` with a unit cotangent, then `jax.tree_util.tree_map(lambda leaf: jnp.asarray(leaf, dtype=dtype), grad)` to coerce the gradient pytree to the right dtype.

#### A.8.b — `_cached_private_solver` double-checked locking

Lines 205-228: cache-on-attribute pattern with `_PRIVATE_SOLVER_CACHE_LOCK`. Acquires the lock only on cache miss. Under contention, two concurrent threads can both miss the cache and both compile — the second `compiled` overwrites the first when the lock is released. Classic "wasted work, no corruption" pattern.

**Cross-reference Finding 4** — the actual L-BFGS-B kernels don't go through this cache today; fixing that will surface the contention pattern as a real cost.

#### A.8.c — `_require_private_optimizer_runtime`

Line 239-259: gates the on-device optimizer behind two preconditions: JAX version ≥ `PRIVATE_OPTIMIZER_JAX_VERSION = 0.9.2` and `jax_enable_x64=True`. Throws a `RuntimeError` if either fails. Defensive but justified — float32 LBFGS is numerically unsafe for the typical Boozer problem condition numbers.

The `x0.ndim != 1` check (line 255-258) enforces flat decision vectors.

#### A.8.d — `_emit_debug_callback`

Line 149-157: shared `jax.debug.callback(..., ordered=False)` to avoid strict transfer-guard tripping.

**No issues.**

### A.9 `optimizer_jax_private/_result_converters.py` (201 LOC)

#### A.9.a — `_lbfgs_success` and `_lbfgs_message` dual-source

Lines 80-92: when `state.task is not None` (the SciPy L-BFGS-B port), success is gated on `status == 0` AND the message comes from `lbfgsb_task_message(task)`. When `task is None` (legacy paths), it falls back to `_status_message_lbfgs` and `_LBFGS_SUCCESS_STATUSES = {0, 4}`. Correct dual-source dispatch.

#### A.9.b — `_lbfgsb_hess_inv_from_state` host materialization

Lines 67-77: builds `LbfgsInvHessProduct(_as_host_numpy(s)[:n_corrs], _as_host_numpy(y)[:n_corrs])`. The `[:n_corrs]` slice trims the buffer to the number of valid corrections. Host-side because SciPy's class expects NumPy.

**No issues.**

### A.10 `optimizer_jax_private/_types.py` (143 LOC)

Pure data structures; no algorithm code. NamedTuples for BFGS, line-search, L-BFGS, invalid-step-log result containers. The `_LBFGSResults.optimizer_state_trace` field default `()` is a `tuple[dict, ...]` (line 115) — not a JAX-traceable structure, so consumers that want device-resident state must look at `s_history`/`y_history`/`gamma` directly.

**No issues.**

---

## B. L-BFGS Ring-Buffer Correctness Audit

### B.1 Architecture choice — port of SciPy L-BFGS-B, not classical two-loop recursion

As noted in §A.5.a, the on-device L-BFGS is a faithful port of SciPy 1.17.1's Fortran `_lbfgsb` routine, which uses the **compact-form L-BFGS-B** of Byrd, Lu, Nocedal, and Zhu (SIAM J. Sci. Comput. 16(5):1190-1208, 1995). This is mathematically equivalent to the two-loop recursion but is implemented via a structured `[D L^T; L -SS^T θ]` block-matrix factorization that enables bound-constrained subproblems efficiently.

The implementation correctly tracks:
- `head` (oldest correction index, advances mod m on rollover) — `_lbfgsb_scipy.py:2237`.
- `itail` (where next correction lands) — `_lbfgsb_scipy.py:2236`.
- `iupdat` (cumulative update count; saturates at m for the `next_col` calculation) — `_lbfgsb_scipy.py:2235`.
- `col = min(iupdat, m)` (number of active corrections).
- `theta = (y^T y) / (y^T s)` (initial Hessian scaling) — `_lbfgsb_scipy.py:2240`.

This is consistent with the standard `gamma = (s_k^T y_k) / (y_k^T y_k)` for L-BFGS up to the inversion convention (this port tracks the Hessian, not the inverse, so the formula is reciprocal).

### B.2 Wrap order and indexing correctness

In SciPy's Fortran, when `iupdat > m`:
- The oldest correction at `head` is overwritten by the new one at `(itail + 1) % m`.
- `head = (head + 1) % m` advances by one slot.

JAX port at line 2236-2237:
```python
next_itail = jnp.where(iupdat <= m, (head + iupdat - 1) % m, (itail + 1) % m)
next_head = jnp.where(iupdat <= m, head, (head + 1) % m)
```

For `iupdat <= m`: `next_itail = (head + iupdat - 1) % m`. With `head = 0` initially, after iupdat=1 the new correction goes at slot 0. After iupdat=2 at slot 1. Etc. PASS. Standard fill-from-head.

For `iupdat > m`: `next_itail = (itail + 1) % m` increments forward, `next_head = (head + 1) % m` also increments. So head and itail both advance together. PASS. Standard ring-buffer rollover.

**The ring-buffer arithmetic is correct vs. SciPy.**

### B.3 SS / SY upper-triangular rollover (matupd inner loops)

Lines 2244-2255 of `_lbfgsb_scipy.py` reshape the `sy` and `ss` inner-product tables on rollover.

JAX port:
```python
for j in range(1, m):
    for offset in range(j):
        value = ss[offset + 1, j]
        ss = ss.at[offset, j - 1].set(jnp.where(rollover & (j < next_col), value, ss[offset, j - 1]))
    for offset in range(m - j):
        active = rollover & (j < next_col) & (offset < (next_col - j))
        value = sy[j + offset, j]
        sy = sy.at[j - 1 + offset, j - 1].set(jnp.where(active, value, sy[j - 1 + offset, j - 1]))
```

This shifts `ss[offset, j-1] ← ss[offset+1, j]` and `sy[j-1+offset, j-1] ← sy[j+offset, j]` — column shift of column j into column j-1, dropping the first column entirely. Matches SciPy's column-shift convention.

The mask `(j < next_col)` ensures inactive columns (when the buffer isn't full) are not modified.

### B.4 Initial Hessian scaling (`theta`)

`theta = rr / dr` (line 2240) where `rr = ||y_k||²` and `dr = y_k^T s_k`. So `θ = (y^T y) / (y^T s)`.

The compact-form Hessian inverse approximation initializes as `H^{(0)} = θ^{-1} I = (y^T s) / (y^T y) * I`, which IS the standard L-BFGS initial scaling `H^{(0)} = γ I` with `γ = (s^T y) / (y^T y)`. Algorithm-equivalent.

### B.5 Curvature condition

The SciPy Fortran skips the update when `y^T s ≤ epsilon * ||y|| * ||s||`. The JAX port follows the same skip pattern via `skip_update` flag. When `skip_update` is True, the existing `ws`/`wy`/`sy`/`ss`/`theta` remain unchanged. Matches SciPy.

### B.6 Comparison to Optax / JAXopt L-BFGS

- **Optax `optax.scale_by_lbfgs`**: classical two-loop recursion on a ring buffer with `m_history` stored as JAX arrays, recurrence run inside a `lax.scan`. Bound-unconstrained-only.
- **JAXopt `LBFGS`**: also two-loop recursion in `lax.scan`, also unconstrained-only.
- **This port (`optimizer_jax_private/_lbfgsb_scipy.py`)**: compact-form L-BFGS-B with the full bound-constrained Cauchy point / subspace minimization machinery, achieving byte-parity (modulo Finding 1) with SciPy 1.17.1.

The choice of porting SciPy L-BFGS-B is **architecturally correct** for the simsopt-jax goal of "JAX target lane that matches SciPy reference lane on the Stage 2 / single-stage optimization workflow". Optax/JAXopt LBFGS would NOT achieve that parity because they are different algorithms.

### B.7 dtype handling

- `lbfgsb_initial_state` (`_lbfgsb_scipy.py:370-399`) enforces `jnp.float64` on x, l, u, g, factr, pgtol. PASS.
- `_require_private_optimizer_runtime` (`_common.py:239-259`) enforces `jax_enable_x64=True` and casts x0 to float64 (line 254). PASS.
- All workspace arrays (`wa`, `iwa`, `lsave`, `isave`, `dsave`) are initialized at float64 / int32 in `lbfgsb_empty_workspace` (line 358-367). PASS.

**dtype contract holds.**

### B.8 Verdict: L-BFGS ring buffer is correct except for the RESTART task path (Finding 1)

The ring-buffer arithmetic, scaling, curvature check, and dtype handling are all SciPy-faithful. The one remaining correctness gap is that 4 of 5 SciPy failure modes don't trigger memory refresh + RESTART, leading to livelock on ill-conditioned exact-Jacobian problems (Finding 1).

---

## C. Adjoint Contract and IFT Correctness

### C.1 IFT formula

CLAUDE.md "M5 implicit differentiation" defines the IFT adjoint formula:
```
dJ/d_coils = ∂J/∂coils − adj^T ∂g/∂coils
```
where `adj` solves the transposed inner linearization `(∂g/∂x)^T adj = (∂J/∂x)^T`.

### C.2 Implementation in `surfaceobjectives_jax.py`

The custom_vjp lives at `surfaceobjectives_jax.py:4446-4492`:

```python
@jax.custom_vjp
def f(coil_dofs):
    coil_dofs = _as_jax_float64(coil_dofs)
    return compiled_forward_result_for(coil_dofs)["value"]

def f_fwd(coil_dofs):
    coil_dofs = _as_jax_float64(coil_dofs)
    result = compiled_forward_result_for(coil_dofs)
    return result["value"], (
        coil_dofs,
        lax.stop_gradient(result["x"]),
        jax.tree_util.tree_map(lax.stop_gradient, result["linear_solve_factors"]),
        result["primal_success"],
    )

def f_bwd(saved_state, cotangent):
    coil_dofs, solved_x, solved_linear_solve_factors, primal_success = saved_state

    def _success(_):
        grad, linear_solve_success = compiled_total_gradient_for(
            coil_dofs, solved_x, solved_linear_solve_factors,
        )
        return _traceable_adjoint_gradient_or_nan(grad, linear_solve_success)

    def _failure(_):
        return _traceable_adjoint_fail_gradient_like(coil_dofs)

    grad = jax.lax.cond(primal_success, _success, _failure, operand=None)
    return (_as_runtime_float64(cotangent, reference=grad) * grad,)
```

**Verification points:**

1. **`lax.stop_gradient` on `result["x"]` and `result["linear_solve_factors"]`** (lines 4465-4466): the IFT adjoint backward pass MUST NOT retrace into the inner-solve `while_loop` or the LU factorization. `stop_gradient` is required to prevent that. PASS.

2. **`primal_success` gating**: when the primal solve failed, `_failure` returns the "fail gradient" (NaN-like surface that propagates a non-finite gradient) instead of the IFT adjoint. CLAUDE.md: "A successful traceable forward solve with a failed adjoint solve must surface a non-finite gradient, not a finite direct-gradient or failure-penalty fallback." Implemented via `_traceable_adjoint_gradient_or_nan` (returns NaN when adjoint failed). PASS.

3. **`_traceable_total_gradient_with_status`** at `surfaceobjectives_jax.py:3616-3641` calls `_traceable_objective_gradient_parts` (lines 3644-3759):

```python
# Direct term
direct_grad = _strict_scalar_grad(_evaluate_objective_of_coils, coil_dofs)

# Adjoint solve: (∂g/∂x)^T adj = ∂J/∂x
dJ_dx = _strict_scalar_grad(lambda x: _evaluate_objective(x, coil_dofs, coil_set_spec), solved_x)
adjoint, linear_solve_success = _traceable_solve_linearization(
    booz_jax, solved_x, dJ_dx, coil_set_spec, objective_kwargs,
    linear_solve_factors=solved_linear_solve_factors,
    linearization_kind=linearization_kind,
    linear_solve_tol=linear_solve_tol,
    linear_solve_stab=linear_solve_stab,
    transpose=True,  # critical: solves (∂g/∂x)^T adj = ∂J/∂x
)

# Implicit term: differentiate adj^T g(x, coils) wrt coils
def directional_stationarity_of_coils(current_coil_dofs):
    return _traceable_directional_inner_stationarity(
        solved_x, adjoint, coil_set_spec_from_dofs(current_coil_dofs),
        **inner_objective_kwargs,
    )
implicit_grad = _strict_scalar_grad(directional_stationarity_of_coils, coil_dofs)

# Total
total_grad = _traceable_adjoint_gradient_or_nan(
    direct_grad - implicit_grad,
    linear_solve_success,
)
```

The `directional_stationarity` formulation evaluates `adj^T g(x, coils)` as a scalar function of coils, then `_strict_scalar_grad` differentiates wrt coils, yielding `adj^T ∂g/∂coils`. This is the standard VJP trick to avoid materializing the cross-Jacobian. **Mathematically correct.**

### C.3 Boundary case: `depends_on_x_inner = False`

When the objective doesn't depend on the inner state (e.g., a pure coil-geometry penalty), the implicit term is zero. The code at lines 3706-3709 correctly short-circuits: `dJ_dx = 0`, `adjoint = 0`, `linear_solve_success = True`. The `not depends_on_x_inner` branch at line 3737-3739 then returns `direct_grad` as both implicit and total. PASS.

### C.4 Adjoint solve dispatch

`_traceable_solve_linearization` (line 3271-3307) routes by `linearization_kind`:
- `"hessian"` → `_traceable_solve_hessian_linearization` (line 3285-3295), which uses `_traceable_solve_plu_linearization` when `linear_solve_factors` are available, else operator GMRES.
- `"exact_jacobian"` → `_traceable_solve_exact_linearization` (line 3296-3304), which uses `_optimizer_jax._solve_jacobian_system_with_status` (operator-GMRES).

Both branches return `(solution, success)`, with success gating downstream NaN-fallback. PASS.

### C.5 Failure propagation contract

`_traceable_adjoint_gradient_or_nan(direct_grad - implicit_grad, linear_solve_success)` at line 3755-3758 is the gate. When `linear_solve_success` is False, `_traceable_adjoint_fail_gradient_like` returns a NaN-filled array. PASS.

### C.6 Verdict: IFT adjoint is mathematically and contract-correct

The IFT adjoint implementation is faithful to the formula and respects all CLAUDE.md contract guards.

---

## D. Exact-Mode No-Fallback Contract

### D.1 Runtime path contract

The exact lane (`linearization_kind == "exact_jacobian"`) MUST:
1. Solve linearizations operator-only at runtime (no dense Jacobian materialization at runtime).
2. Be addressable via `_normalize_solver_options` which strips the `optimizer_backend` key from exact-lane options.
3. Process adjoint RHS one column at a time in `production_operator` mode.

### D.2 Verification

**1. Runtime adjoint:** `boozersurface_jax.py:3664-3707` `linearization_kind == "exact_jacobian"` branch:
- Calls `_jacobian_linear_operator(residual_fn, x)` (`optimizer_jax.py:2741-2766`) which only builds JVP/VJP closures, no dense matrix.
- Provides `solve_forward` and `solve_transpose` that call `_solve_jacobian_system[_with_status]` — operator-GMRES via `_solve_square_array_system_operator_only`.
- The runtime callbacks never touch `res["PLU"]` even when populated. PASS.

**2. `_normalize_solver_options` strip:** line 3195-3196: `if boozer_type == "exact": normalized_options.pop("optimizer_backend", None)`. PASS.

**3. Batched exact adjoints one RHS at a time:** `_solve_square_array_system_operator_only` (`optimizer_jax.py:2568-2582`) handles vector and column-batched RHS via `jax.vmap(solve_column, in_axes=1, out_axes=(1, 0))`. Each column goes through `_solve_square_vector_system_operator_only` independently. PASS.

### D.3 Diagnostic dense materialization (Finding 2)

The public `solve_residual_equation_exactly_newton` path at `boozersurface_jax.py:5757-5762` materializes a dense LU via `jax.scipy.linalg.lu(J)` for the result dict. This is the `cpp_compatible_probe` diagnostic mentioned in CLAUDE.md — a reporting artifact. The contract holds because:
- The LU never feeds the runtime adjoint (verified at §A.1.f).
- The result dict's `linear_solve_backend == "operator"` is unchanged.
- The `EXACT_FACTORIZATION_BACKEND = "operator-gmres"` constant is set in `res["exact_factorization_backend"]`.
- The `dense_linear_solve_factors_available` field is set TRUE when the LU is materialized — which is the source of the misleading public-field concern in Finding 2.

The diagnostic LU computation IS extra wall-clock work. For the production "production_operator" lane, it would be cleaner to gate this LU computation on a debug flag rather than always-on. But this is a polish item, not a contract break.

### D.4 Failure-stage and category reporting

The exact lane reports:
- `failure_category = "scaling_limit"` when `J_bytes > max_dense_jacobian_bytes` (`optimizer_jax.py:1967-1968`).
- `failure_stage = "dense_jacobian_finalization"` in the same case.
- `jacobian_materialized = False`, `dense_jacobian_shape = (rows, cols)`, `dense_jacobian_bytes`, `max_dense_jacobian_bytes` populated correctly.

`_exact_newton_dense_jacobian_message` at `optimizer_jax.py:1940-1947` provides a human-readable message.

The schema fields are surfaced through `_exact_newton_reporting_fields(result)` (line 2858-2867) and consumed via `**exact_reporting` in the `self.res` dict at lines 5746, 5839 of `boozersurface_jax.py`.

### D.5 Verdict: Exact-mode no-fallback contract holds

The runtime adjoint is operator-only on all paths. The diagnostic LU materialization for the public result dict is an acknowledged reporting artifact and does not violate the contract. The only nit is that the `dense_linear_solve_factors_available = True` field could mislead a downstream consumer into thinking the exact adjoint can be back-fed dense factors — clarification of the field's semantics in the schema docs would help.

---

## E. Positive Notes

### E.1 Discipline wins

- **`int()`/`bool()` boundary conversions**: applied consistently across all `self.res` writes. The 2026-03-18 code-review fix is preserved.
- **No `try`/`except`**: zero defensive-fallback `try` blocks in optimizer_jax_private files. Discipline matches user guardrails.
- **No `id(callable)` cache keys**: identity-based equality (`is`) is used via Python's default `object.__hash__` and a `__eq__` that explicitly compares `callback is other.callback`. Correct cache-sharing contract.
- **`linear_solve_factors` LS load-bearing vs. exact metadata**: enforced at the runtime callback level (`_build_runtime_linear_solve_callbacks`), with the exact-lane runtime never reading PLU/LU_PIV factors from the result dict.
- **VJP signature enforcement**: `_require_boozer_vjp_callback_signature` (`boozersurface_jax.py:680-691`) fail-fast checks the 4-arg `(lm, booz_surf, iota, G)` contract on every result-dict callback.
- **Freshness guards**: `_guard_solver_callback_freshness` (`boozersurface_jax.py:694-715`) prevents stale adjoint requests after the Boozer solve state is mutated.
- **`_advance_solver_generation` invalidates downstream caches**: nulls `booz_surf._traceable_runtime_entry_cache` and advances both `_solver_generation` and `_traceable_solve_state_token` (`boozersurface_jax.py:718-723`). Triple cache invalidation.

### E.2 Architectural choices

- **Compile-time specialization for stellsym vs. non-stellsym exact residual**: `_select_exact_residual_fn` at line 1972-1981 returns a different Python callable for each case, avoiding a traced `lax.cond` that would have shape-mismatched outputs.
- **Compact-form L-BFGS-B port**: enables bound-constrained subproblems via the same algorithm SciPy uses. The cost is a 3737-line port file, but the result is SciPy-byte-parity (modulo Finding 1).
- **Eisenstat-Walker Choice-2 forcing terms**: traceable Newton paths use the EW Choice-2 strategy for inner GMRES tolerance, which gives quadratic local convergence at lower per-iteration cost than fixed-tolerance Newton. Published algorithm faithfully implemented (`optimizer_jax.py:2397-2431`).
- **Moore-Penrose normal-equation fallback**: for gauge-null Hessian directions on the LS lane, `_solve_hessian_least_squares_system_with_status` (`optimizer_jax.py:2702-2738`) solves `H^T H y = H^T b` via operator-GMRES.
- **Operator-GMRES seam in `_run_operator_gmres`** (`optimizer_jax.py:2288-2307`): single scoped `transfer_guard("allow")` for the JAX library implementation, keeping the surrounding solve path strict-transfer clean.

### E.3 Code clarity

- Module-level docstrings (`optimizer_jax.py:1-74`, `boozersurface_jax.py:1-27`) clearly state the lane architecture and contract intent.
- The `_BoozerResultSchema` system (`boozersurface_jax.py:177-436`) provides a schema-by-schema breakdown of which keys are required vs. forbidden for each solver type.
- The IFT adjoint docstring at `surfaceobjectives_jax.py:3592` ("Implicit total derivative of the pure traceable objective") concisely describes the mathematical contract.

---

## F. Verdict Per Module

| Module | Verdict |
|---|---|
| `boozersurface_jax.py` | **CONCERNS — one MEDIUM (Finding 2: diagnostic LU on success path), several LOW (Finding 9: split file; Finding 8: factor double-copy).** Optimizable contract, schema completeness, JIT closure caching, traceable bundle cache, and adjoint contract all hold. The exact-mode no-fallback runtime contract holds. The IFT adjoint is mathematically correct. |
| `optimizer_jax.py` | **CONCERNS — one MEDIUM (Finding 5: `delta` is not a trust-region radius; naming clarity), one LOW (Finding 10: `optimizer_jax_reference.py` is not a true reference oracle for LM).** LM matrix-free algorithm is correct (predicted_reduction formula verified). Newton polish/exact host-and-traceable variants are correct. Eisenstat-Walker forcing terms correctly implemented. Adam is standard. Private-package lazy-load is well-engineered. |
| `optimizer_jax_reference.py` | **PASS — one LOW polish (Finding 10).** Module is a thin host-NumPy boundary for SciPy `minimize`. The reference LS path is intentionally the same JAX LM kernel as target LS; this should be made more explicit in the file's docstring. |
| `optimizer_jax_private/_bfgs.py` | **PASS.** Dense BFGS implementation matches textbook with proper curvature skip and strong-Wolfe gating. Cache key is appropriately keyed on `(norm, line_search_maxiter, gtol, maxiter)`. |
| `optimizer_jax_private/_lbfgs.py` | **CONCERNS — one HIGH (Finding 4: kernels re-jit per call).** L-BFGS public surface routes through the SciPy L-BFGS-B port; the value-and-grad kernel is correctly cached but the initial-state and main-loop kernels are not. |
| `optimizer_jax_private/_lbfgsb_scipy.py` | **CONCERNS — one HIGH (Finding 1: RESTART task path unimplemented), one HIGH perf (Finding 3: Python-unrolled `_lbfgsb_ddot`).** The compact-form L-BFGS-B port is otherwise faithful to SciPy 1.17.1. The ring-buffer arithmetic, scaling, curvature check, and dtype contract all hold. The RESTART gap causes livelock on ill-conditioned exact-Jacobian inputs and should be fixed before claiming SciPy parity on those inputs. |
| `optimizer_jax_private/_line_search.py` | **PASS.** Strong-Wolfe line search with cubic/quadratic interpolation correctly ported from JAX 0.9.2 upstream. Best-finite-sample fallback handles zoom failures gracefully. Float32 alpha clamping is no-op for the production float64 stack. |
| `optimizer_jax_private/_common.py` | **PASS.** Shared utilities (dot, norm, scalar value-and-grad, cached solver attr). Double-checked locking pattern in `_cached_private_solver` is benign under contention (wasted work, no corruption). |
| `optimizer_jax_private/_result_converters.py` | **PASS.** Clean dual-source dispatch on `state.task` (SciPy port vs. legacy). `LbfgsInvHessProduct` materialization is host-side, which is correct because SciPy's class expects NumPy. |
| `optimizer_jax_private/_types.py` | **PASS.** Pure NamedTuple definitions; no algorithm code. |

---

## G. Recommended Remediation Priority

Order by ship-blocker impact:

1. **Finding 1 (RESTART task path)** — landing this closes the SciPy-L-BFGS-B parity gap on ill-conditioned inputs. Scope: 4 callsite updates + 1 refresh helper + test fixtures. Estimated 1–2 reviewer-days.
2. **Finding 4 (L-BFGS-B kernel re-jit)** — large GPU/CPU compile-cost regression in deployed code. Route through `_cached_private_solver` keyed on `(n, m, maxls, ftol, gtol)`. Scope: 2 callsite updates. Estimated 0.5 reviewer-day.
3. **Finding 2 (diagnostic LU on success path)** — gate the LU materialization on a debug flag or `verbose=True`; clarify the public schema field `dense_linear_solve_factors_available` to mean "diagnostic factor present, not runtime-load-bearing". Scope: 1 callsite + 1 schema doc update. Estimated 0.5 reviewer-day.
4. **Finding 3 (`_lbfgsb_ddot` Python-unrolled loop)** — replace with `jnp.sum(jnp.where(x * y != 0.0, x * y, 0.0))`. Scope: 1 function. Verify against the existing parity tests. Estimated 1 reviewer-day with validation.
5. **Finding 5 (rename `delta` and document its role)** — cosmetic, but in a 3832-line file naming matters. Estimated 0.5 reviewer-day including comment updates.
6. **Finding 6 (callback-driven halt)** — public-API divergence; low priority unless a researcher requests it. Estimated 1 reviewer-day to plumb `STOP_CALLB` through the trace.
7. **Finding 9 (split `boozersurface_jax.py`)** — maintainability win, not a correctness issue. Estimated 1–2 reviewer-days with care for cyclic-import safety.
8. **Findings 7, 8, 10** — polish; no functional impact.

---

**End of review.**

**Files cited (absolute paths):**

- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/boozersurface_jax.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/optimizer_jax.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/optimizer_jax_reference.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/optimizer_jax_private/__init__.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/optimizer_jax_private/_bfgs.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/optimizer_jax_private/_common.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/optimizer_jax_private/_lbfgs.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/optimizer_jax_private/_lbfgsb_scipy.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/optimizer_jax_private/_line_search.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/optimizer_jax_private/_result_converters.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/optimizer_jax_private/_types.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/surfaceobjectives_jax.py` (cross-referenced for IFT adjoint and cache key)
- `/Users/suhjungdae/code/columbia/simsopt-jax/CLAUDE.md`
- `/Users/suhjungdae/code/columbia/simsopt-jax/.artifacts/lbfgsb_parity_review_2026-05-16/00_SYNTHESIS.md` (cross-referenced for Finding 1 corroboration)
