# JAX convention/best-practice review — objectives, solve, backend layer

Worktree: `/Users/suhjungdae/code/columbia/simsopt-jax`
Branch: `gpu-purity-stage2-20260405`
Runtime: JAX 0.10.0 / jaxlib 0.10.0 / NumPy 2.x / Python 3.11
Date: 2026-05-16
Auditor: Opus 4.7 (1M context)

Files audited (LOC):

- `src/simsopt/geo/surfaceobjectives_jax.py` (5935 LOC)
- `src/simsopt/objectives/fluxobjective_jax.py` (433 LOC)
- `src/simsopt/objectives/integral_bdotn_jax.py` (156 LOC)
- `src/simsopt/objectives/stage2_target_objective_jax.py` (1292 LOC)
- `src/simsopt/solve/permanent_magnet_optimization_jax.py` (798 LOC)
- `src/simsopt/solve/wireframe_optimization_jax.py` (885 LOC)
- `src/simsopt/backend.py` (104 LOC) + delegated `src/simsopt/backend/runtime.py` (1774 LOC)

Cross-referenced CPU siblings:

- `src/simsopt/geo/surfaceobjectives.py` (1730 LOC)
- `src/simsopt/objectives/fluxobjective.py` (133 LOC)
- `src/simsopt/solve/permanent_magnet_optimization.py` (527 LOC)
- `src/simsopt/solve/wireframe_optimization.py` (859 LOC)
- `src/simsopt/jax_core/pm_optimization.py` (2522 LOC)
- `src/simsopt/jax_core/wireframe.py` (527 LOC)

Severity legend used below: BLOCKER → must fix before merge / publication; HIGH → correctness or contract risk, scope a fix this cycle; MEDIUM → maintainability or robustness; LOW → polish / documentation / cosmetic; NOTE → positive observation or non-blocking convention deviation.

## Executive summary — top findings (BLOCKER / HIGH first)

1. **HIGH — `_traceable_solve_plu_linearization` condition-estimator assumes Hessian symmetry but is also reached during forward warm-start solves** (`surfaceobjectives_jax.py:3219-3231`). The comment claims "only reached for the LS lane (linearization_kind == 'hessian') where matrix is the symmetric Hessian and so κ_1(matrix) == κ_1(matrix.T)", but `_traceable_solve_hessian_linearization` (`surfaceobjectives_jax.py:3073-3103`) is called by both adjoint (`transpose=True`) and forward warm-start (`transpose=False`) paths, and the LS Hessian symmetry argument should still hold because the inner LS Hessian IS symmetric — but the comment is read out-of-context and the condition estimator passes the native (non-transposed) matrix to `_dense_matrix_condition_estimate` even when `transpose=True`. The reasoning is correct in this codebase (Hessian symmetry) but the comment text mixes "only reached for LS" with "regardless of transpose". Recommend tightening the comment to "the LS Hessian is exactly symmetric" so the rationale survives a future predictor-mode addition.

2. **HIGH — `_make_traceable_objective_from_compiled_bundle` returns a `jax.jit(f)` for the custom-VJP scalar but does not `donate_argnums` the cotangent or coil_dofs argument** (`surfaceobjectives_jax.py:4492`). The custom-VJP scalar is on the hot single-stage path and is invoked many times per outer iteration; `donate_argnums=(0,)` for the `f`-wrapper (or at minimum on the `compiled_value_and_grad_for` callable at `surfaceobjectives_jax.py:4084`) would let XLA reuse the coil-dof buffer. Not a correctness bug, but a memory-and-throughput regression on GPU for large coil sets.

3. **HIGH — `_traceable_runtime_entry_cache` lives as a mutable attribute on `BoozerSurfaceJAX` but is invalidated only by `_traceable_solve_state_token` changes — not by changes to `bs_jax`'s coil-DOF state alone** (`surfaceobjectives_jax.py:4194-4196`). The cache key DOES include `coil_dof_state_token`, but the cache slot is keyed by `booz_jax` identity; if a caller swaps the `bs_jax` argument to `make_traceable_objective(booz_jax, bs_jax_NEW, ...)` while the same `booz_jax` is reused, the comparison `cached_entry["cache_key"] == cache_key` will see a different token and rebuild — correct. The only attack surface is if a caller mutates `bs_jax._coil_dof_state_token` directly (bypassing `_advance_coil_dof_state`), but that is documented as private API. Verified safe; flagged here only because the cache slot identity (`booz_jax._traceable_runtime_entry_cache`) makes the cross-`bs_jax` lifecycle non-obvious; consider documenting that the cache is `booz_jax`-scoped.

4. **HIGH — Failure-mode silent fallback for ondevice LS adjoint in `_traceable_solve_hessian_linearization`** (`surfaceobjectives_jax.py:3085-3103`): when `linear_solve_factors is None`, the helper falls back to `_solve_hessian_least_squares_system_with_status` which is the LIVE solver. CLAUDE.md "Adjoint / warm-start operator solves" explicitly states "A successful traceable forward solve with a failed adjoint solve must surface a non-finite gradient, not a finite direct-gradient or failure-penalty fallback." The current code does honor that: on linear-solve failure `_traceable_adjoint_gradient_or_nan` returns NaN-gradient. But the fallback path of "call the live solver inside `jit`" violates "Traceable adjoint must NOT call the live solver inside `jit`" — except this fallback is only reached when `linear_solve_factors is None`, which only happens for the operator-only exact lane. For the LS lane, factors are always populated. Verified, but the dead-fallback branch is a latent correctness trap if a future change starts passing `factors=None` for the LS lane.

5. **HIGH — `NonQuasiSymmetricRatioJAX._compute_value` uses JAX-pure surface reconstruction (`_qs_ratio_pure`) for value computation** (`surfaceobjectives_jax.py:2612-2617`), departing from the CLAUDE.md M5 adapter contract that states "CPU surface objects (`surface.gamma()`, `label.J()`) for value computation, and JAX autodiff … for gradient computation." Same pattern in `BoozerResidualJAX._compute_value_from_solved_state` (`surfaceobjectives_jax.py:2388-2404`). This is a *design upgrade* (single coherent JAX pipeline) but the CLAUDE.md documentation is now out-of-date. Either update the M5 adapter pattern description in CLAUDE.md to "pure JAX from solved state, CPU spec/DOF as source-of-truth", or restore CPU-side value-of-record. The numerical equivalence is preserved (both paths produce the same scalar), so this is documentation drift rather than a correctness bug — but the gap will mislead future maintainers about the boundary contract.

6. **MEDIUM — `SquaredFluxJAX` field-points version check uses `==` over an integer counter that is incremented across the bus** (`fluxobjective_jax.py:225, 359-365`). If the field's `_points_version` was advanced for an unrelated reason (e.g. an external `set_points()` from a sibling Optimizable) and then advanced back to the same value, the drift check would not fire. The current implementation is a monotonic counter so collision is unlikely, but the contract is "captured at construction"; a stricter check would clone the points array and `np.array_equal` against the live view, or store a UUID-on-set. The current implementation is materially sufficient.

7. **MEDIUM — `wireframe_optimization_jax.bnorm_obj_matrices_jax` calls `ext_field.B()` (CPU compatibility seam) even when running on the JAX lane** (`wireframe_optimization_jax.py:673-677`). For an `ext_field` that is itself a JAX field, this still works because the M2 adapter renders `B()` as a JAX-result-host-materialized array, but the result is then `np.sum(...)` host arithmetic. This deviates from the "no CPU compatibility seam on JAX lane" contract that `SquaredFluxJAX` adheres to. Recommend exposing a `B_jax`-style native call when `ext_field` is a JAX magnetic field, or document explicitly that wireframe RCLS treats `ext_field` as a snapshotted host array.

8. **MEDIUM — `stage2_target_objective_jax._fixed_curve_penalty` uses a Python double-loop (`stage2_target_objective_jax.py:236-250`)** at construction time over coil-pair gammas. The result is unrolled into the JIT closure, so this is a one-time setup cost, not a runtime regression. Flag for awareness — for `O(n_coil^2)` large counts (>50 fixed coils) construction-time tracing will grow quadratically.

9. **MEDIUM — `make_traceable_objective` docstring does not state that the returned scalar must NOT be called with NumPy/list inputs** (`surfaceobjectives_jax.py:5235-5285`). The boundary rejection lives at `_traceable_runtime_reject_host_input` (`surfaceobjectives_jax.py:3963-3970`), but the public-API docstring just says "Returns: f(coil_dofs) -> jax.Array". Add a one-line statement that host inputs must enter through `host_value_and_grad` or via explicit `jax.device_put`.

10. **LOW — `_invalidate_distributed_tuning_caches` is called in a context where the caller already holds `_backend_runtime_lock`** (`backend/runtime.py:1305-1310, 1492`). Re-acquiring the same RLock is correct in Python, but it makes the lock contract harder to audit. Not a bug.

## Per-module findings

### `src/simsopt/backend.py` + `src/simsopt/backend/runtime.py`

#### A. Simsopt-convention compliance

- **Legacy env vars** — `STAGE2_BACKEND` and `SIMSOPT_JAX_BACKEND` are still recognized at `backend/runtime.py:35-39` and resolved at `_resolve_legacy_value` (line 1161) and `_resolve_legacy_platform` (line 1176). New env vars `SIMSOPT_BACKEND` and `SIMSOPT_JAX_PLATFORM` take precedence. CLAUDE.md contract satisfied.
- **`get_backend()`, `is_jax_backend()`, `get_jax_platform()`** are all read-at-call-time but cached via `_cached_backend_config` (line 1198). The cache is documented as requiring `invalidate_backend_cache()` after monkeypatching env vars (line 1612-1623). MEDIUM convention: this differs from "read at use-time" if a test harness changes env without calling invalidate. The cache is process-scoped, not module-import-scoped, so it satisfies the basic "no module-import-time read" requirement (`get_backend_config` only resolves on first call). Verified by tracing: no top-level `get_backend()` or `get_backend_config()` call in `runtime.py` import path.
- **No implicit `jax.config.update` at import** — `apply_jax_runtime_config` (`runtime.py:1716`) is an explicit call-only API. `should_eagerly_configure_jax` (line 1658) gates eager configuration on `is_jax_backend() and any explicit selector env present`. Correct.

#### B. JAX best-practices compliance

- **`apply_jax_runtime_config` imports `jax` inside the function** (line 1723) — good defensive pattern; lets `simsopt.backend` import on machines without JAX.
- **`_validate_initialized_jax_runtime`** (`runtime.py:1674-1690`) correctly checks `jax.default_backend()` after the fact and raises in strict mode if the resolved platform mismatches. Good.

Verdict: **PASS**. The runtime is well-designed; the only minor issue is the cache invalidation contract that tests must honor.

### `src/simsopt/objectives/integral_bdotn_jax.py`

#### A. Simsopt-convention compliance

- Three definitions ("quadratic flux", "normalized", "local") mirror the CPU contract.
- Empty-grid behavior matches `simsoptpp.integral_BdotN`: `normalized` → `inf`, others → `nan` (line 139-143). Confirmed via `jnp.exp(zero)/zero = inf` and `zero/zero = nan` constructions that survive autodiff.
- `local` singular case handled via `inf_with_nan_jvp` (`_math_utils.py:110-123`) — a custom-VJP that returns `inf` value and a `nan` cotangent on the singular branch. Correct; SquaredFlux contract demands NaN gradient (the CPU sibling raises `ObjectiveFailure`).

#### B. JAX best-practices compliance

- `@partial(jax.jit, static_argnames=("definition", "reduction_mode"))` (line 112) — correct static argument.
- `jax.lax.cond` is used (line 93) for the `local` singular branch; this is the right primitive for a non-trivial JVP-safe conditional.
- `pairwise_sum_flat` is used for the `normalized` denominator — good numerical stability over a naive `jnp.sum`.

Verdict: **PASS**.

### `src/simsopt/objectives/fluxobjective_jax.py`

#### A. Simsopt-convention compliance

- `SquaredFluxJAX(Optimizable)` (line 172) — `depends_on=[field]` (line 237). Correct.
- `J()` and `dJ()` mirror the CPU sibling. `recompute_bell` is overridden (line 355).
- **Set-points-at-construction contract** — `field.set_points_from_spec(field_eval_spec)` at line 224 + drift detection in `_raise_if_field_points_drifted` (line 358-365). The error message explicitly directs the user to rebuild. **MATCHES CLAUDE.md.**
- **DOF-layout drift check** (line 367-376) and **surface-DOF drift check** (line 378-386) extend the same pattern to layout and surface DOFs. Surface-DOFs use `blake2b` over `local_full_x` (line 87-103) — a strong fingerprint. Good.
- **Strict native contract** — uses `coil_dof_extraction_spec` rather than `field.B()` / `field.B_vjp()` (line 67-73 and 325-337). Matches CLAUDE.md "unsupported fields are rejected by the native contract".
- **ObjectiveFailure on non-finite gradient** (line 54-63 and 428-432) — matches the CPU sibling's failure mode for `normalized` and `local` definitions when `|B|² = 0`.

#### B. JAX best-practices compliance

- **JIT closure capture** — fixed surface arrays (`_flux_spec`, `_normal_jax`, `_target_jax`) baked into the JIT (line 215-221). `_jit_forward_dofs` and `_jit_val_grad_dofs` use closure-captured `self._flux_spec` (line 254-258). **MATCHES CLAUDE.md "JIT closure strategy".**
- **`jax.value_and_grad(forward, argnums=0)`** (line 252) — single fused forward+grad call. Good.
- **Host boundary** — `_host_scalar(value, dtype=np.float64)` and `_host_array(grad, dtype=np.float64)` at line 426-427. Explicit dtype, correct.
- No `donate_argnums` — the surface is fixed and the DOF vector is small, so not a real-world concern.
- The fast-path for `_uses_uniform_curve_xyz_fourier_fastpath` (line 263-323) unrolls the coil loop with `for ci in range(n_coils)` and stacks at the end. For >50 coils this could blow up tracing time, but the fast-path predicate guards against unintended use; the spec-native path (line 325-337) is the general case.

Verdict: **PASS**. Best-in-class M2 wrapper.

### `src/simsopt/geo/surfaceobjectives_jax.py`

This is the central M5 module. I will break findings out by sub-area.

#### A. Optimizable contract

- `_SurfaceScalarMetricJAX(Optimizable)` (line 746) — base class for `AreaJAX`, `VolumeJAX`, `AspectRatioJAX`. `depends_on=[self.surface]`. Correct.
- `PrincipalCurvatureJAX(Optimizable)` (line 803) — `depends_on=[surface]`. Correct.
- `QfmResidualJAX(Optimizable)` (line 850) — `depends_on=[surface, biotsavart]`. Also appends parent (line 856). Correct.
- `_BoozerObjectiveBase(Optimizable)` (line 2251) — shared base for `BoozerResidualJAX`, `IotasJAX`, `MajorRadiusJAX`, `NonQuasiSymmetricRatioJAX`. `depends_on=[boozer_surface]` at line 2256 or 2258. Correct.
- `recompute_bell` defined (line 2265) — clears `_J`, `_dJ`, `_dJ_by_dcoil_dofs`. Correct.

#### B. M5 IFT adjoint correctness (BoozerResidualJAX, IotasJAX, NonQuasiSymmetricRatioJAX)

See dedicated section below; all three follow `dJ/dcoils = ∂J/∂coils − adj^T ∂g/∂coils` with the correct sign.

#### C. BoozerSurfaceJAX.get_adjoint_runtime_state() SSOT compliance

- `_resolved_boozer_adjoint_runtime_state` (line 2085-2090) is the ONLY entrypoint that fetches adjoint state, calling `get_adjoint_runtime_state()` on the BoozerSurfaceJAX object. Every wrapper goes through this. **MATCHES CLAUDE.md SSOT contract.**
- Direct `booz_surf.res` accesses are limited to:
  - `_log_boozer_solve_state` (line 2040-2052) — read-only diagnostic.
  - `_ensure_solved_value_state` (line 2061-2076) — read-only guard.
  - `_build_traceable_objective_state` (line 3924) — reads `linearization_kind` for cache-key purposes (this is metadata).
  - Docstring reference at line 5253.
  None of these poke runtime data — they only read metadata or guard against unsolved state.

#### D. Bundle cache invariants (`_traceable_runtime_cache_key`)

- `solve_state_token` (BoozerSurfaceJAX integer counter, advances on successful re-solve) — line 4161.
- `coil_dof_state_token` (BiotSavartJAX integer counter, advances on aggregate `x` writes AND on ancestor invalidation per `set_recompute_flag` at `biotsavart_jax_backend.py:1052-1059`) — line 4162.
- `coil_layout_signature` (structural via `_traceable_contract_tree_signature`) — line 4163.
- `optimize_G` (Python bool) and `predictor_kind` (str) — lines 4164-4165.
- `objective_contract_signature` — structural signature of `objective_kwargs` via `_traceable_contract_tree_signature` — line 4166.
- `option_signature` — `_traceable_runtime_option_signature` (line 4099-4110) hashes `booz_jax.options` for the keys in `_TRACEABLE_RUNTIME_OPTION_KEYS` (`surfaceobjectives_jax.py:171-183`). Good.
- `success_filter_signature` — `_traceable_success_filter_signature` (line 4127-4135) first checks `_traceable_runtime_cache_signature` (structural sharing), then falls back to `_TraceableCallableSignature` (line 4113-4124) which uses `is` identity via `object.__hash__(self.callback)` and `__eq__` returning `self.callback is other.callback`. **MATCHES CLAUDE.md cache contract.**

No `id()` calls and no per-instance adapter tokens. Cache slot lives on `booz_jax._traceable_runtime_entry_cache` (line 4194-4196), keyed by full cache-key tuple. Verified safe.

#### E. LS-lane PLU as load-bearing factor (`surfaceobjectives_jax.py:3167-3220`)

- `_traceable_solve_plu_linearization` consumes the 5-tuple `(P, L, U, lu, piv)` when available (line 3184-3191) via `jsp_linalg.lu_solve` — **byte-shared dispatch with the forward path's factors**. Matches CLAUDE.md "Phase 2 §5.3".
- 3-tuple `(P, L, U)` fallback path (line 3192-3201) uses `solve_triangular(L, P.T @ rhs)` then `solve_triangular(U, ...)`. Mathematically equivalent and correct.
- `_traceable_plu_matvec` (line 3126-3130) does `P @ (L @ (U @ vector))` directly without materializing the dense `H_host` — preferred for jit because it lets XLA fuse the chained matmuls. The CLAUDE.md note "consumer code computes `H_host = P @ L @ U` and uses it" describes the SciPy reference backend (`boozersurface_jax.py:3514-3540`) which IS host-resident and builds the dense product; the on-device traceable consumer uses the same factors but routes them through `lu_solve` / triangular solves. Both are correct uses of "load-bearing PLU".
- Residual-tolerance success gate (line 3138-3164) implements a Wilkinson-style backward-error pad — `safety * dimension * eps * (||matrix|| * ||solution|| + ||rhs||)`. Good numerical practice.
- Condition-estimator hand-off to `_dense_matrix_condition_estimate` (line 3227-3230) with `(lu, piv)` reused — avoids the O(n³) refactorization. The "LS Hessian symmetry" rationale is correct because the Boozer LS Hessian *is* symmetric by construction.

#### F. Failure-mode gradient propagation

- `_traceable_adjoint_gradient_or_nan` (line 3606-3613) — on `linear_solve_success=False`, returns `jnp.full_like(gradient, nan)`. **MATCHES CLAUDE.md "must surface a non-finite gradient, not a finite direct-gradient or failure-penalty fallback".**
- Custom-VJP backward at `_make_traceable_objective_from_compiled_bundle` (line 4470-4486): on `primal_success=False`, returns `_traceable_adjoint_fail_gradient_like(coil_dofs)` (full-NaN). On `primal_success=True` but adjoint failure, the `_success` branch invokes `_traceable_adjoint_gradient_or_nan(grad, linear_solve_success)` — correct.

#### G. JAX best-practices

- **`stop_gradient` discipline** — `lax.stop_gradient(result["x"])` and `tree_map(lax.stop_gradient, result["linear_solve_factors"])` at `f_fwd` (line 4465-4466) and `_objective_fwd` (line 5820-5823). Prevents the IFT adjoint from re-tracing into the linear-solve factorization. Correct per CLAUDE.md.
- **`jax.custom_vjp` on `f`** (line 4451) — correct pattern for IFT.
- **`jax.lax.cond`** at line 4484, 3577 — used for the baseline-fast-path vs general-path branch. Correct for traceable conditionals.
- **`jax.lax.scan`** — not used here (single-step solver); the iterative loop lives in `BoozerSurfaceJAX.run_code_traceable` which is called from `_run_traceable_solve`. The Lax discipline holds.
- **PyTree shape stability** — `_pack_traceable_forward_result` (line 3310-3333) returns a flat dict with stable keys; works fine with `jax.tree_util.tree_map` for `stop_gradient`.

#### H. dtype consistency

- `_as_jax_float64` and `_as_runtime_float64` are used pervasively. Float64 is enforced wherever `requires_x64()` is true (which is all production parity modes per `backend/runtime.py:132-191`).
- `_runtime_bool` wraps booleans into `jax.Array(bool)` (line 946-947). Good.

#### I. Memory and `block_until_ready`

- No `block_until_ready` calls in the module. For the host-boundary wrappers (`_make_traceable_host_objective` line 4528, `_make_traceable_host_value_and_grad` line 4580), the `_host_scalar` / `_host_array` calls force device → host sync. Correct.
- No `donate_argnums`. **MEDIUM** finding: see #2 in executive summary.

#### J. Specific issue: M5 boundary

- `_compute_value_from_solved_state` in `BoozerResidualJAX` (line 2388-2404), `IotasJAX` (line 2496-2497), `NonQuasiSymmetricRatioJAX` (line 2673-2675), `MajorRadiusJAX` (line 2557-2558) all use JAX-pure functions for value. **None invoke `surface.gamma()` or `label.J()` on the CPU side at evaluation time.** The CLAUDE.md claim of "CPU surface objects for value, JAX autodiff for gradient" appears to be aspirational. The actual implementation is a fully JAX-pure pipeline that reconstructs surface geometry from `solved_state.sdofs` via `_surface_geometry_from_dofs`. This is a design upgrade — single coherent pipeline — but the docs should be updated to match. **HIGH finding #5.**

Verdict: **PASS WITH NOTES**. The IFT adjoint correctness is solid; bundle cache invariants are correctly modeled; LS-PLU is load-bearing per contract; failure-mode propagation correctly surfaces NaN. Two documentation gaps (M5 boundary claim, `make_traceable_objective` host-input contract) and two best-practice gaps (no `donate_argnums`, condition-estimator comment clarity).

### `src/simsopt/objectives/stage2_target_objective_jax.py`

#### A. Simsopt-convention compliance

- This file does NOT subclass `Optimizable` — it builds a runtime bundle (`Stage2TargetObjectiveBundle`, line 136) that is consumed by the ondevice target lane. This is correct for a "JAX-native objective constructor" pattern; the CPU Stage 2 lane uses Optimizable composition (`SquaredFlux`, `LpCurveLength`, etc.), but the JAX target lane wraps the whole composite into one fused JIT.
- `Stage2TargetOptimizerState` (line 122-133) is a registered `jax.tree_util` dataclass — pytree-flat by design. Good.
- `final_specs_from_dofs` (line 721-738) and `_dynamic_curve_runtime_state` (line 681-719) reconstruct the coil set spec from the optimizer state without object mutation. Pure functional pattern.

#### B. JAX best-practices compliance

- **`jax.jit` placement** — `raw_terms`, `least_squares_residual`, `reporting_summary`, `objective`, `value_and_grad` all jitted at the public seam (line 1004-1006, 1189-1191). Good.
- **`_mark_cacheable_jit_value_and_grad`** (line 1145, 1191) is applied to all reused value/grad bundles so the private optimizer cache can dedupe. Good.
- **`_mark_structured_private_solver_cacheable`** (line 1193-1196) tags the bundle with a structural cache token. Good.
- **`jax.value_and_grad(_alm_objective_impl)`** (line 1146) — fuses the ALM evaluation and gradient. Good.
- **`jax.lax.scan` over coil-coil pairwise penalties** (`_pairwise_curve_distance_penalty_scan`, line 375-446) — correct use of scan for traceable pairwise reductions; falls back to Python double-loop only at construction time for `_fixed_curve_penalty` (already-fixed TF coils). **MEDIUM #8 — note construction-time quadratic cost.**
- **Sharding support** — `maybe_shard_pairwise_row_trees` is invoked (line 401-404), `field_sharding_summary` and `pairwise_penalty_sharding_summary` provide diagnostics.

#### C. Specific issues

- **`_selected_smoothmax`** (line 168-182) — uses `jax.nn.logsumexp` and clips temperature against `np.finfo(np.float64).eps`. Numerically safe. Good.
- `Stage2PenaltyConfig` uses `NamedTuple` with `definition` defaulting to `"quadratic flux"` (line 109-118). Good immutable contract.

Verdict: **PASS**.

### `src/simsopt/solve/permanent_magnet_optimization_jax.py`

#### A. Simsopt-convention compliance

- All public functions (`GPMO_baseline_jax`, `GPMO_multi_jax`, `GPMO_ArbVec_jax`, `GPMO_backtracking_jax`, `GPMO_ArbVec_backtracking_jax`, `relax_and_split_jax`) are thin adapters over `jax_core/pm_optimization.py` solver kernels (`gpmo_*_solve`, `mwpgp_solve`). **MATCHES the "thin adapter over jax_core" contract.**
- Result dataclasses (`GPMOBaselineResult`, `GPMOMultiResult`, etc.) are registered as `jax.tree_util` dataclasses (line 98, 127, 160, 196, 231, 269) — pytree-flat, work under `jit`/`grad`/`vmap`.
- No object mutation — wrappers return immutable result objects; the host-side caller decides whether to write back into a CPU `PermanentMagnetGrid`.

#### B. JAX best-practices compliance

- **Immutable result dataclasses** are `@dataclass(frozen=True)` (line 86, 112, 144, 178, 216, 248). Good.
- **`_is_tracing(value)`** (line 65-66) — uses `isinstance(value, jax.core.Tracer)`. Correct.
- **`_device_to_host_transfer_disallowed`** (line 69-73) — reads `jax.config.jax_transfer_guard_device_to_host` first, falls back to `jax_transfer_guard`. Correct.
- **`_raise_if_infeasible_initial_condition`** (line 76-83) is guarded by both `_is_tracing` and `_device_to_host_transfer_disallowed` (line 357-359) so it never raises under jit or strict transfer guard. Good.
- **No `block_until_ready`** — wrappers return jax arrays; consumers materialize lazily. Acceptable.
- **No `donate_argnums`** — would benefit `mwpgp_solve` for large `m_history` arrays, but is internal to `jax_core/pm_optimization`. Out of scope here.
- **`prox_l0_jax` / `prox_l1_jax`** (line 303-340) — branch-free (`jnp.where` instead of `if`), trace-clean. Good port of the CPU rules.
- **`projection_L2_balls_jax`** (line 292-300) — thin wrapper over `projection_l2_balls` (in `jax_core/pm_optimization`). Good.

Verdict: **PASS**. Thin adapter pattern is correctly applied.

### `src/simsopt/solve/wireframe_optimization_jax.py`

#### A. Simsopt-convention compliance

- `gsco_wireframe_jax` and `rcls_wireframe_jax` use the same parameter signatures as the CPU siblings. `optimize_wireframe_jax` dispatches on `algorithm.lower() == "rcls" | "gsco"`.
- Results (`WireframeRCLSResult`, `WireframeGSCOResult`) are immutable dataclasses registered as pytrees (line 43-47, 65-79).
- **Difference from PM**: this module is NOT a thin adapter — it contains the GSCO scan kernel directly (line 303-526). `jax_core/wireframe.py` is **field-kernel-only** (item 29), and the GSCO solve (item 31) was kept in `solve/wireframe_optimization_jax.py`. This is a deliberate split (per module docstring at `jax_core/wireframe.py:1-11`). **MATCHES contract**, but the layout diverges from `permanent_magnet_optimization_jax.py`'s thinner-adapter pattern.

#### B. JAX best-practices compliance

- **`jax.lax.scan` for the GSCO inner loop** (line 493-497) — correct primitive for a fixed-iteration count with carry state.
- **`jnp.where` instead of `if`** for `accept_loop`, `stop_now`, etc. (line 429-456) — correct trace-clean branching.
- **`jnp.argmin`** for candidate selection (line 404) — correct.
- **History arrays use `.at[...]` set updates** (line 437-443) — JAX-functional. Good.
- **Explicit `jax.transfer_guard("allow")`** in `_host_array` (line 164-168) — required to cross the device→host boundary under `disallow`. Correct.

#### C. Specific issues

- **`bnorm_obj_matrices_jax` falls back to CPU `ext_field.B()`** (line 673-677, see HIGH #7 above).
- **`_write_wireframe_currents`** (line 734-736) mutates the host CPU `wframe.currents` after the JAX solve completes. This is a deliberate "host snapshot of result" pattern; documented.

Verdict: **PASS WITH NOTES**. Layout split (kernel vs solver) is justified; one CPU-fallback seam should be documented or replaced.

## M5 IFT adjoint correctness across the LS/exact lanes

CLAUDE.md formula:

> `dJ/d_coils = ∂J/∂coils − adj^T ∂g/∂coils`

with `adj` solving the transposed inner linearization `(dg/dx_inner)^T adj = ∂J/∂x_inner`.

### `BoozerResidualJAX._value_and_dJ_by_dcoil_dofs` (`surfaceobjectives_jax.py:2356-2386`)

```
value, direct_gradient = _value_and_direct_coil_gradient(...)             # ∂J/∂coils
dJ_ds = self._compute_dJ_ds(coil_set_spec, iota, G, weight_inv_modB)      # ∂J/∂x_inner
adjoint = _solve_boozer_adjoint(adjoint_state, dJ_ds)                     # adj
adjoint_gradient = _adjoint_coil_dofs_gradient(stream_group_vjps, adjoint, ...)
                                                                          # adj^T ∂g/∂coils
return value, direct_gradient - adjoint_gradient                          # ∂J/∂coils − adj^T ∂g/∂coils
```

Sign and shape correct. Matches CPU `BoozerResidual.compute` (`surfaceobjectives.py:1414-1419`): `self._dJ = dJ_by_dcoils - adjoint_derivative`. **PASS**.

### `IotasJAX._value_and_dJ_by_dcoil_dofs` (`surfaceobjectives_jax.py:2479-2494`)

```
# J = iota, so ∂J/∂coils = 0 (no direct term)
# ∂J/∂x_inner = e_iota (unit cotangent at the iota position in x_inner)
lhs_dtype = _adjoint_state_dtype(adjoint_state)
n = _adjoint_state_decision_size(adjoint_state)
dJ_ds = _explicit_cotangent_basis(n, n-2 if solved_state.G is not None else n-1, dtype=lhs_dtype)
adjoint = _solve_boozer_adjoint(adjoint_state, dJ_ds)
adjoint_gradient = _adjoint_coil_dofs_gradient(...)
return solved_state.iota, -adjoint_gradient                                # 0 − adj^T ∂g/∂coils
```

Sign matches CPU `Iotas.compute` (`surfaceobjectives.py:1210-1215`): `self._dJ = -1.0 * adjoint_derivative`. **PASS**.

### `NonQuasiSymmetricRatioJAX._value_and_dJ_by_dcoil_dofs` (`surfaceobjectives_jax.py:2649-2671`)

```
value = self._compute_value(sdofs, coil_set_spec)                          # J(x*, coils)
direct_gradient = self._direct_coil_gradient(current_coil_dofs, sdofs)    # ∂J/∂coils
dJ_ds = self._compute_dJ_ds(coil_set_spec, sdofs, decision_size)          # ∂J/∂x_inner (with G-zeros tail)
adjoint = _solve_boozer_adjoint(adjoint_state, dJ_ds)                      # adj
adjoint_gradient = _adjoint_coil_dofs_gradient(...)
return value, direct_gradient - adjoint_gradient
```

Sign correct. Note: `_compute_dJ_ds` (line 2632-2647) pads the surface-only gradient with zeros for `iota` and `G` slots to match the decision-size — this is correct because `J_QS` does not depend on `iota` or `G`.

### `MajorRadiusJAX._value_and_dJ_by_dcoil_dofs` (`surfaceobjectives_jax.py:2540-2555`)

Same pattern as `NonQuasiSymmetricRatioJAX` but with `direct_gradient = 0` (line 2555: `return value, -adjoint_gradient`). Major radius depends only on surface DOFs, no direct coil term. **PASS**.

### LS-lane adjoint solve

- `_solve_boozer_adjoint` (line 1723-1733) → `_checked_boozer_linear_solve(transpose=True)` (line 1741-1763) → consumes `adjoint_state.solve_transpose_with_status` from `get_adjoint_runtime_state()`. The LS lane provides this via the operator-backed seam at `boozersurface_jax.py:3460-3519` (SOLVE via `_lu_solve_dense_hessian` on packed `(lu, piv)` factors).
- The LS factors `(P, L, U, lu, piv)` are stored under `lax.stop_gradient` in the traceable lane (per `f_fwd` line 4466) so the adjoint backward pass cannot retrace into the factorization graph. **MATCHES CLAUDE.md Phase 2 §5.3.**

### Exact-lane adjoint solve

- `_traceable_solve_exact_linearization` (`surfaceobjectives_jax.py:3246-3268`) defines a closure `residual_fn(x_inner) = _boozer_exact_residual(x_inner, coil_set_spec, **kwargs)` and hands it to `_optimizer_jax._solve_jacobian_system_with_status(residual_fn, solved_x, rhs, transpose=True, ...)`. This is an **operator-backed solve** — no dense PLU materialization.
- `production_operator` lane in `boozersurface_jax.py` exposes `solve_transpose_with_status` and `solve_forward_with_status` for the exact path; the JAX wrapper consumes them through `_resolved_boozer_adjoint_runtime_state` (line 2085-2090). **MATCHES CLAUDE.md "Exact Boozer scaling-limit contract".**

### Batched adjoint via `compute_standard_surface_objective_gradients` (line 2686-2798)

- Stacks the three RHS into `rhs_batch = jnp.stack((residual_rhs, iota_rhs, non_qs_rhs))` (line 2764).
- `_solve_boozer_adjoint_batch` (line 1766-1777) calls `_solve_boozer_adjoint` once per RHS via a Python list-comprehension — **not** a true `vmap`. CLAUDE.md notes "Batched exact adjoints in `production_operator` solve one RHS at a time through the same operator seam." Verified consistent with that contract.
- Per-RHS extraction at line 2769-2772 uses `jnp.split` and `jnp.squeeze` to keep work device-side — good for strict transfer guard.
- Sign convention preserved per term (line 2784-2786): residual = direct - adjoint, iota = -adjoint, non_qs = direct - adjoint.

**Verdict for M5 IFT correctness: PASS.** All four wrappers implement the IFT formula correctly with appropriate signs and shapes; LS and exact lanes both route through the SSOT `get_adjoint_runtime_state()`; failure-mode propagation correctly surfaces NaN gradients.

## Bundle cache invariants and `id()` discipline

### Cache key composition (`_TraceableRuntimeCacheKey` at `surfaceobjectives_jax.py:4138-4147`)

| Field | Source | Invalidation trigger |
|---|---|---|
| `solve_state_token` | `booz_jax._traceable_solve_state_token` (int counter) | Successful re-solve in `BoozerSurfaceJAX` (`boozersurface_jax.py:721, 3266`) |
| `coil_dof_state_token` | `bs_jax._coil_dof_state_token` (int counter) | Aggregate `x`/`full_x` writes (`biotsavart_jax_backend.py:499`, line 1050) AND ancestor `set_recompute_flag` (line 1052-1058) |
| `coil_layout_signature` | `_traceable_contract_tree_signature(coil_dof_extraction_spec)` | Structural — any change to the extraction spec rebuilds |
| `optimize_G` | Python bool | Statically baked in solver options |
| `predictor_kind` | `booz_jax.boozer_type` (`"ls"` or `"exact"`) | Switched by user explicitly |
| `objective_contract_signature` | `_traceable_contract_tree_signature(objective_kwargs)` | Any change to objective_kwargs (quadrature, label, target, weights, etc.) |
| `option_signature` | Hash of `booz_jax.options` for `_TRACEABLE_RUNTIME_OPTION_KEYS` | User-modified solver options |
| `success_filter_signature` | Structural via `_traceable_runtime_cache_signature` attribute, else `is` identity | Custom user filters |

### `id()` discipline

- **No `id()` calls** in cache-key construction. Verified.
- `_TraceableCallableSignature` uses `object.__hash__(self.callback)` for hashing — this is **NOT** `id(self.callback)`; it is the default object hash which is based on memory address but goes through `object.__hash__` rather than `id()`. The `__eq__` is `self.callback is other.callback`. CLAUDE.md says "compare with `is`, not `id(callable)` or user-defined callable equality" — **MATCHES**.
- `coil_layout_signature` uses `_traceable_contract_tree_signature` (line 3865) which descends into the tree and builds structural signatures for each leaf. Scalar leaves get exact values; large array leaves get `(dtype, shape)` metadata only — relying on `solve_state_token` and `coil_dof_state_token` to detect content changes. This is the documented "semantic sharing" pattern.
- `objective_contract_signature` uses the same `_traceable_contract_tree_signature` — same semantic-sharing trade-off, with the constraint that callers must not mutate captured `objective_kwargs` arrays in place (documented at `surfaceobjectives_jax.py:5578-5582`).

### Cache slot identity

- The cache slot lives on `booz_jax._traceable_runtime_entry_cache` (`surfaceobjectives_jax.py:4194-4196, 4226`). Initialized to `None` at `boozersurface_jax.py:722` and at `boozersurface_jax.py:3267`.
- One entry per `booz_jax` object; cache reads via `cached_entry["cache_key"] == cache_key` (line 4195).
- If a caller swaps `bs_jax` while reusing the same `booz_jax`, the `coil_dof_state_token` will differ and the cache rebuilds. Verified correct.
- ALM bundle cache uses an inner dict `runtime_entry["alm_runtime_bundles"]` keyed by `_traceable_contract_tree_signature(normalized_alm_config)` (line 5692-5694). Structural, safe.

**Verdict: PASS.** Bundle cache invariants are correctly modeled; no `id()` discipline violations.

## Backend selection at boundary (env, programmatic)

### Env-var contract

- `SIMSOPT_BACKEND` (`backend/runtime.py:35`) — primary.
- `STAGE2_BACKEND` (`runtime.py:36`) — legacy, resolved through `_resolve_legacy_value` (line 1161-1173). Primary wins.
- `SIMSOPT_JAX_PLATFORM` (line 37) — primary.
- `SIMSOPT_JAX_BACKEND` (line 38) — legacy, resolved through `_resolve_legacy_platform` (line 1176-1185). Primary wins.
- `SIMSOPT_BACKEND_MODE` (line 39) — new mode-based API; takes precedence over backend+platform pair if set (line 1213-1215).
- `SIMSOPT_BACKEND_STRICT` (line 40) — boolean. **MATCHES CLAUDE.md "Backend selection conventions".**

### Cache and re-read semantics

- `_cached_backend_config` (line 1198) is process-scoped. `get_backend_config()` populates on first call (line 1209-1230). Re-reads require `invalidate_backend_cache()` (line 1612-1623) which clears the config and all derived caches.
- `set_backend(mode, ...)` (line 1738-1773) updates the cache AND writes back to env vars via `_SYNCED_RUNTIME_ENV_VALUES` (line 89-100), so subprocess helpers see the new values.

### No implicit JAX configuration at import

- `apply_jax_runtime_config` (line 1716-1735) is an explicit call. Calls `jax.config.update` only when explicitly invoked.
- `should_eagerly_configure_jax` (line 1658-1663) gates eager configuration on `is_jax_backend() and (explicit selector env present)`. So a user that imports `simsopt.backend` without setting any env var sees no JAX side effects.
- Cuda parity-determinism check (`_validate_cuda_parity_determinism_env`, line 1693-1713) only raises if `mode == 'jax_gpu_parity'` or `strict=True`. Good.

### Programmatic API

- `get_backend()`, `is_jax_backend()`, `get_jax_platform()` — all read-at-call-time but cache-backed.
- `set_backend(mode, ...)` — atomic update of config + env + caches under `_backend_runtime_lock`.

**Verdict: PASS.** Backend selection at the boundary is well-engineered. The cache invalidation requirement is documented and self-consistent.

## Positive notes

- **`SquaredFluxJAX`** is a textbook M2 wrapper: fixed-surface JIT closure, surface-DOF fingerprint drift detection, field-points drift detection, strict native contract, ObjectiveFailure on singular gradient. Best-in-class.
- **Integral-BdotN custom-VJP for `local` singular branch** uses `inf_with_nan_jvp` to surface `inf` value with `nan` cotangent — preserves the CPU "ObjectiveFailure" semantics in a JIT-safe way.
- **`compute_standard_surface_objective_gradients`** (`surfaceobjectives_jax.py:2686-2798`) is a thoughtful batched-adjoint optimization that shares one solved baseline and one stacked RHS across three wrappers, then unpacks per-term with `jnp.split` on the device. Strict-transfer-guard-safe.
- **Bundle cache key tokenization** correctly separates "structural" (layout, dtype, shape) from "content" (solve_state_token, coil_dof_state_token). Large arrays don't need value-hashing on every lookup.
- **All result dataclasses** in `permanent_magnet_optimization_jax.py` and `wireframe_optimization_jax.py` are registered as `jax.tree_util` dataclasses (frozen + flat pytree). This is the canonical JAX result-container pattern.
- **`jax_core/wireframe.py` field kernel** vs. **`solve/wireframe_optimization_jax.py` GSCO/RCLS solver** are correctly split — kernel is item 29, solver is item 31, distinct concerns. The module docstring at `jax_core/wireframe.py:1-11` explicitly notes the boundary.
- **`stage2_target_objective_jax.py`** uses `_pairwise_curve_distance_penalty_scan` with `jax.lax.scan` and `maybe_shard_pairwise_row_trees` for sharding-aware reductions. Production-grade GPU-friendly design.

## Verdict per module

| Module | Verdict |
|---|---|
| `backend.py` + `backend/runtime.py` | **PASS** |
| `objectives/integral_bdotn_jax.py` | **PASS** |
| `objectives/fluxobjective_jax.py` | **PASS** |
| `objectives/stage2_target_objective_jax.py` | **PASS** |
| `geo/surfaceobjectives_jax.py` | **PASS WITH NOTES** (M5 boundary documentation gap, condition-estimator comment clarity, no `donate_argnums`, `make_traceable_objective` docstring lacks host-input contract statement) |
| `solve/permanent_magnet_optimization_jax.py` | **PASS** (thin adapter pattern correct) |
| `solve/wireframe_optimization_jax.py` | **PASS WITH NOTES** (CPU `ext_field.B()` fallback seam, layout split justified by item 29 vs item 31 contract) |

## Cross-check on flagged side issues

- **`B2EnergyJAX` / `LpCurveForceJAX`** — confirmed alias-identical at `src/simsopt/field/force.py:1320` (`B2EnergyJAX = B2Energy`) and line 2284 (`LpCurveForceJAX = LpCurveForce`). Out of scope for this audit (in `field/` not `objectives/`), but the OpenMemory note is accurate.

## Recommended fix list (priority order)

1. **Update CLAUDE.md M5 adapter pattern description** to match the actual implementation: "wrappers use the solved-state runtime summary plus JAX-pure surface reconstruction for both value and gradient; CPU surface objects are the spec/DOF source only" (HIGH #5).
2. **Add `donate_argnums=(0,)` to `_make_traceable_objective_from_compiled_bundle`'s outer `jax.jit(f)` and to `compiled_value_and_grad_for`** (`surfaceobjectives_jax.py:4084, 4492`) to free coil-DOF buffers on the hot path (HIGH #2).
3. **Tighten the condition-estimator comment** in `_traceable_solve_plu_linearization` (`surfaceobjectives_jax.py:3219-3231`) to state "the LS Hessian is symmetric by construction" rather than "only reached for LS lane" (HIGH #1).
4. **Document the host-input rejection contract** in `make_traceable_objective`'s docstring (`surfaceobjectives_jax.py:5235-5285`) — one line stating that callers must use the explicit `host_value_and_grad` wrapper or `jax.device_put` (MEDIUM #9).
5. **Decide on the `ext_field.B()` fallback in `bnorm_obj_matrices_jax`**: either document it as an accepted CPU compatibility seam or extend the native contract to consume a `BiotSavartJAX`-style `ext_field` natively (MEDIUM #7).
6. **Annotate `_traceable_runtime_entry_cache`'s cross-`bs_jax` lifecycle** in its docstring at `surfaceobjectives_jax.py:4174-4181` (HIGH #3).
