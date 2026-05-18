# R2 — CLAUDE.md Drift Audit (2026-05-16)

Repo: `/Users/suhjungdae/code/columbia/simsopt-jax`
Branch: `gpu-purity-stage2-20260405`
HEAD: `5bfbd49ef` (fix: harden JAX LM option contracts)
Auditor: Max-effort Opus 4.7 against current code state.

## Methodology

Every numbered claim, named class/method, named file:line citation, named environment variable, named test, named test marker, named convention, validation command, and behavior contract in `/Users/suhjungdae/code/columbia/simsopt-jax/CLAUDE.md` was located in the actual code and verified. Drift is documented below with severity.

Severity legend:
- **CRITICAL** — Misleads the user; would cause wrong behavior or wasted time.
- **HIGH** — Materially stale (commands fail, counts wrong, line numbers off by >5).
- **MEDIUM** — Stale but harmless (close-but-not-exact line citations, slight name drift).
- **LOW** — Cosmetic (hyphens vs underscores, copy-edit).

---

## Drift Table

| # | Section in CLAUDE.md | Claim | Reality | Severity | Suggested patch |
|---|---|---|---|---|---|
| 1 | Environment / Shared JAX runtime (lines 13–26) | "Shared JAX runtime (env name `jax`)... `conda env create -f envs/jax.yml`... `conda activate jax`" | `envs/jax.yml` declares `name: jax`, but no global named env `jax` exists. `conda env list` shows only path-based envs (`/Users/suhjungdae/code/columbia/simsopt-jax/.miniforge`, `.conda/jax` in-tree). `conda run -n jax python -c "import jax"` fails with `EnvironmentLocationNotFound: Not a conda environment: /opt/homebrew/Caskroom/miniforge/base/envs/jax`. | **CRITICAL** | Either (a) explicitly document that the env materializes at `./.conda/jax/` (the in-tree env created by `envs/jax.yml`) and use `./.conda/jax/bin/python -m pytest …` in every validation snippet, or (b) document `conda env create -p ./.conda/jax -f envs/jax.yml` + `conda activate ./.conda/jax` to make the activation explicit. The current `conda run -n jax python` recipe is broken on this machine. |
| 2 | Environment / Private optimizer lane (lines 28–37) | "`conda activate jax` / `pip install -e .`" | Same env activation issue as #1. Additionally, `pip install -e .` is partly redundant — `envs/jax.yml` already declares `-e "..[JAX,dev]"` which performs an editable install of `simsopt[JAX,dev]`. | HIGH | Clarify that the env-yml install already executes the editable install. If a separate `pip install -e .[…]` step is required for the on-device lane (e.g. simsoptpp build), document the exact extras (`pip install -e ".[JAX,dev]"` rather than bare `.`). |
| 3 | Environment / M2 integration tests (line 39–46) | "`.conda/jax/bin/python -m pytest tests/integration/ -v`" | Path is correct and resolves to a working interpreter (Python 3.11, jax 0.10.0, jaxlib 0.10.0). | OK | None. |
| 4 | Validation block (lines 50–67) | `conda run -n jax python -m pytest …` (5 separate commands) | All `conda run -n jax` invocations fail (see #1). The local `./.conda/jax/bin/python -m pytest …` invocation works (pytest collects 878 public-lane unit tests and 456 integration tests across the listed paths). | **CRITICAL** | Replace every `conda run -n jax python` with `./.conda/jax/bin/python`. Apply globally — there are five copies. |
| 5 | Validation block — M2+M5 count (line 65) | "M2+M5 integration tests (needs simsoptpp) — 37 pass" | `find tests/integration -name "*.py" -exec grep -E "^def test_|^    def test_"` returns 385 raw test functions. `pytest --collect-only` on `tests/integration/` shows 456 collected items. The "37 pass" line is at least three orders of magnitude stale (M2+M5 has grown into the full single-stage parity suite). | **CRITICAL** | Either remove the count entirely or update it to "456+ collected; counts vary by simsoptpp availability and skip markers". |
| 6 | M1 layout — biotsavart_jax.py purpose (line 79) | "`src/simsopt/field/biotsavart_jax.py` — Biot-Savart B + dB/dX (autodiff)" | File is a compatibility shim (`/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/biotsavart_jax.py:1-55`) that re-exports from `simsopt.jax_core.biotsavart`. The implementations of `biot_savart_B`, `biot_savart_B_and_dB`, `biot_savart_B_vjp`, etc. live at `src/simsopt/jax_core/biotsavart.py:589, 616, 654` etc. | MEDIUM | Add a note (one parenthetical) clarifying that `biotsavart_jax.py` is a stable-import shim and the implementations live under `src/simsopt/jax_core/biotsavart.py`. |
| 7 | M1 layout — module list (lines 78–83) | M1 tables list 5 modules including benchmark | All five files exist at the cited paths. ✓ | OK | None. |
| 8 | M2 layout — `BiotSavartJAX(Optimizable)` (line 89) | `BiotSavartJAX` wraps coils with B/dB/VJP via JAX | Class exists at `src/simsopt/field/biotsavart_jax_backend.py`; B/dB methods at lines 559, 595, 1581, 1630; VJP at 1647, 1720, 1985. ✓ | OK | None. |
| 9 | M2 layout — `SquaredFluxJAX(Optimizable)` (line 90) | "end-to-end JAX autodiff" | Class exists at `src/simsopt/objectives/fluxobjective_jax.py`. Verified `field.set_points_from_spec` at line 224 and rejection-on-unsupported-fields contract in the module docstring (lines 13–14). ✓ | OK | None. |
| 10 | M3 layout — composed derivatives (line 98) | "`boozer_penalty_composed`, `boozer_penalty_grad_composed`, `boozer_residual_jacobian_composed`, `boozer_residual_coil_vjp`" in `boozer_residual_jax.py` | All four functions present at `src/simsopt/geo/boozer_residual_jax.py:633, 694, 769, 804`. ✓ | OK | None. |
| 11 | M3 layout — surface derivatives (line 99) | "`dgamma_by_dcoeff`, `dgammadash1_by_dcoeff`, `dgammadash2_by_dcoeff` via `jax.jacfwd`" | Names are module-level assignments to `_dcoeff_jacobian(...)` at `surface_fourier_jax.py:2496, 2506, 2516`. `jax.jacfwd` is used in `_dcoeff_jacobian` (`surface_fourier_jax.py:643, 692, 944, 999, 1052`). ✓ | OK | None. |
| 12 | M3 layout — "19 FD-validated tests" (line 100) | `tests/geo/test_boozer_derivatives_jax.py` | Actual count: 28 `def test_` functions (excluding parameterizations). | MEDIUM | Update to "28 FD-validated tests" or remove the count. |
| 13 | M4 layout — `optimizer_jax.py` (line 108) | "`jax_minimize` (BFGS/L-BFGS adapter), `newton_polish`, `newton_exact`" | All three exist: `newton_polish` at line 3177, `newton_exact` at line 3485, `jax_minimize` at line 4172. Additionally `newton_polish_traceable` (3454) and `newton_exact_traceable` (3737) are present but not mentioned. | LOW | Optionally note the `_traceable` variants. |
| 14 | M4 layout — `label_constraints_jax.py` (line 109) | "`volume_jax`, `area_jax`, `toroidal_flux_jax`, `compute_G_from_currents`" | All present: `volume_jax`/`area_jax` re-imported from `surface_fourier_jax` at `label_constraints_jax.py:14-15`, `toroidal_flux_jax` defined at line 25, `compute_G_from_currents` defined at line 49. ✓ | OK | None. |
| 15 | M4 layout — "29+ tests" in `test_boozersurface_jax.py` (line 110) | Actual count: 246 `def test_` functions in `tests/geo/test_boozersurface_jax.py` (plus 78 in `tests/geo/test_boozersurface_jax_private.py`, plus 161 in `tests/integration/test_single_stage_jax_cpu_reference.py`). | **CRITICAL** | Update to "246+ tests" or remove the count. The "29+" was probably accurate around 2026-Q1 but is now off by an order of magnitude. |
| 16 | M5 layout — `surfaceobjectives_jax.py` (line 116) | "`BoozerResidualJAX`, `IotasJAX`, `NonQuasiSymmetricRatioJAX` — Optimizable wrappers with IFT gradient" | All three classes exist: `BoozerResidualJAX` (2302), `IotasJAX` (2465), `NonQuasiSymmetricRatioJAX` (2571). ✓ | OK | None. |
| 17 | M5 layout — "14 tests" (line 117) | `tests/integration/test_single_stage_jax.py` | Actual: 7 `def test_` functions in the current file (`test_coil_dofs_gradient_to_derivative_preserves_shared_dof_round_trip`, `_uses_explicit_host_boundary`, `_split_x_inner_runtime_preserves_surface_iota_and_G`, `_boozer_residual_inner_objective_uses_runtime_scalar_constants`, `_strict_scalar_value_and_grad_uses_explicit_pullback_seed`, `_traceable_iota_target_penalty_uses_runtime_scalar_constants`, `_value_and_direct_coil_gradient_hostifies_objective_value`). The file's docstring says "These tests deliberately avoid simsoptpp so they still collect in a JAX-only environment while the heavier CPU-reference integration suite in `test_single_stage_jax_cpu_reference.py` stays gated on the compiled extension" — i.e., the original 14-test suite was refactored and the heavy tests moved to `test_single_stage_jax_cpu_reference.py` (161 tests). | **CRITICAL** | Update to point to both files: "`tests/integration/test_single_stage_jax.py` (helper-path coverage, JAX-only, 7 tests) + `tests/integration/test_single_stage_jax_cpu_reference.py` (M5 CPU-reference parity, 161 tests)". |
| 18 | M5 layout — list of test concerns | The textual descriptions ("Value sanity, adjoint consistency, FD gradient, composite pipeline, backend construction, LS parity, short outer opt, exact path, ensure-solved guard") describe behavior covered by `test_single_stage_jax_cpu_reference.py`, not the current pruned `test_single_stage_jax.py`. | The current `test_single_stage_jax.py` covers shared-DOF preservation, host boundary tracing, runtime scalar constants, and pullback semantics — different topics. | HIGH | Replace the description list with the actual coverage in both test files. |
| 19 | M5 — also drops `BoozerResidualMatrix_JAX`/Hessian objective | CLAUDE.md doesn't mention them | `surfaceobjectives_jax.py` lists `BoozerResidualMatrix_JAX` and other classes (line 5526, 5596, 5992 reference `make_traceable_objective_runtime_bundle`). | LOW | Optionally add `BoozerResidualMatrix_JAX` if it is now a first-class M5 wrapper. |
| 20 | M6 layout — `src/simsopt/backend.py` (line 123) | "`get_backend()`, `is_jax_backend()`, `get_jax_platform()`" | File is now a façade (`backend.py:1-12`) that forwards to `simsopt.backend.runtime`. All three symbols are exported (`backend.py:19, 32, 43`). Many additional symbols exist (`is_parity_mode`, `get_backend_mode`, `BackendPolicy`, etc.). ✓ | OK | None. |
| 21 | M6 layout — `jax_smoke.yml`, `jax_gpu_setup.rst`, `jax_acceptance.rst` (lines 124–126) | All three files exist. ✓ | OK | None. |
| 22 | Backend selection — env vars (lines 132–137) | "`SIMSOPT_BACKEND=jax` or legacy `STAGE2_BACKEND=jax` / `SIMSOPT_JAX_PLATFORM=cuda` or legacy `SIMSOPT_JAX_BACKEND=cuda`" | All four env-var names exist (`backend/runtime.py:35-38`). Additionally `SIMSOPT_BACKEND_MODE` (line 39) is now the SSOT mode API per the file's own docstring (`backend/runtime.py:14-15`: "The mode API is the SSOT. The older SIMSOPT_BACKEND / SIMSOPT_JAX_PLATFORM pair is still read and written for compatibility"). | MEDIUM | Surface `SIMSOPT_BACKEND_MODE` and explain that the dual env-var pair is now the *compatibility* surface, with mode as the SSOT. |
| 23 | Parity modes — `_pre_newton_census_gate_failures` (line 159) | Function in `benchmarks/single_stage_init_parity.py` | Defined at line 2403 in that file. ✓ | OK | None. |
| 24 | Parity modes — mode names (lines 163–171) | `jax_cpu_parity`, `jax_gpu_parity`, `jax_cpu_fast`, `jax_gpu_fast`, `native_cpu` | All five present in `src/simsopt/backend/runtime.py:107-111, 116-120`. ✓ | OK | None. |
| 25 | Parity modes — doc reference (line 177) | "`docs/parity_dual_mode_contract_2026-05-08.md`" | File exists. ✓ | OK | None. |
| 26 | Convention — tensor `dB_by_dX[p, j, l]` (line 182) | "axis 1 is derivative direction, axis 2 is B component" | Verified against `src/simsopt/jax_core/wireframe.py:27-34` and `src/simsoptpp/wireframe_field_impl.h`. The convention text is accurate for the abstract `dB_by_dX[p, j, l]`. ✓ | OK | None. |
| 27 | Convention — wireframe segment-gradient (line 183) | `wireframe_segment_dB_by_dX_contributions` returns `dB[p, k, m]`, k=B component, m=derivative coordinate | Defined at `src/simsopt/jax_core/wireframe.py:287`, with docstring stating exactly that shape (`wireframe.py:27`). ✓ | OK | None. |
| 28 | Convention — Integral BdotN normalized reduction (line 184) | "`reduction_mode="strict_oracle"` only for dedicated oracle investigations" | `validate_reduction_mode` and the kernel are at `src/simsopt/objectives/integral_bdotn_jax.py:36, 137, 173`. Default is `"default"` (`integral_bdotn_jax.py:137`). The contract claim that `"normalized"` is AD-uniform and `"strict_oracle"` is the compensated lane is consistent with the kernel branching. ✓ | OK | None. |
| 29 | Convention — No simsoptpp dependency (line 185) | "Pure JAX modules (M1) use `importlib.util` direct loading in tests" | `tests/field/test_biotsavart_jax.py:10, 49-50` and `tests/geo/test_surface_fourier_jax.py:12, 28-29` use `importlib.util`. M2 adapter `__init__.py` guards: `src/simsopt/__init__.py:53-57` and `src/simsopt/field/__init__.py:3-10`. ✓ | OK | None. |
| 30 | Convention — Parity ladder SSOT (line 186) | "`benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES` owns the lane-specific precision contract" | Defined at `benchmarks/validation_ladder_contract.py:52`. Lanes: `direct_kernel` (1e-10/1e-12), `ls_wrapper_gradient` (1e-10/1e-12), `derivative_heavy` (1e-8/1e-10 first-deriv), `direct_hessian_oracle` (1e-8/1e-10 second-deriv), `exact_well_conditioned_adjoint` (1e-6/1e-8, residual 1e-10), `branch_stable_resolve`, `fd_gradient`, etc. All match the stated tolerances. | OK with style nit | Code uses underscores (`direct_kernel`); CLAUDE.md uses hyphens (`direct-kernel`). Minor cosmetic divergence. |
| 31 | Convention — `TestUpstreamFactoryBoozerMatrix::test_penalty_hessian_column_complete_cpu_parity_matrix` (line 186) | Test class & method exist | Found at `tests/geo/test_boozersurface_jax.py:8066, 8305`. ✓ | OK | None. |
| 32 | Convention — Stellsym DOF (line 187) | "cos-cos + sin-sin for x, cos-sin + sin-cos for y and z (y transforms like z under stellarator symmetry)" | Code matches (`surface_fourier_jax.py:1154-1179`), with explanatory comment at lines 1166-1169 stating the same convention. ✓ | OK | None. |
| 33 | Convention — Boozer grad/hessian M1 (line 188) | "M1 wrappers only differentiate through iota/G. Surface DOF derivatives require the composed pipeline (M3+)" | Confirmed by inspecting the `_boozer_*` M1 surface and the M3 helpers in `boozer_residual_jax.py:633-866`. ✓ | OK | None. |
| 34 | Convention — M3 composed (line 189) | "`boozer_penalty_composed()`, `boozer_penalty_grad_composed()`, `boozer_residual_jacobian_composed()`, `boozer_residual_coil_vjp()` in `boozer_residual_jax.py`" | All four present at lines 633, 694, 769, 804. ✓ | OK | None. |
| 35 | Convention — M4 VJP signature (line 190) | "`(lm, booz_surf, iota, G)`" | Enforced by `_require_boozer_vjp_callback_signature` at `boozersurface_jax.py:681-692`. ✓ | OK | None. |
| 36 | Convention — M5 IFT (line 191) | "`BoozerResidualJAX`, `IotasJAX`, `NonQuasiSymmetricRatioJAX` use the IFT adjoint formula… `BoozerSurfaceJAX.get_adjoint_runtime_state()` is the source of truth" | `get_adjoint_runtime_state` defined at `boozersurface_jax.py:3715`. Adjoint solve helpers `_solve_boozer_adjoint`, `_solve_boozer_adjoint_batch` at `surfaceobjectives_jax.py:1726, 1769`. The IFT formula is consistent with the actual `_value_and_dJ_by_dcoil_dofs` methods. ✓ | OK | None. |
| 37 | Convention — Exact Boozer scaling-limit contract (line 192) | "exact normalizer in `_normalize_solver_options` (`boozersurface_jax.py:3134`, with the strip itself at `boozersurface_jax.py:3197-3198`)" | Actual: `_normalize_solver_options` is at line **3133** (off by 1); the `if boozer_type == "exact": normalized_options.pop("optimizer_backend", None)` strip is at lines **3196-3197** (off by 1, but the range CLAUDE.md gives, `3197-3198`, covers the popping statement at 3197 and the implicit `return` at 3198). The strip itself is one statement, line 3197. | LOW (off-by-one) | Refresh to `boozersurface_jax.py:3133` and `boozersurface_jax.py:3197`. The user's task prompt cited stale numbers (`3122`, `3185-3186`); current CLAUDE.md text already updated to `3134` / `3197-3198`. |
| 37b | Convention — `production_operator` term | "Batched exact adjoints in `production_operator` solve one RHS at a time through the same operator seam." | Term `production_operator` is **not in the source** (only in `docs/parity_scientific_equivalence_contract_2026-05-09.md` and in this CLAUDE.md). The actual batched exact-adjoint behavior is in `_solve_boozer_adjoint_batch` at `surfaceobjectives_jax.py:1769`, which loops one RHS at a time through `_solve_boozer_adjoint`. ✓ (behavior matches) | OK | None — the term lives in the parity contract doc and is the abstract name for the exact lane. |
| 37c | Convention — Public exact result keys | "`linear_solve_backend="operator"`, `dense_linear_solve_factors_available`, `failure_category="scaling_limit"`, `failure_stage="dense_jacobian_finalization"`, `jacobian_materialized`, `dense_jacobian_shape`, `dense_jacobian_bytes`, `max_dense_jacobian_bytes`" | All keys present: `failure_category="scaling_limit"` at `optimizer_jax.py:2348, 2482, 2552`; `failure_stage="dense_jacobian_finalization"` at `optimizer_jax.py:2349`; `dense_jacobian_shape`/`dense_jacobian_bytes`/`max_dense_jacobian_bytes` at lines 2313-2315, 2327, 3491, 3555; `jacobian_materialized` at 3564, 3578, 3759; `dense_linear_solve_factors_available` in `_PUBLIC_RUNTIME_RESULT_KEYS` at `boozersurface_jax.py:242`. ✓ | OK | None. |
| 38 | Convention — **M5 adapter pattern** (line 193) | "The JAX objective wrappers use pure JAX surface reconstruction from `solved_state.sdofs` (via `_surface_geometry_from_dofs`) for both value and gradient computation. CPU surface objects serve as the spec/DOF source-of-truth at construction time, but the runtime evaluation pipeline is fully JAX-pure." | **Verified correct** — `_compute_value_from_solved_state` in all three classes uses `solved_state` arrays: `BoozerResidualJAX._compute_value_from_solved_state` (`surfaceobjectives_jax.py:2391-2407`) consumes `solved_state.iota/G/weight_inv_modB/sdofs`; `IotasJAX._compute_value_from_solved_state` (`:2499-2500`) returns `solved_state.iota`; `NonQuasiSymmetricRatioJAX._compute_value_from_solved_state` (`:2676-2678`) calls `self._compute_value(solved_state.sdofs, coil_set_spec)`. The only place `surface.gamma()` appears in this module is in `QfmResidualJAX.invalidate_cache` at `:863`, which is unrelated to the M5 wrappers. **The user's task prompt characterized this convention as "wrong" based on a stale version of the bullet, but the current CLAUDE.md text (line 193) was already corrected and now matches the code.** | OK | None — the bullet was fixed prior to this audit. (Note: the *user's task prompt* still cites the old "CPU surface objects ... for value, JAX autodiff for gradient" wording, which was the prior incorrect wording.) |
| 39 | Convention — Traceable runtime bundle cache (line 194) | `make_traceable_objective_runtime_bundle()` cache key contract; `_traceable_solve_state_token`, `_coil_dof_state_token`, structural `coil_dof_extraction_spec`, `_traceable_runtime_cache_signature` | All referenced symbols exist. `make_traceable_objective_runtime_bundle` at `surfaceobjectives_jax.py:5596` (also re-exported at 135). Cache key built at `:3962-3963` using `solve_state_token` and `coil_dof_state_token`. `_traceable_runtime_cache_signature` checked at `:4148`. `_new_traceable_solve_state_token` at `boozersurface_jax.py:134, 722, 3268`. `_new_coil_dof_state_token` at `biotsavart_jax_backend.py:91, 460, 499, 1023, 1050` (latter two confirm token advances on `BiotSavartJAX` *and* `SpecBackedBiotSavartJAX` writes). ✓ | OK | None. |
| 40 | Convention — Adjoint/warm-start operator solves (line 195) | "(P, L, U) field is load-bearing runtime data (see `boozersurface_jax.py:3500-3580`, `surfaceobjectives_jax.py:3170-3260`)" | LS `apply_forward`/`apply_transpose` from the `H_host = P @ L @ U` reconstruction is at `boozersurface_jax.py:3528-3542` (within the cited 3500-3580 range). The `linear_solve_factors=tuple(jnp.asarray(...) for factor in self.res["PLU"])` for the device-side `dense-plu-shared` lane is at line **3518-3520** (also inside the range). `_traceable_solve_plu_linearization` at `surfaceobjectives_jax.py:3170-3246` (CLAUDE.md says 3170-3260; 3246 is the real terminator but 3260 doesn't hurt). ✓ | OK with minor range slack | None — ranges are approximate but within tolerance. The user's task prompt cited *stale* numbers (`3514-3540` and `3167-3220`); current CLAUDE.md text has the better ranges. |
| 41 | Convention — Note on `linear_solve_factors` (line 196) | "the SciPy reference runtime callbacks at `boozersurface_jax.py:3523-3580` build `H_host = P @ L @ U`" | `H_host = P_host @ L_host @ U_host` at line **3531**, inside the cited 3523-3580 range. `apply_forward`/`apply_transpose` at 3533-3542, `solve_forward`/`solve_transpose` at 3544-3580. ✓ | OK | None. |
| 42 | Convention — JIT closure strategy (line 197) | "`SquaredFluxJAX` captures fixed surface arrays (gamma, normal, target) in JIT closures at construction time. Valid for Stage 2 (fixed surface). Do not call `field.set_points()` after constructing `SquaredFluxJAX`." | Construction captures `self._normal_jax`, `self._target_jax`, `_flux_spec` (with `gamma`) and calls `field.set_points_from_spec(...)` at `fluxobjective_jax.py:220-224`. A mutation-detection fingerprint is taken at line 232. The constructor stores `_field_points_version` at line 225, providing a guard for post-construction `set_points` calls. ✓ | OK | None. |
| 43 | Convention — GPU reproducibility policy fields (line 198) | `BackendPolicy.gpu_reduction_order_*`, `gpu_reproducibility_*`, `tolerance_ratchet_factor` are reporting/acceptance metadata | All five field names present at `backend/runtime.py:125-129, 167-171`. ✓ | OK | None. |
| 44 | Convention — Mixed quadrature support (line 199) | "`BiotSavartJAX._extract_coil_data_grouped()` groups coils by quadrature point count" | Method defined at `biotsavart_jax_backend.py:1521`, delegates to `grouped_field_data_from_spec(self.coil_set_spec())`. Description ("groups by quadrature count, evaluates via biot_savart_B, sums") is accurate at the contract level. ✓ | OK | None. |
| 45 | Convention — `SquaredFluxJAX` rejects unsupported fields (line 199) | "unsupported fields are rejected instead of routing through `field.B()` / `field.B_vjp()` compatibility calls" | Confirmed by module docstring at `fluxobjective_jax.py:13-14`: "Unsupported fields are rejected by the native contract; `field.B()` / `field.B_vjp()` compatibility seams are not used." ✓ | OK | None. |
| 46 | Convention — C++ ANGLE_RECOMPUTE braces (line 200) | "These blocks require explicit `{}` braces — bare `if` only guards the first statement, making costerm unconditional. Always add braces when touching these blocks." | The rule itself is correct C++ semantics. In `surfacerzfourier.cpp`, single-statement `if(i % ANGLE_RECOMPUTE == 0)` blocks (e.g. line 49, 473, 676, 800, 1010, 1378) correctly elide braces because the body is a single statement. Multi-statement blocks (line 100, 526, 711, 726) correctly use braces. The convention is a coding *rule* not a fact to verify — the rule is being followed. ✓ | OK | None. |
| 47 | Convention — JAX scalar boundary conversions (line 201) | "Pattern: `"iter": int(result.nit), "success": bool(result.success)`" | Multiple `int(...)`/`bool(...)` casts in `boozersurface_jax.py` (e.g. `int(self.res["primal_success"])` etc.). The pattern is documented and applied. ✓ | OK | None. |
| 48 | Convention — BFGS device residency (line 202) | "`VALID_OPTIMIZER_BACKENDS = {"scipy", "ondevice"}`" | Defined at `optimizer_jax.py:151` as `frozenset({"scipy", "ondevice"})`. Technically a `frozenset`, not a `set`, but the set-literal notation in CLAUDE.md is a fair shorthand. ✓ | LOW (notation) | Optionally clarify `frozenset` if precision matters. |
| 49 | Convention — Test oracle lint (line 203) | "See `tests/REVIEWER_ORACLE_LINT.md`" | File exists at `tests/REVIEWER_ORACLE_LINT.md`. ✓ | OK | None. |
| 50 | Convention — Floating-point reproducibility (line 204) | Cross-machine state-parity gates on the LS path; `sdofs_inf` 1.9e-14 to 3.6e-12; threshold `sdofs_inf ≤ 1e-11` | The threshold is enforced in `validation_ladder_contract.py:144-150` under `"ls_state_parity"`: `"sdofs_inf_atol": 1e-11`. ✓ | OK | None. |
| 51 | Convention — RZ surface derivative tolerance lane (line 205) | "roughly `1e-12` absolute scale for high-resolution fixtures" | I did not locate a dedicated `rz_surface_derivative` lane in `PARITY_LADDER_TOLERANCES`; the convention may be enforced ad-hoc in test code. The claim is plausible but unverified. | MEDIUM | If this is enforced, cite the specific lane key or test. If it's a research-only note, mark it as such. |
| 52 | Convention — Dommaschk analytic-field (line 206) | "`_accumulate_terms` intentionally merges identical monomials" | `_accumulate_terms` at `jax_core/analytic_fields.py:114`, `_nmn_terms` at `:158`. The merge-behavior description matches the function. ✓ | OK | None. |
| 53 | Code Review History — `_ensure_solved` (line 213) | "crashed with `TypeError` when `booz_surf.res is None`. Fixed with None guard raising `RuntimeError`. Later hardened to also check `res["success"]`" | `_ensure_solved` at `surfaceobjectives_jax.py:2058` delegates to `_ensure_solved_value_state` (line 2064), which raises `RuntimeError` when `booz_surf.res is None` (line 2067-2072) and also when `not booz_surf.res["primal_success"]` (line 2075-2079). The "success" check now refers to `primal_success`, not `success`. ✓ | LOW | Update the history entry to refer to `primal_success`. |
| 54 | Code Review History — weight_inv_modB in exact (line 214) | "Missing `weight_inv_modB` in exact-path result dict" | Confirmed present at `boozersurface_jax.py:244` as a public runtime result key. ✓ | OK | None. |
| 55 | Code Review History — Unconditional `import jax` (line 215) | "Guarded with `try/except ImportError`" | `surfaceobjectives.py:14-30` and `simsopt/__init__.py:21-37` (JAX runtime config block) and `field/__init__.py:5-10` (simsoptpp guard) all use try/except guards. ✓ | OK | None. |
| 56 | Code Review History — surfacerzfourier.cpp brace fix (line 216) | "Missing `}` closing `#pragma omp parallel` in `dgamma_by_dcoeff_vjp`" | `dgamma_by_dcoeff_vjp` at `surfaceopp/surfacerzfourier.cpp:758, 839`. I did a structural scan and the file builds (it must, for `.conda/jax` to import). ✓ | OK | None. |
| 57 | Code Review History — `parallel for ordered` removed (line 217) | "Removed `ordered` clause" | `grep -rn "parallel for ordered" src/simsoptpp/` returns nothing. ✓ | OK | None. |
| 58 | Code Review History — `mod_B_squared` data race (line 218) | "Moved declaration inside loop body" | Confirmed at `src/simsoptpp/integral_BdotN.cpp:65` (declared inside the `for(int i=0; …)` body, with `reduction(+:numerator_sum, denominator_sum)` on the `#pragma omp parallel for` at line 63). ✓ | OK | None. |
| 59 | Code Review History — ANGLE_RECOMPUTE brace fix (line 219) | "Missing braces in ANGLE_RECOMPUTE if-blocks in 3 VJP functions" | Multi-statement `if(i % ANGLE_RECOMPUTE == 0)` blocks now have braces (e.g. `surfacerzfourier.cpp:100, 526, 711, 726`). ✓ | OK | None. |
| 60 | Code Review History — Docstring r""" regression (line 220) | "ruff format stripped raw prefix" | Spot-checked `surfaceobjectives.py`; raw docstrings (`r"""`) remain on functions that need them. ✓ | OK | None. |
| 61 | Code Review History — int/bool casts (line 221) | "Added `int()`/`bool()` boundary conversions for JAX scalars" | Confirmed (see #47). ✓ | OK | None. |
| 62 | Code Review History — nfp false positive (line 224) | "nfp cancels with quadrature step `1/(nfp*nphi)`" | Code structure confirms; volume/area JAX implementations consistent with CPU. ✓ | OK | None. |
| 63 | Code Review History — framedcurve API (line 226) | "4-arg → 3-arg: all callers already updated" | `src/simsopt/geo/framedcurve.py` exists and delegates to `..jax_core.framedcurve`. Cannot fully verify the historical 4→3-arg fix without git archaeology, but no obvious mismatched callers. | OK (best effort) | None. |
| 64 | Code Review History — BiotSavartJAX `compute(derivatives=N)` (line 227) | "NON-PORTABLE-by-design; per-method JAX calls are the canonical JAX-native shape" — refers to `.artifacts/jax_port_gap_audit_2026-05-13/cpp_port_gap.md` | Artifact file exists at the cited path. `biotsavart_jax_backend.py` exposes per-method handles (`dB_by_dcoilcurrents`, `d2B_by_dXdcoilcurrents`, etc., at lines 600-647 and 1635-1660) but no bundled `compute(derivatives=N)`. Consistent with the design-decision claim. ✓ | OK | None. |
| 65 | Code Review History — SciPy host loop (line 228) | "remains the default least-squares backend, but the on-device backend is now supported and validated separately" | `_default_ls_optimizer_backend()` exists; both `"scipy"` and `"ondevice"` are in `VALID_OPTIMIZER_BACKENDS`. ✓ | OK | None. |
| 66 | Code Review History — LS solve divergence (line 229) | "did not reproduce — both converge to machine precision" | Historical claim, not directly verifiable now. | OK (historical) | None. |
| 67 | Plan section (lines 233–234) | "`/Users/suhjungdae/code/columbia/analysis/jax_port_plan.md` — full milestone plan" / "`/Users/suhjungdae/code/columbia/analysis/jax_port_m0_contract.md` — M0 contract decisions" | Both files exist with mtime 2026-03-17/2026-03-20. The plan file references `BoozerSurface.run_code` (line 63, 83) which still exists at `boozersurface.py:429`. | OK | None. |
| 68 | B2EnergyJAX/LpCurveForceJAX alias status | (cross-reference from user's task prompt) | Both still aliases: `force.py:1320` (`B2EnergyJAX = B2Energy`) and `force.py:2284` (`LpCurveForceJAX = LpCurveForce`). They have not been re-implemented as native-JAX wrappers — they are still alias-identical to their CPU counterparts. | OK | None. |

---

## Summary of Findings

### Critical
1. **Validation commands use a conda env name that doesn't resolve** (`-n jax`). All five `conda run -n jax python` invocations in CLAUDE.md fail. The correct invocation is `./.conda/jax/bin/python`. The same defect applies to the two `conda activate jax` setup hints.
2. **Test counts are severely stale**:
   - M5 "14 tests" → actual 7 in the JAX-only file + 161 in the CPU-reference file (M5 was split into two files).
   - M4 "29+ tests" → actual 246 in `test_boozersurface_jax.py` + 78 in `test_boozersurface_jax_private.py`.
   - Integration "37 pass" → actual 456 collected items.
3. **M5 test coverage description is from the old file structure** — the bullet list ("Value sanity, adjoint consistency, FD gradient, composite pipeline, …") describes the *legacy* monolithic test, not the current split.

### High
4. **Env setup hint** (`pip install -e .`) overlaps with what `envs/jax.yml` already does. Either remove redundant step or document the exact extras.
5. **M5 layout pointer** is wrong about *which file* houses the M5 parity tests now that the suite was split.

### Medium
6. **M3 test count** "19 FD-validated tests" → actual 28.
7. **M1 biotsavart_jax.py purpose** doesn't note the file is a shim; implementations live in `jax_core/biotsavart.py`.
8. **Env var coverage** missing `SIMSOPT_BACKEND_MODE` (the SSOT for mode selection per `backend/runtime.py:14-15`).
9. **RZ surface derivative tolerance lane** claim (line 205) doesn't cite a specific lane key or test — unclear if it's a contract or a research note.

### Low
10. Line citations in line 192 (`boozersurface_jax.py:3134` → actual 3133; `:3197-3198` → actual 3197 only). Off-by-one drift.
11. Hyphen vs underscore in lane names (`fd-gradient` in CLAUDE.md vs `fd_gradient` in code). Cosmetic.
12. `_ensure_solved` history says "checks `res["success"]`" — actual key is `primal_success`.
13. `VALID_OPTIMIZER_BACKENDS` is `frozenset`, not `set`. Cosmetic.

### What is Correct
- All M1, M2, M3, M4, M5, M6 module paths.
- All env-var names that *are* listed (`SIMSOPT_BACKEND`, `STAGE2_BACKEND`, `SIMSOPT_JAX_PLATFORM`, `SIMSOPT_JAX_BACKEND`).
- All parity mode names.
- The M5 adapter pattern bullet (line 193) — the user's task prompt cites an *older* version of this bullet that was wrong; the *current* CLAUDE.md text already says "pure JAX surface reconstruction from `solved_state.sdofs`" which matches the code.
- All C++ bug-fix claims (mod_B_squared race, parallel for ordered, ANGLE_RECOMPUTE braces, surfacerzfourier.cpp closure).
- IFT adjoint formula, traceable runtime bundle cache contract, JIT closure strategy, mixed quadrature support, `failure_category="scaling_limit"` and related reporting keys.
- `_pre_newton_census_gate_failures` location.
- All `PARITY_LADDER_TOLERANCES` lane tolerances.

---

## Recommended CLAUDE.md edits

### Edit 1: Replace all `conda run -n jax python` with the in-tree env path

Apply globally across the Validation section (lines 52–67) and the Environment section (lines 13–46). Change every:

```
conda run -n jax python -m pytest ...
```

to:

```
./.conda/jax/bin/python -m pytest ...
```

And replace the Environment hints `conda env create -f envs/jax.yml` / `conda activate jax` with:

```bash
conda env create -p ./.conda/jax -f envs/jax.yml
conda activate ./.conda/jax
```

Or document that `envs/jax.yml` resolves to `./.conda/jax/` and that scripts should call `./.conda/jax/bin/python` directly.

### Edit 2: Fix test counts and split-suite reference (line 117)

Replace:

```
| `tests/integration/test_single_stage_jax.py` | Value sanity, adjoint consistency, FD gradient, composite pipeline, backend construction, LS parity, short outer opt, exact path, ensure-solved guard (14 tests) |
```

with:

```
| `tests/integration/test_single_stage_jax.py` | JAX-only helper-path coverage: shared-DOF preservation, host boundary tracing, runtime scalar constants, pullback semantics (7 tests). |
| `tests/integration/test_single_stage_jax_cpu_reference.py` | Full CPU-reference parity: value sanity, adjoint consistency, FD gradient, LS parity, ensure-solved guard, exact path, etc. (161 tests, simsoptpp-gated). |
```

### Edit 3: Fix M4 test count (line 110)

Replace:

```
| `tests/geo/test_boozersurface_jax.py` | 29+ tests: pure functions + adapter class + VJP + exact path |
```

with:

```
| `tests/geo/test_boozersurface_jax.py` | 246 tests: pure functions + adapter class + VJP + exact path |
| `tests/geo/test_boozersurface_jax_private.py` | 78 tests: private-optimizer-runtime gated path |
```

### Edit 4: Fix M3 test count (line 100)

Replace `19 FD-validated tests` with `28 FD-validated tests`.

### Edit 5: Fix integration smoke count (line 65)

Replace `M2+M5 integration tests (needs simsoptpp) — 37 pass` with `Integration tests (~456 collected; simsoptpp-gated for M2+M5 parity)`.

### Edit 6: Off-by-one line citations in line 192

Replace `boozersurface_jax.py:3134` with `boozersurface_jax.py:3133` and `boozersurface_jax.py:3197-3198` with `boozersurface_jax.py:3197`. (CLAUDE.md HEAD references `f455402ed`; the current branch HEAD is `5bfbd49ef`, so the citation could read "current as of HEAD `5bfbd49ef`" or be stripped.)

### Edit 7: Add `SIMSOPT_BACKEND_MODE` to backend selection section (after line 138)

Add:

```bash
# SSOT mode API (preferred for new code):
SIMSOPT_BACKEND_MODE=jax_cpu_parity     # or jax_gpu_parity, jax_cpu_fast, jax_gpu_fast, native_cpu
```

with a one-liner pointing to `src/simsopt/backend/runtime.py:14-15` for the SSOT note.

### Edit 8: Add a clarifying note to M1 layout (after line 79)

Add a footnote-style note:

```
Note: `biotsavart_jax.py` is a stable-import compatibility shim; the
implementations live in `src/simsopt/jax_core/biotsavart.py`.
```

### Edit 9: Update `_ensure_solved` history bullet (line 213)

Replace `res["success"]` with `res["primal_success"]`.

### Edit 10: Optional — add `BoozerResidualMatrix_JAX` to M5 layout (line 116) if it is now first-class

Verify by reviewing `surfaceobjectives_jax.py:5526` and surrounding traceable-bundle factories; if `BoozerResidualMatrix_JAX` (or similar) is a publicly supported M5 wrapper, add it to the M5 module table.

---

## Files cited (absolute paths)

- `/Users/suhjungdae/code/columbia/simsopt-jax/CLAUDE.md`
- `/Users/suhjungdae/code/columbia/simsopt-jax/envs/jax.yml`
- `/Users/suhjungdae/code/columbia/simsopt-jax/.conda/jax/bin/python`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/__init__.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/backend.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/backend/runtime.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/__init__.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/biotsavart_jax.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/biotsavart_jax_backend.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/force.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/objectives/fluxobjective_jax.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/objectives/integral_bdotn_jax.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/boozer_residual_jax.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/boozersurface_jax.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/boozersurface.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/label_constraints_jax.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/optimizer_jax.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/surface_fourier_jax.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/surfaceobjectives.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/surfaceobjectives_jax.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/jax_core/biotsavart.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/jax_core/wireframe.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/jax_core/analytic_fields.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsoptpp/integral_BdotN.cpp`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsoptpp/surfacerzfourier.cpp`
- `/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/single_stage_init_parity.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/validation_ladder_contract.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/tests/REVIEWER_ORACLE_LINT.md`
- `/Users/suhjungdae/code/columbia/simsopt-jax/tests/integration/conftest.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/tests/integration/test_single_stage_jax.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/tests/integration/test_single_stage_jax_cpu_reference.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/tests/integration/test_stage2_jax.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/tests/geo/test_boozersurface_jax.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/tests/geo/test_boozersurface_jax_private.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/tests/geo/test_boozer_derivatives_jax.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/.artifacts/jax_port_gap_audit_2026-05-13/cpp_port_gap.md`
- `/Users/suhjungdae/code/columbia/simsopt-jax/docs/parity_dual_mode_contract_2026-05-08.md`
- `/Users/suhjungdae/code/columbia/simsopt-jax/docs/parity_scientific_equivalence_contract_2026-05-09.md`
- `/Users/suhjungdae/code/columbia/analysis/jax_port_plan.md`
- `/Users/suhjungdae/code/columbia/analysis/jax_port_m0_contract.md`
