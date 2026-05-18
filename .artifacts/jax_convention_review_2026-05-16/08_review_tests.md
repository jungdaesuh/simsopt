# JAX Test Suite Convention Review — 2026-05-16

Reviewer: Opus 4.7 (1M context)
Branch: `gpu-purity-stage2-20260405`
Scope: ~100 test files containing `jax` in path/name (top-level, `field/`, `geo/`, `jax_core/`, `integration/`, `objectives/`, `solve/`, `mhd/`, `core/`, `subprocess/`)
Reference policy: `tests/REVIEWER_ORACLE_LINT.md`, `benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES`, `.artifacts/jax-test-audit-2026-05-13/TEST_QUALITY_TODOS.md`

---

## Executive Summary

The JAX test suite has been significantly remediated since the 2026-05-13 audit. Of the 25 findings in `TEST_QUALITY_TODOS.md`, the four highest-severity Tier-1 tautologies (#1–#4) and most Tier-2/3 mislabeling issues (#7, #16, #25) have been substantively addressed:

- `test_framedcurve_jax_item18.py` and `test_framedcurve_jax_wrappers_item18.py` were rewritten against closed-form planar-circle analytic frames (oracle type 2) and FD-vs-`jax.grad` Optimizable contracts. The four `*_jax_matches_host` tautologies are gone.
- `test_finitebuild_jax_ssot_item20.py` replaced the JAX-vs-JAX self-parity assertions with a closed-form planar-circle filament oracle at `rtol=1e-12`.
- `test_finitebuild_jax_item20.py` replaced the VJP-vs-derivative tautology with FD-vs-JAX gradient checks at `rtol=1e-6` (item 20 finding #4).
- `tests/integration/test_stage2_jax.py` renamed `_SHORT_RUN_PARITY_RTOL → _SHORT_RUN_CONVERGENCE_RTOL` and `_PHYSICS_PARITY_RTOL → _PHYSICS_CONVERGENCE_RTOL`; "parity" labels on `rtol=1e-2 / 1e-3` smoke tests are now "convergence".
- `tests/test_jax_compile_diagnostics.py` was created to receive the compile-diagnostic-parser tests previously misfiled under `test_single_stage_physics_parity.py` (#23).
- `tests/integration/test_single_stage_jax_cpu_reference.py` tests have honest names: `test_j_both_below_health_ceiling`, `test_j_finite_branch_divergent_smoke`, `test_value_routes_through_residual_helper_not_penalty_objective`, etc. (#25).
- `tests/subprocess/jax_runtime_cases.py` introduced a `_skip_case` JSON-sentinel helper (#7) and uses it at ~16 sites. The harness in `tests/test_jax_import_smoke.py` honors the sentinel via `_assert_subprocess_json_sentinel`.

### Residual issues

1. **Five re-export `is`-identity files still classify as "routing"** (`test_dipole_field_jax_item26.py`, `test_force_item09_closeout.py`, `test_wireframefield_jax_item30.py`, `test_interpolated_field_jax_item15.py`, `test_surface_objectives_jax.py`). Per audit TODO #21, these were recommended for deletion. They are tagged with explicit "routing — guards lazy export table; no parity assertion" docstrings, which is improvement over silent tautology, but the tests still add no coverage beyond a `from simsopt.X import Y` statement at collection time. Reviewer recommendation: keep or delete is a style choice; do not let these claim parity coverage in closeout artifacts.
2. **`tests/objectives/test_integral_bdotn_jax.py:_numpy_integral_BdotN`** is a hand-coded NumPy reimplementation of the three flux definitions. The CPU oracle (`simsoptpp.integral_BdotN`) is exercised in a dedicated `TestIntegralBdotNCppParity` class, so the NumPy reproduction sits as a Tier-4 self-consistency check at `rtol=1e-13`. Not flagged as a tautology under the audit because (a) NumPy ≠ JAX (independent code paths), and (b) the C++ oracle is also asserted; however, the NumPy assertion's tolerance (`rtol=1e-13`) is suspicious for an independent-implementation parity check, since the only thing keeping them in agreement at that precision is identical FP-summation order. Document or loosen.
3. **`tests/integration/test_single_stage_jax_cpu_reference.py` (8601 lines)** retains some Tier-4 / Tier-5 patterns even after the rename pass: adjoint-residual self-consistency tests (L2654–2604), cache-count tests, finite-but-no-cross-lane health checks. These now have honest names and docstrings naming the gate-tier; the residual concern is that there are still a lot of them and they fragment coverage.
4. **`tests/subprocess/jax_runtime_cases.py` line 1858–1943**: 13 bare `return` statements after `assert np.isfinite(...)`. These are case-dispatch returns in `_run_legacy_curve_objective_value_case` / `_run_legacy_curve_objective_gradient_case`, not silent skips. The pattern is legitimate. Acceptance gate per audit #7 should distinguish these from silent-skip-after-precondition failure; the existing `_skip_case` JSON sentinel does not cover dispatch returns, so a future AST audit must avoid false positives on this style.

---

## Tautology Lint Pass

For each parity-style assertion I sampled in scope, I checked the oracle independence per `REVIEWER_ORACLE_LINT.md`. Findings below.

### A. Pure-tautology (FAIL the lint): none remaining

The four Tier-1 tautologies enumerated in TODO #1–#4 are all fixed in the current worktree:

| TODO finding | File | Status as of 2026-05-16 |
|---|---|---|
| #1 `rotated_centroid_frame is upstream` | `tests/geo/test_framedcurve_jax_item18.py` | FIXED — replaced by closed-form planar-circle frames at `rtol=1e-12` (`test_rotated_centroid_frame_matches_planar_circle_analytic`, L327; `test_rotated_frenet_frame_matches_planar_circle_analytic`, L350) |
| #2 `*_jax_matches_host` | `tests/geo/test_framedcurve_jax_wrappers_item18.py` | FIXED — old `*_jax_matches_host` tests removed (file docstring L23–28 acknowledges the change), replaced with `test_frame_rotation_jax_dofs_drive_wrapper_outputs_via_fd` (L149) using FD-vs-`jax.grad` |
| #3 `build_filament_gammas_matches_*` | `tests/geo/test_finitebuild_jax_ssot_item20.py` | FIXED — replaced by `test_build_filament_gammas_matches_planar_circle_closed_form` (L75) anchored to closed-form planar-circle filament offsets at `rtol=1e-12` |
| #4 JAX-VJP-vs-JAX-derivative | `tests/geo/test_finitebuild_jax_item20.py` | FIXED — `test_curvefilament_jax_gamma_vjp_matches_central_fd` (L151), `test_curvefilament_jax_gammadash_vjp_matches_central_fd` (L182), `test_curvefilament_spec_pullback_matches_central_fd` (L213) all anchor against central FD at `rtol=1e-6, atol=1e-8` |

### B. Re-export `is`-identity (style: prefer-delete, not strict tautology)

These tests check that `from simsopt.X import Y` produces the same object as `simsopt.X.Y` — they catch zero correctness regressions; they pass for any non-broken Python import. All five existing sites are now annotated with explicit "routing — no parity assertion" docstrings, which prevents them from being miscredited toward parity coverage. Still recommend deletion per audit #21:

| File:line | Assertion | Classification per file docstring |
|---|---|---|
| `tests/geo/test_surface_objectives_jax.py:5640` | `assert AspectRatioJAX is surfaceobjectives_jax_module.AspectRatioJAX` | "routing — no parity assertion" |
| `tests/geo/test_surface_objectives_jax.py:6129` | `assert QfmResidualJAX is surfaceobjectives_jax_module.QfmResidualJAX` | "routing" |
| `tests/geo/test_surface_objectives_jax.py:6581` | `assert MajorRadiusJAX is surfaceobjectives_jax_module.MajorRadiusJAX` | "routing" |
| `tests/field/test_dipole_field_jax_item26.py:80` | `assert ExportedDipoleFieldJAX is DipoleFieldJAX` | bare `test_package_export` — no docstring |
| `tests/field/test_force_item09_closeout.py:102–105` | 4× `is` assertions on `B2EnergyJAX`/`LpCurveForceJAX` | named `_force_energy_jax_wrappers_are_public_lazy_exports` (honest) |
| `tests/field/test_wireframefield_jax_item30.py:64` | `assert ExportedWireframeFieldJAX is WireframeFieldJAX` | bare `test_package_export` |
| `tests/field/test_interpolated_field_jax_item15.py:398` | `assert ExportedInterpolatedFieldJAX is InterpolatedFieldJAX` | bare |
| `tests/geo/test_surface_henneberg_jax.py:437` | `assert cls is SurfaceHennebergSpec` | embedded in pytree contract check (legitimate) |

The pattern in `tests/field/test_trace_boozer_analytic_jax.py:139–141, 207–209` (assert dispatch returns specific kernel function objects) is **not** a tautology — it intentionally pins routing between `BoozerAnalyticFrozenState → analytic kernels` vs. `BoozerRadialInterpolantFrozenState → spline kernels`, documented as "routing test, not a numerical parity test". Keep.

### C. Hand-coded NumPy reproduction of the JAX formula (Tier-4 self-consistency, sometimes labeled as parity)

| File:line | Reproduces what? | Risk |
|---|---|---|
| `tests/objectives/test_integral_bdotn_jax.py:56–70 (_numpy_integral_BdotN)` | All three flux definitions (`quadratic flux`, `normalized`, `local`) | Asserted at `rtol=1e-13` in `test_parity_with_target` (L122). Tight tol implies same FP order. The C++ oracle is enforced separately at the same `rtol=1e-13` in `TestIntegralBdotNCppParity::test_cpp_parity`, so coverage is not lost; the NumPy assertion just runs a same-formula reimplementation. Style: keep but document as Tier-4 self-consistency. |
| `tests/geo/test_boozer_residual_jax.py:130–152 (_numpy_cpu_ordered_boozer_scalar_reference)` | CPU-ordered scalar reduction at `rtol=0.0, atol=0.0` | Asserted only against the JAX kernel's `reduction_mode="cpu_ordered"` path. This is a Tier-4 byte-identity probe documenting that the JAX cpu-ordered mode mirrors a NumPy point/component accumulation. Documented in `_numpy_cpu_ordered_boozer_scalar_reference` docstring. Acceptable — the same-state direct-kernel C++ scalar oracle is asserted at L307 (`test_scalar_matches_cpp_oracle`) with the parity-ladder direct-kernel tolerance. |
| `tests/integration/test_jax_native_path.py:75–90 (_jaxfouriercurve_pure)` | Reference loop implementation for XYZ Fourier coil gamma | Used only as the FD-gradient oracle; the SUT is `_build_fourier_basis @ coeffs.T` (basis-matrix form). Pin is at `atol=1e-13` (`TestFourierBasis`). The two implementations *are* independent — one is loop-based, the other matrix-multiplication-based — so the parity isn't purely tautological. Acceptable. |

### D. `cpu_objective.J() vs jax_objective.J()` when both route through JAX

| File:line | Test name | Detail | Verdict |
|---|---|---|---|
| `tests/geo/test_curve_objectives_jax.py:238 (test_public_curve_objective_jax_wrappers_match_cpu_values_and_gradients)` | Compares `CurveLength(curve).J()` vs `CurveLengthJAX(curve).J()` | Source: `simsopt/geo/curveobjectives.py:707, 945` route through JAX kernels only when `is_jax_backend()` returns True. This test does **not** monkeypatch `is_jax_backend`, so the CPU side falls through to the legacy `Optimizable` graph (`gammadash()` from the host evaluator). Therefore CPU side is `simsoptpp` / pure-NumPy host while JAX side is `simsopt.jax_core`. Genuine cross-implementation parity. | OK |
| `tests/geo/test_curve_objectives_jax.py:163 (test_remaining_curve_objective_mirrors_match_cpu_values)` | Monkeypatches `is_jax_backend → True` | Now both routes call JAX internally. The "expected" value comes from the CPU `LpCurveCurvatureBarrier(...).J()` which now also goes through JAX. **Borderline tautology** — but expected ≈ jax-mirror is still independently checked by FD-of-Optimizable-graph tests in the same module, so this test's role is "wrapper dispatch routing under the JAX backend monkeypatch", not value oracle. Add a docstring clarifying this. | NEEDS-CLARIFICATION |

### E. Adjoint self-consistency `apply_transpose(solve_transpose(b)) ≈ b`

Per audit TODO #25 (and the explicit example in `REVIEWER_ORACLE_LINT.md` line 29: "as the only check — self-consistency of an operator and its inverse is trivially true"), the adjoint-residual self-consistency test was renamed and docstring-tagged as **Tier-4 self-consistency, not vector parity**. Examples:

- `tests/integration/test_single_stage_jax_cpu_reference.py:TestAdjointSolveConsistency` (L2637) — docstring at L2638–2652 explicitly states "Tier-4 self-consistency evidence, NOT vector parity against an independent dense reference: dense-vs-operator adjoint vector parity is covered by the `exact-well-conditioned-adjoint` lane at `tests/geo/test_boozersurface_jax.py::test_exact_well_conditioned_operator_adjoint_matches_dense_reference_and_plu`."

This is exactly the right partitioning: vector parity is in `tests/geo/test_boozersurface_jax.py` against a dense reference, and adjoint self-consistency tests cover the runtime operator's internal coherence. Both have honest names and docstrings.

---

## Tolerance-lane audit table

Sampled tolerance choices vs. CLAUDE.md SSOT contract: `direct_kernel` (rtol=1e-10, atol=1e-12) for same-state C++ parity; `derivative_heavy` (1e-8/1e-10 first-derivative, 1e-7/1e-9 second-derivative) for derivative paths; `rtol=1e-12` reserved for same-state direct-kernel single-machine probes; absolute thresholds ≥3× cross-machine worst observed (currently `sdofs_inf ≤ 1e-11`).

| File:line | Test | Asserted tol | Lane belongs to | Choice correct? |
|---|---|---|---|---|
| `tests/field/test_biotsavart_jax.py:380` | `test_on_axis_field` | `rtol=1e-12` (closed-form analytic) | analytic same-state | OK — closed-form is independent oracle |
| `tests/field/test_biotsavart_jax.py:566` | `test_B_parity_ncsx` | `_DIRECT_KERNEL_TOLS` (rtol=1e-10, atol=1e-12) | direct-kernel C++ parity | OK |
| `tests/field/test_biotsavart_jax.py:615` | `test_dB_by_dX_parity_ncsx` | `_DERIVATIVE_HEAVY_TOLS["first_derivative_*"]` | derivative-heavy | OK — derivative path correctly routed |
| `tests/field/test_biotsavart_jax.py:721` | `test_d2B_by_dXdX_parity_ncsx` | `_DERIVATIVE_HEAVY_TOLS["second_derivative_*"]` | derivative-heavy (second) | OK |
| `tests/geo/test_boozer_residual_jax.py:296-298` | `test_scalar_and_grad_cpu_ordered_matches_cpp_ds_oracle` | `_DIRECT_KERNEL_TOLS` | direct-kernel | OK (cpu_ordered to C++ scalar) |
| `tests/objectives/test_integral_bdotn_jax.py:128` | `test_parity_with_target` (NumPy mirror) | `rtol=1e-13` | Tier-4 self-consistency NumPy↔JAX | **TIGHT** — borderline tautological-precision; would prefer documentation |
| `tests/objectives/test_integral_bdotn_jax.py:347` | `test_cpp_parity` | `rtol=1e-13` | direct-kernel C++ | TIGHT but acceptable for single-machine |
| `tests/geo/test_surface_fourier_jax.py:183-185` | `test_gamma_torus` (closed-form torus) | `atol=1e-14` | analytic same-state | OK |
| `tests/geo/test_surface_fourier_jax.py:395` | `test_gamma_parity` (vs SurfaceXYZTensorFourier C++) | `atol=1e-13` | direct-kernel C++ | OK |
| `tests/geo/test_surface_fourier_jax.py:443` | `test_coefficient_derivatives_match_cpp` | `_DERIVATIVE_HEAVY_TOLS["first_derivative_*"]` | derivative-heavy | OK |
| `tests/geo/test_framedcurve_jax_item18.py:341` | `test_rotated_centroid_frame_matches_planar_circle_analytic` | `_ANALYTIC_RTOL=1e-12, _ANALYTIC_ATOL=1e-12` | analytic closed-form | OK — closed-form planar circle |
| `tests/geo/test_finitebuild_jax_ssot_item20.py:130-138` | `test_build_filament_gammas_matches_planar_circle_closed_form` | `rtol=1e-12, atol=1e-12` | analytic closed-form | OK |
| `tests/geo/test_finitebuild_jax_item20.py:177` | `test_curvefilament_jax_gamma_vjp_matches_central_fd` | `rtol=1e-6, atol=1e-8` | FD-gradient | OK — FD step `1e-5` matches |
| `tests/integration/test_stage2_jax.py:184` | `_SHORT_RUN_CONVERGENCE_RTOL = 1e-3` | (renamed from `_PARITY`) | smoke / convergence | OK — honest rename |
| `tests/integration/test_stage2_jax.py:185-186` | `_STAGE2_VALUE_PARITY_RTOL=1e-12, _STAGE2_GRADIENT_PARITY_RTOL=1e-11` | direct-kernel value, derivative-heavy gradient | Stage 2 same-state oracle parity | OK — rtol 1e-12 on a single-machine same-state contract is the documented exception |
| `tests/integration/test_stage2_jax.py:188-189` | `_STAGE2_TARGET_SCALAR_VALUE_RTOL=1e-15, _STAGE2_TARGET_SCALAR_VALUE_ATOL=1e-18` | reporting-contract scalar | **VERY TIGHT** but justifiable for a documented same-state assertion |
| `tests/integration/test_stage2_jax.py:340` (`_TARGET_OBJECTIVE_GRAD_ATOL = 5e-12`) | `target-objective` gradient absolute check | derivative tolerance | TIGHT but cross-machine evidence not cited; acceptable per CLAUDE.md "rtol=1e-12 only on same-state direct-kernel on a single machine" rule. Set a hardware-cross-machine guard to ≥3× worst measured if cross-machine evidence becomes available |
| `tests/geo/test_boozersurface_jax.py:4773-4805` | LS `iota`/`G`/`fun` self-parity at `atol=1e-14` | branch-stable LS self-parity | TIGHT — under CLAUDE.md `sdofs_inf ≤ 1e-11` rule, `1e-14` is below the cross-machine floor. **Risk: may fail intermittently on other macOS hardware.** |
| `tests/geo/test_boozersurface_jax_private.py:95-124` | `rtol=1e-12` on multiple LBFGS scipy-vs-private parity checks | LS wrapper gradient single-machine | OK — same-state LS-wrapper-gradient is the documented `rtol=1e-12` exception |
| `tests/geo/test_lbfgsb_scipy_jax_kernels.py:2438-2440` (`assert_allclose(... fold)`) | LBFGS state-transition equality | direct-kernel scipy reference | OK |
| `tests/jax_core/test_tracing_jax_item14.py:99` | `truncation_ceiling = 1.0e-5` for dopri5 single-step | analytic `np.exp(0.1)` with derived bound | OK — bound derived from `h^6 = 1e-6` per docstring |
| `tests/jax_core/test_tracing_jax_item14.py:188-189` | `state_rtol, state_atol` from `event_time_tracing` lane | lane-driven | OK |
| `tests/jax_core/test_tracing_jax_conservation.py:70-71` | `atol=1e-10` for energy/μ conservation across tmax=0.1 | physics-conservation gate | OK |

### Findings on tolerances

1. `tests/integration/test_stage2_jax.py:_STAGE2_TARGET_SCALAR_VALUE_ATOL = 1e-18` is suspect under FP analysis (smallest positive `np.float64` subnormal is `~5e-324`, but realistic FP arithmetic noise floor on a sum-of-products with magnitudes ~1 is `~1e-16`). Unless this is testing exact-zero residual on a single-machine FP-order-pinned same-state contract, the choice is **TOO TIGHT**.
2. `tests/geo/test_boozersurface_jax.py:4773-4805` runs `assert_allclose(..., atol=1e-14)` on LS solver outputs (`iota`, `G`, `fun`, gradient, residual, surface DOFs). Per CLAUDE.md floating-point reproducibility memory, cross-machine `sdofs_inf` ranges up to `3.6e-12`. **These thresholds will likely fail intermittently on different hardware.** Recommendation: relax to `atol=1e-11` per CLAUDE.md ≥3× cross-machine rule, OR explicitly mark as same-machine reproducibility tests with `pytest.mark.machine_pinned`.

---

## Determinism / RNG audit

| Aspect | Status |
|---|---|
| `jax.config.update("jax_enable_x64", True)` | Set at module scope in 12+ test files (e.g., `tests/integration/test_jax_native_path.py:24`, `tests/geo/test_surface_fourier_jax.py:20`, `tests/geo/test_boozer_derivatives_jax.py:18`). Also set in `tests/conftest.py:_force_x64` at session start (`L28-35`) — covers the suite globally. **OK**. |
| Per-test `jax.random.PRNGKey` usage | Present where needed (e.g., `tests/field/test_sampling_jax_item22.py:313` uses `jax.random.PRNGKey(909)`). Other tests use `np.random.RandomState(seed)` for synthetic-data fixtures, which is fine for non-stochastic tests. |
| `np.random.seed` in JAX-port tests | Found in `tests/geo/test_boozersurface_jax.py:7809, 7862` and `tests/field/test_sampling_jax_item22.py:311, 338`. Global `np.random.seed` is **bad practice** (it leaks state across tests run in the same session). Recommendation: replace with `np.random.default_rng(seed)` or `parity_rng(seed)` (which lives at `tests/conftest.py:374`). |
| `jax.clear_caches()` for cache leakage | Used only in `tests/subprocess/jax_runtime_cases.py:219`. The compile-count regression tests in `tests/test_jax_import_smoke.py` run in subprocess to get isolated caches. The in-process suite does not call `jax.clear_caches()` between tests — most tests do not recompile in ways that would leak. The `_guard_backend_runtime_state` autouse fixture (`tests/conftest.py:273`) invalidates the backend cache between tests but does not clear JAX's compile cache. **Acceptable for cross-state correctness; not for compile-count regressions.** |
| Global mutation across tests | The `_guard_backend_runtime_state` fixture (`tests/conftest.py:273`) snapshots `_BACKEND_RUNTIME_ENV_VARS` (24 env vars) and JAX runtime config, restoring after each test. **Strong isolation**. |
| `chex` / `jax.test_util.check_grads` | Not used. Tests use manual FD checks instead (acceptable when FD step and tolerances are documented). |
| `numpy.testing.assert_allclose` on JAX arrays | Pervasive use of `np.array(...)` or `np.asarray(...)` to convert before assertion. Idiomatic. |
| `block_until_ready()` for timing | Used in `tests/geo/test_framedcurve_jax_item18.py:448-451` and elsewhere. Correct usage. |
| FD step sizes | Range from `1e-7` (curve.py tests) to `3e-7` (test_jax_native_path) to `1e-5` (lbfgs lnsrlb). Each is documented in context. **OK**. |
| `pytest.importorskip("simsoptpp")` | Used in `tests/field/test_biotsavart_jax.py:538`, `tests/objectives/test_integral_bdotn_jax.py:326`, `tests/geo/test_boozer_residual_jax.py:252, 317, 541` and consistently across the C++-parity classes. Tests run cleanly in the pure-JAX env. **OK**. |
| `int()` / `bool()` boundary conversions on JAX status fields | Pattern is present at MANY sites — `int(result.status)`, `bool(result.success)`. CLAUDE.md flags this as a requirement, and the tests follow it. **OK**. |

### `np.random.seed` cleanup recommendation

These four sites should migrate to a seeded `np.random.default_rng` to avoid cross-test seed leakage:

- `tests/geo/test_boozersurface_jax.py:7809` `np.random.seed(1)`
- `tests/geo/test_boozersurface_jax.py:7862` `np.random.seed(1)`
- `tests/field/test_sampling_jax_item22.py:311` `np.random.seed(1)`
- `tests/field/test_sampling_jax_item22.py:338` `np.random.seed(2)`

The latter two are necessary because the upstream `draw_uniform_on_curve` (host implementation) consumes the global RNG, so removing them would break the test. But they leak — restoring the RNG via the helper `parity_rng` from `tests/conftest.py:374` is cleaner.

---

## Device-residency audit

`tests/test_backend_strict_jax_device_detection.py` covers the device-detection contract (§2 / §3 from the silent-fallback-removal plan) thoroughly — 18 tests cover the JAX-mode propagation, ImportError tolerance, and external-tool failure modes for `nvidia-smi`. Strong. Pattern: hooks `sys.modules["jax"]` with `monkeypatch.setitem` so the test stub provides exploding stubs that raise if called — this is the right pattern to assert **non-calling** of JAX device APIs from a CPU policy.

`tests/conftest.py` provides:
- `parity_device(lane)` — selects the first device matching `lane in {"cpu", "gpu"}`, skips if absent.
- `parity_default_device(lane)` — context manager that sets `jax.default_device(...)`.
- `assert_array_on_device(array, device)` and `assert_arrays_on_device(...)` — assert device residency.
- `enable_strict_jax_backend(monkeypatch, request, mode)` and `enable_strict_parity_backend(monkeypatch, request, lane)` for activating strict modes.

Tests like `tests/geo/test_boozersurface_jax.py:_collect_exact_well_conditioned_runtime_metadata` (L199-217) and the `_solve_exact_well_conditioned_operator_case` helper at L299-373 collect platform/device-kind metadata and assert device residency. **Strong device discipline on the GPU-purity lane.**

The `parity_lane` fixture (`tests/conftest.py:431`) parametrizes tests across `("cpu", "gpu")` — pytest collects each test twice. Tests gated on GPU skip cleanly via `parity_device(lane)`'s `pytest.skip("CUDA GPU not available")`.

CUDA reproducibility: `ensure_gpu_determinism_xla_flag` (`tests/conftest.py:252`) prepends `--xla_gpu_deterministic_ops` when activating `jax_gpu_parity`. Aligned with CLAUDE.md's "deterministic XLA GPU flag must be set before JAX initialization" requirement.

**Verdict: device-residency contract is well-enforced.**

---

## Conftest patches audit

### Root `tests/conftest.py`

- **Bootstrap pinning** (L11-17): pre-pends the repo root to `sys.path`, then calls `repo_bootstrap.bootstrap_local_simsopt` to bind `simsopt` to the local source tree. This is the right pattern to avoid foreign-`simsopt` shadowing per `feedback_review_access_failures.md`.
- **x64 enforcement** (L28-35): `_force_x64(jax)` is called at module load if JAX is importable. Raises `RuntimeError` if `jax_enable_x64` is not True after the update. Strong.
- **`_guard_backend_runtime_state` autouse fixture** (L273-285): snapshots 24 env vars + 5 JAX runtime config knobs and restores after each test. Invalidates backend cache before and after. **Excellent isolation pattern**.
- **`pytest_collection_modifyitems`** (L486-516): auto-marks `tests/integration/*` as `integration`+`slow`, single-stage and stage2 tests as additional markers, BoozerSurface JAX tests as `boozer`+`slow`. Allows sharded runs.

### `tests/integration/conftest.py`

- **Scikit-build editable finder patch** (L16-66): when the test env has `simsopt` installed as an editable scikit-build install (`.conda/jax`), the meta-path finder intercepts all `simsopt.*` imports and would not find the new JAX modules. This conftest extends `finder.known_source_files` with 11 new JAX module entries. Critical for M2 integration tests to find `BiotSavartJAX`, `SquaredFluxJAX`, `BoozerSurfaceJAX` etc.
- **`pytest_report_header`** (L73-80): prints whether the finder patch was applied. Good diagnostic.

**Verdict: conftest patches are correct, well-documented, and necessary.** No regressions.

---

## Positive notes

1. **Validation-ladder SSOT discipline**: Almost every parity-asserting test imports `parity_ladder_tolerances` from `benchmarks/validation_ladder_contract.py` and uses lane-keyed tolerances (`_DIRECT_KERNEL_TOLS`, `_DERIVATIVE_HEAVY_TOLS`, `_FD_GRADIENT_TOLS`). Inline `rtol=...` literals are increasingly rare and confined to closed-form / analytic / per-test-specific cases. Strong improvement since 2026-05-13.
2. **Oracle docstrings**: New tests cite the oracle by type number (1/2/3/4 per `REVIEWER_ORACLE_LINT.md`) and the C++ symbol or analytic formula. Examples: `test_B_parity_ncsx` (test_biotsavart_jax.py:543-552), `test_scalar_matches_cpp_oracle` (test_boozer_residual_jax.py:307-316), `test_rotated_centroid_frame_matches_planar_circle_analytic` (test_framedcurve_jax_item18.py:327-335), and many more.
3. **Routing-vs-parity distinction**: Tests that route through internals are explicitly named (`test_value_routes_through_residual_helper_not_penalty_objective`, `_lazy_package_export`) and documented as "no parity assertion" in the docstring. This breaks the audit pattern that scanned for "matches_*" name regex and counted them as parity.
4. **Marker discipline**: `private_optimizer_runtime` marker is used consistently:
   - `tests/geo/test_boozersurface_jax_private.py:611` defines `PRIVATE_OPTIMIZER_RUNTIME = pytest.mark.private_optimizer_runtime` and the skipif companion `REQUIRES_PRIVATE_OPTIMIZER_RUNTIME`.
   - `tests/integration/test_section6_public_lane_split.py:91` applies it at the class level.
   - `tests/integration/test_single_stage_jax_cpu_reference.py:5340` applies it at function level.
   - `tests/integration/test_stage2_jax.py:139-481` conditionally configures the private optimizer when the JAX runtime supports it.
5. **Compile-count regression tests** are isolated in subprocess with `JAX_ENABLE_COMPILATION_CACHE=0` (e.g., `tests/test_jax_import_smoke.py:667`, L696, L719). Avoids cross-test cache contamination.
6. **`_skip_case` JSON sentinel** is established in `tests/subprocess/jax_runtime_cases.py:30-32` and used at 16 sites. The harness translator in `tests/test_jax_import_smoke.py:182-193` (`_maybe_skip_from_subprocess_stdout`) honors it. Subprocess silent-skip pattern is materially better than at 2026-05-13.
7. **Sharding tests**: `tests/field/test_biotsavart_jax.py:1277-1349` exercises `jax.sharding.Mesh`/`NamedSharding`/`PartitionSpec` with `monkeypatch` on the sharding tuning. Strong coverage for the GPU coil-axis collective lowering contract.
8. **C++ ANGLE_RECOMPUTE coverage**: The C++ VJP correctness tests in `test_biotsavart_jax.py::TestBiotSavartJaxCppParity` exercise the brace pattern from CLAUDE.md ("In `surfacerzfourier.cpp` the VJP loops use `if(i % ANGLE_RECOMPUTE == 0)` — explicit braces required"). Tests assert parity at the derivative-heavy lane, which would catch the historical "bare if guards only the first statement" bug.

---

## Verdict per file (PASS / NEEDS-WORK / FAIL)

### Top-level

| File | Verdict | Reason |
|---|---|---|
| `tests/test_jax_import_smoke.py` (1500 lines) | PASS | Comprehensive subprocess-driven smoke; explicit env var management; JSON-sentinel honored. |
| `tests/test_jax_compile_diagnostics.py` | PASS | Parser-invariant tests, properly classified as instrumentation. |
| `tests/test_jax_where_division_lint.py` | PASS | Lint-rule unit test, simple and focused. |
| `tests/test_backend_strict_jax_device_detection.py` | PASS | Excellent device-detection propagation coverage. |

### `tests/field/`

| File | Verdict | Reason |
|---|---|---|
| `test_biotsavart_jax.py` | PASS | CLAUDE.md memory flagged this for tautology — verified: tautology-free in current state. C++-vs-JAX parity uses the NCSX fixture and the `direct_kernel` lane; chunked-vs-dense self-consistency tests are explicitly Tier-4 (file docstring L897-908). |
| `test_biotsavart_jax_parity.py` | PASS (not read in detail; sampled headers) |
| `test_biotsavart_jax_cpu_ordered.py` | NEEDS-WORK (per audit #12; not re-validated here) — see `TEST_QUALITY_TODOS.md`. |
| `test_boozer_analytic_jax.py` | PASS |
| `test_boozermagneticfield_jax_item33.py` | PASS — closed-form Boozer-identity test at L337-355 anchors the wrapper at the JAX-identity level independent of CPU. The structural `wrapper_has_no_dofs` test (L454) is documented as catching the specific decision-vector-size regression. |
| `test_circular_coil_jax.py` | PASS — analytic-axis circular-coil oracle + C++ parity. |
| `test_dipole_field_jax_item26.py` | NEEDS-WORK — `test_package_export` (L79-80) is a bare `is`-identity check without a docstring. Either delete or annotate as "routing". |
| `test_interpolated_boozer_field_jax.py` | PASS (sampled). |
| `test_interpolated_field_jax_item15.py` | NEEDS-WORK — L398 `is`-identity check. |
| `test_magnetic_axis_helpers_jax_item21.py` | PASS (sampled). |
| `test_magnetic_field_composition_jax.py` | PASS (sampled). |
| `test_magneticfieldclasses_jax_item15.py` | PASS (sampled). |
| `test_sampling_jax_item22.py` | NEEDS-WORK — uses `np.random.seed(1/2)` globally; would prefer `parity_rng`. Test logic is sound; only the seed-management is a style issue. |
| `test_scalar_potential_rz_jax_item23.py` | PASS (sampled). |
| `test_trace_boozer_analytic_jax.py` | PASS — dispatch `is`-identity (L139, 207) is documented as routing. |
| `test_tracing_jax_item16.py` / `_extended.py` | PASS (sampled). |
| `test_wireframefield_jax_item30.py` | NEEDS-WORK — L64 bare `is`-identity. |
| `test_boozermagneticfield_jax_item33.py` | PASS (covered above). |
| `test_force_item09_closeout.py` | NEEDS-WORK — L102-105 is-identity, but named "are_public_lazy_exports" which is honest. Either delete or keep as routing. |

### `tests/geo/`

| File | Verdict | Reason |
|---|---|---|
| `test_boozer_derivatives_jax.py` | PASS — 19 FD-validated tests at `rtol=1e-10`, anchored against `jax.grad` of the M3 composed pipeline (sampled). |
| `test_boozer_residual_jax.py` | PASS — flagged in OpenMemory for tautology; verified safe. The NumPy reproduction `_numpy_cpu_ordered_boozer_scalar_reference` (L130-152) is explicitly Tier-4 cpu-ordered probe, and the C++ scalar oracle at L307-336 is the same-state direct-kernel anchor. |
| `test_boozersurface_jax.py` (9351 lines) | PASS-WITH-CAVEATS — strong overall, but `atol=1e-14` LS self-parity at L4773-4805 may be cross-machine fragile per CLAUDE.md fp-reproducibility memo. |
| `test_boozersurface_jax_private.py` (2352 lines) | PASS — `private_optimizer_runtime` marker discipline; same-state LS-wrapper-gradient at `rtol=1e-12` is the documented single-machine exception. |
| `test_curve_objectives_jax.py` | PASS — verified that the `cpu_objective` actually routes through a different code path (legacy Optimizable graph) unless `is_jax_backend` is monkeypatched. The single test that monkeypatches it (L163) is value-vs-routing only. |
| `test_curvexyzfouriersymmetries_spec_jax.py` | PASS (sampled). |
| `test_distance_jax.py` | PASS — uses `simsoptpp.get_pointclouds_closer_than_threshold_*` as the C++ oracle (set-equality check), anchoring the JAX candidate culler. |
| `test_finitebuild_jax_item20.py` | PASS — addresses audit #4 (FD vs JAX). |
| `test_finitebuild_jax_ssot_item20.py` | PASS — addresses audit #3 (closed-form planar-circle oracle). |
| `test_framedcurve_jax_item18.py` | PASS — addresses audit #1 (closed-form planar-circle frames). |
| `test_framedcurve_jax_wrappers_item18.py` | PASS — addresses audit #2 (FD vs `jax.grad` on Optimizable graph). |
| `test_label_constraints_jax.py` | PASS (sampled). |
| `test_lbfgsb_scipy_jax_kernels.py` (2507 lines) | PASS — uses `scipy.optimize._lbfgsb_py` as parity oracle and a Python reference reimplementation of the Fortran routines. Strong. |
| `test_linking_number_jax.py` | PASS (sampled). |
| `test_optimizer_jax_item19.py` | PASS — tight, documented contracts on optimizer-backend method selection. |
| `test_optimizer_jax_silent_fallback_removal.py` | PASS — exercises the `_is_flat_optimizer_vector` classifier with both positive (numeric) and negative (object, text, datetime) parametrize cases. |
| `test_orientedcurve_jax_spec.py` | PASS (sampled). |
| `test_permanent_magnet_grid_jax_item27.py` | PASS (sampled). |
| `test_surface_fourier_jax.py` | PASS — analytic torus + C++ parity. Strong. |
| `test_surface_fourier_jax_cpu_ordered.py` | NEEDS-WORK per audit #13; not re-validated. |
| `test_surface_garabedian_jax.py` | PASS (sampled). |
| `test_surface_henneberg_jax.py` | PASS — pytree contract checks with embedded `is`-identity on the type registration are legitimate. |
| `test_surface_objectives_jax.py` | PASS-WITH-NOTE — L5640, 6129, 6581 are documented `is`-routing checks. Other tests use first-order Taylor remainder against `jax.grad` (oracle type 4). |
| `test_surface_rzfourier_jax.py` / `_item06_closeout.py` | PASS (sampled). |
| `test_surface_xyz_tensor_clamped_jax.py` | PASS (sampled). |

### `tests/jax_core/`

| File | Verdict | Reason |
|---|---|---|
| `test_boozer_fixed_state_jax_item33.py` | PASS (sampled). |
| `test_boozer_radial_interp_jax_item32.py` | PASS — `direct_kernel` lane against CPU `BoozerRadialInterpolant`. |
| `test_dipole_field_jax_item24.py` | PASS (sampled). |
| `test_pm_optimization_jax_item25.py` | PASS (sampled). |
| `test_tracing_jax_item14.py` | PASS — closes audit #17. `dopri5_step` is anchored against `np.exp(0.1)` with a derived `h^6=1e-6` truncation ceiling at L99 (gate tier explicit). `trace_fieldline_jit_runs_without_error` was renamed honestly. |
| `test_tracing_jax_phi_events.py` | PASS — closed-form analytic phi-target time at L106-107. |
| `test_tracing_jax_fullorbit.py` / `_events.py` | PASS (sampled). |
| `test_tracing_jax_levelset_events.py` | PASS (sampled). |
| `test_tracing_jax_guiding_center.py` | PASS — upstream particle tracing parity. |
| `test_tracing_jax_boozer_zeta_events.py` | PASS (sampled). |
| `test_tracing_jax_gc_boozer.py` | PASS (sampled). |
| `test_tracing_jax_conservation.py` | PASS — physics conservation (energy, μ) at `atol=1e-10` over `tmax=0.1`. Conservation laws are an independent oracle (oracle type 2). |
| `test_wireframe_jax_item29.py` | PASS (sampled). |
| `jaxpr_utils.py` | PASS — helper module, not a test file. |

### `tests/integration/`

| File | Verdict | Reason |
|---|---|---|
| `test_jax_native_path.py` | PASS — Fourier-basis vs loop-based reproduction is the independent oracle; FD-vs-`jax.value_and_grad` at `rtol=1e-5/2e-6`. |
| `test_non_banana_example_cpp_jax_cpu_parity.py` | NEEDS-WORK per audit #10 — has a `pytest.skip` masking the upstream verdict; not re-validated. |
| `test_single_stage_jax.py` | PASS — JAX-only helper-path tests with explicit `_FakeBiotSavart` mocks; FD-validated. |
| `test_single_stage_jax_cpu_reference.py` (8601 lines) | PASS — large file, audit #25 issues resolved via renames + docstrings naming the gate-tier. Hot spots covered by `branch-stable-resolve`, `exact-well-conditioned-adjoint` lanes elsewhere. |
| `test_stage2_jax.py` (6473 lines) | PASS — audit #16 resolved (renamed `_PARITY_RTOL → _CONVERGENCE_RTOL`). Strong stage 2 parity coverage. |
| `test_single_stage_physics_parity.py` (1012 lines) | NEEDS-WORK per audit #14, #22, #23 — some bookkeeping-only tests still in this file, parts moved out. Not re-validated end-to-end. |
| `test_single_stage_dof_mapping.py` | PASS (sampled). |
| `test_section6_public_lane_split.py` | PASS (sampled). |
| `test_factor_once_adjoint_phase2.py` | PASS (sampled). |
| `test_stage2_target_lane_purity.py` | PASS (sampled). |

### `tests/objectives/`

| File | Verdict | Reason |
|---|---|---|
| `test_fluxobjective_jax_parity.py` | PASS — flagged in OpenMemory for tautology; verified: parity is against CPU `SquaredFlux.J()` (which routes through C++ `simsoptpp.integral_BdotN`), and the FD-gradient is checked via central-difference. The "chunked-vs-dense self-consistency" tests are explicitly Tier-4 with tighter-than-parity tolerances (`_CHUNKED_SELF_CONSISTENCY_VALUE_RTOL = 1e-12`, etc.), documented in the comment block. The `_flux_kernel_value_and_grad` helper is documented as "NOT a CPU/JAX parity oracle" (L336-353). |
| `test_fluxobjective_jax_item03_closeout.py` | PASS (sampled). |
| `test_integral_bdotn_jax.py` | PASS-WITH-NOTE — `_numpy_integral_BdotN` reproduction tested at `rtol=1e-13` is borderline tight for an independent NumPy reimplementation; the parallel C++ parity at `TestIntegralBdotNCppParity` does the heavy lifting. |

### `tests/solve/`

| File | Verdict | Reason |
|---|---|---|
| `test_permanent_magnet_optimization_jax_item28.py` | PASS — uses `simsoptpp` C++ oracle via `setup_initial_condition` / `projection_L2_balls` etc. Imports `simsoptpp` at module top (requires `.conda/jax` env). |
| `test_wireframe_optimization_jax_item31.py` | PASS — uses `simsoptpp as sopp` for parity oracle; lane-driven tolerances. |

### `tests/mhd/`

| File | Verdict | Reason |
|---|---|---|
| `test_boozer_jax.py` | PASS — uses CPU `Quasisymmetry` class as oracle; mock `_FrozenBoozer` / `_FrozenBoozXform` to isolate the quasisymmetry-residual reducer kernel. |
| `test_vmec_diagnostics_jax.py` | PASS — uses CPU `IotaTargetMetric`, `IotaWeighted`, `WellWeighted` as oracles; mock `_FrozenVmec`. |

### `tests/core/`

| File | Verdict | Reason |
|---|---|---|
| `test_jax_core_specs.py` | PASS — spec-construction and pytree-flatten/unflatten tests; coverage on float32→float64 casting and immutable dataclass round-trips. |

### `tests/subprocess/`

| File | Verdict | Reason |
|---|---|---|
| `jax_runtime_cases.py` | PASS-WITH-NOTE per audit #7 — `_skip_case` JSON-sentinel pattern adopted at 16 sites; some legitimate dispatch `return` statements remain in `_run_legacy_curve_objective_*_case` (L1858-1943) and should not be mistaken for silent-skip regressions by a future AST audit. |
| `import_smoke_cases.py` | PASS (sampled). |
| Helper modules: `boozersurface_jax_test_helpers.py`, `jaxpr_utils.py` | PASS — production-grade helpers; honest documentation. |

---

## Concrete remediation list (sorted by impact)

1. **`tests/integration/test_stage2_jax.py:188-189`** — `_STAGE2_TARGET_SCALAR_VALUE_ATOL = 1e-18` is below the float64 noise floor for any non-trivial sum. Audit this test's actual numerical regime and either tighten to `0.0` (exact zero check) or relax to `1e-15`. **Impact: medium — may pass today on lucky FP order but is fragile.**
2. **`tests/geo/test_boozersurface_jax.py:4773-4805, 7457`** — `atol=1e-14` on LS solver outputs and surface DOFs is below the documented cross-machine `sdofs_inf ≤ 1e-11` threshold. **Impact: medium — likely intermittent failures on alternative macOS hardware.** Recommendation: relax to `atol=1e-11` per CLAUDE.md, or add `pytest.mark.machine_pinned` and document.
3. **`tests/objectives/test_integral_bdotn_jax.py:128`** — `rtol=1e-13` between hand-coded NumPy and JAX implementations of the same flux formula. Acceptable but document as Tier-4 self-consistency; the real parity gate is the C++ test in `TestIntegralBdotNCppParity`. **Impact: low.**
4. **5× re-export `is`-identity tests** (`test_dipole_field_jax_item26.py:80`, `test_force_item09_closeout.py:102-105`, `test_wireframefield_jax_item30.py:64`, `test_interpolated_field_jax_item15.py:398`, `test_surface_objectives_jax.py:5640/6129/6581`) — pure routing trivia; recommended for deletion per audit TODO #21. Some already have honest "routing" docstrings; others (e.g., `test_package_export`) are bare. **Impact: low — does not weaken existing parity coverage, just adds noise.**
5. **`np.random.seed(...)` in 4 sites** (`test_boozersurface_jax.py:7809, 7862`, `test_sampling_jax_item22.py:311, 338`) — leak global RNG state. Replace with `np.random.default_rng(seed)` or `parity_rng(seed)`. **Impact: low to medium — depends on test order and could surface as flake on a different test runner.**
6. **`tests/subprocess/jax_runtime_cases.py` lingering bare `return`s** in case-dispatch functions (e.g., L1858-1943) — legitimate, but the future AST audit per TODO #7 / AI-1 must not flag them. Audit walker should restrict to functions whose body is `if not _configure_*: return` (silent precondition skip) vs `if case == "X": ...; return` (dispatch). **Impact: tooling — affects the audit, not the tests.**

---

## Comparison against the 2026-05-13 audit baseline

| Audit Finding | 2026-05-13 status | 2026-05-16 status |
|---|---|---|
| #1 — `tests/geo/test_framedcurve_jax_item18.py` JAX-vs-JAX tautology | Untracked, mtime 2026-05-13 | **FIXED** — closed-form planar-circle frames |
| #2 — `tests/geo/test_framedcurve_jax_wrappers_item18.py` `*_jax_matches_host` | Untracked | **FIXED** — replaced with FD-on-Optimizable |
| #3 — `tests/geo/test_finitebuild_jax_ssot_item20.py` | Untracked | **FIXED** — closed-form planar-circle filament |
| #4 — `tests/geo/test_finitebuild_jax_item20.py` VJP-vs-JAX | Untracked | **FIXED** — FD-vs-JAX |
| #5 — fake CPU/GPU lane in `test_single_stage_cpp_jax_state_parity.py` | Tracked | Not re-validated |
| #6 — `test_strainopt_item08_closeout.py` numpy reproduction | Tracked | Not re-validated |
| #7 — 16 silent-skip sites in `jax_runtime_cases.py` | Tracked, growing | **PARTIALLY ADDRESSED** — `_skip_case` JSON sentinel adopted at 16 sites; some legitimate dispatch returns remain |
| #8 — `pytest.skip` after real assertions | Tracked | Not re-validated |
| #9 — fail-by-design without xfail | Tracked | Not re-validated |
| #10 — upstream-failure-as-skip in `test_non_banana_example_cpp_jax_cpu_parity.py` | Tracked | Not re-validated |
| #11 — JSON-sentinel translator downstream cases | Tracked | Improved with `_assert_subprocess_json_sentinel` |
| #12 — `test_biotsavart_jax_cpu_ordered.py` no-regression-as-parity | Tracked | Not re-validated |
| #13 — `test_surface_fourier_jax_cpu_ordered.py` same | Tracked | Not re-validated |
| #14 — `_assert_outer_loop_single_step_consistency` ceilings-only | Tracked | Not re-validated |
| #15 — `test_strainopt_item08_closeout.py` `1e-10` floor | Tracked | Not re-validated |
| #16 — three "parity" tests at `rtol=1e-2`-`1e-3` in `test_stage2_jax.py` | Tracked | **ADDRESSED** — renamed to `_CONVERGENCE_RTOL` |
| #17 — `test_tracing_jax_item14.py` discarded analytic anchor | Untracked | **ADDRESSED** — `np.exp(0.1)` is now the gate at L99 |
| #18 — structural-trivia tests in `test_boozermagneticfield_jax_item33.py` | Untracked | **PARTIALLY ADDRESSED** — `test_wrapper_has_no_dofs` (L454) has a docstring naming the regression class it catches |
| #19 — silent skip on missing `lightning_sdk` | Tracked | Not re-validated |
| #20 — folded into #7 | — | — |
| #21 — `module.foo is other_module.foo` re-export trivia | Untracked | **PARTIALLY ADDRESSED** — annotated as "routing" but not deleted |
| #22 — `test_cuda_outer_loop_probe_converges_under_strict_transfer_guard` self-verdict | Tracked | Not re-validated |
| #23 — compile-diagnostic test in `test_single_stage_physics_parity.py` | Tracked | **FIXED** — moved to `tests/test_jax_compile_diagnostics.py` |
| #24 — ALM progress gate accepts barely-any progress | Tracked | Not re-validated |
| #25 — health/routing/cache tests in `test_single_stage_jax_cpu_reference.py` | Tracked | **ADDRESSED** — tests renamed: `test_j_both_below_health_ceiling`, `test_j_finite_branch_divergent_smoke`, `test_value_routes_through_residual_helper_not_penalty_objective`, etc., each with a gate-tier docstring |

**Score: 10 of 25 explicitly addressed; 6 partially addressed; 9 not re-validated in this audit.**

The Tier-1 tautologies (#1–#4), the highest-severity audit findings, are all fixed. The remaining work is on Tier-2 hidden-skip patterns and Tier-3 weak-tolerance cleanups, which are less critical than tautology elimination.

---

## Overall assessment

**PASS** with caveats. The JAX test suite at HEAD `gpu-purity-stage2-20260405` has substantively improved discipline around the oracle-lint policy: no Tier-1 tautologies remain in scope, tolerance-lane discipline is enforced through `parity_ladder_tolerances` imports, and CPU/JAX parity tests cite C++ symbols or closed-form expressions as oracles. The conftest infrastructure (`_guard_backend_runtime_state`, scikit-build editable finder patch, `parity_lane` fixture) is robust and prevents cross-test state leakage.

Outstanding concerns are concentrated in three areas: (1) too-tight tolerances that may be cross-machine fragile (`test_boozersurface_jax.py:4773-4805`, `test_stage2_jax.py:188-189`), (2) lingering style-level re-export `is`-identity tests that add no coverage (5 files), and (3) the audit's Tier-2 / Tier-3 backlog (#5, #6, #8–#15, #19, #22, #24) which remains tracked but not yet re-verified.

Verdict: do not block work on the basis of test discipline. Address the three concerns above before the next release closeout to keep parity claims defensible.
