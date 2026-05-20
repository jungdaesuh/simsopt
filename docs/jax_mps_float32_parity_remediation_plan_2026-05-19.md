# JAX MPS / Float32 Smoke and Production Parity Remediation Plan

Date: 2026-05-19

Status: Implementation in progress. Waves 1-5 have code changes and focused local proof in the current tree. The 2026-05-20 review fixed a strict transfer-guard Boozer decision-vector regression and reran the feasible CPU/MPS smoke slice. Later 2026-05-20 Wave 6 passes fixed the `surface_area_volume_simple` strict transfer-guard CPU/MPS path, recorded the full non-banana MPS all-supported failure artifact, and reran banana CPU/MPS float32 `scipy-jax` maxiter=7. Both banana float32 runs fail closed on all-NaN target-lane gradients and write no accepted `results.json`. Wave 8 adjacent-regression cleanup is partially implemented; Wave 7 CPU/CUDA production parity and final Wave 8 signoff remain pending.

## Review Evidence - 2026-05-20

- [x] Target proof interpreter recorded: `.conda/jax-0.10.0/bin/python` imports `jax==0.10.0`, `jaxlib==0.10.0`, backend `cpu`, devices `[CpuDevice(id=0)]`.
- [x] MPS smoke interpreter recorded: `.conda/jax-mps/bin/python` imports `jax==0.10.0`, `jaxlib==0.10.0`, backend `mps`, devices `[MpsDevice(id=0)]`.
- [x] Non-target base interpreter recorded: `/opt/homebrew/Caskroom/miniforge/base/bin/python` imports `jax==0.9.2`, `jaxlib==0.9.2`, backend `cpu`; do not use base `python` as plan-grade proof for this document.
- [x] Official docs rechecked for JAX transfer guard, `jax.custom_vjp`, `jax.lax.slice_in_dim`, JAX CUDA installation support, NVIDIA CUDA driver/toolkit compatibility, SIMSOPT Boozer residual/BoozerSurface contracts, and SciPy L-BFGS-B status semantics.
- [x] Context7 lookup was attempted during the doc review but quota-limited; this document's authoritative references are the direct official upstream links in "Official Documentation Constraints" below.
- [x] Fixed strict transfer-guard failure in Boozer decision-vector splitting: reverse-mode eager VJP uses a custom VJP split whose transpose concatenates cotangents without introducing a host-to-device scalar transfer; HVP/JVP/linearize callers use the plain `lax.slice_in_dim` split because JAX `custom_vjp` does not support forward-mode autodiff.
- [x] Split mode is now fail-fast: only `"reverse"` and `"jvp"` are accepted; unknown internal modes raise `ValueError`.
- [x] Tightened the repo-local non-finite optimizer status contract: status `6` now reports non-finite objective, iterate, or gradient consistently across host L-BFGS, private L-BFGS result conversion, SciPy reference normalization, and single-stage/continuation invalid-state parsing.
- [x] Pinned CPU validation passed:
  - `.conda/jax-0.10.0/bin/python -m pytest -q tests/geo/test_boozer_residual_jax.py -k "split_decision_vector or unpack_decision_vector"` -> `4 passed, 4 skipped, 42 deselected`.
  - `.conda/jax-0.10.0/bin/python -m pytest -q tests/test_runtime_dtype_policy.py` -> `21 passed`.
  - `.conda/jax-0.10.0/bin/python -m pytest -q tests/geo/test_boozersurface_jax.py -k "lbfgs_allows_mps_smoke_policy_default_reference_ls_lane or linear_solve_tolerance_uses_float32_smoke_floor"` -> `2 passed, 437 deselected`.
  - `.conda/jax-0.10.0/bin/python -m pytest -q tests/geo/test_surface_objectives_jax.py -k "traceable_hessian_solve_uses_dense_plu_forward_and_transpose or traceable_hessian_plu_solve_requires_forward_error_gate or traceable_hessian_plu_solve_is_jittable"` -> `3 passed, 313 deselected`.
  - `.conda/jax-0.10.0/bin/python -m pytest -q tests/test_jax_import_smoke.py -k "transfer_guard_disallow_enforces_single_stage_target_runtime_boundaries or transfer_guard_disallow_allows_squaredfluxjax_construction or transfer_guard_disallow_allows_stage2_target_objective_ondevice_entry"` -> `2 passed, 1 skipped, 118 deselected`; the skipped case is GPU-only (`stage2-target-objective-ondevice-entry`) because no CUDA device exists locally.
  - `.conda/jax-0.10.0/bin/python -m pytest -q tests/geo/test_optimizer_jax_reference.py tests/geo/test_optimizer_result_converters.py -k "nonfinite or finite_success or status_six"` -> `7 passed, 7 deselected`.
  - `.conda/jax-0.10.0/bin/python -m pytest -q tests/geo/test_single_stage_example.py -k "extract_optimizer_diagnostics_uses_nonfinite_message_fallback"` -> `1 passed, 328 deselected`.
  - `.conda/jax-0.10.0/bin/python -m pytest -q tests/geo/test_single_stage_continuation.py -k "reruns_invalid_completed_nonfinal_stage or stops_promotion_when_nonfinal_stage_contract_fails"` -> `2 passed, 44 deselected`.
  - `.conda/jax-0.10.0/bin/python -m pytest -q tests/geo/test_boozersurface_jax_private.py -k "minimize_lbfgs_host_core_nonfinite_step_terminates_status_nonfinite or lbfgs_result_status"` -> `1 passed, 95 deselected`.
  - `.conda/jax-0.10.0/bin/python -m pytest -q tests/test_jax_import_smoke.py::test_public_jax_helpers_are_exposed_on_package_roots tests/test_benchmark_helpers.py::test_artifact_host_helpers_use_direction_specific_transfer_guard tests/solve/test_wireframe_optimization_jax_item31.py::test_gsco_opposite_candidate_index_wraps_negative_to_positive` -> `3 passed`.
- [x] MPS smoke validation passed:
  - `.conda/jax-mps/bin/python -m pytest -q tests/test_jax_mps_smoke.py -m mps` -> `1 passed`.
  - `.conda/jax-mps/bin/python -m pytest -q tests/test_mps_smoke_dtype.py -m mps` -> `1 passed`.
  - `.conda/jax-mps/bin/python -m pytest -q tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_lbfgs_allows_mps_smoke_policy_default_reference_ls_lane` -> `1 passed`.
- [x] Direction-specific transfer-boundary review fix passed:
  - Removed remaining broad `jax.transfer_guard("allow")` wrappers from `surfaceobjectives_jax.py`; actual host conversions now use `jax.transfer_guard_device_to_host("allow")`.
  - `ruff format --check src/simsopt/geo/surfaceobjectives_jax.py && ruff check src/simsopt/geo/surfaceobjectives_jax.py && .conda/jax-0.10.0/bin/python -m py_compile src/simsopt/geo/surfaceobjectives_jax.py` -> pass.
  - `.conda/jax-0.10.0/bin/python -m pytest -q tests/geo/test_surface_objectives_jax.py::test_ensure_traceable_runtime_host_wrappers_defers_reporting_metrics_until_used tests/geo/test_surface_objectives_jax.py::test_traceable_seeded_initial_value_surfaces_failed_solve_gradient tests/geo/test_surface_objectives_jax.py::test_traceable_seeded_value_and_grad_builds_general_only_bundle tests/geo/test_surface_objectives_jax.py::test_traceable_runtime_host_wrappers_peel_baseline_without_touching_jitted_boundaries tests/geo/test_surface_objectives_jax.py::test_traceable_runtime_host_wrappers_surface_failed_solve_baseline_gradient` -> `5 passed`.
  - `rg -n 'transfer_guard\("allow"\)|transfer_guard\('allow'\)' src/simsopt/geo/surfaceobjectives_jax.py` -> no matches.
- [x] Wave 6 surface scalar strict transfer-guard rerun passed:
  - `SIMSOPT_BACKEND_MODE=jax_cpu_parity SIMSOPT_JAX_TRANSFER_GUARD=disallow ... .conda/jax-0.10.0/bin/python benchmarks/non_banana_example_cpp_jax_cpu_parity.py --fixtures surface_area_volume_simple --output-json .artifacts/jax_mps_float32_20260520/debug_surface_area_volume_simple_after_directional_guard_fix.json` -> exit `0`; fixture verdict `pass`.
  - `SIMSOPT_BACKEND_MODE=jax_mps_smoke SIMSOPT_JAX_TRANSFER_GUARD=disallow ... .conda/jax-mps/bin/python benchmarks/non_banana_example_cpp_jax_cpu_parity.py --fixtures surface_area_volume_simple --lanes cpu_cpp,jax_mps --baseline-json .artifacts/jax_mps_float32_20260520/non_banana_cpu_x64_baseline_all_supported_no_transfer_guard.json --output-json .artifacts/jax_mps_float32_20260520/non_banana_mps_surface_area_volume_simple_after_directional_guard_fix.json` -> exit `0`; fixture verdict `pass`.
  - The MPS artifact records `backend_mode=jax_mps_smoke`, `runtime_dtype=float32`, `host_dtype=float32`, `tolerance_tier=float32_smoke`, `parity_mode=false`, `transfer_guard=disallow`, `compute_transfer_guard=disallow`, `jax_mps_version=0.10.1`, and `float64_production_lane_exclusion` citing the `tillahoffmann/jax-mps` README constraint `MLX only supports float32.`.
- [x] Full non-banana MPS all-supported harness was run and failed with an artifact:
  - `SIMSOPT_BACKEND_MODE=jax_mps_smoke SIMSOPT_JAX_TRANSFER_GUARD=disallow ... .conda/jax-mps/bin/python benchmarks/non_banana_example_cpp_jax_cpu_parity.py --fixtures all-supported --lanes cpu_cpp,jax_mps --baseline-json .artifacts/jax_mps_float32_20260520/non_banana_cpu_x64_baseline_all_supported_no_transfer_guard.json --output-json .artifacts/jax_mps_float32_20260520/non_banana_mps_all_supported_after_directional_guard_fix.json` -> exit `1`.
  - Artifact summary: `27` fixtures, `1` pass, `26` fail: `15` strict transfer-guard host-to-device failures, `7` strict transfer-guard device-to-host failures, and `4` missing JAX-native wrapper failures in tracing fixtures. This artifact is evidence of remaining Wave 6 work, not acceptance.
- [x] Banana single-stage CPU/MPS float32 `scipy-jax` maxiter=7 reruns were executed under strict transfer guard after two explicit host-boundary fixes:
  - Added a single-stage initial/logging hardware-status D->H reporting boundary (`_evaluate_single_stage_hardware_status_reporting_boundary`) and verified it with `.conda/jax-0.10.0/bin/python -m pytest -q tests/geo/test_single_stage_example.py::HardwareConstraintTests::test_single_stage_hardware_status_reporting_boundary_allows_d2h` -> `1 passed`.
  - Added explicit D->H/H->D transfer boundaries to the SciPy-control target-lane bridge (`optimizer_jax_reference.py`) and verified them with `.conda/jax-0.10.0/bin/python -m pytest -q tests/geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_target_scipy_jax_marks_host_optimizer_transfer_boundaries tests/geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_target_scipy_jax_uses_scipy_control_with_jax_value_grad tests/geo/test_single_stage_example.py::HardwareConstraintTests::test_single_stage_hardware_status_reporting_boundary_allows_d2h` -> `4 passed`.
  - CPU run: `SIMSOPT_BACKEND_MODE=jax_cpu_float32_smoke SIMSOPT_BACKEND_STRICT=1 SIMSOPT_JAX_TRANSFER_GUARD=disallow JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 ... .conda/jax-0.10.0/bin/python examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py --backend jax --optimizer-backend scipy-jax --maxiter 7 --minimal-artifacts --output-root .artifacts/jax_mps_float32_20260520/banana_cpu_float32_single_stage_scipyjax_maxiter7_after_scipy_bridge_fix` -> exit `1`. Progress file: `.artifacts/jax_mps_float32_20260520/banana_cpu_float32_single_stage_scipyjax_maxiter7_after_scipy_bridge_fix/mpol=2-ntor=2-5e6eaec5/outer_optimizer_progress.json`.
  - MPS run: `SIMSOPT_BACKEND_MODE=jax_mps_smoke SIMSOPT_BACKEND_STRICT=1 SIMSOPT_JAX_TRANSFER_GUARD=disallow JAX_PLATFORMS=mps JAX_ENABLE_X64=0 ... .conda/jax-mps/bin/python examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py --backend jax --optimizer-backend scipy-jax --maxiter 7 --minimal-artifacts --output-root .artifacts/jax_mps_float32_20260520/banana_mps_single_stage_scipyjax_maxiter7_after_scipy_bridge_fix` -> exit `1`. Progress file: `.artifacts/jax_mps_float32_20260520/banana_mps_single_stage_scipyjax_maxiter7_after_scipy_bridge_fix/mpol=2-ntor=2-5e6eaec5/outer_optimizer_progress.json`.
  - Both CPU and MPS progress artifacts record `phase2_maxiter=7`, finite target-lane initial objective `1.1327383518218994`, all-NaN target-lane initial gradient (`nonfinite_count=11`, `size=11`, first classification `nan`), optimizer status `6`, message `Non-finite objective, iterate, or gradient encountered during iteration.`, diagnostics `fun_finite=True`, `jac_finite=False`, `x_finite=False`, `invalid_state=True`, and `results_json_count=0`. This is a correct fail-closed rejection, not a passing smoke result.
- [ ] Not yet run locally: all CUDA production parity checks.

## Purpose

Define a root-fix plan for the current JAX MPS / float32 smoke failures, adjacent parity-gate defects, and dirty-tree regressions surfaced during review without weakening production parity. This plan turns the validated issue inventory into executable work items with clear acceptance criteria.

The plan treats MPS as a float32 smoke lane unless and until a separate FP32 production contract is explicitly approved. Float64 production parity remains CPU C++ / SciPy oracle -> JAX CPU x64 -> JAX CUDA x64.

## Goals

- Preserve the existing production parity contract for float64 CPU/CUDA lanes.
- Keep float32 CPU and MPS lanes strict and non-production until they have their own accepted FP32 contract.
- Remove silent-success paths that allow non-finite optimizer state or non-finite JSON payloads into accepted artifacts.
- Fix the float32 adjoint failure at the linear-solve contract level, not by hiding NaNs or substituting fallback gradients.
- Make runtime dtype policy the single source of truth for JAX array construction.
- Restore clean public API boundaries so examples and core code do not depend on private runtime internals.
- Produce reproducible artifacts that record maxiter, backend mode, dtype policy, fixture/input hash, performance, memory, and parity status separately.

## Non-Goals

- Do not loosen float64 production tolerances.
- Do not promote MPS to production parity in this plan.
- Do not introduce synthetic gradients, CPU substitutions, silent retries, or fallback lanes.
- Do not treat performance or memory wins as correctness waivers.
- Do not rewrite unrelated JAX ports while fixing the MPS/float32 and artifact-gate contract.
- Do not promote broad dtype rewrites without proving the target path is part of float32 smoke or production parity.

## Validated State and Remaining Risks

- Float32 single-stage target-lane gradients fail closed with NaN sentinels (`surfaceobjectives_jax.py:3778-3785`) when the adjoint solve fails its success contract; the scalar objective stays finite because it reads cached baseline state.
- The target LS Boozer runtime lane intentionally carries `linear_solve_factors=None` (`surfaceobjectives_jax.py:3500-3508`) so compiled adjoint solves stay matrix-free. In the current tree this does **not** mean normal equations: `_solve_hessian_least_squares_system_with_status` documents "without forming normal equations", uses dense `lstsq` when the square operator can be materialized, otherwise uses operator-only GMRES, and then validates the original residual (`optimizer_jax.py:3970-3999`). The remaining banana CPU/MPS float32 smoke failure is therefore a real original-residual adjoint failure, not a current `H^T H` fallback.
- The current solve-status gate is `||A x - b|| / max(||b||, eps_runtime) <= effective_linear_solve_tolerance(policy, requested_tol)` (`optimizer_jax.py:3492-3569`). Float32 smoke now has `linear_solve_tolerance_floor=sqrt(eps(float32))≈3.45e-4` and `linear_solve_tolerance_cap=1e-3` (`runtime.py:190-201`; `tests/test_runtime_dtype_policy.py:52-66`), so requested tolerances are clamped into the float32 backward-error scale.
- The "11/11 NaN" diagnostic counts the 11-DOF gradient vector, not 11 term components; the term diagnostic enumerates the `_TRACEABLE_SINGLE_STAGE_OUTER_TERM_SPECS` list (`surfaceobjectives_jax.py:214-223, 5439-5482`), so `non_qs` always reports first because it is index 0.
- The CPU float32 smoke rerun now marks the optimizer result as failed when `fun`, `jac`, or `x` is non-finite (`optimizer_jax_reference.py:105-124`, `optimizer_jax_private/_result_converters.py:20-45`, `optimizer_host_lbfgs.py:1527-1553,1604-1624`).
- Pre-gate MPS smoke artifacts do not prove post-fix behavior. The stage-2 artifact (`.artifacts/production_parity_maxiter7_20260519/mps_stage2_smoke_r3/stage2_mps_trajectory.json`) shows a suspicious constant gradient-norm trajectory, and its nested `results.json` reports `OPTIMIZER_SUCCESS=True` with finite `FINAL_OBJECTIVE`/`FINAL_DOFS` but null `OPTIMIZER_FUN_FINITE`/`OPTIMIZER_JAC_FINITE`/`OPTIMIZER_INVALID_STATE` flags. A separate single-stage smoke artifact (`mps_single_stage_scipyjax_smoke_r1/boozer_init_progress.json`) reports `solve_success=true, iterations=0.0`. Wave 1 has added the current result-acceptance and independent finiteness gate; these old artifacts remain baseline evidence only.
- The packed-PLU runtime callbacks (`boozersurface_jax.py:3730-3865`) and the traceable PLU linearization (`surfaceobjectives_jax.py:3329-3407`) now report residual-quality status, not only finite solution entries.
- Diagnostic serialization is split from accepted serialization: `sanitize_diagnostic_payload` keeps NaN/Inf masking for diagnostic payloads, while `strict_accepted_payload` / `accepted_result_payload` reject non-finite accepted artifacts (`hardware_constraints.py:26-168`).
- Single-stage accepted artifact writes now route through `write_single_stage_final_artifact` and `write_single_stage_results_json` (`single_stage_banana_example.py:11227-11252`), so final-result finiteness is checked independently of the old target-lane metrics guard.
- MPS policy is float32 smoke by design (`runtime.py:310-326`): `runtime_dtype=float32`, `requires_x64=False`, `tolerance_tier=float32_smoke`, `default_optimizer_backend="scipy"`.
- The backend facade skew is closed in the current tree: the shadowed legacy `src/simsopt/backend.py` shim is deleted, and the package facade in `src/simsopt/backend/__init__.py` is the only public backend facade.
- Float32-smoke-critical dtype leaks are closed in the current tree: `SquaredFluxJAX._gather_field_free_dofs` now uses `runtime_jnp_dtype()` (`objectives/fluxobjective_jax.py:373-375`), `QfmSurfaceJAX._coil_set_spec` uses `runtime_jnp_dtype()` (`geo/qfmsurface_jax.py:73-76`), and curve geometry helper names no longer imply float64 (`jax_core/curve_geometry.py:60-84`). Off-critical-path float64 leaks (bootstrap, VMEC, PM/wireframe workflows) are still tracked in Wave 8.

## Official Documentation Constraints

The implementation must respect these upstream contracts. Pinned runtime: `jax==0.10.0`, `jaxlib==0.10.0`.

- JAX X64 is a process-global flag set at startup (`jax.config.update("jax_enable_x64", True)` or env `JAX_ENABLE_X64=1`); 64-bit dtypes are not a local per-call assumption. Reference: [JAX default dtypes and the X64 flag](https://docs.jax.dev/en/latest/default_dtypes.html).
- JAX transfer guard distinguishes explicit transfers (`jax.device_put*()`, `jax.device_get()`) from implicit transfers. Direction-specific settings are `jax_transfer_guard_host_to_device`, `jax_transfer_guard_device_to_device`, `jax_transfer_guard_device_to_host`, plus the `with jax.transfer_guard(level): ...` context manager. Only `disallow_explicit` blocks `device_get`/`device_put`; `log` and `disallow` target implicit transfers. Reference: [JAX transfer guard](https://docs.jax.dev/en/latest/transfer_guard.html).
- `jax.device_get()` is the explicit host materialization API and is classified as an explicit transfer by the guard. Reference: [jax.device_get](https://docs.jax.dev/en/latest/_autosummary/jax.device_get.html).
- The active MPS backend in this repo is `tillahoffmann/jax-mps` (MLX-backed PJRT plugin pinned to jaxlib 0.10.x). MLX supports float32 only, so float64 is structurally unavailable on this lane. Reference: [tillahoffmann/jax-mps](https://github.com/tillahoffmann/jax-mps). The legacy Apple `jax-metal` path was removed in this repo and now hard-fails as incompatible with jaxlib 0.10 (`runtime.py:41-55`); keep Apple's page only as historical context: [Apple Accelerated JAX on Mac](https://developer.apple.com/metal/jax/).
- SIMSOPT's BoozerSurface contract is a constrained least-squares problem whose residual vector comes from `boozer_surface_residual`; endpoint optimizer claims must therefore be grounded in residual/objective checks, not only a solver success flag. Reference: [SIMSOPT BoozerSurface docs](https://simsopt.readthedocs.io/latest/simsopt.geo.html#simsopt.geo.boozersurface.BoozerSurface).
- SciPy `OptimizeResult.status` is solver-specific and `message` is the explanatory contract. Reference: [SciPy OptimizeResult](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html).
- SciPy's low-level L-BFGS-B interface returns `(x, f, d)` with `d["warnflag"] ∈ {0, 1, 2}` and `d["task"]`; status `6` is not part of that low-level `warnflag` contract, so the repo-local `LBFGS_STATUS_NONFINITE = 6` (`optimizer_jax_private/_types.py:15`) is reserved for non-finite state in this codebase only and must not be presented as universal. Reference: [SciPy fmin_l_bfgs_b](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html).

## Rationale

The current failure is valuable signal: the target lane is correctly refusing to publish a gradient when the underlying adjoint solve cannot meet its contract. The remediation should keep that fail-closed behavior and fix the contracts around it.

There are three separate concerns that must not be mixed:

1. Correctness parity: objective, gradient, geometry, and optimizer endpoint agreement against the accepted oracle.
2. Numerical precision policy: float64 production versus float32 smoke or future FP32 production.
3. Operational metrics: performance and memory, reported after correctness is established.

The most important design choice is to make accepted artifacts strict and diagnostic artifacts explicit. JSON serialization may materialize device arrays and may sanitize values for diagnostics, but no accepted `results.json` should contain masked non-finite state.

## Math and Computation Contract

- The scalar objective contract is the production API. Optimizer endpoint comparisons are meaningful only after fixed-state value and gradient checks pass.
- A preconditioner or transformed system can be used only if the final solution passes the original residual contract for the original equation.
- Normal-equation solves square conditioning and are not an acceptable default for float32 adjoint parity unless the lane explicitly accepts the resulting precision limit.
- A singular/gauge-null least-squares adjoint contract must validate the original primal residual and not only the transformed normal residual.
- Boozer solve success must be tied to residual quality and objective/gradient validity, not only finite vector entries.
- Physics-facing quantities such as Boozer residual, iota, geometry distances, curvature, and surface/coil state must be checked at fixed state before optimizer endpoint claims.

## Blocking Decisions Before Implementation

These must be pinned before the corresponding wave runs.

- [x] (gates Wave 1) Decide whether rejected-run artifacts remain in the main run directory with a rejection marker, or move under `diagnostics/`. SSOT: rejected runs write `REJECTED.json` alongside diagnostic artifacts in the run directory.
- [x] (gates Wave 1) Decide whether `sanitize_json_payload` is split into `sanitize_diagnostic_payload` + (no sanitization for accepted artifacts) or kept as one function gated by an explicit `is_accepted=False` argument. KISS: prefer the split.
- [x] (gates Wave 2) Pin the singular/gauge-null LS adjoint solver contract. Chosen contract: original-operator solve/status using tier-scaled tolerance; no normal equations and no κ² success gate.
- [x] (gates Wave 2) Pin the residual metric and tolerance source for packed-factor solve success. SSOT: `||A x - b|| / max(||b||, eps_runtime) ≤ effective_linear_solve_tolerance(policy, requested_tol)`, where the helper clamps by `BackendPolicy.linear_solve_tolerance_floor/cap` (`parity` cap `1e-10`; current `float32_smoke` floor `sqrt(eps(float32))≈3.45e-4`, cap `1e-3`). The packed-factor and operator-only solves report the same status object shape.
- [x] (gates Wave 6) Decide whether float32 smoke gradients are diagnostic-only in every harness or gate a separate smoke contract. SSOT: float32 smoke gradients are diagnostic-only and do not gate the fixture verdict unless/until a separate FP32 production contract is approved. `validation_ladder_contract.py` records `production_parity=False` and `gradient_diagnostic_only=True`; `non_banana_example_cpp_jax_cpu_parity.py` tags float32-smoke gradient failures with `diagnostic_only=true` and excludes them from verdict-gating failures.
- [ ] (gates Wave 6+) Decide whether a formal FP32 production parity contract will be created later for MPS. Out of scope for this plan if deferred.

## Execution Plan

### Wave 0 - Pin Contracts Before Fixes

- [x] Add this plan to the remediation tracking checklist.
- [ ] Re-record the exact current source snapshot, branch (`gpu-purity-stage2-20260405`), and dirty-file list before the next implementation wave starts. The current branch is a dirty integration tree; do not reuse older line-number evidence as production proof.
- [x] Record the existing failing CPU float32 smoke artifact path and the pre-fix MPS trajectory artifact path (`.artifacts/production_parity_maxiter7_20260519/mps_stage2_smoke_r3/`) as baseline evidence.
- [x] Pin the Wave-1-gating decisions (rejection-marker location, sanitize-helper split). Pin the Wave-2-gating decisions (singular-LS solver contract, packed-factor residual metric).
- [x] Record the exact official-doc references used for dtype, transfer, MPS, and SciPy optimizer behavior (see Official Documentation Constraints above; cite the jax-mps README for the float32-only constraint).
- [x] Inventory root-level debug artifacts and decide delete, move, or gitignore before cleanup. Decision: gitignore confirmed root-local diagnostics; do not delete user artifacts during this plan.

Acceptance criteria:

- [x] Baseline evidence is listed in the implementation notes.
- [x] The production parity contract is not changed.
- [x] MPS is still documented as float32 smoke, not production parity.
- [x] Wave 1 cannot start before its gating decisions are pinned; Wave 2 cannot start before its gating decisions are pinned.

### Wave 1 - Accepted Artifact Gate SSOT

Purpose: prevent any accepted artifact from masking non-finite optimizer or reporting state.

Implementation tasks:

- [x] Split `sanitize_json_payload` (`examples/single_stage_optimization/hardware_constraints.py:23`) into `sanitize_diagnostic_payload` (current NaN/Inf→None behavior) and a strict accepted-payload contract that fails closed on non-finite entries. SSOT: one function per purpose, no `is_accepted` flag.
- [x] Introduce one result-acceptance helper covering both target-lane and non-target-lane writes. The helper proves:
  - [x] optimizer success is true;
  - [x] optimizer status is accepted for the selected optimizer contract (not `LBFGS_STATUS_NONFINITE=6`);
  - [x] objective value is finite (`jnp.isfinite` and not a sanitized `None`);
  - [x] final DOFs are finite;
  - [x] final gradient is finite when the lane contract requires a gradient;
  - [x] reporting metrics declared required by the artifact schema are finite;
  - [x] `backend_mode`, `runtime_dtype`, `host_dtype`, `tolerance_tier`, and `maxiter` are recorded.
- [x] Gate both target-lane and non-target-lane final artifact writes through `write_single_stage_final_artifact` / `write_single_stage_results_json` (`single_stage_banana_example.py:11227-11252`); delete the duplicated target-lane-only check.
- [x] If side artifacts (`outer_optimizer_progress.json`, `boozer_init_progress.json`, `target_lane_gradient_diagnosis.json`, etc.) continue to be written after failure, write one explicit `REJECTED.json` rejection marker alongside them so downstream validators do not have to scan logs.
- [x] Per-iteration `nonfinite_step` events (`optimizer_host_lbfgs.py:1359-1366`) escalate to `LBFGS_STATUS_NONFINITE` at termination instead of resolving silently to `status=0/1`.

Tests:

- [x] Unit test: failed optimizer refuses accepted `results.json` (`tests/geo/test_single_stage_example.py::test_write_single_stage_results_json_rejects_failed_optimizer`).
- [x] Unit test: optimizer success with non-finite final gradient refuses accepted `results.json` (`tests/geo/test_single_stage_example.py::test_write_single_stage_results_json_rejects_nonfinite_gradient`).
- [x] Unit test: shared final-artifact helper writes `REJECTED.json` and refuses accepted `results.json` for failed optimizer state (`tests/geo/test_single_stage_example.py::test_write_single_stage_final_artifact_writes_rejected_marker`).
- [x] Unit test: diagnostic artifact can be written for rejected run via `sanitize_diagnostic_payload` and is labeled rejected/diagnostic (`tests/geo/test_single_stage_example.py::test_single_stage_rejected_marker_records_failed_gate`).
- [x] Unit test: a transient non-finite iteration followed by line-search recovery still escalates to `LBFGS_STATUS_NONFINITE` at termination.
- [x] Integration smoke: rejected Stage 2 run writes no accepted `results.json` and writes one `REJECTED.json` (`tests/integration/test_stage2_jax.py:4528-4543`); banana CPU/MPS float32 maxiter=7 reruns also recorded `results_json_count=0`.
- [x] Regression test: the strict accepted-payload contract never substitutes `None` for NaN/Inf (`tests/geo/test_single_stage_example.py::test_accepted_result_payload_rejects_non_finite_numbers`).

Acceptance criteria:

- [x] No accepted result artifact contains `FINAL_OBJECTIVE: null`, `FINAL_DOFS: [null, ...]`, or equivalent masked non-finite values.
- [x] Downstream validators can determine accepted versus rejected state from the artifact contract without reading console logs.
- [x] One result-acceptance helper is the only accepted-artifact gate; example wrappers route final writes through it.

### Wave 2 - Linear-Solve Contract Root Fix

Purpose: fix the adjoint-solve failure at the numerical contract boundary.

Implementation tasks:

- [x] Split the adjoint linear-solve contract into three explicit solve kinds with one shared status-object shape `{success: bool, residual: float, residual_relative: float, iterations: int}`:
  - [x] square Hessian solve via `_solve_square_array_system_operator_only` (`optimizer_jax.py:3890-3909`);
  - [x] singular/gauge-null least-squares solve (current `_solve_hessian_least_squares_system_with_status` at `optimizer_jax.py:3970-3999`);
  - [x] packed-factor solve via the LS lane PLU triangular solves (`boozersurface_jax.py:3730-3865`, `surfaceobjectives_jax.py:3329-3407`).
- [x] Keep the no-factor LS runtime lane matrix-free and original-residual-gated. `surfaceobjectives_jax.py:3500-3508` intentionally returns `linear_solve_factors=None`, and `surfaceobjectives_jax.py:3250-3290` routes that case to the operator-backed solve; the root fix is that absence of PLU factors no longer means a normal-equation solve, CPU dense fallback, or unchecked success.
- [x] Pin the singular/gauge-null LS contract per the Wave-0 decision. Per-tier tolerance source: `BackendPolicy.linear_solve_tolerance_floor` / `cap` (`runtime.py:190-201`). Current `float32_smoke` uses floor `sqrt(eps(float32))≈3.45e-4` with cap `1e-3`, and the convergence gate must be the **original-operator** residual, not the κ²-squared normal-equation residual.
- [x] Replace the legacy operator-GMRES success gate with `||A x - b|| / max(||b||, eps_runtime) ≤ effective_linear_solve_tolerance(policy, requested_tol)` (`optimizer_jax.py:3492-3569`) so the gate uses the same floor/cap helper as Boozer solves. Float64 production lanes keep cap `1e-10`.
- [x] Add residual and finiteness checks to the packed-factor forward and transpose solve callbacks (`boozersurface_jax.py:3730-3865` and the traceable LS adjoint at `surfaceobjectives_jax.py:3329-3407`), using the same per-tier residual metric.
- [x] Preserve the fail-closed NaN sentinel (`_traceable_adjoint_gradient_or_nan`) when the selected solve contract fails.

Design constraints:

- [ ] No synthetic gradient fallback.
- [ ] No silent dense CPU substitution.
- [ ] No hidden retry path that changes backend or dtype.
- [ ] No tolerance change that applies to float64 production parity (`parity` tier cap remains 1e-10).
- [ ] No new defensive try/except around the solver entrypoints; status is reported through the shared status object.
- [ ] No κ² conditioning in the float32 lane: do not form `H^T H` or use the corresponding operator-action GMRES on the normal equations.

Tests:

- [ ] Fixed-state unit test for operator-only adjoint solve success/failure reporting under both `parity` and `float32_smoke` tiers. Current tree covers status shape and tolerance policy; the full two-tier solve matrix remains pending.
- [x] Fixed-state unit test for packed-factor residual-gated failure (finite-but-high-residual solution must fail).
- [x] Regression test showing the singular-LS contract validates against the original operator, not the transformed normal equation.
- [ ] CPU float32 smoke gradient diagnosis records solve status, original-operator residual, and iteration count.
- [ ] Float64 CPU parity gradient checks remain within existing tolerance (no drift in `tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py` numerical bounds).

Acceptance criteria:

- [ ] A failed adjoint solve still produces a failed gradient status, not an accepted fallback.
- [ ] A passed adjoint solve has finite solution and original-operator residual within the lane-specific contract.
- [ ] Float64 production tolerances are unchanged.
- [ ] Operator-only and packed-factor solves report the same status object shape.

### Wave 3 - Runtime Dtype Policy Cleanup (Float32 Smoke Critical Path Only)

Purpose: make float32 smoke actually exercise float32 paths end to end. YAGNI: only fix call sites reachable from the banana single-stage / non-banana parity smoke harnesses. Off-critical-path float64 leaks are deferred to Wave 8.

Critical-path implementation tasks (banana smoke reachability confirmed):

- [x] Replace the hardcoded `_as_jax_float64(self.field.x)` cast in `SquaredFluxJAX._gather_field_free_dofs` with the runtime-policy dtype helper (`objectives/fluxobjective_jax.py:373-375`).
- [x] Replace the hardcoded `jnp.float64` cast on `biotsavart.x` in `QfmSurfaceJAX._coil_set_spec` (`geo/qfmsurface_jax.py:73-76`) with the runtime-policy dtype helper.
- [x] Rename the former `_as_explicit_float64`, `_explicit_scalar`, `_ones_like_float64`, and `_zeros_like_float64` helpers to runtime-dtype names (`jax_core/curve_geometry.py:60-84`).
- [ ] Audit single-stage banana helper casts (`single_stage_banana_example.py`) and classify each as (a) runtime-policy, (b) host/SciPy boundary, or (c) intentional float64 production-only. Record the classification next to each call site.

Tests:

- [x] Unit test for `SquaredFluxJAX._gather_field_free_dofs` under both `jax_cpu_float32_smoke` and `jax_cpu_parity` policies.
- [x] Unit test for `QfmSurfaceJAX._coil_set_spec` under both policies.
- [x] Runtime dtype policy test that verifies representative arrays in the smoke critical path are float32 in `jax_cpu_float32_smoke` and `jax_mps_smoke`, and remain float64 in `jax_cpu_parity` and `jax_gpu_parity`.

Acceptance criteria:

- [x] Float32 smoke lanes do not upcast through helper or entrypoint paths on the smoke critical path.
- [x] Float64 production lanes still construct float64 arrays.
- [ ] Renamed helpers compile under the same call sites; no caller change needed beyond the rename.

Out of scope for this wave (tracked in Wave 8): bootstrap JAX profile derivatives, VMEC fieldline diagnostics, VMEC geometry helpers, PM workflow paths, wireframe workflow paths, MHD frozen-state casts. None of these are imported by the banana single-stage example or the non-banana CPU/MPS smoke harnesses (verified by grep against `tests/test_jax_mps_smoke.py`, `tests/test_mps_smoke_dtype.py`, and `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`).

### Wave 4 - API Boundary Cleanup

Purpose: restore clean public import boundaries and remove visible private-module dependencies from examples.

Implementation tasks:

- [x] Delete `src/simsopt/backend.py`. It is shadowed by the `src/simsopt/backend/` package — Python resolves `import simsopt.backend` to `backend/__init__.py`, so the shim is dead code at runtime and drifts from the SSOT facade. SSOT: one facade in `backend/__init__.py`.
- [x] Update `src/simsopt/_core/optimizable.py` and `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:88` to import `get_tolerance_tier` from `simsopt.backend` instead of `simsopt.backend.runtime`. Apply the same fix to any other public-helper imports that reach into `simsopt.backend.runtime`.
- [x] Promote `_coil_dofs_gradient_to_derivative` (`surfaceobjectives_jax.py:1962`) to public `coil_dofs_gradient_to_derivative`, re-export from `simsopt.geo`, and update `single_stage_banana_example.py:104`.
- [x] Treat `src/simsopt/jax_core/_math_utils.py` as a documented compatibility facade (its module docstring already states "Compatibility facade for backend-owned JAX dtype helpers"). Either rename it to drop the underscore or add an explicit `# public-facade: underscore-prefixed for legacy compatibility` marker so the audit grep does not re-flag it.
- [x] Remove `_gsco_opposite_candidate_index` from `__all__` in `src/simsopt/solve/wireframe_optimization_jax.py:35` (private name should not be exported).
- [ ] Promote, relocate, or remove the banana example's remaining private optimizer import `_mark_cacheable_jit_value_and_grad` (`single_stage_banana_example.py:118`) before closing the "No example requires a private core symbol" acceptance item.

Tests:

- [x] Import test: `from simsopt.backend import get_tolerance_tier, raise_if_target_lane_bypass, strict_target_lane_purity, target_lane_purity_active, target_lane_purity_requested` resolves cleanly.
- [x] Import test runs under an isolated current-repo import path (for example `python -S` with `src` inserted) so ambient editable installs cannot hide backend module/package resolution.
- [x] Import test: `from simsopt.geo import coil_dofs_gradient_to_derivative` (or chosen public name) resolves.
- [ ] Example import smoke test for single-stage banana entrypoint.
- [x] Static grep check that no example or production source under `src/` imports `simsopt.backend.runtime` symbols that are already re-exported by `simsopt.backend`.

Acceptance criteria:

- [x] Public helper imports resolve through the public facade.
- [ ] No example requires a private core symbol for normal operation.
- [x] No dead facade module exists in the package tree.

### Wave 5 - Transfer Guard and Host Materialization Boundary

Purpose: keep compute/probe paths strict while allowing explicit artifact and MPI materialization boundaries.

Implementation tasks:

- [x] Wrap the MPI Jacobian materialization at `src/simsopt/solve/mpi_jax.py:70` (`np.asarray(jax.device_get(local_columns))`) in `jax.transfer_guard_device_to_host("allow")`.
- [x] Replace the per-row `jax.device_get` at `src/simsopt/solve/serial_jax.py:100-102` with a batched materialization at write boundary, also under `jax.transfer_guard_device_to_host("allow")`.
- [x] Audit `mpi_jax.py:114-127` non-leader worker loop: ensure shutdown handles `command is None` as well as `STOP` to avoid hangs when sentinel propagation differs (e.g., mock comm in tests).
- [ ] Keep accepted compute and gradient validations under strict transfer guard (`disallow` for production parity lanes and the MPS smoke rerun). The full all-supported MPS artifact still reports strict-transfer failures, so this remains open.
- [x] Consolidate duplicated artifact materialization helpers: `_jax_artifact_host_array` (`benchmarks/non_banana_example_parity_fixtures.py:35`) and `_artifact_host_value` / `_host_float_array` (`benchmarks/non_banana_example_cpp_jax_cpu_parity.py:344`) into one helper in `benchmarks/run_code_benchmark_common.py` (or a dedicated `benchmarks/_host_io.py`). DRY: one materialization helper.
- [x] Use direction-specific `jax.transfer_guard_device_to_host("allow")` at host materialization boundaries instead of broad `jax.transfer_guard("allow")`.

Tests:

- [ ] Transfer-guard strict test for compute path (`SquaredFluxJAX.dJ`, `BoozerSurfaceJAX.run_code`) — no D→H transfers during compute.
- [x] Transfer-guard allowed test for artifact serialization boundary.
- [ ] Transfer-guard allowed test for MPI/serial host-solver boundary if those modules are kept; if `mpi_jax.py` / `serial_jax.py` are out of scope for production, mark them as such and add the test only at the boundary entrypoint.
- [x] MPS smoke test with `SIMSOPT_JAX_TRANSFER_GUARD=disallow` configured for the proven narrow fixture and the failing all-supported artifact.

Acceptance criteria:

- [ ] JAX transfer-guard logs or errors do not appear from compute paths.
- [x] Artifact and logging boundaries materialize arrays explicitly and locally.
- [x] Exactly one artifact-materialization helper exists in the benchmarks tree.
- [x] Host materialization boundaries are auditable by a single grep on `transfer_guard_device_to_host`.

### Wave 6 - MPS and Float32 Smoke Rerun

Purpose: rerun MPS only after the artifact gate, dtype policy, transfer boundary, and solve-status fixes are in place.

Run order:

- [x] `tests/test_jax_mps_smoke.py -m mps`
- [x] `tests/test_mps_smoke_dtype.py -m mps`
- [x] `tests/test_runtime_dtype_policy.py`
- [x] `tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_lbfgs_allows_mps_smoke_policy_default_reference_ls_lane`
- [x] Non-banana parity harness with `SIMSOPT_BACKEND_MODE=jax_mps_smoke` executed. Narrow `surface_area_volume_simple` passes; full `all-supported` run exits `1` with `.artifacts/jax_mps_float32_20260520/non_banana_mps_all_supported_after_directional_guard_fix.json` (`1` pass, `26` fail). Remaining failures block acceptance.
- [x] Banana single-stage CPU float32 smoke, maxiter=7 executed; failed closed on all-NaN target-lane gradient and wrote no accepted `results.json`.
- [x] Banana single-stage MPS float32 smoke, maxiter=7 executed; failed closed on all-NaN target-lane gradient and wrote no accepted `results.json`.

Artifact requirements:

- [x] `backend_mode` recorded for the proven MPS surface artifact and failed full MPS artifact.
- [x] `runtime_dtype` recorded for the proven MPS surface artifact and failed full MPS artifact.
- [x] `host_dtype` recorded for the proven MPS surface artifact and failed full MPS artifact.
- [x] `tolerance_tier` recorded for the proven MPS surface artifact and failed full MPS artifact.
- [x] `maxiter=7` recorded in banana CPU/MPS `outer_optimizer_progress.json` artifacts.
- [x] fixture/input hash recorded for non-banana artifacts.
- [x] source snapshot recorded for non-banana artifacts.
- [x] correctness verdict recorded for non-banana artifacts.
- [x] performance metrics recorded separately as per-lane timing for non-banana artifacts.
- [ ] memory metrics recorded separately.

Acceptance criteria:

- [ ] MPS smoke either passes its documented smoke contract or fails with a rejected artifact. Current status: the narrow surface fixture passes; full non-banana all-supported MPS still fails with a non-acceptance artifact; banana MPS maxiter=7 fails closed with no accepted `results.json` because the target-lane gradient is all NaN.
- [x] MPS smoke is not reported as production parity in the proven non-banana MPS artifacts.
- [x] No CPU fallback or silent backend substitution occurs in the proven non-banana MPS artifacts; metadata records `jax_backend=mps` and selected lane `jax_mps`.
- [x] MPS artifact records the `tillahoffmann/jax-mps` (MLX) float32-only constraint as the reason it is not a float64 production lane, citing the jax-mps README rather than the unmaintained Apple `jax-metal` page.

### Wave 7 - Float64 Production Parity Rerun

Purpose: re-establish production-grade proof after dtype-policy and gate changes.

Run order:

- [ ] CPU C++ / SciPy fixed-state oracle checks.
- [ ] JAX CPU x64 fixed-state checks against the oracle.
- [ ] JAX CPU x64 non-banana parity matrix.
- [ ] JAX CPU x64 banana single-stage maxiter=7.
- [ ] JAX CUDA x64 fixed-state checks against the same oracle.
- [ ] JAX CUDA x64 non-banana parity matrix.
- [ ] JAX CUDA x64 banana single-stage maxiter=7.
- [ ] CPU/CUDA trajectory, endpoint, performance, and memory artifact comparison.

Acceptance criteria:

- [ ] Same source snapshot across lanes.
- [ ] Same fixture/input hash across lanes.
- [ ] Same seed/config across lanes.
- [ ] Same scalar objective contract across lanes.
- [ ] Fixed-state value and gradient checks pass before interpreting optimizer endpoints.
- [ ] End-of-run checks compare final objective, gradient/constraint state, geometry outputs, performance, and memory.
- [ ] Performance and memory are reported separately and do not waive correctness.

### Wave 8 - Adjacent Dirty-Tree Regression Audit and Off-Critical-Path Dtype Hygiene

Purpose: track validated regressions, stale-code risks, and off-critical-path dtype hygiene that do not block the MPS smoke rerun but can invalidate downstream production claims.

Implementation tasks (adjacent regressions):

- [x] QFM augmented Lagrangian: `QfmAugmentedLagrangianInfo.fun` is the raw QFM value, and `augmented_value` carries the augmented objective. The contract is pinned by `tests/geo/test_qfmsurface_jax.py`.
- [x] Relax-and-split JAX: default `epsilon_RS=1.0e-3` short-circuit is intentional parity with the CPU implementation and is covered by a default-argument CPU/JAX oracle test in `tests/solve/test_permanent_magnet_optimization_jax_item28.py`.
- [x] Optimizer status code: document `LBFGS_STATUS_NONFINITE = 6` (`optimizer_jax_private/_types.py:10-14`) as repo-local and verify `success`, `status`, and `message` coverage for both the SciPy adapter (`optimizer_jax_reference.py:113-119`) and on-device/private adapters (`tests/geo/test_optimizer_jax_reference.py`, `tests/geo/test_optimizer_result_converters.py`, `tests/geo/test_boozersurface_jax_private.py`).
- [x] BFGS curvature criterion in QFM (`jax_core/qfm_solver.py`): restore the canonical `sqrt(eps) * ||y|| * ||s||` floor and pin the float32 boundary against the shared private-BFGS oracle in `tests/geo/test_qfmsurface_jax.py`.
- [x] `_normalize_scipy_result` status 6 is documented as repo-local in the constant comment; SciPy's low-level L-BFGS-B `warnflag` contract remains 0/1/2.
- [x] Float32 smoke harness: gradient entries tagged `diagnostic_only` no longer make the fixture verdict fail. SSOT: `validation_ladder_contract.py` marks the tier as `production_parity=False` / `gradient_diagnostic_only=True`, and `non_banana_example_cpp_jax_cpu_parity.py` excludes diagnostic-only gradient failures from verdict-gating failures.
- [ ] Tautology tests: replace change-pinning tests with CPU, analytic, or committed-fixture oracle tests where the behavior is contract-bearing. Current QFM augmented-Lagrangian tests now include upstream exact/KKT oracle coverage (`tests/geo/test_qfmsurface_jax.py:300-430`); the remaining concrete candidate is the high-`epsilon_RS` forced scan-stop test (`tests/solve/test_permanent_magnet_optimization_jax_item28.py:545-560`).

Implementation tasks (off-critical-path dtype hygiene; deferred from Wave 3):

- [ ] Replace hardcoded float64 in bootstrap JAX profile derivative paths (`src/simsopt/mhd/bootstrap_jax.py:144, 174`).
- [ ] Replace hardcoded float64 in VMEC fieldline diagnostics (`src/simsopt/mhd/vmec_diagnostics_jax.py:60, 61, 64, 74`).
- [ ] Replace hardcoded float64 in VMEC geometry helpers (`src/simsopt/jax_core/vmec_geometry.py:289`).
- [ ] Audit and replace runtime-float64 helper usage in PM workflow paths where the value should follow runtime dtype. Current grep shows `_as_jax_float64` call sites in `src/simsopt/jax_core/pm_workflow.py` around lines `291-1092`; classify physics-required host precision separately from runtime-policy arrays before editing.
- [ ] Audit and replace runtime-float64 helper usage in wireframe workflow paths where the value should follow runtime dtype. Current grep shows `_as_jax_float64` call sites in `src/simsopt/jax_core/wireframe_workflow.py` around lines `546-1118`; classify physics-required host precision separately from runtime-policy arrays before editing.
- [ ] Audit tracing, magnetic-axis, and MHD frozen-state float64 casts before claiming float32 end-to-end coverage outside the banana smoke perimeter.

Cleanup tasks:

- [x] MPI and serial JAX solvers remain in scope: the Wave-5 transfer-boundary and Wave-1 result-gate contracts apply to `src/simsopt/solve/mpi_jax.py` / `serial_jax.py`, with coverage in `tests/solve/test_mpi_jax.py` and `tests/solve/test_serial_jax.py`.
- [x] Root-level debug outputs: remove or gitignore `jax_mem_test.py`, `objective_runtimes_semilogy.png`, `taylor_errors.png`, `test_coil.vtu`, and `.gpd/state.json.bak`.
- [x] `simsopt/mhd/__init__.py` no longer imports `jax as _`; it uses `importlib.util.find_spec("jax")` and `_jax_spec` instead.

Acceptance criteria:

- [ ] Adjacent regressions are either fixed, moved to a separate dated plan, or explicitly declared out of scope before production parity is claimed.
- [ ] No test is accepted as an oracle if it only reasserts the implementation under test.
- [ ] No untracked solver module is used in validation without being added to the artifact/test contract.
- [ ] Off-critical-path dtype hygiene completes after the float32 smoke rerun closes; production parity is not claimed before the hygiene is closed.

## Priority TODO List

- [x] P0: Split `sanitize_json_payload` into a strict accepted-payload helper and a `sanitize_diagnostic_payload`.
- [x] P0: Implement one accepted-artifact gate SSOT covering both target and non-target lanes.
- [x] P0: Add final-result finiteness checks independent of optimizer success, including escalation of transient `nonfinite_step` events to `LBFGS_STATUS_NONFINITE` at termination.
- [x] P0: Tier-scale the operator-GMRES success gate by `effective_linear_solve_tolerance(policy, requested_tol)`; no change to float64 parity.
- [x] P0: Keep the no-factor LS adjoint path matrix-free and original-residual-gated; absence of PLU factors no longer implies a normal-equation solve, CPU substitution, or unchecked success.
- [x] P0: Pin the residual metric `||A x - b|| / max(||b||, eps_runtime)` as the SSOT solve-success criterion for both operator-only and packed-factor solves.
- [x] P0: Add residual gates to packed-factor LS callbacks (`boozersurface_jax.py:3730-3865`, `surfaceobjectives_jax.py:3329-3407`).
- [x] P1: Clean runtime dtype policy leaks on the float32 smoke critical path (`objectives/fluxobjective_jax.py:373-375`, `geo/qfmsurface_jax.py:73-76`).
- [x] P1: Delete the shadowed `src/simsopt/backend.py` shim; the package `backend/__init__.py` is the only facade.
- [ ] P1: Finish banana-example API import cleanup. Public backend/geo imports are cleaned (`single_stage_banana_example.py:88,104`), `_math_utils` is documented as a compatibility facade, but the private optimizer import `_mark_cacheable_jit_value_and_grad` remains tracked in Wave 4.
- [x] P1: Add transfer-guard materialization boundaries at `mpi_jax.py:70` and `serial_jax.py:100-102`.
- [x] P1: Consolidate duplicated artifact-materialization helpers in `benchmarks/`.
- [ ] P1: Rerun CPU float32 smoke and MPS float32 smoke (Wave 6). Current status: non-banana MPS narrow surface fixture passes, full all-supported run produced a failing artifact, and banana CPU/MPS `scipy-jax` maxiter=7 both fail closed on all-NaN target-lane gradients. Acceptance remains blocked.
- [ ] P1: Rerun float64 CPU production parity (Wave 7) — the latest CPU parity artifact on this branch is dated 2026-05-18, before the recent dtype-centralization commits.
- [x] P1: Audit QFM, relax-and-split, and optimizer-status adjacent regressions before production signoff.
- [ ] P2: Rerun CUDA production parity on Perlmutter.
- [x] P2: Clean root-level debug artifacts by gitignoring the confirmed root-local diagnostics.
- [x] P2: Rename stale `*_float64`-suffixed helpers in `curve_geometry.py` to runtime-dtype names (`jax_core/curve_geometry.py:60-84`).
- [ ] P2: Off-critical-path dtype hygiene (bootstrap, VMEC, PM, wireframe) per Wave 8.

## Review Checklist

- [ ] No fallbacks were added.
- [ ] No defensive try/except wrappers were added.
- [ ] No production tolerance was loosened (parity tier cap remains 1e-10).
- [ ] No κ² conditioning in the float32 lane — no normal-equation operator on `H^T H`.
- [ ] No accepted artifact masks NaN/Inf as null.
- [ ] No MPS artifact is labeled production parity.
- [ ] No performance result is used to waive correctness.
- [ ] No dead facade module remains (`backend.py` shim deleted).
- [ ] Public imports use public facades.
- [ ] Dtype follows backend runtime policy on the float32 smoke critical path.
- [ ] Transfer guard is strict outside explicit materialization boundaries, using `jax_transfer_guard_device_to_host` rather than the broad guard.
- [ ] Tests include both float32 smoke and float64 parity paths.
- [ ] Solver changes are validated against original residuals, not only transformed systems.
- [ ] Operator-only and packed-factor solves report the same status object shape.
- [ ] Physics outputs are checked at fixed state before endpoint interpretation.
- [ ] Tautology tests are not counted as production proof.
- [x] Official-doc constraints above are still current at implementation time (jax-mps for MPS, jax-metal as historical context only).

## Open Decisions

- [x] Decide whether root debug artifacts should be deleted, moved, or gitignored (Wave 8 cleanup). Decision: gitignore root-local diagnostics, including `.gpd/state.json.bak`.
- [x] Decide whether adjacent dirty-tree regressions (QFM augmented Lagrangian semantics, relax-and-split `epsilon_RS`, QFM BFGS curvature floor) are fixed in this plan or split into a separate dated plan after Wave 1. Decision: fix and test them in this plan.
