# JAX MPS / Float32 Smoke and Production Parity Remediation Plan

Date: 2026-05-19

Status: Draft plan after live-tree review. No implementation is implied by this file.

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

## Validated Current State

- Float32 single-stage target-lane gradients fail closed with NaN sentinels (`surfaceobjectives_jax.py:3762-3769`) when the adjoint solve fails its success contract; the scalar objective stays finite because it reads cached baseline state.
- The target LS Boozer path forces `linear_solve_factors=None` (`surfaceobjectives_jax.py:3484-3492, 4081`) and routes adjoint solves through operator GMRES on the normal equations (`optimizer_jax.py:3445-3481`), which squares conditioning; the CPU float32 smoke diagnosis shows this path fails the residual gate at the smoke fixture.
- The current operator-GMRES success gate `||residual|| ≤ max(1e-12, 10·tol·||rhs||)` uses the effective Boozer linear-solve tolerance (`boozersurface_jax.py:3563-3585`). Float32 smoke currently has `linear_solve_tolerance_floor=1e-6` and `linear_solve_tolerance_cap=None` (`runtime.py:190-201`; `tests/test_runtime_dtype_policy.py:52-66`), so the smoke tolerance bottoms out at `1e-6` rather than being capped at `1e-6`.
- The "11/11 NaN" diagnostic counts the 11-DOF gradient vector, not 11 term components; the term diagnostic enumerates the `_TRACEABLE_SINGLE_STAGE_OUTER_TERM_SPECS` list (`surfaceobjectives_jax.py:212-221, 5134-5148`), so `non_qs` always reports first because it is index 0.
- The CPU float32 smoke rerun now marks the optimizer result as failed when `fun`, `jac`, or `x` is non-finite (`optimizer_jax_reference.py:93-105`, `optimizer_jax_private/_result_converters.py:20-24`, `optimizer_host_lbfgs.py:1513-1536`).
- Pre-gate MPS smoke artifacts do not prove post-fix behavior. The stage-2 artifact (`.artifacts/production_parity_maxiter7_20260519/mps_stage2_smoke_r3/stage2_mps_trajectory.json`) shows a suspicious constant gradient-norm trajectory, and its nested `results.json` reports `OPTIMIZER_SUCCESS=True` with finite `FINAL_OBJECTIVE`/`FINAL_DOFS` but null `OPTIMIZER_FUN_FINITE`/`OPTIMIZER_JAC_FINITE`/`OPTIMIZER_INVALID_STATE` flags. A separate single-stage smoke artifact (`mps_single_stage_scipyjax_smoke_r1/boozer_init_progress.json`) reports `solve_success=true, iterations=0.0`. The gap is the missing result-acceptance and independent finiteness gate that Wave 1 closes.
- The packed-PLU runtime callbacks (`boozersurface_jax.py:3700-3706`) and the traceable PLU linearization (`surfaceobjectives_jax.py:3217-3299`) only check finite solution entries and do not check residual quality.
- `sanitize_json_payload` (`examples/single_stage_optimization/hardware_constraints.py:23-39`) maps NaN/Inf floats to `None`. The same helper is invoked by both accepted-artifact writes (`banana_coil_solver.py:2471`, `single_stage_banana_example.py:9169`) and diagnostic writes; no caller-side finiteness gate exists.
- The target-lane final metrics gate (`single_stage_banana_example.py:5316-5320`) only checks `optimizer_success is False`; it does not independently validate final-result finiteness, and the non-target-lane path (`:5326-5377`) lacks any final optimizer-result gate.
- MPS policy is float32 smoke by design (`runtime.py:310-326`): `runtime_dtype=float32`, `requires_x64=False`, `tolerance_tier=float32_smoke`, `default_optimizer_backend="scipy"`.
- The backend facade skew is real but narrower than reported. `get_tolerance_tier` is already re-exported by `src/simsopt/backend/__init__.py`. The four target-lane purity helpers (`raise_if_target_lane_bypass`, `strict_target_lane_purity`, `target_lane_purity_active`, `target_lane_purity_requested`) live in the package facade as well; `src/simsopt/backend.py` is a shadowed legacy module — Python resolves `import simsopt.backend` to the package, so the shim is dead at runtime.
- Float32-smoke-critical paths that still force float64: `SquaredFluxJAX._gather_field_free_dofs` (`fluxobjective_jax.py:346-348`) and `QfmSurfaceJAX._coil_set_spec` (`qfmsurface_jax.py:73`). Off-critical-path float64 leaks (bootstrap, VMEC, PM/wireframe workflows) are tracked separately in Wave 8.

## Official Documentation Constraints

The implementation must respect these upstream contracts. Pinned runtime: `jax==0.10.0`, `jaxlib==0.10.0`.

- JAX X64 is a process-global flag set at startup (`jax.config.update("jax_enable_x64", True)` or env `JAX_ENABLE_X64=1`); 64-bit dtypes are not a local per-call assumption. Reference: [JAX default dtypes and the X64 flag](https://docs.jax.dev/en/latest/default_dtypes.html).
- JAX transfer guard distinguishes explicit transfers (`jax.device_put*()`, `jax.device_get()`) from implicit transfers. Direction-specific settings are `jax_transfer_guard_host_to_device`, `jax_transfer_guard_device_to_device`, `jax_transfer_guard_device_to_host`, plus the `with jax.transfer_guard(level): ...` context manager. Only `disallow_explicit` blocks `device_get`/`device_put`; `log` and `disallow` target implicit transfers. Reference: [JAX transfer guard](https://docs.jax.dev/en/latest/transfer_guard.html).
- `jax.device_get()` is the explicit host materialization API and is classified as an explicit transfer by the guard. Reference: [jax.device_get](https://docs.jax.dev/en/latest/_autosummary/jax.device_get.html).
- The active MPS backend in this repo is `tillahoffmann/jax-mps` (MLX-backed PJRT plugin pinned to jaxlib 0.10.x). MLX supports float32 only, so float64 is structurally unavailable on this lane. Reference: [tillahoffmann/jax-mps](https://github.com/tillahoffmann/jax-mps). The legacy Apple `jax-metal` path was removed in this repo and now hard-fails as incompatible with jaxlib 0.10 (`runtime.py:41-55`); keep Apple's page only as historical context: [Apple Accelerated JAX on Mac](https://developer.apple.com/metal/jax/).
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

- [ ] (gates Wave 1) Decide whether rejected-run artifacts remain in the main run directory with a rejection marker, or move under `diagnostics/`. SSOT: pick one location.
- [ ] (gates Wave 1) Decide whether `sanitize_json_payload` is split into `sanitize_diagnostic_payload` + (no sanitization for accepted artifacts) or kept as one function gated by an explicit `is_accepted=False` argument. KISS: prefer the split.
- [ ] (gates Wave 2) Pin the singular/gauge-null LS adjoint solver contract. Options to choose from: (a) tier-scaled tolerance on the existing `_solve_square_array_system_operator_only` path using `BackendPolicy.linear_solve_tolerance_floor/cap` (no κ²-squaring, only tolerance widening for `float32_smoke`); (b) LSQR/LSMR on the original operator (no normal equations); (c) iterative refinement around a packed-factor inner solve. The chosen contract must validate against the **original** (un-transformed) residual.
- [ ] (gates Wave 2) Pin the residual metric and tolerance source for packed-factor solve success. SSOT: `||A x - b|| / max(||b||, eps_runtime) ≤ effective_linear_solve_tolerance(policy, requested_tol)`, where the helper clamps by `BackendPolicy.linear_solve_tolerance_floor/cap` (`parity` cap `1e-10`; current `float32_smoke` floor `1e-6`, cap `None`). If implementation chooses a finite smoke cap, update `runtime.py` and policy tests in the same wave. The packed-factor and operator-only solves report the same status object shape.
- [ ] (gates Wave 6) Decide whether float32 smoke gradients are diagnostic-only in every harness or gate a separate smoke contract. The current `validation_ladder_contract.py` marks float32 smoke as `production_parity=False`, but `non_banana_example_cpp_jax_cpu_parity.py:596, 615-617, 2740-2745` still computes a `verdict="fail"` for diagnostic-tagged comparisons. SSOT: pick one rule.
- [ ] (gates Wave 6+) Decide whether a formal FP32 production parity contract will be created later for MPS. Out of scope for this plan if deferred.

## Execution Plan

### Wave 0 - Pin Contracts Before Fixes

- [ ] Add this plan to the remediation tracking checklist.
- [ ] Record the exact current source snapshot, branch (`gpu-purity-stage2-20260405`), and dirty-file list before implementation starts.
- [ ] Record the existing failing CPU float32 smoke artifact path and the pre-fix MPS trajectory artifact path (`.artifacts/production_parity_maxiter7_20260519/mps_stage2_smoke_r3/`) as baseline evidence.
- [ ] Pin the Wave-1-gating decisions (rejection-marker location, sanitize-helper split). Pin the Wave-2-gating decisions (singular-LS solver contract, packed-factor residual metric).
- [ ] Record the exact official-doc references used for dtype, transfer, MPS, and SciPy optimizer behavior (see Official Documentation Constraints above; cite the jax-mps README for the float32-only constraint).
- [ ] Inventory root-level debug artifacts and decide delete, move, or gitignore before cleanup.

Acceptance criteria:

- [ ] Baseline evidence is listed in the implementation notes.
- [ ] The production parity contract is not changed.
- [ ] MPS is still documented as float32 smoke, not production parity.
- [ ] Wave 1 cannot start before its gating decisions are pinned; Wave 2 cannot start before its gating decisions are pinned.

### Wave 1 - Accepted Artifact Gate SSOT

Purpose: prevent any accepted artifact from masking non-finite optimizer or reporting state.

Implementation tasks:

- [ ] Split `sanitize_json_payload` (`examples/single_stage_optimization/hardware_constraints.py:23`) into `sanitize_diagnostic_payload` (current NaN/Inf→None behavior) and a strict accepted-payload contract that fails closed on non-finite entries. SSOT: one function per purpose, no `is_accepted` flag.
- [ ] Introduce one result-acceptance helper covering both target-lane and non-target-lane writes. The helper proves:
  - [ ] optimizer success is true;
  - [ ] optimizer status is accepted for the selected optimizer contract (not `LBFGS_STATUS_NONFINITE=6`);
  - [ ] objective value is finite (`jnp.isfinite` and not a sanitized `None`);
  - [ ] final DOFs are finite;
  - [ ] final gradient is finite when the lane contract requires a gradient;
  - [ ] reporting metrics declared required by the artifact schema are finite;
  - [ ] `backend_mode`, `runtime_dtype`, `host_dtype`, `tolerance_tier`, and `maxiter` are recorded.
- [ ] Gate both target-lane (`single_stage_banana_example.py:5316-5320`) and non-target-lane (`:5326-5377`) final reporting through that one helper; delete the duplicated target-lane-only check.
- [ ] If side artifacts (`outer_optimizer_progress.json`, `boozer_init_progress.json`, `target_lane_gradient_diagnosis.json`, etc.) continue to be written after failure, write one explicit `REJECTED.json` rejection marker alongside them so downstream validators do not have to scan logs.
- [ ] Per-iteration `nonfinite_step` events (`optimizer_host_lbfgs.py:1359-1366`) escalate to `LBFGS_STATUS_NONFINITE` at termination instead of resolving silently to `status=0/1`.

Tests:

- [ ] Unit test: target lane with `optimizer_success=False` refuses accepted `results.json`.
- [ ] Unit test: target lane with `optimizer_success=True` but non-finite final gradient refuses accepted `results.json`.
- [ ] Unit test: non-target lane with failed optimizer refuses accepted `results.json`.
- [ ] Unit test: diagnostic artifact can be written for rejected run via `sanitize_diagnostic_payload` and is labeled rejected/diagnostic.
- [ ] Unit test: a transient non-finite iteration followed by line-search recovery still escalates to `LBFGS_STATUS_NONFINITE` at termination.
- [ ] Integration smoke: CPU float32 rejected run writes no accepted `results.json` and writes one `REJECTED.json`.
- [ ] Regression test: the strict accepted-payload contract never substitutes `None` for NaN/Inf.

Acceptance criteria:

- [ ] No accepted result artifact contains `FINAL_OBJECTIVE: null`, `FINAL_DOFS: [null, ...]`, or equivalent masked non-finite values.
- [ ] Downstream validators can determine accepted versus rejected state from the artifact contract without reading console logs.
- [ ] One result-acceptance helper is the only gate; example wrappers do not duplicate logic.

### Wave 2 - Linear-Solve Contract Root Fix

Purpose: fix the adjoint-solve failure at the numerical contract boundary.

Implementation tasks:

- [ ] Split the adjoint linear-solve contract into three explicit solve kinds with one shared status-object shape `{success: bool, residual: float, residual_relative: float, iterations: int}`:
  - [ ] square Hessian solve via `_solve_square_array_system_operator_only` (`optimizer_jax.py:3083-3094, 3296-3300`);
  - [ ] singular/gauge-null least-squares solve (current `_solve_hessian_least_squares_system_with_status` at `optimizer_jax.py:3445-3481`);
  - [ ] packed-factor solve via the LS lane PLU triangular solves (`boozersurface_jax.py:3514-3540, 3700-3706`, `surfaceobjectives_jax.py:3217-3299`).
- [ ] Stop routing a successful direct square operator solve through the normal-equation LS path. Today, `surfaceobjectives_jax.py:3484-3492, 4081` forces `linear_solve_factors=None`, which forces the LS fallback regardless of whether the direct operator already meets the original residual contract. Gate that switch on actual direct-solve failure, not on the absence of stored factors.
- [ ] Pin the singular/gauge-null LS contract per the Wave-0 decision. Per-tier tolerance source: `BackendPolicy.linear_solve_tolerance_floor` / `cap` (`runtime.py:190-201`). Current `float32_smoke` uses floor `1e-6` with cap `None`, and the convergence gate must be the **original-operator** residual, not the κ²-squared normal-equation residual.
- [ ] Replace the operator-GMRES success gate `||residual|| ≤ max(1e-12, 10·tol·||rhs||)` (`optimizer_jax.py:3083-3094`) with `||A x - b|| / max(||b||, eps_runtime) ≤ effective_linear_solve_tolerance(policy, requested_tol)` so the gate uses the same floor/cap helper as Boozer solves. Float64 production lanes keep cap `1e-10`.
- [ ] Add residual and finiteness checks to the packed-factor forward and transpose solve callbacks (`boozersurface_jax.py:3700-3706` and the traceable LS adjoint at `surfaceobjectives_jax.py:3217-3299`), using the same per-tier residual metric.
- [ ] Preserve the fail-closed NaN sentinel (`_traceable_adjoint_gradient_or_nan`) when the selected solve contract fails.

Design constraints:

- [ ] No synthetic gradient fallback.
- [ ] No silent dense CPU substitution.
- [ ] No hidden retry path that changes backend or dtype.
- [ ] No tolerance change that applies to float64 production parity (`parity` tier cap remains 1e-10).
- [ ] No new defensive try/except around the solver entrypoints; status is reported through the shared status object.
- [ ] No κ² conditioning in the float32 lane: do not form `H^T H` or use the corresponding operator-action GMRES on the normal equations.

Tests:

- [ ] Fixed-state unit test for operator-only adjoint solve success/failure reporting under both `parity` and `float32_smoke` tiers.
- [ ] Fixed-state unit test for packed-factor residual-gated failure (finite-but-high-residual solution must fail).
- [ ] Regression test showing the singular-LS contract validates against the original operator, not the transformed normal equation.
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

- [ ] Replace the hardcoded `_as_jax_float64(self.field.x)` cast in `SquaredFluxJAX._gather_field_free_dofs` (`fluxobjective_jax.py:346-348`) with the runtime-policy dtype helper.
- [ ] Replace the hardcoded `jnp.float64` cast on `biotsavart.x` in `QfmSurfaceJAX._coil_set_spec` (`qfmsurface_jax.py:73-76`) with the runtime-policy dtype helper.
- [ ] Rename `_as_explicit_float64`, `_explicit_scalar`, `_ones_like_float64`, and `_zeros_like_float64` in `jax_core/curve_geometry.py:60, 75, 79, 83` to drop the `float64` suffix; they already route through `_as_runtime_array` and honor the runtime dtype, so the names lie.
- [ ] Audit single-stage banana helper casts (`single_stage_banana_example.py`) and classify each as (a) runtime-policy, (b) host/SciPy boundary, or (c) intentional float64 production-only. Record the classification next to each call site.

Tests:

- [ ] Unit test for `SquaredFluxJAX._gather_field_free_dofs` under both `jax_cpu_float32_smoke` and `jax_cpu_parity` policies.
- [ ] Unit test for `QfmSurfaceJAX._coil_set_spec` under both policies.
- [ ] Runtime dtype policy test that verifies representative arrays in the smoke critical path are float32 in `jax_cpu_float32_smoke` and `jax_mps_smoke`, and remain float64 in `jax_cpu_parity` and `jax_gpu_parity`.

Acceptance criteria:

- [ ] Float32 smoke lanes do not upcast through helper or entrypoint paths on the smoke critical path.
- [ ] Float64 production lanes still construct float64 arrays.
- [ ] Renamed helpers compile under the same call sites; no caller change needed beyond the rename.

Out of scope for this wave (tracked in Wave 8): bootstrap JAX profile derivatives, VMEC fieldline diagnostics, VMEC geometry helpers, PM workflow paths, wireframe workflow paths, MHD frozen-state casts. None of these are imported by the banana single-stage example or the non-banana CPU/MPS smoke harnesses (verified by grep against `tests/test_jax_mps_smoke.py`, `tests/test_mps_smoke_dtype.py`, and `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`).

### Wave 4 - API Boundary Cleanup

Purpose: restore clean public import boundaries and remove visible private-module dependencies from examples.

Implementation tasks:

- [ ] Delete `src/simsopt/backend.py`. It is shadowed by the `src/simsopt/backend/` package — Python resolves `import simsopt.backend` to `backend/__init__.py`, so the shim is dead code at runtime and drifts from the SSOT facade. SSOT: one facade in `backend/__init__.py`.
- [ ] Update `src/simsopt/_core/optimizable.py:32` and `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:86` to import `get_tolerance_tier` from `simsopt.backend` instead of `simsopt.backend.runtime`. Apply the same fix to any other public-helper imports that reach into `simsopt.backend.runtime`.
- [ ] Promote `_coil_dofs_gradient_to_derivative` (`surfaceobjectives_jax.py:1931`) to a non-underscore public name and re-export from `simsopt.geo`, then update `single_stage_banana_example.py:126`.
- [ ] Treat `src/simsopt/jax_core/_math_utils.py` as a documented compatibility facade (its module docstring already states "Compatibility facade for backend-owned JAX dtype helpers"). Either rename it to drop the underscore or add an explicit `# public-facade: underscore-prefixed for legacy compatibility` marker so the audit grep does not re-flag it.
- [ ] Remove `_gsco_opposite_candidate_index` from `__all__` in `src/simsopt/solve/wireframe_optimization_jax.py:35` (private name should not be exported).

Tests:

- [ ] Import test: `from simsopt.backend import get_tolerance_tier, raise_if_target_lane_bypass, strict_target_lane_purity, target_lane_purity_active, target_lane_purity_requested` resolves cleanly.
- [ ] Import test runs under an isolated current-repo import path (for example `python -S` with `src` inserted) so ambient editable installs cannot hide backend module/package resolution.
- [ ] Import test: `from simsopt.geo import coil_dofs_gradient_to_derivative` (or chosen public name) resolves.
- [ ] Example import smoke test for single-stage banana entrypoint.
- [ ] Static grep check that no example or production source under `src/` imports `simsopt.backend.runtime` symbols that are already re-exported by `simsopt.backend`.

Acceptance criteria:

- [ ] Public helper imports resolve through the public facade.
- [ ] No example requires a private core symbol for normal operation.
- [ ] No dead facade module exists in the package tree.

### Wave 5 - Transfer Guard and Host Materialization Boundary

Purpose: keep compute/probe paths strict while allowing explicit artifact and MPI materialization boundaries.

Implementation tasks:

- [ ] Wrap the MPI Jacobian materialization at `src/simsopt/solve/mpi_jax.py:70` (`np.asarray(jax.device_get(local_columns))`) in `jax.transfer_guard_device_to_host("allow")`.
- [ ] Replace the per-row `jax.device_get` at `src/simsopt/solve/serial_jax.py:100-102` with a batched materialization at write boundary, also under `jax.transfer_guard_device_to_host("allow")`.
- [ ] Audit `mpi_jax.py:114-127` non-leader worker loop: ensure shutdown handles `command is None` as well as `STOP` to avoid hangs when sentinel propagation differs (e.g., mock comm in tests).
- [ ] Keep compute and gradient paths under strict transfer guard (`disallow` for production parity lanes, default for smoke).
- [ ] Consolidate duplicated artifact materialization helpers: `_jax_artifact_host_array` (`benchmarks/non_banana_example_parity_fixtures.py:35`) and `_artifact_host_value` / `_host_float_array` (`benchmarks/non_banana_example_cpp_jax_cpu_parity.py:344`) into one helper in `benchmarks/run_code_benchmark_common.py` (or a dedicated `benchmarks/_host_io.py`). DRY: one materialization helper.
- [ ] Use direction-specific `jax.transfer_guard_device_to_host("allow")` at host materialization boundaries instead of broad `jax.transfer_guard("allow")`.

Tests:

- [ ] Transfer-guard strict test for compute path (`SquaredFluxJAX.dJ`, `BoozerSurfaceJAX.run_code`) — no D→H transfers during compute.
- [ ] Transfer-guard allowed test for artifact serialization boundary.
- [ ] Transfer-guard allowed test for MPI/serial host-solver boundary if those modules are kept; if `mpi_jax.py` / `serial_jax.py` are out of scope for production, mark them as such and add the test only at the boundary entrypoint.
- [ ] MPS smoke test with `SIMSOPT_JAX_TRANSFER_GUARD=disallow` configured.

Acceptance criteria:

- [ ] JAX transfer-guard logs or errors do not appear from compute paths.
- [ ] Artifact and logging boundaries materialize arrays explicitly and locally.
- [ ] Exactly one artifact-materialization helper exists in the benchmarks tree.
- [ ] Host materialization boundaries are auditable by a single grep on `transfer_guard_device_to_host`.

### Wave 6 - MPS and Float32 Smoke Rerun

Purpose: rerun MPS only after the artifact gate, dtype policy, transfer boundary, and solve-status fixes are in place.

Run order:

- [ ] `tests/test_jax_mps_smoke.py -m mps`
- [ ] `tests/test_mps_smoke_dtype.py -m mps`
- [ ] `tests/test_runtime_dtype_policy.py`
- [ ] `tests/geo/test_boozersurface_jax.py::test_lbfgs_allows_mps_smoke_policy_default_reference_ls_lane`
- [ ] Non-banana parity harness with `SIMSOPT_BACKEND_MODE=jax_mps_smoke`.
- [ ] Banana single-stage CPU float32 smoke, maxiter=7.
- [ ] Banana single-stage MPS float32 smoke, maxiter=7.

Artifact requirements:

- [ ] `backend_mode` recorded.
- [ ] `runtime_dtype` recorded.
- [ ] `host_dtype` recorded.
- [ ] `tolerance_tier` recorded.
- [ ] `maxiter=7` recorded.
- [ ] fixture/input hash recorded.
- [ ] source snapshot recorded.
- [ ] correctness verdict recorded.
- [ ] performance metrics recorded separately.
- [ ] memory metrics recorded separately.

Acceptance criteria:

- [ ] MPS smoke either passes its documented smoke contract or fails with a rejected artifact.
- [ ] MPS smoke is not reported as production parity.
- [ ] No CPU fallback or silent backend substitution occurs.
- [ ] MPS artifact records the `tillahoffmann/jax-mps` (MLX) float32-only constraint as the reason it is not a float64 production lane, citing the jax-mps README rather than the unmaintained Apple `jax-metal` page.

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

- [ ] QFM augmented Lagrangian: decide whether `QfmAugmentedLagrangianInfo.fun` (`jax_core/qfm_solver.py:64`, write site at `:704`) means augmented value or raw QFM value, then add a CPU/analytic oracle test before changing behavior. The current diff flipped semantics from `result.fun` (augmented) to `metrics.qfm_value` (raw) and only added a change-pinning test (`tests/geo/test_qfmsurface_jax.py:475`).
- [ ] Relax-and-split JAX: compare the new `epsilon_RS=1.0e-3` short-circuit (`permanent_magnet_optimization_jax.py:805, 821`) against the previous fixed-iteration loop; either restore behavior or document the API delta with an oracle test against the CPU implementation.
- [ ] Optimizer status code: document `LBFGS_STATUS_NONFINITE = 6` (`optimizer_jax_private/_types.py:15`) as repo-local; add tests around `success`, `status`, and `message` for both the SciPy adapter (`_mark_scipy_result_invalid_state` in `optimizer_jax_reference.py:101`) and the on-device adapter (`_result_converters._mark_nonfinite`). Verify parity between the two — coverage already exists at `tests/geo/test_optimizer_result_converters.py:172, 190, 226`; add gap tests only if behavior diverges.
- [ ] BFGS curvature criterion in QFM (`jax_core/qfm_solver.py:460-464`): the floor was loosened from `sqrt(eps)` to `eps * ||y|| * ||s||`. Add an oracle test against the SciPy BFGS reference trajectory before locking the new floor.
- [ ] `_normalize_scipy_result` status 6 is currently safe vs SciPy upstream (SciPy uses warnflag 0/1/2 only) but is still a private contract; document this in the constant's docstring.
- [ ] Float32 smoke harness: resolve whether gradient entries tagged `diagnostic_only` can still make the fixture verdict fail (`non_banana_example_cpp_jax_cpu_parity.py:596, 615-617, 2740-2745` vs `validation_ladder_contract.py:65-78` `production_parity=False`); encode one SSOT rule.
- [ ] Tautology tests: replace change-pinning tests with CPU, analytic, or committed-fixture oracle tests where the behavior is contract-bearing. Concrete candidates: `tests/geo/test_qfmsurface_jax.py:475`, `tests/solve/test_permanent_magnet_optimization_jax_item28.py:537`.

Implementation tasks (off-critical-path dtype hygiene; deferred from Wave 3):

- [ ] Replace hardcoded float64 in bootstrap JAX profile derivative paths (`src/simsopt/mhd/bootstrap_jax.py:144, 174`).
- [ ] Replace hardcoded float64 in VMEC fieldline diagnostics (`src/simsopt/mhd/vmec_diagnostics_jax.py:60, 61, 64, 74`).
- [ ] Replace hardcoded float64 in VMEC geometry helpers (`src/simsopt/jax_core/vmec_geometry.py:289`).
- [ ] Replace runtime-float64 helper usage in PM workflow paths (`src/simsopt/jax_core/pm_workflow.py:83, 84, 164`) where the value should follow runtime dtype.
- [ ] Replace runtime-float64 helper usage in wireframe workflow paths (`src/simsopt/jax_core/wireframe_workflow.py:345, 346, 547`) where the value should follow runtime dtype.
- [ ] Audit tracing, magnetic-axis, and MHD frozen-state float64 casts before claiming float32 end-to-end coverage outside the banana smoke perimeter.

Cleanup tasks:

- [ ] MPI and serial JAX solvers: if `src/simsopt/solve/mpi_jax.py` / `serial_jax.py` remain in scope, the Wave-5 transfer-boundary and Wave-1 result-gate contracts apply.
- [ ] Root-level debug outputs: remove or gitignore `jax_mem_test.py`, `objective_runtimes_semilogy.png`, `taylor_errors.png`, `test_coil.vtu`, and `.gpd/state.json.bak`.
- [ ] `simsopt/mhd/__init__.py:17` `import jax as _` shadows Python's last-result convention; rename to `_jax` or drop the alias.

Acceptance criteria:

- [ ] Adjacent regressions are either fixed, moved to a separate dated plan, or explicitly declared out of scope before production parity is claimed.
- [ ] No test is accepted as an oracle if it only reasserts the implementation under test.
- [ ] No untracked solver module is used in validation without being added to the artifact/test contract.
- [ ] Off-critical-path dtype hygiene completes after the float32 smoke rerun closes; production parity is not claimed before the hygiene is closed.

## Priority TODO List

- [ ] P0: Split `sanitize_json_payload` into a strict accepted-payload helper and a `sanitize_diagnostic_payload`.
- [ ] P0: Implement one accepted-artifact gate SSOT covering both target and non-target lanes.
- [ ] P0: Add final-result finiteness checks independent of optimizer success, including escalation of transient `nonfinite_step` events to `LBFGS_STATUS_NONFINITE` at termination.
- [ ] P0: Tier-scale the operator-GMRES success gate by `effective_linear_solve_tolerance(policy, requested_tol)`; no change to float64 parity.
- [ ] P0: Stop forcing the LS adjoint to the normal-equation path when the direct operator solve already meets the original-operator residual; gate the switch on actual direct-solve failure.
- [ ] P0: Pin the residual metric `||A x - b|| / max(||b||, eps_runtime)` as the SSOT solve-success criterion for both operator-only and packed-factor solves.
- [ ] P0: Add residual gates to packed-factor LS callbacks (`boozersurface_jax.py:3700-3706`, `surfaceobjectives_jax.py:3217-3299`).
- [ ] P1: Clean runtime dtype policy leaks on the float32 smoke critical path (`fluxobjective_jax.py:346-348`, `qfmsurface_jax.py:73`).
- [ ] P1: Delete the shadowed `src/simsopt/backend.py` shim; the package `backend/__init__.py` is the only facade.
- [ ] P1: Clean public/private API imports in the banana example (`single_stage_banana_example.py:86, 89-94, 126`).
- [ ] P1: Add transfer-guard materialization boundaries at `mpi_jax.py:70` and `serial_jax.py:100-102`.
- [ ] P1: Consolidate duplicated artifact-materialization helpers in `benchmarks/`.
- [ ] P1: Rerun CPU float32 smoke and MPS float32 smoke (Wave 6).
- [ ] P1: Rerun float64 CPU production parity (Wave 7) — the latest CPU parity artifact on this branch is dated 2026-05-18, before the recent dtype-centralization commits.
- [ ] P1: Audit QFM, relax-and-split, and optimizer-status adjacent regressions before production signoff.
- [ ] P2: Rerun CUDA production parity on Perlmutter.
- [ ] P2: Clean root-level debug artifacts.
- [ ] P2: Rename stale `*_float64`-suffixed helpers (`curve_geometry.py:60, 75, 79, 83`) that route through runtime dtype.
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
- [ ] Official-doc constraints above are still current at implementation time (jax-mps for MPS, jax-metal as historical context only).

## Open Decisions

- [ ] Decide whether root debug artifacts should be deleted, moved, or gitignored (Wave 8 cleanup).
- [ ] Decide whether adjacent dirty-tree regressions (QFM augmented Lagrangian semantics, relax-and-split `epsilon_RS`, QFM BFGS curvature floor) are fixed in this plan or split into a separate dated plan after Wave 1.
