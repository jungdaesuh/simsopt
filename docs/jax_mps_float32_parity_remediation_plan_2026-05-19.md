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

- Float32 single-stage target-lane gradients fail closed with NaN sentinels when the adjoint solve fails.
- The target LS Boozer path intentionally returns no dense linear-solve factors and routes adjoint solves through operator-backed linear systems.
- The current least-squares adjoint path can solve normal equations, which squares conditioning and is not a stable float32 contract.
- The CPU float32 smoke rerun now marks the optimizer result as failed when `fun`, `jac`, or `x` is non-finite.
- The old MPS smoke artifact predates that fix and contains a false optimizer success with non-finite `jac` and `x`.
- The packed-PLU/shared LS runtime callback only checks finite solution entries and does not check residual quality.
- `sanitize_json_payload` maps NaN/Inf floats to `None`, which is acceptable only for rejected diagnostic artifacts, not accepted result artifacts.
- The target-lane final metrics gate only checks `optimizer_success is False`; it does not independently validate final result finiteness.
- The non-target-lane metrics path lacks the same final optimizer-result gate.
- MPS policy is float32 smoke by design: `runtime_dtype=float32`, `requires_x64=False`, `tolerance_tier=float32_smoke`.
- The backend facade skew is real but narrower than reported: `get_tolerance_tier` is already exported; the missing symbols are the target-lane purity helpers.
- Several JAX paths still force float64 and therefore do not fully exercise float32 runtime policy.

## Official Documentation Constraints

The implementation must respect these upstream contracts:

- JAX default dtype behavior is controlled by the X64 flag; 64-bit dtypes are not a local per-call assumption. Reference: [JAX default dtypes and the X64 flag](https://docs.jax.dev/en/latest/default_dtypes.html).
- JAX transfer guard treats explicit `jax.device_get()` and `jax.device_put()` as transfers that can be logged or disallowed by direction-specific settings. Reference: [JAX transfer guard](https://docs.jax.dev/en/latest/transfer_guard.html).
- `jax.device_get()` is the explicit host materialization API. Reference: [jax.device_get](https://docs.jax.dev/en/latest/_autosummary/jax.device_get.html).
- Apple documents the JAX Metal plugin as experimental and lists `np.float64`, `np.complex64`, and `np.complex128` as unsupported data types. Reference: [Apple Accelerated JAX on Mac](https://developer.apple.com/metal/jax/).
- SciPy `OptimizeResult.status` is solver-specific and `message` is the explanatory contract. Reference: [SciPy OptimizeResult](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html).
- SciPy's low-level L-BFGS-B interface reports `warnflag` and `task`; do not present this repo's private status code `6` as a SciPy-reserved universal code. Reference: [SciPy fmin_l_bfgs_b](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html).

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

- [ ] Choose the exact non-normal-equation solver contract for singular/gauge-null LS adjoints.
- [ ] Choose the residual metric and tolerance source for packed-factor solve success.
- [ ] Decide whether rejected-run artifacts remain in the main run directory with a rejection marker or move under `diagnostics/`.
- [ ] Decide whether float32 smoke gradients are diagnostic-only in every harness or gate a separate smoke contract.
- [ ] Decide whether a formal FP32 production parity contract will be created later for MPS.

## Execution Plan

### Wave 0 - Pin Contracts Before Fixes

- [ ] Add this plan to the remediation tracking checklist.
- [ ] Record the exact current source snapshot, branch, and dirty-file list before implementation starts.
- [ ] Record the existing failing CPU float32 smoke artifact path and old MPS smoke artifact path as baseline evidence.
- [ ] Resolve all blocking decisions above before touching linear-solve or artifact-gate code.
- [ ] Record the exact official-doc references used for dtype, transfer, MPS, and SciPy optimizer behavior.
- [ ] Inventory root-level debug artifacts and decide delete, move, or gitignore before cleanup.

Acceptance criteria:

- [ ] Baseline evidence is listed in the implementation notes.
- [ ] The production parity contract is not changed.
- [ ] MPS is still documented as float32 smoke, not production parity.
- [ ] The implementation order cannot start Wave 2 before the solver contract is chosen.

### Wave 1 - Accepted Artifact Gate SSOT

Purpose: prevent any accepted artifact from masking non-finite optimizer or reporting state.

Implementation tasks:

- [ ] Introduce one result-acceptance helper for single-stage artifacts.
- [ ] Gate both target-lane and non-target-lane final reporting through that helper.
- [ ] Require accepted artifacts to prove:
  - [ ] optimizer success is true;
  - [ ] optimizer status is accepted for the selected optimizer contract;
  - [ ] objective value is finite;
  - [ ] final DOFs are finite;
  - [ ] final gradient is finite when the lane contract requires a gradient;
  - [ ] reporting metrics are finite where the artifact schema declares them required;
  - [ ] backend mode, dtype policy, tolerance tier, and maxiter are recorded.
- [ ] Reject accepted `results.json` creation before JSON sanitization can replace NaN/Inf with `None`.
- [ ] Keep diagnostic JSON allowed to contain sanitized `None`, but label it as rejected or diagnostic.
- [ ] Add a rejection metadata artifact if side artifacts continue to be written after failure.
- [ ] Remove duplicate gate logic from example wrapper sites.

Tests:

- [ ] Unit test: target lane with `optimizer_success=False` refuses `results.json`.
- [ ] Unit test: target lane with `optimizer_success=True` but non-finite final gradient refuses `results.json`.
- [ ] Unit test: non-target lane with failed optimizer refuses `results.json`.
- [ ] Unit test: diagnostic artifact can be written for rejected run but is not accepted as final result.
- [ ] Integration smoke: CPU float32 rejected run writes no accepted `results.json`.
- [ ] Regression test: JSON sanitization is never used to make an accepted result payload finite.

Acceptance criteria:

- [ ] No accepted result artifact contains `FINAL_OBJECTIVE: null`, `FINAL_DOFS: [null, ...]`, or equivalent masked non-finite values.
- [ ] Downstream validators can determine accepted versus rejected state from the artifact contract without reading console logs.

### Wave 2 - Linear-Solve Contract Root Fix

Purpose: fix the adjoint-solve failure at the numerical contract boundary.

Implementation tasks:

- [ ] Split the adjoint linear-solve contract into explicit solve kinds:
  - [ ] square Hessian solve;
  - [ ] singular/gauge-null least-squares solve;
  - [ ] packed-factor solve.
- [ ] Stop routing a successful direct square operator solve into a normal-equation least-squares path when the original residual contract already passes.
- [ ] For singular/gauge-null least-squares, implement one documented solver contract that does not blindly square conditioning.
- [ ] Add residual and finiteness checks to packed-factor forward and transpose solve callbacks.
- [ ] Use the same success-status object shape for operator-only and packed-factor solves.
- [ ] Preserve the fail-closed NaN sentinel when the selected solve contract fails.
- [ ] Keep float32 smoke tolerances separate from float64 production tolerances.

Design constraints:

- [ ] No synthetic gradient fallback.
- [ ] No silent dense CPU substitution.
- [ ] No hidden retry path that changes backend or dtype.
- [ ] No tolerance change that applies to float64 production parity.

Tests:

- [ ] Fixed-state unit test for operator-only adjoint solve success/failure reporting.
- [ ] Fixed-state unit test for packed-factor residual-gated failure.
- [ ] Regression test showing finite-but-high-residual packed-factor solutions fail.
- [ ] Regression test proving transformed/preconditioned solves pass the original residual gate.
- [ ] CPU float32 smoke gradient diagnosis shows solve status and residual details.
- [ ] Float64 CPU parity gradient checks remain within existing tolerance.

Acceptance criteria:

- [ ] A failed adjoint solve still produces a failed gradient status, not an accepted fallback.
- [ ] A passed adjoint solve has finite solution and residual within the lane-specific contract.
- [ ] Float64 production tolerances are unchanged.

### Wave 3 - Runtime Dtype Policy Cleanup

Purpose: make float32 smoke actually exercise float32 paths end to end.

Implementation tasks:

- [ ] Replace hardcoded float64 casts in `SquaredFluxJAX._gather_field_free_dofs`.
- [ ] Replace hardcoded float64 casts in `QfmSurfaceJAX._coil_set_spec`.
- [ ] Replace hardcoded float64 casts in bootstrap JAX profile derivative paths.
- [ ] Replace hardcoded float64 casts in VMEC fieldline diagnostics.
- [ ] Replace hardcoded float64 casts in VMEC geometry helpers.
- [ ] Replace runtime-float64 helper usage in PM workflow paths where the value should follow runtime dtype.
- [ ] Replace runtime-float64 helper usage in wireframe workflow paths where the value should follow runtime dtype.
- [ ] Rename stale helper names that imply float64 while routing through runtime dtype.
- [ ] Audit single-stage banana helper casts and classify each as runtime-policy, host/SciPy boundary, or intentional float64 production-only.
- [ ] Audit tracing, magnetic-axis, and MHD frozen-state float64 casts before claiming float32 end-to-end coverage.

Tests:

- [ ] Unit test for each touched module under float32 smoke policy.
- [ ] Unit test for each touched module under float64 parity policy.
- [ ] Runtime dtype policy test that verifies representative arrays are float32 in `jax_cpu_float32_smoke` and `jax_mps_smoke`.
- [ ] Runtime dtype policy test that verifies representative arrays remain float64 in CPU/CUDA parity modes.

Acceptance criteria:

- [ ] Float32 smoke lanes do not upcast through helper or entrypoint paths unless a documented interface requires host float64.
- [ ] Float64 production lanes still construct float64 arrays.
- [ ] Remaining hardcoded float64 paths are listed with an owner and a reason.

### Wave 4 - API Boundary Cleanup

Purpose: restore clean public import boundaries and remove visible private-module dependencies from examples.

Implementation tasks:

- [ ] Patch `src/simsopt/backend.py` to export the missing target-lane purity helpers already exported by `src/simsopt/backend/__init__.py`.
- [ ] Update imports that reach into `simsopt.backend.runtime` for public helpers to use `simsopt.backend`.
- [ ] Replace direct example dependency on `_coil_dofs_gradient_to_derivative` with a public helper or move the helper to a non-private API location.
- [ ] Review `_math_utils` imports separately; do not classify them as private API solely because the module name starts with an underscore if it is the established compatibility facade.
- [ ] Remove private symbols from public `__all__` surfaces unless they are intentionally public.

Tests:

- [ ] Import test for `simsopt.backend` facade parity.
- [ ] Example import smoke test for single-stage banana entrypoint.
- [ ] Static grep check for new direct imports from `simsopt.backend.runtime` outside facade modules.

Acceptance criteria:

- [ ] Public helper imports resolve through the public facade.
- [ ] No example requires a private core symbol for normal operation.

### Wave 5 - Transfer Guard and Host Materialization Boundary

Purpose: keep compute/probe paths strict while allowing explicit artifact and MPI materialization boundaries.

Implementation tasks:

- [ ] Identify every intentional device-to-host materialization site.
- [ ] Wrap only artifact serialization, logging, MPI gather, and SciPy host-solver boundaries with direction-specific transfer allowance.
- [ ] Keep compute and gradient paths under strict transfer guard.
- [ ] Move repeated artifact materialization helpers to one SSOT utility.
- [ ] Avoid per-row `device_get` in logging when a batch materialization is possible.
- [ ] Use direction-specific `transfer_guard_device_to_host("allow")` at host materialization boundaries instead of broad `transfer_guard("allow")`.

Tests:

- [ ] Transfer-guard strict test for compute path.
- [ ] Transfer-guard allowed test for artifact serialization boundary.
- [ ] Transfer-guard allowed test for MPI/serial host-solver boundary if those modules are kept.
- [ ] MPS smoke test with transfer guard configured.

Acceptance criteria:

- [ ] JAX transfer-guard logs or errors do not appear from compute paths.
- [ ] Artifact and logging boundaries materialize arrays explicitly and locally.
- [ ] Host materialization boundaries are small enough to audit by grep.

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
- [ ] MPS artifact records Apple/JAX Metal float64 limitation as the reason it is not a float64 production lane.

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

### Wave 8 - Adjacent Dirty-Tree Regression Audit

Purpose: track validated regressions and stale-code risks that are not solved by the MPS smoke rerun but can invalidate downstream production claims.

Implementation tasks:

- [ ] QFM augmented Lagrangian: decide whether `QfmAugmentedLagrangianInfo.fun` means augmented value or raw QFM value, then add a CPU/analytic oracle test before changing behavior.
- [ ] Relax-and-split JAX: compare the new `epsilon_RS` short-circuit semantics against the previous fixed-iteration loop and CPU behavior; either restore behavior or document and test the API delta.
- [ ] Optimizer status code: keep the repo-local non-finite status constant explicit and do not describe it as SciPy-reserved; add tests around `success`, `status`, and `message`.
- [ ] Private on-device optimizer lane: add the same fail-closed non-finite test coverage that exists for the SciPy adapter path.
- [ ] Float32 smoke harness: resolve whether gradient entries tagged `diagnostic_only` can still make the fixture verdict fail; encode one SSOT rule.
- [ ] Tautology tests: replace change-pinning tests with CPU, analytic, or committed-fixture oracle tests where the behavior is contract-bearing.
- [ ] MPI and serial JAX solvers: if these untracked modules remain in scope, apply the same transfer-boundary and result-gate contract as the single-stage artifacts.
- [ ] Root-level debug outputs: remove, move, or gitignore `jax_mem_test.py`, `objective_runtimes_semilogy.png`, `taylor_errors.png`, `test_coil.vtu`, and `.gpd/state.json.bak`.

Acceptance criteria:

- [ ] Adjacent regressions are either fixed, moved to a separate dated plan, or explicitly declared out of scope before production parity is claimed.
- [ ] No test is accepted as an oracle if it only reasserts the implementation under test.
- [ ] No untracked solver module is used in validation without being added to the artifact/test contract.

## Priority TODO List

- [ ] P0: Implement accepted artifact gate SSOT.
- [ ] P0: Gate non-target-lane artifact writes.
- [ ] P0: Add final-result finiteness checks independent of optimizer success.
- [ ] P0: Add residual gates to packed-factor LS callbacks.
- [ ] P0: Separate direct operator solve success from normal-equation least-squares solve selection.
- [ ] P1: Clean runtime dtype policy leaks.
- [ ] P1: Fix backend facade skew.
- [ ] P1: Clean public/private API imports in the banana example.
- [ ] P1: Add transfer-guard materialization boundaries.
- [ ] P1: Rerun CPU float32 smoke and MPS float32 smoke.
- [ ] P1: Rerun float64 CPU production parity.
- [ ] P1: Audit QFM, relax-and-split, and optimizer-status adjacent regressions before production signoff.
- [ ] P2: Rerun CUDA production parity.
- [ ] P2: Clean root-level debug artifacts.
- [ ] P2: Rename stale helper names that imply float64 after runtime-dtype conversion.

## Review Checklist

- [ ] No fallbacks were added.
- [ ] No defensive try/except wrappers were added.
- [ ] No production tolerance was loosened.
- [ ] No accepted artifact masks NaN/Inf as null.
- [ ] No MPS artifact is labeled production parity.
- [ ] No performance result is used to waive correctness.
- [ ] Public imports use public facades.
- [ ] Dtype follows backend runtime policy.
- [ ] Transfer guard is strict outside explicit materialization boundaries.
- [ ] Tests include both float32 smoke and float64 parity paths.
- [ ] Solver changes are validated against original residuals, not only transformed systems.
- [ ] Physics outputs are checked at fixed state before endpoint interpretation.
- [ ] Tautology tests are not counted as production proof.
- [ ] Official-doc constraints above are still current at implementation time.

## Open Decisions

- [ ] Decide whether root debug artifacts should be deleted, moved, or gitignored.
- [ ] Decide whether adjacent dirty-tree regressions will be fixed in this plan or split into a separate dated plan after Wave 1.
