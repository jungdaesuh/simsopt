# Revised Two-PR Plan: Seeded Target-Lane L-BFGS and Boozer Linear-Solve Abstraction

Date: 2026-04-19

Status: Planning document

Scope: single-stage target-lane seeded outer optimization plus exact-path Boozer
linear-solve abstraction in `simsopt-jax`

## Goal

Land two tightly scoped PRs without regressing the current target-lane or exact
Boozer contracts:

1. PR1 adds an optional seeded `initial_value_and_grad` path for the single-stage
   target-lane `lbfgs-ondevice` outer optimizer.
2. PR2 generalizes the Boozer adjoint runtime state from dense `PLU` storage to
   an abstract linear-solve state, then adds matrix-free adjoint / warm-start
   solves behind that seam.

This plan is code-anchored to the current tree and already folds in the
validation refinements required before either PR can merge.

## Non-Goals

- Do not change the public `runtime_bundle["objective"]` semantics.
- Do not mutate the existing `runtime_bundle` dict contract just to thread an
  optimizer seed.
- Do not make matrix-free adjoint solves the default exact-path behavior in the
  same change that introduces the abstraction.
- Do not rewrite the single-stage retry policy beyond the minimal seed-plumbing
  needed for phase-1 first-attempt reuse.

## Confirmed Live Call Graph

### PR1 target-lane seed path

1. `build_target_lane_outer_objectives(...)` at
   `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:5292`
   builds the runtime bundle and exposes the current fused target-lane
   `value_and_grad`.
2. `make_traceable_objective_value_and_grad(...)` at
   `src/simsopt/geo/surfaceobjectives_jax.py:3385` is the current thin alias
   around `make_traceable_objective_runtime_bundle(... )["value_and_grad"]`.
3. `run_single_stage_optimizer(...)` at
   `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:6660`
   routes the target lane through `target_minimize(...)`.
4. `target_minimize(...)` at `src/simsopt/geo/optimizer_jax.py:2275` routes
   `value_and_grad=True` + `method="lbfgs-ondevice"` to
   `_minimize_lbfgs_private_value_and_grad(...)` at line 2338.
5. `_minimize_lbfgs_private_value_and_grad(...)` at
   `src/simsopt/geo/optimizer_jax_private/_lbfgs.py:786` delegates to
   `_minimize_lbfgs_private_impl(...)` at line 306.
6. `_minimize_lbfgs_private_impl(...)` currently evaluates the initial
   `f_0, g_0` unconditionally at line 351 and re-evaluates the final iterate
   unconditionally at line 667.

### PR2 Boozer linear-solve path

Citation convention: `Symbol (def file:line)` for the function/method/class
definition, and explicit `at file:line` annotations for the lines actually
affected by the change. This way every cited line is unambiguously either an
entry point or a target operation.

1. `_BoozerAdjointRuntimeState` (def `src/simsopt/geo/boozersurface_jax.py:180`)
   currently stores a hard-coded dense `plu` field at `:184`.
2. `BoozerSurfaceJAX.get_adjoint_runtime_state(...)` (def
   `src/simsopt/geo/boozersurface_jax.py:1635`) reads `self.res["PLU"]` and
   constructs `_BoozerAdjointRuntimeState(... plu=self.res["PLU"], ...)` at
   `:1654-1658`.
3. `_solve_boozer_adjoint(...)` (def
   `src/simsopt/geo/surfaceobjectives_jax.py:963`) dereferences
   `adjoint_state.plu` at `:972` and dispatches the transposed dense solve at
   `:973`.
4. `_traceable_forward_result(...)` (def
   `src/simsopt/geo/surfaceobjectives_jax.py:2145`) threads `baseline_plu`
   through both the `baseline_case` (`:2164-2174`) and `general_case`
   (`:2176-2233`) branches.
5. `_traceable_objective_gradient_parts(...)` (def
   `src/simsopt/geo/surfaceobjectives_jax.py:2258`) calls
   `_solve_plu_transpose_with_refinement(*solved_plu, dJ_dx)` at `:2304`.
6. `_traceable_predict_warmstart_x(...)` (def
   `src/simsopt/geo/surfaceobjectives_jax.py:2332`) calls
   `_solve_plu_with_refinement(*baseline_plu, -forcing)` at `:2377` for the
   warm-start predictor.
7. `_build_traceable_objective_state(...)` (def
   `src/simsopt/geo/surfaceobjectives_jax.py:2381`) captures the solved
   baseline `PLU` from `booz_jax.res["PLU"]` at `:2456` and threads it into
   the returned runtime state dict (`baseline_plu` field at `:2478`, full dict
   at `:2474-2486`).
8. `run_code_traceable(...)` (def
   `src/simsopt/geo/boozersurface_jax.py:2322`) exact-mode branch at
   `:2337-2391` materializes the dense Jacobian (`:2360-2366`), builds dense
   `PLU` (`:2362-2366`), and sets
   `"success": result["success"] & finite & jacobian_available_jax,` at `:2387`.

## Hard Invariants

- Public `runtime_bundle["objective"]` and `runtime_bundle["value_and_grad"]`
  stay baseline-aware.
  The current forward path at
  `src/simsopt/geo/surfaceobjectives_jax.py:2145-2174` must keep returning the
  exact baseline objective and baseline solve state when `coil_dofs` exactly
  match the cached baseline.
- The seeded optimizer path must be additive, not invasive.
  The new seedable entrypoint must live beside the existing runtime bundle
  contract, not inside it.
- Seed use must be restricted to the current supported lane:
  `value_and_grad=True` with `method="lbfgs-ondevice"`.
- Cache behavior must remain tied to the same baseline solve generation already
  used by `_traceable_runtime_cache_key(...)` at
  `src/simsopt/geo/surfaceobjectives_jax.py:2594`.
- Exact-path matrix-free work must preserve the current dense default until the
  new abstraction and validation land together.

## PR1: Seeded Target-Lane L-BFGS

### Objective

Allow the single-stage target lane to reuse the baseline objective value and
gradient for the first `lbfgs-ondevice` iteration, without changing the public
traceable runtime bundle semantics.

### Required design

- Add an optional `initial_value_and_grad` parameter to
  `target_minimize(...)`,
  `_minimize_lbfgs_private_value_and_grad(...)`, and
  `_minimize_lbfgs_private_impl(...)`.
- Keep the seed presence check at Python time:
  `initial_value_and_grad is not None`.
  Do not add a traced branch to the non-seeded path.
- Introduce a dedicated seeded helper in
  `src/simsopt/geo/surfaceobjectives_jax.py`, for example
  `make_traceable_objective_seeded_value_and_grad(...)`, that returns a
  `NamedTuple` instead of mutating `make_traceable_objective_runtime_bundle()`.
- That `NamedTuple` should expose:
  - `value_and_grad`: the pure-JAX callable used by the optimizer
  - `optimizer_initial_value_and_grad`: the cached `(baseline_value, baseline_grad)`
    seed for the baseline coil state
- Keep the public runtime bundle unchanged so existing caches, tests, and
  downstream callers continue to consume the same dict keys.

### Runtime-bundle work

1. Build the seed from the same solved baseline state already captured by
   `_build_traceable_objective_state(...)`.
2. Compute the baseline gradient from the same compiled implicit-gradient path
   used by the optimizer, not a separate host-only shortcut.
3. Materialize that baseline gradient at bundle-build time under
   `with jax.transfer_guard("allow"):` to match the existing host-boundary
   pattern used by `_ensure_traceable_runtime_host_wrappers(...)`.
4. Introduce a dedicated general-only forward helper for the seeded optimizer
   path so the seeded lowering does not accidentally retain the public
   baseline-aware `same_coils` branch.

### Optimizer work

1. Reuse the seed for the initial `f_0, g_0` in
   `_minimize_lbfgs_private_impl(...)` instead of always calling
   `_coerce_value_and_grad_result(...)` at line 351.
2. Preserve the current unconditional final re-evaluation semantics for the
   general path, but allow the seeded path to reuse the initial seed only when
   `state.k == 0`.
3. The reuse-vs-reeval branch inside the seeded variant must return identical
   pytree structure, dtypes, and shapes on both sides of the `lax.cond`.
   The safe implementation is to extract a shared coercion helper used by both
   the cached `(value, grad)` reuse path and the live reevaluation path.

### Single-stage wiring

1. Extend `run_single_stage_optimizer(...)` so the target-lane
   `target_minimize_kwargs` assembled at
   `single_stage_banana_example.py:6704-6716` can thread
   `initial_value_and_grad=...` when present.
2. Add an explicit one-shot
   `optimizer_initial_value_and_grad: Optional[tuple[value, grad]] = None`
   parameter to `run_single_stage_optimizer(...)`.
   This is the only supported path from the seeded helper into the target-lane
   optimizer call chain.
   That ownership must cover every direct phase-1 target-lane caller, including
   the scaled diagnostic path in
   `build_target_lane_scaled_phase1_diagnosis(...)` at
   `single_stage_banana_example.py:5698-5710`.
3. Add the same explicit one-shot parameter to
   `run_single_stage_target_lane_optimizer_with_retries(...)`, pass it to the
   first `run_single_stage_optimizer(...)` call at lines 6802-6816, and clear
   it before every retry call at lines 6863-6877.
4. Build that one-shot seed from the new seeded helper only for the phase-1
   fused `value_and_grad` lane.
5. Only pass that seed when all of the following are true:
   - the lane is target/ondevice
   - the objective is using the fused `value_and_grad` path
   - the run is the phase-1 outer optimization attempt
6. Extend `run_single_stage_target_lane_optimizer_with_retries(...)` so the seed
   is consumed on the first phase-1 attempt only and dropped on all retries.

### PR1 merge gates from validation

- `lax.cond` structural equality is tested explicitly.
- Seed materialization is strict-transfer-safe because it is guarded by
  `jax.transfer_guard("allow")`.
- Cache invariance is documented and tested:
  repeated bundle lookup with unchanged solve generation returns the same cached
  `optimizer_initial_value_and_grad` object identity.
- The no-seed path still lowers without carrying the seeded branch.
- The seeded path has an automated lowering regression, not a manual
  `JAX_LOG_COMPILES=1` note.

### PR1 targeted tests

- `tests/geo/test_boozersurface_jax_private.py`
  - seed reuse for the initial evaluation
  - seed reuse only when `state.k == 0`
  - `lax.cond` branch structural-equality regression
- `tests/geo/test_surface_objectives_jax.py`
  - seeded helper leaves the public runtime bundle unchanged
  - cached seeded helper / seed object identity is stable across repeated bundle
    construction
  - seed build is strict-transfer-safe
- `tests/geo/test_single_stage_example.py`
  - one-shot `optimizer_initial_value_and_grad` parameter is threaded from the
    phase-1 call site into `run_single_stage_optimizer(...)`
  - the scaled phase-1 diagnosis path at
    `build_target_lane_scaled_phase1_diagnosis(...)` also receives the seed
    when it is running the fused target-lane `value_and_grad` path
  - `run_single_stage_optimizer(...)` threads the seed via
    `target_minimize_kwargs`
  - `run_single_stage_target_lane_optimizer_with_retries(...)` passes the seed
    only on `phase == "phase1"` and only on the first attempt
- `tests/integration/test_single_stage_jax_cpu_reference.py`
  - lowered seeded helper HLO / metadata proves the seeded path is using the
    general-only forward entrypoint and is not relying on the public
    baseline-aware forward branch

## PR2: Boozer Linear-Solve Abstraction and Matrix-Free Adjoint Mode

### Objective

Abstract exact-path adjoint and warm-start solves away from dense `PLU`, then
add a matrix-free implementation that routes through the same runtime seam.

### Required sequencing

1. Generalize `_BoozerAdjointRuntimeState` first.
   Replace the hard-coded `plu` field with an abstract linear-solve state that
   can represent either dense `PLU` or matrix-free operator-backed solves.
   In the same PR step, migrate the current in-tree readers that dereference
   `adjoint_state.plu` directly:
   - `IotasJAX.compute(...)` (def `surfaceobjectives_jax.py:1666`) reads
     `adjoint_state.plu[1]` at `:1677` and dispatches `_solve_boozer_adjoint`
     at `:1684`.
   - `NonQuasiSymmetricRatioJAX.compute(...)` (def
     `surfaceobjectives_jax.py:1801`) reads `adjoint_state.plu[1].shape[0]` at
     `:1814` and dispatches `_solve_boozer_adjoint` at `:1817`.
   - `compute_standard_surface_objective_gradients(...)` (def
     `surfaceobjectives_jax.py:1828`) reads `adjoint_state.plu[1].dtype` at
     `:1883` and `adjoint_state.plu[1].shape[0]` at `:1884`.
   If a temporary compatibility shim is needed for dense mode, keep it
   explicitly dense-only and do not let the matrix-free path depend on it.
2. Rename the traceable runtime plumbing from `baseline_plu` / `solved_plu` to
   neutral names such as `baseline_linear_solve_state` /
   `solved_linear_solve_state` while preserving dense behavior.
3. Only after those renames land should the matrix-free implementation be wired
   in as an alternative exact-path solve state.

### Solver abstraction work

- Route the adjoint path through `_solve_boozer_adjoint(...)` instead of calling
  `_solve_plu_transpose_with_refinement(...)` directly inside
  `_traceable_objective_gradient_parts(...)`.
- Add a symmetric `_solve_boozer_forward(...)` helper and route the warm-start
  predictor through it instead of calling `_solve_plu_with_refinement(...)`
  directly inside `_traceable_predict_warmstart_x(...)` (def
  `surfaceobjectives_jax.py:2332`) at `:2377`.
- Keep the dense implementation backed by the current iterative-refinement `PLU`
  solves.
- Add a matrix-free implementation backed by the exact residual Jacobian
  operator and GMRES only after the abstraction exists.

### Exact runtime-state work

1. Split the exact-path status in `run_code_traceable(...)` into:
   - `primal_success`: finite primal solve / objective viability
   - `adjoint_linear_solve_available`: whether the returned exact solve state
     can support implicit-gradient evaluation
2. Update the current exact return payload so matrix-free mode does not
   accidentally report `success=False` just because dense `jacobian` / `PLU`
   are absent.
3. Update downstream consumers so they stop assuming
   `result["success"] => dense PLU exists`.
   The current consumer sites include:
   - `_build_traceable_objective_compiled_bundle_from_state(...)` (def
     `surfaceobjectives_jax.py:2489`), value/grad gating inside
     `_value_and_grad_for(...)` at `:2543-2559` (`jax.lax.cond(result["success"], ...)`
     at `:2553-2558`).
   - `_make_traceable_objective_from_compiled_bundle(...)` (def
     `surfaceobjectives_jax.py:2818`), backward-path gating inside `f_bwd`
     (def `:2839`) at `:2848` (`jax.lax.cond(success, _success, _failure, ...)`).
4. Forward-only objective viability should key off `primal_success`, while
   implicit-gradient paths should key off
   `primal_success & adjoint_linear_solve_available`.
   The plan must keep those two notions separate instead of reusing one vague
   `success` flag.
5. Preserve the existing dense default return behavior so current callers that
   expect dense `PLU` continue to work until they opt into the new mode.

### Matrix-free implementation notes

- Reuse the current exact residual callable and operator construction instead of
  introducing a second exact residual definition.
- Keep the exact dense Jacobian materialization policy available because the
  current compile-graph audit already validated that changing the default
  success / `PLU` contract would be user-visible.
- Use `lineax.JacobianLinearOperator(..., jac="bwd")` plus GMRES only behind the
  new abstract solve-state seam.

### PR2 merge gates from validation

- Warm-start forward solves are covered by the same abstraction as adjoint
  solves.
- The success-flag split lands in the same PR step as the new exact runtime
  state, not later.
- Dense mode remains the default and continues to satisfy the current
  `jacobian` / `PLU`-available tests.
- A standalone `lineax` parity test proves that
  `JacobianLinearOperator(..., jac="bwd").T` matches the corresponding
  `jax.vjp` pullback for a small fixture before the production path depends on
  it.

### PR2 targeted tests

- `tests/geo/test_surface_objectives_jax.py`
  - `_solve_boozer_adjoint(...)` dispatches through the abstract solve state
  - `_solve_boozer_forward(...)` dispatches through the abstract solve state
  - warm-start predictor works with both dense and abstract solve-state inputs
  - current objective-wrapper readers no longer depend on `adjoint_state.plu`
    shape/dtype access
- `tests/geo/test_boozersurface_jax.py`
  - dense exact mode still reports success and dense `PLU` exactly as before
  - matrix-free exact mode reports primal success plus adjoint-state
    availability without forcing dense `jacobian` materialization
  - dense-only compatibility shim, if kept temporarily, is covered explicitly
- `tests/geo/test_boozersurface_jax_private.py`
  - standalone `lineax` operator-transpose parity test against `jax.vjp`
- `tests/integration/test_single_stage_jax_cpu_reference.py`
  - exact-path forward-only consumers accept `primal_success` without requiring
    dense `PLU`
  - exact-path fused value/gradient consumers gate on
    `adjoint_linear_solve_available`
  - exact-path objective / gradient consumers keep working through the renamed
    solve-state fields

## Suggested Landing Order

### PR1

1. Add the optimizer seed parameter plumbing.
2. Add the seeded runtime-bundle helper and its `NamedTuple`.
3. Build the baseline seed under explicit transfer-guard allowance.
4. Thread the seed through the single-stage phase-1 first attempt only.
5. Add structural-equality, cache-invariance, and lowering tests.

### PR2

1. Generalize `_BoozerAdjointRuntimeState` and rename solve-state plumbing.
2. Add `_solve_boozer_forward(...)` and route the warm-start predictor through
   it.
3. Route adjoint consumers through `_solve_boozer_adjoint(...)`.
4. Add exact-path primal-success vs adjoint-state-availability split.
5. Add the standalone `lineax` transpose parity test.
6. Add the matrix-free implementation behind the new abstraction.

## Exit Criteria

This plan is complete only when all of the following are true:

- PR1 preserves the public runtime-bundle contract and adds a separate seeded
  helper with automated lowering coverage.
- PR1 phase-1 first-attempt seed reuse is wired into the single-stage retry
  wrapper exactly once and nowhere else.
- PR2 covers both transposed adjoint solves and non-transposed warm-start solves
  through the same abstraction.
- PR2 does not silently demote exact-path success to failure just because dense
  `PLU` is absent.
- Dense default behavior remains stable until matrix-free exact mode is
  explicitly selected and tested.
