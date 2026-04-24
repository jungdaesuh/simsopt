# JAX-Native Remaining Implementation Plan

Date: 2026-04-24

## Context

The original "make banana coils JAX-native" plan is no longer a greenfield
implementation plan. Current branch history and source review show that most
of the native lane already exists:

- `src/simsopt/jax_core/` contains immutable pytree specs, grouped field
  kernels, surface specs, fixed-surface flux specs, and single-stage runtime
  seed specs.
- `src/simsopt/jax_core/field.py` implements coil-axis `jax.shard_map` plus
  `jax.lax.psum` for `coil_groups`.
- `src/simsopt/geo/surfaceobjectives_jax.py` exposes traceable objective
  bundles whose hot path is pure JAX arrays.
- `src/simsopt/geo/boozersurface_jax.py` owns the operator-backed adjoint
  runtime state via `BoozerSurfaceJAX.get_adjoint_runtime_state()`.
- `src/simsopt/geo/optimizer_jax_private/` uses JAX-array optimizer state for
  BFGS, L-BFGS, and line search.

The remaining work is therefore not "port SIMSOPT to JAX." The remaining work
is to formalize the native equivalents of SIMSOPT object concepts, remove the
last script-local/native-boundary seams, and extend multi-device sharding.

## Wave 1 Launch Status

- [x] P1: added `BiotSavartJAX.B_pullback_native(v)` and
  `B_cotangents(v)` as the native grouped cotangent API.
- [x] P1: kept `BiotSavartJAX.B_vjp(v) -> Derivative` and made it delegate
  through the native pullback payload.
- [x] P2: promoted `SingleStageRuntimeSpecBiotSavartJAX` and its spec-backed
  coil/current/curve views into package code.
- [x] P3: scaffolded strict `points_coils` runtime/sharding metadata with
  explicit point and coil device counts.
- [x] P3: kept `backend.should_shard_points()` false for `points_coils` until
  grouped-field 2D execution is implemented.
- [x] P4: closed the Stage 2 ALM target seam so `backend='jax'` rejects
  `optimizer_backend='scipy'` and uses the target optimizer contract for
  `optimizer_backend='ondevice'`.
- [x] P4: audited the single-stage target startup contract; no new code change
  was needed there.
- [x] Code-simplifier pass: scoped to the Wave 1 files.
- [x] Validation: `py_compile`, `git diff --check`, full
  `tests/test_backend.py`, targeted `tests/test_jax_import_smoke.py`, and
  targeted Stage 2/single-stage integration tests pass.

## Architecture Decision

- [x] Do not rewrite upstream SIMSOPT `Optimizable` as a JAX-native class.
- [x] Do not rewrite upstream SIMSOPT `Derivative` as a JAX-native class.
- [x] Keep upstream CPU/C++ `BiotSavart` untouched as the parity oracle.
- [x] Keep public SIMSOPT APIs returning `Derivative` where callers expect it.
- [x] Use immutable specs, explicit DOF vectors, and pytrees as the JAX-native
  replacement for `Optimizable`.
- [x] Use grouped cotangent arrays, flat gradients, and pytrees as the
  JAX-native replacement for `Derivative`.
- [ ] Make the compiled lane consume the native equivalents directly.
- [x] Keep projection back into `Derivative` only at compatibility boundaries.

## Non-Goals

- [ ] Do not add silent CPU fallback paths to JAX target mode.
- [ ] Do not make `Derivative` a pytree keyed by Python `Optimizable` objects.
- [ ] Do not replace `get_adjoint_runtime_state()` with a new gradient
  abstraction.
- [ ] Do not inline new numerical tolerances outside the parity ladder SSOT.
- [ ] Do not refactor upstream SIMSOPT object graph semantics.

## Source Contracts

Official docs checked for this plan:

- JAX JIT and pure-function model:
  `https://docs.jax.dev/en/latest/jit-compilation.html`
- JAX `shard_map` and `psum` collective semantics:
  `https://docs.jax.dev/en/latest/notebooks/shard_map.html`
- SIMSOPT field API:
  `https://simsopt.readthedocs.io/v1.8.3/fields.html`
- NVIDIA NCCL collective semantics:
  `https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html`

Local source contracts:

- Upstream `Optimizable` is a mutable object graph:
  `/Users/suhjungdae/code/opensource/simsopt/src/simsopt/_core/optimizable.py`
- Upstream `Derivative` is keyed by `Optimizable` instances:
  `/Users/suhjungdae/code/opensource/simsopt/src/simsopt/_core/derivative.py`
- Upstream `BiotSavart` defines the additive coil field and VJP contract:
  `/Users/suhjungdae/code/opensource/simsopt/src/simsopt/field/biotsavart.py`
- Public field summation contract lives in:
  `/Users/suhjungdae/code/opensource/simsopt/src/simsopt/field/magneticfield.py`
- Current native specs live in:
  `src/simsopt/jax_core/specs.py`
- Current coil collective field path lives in:
  `src/simsopt/jax_core/field.py`
- Current parity tolerance SSOT lives in:
  `benchmarks/validation_ladder_contract.py`
- Current adjoint SSOT lives in:
  `src/simsopt/geo/boozersurface_jax.py::BoozerSurfaceJAX.get_adjoint_runtime_state`

## P0 - Formalize Native Equivalents

Goal: make the architecture explicit in code and tests.

### Optimizable Equivalent

- [x] Use immutable specs for curve, coil, current, surface, and runtime state.
- [x] Use explicit DOF vectors / pytrees as the ownership boundary.
- [x] Use `OptimizableDofMapSpec` and coil extraction specs for DOF mapping.
- [x] Document the native replacement rule in a repo plan/status doc:
  "compiled JAX lane takes specs and arrays, not mutable Optimizable graphs."
- [ ] Add or update tests that verify native functions accept specs directly
  without reading mutable wrapper state.

### Derivative Equivalent

- [x] Use JAX VJPs internally for Biot-Savart and objective gradients.
- [x] Use grouped coil cotangents internally for field pullbacks.
- [x] Add a supported public native cotangent API on `BiotSavartJAX`.
- [ ] Make internal objective/adjoint paths consume native cotangents where
  they do not need public SIMSOPT compatibility.
- [x] Keep `Derivative` projection as a boundary adapter only.

## P1 - Native Biot-Savart Cotangent API

Goal: expose a JAX-native pullback result before projection into `Derivative`.

Current state:

- `src/simsopt/jax_core/field.py::biot_savart_B_vjp_maybe_collective` returns
  JAX cotangents for `(gammas, gammadashs, currents)`.
- `src/simsopt/field/biotsavart_jax_backend.py::BiotSavartJAX.B_vjp` now
  projects the native pullback payload into `Derivative`.
- `BiotSavartJAX.coil_cotangents_to_derivative(...)` already owns the public
  projection boundary.

Tasks:

- [x] Add `BiotSavartJAX.B_pullback_native(v)` or
  `BiotSavartJAX.B_cotangents(v)`.
- [x] Return a typed native payload:
  grouped cotangents plus corresponding coil index lists.
- [x] Reimplement `BiotSavartJAX.B_vjp(v)` as:
  `B_pullback_native(v)` -> `coil_cotangents_to_derivative(...)`.
- [ ] Route Boozer/objective internals through the native cotangent API when
  the caller does not need a `Derivative`.
- [x] Preserve `B_vjp(v) -> Derivative` for public SIMSOPT compatibility.
- [x] Add parity tests comparing:
  native cotangents -> projected `Derivative` vs current `B_vjp`.
- [ ] Add multi-device tests proving native cotangents still lower through
  the collective path when `SIMSOPT_JAX_SHARDING=coil_groups`.

Files likely touched:

- [ ] `src/simsopt/field/biotsavart_jax_backend.py`
- [ ] `src/simsopt/jax_core/field.py`
- [ ] `src/simsopt/geo/surfaceobjectives_jax.py`
- [ ] `tests/integration/test_stage2_jax.py`
- [ ] `tests/integration/test_single_stage_jax_cpu_reference.py`
- [ ] `tests/subprocess/jax_runtime_cases.py`

## P2 - Promote Runtime-Spec Biot-Savart Adapter

Goal: remove the script-local native adapter from the single-stage example.

Current state:

- `SingleStageRuntimeSpec` is already in `src/simsopt/jax_core/specs.py`.
- `SingleStageRuntimeSpecBiotSavartJAX` now lives in
  `src/simsopt/field/biotsavart_jax_backend.py`.
- The single-stage example imports the packaged adapter.
- Tests import the adapter from package code.

Tasks:

- [x] Move `SingleStageRuntimeSpecBiotSavartJAX` into package code.
- [x] Move `SpecBackedCoil`, `SpecBackedCurve`, and `SpecBackedCurrent` with it.
- [x] Prefer `src/simsopt/field/biotsavart_jax_backend.py` if the class remains
  adapter-like.
- [x] Keep pure spec construction helpers in `src/simsopt/jax_core/field.py`.
- [x] Update the single-stage example to import the packaged class.
- [x] Update tests to import from package code instead of the example module.
- [x] Delete the script-local class after parity.

Files likely touched:

- [ ] `src/simsopt/field/biotsavart_jax_backend.py`
- [ ] `src/simsopt/field/biotsavart_jax.py`
- [ ] `src/simsopt/jax_core/__init__.py`
- [ ] `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
- [ ] `tests/geo/test_single_stage_example.py`

## P3 - Implement `points_coils` 2D Sharding

Goal: extend from coil-axis collectives to point-axis plus coil-axis sharding.

Current state:

- `coil_groups` is registered in `src/simsopt/backend/runtime.py`.
- `coil_groups` uses `jax.shard_map` plus `jax.lax.psum`.
- `points_coils` is registered as a valid sharding strategy.
- `points_coils` requires explicit point and coil device counts and rejects
  product mismatches against the detected JAX device count.
- Point sharding and coil collectives are separate code paths.
- `points_coils` does not yet route grouped-field execution through a 2D
  `shard_map`; that is Wave 2.

Tasks:

- [x] Add `points_coils` to `_VALID_SHARDING_STRATEGIES`.
- [x] Extend `ShardingTuning` with point-axis and coil-axis mesh dimensions.
- [x] Add env/config parsing for point device count and coil device count.
- [x] Build a 2D mesh helper requiring:
  `point_devices * coil_devices == device_count`.
- [ ] Add a grouped-field 2D collective kernel:
  points sharded on point axis, coils sharded on coil axis.
- [ ] Reduce over coil axis with `lax.psum`.
- [ ] Keep output point-sharded after the coil reduction.
- [x] Extend summaries to report:
  `strategy`, `mesh_axes`, `point_axis`, `coil_axis`, `reduced_axis`,
  `field_collective`, and device counts.
- [ ] Add StableHLO lowering tests asserting `all_reduce`.
- [ ] Add parity tests for non-divisible coil counts and mixed quadrature
  groups.

Files likely touched:

- [ ] `src/simsopt/backend/runtime.py`
- [ ] `src/simsopt/jax_core/sharding.py`
- [ ] `src/simsopt/jax_core/field.py`
- [ ] `tests/test_backend.py`
- [ ] `tests/subprocess/jax_runtime_cases.py`

## P4 - Close Host-Driven JAX Target Seams

Goal: make JAX target mode consistently use native/on-device contracts.

Current state:

- Target objective bundles are JAX-native.
- Target optimizer contracts require `optimizer_backend='ondevice'`.
- Stage 2 ALM now resolves its inner optimizer through the same target
  optimizer contract.
- Some startup/parity artifact paths can still construct host-driven objects
  for diagnostics or compatibility.

Tasks:

- [x] Audit Stage 2 JAX target startup for accidental SciPy/reference optimizer
  use.
- [x] Audit single-stage JAX target startup for accidental SciPy/reference
  optimizer use.
- [x] Keep CPU/reference mode explicit and separate.
- [ ] Keep host reporting and artifact writing outside compiled kernels.
- [ ] Remove fallback wording from docs/tests when the path has been deleted.
- [x] Add tests that reject JAX target mode with host/SciPy optimizer contracts.

Files likely touched:

- [ ] `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py`
- [ ] `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
- [ ] `examples/single_stage_optimization/alm_utils.py`
- [ ] `src/simsopt/geo/optimizer_jax.py`
- [ ] `tests/geo/test_single_stage_alm_integration.py`
- [ ] `tests/integration/test_stage2_jax.py`

## P5 - Delete Stale Seams After Coverage

Goal: remove compatibility leftovers only after native parity is proven.

Delete candidates:

- [ ] Script-local runtime-spec Biot-Savart adapter.
- [ ] Duplicate runtime-spec loading helpers.
- [ ] Tests that patch removed fallback helpers.
- [ ] Docs claiming CPU fallback behavior that no longer exists.
- [ ] Dead host pullback helpers that are no longer reachable.

Keep:

- [ ] Public `Derivative` returns.
- [ ] `Optimizable` compatibility wrappers.
- [ ] CPU/reference parity oracle.
- [ ] `BoozerSurfaceJAX.get_adjoint_runtime_state()`.
- [ ] Host reporting and artifact materialization outside compiled kernels.

## Validation Matrix

Use `benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES`.
Do not inline new tolerances.

### Direct Kernel Lane

- [ ] `B`
- [ ] `A`
- [ ] grouped field
- [ ] fixed-surface flux
- [ ] raw Boozer residual

### Derivative-Heavy Lane

- [ ] `dB/dX`
- [x] Biot-Savart native cotangents
- [x] projected `B_vjp`
- [ ] surface coefficient Jacobians
- [ ] Boozer residual derivatives

### Adjoint Lanes

- [ ] `exact_well_conditioned_adjoint`: operator-vs-dense vector parity.
- [ ] `exact_ill_conditioned_adjoint`: residual/failure behavior only.
- [ ] Confirm dense PLU metadata never replaces operator-backed runtime solves.

### Multi-Device Lowering

- [ ] CPU subprocess test with:
  `XLA_FLAGS=--xla_force_host_platform_device_count=4`
- [x] Assert StableHLO text contains `all_reduce`.
- [x] Assert `grouped_field_sharding_summary(...)["field_collective"] is True`.
- [x] Run with `SIMSOPT_JAX_SHARDING=coil_groups`.
- [ ] Run with `SIMSOPT_JAX_SHARDING=points_coils` after P3.

### CUDA Smoke

- [ ] `JAX_PLATFORMS=cuda,cpu`
- [ ] `SIMSOPT_JAX_SHARDING=coil_groups`
- [ ] `SIMSOPT_JAX_SHARDING=points_coils`
- [ ] `NCCL_DEBUG=WARN`
- [ ] Validate parity and active collective summaries.

## Definition of Done

- [ ] JAX target lane has native replacements for `Optimizable` and
  `Derivative` semantics in the compiled path.
- [ ] Public SIMSOPT compatibility remains intact.
- [x] `BiotSavartJAX.B_vjp` still returns `Derivative`, but delegates to a
  native cotangent API.
- [x] Single-stage runtime-spec Biot-Savart adapter is package-owned, not
  script-owned.
- [ ] `coil_groups` and `points_coils` both lower to collective reductions.
- [x] Stage 2 and single-stage JAX target modes reject host optimizer seams.
- [ ] Validation uses the parity ladder SSOT.
- [ ] Stale fallback code/docs/tests are removed only after parity coverage.
