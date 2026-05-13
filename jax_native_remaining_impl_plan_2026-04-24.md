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

## 2026-05-13 Full Remainder Update

This section updates the plan after the CPU-only JAX-port closeout in
`.artifacts/jax_port_goal/REPORT.md`. That closeout completed the default
active scope (`P0`-`P2`) under `active_scope_profile=port_closure`, but it did
not implement the future-scope inventory (`P3`-`P5`) and it intentionally left
items `14`, `15`, and `16` as dependency-blocked.

Items `1`-`11`, `13`, and `17` are closeout-complete in
`.artifacts/jax_port_goal/REPORT.md`. Item `12` is complete except for the
documented `12-circularcoil` sub-item. The full implementation remainder is the
union of:

- [ ] `12-circularcoil` (ready, blocked only by implementation): JAX-native
  complete elliptic-integral support and the `CircularCoil` kernel/wrapper.
- [ ] Item `14` (blocked-by: Boozer field, surface classifier, event-time
  tolerance lane): JAX-native tracing RK path.
- [ ] Item `15` (blocked-by: `12-circularcoil` and `InterpolatedFieldJAX`
  wrapper spec): complete the remaining `CircularCoil` and `InterpolatedField`
  public wrappers.
- [ ] Item `16` (blocked-by: item `14`): public `field/tracing.py` JAX wrapper
  routing.
- [ ] Items `18`-`23` (ready after individual state rows): prompt `P3`
  geometry / optimizer / sampler / scalar potential lanes.
- [ ] Items `24`-`31` (ready after individual state rows): prompt `P4`
  permanent-magnet and wireframe lanes.
- [ ] Items `32`-`33` (execution-prerequisite for item `14` despite prompt tier
  `P5`): Boozer radial interpolant and Boozer-magnetic-field lanes.
- [ ] Existing native-sharding remainder (ready, atomic change-set required):
  finish `points_coils` 2D collective lowering and its CPU multi-device proof.

### Scope Activation Rule

- [ ] Before implementation, update `.artifacts/jax_port_goal/state.json` or a
  successor state file so `active_scope` explicitly includes the tiers being
  worked. Do not rely on the old closeout's skipped `P3`-`P5` aggregate row as
  execution state.
- [ ] Split `18-33` into individual state rows before coding those items. Each
  row needs the same evidence fields used by completed `P0`-`P2` work:
  `source_audit`, `upstream_oracle`, `oracle_contract`,
  `jax_transform_plan`, `math_physics_invariants`, `coverage_matrix`,
  `red_evidence`, `parity_test`, `transfer_guard_test`, `bench_artifact`, and
  `cuda_proof.status`.
- [ ] Keep CUDA proof `not_claimed` unless the user explicitly starts a
  `cuda_perf_release` run. CPU JAX full implementation is a valid next target;
  CUDA release proof is a separate profile.

### 2026-05-13 Review Corrections

- Base HEAD reviewed for this plan update:
  `0489cef28278415f7e933edd7434bddc1d9e8f00`.
- `points_coils` remains a future runtime-plus-kernel change, not a registered
  sharding strategy.
- The JAX target optimizer contract is `ondevice` for production/default
  execution, with explicit `scipy-jax` and `scipy-jax-fullgraph` parity/control
  lanes.
- Official source links were refreshed to current JAX, SIMSOPT, CUDA, and NCCL
  documentation.

### Runtime Version Note

- The repo-local interpreter path and conda env name remain historical:
  `.conda/jax-0.9.2/bin/python` / `jax-0.9.2`.
- The checked local environment currently imports `jax==0.10.0` and
  `jaxlib==0.10.0`; fresh environment resolution is governed by
  `pyproject.toml`.
- The `12-circularcoil` blocker was validated against JAX/JAXLIB `0.10.0`:
  `jax.scipy.special` still does not expose `ellipk` / `ellipe`.
- Do not raise the `pyproject.toml` lower bound above `jax>=0.9.2,<1` unless a
  future implementation lands a real `0.10`-only API dependency; record the
  actual imported versions in every validation artifact.

### Cross-Cutting Constraints

- No silent fallback.
- No broad `try/except`.
- No host callbacks inside production compiled correctness paths. Diagnostic
  callbacks are allowed only in explicitly named diagnostic/probe modes and
  must not become correctness or fallback mechanisms.
- No dynamic imports in product code.
- No inlined tolerance literals outside the parity ladder SSOT.

### Concurrent Tracks

- Current branch: `gpu-purity-stage2-20260405`.
- Strict CUDA / transfer-contract hardening is concurrent evidence work, not a
  prerequisite for CPU-only full implementation.
- CUDA release proof remains `not_claimed` in this plan unless the user
  explicitly authorizes a `cuda_perf_release` profile.

### Dependency Graph

The full remainder should not be executed strictly by item number. Several
lower-numbered items are downstream of future-scope prerequisites. Prompt tier
labels (`P3`/`P4`/`P5`) record the original prompt inventory order, not strict
execution order; Wave R2 must precede Wave R3 even though its items are prompt
tier `P5`.

| Gate | Unlocks | Tier Cost Rough | Requirement |
| --- | --- | --- | --- |
| Elliptic helper | `12-circularcoil`, item `15` `CircularCoil` wrapper | M | Implement JAX-native complete elliptic integrals in `src/simsopt/jax_core/_elliptic.py` with SciPy parity. |
| InterpolatedField wrapper spec | Item `15`; part of tracing surface validation | L | Build a JAX wrapper-level spec over item-13 `regular_grid_interp`, including cylindrical coordinates, `nfp`, `stellsym`, skip masks, and out-of-domain semantics. |
| Boozer radial interpolant | Items `33`, `14`, `16` | XL | Port `simsoptpp/boozerradialinterpolant.cpp` and `boozermagneticfield*.h` into a JAX Boozer field spec/kernel. |
| Event-time tolerance lane | Items `14`, `16` | L | Add a parity-ladder SSOT lane for adaptive RK dense-output and root-localization accuracy before replacing Boost TOMS748/DOPRI behavior. |
| Framed curve kernels | Items `18`, `20` | M | Port `geo/framedcurve.py` frame ODE/framing kernels before finite-build geometry can claim JAX-native closure. |
| Dipole / PM kernels | Items `26`-`28` | XL | Port item `24` dipole field and item `25` PM optimization kernels before wrapper/grid/solve layers. |
| Wireframe kernels | Items `30`-`31` | XL | Port item `29` wireframe field and optimization kernels before public field/solve wrappers. |
| `points_coils` 2D collective | Multi-device release track | L | Add 2D strategy registration/config and grouped-field 2D lowering before any CPU forced-device or CUDA collective signoff. |

### Execution Waves

#### Wave R0 - Reconcile State And Plan Inputs

- [x] Commit or otherwise preserve the final closeout reconciliation files
  (committed as `51d4d2b7c`):
  `.artifacts/jax_port_goal/REPORT.md`, `.artifacts/jax_port_goal/state.json`,
  item-15 plan artifacts, and the lazy-export fix in
  `src/simsopt/field/__init__.py`.
- [ ] Decide whether the next run is CPU-only full implementation or
  CUDA-performance release. Default to CPU-only unless the user explicitly
  authorizes GPUs.
- [ ] Expand the state schema from aggregate `18-33` skipped row into individual
  pending item rows before coding prompt `P3`-`P5`.
- [ ] Enforce the top-level cross-cutting constraints for every activated wave.

#### Wave R1 - Complete The Item-15 Math And Wrapper Surface

- [ ] Implement `src/simsopt/jax_core/_elliptic.py` using Carlson `R_F` and
  `R_D` fixed-iteration `jax.lax.scan` kernels.
- [ ] Add direct parity tests against `scipy.special.ellipk` and `ellipe` over
  `m in [0, 1 - eps]`, including near-zero and near-one stress points. Use the
  parity ladder for tolerances.
- [ ] Implement `CircularCoil` B and `dB_by_dX` kernels. Prefer `jacfwd` of the
  B kernel only if it meets parity and memory gates; otherwise use the closed
  derivative formulas.
- [ ] Add `CircularCoilJAX` to `src/simsopt/field/magneticfieldclasses_jax.py`
  and export it only when both JAX and `simsoptpp` are available.
- [ ] Implement an `InterpolatedFieldJAX` construction contract:
  explicit source-field sampling at construction, immutable grid/spec arrays,
  cylindrical-to-Cartesian coordinate conversion, `nfp` rotational folding,
  `stellsym` z-folding, skip-mask behavior, and documented out-of-bounds
  behavior.
- [ ] Add wrapper parity for in-domain, folded, skip-mask, derivative, and
  out-of-domain cases against the CPU `InterpolatedField` oracle.
- [ ] Promote item `15` from `blocked_dependency` to `cpu_oracle_complete` only
  after both `CircularCoilJAX` and `InterpolatedFieldJAX` pass strict
  transfer-guard validation.

#### Wave R2 - Boozer Field Before Tracing

- [ ] Port `simsoptpp/boozerradialinterpolant.cpp` and related
  `boozermagneticfield*.h` data contracts into
  `src/simsopt/jax_core/boozer_radial_interp.py`.
- [ ] Define immutable Boozer grid/interpolant specs with explicit coordinate
  conventions, periodicity, derivative shape conventions, and field units.
- [ ] Add direct fixed-state parity against the C++ Boozer radial interpolant
  for B, derivatives, boundary/periodic points, and representative production
  fixtures.
- [ ] Implement `field/boozermagneticfield.py` JAX wrappers only after the
  JAX core Boozer kernels are green.
- [ ] Keep item `32` and item `33` separate: item `32` owns kernels/specs;
  item `33` owns public wrapper routing and restart/serialization behavior.

#### Wave R3 - Tracing Core And Public Tracing Wrappers

- [ ] Add a new parity-ladder lane for tracing event-time / Poincare crossing
  accuracy. This must be a contract update before coding item `14`, not an
  after-the-fact tolerance exception.
- [ ] Implement an in-repo JAX RK path for the C++ tracing surface:
  fieldline RHS first, then guiding-center and full-orbit RHS after Boozer
  fields are available.
- [ ] Use fixed-shape carries with max-step caps and masks for JAX loops; do
  not append Python lists or emit dynamic host objects from compiled kernels.
- [ ] Implement a JAX-compatible bracketed event localizer with the chosen
  tolerance lane; document why it is the accepted replacement for Boost
  TOMS748.
- [ ] Implement the JAX surface classifier used by
  `LevelsetStoppingCriterion`; reuse item-13 / `InterpolatedFieldJAX` grid
  specs where possible.
- [ ] Add CPU parity against `tests/field/test_fieldline.py`,
  `tests/field/test_particle.py`, and targeted Poincare/event fixtures.
- [ ] Only after item `14` is green, wire `field/tracing.py` item `16` public
  wrappers to the JAX backend. Do not add a placeholder backend or fallback.

#### Wave R4 - Prompt P3 Geometry, Optimizer, Sampling, Scalar Potential

- [ ] Item `18`: port `geo/framedcurve.py` ODE / framing operations to specs
  and JAX kernels. Cover `FramedCurveFrenet`, `FramedCurveCentroid`,
  `FrameRotation`, `ZeroRotation`, `rotated_centroid_frame`, and
  `rotated_frenet_frame` contracts. Upstream has no `geo/orientedcurve.py`.
- [ ] Item `20`: port `geo/finitebuild.py` after item `18`; parity must include
  filament construction, frame offsets, and derivative/VJP behavior.
- [ ] Item `19`: finish the private on-device optimizer contract audit for
  `qfmsurface.py`, `optimizer_jax.py`, `optimizer_jax_private/*`,
  `optimizer_jax_reference.py`, and `optimizer_host_lbfgs.py`. Keep host
  reference oracles explicit and outside compiled target mode.
- [ ] Item `21`: implement `field/magnetic_axis_helpers.py` on-axis iota ODE
  with an in-repo JAX RK/scan path and field-spec input. Reuse the tracing
  tolerance lane if the same event/ODE accuracy contract applies.
- [ ] Item `22`: port `field/sampling.py` with an explicit
  `jax.random.PRNGKey` contract. No hidden global RNG state.
- [ ] Item `23`: evaluate `ScalarPotentialRZMagneticField`. Proceed only if
  symbolic expressions can be lowered to a static JAX expression/spec before
  compile time; block if runtime SymPy/lambdify is the only path.

#### Wave R5 - Prompt P4 Permanent Magnet And Wireframe Lanes

- [ ] Item `24`: port `simsoptpp/dipole_field.cpp` to
  `src/simsopt/jax_core/dipole_field.py`; include field, derivative, and
  production-grid parity.
- [ ] Item `25`: port `simsoptpp/permanent_magnet_optimization.cpp` to
  `src/simsopt/jax_core/pm_optimization.py`; define immutable PM grid and
  optimizer-state specs.
- [ ] During items `25` and `28`, audit source-level initialization for any RNG
  or orientation sampling. If present, route it through the item `22`
  explicit-`PRNGKey` contract; do not introduce hidden global RNG state.
- [ ] Item `26`: implement `DipoleFieldJAX` after item `24`.
- [ ] Item `27`: port `geo/permanent_magnet_grid.py` after items `24` and
  `25`; preserve file/export behavior outside compiled kernels.
- [ ] Item `28`: port `solve/permanent_magnet_optimization.py` after item
  `25`; optimizer state must be explicit arrays/specs, not mutable globals.
- [ ] Item `29`: port `wireframe_optimization.cpp`,
  `magneticfield_wireframe.cpp`, and `wireframe_field_impl.h` to
  `src/simsopt/jax_core/wireframe.py`.
- [ ] Item `30`: implement `WireframeFieldJAX` after item `29`.
- [ ] Item `31`: port `solve/wireframe_optimization.py` after item `29`.

#### Wave R6 - `points_coils` 2D Sharding And Release Evidence

- [ ] Land `points_coils` as a single atomic change-set: runtime strategy
  registration, 2D device-count resolution, grouped-field kernel, replacement
  tests, and summary fields must be reviewed together.
- [ ] Finish the grouped-field 2D collective kernel: points sharded on the
  point axis, coils sharded on the coil axis, `lax.psum` over coil axis, and
  point-sharded output after reduction.
- [ ] Add CPU forced-device StableHLO tests proving the `points_coils` lowering
  contains the expected collective reduction.
- [ ] Replace
  `tests/test_backend.py::test_sharding_tuning_rejects_points_coils_strategy`
  with active `points_coils` registration/config tests in `tests/test_backend.py`
  in the same change-set.
- [ ] Add parity tests for non-divisible coil counts and mixed quadrature
  groups.
- [ ] If and only if the user authorizes GPU work, run CUDA/NCCL smoke for
  `coil_groups` and `points_coils` and populate `cuda_proof` artifacts from
  real CUDA execution.

### Full Remainder Definition Of Done

- [ ] Every item `14`-`33` has an individual state row with status
  `complete`, `blocked`, or `skipped`; no aggregate `18-33` row remains for an
  active full-implementation run.
- [ ] Every completed item has `closure_level=cpu_oracle_complete` or
  `closure_level=cuda_verified`, an oracle contract, coverage matrix,
  JAX-transform plan, math/physics invariants, red evidence, restart note,
  bench artifact, and targeted tests.
- [ ] Every blocked item has a blocker artifact with category, specific missing
  dependency, two-timebox evidence when applicable, and a proposed user
  decision.
- [ ] All new tolerances live in
  `benchmarks/validation_ladder_contract.py`; tests import those tolerances
  instead of inlining numeric `rtol` / `atol`.
- [ ] Strict transfer-guard tests pass for every item that claims a JAX-native
  host/device boundary.
- [ ] Public CPU/SIMSOPT compatibility remains intact; upstream CPU/C++
  behavior stays the oracle, not a path to edit away.
- [ ] CUDA is either explicitly `not_claimed` or proven by real CUDA artifacts;
  CPU/HLO proxies are never recorded as CUDA verification.

## Wave 1 Launch Status

- [x] P1: added `BiotSavartJAX.B_pullback_native(v)` and
  `B_cotangents(v)` as the native grouped cotangent API.
- [x] P1: kept `BiotSavartJAX.B_vjp(v) -> Derivative` and made it delegate
  through the native pullback payload.
- [x] P2: promoted `SingleStageRuntimeSpecBiotSavartJAX` and its spec-backed
  coil/current/curve views into package code.
- [x] P3: added generic sharding metadata fields for point and coil device
  counts without registering `points_coils`.
- [x] P3: kept the runtime from activating `points_coils`; the current
  environment value is rejected until grouped-field 2D execution is
  implemented.
- [x] P4: closed the Stage 2 ALM target seam so `backend='jax'` rejects
  `optimizer_backend='scipy'` and uses the target optimizer contract for
  `optimizer_backend='ondevice'` plus explicit JAX value/grad control lanes.
- [x] P4: audited the single-stage target startup contract; no new code change
  was needed there.
- [x] Code-simplifier pass: scoped to the Wave 1 files.
- [x] Validation: `py_compile`, `git diff --check`, full
  `tests/test_backend.py`, targeted `tests/test_jax_import_smoke.py`, and
  targeted Stage 2/single-stage integration tests pass.

## Wave 4A Closure Status

- [x] Native `B` pullbacks are checked directly against
  `biot_savart_B_vjp_maybe_collective(...)`, not only through projected public
  `Derivative` output.
- [x] Native `A`, `dA/dX`, `dB/dX`, `A_and_dA`, and `B_and_dB` pullbacks are
  checked against grouped JAX forward-kernel VJPs.
- [x] Native field pullbacks build payloads only from free-coil groups; a
  fixed-coil regression verifies fixed coils stay out of native cotangent
  metadata.
- [x] Forced CPU four-device `coil_groups` subprocess coverage proves the
  native pullback lowerings still contain a device collective. Forward grouped
  field lowering still asserts `all_reduce`; native pullback lowering asserts
  the compiled collective path with `all-gather`.
- [ ] Real CUDA `coil_groups` smoke remains open.

## Architecture Decision

- [x] Do not rewrite upstream SIMSOPT `Optimizable` as a JAX-native class.
- [x] Do not rewrite upstream SIMSOPT `Derivative` as a JAX-native class.
- [x] Keep upstream CPU/C++ `BiotSavart` untouched as the parity oracle.
- [x] Keep public SIMSOPT APIs returning `Derivative` where callers expect it.
- [x] Use immutable specs, explicit DOF vectors, and pytrees as the JAX-native
  replacement for `Optimizable`.
- [x] Use grouped cotangent arrays, flat gradients, and pytrees as the
  JAX-native replacement for `Derivative`.
- [x] Make the compiled lane consume the native equivalents directly.
- [x] Keep projection back into `Derivative` only at compatibility boundaries.

## Non-Goals

- [x] Do not add silent CPU fallback paths to JAX target mode.
- [x] Do not make `Derivative` a pytree keyed by Python `Optimizable` objects.
- [x] Do not replace `get_adjoint_runtime_state()` with a new gradient
  abstraction.
- [ ] Do not inline new numerical tolerances outside the parity ladder SSOT.
- [x] Do not refactor upstream SIMSOPT object graph semantics.

## Source Contracts

Official docs checked for this plan:

- JAX JIT and pure-function model:
  `https://docs.jax.dev/en/latest/jit-compilation.html`
- JAX explicit host/device transfer boundaries:
  `https://docs.jax.dev/en/latest/transfer_guard.html`,
  `https://docs.jax.dev/en/latest/_autosummary/jax.device_get.html`,
  `https://docs.jax.dev/en/latest/_autosummary/jax.block_until_ready.html`
- JAX default dtype and `jax_enable_x64` contract:
  `https://docs.jax.dev/en/latest/default_dtypes.html`
- JAX `shard_map` and `psum` collective semantics:
  `https://docs.jax.dev/en/latest/notebooks/shard_map.html`
- JAX NVIDIA GPU installation and CUDA plugin contract:
  `https://docs.jax.dev/en/latest/installation.html`
- SIMSOPT public field API baseline. The local upstream source remains the
  source of truth for `1.10.7.dev` deltas:
  `https://simsopt.readthedocs.io/v1.10.6/simsopt.field.html`
- NVIDIA NCCL collective semantics:
  `https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html`
- NVIDIA CUDA programming model and memory/thread hierarchy:
  `https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html`

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
- Current sharding strategy/config SSOT lives in:
  `src/simsopt/backend/runtime.py`
- Current negative `points_coils` proof lives in:
  `tests/test_backend.py::test_sharding_tuning_rejects_points_coils_strategy`
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
- [x] Add or update tests that verify native functions accept specs directly
  without reading mutable wrapper state.

### Derivative Equivalent

- [x] Use JAX VJPs internally for Biot-Savart and objective gradients.
- [x] Use grouped coil cotangents internally for field pullbacks.
- [x] Add a supported public native cotangent API on `BiotSavartJAX`.
- [x] Make internal objective/adjoint paths consume native cotangents where
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
- `BiotSavartJAX` exposes native pullbacks for `A`, `dA/dX`, `dB/dX`,
  `A_and_dA`, and `B_and_dB`; public VJP methods continue to project those
  payloads back into `Derivative`.

Tasks:

- [x] Add `BiotSavartJAX.B_pullback_native(v)` or
  `BiotSavartJAX.B_cotangents(v)`.
- [x] Return a typed native payload:
  grouped cotangents plus corresponding coil index lists.
- [x] Reimplement `BiotSavartJAX.B_vjp(v)` as:
  `B_pullback_native(v)` -> `coil_cotangents_to_derivative(...)`.
- [x] Route Boozer/objective internals through the native cotangent API when
  the caller does not need a `Derivative`.
- [x] Preserve `B_vjp(v) -> Derivative` for public SIMSOPT compatibility.
- [x] Add parity tests comparing:
  native cotangents -> projected `Derivative` vs current `B_vjp`.
- [x] Add multi-device tests proving native cotangents still lower through
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
- `points_coils` is not registered today. `SIMSOPT_JAX_SHARDING=points_coils`
  is intentionally rejected by
  `tests/test_backend.py::test_sharding_tuning_rejects_points_coils_strategy`.
- Existing point-axis strategies (`points`, `pairwise_rows`, `hybrid`) and
  coil-axis collectives are separate code paths.
- `points_coils` must be added as one atomic runtime-plus-kernel change:
  strategy validation, 2D mesh/device-count product check, grouped-field 2D
  `shard_map`, and StableHLO/parity tests.

Tasks:

- [ ] Add `points_coils` to `_VALID_SHARDING_STRATEGIES` only in the same
  change that wires the 2D grouped-field kernel.
- [ ] Wire existing `ShardingTuning` point/coil metadata into the 2D strategy.
- [ ] Add explicit point-axis and coil-axis device-count resolution for the 2D
  strategy.
- [ ] Build a 2D mesh helper requiring:
  `point_devices * coil_devices == device_count`.
- [ ] Add a grouped-field 2D collective kernel:
  points sharded on point axis, coils sharded on coil axis.
- [ ] Reduce over coil axis with `lax.psum`.
- [ ] Keep output point-sharded after the coil reduction.
- [ ] Replace the current reject test with registration/config tests when the
  kernel lands.
- [ ] Extend summaries to report:
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
- The production/default JAX target optimizer contract is
  `optimizer_backend='ondevice'`.
- Explicit parity/control lanes may use `optimizer_backend='scipy-jax'` or
  `optimizer_backend='scipy-jax-fullgraph'` with JAX value/grad evaluation.
  Plain `optimizer_backend='scipy'` remains rejected for `backend='jax'`.
- Stage 2 ALM now resolves its inner optimizer through the same target
  optimizer contract.
- Host reporting, parity artifacts, and explicit diagnostic callbacks must stay
  outside production compiled correctness paths.

Tasks:

- [x] Audit Stage 2 JAX target startup for accidental SciPy/reference optimizer
  use.
- [x] Audit single-stage JAX target startup for accidental SciPy/reference
  optimizer use.
- [x] Keep CPU/reference mode explicit and separate.
- [ ] Keep host reporting, artifact writing, and diagnostic callbacks outside
  production compiled correctness paths.
- [x] Remove fallback wording from docs/tests when the path has been deleted.
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

- [x] Script-local runtime-spec Biot-Savart adapter.
- [ ] Duplicate runtime-spec loading helpers, if a targeted grep finds an
  actual duplicate owner. Do not delete generic JSON/spec readers just because
  they share the `single_stage_jax_runtime_spec.json` artifact name.
- [x] Tests that patch removed host-compatibility helpers.
- [x] Docs claiming CPU fallback behavior that no longer exists.
- [x] Dead host pullback helpers that are no longer reachable.

Keep:

- [x] Public `Derivative` returns.
- [x] `Optimizable` compatibility wrappers.
- [x] CPU/reference parity oracle.
- [x] `BoozerSurfaceJAX.get_adjoint_runtime_state()`.
- [x] Host reporting and artifact materialization outside compiled kernels.

## Validation Matrix

Use `benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES`.
Do not inline new tolerances.

### Direct Kernel Lane

- [x] `B`: `tests/objectives/test_integral_bdotn_item10_closeout.py` and
  `tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py`.
- [ ] `A`: CPU parity exists in `tests/integration/test_stage2_jax.py`, but it
  still needs a parity-ladder-owned direct-kernel closeout row before this
  checklist can mark it complete.
- [x] grouped field: `tests/subprocess/jax_runtime_cases.py` covers
  `coil_groups` grouped B/A/derivative parity and collective lowering; the
  `points_coils` grouped field remains open under P3/R6.
- [x] fixed-surface flux:
  `tests/objectives/test_integral_bdotn_item10_closeout.py`.
- [x] raw Boozer residual:
  `tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py`.

### Derivative-Heavy Lane

- [x] `dB/dX`: `tests/field/test_biotsavart_jax.py` and
  `tests/field/test_biotsavart_jax_cpu_ordered.py`.
- [x] Biot-Savart native cotangents
- [x] projected `B_vjp`
- [x] surface coefficient Jacobians: `tests/geo/test_boozer_derivatives_jax.py`.
- [x] Boozer residual derivatives: `tests/geo/test_boozer_residual_jax.py` and
  `tests/geo/test_boozer_derivatives_jax.py`.

### Adjoint Lanes

Status note, 2026-05-05: Boozer adjoint closure tracking moved to
`docs/boozer_full_parity_plan_2026-05-04.md`. That newer plan supersedes this
April checklist for Boozer-specific exact-adjoint lane status.

- [x] `exact_well_conditioned_adjoint`: operator-vs-dense vector parity.
- [ ] `exact_ill_conditioned_adjoint`: true rank-deficient fixture remains
  future residual/failure-only coverage; current mixed-RHS operator-status
  coverage exercises the residual/failure branch.
- [x] Confirm dense PLU metadata never replaces operator-backed runtime solves.

### Multi-Device Lowering

- [x] CPU subprocess test with:
  `XLA_FLAGS=--xla_force_host_platform_device_count=4`
- [x] Assert StableHLO text contains `all_reduce`.
- [x] Assert `grouped_field_sharding_summary(...)["field_collective"] is True`.
- [x] Run with `SIMSOPT_JAX_SHARDING=coil_groups`.
- [ ] Run with `SIMSOPT_JAX_SHARDING=points_coils` only after P3 registers the
  strategy and replaces the current reject test.

### CUDA Smoke

- [ ] `JAX_PLATFORMS=cuda,cpu`
- [ ] `SIMSOPT_JAX_SHARDING=coil_groups`
- [ ] `SIMSOPT_JAX_SHARDING=points_coils`
- [ ] `NCCL_DEBUG=WARN`
- [ ] Validate parity and active collective summaries.

## Definition of Done

- [x] JAX target lane has native replacements for `Optimizable` and
  `Derivative` semantics in the compiled path.
- [x] Public SIMSOPT compatibility remains intact.
- [x] `BiotSavartJAX.B_vjp` still returns `Derivative`, but delegates to a
  native cotangent API.
- [x] Single-stage runtime-spec Biot-Savart adapter is package-owned, not
  script-owned.
- [ ] `coil_groups` lowers to collective reductions today; `points_coils`
  registration plus 2D lowering has landed and lowers to collective reductions.
- [x] Stage 2 and single-stage JAX target modes reject host optimizer seams.
- [ ] Validation uses the parity ladder SSOT.
- [x] Stale fallback code/docs/tests are removed only after parity coverage.
