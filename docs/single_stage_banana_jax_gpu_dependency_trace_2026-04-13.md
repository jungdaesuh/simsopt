# single_stage_banana_example.py JAX/GPU dependency trace

## Scope

This note traces the execution path of `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py` and identifies what still needs to be ported, or intentionally isolated, for truly end-to-end JAX/GPU execution.

The key conclusion is:

- [x] The core single-stage target lane is already on the JAX/ondevice path.
- [ ] The remaining gaps are mostly at startup, accepted-step synchronization, diagnostics, artifact generation, and final reporting.
- [x] The current repo backlog is consistent with this: `jax_gpu_port_todos_2026-04-08.md:115` says the main remaining single-stage work after the GPU proof is algorithmic donor/seed/search policy, not more proof scaffolding.

## Entry-point dependency spine

Primary entrypoint imports:

- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:25`
  - `repo_bootstrap.py`
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:39`
  - `src/simsopt/config.py`
  - `src/simsopt/backend/runtime.py`
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:40`
  - CPU field API via `src/simsopt/field/__init__.py`
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:41`
  - JAX scalar/device helpers via `src/simsopt/jax_core/_math_utils.py`
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:45`
  - geometry/objective graph entrypoints via `src/simsopt/geo/__init__.py`
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:53`
  - pure pairwise distance helpers and legacy curve objective wrappers via `src/simsopt/geo/curveobjectives.py`
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:59`
  - CPU surface objectives plus `surface_to_surface_distance_pure` source via `src/simsopt/geo/surfaceobjectives.py`
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:70`
  - `examples/single_stage_optimization/hardware_constraints.py`
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:74`
  - explicit host materialization helpers via `examples/single_stage_optimization/jax_host_boundary.py` and `src/simsopt/_core/jax_host_boundary.py`
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:75`
  - plotting/artifact helpers via `examples/single_stage_optimization/plotting_utils.py`
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:76`
  - provenance helpers via `examples/single_stage_optimization/run_metadata.py`

Runtime dependency spine for the JAX/ondevice lane:

1. Stage-2 seed + warm start load
   - `load(stage2_bs_path)` at `single_stage_banana_example.py:4715`
   - `load_single_stage_warm_start_state(...)` at `single_stage_banana_example.py:4721`
2. Field backend wrap
   - `BiotSavartJAX(bs.coils)` at `single_stage_banana_example.py:4742-4745`
   - implementation: `src/simsopt/field/biotsavart_jax_backend.py`
   - transitive core: `src/simsopt/jax_core/field.py`, `src/simsopt/jax_core/biotsavart.py`, `src/simsopt/jax_core/specs.py`
3. Boozer initialization
   - `initialize_boozer_surface(...)` at `single_stage_banana_example.py:4957`
   - JAX path selects `BoozerSurfaceJAX` at `single_stage_banana_example.py:2041-2044`
   - implementation: `src/simsopt/geo/boozersurface_jax.py`
   - transitive core: `src/simsopt/geo/boozer_residual_jax.py`, `src/simsopt/geo/label_constraints_jax.py`, `src/simsopt/geo/optimizer_jax.py`
4. JAX objective bundle
   - lazy-loaded builders at `single_stage_banana_example.py:2232-2266`
   - target-lane bundle setup at `single_stage_banana_example.py:5228`
   - implementation: `src/simsopt/geo/surfaceobjectives_jax.py`
   - transitive core: `curveobjectives.py` pure kernels, `surfaceobjectives.py` pure kernels, `jax_core/curve_geometry.py`, `jax_core/field.py`
5. JAX outer optimizer
   - `run_single_stage_optimizer(...)` at `single_stage_banana_example.py:3858`
   - ondevice route goes to `target_minimize(...)` at `single_stage_banana_example.py:3899-3923`
   - implementation: `src/simsopt/geo/optimizer_jax.py`

## Already on the JAX/ondevice path

These are not the main remaining porting blockers:

- Field evaluation:
  - `BiotSavartJAX` is already a JAX-native adapter over immutable coil specs.
  - see `src/simsopt/field/biotsavart_jax_backend.py:1`
- Boozer inner solve:
  - `BoozerSurfaceJAX` explicitly distinguishes the host/reference lane from the `optimizer_backend="ondevice"` target lane.
  - see `src/simsopt/geo/boozersurface_jax.py:1`
- Traceable outer objective:
  - `surfaceobjectives_jax.py` already provides JAX wrappers plus the traceable runtime bundle used by the target lane.
  - see `src/simsopt/geo/surfaceobjectives_jax.py:1`
- Outer optimizer:
  - `target_minimize()` already routes the target lane to ondevice BFGS/L-BFGS implementations rather than the SciPy adapter.
  - see `src/simsopt/geo/optimizer_jax.py:1`
- Hardware success filter:
  - `build_single_stage_target_lane_hardware_success_filter(...)` is already written as a pure-JAX feasibility predicate.
  - see `single_stage_banana_example.py:2349-2577`

## Remaining host-bound seams

Tracking checklist:

- [ ] Startup object loading and warm-start projection
- [ ] Boozer pre-fit and self-intersection postprocess
- [ ] Accepted-step synchronization and callback diagnostics
- [ ] Initial/final artifacts and plotting
- [ ] Final metrics and results assembly
- [ ] Optional diagnosis modes remain clearly separated from the GPU-pure target lane

### 1. Startup object loading and warm-start projection

Code:

- `single_stage_banana_example.py:4715-4721`
- `single_stage_banana_example.py:1151-1208`

Why it is still host-bound:

- `load(...)` restores Python/SIMSOPT objects.
- warm-start projection uses Python file IO, CPU `SurfaceXYZTensorFourier`, `np.asarray(...)`, and `least_squares_fit(...)`.

What would need porting:

- a JAX-native seed/spec loader for stage-2 coil state
- a device-native warm-start surface projection path, or an immutable precomputed surface-spec artifact

### 2. Boozer pre-fit and self-intersection postprocess

Code:

- `single_stage_banana_example.py:2025-2036`
- `single_stage_banana_example.py:2133-2153`
- `single_stage_banana_example.py:2220`

Why it is still host-bound:

- the seed surface fit uses `surf_prev.gamma()` and `least_squares_fit(...)` on host objects
- self-intersection uses `surface.is_self_intersecting()` with optional `ground` / `shapely` style backends

What would need porting:

- a JAX-native seed surface fit path, if startup must also remain on device
- either a JAX topology/self-intersection approximation or an explicit decision that this remains a host-only validation gate outside the core target lane

### 3. Accepted-step synchronization and callback diagnostics

Code:

- sync-policy gate: `single_stage_banana_example.py:923-948`
- callback body: `single_stage_banana_example.py:4238-4355`
- final sync path: `single_stage_banana_example.py:5701-5712`
- ondevice optimizer callback plumbing: `src/simsopt/geo/optimizer_jax.py:1028-1045`, `src/simsopt/geo/optimizer_jax.py:1107-1126`

Why it is still host-bound:

- the ondevice optimizer uses `jax.debug.callback(...)` to execute Python callbacks
- the callback hostifies arrays and then evaluates mutable graph objects:
  - `obj.J()` / `obj.dJ()`
  - `banana_curve.gamma()`
  - `boozer_surface.surface.gamma()` / `.unitnormal()`
  - `bs.B()`
  - file append logging

What would need porting:

- replace callback-driven mutable-graph diagnostics with explicit returned device metrics/state
- keep the default JAX path on `final-only` sync, or move accepted-step observability into a separate opt-in diagnostic mode

### 4. Initial/final artifacts and plotting

Code:

- initial artifacts: `single_stage_banana_example.py:4993-5028`
- final artifacts: `single_stage_banana_example.py:5723-5779`

Why it is still host-bound:

- VTK/JSON export and matplotlib plots are CPU-side
- `bs_diag.save(...)`, `curves_to_vtk(...)`, `surface.to_vtk(...)`, `surface.save(...)`, `normPlot(...)`, and `cross_section_plot(...)` all imply host materialization

What would need porting:

- either make these explicitly non-default for JAX/GPU runs
- or create a separate post-run export stage that consumes saved device-independent specs/results

### 5. Final metrics and results assembly

Code:

- `single_stage_banana_example.py:5793-5826`
- `single_stage_banana_example.py:5978-6091`

Why it is still host-bound:

- final reporting still calls host-facing wrapper objects:
  - `curvelength.J()`
  - `JCurveCurve.shortest_distance()`
  - `JCurveSurface.shortest_distance()`
  - `JSurfSurf.shortest_distance()`
  - objective `.J()` calls for the final results JSON

What would need porting:

- expose final scalar term values directly from the traceable runtime bundle, rather than re-entering the mutable `Optimizable` graph for reporting

### 6. Optional diagnosis modes intentionally reintroduce host wrappers

Code:

- `single_stage_banana_example.py:2780-3009`
- especially `include_host_wrappers=True` at `single_stage_banana_example.py:2896-2903`

Why it matters:

- this is useful for debugging, but it is not a GPU-pure path
- it should stay clearly separated from the production target lane

## What does not need a fresh JAX rewrite right now

The trace does not point to the core field, Boozer, or traceable-objective math as the next porting target. Those are already the ondevice lane:

- `src/simsopt/field/biotsavart_jax_backend.py`
- `src/simsopt/geo/boozersurface_jax.py`
- `src/simsopt/geo/surfaceobjectives_jax.py`
- `src/simsopt/geo/optimizer_jax.py`

The highest-leverage remaining work is to stop the default JAX script from falling back into host-side diagnostics and reporting once the core ondevice solve is done.

## Recommended port order

- [ ] Remove mutable-graph final reporting from the default JAX path.
  Return final objective term scalars and hardware metrics from the traceable runtime bundle.
- [ ] Keep the default JAX path artifact-light.
  No VTK/plot export unless explicitly requested.
- [ ] Keep accepted-step sync host-free by default.
  Preserve `final-only` sync; make `per-accept` a clearly host-observability mode.
- [ ] Decide whether startup must be GPU-pure or just the optimization lane.
  If truly end-to-end, port seed loading and warm-start projection to immutable JAX specs.
- [ ] Treat self-intersection as a separate design decision.
  Either leave it as an explicit host validation gate or implement a JAX approximation; do not leave it as an implicit mixed-mode dependency.

## Official JAX references

These are the upstream references most relevant to the remaining seams:

1. Stateful computations
   - https://docs.jax.dev/en/latest/stateful-computations.html
   - JAX shows that in-place state updates inside jitted code do not behave like normal Python state and recommends explicit state threading instead.
2. JIT compilation and side effects
   - https://docs.jax.dev/en/latest/jit-compilation.html
   - JAX states that impure functions and Python side effects are dangerous under transformations because side effects are not part of the traced jaxpr.
3. Transfer guard
   - https://docs.jax.dev/en/latest/transfer_guard.html
   - JAX can log or disallow implicit host-device transfers, including accidental device-to-host fetches.
4. Asynchronous dispatch
   - https://docs.jax.dev/en/latest/async_dispatch.html
   - printing or converting a `jax.Array` on the host forces synchronization, which is exactly what the host diagnostic/report path does.
5. Persistent compilation cache
   - https://docs.jax.dev/en/latest/persistent_compilation_cache.html
   - cache setup must happen before first compilation, `jax_explain_cache_misses` exists for tracing cache misses, and functions with host callbacks are not cached.
6. `jax.device_get`
   - https://docs.jax.dev/en/latest/_autosummary/jax.device_get.html
   - explicit host fetches are first-class transfers and should stay out of the hot path.
7. `jax.scipy.optimize.minimize`
   - https://docs.jax.dev/en/latest/_autosummary/jax.scipy.optimize.minimize.html
   - the built-in JAX SciPy minimize path is still limited, which helps justify keeping the custom `optimizer_jax` target lane instead of trying to collapse onto the stock API.

## Bottom line

If the question is "what else must be ported for full JAX/GPU support?", the answer is not "more field/Boozer kernel math." The next porting surface is:

- [ ] Startup deserialization/projection if end-to-end purity matters
- [ ] Accepted-step synchronization and diagnostics
- [ ] Artifact generation and final reporting

The core optimize path is already on the target lane. The remaining work is mostly about preventing the script from re-entering host-side SIMSOPT objects once the ondevice solve is finished.
