# single_stage_banana_example.py JAX/GPU dependency trace

## Scope

This note traces the execution path of `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py` and identifies what still needs to be ported, or intentionally isolated, for truly end-to-end JAX/GPU execution.

For the callback-density and persistent-compilation-cache follow-up on the same
path, see `docs/single_stage_banana_ondevice_hot_path_diagnosis_2026-04-19.md`.

The key conclusion is:

- [x] The core single-stage target lane is already on the JAX/ondevice path.
- [ ] The remaining gaps are narrower than the original startup/sync wording implied: startup object loading still restores Python/SIMSOPT objects, warm-start reprojection is an explicit setup-time host fit, self-intersection remains a host validation gate, and requested artifact/result-file assembly still runs on the host; the supported JAX Boozer setup, accepted-step compute, and final penalty metrics already stay on the deferred/device-first lane.
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
   - `load(stage2_bs_path)` / `load_single_stage_warm_start_state(...)` at `single_stage_banana_example.py:5719-5725`
2. Field backend wrap
   - `BiotSavartJAX(bs.coils)` at `single_stage_banana_example.py:5741-5749`
   - implementation: `src/simsopt/field/biotsavart_jax_backend.py`
   - transitive core: `src/simsopt/jax_core/field.py`, `src/simsopt/jax_core/biotsavart.py`, `src/simsopt/jax_core/specs.py`
3. Boozer initialization
   - `initialize_boozer_surface(...)` at `single_stage_banana_example.py:5961`
   - JAX path selects `BoozerSurfaceJAX` at `single_stage_banana_example.py:2602-2610`
   - implementation: `src/simsopt/geo/boozersurface_jax.py`
   - transitive core: `src/simsopt/geo/boozer_residual_jax.py`, `src/simsopt/geo/label_constraints_jax.py`, `src/simsopt/geo/optimizer_jax.py`
4. JAX objective bundle
   - lazy-loaded builders at `single_stage_banana_example.py:3020-3044`
   - target-lane bundle setup at `single_stage_banana_example.py:3068-3076`
   - implementation: `src/simsopt/geo/surfaceobjectives_jax.py`
   - transitive core: `curveobjectives.py` pure kernels, `surfaceobjectives.py` pure kernels, `jax_core/curve_geometry.py`, `jax_core/field.py`
5. JAX outer optimizer
   - `run_single_stage_optimizer(...)` at `single_stage_banana_example.py:3908`
   - ondevice route goes to `target_minimize(...)` at `single_stage_banana_example.py:4869`
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
  - see `single_stage_banana_example.py:3304-3538`

## Remaining host-bound seams

Tracking checklist:

- [ ] Startup object loading and warm-start reprojection remain setup-time host seams
- [ ] Boozer setup compatibility and self-intersection postprocess remain host-visible seams
- [x] Accepted-step synchronization compute is device-first; callback diagnostics remain optional host observability
- [x] Heavy plotting/VTK artifacts are no longer default on the JAX path; requested artifact export remains host-side
- [x] Final penalty metrics come from the traceable runtime bundle on the target lane; final JSON assembly remains host-side
- [x] Optional diagnosis modes remain clearly separated from the GPU-pure target lane

### 1. Startup object loading and warm-start reprojection

Code:

- `single_stage_banana_example.py:5719-5725`
- `single_stage_banana_example.py:5768-5779`
- `single_stage_banana_example.py:1526-1671`

Why it is still host-bound:

- `load(...)` restores Python/SIMSOPT objects.
- the supported-surface path is narrower than the legacy object path:
  - target geometry extraction already uses JAX-backed `surface_gamma_from_dofs(...)` / `surface_rz_fourier_gamma_from_dofs(...)` for `SurfaceXYZTensorFourier`, `SurfaceRZFourier`, and serialized equivalents.
  - unsupported surfaces are rejected at this boundary rather than silently falling back to a host sampling path.
- the reprojection fit is an explicit setup-time host boundary, not a hidden target-lane transfer:
  - `_fit_surface_xyz_tensor_dofs_to_gamma(...)` hostifies the target geometry/design matrix and solves with `np.linalg.lstsq(...)`
  - this avoids the prior GPU `lstsq`/cuSolver startup failure mode while keeping the optimization hot path device-first
  - the legacy alias-equivalent coefficient convention still uses host `SurfaceXYZTensorFourier.least_squares_fit(...)` only while resolving the canonical representative for degenerate basis groups

What would need porting:

- a JAX-native seed/spec loader for stage-2 coil state
- a fully device-native replacement for the warm-start fit and alias-convention canonicalization, or an explicit decision that this setup-time compatibility fit is outside the GPU-pure optimization contract

### 2. Boozer pre-fit and self-intersection postprocess

Code:

- `single_stage_banana_example.py:2488-2531`
- `single_stage_banana_example.py:2573-2581`
- `single_stage_banana_example.py:2598-2617`
- `single_stage_banana_example.py:3003-3008`

Why it is still host-bound:

- initialization already reuses projected DOFs before setup and passes `surface_runtime_state` into `BoozerSurfaceJAX`, so the remaining seam is narrower than a generic "Boozer pre-fit" label suggests.
- the checked-in JAX initialization path now constructs a `DeferredSurfaceXYZTensorFourier` and passes `surface_runtime_state` into `BoozerSurfaceJAX` without eagerly materializing a host `SurfaceXYZTensorFourier`.
- self-intersection uses `surface.is_self_intersecting()` with optional `ground` / `shapely` style backends

What would need porting:

- only if startup must become fully spec-only: a `BoozerSurfaceJAX` setup contract that can avoid keeping even the deferred public surface object for compatibility
- either a JAX topology/self-intersection approximation or an explicit decision that this remains a host-only validation gate outside the core target lane

### 3. Accepted-step synchronization and callback diagnostics

Code:

- target-lane accepted-step sync builder: `single_stage_banana_example.py:3054-3134`
- host accepted-step callback path: `single_stage_banana_example.py:5225-5340`
- adapter sync plumbing: `single_stage_banana_example.py:5418-5460`
- ondevice optimizer callback plumbing: `src/simsopt/geo/optimizer_jax.py:1028-1045`, `src/simsopt/geo/optimizer_jax.py:1107-1126`

Why it is no longer the main compute-path blocker:

- the accepted-step target-lane path now computes reporting metrics from the runtime bundle first and caches one accepted-step summary in array-native run state.
- when the final DOFs still match that accepted step, final reporting reuses the cached summary instead of re-entering the mutable host objective graph just to recompute the same accepted-step metrics.
- runtime snapshots now prefer an explicit solved-state contract from `BoozerSurfaceJAX.get_solved_runtime_state()` when available, so the accepted-step/reporting seam no longer has to reconstruct solved surface state indirectly from host wrappers.
- hardware constraints are evaluated from pure reporting metrics before the explicit host-formatting boundary.
- the remaining host seam is the explicit observability/reporting boundary:
  - `jax.debug.callback(...)` still executes Python callbacks when accepted-step logging is enabled
  - logging/file append and mutable-graph diagnostics still run on host objects
  - benchmark/diagnostic modes can still re-enter mutable wrappers on purpose

What would need porting:

- keep callback-driven observability clearly optional and outside the core target-lane compute contract
- continue shrinking mutable-graph diagnostics if per-accept host logging must become GPU-pure instead of explicitly host-observability-only

### 4. Initial/final artifacts and plotting

Code:

- initial artifacts: `single_stage_banana_example.py:5996-6033`
- final artifacts: `single_stage_banana_example.py:6754-6779`

Why it is still host-bound:

- restart JSON export, requested VTK export, and matplotlib plots are CPU-side
- `bs_diag.save(...)`, `curves_to_vtk(...)`, `surface.to_vtk(...)`, `surface.save(...)`, `normPlot(...)`, and `cross_section_plot(...)` all imply host materialization
- the default JAX path now skips heavy VTK/plot artifacts unless `--full-artifacts` is explicitly requested; benchmark mode still skips restart JSON as well

What would need porting:

- keep heavy visualization artifacts as explicit opt-in for JAX/GPU runs
- create a separate post-run export stage if artifact generation itself must become decoupled from live host SIMSOPT objects

### 5. Final metrics and results assembly

Code:

- `single_stage_banana_example.py:2875-2963`
- final penalty/result assembly in `single_stage_banana_example.py:6839-7099`
- optimizer diagnostics and final JSON emission in `single_stage_banana_example.py:7079-7166`

Why it is no longer the main compute-path blocker:

- hardware-constraint evaluation itself is already pure on the target lane:
  - `evaluate_single_stage_hardware_constraints_pure(...)` computes the feasibility result in JAX space
  - `_hostify_single_stage_hardware_constraints(...)` is only the public/report formatting boundary
  - `resolve_single_stage_final_penalty_metrics(...)` uses the pure hardware evaluator before hostifying once on the target lane
- accepted-step reporting is also narrower than the older mixed-mode wording implied:
  - the final path first checks whether the accepted-step summary cached in run state is still valid for the final DOFs and benchmark mode
  - when it is valid, the final report reuses that cached target-lane summary instead of recomputing through host wrappers
- on the target lane, final penalty/reporting scalars now come from the traceable runtime bundle rather than re-entering host objective wrappers.
- the host-wrapper calls still exist on the legacy/reference fallback path:
  - `curvelength.J()`
  - `JCurveCurve.shortest_distance()`
  - `JCurveSurface.shortest_distance()`
  - `JSurfSurf.shortest_distance()`
  - objective `.J()` calls for the final results JSON
- the surrounding results assembly path still performs optimizer diagnostics extraction and final JSON writes on the host

What would need porting:

- if result writing must be GPU-pure end to end, assemble the final JSON from device-independent specs/results in a separate host export step instead of live mutable objects

### 6. Optional diagnosis modes intentionally reintroduce host wrappers

Code:

- `single_stage_banana_example.py:3020-3044`
- target-lane runtime bundle keeps `include_host_wrappers=False` at `single_stage_banana_example.py:3068-3076`
- optional wrapper reintroduction lives in `src/simsopt/geo/surfaceobjectives_jax.py:2885-2919`

Why it matters:

- this is useful for debugging, but it is not a GPU-pure path
- it should stay clearly separated from the production target lane

## What does not need a fresh JAX rewrite right now

The trace does not point to the core field, Boozer, or traceable-objective math as the next porting target. Those are already the ondevice lane:

- `src/simsopt/field/biotsavart_jax_backend.py`
- `src/simsopt/geo/boozersurface_jax.py`
- `src/simsopt/geo/surfaceobjectives_jax.py`
- `src/simsopt/geo/optimizer_jax.py`

The highest-leverage remaining work is to keep shrinking the explicit host seams around startup/setup and reporting, not to rewrite the core field/Boozer/outer-optimizer math again.

## Recommended port order

- [x] Remove mutable-graph final reporting from the default JAX path.
  Final objective term scalars and hardware metrics now come from the traceable runtime bundle rather than host wrapper recomputation on the target lane.
- [x] Keep the default JAX path artifact-light.
  Heavy VTK/plot export is skipped unless `--full-artifacts` is explicitly requested.
- [x] Keep accepted-step sync compute device-first by default.
  Preserve the current runtime-bundle-first path and keep `per-accept` logging a clearly host-observability mode.
- [ ] Decide whether startup must be GPU-pure or just the optimization lane.
  If truly end-to-end, port seed loading and the setup-time warm-start fit/canonicalization to immutable JAX specs and a device-native Boozer setup contract.
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

If the question is "what else must be ported for full JAX/GPU support?", the answer is still not "more field/Boozer kernel math." The next porting surface is:

- [ ] Warm-start reprojection and Boozer setup, if startup purity matters end to end
- [x] Accepted-step sync compute path is already device-first; keep host work explicit at the reporting/logging boundary
- [x] Hardware-constraint evaluation is already pure on the target lane; keep host normalization explicit at the public/report boundary
- [x] Default heavy artifact generation is opt-in on the JAX path
- [x] Final penalty metrics are runtime-bundle-backed on the target lane
- [ ] Final result-file assembly and requested artifact export remain host-side

The core optimize path is already on the target lane. The remaining work is mostly about preventing startup/setup and post-solve reporting paths from re-entering host-side SIMSOPT objects beyond their now-explicit boundaries.
