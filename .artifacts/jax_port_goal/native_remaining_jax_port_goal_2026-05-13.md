# `/goal` prompt - native SIMSOPT remaining JAX-port candidates

Date: 2026-05-13
Repo: `/Users/suhjungdae/code/columbia/simsopt-jax`
Baseline closeout: `.artifacts/jax_port_goal/REPORT.md`
Baseline state: `.artifacts/jax_port_goal/state.json`

## Purpose

This document is a `/goal`-ready prompt for a second JAX-port wave after the
CPU-JAX/no-GPU closeout of the active manifest. The completed manifest items
`01`-`33` plus `points_coils` are baseline evidence, not work to redo.

The new objective is narrower and more precise than "port all SIMSOPT":
classify and close the remaining **native SIMSOPT** surfaces that are
JAX-portable but not yet fully JAX-ported.

Native SIMSOPT means repo-owned code under `src/simsopt/` and
`src/simsoptpp/`. External solver calls are not targets unless the work is a
fixed-output native post-processing kernel.

## How to use

Paste the contents of the `## Prompt` section into a `/goal` autonomous
orchestrator. Before launch, edit only:

- `active_scope`: which item ids to execute.
- `inventory_only_scope`: which item ids to classify but not implement.
- `scope_profile`: `cpu_port_closure` by default, or `cuda_perf_release` only
  with explicit user approval and a real CUDA environment.
- `skip_items`: items the human explicitly wants excluded.

---

## Prompt

~~~markdown
# Goal: close remaining native SIMSOPT JAX-port candidates

You are working in:

`/Users/suhjungdae/code/columbia/simsopt-jax`

You must iterate until every item in `active_scope` is one of:

- COMPLETE: implemented with source audit, CPU oracle parity, strict-transfer
  validation, and updated status artifacts.
- BLOCKED: a root blocker is documented with code evidence and no silent
  workaround.
- SKIPPED: explicitly out of scope for this run, with a short reason.

For every item in `inventory_only_scope`, do source/docs classification only.
Do not implement it unless the human first promotes it to `active_scope`.
Record each inventory-only row with `closure_level=inventory_only` and a
final status of `SKIPPED` or `BLOCKED`.

Default `scope_profile`: `cpu_port_closure`.

Default `active_scope`:

- `N01_boozer_analytic`
- `N02_interpolated_boozer_field`
- `N03_boozer_field_route_inventory`
- `N04_garabedian_henneberg_surfaces`
- `N05_curvexyzfourier_symmetries_spec`
- `N06_surface_xyz_tensor_clamped_dims`
- `N07_linking_number`
- `N09_magnetic_field_composition`
- `N11_live_pm_grid_host_workflow_decision`

Default `inventory_only_scope`:

- `N08_mhd_fixed_output_postprocessing`
- `N10_generic_solver_orchestration_inventory`
- `N12_qfm_surface_host_orchestration_inventory`
- `N13_mgrid_io_inventory`
- `N14_fourier_interpolation_utility_inventory`

Default `skip_items`: none.

`scope_profile`, `active_scope`, and `inventory_only_scope` are plan controls,
not repo runtime modes. When tests need the repo runtime lane, use the real
`SIMSOPT_BACKEND_MODE` values such as `jax_cpu_parity` or `jax_gpu_parity`.

## Hard boundaries

- Do not re-port completed manifest items `01`-`33` or `points_coils` unless
  required as a dependency for a new item.
- Do not claim real CUDA proof unless `scope_profile=cuda_perf_release`, the
  user explicitly approved GPU work, and the artifact was produced on a real
  CUDA device with CUDA-resident `jaxlib`.
- Do not port VMEC, SPEC, BOOZ_XFORM, `virtual_casing`, or `mpi4py` themselves.
  They are external solver/runtime dependencies.
- Native post-processing of fixed VMEC/Boozer/SPEC outputs may be ported if the
  compiled JAX path consumes immutable arrays/specs and does not run the
  external solver.
- Do not silently bridge CPU objects into JAX paths. If a route cannot be made
  native, keep it fail-fast and document the boundary.
- Do not add broad `try/except`, silent fallback, runtime mode flags that hide
  CPU fallback, dynamic imports, casts to `Any`, or new dependencies without
  explicit human approval.
- Do not add public exports until the implementation has CPU oracle tests and
  downstream import/use coverage.
- Preserve existing CPU/C++ oracle behavior. The trust chain is:
  existing SIMSOPT C++/SciPy behavior -> JAX CPU matches -> JAX GPU matches
  only if GPU is explicitly in scope.

## Official-source gate

Before changing code for an item, validate the item against current project
source plus the relevant official docs.

Context7 library ids selected during the 2026-05-13 doc review:

- JAX: `/google/jax`
- SIMSOPT: `/hiddensymmetries/simsopt`
- CUDA Toolkit: `/websites/nvidia_cuda`

- JAX:
  - Transfer guard: https://docs.jax.dev/en/latest/transfer_guard.html
  - `jax.jit`: https://docs.jax.dev/en/latest/_autosummary/jax.jit.html
  - Buffer donation:
    https://docs.jax.dev/en/latest/buffer_donation.html
  - `shard_map` / `pmap` migration:
    https://docs.jax.dev/en/latest/migrate_pmap.html
  - Installation / CUDA wheels:
    https://docs.jax.dev/en/latest/installation.html
- SIMSOPT:
  - Current official docs: https://simsopt.readthedocs.io/latest/
  - MHD docs:
    https://simsopt.readthedocs.io/latest/mhd.html
- CUDA:
  - CUDA Programming Guide:
    https://docs.nvidia.com/cuda/cuda-programming-guide/index.html
  - CUDA environment variables:
    https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html

Official-doc constraints to preserve:

- JAX `jit` functions must be pure at the compiled boundary, with array/scalar
  or standard-container arguments. Static arguments are compile-cache keys.
- JAX transfer guard `disallow` blocks unintended implicit transfers, but
  explicit `device_put`/`device_get` transfers are still allowed and CPU
  device-to-host fetches are always allowed. It is a guard, not by itself a
  proof that no host/device movement exists.
  The `jax.transfer_guard(...)` context manager is thread-local; spawned threads
  use the global option, so threaded/process validation must set the process
  config or environment, not only a local context manager.
- JAX buffer donation belongs on `jit`, `pmap`, or `pjit` boundaries. For
  `jit(shard_map(...))`, donation belongs on the `jit` wrapper, and donated
  input buffers must not be reused after the call. Donation is positional:
  keyword-argument calls do not donate buffers, so donation tests must exercise
  the real positional call path.
- `pmap` / `shard_map` migration-sensitive code must use explicit sharding and
  explicit `device_put` placement. Do not rely on implicit resharding or
  replicated-array indexing as a correctness or performance proof.
- A CUDA claim requires a CUDA-enabled `jaxlib`, real CUDA-visible devices, and
  provenance from `jax.devices()`, `nvidia-smi`, driver/runtime versions, and
  `CUDA_VISIBLE_DEVICES`. Do not infer CUDA proof from package metadata alone.
  `CUDA_VISIBLE_DEVICES` changes which GPUs are visible and their enumeration
  order; an empty value means no GPUs are visible. Record `nvidia-smi -L` plus
  `jax.devices()` and reject invalid/empty selectors for CUDA proof.
- JAX's current installation docs recommend CUDA 13 wheels for new installs
  while still documenting CUDA 12 wheels. The repo's `pyproject.toml` JAX extras
  remain the source of truth for this run; record any upstream wheel-family
  mismatch instead of changing dependencies inside a port item.
- SIMSOPT MHD docs state VMEC/SPEC are external equilibrium-code interfaces
  that must be installed separately. Do not treat those solvers as native JAX
  port targets.

## Baseline evidence to verify first

- [ ] Read `.artifacts/jax_port_goal/REPORT.md`.
- [ ] Read `.artifacts/jax_port_goal/state.json`.
- [ ] Confirm `stop_condition=met_cpu_jax_no_gpu`.
- [ ] Confirm items `01`-`33` plus `points_coils` are complete for
  `cpu_oracle_complete` scope.
- [ ] Confirm CUDA proof remains `not_claimed`.
- [ ] Confirm current tree status and record unrelated dirty files before
  editing.
- [ ] Confirm `pyproject.toml` JAX/JAX_GPU extras and official docs URL.
- [ ] Confirm each candidate is still unported in the current tree before
  implementing it.
- [ ] Confirm baseline-covered native candidates are not reopened as new active
  work: `CurvePlanarFourier`, `CurvePerturbed`, `CurveCWSFourier`,
  `CurveXYZFourier`, `CurveRZFourier`, `RotatedCurve` where already covered by
  grouped-coil/curve wrappers, `SurfaceXYZFourier`, distance candidate spatial
  hash routes, Boozer radial helper primitives (`sopp.compute_kmnc_kmns`,
  `sopp.inverse_fourier_transform_*`), permanent-magnet fixed-state solve
  wrappers, and wireframe fixed-state solve wrappers.
- [ ] For any candidate named in an external audit, first classify whether it is
  already covered by baseline items `01`-`33` / `points_coils`, a new portable
  gap, host orchestration, IO/visualization, or an external solver wrapper.

Recommended initial commands:

```bash
git status --short
rg -n "stop_condition|cpu_oracle_complete|not_claimed|open_gaps" \
  .artifacts/jax_port_goal/state.json .artifacts/jax_port_goal/REPORT.md
rg -n "JAX =|JAX_GPU =|Documentation =" pyproject.toml
rg -n "class .*\\(sopp|sopp\\.|import simsoptpp|from simsoptpp" \
  src/simsopt -g '*.py'
rg -n "def is_jax_backend|raise_if_target_lane_bypass|raise_if_strict_jax_fallback" \
  src/simsopt/backend/runtime.py src/simsopt
rg --files src/simsoptpp src/simsopt/jax_core src/simsopt | sort
```

## Required artifacts

Create a new artifact root:

`.artifacts/native_remaining_jax_port_goal/`

Required files:

- [ ] `state.json`: one row per item with status, closure level, owner files,
  oracle contract, upstream source refs, downstream consumers, official-doc
  refs, validation command, and residual risk.
- [ ] `REPORT.md`: final summary with complete/block/skip table.
- [ ] `plans/<id>.md`: item plan and implementation notes.
- [ ] `plans/<id>-coverage.md`: coverage matrix.
- [ ] `plans/<id>-jax-transform.md`: compiled boundary and transform plan.
- [ ] `plans/<id>-invariants.md`: math/API invariants.
- [ ] `plans/<id>-source-refs.md`: current source lines and official-doc
  constraints used for the item.
- [ ] `inventory/<id>.md`: source classification for inventory-only items.
  This may replace `plans/<id>.md` only for inventory-only rows.
- [ ] `red/<id>.txt`: failing pre-implementation test evidence for
  implementation items, or explicit `N/A` with reason for inventory-only,
  blocked, or decision-only items.
- [ ] `bench/<id>.json`: validation and performance/proxy metadata when
  applicable.
- [ ] `restart/<id>.md`: required when an item creates or changes a wrapper,
  frozen state, serialization path, or restartable artifact. Use explicit `N/A`
  with reason otherwise.

## Per-item closure checklist

For every active item:

- [ ] Source audit identifies CPU/C++ implementation, public Python wrappers,
  existing JAX coverage, upstream SIMSOPT behavior/docs, and downstream
  consumers.
- [ ] Classify each public surface as `complete`, `partial`, `portable_gap`,
  `orchestration_only`, `external_solver_wrapper`, `io_visualization`, or
  `skip`.
- [ ] Define an oracle contract from current CPU/C++ behavior before coding.
- [ ] Add or update tests that compare JAX CPU to CPU/C++/SciPy oracle.
- [ ] Add downstream import/use tests when the item changes a public export,
  dispatcher, spec converter, or wrapper route.
- [ ] Run the relevant tests with:
  `JAX_ENABLE_X64=True JAX_PLATFORMS=cpu SIMSOPT_BACKEND_MODE=jax_cpu_parity SIMSOPT_BACKEND_STRICT=1 SIMSOPT_JAX_TRANSFER_GUARD=disallow`.
- [ ] Ensure the compiled path consumes immutable specs and explicit arrays.
- [ ] Ensure no compiled path reads mutable `Optimizable` state.
- [ ] Ensure no silent fallback to C++/SciPy occurs in production JAX paths.
- [ ] Ensure no new mutable module-level caches or cross-thread shared state
  are introduced for compiled kernels.
- [ ] If donation is used, prove the donated positional buffers are not reused
  after the call and document the JAX boundary that owns `donate_argnums`.
- [ ] Inspect lowered/JAXPR or equivalent compiled-boundary evidence when a
  transfer-guard run could miss CPU device-to-host movement.
- [ ] Record numerical tolerance source; do not loosen tolerances to hide
  drift.
- [ ] Update exports only when the public API has real implementation and tests.
- [ ] Record remaining unsupported cases as explicit BLOCKED or SKIPPED rows.
- [ ] Do not use absence from this N01-N14 list as evidence of an unported gap
  until the baseline manifest and current source refs have been checked.

For every inventory-only item:

- [ ] Source audit and downstream inventory are complete.
- [ ] No implementation is attempted unless the human promotes the item to
  `active_scope`.
- [ ] The final row has `status=SKIPPED` or `status=BLOCKED`,
  `closure_level=inventory_only`, and a separate classification such as
  `orchestration_only`, `external_solver_wrapper`, `portable_gap`,
  `io_visualization`, or `skip`, with exact source evidence.

Math, physics, and computation invariants to check where applicable:

- [ ] Coordinate convention and shape convention match the CPU oracle.
- [ ] `nfp`, stellarator symmetry, periodicity, and toroidal/poloidal ordering
  are preserved.
- [ ] Derivative bundles have the same tensor order, axes, units, and
  sign conventions as the CPU/C++ path.
- [ ] Boozer-coordinate fields preserve `G`, `I`, `iota`, `psip`, `modB`,
  derivative, interpolation-domain, and extrapolation semantics.
- [ ] Surface and curve kernels preserve DOF ordering, clamped/masked DOF
  semantics, quadrature grids, normal orientation, area, and volume contracts.
- [ ] Integer/topological outputs, such as linking number, preserve algorithmic
  edge cases and are not advertised as differentiable unless tests prove it.

## Work manifest

### N01_boozer_analytic

Question: can `simsopt.field.boozermagneticfield.BoozerAnalytic` be exposed as
a JAX-native analytic Boozer field?

Owner candidates:

- `src/simsopt/field/boozermagneticfield.py`
- `src/simsopt/field/boozermagneticfield_jax.py`
- `src/simsopt/jax_core/boozer_radial_interp.py`
- new `src/simsopt/jax_core/boozer_analytic.py` if justified

Tasks:

- [ ] Audit `BoozerAnalytic` public methods and CPU equations.
- [ ] Prove whether the proposed JAX route can avoid inherited
  `BoozerMagneticField`/`sopp.BoozerMagneticField` helper paths, including the
  current `sopp.compute_kmnc_kmns` and `sopp.inverse_fourier_transform_*`
  call sites in `src/simsopt/field/boozermagneticfield.py`.
- [ ] Decide whether to add `BoozerAnalyticJAX` or a shared Boozer analytic
  spec consumed by tracing.
- [ ] Add CPU/JAX parity tests for `modB`, derivatives, `G`, `I`, `iota`,
  `psip`, and supported route integration.
- [ ] Validate coordinate, periodicity, and derivative sign conventions against
  the CPU class before routing it into tracing.
- [ ] Keep generic CPU `BoozerMagneticField` rejection if no native JAX wrapper
  exists for the concrete field type.

### N02_interpolated_boozer_field

Question: can `InterpolatedBoozerField` be ported as a JAX-native fixed-grid
Boozer field?

Owner candidates:

- `src/simsopt/field/boozermagneticfield.py`
- `src/simsopt/field/boozermagneticfield_jax.py`
- `src/simsopt/jax_core/boozer_radial_interp.py`
- `src/simsopt/jax_core/regular_grid_interp.py`

Tasks:

- [ ] Audit `InterpolatedBoozerField` C++/Python behavior and stored grid
  layout.
- [ ] Use baseline item `33` / `BoozerRadialInterpolantJAX` as the structural
  precedent: freeze CPU-owned Boozer data into an immutable pytree state before
  calling compiled kernels.
- [ ] Decide whether a frozen immutable grid spec can represent all needed
  public methods.
- [ ] Implement only if the JAX path can evaluate from arrays without CPU
  callbacks.
- [ ] Add parity for field values, derivatives, extrapolation behavior, `nfp`,
  and `stellsym` handling.
- [ ] Add an explicit fixture proving unsupported grid metadata stays BLOCKED
  instead of silently reusing the CPU object.
- [ ] If the CPU class does not expose enough grid metadata, mark BLOCKED with
  exact missing fields.

### N03_boozer_field_route_inventory

Question: after N01/N02, which Boozer field classes can be accepted by JAX
tracing without CPU fallback?

Owner candidates:

- `src/simsopt/field/tracing.py`
- `src/simsopt/field/boozermagneticfield_jax.py`
- `src/simsopt/jax_core/tracing.py`

Tasks:

- [ ] Inventory all concrete `BoozerMagneticField` subclasses.
- [ ] Classify raw CPU Boozer fields separately from immutable JAX wrappers.
- [ ] Inventory non-Boozer field types that share tracing dispatch, such as
  `ToroidalField`, `PoloidalField`, and scalar-potential field routes, and prove
  they do not route through Boozer-only conversion code.
- [ ] Keep `trace_particles_boozer` fail-fast for unsupported CPU field types.
- [ ] Add acceptance tests only for concrete native JAX field wrappers.
- [ ] Add rejection tests for unsupported raw CPU Boozer fields.

### N04_garabedian_henneberg_surfaces

Question: can `SurfaceGarabedian` and `SurfaceHenneberg` get immutable JAX
surface specs and geometry kernels?

Owner candidates:

- `src/simsopt/geo/surfacegarabedian.py`
- `src/simsopt/geo/surfacehenneberg.py`
- `src/simsopt/geo/surfacerzfourier.py`
- `src/simsopt/jax_core/surface_fourier.py`
- `src/simsopt/jax_core/specs.py`

Tasks:

- [ ] Audit CPU parameterization, DOF layout, symmetries, and derivatives.
- [ ] Add spec dataclasses only if formulas can be represented without mutable
  wrapper reads.
- [ ] Add parity for `gamma`, derivatives, normals, area/volume if supported.
- [ ] Include `to_RZFourier()` conversion as oracle-only host behavior unless a
  separate native spec route is proven.
- [ ] Classify unsupported geometry methods explicitly.

### N05_curvexyzfourier_symmetries_spec

Question: close the documented `CurveXYZFourierSymmetries` immutable-spec
routing gap.

Owner candidates:

- `src/simsopt/geo/curvexyzfouriersymmetries.py`
- `src/simsopt/geo/curve.py`
- `src/simsopt/jax_core/curve_geometry.py`
- `src/simsopt/jax_core/specs.py`

Tasks:

- [ ] Preserve the existing pure-JAX `JaxCurve` forward geometry path.
- [ ] Use `src/simsopt/geo/curvexyzfourier.py::JaxCurveXYZFourier` as the
  closest structural precedent before adding a new immutable spec.
- [ ] Add `CurveXYZFourierSymmetriesSpec` if needed.
- [ ] Add `to_spec()` / routing into `curve_spec_from_curve`.
- [ ] Add parity for `gamma`, `gammadash`, higher derivatives, and VJP shape.
- [ ] Remove the documented blocker only after downstream wrapper tests pass.

### N06_surface_xyz_tensor_clamped_dims

Question: close the `SurfaceXYZTensorFourier.surface_spec()` clamped-dims gap.

Owner candidates:

- `src/simsopt/geo/surfacexyztensorfourier.py`
- `src/simsopt/jax_core/surface_fourier.py`
- `src/simsopt/jax_core/specs.py`

Tasks:

- [ ] Audit CPU clamped-dims semantics and DOF masking.
- [ ] Start from the current fail-fast source:
  `src/simsopt/geo/surfacexyztensorfourier.py:188-193`.
- [ ] Extend the JAX spec to preserve clamped DOFs without changing the public
  scalar objective.
- [ ] Prove `surface_spec()` no longer rejects clamped dimensions only after
  parity tests cover all clamped coordinate combinations.
- [ ] Add parity for unclamped and clamped cases.
- [ ] Keep existing fail-fast behavior until parity lands.

### N07_linking_number

Question: port `sopp.compute_linking_number` to a JAX-native kernel.

Owner candidates:

- `src/simsopt/geo/curveobjectives.py`
- `src/simsopt/jax_core/curve_geometry.py`
- new `src/simsopt/jax_core/linking.py` if justified

Tasks:

- [ ] Audit C++ algorithm and numerical assumptions.
- [ ] Add fixed-state JAX kernel with CPU/C++ parity tests.
- [ ] Decide whether gradients are required; if not, document non-differentiable
  or integer-valued contract.
- [ ] Include crossing/near-degenerate edge cases from the C++ assumptions.
- [ ] Wire the accepted kernel into `LinkingNumber.J` in
  `src/simsopt/geo/curveobjectives.py` behind the existing
  `simsopt.backend.runtime.is_jax_backend()` contract only after parity and
  edge-case tests pass.
- [ ] Preserve the existing target-lane bypass guard pattern used by nearby
  curve objectives: `raise_if_target_lane_bypass(...)` before legacy entry,
  with a route-specific name for `LinkingNumber.J`.
- [ ] Preserve the CPU `sopp.compute_linking_number` oracle path and prove the
  JAX route does not silently fall back to it in strict backend mode.

### N08_mhd_fixed_output_postprocessing

Question: which native MHD post-processing functions can be ported for fixed
VMEC/Boozer/SPEC output arrays?

Default status: inventory-only. Promote to `active_scope` only after selecting
specific pure array kernels. Do not launch a broad MHD rewrite.

Owner candidates:

- `src/simsopt/mhd/boozer.py`
- `src/simsopt/mhd/vmec.py`
- `src/simsopt/mhd/spec.py`
- `src/simsopt/mhd/virtual_casing.py`
- `src/simsopt/mhd/vmec_diagnostics.py`
- `src/simsopt/mhd/bootstrap.py`
- `src/simsopt/mhd/profiles.py`

Tasks:

- [ ] Classify VMEC/SPEC/BOOZ_XFORM execution as external and out of scope.
- [ ] Record explicit inventory rows for `Boozer`, `Quasisymmetry`, `Vmec`,
  `Spec`, and virtual-casing wrapper surfaces as `external_solver_wrapper` or
  host orchestration, not implementation targets.
- [ ] Cite the official SIMSOPT MHD docs and current source lines proving which
  classes are external solver interfaces.
- [ ] Identify pure array post-processing kernels, such as VMEC geometry and
  Redl/bootstrap formulas.
- [ ] Port only fixed-output kernels that can consume immutable arrays/specs.
- [ ] Add parity tests using fixture arrays or existing test files; do not run
  external solvers inside JAX validation.
- [ ] Leave plotting, file I/O, and solver-launch paths as host-only.

### N09_magnetic_field_composition

Question: can generic field composition (`MagneticFieldSum`,
`MagneticFieldMultiply`) route over JAX-native component fields?

Owner candidates:

- `src/simsopt/field/magneticfield.py`
- `src/simsopt/field/_jax_common.py`
- `src/simsopt/jax_core/field.py`
- new `src/simsopt/jax_core/magneticfield_composition.py` if justified

Tasks:

- [ ] Audit current composition behavior and cache invalidation.
- [ ] Document how `MagneticField` point caches, `_set_points_cb`, and parent
  invalidation interact with immutable JAX specs before adding composition
  routing.
- [ ] Treat `src/simsopt/jax_core/field.py` as grouped-coil field-kernel
  precedent, not evidence that `MagneticFieldSum` or `MagneticFieldMultiply`
  already have native JAX entry points.
- [ ] Add JAX composition only for component fields with native JAX specs.
- [ ] Preserve fail-fast behavior when any component is CPU-only.
- [ ] Add parity for `B`, `dB_by_dX`, `A`, and derivative bundles when
  supported.

### N10_generic_solver_orchestration_inventory

Question: should generic `serial.py` / `mpi.py` SciPy solve wrappers get a JAX
opt-in path, or remain host orchestration?

Default status: inventory-only. A generic optimizer/orchestration rewrite is a
separate architecture goal unless the human promotes a narrow contract.

Owner candidates:

- `src/simsopt/solve/serial.py`
- `src/simsopt/solve/mpi.py`
- `src/simsopt/geo/optimizer_jax.py`
- `src/simsopt/geo/optimizer_jax_private/`
- `src/simsopt/solve/permanent_magnet_optimization_jax.py`
- `src/simsopt/solve/wireframe_optimization_jax.py`

Tasks:

- [ ] Inventory generic solve wrappers and downstream callers.
- [ ] Inventory already-shipped strict opt-in wrappers,
  `permanent_magnet_optimization_jax.py` and `wireframe_optimization_jax.py`,
  as precedent before proposing any generic solve route.
- [ ] Decide whether any pure-JAX objective family justifies a generic opt-in
  wrapper.
- [ ] Do not replace SciPy/MPI host orchestration globally.
- [ ] If implementing, add a strict opt-in path that accepts only native JAX
  objective contracts and rejects CPU objectives.
- [ ] Otherwise mark `orchestration_only` with rationale.

### N11_live_pm_grid_host_workflow_decision

Question: close or explicitly defer the live mutable `PermanentMagnetGrid`
host-loop workflow gap.

Owner candidates:

- `src/simsopt/geo/permanent_magnet_grid.py`
- `src/simsopt/geo/permanent_magnet_grid_jax.py`
- `src/simsopt/solve/permanent_magnet_optimization.py`
- `src/simsopt/solve/permanent_magnet_optimization_jax.py`

Tasks:

- [ ] Re-read the existing item-28 deferred note in
  `.artifacts/jax_port_goal/state.json`.
- [ ] Decide whether to implement an immutable batch replacement or keep the
  live mutating workflow host-only.
- [ ] Pick one final closure label explicitly: `BLOCKED` for a root missing
  capability, or `SKIPPED` with a deferred-decision marker when the supported
  native contract remains immutable `PermanentMagnetGridJAX`.
- [ ] Do not silently export a CPU-style dispatcher as JAX.
- [ ] If implemented, add parity for state updates, objective history, and
  solved magnet vectors.
- [ ] If deferred, record why immutable `PermanentMagnetGridJAX` is the
  supported native contract.

### N12_qfm_surface_host_orchestration_inventory

Question: is `QfmSurface` a JAX-port target, or host solver orchestration around
already-ported residual kernels?

Default status: inventory-only. Promote only if the human asks for a scoped
JAX-native QFM solve wrapper.

Owner candidates:

- `src/simsopt/geo/qfmsurface.py`
- `src/simsopt/geo/surfaceobjectives.py`
- `src/simsopt/geo/surfaceobjectives_jax.py`

Tasks:

- [ ] Record `QfmSurface` as host orchestration if its live contract mutates
  `surface.x` and calls SciPy minimization.
- [ ] Cite the current mutation and SciPy call sites:
  `src/simsopt/geo/qfmsurface.py:49`, `src/simsopt/geo/qfmsurface.py:73`,
  `src/simsopt/geo/qfmsurface.py:97`, `src/simsopt/geo/qfmsurface.py:133`,
  and `src/simsopt/geo/qfmsurface.py:170`.
- [ ] Verify the existing `QfmResidualJAX` coverage before opening any new QFM
  physics/residual port task.
- [ ] Classify any remaining gap as solver orchestration, not a missing residual
  kernel, unless current source proves otherwise.
- [ ] Leave this item `SKIPPED` unless a narrow immutable-state QFM solve
  contract is promoted to `active_scope`.

### N13_mgrid_io_inventory

Question: is `MGrid` JAX-portable numerical work, or MAKEGRID file I/O and host
data loading?

Default status: inventory-only.

Owner candidates:

- `src/simsopt/field/mgrid.py`

Tasks:

- [ ] Classify `MGrid` public methods as `io_visualization`,
  `orchestration_only`, or fixed-array post-processing with source refs.
- [ ] Include downstream MGrid import/use coverage from
  `tests/field/test_mgrid.py`, `tests/field/test_magneticfields.py`, and
  `tests/field/test_coilset.py` before deciding it has no JAX consumer.
- [ ] Do not port file parsing/writing paths to JAX.
- [ ] Promote only if a pure fixed-array kernel is found that downstream JAX
  routes actually consume.

### N14_fourier_interpolation_utility_inventory

Question: is `util/fourier_interpolation.py` on a current JAX production route,
or only a host utility/test helper?

Default status: inventory-only.

Owner candidates:

- `src/simsopt/util/fourier_interpolation.py`
- `tests/util/test_fourier_interpolation.py`

Tasks:

- [ ] Inventory all source and test consumers before classifying this as a
  Boozer or field-route blocker.
- [ ] If it is only a NumPy host utility, mark `orchestration_only` or `skip`
  with source refs.
- [ ] Prove any claimed Boozer dependency with current `src/simsopt` references,
  not test-only references.
- [ ] Promote only if a compiled JAX production route actually depends on this
  interpolation behavior.

## Final validation gate

Do not report success until:

- [ ] Every active item is COMPLETE, BLOCKED, or SKIPPED in
  `.artifacts/native_remaining_jax_port_goal/state.json`.
- [ ] Every inventory-only item is recorded with `closure_level=inventory_only`
  and exact source/docs evidence.
- [ ] `.artifacts/native_remaining_jax_port_goal/REPORT.md` has a summary table.
- [ ] `git diff --check` passes for touched files.
- [ ] Relevant tests pass under strict CPU transfer guard.
- [ ] No new GPU/CUDA claim appears unless `scope_profile=cuda_perf_release`.
- [ ] No completed item relies on CPU fallback in the production JAX route.
- [ ] Official-source refs are recorded for every COMPLETE or BLOCKED item.
- [ ] Downstream public imports and examples touched by the item still pass or
  are explicitly out of scope with source evidence.

Suggested strict CPU validation command pattern:

```bash
JAX_ENABLE_X64=True \
JAX_PLATFORMS=cpu \
SIMSOPT_BACKEND_MODE=jax_cpu_parity \
SIMSOPT_BACKEND_STRICT=1 \
SIMSOPT_JAX_TRANSFER_GUARD=disallow \
.conda/jax-0.9.2/bin/python -m pytest -q <item-specific-tests>
```

Suggested provenance command:

```bash
JAX_PLATFORMS=cpu .conda/jax-0.9.2/bin/python - <<'PY'
import jax
import jaxlib

print("jax", jax.__version__)
print("jaxlib", jaxlib.__version__)
print("devices", jax.devices())
PY
```

Expected CPU-only provenance shape:

```text
jax <version>
jaxlib <version>
devices [CpuDevice(id=0)]
```

Accept equivalent current-JAX CPU reprs only if every reported device has
`platform == "cpu"` and no GPU/CUDA device appears.

Additional CUDA validation is allowed only under `scope_profile=cuda_perf_release`
and must include:

```bash
nvidia-smi
nvidia-smi -L
CUDA_VISIBLE_DEVICES=<approved-devices> \
JAX_ENABLE_X64=True \
JAX_PLATFORMS=cuda \
SIMSOPT_BACKEND_MODE=jax_gpu_parity \
SIMSOPT_BACKEND_STRICT=1 \
SIMSOPT_JAX_TRANSFER_GUARD=disallow \
.conda/jax-0.9.2/bin/python -m pytest -q <item-specific-gpu-tests>
```

## Reporting format

Final answer must start with one of:

- `COMPLETE: all active native remaining items closed`
- `PARTIAL: some active native remaining items blocked/skipped`
- `BLOCKED: no active item could proceed`

Then include:

- item status table,
- exact test commands and results,
- files changed,
- official docs/source refs used,
- remaining unsupported boundaries,
- whether CUDA was claimed (`yes` or `no`).
~~~

## Launch checklist

- [ ] Confirm the user wants to start this as a new `/goal` run.
- [ ] Select `active_scope`.
- [ ] Decide whether to promote any inventory-only item from
  `inventory_only_scope` to `active_scope`.
- [ ] Keep `scope_profile=cpu_port_closure` unless real CUDA proof is explicitly
  requested.
- [ ] Start from a clean understanding of dirty worktree scope; do not revert
  unrelated files.
