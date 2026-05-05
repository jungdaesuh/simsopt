# banana coil optimization — CPU/C++ (`simsoptpp`) dependency manifest

## Scope

This note traces the legacy CPU/C++ execution path — `simsoptpp` pybind11 bindings plus pure-Python `simsopt` wrappers — for both banana product entrypoints:

- Stage 2: `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py`
- Single-stage: `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`

It is the inverse of the JAX/GPU dependency trace at `docs/single_stage_banana_jax_gpu_dependency_trace_2026-04-13.md`. The JAX trace asks "what runs on device when `SIMSOPT_BACKEND=jax`?". This doc asks "what runs through `simsoptpp` when `SIMSOPT_BACKEND` is unset or `cpu`?".

Companion docs:
- `docs/banana_single_stage_stage2_lavish_validation_plan_2026-04-27.md` — product-surface inventory and validation lanes
- `/Users/suhjungdae/code/columbia/analysis/jax_gpu_port_dependency_graph_2026-04-17.md` — JAX-port-centric module graph that lives in the sibling `columbia/analysis` repo and lists each C++ replacement target

## Surrounding source files

Beyond the two `.py` entrypoints, the banana product ships:

- **Shell drivers** — `examples/single_stage_optimization/STAGE_2/banana-scan.sh`, `examples/single_stage_optimization/SINGLE_STAGE/single-scan.sh` (parameter scan launchers around the Python entrypoints)

Stage 2 → Single-Stage handoff is performed at runtime via JSON state files written by Stage 2 and rehydrated by Single-Stage through `load(stage2_bs_path)`; those files are runtime data artifacts, not source.

## Backend selection — when this lane is active

The CPU/C++ lane is active when `args.backend != "jax"`, which is the default if `SIMSOPT_BACKEND` is unset. Both entrypoints accept `--backend` and fall back to the env var.

Stage 2 routing branch:

- `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py:726-732` — `resolve_stage2_default_optimizer_backend()` returns `"scipy"` when field backend is not JAX
- `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py:730` — `if field_backend == "jax":` (optimizer-backend resolver)
- `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py:1812` — second `if field_backend == "jax":` guard inside `resolve_optimizer_contract_for_stage2_outer_loop`
- `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py:2830-2847` — primary objective lane branch; CPU `else` branch constructs `Jf = SquaredFlux(new_surf, new_bs)` (line 2846) and prints `Stage 2 backend: CPU (simsoptpp)` (line 2847)

Single-stage routing branches:

- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:1717` — `if args.backend == "jax":` (early backend gate)
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:10515` — `use_jax = args.backend == "jax"` (canonical lane flag used downstream)
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:4951-4959` — `BoozerCls = BoozerSurfaceJAX` vs `BoozerCls = BoozerSurface`
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:7486` — CPU branch returns the legacy `BoozerResidual` class
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:10687` — CPU `iota_cls = Iotas`
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:11203-11217` — CPU `else` branch builds `bs_obj = BiotSavart(coils)`, `NonQuasiSymmetricRatio`, and the legacy Iotas/BoozerResidual graph

The `is_jax_backend()` helper from `src/simsopt/backend/runtime.py` exists and is referenced by lower-level modules, but neither entrypoint uses it directly — both gate on `args.backend == "jax"` or the local `use_jax` flag.

## Stage 2 — CPU/C++ dependency spine

Entrypoint top-level imports (CPU-relevant subset):

- `repo_bootstrap.py` — `bootstrap_local_simsopt()`, `configure_entrypoint_jax_runtime()` for the resolved local source tree (loaded for both lanes; `simsoptpp` pybind11 module is imported here)
- `src/simsopt/field/__init__.py` — `BiotSavart`, `Coil`, `Current`, `coils_via_symmetries`
- `src/simsopt/objectives/__init__.py` — `SquaredFlux`, `QuadraticPenalty`
- `src/simsopt/geo/__init__.py` — `CurveCWSFourier`/`CurveXYZFourier` constructors, `SurfaceRZFourier`, `SurfaceXYZTensorFourier`, `create_equally_spaced_curves`
- `src/simsopt/geo/curveobjectives.py` — `CurveLength`, `LpCurveCurvature`, `CurveCurveDistance`, `CurveSurfaceDistance`
- `src/simsopt/_core/jax_host_boundary.py` — `host_array`, `host_float` boundary helpers (used even on the CPU lane to materialize JAX-typed scalars from internal pure helpers)

Runtime spine on the CPU/C++ lane:

1. **Coil construction**
   - `create_equally_spaced_curves(...)` plus `Current(...)` and `Coil(curve, current)` build the optimizable coil graph
   - `coils_via_symmetries(...)` applies `nfp` and stellarator symmetry
   - all curve geometry is C++-backed (`sopp.CurveXYZFourier` etc.)
2. **Field backend wrap**
   - `bs = BiotSavart(coils)` then `bs.set_points(surf.gamma().reshape((-1, 3)))`
   - implementation: `src/simsopt/field/biotsavart.py:10` — `class BiotSavart(sopp.BiotSavart, MagneticField):`
3. **Objective assembly** at `banana_coil_solver.py:2846` (CPU else-branch)
   - `Jf = SquaredFlux(new_surf, new_bs)` from `src/simsopt/objectives/fluxobjective.py`
   - `Jls = CurveLength(new_banana_curve)` from `src/simsopt/geo/curveobjectives.py`
   - `Jccdist = CurveCurveDistance(new_curves, CC_THRESHOLD)`
   - `Jcsdist = CurveSurfaceDistance(new_curves, new_surf, CS_THRESHOLD)`
   - `Jc = LpCurveCurvature(new_banana_curve, args.curvature_p_norm, CURVATURE_THRESHOLD)`
   - `JF = SQUARED_FLUX_WEIGHT * Jf + LENGTH_WEIGHT * Jls_penalty + CC_WEIGHT * Jccdist + CC_WEIGHT * Jcsdist + CURVATURE_WEIGHT * Jc` — composition via `_core/optimizable.py` operator overloads
4. **Outer optimizer**
   - `resolve_stage2_default_optimizer_backend(field_backend, ...)` returns `"scipy"` for the CPU lane (`banana_coil_solver.py:726-732`)
   - the timed driver `run_stage2_optimizer_timed(...)` ultimately calls `scipy.optimize.minimize` (L-BFGS-B by default) or `scipy.optimize.least_squares` for the LS variant

## Single-stage — CPU/C++ dependency spine

Entrypoint top-level imports (CPU-relevant subset):

- `single_stage_banana_example.py:47-50` — `from alm_utils import ALMSettings, minimize_alm, ...` — outer ALM loop driver, shared between lanes
- `single_stage_banana_example.py:80` — `from banana_opt.single_stage_objectives import evaluate_alm_objective` plus the rest of `examples/single_stage_optimization/banana_opt/*.py` (hardware contracts, current contracts, single-stage constraints)
- `single_stage_banana_example.py:86` — `from simsopt.field import BiotSavart`
- `single_stage_banana_example.py:93` — `BoozerSurface` from `src/simsopt/geo/boozersurface.py`
- `single_stage_banana_example.py:115-117` — `BoozerResidual`, `Iotas`, `NonQuasiSymmetricRatio` from `src/simsopt/geo/surfaceobjectives.py`
- `examples/single_stage_optimization/hardware_constraints.py`, `plotting_utils.py`, `run_metadata.py` — shared host helpers

Runtime spine on the CPU/C++ lane:

1. **Stage-2 seed and warm-start load**
   - `build_stage2_bs_path(args)` at `single_stage_banana_example.py:1835`
   - `load_stage2_results(stage2_bs_path)` at `single_stage_banana_example.py:1901-1902` reads the sibling `results.json`
   - `load_single_stage_warm_start_state(run_dir)` at `single_stage_banana_example.py:2107` rehydrates Python/SIMSOPT objects from JSON
   - both routes go through `src/simsopt/_core/json.py::load(...)` which deserializes `Optimizable` graphs including the `BiotSavart` field
2. **Field backend wrap**
   - `bs_obj = BiotSavart(coils)` at `single_stage_banana_example.py:11206` (CPU else-branch)
   - same class as Stage 2: `src/simsopt/field/biotsavart.py:10`
   - VJP plumbing: `dJ_by_dcoils = self.biotsavart.B_vjp(dJ_by_dB)` at `single_stage_banana_example.py:4767` calls `sopp.biot_savart_vjp_graph` through `BiotSavart.B_vjp`
3. **Boozer surface initialization**
   - class selection: `BoozerCls = BoozerSurface` at `single_stage_banana_example.py:4959`
   - constructions at `single_stage_banana_example.py:4990, 5000, 5025, 5035` for the four supported (LS-penalty / exact) × (with-volume / with-toroidal-flux) combinations
   - implementation: `src/simsopt/geo/boozersurface.py` — `class BoozerSurface(Optimizable)`
   - inner solvers in the same file: `minimize_boozer_penalty_constraints_LBFGS()`, `minimize_boozer_penalty_constraints_newton()`, `minimize_boozer_penalty_constraints_ls()`, `minimize_boozer_exact_constraints_newton()`
   - residual kernels are dispatched to C++ via wrappers: `_call_boozer_residual`, `_call_boozer_residual_ds`, `_call_boozer_residual_ds2`
4. **Outer objective assembly** at `single_stage_banana_example.py:11214-11259`
   - `nonQSs = [NonQuasiSymmetricRatio(boozer_surface, bs_obj)]` (line 11217)
   - `brs = [build_boozer_residual_objective(boozer_surface, bs_obj, boozer_residual_cls)]` (line 11218); `boozer_residual_cls` resolves to `BoozerResidual` on the CPU lane via the dispatch at line 7486
   - `iota = build_iota_objective(boozer_surface, Iotas)` and `Jiota = QuadraticPenalty(iota, iota_target)`
   - `JCurveLength = QuadraticPenalty(CurveLength(banana_curves[0]), length_target, "max")`
   - `JCurveCurve = CurveCurveDistance(curves, CC_DIST)`
   - `JCurveSurface = CurveSurfaceDistance(curves, boozer_surface.surface, CS_DIST)`
   - `JSurfSurf = SurfaceSurfaceDistance(boozer_surface.surface, VV, SS_DIST)`
   - `JCurvature = LpCurveCurvature(banana_curves[0], 2, CURVATURE_THRESHOLD)`
   - composite `JF = JnonQSRatio + RES_WEIGHT*JBoozerResidual + IOTAS_WEIGHT*Jiota + LENGTH_WEIGHT*JCurveLength + CC_WEIGHT*JCurveCurve + CS_WEIGHT*JCurveSurface + SURF_DIST_WEIGHT*JSurfSurf + CURVATURE_WEIGHT*JCurvature` (lines 11249-11259)
5. **Outer ALM loop**
   - `minimize_alm(...)` call at `single_stage_banana_example.py:11776`
   - implementation: `examples/single_stage_optimization/alm_utils.py`
   - inner step calls `scipy.optimize.minimize` against the augmented Lagrangian penalty form built from `JF`
6. **Hardware-constraint and self-intersection gates**
   - hardware constraints evaluated by `examples/single_stage_optimization/hardware_constraints.py` and the `banana_opt/*.py` contracts; distance constraints reuse the same `CurveCurveDistance`, `CurveSurfaceDistance`, `SurfaceSurfaceDistance` objectives instantiated above
   - self-intersection gate uses `surface.is_self_intersecting()` — a host validation gate that is identical on both lanes
7. **Final results assembly**
   - host `JF.J()` / per-term `.J()` calls populate the final results JSON; on the CPU lane this is the same compute path as the optimizer, not a special reporting branch

## Python ↔ C++ binding map

All bindings live in `src/simsoptpp/python.cpp` (and the per-module `python_*.cpp` registrars). C++ implementation files are in `src/simsoptpp/` alongside the binding wrappers.

### Biot-Savart — field, gradient, VJP

| Python wrapper | C++ source | `sopp.<binding>` | Used by |
|---|---|---|---|
| `BiotSavart.compute(...)`, `.B()`, `.dB_by_dX()` (`src/simsopt/field/biotsavart.py:10`) | `biot_savart_py.cpp`, `biot_savart_c.cpp`, `biot_savart_impl.h`, `magneticfield_biotsavart.cpp/.h` | `sopp.biot_savart` (`python.cpp:57`), `sopp.biot_savart_B` (`python.cpp:58`), `sopp.BiotSavart` class (`python_magneticfield.cpp`) | both entrypoints |
| `BiotSavart.B_vjp(v)`, `BiotSavart.B_and_dB_vjp(v, vgrad)` | `biot_savart_vjp_py.cpp`, `biot_savart_vjp_c.cpp`, `biot_savart_vjp_impl.h` | `sopp.biot_savart_vjp_graph` (`python.cpp:60`) | `SquaredFlux.dJ()`, single-stage VJP at `single_stage_banana_example.py:4767` |
| `BiotSavart.dB_by_dcoilcurrents()`, `BiotSavart.dA_by_dcoilcurrents()` | `biot_savart_impl.h` | `sopp.BiotSavart` cache methods | both |

### Surface and curve geometry

| Python wrapper | C++ source | `sopp.<binding>` | Used by |
|---|---|---|---|
| `SurfaceRZFourier` (`src/simsopt/geo/surfacerzfourier.py`) | `surfacerzfourier.cpp`, `surface.cpp` | `sopp.SurfaceRZFourier` (registered in `python_surfaces.cpp`) | single-stage plasma surface, vessel surface |
| `SurfaceXYZTensorFourier` (`src/simsopt/geo/surfacexyztensorfourier.py`) | `surfacexyzfourier.cpp`, `surfacexyztensorfourier.h`, `surface.cpp` | `sopp.SurfaceXYZTensorFourier` | single-stage Boozer surface |
| `CurveXYZFourier`, `CurveRZFourier`, `CurvePlanarFourier` | `curvexyzfourier.cpp`, `curverzfourier.cpp`, `curveplanarfourier.cpp`, `curve.cpp/.h` | `sopp.CurveXYZFourier`, `sopp.CurveRZFourier`, `sopp.CurvePlanarFourier` (`python_curves.cpp`) | both entrypoints (TF + banana coils) |
| `Curve.gamma`, `gammadash`, `kappa`, `incremental_arclength`, `dgamma_by_dcoeff_vjp`, `dkappa_by_dcoeff_vjp` | `curve.cpp` | inherited from `sopp.Curve*` classes | curve objectives, Boozer residual, distance objectives |
| `Surface.area`, `Surface.volume`, `Surface.normal`, `Surface.gamma`, `gamma1`, `gamma2`, `is_self_intersecting` | `surface.cpp` | inherited from `sopp.Surface*` classes | both entrypoints |

### Boozer residual

| Python wrapper | C++ source | `sopp.<binding>` | Used by |
|---|---|---|---|
| `BoozerSurface.{minimize_boozer_*}` and the `_call_boozer_residual*` helpers in `src/simsopt/geo/boozersurface.py` | `boozerresidual_py.cpp`, `boozerresidual_impl.h` | `sopp.boozer_residual` (`python.cpp:136`), `sopp.boozer_residual_ds` (`python.cpp:137`), `sopp.boozer_residual_ds2` (`python.cpp:138`) | single-stage inner solve |
| `BoozerResidual.J()` / `dJ()` (`src/simsopt/geo/surfaceobjectives.py`) | `boozerresidual_impl.h` (residual) + `boozer_dresidual_dc` lambda in `python.cpp:106` | `sopp.boozer_dresidual_dc` (`python.cpp:106`) | single-stage outer objective |

### Flux and label objectives

| Python wrapper | C++ source | `sopp.<binding>` | Used by |
|---|---|---|---|
| `SquaredFlux.J()` (`src/simsopt/objectives/fluxobjective.py`) | `integral_BdotN.cpp`, `integral_BdotN.h` | `sopp.integral_BdotN` (`python.cpp:91`) | Stage 2 primary objective |
| `Volume`, `Area`, `ToroidalFlux` (`src/simsopt/geo/surfaceobjectives.py`) | `surface.cpp` (label methods on the C++ Surface class) | inherited from `sopp.Surface*` | single-stage Boozer label constraints |

### Distance objectives

| Python wrapper | C++ source | `sopp.<binding>` | Used by |
|---|---|---|---|
| `CurveCurveDistance.compute_candidates()` (`src/simsopt/geo/curveobjectives.py`) — point-cloud culling | `python_distance.cpp` | `sopp.get_pointclouds_closer_than_threshold_within_collection`, `..._between_two_collections` | both entrypoints |
| `CurveCurveDistance`, `CurveSurfaceDistance`, `SurfaceSurfaceDistance` final J/dJ | host Python on top of culled point clouds; sums of pairwise smoothed distance penalties | (uses C++ outputs above) | both entrypoints |

Note: the `Curve*Distance` and `Surface*Distance` `J`/`dJ` value/gradient code is host Python (with optional `@jit`-decorated pure kernels that compile to CPU when JAX platform is `cpu`); only the candidate-pruning step is a pybind11 call. This is the same on both lanes.

## Shared host-side machinery (CPU lane)

These modules are pure Python and are exercised on the CPU lane regardless of `SIMSOPT_BACKEND`.

Core glue:
- `src/simsopt/_core/optimizable.py` — `Optimizable` base, dependency DAG, DOF management, operator overloads (`__add__`, `__mul__`) used to compose `JF`
- `src/simsopt/_core/derivative.py` — `Derivative` and `@derivative_dec`; the chain-rule glue that calls per-objective `dJ()` and combines partials
- `src/simsopt/_core/dofs.py` — flat DOF index/extraction
- `src/simsopt/_core/json.py` — `save()` / `load()` for `Optimizable` graphs (single-stage seed and warm-start path uses `load()`)
- `src/simsopt/_core/jax_host_boundary.py` — `host_array`, `host_float`, `host_bool` (used even on the CPU lane because some pure kernels in `curveobjectives.py` are `@jit`-decorated and run on the CPU JAX platform; the helpers materialize their outputs back to plain NumPy/Python scalars)
- `src/simsopt/_core/{util,types,dev,descriptor}.py` — utility infrastructure pulled in transitively

Field side:
- `src/simsopt/field/coil.py` — `Coil`, `Current`, `coils_via_symmetries`
- `src/simsopt/field/magneticfield.py`, `magneticfieldclasses.py` — base classes that `BiotSavart` inherits from on top of `sopp.BiotSavart`
- `src/simsopt/field/tracing.py` — fieldline / Poincaré tracing (imported by Stage 2 but unused in the optimizer hot path)

Geometry side:
- `src/simsopt/geo/{surface,curve}.py` — Python sides of the C++ Surface / Curve hierarchies
- `src/simsopt/geo/{surfacerzfourier,surfacexyztensorfourier,surfacexyzfourier,curvexyzfourier,curvecwsfourier}.py` — concrete Python wrappers
- `src/simsopt/geo/jit.py` — JAX `jit` helper used by curve objectives' pure kernels
- `src/simsopt/geo/plotting.py`, `src/simsopt/geo/curveobjectives.py`, `src/simsopt/geo/surfaceobjectives.py`, `src/simsopt/geo/boozersurface.py` — full CPU-lane Boozer + objective stack

Objectives side:
- `src/simsopt/objectives/fluxobjective.py` — `SquaredFlux`
- `src/simsopt/objectives/utilities.py` — `QuadraticPenalty`, `forward_backward` (the latter used by `BoozerResidualExact` adjoint solves)
- `src/simsopt/objectives/least_squares.py`, `constrained.py` — least-squares and constrained-objective utilities

Util side:
- `src/simsopt/util/{constants,mpi,logger}.py` — physical constants, MPI helpers, logging

Single-stage-specific drivers:
- `examples/single_stage_optimization/alm_utils.py` — `ALMSettings`, `minimize_alm`, augmented-Lagrangian outer loop
- `examples/single_stage_optimization/banana_opt/*.py` — hardware contracts, current contracts, constraint schemas, smoothing helpers, reference surfaces
- `examples/single_stage_optimization/hardware_constraints.py` — hardware feasibility evaluation
- `examples/single_stage_optimization/{plotting_utils,run_metadata}.py` — artifact writing and provenance

Outer optimizer:
- SciPy `scipy.optimize.minimize` (L-BFGS-B) and `scipy.optimize.least_squares` (Trust-Region)

Both `src/simsopt/geo/__init__.py` and `src/simsopt/field/__init__.py` are wildcard barrel imports that pull every sibling module at startup; the static import closure includes additional modules (e.g. `dipole_field`, `magneticfield_wireframe`, MHD wrappers) that are loaded but not exercised by the banana entrypoints.

## What is NOT exercised on the CPU lane

Modules that are imported (sometimes unconditionally, since the JAX runtime is initialized for both lanes) but not called when `args.backend != "jax"`:

- `src/simsopt/field/biotsavart_jax.py`, `src/simsopt/field/biotsavart_jax_backend.py` — `BiotSavartJAX`, `SingleStageRuntimeSpecBiotSavartJAX`
- `src/simsopt/objectives/fluxobjective_jax.py` — `SquaredFluxJAX` (Stage 2)
- `src/simsopt/objectives/stage2_target_objective_jax.py` — `Stage2TargetObjectiveBundle` (Stage 2 ondevice lane)
- `src/simsopt/objectives/integral_bdotn_jax.py`
- `src/simsopt/geo/boozersurface_jax.py` — `BoozerSurfaceJAX` (single-stage)
- `src/simsopt/geo/boozer_residual_jax.py`, `src/simsopt/geo/label_constraints_jax.py`, `src/simsopt/geo/surface_fourier_jax.py`
- `src/simsopt/geo/surfaceobjectives_jax.py` — `BoozerResidualJAX`, `IotasJAX`, `NonQuasiSymmetricRatioJAX`
- `src/simsopt/geo/optimizer_jax.py` and `src/simsopt/geo/optimizer_jax_private/*` — `target_minimize`, ondevice BFGS / L-BFGS, line searches
- `src/simsopt/jax_core/*` — pure-JAX kernel layer (`biotsavart.py`, `curve_geometry.py`, `surface_rzfourier.py`, `field.py`, `specs.py`)
- `src/simsopt/backend/runtime.py` JAX-platform configuration paths beyond initialization

The JAX runtime is initialized regardless (because `repo_bootstrap.py` calls `configure_entrypoint_jax_runtime(...)` early), but no `jax.jit`-traced functions are executed on the CPU compute path beyond a few `@jit`-decorated pure helpers in `curveobjectives.py` that compile to the CPU JAX platform. The C++ kernels do not depend on JAX in any way.

## Routing summary

| Stage | CPU/C++ class / entry | JAX class / entry | Routing site |
|---|---|---|---|
| Stage 2 field | `BiotSavart` (`src/simsopt/field/biotsavart.py:10`) | `BiotSavartJAX` | `banana_coil_solver.py:2830` |
| Stage 2 flux objective | `SquaredFlux` (`src/simsopt/objectives/fluxobjective.py`) | `SquaredFluxJAX` | `banana_coil_solver.py:2830-2847` |
| Stage 2 outer optimizer | `scipy.optimize.minimize` / `scipy.optimize.least_squares` | `target_minimize` (ondevice) | `banana_coil_solver.py:726-732` |
| Single-stage field | `BiotSavart` | `SingleStageRuntimeSpecBiotSavartJAX` | `single_stage_banana_example.py:11206` |
| Single-stage Boozer | `BoozerSurface` (`src/simsopt/geo/boozersurface.py`) | `BoozerSurfaceJAX` | `single_stage_banana_example.py:4951-4959` |
| Single-stage outer objectives | `BoozerResidual`, `Iotas`, `NonQuasiSymmetricRatio` (`src/simsopt/geo/surfaceobjectives.py`) | `BoozerResidualJAX`, `IotasJAX`, `NonQuasiSymmetricRatioJAX` | `single_stage_banana_example.py:7486, 10687, 11215-11217` |
| Single-stage ALM outer loop | `minimize_alm` (`alm_utils.py`) using `scipy.optimize.minimize` inside | same `minimize_alm` driver, JAX-backed inner step | `single_stage_banana_example.py:11776` (lane-agnostic) |

## Build-time C++ dependency closure

The banana product loads exactly one pybind11 extension, `simsoptpp`. The full `pybind11_add_module` source list lives in `CMakeLists.txt` and includes more files than the banana lane actually exercises. For completeness:

```
src/simsoptpp/python.cpp
src/simsoptpp/python_surfaces.cpp
src/simsoptpp/python_curves.cpp
src/simsoptpp/python_magneticfield.cpp
src/simsoptpp/python_tracing.cpp
src/simsoptpp/python_distance.cpp
src/simsoptpp/python_boozermagneticfield.cpp
src/simsoptpp/boozerresidual_py.cpp
src/simsoptpp/biot_savart_py.cpp
src/simsoptpp/biot_savart_vjp_py.cpp
src/simsoptpp/regular_grid_interpolant_3d_py.cpp
src/simsoptpp/curve.cpp
src/simsoptpp/curverzfourier.cpp
src/simsoptpp/curvexyzfourier.cpp
src/simsoptpp/curveplanarfourier.cpp
src/simsoptpp/surface.cpp
src/simsoptpp/surfacerzfourier.cpp
src/simsoptpp/surfacexyzfourier.cpp
src/simsoptpp/integral_BdotN.cpp
src/simsoptpp/magneticfield_biotsavart.cpp
src/simsoptpp/tracing.cpp
src/simsoptpp/boozerradialinterpolant.cpp
src/simsoptpp/magneticfield_wireframe.cpp
src/simsoptpp/dipole_field.cpp
src/simsoptpp/permanent_magnet_optimization.cpp
src/simsoptpp/dommaschk.cpp
src/simsoptpp/reiman.cpp
src/simsoptpp/wireframe_optimization.cpp
```

The banana lane actively exercises only: `python.cpp`, `python_curves.cpp`, `python_surfaces.cpp`, `python_magneticfield.cpp`, `python_distance.cpp`, `curve.cpp`, `surface.cpp`, `surfacerzfourier.cpp`, `surfacexyzfourier.cpp`, `magneticfield_biotsavart.cpp`, `biot_savart_py.cpp`, `biot_savart_vjp_py.cpp`, `boozerresidual_py.cpp`, `integral_BdotN.cpp` (plus their `_c.cpp` / `_impl.h` siblings). The remaining sources (tracing, wireframe, dipole field, permanent-magnet optimization, Dommaschk, Reiman, Boozer magnetic-field interpolants) are linked into `simsoptpp` but unused by either banana entrypoint.

## Caveats

- This trace was written against `gpu-purity-stage2-20260405` HEAD as of 2026-05-05. Line citations are valid for that commit; binding line numbers in `src/simsoptpp/python.cpp` (57, 58, 60, 91, 106, 136, 137, 138) are stable but worth re-confirming if `python.cpp` is reordered.
- JAX-lane correction as of 2026-05-05: the CPU/C++ VJP rows above do not imply a live `CurveCWSFourierCPP` port blocker. The current JAX path supports CWS forward and VJP natively through the `curve.surf` + `surface_spec()` branch in `_supports_native_curve_geometry` (`src/simsopt/field/biotsavart_jax_backend.py:629`) and `curve_spec_from_curve` (`src/simsopt/jax_core/curve_geometry.py:99`). No `CurveCWSFourierCPP.to_spec()` shim is required for the banana target lane.
- Some `Curve*Distance` and `Surface*Distance` objectives mix C++ candidate culling with host-Python distance accumulation. The same code path runs on both lanes; only the gradient back-propagation differs (CPU lane uses curve VJP methods on the C++ curve class; JAX lane uses autodiff through pure kernels).
- The ALM outer loop in `alm_utils.py` is shared between lanes. It only differs in the inner-step optimizer it invokes — `scipy.optimize.minimize` on CPU vs. JAX-traced inner steps on the JAX lane.
- The single-stage `SurfaceSurfaceDistance(boozer_surface.surface, VV, SS_DIST)` term builds on the same C++ surface kernels as the Stage 2 surface evaluation; the vessel surface `VV` is a `SurfaceRZFourier` instance.
- Stage 2 imports fieldline/Poincaré symbols from `src/simsopt/field/tracing.py` but does not call them in the optimizer path; they are present as future-use scaffolding.
- `BoozerResidualExact` (`single_stage_banana_example.py:4621`) is a separate residual class used by the JAX-Exact lane; on the default single-stage CPU lane the dispatch returns the legacy `BoozerResidual` (`single_stage_banana_example.py:7486`). Its dependencies (`boozer_surface_residual`, `boozer_surface_residual_dB` from `src/simsopt/geo/surfaceobjectives.py`; `forward_backward` from `src/simsopt/objectives/utilities.py`) are imported at lines 120, 121, and 124 respectively and are wired correctly — `BoozerResidualExact` is functional, not stubbed.
