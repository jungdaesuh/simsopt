# JAX Port Status - 2026-05-15

This document summarizes the current JAX port surface, CPU/JAX precision parity
evidence, remaining partial rows, and next validation steps. It is intentionally
scoped: it does not claim full SIMSOPT workflow parity, full CUDA acceptance, or
universal precision for every possible input.

## Executive Snapshot

| Area | Current status | Evidence boundary |
| --- | --- | --- |
| Smooth fixed-state JAX kernels | Broadly ported | Field, geometry, flux, Boozer, PM, wireframe, tracing, and objective kernels have local CPU/JAX coverage. |
| CPU C++/SciPy vs JAX CPU precision parity | Good for covered rows | Latest local matrix has 27 fixtures, 21 pass, 6 partial, 0 fail, and 251 supported comparisons passing. |
| GPU/CUDA parity | Open | Local machine is macOS Apple Silicon; latest matrix records `jax_gpu.status = runtime_required`. |
| Full host workflow parity | Partial by design | Remaining gaps are host orchestration, external solvers, output/reporting, non-unique solver identity, or discrete mutation/event behavior. |
| Toy/tautology risk | Controlled, not absent | Oracle-lint policy rejects JAX-vs-JAX as parity proof; self-consistency tests exist but are not counted as CPU/C++ parity rows. |

Current source HEAD when this file was written:

```text
5c488d809 feat: add JAX QFM penalty current parity
```

The latest full matrix artifact used here records:

```text
artifact: .artifacts/parity/20260514-partial-closeout/all-fixtures.json
artifact git_head: ab2bb12bc98b8c570db9c006f1367012a7707b85
selected lanes: cpu_cpp,jax_cpu
jax_backend: cpu
jax_gpu.status: runtime_required
```

That artifact predates the final committed QFM/current cleanup commit, so it is
strong local evidence but not a fresh clean-HEAD release stamp. The final commit
adds stricter research-fixture QFM/current parity tests on top.

## Ported Surface

### Core Field And Geometry Kernels

| Ported JAX surface | Original/reference surface | Status |
| --- | --- | --- |
| `src/simsopt/jax_core/biotsavart.py` and `field/biotsavart_jax*.py` | `src/simsoptpp/biot_savart_impl.h`, `field/biotsavart.py` | Ported for covered B, dB, A, dA, VJP, and grouped fixed-state lanes. |
| `src/simsopt/jax_core/dipole_field.py` and `field/dipole_field_jax.py` | `src/simsoptpp/dipole_field.*` | Ported for PM field/objective rows. |
| `src/simsopt/jax_core/analytic_fields.py` and analytic wrappers | `src/simsoptpp/dommaschk.*`, `src/simsoptpp/reiman.*`, analytic Python field classes | Ported for covered analytic field wrappers. |
| `src/simsopt/jax_core/curve_geometry.py` | `geo/curve*.py`, `src/simsoptpp/curve*` | Ported for supported curve families and derivative paths used by target lanes. |
| `src/simsopt/jax_core/surface_rzfourier.py`, `surface_fourier.py`, `geo/surface_fourier_jax.py` | `geo/surface*.py`, `src/simsoptpp/surface*` | Ported for covered RZ/XYZ/Tensor surface geometry, derivatives, scalar metrics, and objective consumers. |
| `src/simsopt/jax_core/interpolated_field.py`, `interpolated_boozer_field.py`, `tracing.py` | `field/magneticfieldclasses.py`, `field/tracing.py`, `src/simsoptpp/tracing.*` | Ported for reduced fixed-state tracing/field endpoint evidence; differentiable event optimization is not claimed. |

### Objective And Optimizer-Consumed Kernels

| Ported JAX surface | Original/reference surface | Status |
| --- | --- | --- |
| `objectives/fluxobjective_jax.py` and `objectives/integral_bdotn_jax.py` | `objectives/fluxobjective.py`, `src/simsoptpp/integral_BdotN.*` | Contract-complete for covered fixed-state value/gradient lanes. |
| `objectives/stage2_target_objective_jax.py` | Stage-II objective assembly over flux + curve penalties | Reduced-strict integration path is covered on CPU/JAX; CUDA remains open. |
| `geo/surfaceobjectives_jax.py` | `geo/surfaceobjectives.py`, C++ surface metric methods | Ported for covered `Area`, `Volume`, `ToroidalFlux`, `AspectRatio`, `PrincipalCurvature`, `QfmResidual`, `MajorRadius`, `Iotas`, `NonQuasiSymmetricRatio`, and QFM penalty paths. |
| `geo/curveobjectives_jax.py` | `geo/curveobjectives.py` | Ported for covered length, curve-curve distance, curve-surface distance, curvature penalties, linking number, force, and energy rows. |
| `geo/boozersurface_jax.py`, `geo/boozer_residual_jax.py` | `geo/boozersurface.py`, `src/simsoptpp/boozerresidual_impl.h` | CPU contract is covered for fixed-state values, derivatives, Hessian/adjoint lanes, and integration slices. |
| `jax_core/pm_optimization.py`, `solve/permanent_magnet_optimization_jax.py` | `solve/permanent_magnet_optimization.py`, `src/simsoptpp/permanent_magnet_optimization.*` | Ported for supported fixed-state PM algorithms and reduced example rows. |
| `jax_core/wireframe.py`, `field/wireframefield_jax.py`, `solve/wireframe_optimization_jax.py` | `field/wireframefield.py`, `solve/wireframe_optimization.py`, `src/simsoptpp/wireframe_*` | Ported for supported fixed-state RCLS/GSCO field, matrix, and history rows. |

### Infrastructure

| Surface | Role |
| --- | --- |
| `src/simsopt/jax_core/specs.py` | Immutable specs for curves, surfaces, fields, PM grids, and runtime payloads. |
| `src/simsopt/_core/jax_host_boundary.py` | Host/JAX boundary guard. |
| `src/simsopt/geo/optimizer_jax.py` and `optimizer_jax_private/*` | JAX optimizer adapter, BFGS/LBFGS, line search, and result conversion. |
| `src/simsopt/jax_core/sharding.py`, `reductions.py`, `regular_grid_interp.py` | Device/reduction/interpolation support infrastructure. |

## Precision Parity

Latest local matrix evidence:

| Metric | Value |
| --- | --- |
| Fixtures | 27 |
| Pass | 21 |
| Partial | 6 |
| Fail | 0 |
| Supported CPU/JAX comparisons | 251 |
| Tolerance buckets present | `direct_kernel`, `derivative_heavy`, `ls_wrapper_gradient`, `event_time_tracing` |
| Worst absolute difference | `1.668088953010738e-5` |
| Worst relative difference | `1.9296860502983134e-4` |

Interpretation:

- The worst absolute difference is a tracing particle trajectory endpoint under
  the `event_time_tracing` lane (`rtol=1e-6`, `atol=1e-8`), not a smooth
  objective-gradient lane.
- The worst relative difference occurs on a near-zero direct-kernel quantity;
  the absolute difference is at roundoff scale, so the row passes by absolute
  tolerance.
- The smooth objective and derivative lanes use tighter tolerance buckets:
  `direct_kernel` is typically `rtol=1e-10`, `atol=1e-12`; derivative-heavy
  rows are typically `rtol=1e-8`, `atol=1e-10`.

Precision parity conclusion:

```text
Covered CPU C++/SciPy vs JAX CPU rows look good.
No native-supported matrix comparison is failing.
CUDA/GPU precision parity is not proven by this artifact.
```

## Test Coverage

Current test-file inventory:

| Test inventory | Count |
| --- | ---: |
| Total Python test files under `tests/` | 214 |
| JAX/parity-named test files | 90 |
| Non-integration JAX/parity files | 84 |
| Integration JAX/parity files | 6 |

Integration JAX/parity files:

- `tests/integration/test_jax_native_path.py`
- `tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py`
- `tests/integration/test_single_stage_jax.py`
- `tests/integration/test_single_stage_jax_cpu_reference.py`
- `tests/integration/test_single_stage_physics_parity.py`
- `tests/integration/test_stage2_jax.py`

Recent focused validation after the latest QFM/current commit:

| Validation | Result |
| --- | --- |
| QFM/current focused slice | `7 passed, 3 skipped` |
| Broader QFM/flux related slice | `50 passed, 31 skipped` |
| `compileall` on edited parity tests | passed |
| `ruff check` on edited parity tests | passed |
| `git diff --check` on commit diff | passed |

Older documented broad non-CUDA gates remain useful context:

| Gate | Documented result |
| --- | --- |
| Boozer result-contract non-CUDA bundle | `758 passed, 1 skipped, 65 deselected, 56 subtests passed` |
| Wave 2 JAX port gate | `722 passed, 60 skipped` |
| Stage 2 integration file | `173 passed` |
| Single-stage CPU reference integration file | `173 passed, 5 skipped` |

## Oracle Quality

The parity standard is:

```text
Existing SIMSOPT C++/SciPy behavior -> JAX CPU matches -> JAX GPU matches -> JAX CPU and GPU match.
```

The current local proof covers the first arrow for the supported rows:

```text
Existing SIMSOPT C++/SciPy behavior -> JAX CPU matches
```

Controls against toy or tautological tests:

- `tests/REVIEWER_ORACLE_LINT.md` requires every equality assertion to name an
  independent oracle and rejects JAX-vs-JAX equality as CPU/C++ parity proof.
- `tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py` requires
  `simsoptpp` for the CPU oracle.
- The non-banana parity harness checks that pass/partial rows contain numeric
  `cpu_cpp_value` and `jax_cpu_value`.
- The harness checks CPU and JAX active DOF vectors use matching structural
  names and hashes before comparing gradients.
- Self-consistency checks still exist, for example chunked-vs-dense JAX tests,
  but those are labelled Tier-4/self-consistency and are not counted as
  CPU/C++ parity oracle rows.

This means the current matrix is not a formal proof over all possible inputs,
but it is real acceptance/regression evidence for the covered fixed-state
research fixtures.

## Current Pass Rows

These rows currently pass all native-supported CPU/JAX comparisons:

| Row | Scope |
| --- | --- |
| `minimal_stage2_flux_length_gap` | Stage-II minimal flux plus length penalty. |
| `surface_area_volume_simple` | Surface geometry, area, volume, gradients. |
| `cws_saved_local_flux_nfp2` | Saved CWS local-flux fixture, nfp=2. |
| `cws_saved_local_flux_nfp3` | Saved CWS local-flux fixture, nfp=3. |
| `full_stage2_composite` | Full Stage-II composite with curve penalties. |
| `planar_stage2_composite` | Planar Stage-II composite with linking number. |
| `position_orientation_flux_support_gate` | Position/orientation fixed-state flux. |
| `boozer_surface_basic` | NCSX Boozer residual + labels. |
| `boozer_qa_wrappers` | Boozer QA solved-state scalar wrappers. |
| `finite_beta_target_flux` | Finite-beta target flux with cached W7-X target. |
| `finitebuild_multifilament_support_gate` | Finite-build multifilament flux/penalties. |
| `pm_simple_fixed_state_gpmo_baseline` | Reduced PM simple GPMO baseline. |
| `tracing_fieldlines_qa_reduced_endpoint` | QA fieldline endpoint/hit-count parity. |
| `pm_muse_famus` | Reduced MUSE PM FAMUS fixed-state path. |
| `pm_pm4stell_backtracking` | Reduced PM4Stell backtracking path. |
| `wireframe_gsco_modular_fixed_state` | Reduced modular GSCO fixed-state path. |
| `wireframe_gsco_sector_saddle_fixed_state` | Reduced sector/saddle GSCO path. |
| `tracing_fieldlines_ncsx_reduced_endpoint` | NCSX fieldline endpoint/hit-count parity. |
| `tracing_particle_gc_vac_reduced_endpoint` | NCSX particle guiding-center endpoint parity. |
| `strain_optimization_support_gate` | HSX framed-curve strain fixed-state path. |
| `coil_forces_support_gate` | Coil force/energy fixed-state subproblem. |

## Remaining Partial Rows

The 6 partial rows are not numerical parity failures. They have passing
native-supported comparisons plus explicit unsupported components.

| Row | Unsupported component | Why it remains partial |
| --- | --- | --- |
| `qfm_surface` | `QfmSurface_host_solver` | Fixed-state QFM residual/label/penalty pieces are covered. Full host `QfmSurface` LBFGS/SLSQP solve orchestration is not JAX parity. |
| `wireframe_rcls_basic_fixed_state` | `RCLS_current_vector_nonunique_nullspace` | Fields, matrices, objective components, constraints, and Bnormal agree. Raw current vector identity is not unique because of nullspace. |
| `pm_qa_fixed_state_gpmo_arbvec_or_multi` | `qa_coil_current_optimization`, `qa_plot_and_famus_outputs` | Reduced relax-and-split fixed-state pieces compare. Coil-current optimization/output writing remain host orchestration. |
| `wireframe_rcls_ports_constraint_gate` | `RCLS_current_vector_nonunique_nullspace` | Same nullspace issue as basic RCLS, with port constraints preserved. |
| `wireframe_gsco_multistep_reduced_diagnostic` | mutation/pruning/final-adjust/output loop | First-step GSCO diagnostic is covered. Full mutating multistep workflow is discrete host orchestration. |
| `tracing_boozer_gc_reduced_endpoint` | `VMEC_input_external_solver` | Cached VMEC/BOOZXFORM-state endpoint path compares. Running VMEC/BOOZXFORM from inputs is an external-solver boundary. |

## What Is Next

### Release Acceptance

1. Regenerate the full parity matrix from the current clean HEAD.
2. Run the same fixed-state matrix on real CUDA hardware.
3. Produce CPU C++/SciPy vs JAX GPU and JAX CPU vs JAX GPU artifacts with:
   - git SHA,
   - clean/dirty tree state,
   - JAX and jaxlib versions,
   - `JAX_ENABLE_X64=1`,
   - device model,
   - CUDA/runtime metadata,
   - exact commands.
4. Only then close GPU/CUDA checkboxes in the manifest or plan docs.

### Engineering Scope Decisions

These should not be blindly ported unless they are real downstream product
paths:

| Candidate | Recommendation |
| --- | --- |
| `QfmSurface` full solver orchestration | Port only if solved QFM surfaces must be differentiable product outputs. Otherwise fixed-state QFM objective/penalty coverage is enough. |
| PM QA coil-current optimization | Port only if coil-current optimization is a required downstream workflow. Plot/FAMUS/output paths should stay host-side. |
| Differentiable event tracing | Treat as a research project, not a simple port. Smooth fixed-time tracing is different from differentiable event/status behavior. |
| VMEC/BOOZXFORM/SPEC execution | Do not treat as a JAX-port task. Use cached states, surrogate/adjoint work, or external-solver boundaries. |
| Generic `Optimizable`/SciPy/MPI orchestration | Keep host-side unless a concrete target-lane workflow needs a JAX-native replacement. |

### Documentation Hygiene

- Keep `docs/jax_parity_manifest.md` as the source of truth for parity rows.
- Do not upgrade a row from partial to pass based only on JAX-vs-JAX agreement.
- Keep reduced-real fixtures labelled as reduced-real, not full example runtime.
- Keep GPU status separate from CPU/JAX status.
- Keep self-consistency tests labelled as self-consistency, not CPU/C++ oracle
  evidence.

## Bottom Line

```text
CPU/JAX precision parity for the covered fixed-state research fixtures is in good shape.
The remaining partial rows are mostly explicit host/external/discrete boundaries.
The main blocker for release-grade JAX port acceptance is real CUDA evidence,
not another round of toy or JAX-vs-JAX tests.
```
