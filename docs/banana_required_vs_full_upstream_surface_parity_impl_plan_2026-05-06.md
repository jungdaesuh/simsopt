# Banana-Required vs Full-Upstream Surface Parity Implementation Plan

Status: implementation plan only.
Date: 2026-05-06.
Base commit: `0bb26bb0a`.
Validation basis: current working tree on 2026-05-06. This plan file is untracked, and unrelated dirty files were left unchanged.

## Verdict

For strict banana-required CPU/C++ precision and function parity, do not require every partial surface-family file to become a full legacy mirror. The banana contract is narrower than full upstream SIMSOPT parity:

- Field and flux kernels used by Stage 2.
- Stage 2 target bundle and reporting path.
- Boozer/single-stage objective path.
- Surface geometry/spec paths actually consumed by Stage 2 and single-stage workflows.

The larger "partial" labels mostly come from upstream surface API breadth: second-order surface tangents, forms/curvatures, scalar metric Hessians, host object utilities, broad I/O/copy behavior, and missing upstream surface objective wrappers. Those are valid full-upstream parity requirements, but they are not all banana ship blockers.

## Source Of Truth Split

- `docs/jax_parity_manifest.md` remains the status SSOT for parity rows.
- This file is the implementation plan for closing the quoted partial-file requirements.
- Banana readiness is judged by Requirement Set A plus current-sha CUDA artifacts where required by P5.
- Full legacy/upstream parity is judged by Requirement Set B.
- Do not mark manifest rows complete until the named tests and artifact evidence exist.

## Requirement Set A: Banana-Required Closure

### A0 Scope Lock

- [x] Treat existing SIMSOPT C++/SciPy behavior as the oracle.
- [x] Require same-state C++/SciPy -> JAX CPU parity before interpreting optimizer behavior.
- [x] Require JAX CPU -> JAX CUDA and CPU/GPU agreement before marking CUDA rows complete.
- [x] Keep JAX-vs-JAX agreement insufficient by itself.
- [x] Keep `BiotSavartJAX`, fixed-surface flux, Stage 2 target bundle, Boozer/single-stage objectives, and consumed surface specs as the banana product surface.
- [x] Link this Set A/Set B scope split from the `docs/jax_parity_manifest.md` preamble.
- [ ] Keep this scope decision reflected in `docs/jax_parity_manifest.md` after every closure PR.
- [ ] Leave full-upstream surface/API parity as backlog unless the banana product path starts loading those APIs directly.

### A1 Field And Flux Lanes

Files:

- `src/simsopt/jax_core/biotsavart.py`
- `src/simsopt/field/biotsavart_jax_backend.py`
- `src/simsopt/objectives/fluxobjective_jax.py`
- `src/simsopt/jax_core/objectives_flux.py`

Required state:

- [x] Keep Biot-Savart value and derivative parity exact for banana-required kernels.
- [x] Keep `BiotSavartJAX`, `SpecBackedBiotSavartJAX`, and `SingleStageRuntimeSpecBiotSavartJAX` contract-complete for required field paths.
- [x] Keep `SquaredFluxJAX` and fixed-surface flux kernels contract-complete for Stage 2.
- [x] Keep CPU/C++ precision checks separate from optimizer trace diagnostics.
- [ ] Preserve existing tolerances from the validation ladder; do not loosen tolerances to hide drift.
- [ ] Treat JSON/getter breadth and raw-kernel Taylor polish as full-repo parity polish unless the banana manifest promotes them.

Acceptance:

- [ ] `tests/objectives/test_fluxobjective_jax_parity.py`
- [ ] `tests/objectives/test_integral_bdotn_jax.py`
- [ ] Stage 2 fixed-state value and gradient artifacts when claiming P5 CUDA closure.

### A2 SurfaceRZFourier Banana Geometry/Spec Path

Files:

- `src/simsopt/jax_core/surface_rzfourier.py`
- `src/simsopt/geo/surfacerzfourier.py`
- `src/simsopt/geo/surface.py`
- `src/simsopt/jax_core/specs.py`

Banana-required functions:

- [x] `gamma`
- [x] `gammadash1`
- [x] `gammadash2`
- [x] `normal`
- [x] `unitnormal`
- [x] `area`
- [x] `volume`
- [x] `darea_by_dcoeff`
- [x] `dvolume_by_dcoeff`
- [x] DOF roundtrip from mutable surface object to immutable spec.
- [x] Spec roundtrip from immutable spec to JAX kernels.
- [x] Loaders used by Stage 2 and single-stage workflows.

Required maintenance tasks:

- [ ] Preserve CPU/C++ DOF order exactly.
- [ ] Keep RZ mutable-wrapper methods thin: snapshot state into a spec and call kernel functions.
- [ ] Keep new math in `src/simsopt/jax_core/surface_rzfourier.py`, not in objective wrappers.
- [ ] Keep `SurfaceRZFourierSpec` immutable and pytree-compatible in `src/simsopt/jax_core/specs.py`.
- [ ] Add a banana-focused regression test if a Stage 2 or single-stage artifact loader starts consuming any new RZ host utility.

Acceptance:

- [ ] `tests/geo/test_surface_rzfourier_jax.py`
- [ ] Any banana Stage 2/single-stage artifact loader tests that consume RZ specs.

### A3 SurfaceXYZTensorFourier Support Consumed By Single-Stage

Files:

- `src/simsopt/geo/surface_fourier_jax.py`
- `src/simsopt/jax_core/surface_fourier.py`
- `src/simsopt/geo/surfacexyztensorfourier.py`
- `src/simsopt/jax_core/specs.py`

Banana-required state:

- [x] `SurfaceXYZTensorFourier.surface_spec()` / `to_spec()` builds `SurfaceXYZTensorFourierSpec` for unclamped tensor surfaces.
- [x] `SurfaceXYZTensorFourier.surface_spec()` rejects `clamped_dims`.
- [x] Single-stage JAX runtime seed payloads use the canonical `"SurfaceXYZTensorFourier"` surface class.
- [x] Single-stage JAX runtime seed loading rejects non-`SurfaceXYZTensorFourier` surface classes.
- [x] JAX spec wrappers cover `gamma`, `gammadash1`, `gammadash2`, and `normal` for `SurfaceXYZTensorFourierSpec`.
- [ ] Add explicit acceptance coverage that tensor `clamped_dims=True` remains rejected for JAX specs.
- [ ] Strengthen banana-required spec parity for `SurfaceXYZTensorFourierSpec` across:
  - [ ] `gamma`
  - [ ] `gammadash1`
  - [ ] `gammadash2`
  - [ ] `normal`
  - [ ] stellsym true/false where applicable
  - [ ] `nfp > 1`
  - [ ] nontrivial `mpol` and `ntor`
  - [ ] nondefault quadrature point ranges
- [ ] Add `SurfaceXYZTensorFourierSpec` area/volume parity against CPU for banana diagnostics/output.
- [ ] Add artifact/load-spec acceptance: a legacy or JAX-emitted single-stage tensor surface artifact loads into immutable specs and evaluates `gamma`/`normal` without compiled target code calling host `surface.gamma()`.
- [ ] Add single-stage surface-distance/self-intersection acceptance that the JAX tensor point cloud equals CPU geometry on the exact banana workflow grid.
- [ ] Keep `SurfaceXYZFourierSpec` in generic dispatch/full-upstream backlog unless a real banana artifact starts using it.

Acceptance:

- [ ] `tests/geo/test_surface_fourier_jax.py`
- [ ] Single-stage artifact/spec loader tests.
- [ ] Single-stage surface-distance/self-intersection parity tests if those checks are claimed in the product path.

### A4 Single-Stage Objective Wrappers Used By Banana

Files:

- `src/simsopt/geo/surfaceobjectives_jax.py`
- `src/simsopt/geo/boozer_residual_jax.py`
- `src/simsopt/geo/boozersurface_jax.py`

Banana-required wrappers:

- [x] `BoozerResidualJAX`
- [x] `IotasJAX`
- [x] `NonQuasiSymmetricRatioJAX`
- [x] Traceable runtime objective bundle.
- [x] `ToroidalFlux` where used.

Required maintenance tasks:

- [ ] Route objective geometry through the existing `surface_kind` dispatch.
- [ ] Preserve exact Boozer solve support scope; do not expand `SurfaceXYZTensorFourier` exact-solve support as a side effect of surface parity work.
- [ ] Keep host materialization only in named host wrappers.
- [ ] Do not add `jax.pure_callback` bridges to compiled objective paths.
- [ ] Keep Stage 2 target-lane reporting pure through `Stage2TargetObjectiveBundle.reporting_summary`.

Not banana blockers today:

- [ ] `MajorRadiusJAX`
- [ ] `PrincipalCurvatureJAX`
- [ ] `QfmResidualJAX`
- [ ] `AspectRatioJAX`

Acceptance:

- [ ] `tests/geo/test_surface_objectives_jax.py`
- [ ] `tests/geo/test_boozersurface_jax.py`
- [ ] `tests/geo/test_boozer_derivatives_jax.py`
- [ ] `tests/integration/test_single_stage_jax_cpu_reference.py`

### A5 Current-SHA CUDA Artifact Gate

This is required for banana P5 CUDA closure, but it is not the same thing as full upstream surface parity.

- [ ] Do not mark CUDA rows complete without real CUDA artifacts from the current pushed SHA.
- [ ] Record git SHA and dirty-tree status.
- [ ] Record command line and environment.
- [ ] Record Python, JAX, CUDA, driver, device, x64, and XLA metadata.
- [ ] Record host RSS and GPU memory telemetry where available.
- [ ] Preserve pass/fail reason and artifact path in the manifest.
- [ ] Keep CPU-only local proof labeled as CPU evidence only.

Required CUDA rows if still open in the manifest:

- [ ] Stage 2 fixed-state value.
- [ ] Stage 2 fixed-state gradient.
- [ ] Stage 2 reduced end-to-end strict run.
- [ ] Single-stage initialization.
- [ ] Boozer well-conditioned adjoint.
- [ ] CPU/GPU reduction stress.

## Requirement Set B: Full Legacy/Upstream Coverage

Set B is the real requirement list if the goal is to remove the "partial" label from the listed surface-family files completely. This is a broader surface/objective port, not a narrow banana validation task.

### B0 Cross-Cutting Architecture Review Checklist

These items are PR review checks unless a later implementation adds an explicit lint, grep, or CI owner for them.

- [ ] Keep all pure surface math in `src/simsopt/jax_core/*` or the existing JAX surface modules.
- [ ] Keep mutable Python object wrappers thin.
- [ ] Preserve CPU/reference behavior unchanged.
- [ ] Preserve immutable spec constructors as the SSOT for JAX surface state.
- [ ] Preserve DOF ordering, stellsym skipped modes, and coefficient layout exactly.
- [ ] Do not introduce dynamic imports.
- [ ] Do not introduce `Any` casts or `typing.cast`.
- [ ] Do not add defensive try/except fallbacks.
- [ ] Do not auto-convert host inputs inside JIT/runtime boundaries.
- [ ] Do not introduce callback bridges in compiled paths.
- [ ] Avoid naive production Hessians that allocate avoidable `O(ndofs^2)` intermediates outside tests or explicit Hessian APIs.
- [ ] Keep singular scalar-metric tests away from zero-area/zero-volume cases unless the CPU oracle explicitly defines those limits.

### B1 RZ Second-Order Geometry Core

Files:

- `src/simsopt/jax_core/surface_rzfourier.py`
- `src/simsopt/geo/surfacerzfourier.py`
- `tests/geo/test_surface_rzfourier_jax.py`
- `tests/geo/test_surface_taylor.py`

Implementation:

- [ ] Add direct JAX spec/from-dofs function for `gammadash1dash1`.
- [ ] Add direct JAX spec/from-dofs function for `gammadash1dash2`.
- [ ] Add direct JAX spec/from-dofs function for `gammadash2dash2`.
- [ ] Include Cartesian frame derivative terms, not only Fourier `R/Z` mode derivatives.
- [ ] Add optional fused second-geometry output only if it removes real duplicated work.
- [ ] Add coefficient Jacobians/VJPs for `gammadash1dash1`.
- [ ] Add coefficient Jacobians/VJPs for `gammadash1dash2`.
- [ ] Add coefficient Jacobians/VJPs for `gammadash2dash2`.
- [ ] Add thin public `_jax` wrappers in `src/simsopt/geo/surfacerzfourier.py` only after kernel tests pass.
- [ ] Export new functions from `src/simsopt/jax_core/__init__.py` if existing export conventions require it.

Tests:

- [ ] Value parity against legacy `SurfaceRZFourier.gammadash1dash1()`.
- [ ] Value parity against legacy `SurfaceRZFourier.gammadash1dash2()`.
- [ ] Value parity against legacy `SurfaceRZFourier.gammadash2dash2()`.
- [ ] Coefficient-derivative parity against legacy `dgammadash*_by_dcoeff()` methods.
- [ ] Taylor tests mirroring `tests/geo/test_surface_taylor.py`.
- [ ] stellsym true coverage.
- [ ] stellsym false coverage.
- [ ] `nfp > 1` coverage.
- [ ] Nondefault quadrature point coverage.
- [ ] HLO/transfer-guard smoke if a fused kernel is added.

### B2 RZ Forms And Curvatures

Files:

- `src/simsopt/jax_core/surface_rzfourier.py`
- `src/simsopt/geo/surfacerzfourier.py`
- `src/simsoptpp/surface.h`
- `src/simsoptpp/surface.cpp`
- `src/simsoptpp/python_surfaces.cpp`
- `tests/geo/test_surface_rzfourier_jax.py`
- `tests/geo/test_surface_taylor.py`
- `tests/geo/test_surface.py`

Implementation:

- [ ] Add `first_fund_form` with legacy ordering `[E, F, G]`.
- [ ] Add `second_fund_form` with legacy ordering `[L, M, N]`.
- [ ] Add `surface_curvatures` with legacy ordering `[H, K, kappa1, kappa2]`.
- [ ] Add `dfirst_fund_form_by_dcoeff` if parity claim includes form derivatives.
- [ ] Add `dsecond_fund_form_by_dcoeff` if parity claim includes form derivatives.
- [ ] Add `dsurface_curvatures_by_dcoeff`.
- [ ] Preserve normal orientation from `gammadash1 x gammadash2`.
- [ ] Document the sign convention in comments only where the formula is otherwise ambiguous.

Tests:

- [ ] Value parity for first fundamental form.
- [ ] Value parity for second fundamental form.
- [ ] Value parity for surface curvatures.
- [ ] Derivative parity for form derivatives if implemented.
- [ ] Derivative parity for `dsurface_curvatures_by_dcoeff`.
- [ ] Taylor finite-difference checks.
- [ ] Gauss-Bonnet style coverage against upstream `tests/geo/test_surface.py`.
- [ ] Curvature sign regression on at least one nontrivial non-stellsym surface.

### B3 RZ Scalar Metric Hessians

Files:

- `src/simsopt/jax_core/surface_rzfourier.py`
- `src/simsopt/geo/surfacerzfourier.py`
- `src/simsoptpp/surface.h`
- `src/simsoptpp/surface.cpp`
- `src/simsoptpp/python_surfaces.cpp`
- `tests/geo/test_surface_rzfourier_jax.py`
- `tests/geo/test_surface_taylor.py`
- `tests/geo/test_surface_rzfourier.py`

Implementation:

- [ ] Add `mean_cross_sectional_area` JAX scalar helper.
- [ ] Add `minor_radius` JAX scalar helper.
- [ ] Add `major_radius` JAX scalar helper.
- [ ] Add `aspect_ratio` JAX scalar helper.
- [ ] Add JAX `d2area_by_dcoeffdcoeff` mirroring the C++/pybind CPU oracle.
- [ ] Add JAX `d2volume_by_dcoeffdcoeff` mirroring the C++/pybind CPU oracle.
- [ ] Add `d2minor_radius_by_dcoeff_dcoeff`.
- [ ] Add `d2major_radius_by_dcoeff_dcoeff`.
- [ ] Add `d2aspect_ratio_by_dcoeff_dcoeff`.
- [ ] Keep Hessian APIs explicit so production paths do not allocate Hessians accidentally.

Tests:

- [ ] CPU/JAX value parity for `mean_cross_sectional_area`.
- [ ] CPU/JAX value parity for `minor_radius`.
- [ ] CPU/JAX value parity for `major_radius`.
- [ ] CPU/JAX value parity for `aspect_ratio`.
- [ ] Gradient parity for each scalar metric.
- [ ] Hessian parity for each scalar metric with upstream tolerances.
- [ ] Second-order Taylor tests.
- [ ] Area Hessian parity against the C++/pybind CPU oracle.
- [ ] Volume Hessian parity against the C++/pybind CPU oracle.
- [ ] Tests avoid near-zero singular cases unless explicitly testing CPU-defined behavior.

### B4 Broader SurfaceRZFourier Host API Behavior

Files:

- `src/simsopt/geo/surfacerzfourier.py`
- `src/simsopt/jax_core/surface_rzfourier.py`
- `tests/geo/test_surface_rzfourier_jax.py`
- `tests/geo/test_surface_rzfourier.py`

Implementation and coverage:

- [ ] Verify `from_focus` output can produce a JAX spec and match CPU geometry.
- [ ] Verify `from_pyQSC` output can produce a JAX spec and match CPU geometry.
- [ ] Verify `make_rotating_ellipse` output can produce a JAX spec and match CPU geometry.
- [ ] Verify `change_resolution` preserves JAX spec roundtrip.
- [ ] Verify `condense_spectrum` preserves JAX spec roundtrip.
- [ ] Verify `extend_via_normal` preserves JAX spec roundtrip.
- [ ] Verify `copy` and object-independence semantics for JAX spec snapshots.
- [ ] Add serialization/GSON roundtrip tests if I/O parity is claimed.
- [ ] Add `to_vtk` smoke/file-exists coverage only if I/O parity is claimed.
- [ ] Treat optional dependency tests as skipped when the upstream CPU tests skip for missing optional dependencies.

Acceptance:

- [ ] Existing CPU host tests still pass unchanged.
- [ ] JAX spec tests prove the resulting surfaces evaluate the same `gamma`, tangents, normals, area, and volume as CPU.

### B5 Non-RZ Geometry And Derivative Parity

Files:

- `src/simsopt/geo/surface_fourier_jax.py`
- `src/simsopt/jax_core/surface_fourier.py`
- `src/simsopt/geo/surfacexyzfourier.py`
- `src/simsopt/geo/surfacexyztensorfourier.py`
- `tests/geo/test_surface_fourier_jax.py`
- `tests/geo/test_surface_taylor.py`

Implementation:

- [ ] Add `SurfaceXYZFourier` coefficient derivative parity for `dgamma_by_dcoeff`.
- [ ] Add `SurfaceXYZFourier` coefficient derivative parity for `dgammadash1_by_dcoeff`.
- [ ] Add `SurfaceXYZFourier` coefficient derivative parity for `dgammadash2_by_dcoeff`.
- [ ] Add `gammadash1dash1` for `SurfaceXYZFourierSpec`.
- [ ] Add `gammadash1dash2` for `SurfaceXYZFourierSpec`.
- [ ] Add `gammadash2dash2` for `SurfaceXYZFourierSpec`.
- [ ] Add `gammadash1dash1` for `SurfaceXYZTensorFourierSpec`.
- [ ] Add `gammadash1dash2` for `SurfaceXYZTensorFourierSpec`.
- [ ] Add `gammadash2dash2` for `SurfaceXYZTensorFourierSpec`.
- [ ] Add coefficient derivatives for second coordinate derivatives where upstream exposes them.
- [ ] Add CPU parity tests for existing `gamma_lin` / `surface_gamma_lin_from_dofs` if paired-point APIs are in full-upstream scope.
- [ ] Add `gammadash1_lin`.
- [ ] Add `gammadash2_lin`.
- [ ] Add higher `*_lin` paired-point APIs only if full legacy parity explicitly includes them.
- [ ] Add `unitnormal`.
- [ ] Add `dnormal_by_dcoeff`.
- [ ] Add `d2normal_by_dcoeffdcoeff` only as an explicit heavy API.
- [ ] Add `dunitnormal_by_dcoeff`.
- [ ] Add full-upstream non-RZ area/volume value APIs for surface families not owned by A3 tensor diagnostics.
- [ ] Add `darea`, `d2area`, `dvolume`, and `d2volume` parity for full-upstream non-RZ scope.

Tests:

- [ ] CPU/JAX parity for all new non-RZ coordinate derivatives.
- [ ] CPU/JAX parity for coefficient derivatives.
- [ ] First- and second-order Taylor tests.
- [ ] stellsym true/false coverage where supported.
- [ ] `nfp > 1` coverage.
- [ ] nondefault quadrature coverage.
- [ ] Tensor unclamped coverage.
- [ ] Explicit rejection coverage for tensor `clamped_dims` unless full upstream scope decides to support it.

### B6 Non-RZ Object API Breadth

Files:

- `src/simsopt/geo/surface_fourier_jax.py`
- `src/simsopt/jax_core/surface_fourier.py`
- `src/simsopt/geo/surfacexyzfourier.py`
- `src/simsopt/geo/surfacexyztensorfourier.py`
- `tests/geo/test_surface_fourier_jax.py`
- `tests/geo/test_surface_xyzfourier.py`

Implementation and coverage:

- [ ] Add copy/deepcopy/object-independence tests for `SurfaceXYZFourier`.
- [ ] Add copy/deepcopy/object-independence tests for `SurfaceXYZTensorFourier`.
- [ ] Add direct JSON/GSON roundtrip tests for `SurfaceXYZFourier`.
- [ ] Add direct JSON/GSON roundtrip tests for `SurfaceXYZTensorFourier`.
- [ ] Add VTK smoke/file-exists coverage for tensor surfaces if I/O parity is claimed.
- [ ] Add object API tests for `to_RZFourier`.
- [ ] Add object API tests for `cross_section`.
- [ ] Add object API tests for `least_squares_fit`.
- [ ] Add object API tests for `fit_to_curve`.
- [ ] Add object API tests for `scale`.
- [ ] Add object API tests for `extend_via_normal`.
- [ ] Add object API tests for `extend_via_projected_normal`.
- [ ] Fix or add the intended `test_surface_conversion` coverage if the current test body is exercising the wrong helper.

Acceptance:

- [ ] `tests/geo/test_surface_fourier_jax.py`
- [ ] `tests/geo/test_surface_xyzfourier.py`
- [ ] `tests/geo/test_surface.py`
- [ ] `tests/geo/test_surface_taylor.py`

### B7 Missing Surface Objective Wrappers

File:

- `src/simsopt/geo/surfaceobjectives_jax.py`

Related dependencies:

- `src/simsopt/jax_core/surface_rzfourier.py`
- `src/simsopt/geo/surface_fourier_jax.py`
- `src/simsopt/jax_core/surface_fourier.py`
- `src/simsopt/geo/boozersurface_jax.py`
- `tests/geo/test_surface_objectives_jax.py`
- `tests/geo/test_surface_objectives.py`

Implementation order:

- [ ] Add shared pure-JAX scalar helpers first:
  - [ ] `mean_cross_sectional_area`
  - [ ] `minor_radius`
  - [ ] `major_radius`
  - [ ] `aspect_ratio`
  - [ ] gradients for all four
  - [ ] Hessians where upstream exposes them
- [ ] Implement `AspectRatioJAX`.
- [ ] Implement `MajorRadiusJAX`.
- [ ] Implement `QfmResidualJAX`.
- [ ] Implement `PrincipalCurvatureJAX` last, after curvature kernels exist.
- [ ] Export new wrappers through `surfaceobjectives_jax.__all__`.
- [ ] Add import/lazy-access smoke coverage through `simsopt.geo` if existing conventions require it.
- [ ] Add `aspect_ratio` label support to `BoozerSurfaceJAX` only if full label-test parity is in scope.

`AspectRatioJAX` requirements:

- [ ] Mirror upstream value behavior.
- [ ] Mirror `dJ`.
- [ ] Mirror `dJ_by_dsurfacecoefficients`.
- [ ] Mirror `d2J_by_dsurfacecoefficientsdsurfacecoefficients`.
- [ ] Test CPU/JAX value parity.
- [ ] Test surface-gradient parity.
- [ ] Test Hessian parity.
- [ ] Test first-order Taylor.
- [ ] Test second-order Taylor.
- [ ] Cover `SurfaceRZFourier`.
- [ ] Cover `SurfaceXYZFourier`.
- [ ] Cover `SurfaceXYZTensorFourier`.
- [ ] Cover stellsym true/false where supported.

`MajorRadiusJAX` requirements:

- [ ] Reuse the existing Boozer objective base if it fits the derivative contract.
- [ ] Value is solved-surface major radius.
- [ ] Direct coil gradient is zero.
- [ ] Adjoint RHS is the major-radius surface gradient padded into `[surface_dofs, iota, G]`.
- [ ] Test value parity vs CPU `MajorRadius`.
- [ ] Test public `Derivative` projection parity.
- [ ] Test re-solve directional Taylor/finite difference with respect to coil DOFs for LS where supported.
- [ ] Test exact solve only where `BoozerSurfaceJAX` supports the surface family.

`QfmResidualJAX` requirements:

- [ ] Implement pure scalar `qfm_residual` from surface DOFs, coil-set spec, and surface metadata.
- [ ] Use `jax.grad` with respect to surface DOFs for the surface derivative.
- [ ] Use existing `BiotSavartJAX` field and pullback APIs.
- [ ] Test value parity vs CPU `QfmResidual`.
- [ ] Test surface-gradient parity.
- [ ] Test first-order Taylor with respect to surface DOFs.
- [ ] Test cache/update behavior when surface DOFs change.

`PrincipalCurvatureJAX` requirements:

- [ ] Depend on `surface_curvatures`.
- [ ] Depend on `dsurface_curvatures_by_dcoeff`.
- [ ] Test value parity vs CPU `PrincipalCurvature`.
- [ ] Test surface-gradient parity.
- [ ] Test first-order Taylor.
- [ ] Do not add Hessian tests unless upstream exposes a Hessian contract.

Banana classification:

- [ ] Keep `MajorRadiusJAX` classified as banana-adjacent but not a current banana blocker.
- [ ] Keep `AspectRatioJAX` classified as upstream parity backlog unless it becomes a JAX Boozer label or QFM constraint in the product path.
- [ ] Keep `QfmResidualJAX` classified as upstream/QFM workflow backlog unless product scope changes.
- [ ] Keep `PrincipalCurvatureJAX` classified as upstream parity backlog; banana currently uses curve curvature, not surface principal curvature.

## File Ownership Map

Production files:

- [ ] `src/simsopt/jax_core/surface_rzfourier.py`: RZ pure JAX kernels, derivative kernels, forms, curvatures, scalar metrics, explicit Hessians.
- [ ] `src/simsopt/geo/surfacerzfourier.py`: thin RZ object wrappers and spec snapshot access only.
- [ ] `src/simsopt/geo/surface.py`: CPU/reference base API remains unchanged unless a pure wrapper needs a documented parity hook.
- [ ] `src/simsoptpp/surface.h`, `src/simsoptpp/surface.cpp`, `src/simsoptpp/python_surfaces.cpp`: C++/pybind CPU oracle for forms, curvatures, area/volume Hessians, and derivative-heavy surface parity.
- [ ] `src/simsopt/geo/surface_fourier_jax.py`: non-RZ pure JAX geometry and derivative primitives.
- [ ] `src/simsopt/jax_core/surface_fourier.py`: immutable non-RZ spec wrappers.
- [ ] `src/simsopt/jax_core/specs.py`: immutable specs only when new state is actually required.
- [ ] `src/simsopt/jax_core/__init__.py`: exports only after kernel APIs are stable.
- [ ] `src/simsopt/geo/surfaceobjectives_jax.py`: missing objective wrappers and objective-specific plumbing only.
- [ ] `src/simsopt/geo/boozersurface_jax.py`: label support only if full label parity is in scope.

Test files:

- [ ] `tests/geo/test_surface_rzfourier_jax.py`: RZ JAX parity, transfer guards, spec tests.
- [ ] `tests/geo/test_surface_fourier_jax.py`: non-RZ JAX parity, spec tests.
- [ ] `tests/geo/test_surface_objectives_jax.py`: JAX objective wrappers.
- [ ] `tests/geo/test_surface_taylor.py`: CPU oracle/Taylor reference stays authoritative.
- [ ] `tests/geo/test_surface_rzfourier.py`: CPU host/API oracle stays authoritative.
- [ ] `tests/geo/test_surface.py`: base surface oracle and Gauss-Bonnet/form coverage.
- [ ] `tests/geo/test_surface_objectives.py`: CPU objective oracle stays authoritative.
- [ ] `tests/docs/test_banana_parity_coverage_manifest.py`: manifest status validation after evidence exists.

## Milestone Order

The sequence below is safe for one engineer. For parallel implementation, M4 and M5 can start after M0/M1, M3 starts after M2, and M6 can split by dependency: `QfmResidualJAX` after existing field/pullback contracts, `AspectRatioJAX` / `MajorRadiusJAX` after scalar metric helpers, and `PrincipalCurvatureJAX` after curvature kernels.

### M0 Scope And Baseline

- [ ] Freeze the Set A vs Set B scope split in this file.
- [ ] Confirm `docs/jax_parity_manifest.md` still reflects banana rows accurately.
- [ ] Confirm local interpreter and x64 settings before running parity tests.
- [ ] Confirm no source edits are needed for banana Set A unless current tests fail.

### M1 Banana Non-CUDA Acceptance Tightening

- [ ] Add tensor `clamped_dims` rejection test.
- [ ] Strengthen `SurfaceXYZTensorFourierSpec` parity across banana-relevant grids.
- [ ] Add tensor area/volume spec-level parity if banana diagnostics depend on it.
- [ ] Add artifact/load-spec acceptance for single-stage tensor surfaces.
- [ ] Add self-intersection/surface-distance point-cloud equality if claimed in the product path.
- [ ] Run banana CPU/JAX gates.
- [ ] Update manifest only for evidence-backed rows.

### M2 RZ Full Legacy Geometry

- [ ] Implement RZ second-order coordinate derivatives.
- [ ] Implement RZ coefficient derivatives/VJPs for second-order geometry.
- [ ] Add RZ Taylor and CPU parity tests.
- [ ] Run RZ JAX and CPU oracle tests.

### M3 RZ Forms, Curvatures, Metrics

- [ ] Implement fundamental forms.
- [ ] Implement surface curvatures.
- [ ] Implement curvature derivatives.
- [ ] Implement scalar metric helpers.
- [ ] Implement explicit scalar metric Hessians.
- [ ] Add parity, derivative, Taylor, and Gauss-Bonnet tests.

### M4 Non-RZ Full Geometry

- [ ] Implement non-RZ second coordinate derivatives.
- [ ] Implement non-RZ coefficient derivatives.
- [ ] Implement non-RZ normal/unitnormal derivative APIs.
- [ ] Implement non-RZ area/volume derivative APIs.
- [ ] Add parity and Taylor tests.

### M5 Host/Object API Breadth

- [ ] Add RZ host utility/spec roundtrip coverage.
- [ ] Add non-RZ copy/I/O/object API breadth coverage.
- [ ] Keep optional dependency skips aligned with CPU tests.
- [ ] Keep CPU behavior unchanged.

### M6 Missing Objective Wrappers

- [ ] Implement shared scalar helpers.
- [ ] Implement `AspectRatioJAX`.
- [ ] Implement `MajorRadiusJAX`.
- [ ] Implement `QfmResidualJAX`.
- [ ] Implement `PrincipalCurvatureJAX`.
- [ ] Add CPU/JAX value, derivative, and Taylor tests.
- [ ] Update exports/import smoke tests.

### M7 CUDA And Documentation Evidence

- [ ] Run current-sha CUDA artifacts for banana P5 rows if banana CUDA closure is the goal.
- [ ] Attach artifact metadata to the manifest or linked proof doc.
- [ ] Update `docs/jax_parity_manifest.md`.
- [ ] Update the existing banana coverage plan if rows have moved.
- [ ] Keep Set B backlog rows separate from banana blockers.

## Validation Commands

Use the repo-local interpreter when available:

```bash
cd /Users/suhjungdae/code/columbia/simsopt-jax
export PY=/Users/suhjungdae/code/columbia/simsopt-jax/.conda/jax-0.9.2/bin/python
export PYTHONPATH=/Users/suhjungdae/code/columbia/simsopt-jax/src
export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cpu
```

Banana manifest and harness gate:

```bash
$PY -m pytest -q \
  tests/docs/test_banana_parity_coverage_manifest.py \
  tests/test_hf_production_gpu_proof.py \
  tests/test_benchmark_helpers.py::test_single_stage_init_fixture_files_are_vendored \
  tests/test_benchmark_helpers.py::test_single_stage_init_fixture_runtime_seed_spec_loads
```

Backend / smoke / native-path gate:

```bash
$PY -m pytest -q \
  tests/test_backend.py \
  tests/test_jax_import_smoke.py \
  tests/integration/test_jax_native_path.py
```

Banana CPU/JAX parity gate:

```bash
$PY -m pytest -q \
  tests/objectives/test_fluxobjective_jax_parity.py \
  tests/objectives/test_integral_bdotn_jax.py \
  tests/geo/test_surface_rzfourier_jax.py \
  tests/geo/test_surface_fourier_jax.py \
  tests/geo/test_surface_objectives_jax.py \
  tests/integration/test_stage2_target_lane_purity.py \
  tests/integration/test_stage2_jax.py
```

Boozer focused wrapper gate:

```bash
$PY -m pytest -q \
  tests/geo/test_boozersurface_jax.py \
  tests/geo/test_boozer_derivatives_jax.py \
  -k "boozer or Boozer"
```

Single-stage CPU reference closure gate:

```bash
$PY -m pytest -q \
  tests/integration/test_single_stage_jax_cpu_reference.py
```

Full RZ surface parity gate:

```bash
$PY -m pytest -q \
  tests/geo/test_surface_rzfourier_jax.py \
  tests/geo/test_surface_rzfourier.py \
  tests/geo/test_surface_taylor.py \
  tests/geo/test_surface.py
```

Full non-RZ surface parity gate:

```bash
$PY -m pytest -q \
  tests/geo/test_surface_fourier_jax.py \
  tests/geo/test_surface.py \
  tests/geo/test_surface_taylor.py \
  tests/geo/test_surface_xyzfourier.py
```

Surface objective wrapper gate:

```bash
$PY -m pytest -q \
  tests/geo/test_surface_objectives_jax.py \
  tests/geo/test_surface_objectives.py \
  -k "ToroidalFlux or MajorRadius or PrincipalCurvature or QfmResidual or AspectRatio"
```

Release-gate unit/schema checks:

```bash
$PY -m pytest -q \
  tests/test_single_stage_cpp_jax_state_parity.py \
  tests/integration/test_single_stage_dof_mapping.py \
  tests/test_benchmark_helpers.py \
  -k "release_gate or fixed_state or coordinate_mapping or single_stage_parity_matrix"
```

Local CPU/C++ fixed-state and mapping artifacts:

```bash
mkdir -p .artifacts/parity
$PY benchmarks/single_stage_dof_mapping_proof.py \
  --output-json .artifacts/parity/coordinate-mapping-proof.json
$PY benchmarks/single_stage_cpp_jax_state_parity.py \
  --platform cpu \
  --output-json .artifacts/parity/fixed-state-cpu.json
```

Matrix gate after a same-seed run report exists:

```bash
$PY benchmarks/single_stage_parity_matrix.py \
  --fixed-state-parity-json .artifacts/parity/fixed-state-cpu.json \
  --coordinate-mapping-json .artifacts/parity/coordinate-mapping-proof.json \
  --parity-report-json <merged-same-seed-report.json> \
  --output-json .artifacts/parity/release-matrix-cpu.json \
  --output-md .artifacts/parity/release-matrix-cpu.md
```

Optional CUDA/H200 gate only after commit, push, and image:

```bash
SIMSOPT_HF_GPU_IMAGE=<registry>/simsopt-jax:cuda12-jax092 \
$PY benchmarks/hf_jobs/launch_production_gpu_proof.py \
  --repo-url https://github.com/jungdaesuh/simsopt.git \
  --repo-ref <branch> \
  --repo-sha <pushed-current-sha> \
  --hardware h200 \
  --platform cuda \
  --single-stage-mpol 10 \
  --single-stage-ntor 10 \
  --single-stage-jax-runtime-seed-spec benchmarks/fixtures/single_stage_seed_iota15 \
  --no-detach
```

## Review Checklist And Enforced Gates

Checklist items below are human review requirements unless an automated owner is named.

- [ ] Every new JAX value API has CPU oracle parity tests.
- [ ] Every new JAX derivative API has CPU oracle derivative parity or a Taylor test.
- [ ] Every Hessian API has explicit Hessian parity or second-order Taylor coverage.
- [ ] Every product-path CUDA claim has a current-sha CUDA artifact.
- [ ] No manifest row is marked complete from CPU-only evidence when CUDA evidence is required.
- [ ] No tolerance changes are made without updating the validation ladder contract and explaining why.
- [ ] No broad host API/I/O parity is treated as banana-required unless a banana workflow directly consumes it.
- [ ] Dirty unrelated files remain untouched during implementation.
- [x] `tests/docs/test_banana_parity_coverage_manifest.py` is wired into `.github/workflows/jax_smoke.yml` so manifest status edits run the machine-checkable banana inventory guard.

## Recommended Scope Decision

- [ ] If the goal is banana ship readiness, execute Set A and P5 artifact closure only. Do not implement Set B now.
- [ ] If the goal is zero-gap JAX-vs-C++/Python surface parity, execute Set B in milestones M2 through M6.
- [ ] Keep Set B as a full-upstream parity backlog until the product requirement changes.
