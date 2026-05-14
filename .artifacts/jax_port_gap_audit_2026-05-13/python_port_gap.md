# SIMSOPT Python -> JAX Port Gap Audit

**Date:** 2026-05-13
**Branch:** `gpu-purity-stage2-20260405`
**Worktree:** `/Users/suhjungdae/code/columbia/simsopt-jax`

## Executive Summary

Across `src/simsopt/{field,geo,objectives,solve,mhd,configs}/` this audit
classifies ~140 numerical class/function entries. Counts (rounded by
top-level entry, excluding pure helpers and stopping-criterion subclass
families):

- **PORTED:** ~38 (BiotSavart B+dB+A, SurfaceXYZTensorFourier and
  SurfaceXYZFourier evaluators, BoozerSurface penalty/exact solver,
  curve objectives, dipole/wireframe/permanent-magnet evaluation
  kernels, framedcurve frames, sampling, fluxobjective, integral
  BdotN, force kernels (B2/NetFluxes/SquaredMeanForce/LpForce/
  LpTorque/SquaredMeanTorque), Boozer analytic/radial/interp wrappers,
  CircularCoil/Toroidal/Poloidal/Dommaschk/Reiman/Mirror analytic
  fields, on-axis iota, RK fieldline + guiding-center tracing).
- **PARTIAL:** ~14 — gaps in second-Cartesian-derivative, vector
  potential coverage, missing Optimizable adapters around solved JAX
  kernels, missing surface kinds in JAX label adapters.
- **UNPORTED (high-priority):** ~17 — most notably the `Vmec/Spec`
  reduction `Quasisymmetry/QuasisymmetryRatioResidual`, `Boozer`
  (xform wrapper), `RedlGeomVmec/Boozer/VmecRedlBootstrapMismatch`,
  `Iota{Target,Weighted}`/`WellWeighted`, `ConstrainedProblem` and
  `LeastSquaresProblem`, `MGrid` reader, `QfmSurface` adapter,
  `CoilSet/ReducedCoilSet`, `NormalField/CoilNormalField`,
  `MagneticFieldSum/Multiply`, `Vmec.boundary` shape derivatives,
  `vmec_compute_geometry`, `B_cartesian`, full coilset distance
  toolbox).
- **NON-PORTABLE:** ~24 — VMEC/SPEC Fortran wrappers, MPI glue,
  NetCDF readers, VTK writers, file IO/serialisation, plotting,
  schema-only modules.

**Top 5 highest-leverage gaps**

1. **`SquaredFluxJAX` cannot consume composite fields** —
   `MagneticFieldSum`/`MagneticFieldMultiply` (`magneticfield.py:218,270`)
   have no JAX counterpart, so any pipeline that adds a vacuum field to
   a Biot-Savart coil set drops back to CPU.
2. **No `CoilSetJAX(Optimizable)`** —
   `field/coilset.py:18 CoilSet` and `:383 ReducedCoilSet` glue
   together coil flux + geometric penalties; downstream Stage 1/2
   workflows still hit `BiotSavart(sopp)` + CPU SquaredFlux. The
   underlying spec building blocks exist in
   `jax_core/specs.py:322 CoilSetDofExtractionSpec`.
3. **MHD reduction targets** —
   `mhd/boozer.py:244 Quasisymmetry` and
   `mhd/vmec_diagnostics.py:32 QuasisymmetryRatioResidual`,
   `:373 IotaTargetMetric`, `:486 IotaWeighted`, `:595 WellWeighted`
   compute the actual objectives consumed by VMEC/SPEC single-stage
   work. There is no JAX-side equivalent and no shared spec.
4. **`QfmSurfaceJAX` adapter missing** —
   pure-function residuals exist
   (`surfaceobjectives_jax.py:691 QfmResidualJAX`,
   plus `surface_qfm_residual_jax_from_dofs` and gradient at
   `:5605..5670`), but the LBFGS/SLSQP solver
   (`qfmsurface.py:117 minimize_qfm_penalty_constraints_LBFGS`,
   `:147 minimize_qfm_exact_constraints_SLSQP`) is CPU-only and
   blocks QFM-mode single-stage runs from going on-device.
5. **`BiotSavart` derivative ladder beyond `dB/dX`** —
   `BiotSavartJAX` (`biotsavart_jax_backend.py:878`) covers
   `B`, `dB/dX`, `A`, `dA/dX` and their VJPs, but lacks
   `d2B_by_dXdX`, `dB_by_dcoilcurrents`,
   `d2B_by_dXdcoilcurrents`, `d3B_by_dXdXdcoilcurrents`, and
   the matching `A`-side current-derivatives — these are the
   path that tracing/mgrid/boozer-interp consumers rely on
   today (`field/biotsavart.py:30..162`).

## Conventions

- **Status legend**: P = PORTED, p = PARTIAL, U = UNPORTED,
  N = NON-PORTABLE.
- **`sopp`**: Y = class/method dispatches into the `simsoptpp` C++
  extension; n = pure Python / JAX; — = N/A.
- **`Opt`**: Y = `Optimizable` subclass (relevant for adapter
  design); n = otherwise.
- **Risk** keys to the JAX porting cost (LOW = mechanical map,
  MED = solver/adapter scaffolding, HIGH = new physics binding or
  symbolic engine).

---

## `field/`

`src/simsopt/field/biotsavart.py`, `coil.py`, `boozermagneticfield.py`,
`magneticfield.py`, `magneticfieldclasses.py`, `coilset.py`,
`coilobjective.py`, `mgrid.py`, `normal_field.py`, `sampling.py`,
`selffield.py`, `force.py`, `wireframefield.py`, `tracing.py`,
`magnetic_axis_helpers.py`.

| Python symbol | Status | sopp | Opt | JAX target | Notes |
|---|---|---|---|---|---|
| `biotsavart.py:10 BiotSavart` | P | Y | Y | `biotsavart_jax_backend.py:878 BiotSavartJAX` (+ kernels in `jax_core/biotsavart.py`) | LOW: drop-in. |
| `biotsavart.py:30 dB_by_dcoilcurrents` (and `d2B_by_dXdcoilcurrents`, `d3B_by_dXdXdcoilcurrents`, `dA_by_dcoilcurrents`, `d2A_by_dXdcoilcurrents`, `d3A_by_dXdXdcoilcurrents`) | p | Y | Y | `biotsavart_jax_backend.py` (extend) | **PARTIAL**: JAX `BiotSavartJAX` has `B/dB/dX/A/dA/dX` + VJPs but no current-derivative API. MED risk: re-uses existing `grouped_biot_savart_*` primitives; needs per-coil-current Jacobians and `vgrad`/`v` plumbing parity with `B_and_dB_vjp` (`biotsavart_jax_backend.py:1628`). |
| `biotsavart.py: d2B_by_dXdX implicit` | U | Y | — | (none) | CPU `BiotSavart` exposes `d2B_by_dXdX` for tracing; JAX side has no second-Cartesian derivative. MED risk. |
| `coil.py:30 Coil` | P | Y | Y | `field/coil.py:30` (same class, JAX path enabled via spec) + `jax_core/specs.py:289 CoilSpec` | LOW. |
| `coil.py:99 RegularizedCoil`, `:263 CircularRegularizedCoil`, `:282 RectangularRegularizedCoil` | P | Y | Y | `coil.py` (same; selffield JAX in `selffield.py`) | LOW. |
| `coil.py:303 CurrentBase`, `:399 Current`, `:450 ScaledCurrent`, `:492 CurrentSum` | P | Y | Y | `jax_core/specs.py:259 CurrentValueSpec` + same module | LOW: pure pytree leaves. |
| `coil.py:531 apply_symmetries_to_curves`, `:561 apply_symmetries_to_currents`, `:694 coils_via_symmetries` | P | — | — | `jax_core/specs.py:273 CoilSymmetrySpec` | LOW: pure helpers, currently composable from JAX specs. |
| `coil.py:589 coils_to_vtk`, `:732 load_coils_from_makegrid_file`, `:782 coils_to_makegrid`, `:829 coils_to_focus` | N | — | — | — | File IO / VTK; not relevant to JAX hot path. |
| `coilobjective.py:82 CurrentPenalty` | P | n | Y | already JAX | LOW. |
| `coilset.py:18 CoilSet` | U | n | Y | (proposed `CoilSetJAX`) | **HIGH-PRIORITY**. Composes coils + surface + curve/distance penalties; currently calls CPU `BiotSavart` + CPU `SquaredFlux`. MED risk: refactor to use `BiotSavartJAX` + `SquaredFluxJAX` and `CoilSetDofExtractionSpec`. |
| `coilset.py:383 ReducedCoilSet` | U | n | Y | (proposed `ReducedCoilSetJAX`) | Derived from CoilSet; same blocker. MED. |
| `magneticfield.py:43 MagneticField` (base) | P | Y | Y | shared base + `_is_jax_native_field` gate at `magneticfield.py:12` | LOW: most JAX subclasses derive from this. |
| `magneticfield.py:218 MagneticFieldMultiply` | U | Y | Y | (none) | **HIGH-PRIORITY**: blocks scalar-rescaled JAX fields; would need a JAX adapter or first-class composition op in the native-field gate at `magneticfield.py:23`. LOW–MED. |
| `magneticfield.py:270 MagneticFieldSum` | U | Y | Y | (none) | **HIGH-PRIORITY**: blocks combining `BiotSavartJAX` with vacuum analytic fields; today such compositions silently fall back to CPU via the gate at `magneticfield.py:23`. LOW–MED. |
| `magneticfieldclasses.py:24 ToroidalField` | P | n | Y | `toroidal_field_jax.py:24 ToroidalFieldJAX` | full B + dB + d2B + A coverage. LOW. |
| `magneticfieldclasses.py:128 PoloidalField` | p | n | Y | `poloidal_field_jax.py:21 PoloidalFieldJAX` | **PARTIAL**: missing `_d2B_by_dXdX_impl` and `_A_impl` (CPU has neither for PoloidalField, so de-facto parity; flag for tracing-mgrid use cases). LOW. |
| `magneticfieldclasses.py:236 ScalarPotentialRZMagneticField` | U | Y | Y | (proposed) | **BLOCKED**: see `.artifacts/jax_native_remaining_2026-05-13/HANDOFF.md` (item 23). Requires a SymPy->JAX printer (~600-800 LOC) since the upstream class ingests SymPy strings. HIGH. The `scalar_potential_rz_jax.py:17 ScalarPotentialRZMagneticFieldJAX` stub exists but is non-functional for the SymPy entry-point. |
| `magneticfieldclasses.py:335 CircularCoil` | p | n | Y | `circular_coil_jax.py:21 CircularCoilJAX` | **PARTIAL**: missing `_A_impl` and `_d2B_by_dXdX_impl`. CPU `magneticfieldclasses.py:502 CircularCoil._A_impl` exists. LOW. |
| `magneticfieldclasses.py:572 DipoleField` | p | n | Y | `dipole_field_jax.py:163 DipoleFieldJAX` (+ kernels in `jax_core/dipole_field.py`) | **PARTIAL**: no `_d2B_by_dXdX_impl`; `_A_impl` present (line 286). LOW. |
| `magneticfieldclasses.py:752 Dommaschk` | p | n | Y | `dommaschk_jax.py:44 DommaschkJAX` | **PARTIAL**: missing `_A_impl` and `_d2B_by_dXdX_impl`. LOW. |
| `magneticfieldclasses.py:802 Reiman` | p | n | Y | `reiman_jax.py:22 ReimanJAX` | **PARTIAL**: missing `_d2B_by_dXdX_impl`; CPU also omits `_A_impl`. LOW. |
| `magneticfieldclasses.py:847 UniformInterpolationRule`, `:851 ChebyshevInterpolationRule` | P | Y | n | `jax_core/regular_grid_interp.py UniformInterpolationRule` | LOW: pure tables. |
| `magneticfieldclasses.py:855 InterpolatedField` | P | Y | Y | `interpolated_field_jax.py:142 InterpolatedFieldJAX` (+ `jax_core/interpolated_field.py`) | LOW. |
| `magneticfieldclasses.py:919 MirrorModel` | p | n | Y | `mirror_model_jax.py:21 MirrorModelJAX` | **PARTIAL**: missing `_d2B_by_dXdX_impl` and `_A_impl`. LOW. |
| `boozermagneticfield.py:27 BoozerMagneticField` | P | Y | Y | `boozermagneticfield_jax.py:1271 InterpolatedBoozerFieldJAX` (base surface mirrored) | LOW. |
| `boozermagneticfield.py:110 BoozerAnalytic` | P | Y | Y | `boozermagneticfield_jax.py:1054 BoozerAnalyticJAX` (kernels at `jax_core/boozer_analytic.py`) | LOW. |
| `boozermagneticfield.py:302 BoozerRadialInterpolant` | P | Y | Y | `boozermagneticfield_jax.py:810 BoozerRadialInterpolantJAX` (kernels at `jax_core/boozer_radial_interp.py`, frozen state at `:92`) | LOW. |
| `boozermagneticfield.py:1087 InterpolatedBoozerField` | P | Y | Y | `boozermagneticfield_jax.py:1271 InterpolatedBoozerFieldJAX` (frozen state via `jax_core/interpolated_boozer_field.py`) | LOW. |
| `coil.py:589 coils_to_vtk`, `coil.py:782 coils_to_makegrid`, `:829 coils_to_focus`, `:732 load_coils_from_makegrid_file` | N | — | — | — | I/O. |
| `mgrid.py:22 MGrid` | U | n | n | (proposed `MGridJAX`) | NetCDF read + cylindrical grid evaluation; reader is non-portable (file IO) but evaluator/B-grid sampling on-device could live in `jax_core/regular_grid_interp.py`. **MED** if extending kernels; **N** for the file reader itself. |
| `normal_field.py:20 NormalField` | U | n | Y | (proposed `NormalFieldJAX`) | Used to couple SPEC plasma to coils via `coilset.py`. Holds Fourier coefficients; gradient adaptation maps coil dofs to surface flux. MED. |
| `normal_field.py:522 CoilNormalField` | U | n | Y | (proposed) | Derived from `NormalField`. MED. |
| `sampling.py:4 draw_uniform_on_curve`, `:28 draw_uniform_on_surface` | P | n | — | `sampling_jax.py` -> `jax_core/sampling.py:81 draw_uniform_on_curve_jax`, `:119 draw_uniform_on_surface_jax`, `:48 sample_weighted_indices_jax` | LOW. |
| `selffield.py:63 regularization_circ`, `:80 regularization_rect`, `:98 B_regularized_singularity_term`, `:117 B_regularized_pure` | P | n | — | `jax_core/finitebuild.py` + same module is already pure JAX | LOW. |
| `force.py:1229 B2Energy`, `:1457 NetFluxes`, `:1699 SquaredMeanForce`, `:2059 LpCurveForce`, `:2446 LpCurveTorque`, `:2799 SquaredMeanTorque` | P | n | Y | already JAX-native via `jax_core` helpers | LOW. |
| `wireframefield.py:13 WireframeField` | P | Y | Y | `wireframefield_jax.py:37 WireframeFieldJAX` (kernels at `jax_core/wireframe.py`) | LOW. |
| `wireframefield.py:110 enclosed_current` | P | n | — | derivable from JAX wireframe primitives | LOW. |
| `tracing.py:150 trace_particles_boozer` | p | Y | — | `tracing.py:316 _trace_particles_boozer_jax` (gated dual path) | **PARTIAL**: JAX path implemented for vacuum guiding-center variants only; full-orbit Boozer + non-vacuum branches remain in `simsoptpp`. MED. |
| `tracing.py:487 trace_particles` | p | Y | — | `tracing.py:695 _trace_particles_jax_guiding_center_vacuum`, `:850 _trace_particles_jax_fullorbit_vacuum` | **PARTIAL**: JAX path covers vacuum guiding-center and vacuum full-orbit only; production-grade collisional / mixed branches at `:648-:662` still call `sopp.particle_*`. MED. |
| `tracing.py:1013 trace_particles_starting_on_curve`, `:1092 trace_particles_starting_on_surface` | p | Y | — | inherits from `trace_particles` JAX path | **PARTIAL**: rest after sampling. LOW. |
| `tracing.py:1172 compute_resonances`, `:1327 compute_toroidal_transits`, `:1360 compute_poloidal_transits` | U | Y | — | (proposed in `jax_core/tracing.py`) | Post-processing analysis on traces. Uses `sopp.get_phi` (line 1347/1403). LOW. |
| `tracing.py:1422 compute_fieldlines` | p | Y | — | `tracing.py:1641 _compute_fieldlines_jax` (Dormand-Prince), kernels at `jax_core/tracing.py:801 trace_fieldline` | **PARTIAL**: JAX path active for the dual-mode lane; full feature parity with `sopp.fieldline_tracing` (e.g. some stopping criteria) requires `jax_core/tracing.py` extensions. LOW. |
| `tracing.py:1740 particles_to_vtk` | N | — | — | — | VTK output. |
| `tracing.py:1757..1915 *StoppingCriterion` (Levelset, MinToroidalFlux, MaxToroidalFlux, ToroidalTransit, Iteration, Min/Max R/Z) | P | Y | n | `jax_core/tracing.py:346..428` stopping criteria classes | LOW. |
| `magnetic_axis_helpers.py:8 compute_on_axis_iota` | P | n | — | `jax_core/magnetic_axis_helpers.py:524 on_axis_iota_rk` (Dormand-Prince integration) | LOW. |

---

## `geo/`

`src/simsopt/geo/curve.py`, `curve*fourier*.py`, `surface*.py`,
`framedcurve*.py`, `boozersurface*.py`, `surfaceobjectives*.py`,
`curveobjectives.py`, `qfmsurface.py`, `finitebuild.py`,
`strain_optimization.py`, `accessibility.py`, `hull.py`, `ports.py`,
`orientedcurve.py`, `permanent_magnet_grid*.py`,
`wireframe_toroidal.py`.

| Python symbol | Status | sopp | Opt | JAX target | Notes |
|---|---|---|---|---|---|
| `curve.py:458 Curve` (base) | P | n | Y | shared base used by `JaxCurve`; pure-Python with JAX numerics | LOW. |
| `curve.py:978 JaxCurve` | P | Y | Y | same class; uses JAX for gradient/jit | LOW. |
| `curve.py:1217 RotatedCurve` | P | Y | Y | spec covered via rotmat in `jax_core/specs.py:289 CoilSpec` | LOW. |
| `curve.py:2008 CurveCWSFourier` | P | Y | Y | `jax_core/specs.py:598 CurveCWSFourierRZSpec` + `curve_geometry.py` (works for RZ-Fourier surface) | LOW for RZFourier; XYZTensorFourier branch is still CPU-tied via `surf.gamma()` lookups inside `__init__` (`curvecwsfourier.py:172 CurveCWSFourierCPP` is the C++ variant). MED for full XYZTensor coverage. |
| `curvexyzfourier.py:99 CurveXYZFourier`, `:346 JaxCurveXYZFourier` | P | Y | Y | `jax_core/specs.py:26 CurveXYZFourierSpec` + `curve_geometry.py` | LOW. |
| `curverzfourier.py:60 CurveRZFourier` | P | Y | Y | `jax_core/specs.py:42 CurveRZFourierSpec` | LOW. |
| `curveplanarfourier.py:101 CurvePlanarFourier`, `:215 JaxCurvePlanarFourier` | P | Y | Y | `jax_core/specs.py:60 CurvePlanarFourierSpec` | LOW. |
| `curvehelical.py:41 CurveHelical` | P | n | Y | `jax_core/specs.py:76 CurveHelicalSpec` | LOW. |
| `curvexyzfouriersymmetries.py:62 CurveXYZFourierSymmetries` | P | n | Y | `jax_core/specs.py:96 CurveXYZFourierSymmetriesSpec` | LOW. |
| `curveperturbed.py:152 CurvePerturbed` (+ `:26 GaussianSampler`, `:113 PerturbationSample`) | P | Y | Y | `jax_core/specs.py:623 CurvePerturbedSpec`, kernels in `jax_core/curve_geometry.py:321` | LOW. |
| `curvecwsfourier.py:172 CurveCWSFourierCPP` | U | Y | Y | (subset of `CurveCWSFourier` JAX path) | C++-only variant of curve on surface; subsumed by `CurveCWSFourier` once XYZTensor surface support lands. LOW once that lands. |
| `orientedcurve.py:92 OrientedCurveXYZFourier` | P | n | Y | `jax_core/curve_geometry.py` (uses `JaxCurve`) | LOW. |
| `surface.py:35 Surface` (base) | P | Y | Y | shared base; consumed by all subclasses | LOW. |
| `surface.py:936 SurfaceClassifier` | P | n | n | `jax_core/surface_classifier.py` | LOW. |
| `surface.py:1052 SurfaceScaled` | U | n | Y | (proposed `SurfaceScaledJAX` wrapper) | Wraps a surface and rescales DOFs; adapter pattern. LOW. |
| `surfacerzfourier.py:192 SurfaceRZFourier` | P | Y | Y | `jax_core/surface_rzfourier.py` + `jax_core/specs.py:432 SurfaceRZFourierSpec` (gamma/gammadash/normal/area/volume + their from_dofs variants) | LOW. |
| `surfacerzfourier.py:2520 SurfaceRZPseudospectral` | U | n | Y | (none) | Spectral collocation surface; not part of JAX seed path. MED. |
| `surfacexyzfourier.py:72 SurfaceXYZFourier` | P | Y | Y | `jax_core/specs.py:462 SurfaceXYZFourierSpec` + `surface_fourier_jax.py` functions (`surface_gamma_lin`, `surface_gammadash{1,2}_lin`, `*_from_dofs`, derivative variants at `:1463..1986`) | LOW. |
| `surfacexyztensorfourier.py:53 SurfaceXYZTensorFourier` | P | Y | Y | `surface_fourier_jax.py:518 surface_gamma`/+derivatives + `surface_fourier_jax_cpu_ordered.py` + `jax_core/specs.py:490 SurfaceXYZTensorFourierSpec` | LOW. |
| `surfacegarabedian.py:15 SurfaceGarabedian` | p | Y | Y | `jax_core/specs.py:121 SurfaceGarabedianSpec` (+ `garabedian_to_rzfourier_spec`) | **PARTIAL**: spec + conversion to RZFourier exist; no direct gamma/derivative kernels matching CPU `SurfaceGarabedian` get_dofs/set_dofs surface; consumers go through the RZFourier translation. LOW for the supported pipeline; MED if full Garabedian-native eval is required. |
| `surfacehenneberg.py:21 SurfaceHenneberg` | P | Y | Y | `jax_core/surface_henneberg.py:213 surface_henneberg_gamma_from_spec` + `jax_core/specs.py:155 SurfaceHennebergSpec` | LOW. |
| `framedcurve.py:41 FramedCurve`, `:90 FramedCurveFrenet`, `:271 FramedCurveCentroid`, `:427 FrameRotation`, `:473 ZeroRotation` | P | Y | Y | `framedcurve_jax.py:136 FrameRotationJAX`, `:214 ZeroRotationJAX`, `:262 FramedCurveFrenetJAX`, `:307 FramedCurveCentroidJAX` (+ `jax_core/framedcurve.py`) | LOW. |
| `finitebuild.py:32 CurveFilament`, `:242 create_multifilament_grid` | P | n | Y | `jax_core/finitebuild.py` + `jax_core/specs.py:653 CurveFilamentSpec` | LOW. |
| `strain_optimization.py:53 LPBinormalCurvatureStrainPenalty`, `:117 LPTorsionalStrainPenalty`, `:180 CoilStrain` | P | n | Y | same module (uses JAX inside) | LOW. |
| `curveobjectives.py:157 CurveLength`, `:236 LpCurveCurvature`, `:290 LpCurveCurvatureBarrier`, `:361 LpCurveTorsion`, `:580 CurveCurveDistanceBarrier`, `:668 CurveCurveDistance`, `:906 CurveSurfaceDistance`, `:1063 ArclengthVariation`, `:1159 MeanSquaredCurvature`, `:1190 MinimumDistance`, `:1194 LinkingNumber`, `:1330 FramedCurveTwist`, `:1457 MinCurveCurveDistance` | P | n | Y | same module (JAX-native) | LOW. |
| `boozersurface.py:214 BoozerSurface` | P | Y | Y | `boozersurface_jax.py:3190 BoozerSurfaceJAX` (penalty/LBFGS, LS, newton, exact) | LOW. |
| `surfaceobjectives.py:162 AspectRatio`, `:220 Area`, `:278 Volume`, `:336 ToroidalFlux`, `:456 PrincipalCurvature` | p | Y | Y | `surfaceobjectives_jax.py:609 AspectRatioJAX`, `:644 PrincipalCurvatureJAX`; `Area`/`Volume`/`ToroidalFlux` only have **pure helpers** at `surface_fourier_jax.py:2399 surface_volume`, `:2420 surface_area`, `label_constraints_jax.py:25 toroidal_flux_jax` and **no `Optimizable` adapter** | **PARTIAL**: `AreaJAX`/`VolumeJAX`/`ToroidalFluxJAX` adapters do not exist. JAX label_constraints_jax.toroidal_flux_jax is a free function only. MED: build adapter classes around the existing pure-function kernels. |
| `surfaceobjectives.py:799 QfmResidual` | P | n | Y | `surfaceobjectives_jax.py:691 QfmResidualJAX` (+ `:567 surface_qfm_residual_jax_from_dofs`) | LOW. |
| `surfaceobjectives.py:878 MajorRadius`, `:930 NonQuasiSymmetricRatio`, `:1159 Iotas`, `:1297 BoozerResidual` | P | n | Y | `surfaceobjectives_jax.py:2140 BoozerResidualJAX`, `:2303 IotasJAX`, `:2348 MajorRadiusJAX`, `:2409 NonQuasiSymmetricRatioJAX` | LOW. |
| `surfaceobjectives.py: parameter_derivatives` (free fn, line 765) | P | n | — | same module + JAX kernels | LOW. |
| `qfmsurface.py:9 QfmSurface` | U | n | n (GSONable) | (proposed `QfmSurfaceJAX` adapter) | **HIGH-PRIORITY**. Wraps CPU SciPy `L-BFGS-B`/`SLSQP` on the residual. Residual + Jacobian are already in `surfaceobjectives_jax.py:691 QfmResidualJAX` and `:5605..5670`. MED: implement adapter that uses `optimizer_jax` LM/Newton lanes on the JAX residual. |
| `accessibility.py:28 PortSize`, `:402 ProjectedEnclosedArea`, `:473 DirectedFacingPort` + pure helpers | P | n | Y | same module (JAX-native, value/grad/hessian written in JAX) | LOW. |
| `hull.py:7 hull2D` | U | n | n | (proposed) | 2D convex hull util; pure NumPy/SciPy. LOW if needed; not in JAX hot path. |
| `ports.py:20 PortSet`, `:388 Port`, `:449 CircularPort`, `:737 RectangularPort` | U | n | n | (proposed) | Geometric ports + distance penalties for accessibility. Used by `accessibility.py`. NumPy-only. MED. |
| `permanent_magnet_grid.py:14 PermanentMagnetGrid` | P | Y | n | `permanent_magnet_grid_jax.py:34 PermanentMagnetGridJAX` (kernels at `jax_core/pm_optimization.py`) | LOW. |
| `wireframe_toroidal.py:17 ToroidalWireframe` | U | n | n | (proposed) | Topology + segment-grid builder for wireframes; pure Python/NumPy. The downstream **field evaluation** for any wireframe is already JAX-ported (`wireframefield_jax.py`). MED if/when topology editing needs to live on-device. |

---

## `objectives/`

| Python symbol | Status | sopp | Opt | JAX target | Notes |
|---|---|---|---|---|---|
| `fluxobjective.py:20 SquaredFlux` | P | Y | Y | `fluxobjective_jax.py:110 SquaredFluxJAX` (+ `integral_bdotn_jax.py:38 residual_BdotN`, `:93 integral_BdotN`, `:86 signed_BdotN_flux`) | LOW. |
| `integral_bdotn_jax.py` (functions) | P | — | — | same module | LOW. |
| `functions.py:20 Identity`, `:75 Adder`, `:115 Rosenbrock`, `:198 TestObject1`, `:248 Affine`, `:275 Failer`, `:318 Beale` | P | n | Y | same module | LOW. Test fixtures, lightweight. |
| `utilities.py:63 MPIOptimizable` | N | n | Y | — | MPI ensemble glue. |
| `utilities.py:122 MPIObjective` | N | n | Y | — | MPI ensemble glue. |
| `utilities.py:177 QuadraticPenalty` | P | n | Y | same module, pure Python, autodiff-safe | LOW. |
| `utilities.py:22 forward_backward` | P | n | — | same module (pure NumPy) | LOW. |
| `utilities.py:226 Weight` | P | n | n | same module | LOW. |
| `constrained.py:27 ConstrainedProblem` | U | n | Y | (proposed `ConstrainedProblemJAX`) | Generic outer-loop wrapper used by `solve/{serial,mpi}.constrained_*_solve`. Calls `prob.J()`/`prob.dJ()` per child. **No JAX-native equivalent**: the wrappers do not assume an autodiff inner objective. MED. |
| `least_squares.py:30 LeastSquaresProblem` | U | n | Y | (proposed `LeastSquaresProblemJAX`) | Standard LS bundle. MED. |
| `stage2_target_objective_jax.py:*` (FinalSpecBundle, Stage2TargetObjectiveTerm, Stage2TargetReportingSummary, Stage2PenaltyConfig, Stage2TargetOptimizerState, Stage2TargetObjectiveBundle + helpers, `build_stage2_target_objective`) | P | — | — | same module (no CPU counterpart) | This is the JAX-native Stage2 outer pipeline. LOW. |

---

## `solve/`

| Python symbol | Status | sopp | Opt | JAX target | Notes |
|---|---|---|---|---|---|
| `serial.py:32 least_squares_serial_solve`, `:179 serial_solve`, `:288 constrained_serial_solve` | U | n | — | (proposed `target_*_solve` family in `geo/optimizer_jax.py`) | Driver glue that calls `scipy.optimize`. Already mirrored by JAX-side optimizer (`optimizer_jax.py target_minimize`, `target_scipy_minimize_value_and_grad`). MED to wrap into a public `solve_jax.py`. |
| `mpi.py:77 least_squares_mpi_solve`, `:310 constrained_mpi_solve` | N | n | — | — | MPI ensemble glue. |
| `permanent_magnet_optimization.py:13 prox_l0`, `:42 prox_l1`, `:67 projection_L2_balls`, `:87 setup_initial_condition`, `:118 relax_and_split`, `:278 GPMO` | P | Y | — | `permanent_magnet_optimization_jax.py:272 projection_L2_balls_jax`, `:283 prox_l0_jax`, `:294 prox_l1_jax`, `:309 setup_initial_condition_jax`, `:333..577 GPMO_*`, `:620 relax_and_split_jax` (+ `jax_core/pm_optimization.py`) | LOW. |
| `wireframe_optimization.py:18 optimize_wireframe`, `:334 bnorm_obj_matrices`, `:456 rcls_wireframe`, `:553 gsco_wireframe`, `:723 regularized_constrained_least_squares` | P | Y | — | `wireframe_optimization_jax.py:102 regularized_constrained_least_squares_jax`, `:296 greedy_stellarator_coil_optimization_jax`, `:519 gsco_wireframe_jax` (+ `jax_core/wireframe.py`) | LOW. |

---

## `mhd/`

All entries here couple to Fortran VMEC/SPEC, NetCDF, MPI ensembles,
external `vmec` / `spec` Python packages, or `pyoculus`. Numerical hot
paths that *consume* VMEC output (Boozer transform, geometric
post-processing) are still defined here and remain unported.

| Python symbol | Status | sopp | Opt | JAX target | Notes |
|---|---|---|---|---|---|
| `vmec.py:114 Vmec` | N | n | Y | — | Wraps Fortran `vmec`. Non-portable. |
| `spec.py:59 Spec`, `:1321 Residue` | N | n | Y | — | Wraps Fortran `spec` + `pyoculus`. |
| `boozer.py:37 Boozer` | N | n | Y | — | Wraps BOOZXFORM Fortran code. (The JAX-side `BoozerRadialInterpolantJAX` ingests its output via `freeze_boozer_radial_state`.) |
| `boozer.py:244 Quasisymmetry` | U | n | Y | (proposed) | **HIGH-PRIORITY**. Modal residual on BOOZXFORM output; pure NumPy reduction. Could live on-device once Boozer payload is JAX-frozen. MED. |
| `virtual_casing.py:33 VirtualCasing` | N | n | n | — | Wraps `virtual_casing` Fortran code. |
| `vmec_diagnostics.py:32 QuasisymmetryRatioResidual` | U | n | Y | (proposed) | **HIGH-PRIORITY**. Computes the M-N-weighted residual from VMEC output; once data lands in JAX, the reduction is a clean port. MED. |
| `vmec_diagnostics.py:373 IotaTargetMetric`, `:486 IotaWeighted`, `:595 WellWeighted` | U | n | Y | (proposed) | **HIGH-PRIORITY**. MHD reductions over VMEC profiles; pure NumPy. MED. |
| `vmec_diagnostics.py:291 B_cartesian`, `:713 vmec_splines`, `:850 VmecGeometryResults`, `:1208 vmec_compute_geometry`, `:1770 vmec_fieldlines` | U | n | n | (proposed) | NumPy/SciPy post-processing over Vmec output; non-trivial because of profile splines + fieldline integration. HIGH effort if fully ported. |
| `bootstrap.py:27 compute_trapped_fraction`, `:173 j_dot_B_Redl`, `:405 RedlGeomVmec`, `:517 RedlGeomBoozer`, `:635 VmecRedlBootstrapMismatch` | U | n | Y | (proposed) | Bootstrap-current reduction over VMEC / Boozer output. MED. |
| `profiles.py:26 Profile`, `:60 ProfileSpec`, `:142 ProfilePolynomial`, `:167 ProfileScaled`, `:195 ProfileSpline`, `:246 ProfilePressure` | U | n | Y | (proposed) | Profile evaluators (pure NumPy / SciPy splines). LOW. |

---

## `configs/`

| Python symbol | Status | sopp | Opt | JAX target | Notes |
|---|---|---|---|---|---|
| `zoo.py:38 get_data` (dispatch), `:458 get_ncsx_data`, `:488 get_hsx_data`, `:518 get_giuliani_data`, `:553 get_w7x_data`, `:588 download_ID_from_QUASR_database`, `:674 _prune_cache` | N | n | — | — | Bundled-configuration factories that read data files and return `CurveXYZFourier`/`Current`/`CurveRZFourier` lists. The factories themselves are non-portable IO + dataset metadata. |
| `LHD_like.py:6 get_LHD_like_data` | N | n | — | — | Same pattern, helical-coil LHD geometry. |

---

## Unported high-priority (recap)

The following entries are not covered by any JAX module today and
block on-device single-stage / coil-set workflows or VMEC-driven
optimization. Listed in approximate leverage order:

1. **`coilset.py:18 CoilSet` / `:383 ReducedCoilSet`** — primary
   composition layer for coil + surface optimization; without a JAX
   variant Stage 2 work goes through CPU `BiotSavart`.
2. **`magneticfield.py:218 MagneticFieldMultiply` and
   `:270 MagneticFieldSum`** — composition of fields. Without them
   pipelines that overlay `BiotSavartJAX` with vacuum analytic
   fields revert to CPU evaluation.
3. **`qfmsurface.py:9 QfmSurface`** — QFM solver; residual is
   JAX-native but the optimizer plumbing isn't.
4. **`mhd/boozer.py:244 Quasisymmetry`,
   `mhd/vmec_diagnostics.py:32 QuasisymmetryRatioResidual`,
   `:373 IotaTargetMetric`, `:486 IotaWeighted`,
   `:595 WellWeighted`** — MHD-reduction objectives.
5. **`mhd/bootstrap.py:173 j_dot_B_Redl`, `:405 RedlGeomVmec`,
   `:517 RedlGeomBoozer`, `:635 VmecRedlBootstrapMismatch`** —
   bootstrap-current targets.
6. **`objectives/constrained.py:27 ConstrainedProblem` and
   `objectives/least_squares.py:30 LeastSquaresProblem`** — outer
   problem objects that the user-visible API exposes. JAX optimization
   today goes through `target_minimize` directly without a `Problem`
   wrapper.
7. **`magneticfieldclasses.py:236 ScalarPotentialRZMagneticField`** —
   blocked on SymPy->JAX translator; documented in
   `.artifacts/jax_native_remaining_2026-05-13/HANDOFF.md`.
8. **`field/normal_field.py:20 NormalField` and `:522 CoilNormalField`** —
   plasma boundary <-> coil flux coupling for SPEC integration.
9. **`mgrid.py:22 MGrid`** — NetCDF mgrid reader. File IO is
   non-portable but the underlying grid evaluator is reusable.
10. **`mhd/vmec_diagnostics.py:1208 vmec_compute_geometry`,
    `:1770 vmec_fieldlines`** — VMEC post-processing utilities that
    several other objectives need.

---

## Partial gaps (recap)

For each entry below, the JAX adapter exists but is missing a precise
slice of the CPU surface. The "missing" column lists exact methods or
keyword paths.

| Python symbol | JAX file | Missing surface |
|---|---|---|
| `biotsavart.py:30..162 BiotSavart` (current-derivative ladder) | `biotsavart_jax_backend.py:878 BiotSavartJAX` | `dB_by_dcoilcurrents`, `d2B_by_dXdcoilcurrents`, `d3B_by_dXdXdcoilcurrents`, `dA_by_dcoilcurrents`, `d2A_by_dXdcoilcurrents`, `d3A_by_dXdXdcoilcurrents`, and `d2B_by_dXdX` |
| `magneticfieldclasses.py:335 CircularCoil` | `circular_coil_jax.py:21 CircularCoilJAX` | `_A_impl`, `_d2B_by_dXdX_impl` |
| `magneticfieldclasses.py:572 DipoleField` | `dipole_field_jax.py:163 DipoleFieldJAX` | `_d2B_by_dXdX_impl` |
| `magneticfieldclasses.py:752 Dommaschk` | `dommaschk_jax.py:44 DommaschkJAX` | `_A_impl`, `_d2B_by_dXdX_impl` |
| `magneticfieldclasses.py:802 Reiman` | `reiman_jax.py:22 ReimanJAX` | `_d2B_by_dXdX_impl` |
| `magneticfieldclasses.py:128 PoloidalField` | `poloidal_field_jax.py:21 PoloidalFieldJAX` | `_A_impl` (CPU also lacks; document as parity), `_d2B_by_dXdX_impl` |
| `magneticfieldclasses.py:919 MirrorModel` | `mirror_model_jax.py:21 MirrorModelJAX` | `_A_impl`, `_d2B_by_dXdX_impl` |
| `surfaceobjectives.py:220 Area`, `:278 Volume`, `:336 ToroidalFlux` | pure JAX helpers in `surface_fourier_jax.py:2399 surface_volume`, `:2420 surface_area`, `label_constraints_jax.py:25 toroidal_flux_jax` | **No `Optimizable` adapter classes** (`AreaJAX`, `VolumeJAX`, `ToroidalFluxJAX`) — single-stage label-constraint plumbing has to wrap by hand or use CPU classes. |
| `surfacegarabedian.py:15 SurfaceGarabedian` | `jax_core/specs.py:121 SurfaceGarabedianSpec` (+ Garabedian->RZFourier conversion) | No direct gamma/derivative kernels matching `SurfaceGarabedian.gamma()`; only via RZ-Fourier conversion. Adequate for current consumers; gap if Garabedian-native eval is required. |
| `field/tracing.py:150 trace_particles_boozer`, `:487 trace_particles` | `tracing.py:316 _trace_particles_boozer_jax`, `:695 _trace_particles_jax_guiding_center_vacuum`, `:850 _trace_particles_jax_fullorbit_vacuum` | JAX path covers vacuum guiding-center and vacuum full-orbit. Non-vacuum / collisional / `sopp.particle_*` branches at `tracing.py:280, :648, :662` still need C++ for production. |
| `field/tracing.py:1422 compute_fieldlines` | `tracing.py:1641 _compute_fieldlines_jax` | JAX path active, but full stopping-criterion parity with `sopp.fieldline_tracing` needs `jax_core/tracing.py` extensions for any criteria not yet listed in `tracing.py:1519 _translate_stopping_criteria_to_jax`. |
| `geo/curve.py:2008 CurveCWSFourier` (XYZTensorFourier branch) | `jax_core/specs.py:598 CurveCWSFourierRZSpec` + `curve_geometry.py` | Spec covers the RZ-Fourier surface case (`surf_type == "RZ_Fourier"`); the XYZTensorFourier branch in `curve.py:2040` still depends on `surf.gamma()` host evaluations. |

---

## Methodology notes / Uncertain entries

- **UNCLEAR**: Whether `field/coilset.py:18 CoilSet`'s `flux_penalty`,
  `length_penalty`, etc. methods can be retargeted at JAX
  equivalents (`SquaredFluxJAX`, `CurveLength`, ...) by passing a
  `BiotSavartJAX` instead of `BiotSavart` without further refactor.
  The implementation reads `self.bs.B()` in places and may need a
  reshape of the cache contract.
- **UNCLEAR**: Whether `magneticfield.py:23 _raise_if_strict_jax_mixed_composition`
  could be relaxed to allow `MagneticFieldSum([BiotSavartJAX, ToroidalFieldJAX])`
  rather than blocking it. A first step toward a `MagneticFieldSumJAX`
  would be to define the JAX-side reduction operator over the
  per-field B/dB/A pytrees.
- **NOT verified**: Each `*_jax_*.py` JAX function was tested against
  its CPU twin via the parity ladder, but this audit did not
  re-execute the tests. Status reflects file presence + API surface,
  not numerical parity. Numerical parity status is tracked in
  `benchmarks/validation_ladder_contract.py` and
  `.artifacts/jax_test_audit_2026-04-25/`.
- The `optimize/budget.py` module (single file) is not in the
  enumerated audit scope; it manages compute-budget heuristics and
  has no computational hot path.
