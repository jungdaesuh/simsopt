# JAX Test Mirror Gap Audit (Part 4)

**Date:** 2026-05-13
**Scope:** Pair every `tests/<dir>/test_*.py` (non-`_jax.py`) original with its
JAX-mirror counterpart in `tests/<dir>/test_*_jax*.py` and identify unmirrored
scenarios where the underlying functionality IS JAX-ported.

Oracle-lint rules in `tests/REVIEWER_ORACLE_LINT.md:1-86` apply to all
"PARTIAL" / "MIRRORED" claims; a mirror that re-asserts wrapper-equals-kernel
or jax-equals-host-that-routes-through-jax is flagged.

---

## Executive Summary

- Python originals walked (non-`_jax.py`, `tests/**/test_*.py`): **63 files**
  with **≈ 524 `def test_*` functions**.
- JAX mirror files (`tests/**/test_*_jax*.py` plus dedicated closeouts): **51
  files** with **≈ 1,160 `def test_*` functions** (most of the volume comes
  from `tests/geo/test_boozersurface_jax.py:237`, `test_surface_objectives_jax.py`,
  and `test_surface_rzfourier_jax.py`).
- Coarse mirroring rate by Python-test scenario coverage: **≈ 60 % MIRRORED or
  PARTIAL**, **≈ 8 % UNMIRRORED with a real JAX surface to assert against**,
  and **≈ 32 % N/A — NOT PORTED** (file IO, plotting, MPI, VMEC, SPEC,
  bootstrap, virtual casing, particle tracing C++ reference, focus loader,
  `to_vtk`).

### Top 5 highest-leverage UNMIRRORED tests (cited file:line)

1. **`tests/field/test_biotsavart.py:402` `test_biotsavart_vector_potential_coil_current_taylortest`
   + `tests/field/test_biotsavart.py:276` `test_biotsavart_coil_current_taylortest`** —
   linearity of `B`/`A` and their derivatives in coil current (including
   `d3B_by_dXdXdcoilcurrents` and `d3A_by_dXdXdcoilcurrents`). JAX surface
   exposes `BiotSavartJAX.B_vjp` / `A_vjp` / `B_and_dB_vjp` / `A_and_dA_vjp` /
   `d2A_by_dXdX` (see `src/simsopt/field/biotsavart_jax_backend.py:529,
   1616-1628`), but the JAX parity suite stops at first derivatives; no
   coil-current Taylor or third-mixed-derivative check exists in
   `test_biotsavart_jax_parity.py:229-528` or `test_biotsavart_jax.py:344-1008`.

2. **`tests/geo/test_curve_objectives.py:780` `test_curve_minimum_distance_taylor_test`
   + `:826` `test_curve_minimum_distance_barrier_taylor_test` +
   `:862` `test_curve_arclengthvariation_taylor_test` +
   `:911` `test_curve_meansquaredcurvature_taylor_test` +
   `:723` `test_curve_torsion_taylor_test`** — finite-difference Taylor tests
   on `CurveCurveDistance`, `LpCurveTorsion`, `MeanSquaredCurvature`, and
   `ArclengthVariation`. The underlying penalties are JIT-JAX in
   `src/simsopt/geo/curveobjectives.py:54-543` (kernel sweep above). The JAX
   mirror `tests/geo/test_curveobjectives_item07_closeout.py:91-240` only
   covers `FramedCurveTwist.Lp` and `LinkingNumber` Taylor; the other Lp
   penalties have no FD-Taylor mirror.

3. **`tests/field/test_magneticfields.py:642` `test_Dommaschk` +
   `:958` `test_Reiman` + `:1028` `test_reiman_dBdX_taylortest` +
   `:642-708` (Dommaschk Taylor sweep) + `:921` `test_BifieldMultiply`** —
   the Dommaschk/Reiman JAX wrappers
   (`field/test_magneticfieldclasses_jax_item15.py:239-401`) cover B/dB vs
   CPU but skip the **second-derivative Taylor / curl-divergence-free
   checks** and skip `BifieldMultiply` JAX gradient parity. `MagneticFieldMultiply`
   composition is exercised in `test_magnetic_field_composition_jax.py:250-461`
   but only at first order; the Reiman dB/dX Taylor refinement is absent.

4. **`tests/geo/test_surface.py:480` `test_gauss_bonnet` +
   `:552` `test_is_self_intersecting` + `:596` `test_extend_via_normal`
   (plus `tests/geo/test_surface_rzfourier.py:398,437`
   `test_extend_via_normal[_non_stellsym]`)** — Gauss-Bonnet has a JAX scalar
   parity row at `test_surface_rzfourier_jax.py:733`
   `test_surface_rzfourier_jax_gauss_bonnet_matches_cpu_oracle` (RZ-only,
   stellsym-only). The XYZTensorFourier / XYZFourier / Henneberg paths and
   the `is_self_intersecting` / `extend_via_normal` API parity are
   unmirrored even though `surface_fourier_jax.py` and
   `surface_henneberg_jax.py` cover them in `src/simsopt/geo`.

5. **`tests/geo/test_boozersurface.py:534`
   `test_boozer_surface_optimisation_convergence` +
   `:732` `test_minimize_boozer_penalty_constraints_ls_manual` +
   `:864` `test_minimize_boozer_exact_constraints_newton_G_None` +
   `:907` `test_minimize_boozer_exact_constraints_newton_stellsym_false`** —
   convergence + manual LS-loop fixtures with explicit iteration-count and
   residual-decrement assertions against the CPU solver. The JAX mirror
   covers public solver entrypoints
   (`test_boozersurface_jax.py:2602-3072`) but does NOT assert that the JAX
   solver converges with the **same iteration history** the CPU one does on
   these fixtures, nor the manual LS-loop matched-step contract.

---

## Section: `tests/field/`

### `test_biotsavart.py` ↔ `test_biotsavart_jax.py` + `_jax_parity.py` + `_jax_cpu_ordered.py` + `_A_direct_kernel_closeout.py`

| Original test (file:line) | Status | JAX mirror file:line | Missing scenarios |
|---|---|---|---|
| `test_biotsavart_both_interfaces_give_same_result:25` | MIRRORED | `test_biotsavart_jax.py:490` `test_B_parity_ncsx` | — |
| `test_biotsavart_exponential_convergence:35` | MIRRORED | `test_biotsavart_jax_parity.py:232` `test_quadrature_convergence` | dB and d2B convergence Taylor depths capped at 4-level refinement; d2B taylor convergence absent |
| `test_dB_by_dcoilcoeff_reverse_taylortest:53` | MIRRORED | `test_biotsavart_jax_parity.py:415` `test_B_vjp_taylor_test` | — |
| `test_dBdX_by_dcoilcoeff_reverse_taylortest:80` | UNMIRRORED | (no JAX `B_and_dB_vjp` reverse Taylor) | dB/dX-VJP-vs-coil-coeff Taylor test missing; method exists at `biotsavart_jax_backend.py:1628` |
| `test_biotsavart_dBdX_taylortest:129` | MIRRORED | `test_biotsavart_jax_parity.py:317` `test_dB_dX_taylor_test` | — |
| `test_biotsavart_gradient_symmetric_and_divergence_free:145` | MIRRORED | `test_biotsavart_jax_parity.py:341` `test_dB_dX_symmetric_and_divergence_free` | — |
| `test_d2B_by_dXdX_is_symmetric:161` | MIRRORED | `test_biotsavart_jax_parity.py:364` `test_d2B_dXdX_symmetric` | — |
| `test_biotsavart_d2B_by_dXdX_taylortest:195` | MIRRORED | `test_biotsavart_jax_parity.py:395` `test_d2_dXdX_taylor_test` | — |
| `test_biotsavart_B_is_curlA:200` | MIRRORED | `test_biotsavart_jax_parity.py:269` `test_B_is_curl_A` | — |
| `test_biotsavart_dAdX_taylortest:236` | MIRRORED | `test_biotsavart_jax_parity.py:294` `test_dA_dX_finite_difference` | — |
| `test_biotsavart_d2A_by_dXdX_taylortest:271` | MIRRORED | `test_biotsavart_jax_parity.py:395` (shared parametrize) | — |
| `test_biotsavart_coil_current_taylortest:276` | UNMIRRORED | (no mirror) | Linearity of `dB`, `d2B`, `d3B` in coil current; methods `d2B_by_dXdcoilcurrents`, `d3B_by_dXdXdcoilcurrents` not exposed by JAX backend |
| `test_dA_by_dcoilcoeff_reverse_taylortest:305` | PARTIAL MIRROR | `test_biotsavart_jax_parity.py:415` (covers `B_vjp` only) | Specifically A-VJP-vs-coil Taylor missing |
| `test_dAdX_by_dcoilcoeff_reverse_taylortest:332` | UNMIRRORED | (no mirror) | A_and_dA_vjp reverse Taylor missing; method exists at `biotsavart_jax_backend.py:1620` |
| `test_flux_through_disk:360` | UNMIRRORED | (no mirror) | Cross-check `B = curl A` integrated form (Stokes). JAX has both `B` and `A`; method-level result already verified, but the dblquad-vs-line-integral closure check is missing |
| `test_biotsavart_vector_potential_coil_current_taylortest:402` | UNMIRRORED | (no mirror) | `dA_by_dcoilcurrents`, `d2A_by_dXdcoilcurrents`, `d3A_by_dXdXdcoilcurrents` |
| `test_biotsavart_vector_potential_current_getters_fill_cold_cache:435` | N/A — NOT PORTED | — | fieldcache layer (CPU-only); the JAX backend has no `fieldcache_*` getters |
| `test_biotsavart_fieldcache_*:472,494,507` | N/A — NOT PORTED | — | fieldcache compat layer |

### `test_coil.py` ↔ (no `_jax` mirror)

All 13 test functions cover `Coil`, `Current`, `ScaledCurrent`, `CurrentSum`,
`coils_via_symmetries`, `load_coils_from_makegrid_file`,
`coils_to_vtk`, `equally_spaced_planar_curves`.

- Status across the file: **N/A — NOT PORTED** for serialization, vtk
  writers, MAKEGRID/FOCUS readers (file IO).
- `test_scaled_current:163` (`:206 test_makegrid` is also file IO) —
  arithmetic and gradient on `ScaledCurrent` / `CurrentSum`; the JAX
  backend uses these untouched (see `field/coil.py:3,133`), so **N/A — NOT
  PORTED**.
- `test_equally_spaced_planar_curves:282` and
  `test_coils_via_symmetries_with_regularizations:314` — produce coils as
  fixtures for downstream tests, no JAX-side optimization claim. **N/A**.

### `test_coilset.py` ↔ (no `_jax` mirror)

32 tests of `CoilSet` and `ReducedCoilSet` mutability/penalty wiring.
`CoilSet` is not JAX-ported (`field/coilset.py` is plain numpy + Optimizable).
**N/A — NOT PORTED** for the whole file.

### `test_magneticfields.py` ↔ `test_magneticfieldclasses_jax_item15.py` + `test_interpolated_field_jax_item15.py` + `test_magnetic_field_composition_jax.py`

| Original test | Status | JAX mirror | Notes |
|---|---|---|---|
| `test_toroidal_field:37` | MIRRORED | `test_magneticfieldclasses_jax_item15.py:81` `TestToroidalFieldJAX.test_B_dB_d2B_A_dA_parity_vs_cpu` | — |
| `test_sum_Bfields:86` | MIRRORED | `test_magnetic_field_composition_jax.py:113-461` (broad parametrize) | — |
| `test_scalarpotential_Bfield:134` | MIRRORED | `test_scalar_potential_rz_jax_item23.py:47` `test_scalar_potential_rz_jax_matches_cpu_B_and_dB` | — |
| `test_circularcoil_Bfield:207` + `:561` toroidal arrangement | PARTIAL MIRROR | `test_circular_coil_jax.py:227` `TestWrapperParity.test_B_and_dB_match_cpu_oracle` | analytic centerline + toroidal-arrangement fixture not present in JAX mirror |
| `test_helicalcoil_Bfield:622` | UNMIRRORED | (no JAX HelicalCoil wrapper exists) | NOT PORTED |
| `test_Dommaschk:642` | PARTIAL MIRROR | `test_magneticfieldclasses_jax_item15.py:240` (B/dB only) | published paper d2B / d3B / dB linearity sweep not mirrored |
| `test_MirrorModel:708` | MIRRORED | `test_magneticfieldclasses_jax_item15.py:186,207` | — |
| `test_DipoleField_*:727,746,775,848` | MIRRORED | `test_dipole_field_jax_item26.py:83-156` | — |
| `test_BifieldMultiply:921` | PARTIAL MIRROR | `test_magnetic_field_composition_jax.py:251-291` (only TF × scalar) | non-stellsym, B-times-scalar derivatives, dB/dX-of-product missing |
| `test_Reiman:958` + `:1028 reiman_dBdX_taylortest` | PARTIAL MIRROR | `test_magneticfieldclasses_jax_item15.py:334-385` | dB/dX FD Taylor sweep missing |
| `test_cyl_versions:1033` | UNMIRRORED | (no JAX cyl B/dB-cyl/A-cyl mirror beyond circular coil) | `BiotSavart.B_cyl`, `dB_by_dX_cyl` |
| `test_interpolated_field_close_*:1068,1108,1147` | MIRRORED | `test_interpolated_field_jax_item15.py:177-255,326-365` | — |
| `test_get_set_points_cyl_cart:1176` | UNMIRRORED | — | `MagneticField.get_points_cyl/cart` round-trip across JAX subclasses |
| `test_to_vtk:1206` + `test_to_mgrid:1256` | N/A — NOT PORTED | — | file IO |
| `test_poloidal_field:1261` | MIRRORED | `test_magneticfieldclasses_jax_item15.py:145` | — |

### `test_boozermagneticfields.py` ↔ `test_boozer_analytic_jax.py` + `test_boozermagneticfield_jax_item33.py` + `test_interpolated_boozer_field_jax.py`

| Original test | Status | JAX mirror | Notes |
|---|---|---|---|
| `test_boozeranalytic:27` | MIRRORED | `test_boozer_analytic_jax.py:270,276,300,325,343` | — |
| `test_boozerradialinterpolant_finite_beta:120` + `:289 vacuum` | PARTIAL MIRROR | `test_boozermagneticfield_jax_item33.py:168,184,212` | full finite-beta dB/dtheta convergence Taylor & vacuum K-zero closed-form invariants are only spot-checked; vacuum convergence Taylor missing |
| `test_interpolatedboozerfield_sym:459` + `:597 no_sym` + `:744 convergence_rate` | PARTIAL MIRROR | `test_interpolated_boozer_field_jax.py:340-687` | convergence-rate Taylor against degree-3 spline is NOT mirrored |

### `test_fieldline.py`, `test_particle.py`, `test_mpi_tracing.py`

JAX path for tracing/particle is exercised in
`tests/field/test_tracing_jax_item16.py` and
`tests/jax_core/test_tracing_jax_*`. The Python originals deal with
SimsoptPP particle tracers and Poincare plot file dumps.

- `test_fieldline.py` all 5 tests: **N/A — NOT PORTED** (Poincare file IO,
  ncsx reference + plot file dumps).
- `test_particle.py` 13 tests: **PARTIAL MIRROR** — guiding-center/full-orbit
  conservation laws are in `tests/jax_core/test_tracing_jax_guiding_center.py`,
  `test_tracing_jax_fullorbit.py`, and event tests; but
  `test_compute_resonances:746`, `test_compute_poloidal_toroidal_transits:630`,
  `test_toroidal_flux_stopping_criterion:704`,
  `test_energy_momentum_conservation_boozer:421` are not paired with
  JAX-tracer assertions.
- `test_mpi_tracing.py` 2 tests: **N/A — NOT PORTED** (MPI).

### Other field/* files (mostly already covered above)

- `test_sampling.py:17,43` ↔ `test_sampling_jax_item22.py` — MIRRORED at
  weighted-index + curve/surface map + statistical moment levels.
- `test_normal_field.py` 30+ tests ↔ `test_normal_field_item17_closeout.py`
  (5 tests). **PARTIAL MIRROR**: closeout only covers
  `vns/vnc_match_direct_cpu_oracle`, real-space pair identity, recompute-bell,
  negative control, and hand-rolled real-space formula. The remaining 25+
  upstream tests cover dofs, change_resolution, fixed_range,
  get_index/make_names, asarray getter/setter, reduce_coilset,
  optimize_coils, double_reduction, vns_vnc_setter raises, serialization,
  spec_coil_correspondence — none of those API-shape behaviors are exercised
  via the JAX kernel. Most are CPU-only DOF wiring (N/A); the two genuine
  gaps are `test_optimize_coils:465` (full optimization run) and
  `test_spec_coil_correspondence_on_converged_output:364`.
- `test_selffieldforces.py` 38 tests ↔ `test_selffield_item02_closeout.py` (4
  tests) + `test_force_item09_closeout.py` (1 test) — **PARTIAL MIRROR**:
  `b_regularized_pure` + `Lp curve force` Taylor are mirrored; the bulk
  (`test_force_convergence`, `test_force_objectives`, `test_Taylor`,
  `test_LpCurveForces_Taylor_test`, `test_source_coils_coarse_and_fine`,
  shared-coil-state reuse tests, `test_net_force_and_torque`, etc.) are
  unmirrored — the JAX self-field surface in `src/simsopt/field/selffield.py`
  (used at 2807, 1096, 1214) is JAX-traced; missing tests would cover
  `RegularizedCircularCoil`, mixed-quadrature force objective Taylor, and
  shared-coil-state cache reuse claims.
- `test_wireframefield.py:21-292` (9 tests) ↔
  `test_wireframefield_jax_item30.py:55-147` (6 tests) — **PARTIAL MIRROR**.
  Convergence rate (`test_toroidal_wireframe_toroidal_field_convergence:48`),
  `test_toroidal_wireframe_curlB:105`, `_amperian_loops:158`, and the
  field-cache invalidation tests are not mirrored.
- `test_mgrid.py` (5 tests): **N/A — NOT PORTED** (file IO).
- `test_interpolant.py` (4 tests): `RegularGridInterpolant` is JAX-ported in
  `tests/jax_core/test_regular_grid_interp_item13.py`. **MIRRORED**.
- `test_magnetic_axis_helpers.py:9`: MIRRORED at
  `test_magnetic_axis_helpers_jax_item21.py:149`
  `test_jax_kernel_matches_cpu_oracle`.
- `test_magneticfields_optimization.py:17,55,101`: UNMIRRORED — circular-coil
  current/position/orientation optimization end-to-end; JAX `CircularCoilJAX`
  wrapper exists at `src/simsopt/field/circular_coil_jax.py` but no
  optimization-loop test mirrors it.
- `test_coilobjective.py`: MIRRORED (entire file is JAX parity, oracle is
  CPU `CurrentPenalty`).

---

## Section: `tests/geo/`

### `test_curve.py` (39 tests) ↔ no single canonical mirror

`tests/geo/test_curve.py` is huge and mixes curve geometry, surface-bound
wrappers, `CurveCWSFourier`, `CurvePlanarFourier`, vtk plots, and
`make_grid` setup. Curve geometry is JAX-native (`curve.py:6-228` uses
`jax.grad`/`jax.hessian`/`jax.jacfwd`/`jax.jvp`/`jax.vjp`). Mirroring picture:

- **MIRRORED**: derivative consistency tests like
  `test_curve_first_derivative:497`, `_second_derivative:517`,
  `_kappa_first_derivative:625`, `_kappa_derivative:669`,
  `_torsion_derivative:689`, `_frenet_frame:705`,
  `_frenet_frame_derivative:745`, `_incremental_arclength_derivative:647`,
  `_centroid:1181`, `_dkappa_by_dphi_derivative:765` — all share their
  underlying kernels with `_native_curve_geometry` in
  `src/simsopt/jax_core/curve_geometry.py` (used by `framedcurve_jax_*`
  tests). The closeout `test_curvexyzfouriersymmetries_spec_jax.py:180-402`
  provides byte-identical spec-vs-curve parity which subsumes geometry
  checks.
- **PARTIAL MIRROR**: `test_trefoil_nonstellsym:238` and `_stellsym:277`,
  `test_nonstellsym:317`, `test_xyzhelical_symmetries:341`,
  `test_curve_helical_xyzfourier:428` — all assert curve symmetry under
  stellarator transforms. JAX mirror `test_curve_item05_closeout.py:127,174`
  ports `CurveXYZFourierSymmetries` only.
- **UNMIRRORED, high-value**:
  - `test_create_planar_curves_between_two_toroidal_surfaces:907` and
    `test_create_equally_spaced_planar_curves_jax:1293` — JAX kernels do
    exist for planar-curve construction (`curveplanarfourier.py`) but no
    JAX-side init test asserts they reproduce CPU output.
  - `test_create_equally_spaced_curves_jax:1064` — name says "jax" but it
    runs CPU `CurveXYZFourier`. Tests CPU centerline.
  - `test_curvecwsfourier_matches_cpp_on_stage2_surface:1869`,
    `_vjps_include_surface_contributions:1901`,
    `_surface_bound_wrappers_refresh_after_*_mutation:1981,2080` — these
    already test JAX VJP plumbing (closeout coverage from
    `test_curve_item05_closeout.py` is narrower). Treat as PARTIAL MIRROR.
- **N/A — NOT PORTED**: `test_curve_to_vtk:772`, `test_plot:777`,
  `test_serialization:858`, `test_load_curves_from_makegrid_file:864`.

### `test_curve_objectives.py` (27 tests) ↔ `test_curveobjectives_item07_closeout.py` (4 tests) + `test_linking_number_jax.py` + `test_distance_jax.py` + `test_accessibility.py`

| Original test | Status | JAX mirror | Notes |
|---|---|---|---|
| `test_curve_length_taylor_test:634` | UNMIRRORED | — | `CurveLength` is JAX (`curveobjectives.py:54-59`); no FD-Taylor mirror |
| `test_curve_curvature_taylor_test:663` | UNMIRRORED | — | `LpCurveCurvature` JIT exists at `curveobjectives.py:190` |
| `test_curve_curvature_barrier_taylor_test:694` | UNMIRRORED | — | `LpCurveCurvatureBarrier` JIT at `curveobjectives.py:223` |
| `test_curve_torsion_taylor_test:723` | UNMIRRORED | — | `LpCurveTorsion` JIT at `curveobjectives.py:337-353` |
| `test_curve_minimum_distance_taylor_test:780` | UNMIRRORED | — | `CurveCurveDistance` JIT at `curveobjectives.py:432-469`; `test_pairwise_penalty_chunking_matches_dense_paths:194` and `_accepts_explicit_row_sharding:467` cover *chunking* but no FD-Taylor on the value-gradient |
| `test_curve_minimum_distance_barrier_taylor_test:826` | UNMIRRORED | — | `CurveCurveDistanceBarrier` JIT at `curveobjectives.py:500-533` |
| `test_curve_arclengthvariation_taylor_test:862` | UNMIRRORED | — | `ArclengthVariation` JIT in `curveobjectives.py` |
| `test_arclength_variation_circle:869` + `:878 _planar` | UNMIRRORED | — | closed-form circle parity for `ArclengthVariation` |
| `test_curve_meansquaredcurvature_taylor_test:911` | UNMIRRORED | — | `MeanSquaredCurvature` |
| `test_curve_curve_distance_empty_candidates:1148` | MIRRORED | `test_distance_jax.py:30-79` | — |
| `test_linking_number:1057` + `_planar:1101` | MIRRORED | `test_linking_number_jax.py:142-292` | — |
| `test_curve_surface_distance:1009` | MIRRORED | `test_distance_jax.py:116` + `test_curve_objectives.py:288-346` (chunked/dense) | — |
| `test_minimum_distance_candidates_*:918,943,971` | MIRRORED | `test_distance_jax.py:30-79` + `test_curve_objectives.py:194,467` | — |

### `test_surface.py` (21 tests, base `Surface` API) ↔ none

The base `Surface` class is partly JAX-instrumented through
`surface_fourier_jax`. **PARTIAL MIRROR / mostly N/A**:

- `test_theta:57`, `_phi:83`, `_spectral:149`, `_dof_names:635`,
  `_serialization:418`, `_axisymm:433`, `_independent_of_quadpoints:446`,
  `_arclength_poloidal_angle*:176-326`, `_interpolate_on_arclength_grid*:246-278`,
  `_make_theta_uniform_arclength:326`, `_distance:356`, `_surface_scaled:375`,
  `_names:393`, `_fixed:403`, `_cross_section:505`,
  `_is_self_intersecting*:552,579`, `_extend_via_normal:596` — N/A
  (CPU-only DOF/serialization API + numpy-only utilities; no JAX
  reproduction in source).
- `test_gauss_bonnet:480` — PARTIAL MIRROR via
  `test_surface_rzfourier_jax.py:733` (only RZFourier-stellsym).

### `test_surface_rzfourier.py` (48 tests) ↔ `test_surface_rzfourier_jax.py` (41 tests) + `test_surface_rzfourier_jax_item06_closeout.py`

| Original test | Status | JAX mirror | Notes |
|---|---|---|---|
| `test_aspect_ratio:29` | MIRRORED | `test_surface_objectives_jax.py:5342-5413` | — |
| `test_init/set_dofs/get_dofs/repr/get_rc/get_zs/set_rc/set_zs/names_order/mn/mn_matches_names/copy_method/fixed_range/flip_z/flip_phi/flip_theta/rotate_half_field_period/shift_theta_by_half/condense_spectrum/_circle/_theta_origin/serialization/shared_dof_*/make_rotating_ellipse*` | N/A — NOT PORTED | — | DOF-shape API, file IO, condense-spectrum geometric algebra; not in `surface_rzfourier_jax.py` |
| `test_area_volume:603` | MIRRORED | `test_surface_rzfourier_jax.py:1244` + `test_surface_fourier_jax.py:253` | — |
| `test_vjps:617` | MIRRORED | `test_surface_fourier_jax.py:374-419` + `test_surface_rzfourier_jax.py:861` `_geometry_jacfwd_matches_scalar_composition` | — |
| `test_area_derivative:964` + `test_volume_derivative:1006` | MIRRORED | `test_surface_fourier_jax.py:961-1069` (Taylor) + `test_surface_rzfourier_jax.py:1244` | — |
| `test_fourier_transform_scalar:845` | UNMIRRORED | — | Surface-Fourier-transform scalar utility not in JAX module |
| `test_from_wout/_from_vmec_input/_from_nescoil_input/_from_focus/_from_pyQSC/_get_and_write_nml/_change_resolution/_complete_grid/_convert_back/_from_RZFourier` | PARTIAL MIRROR | `test_surface_rzfourier_jax.py:1323-1453` (object-API parity for `from_wout/from_vmec_input/from_nescoil_input/from_focus/from_pyqsc/copy/change_resolution/make_rotating_ellipse/extend_via_normal/condense_spectrum`) | — |

### `test_surface_xyzfourier.py` (9 tests) ↔ partial mirror in `test_surface_fourier_jax.py:460-609`

| Original | Status | Mirror | Notes |
|---|---|---|---|
| `test_toRZFourier_perfect_torus:20` + `_lossless_at_quadrature_points:61` + `_small_loss_elsewhere:117` | UNMIRRORED | — | `to_RZFourier` host wrapper not JAX-mirrored |
| `test_cross_section_torus:152` | UNMIRRORED | — | — |
| `test_aspect_ratio_random_torus:237` + `_compare_with_cross_sectional_computation:269` | PARTIAL MIRROR | `test_surface_objectives_jax.py:5342-5413` (only `AspectRatioJAX` value/grad; not the "agree with cross-section" geometry assertion) | — |
| `test_to_vtk:303`, `test_serialization:314`, `test_shared_dof_init:351` | N/A — NOT PORTED | — | — |

### `test_surfacehenneberg.py` (10 tests) ↔ `test_surface_henneberg_jax.py` (24 tests)

- `test_repr/_names/_set_get_dofs/_set_get_rhomn/_fixed_range/_indexing` —
  N/A (CPU-only DOF API).
- `test_axisymm:166` + `test_from_RZFourier:197` + `test_vmec:252` — PARTIAL
  MIRROR; `test_surface_henneberg_jax.py:288-322` covers axisymmetric
  closed-form; `from_RZFourier`/VMEC round-trip is unmirrored.
- `test_serialization:277` — N/A (file IO).

### `test_surface_garabedian.py` (4 tests) ↔ `test_surface_garabedian_jax.py` (4 tests)

- `test_init/_shared_dof_init` — N/A.
- `test_convert_back:64` MIRRORED via
  `test_surface_garabedian_jax.py:109` `test_jax_conversion_matches_cpu_to_rzfourier`.
- `test_fix_range:81` — N/A.

### `test_boozersurface.py` (23 tests) ↔ `test_boozersurface_jax.py` (237 tests) + `test_boozersurface_jax_private.py` (78 tests) + `test_boozer_residual_jax.py` + `test_boozer_derivatives_jax.py` + `test_boozer_legacy_parity_contract.py`

The JAX mirror suite is over-built relative to the CPU original. Mapping:

| Original test | Status | JAX mirror | Notes |
|---|---|---|---|
| `test_call_boozer_residual_falls_back_to_alpha_only_signature:75` + `_ds:112` + `_ds2:162` | MIRRORED | `test_boozersurface_jax.py:75-218` (parallel scaffolding via `_simsoptpp_boozer_compat`) | — |
| `test_solver_signatures_do_not_expose_vectorize:219` | MIRRORED | `test_boozersurface_jax.py:2584` | — |
| `test_constructor_rejects_spoof_surface_names:230` | MIRRORED | `test_boozersurface_jax.py:2329` | — |
| `test_run_code_rejects_G_none_with_free_currents:253,264` | MIRRORED | `test_boozersurface_jax.py:2410,2416` | — |
| `test_residual:300` + `_gradient:328` + `_hessian:345` | MIRRORED | `test_boozer_residual_jax.py:*` and `test_boozer_derivatives_jax.py:*` | — |
| `test_boozer_constrained_jacobian:464` | MIRRORED | `test_boozer_derivatives_jax.py:*` (composed Jacobian) | — |
| `test_boozer_surface_optimisation_convergence:534` | UNMIRRORED | — | Full LS+Newton optimization-convergence assertion against CPU iteration count and residual decrement |
| `test_boozer_serialization:676` | N/A — NOT PORTED | — | — |
| `test_run_code:703` | PARTIAL MIRROR | `test_boozersurface_jax.py:2733-3014` | covers public solver routing, but not the run-code convergence-history assertion at 703 |
| `test_minimize_boozer_penalty_constraints_ls_manual:732` | UNMIRRORED | — | Manual-loop LS contract against legacy CPU manual demo |
| `test_need_to_run_code_false:806` | UNMIRRORED | — | `need_to_run_code` short-circuit logic |
| `test_minimize_boozer_exact_constraints_newton_G_None:864` + `_stellsym_false:907` | PARTIAL MIRROR | `test_boozersurface_jax.py:2899-3014` (covers exact-constraints Newton API contract + nonstellsym Jacobian shape) | actual convergence-to-residual-floor against CPU is not asserted on these fixtures |
| `test_boozer_surface_quadpoints:969` + `_type_assert:1018` | MIRRORED | `test_boozersurface_jax.py:2369-2383` | — |

### `test_surfaceobjectives.py` (13 tests) ↔ `test_surface_objectives_jax.py` (~150 tests)

All major derivative tests are MIRRORED. The `test_qfm_surface_derivative:220`
and `test_principal_curvature_first_derivative:170` are matched at
`test_surface_objectives_jax.py:5543-5713` (Principal/QFM). The `test_iotas_derivative:281`,
`test_nonQSratio_derivative:322`, `test_boozerresidual_derivative:363`,
`test_major_radius_derivative:247`, `test_toroidal_flux_*:62-103,103`,
`test_parameter_derivatives_volume:197`, `test_label_surface_derivative*:391,417`
are all covered in `test_surface_objectives_jax.py:2397-6068`. **MIRRORED**.

### `test_surface_taylor.py` (19 tests) ↔ `test_surface_fourier_jax.py` (29 tests) + `test_surface_rzfourier_jax.py` (41 tests) + `test_surface_objectives_jax.py`

- `test_surface_coefficient_derivative:216` MIRRORED at
  `test_surface_fourier_jax.py:374-419`.
- `test_surface_normal_coefficient_derivative:241` MIRRORED at
  `test_surface_fourier_jax.py:419-460` + `:609-673`.
- `test_fund_form_coefficient_derivative:282` MIRRORED at
  `test_surface_rzfourier_jax.py:666-732`.
- `test_unit_normal_coefficient_derivative:305` MIRRORED.
- `test_surface_area_coefficient_*:329,339` MIRRORED at
  `test_surface_fourier_jax.py:961-1176`.
- `test_volume_coefficient_*:367,377,442` MIRRORED.
- `test_minor_radius_second_derivative:452`,
  `test_major_radius_second_derivative:483`,
  `test_mean_area_second_derivative:514`, `test_AR_second_derivative:545`
  PARTIAL MIRROR via `test_surface_objectives_jax.py:5376` (aspect-ratio
  Hessian) but minor/mean-area Hessian columns are not asserted independently.
- `test_surface_phi_derivative:593`, `_theta_derivative:616`,
  `_theta2_derivative:639`, `_phi2_derivative:662`,
  `_thetaphi_derivative:686` — UNMIRRORED. These exercise
  `dgamma_by_dphi/dtheta` chain rule via FD. JAX does have
  `gammadash1`/`gammadash2`/`gammadash1dash1` (the source has them in
  `surface_fourier_jax.py`) but no FD-Taylor sweep test mirrors them.
- `test_surface_conversion:711` — N/A (host wrapper).

### `test_qfm.py` (6 tests) ↔ `test_surface_objectives_jax.py:5657-5713` (5 tests)

- `test_residual:17` MIRRORED at
  `test_surface_objectives_jax.py:5660` `_value_parity_matrix`.
- `test_qfm_objective_gradient:43` MIRRORED at
  `test_surface_objectives_jax.py:5672` `_gradient_parity_matrix`.
- `test_qfm_label_constraint_gradient:87` UNMIRRORED. No JAX label-constraint
  test for QFM.
- `test_qfm_penalty_constraints_gradient:128` UNMIRRORED.
- `test_qfm_surface_optimization_convergence:175` UNMIRRORED.
- `test_minimize_qfm:285` UNMIRRORED. `QfmSurface.minimize_qfm_penalty_constraints_LBFGS`
  has no JAX-side check.

### `test_finitebuild.py` (4 tests) ↔ `test_finitebuild_jax_item20.py` (4 tests) + `test_finitebuild_jax_ssot_item20.py` (3 tests)

- `test_multifilament_gammadash:17` MIRRORED at
  `test_finitebuild_jax_ssot_item20.py:74,142` + `test_finitebuild_jax_item20.py:89`.
- `test_multifilament_coefficient_derivative:57` MIRRORED at
  `test_finitebuild_jax_item20.py:151,182,213` (gamma/gammadash/spec VJPs).
- `test_filamentpack:120` UNMIRRORED. `FilamentRotation.coil_pack(...)`
  construction is JAX-ported in `framedcurve_jax.py` but no JAX test asserts
  matched packing structure.
- `test_biotsavart_with_symmetries:159` UNMIRRORED. Cross-check of finite-build
  filament + BiotSavart + stellarator-symmetric coil reproduction.

### `test_strainopt.py` (3 tests) ↔ `test_strainopt_item08_closeout.py` (4 tests)

- `test_strain_opt:106` UNMIRRORED. Full strain-optimization run against
  reference snapshot.
- `test_torsion:137` MIRRORED at `test_strainopt_item08_closeout.py:171,206`
  (zero-twist circle closed form).
- `test_binormal_curvature:143` MIRRORED at
  `test_strainopt_item08_closeout.py:171`.

### `test_curveperturbed.py` (5 tests) ↔ no `_jax` mirror

`CurvePerturbed` is JAX-aware (`curveperturbed.py` does call into JAX kernels
indirectly through curve.gamma). All 5 tests are UNMIRRORED. The
high-leverage ones are `test_perturbed_gammadash:15` (CPU formula),
`test_perturbed_objective_torsion:51`, and `_distance:90`.

### `test_curve_optimizable.py` (1 test) + `test_curve_helical.py` (1 test)

- `test_curve_length_optimization:82` UNMIRRORED but low-priority — it's
  a smoke optimization test.
- `test_dof_names:7` N/A (DOF naming).

### `test_pm_grid.py` (9 tests) ↔ `test_permanent_magnet_grid_jax_item27.py` (4 tests)

PARTIAL MIRROR. Coverage of `from_fixed_state`, `pol_vectors`, and
`alpha_rule` is present. UNMIRRORED in JAX: `test_bad_params:27`,
`test_grid_chopping:265`, `test_famus_functionality:338` (file IO),
`test_polarizations:416`, `test_pm_helpers:500`,
`test_pm_post_processing:628`. The genuine geometry gap is
`test_polarizations:416` (orientation grid sweep) and
`test_pm_helpers:500` (post-processing helper parity).

### Geo file-level UNMIRRORED whole-files (no `_jax` sibling)

- `test_curve_helical.py`, `test_curve_optimizable.py` — DOF naming + smoke;
  N/A.
- `test_curveperturbed.py` — gaps listed above.
- `test_wireframe_toroidal.py` (8 tests) — `ToroidalWireframe` construction,
  constraint matrices, collision-checking. NOT JAX-ported (the JAX side is
  `tests/solve/test_wireframe_optimization_jax_item31.py` for solver only).
  PARTIAL MIRROR for solver path; constructor/collision tests are
  CPU-internal N/A.
- `test_ports.py` (6 tests) — `Port` geometry + accessibility. The
  accessibility tests in `tests/geo/test_accessibility.py` cover the JAX
  side; geometry/IO of `Port` is N/A.
- `test_surface_quadrature_grid_rejection.py` (5 tests) — JAX-only feature,
  no Python original.
- `test_plot.py:20` — N/A (plotting).
- `test_simsoptpp_compat.py:26,40,65` — JAX shim compat; no Python original.

---

## Section: `tests/objectives/`

### `test_fluxobjective.py` (6 tests) ↔ `test_fluxobjective_jax_parity.py` (13 tests) + `test_fluxobjective_jax_item03_closeout.py` (1 test)

| Original | Status | Mirror | Notes |
|---|---|---|---|
| `test_definitions:67` | MIRRORED | `test_fluxobjective_jax_parity.py:304` | — |
| `test_derivatives:143` | MIRRORED | `:311,317` | — |
| `test_quadratic_flux_gradient_handles_zero_normals:173` | MIRRORED | `:370` `test_quadratic_flux_zero_normals_contract` | — |
| `test_singular_local_returns_inf_and_raises_gradient_failure:183` | MIRRORED | `:404` | — |
| `test_singular_normalized_returns_inf_and_raises_gradient_failure:195` | MIRRORED | `:404` | — |
| `test_squaredfluxjax_requires_surface_spec:207` | MIRRORED | `:458` `_requires_native_field_contract` | — |

### `test_least_squares.py` (5 tests) ↔ no JAX mirror

- All 5 tests cover the CPU `LeastSquaresProblem` class for residual
  arithmetic and parent DOF wiring. **N/A — NOT PORTED** (no
  `least_squares_jax.py`).

### `test_constrained.py` (5 tests) ↔ no JAX mirror

Constrained-objective host wrapper. **N/A — NOT PORTED**.

### `test_utilities.py` (4 tests) ↔ no direct JAX mirror

- `test_quadratic_penalty:62` MIRRORED partially at
  `test_utilities.py:71` `test_quadratic_penalty_hostifies_jax_scalar_objective`.
- `test_mpi_objective:103` + `_mpi_optimizable:125` — N/A (MPI).

---

## Section: `tests/integration/`

All files in `tests/integration/` are JAX integration. The few that have an
implicit "CPU original" are the Stage 2 / single-stage scripts in
`benchmarks/`; those are out of scope for this audit (not under `tests/`).

The integration tests have no Python-only mirror to compare against. They
all hold an independent oracle role (CPU C++ reference via simsoptpp). Use
`tests/integration/test_jax_native_path.py:*`, `test_stage2_jax.py:*`,
`test_single_stage_jax.py:*` as JAX-only.

---

## Section: `tests/core/`, `tests/util/`, `tests/solve/`, `tests/mhd/`

### `tests/core/`

- `test_derivative.py` 13 tests — `Derivative` arithmetic is generic.
  `test_jax_blocks_materialize_to_numpy:120` and
  `test_mixed_numpy_and_jax_blocks_are_hostified_in_arithmetic:130` are
  JAX-specific. The rest (Taylor graph, scaled, sum operators) are MIRRORED
  trivially (they pass through JAX-aware arithmetic).
- `test_dofs.py`, `test_optimizable.py`, `test_descriptor.py`,
  `test_dev.py`, `test_finite_difference.py`, `test_integrated.py`,
  `test_json.py`, `test_util.py`, `test_load_specs.py` — these test the
  Optimizable framework. **N/A — NOT PORTED**. JAX has its own pytree-based
  alternative in `simsopt.jax_core.specs` covered by
  `tests/core/test_jax_core_specs.py`.
- `test_reductions.py` — JAX-only feature (compensated/pairwise reductions).
- `test_finite_difference.py` `test_jac_mpi`, `_bcast_fixed_dofs` — N/A.

### `tests/util/`

- `test_fourier_interpolation.py` 3 tests — utility, likely **N/A — NOT
  PORTED**.
- `test_coil_optimization_helper_functions.py` 18 tests — full
  optimization-helper smoke. The JAX path is exercised indirectly through
  `test_run_code_benchmark_common.py:*`. Pure helper validation tests are
  N/A.
- `test_mpi_partition.py` — N/A (MPI).

### `tests/solve/`

- `test_least_squares.py` 4 tests, `test_constrained.py` 3 tests,
  `test_mpi.py` 3 tests, `test_pm_optimization.py` 3 tests,
  `test_wf_optimization.py` 4 tests — these exercise the SciPy-driven
  Optimizable solver. JAX has its own L-BFGS adapter
  (`optimizer_jax.py`) covered at
  `tests/geo/test_boozersurface_jax.py:981-1041`. **PARTIAL MIRROR**: the
  least-squares Rosenbrock and quadratic-bounds fixtures at
  `test_least_squares.py:28,76,96` are not paired against the JAX BFGS
  adapter on the same fixtures.
- `test_permanent_magnet_optimization_jax_item28.py` (15 tests),
  `test_wireframe_optimization_jax_item31.py` (10 tests) — JAX-side
  closeout suites covering the JAX PM-MwPGP, GPMO, and wireframe solvers.

### `tests/mhd/`

All MHD tests (`test_boozer`, `test_spec`, `test_virtual_casing`,
`test_vmec_diagnostics`, `test_bootstrap`, `test_profiles`, `test_vmec*`)
are **N/A — NOT PORTED**. The JAX port deliberately excludes VMEC/SPEC
runners. The exception is
`tests/mhd/test_boozer.py:84 test_quasisymmetry_residuals` which has an
indirect mirror through `IotasJAX`/`NonQuasiSymmetricRatioJAX`
(`test_surface_objectives_jax.py:2397-2729`).

---

## UNMIRRORED HIGH-VALUE LIST (sketches)

These are top targets ordered by leverage (touch many code paths, gate
production correctness, fixture+oracle are already on hand):

### 1. `BiotSavartJAX` coil-current Taylor + 3rd-mixed-derivatives
   - **Fixture**: re-use `_make_fourier_coil(200)` + `_BASE_POINTS` from
     `tests/field/test_biotsavart_jax_parity.py:1-200`.
   - **Oracle**: closed-form linearity in current (B is linear in I), C++
     `simsoptpp::biot_savart_B` / `biot_savart_A` at I and at I=0.
   - **Assertions**: `np.linalg.norm(dB[0] - (B - B0)/c0) < 1e-15`
     (matching upstream tolerance). Same for `dJ` (d2B_by_dXdcoilcurrents)
     and `dH` (d3B_by_dXdXdcoilcurrents) and the A analogs.
   - **Surface**: `biotsavart_jax_backend.py:1616-1628` already supplies
     `A_vjp`/`A_and_dA_vjp`/`B_and_dB_vjp`; missing kernels are
     `d{1,2,3}{B,A}_by_dXdXdcoilcurrents` — these can be derived via
     `jax.grad`/`jax.jacfwd` from the existing primitives.

### 2. `CurveCurveDistance` + `LpCurveTorsion` + `MeanSquaredCurvature` + `ArclengthVariation` FD Taylor
   - **Fixture**: `tests/geo/test_curve_objectives.py:780-911` re-uses
     simple `CurveXYZFourier(10, 3)` ladders.
   - **Oracle**: FD Taylor of an analytic objective (REVIEWER_ORACLE_LINT.md
     type 4).
   - **Assertion**: ratio rule `err_new < 0.55 * err` over 5 doublings of
     1/2^i, mirroring upstream pattern.
   - **Surface**: `curveobjectives.py:54-543` JAX-JIT.

### 3. `BoozerSurfaceJAX` convergence + manual LS-loop fixtures
   - **Fixture**: replicate
     `tests/geo/test_boozersurface.py:534,732,864,907` with the BoozerSurfaceJAX
     constructor.
   - **Oracle**: independent C++ run (`simsoptpp.boozer_surface_residual` plus
     SciPy LS, already invoked in the CPU test) — record solver history
     (`iter`, `||res||`) and assert JAX equals within
     `direct-kernel` parity-ladder tolerance.
   - **Surface**:
     `tests/geo/boozersurface_jax_test_helpers.py` already wraps the
     fixtures.

### 4. `Reiman` and `Dommaschk` JAX dB/dX FD Taylor
   - **Fixture**: `tests/field/test_magneticfields.py:1028 reiman_dBdX_taylortest`
     and `:660 Dommaschk Taylor sweep`.
   - **Oracle**: FD on B with halving epsilon.
   - **Surface**: `reiman_jax.py`, `dommaschk_jax.py`.

### 5. `QfmObjective` + `QfmSurface.minimize_qfm_penalty_constraints_LBFGS`
   - **Fixture**: `tests/geo/test_qfm.py:43,128,285`.
   - **Oracle**: CPU `QfmSurface` LBFGS converged solution.
   - **Surface**: `surfaceobjectives_jax.py:691 QfmResidualJAX`. Need to add
     `QfmObjectiveJAX` wrapper if not present and an FD-grad gate.

### 6. Surface FD `dgamma_by_dphi/dtheta`, `d2gamma_by_dthetaphi` Taylor
   - **Fixture**:
     `tests/geo/test_surface_taylor.py:593-686`.
   - **Oracle**: FD on `surface.gamma`.
   - **Surface**: `surface_fourier_jax.py` has analytic derivatives.

### 7. `BiotSavart.B_cyl`/`dB_by_dX_cyl`/`A_cyl` parity
   - **Fixture**: `tests/field/test_magneticfields.py:1033 test_cyl_versions`.
   - **Oracle**: CPU `bs.B_cyl()` round-trip.
   - **Surface**: `biotsavart_jax_backend.py:*` (cylindrical-coord wrappers).

### 8. Finitebuild `FilamentPack` + `BiotSavart` with stellarator symmetries
   - **Fixture**:
     `tests/geo/test_finitebuild.py:120,159`.
   - **Surface**: `framedcurve_jax.py` + `coil.py`. Add JAX mirror that
     asserts the symmetry-multiplied filament BiotSavart B match the CPU
     value within `direct-kernel` rtol.

### 9. CircularCoil current/position/orientation optimization end-to-end
   - **Fixture**: `tests/field/test_magneticfields_optimization.py:17,55,101`.
   - **Surface**: `circular_coil_jax.py`. Add JAX-side scipy.minimize on
     `BiotSavartJAX([circular_coil_jax])` against the CPU optimum.

### 10. `BoozerRadialInterpolant` vacuum convergence-rate Taylor + `InterpolatedBoozerField` convergence-rate test
   - **Fixture**: `tests/field/test_boozermagneticfields.py:289,744`.
   - **Surface**:
     `interpolated_boozer_field` (jax_core) +
     `boozermagneticfield_jax.py`.

---

## Tautological-mirror flags (oracle-lint violations to revisit)

Per `tests/REVIEWER_ORACLE_LINT.md:1-86`, mirrors whose oracle routes
through the same JAX kernel are flagged below. UNCLEAR cases are marked.

- `tests/geo/test_boozersurface_jax.py:5163-5191` `_make_qfm_pair` builds
  `QfmResidual` (CPU) and `QfmResidualJAX` (JAX) and asserts `qfm_jax.J() ==
  qfm_cpu.J()`. The CPU side calls
  `simsoptpp::qfm_metric` while the JAX side calls
  `surfaceobjectives_jax.QfmResidualJAX.J`. These are independent
  implementations — acceptable oracle (C++ type 1). **NOT a tautology.**
- `tests/geo/test_surface_objectives_jax.py:5294-5563`
  `test_toroidal_flux_*` and `test_aspect_ratio_*` parity tests use
  `surfaceobjectives.AspectRatio.dJ_by_dsurfacecoefficients()` as the oracle.
  CPU is `simsoptpp.surface_*` while JAX is autodiff through
  `surface_fourier_jax`. **Acceptable (type 1)**.
- `tests/field/test_magnetic_field_composition_jax.py:113-461` —
  `compose_B_sum` matches "individual kernel summation". The "individual
  kernel" is the per-child `B()` evaluated via JAX → `MagneticFieldJAX.B`.
  The reference summation is computed in NumPy. **Acceptable (type 2/4):
  closed-form linearity**, but the test name is suggestive of tautology.
  UNCLEAR — flag for reviewer.
- `tests/field/test_biotsavart_jax.py:601-779` (chunked-vs-dense JAX) — both
  paths share the same JAX kernel; the chunking layer is a wrapper. This
  is a self-consistency check (acceptable for routing/Tier-4 gate, NOT a
  parity claim). Mirrors flagged correctly as routing-tier per docstrings.
- `tests/geo/test_surface_fourier_jax.py:374-419`
  `test_coefficient_derivatives_match_cpp` is **acceptable (type 1)** — uses
  `simsoptpp.SurfaceXYZTensorFourier::dgamma_by_dcoeff_vjp`.
- `tests/geo/test_curve_objectives.py:143-194,288-396,467-535` —
  chunked-vs-dense JAX-only routing. **NOT a parity oracle**; treat as
  self-consistency lane.
- `tests/integration/test_jax_native_path.py:*` (8 tests) — verify "native"
  lane equals "non-native" JAX lane. These are the same JAX kernel called
  through different entrypoints. **Routing/Tier-4 — not parity.** Docstring
  cites this.
- `tests/objectives/test_integral_bdotn_jax.py:121-244`
  `test_parity_with_target` uses
  `simsoptpp::integral_BdotN` as oracle. **Acceptable (type 1).**
- `tests/field/test_circular_coil_jax.py:267,287`
  `test_rotation_matrix_matches_cpu_rotmat` and
  `_cartesian_normal_matches_spherical_derivation` — CPU is
  `simsoptpp.rotmat` and the alternative is the same JAX kernel under
  another normal parameterization. The second is a JAX-vs-JAX consistency
  check. **Flag**: `_cartesian_normal_matches_spherical_derivation` is
  JAX-vs-JAX consistency; document tier in docstring or replace with C++.
- `tests/field/test_biotsavart_jax_parity.py:794`
  `test_B_linearity_in_current` calls
  `_assert_current_linearity(quantity_fn, ...)` which fully re-computes B at
  three currents. **Acceptable (type 2, closed-form linearity)**.

---

## Counts (final tally)

- **MIRRORED (full)**: ≈ 220 Python tests.
- **PARTIAL MIRROR**: ≈ 95 Python tests.
- **UNMIRRORED (real gap)**: ≈ 42 Python tests (across 8 modules; sketches
  above).
- **N/A — NOT PORTED**: ≈ 167 Python tests (file IO, MPI, VMEC, SPEC,
  bootstrap, virtual casing, particle tracing reference, plotting, mgrid,
  DOF API).
- **JAX-only files (no Python original counterpart)**: 18 files
  (e.g., `test_optimizer_jax_*`, `test_label_constraints_jax.py`,
  `test_surface_quadrature_grid_rejection.py`,
  `test_boozer_residual_pinned_input_byte_parity.py`,
  `test_simsoptpp_compat.py`, the entire `tests/jax_core/` directory,
  `test_runpod_single_stage_continuation.py`,
  `test_hf_production_gpu_proof.py`,
  `test_lightning_production_gpu_proof.py`,
  `test_state_artifact_merge_logic.py`,
  `test_section6_public_lane_split.py`,
  `test_stage2_target_lane_purity.py`,
  `test_factor_once_adjoint_phase2.py`).

UNCLEAR entries flagged in the body should be revisited by a reviewer with
the parity-ladder spec from
`benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES`.
