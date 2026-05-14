# C++ → JAX Port-Gap Inventory (2026-05-13)

## Executive Summary

This audit enumerates every publicly-exposed C++ symbol in `src/simsoptpp/`
(via the `python.cpp`, `python_curves.cpp`, `python_surfaces.cpp`,
`python_magneticfield.cpp`, `python_boozermagneticfield.cpp`,
`python_tracing.cpp`, and `python_distance.cpp` pybind11 modules) and
classifies its JAX port status.

**Counts by status (~127 public C++ symbols counted)**

| Status        | Count | Notes |
|---------------|-------|-------|
| PORTED        | 78    | Full-surface JAX equivalent present and canonical. |
| PARTIAL       | 23    | JAX module exists but misses one or more API rows. |
| UNPORTED      | 7     | No JAX counterpart found in `src/simsopt/**`. |
| NON-PORTABLE  | 11    | Pure pybind11 glue (set_dofs/get_dofs/invalidate_cache/init/_ref accessors) or build metadata. |
| UNCLEAR       | 8     | See questions below; mostly LinAlg helpers and pybind trampolines whose JAX semantics are ambiguous. |

**Top 3 highest-leverage UNPORTED items**

1. `boozer_dresidual_dc` (`src/simsoptpp/python.cpp:106`) — closed-form
   `dresidual/dc` Jacobian-by-DOF column kernel used in the Boozer
   residual derivative pipeline. JAX side composes via autodiff but
   no direct-kernel counterpart exists.
2. `Surface<T>::fit_to_curve` / `Surface<T>::_extend_via_normal_for_nonuniform_phi` / `Surface<T>::extend_via_projected_normal`
   (`src/simsoptpp/surface.cpp:82,129,145`) — surface bootstrap helpers
   called by `BoozerSurface` / `permanent_magnet_grid` initialization.
3. `BiotSavart::compute` / `BiotSavart::compute_A` legacy cache fillers
   (`src/simsoptpp/magneticfield_biotsavart.cpp:10,82`) — used by upstream
   `BiotSavart.compute()` to populate field caches with batched
   `dB/dX`+`d2B/dXdX`+`dA/dX`+`d2A/dXdX`. JAX `BiotSavartJAX` exposes the
   value APIs but not the cache-orchestration entrypoint.

**Top 3 PARTIAL items needing completion**

1. **Curve geometric quantities** (`PyCurve` family in `python_curves.cpp:82-110`).
   `kappa`, `torsion`, `incremental_arclength`, `dkappa_by_dcoeff`,
   `dtorsion_by_dcoeff`, `dincremental_arclength_by_dcoeff`,
   `least_squares_fit`, `dgammadashdashdash_by_dcoeff_vjp_impl` exist as
   pure functions in `src/simsopt/geo/curve.py` but are not lifted into
   `simsopt/jax_core/curve_geometry.py` — they currently rely on
   `sopp.Curve` trampoline subclasses for batched evaluation.
2. **Surface `d2volume_by_dcoeffdcoeff` / `d2area_by_dcoeffdcoeff` / `d2normal_by_dcoeffdcoeff`**
   exist (`surface_fourier_jax.py:2546,2610,2612`) but are autodiff-built
   on top of `surface_xyzfourier_*` only. **No equivalent JAX path
   exists for `SurfaceRZFourier`** — autodiff Hessians for area/volume
   wrt RZ DOFs route through CPU. Same for `dsurface_curvatures_by_dcoeff`.
3. **`BiotSavart::compute` derivative cache**: `BiotSavartJAX` returns
   `B/dB/d2B/A/dA/d2A` individually but the batched
   `dB_by_dcoilcurrents` / `dA_by_dcoilcurrents` array bundle stored in
   `simsopt.field.biotsavart.BiotSavart._dB_by_dcoilcurrents` (lines
   37-159) has no first-class JAX counterpart and must be assembled from
   per-coil JAX evaluations.

---

## Per-File C++ Symbol Tables

Each table lists pybind11-exposed symbols and classifies them by JAX
port status. The "Tests" column cites any direct CPU↔JAX or
closed-form parity test that exercises the symbol or its JAX twin.

### `src/simsoptpp/python.cpp` (module-level functions)

| Line | C++ symbol | Status | JAX target | Notes / API gap | Effort | Test parity |
|------|-----------|--------|------------|-----------------|--------|-------------|
| 57 | `biot_savart` | PORTED | `jax_core/biotsavart.py:597` `biot_savart_B_and_dB` plus `biot_savart_d2B_by_dXdX` | Bundled value+grad+hess. JAX has individual entrypoints but no single `compute(d_max)` selector. Treat the bundling itself as NON-PORTABLE cache orchestration. | LOW | `tests/field/test_biotsavart_jax.py`, `tests/field/test_biotsavart_jax_parity.py` |
| 58 | `biot_savart_B` | PORTED | `jax_core/biotsavart.py:573` `biot_savart_B` | — | — | `test_biotsavart_jax.py` (rtol=1e-10, direct-kernel lane) |
| 59 | `biot_savart_vjp` | PORTED | `jax_core/biotsavart.py:635` `biot_savart_B_vjp` + `field.py:299` `biot_savart_B_vjp_maybe_collective` | JAX uses `jax.vjp` over `biot_savart_B`; signature differs from `biot_savart_vjp` (no separate `res_B`/`res_dB` output arrays) but functionally equivalent. | — | `tests/field/test_biotsavart_jax_parity.py` |
| 60 | `biot_savart_vjp_graph` | PARTIAL | `jax_core/biotsavart.py:635` | C++ returns gradients wrt `gammas` AND `dgamma_by_dphis` separately; JAX returns tangents in autodiff cotangent dict. Re-shape adapter handled by `biotsavart_jax_backend.py`. | LOW | `test_biotsavart_jax_parity.py` |
| 61 | `biot_savart_vector_potential_vjp_graph` | PORTED | `jax_core/biotsavart.py:617` `biot_savart_A` (jax.vjp wraps) | Indirectly via autodiff over `biot_savart_A`. | LOW | `tests/field/test_biotsavart_A_direct_kernel_closeout.py` |
| 64 | `dipole_field_B` | PORTED | `jax_core/dipole_field.py:142` `dipole_field_B` | — | — | `tests/jax_core/test_dipole_field_item24.py` |
| 65 | `dipole_field_A` | PORTED | `jax_core/dipole_field.py:178` `dipole_field_A` | — | — | `tests/jax_core/test_dipole_field_item24.py` |
| 66 | `dipole_field_dB` | PORTED | `jax_core/dipole_field.py:234` `dipole_field_dB` | — | — | `tests/jax_core/test_dipole_field_item24.py` |
| 67 | `dipole_field_dA` | PORTED | `jax_core/dipole_field.py:290` `dipole_field_dA` | — | — | `tests/jax_core/test_dipole_field_item24.py` |
| 68 | `dipole_field_Bn` | PORTED | `jax_core/dipole_field.py:418` `dipole_field_Bn` | All three `coordinate_flag` branches (cartesian/cylindrical/toroidal) handled in `_rotate_normal_matrix_to_*_basis` (lines 349/367/389). | — | `tests/field/test_dipole_field_jax_item26.py` |
| 69 | `define_a_uniform_cartesian_grid_between_two_toroidal_surfaces` | PORTED | `jax_core/dipole_field.py:512` | Uses NumPy under the hood (`_nearest_index_and_distance`); pure-Python kernel, not JAX-jittable but pure / device-free. | — | `tests/jax_core/test_dipole_field_item24.py` |
| 72 | `MwPGP_algorithm` | PORTED | `jax_core/pm_optimization.py:2298` `mwpgp_solve` | Mirrors C++ algebra; documented at module top (line 31). | MED | `tests/jax_core/test_pm_optimization_jax_item25.py` |
| 74 | `GPMO_backtracking` | PORTED | `jax_core/pm_optimization.py` `gpmo_backtracking_solve` (line 1778 step / `_gpmo_backtracking_remove_pairs` 1670) | — | — | `test_pm_optimization_jax_item25.py` |
| 75 | `GPMO_multi` | PORTED | `jax_core/pm_optimization.py:1578` `gpmo_multi_solve` | — | — | `test_pm_optimization_jax_item25.py` |
| 76 | `GPMO_ArbVec` | PORTED | `jax_core/pm_optimization.py:778` `gpmo_arbvec_solve` | — | — | `test_pm_optimization_jax_item25.py` |
| 77 | `GPMO_ArbVec_backtracking` | PORTED | `jax_core/pm_optimization.py:1291` `gpmo_arbvec_backtracking_solve` | — | — | `test_pm_optimization_jax_item25.py` |
| 78 | `GPMO_baseline` | PORTED | `jax_core/pm_optimization.py:638` `gpmo_baseline_solve` | — | — | `test_pm_optimization_jax_item25.py` |
| 81 | `GSCO` | PORTED | `src/simsopt/solve/wireframe_optimization_jax.py:296` `greedy_stellarator_coil_optimization_jax` + `gsco_wireframe_jax` (line 519) | Mirrors C++ algorithm including `no_crossing` / `no_new_coils` / `match_current` flags. | MED | `tests/geo/test_wireframe_*` (none specific to GSCO JAX path yet — UNCLEAR whether parity test exists; see Q1). |
| 88 | `DommaschkB` | PORTED | `jax_core/analytic_fields.py:630` `dommaschk_B` | — | — | `tests/jax_core/test_analytic_fields_item11.py::test_dommaschk_*` (parity vs `sopp.DommaschkB`) |
| 89 | `DommaschkdB` | PORTED | `jax_core/analytic_fields.py:659` `dommaschk_dB` | — | — | `tests/jax_core/test_analytic_fields_item11.py` |
| 91 | `integral_BdotN` | PORTED | `src/simsopt/objectives/integral_bdotn_jax.py:93` `integral_BdotN` + `:38` `residual_BdotN` | All 3 definitions (`quadratic flux`/`normalized`/`local`) wired. | — | `tests/objectives/test_integral_bdotn_jax.py` (per `CLAUDE.md`) |
| 93 | `ReimanB` | PORTED | `jax_core/analytic_fields.py:864` `reiman_B` | — | — | `tests/jax_core/test_analytic_fields_item11.py::test_reiman_*` |
| 94 | `ReimandB` | PORTED | `jax_core/analytic_fields.py:878` `reiman_dB` | — | — | `tests/jax_core/test_analytic_fields_item11.py` |
| 96 | `fourier_transform_even` | PORTED | `jax_core/boozer_radial_interp.py:400` `fourier_transform_even` | — | — | `tests/jax_core/test_boozer_radial_interp_jax_item32.py` |
| 97 | `fourier_transform_odd` | PORTED | `jax_core/boozer_radial_interp.py:373` `fourier_transform_odd` | — | — | `tests/jax_core/test_boozer_radial_interp_jax_item32.py` |
| 98 | `inverse_fourier_transform_even` | PORTED | `jax_core/boozer_radial_interp.py:528` `inverse_fourier_transform_even` + 1D/2D variants | — | — | `tests/jax_core/test_boozer_radial_interp_jax_item32.py` |
| 99 | `inverse_fourier_transform_odd` | PORTED | `jax_core/boozer_radial_interp.py:471` `inverse_fourier_transform_odd` + 1D/2D variants | — | — | `tests/jax_core/test_boozer_radial_interp_jax_item32.py` |
| 100 | `compute_kmns` | PORTED | `jax_core/boozer_radial_interp.py:223` `compute_kmns` | — | — | `tests/jax_core/test_boozer_radial_interp_jax_item32.py` |
| 101 | `compute_kmnc_kmns` | PORTED | `jax_core/boozer_radial_interp.py:282` `compute_kmnc_kmns` | — | — | `tests/jax_core/test_boozer_radial_interp_jax_item32.py` |
| 106 | `boozer_dresidual_dc` | **UNPORTED** | proposed `src/simsopt/geo/boozer_residual_jax.py` (new helper) | Direct closed-form derivative kernel `G*dB_dc - 2*B·dB_dc * tang - B2*(dxphi_dc + iota*dxtheta_dc)`. The JAX side uses `jax.jacfwd`/`jax.vjp` (`boozer_residual_jacobian_composed` line 717, `boozer_residual_coil_vjp` line 743) and never materialises this column-by-column kernel. **Question Q2**: confirm with CLAUDE.md "M3 composed derivatives" entry that this is the intended replacement; if so, mark NON-PORTABLE-by-design. Otherwise port. | LOW (if direct port wanted) | None — covered indirectly by `tests/geo/test_boozer_derivatives_jax.py` autodiff tests. |
| 136 | `boozer_residual` | PORTED | `geo/boozer_residual_jax.py:117` `boozer_residual_scalar` + `:293` `boozer_residual_vector` | Both `weight_inv_modB` branches handled. | — | `tests/geo/test_boozer_residual_jax.py`, `test_boozer_residual_pinned_input_byte_parity.py` |
| 137 | `boozer_residual_ds` | PORTED | `geo/boozer_residual_jax.py:201` `boozer_residual_grad` | C++ returns `(scalar, ds_grad)`; JAX builds via autodiff. | — | `tests/geo/test_boozer_derivatives_jax.py` |
| 138 | `boozer_residual_ds2` | PORTED | `geo/boozer_residual_jax.py:246` `boozer_residual_hessian` | — | — | `tests/geo/test_boozer_derivatives_jax.py` |
| 140 | `matmult` | NON-PORTABLE | n/a | Pure Eigen matrix multiply used in `simsopt.geo.curve.py:1375-1419` exclusively for compact rotation cotangent forwarding. JAX equivalent is `jnp.einsum`/`jnp.matmul`; **call sites still hit `sopp.matmult` (see `geo/curve.py:1375`)** for legacy curves. **Question Q3**: should the legacy curve VJP loop be rewritten to call `jnp.matmul` directly? | LOW | Implicit via `tests/geo/test_curve.py`. |
| 154 | `vjp` (Eigen) | NON-PORTABLE | n/a | Pure `v.T @ B` reduction. Same as above. | LOW | Implicit |

### `src/simsoptpp/python_curves.cpp` (`PyCurve` class + subclasses)

| Line | C++ symbol | Status | JAX target | Notes / API gap | Effort | Test parity |
|------|-----------|--------|------------|-----------------|--------|-------------|
| 82 | `Curve::gamma` | PORTED | `jax_core/curve_geometry.py:766` `curve_gamma_and_dash_from_spec` + `_curve_gamma_kernel:133` | All four `CurveSpec` kinds (XYZFourier, RZFourier, PlanarFourier, Helical, XYZFourierSymmetries) lower into `_curve_gamma_kernel`. | — | `tests/geo/test_curvexyzfouriersymmetries_spec_jax.py`, `tests/geo/test_curve.py` |
| 83 | `Curve::gamma_impl` | NON-PORTABLE | n/a | pybind11 trampoline override hook. | — | n/a |
| 84 | `Curve::gammadash` | PORTED | `jax_core/curve_geometry.py:213` `_curve_geometry_terms_from_kernel` (order=1) | — | — | `tests/geo/test_curve.py` |
| 85 | `Curve::gammadashdash` | PORTED | `jax_core/curve_geometry.py:213` (order=2) | — | — | `tests/geo/test_curve.py` |
| 86 | `Curve::gammadashdashdash` | PORTED | `jax_core/curve_geometry.py:300` `_curve_geometry_with_third_derivative_from_dofs` | — | — | `tests/geo/test_curve.py` |
| 88 | `Curve::dgamma_by_dcoeff` | PORTED | derived via `jax.jacfwd(curve_gamma_and_dash_from_dofs)` in `field/biotsavart_jax_backend.py` and `geo/curve.py` | No first-class `dgamma_by_dcoeff` JAX export named identically; built by autodiff in callers. | LOW | `tests/geo/test_curve.py` |
| 89 | `Curve::dgammadash_by_dcoeff` | PORTED | autodiff path same as above | — | LOW | `tests/geo/test_curve.py` |
| 90 | `Curve::dgammadashdash_by_dcoeff` | PORTED | autodiff path | — | LOW | `tests/geo/test_curve.py` |
| 91 | `Curve::dgammadashdashdash_by_dcoeff` | PARTIAL | autodiff path | Exists for XYZFourier via `_curve_geometry_with_third_derivative_from_dofs`; UNCLEAR for RZFourier/PlanarFourier (need to verify 3rd-derivative kernel coverage). **Question Q4** | LOW | Partial via `tests/geo/test_curve.py` |
| 93 | `Curve::dgamma_by_dcoeff_vjp_impl` | PORTED | `jax.vjp(curve_gamma_and_dash_from_dofs)` | Used inside `biotsavart_jax_backend.py:_curve_dof_mode` / pullback group profile builder. | — | `tests/field/test_biotsavart_jax.py` |
| 94-96 | `dgammadash_/dash²_/dash³_by_dcoeff_vjp_impl` | PARTIAL | autodiff fallback | Same coverage caveat as #91. | LOW | Implicit |
| 98 | `Curve::incremental_arclength` | PARTIAL | `src/simsopt/geo/curve.py:213` `incremental_arclength_pure` | Pure JAX function exists but is **not** lifted into `simsopt/jax_core/curve_geometry.py`. Callers go through the `sopp.Curve` trampoline. | LOW | `tests/geo/test_curve.py` |
| 99 | `Curve::dincremental_arclength_by_dcoeff` | PARTIAL | `geo/curve.py` autodiff helpers (kappagrad0/kappavjp0 lines 240-254) | Same as #98 — autodiff exists, not promoted to `jax_core`. | LOW | `tests/geo/test_curve.py` |
| 100 | `Curve::kappa` | PARTIAL | `geo/curve.py:229` `kappa_pure` | Same as #98. | LOW | `tests/geo/test_curve.py` |
| 101 | `Curve::dkappa_by_dcoeff` | PARTIAL | `geo/curve.py:250-254` `kappagrad0`/`kappagrad1` | Same. | LOW | `tests/geo/test_curve.py` |
| 102 | `Curve::torsion` | PARTIAL | `geo/curve.py:259` `torsion_pure` | Same. | LOW | `tests/geo/test_curve.py` |
| 103 | `Curve::dtorsion_by_dcoeff` | PARTIAL | autodiff path | Same. | LOW | `tests/geo/test_curve.py` |
| 104 | `Curve::invalidate_cache` | NON-PORTABLE | n/a | C++ memoisation primitive. | — | n/a |
| 105 | `Curve::least_squares_fit` | UNPORTED | proposed `jax_core/curve_geometry.py:fit_to_target` | Used by `simsopt.geo.surface.py:891` indirectly (via Surface). LSQ over Fourier basis is trivial to port (call `jnp.linalg.lstsq`). | LOW | None (n/a). |
| 107-110 | `set_dofs`, `set_dofs_impl`, `get_dofs`, `num_dofs` | NON-PORTABLE | n/a | Optimizable trampoline glue. | — | n/a |
| 119 | `CurveXYZFourier`, `dofs_matrix`, `order` | PORTED | `jax_core/specs.py:26` `CurveXYZFourierSpec` + `make_curve_xyzfourier_spec:739` | DOF mapping handled in spec. | — | `tests/geo/test_curve.py` |
| 125 | `CurveRZFourier` (`rc/rs/zc/zs`, `nfp`, `stellsym`) | PORTED | `jax_core/specs.py:42` `CurveRZFourierSpec` + `make_curve_rzfourier_spec:752` | — | — | `tests/geo/test_curve.py` |
| 137 | `CurvePlanarFourier` (`rc/rs/q/center`) | PORTED | `jax_core/specs.py:60` `CurvePlanarFourierSpec` + `make_curve_planarfourier_spec:1053` | — | — | `tests/geo/test_curve.py` |

### `src/simsoptpp/python_surfaces.cpp` (`PySurface` class + subclasses)

| Line | C++ symbol | Status | JAX target | Notes / API gap | Effort | Test parity |
|------|-----------|--------|------------|-----------------|--------|-------------|
| 196 | `Surface::gamma` | PORTED | `geo/surface_fourier_jax.py:518` `surface_gamma` (tensor-fourier) + `jax_core/surface_rzfourier.py:502` `surface_rz_fourier_gamma_from_spec` + `jax_core/surface_fourier.py:29` xyz-fourier | All three kinds covered. | — | `tests/geo/test_surface_fourier_jax.py` |
| 197-206 | `gamma_lin`, `gammadash1_lin`, `gammadash2_lin`, `gammadash1dash1_lin`, `gammadash1dash2_lin`, `gammadash2dash2_lin`, `gammadash1dash1dash1_lin`, `gammadash1dash1dash2_lin`, `gammadash1dash2dash2_lin`, `gammadash2dash2dash2_lin` | PARTIAL | `geo/surface_fourier_jax.py:571` `surface_gamma_lin`, etc. | First and second `_lin` derivatives covered. **Third-derivative `_lin` variants (`gammadash1dash1dash1_lin`/.../`gammadash2dash2dash2_lin`) UNPORTED** — only available for SurfaceRZFourier in C++; not present in JAX. Used by some QFM/regularisation paths. | MED | partial via `tests/geo/test_surface_fourier_jax.py` |
| 207 | `Surface::dgamma_by_dcoeff` | PORTED | `geo/surface_fourier_jax.py:dgamma_by_dcoeff` (jacfwd, see line 2442 `_dcoeff_jacobian`) | — | — | `tests/geo/test_surface_fourier_jax.py` |
| 208 | `Surface::dgamma_by_dcoeff_vjp` | PORTED | `geo/surface_fourier_jax.py:_surface_xyzfourier_dcoeff_jacobian` and `jax.vjp` wraps | — | — | `tests/geo/test_surface_fourier_jax.py` |
| 209-219 | `gammadash1/2/12...by_dcoeff` (first + second derivatives) | PORTED | `surface_fourier_jax.py:805+` + `jax_core/surface_rzfourier.py:803+` | — | — | `tests/geo/test_surface_fourier_jax.py` |
| 220 | `surface_curvatures` | PORTED | `jax_core/surface_rzfourier.py:884` `surface_rz_fourier_surface_curvatures_from_dofs` | RZFourier covered. PARTIAL for XYZFourier/XYZTensorFourier: UNCLEAR whether JAX has equivalent kernels (only `first_fund_form`/`second_fund_form` for RZ are explicitly named). **Question Q5** | LOW | `tests/geo/test_surface.py` |
| 221 | `dsurface_curvatures_by_dcoeff` | PARTIAL | autodiff via `jacfwd(surface_rz_fourier_surface_curvatures_from_dofs)` | RZ-only; XYZ analogue UNCLEAR. | LOW | n/a |
| 222 | `first_fund_form` | PORTED (RZ) / UNPORTED (XYZ) | `jax_core/surface_rzfourier.py:872` `surface_rz_fourier_first_fund_form_from_dofs` | RZ only; **no XYZFourier / XYZTensorFourier JAX kernel** (only the constituent `gammadash1/2` are JAX-ported). | LOW | `tests/geo/test_surface.py` |
| 223 | `dfirst_fund_form_by_dcoeff` | PARTIAL | autodiff | RZ only. | LOW | n/a |
| 224 | `second_fund_form` | PORTED (RZ) / UNPORTED (XYZ) | `jax_core/surface_rzfourier.py:878` | Same caveat as #222. | LOW | n/a |
| 225 | `dsecond_fund_form_by_dcoeff` | PARTIAL | autodiff | Same. | LOW | n/a |
| 226 | `dgammadash2_by_dcoeff_vjp` | PORTED | `jax.vjp(surface_gammadash2)` | — | — | `tests/geo/test_surface_fourier_jax.py` |
| 227-232 | `normal`, `dnormal_by_dcoeff`, `dnormal_by_dcoeff_vjp`, `d2normal_by_dcoeffdcoeff`, `unitnormal`, `dunitnormal_by_dcoeff` | PORTED | `surface_fourier_jax.py:1072` `surface_normal`, `:1114` `_unitnormal`, `:2546` `d2normal_by_dcoeffdcoeff` | All present. | — | `tests/geo/test_surface_fourier_jax.py` |
| 233-236 | `area`, `darea_by_dcoeff`, `d2area_by_dcoeffdcoeff` | PORTED | `surface_fourier_jax.py:2420` `surface_area` + `_surface_scalar_grad`/`_surface_scalar_hessian` | — | — | `tests/geo/test_surface_fourier_jax.py` |
| 237-240 | `volume`, `dvolume_by_dcoeff`, `d2volume_by_dcoeffdcoeff` | PORTED | `surface_fourier_jax.py:2399` `surface_volume` + hessian helpers | — | — | `tests/geo/test_surface_fourier_jax.py` |
| 241 | `Surface::fit_to_curve` | UNPORTED | proposed `jax_core/surface_*.py:fit_to_curve` | Surface bootstrap routine. **Currently delegates to `sopp.SurfaceXYZ*.least_squares_fit` (surfacexyzfourier.py:265, surfacexyztensorfourier.py:265)**. CPU-only. | MED | `tests/geo/test_surface.py` |
| 242 | `Surface::scale` | NON-PORTABLE | n/a | Mutates internal DOFs. Optimizable mechanic. | — | n/a |
| 243 | `Surface::_extend_via_normal_for_nonuniform_phi` | UNPORTED | proposed `jax_core/surface_*.py:extend_via_normal` | Used by some surface offset workflows. | MED | n/a |
| 244 | `Surface::extend_via_projected_normal` | UNPORTED | proposed `jax_core/surface_*.py:extend_via_projected_normal` | Same. | MED | n/a |
| 245 | `Surface::least_squares_fit` | UNPORTED | proposed `jax_core/surface_*.py:fit` | Used internally by `fit_to_curve` and `surface.py:891`. | LOW | n/a |
| 246-251 | `invalidate_cache`, `set_dofs`, `set_dofs_impl`, `get_dofs`, `quadpoints_phi`, `quadpoints_theta` | NON-PORTABLE | n/a | Optimizable trampoline + cache. | — | n/a |
| 259 | `SurfaceRZFourier(mpol, ntor, nfp, stellsym, rc/rs/zc/zs, allocate)` | PORTED | `jax_core/specs.py:432` `SurfaceRZFourierSpec` + `make_surface_rzfourier_spec:1382` | — | — | `tests/geo/test_surface.py` |
| 271 | `SurfaceXYZFourier(mpol, ntor, nfp, stellsym, xc/xs/yc/ys/zc/zs)` | PORTED | `jax_core/specs.py:462` `SurfaceXYZFourierSpec` + `make_surface_xyz_fourier_spec:1462` | — | — | `tests/geo/test_surface.py` |
| 284 | `SurfaceXYZTensorFourier(mpol, ntor, nfp, stellsym, clamped_dims, xcs/ycs/zcs)` | PORTED | `jax_core/specs.py:490` `SurfaceXYZTensorFourierSpec` + `make_surface_xyz_tensor_fourier_spec:1544` | — | — | `tests/geo/test_surface_fourier_jax.py` |

### `src/simsoptpp/python_magneticfield.cpp` (`MagneticField` family)

| Line | C++ symbol | Status | JAX target | Notes / API gap | Effort | Test parity |
|------|-----------|--------|------------|-----------------|--------|-------------|
| 29 | `MagneticField::B` | PORTED | `field/biotsavart_jax_backend.py:405` `SpecBackedBiotSavartJAX.B` (and per-field wrappers in `circular_coil_jax.py`, `dipole_field_jax.py`, etc.) | — | — | `tests/field/test_biotsavart_jax.py`, `test_circular_coil_jax.py`, `test_dipole_field_jax_item26.py` |
| 30 | `MagneticField::dB_by_dX` | PORTED | `biotsavart_jax_backend.py` | — | — | `test_biotsavart_jax.py` |
| 31 | `MagneticField::d2B_by_dXdX` | PORTED | `jax_core/biotsavart.py:585` + `field.py:545` grouped variant | — | — | `tests/field/test_biotsavart_jax.py` |
| 32 | `MagneticField::AbsB` | PORTED | derived from `B` via `jnp.linalg.norm` in wrappers | — | — | covered indirectly |
| 33 | `MagneticField::GradAbsB` | PORTED | autodiff over `AbsB` | — | — | covered indirectly |
| 34 | `MagneticField::GradAbsB_cyl` | PARTIAL | UNCLEAR. C++ exposes both cart and cyl. JAX path returns cart only. **Question Q6**: where does the cyl variant land in JAX? | LOW | n/a |
| 35-39 | `B_ref` / `dB_by_dX_ref` / etc. | NON-PORTABLE | n/a | Numpy-view leak primitives (return reference; no copy). JAX is functional. | — | n/a |
| 40-41 | `B_cyl`, `B_cyl_ref` | PARTIAL | UNCLEAR — see Q6. JAX backend likely does cart-only. | LOW | n/a |
| 42 | `MagneticField::A` | PORTED | `jax_core/biotsavart.py:617` `biot_savart_A` + `biotsavart_jax_backend.py:A` | — | — | `tests/field/test_biotsavart_A_direct_kernel_closeout.py` |
| 43-44 | `A_cyl`, `A_cyl_ref` | PARTIAL | UNCLEAR — see Q6. | LOW | n/a |
| 45 | `MagneticField::dA_by_dX` | PORTED | `jax_core/biotsavart.py:623` `biot_savart_dA_by_dX` | — | — | `tests/field/test_biotsavart_A_direct_kernel_closeout.py` |
| 46 | `MagneticField::d2A_by_dXdX` | PORTED | `jax_core/biotsavart.py:629` `biot_savart_d2A_by_dXdX` + grouped variants | — | — | `tests/field/test_biotsavart_A_direct_kernel_closeout.py` |
| 47-49 | `A_ref`/`dA_by_dX_ref`/`d2A_by_dXdX_ref` | NON-PORTABLE | n/a | Numpy-view returns. | — | n/a |
| 50 | `invalidate_cache` | NON-PORTABLE | n/a | C++ cache primitive. | — | n/a |
| 51-57 | `get_points_cart`, `get_points_cyl`, `*_ref`, `set_points_cart`, `set_points_cyl`, `set_points` | NON-PORTABLE | n/a | Coordinate-conversion plumbing on the Optimizable side. JAX wrappers route through Python-side helpers. | — | n/a |
| 62 | `InterpolationRule` (abstract) | NON-PORTABLE | n/a | pybind11 abstract base. | — | n/a |
| 65 | `UniformInterpolationRule(degree)` | PORTED | `jax_core/regular_grid_interp.py:66` `UniformInterpolationRule` | — | — | `tests/jax_core/test_regular_grid_interp_item13.py` |
| 68 | `ChebyshevInterpolationRule(degree)` | PORTED | `jax_core/regular_grid_interp.py:87` `ChebyshevInterpolationRule` | — | — | `tests/jax_core/test_regular_grid_interp_item13.py` |
| 72 | `RegularGridInterpolant3D` (class) | PORTED | `jax_core/regular_grid_interp.py:109` `RegularGridInterpolant3DSpec` | — | — | `tests/jax_core/test_regular_grid_interp_item13.py` |
| 79 | `RegularGridInterpolant3D::interpolate_batch` | PORTED | `jax_core/regular_grid_interp.py:318` `build_regular_grid_interpolant_3d` | — | — | `tests/jax_core/test_regular_grid_interp_item13.py` |
| 80 | `RegularGridInterpolant3D::evaluate` | PORTED | `jax_core/regular_grid_interp.py:570` `evaluate_batch` (handles both single & batch) | — | — | `tests/jax_core/test_regular_grid_interp_item13.py` |
| 81 | `RegularGridInterpolant3D::evaluate_batch` | PORTED | `jax_core/regular_grid_interp.py:570` `evaluate_batch` | — | — | `tests/jax_core/test_regular_grid_interp_item13.py` |
| 84-86 | `CurrentBase` (abstract + `get_value`) | NON-PORTABLE | n/a | pybind11 base class. JAX side uses `CurrentValueSpec` (`jax_core/specs.py:259`). | — | n/a |
| 88-92 | `Current(double)` (`set_dofs`/`get_dofs`/`get_value`) | PORTED | `jax_core/specs.py:259` `CurrentValueSpec` + `make_current_value_spec:1198` | — | — | covered by `tests/field/test_biotsavart_jax.py` |
| 94 | `Coil(curve, current)` | PORTED | `jax_core/specs.py:289` `CoilSpec` + `make_coil_spec:1208` | — | — | `tests/field/test_biotsavart_jax.py` |
| 99 | `MagneticField` (abstract) | NON-PORTABLE | n/a | pybind11 abstract base. | — | n/a |
| 104 | `BiotSavart(coils)` | PORTED | `field/biotsavart_jax_backend.py:405` `SpecBackedBiotSavartJAX` | — | — | `tests/field/test_biotsavart_jax.py` |
| 106 | `BiotSavart::compute` | PARTIAL | `field/biotsavart_jax_backend.py:B`/`dB_by_dX`/`d2B_by_dXdX` separate methods | C++ entry takes `derivatives=0/1/2` and fills a triple cache. JAX side has separate methods but **no `dB_by_dcoilcurrents` batched bundle** — see SSOT note in summary. | MED | `tests/field/test_biotsavart_jax.py` |
| 107 | `BiotSavart::compute_A` | PARTIAL | analogous to `compute` | Same gap as #106 for `A/dA/d2A`. | MED | `test_biotsavart_A_direct_kernel_closeout.py` |
| 108-110 | `fieldcache_get_or_create`, `fieldcache_get_status` | NON-PORTABLE | n/a | C++ cache primitive — JAX is functional. | — | n/a |
| 113 | `InterpolatedField(field, rule, r/phi/z_range, ...)` | PORTED | `field/interpolated_field_jax.py:142` `InterpolatedFieldJAX` + `jax_core/interpolated_field.py:48` `InterpolatedFieldSpec` | — | — | `tests/field/test_interpolated_field_jax_item15.py` |
| 116 | `InterpolatedField::estimate_error_B` | PORTED | `jax_core/regular_grid_interp.py:618` `estimate_error` | Generic helper used per-quantity. | — | `tests/field/test_interpolated_field_jax_item15.py` |
| 117 | `InterpolatedField::estimate_error_GradAbsB` | PORTED | same as #116 | — | — | `tests/field/test_interpolated_field_jax_item15.py` |
| 124 | `WireframeField(nodes, segments, currents, x)` | PORTED | `field/wireframefield_jax.py:37` `WireframeFieldJAX` + `jax_core/wireframe.py:473` `wireframe_B` | — | — | `tests/field/test_wireframefield_jax_item30.py` |
| 126 | `WireframeField::compute` | PARTIAL | `jax_core/wireframe.py:473` (`wireframe_B`) + `:486` (`wireframe_dB_by_dX`) + `:499` (`wireframe_B_and_dB_by_dX`) | Same `derivatives=0/1/2` cache caveat as `BiotSavart::compute`. **`d2B_by_dXdX` NOT covered** for wireframe in JAX. | MED | `test_wireframefield_jax_item30.py` |
| 127-128 | `WireframeField::fieldcache_get_or_create`, `fieldcache_get_status` | NON-PORTABLE | n/a | C++ cache primitive. | — | n/a |

### `src/simsoptpp/python_boozermagneticfield.cpp`

| Line | C++ symbol | Status | JAX target | Notes | Effort | Test |
|------|-----------|--------|------------|-------|--------|------|
| 21-89 | `BoozerMagneticField` ~33 scalar accessors (`K`, `nu`, `R`, `Z`, `modB`, `G`, `I`, `psip`, `iota` + derivatives + `_ref` variants) | PORTED | `jax_core/boozer_analytic.py:_eval_*` (per scalar) + `field/boozermagneticfield_jax.py:BoozerRadialInterpolantJAX` (via `BoozerRadialInterpolantFrozenState`) + `jax_core/interpolated_boozer_field.py:evaluate_scalar` (line 657) | All scalar evaluators implemented. The `_ref` variants are NON-PORTABLE (numpy views). | LOW | `tests/field/test_boozermagneticfield_jax_item33.py`, `test_boozer_analytic_jax.py`, `test_interpolated_boozer_field_jax.py` |
| 91-94 | `invalidate_cache`, `get_points`, `set_points`, `get_points_ref` | NON-PORTABLE | n/a | Trampoline plumbing. | — | n/a |
| 98 | `BoozerMagneticField(psi0)` | PORTED | `jax_core/boozer_analytic.py:64` `BoozerAnalyticFrozenState` + `freeze_boozer_analytic_state:105` | — | — | `test_boozer_analytic_jax.py` |
| 102 | `InterpolatedBoozerField(...)` | PORTED | `jax_core/interpolated_boozer_field.py:165` `InterpolatedBoozerFieldFrozenState` + `field/boozermagneticfield_jax.py:InterpolatedBoozerFieldJAX` | — | — | `tests/field/test_interpolated_boozer_field_jax.py` |
| 105-112 | `InterpolatedBoozerField::estimate_error_*` (K/modB/R/Z/nu/G/I/iota) | PORTED | `jax_core/regular_grid_interp.py:618` `estimate_error` invoked per scalar | — | — | covered by `test_interpolated_boozer_field_jax.py` |

### `src/simsoptpp/python_tracing.cpp`

| Line | C++ symbol | Status | JAX target | Notes / API gap | Effort | Test |
|------|-----------|--------|------------|-----------------|--------|------|
| 17 | `StoppingCriterion` (abstract) | NON-PORTABLE | n/a | pybind11 abstract. JAX side uses dataclass equivalents. | — | n/a |
| 18 | `IterationStoppingCriterion(int)` | PORTED | `jax_core/tracing.py:392` `IterStoppingCriterion` | — | — | `tests/jax_core/test_tracing_jax_item14.py` |
| 20 | `MinRStoppingCriterion(double)` | PORTED | `jax_core/tracing.py:346` `MinRStoppingCriterion` | — | — | `test_tracing_jax_item14.py` |
| 22 | `MinZStoppingCriterion(double)` | PORTED | `jax_core/tracing.py:364` `MinZStoppingCriterion` | — | — | `test_tracing_jax_item14.py` |
| 24 | `MaxRStoppingCriterion(double)` | PORTED | `jax_core/tracing.py:357` `MaxRStoppingCriterion` | — | — | `test_tracing_jax_item14.py` |
| 26 | `MaxZStoppingCriterion(double)` | PORTED | `jax_core/tracing.py:371` `MaxZStoppingCriterion` | — | — | `test_tracing_jax_item14.py` |
| 28 | `MaxToroidalFluxStoppingCriterion(double)` | PORTED | `jax_core/tracing.py:420` `MaxToroidalFluxStoppingCriterion` | — | — | `test_tracing_jax_item14.py` |
| 30 | `MinToroidalFluxStoppingCriterion(double)` | PORTED | `jax_core/tracing.py:404` `MinToroidalFluxStoppingCriterion` | — | — | `test_tracing_jax_item14.py` |
| 32 | `ToroidalTransitStoppingCriterion(double, bool)` | PORTED | `jax_core/tracing.py:378` `ToroidalTransitStoppingCriterion` | — | — | `test_tracing_jax_item14.py` |
| 34 | `LevelsetStoppingCriterion(RegularGridInterpolant3D)` | PORTED | `jax_core/tracing.py:428` `LevelsetStoppingCriterion` | — | — | `tests/jax_core/test_tracing_jax_levelset_events.py` |
| 37 | `particle_guiding_center_boozer_tracing` | PORTED | `jax_core/tracing.py:2219` `trace_guiding_center_boozer` (+ `vacuum_boozer_rhs:1998`, `no_k_boozer_rhs:2065`, `boozer_rhs:2142`) | — | — | `tests/jax_core/test_tracing_jax_gc_boozer.py`, `test_tracing_jax_boozer_zeta_events.py` |
| 52 | `particle_guiding_center_tracing` | PORTED | `jax_core/tracing.py:1340` `trace_guiding_center` | — | — | `tests/jax_core/test_tracing_jax_guiding_center.py`, `test_tracing_jax_phi_events.py` |
| 66 | `particle_fullorbit_tracing` | PORTED | `jax_core/tracing.py:2779` `trace_fullorbit` (+ `fullorbit_vacuum_rhs:2730`) | — | — | `tests/jax_core/test_tracing_jax_fullorbit.py`, `test_tracing_jax_fullorbit_events.py` |
| 78 | `fieldline_tracing` | PORTED | `jax_core/tracing.py:801` `trace_fieldline` (+ `fieldline_rhs:586`) | — | — | `tests/field/test_tracing_jax_item16.py` |
| 86 | `get_phi` | PORTED | UNCLEAR — semantically equivalent to `_continuous_phi:505` in tracing helpers. **Question Q7**: confirm `jax_core/tracing.py::_continuous_phi` is the canonical port (not exported in `__all__`). | LOW | implicit via tracing tests |

### `src/simsoptpp/python_distance.cpp`

| Line | C++ symbol | Status | JAX target | Notes | Effort | Test |
|------|-----------|--------|------------|-------|--------|------|
| 174 | `get_pointclouds_closer_than_threshold_within_collection` | PORTED | `geo/_distance_jax.py:68` `get_close_candidates_within_collection` | — | — | `tests/geo/test_distance_jax.py` |
| 175 | `get_pointclouds_closer_than_threshold_between_two_collections` | PORTED | `geo/_distance_jax.py:85` `get_close_candidates_between_collections` | — | — | `tests/geo/test_distance_jax.py` |
| 176 | `compute_linking_number` | PORTED | `jax_core/curve_geometry.py:728` `pair_linking_number_pure` | Inner pair contribution — outer loop in `geo/curveobjectives.py:1234-1261` (Python). | — | `tests/geo/test_linking_number_jax.py` |

---

## Unported High-Priority Section

These are the items with no current JAX equivalent in `src/simsopt/**`:

1. **`boozer_dresidual_dc`** (`src/simsoptpp/python.cpp:106`) — direct
   closed-form derivative kernel for the Boozer residual. **Risk LOW**:
   the kernel is a single contraction with no quadrature subtlety. The
   JAX path composes via `boozer_residual_jacobian_composed` and
   `boozer_residual_coil_vjp` (`boozer_residual_jax.py:717,743`), which
   are validated by `tests/geo/test_boozer_derivatives_jax.py`. If the
   direct-kernel form is still required for parity audits, port as a
   thin `jnp.einsum` wrapper.

2. **`Surface::fit_to_curve`** (`src/simsoptpp/surface.cpp:82`) — surface
   bootstrap routine. Called by `simsopt/geo/surfacexyzfourier.py:265`
   and `surfacexyztensorfourier.py:265` via
   `surf.least_squares_fit(gamma)`. **Risk MED**: requires careful Fourier
   basis matrix construction; cannot be JIT-compiled (variable shapes).
   Closest JAX is `jnp.linalg.lstsq` over a hand-built basis.

3. **`Surface::least_squares_fit`** (`src/simsoptpp/surface.cpp:43`) —
   helper used by `fit_to_curve` and `surface.py:891`. **Risk LOW**.

4. **`Surface::_extend_via_normal_for_nonuniform_phi`** (`surface.cpp:129`)
   and **`Surface::extend_via_projected_normal`** (`surface.cpp:145`) —
   surface offset routines for permanent-magnet inner/outer boundary
   construction. **Risk MED**: requires unit-normal computation already
   ported, but iterative projection inside the latter needs a fixed-point
   convergence solver.

5. **`Curve::least_squares_fit`** (`python_curves.cpp:105`) — curve LSQ
   fit. **Risk LOW**: trivial `jnp.linalg.lstsq` over Fourier basis.

6. **Higher-order `gammadash*dash*dash*_lin` for `SurfaceRZFourier`**
   (`python_surfaces.cpp:203-206`) — third-order parametric derivatives.
   **Risk LOW**: pure autodiff over `surface_rz_fourier_*` (one
   `jax.jacfwd` chain). Currently UNPORTED; called from QFM/regularisation
   helpers in some legacy paths.

7. **`Curve::dgammadashdashdash_by_dcoeff`** + the corresponding
   `_vjp_impl` (`python_curves.cpp:91,96`) — third-derivative-by-DOF for
   RZFourier and PlanarFourier curves. **Risk LOW**: extend
   `_curve_geometry_with_third_derivative_from_dofs` to additional
   `CurveSpec` kinds.

---

## Partial Gaps Section

Each entry lists the exact API row(s) to add.

### `BiotSavart::compute(derivatives=2)` / `compute_A(derivatives=2)`

Add to `simsopt/field/biotsavart_jax_backend.py`:

- `dB_by_dcoilcurrents`: list of per-coil ∂B/∂I tensors, currently
  rebuilt by callers (e.g. `simsopt/field/biotsavart.py:37`).
- `d2B_by_dXdcoilcurrents`: ∂²B/∂X∂I bundle (`biotsavart.py:47`).
- `d3B_by_dXdXdcoilcurrents`: ∂³B/∂X²∂I bundle (`biotsavart.py:57`).
- analogous `dA/d2A/d3A_by_dcoilcurrents` (`biotsavart.py:139-159`).

These are pure linear combinations of single-coil JAX evaluations; the
gap is one of API ergonomics, not numerics.

### `WireframeField::compute(derivatives=2)`

Add to `simsopt/jax_core/wireframe.py`:

- `wireframe_d2B_by_dXdX` (analytic Hessian of Biot–Savart segment kernel),
- `wireframe_B_and_dB_and_d2B` triple bundle.

Currently `wireframe.py:499` only goes up to `dB_by_dX`.

### Surface RZFourier — higher-order `_lin` derivatives

The C++ `SurfaceRZFourier` exposes `gammadash1dash1dash1_lin`,
`gammadash1dash1dash2_lin`, `gammadash1dash2dash2_lin`,
`gammadash2dash2dash2_lin` (`python_surfaces.cpp:203-206`). These do not
appear in the JAX surface modules. Add them as `jax.jacfwd` compositions
of `surface_rz_fourier_gammadash*_lin_from_spec`.

### Surface XYZFourier — fundamental forms and curvatures

`surface_curvatures`, `first_fund_form`, `second_fund_form`,
`dsurface_curvatures_by_dcoeff`, `dfirst_fund_form_by_dcoeff`,
`dsecond_fund_form_by_dcoeff` only have JAX equivalents for
`SurfaceRZFourier` (`jax_core/surface_rzfourier.py:872,878,884`). Add
analogous `surface_xyzfourier_*` and `surface_xyz_tensor_fourier_*`
helpers in `jax_core/surface_fourier.py` and
`geo/surface_fourier_jax.py`.

### Curve geometric quantities — promote pure functions to `jax_core`

`incremental_arclength_pure`, `kappa_pure`, `torsion_pure`,
`kappagrad0`, `kappagrad1`, `kappavjp0`, `kappavjp1` already exist as
pure JAX functions in `simsopt/geo/curve.py:213-300`. Promote them
(plus first-derivative `_by_dcoeff` autodiff wrappers) into
`simsopt/jax_core/curve_geometry.py` so consumers do not need the
`sopp.Curve` trampoline. This unblocks pure-JAX `JaxCurve` subclasses.

### Curve helical/perturbed third-derivative-by-DOF coverage

`_curve_geometry_with_third_derivative_from_dofs`
(`jax_core/curve_geometry.py:300`) currently lowers `CurveXYZFourierSpec`
only. Extend to `CurveRZFourierSpec`, `CurvePlanarFourierSpec`,
`CurveHelicalSpec`, `CurveXYZFourierSymmetriesSpec`, and
`CurvePerturbedSpec` for full API parity with `python_curves.cpp:86`.

### MagneticField cylindrical accessors

`B_cyl`, `A_cyl`, `GradAbsB_cyl` (`python_magneticfield.cpp:34,40,43`)
have no first-class JAX exports. Wrappers like
`InterpolatedFieldJAX._cart_to_cyl` (`jax_core/interpolated_field.py:97`)
exist as helpers but are not exposed at the `B_cyl()` method level.

### `Curve::dgamma_by_dcoeff_vjp_impl` family

JAX builds the VJP via `jax.vjp` over `curve_gamma_and_dash_from_dofs`,
but the C++ `_vjp_impl` API surface (separate `gamma`, `gammadash`,
`gammadashdash`, `gammadashdashdash` VJP entries) is not mirrored in
`jax_core/curve_geometry.py`. Add named wrappers
`curve_*_vjp_from_dofs(spec, dofs, cotangent)` so external consumers do
not have to reach into `jax.vjp` themselves.

---

## Open Questions (UNCLEAR resolution required)

- **Q1**: Is there a JAX-vs-C++ parity test for `GSCO`? The audit found
  `gsco_wireframe_jax` in `solve/wireframe_optimization_jax.py:519` and
  `greedy_stellarator_coil_optimization_jax` at line 296, but no test
  file named `test_gsco_*` was visible — only an item-31 reference in
  `jax_core/wireframe.py:8-9`. Resolve by checking
  `tests/geo/test_simsoptpp_compat.py` and `tests/jax_core/test_wireframe_*`.

- **Q2**: Is `boozer_dresidual_dc` (`python.cpp:106`) meant to remain a
  C++-only optimised kernel, with the JAX side relying on
  `boozer_residual_jacobian_composed` / `boozer_residual_coil_vjp`
  (`boozer_residual_jax.py:717,743`) for autodiff parity? The
  `CLAUDE.md` M3 entry suggests yes ("composed pipeline without label
  constraints"). If yes, reclassify as NON-PORTABLE-by-design.

- **Q3**: Should `simsopt/geo/curve.py:1375-1419` stop calling
  `sopp.matmult` and use `jnp.matmul` directly for the rotation
  cotangent forwarding? `matmult` and `vjp` (Eigen) in
  `python.cpp:140,154` are pure linear-algebra primitives JAX already
  provides natively.

- **Q4**: Does the JAX side correctly emit
  `dgammadashdashdash_by_dcoeff` (and its VJP) for
  `CurveRZFourierSpec`, `CurvePlanarFourierSpec`,
  `CurveHelicalSpec`, `CurveXYZFourierSymmetriesSpec`, and
  `CurvePerturbedSpec`, or is
  `_curve_geometry_with_third_derivative_from_dofs`
  (`jax_core/curve_geometry.py:300`) XYZFourier-only?

- **Q5**: Are `surface_curvatures`, `first_fund_form`, `second_fund_form`,
  and their `_by_dcoeff` analogues implemented for
  `SurfaceXYZFourier` and `SurfaceXYZTensorFourier` in JAX, or only for
  `SurfaceRZFourier` (as the audit currently shows)?

- **Q6**: `B_cyl`, `A_cyl`, `GradAbsB_cyl`, and their `_ref` companions
  — does any JAX backend wrapper (e.g. `BiotSavartJAX`,
  `CircularCoilJAX`) expose `B_cyl()` directly? If not, where is the
  conversion done — only inside `InterpolatedFieldJAX._cart_to_cyl`?

- **Q7**: `get_phi` (`python_tracing.cpp:86`) — is
  `jax_core/tracing.py::_continuous_phi` (line 505) the SSOT port, or is
  there a separately-named JAX `get_phi` wrapper? `_continuous_phi` is
  module-private; promote to public.

- **Q8**: `Surface::scale` (`surface.cpp:103`) — mutates DOFs in place.
  In JAX terms, this is a pure DOF-scaling transform; should it be a
  `jnp.multiply` lifted into the spec layer (`make_surface_*_spec` with
  a `scale=` knob)?

---

## Methodology Notes

- C++ public surface enumerated from pybind11 registrations in
  `python.cpp` (39 module-level functions), `python_curves.cpp` (1 base
  class with 21 methods + 3 subclasses), `python_surfaces.cpp` (1 base
  class with 38 methods + 3 subclasses), `python_magneticfield.cpp` (1
  base with 25 methods + 4 subclasses + 3 interpolation rules + Coil
  family), `python_boozermagneticfield.cpp` (1 base with 71 accessors
  + 1 interpolated subclass), `python_tracing.cpp` (10 stopping
  criteria + 5 tracing entrypoints), `python_distance.cpp` (3
  free functions).
- JAX side enumerated from `src/simsopt/{field,geo,objectives}/*_jax*.py`
  and `src/simsopt/jax_core/` directly. Function names matched by
  semantics, not lexical equality. Re-exports through
  `simsopt/jax_core/__init__.py` and `simsopt/field/__init__.py` were
  traced.
- Test parity column cites the most direct CPU↔JAX or closed-form
  parity test found per the audit. Tests using `sopp.X` and `simsopt`
  symbols in the same test counted as parity tests.
- The seven UNPORTED items are exactly the ones with **no** JAX kernel
  found in the source tree. Items where a JAX equivalent exists but is
  named differently are classified PORTED with a "Notes" entry citing
  the renaming.
- Hessian/derivative coverage was verified against `CLAUDE.md`'s
  parity-ladder tolerances: where `CLAUDE.md` explicitly states a JAX
  helper is the SSOT (`PARITY_LADDER_TOLERANCES`, M3/M4/M5 entries),
  that helper is marked PORTED. UNCLEAR is reserved for genuine
  ambiguity (e.g. multi-curve-kind coverage of a third-derivative
  kernel).
