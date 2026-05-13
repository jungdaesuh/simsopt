# Item 06 Coverage Matrix

Status: complete for CPU/JAX oracle closure. CUDA proof is not claimed.

Upstream audit SHA:
`1b0cc3a96063197cdbdd01559e04c25456fbe6ff`.

Search seeds:
`git grep -nE 'SurfaceRZFourier|SurfaceXYZFourier|SurfaceXYZTensorFourier|surface_fourier_jax|stellsym_scatter|surface_normal|surface_gamma|surface_area|surface_volume'`
against `tests/` and `src/simsopt/jax_core/surface_*.py`, plus
`git -C /Users/suhjungdae/code/opensource/simsopt grep ...` against
upstream `tests/geo/` for `SurfaceRZFourier`, `SurfaceXYZFourier`, and
`SurfaceXYZTensorFourier`.

## JAX SurfaceRZFourier kernel parity (current repo)

| Source | Path / Node | Intent | Classification | Evidence |
| --- | --- | --- | --- | --- |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_jax_parity_stellsym` | `surface_rz_fourier_*` parity, stellsym=True, nphi=9, ntheta=10 | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_jax_parity_non_stellsym` | `surface_rz_fourier_*` parity, stellsym=False, nphi=9, ntheta=10 | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax_item06_closeout.py::test_surface_rzfourier_jax_production_scale_non_stellsym_parity` | Production-scale non-stellsym gamma/gd1/gd2/normal/area/volume parity, nphi=32, ntheta=16 | `covered_by_unit_parity` | item 06 closeout witness; same module |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_second_geometry_jacobians_match_cpu` (parametrized stellsym=True/False) | Second-order coordinate derivative parity | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_second_geometry_directional_taylor` (parametrized) | Directional Taylor residuals | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_forms_and_curvatures_derivatives_match_cpu` (parametrized) | First/second fundamental forms and curvatures derivatives | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_forms_and_curvatures_directional_taylor` (parametrized) | Taylor residual on forms/curvatures | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_jax_gauss_bonnet_matches_cpu_oracle` | Gauss-Bonnet check, nphi=32, ntheta=33, stellsym=True | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_spec_is_jittable` | Spec is a jittable pytree | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_geometry_from_spec_matches_scalar_composition` (parametrized) | Fused geometry from spec matches scalar composition | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_geometry_jvp_matches_scalar_composition` | JVP linearity through fused path | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_geometry_vjp_cotangent_matches_scalar_composition` | VJP cotangent parity | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_geometry_jacfwd_matches_scalar_composition` | `jacfwd` parity | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_geometry_hessian_trace_smoke` | Hessian trace smoke | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_geometry_allows_strict_transfer_guard` | Strict transfer guard | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_fused_geometry_reduces_hlo_work` | HLO compression for fused entrypoint | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_scalar_gamma_hlo_stays_single_output` | Single-output HLO for scalar gamma | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rz_geometry_hlo_probe_entrypoint_uses_local_package` | Subprocess HLO probe uses local package | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_geometry_avoids_jnp_arange` | `jnp.arange` not in lowered HLO | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_unitnormal_degenerate_surface_matches_cpu_singularity` | Degenerate-surface unitnormal singularity parity | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_geometry_from_dofs_matches_boozer_hot_path` | Boozer hot path geometry composition parity | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_jax_jacobian_parity_stellsym` | dnormal_by_dcoeff / dunitnormal_by_dcoeff stellsym=True parity | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_jax_jacobian_parity_non_stellsym` | dnormal_by_dcoeff / dunitnormal_by_dcoeff stellsym=False parity | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_area_volume_gradient_parity_stellsym` | darea/dvolume_by_dcoeff stellsym=True parity | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_area_volume_gradient_parity_non_stellsym` | darea/dvolume_by_dcoeff stellsym=False parity | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_scalar_metric_parity_stellsym` | Mean cross-sectional area / minor / major / aspect ratio gradient parity stellsym=True | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_scalar_metric_parity_non_stellsym` | same, stellsym=False | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_scalar_metrics_second_order_taylor` (parametrized stellsym=True/False) | Second-order Taylor on scalar metrics | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_spec_from_dofs_round_trip` | spec_from_dofs round-trip | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_dofs_round_trip_stellsym` | dofs_from_spec round-trip stellsym=True | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_dofs_round_trip_non_stellsym` | dofs_from_spec round-trip stellsym=False | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_from_wout_object_api_parity` | Surface from VMEC wout + JAX parity | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_from_vmec_input_object_api_parity` | Surface from VMEC input + JAX parity | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_from_nescoil_input_object_api_parity` | Surface from NESCOIL input + JAX parity | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_from_focus_object_api_parity` | Surface from FOCUS + JAX parity | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_from_pyqsc_object_api_parity` | Surface from pyQSC + JAX parity | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_copy_object_api_parity` | `copy` preserves JAX parity | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_copy_nfp_recomputes_field_period_grid` | `copy(nfp=...)` recomputes quadpoints | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_copy_object_api_independent_dofs` | independent DOFs after copy | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_change_resolution_object_api_parity` | change_resolution + JAX parity | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_make_rotating_ellipse_object_api_parity` | rotating ellipse helper + JAX parity | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_extend_via_normal_object_api_parity` | extend_via_normal + JAX parity | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_rzfourier_jax.py::test_surface_rzfourier_condense_spectrum_object_api_parity` | condense_spectrum + JAX parity | `covered_by_unit_parity` | same node |

## JAX SurfaceXYZ(Tensor)Fourier kernel parity (current repo)

| Source | Path / Node | Intent | Classification | Evidence |
| --- | --- | --- | --- | --- |
| current | `tests/geo/test_surface_fourier_jax.py::TestSurfaceFourierJaxSimpleTorus::test_gamma_torus` | XYZ gamma vs analytic torus, nphi=20, ntheta=20 | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_fourier_jax.py::TestSurfaceFourierJaxSimpleTorus::test_gammadash1_finite_difference` | FD validation of gammadash1 | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_fourier_jax.py::TestSurfaceFourierJaxSimpleTorus::test_gammadash2_finite_difference` | FD validation of gammadash2 | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_fourier_jax.py::TestSurfaceFourierJaxSimpleTorus::test_normal_matches_analytic_torus_geometry` | Normal vector parity on analytic torus | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_fourier_jax.py::TestSurfaceFourierJaxSimpleTorus::test_area_and_volume_match_analytic_torus` | Area/volume vs analytic torus | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_fourier_jax.py::TestSurfaceFourierJaxHigherOrder::test_nontrivial_modes` | Higher-order modes round-trip, nphi=32, ntheta=32 | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_fourier_jax.py::TestSurfaceFourierJaxHigherOrder::test_basis_shape` | Basis shapes | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_fourier_jax.py::TestSurfaceFourierJaxCppParity::test_gamma_parity` | CPU gamma vs JAX, stellsym=True | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_fourier_jax.py::TestSurfaceFourierJaxCppParity::test_coefficient_derivatives_match_cpp` (parametrized stellsym=True/False) | Coefficient derivatives parity | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_fourier_jax.py::TestSurfaceFourierJaxCppParity::test_second_coordinate_derivatives_match_cpp` (parametrized stellsym=True/False) | Second coordinate derivatives parity | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_fourier_jax.py::TestSurfaceXYZFourierJaxCppParity::test_geometry_and_tangents_match_cpp` (parametrized stellsym=True/False) | XYZ geometry & tangents | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_fourier_jax.py::TestSurfaceXYZFourierJaxCppParity::test_second_coordinate_derivative_dcoeff_match_cpp` (parametrized stellsym=True/False) | XYZ second derivative dcoeff | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_fourier_jax.py::TestSurfaceFourierSecondNormalDerivativeParity::test_d2normal_by_dcoeffdcoeff_matches_cpp` (parametrized SurfaceXYZFourier, SurfaceXYZTensorFourier x stellsym=True/False) | d2normal / coeff-coeff parity | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_fourier_jax.py::TestSurfaceFourierPairedPointParity::test_gamma_and_tangent_lin_match_cpp` (parametrized SurfaceXYZFourier, SurfaceXYZTensorFourier x stellsym=True/False) | Paired-point linearization parity | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_fourier_jax.py::TestSurfaceFourierSpecCppParity::test_tensor_surface_spec_rejects_clamped_dims` | Spec rejects clamped dims | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_fourier_jax.py::TestSurfaceFourierSpecCppParity::test_spec_geometry_and_normals_match_cpp` (parametrized SurfaceXYZFourier, SurfaceXYZTensorFourier x stellsym=True/False) | Spec-driven geometry/normal parity | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_fourier_jax.py::TestSurfaceFourierSpecCppParity::test_spec_second_coordinate_derivatives_match_cpp` (parametrized) | Spec second coord derivatives parity | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_fourier_jax.py::TestSurfaceFourierSpecCppParity::test_area_volume_derivatives_match_cpp` (parametrized) | Spec area/volume derivatives | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_fourier_jax.py::TestSurfaceFourierSpecCppParity::test_area_volume_derivative_taylor_residuals` (parametrized) | Spec area/volume Taylor residuals | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_fourier_jax.py::TestSurfaceFourierSpecCppParity::test_tensor_spec_tangents_area_and_volume_match_cpp` (parametrized stellsym=True/False) | Tensor spec tangents/area/volume parity | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_fourier_jax.py::TestSurfaceFourierObjectApiParity::test_copy_object_api_independent_dofs` (parametrized SurfaceXYZFourier, SurfaceXYZTensorFourier x stellsym=True/False) | Copy + independent dofs | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_fourier_jax.py::TestSurfaceFourierObjectApiParity::test_copy_module_protocol_independent_dofs` (parametrized) | Copy via module protocol | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_fourier_jax.py::TestSurfaceFourierObjectApiParity::test_copy_object_api_variants_preserve_spec_parity` (parametrized) | Copy variants preserve spec parity | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_fourier_jax.py::TestSurfaceFourierObjectApiParity::test_to_rzfourier_and_cross_section_object_api_parity` (parametrized) | to_RZFourier and cross_section object API parity | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_fourier_jax.py::TestSurfaceFourierObjectApiParity::test_least_squares_fit_object_api_parity` (parametrized) | Least-squares fit | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_fourier_jax.py::TestSurfaceFourierObjectApiParity::test_fit_to_curve_object_api_parity` (parametrized) | Fit-to-curve | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_fourier_jax.py::TestSurfaceFourierObjectApiParity::test_scale_object_api_parity` (parametrized) | Scale | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_fourier_jax.py::TestSurfaceFourierObjectApiParity::test_extend_object_api_parity` (parametrized) | Extend | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_fourier_jax.py::TestSurfaceFourierObjectApiParity::test_serialization_object_api_parity` | Serialization | `covered_by_unit_parity` | same node |

## JAX SurfaceFourier CPU-ordered census twins

| Source | Path / Node | Intent | Classification | Evidence |
| --- | --- | --- | --- | --- |
| current | `tests/geo/test_surface_fourier_jax_cpu_ordered.py::test_surface_gamma_cpu_ordered_matches_cpp_within_ulp` | CPU-ordered gamma within ULP of CPP, 20x20 stellsym=True / 13x9 stellsym=False | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_fourier_jax_cpu_ordered.py::test_surface_gammadash_cpu_ordered_matches_cpp` | CPU-ordered gammadash parity | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_fourier_jax_cpu_ordered.py::test_dgamma_by_dcoeff_cpu_ordered_matches_cpp` | CPU-ordered dgamma_by_dcoeff parity | `covered_by_unit_parity` | same node |
| current | `tests/geo/test_surface_fourier_jax_cpu_ordered.py::test_parity_policy_routes_through_cpu_ordered_kernels` | Parity policy enforces CPU-ordered twin | `covered_by_unit_parity` | same node |

## Surface Optimizable / serialization / IO (current repo)

| Source | Path / Node | Intent | Classification | Evidence |
| --- | --- | --- | --- | --- |
| current | `tests/geo/test_surface_rzfourier.py` (all classes; mirrors upstream) | SurfaceRZFourier Optimizable / I/O / VMEC / FOCUS / pyQSC | `oracle_only` | exercises CPU SurfaceRZFourier; JAX parity is asserted by `test_surface_rzfourier_jax.py` and the new closeout test |
| current | `tests/geo/test_surface_xyzfourier.py` | SurfaceXYZFourier Optimizable / serialization | `oracle_only` | exercises CPU SurfaceXYZFourier; JAX parity asserted by `test_surface_fourier_jax.py` |
| current | `tests/geo/test_surface.py` | abstract Surface helpers, quadrature, Gauss-Bonnet, cross-section, self-intersection | `oracle_only` | non-JAX surface helper coverage |
| current | `tests/geo/test_surface_taylor.py` | CPU Taylor / coefficient derivatives | `oracle_only` | CPU validation only |
| current | `tests/geo/test_surface_garabedian.py` | SurfaceGarabedian (auxiliary; skip list) | `not_applicable` | per prompt skip list (`surfacegarabedian.py` and `surfacehenneberg.py`) |
| current | `tests/geo/test_surface_objectives.py` | Surface objectives (CPU) | `oracle_only` | covered separately under item 04 scope; not item 06 |
| current | `tests/geo/test_surface_objectives_jax.py` | JAX surface objectives | `wrapper_only` | item 04 / item 05 scope, exercises downstream Boozer/QFM objectives, not item 06 kernels |

## Upstream SIMSOPT tests (audited at `1b0cc3a96063197cdbdd01559e04c25456fbe6ff`)

| Source | Path / Node | Intent | Classification | Evidence |
| --- | --- | --- | --- | --- |
| upstream | `upstream_hss/master:tests/geo/test_surface_rzfourier.py::SurfaceRZFourierTests::test_aspect_ratio` | CPU aspect-ratio oracle | `oracle_only` | aspect_ratio JAX path validated by `test_surface_rzfourier_jax.py::test_surface_rzfourier_scalar_metric_parity_*` |
| upstream | `upstream_hss/master:tests/geo/test_surface_rzfourier.py::SurfaceRZFourierTests::test_init` | CPU init oracle | `oracle_only` | CPU-only init contract |
| upstream | `upstream_hss/master:tests/geo/test_surface_rzfourier.py::SurfaceRZFourierTests::test_shared_dof_init` | shared-DOF init oracle | `oracle_only` | CPU-only init contract |
| upstream | `upstream_hss/master:tests/geo/test_surface_rzfourier.py::SurfaceRZFourierTests::test_get_dofs`, `::test_set_dofs` | CPU DOF accessor oracles | `oracle_only` | DOF round-trip is `covered_by_unit_parity` via `test_surface_rzfourier_spec_from_dofs_round_trip` and `test_surface_rzfourier_dofs_round_trip_*` |
| upstream | `upstream_hss/master:tests/geo/test_surface_rzfourier.py::SurfaceRZFourierTests::test_from_wout`, `::test_from_vmec_input`, `::test_from_nescoil_input`, `::test_from_nescoil_input_distance`, `::test_from_vmec_2_ways`, `::test_get_and_write_nml`, `::test_from_focus`, `::test_from_pyQSC` | VMEC / NESCOIL / FOCUS / pyQSC IO oracles | `oracle_only` | object-API parity covered by `test_surface_rzfourier_from_*_object_api_parity` |
| upstream | `upstream_hss/master:tests/geo/test_surface_rzfourier.py::SurfaceRZFourierTests::test_extend_via_normal`, `::test_extend_via_normal_non_stellsym` | extend_via_normal oracle (CPU) | `oracle_only` | object-API parity covered by `test_surface_rzfourier_extend_via_normal_object_api_parity` |
| upstream | `upstream_hss/master:tests/geo/test_surface_rzfourier.py::SurfaceRZFourierTests::test_change_resolution` | change_resolution oracle | `oracle_only` | object-API parity covered by `test_surface_rzfourier_change_resolution_object_api_parity` |
| upstream | `upstream_hss/master:tests/geo/test_surface_rzfourier.py::SurfaceRZFourierTests::test_area_volume` | CPU area/volume oracle | `oracle_only` | JAX parity at production scale via item 06 closeout test (stellsym=False) and `test_surface_rzfourier_jax_parity_*` (smaller scale) |
| upstream | `upstream_hss/master:tests/geo/test_surface_rzfourier.py::SurfaceRZFourierTests::test_vjps` | CPU VJP oracle | `oracle_only` | JAX VJP parity covered by `test_surface_rzfourier_second_geometry_jacobians_match_cpu` and `test_surface_rzfourier_jax_jacobian_parity_*` |
| upstream | `upstream_hss/master:tests/geo/test_surface_rzfourier.py::SurfaceRZFourierTests::test_names_order`, `::test_mn`, `::test_mn_matches_names`, `::test_get_rc`, `::test_get_zs`, `::test_set_rc`, `::test_set_zs`, `::test_repr` | CPU naming / mn / accessor oracles | `oracle_only` | CPU bookkeeping; no JAX path |
| upstream | `upstream_hss/master:tests/geo/test_surface_rzfourier.py::SurfaceRZFourierTests::test_serialization`, `::test_shared_dof_serialization` | Surface serialization oracle | `oracle_only` | object-API parity covered by `test_surface_rzfourier_jax.py` object-API tests and `test_serialization_object_api_parity` |
| upstream | `upstream_hss/master:tests/geo/test_surface_rzfourier.py::SurfaceRZFourierTests::test_make_rotating_ellipse` | rotating ellipse helper oracle | `oracle_only` | object-API parity covered by `test_surface_rzfourier_make_rotating_ellipse_object_api_parity` |
| upstream | `upstream_hss/master:tests/geo/test_surface_xyzfourier.py::SurfaceXYZFourierTests::test_toRZFourier_perfect_torus`, `::test_toRZFourier_lossless_at_quadrature_points`, `::test_toRZFourier_small_loss_elsewhere`, `::test_cross_section_torus`, `::test_aspect_ratio_random_torus`, `::test_aspect_ratio_compare_with_cross_sectional_computation` | CPU SurfaceXYZFourier oracles | `oracle_only` | JAX parity covered by `test_surface_fourier_jax.py::TestSurfaceXYZFourierJaxCppParity` and `TestSurfaceFourierObjectApiParity` |
| upstream | `upstream_hss/master:tests/geo/test_surface_xyzfourier.py::SurfaceXYZFourierTests::test_to_vtk`, `::test_serialization`, `::test_shared_dof_init` | VTK / serialization / shared DOF init | `oracle_only` | object-API parity covered by `test_serialization_object_api_parity`; VTK is IO/visualization (skip list) |
| upstream | `upstream_hss/master:tests/geo/test_surface.py` (all Quadpoints/Arclength/Distance/Scaled/BestNphi/Curvature/SelfIntersecting/Util/DofNames tests) | abstract Surface helper / curvature / arclength / cross-section / Gauss-Bonnet oracles | `oracle_only` | JAX uses CPU `Surface` helpers at the wrapper boundary; the JAX kernel path consumes the resulting quadpoints arrays |
| upstream | `upstream_hss/master:tests/geo/test_surface_taylor.py` (all Surface coefficient derivative tests) | CPU coefficient-derivative Taylor oracles | `oracle_only` | JAX FD/Taylor parity covered by `test_surface_rzfourier_second_geometry_directional_taylor`, `test_surface_rzfourier_forms_and_curvatures_directional_taylor`, and `TestSurfaceFourierSpecCppParity::test_area_volume_derivative_taylor_residuals` |
| upstream | `upstream_hss/master:tests/geo/test_surface_objectives.py` | CPU surface objectives | `oracle_only` | item 04 scope (Boozer/QFM/label constraints); not item 06 kernels |
| upstream | `upstream_hss/master:tests/geo/test_surface_garabedian.py` | SurfaceGarabedian / SurfaceHenneberg | `not_applicable` | per prompt skip list |
| upstream | `upstream_hss/master:tests/geo/test_surfacehenneberg.py` | SurfaceHenneberg | `not_applicable` | per prompt skip list |
| upstream | `upstream_hss/master:tests/geo/test_boozersurface.py` | SurfaceRZFourier as Boozer initial state | `wrapper_only` | item 04 owns BoozerSurface; here only the surface DOF/spec contract is exercised, which is `covered_by_unit_parity` via the item 06 kernel rows |
| upstream | `upstream_hss/master:tests/geo/test_qfm.py` | SurfaceRZFourier under QFM surface objective | `wrapper_only` | item 04 owns QFM; surface DOF/spec contract is `covered_by_unit_parity` here |
| upstream | `upstream_hss/master:tests/geo/test_curve.py`, `::test_curve_objectives.py`, `::test_finitebuild.py`, `::test_plot.py`, `::test_pm_grid.py`, `::test_wireframe_toroidal.py` | downstream Surface consumers (distance objectives, plotting, PM grids, wireframes) | `wrapper_only` | downstream wrappers exercised by their own item manifests (01, 07, 08, etc.); item 06 owns only the surface adapter contract |

No matrix row is unclassified. The previously missing
production-scale stellsym=False SurfaceRZFourier parity row is closed
by the new test
`tests/geo/test_surface_rzfourier_jax_item06_closeout.py::test_surface_rzfourier_jax_production_scale_non_stellsym_parity`.
