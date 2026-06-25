"""Public static facade for the pure JAX kernel layer."""

from . import biotsavart as _biotsavart
from . import circular_coil as _circular_coil
from . import curve_geometry as _curve_geometry
from . import field as _field
from . import finitebuild as _finitebuild
from . import framedcurve as _framedcurve
from . import interpolated_boozer_field as _interpolated_boozer_field
from . import mhd_reductions as _mhd_reductions
from . import objectives_flux as _objectives_flux
from . import reductions as _reductions
from . import specs as _specs
from . import banana as _banana
from . import surface_fourier as _surface_fourier
from . import surface_henneberg as _surface_henneberg
from . import surface_rzfourier as _surface_rzfourier

_JAX_CORE_SPEC_REGISTRATION_MODULES = (_banana,)

_JAX_CORE_MODULES = (
    ("specs", _specs),
    ("finitebuild", _finitebuild),
    ("framedcurve", _framedcurve),
    ("objectives_flux", _objectives_flux),
    ("curve_geometry", _curve_geometry),
    ("field", _field),
    ("circular_coil", _circular_coil),
    ("biotsavart", _biotsavart),
    ("interpolated_boozer_field", _interpolated_boozer_field),
    ("reductions", _reductions),
    ("mhd_reductions", _mhd_reductions),
    ("surface_rzfourier", _surface_rzfourier),
    ("surface_fourier", _surface_fourier),
    ("surface_henneberg", _surface_henneberg),
)

_PACKAGE_EXPORTS_BY_MODULE = {
    "biotsavart": (
        "biot_savart_A",
        "biot_savart_B",
        "biot_savart_B_and_dB",
        "biot_savart_B_vjp",
        "biot_savart_dA_by_dX",
        "biot_savart_dB_by_dX",
        "group_coil_data",
        "grouped_biot_savart_A",
        "grouped_biot_savart_B",
        "invalidate_kernel_cache",
    ),
    "interpolated_boozer_field": (
        "FLUX_FUNCTION_SCALARS",
        "InterpolatedBoozerFieldFrozenState",
        "SYMMETRY_EXPLOIT_SCALARS",
        "build_flux_function_interpolant",
        "build_symmetry_exploit_interpolant",
        "evaluate_interpolated_boozer_scalar",
        "fold_points_for_symmetry",
        "freeze_interpolated_boozer_field_state",
    ),
}

_PACKAGE_EXPORT_ORDER = tuple(
    """
BiotSavartSpec build_filament_gamma_and_dash build_filament_gammas centroid_frame
centroid_frame_dash compute_filament_offsets frenet_frame frenet_frame_dash
rotated_centroid_frame rotated_centroid_frame_dash rotated_frenet_frame rotated_frenet_frame_dash
rotation_alpha rotation_alphadash rotation_dcoeff rotationdash_dcoeff
CoilSpec CoilGroupSpec CoilDofExtractionSpec CoilSetDofExtractionSpec
CoilSymmetrySpec apply_coil_symmetry CurveCWSFourierRZSpec CurveFilamentSpec
CurveHelicalSpec OrientedCurveXYZFourierSpec make_oriented_curve_xyzfourier_spec CurvePlanarFourierSpec
CurveSpec CurveSpecKind CurvePerturbedSpec CurrentValueSpec
CurveRZFourierSpec CurveXYZFourierSpec CurveXYZFourierSymmetriesSpec FieldEvalSpec
FrameRotationSpec FixedSurfaceFluxSpec GroupedCoilSetSpec OptimizableDofMapSpec
RotationSpec SingleStageRuntimeSpec SingleStageSeedSpec SurfaceGarabedianSpec
SurfaceHennebergSpec SurfaceRZFourierSpec SurfaceSpec SurfaceSpecKind
SurfaceXYZFourierSpec SurfaceXYZTensorFourierSpec ZeroRotationSpec build_fourier_basis
closed_curve_self_intersection_min_distance closed_curve_self_intersection_penalty closed_curve_self_intersection_summary closed_curve_self_intersection_tolerance
curve_dincremental_arclength_by_dcoeff_from_dofs curve_dincremental_arclength_by_dcoeff_vjp_from_dofs curve_dkappa_by_dcoeff_from_dofs curve_dkappa_by_dcoeff_vjp_from_dofs
curve_dtorsion_by_dcoeff_from_dofs curve_dtorsion_by_dcoeff_vjp_from_dofs curve_gamma_and_dash_from_dofs curve_gamma_and_dash_from_spec
curve_gamma_vjp_from_dofs curve_geometry_from_dofs curve_geometry_from_spec curve_gammadash_vjp_from_dofs
curve_gammadashdash_vjp_from_dofs curve_gammadashdashdash_vjp_from_dofs curve_incremental_arclength_from_dofs curve_incremental_arclength_from_spec
curve_kappa_from_dofs curve_kappa_from_spec curve_pullback_from_dofs curve_pullback_from_spec
curve_spec_kind surface_spec_kind curve_spec_from_curve curve_spec_with_dofs
curve_spec_with_quadpoints curve_torsion_from_dofs curve_torsion_from_spec coil_set_spec_from_dof_extraction_spec
coil_specs_from_dof_extraction_spec CircularCoilSpec circular_coil_A circular_coil_B
circular_coil_dB biot_savart_A biot_savart_B biot_savart_B_and_dB
biot_savart_B_vjp biot_savart_dA_by_dX biot_savart_dB_by_dX group_coil_data
grouped_biot_savart_A grouped_biot_savart_B invalidate_kernel_cache FLUX_FUNCTION_SCALARS
InterpolatedBoozerFieldFrozenState SYMMETRY_EXPLOIT_SCALARS build_flux_function_interpolant build_symmetry_exploit_interpolant
evaluate_interpolated_boozer_scalar fold_points_for_symmetry freeze_interpolated_boozer_field_state fixed_surface_flux_specs_from_surface
fixed_surface_flux_integral fixed_surface_flux_integral_from_B VALID_REDUCTION_MODES compensated_sum_flat
pairwise_sum_axis pairwise_sum_flat scalar_square_sum validate_reduction_mode
boozer_quasisymmetry_mode_indices boozer_quasisymmetry_residuals iota_target_metric_j iota_weighted_j
well_weighted_j grouped_coil_set_spec_from_coil_specs grouped_biot_savart_A_from_inputs grouped_biot_savart_A_from_spec
grouped_biot_savart_B_and_dB_from_spec grouped_biot_savart_B_from_inputs grouped_biot_savart_B_from_spec grouped_biot_savart_d2A_by_dXdX_from_inputs
grouped_biot_savart_d2A_by_dXdX_from_spec grouped_biot_savart_d2B_by_dXdX_from_inputs grouped_biot_savart_d2B_by_dXdX_from_spec grouped_biot_savart_dA_by_dX_from_inputs
grouped_biot_savart_dA_by_dX_from_spec grouped_biot_savart_dB_by_dX_from_inputs grouped_biot_savart_dB_by_dX_from_spec grouped_coil_currents_from_inputs
grouped_coil_currents_from_spec grouped_coil_index_lists_from_spec grouped_coil_set_spec_from_grouped_data grouped_coil_set_spec_from_inputs
grouped_coil_set_spec_from_lists grouped_coil_set_spec_from_source grouped_field_data_from_spec grouped_field_inputs_from_spec
make_biot_savart_spec make_coil_spec make_coil_dof_extraction_spec make_coil_symmetry_spec
make_coil_group_spec make_coil_set_dof_extraction_spec make_current_value_spec make_curve_cwsfourier_rz_spec
make_curve_filament_spec make_curve_helical_spec make_curve_planarfourier_spec make_curve_perturbed_spec
make_curve_rzfourier_spec make_curve_xyzfourier_spec make_curve_xyzfouriersymmetries_spec make_field_eval_spec
make_frame_rotation_spec make_fixed_surface_flux_spec make_grouped_coil_set_spec make_optimizable_dof_map_spec
make_single_stage_runtime_spec make_single_stage_seed_spec make_surface_garabedian_spec make_surface_henneberg_spec
make_surface_rzfourier_spec make_zero_rotation_spec garabedian_to_rzfourier_spec pair_linking_number_pure
segment_segment_distance_pure surface_rz_fourier_aspect_ratio_from_spec surface_rz_fourier_aspect_ratio_from_dofs surface_rz_fourier_area_from_spec
surface_rz_fourier_area_from_dofs surface_rz_fourier_d2area_from_dofs surface_rz_fourier_d2aspect_ratio_from_dofs surface_rz_fourier_d2major_radius_from_dofs
surface_rz_fourier_d2minor_radius_from_dofs surface_rz_fourier_d2volume_from_dofs surface_rz_fourier_darea_from_dofs surface_rz_fourier_daspect_ratio_from_dofs
surface_rz_fourier_dfirst_fund_form_from_dofs surface_rz_fourier_dofs_from_spec surface_rz_fourier_dmajor_radius_from_dofs surface_rz_fourier_dmean_cross_sectional_area_from_dofs
surface_rz_fourier_dminor_radius_from_dofs surface_rz_fourier_dsecond_fund_form_from_dofs surface_rz_fourier_dsurface_curvatures_from_dofs surface_rz_fourier_dvolume_from_dofs
surface_rz_fourier_first_fund_form_from_spec surface_rz_fourier_first_fund_form_from_dofs surface_rz_fourier_gamma_from_spec surface_rz_fourier_gamma_from_dofs surface_rz_fourier_gamma_lin_from_spec surface_rz_fourier_gamma_lin_from_dofs
surface_rz_fourier_gammadash1dash1_from_spec surface_rz_fourier_gammadash1dash1_from_dofs surface_rz_fourier_gammadash1dash1dash1_lin_from_spec surface_rz_fourier_gammadash1dash1dash1_lin_from_dofs
surface_rz_fourier_gammadash1dash1_vjp_from_dofs surface_rz_fourier_gammadash1dash2_from_spec surface_rz_fourier_gammadash1dash2_from_dofs surface_rz_fourier_gammadash1dash1dash2_lin_from_spec
surface_rz_fourier_gammadash1dash1dash2_lin_from_dofs surface_rz_fourier_gammadash1dash2dash2_lin_from_spec surface_rz_fourier_gammadash1dash2dash2_lin_from_dofs surface_rz_fourier_gammadash1dash2_vjp_from_dofs
surface_rz_fourier_gammadash1_from_spec surface_rz_fourier_gammadash1_from_dofs surface_rz_fourier_gammadash2dash2_from_spec surface_rz_fourier_gammadash2dash2_from_dofs
surface_rz_fourier_gammadash2dash2dash2_lin_from_spec surface_rz_fourier_gammadash2dash2dash2_lin_from_dofs surface_rz_fourier_gammadash2dash2_vjp_from_dofs surface_rz_fourier_gammadash2_from_spec
surface_rz_fourier_gammadash2_from_dofs surface_rz_fourier_major_radius_from_spec surface_rz_fourier_major_radius_from_dofs surface_rz_fourier_mean_cross_sectional_area_from_spec
surface_rz_fourier_mean_cross_sectional_area_from_dofs surface_rz_fourier_minor_radius_from_spec surface_rz_fourier_minor_radius_from_dofs surface_rz_fourier_normal_from_spec
surface_rz_fourier_normal_from_dofs surface_rz_fourier_second_fund_form_from_spec surface_rz_fourier_second_fund_form_from_dofs surface_rz_fourier_dgammadash1dash1_from_dofs
surface_rz_fourier_dgammadash1dash2_from_dofs surface_rz_fourier_dgammadash2dash2_from_dofs surface_rz_fourier_dnormal_from_dofs surface_rz_fourier_spec_from_dofs
surface_rz_fourier_surface_curvatures_from_spec surface_rz_fourier_surface_curvatures_from_dofs surface_rz_fourier_unitnormal_from_spec surface_rz_fourier_unitnormal_from_dofs
surface_rz_fourier_dunitnormal_from_dofs surface_rz_fourier_volume_from_spec surface_rz_fourier_volume_from_dofs surface_xyz_fourier_area_from_spec
surface_xyz_fourier_dfirst_fund_form_from_dofs surface_xyz_fourier_dsecond_fund_form_from_dofs surface_xyz_fourier_dsurface_curvatures_from_dofs surface_xyz_fourier_first_fund_form_from_spec
surface_xyz_fourier_first_fund_form_from_dofs surface_xyz_fourier_gamma_from_spec surface_xyz_fourier_gammadash1_from_spec surface_xyz_fourier_gammadash1dash1_from_spec
surface_xyz_fourier_gammadash1dash1_lin_from_spec surface_xyz_fourier_gammadash1dash1_lin_from_dofs surface_xyz_fourier_gammadash1dash1dash1_lin_from_spec surface_xyz_fourier_gammadash1dash1dash1_lin_from_dofs
surface_xyz_fourier_gammadash1dash1dash2_lin_from_spec surface_xyz_fourier_gammadash1dash1dash2_lin_from_dofs surface_xyz_fourier_gammadash1dash2_from_spec surface_xyz_fourier_gammadash1dash2_lin_from_spec
surface_xyz_fourier_gammadash1dash2_lin_from_dofs surface_xyz_fourier_gammadash1dash2dash2_lin_from_spec surface_xyz_fourier_gammadash1dash2dash2_lin_from_dofs surface_xyz_fourier_gammadash2_from_spec
surface_xyz_fourier_gammadash2dash2_from_spec surface_xyz_fourier_gammadash2dash2_lin_from_spec surface_xyz_fourier_gammadash2dash2_lin_from_dofs surface_xyz_fourier_gammadash2dash2dash2_lin_from_spec
surface_xyz_fourier_gammadash2dash2dash2_lin_from_dofs surface_xyz_fourier_normal_from_spec surface_xyz_fourier_second_fund_form_from_spec surface_xyz_fourier_second_fund_form_from_dofs
surface_xyz_fourier_surface_curvatures_from_spec surface_xyz_fourier_surface_curvatures_from_dofs surface_xyz_fourier_unitnormal_from_spec surface_xyz_fourier_volume_from_spec
surface_henneberg_area_from_spec surface_henneberg_gamma_from_spec surface_henneberg_gammadash1_from_spec surface_henneberg_gammadash2_from_spec
surface_henneberg_normal_from_spec surface_henneberg_unitnormal_from_spec surface_henneberg_volume_from_spec surface_xyz_tensor_fourier_area_from_spec
surface_xyz_tensor_fourier_dfirst_fund_form_from_dofs surface_xyz_tensor_fourier_dsecond_fund_form_from_dofs surface_xyz_tensor_fourier_dsurface_curvatures_from_dofs surface_xyz_tensor_fourier_first_fund_form_from_spec
surface_xyz_tensor_fourier_first_fund_form_from_dofs surface_xyz_tensor_fourier_gamma_from_spec surface_xyz_tensor_fourier_gammadash1_from_spec surface_xyz_tensor_fourier_gammadash1dash1_from_spec
surface_xyz_tensor_fourier_gammadash1dash1_lin_from_spec surface_xyz_tensor_fourier_gammadash1dash1_lin_from_dofs surface_xyz_tensor_fourier_gammadash1dash1dash1_lin_from_spec surface_xyz_tensor_fourier_gammadash1dash1dash1_lin_from_dofs
surface_xyz_tensor_fourier_gammadash1dash1dash2_lin_from_spec surface_xyz_tensor_fourier_gammadash1dash1dash2_lin_from_dofs surface_xyz_tensor_fourier_gammadash1dash2_from_spec surface_xyz_tensor_fourier_gammadash1dash2_lin_from_spec
surface_xyz_tensor_fourier_gammadash1dash2_lin_from_dofs surface_xyz_tensor_fourier_gammadash1dash2dash2_lin_from_spec surface_xyz_tensor_fourier_gammadash1dash2dash2_lin_from_dofs surface_xyz_tensor_fourier_gammadash2_from_spec
surface_xyz_tensor_fourier_gammadash2dash2_from_spec surface_xyz_tensor_fourier_gammadash2dash2_lin_from_spec surface_xyz_tensor_fourier_gammadash2dash2_lin_from_dofs surface_xyz_tensor_fourier_gammadash2dash2dash2_lin_from_spec
surface_xyz_tensor_fourier_gammadash2dash2dash2_lin_from_dofs surface_xyz_tensor_fourier_normal_from_spec surface_xyz_tensor_fourier_second_fund_form_from_spec surface_xyz_tensor_fourier_second_fund_form_from_dofs
surface_xyz_tensor_fourier_surface_curvatures_from_spec surface_xyz_tensor_fourier_surface_curvatures_from_dofs surface_xyz_tensor_fourier_unitnormal_from_spec surface_xyz_tensor_fourier_volume_from_spec
make_surface_xyz_fourier_spec make_surface_xyz_tensor_fourier_spec
""".split()
)

def _build_static_export_map():
    export_to_module = {}
    export_order = []
    for module_name, module in _JAX_CORE_MODULES:
        module_exports = _PACKAGE_EXPORTS_BY_MODULE.get(
            module_name, tuple(module.__all__)
        )
        for export_name in module_exports:
            previous_module = export_to_module.get(export_name)
            if previous_module is not None:
                raise RuntimeError(
                    f"duplicate package export {export_name!r}: "
                    f"{previous_module.__name__!r} and {module.__name__!r}"
                )
            export_to_module[export_name] = module
            export_order.append(export_name)

    package_export_names = set()
    duplicate_order_names = set()
    for name in _PACKAGE_EXPORT_ORDER:
        if name in package_export_names:
            duplicate_order_names.add(name)
        package_export_names.add(name)
    if duplicate_order_names:
        raise RuntimeError(
            f"duplicate package export order entries: {sorted(duplicate_order_names)}"
        )
    missing_order_names = [
        name for name in _PACKAGE_EXPORT_ORDER if name not in export_to_module
    ]
    if missing_order_names:
        raise RuntimeError(
            f"package export order references unknown exports: {missing_order_names}"
        )
    extra_export_names = [
        name for name in export_order if name not in package_export_names
    ]
    if extra_export_names:
        raise RuntimeError(
            f"package export order omits module exports: {extra_export_names}"
        )

    return {
        export_name: export_to_module[export_name]
        for export_name in _PACKAGE_EXPORT_ORDER
    }


_EXPORT_TO_MODULE = _build_static_export_map()
__all__ = _PACKAGE_EXPORT_ORDER
globals().update(
    {
        export_name: getattr(module, export_name)
        for export_name, module in _EXPORT_TO_MODULE.items()
    }
)


def __dir__():
    return sorted(set(globals()) | set(__all__))
