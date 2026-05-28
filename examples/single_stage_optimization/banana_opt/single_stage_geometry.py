import math
import os

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from simsopt.geo import SurfaceRZFourier

from banana_opt.artifact_contracts import (
    validate_boozer_surface_json_current_lineage,
    validate_vacuum_boozer_surface_json,
)
from banana_opt.boozer_warm_start import save_boozer_surface_with_state
from banana_opt.hardware_contracts import COIL_LENGTH_MIN_FRACTION
from banana_opt.hardware_constraint_schema import (
    build_hardware_constraint_status,
    build_threshold_overrides,
    get_hardware_constraint_spec,
    get_hardware_constraint_spec_for_alm_name,
    hardware_constraint_schema_name_for_alm_name,
)
from topology_scorer import (
    score_topology as _score_topology,
    stop_reasons_indicate_broken as _topology_stop_reasons_indicate_broken,
    topology_transport_diagnostics_not_evaluated as _topology_transport_diagnostics_not_evaluated,
)
from banana_opt.surface_mode_contracts import (
    SURFACE_STACK_POLICY_PUBLISHED_FIXED_STACK,
    SurfaceModeContract,
    surface_mode_surface_names,
)
from workflow_helpers import validate_normalized_toroidal_flux


_SURFACE_GOES_BACK_ERROR_FRAGMENT = "surface 'goes back' on itself"
INTERIOR_IOTA_LOW_ORDER_RATIONAL_MAX_DENOMINATOR = 8
INTERIOR_IOTA_LOW_ORDER_RATIONAL_TOLERANCE = 5.0e-3


def _is_surface_goes_back_error(error):
    return _SURFACE_GOES_BACK_ERROR_FRAGMENT in str(error)


def surface_self_intersection_status(surface):
    try:
        return bool(surface.is_self_intersecting()), True
    except Exception as error:
        if not _is_surface_goes_back_error(error):
            raise
        return True, False


def surface_is_self_intersecting(surface):
    return surface_self_intersection_status(surface)[0]


def scale_surface_to_major_radius(surface, major_radius):
    scale = major_radius / surface.major_radius()
    surface.set_dofs(surface.get_dofs() * scale)
    return surface


def build_surface_configs(
    file_loc,
    nphi,
    ntheta,
    seed_label,
    major_radius,
    outer_target_volume,
    num_surfaces,
    inner_surface_ratio,
    surface_factory=SurfaceRZFourier,
):
    if num_surfaces not in {1, 2}:
        raise ValueError(
            "Legacy build_surface_configs supports only 1 or 2 surfaces; "
            "use build_surface_configs_for_contract for published multisurface stacks."
        )
    seed_label = validate_normalized_toroidal_flux(
        seed_label,
        field_name="single-stage surface seed_label",
    )
    outer_reference = surface_factory.from_wout(
        file_loc,
        range="half period",
        nphi=nphi,
        ntheta=ntheta,
        s=seed_label,
    )
    outer_reference = scale_surface_to_major_radius(outer_reference, major_radius)
    configs = [
        {
            "name": "outer",
            "seed_label": seed_label,
            "target_volume": outer_target_volume,
            "initial_surface": outer_reference,
        }
    ]
    if num_surfaces == 1:
        return configs

    if not (0.0 < inner_surface_ratio < 1.0):
        raise ValueError(
            "--inner-surface-ratio must be between 0 and 1 when --num-surfaces=2"
        )

    inner_label = seed_label * inner_surface_ratio
    inner_label = validate_normalized_toroidal_flux(
        inner_label,
        field_name="single-stage inner surface seed_label",
    )
    inner_reference = surface_factory.from_wout(
        file_loc,
        range="half period",
        nphi=nphi,
        ntheta=ntheta,
        s=inner_label,
    )
    inner_reference = scale_surface_to_major_radius(inner_reference, major_radius)
    inner_volume_ratio = inner_reference.volume() / outer_reference.volume()
    inner_target_volume = outer_target_volume * inner_volume_ratio
    if not (0.0 < inner_target_volume < outer_target_volume):
        raise RuntimeError(
            "Derived inner target volume is not strictly inside the outer target volume"
        )

    return [
        {
            "name": "inner",
            "seed_label": inner_label,
            "target_volume": inner_target_volume,
            "initial_surface": inner_reference,
        },
        configs[0],
    ]


def build_surface_configs_for_contract(
    file_loc,
    nphi,
    ntheta,
    seed_label,
    major_radius,
    outer_target_volume,
    surface_mode_contract: SurfaceModeContract,
    surface_factory=SurfaceRZFourier,
):
    seed_label = validate_normalized_toroidal_flux(
        seed_label,
        field_name="single-stage surface seed_label",
    )
    names = surface_mode_surface_names(surface_mode_contract)
    references = []
    for name, label_fraction in zip(names, surface_mode_contract.label_fractions):
        surface_label = validate_normalized_toroidal_flux(
            seed_label * float(label_fraction),
            field_name=f"single-stage {name} surface seed_label",
        )
        reference = surface_factory.from_wout(
            file_loc,
            range="half period",
            nphi=nphi,
            ntheta=ntheta,
            s=surface_label,
        )
        reference = scale_surface_to_major_radius(reference, major_radius)
        references.append(
            {
                "name": name,
                "seed_label": surface_label,
                "reference_volume": reference.volume(),
                "initial_surface": reference,
            }
        )

    outer_reference_volume = references[-1]["reference_volume"]
    configs = []
    for reference in references:
        target_volume = (
            outer_target_volume
            * float(reference["reference_volume"])
            / float(outer_reference_volume)
        )
        configs.append(
            {
                "name": reference["name"],
                "seed_label": reference["seed_label"],
                "target_volume": target_volume,
                "initial_surface": reference["initial_surface"],
            }
        )

    if len(configs) > 1:
        target_volumes = [config["target_volume"] for config in configs]
        if not all(
            left_volume < right_volume
            for left_volume, right_volume in zip(
                target_volumes[:-1],
                target_volumes[1:],
            )
        ):
            raise RuntimeError(
                "Derived multisurface target volumes are not strictly ordered "
                "from inner to outer."
            )
    return configs


def surface_pointcloud_gap(surface_a, surface_b):
    points_a = surface_a.gamma().reshape((-1, 3))
    points_b = surface_b.gamma().reshape((-1, 3))
    nearest, _ = cKDTree(points_b).query(points_a, k=1)
    return float(np.min(nearest))


def surface_cross_section_rz(surface, phi, theta_samples=128):
    cross_section = surface.cross_section(phi, thetas=theta_samples)
    rz = np.zeros((cross_section.shape[0], 2))
    rz[:, 0] = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
    rz[:, 1] = cross_section[:, 2]
    return rz


def planar_segments_intersect(p1, p2, q1, q2, tol=1e-12):
    def orientation(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a, b, c):
        return (
            min(a[0], b[0]) - tol <= c[0] <= max(a[0], b[0]) + tol
            and min(a[1], b[1]) - tol <= c[1] <= max(a[1], b[1]) + tol
        )

    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    if (o1 > tol and o2 < -tol or o1 < -tol and o2 > tol) and (
        o3 > tol and o4 < -tol or o3 < -tol and o4 > tol
    ):
        return True

    if abs(o1) <= tol and on_segment(p1, p2, q1):
        return True
    if abs(o2) <= tol and on_segment(p1, p2, q2):
        return True
    if abs(o3) <= tol and on_segment(q1, q2, p1):
        return True
    if abs(o4) <= tol and on_segment(q1, q2, p2):
        return True
    return False


def planar_polygons_intersect(poly_a, poly_b):
    edges_a = list(zip(poly_a, np.roll(poly_a, -1, axis=0)))
    edges_b = list(zip(poly_b, np.roll(poly_b, -1, axis=0)))
    return any(
        planar_segments_intersect(a0, a1, b0, b1)
        for a0, a1 in edges_a
        for b0, b1 in edges_b
    )


def cross_sections_are_nested(
    inner_surface,
    outer_surface,
    nphi_slices=9,
    theta_samples=128,
):
    bad_phis = []
    for phi in np.linspace(0.0, 1.0 / outer_surface.nfp, nphi_slices, endpoint=False):
        inner_rz = surface_cross_section_rz(
            inner_surface,
            phi,
            theta_samples=theta_samples,
        )
        outer_rz = surface_cross_section_rz(
            outer_surface,
            phi,
            theta_samples=theta_samples,
        )
        outer_path = MplPath(np.vstack([outer_rz, outer_rz[0]]))
        inner_inside_outer = bool(
            np.all(outer_path.contains_points(inner_rz, radius=1e-10))
        )
        disjoint = not planar_polygons_intersect(inner_rz, outer_rz)
        if not (inner_inside_outer and disjoint):
            bad_phis.append(float(phi))
    return len(bad_phis) == 0, bad_phis


def evaluate_surface_stack(
    surface_data,
    vessel_surface=None,
    surface_gap_threshold=0.0,
    enforce_nesting=True,
):
    volumes = [entry["boozer_surface"].surface.volume() for entry in surface_data]
    iotas = [entry["boozer_surface"].res["iota"] for entry in surface_data]
    solve_success = [
        bool(entry["boozer_surface"].res["success"]) for entry in surface_data
    ]

    self_intersections = [
        surface_is_self_intersecting(entry["boozer_surface"].surface)
        for entry in surface_data
    ]
    adjacent_gaps = []
    for left, right in zip(surface_data[:-1], surface_data[1:]):
        adjacent_gaps.append(
            surface_pointcloud_gap(
                left["boozer_surface"].surface,
                right["boozer_surface"].surface,
            )
        )

    outer_vessel_gap = None
    if vessel_surface is not None:
        outer_vessel_gap = surface_pointcloud_gap(
            surface_data[-1]["boozer_surface"].surface,
            vessel_surface,
        )

    volumes_ordered = np.all(np.diff(volumes) > 0.0) if len(volumes) > 1 else True
    gap_ok = all(gap > surface_gap_threshold for gap in adjacent_gaps)
    nesting_ok = True
    bad_nesting_phis = []
    bad_nesting_pairs = []
    if (
        enforce_nesting
        and len(surface_data) > 1
        and all(
            hasattr(entry["boozer_surface"].surface, "cross_section")
            for entry in surface_data
        )
    ):
        for inner_index, (inner_entry, outer_entry) in enumerate(
            zip(surface_data[:-1], surface_data[1:])
        ):
            outer_index = inner_index + 1
            try:
                pair_nesting_ok, pair_bad_phis = cross_sections_are_nested(
                    inner_entry["boozer_surface"].surface,
                    outer_entry["boozer_surface"].surface,
                )
            except RuntimeError as exc:
                if not _is_surface_goes_back_error(exc):
                    raise
                pair_nesting_ok = False
                pair_bad_phis = []
            if not pair_nesting_ok:
                bad_nesting_pairs.append(
                    {
                        "inner_index": inner_index,
                        "outer_index": outer_index,
                        "inner_name": inner_entry.get("name"),
                        "outer_name": outer_entry.get("name"),
                        "bad_phis": pair_bad_phis,
                    }
                )
                bad_nesting_phis.extend(pair_bad_phis)
        nesting_ok = len(bad_nesting_pairs) == 0

    success = (
        all(solve_success)
        and not any(self_intersections)
        and volumes_ordered
        and gap_ok
        and nesting_ok
    )
    return {
        "success": success,
        "solve_success": solve_success,
        "self_intersections": self_intersections,
        "volumes": volumes,
        "iotas": iotas,
        "adjacent_gaps": adjacent_gaps,
        "outer_vessel_gap": outer_vessel_gap,
        "volumes_ordered": volumes_ordered,
        "gap_ok": gap_ok,
        "nesting_ok": nesting_ok,
        "bad_nesting_phis": bad_nesting_phis,
        "bad_nesting_pairs": bad_nesting_pairs,
    }


def evaluate_single_stage_hardware_constraints(
    curve_curve_min_dist,
    cc_dist,
    curve_surface_min_dist,
    cs_dist,
    surface_vessel_min_dist,
    ss_dist,
    max_curvature,
    curvature_threshold,
    *,
    coil_length=None,
    length_target=None,
    poloidal_extent_rad=None,
    poloidal_extent_threshold_rad=None,
    tf_current_A=None,
    tf_current_limit_A=None,
    banana_current_A=None,
    banana_current_max_A=None,
    coil_width=None,
    width_min_threshold=None,
    width_max_threshold=None,
    self_intersect_penalty=None,
    self_intersect_threshold=None,
    lcfs_major_radius_m=None,
    lcfs_minor_radius_m=None,
):
    length_min_target = None
    if length_target is not None:
        length_min_target = COIL_LENGTH_MIN_FRACTION * float(length_target)
    shared_threshold_inputs = (
        ("coil_coil_spacing", cc_dist),
        ("coil_surface_spacing", cs_dist),
        ("max_curvature", curvature_threshold),
        ("poloidal_extent", poloidal_extent_threshold_rad),
        ("width_min", width_min_threshold),
        ("width_max", width_max_threshold),
        ("self_intersect", self_intersect_threshold),
        ("tf_current", tf_current_limit_A),
        ("banana_current", banana_current_max_A),
    )
    search_threshold_overrides = build_threshold_overrides(
        shared_threshold_inputs
        + (
            ("coil_length", length_target),
            ("coil_length_min", length_min_target),
        )
    )
    artifact_threshold_overrides = build_threshold_overrides(
        shared_threshold_inputs + (("coil_length_min", length_min_target),)
    )
    measured_values = {
        "coil_coil_spacing": curve_curve_min_dist,
        "coil_surface_spacing": curve_surface_min_dist,
        "max_curvature": max_curvature,
        "coil_length": coil_length,
        "coil_length_min": coil_length,
        "poloidal_extent": poloidal_extent_rad,
        "width_min": coil_width,
        "width_max": coil_width,
        "self_intersect": self_intersect_penalty,
        "tf_current": tf_current_A,
        "banana_current": banana_current_A,
        "lcfs_major_radius": lcfs_major_radius_m,
        "lcfs_minor_radius": lcfs_minor_radius_m,
    }
    search_hardware_status = build_hardware_constraint_status(
        measured_values,
        applies_to="penalty",
        threshold_overrides=search_threshold_overrides,
    )
    artifact_hardware_status = build_hardware_constraint_status(
        measured_values,
        applies_to="artifact",
        threshold_overrides=artifact_threshold_overrides,
        require_values=True,
    )
    return {
        "success": search_hardware_status["success"],
        "violations": list(search_hardware_status["violations"]),
        "constraints": search_hardware_status["constraints"],
        "search_hardware_status": search_hardware_status,
        "artifact_hardware_status": artifact_hardware_status,
        "curve_curve_min_dist": float(curve_curve_min_dist),
        "curve_curve_distance_metric_kind": "banana_coils",
        "cc_dist": float(cc_dist),
        "curve_surface_min_dist": float(curve_surface_min_dist),
        "cs_dist": float(cs_dist),
        "surface_vessel_min_dist": float(surface_vessel_min_dist),
        "ss_dist": float(ss_dist),
        "max_curvature": float(max_curvature),
        "curvature_threshold": float(curvature_threshold),
        "coil_length": _optional_float(coil_length),
        "length_target": _optional_float(length_target),
        "length_min_target": _optional_float(length_min_target),
        "poloidal_extent_rad": _optional_float(poloidal_extent_rad),
        "poloidal_extent_threshold_rad": _optional_float(poloidal_extent_threshold_rad),
        "tf_current_A": _optional_float(tf_current_A),
        "tf_current_limit_A": _optional_float(tf_current_limit_A),
        "banana_current_A": _optional_float(banana_current_A),
        "banana_current_max_A": _optional_float(banana_current_max_A),
        "coil_width": _optional_float(coil_width),
        "width_min_threshold": _optional_float(width_min_threshold),
        "width_max_threshold": _optional_float(width_max_threshold),
        "self_intersect_penalty": _optional_float(self_intersect_penalty),
        "self_intersect_threshold": _optional_float(self_intersect_threshold),
        "lcfs_major_radius_m": _optional_float(lcfs_major_radius_m),
        "lcfs_minor_radius_m": _optional_float(lcfs_minor_radius_m),
    }


def _lower_bound_measurement_from_signed(threshold, signed_value):
    return float(threshold) - float(signed_value)


def _upper_bound_measurement_from_signed(threshold, signed_value):
    return float(threshold) + float(signed_value)


def _optional_float(value):
    return None if value is None else float(value)


def _threshold_from_override_or_schema(name, threshold_override):
    if threshold_override is not None:
        return float(threshold_override)
    return float(get_hardware_constraint_spec(name).threshold)


def _constraint_signed_value_by_name(objective_eval, payload_kind):
    names = list(objective_eval["constraint_names"])
    value_field = "dual_update_values"
    if payload_kind == "signed_residual" and "raw_dual_update_values" in objective_eval:
        value_field = "raw_dual_update_values"
    values = np.asarray(objective_eval[value_field], dtype=float)
    return {str(name): float(value) for name, value in zip(names, values, strict=True)}


def _status_constraint_names(objective_eval, payload_kind):
    if payload_kind != "signed_residual":
        return None
    constraint_names = tuple(str(name) for name in objective_eval["constraint_names"])
    constraint_blocks = objective_eval.get("constraint_blocks")
    if constraint_blocks is None:
        return tuple(
            dict.fromkeys(
                get_hardware_constraint_spec_for_alm_name(name).name
                for name in constraint_names
            )
        )
    schema_names = []
    for name, block in zip(constraint_names, constraint_blocks, strict=True):
        schema_name = hardware_constraint_schema_name_for_alm_name(name)
        if schema_name is None:
            if str(block) == "physics":
                continue
            raise KeyError(f"Unknown ALM hardware constraint {name!r}.")
        schema_names.append(schema_name)
    return tuple(dict.fromkeys(schema_names))


def _signed_values_for_hardware_constraint(signed_values, schema_name):
    return tuple(
        float(signed_value)
        for alm_name, signed_value in signed_values.items()
        if hardware_constraint_schema_name_for_alm_name(str(alm_name)) == schema_name
    )


def _penalty_objective_constraint_entry(name, value):
    penalty_value = max(float(value), 0.0)
    return {
        "name": str(name),
        "kind": "penalty_objective",
        "threshold": 0.0,
        "value": penalty_value,
        "signed_value": penalty_value,
        "violation": penalty_value,
        "success": penalty_value == 0.0,
        "applies_to": ("penalty",),
        "traversal_policy": "allowed",
    }


def _add_penalty_objective_status_entries(search_hardware_status, constraint_values):
    violation_ratios = {}
    allowed_status = search_hardware_status["allowed_traversal_status"]
    for name, value in constraint_values.items():
        entry = _penalty_objective_constraint_entry(name, value)
        search_hardware_status["constraints"][name] = entry
        allowed_status["constraints"][name] = entry
        violation_ratios[f"{name}_penalty"] = float(entry["violation"])
        if not entry["success"]:
            message = f"{name} penalty {entry['violation']:.6e} exceeds 0"
            search_hardware_status["violations"].append(message)
            allowed_status["violations"].append(message)
            search_hardware_status["success"] = False
            allowed_status["success"] = False
    search_hardware_status["violation_ratios"] = violation_ratios


def evaluate_single_stage_search_hardware_snapshot(
    objective_eval,
    cc_dist,
    cs_dist,
    ss_dist,
    curvature_threshold,
    *,
    coil_length=None,
    length_target=None,
    poloidal_extent_rad=None,
    poloidal_extent_threshold_rad=None,
    tf_current_A=None,
    tf_current_limit_A=None,
    banana_current_A=None,
    banana_current_max_A=None,
    coil_width=None,
    width_min_threshold=None,
    width_max_threshold=None,
    self_intersect_penalty=None,
    self_intersect_threshold=None,
    lcfs_major_radius_m=None,
    lcfs_minor_radius_m=None,
    lcfs_major_radius_threshold=None,
    lcfs_minor_radius_threshold=None,
):
    payload_kind = objective_eval["search_hardware_constraint_payload_kind"]
    if payload_kind not in {"signed_residual", "penalty_objective"}:
        raise ValueError(
            "search hardware constraint payload kind must be "
            "'signed_residual' or 'penalty_objective'."
        )
    signed_values = _constraint_signed_value_by_name(objective_eval, payload_kind)
    length_min_target = (
        None
        if length_target is None
        else COIL_LENGTH_MIN_FRACTION * float(length_target)
    )
    resolved_coil_length_threshold = _threshold_from_override_or_schema(
        "coil_length",
        length_target,
    )
    if coil_length is None:
        coil_length_signed_value = signed_values.get("coil_length_upper_bound")
        if coil_length_signed_value is not None:
            coil_length = _upper_bound_measurement_from_signed(
                resolved_coil_length_threshold,
                coil_length_signed_value,
            )
        elif length_min_target is not None:
            coil_length_min_signed_value = signed_values.get("coil_length_min")
            if coil_length_min_signed_value is not None:
                coil_length = _lower_bound_measurement_from_signed(
                    length_min_target,
                    coil_length_min_signed_value,
                )
    curve_curve_min_dist = None
    if payload_kind == "signed_residual" and "coil_coil_spacing" in signed_values:
        curve_curve_min_dist = _lower_bound_measurement_from_signed(
            cc_dist,
            signed_values["coil_coil_spacing"],
        )
    curve_surface_min_dist = None
    if payload_kind == "signed_residual" and "coil_surface_spacing" in signed_values:
        curve_surface_min_dist = _lower_bound_measurement_from_signed(
            cs_dist,
            signed_values["coil_surface_spacing"],
        )
    surface_vessel_min_dist = None
    max_curvature = None
    if payload_kind == "signed_residual" and "max_curvature" in signed_values:
        max_curvature = _upper_bound_measurement_from_signed(
            curvature_threshold,
            signed_values["max_curvature"],
        )
    if banana_current_A is None and banana_current_max_A is not None:
        banana_current_signed_values = _signed_values_for_hardware_constraint(
            signed_values,
            "banana_current",
        )
        if banana_current_signed_values:
            banana_current_A = max(
                _upper_bound_measurement_from_signed(
                    banana_current_max_A,
                    signed_value,
                )
                for signed_value in banana_current_signed_values
            )
    if poloidal_extent_rad is None and poloidal_extent_threshold_rad is not None:
        poloidal_extent_signed_value = signed_values.get("poloidal_extent")
        if poloidal_extent_signed_value is not None:
            poloidal_extent_rad = _upper_bound_measurement_from_signed(
                poloidal_extent_threshold_rad,
                poloidal_extent_signed_value,
            )
    resolved_lcfs_major_radius_threshold = _threshold_from_override_or_schema(
        "lcfs_major_radius",
        lcfs_major_radius_threshold,
    )
    if lcfs_major_radius_m is None:
        lcfs_major_radius_signed_value = signed_values.get("lcfs_major_radius")
        if lcfs_major_radius_signed_value is not None:
            lcfs_major_radius_m = _upper_bound_measurement_from_signed(
                resolved_lcfs_major_radius_threshold,
                lcfs_major_radius_signed_value,
            )
    resolved_lcfs_minor_radius_threshold = _threshold_from_override_or_schema(
        "lcfs_minor_radius",
        lcfs_minor_radius_threshold,
    )
    if lcfs_minor_radius_m is None:
        lcfs_minor_radius_signed_value = signed_values.get("lcfs_minor_radius")
        if lcfs_minor_radius_signed_value is not None:
            lcfs_minor_radius_m = _upper_bound_measurement_from_signed(
                resolved_lcfs_minor_radius_threshold,
                lcfs_minor_radius_signed_value,
            )

    threshold_overrides = build_threshold_overrides(
        (
            ("coil_coil_spacing", cc_dist),
            ("coil_surface_spacing", cs_dist),
            ("max_curvature", curvature_threshold),
            ("coil_length", length_target),
            ("coil_length_min", length_min_target),
            ("poloidal_extent", poloidal_extent_threshold_rad),
            ("width_min", width_min_threshold),
            ("width_max", width_max_threshold),
            ("self_intersect", self_intersect_threshold),
            ("banana_current", banana_current_max_A),
            ("lcfs_major_radius", lcfs_major_radius_threshold),
            ("lcfs_minor_radius", lcfs_minor_radius_threshold),
        )
    )
    measured_values = {
        "coil_coil_spacing": curve_curve_min_dist,
        "coil_surface_spacing": curve_surface_min_dist,
        "max_curvature": max_curvature,
        "coil_length": coil_length,
        "coil_length_min": coil_length,
        "poloidal_extent": poloidal_extent_rad,
        "width_min": coil_width,
        "width_max": coil_width,
        "self_intersect": self_intersect_penalty,
        "tf_current": tf_current_A,
        "banana_current": banana_current_A,
        "lcfs_major_radius": lcfs_major_radius_m,
        "lcfs_minor_radius": lcfs_minor_radius_m,
    }
    search_hardware_status = build_hardware_constraint_status(
        measured_values,
        applies_to="alm" if payload_kind == "signed_residual" else "penalty",
        names=_status_constraint_names(objective_eval, payload_kind),
        threshold_overrides=threshold_overrides,
    )
    if payload_kind == "penalty_objective":
        _add_penalty_objective_status_entries(search_hardware_status, signed_values)
    search_hardware_status.update(
        {
            "curve_curve_min_dist": curve_curve_min_dist,
            "cc_dist": float(cc_dist),
            "curve_surface_min_dist": curve_surface_min_dist,
            "cs_dist": float(cs_dist),
            "surface_vessel_min_dist": surface_vessel_min_dist,
            "ss_dist": float(ss_dist),
            "max_curvature": max_curvature,
            "curvature_threshold": float(curvature_threshold),
            "coil_length": _optional_float(coil_length),
            "length_target": _optional_float(length_target),
            "length_min_target": _optional_float(length_min_target),
            "poloidal_extent_rad": _optional_float(poloidal_extent_rad),
            "poloidal_extent_threshold_rad": _optional_float(
                poloidal_extent_threshold_rad
            ),
            "coil_width": _optional_float(coil_width),
            "width_min_threshold": _optional_float(width_min_threshold),
            "width_max_threshold": _optional_float(width_max_threshold),
            "self_intersect_penalty": _optional_float(self_intersect_penalty),
            "self_intersect_threshold": _optional_float(self_intersect_threshold),
            "lcfs_major_radius_m": _optional_float(lcfs_major_radius_m),
            "lcfs_minor_radius_m": _optional_float(lcfs_minor_radius_m),
            "lcfs_major_radius_threshold": _optional_float(
                resolved_lcfs_major_radius_threshold
            ),
            "lcfs_minor_radius_threshold": _optional_float(
                resolved_lcfs_minor_radius_threshold
            ),
            "tf_current_A": _optional_float(tf_current_A),
            "tf_current_limit_A": _optional_float(tf_current_limit_A),
            "banana_current_A": _optional_float(banana_current_A),
            "banana_current_max_A": _optional_float(banana_current_max_A),
        }
    )
    return {
        "success": search_hardware_status["success"],
        "violations": list(search_hardware_status["violations"]),
        "constraints": search_hardware_status["constraints"],
        "search_hardware_status": search_hardware_status,
        "artifact_hardware_status": None,
        "curve_curve_min_dist": curve_curve_min_dist,
        "curve_curve_distance_metric_kind": "banana_coils",
        "cc_dist": float(cc_dist),
        "curve_surface_min_dist": curve_surface_min_dist,
        "cs_dist": float(cs_dist),
        "surface_vessel_min_dist": surface_vessel_min_dist,
        "ss_dist": float(ss_dist),
        "max_curvature": max_curvature,
        "curvature_threshold": float(curvature_threshold),
        "coil_length": _optional_float(coil_length),
        "length_target": _optional_float(length_target),
        "length_min_target": _optional_float(length_min_target),
        "poloidal_extent_rad": _optional_float(poloidal_extent_rad),
        "poloidal_extent_threshold_rad": _optional_float(poloidal_extent_threshold_rad),
        "coil_width": _optional_float(coil_width),
        "width_min_threshold": _optional_float(width_min_threshold),
        "width_max_threshold": _optional_float(width_max_threshold),
        "self_intersect_penalty": _optional_float(self_intersect_penalty),
        "self_intersect_threshold": _optional_float(self_intersect_threshold),
        "lcfs_major_radius_m": _optional_float(lcfs_major_radius_m),
        "lcfs_minor_radius_m": _optional_float(lcfs_minor_radius_m),
        "lcfs_major_radius_threshold": _optional_float(
            resolved_lcfs_major_radius_threshold
        ),
        "lcfs_minor_radius_threshold": _optional_float(
            resolved_lcfs_minor_radius_threshold
        ),
        "tf_current_A": _optional_float(tf_current_A),
        "tf_current_limit_A": _optional_float(tf_current_limit_A),
        "banana_current_A": _optional_float(banana_current_A),
        "banana_current_max_A": _optional_float(banana_current_max_A),
    }


def compute_single_stage_surface_vessel_min_dist(
    surface_status,
    outer_surface=None,
    vessel_surface=None,
):
    outer_vessel_gap = surface_status.get("outer_vessel_gap")
    if outer_vessel_gap is not None:
        return float(outer_vessel_gap)
    if outer_surface is None or vessel_surface is None:
        raise ValueError(
            "Need outer_surface and vessel_surface when no cached gap is available."
        )
    return float(
        np.min(
            cdist(
                outer_surface.gamma().reshape((-1, 3)),
                vessel_surface.gamma().reshape((-1, 3)),
            )
        )
    )


def evaluate_single_stage_hardware_snapshot(
    curve_curve_distance_obj,
    cc_dist,
    curve_surface_distance_obj,
    cs_dist,
    surface_status,
    ss_dist,
    banana_curve,
    curvature_threshold,
    outer_surface=None,
    vessel_surface=None,
    coil_length=None,
    length_target=None,
    poloidal_extent_rad=None,
    poloidal_extent_threshold_rad=None,
    tf_current_A=None,
    tf_current_limit_A=None,
    banana_current_A=None,
    banana_current_max_A=None,
    coil_width=None,
    width_min_threshold=None,
    width_max_threshold=None,
    self_intersect_penalty=None,
    self_intersect_threshold=None,
    lcfs_major_radius_m=None,
    lcfs_minor_radius_m=None,
):
    curve_curve_min_dist = float(curve_curve_distance_obj.shortest_distance())
    curve_surface_min_dist = float(curve_surface_distance_obj.shortest_distance())
    surface_vessel_min_dist = compute_single_stage_surface_vessel_min_dist(
        surface_status,
        outer_surface,
        vessel_surface,
    )
    resolved_lcfs_major_radius_m = lcfs_major_radius_m
    resolved_lcfs_minor_radius_m = lcfs_minor_radius_m
    if outer_surface is not None:
        resolved_lcfs_major_radius_m = outer_surface.major_radius()
        resolved_lcfs_minor_radius_m = outer_surface.minor_radius()
    max_curvature = float(np.max(banana_curve.kappa()))
    return evaluate_single_stage_hardware_constraints(
        curve_curve_min_dist,
        cc_dist,
        curve_surface_min_dist,
        cs_dist,
        surface_vessel_min_dist,
        ss_dist,
        max_curvature,
        curvature_threshold,
        coil_length=coil_length,
        length_target=length_target,
        poloidal_extent_rad=poloidal_extent_rad,
        poloidal_extent_threshold_rad=poloidal_extent_threshold_rad,
        tf_current_A=tf_current_A,
        tf_current_limit_A=tf_current_limit_A,
        banana_current_A=banana_current_A,
        banana_current_max_A=banana_current_max_A,
        coil_width=coil_width,
        width_min_threshold=width_min_threshold,
        width_max_threshold=width_max_threshold,
        self_intersect_penalty=self_intersect_penalty,
        self_intersect_threshold=self_intersect_threshold,
        lcfs_major_radius_m=resolved_lcfs_major_radius_m,
        lcfs_minor_radius_m=resolved_lcfs_minor_radius_m,
    )


def snapshot_surface_states(surface_data):
    return {
        "sdofs": [entry["boozer_surface"].surface.x.copy() for entry in surface_data],
        "iota": [entry["boozer_surface"].res["iota"] for entry in surface_data],
        "G": [entry["boozer_surface"].res["G"] for entry in surface_data],
    }


def restore_surface_states(surface_data, state):
    for entry, sdofs, iota, G in zip(
        surface_data,
        state["sdofs"],
        state["iota"],
        state["G"],
    ):
        entry["boozer_surface"].surface.x = sdofs.copy()
        entry["boozer_surface"].res["iota"] = iota
        entry["boozer_surface"].res["G"] = G


def solve_surface_stack_at_dofs(
    x,
    objective,
    surface_data,
    state,
    vessel_surface=None,
    surface_gap_threshold=0.0,
    enforce_nesting=True,
):
    restore_surface_states(surface_data, state)
    objective.x = x
    for entry, iota, G in zip(surface_data, state["iota"], state["G"]):
        entry["boozer_surface"].run_code(iota, G)
    return evaluate_surface_stack(
        surface_data,
        vessel_surface=vessel_surface,
        surface_gap_threshold=surface_gap_threshold,
        enforce_nesting=enforce_nesting,
    )


def continuation_inner_surface_weight(
    num_surfaces,
    accepted_iterations,
    ramp_iterations,
    initial_weight,
):
    if num_surfaces <= 1:
        return 1.0
    if not (0.0 <= initial_weight <= 1.0):
        raise ValueError("inner-surface initial weight must be between 0 and 1")
    if ramp_iterations <= 0:
        return 1.0
    progress = min(max(accepted_iterations, 0), ramp_iterations) / ramp_iterations
    return initial_weight + (1.0 - initial_weight) * progress


def build_surface_search_weights(
    num_surfaces,
    accepted_iterations,
    ramp_iterations,
    initial_inner_weight,
):
    """Return diagnostic surface weights for multisurface search telemetry.

    The experimental continuation ramp is solver/gate-scoped. Production
    objective wrappers use fixed global QS/Boozer objectives, so these weights
    are recorded with diagnostics and do not retarget the descended objective.
    """
    if num_surfaces <= 0:
        raise ValueError("num_surfaces must be positive")
    weights = np.ones(num_surfaces)
    if num_surfaces > 1:
        weights[:-1] = continuation_inner_surface_weight(
            num_surfaces,
            accepted_iterations,
            ramp_iterations,
            initial_inner_weight,
        )
    return weights


def build_surface_search_weights_for_contract(
    surface_mode_contract: SurfaceModeContract,
    accepted_iterations,
    ramp_iterations,
    initial_inner_weight,
):
    if surface_mode_contract.stack_policy == SURFACE_STACK_POLICY_PUBLISHED_FIXED_STACK:
        if len(surface_mode_contract.weights) != surface_mode_contract.num_surfaces:
            raise ValueError(
                "Surface-mode contract weights must match label fractions."
            )
        return np.asarray(surface_mode_contract.weights, dtype=float)
    return build_surface_search_weights(
        surface_mode_contract.num_surfaces,
        accepted_iterations,
        ramp_iterations,
        initial_inner_weight,
    )


def nearest_low_order_rational(value, max_denominator):
    if max_denominator <= 0:
        raise ValueError("max_denominator must be positive")
    iota = float(value)
    if not np.isfinite(iota):
        return None
    best = None
    for denominator in range(1, int(max_denominator) + 1):
        numerator = int(round(iota * denominator))
        divisor = math.gcd(abs(numerator), denominator)
        reduced_numerator = numerator // divisor
        reduced_denominator = denominator // divisor
        rational_value = float(reduced_numerator) / float(reduced_denominator)
        abs_error = abs(iota - rational_value)
        candidate = {
            "numerator": int(reduced_numerator),
            "denominator": int(reduced_denominator),
            "rational_value": rational_value,
            "abs_error": abs_error,
        }
        if best is None or (
            abs_error,
            reduced_denominator,
        ) < (
            best["abs_error"],
            best["denominator"],
        ):
            best = candidate
    return best


def interior_iota_low_order_rational_diagnostics(
    surface_data,
    final_surface_iotas,
    *,
    max_denominator=INTERIOR_IOTA_LOW_ORDER_RATIONAL_MAX_DENOMINATOR,
    tolerance=INTERIOR_IOTA_LOW_ORDER_RATIONAL_TOLERANCE,
):
    if tolerance <= 0.0:
        raise ValueError("interior iota rational tolerance must be positive")
    if max_denominator <= 0:
        raise ValueError("interior iota rational max denominator must be positive")
    matches = []
    for surface_index, (entry, iota) in enumerate(
        zip(surface_data[:-1], final_surface_iotas[:-1])
    ):
        nearest = nearest_low_order_rational(iota, max_denominator)
        if nearest is None or nearest["abs_error"] > float(tolerance):
            continue
        matches.append(
            {
                "surface_index": int(surface_index),
                "surface_name": entry["name"],
                "iota": float(iota),
                **nearest,
            }
        )
    return {
        "near_low_order_rational": bool(matches),
        "matches": matches,
        "max_denominator": int(max_denominator),
        "tolerance": float(tolerance),
        "convention": "signed_iota=p/q; interior_surfaces_exclude_outer",
    }


def build_surface_search_gate(
    num_surfaces,
    accepted_iterations,
    ramp_iterations,
    initial_inner_weight,
    surface_gap_threshold,
):
    """Return the acceptance gate controlled by the multisurface ramp."""
    if num_surfaces <= 1:
        return {
            "surface_gap_threshold": float(surface_gap_threshold),
            "enforce_nesting": True,
            "gate_scale": 1.0,
        }

    gate_scale = continuation_inner_surface_weight(
        num_surfaces,
        accepted_iterations,
        ramp_iterations,
        initial_inner_weight,
    )
    return {
        "surface_gap_threshold": float(surface_gap_threshold) * gate_scale,
        "enforce_nesting": bool(gate_scale >= 1.0),
        "gate_scale": float(gate_scale),
    }


def build_surface_search_gate_for_contract(
    surface_mode_contract: SurfaceModeContract,
    accepted_iterations,
    ramp_iterations,
    initial_inner_weight,
    surface_gap_threshold,
):
    if surface_mode_contract.stack_policy == SURFACE_STACK_POLICY_PUBLISHED_FIXED_STACK:
        return {
            "surface_gap_threshold": float(surface_gap_threshold),
            "enforce_nesting": True,
            "gate_scale": 1.0,
        }
    return build_surface_search_gate(
        surface_mode_contract.num_surfaces,
        accepted_iterations,
        ramp_iterations,
        initial_inner_weight,
        surface_gap_threshold,
    )


def build_scaled_outer_problem(base_fun, base_callback, anchor_x, step_scale):
    if not (0.0 < step_scale <= 1.0):
        raise ValueError("step_scale must be in (0, 1]")

    def scaled_fun(z):
        x = anchor_x + step_scale * z
        J, dJ = base_fun(x)
        return J, step_scale * dJ

    def scaled_callback(z):
        base_callback(anchor_x + step_scale * z)

    return scaled_fun, scaled_callback


def build_scipy_bounds(lower_bounds, upper_bounds):
    lower = np.asarray(lower_bounds, dtype=float)
    upper = np.asarray(upper_bounds, dtype=float)
    if lower.ndim != 1 or upper.ndim != 1:
        raise ValueError("Bounds must be one-dimensional")
    if lower.shape != upper.shape:
        raise ValueError("Lower and upper bounds must have the same shape")
    if not (np.isfinite(lower).any() or np.isfinite(upper).any()):
        return None
    return list(zip(lower.tolist(), upper.tolist()))


def build_local_relative_bounds(anchor_x, relative_radius, lower_bounds, upper_bounds):
    if relative_radius is None:
        return build_scipy_bounds(lower_bounds, upper_bounds)
    if relative_radius <= 0.0:
        raise ValueError("relative_radius must be positive when provided")
    anchor = np.asarray(anchor_x, dtype=float)
    lower = np.asarray(lower_bounds, dtype=float)
    upper = np.asarray(upper_bounds, dtype=float)
    if anchor.ndim != 1:
        raise ValueError("anchor_x must be one-dimensional")
    if anchor.shape != lower.shape or anchor.shape != upper.shape:
        raise ValueError("Local bounds inputs must have matching shapes")
    widths = float(relative_radius) * np.maximum(1.0, np.abs(anchor))
    local_lower = np.where(
        np.isfinite(lower), np.maximum(lower, anchor - widths), anchor - widths
    )
    local_upper = np.where(
        np.isfinite(upper), np.minimum(upper, anchor + widths), anchor + widths
    )
    return build_scipy_bounds(local_lower, local_upper)


def build_scaled_outer_bounds(anchor_x, step_scale, lower_bounds, upper_bounds):
    if not (0.0 < step_scale <= 1.0):
        raise ValueError("step_scale must be in (0, 1]")
    anchor = np.asarray(anchor_x, dtype=float)
    lower = np.asarray(lower_bounds, dtype=float)
    upper = np.asarray(upper_bounds, dtype=float)
    if anchor.ndim != 1:
        raise ValueError("anchor_x must be one-dimensional")
    if anchor.shape != lower.shape or anchor.shape != upper.shape:
        raise ValueError("Scaled bounds inputs must have matching shapes")
    scaled_lower = np.where(np.isfinite(lower), (lower - anchor) / step_scale, -np.inf)
    scaled_upper = np.where(np.isfinite(upper), (upper - anchor) / step_scale, np.inf)
    return build_scipy_bounds(scaled_lower, scaled_upper)


def build_scaled_local_outer_bounds(
    anchor_x,
    step_scale,
    lower_bounds,
    upper_bounds,
    relative_radius,
):
    if relative_radius is None:
        return build_scaled_outer_bounds(
            anchor_x, step_scale, lower_bounds, upper_bounds
        )
    if not (0.0 < step_scale <= 1.0):
        raise ValueError("step_scale must be in (0, 1]")
    anchor = np.asarray(anchor_x, dtype=float)
    lower = np.asarray(lower_bounds, dtype=float)
    upper = np.asarray(upper_bounds, dtype=float)
    if anchor.ndim != 1:
        raise ValueError("anchor_x must be one-dimensional")
    if anchor.shape != lower.shape or anchor.shape != upper.shape:
        raise ValueError("Scaled local bounds inputs must have matching shapes")
    widths = float(relative_radius) * np.maximum(1.0, np.abs(anchor))
    local_lower = np.where(
        np.isfinite(lower), np.maximum(lower, anchor - widths), anchor - widths
    )
    local_upper = np.where(
        np.isfinite(upper), np.minimum(upper, anchor + widths), anchor + widths
    )
    scaled_lower = np.where(
        np.isfinite(local_lower), (local_lower - anchor) / step_scale, -np.inf
    )
    scaled_upper = np.where(
        np.isfinite(local_upper), (local_upper - anchor) / step_scale, np.inf
    )
    return build_scipy_bounds(scaled_lower, scaled_upper)


def evaluate_topology_gate(
    surface,
    bfield,
    nfieldlines,
    tmax,
    tol,
    survival_threshold,
    score_topology_fn=_score_topology,
):
    if nfieldlines <= 0:
        return disabled_topology_gate_status(tmax, tol, survival_threshold)

    topology_result = score_topology_fn(
        surface,
        bfield,
        nfieldlines=nfieldlines,
        tmax=tmax,
        tol=tol,
        nphis=1,
        inset_fraction=0.05,
        field_policy="never",
        compute_transport_diagnostics=False,
        compute_invariant_torus_classification=False,
    )
    earliest_exit = topology_result["first_exit"]
    return _finalize_topology_gate_status(
        {
            "enabled": True,
            "success": bool(topology_result["survival_fraction"] >= survival_threshold),
            "nfieldlines": int(nfieldlines),
            "survived_lines": int(topology_result["survived_lines"]),
            "survival_fraction": float(topology_result["survival_fraction"]),
            "survival_threshold": float(survival_threshold),
            "tmax": float(tmax),
            "tol": float(tol),
            "stop_reason_counts": topology_result["stop_reason_counts"],
            **_topology_gate_first_exit_fields(earliest_exit),
            "evaluation_error": None,
            "evaluation_error_type": None,
            "seed_contract": topology_result.get("seed_contract"),
            "field_model": topology_result.get("field_model"),
            "transport_diagnostics": topology_result.get("transport_diagnostics"),
        }
    )


def _topology_gate_first_exit_fields(earliest_exit):
    if earliest_exit is None:
        return {
            "first_exit_time": None,
            "first_exit_angle": None,
            "first_exit_reason": None,
        }
    return {
        "first_exit_time": earliest_exit["first_exit_time"],
        "first_exit_angle": earliest_exit["first_exit_angle"],
        "first_exit_reason": earliest_exit["stop_reason"],
    }


def topology_gate_state(status):
    if not bool(status.get("enabled", True)):
        return None
    if bool(status.get("broken", False)):
        return "broken"
    stop_reason_counts = status.get("stop_reason_counts") or {}
    survival_fraction = float(status.get("survival_fraction", 0.0))
    survival_threshold = float(status.get("survival_threshold", 0.0))
    finite_metrics = np.isfinite(survival_fraction) and np.isfinite(survival_threshold)
    if not finite_metrics or _topology_stop_reasons_indicate_broken(stop_reason_counts):
        return "broken"
    if bool(status.get("success", False)):
        return "feasible"
    return "modeled_infeasible"


def _finalize_topology_gate_status(status):
    state = topology_gate_state(status)
    finalized = dict(status)
    if state is None:
        finalized["state"] = None
        finalized["broken"] = False
        return finalized
    if state == "broken":
        finalized["success"] = False
    finalized["state"] = state
    finalized["broken"] = state == "broken"
    return finalized


def broken_topology_gate_status(
    tmax,
    tol,
    survival_threshold,
    *,
    nfieldlines,
    error_message,
    error_type,
):
    return _finalize_topology_gate_status(
        {
            "enabled": bool(nfieldlines > 0),
            "success": False,
            "nfieldlines": int(max(nfieldlines, 0)),
            "survived_lines": 0,
            "survival_fraction": 0.0,
            "survival_threshold": float(survival_threshold),
            "tmax": float(tmax),
            "tol": float(tol),
            "stop_reason_counts": {},
            "first_exit_time": None,
            "first_exit_angle": None,
            "first_exit_reason": None,
            "evaluation_error": str(error_message),
            "evaluation_error_type": str(error_type),
            "broken": True,
            "transport_diagnostics": _topology_transport_diagnostics_not_evaluated(
                "topology_gate_broken"
            ),
        }
    )


def disabled_topology_gate_status(tmax, tol, survival_threshold):
    return _finalize_topology_gate_status(
        {
            "enabled": False,
            "success": True,
            "nfieldlines": 0,
            "survived_lines": 0,
            "survival_fraction": 1.0,
            "survival_threshold": float(survival_threshold),
            "tmax": float(tmax),
            "tol": float(tol),
            "stop_reason_counts": {},
            "first_exit_time": None,
            "first_exit_angle": None,
            "first_exit_reason": None,
            "evaluation_error": None,
            "evaluation_error_type": None,
            "transport_diagnostics": _topology_transport_diagnostics_not_evaluated(
                "topology_gate_disabled"
            ),
        }
    )


def topology_gate_deficit(status):
    if not status["enabled"]:
        return 0.0
    return max(
        0.0, float(status["survival_threshold"]) - float(status["survival_fraction"])
    )


def topology_gate_rejection_increment(last_objective, status, penalty_scale):
    base_increment = max(abs(last_objective), 1.0)
    deficit = topology_gate_deficit(status)
    return base_increment * (1.0 + penalty_scale * deficit)


def save_surface_artifacts(
    surface_data,
    biotsavart,
    out_dir,
    stem,
    also_write_outer_legacy,
    boozer_state_interchange_manifest=None,
):
    outer_entry = surface_data[-1]

    def write_surface_artifact(entry, path_stem):
        boozer_surface = entry["boozer_surface"]
        surface = boozer_surface.surface
        biotsavart.set_points(surface.gamma().reshape((-1, 3)))
        unitn = surface.unitnormal()
        field = biotsavart.B().reshape(unitn.shape)
        point_data = {
            "B_N/B": np.sum(field * unitn, axis=2)[:, :, None]
            / np.sqrt(np.sum(field**2, axis=2))[:, :, None]
        }
        surface.to_vtk(path_stem, extra_data=point_data)
        surface.save(path_stem + ".json")
        boozer_surface_path = path_stem + "_boozer_surface.json"
        if boozer_state_interchange_manifest is None:
            state_path = save_boozer_surface_with_state(
                boozer_surface,
                boozer_surface_path,
            )
            validation = validate_boozer_surface_json_current_lineage(
                boozer_surface_path
            )
            if not validation["passed"]:
                raise ValueError(
                    "Zero-current Boozer artifacts must use "
                    "simsopt.geo.BoozerSurface without an I field: "
                    f"{boozer_surface_path}"
                )
        else:
            state_path = save_boozer_surface_with_state(
                boozer_surface,
                boozer_surface_path,
                interchange_manifest=boozer_state_interchange_manifest,
                surface_validator=validate_vacuum_boozer_surface_json,
            )
        if boozer_state_interchange_manifest is not None and state_path is None:
            raise ValueError(
                "Baseline-replayable Boozer artifacts require a solved iota/G "
                f"state sidecar: {boozer_surface_path}"
            )

    for entry in surface_data:
        write_surface_artifact(entry, os.path.join(out_dir, f"{stem}_{entry['name']}"))

    if also_write_outer_legacy:
        write_surface_artifact(outer_entry, os.path.join(out_dir, stem))


def collect_surface_run_metadata(
    surface_data,
    run_status,
    initial_surface_volumes,
    initial_surface_iotas,
    final_surface_volumes,
    final_surface_iotas,
):
    interior_iota_diagnostics = interior_iota_low_order_rational_diagnostics(
        surface_data,
        final_surface_iotas,
    )
    return {
        "SURFACE_NAMES": [entry["name"] for entry in surface_data],
        "SURFACE_SEED_LABELS": [float(entry["seed_label"]) for entry in surface_data],
        "SURFACE_TARGET_VOLUMES": [
            float(entry["target_volume"]) for entry in surface_data
        ],
        "SURFACE_INITIALIZATION_PROVENANCE": [
            entry["initialization_provenance"] for entry in surface_data
        ],
        "FINAL_SURFACE_VOLUMES": [float(value) for value in final_surface_volumes],
        "FINAL_SURFACE_IOTAS": [float(value) for value in final_surface_iotas],
        "SURFACE_SELF_INTERSECTING": [
            bool(value) for value in run_status["self_intersections"]
        ],
        "ADJACENT_SURFACE_GAPS": [
            float(value) for value in run_status["adjacent_gaps"]
        ],
        "OUTER_VESSEL_GAP": None
        if run_status["outer_vessel_gap"] is None
        else float(run_status["outer_vessel_gap"]),
        "SURFACES_NESTED": bool(run_status["nesting_ok"]),
        "BAD_NESTING_PHIS": [float(value) for value in run_status["bad_nesting_phis"]],
        "INITIAL_SURFACE_VOLUMES": [float(value) for value in initial_surface_volumes],
        "INITIAL_SURFACE_IOTAS": [float(value) for value in initial_surface_iotas],
        "FINAL_INTERIOR_IOTA_NEAR_LOW_ORDER_RATIONAL": interior_iota_diagnostics[
            "near_low_order_rational"
        ],
        "FINAL_INTERIOR_IOTA_LOW_ORDER_RATIONAL_MATCHES": interior_iota_diagnostics[
            "matches"
        ],
        "FINAL_INTERIOR_IOTA_LOW_ORDER_RATIONAL_MAX_DENOMINATOR": (
            interior_iota_diagnostics["max_denominator"]
        ),
        "FINAL_INTERIOR_IOTA_LOW_ORDER_RATIONAL_TOLERANCE": (
            interior_iota_diagnostics["tolerance"]
        ),
        "FINAL_INTERIOR_IOTA_LOW_ORDER_RATIONAL_CONVENTION": (
            interior_iota_diagnostics["convention"]
        ),
    }
