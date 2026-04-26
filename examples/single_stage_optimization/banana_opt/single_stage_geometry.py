import os

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from simsopt.geo import SurfaceRZFourier

from banana_opt.hardware_constraint_schema import (
    build_hardware_constraint_status,
    build_threshold_overrides,
)
from topology_scorer import (
    score_topology as _score_topology,
    stop_reasons_indicate_broken as _topology_stop_reasons_indicate_broken,
    topology_transport_diagnostics_not_evaluated as _topology_transport_diagnostics_not_evaluated,
)
from workflow_helpers import validate_normalized_toroidal_flux


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

    if (
        (o1 > tol and o2 < -tol or o1 < -tol and o2 > tol)
        and (o3 > tol and o4 < -tol or o3 < -tol and o4 > tol)
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
    vessel_gap_threshold=0.0,
    enforce_nesting=True,
):
    volumes = [entry["boozer_surface"].surface.volume() for entry in surface_data]
    iotas = [entry["boozer_surface"].res["iota"] for entry in surface_data]
    solve_success = [bool(entry["boozer_surface"].res["success"]) for entry in surface_data]

    def safe_is_self_intersecting(surface):
        try:
            return bool(surface.is_self_intersecting())
        except Exception:
            return True

    self_intersections = [
        safe_is_self_intersecting(entry["boozer_surface"].surface)
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
    vessel_gap_ok = (
        outer_vessel_gap is None or outer_vessel_gap > vessel_gap_threshold
    )
    nesting_ok = True
    bad_nesting_phis = []
    if (
        enforce_nesting
        and len(surface_data) > 1
        and all(
            hasattr(entry["boozer_surface"].surface, "cross_section")
            for entry in surface_data
        )
    ):
        try:
            nesting_ok, bad_nesting_phis = cross_sections_are_nested(
                surface_data[0]["boozer_surface"].surface,
                surface_data[-1]["boozer_surface"].surface,
            )
        except Exception:
            nesting_ok = False

    success = (
        all(solve_success)
        and not any(self_intersections)
        and volumes_ordered
        and gap_ok
        and vessel_gap_ok
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
        "vessel_gap_ok": vessel_gap_ok,
        "nesting_ok": nesting_ok,
        "bad_nesting_phis": bad_nesting_phis,
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
):
    shared_threshold_inputs = (
        ("coil_coil_spacing", cc_dist),
        ("coil_surface_spacing", cs_dist),
        ("surface_vessel_spacing", ss_dist),
        ("max_curvature", curvature_threshold),
        ("poloidal_extent", poloidal_extent_threshold_rad),
        ("tf_current", tf_current_limit_A),
        ("banana_current", banana_current_max_A),
    )
    search_threshold_overrides = build_threshold_overrides(
        shared_threshold_inputs + (("coil_length", length_target),)
    )
    artifact_threshold_overrides = build_threshold_overrides(
        shared_threshold_inputs
    )
    measured_values = {
        "coil_coil_spacing": curve_curve_min_dist,
        "coil_surface_spacing": curve_surface_min_dist,
        "surface_vessel_spacing": surface_vessel_min_dist,
        "max_curvature": max_curvature,
        "coil_length": coil_length,
        "poloidal_extent": poloidal_extent_rad,
        "tf_current": tf_current_A,
        "banana_current": banana_current_A,
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
    )
    return {
        "success": search_hardware_status["success"],
        "violations": list(search_hardware_status["violations"]),
        "constraints": search_hardware_status["constraints"],
        "search_hardware_status": search_hardware_status,
        "artifact_hardware_status": artifact_hardware_status,
        "curve_curve_min_dist": float(curve_curve_min_dist),
        "cc_dist": float(cc_dist),
        "curve_surface_min_dist": float(curve_surface_min_dist),
        "cs_dist": float(cs_dist),
        "surface_vessel_min_dist": float(surface_vessel_min_dist),
        "ss_dist": float(ss_dist),
        "max_curvature": float(max_curvature),
        "curvature_threshold": float(curvature_threshold),
        "coil_length": _optional_float(coil_length),
        "length_target": _optional_float(length_target),
        "poloidal_extent_rad": _optional_float(poloidal_extent_rad),
        "poloidal_extent_threshold_rad": _optional_float(
            poloidal_extent_threshold_rad
        ),
        "tf_current_A": _optional_float(tf_current_A),
        "tf_current_limit_A": _optional_float(tf_current_limit_A),
        "banana_current_A": _optional_float(banana_current_A),
        "banana_current_max_A": _optional_float(banana_current_max_A),
    }


def _lower_bound_measurement_from_signed(threshold, signed_value):
    return float(threshold) - float(signed_value)


def _upper_bound_measurement_from_signed(threshold, signed_value):
    return float(threshold) + float(signed_value)


def _optional_float(value):
    return None if value is None else float(value)


def _constraint_signed_value_by_name(objective_eval):
    names = list(objective_eval["constraint_names"])
    values = np.asarray(objective_eval["dual_update_values"], dtype=float)
    return {
        str(name): float(value)
        for name, value in zip(names, values, strict=True)
    }


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
):
    signed_values = _constraint_signed_value_by_name(objective_eval)
    payload_kind = objective_eval["search_hardware_constraint_payload_kind"]
    if payload_kind not in {"signed_residual", "penalty_objective"}:
        raise ValueError(
            "search hardware constraint payload kind must be "
            "'signed_residual' or 'penalty_objective'."
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
    if payload_kind == "signed_residual" and "surface_vessel_spacing" in signed_values:
        surface_vessel_min_dist = _lower_bound_measurement_from_signed(
            ss_dist,
            signed_values["surface_vessel_spacing"],
        )
    max_curvature = None
    if payload_kind == "signed_residual" and "max_curvature" in signed_values:
        max_curvature = _upper_bound_measurement_from_signed(
            curvature_threshold,
            signed_values["max_curvature"],
        )
    if banana_current_A is None and banana_current_max_A is not None:
        banana_current_signed_value = signed_values.get("banana_current_upper_bound")
        if banana_current_signed_value is not None:
            banana_current_A = _upper_bound_measurement_from_signed(
                banana_current_max_A,
                banana_current_signed_value,
            )
    if poloidal_extent_rad is None and poloidal_extent_threshold_rad is not None:
        poloidal_extent_signed_value = signed_values.get("poloidal_extent")
        if poloidal_extent_signed_value is not None:
            poloidal_extent_rad = _upper_bound_measurement_from_signed(
                poloidal_extent_threshold_rad,
                poloidal_extent_signed_value,
            )

    threshold_overrides = build_threshold_overrides(
        (
            ("coil_coil_spacing", cc_dist),
            ("coil_surface_spacing", cs_dist),
            ("surface_vessel_spacing", ss_dist),
            ("max_curvature", curvature_threshold),
            ("poloidal_extent", poloidal_extent_threshold_rad),
            ("banana_current", banana_current_max_A),
        )
    )
    measured_values = {
        "coil_coil_spacing": curve_curve_min_dist,
        "coil_surface_spacing": curve_surface_min_dist,
        "surface_vessel_spacing": surface_vessel_min_dist,
        "max_curvature": max_curvature,
        "coil_length": coil_length,
        "poloidal_extent": poloidal_extent_rad,
        "tf_current": tf_current_A,
        "banana_current": banana_current_A,
    }
    search_hardware_status = build_hardware_constraint_status(
        measured_values,
        applies_to="penalty",
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
            "poloidal_extent_rad": _optional_float(poloidal_extent_rad),
            "poloidal_extent_threshold_rad": _optional_float(
                poloidal_extent_threshold_rad
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
        "cc_dist": float(cc_dist),
        "curve_surface_min_dist": curve_surface_min_dist,
        "cs_dist": float(cs_dist),
        "surface_vessel_min_dist": surface_vessel_min_dist,
        "ss_dist": float(ss_dist),
        "max_curvature": max_curvature,
        "curvature_threshold": float(curvature_threshold),
        "coil_length": _optional_float(coil_length),
        "length_target": _optional_float(length_target),
        "poloidal_extent_rad": _optional_float(poloidal_extent_rad),
        "poloidal_extent_threshold_rad": _optional_float(
            poloidal_extent_threshold_rad
        ),
        "tf_current_A": _optional_float(tf_current_A),
        "tf_current_limit_A": _optional_float(tf_current_limit_A),
        "banana_current_A": _optional_float(banana_current_A),
        "banana_current_max_A": _optional_float(banana_current_max_A),
    }


def compute_single_stage_surface_vessel_min_dist(
    surface_vessel_distance_obj,
    surface_status,
    outer_surface=None,
    vessel_surface=None,
):
    if surface_vessel_distance_obj is not None and hasattr(
        surface_vessel_distance_obj,
        "shortest_distance",
    ):
        return float(surface_vessel_distance_obj.shortest_distance())
    outer_vessel_gap = surface_status.get("outer_vessel_gap")
    if outer_vessel_gap is not None:
        return float(outer_vessel_gap)
    if outer_surface is None or vessel_surface is None:
        raise ValueError(
            "Need outer_surface and vessel_surface when no surface-vessel "
            "distance object or cached gap is available."
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
    surface_vessel_distance_obj,
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
):
    curve_curve_min_dist = float(curve_curve_distance_obj.shortest_distance())
    curve_surface_min_dist = float(curve_surface_distance_obj.shortest_distance())
    surface_vessel_min_dist = compute_single_stage_surface_vessel_min_dist(
        surface_vessel_distance_obj,
        surface_status,
        outer_surface,
        vessel_surface,
    )
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
    vessel_gap_threshold=0.0,
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
        vessel_gap_threshold=vessel_gap_threshold,
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


def build_surface_search_gate(
    num_surfaces,
    accepted_iterations,
    ramp_iterations,
    initial_inner_weight,
    surface_gap_threshold,
    vessel_gap_threshold,
):
    if num_surfaces <= 1:
        return {
            "surface_gap_threshold": float(surface_gap_threshold),
            "vessel_gap_threshold": float(vessel_gap_threshold),
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
        "vessel_gap_threshold": float(vessel_gap_threshold) * gate_scale,
        "enforce_nesting": bool(gate_scale >= 1.0),
        "gate_scale": float(gate_scale),
    }


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
    local_lower = np.where(np.isfinite(lower), np.maximum(lower, anchor - widths), anchor - widths)
    local_upper = np.where(np.isfinite(upper), np.minimum(upper, anchor + widths), anchor + widths)
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
        return build_scaled_outer_bounds(anchor_x, step_scale, lower_bounds, upper_bounds)
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
    local_lower = np.where(np.isfinite(lower), np.maximum(lower, anchor - widths), anchor - widths)
    local_upper = np.where(np.isfinite(upper), np.minimum(upper, anchor + widths), anchor + widths)
    scaled_lower = np.where(np.isfinite(local_lower), (local_lower - anchor) / step_scale, -np.inf)
    scaled_upper = np.where(np.isfinite(local_upper), (local_upper - anchor) / step_scale, np.inf)
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
    return max(0.0, float(status["survival_threshold"]) - float(status["survival_fraction"]))


def topology_gate_rejection_increment(last_objective, status, penalty_scale):
    base_increment = max(abs(last_objective), 1.0)
    deficit = topology_gate_deficit(status)
    return base_increment * (1.0 + penalty_scale * deficit)


def save_surface_artifacts(surface_data, biotsavart, out_dir, stem, also_write_outer_legacy):
    outer_entry = surface_data[-1]

    def write_surface_artifact(entry, path_stem):
        boozer_surface = entry["boozer_surface"]
        surface = boozer_surface.surface
        biotsavart.set_points(surface.gamma().reshape((-1, 3)))
        unitn = surface.unitnormal()
        field = biotsavart.B().reshape(unitn.shape)
        point_data = {
            "B_N/B": np.sum(field * unitn, axis=2)[:, :, None]
            / np.sqrt(np.sum(field ** 2, axis=2))[:, :, None]
        }
        surface.to_vtk(path_stem, extra_data=point_data)
        surface.save(path_stem + ".json")
        boozer_surface.save(path_stem + "_boozer_surface.json")

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
    return {
        "SURFACE_NAMES": [entry["name"] for entry in surface_data],
        "SURFACE_SEED_LABELS": [float(entry["seed_label"]) for entry in surface_data],
        "SURFACE_TARGET_VOLUMES": [float(entry["target_volume"]) for entry in surface_data],
        "FINAL_SURFACE_VOLUMES": [float(value) for value in final_surface_volumes],
        "FINAL_SURFACE_IOTAS": [float(value) for value in final_surface_iotas],
        "SURFACE_SELF_INTERSECTING": [bool(value) for value in run_status["self_intersections"]],
        "ADJACENT_SURFACE_GAPS": [float(value) for value in run_status["adjacent_gaps"]],
        "OUTER_VESSEL_GAP": None
        if run_status["outer_vessel_gap"] is None
        else float(run_status["outer_vessel_gap"]),
        "SURFACES_NESTED": bool(run_status["nesting_ok"]),
        "BAD_NESTING_PHIS": [float(value) for value in run_status["bad_nesting_phis"]],
        "INITIAL_SURFACE_VOLUMES": [float(value) for value in initial_surface_volumes],
        "INITIAL_SURFACE_IOTAS": [float(value) for value in initial_surface_iotas],
    }
