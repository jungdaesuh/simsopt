import os

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from simsopt.field import (
    LevelsetStoppingCriterion,
    MaxRStoppingCriterion,
    MaxZStoppingCriterion,
    MinRStoppingCriterion,
    MinZStoppingCriterion,
    compute_fieldlines,
)
from simsopt.geo import SurfaceClassifier, SurfaceRZFourier

from topology_scorer import (
    midplane_seed_radii as _midplane_seed_radii,
    stop_reason_label as _topology_stop_reason,
    toroidal_angle as _topology_toroidal_angle,
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
):
    violations = []
    if curve_curve_min_dist < cc_dist:
        violations.append(
            f"coil_coil_min_dist {curve_curve_min_dist:.6f} below threshold {cc_dist:.6f}"
        )
    if curve_surface_min_dist < cs_dist:
        violations.append(
            f"coil_surface_min_dist {curve_surface_min_dist:.6f} below threshold {cs_dist:.6f}"
        )
    if surface_vessel_min_dist < ss_dist:
        violations.append(
            f"surface_vessel_min_dist {surface_vessel_min_dist:.6f} below threshold {ss_dist:.6f}"
        )
    if max_curvature > curvature_threshold:
        violations.append(
            f"max_curvature {max_curvature:.6f} exceeds threshold {curvature_threshold:.6f}"
        )
    return {
        "success": len(violations) == 0,
        "violations": violations,
        "curve_curve_min_dist": float(curve_curve_min_dist),
        "cc_dist": float(cc_dist),
        "curve_surface_min_dist": float(curve_surface_min_dist),
        "cs_dist": float(cs_dist),
        "surface_vessel_min_dist": float(surface_vessel_min_dist),
        "ss_dist": float(ss_dist),
        "max_curvature": float(max_curvature),
        "curvature_threshold": float(curvature_threshold),
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
    status = evaluate_single_stage_hardware_constraints(
        curve_curve_min_dist,
        cc_dist,
        curve_surface_min_dist,
        cs_dist,
        surface_vessel_min_dist,
        ss_dist,
        max_curvature,
        curvature_threshold,
    )
    return {
        "curve_curve_min_dist": curve_curve_min_dist,
        "curve_surface_min_dist": curve_surface_min_dist,
        "surface_vessel_min_dist": surface_vessel_min_dist,
        "max_curvature": max_curvature,
        "status": status,
    }


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
    surface_classifier_factory=SurfaceClassifier,
    levelset_stopping_criterion_cls=LevelsetStoppingCriterion,
    max_z_stopping_criterion_cls=MaxZStoppingCriterion,
    min_z_stopping_criterion_cls=MinZStoppingCriterion,
    min_r_stopping_criterion_cls=MinRStoppingCriterion,
    max_r_stopping_criterion_cls=MaxRStoppingCriterion,
    compute_fieldlines_fn=compute_fieldlines,
    midplane_seed_radii_fn=_midplane_seed_radii,
    topology_stop_reason_fn=_topology_stop_reason,
    topology_toroidal_angle_fn=_topology_toroidal_angle,
):
    if nfieldlines <= 0:
        return disabled_topology_gate_status(tmax, tol, survival_threshold)

    cross_section = surface.cross_section(phi=0.0, thetas=512)
    r = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
    z = cross_section[:, 2]
    rmin = float(np.min(r))
    rmax = float(np.max(r))
    zmax = float(np.max(np.abs(z)))
    classifier = surface_classifier_factory(surface, h=0.03, p=2)
    stopping_criteria = [
        levelset_stopping_criterion_cls(classifier.dist),
        max_z_stopping_criterion_cls(zmax * 1.05),
        min_z_stopping_criterion_cls(-zmax * 1.05),
        min_r_stopping_criterion_cls(rmin * 0.95),
        max_r_stopping_criterion_cls(rmax * 1.05),
    ]
    stop_labels = [
        "surface_exit",
        "max_z_guardrail",
        "min_z_guardrail",
        "min_r_guardrail",
        "max_r_guardrail",
    ]
    R0 = midplane_seed_radii_fn(surface, nfieldlines)
    Z0 = np.zeros((nfieldlines,))
    _, fieldlines_phi_hits = compute_fieldlines_fn(
        bfield,
        R0,
        Z0,
        tmax=tmax,
        tol=tol,
        phis=[0.0],
        stopping_criteria=stopping_criteria,
    )

    survived = 0
    earliest_exit = None
    stop_reason_counts = {label: 0 for label in stop_labels}
    for hits in fieldlines_phi_hits:
        hits = np.asarray(hits)
        if hits.size == 0:
            survived += 1
            continue
        if hits.ndim == 1:
            hits = hits[None, :]
        negative_hits = hits[hits[:, 1] < 0]
        if negative_hits.size == 0:
            survived += 1
            continue
        first_stop = negative_hits[0]
        stop_index = int(-first_stop[1]) - 1
        stop_reason = topology_stop_reason_fn(stop_index, stop_labels)
        stop_reason_counts.setdefault(stop_reason, 0)
        stop_reason_counts[stop_reason] += 1
        exit_time = float(first_stop[0])
        exit_angle = topology_toroidal_angle_fn(first_stop[2], first_stop[3])
        if earliest_exit is None or exit_time < earliest_exit["first_exit_time"]:
            earliest_exit = {
                "first_exit_time": exit_time,
                "first_exit_angle": exit_angle,
                "stop_reason": stop_reason,
            }

    survival_fraction = survived / nfieldlines
    return {
        "enabled": True,
        "success": bool(survival_fraction >= survival_threshold),
        "nfieldlines": int(nfieldlines),
        "survived_lines": int(survived),
        "survival_fraction": float(survival_fraction),
        "survival_threshold": float(survival_threshold),
        "tmax": float(tmax),
        "tol": float(tol),
        "stop_reason_counts": stop_reason_counts,
        "first_exit_time": None
        if earliest_exit is None
        else earliest_exit["first_exit_time"],
        "first_exit_angle": None
        if earliest_exit is None
        else earliest_exit["first_exit_angle"],
        "first_exit_reason": None
        if earliest_exit is None
        else earliest_exit["stop_reason"],
    }


def disabled_topology_gate_status(tmax, tol, survival_threshold):
    return {
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
    }


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
