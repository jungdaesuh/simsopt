import inspect

import numpy as np

from alm_utils import (
    ALMSettings,
    alm_result_diagnostics_fields,
    augmented_inequality_objective,
    lower_bound_residual,
    upper_bound_residual,
    zero_gradient_like,
)
from banana_opt.hardware_contracts import (
    TF_CURRENT_HARD_LIMIT_A,
    fixed_stage2_clearance_contract,
)
from banana_opt.hardware_constraint_schema import (
    build_hardware_constraint_artifact_payload_fields,
    build_hardware_constraint_status,
    build_threshold_overrides,
    hardware_constraint_alm_names,
)
from banana_opt.smoothing import smoothmax_selected, smoothmin_selected


_SMOOTHING_EPS = float(np.finfo(float).eps)


def _new_derivative():
    from simsopt._core.derivative import Derivative

    return Derivative({})


def build_stage2_alm_settings(args):
    return ALMSettings(
        max_outer_iterations=args.alm_max_outer_iters,
        max_subproblem_continuations=args.alm_max_subproblem_continuations,
        penalty_init=args.alm_penalty_init,
        penalty_scale=args.alm_penalty_scale,
        penalty_max=args.alm_penalty_max,
        feasibility_tol=args.alm_feas_tol,
        stationarity_tol=args.alm_stationarity_tol,
        trust_radius_init=(
            None if args.alm_trust_radius_init == 0.0 else args.alm_trust_radius_init
        ),
        trust_radius_min=args.alm_trust_radius_min,
        trust_radius_shrink=args.alm_trust_radius_shrink,
        trust_radius_grow=args.alm_trust_radius_grow,
        max_inner_attempts=args.alm_max_inner_attempts,
    )


def _build_stage2_artifact_hardware_snapshot(
    *,
    hardware_status,
    final_coil_length,
    length_target,
    final_curve_curve_min_dist,
    final_max_curvature,
    final_curve_surface_min_dist,
    plasma_vessel_min_dist,
    banana_current_A,
    banana_current_max_A,
    tf_current_A,
):
    return {
        "coil_length": final_coil_length,
        "length_target": length_target,
        "curve_curve_min_dist": final_curve_curve_min_dist,
        "max_curvature": final_max_curvature,
        "curve_surface_min_dist": final_curve_surface_min_dist,
        "surface_vessel_min_dist": plasma_vessel_min_dist,
        "banana_current_A": banana_current_A,
        "banana_current_max_A": banana_current_max_A,
        "tf_current_A": tf_current_A,
        "tf_current_limit_A": TF_CURRENT_HARD_LIMIT_A,
        "artifact_hardware_status": hardware_status,
    }


def _stage2_constraint_names(*, include_coil_surface: bool) -> tuple[str, ...]:
    requested_names = [
        "coil_length",
        "coil_coil_spacing",
        "max_curvature",
        "banana_current",
    ]
    if include_coil_surface:
        requested_names.insert(2, "coil_surface_spacing")
    return hardware_constraint_alm_names(names=tuple(requested_names))


def _legacy_stage2_constraint_names(*, include_coil_surface: bool) -> tuple[str, ...]:
    if include_coil_surface:
        return (
            "coil_length_upper_bound",
            "coil_coil_spacing",
            "coil_surface_spacing",
            "max_curvature",
            "banana_current_upper_bound",
        )
    return (
        "coil_length_upper_bound",
        "coil_coil_spacing",
        "max_curvature",
        "banana_current_upper_bound",
    )


def _ordered_constraint_values(
    constraint_names: tuple[str, ...],
    values_by_name: dict[str, object],
) -> list[object]:
    return [values_by_name[name] for name in constraint_names]


def build_stage2_results(
    *,
    args,
    plasma_surf_filename,
    file_loc,
    stage2_bs_path,
    tf_current_A,
    tf_current_sum_abs_A,
    num_tf_coils,
    initial_banana_current_A,
    banana_current_A,
    banana_to_tf_current_ratio,
    cc_threshold,
    cc_weight,
    curvature_weight,
    curvature_threshold,
    length_weight,
    constraint_method,
    theta_center,
    phi_center,
    theta_width,
    phi_width,
    length_target,
    major_radius,
    toroidal_flux,
    nfp,
    banana_surf_radius,
    order,
    max_iterations,
    iterations,
    termination_message,
    optimizer_success,
    basin_seed,
    basin_iterations,
    basin_minimization_failures,
    basin_accepted_hops,
    basin_rejected_hops,
    basin_best_objective,
    basin_accept_test_rejections,
    basin_accept_test_triggered,
    basin_nonfinite_rejections,
    basin_normalized_step_rejections,
    basin_completed_hops,
    basin_initial_objective,
    basin_best_hop_objective,
    basin_best_hop_index,
    basin_best_result_source,
    basin_objective_improvement,
    alm_result,
    alm_taylor_result,
    final_volume,
    field_error,
    intersecting,
    final_max_curvature,
    final_coil_length,
    final_curve_curve_min_dist,
    hardware_status,
    final_curve_surface_min_dist=None,
    plasma_vessel_min_dist=None,
):
    alm_enabled = constraint_method == "alm"
    basin_hops = int(getattr(args, "basin_hops", 0))
    hardware_snapshot = _build_stage2_artifact_hardware_snapshot(
        hardware_status=hardware_status,
        final_coil_length=final_coil_length,
        length_target=length_target,
        final_curve_curve_min_dist=final_curve_curve_min_dist,
        final_max_curvature=final_max_curvature,
        final_curve_surface_min_dist=final_curve_surface_min_dist,
        plasma_vessel_min_dist=plasma_vessel_min_dist,
        banana_current_A=banana_current_A,
        banana_current_max_A=float(args.banana_current_max_A),
        tf_current_A=tf_current_A,
    )
    return {
        "PLASMA_SURF_FILENAME": plasma_surf_filename,
        "PLASMA_SURF_PATH": file_loc,
        "STAGE2_BS_PATH": stage2_bs_path,
        "TF_CURRENT_A": float(tf_current_A),
        "TF_CURRENT_SUM_ABS_A": float(tf_current_sum_abs_A),
        "NUM_TF_COILS": int(num_tf_coils),
        "BANANA_INIT_CURRENT_A": float(initial_banana_current_A),
        "BANANA_CURRENT_MAX_A": float(args.banana_current_max_A),
        "BANANA_CURRENT_A": float(banana_current_A),
        "BANANA_TO_TF_CURRENT_RATIO": float(banana_to_tf_current_ratio),
        "CC_THRESHOLD": cc_threshold,
        "CC_WEIGHT": cc_weight,
        "CURVATURE_WEIGHT": curvature_weight,
        "CURVATURE_THRESHOLD": curvature_threshold,
        "LENGTH_WEIGHT": length_weight,
        **fixed_stage2_clearance_contract(),
        "CONSTRAINT_METHOD": constraint_method,
        "theta_center": theta_center,
        "phi_center": phi_center,
        "theta_width": theta_width,
        "phi_width": phi_width,
        "LENGTH_TARGET": length_target,
        "MAJOR_RADIUS": major_radius,
        "TOROIDAL_FLUX": toroidal_flux,
        "NFP": int(nfp),
        "banana_surf_radius": banana_surf_radius,
        "order": order,
        "init_only": args.init_only,
        "max_iterations": max_iterations,
        "iterations": iterations,
        "TERMINATION_MESSAGE": termination_message,
        "OPTIMIZER_SUCCESS": optimizer_success,
        "basin_hops": basin_hops,
        "basin_stepsize": getattr(args, "basin_stepsize", None) if basin_hops > 0 else None,
        "basin_temperature": (
            getattr(args, "basin_temperature", None) if basin_hops > 0 else None
        ),
        "basin_niter_success": (
            getattr(args, "basin_niter_success", None)
            if basin_hops > 0 and getattr(args, "basin_niter_success", 0) > 0
            else None
        ),
        "basin_seed": basin_seed if basin_hops > 0 else None,
        "basin_iterations": basin_iterations,
        "basin_minimization_failures": basin_minimization_failures,
        "basin_accepted_hops": basin_accepted_hops,
        "basin_rejected_hops": basin_rejected_hops,
        "basin_best_objective": basin_best_objective,
        "basin_accept_test_rejections": basin_accept_test_rejections,
        "basin_accept_test_triggered": basin_accept_test_triggered,
        "basin_nonfinite_rejections": basin_nonfinite_rejections,
        "basin_normalized_step_rejections": basin_normalized_step_rejections,
        "basin_completed_hops": basin_completed_hops,
        "basin_initial_objective": basin_initial_objective,
        "basin_best_hop_objective": basin_best_hop_objective,
        "basin_best_hop_index": basin_best_hop_index,
        "basin_best_result_source": basin_best_result_source,
        "basin_objective_improvement": basin_objective_improvement,
        "ALM_MAX_OUTER_ITERS": args.alm_max_outer_iters if alm_enabled else None,
        "ALM_MAX_SUBPROBLEM_CONTINUATIONS": (
            args.alm_max_subproblem_continuations if alm_enabled else None
        ),
        "ALM_OUTER_ITERATIONS": getattr(alm_result, "outer_iterations", None),
        "ALM_PENALTY_INIT": args.alm_penalty_init if alm_enabled else None,
        "ALM_PENALTY_SCALE": args.alm_penalty_scale if alm_enabled else None,
        "ALM_PENALTY_MAX": args.alm_penalty_max if alm_enabled else None,
        "ALM_FEAS_TOL": args.alm_feas_tol if alm_enabled else None,
        "ALM_STATIONARITY_TOL": args.alm_stationarity_tol if alm_enabled else None,
        "ALM_TRUST_RADIUS_INIT": args.alm_trust_radius_init if alm_enabled else None,
        "ALM_TRUST_RADIUS_MIN": args.alm_trust_radius_min if alm_enabled else None,
        "ALM_TRUST_RADIUS_SHRINK": args.alm_trust_radius_shrink if alm_enabled else None,
        "ALM_TRUST_RADIUS_GROW": args.alm_trust_radius_grow if alm_enabled else None,
        "ALM_MAX_INNER_ATTEMPTS": args.alm_max_inner_attempts if alm_enabled else None,
        "ALM_DISTANCE_SMOOTHING": args.alm_distance_smoothing if alm_enabled else None,
        "ALM_CURVATURE_SMOOTHING": args.alm_curvature_smoothing if alm_enabled else None,
        "ALM_TAYLOR_TEST_ENABLED": (
            getattr(args, "alm_taylor_test", None) if alm_enabled else None
        ),
        "ALM_TAYLOR_TEST_SEED": (
            getattr(args, "alm_taylor_test_seed", None) if alm_enabled else None
        ),
        "ALM_TAYLOR_RESULT": alm_taylor_result,
        "ALM_TERMINATION_REASON": getattr(alm_result, "termination_reason", None),
        "ALM_CONVERGED": getattr(alm_result, "converged_to_tolerances", None),
        "ALM_RESTORED_BEST_FEASIBLE": getattr(alm_result, "restored_best_feasible", None),
        "ALM_RESTORED_BEST_FEASIBLE_REASON": getattr(
            alm_result,
            "restored_best_feasible_reason",
            None,
        ),
        "ALM_INNER_OPTIMIZER_SUCCESS": getattr(alm_result, "optimizer_success", None),
        "ALM_INNER_OPTIMIZER_MESSAGE": getattr(alm_result, "optimizer_message", None),
        "ALM_FINAL_MAX_FEASIBILITY_VIOLATION": getattr(
            alm_result,
            "final_max_feasibility_violation",
            None,
        ),
        "ALM_FINAL_STATIONARITY_NORM": getattr(alm_result, "final_stationarity_norm", None),
        "ALM_FINAL_RAW_STATIONARITY_NORM": getattr(
            alm_result,
            "final_raw_stationarity_norm",
            None,
        ),
        "ALM_FINAL_KKT_STATIONARITY_NORM": getattr(
            alm_result,
            "final_kkt_stationarity_norm",
            None,
        ),
        "ALM_FINAL_FEASIBILITY_TOL": getattr(
            alm_result,
            "final_feasibility_tolerance",
            None,
        ),
        "ALM_FINAL_STATIONARITY_TOL": getattr(
            alm_result,
            "final_stationarity_tolerance",
            None,
        ),
        "ALM_FINAL_PENALTY": getattr(alm_result, "penalty", None),
        "ALM_FINAL_MULTIPLIERS": getattr(alm_result, "multipliers", None),
        "ALM_FINAL_CONSTRAINT_VALUES": getattr(alm_result, "constraint_values", None),
        "ALM_FINAL_SOLVER_CONSTRAINT_VALUES": getattr(
            alm_result,
            "solver_constraint_values",
            None,
        ),
        "ALM_FINAL_HARD_SIGNED_CONSTRAINT_VALUES": getattr(
            alm_result,
            "hard_signed_constraint_values",
            None,
        ),
        "ALM_FINAL_HARD_VIOLATION_VALUES": getattr(
            alm_result,
            "hard_violation_values",
            None,
        ),
        "ALM_FINAL_SURROGATE_SIGNED_CONSTRAINT_VALUES": getattr(
            alm_result,
            "surrogate_signed_constraint_values",
            None,
        ),
        "ALM_FINAL_HARD_MAX_VIOLATION": getattr(
            alm_result,
            "final_hard_max_violation",
            None,
        ),
        "ALM_FINAL_SURROGATE_MAX_VALUE": getattr(
            alm_result,
            "final_surrogate_max_value",
            None,
        ),
        "ALM_FINAL_HARD_POSITIVE_SHIFT_ZERO": getattr(
            alm_result,
            "hard_positive_shift_zero",
            None,
        ),
        "ALM_FINAL_SIGNAL_MISMATCH_ACTIVE": getattr(
            alm_result,
            "signal_mismatch_active",
            None,
        ),
        "ALM_FINAL_PENALTY_GRADIENT_NORM": getattr(
            alm_result,
            "final_penalty_gradient_norm",
            None,
        ),
        "ALM_FINAL_TRUST_RADIUS": getattr(alm_result, "trust_radius", None),
        **alm_result_diagnostics_fields(alm_result),
        "ALM_HISTORY": getattr(alm_result, "history", None),
        "FINAL_VOLUME": float(final_volume),
        "FIELD_ERROR": float(field_error),
        "SELF_INTERSECTING": intersecting,
        **build_hardware_constraint_artifact_payload_fields(hardware_snapshot),
    }


def make_stage2_fun(JF, new_bs, new_surf, Jf, Jls, Jccdist, Jc):
    def fun(dofs):
        JF.x = dofs
        J = JF.J()
        grad = JF.dJ()
        unitn = new_surf.unitnormal()
        BdotN = np.mean(np.abs(np.sum(new_bs.B().reshape(unitn.shape) * unitn, axis=2)))
        outstr = f"J={J:.1e}, Jf={Jf.J():.1e}, ⟨B·n⟩={BdotN:.1e}"
        outstr += f", Len={Jls.J():.1f}m"
        outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}m"
        outstr += f", Curvature={Jc.J():.2f}"
        outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
        print(outstr)
        return J, grad

    return fun


def evaluate_stage2_hardware_constraints(
    coil_length,
    length_target,
    curve_curve_min_dist,
    cc_threshold,
    max_curvature,
    curvature_threshold,
    curve_surface_min_dist=None,
    coil_surface_threshold=None,
    plasma_vessel_min_dist=None,
    plasma_vessel_threshold=None,
    banana_current_A=None,
    banana_current_threshold=None,
    tf_current_A=None,
    tf_current_threshold=None,
):
    threshold_overrides = build_threshold_overrides(
        (
            ("coil_length", length_target),
            ("coil_coil_spacing", cc_threshold),
            ("max_curvature", curvature_threshold),
            ("coil_surface_spacing", coil_surface_threshold),
            ("surface_vessel_spacing", plasma_vessel_threshold),
            ("banana_current", banana_current_threshold),
            ("tf_current", tf_current_threshold),
        )
    )
    measured_values = {
        "coil_length": coil_length,
        "coil_coil_spacing": curve_curve_min_dist,
        "max_curvature": max_curvature,
        "coil_surface_spacing": curve_surface_min_dist,
        "surface_vessel_spacing": plasma_vessel_min_dist,
        "banana_current": banana_current_A,
        "tf_current": tf_current_A,
    }
    status = build_hardware_constraint_status(
        measured_values,
        applies_to="artifact",
        threshold_overrides=threshold_overrides,
    )
    status.update(
        {
            "coil_length": float(coil_length),
            "length_target": float(length_target),
            "curve_curve_min_dist": float(curve_curve_min_dist),
            "cc_threshold": float(cc_threshold),
            "max_curvature": float(max_curvature),
            "curvature_threshold": float(curvature_threshold),
        }
    )
    if curve_surface_min_dist is not None and coil_surface_threshold is not None:
        status["curve_surface_min_dist"] = float(curve_surface_min_dist)
        status["coil_surface_threshold"] = float(coil_surface_threshold)
    if plasma_vessel_min_dist is not None and plasma_vessel_threshold is not None:
        status["plasma_vessel_min_dist"] = float(plasma_vessel_min_dist)
        status["plasma_vessel_threshold"] = float(plasma_vessel_threshold)
    if banana_current_A is not None and banana_current_threshold is not None:
        status["banana_current_A"] = float(banana_current_A)
        status["banana_current_threshold"] = float(banana_current_threshold)
    if tf_current_A is not None and tf_current_threshold is not None:
        status["tf_current_A"] = float(tf_current_A)
        status["tf_current_threshold"] = float(tf_current_threshold)
    return status


def stage2_constraint_activity_tolerances(
    distance_smoothing: float,
    curvature_smoothing: float,
    *,
    length_tolerance: float = 1e-3,
    banana_current_tolerance: float = 1e-3,
    include_coil_surface: bool = False,
):
    tolerances = [
        length_tolerance,
        max(4.0 * float(distance_smoothing), _SMOOTHING_EPS),
        max(4.0 * float(curvature_smoothing), _SMOOTHING_EPS),
        banana_current_tolerance,
    ]
    if include_coil_surface:
        return [
            tolerances[0],
            tolerances[1],
            tolerances[1],
            tolerances[2],
            tolerances[3],
        ]
    return tolerances


def resolve_stage2_constraint_activity_tolerances(
    stage2_constraint_activity_tolerances_fn,
    distance_smoothing: float,
    curvature_smoothing: float,
    *,
    include_coil_surface: bool,
):
    parameters = inspect.signature(stage2_constraint_activity_tolerances_fn).parameters
    if "include_coil_surface" in parameters:
        raw_tolerances = stage2_constraint_activity_tolerances_fn(
            distance_smoothing,
            curvature_smoothing,
            include_coil_surface=include_coil_surface,
        )
    else:
        raw_tolerances = stage2_constraint_activity_tolerances_fn(
            distance_smoothing,
            curvature_smoothing,
        )
    tolerance_values = [float(value) for value in raw_tolerances]
    constraint_names = _legacy_stage2_constraint_names(
        include_coil_surface=include_coil_surface,
    )
    if len(tolerance_values) != len(constraint_names):
        raise ValueError(
            "Stage 2 activity tolerance helper returned "
            f"{len(tolerance_values)} values for {len(constraint_names)} constraints."
        )
    return {
        name: value
        for name, value in zip(constraint_names, tolerance_values)
    }


def _sanitize_stage2_alm_inputs(
    base_value,
    base_grad,
    constraint_values,
    constraint_grads,
):
    invalid_fields: list[str] = []

    sanitized_base_grad = np.asarray(base_grad, dtype=float)
    if not np.all(np.isfinite(sanitized_base_grad)):
        invalid_fields.append("base_grad")
        sanitized_base_grad = zero_gradient_like(sanitized_base_grad)

    sanitized_base_value = float(base_value)
    if not np.isfinite(sanitized_base_value):
        invalid_fields.append("base_value")
        sanitized_base_value = max(float(np.linalg.norm(sanitized_base_grad)), 1.0)

    sanitized_constraint_values = []
    for index, constraint_value in enumerate(constraint_values):
        scalar_value = float(constraint_value)
        if not np.isfinite(scalar_value):
            invalid_fields.append(f"constraint_values[{index}]")
            scalar_value = 1.0
        sanitized_constraint_values.append(float(scalar_value))

    sanitized_constraint_grads = []
    for index, constraint_grad in enumerate(constraint_grads):
        grad_array = np.asarray(constraint_grad, dtype=float)
        if (
            grad_array.shape != sanitized_base_grad.shape
            or not np.all(np.isfinite(grad_array))
        ):
            invalid_fields.append(f"constraint_grads[{index}]")
            grad_array = zero_gradient_like(sanitized_base_grad)
        sanitized_constraint_grads.append(grad_array)

    return (
        float(sanitized_base_value),
        sanitized_base_grad,
        sanitized_constraint_values,
        sanitized_constraint_grads,
        invalid_fields,
    )


def _sanitize_stage2_feasibility_values(
    feasibility_values,
    *,
    constraint_values,
    field_prefix: str = "feasibility_values",
) -> tuple[list[float], list[str]]:
    sanitized = []
    invalid_fields: list[str] = []
    for index, (feasibility_value, constraint_value) in enumerate(
        zip(feasibility_values, constraint_values)
    ):
        scalar_value = float(feasibility_value)
        if not np.isfinite(scalar_value):
            invalid_fields.append(f"{field_prefix}[{index}]")
            scalar_value = max(1.0, max(float(constraint_value), 0.0))
        sanitized.append(float(scalar_value))
    return sanitized, invalid_fields


def _sanitize_stage2_signal_values(
    values,
    *,
    fallback_values,
    field_prefix: str,
) -> tuple[list[float], list[str]]:
    sanitized = []
    invalid_fields: list[str] = []
    for index, (value, fallback_value) in enumerate(zip(values, fallback_values)):
        scalar_value = float(value)
        if not np.isfinite(scalar_value):
            invalid_fields.append(f"{field_prefix}[{index}]")
            scalar_value = float(fallback_value)
        sanitized.append(float(scalar_value))
    return sanitized, invalid_fields


def smooth_max_curvature_signed_constraint(
    curve,
    threshold: float,
    temperature: float,
    base_objective_optimizable,
):
    kappa = np.asarray(curve.kappa(), dtype=float)
    hard_max = float(np.max(kappa))
    active_mask = kappa >= (hard_max - 4.0 * float(temperature))
    if not np.any(active_mask):
        active_mask[np.argmax(kappa)] = True
    smooth_max, active_weights = smoothmax_selected(
        kappa[active_mask],
        temperature,
        _SMOOTHING_EPS,
    )
    full_weights = np.zeros_like(kappa)
    full_weights[active_mask] = active_weights
    grad = np.asarray(
        curve.dkappa_by_dcoeff_vjp(full_weights)(base_objective_optimizable),
        dtype=float,
    )
    return smooth_max - float(threshold), grad


def smooth_min_distance_signed_constraint(
    curves,
    minimum_distance: float,
    temperature: float,
    base_objective_optimizable,
):
    pair_blocks = []
    hard_min = np.inf
    for i, curve_i in enumerate(curves):
        gamma_i = np.asarray(curve_i.gamma(), dtype=float)
        for j in range(i):
            curve_j = curves[j]
            gamma_j = np.asarray(curve_j.gamma(), dtype=float)
            diffs = gamma_i[:, None, :] - gamma_j[None, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            hard_min = min(hard_min, float(np.min(dists)))
            pair_blocks.append((i, j, diffs, dists))

    if not pair_blocks:
        return float(minimum_distance), zero_gradient_like(base_objective_optimizable.x)

    selection_window = 4.0 * float(temperature)
    selected_distances = []
    selected_entries = []
    for i, j, diffs, dists in pair_blocks:
        mask = dists <= (hard_min + selection_window)
        if not np.any(mask):
            mask[np.unravel_index(np.argmin(dists), dists.shape)] = True
        rows, cols = np.nonzero(mask)
        selected_distances.append(dists[rows, cols])
        selected_entries.append((i, j, rows, cols, diffs[rows, cols], dists[rows, cols]))

    flat_distances = np.concatenate(selected_distances)
    smooth_min, flat_weights = smoothmin_selected(
        flat_distances,
        temperature,
        _SMOOTHING_EPS,
    )

    point_gradients = [np.zeros_like(np.asarray(curve.gamma(), dtype=float)) for curve in curves]
    offset = 0
    for i, j, rows, cols, diffs, distances in selected_entries:
        count = len(distances)
        local_weights = flat_weights[offset:offset + count]
        offset += count
        directions = diffs / np.maximum(distances[:, None], _SMOOTHING_EPS)
        np.add.at(point_gradients[i], rows, local_weights[:, None] * directions)
        np.add.at(point_gradients[j], cols, -local_weights[:, None] * directions)

    derivative = _new_derivative()
    for curve, point_gradient in zip(curves, point_gradients):
        if np.any(point_gradient):
            derivative += curve.dgamma_by_dcoeff_vjp(point_gradient)
    grad = np.asarray(derivative(base_objective_optimizable), dtype=float)
    # grad = d(smooth_min)/dx, but signed_value = min_dist - smooth_min,
    # so d(signed_value)/dx = -d(smooth_min)/dx = -grad.
    return float(minimum_distance) - smooth_min, -grad


def smooth_min_curve_surface_signed_constraint(
    curves,
    surface,
    minimum_distance: float,
    temperature: float,
    base_objective_optimizable,
):
    if not curves:
        return float(minimum_distance), zero_gradient_like(base_objective_optimizable.x)

    surface_points = np.asarray(surface.gamma(), dtype=float).reshape((-1, 3))
    curve_blocks = []
    hard_min = np.inf
    for curve_index, curve in enumerate(curves):
        gamma = np.asarray(curve.gamma(), dtype=float)
        diffs = gamma[:, None, :] - surface_points[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        hard_min = min(hard_min, float(np.min(dists)))
        curve_blocks.append((curve_index, diffs, dists))

    selection_window = 4.0 * float(temperature)
    selected_distances = []
    selected_entries = []
    for curve_index, diffs, dists in curve_blocks:
        mask = dists <= (hard_min + selection_window)
        if not np.any(mask):
            mask[np.unravel_index(np.argmin(dists), dists.shape)] = True
        rows, cols = np.nonzero(mask)
        selected_distances.append(dists[rows, cols])
        selected_entries.append((curve_index, rows, diffs[rows, cols], dists[rows, cols]))

    flat_distances = np.concatenate(selected_distances)
    smooth_min, flat_weights = smoothmin_selected(
        flat_distances,
        temperature,
        _SMOOTHING_EPS,
    )

    point_gradients = [np.zeros_like(np.asarray(curve.gamma(), dtype=float)) for curve in curves]
    offset = 0
    for curve_index, rows, diffs, distances in selected_entries:
        count = len(distances)
        local_weights = flat_weights[offset:offset + count]
        offset += count
        directions = diffs / np.maximum(distances[:, None], _SMOOTHING_EPS)
        np.add.at(point_gradients[curve_index], rows, local_weights[:, None] * directions)

    derivative = _new_derivative()
    for curve, point_gradient in zip(curves, point_gradients):
        if np.any(point_gradient):
            derivative += curve.dgamma_by_dcoeff_vjp(point_gradient)
    grad = np.asarray(derivative(base_objective_optimizable), dtype=float)
    # grad = d(smooth_min)/dx, but signed_value = min_dist - smooth_min,
    # so d(signed_value)/dx = -d(smooth_min)/dx = -grad.
    return float(minimum_distance) - smooth_min, -grad


def evaluate_stage2_alm_problem(
    dofs,
    base_objective,
    new_bs,
    new_surf,
    Jf,
    Jls,
    length_target,
    Jccdist,
    Jc,
    banana_current,
    banana_current_max_A,
    distance_smoothing,
    curvature_smoothing,
    multipliers,
    penalty,
    stage2_constraint_activity_tolerances,
    smooth_min_distance_signed_constraint,
    smooth_max_curvature_signed_constraint,
    Jcsdist=None,
    smooth_min_curve_surface_signed_constraint=None,
):
    base_objective.x = dofs
    base_value = float(base_objective.J())
    base_grad = np.asarray(base_objective.dJ(), dtype=float)
    base_objective_optimizable = base_objective

    coil_length = float(Jls.J())
    length_violation = upper_bound_residual(coil_length, length_target)
    length_grad = np.asarray(Jls.dJ(partials=True)(base_objective_optimizable), dtype=float)

    curve_curve_min_dist = float(Jccdist.shortest_distance())
    curve_curve_violation = lower_bound_residual(
        curve_curve_min_dist,
        Jccdist.minimum_distance,
    )
    curve_curve_signed_value, curve_curve_grad = smooth_min_distance_signed_constraint(
        Jccdist.curves,
        Jccdist.minimum_distance,
        distance_smoothing,
        base_objective_optimizable,
    )
    include_coil_surface = (
        Jcsdist is not None and smooth_min_curve_surface_signed_constraint is not None
    )
    if include_coil_surface:
        curve_surface_min_dist = float(Jcsdist.shortest_distance())
        curve_surface_violation = lower_bound_residual(
            curve_surface_min_dist,
            Jcsdist.minimum_distance,
        )
        curve_surface_signed_value, curve_surface_grad = (
            smooth_min_curve_surface_signed_constraint(
                Jcsdist.curves,
                Jcsdist.surface,
                Jcsdist.minimum_distance,
                distance_smoothing,
                base_objective_optimizable,
            )
        )

    max_curvature = float(np.max(Jc.curve.kappa()))
    curvature_violation = upper_bound_residual(max_curvature, Jc.threshold)
    curvature_signed_value, curvature_grad = smooth_max_curvature_signed_constraint(
        Jc.curve,
        Jc.threshold,
        curvature_smoothing,
        base_objective_optimizable,
    )

    (
        banana_current_abs_A,
        banana_current_violation,
        banana_current_signed_value,
        banana_current_grad,
    ) = evaluate_banana_current_upper_bound(
        banana_current,
        banana_current_max_A,
        base_objective_optimizable,
    )

    active_names = _stage2_constraint_names(include_coil_surface=include_coil_surface)
    hard_by_name = {
        "coil_length_upper_bound": coil_length - length_target,
        "coil_coil_spacing": Jccdist.minimum_distance - curve_curve_min_dist,
        "max_curvature": max_curvature - Jc.threshold,
        "banana_current_upper_bound": banana_current_signed_value,
    }
    surrogate_by_name = {
        "coil_length_upper_bound": coil_length - length_target,
        "coil_coil_spacing": curve_curve_signed_value,
        "max_curvature": curvature_signed_value,
        "banana_current_upper_bound": banana_current_signed_value,
    }
    grad_by_name = {
        "coil_length_upper_bound": length_grad,
        "coil_coil_spacing": curve_curve_grad,
        "max_curvature": curvature_grad,
        "banana_current_upper_bound": banana_current_grad,
    }
    feasibility_by_name = {
        "coil_length_upper_bound": length_violation,
        "coil_coil_spacing": curve_curve_violation,
        "max_curvature": curvature_violation,
        "banana_current_upper_bound": banana_current_violation,
    }
    if include_coil_surface:
        hard_by_name["coil_surface_spacing"] = (
            Jcsdist.minimum_distance - curve_surface_min_dist
        )
        surrogate_by_name["coil_surface_spacing"] = curve_surface_signed_value
        grad_by_name["coil_surface_spacing"] = curve_surface_grad
        feasibility_by_name["coil_surface_spacing"] = curve_surface_violation
    hard_signed_constraint_values = _ordered_constraint_values(active_names, hard_by_name)
    surrogate_signed_constraint_values = _ordered_constraint_values(
        active_names,
        surrogate_by_name,
    )
    constraint_grads = _ordered_constraint_values(active_names, grad_by_name)
    hard_violation_values = _ordered_constraint_values(active_names, feasibility_by_name)
    (
        sanitized_base_value,
        sanitized_base_grad,
        sanitized_surrogate_signed_constraint_values,
        sanitized_constraint_grads,
        sanitized_invalid_fields,
    ) = _sanitize_stage2_alm_inputs(
        base_value,
        base_grad,
        surrogate_signed_constraint_values,
        constraint_grads,
    )
    sanitized_hard_signed_constraint_values, invalid_hard_signed_fields = (
        _sanitize_stage2_signal_values(
            hard_signed_constraint_values,
            fallback_values=sanitized_surrogate_signed_constraint_values,
            field_prefix="hard_signed_constraint_values",
        )
    )
    sanitized_hard_violation_values, invalid_hard_violation_fields = (
        _sanitize_stage2_feasibility_values(
            hard_violation_values,
            constraint_values=sanitized_hard_signed_constraint_values,
            field_prefix="hard_violation_values",
        )
    )

    evaluation = augmented_inequality_objective(
        sanitized_base_value,
        sanitized_base_grad,
        sanitized_surrogate_signed_constraint_values,
        sanitized_constraint_grads,
        multipliers,
        penalty,
    )
    tolerance_by_name = resolve_stage2_constraint_activity_tolerances(
        stage2_constraint_activity_tolerances,
        distance_smoothing,
        curvature_smoothing,
        include_coil_surface=include_coil_surface,
    )
    invalid_fields = (
        sanitized_invalid_fields
        + invalid_hard_signed_fields
        + invalid_hard_violation_fields
    )
    evaluation.update(
        {
            "base_value": sanitized_base_value,
            "constraint_names": list(active_names),
            "dual_update_values": sanitized_surrogate_signed_constraint_values,
            "constraint_grads": sanitized_constraint_grads,
            "constraint_activity_tolerances": np.asarray(
                _ordered_constraint_values(active_names, tolerance_by_name),
                dtype=float,
            ),
            "feasibility_values": sanitized_hard_violation_values,
            "hard_signed_constraint_values": sanitized_hard_signed_constraint_values,
            "hard_violation_values": sanitized_hard_violation_values,
            "surrogate_signed_constraint_values": sanitized_surrogate_signed_constraint_values,
            "hard_dual_update_values": sanitized_hard_signed_constraint_values,
            "max_feasibility_violation": max(sanitized_hard_violation_values),
            "nonfinite_inputs_sanitized": bool(invalid_fields),
            "nonfinite_input_fields": invalid_fields,
        }
    )
    if invalid_fields:
        # Keep the diagnostic payload finite enough to inspect, but mark the
        # evaluation itself invalid so generic ALM rejection/salvage logic
        # handles it instead of accepting a fabricated finite sample.
        evaluation["total"] = float("nan")
        evaluation["nonfinite_evaluation"] = True
        evaluation["nonfinite_fields"] = list(invalid_fields)

    unitn = new_surf.unitnormal()
    BdotN = np.mean(np.abs(np.sum(new_bs.B().reshape(unitn.shape) * unitn, axis=2)))
    outstr = (
        f"ALM J={evaluation['total']:.1e}, Jflux={sanitized_base_value:.1e}, "
        f"Jf={Jf.J():.1e}, ⟨B·n⟩={BdotN:.1e}"
    )
    outstr += (
        f", Len={coil_length:.1f}m, Len+={length_violation:.2e}, "
        f"Leng={coil_length - length_target:.2e}"
    )
    outstr += (
        f", C-C-Sep={curve_curve_min_dist:.2f}m, CC+={curve_curve_violation:.2e}, "
        f"CCg={curve_curve_signed_value:.2e}"
    )
    if include_coil_surface:
        outstr += (
            f", C-S-Sep={curve_surface_min_dist:.2f}m, CS+={curve_surface_violation:.2e}, "
            f"CSg={curve_surface_signed_value:.2e}"
        )
    outstr += (
        f", Curvature={max_curvature:.2f}, Curv+={curvature_violation:.2e}, "
        f"Curvg={curvature_signed_value:.2e}"
    )
    outstr += (
        f", |BananaI|={banana_current_abs_A:.2f}A, BananaI+={banana_current_violation:.2e}, "
        f"BananaIg={banana_current_signed_value:.2e}"
    )
    outstr += f", ║∇L_A║={evaluation['stationarity_norm']:.1e}, μ={penalty:.1e}"
    print(outstr)
    return evaluation


def evaluate_banana_current_upper_bound(
    banana_current,
    banana_current_max_A,
    base_objective_optimizable,
):
    banana_current_A = float(banana_current.get_value())
    banana_current_abs_A = abs(banana_current_A)
    banana_current_violation = upper_bound_residual(
        banana_current_abs_A,
        banana_current_max_A,
    )
    banana_current_signed_value = banana_current_abs_A - float(banana_current_max_A)
    banana_current_sign = 1.0 if banana_current_A >= 0.0 else -1.0
    banana_current_cotangent = np.array([banana_current_sign], dtype=float)
    banana_current_grad = np.asarray(
        banana_current.vjp(banana_current_cotangent)(base_objective_optimizable),
        dtype=float,
    )
    return (
        banana_current_abs_A,
        banana_current_violation,
        banana_current_signed_value,
        banana_current_grad,
    )
