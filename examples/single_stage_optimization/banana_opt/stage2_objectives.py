import numpy as np

from alm_utils import (
    ALMSettings,
    augmented_inequality_objective,
    lower_bound_residual,
    upper_bound_residual,
    zero_gradient_like,
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
    alm_result,
    alm_taylor_result,
    final_volume,
    field_error,
    intersecting,
    final_max_curvature,
    final_coil_length,
    final_curve_curve_min_dist,
    hardware_status,
):
    alm_enabled = constraint_method == "alm"
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
        "basin_hops": args.basin_hops,
        "basin_stepsize": args.basin_stepsize if args.basin_hops > 0 else None,
        "basin_temperature": args.basin_temperature if args.basin_hops > 0 else None,
        "basin_niter_success": (
            args.basin_niter_success
            if args.basin_hops > 0 and args.basin_niter_success > 0
            else None
        ),
        "basin_seed": basin_seed if args.basin_hops > 0 else None,
        "basin_iterations": basin_iterations,
        "basin_minimization_failures": basin_minimization_failures,
        "basin_accepted_hops": basin_accepted_hops,
        "basin_rejected_hops": basin_rejected_hops,
        "basin_best_objective": basin_best_objective,
        "basin_accept_test_rejections": basin_accept_test_rejections,
        "basin_accept_test_triggered": basin_accept_test_triggered,
        "ALM_MAX_OUTER_ITERS": args.alm_max_outer_iters if alm_enabled else None,
        "ALM_MAX_SUBPROBLEM_CONTINUATIONS": (
            args.alm_max_subproblem_continuations if alm_enabled else None
        ),
        "ALM_OUTER_ITERATIONS": getattr(alm_result, "outer_iterations", None),
        "ALM_PENALTY_INIT": args.alm_penalty_init if alm_enabled else None,
        "ALM_PENALTY_SCALE": args.alm_penalty_scale if alm_enabled else None,
        "ALM_FEAS_TOL": args.alm_feas_tol if alm_enabled else None,
        "ALM_STATIONARITY_TOL": args.alm_stationarity_tol if alm_enabled else None,
        "ALM_TRUST_RADIUS_INIT": args.alm_trust_radius_init if alm_enabled else None,
        "ALM_TRUST_RADIUS_MIN": args.alm_trust_radius_min if alm_enabled else None,
        "ALM_TRUST_RADIUS_SHRINK": args.alm_trust_radius_shrink if alm_enabled else None,
        "ALM_TRUST_RADIUS_GROW": args.alm_trust_radius_grow if alm_enabled else None,
        "ALM_MAX_INNER_ATTEMPTS": args.alm_max_inner_attempts if alm_enabled else None,
        "ALM_DISTANCE_SMOOTHING": args.alm_distance_smoothing if alm_enabled else None,
        "ALM_CURVATURE_SMOOTHING": args.alm_curvature_smoothing if alm_enabled else None,
        "ALM_TAYLOR_TEST_ENABLED": args.alm_taylor_test if alm_enabled else None,
        "ALM_TAYLOR_TEST_SEED": args.alm_taylor_test_seed if alm_enabled else None,
        "ALM_TAYLOR_RESULT": alm_taylor_result,
        "ALM_FINAL_PENALTY": getattr(alm_result, "penalty", None),
        "ALM_FINAL_MULTIPLIERS": getattr(alm_result, "multipliers", None),
        "ALM_FINAL_CONSTRAINT_VALUES": getattr(alm_result, "constraint_values", None),
        "ALM_FINAL_SOLVER_CONSTRAINT_VALUES": getattr(
            alm_result,
            "solver_constraint_values",
            None,
        ),
        "ALM_FINAL_TRUST_RADIUS": getattr(alm_result, "trust_radius", None),
        "ALM_HISTORY": getattr(alm_result, "history", None),
        "FINAL_VOLUME": float(final_volume),
        "FIELD_ERROR": float(field_error),
        "SELF_INTERSECTING": intersecting,
        "MAX_CURVATURE": final_max_curvature,
        "COIL_LENGTH": final_coil_length,
        "CURVE_CURVE_MIN_DIST": final_curve_curve_min_dist,
        "HARDWARE_CONSTRAINTS_OK": hardware_status["success"],
        "HARDWARE_CONSTRAINT_VIOLATIONS": hardware_status["violations"],
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
):
    violations = []
    if coil_length > length_target:
        violations.append(
            f"coil_length {coil_length:.6f} exceeds target {length_target:.6f}"
        )
    if curve_curve_min_dist < cc_threshold:
        violations.append(
            f"coil_coil_min_dist {curve_curve_min_dist:.6f} below threshold {cc_threshold:.6f}"
        )
    if max_curvature > curvature_threshold:
        violations.append(
            f"max_curvature {max_curvature:.6f} exceeds threshold {curvature_threshold:.6f}"
        )
    return {
        "success": len(violations) == 0,
        "violations": violations,
        "coil_length": float(coil_length),
        "length_target": float(length_target),
        "curve_curve_min_dist": float(curve_curve_min_dist),
        "cc_threshold": float(cc_threshold),
        "max_curvature": float(max_curvature),
        "curvature_threshold": float(curvature_threshold),
    }


def stage2_constraint_activity_tolerances(
    distance_smoothing: float,
    curvature_smoothing: float,
    *,
    length_tolerance: float = 1e-3,
    banana_current_tolerance: float = 1e-3,
):
    return [
        float(length_tolerance),
        max(4.0 * float(distance_smoothing), _SMOOTHING_EPS),
        max(4.0 * float(curvature_smoothing), _SMOOTHING_EPS),
        float(banana_current_tolerance),
    ]


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
    return float(minimum_distance) - smooth_min, grad


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

    evaluation = augmented_inequality_objective(
        base_value,
        base_grad,
        [
            coil_length - length_target,
            curve_curve_signed_value,
            curvature_signed_value,
            banana_current_signed_value,
        ],
        [length_grad, curve_curve_grad, curvature_grad, banana_current_grad],
        multipliers,
        penalty,
    )
    evaluation.update(
        {
            "base_value": base_value,
            "constraint_names": [
                "coil_length_upper_bound",
                "coil_coil_spacing",
                "max_curvature",
                "banana_current_upper_bound",
            ],
            "dual_update_values": [
                coil_length - length_target,
                curve_curve_signed_value,
                curvature_signed_value,
                banana_current_signed_value,
            ],
            "constraint_grads": [
                length_grad,
                curve_curve_grad,
                curvature_grad,
                banana_current_grad,
            ],
            "constraint_activity_tolerances": stage2_constraint_activity_tolerances(
                distance_smoothing,
                curvature_smoothing,
            ),
            "feasibility_values": [
                length_violation,
                curve_curve_violation,
                curvature_violation,
                banana_current_violation,
            ],
            "max_feasibility_violation": max(
                length_violation,
                curve_curve_violation,
                curvature_violation,
                banana_current_violation,
            ),
        }
    )

    unitn = new_surf.unitnormal()
    BdotN = np.mean(np.abs(np.sum(new_bs.B().reshape(unitn.shape) * unitn, axis=2)))
    outstr = (
        f"ALM J={evaluation['total']:.1e}, Jflux={base_value:.1e}, "
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
    banana_current_grad = np.asarray(
        banana_current.vjp(banana_current_sign)(base_objective_optimizable),
        dtype=float,
    )
    return (
        banana_current_abs_A,
        banana_current_violation,
        banana_current_signed_value,
        banana_current_grad,
    )
