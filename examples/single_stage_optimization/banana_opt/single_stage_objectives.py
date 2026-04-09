import numpy as np

from alm_utils import augmented_inequality_objective
from banana_opt.single_stage_constraints import single_stage_constraint_activity_tolerances


def average_surface_objectives(objectives, weights=None):
    if len(objectives) == 0:
        raise ValueError("Need at least one surface objective to average")
    if weights is None:
        weights = np.ones(len(objectives))
    if len(objectives) != len(weights):
        raise ValueError("Number of objectives and weights must match")
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        raise ValueError("Sum of weights must be positive")

    weighted_sum = None
    for weight, objective in zip(weights, objectives):
        weighted_objective = weight * objective
        weighted_sum = weighted_objective if weighted_sum is None else weighted_sum + weighted_objective
    return (1.0 / total_weight) * weighted_sum


def build_total_objective(
    JnonQSRatio,
    RES_WEIGHT,
    JBoozerResidual,
    IOTAS_WEIGHT,
    Jiota,
    LENGTH_WEIGHT,
    JCurveLength,
    CC_WEIGHT,
    JCurveCurve,
    CS_WEIGHT,
    JCurveSurface,
    CURVATURE_WEIGHT,
    JCurvature,
    SURF_DIST_WEIGHT=0.0,
    JSurfSurf=None,
):
    objective = (
        JnonQSRatio
        + RES_WEIGHT * JBoozerResidual
        + IOTAS_WEIGHT * Jiota
        + LENGTH_WEIGHT * JCurveLength
        + CC_WEIGHT * JCurveCurve
        + CS_WEIGHT * JCurveSurface
        + CURVATURE_WEIGHT * JCurvature
    )
    if JSurfSurf is not None:
        objective = objective + SURF_DIST_WEIGHT * JSurfSurf
    return objective


def _surface_objective_pair(surface_weights, nonQSs, brs):
    J_QS_obj = average_surface_objectives(nonQSs, weights=surface_weights)
    J_Boozer_obj = average_surface_objectives(brs, weights=surface_weights)
    return J_QS_obj, J_Boozer_obj


def _objective_gradient(objective, objective_optimizable=None):
    if objective_optimizable is None:
        return np.asarray(objective.dJ(), dtype=float)
    return np.asarray(objective.dJ(partials=True)(objective_optimizable), dtype=float)


def _objective_upper_bound_constraint(objective, threshold, objective_optimizable):
    if threshold is None:
        raise ValueError("Gil ALM formulation requires explicit objective thresholds")
    signed_value = float(objective.J()) - float(threshold)
    grad = _objective_gradient(objective, objective_optimizable)
    return signed_value, grad, max(0.0, signed_value)


def evaluate_total_objective(
    surface_weights,
    nonQSs,
    brs,
    RES_WEIGHT,
    Jiota,
    IOTAS_WEIGHT,
    JCurveLength,
    LENGTH_WEIGHT,
    JCurveCurve,
    CC_WEIGHT,
    JCurveSurface,
    CS_WEIGHT,
    JCurvature,
    CURVATURE_WEIGHT,
    JSurfSurf=None,
    SURF_DIST_WEIGHT=0.0,
):
    J_QS_obj, J_Boozer_obj = _surface_objective_pair(surface_weights, nonQSs, brs)
    total_objective = build_total_objective(
        J_QS_obj,
        RES_WEIGHT,
        J_Boozer_obj,
        IOTAS_WEIGHT,
        Jiota,
        LENGTH_WEIGHT,
        JCurveLength,
        CC_WEIGHT,
        JCurveCurve,
        CS_WEIGHT,
        JCurveSurface,
        CURVATURE_WEIGHT,
        JCurvature,
        SURF_DIST_WEIGHT=SURF_DIST_WEIGHT,
        JSurfSurf=JSurfSurf,
    )
    total_grad = np.asarray(total_objective.dJ(), dtype=float)
    return {
        "total": float(total_objective.J()),
        "grad": total_grad,
        "J_QS": float(J_QS_obj.J()),
        "dJ_QS": np.asarray(J_QS_obj.dJ(), dtype=float),
        "J_Boozer": float(J_Boozer_obj.J()),
        "dJ_Boozer": np.asarray(J_Boozer_obj.dJ(), dtype=float),
        "J_iota": float(Jiota.J()),
        "dJ_iota": np.asarray(Jiota.dJ(), dtype=float),
        "J_len": float(JCurveLength.J()),
        "dJ_len": np.asarray(JCurveLength.dJ(), dtype=float),
        "J_cc": float(JCurveCurve.J()),
        "dJ_cc": np.asarray(JCurveCurve.dJ(), dtype=float),
        "J_cs": float(JCurveSurface.J()),
        "dJ_cs": np.asarray(JCurveSurface.dJ(), dtype=float),
        "J_surf": 0.0 if JSurfSurf is None else float(JSurfSurf.J()),
        "dJ_surf": (
            np.zeros_like(total_grad)
            if JSurfSurf is None
            else np.asarray(JSurfSurf.dJ(), dtype=float)
        ),
        "J_curvature": float(JCurvature.J()),
        "dJ_curvature": np.asarray(JCurvature.dJ(), dtype=float),
        "surface_weights": np.asarray(surface_weights, dtype=float).copy(),
    }


def evaluate_base_objective(
    surface_weights,
    nonQSs,
    brs,
    RES_WEIGHT,
    Jiota,
    IOTAS_WEIGHT,
    JCurveLength,
    LENGTH_WEIGHT,
    *,
    objective_optimizable=None,
    alm_formulation="legacy",
    _surface_pair=None,
):
    if _surface_pair is not None:
        J_QS_obj, J_Boozer_obj = _surface_pair
    else:
        J_QS_obj, J_Boozer_obj = _surface_objective_pair(surface_weights, nonQSs, brs)
    base_objective = (
        J_QS_obj
        + RES_WEIGHT * J_Boozer_obj
        + IOTAS_WEIGHT * Jiota
        + LENGTH_WEIGHT * JCurveLength
    )
    physics_total = float(base_objective.J())
    base_grad = _objective_gradient(base_objective, objective_optimizable)
    if alm_formulation == "gil":
        total = 0.0
        grad = np.zeros_like(base_grad)
    elif alm_formulation == "legacy":
        total = physics_total
        grad = base_grad
    else:
        raise ValueError(f"Unsupported ALM formulation {alm_formulation!r}")
    return {
        "total": total,
        "grad": grad,
        "physics_total": physics_total,
        "J_QS": float(J_QS_obj.J()),
        "dJ_QS": np.asarray(J_QS_obj.dJ(), dtype=float),
        "J_Boozer": float(J_Boozer_obj.J()),
        "dJ_Boozer": np.asarray(J_Boozer_obj.dJ(), dtype=float),
        "J_iota": float(Jiota.J()),
        "dJ_iota": np.asarray(Jiota.dJ(), dtype=float),
        "J_len": float(JCurveLength.J()),
        "dJ_len": np.asarray(JCurveLength.dJ(), dtype=float),
        "surface_weights": np.asarray(surface_weights, dtype=float).copy(),
    }


def evaluate_alm_objective(
    surface_weights,
    nonQSs,
    brs,
    RES_WEIGHT,
    Jiota,
    IOTAS_WEIGHT,
    JCurveLength,
    LENGTH_WEIGHT,
    JCurveCurve,
    JCurveSurface,
    JCurvature,
    multipliers,
    penalty,
    *,
    objective_optimizable,
    curves,
    curve_curve_min_distance,
    outer_surface,
    curve_surface_min_distance,
    banana_curve,
    curvature_threshold,
    distance_smoothing,
    curvature_smoothing,
    constraint_names,
    curve_curve_constraint_fn,
    curve_surface_constraint_fn,
    curvature_constraint_fn,
    JSurfSurf=None,
    vessel_surface=None,
    surface_surface_min_distance=0.0,
    surface_surface_constraint_fn=None,
    augmented_inequality_objective_fn=augmented_inequality_objective,
    activity_tolerances_fn=single_stage_constraint_activity_tolerances,
    alm_formulation="legacy",
    qs_threshold=None,
    boozer_threshold=None,
    iota_penalty_threshold=None,
    length_penalty_threshold=0.0,
):
    surface_pair = _surface_objective_pair(surface_weights, nonQSs, brs)
    J_QS_obj, J_Boozer_obj = surface_pair
    base_eval = evaluate_base_objective(
        surface_weights,
        nonQSs,
        brs,
        RES_WEIGHT,
        Jiota,
        IOTAS_WEIGHT,
        JCurveLength,
        LENGTH_WEIGHT,
        objective_optimizable=objective_optimizable,
        alm_formulation=alm_formulation,
        _surface_pair=surface_pair,
    )

    curve_curve_signed_value, curve_curve_grad, curve_curve_violation = curve_curve_constraint_fn(
        curves,
        curve_curve_min_distance,
        distance_smoothing,
        objective_optimizable,
    )
    curve_surface_signed_value, curve_surface_grad, curve_surface_violation = (
        curve_surface_constraint_fn(
            curves,
            outer_surface,
            curve_surface_min_distance,
            distance_smoothing,
            objective_optimizable,
        )
    )
    curvature_signed_value, curvature_grad, curvature_violation = curvature_constraint_fn(
        banana_curve,
        curvature_threshold,
        curvature_smoothing,
        objective_optimizable,
    )

    active_constraint_names = list(constraint_names[:3])
    constraint_values = [
        curve_curve_signed_value,
        curve_surface_signed_value,
        curvature_signed_value,
    ]
    constraint_grads = [
        curve_curve_grad,
        curve_surface_grad,
        curvature_grad,
    ]
    feasibility_values = [
        curve_curve_violation,
        curve_surface_violation,
        curvature_violation,
    ]
    if JSurfSurf is not None:
        surface_surface_signed_value, surface_surface_grad, surface_surface_violation = (
            surface_surface_constraint_fn(
                outer_surface,
                vessel_surface,
                surface_surface_min_distance,
                distance_smoothing,
                objective_optimizable,
            )
        )
        active_constraint_names.append(constraint_names[3])
        constraint_values.append(surface_surface_signed_value)
        constraint_grads.append(surface_surface_grad)
        feasibility_values.append(surface_surface_violation)

    n_physics_constraints = 0
    if alm_formulation == "gil":
        physics_constraints = [
            _objective_upper_bound_constraint(
                J_QS_obj,
                qs_threshold,
                objective_optimizable,
            ),
            _objective_upper_bound_constraint(
                J_Boozer_obj,
                boozer_threshold,
                objective_optimizable,
            ),
            _objective_upper_bound_constraint(
                Jiota,
                iota_penalty_threshold,
                objective_optimizable,
            ),
            _objective_upper_bound_constraint(
                JCurveLength,
                length_penalty_threshold,
                objective_optimizable,
            ),
        ]
        n_physics_constraints = len(physics_constraints)
        if len(constraint_names) < len(active_constraint_names) + n_physics_constraints:
            raise ValueError(
                f"Gil ALM formulation requires {n_physics_constraints} additional "
                "physics constraint names"
            )
        for constraint_name, (signed_value, grad, violation) in zip(
            constraint_names[len(active_constraint_names):],
            physics_constraints,
        ):
            active_constraint_names.append(constraint_name)
            constraint_values.append(signed_value)
            constraint_grads.append(grad)
            feasibility_values.append(violation)

    alm_eval = augmented_inequality_objective_fn(
        base_eval["total"],
        base_eval["grad"],
        constraint_values,
        constraint_grads,
        multipliers,
        penalty,
    )
    base_total = float(base_eval["physics_total"])
    base_eval.update(alm_eval)
    base_eval["base_total"] = base_total
    base_eval["constraint_names"] = active_constraint_names
    base_eval["dual_update_values"] = np.asarray(constraint_values, dtype=float)
    base_eval["feasibility_values"] = np.asarray(feasibility_values, dtype=float)
    base_eval["max_feasibility_violation"] = float(max(feasibility_values))
    base_eval["constraint_grads"] = [np.asarray(grad, dtype=float) for grad in constraint_grads]
    constraint_activity_tolerances = np.asarray(
        activity_tolerances_fn(
            distance_smoothing,
            curvature_smoothing,
            include_surface_surface=JSurfSurf is not None,
        ),
        dtype=float,
    )
    if n_physics_constraints > 0:
        constraint_activity_tolerances = np.concatenate(
            [
                constraint_activity_tolerances,
                np.zeros(n_physics_constraints, dtype=float),
            ]
        )
    base_eval["constraint_activity_tolerances"] = constraint_activity_tolerances
    base_eval["J_cc"] = float(JCurveCurve.J())
    base_eval["dJ_cc"] = np.asarray(JCurveCurve.dJ(), dtype=float)
    base_eval["J_cs"] = float(JCurveSurface.J())
    base_eval["dJ_cs"] = np.asarray(JCurveSurface.dJ(), dtype=float)
    base_eval["J_surf"] = 0.0 if JSurfSurf is None else float(JSurfSurf.J())
    base_eval["dJ_surf"] = (
        np.zeros_like(np.asarray(base_eval["grad"], dtype=float))
        if JSurfSurf is None
        else np.asarray(JSurfSurf.dJ(), dtype=float)
    )
    base_eval["J_curvature"] = float(JCurvature.J())
    base_eval["dJ_curvature"] = np.asarray(JCurvature.dJ(), dtype=float)
    base_eval["alm_formulation"] = alm_formulation
    return base_eval
