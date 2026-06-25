import numpy as np

from alm_utils import augmented_inequality_objective
from .hardware_constraint_schema import get_hardware_constraint_spec
from .single_stage_constraints import single_stage_constraint_activity_tolerances


def _annotate_search_evaluation_finiteness(evaluation):
    annotated = dict(evaluation)
    total_value = float(annotated["total"])
    grad = np.asarray(annotated["grad"], dtype=float)
    annotated["total_finite"] = bool(np.isfinite(total_value))
    annotated["grad_all_finite"] = bool(np.all(np.isfinite(grad)))
    annotated["all_finite"] = annotated["total_finite"] and annotated["grad_all_finite"]
    return annotated


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
    try:
        partial_gradient = objective.dJ(partials=True)
    except TypeError:
        return np.asarray(objective.dJ(), dtype=float)
    if callable(partial_gradient):
        return np.asarray(partial_gradient(objective_optimizable), dtype=float)
    return np.asarray(partial_gradient, dtype=float)


def _objective_upper_bound_constraint(objective, threshold, objective_optimizable):
    if threshold is None:
        raise ValueError(
            "thresholded_physics ALM formulation requires explicit objective thresholds"
        )
    signed_value = float(objective.J()) - float(threshold)
    grad = _objective_gradient(objective, objective_optimizable)
    return signed_value, grad, max(0.0, signed_value)


def _scalar_abs_upper_bound_constraint(optimizable, threshold, objective_optimizable):
    value = float(optimizable.get_value())
    signed_value = abs(value) - float(threshold)
    sign = 1.0 if value >= 0.0 else -1.0
    cotangent = np.array([sign], dtype=float)
    grad = np.asarray(
        optimizable.vjp(cotangent)(objective_optimizable),
        dtype=float,
    )
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
    objective_optimizable=None,
):
    raw_J_QS_obj, raw_J_Boozer_obj = _surface_objective_pair(
        surface_weights,
        nonQSs,
        brs,
    )
    total_objective = build_total_objective(
        raw_J_QS_obj,
        RES_WEIGHT,
        raw_J_Boozer_obj,
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
    total_grad = _objective_gradient(total_objective, objective_optimizable)
    return _annotate_search_evaluation_finiteness(
        {
            "total": float(total_objective.J()),
            "grad": total_grad,
            "J_QS": float(raw_J_QS_obj.J()),
            "dJ_QS": _objective_gradient(raw_J_QS_obj, objective_optimizable),
            "J_Boozer": float(raw_J_Boozer_obj.J()),
            "dJ_Boozer": _objective_gradient(raw_J_Boozer_obj, objective_optimizable),
            "J_iota": float(Jiota.J()),
            "dJ_iota": _objective_gradient(Jiota, objective_optimizable),
            "J_len": float(JCurveLength.J()),
            "dJ_len": _objective_gradient(JCurveLength, objective_optimizable),
            "J_cc": float(JCurveCurve.J()),
            "dJ_cc": _objective_gradient(JCurveCurve, objective_optimizable),
            "J_cs": float(JCurveSurface.J()),
            "dJ_cs": _objective_gradient(JCurveSurface, objective_optimizable),
            "J_surf": 0.0 if JSurfSurf is None else float(JSurfSurf.J()),
            "dJ_surf": (
                np.zeros_like(total_grad)
                if JSurfSurf is None
                else _objective_gradient(JSurfSurf, objective_optimizable)
            ),
            "J_curvature": float(JCurvature.J()),
            "dJ_curvature": _objective_gradient(JCurvature, objective_optimizable),
            "surface_weights": np.asarray(surface_weights, dtype=float).copy(),
        }
    )


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
    alm_formulation="weighted_sum",
    _surface_pair=None,
):
    if _surface_pair is not None:
        raw_J_QS_obj, raw_J_Boozer_obj = _surface_pair
    else:
        raw_J_QS_obj, raw_J_Boozer_obj = _surface_objective_pair(
            surface_weights,
            nonQSs,
            brs,
        )
    base_objective = (
        raw_J_QS_obj
        + RES_WEIGHT * raw_J_Boozer_obj
        + IOTAS_WEIGHT * Jiota
        + LENGTH_WEIGHT * JCurveLength
    )
    physics_total = float(base_objective.J())
    base_grad = _objective_gradient(base_objective, objective_optimizable)
    if alm_formulation == "thresholded_physics":
        total = 0.0
        grad = np.zeros_like(base_grad)
    elif alm_formulation == "weighted_sum":
        total = physics_total
        grad = base_grad
    else:
        raise ValueError(f"Unsupported ALM formulation {alm_formulation!r}")
    return _annotate_search_evaluation_finiteness(
        {
            "total": total,
            "grad": grad,
            "physics_total": physics_total,
            "J_QS": float(raw_J_QS_obj.J()),
            "dJ_QS": _objective_gradient(raw_J_QS_obj, objective_optimizable),
            "J_Boozer": float(raw_J_Boozer_obj.J()),
            "dJ_Boozer": _objective_gradient(raw_J_Boozer_obj, objective_optimizable),
            "J_iota": float(Jiota.J()),
            "dJ_iota": _objective_gradient(Jiota, objective_optimizable),
            "J_len": float(JCurveLength.J()),
            "dJ_len": _objective_gradient(JCurveLength, objective_optimizable),
            "surface_weights": np.asarray(surface_weights, dtype=float).copy(),
        }
    )


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
    alm_formulation="weighted_sum",
    qs_threshold=None,
    boozer_threshold=None,
    iota_penalty_threshold=None,
    length_penalty_threshold=0.0,
    coil_length_objective=None,
    coil_length_threshold=None,
    banana_current=None,
    banana_current_threshold=None,
):
    raw_surface_pair = _surface_objective_pair(surface_weights, nonQSs, brs)
    raw_J_QS_obj, raw_J_Boozer_obj = raw_surface_pair
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
        _surface_pair=raw_surface_pair,
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

    hardware_constraints = {
        "coil_coil_spacing": (
            curve_curve_signed_value,
            curve_curve_grad,
            curve_curve_violation,
        ),
        "coil_surface_spacing": (
            curve_surface_signed_value,
            curve_surface_grad,
            curve_surface_violation,
        ),
        "max_curvature": (
            curvature_signed_value,
            curvature_grad,
            curvature_violation,
        ),
    }
    if JSurfSurf is not None:
        if vessel_surface is None or surface_surface_constraint_fn is None:
            raise ValueError(
                "surface-surface ALM constraints require vessel_surface and "
                "surface_surface_constraint_fn"
            )
        (
            surface_surface_signed_value,
            surface_surface_grad,
            surface_surface_violation,
        ) = surface_surface_constraint_fn(
            outer_surface,
            vessel_surface,
            surface_surface_min_distance,
            distance_smoothing,
            objective_optimizable,
        )
        hardware_constraints["surface_vessel_spacing"] = (
            surface_surface_signed_value,
            surface_surface_grad,
            surface_surface_violation,
        )
    if coil_length_objective is not None and coil_length_threshold is not None:
        hardware_constraints["coil_length_upper_bound"] = _objective_upper_bound_constraint(
            coil_length_objective,
            coil_length_threshold,
            objective_optimizable,
        )
    if banana_current is not None and banana_current_threshold is not None:
        hardware_constraints["banana_current_upper_bound"] = (
            _scalar_abs_upper_bound_constraint(
                banana_current,
                banana_current_threshold,
                objective_optimizable,
            )
        )

    physics_constraints = {}
    if alm_formulation == "thresholded_physics":
        physics_constraints = {
            "qs_error": _objective_upper_bound_constraint(
                raw_J_QS_obj,
                qs_threshold,
                objective_optimizable,
            ),
            "boozer_residual": _objective_upper_bound_constraint(
                raw_J_Boozer_obj,
                boozer_threshold,
                objective_optimizable,
            ),
            "iota_penalty": _objective_upper_bound_constraint(
                Jiota,
                iota_penalty_threshold,
                objective_optimizable,
            ),
            "length_penalty": _objective_upper_bound_constraint(
                JCurveLength,
                length_penalty_threshold,
                objective_optimizable,
            ),
        }

    active_constraint_names = []
    constraint_values = []
    constraint_grads = []
    feasibility_values = []
    for constraint_name in constraint_names:
        if constraint_name in hardware_constraints:
            signed_value, grad, violation = hardware_constraints[constraint_name]
        elif constraint_name in physics_constraints:
            signed_value, grad, violation = physics_constraints[constraint_name]
        else:
            raise ValueError(f"Unknown ALM constraint name {constraint_name!r}.")
        active_constraint_names.append(constraint_name)
        constraint_values.append(signed_value)
        constraint_grads.append(grad)
        feasibility_values.append(violation)

    alm_eval = augmented_inequality_objective(
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
    geometry_tolerances = np.asarray(
        single_stage_constraint_activity_tolerances(
            distance_smoothing,
            curvature_smoothing,
            include_surface_surface=JSurfSurf is not None,
        ),
        dtype=float,
    )
    geometry_names = ["coil_coil_spacing", "coil_surface_spacing", "max_curvature"]
    if JSurfSurf is not None:
        geometry_names.append("surface_vessel_spacing")
    constraint_tolerance_by_name = {
        name: float(value)
        for name, value in zip(geometry_names, geometry_tolerances)
    }
    for exact_constraint_name in ("coil_length_upper_bound", "banana_current_upper_bound"):
        if exact_constraint_name in hardware_constraints:
            constraint_tolerance_by_name[exact_constraint_name] = 1.0e-3
    base_eval["constraint_activity_tolerances"] = np.asarray(
        [
            constraint_tolerance_by_name.get(constraint_name, 0.0)
            for constraint_name in active_constraint_names
        ],
        dtype=float,
    )
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
    if coil_length_objective is not None:
        coil_length_spec = get_hardware_constraint_spec("coil_length")
        base_eval["coil_length_upper_bound_threshold"] = (
            coil_length_spec.threshold
            if coil_length_threshold is None
            else float(coil_length_threshold)
        )
    if banana_current_threshold is not None:
        banana_current_spec = get_hardware_constraint_spec("banana_current")
        base_eval["banana_current_upper_bound_threshold"] = min(
            banana_current_spec.threshold,
            float(banana_current_threshold),
        )
    base_eval["alm_formulation"] = alm_formulation
    return _annotate_search_evaluation_finiteness(base_eval)
