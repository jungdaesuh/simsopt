import numpy as np

from alm_utils import augmented_inequality_objective
from banana_opt.frontier_constraints import annotate_search_evaluation_finiteness
from banana_opt.frontier_scalarization import (
    FRONTIER_REFERENCE_MODE_ACHIEVEMENT,
    FRONTIER_REFERENCE_MODE_EPSILON,
)
from banana_opt.hardware_constraint_schema import get_hardware_constraint_spec
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
    VOLUME_WEIGHT,
    JVolume,
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
    if JVolume is not None:
        objective = objective + VOLUME_WEIGHT * JVolume
    if JSurfSurf is not None:
        objective = objective + SURF_DIST_WEIGHT * JSurfSurf
    return objective


def _surface_objective_pair(surface_weights, nonQSs, brs):
    J_QS_obj = average_surface_objectives(nonQSs, weights=surface_weights)
    J_Boozer_obj = average_surface_objectives(brs, weights=surface_weights)
    return J_QS_obj, J_Boozer_obj


def _resolve_surface_objective_terms(
    surface_weights,
    nonQSs,
    brs,
    *,
    JNonQSObjective=None,
    JBoozerObjective=None,
):
    raw_J_QS_obj, raw_J_Boozer_obj = _surface_objective_pair(surface_weights, nonQSs, brs)
    objective_J_QS_obj = raw_J_QS_obj if JNonQSObjective is None else JNonQSObjective
    objective_J_Boozer_obj = raw_J_Boozer_obj if JBoozerObjective is None else JBoozerObjective
    return raw_J_QS_obj, raw_J_Boozer_obj, objective_J_QS_obj, objective_J_Boozer_obj


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


def augment_frontier_metric_state(
    objective_eval,
    *,
    surface_iota_term,
    surface_volume_term,
    objective_optimizable=None,
):
    annotated = dict(objective_eval)
    annotated["J_iota_metric"] = float(surface_iota_term.J())
    annotated["dJ_iota_metric"] = _objective_gradient(
        surface_iota_term,
        objective_optimizable,
    )
    if surface_volume_term is None:
        annotated["J_volume_metric"] = 0.0
        annotated["dJ_volume_metric"] = np.zeros_like(
            np.asarray(annotated["grad"], dtype=float)
        )
    else:
        annotated["J_volume_metric"] = float(surface_volume_term.J())
        annotated["dJ_volume_metric"] = _objective_gradient(
            surface_volume_term,
            objective_optimizable,
        )
    return annotated


def apply_frontier_scalarization_override(
    objective_eval,
    *,
    enabled,
    frontier_goal_config,
    surface_iota_term,
    surface_volume_term,
    effective_res_weight,
    effective_iotas_weight,
    effective_volume_weight,
    length_weight,
    cc_weight,
    cs_weight,
    curvature_weight,
    surf_dist_weight,
    objective_optimizable=None,
    alm_formulation="weighted_sum",
    alm_multipliers=None,
    alm_penalty=None,
):
    if not enabled or frontier_goal_config is None:
        return dict(objective_eval)
    annotated = augment_frontier_metric_state(
        objective_eval,
        surface_iota_term=surface_iota_term,
        surface_volume_term=surface_volume_term,
        objective_optimizable=objective_optimizable,
    )
    annotated["frontier_scalarization_type"] = frontier_goal_config.scalarization_type
    frontier_goal_total, frontier_goal_grad = _frontier_goal_component_total_grad(
        annotated,
        effective_res_weight=effective_res_weight,
        effective_iotas_weight=effective_iotas_weight,
        effective_volume_weight=effective_volume_weight,
    )
    replacement_total = float(frontier_goal_total)
    replacement_grad = np.asarray(frontier_goal_grad, dtype=float)

    if (
        frontier_goal_config.scalarization_type
        == FRONTIER_REFERENCE_MODE_ACHIEVEMENT
    ):
        chebyshev_eval = _frontier_chebyshev_goal(annotated, frontier_goal_config)
        replacement_total = float(chebyshev_eval["frontier_scalarization_total"])
        replacement_grad = np.asarray(
            chebyshev_eval["frontier_scalarization_grad"],
            dtype=float,
        )
        annotated.update(chebyshev_eval)
    elif (
        frontier_goal_config.scalarization_type
        == FRONTIER_REFERENCE_MODE_EPSILON
    ):
        epsilon_penalties = _frontier_epsilon_penalties(
            annotated,
            frontier_goal_config=frontier_goal_config,
        )
        epsilon_penalty_total = float(
            sum(entry["penalty"] for entry in epsilon_penalties.values())
        )
        epsilon_penalty_grad = sum(
            (
                np.asarray(entry["grad"], dtype=float)
                for entry in epsilon_penalties.values()
            ),
            np.zeros_like(replacement_grad),
        )
        replacement_total += epsilon_penalty_total
        replacement_grad = replacement_grad + epsilon_penalty_grad
        annotated["frontier_epsilon_penalty"] = float(epsilon_penalty_total)
        annotated["frontier_epsilon_constraints"] = {
            metric_name: {
                "threshold": entry["threshold"],
                "excess": entry["excess"],
                "excess_ratio": entry["excess_ratio"],
                "penalty": entry["penalty"],
            }
            for metric_name, entry in epsilon_penalties.items()
        }

    annotated["frontier_goal_total"] = float(replacement_total)
    annotated["frontier_goal_grad"] = np.asarray(replacement_grad, dtype=float)

    if alm_formulation == "weighted_sum":
        if "constraint_values" in annotated and "constraint_grads" in annotated:
            if alm_multipliers is None or alm_penalty is None:
                raise ValueError(
                    "ALM frontier scalarization override requires explicit multipliers and penalty"
                )
            base_total, base_grad = _frontier_alm_base_total_grad(
                annotated,
                length_weight=length_weight,
            )
            base_total += float(replacement_total)
            base_grad = base_grad + replacement_grad
            alm_eval = augmented_inequality_objective(
                base_total,
                base_grad,
                annotated["constraint_values"],
                annotated["constraint_grads"],
                np.asarray(alm_multipliers, dtype=float),
                float(alm_penalty),
            )
            annotated.update(alm_eval)
            annotated["physics_total"] = float(base_total)
            annotated["base_total"] = float(base_total)
        else:
            penalty_total, penalty_grad = _frontier_penalty_geometry_total_grad(
                annotated,
                length_weight=length_weight,
                cc_weight=cc_weight,
                cs_weight=cs_weight,
                curvature_weight=curvature_weight,
                surf_dist_weight=surf_dist_weight,
            )
            annotated["total"] = penalty_total + float(replacement_total)
            annotated["grad"] = penalty_grad + replacement_grad
    return annotated


def _frontier_goal_component_total_grad(
    objective_eval,
    *,
    effective_res_weight,
    effective_iotas_weight,
    effective_volume_weight,
):
    total = (
        float(objective_eval["J_QS_objective"])
        + float(effective_res_weight) * float(objective_eval["J_Boozer_objective"])
        + float(effective_iotas_weight) * float(objective_eval["J_iota"])
        + float(effective_volume_weight) * float(objective_eval.get("J_volume", 0.0))
    )
    grad = (
        np.asarray(objective_eval["dJ_QS_objective"], dtype=float)
        + float(effective_res_weight)
        * np.asarray(objective_eval["dJ_Boozer_objective"], dtype=float)
        + float(effective_iotas_weight)
        * np.asarray(objective_eval["dJ_iota"], dtype=float)
        + float(effective_volume_weight)
        * np.asarray(objective_eval.get("dJ_volume", 0.0), dtype=float)
    )
    return float(total), grad


def _frontier_penalty_geometry_total_grad(
    objective_eval,
    *,
    length_weight,
    cc_weight,
    cs_weight,
    curvature_weight,
    surf_dist_weight,
):
    total = (
        float(length_weight) * float(objective_eval["J_len"])
        + float(cc_weight) * float(objective_eval["J_cc"])
        + float(cs_weight) * float(objective_eval["J_cs"])
        + float(curvature_weight) * float(objective_eval["J_curvature"])
        + float(surf_dist_weight) * float(objective_eval.get("J_surf", 0.0))
    )
    grad = (
        float(length_weight) * np.asarray(objective_eval["dJ_len"], dtype=float)
        + float(cc_weight) * np.asarray(objective_eval["dJ_cc"], dtype=float)
        + float(cs_weight) * np.asarray(objective_eval["dJ_cs"], dtype=float)
        + float(curvature_weight)
        * np.asarray(objective_eval["dJ_curvature"], dtype=float)
        + float(surf_dist_weight)
        * np.asarray(objective_eval.get("dJ_surf", 0.0), dtype=float)
    )
    return float(total), grad


def _frontier_alm_base_total_grad(
    objective_eval,
    *,
    length_weight,
):
    total = float(length_weight) * float(objective_eval["J_len"])
    grad = float(length_weight) * np.asarray(objective_eval["dJ_len"], dtype=float)
    return float(total), grad


def _frontier_chebyshev_goal(objective_eval, frontier_goal_config):
    deltas = np.asarray(
        [
            frontier_goal_config.chebyshev_weight_iota
            * (
                (frontier_goal_config.iota_reference - float(objective_eval["J_iota_metric"]))
                / frontier_goal_config.iota_scale
            ),
            frontier_goal_config.chebyshev_weight_volume
            * (
                (frontier_goal_config.volume_reference - float(objective_eval["J_volume_metric"]))
                / frontier_goal_config.volume_scale
            ),
            frontier_goal_config.chebyshev_weight_qa
            * (
                (float(objective_eval["J_QS"]) - frontier_goal_config.qs_reference)
                / frontier_goal_config.qs_reference
            ),
            frontier_goal_config.chebyshev_weight_boozer
            * (
                (float(objective_eval["J_Boozer"]) - frontier_goal_config.boozer_reference)
                / frontier_goal_config.boozer_reference
            ),
        ],
        dtype=float,
    )
    sharpness = float(frontier_goal_config.chebyshev_sharpness)
    max_delta = float(np.max(deltas))
    exp_shifted = np.exp(sharpness * (deltas - max_delta))
    sum_exp = float(np.sum(exp_shifted))
    softmax_weights = exp_shifted / sum_exp
    chebyshev_total = (
        max_delta
        + np.log(sum_exp) / sharpness
        + frontier_goal_config.chebyshev_rho * float(np.sum(deltas))
    )
    directional_grads = np.stack([
        -frontier_goal_config.chebyshev_weight_iota
        * np.asarray(objective_eval["dJ_iota_metric"], dtype=float)
        / frontier_goal_config.iota_scale,
        -frontier_goal_config.chebyshev_weight_volume
        * np.asarray(objective_eval["dJ_volume_metric"], dtype=float)
        / frontier_goal_config.volume_scale,
        frontier_goal_config.chebyshev_weight_qa
        * np.asarray(objective_eval["dJ_QS"], dtype=float)
        / frontier_goal_config.qs_reference,
        frontier_goal_config.chebyshev_weight_boozer
        * np.asarray(objective_eval["dJ_Boozer"], dtype=float)
        / frontier_goal_config.boozer_reference,
    ])
    coeffs = softmax_weights + frontier_goal_config.chebyshev_rho
    chebyshev_grad = (coeffs[:, None] * directional_grads).sum(axis=0)
    return {
        "frontier_scalarization_total": float(chebyshev_total),
        "frontier_scalarization_grad": chebyshev_grad,
        "frontier_chebyshev_deltas": deltas.tolist(),
        "frontier_chebyshev_softmax_weights": softmax_weights.tolist(),
    }


def _frontier_epsilon_penalties(
    objective_eval,
    *,
    frontier_goal_config,
):
    epsilon_penalties: dict[str, dict[str, object]] = {}
    if frontier_goal_config.epsilon_constraint_qa_max is not None:
        epsilon_penalties["qa_error"] = _frontier_excess_penalty(
            objective_eval["J_QS"],
            objective_eval["dJ_QS"],
            threshold=frontier_goal_config.epsilon_constraint_qa_max,
            scale=max(frontier_goal_config.qs_reference, 1.0e-6),
            penalty_weight=frontier_goal_config.epsilon_penalty_weight,
        )
    if frontier_goal_config.epsilon_constraint_boozer_max is not None:
        epsilon_penalties["boozer_residual"] = _frontier_excess_penalty(
            objective_eval["J_Boozer"],
            objective_eval["dJ_Boozer"],
            threshold=frontier_goal_config.epsilon_constraint_boozer_max,
            scale=max(frontier_goal_config.boozer_reference, 1.0e-6),
            penalty_weight=frontier_goal_config.epsilon_penalty_weight,
        )
    return epsilon_penalties


def _frontier_excess_penalty(
    value,
    grad,
    *,
    threshold,
    scale,
    penalty_weight,
):
    excess = max(float(value) - float(threshold), 0.0)
    excess_ratio = excess / float(scale)
    penalty = float(penalty_weight) * float(excess_ratio**2)
    penalty_grad = (
        np.zeros_like(np.asarray(grad, dtype=float))
        if excess <= 0.0
        else (
            float(penalty_weight)
            * 2.0
            * excess_ratio
            / float(scale)
            * np.asarray(grad, dtype=float)
        )
    )
    return {
        "enabled": True,
        "threshold": float(threshold),
        "scale": float(scale),
        "excess": float(excess),
        "excess_ratio": float(excess_ratio),
        "penalty": float(penalty),
        "grad": np.asarray(penalty_grad, dtype=float),
    }


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
    JNonQSObjective=None,
    JBoozerObjective=None,
    JVolume=None,
    VOLUME_WEIGHT=0.0,
    objective_optimizable=None,
    include_diagnostics=True,
):
    (
        raw_J_QS_obj,
        raw_J_Boozer_obj,
        objective_J_QS_obj,
        objective_J_Boozer_obj,
    ) = _resolve_surface_objective_terms(
        surface_weights,
        nonQSs,
        brs,
        JNonQSObjective=JNonQSObjective,
        JBoozerObjective=JBoozerObjective,
    )
    total_objective = build_total_objective(
        objective_J_QS_obj,
        RES_WEIGHT,
        objective_J_Boozer_obj,
        IOTAS_WEIGHT,
        Jiota,
        VOLUME_WEIGHT,
        JVolume,
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
    evaluation = {
        "total": float(total_objective.J()),
        "grad": total_grad,
        "surface_weights": np.asarray(surface_weights, dtype=float).copy(),
        "diagnostics_included": False,
    }
    if not include_diagnostics:
        return annotate_search_evaluation_finiteness(evaluation)

    volume_grad = (
        np.zeros_like(total_grad)
        if JVolume is None
        else _objective_gradient(JVolume, objective_optimizable)
    )
    evaluation.update({
        "diagnostics_included": True,
        "J_QS": float(raw_J_QS_obj.J()),
        "dJ_QS": _objective_gradient(raw_J_QS_obj, objective_optimizable),
        "J_QS_objective": float(objective_J_QS_obj.J()),
        "dJ_QS_objective": _objective_gradient(
            objective_J_QS_obj,
            objective_optimizable,
        ),
        "J_Boozer": float(raw_J_Boozer_obj.J()),
        "dJ_Boozer": _objective_gradient(raw_J_Boozer_obj, objective_optimizable),
        "J_Boozer_objective": float(objective_J_Boozer_obj.J()),
        "dJ_Boozer_objective": _objective_gradient(
            objective_J_Boozer_obj,
            objective_optimizable,
        ),
        "J_iota": float(Jiota.J()),
        "dJ_iota": _objective_gradient(Jiota, objective_optimizable),
        "J_volume": 0.0 if JVolume is None else float(JVolume.J()),
        "dJ_volume": volume_grad,
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
    })
    return annotate_search_evaluation_finiteness(evaluation)


def evaluate_base_objective(
    surface_weights,
    nonQSs,
    brs,
    RES_WEIGHT,
    Jiota,
    IOTAS_WEIGHT,
    JVolume,
    VOLUME_WEIGHT,
    JCurveLength,
    LENGTH_WEIGHT,
    *,
    objective_optimizable=None,
    alm_formulation="weighted_sum",
    _surface_pair=None,
    JNonQSObjective=None,
    JBoozerObjective=None,
    include_diagnostics=True,
):
    if _surface_pair is not None:
        raw_J_QS_obj, raw_J_Boozer_obj = _surface_pair
    else:
        raw_J_QS_obj, raw_J_Boozer_obj = _surface_objective_pair(surface_weights, nonQSs, brs)
    objective_J_QS_obj = raw_J_QS_obj if JNonQSObjective is None else JNonQSObjective
    objective_J_Boozer_obj = raw_J_Boozer_obj if JBoozerObjective is None else JBoozerObjective
    base_objective = (
        objective_J_QS_obj
        + RES_WEIGHT * objective_J_Boozer_obj
        + IOTAS_WEIGHT * Jiota
        + LENGTH_WEIGHT * JCurveLength
    )
    if JVolume is not None:
        base_objective = base_objective + VOLUME_WEIGHT * JVolume
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
    evaluation = {
        "total": total,
        "grad": grad,
        "physics_total": physics_total,
        "surface_weights": np.asarray(surface_weights, dtype=float).copy(),
        "diagnostics_included": False,
    }
    if not include_diagnostics:
        return annotate_search_evaluation_finiteness(evaluation)

    volume_grad = (
        np.zeros_like(base_grad)
        if JVolume is None
        else _objective_gradient(JVolume, objective_optimizable)
    )
    evaluation.update({
        "diagnostics_included": True,
        "J_QS": float(raw_J_QS_obj.J()),
        "dJ_QS": _objective_gradient(raw_J_QS_obj, objective_optimizable),
        "J_QS_objective": float(objective_J_QS_obj.J()),
        "dJ_QS_objective": _objective_gradient(objective_J_QS_obj, objective_optimizable),
        "J_Boozer": float(raw_J_Boozer_obj.J()),
        "dJ_Boozer": _objective_gradient(raw_J_Boozer_obj, objective_optimizable),
        "J_Boozer_objective": float(objective_J_Boozer_obj.J()),
        "dJ_Boozer_objective": _objective_gradient(objective_J_Boozer_obj, objective_optimizable),
        "J_iota": float(Jiota.J()),
        "dJ_iota": _objective_gradient(Jiota, objective_optimizable),
        "J_volume": 0.0 if JVolume is None else float(JVolume.J()),
        "dJ_volume": volume_grad,
        "J_len": float(JCurveLength.J()),
        "dJ_len": _objective_gradient(JCurveLength, objective_optimizable),
    })
    return annotate_search_evaluation_finiteness(evaluation)


def evaluate_alm_objective(
    surface_weights,
    nonQSs,
    brs,
    RES_WEIGHT,
    Jiota,
    IOTAS_WEIGHT,
    JVolume,
    VOLUME_WEIGHT,
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
    alm_formulation="weighted_sum",
    qs_threshold=None,
    boozer_threshold=None,
    iota_penalty_threshold=None,
    length_penalty_threshold=0.0,
    coil_length_objective=None,
    coil_length_threshold=None,
    banana_current=None,
    banana_current_threshold=None,
    JNonQSObjective=None,
    JBoozerObjective=None,
    include_diagnostics=True,
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
        JVolume,
        VOLUME_WEIGHT,
        JCurveLength,
        LENGTH_WEIGHT,
        objective_optimizable=objective_optimizable,
        alm_formulation=alm_formulation,
        _surface_pair=raw_surface_pair,
        JNonQSObjective=JNonQSObjective,
        JBoozerObjective=JBoozerObjective,
        include_diagnostics=include_diagnostics,
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

    hardware_constraints: dict[str, tuple[float, np.ndarray, float]] = {
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
        surface_surface_signed_value, surface_surface_grad, surface_surface_violation = (
            surface_surface_constraint_fn(
                outer_surface,
                vessel_surface,
                surface_surface_min_distance,
                distance_smoothing,
                objective_optimizable,
            )
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

    physics_constraints: dict[str, tuple[float, np.ndarray, float]] = {}
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

    active_constraint_names: list[str] = []
    constraint_values: list[float] = []
    constraint_grads: list[np.ndarray] = []
    feasibility_values: list[float] = []
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
    geometry_tolerances = np.asarray(
        activity_tolerances_fn(
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
    constraint_activity_tolerances = np.asarray(
        [
            constraint_tolerance_by_name.get(constraint_name, 0.0)
            for constraint_name in active_constraint_names
        ],
        dtype=float,
    )
    base_eval["constraint_activity_tolerances"] = constraint_activity_tolerances
    if include_diagnostics:
        base_eval["diagnostics_included"] = True
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
    return annotate_search_evaluation_finiteness(base_eval)
