import copy
import os
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from banana_opt.basin_hopping import _normalized_step_rms as basin_normalized_step_rms
from banana_opt.incumbents import (
    restore_single_stage_incumbent_state,
    snapshot_single_stage_incumbent_state,
)
from banana_opt.single_stage_geometry import (
    build_local_relative_bounds,
    build_scaled_local_outer_bounds,
    build_scaled_outer_bounds,
    build_scaled_outer_problem,
    build_scipy_bounds,
)

_PENALTY_FEASIBLE_START_LOCAL_MAXITER = int(
    os.environ.get("PENALTY_FEASIBLE_START_LOCAL_MAXITER", "5")
)
_PENALTY_FEASIBLE_START_LOCAL_RELATIVE_RADIUS = float(
    os.environ.get("PENALTY_FEASIBLE_START_LOCAL_RELATIVE_RADIUS", "0.05")
)
_PENALTY_FEASIBLE_START_LOCAL_MAX_ATTEMPTS = int(
    os.environ.get("PENALTY_FEASIBLE_START_LOCAL_MAX_ATTEMPTS", "3")
)
_PENALTY_FEASIBLE_START_LOCAL_RADIUS_SHRINK = float(
    os.environ.get("PENALTY_FEASIBLE_START_LOCAL_RADIUS_SHRINK", "0.5")
)
_PENALTY_FEASIBLE_START_REJECT_RADIUS_SHRINK = float(
    os.environ.get("PENALTY_FEASIBLE_START_REJECT_RADIUS_SHRINK", "0.25")
)
_PENALTY_FEASIBLE_START_SAFE_STEP_RMS_LIMIT = float(
    os.environ.get("PENALTY_FEASIBLE_START_SAFE_STEP_RMS_LIMIT", "0.02")
)
_PENALTY_FEASIBLE_START_PHASE2_RADIUS_SCALE = float(
    os.environ.get("PENALTY_FEASIBLE_START_PHASE2_RADIUS_SCALE", "0.5")
)
_PENALTY_FEASIBLE_START_MIN_ACCEPTED_STEP_RMS = 1.0e-6
_FRONTIER_FEASIBLE_START_PHASE1_SCALE = 0.05
_FRONTIER_FEASIBLE_START_LOCAL_RELATIVE_RADIUS = 0.01

_SEED_REGIME_PRESERVE_FIRST = "preserve_first"
_SEED_REGIME_REPAIR_FIRST = "repair_first"
_SEED_REGIME_BRIDGE_ONLY = "bridge_only"


@dataclass(frozen=True)
class Phase1Config:
    cc_weight: float = 0.0
    cs_weight: float = 0.0
    curvature_weight: float = 0.0
    surf_dist_weight: float = 0.0
    local_maxiter: int = _PENALTY_FEASIBLE_START_LOCAL_MAXITER
    local_relative_radius: float = _PENALTY_FEASIBLE_START_LOCAL_RELATIVE_RADIUS
    local_max_attempts: int = _PENALTY_FEASIBLE_START_LOCAL_MAX_ATTEMPTS
    local_radius_shrink: float = _PENALTY_FEASIBLE_START_LOCAL_RADIUS_SHRINK
    reject_radius_shrink: float = _PENALTY_FEASIBLE_START_REJECT_RADIUS_SHRINK
    safe_step_rms_limit: float = _PENALTY_FEASIBLE_START_SAFE_STEP_RMS_LIMIT
    phase2_radius_scale: float = _PENALTY_FEASIBLE_START_PHASE2_RADIUS_SCALE
    min_accepted_step_rms: float = _PENALTY_FEASIBLE_START_MIN_ACCEPTED_STEP_RMS
    frontier_phase1_scale: float = _FRONTIER_FEASIBLE_START_PHASE1_SCALE
    frontier_local_relative_radius: float = (
        _FRONTIER_FEASIBLE_START_LOCAL_RELATIVE_RADIUS
    )


def build_phase1_config(
    *,
    cc_weight=0.0,
    cs_weight=0.0,
    curvature_weight=0.0,
    surf_dist_weight=0.0,
    local_maxiter=_PENALTY_FEASIBLE_START_LOCAL_MAXITER,
    local_relative_radius=_PENALTY_FEASIBLE_START_LOCAL_RELATIVE_RADIUS,
    local_max_attempts=_PENALTY_FEASIBLE_START_LOCAL_MAX_ATTEMPTS,
    local_radius_shrink=_PENALTY_FEASIBLE_START_LOCAL_RADIUS_SHRINK,
    reject_radius_shrink=_PENALTY_FEASIBLE_START_REJECT_RADIUS_SHRINK,
    safe_step_rms_limit=_PENALTY_FEASIBLE_START_SAFE_STEP_RMS_LIMIT,
    phase2_radius_scale=_PENALTY_FEASIBLE_START_PHASE2_RADIUS_SCALE,
    min_accepted_step_rms=_PENALTY_FEASIBLE_START_MIN_ACCEPTED_STEP_RMS,
    frontier_phase1_scale=_FRONTIER_FEASIBLE_START_PHASE1_SCALE,
    frontier_local_relative_radius=_FRONTIER_FEASIBLE_START_LOCAL_RELATIVE_RADIUS,
):
    return Phase1Config(
        cc_weight=float(cc_weight),
        cs_weight=float(cs_weight),
        curvature_weight=float(curvature_weight),
        surf_dist_weight=float(surf_dist_weight),
        local_maxiter=int(local_maxiter),
        local_relative_radius=float(local_relative_radius),
        local_max_attempts=int(local_max_attempts),
        local_radius_shrink=float(local_radius_shrink),
        reject_radius_shrink=float(reject_radius_shrink),
        safe_step_rms_limit=float(safe_step_rms_limit),
        phase2_radius_scale=float(phase2_radius_scale),
        min_accepted_step_rms=float(min_accepted_step_rms),
        frontier_phase1_scale=float(frontier_phase1_scale),
        frontier_local_relative_radius=float(frontier_local_relative_radius),
    )


DEFAULT_PHASE1_CONFIG = build_phase1_config()


def resolve_initial_step_phase_maxiter(
    total_maxiter,
    initial_step_scale,
    initial_step_maxiter,
):
    if initial_step_maxiter <= 0:
        return 0
    if not (0.0 < initial_step_scale < 1.0):
        return 0
    return min(total_maxiter, initial_step_maxiter)


def penalty_seed_regime_uses_local_startup(seed_regime):
    return seed_regime in (
        _SEED_REGIME_REPAIR_FIRST,
        _SEED_REGIME_BRIDGE_ONLY,
    )


def resolve_penalty_phase1_settings(
    total_maxiter,
    initial_step_scale,
    initial_step_maxiter,
    *,
    phase1_config=DEFAULT_PHASE1_CONFIG,
    enable_local_preservation,
    seed_regime=_SEED_REGIME_PRESERVE_FIRST,
    is_frontier_mode=False,
):
    explicit_phase1_maxiter = max(int(initial_step_maxiter), 0)
    phase1_maxiter = resolve_initial_step_phase_maxiter(
        total_maxiter,
        initial_step_scale,
        explicit_phase1_maxiter,
    )
    baseline_local_relative_radius = float(phase1_config.local_relative_radius)
    auto_enabled = False
    phase1_scale = float(initial_step_scale)
    local_relative_radius = (
        baseline_local_relative_radius
        if enable_local_preservation and phase1_maxiter > 0
        else None
    )
    if enable_local_preservation and phase1_maxiter == 0 and total_maxiter > 0:
        fallback_maxiter = (
            explicit_phase1_maxiter
            if explicit_phase1_maxiter > 0
            else int(phase1_config.local_maxiter)
        )
        phase1_maxiter = min(total_maxiter, fallback_maxiter)
        phase1_scale = (
            float(phase1_config.frontier_phase1_scale) if is_frontier_mode else 1.0
        )
        local_relative_radius = (
            float(phase1_config.frontier_local_relative_radius)
            if is_frontier_mode
            else baseline_local_relative_radius
        )
        auto_enabled = True
    use_phase1 = phase1_maxiter > 0
    use_local_bounds = bool(enable_local_preservation and use_phase1)
    local_max_attempts = max(int(phase1_config.local_max_attempts), 1)
    if seed_regime == _SEED_REGIME_REPAIR_FIRST and use_local_bounds:
        local_max_attempts += 1
    return {
        "use_phase1": use_phase1,
        "phase1_maxiter": int(phase1_maxiter),
        "phase1_scale": float(phase1_scale),
        "auto_enabled": auto_enabled,
        "use_local_bounds": use_local_bounds,
        "local_relative_radius": local_relative_radius if use_local_bounds else None,
        "local_max_attempts": local_max_attempts if use_local_bounds else 1,
    }


def _build_penalty_phase1_result(
    *,
    used_phase1,
    phase1_iterations,
    phase1_termination_message,
    phase1_success,
    phase1_outcome,
    continue_search,
    next_dofs,
    local_preservation_used,
    local_preservation_preserved_start,
    local_preservation_attempts,
    local_preservation_radius,
    local_preservation_step_rms,
    phase1_first_accepted_step_rms,
    phase1_max_accepted_step_rms,
    phase1_anchor_restore_used,
    phase1_unsafe_accept_rollbacks,
    phase1_invalid_reject_attempts,
    startup_local_phase_regime,
    startup_local_recovery_achieved,
    bridge_local_donor_ready,
):
    return {
        "used_phase1": bool(used_phase1),
        "phase1_iterations": phase1_iterations,
        "phase1_termination_message": phase1_termination_message,
        "phase1_success": phase1_success,
        "phase1_outcome": phase1_outcome,
        "continue_search": bool(continue_search),
        "next_dofs": np.asarray(next_dofs, dtype=float).copy(),
        "local_preservation_used": bool(local_preservation_used),
        "local_preservation_preserved_start": bool(
            local_preservation_preserved_start
        ),
        "local_preservation_attempts": int(local_preservation_attempts),
        "local_preservation_radius": local_preservation_radius,
        "phase2_local_preservation_radius": local_preservation_radius,
        "local_preservation_step_rms": local_preservation_step_rms,
        "phase1_first_accepted_step_rms": phase1_first_accepted_step_rms,
        "phase1_max_accepted_step_rms": phase1_max_accepted_step_rms,
        "phase1_anchor_restore_used": bool(phase1_anchor_restore_used),
        "phase1_unsafe_accept_rollbacks": int(phase1_unsafe_accept_rollbacks),
        "phase1_invalid_reject_attempts": int(phase1_invalid_reject_attempts),
        "phase1_recovery_used": bool(
            phase1_anchor_restore_used or phase1_invalid_reject_attempts > 0
        ),
        "startup_local_phase_regime": startup_local_phase_regime,
        "startup_local_recovery_achieved": bool(startup_local_recovery_achieved),
        "bridge_local_donor_ready": bool(bridge_local_donor_ready),
    }


def snapshot_penalty_phase1_anchor(run_dict):
    return {
        "incumbent": snapshot_single_stage_incumbent_state(run_dict),
        "accepted_iterations": int(run_dict.get("accepted_iterations", 0)),
        "x_prev": np.asarray(
            run_dict.get("x_prev", run_dict["accepted_x"]),
            dtype=float,
        ).copy(),
        "intersecting": bool(run_dict.get("intersecting", False)),
        "accepted_boozer_stage": run_dict.get("accepted_boozer_stage"),
        "frontier_trust_status": copy.deepcopy(run_dict.get("frontier_trust_status")),
        "best_accepted_incumbent": copy.deepcopy(run_dict.get("best_accepted_incumbent")),
        "best_accepted_metric": run_dict.get("best_accepted_metric"),
        "best_accepted_stage": copy.deepcopy(run_dict.get("best_accepted_stage")),
        "best_feasible_incumbent": copy.deepcopy(run_dict.get("best_feasible_incumbent")),
        "best_feasible_metric": run_dict.get("best_feasible_metric"),
        "best_feasible_stage": copy.deepcopy(run_dict.get("best_feasible_stage")),
        "it": int(run_dict.get("it", 0)),
    }


def restore_penalty_phase1_anchor(run_dict, anchor_state):
    restore_single_stage_incumbent_state(run_dict, anchor_state["incumbent"])
    run_dict["accepted_iterations"] = int(anchor_state["accepted_iterations"])
    run_dict["x_prev"] = np.asarray(anchor_state["x_prev"], dtype=float).copy()
    run_dict["intersecting"] = bool(anchor_state["intersecting"])
    run_dict["accepted_boozer_stage"] = anchor_state["accepted_boozer_stage"]
    run_dict["frontier_trust_status"] = copy.deepcopy(
        anchor_state["frontier_trust_status"]
    )
    run_dict["best_accepted_incumbent"] = copy.deepcopy(
        anchor_state["best_accepted_incumbent"]
    )
    run_dict["best_accepted_metric"] = anchor_state["best_accepted_metric"]
    run_dict["best_accepted_stage"] = copy.deepcopy(anchor_state["best_accepted_stage"])
    run_dict["best_feasible_incumbent"] = copy.deepcopy(
        anchor_state["best_feasible_incumbent"]
    )
    run_dict["best_feasible_metric"] = anchor_state["best_feasible_metric"]
    run_dict["best_feasible_stage"] = copy.deepcopy(anchor_state["best_feasible_stage"])
    run_dict["it"] = int(anchor_state["it"])
    run_dict["trial_hardware_status"] = None


def resolve_penalty_phase2_local_radius(
    step_rms,
    *,
    local_radius,
    seed_regime,
    phase1_config=DEFAULT_PHASE1_CONFIG,
):
    if not np.isfinite(step_rms):
        return None
    radius_ceiling = (
        float(phase1_config.safe_step_rms_limit)
        if seed_regime == _SEED_REGIME_PRESERVE_FIRST
        else local_radius
    )
    if radius_ceiling is None or not np.isfinite(radius_ceiling):
        radius_ceiling = float(phase1_config.safe_step_rms_limit)
    base_radius = min(
        max(float(step_rms), float(phase1_config.min_accepted_step_rms)),
        float(radius_ceiling),
    )
    return max(
        float(phase1_config.min_accepted_step_rms),
        float(base_radius) * float(phase1_config.phase2_radius_scale),
    )


def _resolve_phase1_accept_outcome(
    *,
    seed_regime,
    donor_ready_local_accept,
    recovered_local_accept,
    repair_local_accept,
):
    if seed_regime == _SEED_REGIME_PRESERVE_FIRST:
        return donor_ready_local_accept, "safe_local_accept", recovered_local_accept
    if seed_regime == _SEED_REGIME_REPAIR_FIRST:
        return repair_local_accept, "repair_local_recovery", repair_local_accept
    if seed_regime == _SEED_REGIME_BRIDGE_ONLY:
        phase1_outcome = (
            "bridge_local_donor_ready"
            if donor_ready_local_accept
            else "bridge_local_recovery_only"
        )
        return recovered_local_accept, phase1_outcome, recovered_local_accept
    return False, "nonlocal_phase1_continue", recovered_local_accept


def evaluate_penalty_phase1_local_accept(
    anchor_x,
    run_dict,
    *,
    local_radius,
    seed_regime,
    anchor_repair_state,
    phase1_config=DEFAULT_PHASE1_CONFIG,
    refinement_eligible_fn,
    repair_progress_state_fn,
):
    accepted_x = np.asarray(run_dict["accepted_x"], dtype=float)
    step_rms = basin_normalized_step_rms(anchor_x, accepted_x)
    meaningful_step = bool(
        np.isfinite(step_rms)
        and step_rms > float(phase1_config.min_accepted_step_rms)
    )
    refinement_ready = refinement_eligible_fn(run_dict)
    repair_state = repair_progress_state_fn(run_dict)
    repair_state_improved = bool(repair_state < anchor_repair_state)
    within_local_radius = bool(
        local_radius is None
        or (
            np.isfinite(step_rms)
            and step_rms <= float(local_radius) + 1.0e-12
        )
    )
    recovered_local_accept = bool(
        refinement_ready and meaningful_step and within_local_radius
    )
    repair_local_accept = bool(
        repair_state_improved and meaningful_step and within_local_radius
    )
    donor_ready_local_accept = bool(
        recovered_local_accept
        and step_rms <= float(phase1_config.safe_step_rms_limit)
    )
    (
        phase1_graduated,
        phase1_outcome,
        reported_recovered_local_accept,
    ) = _resolve_phase1_accept_outcome(
        seed_regime=seed_regime,
        donor_ready_local_accept=donor_ready_local_accept,
        recovered_local_accept=recovered_local_accept,
        repair_local_accept=repair_local_accept,
    )
    return {
        "step_rms": float(step_rms),
        "meaningful_step": meaningful_step,
        "refinement_ready": refinement_ready,
        "within_local_radius": within_local_radius,
        "recovered_local_accept": reported_recovered_local_accept,
        "repair_state": repair_state,
        "repair_state_improved": repair_state_improved,
        "repair_local_accept": repair_local_accept,
        "donor_ready_local_accept": donor_ready_local_accept,
        "phase1_graduated": phase1_graduated,
        "phase1_outcome": phase1_outcome,
        "phase2_radius": (
            resolve_penalty_phase2_local_radius(
                step_rms,
                local_radius=local_radius,
                seed_regime=seed_regime,
                phase1_config=phase1_config,
            )
            if phase1_graduated
            else None
        ),
    }


def _build_penalty_phase1_problem(
    anchor_x,
    *,
    phase1_scale,
    use_local_bounds,
    local_radius,
    lower_bounds,
    upper_bounds,
    objective_fn,
    callback_fn,
):
    if phase1_scale < 1.0:
        phase1_fun, phase1_callback = build_scaled_outer_problem(
            objective_fn,
            callback_fn,
            anchor_x,
            phase1_scale,
        )
        x0 = np.zeros_like(anchor_x)
        bounds = (
            build_scaled_local_outer_bounds(
                anchor_x,
                phase1_scale,
                lower_bounds,
                upper_bounds,
                local_radius,
            )
            if use_local_bounds
            else build_scaled_outer_bounds(
                anchor_x,
                phase1_scale,
                lower_bounds,
                upper_bounds,
            )
        )
        return phase1_fun, phase1_callback, x0, bounds

    bounds = (
        build_local_relative_bounds(
            anchor_x,
            local_radius,
            lower_bounds,
            upper_bounds,
        )
        if use_local_bounds
        else build_scipy_bounds(lower_bounds, upper_bounds)
    )
    return objective_fn, callback_fn, anchor_x.copy(), bounds


def build_penalty_phase2_bounds(
    anchor_x,
    *,
    lower_bounds,
    upper_bounds,
    phase1_result,
):
    if (
        not phase1_result.get("local_preservation_used", False)
        or phase1_result.get("local_preservation_preserved_start", False)
    ):
        return build_scipy_bounds(lower_bounds, upper_bounds)
    local_radius = phase1_result.get(
        "phase2_local_preservation_radius",
        phase1_result.get("local_preservation_radius"),
    )
    return build_local_relative_bounds(
        anchor_x,
        local_radius,
        lower_bounds,
        upper_bounds,
    )


def _repair_phase1_total_grad(objective_eval, *, phase1_config):
    total = 0.0
    grad = np.zeros_like(np.asarray(objective_eval["grad"], dtype=float))
    used_term = False
    for value_key, grad_key, weight in (
        ("J_cc", "dJ_cc", phase1_config.cc_weight),
        ("J_cs", "dJ_cs", phase1_config.cs_weight),
        ("J_curvature", "dJ_curvature", phase1_config.curvature_weight),
        ("J_surf", "dJ_surf", phase1_config.surf_dist_weight),
    ):
        if value_key not in objective_eval or grad_key not in objective_eval:
            continue
        total += float(weight) * float(objective_eval[value_key])
        grad = grad + float(weight) * np.asarray(objective_eval[grad_key], dtype=float)
        used_term = True
    if used_term:
        return float(total), grad
    return float(objective_eval["total"]), np.asarray(objective_eval["grad"], dtype=float)


def _build_repair_phase1_objective(*, objective_fn, objective_eval_fn, seed_regime, phase1_config):
    if seed_regime != _SEED_REGIME_REPAIR_FIRST or objective_eval_fn is None:
        return objective_fn

    def phase1_objective(xk):
        evaluation = objective_eval_fn(xk)
        return _repair_phase1_total_grad(
            evaluation,
            phase1_config=phase1_config,
        )

    return phase1_objective


def _local_phase_failure_result(seed_regime):
    if seed_regime == _SEED_REGIME_PRESERVE_FIRST:
        return (
            "preserved_feasible_start_no_safe_local_step",
            "preserved_start_no_safe_step",
            True,
        )
    if seed_regime == _SEED_REGIME_BRIDGE_ONLY:
        return "bridge_only_no_local_donor", "bridge_only_no_local_donor", False
    if seed_regime == _SEED_REGIME_REPAIR_FIRST:
        return "repair_first_no_local_recovery", "repair_first_no_local_recovery", False
    return "local_phase1_no_progress", "local_phase1_no_progress", False


def run_penalty_phase1(
    dofs,
    *,
    total_maxiter,
    maxcor,
    ftol,
    gtol,
    initial_step_scale,
    initial_step_maxiter,
    enable_local_preservation,
    seed_regime=_SEED_REGIME_PRESERVE_FIRST,
    is_frontier_mode=False,
    lower_bounds,
    upper_bounds,
    run_dict,
    objective_fn,
    callback_fn,
    refinement_eligible_fn,
    repair_progress_state_fn,
    phase1_config=DEFAULT_PHASE1_CONFIG,
    objective_eval_fn=None,
    normalize_message_fn,
    restore_accepted_state_fn,
    refresh_preserved_timeout_artifacts_fn=None,
    minimize_fn=minimize,
):
    use_local_startup = bool(
        enable_local_preservation or penalty_seed_regime_uses_local_startup(seed_regime)
    )
    settings = resolve_penalty_phase1_settings(
        total_maxiter,
        initial_step_scale,
        initial_step_maxiter,
        phase1_config=phase1_config,
        enable_local_preservation=use_local_startup,
        seed_regime=seed_regime,
        is_frontier_mode=is_frontier_mode,
    )
    if not settings["use_phase1"]:
        return _build_penalty_phase1_result(
            used_phase1=False,
            phase1_iterations=None,
            phase1_termination_message=None,
            phase1_success=None,
            phase1_outcome="bypassed",
            continue_search=True,
            next_dofs=dofs,
            local_preservation_used=False,
            local_preservation_preserved_start=False,
            local_preservation_attempts=0,
            local_preservation_radius=None,
            local_preservation_step_rms=None,
            phase1_first_accepted_step_rms=None,
            phase1_max_accepted_step_rms=None,
            phase1_anchor_restore_used=False,
            phase1_unsafe_accept_rollbacks=0,
            phase1_invalid_reject_attempts=0,
            startup_local_phase_regime=None,
            startup_local_recovery_achieved=False,
            bridge_local_donor_ready=False,
        )

    phase1_iterations = 0
    phase1_messages = []
    remaining_maxiter = settings["phase1_maxiter"]
    local_attempts_used = 0
    local_radius = settings["local_relative_radius"]
    phase1_success = False
    phase1_first_accepted_step_rms = None
    phase1_max_accepted_step_rms = None
    phase1_anchor_restore_used = False
    phase1_unsafe_accept_rollbacks = 0
    phase1_invalid_reject_attempts = 0
    startup_local_recovery_achieved = False
    bridge_local_donor_ready = False

    while remaining_maxiter > 0:
        local_attempts_used += 1
        restore_accepted_state_fn()
        anchor_state = snapshot_penalty_phase1_anchor(run_dict)
        anchor_x = np.asarray(run_dict["accepted_x"], dtype=float).copy()
        anchor_repair_state = repair_progress_state_fn(run_dict)
        accepted_before_attempt = int(run_dict.get("accepted_iterations", 0))
        invalid_rejects_before_attempt = int(
            run_dict.get("invalid_state_rejects_total", 0)
        )
        phase1_scale = settings["phase1_scale"]

        def record_phase1_accepted_step_rms(accepted_step_rms):
            nonlocal phase1_first_accepted_step_rms, phase1_max_accepted_step_rms
            if phase1_first_accepted_step_rms is None:
                phase1_first_accepted_step_rms = accepted_step_rms
            phase1_max_accepted_step_rms = (
                accepted_step_rms
                if phase1_max_accepted_step_rms is None
                else max(
                    phase1_max_accepted_step_rms,
                    accepted_step_rms,
                )
            )

        def tracked_phase1_callback(xk):
            accepted_before_callback = int(run_dict.get("accepted_iterations", 0))
            phase1_callback(xk)
            if int(run_dict.get("accepted_iterations", 0)) > accepted_before_callback:
                record_phase1_accepted_step_rms(
                    basin_normalized_step_rms(
                        anchor_x,
                        np.asarray(run_dict["accepted_x"], dtype=float),
                    )
                )

        phase1_objective = _build_repair_phase1_objective(
            objective_fn=objective_fn,
            objective_eval_fn=objective_eval_fn,
            seed_regime=seed_regime,
            phase1_config=phase1_config,
        )

        phase1_fun, phase1_callback, x0, bounds = _build_penalty_phase1_problem(
            anchor_x,
            phase1_scale=phase1_scale,
            use_local_bounds=settings["use_local_bounds"],
            local_radius=local_radius,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            objective_fn=phase1_objective,
            callback_fn=callback_fn,
        )

        run_dict["phase1_repair_mode_active"] = (
            seed_regime == _SEED_REGIME_REPAIR_FIRST
        )
        try:
            last_result = minimize_fn(
                phase1_fun,
                x0,
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                callback=tracked_phase1_callback,
                options={
                    "maxiter": remaining_maxiter,
                    "maxcor": maxcor,
                    "ftol": ftol,
                    "gtol": gtol,
                },
            )
        finally:
            run_dict["phase1_repair_mode_active"] = False
        phase1_iterations += int(last_result.nit)
        remaining_maxiter = max(settings["phase1_maxiter"] - phase1_iterations, 0)
        phase1_success = bool(last_result.success)
        attempt_message = normalize_message_fn(
            last_result.message,
            success=phase1_success,
            status=getattr(last_result, "status", None),
            invalid_state_rejects_total=run_dict["invalid_state_rejects_total"],
            surface_solve_rejects=run_dict["surface_solve_rejects"],
            hardware_rejects=run_dict["hardware_rejects"],
            topology_gate_rejects=run_dict["topology_gate_rejects"],
        )
        phase1_messages.append(
            f"attempt{local_attempts_used}={attempt_message}"
            if settings["use_local_bounds"]
            else attempt_message
        )
        if int(run_dict.get("accepted_iterations", 0)) > accepted_before_attempt:
            accept_summary = evaluate_penalty_phase1_local_accept(
                anchor_x,
                run_dict,
                local_radius=local_radius,
                seed_regime=seed_regime,
                anchor_repair_state=anchor_repair_state,
                phase1_config=phase1_config,
                refinement_eligible_fn=refinement_eligible_fn,
                repair_progress_state_fn=repair_progress_state_fn,
            )
            startup_local_recovery_achieved = bool(
                accept_summary["recovered_local_accept"]
            )
            bridge_local_donor_ready = bool(
                accept_summary["donor_ready_local_accept"]
            )
            if accept_summary["phase1_graduated"]:
                return _build_penalty_phase1_result(
                    used_phase1=True,
                    phase1_iterations=phase1_iterations,
                    phase1_termination_message="; ".join(phase1_messages),
                    phase1_success=phase1_success,
                    phase1_outcome=(
                        f"{accept_summary['phase1_outcome']}_after_recovery"
                        if phase1_anchor_restore_used
                        else accept_summary["phase1_outcome"]
                    ),
                    continue_search=True,
                    next_dofs=run_dict["accepted_x"],
                    local_preservation_used=settings["use_local_bounds"],
                    local_preservation_preserved_start=False,
                    local_preservation_attempts=local_attempts_used,
                    local_preservation_radius=accept_summary["phase2_radius"],
                    local_preservation_step_rms=accept_summary["step_rms"],
                    phase1_first_accepted_step_rms=phase1_first_accepted_step_rms,
                    phase1_max_accepted_step_rms=phase1_max_accepted_step_rms,
                    phase1_anchor_restore_used=phase1_anchor_restore_used,
                    phase1_unsafe_accept_rollbacks=phase1_unsafe_accept_rollbacks,
                    phase1_invalid_reject_attempts=phase1_invalid_reject_attempts,
                    startup_local_phase_regime=(
                        seed_regime if settings["use_local_bounds"] else None
                    ),
                    startup_local_recovery_achieved=startup_local_recovery_achieved,
                    bridge_local_donor_ready=bridge_local_donor_ready,
                )
            phase1_messages.append(
                "unsafe_local_accept("
                f"step_rms={accept_summary['step_rms']:.3e}, "
                f"meaningful={accept_summary['meaningful_step']}, "
                f"refinement_ready={accept_summary['refinement_ready']}, "
                f"repair_state_improved={accept_summary['repair_state_improved']}, "
                f"within_local_radius={accept_summary['within_local_radius']}, "
                f"donor_ready={accept_summary['donor_ready_local_accept']})"
            )
            phase1_unsafe_accept_rollbacks += 1
            phase1_anchor_restore_used = True
            restore_penalty_phase1_anchor(run_dict, anchor_state)
            restore_accepted_state_fn()
            if refresh_preserved_timeout_artifacts_fn is not None:
                refresh_preserved_timeout_artifacts_fn()
        if not settings["use_local_bounds"]:
            break
        if local_attempts_used >= settings["local_max_attempts"]:
            break
        invalid_rejects_delta = int(run_dict.get("invalid_state_rejects_total", 0)) - (
            invalid_rejects_before_attempt
        )
        if invalid_rejects_delta > 0:
            phase1_invalid_reject_attempts += 1
        shrink_factor = (
            float(phase1_config.reject_radius_shrink)
            if invalid_rejects_delta > 0
            else float(phase1_config.local_radius_shrink)
        )
        local_radius *= shrink_factor

    restore_accepted_state_fn()
    if settings["use_local_bounds"]:
        failure_message, phase1_outcome, preserved_start = _local_phase_failure_result(
            seed_regime
        )
        phase1_messages.append(failure_message)
        return _build_penalty_phase1_result(
            used_phase1=True,
            phase1_iterations=phase1_iterations,
            phase1_termination_message="; ".join(phase1_messages),
            phase1_success=False,
            phase1_outcome=phase1_outcome,
            continue_search=False,
            next_dofs=run_dict["accepted_x"],
            local_preservation_used=True,
            local_preservation_preserved_start=preserved_start,
            local_preservation_attempts=local_attempts_used,
            local_preservation_radius=local_radius,
            local_preservation_step_rms=None,
            phase1_first_accepted_step_rms=phase1_first_accepted_step_rms,
            phase1_max_accepted_step_rms=phase1_max_accepted_step_rms,
            phase1_anchor_restore_used=phase1_anchor_restore_used,
            phase1_unsafe_accept_rollbacks=phase1_unsafe_accept_rollbacks,
            phase1_invalid_reject_attempts=phase1_invalid_reject_attempts,
            startup_local_phase_regime=seed_regime,
            startup_local_recovery_achieved=startup_local_recovery_achieved,
            bridge_local_donor_ready=bridge_local_donor_ready,
        )

    return _build_penalty_phase1_result(
        used_phase1=True,
        phase1_iterations=phase1_iterations,
        phase1_termination_message="; ".join(phase1_messages),
        phase1_success=phase1_success,
        phase1_outcome="phase1_complete",
        continue_search=True,
        next_dofs=run_dict["accepted_x"],
        local_preservation_used=False,
        local_preservation_preserved_start=False,
        local_preservation_attempts=local_attempts_used,
        local_preservation_radius=None,
        local_preservation_step_rms=None,
        phase1_first_accepted_step_rms=phase1_first_accepted_step_rms,
        phase1_max_accepted_step_rms=phase1_max_accepted_step_rms,
        phase1_anchor_restore_used=phase1_anchor_restore_used,
        phase1_unsafe_accept_rollbacks=phase1_unsafe_accept_rollbacks,
        phase1_invalid_reject_attempts=phase1_invalid_reject_attempts,
        startup_local_phase_regime=None,
        startup_local_recovery_achieved=False,
        bridge_local_donor_ready=False,
    )
