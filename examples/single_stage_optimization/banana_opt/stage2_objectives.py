import inspect
import time
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from alm_utils import (
    ALMSettings,
    alm_result_diagnostics_fields,
    augmented_inequality_objective,
    normalize_alm_constraints,
    normalize_alm_constraint_signals,
    upper_bound_residual,
    zero_gradient_like,
)
from banana_opt.coil_groups import (
    COIL_GROUPS_RESULTS_KEY,
    build_contiguous_manifest,
)
from banana_opt.hardware_contracts import (
    TF_CURRENT_HARD_LIMIT_A,
    fixed_stage2_clearance_contract,
    is_major_radius_offspec,
)
from banana_opt.hardware_constraint_schema import (
    ALMConstraintMetadata,
    ALM_OBJECTIVE_SCALE_FLOOR,
    alm_constraint_metadata_payload,
    build_hardware_constraint_artifact_payload_fields,
    build_hardware_constraint_status,
    build_threshold_overrides,
    hardware_constraint_alm_metadata,
    hardware_constraint_alm_names,
)
from banana_opt.poloidal_extent import poloidal_extent_rad_from_objective
from banana_opt.single_stage_geometry import build_surface_configs
from banana_opt.smoothing import smoothmax_selected, smoothmin_selected
from banana_opt.smooth_distance_selection import (
    pairwise_block_min,
    point_tree,
    select_pairwise_near_min,
    softmin_selection_window,
    surface_dgamma_by_dcoeff_derivative,
    surface_points_tree_shape,
)
from banana_opt.stage2_single_stage_handoff import (
    BOOZER_FAILURE_POLICY_RESTORE_LAST_SUCCESS,
    attempt_initialize_boozer_surface,
    compute_tf_G0,
    restore_boozer_solve_state,
    run_boozer_with_failure_policy,
    snapshot_boozer_solve_state,
)
from simsopt.geo.surfaceobjectives import Iotas
from simsopt.objectives import QuadraticPenalty


_SMOOTHING_EPS = float(np.finfo(float).eps)
_STAGE2_SOLVE_FAILURE_REJECT_VIOLATION = 1.0
_STAGE2_HOT_LOOP_SELF_INTERSECTION_ANGLE = 0.0

_STAGE2_FAILURE_REASON_NONE = None
_STAGE2_FAILURE_REASON_SOLVE = "solve_failed"
_STAGE2_FAILURE_REASON_SELF_INTERSECTION = "self_intersecting"


def _boozer_surface_is_self_intersecting(boozer_surface) -> bool:
    return bool(
        boozer_surface.surface.is_self_intersecting(
            angle=_STAGE2_HOT_LOOP_SELF_INTERSECTION_ANGLE,
        )
    )


def _new_derivative():
    from simsopt._core.derivative import Derivative

    return Derivative({})


@dataclass
class Stage2IotaRuntimeStats:
    bootstrap_seconds: float
    runtime_seconds: float = 0.0
    runtime_calls: int = 0


@dataclass(frozen=True)
class Stage2IotaState:
    iota: float
    penalty: float
    abs_error: float
    feasible: bool
    solve_failed: bool = False


@dataclass(frozen=True)
class Stage2IotaEvaluation:
    state: Stage2IotaState
    penalty_grad: np.ndarray | None = None


class Stage2GuardedBoozerEvaluator:
    def __init__(self, boozer_surface):
        self.boozer_surface = boozer_surface
        self.last_successful_state = snapshot_boozer_solve_state(boozer_surface)
        self.last_solve_failed = False
        self.last_failure_reason: str | None = _STAGE2_FAILURE_REASON_NONE

    def _refresh_if_needed(self) -> None:
        if self.boozer_surface.need_to_run_code:
            self.run_guarded()

    def _build_failed_state(
        self,
        *,
        target: float,
        tolerance: float,
    ) -> Stage2IotaState:
        return _build_stage2_iota_state_from_iota(
            self.last_successful_state.iota,
            target=target,
            tolerance=tolerance,
            solve_failed=True,
        )

    def run_guarded(self):
        res = self.boozer_surface.res
        solve_attempt = run_boozer_with_failure_policy(
            self.boozer_surface,
            res["iota"],
            res["G"],
            failure_policy=BOOZER_FAILURE_POLICY_RESTORE_LAST_SUCCESS,
            last_successful_state=self.last_successful_state,
        )
        if not solve_attempt.solve_success:
            self.last_solve_failed = True
            self.last_failure_reason = _STAGE2_FAILURE_REASON_SOLVE
            return {"success": False, "reason": self.last_failure_reason}
        self_intersection_check_completed = False
        try:
            self_intersecting = _boozer_surface_is_self_intersecting(
                self.boozer_surface,
            )
            self_intersection_check_completed = True
        finally:
            if not self_intersection_check_completed:
                restore_boozer_solve_state(
                    self.boozer_surface,
                    self.last_successful_state,
                )
        if self_intersecting:
            restore_boozer_solve_state(
                self.boozer_surface,
                self.last_successful_state,
            )
            self.last_solve_failed = True
            self.last_failure_reason = _STAGE2_FAILURE_REASON_SELF_INTERSECTION
            return {"success": False, "reason": self.last_failure_reason}
        self.last_solve_failed = False
        self.last_failure_reason = _STAGE2_FAILURE_REASON_NONE
        self.last_successful_state = snapshot_boozer_solve_state(self.boozer_surface)
        return {"success": True, "reason": None}

    def evaluate(
        self,
        iota_term,
        penalty_objective,
        *,
        target: float,
        tolerance: float,
    ) -> Stage2IotaEvaluation:
        self._refresh_if_needed()
        if self.last_solve_failed:
            return Stage2IotaEvaluation(
                state=self._build_failed_state(target=target, tolerance=tolerance)
            )
        return _evaluate_stage2_iota_terms(
            iota_term,
            penalty_objective,
            target=target,
            tolerance=tolerance,
        )

    def evaluate_state(
        self,
        iota_term,
        penalty_objective,
        *,
        target: float,
        tolerance: float,
    ) -> Stage2IotaState:
        self._refresh_if_needed()
        if self.last_solve_failed:
            return self._build_failed_state(target=target, tolerance=tolerance)
        return _build_stage2_iota_state(
            iota_term,
            penalty_objective,
            target=target,
            tolerance=tolerance,
        )


@dataclass
class Stage2IotaRuntime:
    mode: str
    boozer_surface: object
    iota_term: object
    penalty_objective: object
    target: float
    tolerance: float
    weight: float
    penalty_threshold: float
    vol_target: float
    constraint_weight: float | None
    num_tf_coils: int
    nphi: int
    ntheta: int
    mpol: int
    ntor: int
    stats: Stage2IotaRuntimeStats
    initial_state: Stage2IotaState
    guarded_boozer_evaluator: Stage2GuardedBoozerEvaluator | None = None
    effective_weight: float | None = None


def stage2_iota_penalty_threshold(iota_tolerance: float) -> float:
    tolerance = float(iota_tolerance)
    if tolerance <= 0.0:
        raise ValueError("Stage 2 iota tolerance must be positive.")
    return 0.5 * tolerance * tolerance


def _stage2_iota_penalty(iota: float, target: float) -> float:
    delta = float(iota) - float(target)
    return 0.5 * delta * delta


def _build_stage2_iota_state_from_iota(
    iota: float,
    *,
    target: float,
    tolerance: float,
    solve_failed: bool = False,
) -> Stage2IotaState:
    iota_value = float(iota)
    abs_error = abs(iota_value - float(target))
    return Stage2IotaState(
        iota=iota_value,
        penalty=_stage2_iota_penalty(iota_value, target),
        abs_error=abs_error,
        feasible=not solve_failed and abs_error <= float(tolerance),
        solve_failed=bool(solve_failed),
    )


def _build_stage2_soft_failure_reject_value_and_grad(
    objective_value: float,
    objective_grad: np.ndarray,
) -> tuple[float, np.ndarray]:
    base_objective = float(objective_value)
    base_grad = np.asarray(objective_grad, dtype=float).copy()
    if base_objective >= 1.0:
        return 2.0 * base_objective, 2.0 * base_grad
    return base_objective + 1.0, base_grad


def _resolve_stage2_soft_effective_weight(
    stage2_iota_runtime: Stage2IotaRuntime,
    *,
    objective_value: float,
    penalty_value: float,
) -> float:
    cached_weight = stage2_iota_runtime.effective_weight
    if cached_weight is not None:
        return float(cached_weight)
    penalty_floor = max(
        float(stage2_iota_runtime.penalty_threshold),
        _SMOOTHING_EPS,
    )
    effective_weight = (
        float(stage2_iota_runtime.weight)
        * float(objective_value)
        / max(float(penalty_value), penalty_floor)
    )
    stage2_iota_runtime.effective_weight = effective_weight
    return effective_weight


def _build_stage2_iota_state(
    iota_term,
    penalty_objective,
    *,
    target: float,
    tolerance: float,
    solve_failed: bool = False,
) -> Stage2IotaState:
    iota = float(iota_term.J())
    penalty = float(penalty_objective.J())
    abs_error = abs(iota - float(target))
    return Stage2IotaState(
        iota=iota,
        penalty=penalty,
        abs_error=abs_error,
        feasible=not solve_failed and abs_error <= float(tolerance),
        solve_failed=bool(solve_failed),
    )


def _evaluate_stage2_iota_terms(
    iota_term,
    penalty_objective,
    *,
    target: float,
    tolerance: float,
    solve_failed: bool = False,
) -> Stage2IotaEvaluation:
    state = _build_stage2_iota_state(
        iota_term,
        penalty_objective,
        target=target,
        tolerance=tolerance,
        solve_failed=solve_failed,
    )
    return Stage2IotaEvaluation(
        state=state,
        penalty_grad=np.asarray(penalty_objective.dJ(), dtype=float),
    )


def evaluate_stage2_iota(
    stage2_iota_runtime: Stage2IotaRuntime,
) -> Stage2IotaEvaluation:
    if stage2_iota_runtime.guarded_boozer_evaluator is not None:
        return stage2_iota_runtime.guarded_boozer_evaluator.evaluate(
            stage2_iota_runtime.iota_term,
            stage2_iota_runtime.penalty_objective,
            target=stage2_iota_runtime.target,
            tolerance=stage2_iota_runtime.tolerance,
        )
    return _evaluate_stage2_iota_terms(
        stage2_iota_runtime.iota_term,
        stage2_iota_runtime.penalty_objective,
        target=stage2_iota_runtime.target,
        tolerance=stage2_iota_runtime.tolerance,
    )


def evaluate_stage2_iota_state(
    stage2_iota_runtime: Stage2IotaRuntime,
) -> Stage2IotaState:
    if stage2_iota_runtime.guarded_boozer_evaluator is not None:
        return stage2_iota_runtime.guarded_boozer_evaluator.evaluate_state(
            stage2_iota_runtime.iota_term,
            stage2_iota_runtime.penalty_objective,
            target=stage2_iota_runtime.target,
            tolerance=stage2_iota_runtime.tolerance,
        )
    return _build_stage2_iota_state(
        stage2_iota_runtime.iota_term,
        stage2_iota_runtime.penalty_objective,
        target=stage2_iota_runtime.target,
        tolerance=stage2_iota_runtime.tolerance,
    )


def _coerce_stage2_partition_counts(
    *,
    total_coils,
    num_tf_coils,
    num_banana_coils,
    num_proxy_coils,
    num_vf_coils,
) -> tuple[int, int, int, int, int]:
    return (
        int(total_coils),
        int(num_tf_coils),
        int(num_banana_coils),
        int(num_proxy_coils),
        int(num_vf_coils),
    )


def validate_stage2_coil_partition_counts(
    *,
    total_coils,
    num_tf_coils,
    num_banana_coils,
    num_proxy_coils,
    num_vf_coils,
    context: str,
) -> None:
    (
        total_coils_int,
        num_tf_coils_int,
        num_banana_coils_int,
        num_proxy_coils_int,
        num_vf_coils_int,
    ) = _coerce_stage2_partition_counts(
        total_coils=total_coils,
        num_tf_coils=num_tf_coils,
        num_banana_coils=num_banana_coils,
        num_proxy_coils=num_proxy_coils,
        num_vf_coils=num_vf_coils,
    )
    expected_total_coils = (
        num_tf_coils_int + num_banana_coils_int + num_proxy_coils_int + num_vf_coils_int
    )
    if total_coils_int != expected_total_coils:
        raise ValueError(
            f"{context} does not match the loaded BiotSavart coil count: "
            f"total={total_coils_int}, expected={expected_total_coils} "
            f"(TF={num_tf_coils_int}, banana={num_banana_coils_int}, "
            f"proxy={num_proxy_coils_int}, vf={num_vf_coils_int})."
        )


def build_stage2_iota_runtime(
    *,
    equilibrium_file: str,
    bs,
    tf_coils,
    major_radius: float,
    toroidal_flux: float,
    nphi: int,
    ntheta: int,
    mpol: int,
    ntor: int,
    vol_target: float,
    iota_target: float,
    iota_tolerance: float,
    constraint_weight: float | None,
    num_tf_coils: int,
    mode: str,
    weight: float = 1.0,
    build_surface_configs_fn=build_surface_configs,
    attempt_initialize_boozer_surface_fn=attempt_initialize_boozer_surface,
    compute_tf_G0_fn=compute_tf_G0,
    iotas_cls=Iotas,
    quadratic_penalty_cls=QuadraticPenalty,
) -> Stage2IotaRuntime:
    if len(tf_coils) != int(num_tf_coils):
        raise ValueError(
            "Stage 2 hot-loop iota setup requires --stage2-iota-num-tf-coils to "
            f"match the actual TF-coil count ({len(tf_coils)}), got {num_tf_coils}."
        )

    outer_surface_config = build_surface_configs_fn(
        equilibrium_file,
        int(nphi),
        int(ntheta),
        float(toroidal_flux),
        float(major_radius),
        float(vol_target),
        1,
        0.8,
    )[-1]
    bootstrap_start = time.perf_counter()
    initialization = attempt_initialize_boozer_surface_fn(
        outer_surface_config["initial_surface"],
        int(mpol),
        int(ntor),
        bs,
        outer_surface_config["target_volume"],
        constraint_weight,
        float(iota_target),
        compute_tf_G0_fn(tf_coils),
        nfp=outer_surface_config["initial_surface"].nfp,
    )
    bootstrap_seconds = time.perf_counter() - bootstrap_start
    if not initialization.success or initialization.boozer_surface is None:
        details = [
            f"solve_success={initialization.solve_success}",
            f"self_intersecting={initialization.self_intersecting}",
            f"solved_iota={initialization.solved_iota}",
        ]
        if initialization.error_type is not None:
            details.append(
                f"{initialization.error_type}: {initialization.error_message}"
            )
        raise RuntimeError(
            "Stage 2 Boozer/iota hot-loop initialization failed: " + ", ".join(details)
        )

    boozer_surface = initialization.boozer_surface
    stats = Stage2IotaRuntimeStats(bootstrap_seconds=bootstrap_seconds)
    original_run_code = boozer_surface.run_code

    def timed_run_code(iota, G=None):
        run_start = time.perf_counter()
        try:
            return original_run_code(iota, G)
        finally:
            stats.runtime_calls += 1
            stats.runtime_seconds += time.perf_counter() - run_start

    boozer_surface.run_code = timed_run_code
    guarded_boozer_evaluator = Stage2GuardedBoozerEvaluator(boozer_surface)
    iota_term = iotas_cls(boozer_surface)
    penalty_objective = quadratic_penalty_cls(iota_term, float(iota_target))
    initial_state = _build_stage2_iota_state(
        iota_term,
        penalty_objective,
        target=float(iota_target),
        tolerance=float(iota_tolerance),
    )
    return Stage2IotaRuntime(
        mode=str(mode),
        boozer_surface=boozer_surface,
        iota_term=iota_term,
        penalty_objective=penalty_objective,
        target=float(iota_target),
        tolerance=float(iota_tolerance),
        weight=float(weight),
        penalty_threshold=stage2_iota_penalty_threshold(iota_tolerance),
        vol_target=float(vol_target),
        constraint_weight=(
            None if constraint_weight is None else float(constraint_weight)
        ),
        num_tf_coils=int(num_tf_coils),
        nphi=int(nphi),
        ntheta=int(ntheta),
        mpol=int(mpol),
        ntor=int(ntor),
        stats=stats,
        initial_state=initial_state,
        guarded_boozer_evaluator=guarded_boozer_evaluator,
    )


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
    final_poloidal_extent_rad,
    poloidal_extent_threshold_rad,
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
        "poloidal_extent_rad": final_poloidal_extent_rad,
        "poloidal_extent_threshold_rad": poloidal_extent_threshold_rad,
        "banana_current_A": banana_current_A,
        "banana_current_max_A": banana_current_max_A,
        "tf_current_A": tf_current_A,
        "tf_current_limit_A": TF_CURRENT_HARD_LIMIT_A,
        "artifact_hardware_status": hardware_status,
    }


def _stage2_constraint_names(
    *,
    include_coil_surface: bool,
    include_poloidal_extent: bool = False,
    include_iota_penalty: bool = False,
) -> tuple[str, ...]:
    requested_names = [
        "coil_length",
        "coil_coil_spacing",
        "max_curvature",
        "banana_current",
    ]
    if include_coil_surface:
        requested_names.insert(2, "coil_surface_spacing")
    if include_poloidal_extent:
        requested_names.append("poloidal_extent")
    constraint_names = list(hardware_constraint_alm_names(names=tuple(requested_names)))
    if include_iota_penalty:
        constraint_names.append("iota_penalty")
    return tuple(constraint_names)


def _legacy_stage2_constraint_names(
    *,
    include_coil_surface: bool,
    include_poloidal_extent: bool = False,
    include_iota_penalty: bool = False,
) -> tuple[str, ...]:
    if include_coil_surface:
        constraint_names = [
            "coil_length_upper_bound",
            "coil_coil_spacing",
            "coil_surface_spacing",
            "max_curvature",
            "banana_current_upper_bound",
        ]
    else:
        constraint_names = [
            "coil_length_upper_bound",
            "coil_coil_spacing",
            "max_curvature",
            "banana_current_upper_bound",
        ]
    if include_poloidal_extent:
        constraint_names.append("poloidal_extent")
    if include_iota_penalty:
        constraint_names.append("iota_penalty")
    return tuple(constraint_names)


def _ordered_constraint_values(
    constraint_names: tuple[str, ...],
    values_by_name: dict[str, object],
) -> list[object]:
    return [values_by_name[name] for name in constraint_names]


def _stage2_alm_signal_values(
    constraint_names: tuple[str, ...],
    metadata_by_name: Mapping[str, ALMConstraintMetadata],
    *,
    hard_signed_values: np.ndarray,
    hard_violation_values: np.ndarray,
    surrogate_signed_values: np.ndarray,
    surrogate_violation_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dual_update_values = np.empty(len(constraint_names), dtype=float)
    feasibility_values = np.empty(len(constraint_names), dtype=float)
    for index, constraint_name in enumerate(constraint_names):
        metadata = metadata_by_name[constraint_name]
        dual_update_values[index] = (
            hard_signed_values[index]
            if metadata.dual_update_value_kind == "hard"
            else surrogate_signed_values[index]
        )
        feasibility_values[index] = (
            hard_violation_values[index]
            if metadata.feasibility_value_kind == "hard"
            else surrogate_violation_values[index]
        )
    return dual_update_values, feasibility_values


def _stage2_hardware_threshold_overrides(
    *,
    length_target,
    Jccdist,
    Jc,
    banana_current_max_A,
    Jcsdist=None,
    poloidal_extent_threshold_rad=None,
) -> dict[str, float]:
    return build_threshold_overrides(
        (
            ("coil_length", length_target),
            ("coil_coil_spacing", Jccdist.minimum_distance),
            (
                "coil_surface_spacing",
                None if Jcsdist is None else Jcsdist.minimum_distance,
            ),
            ("max_curvature", Jc.threshold),
            ("banana_current", banana_current_max_A),
            ("poloidal_extent", poloidal_extent_threshold_rad),
        )
    )


def _require_explicit_stage2_alm_threshold(name: str, value) -> float:
    if value is None:
        raise ValueError(
            f"Stage 2 ALM constraint {name!r} requires an explicit threshold."
        )
    return float(value)


def _stage2_alm_constraint_metadata(
    constraint_names: tuple[str, ...],
    *,
    threshold_overrides: Mapping[str, float],
    activity_tolerance_by_name: Mapping[str, float],
    iota_penalty_threshold: float | None = None,
) -> dict[str, ALMConstraintMetadata]:
    metadata_by_name: dict[str, ALMConstraintMetadata] = {}
    surrogate_hardware_names = {
        "coil_coil_spacing",
        "coil_surface_spacing",
        "max_curvature",
        "poloidal_extent",
    }
    for constraint_name in constraint_names:
        activity_tolerance = float(activity_tolerance_by_name[constraint_name])
        if constraint_name == "iota_penalty":
            raw_threshold = _require_explicit_stage2_alm_threshold(
                "iota_penalty",
                iota_penalty_threshold,
            )
            metadata_by_name[constraint_name] = ALMConstraintMetadata(
                scale=max(raw_threshold, ALM_OBJECTIVE_SCALE_FLOOR),
                block="physics",
                activity_tolerance=activity_tolerance,
                raw_threshold=raw_threshold,
                source="stage2_iota_penalty_threshold",
                objective_value_kind="raw_physics",
                gradient_value_kind="raw_physics",
                dual_update_value_kind="hard",
                feasibility_value_kind="hard",
                certification_value_kind="hard",
            )
            continue
        if (
            constraint_name == "coil_length_upper_bound"
            and "coil_length" not in threshold_overrides
        ):
            _require_explicit_stage2_alm_threshold("coil_length_upper_bound", None)
        uses_surrogate = constraint_name in surrogate_hardware_names
        metadata_by_name[constraint_name] = hardware_constraint_alm_metadata(
            constraint_name,
            threshold_overrides=threshold_overrides,
            activity_tolerance=activity_tolerance,
            objective_value_kind="surrogate" if uses_surrogate else "hard",
            gradient_value_kind="surrogate" if uses_surrogate else "hard",
            dual_update_value_kind="hard",
            feasibility_value_kind="hard",
        )
    return metadata_by_name


def build_stage2_results(
    *,
    args,
    plasma_surf_filename,
    file_loc,
    stage2_bs_path,
    tf_current_A,
    tf_current_sum_abs_A,
    num_tf_coils,
    num_banana_coils,
    num_proxy_coils,
    num_vf_coils,
    initial_banana_current_A,
    banana_current_A,
    banana_to_tf_current_ratio,
    finite_current_mode,
    boozer_current_convention,
    proxy_plasma_current_A,
    vf_current_A,
    vf_template_path,
    total_coils,
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
    final_plasma_major_radius_m,
    final_plasma_minor_radius_m,
    field_error,
    intersecting,
    final_max_curvature,
    final_coil_length,
    final_curve_curve_min_dist,
    hardware_status,
    final_curve_surface_min_dist=None,
    plasma_vessel_min_dist=None,
    final_poloidal_extent_rad=None,
    poloidal_extent_threshold_rad=None,
):
    alm_enabled = constraint_method == "alm"
    hardware_snapshot = _build_stage2_artifact_hardware_snapshot(
        hardware_status=hardware_status,
        final_coil_length=final_coil_length,
        length_target=length_target,
        final_curve_curve_min_dist=final_curve_curve_min_dist,
        final_max_curvature=final_max_curvature,
        final_curve_surface_min_dist=final_curve_surface_min_dist,
        plasma_vessel_min_dist=plasma_vessel_min_dist,
        final_poloidal_extent_rad=final_poloidal_extent_rad,
        poloidal_extent_threshold_rad=poloidal_extent_threshold_rad,
        banana_current_A=banana_current_A,
        banana_current_max_A=float(args.banana_current_max_A),
        tf_current_A=tf_current_A,
    )
    validate_stage2_coil_partition_counts(
        total_coils=total_coils,
        num_tf_coils=num_tf_coils,
        num_banana_coils=num_banana_coils,
        num_proxy_coils=num_proxy_coils,
        num_vf_coils=num_vf_coils,
        context="Stage 2 coil partition metadata",
    )
    coil_groups_manifest = build_contiguous_manifest(
        num_tf_coils=int(num_tf_coils),
        num_banana_coils=int(num_banana_coils),
        num_proxy_coils=int(num_proxy_coils),
        num_vf_coils=int(num_vf_coils),
    )
    return {
        "PLASMA_SURF_FILENAME": plasma_surf_filename,
        "PLASMA_SURF_PATH": file_loc,
        "STAGE2_BS_PATH": stage2_bs_path,
        "TF_CURRENT_A": float(tf_current_A),
        "TF_CURRENT_SUM_ABS_A": float(tf_current_sum_abs_A),
        "NUM_TF_COILS": int(num_tf_coils),
        "NUM_BANANA_COILS": int(num_banana_coils),
        "NUM_PROXY_COILS": int(num_proxy_coils),
        "NUM_VF_COILS": int(num_vf_coils),
        COIL_GROUPS_RESULTS_KEY: coil_groups_manifest.to_json_payload(),
        "BANANA_INIT_CURRENT_A": float(initial_banana_current_A),
        "BANANA_CURRENT_MAX_A": float(args.banana_current_max_A),
        "BANANA_CURRENT_A": float(banana_current_A),
        "BANANA_TO_TF_CURRENT_RATIO": float(banana_to_tf_current_ratio),
        "FINITE_CURRENT_MODE": str(finite_current_mode),
        "BOOZER_CURRENT_CONVENTION": str(boozer_current_convention),
        "PROXY_PLASMA_CURRENT_A": float(proxy_plasma_current_A),
        "VF_CURRENT_A": float(vf_current_A),
        "VF_TEMPLATE_PATH": (
            None if vf_template_path in {None, ""} else str(vf_template_path)
        ),
        "CC_THRESHOLD": cc_threshold,
        "CC_WEIGHT": cc_weight,
        "CURVATURE_WEIGHT": curvature_weight,
        "CURVATURE_THRESHOLD": curvature_threshold,
        "POLOIDAL_EXTENT_RAD": (
            None if final_poloidal_extent_rad is None else float(final_poloidal_extent_rad)
        ),
        "POLOIDAL_EXTENT_THRESHOLD_RAD": (
            None
            if poloidal_extent_threshold_rad is None
            else float(poloidal_extent_threshold_rad)
        ),
        "LENGTH_WEIGHT": length_weight,
        **fixed_stage2_clearance_contract(),
        "CONSTRAINT_METHOD": constraint_method,
        "theta_center": theta_center,
        "phi_center": phi_center,
        "theta_width": theta_width,
        "phi_width": phi_width,
        "LENGTH_TARGET": length_target,
        "MAJOR_RADIUS": major_radius,
        "R0_OFF_SPEC": is_major_radius_offspec(major_radius),
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
        "ALM_TRUST_RADIUS_SHRINK": args.alm_trust_radius_shrink
        if alm_enabled
        else None,
        "ALM_TRUST_RADIUS_GROW": args.alm_trust_radius_grow if alm_enabled else None,
        "ALM_MAX_INNER_ATTEMPTS": args.alm_max_inner_attempts if alm_enabled else None,
        "ALM_DISTANCE_SMOOTHING": args.alm_distance_smoothing if alm_enabled else None,
        "ALM_CURVATURE_SMOOTHING": args.alm_curvature_smoothing
        if alm_enabled
        else None,
        "ALM_TAYLOR_TEST_ENABLED": args.alm_taylor_test if alm_enabled else None,
        "ALM_TAYLOR_TEST_SEED": args.alm_taylor_test_seed if alm_enabled else None,
        "ALM_TAYLOR_RESULT": alm_taylor_result,
        "ALM_TERMINATION_REASON": getattr(alm_result, "termination_reason", None),
        "ALM_CONVERGED": getattr(alm_result, "converged_to_tolerances", None),
        "ALM_RESTORED_BEST_FEASIBLE": getattr(
            alm_result, "restored_best_feasible", None
        ),
        "ALM_RESTORED_BEST_FEASIBLE_REASON": getattr(
            alm_result,
            "restored_best_feasible_reason",
            None,
        ),
        "ALM_INNER_OPTIMIZER_SUCCESS": getattr(alm_result, "optimizer_success", None),
        "ALM_INNER_OPTIMIZER_MESSAGE": getattr(alm_result, "optimizer_message", None),
        "ALM_SCHEMA_VERSION": getattr(alm_result, "alm_schema_version", None),
        "ALM_FINAL_MAX_FEASIBILITY_VIOLATION": getattr(
            alm_result,
            "final_max_feasibility_violation",
            None,
        ),
        "ALM_FINAL_MAX_NORMALIZED_VIOLATION": getattr(
            alm_result,
            "final_max_feasibility_violation",
            None,
        ),
        "ALM_FINAL_STATIONARITY_NORM": getattr(
            alm_result, "final_stationarity_norm", None
        ),
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
        "ALM_FINAL_RAW_DUAL_ESTIMATES": getattr(alm_result, "raw_dual_estimates", None),
        "ALM_CONSTRAINT_SCALES": getattr(alm_result, "constraint_scales", None),
        "ALM_CONSTRAINT_BLOCKS": getattr(alm_result, "constraint_blocks", None),
        "ALM_CONSTRAINT_SCALE_SOURCES": getattr(
            alm_result,
            "constraint_scale_sources",
            None,
        ),
        "ALM_FINAL_CONSTRAINT_VALUES": getattr(
            alm_result,
            "raw_constraint_values",
            getattr(alm_result, "constraint_values", None),
        ),
        "ALM_FINAL_NORMALIZED_CONSTRAINT_VALUES": getattr(
            alm_result,
            "normalized_constraint_values",
            getattr(alm_result, "constraint_values", None),
        ),
        "ALM_FINAL_SOLVER_CONSTRAINT_VALUES": getattr(
            alm_result,
            "raw_solver_constraint_values",
            getattr(alm_result, "solver_constraint_values", None),
        ),
        "ALM_FINAL_NORMALIZED_SOLVER_CONSTRAINT_VALUES": getattr(
            alm_result,
            "normalized_solver_constraint_values",
            getattr(alm_result, "solver_constraint_values", None),
        ),
        "ALM_FINAL_HARD_SIGNED_CONSTRAINT_VALUES": getattr(
            alm_result,
            "raw_hard_signed_constraint_values",
            getattr(alm_result, "hard_signed_constraint_values", None),
        ),
        "ALM_FINAL_NORMALIZED_HARD_SIGNED_CONSTRAINT_VALUES": getattr(
            alm_result,
            "hard_signed_constraint_values",
            None,
        ),
        "ALM_FINAL_HARD_VIOLATION_VALUES": getattr(
            alm_result,
            "raw_hard_violation_values",
            getattr(alm_result, "hard_violation_values", None),
        ),
        "ALM_FINAL_RAW_HARD_VIOLATION_BY_CONSTRAINT": getattr(
            alm_result,
            "raw_hard_violation_values",
            getattr(alm_result, "hard_violation_values", None),
        ),
        "ALM_FINAL_NORMALIZED_HARD_VIOLATION_VALUES": getattr(
            alm_result,
            "hard_violation_values",
            None,
        ),
        "ALM_FINAL_SURROGATE_SIGNED_CONSTRAINT_VALUES": getattr(
            alm_result,
            "raw_surrogate_signed_constraint_values",
            getattr(alm_result, "surrogate_signed_constraint_values", None),
        ),
        "ALM_FINAL_NORMALIZED_SURROGATE_SIGNED_CONSTRAINT_VALUES": getattr(
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
        "FINAL_LCFS_MAJOR_RADIUS_M": float(final_plasma_major_radius_m),
        "FINAL_LCFS_MINOR_RADIUS_M": float(final_plasma_minor_radius_m),
        "FIELD_ERROR": float(field_error),
        "SELF_INTERSECTING": intersecting,
        **build_hardware_constraint_artifact_payload_fields(hardware_snapshot),
    }


def make_stage2_fun(
    JF,
    new_bs,
    new_surf,
    Jf,
    Jls,
    Jccdist,
    Jc,
    stage2_iota_runtime: Stage2IotaRuntime | None = None,
    *,
    emit_diagnostics=False,
):
    soft_mode_enabled = (
        stage2_iota_runtime is not None and stage2_iota_runtime.mode == "soft"
    )

    def fun(dofs):
        JF.x = dofs
        J = float(JF.J())
        grad = np.asarray(JF.dJ(), dtype=float)
        iota_state = None
        iota_evaluation = None
        if soft_mode_enabled:
            iota_evaluation = evaluate_stage2_iota(stage2_iota_runtime)
            iota_state = iota_evaluation.state
            if iota_state.solve_failed:
                J, grad = _build_stage2_soft_failure_reject_value_and_grad(
                    J,
                    grad,
                )
            else:
                effective_weight = _resolve_stage2_soft_effective_weight(
                    stage2_iota_runtime,
                    objective_value=J,
                    penalty_value=iota_state.penalty,
                )
                J += effective_weight * iota_state.penalty
                grad = grad + (
                    effective_weight
                    * np.asarray(iota_evaluation.penalty_grad, dtype=float)
                )
        if emit_diagnostics:
            unitn = new_surf.unitnormal()
            BdotN = np.mean(
                np.abs(np.sum(new_bs.B().reshape(unitn.shape) * unitn, axis=2))
            )
            outstr = f"J={J:.1e}, Jf={Jf.J():.1e}, ⟨B·n⟩={BdotN:.1e}"
            outstr += f", Len={Jls.J():.1f}m"
            outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}m"
            outstr += f", Curvature={Jc.J():.2f}"
            if stage2_iota_runtime is not None:
                if iota_state is None:
                    iota_state = evaluate_stage2_iota_state(stage2_iota_runtime)
                outstr += (
                    f", Iota={iota_state.iota:.4f}, Jiota={iota_state.penalty:.2e}"
                )
                if iota_state.solve_failed:
                    evaluator = getattr(
                        stage2_iota_runtime,
                        "guarded_boozer_evaluator",
                        None,
                    )
                    reason = (
                        evaluator.last_failure_reason
                        if evaluator is not None
                        and getattr(evaluator, "last_failure_reason", None) is not None
                        else _STAGE2_FAILURE_REASON_SOLVE
                    )
                    outstr += f", IotaSolveFailed=1, IotaFailureReason={reason}"
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
    poloidal_extent_rad=None,
    poloidal_extent_threshold_rad=None,
    banana_current_A=None,
    banana_current_threshold=None,
    tf_current_A=None,
    tf_current_threshold=None,
):
    artifact_threshold_overrides = build_threshold_overrides(
        (
            ("coil_length", length_target),
            ("coil_coil_spacing", cc_threshold),
            ("max_curvature", curvature_threshold),
            ("coil_surface_spacing", coil_surface_threshold),
            ("surface_vessel_spacing", plasma_vessel_threshold),
            ("poloidal_extent", poloidal_extent_threshold_rad),
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
        "poloidal_extent": poloidal_extent_rad,
        "banana_current": banana_current_A,
        "tf_current": tf_current_A,
    }
    status = build_hardware_constraint_status(
        measured_values,
        applies_to="artifact",
        threshold_overrides=artifact_threshold_overrides,
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
    if poloidal_extent_rad is not None and poloidal_extent_threshold_rad is not None:
        status["poloidal_extent_rad"] = float(poloidal_extent_rad)
        status["poloidal_extent_threshold_rad"] = float(poloidal_extent_threshold_rad)
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
    include_poloidal_extent: bool = False,
    include_iota_penalty: bool = False,
    iota_tolerance: float = 0.0,
):
    distance_activity_tolerance = max(
        softmin_selection_window(distance_smoothing),
        _SMOOTHING_EPS,
    )
    tolerances = [
        length_tolerance,
        distance_activity_tolerance,
        max(4.0 * float(curvature_smoothing), _SMOOTHING_EPS),
        banana_current_tolerance,
    ]
    if include_coil_surface:
        tolerances = [
            tolerances[0],
            tolerances[1],
            tolerances[1],
            tolerances[2],
            tolerances[3],
        ]
    if include_poloidal_extent:
        tolerances.append(max(float(curvature_smoothing), _SMOOTHING_EPS))
    if include_iota_penalty:
        tolerances.append(
            max(stage2_iota_penalty_threshold(iota_tolerance), _SMOOTHING_EPS)
        )
    return tolerances


def resolve_stage2_constraint_activity_tolerances(
    stage2_constraint_activity_tolerances_fn,
    distance_smoothing: float,
    curvature_smoothing: float,
    *,
    include_coil_surface: bool,
    include_poloidal_extent: bool = False,
    include_iota_penalty: bool = False,
    iota_tolerance: float | None = None,
):
    parameters = inspect.signature(stage2_constraint_activity_tolerances_fn).parameters
    call_kwargs = {}
    if "include_coil_surface" in parameters:
        call_kwargs["include_coil_surface"] = include_coil_surface
    if "include_poloidal_extent" in parameters:
        call_kwargs["include_poloidal_extent"] = include_poloidal_extent
    if "include_iota_penalty" in parameters:
        call_kwargs["include_iota_penalty"] = include_iota_penalty
    if "iota_tolerance" in parameters and iota_tolerance is not None:
        call_kwargs["iota_tolerance"] = iota_tolerance
    raw_tolerances = stage2_constraint_activity_tolerances_fn(
        distance_smoothing,
        curvature_smoothing,
        **call_kwargs,
    )
    tolerance_values = [float(value) for value in raw_tolerances]
    constraint_names = _legacy_stage2_constraint_names(
        include_coil_surface=include_coil_surface,
        include_poloidal_extent=include_poloidal_extent,
        include_iota_penalty=include_iota_penalty,
    )
    if len(tolerance_values) != len(constraint_names):
        raise ValueError(
            "Stage 2 activity tolerance helper returned "
            f"{len(tolerance_values)} values for {len(constraint_names)} constraints."
        )
    return {name: value for name, value in zip(constraint_names, tolerance_values)}


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
        if grad_array.shape != sanitized_base_grad.shape or not np.all(
            np.isfinite(grad_array)
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
    curve_points = [np.asarray(curve.gamma(), dtype=float) for curve in curves]
    curve_trees = [point_tree(points) for points in curve_points]
    pair_blocks = []
    hard_min = np.inf
    for i, gamma_i in enumerate(curve_points):
        for j in range(i):
            block_min = pairwise_block_min(
                gamma_i,
                curve_points[j],
                right_tree=curve_trees[j],
            )
            hard_min = min(hard_min, block_min)
            pair_blocks.append((i, j, block_min))

    if not pair_blocks:
        hard_signed_value = float(minimum_distance)
        return (
            hard_signed_value,
            zero_gradient_like(base_objective_optimizable.x),
            hard_signed_value,
        )

    selection_window = softmin_selection_window(temperature)
    selected_distances = []
    selected_entries = []
    selection_threshold = hard_min + selection_window
    for i, j, block_min in pair_blocks:
        if block_min > selection_threshold:
            continue
        rows, cols, diffs, distances = select_pairwise_near_min(
            curve_points[i],
            curve_points[j],
            selection_threshold,
            left_tree=curve_trees[i],
            right_tree=curve_trees[j],
        )
        selected_distances.append(distances)
        selected_entries.append((i, j, rows, cols, diffs, distances))

    flat_distances = np.concatenate(selected_distances)
    smooth_min, flat_weights = smoothmin_selected(
        flat_distances,
        temperature,
        _SMOOTHING_EPS,
    )

    point_gradients = [np.zeros_like(gamma) for gamma in curve_points]
    offset = 0
    for i, j, rows, cols, diffs, distances in selected_entries:
        count = len(distances)
        local_weights = flat_weights[offset : offset + count]
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
    signed_value = float(minimum_distance) - smooth_min
    hard_signed_value = float(minimum_distance) - float(hard_min)
    return signed_value, -grad, hard_signed_value


def smooth_min_curve_surface_signed_constraint(
    curves,
    surface,
    minimum_distance: float,
    temperature: float,
    base_objective_optimizable,
):
    if not curves:
        hard_signed_value = float(minimum_distance)
        return (
            hard_signed_value,
            zero_gradient_like(base_objective_optimizable.x),
            hard_signed_value,
        )

    surface_points, surface_tree, surface_gamma_shape = surface_points_tree_shape(
        surface
    )
    curve_points = [np.asarray(curve.gamma(), dtype=float) for curve in curves]
    curve_trees = [None] * len(curve_points)
    curve_blocks = []
    hard_min = np.inf
    for curve_index, gamma in enumerate(curve_points):
        block_min = pairwise_block_min(gamma, surface_points, right_tree=surface_tree)
        hard_min = min(hard_min, block_min)
        curve_blocks.append((curve_index, block_min))

    selection_window = softmin_selection_window(temperature)
    selected_distances = []
    selected_entries = []
    selection_threshold = hard_min + selection_window
    for curve_index, block_min in curve_blocks:
        if block_min > selection_threshold:
            continue
        if curve_trees[curve_index] is None:
            curve_trees[curve_index] = point_tree(curve_points[curve_index])
        rows, cols, diffs, distances = select_pairwise_near_min(
            curve_points[curve_index],
            surface_points,
            selection_threshold,
            left_tree=curve_trees[curve_index],
            right_tree=surface_tree,
        )
        selected_distances.append(distances)
        selected_entries.append((curve_index, rows, cols, diffs, distances))

    flat_distances = np.concatenate(selected_distances)
    smooth_min, flat_weights = smoothmin_selected(
        flat_distances,
        temperature,
        _SMOOTHING_EPS,
    )

    curve_gradients = [np.zeros_like(gamma) for gamma in curve_points]
    surface_gradient = np.zeros_like(surface_points)
    offset = 0
    for curve_index, rows, cols, diffs, distances in selected_entries:
        count = len(distances)
        local_weights = flat_weights[offset : offset + count]
        offset += count
        directions = diffs / np.maximum(distances[:, None], _SMOOTHING_EPS)
        np.add.at(
            curve_gradients[curve_index], rows, local_weights[:, None] * directions
        )
        np.add.at(surface_gradient, cols, -local_weights[:, None] * directions)

    derivative = _new_derivative()
    for curve, point_gradient in zip(curves, curve_gradients):
        if np.any(point_gradient):
            derivative += curve.dgamma_by_dcoeff_vjp(point_gradient)
    if np.any(surface_gradient):
        derivative += surface_dgamma_by_dcoeff_derivative(
            surface,
            surface_gradient.reshape(surface_gamma_shape),
        )
    grad = np.asarray(derivative(base_objective_optimizable), dtype=float)
    # grad = d(smooth_min)/dx, but signed_value = min_dist - smooth_min,
    # so d(signed_value)/dx = -d(smooth_min)/dx = -grad.
    signed_value = float(minimum_distance) - smooth_min
    hard_signed_value = float(minimum_distance) - float(hard_min)
    return signed_value, -grad, hard_signed_value


def _stage2_distance_minimum(distance_obj, hard_signed_value, emit_diagnostics):
    if emit_diagnostics:
        return float(distance_obj.shortest_distance())
    return float(distance_obj.minimum_distance) - float(hard_signed_value)


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
    Jpoloidal=None,
    poloidal_extent_threshold_rad=None,
    poloidal_extent_smoothing=None,
    smooth_poloidal_extent_signed_constraint=None,
    stage2_iota_runtime: Stage2IotaRuntime | None = None,
    emit_diagnostics=False,
):
    length_target = _require_explicit_stage2_alm_threshold(
        "coil_length_upper_bound",
        length_target,
    )
    base_objective.x = dofs
    base_value = float(base_objective.J())
    base_grad = np.asarray(base_objective.dJ(), dtype=float)
    base_objective_optimizable = base_objective

    coil_length = float(Jls.J())
    length_violation = upper_bound_residual(coil_length, length_target)
    length_grad = np.asarray(
        Jls.dJ(partials=True)(base_objective_optimizable), dtype=float
    )

    (
        curve_curve_signed_value,
        curve_curve_grad,
        curve_curve_hard_signed_value,
    ) = smooth_min_distance_signed_constraint(
        Jccdist.curves,
        Jccdist.minimum_distance,
        distance_smoothing,
        base_objective_optimizable,
    )
    curve_curve_min_dist = _stage2_distance_minimum(
        Jccdist,
        curve_curve_hard_signed_value,
        emit_diagnostics,
    )
    curve_curve_violation = upper_bound_residual(
        curve_curve_hard_signed_value,
        0.0,
    )
    include_coil_surface = (
        Jcsdist is not None and smooth_min_curve_surface_signed_constraint is not None
    )
    if include_coil_surface:
        (
            curve_surface_signed_value,
            curve_surface_grad,
            curve_surface_hard_signed_value,
        ) = (
            smooth_min_curve_surface_signed_constraint(
                Jcsdist.curves,
                Jcsdist.surface,
                Jcsdist.minimum_distance,
                distance_smoothing,
                base_objective_optimizable,
            )
        )
        curve_surface_min_dist = _stage2_distance_minimum(
            Jcsdist,
            curve_surface_hard_signed_value,
            emit_diagnostics,
        )
        curve_surface_violation = upper_bound_residual(
            curve_surface_hard_signed_value,
            0.0,
        )

    max_curvature = float(np.max(Jc.curve.kappa()))
    curvature_violation = upper_bound_residual(max_curvature, Jc.threshold)
    curvature_signed_value, curvature_grad = smooth_max_curvature_signed_constraint(
        Jc.curve,
        Jc.threshold,
        curvature_smoothing,
        base_objective_optimizable,
    )
    include_poloidal_extent = (
        Jpoloidal is not None
        and poloidal_extent_threshold_rad is not None
        and smooth_poloidal_extent_signed_constraint is not None
    )
    if include_poloidal_extent:
        poloidal_extent_rad = poloidal_extent_rad_from_objective(Jpoloidal)
        poloidal_extent_smoothing_value = (
            curvature_smoothing
            if poloidal_extent_smoothing is None
            else poloidal_extent_smoothing
        )
        (
            poloidal_extent_signed_value,
            poloidal_extent_grad,
            _poloidal_extent_surrogate_violation,
            poloidal_extent_hard_signed_value,
            poloidal_extent_hard_violation,
        ) = smooth_poloidal_extent_signed_constraint(
            Jpoloidal.curve,
            Jpoloidal.R_winding,
            poloidal_extent_threshold_rad,
            poloidal_extent_smoothing_value,
            base_objective_optimizable,
            Z_winding=Jpoloidal.Z_winding,
            include_hard_signal=True,
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
    iota_state = None
    iota_violation = None
    iota_signed_value = None
    include_iota_penalty = (
        stage2_iota_runtime is not None and stage2_iota_runtime.mode == "alm"
    )
    stage2_iota_penalty_threshold_value = None
    if stage2_iota_runtime is not None:
        stage2_iota_penalty_threshold_value = _require_explicit_stage2_alm_threshold(
            "iota_penalty",
            stage2_iota_runtime.penalty_threshold,
        )
        iota_evaluation = evaluate_stage2_iota(stage2_iota_runtime)
        iota_state = iota_evaluation.state
        if iota_state.solve_failed:
            iota_violation = max(
                stage2_iota_penalty_threshold_value,
                _STAGE2_SOLVE_FAILURE_REJECT_VIOLATION,
            )
            iota_signed_value = iota_violation
            iota_grad = zero_gradient_like(base_objective_optimizable.x)
        else:
            iota_violation = upper_bound_residual(
                iota_state.penalty,
                stage2_iota_penalty_threshold_value,
            )
            iota_signed_value = (
                iota_state.penalty - stage2_iota_penalty_threshold_value
            )
            iota_grad = np.asarray(iota_evaluation.penalty_grad, dtype=float)

    active_names = _stage2_constraint_names(
        include_coil_surface=include_coil_surface,
        include_poloidal_extent=include_poloidal_extent,
        include_iota_penalty=include_iota_penalty,
    )
    hard_by_name = {
        "coil_length_upper_bound": coil_length - length_target,
        "coil_coil_spacing": curve_curve_hard_signed_value,
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
    if include_poloidal_extent:
        hard_by_name["poloidal_extent"] = poloidal_extent_hard_signed_value
        surrogate_by_name["poloidal_extent"] = poloidal_extent_signed_value
        grad_by_name["poloidal_extent"] = poloidal_extent_grad
        feasibility_by_name["poloidal_extent"] = poloidal_extent_hard_violation
    if include_coil_surface:
        hard_by_name["coil_surface_spacing"] = curve_surface_hard_signed_value
        surrogate_by_name["coil_surface_spacing"] = curve_surface_signed_value
        grad_by_name["coil_surface_spacing"] = curve_surface_grad
        feasibility_by_name["coil_surface_spacing"] = curve_surface_violation
    if include_iota_penalty:
        hard_by_name["iota_penalty"] = iota_signed_value
        surrogate_by_name["iota_penalty"] = iota_signed_value
        grad_by_name["iota_penalty"] = iota_grad
        feasibility_by_name["iota_penalty"] = iota_violation
    hard_signed_constraint_values = _ordered_constraint_values(
        active_names, hard_by_name
    )
    surrogate_signed_constraint_values = _ordered_constraint_values(
        active_names,
        surrogate_by_name,
    )
    constraint_grads = _ordered_constraint_values(active_names, grad_by_name)
    hard_violation_values = _ordered_constraint_values(
        active_names, feasibility_by_name
    )
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

    tolerance_by_name = resolve_stage2_constraint_activity_tolerances(
        stage2_constraint_activity_tolerances,
        distance_smoothing,
        curvature_smoothing,
        include_coil_surface=include_coil_surface,
        include_poloidal_extent=include_poloidal_extent,
        include_iota_penalty=include_iota_penalty,
        iota_tolerance=(
            None if stage2_iota_runtime is None else stage2_iota_runtime.tolerance
        ),
    )
    raw_constraint_activity_tolerances = np.asarray(
        _ordered_constraint_values(active_names, tolerance_by_name),
        dtype=float,
    )
    threshold_overrides = _stage2_hardware_threshold_overrides(
        length_target=length_target,
        Jccdist=Jccdist,
        Jc=Jc,
        banana_current_max_A=banana_current_max_A,
        Jcsdist=Jcsdist,
        poloidal_extent_threshold_rad=poloidal_extent_threshold_rad,
    )
    metadata_by_name = _stage2_alm_constraint_metadata(
        active_names,
        threshold_overrides=threshold_overrides,
        activity_tolerance_by_name=tolerance_by_name,
        iota_penalty_threshold=(
            None
            if stage2_iota_runtime is None
            else stage2_iota_penalty_threshold_value
        ),
    )
    metadata_payload = alm_constraint_metadata_payload(active_names, metadata_by_name)
    constraint_scales = np.asarray(metadata_payload["constraint_scales"], dtype=float)
    normalized_payload = normalize_alm_constraints(
        sanitized_surrogate_signed_constraint_values,
        sanitized_constraint_grads,
        sanitized_hard_violation_values,
        raw_constraint_activity_tolerances,
        constraint_scales,
    )
    normalized_surrogate_signed_constraint_values = normalized_payload[
        "normalized_signed_values"
    ]
    normalized_constraint_grads = normalized_payload["normalized_constraint_grads"]
    normalized_hard_violation_values = normalized_payload[
        "normalized_feasibility_values"
    ]
    normalized_constraint_activity_tolerances = normalized_payload[
        "normalized_activity_tolerances"
    ]
    normalized_hard_signed_constraint_values = (
        np.asarray(sanitized_hard_signed_constraint_values, dtype=float)
        / constraint_scales
    )
    raw_surrogate_feasibility_values = np.maximum(
        sanitized_surrogate_signed_constraint_values,
        0.0,
    )
    raw_dual_update_values, raw_feasibility_values = _stage2_alm_signal_values(
        active_names,
        metadata_by_name,
        hard_signed_values=sanitized_hard_signed_constraint_values,
        hard_violation_values=sanitized_hard_violation_values,
        surrogate_signed_values=sanitized_surrogate_signed_constraint_values,
        surrogate_violation_values=raw_surrogate_feasibility_values,
    )
    normalized_signal_payload = normalize_alm_constraint_signals(
        raw_dual_update_values,
        raw_feasibility_values,
        raw_constraint_activity_tolerances,
        constraint_scales,
    )
    normalized_dual_update_values = normalized_signal_payload[
        "normalized_signed_values"
    ]
    normalized_feasibility_values = normalized_signal_payload[
        "normalized_feasibility_values"
    ]
    invalid_fields = (
        sanitized_invalid_fields
        + invalid_hard_signed_fields
        + invalid_hard_violation_fields
    )
    evaluation = augmented_inequality_objective(
        sanitized_base_value,
        sanitized_base_grad,
        normalized_surrogate_signed_constraint_values,
        normalized_constraint_grads,
        multipliers,
        penalty,
    )
    evaluation.update(
        {
            "base_value": sanitized_base_value,
            "constraint_names": list(active_names),
            "dual_update_values": normalized_dual_update_values,
            "constraint_grads": normalized_constraint_grads,
            "constraint_activity_tolerances": normalized_constraint_activity_tolerances,
            "feasibility_values": normalized_feasibility_values,
            "hard_signed_constraint_values": normalized_hard_signed_constraint_values,
            "hard_violation_values": normalized_hard_violation_values,
            "surrogate_signed_constraint_values": normalized_surrogate_signed_constraint_values,
            "hard_dual_update_values": normalized_hard_signed_constraint_values,
            "normalized_signed_constraint_values": normalized_surrogate_signed_constraint_values,
            "normalized_feasibility_values": normalized_feasibility_values,
            "raw_constraint_values": sanitized_surrogate_signed_constraint_values,
            "raw_solver_constraint_values": sanitized_surrogate_signed_constraint_values,
            "raw_dual_update_values": raw_dual_update_values,
            "raw_feasibility_values": raw_feasibility_values,
            "raw_hard_signed_constraint_values": sanitized_hard_signed_constraint_values,
            "raw_hard_violation_values": sanitized_hard_violation_values,
            "raw_surrogate_signed_constraint_values": sanitized_surrogate_signed_constraint_values,
            "raw_hard_dual_update_values": sanitized_hard_signed_constraint_values,
            "raw_constraint_grads": sanitized_constraint_grads,
            "raw_constraint_activity_tolerances": raw_constraint_activity_tolerances,
            "max_feasibility_violation": max(normalized_feasibility_values),
            "nonfinite_inputs_sanitized": bool(invalid_fields),
            "nonfinite_input_fields": invalid_fields,
            **metadata_payload,
        }
    )
    if invalid_fields:
        # Keep the diagnostic payload finite enough to inspect, but mark the
        # evaluation itself invalid so generic ALM rejection/salvage logic
        # handles it instead of accepting a fabricated finite sample.
        evaluation["total"] = float("nan")
        evaluation["nonfinite_evaluation"] = True
        evaluation["nonfinite_fields"] = list(invalid_fields)

    if emit_diagnostics:
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
            f", C-C-Sep={curve_curve_min_dist:.2f}m, "
            f"CC+={curve_curve_violation:.2e}, "
            f"CCg={curve_curve_signed_value:.2e}"
        )
        if include_coil_surface:
            outstr += (
                f", C-S-Sep={curve_surface_min_dist:.2f}m, "
                f"CS+={curve_surface_violation:.2e}, "
                f"CSg={curve_surface_signed_value:.2e}"
            )
        outstr += (
            f", Curvature={max_curvature:.2f}, Curv+={curvature_violation:.2e}, "
            f"Curvg={curvature_signed_value:.2e}"
        )
        outstr += (
            f", |BananaI|={banana_current_abs_A:.2f}A, "
            f"BananaI+={banana_current_violation:.2e}, "
            f"BananaIg={banana_current_signed_value:.2e}"
        )
        if iota_state is not None:
            outstr += f", Iota={iota_state.iota:.4f}, Jiota={iota_state.penalty:.2e}"
        if include_iota_penalty:
            outstr += f", Iota+={iota_violation:.2e}, Iotag={iota_signed_value:.2e}"
        if include_poloidal_extent:
            outstr += (
                f", Poloidal={poloidal_extent_rad:.3f}rad, "
                f"Poloidal+={poloidal_extent_hard_violation:.2e}, "
                f"Poloidalg={poloidal_extent_signed_value:.2e}"
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
