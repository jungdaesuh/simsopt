from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path

from .frontier_contracts import EpsilonThresholds
from .frontier_progress_state import (
    FrontierLaneContract,
    FrontierLaneRecord,
    build_frontier_lane_contract,
)
from .frontier_runtime_calibration import effective_lane_budget
from .frontier_scalarization import (
    FRONTIER_REFERENCE_MODE_ACHIEVEMENT,
    FRONTIER_REFERENCE_MODE_EPSILON,
    FrontierLaneSpec,
)
from .single_stage_search_contracts import resolve_frontier_invariant_torus_min

FRONTIER_LANE_WARM_START_MODE_SEED = "seed"
FRONTIER_LANE_WARM_START_MODE_REUSE_LATEST_CERTIFIED = "reuse_latest_certified"


@dataclass(frozen=True)
class FrontierLaneExecution:
    lane_index: int
    lane_spec: FrontierLaneSpec
    lane_args: argparse.Namespace
    lane_budget: int
    stage2_bs_path: Path
    warm_start_source: str
    lane_contract: FrontierLaneContract
    output_root: Path


@dataclass(frozen=True)
class FrontierLaneExecutionResult:
    execution: FrontierLaneExecution
    lane_payload: dict[str, object]


def build_frontier_lane_args(
    args: argparse.Namespace,
    lane_spec: FrontierLaneSpec,
) -> argparse.Namespace:
    lane_args = argparse.Namespace(**vars(args))
    lane_args.frontier_scalarization_type = lane_spec.scalarization_type
    lane_args.iotas_weight = lane_spec.iotas_weight
    lane_args.frontier_volume_weight = lane_spec.frontier_volume_weight
    lane_args.res_weight = lane_spec.res_weight
    if lane_spec.lane_budget is not None:
        lane_args.maxiter = int(lane_spec.lane_budget)
    for attribute_name in (
        "frontier_reference_iota",
        "frontier_reference_iota_scale",
        "frontier_reference_volume",
        "frontier_reference_volume_scale",
        "frontier_reference_qa",
        "frontier_reference_qa_scale",
        "frontier_reference_boozer",
        "frontier_reference_boozer_scale",
        "frontier_boozer_trust_threshold",
        "frontier_boozer_trust_penalty_scale",
        "frontier_chebyshev_rho",
        "frontier_chebyshev_sharpness",
        "frontier_chebyshev_weight_iota",
        "frontier_chebyshev_weight_volume",
        "frontier_chebyshev_weight_qa",
        "frontier_chebyshev_weight_boozer",
        EpsilonThresholds.QA_MAX_KEY,
        EpsilonThresholds.BOOZER_MAX_KEY,
        "frontier_epsilon_penalty_weight",
    ):
        setattr(
            lane_args,
            attribute_name,
            lane_spec.scalarization_params.get(attribute_name),
        )
    return lane_args


def lane_rng_seed(base_seed: int, *, lane_index: int) -> int:
    return int(base_seed) + lane_index


def build_lane_rerun_contract(
    args: argparse.Namespace,
    lane_spec: FrontierLaneSpec,
    *,
    stage2_bs_path: Path,
) -> dict[str, object]:
    return {
        "single_stage_goal_mode": "frontier",
        "stage2_bs_path": str(stage2_bs_path),
        "plasma_surf_filename": Path(args.plasma_surf_filename).name,
        "iotas_weight": lane_spec.iotas_weight,
        "frontier_volume_weight": lane_spec.frontier_volume_weight,
        "frontier_invariant_torus_min": _frontier_invariant_torus_min(args),
        "res_weight": lane_spec.res_weight,
        "maxiter": args.maxiter
        if lane_spec.lane_budget is None
        else lane_spec.lane_budget,
        "constraint_method": args.constraint_method,
        "constraint_mode": _lane_constraint_mode(lane_spec),
        "hardware_search_mode": args.hardware_search_mode,
        "scalarization_type": lane_spec.scalarization_type,
        "scalarization_params": dict(lane_spec.scalarization_params),
    }


def _frontier_invariant_torus_min(args: argparse.Namespace) -> float:
    return resolve_frontier_invariant_torus_min(
        getattr(args, "frontier_invariant_torus_min", None),
        getattr(args, "frontier_kam_min", None),
        environment=os.environ,
    )


def _lane_reference_point(lane_spec: FrontierLaneSpec) -> dict[str, float] | None:
    metric_key_map = {
        "frontier_reference_iota": "iota",
        "frontier_reference_volume": "volume",
        "frontier_reference_qa": "qa_error",
        "frontier_reference_boozer": "boozer_residual",
    }
    reference_point = {
        metric_name: float(lane_spec.scalarization_params[scalarization_key])
        for scalarization_key, metric_name in metric_key_map.items()
        if scalarization_key in lane_spec.scalarization_params
    }
    return reference_point or None


def build_frontier_lane_contract_for_spec(
    args: argparse.Namespace,
    lane_spec: FrontierLaneSpec,
    *,
    campaign_id: str,
    stage2_bs_path: Path,
    warm_start_source: str,
    lane_budget: int,
    lane_index: int,
) -> FrontierLaneContract:
    return build_frontier_lane_contract(
        campaign_id=campaign_id,
        lane_id=lane_spec.lane_id,
        engine=args.frontier_engine,
        reference_point=_lane_reference_point(lane_spec),
        scalarization_type=lane_spec.scalarization_type,
        scalarization_params=lane_spec.scalarization_params,
        constraint_mode=_lane_constraint_mode(lane_spec),
        warm_start_source=warm_start_source,
        optimizer_budget=lane_budget,
        rng_seed=lane_rng_seed(args.frontier_rng_seed, lane_index=lane_index),
        rerun_contract=build_lane_rerun_contract(
            args,
            lane_spec,
            stage2_bs_path=stage2_bs_path,
        ),
    )


def resolve_frontier_lane_warm_start(
    *,
    base_stage2_bs_path: Path,
    lane_records_by_id: dict[str, FrontierLaneRecord],
    lane_specs: list[FrontierLaneSpec],
    lane_index: int,
    warm_start_mode: str,
) -> tuple[Path, str]:
    resolved_base_path = Path(base_stage2_bs_path).resolve()
    if warm_start_mode != FRONTIER_LANE_WARM_START_MODE_REUSE_LATEST_CERTIFIED:
        return resolved_base_path, str(resolved_base_path)
    for candidate_lane_spec in reversed(lane_specs[:lane_index]):
        lane_record = lane_records_by_id.get(candidate_lane_spec.lane_id)
        if lane_record is None or not lane_record.final_certified:
            continue
        warm_start_path = _warm_start_path_from_lane_record(lane_record)
        if warm_start_path is None or not warm_start_path.exists():
            continue
        return warm_start_path.resolve(), str(warm_start_path.resolve())
    return resolved_base_path, str(resolved_base_path)


def frontier_lanes_require_ordered_execution(
    *,
    warm_start_mode: str,
    early_stop_patience_lanes: int,
    lane_workers: int,
) -> bool:
    return (
        int(lane_workers) == 1
        or warm_start_mode == FRONTIER_LANE_WARM_START_MODE_REUSE_LATEST_CERTIFIED
        or int(early_stop_patience_lanes) > 0
    )


def build_frontier_lane_execution_groups(
    lane_specs: list[FrontierLaneSpec],
    *,
    lane_records_by_id: dict[str, FrontierLaneRecord],
    warm_start_mode: str,
    early_stop_patience_lanes: int,
    lane_workers: int,
) -> list[list[tuple[int, FrontierLaneSpec]]]:
    def needs_run(lane_spec: FrontierLaneSpec) -> bool:
        lane_record = lane_records_by_id.get(lane_spec.lane_id)
        return lane_record is None or lane_record.status != "completed"

    pending_lanes = [
        (lane_index, lane_spec)
        for lane_index, lane_spec in enumerate(lane_specs)
        if needs_run(lane_spec)
    ]
    if frontier_lanes_require_ordered_execution(
        warm_start_mode=warm_start_mode,
        early_stop_patience_lanes=early_stop_patience_lanes,
        lane_workers=lane_workers,
    ):
        return [[lane] for lane in pending_lanes]
    return [pending_lanes] if pending_lanes else []


def build_frontier_lane_execution(
    args: argparse.Namespace,
    lane_spec: FrontierLaneSpec,
    *,
    campaign_id: str,
    lane_index: int,
    lane_records_by_id: dict[str, FrontierLaneRecord],
    lane_specs: list[FrontierLaneSpec],
    runtime_defaults,
    stage2_bs_path: Path,
    output_root: Path,
) -> FrontierLaneExecution:
    lane_args = build_frontier_lane_args(args, lane_spec)
    lane_budget = effective_lane_budget(
        lane_spec.lane_budget,
        runtime_defaults,
    )
    lane_args.maxiter = int(lane_budget)
    lane_args.checkpoint_every = int(runtime_defaults.checkpoint_every)
    lane_stage2_bs_path, warm_start_source = resolve_frontier_lane_warm_start(
        base_stage2_bs_path=stage2_bs_path,
        lane_records_by_id=lane_records_by_id,
        lane_specs=lane_specs,
        lane_index=lane_index,
        warm_start_mode=args.frontier_lane_warm_start_mode,
    )
    lane_contract = build_frontier_lane_contract_for_spec(
        lane_args,
        lane_spec,
        campaign_id=campaign_id,
        stage2_bs_path=lane_stage2_bs_path,
        warm_start_source=warm_start_source,
        lane_budget=lane_budget,
        lane_index=lane_index,
    )
    return FrontierLaneExecution(
        lane_index=lane_index,
        lane_spec=lane_spec,
        lane_args=lane_args,
        lane_budget=lane_budget,
        stage2_bs_path=lane_stage2_bs_path,
        warm_start_source=warm_start_source,
        lane_contract=lane_contract,
        output_root=output_root / "lanes" / lane_spec.lane_id,
    )


def _warm_start_path_from_lane_record(
    lane_record: FrontierLaneRecord,
) -> Path | None:
    if lane_record.result_source != "final" or lane_record.results_path is None:
        return None
    results_path = Path(lane_record.results_path)
    return results_path.with_name("biot_savart_opt.json")


def _lane_constraint_mode(lane_spec: FrontierLaneSpec) -> str:
    if lane_spec.scalarization_type == FRONTIER_REFERENCE_MODE_EPSILON:
        return "frontier_epsilon_constraint_v1"
    if lane_spec.scalarization_type == FRONTIER_REFERENCE_MODE_ACHIEVEMENT:
        return "frontier_achievement_chebyshev_v1"
    return "frontier_v2_single_lane_contract"
