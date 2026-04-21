from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

from .frontier_engine_multilane_local import (
    FrontierLaneSpec,
    generate_multilane_local_specs,
)

FRONTIER_REFERENCE_MODE_SHARED = "shared_seed_relative_frontier_v2"
FRONTIER_REFERENCE_MODE_REFERENCE_POINTS = "reference_point_sweep_v1"
FRONTIER_REFERENCE_MODE_EPSILON = "epsilon_constraint_sweep_v1"
FRONTIER_REFERENCE_MODE_ACHIEVEMENT = "achievement_chebyshev_sweep_v1"
FRONTIER_REFERENCE_MODE_ACHIEVEMENT_FULL_SIMPLEX = (
    "achievement_chebyshev_full_simplex_v1"
)
SUPPORTED_FRONTIER_REFERENCE_MODES = (
    FRONTIER_REFERENCE_MODE_SHARED,
    FRONTIER_REFERENCE_MODE_REFERENCE_POINTS,
    FRONTIER_REFERENCE_MODE_EPSILON,
    FRONTIER_REFERENCE_MODE_ACHIEVEMENT,
    FRONTIER_REFERENCE_MODE_ACHIEVEMENT_FULL_SIMPLEX,
)
FRONTIER_REFERENCE_POINTS_SCHEMA_VERSION = "frontier_reference_points_v1"
FRONTIER_EPSILON_SPEC_SCHEMA_VERSION = "frontier_epsilon_spec_v1"
FRONTIER_ACHIEVEMENT_SPEC_SCHEMA_VERSION = "frontier_achievement_spec_v1"
_STRICT_TOP_LEVEL_KEYS = frozenset({"schema_version", "SCHEMA_VERSION", "lanes"})
_REFERENCE_METRIC_KEYS = frozenset(
    {"iota", "volume", "qa_error", "boozer_residual"}
)
_EPSILON_METRIC_KEYS = frozenset({"qa_error", "boozer_residual"})


def frontier_scalarization_family(
    lane_specs: list[FrontierLaneSpec],
) -> str:
    families = sorted({lane.scalarization_type for lane in lane_specs})
    if not families:
        return "empty"
    if len(families) == 1:
        return families[0]
    return "mixed:" + ",".join(families)


def _read_json_payload(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _coerce_float_mapping(payload: Mapping[str, object]) -> dict[str, float]:
    return {str(key): float(value) for key, value in payload.items()}


def _optional_float(payload: Mapping[str, object], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    return float(value)


def _require_allowed_keys(
    payload: Mapping[str, object],
    *,
    allowed_keys: frozenset[str],
    context: str,
) -> None:
    unknown_keys = sorted(set(payload) - set(allowed_keys))
    if unknown_keys:
        raise ValueError(f"{context} contains unsupported keys: {unknown_keys}")


def _require_lane_entries(
    payload: Mapping[str, object],
    *,
    schema_version: str,
    path: str | Path,
) -> list[dict[str, object]]:
    _require_allowed_keys(
        payload,
        allowed_keys=_STRICT_TOP_LEVEL_KEYS,
        context=str(path),
    )
    observed_schema = payload.get("schema_version", payload.get("SCHEMA_VERSION"))
    if observed_schema != schema_version:
        raise ValueError(
            f"{path} must declare schema_version={schema_version!r}; "
            f"got {observed_schema!r}"
        )
    lanes_payload = payload.get("lanes")
    if not isinstance(lanes_payload, list) or not lanes_payload:
        raise ValueError(f"{path} must contain a non-empty 'lanes' list")
    lanes: list[dict[str, object]] = []
    for item in lanes_payload:
        if not isinstance(item, dict):
            raise ValueError(f"{path} lane entries must be JSON objects")
        lanes.append(dict(item))
    return lanes


def _total_reward_weight(
    iotas_weight: float,
    frontier_volume_weight: float | None,
) -> float:
    base_volume_weight = (
        float(iotas_weight)
        if frontier_volume_weight is None
        else float(frontier_volume_weight)
    )
    return float(iotas_weight) + base_volume_weight


def _resolve_reward_weights(
    lane_payload: Mapping[str, object],
    *,
    default_iotas_weight: float,
    default_frontier_volume_weight: float | None,
) -> tuple[float, float]:
    explicit_iota_weight = _optional_float(lane_payload, "iotas_weight")
    explicit_volume_weight = _optional_float(
        lane_payload,
        "frontier_volume_weight",
    )
    if explicit_iota_weight is not None or explicit_volume_weight is not None:
        iota_weight = (
            float(default_iotas_weight)
            if explicit_iota_weight is None
            else explicit_iota_weight
        )
        volume_weight = (
            float(default_iotas_weight)
            if explicit_volume_weight is None and default_frontier_volume_weight is None
            else (
                float(default_frontier_volume_weight)
                if explicit_volume_weight is None
                else explicit_volume_weight
            )
        )
    else:
        iota_share = _optional_float(lane_payload, "iota_share")
        volume_share = _optional_float(lane_payload, "volume_share")
        if iota_share is None and volume_share is None:
            iota_weight = float(default_iotas_weight)
            volume_weight = (
                float(default_iotas_weight)
                if default_frontier_volume_weight is None
                else float(default_frontier_volume_weight)
            )
        else:
            if iota_share is None or volume_share is None:
                raise ValueError(
                    "lane payload must provide both iota_share and volume_share"
                )
            total_reward = _total_reward_weight(
                default_iotas_weight,
                default_frontier_volume_weight,
            )
            iota_weight = total_reward * float(iota_share)
            volume_weight = total_reward * float(volume_share)
    if iota_weight < 0.0 or volume_weight < 0.0:
        raise ValueError("frontier lane reward weights must be non-negative")
    if iota_weight == 0.0 and volume_weight == 0.0:
        raise ValueError("frontier lane must keep at least one positive reward weight")
    return float(iota_weight), float(volume_weight)


def _resolve_lane_budget(
    lane_payload: Mapping[str, object],
    default_lane_budget: int | None,
) -> int | None:
    lane_budget = lane_payload.get("lane_budget")
    if lane_budget is None:
        return default_lane_budget
    return int(lane_budget)


def _seed_reference_metrics(
    stage2_results: Mapping[str, object] | None,
) -> dict[str, float]:
    if stage2_results is None:
        return {}
    metrics: dict[str, float] = {}
    field_map = {
        "FINAL_IOTA": "iota",
        "FINAL_VOLUME": "volume",
        "NONQS_RATIO": "qa_error",
        "BOOZER_RESIDUAL": "boozer_residual",
    }
    for results_key, metric_name in field_map.items():
        value = stage2_results.get(results_key)
        if value is not None:
            metrics[metric_name] = float(value)
    return metrics


def _reference_scalarization_params(
    lane_payload: Mapping[str, object],
) -> tuple[dict[str, float], dict[str, float] | None]:
    params: dict[str, float] = {}
    reference_point_payload = lane_payload.get("reference_point")
    reference_point = None
    if isinstance(reference_point_payload, Mapping):
        _require_allowed_keys(
            reference_point_payload,
            allowed_keys=_REFERENCE_METRIC_KEYS,
            context="reference_point",
        )
        reference_point = _coerce_float_mapping(reference_point_payload)
        if "iota" in reference_point:
            params["frontier_reference_iota"] = reference_point["iota"]
        if "volume" in reference_point:
            params["frontier_reference_volume"] = reference_point["volume"]
        if "qa_error" in reference_point:
            params["frontier_reference_qa"] = max(reference_point["qa_error"], 1e-6)
        if "boozer_residual" in reference_point:
            params["frontier_reference_boozer"] = max(
                reference_point["boozer_residual"],
                1e-6,
            )
    for payload_key, param_key, minimum in (
        ("frontier_reference_iota", "frontier_reference_iota", None),
        ("frontier_reference_iota_scale", "frontier_reference_iota_scale", 1e-6),
        ("frontier_reference_volume", "frontier_reference_volume", None),
        ("frontier_reference_volume_scale", "frontier_reference_volume_scale", 1e-6),
        ("frontier_reference_qa", "frontier_reference_qa", 1e-6),
        ("frontier_reference_boozer", "frontier_reference_boozer", 1e-6),
        ("frontier_boozer_trust_threshold", "frontier_boozer_trust_threshold", 1e-5),
        (
            "frontier_boozer_trust_penalty_scale",
            "frontier_boozer_trust_penalty_scale",
            1e-6,
        ),
        ("frontier_chebyshev_sharpness", "frontier_chebyshev_sharpness", 1.0e-12),
        (
            "frontier_epsilon_penalty_weight",
            "frontier_epsilon_penalty_weight",
            0.0,
        ),
    ):
        value = _optional_float(lane_payload, payload_key)
        if value is None:
            continue
        params[param_key] = max(value, minimum) if minimum is not None else value
    return params, reference_point


def _reference_point_lane_specs(
    *,
    path: str | Path,
    default_iotas_weight: float,
    default_frontier_volume_weight: float | None,
    default_res_weight: float,
    default_lane_budget: int | None,
) -> list[FrontierLaneSpec]:
    payload = _read_json_payload(path)
    lanes_payload = _require_lane_entries(
        payload,
        schema_version=FRONTIER_REFERENCE_POINTS_SCHEMA_VERSION,
        path=path,
    )
    lane_specs: list[FrontierLaneSpec] = []
    for index, lane_payload in enumerate(lanes_payload):
        scalarization_params, _ = _reference_scalarization_params(lane_payload)
        iota_weight, volume_weight = _resolve_reward_weights(
            lane_payload,
            default_iotas_weight=default_iotas_weight,
            default_frontier_volume_weight=default_frontier_volume_weight,
        )
        lane_specs.append(
            FrontierLaneSpec(
                lane_id=str(
                    lane_payload.get("lane_id", f"lane_{index + 1:02d}")
                ),
                scalarization_type=FRONTIER_REFERENCE_MODE_REFERENCE_POINTS,
                scalarization_params=scalarization_params,
                iotas_weight=iota_weight,
                frontier_volume_weight=volume_weight,
                res_weight=float(
                    lane_payload.get("res_weight", default_res_weight)
                ),
                lane_budget=_resolve_lane_budget(
                    lane_payload,
                    default_lane_budget,
                ),
            )
        )
    return lane_specs


def _das_dennis_integer_compositions(
    total: int,
    parts: int,
) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(total,)]
    compositions: list[tuple[int, ...]] = []
    for value in range(total + 1):
        for remainder in _das_dennis_integer_compositions(total - value, parts - 1):
            compositions.append((value, *remainder))
    return compositions


def _das_dennis_reference_directions(
    *,
    n_dim: int,
    partitions: int,
) -> list[tuple[float, ...]]:
    if partitions <= 0:
        raise ValueError("frontier full-simplex partitions must be positive")
    return [
        tuple(float(component) / float(partitions) for component in composition)
        for composition in _das_dennis_integer_compositions(partitions, n_dim)
    ]


def _direction_count(
    *,
    n_dim: int,
    partitions: int,
) -> int:
    return math.comb(partitions + n_dim - 1, n_dim - 1)


def _select_reference_directions(
    *,
    requested_num_directions: int,
    n_dim: int,
    partitions: int | None,
) -> list[tuple[float, ...]]:
    if requested_num_directions <= 0:
        raise ValueError("--frontier-num-lanes must be positive")
    if partitions is not None:
        return _das_dennis_reference_directions(
            n_dim=n_dim,
            partitions=partitions,
        )
    resolved_partitions = 1
    while _direction_count(
        n_dim=n_dim,
        partitions=resolved_partitions,
    ) < requested_num_directions:
        resolved_partitions += 1
    directions = _das_dennis_reference_directions(
        n_dim=n_dim,
        partitions=resolved_partitions,
    )
    if len(directions) <= requested_num_directions:
        return directions
    if requested_num_directions == 1:
        return [directions[0]]
    selected_indices = [
        round(index * (len(directions) - 1) / float(requested_num_directions - 1))
        for index in range(requested_num_directions)
    ]
    unique_indices: list[int] = []
    seen: set[int] = set()
    for index in selected_indices:
        if index in seen:
            continue
        unique_indices.append(index)
        seen.add(index)
    next_index = 0
    while len(unique_indices) < requested_num_directions:
        if next_index not in seen:
            unique_indices.append(next_index)
            seen.add(next_index)
        next_index += 1
    unique_indices.sort()
    return [directions[index] for index in unique_indices]


def generate_frontier_reference_directions(
    *,
    requested_num_directions: int,
    n_dim: int = 4,
    partitions: int | None = None,
) -> list[tuple[float, ...]]:
    return _select_reference_directions(
        requested_num_directions=requested_num_directions,
        n_dim=n_dim,
        partitions=partitions,
    )


def _resolve_metric_weights(
    lane_payload: Mapping[str, object],
) -> dict[str, float]:
    metric_weights_payload = lane_payload.get("metric_weights", {})
    if metric_weights_payload is None:
        metric_weights_payload = {}
    if not isinstance(metric_weights_payload, Mapping):
        raise ValueError("metric_weights must be a JSON object when provided")
    metric_weights = {
        "iota": 1.0,
        "volume": 1.0,
        "qa_error": 1.0,
        "boozer_residual": 1.0,
    }
    for metric_name in metric_weights:
        if metric_name in metric_weights_payload:
            metric_weights[metric_name] = float(metric_weights_payload[metric_name])
    if any(weight <= 0.0 for weight in metric_weights.values()):
        raise ValueError("achievement/Chebyshev metric weights must be positive")
    return metric_weights


def _achievement_chebyshev_lane_specs(
    *,
    path: str | Path,
    default_iotas_weight: float,
    default_frontier_volume_weight: float | None,
    default_res_weight: float,
    default_lane_budget: int | None,
) -> list[FrontierLaneSpec]:
    payload = _read_json_payload(path)
    lanes_payload = _require_lane_entries(
        payload,
        schema_version=FRONTIER_ACHIEVEMENT_SPEC_SCHEMA_VERSION,
        path=path,
    )
    lane_specs: list[FrontierLaneSpec] = []
    for index, lane_payload in enumerate(lanes_payload):
        scalarization_params, reference_point = _reference_scalarization_params(
            lane_payload
        )
        if reference_point is None:
            raise ValueError(
                "achievement/Chebyshev lanes require a reference_point"
            )
        metric_weights = _resolve_metric_weights(lane_payload)
        scalarization_params.update(
            {
                "frontier_chebyshev_rho": max(
                    0.0,
                    1.0e-3
                    if _optional_float(lane_payload, "rho") is None
                    else float(_optional_float(lane_payload, "rho")),
                ),
                "frontier_chebyshev_sharpness": max(
                    1.0e-12,
                    float(
                        scalarization_params.get("frontier_chebyshev_sharpness", 12.0)
                    )
                    if _optional_float(lane_payload, "sharpness") is None
                    else float(_optional_float(lane_payload, "sharpness")),
                ),
                "frontier_chebyshev_weight_iota": metric_weights["iota"],
                "frontier_chebyshev_weight_volume": metric_weights["volume"],
                "frontier_chebyshev_weight_qa": metric_weights["qa_error"],
                "frontier_chebyshev_weight_boozer": metric_weights["boozer_residual"],
            }
        )
        iota_weight, volume_weight = _resolve_reward_weights(
            lane_payload,
            default_iotas_weight=default_iotas_weight,
            default_frontier_volume_weight=default_frontier_volume_weight,
        )
        lane_specs.append(
            FrontierLaneSpec(
                lane_id=str(lane_payload.get("lane_id", f"lane_{index + 1:02d}")),
                scalarization_type=FRONTIER_REFERENCE_MODE_ACHIEVEMENT,
                scalarization_params=scalarization_params,
                iotas_weight=iota_weight,
                frontier_volume_weight=volume_weight,
                res_weight=float(lane_payload.get("res_weight", default_res_weight)),
                lane_budget=_resolve_lane_budget(
                    lane_payload,
                    default_lane_budget,
                ),
            )
        )
    return lane_specs


def _achievement_full_simplex_lane_specs(
    *,
    num_lanes: int,
    default_iotas_weight: float,
    default_frontier_volume_weight: float | None,
    default_res_weight: float,
    default_lane_budget: int | None,
    seed_reference_metrics: Mapping[str, float],
    full_simplex_partitions: int | None,
) -> list[FrontierLaneSpec]:
    required_metrics = ("iota", "volume", "qa_error", "boozer_residual")
    if any(metric_name not in seed_reference_metrics for metric_name in required_metrics):
        missing_metrics = [
            metric_name
            for metric_name in required_metrics
            if metric_name not in seed_reference_metrics
        ]
        raise ValueError(
            "full-simplex achievement mode requires seed reference metrics for "
            f"{missing_metrics}"
        )
    directions = _select_reference_directions(
        requested_num_directions=num_lanes,
        n_dim=4,
        partitions=full_simplex_partitions,
    )
    default_volume_weight = (
        float(default_iotas_weight)
        if default_frontier_volume_weight is None
        else float(default_frontier_volume_weight)
    )
    return [
        FrontierLaneSpec(
            lane_id=f"simplex_{index + 1:02d}",
            scalarization_type=FRONTIER_REFERENCE_MODE_ACHIEVEMENT,
            scalarization_params={
                "frontier_reference_iota": float(seed_reference_metrics["iota"]),
                "frontier_reference_volume": float(seed_reference_metrics["volume"]),
                "frontier_reference_qa": max(
                    float(seed_reference_metrics["qa_error"]),
                    1.0e-6,
                ),
                "frontier_reference_boozer": max(
                    float(seed_reference_metrics["boozer_residual"]),
                    1.0e-6,
                ),
                "frontier_chebyshev_rho": 1.0e-3,
                "frontier_chebyshev_sharpness": 12.0,
                "frontier_chebyshev_weight_iota": max(direction[0], 1.0e-12),
                "frontier_chebyshev_weight_volume": max(direction[1], 1.0e-12),
                "frontier_chebyshev_weight_qa": max(direction[2], 1.0e-12),
                "frontier_chebyshev_weight_boozer": max(direction[3], 1.0e-12),
            },
            iotas_weight=float(default_iotas_weight),
            frontier_volume_weight=default_volume_weight,
            res_weight=float(default_res_weight),
            lane_budget=default_lane_budget,
        )
        for index, direction in enumerate(directions)
    ]


def _epsilon_constraint_lane_specs(
    *,
    path: str | Path,
    default_iotas_weight: float,
    default_frontier_volume_weight: float | None,
    default_res_weight: float,
    default_lane_budget: int | None,
    seed_reference_metrics: Mapping[str, float],
) -> list[FrontierLaneSpec]:
    payload = _read_json_payload(path)
    lanes_payload = _require_lane_entries(
        payload,
        schema_version=FRONTIER_EPSILON_SPEC_SCHEMA_VERSION,
        path=path,
    )
    allowed_lane_keys = frozenset(
        {
            "lane_id",
            "objective",
            "epsilon_constraints",
            "reference_point",
            "iotas_weight",
            "frontier_volume_weight",
            "iota_share",
            "volume_share",
            "res_weight",
            "lane_budget",
            "frontier_reference_iota",
            "frontier_reference_iota_scale",
            "frontier_reference_volume",
            "frontier_reference_volume_scale",
            "frontier_reference_qa",
            "frontier_reference_boozer",
            "frontier_boozer_trust_threshold",
            "frontier_boozer_trust_penalty_scale",
            "frontier_epsilon_penalty_weight",
        }
    )
    total_reward = _total_reward_weight(
        default_iotas_weight,
        default_frontier_volume_weight,
    )
    lane_specs: list[FrontierLaneSpec] = []
    for index, lane_payload in enumerate(lanes_payload):
        _require_allowed_keys(
            lane_payload,
            allowed_keys=allowed_lane_keys,
            context=f"epsilon lane {index + 1}",
        )
        objective_name = str(lane_payload.get("objective", "iota"))
        if objective_name == "iota":
            iota_weight = total_reward
            volume_weight = 0.0
        elif objective_name == "volume":
            iota_weight = 0.0
            volume_weight = total_reward
        else:
            raise ValueError(
                "epsilon frontier lanes require objective='iota' or 'volume'"
            )
        scalarization_params, reference_point = _reference_scalarization_params(
            lane_payload
        )
        epsilon_payload = lane_payload.get("epsilon_constraints")
        if isinstance(epsilon_payload, Mapping):
            _require_allowed_keys(
                epsilon_payload,
                allowed_keys=_EPSILON_METRIC_KEYS,
                context=f"epsilon lane {index + 1} epsilon_constraints",
            )
            epsilon_constraints = _coerce_float_mapping(epsilon_payload)
        else:
            epsilon_constraints = {}
        epsilon_penalty_weight = _optional_float(
            lane_payload,
            "frontier_epsilon_penalty_weight",
        )
        if epsilon_penalty_weight is not None:
            scalarization_params["frontier_epsilon_penalty_weight"] = max(
                float(epsilon_penalty_weight),
                0.0,
            )
        qa_epsilon = epsilon_constraints.get("qa_error")
        if qa_epsilon is not None:
            scalarization_params["epsilon_constraint_qa_max"] = float(qa_epsilon)
            scalarization_params.setdefault(
                "frontier_reference_qa",
                max(float(qa_epsilon), 1e-6),
            )
        boozer_epsilon = epsilon_constraints.get("boozer_residual")
        if boozer_epsilon is not None:
            scalarization_params["epsilon_constraint_boozer_max"] = float(
                boozer_epsilon
            )
            scalarization_params.setdefault(
                "frontier_reference_boozer",
                max(float(boozer_epsilon), 1e-6),
            )
            scalarization_params.setdefault(
                "frontier_boozer_trust_threshold",
                max(float(boozer_epsilon), 1e-5),
            )
        if reference_point is None:
            reference_point = {
                metric_name: float(metric_value)
                for metric_name, metric_value in seed_reference_metrics.items()
                if metric_name in {"iota", "volume", "qa_error", "boozer_residual"}
            }
        if "iota" in reference_point:
            scalarization_params.setdefault(
                "frontier_reference_iota",
                float(reference_point["iota"]),
            )
        if "volume" in reference_point:
            scalarization_params.setdefault(
                "frontier_reference_volume",
                float(reference_point["volume"]),
            )
        if "qa_error" in reference_point:
            scalarization_params.setdefault(
                "frontier_reference_qa",
                max(float(reference_point["qa_error"]), 1e-6),
            )
        if "boozer_residual" in reference_point:
            scalarization_params.setdefault(
                "frontier_reference_boozer",
                max(float(reference_point["boozer_residual"]), 1e-6),
            )
        lane_specs.append(
            FrontierLaneSpec(
                lane_id=str(
                    lane_payload.get("lane_id", f"lane_{index + 1:02d}")
                ),
                scalarization_type=FRONTIER_REFERENCE_MODE_EPSILON,
                scalarization_params=scalarization_params,
                iotas_weight=iota_weight,
                frontier_volume_weight=volume_weight,
                res_weight=float(
                    lane_payload.get("res_weight", default_res_weight)
                ),
                lane_budget=_resolve_lane_budget(
                    lane_payload,
                    default_lane_budget,
                ),
            )
        )
    return lane_specs


def generate_frontier_lane_specs(
    *,
    reference_mode: str,
    num_lanes: int,
    iotas_weight: float,
    frontier_volume_weight: float | None,
    res_weight: float,
    lane_budget: int | None,
    stage2_results: Mapping[str, object] | None,
    reference_points_file: str | None,
    epsilon_spec_file: str | None,
    full_simplex_partitions: int | None = None,
) -> list[FrontierLaneSpec]:
    if reference_mode == FRONTIER_REFERENCE_MODE_SHARED:
        return generate_multilane_local_specs(
            num_lanes=num_lanes,
            iotas_weight=iotas_weight,
            frontier_volume_weight=frontier_volume_weight,
            res_weight=res_weight,
            lane_budget=lane_budget,
        )
    if reference_mode == FRONTIER_REFERENCE_MODE_REFERENCE_POINTS:
        if reference_points_file is None:
            raise ValueError(
                "--frontier-reference-points-file is required for "
                f"{FRONTIER_REFERENCE_MODE_REFERENCE_POINTS}"
            )
        return _reference_point_lane_specs(
            path=reference_points_file,
            default_iotas_weight=iotas_weight,
            default_frontier_volume_weight=frontier_volume_weight,
            default_res_weight=res_weight,
            default_lane_budget=lane_budget,
        )
    if reference_mode == FRONTIER_REFERENCE_MODE_EPSILON:
        if epsilon_spec_file is None:
            raise ValueError(
                "--frontier-epsilon-spec-file is required for "
                f"{FRONTIER_REFERENCE_MODE_EPSILON}"
            )
        return _epsilon_constraint_lane_specs(
            path=epsilon_spec_file,
            default_iotas_weight=iotas_weight,
            default_frontier_volume_weight=frontier_volume_weight,
            default_res_weight=res_weight,
            default_lane_budget=lane_budget,
            seed_reference_metrics=_seed_reference_metrics(stage2_results),
        )
    if reference_mode == FRONTIER_REFERENCE_MODE_ACHIEVEMENT:
        if reference_points_file is None:
            raise ValueError(
                "--frontier-reference-points-file is required for "
                f"{FRONTIER_REFERENCE_MODE_ACHIEVEMENT}"
            )
        return _achievement_chebyshev_lane_specs(
            path=reference_points_file,
            default_iotas_weight=iotas_weight,
            default_frontier_volume_weight=frontier_volume_weight,
            default_res_weight=res_weight,
            default_lane_budget=lane_budget,
        )
    if reference_mode == FRONTIER_REFERENCE_MODE_ACHIEVEMENT_FULL_SIMPLEX:
        return _achievement_full_simplex_lane_specs(
            num_lanes=num_lanes,
            default_iotas_weight=iotas_weight,
            default_frontier_volume_weight=frontier_volume_weight,
            default_res_weight=res_weight,
            default_lane_budget=lane_budget,
            seed_reference_metrics=_seed_reference_metrics(stage2_results),
            full_simplex_partitions=full_simplex_partitions,
        )
    raise ValueError(f"Unsupported frontier reference mode {reference_mode!r}")
