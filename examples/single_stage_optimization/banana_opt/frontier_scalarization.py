from __future__ import annotations

"""Frontier scalarization contracts and lane-spec generation.

The legacy shared reference mode is a local iota/volume reward-share sweep. It
does not sweep QA error or Boozer residual reference directions; use the
achievement full-simplex mode for generated 4-objective reference directions.
"""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np

from alm_utils import augmented_inequality_objective
from .frontier_contracts import EpsilonThresholds
from .objective_gradients import objective_gradient as _objective_gradient
from .search_evaluation import annotate_search_evaluation_finiteness

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
# v2 mandates ``frontier_reference_qa_scale`` / ``frontier_reference_boozer_scale``
# alongside the existing reference targets so the Chebyshev gradient and
# epsilon excess penalty stop conflating scale with target. There is no
# back-compat path; v1 specs raise at load.
FRONTIER_ACHIEVEMENT_SPEC_SCHEMA_VERSION = "frontier_achievement_spec_v2"
_REFERENCE_METRIC_FLOOR = 1.0e-6
_TRUST_THRESHOLD_FLOOR = 1.0e-5
# Positive floor keeps Chebyshev divisions finite for pure-axis directions.
# The flooring breaks simplex unit-sum, so consumers must renormalize via
# ``_floor_and_normalize_simplex_weights`` before persisting weights.
_WEIGHT_FLOOR = 1.0e-12


def _floor_and_normalize_simplex_weights(
    direction: tuple[float, ...],
) -> tuple[float, ...]:
    """Apply ``_WEIGHT_FLOOR`` per component and renormalize to unit sum.

    Pure-axis Das-Dennis directions contain zeros. Without flooring the
    Chebyshev gradient divides by zero; without renormalization the
    flooring breaks the documented ``Σwᵢ = 1`` lane-payload invariant.
    """
    floored = tuple(max(component, _WEIGHT_FLOOR) for component in direction)
    total = sum(floored)
    return tuple(weight / total for weight in floored)
_DEFAULT_CHEBYSHEV_RHO = 1.0e-3
_DEFAULT_CHEBYSHEV_SHARPNESS = 12.0
_STRICT_TOP_LEVEL_KEYS = frozenset({"schema_version", "lanes"})
_REFERENCE_METRIC_KEYS = frozenset(
    {"iota", "volume", "qa_error", "boozer_residual"}
)
_EPSILON_METRIC_KEYS = frozenset({"qa_error", "boozer_residual"})


@dataclass(frozen=True, slots=True)
class FrontierLaneSpec:
    lane_id: str
    scalarization_type: str
    scalarization_params: dict[str, float]
    iotas_weight: float
    frontier_volume_weight: float
    res_weight: float
    lane_budget: int | None

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_json_dict(
        cls,
        payload: dict[str, object],
    ) -> "FrontierLaneSpec":
        scalarization_params_payload = payload.get("scalarization_params", {})
        return cls(
            lane_id=str(payload["lane_id"]),
            scalarization_type=str(payload["scalarization_type"]),
            scalarization_params={
                str(key): float(value)
                for key, value in scalarization_params_payload.items()
            },
            iotas_weight=float(payload["iotas_weight"]),
            frontier_volume_weight=float(payload["frontier_volume_weight"]),
            res_weight=float(payload["res_weight"]),
            lane_budget=None
            if payload.get("lane_budget") is None
            else int(payload["lane_budget"]),
        )


@dataclass(frozen=True, slots=True)
class _FrontierLaneSpecRequest:
    num_lanes: int
    iotas_weight: float
    frontier_volume_weight: float | None
    res_weight: float
    lane_budget: int | None
    stage2_results: Mapping[str, object] | None
    reference_points_file: str | None
    epsilon_spec_file: str | None
    full_simplex_partitions: int | None


_FrontierLaneSpecGenerator = Callable[
    [_FrontierLaneSpecRequest],
    list[FrontierLaneSpec],
]


def generate_multilane_local_specs(
    *,
    num_lanes: int,
    iotas_weight: float,
    frontier_volume_weight: float | None,
    res_weight: float,
    lane_budget: int | None,
) -> list[FrontierLaneSpec]:
    """Generate the legacy local iota/volume share sweep.

    This mode varies only the iota and volume reward shares on the clamped
    interval [0.2, 0.8]. QS and Boozer residual weights are not swept here; use
    achievement_chebyshev_full_simplex_v1 for 4-objective reference directions.
    """
    if num_lanes <= 0:
        raise ValueError("--frontier-num-lanes must be positive")

    base_volume_weight = (
        float(iotas_weight)
        if frontier_volume_weight is None
        else float(frontier_volume_weight)
    )
    total_reward_weight = float(iotas_weight) + base_volume_weight
    if num_lanes == 1:
        shares = [0.5]
    else:
        min_share = 0.2
        max_share = 0.8
        step = (max_share - min_share) / float(num_lanes - 1)
        shares = [min_share + step * index for index in range(num_lanes)]

    lane_specs: list[FrontierLaneSpec] = []
    for index, iota_share in enumerate(shares):
        volume_share = 1.0 - iota_share
        lane_specs.append(
            FrontierLaneSpec(
                lane_id=f"lane_{index + 1:02d}",
                scalarization_type="weight_schedule_v1",
                scalarization_params={
                    "iota_share": float(iota_share),
                    "volume_share": float(volume_share),
                },
                iotas_weight=total_reward_weight * float(iota_share),
                frontier_volume_weight=total_reward_weight * float(volume_share),
                res_weight=float(res_weight),
                lane_budget=lane_budget,
            )
        )
    return lane_specs


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
    poloidal_extent_weight=0.0,
    width_weight=0.0,
    selfint_weight=0.0,
    hardware_keepout_weight=0.0,
    vessel_keepout_weight=0.0,
    available_envelope_reward_weight=0.0,
    hardware_sdf_free_space_reward_weight=0.0,
    msc_weight=0.0,
    arclen_weight=0.0,
    link_weight=0.0,
    force_weight=0.0,
    magnetic_well_weight=0.0,
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

    if frontier_goal_config.scalarization_type == FRONTIER_REFERENCE_MODE_ACHIEVEMENT:
        chebyshev_eval = _frontier_chebyshev_goal(annotated, frontier_goal_config)
        replacement_total = float(chebyshev_eval["frontier_scalarization_total"])
        replacement_grad = np.asarray(
            chebyshev_eval["frontier_scalarization_grad"],
            dtype=float,
        )
        annotated.update(chebyshev_eval)
    elif frontier_goal_config.scalarization_type == FRONTIER_REFERENCE_MODE_EPSILON:
        # Hybrid epsilon method: keep the weighted base objective and add a fixed
        # quadratic constraint penalty rather than replacing the objective by pure f_k.
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
        # SSOT: both ALM and non-ALM branches sum the full geometry-penalty
        # bundle (length, cc, cs, curvature, poloidal_extent, width,
        # self-intersection) into
        # ``base_total`` / ``physics_total`` so cross-formulation comparisons
        # operate on the same physical denominator.
        penalty_total, penalty_grad = _frontier_penalty_geometry_total_grad(
            annotated,
            length_weight=length_weight,
            cc_weight=cc_weight,
            cs_weight=cs_weight,
            curvature_weight=curvature_weight,
            poloidal_extent_weight=poloidal_extent_weight,
            width_weight=width_weight,
            selfint_weight=selfint_weight,
            hardware_keepout_weight=hardware_keepout_weight,
            vessel_keepout_weight=vessel_keepout_weight,
            available_envelope_reward_weight=available_envelope_reward_weight,
            hardware_sdf_free_space_reward_weight=(
                hardware_sdf_free_space_reward_weight
            ),
            msc_weight=msc_weight,
            arclen_weight=arclen_weight,
            link_weight=link_weight,
            force_weight=force_weight,
            magnetic_well_weight=magnetic_well_weight,
        )
        if "constraint_values" in annotated and "constraint_grads" in annotated:
            if alm_multipliers is None or alm_penalty is None:
                raise ValueError(
                    "ALM frontier scalarization override requires explicit multipliers and penalty"
                )
            base_total = float(penalty_total) + float(replacement_total)
            base_grad = penalty_grad + replacement_grad
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
            annotated["total"] = float(penalty_total) + float(replacement_total)
            annotated["grad"] = penalty_grad + replacement_grad
    return annotate_search_evaluation_finiteness(annotated)


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
    poloidal_extent_weight=0.0,
    width_weight=0.0,
    selfint_weight=0.0,
    hardware_keepout_weight=0.0,
    vessel_keepout_weight=0.0,
    available_envelope_reward_weight=0.0,
    hardware_sdf_free_space_reward_weight=0.0,
    msc_weight=0.0,
    arclen_weight=0.0,
    link_weight=0.0,
    force_weight=0.0,
    magnetic_well_weight=0.0,
):
    total = (
        float(length_weight) * float(objective_eval["J_len"])
        + float(cc_weight) * float(objective_eval["J_cc"])
        + float(cs_weight) * float(objective_eval["J_cs"])
        + float(curvature_weight) * float(objective_eval["J_curvature"])
    )
    grad = (
        float(length_weight) * np.asarray(objective_eval["dJ_len"], dtype=float)
        + float(cc_weight) * np.asarray(objective_eval["dJ_cc"], dtype=float)
        + float(cs_weight) * np.asarray(objective_eval["dJ_cs"], dtype=float)
        + float(curvature_weight)
        * np.asarray(objective_eval["dJ_curvature"], dtype=float)
        + float(poloidal_extent_weight)
        * np.asarray(objective_eval.get("dJ_poloidal_extent", 0.0), dtype=float)
    )
    total += float(poloidal_extent_weight) * float(
        objective_eval.get("J_poloidal_extent", 0.0)
    )
    if float(width_weight) != 0.0:
        width_total, width_grad = _frontier_width_penalty_total_grad(objective_eval)
        total += float(width_weight) * width_total
        grad = grad + float(width_weight) * width_grad
    if float(selfint_weight) != 0.0:
        total += float(selfint_weight) * float(objective_eval["J_self_intersect"])
        grad = grad + float(selfint_weight) * np.asarray(
            objective_eval["dJ_self_intersect"],
            dtype=float,
        )
    if float(hardware_keepout_weight) != 0.0:
        total += float(hardware_keepout_weight) * float(
            objective_eval["J_hardware_keepout"]
        )
        grad = grad + float(hardware_keepout_weight) * np.asarray(
            objective_eval["dJ_hardware_keepout"],
            dtype=float,
        )
    if float(vessel_keepout_weight) != 0.0:
        total += float(vessel_keepout_weight) * float(
            objective_eval["J_vessel_keepout"]
        )
        grad = grad + float(vessel_keepout_weight) * np.asarray(
            objective_eval["dJ_vessel_keepout"],
            dtype=float,
        )
    if float(available_envelope_reward_weight) != 0.0:
        total += float(available_envelope_reward_weight) * float(
            objective_eval["J_available_envelope_reward"]
        )
        grad = grad + float(available_envelope_reward_weight) * np.asarray(
            objective_eval["dJ_available_envelope_reward"],
            dtype=float,
        )
    if float(hardware_sdf_free_space_reward_weight) != 0.0:
        total += float(hardware_sdf_free_space_reward_weight) * float(
            objective_eval["J_hardware_sdf_free_space_reward"]
        )
        grad = grad + float(hardware_sdf_free_space_reward_weight) * np.asarray(
            objective_eval["dJ_hardware_sdf_free_space_reward"],
            dtype=float,
        )
    # Opt-in SIMSOPT coil regularizers (guarded by weight so default runs read no
    # extra diagnostics and stay byte-identical with the prior frontier objective).
    if float(msc_weight) != 0.0:
        total += float(msc_weight) * float(objective_eval["J_msc"])
        grad = grad + float(msc_weight) * np.asarray(
            objective_eval["dJ_msc"], dtype=float
        )
    if float(arclen_weight) != 0.0:
        total += float(arclen_weight) * float(objective_eval["J_arclen"])
        grad = grad + float(arclen_weight) * np.asarray(
            objective_eval["dJ_arclen"], dtype=float
        )
    if float(link_weight) != 0.0:
        total += float(link_weight) * float(objective_eval["J_link"])
        grad = grad + float(link_weight) * np.asarray(
            objective_eval["dJ_link"], dtype=float
        )
    if float(force_weight) != 0.0:
        total += float(force_weight) * float(objective_eval["J_coil_force"])
        grad = grad + float(force_weight) * np.asarray(
            objective_eval["dJ_coil_force"], dtype=float
        )
    if float(magnetic_well_weight) != 0.0:
        total += float(magnetic_well_weight) * float(
            objective_eval["J_magnetic_well"]
        )
        grad = grad + float(magnetic_well_weight) * np.asarray(
            objective_eval["dJ_magnetic_well"], dtype=float
        )
    return float(total), grad


def _frontier_width_penalty_total_grad(objective_eval):
    width = float(objective_eval["J_coil_width"])
    width_grad = np.asarray(objective_eval["dJ_coil_width"], dtype=float)
    min_threshold = float(objective_eval["coil_width_min_threshold"])
    max_threshold = float(objective_eval["coil_width_max_threshold"])
    min_hinge = min(width - min_threshold, 0.0)
    max_hinge = max(width - max_threshold, 0.0)
    total = 0.5 * min_hinge * min_hinge + 0.5 * max_hinge * max_hinge
    grad = (min_hinge + max_hinge) * width_grad
    return float(total), grad


def _frontier_chebyshev_goal(objective_eval, frontier_goal_config):
    deltas = np.asarray(
        [
            frontier_goal_config.chebyshev_weight_iota
            * (
                (
                    frontier_goal_config.iota_reference
                    - float(objective_eval["J_iota_metric"])
                )
                / frontier_goal_config.iota_scale
            ),
            frontier_goal_config.chebyshev_weight_volume
            * (
                (
                    frontier_goal_config.volume_reference
                    - float(objective_eval["J_volume_metric"])
                )
                / frontier_goal_config.volume_scale
            ),
            frontier_goal_config.chebyshev_weight_qa
            * (
                (float(objective_eval["J_QS"]) - frontier_goal_config.qs_reference)
                / frontier_goal_config.qs_scale
            ),
            frontier_goal_config.chebyshev_weight_boozer
            * (
                (
                    float(objective_eval["J_Boozer"])
                    - frontier_goal_config.boozer_reference
                )
                / frontier_goal_config.boozer_scale
            ),
        ],
        dtype=float,
    )
    sharpness = float(frontier_goal_config.chebyshev_sharpness)
    if not (math.isfinite(sharpness) and sharpness > 0.0):
        raise ValueError(
            "frontier_chebyshev_sharpness must be a positive finite float, "
            f"got {sharpness!r}"
        )
    max_delta = float(np.max(deltas))
    exp_shifted = np.exp(sharpness * (deltas - max_delta))
    sum_exp = float(np.sum(exp_shifted))
    softmax_weights = exp_shifted / sum_exp
    chebyshev_total = (
        max_delta
        + np.log(sum_exp) / sharpness
        + frontier_goal_config.chebyshev_rho * float(np.sum(deltas))
    )
    coeffs = softmax_weights + frontier_goal_config.chebyshev_rho
    chebyshev_grad = (
        (
            -float(coeffs[0])
            * float(frontier_goal_config.chebyshev_weight_iota)
            / float(frontier_goal_config.iota_scale)
        )
        * np.asarray(objective_eval["dJ_iota_metric"], dtype=float)
        + (
            -float(coeffs[1])
            * float(frontier_goal_config.chebyshev_weight_volume)
            / float(frontier_goal_config.volume_scale)
        )
        * np.asarray(objective_eval["dJ_volume_metric"], dtype=float)
        + (
            float(coeffs[2])
            * float(frontier_goal_config.chebyshev_weight_qa)
            / float(frontier_goal_config.qs_scale)
        )
        * np.asarray(objective_eval["dJ_QS"], dtype=float)
        + (
            float(coeffs[3])
            * float(frontier_goal_config.chebyshev_weight_boozer)
            / float(frontier_goal_config.boozer_scale)
        )
        * np.asarray(objective_eval["dJ_Boozer"], dtype=float)
    )
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
    thresholds = EpsilonThresholds.from_goal_config(frontier_goal_config)
    if thresholds.qa_max is not None:
        # qs_scale already enforces ``_REFERENCE_METRIC_FLOOR`` at config-build
        # time, so the historical ``max(..., 1e-6)`` clamp is redundant.
        epsilon_penalties["qa_error"] = _frontier_excess_penalty(
            objective_eval["J_QS"],
            objective_eval["dJ_QS"],
            threshold=thresholds.qa_max,
            scale=frontier_goal_config.qs_scale,
            penalty_weight=frontier_goal_config.epsilon_penalty_weight,
        )
    if thresholds.boozer_max is not None:
        epsilon_penalties["boozer_residual"] = _frontier_excess_penalty(
            objective_eval["J_Boozer"],
            objective_eval["dJ_Boozer"],
            threshold=thresholds.boozer_max,
            scale=frontier_goal_config.boozer_scale,
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
    observed_schema = payload.get("schema_version")
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
    default_iota_weight = float(default_iotas_weight)
    default_volume_weight = (
        default_iota_weight
        if default_frontier_volume_weight is None
        else float(default_frontier_volume_weight)
    )
    explicit_iota_weight = _optional_float(lane_payload, "iotas_weight")
    explicit_volume_weight = _optional_float(
        lane_payload,
        "frontier_volume_weight",
    )
    if explicit_iota_weight is not None or explicit_volume_weight is not None:
        iota_weight = (
            default_iota_weight
            if explicit_iota_weight is None
            else explicit_iota_weight
        )
        volume_weight = (
            default_volume_weight
            if explicit_volume_weight is None
            else explicit_volume_weight
        )
    else:
        iota_share = _optional_float(lane_payload, "iota_share")
        volume_share = _optional_float(lane_payload, "volume_share")
        if iota_share is None and volume_share is None:
            iota_weight = default_iota_weight
            volume_weight = default_volume_weight
        else:
            if iota_share is None or volume_share is None:
                raise ValueError(
                    "lane payload must provide both iota_share and volume_share"
                )
            total_reward = default_iota_weight + default_volume_weight
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
        if value is None:
            continue
        coerced = float(value)
        # Stage-2 may emit NaN when a surface solve does not converge. NaN
        # propagates through `max(NaN, floor) → NaN` into the runtime
        # Chebyshev scalarization and silently corrupts every lane spec
        # that depends on the reference vector.
        if not math.isfinite(coerced):
            continue
        metrics[metric_name] = coerced
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
            params["frontier_reference_qa"] = max(
                reference_point["qa_error"], _REFERENCE_METRIC_FLOOR
            )
        if "boozer_residual" in reference_point:
            params["frontier_reference_boozer"] = max(
                reference_point["boozer_residual"],
                _REFERENCE_METRIC_FLOOR,
            )
    for payload_key, minimum in (
        ("frontier_reference_iota", None),
        ("frontier_reference_iota_scale", _REFERENCE_METRIC_FLOOR),
        ("frontier_reference_volume", None),
        ("frontier_reference_volume_scale", _REFERENCE_METRIC_FLOOR),
        ("frontier_reference_qa", _REFERENCE_METRIC_FLOOR),
        ("frontier_reference_qa_scale", _REFERENCE_METRIC_FLOOR),
        ("frontier_reference_boozer", _REFERENCE_METRIC_FLOOR),
        ("frontier_reference_boozer_scale", _REFERENCE_METRIC_FLOOR),
        ("frontier_boozer_trust_threshold", _TRUST_THRESHOLD_FLOOR),
        ("frontier_boozer_trust_penalty_scale", _REFERENCE_METRIC_FLOOR),
        ("frontier_chebyshev_sharpness", None),
        ("frontier_epsilon_penalty_weight", 0.0),
    ):
        value = _optional_float(lane_payload, payload_key)
        if value is None:
            continue
        params[payload_key] = max(value, minimum) if minimum is not None else value
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
    """Emit the full Das-Dennis reference-direction family for a partition count.

    When ``partitions`` is provided, the full Das-Dennis family for that
    partition count is emitted (``requested_num_directions`` is ignored).

    When ``partitions`` is omitted, ``requested_num_directions`` must equal
    ``C(p + n - 1, n - 1)`` for some partition ``p``. Selection-by-rounding
    over a larger family is no longer permitted because it produces
    non-uniform geometric coverage.
    """
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
    if (
        _direction_count(n_dim=n_dim, partitions=resolved_partitions)
        != requested_num_directions
    ):
        smaller_count = _direction_count(
            n_dim=n_dim,
            partitions=resolved_partitions - 1,
        )
        larger_count = _direction_count(
            n_dim=n_dim,
            partitions=resolved_partitions,
        )
        raise ValueError(
            f"--frontier-num-lanes={requested_num_directions} does not match "
            f"any Das-Dennis reference-direction count for n_dim={n_dim}. "
            "Selection-by-rounding produces non-uniform geometric coverage "
            "and is no longer permitted; supply --frontier-full-simplex-partitions "
            "explicitly to pick a partition. Nearest valid counts: "
            f"{smaller_count} (partitions={resolved_partitions - 1}) or "
            f"{larger_count} (partitions={resolved_partitions})."
        )
    return _das_dennis_reference_directions(
        n_dim=n_dim,
        partitions=resolved_partitions,
    )


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
        # v2 requires explicit qa/boozer scales so the Chebyshev gradient and
        # epsilon excess penalty stop conflating scale with target. v1 specs
        # are rejected at the schema_version gate above; specs that pass the
        # gate but omit the scale keys are caught here.
        for required_scale_key in (
            "frontier_reference_qa_scale",
            "frontier_reference_boozer_scale",
        ):
            if required_scale_key not in scalarization_params:
                raise ValueError(
                    f"achievement/Chebyshev lane {index + 1} is missing "
                    f"required key {required_scale_key!r}"
                )
        metric_weights = _resolve_metric_weights(lane_payload)
        scalarization_params.update(
            {
                "frontier_chebyshev_rho": max(
                    0.0,
                    _DEFAULT_CHEBYSHEV_RHO
                    if _optional_float(lane_payload, "rho") is None
                    else float(_optional_float(lane_payload, "rho")),
                ),
                "frontier_chebyshev_sharpness": float(
                    scalarization_params.get(
                        "frontier_chebyshev_sharpness",
                        _DEFAULT_CHEBYSHEV_SHARPNESS,
                    )
                )
                if _optional_float(lane_payload, "sharpness") is None
                else float(_optional_float(lane_payload, "sharpness")),
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
    # The 0.25 factor matches the existing iota_scale heuristic in
    # ``build_frontier_goal_config`` and gives a Chebyshev delta range of
    # roughly ``±4`` for a metric drift of one reference unit, which keeps the
    # log-sum-exp smoothing well-conditioned without saturating the softmax.
    qa_scale = max(
        float(seed_reference_metrics["qa_error"]) * 0.25,
        _REFERENCE_METRIC_FLOOR,
    )
    boozer_scale = max(
        float(seed_reference_metrics["boozer_residual"]) * 0.25,
        _REFERENCE_METRIC_FLOOR,
    )
    lane_specs: list[FrontierLaneSpec] = []
    for index, direction in enumerate(directions):
        normalized_weights = _floor_and_normalize_simplex_weights(direction)
        lane_specs.append(
            FrontierLaneSpec(
                lane_id=f"simplex_{index + 1:02d}",
                scalarization_type=FRONTIER_REFERENCE_MODE_ACHIEVEMENT,
                scalarization_params={
                    "frontier_reference_iota": float(seed_reference_metrics["iota"]),
                    "frontier_reference_volume": float(
                        seed_reference_metrics["volume"]
                    ),
                    "frontier_reference_qa": max(
                        float(seed_reference_metrics["qa_error"]),
                        _REFERENCE_METRIC_FLOOR,
                    ),
                    "frontier_reference_qa_scale": qa_scale,
                    "frontier_reference_boozer": max(
                        float(seed_reference_metrics["boozer_residual"]),
                        _REFERENCE_METRIC_FLOOR,
                    ),
                    "frontier_reference_boozer_scale": boozer_scale,
                    "frontier_chebyshev_rho": _DEFAULT_CHEBYSHEV_RHO,
                    "frontier_chebyshev_sharpness": _DEFAULT_CHEBYSHEV_SHARPNESS,
                    "frontier_chebyshev_weight_iota": normalized_weights[0],
                    "frontier_chebyshev_weight_volume": normalized_weights[1],
                    "frontier_chebyshev_weight_qa": normalized_weights[2],
                    "frontier_chebyshev_weight_boozer": normalized_weights[3],
                },
                iotas_weight=float(default_iotas_weight),
                frontier_volume_weight=default_volume_weight,
                res_weight=float(default_res_weight),
                lane_budget=default_lane_budget,
            )
        )
    return lane_specs


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
            scalarization_params.update(
                EpsilonThresholds(qa_max=float(qa_epsilon)).as_payload()
            )
            scalarization_params.setdefault(
                "frontier_reference_qa",
                max(float(qa_epsilon), _REFERENCE_METRIC_FLOOR),
            )
        boozer_epsilon = epsilon_constraints.get("boozer_residual")
        if boozer_epsilon is not None:
            scalarization_params.update(
                EpsilonThresholds(boozer_max=float(boozer_epsilon)).as_payload()
            )
            scalarization_params.setdefault(
                "frontier_reference_boozer",
                max(float(boozer_epsilon), _REFERENCE_METRIC_FLOOR),
            )
            scalarization_params.setdefault(
                "frontier_boozer_trust_threshold",
                # This intentionally ties the trust-residual cap to the epsilon cap:
                # both penalties apply additively when Boozer exceeds the shared limit.
                max(float(boozer_epsilon), _TRUST_THRESHOLD_FLOOR),
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
                max(float(reference_point["qa_error"]), _REFERENCE_METRIC_FLOOR),
            )
        if "boozer_residual" in reference_point:
            scalarization_params.setdefault(
                "frontier_reference_boozer",
                max(float(reference_point["boozer_residual"]), _REFERENCE_METRIC_FLOOR),
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


def _shared_lane_specs_from_request(
    request: _FrontierLaneSpecRequest,
) -> list[FrontierLaneSpec]:
    return generate_multilane_local_specs(
        num_lanes=request.num_lanes,
        iotas_weight=request.iotas_weight,
        frontier_volume_weight=request.frontier_volume_weight,
        res_weight=request.res_weight,
        lane_budget=request.lane_budget,
    )


def _reference_point_lane_specs_from_request(
    request: _FrontierLaneSpecRequest,
) -> list[FrontierLaneSpec]:
    if request.reference_points_file is None:
        raise ValueError(
            "--frontier-reference-points-file is required for "
            f"{FRONTIER_REFERENCE_MODE_REFERENCE_POINTS}"
        )
    return _reference_point_lane_specs(
        path=request.reference_points_file,
        default_iotas_weight=request.iotas_weight,
        default_frontier_volume_weight=request.frontier_volume_weight,
        default_res_weight=request.res_weight,
        default_lane_budget=request.lane_budget,
    )


def _epsilon_lane_specs_from_request(
    request: _FrontierLaneSpecRequest,
) -> list[FrontierLaneSpec]:
    if request.epsilon_spec_file is None:
        raise ValueError(
            "--frontier-epsilon-spec-file is required for "
            f"{FRONTIER_REFERENCE_MODE_EPSILON}"
        )
    return _epsilon_constraint_lane_specs(
        path=request.epsilon_spec_file,
        default_iotas_weight=request.iotas_weight,
        default_frontier_volume_weight=request.frontier_volume_weight,
        default_res_weight=request.res_weight,
        default_lane_budget=request.lane_budget,
        seed_reference_metrics=_seed_reference_metrics(request.stage2_results),
    )


def _achievement_lane_specs_from_request(
    request: _FrontierLaneSpecRequest,
) -> list[FrontierLaneSpec]:
    if request.reference_points_file is None:
        raise ValueError(
            "--frontier-reference-points-file is required for "
            f"{FRONTIER_REFERENCE_MODE_ACHIEVEMENT}"
        )
    return _achievement_chebyshev_lane_specs(
        path=request.reference_points_file,
        default_iotas_weight=request.iotas_weight,
        default_frontier_volume_weight=request.frontier_volume_weight,
        default_res_weight=request.res_weight,
        default_lane_budget=request.lane_budget,
    )


def _achievement_full_simplex_lane_specs_from_request(
    request: _FrontierLaneSpecRequest,
) -> list[FrontierLaneSpec]:
    return _achievement_full_simplex_lane_specs(
        num_lanes=request.num_lanes,
        default_iotas_weight=request.iotas_weight,
        default_frontier_volume_weight=request.frontier_volume_weight,
        default_res_weight=request.res_weight,
        default_lane_budget=request.lane_budget,
        seed_reference_metrics=_seed_reference_metrics(request.stage2_results),
        full_simplex_partitions=request.full_simplex_partitions,
    )


_FRONTIER_LANE_SPEC_GENERATORS: Mapping[str, _FrontierLaneSpecGenerator] = (
    MappingProxyType(
        {
            FRONTIER_REFERENCE_MODE_SHARED: _shared_lane_specs_from_request,
            FRONTIER_REFERENCE_MODE_REFERENCE_POINTS: (
                _reference_point_lane_specs_from_request
            ),
            FRONTIER_REFERENCE_MODE_EPSILON: _epsilon_lane_specs_from_request,
            FRONTIER_REFERENCE_MODE_ACHIEVEMENT: _achievement_lane_specs_from_request,
            FRONTIER_REFERENCE_MODE_ACHIEVEMENT_FULL_SIMPLEX: (
                _achievement_full_simplex_lane_specs_from_request
            ),
        }
    )
)


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
    generator = _FRONTIER_LANE_SPEC_GENERATORS.get(reference_mode)
    if generator is None:
        raise ValueError(f"Unsupported frontier reference mode {reference_mode!r}")
    return generator(
        _FrontierLaneSpecRequest(
            num_lanes=num_lanes,
            iotas_weight=iotas_weight,
            frontier_volume_weight=frontier_volume_weight,
            res_weight=res_weight,
            lane_budget=lane_budget,
            stage2_results=stage2_results,
            reference_points_file=reference_points_file,
            epsilon_spec_file=epsilon_spec_file,
            full_simplex_partitions=full_simplex_partitions,
        )
    )
