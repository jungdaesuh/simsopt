from __future__ import annotations

import math
import warnings
from typing import Mapping

import numpy as np

from .single_stage_geometry import (
    topology_gate_deficit,
    topology_gate_rejection_increment,
)
from .single_stage_search_policy import (
    HardwareSearchPolicy,
    SearchContext,
    decide_hardware_search_action,
)
from .search_evaluation import (
    FINITE_EPS,
    annotate_search_evaluation_finiteness,
)
from .topology.kam_birkhoff import KAM_FRACTION_SEMANTICS

_DEFAULT_FRONTIER_SEARCH_CONTRACT_PENALTY_SCALE = 4.0
# Conservative interim floor from the current known-good nested donor envelope:
# it makes the WBA fraction non-vacuous while calibration remains artifact-local.
DEFAULT_FRONTIER_INVARIANT_TORUS_MIN = 0.30
FRONTIER_INVARIANT_TORUS_MIN_RATIONALE = (
    "conservative_nonzero_interim_floor_from_known_good_nested_donors"
)
# Emitted (loud, capturable) when an explicit/env-configured WBA floor is <= 0 or non-finite
# and is therefore hard-clamped up to DEFAULT_FRONTIER_INVARIANT_TORUS_MIN. A configured 0.0
# would otherwise re-vacuate the binding invariant-torus certification gate (the deficit is
# max(min - fraction, 0), so a 0.0 floor certifies every nested-or-not config). This mirrors
# the calibrator's own clamp (frontier_kam_calibration.calibration_summary: recommended_raw
# <= 0 -> DEFAULT + VACUOUS_FRONTIER_INVARIANT_TORUS_WARNING), keeping the two surfaces SSOT.
CLAMPED_VACUOUS_FRONTIER_INVARIANT_TORUS_WARNING = (
    "configured frontier invariant-torus floor is <= 0 (vacuous); hard-clamping up to the "
    f"default nonzero floor {DEFAULT_FRONTIER_INVARIANT_TORUS_MIN:g} so the binding "
    "certification gate is not silently re-vacuated"
)
_DIAGNOSTIC_HARDWARE_RATIO_NAMES = frozenset(
    {
        "surface_vessel_min_dist",
        "surface_vessel_spacing",
        "surface_vessel_spacing_penalty",
        "plasma_vessel_min_dist",
        "plasma_vessel_min_dist_m",
        "plasma_vessel_spacing",
        "plasma_vessel_spacing_penalty",
        "lcfs_vessel_spacing",
    }
)


def _finite_float_or_none(value: object) -> float | None:
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return numeric


def default_frontier_invariant_torus_min() -> float:
    """Return the non-vacuous default WBA floor for frontier certification."""

    return DEFAULT_FRONTIER_INVARIANT_TORUS_MIN


def _clamp_configured_invariant_torus_min(value: object) -> float:
    """Coerce an explicit/env-configured WBA floor, hard-clamping <=0 / non-finite up.

    A configured floor of 0.0 (or negative / non-finite) re-vacuates the binding
    invariant-torus certification gate, so it is clamped up to the default nonzero floor
    with a loud warning -- mirroring the calibrator's recommended-floor clamp so the two
    surfaces stay SSOT. A finite value in (0, inf) is returned verbatim (the calibrator's
    own validation later rejects anything outside [0, 1]).
    """

    floor = float(value)
    if not math.isfinite(floor) or floor <= 0.0:
        warnings.warn(
            CLAMPED_VACUOUS_FRONTIER_INVARIANT_TORUS_WARNING,
            stacklevel=2,
        )
        return DEFAULT_FRONTIER_INVARIANT_TORUS_MIN
    return floor


def resolve_frontier_invariant_torus_min(
    frontier_invariant_torus_min: object | None,
    frontier_kam_min: object | None,
    *,
    environment: Mapping[str, str] | None = None,
) -> float:
    """Resolve the canonical frontier WBA floor from explicit values, env, or default.

    An explicit value (or its ``frontier_kam_min`` / env-var aliases) of <=0 or non-finite
    is hard-clamped up to ``DEFAULT_FRONTIER_INVARIANT_TORUS_MIN`` with a loud warning (see
    ``_clamp_configured_invariant_torus_min``); it never silently re-vacuates the binding
    certification gate. The unconfigured path returns the nonzero default with no warning.
    """

    if frontier_invariant_torus_min is not None:
        return _clamp_configured_invariant_torus_min(frontier_invariant_torus_min)
    if frontier_kam_min is not None:
        return _clamp_configured_invariant_torus_min(frontier_kam_min)
    if environment is not None:
        if "FRONTIER_INVARIANT_TORUS_MIN" in environment:
            return _clamp_configured_invariant_torus_min(
                environment["FRONTIER_INVARIANT_TORUS_MIN"]
            )
        if "FRONTIER_KAM_MIN" in environment:
            return _clamp_configured_invariant_torus_min(
                environment["FRONTIER_KAM_MIN"]
            )
    return default_frontier_invariant_torus_min()


def _topology_entry_invariant_torus_fraction(
    topology_entry: Mapping[str, object],
) -> float | None:
    explicit_fraction = _finite_float_or_none(
        topology_entry.get("invariant_torus_fraction")
    )
    if explicit_fraction is not None:
        return explicit_fraction
    if topology_entry.get("kam_fraction_semantics") != KAM_FRACTION_SEMANTICS:
        return None
    return _finite_float_or_none(topology_entry.get("kam_fraction"))


def _frontier_invariant_torus_certification_status(
    *,
    enabled: bool,
    ok: bool | None,
    reason: str,
    hardware_ok: bool | None,
    topology_evaluated: bool | None,
    topology_broken: bool | None,
    accepted_iteration: int | None,
    topology_accepted_iteration: int | None,
    invariant_torus_fraction: float | None,
    invariant_torus_min: float | None,
    invariant_torus_deficit: float | None,
) -> dict[str, object]:
    return {
        "enabled": enabled,
        "ok": ok,
        "reason": reason,
        "hardware_ok": hardware_ok,
        "topology_evaluated": topology_evaluated,
        "topology_broken": topology_broken,
        "accepted_iteration": accepted_iteration,
        "topology_accepted_iteration": topology_accepted_iteration,
        "invariant_torus_fraction": invariant_torus_fraction,
        "invariant_torus_min": invariant_torus_min,
        "invariant_torus_deficit": invariant_torus_deficit,
        "kam_fraction": invariant_torus_fraction,
        "kam_fraction_semantics": (
            KAM_FRACTION_SEMANTICS if invariant_torus_fraction is not None else None
        ),
        "kam_min": invariant_torus_min,
        "kam_deficit": invariant_torus_deficit,
    }


def evaluate_frontier_trust_status(
    search_eval: Mapping[str, object],
    *,
    enabled: bool,
    threshold: float | None = None,
) -> dict[str, object]:
    if not enabled:
        return {
            "enabled": False,
            "ok": None,
            "residual": None,
            "threshold": None,
            "excess": None,
        }
    if threshold is None:
        raise ValueError("frontier trust evaluation requires a threshold")
    residual = float(search_eval["J_Boozer"])
    trust_threshold = float(threshold)
    excess = max(residual - trust_threshold, 0.0)
    return {
        "enabled": True,
        "ok": bool(np.isfinite(residual) and residual <= trust_threshold),
        "residual": residual,
        "threshold": trust_threshold,
        "excess": excess,
    }


def evaluate_frontier_kam_certification(
    topology_entry: Mapping[str, object] | None,
    *,
    enabled: bool,
    hardware_ok: bool | None,
    kam_min: float | None,
    accepted_iteration: int | None = None,
) -> dict[str, object]:
    if not enabled:
        return _frontier_invariant_torus_certification_status(
            enabled=False,
            ok=None,
            reason="disabled",
            hardware_ok=None,
            topology_evaluated=None,
            topology_broken=None,
            accepted_iteration=accepted_iteration,
            topology_accepted_iteration=None,
            invariant_torus_fraction=None,
            invariant_torus_min=None,
            invariant_torus_deficit=None,
        )
    if kam_min is None:
        raise ValueError("frontier invariant-torus certification requires kam_min")
    resolved_invariant_torus_min = float(kam_min)
    if not np.isfinite(resolved_invariant_torus_min) or not (
        0.0 <= resolved_invariant_torus_min <= 1.0
    ):
        raise ValueError(
            "frontier invariant-torus certification requires kam_min in [0, 1]"
        )
    if topology_entry is None:
        return _frontier_invariant_torus_certification_status(
            enabled=True,
            ok=False,
            reason=(
                "topology_not_evaluated" if hardware_ok is True else "hardware_failed"
            ),
            hardware_ok=bool(hardware_ok),
            topology_evaluated=False,
            topology_broken=None,
            accepted_iteration=accepted_iteration,
            topology_accepted_iteration=None,
            invariant_torus_fraction=None,
            invariant_torus_min=resolved_invariant_torus_min,
            invariant_torus_deficit=None,
        )

    topology_accepted_iteration = topology_entry.get("accepted_iteration")
    topology_broken = bool(topology_entry.get("topology_broken", False))
    raw_invariant_torus_fraction = _topology_entry_invariant_torus_fraction(
        topology_entry
    )
    invariant_torus_fraction_out_of_range = (
        raw_invariant_torus_fraction is not None
        and not (0.0 <= raw_invariant_torus_fraction <= 1.0)
    )
    invariant_torus_fraction = (
        None if invariant_torus_fraction_out_of_range else raw_invariant_torus_fraction
    )
    if accepted_iteration is not None and topology_accepted_iteration is not None:
        if int(topology_accepted_iteration) != int(accepted_iteration):
            return _frontier_invariant_torus_certification_status(
                enabled=True,
                ok=False,
                reason="topology_not_current",
                hardware_ok=bool(hardware_ok),
                topology_evaluated=True,
                topology_broken=topology_broken,
                accepted_iteration=accepted_iteration,
                topology_accepted_iteration=int(topology_accepted_iteration),
                invariant_torus_fraction=raw_invariant_torus_fraction,
                invariant_torus_min=resolved_invariant_torus_min,
                invariant_torus_deficit=None,
            )

    topology_iteration = (
        None
        if topology_accepted_iteration is None
        else int(topology_accepted_iteration)
    )
    if topology_broken:
        return _frontier_invariant_torus_certification_status(
            enabled=True,
            ok=False,
            reason="topology_broken",
            hardware_ok=bool(hardware_ok),
            topology_evaluated=True,
            topology_broken=True,
            accepted_iteration=accepted_iteration,
            topology_accepted_iteration=topology_iteration,
            invariant_torus_fraction=raw_invariant_torus_fraction,
            invariant_torus_min=resolved_invariant_torus_min,
            invariant_torus_deficit=(
                None
                if invariant_torus_fraction is None
                else max(
                    resolved_invariant_torus_min - invariant_torus_fraction,
                    0.0,
                )
            ),
        )
    if invariant_torus_fraction_out_of_range:
        return _frontier_invariant_torus_certification_status(
            enabled=True,
            ok=False,
            reason="invariant_torus_fraction_out_of_range",
            hardware_ok=bool(hardware_ok),
            topology_evaluated=True,
            topology_broken=False,
            accepted_iteration=accepted_iteration,
            topology_accepted_iteration=topology_iteration,
            invariant_torus_fraction=raw_invariant_torus_fraction,
            invariant_torus_min=resolved_invariant_torus_min,
            invariant_torus_deficit=None,
        )
    if invariant_torus_fraction is None:
        return _frontier_invariant_torus_certification_status(
            enabled=True,
            ok=False,
            reason=(
                "invariant_torus_fraction_missing"
                if hardware_ok is True
                else "hardware_failed"
            ),
            hardware_ok=bool(hardware_ok),
            topology_evaluated=True,
            topology_broken=False,
            accepted_iteration=accepted_iteration,
            topology_accepted_iteration=topology_iteration,
            invariant_torus_fraction=None,
            invariant_torus_min=resolved_invariant_torus_min,
            invariant_torus_deficit=None,
        )

    invariant_torus_deficit = max(
        resolved_invariant_torus_min - invariant_torus_fraction,
        0.0,
    )
    certified = hardware_ok is True and invariant_torus_deficit == 0.0
    if invariant_torus_deficit > 0.0:
        reason = "invariant_torus_fraction_below_min"
    elif hardware_ok is not True:
        reason = "hardware_failed"
    else:
        reason = "certified"
    return _frontier_invariant_torus_certification_status(
        enabled=True,
        ok=bool(certified),
        reason=reason,
        hardware_ok=bool(hardware_ok),
        topology_evaluated=True,
        topology_broken=False,
        accepted_iteration=accepted_iteration,
        topology_accepted_iteration=topology_iteration,
        invariant_torus_fraction=invariant_torus_fraction,
        invariant_torus_min=resolved_invariant_torus_min,
        invariant_torus_deficit=float(invariant_torus_deficit),
    )


def target_kam_floor_satisfied(
    topology_entry: Mapping[str, object] | None,
    *,
    floor: float | None,
    current_geometry_key: object = None,
) -> bool:
    """Target-mode invariant-torus (KAM) floor for incumbent promotion.

    floor: minimum required invariant-torus fraction in [0, 1]; None disables the
    floor (always satisfied). current_geometry_key: a fingerprint of the geometry
    under test; when given, the floor additionally fails closed unless the topology
    entry's stored "geometry_key" equals it — so a config is never promoted on a
    STALE entry left over from a different config (the in-loop scorer is not re-run
    for every incumbent check and the entry is not carried through incumbent
    restore). The key tracks the geometry itself, so it is correct regardless of any
    iteration-counter timing. Pass None only when the caller guarantees the entry
    matches the config under test (e.g. unit tests of the fraction logic).

    Returns True only when the topology entry is current (if a key is given) and
    reports an evaluated invariant-torus fraction >= floor. A missing entry, a stale
    entry (different geometry), or an unevaluated fraction (None) returns False —
    fail-closed: an unconfined config whose lines escape before classification, or
    whose KAM was measured for a different config, can never slip through. A too-high
    floor or under-resolved in-loop scorer therefore surfaces as an empty incumbent
    set ("no confined config found"), never a silently promoted unconfined config.
    KAM fraction is a discrete classifier output, so this is a promotion gate, not a
    gradient objective.
    """
    if floor is None:
        return True
    resolved_floor = float(floor)
    if not np.isfinite(resolved_floor) or not (0.0 <= resolved_floor <= 1.0):
        raise ValueError("target KAM floor requires a fraction in [0, 1]")
    if topology_entry is None:
        return False
    if current_geometry_key is not None:
        if topology_entry.get("geometry_key") != current_geometry_key:
            return False
    fraction = _topology_entry_invariant_torus_fraction(topology_entry)
    if fraction is None:
        return False
    return fraction >= resolved_floor


def evaluate_frontier_trust_penalty(
    search_eval: Mapping[str, object],
    *,
    enabled: bool,
    threshold: float | None = None,
    penalty_scale: float | None = None,
) -> dict[str, object]:
    """One-sided residual-cap penalty on J_Boozer, not a DOF-space trust region."""
    grad = np.asarray(search_eval["grad"], dtype=float)
    if not enabled:
        return {
            "enabled": False,
            "penalty": 0.0,
            "grad": np.zeros_like(grad),
            "scale": None,
            "excess_ratio": None,
        }
    if threshold is None or penalty_scale is None:
        raise ValueError("frontier trust penalty requires threshold and penalty_scale")
    scale = max(float(penalty_scale), FINITE_EPS)
    residual = float(search_eval["J_Boozer"])
    excess_ratio = max((residual - float(threshold)) / scale, 0.0)
    if excess_ratio == 0.0:
        penalty_grad = np.zeros_like(grad)
    else:
        penalty_grad = (2.0 * excess_ratio / scale) * np.asarray(
            search_eval["dJ_Boozer"], dtype=float
        )
    return {
        "enabled": True,
        "penalty": float(excess_ratio**2),
        "grad": penalty_grad,
        "scale": scale,
        "excess_ratio": float(excess_ratio),
    }


def annotate_frontier_search_eval(
    search_eval: Mapping[str, object],
    *,
    enabled: bool,
    threshold: float | None = None,
    penalty_scale: float | None = None,
) -> dict[str, object]:
    annotated = annotate_search_evaluation_finiteness(search_eval)
    trust_status = evaluate_frontier_trust_status(
        annotated,
        enabled=enabled,
        threshold=threshold,
    )
    trust_penalty = evaluate_frontier_trust_penalty(
        annotated,
        enabled=enabled,
        threshold=threshold,
        penalty_scale=penalty_scale,
    )
    if not enabled:
        return annotated
    annotated["frontier_base_total"] = float(annotated["total"])
    annotated["frontier_rank_total"] = (
        float(annotated["total"]) + trust_penalty["penalty"]
    )
    annotated["total"] = annotated["frontier_rank_total"]
    annotated["grad"] = (
        np.asarray(annotated["grad"], dtype=float) + trust_penalty["grad"]
    )
    annotated["frontier_trust_ok"] = trust_status["ok"]
    annotated["frontier_boozer_trust_threshold"] = trust_status["threshold"]
    annotated["frontier_boozer_trust_excess"] = trust_status["excess"]
    annotated["frontier_boozer_trust_penalty_scale"] = trust_penalty["scale"]
    annotated["frontier_boozer_trust_excess_ratio"] = trust_penalty["excess_ratio"]
    annotated["frontier_trust_penalty"] = trust_penalty["penalty"]
    return annotate_search_evaluation_finiteness(annotated)


def evaluate_frontier_hardware_search_contract(
    hardware_status: Mapping[str, object],
    *,
    policy: HardwareSearchPolicy,
    context: SearchContext,
) -> dict[str, object]:
    decision = decide_hardware_search_action(policy, hardware_status, context)
    violation_ratios = hardware_violation_ratios(hardware_status)
    return {
        "success": bool(hardware_status["success"]),
        "violations": list(hardware_status.get("violations", [])),
        "violation_ratios": violation_ratios,
        "max_violation_ratio": max(violation_ratios.values(), default=0.0),
        "reject": decision.reject,
        "warning_only": decision.warning_only,
        "rejection_increment": decision.rejection_increment,
        "reason": decision.reason,
    }


def evaluate_frontier_hardware_search_penalty(
    hardware_status: Mapping[str, object],
    *,
    previous_objective: float,
    penalty_scale: float = _DEFAULT_FRONTIER_SEARCH_CONTRACT_PENALTY_SCALE,
) -> dict[str, object]:
    violation_ratios = hardware_violation_ratios(hardware_status)
    max_violation_ratio = max(violation_ratios.values(), default=0.0)
    penalty = (
        _frontier_search_contract_penalty_base(previous_objective)
        * max(float(penalty_scale), 0.0)
        * max_violation_ratio
    )
    return {
        "success": bool(hardware_status["success"]),
        "violations": list(hardware_status.get("violations", [])),
        "violation_ratios": violation_ratios,
        "max_violation_ratio": float(max_violation_ratio),
        "penalty": float(penalty),
        "penalty_scale": float(max(float(penalty_scale), 0.0)),
    }


def evaluate_frontier_topology_search_contract(
    topology_status: Mapping[str, object],
    *,
    previous_objective: float,
    penalty_scale: float,
) -> dict[str, object]:
    enabled = bool(topology_status["enabled"])
    success = bool(topology_status["success"])
    deficit = topology_gate_deficit(topology_status)
    rejection_increment = None
    if enabled and not success:
        rejection_increment = topology_gate_rejection_increment(
            previous_objective,
            topology_status,
            penalty_scale,
        )
    return {
        "enabled": enabled,
        "success": success,
        "deficit": float(deficit),
        "reject": bool(enabled and not success),
        "rejection_increment": rejection_increment,
    }


def evaluate_frontier_topology_search_penalty(
    topology_status: Mapping[str, object],
    *,
    previous_objective: float,
    penalty_scale: float,
) -> dict[str, object]:
    deficit = topology_gate_deficit(topology_status)
    penalty = (
        _frontier_search_contract_penalty_base(previous_objective)
        * max(float(penalty_scale), 0.0)
        * float(deficit)
    )
    return {
        "enabled": bool(topology_status["enabled"]),
        "success": bool(topology_status["success"]),
        "deficit": float(deficit),
        "penalty": float(penalty),
        "penalty_scale": float(max(float(penalty_scale), 0.0)),
    }


def apply_frontier_search_contract_penalties(
    search_eval: Mapping[str, object],
    *,
    hardware_penalty: Mapping[str, object] | None = None,
    topology_penalty: Mapping[str, object] | None = None,
) -> dict[str, object]:
    annotated = dict(search_eval)
    existing_rank_total = float(
        annotated.get("frontier_rank_total", annotated["total"])
    )
    contract_penalty = 0.0

    if hardware_penalty is not None:
        contract_penalty += float(hardware_penalty["penalty"])
        annotated["frontier_hardware_penalty"] = float(hardware_penalty["penalty"])
        annotated["frontier_hardware_penalty_scale"] = float(
            hardware_penalty["penalty_scale"]
        )
        annotated["frontier_hardware_violation_ratios"] = dict(
            hardware_penalty["violation_ratios"]
        )
        annotated["frontier_hardware_max_violation_ratio"] = float(
            hardware_penalty["max_violation_ratio"]
        )

    if topology_penalty is not None:
        contract_penalty += float(topology_penalty["penalty"])
        annotated["frontier_topology_penalty"] = float(topology_penalty["penalty"])
        annotated["frontier_topology_penalty_scale"] = float(
            topology_penalty["penalty_scale"]
        )
        annotated["frontier_topology_deficit"] = float(topology_penalty["deficit"])

    annotated["frontier_contract_penalty"] = float(contract_penalty)
    annotated["frontier_rank_total"] = existing_rank_total + float(contract_penalty)
    annotated["total"] = annotated["frontier_rank_total"]
    return annotated


def evaluate_frontier_hard_invalidation(
    *,
    search_eval: Mapping[str, object] | None,
    surface_success: bool,
    surface_status: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not surface_success:
        if surface_status is not None:
            if not all(
                bool(value) for value in surface_status.get("solve_success", [])
            ):
                return {
                    "invalid": True,
                    "reason": "surface_solve_failed",
                    "fields": ["solve_success"],
                }
            if any(
                bool(value) for value in surface_status.get("self_intersections", [])
            ):
                return {
                    "invalid": True,
                    "reason": "geometry_state_unrestorable",
                    "fields": ["self_intersections"],
                }
            if not bool(surface_status.get("volumes_ordered", True)):
                return {
                    "invalid": True,
                    "reason": "geometry_state_unrestorable",
                    "fields": ["volumes"],
                }
            if not bool(surface_status.get("gap_ok", True)):
                return {
                    "invalid": True,
                    "reason": "geometry_state_unrestorable",
                    "fields": ["adjacent_gaps"],
                }
            if not bool(surface_status.get("nesting_ok", True)):
                return {
                    "invalid": True,
                    "reason": "geometry_state_unrestorable",
                    "fields": ["bad_nesting_phis"],
                }
        return {
            "invalid": True,
            "reason": "surface_solve_failed",
            "fields": ["surface_status"],
        }
    if search_eval is None:
        return {
            "invalid": True,
            "reason": "missing_search_eval",
            "fields": ["search_eval"],
        }
    invalid_fields = list(search_eval.get("nonfinite_fields", []))
    if search_eval.get("finite_eval_ok") is False or invalid_fields:
        return {
            "invalid": True,
            "reason": "nonfinite_evaluation",
            "fields": invalid_fields,
        }
    return {
        "invalid": False,
        "reason": None,
        "fields": [],
    }


def hardware_violation_ratios(
    hardware_status: Mapping[str, object],
) -> dict[str, float]:
    ratios = {
        "curve_curve_min_dist": _lower_bound_violation_ratio(
            hardware_status.get("curve_curve_min_dist"),
            hardware_status.get("cc_dist"),
        ),
        "curve_surface_min_dist": _lower_bound_violation_ratio(
            hardware_status.get("curve_surface_min_dist"),
            hardware_status.get("cs_dist"),
        ),
        "max_curvature": _upper_bound_violation_ratio(
            hardware_status.get("max_curvature"),
            hardware_status.get("curvature_threshold"),
        ),
        "banana_current": _box_bound_violation_ratio(
            hardware_status.get("banana_current_A"),
            hardware_status.get("banana_current_max_A"),
        ),
        "tf_current": _box_bound_violation_ratio(
            hardware_status.get("tf_current_A"),
            hardware_status.get("tf_current_limit_A"),
        ),
    }
    for name, entry in dict(hardware_status.get("constraints", {})).items():
        if _is_diagnostic_hardware_ratio_name(str(name)):
            continue
        threshold = abs(float(entry["threshold"]))
        violation = float(entry["violation"])
        if threshold > 0.0:
            ratios[str(name)] = float(entry["violation"]) / max(
                threshold,
                FINITE_EPS,
            )
        elif violation > 0.0:
            ratios[str(name)] = violation
    explicit_ratios = hardware_status.get("violation_ratios")
    if explicit_ratios is not None:
        for name, value in dict(explicit_ratios).items():
            if _is_diagnostic_hardware_ratio_name(str(name)):
                continue
            ratios[str(name)] = float(value)
    return ratios


def _is_diagnostic_hardware_ratio_name(name: str) -> bool:
    return name in _DIAGNOSTIC_HARDWARE_RATIO_NAMES


def _lower_bound_violation_ratio(value, minimum) -> float:
    if value is None or minimum is None:
        return 0.0
    minimum_value = max(abs(float(minimum)), FINITE_EPS)
    return max(float(minimum) - float(value), 0.0) / minimum_value


def _upper_bound_violation_ratio(value, maximum) -> float:
    if value is None or maximum is None:
        return 0.0
    maximum_value = max(abs(float(maximum)), FINITE_EPS)
    return max(float(value) - float(maximum), 0.0) / maximum_value


def _box_bound_violation_ratio(value, maximum) -> float:
    if value is None or maximum is None:
        return 0.0
    maximum_value = max(abs(float(maximum)), FINITE_EPS)
    return max(abs(float(value)) - float(maximum), 0.0) / maximum_value


def _frontier_search_contract_penalty_base(previous_objective: float) -> float:
    return max(abs(float(previous_objective)), 1.0)
