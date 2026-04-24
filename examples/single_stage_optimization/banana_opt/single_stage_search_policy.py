from dataclasses import dataclass
from typing import Literal


HardwareSearchMode = Literal["hard", "warn", "adaptive"]


@dataclass(frozen=True)
class HardwareSearchPolicy:
    mode: HardwareSearchMode
    soft_iterations: int = 0

    def __post_init__(self):
        if self.mode not in ("hard", "warn", "adaptive"):
            raise ValueError(
                "hardware search mode must be one of 'hard', 'warn', or 'adaptive'"
            )
        if self.soft_iterations < 0:
            raise ValueError("hardware search soft_iterations must be non-negative")


@dataclass(frozen=True)
class SearchContext:
    accepted_iterations: int
    gate_scale: float
    previous_objective: float


@dataclass(frozen=True)
class SearchDecision:
    reject: bool
    warning_only: bool
    rejection_increment: float | None
    reason: str | None


@dataclass(frozen=True)
class CurvatureTraversalPolicy:
    band_ratio: float = 0.0
    eval_budget: int = 0

    def __post_init__(self):
        if self.band_ratio < 0.0:
            raise ValueError("curvature traversal band_ratio must be non-negative")
        if self.eval_budget < 0:
            raise ValueError("curvature traversal eval_budget must be non-negative")


@dataclass(frozen=True)
class CurvatureTraversalDecision:
    allow_boozer_eval: bool
    over_threshold: bool
    reason: str
    far_invalid_limit: float


def hardware_rejection_increment(previous_objective):
    return max(abs(float(previous_objective)), 1.0)


def decide_curvature_traversal(
    *,
    max_curvature,
    curvature_threshold,
    policy: CurvatureTraversalPolicy,
    used_budget,
):
    far_invalid_limit = float(curvature_threshold) * (1.0 + float(policy.band_ratio))
    if float(max_curvature) <= float(curvature_threshold):
        return CurvatureTraversalDecision(
            allow_boozer_eval=True,
            over_threshold=False,
            reason="within_threshold",
            far_invalid_limit=far_invalid_limit,
        )
    if float(max_curvature) > far_invalid_limit:
        return CurvatureTraversalDecision(
            allow_boozer_eval=False,
            over_threshold=True,
            reason="far_invalid_curvature",
            far_invalid_limit=far_invalid_limit,
        )
    if int(used_budget) >= int(policy.eval_budget):
        return CurvatureTraversalDecision(
            allow_boozer_eval=False,
            over_threshold=True,
            reason="curvature_traversal_budget_exhausted",
            far_invalid_limit=far_invalid_limit,
        )
    return CurvatureTraversalDecision(
        allow_boozer_eval=True,
        over_threshold=True,
        reason="within_traversal_band",
        far_invalid_limit=far_invalid_limit,
    )


def _is_adaptive_soft_phase(policy: HardwareSearchPolicy, context: SearchContext):
    if float(context.gate_scale) >= 1.0:
        return False
    if int(policy.soft_iterations) <= 0:
        return True
    return context.accepted_iterations < int(policy.soft_iterations)


def decide_hardware_search_action(
    policy: HardwareSearchPolicy,
    hardware_status,
    context: SearchContext,
):
    if hardware_status["success"]:
        return SearchDecision(
            reject=False,
            warning_only=False,
            rejection_increment=None,
            reason=None,
        )

    if policy.mode == "warn":
        return SearchDecision(
            reject=False,
            warning_only=True,
            rejection_increment=None,
            reason="warn_mode",
        )

    if policy.mode == "adaptive" and _is_adaptive_soft_phase(policy, context):
        return SearchDecision(
            reject=False,
            warning_only=True,
            rejection_increment=None,
            reason="adaptive_soft_phase",
        )

    return SearchDecision(
        reject=True,
        warning_only=False,
        rejection_increment=hardware_rejection_increment(context.previous_objective),
        reason="hard_reject",
    )
