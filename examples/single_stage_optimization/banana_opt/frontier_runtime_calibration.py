from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

from .frontier_archive import certified_archive_members, frontier_archive_hypervolume

FRONTIER_RUNTIME_CALIBRATION_SCHEMA_VERSION = "frontier_runtime_calibration_v1"


@dataclass(frozen=True, slots=True)
class FrontierRuntimeCalibrationProfile:
    schema_version: str
    profile_name: str
    default_num_lanes: int
    default_lane_budget: int
    default_checkpoint_every: int
    default_early_stop_patience_lanes: int
    default_early_stop_min_certified: int
    default_early_stop_min_hypervolume_gain: float
    calibration_basis: tuple[str, ...]

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["calibration_basis"] = list(self.calibration_basis)
        return payload


@dataclass(frozen=True, slots=True)
class FrontierResolvedRuntimeDefaults:
    calibration_profile: FrontierRuntimeCalibrationProfile
    num_lanes: int
    lane_budget: int
    total_budget: int
    checkpoint_every: int
    early_stop_patience_lanes: int
    early_stop_min_certified: int
    early_stop_min_hypervolume_gain: float

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": FRONTIER_RUNTIME_CALIBRATION_SCHEMA_VERSION,
            "profile": self.calibration_profile.to_json_dict(),
            "resolved_defaults": {
                "num_lanes": self.num_lanes,
                "lane_budget": self.lane_budget,
                "total_budget": self.total_budget,
                "checkpoint_every": self.checkpoint_every,
                "early_stop_patience_lanes": self.early_stop_patience_lanes,
                "early_stop_min_certified": self.early_stop_min_certified,
                "early_stop_min_hypervolume_gain": self.early_stop_min_hypervolume_gain,
            },
        }


# DC.21 (frontier bloat-reduction plan, r48): the two-profile registry is intentionally
# retained instead of collapsing to a single FrontierRuntimeDefaults dataclass + factory
# (deferred 8.2.2). D3 = "preserve shape" → external lab-note consumers pin the
# profile-bearing FRONTIER_RUNTIME_CALIBRATION JSON contract. Closed WONTFIX 2026-05-08.
_FRONTIER_RUNTIME_CALIBRATION_PROFILES: Mapping[
    str, FrontierRuntimeCalibrationProfile
] = MappingProxyType(
    {
        "canonical_seed_v1": FrontierRuntimeCalibrationProfile(
            schema_version=FRONTIER_RUNTIME_CALIBRATION_SCHEMA_VERSION,
            profile_name="canonical_seed_v1",
            default_num_lanes=3,
            default_lane_budget=300,
            default_checkpoint_every=5,
            default_early_stop_patience_lanes=3,
            default_early_stop_min_certified=1,
            default_early_stop_min_hypervolume_gain=1.0e-6,
            calibration_basis=(
                "canonical_seed_bridge_smoke",
                "canonical_seed_resume_smoke",
            ),
        ),
        "reduced_fixture_v1": FrontierRuntimeCalibrationProfile(
            schema_version=FRONTIER_RUNTIME_CALIBRATION_SCHEMA_VERSION,
            profile_name="reduced_fixture_v1",
            default_num_lanes=3,
            default_lane_budget=300,
            default_checkpoint_every=5,
            default_early_stop_patience_lanes=2,
            default_early_stop_min_certified=1,
            default_early_stop_min_hypervolume_gain=1.0e-6,
            calibration_basis=(
                "reduced_fixture_multilane_smoke",
                "deterministic_resume_smoke",
            ),
        ),
    }
)
SUPPORTED_FRONTIER_RUNTIME_CALIBRATION_PROFILES = tuple(
    _FRONTIER_RUNTIME_CALIBRATION_PROFILES
)


def get_frontier_runtime_calibration_profile(
    profile_name: str,
) -> FrontierRuntimeCalibrationProfile:
    profile = _FRONTIER_RUNTIME_CALIBRATION_PROFILES.get(profile_name)
    if profile is None:
        raise ValueError(
            f"Unsupported frontier runtime calibration profile {profile_name!r}"
        )
    return profile


def resolve_frontier_runtime_defaults(
    *,
    profile_name: str,
    requested_num_lanes: int | None,
    requested_lane_budget: int | None,
    requested_total_budget: int | None,
    requested_checkpoint_every: int | None,
    requested_early_stop_patience_lanes: int | None,
    requested_early_stop_min_certified: int | None,
    requested_early_stop_min_hypervolume_gain: float | None,
) -> FrontierResolvedRuntimeDefaults:
    profile = get_frontier_runtime_calibration_profile(profile_name)
    num_lanes = (
        profile.default_num_lanes
        if requested_num_lanes is None
        else int(requested_num_lanes)
    )
    lane_budget = (
        profile.default_lane_budget
        if requested_lane_budget is None
        else int(requested_lane_budget)
    )
    total_budget = (
        int(requested_total_budget)
        if requested_total_budget is not None
        else int(num_lanes * lane_budget)
    )
    checkpoint_every = (
        profile.default_checkpoint_every
        if requested_checkpoint_every is None or int(requested_checkpoint_every) <= 0
        else int(requested_checkpoint_every)
    )
    early_stop_patience_lanes = (
        profile.default_early_stop_patience_lanes
        if requested_early_stop_patience_lanes is None
        else int(requested_early_stop_patience_lanes)
    )
    if early_stop_patience_lanes < 0:
        raise ValueError(
            "frontier early_stop_patience_lanes must be >= 0 "
            f"(0 disables early stop), got {early_stop_patience_lanes}"
        )
    early_stop_min_certified = (
        profile.default_early_stop_min_certified
        if requested_early_stop_min_certified is None
        else int(requested_early_stop_min_certified)
    )
    if early_stop_min_certified < 0:
        raise ValueError(
            "frontier early_stop_min_certified must be >= 0, "
            f"got {early_stop_min_certified}"
        )
    early_stop_min_hypervolume_gain = (
        profile.default_early_stop_min_hypervolume_gain
        if requested_early_stop_min_hypervolume_gain is None
        else float(requested_early_stop_min_hypervolume_gain)
    )
    if not (
        math.isfinite(early_stop_min_hypervolume_gain)
        and early_stop_min_hypervolume_gain >= 0.0
    ):
        raise ValueError(
            "frontier early_stop_min_hypervolume_gain must be a finite "
            f"non-negative float, got {early_stop_min_hypervolume_gain}"
        )
    return FrontierResolvedRuntimeDefaults(
        calibration_profile=profile,
        num_lanes=int(num_lanes),
        lane_budget=int(lane_budget),
        total_budget=int(total_budget),
        checkpoint_every=int(checkpoint_every),
        early_stop_patience_lanes=int(early_stop_patience_lanes),
        early_stop_min_certified=int(early_stop_min_certified),
        early_stop_min_hypervolume_gain=float(early_stop_min_hypervolume_gain),
    )


def resolve_frontier_runtime_defaults_from_args(
    args: object,
    *,
    requested_num_lanes: int | None = None,
) -> FrontierResolvedRuntimeDefaults:
    return resolve_frontier_runtime_defaults(
        profile_name=args.frontier_runtime_calibration_profile,
        requested_num_lanes=(
            args.frontier_num_lanes
            if requested_num_lanes is None
            else requested_num_lanes
        ),
        requested_lane_budget=args.frontier_lane_budget,
        requested_total_budget=args.frontier_total_budget,
        requested_checkpoint_every=args.checkpoint_every,
        requested_early_stop_patience_lanes=args.frontier_early_stop_patience_lanes,
        requested_early_stop_min_certified=args.frontier_early_stop_min_certified,
        requested_early_stop_min_hypervolume_gain=(
            args.frontier_early_stop_min_hypervolume_gain
        ),
    )


def effective_lane_budget(
    lane_budget: int | None,
    runtime_defaults: FrontierResolvedRuntimeDefaults,
) -> int:
    return (
        runtime_defaults.lane_budget
        if lane_budget is None
        else int(lane_budget)
    )


def effective_total_budget(
    lane_budgets: Iterable[int],
    runtime_defaults: FrontierResolvedRuntimeDefaults,
) -> int:
    resolved_lane_budgets = [int(budget) for budget in lane_budgets]
    if not resolved_lane_budgets:
        return int(runtime_defaults.total_budget)
    return int(sum(resolved_lane_budgets))


def build_frontier_early_stop_policy(
    runtime_defaults: FrontierResolvedRuntimeDefaults,
) -> dict[str, object]:
    return {
        "patience_lanes": runtime_defaults.early_stop_patience_lanes,
        "min_certified": runtime_defaults.early_stop_min_certified,
        "min_hypervolume_gain": runtime_defaults.early_stop_min_hypervolume_gain,
    }


def build_initial_frontier_early_stop_status(
    *,
    runtime_defaults: FrontierResolvedRuntimeDefaults,
    archive_members,
) -> dict[str, object]:
    return {
        "policy": build_frontier_early_stop_policy(runtime_defaults),
        "triggered": False,
        "reason": None,
        "no_improvement_streak": 0,
        "best_hypervolume": None,
        "best_archive_size": len(certified_archive_members(archive_members)),
        "stopped_after_lane_id": None,
    }


def update_frontier_early_stop_status(
    *,
    status: dict[str, object] | None,
    certified_archive_members_list,
    hypervolume_reference,
    runtime_defaults: FrontierResolvedRuntimeDefaults,
) -> dict[str, object]:
    policy = build_frontier_early_stop_policy(runtime_defaults)
    previous_status = (
        build_initial_frontier_early_stop_status(
            runtime_defaults=runtime_defaults,
            archive_members=[],
        )
        if status is None
        else dict(status)
    )

    certified_members = certified_archive_members(certified_archive_members_list)
    if len(certified_members) < runtime_defaults.early_stop_min_certified:
        previous_status["policy"] = policy
        previous_status["best_archive_size"] = max(
            int(previous_status.get("best_archive_size", 0)),
            len(certified_members),
        )
        return previous_status

    current_hypervolume = frontier_archive_hypervolume(
        certified_members,
        hypervolume_reference=hypervolume_reference,
    )
    previous_best_hypervolume = previous_status.get("best_hypervolume")
    previous_best_archive_size = int(previous_status.get("best_archive_size", 0))

    improved = False
    if current_hypervolume is not None:
        improved = (
            previous_best_hypervolume is None
            or float(current_hypervolume) >= float(previous_best_hypervolume)
            + runtime_defaults.early_stop_min_hypervolume_gain
        )
    else:
        improved = len(certified_members) > previous_best_archive_size

    if improved:
        previous_status["no_improvement_streak"] = 0
        previous_status["best_hypervolume"] = current_hypervolume
        previous_status["best_archive_size"] = len(certified_members)
    else:
        previous_status["no_improvement_streak"] = int(
            previous_status.get("no_improvement_streak", 0)
        ) + 1
        previous_status["best_archive_size"] = max(
            previous_best_archive_size,
            len(certified_members),
        )
        if previous_best_hypervolume is None:
            previous_status["best_hypervolume"] = current_hypervolume

    previous_status["policy"] = policy
    triggered = (
        runtime_defaults.early_stop_patience_lanes > 0
        and int(previous_status["no_improvement_streak"])
        >= runtime_defaults.early_stop_patience_lanes
    )
    previous_status["triggered"] = triggered
    if triggered and previous_status.get("reason") is None:
        previous_status["reason"] = "archive_stagnation"
    return previous_status
