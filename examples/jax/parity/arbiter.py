"""Fail-closed scientific comparison of isolated parity lane receipts."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping

import numpy as np
from examples.jax.parity._manifest import ComparisonRoute
from examples.jax.parity.contracts import (
    ComparisonResult,
    QualityBand,
    QualityBandResult,
)
from examples.jax.parity.provenance import LaneProvenance
from simsopt_jax.config import ExecutionIntent
from simsopt_jax.examples import ExecutionScale
from simsopt_jax.parity_tolerances import parity_ladder_tolerances

_REQUIRED_LANES = frozenset({"native-cpu", "jax-cpu", "jax-gpu"})
_REQUIRED_PAIRS = frozenset(
    {"native-cpu:jax-cpu", "native-cpu:jax-gpu", "jax-cpu:jax-gpu"}
)
QUALITY_BAND_SCALE: ExecutionScale = "native_default"
QUALITY_BAND_VERDICT = "quality-band"
_INFORMATIONAL_DIAGNOSTIC_PREFIX = "informational (quality-band, non-certifying): "


class ArbitrationError(ValueError):
    """Lane evidence is structurally incapable of supporting parity."""


@dataclass(frozen=True)
class LaneObservation:
    lane: str
    backend_mode: str
    platform: str
    precision: str
    scale: ExecutionScale
    input_fingerprint: str
    configuration_fingerprint: str
    effective_construction_fingerprint: str
    driver: str
    normalized_status: str
    raw_status: str
    success: bool
    nit: int | None
    nfev: int | None
    njev: int | None
    completed_workflow_stages: tuple[str, ...]
    provenance: LaneProvenance | None
    values: Mapping[str, np.ndarray]
    applicability: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = dict(self.values)
        applicability = {
            **{key: True for key in values},
            "optimizer_outcome": self.normalized_status != "not_applicable",
        }
        supplied_applicability = dict(self.applicability)
        unexpected = set(supplied_applicability) - set(applicability)
        if unexpected:
            raise ValueError(
                f"lane applicability has unknown keys: {sorted(unexpected)}"
            )
        applicability.update(supplied_applicability)
        if applicability["optimizer_outcome"] != (
            self.normalized_status != "not_applicable"
        ):
            raise ValueError("optimizer applicability must match normalized status")
        object.__setattr__(self, "values", MappingProxyType(values))
        object.__setattr__(self, "applicability", MappingProxyType(applicability))


@dataclass(frozen=True)
class ArbitrationResult:
    verdict: str
    comparisons: tuple[ComparisonResult, ...]
    quality_band_results: tuple[QualityBandResult, ...] = ()


def _validate_lanes(
    observations: Mapping[str, LaneObservation],
    required_lanes: frozenset[str],
    expected_workflow_stages: tuple[str, ...] | None,
    execution_intent: ExecutionIntent = "parity",
    quality_band: QualityBand | None = None,
) -> None:
    if execution_intent not in ("fast", "parity"):
        raise ArbitrationError(f"invalid execution intent: {execution_intent}")
    if not required_lanes or not required_lanes <= _REQUIRED_LANES:
        raise ArbitrationError(f"invalid required lanes: {sorted(required_lanes)}")
    missing = required_lanes - set(observations)
    if missing:
        raise ArbitrationError(f"missing required lane: {sorted(missing)}")
    if quality_band is not None and {
        observations[lane].scale for lane in required_lanes
    } != {QUALITY_BAND_SCALE}:
        raise ArbitrationError(
            f"quality-band certification requires the {QUALITY_BAND_SCALE} scale"
        )
    budget_exhausted_admissible = quality_band is not None and {
        observations[lane].normalized_status for lane in required_lanes
    } == {"budget_exhausted"}
    expected_runtime = {
        "native-cpu": ("native_cpu", "cpu", "fp64"),
        "jax-cpu": (f"jax_cpu_{execution_intent}", "cpu", "fp64"),
        "jax-gpu": (f"jax_gpu_{execution_intent}", "gpu", "fp64"),
    }
    expected_gpu_transfer_guard = "log" if execution_intent == "fast" else "disallow"
    for lane in sorted(required_lanes):
        backend, platform, precision = expected_runtime[lane]
        observation = observations[lane]
        if observation.provenance is None:
            raise ArbitrationError(f"{lane} is missing provenance")
        provenance = observation.provenance
        if observation.lane != lane:
            raise ArbitrationError(f"{lane} receipt names lane {observation.lane}")
        if observation.backend_mode != backend:
            raise ArbitrationError(f"{lane} backend_mode must be {backend}")
        if observation.platform != platform:
            raise ArbitrationError(f"{lane} platform must be {platform}")
        if observation.precision != precision:
            raise ArbitrationError(f"{lane} precision must be {precision}")
        if lane.startswith("jax-") and any(
            forbidden in observation.driver.lower()
            for forbidden in ("scipy", "optimistix", "optax", "host_callback")
        ):
            raise ArbitrationError(
                f"{lane} uses forbidden parity driver {observation.driver}"
            )
        if provenance.lane_environment_policy.get("SIMSOPT_BACKEND_MODE") != backend:
            raise ArbitrationError(f"{lane} provenance backend policy mismatch")
        if lane == "jax-gpu" and (
            provenance.lane_environment_policy.get("SIMSOPT_JAX_TRANSFER_GUARD")
            != expected_gpu_transfer_guard
            or provenance.lane_environment_policy.get("JAX_TRANSFER_GUARD")
            != expected_gpu_transfer_guard
            or set(provenance.jax_effective_transfer_guards.values())
            != {expected_gpu_transfer_guard}
            or set(provenance.jax_effective_transfer_guards)
            != {"device_to_device", "device_to_host", "host_to_device"}
        ):
            raise ArbitrationError(
                f"jax-gpu effective transfer guards must {expected_gpu_transfer_guard}"
            )
        if lane.startswith("jax-"):
            device_platforms = {device.platform for device in provenance.devices}
            expected_device_platforms = (
                {"cuda", "gpu"} if lane == "jax-gpu" else {"cpu"}
            )
            if not device_platforms & expected_device_platforms:
                raise ArbitrationError(f"{lane} device provenance mismatch")
        if observation.normalized_status not in {
            "converged",
            "budget_exhausted",
            "failed",
            "not_applicable",
        }:
            raise ArbitrationError(
                f"{lane} has invalid normalized status {observation.normalized_status}"
            )
        if not observation.success and not budget_exhausted_admissible:
            raise ArbitrationError(f"{lane} did not report scientific success")
        if observation.normalized_status == "not_applicable" and (
            observation.nit is not None
            or observation.nfev is not None
            or observation.njev is not None
        ):
            raise ArbitrationError(
                f"{lane} optimizer counters must be null when outcome is not applicable"
            )
        for counter_name in ("nit", "nfev", "njev"):
            counter = getattr(observation, counter_name)
            if counter is not None and counter < 0:
                raise ArbitrationError(f"{lane} has invalid {counter_name}: {counter}")
    if budget_exhausted_admissible:
        budgets = {observations[lane].nit for lane in required_lanes}
        if len(budgets) != 1 or None in budgets:
            raise ArbitrationError(
                "quality-band budget_exhausted lanes must share one matched "
                f"iteration budget: {sorted(budgets, key=repr)}"
            )
    stage_contract = (
        expected_workflow_stages
        if expected_workflow_stages is not None
        else observations[next(iter(sorted(required_lanes)))].completed_workflow_stages
    )
    if not stage_contract or len(stage_contract) != len(set(stage_contract)):
        raise ArbitrationError("workflow stage contract must be non-empty and unique")
    for lane in sorted(required_lanes):
        if observations[lane].completed_workflow_stages != stage_contract:
            raise ArbitrationError(f"{lane} workflow stage mismatch")
    normalized_statuses = {
        observations[lane].normalized_status for lane in sorted(required_lanes)
    }
    if len(normalized_statuses) != 1:
        raise ArbitrationError("normalized convergence category mismatch")
    scales = {observations[lane].scale for lane in sorted(required_lanes)}
    if len(scales) != 1:
        raise ArbitrationError("execution scale mismatch")
    fingerprint_fields = (
        ("input", "input_fingerprint"),
        ("configuration", "configuration_fingerprint"),
        ("effective construction", "effective_construction_fingerprint"),
    )
    for label, field_name in fingerprint_fields:
        values = {
            getattr(observations[lane], field_name) for lane in sorted(required_lanes)
        }
        if len(values) != 1:
            raise ArbitrationError(f"{label} fingerprint mismatch")
    commits: set[str] = set()
    source_manifests: dict[str, dict[str, str]] = {}
    for lane in sorted(required_lanes):
        provenance = observations[lane].provenance
        assert provenance is not None
        commits.add(provenance.repository_commit)
        source_map = {
            source.path: source.sha256 for source in provenance.executed_sources
        }
        if len(source_map) != len(provenance.executed_sources):
            raise ArbitrationError(f"{lane} has duplicate executed source paths")
        source_manifests[lane] = source_map
    if len(commits) != 1:
        raise ArbitrationError("repository provenance mismatch: repository_commit")
    lanes = sorted(required_lanes)
    for left_index, left_lane in enumerate(lanes):
        for right_lane in lanes[left_index + 1 :]:
            shared_paths = set(source_manifests[left_lane]) & set(
                source_manifests[right_lane]
            )
            for path in sorted(shared_paths):
                if (
                    source_manifests[left_lane][path]
                    != source_manifests[right_lane][path]
                ):
                    raise ArbitrationError(
                        f"executed source mismatch: {path} ({left_lane}:{right_lane})"
                    )


def _required_lane_pairs(required_lanes: frozenset[str]) -> frozenset[str]:
    return frozenset(
        pair
        for pair in _REQUIRED_PAIRS
        if set(pair.split(":", maxsplit=1)) <= required_lanes
    )


def _validate_route_matrix(
    routes: tuple[ComparisonRoute, ...],
    observations: Mapping[str, LaneObservation],
    required_lanes: frozenset[str],
) -> None:
    grouped: dict[tuple[str, str], set[str]] = {}
    applicability_by_key: dict[tuple[str, str], set[bool]] = {}
    for route in routes:
        key = (route.phase, route.observable)
        grouped.setdefault(key, set()).add(route.lane_pair)
        applicability_by_key.setdefault(key, set()).add(route.applicable)
    required_pairs = _required_lane_pairs(required_lanes)
    observable_keys = {
        lane: {
            key
            for key, applicable in observations[lane].applicability.items()
            if key != "optimizer_outcome" and applicable
        }
        for lane in required_lanes
    }
    if len({frozenset(keys) for keys in observable_keys.values()}) != 1:
        raise ArbitrationError("lane observable applicability mismatch")
    required_keys = next(iter(observable_keys.values()))
    grouped_keys = {f"{phase}:{observable}" for phase, observable in grouped}
    if grouped_keys != required_keys:
        raise ArbitrationError(
            "applicable observables require a complete direct lane-pair matrix"
        )
    for key, lane_pairs in grouped.items():
        if lane_pairs != required_pairs:
            raise ArbitrationError(f"{key} requires a complete direct lane-pair matrix")
        if len(applicability_by_key[key]) != 1:
            raise ArbitrationError(f"{key} has inconsistent comparison applicability")


def _route_tolerance(route: ComparisonRoute) -> tuple[float, float]:
    tolerance = parity_ladder_tolerances(route.tolerance_bucket)
    derivative = any(
        token in route.observable.lower()
        for token in ("gradient", "jacobian", "derivative")
    )
    if route.tolerance_bucket == "gpu_runtime":
        if route.phase == "final":
            keys = ("whole_solve_value_rtol", "whole_solve_value_atol")
        elif derivative:
            keys = ("same_state_gradient_rtol", "same_state_gradient_atol")
        else:
            keys = ("same_state_forward_rtol", "same_state_forward_atol")
    elif route.tolerance_bucket == "native_workflow":
        if route.phase == "final":
            keys = ("whole_solve_value_rtol", "whole_solve_value_atol")
        elif derivative:
            keys = ("same_state_derivative_rtol", "same_state_derivative_atol")
        else:
            keys = ("same_state_value_rtol", "same_state_value_atol")
    else:
        keys = ("rtol", "atol")
    rtol, atol = tolerance.get(keys[0]), tolerance.get(keys[1])
    if not isinstance(rtol, float) or not isinstance(atol, float):
        raise ArbitrationError(
            f"tolerance bucket {route.tolerance_bucket} lacks {keys}"
        )
    return rtol, atol


def _quality_band_results(
    quality_band: QualityBand,
    observations: Mapping[str, LaneObservation],
    required_lanes: frozenset[str],
) -> tuple[QualityBandResult, ...]:
    """Measure every compared lane against one case-owned endpoint floor."""
    results: list[QualityBandResult] = []
    for lane in sorted(required_lanes):
        observation = observations[lane]
        # A receipt's applicability keys are exactly its published value keys,
        # so this one gate covers both an absent and a disclaimed observable.
        if not observation.applicability.get(quality_band.observable, False):
            raise ArbitrationError(
                "quality-band observable is not applicable "
                f"{quality_band.observable}: {lane}"
            )
        value = np.asarray(observation.values[quality_band.observable])
        if value.dtype != np.dtype(np.float64):
            raise ArbitrationError(
                f"{lane} quality-band observable must be FP64: "
                f"{quality_band.observable}"
            )
        if value.size == 0 or not bool(np.all(np.isfinite(value))):
            raise ArbitrationError(
                f"non-finite quality-band observable {quality_band.observable}: {lane}"
            )
        observed_value = float(np.max(value))
        results.append(
            QualityBandResult(
                lane=lane,
                observable=quality_band.observable,
                max_value=quality_band.max_value,
                observed_value=observed_value,
                passed=observed_value <= quality_band.max_value,
            )
        )
    return tuple(results)


def arbitrate(
    routes: tuple[ComparisonRoute, ...],
    observations: Mapping[str, LaneObservation],
    *,
    required_lanes: frozenset[str] = _REQUIRED_LANES,
    expected_workflow_stages: tuple[str, ...] | None = None,
    execution_intent: ExecutionIntent = "parity",
    quality_band: QualityBand | None = None,
) -> ArbitrationResult:
    """Compare every direct pair under the declared JAX execution policy.

    The default retains the certification-oriented parity backend and guards.

    A case that declares a ``quality_band`` opts that case into the 2026-08-15
    ruling's rule-3 comparator at ``native_default``: ``budget_exhausted`` is an
    admissible terminal state when every compared lane exhausts one identical
    matched budget, certification is the declared endpoint floor rather than
    final-value equality, and the verdict is labelled ``quality-band`` so it can
    never be read as equivalence. Pairwise comparisons are still computed and
    recorded, marked informational, and do not decide the verdict.
    """
    _validate_lanes(
        observations,
        required_lanes,
        expected_workflow_stages,
        execution_intent,
        quality_band,
    )
    selected_routes = tuple(
        route
        for route in routes
        if set(route.lane_pair.split(":", maxsplit=1)) <= required_lanes
    )
    _validate_route_matrix(selected_routes, observations, required_lanes)
    comparisons: list[ComparisonResult] = []
    for route in selected_routes:
        if not route.applicable:
            continue
        left_lane, right_lane = route.lane_pair.split(":", maxsplit=1)
        value_key = f"{route.phase}:{route.observable}"
        try:
            if not observations[left_lane].applicability.get(value_key, False) or not (
                observations[right_lane].applicability.get(value_key, False)
            ):
                raise ArbitrationError(
                    f"required observable is marked not applicable {value_key}: "
                    f"{route.lane_pair}"
                )
            left = np.asarray(observations[left_lane].values[value_key])
            right = np.asarray(observations[right_lane].values[value_key])
        except KeyError as error:
            raise ArbitrationError(
                f"missing required observable {value_key}: {route.lane_pair}"
            ) from error
        for lane, value in ((left_lane, left), (right_lane, right)):
            if value.dtype.kind == "f" and value.dtype != np.dtype(np.float64):
                raise ArbitrationError(
                    f"{lane} required floating observable must be FP64: {value_key}"
                )
            if value.dtype.kind == "c" and value.dtype != np.dtype(np.complex128):
                raise ArbitrationError(
                    f"{lane} required complex observable must be FP64: {value_key}"
                )
        if not bool(np.all(np.isfinite(left))) or not bool(np.all(np.isfinite(right))):
            raise ArbitrationError(
                f"non-finite required observable {value_key}: {route.lane_pair}"
            )
        if left.shape != right.shape:
            passed = False
            diagnostic = f"shape mismatch: {left.shape} != {right.shape}"
        elif route.comparator == "exact":
            passed = bool(np.array_equal(left, right))
            diagnostic = "exact comparison"
        elif route.comparator == "allclose":
            rtol, atol = _route_tolerance(route)
            passed = bool(np.allclose(left, right, rtol=rtol, atol=atol))
            diagnostic = f"allclose rtol={rtol} atol={atol}"
        elif route.comparator == "not_worse":
            rtol, atol = _route_tolerance(route)
            upper_bound = left + rtol * np.abs(left) + atol
            passed = bool(np.all(right <= upper_bound))
            diagnostic = f"not_worse rtol={rtol} atol={atol}"
        else:
            raise ArbitrationError(
                f"equivalent comparator requires a case-owned invariant: {value_key}"
            )
        comparisons.append(
            ComparisonResult(
                phase=route.phase,
                observable=route.observable,
                lane_pair=route.lane_pair,
                passed=passed,
                tolerance_bucket=route.tolerance_bucket,
                diagnostic=diagnostic,
            )
        )
    if not comparisons:
        raise ArbitrationError("no applicable comparison routes")
    if quality_band is None:
        return ArbitrationResult(
            verdict="pass" if all(item.passed for item in comparisons) else "fail",
            comparisons=tuple(comparisons),
        )
    band_results = _quality_band_results(quality_band, observations, required_lanes)
    return ArbitrationResult(
        verdict=(
            QUALITY_BAND_VERDICT
            if all(item.passed for item in band_results)
            else "fail"
        ),
        comparisons=tuple(
            replace(
                comparison,
                diagnostic=(
                    f"{_INFORMATIONAL_DIAGNOSTIC_PREFIX}{comparison.diagnostic}"
                ),
            )
            for comparison in comparisons
        ),
        quality_band_results=band_results,
    )
