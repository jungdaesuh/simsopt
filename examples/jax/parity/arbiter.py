"""Fail-closed scientific comparison of isolated parity lane receipts."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping

import numpy as np
from examples.jax.outer_optimizer_policy import (
    OuterOptimizerPolicy,
    policy_owns_parity_case,
)
from examples.jax.parity._manifest import ComparisonRoute
from examples.jax.parity.contracts import (
    AdmittedTerminalOutcome,
    ComparisonResult,
    EndStateResult,
    QualityBand,
    QualityBandResult,
    StageWiseContract,
    UpstreamEndState,
    UpstreamEndStates,
)
from examples.jax.parity.provenance import LaneProvenance
from examples.jax.parity.work_budget import WorkBudgetContract
from simsopt_jax.config import ExecutionIntent
from simsopt_jax.examples import ExecutionScale
from simsopt_jax.parity_tolerances import parity_ladder_tolerances

_REQUIRED_LANES = frozenset({"native-cpu", "jax-cpu", "jax-gpu"})
_REQUIRED_PAIRS = frozenset(
    {"native-cpu:jax-cpu", "native-cpu:jax-gpu", "jax-cpu:jax-gpu"}
)
QUALITY_BAND_VERDICT = "quality-band"
_INFORMATIONAL_DIAGNOSTIC_PREFIX = "informational (quality-band, non-certifying): "


class ArbitrationError(ValueError):
    """Lane evidence is structurally incapable of supporting parity."""


class LaneOutcomeRejection(ArbitrationError):
    """A compared lane reported no scientific success of its own.

    Raised only where the arbiter reads a lane's OWN terminal outcome, never
    for a harness, contract or integrity violation. A caller may record this
    as the scientific result of an attempt; every other ``ArbitrationError``
    says the evidence is unusable and must abort the run.
    """


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
    work_budget_admitted: bool = False
    admitted_terminal_lanes: tuple[tuple[str, str], ...] = ()
    end_state_results: tuple[EndStateResult, ...] = ()
    stage_wise: StageWiseContract | None = None


@dataclass(frozen=True)
class _TerminalAdmission:
    """Which non-success terminal outcomes a case's contracts let stand."""

    work_budget_admitted: bool
    admitted_terminal_lanes: tuple[tuple[str, str], ...]


def _validate_lanes(
    observations: Mapping[str, LaneObservation],
    required_lanes: frozenset[str],
    expected_workflow_stages: tuple[str, ...] | None,
    execution_intent: ExecutionIntent = "parity",
    quality_band: QualityBand | None = None,
    case_id: str | None = None,
    example_id: str | None = None,
    outer_optimizer_policy: OuterOptimizerPolicy | None = None,
    work_budget_contract: WorkBudgetContract | None = None,
    admitted_terminal_outcomes: tuple[AdmittedTerminalOutcome, ...] = (),
    upstream_end_states: UpstreamEndStates | None = None,
    stage_wise: StageWiseContract | None = None,
) -> _TerminalAdmission:
    if outer_optimizer_policy is not None and not policy_owns_parity_case(
        outer_optimizer_policy, case_id=case_id, example_id=example_id
    ):
        raise ArbitrationError(
            "outer optimizer policy does not own this parity case/example"
        )
    if execution_intent not in ("fast", "parity"):
        raise ArbitrationError(f"invalid execution intent: {execution_intent}")
    if not required_lanes or not required_lanes <= _REQUIRED_LANES:
        raise ArbitrationError(f"invalid required lanes: {sorted(required_lanes)}")
    missing = required_lanes - set(observations)
    if missing:
        raise ArbitrationError(f"missing required lane: {sorted(missing)}")
    required_scales = {observations[lane].scale for lane in required_lanes}
    if quality_band is not None and required_scales != {quality_band.scale}:
        raise ArbitrationError(
            f"quality-band certification requires the {quality_band.scale} scale"
        )
    if admitted_terminal_outcomes and quality_band is None:
        raise ArbitrationError("admitted terminal outcomes require a quality band")
    if (
        admitted_terminal_outcomes
        and quality_band is not None
        and quality_band.scale != "native_default"
    ):
        raise ArbitrationError(
            "admitted terminal outcomes require a native_default quality band"
        )
    # The end-state contract is bound at this seam like an admission: to ONE
    # case and ONE scale, and alone. It admits no terminal outcome, so every
    # lane still has to pass the success gate below.
    if upstream_end_states is not None:
        if quality_band is not None or admitted_terminal_outcomes:
            raise ArbitrationError(
                "upstream end states cannot combine with a quality band or "
                "admitted terminal outcomes"
            )
        if case_id is None:
            raise ArbitrationError("upstream end states require the arbitrated case_id")
        if upstream_end_states.case_id != case_id:
            raise ArbitrationError(
                "upstream end states belong to another case: "
                f"{upstream_end_states.case_id!r} != {case_id!r}"
            )
        if required_scales != {upstream_end_states.scale}:
            raise ArbitrationError(
                "upstream end-state acceptance requires the "
                f"{upstream_end_states.scale} scale"
            )
    # A stage-wise contract is bound like the end-state set: to ONE case and
    # ONE scale. It admits no terminal outcome and cannot stand beside a band.
    if stage_wise is not None:
        if quality_band is not None or admitted_terminal_outcomes:
            raise ArbitrationError(
                "a stage-wise contract cannot combine with a quality band or "
                "admitted terminal outcomes"
            )
        if case_id is None:
            raise ArbitrationError(
                "a stage-wise contract requires the arbitrated case_id"
            )
        if stage_wise.case_id != case_id:
            raise ArbitrationError(
                "stage-wise contract belongs to another case: "
                f"{stage_wise.case_id!r} != {case_id!r}"
            )
        if required_scales != {stage_wise.scale}:
            raise ArbitrationError(
                f"the stage-wise contract requires the {stage_wise.scale} scale"
            )
    # Ownership is checked at this seam, not only in the registry: an admission
    # authorized for one case can never be handed to another.
    if admitted_terminal_outcomes and case_id is None:
        raise ArbitrationError(
            "admitted terminal outcomes require the arbitrated case_id"
        )
    foreign = sorted(
        outcome.case_id
        for outcome in admitted_terminal_outcomes
        if outcome.case_id != case_id
    )
    if foreign:
        raise ArbitrationError(
            f"admitted terminal outcome belongs to another case: {foreign} != {case_id!r}"
        )
    admitted_outcomes = frozenset(
        (outcome.lane, outcome.raw_status) for outcome in admitted_terminal_outcomes
    )
    # An admission matches the receipt's published strings exactly; the lane's
    # own ``failed``/``success`` verdict is carried through untouched.
    admitted_terminal_lanes = tuple(
        sorted(
            (lane, observations[lane].raw_status)
            for lane in required_lanes
            if observations[lane].normalized_status == "failed"
            and observations[lane].success is False
            and (lane, observations[lane].raw_status) in admitted_outcomes
        )
    )
    admitted_lanes = {lane for lane, _ in admitted_terminal_lanes}
    budget_exhausted_lanes = {
        lane
        for lane in required_lanes
        if observations[lane].normalized_status == "budget_exhausted"
    }
    all_budget_exhausted = budget_exhausted_lanes == set(required_lanes)
    quality_band_budget_admitted = quality_band is not None and (
        budget_exhausted_lanes | admitted_lanes == set(required_lanes)
    )
    observed_scales = {observations[lane].scale for lane in required_lanes}
    work_budget_admitted = (
        quality_band is None
        and work_budget_contract is not None
        and all_budget_exhausted
        and len(observed_scales) == 1
        and next(iter(observed_scales)) in work_budget_contract.scales
    )
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
        if (
            lane.startswith("jax-")
            and outer_optimizer_policy is not None
            and observation.driver != outer_optimizer_policy.expected_driver
        ):
            raise ArbitrationError(
                f"{lane} driver differs from its declared outer optimizer policy"
            )
        if (
            lane.startswith("jax-")
            and outer_optimizer_policy is None
            and any(
                forbidden in observation.driver.lower()
                for forbidden in ("scipy", "optimistix", "optax", "host_callback")
            )
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
        if observation.normalized_status == "budget_exhausted" and observation.success:
            raise LaneOutcomeRejection(f"{lane} budget_exhausted cannot report success")
        if observation.normalized_status == "failed" and observation.success:
            raise LaneOutcomeRejection(f"{lane} provider reported failure")
        if not observation.success and not (
            quality_band_budget_admitted or work_budget_admitted
        ):
            raise LaneOutcomeRejection(f"{lane} did not report scientific success")
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
    if quality_band_budget_admitted:
        budgets = {observations[lane].nit for lane in budget_exhausted_lanes}
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
        observations[lane].normalized_status
        for lane in sorted(required_lanes)
        if lane not in admitted_lanes
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
    return _TerminalAdmission(
        work_budget_admitted=work_budget_admitted,
        admitted_terminal_lanes=admitted_terminal_lanes,
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


def _require_fp64(lane: str, value_key: str, value: np.ndarray) -> None:
    if value.dtype.kind == "f" and value.dtype != np.dtype(np.float64):
        raise ArbitrationError(
            f"{lane} required floating observable must be FP64: {value_key}"
        )
    if value.dtype.kind == "c" and value.dtype != np.dtype(np.complex128):
        raise ArbitrationError(
            f"{lane} required complex observable must be FP64: {value_key}"
        )


def stopping_bound_gaps(
    left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row solved-state gap and its bound (PLAN.md amendments 6 and 7, B1', B1'').

    Each row of ``left`` and ``right`` is one lane's solution of one penalty
    least-squares solve: the state x, the solve's label target t, its end
    ``norm(J^T r)``, ``lambda_min(J^T J)`` at x and ``s = norm(dx*/dt)``, all
    from the lane's own run. With ``b = J^T r`` the gradient of
    ``f_t = 1/2 |r(x, t)|^2`` and ``mu`` a lower bound on the curvature along the
    segment to the root, each lane lies within ``|b| / mu`` of its own root
    x*(t); the two lanes' roots differ by at most ``s |t_l - t_r|`` to first
    order; so ``|x_l - x_r| <= (|b_l| + |b_r|) / mu + s |t_l - t_r|`` with ``mu``
    the smaller Gauss-Newton curvature and ``s`` the larger sensitivity.
    Returns the Euclidean gap per row and that bound per row.
    """
    gap = np.linalg.norm(left[..., :-4] - right[..., :-4], axis=-1)
    bound = (left[..., -3] + right[..., -3]) / np.minimum(
        left[..., -2], right[..., -2]
    ) + np.maximum(left[..., -1], right[..., -1]) * np.abs(
        left[..., -4] - right[..., -4]
    )
    return gap, bound


def _compare(
    route: ComparisonRoute, left: np.ndarray, right: np.ndarray
) -> tuple[bool, str]:
    """Apply ``route``'s comparator to ``left`` against ``right``.

    The one comparison every lane-pair route and every lane-versus-upstream
    end-state match runs, so both use the same comparator and tolerance.
    """
    if left.shape != right.shape:
        return False, f"shape mismatch: {left.shape} != {right.shape}"
    if route.comparator == "exact":
        return bool(np.array_equal(left, right)), "exact comparison"
    if route.comparator == "allclose":
        rtol, atol = _route_tolerance(route)
        return (
            bool(np.allclose(left, right, rtol=rtol, atol=atol)),
            f"allclose rtol={rtol} atol={atol}",
        )
    if route.comparator == "not_worse":
        rtol, atol = _route_tolerance(route)
        upper_bound = left + rtol * np.abs(left) + atol
        return bool(np.all(right <= upper_bound)), f"not_worse rtol={rtol} atol={atol}"
    if route.comparator == "stopping_bound":
        gap, bound = stopping_bound_gaps(left, right)
        curvature = np.minimum(left[..., -2], right[..., -2])
        passed = bool(np.all(curvature > 0.0) and np.all(gap <= bound))
        ratio = np.where(
            gap == 0.0,
            0.0,
            np.where(bound > 0.0, gap / np.where(bound > 0.0, bound, 1.0), np.inf),
        )
        return passed, f"stopping_bound: max gap/bound {float(np.max(ratio)):.3e}"
    raise ArbitrationError(
        "equivalent comparator requires a case-owned invariant: "
        f"{route.phase}:{route.observable}"
    )


def _end_state_routes(
    routes: tuple[ComparisonRoute, ...], upstream_end_states: UpstreamEndStates
) -> Mapping[str, ComparisonRoute]:
    """Pick, per judged key, the one route whose comparator judges the upstream match.

    Every selected route of a judged key must be applicable and share one
    ``exact`` or ``allclose`` comparator and one tolerance bucket, so the
    lane-versus-upstream comparison is exactly what the case already requires
    of two lanes. ``not_worse`` is one-sided and cannot say two states match.
    """
    judged: dict[str, ComparisonRoute] = {}
    for key in upstream_end_states.observables:
        key_routes = tuple(
            route for route in routes if f"{route.phase}:{route.observable}" == key
        )
        if not key_routes or not all(route.applicable for route in key_routes):
            raise ArbitrationError(
                f"upstream end-state observable has no applicable route: {key}"
            )
        comparators = {route.comparator for route in key_routes}
        buckets = {route.tolerance_bucket for route in key_routes}
        if (
            len(comparators) != 1
            or not comparators <= {"exact", "allclose"}
            or len(buckets) != 1
        ):
            raise ArbitrationError(
                "upstream end-state observable requires one exact or allclose "
                f"comparator and one tolerance bucket: {key}"
            )
        judged[key] = key_routes[0]
    return MappingProxyType(judged)


def _checked_end_state(
    judged_routes: Mapping[str, ComparisonRoute], values: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """The judged values of one end state, present, finite and FP64, in receipt form."""
    checked: dict[str, np.ndarray] = {}
    for key in judged_routes:
        if key not in values:
            raise ArbitrationError(f"missing upstream end-state observable {key}")
        value = np.atleast_1d(np.asarray(values[key]))
        _require_fp64("end state", key, value)
        if not bool(np.all(np.isfinite(value))):
            raise ArbitrationError(f"non-finite upstream end-state observable {key}")
        checked[key] = value
    return checked


def _same_branch(
    judged_routes: Mapping[str, ComparisonRoute],
    candidate: Mapping[str, np.ndarray],
    representative: UpstreamEndState,
) -> bool:
    """Whether ``candidate`` (``left``) passes every judged route against ``representative`` (``right``)."""
    return all(
        _compare(route, candidate[key], representative.values[key])[0]
        for key, route in judged_routes.items()
    )


def upstream_branch_representatives(
    upstream_end_states: UpstreamEndStates,
    routes: tuple[ComparisonRoute, ...],
) -> tuple[UpstreamEndState, ...]:
    """Upstream's distinct end-state branches, each as its lowest-``k`` draw.

    The draws are visited in ascending ``k``; a draw joins the branch of the
    first representative it matches under every judged key's route comparator
    (the draw as ``left``, the representative as ``right``, as a lane is
    judged), and otherwise starts a new branch. A set whose draws all lie on
    ONE branch is refused: one branch means the end state is a function of the
    input, and the lane-versus-lane routes must decide instead.
    """
    judged_routes = _end_state_routes(routes, upstream_end_states)
    representatives: list[UpstreamEndState] = []
    for state in upstream_end_states.states:
        values = _checked_end_state(judged_routes, state.values)
        if not any(
            _same_branch(judged_routes, values, representative)
            for representative in representatives
        ):
            representatives.append(state)
    if len(representatives) < 2:
        raise ArbitrationError(
            "upstream end states span one branch under the case's route "
            "comparators; the lane-versus-lane routes decide such a case"
        )
    return tuple(representatives)


def upstream_end_state_matches(
    upstream_end_states: UpstreamEndStates,
    routes: tuple[ComparisonRoute, ...],
    values: Mapping[str, np.ndarray],
) -> tuple[int, ...]:
    """Return the ascending ``k`` of every upstream branch representative one end state matches.

    The representatives are :func:`upstream_branch_representatives`: a draw
    that is not its branch's lowest-``k`` member is never matched on its own,
    so the acceptance region is one comparator ball per branch. A
    representative matches when, for EVERY judged key, ``values[key]`` (as
    ``left``) passes that key's route comparator against the representative's
    value (as ``right``); a shape mismatch is no match. A scalar is compared in
    the published receipt form, a one-element array, whether it comes from a
    receipt or from an in-process observation. The judged routes are chosen
    from ``routes`` under the arbiter's own refusal rules, and each value must
    be present, finite and FP64. An engineering acceptance against upstream's
    own scatter, never an equivalence test.
    """
    judged_routes = _end_state_routes(routes, upstream_end_states)
    checked = _checked_end_state(judged_routes, values)
    return tuple(
        representative.k
        for representative in upstream_branch_representatives(
            upstream_end_states, routes
        )
        if _same_branch(judged_routes, checked, representative)
    )


def _end_state_results(
    upstream_end_states: UpstreamEndStates,
    routes: tuple[ComparisonRoute, ...],
    observations: Mapping[str, LaneObservation],
    required_lanes: frozenset[str],
) -> tuple[EndStateResult, ...]:
    """Match every compared lane's end state against upstream's branch representatives."""
    results: list[EndStateResult] = []
    for lane in sorted(required_lanes):
        observation = observations[lane]
        for key in upstream_end_states.observables:
            if not observation.applicability.get(key, False):
                raise ArbitrationError(
                    f"upstream end-state observable is not applicable {key}: {lane}"
                )
        matched_draws = upstream_end_state_matches(
            upstream_end_states, routes, observation.values
        )
        results.append(
            EndStateResult(
                lane=lane, matched_draws=matched_draws, passed=bool(matched_draws)
            )
        )
    return tuple(results)


def _require_stage_wise_routes(
    stage_wise: StageWiseContract,
    routes: tuple[ComparisonRoute, ...],
    required_lanes: frozenset[str],
) -> None:
    """Refuse a stage-wise contract whose keys the route matrix does not judge.

    A deciding key must be applicable on every required lane pair (it is the
    harness half of the stage-wise proof); an informational key must be an
    applicable route too, or there would be nothing to record.
    """
    required_pairs = _required_lane_pairs(required_lanes)
    for key in (
        *stage_wise.deciding_observables,
        *stage_wise.informational_observables,
    ):
        key_routes = tuple(
            route for route in routes if f"{route.phase}:{route.observable}" == key
        )
        if (
            not key_routes
            or not all(route.applicable for route in key_routes)
            or {route.lane_pair for route in key_routes} != required_pairs
        ):
            raise ArbitrationError(
                f"stage-wise observable has no applicable complete route matrix: {key}"
            )


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
    case_id: str | None = None,
    example_id: str | None = None,
    outer_optimizer_policy: OuterOptimizerPolicy | None = None,
    work_budget_contract: WorkBudgetContract | None = None,
    admitted_terminal_outcomes: tuple[AdmittedTerminalOutcome, ...] = (),
    upstream_end_states: UpstreamEndStates | None = None,
    stage_wise: StageWiseContract | None = None,
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

    A case-owned ``work_budget_contract`` admits all required lanes' honest
    budget exits at its declared scales; numerical comparisons still decide.

    A case-owned ``admitted_terminal_outcomes`` widens the band path's
    admissible terminal states from ``budget_exhausted`` alone to those exact
    published outcomes, for each declared lane whose receipt matches its
    declared raw status string for string. An admitted lane is exempt from
    the matched-budget clause (it stopped early by definition) and from the
    single-category rule; every other required lane must still be
    ``budget_exhausted`` at one shared budget, so at least one such lane must
    exist. The admitted lane keeps its own ``failed`` status and ``success``
    false, the band still decides, the ceiling is still ``quality-band``, and
    every admitted lane is named in ``admitted_terminal_lanes``.

    A case-owned ``upstream_end_states`` serves a workflow whose end state its
    input does not determine (upstream's own one-ulp draws land on several
    end states). It is an engineering acceptance against upstream's own
    scatter, never equivalence. Each lane must still pass the success gate
    (the contract admits no budget or failure outcome); each lane's end state
    must then match one upstream branch representative (the lowest-``k`` draw
    of each distinct branch, :func:`upstream_branch_representatives`) on every
    judged key, under that key's own route comparator and tolerance. The lane-pair comparisons of the
    judged keys are recorded as informational; every other route still
    decides. The verdict is ``quality-band`` at best, never ``pass``, and
    ``end_state_results`` names the representatives each lane matched.

    A case-owned ``stage_wise`` contract serves a workflow judged stage by
    stage from shared states. Its deciding keys must be applicable routes on
    every required pair and decide like any route; the lane-pair comparisons
    of its informational keys (the chained, path-dependent end point) are
    recorded as informational. Every lane must still pass the success gate
    (a work budget may admit budget exits as usual). An upstream end-state set
    passed beside it is matched and recorded, informational: under a
    stage-wise contract ``end_state_results`` decide nothing. The verdict is
    ``quality-band`` at best, never ``pass``.
    """
    admission = _validate_lanes(
        observations,
        required_lanes,
        expected_workflow_stages,
        execution_intent,
        quality_band,
        case_id,
        example_id,
        outer_optimizer_policy,
        work_budget_contract,
        admitted_terminal_outcomes,
        upstream_end_states,
        stage_wise,
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
        _require_fp64(left_lane, value_key, left)
        _require_fp64(right_lane, value_key, right)
        if not bool(np.all(np.isfinite(left))) or not bool(np.all(np.isfinite(right))):
            raise ArbitrationError(
                f"non-finite required observable {value_key}: {route.lane_pair}"
            )
        passed, diagnostic = _compare(route, left, right)
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
    if stage_wise is not None:
        _require_stage_wise_routes(stage_wise, selected_routes, required_lanes)
        informational = frozenset(stage_wise.informational_observables)
        return ArbitrationResult(
            verdict=(
                QUALITY_BAND_VERDICT
                if all(
                    comparison.passed
                    for comparison in comparisons
                    if f"{comparison.phase}:{comparison.observable}"
                    not in informational
                )
                else "fail"
            ),
            comparisons=tuple(
                replace(
                    comparison,
                    diagnostic=(
                        f"{_INFORMATIONAL_DIAGNOSTIC_PREFIX}{comparison.diagnostic}"
                    ),
                )
                if f"{comparison.phase}:{comparison.observable}" in informational
                else comparison
                for comparison in comparisons
            ),
            work_budget_admitted=admission.work_budget_admitted,
            end_state_results=(
                _end_state_results(
                    upstream_end_states, selected_routes, observations, required_lanes
                )
                if upstream_end_states is not None
                else ()
            ),
            stage_wise=stage_wise,
        )
    if upstream_end_states is not None:
        end_state_results = _end_state_results(
            upstream_end_states, selected_routes, observations, required_lanes
        )
        judged = frozenset(upstream_end_states.observables)
        deciding_passed = all(
            comparison.passed
            for comparison in comparisons
            if f"{comparison.phase}:{comparison.observable}" not in judged
        )
        return ArbitrationResult(
            verdict=(
                QUALITY_BAND_VERDICT
                if deciding_passed and all(item.passed for item in end_state_results)
                else "fail"
            ),
            comparisons=tuple(
                replace(
                    comparison,
                    diagnostic=(
                        f"{_INFORMATIONAL_DIAGNOSTIC_PREFIX}{comparison.diagnostic}"
                    ),
                )
                if f"{comparison.phase}:{comparison.observable}" in judged
                else comparison
                for comparison in comparisons
            ),
            work_budget_admitted=admission.work_budget_admitted,
            end_state_results=end_state_results,
        )
    if quality_band is None:
        return ArbitrationResult(
            verdict="pass" if all(item.passed for item in comparisons) else "fail",
            comparisons=tuple(comparisons),
            work_budget_admitted=admission.work_budget_admitted,
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
        admitted_terminal_lanes=admission.admitted_terminal_lanes,
    )
