"""The arbiter's case-owned admission of one published terminal outcome.

Every fixture here is built from literal ``LaneObservation`` values rather than
the case registry: these tests own the arbitration RULE, not any case's
declaration of it.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from examples.jax.parity._manifest import ComparisonRoute
from examples.jax.parity.arbiter import (
    ArbitrationError,
    LaneObservation,
    LaneOutcomeRejection,
    arbitrate,
)
from examples.jax.parity.contracts import AdmittedTerminalOutcome, QualityBand
from examples.jax.parity.provenance import (
    DeviceMetadata,
    ExecutedSource,
    LaneProvenance,
)

# The coil-forces mirror's published JAX GPU stage-one outcome: SciPy
# L-BFGS-B status 2 for stage one and its inert stage-two cold restart.
_CASE_ID = "native-coil-forces"
_ADMITTED_RAW_STATUS = "2,2"
_BUDGET_RAW_STATUS = "stopping_reason=iteration-limit"
_MATCHED_BUDGET = 400
_ADMITTED_NIT = 381
_FINAL_OBJECTIVE = {
    "native-cpu": 4.3972015892540963e-08,
    "jax-cpu": 4.5074090114235766e-08,
    "jax-gpu": 4.561437157279435e-08,
}
_BAND = QualityBand(
    observable="final:objective",
    max_value=1.0e-07,
    derivation="test fixture mirroring the official coil-forces native_default run",
)
_OUTCOME = AdmittedTerminalOutcome(
    case_id=_CASE_ID,
    lane="jax-gpu",
    raw_status=_ADMITTED_RAW_STATUS,
    normalized_status="failed",
    upstream_evidence=(
        "upstream's own official script terminates stage one at status 2 under "
        "a one-ulp start perturbation (sensitivity sample k=5); upstream's "
        "composite outcome there is '2,1', so the later stages differ"
    ),
)


def _provenance(backend_mode: str) -> LaneProvenance:
    is_jax = backend_mode != "native_cpu"
    platform = "gpu" if backend_mode.startswith("jax_gpu_") else "cpu"
    policy = {"SIMSOPT_BACKEND_MODE": backend_mode}
    if platform == "gpu":
        policy.update(
            {
                "SIMSOPT_JAX_TRANSFER_GUARD": "disallow",
                "JAX_TRANSFER_GUARD": "disallow",
            }
        )
    return LaneProvenance(
        repository_commit="d" * 40,
        repository_dirty=False,
        tracked_diff_sha256="e" * 64,
        untracked_files=(),
        executed_sources=(
            ExecutedSource("examples/jax/run_parity.py", "f" * 64, "a" * 40),
        ),
        python_version="3.11.0",
        jax_version="0.10.0" if is_jax else None,
        simsopt_version="1.10.0",
        simsopt_version_commit="g" + "d" * 9,
        simsopt_version_checkout_compatible=True,
        lane_environment_policy=policy,
        jax_effective_transfer_guards=(
            {
                "device_to_device": "disallow",
                "device_to_host": "disallow",
                "host_to_device": "disallow",
            }
            if platform == "gpu"
            else {}
        ),
        devices=(DeviceMetadata(0, platform, "test", 0),) if is_jax else (),
        host_peak_rss_bytes=1024,
        host_peak_rss_method="test fixture",
        device_memory_peak_bytes=None,
        device_memory_status="unavailable: test fixture",
        memory_measurement_scope=(
            "combined import, compile/warmup, and one bounded execution"
        ),
        steady_state_memory_measured=False,
        measurement_synchronization=(
            "jax.block_until_ready over published observation values"
            if is_jax
            else "native synchronous execution"
        ),
        simsoptpp_path=None,
        simsoptpp_sha256=None,
        simsoptpp_version=None,
        simsoptpp_build_commit=None,
        simsoptpp_checkout_compatible=None,
        authoritative=True,
    )


def _routes() -> tuple[ComparisonRoute, ...]:
    return tuple(
        ComparisonRoute(
            phase=phase,
            observable=observable,
            lane_pair=lane_pair,
            applicable=True,
            comparator="allclose",
            tolerance_bucket=bucket,
        )
        for phase, observable, bucket in (
            ("initial", "objective_sum_squares", "native_workflow"),
            ("final", "objective", "mirror_single_stage_final_value"),
        )
        for lane_pair in (
            "native-cpu:jax-cpu",
            "native-cpu:jax-gpu",
            "jax-cpu:jax-gpu",
        )
    )


def _lane(
    lane: str,
    *,
    normalized_status: str,
    raw_status: str,
    success: bool,
    nit: int,
    final_objective: float,
) -> LaneObservation:
    backend_mode = {
        "native-cpu": "native_cpu",
        "jax-cpu": "jax_cpu_parity",
        "jax-gpu": "jax_gpu_parity",
    }[lane]
    return LaneObservation(
        lane=lane,
        backend_mode=backend_mode,
        platform="gpu" if lane == "jax-gpu" else "cpu",
        precision="fp64",
        scale="native_default",
        input_fingerprint="a" * 64,
        configuration_fingerprint="b" * 64,
        effective_construction_fingerprint="c" * 64,
        driver="simsopt_lm_gmres",
        normalized_status=normalized_status,
        raw_status=raw_status,
        success=success,
        nit=nit,
        nfev=nit,
        njev=nit,
        completed_workflow_stages=("construct", "optimize"),
        provenance=_provenance(backend_mode),
        values={
            "initial:objective_sum_squares": np.asarray(1.0, dtype=np.float64),
            "final:objective": np.asarray([final_objective], dtype=np.float64),
        },
    )


def _observations(
    *,
    final_objective: dict[str, float] | None = None,
    budgets: dict[str, int] | None = None,
) -> dict[str, LaneObservation]:
    objectives = final_objective or _FINAL_OBJECTIVE
    nits = budgets or {
        "native-cpu": _MATCHED_BUDGET,
        "jax-cpu": _MATCHED_BUDGET,
        "jax-gpu": _ADMITTED_NIT,
    }
    terminal = {
        "native-cpu": ("budget_exhausted", _BUDGET_RAW_STATUS),
        "jax-cpu": ("budget_exhausted", _BUDGET_RAW_STATUS),
        "jax-gpu": ("failed", _ADMITTED_RAW_STATUS),
    }
    return {
        lane: _lane(
            lane,
            normalized_status=terminal[lane][0],
            raw_status=terminal[lane][1],
            success=False,
            nit=nits[lane],
            final_objective=objectives[lane],
        )
        for lane in ("native-cpu", "jax-cpu", "jax-gpu")
    }


def test_declared_terminal_outcome_certifies_at_most_a_quality_band() -> None:
    result = arbitrate(
        _routes(),
        _observations(),
        quality_band=_BAND,
        case_id=_CASE_ID,
        admitted_terminal_outcomes=(_OUTCOME,),
    )

    assert result.verdict == "quality-band"
    assert result.verdict != "pass"
    assert result.admitted_terminal_lanes == (("jax-gpu", _ADMITTED_RAW_STATUS),)
    assert all(item.passed for item in result.quality_band_results)


def test_admission_never_relabels_the_published_lane_outcome() -> None:
    observations = _observations()

    arbitrate(
        _routes(),
        observations,
        quality_band=_BAND,
        case_id=_CASE_ID,
        admitted_terminal_outcomes=(_OUTCOME,),
    )

    assert observations["jax-gpu"].normalized_status == "failed"
    assert observations["jax-gpu"].raw_status == _ADMITTED_RAW_STATUS
    assert observations["jax-gpu"].success is False


def test_the_same_receipts_are_rejected_without_a_declaration() -> None:
    with pytest.raises(LaneOutcomeRejection, match="did not report scientific success"):
        arbitrate(_routes(), _observations(), quality_band=_BAND)


def test_a_declaration_without_a_band_is_refused() -> None:
    with pytest.raises(
        ArbitrationError, match="admitted terminal outcomes require a quality band"
    ):
        arbitrate(
            _routes(),
            _observations(),
            case_id=_CASE_ID,
            admitted_terminal_outcomes=(_OUTCOME,),
        )


def test_a_raw_status_that_differs_by_one_character_is_not_admitted() -> None:
    other = dataclasses.replace(_OUTCOME, raw_status="2,1")

    with pytest.raises(LaneOutcomeRejection, match="did not report scientific success"):
        arbitrate(
            _routes(),
            _observations(),
            quality_band=_BAND,
            case_id=_CASE_ID,
            admitted_terminal_outcomes=(other,),
        )


def test_a_declared_raw_status_cannot_launder_a_false_success_claim() -> None:
    observations = _observations()
    observations["jax-cpu"] = dataclasses.replace(
        observations["jax-cpu"],
        normalized_status="failed",
        raw_status=_ADMITTED_RAW_STATUS,
        success=True,
        applicability={},
    )

    with pytest.raises(LaneOutcomeRejection, match="provider reported failure"):
        arbitrate(
            _routes(),
            observations,
            quality_band=_BAND,
            case_id=_CASE_ID,
            admitted_terminal_outcomes=(_OUTCOME,),
        )


def test_the_matched_budget_rule_still_binds_the_budget_exhausted_lanes() -> None:
    with pytest.raises(ArbitrationError, match="matched\n?\\s?iteration budget"):
        arbitrate(
            _routes(),
            _observations(
                budgets={
                    "native-cpu": _MATCHED_BUDGET,
                    "jax-cpu": _MATCHED_BUDGET - 1,
                    "jax-gpu": _ADMITTED_NIT,
                }
            ),
            quality_band=_BAND,
            case_id=_CASE_ID,
            admitted_terminal_outcomes=(_OUTCOME,),
        )


def test_an_admission_cannot_replace_the_last_budget_exhausted_lane() -> None:
    observations = _observations()
    for lane in ("native-cpu", "jax-cpu"):
        observations[lane] = dataclasses.replace(
            observations[lane],
            normalized_status="failed",
            raw_status=_ADMITTED_RAW_STATUS,
            applicability={},
        )

    # One declaration per lane: even with every lane admitted, no lane is left
    # to carry the matched budget, so the band path refuses.
    every_lane = tuple(
        dataclasses.replace(_OUTCOME, lane=lane)
        for lane in ("native-cpu", "jax-cpu", "jax-gpu")
    )

    with pytest.raises(ArbitrationError, match="matched\n?\\s?iteration budget"):
        arbitrate(
            _routes(),
            observations,
            quality_band=_BAND,
            case_id=_CASE_ID,
            admitted_terminal_outcomes=every_lane,
        )


def test_an_admitted_lane_outside_the_band_still_fails() -> None:
    result = arbitrate(
        _routes(),
        _observations(final_objective={**_FINAL_OBJECTIVE, "jax-gpu": 4.6e-07}),
        quality_band=_BAND,
        case_id=_CASE_ID,
        admitted_terminal_outcomes=(_OUTCOME,),
    )

    assert result.verdict == "fail"
    assert result.admitted_terminal_lanes == (("jax-gpu", _ADMITTED_RAW_STATUS),)
    assert {item.lane: item.passed for item in result.quality_band_results} == {
        "native-cpu": True,
        "jax-cpu": True,
        "jax-gpu": False,
    }


@pytest.mark.parametrize(
    (
        "case_id",
        "lane",
        "raw_status",
        "normalized_status",
        "upstream_evidence",
        "expected",
    ),
    (
        (
            _CASE_ID,
            "jax-gpu",
            "2,2",
            "budget_exhausted",
            "measured",
            "normalized status must be 'failed'",
        ),
        (
            _CASE_ID,
            "jax-gpu",
            "2,2",
            "converged",
            "measured",
            "normalized status must be 'failed'",
        ),
        (_CASE_ID, "jax-gpu", "", "failed", "measured", "requires a raw status"),
        (_CASE_ID, "jax-gpu", "2,2", "failed", "", "requires upstream evidence"),
        ("", "jax-gpu", "2,2", "failed", "measured", "requires an owning case_id"),
        (_CASE_ID, "", "2,2", "failed", "measured", "requires the admitted lane"),
    ),
)
def test_admitted_terminal_outcome_refuses_an_unbound_declaration(
    case_id: str,
    lane: str,
    raw_status: str,
    normalized_status: str,
    upstream_evidence: str,
    expected: str,
) -> None:
    with pytest.raises(ValueError, match=expected):
        AdmittedTerminalOutcome(
            case_id=case_id,
            lane=lane,
            raw_status=raw_status,
            normalized_status=normalized_status,
            upstream_evidence=upstream_evidence,
        )


def test_an_admission_authorized_for_another_case_is_refused() -> None:
    foreign = dataclasses.replace(_OUTCOME, case_id="native-boozerqa")

    with pytest.raises(ArbitrationError, match="belongs to another case"):
        arbitrate(
            _routes(),
            _observations(),
            quality_band=_BAND,
            case_id=_CASE_ID,
            admitted_terminal_outcomes=(foreign,),
        )


def test_an_admission_without_the_arbitrated_case_id_is_refused() -> None:
    with pytest.raises(ArbitrationError, match="require the arbitrated case_id"):
        arbitrate(
            _routes(),
            _observations(),
            quality_band=_BAND,
            admitted_terminal_outcomes=(_OUTCOME,),
        )


def test_an_admission_at_the_bounded_scale_is_refused() -> None:
    bounded = {
        lane: dataclasses.replace(observation, scale="bounded")
        for lane, observation in _observations().items()
    }

    with pytest.raises(ArbitrationError, match="requires the native_default scale"):
        arbitrate(
            _routes(),
            bounded,
            quality_band=_BAND,
            case_id=_CASE_ID,
            admitted_terminal_outcomes=(_OUTCOME,),
        )


def test_a_non_finite_end_value_is_refused_even_when_the_outcome_is_admitted() -> None:
    non_finite = _observations(
        final_objective={**_FINAL_OBJECTIVE, "jax-gpu": float("nan")}
    )

    with pytest.raises(ArbitrationError, match="non-finite"):
        arbitrate(
            _routes(),
            non_finite,
            quality_band=_BAND,
            case_id=_CASE_ID,
            admitted_terminal_outcomes=(_OUTCOME,),
        )


@pytest.mark.parametrize("failed_lane", ("native-cpu", "jax-cpu"))
def test_the_same_raw_status_on_another_lane_is_not_admitted(failed_lane: str) -> None:
    observations = _observations()
    swapped = {
        **observations,
        failed_lane: dataclasses.replace(
            observations[failed_lane],
            normalized_status="failed",
            raw_status=_ADMITTED_RAW_STATUS,
            nit=_ADMITTED_NIT,
            nfev=_ADMITTED_NIT,
            njev=_ADMITTED_NIT,
        ),
        "jax-gpu": dataclasses.replace(
            observations["jax-gpu"],
            normalized_status="budget_exhausted",
            raw_status=_BUDGET_RAW_STATUS,
            nit=_MATCHED_BUDGET,
            nfev=_MATCHED_BUDGET,
            njev=_MATCHED_BUDGET,
        ),
    }

    with pytest.raises(LaneOutcomeRejection, match="did not report scientific success"):
        arbitrate(
            _routes(),
            swapped,
            quality_band=_BAND,
            case_id=_CASE_ID,
            admitted_terminal_outcomes=(_OUTCOME,),
        )
