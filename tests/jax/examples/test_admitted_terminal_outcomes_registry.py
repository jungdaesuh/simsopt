"""Admitted terminal outcomes in the case registry are narrow, band-bound and backed by upstream's own samples."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import pytest
from examples.jax.parity.cases import (
    COIL_FORCES_UPSTREAM_TERMINAL_OUTCOMES,
    CaseDefinition,
    get_case,
    implemented_case_ids,
)
from examples.jax.parity.contracts import AdmittedTerminalOutcome
from examples.jax.parity.official_reference import load_official_sensitivity

#: The only case that declares an admitted provider failure, and the exact outcomes it admits: every
#: composite outcome other than the budget pair that upstream's own one-ulp draws produced, on each lane.
ADMITTING_CASES = {
    "native-coil-forces": tuple(
        (lane, raw_status)
        for lane in ("native-cpu", "jax-cpu", "jax-gpu")
        for raw_status in ("2,1", "1,2", "2,2")
    )
}


def _admitting_cases() -> dict[str, tuple[tuple[str, str], ...]]:
    return {
        case_id: tuple(
            (outcome.lane, outcome.raw_status)
            for outcome in get_case(case_id).native_default_admitted_terminal_outcomes
        )
        for case_id in implemented_case_ids()
        if get_case(case_id).native_default_admitted_terminal_outcomes
    }


def test_only_coil_forces_admits_a_provider_failure_and_only_the_abnormal_pair() -> (
    None
):
    assert _admitting_cases() == ADMITTING_CASES


@pytest.mark.parametrize("case_id", sorted(ADMITTING_CASES))
def test_every_admitted_outcome_is_band_bound_and_keeps_the_failed_category(
    case_id: str,
) -> None:
    case = get_case(case_id)
    assert case.quality_band("native_default") is not None
    assert case.work_budget_contract is not None
    assert "native_default" not in case.work_budget_contract.scales
    for outcome in case.native_default_admitted_terminal_outcomes:
        assert outcome.case_id == case_id
        assert outcome.normalized_status == "failed"
        assert outcome.upstream_evidence


def test_every_admitted_coil_forces_outcome_is_one_upstream_produced() -> None:
    """Each admitted composite status names the upstream draws that produced it.

    Draws inside the tracked nine (k = 0..8) are checked against the tracked
    sensitivity record here; the draws of the pre-registered extension
    (k = 9..40) are named in the evidence text, and their per-run records are
    local evidence outside this repository.
    """
    sensitivity = load_official_sensitivity("native-coil-forces")
    tracked = {
        run.k: ",".join(str(call.status) for call in run.provider_calls)
        for run in sensitivity.runs
    }
    assert "1,1" not in COIL_FORCES_UPSTREAM_TERMINAL_OUTCOMES
    for raw_status, draws in COIL_FORCES_UPSTREAM_TERMINAL_OUTCOMES.items():
        assert draws
        for k in draws:
            if k in tracked:
                assert tracked[k] == raw_status, (k, tracked[k], raw_status)
    # The tracked nine show exactly one non-budget outcome, and it is listed.
    assert {k: status for k, status in tracked.items() if status != "1,1"} == {
        5: "2,1"
    }
    for outcome in get_case("native-coil-forces").native_default_admitted_terminal_outcomes:
        draws = COIL_FORCES_UPSTREAM_TERMINAL_OUTCOMES[outcome.raw_status]
        assert f"k = {', '.join(map(str, draws))} of k = 0..40" in outcome.upstream_evidence


def test_an_admitted_outcome_without_a_band_is_refused() -> None:
    case = get_case("native-coil-forces")
    with pytest.raises(ValueError, match="require a native_default quality band"):
        CaseDefinition(
            case_id=case.case_id,
            create_input=case.create_input,
            execute=case.execute,
            work_budget_contract=case.work_budget_contract,
            native_default_admitted_terminal_outcomes=(
                AdmittedTerminalOutcome(
                    case_id=case.case_id,
                    lane="jax-gpu",
                    raw_status="2,2",
                    normalized_status="failed",
                    upstream_evidence="test",
                ),
            ),
        )
