"""Admitted terminal outcomes in the case registry are narrow, band-bound and backed by upstream's own samples (C14)."""

from __future__ import annotations

import pytest
from examples.jax.parity.cases import CaseDefinition, get_case, implemented_case_ids
from examples.jax.parity.contracts import AdmittedTerminalOutcome
from examples.jax.parity.official_reference import load_official_sensitivity

#: The only case that declares an admitted provider failure, and the exact outcome it admits.
ADMITTING_CASES = {"native-coil-forces": (("jax-gpu", "2,2"),)}


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
    assert case.native_default_quality_band is not None
    assert case.work_budget_contract is not None
    assert "native_default" not in case.work_budget_contract.scales
    for outcome in case.native_default_admitted_terminal_outcomes:
        assert outcome.case_id == case_id
        assert outcome.normalized_status == "failed"
        assert outcome.upstream_evidence


@pytest.mark.parametrize("case_id", sorted(ADMITTING_CASES))
def test_the_admitted_stage_one_status_occurs_in_upstreams_own_samples(
    case_id: str,
) -> None:
    sensitivity = load_official_sensitivity(case_id)
    upstream_stage_one = {run.provider_calls[0].status for run in sensitivity.runs}
    for outcome in get_case(case_id).native_default_admitted_terminal_outcomes:
        stage_one = int(outcome.raw_status.split(",")[0])
        assert stage_one in upstream_stage_one, (outcome.raw_status, upstream_stage_one)
        assert stage_one == 2


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
