"""Artifact-free contract coverage for the F3 campaign's frozen arithmetic.

The campaign harness itself only resolves under the pinned instrument tree and
only runs when the box is free, so none of it is exercisable here.  Its
arithmetic is: policy identity, per-row contract binding, counter liveness, the
three anchor formulas, the quality gate, the dual verdict rule, cap accounting,
and ``validate``.  All of that lives in the contract module beside it, which
imports nothing but the standard library — so every number the charter calls
non-amendable is checked here, on synthetic inputs, in the ordinary test
environment.

The charter's own worked figures are used as oracles wherever it states one, so
a drift in these formulas fails against the document rather than against a
value this file invented.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from benchmarks.flat675_fused_campaign_contract import (
    ARCHIVED_B3_PROCESS_WALL,
    ARCHIVED_STEADY_PER_EVAL,
    BQ_MAX_MAXITER,
    BQ_MAX_PROBES_PER_LANE,
    BQ_MAX_SEARCH_SECONDS,
    F3_CHARTER_COMMIT,
    F3_CHARTER_LINEAGE,
    F3_CHARTER_SHA256,
    F3_ROW_SCHEMA,
    F3_RUN_MANIFEST_SCHEMA,
    GRADIENT_FACTOR_K,
    L1_LANE,
    L2_LANE,
    MAX_SOLVE_CHILD_PROCESSES,
    MAX_TIMED_LEGS,
    MAXFUN_MULTIPLIER,
    PAIR_COUNT,
    POLICY_FIELDS,
    POLICY_MAXCOR,
    POLICY_MAXLS,
    RUNG_BUDGETS,
    CapLedger,
    F3ContractError,
    Verdict,
    adjudicate_rung,
    b3_anchor,
    b37_anchor,
    bq_anchor,
    bq_quality_failures,
    budget_search_breaches,
    counter_liveness_failures,
    f3_contract_sha256,
    fixed_budget_quality_failures,
    observed_policy_sha256,
    pair_speedups,
    policy_identity_failures,
    policy_identity_sha256,
    policy_payload,
    validate_run_dir,
)

# The charter's frozen figures, transcribed from the document under test.
CHARTER_ARCHIVED_B3_WALL = 58.702
CHARTER_ARCHIVED_PER_EVAL = 52.807 / 9
CHARTER_MAXCOR = 300
CHARTER_MAXLS = 8
CHARTER_GTOL = 0.001
CHARTER_FTOL = 0.0

FAIR_BAR_SHA = "6ca00d035ca374bd16085ae8a8cca814ccaa48ed9637a6184f31811fe7e7b87c"
BUNDLE_SHA = "84febc05d195d84c0802205b2b4c85ea1fa38faa7ff856efca7c12d980647c0c"
PRODUCTION_COMMIT = "0" * 40
INSTRUMENT_COMMIT = "1c23f6c5f8964c74cc60f63d81b7f93f2db852f3"


# --------------------------------------------------------------------------
# Charter identity
# --------------------------------------------------------------------------


def test_charter_identity_is_the_frozen_document() -> None:
    """The harness must bind the charter this campaign was written against."""
    assert F3_CHARTER_SHA256 == (
        "0a61ed647afc08424a149a06a6e247535d4da931136bc5d2294874634b9564dc"
    )
    assert F3_CHARTER_COMMIT == "b7ec63b6e"
    # Append-only: the freeze seeds the lineage and is never dropped from it.
    assert F3_CHARTER_LINEAGE[0] == F3_CHARTER_SHA256
    assert len(set(F3_CHARTER_LINEAGE)) == len(F3_CHARTER_LINEAGE)


def test_frozen_rung_budgets_and_thresholds_match_the_charter() -> None:
    assert RUNG_BUDGETS == {"b3": 3, "b37": 37}
    assert PAIR_COUNT == 5
    assert MAX_TIMED_LEGS == 51
    assert MAX_SOLVE_CHILD_PROCESSES == 130
    assert GRADIENT_FACTOR_K == 2.0
    assert ARCHIVED_B3_PROCESS_WALL == CHARTER_ARCHIVED_B3_WALL
    assert ARCHIVED_STEADY_PER_EVAL == CHARTER_ARCHIVED_PER_EVAL


# --------------------------------------------------------------------------
# Policy identity (charter: mismatch voids the leg)
# --------------------------------------------------------------------------


def test_policy_payload_is_the_archived_lbfgsb_policy_at_the_rung() -> None:
    payload = policy_payload(37)

    assert set(payload) == set(POLICY_FIELDS)
    assert payload["method"] == "L-BFGS-B"
    assert payload["maxiter"] == 37
    assert payload["maxcor"] == POLICY_MAXCOR == CHARTER_MAXCOR
    assert payload["maxls"] == POLICY_MAXLS == CHARTER_MAXLS
    assert payload["gtol"] == CHARTER_GTOL
    assert payload["ftol"] == CHARTER_FTOL
    # The charter states this cap cannot bind: 37 x 9 = 333 < 740.
    assert payload["maxfun"] == 37 * MAXFUN_MULTIPLIER
    assert 37 * (POLICY_MAXLS + 1) < 37 * MAXFUN_MULTIPLIER


def test_policy_sha_separates_the_rungs_and_is_stable() -> None:
    b3_sha = policy_identity_sha256(RUNG_BUDGETS["b3"])
    b37_sha = policy_identity_sha256(RUNG_BUDGETS["b37"])

    assert b3_sha != b37_sha
    assert b3_sha == policy_identity_sha256(RUNG_BUDGETS["b3"])
    assert len(b3_sha) == 64
    # A lane that reproduces the chartered policy hashes to the rung constant.
    assert observed_policy_sha256(policy_payload(3)) == b3_sha


def test_policy_sha_ignores_fields_outside_the_chartered_seven() -> None:
    """A row may carry extra keys; identity is the seven the charter names."""
    decorated = dict(policy_payload(3))
    decorated["disclosure"] = "native lane imposes no evaluation cap"

    assert observed_policy_sha256(decorated) == policy_identity_sha256(3)


def test_policy_sha_refuses_a_policy_missing_a_chartered_field() -> None:
    incomplete = dict(policy_payload(3))
    del incomplete["maxls"]

    with pytest.raises(F3ContractError, match="missing"):
        observed_policy_sha256(incomplete)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("maxls", 20, id="unpinned-line-search"),
        pytest.param("maxcor", 10, id="finite-build-history"),
        pytest.param("gtol", 1.0e-12, id="test-tolerance-gtol"),
        pytest.param("ftol", 1.0e-15, id="test-tolerance-ftol"),
        pytest.param("method", "BFGS", id="wrong-method"),
    ],
)
def test_policy_identity_names_every_field_a_lane_got_wrong(
    field: str, value: object
) -> None:
    """The pre-pin defaults are exactly what this gate exists to catch."""
    observed = dict(policy_payload(3))
    observed[field] = value

    failures = policy_identity_failures(observed, budget=3)

    assert len(failures) == 1
    assert failures[0].startswith(f"policy_{field}_")
    assert observed_policy_sha256(observed) != policy_identity_sha256(3)


def test_policy_identity_accepts_the_chartered_policy() -> None:
    assert policy_identity_failures(policy_payload(37), budget=37) == []


# --------------------------------------------------------------------------
# Per-row contract binding
# --------------------------------------------------------------------------


def _contract(**overrides: Any) -> str:
    arguments: dict[str, Any] = {
        "campaign_manifest_sha256": BUNDLE_SHA,
        "budget": 37,
        "production_commit": PRODUCTION_COMMIT,
        "instrument_commit": INSTRUMENT_COMMIT,
        "fair_bar_charter_sha256": FAIR_BAR_SHA,
    }
    arguments.update(overrides)
    return f3_contract_sha256(**arguments)


@pytest.mark.parametrize(
    "override",
    [
        pytest.param({"campaign_manifest_sha256": "f" * 64}, id="bundle"),
        pytest.param({"budget": 3}, id="rung-policy"),
        pytest.param({"production_commit": "1" * 40}, id="production-tree"),
        pytest.param({"instrument_commit": "2" * 40}, id="instrument-tree"),
        pytest.param({"fair_bar_charter_sha256": "3" * 64}, id="fair-bar-charter"),
        pytest.param({"charter_sha256": "4" * 64}, id="f3-charter"),
    ],
)
def test_every_chartered_component_moves_the_row_contract(
    override: dict[str, Any],
) -> None:
    """The charter enumerates six components; each must bind the row."""
    assert _contract(**override) != _contract()


def test_row_contract_is_reproducible() -> None:
    assert _contract() == _contract()
    assert len(_contract()) == 64


# --------------------------------------------------------------------------
# Counter liveness (charter: a silent zero must void, never reach an anchor)
# --------------------------------------------------------------------------


def test_counter_liveness_accepts_two_live_counters() -> None:
    assert counter_liveness_failures(l1_nfev=9, l2_compact_evaluations=9) == []


@pytest.mark.parametrize(
    ("l1", "l2", "expected"),
    [
        pytest.param(0, 9, "l1_nfev_nonpositive_0", id="fused-counter-defaulted"),
        pytest.param(
            9, 0, "l2_compact_candidate_evaluations_nonpositive_0", id="native-zero"
        ),
        pytest.param(-1, 9, "l1_nfev_nonpositive_-1", id="negative"),
        pytest.param(None, 9, "l1_nfev_not_an_integer", id="absent"),
        pytest.param(True, 9, "l1_nfev_not_an_integer", id="boolean-is-not-a-count"),
    ],
)
def test_counter_liveness_voids_a_dead_counter(
    l1: object, l2: object, expected: str
) -> None:
    """dispatch.py defaults nfev to 0; that must void rather than anchor."""
    failures = counter_liveness_failures(l1_nfev=l1, l2_compact_evaluations=l2)

    assert expected in failures


# --------------------------------------------------------------------------
# Anchor formulas (charter "Verdict rule" rule 2 — non-amendable)
# --------------------------------------------------------------------------


def test_b3_anchor_is_the_archived_process_wall() -> None:
    assert b3_anchor() == CHARTER_ARCHIVED_B3_WALL


def test_b37_anchor_prices_the_smaller_of_the_two_lane_medians() -> None:
    """min() is the charter's: never credit the GPU with work it did not do."""
    native_heavier = b37_anchor(
        l2_compact_evaluations=[40, 41, 42], l1_nfev=[30, 31, 32]
    )
    fused_heavier = b37_anchor(
        l2_compact_evaluations=[30, 31, 32], l1_nfev=[40, 41, 42]
    )

    assert native_heavier == pytest.approx(CHARTER_ARCHIVED_PER_EVAL * 31)
    assert fused_heavier == pytest.approx(CHARTER_ARCHIVED_PER_EVAL * 31)
    assert native_heavier == fused_heavier


def test_b37_anchor_uses_medians_not_means() -> None:
    """One slow pair must not move the anchor the way a mean would."""
    assert b37_anchor(
        l2_compact_evaluations=[9, 9, 900], l1_nfev=[9, 9, 900]
    ) == pytest.approx(CHARTER_ARCHIVED_PER_EVAL * 9)


def test_b37_anchor_refuses_a_lane_with_no_counters() -> None:
    with pytest.raises(F3ContractError, match="both lanes"):
        b37_anchor(l2_compact_evaluations=[9, 9], l1_nfev=[])


def test_bq_anchor_prices_native_work_at_n_star() -> None:
    """BQ's currency is native's minimal cost of producing Q*."""
    assert bq_anchor(l2_compact_evaluations_at_nstar=[20, 21, 22]) == pytest.approx(
        CHARTER_ARCHIVED_PER_EVAL * 21
    )


def test_bq_anchor_refuses_an_empty_counter_set() -> None:
    with pytest.raises(F3ContractError, match="at n\\*"):
        bq_anchor(l2_compact_evaluations_at_nstar=[])


# --------------------------------------------------------------------------
# Quality gate (one-sided objective, K=2 gradient)
# --------------------------------------------------------------------------


def test_fixed_budget_gate_accepts_an_equal_or_better_fused_endpoint() -> None:
    assert (
        fixed_budget_quality_failures(
            l1_oracle_objective=1.0,
            l2_oracle_objective=1.0,
            l1_oracle_gradient_inf=2.0,
            l2_oracle_gradient_inf=2.0,
        )
        == []
    )


def test_fixed_budget_gate_is_one_sided_on_the_objective() -> None:
    """A better fused endpoint passes; the campaign never demands identity."""
    assert (
        fixed_budget_quality_failures(
            l1_oracle_objective=0.5,
            l2_oracle_objective=1.0,
            l1_oracle_gradient_inf=1.0,
            l2_oracle_gradient_inf=1.0,
        )
        == []
    )
    assert "l1_objective_above_paired_native" in fixed_budget_quality_failures(
        l1_oracle_objective=1.0 + 1.0e-6,
        l2_oracle_objective=1.0,
        l1_oracle_gradient_inf=1.0,
        l2_oracle_gradient_inf=1.0,
    )


def test_fixed_budget_gate_tolerates_exactly_the_chartered_1e_minus_10() -> None:
    assert (
        fixed_budget_quality_failures(
            l1_oracle_objective=1.0 + 1.0e-11,
            l2_oracle_objective=1.0,
            l1_oracle_gradient_inf=1.0,
            l2_oracle_gradient_inf=1.0,
        )
        == []
    )


def test_fixed_budget_gate_rejects_a_non_descending_endpoint() -> None:
    """Lower objective but a gradient past K x native is the case K exists for."""
    failures = fixed_budget_quality_failures(
        l1_oracle_objective=0.1,
        l2_oracle_objective=1.0,
        l1_oracle_gradient_inf=2.0001,
        l2_oracle_gradient_inf=1.0,
    )

    assert failures == ["l1_gradient_above_k_times_native"]


def test_bq_gate_requires_both_endpoints_at_the_quality_target() -> None:
    assert (
        bq_quality_failures(
            l1_oracle_objective=0.9,
            l2_oracle_objective=0.95,
            l1_oracle_gradient_inf=1.0,
            l2_oracle_gradient_inf=1.0,
            quality_target=1.0,
        )
        == []
    )
    failures = bq_quality_failures(
        l1_oracle_objective=1.5,
        l2_oracle_objective=2.0,
        l1_oracle_gradient_inf=1.0,
        l2_oracle_gradient_inf=1.0,
        quality_target=1.0,
    )
    assert failures == [
        "l1_objective_above_quality_target",
        "l2_objective_above_quality_target",
    ]


# --------------------------------------------------------------------------
# Verdict rule (dual; both must hold)
# --------------------------------------------------------------------------


def test_pair_speedups_are_native_over_fused() -> None:
    assert pair_speedups(l1_walls=[10.0, 5.0], l2_walls=[12.0, 10.0]) == (1.2, 2.0)


def test_pair_speedups_refuse_a_nonpositive_fused_wall() -> None:
    with pytest.raises(F3ContractError, match="must be positive"):
        pair_speedups(l1_walls=[0.0], l2_walls=[1.0])


def test_rung_wins_when_both_rules_hold() -> None:
    outcome = adjudicate_rung(
        l1_walls=[10.0] * PAIR_COUNT,
        l2_walls=[12.0] * PAIR_COUNT,
        anchor_seconds=58.702,
    )

    assert outcome.verdict is Verdict.WIN
    assert outcome.median_speedup == pytest.approx(1.2)
    assert outcome.anchor_over_l1_median == pytest.approx(5.8702)
    assert outcome.live_rule_holds and outcome.anchor_rule_holds


def test_rung_is_bounded_negative_when_one_pair_fails_the_pair_threshold() -> None:
    """Median 1.10 is not enough: every pair must exceed 1.00."""
    outcome = adjudicate_rung(
        l1_walls=[10.0, 10.0, 10.0, 10.0, 10.0],
        l2_walls=[13.0, 13.0, 11.0, 13.0, 9.9],
        anchor_seconds=58.702,
    )

    assert outcome.verdict is Verdict.CLOSED_BOUNDED_NEGATIVE
    assert outcome.minimum_speedup == pytest.approx(0.99)
    assert outcome.live_rule_holds is False


def test_rung_is_bounded_negative_when_only_the_anchor_rule_fails() -> None:
    outcome = adjudicate_rung(
        l1_walls=[10.0] * PAIR_COUNT,
        l2_walls=[12.0] * PAIR_COUNT,
        anchor_seconds=10.5,
    )

    assert outcome.live_rule_holds is True
    assert outcome.anchor_rule_holds is False
    assert outcome.verdict is Verdict.CLOSED_BOUNDED_NEGATIVE


def test_rung_median_exactly_at_the_threshold_still_wins() -> None:
    outcome = adjudicate_rung(
        l1_walls=[10.0] * PAIR_COUNT,
        l2_walls=[11.0] * PAIR_COUNT,
        anchor_seconds=11.0,
    )

    assert outcome.median_speedup == pytest.approx(1.10)
    assert outcome.verdict is Verdict.WIN


def test_rung_is_not_produced_when_three_pairs_voided() -> None:
    outcome = adjudicate_rung(
        l1_walls=[10.0, 10.0],
        l2_walls=[12.0, 12.0],
        anchor_seconds=58.702,
        not_produced_pairs=3,
    )

    assert outcome.verdict is Verdict.NOT_PRODUCED
    assert any("rung_aborted" in failure for failure in outcome.failures)


def test_rung_is_not_produced_when_a_pair_is_missing() -> None:
    outcome = adjudicate_rung(
        l1_walls=[10.0] * 4,
        l2_walls=[12.0] * 4,
        anchor_seconds=58.702,
    )

    assert outcome.verdict is Verdict.NOT_PRODUCED
    assert any("pair_count_4" in failure for failure in outcome.failures)


def test_rung_is_not_produced_without_an_anchor() -> None:
    outcome = adjudicate_rung(
        l1_walls=[10.0] * PAIR_COUNT,
        l2_walls=[12.0] * PAIR_COUNT,
        anchor_seconds=None,
    )

    assert outcome.anchor_rule_holds is False
    assert outcome.verdict is Verdict.CLOSED_BOUNDED_NEGATIVE
    assert "anchor_unavailable" in outcome.failures


# --------------------------------------------------------------------------
# Cap accounting (charter "Caps and aborts")
# --------------------------------------------------------------------------


def test_the_chartered_campaign_shape_fits_inside_its_caps() -> None:
    """The charter's own arithmetic: 51 timed, 121 solve-executing children."""
    ledger = CapLedger().with_legs(timed=51, solve_children=121, wall_seconds=3600.0)

    assert ledger.breaches() == []
    assert ledger.timed_legs == MAX_TIMED_LEGS


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        pytest.param({"timed": 52}, "timed_legs_52_over_51", id="timed-legs"),
        pytest.param(
            {"solve_children": 131},
            "solve_children_131_over_130",
            id="solve-children",
        ),
        pytest.param(
            {"wall_seconds": 12 * 3600.0 + 1.0},
            "campaign_wall_43201s_over_43200s",
            id="campaign-wall",
        ),
    ],
)
def test_each_cap_breach_is_named(kwargs: dict[str, Any], expected: str) -> None:
    assert expected in CapLedger().with_legs(**kwargs).breaches()


def test_budget_search_caps_are_the_chartered_ones() -> None:
    assert (
        budget_search_breaches(
            probe_count=BQ_MAX_PROBES_PER_LANE,
            largest_maxiter=BQ_MAX_MAXITER,
            search_wall_seconds=BQ_MAX_SEARCH_SECONDS,
        )
        == []
    )
    assert budget_search_breaches(
        probe_count=BQ_MAX_PROBES_PER_LANE + 1,
        largest_maxiter=BQ_MAX_MAXITER + 1,
        search_wall_seconds=BQ_MAX_SEARCH_SECONDS + 1.0,
    ) == [
        "probes_13_over_12",
        "maxiter_1025_over_1024",
        "search_wall_7201s_over_7200s",
    ]


# --------------------------------------------------------------------------
# validate (charter "Validate entrypoint": from the run dir alone)
# --------------------------------------------------------------------------


def _row(
    *,
    lane: str,
    wall: float,
    evaluations: int,
    budget: int = 37,
    oracle_objective: float = 1.0,
) -> dict[str, Any]:
    policy = policy_payload(budget)
    return {
        "schema": F3_ROW_SCHEMA,
        "lane": lane,
        "role": "timed",
        "rung": "b37",
        "budget": budget,
        "timed": True,
        "process_wall_seconds": wall,
        "evaluation_count": evaluations,
        "evaluation_counter_name": (
            "nfev" if lane == L1_LANE else "compact_candidate_evaluations"
        ),
        "nit": budget,
        "policy": policy,
        "policy_identity_sha256": observed_policy_sha256(policy),
        "oracle_objective": oracle_objective,
        "f3_charter_sha256": F3_CHARTER_SHA256,
        "fair_bar_charter_sha256": FAIR_BAR_SHA,
        "campaign_input_manifest_sha256": BUNDLE_SHA,
        "production_commit": PRODUCTION_COMMIT,
        "instrument_commit": INSTRUMENT_COMMIT,
        "campaign_contract_sha256": _contract(budget=budget),
    }


def _write_run_dir(
    root: Path,
    *,
    l1_walls: list[float],
    l2_walls: list[float],
    manifest_overrides: dict[str, Any] | None = None,
    row_mutation: Any = None,
) -> Path:
    l1_nfev = [30] * len(l1_walls)
    l2_compact = [31] * len(l2_walls)
    for index, (l1_wall, l2_wall) in enumerate(zip(l1_walls, l2_walls, strict=True)):
        for lane, wall, evaluations in (
            (L1_LANE, l1_wall, l1_nfev[index]),
            (L2_LANE, l2_wall, l2_compact[index]),
        ):
            row = _row(lane=lane, wall=wall, evaluations=evaluations)
            if row_mutation is not None:
                row = row_mutation(row, index, lane)
            leg = root / f"pair{index}-{'l1' if lane == L1_LANE else 'l2'}"
            leg.mkdir(parents=True, exist_ok=True)
            (leg / "row.json").write_text(json.dumps(row, indent=2, sort_keys=True))
    anchor = b37_anchor(l2_compact_evaluations=l2_compact, l1_nfev=l1_nfev)
    outcome = adjudicate_rung(
        l1_walls=l1_walls, l2_walls=l2_walls, anchor_seconds=anchor
    )
    manifest: dict[str, Any] = {
        "schema": F3_RUN_MANIFEST_SCHEMA,
        "rung": "b37",
        "budget": 37,
        "f3_charter_sha256": F3_CHARTER_SHA256,
        "anchor_process_wall_seconds": anchor,
        "verdict": outcome.verdict.value,
        "not_produced_pairs": 0,
        "gate_failures": [],
        "timed_legs": 2 * len(l1_walls),
        "solve_child_processes": 4 * len(l1_walls),
        "campaign_wall_seconds": 100.0,
    }
    manifest.update(manifest_overrides or {})
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return root


def test_validate_accepts_a_consistent_run_directory(tmp_path: Path) -> None:
    """The control every rejection below is measured against."""
    _write_run_dir(tmp_path, l1_walls=[10.0] * PAIR_COUNT, l2_walls=[12.0] * PAIR_COUNT)

    report = validate_run_dir(tmp_path)

    assert report.findings == ()
    assert report.valid is True
    assert report.timed_pair_count == PAIR_COUNT
    assert report.recomputed_verdict is Verdict.WIN
    assert report.median_speedup == pytest.approx(1.2)


def test_validate_recomputes_the_verdict_rather_than_trusting_it(
    tmp_path: Path,
) -> None:
    """A run dir claiming WIN over losing walls must not validate."""
    _write_run_dir(
        tmp_path,
        l1_walls=[12.0] * PAIR_COUNT,
        l2_walls=[10.0] * PAIR_COUNT,
        manifest_overrides={"verdict": Verdict.WIN.value},
    )

    report = validate_run_dir(tmp_path)

    assert report.valid is False
    assert report.recomputed_verdict is Verdict.CLOSED_BOUNDED_NEGATIVE
    assert any("recorded verdict" in finding for finding in report.findings)


def test_validate_rejects_a_row_bound_to_a_foreign_contract(
    tmp_path: Path,
) -> None:
    def _forge(row: dict[str, Any], index: int, lane: str) -> dict[str, Any]:
        if index == 0 and lane == L1_LANE:
            row["campaign_contract_sha256"] = "f" * 64
        return row

    _write_run_dir(
        tmp_path,
        l1_walls=[10.0] * PAIR_COUNT,
        l2_walls=[12.0] * PAIR_COUNT,
        row_mutation=_forge,
    )

    report = validate_run_dir(tmp_path)

    assert report.valid is False
    assert any("foreign contract" in finding for finding in report.findings)


def test_validate_rejects_a_row_whose_policy_left_the_rung(tmp_path: Path) -> None:
    """The pre-pin maxls=20 default is what this check exists to catch."""

    def _unpin(row: dict[str, Any], index: int, lane: str) -> dict[str, Any]:
        if index == 0 and lane == L1_LANE:
            row["policy"] = {**row["policy"], "maxls": 20}
            row["policy_identity_sha256"] = observed_policy_sha256(row["policy"])
        return row

    _write_run_dir(
        tmp_path,
        l1_walls=[10.0] * PAIR_COUNT,
        l2_walls=[12.0] * PAIR_COUNT,
        row_mutation=_unpin,
    )

    report = validate_run_dir(tmp_path)

    assert report.valid is False
    assert any(
        "differs from the rung constant" in finding for finding in report.findings
    )


def test_validate_rejects_a_recorded_policy_sha_that_does_not_hash_its_policy(
    tmp_path: Path,
) -> None:
    def _forge(row: dict[str, Any], index: int, lane: str) -> dict[str, Any]:
        if index == 0 and lane == L2_LANE:
            row["policy_identity_sha256"] = "0" * 64
        return row

    _write_run_dir(
        tmp_path,
        l1_walls=[10.0] * PAIR_COUNT,
        l2_walls=[12.0] * PAIR_COUNT,
        row_mutation=_forge,
    )

    report = validate_run_dir(tmp_path)

    assert report.valid is False
    assert any("recorded policy sha differs" in finding for finding in report.findings)


def test_validate_rejects_an_anchor_off_the_charter_formula(tmp_path: Path) -> None:
    _write_run_dir(
        tmp_path,
        l1_walls=[10.0] * PAIR_COUNT,
        l2_walls=[12.0] * PAIR_COUNT,
        manifest_overrides={"anchor_process_wall_seconds": 999.0},
    )

    report = validate_run_dir(tmp_path)

    assert report.valid is False
    assert any(
        "differs from the charter formula" in finding for finding in report.findings
    )


def test_validate_reports_a_dead_counter_in_a_timed_row(tmp_path: Path) -> None:
    def _kill(row: dict[str, Any], index: int, lane: str) -> dict[str, Any]:
        if index == 2 and lane == L1_LANE:
            row["evaluation_count"] = 0
        return row

    _write_run_dir(
        tmp_path,
        l1_walls=[10.0] * PAIR_COUNT,
        l2_walls=[12.0] * PAIR_COUNT,
        row_mutation=_kill,
    )

    report = validate_run_dir(tmp_path)

    assert report.valid is False
    assert any("l1_nfev_nonpositive_0" in finding for finding in report.findings)


def test_validate_refuses_a_foreign_charter_lineage(tmp_path: Path) -> None:
    _write_run_dir(
        tmp_path,
        l1_walls=[10.0] * PAIR_COUNT,
        l2_walls=[12.0] * PAIR_COUNT,
        manifest_overrides={"f3_charter_sha256": "e" * 64},
    )

    with pytest.raises(F3ContractError, match="append-only lineage"):
        validate_run_dir(tmp_path)


def test_validate_refuses_a_foreign_manifest_schema(tmp_path: Path) -> None:
    _write_run_dir(
        tmp_path,
        l1_walls=[10.0] * PAIR_COUNT,
        l2_walls=[12.0] * PAIR_COUNT,
        manifest_overrides={"schema": "some-other-campaign.v1"},
    )

    with pytest.raises(F3ContractError, match="run manifest schema"):
        validate_run_dir(tmp_path)


def test_validate_reports_a_cap_breach_recorded_in_the_manifest(
    tmp_path: Path,
) -> None:
    _write_run_dir(
        tmp_path,
        l1_walls=[10.0] * PAIR_COUNT,
        l2_walls=[12.0] * PAIR_COUNT,
        manifest_overrides={"solve_child_processes": 131},
    )

    report = validate_run_dir(tmp_path)

    assert report.valid is False
    assert any("solve_children_131" in finding for finding in report.findings)


def test_validate_refuses_a_directory_with_no_rows(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": F3_RUN_MANIFEST_SCHEMA,
                "rung": "b37",
                "budget": 37,
                "f3_charter_sha256": F3_CHARTER_SHA256,
            }
        )
    )

    with pytest.raises(F3ContractError, match="holds no rows"):
        validate_run_dir(tmp_path)
