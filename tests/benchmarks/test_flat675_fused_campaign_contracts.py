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
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from benchmarks.flat675_fused_campaign_contract import (
    ARCHIVED_B3_PROCESS_WALL,
    ARCHIVED_STEADY_PER_EVAL,
    BQ_MAX_MAXITER,
    BQ_MAX_PROBES_PER_LANE,
    BQ_MAX_SEARCH_SECONDS,
    BQ_SEARCH_START,
    DISCLOSURE_RUNG_SUFFIX,
    DISCLOSURE_RUNGS,
    F3_CHARTER_AMENDMENT_1_SHA256,
    F3_CHARTER_COMMIT,
    F3_CHARTER_FREEZE_SHA256,
    F3_CHARTER_LINEAGE,
    F3_CHARTER_SHA256,
    F3_ROW_DIRECTORY,
    F3_ROW_SCHEMA,
    F3_RUN_MANIFEST_SCHEMA,
    GRADIENT_FACTOR_K,
    L1_LANE,
    L2_LANE,
    MAX_SOLVE_CHILD_PROCESSES,
    MAX_TIMED_LEGS,
    MAXFUN_MULTIPLIER,
    OBSOLETE_PRECISION_ENV,
    PAIR_COUNT,
    POLICY_FIELDS,
    POLICY_MAXCOR,
    POLICY_MAXLS,
    RUNG_BUDGETS,
    SOLVE_CHILDREN_PER_COLD_PAIR,
    SOLVE_CHILDREN_PER_WARM_PAIR,
    CampaignState,
    CapLedger,
    F3ContractError,
    Verdict,
    adjudicate_rows,
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
    pair_rows,
    pair_speedups,
    parse_row,
    policy_identity_failures,
    policy_identity_sha256,
    policy_payload,
    production_child_environment,
    resolve_disclosure_budgets,
    row_paths,
    search_minimal_budget,
    select_sweep_config,
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


def test_charter_identity_is_the_current_amendment() -> None:
    """The harness must bind the charter bytes it is executing under."""
    assert F3_CHARTER_SHA256 == (
        "86db1058962d25048f3f16a5e616978985aaa86e08028d9b1c2dbe868ddfb994"
    )
    assert F3_CHARTER_COMMIT == "e8625f691"


def test_charter_lineage_is_append_only() -> None:
    """Every earlier sha survives, in order, with the current one last."""
    assert F3_CHARTER_FREEZE_SHA256 == (
        "0a61ed647afc08424a149a06a6e247535d4da931136bc5d2294874634b9564dc"
    )
    assert F3_CHARTER_AMENDMENT_1_SHA256 == (
        "b710ff423667b7fa3c2d9e194ee1e3ccca94ed4821df7c9081fb4deb76e298d2"
    )
    assert F3_CHARTER_LINEAGE == (
        F3_CHARTER_FREEZE_SHA256,
        F3_CHARTER_AMENDMENT_1_SHA256,
        F3_CHARTER_SHA256,
    )
    assert len(set(F3_CHARTER_LINEAGE)) == len(F3_CHARTER_LINEAGE)


@pytest.mark.parametrize(
    "charter_sha",
    [
        pytest.param(F3_CHARTER_FREEZE_SHA256, id="freeze"),
        pytest.param(F3_CHARTER_AMENDMENT_1_SHA256, id="amendment-1"),
    ],
)
def test_validate_accepts_a_run_bound_to_any_earlier_lineage_member(
    tmp_path: Path, charter_sha: str
) -> None:
    """A run that executed under an earlier amendment stays validatable."""
    _write_run_dir(
        tmp_path,
        l1_walls=[10.0] * PAIR_COUNT,
        l2_walls=[12.0] * PAIR_COUNT,
        manifest_overrides={"f3_charter_sha256": charter_sha},
        charter_sha=charter_sha,
    )

    assert validate_run_dir(tmp_path).valid is True


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


def test_bq_gate_compares_to_the_target_exactly() -> None:
    """The charter grants BQ no tolerance; the 1e-10 is the fixed-budget gate's."""
    assert (
        bq_quality_failures(
            l1_oracle_objective=1.0,
            l2_oracle_objective=1.0,
            l1_oracle_gradient_inf=1.0,
            l2_oracle_gradient_inf=1.0,
            quality_target=1.0,
        )
        == []
    )
    assert bq_quality_failures(
        l1_oracle_objective=1.0 + 1.0e-11,
        l2_oracle_objective=1.0,
        l1_oracle_gradient_inf=1.0,
        l2_oracle_gradient_inf=1.0,
        quality_target=1.0,
    ) == ["l1_objective_above_quality_target"]


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
    pair_index: int,
    wall: float,
    evaluations: int,
    budget: int = 37,
    oracle_objective: float = 1.0,
    oracle_gradient: float = 1.0,
    charter_sha: str = F3_CHARTER_SHA256,
) -> dict[str, Any]:
    policy = policy_payload(budget)
    return {
        "schema": F3_ROW_SCHEMA,
        "lane": lane,
        "role": "timed",
        "rung": "b37",
        "pair_index": pair_index,
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
        "oracle_gradient_inf_norm": oracle_gradient,
        "host_transfer_ledger": {"initialization": 0, "final_result": 15},
        "f3_charter_sha256": charter_sha,
        "fair_bar_charter_sha256": FAIR_BAR_SHA,
        "campaign_input_manifest_sha256": BUNDLE_SHA,
        "production_commit": PRODUCTION_COMMIT,
        "instrument_commit": INSTRUMENT_COMMIT,
        "campaign_contract_sha256": _contract(
            budget=budget, charter_sha256=charter_sha
        ),
    }


def _write_run_dir(
    root: Path,
    *,
    l1_walls: list[float],
    l2_walls: list[float],
    manifest_overrides: dict[str, Any] | None = None,
    row_mutation: Any = None,
    charter_sha: str = F3_CHARTER_SHA256,
) -> Path:
    l1_nfev = [30] * len(l1_walls)
    l2_compact = [31] * len(l2_walls)
    rows: list[dict[str, Any]] = []
    for index, (l1_wall, l2_wall) in enumerate(zip(l1_walls, l2_walls, strict=True)):
        for lane, wall, evaluations in (
            (L1_LANE, l1_wall, l1_nfev[index]),
            (L2_LANE, l2_wall, l2_compact[index]),
        ):
            row = _row(
                lane=lane,
                pair_index=index,
                wall=wall,
                evaluations=evaluations,
                charter_sha=charter_sha,
            )
            if row_mutation is not None:
                row = row_mutation(row, index, lane)
            rows.append(row)
    _write_rows(root, rows)
    anchor = b37_anchor(l2_compact_evaluations=l2_compact, l1_nfev=l1_nfev)
    outcome = adjudicate_rung(
        l1_walls=l1_walls, l2_walls=l2_walls, anchor_seconds=anchor
    )
    manifest: dict[str, Any] = {
        "schema": F3_RUN_MANIFEST_SCHEMA,
        "rung": "b37",
        "budget": 37,
        "f3_charter_sha256": charter_sha,
        "anchor_process_wall_seconds": anchor,
        "verdict": outcome.verdict.value,
        "gate_failures": [],
        "timed_legs": 2 * len(l1_walls),
        "solve_child_processes": SOLVE_CHILDREN_PER_WARM_PAIR * len(l1_walls),
        "campaign_wall_seconds": 100.0,
        "production_commit": PRODUCTION_COMMIT,
        "instrument_commit": INSTRUMENT_COMMIT,
        "campaign_input_manifest_sha256": BUNDLE_SHA,
    }
    manifest.update(manifest_overrides or {})
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return root


def _write_rows(root: Path, rows: list[dict[str, Any]]) -> None:
    """Place rows exactly where the producer places them."""
    row_dir = root / F3_ROW_DIRECTORY
    row_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        name = f"pair{row['pair_index']}-{'l1' if row['lane'] == L1_LANE else 'l2'}"
        (row_dir / f"{name}.json").write_text(json.dumps(row, indent=2, sort_keys=True))


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
        "differs from the constant for its budget" in finding
        for finding in report.findings
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


# --------------------------------------------------------------------------
# Producer/validator round trip: the layout the orchestrator actually writes
# --------------------------------------------------------------------------


def _producer_row(
    *,
    lane: str,
    pair_index: int,
    rung: str,
    budget: int,
    wall: float,
    evaluations: int,
    oracle_objective: float,
    oracle_gradient: float = 1.0,
) -> dict[str, Any]:
    """A row shaped exactly as ``_write_row`` shapes it, including its key set."""
    policy = policy_payload(budget)
    return {
        "schema": F3_ROW_SCHEMA,
        "lane": lane,
        "role": "timed",
        "rung": rung,
        "pair_index": pair_index,
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
        "oracle_gradient_inf_norm": oracle_gradient,
        "host_transfer_ledger": {"initialization": 0, "final_result": 15},
        "f3_charter_sha256": F3_CHARTER_SHA256,
        "fair_bar_charter_sha256": FAIR_BAR_SHA,
        "campaign_input_manifest_sha256": BUNDLE_SHA,
        "production_commit": PRODUCTION_COMMIT,
        "instrument_commit": INSTRUMENT_COMMIT,
        "campaign_contract_sha256": _contract(budget=budget),
    }


def _publish_like_the_producer(
    root: Path,
    rows: list[dict[str, Any]],
    *,
    rung: str,
    budget: int,
    quality_target: float | None = None,
) -> Any:
    """Write rows and publish the manifest through the shared rule.

    This mirrors what ``_publish_rung`` does — parse the rows just written,
    adjudicate them, record the outcome — so a round trip here exercises the
    same agreement the orchestrator relies on.
    """
    _write_rows(root, rows)
    parsed = [
        parse_row(row, source=f"{row['lane']}-pair{row['pair_index']}") for row in rows
    ]
    outcome, pairs = adjudicate_rows(parsed, rung=rung, quality_target=quality_target)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": F3_RUN_MANIFEST_SCHEMA,
                "rung": rung,
                "budget": budget,
                "quality_target": quality_target,
                "f3_charter_sha256": F3_CHARTER_SHA256,
                "anchor_process_wall_seconds": outcome.anchor_seconds,
                "verdict": outcome.verdict.value,
                "gate_failures": list(outcome.failures),
                "not_produced_pairs": outcome.not_produced_pairs,
                "timed_legs": 2 * len(pairs),
                "solve_child_processes": SOLVE_CHILDREN_PER_WARM_PAIR * len(pairs),
                "campaign_wall_seconds": 100.0,
                "production_commit": PRODUCTION_COMMIT,
                "instrument_commit": INSTRUMENT_COMMIT,
                "campaign_input_manifest_sha256": BUNDLE_SHA,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return outcome


def test_row_paths_ignore_a_fair_bar_row_beside_a_leg(tmp_path: Path) -> None:
    """The bug this layout fixes: run_leg writes its own row.json per leg."""
    _write_rows(
        tmp_path,
        [
            _producer_row(
                lane=lane,
                pair_index=0,
                rung="b3",
                budget=3,
                wall=10.0,
                evaluations=9,
                oracle_objective=1.0,
            )
            for lane in (L1_LANE, L2_LANE)
        ],
    )
    foreign = tmp_path / "pair0-l2"
    foreign.mkdir()
    (foreign / "row.json").write_text(
        json.dumps({"schema": "genuine-675-fair-bar-row.v1", "lane": "native_cpp_cpu"})
    )

    discovered = row_paths(tmp_path)

    assert [path.name for path in discovered] == ["pair0-l1.json", "pair0-l2.json"]


@pytest.mark.parametrize(
    ("rung", "l1_budget", "l2_budget", "quality_target"),
    [
        pytest.param("b3", 3, 3, None, id="b3"),
        pytest.param("b37", 37, 37, None, id="b37"),
        pytest.param("bq", 22, 31, 1.5, id="bq-m-star-differs-from-n-star"),
    ],
)
def test_producer_layout_round_trips_through_validate(
    tmp_path: Path,
    rung: str,
    l1_budget: int,
    l2_budget: int,
    quality_target: float | None,
) -> None:
    """A directory the producer wrote must validate, with the same verdict.

    BQ is included with m* != n* because that is the case a validator keyed on
    one campaign-wide budget silently mis-hashes.
    """
    rows: list[dict[str, Any]] = []
    for index in range(PAIR_COUNT):
        rows.append(
            _producer_row(
                lane=L1_LANE,
                pair_index=index,
                rung=rung,
                budget=l1_budget,
                wall=10.0,
                evaluations=30,
                oracle_objective=1.0,
            )
        )
        rows.append(
            _producer_row(
                lane=L2_LANE,
                pair_index=index,
                rung=rung,
                budget=l2_budget,
                wall=12.0,
                evaluations=31,
                oracle_objective=1.0,
            )
        )
    outcome = _publish_like_the_producer(
        tmp_path, rows, rung=rung, budget=l1_budget, quality_target=quality_target
    )

    report = validate_run_dir(tmp_path)

    assert report.findings == ()
    assert report.valid is True
    assert report.timed_pair_count == PAIR_COUNT
    assert report.recomputed_verdict is outcome.verdict
    assert report.recorded_verdict == outcome.verdict.value
    assert report.recomputed_anchor_seconds == outcome.anchor_seconds


def test_bq_round_trip_hashes_each_lane_at_its_own_budget(tmp_path: Path) -> None:
    """m* and n* differ, so the two lanes' policy shas must differ too."""
    l1 = _producer_row(
        lane=L1_LANE,
        pair_index=0,
        rung="bq",
        budget=22,
        wall=10.0,
        evaluations=30,
        oracle_objective=1.0,
    )
    l2 = _producer_row(
        lane=L2_LANE,
        pair_index=0,
        rung="bq",
        budget=31,
        wall=12.0,
        evaluations=31,
        oracle_objective=1.0,
    )

    assert l1["policy_identity_sha256"] != l2["policy_identity_sha256"]
    assert l1["policy_identity_sha256"] == policy_identity_sha256(22)
    assert l2["policy_identity_sha256"] == policy_identity_sha256(31)
    assert l1["campaign_contract_sha256"] != l2["campaign_contract_sha256"]
    _write_rows(tmp_path, [l1, l2])
    assert len(row_paths(tmp_path)) == 2


# --------------------------------------------------------------------------
# Forgery battery: a manifest never overrides what the rows say
# --------------------------------------------------------------------------


def _forged_run_dir(tmp_path: Path, mutate: Any, **manifest: Any) -> Path:
    rows: list[dict[str, Any]] = []
    for index in range(PAIR_COUNT):
        for lane, wall, evaluations in (
            (L1_LANE, 10.0, 30),
            (L2_LANE, 12.0, 31),
        ):
            row = _producer_row(
                lane=lane,
                pair_index=index,
                rung="b37",
                budget=37,
                wall=wall,
                evaluations=evaluations,
                oracle_objective=1.0,
            )
            rows.append(mutate(row, index, lane))
    _publish_like_the_producer(tmp_path, rows, rung="b37", budget=37)
    if manifest:
        payload = json.loads((tmp_path / "manifest.json").read_text())
        payload.update(manifest)
        (tmp_path / "manifest.json").write_text(json.dumps(payload, sort_keys=True))
    return tmp_path


def test_forged_manifest_cannot_hide_a_quality_gate_violation(
    tmp_path: Path,
) -> None:
    """gate_failures:[] with a violating row must still fail validation."""

    def _violate(row: dict[str, Any], index: int, lane: str) -> dict[str, Any]:
        if index == 1 and lane == L1_LANE:
            row["oracle_objective"] = 5.0
        return row

    _forged_run_dir(
        tmp_path,
        _violate,
        gate_failures=[],
        verdict=Verdict.WIN.value,
    )

    report = validate_run_dir(tmp_path)

    assert report.valid is False
    assert any(
        "l1_objective_above_paired_native" in finding for finding in report.findings
    )


def test_forged_manifest_cannot_hide_a_gradient_violation(tmp_path: Path) -> None:
    def _violate(row: dict[str, Any], index: int, lane: str) -> dict[str, Any]:
        if index == 0 and lane == L1_LANE:
            row["oracle_gradient_inf_norm"] = 2.5
        return row

    _forged_run_dir(tmp_path, _violate, gate_failures=[], verdict=Verdict.WIN.value)

    report = validate_run_dir(tmp_path)

    assert report.valid is False
    assert any(
        "l1_gradient_above_k_times_native" in finding for finding in report.findings
    )


def test_forged_manifest_cannot_hide_a_dead_counter(tmp_path: Path) -> None:
    def _kill(row: dict[str, Any], index: int, lane: str) -> dict[str, Any]:
        if index == 2 and lane == L2_LANE:
            row["evaluation_count"] = 0
        return row

    _forged_run_dir(tmp_path, _kill, gate_failures=[], verdict=Verdict.WIN.value)

    report = validate_run_dir(tmp_path)

    assert report.valid is False
    assert any(
        "l2_compact_candidate_evaluations_nonpositive_0" in finding
        for finding in report.findings
    )


def test_forged_manifest_cannot_hide_a_native_short_of_its_budget(
    tmp_path: Path,
) -> None:
    """Work matching is fail-closed on BOTH lanes, not just the fused one."""

    def _short(row: dict[str, Any], index: int, lane: str) -> dict[str, Any]:
        if index == 0 and lane == L2_LANE:
            row["nit"] = 36
        return row

    _forged_run_dir(tmp_path, _short, gate_failures=[], verdict=Verdict.WIN.value)

    report = validate_run_dir(tmp_path)

    assert report.valid is False
    assert any("l2_nit_36_expected_37" in finding for finding in report.findings)


def test_forged_manifest_cannot_hide_a_fused_host_transfer(tmp_path: Path) -> None:
    def _leak(row: dict[str, Any], index: int, lane: str) -> dict[str, Any]:
        if index == 3 and lane == L1_LANE:
            row["host_transfer_ledger"] = {"advance": 37, "final_result": 15}
        return row

    _forged_run_dir(tmp_path, _leak, gate_failures=[], verdict=Verdict.WIN.value)

    report = validate_run_dir(tmp_path)

    assert report.valid is False
    assert any("l1_host_advance_transfers" in finding for finding in report.findings)


def test_forged_row_identity_must_match_the_manifest(tmp_path: Path) -> None:
    def _swap(row: dict[str, Any], index: int, lane: str) -> dict[str, Any]:
        if index == 0 and lane == L1_LANE:
            row["production_commit"] = "9" * 40
            row["campaign_contract_sha256"] = _contract(
                budget=37, production_commit="9" * 40
            )
        return row

    _forged_run_dir(tmp_path, _swap)

    report = validate_run_dir(tmp_path)

    assert report.valid is False
    assert any(
        "production_commit" in finding and "differs from the manifest" in finding
        for finding in report.findings
    )


def test_unpaired_timed_row_is_a_contract_error(tmp_path: Path) -> None:
    _write_rows(
        tmp_path,
        [
            _producer_row(
                lane=L1_LANE,
                pair_index=0,
                rung="b3",
                budget=3,
                wall=10.0,
                evaluations=9,
                oracle_objective=1.0,
            )
        ],
    )

    with pytest.raises(F3ContractError, match="missing its"):
        pair_rows(
            [
                parse_row(
                    json.loads(
                        (tmp_path / F3_ROW_DIRECTORY / "pair0-l1.json").read_text()
                    ),
                    source="pair0-l1",
                )
            ]
        )


# --------------------------------------------------------------------------
# BQ budget search, as a pure algorithm against a stubbed probe
# --------------------------------------------------------------------------


def _recording_probe(threshold: int) -> Any:
    """A probe that reaches the target exactly at ``threshold`` iterations."""
    seen: list[int] = []

    def probe(maxiter: int) -> float:
        seen.append(maxiter)
        return 0.5 if maxiter >= threshold else 5.0

    probe.seen = seen  # type: ignore[attr-defined]
    return probe


def test_budget_search_bisects_down_when_the_start_already_reaches() -> None:
    probe = _recording_probe(20)

    search = search_minimal_budget(probe, quality_target=1.0)

    assert search.star == 20
    assert search.breaches == ()
    assert probe.seen[0] == 37  # the chartered start
    assert all(objective is not None for objective in probe.seen)


def test_budget_search_doubles_upward_before_bisecting() -> None:
    probe = _recording_probe(100)

    search = search_minimal_budget(probe, quality_target=1.0)

    assert probe.seen[:3] == [37, 74, 148]
    assert search.star == 100
    assert search.breaches == ()


def test_budget_search_reports_the_smallest_reaching_probe_it_saw() -> None:
    """star is the minimum over reaching probes, not simply the last one.

    Bisection can end on a probe that did NOT reach (the lower bound), so the
    invariant is over the recorded history rather than its final entry.
    """
    search = search_minimal_budget(_recording_probe(20), quality_target=1.0)

    reaching = [
        int(str(probe["maxiter"])) for probe in search.probes if probe["reached_target"]
    ]
    assert search.probes[0]["maxiter"] == BQ_SEARCH_START
    assert all("oracle_objective" in probe for probe in search.probes)
    assert reaching
    assert search.star == min(reaching) == 20


def test_budget_search_stops_at_the_probe_cap() -> None:
    """A long bisection must end at the probe cap rather than run on."""
    search = search_minimal_budget(_recording_probe(1000), quality_target=1.0, start=1)

    assert search.star is None
    assert any("probes_13_over_12" in breach for breach in search.breaches)
    assert len(search.probes) == BQ_MAX_PROBES_PER_LANE


def test_budget_search_that_never_reaches_breaches_the_maxiter_cap() -> None:
    """Doubling from 37 passes 1024 before it passes twelve probes."""
    search = search_minimal_budget(lambda maxiter: 5.0, quality_target=1.0)

    assert search.star is None
    assert any("over_1024" in breach for breach in search.breaches)


def test_budget_search_stops_at_the_maxiter_cap() -> None:
    search = search_minimal_budget(lambda maxiter: 5.0, quality_target=1.0, start=1024)

    assert search.star is None
    assert any("maxiter_2048_over_1024" in breach for breach in search.breaches)


def test_budget_search_stops_at_the_wall_cap() -> None:
    search = search_minimal_budget(
        _recording_probe(20),
        quality_target=1.0,
        elapsed_seconds=lambda: BQ_MAX_SEARCH_SECONDS + 1.0,
    )

    assert search.star is None
    assert any("search_wall_7201s" in breach for breach in search.breaches)
    assert search.probes == ()


# --------------------------------------------------------------------------
# Native sweep selection
# --------------------------------------------------------------------------


def test_sweep_selects_the_fastest_median_config() -> None:
    assert (
        select_sweep_config(
            {"omp1": 210.0, "omp2": 120.0, "omp4": 70.0, "omp8": 55.0, "omp16": 52.7}
        )
        == "omp16"
    )


def test_sweep_selection_is_deterministic_under_a_tie() -> None:
    """Two configs at the same median must not select by dict order."""
    assert select_sweep_config({"omp8": 52.7, "omp16": 52.7}) == "omp16"
    assert select_sweep_config({"omp16": 52.7, "omp8": 52.7}) == "omp16"


def test_sweep_refuses_an_empty_matrix() -> None:
    with pytest.raises(F3ContractError, match="at least one config"):
        select_sweep_config({})


# --------------------------------------------------------------------------
# Campaign state: cap accumulation and the rung-boundary stop
# --------------------------------------------------------------------------


def test_campaign_state_accumulates_across_three_rungs() -> None:
    state = CampaignState()
    for rung in ("b3", "b37", "bq"):
        assert (
            state.admits(
                timed=2 * PAIR_COUNT,
                solve_children=SOLVE_CHILDREN_PER_WARM_PAIR * PAIR_COUNT,
            )
            == []
        )
        state = state.completing(
            rung,
            timed=2 * PAIR_COUNT,
            solve_children=SOLVE_CHILDREN_PER_WARM_PAIR * PAIR_COUNT,
        )

    assert state.completed_rungs == ("b3", "b37", "bq")
    assert state.ledger.timed_legs == 30
    assert state.ledger.solve_child_processes == 90
    assert state.stopped is False


def test_campaign_state_counts_oracle_children_in_a_warm_pair() -> None:
    """Six per warm pair: two primers, two timed legs, two oracle children."""
    assert SOLVE_CHILDREN_PER_WARM_PAIR == 6
    assert SOLVE_CHILDREN_PER_COLD_PAIR == 4


def test_campaign_state_refuses_to_start_a_rung_that_would_breach() -> None:
    """The stop lands at a rung boundary, before the rung begins."""
    state = CampaignState(CapLedger(timed_legs=MAX_TIMED_LEGS - 2))

    assert state.admits(timed=2, solve_children=6) == []
    assert any(
        "timed_legs" in breach for breach in state.admits(timed=4, solve_children=6)
    )


def test_campaign_state_stops_after_a_rung_that_exhausted_a_cap() -> None:
    state = CampaignState(
        CapLedger(solve_child_processes=MAX_SOLVE_CHILD_PROCESSES - 1)
    ).completing("b37", timed=2, solve_children=6)

    assert state.stopped is True
    assert state.completed_rungs == ("b37",)
    # A completed rung keeps its verdict; the NEXT rung is refused.
    assert state.admits(timed=2, solve_children=6) == [
        "campaign_already_stopped_at_a_rung_boundary"
    ]


def test_campaign_state_round_trips_through_its_payload() -> None:
    state = CampaignState(
        CapLedger(timed_legs=10, solve_child_processes=30, campaign_wall_seconds=99.5),
        completed_rungs=("b3",),
    )

    assert CampaignState.from_payload(state.as_payload()) == state


def test_campaign_state_refuses_a_foreign_schema() -> None:
    with pytest.raises(F3ContractError, match="campaign state schema"):
        CampaignState.from_payload({"schema": "something-else.v1"})


# --------------------------------------------------------------------------
# not_produced_pairs counts PAIRS, not the rung-wide failure list
# --------------------------------------------------------------------------


def _rows_with_one_failing_pair() -> list[Any]:
    """Five pairs, of which pair 2 alone violates the quality gate."""
    rows: list[dict[str, Any]] = []
    for index in range(PAIR_COUNT):
        rows.append(
            _producer_row(
                lane=L1_LANE,
                pair_index=index,
                rung="b37",
                budget=37,
                wall=10.0,
                evaluations=30,
                oracle_objective=5.0 if index == 2 else 1.0,
            )
        )
        rows.append(
            _producer_row(
                lane=L2_LANE,
                pair_index=index,
                rung="b37",
                budget=37,
                wall=12.0,
                evaluations=31,
                oracle_objective=1.0,
            )
        )
    return [
        parse_row(row, source=f"{row['lane']}-pair{row['pair_index']}") for row in rows
    ]


def test_not_produced_pairs_counts_only_the_pairs_that_failed() -> None:
    """One failing pair of five is one voided pair, not five."""
    outcome, pairs = adjudicate_rows(_rows_with_one_failing_pair(), rung="b37")

    assert len(pairs) == PAIR_COUNT
    assert outcome.not_produced_pairs == 1
    # Below the three-pair abort, so the rung is adjudicated, not aborted.
    assert not any("rung_aborted" in failure for failure in outcome.failures)


def test_not_produced_pairs_is_zero_when_every_pair_passes() -> None:
    rows = [
        parse_row(row, source=f"{row['lane']}-pair{row['pair_index']}")
        for index in range(PAIR_COUNT)
        for row in (
            _producer_row(
                lane=L1_LANE,
                pair_index=index,
                rung="b37",
                budget=37,
                wall=10.0,
                evaluations=30,
                oracle_objective=1.0,
            ),
            _producer_row(
                lane=L2_LANE,
                pair_index=index,
                rung="b37",
                budget=37,
                wall=12.0,
                evaluations=31,
                oracle_objective=1.0,
            ),
        )
    ]

    outcome, _pairs = adjudicate_rows(rows, rung="b37")

    assert outcome.not_produced_pairs == 0
    assert outcome.verdict is Verdict.WIN


def test_a_published_not_produced_count_must_match_the_recomputed_one(
    tmp_path: Path,
) -> None:
    """The degenerate count published five voided pairs where one had failed."""
    rows: list[dict[str, Any]] = []
    for index in range(PAIR_COUNT):
        rows.append(
            _producer_row(
                lane=L1_LANE,
                pair_index=index,
                rung="b37",
                budget=37,
                wall=10.0,
                evaluations=30,
                oracle_objective=5.0 if index == 2 else 1.0,
            )
        )
        rows.append(
            _producer_row(
                lane=L2_LANE,
                pair_index=index,
                rung="b37",
                budget=37,
                wall=12.0,
                evaluations=31,
                oracle_objective=1.0,
            )
        )
    outcome = _publish_like_the_producer(tmp_path, rows, rung="b37", budget=37)
    manifest = json.loads((tmp_path / "manifest.json").read_text())

    assert outcome.not_produced_pairs == 1
    assert manifest["not_produced_pairs"] == 1
    assert manifest["not_produced_pairs"] != PAIR_COUNT


def test_three_voided_pairs_abort_the_rung() -> None:
    """The abort rule reads the pair count, so it now fires when it should."""
    rows: list[dict[str, Any]] = []
    for index in range(PAIR_COUNT):
        rows.append(
            _producer_row(
                lane=L1_LANE,
                pair_index=index,
                rung="b37",
                budget=37,
                wall=10.0,
                evaluations=30,
                oracle_objective=5.0 if index < 3 else 1.0,
            )
        )
        rows.append(
            _producer_row(
                lane=L2_LANE,
                pair_index=index,
                rung="b37",
                budget=37,
                wall=12.0,
                evaluations=31,
                oracle_objective=1.0,
            )
        )
    parsed = [
        parse_row(row, source=f"{row['lane']}-pair{row['pair_index']}") for row in rows
    ]

    outcome, _pairs = adjudicate_rows(parsed, rung="b37")

    assert outcome.not_produced_pairs == 3
    assert outcome.verdict is Verdict.NOT_PRODUCED
    assert any(
        "rung_aborted_on_3_not_produced_pairs" in failure
        for failure in outcome.failures
    )


# --------------------------------------------------------------------------
# Required manifest identity (absence is a finding, not a pass)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        pytest.param("production_commit", id="production-commit"),
        pytest.param("instrument_commit", id="instrument-commit"),
        pytest.param("campaign_input_manifest_sha256", id="bundle-sha"),
    ],
)
def test_validate_requires_every_manifest_identity_field(
    tmp_path: Path, field: str
) -> None:
    """Omitting the field must not buy a pass the way a mismatch cannot."""
    _write_run_dir(tmp_path, l1_walls=[10.0] * PAIR_COUNT, l2_walls=[12.0] * PAIR_COUNT)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    del manifest[field]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))

    report = validate_run_dir(tmp_path)

    assert report.valid is False
    assert any(
        f"missing required identity field {field!r}" in finding
        for finding in report.findings
    )


# --------------------------------------------------------------------------
# FRESH_REPORTED: the disclosure token (Amendment 2)
# --------------------------------------------------------------------------


def _disclosure_rows(*, oracle_objective: float = 1.0) -> list[Any]:
    """The single cold pair a disclosure rung produces."""
    return [
        parse_row(row, source=f"{row['lane']}-pair0")
        for row in (
            _producer_row(
                lane=L1_LANE,
                pair_index=0,
                rung="b3-cold-disclosure",
                budget=3,
                wall=140.0,
                evaluations=9,
                oracle_objective=oracle_objective,
            ),
            _producer_row(
                lane=L2_LANE,
                pair_index=0,
                rung="b3-cold-disclosure",
                budget=3,
                wall=60.0,
                evaluations=9,
                oracle_objective=1.0,
            ),
        )
    ]


def test_a_disclosure_pair_is_fresh_reported_not_not_produced() -> None:
    """One cold pair is the whole shape; the five-pair rule does not apply."""
    outcome, pairs = adjudicate_rows(
        _disclosure_rows(), rung="b3-cold-disclosure", disclosure=True
    )

    assert len(pairs) == 1
    assert outcome.verdict is Verdict.FRESH_REPORTED
    assert outcome.failures == ()
    # A cold fused leg is slower here, and that must not read as a verdict.
    assert outcome.median_speedup is not None
    assert outcome.median_speedup < 1.0


def test_the_same_single_pair_is_not_produced_as_a_verdict_rung() -> None:
    """Without the disclosure flag the pair-count rule still voids N=1."""
    outcome, _pairs = adjudicate_rows(_disclosure_rows(), rung="b3")

    assert outcome.verdict is Verdict.NOT_PRODUCED
    assert any("pair_count_1_expected_5" in failure for failure in outcome.failures)


def test_a_disclosure_whose_gates_fail_is_still_voided() -> None:
    """FRESH_REPORTED labels evidence, it does not excuse a failed gate."""
    outcome, _pairs = adjudicate_rows(
        _disclosure_rows(oracle_objective=5.0),
        rung="b3-cold-disclosure",
        disclosure=True,
    )

    assert outcome.verdict is Verdict.NOT_PRODUCED
    assert any(
        "l1_objective_above_paired_native" in failure for failure in outcome.failures
    )


def test_validate_reads_the_disclosure_flag_from_the_manifest(
    tmp_path: Path,
) -> None:
    """A disclosure run directory round-trips as FRESH_REPORTED."""
    rows = [
        _producer_row(
            lane=L1_LANE,
            pair_index=0,
            rung="b3-cold-disclosure",
            budget=3,
            wall=140.0,
            evaluations=9,
            oracle_objective=1.0,
        ),
        _producer_row(
            lane=L2_LANE,
            pair_index=0,
            rung="b3-cold-disclosure",
            budget=3,
            wall=60.0,
            evaluations=9,
            oracle_objective=1.0,
        ),
    ]
    _write_rows(tmp_path, rows)
    parsed = [parse_row(row, source=f"{row['lane']}-pair0") for row in rows]
    outcome, pairs = adjudicate_rows(parsed, rung="b3-cold-disclosure", disclosure=True)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": F3_RUN_MANIFEST_SCHEMA,
                "rung": "b3-cold-disclosure",
                "budget": 3,
                "disclosure_only": True,
                "f3_charter_sha256": F3_CHARTER_SHA256,
                "anchor_process_wall_seconds": outcome.anchor_seconds,
                "verdict": outcome.verdict.value,
                "not_produced_pairs": outcome.not_produced_pairs,
                "timed_legs": 2,
                "solve_child_processes": SOLVE_CHILDREN_PER_COLD_PAIR,
                "campaign_wall_seconds": 300.0,
                "production_commit": PRODUCTION_COMMIT,
                "instrument_commit": INSTRUMENT_COMMIT,
                "campaign_input_manifest_sha256": BUNDLE_SHA,
            },
            sort_keys=True,
        )
    )

    report = validate_run_dir(tmp_path)

    assert report.findings == ()
    assert report.valid is True
    assert report.recomputed_verdict is Verdict.FRESH_REPORTED
    assert len(pairs) == 1


def test_fresh_reported_is_not_a_rung_verdict() -> None:
    """The three rung verdicts stay exactly what the charter froze."""
    assert {verdict.value for verdict in Verdict} == {
        "WIN",
        "CLOSED_BOUNDED_NEGATIVE",
        "NOT_PRODUCED",
        "FRESH_REPORTED",
    }
    for walls, anchor in (([10.0] * PAIR_COUNT, 58.702),):
        outcome = adjudicate_rung(
            l1_walls=walls, l2_walls=[12.0] * PAIR_COUNT, anchor_seconds=anchor
        )
        assert outcome.verdict is not Verdict.FRESH_REPORTED


# --------------------------------------------------------------------------
# Amendment 2's process-cap arithmetic
# --------------------------------------------------------------------------


def test_the_chartered_campaign_shape_is_127_children() -> None:
    """Amendment 2 pins the arithmetic at this instrument's own accounting."""
    warm_pairs = 3 * PAIR_COUNT
    cold_pairs = 3
    bq_search_children = 1 + BQ_MAX_PROBES_PER_LANE + BQ_MAX_PROBES_PER_LANE
    total = (
        warm_pairs * SOLVE_CHILDREN_PER_WARM_PAIR
        + cold_pairs * SOLVE_CHILDREN_PER_COLD_PAIR
        + bq_search_children
    )

    assert (warm_pairs, cold_pairs, bq_search_children) == (15, 3, 25)
    assert total == 127
    assert CapLedger(solve_child_processes=total).breaches() == []
    assert total <= MAX_SOLVE_CHILD_PROCESSES


# --------------------------------------------------------------------------
# The BQ disclosure pair (charter: one fresh-cache pair PER RUNG, three total)
# --------------------------------------------------------------------------


def test_three_rungs_each_get_a_disclosure_pair() -> None:
    """BQ is a disclosure rung even though it has no fixed budget."""
    assert DISCLOSURE_RUNGS == ("b3", "b37", "bq")
    assert set(RUNG_BUDGETS) < set(DISCLOSURE_RUNGS)
    assert DISCLOSURE_RUNG_SUFFIX == "-cold-disclosure"


def test_a_bq_disclosure_rung_keeps_the_quality_gate() -> None:
    """Its lanes ran different budgets, so a fixed-budget gate would misjudge.

    The fused endpoint here is worse than the native one — legitimate when the
    two ran m* and n* — and only the Q*-relative gate reads that correctly.
    """
    rows = [
        parse_row(row, source=f"{row['lane']}-pair0")
        for row in (
            _producer_row(
                lane=L1_LANE,
                pair_index=0,
                rung=f"bq{DISCLOSURE_RUNG_SUFFIX}",
                budget=22,
                wall=140.0,
                evaluations=24,
                oracle_objective=1.4,
            ),
            _producer_row(
                lane=L2_LANE,
                pair_index=0,
                rung=f"bq{DISCLOSURE_RUNG_SUFFIX}",
                budget=31,
                wall=60.0,
                evaluations=31,
                oracle_objective=1.2,
            ),
        )
    ]

    with_target, _pairs = adjudicate_rows(
        rows,
        rung=f"bq{DISCLOSURE_RUNG_SUFFIX}",
        quality_target=1.5,
        disclosure=True,
    )
    without_target, _pairs = adjudicate_rows(
        rows, rung=f"bq{DISCLOSURE_RUNG_SUFFIX}", disclosure=True
    )

    assert with_target.verdict is Verdict.FRESH_REPORTED
    assert with_target.failures == ()
    # Absent Q* the pair cannot be judged at all, and says so.
    assert any(
        "bq_quality_target_absent" in failure for failure in without_target.failures
    )


def test_bq_disclosure_round_trips_at_m_star_not_equal_n_star(
    tmp_path: Path,
) -> None:
    """The whole point: a cold BQ pair at searched budgets must validate."""
    rows = [
        _producer_row(
            lane=L1_LANE,
            pair_index=0,
            rung=f"bq{DISCLOSURE_RUNG_SUFFIX}",
            budget=22,
            wall=140.0,
            evaluations=24,
            oracle_objective=1.4,
        ),
        _producer_row(
            lane=L2_LANE,
            pair_index=0,
            rung=f"bq{DISCLOSURE_RUNG_SUFFIX}",
            budget=31,
            wall=60.0,
            evaluations=31,
            oracle_objective=1.2,
        ),
    ]
    _write_rows(tmp_path, rows)
    parsed = [parse_row(row, source=f"{row['lane']}-pair0") for row in rows]
    outcome, pairs = adjudicate_rows(
        parsed,
        rung=f"bq{DISCLOSURE_RUNG_SUFFIX}",
        quality_target=1.5,
        disclosure=True,
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": F3_RUN_MANIFEST_SCHEMA,
                "rung": f"bq{DISCLOSURE_RUNG_SUFFIX}",
                "budget": 22,
                "native_budget": 31,
                "quality_target": 1.5,
                "disclosure_only": True,
                "f3_charter_sha256": F3_CHARTER_SHA256,
                "anchor_process_wall_seconds": outcome.anchor_seconds,
                "verdict": outcome.verdict.value,
                "not_produced_pairs": outcome.not_produced_pairs,
                "timed_legs": 2,
                "solve_child_processes": SOLVE_CHILDREN_PER_COLD_PAIR,
                "campaign_wall_seconds": 400.0,
                "production_commit": PRODUCTION_COMMIT,
                "instrument_commit": INSTRUMENT_COMMIT,
                "campaign_input_manifest_sha256": BUNDLE_SHA,
            },
            sort_keys=True,
        )
    )

    report = validate_run_dir(tmp_path)

    assert report.findings == ()
    assert report.valid is True
    assert report.recomputed_verdict is Verdict.FRESH_REPORTED
    assert len(pairs) == 1
    # Each lane hashed at its own searched budget, as the timed BQ pairs do.
    assert rows[0]["policy_identity_sha256"] == policy_identity_sha256(22)
    assert rows[1]["policy_identity_sha256"] == policy_identity_sha256(31)


@pytest.mark.parametrize("rung", ["b3", "b37"])
def test_a_fixed_budget_disclosure_uses_its_chartered_budget(rung: str) -> None:
    budgets = resolve_disclosure_budgets(rung)

    assert budgets.fused_maxiter == budgets.native_maxiter == RUNG_BUDGETS[rung]
    assert budgets.quality_target is None
    assert budgets.rung_label == f"{rung}{DISCLOSURE_RUNG_SUFFIX}"


@pytest.mark.parametrize("rung", ["b3", "b37"])
@pytest.mark.parametrize(
    "supply",
    [
        pytest.param(
            lambda rung: resolve_disclosure_budgets(rung, fused_maxiter=22),
            id="fused-maxiter",
        ),
        pytest.param(
            lambda rung: resolve_disclosure_budgets(rung, native_maxiter=31),
            id="native-maxiter",
        ),
        pytest.param(
            lambda rung: resolve_disclosure_budgets(rung, quality_target=1.5),
            id="quality-target",
        ),
    ],
)
def test_a_fixed_budget_disclosure_refuses_a_searched_budget(
    rung: str, supply: Callable[[str], object]
) -> None:
    """B3/B37 disclose at the chartered budget; a searched one is a mistake."""
    with pytest.raises(F3ContractError, match="takes no"):
        supply(rung)


def test_a_bq_disclosure_takes_the_searched_budgets() -> None:
    budgets = resolve_disclosure_budgets(
        "bq", fused_maxiter=22, native_maxiter=31, quality_target=1.5
    )

    assert (budgets.fused_maxiter, budgets.native_maxiter) == (22, 31)
    assert budgets.quality_target == 1.5
    assert budgets.rung_label == f"bq{DISCLOSURE_RUNG_SUFFIX}"


@pytest.mark.parametrize(
    "omitted",
    [
        pytest.param("fused_maxiter", id="no-m-star"),
        pytest.param("native_maxiter", id="no-n-star"),
        pytest.param("quality_target", id="no-quality-target"),
    ],
)
def test_a_bq_disclosure_requires_every_searched_value(omitted: str) -> None:
    """m*, n* and Q* have no chartered defaults, so none may be inferred."""
    with pytest.raises(F3ContractError, match="requires"):
        resolve_disclosure_budgets(
            "bq",
            fused_maxiter=None if omitted == "fused_maxiter" else 22,
            native_maxiter=None if omitted == "native_maxiter" else 31,
            quality_target=None if omitted == "quality_target" else 1.5,
        )


def test_only_the_three_chartered_rungs_may_disclose() -> None:
    with pytest.raises(F3ContractError, match="not a disclosure rung"):
        resolve_disclosure_budgets("b50")


# --------------------------------------------------------------------------
# Cross-tree child environment (the L1 lane runs a production-tree child from
# an environment the instrument built)
# --------------------------------------------------------------------------

# The variable set the instrument's builder actually produced for a GPU child,
# transcribed from a live gpu_environment() call.
INSTRUMENT_GPU_ENVIRONMENT = {
    "JAX_ENABLE_X64": "1",
    "JAX_PLATFORMS": "cuda,cpu",
    "SIMSOPT_ADJOINT_LINEAR_SOLVER": "dense",
    "SIMSOPT_BACKEND": "jax",
    "SIMSOPT_BACKEND_MODE": "jax_gpu_parity",
    "SIMSOPT_BACKEND_STRICT": "1",
    "SIMSOPT_JAX_BACKEND": "cuda",
    "SIMSOPT_JAX_CHUNK_AUTOTUNE": "1",
    "SIMSOPT_JAX_CUDA_LIBRARY_MODE": "bundled",
    "SIMSOPT_JAX_DEBUG_NANS": "0",
    "SIMSOPT_JAX_DISABLE_JIT": "0",
    "SIMSOPT_JAX_GPU_PREALLOCATE": "false",
    "SIMSOPT_JAX_PLATFORM": "cuda",
    "SIMSOPT_JAX_SHARDING": "none",
    "SIMSOPT_JAX_TRANSFER_GUARD": "disallow",
    "SIMSOPT_LBFGS_DEBUG": "0",
    "SIMSOPT_MAX_DENSE_JACOBIAN_BYTES_GPU": "268435456",
    "SIMSOPT_MIXED_PRECISION": "0",
    "SIMSOPT_TRACEABLE_NEWTON_LINEAR_SOLVER": "hybrid_final_dense_ir",
    "STAGE2_BACKEND": "jax",
}


def test_the_adapter_removes_the_variable_production_refuses() -> None:
    adapted = production_child_environment(INSTRUMENT_GPU_ENVIRONMENT)

    assert OBSOLETE_PRECISION_ENV in INSTRUMENT_GPU_ENVIRONMENT
    assert OBSOLETE_PRECISION_ENV not in adapted


def test_the_adapter_names_no_precision_of_its_own() -> None:
    """Absence resolves to fp64; naming it would be a second source for it."""
    adapted = production_child_environment(INSTRUMENT_GPU_ENVIRONMENT)

    assert "SIMSOPT_PRECISION" not in adapted


def test_the_adapter_changes_nothing_else() -> None:
    """Exactly one variable moves, so the lane under test is otherwise intact."""
    adapted = production_child_environment(INSTRUMENT_GPU_ENVIRONMENT)

    assert set(INSTRUMENT_GPU_ENVIRONMENT) - set(adapted) == {OBSOLETE_PRECISION_ENV}
    assert all(
        adapted[name] == value
        for name, value in INSTRUMENT_GPU_ENVIRONMENT.items()
        if name != OBSOLETE_PRECISION_ENV
    )


def test_the_adapter_does_not_mutate_its_input() -> None:
    original = dict(INSTRUMENT_GPU_ENVIRONMENT)

    production_child_environment(INSTRUMENT_GPU_ENVIRONMENT)

    assert INSTRUMENT_GPU_ENVIRONMENT == original


def test_the_adapter_is_idempotent_and_accepts_a_clean_environment() -> None:
    once = production_child_environment(INSTRUMENT_GPU_ENVIRONMENT)

    assert production_child_environment(once) == once


def _resolve_production_policy() -> Any:
    """Resolve the production backend policy from the ambient environment."""
    from simsopt_jax.backend._runtime_policy import (
        _config_from_mode,
        _policy_from_config,
    )

    return _policy_from_config(_config_from_mode("jax_gpu_parity", strict=True))


def test_the_unadapted_environment_is_what_killed_the_l1_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bug, reproduced against the production tree's own resolver."""
    for name, value in INSTRUMENT_GPU_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match="SIMSOPT_MIXED_PRECISION is not supported"):
        _resolve_production_policy()


def test_the_adapted_environment_resolves_to_float64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The audit, executable: the whole instrument set minus one is accepted.

    This is what makes "exactly one variable" a measurement rather than a
    reading of the source, and it pins the fused lane's fp64 contract to the
    absence of SIMSOPT_PRECISION rather than to a variable someone set.
    """
    monkeypatch.delenv(OBSOLETE_PRECISION_ENV, raising=False)
    adapted = production_child_environment(INSTRUMENT_GPU_ENVIRONMENT)
    for name, value in adapted.items():
        monkeypatch.setenv(name, value)

    policy = _resolve_production_policy()

    assert policy.resolved_precision == "fp64"
