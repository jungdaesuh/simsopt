"""Unit tests for pure mixed pre-Newton seed / fallback policy (R05 SSOT)."""

from __future__ import annotations

import numpy as np
import pytest

from simsopt_jax_adapters.geo.pre_newton_seed_policy import (
    MixedPreNewtonSeedPolicyResult,
    combine_mixed_seed_gate,
    mixed_bounded_result_flags,
    mixed_canonical_fallback_flags,
    resolve_mixed_pre_newton_seed_policy,
    route_after_bounded_attempt,
    route_after_seed_gate,
)

_PROPOSAL = np.asarray([0.5, -1.0], dtype=np.float64)
_ORIGINAL = np.asarray([1.0, -2.0], dtype=np.float64)


def _assert_selected(
    result: MixedPreNewtonSeedPolicyResult, expected: np.ndarray
) -> None:
    np.testing.assert_array_equal(result.selected_seed_x, expected)
    assert result.selected_seed_x is not expected
    assert result.selected_seed_x.flags.writeable


@pytest.mark.parametrize(
    (
        "case",
        "proposal_success",
        "seed_candidate_accepted",
        "bounded_certificate_success",
        "expect_gate",
        "expect_source",
        "expect_attempt_bounded",
        "expect_fallback",
        "expect_mixed_seed",
        "expect_bounded_cert",
        "expect_stage",
        "expect_x",
    ),
    [
        (
            "success_pending_bounded",
            True,
            True,
            None,
            True,
            "proposal",
            True,
            False,
            True,
            False,
            "pending_bounded",
            _PROPOSAL,
        ),
        (
            "success_bounded_kept",
            True,
            True,
            True,
            True,
            "proposal",
            False,
            False,
            True,
            True,
            "proposal_bounded",
            _PROPOSAL,
        ),
        (
            "success_bounded_fail_canonical",
            True,
            True,
            False,
            True,
            "original",
            False,
            True,
            True,
            False,
            "canonical_fallback",
            _ORIGINAL,
        ),
        (
            "maxiter_pre_newton_reject",
            False,
            True,
            None,
            False,
            "original",
            False,
            True,
            False,
            False,
            "canonical_fallback",
            _ORIGINAL,
        ),
        (
            "line_search_fail_pre_newton_reject",
            False,
            True,
            False,
            False,
            "original",
            False,
            True,
            False,
            False,
            "canonical_fallback",
            _ORIGINAL,
        ),
        (
            "gate_reject_merit",
            True,
            False,
            None,
            False,
            "original",
            False,
            True,
            False,
            False,
            "canonical_fallback",
            _ORIGINAL,
        ),
        (
            "nonfinite_via_merit_reject",
            True,
            False,
            True,
            False,
            "original",
            False,
            True,
            False,
            False,
            "canonical_fallback",
            _ORIGINAL,
        ),
        (
            "both_reject",
            False,
            False,
            None,
            False,
            "original",
            False,
            True,
            False,
            False,
            "canonical_fallback",
            _ORIGINAL,
        ),
    ],
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_resolve_mixed_pre_newton_seed_policy_table(
    case,
    proposal_success,
    seed_candidate_accepted,
    bounded_certificate_success,
    expect_gate,
    expect_source,
    expect_attempt_bounded,
    expect_fallback,
    expect_mixed_seed,
    expect_bounded_cert,
    expect_stage,
    expect_x,
):
    del case
    result = resolve_mixed_pre_newton_seed_policy(
        proposal_success=proposal_success,
        seed_candidate_accepted=seed_candidate_accepted,
        proposal_x=_PROPOSAL,
        original_x=_ORIGINAL,
        bounded_certificate_success=bounded_certificate_success,
    )

    assert result.seed_gate_accepted is expect_gate
    assert result.seed_source == expect_source
    assert result.attempt_bounded_mixed is expect_attempt_bounded
    assert result.canonical_fallback_used is expect_fallback
    assert result.mixed_seed_accepted is expect_mixed_seed
    assert result.mixed_bounded_certificate_accepted is expect_bounded_cert
    assert result.accepted_stage == expect_stage
    _assert_selected(result, expect_x)


def test_combine_mixed_seed_gate_matches_historical_and() -> None:
    assert (
        combine_mixed_seed_gate(
            proposal_success=True,
            seed_candidate_accepted=True,
        )
        is True
    )
    assert (
        combine_mixed_seed_gate(
            proposal_success=True,
            seed_candidate_accepted=False,
        )
        is False
    )
    assert (
        combine_mixed_seed_gate(
            proposal_success=False,
            seed_candidate_accepted=True,
        )
        is False
    )
    assert (
        combine_mixed_seed_gate(
            proposal_success=False,
            seed_candidate_accepted=False,
        )
        is False
    )


def test_mixed_telemetry_flag_helpers_match_publish_contract() -> None:
    bounded = mixed_bounded_result_flags(bounded_certificate_success=True)
    assert bounded == {
        "canonical_fallback_used": False,
        "mixed_seed_accepted": True,
        "mixed_bounded_certificate_accepted": True,
    }
    bounded_fail = mixed_bounded_result_flags(bounded_certificate_success=False)
    assert bounded_fail["canonical_fallback_used"] is False
    assert bounded_fail["mixed_seed_accepted"] is True
    assert bounded_fail["mixed_bounded_certificate_accepted"] is False

    canon_after_gate_reject = mixed_canonical_fallback_flags(seed_gate_accepted=False)
    assert canon_after_gate_reject == {
        "canonical_fallback_used": True,
        "mixed_seed_accepted": False,
        "mixed_bounded_certificate_accepted": False,
    }
    canon_after_bounded_fail = mixed_canonical_fallback_flags(seed_gate_accepted=True)
    assert canon_after_bounded_fail == {
        "canonical_fallback_used": True,
        "mixed_seed_accepted": True,
        "mixed_bounded_certificate_accepted": False,
    }


def test_selected_seed_is_independent_copy() -> None:
    proposal = np.asarray([3.0, 4.0])
    original = np.asarray([5.0, 6.0])
    result = resolve_mixed_pre_newton_seed_policy(
        proposal_success=True,
        seed_candidate_accepted=True,
        proposal_x=proposal,
        original_x=original,
        bounded_certificate_success=None,
    )
    result.selected_seed_x[0] = -99.0
    assert proposal[0] == 3.0


def test_route_after_seed_gate_matches_ssot_tree() -> None:
    accepted = route_after_seed_gate(
        True,
        when_accepted=lambda _: np.int32(1),
        when_rejected=lambda _: np.int32(0),
    )
    rejected = route_after_seed_gate(
        False,
        when_accepted=lambda _: np.int32(1),
        when_rejected=lambda _: np.int32(0),
    )
    assert int(accepted) == 1
    assert int(rejected) == 0


def test_route_after_bounded_attempt_matches_ssot_tree() -> None:
    keep = route_after_bounded_attempt(
        True,
        keep_bounded=lambda _: np.int32(2),
        run_canonical=lambda _: np.int32(3),
    )
    fallback = route_after_bounded_attempt(
        False,
        keep_bounded=lambda _: np.int32(2),
        run_canonical=lambda _: np.int32(3),
    )
    assert int(keep) == 2
    assert int(fallback) == 3


def test_production_mixed_pipeline_imports_route_ssot() -> None:
    """Production decision tree must own the SSOT route helpers, not re-inline them."""
    from pathlib import Path

    source = Path("src/simsopt_jax_adapters/geo/boozer_surface.py").read_text(
        encoding="utf-8"
    )
    assert "route_after_seed_gate" in source
    assert "route_after_bounded_attempt" in source
    assert "combine_mixed_seed_gate" in source
    # Inline lax.cond for these two decisions is no longer the SSOT owner.
    assert "jax.lax.cond(\n            seed_gate_accepted," not in source
