"""Inner-solve contract for the nested Boozer-LS Schur-Newton walk.

Two Phase-3 rungs of ``docs/nested_ls_upgrade_implementation_plan.md``, both
pure and both testable without a solve: the three-valued exit status, and the
Delta-c sub-step ladder's point planner.

The walk has always published one bit, ``success``. ``exit_status`` splits the failing
half into ``coarse_converged`` and ``failed`` without moving ``success``, so
these tests exist to prove two things at once: that the new status
discriminates where it claims to, and that it changed nothing a consumer
already reads.
"""

from __future__ import annotations

import dataclasses
import inspect
import math

import numpy as np
import pytest
from simsopt_jax_adapters.geo.nested_ls_contract import (
    NESTED_LS_COARSE_AMPLIFICATION_GRADIENT,
    NESTED_LS_COARSE_AMPLIFICATION_VALUE,
    NESTED_LS_COARSE_USE_COMMITTED_ANCHOR,
    NESTED_LS_COARSE_USE_LINE_SEARCH_VALUE,
    NESTED_LS_COARSE_USE_OUTER_GRADIENT,
    NESTED_LS_COARSE_USES,
    NESTED_LS_INNER_POLICY_NAME,
    NESTED_LS_INNER_SUBSTEP_LEGS,
    NESTED_LS_NEWTON_COARSE_TOL,
    NESTED_LS_NEWTON_MAXITER,
    NESTED_LS_NEWTON_EXIT_COARSE_CONVERGED,
    NESTED_LS_NEWTON_EXIT_CONVERGED,
    NESTED_LS_NEWTON_EXIT_FAILED,
    NESTED_LS_NEWTON_EXIT_STATUSES,
    NESTED_LS_NEWTON_TOL,
    NESTED_LS_OUTER_FD0_REL_TOL,
    NESTED_LS_OUTER_IOTA_BRANCH_GUARD,
    NESTED_LS_OUTER_JAX_CHILD_SCHEMA,
    NESTED_LS_OUTER_NATIVE_CHILD_SCHEMA,
    NESTED_LS_PREDICTOR_ARM_BARE,
    NESTED_LS_PREDICTOR_ARM_PREDICTED,
    NESTED_LS_PREDICTOR_TRUST_REGION_RATIO,
    nested_ls_coarse_tier_admits,
    nested_ls_coarse_tier_error_bound,
    nested_ls_inner_policy,
    nested_ls_inner_substep_points,
    nested_ls_predictor_arm,
    nested_ls_predictor_trust_region,
    nested_ls_newton_exit_status,
)

#: ``||g||_2`` at the three late rejections of the B37 v2 diagnostic
#: (``docs/receipts/evidence/nested_ls_outer_b37v2_20260824_jax_diagnostic.json``,
#: outer evaluations 39, 43 and 53), read out of ``rejection_detail``.
#:
#: Each spent 9, 10 and 10 of ``NESTED_LS_NEWTON_MAXITER = 10`` Newton
#: iterations -- 43 and 53 exhausted the budget outright. The ledger string
#: labels that count "iterations left" while carrying iterations completed,
#: so these figures are read against the code, not the label; see
#: ``NestedLsInnerSolveFailed``, where the label is now corrected.
B37_LATE_REJECTION_GRAD_L2 = (
    1.8926946763102574e-2,
    9.934756610314045e-4,
    4.9912673146793665e-3,
)


def _status(grad_l2: float, *, persisted: bool = True, finite: bool = True) -> str:
    return nested_ls_newton_exit_status(
        persisted=persisted,
        finite_iterate=finite,
        reduced_gradient_l2=grad_l2,
        tol=NESTED_LS_NEWTON_TOL,
    )


def test_statuses_tuple_is_the_three_distinct_constants() -> None:
    assert NESTED_LS_NEWTON_EXIT_STATUSES == (
        NESTED_LS_NEWTON_EXIT_CONVERGED,
        NESTED_LS_NEWTON_EXIT_COARSE_CONVERGED,
        NESTED_LS_NEWTON_EXIT_FAILED,
    )
    assert len(set(NESTED_LS_NEWTON_EXIT_STATUSES)) == 3


def test_coarse_band_sits_strictly_above_the_certification_tolerance() -> None:
    """A coarse band at or under ``tol`` would be an empty band."""

    assert NESTED_LS_NEWTON_COARSE_TOL > NESTED_LS_NEWTON_TOL
    assert _status(NESTED_LS_NEWTON_COARSE_TOL) == (
        NESTED_LS_NEWTON_EXIT_COARSE_CONVERGED
    )


@pytest.mark.parametrize(
    ("grad_l2", "expected"),
    [
        (0.0, NESTED_LS_NEWTON_EXIT_CONVERGED),
        (NESTED_LS_NEWTON_TOL * 0.1, NESTED_LS_NEWTON_EXIT_CONVERGED),
        (NESTED_LS_NEWTON_TOL, NESTED_LS_NEWTON_EXIT_CONVERGED),
        (NESTED_LS_NEWTON_TOL * 10.0, NESTED_LS_NEWTON_EXIT_COARSE_CONVERGED),
        (NESTED_LS_NEWTON_COARSE_TOL, NESTED_LS_NEWTON_EXIT_COARSE_CONVERGED),
        (NESTED_LS_NEWTON_COARSE_TOL * 10.0, NESTED_LS_NEWTON_EXIT_FAILED),
        (math.inf, NESTED_LS_NEWTON_EXIT_FAILED),
    ],
)
def test_bands_are_closed_on_the_right(grad_l2: float, expected: str) -> None:
    assert _status(grad_l2) == expected


def test_the_measured_b37_late_rejections_are_not_rescued_by_the_coarse_tier() -> None:
    """The coarse tier does not reclassify the failures that motivated it.

    This is the honest, load-bearing negative. All three late rejections
    sit at least five orders of magnitude above
    ``NESTED_LS_NEWTON_COARSE_TOL``, so licensing a coarse tier in Phase 4
    cannot turn any of them into a usable result -- and two of the three
    exhausted the inner budget, which no exit-status refinement addresses
    either. If someone later widens the coarse band far enough to swallow
    them, this test fails and forces that decision into the open.
    """

    for grad_l2 in B37_LATE_REJECTION_GRAD_L2:
        assert grad_l2 > NESTED_LS_NEWTON_COARSE_TOL * 1.0e4
        assert _status(grad_l2) == NESTED_LS_NEWTON_EXIT_FAILED


def test_a_non_persisted_walk_is_failed_even_at_a_converged_gradient() -> None:
    """The ``persisted`` gate is not decoration.

    A non-persisted walk reports the START surface and start gradient. Its
    norm describes a point the result does not carry, so classifying by it
    would stamp the caller's own input as a solve outcome. Dropping the
    gate is the obvious "simplification" of this function; this is the
    assertion that catches it.
    """

    assert _status(0.0, persisted=False) == NESTED_LS_NEWTON_EXIT_FAILED
    assert _status(NESTED_LS_NEWTON_TOL * 10.0, persisted=False) == (
        NESTED_LS_NEWTON_EXIT_FAILED
    )


def test_a_non_finite_iterate_is_failed_at_every_norm() -> None:
    for grad_l2 in (0.0, NESTED_LS_NEWTON_TOL, math.nan, math.inf):
        assert _status(grad_l2, finite=False) == NESTED_LS_NEWTON_EXIT_FAILED


def test_an_inverted_band_degrades_to_two_valued_and_never_demotes() -> None:
    """``coarse_tol < tol`` must not demote a converged walk."""

    assert (
        nested_ls_newton_exit_status(
            persisted=True,
            finite_iterate=True,
            reduced_gradient_l2=NESTED_LS_NEWTON_TOL * 0.1,
            tol=NESTED_LS_NEWTON_TOL,
            coarse_tol=NESTED_LS_NEWTON_TOL * 1.0e-7,
        )
        == NESTED_LS_NEWTON_EXIT_CONVERGED
    )
    assert (
        nested_ls_newton_exit_status(
            persisted=True,
            finite_iterate=True,
            reduced_gradient_l2=NESTED_LS_NEWTON_TOL * 10.0,
            tol=NESTED_LS_NEWTON_TOL,
            coarse_tol=NESTED_LS_NEWTON_TOL * 1.0e-7,
        )
        == NESTED_LS_NEWTON_EXIT_FAILED
    )


def test_converged_is_identically_the_success_predicate_over_a_swept_grid() -> None:
    """``exit_status == "converged"`` must equal what ``success`` already was.

    The solver computes ``committed_success = persist and finite and
    final_norm <= tol``. If the two ever disagree, a consumer reading the
    new field sees a different lane than one reading the old bit, which is
    precisely the SSOT break this refinement is supposed to avoid. Swept
    rather than spot-checked, and deliberately dense across both band
    edges.
    """

    norms = np.concatenate(
        [
            np.array([0.0, math.inf]),
            np.logspace(-16.0, 2.0, 73),
            np.array(
                [
                    NESTED_LS_NEWTON_TOL,
                    np.nextafter(NESTED_LS_NEWTON_TOL, 0.0),
                    np.nextafter(NESTED_LS_NEWTON_TOL, math.inf),
                    NESTED_LS_NEWTON_COARSE_TOL,
                    np.nextafter(NESTED_LS_NEWTON_COARSE_TOL, 0.0),
                    np.nextafter(NESTED_LS_NEWTON_COARSE_TOL, math.inf),
                ]
            ),
        ]
    )
    checked = 0
    for persisted in (False, True):
        for finite in (False, True):
            for norm in norms:
                committed_success = bool(
                    persisted and finite and norm <= (NESTED_LS_NEWTON_TOL)
                )
                status = nested_ls_newton_exit_status(
                    persisted=persisted,
                    finite_iterate=finite,
                    reduced_gradient_l2=float(norm),
                    tol=NESTED_LS_NEWTON_TOL,
                )
                assert (status == NESTED_LS_NEWTON_EXIT_CONVERGED) is committed_success
                assert status in NESTED_LS_NEWTON_EXIT_STATUSES
                checked += 1
    assert checked == 4 * norms.size


def test_every_returned_status_is_a_declared_status() -> None:
    for persisted in (False, True):
        for finite in (False, True):
            for norm in (0.0, 1.0e-14, 1.0e-10, 1.0e-3, math.nan, math.inf):
                assert (
                    nested_ls_newton_exit_status(
                        persisted=persisted,
                        finite_iterate=finite,
                        reduced_gradient_l2=norm,
                        tol=NESTED_LS_NEWTON_TOL,
                    )
                    in NESTED_LS_NEWTON_EXIT_STATUSES
                )


def test_the_classifier_takes_keyword_arguments_only() -> None:
    """Positional calls would silently transpose ``persisted``/``finite``."""

    parameters = inspect.signature(nested_ls_newton_exit_status).parameters
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in parameters.values()
    )


def test_the_result_carries_exit_status_beside_success() -> None:
    """The field is wired onto the walk's result, not only defined."""

    from simsopt_jax_adapters.geo.nested_ls_reduced import NestedLsSchurNewtonResult

    fields = {
        field.name: field for field in dataclasses.fields(NestedLsSchurNewtonResult)
    }
    assert "exit_status" in fields
    assert fields["exit_status"].type in ("str", str)
    assert "success" in fields


# --------------------------------------------------------------------------
# Inner Delta-c sub-stepping (Phase 3)
# --------------------------------------------------------------------------


def test_one_leg_is_the_undivided_step_and_returns_the_trial_itself() -> None:
    """The first ladder rung must cost nothing and change nothing.

    This is what makes sub-stepping free to enable: rung 1 is the step the
    un-sub-stepped lane already takes, so a run with the ladder on
    reproduces a run with it off exactly until the first failure.
    """

    anchor = np.array([0.25, -0.5, 1.0e-9])
    trial = np.array([0.2531863066098380, -0.5031863066098380, 2.0e-9])
    points = nested_ls_inner_substep_points(
        anchor_coil_dofs=anchor, trial_coil_dofs=trial, legs=1
    )
    assert len(points) == 1
    assert np.array_equal(points[0], trial)


@pytest.mark.parametrize("legs", NESTED_LS_INNER_SUBSTEP_LEGS)
def test_the_last_point_is_the_trial_bitwise_at_every_rung(legs: int) -> None:
    """The endpoint is the caller's coils, not a reconstruction of them.

    The outer objective is evaluated at the coils the caller named, so a
    solve that landed at a neighbour would publish ``s*(c')`` labelled
    ``s*(c)``. Asserted bitwise at every rung of the sealed ladder.
    """

    anchor = np.array([0.25, -0.5])
    trial = np.array([0.2531863066098380, -0.5031863066098380])
    points = nested_ls_inner_substep_points(
        anchor_coil_dofs=anchor, trial_coil_dofs=trial, legs=legs
    )
    assert len(points) == legs
    assert np.array_equal(points[-1], trial)
    # And it is a copy, so a caller mutating the result cannot reach back
    # into the trial it was handed.
    assert points[-1] is not trial
    # Known EQUIVALENT MUTANT, recorded rather than left as an apparent gap:
    # replacing the copy with ``anchor + delta`` passes this file. It has to
    # -- probing 120 000 random pairs across six magnitude regimes found no
    # binary64 case where ``a + (t - a) != t``, so no value assertion can
    # separate them. The copy is a guarantee where the alternative is an
    # empirical regularity; that difference is real but is not observable
    # from outside, and pretending otherwise with a contrived assertion
    # would be worse than saying so here.


@pytest.mark.parametrize("legs", NESTED_LS_INNER_SUBSTEP_LEGS)
def test_the_points_advance_monotonically_along_the_displacement(
    legs: int,
) -> None:
    """Each leg is strictly further from the anchor than the last.

    A ladder that repeated or reversed a point would spend solves without
    shortening any step, which is the entire mechanism.
    """

    anchor = np.array([1.0, -2.0, 0.5])
    trial = np.array([1.007, -2.003, 0.5072])
    points = nested_ls_inner_substep_points(
        anchor_coil_dofs=anchor, trial_coil_dofs=trial, legs=legs
    )
    distances = [float(np.linalg.norm(point - anchor)) for point in points]
    assert distances == sorted(distances)
    assert len(set(distances)) == legs
    # No point overshoots the trial.
    total = float(np.linalg.norm(trial - anchor))
    assert distances[-1] == pytest.approx(total, rel=0.0, abs=0.0)


def test_the_anchor_itself_is_never_a_point_to_solve_at() -> None:
    """The anchor is already solved; re-solving it would waste a leg."""

    anchor = np.array([0.25, -0.5])
    trial = np.array([0.3, -0.45])
    for legs in NESTED_LS_INNER_SUBSTEP_LEGS:
        points = nested_ls_inner_substep_points(
            anchor_coil_dofs=anchor, trial_coil_dofs=trial, legs=legs
        )
        assert not any(np.array_equal(point, anchor) for point in points)


def test_each_leg_shortens_the_maximum_step_by_the_leg_count() -> None:
    """The property the whole rung rests on, stated as a number.

    The B37 failures exhaust a fixed Newton budget on a displacement of
    3.2e-3 to 7.3e-3. Dividing into ``legs`` legs makes every step
    ``||dc||/legs``, which is the only lever here -- the budget per leg is
    unchanged.
    """

    anchor = np.array([1.0, -2.0])
    trial = anchor + np.array([7.305683117781034e-3, 0.0])
    total = float(np.linalg.norm(trial - anchor))
    for legs in NESTED_LS_INNER_SUBSTEP_LEGS:
        points = nested_ls_inner_substep_points(
            anchor_coil_dofs=anchor, trial_coil_dofs=trial, legs=legs
        )
        walk = [anchor, *points]
        steps = [
            float(np.linalg.norm(walk[index + 1] - walk[index]))
            for index in range(len(walk) - 1)
        ]
        assert max(steps) == pytest.approx(total / legs)


def test_a_zero_displacement_still_yields_the_trial_at_every_rung() -> None:
    """A trial at the anchor is a real case: scipy re-evaluates x0."""

    anchor = np.array([0.25, -0.5])
    for legs in NESTED_LS_INNER_SUBSTEP_LEGS:
        points = nested_ls_inner_substep_points(
            anchor_coil_dofs=anchor, trial_coil_dofs=anchor, legs=legs
        )
        assert len(points) == legs
        assert np.array_equal(points[-1], anchor)


def test_a_non_positive_leg_count_is_refused() -> None:
    anchor = np.array([0.25, -0.5])
    for legs in (0, -1):
        with pytest.raises(ValueError, match="at least 1"):
            nested_ls_inner_substep_points(
                anchor_coil_dofs=anchor, trial_coil_dofs=anchor, legs=legs
            )


def test_mismatched_coil_blocks_are_refused_rather_than_broadcast() -> None:
    """numpy would happily broadcast (1,) against (11,) and solve nonsense."""

    with pytest.raises(ValueError, match="same shape"):
        nested_ls_inner_substep_points(
            anchor_coil_dofs=np.array([0.25]),
            trial_coil_dofs=np.array([0.25, -0.5]),
            legs=2,
        )


def test_the_sealed_ladder_starts_undivided_and_only_refines() -> None:
    """Ladder shape, pinned: starts at 1, strictly increasing, all positive."""

    ladder = NESTED_LS_INNER_SUBSTEP_LEGS
    assert ladder[0] == 1
    assert list(ladder) == sorted(ladder)
    assert len(set(ladder)) == len(ladder)
    assert all(isinstance(legs, int) and legs >= 1 for legs in ladder)


# --------------------------------------------------------------------------
# Licensed coarse tier and its red test (Phase 4)
# --------------------------------------------------------------------------


def test_a_committed_anchor_is_never_admitted_at_a_coarse_residual() -> None:
    """The red test: coarse bytes must not become a warm start.

    An anchor is the point every later evaluation inherits and the point
    FD-0 differences about, so error there is propagated rather than
    attenuated. This is the assertion that fails if someone widens the
    anchor tier to reuse a cheap solve.
    """

    assert nested_ls_coarse_tier_admits(
        achieved_residual_l2=NESTED_LS_NEWTON_TOL,
        use=NESTED_LS_COARSE_USE_COMMITTED_ANCHOR,
    )
    for residual in (
        np.nextafter(NESTED_LS_NEWTON_TOL, math.inf),
        NESTED_LS_NEWTON_COARSE_TOL,
        1.0e-3,
    ):
        assert not nested_ls_coarse_tier_admits(
            achieved_residual_l2=float(residual),
            use=NESTED_LS_COARSE_USE_COMMITTED_ANCHOR,
        ), f"a coarse residual {residual!r} was admitted as a committed anchor"


def test_a_coarse_result_outside_the_budget_is_refused_for_the_adjoint() -> None:
    """The red test the plan asks for, stated at the tier boundary."""

    assert nested_ls_coarse_tier_admits(
        achieved_residual_l2=NESTED_LS_NEWTON_COARSE_TOL,
        use=NESTED_LS_COARSE_USE_OUTER_GRADIENT,
    )
    assert not nested_ls_coarse_tier_admits(
        achieved_residual_l2=float(np.nextafter(NESTED_LS_NEWTON_COARSE_TOL, math.inf)),
        use=NESTED_LS_COARSE_USE_OUTER_GRADIENT,
    )


def test_production_cannot_reach_the_adjoint_with_a_coarse_result_today() -> None:
    """The tier is a licence nothing has claimed yet, and that is the point.

    ``success`` stays true only for ``converged``, and the outer objective
    raises on a non-successful inner solve BEFORE assembling any adjoint
    (``_nested_ls_outer_objective`` calls ``_solve_nested_inner_at_coils``
    first). So a coarse-converged result is already fail-closed against the
    adjoint without any consumer honouring the tier. This test pins that
    interlock, so licensing the tier later is a deliberate act rather than
    something that happens by a status becoming readable.
    """

    coarse = nested_ls_newton_exit_status(
        persisted=True,
        finite_iterate=True,
        reduced_gradient_l2=NESTED_LS_NEWTON_COARSE_TOL,
        tol=NESTED_LS_NEWTON_TOL,
    )
    assert coarse == NESTED_LS_NEWTON_EXIT_COARSE_CONVERGED
    assert coarse != NESTED_LS_NEWTON_EXIT_CONVERGED


def test_the_value_tier_is_not_looser_than_the_gradient_tier() -> None:
    """The measured inversion, pinned so a future edit cannot undo it quietly.

    The plan ordered its tiers with line-search values as the loosest. The
    measurement says the objective's relative error is ~3.8x the gradient's
    at the same achieved residual, so the value is the BINDING constraint.
    If someone later re-separates the tiers with the value looser, this
    fails.
    """

    assert (
        NESTED_LS_COARSE_AMPLIFICATION_VALUE > NESTED_LS_COARSE_AMPLIFICATION_GRADIENT
    )
    for residual in (1.0e-13, 1.0e-10, NESTED_LS_NEWTON_COARSE_TOL):
        value_admitted = nested_ls_coarse_tier_admits(
            achieved_residual_l2=residual,
            use=NESTED_LS_COARSE_USE_LINE_SEARCH_VALUE,
        )
        gradient_admitted = nested_ls_coarse_tier_admits(
            achieved_residual_l2=residual,
            use=NESTED_LS_COARSE_USE_OUTER_GRADIENT,
        )
        assert value_admitted == gradient_admitted
        assert nested_ls_coarse_tier_error_bound(
            achieved_residual_l2=residual,
            use=NESTED_LS_COARSE_USE_LINE_SEARCH_VALUE,
        ) > nested_ls_coarse_tier_error_bound(
            achieved_residual_l2=residual,
            use=NESTED_LS_COARSE_USE_OUTER_GRADIENT,
        )


def test_the_licensed_tier_sits_far_under_the_fd0_band() -> None:
    """Why 1e-8 is the tier, in the units the decision was made in.

    FD-0's relative band is ``NESTED_LS_OUTER_FD0_REL_TOL``. At the coarse
    tolerance the worst measured amplification puts both the gradient and
    the value error orders of magnitude under it; that margin is the
    licence. Recomputed here rather than quoted, so moving either the tier
    or an amplification constant moves this assertion.
    """

    for use in (
        NESTED_LS_COARSE_USE_OUTER_GRADIENT,
        NESTED_LS_COARSE_USE_LINE_SEARCH_VALUE,
    ):
        bound = nested_ls_coarse_tier_error_bound(
            achieved_residual_l2=NESTED_LS_NEWTON_COARSE_TOL, use=use
        )
        assert bound < NESTED_LS_OUTER_FD0_REL_TOL / 50.0, (
            f"{use} inherits {bound:.3e} at the coarse tier, which is not "
            f"comfortably under the FD-0 band {NESTED_LS_OUTER_FD0_REL_TOL:.0e}"
        )


def test_an_unknown_use_fails_closed_rather_than_defaulting_to_admitted() -> None:
    """A default-admit branch would license the next consumer silently."""

    for function in (nested_ls_coarse_tier_admits, nested_ls_coarse_tier_error_bound):
        with pytest.raises(ValueError, match="unknown nested-LS coarse-tier use"):
            function(achieved_residual_l2=1.0e-14, use="whatever_lands_next")


def test_a_non_finite_residual_is_never_admitted() -> None:
    """NaN fails every comparison, so it must be refused explicitly."""

    for use in NESTED_LS_COARSE_USES:
        for residual in (math.nan, -1.0, -math.inf):
            assert not nested_ls_coarse_tier_admits(
                achieved_residual_l2=residual, use=use
            )


# --------------------------------------------------------------------------
# Predictor trust region and arm rule (Phase 2), now one implementation
# --------------------------------------------------------------------------


def test_the_trust_region_scales_and_never_rejects() -> None:
    """DESC ``tr_ratio``: an oversized step is clipped, not discarded."""

    anchor = np.array([3.0, 4.0])  # norm 5, cap 0.5 at ratio 0.1
    delta = np.array([2.0, 0.0])
    applied, raw, applied_norm, cap, scaled = nested_ls_predictor_trust_region(
        delta_surface=delta, anchor_surface_dofs=anchor, ratio=0.1
    )
    assert scaled is True
    assert cap == pytest.approx(0.5)
    assert raw == pytest.approx(2.0)
    assert applied_norm == pytest.approx(cap)
    # Direction preserved: clipping, not rejection.
    np.testing.assert_allclose(applied, np.array([0.5, 0.0]))


def test_a_step_exactly_at_the_bound_is_left_untouched() -> None:
    """Strict ``>`` triggers scaling, so the boundary is inside the region."""

    anchor = np.array([3.0, 4.0])
    delta = np.array([0.5, 0.0])
    applied, _raw, _applied_norm, cap, scaled = nested_ls_predictor_trust_region(
        delta_surface=delta, anchor_surface_dofs=anchor, ratio=0.1
    )
    assert scaled is False
    assert np.array_equal(applied, delta)
    assert float(np.linalg.norm(delta)) == pytest.approx(cap)


def test_the_applied_step_never_aliases_the_caller_s_array() -> None:
    """Both branches return a fresh array, so a caller cannot write back."""

    anchor = np.array([3.0, 4.0])
    for delta in (np.array([0.1, 0.0]), np.array([9.0, 0.0])):
        applied, *_ = nested_ls_predictor_trust_region(
            delta_surface=delta, anchor_surface_dofs=anchor, ratio=0.1
        )
        assert applied is not delta


def test_a_tie_keeps_the_prediction_so_a_zero_step_is_an_identity() -> None:
    """The tie-break is load-bearing, not arbitrary.

    At ``Δc = 0`` the predicted step is zero, the two starts are the same
    vector, and the two envelope gradients are bitwise equal. Keeping the
    prediction on a tie is what makes a predictor-ON run reproduce a
    predictor-OFF run exactly at an unmoved point.
    """

    assert (
        nested_ls_predictor_arm(bare_gradient_l2=1.5, predicted_gradient_l2=1.5)
        == NESTED_LS_PREDICTOR_ARM_PREDICTED
    )


def test_the_bare_anchor_wins_only_when_the_prediction_is_strictly_worse() -> None:
    assert (
        nested_ls_predictor_arm(bare_gradient_l2=1.0, predicted_gradient_l2=1.0 + 1e-15)
        == NESTED_LS_PREDICTOR_ARM_BARE
    )
    assert (
        nested_ls_predictor_arm(bare_gradient_l2=1.0, predicted_gradient_l2=1.0 - 1e-15)
        == NESTED_LS_PREDICTOR_ARM_PREDICTED
    )


def test_the_harness_and_the_contract_are_the_same_rule() -> None:
    """The measured arithmetic and the shipped arithmetic must be one thing.

    The replay harness is what produced the Phase-2 evidence, so a second
    copy of these rules living there is how a receipt's number quietly stops
    describing the lane it claims to describe. The harness now delegates;
    this pins that it still does.
    """

    from benchmarks import nested_ls_outer_predictor_replay as probe

    assert probe.TRUST_REGION_RATIO == NESTED_LS_PREDICTOR_TRUST_REGION_RATIO
    anchor = np.array([3.0, 4.0])
    for delta in (np.array([2.0, 0.0]), np.array([0.1, 0.0]), np.zeros(2)):
        harness = probe.apply_trust_region(delta, anchor, 0.1)
        shared = nested_ls_predictor_trust_region(
            delta_surface=delta, anchor_surface_dofs=anchor, ratio=0.1
        )
        assert np.array_equal(harness[0], shared[0])
        assert harness[1:] == shared[1:]
    assert probe.select_arm(1.0, 1.0) == nested_ls_predictor_arm(
        bare_gradient_l2=1.0, predicted_gradient_l2=1.0
    )


# --------------------------------------------------------------------------
# Predictor fallback semantics in production (Phase 2 wiring)
# --------------------------------------------------------------------------


def _outer_state(**overrides: object):
    """A ``NestedLsOuterState`` carrying only what the fallback branches read.

    The three branches under test return before touching ``jax_boozer``, the
    flat program, or any device, which is exactly why they are worth testing
    on CPU: they are the paths that decide NOT to predict, and a predictor
    that silently predicts anyway is the failure this covers.
    """

    from simsopt_jax_adapters.geo.nested_ls_reduced_scale import (
        NestedLsOuterAnchor,
        NestedLsOuterState,
        NestedLsOuterTrialReadout,
    )

    anchor = NestedLsOuterAnchor.at(
        coil_dofs=np.array([0.25, -0.5]),
        surface_dofs=np.array([1.0, 2.0, 3.0]),
        iota=0.14,
        G=2.0,
        schur_lu=None,
    )

    class _FakeSurface:
        def __init__(self) -> None:
            self.dofs = None

        def set_dofs(self, dofs: object) -> None:
            self.dofs = np.array(dofs, dtype=np.float64, copy=True)

    class _FakeBiotSavart:
        def __init__(self, owner: object) -> None:
            self._owner = owner
            self._x = None

        @property
        def x(self):
            return self._x

        @x.setter
        def x(self, value: object) -> None:
            self._x = np.array(value, dtype=np.float64, copy=True)
            # Record every coil point installed, in order. WHICH point the
            # mixed term is evaluated at is as load-bearing as which closure
            # family evaluates it: ``source.operator`` was factored at the
            # committed anchor, so evaluating the mixed term anywhere else
            # linearizes about a point the operator does not describe -- and
            # it returns a plausible array rather than raising.
            self._owner.installed_coils.append(self._x)

    class _FakeBoozer:
        def __init__(self) -> None:
            self.surface = _FakeSurface()
            self.installed_coils: list = []
            self.biotsavart = _FakeBiotSavart(self)
            self.refresh_count = 0

        def _refresh_coil_data(self) -> None:
            self.refresh_count += 1

    fields: dict[str, object] = {
        "jax_boozer": _FakeBoozer(),
        "flat_value_and_grad": None,
        "vessel_dofs": np.zeros(1),
        "coil_slice": slice(0, 2),
        "surface_slice": slice(2, 5),
        "anchor": anchor,
        "last_trial": NestedLsOuterTrialReadout(
            anchor=anchor,
            inner_iterations=-1,
            inner_grad_l2=-1.0,
            adjoint_live_eta=-1.0,
        ),
        "inner_substep_legs": (1,),
        "inner_predictor": False,
        "predictor_source": None,
        "last_predictor_arm": NESTED_LS_PREDICTOR_ARM_BARE,
        "last_predictor_raw_delta_l2": -1.0,
        "last_predictor_applied_delta_l2": -1.0,
        "last_predictor_scaled": False,
        "record_mixed_cross_check": False,
        "mixed_cross_check_gradient": None,
        "mixed_cross_check_max_abs": None,
    }
    fields.update(overrides)
    return NestedLsOuterState(**fields)  # type: ignore[arg-type]


def test_the_predictor_is_off_by_default_on_both_new_policies() -> None:
    """A run must opt in to either trajectory change, never inherit it."""

    import inspect

    from simsopt_jax_adapters.geo.nested_ls_reduced_scale import (
        prepare_f3_b37_outer_state,
    )

    parameters = inspect.signature(prepare_f3_b37_outer_state).parameters
    assert parameters["inner_substep"].default is False
    assert parameters["inner_predictor"].default is False
    for name in ("inner_substep", "inner_predictor"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_a_predictor_off_state_starts_from_the_bare_anchor_bytes() -> None:
    """The OFF path must hand back the anchor's own array, untouched.

    The state is given VALID cached machinery, built at the anchor's own
    coils, so the only thing stopping a prediction is the flag. A test that
    left the cache empty would pass whether the flag was checked or not --
    it would prove the source guard twice and the flag guard never.
    """

    from simsopt_jax_adapters.geo.nested_ls_reduced_scale import (
        NestedLsPredictorSource,
        _predicted_inner_start,
    )

    def _explode(_vector):  # pragma: no cover - must never be called
        raise AssertionError("predicted with the predictor switched off")

    state = _outer_state(inner_predictor=False)
    usable = NestedLsPredictorSource(
        coil_dofs=np.array(state.anchor.coil_dofs, copy=True),
        operator=object(),
        apply_lu=_explode,
    )
    assert usable.built_at(state.anchor.coil_dofs)
    state.predictor_source = usable

    surface, arm, raw, applied, scaled = _predicted_inner_start(
        state, np.array([0.3, -0.45])
    )
    assert surface is state.anchor.surface_dofs
    assert arm == NESTED_LS_PREDICTOR_ARM_BARE
    assert (raw, applied, scaled) == (0.0, 0.0, False)


def test_a_predictor_on_state_with_no_cached_machinery_falls_back() -> None:
    """Turning the flag on is not enough; an evaluation must have run."""

    from simsopt_jax_adapters.geo.nested_ls_reduced_scale import (
        _predicted_inner_start,
    )

    state = _outer_state(inner_predictor=True, predictor_source=None)
    surface, arm, _raw, _applied, _scaled = _predicted_inner_start(
        state, np.array([0.3, -0.45])
    )
    assert surface is state.anchor.surface_dofs
    assert arm == NESTED_LS_PREDICTOR_ARM_BARE


def test_machinery_built_at_other_coils_is_refused_not_reused() -> None:
    """The provenance guard, and the reason the cache is safe at all.

    ``predictor_source`` holds whatever the LAST evaluation produced, and
    the last evaluation is frequently a REJECTED trial rather than the
    incumbent. Using that operator would predict a displacement measured
    from a point the committed anchor is not — it would compile, run, and
    return a plausible surface. Nothing else in the system would notice.
    """

    from simsopt_jax_adapters.geo.nested_ls_reduced_scale import (
        NestedLsPredictorSource,
        _predicted_inner_start,
    )

    def _explode(_vector):  # pragma: no cover - must never be called
        raise AssertionError("the predictor used machinery from another point")

    stale = NestedLsPredictorSource(
        coil_dofs=np.array([9.0, 9.0]),  # not the anchor's (0.25, -0.5)
        operator=object(),
        apply_lu=_explode,
    )
    state = _outer_state(inner_predictor=True, predictor_source=stale)
    assert not stale.built_at(state.anchor.coil_dofs)
    surface, arm, _raw, _applied, _scaled = _predicted_inner_start(
        state, np.array([0.3, -0.45])
    )
    assert surface is state.anchor.surface_dofs
    assert arm == NESTED_LS_PREDICTOR_ARM_BARE


def test_built_at_is_bitwise_and_shape_tolerant() -> None:
    """One ULP is a different point; a row vector is the same one."""

    from simsopt_jax_adapters.geo.nested_ls_reduced_scale import (
        NestedLsPredictorSource,
    )

    coils = np.array([0.25, -0.5])
    source = NestedLsPredictorSource(
        coil_dofs=coils, operator=object(), apply_lu=lambda v: v
    )
    assert source.built_at(coils)
    assert source.built_at(coils.reshape(1, -1))
    nudged = np.array(coils, copy=True)
    nudged[0] = np.nextafter(nudged[0], math.inf)
    assert not source.built_at(nudged)


def _predicting_state(monkeypatch, *, raw_delta, bare_grad_l2, predicted_grad_l2):
    """A state whose prediction path runs entirely on fakes.

    ``_predicted_inner_start``'s three collaborators are module-level names
    in the scale module, so they can be replaced the same way the outer
    children's are. Without this the only CPU-reachable branches are the
    ones that decide NOT to predict, and the trust region and the arm rule
    -- the two decisions the prediction path actually makes -- go
    untested. Mutating either survived the suite until this existed.
    """

    import inspect

    from simsopt_jax_adapters.geo import nested_ls_reduced_scale as scale

    state = _outer_state(inner_predictor=True)
    state.predictor_source = scale.NestedLsPredictorSource(
        coil_dofs=np.array(state.anchor.coil_dofs, copy=True),
        operator=object(),
        # The LU returns the NEGATED raw delta, because the production line
        # is ``raw_delta = -apply_lu(mixed)``.
        apply_lu=lambda _mixed: -np.asarray(raw_delta, dtype=np.float64),
    )

    # The two closure families have DIFFERENT arities in production, and the
    # fakes mirror that on purpose. ``nested_ls_reduced_closures`` captures
    # the coils at construction, so its residual takes only the packed
    # decision; ``nested_ls_runtime_coil_closures`` takes
    # ``(packed, coil_dofs)``. The first draft of these fakes handed back
    # bare ``object()``s, which accept anything and therefore could not tell
    # the families apart -- production shipped with the runtime pair wired
    # into the envelope evaluation and these tests stayed green until a real
    # run raised TypeError. Fakes that do not mirror the contract do not
    # defend it.
    def _runtime_residual(_packed, _coil_dofs):  # pragma: no cover - arity only
        raise AssertionError("runtime-coil residual called as a frozen-coil one")

    def _frozen_residual(_packed):  # pragma: no cover - arity only
        raise AssertionError("frozen-coil residual called directly")

    monkeypatch.setattr(
        scale,
        "nested_ls_runtime_coil_closures",
        lambda _b: (_runtime_residual, _runtime_residual, None),
    )
    monkeypatch.setattr(
        scale,
        "nested_ls_reduced_closures",
        lambda _b: (_frozen_residual, _frozen_residual, None),
    )

    def _fake_mixed(residual, objective, _surface, _coils, _step, *, operator):
        # The mirror of the envelope check, in the other direction. The mixed
        # term REQUIRES the runtime-coil family, because it differentiates
        # with respect to the coils; handing it frozen-coil closures does not
        # raise, it silently returns a wrong delta_s. Arity is the only thing
        # that separates them from outside, so assert it.
        del operator
        for closure in (residual, objective):
            parameters = inspect.signature(closure).parameters
            assert len(parameters) == 2, (
                "mixed term evaluated with a frozen-coil closure "
                f"(arity {len(parameters)}); it requires the runtime-coil family"
            )
        return np.zeros_like(np.asarray(raw_delta, dtype=np.float64))

    monkeypatch.setattr(scale, "apply_reduced_mixed_schur_coil_tangent", _fake_mixed)

    def _fake_envelope(residual, objective, surface):
        # Assert the FAMILY, not just that something was passed. A
        # frozen-coil residual takes one argument; the runtime-coil one takes
        # two. This is the check that would have caught the shipped swap.
        for closure in (residual, objective):
            parameters = inspect.signature(closure).parameters
            assert len(parameters) == 1, (
                "envelope evaluated with a runtime-coil closure "
                f"(arity {len(parameters)}); it requires the frozen-coil family"
            )
        is_bare = np.array_equal(surface, state.anchor.surface_dofs)
        norm = bare_grad_l2 if is_bare else predicted_grad_l2
        return 0.0, np.array([float(norm), 0.0, 0.0]), None

    monkeypatch.setattr(scale, "_envelope_value_and_grad", _fake_envelope)
    return state, scale


def test_the_trust_region_is_applied_to_the_predicted_start(monkeypatch) -> None:
    """An oversized prediction must be CLIPPED before it becomes a start.

    ``‖s_anchor‖`` here is ‖(1,2,3)‖ = 3.7417, so the cap at ratio 0.1 is
    0.37417. A raw step of length 10 must arrive scaled to exactly the cap,
    with its direction intact — using the raw step instead survived the
    suite before this test.
    """

    raw = np.array([10.0, 0.0, 0.0])
    state, _scale = _predicting_state(
        monkeypatch, raw_delta=raw, bare_grad_l2=5.0, predicted_grad_l2=1.0
    )
    from simsopt_jax_adapters.geo.nested_ls_reduced_scale import (
        _predicted_inner_start,
    )

    surface, arm, raw_l2, applied_l2, scaled = _predicted_inner_start(
        state, np.array([0.3, -0.45])
    )
    cap = 0.1 * float(np.linalg.norm(state.anchor.surface_dofs))
    # The mixed term is evaluated at the ANCHOR, the envelope comparison at
    # the TRIAL, in that order. Evaluating the mixed term at the trial
    # linearizes about a point ``source.operator`` was not factored at, and
    # returns a plausible array rather than raising.
    installed = state.jax_boozer.installed_coils
    assert len(installed) == 2
    np.testing.assert_array_equal(installed[0], np.asarray(state.anchor.coil_dofs))
    np.testing.assert_array_equal(installed[1], np.array([0.3, -0.45]))
    assert arm == NESTED_LS_PREDICTOR_ARM_PREDICTED
    assert scaled is True
    assert raw_l2 == pytest.approx(10.0)
    assert applied_l2 == pytest.approx(cap)
    # The start is anchor + CLIPPED step, not anchor + raw step.
    np.testing.assert_allclose(
        surface, np.asarray(state.anchor.surface_dofs) + np.array([cap, 0.0, 0.0])
    )
    assert not np.allclose(surface, np.asarray(state.anchor.surface_dofs) + raw)


def test_a_prediction_with_a_worse_envelope_gradient_is_discarded(
    monkeypatch,
) -> None:
    """The arm rule decides, and losing means the bare anchor's own bytes."""

    state, _scale = _predicting_state(
        monkeypatch,
        raw_delta=np.array([0.01, 0.0, 0.0]),
        bare_grad_l2=1.0,
        predicted_grad_l2=2.0,  # strictly worse
    )
    from simsopt_jax_adapters.geo.nested_ls_reduced_scale import (
        _predicted_inner_start,
    )

    surface, arm, raw_l2, _applied, _scaled = _predicted_inner_start(
        state, np.array([0.3, -0.45])
    )
    assert arm == NESTED_LS_PREDICTOR_ARM_BARE
    assert surface is state.anchor.surface_dofs
    # The step was still measured and reported, so a receipt can show that a
    # prediction was made and rejected rather than never attempted.
    assert raw_l2 == pytest.approx(0.01)


def test_a_prediction_with_a_better_envelope_gradient_is_kept(monkeypatch) -> None:
    state, _scale = _predicting_state(
        monkeypatch,
        raw_delta=np.array([0.01, 0.0, 0.0]),
        bare_grad_l2=2.0,
        predicted_grad_l2=1.0,
    )
    from simsopt_jax_adapters.geo.nested_ls_reduced_scale import (
        _predicted_inner_start,
    )

    surface, arm, _raw, applied_l2, scaled = _predicted_inner_start(
        state, np.array([0.3, -0.45])
    )
    assert arm == NESTED_LS_PREDICTOR_ARM_PREDICTED
    assert scaled is False
    assert applied_l2 == pytest.approx(0.01)
    np.testing.assert_allclose(
        surface, np.asarray(state.anchor.surface_dofs) + np.array([0.01, 0.0, 0.0])
    )


# --------------------------------------------------------------------------
# Sealed inner-lane policy (Phase 5)
# --------------------------------------------------------------------------


def _policy(**overrides: object) -> dict:
    fields: dict[str, object] = {
        "ift_stab": 0.0,
        "inner_substep_legs": (1,),
        "inner_predictor": False,
    }
    fields.update(overrides)
    return nested_ls_inner_policy(**fields)  # type: ignore[arg-type]


def test_the_stock_lane_declares_itself_stock() -> None:
    policy = _policy()
    assert policy["trajectory_is_stock"] is True
    assert policy["inner_substep_enabled"] is False
    assert policy["inner_predictor_enabled"] is False
    assert policy["policy"] == NESTED_LS_INNER_POLICY_NAME


@pytest.mark.parametrize(
    "overrides",
    [
        {"inner_substep_legs": NESTED_LS_INNER_SUBSTEP_LEGS},
        {"inner_predictor": True},
        {"inner_substep_legs": NESTED_LS_INNER_SUBSTEP_LEGS, "inner_predictor": True},
    ],
)
def test_every_non_stock_lane_refuses_to_look_stock(overrides: dict) -> None:
    """The whole reason the block exists.

    Before it, a run with either lever enabled published a policy
    byte-identical to a run without them -- two different optimizations, one
    receipt shape, and a consumer free to compare numbers that must not be
    compared. Each lever alone, and both together, must move
    ``trajectory_is_stock`` off True.
    """

    policy = _policy(**overrides)
    assert policy["trajectory_is_stock"] is False
    assert "NON-STOCK" in str(policy["comparability"])


def test_the_stock_and_non_stock_blocks_are_not_equal() -> None:
    """Stated as the comparison a consumer would actually make."""

    assert _policy() != _policy(inner_predictor=True)
    assert _policy() != _policy(inner_substep_legs=NESTED_LS_INNER_SUBSTEP_LEGS)


def test_the_trust_region_ratio_appears_only_when_the_predictor_runs() -> None:
    """A cap reported for a lane that never capped anything is noise."""

    assert _policy()["predictor_trust_region_ratio"] is None
    assert _policy(inner_predictor=True)["predictor_trust_region_ratio"] == (
        NESTED_LS_PREDICTOR_TRUST_REGION_RATIO
    )


def test_the_coarse_tier_is_reported_licensed_but_unclaimed() -> None:
    """``None`` is the honest answer today, not a placeholder.

    Phase 4 licensed a coarse tier; no consumer honours it, because
    ``success`` stays true only for ``converged``. The block says the
    licence exists and is unexercised, and a run that starts honouring it
    has to say so here -- at which point receipts on either side of the
    change stop being comparable.
    """

    policy = _policy()
    assert policy["coarse_tier_licensed_residual"] == NESTED_LS_NEWTON_COARSE_TOL
    assert policy["coarse_tier_honoured"] is None
    assert _policy(coarse_tier_honoured="outer_gradient")["coarse_tier_honoured"] == (
        "outer_gradient"
    )


def test_the_block_records_values_not_a_preset_name() -> None:
    """A preset name is only as honest as the reader's copy of its meaning."""

    policy = _policy(ift_stab=1.0e-4)
    assert policy["ift_stab"] == 1.0e-4
    assert policy["newton_maxiter"] == NESTED_LS_NEWTON_MAXITER
    assert policy["newton_tol"] == NESTED_LS_NEWTON_TOL
    assert policy["iota_branch_guard"] == NESTED_LS_OUTER_IOTA_BRANCH_GUARD


def test_both_child_schemas_moved_for_the_block() -> None:
    """One bump, not three, and it had to move rather than stay additive.

    An old consumer reading a new receipt would still see every field it
    knows, and would still draw the wrong comparison the block exists to
    prevent. Only the version stops it.
    """

    assert NESTED_LS_OUTER_JAX_CHILD_SCHEMA.endswith(".v6")
    assert NESTED_LS_OUTER_NATIVE_CHILD_SCHEMA.endswith(".v5")
