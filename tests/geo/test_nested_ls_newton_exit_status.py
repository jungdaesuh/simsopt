"""Three-valued inner exit for the nested Boozer-LS Schur-Newton walk.

Phase 3 of ``docs/nested_ls_upgrade_implementation_plan.md``. The walk has
always published one bit, ``success``. ``exit_status`` splits the failing
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
    NESTED_LS_NEWTON_COARSE_TOL,
    NESTED_LS_NEWTON_EXIT_COARSE_CONVERGED,
    NESTED_LS_NEWTON_EXIT_CONVERGED,
    NESTED_LS_NEWTON_EXIT_FAILED,
    NESTED_LS_NEWTON_EXIT_STATUSES,
    NESTED_LS_NEWTON_TOL,
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
