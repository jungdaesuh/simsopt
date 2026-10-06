"""Termination labels, the box second-order screen and the termination-record validator.

Test groups: L1 emitter tables, L2 the status-7 split, L4/L4a-L4g the second-order screen, L5/L5a precedence, L6
stagnation, T9 the fallback-label cases and L8 the Hessian-stability decision. Every problem is analytic; nothing
here runs a solver.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import dataclasses
import math

import numpy as np
import pytest
from simsopt_contracts.optimization_endpoint import (
    TERMINAL_STATIONARITY_ATOL,
    stopping_reason_for_status,
)
from simsopt_jax.solve import (
    SCIPY_LBFGSB_UNRESOLVED_STALL_STATUS,
    LbfgsbRestartReason,
    ScipyBounds,
)
from simsopt_jax.solve import dispatch
from simsopt_jax.solve.termination import (
    BOUND_ACTIVITY_RTOL,
    STAGNATION_RTOL,
    STAGNATION_WINDOW,
    Emitter,
    EmitterStop,
    NotApplicable,
    SecondOrderVerdict,
    TerminationLabel,
    classify_coordinates,
    projected_gradient_inf_norm,
    second_order_screen,
    stagnation,
    termination_record_defects,
    termination_report,
    verify_termination_evidence,
)

TOL = TERMINAL_STATIONARITY_ATOL
INF = math.inf


def _box(lower, upper) -> ScipyBounds:
    return ScipyBounds(lower=tuple(lower), upper=tuple(upper))


class _EigenSpy:
    """Records the shape of every matrix handed to numpy's symmetric eigen routines."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.shapes: list[tuple[str, tuple[int, ...]]] = []
        eigh, eigvalsh = np.linalg.eigh, np.linalg.eigvalsh

        def spied_eigh(matrix, *args, **kwargs):
            self.shapes.append(("eigh", np.shape(matrix)))
            return eigh(matrix, *args, **kwargs)

        def spied_eigvalsh(matrix, *args, **kwargs):
            self.shapes.append(("eigvalsh", np.shape(matrix)))
            return eigvalsh(matrix, *args, **kwargs)

        monkeypatch.setattr(np.linalg, "eigh", spied_eigh)
        monkeypatch.setattr(np.linalg, "eigvalsh", spied_eigvalsh)

    @property
    def empty_calls(self) -> list[tuple[str, tuple[int, ...]]]:
        return [call for call in self.shapes if 0 in call[1]]


# ---------------------------------------------------------------------------
# Projected gradient: one owner, SciPy's projgr.


def test_projected_gradient_is_scipy_projgr_and_equals_the_d4_definition() -> None:
    """``projgr`` clips by the distance to the bound the descent step moves toward.

    D4 wrote the same norm as ``|P[l,u](x - g) - x|_inf``; the two agree for any feasible x.
    """
    rng = np.random.default_rng(20260924)
    for _ in range(200):
        n = int(rng.integers(1, 7))
        lower = rng.uniform(-2.0, 0.0, n)
        upper = lower + rng.uniform(0.0, 3.0, n)
        x = lower + rng.uniform(0.0, 1.0, n) * (upper - lower)
        on_bound = rng.integers(0, 3, n)
        x = np.where(on_bound == 1, lower, np.where(on_bound == 2, upper, x))
        gradient = rng.normal(0.0, 2.0, n)
        box = _box(lower, upper)

        pg = projected_gradient_inf_norm(x, gradient, box)

        d4 = float(np.max(np.abs(np.clip(x - gradient, lower, upper) - x)))
        assert pg == pytest.approx(d4, rel=0.0, abs=4.0 * np.finfo(float).eps * 8.0)
    assert projected_gradient_inf_norm(np.zeros(2), np.array([3.0, -4.0]), None) == 4.0


def test_dispatch_uses_the_single_projected_gradient_owner() -> None:
    assert not hasattr(dispatch, "_projected_grad_norm_inf")
    assert dispatch.projected_gradient_inf_norm is projected_gradient_inf_norm


# ---------------------------------------------------------------------------
# L4: the second-order screen.


def test_l4_one_weak_bound_with_an_inward_negative_direction_is_an_exact_saddle() -> (
    None
):
    screen = second_order_screen(
        np.diag([-1.0, 1.0]), np.zeros(2), np.zeros(2), _box([0.0, -INF], [INF, INF])
    )

    assert screen.verdict is SecondOrderVerdict.SADDLE
    assert screen.cone == "exact"
    assert screen.witness == (1.0, 0.0)
    assert screen.lambda_min == -1.0


def test_l4_two_weak_bounds_example_is_inconclusive_not_saddle() -> None:
    """``0.5x^2 + 2xy + 0.5y^2`` on x, y >= 0 at 0: a strict constrained minimum."""
    screen = second_order_screen(
        np.array([[1.0, 2.0], [2.0, 1.0]]),
        np.zeros(2),
        np.zeros(2),
        _box([0.0, 0.0], [INF, INF]),
    )

    assert screen.verdict is SecondOrderVerdict.INCONCLUSIVE
    assert screen.cone == "candidate"
    assert screen.witness is None


def test_l4_strictly_free_positive_definite_passes_exactly() -> None:
    screen = second_order_screen(np.diag([2.0, 3.0]), np.ones(2), np.zeros(2), None)

    assert screen.verdict is SecondOrderVerdict.PASS
    assert screen.cone == "exact"


def test_l4_negative_curvature_only_along_a_strong_bound_passes() -> None:
    screen = second_order_screen(
        np.diag([-1.0, 1.0]),
        np.zeros(2),
        np.array([1.0, 0.0]),
        _box([0.0, -INF], [INF, INF]),
    )

    assert screen.classes.strong == (0,)
    assert screen.verdict is SecondOrderVerdict.PASS


@pytest.mark.parametrize(
    ("epsilon", "verdict", "cone", "witness"),
    [
        (TOL / 2.0, SecondOrderVerdict.INCONCLUSIVE, "candidate", None),
        (0.0, SecondOrderVerdict.SADDLE, "exact", (1.0, 0.0)),
        (2.0 * TOL, SecondOrderVerdict.PASS, "exact", None),
    ],
    ids=["ambiguous", "weak", "strong"],
)
def test_l4a_epsilon_counterexample(epsilon, verdict, cone, witness) -> None:
    """``f = eps x - x^2/2 + y^2/2``, x >= 0, at (0, 0)."""
    screen = second_order_screen(
        np.diag([-1.0, 1.0]),
        np.zeros(2),
        np.array([epsilon, 0.0]),
        _box([0.0, -INF], [INF, INF]),
    )

    assert screen.verdict is verdict
    assert screen.cone == cone
    assert screen.witness == witness
    if verdict is SecondOrderVerdict.PASS:
        assert screen.classes.strong == (0,), "PASS is decided on y alone"
        assert screen.lambda_min == 1.0


def test_l4b_near_bound_but_not_bitwise_is_ambiguous_and_never_moved_by_a_witness() -> (
    None
):
    x = np.array([1e-13, 0.0])
    assert 1e-13 <= BOUND_ACTIVITY_RTOL
    screen = second_order_screen(
        np.diag([-1.0, 1.0]), x, np.zeros(2), _box([0.0, -INF], [INF, INF])
    )

    assert screen.classes.ambiguous == (0,)
    assert screen.classes.on_lower == ()
    assert screen.verdict is SecondOrderVerdict.INCONCLUSIVE
    assert screen.witness is None


def test_l4c_a_fixed_coordinate_takes_no_part() -> None:
    fixed_box = _box([0.0, -INF], [0.0, INF])

    passing = second_order_screen(
        np.diag([-1.0, 1.0]), np.zeros(2), np.zeros(2), fixed_box
    )
    saddle = second_order_screen(
        np.diag([-1.0, -2.0]), np.zeros(2), np.zeros(2), fixed_box
    )

    assert passing.classes.fixed == (0,)
    assert passing.verdict is SecondOrderVerdict.PASS
    assert saddle.verdict is SecondOrderVerdict.SADDLE
    assert saddle.witness is not None and saddle.witness[0] == 0.0
    assert abs(saddle.witness[1]) == 1.0


@pytest.mark.parametrize(
    ("box", "gradient", "fixed", "strong"),
    [
        (_box([0.0, 1.0], [0.0, 1.0]), np.zeros(2), (0, 1), ()),
        (_box([0.0, 0.0], [INF, 1.0]), np.array([1.0, -1.0]), (), (0, 1)),
    ],
    ids=["all-fixed", "all-strong"],
)
def test_l4d_empty_movable_set_is_a_vacuous_pass_with_no_eigen_call(
    monkeypatch, box, gradient, fixed, strong
) -> None:
    spy = _EigenSpy(monkeypatch)
    x = np.array([box.lower[0], box.upper[1]]) if strong else np.array([0.0, 1.0])

    screen = second_order_screen(np.diag([-1.0, -1.0]), x, gradient, box)

    assert spy.shapes == []
    assert (screen.classes.fixed, screen.classes.strong) == (fixed, strong)
    assert screen.verdict is SecondOrderVerdict.PASS
    assert screen.lambda_min == NotApplicable("no_movable_directions")
    assert screen.tau_h == NotApplicable("no_movable_directions")
    assert screen.witness is None


def test_l4d_empty_free_set_skips_the_free_candidate_and_uses_the_weak_one(
    monkeypatch,
) -> None:
    spy = _EigenSpy(monkeypatch)

    screen = second_order_screen(
        np.array([[-1.0]]), np.zeros(1), np.zeros(1), _box([0.0], [INF])
    )

    assert spy.empty_calls == []
    assert spy.shapes == [("eigvalsh", (1, 1)), ("eigh", (1, 1))]
    assert screen.verdict is SecondOrderVerdict.SADDLE
    assert screen.witness == (1.0,)


def test_l4d_prime_only_ambiguous_coordinates_is_inconclusive_without_empty_eigen_calls(
    monkeypatch,
) -> None:
    """1-D ``f = eps x - x^2/2``, x >= 0, at 0, eps = gtol/2: M = Z = {x}; both witness sets are empty."""
    spy = _EigenSpy(monkeypatch)

    screen = second_order_screen(
        np.array([[-1.0]]), np.zeros(1), np.array([TOL / 2.0]), _box([0.0], [INF])
    )

    assert screen.classes.ambiguous == (0,)
    assert spy.empty_calls == []
    assert [name for name, _shape in spy.shapes] == ["eigvalsh"]
    assert screen.verdict is SecondOrderVerdict.INCONCLUSIVE
    assert screen.witness is None


def test_l4e_witnesses_are_embedded_in_full_coordinates() -> None:
    """Two weak lower bounds whose joint negative eigenvector leaves the box in both signs, plus a free
    coordinate with its own negative curvature: the F candidate is the witness, embedded with zeros."""
    hessian = np.array(
        [
            [1.0, 2.0, 0.0, 0.0, 0.0],
            [2.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, -3.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -3.0],
        ]
    )
    x = np.array([0.0, 0.0, 0.25, 2.0, 0.0])
    gradient = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    box = _box([0.0, 0.0, -1.0, 2.0, 0.0], [INF, INF, 1.0, 2.0, INF])

    screen = second_order_screen(hessian, x, gradient, box)

    assert screen.classes.fixed == (3,)
    assert screen.classes.strong == (4,)
    assert screen.classes.weak == (0, 1)
    assert screen.classes.free == (2,)
    assert screen.verdict is SecondOrderVerdict.SADDLE
    witness = np.asarray(screen.witness)
    assert witness.shape == (5,)
    assert np.array_equal(witness[[0, 1, 3, 4]], np.zeros(4))
    assert abs(witness[2]) == 1.0
    assert float(gradient @ witness) <= 0.0
    assert float(witness @ hessian @ witness) < -screen.tau_h


def test_l4f_narrow_box_inward_gradients_give_no_out_of_box_witness() -> None:
    """t = gtol, box [0, t/2]^2, x = 0, g = (-2t, -2t), H eigenvalues +-1."""
    t = TOL
    box = _box([0.0, 0.0], [t / 2.0, t / 2.0])
    gradient = np.array([-2.0 * t, -2.0 * t])
    hessian = np.array([[0.6, 0.8], [0.8, -0.6]])

    assert projected_gradient_inf_norm(np.zeros(2), gradient, box) == t / 2.0
    screen = second_order_screen(hessian, np.zeros(2), gradient, box)

    assert screen.classes.inward == (0, 1)
    assert screen.verdict is SecondOrderVerdict.INCONCLUSIVE
    assert screen.witness is None
    record = screen.to_record()
    assert record["inward_bound_coordinates"] == {"count": 2, "indices": [0, 1]}


def _in_tangent_cone(direction, x, lower, upper) -> bool:
    for i, component in enumerate(direction):
        if lower[i] == upper[i] and component != 0.0:
            return False
        if x[i] == lower[i] and component < 0.0:
            return False
        if x[i] == upper[i] and component > 0.0:
            return False
    return True


def test_l4f_property_every_emitted_witness_is_feasible_and_first_order_compatible() -> (
    None
):
    rng = np.random.default_rng(424242)
    saddles = 0
    for _ in range(3000):
        n = int(rng.integers(1, 6))
        lower = rng.uniform(-1.0, 0.0, n)
        upper = lower + rng.choice([0.0, 1e-8, 0.5, 2.0, INF], n)
        position = rng.integers(0, 4, n)
        interior = lower + 0.5 * np.where(np.isfinite(upper), upper - lower, 1.0)
        x = np.where(
            position == 0,
            lower,
            np.where(
                position == 1, np.where(np.isfinite(upper), upper, lower), interior
            ),
        )
        x = np.where(position == 3, lower + 1e-13, x)
        x = np.minimum(x, upper)
        gradient = rng.choice(
            [0.0, TOL / 2.0, -TOL / 2.0, 3.0 * TOL, -3.0 * TOL, 1.0, -1.0], n
        )
        gradient = gradient * rng.choice([1.0, 0.0], n, p=[0.8, 0.2])
        root = rng.normal(size=(n, n))
        hessian = 0.5 * (root + root.T)
        box = _box(lower, upper)

        screen = second_order_screen(hessian, x, gradient, box)

        if screen.verdict is SecondOrderVerdict.SADDLE:
            saddles += 1
            witness = np.asarray(screen.witness)
            assert witness.shape == (n,)
            assert _in_tangent_cone(witness, x, lower, upper)
            assert float(gradient @ witness) <= 0.0
            assert float(witness @ hessian @ witness) < -screen.tau_h * float(
                witness @ witness
            )
            for index in (
                *screen.classes.ambiguous,
                *screen.classes.strong,
                *screen.classes.fixed,
            ):
                assert witness[index] == 0.0
        else:
            assert screen.witness is None
    assert saddles > 300, "the property sample must exercise the witness path"


def test_screen_rejects_a_point_outside_the_box() -> None:
    with pytest.raises(ValueError, match="outside"):
        second_order_screen(
            np.eye(1), np.array([-1e-300]), np.zeros(1), _box([0.0], [1.0])
        )


def test_classes_partition_every_coordinate() -> None:
    classes = classify_coordinates(
        np.array([0.0, 1.0, 0.5, 1e-13, 2.0, 0.0]),
        np.array([1.0, 1.0, 0.0, 0.0, 0.0, TOL / 4.0]),
        _box([0.0, 0.0, 0.0, 0.0, 2.0, 0.0], [1.0, 1.0, 1.0, 1.0, 2.0, 1.0]),
    )

    assert classes.strong == (0,)
    assert classes.inward == (1,)
    assert classes.free == (2,)
    assert classes.ambiguous == (3, 5)
    assert classes.fixed == (4,)
    assert classes.on_lower == (0, 5)
    assert classes.on_upper == (1,)
    assert classes.movable == (1, 2, 3, 5)


# ---------------------------------------------------------------------------
# Labels: L1 emitter tables, L2 the status-7 split, L5/L5a precedence, T9.

_X = np.array([0.5, -0.25])
_BOX = _box([0.0, -1.0], [1.0, 1.0])
_PD = np.diag([2.0, 1.0])
_STATIONARY = np.array([1e-9, -2e-9])
_NOT_STATIONARY = np.array([1e-3, 0.0])


def _label(
    emitter: Emitter,
    status: int,
    *,
    gradient=_STATIONARY,
    success: bool = False,
    restart: LbfgsbRestartReason | None = None,
    hessian=_PD,
    history=(1.0, 0.5),
    x=_X,
    fun: float = 0.25,
):
    return termination_report(
        EmitterStop(
            emitter=emitter,
            status=status,
            message=f"message {status}",
            success=success,
            last_restart_reason=restart,
        ),
        x=np.asarray(x),
        fun=fun,
        gradient=np.asarray(gradient),
        bounds=_BOX,
        accepted_fun=history,
        hessian=hessian,
    )


_L = TerminationLabel
_TABLE_CASES = [
    # (emitter, status, restart reason, gradient, expected label)
    (Emitter.SCIPY_LBFGSB, 0, None, _STATIONARY, _L.CONVERGED),
    (Emitter.SCIPY_LBFGSB, 0, None, _NOT_STATIONARY, _L.OWN_STOP_NOT_STATIONARY),
    (Emitter.SCIPY_LBFGSB, 1, None, _NOT_STATIONARY, _L.BUDGET_EXHAUSTED),
    (Emitter.SCIPY_LBFGSB, 1, None, _STATIONARY, _L.BUDGET_EXHAUSTED),
    (Emitter.SCIPY_LBFGSB, 2, None, _STATIONARY, _L.LINE_SEARCH_FAILED),
    (
        Emitter.SCIPY_LBFGSB,
        7,
        LbfgsbRestartReason.BUDGET_EXHAUSTED,
        _STATIONARY,
        _L.BUDGET_EXHAUSTED,
    ),
    (
        Emitter.SCIPY_LBFGSB,
        7,
        LbfgsbRestartReason.FRESH_MEMORY_STALL,
        _NOT_STATIONARY,
        _L.UNRESOLVED_STALL,
    ),
    (Emitter.SCIPY_LBFGSB, 7, None, _STATIONARY, _L.FAILED),
    (Emitter.SCIPY_LBFGSB, 99, None, _STATIONARY, _L.FAILED),
    (Emitter.SCIPY_TRUST_CONSTR, 1, None, _STATIONARY, _L.CONVERGED),
    (Emitter.SCIPY_TRUST_CONSTR, 2, None, _STATIONARY, _L.CONVERGED),
    (Emitter.SCIPY_TRUST_CONSTR, 2, None, _NOT_STATIONARY, _L.OWN_STOP_NOT_STATIONARY),
    (Emitter.SCIPY_TRUST_CONSTR, 0, None, _STATIONARY, _L.BUDGET_EXHAUSTED),
    (Emitter.SCIPY_TRUST_CONSTR, 4, None, _STATIONARY, _L.INFEASIBLE),
    (Emitter.SCIPY_TRUST_CONSTR, 3, None, _STATIONARY, _L.FAILED),
    (Emitter.SCIPY_TRUST_EXACT, 0, None, _STATIONARY, _L.CONVERGED),
    (Emitter.SCIPY_TRUST_EXACT, 0, None, _NOT_STATIONARY, _L.OWN_STOP_NOT_STATIONARY),
    (Emitter.SCIPY_TRUST_EXACT, 1, None, _STATIONARY, _L.BUDGET_EXHAUSTED),
    (Emitter.SCIPY_TRUST_EXACT, 2, None, _STATIONARY, _L.FAILED),
    (Emitter.SCIPY_TRUST_EXACT, 3, None, _STATIONARY, _L.FAILED),
]


@pytest.mark.parametrize(
    ("emitter", "status", "restart", "gradient", "expected"), _TABLE_CASES
)
def test_l1_l2_emitter_tables(emitter, status, restart, gradient, expected) -> None:
    for success in (False, True):
        report = _label(
            emitter, status, gradient=gradient, success=success, restart=restart
        )

        assert report.label is expected, "the emitter's success flag is never read"
        assert report.emitter_stop.status == status
        assert report.emitter_stop.message == f"message {status}"
        assert report.emitter_stop.success is success
        assert termination_record_defects(report.to_record()) == ()


def test_l1_lbfgsb_table_agrees_with_the_endpoint_contract() -> None:
    """The endpoint contract owns SciPy L-BFGS-B's integer meanings; the label table must not drift from it."""
    by_reason = {
        "iteration-limit": _L.BUDGET_EXHAUSTED,
        "evaluation-limit": _L.BUDGET_EXHAUSTED,
        "line-search-failed": _L.LINE_SEARCH_FAILED,
        "converged": None,
    }
    for status in (0, 1, 2):
        reason = stopping_reason_for_status(
            status_convention="scipy-lbfgsb",
            provider_success=status == 0,
            provider_status=status,
            finite=True,
        )
        label = _label(Emitter.SCIPY_LBFGSB, status, gradient=_NOT_STATIONARY).label
        expected = by_reason[reason]
        assert label is (_L.OWN_STOP_NOT_STATIONARY if expected is None else expected)


def test_status_seven_is_the_restart_route_status() -> None:
    assert SCIPY_LBFGSB_UNRESOLVED_STALL_STATUS == 7


def test_l5_one_ulp_outside_the_original_bound_is_infeasible_even_at_a_cap_with_zero_pg() -> (
    None
):
    x = np.array([np.nextafter(1.0, 2.0), 0.0])

    report = _label(Emitter.SCIPY_LBFGSB, 1, gradient=np.zeros(2), x=x)

    assert report.label is _L.INFEASIBLE
    assert report.max_bound_excursion == np.nextafter(1.0, 2.0) - 1.0
    assert report.second_order is None
    assert termination_record_defects(report.to_record()) == ()


def test_l5_precedence_order() -> None:
    outside = np.array([1.5, 0.0])
    nan_gradient = np.array([np.nan, 0.0])

    assert (
        _label(Emitter.SCIPY_LBFGSB, 0, gradient=nan_gradient, x=outside).label
        is _L.NONFINITE
    )
    assert _label(Emitter.SCIPY_LBFGSB, 0, x=outside).label is _L.INFEASIBLE
    assert _label(Emitter.SCIPY_LBFGSB, 0, fun=math.inf).label is _L.NONFINITE
    assert (
        _label(Emitter.SCIPY_LBFGSB, 0, hessian=np.full((2, 2), np.nan)).label
        is _L.NONFINITE
    )
    assert _label(Emitter.SCIPY_LBFGSB, 0).label is _L.CONVERGED
    assert (
        _label(Emitter.SCIPY_LBFGSB, 0, hessian=np.diag([-1.0, 1.0])).label is _L.SADDLE
    )
    assert (
        _label(Emitter.SCIPY_LBFGSB, 0, hessian=None).label
        is _L.CONVERGED_FIRST_ORDER_ONLY
    )
    nan_report = _label(Emitter.SCIPY_LBFGSB, 0, gradient=nan_gradient)
    assert nan_report.nonfinite_fields == ("gradient",)
    assert nan_report.projected_grad_norm_inf is None
    assert termination_record_defects(nan_report.to_record()) == ()


def test_l5_second_order_inconclusive_at_an_own_stop() -> None:
    """Two weak lower bounds at the corner of the box with an indefinite coupling."""
    report = termination_report(
        EmitterStop(Emitter.SCIPY_LBFGSB, 0, "CONVERGENCE", True, None),
        x=np.zeros(2),
        fun=0.0,
        gradient=np.zeros(2),
        bounds=_box([0.0, 0.0], [1.0, 1.0]),
        accepted_fun=(0.0,),
        hessian=np.array([[1.0, 2.0], [2.0, 1.0]]),
    )

    assert report.label is _L.SECOND_ORDER_INCONCLUSIVE


_CAPS = [
    (Emitter.SCIPY_LBFGSB, 1, None),
    (Emitter.SCIPY_LBFGSB, 7, LbfgsbRestartReason.BUDGET_EXHAUSTED),
    (Emitter.SCIPY_TRUST_CONSTR, 0, None),
    (Emitter.SCIPY_TRUST_EXACT, 1, None),
]


@pytest.mark.parametrize(("emitter", "status", "restart"), _CAPS)
def test_l5a_a_cap_outranks_stagnation_and_small_pg(emitter, status, restart) -> None:
    flat = tuple(1.0 for _ in range(STAGNATION_WINDOW + 1))
    for gradient in (_STATIONARY, _NOT_STATIONARY):
        report = _label(
            emitter, status, gradient=gradient, restart=restart, history=flat
        )

        assert report.label is _L.BUDGET_EXHAUSTED
        assert report.stagnation.stagnated is True
        assert report.to_record()["stagnated"] is True


def test_l6_stagnation_threshold_over_the_window() -> None:
    def history(last_drop: float) -> tuple[float, ...]:
        values = [2.0] * STAGNATION_WINDOW + [2.0 - last_drop]
        return (7.0, *values)

    below = stagnation(history(0.9 * STAGNATION_RTOL * 2.0))
    above = stagnation(history(1.1 * STAGNATION_RTOL * 2.0))
    short = stagnation(tuple([1.0] * STAGNATION_WINDOW))

    assert below.stagnated is True
    assert above.stagnated is False
    assert below.window_rel_drop == pytest.approx(0.9 * STAGNATION_RTOL, rel=1e-6)
    assert above.window_rel_drop == pytest.approx(1.1 * STAGNATION_RTOL, rel=1e-6)
    assert short.stagnated is False
    assert short.window_rel_drop == NotApplicable("history_shorter_than_window")
    assert stagnation((0.0,) * (STAGNATION_WINDOW + 1)).window_rel_drop == 0.0
    nonfinite = stagnation((*[1.0] * STAGNATION_WINDOW, math.nan))
    assert nonfinite.stagnated is False
    assert nonfinite.window_rel_drop == NotApplicable("nonfinite_history")


def test_t9_fallback_labels_on_synthetic_results() -> None:
    """T9: ftol stop at pg 1e-3; cap; unresolved stall; ftol stop at pg 1e-9; a stagnant history."""
    flat = tuple(1.0 for _ in range(STAGNATION_WINDOW + 1))

    assert (
        _label(Emitter.SCIPY_LBFGSB, 0, gradient=_NOT_STATIONARY).label
        is _L.OWN_STOP_NOT_STATIONARY
    )
    assert _label(Emitter.SCIPY_LBFGSB, 1).label is _L.BUDGET_EXHAUSTED
    assert (
        _label(
            Emitter.SCIPY_LBFGSB, 7, restart=LbfgsbRestartReason.FRESH_MEMORY_STALL
        ).label
        is _L.UNRESOLVED_STALL
    )
    assert _label(Emitter.SCIPY_LBFGSB, 0).label is _L.CONVERGED
    stagnant = _label(Emitter.SCIPY_LBFGSB, 0, gradient=_NOT_STATIONARY, history=flat)
    assert stagnant.label is _L.OWN_STOP_NOT_STATIONARY
    assert stagnant.stagnation.stagnated is True


def test_the_stationarity_threshold_is_the_endpoint_contract_constant() -> None:
    just_above = np.array([np.nextafter(TOL, 1.0), 0.0])

    assert (
        _label(Emitter.SCIPY_LBFGSB, 0, gradient=np.array([TOL, 0.0])).label
        is _L.CONVERGED
    )
    assert (
        _label(Emitter.SCIPY_LBFGSB, 0, gradient=just_above).label
        is _L.OWN_STOP_NOT_STATIONARY
    )


# ---------------------------------------------------------------------------
# L4g and the record validator.


def _vacuous_pass_report():
    return termination_report(
        EmitterStop(Emitter.SCIPY_LBFGSB, 0, "CONVERGENCE", True, None),
        x=np.array([0.0, 1.0]),
        fun=0.0,
        gradient=np.array([5.0, -5.0]),
        bounds=_box([0.0, 1.0], [0.0, 1.0]),
        accepted_fun=(1.0, 0.0),
        hessian=np.diag([-1.0, -1.0]),
    )


def test_l4g_complete_vacuous_pass_record_is_valid() -> None:
    report = _vacuous_pass_report()
    record = report.to_record()

    assert report.label is _L.CONVERGED
    assert record["second_order"]["lambda_min"] == {
        "not_applicable": "no_movable_directions"
    }
    assert record["second_order"]["tau_H"] == {
        "not_applicable": "no_movable_directions"
    }
    assert record["second_order"]["witness"] is None
    assert termination_record_defects(record) == ()


def _with_second_order(record, **changes):
    return {**record, "second_order": {**record["second_order"], **changes}}


@pytest.mark.parametrize("verdict", ["SADDLE", "INCONCLUSIVE"])
def test_l4g_tagged_fields_with_another_verdict_are_corrupt(verdict) -> None:
    record = _with_second_order(_vacuous_pass_report().to_record(), verdict=verdict)

    assert termination_record_defects(record) != ()


def test_l4g_tagged_fields_with_a_nonempty_movable_set_are_corrupt() -> None:
    record = _vacuous_pass_report().to_record()
    classes = {**record["second_order"]["classes"], "fixed": [0], "free": [1]}

    assert termination_record_defects(_with_second_order(record, classes=classes)) != ()


def test_record_validator_rejects_silently_missing_and_inconsistent_fields() -> None:
    record = _label(Emitter.SCIPY_LBFGSB, 0).to_record()
    assert termination_record_defects(record) == ()

    missing = {
        key: value for key, value in record.items() if key != "projected_grad_norm_inf"
    }
    null_pg = {**record, "projected_grad_norm_inf": None}
    null_screen = {**record, "second_order": None}
    wrong_label = {**record, "label": "BUDGET_EXHAUSTED"}
    nan_lambda = _with_second_order(record, lambda_min=math.nan)
    bad_witness = _with_second_order(record, verdict="SADDLE", witness=[-1.0, 0.0])

    for corrupt in (
        missing,
        null_pg,
        null_screen,
        wrong_label,
        nan_lambda,
        bad_witness,
    ):
        assert termination_record_defects(corrupt) != (), corrupt


def test_record_validator_accepts_the_hessian_exception_failure_schema() -> None:
    report = termination_report(
        EmitterStop(Emitter.SCIPY_LBFGSB, 0, "CONVERGENCE", True, None),
        x=_X,
        fun=0.25,
        gradient=_STATIONARY,
        bounds=_BOX,
        accepted_fun=(1.0,),
        hessian_exception="XlaRuntimeError",
    )
    record = report.to_record()

    assert report.label is _L.CONVERGED_FIRST_ORDER_ONLY
    assert record["second_order"] is None
    assert termination_record_defects(record) == ()
    assert termination_record_defects({**record, "hessian_exception": ""}) != ()


def test_report_is_immutable() -> None:
    report = _label(Emitter.SCIPY_LBFGSB, 0)

    with pytest.raises(dataclasses.FrozenInstanceError):
        report.label = _L.FAILED  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Restart metadata belongs to the SciPy L-BFGS-B restart route only.


@pytest.mark.parametrize(
    "emitter",
    [Emitter.SCIPY_TRUST_CONSTR, Emitter.SCIPY_TRUST_EXACT],
)
def test_restart_metadata_from_another_emitter_is_rejected(emitter) -> None:
    with pytest.raises(ValueError, match="restart"):
        EmitterStop(emitter, 1, "m", True, LbfgsbRestartReason.FRESH_MEMORY_STALL)
    record = _label(emitter, 0).to_record()
    mutated = {**record, "last_restart_reason": "fresh_memory_stall_not_restarted"}

    assert termination_record_defects(record) == ()
    assert termination_record_defects(mutated) != ()


# ---------------------------------------------------------------------------
# A record is authoritative only when it is the report its endpoint evidence gives.

_EVIDENCE = {
    "x": _X,
    "fun": 0.25,
    "gradient": _STATIONARY,
    "bounds": _BOX,
    "accepted_fun": (1.0, 0.5),
    "hessian": _PD,
}


def _verify(record, **changes):
    return verify_termination_evidence(record, **{**_EVIDENCE, **changes})


def test_a_record_matching_its_evidence_is_verified() -> None:
    record = _label(Emitter.SCIPY_LBFGSB, 0).to_record()

    verified = _verify(record)

    assert verified.defects == ()
    assert verified.report is not None
    assert verified.report.label is _L.CONVERGED


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"gradient": np.array([1.0, 1.0])}, "label"),
        ({"x": np.array([50.0, -0.25])}, "label"),
        ({"x": np.zeros(0), "gradient": np.zeros(0)}, "at least one"),
        ({"x": np.zeros(0)}, "dimension"),
        ({"gradient": np.zeros(0)}, "dimension"),
        ({"gradient": np.zeros(3)}, "dimension"),
        ({"hessian": np.eye(3)}, "dimension"),
        ({"bounds": _box([0.0], [1.0])}, "dimension"),
        ({"fun": math.nan}, "label"),
        ({"hessian": np.diag([-1.0, 1.0])}, "label"),
        ({"accepted_fun": tuple([1.0] * (STAGNATION_WINDOW + 1))}, "stagnated"),
    ],
    ids=[
        "stale-gradient",
        "infeasible-x",
        "empty-x-and-g",
        "empty-x",
        "empty-g",
        "long-g",
        "hessian-shape",
        "bounds-shape",
        "nan-fun",
        "stale-curvature",
        "stale-history",
    ],
)
def test_a_stale_summary_never_verifies(changes, reason) -> None:
    record = _label(Emitter.SCIPY_LBFGSB, 0).to_record()

    verified = _verify(record, **changes)

    assert verified.defects != ()
    assert any(reason in defect for defect in verified.defects), verified.defects


def test_an_incorrect_cone_kind_is_rejected() -> None:
    """A free coordinate with a nonzero gradient makes the cone a candidate cone."""
    record = _label(Emitter.SCIPY_LBFGSB, 0).to_record()
    assert record["second_order"]["cone"] == "candidate"
    exact = {**record, "second_order": {**record["second_order"], "cone": "exact"}}

    assert termination_record_defects(exact) == (), (
        "schema alone cannot see the gradient"
    )
    assert _verify(exact).defects != ()


def test_an_incorrect_saddle_witness_is_rejected() -> None:
    hessian = np.diag([-1.0, 1.0])
    report = termination_report(
        EmitterStop(Emitter.SCIPY_LBFGSB, 0, "CONVERGENCE", True, None),
        x=_X,
        fun=0.25,
        gradient=_STATIONARY,
        bounds=_BOX,
        accepted_fun=(1.0, 0.5),
        hessian=hessian,
    )
    record = report.to_record()
    assert record["second_order"]["verdict"] == "SADDLE"
    positive = {
        **record,
        "second_order": {**record["second_order"], "witness": [0.0, 1.0]},
    }

    assert termination_record_defects(positive) == (), "schema alone cannot see H"
    assert _verify(positive, hessian=hessian).defects != ()
    assert _verify(record, hessian=hessian).defects == ()


def test_the_hessian_exception_is_part_of_the_evidence() -> None:
    record = termination_report(
        EmitterStop(Emitter.SCIPY_LBFGSB, 0, "CONVERGENCE", True, None),
        x=_X,
        fun=0.25,
        gradient=_STATIONARY,
        bounds=_BOX,
        accepted_fun=(1.0, 0.5),
        hessian_exception="XlaRuntimeError",
    ).to_record()

    assert (
        _verify(record, hessian=None, hessian_exception="XlaRuntimeError").defects == ()
    )
    assert _verify(record).defects != (), (
        "a Hessian was measured, the record says it raised"
    )


# ---------------------------------------------------------------------------
# The metric the stationarity threshold measures (acceptance unchanged).


def test_narrow_box_label_is_the_clipped_projected_gradient_metric() -> None:
    """t = 1e-7, box [0, t/2], x = 0, g = -2t, H = [1], own stop: SciPy's clipped projected gradient is
    t/2 <= t, so the frozen metric labels it CONVERGED although the raw inward gradient is 2t. The label is
    finite-tolerance projected stationarity, not exact KKT stationarity; the INWARD coordinate is reported
    so a reader can see it. This pins the frozen rule; it is not a new acceptance rule."""
    t = TOL
    report = termination_report(
        EmitterStop(Emitter.SCIPY_LBFGSB, 0, "CONVERGENCE", True, None),
        x=np.zeros(1),
        fun=0.0,
        gradient=np.array([-2.0 * t]),
        bounds=_box([0.0], [t / 2.0]),
        accepted_fun=(1.0, 0.0),
        hessian=np.array([[1.0]]),
    )
    record = report.to_record()

    assert report.projected_grad_norm_inf == t / 2.0
    assert report.label is _L.CONVERGED
    assert record["second_order"]["inward_bound_coordinates"] == {
        "count": 1,
        "indices": [0],
    }


# ---------------------------------------------------------------------------
# The eigenvalue and tolerance descriptors are recomputed from the evidence.


@pytest.mark.parametrize(
    ("lambda_min", "tau_h"),
    [(-999.0, 1000.0), (1.0, 1000.0), (-999.0, None), (2e-9, None)],
    ids=["both-false", "tau-false", "lambda-false", "lambda-is-tau"],
)
def test_false_eigen_descriptors_are_rejected(lambda_min, tau_h) -> None:
    record = _label(Emitter.SCIPY_LBFGSB, 0).to_record()
    screen = record["second_order"]
    assert (screen["lambda_min"], screen["tau_H"]) == (1.0, 2e-9)
    false = {
        **record,
        "second_order": {
            **screen,
            "lambda_min": lambda_min,
            "tau_H": screen["tau_H"] if tau_h is None else tau_h,
        },
    }

    defects = _verify(false).defects

    assert any("lambda_min" in d or "tau_H" in d for d in defects), defects


def test_rounding_level_descriptor_differences_are_accepted() -> None:
    """The consistency rule: |d lambda| <= 10 n eps ||H_MM||_2, |d tau| <= 10 n eps tau."""
    record = _label(Emitter.SCIPY_LBFGSB, 0).to_record()
    screen = record["second_order"]
    nudged = {
        **record,
        "second_order": {
            **screen,
            "lambda_min": float(np.nextafter(screen["lambda_min"], 2.0)),
            "tau_H": float(np.nextafter(screen["tau_H"], 1.0)),
        },
    }

    assert _verify(nudged).defects == ()


# ---------------------------------------------------------------------------
# L8: decisions use H_sym and must be stable over H, H^T, H_sym.


def test_l8_second_order_decision_must_be_stable_under_the_skew() -> None:
    """H_sym = diag(-1e-6, 1) plus a skew of norm 2e-7: with the raw H, tau_H =
    10 ||H - H^T|| = 2e-6 hides the -1e-6 curvature (PASS); with H_sym alone tau_H =
    1e-9 exposes it (SADDLE). An unstable decision is INCONCLUSIVE, never CONVERGED."""
    skew = np.array([[0.0, 1e-7], [-1e-7, 0.0]])
    hessian = np.diag([-1e-6, 1.0]) + skew

    screen = second_order_screen(hessian, _X, np.zeros(2), _BOX)
    report = _label(Emitter.SCIPY_LBFGSB, 0, gradient=np.zeros(2), hessian=hessian)

    assert screen.decision_stable is False
    assert screen.verdict is SecondOrderVerdict.INCONCLUSIVE
    assert screen.witness is None
    assert report.label is _L.SECOND_ORDER_INCONCLUSIVE
    assert report.to_record()["second_order"]["decision_stable"] is False
    assert termination_record_defects(report.to_record()) == ()


def test_l8_a_stable_decision_is_unchanged() -> None:
    screen = second_order_screen(np.diag([2.0, 1.0]), _X, np.zeros(2), _BOX)

    assert screen.decision_stable is True
    assert screen.verdict is SecondOrderVerdict.PASS
