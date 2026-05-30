"""Regression tests for the opt-in field-line RHS-evaluation budget.

The Greene-residue probe integrates field lines with an adaptive ``solve_ivp``.
In a near-chaotic / flat-shear region (e.g. the low-iota box corner of the HBT
single-stage lineage) the phi-parametrised ODE forces the integrator into an
enormous number of expensive full Biot-Savart RHS evaluations, so a single
probe can run effectively unbounded. ``FieldlineIntegratorOptions.max_rhs_evaluations``
converts that hang into a deterministic, fast, *gated* failure.

These tests lock the budget contract:
  * default ``None`` is a no-op (byte-identical legacy behaviour),
  * a tight budget raises ``FieldlineEvaluationBudgetError`` mid-integration,
  * the error subclasses ``RuntimeError`` so the existing seed-ranking and
    periodic-orbit-discovery handlers gate it as ``integration_failed`` with no
    caller changes,
  * the option validates its input.
"""

from __future__ import annotations

import numpy as np
import pytest

from examples.single_stage_optimization.banana_opt.topology.fieldline_map import (
    FieldlineEvaluationBudgetError,
    FieldlineIntegratorOptions,
    _budgeted_rhs,
    integrate_full_torus_return_map,
)
from examples.single_stage_optimization.banana_opt.topology.poincare_chart import (
    PoincareChart,
)


class _ConstantToroidalField:
    """Minimal analytic, well-behaved field: purely toroidal (|B_phi|/|B| = 1).

    A field line launched at (R, Z) stays there, so the phi-integration never
    stalls — any non-termination in a test using this field is the budget, not
    the field.
    """

    def __init__(self) -> None:
        self.points = np.zeros((1, 3), dtype=float)

    def set_points(self, points: np.ndarray) -> "_ConstantToroidalField":
        self.points = np.asarray(points, dtype=float)
        return self

    def B(self) -> np.ndarray:
        x = self.points[:, 0]
        y = self.points[:, 1]
        radius = np.sqrt(x**2 + y**2)
        cos_phi = x / radius
        sin_phi = y / radius
        # B = e_phi = (-sin phi, cos phi, 0): unit toroidal field.
        return np.stack([-sin_phi, cos_phi, np.zeros_like(radius)], axis=-1)


def test_budgeted_rhs_none_is_passthrough() -> None:
    def rhs(phi, state):
        return state

    assert _budgeted_rhs(rhs, None) is rhs


def test_budgeted_rhs_raises_after_budget() -> None:
    calls = {"n": 0}

    def rhs(phi, state):
        calls["n"] += 1
        return state

    wrapped = _budgeted_rhs(rhs, 3)
    state = np.zeros(2)
    # First three calls are allowed; the fourth trips the budget.
    for _ in range(3):
        wrapped(0.0, state)
    with pytest.raises(FieldlineEvaluationBudgetError):
        wrapped(0.0, state)
    assert calls["n"] == 3  # the tripping call does not reach the wrapped rhs


def test_integration_completes_with_no_budget() -> None:
    field = _ConstantToroidalField()
    chart = PoincareChart(axis_r=1.0, axis_z=0.0)
    options = FieldlineIntegratorOptions()  # max_rhs_evaluations is None by default
    result = integrate_full_torus_return_map(
        field,
        (1.05, 0.0),
        chart=chart,
        torus_turns=1,
        options=options,
    )
    # Purely toroidal field => the line returns to its launch point.
    assert result.final_state == pytest.approx((1.05, 0.0), abs=1.0e-6)


def test_integration_trips_tight_budget() -> None:
    field = _ConstantToroidalField()
    chart = PoincareChart(axis_r=1.0, axis_z=0.0)
    options = FieldlineIntegratorOptions(max_rhs_evaluations=5)
    with pytest.raises(FieldlineEvaluationBudgetError):
        integrate_full_torus_return_map(
            field,
            (1.05, 0.0),
            chart=chart,
            torus_turns=1,
            options=options,
        )


def test_budget_error_is_runtime_error_for_gating() -> None:
    # Existing handlers (residue_diagnostics seed ranking, periodic-orbit
    # discovery) catch RuntimeError; subclassing it is what lets the budget
    # surface as a gated integration failure with no caller changes.
    assert issubclass(FieldlineEvaluationBudgetError, RuntimeError)


@pytest.mark.parametrize("bad_value", [0, -1, -1000])
def test_options_reject_nonpositive_budget(bad_value: int) -> None:
    with pytest.raises(ValueError):
        FieldlineIntegratorOptions(max_rhs_evaluations=bad_value)


def test_options_accept_positive_budget_and_none() -> None:
    assert FieldlineIntegratorOptions(max_rhs_evaluations=10).max_rhs_evaluations == 10
    assert FieldlineIntegratorOptions().max_rhs_evaluations is None
