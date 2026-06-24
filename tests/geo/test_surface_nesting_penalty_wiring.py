"""Wiring tests for the soft surface-nesting penalty in the single-stage driver.

The pure barrier ``surface_stack_nesting_penalty`` is gradient-verified in
``test_surface_nesting_penalty.py``. THIS suite proves the DRIVER glue
``_apply_surface_nesting_penalty`` actually folds that barrier into the
weighted-sum evaluation the L-BFGS-B optimizer descends -- the "inert-knob" class
of bug (a term that exists but never moves the optimizer) that has slipped through
this codebase before. It drives the real driver function with the SAME margin
contract the ALM path uses, monkeypatching only the (separately-tested) margin
computation so the test stays fast and deterministic.

This closes the wiring half of the lever. The live end-to-end proof -- an enabled
weight moving a real L-BFGS-B step on a real Boozer stack -- is the bounded
penalty-mode smoke run recorded alongside the feature run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

EXAMPLES_ROOT = (
    Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
)
for _p in (str(EXAMPLES_ROOT / "SINGLE_STAGE"), str(EXAMPLES_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import single_stage_banana_example as drv  # noqa: E402
from banana_opt.single_stage_constraints import (  # noqa: E402
    surface_stack_nesting_penalty,
)


def _fake_surface_data(n):
    # The helper only reads entry["boozer_surface"].surface and entry["boozer_surface"];
    # the (monkeypatched) margin fn ignores the geometry, so sentinels suffice.
    return [{"boozer_surface": SimpleNamespace(surface=f"surf{i}")} for i in range(n)]


def _install(monkeypatch, *, weight, clearance, surfaces=2, margin=None):
    """Wire the driver globals + monkeypatch the margin fn; return the recorded calls."""
    calls = []

    def fake_margin(
        surfaces_arg, clearance_arg, temperature_arg, objective_arg, *, boozer_surfaces
    ):
        calls.append(
            {
                "surfaces": surfaces_arg,
                "clearance": clearance_arg,
                "temperature": temperature_arg,
                "objective": objective_arg,
                "boozer_surfaces": boozer_surfaces,
            }
        )
        # (signed_value, signed_grad = d(signed_value)/dx, violation)
        return margin if margin is not None else (0.0, np.zeros(2), 0.0)

    monkeypatch.setattr(drv, "surface_data", _fake_surface_data(surfaces), raising=False)
    monkeypatch.setattr(drv, "SURFACE_NESTING_WEIGHT", float(weight), raising=False)
    monkeypatch.setattr(
        drv, "SURFACE_NESTING_CLEARANCE", float(clearance), raising=False
    )
    monkeypatch.setattr(drv, "JF", SimpleNamespace(name="JF-sentinel"), raising=False)
    monkeypatch.setattr(drv, "current_alm_distance_smoothing", lambda: 0.01)
    monkeypatch.setattr(drv, "_smooth_min_surface_stack_signed_constraint", fake_margin)
    return calls


def _evaluation():
    return {"total": 100.0, "grad": np.array([1.0, 2.0])}


def test_enabled_weight_folds_penalty_into_total_and_grad(monkeypatch):
    # The load-bearing wiring check: an enabled weight on an in-band margin moves
    # BOTH total and grad by exactly the pure barrier, and records a diagnostic.
    signed_value, signed_grad = 0.4, np.array([2.0, -1.0])
    weight, clearance = 3.0, 0.5
    calls = _install(
        monkeypatch,
        weight=weight,
        clearance=clearance,
        margin=(signed_value, signed_grad, max(0.0, signed_value)),
    )
    ev = _evaluation()
    out = drv._apply_surface_nesting_penalty(ev)

    pen_val, pen_grad = surface_stack_nesting_penalty(signed_value, signed_grad, weight)
    assert pen_val > 0.0  # in-band -> a real penalty, not an inert no-op
    assert out["total"] == pytest.approx(100.0 + pen_val)
    np.testing.assert_allclose(out["grad"], np.array([1.0, 2.0]) + pen_grad)
    assert out["J_surface_nesting"] == pytest.approx(pen_val)

    # The margin fn was called once with the wired surfaces / clearance / smoothing
    # temperature / JF objective, proving the glue passes the right contract.
    assert len(calls) == 1
    c = calls[0]
    assert c["surfaces"] == ("surf0", "surf1")
    assert c["clearance"] == clearance
    assert c["temperature"] == 0.01
    assert c["objective"].name == "JF-sentinel"
    assert len(c["boozer_surfaces"]) == 2


def test_zero_weight_is_byte_identical_and_skips_margin(monkeypatch):
    # Default-OFF: the helper returns the evaluation object UNCHANGED and must not
    # even call the (potentially expensive) margin fn.
    calls = _install(
        monkeypatch, weight=0.0, clearance=0.5, margin=(0.4, np.array([2.0, -1.0]), 0.4)
    )
    ev = _evaluation()
    out = drv._apply_surface_nesting_penalty(ev)
    assert out is ev
    assert out["total"] == 100.0
    np.testing.assert_array_equal(out["grad"], np.array([1.0, 2.0]))
    assert "J_surface_nesting" not in out
    assert calls == []  # gated out before any margin computation


def test_zero_clearance_is_inert(monkeypatch):
    calls = _install(
        monkeypatch, weight=3.0, clearance=0.0, margin=(0.4, np.array([2.0, -1.0]), 0.4)
    )
    ev = _evaluation()
    out = drv._apply_surface_nesting_penalty(ev)
    assert out is ev
    assert "J_surface_nesting" not in out
    assert calls == []


def test_single_surface_is_inert(monkeypatch):
    # < 2 surfaces -> no adjacent pair -> the barrier is undefined; must be inert.
    calls = _install(
        monkeypatch,
        weight=3.0,
        clearance=0.5,
        surfaces=1,
        margin=(0.4, np.array([2.0, -1.0]), 0.4),
    )
    ev = _evaluation()
    out = drv._apply_surface_nesting_penalty(ev)
    assert out is ev
    assert "J_surface_nesting" not in out
    assert calls == []


def test_out_of_band_margin_leaves_total_and_grad_unchanged(monkeypatch):
    # Enabled lever but surfaces comfortably spaced (signed_value < 0): the one-sided
    # barrier contributes exactly zero to total and grad; the diagnostic records 0.
    _install(
        monkeypatch,
        weight=3.0,
        clearance=0.5,
        margin=(-0.2, np.array([5.0, 5.0]), 0.0),
    )
    ev = _evaluation()
    out = drv._apply_surface_nesting_penalty(ev)
    assert out["total"] == pytest.approx(100.0)
    np.testing.assert_allclose(out["grad"], np.array([1.0, 2.0]))
    assert out["J_surface_nesting"] == 0.0


def test_evaluate_total_objective_routes_through_surface_nesting_penalty(monkeypatch):
    """The INTEGRATION boundary: evaluate_total_objective must return
    ``_apply_surface_nesting_penalty(evaluation)``, not early-return the scalarized
    evaluation.

    The dead-return bug -- ``return apply_frontier_scalarization_override(...)`` with an
    unreachable ``return _apply_surface_nesting_penalty(evaluation)`` below it -- silently
    disabled the penalty for every multisurface penalty-mode run while keeping the helper
    unit-correct. The helper-only tests above pass either way; this drives the real
    driver function and asserts the wrapper is on the live return path.
    """
    terms = {
        "effective_res_weight": 0.0,
        "effective_iotas_weight": 0.0,
        "JNonQSObjective": None,
        "JBoozerObjective": None,
        "JVolume": None,
        "effective_volume_weight": 0.0,
    }
    monkeypatch.setattr(
        drv, "resolve_current_surface_objective_terms", lambda *a, **k: terms
    )
    monkeypatch.setattr(drv, "SINGLE_STAGE_POLOIDAL_WEIGHT", 0.0, raising=False)
    scalarized = {"total": 42.0, "grad": np.zeros(2)}
    monkeypatch.setattr(drv, "_evaluate_total_objective_impl", lambda *a, **k: scalarized)
    # The frontier override is a passthrough here; the penalty wrapper is under test.
    monkeypatch.setattr(
        drv, "apply_frontier_scalarization_override", lambda evaluation, **k: evaluation
    )
    sentinel = {"total": -1.0, "grad": np.ones(2), "wrapped_by_penalty": True}
    received = {}

    def fake_penalty(evaluation):
        received["evaluation"] = evaluation
        return sentinel

    monkeypatch.setattr(drv, "_apply_surface_nesting_penalty", fake_penalty)

    out = drv.evaluate_total_objective(
        None, None, None, 0.0, None, 0.0, None, 0.0, None, 0.0, None, 0.0, None, 0.0,
        width_min_threshold=0.0,
        width_max_threshold=0.0,
    )
    # The live return path went THROUGH the penalty wrapper, applied to the scalarized
    # evaluation -- the early-return regression would instead return `scalarized` and
    # never populate `received`.
    assert out is sentinel
    assert received["evaluation"] is scalarized
