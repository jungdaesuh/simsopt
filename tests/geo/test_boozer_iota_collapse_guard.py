"""Regression tests for the Boozer iota-collapse early-exit + defense-in-depth reject.

These pin the fix for the single-stage hang in which a large trial coil step drives the
warm-started Boozer BFGS solve toward the trivial magnetic axis (iota -> ~3e-4) and grinds the
full ``bfgs_maxiter`` iterations inside one function evaluation.

Two layers are covered:
  1. ``BoozerSurface.minimize_boozer_penalty_constraints_LBFGS`` / ``run_code`` opt-in early-exit.
  2. ``single_stage_geometry.evaluate_surface_stack`` defense-in-depth iota-collapse reject.
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize

from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import coils_via_symmetries
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.surfaceobjectives import Area
from simsopt.configs.zoo import get_giuliani_data
from .surface_test_helpers import get_surface

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_PATH = _REPO_ROOT / "examples" / "single_stage_optimization"
if str(_EXAMPLES_PATH) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_PATH))
from banana_opt.single_stage_geometry import (  # noqa: E402
    IOTA_COLLAPSE_REJECT_FRACTION,
    evaluate_surface_stack,
)


def _build_giuliani_boozer_surface():
    """A real, small BoozerLS surface (the same fixture the convergence tests use)."""
    curves, currents, ma = get_giuliani_data()
    coils = coils_via_symmetries(curves, currents, ma.nfp, True)
    current_sum = sum(abs(c.current.get_value()) for c in coils)
    bs = BiotSavart(coils)
    s = get_surface("SurfaceXYZTensorFourier", True, nfp=ma.nfp)
    s.fit_to_curve(ma, 0.1)
    G0 = 2.0 * np.pi * current_sum * (4 * np.pi * 1e-7 / (2 * np.pi))
    ar = Area(s)
    cw = s.quadpoints_phi.size * s.quadpoints_theta.size * 3
    boozer = BoozerSurface(bs, s, ar, ar.J(), constraint_weight=100.0 / cw,
                           options={"verbose": False})
    return boozer, G0


class ScipyStopIterationContractTest(unittest.TestCase):
    """The early-exit relies on scipy's documented callback->StopIteration contract.

    Pin it so a scipy upgrade that changes this behavior fails loudly here.
    """

    def test_bfgs_callback_stopiteration_marks_unsuccessful_status_99(self):
        def fun(x):
            return float(np.sum((x - 3.0) ** 2)), 2.0 * (x - 3.0)

        seen = {"n": 0}

        def callback(intermediate_result):
            seen["n"] += 1
            # current iterate is exposed on .x
            assert hasattr(intermediate_result, "x")
            if seen["n"] >= 2:
                raise StopIteration

        res = minimize(fun, np.array([-10.0, -10.0]), jac=True, method="BFGS",
                       callback=callback, options={"maxiter": 1500, "gtol": 1e-12})
        self.assertFalse(res.success)
        self.assertEqual(res.status, 99)
        self.assertLessEqual(res.nit, 2)


class BoozerLBFGSIotaCollapseGuardTest(unittest.TestCase):
    def test_guard_none_is_byte_identical_to_legacy(self):
        # Two independent solves: one with the param omitted, one with iota_collapse_fraction=None.
        boozer_a, G0 = _build_giuliani_boozer_surface()
        res_a = boozer_a.minimize_boozer_penalty_constraints_LBFGS(
            tol=1e-10, maxiter=700, constraint_weight=boozer_a.constraint_weight,
            iota=0.4, G=G0, limited_memory=False)

        boozer_b, G0b = _build_giuliani_boozer_surface()
        res_b = boozer_b.minimize_boozer_penalty_constraints_LBFGS(
            tol=1e-10, maxiter=700, constraint_weight=boozer_b.constraint_weight,
            iota=0.4, G=G0b, limited_memory=False,
            iota_collapse_fraction=None, iota_reference=None)

        # Default-None path: identical trajectory (same iter count, bitwise-identical iota & fun)
        # and the collapse flag is always present and False.
        self.assertEqual(res_a["iter"], res_b["iter"])
        self.assertEqual(res_a["iota"], res_b["iota"])
        self.assertEqual(res_a["fun"], res_b["fun"])
        self.assertFalse(res_a["iota_collapsed"])
        self.assertFalse(res_b["iota_collapsed"])

    def test_healthy_solve_with_guard_enabled_is_unaffected(self):
        boozer, G0 = _build_giuliani_boozer_surface()
        # Reference at the true iota (~0.4); a healthy solve stays well above 0.3*ref and must
        # converge normally (guard does not trip).
        res = boozer.minimize_boozer_penalty_constraints_LBFGS(
            tol=1e-10, maxiter=700, constraint_weight=boozer.constraint_weight,
            iota=0.4, G=G0, limited_memory=False,
            iota_collapse_fraction=IOTA_COLLAPSE_REJECT_FRACTION, iota_reference=0.4)
        self.assertFalse(res["iota_collapsed"])
        self.assertTrue(res["success"])
        # giuliani iota ~ 0.4; comfortably above the 0.12 floor.
        self.assertGreater(abs(res["iota"]), 0.3)

    def test_guard_trips_early_when_iota_below_floor(self):
        # Compare full-solve iter count vs. guarded iter count when the floor is set ABOVE the
        # achievable iota so the iterate is always "collapsed" -> callback fires immediately.
        boozer_full, G0 = _build_giuliani_boozer_surface()
        res_full = boozer_full.minimize_boozer_penalty_constraints_LBFGS(
            tol=1e-10, maxiter=700, constraint_weight=boozer_full.constraint_weight,
            iota=0.4, G=G0, limited_memory=False)
        self.assertFalse(res_full["iota_collapsed"])

        boozer_guard, G0g = _build_giuliani_boozer_surface()
        # iota_reference=10.0 => floor = 0.3*10 = 3.0, far above the real iota (~0.4) => trips.
        res_guard = boozer_guard.minimize_boozer_penalty_constraints_LBFGS(
            tol=1e-10, maxiter=700, constraint_weight=boozer_guard.constraint_weight,
            iota=0.4, G=G0g, limited_memory=False,
            iota_collapse_fraction=IOTA_COLLAPSE_REJECT_FRACTION, iota_reference=10.0)
        self.assertTrue(res_guard["iota_collapsed"])
        self.assertFalse(res_guard["success"])
        # Early-exit: strictly (and dramatically) fewer iterations than the full solve.
        self.assertLess(res_guard["iter"], res_full["iter"])

    def test_run_code_returns_collapsed_result_without_newton_polish(self):
        # run_code on a collapsed BFGS solve should short-circuit (skip Newton) and surface the
        # unsuccessful, collapse-tagged result.
        boozer, G0 = _build_giuliani_boozer_surface()
        res = boozer.run_code(0.4, G=G0,
                              iota_collapse_fraction=IOTA_COLLAPSE_REJECT_FRACTION,
                              iota_reference=10.0)
        self.assertTrue(res.get("iota_collapsed"))
        self.assertFalse(res["success"])
        # Collapsed BFGS result is type 'ls' from the LBFGS path (Newton was skipped).
        self.assertEqual(res["type"], "ls")


class _FakeSurface:
    def __init__(self, volume):
        self._volume = float(volume)

    def volume(self):
        return self._volume

    def is_self_intersecting(self):
        return False

    def cross_section(self, *_args, **_kwargs):
        return np.zeros((4, 3))


def _entry(name, volume, iota, success=True):
    return {
        "name": name,
        "boozer_surface": SimpleNamespace(
            surface=_FakeSurface(volume),
            res={"iota": iota, "success": success},
        ),
    }


class EvaluateSurfaceStackIotaCollapseRejectTest(unittest.TestCase):
    def test_reference_none_does_not_reject_low_iota(self):
        # Backwards-compatible default: no reference => no iota-collapse check.
        surface_data = [_entry("outer", 1.0, 1e-4)]
        result = evaluate_surface_stack(surface_data)
        self.assertTrue(result["iota_collapse_ok"])
        self.assertEqual(result["iota_collapsed"], [False])
        self.assertIsNone(result["reason"])
        self.assertTrue(result["success"])

    def test_collapsed_iota_is_rejected_even_when_solve_succeeded(self):
        # Solve nominally succeeded, no self-intersection, but iota collapsed far below reference.
        surface_data = [_entry("outer", 1.0, 1e-4, success=True)]
        result = evaluate_surface_stack(surface_data, reference_iotas=[0.16])
        self.assertFalse(result["iota_collapse_ok"])
        self.assertEqual(result["iota_collapsed"], [True])
        self.assertEqual(result["reason"], "iota_collapse")
        self.assertFalse(result["success"])

    def test_healthy_iota_passes_with_reference(self):
        surface_data = [_entry("outer", 1.0, 0.157, success=True)]
        result = evaluate_surface_stack(surface_data, reference_iotas=[0.16])
        self.assertTrue(result["iota_collapse_ok"])
        self.assertEqual(result["iota_collapsed"], [False])
        self.assertIsNone(result["reason"])
        self.assertTrue(result["success"])

    def test_threshold_is_named_constant_fraction_of_reference(self):
        # Just above and just below 0.3 * |reference| straddle the boundary.
        ref = 0.20
        floor = IOTA_COLLAPSE_REJECT_FRACTION * ref  # 0.06
        below = _entry("a", 1.0, 0.9 * floor, success=True)
        above = _entry("b", 1.0, 1.1 * floor, success=True)
        below_result = evaluate_surface_stack([below], reference_iotas=[ref])
        above_result = evaluate_surface_stack([above], reference_iotas=[ref])
        self.assertEqual(below_result["iota_collapsed"], [True])
        self.assertEqual(above_result["iota_collapsed"], [False])

    def test_collapse_check_uses_absolute_value_of_iota_and_reference(self):
        # Negative (signed-CW) iota near the negative reference must NOT be flagged as collapsed.
        surface_data = [_entry("outer", 1.0, -0.157, success=True)]
        result = evaluate_surface_stack(surface_data, reference_iotas=[-0.16])
        self.assertEqual(result["iota_collapsed"], [False])
        self.assertTrue(result["iota_collapse_ok"])


if __name__ == "__main__":
    unittest.main()
