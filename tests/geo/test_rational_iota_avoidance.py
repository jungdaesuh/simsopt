"""Tests for the single-stage rational-iota avoidance penalty (BANANA TODO #14).

Physics: a near-shearless vacuum field whose OUTER Boozer iota sits on a low-order
rational p/q (q <= Q) seeds magnetic islands whose half-width grows with the resonant
flux. ``RationalIotaAvoidance`` places a smooth Gaussian well on every reduced rational
with denominator q <= Q (weighted 1/q^2, width sigma0/q) so the penalty peaks ON each
rational and decays to ~0 between them; minimizing it pushes the outer iota off the
nearest low-order rational. The term reuses the SAME outer ``Iotas`` Boozer-adjoint
gradient path the iota-target penalty uses (single source of truth for d(iota)/d(coils)).

These tests construct a controllable iota term (a real ``Optimizable`` whose ``J`` is a
settable iota and whose ``dJ`` is a fixed nonzero d(iota)/dx ``Derivative`` -- a faithful
stand-in for ``Iotas`` through which the chain rule must flow) and assert observable
behavior, NOT the implementation:

  * the penalty PEAKS at a rational and DIPS between rationals (the well shape),
  * the gradient's descent direction pushes iota AWAY from a low-order rational,
  * the analytic d/d(iota) matches a finite difference (chain rule is correct),
  * the analytic coil-dof gradient is the d/d(iota) prefactor times the iota term's
    Boozer-adjoint Derivative (the SSOT gradient path is actually used), and
  * the default-weight-0 build returns ``None`` so the objective graph is byte-identical
    (the opt-in lock).
"""

import sys
import unittest
from pathlib import Path

import numpy as np

from simsopt._core.derivative import Derivative, derivative_dec
from simsopt._core.optimizable import Optimizable

EXAMPLES_ROOT = (
    Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
)
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from banana_opt.single_stage_objectives import (  # noqa: E402
    RATIONAL_IOTA_AVOIDANCE_DEFAULT_SIGMA,
    RATIONAL_IOTA_AVOIDANCE_MAX_DENOMINATOR,
    RationalIotaAvoidance,
    build_single_stage_rational_iota_avoidance_objective,
    build_total_objective,
)


class _ControllableIotaTerm(Optimizable):
    """A real Optimizable standing in for the outer-surface ``Iotas`` objective.

    ``J()`` returns a settable iota; ``dJ()`` returns a fixed d(iota)/dx as a genuine
    ``Derivative`` keyed on this term, so ``derivative_dec`` evaluates it against any
    downstream optimizable exactly as the true Boozer adjoint would. This lets the test
    sweep iota and finite-difference the penalty without solving a BoozerSurface, while
    still exercising the real Derivative chain (no mock of the gradient math).
    """

    def __init__(self, iota, diota_by_dx):
        self._iota = float(iota)
        self._diota_by_dx = np.asarray(diota_by_dx, dtype=float)
        Optimizable.__init__(self, x0=np.zeros(self._diota_by_dx.size))

    def set_iota(self, iota):
        self._iota = float(iota)
        self.recompute_bell()

    def J(self):
        return self._iota

    @derivative_dec
    def dJ(self):
        return Derivative({self: self._diota_by_dx.copy()})

    return_fn_map = {"J": J, "dJ": dJ}


# A nonzero, non-uniform d(iota)/dx so the coil-dof gradient direction is unambiguous.
_DIOTA_BY_DX = np.array([0.5, -0.25, 0.75])


def _make_penalty(iota, *, sigma=RATIONAL_IOTA_AVOIDANCE_DEFAULT_SIGMA, max_denom=10):
    term = _ControllableIotaTerm(iota, _DIOTA_BY_DX)
    penalty = RationalIotaAvoidance(term, max_denominator=max_denom, sigma=sigma)
    return term, penalty


class RationalIotaAvoidanceShapeTests(unittest.TestCase):
    def test_peaks_at_rational_and_dips_between(self):
        """J is a local maximum ON 3/10 and far smaller at the 3/10..1/3 midpoint."""
        _term, peak = _make_penalty(0.30)
        _term_l, just_left = _make_penalty(0.30 - 1e-3)
        _term_r, just_right = _make_penalty(0.30 + 1e-3)
        peak_value = peak.J()
        self.assertGreater(peak_value, just_left.J())
        self.assertGreater(peak_value, just_right.J())

        # Midpoint between the two nearest low-order rationals 3/10 and 1/3 is a basin.
        midpoint = (0.30 + 1.0 / 3.0) / 2.0
        _term_m, basin = _make_penalty(midpoint)
        self.assertLess(basin.J(), 0.1 * peak_value)

    def test_default_sigma_leaves_target_lanes_near_zero(self):
        """The 0.15/0.18 target lanes sit between q<=10 rationals -> negligible J.

        Documents that the default band is narrow enough that the avoidance term does
        not fight the non-rational iota targets even if a user enables it there.
        """
        _peak_term, peak = _make_penalty(0.30)
        peak_value = peak.J()
        for lane_iota in (0.15, 0.18):
            _term, lane = _make_penalty(lane_iota)
            self.assertLess(lane.J(), 0.05 * peak_value)


class RationalIotaAvoidanceGradientTests(unittest.TestCase):
    def test_descent_direction_pushes_iota_away_from_rational(self):
        """Within +/-1 sigma_q of 3/10 the iota-derivative drives iota off the rational.

        The penalty's d/d(iota) has the opposite sign of the descent step -dJ/d(iota);
        the step must point AWAY from 0.30 (down below it, up above it) so a gradient
        step escapes the resonance rather than locking onto it.
        """
        sigma0 = RATIONAL_IOTA_AVOIDANCE_DEFAULT_SIGMA
        sigma_q = sigma0 / 3  # 3/10 has denominator 3
        for offset in (-0.8, -0.4, 0.4, 0.8):
            iota = 0.30 + offset * sigma_q
            diota = self._dJ_diota(iota, sigma0)
            descent_step = -diota
            if iota < 0.30:
                self.assertLess(
                    descent_step,
                    0.0,
                    f"below 3/10 (iota={iota:.5f}) the step must push iota down",
                )
            else:
                self.assertGreater(
                    descent_step,
                    0.0,
                    f"above 3/10 (iota={iota:.5f}) the step must push iota up",
                )

    def test_analytic_iota_derivative_matches_finite_difference(self):
        """d(penalty)/d(iota) recovered from the coil gradient matches a FD on J(iota).

        Projecting the analytic coil-dof gradient back onto the (constant) d(iota)/dx
        isolates the scalar d(penalty)/d(iota); it must match a central difference of
        ``J`` in iota, proving the chain rule through the iota term is correct.
        """
        iota = 0.30 - 0.4 * (RATIONAL_IOTA_AVOIDANCE_DEFAULT_SIGMA / 3)
        analytic_diota = self._dJ_diota(iota, RATIONAL_IOTA_AVOIDANCE_DEFAULT_SIGMA)

        eps = 1e-7
        term_plus, penalty_plus = _make_penalty(iota + eps)
        term_minus, penalty_minus = _make_penalty(iota - eps)
        fd_diota = (penalty_plus.J() - penalty_minus.J()) / (2.0 * eps)
        np.testing.assert_allclose(analytic_diota, fd_diota, rtol=1e-5, atol=1e-9)

    def test_coil_gradient_is_iota_derivative_times_adjoint(self):
        """The coil-dof gradient is d(penalty)/d(iota) * d(iota)/dx (the SSOT path).

        Confirms the penalty routes its gradient through the iota term's Derivative
        rather than fabricating its own coil dependence: grad == scalar * d(iota)/dx.
        """
        iota = 0.30 + 0.5 * (RATIONAL_IOTA_AVOIDANCE_DEFAULT_SIGMA / 3)
        _term, penalty = _make_penalty(iota)
        grad = np.asarray(penalty.dJ(), dtype=float)
        scalar = self._dJ_diota(iota, RATIONAL_IOTA_AVOIDANCE_DEFAULT_SIGMA)
        np.testing.assert_allclose(grad, scalar * _DIOTA_BY_DX, rtol=1e-9, atol=1e-12)
        self.assertGreater(np.linalg.norm(grad), 0.0)

    @staticmethod
    def _dJ_diota(iota, sigma0):
        """Scalar d(penalty)/d(iota) recovered from the analytic coil-dof gradient.

        ``penalty.dJ()`` evaluates the penalty's gradient over its only leaf dofs (the
        iota term's): grad = (d penalty/d iota) * d(iota)/dx with d(iota)/dx a fixed
        nonzero vector, so the projection grad . v / (v . v) returns the scalar exactly.
        """
        _term, penalty = _make_penalty(iota, sigma=sigma0)
        grad = np.asarray(penalty.dJ(), dtype=float)
        return float(grad @ _DIOTA_BY_DX / (_DIOTA_BY_DX @ _DIOTA_BY_DX))


class RationalIotaAvoidanceBuilderTests(unittest.TestCase):
    def test_builder_returns_none_for_missing_iota_term(self):
        self.assertIsNone(
            build_single_stage_rational_iota_avoidance_objective(None)
        )

    def test_builder_uses_module_defaults(self):
        term = _ControllableIotaTerm(0.30, _DIOTA_BY_DX)
        penalty = build_single_stage_rational_iota_avoidance_objective(term)
        self.assertIsInstance(penalty, RationalIotaAvoidance)
        self.assertEqual(
            penalty.max_denominator, RATIONAL_IOTA_AVOIDANCE_MAX_DENOMINATOR
        )
        self.assertEqual(penalty.sigma, RATIONAL_IOTA_AVOIDANCE_DEFAULT_SIGMA)

    def test_invalid_parameters_rejected(self):
        term = _ControllableIotaTerm(0.30, _DIOTA_BY_DX)
        with self.assertRaises(ValueError):
            RationalIotaAvoidance(term, max_denominator=0)
        with self.assertRaises(ValueError):
            RationalIotaAvoidance(term, sigma=0.0)


class _PlainObjective(Optimizable):
    """Minimal real objective so build_total_objective can sum/scale genuine terms."""

    def __init__(self, value, gradient):
        self._value = float(value)
        self._gradient = np.asarray(gradient, dtype=float)
        Optimizable.__init__(self, x0=np.zeros(self._gradient.size))

    def J(self):
        return self._value

    @derivative_dec
    def dJ(self):
        return Derivative({self: self._gradient.copy()})

    return_fn_map = {"J": J, "dJ": dJ}


class RationalIotaAvoidanceDefaultOffLockTests(unittest.TestCase):
    """Lock test: the default (no term / weight 0) objective is byte-identical."""

    @staticmethod
    def _core_terms():
        grad = np.zeros(3)
        return dict(
            JnonQSRatio=_PlainObjective(1.0, grad),
            RES_WEIGHT=0.0,
            JBoozerResidual=_PlainObjective(0.0, grad),
            IOTAS_WEIGHT=0.0,
            Jiota=_PlainObjective(0.0, grad),
            VOLUME_WEIGHT=0.0,
            JVolume=None,
            LENGTH_WEIGHT=0.0,
            JCurveLength=_PlainObjective(0.0, grad),
            CC_WEIGHT=0.0,
            JCurveCurve=_PlainObjective(0.0, grad),
            CS_WEIGHT=0.0,
            JCurveSurface=_PlainObjective(0.0, grad),
            CURVATURE_WEIGHT=0.0,
            JCurvature=_PlainObjective(0.0, grad),
        )

    def test_none_term_is_byte_identical_to_baseline(self):
        """Passing JRationalIotaAvoidance=None leaves J and dJ exactly unchanged."""
        baseline = build_total_objective(**self._core_terms())
        with_none = build_total_objective(
            **self._core_terms(),
            JRationalIotaAvoidance=None,
            RATIONAL_IOTA_AVOIDANCE_WEIGHT=0.0,
        )
        self.assertEqual(float(with_none.J()), float(baseline.J()))
        np.testing.assert_array_equal(
            np.asarray(with_none.dJ(), dtype=float),
            np.asarray(baseline.dJ(), dtype=float),
        )

    def test_enabled_term_changes_the_objective(self):
        """A present term with a positive weight actually perturbs J and contributes dJ.

        Guards against a no-op wiring that silently drops the term even when enabled.
        With every other weight 0, the enabled total's value equals the weighted penalty
        value and its gradient over the penalty's iota-term dofs equals the weighted
        penalty gradient -- so the term is genuinely summed into the objective, not
        dropped. (Baseline and enabled graphs have different leaf sets, so their full dJ
        vectors are not directly comparable; the penalty's own contribution is checked.)
        """
        term = _ControllableIotaTerm(0.30, _DIOTA_BY_DX)
        penalty = RationalIotaAvoidance(term)
        weight = 1.0
        baseline = build_total_objective(**self._core_terms())
        enabled = build_total_objective(
            **self._core_terms(),
            JRationalIotaAvoidance=penalty,
            RATIONAL_IOTA_AVOIDANCE_WEIGHT=weight,
        )
        self.assertNotEqual(float(enabled.J()), float(baseline.J()))
        self.assertAlmostEqual(
            float(enabled.J()) - float(baseline.J()),
            weight * float(penalty.J()),
            places=12,
        )
        # The enabled total's gradient w.r.t. the penalty's iota-term dofs is exactly the
        # weighted penalty gradient (the baseline graph has no dependence on those dofs).
        enabled_grad_on_iota = np.asarray(enabled.dJ(partials=True)(term), dtype=float)
        np.testing.assert_allclose(
            enabled_grad_on_iota,
            weight * np.asarray(penalty.dJ(), dtype=float),
            rtol=1e-9,
            atol=1e-12,
        )
        self.assertGreater(np.linalg.norm(enabled_grad_on_iota), 0.0)


# Noble target in the low-iota buildable window: 1/(11 + 1/phi), phi the golden ratio.
_NOBLE_LOW_IOTA = 1.0 / (11.0 + 1.0 / ((1.0 + 5.0 ** 0.5) / 2.0))  # ~0.086073


class RationalIotaAvoidanceLowIotaWindowTests(unittest.TestCase):
    """At the module-default Q the penalty is LIVE across [0.08, 0.09] (banana P4).

    The previous default Q=10 had no reduced p/q in [0.08, 0.09] (1/11, 1/12, 1/13
    all have q > 10), so the term was a structural no-op across the low-iota
    buildable band. Behavior is asserted at the imported module default, so a future
    retune that still covers the window keeps these green, but a regression to a Q
    with no well in [0.08, 0.09] fails them.
    """

    def test_q10_was_inert_in_window(self):
        """Regression anchor: q<=10 makes 1/12 (inside the band) effectively a no-op.

        There is no reduced p/q with q<=10 in [0.08, 0.09]; the nearest (1/10, 1/9) sit
        >10 sigma_q from 1/12, so the penalty there is ~1e-44 (no usable gradient),
        whereas the new default Q places a real well on 1/12 (see
        test_default_q_is_live_on_1_12_and_1_11).
        """
        _term, penalty = _make_penalty(1.0 / 12.0, max_denom=10)
        self.assertLess(penalty.J(), 1e-20)

    def test_default_q_is_live_on_1_12_and_1_11(self):
        """At the module default Q, J peaks to O(1/q^2) on the band-bounding rationals.

        Asserting against ~1/q^2 (not > 0) is what makes this fail on a Q->10
        regression: at Q=10 those centers are Gaussian-tail dead (J(1/12)~1e-44,
        J(1/11)~3e-15), far below the 0.5/q^2 floor the live wells clear.
        """
        for p, q in ((1, 12), (1, 11)):
            _term, penalty = _make_penalty(
                p / q, max_denom=RATIONAL_IOTA_AVOIDANCE_MAX_DENOMINATOR
            )
            self.assertGreater(penalty.J(), 0.5 / q ** 2)

    def test_noble_sits_in_a_trough_between_1_12_and_1_11(self):
        """J dips at the noble 0.086073 to << its neighbouring 1/12 and 1/11 peaks."""
        _t12, peak_12 = _make_penalty(
            1.0 / 12.0, max_denom=RATIONAL_IOTA_AVOIDANCE_MAX_DENOMINATOR
        )
        _t11, peak_11 = _make_penalty(
            1.0 / 11.0, max_denom=RATIONAL_IOTA_AVOIDANCE_MAX_DENOMINATOR
        )
        _tn, trough = _make_penalty(
            _NOBLE_LOW_IOTA, max_denom=RATIONAL_IOTA_AVOIDANCE_MAX_DENOMINATOR
        )
        self.assertLess(trough.J(), 0.1 * peak_12.J())
        self.assertLess(trough.J(), 0.1 * peak_11.J())

    def test_frontier_3_10_well_survives_at_default_q(self):
        """Raising the default Q must not move the frontier 3/10 well: 0.30 stays a peak."""
        _t, peak = _make_penalty(
            0.30, max_denom=RATIONAL_IOTA_AVOIDANCE_MAX_DENOMINATOR
        )
        _tl, left = _make_penalty(
            0.30 - 1e-3, max_denom=RATIONAL_IOTA_AVOIDANCE_MAX_DENOMINATOR
        )
        _tr, right = _make_penalty(
            0.30 + 1e-3, max_denom=RATIONAL_IOTA_AVOIDANCE_MAX_DENOMINATOR
        )
        self.assertGreater(peak.J(), left.J())
        self.assertGreater(peak.J(), right.J())


if __name__ == "__main__":
    unittest.main()
