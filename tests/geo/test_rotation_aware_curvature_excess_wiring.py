"""Wiring tests for the single-stage rotation-aware per-point curvature steering.

The class ``RotationAwareCurvatureExcessPenalty`` (gradient-verified separately in
``test_rotation_aware_curvature_excess_penalty.py``) is wired into the single-stage
objective as the curvature STEERING term, REPLACING the scalar ``LpCurveCurvature``
in the descent graph when ``--single-stage-rotation-aware-curvature-cap`` is active
(and the fold + a finite-build pack frame exist). When OFF (default) the scalar
curvature path must stay byte-identical.

These tests drive the real module objective builders in ``single_stage_objectives``
with the lightweight algebraic fakes used across the other wiring suites:

- ``build_total_objective`` (penalty path): default-OFF byte-identity, and the
  SWAP-not-add contract (the rotation-aware term REPLACES ``CURVATURE_WEIGHT *
  JCurvature``, it does not add on top).
- ``evaluate_alm_objective`` -> ``evaluate_base_objective`` (ALM path): default-OFF
  byte-identity, and the term enters the ALM physics objective when active.
- A REAL ``RotationAwareCurvatureExcessPenalty`` (built on a real finite-build pack
  framedcurve fixture) presents the J/dJ optimizable contract the wiring relies on
  (the differentiable gradient correctness is proven in
  ``test_rotation_aware_curvature_excess_penalty.py``; the full real-term objective
  assembly is exercised by the example ``--init-only`` smoke).
"""

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

import unittest  # noqa: E402

from banana_opt.fold_buildability import (  # noqa: E402
    RotationAwareCurvatureExcessPenalty,
)
from banana_opt.hardware_contracts import (  # noqa: E402
    TYPE_KK_INNER_RADIUS_MARGIN_M,
)
from simsopt.geo import (  # noqa: E402
    CurveXYZFourier,
    create_multifilament_grid,
)
from banana_opt import single_stage_objectives  # noqa: E402

WINDING_R0 = 0.903


class _FakeAlgebraicObjective:
    """Minimal Optimizable stand-in: scalar value + gradient under +/* algebra.

    Mirrors the fake in ``test_b2_alm_pack_kwargs.py`` /
    ``test_banana_objective_modules.py`` so the module objective builders
    (``A + w * B``) accumulate value and gradient exactly as they do for real
    terms.
    """

    def __init__(self, value, gradient, projected_gradient=None):
        self._value = float(value)
        self._gradient = np.asarray(gradient, dtype=float)
        projected = gradient if projected_gradient is None else projected_gradient
        self._projected_gradient = np.asarray(projected, dtype=float)

    def J(self):
        return self._value

    def dJ(self, partials=False):
        if partials:
            return lambda _objective: self._projected_gradient.copy()
        return self._gradient.copy()

    def __add__(self, other):
        if other == 0:
            return self
        return _FakeAlgebraicObjective(
            self._value + other._value,
            self._gradient + other._gradient,
            self._projected_gradient + other._projected_gradient,
        )

    __radd__ = __add__

    def __mul__(self, scalar):
        return _FakeAlgebraicObjective(
            self._value * scalar,
            self._gradient * scalar,
            self._projected_gradient * scalar,
        )

    __rmul__ = __mul__


def _base_total_objective_args():
    # The 15 positional build_total_objective args; index 13 = CURVATURE_WEIGHT
    # (14.0), index 14 = JCurvature (value 15.0, grad [2,-2]). JVolume is None so
    # the volume term is skipped, matching the other assembly tests.
    return [
        _FakeAlgebraicObjective(1.0, [1.0, 0.0]),
        2.0,
        _FakeAlgebraicObjective(3.0, [0.0, 2.0]),
        4.0,
        _FakeAlgebraicObjective(5.0, [1.0, 1.0]),
        6.0,
        None,
        8.0,
        _FakeAlgebraicObjective(9.0, [0.0, 3.0]),
        10.0,
        _FakeAlgebraicObjective(11.0, [1.0, -1.0]),
        12.0,
        _FakeAlgebraicObjective(13.0, [0.5, 0.5]),
        14.0,
        _FakeAlgebraicObjective(15.0, [2.0, -2.0]),
    ]


CURVATURE_WEIGHT = 14.0
JCURVATURE_VALUE = 15.0
JCURVATURE_GRAD = np.array([2.0, -2.0])


def _real_pack_penalty():
    """A real RotationAwareCurvatureExcessPenalty on a finite-build pack frame.

    Reuses the wobble-loop + shared-frame fixture from the class gradient test so
    the hinge is genuinely active (J > 0), proving the assembled objective stays
    finite when the real term is wired in.
    """
    curve = CurveXYZFourier(64, 2)
    dofs = {name: 0.0 for name in curve.local_dof_names}
    dofs["xc(0)"] = WINDING_R0
    dofs["xc(1)"] = 0.05
    dofs["zs(1)"] = 0.05
    dofs["xc(2)"] = 0.018
    dofs["zs(2)"] = 0.026
    dofs["yc(1)"] = 0.011
    dofs["ys(2)"] = 0.015
    curve.x = np.array([dofs[name] for name in curve.local_dof_names])
    framedcurve = create_multifilament_grid(
        curve,
        2,
        7,
        0.0099,
        0.0066,
        rotation_order=2,
        frame="surface_tangent",
        surface_major_radius=WINDING_R0,
    )[0].framedcurve
    penalty = RotationAwareCurvatureExcessPenalty(
        framedcurve,
        curve,
        finite_build=None,
        margin=TYPE_KK_INNER_RADIUS_MARGIN_M,
        p=2,
    )
    return penalty, curve


class BuildTotalObjectiveSwapTest(unittest.TestCase):
    """Penalty path: the rotation-aware term REPLACES the scalar curvature steering."""

    def setUp(self):
        self.module = single_stage_objectives

    def test_signature_binds_the_rotation_aware_kwargs(self):
        signature = inspect.signature(self.module.build_total_objective)
        signature.bind_partial(
            *_base_total_objective_args(),
            JRotationAwareCurvatureExcess=None,
            ROTATION_AWARE_CURVATURE_EXCESS_WEIGHT=0.0,
        )

    def test_default_off_is_byte_identical(self):
        # Flag OFF (None term) -> the scalar CURVATURE_WEIGHT * JCurvature path is
        # byte-identical to the call that omits the rotation-aware kwargs entirely,
        # even with a nonzero weight (the None guard wins).
        baseline = self.module.build_total_objective(*_base_total_objective_args())
        default_off = self.module.build_total_objective(
            *_base_total_objective_args(),
            JRotationAwareCurvatureExcess=None,
            ROTATION_AWARE_CURVATURE_EXCESS_WEIGHT=0.0,
        )
        self.assertEqual(baseline.J(), default_off.J())
        np.testing.assert_array_equal(baseline.dJ(), default_off.dJ())

        weighted_none = self.module.build_total_objective(
            *_base_total_objective_args(),
            JRotationAwareCurvatureExcess=None,
            ROTATION_AWARE_CURVATURE_EXCESS_WEIGHT=123.0,
        )
        self.assertEqual(baseline.J(), weighted_none.J())
        np.testing.assert_array_equal(baseline.dJ(), weighted_none.dJ())

    def test_active_term_replaces_scalar_curvature_steering(self):
        # SWAP, not add: with the rotation-aware term active, the scalar
        # CURVATURE_WEIGHT * JCurvature contribution is REMOVED from the descent
        # graph and replaced by ROTATION_AWARE_CURVATURE_EXCESS_WEIGHT * the term.
        baseline = self.module.build_total_objective(*_base_total_objective_args())
        excess = _FakeAlgebraicObjective(0.4, [0.6, -0.5])
        weight = 50.0
        with_excess = self.module.build_total_objective(
            *_base_total_objective_args(),
            JRotationAwareCurvatureExcess=excess,
            ROTATION_AWARE_CURVATURE_EXCESS_WEIGHT=weight,
        )

        expected_delta_J = weight * excess.J() - CURVATURE_WEIGHT * JCURVATURE_VALUE
        self.assertAlmostEqual(with_excess.J() - baseline.J(), expected_delta_J)
        np.testing.assert_allclose(
            with_excess.dJ() - baseline.dJ(),
            weight * excess.dJ() - CURVATURE_WEIGHT * JCURVATURE_GRAD,
        )

    def test_real_pack_penalty_presents_optimizable_contract(self):
        # The real differentiable penalty (active hinge, J > 0) presents the J/dJ
        # optimizable contract the wiring composes (``weight * term`` and a finite
        # scalar J). Gradient correctness is proven in the dedicated FD test; the
        # full real-term objective assembly is covered by the example --init-only
        # smoke (mixing this real Optimizable with algebraic fakes is not a real
        # code path).
        penalty, _curve = _real_pack_penalty()
        self.assertGreater(penalty.J(), 0.0)
        self.assertTrue(np.isfinite(penalty.J()))
        scaled = CURVATURE_WEIGHT * penalty
        self.assertTrue(np.isfinite(scaled.J()))


def _zero_constraint_result_2d(*_args, **_kwargs):
    return -0.1, np.array([0.0, 0.0]), 0.0


class EvaluateAlmObjectiveRotationAwareTest(unittest.TestCase):
    """ALM path: the rotation-aware term threads into the ALM physics objective."""

    def setUp(self):
        self.module = single_stage_objectives

    def _evaluate(self, **extra_kwargs):
        zero = _FakeAlgebraicObjective(0.0, [0.0, 0.0])
        return self.module.evaluate_alm_objective(
            np.array([1.0]),
            [zero],
            [zero],
            RES_WEIGHT=0.0,
            Jiota=zero,
            IOTAS_WEIGHT=0.0,
            JVolume=None,
            VOLUME_WEIGHT=0.0,
            JCurveLength=zero,
            LENGTH_WEIGHT=0.0,
            JCurveCurve=zero,
            JCurveSurface=zero,
            JCurvature=zero,
            multipliers=np.array([0.0, 0.0, 0.0]),
            penalty=1.0,
            objective_optimizable=SimpleNamespace(),
            curves=["curve_a"],
            curve_curve_min_distance=0.05,
            outer_surface="outer",
            curve_surface_min_distance=0.02,
            banana_curve="banana",
            curvature_threshold=40.0,
            distance_smoothing=0.01,
            curvature_smoothing=0.05,
            constraint_names=(
                "coil_coil_spacing",
                "coil_surface_spacing",
                "max_curvature",
            ),
            curve_curve_constraint_fn=_zero_constraint_result_2d,
            curve_surface_constraint_fn=_zero_constraint_result_2d,
            curvature_constraint_fn=_zero_constraint_result_2d,
            activity_tolerances_fn=lambda *_a, **_k: np.zeros(3),
            **extra_kwargs,
        )

    def test_signature_binds_the_rotation_aware_kwargs(self):
        signature = inspect.signature(self.module.evaluate_alm_objective)
        signature.bind_partial(
            JRotationAwareCurvatureExcess=None,
            ROTATION_AWARE_CURVATURE_EXCESS_WEIGHT=0.0,
        )

    def test_default_off_is_byte_identical(self):
        baseline = self._evaluate()
        with_none = self._evaluate(
            JRotationAwareCurvatureExcess=None,
            ROTATION_AWARE_CURVATURE_EXCESS_WEIGHT=0.0,
        )
        self.assertEqual(with_none["total"], baseline["total"])
        np.testing.assert_array_equal(with_none["grad"], baseline["grad"])

        # Term present but zero-weight stays inert (opt-in default OFF).
        excess = _FakeAlgebraicObjective(2.0, [1.0, 0.0])
        zero_weight = self._evaluate(
            JRotationAwareCurvatureExcess=excess,
            ROTATION_AWARE_CURVATURE_EXCESS_WEIGHT=0.0,
        )
        self.assertEqual(zero_weight["total"], baseline["total"])
        np.testing.assert_array_equal(zero_weight["grad"], baseline["grad"])

    def test_active_term_enters_alm_physics_objective(self):
        # weighted_sum ALM total = base physics terms; the rotation-aware steering
        # term enters as + weight * term over the baseline.
        excess = _FakeAlgebraicObjective(2.0, [1.0, 3.0])
        baseline = self._evaluate()
        steered = self._evaluate(
            JRotationAwareCurvatureExcess=excess,
            ROTATION_AWARE_CURVATURE_EXCESS_WEIGHT=5.0,
        )
        self.assertAlmostEqual(steered["total"] - baseline["total"], 5.0 * 2.0)
        np.testing.assert_allclose(steered["grad"] - baseline["grad"], [5.0, 15.0])


if __name__ == "__main__":
    unittest.main()
