"""Tests for the Stage-2 seed-gradient diagnostic (``diagnose_seed_gradient``).

The diagnostic is a pure, optimizer-free probe of the objective at the warm-start
seed, built to root-cause why L-BFGS-B fails to take a first step. These tests pin
its load-bearing behaviors on synthetic objectives whose gradient and curvature we
control exactly: (1) for a healthy convex objective ``-grad`` is a genuine descent
direction and the hardware/edge split + scaled-gradient norms are exact; (2) a
gradient that is inconsistent with the function (points uphill) is flagged as
ascent; (3) a scale vector that overshoots makes the untruncated first step climb
uphill even though ``-grad`` itself descends; (4) the hardware-vs-edge split equals
the SSOT hinge contribution. No heavyweight artifacts -- the suite stays fast.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

EXAMPLES_ROOT = (
    Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
)
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from banana_opt.stage2_objectives import (  # noqa: E402
    EDGE_IOTA_HINGE_LINEAR,
    _add_stage2_edge_iota_objective,
    diagnose_seed_gradient,
    format_seed_gradient_diagnostic,
)
from banana_opt.single_stage_geometry import (  # noqa: E402
    CurveSobolevBlock,
    Stage2PenaltyPreconditioner,
)


def _quadratic(curvature):
    """``J(x) = 0.5 * sum(curvature * x**2)`` with exact gradient ``curvature * x``."""
    curvature = np.asarray(curvature, dtype=float)

    def fun(x):
        x = np.asarray(x, dtype=float)
        return 0.5 * float(np.sum(curvature * x * x)), curvature * x

    return fun


class _FakeEdgeObjective:
    """Minimal stand-in for ``Stage2EdgeIotaSteeringObjective``.

    Returns a fixed mean ``delta_abs`` and a fixed ``d(delta_abs)/dx`` regardless
    of state, which is all the hinge fold reads; ``recompute_bell`` is a no-op.
    """

    def __init__(self, delta_abs, grad):
        self._delta_abs = float(delta_abs)
        self._grad = np.asarray(grad, dtype=float)

    def recompute_bell(self):
        return None

    def J(self):
        return self._delta_abs

    def dJ_by_dcoils(self):
        return self._grad


class SeedGradientDiagnosticTest(unittest.TestCase):
    def test_healthy_convex_objective_reports_descent_and_exact_norms(self):
        curvature = np.array([1.0, 4.0, 9.0])
        x0 = np.array([1.0, 1.0, 1.0])
        scale = np.ones(3)
        diag = diagnose_seed_gradient(_quadratic(curvature), x0, scale)

        expected_grad = curvature * x0  # [1, 4, 9]
        self.assertAlmostEqual(diag["grad_norm_2"], float(np.linalg.norm(expected_grad)))
        self.assertAlmostEqual(diag["grad_norm_inf"], 9.0)
        # No edge term -> the whole gradient is hardware, edge split is zero.
        self.assertEqual(diag["grad_edge_norm_2"], 0.0)
        self.assertAlmostEqual(diag["grad_hw_norm_2"], diag["grad_norm_2"])
        # scale == 1 -> the line search sees exactly the physical gradient.
        self.assertAlmostEqual(diag["scaled_grad_norm_2"], diag["grad_norm_2"])
        # -grad is a genuine descent direction at every probed step.
        self.assertTrue(all(d["dJ"] < 0.0 for d in diag["raw_descent"]))
        self.assertEqual(diag["verdict"], "descent_exists_along_minus_grad")
        # The rendered block carries the machine-readable JSON tail.
        self.assertIn(
            "SEED_GRADIENT_DIAGNOSTIC_JSON=", format_seed_gradient_diagnostic(diag)
        )

    def test_gradient_inconsistent_with_function_is_flagged_as_ascent(self):
        # Reports the NEGATED true slope, so -grad points back uphill toward larger J.
        def fun(x):
            x = np.asarray(x, dtype=float)
            return 0.5 * float(np.sum(x * x)), -x

        diag = diagnose_seed_gradient(fun, np.array([1.0, 1.0]), np.ones(2))
        self.assertTrue(all(d["dJ"] > 0.0 for d in diag["raw_descent"]))
        self.assertEqual(
            diag["verdict"], "minus_grad_is_ascent_gradient_inconsistent"
        )

    def test_scale_overshoot_makes_first_step_climb_while_minus_grad_descends(self):
        # Isotropic bowl: J = 0.5*||x||^2, grad = x. With scale^2 = 3 the untruncated
        # first step x0 - grad*scale^2 = x0 - 3*x0 = -2*x0 overshoots past the minimum
        # and climbs (J: 1 -> 4), even though -grad is a fine descent direction.
        x0 = np.array([1.0, 1.0])
        scale = np.full(2, np.sqrt(3.0))
        diag = diagnose_seed_gradient(_quadratic(np.ones(2)), x0, scale)
        self.assertLess(min(d["dJ"] for d in diag["raw_descent"]), 0.0)
        self.assertGreater(diag["first_step_dJ"], 0.0)
        self.assertAlmostEqual(diag["first_step_dJ"], 3.0)  # (||-2x0||^2)/2 - 1 = 4-1
        self.assertAlmostEqual(diag["scale_max"], np.sqrt(3.0))
        # The line-search-visible gradient is grad*scale (= (1,1)*sqrt(3)); pins the
        # scaled-gradient at scale != 1 so a grad/scale mix-up would be caught.
        self.assertAlmostEqual(diag["scaled_grad_norm_inf"], np.sqrt(3.0))

    def test_operator_step_matches_optimizer_transform(self):
        cholesky_factor = np.array([[2.0, 0.0], [0.5, 1.5]])
        preconditioner = Stage2PenaltyPreconditioner(
            diagonal_scale=np.ones(3),
            curve_blocks=(
                CurveSobolevBlock(
                    indices=np.array([1, 2]),
                    cholesky_factor=cholesky_factor,
                    metric_trace_mean=1.0,
                ),
            ),
            metric_kind="h1",
            alpha=1.0,
        )
        curvature = np.array([1.0, 4.0, 9.0])
        x0 = np.array([1.0, 0.5, -0.25])
        fun = _quadratic(curvature)
        grad = curvature * x0
        expected_step = preconditioner.step_from_gradient(grad)
        expected_j0 = fun(x0)[0]
        expected_first_step_dj = fun(x0 - expected_step)[0] - expected_j0
        expected_scaled_grad = preconditioner.grad_to_u(grad)

        diag = diagnose_seed_gradient(fun, x0, preconditioner)

        self.assertAlmostEqual(diag["first_step_dJ"], expected_first_step_dj)
        self.assertAlmostEqual(
            diag["scaled_grad_norm_2"], float(np.linalg.norm(expected_scaled_grad))
        )
        self.assertEqual(diag["STAGE2_PRECONDITIONER_KIND"], "h1")
        self.assertEqual(diag["STAGE2_PRECONDITIONER_CURVE_BLOCK_COUNT"], 1)
        rendered = format_seed_gradient_diagnostic(diag)
        self.assertIn("||grad_u||_2=", rendered)
        self.assertIn("P.step_from_gradient(grad)", rendered)
        self.assertNotIn("||grad*scale||_2=", rendered)
        self.assertNotIn("grad*scale^2", rendered)

    def test_hardware_edge_split_equals_the_hinge_contribution(self):
        curvature = np.array([1.0, 2.0, 3.0])
        x0 = np.array([0.5, 0.5, 0.5])
        scale = np.ones(3)
        fun_hw = _quadratic(curvature)
        edge_grad = np.array([0.1, -0.2, 0.3])
        edge = _FakeEdgeObjective(delta_abs=0.2, grad=edge_grad)
        weight, target_min = 2.0, 0.5  # shortfall 0.3 > 0 -> hinge active

        def fun_full(x):
            j_hw, g_hw = fun_hw(x)
            return _add_stage2_edge_iota_objective(
                j_hw,
                g_hw,
                edge_iota_objective=edge,
                edge_iota_weight=weight,
                edge_iota_target_min=target_min,
                edge_iota_hinge_shape=EDGE_IOTA_HINGE_LINEAR,
            )

        diag = diagnose_seed_gradient(
            fun_full,
            x0,
            scale,
            edge_iota_objective=edge,
            edge_iota_weight=weight,
            edge_iota_target_min=target_min,
            edge_iota_hinge_shape=EDGE_IOTA_HINGE_LINEAR,
        )
        # Linear hinge: penalty grad = -d(delta)/dx, so the edge contribution to the
        # gradient is weight * (-edge_grad); the remainder must be the hardware grad.
        expected_edge = weight * (-edge_grad)
        expected_hw = curvature * x0
        self.assertAlmostEqual(
            diag["grad_edge_norm_2"], float(np.linalg.norm(expected_edge))
        )
        self.assertAlmostEqual(
            diag["grad_hw_norm_2"], float(np.linalg.norm(expected_hw))
        )
        self.assertGreater(diag["grad_edge_norm_2"], 0.0)


if __name__ == "__main__":
    unittest.main()
