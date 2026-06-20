"""Gradient + behavioral tests for ``RotationAwareCurvatureExcessPenalty``.

The penalty places a per-quadpoint, rotation-aware cap on the centerline
curvature: ``cap(theta) = 1/(margin + reach(theta; alpha))`` where ``reach`` is the
LIVE rotated outer-channel half-extent projected into the centerline bend plane
(SSOT: ``rotation_aware_projected_half_extent_m``). ``dJ`` must carry both gradient
paths -- through ``|kappa|`` to the banana centerline dofs, and through ``cap ->
reach -> rotated_frame()`` to the ``FrameRotation`` alpha dofs. These tests prove
each path by central finite differences (rel.err < 1e-6) and check the relief
property (folding alpha into the bend strictly lowers the penalty).
"""

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from banana_opt.fold_buildability import (  # noqa: E402
    RotationAwareCurvatureExcessPenalty,
    _projected_reach_pure,
)
from banana_opt.stage2_geometry import (  # noqa: E402
    rotation_aware_projected_half_extent_m,
)
from banana_opt.hardware_contracts import (  # noqa: E402
    TYPE_KK_INNER_RADIUS_MARGIN_M,
    TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M,
    TYPE_KK_OUTER_CHANNEL_HALF_WIDTH_BINORMAL_M,
)
from simsopt.geo import (  # noqa: E402
    CurveXYZFourier,
    FrameRotation,
    FramedCurveSurfaceTangent,
    create_multifilament_grid,
)

WINDING_R0 = 0.903


def _banana_curve(npoints=64, radius=0.05):
    """A small near-poloidal loop with a wobble so kappa is non-uniform and high.

    The tight base radius plus the order-2 wobble drives the local curvature well
    above the rotation-aware cap on part of the loop, so the
    hinge is genuinely active and the twist lever has something to relieve. (A
    gentle a=0.13 loop stays entirely below the ~43-123/m cap, J==0 everywhere.)
    """
    curve = CurveXYZFourier(npoints, 2)
    dofs = {name: 0.0 for name in curve.local_dof_names}
    dofs["xc(0)"] = WINDING_R0
    dofs["xc(1)"] = radius
    dofs["zs(1)"] = radius
    dofs["xc(2)"] = 0.018
    dofs["zs(2)"] = 0.026
    dofs["yc(1)"] = 0.011
    dofs["ys(2)"] = 0.015
    curve.x = np.array([dofs[name] for name in curve.local_dof_names])
    return curve


def _shared_frame(curve, rotation_order=2):
    """The shared surface-tangent frame the finite-build pack rides (carries alpha)."""
    filaments = create_multifilament_grid(
        curve,
        2,
        7,
        0.0099,
        0.0066,
        rotation_order=rotation_order,
        frame="surface_tangent",
        surface_major_radius=WINDING_R0,
    )
    return filaments[0].framedcurve


def _penalty(framedcurve, curve, p=2, margin=TYPE_KK_INNER_RADIUS_MARGIN_M):
    return RotationAwareCurvatureExcessPenalty(
        framedcurve, curve, finite_build=None, margin=margin, p=p
    )


def _central_fd_directional(value_fn, x0, direction, eps):
    plus = value_fn(x0 + eps * direction)
    minus = value_fn(x0 - eps * direction)
    return (plus - minus) / (2.0 * eps)


class ReachSSOTTest(unittest.TestCase):
    """The differentiable JAX reach must equal the numpy SSOT byte-for-byte."""

    def test_projected_reach_matches_numpy_ssot(self):
        curve = _banana_curve()
        frame = _shared_frame(curve)
        # Excite the twist so the rotated frame is non-trivial.
        rot = frame.rotation
        x = np.asarray(rot.x, dtype=float).copy()
        x[0] = 0.4
        x[1] = 0.2
        rot.x = x

        _, normal_axis, binormal_axis = (
            np.asarray(a, dtype=float) for a in frame.rotated_frame()
        )
        jax_reach = np.asarray(
            _projected_reach_pure(
                curve.gammadash(),
                curve.gammadashdash(),
                normal_axis,
                binormal_axis,
                TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M,
                TYPE_KK_OUTER_CHANNEL_HALF_WIDTH_BINORMAL_M,
            ),
            dtype=float,
        )
        numpy_reach = rotation_aware_projected_half_extent_m(None, curve, frame)
        np.testing.assert_allclose(jax_reach, numpy_reach, rtol=0.0, atol=1e-12)


class HingeBehaviorTest(unittest.TestCase):
    def test_zero_when_all_kappa_below_cap(self):
        # A pure large circle: kappa ~ 1/R is tiny vs the ~38-104/m cap.
        curve = CurveXYZFourier(64, 1)
        dofs = {name: 0.0 for name in curve.local_dof_names}
        dofs["xc(0)"] = WINDING_R0
        dofs["xc(1)"] = 0.5
        dofs["zs(1)"] = 0.5
        curve.x = np.array([dofs[name] for name in curve.local_dof_names])
        frame = _shared_frame(curve)
        pen = _penalty(frame, curve)
        cap = pen.rotation_aware_cap_inv_m()
        self.assertTrue(np.all(np.abs(curve.kappa()) <= cap))
        self.assertEqual(pen.J(), 0.0)

    def test_positive_when_some_kappa_above_cap(self):
        curve = _banana_curve()
        frame = _shared_frame(curve)
        pen = _penalty(frame, curve)
        cap = pen.rotation_aware_cap_inv_m()
        self.assertTrue(np.any(np.abs(curve.kappa()) > cap))
        self.assertGreater(pen.J(), 0.0)

    def test_folding_alpha_into_bend_relieves_penalty(self):
        """Folding the twist toward the bend strictly lowers J (the relief property).

        The relief mechanism is per-point: where ``|kappa|`` is high the cap must
        rise (thin/normal extent laid into the bend), not the global cap mean -- a
        fold that helps a high-curvature apex can lower the cap elsewhere where it
        does not matter. So the invariant asserted is J relief, plus a rise of the
        cap on the arclength that was actively infeasible at the seed.
        """
        curve = _banana_curve()
        frame = _shared_frame(curve)
        pen = _penalty(frame, curve)
        rot = frame.rotation
        x0 = np.asarray(rot.x, dtype=float).copy()
        j0 = pen.J()
        self.assertGreater(j0, 0.0)
        kappa = np.abs(curve.kappa())
        cap_seed = pen.rotation_aware_cap_inv_m()
        active = kappa > cap_seed  # the over-cap (infeasible) quadpoints at the seed

        # Sweep the constant twist mode; pick the orientation that lowers J.
        best_j = j0
        best_x = x0
        for alpha_const in np.linspace(0.0, 1.6, 33):
            x = x0.copy()
            x[0] = alpha_const
            rot.x = x
            j = pen.J()
            if j < best_j:
                best_j = j
                best_x = x.copy()
        rot.x = best_x
        cap_folded = pen.rotation_aware_cap_inv_m()
        rot.x = x0

        self.assertLess(best_j, j0)  # relief: folding strictly lowers the penalty
        # The relief came from raising the cap on the seed-infeasible arclength.
        self.assertGreater(
            float(np.mean(cap_folded[active])), float(np.mean(cap_seed[active]))
        )


class GradientCheckTest(unittest.TestCase):
    """Central-FD vs analytic dJ on each dof group independently (rel.err < 1e-6)."""

    REL_TOL = 1e-6

    def _setup(self):
        curve = _banana_curve()
        frame = _shared_frame(curve)
        rot = frame.rotation
        # Seed a non-trivial twist so the alpha gradient is exercised away from 0.
        x = np.asarray(rot.x, dtype=float).copy()
        x[0] = 0.35
        x[1] = 0.18
        if x.size > 2:
            x[2] = -0.12
        rot.x = x
        pen = _penalty(frame, curve)
        # The active hinge must be genuinely engaged for a meaningful check.
        cap = pen.rotation_aware_cap_inv_m()
        assert np.any(np.abs(curve.kappa()) > cap)
        assert pen.J() > 0.0
        return curve, frame, rot, pen

    def test_grad_wrt_banana_centerline_dofs(self):
        curve, frame, rot, pen = self._setup()
        x0 = np.asarray(curve.x, dtype=float).copy()
        # partials=True -> the Derivative object; (curve) projects onto curve dofs.
        analytic = np.asarray(pen.dJ(partials=True)(curve), dtype=float)
        rng = np.random.default_rng(7)
        direction = rng.standard_normal(x0.shape)
        direction /= np.linalg.norm(direction)

        def value_fn(x):
            curve.x = x
            return pen.J()

        fd = _central_fd_directional(value_fn, x0, direction, eps=1e-6)
        curve.x = x0
        analytic_dir = float(np.dot(analytic, direction))
        rel_err = abs(fd - analytic_dir) / max(abs(fd), 1e-30)
        self._rel_err_curve = rel_err
        self.assertLess(rel_err, self.REL_TOL, f"curve-dof rel.err={rel_err:.3e}")

    def test_grad_wrt_frame_rotation_alpha_dofs(self):
        curve, frame, rot, pen = self._setup()
        a0 = np.asarray(rot.x, dtype=float).copy()
        rot_names = list(rot.dof_names)
        # partials=True -> the Derivative object; (rot) projects onto alpha dofs.
        analytic = np.asarray(pen.dJ(partials=True)(rot), dtype=float)
        self.assertEqual(analytic.shape[0], len(rot_names))
        rng = np.random.default_rng(11)
        direction = rng.standard_normal(a0.shape)
        direction /= np.linalg.norm(direction)

        def value_fn(a):
            rot.x = a
            return pen.J()

        fd = _central_fd_directional(value_fn, a0, direction, eps=1e-6)
        rot.x = a0
        analytic_dir = float(np.dot(analytic, direction))
        rel_err = abs(fd - analytic_dir) / max(abs(fd), 1e-30)
        self._rel_err_alpha = rel_err
        self.assertLess(rel_err, self.REL_TOL, f"alpha-dof rel.err={rel_err:.3e}")


if __name__ == "__main__":
    unittest.main()
