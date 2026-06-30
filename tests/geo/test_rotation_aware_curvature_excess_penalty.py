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
    _centerline_bend_direction,
    rotation_aware_projected_half_extent_m,
    rotation_aware_warm_start_alpha_dofs,
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


class WarmStartTest(unittest.TestCase):
    """The reach-minimizing alpha warm-start escapes the alpha=0 basin.

    validate4 showed the optimizer never folds the pack from alpha=0 (it flattens
    the centerline kappa instead), so the realized cap stays pinned at the
    conservative corner. ``rotation_aware_warm_start_alpha_dofs`` seeds alpha at the
    thin-side-into-bend profile so the descent starts already folded.
    """

    def test_design_matrix_matches_framerotation_basis(self):
        """The analytic Fourier design matrix == the live FrameRotation.alpha (SSOT).

        Pins the warm-start's hand-rolled basis to ``framedcurve.jaxrotation_pure``;
        if that basis ever changes this fails instead of fitting onto stale modes.
        """
        curve = _banana_curve()
        frame = _shared_frame(curve, rotation_order=3)
        rot = frame.rotation
        quadpoints = np.asarray(curve.quadpoints, dtype=float)
        order = int(rot.order)
        scale = float(rot.scale)
        columns = [np.ones_like(quadpoints)]
        for j in range(1, order + 1):
            columns.append(np.sin(2.0 * np.pi * j * quadpoints))
            columns.append(np.cos(2.0 * np.pi * j * quadpoints))
        design = scale * np.stack(columns, axis=1)
        rng = np.random.default_rng(3)
        x = rng.standard_normal(np.asarray(rot.x).shape)
        rot.x = x
        np.testing.assert_allclose(
            design @ x, np.asarray(rot.alpha(quadpoints), dtype=float),
            rtol=0.0, atol=1e-12,
        )

    def test_warm_start_is_pure_and_reference_independent(self):
        """No input mutation, and the result is independent of the rotation's live
        dofs -- the reference frame is read at alpha=0, not the current twist.

        Seeding a non-trivial alpha first exercises the documented invariant (a
        re-warm-start reads the same alpha=0 reference); at the zero state alone the
        explicit-zero and live references coincide and a regression reading live
        dofs as the reference would slip through.
        """
        curve = _banana_curve()
        frame = _shared_frame(curve, rotation_order=3)
        rot = frame.rotation
        dofs_from_zero = rotation_aware_warm_start_alpha_dofs(curve, frame)
        rng = np.random.default_rng(5)
        seeded = rng.standard_normal(np.asarray(rot.x).shape)
        rot.x = seeded
        x_before = np.asarray(rot.x, dtype=float).copy()
        dofs_from_seeded = rotation_aware_warm_start_alpha_dofs(curve, frame)
        # (a) pure: the call did not mutate the live rotation.
        np.testing.assert_array_equal(np.asarray(rot.x, dtype=float), x_before)
        # (b) reference-independent: same result regardless of pre-existing twist.
        np.testing.assert_allclose(
            dofs_from_seeded, dofs_from_zero, rtol=0.0, atol=1e-12
        )

    def test_warm_start_relieves_penalty_from_zero_alpha(self):
        curve = _banana_curve()
        frame = _shared_frame(curve, rotation_order=3)
        pen = _penalty(frame, curve)
        rot = frame.rotation
        rot.x = np.zeros_like(np.asarray(rot.x, dtype=float))
        j_zero = pen.J()
        self.assertGreater(j_zero, 0.0)
        rot.x = rotation_aware_warm_start_alpha_dofs(curve, frame)
        j_warm = pen.J()
        self.assertLess(j_warm, j_zero)

    def test_warm_start_reduces_residual_infeasible_arclength(self):
        curve = _banana_curve()
        frame = _shared_frame(curve, rotation_order=3)
        rot = frame.rotation
        margin = TYPE_KK_INNER_RADIUS_MARGIN_M
        kappa = np.abs(np.asarray(curve.kappa(), dtype=float))
        arclen = np.linalg.norm(np.asarray(curve.gammadash(), dtype=float), axis=1)

        def residual_fraction():
            reach = rotation_aware_projected_half_extent_m(None, curve, frame)
            cap = 1.0 / (margin + reach)
            infeasible = kappa > cap
            return float(np.sum(arclen[infeasible]) / np.sum(arclen))

        rot.x = np.zeros_like(np.asarray(rot.x, dtype=float))
        residual_zero = residual_fraction()
        self.assertGreater(residual_zero, 0.0)
        rot.x = rotation_aware_warm_start_alpha_dofs(curve, frame)
        residual_warm = residual_fraction()
        self.assertLess(residual_warm, residual_zero)

    def test_warm_start_lays_thin_axis_into_bend_on_active_arc(self):
        """On the high-curvature arc the thin normal ends up more aligned with the
        bend than the wide binormal -- the geometric definition of the fold. The
        alpha=0 baseline is pinned so the test proves the FLIP is caused by the
        warm-start (not a fixture that already satisfies it)."""
        curve = _banana_curve()
        frame = _shared_frame(curve, rotation_order=3)
        rot = frame.rotation
        kappa = np.abs(np.asarray(curve.kappa(), dtype=float))
        active = kappa >= np.quantile(kappa, 0.75)
        bend = _centerline_bend_direction(curve)

        def aligns():
            _, normal_axis, binormal_axis = (
                np.asarray(a, dtype=float) for a in frame.rotated_frame()
            )
            normal_align = np.abs(
                np.sum(bend[active] * normal_axis[active], axis=1)
            ).mean()
            binormal_align = np.abs(
                np.sum(bend[active] * binormal_axis[active], axis=1)
            ).mean()
            return normal_align, binormal_align

        rot.x = np.zeros_like(np.asarray(rot.x, dtype=float))
        normal_zero, binormal_zero = aligns()
        # Baseline: the surface-tangent normal is the WIDE-aligned axis at alpha=0
        # (this is exactly why the conservative edgewise cap binds without a twist).
        self.assertLess(normal_zero, binormal_zero)

        rot.x = rotation_aware_warm_start_alpha_dofs(curve, frame)
        normal_warm, binormal_warm = aligns()
        # Warm-start flips it: the thin normal is now the bend-aligned axis.
        self.assertGreater(normal_warm, binormal_warm)
        self.assertGreater(normal_warm, normal_zero)

    def test_warm_start_degenerate_thick_normal_aligns_binormal(self):
        """If the normal were the WIDE axis (half_n > half_b), the warm-start must
        lay the BINORMAL into the bend instead (the +pi/2 degenerate branch)."""
        curve = _banana_curve()
        frame = _shared_frame(curve, rotation_order=3)
        rot = frame.rotation
        kappa = np.abs(np.asarray(curve.kappa(), dtype=float))
        active = kappa >= np.quantile(kappa, 0.75)
        bend = _centerline_bend_direction(curve)
        # Swap the half-extents so the normal is the thick axis.
        rot.x = rotation_aware_warm_start_alpha_dofs(
            curve,
            frame,
            half_n_m=TYPE_KK_OUTER_CHANNEL_HALF_WIDTH_BINORMAL_M,
            half_b_m=TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M,
        )
        _, normal_axis, binormal_axis = (
            np.asarray(a, dtype=float) for a in frame.rotated_frame()
        )
        normal_align = np.abs(
            np.sum(bend[active] * normal_axis[active], axis=1)
        ).mean()
        binormal_align = np.abs(
            np.sum(bend[active] * binormal_axis[active], axis=1)
        ).mean()
        self.assertGreater(binormal_align, normal_align)


class ProjectedReachSupportFunctionTest(unittest.TestCase):
    """The JAX reach is the exact rectangle SUPPORT FUNCTION, not a quadratic blend.

    Regression for the optimistic-reach bug: the reach in the centerline bend
    direction is ``half_n*|n| + half_b*|b|`` (worst inner-corner half-extent), the
    quantity the non-fold condition ``w|kappa_w| + h|kappa_h| < 1`` binds on. The
    old code used ``half_n*n^2 + half_b*b^2`` (a quadratic axis blend), which is
    optimistic everywhere except axis-aligned bends.
    """

    HALF_N = TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M
    HALF_B = TYPE_KK_OUTER_CHANNEL_HALF_WIDTH_BINORMAL_M

    @staticmethod
    def _reach_for_bend(bend_vector):
        """Single-quadpoint reach for a curvature vector along ``bend_vector``.

        tangent = x, rotated normal = y, rotated binormal = z. The curvature vector
        (``gammadashdash`` projected perpendicular to the tangent) is set to
        ``bend_vector``; its in-plane components are exactly (bend.n, bend.b).
        """
        tangent = np.array([[1.0, 0.0, 0.0]])
        frame_n = np.array([[0.0, 1.0, 0.0]])
        frame_b = np.array([[0.0, 0.0, 1.0]])
        gammadash = tangent
        gammadashdash = np.asarray([bend_vector], dtype=float)
        return float(
            np.asarray(
                _projected_reach_pure(
                    gammadash,
                    gammadashdash,
                    frame_n,
                    frame_b,
                    ProjectedReachSupportFunctionTest.HALF_N,
                    ProjectedReachSupportFunctionTest.HALF_B,
                ),
                dtype=float,
            )[0]
        )

    def test_forty_five_degree_misalignment_is_support_function_not_quadratic(self):
        # bend ~ (y + z): equal projection onto normal and binormal, |n| = |b| =
        # 1/sqrt(2). Exact support function -> (half_n + half_b)/sqrt(2). The old
        # quadratic blend half_n*n^2 + half_b*b^2 returns (half_n + half_b)/2, which
        # is ~41% lower -- THIS is what the regression catches: the assertion below
        # passes only for the support function and fails against the quadratic.
        reach = self._reach_for_bend([0.0, 1.0, 1.0])
        support_function = (self.HALF_N + self.HALF_B) / np.sqrt(2.0)
        old_quadratic = (self.HALF_N + self.HALF_B) / 2.0
        self.assertAlmostEqual(
            reach, support_function, places=9,
            msg="45deg reach is not the exact rectangle support function",
        )
        # Guard: the support-function value is genuinely distinct from the old
        # quadratic, so this test cannot pass against the reverted formula.
        self.assertGreater(reach, old_quadratic + 1e-4)

    def test_axis_aligned_reach_equals_quadratic(self):
        # When the bend is along a single axis the support function and the old
        # quadratic agree (|n|=1,|b|=0 -> half_n in both forms; and vice versa).
        self.assertAlmostEqual(
            self._reach_for_bend([0.0, 1.0, 0.0]), self.HALF_N, places=12,
            msg="normal-axis bend reach must be exactly half_n",
        )
        self.assertAlmostEqual(
            self._reach_for_bend([0.0, 0.0, 1.0]), self.HALF_B, places=12,
            msg="binormal-axis bend reach must be exactly half_b",
        )

    def test_exact_exceeds_quadratic_at_intermediate_angles(self):
        # Strict: at every non-axis-aligned angle the support function reports a
        # LARGER (less optimistic) reach than the quadratic blend.
        for theta in np.linspace(0.05, np.pi / 2.0 - 0.05, 9):
            bend = [0.0, float(np.cos(theta)), float(np.sin(theta))]
            reach = self._reach_for_bend(bend)
            n_comp = np.cos(theta)
            b_comp = np.sin(theta)
            quadratic = self.HALF_N * n_comp**2 + self.HALF_B * b_comp**2
            self.assertGreater(
                reach, quadratic + 1e-9,
                f"support function not strictly above quadratic at theta={theta:.3f}",
            )


class GradientFinitenessTest(unittest.TestCase):
    """``dJ()`` returns finite values on a real active-hinge pack.

    The reach uses ``jnp.abs``, which has a benign subgradient kink at projection
    0; this pins that the assembled gradient (both the kappa and alpha dof paths)
    stays finite -- no NaN/inf leaks from the kink into the descent.
    """

    def test_dJ_is_finite_on_active_pack(self):
        curve = _banana_curve()
        frame = _shared_frame(curve)
        rot = frame.rotation
        x = np.asarray(rot.x, dtype=float).copy()
        x[0] = 0.35
        x[1] = 0.18
        rot.x = x
        pen = _penalty(frame, curve)
        self.assertGreater(pen.J(), 0.0)  # the hinge is genuinely active
        grad = np.asarray(pen.dJ(), dtype=float)
        self.assertTrue(
            np.all(np.isfinite(grad)),
            f"dJ produced non-finite entries: {grad[~np.isfinite(grad)]}",
        )
        # The gradient also projects finitely onto each dof group independently.
        grad_curve = np.asarray(pen.dJ(partials=True)(curve), dtype=float)
        grad_alpha = np.asarray(pen.dJ(partials=True)(rot), dtype=float)
        self.assertTrue(np.all(np.isfinite(grad_curve)))
        self.assertTrue(np.all(np.isfinite(grad_alpha)))


if __name__ == "__main__":
    unittest.main()
