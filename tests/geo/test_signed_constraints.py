import subprocess
import sys
import unittest
from unittest import mock

import numpy as np

from simsopt._core import Optimizable
from simsopt.geo import signed_constraints
from simsopt.geo.curveobjectives import CurveCurveDistance
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.signed_constraints import (
    smooth_max_curvature_signed_constraint,
    smooth_min_curve_curve_signed_constraint,
    smooth_min_curve_surface_signed_constraint,
)
from simsopt.geo.surfacerzfourier import SurfaceRZFourier


def _circle(radius, center_z, quadpoint_count=64):
    curve = CurveXYZFourier(quadpoint_count, 1)
    curve.set("xc(1)", radius)
    curve.set("ys(1)", radius)
    curve.set("zc(0)", center_z)
    return curve


def _torus(major_radius, minor_radius):
    surface = SurfaceRZFourier(
        nfp=1,
        stellsym=True,
        mpol=1,
        ntor=1,
        quadpoints_phi=np.linspace(0.0, 1.0, 32, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 16, endpoint=False),
    )
    surface.set("rc(0,0)", major_radius)
    surface.set("rc(1,0)", minor_radius)
    surface.set("zs(1,0)", minor_radius)
    return surface


def _fixed_torus(major_radius, minor_radius):
    surface = _torus(major_radius, minor_radius)
    surface.fix_all()
    return surface


def _circle_with_bump():
    # A unit circle with a second harmonic, so kappa has a distinct maximum.
    curve = CurveXYZFourier(64, 2)
    curve.set("xc(1)", 1.0)
    curve.set("ys(1)", 1.0)
    curve.set("xc(2)", 0.1)
    return curve


class _JointDofs(Optimizable):
    """Owns no dofs; its free dofs are the union of its parents'."""

    def __init__(self, parents):
        super().__init__(depends_on=parents)


def _central_difference(signed_value_fn, dofs_owner, direction, step=1.0e-6):
    x0 = dofs_owner.x.copy()
    dofs_owner.x = x0 + step * direction
    forward = signed_value_fn()
    dofs_owner.x = x0 - step * direction
    backward = signed_value_fn()
    dofs_owner.x = x0
    return (forward - backward) / (2.0 * step)


class CurveCurveSignedConstraintTests(unittest.TestCase):
    def test_separated_circles_are_strictly_feasible(self):
        # Stock CurveCurveDistance.J() is a hinge: zero whenever the curves are
        # farther apart than the threshold, so it cannot tell slack from
        # active. The signed value keeps the slack: g = d_min - distance < 0.
        curves = [_circle(1.0, 0.0), _circle(1.0, 0.5)]
        minimum_distance = 0.1
        objective = CurveCurveDistance(curves, minimum_distance)

        signed_value, grad, hard_signed_value = (
            smooth_min_curve_curve_signed_constraint(
                curves, minimum_distance, 0.01, objective
            )
        )

        self.assertEqual(objective.J(), 0.0)
        self.assertLess(signed_value, 0.0)
        self.assertAlmostEqual(hard_signed_value, minimum_distance - 0.5, places=12)
        # Smooth min <= hard min, so the surrogate is never looser than the
        # hard signal.
        self.assertGreaterEqual(signed_value, hard_signed_value)
        self.assertEqual(grad.shape, objective.x.shape)
        self.assertTrue(np.all(np.isfinite(grad)))

    def test_single_curve_has_no_pairs_and_is_feasible(self):
        # With fewer than two curves there is no pair to violate the spacing,
        # so the constraint is satisfied: both signals are <= 0 and the
        # gradient is exactly zero.
        curve = _circle(1.0, 0.0)
        objective = CurveCurveDistance([curve], 0.1)

        signed_value, grad, hard_signed_value = (
            smooth_min_curve_curve_signed_constraint([curve], 0.1, 0.01, objective)
        )

        self.assertLessEqual(signed_value, 0.0)
        self.assertLessEqual(hard_signed_value, 0.0)
        np.testing.assert_array_equal(grad, np.zeros_like(objective.x))

    def test_gradient_matches_central_difference(self):
        curves = [_circle(1.0, 0.0), _circle(0.9, 0.3)]
        dofs_owner = _JointDofs(curves)
        direction = np.random.default_rng(0).standard_normal(dofs_owner.x.size)

        _signed, grad, _hard = smooth_min_curve_curve_signed_constraint(
            curves, 0.4, 0.01, dofs_owner
        )
        finite_difference = _central_difference(
            lambda: smooth_min_curve_curve_signed_constraint(
                curves, 0.4, 0.01, dofs_owner
            )[0],
            dofs_owner,
            direction,
        )

        np.testing.assert_allclose(grad @ direction, finite_difference, rtol=1.0e-6)


class CurveSurfaceSignedConstraintTests(unittest.TestCase):
    def test_axis_inside_torus_reports_hard_clearance_and_surface_gradient(self):
        # The circle is the torus axis: every surface point is 0.3 away.
        curve = _circle(1.0, 0.0)
        surface = _torus(1.0, 0.3)
        dofs_owner = _JointDofs([curve, surface])
        direction = np.random.default_rng(1).standard_normal(dofs_owner.x.size)

        signed_value, grad, hard_signed_value = (
            smooth_min_curve_surface_signed_constraint(
                [curve], surface, 0.4, 0.01, dofs_owner
            )
        )
        finite_difference = _central_difference(
            lambda: smooth_min_curve_surface_signed_constraint(
                [curve], surface, 0.4, 0.01, dofs_owner
            )[0],
            dofs_owner,
            direction,
        )

        self.assertAlmostEqual(hard_signed_value, 0.4 - 0.3, places=12)
        self.assertGreaterEqual(signed_value, hard_signed_value)
        self.assertTrue(np.any(grad[curve.x.size :] != 0.0), "surface dofs move g")
        np.testing.assert_allclose(grad @ direction, finite_difference, rtol=1.0e-6)

    def test_no_curves_is_feasible(self):
        surface = _torus(1.0, 0.3)
        dofs_owner = _JointDofs([surface])

        signed_value, grad, hard_signed_value = (
            smooth_min_curve_surface_signed_constraint(
                [], surface, 0.4, 0.01, dofs_owner
            )
        )

        self.assertEqual(signed_value, -0.4)
        self.assertEqual(hard_signed_value, -0.4)
        np.testing.assert_array_equal(grad, np.zeros_like(dofs_owner.x))


class MaxCurvatureSignedConstraintTests(unittest.TestCase):
    def test_circle_reports_hard_excess_and_finite_difference_gradient(self):
        curve = _circle_with_bump()
        dofs_owner = _JointDofs([curve])
        direction = np.random.default_rng(2).standard_normal(dofs_owner.x.size)

        signed_value, grad, hard_signed_value = smooth_max_curvature_signed_constraint(
            curve, 1.5, 0.05, dofs_owner
        )
        finite_difference = _central_difference(
            lambda: smooth_max_curvature_signed_constraint(
                curve, 1.5, 0.05, dofs_owner
            )[0],
            dofs_owner,
            direction,
        )

        self.assertAlmostEqual(
            hard_signed_value, float(np.max(curve.kappa())) - 1.5, places=12
        )
        self.assertGreaterEqual(signed_value, hard_signed_value)
        np.testing.assert_allclose(grad @ direction, finite_difference, rtol=1.0e-5)

    def test_precomputed_kappa_gives_the_same_result(self):
        curve = _circle_with_bump()
        dofs_owner = _JointDofs([curve])
        kappa = curve.kappa().copy()

        reference = smooth_max_curvature_signed_constraint(curve, 1.5, 0.05, dofs_owner)
        with mock.patch.object(
            curve,
            "kappa",
            side_effect=AssertionError("kappa was supplied by the caller"),
        ):
            reused = smooth_max_curvature_signed_constraint(
                curve, 1.5, 0.05, dofs_owner, kappa=kappa
            )

        self.assertEqual(reused[0], reference[0])
        self.assertEqual(reused[2], reference[2])
        np.testing.assert_array_equal(reused[1], reference[1])


class SelectionHelperTests(unittest.TestCase):
    def test_kdtree_pairwise_selection_matches_bruteforce_threshold(self):
        left = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        right = np.array([[0.1, 0.0, 0.0], [2.0, 0.0, 0.0], [3.4, 0.0, 0.0]])

        self.assertAlmostEqual(signed_constraints.pairwise_block_min(left, right), 0.1)
        rows, cols, diffs, distances = signed_constraints.select_pairwise_near_min(
            left,
            right,
            threshold=0.45,
        )

        self.assertEqual(set(zip(rows.tolist(), cols.tolist())), {(0, 0), (2, 2)})
        np.testing.assert_allclose(np.linalg.norm(diffs, axis=1), distances)

    def test_kdtree_pairwise_selection_returns_empty_arrays_when_no_pairs_match(self):
        rows, cols, diffs, distances = signed_constraints.select_pairwise_near_min(
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[2.0, 0.0, 0.0]]),
            threshold=0.5,
        )

        self.assertEqual(rows.tolist(), [])
        self.assertEqual(cols.tolist(), [])
        self.assertEqual(diffs.shape, (0, 3))
        self.assertEqual(distances.tolist(), [])

    def test_kdtree_pairwise_selection_reuses_supplied_trees(self):
        left = np.array([[0.0, 0.0, 0.0]])
        right = np.array([[0.25, 0.0, 0.0]])
        left_tree = signed_constraints.point_tree(left)
        right_tree = signed_constraints.point_tree(right)

        with mock.patch.object(
            signed_constraints,
            "point_tree",
            side_effect=AssertionError("selection should reuse supplied trees"),
        ):
            rows, cols, diffs, distances = signed_constraints.select_pairwise_near_min(
                left,
                right,
                threshold=0.5,
                left_tree=left_tree,
                right_tree=right_tree,
            )

        self.assertEqual(rows.tolist(), [0])
        self.assertEqual(cols.tolist(), [0])
        np.testing.assert_allclose(diffs, [[-0.25, 0.0, 0.0]])
        np.testing.assert_allclose(distances, [0.25])

    def test_surface_tree_cache_reuses_while_the_sampled_geometry_is_unchanged(self):
        class Surface:
            def __init__(self):
                self.x = np.array([0.0])
                self.gamma_calls = 0

            def gamma(self):
                self.gamma_calls += 1
                offset = float(self.x[0])
                return np.array([[[offset, 0.0, 0.0], [offset + 1.0, 0.0, 0.0]]])

        surface = Surface()
        points, tree, shape = signed_constraints.surface_points_tree_shape(surface)
        same_points, same_tree, same_shape = (
            signed_constraints.surface_points_tree_shape(surface)
        )

        # gamma() is sampled on every call (it is the cache key); the tree is not rebuilt.
        self.assertEqual(surface.gamma_calls, 2)
        self.assertIs(same_points, points)
        self.assertIs(same_tree, tree)
        self.assertEqual(same_shape, shape)

        surface.x = np.array([2.0])
        updated_points, updated_tree, updated_shape = (
            signed_constraints.surface_points_tree_shape(surface)
        )

        self.assertEqual(surface.gamma_calls, 3)
        self.assertIsNot(updated_points, points)
        self.assertIsNot(updated_tree, tree)
        self.assertEqual(updated_shape, shape)
        np.testing.assert_allclose(updated_points[:, 0], [2.0, 3.0])

    def test_surface_tree_cache_follows_geometry_the_dofs_do_not_hold(self):
        # The sample points move while x stays put (as a quadrature change
        # would): the cache must follow the sampled geometry, not x.
        class Surface:
            x = np.array([0.0])

            def __init__(self):
                self.sample_offset = 0.0

            def gamma(self):
                return np.array([[[self.sample_offset, 0.0, 0.0], [1.0, 0.0, 0.0]]])

        surface = Surface()
        signed_constraints.surface_points_tree_shape(surface)
        surface.sample_offset = 0.5
        points, tree, _shape = signed_constraints.surface_points_tree_shape(surface)

        np.testing.assert_array_equal(points[:, 0], [0.5, 1.0])
        self.assertEqual(tree.query([0.5, 0.0, 0.0])[0], 0.0)

    def test_surface_tree_cache_follows_a_fixed_coefficient_change(self):
        # All dofs fixed: surface.x is empty and cannot see rc(0,0) move.
        surface = _torus(1.0, 0.1)
        surface.fix_all()
        points, tree, shape = signed_constraints.surface_points_tree_shape(surface)
        old_points = points.copy()

        surface.set("rc(0,0)", 2.0)
        fresh_gamma = np.array(surface.gamma())
        new_points, new_tree, new_shape = signed_constraints.surface_points_tree_shape(
            surface
        )

        np.testing.assert_array_equal(new_points, fresh_gamma.reshape((-1, 3)))
        self.assertEqual(new_shape, fresh_gamma.shape)
        self.assertEqual(new_tree.query(fresh_gamma[0, 0])[0], 0.0)
        # The first call's points and tree are one owned snapshot: recomputing
        # the surface's gamma in place does not move them.
        np.testing.assert_array_equal(points, old_points)
        self.assertEqual(tree.query(old_points[0])[0], 0.0)
        self.assertFalse(points.flags.writeable)

    def test_curve_surface_constraint_sees_a_fixed_coefficient_change(self):
        # A circle of radius 2.1 in the z = 0 plane touches the outboard
        # midplane of the torus (R0 = 2, a = 0.1) and clears the R0 = 1 one.
        surface = _torus(1.0, 0.1)
        surface.fix_all()
        curve = _circle(2.1, 0.0, quadpoint_count=128)
        objective = CurveCurveDistance([curve], 0.1)
        smooth_min_curve_surface_signed_constraint([curve], surface, 0.1, 1e-3, objective)

        surface.set("rc(0,0)", 2.0)
        cached = smooth_min_curve_surface_signed_constraint(
            [curve], surface, 0.1, 1e-3, objective
        )
        fresh = smooth_min_curve_surface_signed_constraint(
            [curve], _fixed_torus(2.0, 0.1), 0.1, 1e-3, objective
        )

        self.assertEqual(cached[2], fresh[2])
        self.assertEqual(cached[0], fresh[0])


class SignedConstraintsImportBoundaryTests(unittest.TestCase):
    def test_import_does_not_load_the_alm_solver(self):
        # Fresh interpreter: other tests in this process may already have
        # imported the solver. The geometry rows serve any optimizer.
        probe = (
            "import sys\n"
            "import simsopt.geo.signed_constraints\n"
            "leaked = sorted(name for name in sys.modules\n"
            "                if name.startswith('simsopt.solve.alm'))\n"
            "print(leaked)\n"
            "sys.exit(1 if leaked else 0)\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"simsopt.geo.signed_constraints loaded the ALM solver: "
            f"{completed.stdout}{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
