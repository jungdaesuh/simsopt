import ast
import unittest
from pathlib import Path
import tracemalloc
from unittest import mock

import numpy as np
from scipy.special import logsumexp

from simsopt._core import Optimizable
from simsopt.geo.curveobjectives import CurveCurveDistance
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.surfacerzfourier import SurfaceRZFourier

from simsopt_alm import signed_constraints
from simsopt_alm.signed_constraints import (
    smooth_max_curvature_signed_constraint,
    smooth_min_curve_curve_signed_constraint,
    smooth_min_curve_surface_signed_constraint,
)


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


class SignedConstraintTemperatureTests(unittest.TestCase):
    """Every kernel takes a finite positive temperature and rejects any other
    with a ValueError naming it (zero is not the hard limit: the hard signal
    is the third return value)."""

    INVALID = (-0.01, 0.0, float("nan"), float("inf"), -float("inf"))

    def _kernels(self):
        curves = [_circle(1.0, 0.0), _circle(1.0, 0.5)]
        surface = _torus(1.0, 0.3)
        curve = _circle_with_bump()
        return {
            "curve_curve": lambda t: smooth_min_curve_curve_signed_constraint(
                curves, 0.1, t, _JointDofs(curves)
            ),
            "curve_surface": lambda t: smooth_min_curve_surface_signed_constraint(
                [_circle(0.1, 0.0)], surface, 0.4, t, _JointDofs([surface])
            ),
            "curvature": lambda t: smooth_max_curvature_signed_constraint(
                curve, 1.5, t, _JointDofs([curve])
            ),
        }

    def test_an_invalid_temperature_is_rejected_by_name(self):
        for name, kernel in self._kernels().items():
            for temperature in self.INVALID:
                with self.subTest(kernel=name, temperature=temperature):
                    with self.assertRaisesRegex(ValueError, "temperature"):
                        kernel(temperature)

    def test_a_finite_positive_temperature_is_accepted(self):
        for name, kernel in self._kernels().items():
            with self.subTest(kernel=name):
                signed_value, grad, hard_signed_value = kernel(1.0e-3)
                self.assertTrue(np.isfinite(signed_value))
                self.assertTrue(np.all(np.isfinite(grad)))
                self.assertGreaterEqual(signed_value, hard_signed_value)


class _SupportEdgeCase:
    """A kernel, its dofs owner and one dof, at a temperature that puts a sample
    exactly at ``4 T`` from the extremum (the edge of the support window the
    kernels once truncated to, where they were discontinuous)."""

    def __init__(self, evaluate, owner, curve, dof_name):
        self.evaluate = evaluate
        self.owner = owner
        self.index = list(owner.dof_names).index(f"{curve.name}:{dof_name}")

    def at_offset(self, offset):
        x0 = self.owner.x.copy()
        moved = x0.copy()
        moved[self.index] += offset
        self.owner.x = moved
        try:
            return self.evaluate()
        finally:
            self.owner.x = x0


def _bumped_circle(radius, z, count=32, bump=0.04):
    curve = CurveXYZFourier(count, 2)
    curve.set("xc(1)", radius)
    curve.set("ys(1)", radius)
    curve.set("zc(0)", z)
    curve.set("xc(2)", bump)
    return curve


def _support_edge_cases():
    # Codex round-16 R16-02 reproduction geometry.
    curvature_curve = _bumped_circle(1.0, 0.0, count=64, bump=0.1)
    kappa = curvature_curve.kappa().copy()
    curvature_temperature = float((np.max(kappa) - kappa[13]) / 4)

    first, second = _bumped_circle(1.0, 0.0), _bumped_circle(0.9, 0.3)
    pair_distances = np.linalg.norm(
        first.gamma()[:, None, :] - second.gamma()[None, :, :], axis=2
    ).ravel()
    pair_temperature = float(
        (np.sort(pair_distances)[len(pair_distances) // 5] - np.min(pair_distances)) / 4
    )

    surface_curve = _bumped_circle(1.7, 0.07)
    surface_curve.set("yc(0)", 0.03)
    surface = SurfaceRZFourier(
        nfp=1,
        stellsym=True,
        mpol=1,
        ntor=1,
        quadpoints_phi=np.linspace(0, 1, 16, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, 16, endpoint=False),
    )
    surface.set("rc(0,0)", 1.0)
    surface.set("rc(1,0)", 0.3)
    surface.set("zs(1,0)", 0.3)
    surface.fix_all()
    surface_distances = np.linalg.norm(
        surface_curve.gamma()[:, None, :] - surface.gamma().reshape(-1, 3)[None, :, :],
        axis=2,
    ).ravel()
    surface_temperature = float(
        (np.sort(surface_distances)[200] - np.min(surface_distances)) / 4
    )

    curvature_owner = _JointDofs([curvature_curve])
    pair_owner = _JointDofs([first, second])
    surface_owner = _JointDofs([surface_curve])
    return {
        "curvature": _SupportEdgeCase(
            lambda: smooth_max_curvature_signed_constraint(
                curvature_curve, 1.5, curvature_temperature, curvature_owner
            ),
            curvature_owner,
            curvature_curve,
            "xc(2)",
        ),
        "curve_curve": _SupportEdgeCase(
            lambda: smooth_min_curve_curve_signed_constraint(
                [first, second], 0.4, pair_temperature, pair_owner
            ),
            pair_owner,
            second,
            "xc(1)",
        ),
        "curve_surface": _SupportEdgeCase(
            lambda: smooth_min_curve_surface_signed_constraint(
                [surface_curve], surface, 0.4, surface_temperature, surface_owner
            ),
            surface_owner,
            surface_curve,
            "xc(1)",
        ),
    }


class SupportChangeSmoothnessTests(unittest.TestCase):
    """The rows are smooth where a sample crosses 4 T from the extremum: the
    value is continuous there and the gradient matches finite differences."""

    def test_the_value_is_continuous_across_the_support_edge(self):
        for name, case in _support_edge_cases().items():
            with self.subTest(kernel=name):
                gradient = case.at_offset(0.0)[1][case.index]
                for step in (1.0e-8, 1.0e-10):
                    jump = case.at_offset(step)[0] - case.at_offset(-step)[0]
                    self.assertLessEqual(
                        abs(jump - 2.0 * step * gradient),
                        1.0e-12,
                        f"the {name} row jumps by {jump:.3e} over a {2 * step:.0e} "
                        "move: it is not continuous there",
                    )

    def test_the_gradient_matches_finite_differences_across_the_support_edge(self):
        for name, case in _support_edge_cases().items():
            gradient = case.at_offset(0.0)[1][case.index]
            for step in (1.0e-4, 1.0e-6, 1.0e-8):
                with self.subTest(kernel=name, step=step):
                    finite_difference = (
                        case.at_offset(step)[0] - case.at_offset(-step)[0]
                    ) / (2.0 * step)
                    self.assertAlmostEqual(
                        finite_difference,
                        gradient,
                        delta=1.0e-5 * max(1.0, abs(gradient)),
                        msg=f"the {name} gradient disagrees with a central "
                        f"difference of step {step:.0e}",
                    )


class FullLogSumExpTests(unittest.TestCase):
    def test_distance_rows_are_the_log_sum_exp_over_every_pair(self):
        curves = [_circle(1.0, 0.0, 16), _circle(0.9, 0.3, 12), _circle(1.1, -0.4, 8)]
        surface = _fixed_torus(1.0, 0.3)
        temperature = 0.05
        points = [np.asarray(c.gamma()) for c in curves]
        surface_points = surface.gamma().reshape((-1, 3))

        def reference(pairs, minimum_distance):
            distances = np.concatenate(
                [np.linalg.norm(a[:, None] - b[None], axis=2).ravel() for a, b in pairs]
            )
            soft = -temperature * logsumexp(-distances / temperature)
            return minimum_distance - soft, minimum_distance - np.min(distances)

        curve_pairs = [(points[i], points[j]) for i in range(3) for j in range(i)]
        signed, _grad, hard = smooth_min_curve_curve_signed_constraint(
            curves, 0.4, temperature, _JointDofs(curves)
        )
        np.testing.assert_allclose((signed, hard), reference(curve_pairs, 0.4), rtol=1e-13)

        surface_pairs = [(p, surface_points) for p in points]
        signed, _grad, hard = smooth_min_curve_surface_signed_constraint(
            curves, surface, 0.4, temperature, _JointDofs(curves)
        )
        np.testing.assert_allclose(
            (signed, hard), reference(surface_pairs, 0.4), rtol=1e-13
        )

    def test_curvature_row_is_the_log_sum_exp_over_every_point(self):
        curve = _circle_with_bump()
        kappa = np.asarray(curve.kappa())
        signed, _grad, hard = smooth_max_curvature_signed_constraint(
            curve, 1.5, 0.05, _JointDofs([curve])
        )
        self.assertAlmostEqual(signed, 0.05 * logsumexp(kappa / 0.05) - 1.5, places=13)
        self.assertEqual(hard, float(np.max(kappa)) - 1.5)

    def test_blocking_does_not_change_the_result(self):
        # Blocks of 7 pairs split rows and columns unevenly and move the
        # running shift many times; the sums are the same up to rounding.
        curves = [_circle(1.0, 0.0, 16), _circle(0.9, 0.3, 12)]
        surface = _fixed_torus(1.0, 0.3)
        owner = _JointDofs(curves)
        kernels = {
            "curve_curve": lambda: smooth_min_curve_curve_signed_constraint(
                curves, 0.4, 0.05, owner
            ),
            "curve_surface": lambda: smooth_min_curve_surface_signed_constraint(
                curves, surface, 0.4, 0.05, owner
            ),
        }
        for name, kernel in kernels.items():
            with self.subTest(kernel=name):
                whole = kernel()
                with mock.patch.object(signed_constraints, "_PAIR_BLOCK", 7):
                    blocked = kernel()
                self.assertEqual(blocked[2], whole[2])
                self.assertAlmostEqual(blocked[0], whole[0], delta=1e-14)
                np.testing.assert_allclose(blocked[1], whole[1], rtol=1e-12, atol=1e-15)

    def test_coincident_samples_give_a_finite_gradient(self):
        # Two unit circles in perpendicular planes share the sample (1, 0, 0):
        # that pair has distance 0 and no direction, and contributes nothing.
        xy = _circle(1.0, 0.0, 8)
        xz = CurveXYZFourier(8, 1)
        xz.set("xc(1)", 1.0)
        xz.set("zs(1)", 1.0)
        owner = _JointDofs([xy, xz])
        signed, grad, hard = smooth_min_curve_curve_signed_constraint(
            [xy, xz], 0.1, 0.01, owner
        )
        self.assertEqual(hard, 0.1)
        self.assertGreaterEqual(signed, hard)
        self.assertTrue(np.all(np.isfinite(grad)))

    def test_memory_stays_bounded_at_a_realistic_size(self):
        # 4 coils of 128 points against a 64 x 64 surface: 2.1 million pairs,
        # whose differences alone would take 50 MB if built at once.
        curves = [_circle(1.0 + 0.05 * k, 0.1 * k, 128) for k in range(4)]
        surface = SurfaceRZFourier(
            nfp=1,
            stellsym=True,
            mpol=1,
            ntor=1,
            quadpoints_phi=np.linspace(0.0, 1.0, 64, endpoint=False),
            quadpoints_theta=np.linspace(0.0, 1.0, 64, endpoint=False),
        )
        surface.set("rc(0,0)", 1.0)
        surface.set("rc(1,0)", 0.3)
        surface.set("zs(1,0)", 0.3)
        surface.fix_all()
        owner = _JointDofs(curves)
        tracemalloc.start()
        try:
            signed, grad, hard = smooth_min_curve_surface_signed_constraint(
                curves, surface, 0.4, 0.005, owner
            )
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 16 * 2**20, f"peak scratch memory {peak} bytes")
        self.assertGreaterEqual(signed, hard)
        self.assertTrue(np.all(np.isfinite(grad)))


class SampledGeometryTests(unittest.TestCase):
    def test_the_hard_value_is_the_extremum_over_the_samples_only(self):
        # Codex R16-05: a thin rotated ellipse (semi-axes 1 and 0.01) sampled
        # at 30 points is feasible for a curvature bound of 15, while 30,000
        # samples of the same coefficients reach a curvature of 1e4.
        phase = np.pi / 30
        curve = CurveXYZFourier(30, 2)
        curve.set("xc(1)", np.cos(phase))
        curve.set("xs(1)", -np.sin(phase))
        curve.set("yc(1)", 0.01 * np.sin(phase))
        curve.set("ys(1)", 0.01 * np.cos(phase))
        dense = CurveXYZFourier(30000, 2)
        dense.x = curve.x
        signed, _grad, hard = smooth_max_curvature_signed_constraint(
            curve, 15.0, 0.05, _JointDofs([curve])
        )
        self.assertEqual(hard, float(np.max(curve.kappa())) - 15.0)
        self.assertLess(signed, 0.0)
        self.assertGreater(float(np.max(dense.kappa())), 1.0e3)

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
    def test_the_rows_import_nothing_from_the_solver(self):
        # The geometry rows serve any optimizer: the module imports simsopt
        # and numpy, and no solver module (the
        # package's __init__ still runs, as for any submodule import).
        tree = ast.parse(Path(signed_constraints.__file__).read_text(encoding="utf-8"))
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0, "signed_constraints imports a solver module")
                roots.add(node.module.split(".")[0])
        self.assertEqual(roots, {"numpy", "simsopt"})


if __name__ == "__main__":
    unittest.main()
