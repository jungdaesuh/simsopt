import unittest
import json
import numpy as np

from simsopt._core.json import GSONEncoder, GSONDecoder, SIMSON
from simsopt.geo.curvesuperellipse import CurveSuperEllipse
from simsopt.geo.curve import RotatedCurve
from simsopt.geo import parameters

from .test_curve import taylor_test

parameters['jit'] = False


def get_se_curve(rotated, x=(np.arange(32)+0.5)/32):
    """Mirror of get_curve() in test_curve.py, for CurveSuperEllipse."""
    np.random.seed(2)
    rand_scale = 0.01
    curve = CurveSuperEllipse(x, 0.1, 0.15, 4.0)
    dofs = np.asarray(curve.x).copy()
    dofs[7] = 1.0    # X, so the curve is off the origin
    curve.x = dofs + rand_scale * np.random.rand(len(dofs)).reshape(dofs.shape)
    if rotated:
        curve = RotatedCurve(curve, 0.5, flip=False)
    return curve


class SuperEllipseTesting(unittest.TestCase):

    def test_make_names(self):
        # Names, count, ordering, and the default placement, which must be the
        # identity so the curve is immediately usable and lies in z = 0.
        expected = ['a', 'b', 'n', 'q0', 'qi', 'qj', 'qk', 'X', 'Y', 'Z']
        curve = CurveSuperEllipse(32, 0.1, 0.15, 4.0)
        self.assertEqual(curve._make_names(), expected)
        self.assertEqual(curve.local_dof_names, expected)
        self.assertEqual(curve.num_dofs(), 10)
        np.testing.assert_allclose(
            curve.x, [0.1, 0.15, 4.0, 1., 0., 0., 0., 0., 0., 0.])
        np.testing.assert_allclose(curve.gamma()[:, 2], 0, atol=1e-15)

        for name, val in [('b', 0.2), ('n', 3.0), ('X', 1.0), ('Y', 2.0), ('Z', 3.0)]:
            curve.set(name, val)
        np.testing.assert_allclose(np.mean(curve.gamma(), axis=0),
                                   [1.0, 2.0, 3.0], atol=1e-14)

    def test_init_validates_shape_arguments(self):
        for a, b, n in [(-0.1, 0.15, 4.0), (0.1, 0.0, 4.0),
                        (0.1, 0.15, 1.0), (0.1, 0.15, 0.5)]:
            with self.subTest(a=a, b=b, n=n):
                with self.assertRaises(ValueError):
                    CurveSuperEllipse(32, a, b, n)

    def subtest_implicit_equation(self, a, b, n, nquadpoints):
        curve = CurveSuperEllipse(nquadpoints, a, b, n)
        g = curve.gamma()
        residual = np.abs(g[:, 0]/a)**n + np.abs(g[:, 1]/b)**n - 1
        np.testing.assert_allclose(residual, 0, atol=1e-13)
        np.testing.assert_allclose(g[:, 2], 0, atol=1e-14)
        # Pointwise evaluation reproduces the semi-axes exactly, unlike a
        # truncated Fourier fit of the same shape.
        self.assertAlmostEqual(np.abs(g[:, 0]).max(), a, places=4)
        self.assertAlmostEqual(np.abs(g[:, 1]).max(), b, places=4)

    def test_implicit_equation(self):
        # The last case is a == b, n == 2, i.e. a circle.
        for a, b, n, nq in [(0.1, 0.15, 4.0, 64), (0.1, 0.15, 2.7, 66),
                            (0.2, 0.03, 3.3, 128), (0.3, 0.3, 2.0, 64)]:
            with self.subTest(a=a, b=b, n=n, nquadpoints=nq):
                self.subtest_implicit_equation(a, b, n, nq)

    def test_area(self):
        # Against the shoelace formula, and monotone in n up to the 4ab limit.
        a, b = 0.1, 0.15
        areas = []
        for n in [2.0, 3.0, 4.0, 6.0, 12.0]:
            with self.subTest(n=n):
                curve = CurveSuperEllipse((np.arange(8192)+0.5)/8192, a, b, n)
                g = curve.gamma()
                x, y = g[:, 0], g[:, 1]
                shoelace = 0.5*np.abs(np.sum(x*np.roll(y, -1) - np.roll(x, -1)*y))
                self.assertAlmostEqual(curve.area()/shoelace, 1.0, places=5)
                areas.append(curve.area())
        self.assertTrue(np.all(np.diff(areas) > 0))
        self.assertAlmostEqual(areas[0], np.pi*a*b, places=12)
        self.assertLess(areas[-1], 4*a*b)

    def test_curvature_vs_exponent(self):
        # n > 1 keeps the curvature bounded and it blows up as n -> 1. For
        # n > 2 the shape is locally x ~ a - C|y|^n near a semi-axis, so the
        # curvature vanishes there; approach off-axis to see the true limit.
        curve = CurveSuperEllipse((np.arange(512)+0.5)/512, 0.1, 0.15, 2.0)
        kappas = []
        for n in [1.2, 1.5, 2.0, 3.0, 4.0]:
            curve.x = np.array([0.1, 0.15, n, 1., 0., 0., 0., 0., 0., 0.])
            kappas.append(curve.kappa().max())
        self.assertLess(kappas[-1], 1e3)
        self.assertGreater(kappas[0], kappas[2])

        near_axis = np.array([1e-7, 1e-5, 1e-3])/(2*np.pi)
        for n, bound in [(3.0, 1e-1), (4.0, 1e-2)]:
            with self.subTest(n=n):
                self.assertLess(
                    CurveSuperEllipse(near_axis, 0.1, 0.15, n).kappa().max(), bound)
        np.testing.assert_allclose(
            CurveSuperEllipse(near_axis, 0.1, 0.15, 2.0).kappa(),
            0.1/0.15**2, rtol=1e-3)

    def test_theta_derivatives_on_semi_axes(self):
        # The default grid always includes quadpoint 0.0, which lies on a
        # semi-axis. The branch-free floor in superellipse_rho keeps the
        # theta-derivatives correct there; a where-based guard flattens them
        # and gives kappa = 0 instead.
        for nquadpoints in [32, 64, 128]:
            with self.subTest(nquadpoints=nquadpoints):
                curve = CurveSuperEllipse(nquadpoints, 0.3, 0.3, 2.0)
                np.testing.assert_allclose(curve.kappa(), 1/0.3, rtol=1e-12)
                self.assertTrue(np.all(np.isfinite(curve.dkappa_by_dcoeff())))

    def test_no_nan_when_quadpoints_on_semi_axes(self):
        # Same grid points, but now checking d/dn, which evaluates log(0) there.
        for nquadpoints in [32, 64, 128]:
            with self.subTest(nquadpoints=nquadpoints):
                curve = CurveSuperEllipse(nquadpoints, 0.1, 0.15, 4.0)
                theta = 2*np.pi*np.asarray(curve.quadpoints)
                on_axis = np.isclose(np.cos(theta), 0, atol=1e-15).sum() \
                    + np.isclose(np.sin(theta), 0, atol=1e-15).sum()
                self.assertGreater(on_axis, 0)
                for arr in [curve.gamma(), curve.gammadash(), curve.gammadashdash(),
                            curve.dgamma_by_dcoeff(), curve.dgammadash_by_dcoeff()]:
                    self.assertFalse(np.any(np.isnan(arr)))

    def subtest_coefficient_derivative(self, rotated):
        curve = get_se_curve(rotated)
        coeffs = curve.x
        curve.invalidate_cache()

        for gamma_fn, dgamma_fn in [
                (curve.gamma, curve.dgamma_by_dcoeff),
                (curve.gammadash, curve.dgammadash_by_dcoeff),
                (curve.gammadashdash, curve.dgammadashdash_by_dcoeff),
                (curve.gammadashdashdash, curve.dgammadashdashdash_by_dcoeff)]:
            def f(dofs, gamma_fn=gamma_fn):
                curve.x = dofs
                return gamma_fn().copy()

            def df(dofs, dgamma_fn=dgamma_fn):
                curve.x = dofs
                return dgamma_fn().copy()
            taylor_test(f, df, coeffs)

    def test_coefficient_derivative(self):
        for rotated in [True, False]:
            with self.subTest(rotated=rotated):
                self.subtest_coefficient_derivative(rotated)

    def subtest_kappa_derivative(self, rotated):
        # n = 2 is used here because for n > 2 the curvature vanishes exactly
        # on the semi-axes, where dkappa_by_dcoeff divides by the curvature.
        curve = get_se_curve(rotated)
        curve.x = np.concatenate([[0.1, 0.15, 2.0], np.asarray(curve.x)[3:]])
        coeffs = curve.x

        def f(dofs):
            curve.x = dofs
            return curve.kappa().copy()

        def df(dofs):
            curve.x = dofs
            return curve.dkappa_by_dcoeff().copy()
        taylor_test(f, df, coeffs)

    def test_kappa_derivative(self):
        for rotated in [True, False]:
            with self.subTest(rotated=rotated):
                self.subtest_kappa_derivative(rotated)

    def test_serialization(self):
        for rotated in [True, False]:
            with self.subTest(rotated=rotated):
                curve = get_se_curve(rotated)
                curve_json_str = json.dumps(SIMSON(curve), cls=GSONEncoder, indent=2)
                curve_regen = json.loads(curve_json_str, cls=GSONDecoder)
                self.assertTrue(np.allclose(curve.gamma(), curve_regen.gamma()))

    def test_shared_dofs(self):
        curve1 = CurveSuperEllipse(64, 0.1, 0.15, 4.0)
        curve1.set('X', 1.0)
        curve2 = CurveSuperEllipse(64, 0.1, 0.15, 4.0, dofs=curve1.dofs)
        np.testing.assert_allclose(curve1.gamma(), curve2.gamma())
        curve1.set('a', 0.2)
        np.testing.assert_allclose(curve1.gamma(), curve2.gamma())

    def test_curvature_penalty_gradient(self):
        # safe_kappa_pure keeps dkappa_by_dcoeff_vjp finite where the
        # curvature vanishes, which for n > 2 is the four semi-axes. Without
        # it LpCurveCurvature.dJ() is nan at every exponent above 2.
        from simsopt.geo import LpCurveCurvature
        for n in [2.0, 3.0, 4.0, 6.0]:
            with self.subTest(n=n):
                curve = CurveSuperEllipse(32, 0.04, 0.06, n)
                J = LpCurveCurvature(curve, 2)
                self.assertTrue(np.all(np.isfinite(J.dJ())))

        curve = CurveSuperEllipse(32, 0.04, 0.06, 4.0)
        J = LpCurveCurvature(curve, 2)

        def f(dofs):
            curve.x = dofs
            return J.J()

        def df(dofs):
            curve.x = dofs
            return J.dJ()
        taylor_test(f, df, curve.x.copy())


if __name__ == "__main__":
    unittest.main()
