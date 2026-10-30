import copy
import json
import pickle
import unittest
from unittest import mock
from numpy.random import PCG64DXSM, Generator

import numpy as np

from simsopt.geo.curveperturbed import GaussianSampler, PerturbationSample, CurvePerturbed
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.curveobjectives import LpCurveTorsion, CurveCurveDistance
from simsopt._core.json import GSONDecoder, GSONEncoder, SIMSON


def circle(points, radius):
    """Planar circle of ``radius`` (m) on the quadrature ``points``."""
    curve = CurveXYZFourier(points, 2)
    curve.set("xc(1)", radius)
    curve.set("ys(1)", radius)
    return curve


class CurvePerturbationTesting(unittest.TestCase):

    def test_perturbed_gammadash(self):
        sigma = 1
        length_scale = 0.5
        points = np.linspace(0, 1, 200, endpoint=False)
        sampler = GaussianSampler(points, sigma, length_scale, n_derivs=2)
        rg = Generator(PCG64DXSM(1))
        sample = PerturbationSample(sampler, randomgen=rg)

        dphi = points[1]

        for idx in range(2):
            g = sample[idx + 0]
            gd = sample[idx + 1]

            gdest = (-1/12) * g[4:, :] + (2/3) * g[3:-1, :] + 0 * g[2:-2, :] + (-2/3) * g[1:-3, :] + (1/12) * g[0:-4, :]
            gdest *= 1/dphi
            err = np.abs(gdest - gd[2:-2, :])

            print("np.mean(err)", np.mean(err))
            if idx == 0:
                assert np.mean(err) < 3e-4
            else:
                assert np.mean(err) < 2e-3

    def test_perturbed_periodic(self):
        sigma = 1
        length_scale = 0.5
        n = 100
        points = np.linspace(0, 2, 2*n, endpoint=False)
        sampler = GaussianSampler(points, sigma, length_scale, n_derivs=0)
        rg = Generator(PCG64DXSM(1))
        sample = PerturbationSample(sampler, randomgen=rg)
        periodic_err = np.abs(sample[0][:n, :] - sample[0][n:, :])
        print("periodic_err", np.mean(periodic_err))
        assert np.mean(periodic_err) < 1e-6

    def test_perturbed_objective_torsion(self):
        # test the torsion objective as that covers all derivatives (up to
        # third) of a curve
        sigma = 1
        length_scale = 0.5
        points = np.linspace(0, 1, 200, endpoint=False)
        sampler = GaussianSampler(points, sigma, length_scale, n_derivs=3)
        rg = Generator(PCG64DXSM(1))
        sample = PerturbationSample(sampler, randomgen=rg)

        order = 4
        nquadpoints = 200
        curve = CurveXYZFourier(nquadpoints, order)
        dofs = np.zeros((curve.dof_size, ))
        dofs[1] = 1.
        dofs[2*order+3] = 1.
        dofs[4*order+3] = 1.
        curve.x = dofs

        curve = CurvePerturbed(curve, sample)

        J = LpCurveTorsion(curve, p=2)
        J0 = J.J()
        curve_dofs = curve.x
        h = 1e-3 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
        dJ = J.dJ()
        deriv = np.sum(dJ * h)
        assert np.abs(deriv) > 1e-10
        err = 1e6
        for i in range(10, 20):
            eps = 0.5**i
            curve.x = curve_dofs + eps * h
            Jh = J.J()
            deriv_est = (Jh-J0)/eps
            err_new = np.linalg.norm(deriv_est-deriv)
            print("err_new %s" % (err_new))
            assert err_new < 0.55 * err
            err = err_new

    def test_perturbed_objective_distance(self):
        # test the distance objective as that covers the position and the first
        # derivative of the curve
        sigma = 1
        length_scale = 0.2
        points = np.linspace(0, 1, 200, endpoint=False)
        sampler = GaussianSampler(points, sigma, length_scale, n_derivs=1)
        rg = Generator(PCG64DXSM(1))
        sample1 = PerturbationSample(sampler, randomgen=rg)
        sample2 = PerturbationSample(sampler, randomgen=rg)

        order = 4
        nquadpoints = 200
        curve1 = CurveXYZFourier(nquadpoints, order)
        dofs = np.zeros((curve1.dof_size, ))
        dofs[1] = 1.
        dofs[2*order+3] = 1.
        dofs[4*order+3] = 1.
        curve1.x = dofs

        curve2 = CurveXYZFourier(nquadpoints, order)
        dofs = np.zeros((curve2.dof_size, ))
        dofs[1] = 2.
        dofs[2*order+3] = 2.
        dofs[4*order+3] = 2.
        curve2.x = dofs

        curve1 = CurvePerturbed(curve1, sample1)
        curve2 = CurvePerturbed(curve2, sample2)

        J = CurveCurveDistance([curve1, curve2], 2.0)
        J0 = J.J()
        curve1.resample()
        assert J0 != J.J()
        J0 = J.J()
        curve_dofs = J.x
        h = 1e-3 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
        dJ = J.dJ()
        deriv = np.sum(dJ * h)
        assert np.abs(deriv) > 1e-10
        err = 1e6
        for i in range(2, 10):
            eps = 0.5**i
            J.x = curve_dofs + eps * h
            Jh = J.J()
            deriv_est = (Jh-J0)/eps
            err_new = np.linalg.norm(deriv_est-deriv)
            # print("err_new %s" % (err_new))
            assert err_new < 0.55 * err
            err = err_new

    def test_serialization(self):
        """GSON round trip keeps the geometry, and the rebuilt curve follows a resample of its rebuilt sample."""
        sigma = 1
        length_scale = 0.5
        points = np.linspace(0, 1, 200, endpoint=False)
        sampler = GaussianSampler(points, sigma, length_scale, n_derivs=2)
        sample = PerturbationSample(sampler)

        order = 4
        nquadpoints = 200
        curve = CurveXYZFourier(nquadpoints, order)
        dofs = np.zeros((curve.dof_size,))
        dofs[1] = 1.
        dofs[2 * order + 3] = 1.
        dofs[4 * order + 3] = 1.
        curve.x = dofs
        curve_per = CurvePerturbed(curve, sample)

        curve_str = json.dumps(SIMSON(curve_per), cls=GSONEncoder)
        curve_per_regen = json.loads(curve_str, cls=GSONDecoder)
        self.assertTrue(np.allclose(curve_per.gamma(), curve_per_regen.gamma()))
        # The rebuilt curve is registered with its rebuilt sample.
        curve_per_regen.resample()
        np.testing.assert_allclose(curve_per_regen.gamma(),
                                   curve_per_regen.curve.gamma() + curve_per_regen.sample[0], rtol=0, atol=1e-15)

    def test_resample_refreshes_every_curve_sharing_the_sample(self):
        """Resampling through one curve moves every curve built on that sample.

        Curves sharing a PerturbationSample model one systematic error applied to
        several coils (e.g. the filaments of a finite-build coil), so after a new
        draw each curve must equal its base curve plus the new perturbation,
        including a curve that wraps a sharing curve.
        """
        points = np.linspace(0, 1, 64, endpoint=False)
        sampler = GaussianSampler(points, 0.05, 0.3, n_derivs=1)
        sample = PerturbationSample(sampler, randomgen=Generator(PCG64DXSM(3)))
        outer_sample = PerturbationSample(sampler, randomgen=Generator(PCG64DXSM(4)))
        base1, base2 = circle(points, 1.0), circle(points, 1.5)
        curve1 = CurvePerturbed(base1, sample)
        curve2 = CurvePerturbed(base2, sample)
        nested = CurvePerturbed(curve2, outer_sample)
        for c in (curve1, curve2, nested):
            c.gamma()
            c.gammadash()

        curve1.resample()
        np.testing.assert_allclose(curve2.gamma(), base2.gamma() + sample[0], rtol=0, atol=1e-15)
        np.testing.assert_allclose(curve2.gammadash(), base2.gammadash() + sample[1], rtol=0, atol=1e-14)
        np.testing.assert_allclose(nested.gamma(), base2.gamma() + sample[0] + outer_sample[0],
                                   rtol=0, atol=1e-15)

        sample.resample()
        np.testing.assert_allclose(curve1.gamma(), base1.gamma() + sample[0], rtol=0, atol=1e-15)

    def test_curves_with_equal_names_both_follow_the_sample(self):
        """Curves that compare equal stay separate consumers of a sample.

        ``Optimizable.__eq__`` compares names, so a renamed curve can equal another
        one. Each must still move on a new draw, and registering a curve twice
        must not notify it twice.
        """
        points = np.linspace(0, 1, 64, endpoint=False)
        sampler = GaussianSampler(points, 0.05, 0.3, n_derivs=1)
        sample = PerturbationSample(sampler, randomgen=Generator(PCG64DXSM(5)))
        other_sample = PerturbationSample(sampler, randomgen=Generator(PCG64DXSM(6)))
        base1, base2 = circle(points, 1.0), circle(points, 1.5)
        curve1 = CurvePerturbed(base1, sample)
        curve2 = CurvePerturbed(base2, other_sample)
        curve2.name = curve1.name
        self.assertEqual(curve1, curve2)

        curve2.sample = sample
        sample.add_consumer(curve2)
        curve1.gamma()
        curve2.gamma()
        with mock.patch.object(curve2, "set_recompute_flag", wraps=curve2.set_recompute_flag) as invalidate:
            sample.resample()
        invalidate.assert_called_once()
        np.testing.assert_allclose(curve1.gamma(), base1.gamma() + sample[0], rtol=0, atol=1e-15)
        np.testing.assert_allclose(curve2.gamma(), base2.gamma() + sample[0], rtol=0, atol=1e-15)

    def test_sample_reassignment_moves_the_curve(self):
        """A curve given a new sample takes its draw at once and follows only that sample."""
        points = np.linspace(0, 1, 64, endpoint=False)
        sampler = GaussianSampler(points, 0.05, 0.3, n_derivs=1)
        old_sample = PerturbationSample(sampler, randomgen=Generator(PCG64DXSM(7)))
        new_sample = PerturbationSample(sampler, randomgen=Generator(PCG64DXSM(8)))
        base = circle(points, 1.0)
        curve = CurvePerturbed(base, old_sample)
        curve.gamma()

        curve.sample = new_sample
        np.testing.assert_allclose(curve.gamma(), base.gamma() + new_sample[0], rtol=0, atol=1e-15)
        new_sample.resample()
        np.testing.assert_allclose(curve.gamma(), base.gamma() + new_sample[0], rtol=0, atol=1e-15)
        with mock.patch.object(curve, "set_recompute_flag", wraps=curve.set_recompute_flag) as invalidate:
            old_sample.resample()
        invalidate.assert_not_called()

    def test_copied_sample_keeps_the_draw_but_not_the_curves(self):
        """A pickled or copied sample has its own registry and its own current draw.

        The copy starts with the same draw. Curves built on the copy follow its
        resamples, and curves built on the original keep the original draw.
        ``copy.copy`` shares the random generator with the original, so the
        two draw from one stream; pickle and ``copy.deepcopy`` copy it.
        """
        points = np.linspace(0, 1, 64, endpoint=False)
        sampler = GaussianSampler(points, 0.05, 0.3, n_derivs=1)
        sample = PerturbationSample(sampler, randomgen=Generator(PCG64DXSM(5)))
        base = circle(points, 1.0)
        curve = CurvePerturbed(base, sample)
        gamma = curve.gamma().copy()
        copiers = {"pickle": lambda s: pickle.loads(pickle.dumps(s)), "copy": copy.copy, "deepcopy": copy.deepcopy}
        for name, copier in copiers.items():
            with self.subTest(copier=name):
                duplicate = copier(sample)
                np.testing.assert_array_equal(duplicate[0], sample[0])
                on_duplicate = CurvePerturbed(base, duplicate)
                on_duplicate.gamma()
                duplicate.resample()
                self.assertFalse(np.allclose(duplicate[0], sample[0]))
                np.testing.assert_allclose(on_duplicate.gamma(), base.gamma() + duplicate[0], rtol=0, atol=1e-15)
                np.testing.assert_array_equal(curve.gamma(), gamma)
