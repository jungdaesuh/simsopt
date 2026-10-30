import unittest

import numpy as np
from numpy.random import PCG64DXSM, Generator

from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.curveperturbed import GaussianSampler, PerturbationSample, CurvePerturbed
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import Coil, Current, ScaledCurrent
from simsopt.field.magneticfield import MagneticFieldSum
from simsopt.field.magneticfieldclasses import ToroidalField


def get_curve(num_quadrature_points=200, perturb=False):
    coil = CurveXYZFourier(num_quadrature_points, 3)
    coeffs = coil.dofs_matrix
    coeffs[1][0] = 1.
    coeffs[1][1] = 0.5
    coeffs[2][2] = 0.5
    coil.set_dofs(np.concatenate(coeffs))
    if perturb:
        d = coil.get_dofs()
        coil.set_dofs(d + np.random.uniform(size=d.shape))
    return coil


def circle(radius, z):
    """Horizontal circle of ``radius`` (m) at height ``z`` (m), 64 quadrature points."""
    curve = CurveXYZFourier(64, 2)
    curve.set("xc(1)", radius)
    curve.set("ys(1)", radius)
    curve.set("zc(0)", z)
    return curve


def get_perturbed_coils(seed):
    """Two circular coils (radii 1 m and 1.5 m) sharing one perturbation sample.

    Returns the coils, their base curves, and the shared PerturbationSample.
    """
    sampler = GaussianSampler(np.linspace(0, 1, 64, endpoint=False), 0.05, 0.3, n_derivs=1)
    sample = PerturbationSample(sampler, randomgen=Generator(PCG64DXSM(seed)))
    bases = [circle(1.0, 0.0), circle(1.5, 0.3)]
    coils = [Coil(CurvePerturbed(base, sample), Current(1e5)) for base in bases]
    return coils, bases, sample


def fresh_field(bases, samples, points):
    """B (T, shape (npoints, 3)) of new coils on ``bases``, perturbed by the matching ``samples``."""
    coils = [Coil(CurvePerturbed(base, sample), Current(1e5)) for base, sample in zip(bases, samples)]
    return BiotSavart(coils).set_points(points).B()


RESAMPLE_POINTS = np.array([[0.1, 0.2, 0.05], [0.3, -0.1, 0.2], [1.2, 0.0, 0.1]])


class Testing(unittest.TestCase):

    def assert_field_moved_to(self, B, B_before, B_truth):
        """Assert that ``B`` equals ``B_truth`` and that the change from ``B_before`` is real."""
        # A new draw moves the field by percent-level amounts, so equality is meaningful.
        self.assertGreater(np.max(np.abs(B_truth - B_before)), 1e-3 * np.max(np.abs(B_truth)))
        np.testing.assert_allclose(B, B_truth, rtol=0, atol=1e-13 * np.max(np.abs(B_truth)))

    def test_fully_fixed_field_follows_a_new_draw(self):
        """A field with no free DOF, alone or in a sum, follows a resampled shared perturbation.

        Field-error studies resample coils whose DOFs are all fixed. A draw through
        one coil moves both coils that share the sample, so the cached field must
        equal a field built from scratch with the new draw.
        """
        coils, bases, sample = get_perturbed_coils(seed=2)
        field = BiotSavart(coils)
        field.fix_all()
        background = ToroidalField(1.0, 1.0)
        total = MagneticFieldSum([field, background])
        self.assertEqual(total.dof_size, 0)
        total.set_points(RESAMPLE_POINTS)
        B_before = field.B().copy()
        total_before = total.B().copy()
        background_B = background.B().copy()

        coils[0].curve.resample()
        truth = fresh_field(bases, [sample, sample], RESAMPLE_POINTS)
        self.assert_field_moved_to(total.B(), total_before, truth + background_B)
        self.assert_field_moved_to(field.B(), B_before, truth)

    def test_fully_fixed_field_follows_sample_reassignment(self):
        """A fully fixed field follows a new sample of one coil, and later draws of that sample."""
        coils, bases, sample = get_perturbed_coils(seed=3)
        new_sample = PerturbationSample(sample.sampler, randomgen=Generator(PCG64DXSM(4)))
        field = BiotSavart(coils)
        field.fix_all()
        field.set_points(RESAMPLE_POINTS)
        curve = coils[0].curve
        for change in (lambda: setattr(curve, "sample", new_sample), curve.resample, sample.resample):
            B_before = field.B().copy()
            change()
            self.assert_field_moved_to(field.B(), B_before,
                                       fresh_field(bases, [new_sample, sample], RESAMPLE_POINTS))

    def test_x_assignment_keeps_the_cache_of_a_field_with_no_free_dof(self):
        """Assigning x on a sum spares its fully fixed and DOF-free terms; a new draw does not.

        A background field can be expensive (many coils or magnets), and an x
        assignment on another term gives it nothing new, so its cached B must
        survive. A sentinel written into the native cache through ``B_ref`` stays
        visible only while the cache lives.
        """
        sentinel = -123456.0
        coils, bases, sample = get_perturbed_coils(seed=5)
        fixed = BiotSavart(coils)
        fixed.fix_all()
        dof_free = ToroidalField(1.0, 1.0)
        free = BiotSavart([Coil(circle(0.5, 0.3), Current(1e5))])
        total = MagneticFieldSum([fixed, dof_free, free])
        total.set_points(RESAMPLE_POINTS)
        total.B()
        B_free = free.B().copy()
        B_fixed = fixed.B().copy()
        fixed.B_ref()[0, 0] = sentinel
        dof_free.B_ref()[0, 0] = sentinel

        total.x = np.array(total.x) + 1e-3
        self.assertFalse(np.array_equal(free.B(), B_free), "the free term must follow x")
        self.assertEqual(fixed.B()[0, 0], sentinel, "a fully fixed field lost its cache on an x assignment")
        self.assertEqual(dof_free.B()[0, 0], sentinel, "a DOF-free field lost its cache on an x assignment")

        coils[0].curve.resample()
        self.assert_field_moved_to(fixed.B(), B_fixed, fresh_field(bases, [sample, sample], RESAMPLE_POINTS))

    def test_biotsavart_both_interfaces_give_same_result(self):
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        points = np.asarray(10 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        B1 = BiotSavart([coil]).set_points(points).B()
        from simsoptpp import biot_savart_B
        B2 = biot_savart_B(points, [curve.gamma()], [curve.gammadash()], [1e4])
        assert np.linalg.norm(B1) > 1e-5
        assert np.allclose(B1, B2)

    def test_biotsavart_exponential_convergence(self):
        BiotSavart([Coil(get_curve(), Current(1e4))])
        points = np.asarray(10 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        btrue = BiotSavart([Coil(get_curve(1000), Current(1e4))]).set_points(points).B()
        bcoarse = BiotSavart([Coil(get_curve(10), Current(1e4))]).set_points(points).B()
        bfine = BiotSavart([Coil(get_curve(20), Current(1e4))]).set_points(points).B()
        assert np.linalg.norm(btrue-bfine) < 1e-4 * np.linalg.norm(bcoarse-bfine)

        dbtrue = BiotSavart([Coil(get_curve(1000), Current(1e4))]).set_points(points).dB_by_dX()
        dbcoarse = BiotSavart([Coil(get_curve(10), Current(1e4))]).set_points(points).dB_by_dX()
        dbfine = BiotSavart([Coil(get_curve(20), Current(1e4))]).set_points(points).dB_by_dX()
        assert np.linalg.norm(dbtrue-dbfine) < 1e-4 * np.linalg.norm(dbcoarse-dbfine)

        dbtrue = BiotSavart([Coil(get_curve(1000), Current(1e4))]).set_points(points).d2B_by_dXdX()
        dbcoarse = BiotSavart([Coil(get_curve(10), Current(1e4))]).set_points(points).d2B_by_dXdX()
        dbfine = BiotSavart([Coil(get_curve(20), Current(1e4))]).set_points(points).d2B_by_dXdX()
        assert np.linalg.norm(dbtrue-dbfine) < 1e-4 * np.linalg.norm(dbcoarse-dbfine)

    def test_dB_by_dcoilcoeff_reverse_taylortest(self):
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape)-0.5)

        bs.set_points(points)
        curve_dofs = curve.x
        B = bs.B()
        J0 = np.sum(B**2)
        dJ = bs.B_vjp(B)(curve)

        h = 1e-2 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
        dJ_dh = 2*np.sum(dJ * h)
        err = 1e6
        for i in range(5, 10):
            eps = 0.5**i
            curve.x = curve_dofs + eps * h
            Bh = bs.B()
            Jh = np.sum(Bh**2)
            deriv_est = (Jh-J0)/eps
            err_new = np.linalg.norm(deriv_est-dJ_dh)
            assert err_new < 0.55 * err
            err = err_new

    def test_dBdX_by_dcoilcoeff_reverse_taylortest(self):
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape)-0.5)

        bs.set_points(points)
        curve_dofs = curve.x
        B = bs.B()
        dBdX = bs.dB_by_dX()
        J0 = np.sum(dBdX**2)
        dJ = bs.B_and_dB_vjp(B, dBdX)[1](curve)

        h = 1e-2 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
        dJ_dh = 2*np.sum(dJ * h)
        err = 1e6
        for i in range(5, 10):
            eps = 0.5**i
            curve.x = curve_dofs + eps * h
            dBdXh = bs.dB_by_dX()
            Jh = np.sum(dBdXh**2)
            deriv_est = (Jh-J0)/eps
            err_new = np.linalg.norm(deriv_est-dJ_dh)
            assert err_new < 0.55 * err
            err = err_new

    def subtest_biotsavart_dBdX_taylortest(self, idx):
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape)-0.5)
        bs.set_points(points)
        B0 = bs.B()[idx]
        dB = bs.dB_by_dX()[idx]
        for direction in [np.asarray((1., 0, 0)), np.asarray((0, 1., 0)), np.asarray((0, 0, 1.))]:
            deriv = dB.T.dot(direction)
            err = 1e6
            for i in range(5, 10):
                eps = 0.5**i
                bs.set_points(points + eps * direction)
                Beps = bs.B()[idx]
                deriv_est = (Beps-B0)/(eps)
                new_err = np.linalg.norm(deriv-deriv_est)
                assert new_err < 0.55 * err
                err = new_err

    def test_biotsavart_dBdX_taylortest(self):
        for idx in [0, 16]:
            with self.subTest(idx=idx):
                self.subtest_biotsavart_dBdX_taylortest(idx)

    def subtest_biotsavart_gradient_symmetric_and_divergence_free(self, idx):
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape)-0.5)
        bs.set_points(points)
        dB = bs.dB_by_dX()
        assert abs(dB[idx][0, 0] + dB[idx][1, 1] + dB[idx][2, 2]) < 1e-14
        assert np.allclose(dB[idx], dB[idx].T)

    def test_biotsavart_gradient_symmetric_and_divergence_free(self):
        for idx in [0, 16]:
            with self.subTest(idx=idx):
                self.subtest_biotsavart_gradient_symmetric_and_divergence_free(idx)

    def subtest_d2B_by_dXdX_is_symmetric(self, idx):
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape)-0.5)
        bs.set_points(points)
        d2B_by_dXdX = bs.d2B_by_dXdX()
        for i in range(3):
            assert np.allclose(d2B_by_dXdX[idx, :, :, i], d2B_by_dXdX[idx, :, :, i].T)

    def test_d2B_by_dXdX_is_symmetric(self):
        for idx in [0, 16]:
            with self.subTest(idx=idx):
                self.subtest_d2B_by_dXdX_is_symmetric(idx)

    def subtest_biotsavart_d2B_by_dXdX_taylortest(self, idx):
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        bs.set_points(points)
        d2B_by_dXdX = bs.d2B_by_dXdX()
        for d1 in range(3):
            for d2 in range(3):
                second_deriv = d2B_by_dXdX[idx, d1, d2]
                err = 1e6
                for i in range(5, 10):
                    eps = 0.5**i

                    ed2 = np.zeros((1, 3))
                    ed2[0, d2] = 1.

                    bs.set_points(points + eps * ed2)
                    dB_dXp = bs.dB_by_dX()[idx, d1]

                    bs.set_points(points - eps * ed2)
                    dB_dXm = bs.dB_by_dX()[idx, d1]

                    second_deriv_est = (dB_dXp - dB_dXm)/(2. * eps)

                    new_err = np.linalg.norm(second_deriv-second_deriv_est)
                    assert new_err < 0.30 * err
                    err = new_err

    def test_biotsavart_d2B_by_dXdX_taylortest(self):
        for idx in [0, 16]:
            with self.subTest(idx=idx):
                self.subtest_biotsavart_d2B_by_dXdX_taylortest(idx)

    def test_biotsavart_B_is_curlA(self):
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        bs.set_points(points)
        B, dA_by_dX = bs.B(), bs.dA_by_dX()
        curlA1 = dA_by_dX[:, 1, 2] - dA_by_dX[:, 2, 1]
        curlA2 = dA_by_dX[:, 2, 0] - dA_by_dX[:, 0, 2]
        curlA3 = dA_by_dX[:, 0, 1] - dA_by_dX[:, 1, 0]
        curlA = np.concatenate((curlA1[:, None], curlA2[:, None], curlA3[:, None]), axis=1)
        err = np.max(np.abs(curlA - B))
        assert err < 1e-14

    def subtest_biotsavart_dAdX_taylortest(self, idx):
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape)-0.5)
        bs.set_points(points)
        A0 = bs.A()[idx]
        dA = bs.dA_by_dX()[idx]

        for direction in [np.asarray((1., 0, 0)), np.asarray((0, 1., 0)), np.asarray((0, 0, 1.))]:
            deriv = dA.T.dot(direction)
            err = 1e6
            for i in range(5, 10):
                eps = 0.5**i
                bs.set_points(points + eps * direction)
                Aeps = bs.A()[idx]
                deriv_est = (Aeps-A0)/(eps)
                new_err = np.linalg.norm(deriv-deriv_est)
                assert new_err < 0.55 * err
                err = new_err

    def test_biotsavart_dAdX_taylortest(self):
        for idx in [0, 16]:
            with self.subTest(idx=idx):
                self.subtest_biotsavart_dAdX_taylortest(idx)

    def subtest_biotsavart_d2A_by_dXdX_taylortest(self, idx):
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        bs.set_points(points)
        d2A_by_dXdX = bs.d2A_by_dXdX()
        for d1 in range(3):
            for d2 in range(3):
                second_deriv = d2A_by_dXdX[idx, d1, d2]
                err = 1e6
                for i in range(5, 10):
                    eps = 0.5**i

                    ed2 = np.zeros((1, 3))
                    ed2[0, d2] = 1.

                    bs.set_points(points + eps * ed2)
                    dA_dXp = bs.dA_by_dX()[idx, d1]

                    bs.set_points(points - eps * ed2)
                    dA_dXm = bs.dA_by_dX()[idx, d1]

                    second_deriv_est = (dA_dXp - dA_dXm)/(2. * eps)

                    new_err = np.linalg.norm(second_deriv-second_deriv_est)
                    #print("new_err", new_err)
                    assert new_err < 0.30 * err
                    err = new_err

    def test_biotsavart_d2A_by_dXdX_taylortest(self):
        for idx in [0, 16]:
            with self.subTest(idx=idx):
                self.subtest_biotsavart_d2A_by_dXdX_taylortest(idx)

    def test_biotsavart_coil_current_taylortest(self):
        curve0 = get_curve()
        c0 = 1e4
        current0 = Current(c0)
        curve1 = get_curve(perturb=True)
        current1 = Current(1e3)
        bs = BiotSavart([Coil(curve0, current0), Coil(curve1, current1)])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        bs.set_points(points)
        B = bs.B()
        J = bs.dB_by_dX()
        H = bs.d2B_by_dXdX()
        dB = bs.dB_by_dcoilcurrents()
        dJ = bs.d2B_by_dXdcoilcurrents()
        dH = bs.d3B_by_dXdXdcoilcurrents()

        # the B field is linear in the current, so a small stepsize is not necessary
        current0.x = [0]
        B0 = bs.B()
        J0 = bs.dB_by_dX()
        H0 = bs.d2B_by_dXdX()
        dB_approx = (B-B0)/(c0)
        dJ_approx = (J-J0)/(c0)
        dH_approx = (H-H0)/(c0)
        assert np.linalg.norm(dB[0]-dB_approx) < 1e-15
        assert np.linalg.norm(dJ[0]-dJ_approx) < 1e-15
        print(f"H norm is {np.linalg.norm(dH[0]-dH_approx)}")
        assert np.linalg.norm(dH[0]-dH_approx) < 1e-15

    def test_dA_by_dcoilcoeff_reverse_taylortest(self):
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, ScaledCurrent(Current(1), 1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape)-0.5)

        bs.set_points(points)
        coil_dofs = coil.x
        A = bs.A()
        J0 = np.sum(A**2)
        dJ = bs.A_vjp(A)(coil)

        h = 1e-2 * np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
        dJ_dh = 2*np.sum(dJ * h)
        err = 1e6
        for i in range(5, 10):
            eps = 0.5**i
            coil.x = coil_dofs + eps * h
            Ah = bs.A()
            Jh = np.sum(Ah**2)
            deriv_est = (Jh-J0)/eps
            err_new = np.linalg.norm(deriv_est-dJ_dh)
            assert err_new < 0.55 * err
            err = err_new

    def test_dAdX_by_dcoilcoeff_reverse_taylortest(self):
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, ScaledCurrent(Current(1), 1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape)-0.5)

        bs.set_points(points)
        coil_dofs = coil.x
        A = bs.A()
        dAdX = bs.dA_by_dX()
        J0 = np.sum(dAdX**2)
        dJ = bs.A_and_dA_vjp(A, dAdX)[1](coil)

        h = 1e-2 * np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
        dJ_dh = 2*np.sum(dJ * h)
        err = 1e6
        for i in range(5, 10):
            eps = 0.5**i
            coil.x = coil_dofs + eps * h
            dAdXh = bs.dA_by_dX()
            Jh = np.sum(dAdXh**2)
            deriv_est = (Jh-J0)/eps
            err_new = np.linalg.norm(deriv_est-dJ_dh)
            assert err_new < 0.55 * err
            err = err_new

    def test_flux_through_disk(self):
        # this test makes sure that the toroidal flux through a disk (D)
        # given by \int_D B \cdot n dB = \int_{\partial D} A \cdot dl
        np.random.seed(1)

        from scipy.spatial.transform import Rotation as R
        rot = R.from_euler('zyx', [21.234, 8.431, -4.86392], degrees=True).as_matrix()
        new_n = rot @ np.array([0, 0, 1])

        curve = get_curve(perturb=True)
        coil = Coil(curve, Current(1e4))
        bs = BiotSavart([coil])

        # define the disk
        def f(t, r):
            x = r * np.cos(t).reshape((-1, 1))
            y = r * np.sin(t).reshape((-1, 1))
            pts = np.concatenate((x, y, np.zeros((x.shape[1], 1))), axis=1) @ rot.T
            bs.set_points(pts)
            B = bs.B()
            temp = np.sum(B*new_n[None, :], axis=1)*r
            return temp[0]

        # int_r int_theta B int r dr dtheta
        from scipy import integrate
        r = 0.15
        fluxB = integrate.dblquad(f, 0, r, 0, 2*np.pi, epsabs=1e-15, epsrel=1e-15)

        # num range used to be (20, 60) but this fails for num <= 20-30 for certain
        # random coil initializations since don't have enough quadrature points
        # to integrate to numerical precision.
        for num in range(40, 100):
            npoints = num
            angles = np.linspace(0, 2*np.pi, npoints, endpoint=False).reshape((-1, 1))
            t = np.concatenate((-np.sin(angles), np.cos(angles), np.zeros((angles.size, 1))), axis=1) @ rot.T
            pts = r*np.concatenate((np.cos(angles), np.sin(angles), np.zeros((angles.size, 1))), axis=1) @ rot.T
            bs.set_points(pts)
            A = bs.A()
            fluxA = r*np.sum(A*t) * 2 * np.pi/npoints
            assert np.abs(fluxB[0]-fluxA)/fluxB[0] < 1e-14

    def test_biotsavart_vector_potential_coil_current_taylortest(self):
        curve0 = get_curve()
        c0 = 1e4
        current0 = Current(c0)
        curve1 = get_curve(perturb=True)
        current1 = Current(1e3)
        bs = BiotSavart([Coil(curve0, current0), Coil(curve1, current1)])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        bs.set_points(points)
        A = bs.A()
        J = bs.dA_by_dX()
        H = bs.d2A_by_dXdX()

        #trigger recompute bell for code coverage of field cache
        bs.recompute_bell()
        dA = bs.dA_by_dcoilcurrents()
        bs.recompute_bell()
        dJ = bs.d2A_by_dXdcoilcurrents()
        bs.recompute_bell()
        dH = bs.d3A_by_dXdXdcoilcurrents()

        # the A field is linear in the current, so a small stepsize is not necessary
        current0.x = [0]
        A0 = bs.A()
        J0 = bs.dA_by_dX()
        H0 = bs.d2A_by_dXdX()
        dA_approx = (A-A0)/(c0)
        dJ_approx = (J-J0)/(c0)
        dH_approx = (H-H0)/(c0)
        assert np.linalg.norm(dA[0]-dA_approx) < 1e-15
        assert np.linalg.norm(dJ[0]-dJ_approx) < 1e-15
        assert np.linalg.norm(dH[0]-dH_approx) < 1e-15


if __name__ == "__main__":
    unittest.main()
