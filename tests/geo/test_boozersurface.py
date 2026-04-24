import unittest
from unittest.mock import patch

import numpy as np
import simsopt.geo.boozersurface as boozersurface_module
import simsopt.geo.surfaceobjectives as surfaceobjectives_module
from simsopt.field.coil import coils_via_symmetries
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo import SurfaceXYZTensorFourier, SurfaceRZFourier
from simsopt.geo.surfaceobjectives import (
    Area,
    ToroidalFlux,
    boozer_surface_dexactresidual_dcoils_dcurrents_vjp,
    boozer_surface_dlsqgrad_dcoils_vjp,
    boozer_surface_residual,
    boozer_surface_residual_dB,
)
from simsopt.configs.zoo import get_ncsx_data, get_hsx_data, get_giuliani_data
from .surface_test_helpers import get_surface, get_exact_surface, get_boozer_surface


surfacetypes_list = ["SurfaceXYZFourier", "SurfaceXYZTensorFourier"]
stellsym_list = [True, False]


class BoozerSurfaceTests(unittest.TestCase):
    def test_boozer_residual_helpers_fall_back_to_alpha_only_extension(self):
        alpha = 1.25
        G = 0.5
        current_I = 0.75
        iota = -0.2
        xphi = np.zeros((1, 1, 3))
        xtheta = np.zeros((1, 1, 3))
        B = np.zeros((1, 1, 3))
        dB_dx = np.zeros((1, 1, 3, 3))
        d2B_dx2 = np.zeros((1, 1, 3, 3, 3))
        dx_ds = np.zeros((1, 1, 3, 1))
        dxphi_ds = np.zeros((1, 1, 3, 1))
        dxtheta_ds = np.zeros((1, 1, 3, 1))
        alpha_only_gradient = np.array([3.0, 5.0, 7.0])
        alpha_only_hessian = np.array(
            [
                [11.0, 13.0, 17.0],
                [13.0, 19.0, 23.0],
                [17.0, 23.0, 29.0],
            ]
        )
        residual_calls = []
        residual_ds_calls = []
        residual_ds2_calls = []

        def fake_boozer_residual(*args):
            residual_calls.append(args)
            if len(args) == 7:
                raise TypeError("incompatible function arguments")
            return "alpha-only-residual"

        def fake_boozer_residual_ds(*args):
            residual_ds_calls.append(args)
            if len(args) == 11:
                raise TypeError("incompatible function arguments")
            return ("alpha-only-ds", alpha_only_gradient.copy())

        def fake_boozer_residual_ds2(*args):
            residual_ds2_calls.append(args)
            if len(args) == 12:
                raise TypeError("incompatible function arguments")
            return (
                "alpha-only-ds2",
                alpha_only_gradient.copy(),
                alpha_only_hessian.copy(),
            )

        previous_residual_mode = boozersurface_module._BOOZER_RESIDUAL_CALL_MODE
        previous_residual_ds_mode = boozersurface_module._BOOZER_RESIDUAL_DS_CALL_MODE
        previous_residual_ds2_mode = boozersurface_module._BOOZER_RESIDUAL_DS2_CALL_MODE
        boozersurface_module._BOOZER_RESIDUAL_CALL_MODE = None
        boozersurface_module._BOOZER_RESIDUAL_DS_CALL_MODE = None
        boozersurface_module._BOOZER_RESIDUAL_DS2_CALL_MODE = None
        try:
            with patch.object(boozersurface_module.sopp, "boozer_residual", side_effect=fake_boozer_residual):
                residual = boozersurface_module._call_boozer_residual(
                    alpha, G, current_I, iota, xphi, xtheta, B, True
                )
            with patch.object(boozersurface_module.sopp, "boozer_residual_ds", side_effect=fake_boozer_residual_ds):
                residual_ds = boozersurface_module._call_boozer_residual_ds(
                    alpha, G, current_I, iota, B, dB_dx, xphi, xtheta, dx_ds, dxphi_ds, dxtheta_ds, True
                )
            with patch.object(boozersurface_module.sopp, "boozer_residual_ds2", side_effect=fake_boozer_residual_ds2):
                residual_ds2 = boozersurface_module._call_boozer_residual_ds2(
                    alpha, G, current_I, iota, B, dB_dx, d2B_dx2, xphi, xtheta, dx_ds, dxphi_ds, dxtheta_ds, True
                )
        finally:
            boozersurface_module._BOOZER_RESIDUAL_CALL_MODE = previous_residual_mode
            boozersurface_module._BOOZER_RESIDUAL_DS_CALL_MODE = previous_residual_ds_mode
            boozersurface_module._BOOZER_RESIDUAL_DS2_CALL_MODE = previous_residual_ds2_mode

        self.assertEqual(residual, "alpha-only-residual")
        expected_gradient = np.array(
            [alpha_only_gradient[0], alpha_only_gradient[1] + current_I * alpha_only_gradient[2], alpha_only_gradient[2]]
        )
        expected_hessian = np.array(
            [
                [11.0, 25.75, 17.0],
                [25.75, 69.8125, 44.75],
                [17.0, 44.75, 29.0],
            ]
        )
        self.assertEqual(residual_ds[0], "alpha-only-ds")
        self.assertEqual(residual_ds2[0], "alpha-only-ds2")
        np.testing.assert_allclose(residual_ds[1], expected_gradient)
        np.testing.assert_allclose(residual_ds2[1], expected_gradient)
        np.testing.assert_allclose(residual_ds2[2], expected_hessian)
        self.assertEqual(len(residual_calls), 2)
        self.assertEqual(len(residual_ds_calls), 2)
        self.assertEqual(len(residual_ds2_calls), 2)
        self.assertEqual(len(residual_calls[0]), 7)
        self.assertEqual(len(residual_calls[1]), 6)
        self.assertEqual(len(residual_ds_calls[0]), 11)
        self.assertEqual(len(residual_ds_calls[1]), 10)
        self.assertEqual(len(residual_ds2_calls[0]), 12)
        self.assertEqual(len(residual_ds2_calls[1]), 11)
        self.assertEqual(residual_calls[1][0], alpha)
        self.assertEqual(residual_ds_calls[1][0], alpha)
        self.assertEqual(residual_ds2_calls[1][0], alpha)

    def test_boozer_dresidual_dc_falls_back_to_alpha_only_extension(self):
        alpha = 1.25
        G = 0.5
        current_I = 0.75
        iota = -0.2
        dB_dc = np.zeros((1, 1, 3, 1))
        B = np.zeros((1, 1, 3))
        tang = np.zeros((1, 1, 3))
        B2 = np.zeros((1, 1))
        dxphi_dc = np.zeros((1, 1, 3, 1))
        dxtheta_dc = np.zeros((1, 1, 3, 1))
        calls = []

        def fake_boozer_dresidual_dc(*args):
            calls.append(args)
            if len(args) == 9:
                raise TypeError("incompatible function arguments")
            return "alpha-only"

        previous_mode = surfaceobjectives_module._BOOZER_DRESIDUAL_DC_CALL_MODE
        surfaceobjectives_module._BOOZER_DRESIDUAL_DC_CALL_MODE = None
        try:
            with patch.object(
                surfaceobjectives_module.sopp,
                "boozer_dresidual_dc",
                side_effect=fake_boozer_dresidual_dc,
            ):
                result = surfaceobjectives_module._call_boozer_dresidual_dc(
                    alpha,
                    G,
                    current_I,
                    dB_dc,
                    B,
                    tang,
                    B2,
                    dxphi_dc,
                    iota,
                    dxtheta_dc,
                )
        finally:
            surfaceobjectives_module._BOOZER_DRESIDUAL_DC_CALL_MODE = previous_mode

        self.assertEqual(result, "alpha-only")
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(calls[0]), 9)
        self.assertEqual(len(calls[1]), 8)
        self.assertEqual(calls[1][0], alpha)

    def _make_area_boozer_surface(self, *, current_I, mpol, ntor, phis, thetas, constraint_weight, options):
        curves, currents, ma = get_ncsx_data()
        coils = coils_via_symmetries(curves, currents, 3, True)
        bs = BiotSavart(coils)
        current_sum = sum(abs(c.current.get_value()) for c in coils)
        G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))
        surface = SurfaceXYZTensorFourier(
            mpol=mpol, ntor=ntor, stellsym=True, nfp=3,
            quadpoints_phi=phis, quadpoints_theta=thetas,
        )
        surface.fit_to_curve(ma, 0.1, flip_theta=True)
        label = Area(surface)
        boozer_surface = BoozerSurface(
            bs, surface, label, label.J(), constraint_weight=constraint_weight,
            options=options, I=current_I,
        )
        return bs, G0, boozer_surface

    def _assert_directional_fd_convergence(self, f, coeffs, direction, directional_derivative):
        err_old = 1e9
        epsilons = np.power(2., -np.asarray(range(11, 18)))
        for eps in epsilons:
            dfdx_fd = (f(coeffs + eps * direction) - f(coeffs - eps * direction)) / (2 * eps)
            err = np.abs(dfdx_fd - directional_derivative)
            self.assertLess(err, err_old * 0.31)
            err_old = err

    def _assert_penalty_constraints_cpp_python_match(self, boozer_surface, x, *, optimize_G, weight_inv_modB):
        w = 0.
        f0 = boozer_surface.boozer_penalty_constraints(
            x, derivatives=0, constraint_weight=w, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB)
        f1 = boozer_surface.boozer_penalty_constraints_vectorized(
            x, derivatives=0, constraint_weight=w, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB)
        np.testing.assert_allclose(f0, f1, atol=1e-13, rtol=1e-13)
        print(np.abs(f0-f1)/np.abs(f0))

        f0, J0 = boozer_surface.boozer_penalty_constraints(
            x, derivatives=1, constraint_weight=w, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB)
        f1, J1 = boozer_surface.boozer_penalty_constraints_vectorized(
            x, derivatives=1, constraint_weight=w, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB)
        np.testing.assert_allclose(f0, f1, atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose(J0, J1, atol=1e-11, rtol=1e-11)

        h1 = np.random.rand(J0.size)-0.5
        np.testing.assert_allclose(J0@h1, J1@h1, atol=1e-13, rtol=1e-13)
        print(np.abs(f0-f1)/np.abs(f0), np.abs(J0@h1-J1@h1)/np.abs(J0@h1))

        H0, H1 = self._assert_penalty_constraints_derivatives2_cpp_python_match(
            boozer_surface, x, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB,
        )
        return H0, H1

    def _assert_penalty_constraints_derivatives2_cpp_python_match(self, boozer_surface, x, *, optimize_G, weight_inv_modB):
        w = 0.
        f0, J0, H0 = boozer_surface.boozer_penalty_constraints(
            x, derivatives=2, constraint_weight=w, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB)
        f1, J1, H1 = boozer_surface.boozer_penalty_constraints_vectorized(
            x, derivatives=2, constraint_weight=w, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB)
        np.testing.assert_allclose(f0, f1, atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose(J0, J1, atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(H0, H1, atol=1e-10, rtol=1e-10)

        h1 = np.random.rand(J0.size)-0.5
        h2 = np.random.rand(J0.size)-0.5
        np.testing.assert_allclose(J0@h1, J1@h1, atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose((H0@h1)@h2, (H1@h1)@h2, atol=1e-13, rtol=1e-13)
        print(np.abs(f0-f1)/np.abs(f0), np.abs(J0@h1-J1@h1)/np.abs(J0@h1), np.abs((H0@h1)@h2-(H1@h1)@h2)/np.abs((H0@h1)@h2))
        return H0, H1

    def _print_hessian_differences(self, Ha, Hb):
        diff = np.abs(Ha.flatten() - Hb.flatten())
        rel_diff = diff/np.abs(Ha.flatten())
        ij1 = np.where(diff.reshape(Ha.shape) == np.max(diff))
        i1 = ij1[0][0]
        j1 = ij1[1][0]

        ij2 = np.where(rel_diff.reshape(Ha.shape) == np.max(rel_diff))
        i2 = ij2[0][0]
        j2 = ij2[1][0]
        print(f'max err     ({i1:03}, {j1:03}): {np.max(diff):.6e}, {Ha[i1, j1]:.6e}\nmax rel err ({i2:03}, {j2:03}): {np.max(rel_diff):.6e}, {Ha[i2,j2]:.6e}\n')

    def test_residual(self):
        """
        This test loads a SurfaceXYZFourier that interpolates the xyz
        coordinates of a surface in the NCSX configuration that was computed
        on a previous branch of pyplasmaopt. Here, we verify that the Boozer
        residual at these interpolation points is small.
        """

        s = get_exact_surface()
        curves, currents, ma = get_ncsx_data()
        coils = coils_via_symmetries(curves, currents, 3, True)
        bs = BiotSavart(coils)
        bs_tf = BiotSavart(coils)

        weight = 1.
        tf = ToroidalFlux(s, bs_tf)

        # these data are obtained from `boozer` branch of pyplamsaopt
        tf_target = 0.41431152
        iota = -0.44856192

        boozer_surface = BoozerSurface(bs, s, tf, tf_target)
        x = np.concatenate((s.get_dofs(), [iota]))
        r0 = boozer_surface.boozer_penalty_constraints(
            x, derivatives=0, constraint_weight=weight, optimize_G=False,
            scalarize=False)
        # the residual should be close to zero for all entries apart from the y
        # and z coordinate at phi=0 and theta=0 (and the corresponding rotations)
        ignores_idxs = np.zeros_like(r0)
        ignores_idxs[[1, 2, 693, 694, 695, 1386, 1387, 1388, -2, -1]] = 1
        assert np.max(np.abs(r0[ignores_idxs < 0.5])) < 1e-8
        assert np.max(np.abs(r0[-2:])) < 1e-6

    def test_boozer_penalty_constraints_gradient(self):
        """
        Taylor test to verify the gradient of the scalarized constrained
        optimization problem's objective.
        """
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                for optimize_G in [True, False]:
                    for vectorize in [True, False]:
                        with self.subTest(surfacetype=surfacetype,
                                          stellsym=stellsym,
                                          optimize_G=optimize_G,
                                          vectorize=vectorize):
                            self.subtest_boozer_penalty_constraints_gradient(surfacetype, stellsym, optimize_G, vectorize)

    def test_boozer_penalty_constraints_hessian(self):
        """
        Taylor test to verify the Hessian of the scalarized constrained
        optimization problem's objective.
        """
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                for optimize_G in [True, False]:
                    for vectorize in [True, False]:
                        with self.subTest(surfacetype=surfacetype,
                                          stellsym=stellsym,
                                          optimize_G=optimize_G,
                                          vectorize=vectorize):
                            self.subtest_boozer_penalty_constraints_hessian(
                                surfacetype, stellsym, optimize_G, vectorize)

    def subtest_boozer_penalty_constraints_gradient(self, surfacetype, stellsym,
                                                    optimize_G=False, vectorize=False):
        np.random.seed(1)
        curves, currents, ma = get_ncsx_data()
        coils = coils_via_symmetries(curves, currents, 3, True)
        bs = BiotSavart(coils)
        bs_tf = BiotSavart(coils)
        current_sum = sum(abs(c.current.get_value()) for c in coils)

        s = get_surface(surfacetype, stellsym)
        s.fit_to_curve(ma, 0.1)

        weight = 11.1232

        tf = ToroidalFlux(s, bs_tf, nphi=51, ntheta=51)

        tf_target = 0.1
        boozer_surface = BoozerSurface(bs, s, tf, tf_target)
        fun = boozer_surface.boozer_penalty_constraints_vectorized if vectorize else boozer_surface.boozer_penalty_constraints

        iota = -0.3
        x = np.concatenate((s.get_dofs(), [iota]))
        if optimize_G:
            x = np.concatenate((x, [2.*np.pi*current_sum*(4*np.pi*10**(-7)/(2 * np.pi))]))
        f0, J0 = fun(x, derivatives=1, constraint_weight=weight, optimize_G=optimize_G)
        h = np.random.uniform(size=x.shape)-0.5
        Jex = J0@h

        err_old = 1e9
        epsilons = np.power(2., -np.asarray(range(7, 20)))
        print("###############################################################")
        for eps in epsilons:
            f1 = fun(x + eps*h, derivatives=0, constraint_weight=weight, optimize_G=optimize_G)
            Jfd = (f1-f0)/eps
            err = np.linalg.norm(Jfd-Jex)/np.linalg.norm(Jex)
            print(err/err_old, f0, f1)
            assert err < err_old * 0.55
            err_old = err
        print("###############################################################")

    def subtest_boozer_penalty_constraints_hessian(self, surfacetype, stellsym,
                                                   optimize_G=False, vectorize=False):
        np.random.seed(1)
        curves, currents, ma = get_ncsx_data()
        coils = coils_via_symmetries(curves, currents, 3, True)
        bs = BiotSavart(coils)
        bs_tf = BiotSavart(coils)
        current_sum = sum(abs(c.current.get_value()) for c in coils)

        s = get_surface(surfacetype, stellsym)
        s.fit_to_curve(ma, 0.1)

        tf = ToroidalFlux(s, bs_tf, nphi=51, ntheta=51)

        tf_target = 0.1
        boozer_surface = BoozerSurface(bs, s, tf, tf_target)
        fun = boozer_surface.boozer_penalty_constraints_vectorized if vectorize else boozer_surface.boozer_penalty_constraints

        iota = -0.3
        x = np.concatenate((s.get_dofs(), [iota]))
        if optimize_G:
            x = np.concatenate(
                (x, [2.*np.pi*current_sum*(4*np.pi*10**(-7)/(2 * np.pi))]))

        f0, J0, H0 = fun(x, derivatives=2, optimize_G=optimize_G)
        h1 = np.random.uniform(size=x.shape)-0.5
        h2 = np.random.uniform(size=x.shape)-0.5
        d2f = h1 @ H0 @ h2

        err_old = 1e9
        epsilons = np.power(2., -np.asarray(range(10, 20)))
        print("###############################################################")
        for eps in epsilons:
            fp, Jp = fun(x + eps*h1, derivatives=1, optimize_G=optimize_G)
            d2f_fd = (Jp@h2-J0@h2)/eps
            err = np.abs(d2f_fd-d2f)/np.abs(d2f)
            print(err/err_old)
            assert err < err_old * 0.55
            err_old = err

    def test_boozer_constrained_jacobian(self):
        """
        Taylor test to verify the Jacobian of the first order optimality
        conditions of the exactly constrained optimization problem.
        """
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                for optimize_G in [True, False]:
                    with self.subTest(surfacetype=surfacetype,
                                      stellsym=stellsym,
                                      optimize_G=optimize_G):
                        self.subtest_boozer_constrained_jacobian(
                            surfacetype, stellsym, optimize_G)

    def subtest_boozer_constrained_jacobian(self, surfacetype, stellsym,
                                            optimize_G=False):
        np.random.seed(1)
        curves, currents, ma = get_ncsx_data()
        coils = coils_via_symmetries(curves, currents, 3, True)
        bs = BiotSavart(coils)
        bs_tf = BiotSavart(coils)
        current_sum = sum(abs(c.current.get_value()) for c in coils)

        s = get_surface(surfacetype, stellsym)
        s.fit_to_curve(ma, 0.1)

        tf = ToroidalFlux(s, bs_tf, nphi=51, ntheta=51)

        tf_target = 0.1
        boozer_surface = BoozerSurface(bs, s, tf, tf_target)

        iota = -0.3
        lm = [0., 0.]
        x = np.concatenate((s.get_dofs(), [iota]))
        if optimize_G:
            x = np.concatenate(
                (x, [2.*np.pi*current_sum*(4*np.pi*10**(-7)/(2 * np.pi))]))
        xl = np.concatenate((x, lm))
        res0, dres0 = boozer_surface.boozer_exact_constraints(
            xl, derivatives=1, optimize_G=optimize_G)

        h = np.random.uniform(size=xl.shape)-0.5
        dres_exact = dres0@h

        err_old = 1e9
        epsilons = np.power(2., -np.asarray(range(7, 20)))
        print("###############################################################")
        for eps in epsilons:
            res1 = boozer_surface.boozer_exact_constraints(
                xl + eps*h, derivatives=0, optimize_G=optimize_G)
            dres_fd = (res1-res0)/eps
            err = np.linalg.norm(dres_fd-dres_exact)
            print(err/err_old)
            assert err < err_old * 0.55
            err_old = err
        print("###############################################################")

    def test_boozer_surface_optimisation_convergence(self):
        """
        Test to verify the various optimization algorithms that compute
        the Boozer angles on a surface.
        """

        configs = [
            ("SurfaceXYZTensorFourier", True, True, 'residual_exact'),  # noqa
            ("SurfaceXYZTensorFourier", True, True, 'newton_exact'),  # noqa
            ("SurfaceXYZTensorFourier", True, True, 'newton'),  # noqa
            ("SurfaceXYZTensorFourier", False, True, 'ls'),  # noqa
            ("SurfaceXYZFourier", True, False, 'ls'),  # noqa
        ]
        for surfacetype, stellsym, optimize_G, second_stage in configs:
            for get_data in [get_hsx_data, get_ncsx_data, get_giuliani_data]:
                for vectorize in [True, False]:
                    with self.subTest(
                        surfacetype=surfacetype, stellsym=stellsym,
                            optimize_G=optimize_G, second_stage=second_stage, get_data=get_data, vectorize=vectorize):
                        self.subtest_boozer_surface_optimisation_convergence(
                            surfacetype, stellsym, optimize_G, second_stage, get_data, vectorize)

    def subtest_boozer_surface_optimisation_convergence(self, surfacetype,
                                                        stellsym, optimize_G,
                                                        second_stage, get_data,
                                                        vectorize):
        curves, currents, ma = get_data()
        if stellsym:
            coils = coils_via_symmetries(curves, currents, ma.nfp, True)
        else:
            # Create a stellarator that still has rotational symmetry but
            # doesn't have stellarator symmetry. We do this by first applying
            # stellarator symmetry, then breaking this slightly, and then
            # applying rotational symmetry
            from simsopt.geo.curve import RotatedCurve
            curves_flipped = [RotatedCurve(c, 0, True) for c in curves]
            currents_flipped = [-cur for cur in currents]
            for c in curves_flipped:
                c.rotmat += 0.001*np.random.uniform(low=-1., high=1.,
                                                    size=c.rotmat.shape)
                c.rotmatT = c.rotmat.T
            coils = coils_via_symmetries(curves + curves_flipped,
                                         currents + currents_flipped, ma.nfp, False)
        current_sum = sum(abs(c.current.get_value()) for c in coils)

        bs = BiotSavart(coils)

        s = get_surface(surfacetype, stellsym, nfp=ma.nfp)
        s.fit_to_curve(ma, 0.1)
        if get_data is get_ncsx_data:
            iota = -0.4
        elif get_data is get_giuliani_data:
            iota = 0.4
        elif get_data is get_hsx_data:
            iota = 1.
        else:
            raise Exception("initial guess for rotational transform for this config not given")

        ar = Area(s)
        ar_target = ar.J()
        boozer_surface = BoozerSurface(bs, s, ar, ar_target)

        if optimize_G:
            G = 2.*np.pi*current_sum*(4*np.pi*10**(-7)/(2 * np.pi))
        else:
            G = None

        cw = (s.quadpoints_phi.size * s.quadpoints_theta.size * 3)
        # compute surface first using LBFGS exact and an area constraint
        res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(
            tol=1e-12, maxiter=700, constraint_weight=100/cw, iota=iota, G=G,
            vectorize=vectorize)
        print('Residual norm after LBFGS', res['iter'], np.sqrt(2*res['fun']))

        boozer_surface.recompute_bell()
        if second_stage == 'ls':
            res = boozer_surface.minimize_boozer_penalty_constraints_ls(
                tol=1e-11, maxiter=100, constraint_weight=1000./cw,
                iota=res['iota'], G=res['G'])
        elif second_stage == 'newton':
            res = boozer_surface.minimize_boozer_penalty_constraints_newton(
                tol=1e-10, maxiter=20, constraint_weight=100./cw,
                iota=res['iota'], G=res['G'], stab=1e-4, vectorize=vectorize)
        elif second_stage == 'newton_exact':
            res = boozer_surface.minimize_boozer_exact_constraints_newton(
                tol=1e-10, maxiter=15, iota=res['iota'], G=res['G'])
        elif second_stage == 'residual_exact':
            res = boozer_surface.solve_residual_equation_exactly_newton(
                tol=1e-12, maxiter=15, iota=res['iota'], G=res['G'])

        print('Residual norm after second stage', np.linalg.norm(res['residual']))
        assert res['success']
        assert not boozer_surface.surface.is_self_intersecting(thetas=100)

        # For the stellsym case we have z(0, 0) = y(0, 0) = 0. For the not
        # stellsym case, we enforce z(0, 0) = 0, but expect y(0, 0) \neq 0
        gammazero = s.gamma()[0, 0, :]
        assert np.abs(gammazero[2]) < 1e-10
        if stellsym:
            assert np.abs(gammazero[1]) < 1e-10
        else:
            assert np.abs(gammazero[1]) > 1e-6

        if surfacetype == 'SurfaceXYZTensorFourier':
            assert np.linalg.norm(res['residual']) < 1e-9

        print(ar_target, ar.J())
        print(res['residual'][-10:])
        if surfacetype == 'SurfaceXYZTensorFourier' or second_stage == 'newton_exact':
            assert np.abs(ar_target - ar.J()) < 1e-9
        else:
            assert np.abs(ar_target - ar.J()) < 1e-4

    def test_boozer_serialization(self):
        """
        Test to verify the serialization capability of a BoozerSurface.
        """
        for label in ['Volume', 'Area', 'ToroidalFlux']:
            with self.subTest(label=label):
                self.subtest_boozer_serialization(label)

    def subtest_boozer_serialization(self, label):
        import json
        from simsopt._core.json import GSONDecoder, GSONEncoder, SIMSON

        bs, boozer_surface = get_boozer_surface(label=label)

        # test serialization of BoozerSurface here too
        bs_str = json.dumps(SIMSON(boozer_surface), cls=GSONEncoder)
        bs_regen = json.loads(bs_str, cls=GSONDecoder)

        diff = boozer_surface.surface.x - bs_regen.surface.x
        self.assertAlmostEqual(np.linalg.norm(diff.ravel()), 0)
        self.assertAlmostEqual(boozer_surface.label.J(), bs_regen.label.J())
        self.assertAlmostEqual(boozer_surface.targetlabel, bs_regen.targetlabel)

        # check that BoozerSurface.surface and label.surface are the same surfaces
        assert bs_regen.label.surface is bs_regen.surface

    def test_run_code(self):
        """
        This unit test verifies that the run_code portion of the BoozerSurface class is working as expected
        """
        bs, boozer_surface = get_boozer_surface(boozer_type='ls')
        boozer_surface.run_code(boozer_surface.res['iota'], G=boozer_surface.res['G'])

        # this second time should not actually run
        boozer_surface.run_code(boozer_surface.res['iota'], G=boozer_surface.res['G'])

        for c in bs.coils:
            c.current.fix_all()

        boozer_surface.need_to_run_code = True
        # run without providing value of G
        boozer_surface.run_code(boozer_surface.res['iota'])

        bs, boozer_surface = get_boozer_surface(boozer_type='exact')
        boozer_surface.run_code(boozer_surface.res['iota'], G=boozer_surface.res['G'])

        # this second time should not actually run
        boozer_surface.run_code(boozer_surface.res['iota'], G=boozer_surface.res['G'])

        # run the BoozerExact algorithm without a guess for G
        boozer_surface.need_to_run_code = True
        boozer_surface.solve_residual_equation_exactly_newton(iota=boozer_surface.res['iota'])

    def test_convergence_cpp_and_notcpp_same(self):
        """
        This unit test verifies that that the cpp and not cpp implementations converge to 
        the same solutions
        """
        x_vec = self.subtest_convergence_cpp_and_notcpp_same(True)
        x_nonvec = self.subtest_convergence_cpp_and_notcpp_same(False)
        np.testing.assert_allclose(x_vec, x_nonvec, atol=1e-11)

    def subtest_convergence_cpp_and_notcpp_same(self, vectorize):
        """
        compute a surface using either the vectorized or non-vectorized subroutines
        """
        curves, currents, ma = get_ncsx_data()
        coils = coils_via_symmetries(curves, currents, ma.nfp, True)
        current_sum = sum(abs(c.current.get_value()) for c in coils)
        bs = BiotSavart(coils)

        s = get_surface('SurfaceXYZTensorFourier', True, nfp=ma.nfp)
        s.fit_to_curve(ma, 0.1)
        iota = -0.4

        ar = Area(s)
        ar_target = ar.J()
        boozer_surface = BoozerSurface(bs, s, ar, ar_target)

        G = 2.*np.pi*current_sum*(4*np.pi*10**(-7)/(2 * np.pi))

        cw = 3*s.quadpoints_phi.size * s.quadpoints_theta.size
        # vectorized solution first
        res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(
            tol=1e-10, maxiter=600, constraint_weight=100./cw, iota=iota, G=G,
            vectorize=vectorize)
        print('Residual norm after LBFGS', np.sqrt(2*res['fun']))

        boozer_surface.recompute_bell()
        res = boozer_surface.minimize_boozer_penalty_constraints_newton(
            tol=1e-10, maxiter=20, constraint_weight=100./cw,
            iota=res['iota'], G=res['G'], stab=0., vectorize=vectorize)

        assert res['success']
        x = boozer_surface.surface.x.copy()
        iota = res['iota']
        G = res['G']
        return np.concatenate([x, [iota, G]])

    def test_boozer_penalty_constraints_cpp_notcpp(self):
        """
        Test to verify cpp and python implementations of the BoozerLS objective return the same thing.
        """
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                for weight_inv_modB in [False, True]:
                    for (optimize_G, nphi, ntheta, mpol, ntor) in [(True, 1, 1, 3, 3), (False, 1, 1, 13, 2), (True, 2, 2, 10, 3), (False, 2, 1, 3, 4), (True, 6, 9, 3, 3), (False, 7, 8, 3, 4), (True, 3, 3, 3, 3), (False, 3, 3, 3, 5)]:
                        with self.subTest(surfacetype=surfacetype,
                                          stellsym=stellsym,
                                          optimize_G=optimize_G,
                                          weight_inv_modB=weight_inv_modB,
                                          mpol=mpol,
                                          ntor=ntor):
                            self.subtest_boozer_penalty_constraints_cpp_notcpp(surfacetype, stellsym, optimize_G, nphi, ntheta, weight_inv_modB, mpol, ntor)

    def test_boozer_penalty_constraints_derivatives2_weighted_unweighted_cpp_notcpp(self):
        for weight_inv_modB in [False, True]:
            with self.subTest(weight_inv_modB=weight_inv_modB):
                self.subtest_boozer_penalty_constraints_derivatives2_cpp_notcpp(weight_inv_modB)

    def subtest_boozer_penalty_constraints_derivatives2_cpp_notcpp(self, weight_inv_modB):
        np.random.seed(1)
        curves, currents, ma = get_ncsx_data()
        coils = coils_via_symmetries(curves, currents, 3, True)
        bs = BiotSavart(coils)
        bs_tf = BiotSavart(coils)
        current_sum = sum(abs(c.current.get_value()) for c in coils)

        s = get_surface(
            "SurfaceXYZTensorFourier", True, nphi=2, ntheta=2,
            thetas=[0.2432101234, 0.9832134],
            phis=[0.2234567989, 0.432123451],
            mpol=3, ntor=3,
        )
        s.fit_to_curve(ma, 0.1)
        s.x = s.x + np.random.rand(s.x.size)*1e-6

        tf = ToroidalFlux(s, bs_tf, nphi=51, ntheta=51)
        tf_target = 0.1
        G = 2.*np.pi*current_sum*(4*np.pi*10**(-7)/(2 * np.pi))
        iota = -0.3

        for current_I in [0.0, 0.37]:
            boozer_surface = BoozerSurface(bs, s, tf, tf_target, I=current_I)
            x = np.concatenate((s.get_dofs(), [iota, G]))
            self._assert_penalty_constraints_derivatives2_cpp_python_match(
                boozer_surface, x, optimize_G=True, weight_inv_modB=weight_inv_modB,
            )

    def subtest_boozer_penalty_constraints_cpp_notcpp(self, surfacetype, stellsym, optimize_G, nphi, ntheta, weight_inv_modB, mpol, ntor):

        np.random.seed(1)
        curves, currents, ma = get_ncsx_data()
        coils = coils_via_symmetries(curves, currents, 3, stellsym)
        bs = BiotSavart(coils)
        bs_tf = BiotSavart(coils)
        current_sum = sum(abs(c.current.get_value()) for c in coils)

        phis = None
        thetas = None
        if nphi == 1:
            phis = [0.2234567989]
        elif nphi == 2:
            phis = [0.2234567989, 0.432123451]

        if ntheta == 1:
            thetas = [0.2432101234]
        elif ntheta == 2:
            thetas = [0.2432101234, 0.9832134]

        s = get_surface(surfacetype, stellsym, nphi=nphi, ntheta=ntheta, thetas=thetas, phis=phis, mpol=mpol, ntor=ntor)
        s.fit_to_curve(ma, 0.1)
        s.x = s.x + np.random.rand(s.x.size)*1e-6

        tf = ToroidalFlux(s, bs_tf, nphi=51, ntheta=51)

        tf_target = 0.1
        for current_I in [0.0, 0.37]:
            boozer_surface = BoozerSurface(bs, s, tf, tf_target, I=current_I)

            iota = -0.3
            x = np.concatenate((s.get_dofs(), [iota]))
            if optimize_G:
                x = np.concatenate((x, [2.*np.pi*current_sum*(4*np.pi*10**(-7)/(2 * np.pi))]))
            H0, H1 = self._assert_penalty_constraints_cpp_python_match(
                boozer_surface, x, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB,
            )

        self._print_hessian_differences(H0, H1)

    def test_boozer_exact_coil_vjp_finite_current(self):
        """
        Taylor test for the exact-path coil VJP with nonzero net toroidal current I.
        """
        np.random.seed(2)
        current_I = 0.37
        phis = np.linspace(0, 1/3, 13, endpoint=False)
        thetas = np.linspace(0, 1, 13, endpoint=False)
        bs, G0, boozer_surface = self._make_area_boozer_surface(
            current_I=current_I, mpol=6, ntor=6, phis=phis, thetas=thetas,
            constraint_weight=None,
            options={"verbose": False},
        )
        res = boozer_surface.run_code(-0.406, G=G0)

        coeffs = bs.x.copy()
        direction = np.random.rand(*coeffs.shape) - 0.5
        lm = np.random.rand(int(res["mask"].sum()) + 1) - 0.5
        dfdx = boozer_surface_dexactresidual_dcoils_dcurrents_vjp(
            lm, boozer_surface, res["iota"], res["G"]
        )(bs)
        directional_derivative = dfdx @ direction

        def f(dofs):
            bs.x = dofs
            residual = boozer_surface_residual(
                boozer_surface.surface, res["iota"], res["G"], bs,
                derivatives=0, I=current_I,
            )[0]
            return np.dot(lm[:-1], residual[res["mask"]])

        self._assert_directional_fd_convergence(f, coeffs, direction, directional_derivative)
        bs.x = coeffs

    def test_boozer_lsqgrad_coil_vjp_finite_current(self):
        """
        Taylor test for the least-squares gradient coil VJP with nonzero net toroidal current I.
        """
        np.random.seed(3)
        current_I = 0.37
        phis = np.linspace(0, 1/3, 20, endpoint=False)
        thetas = np.linspace(0, 1, 20, endpoint=False)
        bs, G0, boozer_surface = self._make_area_boozer_surface(
            current_I=current_I, mpol=3, ntor=3, phis=phis, thetas=thetas,
            constraint_weight=100.0,
            options={"verbose": False, "weight_inv_modB": True},
        )
        boozer_surface.res = {"I": current_I}

        coeffs = bs.x.copy()
        direction = np.random.rand(*coeffs.shape) - 0.5
        iota = -0.3
        boozer = boozer_surface_residual_dB(
            boozer_surface.surface, iota, G0, bs, derivatives=1,
            weight_inv_modB=True, I=current_I,
        )
        num_points = 3 * boozer_surface.surface.quadpoints_phi.size * boozer_surface.surface.quadpoints_theta.size
        lm = np.random.rand(boozer[2].shape[1]) - 0.5
        dfdx = boozer_surface_dlsqgrad_dcoils_vjp(
            lm, boozer_surface, iota, G0, weight_inv_modB=True
        )(bs)
        directional_derivative = dfdx @ direction

        def f(dofs):
            bs.x = dofs
            residual_terms = boozer_surface_residual_dB(
                boozer_surface.surface, iota, G0, bs, derivatives=1,
                weight_inv_modB=True, I=current_I,
            )
            residual = residual_terms[0] / np.sqrt(num_points)
            lsq_gradient = residual_terms[2].T @ residual / np.sqrt(num_points)
            return lm @ lsq_gradient

        self._assert_directional_fd_convergence(f, coeffs, direction, directional_derivative)
        bs.x = coeffs

    def test_boozer_surface_quadpoints(self):
        """ 
        this unit test checks that the quadpoints mask for stellarator symmetric Boozer Surfaces are correctly initialized
        """
        for idx in range(4):
            with self.subTest(idx=idx):
                self.subtest_boozer_surface_quadpoints(idx)

    def subtest_boozer_surface_quadpoints(self, idx):
        mpol = 6
        ntor = 6
        nfp = 3

        if idx == 0:
            phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
            thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
            mask_true = np.ones((phis.size, thetas.size), dtype=bool)
            mask_true[:, mpol+1:] = False
            mask_true[ntor+1:, 0] = False
        elif idx == 1:
            phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
            thetas = np.linspace(0, 0.5, mpol+1, endpoint=False)
            mask_true = np.ones((phis.size, thetas.size), dtype=bool)
            mask_true[ntor+1:, 0] = False
        elif idx == 2:
            phis = np.linspace(0, 1/(2*nfp), ntor+1, endpoint=False)
            thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
            mask_true = np.ones((phis.size, thetas.size), dtype=bool)
            mask_true[0, mpol+1:] = False
        elif idx == 3:
            phis = np.linspace(0, 1., 2*ntor+1, endpoint=False)
            thetas = np.linspace(0, 1., 2*mpol+1, endpoint=False)

        s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=True, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)

        if idx < 3:  # the first three quadrature point sets should pass without issue.
            mask = s.get_stellsym_mask()
            assert np.all(mask == mask_true)
        else:
            with self.assertRaises(Exception):
                mask = s.get_stellsym_mask()

    def test_boozer_surface_type_assert(self):
        """
        this unit test checks that an exception is raised if a SurfaceRZFourier is passed to a BoozerSurface
        """
        mpol = 6
        ntor = 6
        nfp = 3
        phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
        thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
        s = SurfaceRZFourier(mpol=mpol, ntor=ntor, stellsym=True, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)

        base_curves, base_currents, ma = get_ncsx_data()
        coils = coils_via_symmetries(base_curves, base_currents, 3, True)
        bs = BiotSavart(coils)

        lab = Area(s)
        lab_target = 0.1

        with self.assertRaises(Exception):
            _ = BoozerSurface(bs, s, lab, lab_target)


if __name__ == "__main__":
    unittest.main()
