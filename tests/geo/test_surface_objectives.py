import unittest
import numpy as np
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo.surfaceobjectives import ToroidalFlux, QfmResidual, parameter_derivatives, Volume, PrincipalCurvature, MajorRadius, Iotas, NonQuasiSymmetricRatio, BoozerResidual
from simsopt.configs.zoo import get_data
from .surface_test_helpers import get_surface, get_exact_surface, get_boozer_surface


surfacetypes_list = ["SurfaceXYZFourier", "SurfaceRZFourier",
                     "SurfaceXYZTensorFourier"]
stellsym_list = [True, False]


def taylor_test1(f, df, x, epsilons=None, direction=None, atol=1e-9):
    np.random.seed(1)
    f(x)
    if direction is None:
        direction = np.random.rand(*(x.shape))-0.5
    dfx = df(x)@direction
    if epsilons is None:
        epsilons = np.power(2., -np.asarray(range(10, 20)))
    print("###################################################################")
    err_old = 1e9
    for eps in epsilons:
        fpluseps = f(x + eps * direction)
        fminuseps = f(x - eps * direction)
        dfest = (fpluseps-fminuseps)/(2*eps)
        err = np.linalg.norm(dfest - dfx)
        print("taylor test1: ", err, err/err_old)
        np.testing.assert_array_less(err, max(atol, 0.35 * err_old),
                                     err_msg=f"Taylor test failed: err={err:.2e}, threshold={max(atol, 0.35 * err_old):.2e}, "
                                             f"err_old={err_old:.2e}, ratio={err/err_old:.4f}")
        err_old = err
    print("###################################################################")


def taylor_test2(f, df, d2f, x, epsilons=None, direction1=None, direction2=None):
    np.random.seed(1)
    if direction1 is None:
        direction1 = np.random.rand(*(x.shape))-0.5
    if direction2 is None:
        direction2 = np.random.rand(*(x.shape))-0.5

    f(x)
    df0 = df(x) @ direction1
    d2fval = direction2.T @ d2f(x) @ direction1
    if epsilons is None:
        epsilons = np.power(2., -np.asarray(range(7, 20)))
    print("###################################################################")
    err_old = 1e9
    for eps in epsilons:
        fpluseps = df(x + eps * direction2) @ direction1
        d2fest = (fpluseps-df0)/eps
        err = np.abs(d2fest - d2fval)
        print('taylor test2: ', err, err/err_old)
        assert err < 0.6 * err_old
        err_old = err
    print("###################################################################")


class ToroidalFluxTests(unittest.TestCase):
    def test_toroidal_flux_is_constant(self):
        """
        this test ensures that the toroidal flux does not change, regardless
        of the cross section (varphi = constant) across which it is computed
        """
        s = get_exact_surface()
        base_curves, base_currents, ma, nfp, bs= get_data("ncsx")
        bs_tf = BiotSavart(bs.coils)

        gamma = s.gamma()
        num_phi = gamma.shape[0]

        tf_list = np.zeros((num_phi,))
        for idx in range(num_phi):
            tf = ToroidalFlux(s, bs_tf, idx=idx)
            tf_list[idx] = tf.J()
        mean_tf = np.mean(tf_list)

        max_err = np.max(np.abs(mean_tf - tf_list)) / mean_tf
        assert max_err < 1e-2

    def test_toroidal_flux_first_derivative(self):
        """
        Taylor test for partial derivatives of toroidal flux with respect to surface coefficients
        """

        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    self.subtest_toroidal_flux1(surfacetype, stellsym)

    def test_toroidal_flux_second_derivative(self):
        """
        Taylor test for Hessian of toroidal flux with respect to surface coefficients
        """

        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    self.subtest_toroidal_flux2(surfacetype, stellsym)

    def test_toroidal_flux_partial_derivatives_wrt_coils(self):
        """
        Taylor test for partial derivative of toroidal flux with respect to surface coefficients
        """

        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    self.subtest_toroidal_flux3(surfacetype, stellsym)

    def subtest_toroidal_flux1(self, surfacetype, stellsym):
        base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
        bs_tf = BiotSavart(bs.coils)
        s = get_surface(surfacetype, stellsym)

        tf = ToroidalFlux(s, bs_tf)
        coeffs = s.x

        def f(dofs):
            s.x = dofs
            return tf.J()

        def df(dofs):
            s.x = dofs
            return tf.dJ_by_dsurfacecoefficients()
        taylor_test1(f, df, coeffs)

    def subtest_toroidal_flux2(self, surfacetype, stellsym):
        base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
        s = get_surface(surfacetype, stellsym)

        tf = ToroidalFlux(s, bs)
        coeffs = s.x

        def f(dofs):
            s.x = dofs
            return tf.J()

        def df(dofs):
            s.x = dofs
            return tf.dJ_by_dsurfacecoefficients()

        def d2f(dofs):
            s.x = dofs
            return tf.d2J_by_dsurfacecoefficientsdsurfacecoefficients()

        taylor_test2(f, df, d2f, coeffs)

    def subtest_toroidal_flux3(self, surfacetype, stellsym):
        base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
        bs_tf = BiotSavart(bs.coils)
        s = get_surface(surfacetype, stellsym)

        tf = ToroidalFlux(s, bs_tf)
        coeffs = bs_tf.x

        def f(dofs):
            bs_tf.x = dofs
            return tf.J()

        def df(dofs):
            bs_tf.x = dofs
            return tf.dJ_by_dcoils()(bs_tf)
        taylor_test1(f, df, coeffs)


class PrincipalCurvatureTests(unittest.TestCase):
    def test_principal_curvature_first_derivative(self):
        """
        Taylor test for gradient of principal curvature metric.
        """
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    self.subtest_principal_curvature(surfacetype, stellsym)

    def subtest_principal_curvature(self, surfacetype, stellsym):
        s = get_surface(surfacetype, stellsym)

        pc = PrincipalCurvature(s, kappamax1=1, kappamax2=2.2, weight1=1, weight2=2.)
        coeffs = s.x

        def f(dofs):
            s.x = dofs
            return pc.J()

        def df(dofs):
            s.x = dofs
            return pc.dJ()

        taylor_test1(f, df, coeffs, epsilons=np.power(2., -np.asarray(range(13, 22))))


class ParameterDerivativesTest(unittest.TestCase):
    def test_parameter_derivatives_volume(self):
        """
        Test that parameter derivatives of volume (shape_gradient = 1) match
        parameter derivatives computed from Volume class.
        """
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    self.subtest_volume(surfacetype, stellsym)

    def subtest_volume(self, surfacetype, stellsym):
        from simsopt.geo import Surface
        s = get_surface(surfacetype, stellsym, mpol=7, ntor=6,
                        ntheta=32, nphi=31, full=True)
        dofs = s.get_dofs()
        vol = Volume(s, range=Surface.RANGE_FIELD_PERIOD)
        dvol_sg = parameter_derivatives(s, np.ones_like(s.gamma()[:, :, 0]))
        dvol = vol.dJ_by_dsurfacecoefficients()
        for i in range(len(dofs)):
            self.assertAlmostEqual(dvol_sg[i], dvol[i], places=10)


class QfmTests(unittest.TestCase):
    def test_qfm_surface_derivative(self):
        """
        Taylor test for derivative of qfm metric wrt surface parameters
        """
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    self.subtest_qfm1(surfacetype, stellsym)

    def subtest_qfm1(self, surfacetype, stellsym):
        base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
        s = get_surface(surfacetype, stellsym)
        coeffs = s.x
        qfm = QfmResidual(s, bs)

        def f(dofs):
            s.x = dofs
            return qfm.J()

        def df(dofs):
            s.x = dofs
            return qfm.dJ_by_dsurfacecoefficients()
        taylor_test1(f, df, coeffs,
                     epsilons=np.power(2., -np.asarray(range(13, 22))))


class MajorRadiusTests(unittest.TestCase):
    def test_major_radius_derivative(self):
        """
        Taylor test for derivative of surface major radius wrt coil parameters
        """
        for boozer_type in ['exact', 'ls']:
            for label in ["Volume", "ToroidalFlux"]:
                for optimize_G in [True, False]:
                    for weight_inv_modB in [True, False]:
                        with self.subTest(label=label, boozer_type=boozer_type, optimize_G=optimize_G):
                            if boozer_type == 'ls' and label == 'ToroidalFlux':
                                continue
                            if boozer_type == 'exact' and optimize_G is False:
                                continue
                            if boozer_type == 'exact' and weight_inv_modB:
                                continue
                            self.subtest_major_radius_surface_derivative(label, boozer_type, optimize_G, weight_inv_modB)

    def subtest_major_radius_surface_derivative(self, label, boozer_type, optimize_G, weight_inv_modB):
        bs, boozer_surface = get_boozer_surface(label=label, nphi=51, ntheta=51, boozer_type=boozer_type, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB)
        coeffs = bs.x
        mr = MajorRadius(boozer_surface)

        def f(dofs):
            bs.x = dofs
            return mr.J()

        def df(dofs):
            bs.x = dofs
            return mr.dJ()
        taylor_test1(f, df, coeffs,
                     epsilons=np.power(2., -np.asarray(range(13, 18))))


class IotasTests(unittest.TestCase):
    def test_iotas_derivative(self):
        """
        Taylor test for derivative of surface rotational transform wrt coil parameters
        """

        for boozer_type in ['exact', 'ls']:
            for label in ["Volume", "ToroidalFlux"]:
                for optimize_G in [True, False]:
                    for weight_inv_modB in [True, False]:
                        if boozer_type == 'ls' and label == 'ToroidalFlux':
                            continue
                        if boozer_type == 'exact' and optimize_G is False:
                            continue
                        if boozer_type == 'exact' and weight_inv_modB:
                            continue
                        with self.subTest(label=label, boozer_type=boozer_type, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB):
                            self.subtest_iotas_derivative(label, boozer_type, optimize_G, weight_inv_modB)

    def subtest_iotas_derivative(self, label, boozer_type, optimize_G, weight_inv_modB):
        """
        Taylor test for derivative of surface rotational transform wrt coil parameters
        """
        np.random.seed(1)  # Fixed seed for reproducibility across platforms

        bs, boozer_surface = get_boozer_surface(label=label, boozer_type=boozer_type, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB)
        coeffs = bs.x
        io = Iotas(boozer_surface)

        def f(dofs):
            bs.x = dofs
            return io.J()

        def df(dofs):
            bs.x = dofs
            return io.dJ()

        taylor_test1(f, df, coeffs,
                     epsilons=np.power(2., -np.asarray(range(13, 19))), atol=2e-8)


class NonQSRatioTests(unittest.TestCase):
    def test_fixed_surface_value_matches_production_value(self):
        bs, boozer_surface = get_boozer_surface(
            label="Volume",
            boozer_type="exact",
            optimize_G=True,
        )
        objective = NonQuasiSymmetricRatio(
            boozer_surface,
            bs,
            quasi_poloidal=False,
        )

        production_value = objective.J()
        fixed_surface_value, _ = objective.fixed_surface_value_and_derivative()

        np.testing.assert_allclose(
            fixed_surface_value,
            production_value,
            rtol=0.0,
            atol=5e-15,
        )

    def test_fixed_surface_physical_partial_matches_directional_difference(self):
        bs, boozer_surface = get_boozer_surface(
            label="Volume",
            boozer_type="exact",
            optimize_G=True,
        )
        objective = NonQuasiSymmetricRatio(
            boozer_surface,
            bs,
            quasi_poloidal=False,
        )
        surface = objective.in_surface
        baseline = surface.get_dofs()
        direction = np.random.default_rng(20260725).standard_normal(baseline.size)
        direction /= np.linalg.norm(direction)

        _, derivative = objective.fixed_surface_value_and_derivative()
        analytic = float(derivative.data[surface] @ direction)
        step = 1.0e-6
        surface.set_dofs(baseline + step * direction)
        plus, _ = objective.fixed_surface_value_and_derivative()
        surface.set_dofs(baseline - step * direction)
        minus, _ = objective.fixed_surface_value_and_derivative()
        surface.set_dofs(baseline)
        finite_difference = (plus - minus) / (2.0 * step)

        np.testing.assert_allclose(
            analytic,
            finite_difference,
            rtol=3.0e-6,
            atol=1.0e-10,
        )

    def test_nonQSratio_derivative(self):
        """
        Taylor test for derivative of surface non QS ratio wrt coil parameters
        """
        for boozer_type in ['exact', 'ls']:
            for label in ["Volume", "ToroidalFlux"]:
                for weight_inv_modB in [True, False]:
                    for optimize_G in [True, False]:
                        for fix_coil_dof in [True, False]:
                            if boozer_type == 'ls' and label == 'ToroidalFlux':
                                continue
                            if boozer_type == 'exact' and optimize_G is False:
                                continue
                            if boozer_type == 'exact' and weight_inv_modB:
                                continue
                            for axis in [False, True]:
                                with self.subTest(label=label, axis=axis, boozer_type=boozer_type, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB, fix_coil_dof=fix_coil_dof):
                                    self.subtest_nonQSratio_derivative(label, axis, boozer_type, optimize_G, weight_inv_modB, fix_coil_dof)

    def subtest_nonQSratio_derivative(self, label, axis, boozer_type, optimize_G, weight_inv_modB, fix_coil_dof):
        bs, boozer_surface = get_boozer_surface(label=label, boozer_type=boozer_type, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB)

        if fix_coil_dof:
            bs.coils[0].curve.fix('xc(0)')

        coeffs = bs.x
        io = NonQuasiSymmetricRatio(boozer_surface, bs, quasi_poloidal=axis)

        def f(dofs):
            bs.x = dofs
            return io.J()

        def df(dofs):
            bs.x = dofs
            return io.dJ()

        taylor_test1(f, df, coeffs,
                     epsilons=np.power(2., -np.asarray(range(13, 19))))


class BoozerResidualTests(unittest.TestCase):
    def test_fixed_surface_value_matches_production_value(self):
        for optimize_G in (True, False):
            with self.subTest(optimize_G=optimize_G):
                bs, boozer_surface = get_boozer_surface(
                    label="Volume",
                    boozer_type="ls",
                    optimize_G=optimize_G,
                    weight_inv_modB=False,
                )
                objective = BoozerResidual(boozer_surface, bs)

                production_value = objective.J()
                solved_G = boozer_surface.res["G"]
                fixed_surface_value, _, y_partial = (
                    objective.fixed_surface_value_derivative_and_y_partial(
                        float(boozer_surface.res["iota"]),
                        None if solved_G is None else float(solved_G),
                        weight_inv_modB=bool(
                            boozer_surface.res["weight_inv_modB"]
                        ),
                    )
                )

                np.testing.assert_allclose(
                    fixed_surface_value,
                    production_value,
                    rtol=0.0,
                    atol=5e-15,
                )
                self.assertEqual(y_partial.shape, (2 if optimize_G else 1,))

    def test_fixed_surface_y_partial_matches_directional_difference(self):
        bs, boozer_surface = get_boozer_surface(
            label="Volume",
            boozer_type="ls",
            optimize_G=True,
            weight_inv_modB=False,
        )
        objective = BoozerResidual(boozer_surface, bs)
        iota = float(boozer_surface.res["iota"])
        G = float(boozer_surface.res["G"])
        weight_inv_modB = bool(boozer_surface.res["weight_inv_modB"])
        _, _, y_partial = objective.fixed_surface_value_derivative_and_y_partial(
            iota,
            G,
            weight_inv_modB=weight_inv_modB,
        )
        direction = np.asarray([0.6, -0.8], dtype=np.float64)
        step = 1.0e-6
        plus, _, _ = objective.fixed_surface_value_derivative_and_y_partial(
            iota + step * direction[0],
            G + step * direction[1],
            weight_inv_modB=weight_inv_modB,
        )
        minus, _, _ = objective.fixed_surface_value_derivative_and_y_partial(
            iota - step * direction[0],
            G - step * direction[1],
            weight_inv_modB=weight_inv_modB,
        )
        finite_difference = (plus - minus) / (2.0 * step)
        analytic = float(y_partial @ direction)

        np.testing.assert_allclose(
            analytic,
            finite_difference,
            rtol=3.0e-6,
            atol=1.0e-10,
        )

    def test_fixed_surface_coil_partial_matches_directional_difference(self):
        bs, boozer_surface = get_boozer_surface(
            label="Volume",
            boozer_type="ls",
            optimize_G=True,
            weight_inv_modB=False,
        )
        objective = BoozerResidual(boozer_surface, bs)
        iota = float(boozer_surface.res["iota"])
        G = float(boozer_surface.res["G"])
        weight_inv_modB = bool(boozer_surface.res["weight_inv_modB"])
        baseline = bs.x
        direction = np.random.default_rng(20260725).standard_normal(baseline.size)
        direction /= np.linalg.norm(direction)
        _, derivative, _ = objective.fixed_surface_value_derivative_and_y_partial(
            iota,
            G,
            weight_inv_modB=weight_inv_modB,
        )
        analytic = float(derivative(bs) @ direction)
        step = 1.0e-6
        bs.x = baseline + step * direction
        plus, _, _ = objective.fixed_surface_value_derivative_and_y_partial(
            iota,
            G,
            weight_inv_modB=weight_inv_modB,
        )
        bs.x = baseline - step * direction
        minus, _, _ = objective.fixed_surface_value_derivative_and_y_partial(
            iota,
            G,
            weight_inv_modB=weight_inv_modB,
        )
        bs.x = baseline
        finite_difference = (plus - minus) / (2.0 * step)

        np.testing.assert_allclose(
            analytic,
            finite_difference,
            rtol=3.0e-6,
            atol=1.0e-10,
        )

    def test_fixed_surface_physical_partial_survives_fix_all(self):
        bs, boozer_surface = get_boozer_surface(
            label="Volume",
            boozer_type="ls",
            optimize_G=True,
            weight_inv_modB=False,
        )
        objective = BoozerResidual(boozer_surface, bs)
        surface = objective.in_surface
        baseline = surface.get_dofs()
        rng = np.random.default_rng(20260725)
        offset = rng.standard_normal(baseline.size)
        offset /= np.linalg.norm(offset)
        baseline = baseline + 1.0e-3 * offset
        surface.set_dofs(baseline)
        surface.fix_all()
        iota = float(boozer_surface.res["iota"])
        G = float(boozer_surface.res["G"])
        weight_inv_modB = bool(boozer_surface.res["weight_inv_modB"])
        _, derivative, _ = objective.fixed_surface_value_derivative_and_y_partial(
            iota,
            G,
            weight_inv_modB=weight_inv_modB,
        )
        physical_partial = derivative.data[surface]
        self.assertEqual(physical_partial.shape, baseline.shape)
        self.assertTrue(np.all(np.isfinite(physical_partial)))

        direction = rng.standard_normal(baseline.size)
        direction /= np.linalg.norm(direction)
        analytic = float(physical_partial @ direction)
        step = 1.0e-6
        surface.set_dofs(baseline + step * direction)
        plus, _, _ = objective.fixed_surface_value_derivative_and_y_partial(
            iota,
            G,
            weight_inv_modB=weight_inv_modB,
        )
        surface.set_dofs(baseline - step * direction)
        minus, _, _ = objective.fixed_surface_value_derivative_and_y_partial(
            iota,
            G,
            weight_inv_modB=weight_inv_modB,
        )
        surface.set_dofs(baseline)
        finite_difference = (plus - minus) / (2.0 * step)

        np.testing.assert_allclose(
            analytic,
            finite_difference,
            rtol=3.0e-6,
            atol=1.0e-10,
        )

    def test_boozerresidual_derivative(self):
        """
        Taylor test for derivative of surface non QS ratio wrt coil parameters
        """
        for label in ["Volume"]:
            for optimize_G in [True, False]:
                for weight_inv_modB in [True, False]:
                    with self.subTest(label=label, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB):
                        self.subtest_boozerresidual_derivative(label, optimize_G, weight_inv_modB)

    def subtest_boozerresidual_derivative(self, label, optimize_G, weight_inv_modB):
        bs, boozer_surface = get_boozer_surface(label=label, boozer_type='ls', optimize_G=optimize_G, weight_inv_modB=weight_inv_modB)
        coeffs = bs.x
        br = BoozerResidual(boozer_surface, bs)

        def f(dofs):
            bs.x = dofs
            return br.J()

        def df(dofs):
            bs.x = dofs
            return br.dJ()

        taylor_test1(f, df, coeffs,
                     epsilons=np.power(2., -np.asarray(range(13, 19))))


class LabelTests(unittest.TestCase):
    def test_label_surface_derivative1(self):
        for label in ["Volume", "ToroidalFlux", "Area", "AspectRatio"]:
            for stellsym in stellsym_list:
                for nphi, ntheta in [(13, 14), (None, None), (13, None), (None, 14)]:
                    with self.subTest(label=label, stellsym=stellsym, converge=stellsym):
                        # don't converge the BoozerSurface when stellsym=False because it takes a long time
                        # for a unit test
                        self.subtest_label_derivative1(label, stellsym=stellsym, converge=stellsym, nphi=nphi, ntheta=ntheta)

    def subtest_label_derivative1(self, label, stellsym, converge, nphi, ntheta):
        bs, boozer_surface = get_boozer_surface(label=label, nphi=nphi, ntheta=ntheta, converge=converge, stellsym=stellsym)
        surface = boozer_surface.surface
        label = boozer_surface.label
        coeffs = surface.x

        def f(dofs):
            surface.x = dofs
            return label.J()

        def df(dofs):
            surface.x = dofs
            return label.dJ(partials=True)(surface)

        taylor_test1(f, df, coeffs,
                     epsilons=np.power(2., -np.asarray(range(12, 18))))

    def test_label_surface_derivative2(self):
        for label in ["Volume", "ToroidalFlux", "Area", "AspectRatio"]:
            with self.subTest(label=label):
                self.subtest_label_derivative2(label)

    def subtest_label_derivative2(self, label):
        bs, boozer_surface = get_boozer_surface(label=label)
        surface = boozer_surface.surface
        label = boozer_surface.label
        coeffs = surface.x

        def f(dofs):
            surface.x = dofs
            return label.J()

        def df(dofs):
            surface.x = dofs
            return label.dJ_by_dsurfacecoefficients()

        def d2f(dofs):
            surface.x = dofs
            return label.d2J_by_dsurfacecoefficientsdsurfacecoefficients()

        taylor_test2(f, df, d2f, coeffs,
                     epsilons=np.power(2., -np.asarray(range(13, 19))))
