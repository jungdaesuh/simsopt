import unittest
import json

import numpy as np

from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.field.coil import coils_via_symmetries, Current
from simsopt.geo.curve import create_equally_spaced_curves
from simsopt.geo.curveobjectives import CurveLength
from simsopt.field.biotsavart import BiotSavart
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt._core.json import GSONDecoder, GSONEncoder, SIMSON


from pathlib import Path
TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
filename = TEST_DIR / 'input.LandremanPaul2021_QA'


class FluxObjectiveTests(unittest.TestCase):

    def test_definitions(self):
        """Verify the available definitions."""
        surf = SurfaceRZFourier.from_vmec_input(filename)
        ntheta = len(surf.quadpoints_theta)
        nphi = len(surf.quadpoints_phi)
        ncoils = 3

        base_curves = create_equally_spaced_curves(
            ncoils, surf.nfp, stellsym=surf.stellsym, R0=1.0, R1=0.5, order=6
        )
        base_currents = [Current(1e5) for i in range(ncoils)]
        coils = coils_via_symmetries(base_curves, base_currents, surf.nfp, surf.stellsym)
        bs = BiotSavart(coils)

        # Test definition = "quadratic flux":
        target = np.ones(surf.gamma().shape[0:2])
        J = SquaredFlux(surf, bs, target, definition="quadratic flux").J()
        bs.set_points(surf.gamma().reshape((-1, 3)))
        B = bs.B()
        normal = surf.normal().reshape((-1, 3))
        norm_normal = np.sqrt(normal[:, 0]**2 + normal[:, 1]**2 + normal[:, 2]**2)
        B_dot_n = np.sum(B * surf.unitnormal().reshape((-1, 3)), axis=1)
        should_be = 0.5 * sum((B_dot_n - target.reshape((-1,)))**2 * norm_normal) / (ntheta * nphi)
        np.testing.assert_allclose(J, should_be)

        # Test definition = "normalized":
        J2 = SquaredFlux(surf, bs, target, definition="normalized").J()
        mod_B_squared = np.sum(B * B, axis=1)
        numerator = 0.5 * sum(
            (B_dot_n - target.reshape((-1,)))**2 * norm_normal
        ) / (ntheta * nphi)
        denominator = sum(mod_B_squared * norm_normal) / (ntheta * nphi)
        np.testing.assert_allclose(J2, numerator / denominator)

        # Test definition = "local":
        J3 = SquaredFlux(surf, bs, target, definition="local").J()
        should_be3 = 0.5 * sum(
            (B_dot_n - target.reshape((-1,)))**2 / mod_B_squared * norm_normal
        ) / (ntheta * nphi)
        np.testing.assert_allclose(J3, should_be3)

        with self.assertRaises(ValueError):
            SquaredFlux(surf, bs, target, definition="foobar")

    def check_taylor_test(self, J):
        dofs = J.x
        np.random.seed(1)
        h = np.random.uniform(size=dofs.shape)
        dJ0 = J.dJ()
        dJh = sum(dJ0 * h)
        err_old = 1e10
        for i in range(11, 17):
            eps = 0.5 ** i
            J.x = dofs + eps * h
            J1 = J.J()
            J.x = dofs - eps * h
            J2 = J.J()
            err = np.abs((J1 - J2) / (2 * eps) - dJh)
            # print(f"i: {i}  err: {err}  err_old: {err_old}  err/err_old: {err/err_old}")
            assert err < 0.6 ** 2 * err_old
            err_old = err

        J_str = json.dumps(SIMSON(J), cls=GSONEncoder)
        J_regen = json.loads(J_str, cls=GSONDecoder)
        self.assertAlmostEqual(J.J(), J_regen.J())

    def test_derivatives(self):
        """Verify correctness of SquaredFlux.dJ()"""
        s = SurfaceRZFourier.from_vmec_input(filename)
        ncoils = 4

        base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=s.stellsym, R0=1.0, R1=0.5, order=6)
        base_currents = [Current(1e5) for i in range(ncoils)]
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, s.stellsym)
        bs = BiotSavart(coils)

        for definition in ["quadratic flux", "normalized", "local"]:
            with self.subTest(definition=definition):
                Jf = SquaredFlux(s, bs, definition=definition)
                self.check_taylor_test(Jf)

                target = np.zeros(s.gamma().shape[0:2])
                Jf2 = SquaredFlux(s, bs, target, definition=definition)
                self.check_taylor_test(Jf2)
                target = np.ones(s.gamma().shape[0:2])
                Jf3 = SquaredFlux(s, bs, target, definition=definition)
                self.check_taylor_test(Jf3)

                Jls = [CurveLength(c) for c in base_curves]

                ALPHA = 1e-5
                JF_scaled_summed = Jf + ALPHA * sum(Jls)
                self.check_taylor_test(JF_scaled_summed)

    def test_surface_dof_change_refreshes_field_points(self):
        s = SurfaceRZFourier(mpol=1, ntor=0, nfp=1)
        base_curves = create_equally_spaced_curves(
            2,
            s.nfp,
            stellsym=s.stellsym,
            R0=1.0,
            R1=0.3,
            order=3,
        )
        base_currents = [Current(1e5) for _ in base_curves]
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, s.stellsym)
        bs = BiotSavart(coils)
        SquaredFlux(s, bs, definition="quadratic flux")

        initial_points = bs.get_points_cart().copy()
        dofs = s.get_dofs()
        dofs[0] *= 1.1
        s.set_dofs(dofs)
        moved_points = s.gamma().reshape((-1, 3))
        fresh_bs = BiotSavart(coils)
        SquaredFlux(s, fresh_bs, definition="quadratic flux")

        self.assertFalse(np.allclose(initial_points, moved_points))
        np.testing.assert_allclose(bs.get_points_cart(), moved_points)
        np.testing.assert_allclose(bs.get_points_cart(), fresh_bs.get_points_cart())
