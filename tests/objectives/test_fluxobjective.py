import unittest
import json

import numpy as np

from simsopt._core.derivative import Derivative
from simsopt._core.optimizable import Optimizable
from simsopt._core.util import ObjectiveFailure
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.field.coil import coils_via_symmetries, Current
from simsopt.field import BiotSavartJAX
from simsopt.geo.curve import create_equally_spaced_curves
from simsopt.geo.curveobjectives import CurveLength
from simsopt.field.biotsavart import BiotSavart
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.objectives.fluxobjective_jax import SquaredFluxJAX
from simsopt.field.coil import Coil
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt._core.json import GSONDecoder, GSONEncoder, SIMSON


from pathlib import Path

TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
filename = TEST_DIR / "input.LandremanPaul2021_QA"


class _FluxObjectiveFakeSurface:
    def __init__(self, normal):
        self._normal = np.asarray(normal, dtype=np.float64)

    def gamma(self):
        return np.zeros_like(self._normal)

    def normal(self):
        return self._normal


class _FluxObjectiveFakeField(Optimizable):
    def __init__(self, B):
        self._B = np.asarray(B, dtype=np.float64)
        super().__init__(x0=np.zeros(self._B.size))

    def recompute_bell(self, parent=None):
        del parent

    def set_points(self, xyz):
        del xyz

    def B(self):
        return self._B.reshape((-1, 3))

    def B_vjp(self, dJdB):
        return Derivative({self: np.asarray(dJdB).reshape((-1,))})


def _make_fake_flux_objective(*, definition, normal, B, target):
    surface = _FluxObjectiveFakeSurface(normal)
    field = _FluxObjectiveFakeField(B)
    objective = SquaredFlux(
        surface, field, target=np.asarray(target), definition=definition
    )
    return objective, field


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
        coils = coils_via_symmetries(
            base_curves, base_currents, surf.nfp, surf.stellsym
        )
        bs = BiotSavart(coils)

        # Test definition = "quadratic flux":
        target = np.ones(surf.gamma().shape[0:2])
        J = SquaredFlux(surf, bs, target, definition="quadratic flux").J()
        bs.set_points(surf.gamma().reshape((-1, 3)))
        B = bs.B()
        normal = surf.normal().reshape((-1, 3))
        norm_normal = np.sqrt(normal[:, 0] ** 2 + normal[:, 1] ** 2 + normal[:, 2] ** 2)
        B_dot_n = np.sum(B * surf.unitnormal().reshape((-1, 3)), axis=1)
        should_be = (
            0.5
            * sum((B_dot_n - target.reshape((-1,))) ** 2 * norm_normal)
            / (ntheta * nphi)
        )
        np.testing.assert_allclose(J, should_be)

        # Test definition = "normalized":
        J2 = SquaredFlux(surf, bs, target, definition="normalized").J()
        mod_B_squared = np.sum(B * B, axis=1)
        numerator = (
            0.5
            * sum((B_dot_n - target.reshape((-1,))) ** 2 * norm_normal)
            / (ntheta * nphi)
        )
        denominator = sum(mod_B_squared * norm_normal) / (ntheta * nphi)
        np.testing.assert_allclose(J2, numerator / denominator)

        # Test definition = "local":
        J3 = SquaredFlux(surf, bs, target, definition="local").J()
        should_be3 = (
            0.5
            * sum((B_dot_n - target.reshape((-1,))) ** 2 / mod_B_squared * norm_normal)
            / (ntheta * nphi)
        )
        np.testing.assert_allclose(J3, should_be3)

        with self.assertRaises(ValueError):
            SquaredFlux(surf, bs, target, definition="foobar")

    def check_taylor_test(self, J):
        dofs = J.x
        rng = np.random.default_rng(1)
        h = rng.uniform(size=dofs.shape)
        dJ0 = J.dJ()
        dJh = sum(dJ0 * h)
        err_old = 1e10
        for i in range(11, 17):
            eps = 0.5**i
            J.x = dofs + eps * h
            J1 = J.J()
            J.x = dofs - eps * h
            J2 = J.J()
            err = np.abs((J1 - J2) / (2 * eps) - dJh)
            # print(f"i: {i}  err: {err}  err_old: {err_old}  err/err_old: {err/err_old}")
            assert err < 0.6**2 * err_old
            err_old = err

        J_str = json.dumps(SIMSON(J), cls=GSONEncoder)
        J_regen = json.loads(J_str, cls=GSONDecoder)
        self.assertAlmostEqual(J.J(), J_regen.J())

    def test_derivatives(self):
        """Verify correctness of SquaredFlux.dJ()"""
        s = SurfaceRZFourier.from_vmec_input(filename)
        ncoils = 4

        base_curves = create_equally_spaced_curves(
            ncoils, s.nfp, stellsym=s.stellsym, R0=1.0, R1=0.5, order=6
        )
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

    def test_quadratic_flux_gradient_handles_zero_normals(self):
        objective, field = _make_fake_flux_objective(
            definition="quadratic flux",
            normal=np.zeros((1, 1, 3)),
            B=np.zeros((1, 1, 3)),
            target=[[1.0]],
        )

        np.testing.assert_allclose(objective.dJ(), np.zeros(field.local_dof_size))

    def test_singular_local_returns_inf_and_raises_gradient_failure(self):
        objective, _field = _make_fake_flux_objective(
            definition="local",
            normal=[[[1.0, 0.0, 0.0]]],
            B=[[[0.0, 0.0, 0.0]]],
            target=[[1.0]],
        )

        self.assertTrue(np.isinf(objective.J()))
        with self.assertRaisesRegex(ObjectiveFailure, "gradient is singular"):
            objective.dJ()

    def test_singular_normalized_returns_inf_and_raises_gradient_failure(self):
        objective, _field = _make_fake_flux_objective(
            definition="normalized",
            normal=[[[1.0, 0.0, 0.0]]],
            B=[[[0.0, 0.0, 0.0]]],
            target=[[1.0]],
        )

        self.assertTrue(np.isinf(objective.J()))
        with self.assertRaisesRegex(ObjectiveFailure, "gradient is singular"):
            objective.dJ()

    def test_squaredfluxjax_requires_surface_spec(self):
        curve = CurveXYZFourier(16, 1)
        curve.x = np.array([1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1])
        field = BiotSavartJAX([Coil(curve, Current(1.0))])
        surface = _FluxObjectiveFakeSurface(np.zeros((1, 1, 3)))

        with self.assertRaisesRegex(NotImplementedError, "surface_spec"):
            SquaredFluxJAX(surface, field)
