import unittest
import json
import numpy as np

from simsopt._core.json import GSONEncoder, GSONDecoder, SIMSON
from simsopt.geo.curveplanaronsurface import CurvePlanarOnSurface
from simsopt.geo.curvesuperellipse import CurveSuperEllipse
from simsopt.geo.curveplanarfourier import CurvePlanarFourier, JaxCurvePlanarFourier
from simsopt.geo.curveplanarellipticalcylindrical import CurvePlanarEllipticalCylindrical
from simsopt.geo.curvehelical import CurveHelical
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo import parameters
from simsopt.field import BiotSavart, Coil, Current
from simsopt.objectives import SquaredFlux

parameters['jit'] = False


def get_surface():
    """A non-axisymmetric surface, so the local frame is not trivial."""
    s = SurfaceRZFourier(mpol=2, ntor=1, nfp=3, stellsym=True,
                         quadpoints_phi=np.linspace(0, 1, 8, endpoint=False),
                         quadpoints_theta=np.linspace(0, 1, 6, endpoint=False))
    s.set_rc(0, 0, 1.0)
    s.set_rc(1, 0, 0.20)
    s.set_zs(1, 0, 0.25)
    s.set_rc(1, 1, 0.03)
    s.set_zs(1, 1, -0.02)
    return s


def get_base(nquadpoints=32):
    return CurveSuperEllipse(nquadpoints, 0.04, 0.06, 4.0)


def all_planar_bases():
    """One instance of every type in PLANAR_CURVE_TYPES."""
    planar_fourier = CurvePlanarFourier(32, 1)
    planar_fourier.set('rc(0)', 0.05)
    planar_fourier.set('q0', 1.0)
    jax_planar_fourier = JaxCurvePlanarFourier(32, 1)
    jax_planar_fourier.set('rc(0)', 0.05)
    jax_planar_fourier.set('q0', 1.0)
    cylindrical = CurvePlanarEllipticalCylindrical(32, 0.05, 0.06)
    cylindrical.set('R0', 1.0)
    return [get_base(), planar_fourier, jax_planar_fourier, cylindrical]


class PlanarOnSurfaceTesting(unittest.TestCase):

    def test_anchored_to_surface(self):
        # Centre on the quadpoint, plane normal on the surface normal, and the
        # in-plane x direction on the requested surface tangent.
        s = get_surface()
        for align, dash in [("theta", s.gammadash2), ("phi", s.gammadash1)]:
            with self.subTest(align=align):
                curve = CurvePlanarOnSurface(get_base(), s, 3, 2, align=align)
                np.testing.assert_allclose(curve.center(), s.gamma()[3, 2])
                np.testing.assert_allclose(curve.unitnormal(), s.unitnormal()[3, 2])

                rotation = curve.rotation_matrix()
                np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
                self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=12)

                nhat = curve.unitnormal()
                tangent = dash()[3, 2]
                tangent = tangent - (tangent @ nhat) * nhat
                tangent /= np.linalg.norm(tangent)
                np.testing.assert_allclose(rotation[:, 0], tangent, atol=1e-13)

    def test_accepts_planar_curve_types(self):
        # Every type on the whitelist must place correctly, including the ones
        # whose canonical pose is not the x-y plane at the origin.
        s = get_surface()
        for base in all_planar_bases():
            with self.subTest(base=type(base).__name__):
                curve = CurvePlanarOnSurface(base, s, 3, 2)
                offset = curve.gamma() - curve.center()
                self.assertLess(np.abs(offset @ curve.unitnormal()).max(), 1e-13)

    def test_rejects_nonplanar_base(self):
        with self.assertRaises(TypeError):
            CurvePlanarOnSurface(CurveHelical(16, 1), get_surface(), 3, 2)

    def test_rejects_invalid_arguments(self):
        s = get_surface()
        nphi = len(s.quadpoints_phi)
        ntheta = len(s.quadpoints_theta)
        with self.assertRaises(ValueError):
            CurvePlanarOnSurface(get_base(), s, 3, 2, align="typo")
        for iphi, itheta in [(nphi, 0), (0, ntheta), (-1, 0)]:
            with self.subTest(iphi=iphi, itheta=itheta):
                with self.assertRaises(ValueError):
                    CurvePlanarOnSurface(get_base(), s, iphi, itheta)

    def test_coil_follows_surface(self):
        # The whole point of the class: the base curve's geometry is fixed, yet
        # moving and reshaping the surface must carry the coil rigidly with it.
        s = get_surface()
        base = get_base()
        curve = CurvePlanarOnSurface(base, s, 3, 2)
        gamma_before = curve.gamma().copy()
        normal_before = curve.unitnormal().copy()
        base_x_before = np.asarray(base.local_full_x).copy()
        local_before = (curve.gamma() - curve.center()) @ curve.rotation_matrix()

        s.set_rc(0, 0, 1.3)          # translate
        s.set_rc(1, 0, 0.35)         # and reshape, which must also retilt it
        s.set_zs(1, 0, 0.10)

        self.assertGreater(np.abs(curve.gamma() - gamma_before).max(), 1e-3)
        self.assertGreater(np.linalg.norm(curve.unitnormal() - normal_before), 1e-3)
        np.testing.assert_allclose(base.local_full_x, base_x_before)
        np.testing.assert_allclose(curve.center(), s.gamma()[3, 2])
        self.assertLess(
            np.abs((curve.gamma() - curve.center()) @ curve.unitnormal()).max(), 1e-13)

        local_after = (curve.gamma() - curve.center()) @ curve.rotation_matrix()
        np.testing.assert_allclose(local_after, local_before, atol=1e-14)

    def test_base_placement_is_fixed(self):
        # The base frame is computed once at construction, so any later change
        # to the base curve's placement would leave it stale and slide the coil
        # off the surface. The class must prevent that.
        s = get_surface()
        base = get_base()
        curve = CurvePlanarOnSurface(base, s, 3, 2)
        gamma_before = curve.gamma().copy()
        for name in ['q0', 'qi', 'qj', 'qk', 'X', 'Y', 'Z']:
            self.assertNotIn(name, base.local_dof_names)
        np.testing.assert_allclose(curve.gamma(), gamma_before)

    def test_taylor(self):
        # Exercises the vjps against both parents at once, through BiotSavart.
        for align in ["theta", "phi"]:
            with self.subTest(align=align):
                s = get_surface()
                curve = CurvePlanarOnSurface(get_base(24), s, 3, 2, align=align)
                current = Current(1e5)
                current.fix_all()
                target = SurfaceRZFourier(
                    mpol=1, ntor=0, nfp=1, stellsym=True,
                    quadpoints_phi=np.linspace(0, 1, 12, endpoint=False),
                    quadpoints_theta=np.linspace(0, 1, 12, endpoint=False))
                target.set_rc(0, 0, 1.0)
                target.set_rc(1, 0, 0.35)
                target.set_zs(1, 0, 0.35)
                target.fix_all()
                J = SquaredFlux(target, BiotSavart([Coil(curve, current)]))

                x0 = J.x.copy()
                np.random.seed(1)
                direction = np.random.rand(x0.size) - 0.5
                dJ = J.dJ() @ direction

                err_old = None
                for k in range(10, 16):
                    eps = 2.0 ** -k
                    J.x = x0 + eps * direction
                    Jplus = J.J()
                    J.x = x0 - eps * direction
                    Jminus = J.J()
                    err = abs((Jplus - Jminus) / (2 * eps) - dJ)
                    if err_old is not None:
                        self.assertLess(err, 0.3 * err_old)
                    err_old = err
                J.x = x0

    def test_serialization(self):
        curve = CurvePlanarOnSurface(get_base(), get_surface(), 3, 2)
        curve_json_str = json.dumps(SIMSON(curve), cls=GSONEncoder, indent=2)
        curve_regen = json.loads(curve_json_str, cls=GSONDecoder)
        self.assertTrue(np.allclose(curve.gamma(), curve_regen.gamma()))


if __name__ == "__main__":
    unittest.main()
