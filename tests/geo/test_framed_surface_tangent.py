"""Tests for the surface-tangent finite-build frame (FramedCurveSurfaceTangent).

The surface-tangent frame lays a finite-build coil pack flat against a circular
toroidal winding surface: the reference normal tracks the surface's outward
normal and the reference binormal lies in the surface tangent plane. These tests
exercise that contract on a CurveCWSFourierCPP banana centerline that lives
exactly on a circular winding torus, so the analytic surface normal is the
ground truth at machine precision.
"""

import unittest

import numpy as np

from simsopt.geo import (
    CurveCWSFourierCPP,
    FrameRotation,
    FramedCurveSurfaceTangent,
    SurfaceRZFourier,
    ZeroRotation,
    create_multifilament_grid,
)
from simsopt.geo.framedcurve import surface_tangent_normal_direction

# Circular winding torus matching the banana hardware contract:
# axis ring radius R0 in the z = Z0 midplane, minor radius A.
WINDING_MAJOR_RADIUS = 0.903
WINDING_MINOR_RADIUS = 0.142
WINDING_MIDPLANE_Z = 0.0


def _winding_surface():
    surf = SurfaceRZFourier(
        nfp=1,
        stellsym=True,
        mpol=1,
        ntor=0,
        quadpoints_phi=np.linspace(0, 1, 32, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, 32, endpoint=False),
    )
    surf.set_rc(0, 0, WINDING_MAJOR_RADIUS)
    surf.set_rc(1, 0, WINDING_MINOR_RADIUS)
    surf.set_zs(1, 0, WINDING_MINOR_RADIUS)
    return surf


def _banana_curve(ppp=80, order=2):
    """A non-trivial banana loop winding poloidally and toroidally on the torus."""
    curve = CurveCWSFourierCPP(
        np.linspace(0, 1, ppp, endpoint=False), order=order, surf=_winding_surface()
    )
    curve.set("phic(0)", 0.0)
    curve.set("phis(1)", 0.10)
    curve.set("phis(2)", 0.03)
    curve.set("thetac(0)", 0.0)
    curve.set("thetas(1)", 1.0)
    curve.set("thetas(2)", 0.15)
    return curve


class SurfaceTangentFrameTesting(unittest.TestCase):
    def test_frame_orthonormal_and_normal_tracks_surface(self):
        """t,n,b are orthonormal and n equals the analytic torus normal (1e-12)."""
        curve = _banana_curve()
        framedcurve = FramedCurveSurfaceTangent(
            curve, WINDING_MAJOR_RADIUS, WINDING_MIDPLANE_Z
        )
        t, n, b = (np.asarray(v) for v in framedcurve.rotated_frame())

        # Orthonormality of the frame.
        self.assertLess(np.abs(np.sum(t * t, axis=1) - 1.0).max(), 1e-12)
        self.assertLess(np.abs(np.sum(n * n, axis=1) - 1.0).max(), 1e-12)
        self.assertLess(np.abs(np.sum(b * b, axis=1) - 1.0).max(), 1e-12)
        self.assertLess(np.abs(np.sum(t * n, axis=1)).max(), 1e-12)
        self.assertLess(np.abs(np.sum(t * b, axis=1)).max(), 1e-12)
        self.assertLess(np.abs(np.sum(n * b, axis=1)).max(), 1e-12)
        self.assertLess(np.abs(np.cross(t, n) - b).max(), 1e-12)

        # On a circular torus the surface normal is purely poloidal, so the
        # curve tangent already lies in the tangent plane and the projected
        # frame normal equals the analytic surface normal exactly.
        gamma = curve.gamma()
        surface_normal = np.asarray(
            surface_tangent_normal_direction(gamma, WINDING_MAJOR_RADIUS, WINDING_MIDPLANE_Z)
        )
        self.assertLess(np.abs(n - surface_normal).max(), 1e-12)
        # The binormal (pack width axis) lies in the tangent plane: it is
        # perpendicular to the surface normal.
        self.assertLess(np.abs(np.sum(b * surface_normal, axis=1)).max(), 1e-12)

    def test_analytic_normal_matches_geometric_torus_normal(self):
        """surface_tangent_normal_direction reproduces the closed-form torus normal."""
        curve = _banana_curve()
        gamma = curve.gamma()
        rho = np.sqrt(gamma[:, 0] ** 2 + gamma[:, 1] ** 2)
        phi = np.arctan2(gamma[:, 1], gamma[:, 0])
        # Poloidal angle of each point about the circular axis ring.
        u = np.arctan2(gamma[:, 2] - WINDING_MIDPLANE_Z, rho - WINDING_MAJOR_RADIUS)
        expected = np.stack(
            [np.cos(u) * np.cos(phi), np.cos(u) * np.sin(phi), np.sin(u)], axis=1
        )
        got = np.asarray(
            surface_tangent_normal_direction(gamma, WINDING_MAJOR_RADIUS, WINDING_MIDPLANE_Z)
        )
        self.assertLess(np.abs(got - expected).max(), 1e-12)

    def test_unrotated_filaments_displaced_along_frame_axes(self):
        """create_multifilament_grid + rotation_order None offsets exactly +/-gapsize."""
        curve = _banana_curve()
        gapsize_n = 0.02
        gapsize_b = 0.029
        framedcurve = FramedCurveSurfaceTangent(
            curve, WINDING_MAJOR_RADIUS, WINDING_MIDPLANE_Z, ZeroRotation(curve.quadpoints)
        )
        _, n, b = (np.asarray(v) for v in framedcurve.rotated_frame())

        filaments = create_multifilament_grid(
            curve,
            2,
            2,
            gapsize_n,
            gapsize_b,
            rotation_order=None,
            frame="surface_tangent",
            surface_major_radius=WINDING_MAJOR_RADIUS,
            surface_midplane_z=WINDING_MIDPLANE_Z,
        )
        self.assertEqual(len(filaments), 4)
        gamma = curve.gamma()
        # 2x2 grid is centered on the curve, so shifts are +/- half a gap.
        for filament, (sign_n, sign_b) in zip(
            filaments, [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        ):
            expected = (
                gamma
                + sign_n * 0.5 * gapsize_n * n
                + sign_b * 0.5 * gapsize_b * b
            )
            self.assertLess(np.abs(np.asarray(filament.gamma()) - expected).max(), 1e-10)

    def test_filament_offsets_lie_in_normal_binormal_plane(self):
        """Each filament's offset from the centerline has zero tangential component."""
        curve = _banana_curve()
        framedcurve = FramedCurveSurfaceTangent(
            curve, WINDING_MAJOR_RADIUS, WINDING_MIDPLANE_Z, ZeroRotation(curve.quadpoints)
        )
        t, _, _ = (np.asarray(v) for v in framedcurve.rotated_frame())
        filaments = create_multifilament_grid(
            curve,
            3,
            3,
            0.02,
            0.029,
            rotation_order=None,
            frame="surface_tangent",
            surface_major_radius=WINDING_MAJOR_RADIUS,
            surface_midplane_z=WINDING_MIDPLANE_Z,
        )
        gamma = curve.gamma()
        for filament in filaments:
            offset = np.asarray(filament.gamma()) - gamma
            self.assertLess(np.abs(np.sum(offset * t, axis=1)).max(), 1e-10)

    def test_create_multifilament_grid_requires_surface_radius(self):
        """surface_tangent frame must reject a missing winding-surface radius."""
        curve = _banana_curve()
        with self.assertRaises(ValueError):
            create_multifilament_grid(
                curve, 2, 2, 0.02, 0.029, rotation_order=None, frame="surface_tangent"
            )

    def test_create_multifilament_grid_rejects_unknown_frame(self):
        """Unknown frames must fail even when Python assertions are optimized out."""
        curve = _banana_curve()
        with self.assertRaisesRegex(ValueError, "unknown_frame"):
            create_multifilament_grid(
                curve, 2, 2, 0.02, 0.029, rotation_order=None, frame="unknown_frame"
            )

    def test_multifilament_coefficient_derivative(self):
        """Finite-build filament gamma/gammadash dof-derivatives match finite differences.

        Mirrors the centroid/frenet derivative test in test_finitebuild so the
        surface-tangent filaments are confirmed differentiable w.r.t. the curve
        and rotation dofs (filament positions enter the BiotSavart graph).
        """
        for order in [None, 1]:
            with self.subTest(order=order):
                self._subtest_coefficient_derivative(order)

    def _subtest_coefficient_derivative(self, order):
        curve = _banana_curve(ppp=20, order=2)
        if order == 1:
            rotation = FrameRotation(curve.quadpoints, order)
            rotation.x = np.array([0.0, 0.1, 0.3])
        else:
            rotation = ZeroRotation(curve.quadpoints)
        filaments = create_multifilament_grid(
            curve,
            2,
            2,
            0.02,
            0.029,
            rotation_order=order,
            frame="surface_tangent",
            surface_major_radius=WINDING_MAJOR_RADIUS,
            surface_midplane_z=WINDING_MIDPLANE_Z,
        )
        filament = filaments[0]

        dofs = filament.x
        g = filament.gamma()
        np.random.seed(1)
        v = np.random.standard_normal(size=g.shape)
        h = np.random.standard_normal(size=dofs.shape)
        df = np.sum(filament.dgamma_by_dcoeff_vjp(v)(filament) * h)
        dg = np.sum(filament.dgammadash_by_dcoeff_vjp(v)(filament) * h)

        errf_old = 1e10
        for i in range(12, 17):
            eps = 0.5**i
            filament.x = dofs + eps * h
            f1 = np.sum(filament.gamma() * v)
            filament.x = dofs - eps * h
            f2 = np.sum(filament.gamma() * v)
            errf = abs((f1 - f2) / (2 * eps) - df)
            self.assertLess(errf, 0.3 * errf_old)
            errf_old = errf

        errg_old = 1e10
        for i in range(10, 17):
            eps = 0.5**i
            filament.x = dofs + eps * h
            g1 = np.sum(filament.gammadash() * v)
            filament.x = dofs - eps * h
            g2 = np.sum(filament.gammadash() * v)
            errg = abs((g1 - g2) / (2 * eps) - dg)
            self.assertLess(errg, 0.3 * errg_old)
            errg_old = errg


if __name__ == "__main__":
    unittest.main()
