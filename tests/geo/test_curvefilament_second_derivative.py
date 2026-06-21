"""Second-derivative (curvature) support for finite-build filaments.

Until this was added, ``CurveFilament`` implemented only ``gamma`` and
``gammadash``; calling ``gammadashdash`` (and therefore ``kappa``) on a
finite-build conductor raised ``gammadashdash_impl was not implemented``. These
tests pin the new behavior:

* ``CurveFilament.gammadashdash`` is the true second curve-parameter derivative
  of the offset conductor, so ``kappa`` works on the conductor itself rather than
  only on the centerline.
* The chain it depends on -- ``rotated_frame_dashdash`` on the centroid and
  surface-tangent frames, ``FrameRotation.alphadashdash``, and
  ``CurveCWSFourierCPP.gammadashdashdash`` -- is exact.
* The Frenet frame, which would need the curve's (unavailable) fourth
  derivative, fails loudly instead of returning a wrong number.

Ground truth is a periodic spectral derivative on a dense uniform grid: a smooth
closed curve sampled on ``N`` equispaced points has an FFT derivative accurate to
machine precision, so a wrong analytic derivative shows up as an O(1) relative
error rather than a near-zero one.
"""

import unittest

import numpy as np

from simsopt.geo import (
    CurveCWSFourierCPP,
    CurveXYZFourier,
    FrameRotation,
    FramedCurveCentroid,
    FramedCurveFrenet,
    FramedCurveSurfaceTangent,
    SurfaceRZFourier,
    ZeroRotation,
)
from simsopt.geo.finitebuild import CurveFilament

WINDING_MAJOR_RADIUS = 0.903
WINDING_MINOR_RADIUS = 0.142
WINDING_MIDPLANE_Z = 0.0
# Dense enough that the FFT reference resolves the centroid frame's projection on
# a CWS curve to ~1e-11 (it is only ~1e-5 at 512), so the spectral truncation
# error stays well below the assertion tolerances and a wrong analytic derivative
# is what fails, not the reference.
N_DENSE = 1024


def _dense_quadpoints(n=N_DENSE):
    return np.linspace(0.0, 1.0, n, endpoint=False)


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


def _banana_curve(n=N_DENSE, order=2):
    """A smooth banana loop winding poloidally and toroidally on the torus."""
    curve = CurveCWSFourierCPP(_dense_quadpoints(n), order=order, surf=_winding_surface())
    curve.set("phis(1)", 0.10)
    curve.set("phis(2)", 0.03)
    curve.set("thetas(1)", 1.0)
    curve.set("thetas(2)", 0.15)
    return curve


def _modest_rotation(quadpoints, order=3, seed=0):
    """A FrameRotation whose angle stays O(1) so its second derivative is well
    resolved on the dense grid (scale=1, small coefficients)."""
    rot = FrameRotation(quadpoints, order, scale=1.0)
    rng = np.random.default_rng(seed)
    rot.x = 0.2 * rng.standard_normal(rot.dof_size)
    return rot


def _spectral_derivative(values):
    """Derivative with respect to the [0,1)-periodic parameter via FFT.

    ``values`` has shape (N,) or (N, k); returns the same shape.
    """
    arr = np.asarray(values)
    flat = arr.reshape(arr.shape[0], -1)
    n = flat.shape[0]
    wavenumbers = np.fft.fftfreq(n, d=1.0 / n)
    deriv = np.fft.ifft((2j * np.pi * wavenumbers)[:, None] * np.fft.fft(flat, axis=0), axis=0)
    return np.real(deriv).reshape(arr.shape)


def _rel_error(actual, reference):
    return np.max(np.abs(actual - reference)) / (np.max(np.abs(reference)) + 1e-30)


class CurveFilamentSecondDerivativeTesting(unittest.TestCase):
    def test_cws_third_derivative_matches_spectral(self):
        """CurveCWSFourierCPP.gammadashdashdash equals the spectral derivative of
        gammadashdash (the dependency the filament curvature needs)."""
        curve = _banana_curve()
        third = np.asarray(curve.gammadashdashdash())
        reference = _spectral_derivative(curve.gammadashdash())
        self.assertLess(_rel_error(third, reference), 1e-7)

    def test_filament_gammadashdash_matches_spectral_surface_tangent(self):
        """Offset, twisted surface-tangent filament: analytic gammadashdash equals
        the spectral derivative of its own gammadash."""
        curve = _banana_curve()
        framedcurve = FramedCurveSurfaceTangent(
            curve, WINDING_MAJOR_RADIUS, WINDING_MIDPLANE_Z,
            _modest_rotation(curve.quadpoints),
        )
        filament = CurveFilament(framedcurve, 0.013, -0.021)
        reference = _spectral_derivative(filament.gammadash())
        self.assertLess(_rel_error(np.asarray(filament.gammadashdash()), reference), 1e-6)

    def test_filament_gammadashdash_matches_spectral_centroid(self):
        """Same exactness for the centroid frame (simsopt's default frame)."""
        curve = _banana_curve()
        framedcurve = FramedCurveCentroid(curve, _modest_rotation(curve.quadpoints, seed=1))
        filament = CurveFilament(framedcurve, -0.018, 0.012)
        reference = _spectral_derivative(filament.gammadash())
        self.assertLess(_rel_error(np.asarray(filament.gammadashdash()), reference), 1e-6)

    def test_filament_kappa_matches_spectral_curvature(self):
        """kappa() now runs on a conductor and equals the curvature built from
        spectral derivatives of the conductor position."""
        curve = _banana_curve()
        framedcurve = FramedCurveSurfaceTangent(
            curve, WINDING_MAJOR_RADIUS, WINDING_MIDPLANE_Z,
            _modest_rotation(curve.quadpoints, seed=2),
        )
        filament = CurveFilament(framedcurve, 0.011, -0.017)
        gamma = np.asarray(filament.gamma())
        gd = _spectral_derivative(gamma)
        gdd = _spectral_derivative(gd)
        speed = np.linalg.norm(gd, axis=1)
        kappa_reference = np.linalg.norm(np.cross(gd, gdd), axis=1) / speed**3
        self.assertLess(_rel_error(np.asarray(filament.kappa()), kappa_reference), 1e-5)

    def test_zero_offset_filament_equals_centerline_second_derivative(self):
        """With dn=db=0 the conductor is the centerline, so its second derivative
        is exactly the centerline's (no frame term contributes)."""
        curve = _banana_curve()
        framedcurve = FramedCurveSurfaceTangent(
            curve, WINDING_MAJOR_RADIUS, WINDING_MIDPLANE_Z,
            _modest_rotation(curve.quadpoints, seed=3),
        )
        filament = CurveFilament(framedcurve, 0.0, 0.0)
        self.assertLess(
            np.max(np.abs(np.asarray(filament.gammadashdash()) - np.asarray(curve.gammadashdash()))),
            1e-12,
        )

    def test_frame_rotation_alphadashdash_matches_closed_form_and_spectral(self):
        """FrameRotation.alphadashdash equals both the analytic Fourier second
        derivative and the spectral derivative of alphadash."""
        quadpoints = _dense_quadpoints()
        rot = _modest_rotation(quadpoints, order=4, seed=4)
        analytic = np.asarray(rot.alphadashdash(quadpoints))

        # Closed form: each mode j contributes -(2*pi*j)**2 * coeff, scaled.
        dofs = np.asarray(rot.x)
        closed = np.zeros_like(quadpoints)
        for j in range(1, rot.order + 1):
            w = 2 * np.pi * j
            closed -= dofs[2 * j - 1] * w**2 * np.sin(w * quadpoints)
            closed -= dofs[2 * j] * w**2 * np.cos(w * quadpoints)
        closed *= rot.scale
        self.assertLess(np.max(np.abs(analytic - closed)), 1e-10)
        self.assertLess(_rel_error(analytic, _spectral_derivative(rot.alphadash(quadpoints))), 1e-7)

    def test_zero_rotation_alphadashdash_is_zero(self):
        quadpoints = _dense_quadpoints()
        rot = ZeroRotation(quadpoints)
        self.assertEqual(np.max(np.abs(np.asarray(rot.alphadashdash(quadpoints)))), 0.0)

    def test_frenet_frame_second_derivative_raises(self):
        """The Frenet frame cannot supply a second frame derivative (needs the
        curve's fourth derivative); it must fail loudly, not silently."""
        curve = _banana_curve(order=2)
        framedcurve = FramedCurveFrenet(curve, _modest_rotation(curve.quadpoints, seed=5))
        with self.assertRaises(NotImplementedError):
            framedcurve.rotated_frame_dashdash()
        filament = CurveFilament(framedcurve, 0.01, 0.01)
        with self.assertRaises(NotImplementedError):
            filament.gammadashdash()

    def test_curvexyzfourier_filament_second_derivative(self):
        """The fix is not CWS-specific: an XYZ-backed conductor's second
        derivative is exact too (independent curve family)."""
        curve = CurveXYZFourier(_dense_quadpoints(), order=4)
        curve.set("xc(0)", 1.0)
        curve.set("xc(1)", 0.30)
        curve.set("ys(1)", 0.30)
        curve.set("zs(1)", 0.12)
        curve.set("xc(2)", 0.04)
        framedcurve = FramedCurveCentroid(curve, _modest_rotation(curve.quadpoints, seed=6))
        filament = CurveFilament(framedcurve, 0.02, -0.015)
        reference = _spectral_derivative(filament.gammadash())
        self.assertLess(_rel_error(np.asarray(filament.gammadashdash()), reference), 1e-6)


if __name__ == "__main__":
    unittest.main()
