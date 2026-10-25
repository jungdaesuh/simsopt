"""Native SurfaceXYZTensorFourier compensated geometry and nonfinite propagation."""
from __future__ import annotations

import math
import unittest

import numpy as np
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier

METHODS = ('gamma', 'gammadash1', 'gammadash2')
# Coefficient indices whose basis terms at node 0 form each method's ordered stream.
STREAM_INDICES = {'gamma': ((0, 0), (0, 1), (0, 2)),
                  'gammadash1': ((0, 5), (0, 6), (0, 8)),
                  'gammadash2': ((5, 0), (6, 0), (8, 0))}
# Exact products [2**54, 1, -2**54] that lose the 1 under naive summation.
CANCELLATION_TERMS = {'gamma': (2.0**54, 1.0, -2.0**54),
                      'gammadash1': (2.0**54, 0.5, -2.0**52),
                      'gammadash2': (2.0**54, 0.5, -2.0**52)}
MAXIMUM = np.finfo(np.float64).max
NONFINITE_TERMS = {'nan': (math.nan, 0.0, 0.0), 'inf': (math.inf, 0.0, 0.0),
                   'negative_inf': (-math.inf, 0.0, 0.0), 'mixed_inf': (math.inf, -math.inf, 0.0),
                   'negative_zero': (-0.0, -0.0, -0.0)}
OVERFLOW_TERMS = {'gamma': (MAXIMUM, MAXIMUM, -MAXIMUM),
                  'gammadash1': (MAXIMUM, MAXIMUM / 2.0, -MAXIMUM / 4.0),
                  'gammadash2': (MAXIMUM, MAXIMUM / 2.0, -MAXIMUM / 4.0)}


def zero_surface(mpol: int = 1, ntor: int = 1, stellsym: bool = False,
                 clamped: bool = False, nodes: tuple[float, ...] = (0.0,),
                 nfp: int = 1) -> SurfaceXYZTensorFourier:
    surface = SurfaceXYZTensorFourier(nfp=nfp, stellsym=stellsym, mpol=mpol,
                                     ntor=ntor, clamped_dims=[clamped] * 3,
                                     quadpoints_phi=list(nodes), quadpoints_theta=list(nodes))
    # Constructor initializes a torus. Zero every stored coefficient, including skipped ones.
    surface.xcs[:] = 0.0
    surface.ycs[:] = 0.0
    surface.zcs[:] = 0.0
    synchronize(surface)
    return surface


def synchronize(surface: SurfaceXYZTensorFourier) -> None:
    surface.set_dofs(surface.get_dofs().copy())
    surface.invalidate_cache()


def geometry(surface: SurfaceXYZTensorFourier) -> tuple[np.ndarray, ...]:
    return surface.gamma().copy(), surface.gammadash1().copy(), surface.gammadash2().copy()


def angles(nodes: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(2.0 * np.pi * np.asarray(nodes), 2.0 * np.pi * np.asarray(nodes), indexing='ij')


def stream(method: str, coefficients: tuple[float, ...], dim: int = 2, nfp: int = 1) -> SurfaceXYZTensorFourier:
    """Zero surface whose ``dim`` component carries ``coefficients`` on ``method``'s stream."""
    surface = zero_surface(mpol=4 if method == 'gammadash2' else 1,
                           ntor=4 if method == 'gammadash1' else 2, nfp=nfp)
    for index, coefficient in zip(STREAM_INDICES[method], coefficients):
        (surface.xcs, surface.ycs, surface.zcs)[dim][index] = coefficient
    synchronize(surface)
    return surface


class SurfaceCompensatedNativeTests(unittest.TestCase):
    def test_single_term_controls(self) -> None:
        for method in METHODS:
            with self.subTest(method=method):
                surface = stream(method, (0.0, 1.0 if method == 'gamma' else 0.5, 0.0))
                values = geometry(surface)
                self.assertEqual(float(values[METHODS.index(method)][0, 0, 2]), 1.0 if method == 'gamma' else 2.0 * math.pi)

    def test_torus_symmetry_nfp_and_cache(self) -> None:
        nodes = (0.0, 0.125, 0.25, 0.5)
        phi, theta = angles(nodes)
        for stellsym in (False, True):
            for nfp in (1, 2, 3):
                with self.subTest(stellsym=stellsym, nfp=nfp):
                    surface = zero_surface(stellsym=stellsym, nodes=nodes, nfp=nfp)
                    surface.xcs[0, 0] = 1.0
                    surface.xcs[1, 0] = 0.1
                    for radius in (0.1, 0.2, 0.1):
                        surface.zcs[2, 0] = radius
                        synchronize(surface)
                        values = geometry(surface)
                        r = 1.0 + 0.1 * np.cos(theta)
                        expected = (np.stack((r * np.cos(phi), r * np.sin(phi), radius * np.sin(theta)), axis=-1),
                                    2.0 * np.pi * np.stack((-r * np.sin(phi), r * np.cos(phi), np.zeros_like(phi)), axis=-1),
                                    2.0 * np.pi * np.stack((-0.1 * np.sin(theta) * np.cos(phi), -0.1 * np.sin(theta) * np.sin(phi), radius * np.cos(theta)), axis=-1))
                        for name, actual, analytic in zip(METHODS, values, expected):
                            np.testing.assert_allclose(actual, analytic, rtol=0.0, atol=1e-13, err_msg=name)
                        # Exercise public invalidation after an actual DOF change and repeated evaluation.
                        surface.invalidate_cache()
                        repeated = geometry(surface)
                        for first, second in zip(values, repeated):
                            self.assertEqual(first.dtype, second.dtype)
                            self.assertEqual(first.tobytes(), second.tobytes())

    def test_all_gamma_and_theta_accumulator_streams(self) -> None:
        for method in ('gamma', 'gammadash2'):
            for dim in range(3):
                with self.subTest(method=method, dim=dim):
                    surface = stream(method, CANCELLATION_TERMS[method], dim)
                    actual = geometry(surface)[METHODS.index(method)][0, 0]
                    expected = np.zeros(3)
                    expected[dim] = 1.0 if method == 'gamma' else 2.0 * math.pi
                    np.testing.assert_array_equal(actual, expected)

    def test_all_phi_derivative_streams_and_nfp_scaling(self) -> None:
        for nfp in (1, 2, 3):
            for dim in range(3):
                with self.subTest(nfp=nfp, dim=dim):
                    surface = stream('gammadash1', CANCELLATION_TERMS['gammadash1'], dim, nfp)
                    expected = np.zeros(3)
                    expected[dim] = 2.0 * math.pi * nfp
                    np.testing.assert_array_equal(surface.gammadash1()[0, 0], expected)

    def test_phi_xhat_yhat_rotation_coupling(self) -> None:
        for nfp in (1, 2, 3):
            for dim in (0, 1):
                with self.subTest(nfp=nfp, dim=dim):
                    # The position stream's xhat/yhat enter the phi tangent through the rotation.
                    surface = stream('gamma', CANCELLATION_TERMS['gamma'], dim, nfp)
                    expected = np.zeros(3)
                    expected[1 - dim] = 2.0 * math.pi * (1.0 if dim == 0 else -1.0)
                    np.testing.assert_array_equal(surface.gammadash1()[0, 0], expected)

    def test_nonzero_modes_and_rotated_first_derivatives(self) -> None:
        nodes = (0.0, 0.03125, 0.125)
        phi, theta = angles(nodes)
        for nfp in (1, 2, 3):
            with self.subTest(nfp=nfp):
                surface = zero_surface(mpol=2, ntor=2, nfp=nfp, nodes=nodes)
                a, b, c = 0.125, -0.25, 0.5
                surface.xcs[1, 3] = a
                surface.ycs[3, 1] = b
                surface.zcs[3, 3] = c
                synchronize(surface)
                x = a * np.cos(theta) * np.sin(nfp * phi)
                y = b * np.sin(theta) * np.cos(nfp * phi)
                z = c * np.sin(theta) * np.sin(nfp * phi)
                xp = a * np.cos(theta) * nfp * np.cos(nfp * phi)
                yp = -b * np.sin(theta) * nfp * np.sin(nfp * phi)
                zp = c * np.sin(theta) * nfp * np.cos(nfp * phi)
                xt = -a * np.sin(theta) * np.sin(nfp * phi)
                yt = b * np.cos(theta) * np.cos(nfp * phi)
                zt = c * np.cos(theta) * np.sin(nfp * phi)
                expected = (np.stack((x * np.cos(phi) - y * np.sin(phi), x * np.sin(phi) + y * np.cos(phi), z), axis=-1),
                            2.0 * np.pi * np.stack((xp * np.cos(phi) - yp * np.sin(phi) - x * np.sin(phi) - y * np.cos(phi),
                                                   xp * np.sin(phi) + yp * np.cos(phi) + x * np.cos(phi) - y * np.sin(phi), zp), axis=-1),
                            2.0 * np.pi * np.stack((xt * np.cos(phi) - yt * np.sin(phi), xt * np.sin(phi) + yt * np.cos(phi), zt), axis=-1))
                for name, actual, analytic in zip(METHODS, geometry(surface), expected):
                    np.testing.assert_allclose(actual, analytic, rtol=0.0, atol=1e-13, err_msg=name)

    def test_clamped_all_components_first_derivatives(self) -> None:
        nodes = (0.0, 0.125, 0.25)
        phi, theta = angles(nodes)
        # The z-only case also requires exactly zero x/y components.
        cases = [(nfp, (0.125, -0.25, 0.5)) for nfp in (1, 2, 3)] + [(1, (0.0, 0.0, 2.0))]
        for nfp, (a, b, c) in cases:
            with self.subTest(nfp=nfp, coefficients=(a, b, c)):
                surface = zero_surface(clamped=True, nodes=nodes, nfp=nfp)
                surface.xcs[0, 0], surface.ycs[0, 0], surface.zcs[0, 0] = a, b, c
                synchronize(surface)
                enforcer = np.sin(nfp * phi / 2.0)**2 + np.sin(theta / 2.0)**2
                ep = nfp * np.cos(nfp * phi / 2.0) * np.sin(nfp * phi / 2.0)
                et = np.cos(theta / 2.0) * np.sin(theta / 2.0)
                xr = a * np.cos(phi) - b * np.sin(phi)
                yr = a * np.sin(phi) + b * np.cos(phi)
                expected = (np.stack((enforcer * xr, enforcer * yr, c * enforcer), axis=-1),
                            2.0 * np.pi * np.stack((ep * xr - enforcer * yr, ep * yr + enforcer * xr, c * ep), axis=-1),
                            2.0 * np.pi * np.stack((et * xr, et * yr, c * et), axis=-1))
                for name, actual, analytic in zip(METHODS, geometry(surface), expected):
                    np.testing.assert_allclose(actual, analytic, rtol=0.0, atol=1e-13, err_msg=name)
                    if a == b == 0.0:
                        np.testing.assert_array_equal(actual[..., :2], 0.0, err_msg=name)

    def test_all_component_nonfinite_and_transient_overflow(self) -> None:
        for method in METHODS:
            cases = {**NONFINITE_TERMS, 'transient_overflow': OVERFLOW_TERMS[method]}
            for dim in range(3):
                for label, terms in cases.items():
                    with self.subTest(method=method, dim=dim, case=label):
                        actual = geometry(stream(method, terms, dim))[METHODS.index(method)][0, 0]
                        if label == 'negative_zero':
                            np.testing.assert_array_equal(actual, 0.0)
                            np.testing.assert_array_equal(np.signbit(actual), False)
                        elif dim == 2:
                            np.testing.assert_array_equal(actual[:2], 0.0)
                            if label in ('nan', 'mixed_inf'):
                                self.assertTrue(math.isnan(float(actual[2])))
                            else:
                                self.assertEqual(float(actual[2]), -math.inf if label == 'negative_inf' else math.inf)
                        else:
                            self.assertEqual(float(actual[2]), 0.0)
                            if label in ('nan', 'mixed_inf') or (method == 'gammadash1' and label in ('inf', 'negative_inf')):
                                np.testing.assert_array_equal(np.isnan(actual[:2]), True)
                            else:
                                self.assertEqual(float(actual[dim]), -math.inf if label == 'negative_inf' else math.inf)
                                self.assertTrue(math.isnan(float(actual[1 - dim])))

    def test_all_components_skip_nonfinite_coefficients(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                surface = zero_surface(stellsym=True)
                # These modes are forbidden by symmetry: get_coeff returns zero before reading them.
                surface.xcs[0, 2] = value
                surface.ycs[0, 0] = value
                surface.zcs[0, 0] = value
                surface.invalidate_cache()
                for actual in geometry(surface):
                    np.testing.assert_array_equal(actual, 0.0)


if __name__ == '__main__':
    unittest.main()
