"""Native tensor geometry accuracy and legacy nonfinite behavior.

The three z cancellation cases fail against retained native b190 before this
candidate. Expanded x/y/coupling cases must also be compared on old/candidate
binaries. Singular fixtures exercise only public geometry and first derivatives.
"""
from __future__ import annotations

import math
import unittest

import numpy as np
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier

METHODS = ('gamma', 'gammadash1', 'gammadash2')


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


def z_stream(method: str, coefficients: tuple[float, ...]) -> SurfaceXYZTensorFourier:
    surface = zero_surface(mpol=4 if method == 'gammadash2' else 1,
                           ntor=4 if method == 'gammadash1' else 2)
    indices = {'gamma': ((0, 0), (0, 1), (0, 2)),
               'gammadash1': ((0, 5), (0, 6), (0, 8)),
               'gammadash2': ((5, 0), (6, 0), (8, 0))}[method]
    for index, coefficient in zip(indices, coefficients):
        surface.zcs[index] = coefficient
    synchronize(surface)
    return surface


class SurfaceCompensatedNativeTests(unittest.TestCase):
    def test_coefficient_order_and_zero_fixture(self) -> None:
        surface = zero_surface(mpol=4, ntor=4)
        for coefficients in (surface.xcs, surface.ycs, surface.zcs):
            np.testing.assert_array_equal(coefficients, 0.0)
        ordered = np.arange(3 * 9 * 9, dtype=np.float64)
        surface.set_dofs(ordered)
        np.testing.assert_array_equal(surface.get_dofs(), ordered)
        for index, coefficients in enumerate((surface.xcs, surface.ycs, surface.zcs)):
            np.testing.assert_array_equal(coefficients, ordered.reshape(3, 9, 9)[index])

    def cancellation(self, method: str) -> None:
        coefficients = (2.0**54, 1.0, -2.0**54) if method == 'gamma' else (2.0**54, 0.5, -2.0**52)
        surface = z_stream(method, coefficients)
        values = geometry(surface)
        actual = float(values[METHODS.index(method)][0, 0, 2])
        expected = 1.0 if method == 'gamma' else 2.0 * math.pi
        self.assertEqual(actual, expected, f'{method}: ordered exact products [2**54,1,-2**54] must retain 1 before the 2pi derivative scale')

    def test_cancellation_gamma(self) -> None:
        self.cancellation('gamma')

    def test_cancellation_phi(self) -> None:
        self.cancellation('gammadash1')

    def test_cancellation_theta(self) -> None:
        self.cancellation('gammadash2')

    def test_single_term_controls(self) -> None:
        for method in METHODS:
            with self.subTest(method=method):
                surface = z_stream(method, (0.0, 1.0 if method == 'gamma' else 0.5, 0.0))
                values = geometry(surface)
                self.assertEqual(float(values[METHODS.index(method)][0, 0, 2]), 1.0 if method == 'gamma' else 2.0 * math.pi)

    def test_torus_symmetry_nfp_and_cache(self) -> None:
        nodes = (0.0, 0.125, 0.25, 0.5)
        phi, theta = np.meshgrid(2.0 * np.pi * np.asarray(nodes), 2.0 * np.pi * np.asarray(nodes), indexing='ij')
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

    def test_clamped_first_derivative_control(self) -> None:
        surface = zero_surface(clamped=True, nodes=(0.0, 0.25))
        surface.zcs[0, 0] = 2.0
        synchronize(surface)
        values = geometry(surface)
        phi, theta = np.meshgrid(2.0 * np.pi * np.asarray((0.0, 0.25)), 2.0 * np.pi * np.asarray((0.0, 0.25)), indexing='ij')
        expected = (2.0 * (np.sin(phi / 2.0)**2 + np.sin(theta / 2.0)**2),
                    4.0 * np.pi * np.cos(phi / 2.0) * np.sin(phi / 2.0),
                    4.0 * np.pi * np.cos(theta / 2.0) * np.sin(theta / 2.0))
        for actual, analytic in zip(values, expected):
            np.testing.assert_allclose(actual[..., 2], analytic, rtol=0.0, atol=1e-13)
            np.testing.assert_array_equal(actual[..., :2], 0.0)

    def test_nonfinite_and_transient_overflow_legacy(self) -> None:
        maximum = np.finfo(np.float64).max
        for method in METHODS:
            scales = (1.0, 1.0, 1.0) if method == 'gamma' else (1.0, 0.5, 0.25)
            cases = (('nan', (math.nan, 0.0, 0.0), 'nan'),
                     ('inf', (math.inf, 0.0, 0.0), 'positive_inf'),
                     ('negative_inf', (-math.inf, 0.0, 0.0), 'negative_inf'),
                     ('mixed_inf', (math.inf, -math.inf, 0.0), 'nan'),
                     ('transient_overflow', (maximum * scales[0], maximum * scales[1], -maximum * scales[2]), 'positive_inf'),
                     ('negative_zero', (-0.0, -0.0, -0.0), 'positive_zero'))
            for label, coefficients, expected in cases:
                with self.subTest(method=method, case=label):
                    values = geometry(z_stream(method, coefficients))
                    actual = float(values[METHODS.index(method)][0, 0, 2])
                    if expected == 'nan':
                        self.assertTrue(math.isnan(actual))
                    elif expected == 'positive_inf':
                        self.assertEqual(actual, math.inf)
                    elif expected == 'negative_inf':
                        self.assertEqual(actual, -math.inf)
                    else:
                        self.assertEqual(actual, 0.0)
                        self.assertFalse(np.signbit(actual))

    def test_skipped_nonfinite_coefficients(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                surface = zero_surface(stellsym=True)
                # z(0,0) is forbidden by symmetry: get_coeff returns zero before reading it.
                surface.zcs[0, 0] = value
                surface.invalidate_cache()
                for actual in geometry(surface):
                    np.testing.assert_array_equal(actual, 0.0)

    def test_all_gamma_and_theta_accumulator_streams(self) -> None:
        for method in ('gamma', 'gammadash2'):
            for dim in range(3):
                with self.subTest(method=method, dim=dim):
                    surface = z_stream(method, (0.0, 0.0, 0.0))
                    coefficients = (surface.xcs, surface.ycs, surface.zcs)[dim]
                    indices = ((0, 0), (0, 1), (0, 2)) if method == 'gamma' else ((5, 0), (6, 0), (8, 0))
                    terms = (2.0**54, 1.0, -2.0**54) if method == 'gamma' else (2.0**54, 0.5, -2.0**52)
                    for index, value in zip(indices, terms):
                        coefficients[index] = value
                    synchronize(surface)
                    actual = geometry(surface)[METHODS.index(method)][0, 0]
                    expected = np.zeros(3)
                    expected[dim] = 1.0 if method == 'gamma' else 2.0 * math.pi
                    np.testing.assert_array_equal(actual, expected)

    def test_all_phi_derivative_streams_and_nfp_scaling(self) -> None:
        for nfp in (1, 2, 3):
            for dim in range(3):
                with self.subTest(nfp=nfp, dim=dim):
                    surface = zero_surface(ntor=4, nfp=nfp)
                    coefficients = (surface.xcs, surface.ycs, surface.zcs)[dim]
                    for index, value in zip((5, 6, 8), (2.0**54, 0.5, -2.0**52)):
                        coefficients[0, index] = value
                    synchronize(surface)
                    expected = np.zeros(3)
                    expected[dim] = 2.0 * math.pi * nfp
                    np.testing.assert_array_equal(surface.gammadash1()[0, 0], expected)

    def test_phi_xhat_yhat_rotation_coupling(self) -> None:
        for nfp in (1, 2, 3):
            for dim in (0, 1):
                with self.subTest(nfp=nfp, dim=dim):
                    surface = zero_surface(ntor=2, nfp=nfp)
                    coefficients = (surface.xcs, surface.ycs)[dim]
                    for index, value in enumerate((2.0**54, 1.0, -2.0**54)):
                        coefficients[0, index] = value
                    synchronize(surface)
                    expected = np.zeros(3)
                    expected[1 - dim] = 2.0 * math.pi * (1.0 if dim == 0 else -1.0)
                    np.testing.assert_array_equal(surface.gammadash1()[0, 0], expected)

    def test_nonzero_modes_and_rotated_first_derivatives(self) -> None:
        nodes = (0.0, 0.03125, 0.125)
        phi, theta = np.meshgrid(2.0 * np.pi * np.asarray(nodes), 2.0 * np.pi * np.asarray(nodes), indexing='ij')
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
        phi, theta = np.meshgrid(2.0 * np.pi * np.asarray(nodes), 2.0 * np.pi * np.asarray(nodes), indexing='ij')
        for nfp in (1, 2, 3):
            with self.subTest(nfp=nfp):
                surface = zero_surface(clamped=True, nodes=nodes, nfp=nfp)
                a, b, c = 0.125, -0.25, 0.5
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

    def test_all_component_nonfinite_and_transient_overflow(self) -> None:
        maximum = np.finfo(np.float64).max
        for method in METHODS:
            for dim in range(3):
                for label in ('nan', 'inf', 'negative_inf', 'mixed_inf', 'transient_overflow', 'negative_zero'):
                    with self.subTest(method=method, dim=dim, case=label):
                        surface = z_stream(method, (0.0, 0.0, 0.0))
                        indices = {'gamma': ((0, 0), (0, 1), (0, 2)),
                                   'gammadash1': ((0, 5), (0, 6), (0, 8)),
                                   'gammadash2': ((5, 0), (6, 0), (8, 0))}[method]
                        overflow = (maximum, maximum, -maximum) if method == 'gamma' else (maximum, maximum / 2.0, -maximum / 4.0)
                        cases = {'nan': (math.nan, 0.0, 0.0), 'inf': (math.inf, 0.0, 0.0),
                                 'negative_inf': (-math.inf, 0.0, 0.0), 'mixed_inf': (math.inf, -math.inf, 0.0),
                                 'transient_overflow': overflow, 'negative_zero': (-0.0, -0.0, -0.0)}
                        coefficients = (surface.xcs, surface.ycs, surface.zcs)[dim]
                        for index, value in zip(indices, cases[label]):
                            coefficients[index] = value
                        synchronize(surface)
                        actual = geometry(surface)[METHODS.index(method)][0, 0]
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
                surface.xcs[0, 2] = value
                surface.ycs[0, 0] = value
                surface.zcs[0, 0] = value
                surface.invalidate_cache()
                for actual in geometry(surface):
                    np.testing.assert_array_equal(actual, 0.0)


if __name__ == '__main__':
    unittest.main()
