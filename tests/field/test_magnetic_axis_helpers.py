#!/usr/bin/env python3
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import coils_via_symmetries
from simsopt.field.magnetic_axis_helpers import (
    _AXIS_RETURN_PENALTY,
    FieldLineBelowAxisThresholdError,
    MagneticAxisNotLocatedError,
    _axis_return_residual,
    _fieldline_rhs_phi,
    compute_on_axis_iota,
    locate_magnetic_axis_point,
)
from simsopt.configs.zoo import get_ncsx_data, get_hsx_data, get_giuliani_data


class _ShiftedAxisField:
    """Analytic field whose magnetic axis closes over the full torus only.

    Field lines rotate at ``iota`` about a center
    ``C(phi) = (major_radius + axis_shift*cos phi, axis_shift*sin phi)`` that is
    2*pi-periodic but not 2*pi/nfp-periodic. The axis (the closed field line at
    ``C(0) = (major_radius + axis_shift, 0)``) is therefore a fixed point of the
    full-torus return map but, for nfp > 1, NOT of the single-field-period map.
    ``axis_shift = 0`` reduces to an axisymmetric field with axis at
    ``major_radius``.
    """

    def __init__(self, *, major_radius, axis_shift, iota):
        self.major_radius = float(major_radius)
        self.axis_shift = float(axis_shift)
        self.iota = float(iota)
        self.points = np.empty((0, 3), dtype=float)

    def set_points(self, points):
        self.points = np.asarray(points, dtype=float)
        return self

    def B(self):
        x = self.points[:, 0]
        y = self.points[:, 1]
        z = self.points[:, 2]
        radius = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        cos_phi = x / radius
        sin_phi = y / radius
        center_r = self.major_radius + self.axis_shift * np.cos(phi)
        center_z = self.axis_shift * np.sin(phi)
        d_center_r = -self.axis_shift * np.sin(phi)
        d_center_z = self.axis_shift * np.cos(phi)
        d_radius_dphi = d_center_r - self.iota * (z - center_z)
        d_z_dphi = d_center_z + self.iota * (radius - center_r)
        b_phi = np.ones_like(radius)
        b_r = d_radius_dphi / radius
        b_z = d_z_dphi / radius
        b_x = b_r * cos_phi - b_phi * sin_phi
        b_y = b_r * sin_phi + b_phi * cos_phi
        return np.stack([b_x, b_y, b_z], axis=-1)


class _BphiNullAxisField:
    """Axisymmetric rotating field whose toroidal component vanishes at R=r_null.

    Field lines circle the axis at ``(major_radius, 0)`` (where b_r=b_z=0 and the
    axis is healthy: |B_phi|/|B| ~ 1). ``b_phi = (radius - r_null)/major_radius``
    passes through zero on the cylinder ``radius = r_null``. A field line whose
    radial oscillation crosses ``r_null`` therefore drives |B_phi|/|B| below the
    axis threshold mid-integration -- the exact transient-null condition the m8n8
    axis solve hit on an off-axis trial step. Reproduces RC1 deterministically.
    """

    def __init__(self, *, major_radius, r_null, iota):
        self.major_radius = float(major_radius)
        self.r_null = float(r_null)
        self.iota = float(iota)
        self.points = np.empty((0, 3), dtype=float)

    def set_points(self, points):
        self.points = np.asarray(points, dtype=float)
        return self

    def B(self):
        x = self.points[:, 0]
        y = self.points[:, 1]
        z = self.points[:, 2]
        radius = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        cos_phi = x / radius
        sin_phi = y / radius
        d_radius_dphi = -self.iota * z
        d_z_dphi = self.iota * (radius - self.major_radius)
        b_phi = (radius - self.r_null) / self.major_radius
        b_r = d_radius_dphi / radius
        b_z = d_z_dphi / radius
        b_x = b_r * cos_phi - b_phi * sin_phi
        b_y = b_r * sin_phi + b_phi * cos_phi
        return np.stack([b_x, b_y, b_z], axis=-1)


class AxisReturnResidualBphiNullRobustness(unittest.TestCase):
    def test_bphi_null_during_integration_returns_penalty_not_raises(self):
        # A trial state whose field line oscillates across the b_phi=0 cylinder
        # trips the integrand guard mid-integration. The residual must absorb it
        # into a large finite penalty so least_squares can step away -- the old
        # behavior let the ValueError escape and abort the whole axis solve.
        field = _BphiNullAxisField(major_radius=1.0, r_null=0.9, iota=0.3)
        # Confirm the integrand itself still trips on the null cylinder (guard intact).
        with self.assertRaises(FieldLineBelowAxisThresholdError):
            _fieldline_rhs_phi(
                0.0,
                np.asarray([0.9, 0.0], dtype=float),
                field,
                1.0e-8,
            )
        residual = _axis_return_residual(
            np.asarray([0.78, 0.0], dtype=float),
            magnetic_field=field,
            phi0=0.0,
            toroidal_span=2.0 * math.pi,
            scale=1.0,
            rtol=1.0e-9,
            atol=1.0e-11,
            max_step=0.05,
            min_bphi_over_b=1.0e-8,
        )
        np.testing.assert_array_equal(
            residual, np.full(2, _AXIS_RETURN_PENALTY)
        )

    def test_bphi_null_trial_does_not_abort_whole_axis_solve(self):
        # End-to-end: with a healthy axis at R=1.0 but a b_phi null at R=0.9, the
        # least_squares solve must NOT die with a bare ValueError when a trial
        # excursion hits the null. It either converges to the true axis or ends
        # as MagneticAxisNotLocatedError (graceful) -- never an escaping
        # FieldLineBelowAxisThresholdError/ValueError.
        field = _BphiNullAxisField(major_radius=1.0, r_null=0.9, iota=0.3)
        try:
            point = locate_magnetic_axis_point(
                field,
                (0.78, 0.05),
                nfp=1,
                r_bounds=(0.6, 1.4),
                z_bounds=(-0.3, 0.3),
            )
        except MagneticAxisNotLocatedError:
            point = None
        if point is not None:
            self.assertAlmostEqual(point["r"], 1.0, places=3)
            self.assertAlmostEqual(point["z"], 0.0, places=3)
            self.assertLess(point["normalized_return_residual"], 1.0e-6)


class LocateMagneticAxisPoint(unittest.TestCase):
    def test_axisymmetric_axis_recovered(self):
        field = _ShiftedAxisField(major_radius=1.0, axis_shift=0.0, iota=0.3)
        point = locate_magnetic_axis_point(
            field,
            (1.05, 0.02),
            nfp=3,
            r_bounds=(0.8, 1.2),
            z_bounds=(-0.2, 0.2),
        )
        self.assertAlmostEqual(point["r"], 1.0, places=5)
        self.assertAlmostEqual(point["z"], 0.0, places=5)

    def test_nfp_asymmetric_axis_needs_full_torus_turn(self):
        # Regression: the axis closes over 2*pi at (1.05, 0) but not over one
        # field period. The full-torus locator must still find it -- the old
        # 2*pi/nfp span could not, which broke the WBA/topology scorer on
        # symmetry-broken (optimized coil) fields.
        field = _ShiftedAxisField(major_radius=1.0, axis_shift=0.05, iota=0.3)
        point = locate_magnetic_axis_point(
            field,
            (1.0, 0.0),
            nfp=3,
            r_bounds=(0.8, 1.2),
            z_bounds=(-0.2, 0.2),
        )
        self.assertAlmostEqual(point["r"], 1.05, places=4)
        self.assertAlmostEqual(point["z"], 0.0, places=4)
        self.assertLess(point["normalized_return_residual"], 1.0e-8)

        # The located axis is provably NOT a single-field-period fixed point.
        field_period_residual = _axis_return_residual(
            np.asarray([point["r"], point["z"]], dtype=float),
            magnetic_field=field,
            phi0=0.0,
            toroidal_span=2.0 * math.pi / 3.0,
            scale=1.0,
            rtol=1.0e-9,
            atol=1.0e-11,
            max_step=0.05,
            min_bphi_over_b=1.0e-8,
        )
        self.assertGreater(float(np.linalg.norm(field_period_residual)), 1.0e-3)

    def test_acceptance_tolerance_is_separate_from_optimizer_tolerance(self):
        captured = {}

        def fake_least_squares(_fun, guess, *, bounds, xtol, ftol, gtol, max_nfev):
            captured.update(
                {
                    "guess": guess,
                    "bounds": bounds,
                    "xtol": xtol,
                    "ftol": ftol,
                    "gtol": gtol,
                    "max_nfev": max_nfev,
                }
            )
            return SimpleNamespace(
                x=np.asarray([1.0, 0.0], dtype=float),
                fun=np.asarray([5.0e-7, 0.0], dtype=float),
                success=False,
                message="optimizer stopped before strict success",
                nfev=4,
            )

        with patch(
            "simsopt.field.magnetic_axis_helpers.least_squares",
            fake_least_squares,
        ):
            point = locate_magnetic_axis_point(
                _ShiftedAxisField(major_radius=1.0, axis_shift=0.0, iota=0.3),
                (1.05, 0.02),
                nfp=3,
                r_bounds=(0.8, 1.2),
                z_bounds=(-0.2, 0.2),
                residual_tolerance=1.0e-6,
                optimizer_tolerance=1.0e-10,
            )

        self.assertEqual(captured["xtol"], 1.0e-10)
        self.assertEqual(captured["ftol"], 1.0e-10)
        self.assertEqual(captured["gtol"], 1.0e-10)
        self.assertEqual(point["normalized_return_residual"], 5.0e-7)
        self.assertEqual(point["residual_accept_tolerance"], 1.0e-6)
        self.assertEqual(point["optimizer_tolerance"], 1.0e-10)
        self.assertFalse(point["optimizer_success"])

    def test_residual_above_acceptance_tolerance_raises_axis_not_located(self):
        def fake_least_squares(_fun, _guess, **_kwargs):
            return SimpleNamespace(
                x=np.asarray([1.0, 0.0], dtype=float),
                fun=np.asarray([2.0e-6, 0.0], dtype=float),
                success=True,
                message="success",
                nfev=3,
            )

        with patch(
            "simsopt.field.magnetic_axis_helpers.least_squares",
            fake_least_squares,
        ):
            with self.assertRaises(MagneticAxisNotLocatedError):
                locate_magnetic_axis_point(
                    _ShiftedAxisField(major_radius=1.0, axis_shift=0.0, iota=0.3),
                    (1.05, 0.02),
                    nfp=3,
                    r_bounds=(0.8, 1.2),
                    z_bounds=(-0.2, 0.2),
                    residual_tolerance=1.0e-6,
                    optimizer_tolerance=1.0e-10,
                )


class MagneticAxisHelpers(unittest.TestCase):
    def test_magnetic_axis_iota(self):
        """
        Verify that the rotational transform can be computed on axis
        """
        for get_data, target_iota in zip(
            [get_hsx_data, get_ncsx_data, get_giuliani_data],
            [1.0418687161633922, 0.39549339846119463, 0.42297724084249616],
        ):
            self.subtest_magnetic_axis_iota(get_data, target_iota)

    def subtest_magnetic_axis_iota(self, get_data, target_iota):
        curves, currents, ma = get_data()
        coils = coils_via_symmetries(curves, currents, ma.nfp, True)
        iota = compute_on_axis_iota(ma, BiotSavart(coils))
        np.testing.assert_allclose(iota, target_iota, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
