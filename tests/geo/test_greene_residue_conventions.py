from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import pytest

from examples.single_stage_optimization.banana_opt.topology.fieldline_map import (
    FieldlineIntegratorOptions,
    LowToroidalFieldError,
    fieldline_rhs_phi,
    integrate_full_torus_return_map,
    integrate_target_return_map,
    target_winding_residual,
)
from examples.single_stage_optimization.banana_opt.topology.greene_residue import (
    GREENE_RESIDUE_ELLIPTIC_O,
    GREENE_RESIDUE_HYPERBOLIC_X,
    GREENE_RESIDUE_PARABOLIC,
    GREENE_RESIDUE_PERIOD_DOUBLING,
    classify_greene_residue,
    greene_residue_diagnostic_from_matrix,
    greene_residue_from_trace,
)
from examples.single_stage_optimization.banana_opt.topology.poincare_chart import (
    PoincareChart,
)
from examples.single_stage_optimization.banana_opt.topology.rational_target import (
    GREENE_BRANCH_O,
    GREENE_BRANCH_X,
    GREENE_FOURIER_CONVENTION,
    GREENE_IOTA_CONVENTION,
    GREENE_MAP_CONVENTION_FULL_TORUS,
    RationalTarget,
)
from simsopt.field.magneticfieldclasses import PoloidalField, ToroidalField
from simsopt.field.tracing import compute_fieldlines


class CircularTransformField:
    def __init__(self, *, axis_r: float, axis_z: float, iota: float):
        self.axis_r = float(axis_r)
        self.axis_z = float(axis_z)
        self.iota = float(iota)
        self.points = np.empty((0, 3), dtype=float)

    def set_points(self, points: np.ndarray) -> "CircularTransformField":
        self.points = np.asarray(points, dtype=float)
        return self

    def B(self) -> np.ndarray:
        x = self.points[:, 0]
        y = self.points[:, 1]
        z = self.points[:, 2]
        radius = np.sqrt(x**2 + y**2)
        cos_phi = x / radius
        sin_phi = y / radius
        d_radius_dphi = -self.iota * (z - self.axis_z)
        d_z_dphi = self.iota * (radius - self.axis_r)
        b_phi = np.ones_like(radius)
        b_r = d_radius_dphi / radius
        b_z = d_z_dphi / radius
        b_x = b_r * cos_phi - b_phi * sin_phi
        b_y = b_r * sin_phi + b_phi * cos_phi
        return np.stack([b_x, b_y, b_z], axis=-1)


class LowToroidalRatioField:
    def __init__(self):
        self.points = np.empty((0, 3), dtype=float)

    def set_points(self, points: np.ndarray) -> "LowToroidalRatioField":
        self.points = np.asarray(points, dtype=float)
        return self

    def B(self) -> np.ndarray:
        x = self.points[:, 0]
        y = self.points[:, 1]
        radius = np.sqrt(x**2 + y**2)
        e_phi = np.stack([-y / radius, x / radius, np.zeros_like(radius)], axis=-1)
        return 1.0e-12 * e_phi + np.asarray([0.0, 0.0, 1.0], dtype=float)


def test_rational_target_locks_iota_fourier_and_manifest_conventions():
    target = RationalTarget(
        p=2,
        q=5,
        radial_label=0.4,
        radial_window=(0.3, 0.5),
        branches=(GREENE_BRANCH_O,),
        phi0=0.25,
        nfp=3,
        fourier_m=5,
        fourier_n=2,
    )

    assert target.iota == Fraction(2, 5)
    assert target.iota_float == pytest.approx(0.4)
    assert target.full_torus_span == pytest.approx((0.25, 0.25 + 2.0 * math.pi))
    assert target.periodic_span == pytest.approx((0.25, 0.25 + 10.0 * math.pi))
    assert target.expected_winding() == 2
    assert target.manifest_key() == (
        "p=2|q=5|iota=2/5|weight=1|radial_label=0.40000000000000002|"
        "radial_window=0.29999999999999999:0.5|branches=O|phi0=0.25|"
        "nfp=3|convention=iota=p/q|map=full_torus_phi_return_map|"
        "fourier=5:2:m*iota-n=0"
    )
    assert target.convention == GREENE_IOTA_CONVENTION
    assert target.map_convention == GREENE_MAP_CONVENTION_FULL_TORUS
    assert GREENE_FOURIER_CONVENTION in target.manifest_key()


def test_rational_target_rejects_ambiguous_or_inconsistent_resonance_metadata():
    with pytest.raises(ValueError, match="p/q must be reduced"):
        RationalTarget(p=2, q=4)
    with pytest.raises(ValueError, match="requires both m and n"):
        RationalTarget(p=1, q=3, fourier_m=3)
    with pytest.raises(ValueError, match="m \\* \\(p/q\\) - n = 0"):
        RationalTarget(p=1, q=3, fourier_m=3, fourier_n=2)
    with pytest.raises(ValueError, match="convention='iota=p/q'"):
        RationalTarget(p=1, q=3, convention="m/n")
    with pytest.raises(ValueError, match="full-torus return map"):
        RationalTarget(p=1, q=3, map_convention="field_period")
    with pytest.raises(ValueError, match="Unknown Greene residue branch"):
        RationalTarget(p=1, q=3, branches=(GREENE_BRANCH_X, "theta_pi"))


def test_poincare_chart_angle_radial_label_and_winding_orientation():
    chart = PoincareChart(axis_r=1.0, axis_z=-0.2, radial_label_scale=0.5)

    assert chart.theta((1.5, -0.2)) == pytest.approx(0.0)
    assert chart.theta((1.0, 0.3)) == pytest.approx(math.pi / 2.0)
    assert chart.radial_label((1.3, 0.2)) == pytest.approx(1.0)
    states = np.asarray(
        [
            [1.5, -0.2],
            [1.0, 0.3],
            [0.5, -0.2],
            [1.0, -0.7],
            [1.5, -0.2],
        ]
    )
    assert chart.winding(states) == pytest.approx(1.0)
    assert PoincareChart(axis_r=1.0, axis_z=-0.2, poloidal_orientation=-1).winding(
        states
    ) == pytest.approx(-1.0)


@pytest.mark.parametrize(
    ("trace_m", "expected_residue", "expected_classification"),
    [
        (1.5, 0.125, GREENE_RESIDUE_ELLIPTIC_O),
        (2.0, 0.0, GREENE_RESIDUE_PARABOLIC),
        (-2.0, 1.0, GREENE_RESIDUE_PERIOD_DOUBLING),
        (3.0, -0.25, GREENE_RESIDUE_HYPERBOLIC_X),
        (-3.0, 1.25, GREENE_RESIDUE_HYPERBOLIC_X),
    ],
)
def test_greene_residue_formula_and_classification(
    trace_m: float,
    expected_residue: float,
    expected_classification: str,
):
    residue = greene_residue_from_trace(trace_m)

    assert residue == pytest.approx(expected_residue)
    assert classify_greene_residue(residue) == expected_classification


def test_greene_residue_diagnostic_reads_trace_from_monodromy_matrix():
    diagnostic = greene_residue_diagnostic_from_matrix(
        np.asarray([[1.2, 0.4], [0.1, 0.3]])
    )

    assert diagnostic.trace_m == pytest.approx(1.5)
    assert diagnostic.residue == pytest.approx(0.125)
    assert diagnostic.classification == GREENE_RESIDUE_ELLIPTIC_O


def test_full_torus_return_map_closes_analytic_p_over_q_orbit():
    target = RationalTarget(p=2, q=3, fourier_m=3, fourier_n=2)
    chart = PoincareChart(axis_r=1.0, axis_z=0.0)
    field = CircularTransformField(
        axis_r=chart.axis_r,
        axis_z=chart.axis_z,
        iota=target.iota_float,
    )
    options = FieldlineIntegratorOptions(
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=0.025,
        samples_per_full_torus=96,
    )

    result = integrate_target_return_map(
        field,
        (1.2, 0.0),
        target=target,
        chart=chart,
        options=options,
    )

    assert result.final_state == pytest.approx(result.initial_state, abs=1.0e-9)
    assert result.winding == pytest.approx(2.0, abs=1.0e-9)
    assert target_winding_residual(result, target) == pytest.approx(0.0, abs=1.0e-9)
    assert result.min_bphi_over_b > options.min_bphi_over_b


def test_phi_return_map_matches_existing_section_hit_geometry_for_tokamak_field():
    axis_r = 1.0
    axis_z = 0.0
    q_safety_factor = 3.2
    initial_state = (1.05, 0.0)
    field = ToroidalField(axis_r, 1.0) + PoloidalField(
        axis_r,
        1.0,
        q_safety_factor,
    )
    chart = PoincareChart(axis_r=axis_r, axis_z=axis_z)
    options = FieldlineIntegratorOptions(
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=0.025,
        samples_per_full_torus=96,
    )

    result = integrate_full_torus_return_map(
        field,
        initial_state,
        chart=chart,
        options=options,
    )
    _, phi_hits = compute_fieldlines(
        field,
        [initial_state[0]],
        [initial_state[1]],
        tmax=10.0,
        tol=1.0e-10,
        phis=[0.0],
        stopping_criteria=[],
    )
    first_return_hit = phi_hits[0][phi_hits[0][:, 0] > 1.0e-6][0]
    hit_state = (
        math.hypot(float(first_return_hit[2]), float(first_return_hit[3])),
        float(first_return_hit[4]),
    )

    assert result.final_state == pytest.approx(hit_state, abs=5.0e-6)


def test_low_toroidal_field_ratio_is_loudly_rejected_before_division():
    with pytest.raises(LowToroidalFieldError, match="B_phi"):
        fieldline_rhs_phi(
            0.0,
            (1.1, 0.0),
            field=LowToroidalRatioField(),
            min_bphi_over_b=1.0e-6,
        )
