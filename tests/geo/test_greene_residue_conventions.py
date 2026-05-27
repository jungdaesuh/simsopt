from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
import math

import numpy as np
import pytest

from examples.single_stage_optimization.banana_opt.topology.fieldline_map import (
    FieldlineIntegratorOptions,
    LowToroidalFieldError,
    fieldline_rhs_phi,
    integrate_full_torus_return_map,
    integrate_tangent_full_torus_return_map,
    integrate_tangent_target_return_map,
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
from examples.single_stage_optimization.banana_opt.topology.periodic_orbit import (
    BRANCH_STATUS_BRANCH_MISMATCH,
    BRANCH_STATUS_CONVERGED,
    BRANCH_STATUS_OUTSIDE_RADIAL_WINDOW,
    BRANCH_STATUS_WRONG_WINDING,
    PeriodicOrbitSolverOptions,
    continue_periodic_orbit,
    discover_periodic_orbit,
    solve_periodic_orbit,
)
from examples.single_stage_optimization.banana_opt.topology.rational_target import (
    GREENE_BRANCH_O,
    GREENE_BRANCH_X,
    GREENE_FOURIER_CONVENTION,
    GREENE_IOTA_CONVENTION,
    GREENE_MAP_CONVENTION_FULL_TORUS,
    RationalTarget,
)
from examples.single_stage_optimization.banana_opt.topology.residue_diagnostics import (
    GREENE_RESIDUE_PROBE_SCHEMA_VERSION,
    radial_multistart_initial_guesses,
    run_residue_probe,
)
from simsopt.field.magneticfieldclasses import PoloidalField, ToroidalField
from simsopt.field.tracing import compute_fieldlines


def _finite_difference_dB_by_dX(
    evaluate_field: Callable[[np.ndarray], np.ndarray],
    points: np.ndarray,
) -> np.ndarray:
    epsilon = 1.0e-6
    jacobian = np.empty((points.shape[0], 3, 3), dtype=float)
    for coordinate_index in range(3):
        direction = np.zeros(3, dtype=float)
        direction[coordinate_index] = epsilon
        plus = evaluate_field(points + direction)
        minus = evaluate_field(points - direction)
        jacobian[:, coordinate_index, :] = (plus - minus) / (2.0 * epsilon)
    return jacobian


class CircularTransformField:
    def __init__(self, *, axis_r: float, axis_z: float, iota: float):
        self.axis_r = float(axis_r)
        self.axis_z = float(axis_z)
        self.iota = float(iota)
        self.points = np.empty((0, 3), dtype=float)

    def set_points(self, points: np.ndarray) -> "CircularTransformField":
        self.points = np.asarray(points, dtype=float)
        return self

    def _B_at(self, points: np.ndarray) -> np.ndarray:
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
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

    def B(self) -> np.ndarray:
        return self._B_at(self.points)

    def dB_by_dX(self) -> np.ndarray:
        return _finite_difference_dB_by_dX(self._B_at, self.points)


class DrivenPeriodicOrbitField:
    def __init__(
        self,
        *,
        axis_r: float,
        axis_z: float,
        target: RationalTarget,
        orbit_radius: float,
        phase0: float,
        tangent_generator: np.ndarray,
        orbit_winding: float | None = None,
    ):
        self.axis_r = float(axis_r)
        self.axis_z = float(axis_z)
        self.target = target
        self.orbit_radius = float(orbit_radius)
        self.phase0 = float(phase0)
        self.angular_rate = (
            target.iota_float
            if orbit_winding is None
            else float(orbit_winding) / float(target.q)
        )
        self.tangent_generator = np.asarray(tangent_generator, dtype=float)
        self.points = np.empty((0, 3), dtype=float)

    def set_points(self, points: np.ndarray) -> "DrivenPeriodicOrbitField":
        self.points = np.asarray(points, dtype=float)
        return self

    def initial_orbit_state(self) -> tuple[float, float]:
        theta = self.phase0
        return (
            self.axis_r + self.orbit_radius * math.cos(theta),
            self.axis_z + self.orbit_radius * math.sin(theta),
        )

    def _orbit_state_and_velocity(
        self,
        phi: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        theta = self.phase0 + self.angular_rate * (phi - self.target.phi0)
        orbit_r = self.axis_r + self.orbit_radius * np.cos(theta)
        orbit_z = self.axis_z + self.orbit_radius * np.sin(theta)
        orbit_dr_dphi = -self.orbit_radius * self.angular_rate * np.sin(theta)
        orbit_dz_dphi = self.orbit_radius * self.angular_rate * np.cos(theta)
        return orbit_r, orbit_z, orbit_dr_dphi, orbit_dz_dphi

    def _B_at(self, points: np.ndarray) -> np.ndarray:
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        radius = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        cos_phi = x / radius
        sin_phi = y / radius
        orbit_r, orbit_z, orbit_dr_dphi, orbit_dz_dphi = self._orbit_state_and_velocity(
            phi
        )
        delta_r = radius - orbit_r
        delta_z = z - orbit_z
        d_radius_dphi = (
            orbit_dr_dphi
            + self.tangent_generator[0, 0] * delta_r
            + self.tangent_generator[0, 1] * delta_z
        )
        d_z_dphi = (
            orbit_dz_dphi
            + self.tangent_generator[1, 0] * delta_r
            + self.tangent_generator[1, 1] * delta_z
        )
        b_phi = np.ones_like(radius)
        b_r = d_radius_dphi / radius
        b_z = d_z_dphi / radius
        b_x = b_r * cos_phi - b_phi * sin_phi
        b_y = b_r * sin_phi + b_phi * cos_phi
        return np.stack([b_x, b_y, b_z], axis=-1)

    def B(self) -> np.ndarray:
        return self._B_at(self.points)

    def dB_by_dX(self) -> np.ndarray:
        return _finite_difference_dB_by_dX(self._B_at, self.points)


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


def test_tangent_full_torus_map_matches_analytic_rotation_monodromy():
    iota = 1.0 / 3.0
    chart = PoincareChart(axis_r=1.0, axis_z=0.0)
    field = CircularTransformField(
        axis_r=chart.axis_r,
        axis_z=chart.axis_z,
        iota=iota,
    )
    options = FieldlineIntegratorOptions(
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=0.025,
        samples_per_full_torus=96,
    )
    rotation_angle = 2.0 * math.pi * iota
    expected_monodromy = np.asarray(
        [
            [math.cos(rotation_angle), -math.sin(rotation_angle)],
            [math.sin(rotation_angle), math.cos(rotation_angle)],
        ]
    )

    result = integrate_tangent_full_torus_return_map(
        field,
        (1.2, 0.0),
        chart=chart,
        options=options,
    )

    assert result.monodromy == pytest.approx(expected_monodromy, abs=1.0e-7)
    assert result.trace_m == pytest.approx(-1.0, abs=1.0e-7)
    assert result.det_m == pytest.approx(1.0, abs=1.0e-7)
    diagnostic = greene_residue_diagnostic_from_matrix(result.monodromy)
    assert diagnostic.residue == pytest.approx(0.75, abs=1.0e-7)
    assert diagnostic.classification == GREENE_RESIDUE_ELLIPTIC_O


def test_tangent_target_map_matches_centered_return_map_perturbation():
    target = RationalTarget(p=1, q=4, fourier_m=4, fourier_n=1)
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
    initial_state = np.asarray([1.2, 0.0], dtype=float)
    perturbation = np.asarray([2.0e-5, -1.0e-5], dtype=float)

    tangent_result = integrate_tangent_target_return_map(
        field,
        initial_state,
        target=target,
        chart=chart,
        options=options,
    )
    base_result = integrate_target_return_map(
        field,
        initial_state,
        target=target,
        chart=chart,
        options=options,
    )
    perturbed_result = integrate_target_return_map(
        field,
        initial_state + perturbation,
        target=target,
        chart=chart,
        options=options,
    )
    observed = np.asarray(perturbed_result.final_state) - np.asarray(
        base_result.final_state
    )

    assert tangent_result.monodromy @ perturbation == pytest.approx(
        observed,
        abs=2.0e-8,
    )
    assert tangent_result.trace_m == pytest.approx(2.0, abs=1.0e-7)
    assert tangent_result.det_m == pytest.approx(1.0, abs=1.0e-7)
    assert greene_residue_diagnostic_from_matrix(
        tangent_result.monodromy
    ).residue == pytest.approx(0.0, abs=3.0e-8)


def test_periodic_orbit_solver_converges_to_known_elliptic_branch():
    target = RationalTarget(
        p=1,
        q=1,
        radial_window=(0.18, 0.23),
        fourier_m=1,
        fourier_n=1,
    )
    chart = PoincareChart(axis_r=1.0, axis_z=0.0)
    period = 2.0 * math.pi * float(target.q)
    tangent_generator = (0.5 * math.pi / period) * np.asarray(
        [[0.0, -1.0], [1.0, 0.0]],
        dtype=float,
    )
    field = DrivenPeriodicOrbitField(
        axis_r=chart.axis_r,
        axis_z=chart.axis_z,
        target=target,
        orbit_radius=0.2,
        phase0=0.35,
        tangent_generator=tangent_generator,
    )
    integrator_options = FieldlineIntegratorOptions(
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=0.025,
        samples_per_full_torus=96,
    )
    solver_options = PeriodicOrbitSolverOptions(
        residual_tolerance=1.0e-9,
        winding_tolerance=1.0e-6,
        max_iterations=8,
        max_step_norm=0.08,
    )
    true_state = np.asarray(field.initial_orbit_state(), dtype=float)
    initial_state = true_state + np.asarray([0.018, -0.012], dtype=float)

    result = solve_periodic_orbit(
        field,
        initial_state,
        target=target,
        chart=chart,
        branch=GREENE_BRANCH_O,
        integrator_options=integrator_options,
        solver_options=solver_options,
    )

    assert result.status == BRANCH_STATUS_CONVERGED
    assert result.converged
    assert result.state == pytest.approx(true_state, abs=4.0e-7)
    assert result.closure_residual_norm < 1.0e-9
    assert result.winding == pytest.approx(1.0, abs=1.0e-7)
    assert result.radial_label == pytest.approx(0.2, abs=4.0e-7)
    assert result.residue_diagnostic.classification == GREENE_RESIDUE_ELLIPTIC_O
    assert result.residue_diagnostic.residue == pytest.approx(0.5, abs=4.0e-6)
    assert result.tangent_result.det_m == pytest.approx(1.0, abs=4.0e-6)

    branch_state = result.to_branch_state(
        generation=3,
        accepted_iteration_id="accepted-0",
    )
    continued_field = DrivenPeriodicOrbitField(
        axis_r=chart.axis_r,
        axis_z=chart.axis_z,
        target=target,
        orbit_radius=0.205,
        phase0=0.35,
        tangent_generator=tangent_generator,
    )
    continued = continue_periodic_orbit(
        continued_field,
        branch_state,
        target=target,
        chart=chart,
        integrator_options=integrator_options,
        solver_options=solver_options,
    )

    assert continued.status == BRANCH_STATUS_CONVERGED
    assert continued.state == pytest.approx(
        continued_field.initial_orbit_state(),
        abs=4.0e-7,
    )
    assert continued.radial_label == pytest.approx(0.205, abs=4.0e-7)


def test_periodic_orbit_discovery_accepts_multistart_initial_guesses():
    target = RationalTarget(
        p=1,
        q=1,
        radial_window=(0.18, 0.23),
        fourier_m=1,
        fourier_n=1,
    )
    chart = PoincareChart(axis_r=1.0, axis_z=0.0)
    period = 2.0 * math.pi * float(target.q)
    field = DrivenPeriodicOrbitField(
        axis_r=chart.axis_r,
        axis_z=chart.axis_z,
        target=target,
        orbit_radius=0.2,
        phase0=-0.25,
        tangent_generator=(0.5 * math.pi / period)
        * np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=float),
    )
    true_state = np.asarray(field.initial_orbit_state(), dtype=float)

    result = discover_periodic_orbit(
        field,
        (
            true_state + np.asarray([0.02, -0.02], dtype=float),
            true_state + np.asarray([-0.01, 0.01], dtype=float),
        ),
        target=target,
        chart=chart,
        branch=GREENE_BRANCH_O,
        integrator_options=FieldlineIntegratorOptions(
            rtol=1.0e-10,
            atol=1.0e-12,
            max_step=0.025,
            samples_per_full_torus=96,
        ),
        solver_options=PeriodicOrbitSolverOptions(
            residual_tolerance=1.0e-9,
            winding_tolerance=1.0e-6,
            max_iterations=8,
            max_step_norm=0.08,
        ),
    )

    assert result.status == BRANCH_STATUS_CONVERGED
    assert result.state == pytest.approx(true_state, abs=4.0e-7)


def test_periodic_orbit_solver_reports_radial_winding_and_branch_failures():
    target = RationalTarget(
        p=1,
        q=1,
        radial_window=(0.18, 0.23),
        fourier_m=1,
        fourier_n=1,
    )
    chart = PoincareChart(axis_r=1.0, axis_z=0.0)
    period = 2.0 * math.pi * float(target.q)
    field = DrivenPeriodicOrbitField(
        axis_r=chart.axis_r,
        axis_z=chart.axis_z,
        target=target,
        orbit_radius=0.2,
        phase0=0.0,
        tangent_generator=(0.5 * math.pi / period)
        * np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=float),
    )
    integrator_options = FieldlineIntegratorOptions(
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=0.025,
        samples_per_full_torus=96,
    )
    solver_options = PeriodicOrbitSolverOptions(
        residual_tolerance=1.0e-9,
        winding_tolerance=1.0e-6,
        max_iterations=8,
        max_step_norm=0.08,
    )
    hyperbolic_field = DrivenPeriodicOrbitField(
        axis_r=chart.axis_r,
        axis_z=chart.axis_z,
        target=target,
        orbit_radius=0.2,
        phase0=0.0,
        tangent_generator=(0.4 / period)
        * np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=float),
    )
    hyperbolic_initial_state = np.asarray(
        hyperbolic_field.initial_orbit_state(),
        dtype=float,
    ) + np.asarray([0.01, -0.015], dtype=float)
    hyperbolic_result = solve_periodic_orbit(
        hyperbolic_field,
        hyperbolic_initial_state,
        target=target,
        chart=chart,
        branch=GREENE_BRANCH_X,
        integrator_options=integrator_options,
        solver_options=solver_options,
    )

    outside_window = solve_periodic_orbit(
        field,
        field.initial_orbit_state(),
        target=RationalTarget(
            p=1,
            q=1,
            radial_window=(0.1, 0.15),
            fourier_m=1,
            fourier_n=1,
        ),
        chart=chart,
        branch=GREENE_BRANCH_O,
        integrator_options=integrator_options,
        solver_options=solver_options,
    )
    branch_mismatch = solve_periodic_orbit(
        field,
        field.initial_orbit_state(),
        target=target,
        chart=chart,
        branch=GREENE_BRANCH_X,
        integrator_options=integrator_options,
        solver_options=solver_options,
    )

    wrong_winding_target = RationalTarget(
        p=2,
        q=1,
        radial_window=(0.18, 0.23),
        fourier_m=1,
        fourier_n=2,
    )
    wrong_winding_field = DrivenPeriodicOrbitField(
        axis_r=chart.axis_r,
        axis_z=chart.axis_z,
        target=wrong_winding_target,
        orbit_radius=0.2,
        phase0=0.0,
        tangent_generator=(0.5 * math.pi / period)
        * np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=float),
        orbit_winding=1.0,
    )
    wrong_winding = solve_periodic_orbit(
        wrong_winding_field,
        wrong_winding_field.initial_orbit_state(),
        target=wrong_winding_target,
        chart=chart,
        branch=GREENE_BRANCH_O,
        integrator_options=integrator_options,
        solver_options=solver_options,
    )

    assert hyperbolic_result.status == BRANCH_STATUS_CONVERGED
    assert (
        hyperbolic_result.residue_diagnostic.classification
        == GREENE_RESIDUE_HYPERBOLIC_X
    )
    assert outside_window.status == BRANCH_STATUS_OUTSIDE_RADIAL_WINDOW
    assert branch_mismatch.status == BRANCH_STATUS_BRANCH_MISMATCH
    assert wrong_winding.status == BRANCH_STATUS_WRONG_WINDING


def test_residue_probe_serializes_required_branch_diagnostics():
    target = RationalTarget(
        p=1,
        q=1,
        radial_label=0.2,
        radial_window=(0.18, 0.23),
        branches=(GREENE_BRANCH_O,),
        fourier_m=1,
        fourier_n=1,
    )
    chart = PoincareChart(axis_r=1.0, axis_z=0.0)
    period = 2.0 * math.pi * float(target.q)
    field = DrivenPeriodicOrbitField(
        axis_r=chart.axis_r,
        axis_z=chart.axis_z,
        target=target,
        orbit_radius=0.2,
        phase0=0.0,
        tangent_generator=(0.5 * math.pi / period)
        * np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=float),
    )
    integrator_options = FieldlineIntegratorOptions(
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=0.025,
        samples_per_full_torus=96,
    )
    solver_options = PeriodicOrbitSolverOptions(
        residual_tolerance=1.0e-9,
        winding_tolerance=1.0e-6,
        max_iterations=8,
        max_step_norm=0.08,
    )

    probe = run_residue_probe(
        field,
        targets=(target,),
        chart=chart,
        integrator_options=integrator_options,
        solver_options=solver_options,
    )

    diagnostic = probe["diagnostics"][0]
    assert probe["schema_version"] == GREENE_RESIDUE_PROBE_SCHEMA_VERSION
    assert probe["branch_status_counts"] == {BRANCH_STATUS_CONVERGED: 1}
    assert diagnostic["target_id"] == target.manifest_key()
    assert diagnostic["branch"] == GREENE_BRANCH_O
    assert diagnostic["branch_status"] == BRANCH_STATUS_CONVERGED
    assert diagnostic["residue"] == pytest.approx(0.5, abs=4.0e-6)
    assert diagnostic["detM"] == pytest.approx(1.0, abs=4.0e-6)
    assert diagnostic["winding"] == pytest.approx(1.0, abs=1.0e-7)
    assert diagnostic["radial_label"] == pytest.approx(0.2, abs=4.0e-7)
    assert diagnostic["min_Bphi_over_B"] > integrator_options.min_bphi_over_b
    assert diagnostic["solver_iterations"] <= solver_options.max_iterations


def test_residue_probe_radial_multistart_requires_target_radial_label():
    target = RationalTarget(p=1, q=1, fourier_m=1, fourier_n=1)
    chart = PoincareChart(axis_r=1.0, axis_z=0.0, radial_label_scale=0.5)

    with pytest.raises(ValueError, match="radial_label"):
        radial_multistart_initial_guesses(target, chart)

    guesses = radial_multistart_initial_guesses(
        RationalTarget(
            p=1,
            q=1,
            radial_label=0.4,
            fourier_m=1,
            fourier_n=1,
        ),
        chart,
        phase_angles=(0.0, math.pi),
    )

    assert np.asarray(guesses, dtype=float) == pytest.approx(
        np.asarray([[1.2, 0.0], [0.8, 0.0]], dtype=float)
    )


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
