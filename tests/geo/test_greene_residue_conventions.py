from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
import json
import math

import numpy as np
import pytest

from examples.single_stage_optimization.banana_opt.topology import residue_diagnostics
from examples.single_stage_optimization.banana_opt.topology.fieldline_map import (
    FieldlineIntegratorOptions,
    LowToroidalFieldError,
    cartesian_from_cylindrical,
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
    BRANCH_STATUS_BAD_DETERMINANT,
    BRANCH_STATUS_BRANCH_MISMATCH,
    BRANCH_STATUS_CONVERGED,
    BRANCH_STATUS_INTEGRATION_FAILED,
    BRANCH_STATUS_OUTSIDE_RADIAL_WINDOW,
    BRANCH_STATUS_WRONG_WINDING,
    PeriodicOrbitDiscoveryError,
    PeriodicOrbitSolverOptions,
    continue_periodic_orbit,
    discover_periodic_orbit,
    solve_periodic_orbit,
    tangent_map_determinant_within_tolerance,
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
    radial_multistart_labels,
    run_residue_probe,
)
from examples.single_stage_optimization.banana_opt.topology.residue_objective import (
    DEFAULT_RESIDUE_OBJECTIVE_WEIGHT,
    DEFAULT_RESIDUE_OBJECTIVE_SAMPLES_PER_FULL_TORUS,
    BiotSavartGreeneResidueObjective,
    ResidueBranchSeed,
    load_residue_objective_seeds,
    load_residue_objective_targets,
    residue_branch_seed_from_payload,
    residue_objective_target_manifest_id,
)
from examples.single_stage_optimization.banana_opt.topology.residue_sensitivity import (
    BIOT_SAVART_BRANCH_RESOLVED_FD_MODE,
    BIOT_SAVART_BRANCH_RESOLVED_TAYLOR_MODE,
    BIOT_SAVART_BRANCH_RESIDUE_GRADIENT_ACTIVE,
    BIOT_SAVART_BRANCH_RESIDUE_GRADIENT_SATISFIED_FROZEN,
    BIOT_SAVART_BRANCH_RESOLVED_VJP_MODE,
    BIOT_SAVART_BRANCH_RESOLVED_VJP_TAYLOR_MODE,
    BRANCH_RESOLVED_FD_MODE,
    DEFAULT_RESIDUE_SATISFIED_THRESHOLD,
    FROZEN_ORBIT_FD_MODE,
    biot_savart_b_and_dB_vjp_dot_test,
    branch_resolved_biot_savart_residue_central_difference,
    branch_resolved_biot_savart_residue_taylor_diagnostic,
    branch_resolved_biot_savart_residue_vjp,
    branch_resolved_biot_savart_residue_vjp_taylor_diagnostic,
    branch_resolved_residue_central_difference,
    frozen_orbit_residue_central_difference,
    solve_biot_savart_residue_branch,
)
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import Coil, Current
from simsopt.field.magneticfieldclasses import PoloidalField, ToroidalField
from simsopt.field.tracing import compute_fieldlines
from simsopt.geo.curvexyzfourier import CurveXYZFourier


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
    def __init__(
        self,
        *,
        axis_r: float,
        axis_z: float,
        iota: float,
        geometric_winding_offset: float = 0.0,
    ):
        self.axis_r = float(axis_r)
        self.axis_z = float(axis_z)
        self.iota = float(iota)
        self.geometric_winding_offset = float(geometric_winding_offset)
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
        geometric_rate = self.iota + self.geometric_winding_offset
        d_radius_dphi = -geometric_rate * (z - self.axis_z)
        d_z_dphi = geometric_rate * (radius - self.axis_r)
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


def _make_toroidal_field_coil(phi: float) -> Coil:
    center_radius = 1.0
    coil_radius = 0.25
    radial = np.asarray([math.cos(phi), math.sin(phi), 0.0], dtype=float)
    center = center_radius * radial
    curve = CurveXYZFourier(64, 1)
    curve.set_dofs(
        [
            center[0],
            coil_radius * radial[0],
            0.0,
            center[1],
            coil_radius * radial[1],
            0.0,
            0.0,
            0.0,
            coil_radius,
        ]
    )
    current = Current(1.0e5)
    current.fix_all()
    return Coil(curve, current)


def _make_toroidal_field_biot_savart(num_coils: int = 8) -> BiotSavart:
    return BiotSavart(
        [
            _make_toroidal_field_coil(2.0 * math.pi * float(index) / num_coils)
            for index in range(num_coils)
        ]
    )


def _unit_biot_savart_direction(field: BiotSavart, *, seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=np.asarray(field.x, dtype=float).shape)
    return direction / np.linalg.norm(direction)


def _residue_vjp_test_points(result) -> np.ndarray:
    return_map = result.tangent_result.return_map
    indices = range(0, return_map.phi_grid.size, 8)
    return np.asarray(
        [
            cartesian_from_cylindrical(
                float(return_map.states[index, 0]),
                float(return_map.phi_grid[index]),
                float(return_map.states[index, 1]),
            )
            for index in indices
        ],
        dtype=float,
    )


def _biot_savart_residue_gate_inputs():
    target = RationalTarget(
        p=0,
        q=1,
        radial_label=math.sqrt(0.1**2 + 0.05**2),
        radial_window=(0.05, 0.2),
        branches=(GREENE_BRANCH_X,),
    )
    integrator_options = FieldlineIntegratorOptions(
        rtol=1.0e-8,
        atol=1.0e-10,
        max_step=0.05,
        samples_per_full_torus=DEFAULT_RESIDUE_OBJECTIVE_SAMPLES_PER_FULL_TORUS,
        min_bphi_over_b=1.0e-7,
    )
    solver_options = PeriodicOrbitSolverOptions(
        residual_tolerance=1.0e-7,
        winding_tolerance=1.0e-4,
        max_iterations=8,
        max_step_norm=0.02,
    )
    return (
        _make_toroidal_field_biot_savart(),
        target,
        PoincareChart(axis_r=1.0, axis_z=0.0),
        integrator_options,
        solver_options,
    )


def _biot_savart_residue_objective_inputs():
    field, _target, chart, integrator_options, _solver_options = (
        _biot_savart_residue_gate_inputs()
    )
    original_x = np.asarray(field.x, dtype=float).copy()
    direction = _unit_biot_savart_direction(field, seed=2)
    target = RationalTarget(
        p=0,
        q=1,
        radial_label=0.126,
        radial_window=(0.10, 0.16),
        branches=(GREENE_BRANCH_O,),
        weight=1.7,
    )
    solver_options = PeriodicOrbitSolverOptions(
        residual_tolerance=1.0e-10,
        winding_tolerance=1.0e-4,
        max_iterations=20,
        max_step_norm=0.02,
    )
    validation_id = "pytest-residue-objective"
    seed = ResidueBranchSeed(
        target_id=target.manifest_key(),
        branch=GREENE_BRANCH_O,
        section_state=(1.1, 0.05),
        validation_id=validation_id,
        optimizer_taylor_validated=True,
    )
    return (
        field,
        target,
        chart,
        integrator_options,
        solver_options,
        seed,
        validation_id,
        original_x,
        direction,
    )


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


def test_target_winding_uses_section_returns_not_continuous_helical_path():
    target = RationalTarget(p=1, q=4, nfp=5, fourier_m=4, fourier_n=1)
    chart = PoincareChart(axis_r=1.0, axis_z=0.0)
    field = CircularTransformField(
        axis_r=chart.axis_r,
        axis_z=chart.axis_z,
        iota=target.iota_float,
        geometric_winding_offset=target.nfp,
    )
    options = FieldlineIntegratorOptions(
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=0.025,
        samples_per_full_torus=256,
    )

    result = integrate_target_return_map(
        field,
        (1.2, 0.0),
        target=target,
        chart=chart,
        options=options,
    )

    assert result.final_state == pytest.approx(result.initial_state, abs=1.0e-9)
    assert chart.winding(result.states) == pytest.approx(21.0, abs=1.0e-9)
    assert result.raw_return_section_winding == pytest.approx(21.0, abs=1.0e-9)
    assert result.raw_return_section_unwrapped_theta[-1] == pytest.approx(
        42.0 * math.pi,
        abs=1.0e-9,
    )
    assert result.return_section_unwrapped_theta.size == target.q + 1
    assert result.return_section_unwrapped_theta[-1] == pytest.approx(
        2.0 * math.pi,
        abs=1.0e-9,
    )
    assert result.winding == pytest.approx(1.0, abs=1.0e-9)
    assert target_winding_residual(result, target) == pytest.approx(0.0, abs=1.0e-9)


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
        p=1,
        q=4,
        radial_window=(0.18, 0.23),
        fourier_m=4,
        fourier_n=1,
    )
    wrong_winding_period = 2.0 * math.pi * float(wrong_winding_target.q)
    wrong_winding_field = DrivenPeriodicOrbitField(
        axis_r=chart.axis_r,
        axis_z=chart.axis_z,
        target=wrong_winding_target,
        orbit_radius=0.2,
        phase0=0.0,
        tangent_generator=(0.5 * math.pi / wrong_winding_period)
        * np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=float),
        orbit_winding=0.0,
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


def test_periodic_orbit_solver_reports_bad_tangent_determinant():
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
        tangent_generator=(math.log(1.02) / period)
        * np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=float),
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
        det_tolerance=1.0e-5,
        max_iterations=8,
        max_step_norm=0.08,
    )

    result = solve_periodic_orbit(
        field,
        field.initial_orbit_state(),
        target=target,
        chart=chart,
        branch=GREENE_BRANCH_O,
        integrator_options=integrator_options,
        solver_options=solver_options,
    )

    assert result.status == BRANCH_STATUS_BAD_DETERMINANT
    assert not tangent_map_determinant_within_tolerance(
        result.tangent_result,
        solver_options,
    )


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
    assert diagnostic["raw_return_section_winding"] == pytest.approx(
        1.0,
        abs=1.0e-7,
    )
    assert diagnostic["radial_label"] == pytest.approx(0.2, abs=4.0e-7)
    assert diagnostic["min_Bphi_over_B"] > integrator_options.min_bphi_over_b
    assert diagnostic["solver_iterations"] <= solver_options.max_iterations


def test_residue_probe_serializes_branch_integration_failure(monkeypatch):
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

    def fail_discovery(
        field,
        initial_guesses,
        *,
        target,
        chart,
        branch,
        integrator_options,
        solver_options,
    ):
        raise PeriodicOrbitDiscoveryError(
            status=BRANCH_STATUS_INTEGRATION_FAILED,
            message="synthetic tangent integration failure",
        )

    monkeypatch.setattr(
        residue_diagnostics,
        "discover_periodic_orbit",
        fail_discovery,
    )

    probe = residue_diagnostics.run_residue_probe(
        object(),
        targets=(target,),
        chart=chart,
        phase_angles=(0.0, math.pi),
    )

    diagnostic = probe["diagnostics"][0]
    assert probe["branch_status_counts"] == {BRANCH_STATUS_INTEGRATION_FAILED: 1}
    assert diagnostic["target_id"] == target.manifest_key()
    assert diagnostic["branch"] == GREENE_BRANCH_O
    assert diagnostic["branch_status"] == BRANCH_STATUS_INTEGRATION_FAILED
    assert diagnostic["converged"] is False
    assert diagnostic["initial_guess_count"] == 8
    assert diagnostic["residue"] is None
    assert diagnostic["failure_message"] == "synthetic tangent integration failure"


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


def test_residue_probe_radial_multistart_spans_target_window():
    chart = PoincareChart(axis_r=1.0, axis_z=0.0, radial_label_scale=0.5)
    target = RationalTarget(
        p=1,
        q=1,
        radial_label=0.4,
        radial_window=(0.2, 0.8),
        fourier_m=1,
        fourier_n=1,
    )

    assert radial_multistart_labels(target) == pytest.approx((0.4, 0.2, 0.5, 0.8))
    guesses = radial_multistart_initial_guesses(
        target,
        chart,
        phase_angles=(0.0,),
    )

    assert np.asarray(guesses, dtype=float) == pytest.approx(
        np.asarray([[1.2, 0.0], [1.1, 0.0], [1.25, 0.0], [1.4, 0.0]], dtype=float)
    )


def test_residue_sensitivity_central_differences_match_analytic_residue_slope():
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
    orbit_radius = 0.2
    phase0 = 0.0

    def field_factory(rotation_angle: float) -> DrivenPeriodicOrbitField:
        return DrivenPeriodicOrbitField(
            axis_r=chart.axis_r,
            axis_z=chart.axis_z,
            target=target,
            orbit_radius=orbit_radius,
            phase0=phase0,
            tangent_generator=(float(rotation_angle) / period)
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
    rotation_angle = 0.5 * math.pi
    step = 1.0e-4
    fixed_state = field_factory(rotation_angle).initial_orbit_state()
    expected_derivative = 0.5 * math.sin(rotation_angle)

    frozen = frozen_orbit_residue_central_difference(
        field_factory,
        parameter_name="rotation_angle",
        parameter_value=rotation_angle,
        step=step,
        fixed_state=fixed_state,
        target=target,
        chart=chart,
        integrator_options=integrator_options,
    )
    branch_resolved = branch_resolved_residue_central_difference(
        field_factory,
        np.asarray(fixed_state, dtype=float) + np.asarray([0.015, -0.01], dtype=float),
        parameter_name="rotation_angle",
        parameter_value=rotation_angle,
        step=step,
        target=target,
        chart=chart,
        branch=GREENE_BRANCH_O,
        integrator_options=integrator_options,
        solver_options=solver_options,
    )

    assert frozen.mode == FROZEN_ORBIT_FD_MODE
    assert frozen.derivative == pytest.approx(expected_derivative, abs=1.0e-6)
    assert branch_resolved.mode == BRANCH_RESOLVED_FD_MODE
    assert branch_resolved.derivative == pytest.approx(
        expected_derivative,
        abs=1.0e-6,
    )
    assert branch_resolved.base_status == BRANCH_STATUS_CONVERGED
    assert branch_resolved.plus_status == BRANCH_STATUS_CONVERGED
    assert branch_resolved.minus_status == BRANCH_STATUS_CONVERGED
    assert branch_resolved.to_json_dict()["parameter_name"] == "rotation_angle"


def test_biot_savart_branch_residue_directional_oracle_uses_real_coil_dofs():
    field, target, chart, integrator_options, solver_options = (
        _biot_savart_residue_gate_inputs()
    )
    original_x = np.asarray(field.x, dtype=float).copy()
    direction = _unit_biot_savart_direction(field)
    step = 1.0e-8

    diagnostic = branch_resolved_biot_savart_residue_central_difference(
        field,
        (1.1, 0.05),
        direction=direction,
        step=step,
        target=target,
        chart=chart,
        branch=GREENE_BRANCH_X,
        integrator_options=integrator_options,
        solver_options=solver_options,
    )

    assert diagnostic.mode == BIOT_SAVART_BRANCH_RESOLVED_FD_MODE
    assert diagnostic.base_status == BRANCH_STATUS_CONVERGED
    assert diagnostic.plus_status == BRANCH_STATUS_CONVERGED
    assert diagnostic.minus_status == BRANCH_STATUS_CONVERGED
    assert diagnostic.direction_norm == pytest.approx(1.0)
    np.testing.assert_allclose(diagnostic.direction, direction)
    assert math.isfinite(diagnostic.derivative)
    assert diagnostic.base_winding == pytest.approx(0.0, abs=1.0e-4)
    assert diagnostic.plus_winding == pytest.approx(0.0, abs=1.0e-4)
    assert diagnostic.minus_winding == pytest.approx(0.0, abs=1.0e-4)
    np.testing.assert_allclose(
        np.asarray(diagnostic.branch_state_derivative),
        (
            np.asarray(diagnostic.plus_state, dtype=float)
            - np.asarray(diagnostic.minus_state, dtype=float)
        )
        / (2.0 * step),
    )
    np.testing.assert_allclose(
        np.asarray(diagnostic.final_state_derivative),
        (
            np.asarray(diagnostic.plus_final_state, dtype=float)
            - np.asarray(diagnostic.minus_final_state, dtype=float)
        )
        / (2.0 * step),
    )
    np.testing.assert_allclose(
        np.asarray(diagnostic.monodromy_derivative),
        (
            np.asarray(diagnostic.plus_monodromy, dtype=float)
            - np.asarray(diagnostic.minus_monodromy, dtype=float)
        )
        / (2.0 * step),
    )
    assert diagnostic.residue_derivative == pytest.approx(diagnostic.derivative)
    assert diagnostic.residue_derivative == pytest.approx(
        (diagnostic.plus_residue - diagnostic.minus_residue) / (2.0 * step)
    )
    assert diagnostic.base_residue == pytest.approx(
        (2.0 - np.trace(np.asarray(diagnostic.base_monodromy, dtype=float))) / 4.0
    )
    assert diagnostic.plus_residue == pytest.approx(
        (2.0 - np.trace(np.asarray(diagnostic.plus_monodromy, dtype=float))) / 4.0
    )
    assert diagnostic.minus_residue == pytest.approx(
        (2.0 - np.trace(np.asarray(diagnostic.minus_monodromy, dtype=float))) / 4.0
    )
    assert diagnostic.residue_derivative == pytest.approx(
        -0.25 * np.trace(np.asarray(diagnostic.monodromy_derivative, dtype=float))
    )
    for section_state, final_state, closure_residual in (
        (
            diagnostic.base_state,
            diagnostic.base_final_state,
            diagnostic.base_closure_residual,
        ),
        (
            diagnostic.plus_state,
            diagnostic.plus_final_state,
            diagnostic.plus_closure_residual,
        ),
        (
            diagnostic.minus_state,
            diagnostic.minus_final_state,
            diagnostic.minus_closure_residual,
        ),
    ):
        np.testing.assert_allclose(
            np.asarray(final_state, dtype=float)
            - np.asarray(section_state, dtype=float),
            np.asarray(closure_residual, dtype=float),
        )
    assert np.asarray(diagnostic.base_monodromy, dtype=float).shape == (2, 2)
    assert np.asarray(diagnostic.plus_monodromy, dtype=float).shape == (2, 2)
    assert np.asarray(diagnostic.minus_monodromy, dtype=float).shape == (2, 2)
    assert np.asarray(diagnostic.monodromy_derivative, dtype=float).shape == (2, 2)
    assert np.all(np.isfinite(np.asarray(diagnostic.branch_state_derivative)))
    assert np.all(np.isfinite(np.asarray(diagnostic.final_state_derivative)))
    assert np.all(np.isfinite(np.asarray(diagnostic.closure_residual_derivative)))
    assert np.all(np.isfinite(np.asarray(diagnostic.monodromy_derivative)))
    for closure_residual in (
        diagnostic.base_closure_residual,
        diagnostic.plus_closure_residual,
        diagnostic.minus_closure_residual,
    ):
        assert np.linalg.norm(np.asarray(closure_residual, dtype=float)) < 1.0e-7
    payload = diagnostic.to_json_dict()
    assert payload["mode"] == BIOT_SAVART_BRANCH_RESOLVED_FD_MODE
    np.testing.assert_allclose(payload["direction"], direction)
    assert payload["residue_derivative"] == pytest.approx(diagnostic.derivative)
    assert payload["derivative"] == pytest.approx(diagnostic.derivative)
    assert np.asarray(payload["base_final_state"], dtype=float).shape == (2,)
    assert np.asarray(payload["plus_final_state"], dtype=float).shape == (2,)
    assert np.asarray(payload["minus_final_state"], dtype=float).shape == (2,)
    assert np.asarray(payload["final_state_derivative"], dtype=float).shape == (2,)
    assert np.asarray(payload["base_closure_residual"], dtype=float).shape == (2,)
    assert np.asarray(payload["plus_closure_residual"], dtype=float).shape == (2,)
    assert np.asarray(payload["minus_closure_residual"], dtype=float).shape == (2,)
    assert np.asarray(payload["closure_residual_derivative"], dtype=float).shape == (2,)
    assert np.asarray(payload["base_monodromy"], dtype=float).shape == (2, 2)
    assert np.asarray(payload["plus_monodromy"], dtype=float).shape == (2, 2)
    assert np.asarray(payload["minus_monodromy"], dtype=float).shape == (2, 2)
    assert np.asarray(payload["monodromy_derivative"], dtype=float).shape == (2, 2)
    np.testing.assert_allclose(field.x, original_x)


def test_biot_savart_branch_residue_directional_taylor_gate_serializes_real_coil_probe():
    field, target, chart, integrator_options, solver_options = (
        _biot_savart_residue_gate_inputs()
    )
    original_x = np.asarray(field.x, dtype=float).copy()
    direction = -_unit_biot_savart_direction(field)

    diagnostic = branch_resolved_biot_savart_residue_taylor_diagnostic(
        field,
        (1.1, 0.05),
        direction=direction,
        derivative_step=1.0e-8,
        probe_steps=(4.0e-7, 2.0e-7, 1.0e-7),
        target=target,
        chart=chart,
        branch=GREENE_BRANCH_X,
        integrator_options=integrator_options,
        solver_options=solver_options,
    )

    assert diagnostic.mode == BIOT_SAVART_BRANCH_RESOLVED_TAYLOR_MODE
    assert diagnostic.base_status == BRANCH_STATUS_CONVERGED
    assert diagnostic.direction_norm == pytest.approx(1.0)
    np.testing.assert_allclose(diagnostic.direction, direction)
    assert len(diagnostic.samples) == 3
    assert len(diagnostic.observed_orders) == 2
    for sample in diagnostic.samples:
        assert sample.status == BRANCH_STATUS_CONVERGED
        assert sample.absolute_residual < 1.0e-6
        assert np.asarray(sample.state, dtype=float).shape == (2,)
    assert np.all(np.isfinite(np.asarray(diagnostic.observed_orders)))
    payload = diagnostic.to_json_dict()
    assert payload["mode"] == BIOT_SAVART_BRANCH_RESOLVED_TAYLOR_MODE
    assert len(payload["samples"]) == 3
    assert len(payload["observed_orders"]) == 2
    np.testing.assert_allclose(field.x, original_x)


def test_biot_savart_residue_branch_rejects_underresolved_rk4_tangent_map():
    (
        field,
        target,
        chart,
        integrator_options,
        solver_options,
        seed,
        _validation_id,
        original_x,
        direction,
    ) = _biot_savart_residue_objective_inputs()
    field.x = original_x + 1.0e-3 * direction
    underresolved_integrator_options = FieldlineIntegratorOptions(
        rtol=integrator_options.rtol,
        atol=integrator_options.atol,
        max_step=integrator_options.max_step,
        samples_per_full_torus=96,
        min_bphi_over_b=integrator_options.min_bphi_over_b,
    )

    result = solve_biot_savart_residue_branch(
        field,
        seed.section_state,
        target=target,
        chart=chart,
        branch=GREENE_BRANCH_O,
        integrator_options=underresolved_integrator_options,
        solver_options=solver_options,
    )

    assert result.status == BRANCH_STATUS_BAD_DETERMINANT
    assert abs(result.tangent_result.det_m - 1.0) > solver_options.det_tolerance


def test_biot_savart_residue_vjp_freezes_near_success_branch():
    field, _target, chart, integrator_options, _solver_options = (
        _biot_savart_residue_gate_inputs()
    )
    target = RationalTarget(
        p=0,
        q=1,
        radial_label=0.04,
        radial_window=(0.03, 0.2),
        branches=(GREENE_BRANCH_O,),
    )
    solver_options = PeriodicOrbitSolverOptions(
        residual_tolerance=1.0e-8,
        winding_tolerance=1.0e-4,
        max_iterations=20,
        max_step_norm=0.02,
    )

    diagnostic = branch_resolved_biot_savart_residue_vjp(
        field,
        (1.1, 0.05),
        target=target,
        chart=chart,
        branch=GREENE_BRANCH_O,
        integrator_options=integrator_options,
        solver_options=solver_options,
    )

    assert diagnostic.mode == BIOT_SAVART_BRANCH_RESOLVED_VJP_MODE
    assert (
        diagnostic.gradient_status
        == BIOT_SAVART_BRANCH_RESIDUE_GRADIENT_SATISFIED_FROZEN
    )
    assert abs(diagnostic.residue) < DEFAULT_RESIDUE_SATISFIED_THRESHOLD
    assert diagnostic.gradient_norm == pytest.approx(0.0)
    np.testing.assert_allclose(diagnostic.gradient, np.zeros_like(field.x))
    np.testing.assert_allclose(diagnostic.derivative(field), np.zeros_like(field.x))


def test_biot_savart_residue_vjp_taylor_gate_is_second_order():
    field, _target, chart, integrator_options, _solver_options = (
        _biot_savart_residue_gate_inputs()
    )
    original_x = np.asarray(field.x, dtype=float).copy()
    direction = _unit_biot_savart_direction(field, seed=2)
    field.x = original_x + 1.0e-3 * direction
    target = RationalTarget(
        p=0,
        q=1,
        radial_label=0.126,
        radial_window=(0.10, 0.16),
        branches=(GREENE_BRANCH_O,),
    )
    solver_options = PeriodicOrbitSolverOptions(
        residual_tolerance=1.0e-10,
        winding_tolerance=1.0e-4,
        max_iterations=20,
        max_step_norm=0.02,
    )

    taylor = branch_resolved_biot_savart_residue_vjp_taylor_diagnostic(
        field,
        (1.1, 0.05),
        direction=direction,
        probe_steps=(1.0e-5, 5.0e-6, 2.5e-6),
        target=target,
        chart=chart,
        branch=GREENE_BRANCH_O,
        integrator_options=integrator_options,
        solver_options=solver_options,
        r_satisfied=0.0,
    )
    gradient_field, _target, gradient_chart, gradient_integrator_options, _solver = (
        _biot_savart_residue_gate_inputs()
    )
    gradient_field.x = original_x + 1.0e-3 * direction
    gradient_diagnostic = branch_resolved_biot_savart_residue_vjp(
        gradient_field,
        (1.1, 0.05),
        target=target,
        chart=gradient_chart,
        branch=GREENE_BRANCH_O,
        integrator_options=gradient_integrator_options,
        solver_options=solver_options,
        r_satisfied=0.0,
    )

    assert gradient_diagnostic.mode == BIOT_SAVART_BRANCH_RESOLVED_VJP_MODE
    assert (
        gradient_diagnostic.gradient_status
        == BIOT_SAVART_BRANCH_RESIDUE_GRADIENT_ACTIVE
    )
    assert gradient_diagnostic.cotangent_point_count > 0
    assert gradient_diagnostic.gradient_norm > 0.0
    assert taylor.mode == BIOT_SAVART_BRANCH_RESOLVED_VJP_TAYLOR_MODE
    assert taylor.base_status == BRANCH_STATUS_CONVERGED
    assert min(taylor.observed_orders) > 1.95
    for sample in taylor.samples:
        assert sample.status == BRANCH_STATUS_CONVERGED
    np.testing.assert_allclose(field.x, original_x + 1.0e-3 * direction)
    np.testing.assert_allclose(gradient_field.x, original_x + 1.0e-3 * direction)


def test_biot_savart_greene_residue_objective_is_disabled_by_default():
    (
        field,
        target,
        chart,
        integrator_options,
        solver_options,
        _seed,
        _validation_id,
        _original_x,
        _direction,
    ) = _biot_savart_residue_objective_inputs()

    objective = BiotSavartGreeneResidueObjective(
        field,
        targets=(target,),
        chart=chart,
        integrator_options=integrator_options,
        solver_options=solver_options,
    )
    payload = objective.to_json_dict()

    assert objective.objective_weight == pytest.approx(DEFAULT_RESIDUE_OBJECTIVE_WEIGHT)
    assert objective.J() == pytest.approx(0.0)
    np.testing.assert_allclose(objective.dJ(), np.zeros_like(field.x))
    assert objective.branch_values() == ()
    assert objective.gradient_diagnostics() == ()
    assert payload["enabled"] is False
    assert payload["target_manifest_id"] == residue_objective_target_manifest_id(
        (target,)
    )
    assert payload["value"] == pytest.approx(0.0)
    assert payload["branches"] == []


def test_biot_savart_greene_residue_objective_requires_validated_manifest():
    (
        field,
        target,
        chart,
        integrator_options,
        solver_options,
        seed,
        validation_id,
        _original_x,
        _direction,
    ) = _biot_savart_residue_objective_inputs()
    target_manifest_id = residue_objective_target_manifest_id((target,))
    payload_seed = residue_branch_seed_from_payload(
        {
            "target_id": target.manifest_key(),
            "branch": GREENE_BRANCH_O,
            "section_state": (1.1, 0.05),
        },
        validation_id=validation_id,
    )

    assert payload_seed.target_id == seed.target_id
    assert payload_seed.branch == seed.branch
    assert payload_seed.section_state == seed.section_state
    assert payload_seed.validation_id == seed.validation_id
    assert payload_seed.optimizer_taylor_validated is False
    with pytest.raises(ValueError, match="validation_id"):
        BiotSavartGreeneResidueObjective(
            field,
            targets=(target,),
            chart=chart,
            branch_seeds=(seed,),
            objective_weight=1.0,
            integrator_options=integrator_options,
            solver_options=solver_options,
        )
    with pytest.raises(ValueError, match="Missing Greene residue objective branch"):
        BiotSavartGreeneResidueObjective(
            field,
            targets=(target,),
            chart=chart,
            branch_seeds=(),
            validation_id=validation_id,
            objective_weight=1.0,
            integrator_options=integrator_options,
            solver_options=solver_options,
        )
    with pytest.raises(ValueError, match="target_manifest_id"):
        BiotSavartGreeneResidueObjective(
            field,
            targets=(target,),
            chart=chart,
            branch_seeds=(seed,),
            validation_id=validation_id,
            target_manifest_id=target_manifest_id + "-stale",
            objective_weight=1.0,
            integrator_options=integrator_options,
            solver_options=solver_options,
        )
    with pytest.raises(ValueError, match="validation_id does not match"):
        BiotSavartGreeneResidueObjective(
            field,
            targets=(target,),
            chart=chart,
            branch_seeds=(
                ResidueBranchSeed(
                    target_id=target.manifest_key(),
                    branch=GREENE_BRANCH_O,
                    section_state=(1.1, 0.05),
                    validation_id="other-validation",
                ),
            ),
            validation_id=validation_id,
            objective_weight=1.0,
            integrator_options=integrator_options,
            solver_options=solver_options,
        )
    with pytest.raises(ValueError, match="optimizer Taylor validation"):
        BiotSavartGreeneResidueObjective(
            field,
            targets=(target,),
            chart=chart,
            branch_seeds=(payload_seed,),
            validation_id=validation_id,
            objective_weight=1.0,
            integrator_options=integrator_options,
            solver_options=solver_options,
        )


def test_residue_objective_seed_loader_requires_passed_validation(tmp_path):
    target_payload = {
        "p": 0,
        "q": 1,
        "radial_label": 0.126,
        "radial_window": [0.10, 0.16],
        "branches": [GREENE_BRANCH_O],
        "weight": 1.7,
    }
    targets_path = tmp_path / "targets.json"
    targets_path.write_text(
        json.dumps({"targets": [target_payload]}),
        encoding="utf-8",
    )
    targets = load_residue_objective_targets(targets_path)
    target_manifest_id = residue_objective_target_manifest_id(targets)
    target_id = targets[0].manifest_key()

    def optimizer_taylor_validation(
        *,
        branch: str = GREENE_BRANCH_O,
        observed_orders: list[float] | None = None,
        base_state: list[float] | None = None,
        residuals: list[float] | None = None,
    ) -> dict[str, object]:
        sample_residuals = (
            [1.0e-10, 2.5e-11, 6.25e-12]
            if residuals is None
            else residuals
        )
        return {
            "target_id": target_id,
            "branch": branch,
            "diagnostic": {
                "mode": BIOT_SAVART_BRANCH_RESOLVED_VJP_TAYLOR_MODE,
                "derivative_step": 1.0e-6,
                "direction_norm": 1.0,
                "direction": [1.0, 0.0],
                "base_residue": 0.01,
                "directional_derivative": 0.2,
                "base_status": BRANCH_STATUS_CONVERGED,
                "base_state": [1.1, 0.05] if base_state is None else base_state,
                "samples": [
                    {
                        "step": 1.0e-5,
                        "residue": 0.010002,
                        "first_order_prediction": 0.010002,
                        "residual": sample_residuals[0],
                        "absolute_residual": sample_residuals[0],
                        "status": BRANCH_STATUS_CONVERGED,
                        "state": [1.1, 0.05],
                    },
                    {
                        "step": 5.0e-6,
                        "residue": 0.010001,
                        "first_order_prediction": 0.010001,
                        "residual": sample_residuals[1],
                        "absolute_residual": sample_residuals[1],
                        "status": BRANCH_STATUS_CONVERGED,
                        "state": [1.1, 0.05],
                    },
                    {
                        "step": 2.5e-6,
                        "residue": 0.0100005,
                        "first_order_prediction": 0.0100005,
                        "residual": sample_residuals[2],
                        "absolute_residual": sample_residuals[2],
                        "status": BRANCH_STATUS_CONVERGED,
                        "state": [1.1, 0.05],
                    },
                ],
                "observed_orders": (
                    [2.0, 2.0] if observed_orders is None else observed_orders
                ),
            },
        }

    seed_payload = {
        "target_manifest_id": target_manifest_id,
        "validation_status": "failed",
        "validation_artifact_id": "validation-artifact",
        "branch_seeds": [
            {
                "target_id": target_id,
                "branch": GREENE_BRANCH_O,
                "section_state": [1.1, 0.05],
            },
        ],
    }
    seeds_path = tmp_path / "seeds.json"
    seeds_path.write_text(json.dumps(seed_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="validation_status"):
        load_residue_objective_seeds(
            seeds_path,
            target_manifest_id=target_manifest_id,
        )

    seed_payload["validation_status"] = "passed"
    seeds_path.write_text(json.dumps(seed_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="optimizer_taylor_validations"):
        load_residue_objective_seeds(
            seeds_path,
            target_manifest_id=target_manifest_id,
        )

    seed_payload["optimizer_taylor_validations"] = []
    seeds_path.write_text(json.dumps(seed_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing Greene residue objective"):
        load_residue_objective_seeds(
            seeds_path,
            target_manifest_id=target_manifest_id,
        )

    seed_payload["optimizer_taylor_validations"] = [
        optimizer_taylor_validation(observed_orders=[1.5, 2.0])
    ]
    seeds_path.write_text(json.dumps(seed_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="observed_orders"):
        load_residue_objective_seeds(
            seeds_path,
            target_manifest_id=target_manifest_id,
        )

    seed_payload["optimizer_taylor_validations"] = [
        optimizer_taylor_validation(
            observed_orders=[1.5, 2.0],
            residuals=[1.0e-10, 1.0e-10 / (2.0**1.5), 1.0e-10 / (2.0**3.5)],
        )
    ]
    seeds_path.write_text(json.dumps(seed_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="minimum order"):
        load_residue_objective_seeds(
            seeds_path,
            target_manifest_id=target_manifest_id,
        )

    seed_payload["optimizer_taylor_validations"] = [
        optimizer_taylor_validation(base_state=[1.11, 0.05])
    ]
    seeds_path.write_text(json.dumps(seed_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="base_state"):
        load_residue_objective_seeds(
            seeds_path,
            target_manifest_id=target_manifest_id,
        )

    seed_payload["optimizer_taylor_validations"] = [optimizer_taylor_validation()]
    seeds_path.write_text(json.dumps(seed_payload), encoding="utf-8")
    validation_id, seeds = load_residue_objective_seeds(
        seeds_path,
        target_manifest_id=target_manifest_id,
    )

    assert validation_id == "validation-artifact"
    assert seeds == (
        ResidueBranchSeed(
            target_id=target_id,
            branch=GREENE_BRANCH_O,
            section_state=(1.1, 0.05),
            validation_id="validation-artifact",
            optimizer_taylor_validated=True,
        ),
    )


def test_biot_savart_greene_residue_objective_taylor_gate_is_second_order():
    (
        field,
        target,
        chart,
        integrator_options,
        solver_options,
        seed,
        validation_id,
        original_x,
        direction,
    ) = _biot_savart_residue_objective_inputs()
    base_x = original_x + 1.0e-3 * direction
    objective_weight = 0.75
    residue_scale = 0.5
    target_manifest_id = residue_objective_target_manifest_id((target,))

    def make_objective() -> BiotSavartGreeneResidueObjective:
        return BiotSavartGreeneResidueObjective(
            field,
            targets=(target,),
            chart=chart,
            branch_seeds=(seed,),
            validation_id=validation_id,
            target_manifest_id=target_manifest_id,
            objective_weight=objective_weight,
            residue_scale=residue_scale,
            integrator_options=integrator_options,
            solver_options=solver_options,
            r_satisfied=0.0,
        )

    field.x = base_x
    objective = make_objective()
    base_value = objective.J()
    gradient = np.asarray(objective.dJ(), dtype=float)
    assert objective._branch_state_by_key == {
        (target.manifest_key(), GREENE_BRANCH_O): seed.section_state
    }
    diagnostic = branch_resolved_biot_savart_residue_vjp(
        field,
        seed.section_state,
        target=target,
        chart=chart,
        branch=GREENE_BRANCH_O,
        integrator_options=integrator_options,
        solver_options=solver_options,
        r_satisfied=0.0,
    )
    expected_value = (
        objective_weight
        * float(target.weight)
        * 0.5
        * (diagnostic.residue / residue_scale) ** 2
    )
    directional_derivative = float(np.dot(gradient, direction))
    residuals: list[float] = []

    for step in (1.0e-5, 5.0e-6, 2.5e-6):
        field.x = base_x + step * direction
        sampled_value = make_objective().J()
        residuals.append(
            abs(sampled_value - base_value - step * directional_derivative)
        )
    field.x = base_x

    observed_orders = tuple(
        math.log(residuals[index] / residuals[index + 1]) / math.log(2.0)
        for index in range(len(residuals) - 1)
    )

    payload = objective.to_json_dict()
    assert payload["target_manifest_id"] == target_manifest_id
    assert payload["enabled"] is True
    assert payload["branches"][0]["status"] == BRANCH_STATUS_CONVERGED
    assert abs(payload["branches"][0]["det_m"] - 1.0) <= solver_options.det_tolerance
    assert base_value == pytest.approx(expected_value, rel=1.0e-5)
    assert np.linalg.norm(gradient) > 0.0
    assert diagnostic.gradient_status == BIOT_SAVART_BRANCH_RESIDUE_GRADIENT_ACTIVE
    assert min(observed_orders) > 1.95


def test_biot_savart_b_and_dB_vjp_dot_test_matches_central_difference():
    field, target, chart, integrator_options, solver_options = (
        _biot_savart_residue_gate_inputs()
    )
    original_x = np.asarray(field.x, dtype=float).copy()
    orbit = solve_periodic_orbit(
        field,
        (1.1, 0.05),
        target=target,
        chart=chart,
        branch=GREENE_BRANCH_X,
        integrator_options=integrator_options,
        solver_options=solver_options,
    )
    points = _residue_vjp_test_points(orbit)
    rng = np.random.default_rng(3)
    b_cotangent = 0.1 * rng.normal(size=(points.shape[0], 3))
    grad_b_cotangent = 0.01 * rng.normal(size=(points.shape[0], 3, 3))

    diagnostic = biot_savart_b_and_dB_vjp_dot_test(
        field,
        points=points,
        b_cotangent=b_cotangent,
        grad_b_cotangent=grad_b_cotangent,
        direction=_unit_biot_savart_direction(field),
        step=1.0e-6,
    )

    assert diagnostic.absolute_error < 1.0e-9
    assert diagnostic.relative_error < 1.0e-9
    assert diagnostic.central_difference == pytest.approx(
        diagnostic.vjp_dot,
        abs=1.0e-9,
    )
    assert diagnostic.to_json_dict()["step"] == pytest.approx(1.0e-6)
    np.testing.assert_allclose(field.x, original_x)


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
