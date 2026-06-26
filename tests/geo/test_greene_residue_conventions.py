from __future__ import annotations

from collections.abc import Callable
import dataclasses
from fractions import Fraction
import json
import math

import numpy as np
import pytest

from examples.single_stage_optimization.banana_opt.topology import (
    residue_diagnostics,
    residue_seed_builder,
)
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
    BRANCH_STATUS_NEWTON_STALLED,
    BRANCH_STATUS_OUTSIDE_RADIAL_WINDOW,
    BRANCH_STATUS_WRONG_WINDING,
    PeriodicOrbitDiscoveryError,
    PeriodicOrbitResult,
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
    target_winding_ranked_initial_guesses,
)
from examples.single_stage_optimization.banana_opt.topology.residue_objective import (
    DEFAULT_RESIDUE_OBJECTIVE_WEIGHT,
    DEFAULT_RESIDUE_OBJECTIVE_SAMPLES_PER_FULL_TORUS,
    GREENE_RESIDUE_BRANCH_LOSS_RAISE,
    GREENE_RESIDUE_BRANCH_LOSS_TREAT_AS_SATISFIED,
    GREENE_RESIDUE_OBJECTIVE_VALIDATION_EVIDENCE_KIND,
    BiotSavartGreeneResidueObjective,
    ResidueBranchSeed,
    load_residue_objective_seeds,
    load_residue_objective_targets,
    residue_branch_seed_from_payload,
    residue_objective_target_manifest_id,
    residue_target_from_payload,
)
from examples.single_stage_optimization.banana_opt.topology.residue_seed_builder import (
    generate_residue_seed_files,
    rank_island_candidates,
)
from examples.single_stage_optimization.banana_opt.topology.residue_sensitivity import (
    BIOT_SAVART_BRANCH_RESOLVED_FD_MODE,
    BIOT_SAVART_BRANCH_RESOLVED_TAYLOR_MODE,
    BIOT_SAVART_BRANCH_RESIDUE_DOF_GRADIENT_METHOD,
    BIOT_SAVART_BRANCH_RESIDUE_DOF_GRADIENT_METHOD_CENTRAL_DIFFERENCE,
    BIOT_SAVART_BRANCH_RESIDUE_GRADIENT_ACTIVE,
    BIOT_SAVART_BRANCH_RESIDUE_GRADIENT_SATISFIED_FROZEN,
    BIOT_SAVART_BRANCH_RESIDUE_LOCAL_SENSITIVITY_LIMITATIONS_FD_FALLBACK,
    BIOT_SAVART_BRANCH_RESIDUE_LOCAL_SENSITIVITY_METHOD,
    BIOT_SAVART_BRANCH_RESIDUE_LOCAL_SENSITIVITY_METHOD_FD_FALLBACK,
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

# Internal helpers exercised by the analytic-vs-finite-difference cross-checks
# (item #6 validation discipline): the analytic RK4 internal Jacobians must
# agree numerically with the central-difference reference they replace.
from examples.single_stage_optimization.banana_opt.topology.residue_sensitivity import (
    _analytic_rk4_residue_state_gradient,
    _analytic_rk4_step_state_jacobian,
    _analytic_rk4_step_vjp,
    _augmented_rhs_state_jacobian,
    _field_hessian_at_point,
    _field_supports_analytic_hessian,
    _rhs_and_jacobian_from_field_data,
    _rk4_residue_state_gradient,
    _rk4_stage_cotangents,
    _rk4_step_state_jacobian,
    _rk4_tangent_return_map,
    _solve_rk4_periodic_orbit,
    _vjp_residue_sensitivity_provenance,
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


class RadialShearTransformField(CircularTransformField):
    def __init__(
        self,
        *,
        axis_r: float,
        axis_z: float,
        reference_iota: float,
        reference_minor_radius: float,
        shear: float,
    ):
        super().__init__(axis_r=axis_r, axis_z=axis_z, iota=reference_iota)
        self.reference_minor_radius = float(reference_minor_radius)
        self.shear = float(shear)

    def _B_at(self, points: np.ndarray) -> np.ndarray:
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        radius = np.sqrt(x**2 + y**2)
        minor_radius = np.sqrt((radius - self.axis_r) ** 2 + (z - self.axis_z) ** 2)
        cos_phi = x / radius
        sin_phi = y / radius
        geometric_rate = self.iota + self.shear * (
            minor_radius - self.reference_minor_radius
        )
        d_radius_dphi = -geometric_rate * (z - self.axis_z)
        d_z_dphi = geometric_rate * (radius - self.axis_r)
        b_phi = np.ones_like(radius)
        b_r = d_radius_dphi / radius
        b_z = d_z_dphi / radius
        b_x = b_r * cos_phi - b_phi * sin_phi
        b_y = b_r * sin_phi + b_phi * cos_phi
        return np.stack([b_x, b_y, b_z], axis=-1)


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


class ResonantIslandField:
    def __init__(
        self,
        *,
        axis_r: float,
        axis_z: float,
        target: RationalTarget,
        orbit_radius: float,
        phase0: float,
        shear: float,
        drive: float,
    ):
        self.axis_r = float(axis_r)
        self.axis_z = float(axis_z)
        self.target = target
        self.orbit_radius = float(orbit_radius)
        self.phase0 = float(phase0)
        self.shear = float(shear)
        self.drive = float(drive)
        self.points = np.empty((0, 3), dtype=float)

    def set_points(self, points: np.ndarray) -> "ResonantIslandField":
        self.points = np.asarray(points, dtype=float)
        return self

    def initial_orbit_state(self) -> tuple[float, float]:
        return (
            self.axis_r + self.orbit_radius * math.cos(self.phase0),
            self.axis_z + self.orbit_radius * math.sin(self.phase0),
        )

    def _B_at(self, points: np.ndarray) -> np.ndarray:
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        radius = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        cos_phi = x / radius
        sin_phi = y / radius
        minor_r = radius - self.axis_r
        minor_z = z - self.axis_z
        minor_radius = np.sqrt(minor_r**2 + minor_z**2)
        theta = np.arctan2(minor_z, minor_r)
        resonant_phase = float(self.target.q) * (theta - self.phase0) - float(
            self.target.p
        ) * (phi - self.target.phi0)
        radial_velocity = -self.drive * np.sin(resonant_phase)
        angular_velocity = self.target.iota_float + self.shear * (
            minor_radius - self.orbit_radius
        )
        d_radius_dphi = radial_velocity * np.cos(
            theta
        ) - minor_radius * angular_velocity * np.sin(theta)
        d_z_dphi = radial_velocity * np.sin(
            theta
        ) + minor_radius * angular_velocity * np.cos(theta)
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
        direct_proxy_consistency_validated=True,
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


def test_target_winding_ranked_initial_guesses_uses_real_nonzero_return_map():
    target = RationalTarget(
        p=1,
        q=4,
        radial_label=0.35,
        radial_window=(0.2, 0.5),
        branches=(GREENE_BRANCH_O,),
        fourier_m=4,
        fourier_n=1,
    )
    chart = PoincareChart(axis_r=1.0, axis_z=0.0)
    field = RadialShearTransformField(
        axis_r=chart.axis_r,
        axis_z=chart.axis_z,
        reference_iota=target.iota_float,
        reference_minor_radius=0.2,
        shear=0.8,
    )
    ranked = target_winding_ranked_initial_guesses(
        field,
        target,
        chart,
        integrator_options=FieldlineIntegratorOptions(
            rtol=1.0e-10,
            atol=1.0e-12,
            max_step=0.025,
            samples_per_full_torus=96,
        ),
        phase_angles=(0.0,),
    )

    assert chart.radial_label(ranked[0]) == pytest.approx(0.2, abs=2.0e-5)


def test_residue_probe_discovers_nonzero_p_branch_from_ranked_scan():
    target = RationalTarget(
        p=1,
        q=1,
        radial_label=0.2,
        radial_window=(0.2, 0.2),
        branches=(GREENE_BRANCH_O,),
        fourier_m=1,
        fourier_n=1,
    )
    chart = PoincareChart(axis_r=1.0, axis_z=0.0)
    period = 2.0 * math.pi * float(target.q)
    phase0 = 0.25 * math.pi
    field = DrivenPeriodicOrbitField(
        axis_r=chart.axis_r,
        axis_z=chart.axis_z,
        target=target,
        orbit_radius=0.2,
        phase0=phase0,
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
        phase_angles=(phase0,),
    )

    diagnostic = probe["diagnostics"][0]
    assert probe["branch_status_counts"] == {BRANCH_STATUS_CONVERGED: 1}
    assert diagnostic["branch_status"] == BRANCH_STATUS_CONVERGED
    assert diagnostic["winding"] == pytest.approx(1.0, abs=1.0e-7)
    assert diagnostic["section_state"] == pytest.approx(
        field.initial_orbit_state(),
        abs=4.0e-7,
    )


def test_residue_probe_discovers_q_greater_than_one_nonzero_p_branch_from_ranked_scan():
    target = RationalTarget(
        p=1,
        q=4,
        radial_label=0.2,
        radial_window=(0.2, 0.2),
        branches=(GREENE_BRANCH_O,),
        fourier_m=4,
        fourier_n=1,
    )
    chart = PoincareChart(axis_r=1.0, axis_z=0.0)
    phase0 = 0.25 * math.pi
    field = ResonantIslandField(
        axis_r=chart.axis_r,
        axis_z=chart.axis_z,
        target=target,
        orbit_radius=0.2,
        phase0=phase0,
        shear=0.1,
        drive=0.01,
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
        phase_angles=(phase0,),
    )

    diagnostic = probe["diagnostics"][0]
    assert probe["branch_status_counts"] == {BRANCH_STATUS_CONVERGED: 1}
    assert diagnostic["branch_status"] == BRANCH_STATUS_CONVERGED
    assert diagnostic["winding"] == pytest.approx(1.0, abs=1.0e-7)
    assert diagnostic["raw_return_section_winding"] == pytest.approx(
        1.0,
        abs=1.0e-7,
    )
    assert diagnostic["section_state"] == pytest.approx(
        field.initial_orbit_state(),
        abs=4.0e-7,
    )


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
    monkeypatch.setattr(
        residue_diagnostics,
        "target_winding_ranked_initial_guesses",
        lambda field, target, chart, integrator_options, phase_angles: (
            radial_multistart_initial_guesses(
                target,
                chart,
                phase_angles=phase_angles,
            )
        ),
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


def test_residue_probe_withholds_residue_for_nonconverged_returned_result(monkeypatch):
    """A returned-but-stalled branch must report NO residue/classification.

    Reproduces the certificate fail-open seen in ``slid_clean``: when no seed
    converges, ``discover_periodic_orbit`` returns the least-bad iterate (a real
    ``PeriodicOrbitResult`` whose ``status`` is ``newton_stalled``) -- NOT a
    discovery exception -- and that result still carries a ``residue_diagnostic``
    computed from the non-fixed-point monodromy. The probe payload must withhold
    that residue (and its O/X classification and traceM), because a residue from a
    point that is not a period-q fixed point is not a real island-stability verdict.

    Against the old behavior (residue serialized unconditionally) this fails:
    ``diagnostic["residue"]`` is the stalled iterate's ~0.5 residue and
    ``residue_classification`` its elliptic label. The audit diagnostics that ARE
    defined off a fixed point (detM, winding, section_state, newton_residual,
    branch_status) remain populated so the non-converged branch stays inspectable.
    """

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

    # A genuine converged solve gives a real residue_diagnostic (~0.5, elliptic).
    converged_result = solve_periodic_orbit(
        field,
        field.initial_orbit_state(),
        target=target,
        chart=chart,
        branch=GREENE_BRANCH_O,
        integrator_options=integrator_options,
        solver_options=solver_options,
    )
    assert converged_result.converged
    assert 0.0 < converged_result.residue_diagnostic.residue < 1.0
    assert (
        converged_result.residue_diagnostic.classification
        == GREENE_RESIDUE_ELLIPTIC_O
    )

    # Flip ONLY the status to newton_stalled: a real, non-trivial residue_diagnostic
    # now rides a non-converged result -- exactly the slid_clean producer's state.
    stalled_result: PeriodicOrbitResult = dataclasses.replace(
        converged_result,
        status=BRANCH_STATUS_NEWTON_STALLED,
    )
    assert stalled_result.converged is False
    assert stalled_result.residue_diagnostic.residue > 0.0

    def return_stalled(
        field,
        initial_guesses,
        *,
        target,
        chart,
        branch,
        integrator_options,
        solver_options,
    ):
        return stalled_result

    monkeypatch.setattr(
        residue_diagnostics,
        "discover_periodic_orbit",
        return_stalled,
    )

    probe = residue_diagnostics.run_residue_probe(
        field,
        targets=(target,),
        chart=chart,
        integrator_options=integrator_options,
        solver_options=solver_options,
        phase_angles=(0.0,),
    )

    diagnostic = probe["diagnostics"][0]
    # Honest non-convergence is reported, not hidden.
    assert diagnostic["branch_status"] == BRANCH_STATUS_NEWTON_STALLED
    assert diagnostic["converged"] is False
    assert probe["branch_status_counts"] == {BRANCH_STATUS_NEWTON_STALLED: 1}
    # Fail-closed: no residue, no O/X verdict, no traceM from a non-fixed point.
    assert diagnostic["residue"] is None
    assert diagnostic["residue_classification"] is None
    assert diagnostic["traceM"] is None
    # Audit fields defined off any iterate stay populated for inspection.
    assert diagnostic["detM"] is not None
    assert diagnostic["winding"] is not None
    assert diagnostic["section_state"] is not None
    assert diagnostic["newton_residual"] is not None


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


def test_target_winding_ranked_initial_guesses_prioritize_requested_basin(monkeypatch):
    chart = PoincareChart(axis_r=1.0, axis_z=0.0)
    target = RationalTarget(
        p=1,
        q=4,
        radial_label=0.2,
        radial_window=(0.2, 0.2),
        fourier_m=4,
        fourier_n=1,
    )
    requested_phase = 0.25 * math.pi
    phase_angles = (0.0, 0.5 * math.pi, requested_phase, math.pi)
    requested_state = residue_diagnostics.section_state_from_chart(
        chart,
        radial_label=0.2,
        theta=requested_phase,
    )

    class FakeReturnMap:
        def __init__(self, *, initial_state, winding: float):
            self.initial_state = initial_state
            self.final_state = initial_state
            self.winding = winding

    def fake_integrate_target_return_map(
        field,
        initial_state,
        *,
        target,
        chart,
        options,
    ):
        state = (float(initial_state[0]), float(initial_state[1]))
        winding = 1.0 if np.allclose(state, requested_state) else 0.0
        return FakeReturnMap(initial_state=state, winding=winding)

    monkeypatch.setattr(
        residue_diagnostics,
        "integrate_target_return_map",
        fake_integrate_target_return_map,
    )

    ranked = target_winding_ranked_initial_guesses(
        object(),
        target,
        chart,
        phase_angles=phase_angles,
    )

    assert ranked[0] == pytest.approx(requested_state)


def test_target_winding_ranked_initial_guesses_defers_prescan_failures(monkeypatch):
    chart = PoincareChart(axis_r=1.0, axis_z=0.0)
    target = RationalTarget(
        p=1,
        q=4,
        radial_label=0.2,
        radial_window=(0.2, 0.2),
        fourier_m=4,
        fourier_n=1,
    )
    failing_phase = 0.0
    requested_phase = 0.5 * math.pi
    failing_state = residue_diagnostics.section_state_from_chart(
        chart,
        radial_label=0.2,
        theta=failing_phase,
    )
    requested_state = residue_diagnostics.section_state_from_chart(
        chart,
        radial_label=0.2,
        theta=requested_phase,
    )

    class FakeReturnMap:
        def __init__(self, *, initial_state, winding: float):
            self.initial_state = initial_state
            self.final_state = initial_state
            self.winding = winding

    def fake_integrate_target_return_map(
        field,
        initial_state,
        *,
        target,
        chart,
        options,
    ):
        state = (float(initial_state[0]), float(initial_state[1]))
        if np.allclose(state, failing_state):
            raise RuntimeError("synthetic pre-scan integration failure")
        winding = 1.0 if np.allclose(state, requested_state) else 0.0
        return FakeReturnMap(initial_state=state, winding=winding)

    monkeypatch.setattr(
        residue_diagnostics,
        "integrate_target_return_map",
        fake_integrate_target_return_map,
    )

    ranked = target_winding_ranked_initial_guesses(
        object(),
        target,
        chart,
        phase_angles=(failing_phase, requested_phase, math.pi),
    )

    assert ranked[0] == pytest.approx(requested_state)
    assert ranked[-1] == pytest.approx(failing_state)


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
    assert math.isfinite(diagnostic.base_raw_return_section_winding)
    assert math.isfinite(diagnostic.plus_raw_return_section_winding)
    assert math.isfinite(diagnostic.minus_raw_return_section_winding)
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
    assert diagnostic.branch == GREENE_BRANCH_X
    assert diagnostic.expected_winding == pytest.approx(target.expected_winding())
    assert diagnostic.base_winding == pytest.approx(
        target.expected_winding(),
        abs=solver_options.winding_tolerance,
    )
    assert abs(diagnostic.base_winding_residual) <= solver_options.winding_tolerance
    assert math.isfinite(diagnostic.base_raw_return_section_winding)
    assert math.isfinite(diagnostic.base_det_m)
    assert diagnostic.base_residue_classification != ""
    assert diagnostic.direction_norm == pytest.approx(1.0)
    np.testing.assert_allclose(diagnostic.direction, direction)
    assert len(diagnostic.samples) == 3
    assert len(diagnostic.observed_orders) == 2
    for sample in diagnostic.samples:
        assert sample.status == BRANCH_STATUS_CONVERGED
        assert sample.branch == GREENE_BRANCH_X
        assert sample.winding == pytest.approx(
            target.expected_winding(),
            abs=solver_options.winding_tolerance,
        )
        assert abs(sample.winding_residual) <= solver_options.winding_tolerance
        assert math.isfinite(sample.raw_return_section_winding)
        assert math.isfinite(sample.det_m)
        assert sample.residue_classification != ""
        assert sample.absolute_residual < 1.0e-6
        assert np.asarray(sample.state, dtype=float).shape == (2,)
    assert np.all(np.isfinite(np.asarray(diagnostic.observed_orders)))
    payload = diagnostic.to_json_dict()
    assert payload["mode"] == BIOT_SAVART_BRANCH_RESOLVED_TAYLOR_MODE
    assert payload["branch"] == GREENE_BRANCH_X
    assert payload["expected_winding"] == pytest.approx(target.expected_winding())
    # This diagnostic drives its DOF gradient by central finite difference over
    # the DOFs (``branch_resolved_biot_savart_residue_central_difference``); it
    # never runs the analytic B/grad-B adjoint. Provenance must report the
    # central-difference path, not the analytic-VJP label.
    assert (
        payload["dof_gradient_method"]
        == BIOT_SAVART_BRANCH_RESIDUE_DOF_GRADIENT_METHOD_CENTRAL_DIFFERENCE
        == "central_finite_difference_dof"
    )
    assert payload["dof_gradient_method"] != BIOT_SAVART_BRANCH_RESIDUE_DOF_GRADIENT_METHOD
    assert (
        payload["local_sensitivity_method"]
        == BIOT_SAVART_BRANCH_RESIDUE_LOCAL_SENSITIVITY_METHOD_FD_FALLBACK
        == "central_finite_difference_rk4"
    )
    assert (
        payload["local_sensitivity_method"]
        != BIOT_SAVART_BRANCH_RESIDUE_LOCAL_SENSITIVITY_METHOD
    )
    assert payload["local_sensitivity_limitations"] == list(
        BIOT_SAVART_BRANCH_RESIDUE_LOCAL_SENSITIVITY_LIMITATIONS_FD_FALLBACK
    )
    assert len(payload["samples"]) == 3
    assert payload["samples"][0]["branch"] == GREENE_BRANCH_X
    assert math.isfinite(payload["samples"][0]["winding"])
    assert len(payload["observed_orders"]) == 2
    np.testing.assert_allclose(field.x, original_x)


def test_vjp_residue_sensitivity_provenance_is_path_aware():
    # The local-sensitivity provenance must describe the path actually taken, not
    # a fixed assumption. A field exposing the analytic Hessian (real BiotSavart)
    # routes the closed-form RK4 state Jacobian / adjoint; a Hessian-less field
    # falls back to the validated central-difference RK4 path and must self-report
    # that FD fallback (and its limitation) instead of the analytic label.
    biot_savart = _make_toroidal_field_biot_savart()
    assert _field_supports_analytic_hessian(biot_savart) is True
    analytic = _vjp_residue_sensitivity_provenance(biot_savart)
    assert analytic.dof_gradient_method == BIOT_SAVART_BRANCH_RESIDUE_DOF_GRADIENT_METHOD
    assert (
        analytic.local_sensitivity_method
        == BIOT_SAVART_BRANCH_RESIDUE_LOCAL_SENSITIVITY_METHOD
        == "analytic_rk4_hessian_vjp"
    )
    assert analytic.local_sensitivity_limitations == ()

    hessian_free_field = CircularTransformField(axis_r=1.0, axis_z=0.0, iota=0.3)
    assert _field_supports_analytic_hessian(hessian_free_field) is False
    fd_fallback = _vjp_residue_sensitivity_provenance(hessian_free_field)
    assert fd_fallback.dof_gradient_method == BIOT_SAVART_BRANCH_RESIDUE_DOF_GRADIENT_METHOD
    assert (
        fd_fallback.local_sensitivity_method
        == BIOT_SAVART_BRANCH_RESIDUE_LOCAL_SENSITIVITY_METHOD_FD_FALLBACK
        == "central_finite_difference_rk4"
    )
    assert (
        fd_fallback.local_sensitivity_limitations
        == BIOT_SAVART_BRANCH_RESIDUE_LOCAL_SENSITIVITY_LIMITATIONS_FD_FALLBACK
        == ("local_state_jacobian_uses_central_finite_difference_rk4",)
    )

    # The two records must genuinely diverge: same DOF-gradient mechanism, but a
    # different local-sensitivity label and a non-empty FD-fallback limitation.
    assert (
        analytic.local_sensitivity_method != fd_fallback.local_sensitivity_method
    )
    assert analytic.local_sensitivity_limitations == ()
    assert len(fd_fallback.local_sensitivity_limitations) == 1


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
    gradient_payload = gradient_diagnostic.to_json_dict()
    assert (
        gradient_payload["dof_gradient_method"]
        == BIOT_SAVART_BRANCH_RESIDUE_DOF_GRADIENT_METHOD
    )
    assert (
        gradient_payload["local_sensitivity_method"]
        == BIOT_SAVART_BRANCH_RESIDUE_LOCAL_SENSITIVITY_METHOD
        == "analytic_rk4_hessian_vjp"
    )
    assert gradient_payload["local_sensitivity_limitations"] == []
    taylor_payload = taylor.to_json_dict()
    assert (
        taylor_payload["dof_gradient_method"]
        == BIOT_SAVART_BRANCH_RESIDUE_DOF_GRADIENT_METHOD
    )
    assert (
        taylor_payload["local_sensitivity_method"]
        == BIOT_SAVART_BRANCH_RESIDUE_LOCAL_SENSITIVITY_METHOD
        == "analytic_rk4_hessian_vjp"
    )
    assert taylor_payload["local_sensitivity_limitations"] == []
    assert taylor.mode == BIOT_SAVART_BRANCH_RESOLVED_VJP_TAYLOR_MODE
    assert taylor.base_status == BRANCH_STATUS_CONVERGED
    assert taylor.branch == GREENE_BRANCH_O
    assert taylor.expected_winding == pytest.approx(target.expected_winding())
    assert taylor.base_winding == pytest.approx(
        target.expected_winding(),
        abs=solver_options.winding_tolerance,
    )
    assert abs(taylor.base_winding_residual) <= solver_options.winding_tolerance
    assert math.isfinite(taylor.base_raw_return_section_winding)
    assert math.isfinite(taylor.base_det_m)
    assert taylor.base_residue_classification != ""
    assert min(taylor.observed_orders) > 1.95
    for sample in taylor.samples:
        assert sample.status == BRANCH_STATUS_CONVERGED
        assert sample.branch == GREENE_BRANCH_O
        assert sample.winding == pytest.approx(
            target.expected_winding(),
            abs=solver_options.winding_tolerance,
        )
        assert abs(sample.winding_residual) <= solver_options.winding_tolerance
        assert math.isfinite(sample.raw_return_section_winding)
        assert math.isfinite(sample.det_m)
        assert sample.residue_classification != ""
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
    assert (
        payload["validation_evidence_kind"]
        == GREENE_RESIDUE_OBJECTIVE_VALIDATION_EVIDENCE_KIND
    )
    assert payload["validation_limitations"] == [
        "loader_validates_json_proof_payloads_not_fresh_field_solves",
        "no_bundled_landreman_2106_14930_poincare_or_spec_reproduction",
    ]
    assert (
        payload["dof_gradient_method"] == BIOT_SAVART_BRANCH_RESIDUE_DOF_GRADIENT_METHOD
    )
    assert (
        payload["local_sensitivity_method"]
        == BIOT_SAVART_BRANCH_RESIDUE_LOCAL_SENSITIVITY_METHOD
        == "analytic_rk4_hessian_vjp"
    )
    assert payload["local_sensitivity_limitations"] == []
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
    for flag_name in (
        "optimizer_taylor_validated",
        "direct_proxy_consistency_validated",
        "real_field_nonzero_winding_validated",
    ):
        with pytest.raises(ValueError, match=flag_name):
            residue_branch_seed_from_payload(
                {
                    "target_id": target.manifest_key(),
                    "branch": GREENE_BRANCH_O,
                    "section_state": (1.1, 0.05),
                    flag_name: "false",
                },
                validation_id=validation_id,
            )
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


def test_biot_savart_greene_residue_objective_requires_physics_validation_gates():
    (
        field,
        target,
        chart,
        integrator_options,
        solver_options,
        _seed,
        validation_id,
        _original_x,
        _direction,
    ) = _biot_savart_residue_objective_inputs()
    missing_direct_proxy = ResidueBranchSeed(
        target_id=target.manifest_key(),
        branch=GREENE_BRANCH_O,
        section_state=(1.1, 0.05),
        validation_id=validation_id,
        optimizer_taylor_validated=True,
    )

    with pytest.raises(ValueError, match="direct-vs-proxy physics validation"):
        BiotSavartGreeneResidueObjective(
            field,
            targets=(target,),
            chart=chart,
            branch_seeds=(missing_direct_proxy,),
            validation_id=validation_id,
            objective_weight=1.0,
            integrator_options=integrator_options,
            solver_options=solver_options,
        )

    nonzero_target = RationalTarget(
        p=1,
        q=1,
        radial_label=0.126,
        radial_window=(0.10, 0.16),
        branches=(GREENE_BRANCH_O,),
        weight=1.7,
        fourier_m=1,
        fourier_n=1,
    )
    missing_real_field_convergence = ResidueBranchSeed(
        target_id=nonzero_target.manifest_key(),
        branch=GREENE_BRANCH_O,
        section_state=(1.1, 0.05),
        validation_id=validation_id,
        optimizer_taylor_validated=True,
        direct_proxy_consistency_validated=True,
    )

    with pytest.raises(ValueError, match="real-field convergence validation"):
        BiotSavartGreeneResidueObjective(
            field,
            targets=(nonzero_target,),
            chart=chart,
            branch_seeds=(missing_real_field_convergence,),
            validation_id=validation_id,
            objective_weight=1.0,
            integrator_options=integrator_options,
            solver_options=solver_options,
        )

    validated_nonzero_seed = ResidueBranchSeed(
        target_id=nonzero_target.manifest_key(),
        branch=GREENE_BRANCH_O,
        section_state=(1.1, 0.05),
        validation_id=validation_id,
        optimizer_taylor_validated=True,
        direct_proxy_consistency_validated=True,
        real_field_nonzero_winding_validated=True,
    )
    objective = BiotSavartGreeneResidueObjective(
        field,
        targets=(nonzero_target,),
        chart=chart,
        branch_seeds=(validated_nonzero_seed,),
        validation_id=validation_id,
        objective_weight=1.0,
        integrator_options=integrator_options,
        solver_options=solver_options,
    )

    assert objective.target_manifest_id == residue_objective_target_manifest_id(
        (nonzero_target,)
    )


def test_residue_objective_target_loader_rejects_unknown_and_missing_keys():
    target = residue_target_from_payload(
        {
            "p": 0,
            "q": 1,
            "convention": GREENE_IOTA_CONVENTION,
            "map_convention": GREENE_MAP_CONVENTION_FULL_TORUS,
        }
    )

    assert target.convention == GREENE_IOTA_CONVENTION
    assert target.map_convention == GREENE_MAP_CONVENTION_FULL_TORUS

    with pytest.raises(ValueError, match="convention"):
        residue_target_from_payload(
            {
                "p": 0,
                "q": 1,
                "convention": "legacy",
            }
        )

    with pytest.raises(ValueError, match="Unknown Greene residue target keys"):
        residue_target_from_payload(
            {
                "p": 0,
                "q": 1,
                "unsupported": True,
            }
        )

    with pytest.raises(ValueError, match="Missing Greene residue target keys"):
        residue_target_from_payload({"p": 0})


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
            [1.0e-10, 2.5e-11, 6.25e-12] if residuals is None else residuals
        )
        sample_steps = [1.0e-5, 5.0e-6, 2.5e-6]
        samples = []
        for step, residual in zip(sample_steps, sample_residuals, strict=True):
            prediction = 0.01 + 0.2 * step
            samples.append(
                {
                    "step": step,
                    "residue": prediction + residual,
                    "first_order_prediction": prediction,
                    "residual": residual,
                    "absolute_residual": abs(residual),
                    "status": BRANCH_STATUS_CONVERGED,
                    "state": [1.1, 0.05],
                    "branch": branch,
                    "winding": 0.0,
                    "winding_residual": 0.0,
                    "raw_return_section_winding": 0.0,
                    "det_m": 1.0,
                    "residue_classification": GREENE_RESIDUE_ELLIPTIC_O,
                }
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
                "branch": branch,
                "expected_winding": 0.0,
                "base_winding": 0.0,
                "base_winding_residual": 0.0,
                "base_raw_return_section_winding": 0.0,
                "base_det_m": 1.0,
                "base_residue_classification": GREENE_RESIDUE_ELLIPTIC_O,
                "samples": samples,
                "observed_orders": (
                    [2.0, 2.0] if observed_orders is None else observed_orders
                ),
            },
        }

    def direct_proxy_validation(
        *,
        section_state: list[float] | None = None,
        direct_residue: float = 0.01,
        proxy_residue: float = 0.01,
    ) -> dict[str, object]:
        return {
            "target_id": target_id,
            "branch": GREENE_BRANCH_O,
            "section_state": [1.1, 0.05] if section_state is None else section_state,
            "validation_status": "passed",
            "direct_residue": direct_residue,
            "proxy_residue": proxy_residue,
            "absolute_residue_difference": abs(direct_residue - proxy_residue),
            "residue_difference_tolerance": 1.0e-8,
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

    branch_mismatch_validation = optimizer_taylor_validation()
    branch_mismatch_validation["diagnostic"]["samples"][0]["branch"] = GREENE_BRANCH_X
    seed_payload["optimizer_taylor_validations"] = [branch_mismatch_validation]
    seeds_path.write_text(json.dumps(seed_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="branch"):
        load_residue_objective_seeds(
            seeds_path,
            target_manifest_id=target_manifest_id,
        )

    winding_mismatch_validation = optimizer_taylor_validation()
    winding_mismatch_validation["diagnostic"]["samples"][0]["winding_residual"] = 1.0
    seed_payload["optimizer_taylor_validations"] = [winding_mismatch_validation]
    seeds_path.write_text(json.dumps(seed_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="winding"):
        load_residue_objective_seeds(
            seeds_path,
            target_manifest_id=target_manifest_id,
        )

    inconsistent_validation = optimizer_taylor_validation()
    inconsistent_validation["diagnostic"]["samples"][0]["residual"] = 5.0e-10
    seed_payload["optimizer_taylor_validations"] = [inconsistent_validation]
    seeds_path.write_text(json.dumps(seed_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="residue minus first_order_prediction"):
        load_residue_objective_seeds(
            seeds_path,
            target_manifest_id=target_manifest_id,
        )

    seed_payload["optimizer_taylor_validations"] = [optimizer_taylor_validation()]
    seed_payload["direct_proxy_consistency_validations"] = [
        direct_proxy_validation(section_state=[1.11, 0.05])
    ]
    seeds_path.write_text(json.dumps(seed_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="section_state"):
        load_residue_objective_seeds(
            seeds_path,
            target_manifest_id=target_manifest_id,
        )

    seed_payload["direct_proxy_consistency_validations"] = [
        direct_proxy_validation(direct_residue=0.011, proxy_residue=0.011)
    ]
    seeds_path.write_text(json.dumps(seed_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="base_residue"):
        load_residue_objective_seeds(
            seeds_path,
            target_manifest_id=target_manifest_id,
        )

    seed_payload["direct_proxy_consistency_validations"] = [direct_proxy_validation()]
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
            direct_proxy_consistency_validated=True,
        ),
    )

    for self_attested_flag in (
        "direct_proxy_consistency_validated",
        "real_field_nonzero_winding_validated",
    ):
        self_attested_payload = json.loads(json.dumps(seed_payload))
        self_attested_payload["branch_seeds"][0][self_attested_flag] = True
        seeds_path.write_text(json.dumps(self_attested_payload), encoding="utf-8")
        with pytest.raises(ValueError, match="self-attested"):
            load_residue_objective_seeds(
                seeds_path,
                target_manifest_id=target_manifest_id,
            )


def test_residue_objective_seed_loader_ties_real_field_validation_to_target_winding(
    tmp_path,
):
    target_payload = {
        "p": 2,
        "q": 3,
        "radial_label": 0.126,
        "radial_window": [0.10, 0.16],
        "branches": [GREENE_BRANCH_O],
    }
    targets_path = tmp_path / "targets.json"
    targets_path.write_text(
        json.dumps({"targets": [target_payload]}),
        encoding="utf-8",
    )
    targets = load_residue_objective_targets(targets_path)
    target_manifest_id = residue_objective_target_manifest_id(targets)
    target_id = targets[0].manifest_key()

    def optimizer_taylor_validation() -> dict[str, object]:
        sample_steps = [1.0e-5, 5.0e-6, 2.5e-6]
        sample_residuals = [1.0e-10, 2.5e-11, 6.25e-12]
        samples = []
        for step, residual in zip(sample_steps, sample_residuals, strict=True):
            prediction = 0.01 + 0.2 * step
            samples.append(
                {
                    "step": step,
                    "residue": prediction + residual,
                    "first_order_prediction": prediction,
                    "residual": residual,
                    "absolute_residual": abs(residual),
                    "status": BRANCH_STATUS_CONVERGED,
                    "state": [1.1, 0.05],
                    "branch": GREENE_BRANCH_O,
                    "winding": 2.0,
                    "winding_residual": 0.0,
                    "raw_return_section_winding": 2.0,
                    "det_m": 1.0,
                    "residue_classification": GREENE_RESIDUE_ELLIPTIC_O,
                }
            )
        return {
            "target_id": target_id,
            "branch": GREENE_BRANCH_O,
            "diagnostic": {
                "mode": BIOT_SAVART_BRANCH_RESOLVED_VJP_TAYLOR_MODE,
                "derivative_step": 1.0e-6,
                "direction_norm": 1.0,
                "direction": [1.0, 0.0],
                "base_residue": 0.01,
                "directional_derivative": 0.2,
                "base_status": BRANCH_STATUS_CONVERGED,
                "base_state": [1.1, 0.05],
                "branch": GREENE_BRANCH_O,
                "expected_winding": 2.0,
                "base_winding": 2.0,
                "base_winding_residual": 0.0,
                "base_raw_return_section_winding": 2.0,
                "base_det_m": 1.0,
                "base_residue_classification": GREENE_RESIDUE_ELLIPTIC_O,
                "samples": samples,
                "observed_orders": [2.0, 2.0],
            },
        }

    def real_field_validation(
        *,
        expected_winding: float,
        section_state: list[float] | None = None,
    ) -> dict[str, object]:
        return {
            "target_id": target_id,
            "branch": GREENE_BRANCH_O,
            "section_state": [1.1, 0.05] if section_state is None else section_state,
            "validation_status": "passed",
            "branch_status": BRANCH_STATUS_CONVERGED,
            "expected_winding": expected_winding,
            "observed_winding": expected_winding,
            "winding_residual": 0.0,
            "winding_tolerance": 1.0e-7,
        }

    seed_payload = {
        "target_manifest_id": target_manifest_id,
        "validation_status": "passed",
        "validation_artifact_id": "validation-artifact",
        "branch_seeds": [
            {
                "target_id": target_id,
                "branch": GREENE_BRANCH_O,
                "section_state": [1.1, 0.05],
            },
        ],
        "optimizer_taylor_validations": [optimizer_taylor_validation()],
        "real_field_nonzero_winding_validations": [
            real_field_validation(expected_winding=1.0)
        ],
    }
    seeds_path = tmp_path / "seeds.json"
    seeds_path.write_text(json.dumps(seed_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="expected_winding"):
        load_residue_objective_seeds(
            seeds_path,
            target_manifest_id=target_manifest_id,
        )

    seed_payload["real_field_nonzero_winding_validations"] = [
        real_field_validation(expected_winding=2.0, section_state=[1.11, 0.05])
    ]
    seeds_path.write_text(json.dumps(seed_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="section_state"):
        load_residue_objective_seeds(
            seeds_path,
            target_manifest_id=target_manifest_id,
        )

    seed_payload["real_field_nonzero_winding_validations"] = [
        real_field_validation(expected_winding=2.0)
    ]
    seeds_path.write_text(json.dumps(seed_payload), encoding="utf-8")
    _, seeds = load_residue_objective_seeds(
        seeds_path,
        target_manifest_id=target_manifest_id,
    )

    assert seeds == (
        ResidueBranchSeed(
            target_id=target_id,
            branch=GREENE_BRANCH_O,
            section_state=(1.1, 0.05),
            validation_id="validation-artifact",
            optimizer_taylor_validated=True,
            real_field_nonzero_winding_validated=True,
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
    assert (
        payload["validation_evidence_kind"]
        == GREENE_RESIDUE_OBJECTIVE_VALIDATION_EVIDENCE_KIND
    )
    assert payload["validation_limitations"] == [
        "loader_validates_json_proof_payloads_not_fresh_field_solves",
        "no_bundled_landreman_2106_14930_poincare_or_spec_reproduction",
    ]
    assert (
        payload["local_sensitivity_method"]
        == BIOT_SAVART_BRANCH_RESIDUE_LOCAL_SENSITIVITY_METHOD
    )
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


def test_residue_objective_branch_loss_policy_treats_lost_island_as_satisfied():
    field, _t, chart, integrator_options, _solver, _seed, validation_id, original_x, _d = (
        _biot_savart_residue_objective_inputs()
    )
    # The p=0 orbit of this coil field closes near radial_label ~0.11; a target
    # window of (0.30, 0.40) is therefore a "drifted/healed away" island that
    # converges geometrically but is flagged OUTSIDE_RADIAL_WINDOW.
    lost_target = RationalTarget(
        p=0,
        q=1,
        radial_label=0.35,
        radial_window=(0.30, 0.40),
        branches=(GREENE_BRANCH_O,),
        weight=1.7,
    )
    solver_options = PeriodicOrbitSolverOptions(
        residual_tolerance=1.0e-9,
        winding_tolerance=1.0e-4,
        max_iterations=20,
        max_step_norm=0.02,
    )
    seed = ResidueBranchSeed(
        target_id=lost_target.manifest_key(),
        branch=GREENE_BRANCH_O,
        section_state=(1.1, 0.05),
        validation_id=validation_id,
        optimizer_taylor_validated=True,
        direct_proxy_consistency_validated=True,
    )

    def make_objective(on_branch_loss: str) -> BiotSavartGreeneResidueObjective:
        return BiotSavartGreeneResidueObjective(
            field,
            targets=(lost_target,),
            chart=chart,
            branch_seeds=(seed,),
            validation_id=validation_id,
            objective_weight=0.75,
            residue_scale=0.5,
            integrator_options=integrator_options,
            solver_options=solver_options,
            r_satisfied=0.0,
            on_branch_loss=on_branch_loss,
        )

    # Default policy keeps a lost island loud.
    with pytest.raises(ValueError, match="branch solve requires"):
        make_objective(GREENE_RESIDUE_BRANCH_LOSS_RAISE).J()

    # treat_as_satisfied: zero J and dJ contribution, recorded + traceable.
    satisfied = make_objective(GREENE_RESIDUE_BRANCH_LOSS_TREAT_AS_SATISFIED)
    assert satisfied.J() == pytest.approx(0.0)
    np.testing.assert_allclose(satisfied.dJ(), np.zeros_like(field.x))
    branch_values = satisfied.branch_values()
    assert len(branch_values) == 1
    assert branch_values[0].status == BRANCH_STATUS_OUTSIDE_RADIAL_WINDOW
    assert branch_values[0].branch_objective == pytest.approx(0.0)
    assert satisfied.gradient_diagnostics() == ()
    payload = satisfied.to_json_dict()
    assert payload["enabled"] is True
    assert payload["on_branch_loss"] == GREENE_RESIDUE_BRANCH_LOSS_TREAT_AS_SATISFIED
    assert payload["value"] == pytest.approx(0.0)
    assert payload["branches"][0]["status"] == BRANCH_STATUS_OUTSIDE_RADIAL_WINDOW
    np.testing.assert_allclose(field.x, original_x)


def test_residue_objective_branch_loss_policy_keeps_integrity_failure_loud():
    field, _t, chart, integrator_options, _solver, _seed, validation_id, original_x, direction = (
        _biot_savart_residue_objective_inputs()
    )
    # An in-window target that converges at full resolution but is forced to
    # BRANCH_STATUS_BAD_DETERMINANT by an under-resolved RK4 tangent map. This
    # is a numerical-integrity failure, not a healed island, so it must stay
    # loud even under treat_as_satisfied.
    integrity_target = RationalTarget(
        p=0,
        q=1,
        radial_label=0.126,
        radial_window=(0.10, 0.16),
        branches=(GREENE_BRANCH_O,),
        weight=1.7,
    )
    seed = ResidueBranchSeed(
        target_id=integrity_target.manifest_key(),
        branch=GREENE_BRANCH_O,
        section_state=(1.1, 0.05),
        validation_id=validation_id,
        optimizer_taylor_validated=True,
        direct_proxy_consistency_validated=True,
    )
    underresolved = FieldlineIntegratorOptions(
        rtol=integrator_options.rtol,
        atol=integrator_options.atol,
        max_step=integrator_options.max_step,
        samples_per_full_torus=96,
        min_bphi_over_b=integrator_options.min_bphi_over_b,
    )
    solver_options = PeriodicOrbitSolverOptions(
        residual_tolerance=1.0e-10,
        winding_tolerance=1.0e-4,
        max_iterations=20,
        max_step_norm=0.02,
    )
    field.x = original_x + 1.0e-3 * direction
    try:
        objective = BiotSavartGreeneResidueObjective(
            field,
            targets=(integrity_target,),
            chart=chart,
            branch_seeds=(seed,),
            validation_id=validation_id,
            objective_weight=0.75,
            residue_scale=0.5,
            integrator_options=underresolved,
            solver_options=solver_options,
            r_satisfied=0.0,
            on_branch_loss=GREENE_RESIDUE_BRANCH_LOSS_TREAT_AS_SATISFIED,
        )
        with pytest.raises(ValueError, match="branch solve requires"):
            objective.J()
    finally:
        field.x = original_x


def test_residue_objective_rejects_unknown_branch_loss_policy():
    field, target, chart, integrator_options, solver_options, _seed, _vid, _x, _d = (
        _biot_savart_residue_objective_inputs()
    )
    with pytest.raises(ValueError, match="on_branch_loss"):
        BiotSavartGreeneResidueObjective(
            field,
            targets=(target,),
            chart=chart,
            integrator_options=integrator_options,
            solver_options=solver_options,
            on_branch_loss="silently_ignore",
        )


def _driven_field_for_rotation(
    chart: PoincareChart,
    target: RationalTarget,
    rotation_angle: float,
) -> DrivenPeriodicOrbitField:
    period = 2.0 * math.pi * float(target.q)
    return DrivenPeriodicOrbitField(
        axis_r=chart.axis_r,
        axis_z=chart.axis_z,
        target=target,
        orbit_radius=0.2,
        phase0=0.0,
        tangent_generator=(float(rotation_angle) / period)
        * np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=float),
    )


def test_rank_island_candidates_scales_width_with_residue():
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

    wide = rank_island_candidates(
        _driven_field_for_rotation(chart, target, 0.5 * math.pi),
        targets=(target,),
        chart=chart,
        integrator_options=integrator_options,
        solver_options=solver_options,
        phase_angles=(0.0,),
    )
    narrow = rank_island_candidates(
        _driven_field_for_rotation(chart, target, 0.1 * math.pi),
        targets=(target,),
        chart=chart,
        integrator_options=integrator_options,
        solver_options=solver_options,
        phase_angles=(0.0,),
    )

    assert len(wide) == 1
    assert len(narrow) == 1
    # Greene residue of the linearized rotation generator is (1 - cos(angle))/2,
    # so the island-width proxy is sqrt of that and grows with the rotation.
    assert wide[0].island_width_proxy == pytest.approx(
        math.sqrt((1.0 - math.cos(0.5 * math.pi)) / 2.0),
        abs=1.0e-4,
    )
    assert wide[0].island_width_proxy > narrow[0].island_width_proxy
    assert wide[0].residue_classification == GREENE_RESIDUE_ELLIPTIC_O
    assert wide[0].branch == GREENE_BRANCH_O


def test_residue_seed_builder_discovers_validates_and_round_trips(tmp_path):
    field, target, chart, integrator_options, _solver, _seed, _vid, original_x, direction = (
        _biot_savart_residue_objective_inputs()
    )
    solver_options = PeriodicOrbitSolverOptions(
        residual_tolerance=1.0e-10,
        winding_tolerance=1.0e-4,
        max_iterations=20,
        max_step_norm=0.02,
    )
    # A 1e-2 coil-DOF perturbation opens a p=0 island deep enough (|R| ~ 3e-4)
    # that the auto-discovered fixed point clears the 1.95 Taylor-order gate. A
    # 1e-3 ripple gives |R| ~ 3e-6, which the builder correctly rejects as below
    # finite-difference resolution.
    field.x = original_x + 1.0e-2 * direction
    targets_path = tmp_path / "targets.json"
    seeds_path = tmp_path / "seeds.json"
    try:
        # Full CLI core: discover + rank + validate + write, with no pinned seeds.
        manifests = generate_residue_seed_files(
            field,
            targets=(target,),
            chart=chart,
            validation_artifact_id="auto-e2e",
            targets_path=targets_path,
            seeds_path=seeds_path,
            integrator_options=integrator_options,
            solver_options=solver_options,
            phase_angles=(0.0, 0.25 * math.pi, 0.5 * math.pi),
            direction=direction,
            r_satisfied=0.0,
        )

        assert manifests.validated
        assert manifests.validated[0].taylor_min_order > 1.95

        # The emitted payloads must be accepted by the objective's own loaders,
        # which re-run every anti-self-attestation gate.
        loaded_targets = load_residue_objective_targets(targets_path)
        manifest_id = residue_objective_target_manifest_id(loaded_targets)
        assert manifest_id == manifests.seeds_payload["target_manifest_id"]
        validation_id, seeds = load_residue_objective_seeds(
            seeds_path,
            target_manifest_id=manifest_id,
        )
        assert validation_id == "auto-e2e"
        assert {seed.branch for seed in seeds} == {
            validated.branch for validated in manifests.validated
        }

        objective = BiotSavartGreeneResidueObjective(
            field,
            targets=loaded_targets,
            chart=chart,
            branch_seeds=seeds,
            validation_id=validation_id,
            objective_weight=1.0,
            integrator_options=integrator_options,
            solver_options=solver_options,
            r_satisfied=0.0,
        )
        assert objective.J() >= 0.0
        assert objective.to_json_dict()["enabled"] is True
    finally:
        field.x = original_x


def test_residue_seed_builder_skips_validation_errors_and_continues(monkeypatch):
    field, target, chart, integrator_options, solver_options, _seed, _vid, _original_x, _direction = (
        _biot_savart_residue_objective_inputs()
    )
    failing_candidate = residue_seed_builder.IslandCandidate(
        target=target,
        branch=GREENE_BRANCH_O,
        section_state=(1.1, 0.05),
        proxy_residue=0.9,
        residue_classification=GREENE_RESIDUE_ELLIPTIC_O,
        winding=target.expected_winding(),
        island_width_proxy=0.9,
    )
    passing_candidate = dataclasses.replace(
        failing_candidate,
        section_state=(1.11, 0.04),
        proxy_residue=0.1,
        island_width_proxy=0.1,
    )

    def fake_validate_candidate(_field, candidate, **_kwargs):
        if candidate is failing_candidate:
            raise ValueError(
                "Branch-resolved residue finite difference requires base branch "
                "status converged, got bad_tangent_determinant"
            )
        return (
            residue_seed_builder.ValidatedBranchSeed(
                target=candidate.target,
                branch=candidate.branch,
                seed_section_state=candidate.section_state,
                direct_residue=0.1,
                proxy_residue=candidate.proxy_residue,
                residue_difference=0.0,
                residue_difference_tolerance=1.0e-3,
                taylor_min_order=2.0,
                taylor_diagnostic={"schema_version": "test"},
                real_field=None,
            ),
            "",
        )

    monkeypatch.setattr(
        residue_seed_builder,
        "_validate_candidate",
        fake_validate_candidate,
    )

    manifests = residue_seed_builder.build_validated_residue_seeds(
        field,
        candidates=(failing_candidate, passing_candidate),
        chart=chart,
        validation_artifact_id="validation-error-continues",
        integrator_options=integrator_options,
        solver_options=solver_options,
    )

    assert [seed.seed_section_state for seed in manifests.validated] == [
        passing_candidate.section_state
    ]
    assert len(manifests.skipped) == 1
    assert manifests.skipped[0].target_id == failing_candidate.target.manifest_key()
    assert manifests.skipped[0].branch == failing_candidate.branch
    assert "bad_tangent_determinant" in manifests.skipped[0].reason


def test_residue_objective_gradient_step_reduces_residue():
    field, target, chart, integrator_options, _solver, _seed, validation_id, original_x, direction = (
        _biot_savart_residue_objective_inputs()
    )
    solver_options = PeriodicOrbitSolverOptions(
        residual_tolerance=1.0e-10,
        winding_tolerance=1.0e-4,
        max_iterations=20,
        max_step_norm=0.02,
    )
    base_x = original_x + 1.0e-2 * direction
    target_manifest_id = residue_objective_target_manifest_id((target,))
    field.x = base_x
    try:
        # Locate the p=0 island that the 1e-2 ripple opens; the converged fixed
        # point is the validated objective seed.
        located = solve_periodic_orbit(
            field,
            (1.11, -0.07),
            target=target,
            chart=chart,
            branch=GREENE_BRANCH_O,
            integrator_options=integrator_options,
            solver_options=solver_options,
        )
        assert located.status == BRANCH_STATUS_CONVERGED
        seed = ResidueBranchSeed(
            target_id=target.manifest_key(),
            branch=GREENE_BRANCH_O,
            section_state=located.state,
            validation_id=validation_id,
            optimizer_taylor_validated=True,
            direct_proxy_consistency_validated=True,
        )

        def objective_at(coil_dofs: np.ndarray) -> BiotSavartGreeneResidueObjective:
            field.x = coil_dofs
            return BiotSavartGreeneResidueObjective(
                field,
                targets=(target,),
                chart=chart,
                branch_seeds=(seed,),
                validation_id=validation_id,
                target_manifest_id=target_manifest_id,
                objective_weight=1.0,
                residue_scale=1.0,
                integrator_options=integrator_options,
                solver_options=solver_options,
                r_satisfied=0.0,
            )

        base_objective = objective_at(base_x)
        base_value = base_objective.J()
        gradient = np.asarray(base_objective.dJ(), dtype=float)
        gradient_norm = float(np.linalg.norm(gradient))
        assert base_value > 0.0
        assert gradient_norm > 0.0

        # One small step along -grad J must reduce the residue^2 objective:
        # concrete evidence the gradient is a usable island-suppression direction.
        step = 1.0e-4 / gradient_norm
        descended_value = objective_at(base_x - step * gradient).J()
        assert descended_value < base_value
    finally:
        field.x = original_x


class _HessianBackedSyntheticField:
    """Wrap a closed-form synthetic field, adding a self-consistent Hessian.

    The synthetic island/orbit fields used elsewhere in this module compute
    ``dB_by_dX`` by central finite difference of their analytic ``_B_at`` and
    expose no ``d2B_by_dXdX``. The analytic RK4 VJP needs that Hessian, so this
    wrapper supplies ``dB_by_dX`` and ``d2B_by_dXdX`` from central differences of
    the same ``_B_at``. The Hessian is therefore consistent with the gradient the
    finite-difference RK4 path reads, which is exactly what makes an
    analytic-vs-finite-difference cross-check meaningful: any residual is the
    finite-difference reference's truncation, not an analytic error.
    """

    _GRADIENT_STEP = 1.0e-6
    _HESSIAN_STEP = 1.0e-4

    def __init__(self, inner):
        self.inner = inner
        self.points = np.empty((0, 3), dtype=float)

    def set_points(self, points: np.ndarray) -> "_HessianBackedSyntheticField":
        self.points = np.asarray(points, dtype=float)
        self.inner.set_points(self.points)
        return self

    def initial_orbit_state(self):
        return self.inner.initial_orbit_state()

    def B(self) -> np.ndarray:
        return self.inner._B_at(self.points)

    def _gradient_at(self, points: np.ndarray) -> np.ndarray:
        step = self._GRADIENT_STEP
        gradient = np.empty((points.shape[0], 3, 3), dtype=float)
        for axis in range(3):
            shift = np.zeros(3, dtype=float)
            shift[axis] = step
            gradient[:, axis, :] = (
                self.inner._B_at(points + shift) - self.inner._B_at(points - shift)
            ) / (2.0 * step)
        return gradient

    def dB_by_dX(self) -> np.ndarray:
        return self._gradient_at(self.points)

    def d2B_by_dXdX(self) -> np.ndarray:
        step = self._HESSIAN_STEP
        hessian = np.empty((self.points.shape[0], 3, 3, 3), dtype=float)
        for axis in range(3):
            shift = np.zeros(3, dtype=float)
            shift[axis] = step
            hessian[:, axis, :, :] = (
                self._gradient_at(self.points + shift)
                - self._gradient_at(self.points - shift)
            ) / (2.0 * step)
        return hessian


def _resonant_island_field_with_hessian():
    target = RationalTarget(
        p=1,
        q=4,
        radial_label=0.2,
        radial_window=(0.2, 0.2),
        branches=(GREENE_BRANCH_O,),
        fourier_m=4,
        fourier_n=1,
    )
    chart = PoincareChart(axis_r=1.0, axis_z=0.0)
    phase0 = 0.25 * math.pi
    field = _HessianBackedSyntheticField(
        ResonantIslandField(
            axis_r=chart.axis_r,
            axis_z=chart.axis_z,
            target=target,
            orbit_radius=0.2,
            phase0=phase0,
            shear=0.1,
            drive=0.01,
        )
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
    return field, target, chart, integrator_options, solver_options


def test_analytic_rk4_internal_jacobians_match_fd_on_resonant_island_p_nonzero():
    # Validation discipline for item #6 on a genuine p != 0, nonzero-winding
    # island return map (residue ~ 0.5): the analytic per-step state Jacobian,
    # the per-stage B/grad-B cotangents, and d(residue)/d(state) must agree with
    # the central-difference reference they replace. The agreement floor here is
    # the synthetic field's own finite-difference dB_by_dX (~1e-6), not the
    # analytic algebra (which is machine-exact on a field with analytic
    # derivatives -- see the BiotSavart ground-truth test below).
    field, target, chart, integrator_options, solver_options = (
        _resonant_island_field_with_hessian()
    )
    orbit = _solve_rk4_periodic_orbit(
        field,
        field.initial_orbit_state(),
        target=target,
        chart=chart,
        branch=GREENE_BRANCH_O,
        integrator_options=integrator_options,
        solver_options=solver_options,
    )
    assert orbit.status == BRANCH_STATUS_CONVERGED
    assert orbit.winding == pytest.approx(1.0, abs=1.0e-6)
    assert target.p != 0
    assert abs(orbit.residue_diagnostic.residue) > 0.1

    _tangent_result, records = _rk4_tangent_return_map(
        field,
        orbit.state,
        target=target,
        chart=PoincareChart(axis_r=orbit.state[0], axis_z=orbit.state[1]),
        integrator_options=integrator_options,
        with_tape=True,
    )
    output_adjoint = np.asarray([0.3, -0.2, -0.25, 0.0, 0.0, -0.25], dtype=float)
    worst_state_jacobian_rel = 0.0
    worst_cotangent_abs = 0.0
    for record in records[::11]:
        fd_jacobian = _rk4_step_state_jacobian(
            field,
            record,
            integrator_options=integrator_options,
            local_difference_step=1.0e-6,
        )
        analytic_jacobian = _analytic_rk4_step_state_jacobian(
            field, record, integrator_options=integrator_options
        )
        worst_state_jacobian_rel = max(
            worst_state_jacobian_rel,
            float(
                np.max(np.abs(analytic_jacobian - fd_jacobian))
                / max(1.0e-12, np.max(np.abs(fd_jacobian)))
            ),
        )
        fd_b_cotangents = []
        fd_grad_b_cotangents = []
        for stage_index in (1, 2, 3, 4):
            _point, b_cotangent, grad_b_cotangent = _rk4_stage_cotangents(
                field,
                record,
                output_adjoint,
                stage_index=stage_index,
                integrator_options=integrator_options,
                local_difference_step=1.0e-6,
            )
            fd_b_cotangents.append(b_cotangent)
            fd_grad_b_cotangents.append(grad_b_cotangent)
        analytic_points: list = []
        analytic_b_cotangents: list = []
        analytic_grad_b_cotangents: list = []
        _analytic_rk4_step_vjp(
            field,
            record,
            output_adjoint,
            integrator_options=integrator_options,
            points=analytic_points,
            b_cotangents=analytic_b_cotangents,
            grad_b_cotangents=analytic_grad_b_cotangents,
        )
        for stage in range(4):
            worst_cotangent_abs = max(
                worst_cotangent_abs,
                float(np.max(np.abs(analytic_b_cotangents[stage] - fd_b_cotangents[stage]))),
                float(
                    np.max(
                        np.abs(
                            analytic_grad_b_cotangents[stage]
                            - fd_grad_b_cotangents[stage]
                        )
                    )
                ),
            )

    # Both sides read the same finite-difference field gradient, so they agree to
    # that gradient's truncation floor (~1e-6); the analytic side carries no extra
    # error of its own.
    assert worst_state_jacobian_rel < 1.0e-5
    assert worst_cotangent_abs < 1.0e-5

    fd_dresidue_dstate = _rk4_residue_state_gradient(
        field,
        orbit.state,
        target=target,
        integrator_options=integrator_options,
        local_difference_step=1.0e-6,
    )
    analytic_dresidue_dstate = _analytic_rk4_residue_state_gradient(
        field,
        orbit.state,
        target=target,
        integrator_options=integrator_options,
    )
    np.testing.assert_allclose(
        analytic_dresidue_dstate,
        fd_dresidue_dstate,
        rtol=1.0e-4,
        atol=1.0e-6,
    )


def test_analytic_rk4_state_jacobian_matches_exact_ground_truth_on_biotsavart():
    # With analytic field derivatives (BiotSavart exposes exact dB_by_dX and
    # d2B_by_dXdX), the analytic augmented-RHS state Jacobian is exact to machine
    # precision, validated against a clean central difference of the augmented
    # RHS that re-reads the analytic field at the moved point. This is the tight
    # proof that the analytic algebra itself -- not just its agreement with the
    # finite-difference RK4 step -- is correct. The check is run at generic
    # off-axis augmented states with a non-trivial tangent matrix so the Jacobian
    # is O(1) (orbit section points have a near-zero radial RHS dominated by field
    # rounding and are unsuitable for a tight relative comparison).
    field, _target, _chart, integrator_options, _solver_options = (
        _biot_savart_residue_gate_inputs()
    )
    min_bphi_over_b = integrator_options.min_bphi_over_b

    def augmented_rhs_total(augmented_state, phi):
        point = cartesian_from_cylindrical(
            float(augmented_state[0]), float(phi), float(augmented_state[1])
        )
        field.set_points(point.reshape((1, 3)))
        b = np.asarray(field.B(), dtype=float)[0]
        grad_b = np.asarray(field.dB_by_dX(), dtype=float)[0]
        state = np.asarray(augmented_state[:2], dtype=float)
        tangent = np.asarray(augmented_state[2:], dtype=float).reshape((2, 2))
        velocity, jacobian = _rhs_and_jacobian_from_field_data(
            float(phi), state, b, grad_b, min_bphi_over_b=min_bphi_over_b
        )
        return np.concatenate([velocity, (jacobian @ tangent).reshape(4)])

    sample_states = (
        (np.asarray([1.18, 0.07, 1.2, -0.3, 0.4, 0.9], dtype=float), 0.3),
        (np.asarray([0.83, -0.06, 0.8, 0.5, -0.2, 1.1], dtype=float), 1.4),
        (np.asarray([1.22, -0.09, 1.0, 0.1, -0.4, 0.7], dtype=float), 4.1),
    )
    checked = 0
    for augmented_state, phi in sample_states:
        point = cartesian_from_cylindrical(
            float(augmented_state[0]), float(phi), float(augmented_state[1])
        )
        field.set_points(point.reshape((1, 3)))
        b_vector = np.asarray(field.B(), dtype=float)[0]
        grad_b = np.asarray(field.dB_by_dX(), dtype=float)[0]
        hessian = _field_hessian_at_point(field, point)
        analytic = _augmented_rhs_state_jacobian(
            float(phi),
            augmented_state,
            b_vector,
            grad_b,
            hessian,
            min_bphi_over_b=min_bphi_over_b,
        )
        ground_truth = np.empty((6, 6), dtype=float)
        for index in range(6):
            step = 1.0e-7 * max(1.0, abs(augmented_state[index]))
            shift = np.zeros(6, dtype=float)
            shift[index] = step
            ground_truth[:, index] = (
                augmented_rhs_total(augmented_state + shift, phi)
                - augmented_rhs_total(augmented_state - shift, phi)
            ) / (2.0 * step)
        jacobian_scale = float(np.max(np.abs(ground_truth)))
        assert jacobian_scale > 1.0e-2
        # Absolute agreement scaled by the Jacobian magnitude; the residual is the
        # central-difference ground truth's own truncation, not analytic error.
        np.testing.assert_allclose(
            analytic,
            ground_truth,
            rtol=0.0,
            atol=1.0e-6 * jacobian_scale,
        )
        checked += 1
    assert checked == len(sample_states)


def test_analytic_residue_vjp_matches_true_residue_slope_on_biotsavart():
    # Decisive end-to-end check: the assembled analytic coil-DOF gradient's
    # directional derivative equals the true central-difference slope of the
    # Greene residue along the same coil-DOF direction. This is the property the
    # gradient exists to deliver, and the apples-to-apples ground truth (a real
    # residue resolve at +/- h), independent of any internal Jacobian.
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
    try:
        diagnostic = branch_resolved_biot_savart_residue_vjp(
            field,
            (1.1, 0.05),
            target=target,
            chart=chart,
            branch=GREENE_BRANCH_O,
            integrator_options=integrator_options,
            solver_options=solver_options,
            r_satisfied=0.0,
        )
        assert diagnostic.gradient_status == BIOT_SAVART_BRANCH_RESIDUE_GRADIENT_ACTIVE
        analytic_directional_derivative = float(
            np.dot(np.asarray(diagnostic.gradient, dtype=float), direction)
        )

        base_x = original_x + 1.0e-3 * direction

        def residue_at(coil_dofs: np.ndarray) -> float:
            field.x = coil_dofs
            solved = _solve_rk4_periodic_orbit(
                field,
                diagnostic.state,
                target=target,
                chart=chart,
                branch=GREENE_BRANCH_O,
                integrator_options=integrator_options,
                solver_options=solver_options,
            )
            assert solved.status == BRANCH_STATUS_CONVERGED
            return float(solved.residue_diagnostic.residue)

        step = 1.0e-5
        true_slope = (
            residue_at(base_x + step * direction) - residue_at(base_x - step * direction)
        ) / (2.0 * step)
    finally:
        field.x = original_x

    assert true_slope == pytest.approx(analytic_directional_derivative, rel=1.0e-3, abs=1.0e-7)
