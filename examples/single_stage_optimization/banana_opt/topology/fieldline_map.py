from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import cos, isfinite, pi, sin
from typing import Protocol

import numpy as np
from scipy.integrate import solve_ivp

from .poincare_chart import PoincareChart
from .rational_target import RationalTarget


class MagneticFieldLike(Protocol):
    def set_points(self, points: np.ndarray) -> object: ...

    def B(self) -> np.ndarray: ...


class DifferentiableMagneticFieldLike(MagneticFieldLike, Protocol):
    """Field with dB_by_dX rows as Cartesian-coordinate derivatives of B."""

    def dB_by_dX(self) -> np.ndarray: ...


class LowToroidalFieldError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FieldlineIntegratorOptions:
    rtol: float = 1.0e-9
    atol: float = 1.0e-11
    max_step: float = 0.05
    samples_per_full_torus: int = 128
    min_bphi_over_b: float = 1.0e-8
    method: str = "DOP853"

    def __post_init__(self) -> None:
        rtol = float(self.rtol)
        atol = float(self.atol)
        max_step = float(self.max_step)
        min_bphi_over_b = float(self.min_bphi_over_b)
        samples = int(self.samples_per_full_torus)
        if rtol <= 0.0 or atol <= 0.0 or max_step <= 0.0:
            raise ValueError(
                "Fieldline integrator tolerances and max_step must be positive"
            )
        if min_bphi_over_b < 0.0 or not isfinite(min_bphi_over_b):
            raise ValueError("Fieldline min_bphi_over_b must be finite and nonnegative")
        if samples < 8:
            raise ValueError("Fieldline samples_per_full_torus must be at least 8")
        object.__setattr__(self, "rtol", rtol)
        object.__setattr__(self, "atol", atol)
        object.__setattr__(self, "max_step", max_step)
        object.__setattr__(self, "min_bphi_over_b", min_bphi_over_b)
        object.__setattr__(self, "samples_per_full_torus", samples)


DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS = FieldlineIntegratorOptions()


@dataclass(frozen=True, slots=True)
class CylindricalFieldComponents:
    b_r: float
    b_phi: float
    b_z: float
    b_norm: float
    bphi_over_b: float


@dataclass(frozen=True, slots=True)
class CylindricalFieldJacobian:
    components: CylindricalFieldComponents
    dbr_dr: float
    dbphi_dr: float
    dbz_dr: float
    dbr_dz: float
    dbphi_dz: float
    dbz_dz: float


@dataclass(frozen=True, slots=True)
class FieldlineReturnResult:
    initial_state: tuple[float, float]
    final_state: tuple[float, float]
    phi_grid: np.ndarray
    states: np.ndarray
    unwrapped_theta: np.ndarray
    winding: float
    min_bphi_over_b: float


@dataclass(frozen=True, slots=True)
class FieldlineTangentReturnResult:
    return_map: FieldlineReturnResult
    monodromy: np.ndarray
    trace_m: float
    det_m: float


def cartesian_from_cylindrical(radius: float, phi: float, z: float) -> np.ndarray:
    return np.asarray(
        [float(radius) * cos(float(phi)), float(radius) * sin(float(phi)), float(z)],
        dtype=float,
    )


def cylindrical_basis(phi: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = cos(float(phi))
    s = sin(float(phi))
    return (
        np.asarray([c, s, 0.0], dtype=float),
        np.asarray([-s, c, 0.0], dtype=float),
        np.asarray([0.0, 0.0, 1.0], dtype=float),
    )


def _cartesian_field_vector(
    field: MagneticFieldLike,
    *,
    radius: float,
    phi: float,
    z: float,
) -> np.ndarray:
    point = cartesian_from_cylindrical(radius, phi, z).reshape((1, 3))
    field.set_points(point)
    cartesian_b = np.asarray(field.B(), dtype=float)
    if cartesian_b.shape != (1, 3):
        raise ValueError(
            f"Magnetic field B() must have shape (1, 3), got {cartesian_b.shape}"
        )
    b_vector = cartesian_b[0]
    if not np.all(np.isfinite(b_vector)):
        raise ValueError("Magnetic field B() returned NaN/Inf values")
    return b_vector


def cylindrical_field_components(
    field: MagneticFieldLike,
    *,
    radius: float,
    phi: float,
    z: float,
) -> CylindricalFieldComponents:
    b_vector = _cartesian_field_vector(field, radius=radius, phi=phi, z=z)
    e_r, e_phi, e_z = cylindrical_basis(phi)
    b_norm = float(np.linalg.norm(b_vector))
    if b_norm <= 0.0:
        raise ValueError("Magnetic field norm must be positive")
    b_phi = float(np.dot(b_vector, e_phi))
    return CylindricalFieldComponents(
        b_r=float(np.dot(b_vector, e_r)),
        b_phi=b_phi,
        b_z=float(np.dot(b_vector, e_z)),
        b_norm=b_norm,
        bphi_over_b=abs(b_phi) / b_norm,
    )


def cylindrical_field_jacobian(
    field: DifferentiableMagneticFieldLike,
    *,
    radius: float,
    phi: float,
    z: float,
) -> CylindricalFieldJacobian:
    b_vector = _cartesian_field_vector(field, radius=radius, phi=phi, z=z)
    dB_by_dX = np.asarray(field.dB_by_dX(), dtype=float)
    if dB_by_dX.shape != (1, 3, 3):
        raise ValueError(
            f"Magnetic field dB_by_dX() must have shape (1, 3, 3), got {dB_by_dX.shape}"
        )
    field_jacobian_xyz = dB_by_dX[0]
    if not np.all(np.isfinite(field_jacobian_xyz)):
        raise ValueError("Magnetic field dB_by_dX() returned NaN/Inf values")

    e_r, e_phi, e_z = cylindrical_basis(phi)
    b_norm = float(np.linalg.norm(b_vector))
    if b_norm <= 0.0:
        raise ValueError("Magnetic field norm must be positive")
    b_phi = float(np.dot(b_vector, e_phi))
    components = CylindricalFieldComponents(
        b_r=float(np.dot(b_vector, e_r)),
        b_phi=b_phi,
        b_z=float(np.dot(b_vector, e_z)),
        b_norm=b_norm,
        bphi_over_b=abs(b_phi) / b_norm,
    )

    dB_dR = e_r @ field_jacobian_xyz
    dB_dZ = e_z @ field_jacobian_xyz
    return CylindricalFieldJacobian(
        components=components,
        dbr_dr=float(np.dot(dB_dR, e_r)),
        dbphi_dr=float(np.dot(dB_dR, e_phi)),
        dbz_dr=float(np.dot(dB_dR, e_z)),
        dbr_dz=float(np.dot(dB_dZ, e_r)),
        dbphi_dz=float(np.dot(dB_dZ, e_phi)),
        dbz_dz=float(np.dot(dB_dZ, e_z)),
    )


def fieldline_rhs_phi(
    phi: float,
    state: Sequence[float],
    *,
    field: MagneticFieldLike,
    min_bphi_over_b: float,
) -> np.ndarray:
    if len(state) != 2:
        raise ValueError("Field-line state must be [R, Z]")
    radius = float(state[0])
    z = float(state[1])
    if radius <= 0.0 or not isfinite(radius) or not isfinite(z):
        raise ValueError("Field-line state requires finite R > 0 and finite Z")
    components = cylindrical_field_components(
        field,
        radius=radius,
        phi=phi,
        z=z,
    )
    if components.bphi_over_b <= float(min_bphi_over_b):
        raise LowToroidalFieldError(
            "|B_phi|/|B| fell below the Greene return-map threshold"
        )
    return np.asarray(
        [
            radius * components.b_r / components.b_phi,
            radius * components.b_z / components.b_phi,
        ],
        dtype=float,
    )


def fieldline_rhs_and_jacobian_phi(
    phi: float,
    state: Sequence[float],
    *,
    field: DifferentiableMagneticFieldLike,
    min_bphi_over_b: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(state) != 2:
        raise ValueError("Field-line state must be [R, Z]")
    radius = float(state[0])
    z = float(state[1])
    if radius <= 0.0 or not isfinite(radius) or not isfinite(z):
        raise ValueError("Field-line state requires finite R > 0 and finite Z")
    field_jacobian = cylindrical_field_jacobian(
        field,
        radius=radius,
        phi=phi,
        z=z,
    )
    components = field_jacobian.components
    if components.bphi_over_b <= float(min_bphi_over_b):
        raise LowToroidalFieldError(
            "|B_phi|/|B| fell below the Greene return-map threshold"
        )

    inv_bphi = 1.0 / components.b_phi
    inv_bphi_squared = inv_bphi * inv_bphi
    velocity = np.asarray(
        [
            radius * components.b_r * inv_bphi,
            radius * components.b_z * inv_bphi,
        ],
        dtype=float,
    )
    jacobian = np.asarray(
        [
            [
                components.b_r * inv_bphi
                + radius
                * (
                    field_jacobian.dbr_dr * components.b_phi
                    - components.b_r * field_jacobian.dbphi_dr
                )
                * inv_bphi_squared,
                radius
                * (
                    field_jacobian.dbr_dz * components.b_phi
                    - components.b_r * field_jacobian.dbphi_dz
                )
                * inv_bphi_squared,
            ],
            [
                components.b_z * inv_bphi
                + radius
                * (
                    field_jacobian.dbz_dr * components.b_phi
                    - components.b_z * field_jacobian.dbphi_dr
                )
                * inv_bphi_squared,
                radius
                * (
                    field_jacobian.dbz_dz * components.b_phi
                    - components.b_z * field_jacobian.dbphi_dz
                )
                * inv_bphi_squared,
            ],
        ],
        dtype=float,
    )
    return velocity, jacobian


def _normalize_initial_state(initial_state: Sequence[float]) -> tuple[float, float]:
    if len(initial_state) != 2:
        raise ValueError("Field-line initial_state must be [R, Z]")
    radius = float(initial_state[0])
    z = float(initial_state[1])
    if radius <= 0.0 or not isfinite(radius) or not isfinite(z):
        raise ValueError("Field-line initial_state requires finite R > 0 and finite Z")
    return (radius, z)


def integrate_full_torus_return_map(
    field: MagneticFieldLike,
    initial_state: Sequence[float],
    *,
    chart: PoincareChart,
    phi0: float = 0.0,
    torus_turns: int = 1,
    options: FieldlineIntegratorOptions = DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
) -> FieldlineReturnResult:
    start_state = _normalize_initial_state(initial_state)
    turns = int(torus_turns)
    phi_start = float(phi0)
    if turns <= 0:
        raise ValueError("torus_turns must be positive")
    if not isfinite(phi_start):
        raise ValueError("phi0 must be finite")
    sample_count = int(options.samples_per_full_torus) * turns + 1
    phi_stop = phi_start + 2.0 * pi * float(turns)
    phi_grid = np.linspace(phi_start, phi_stop, sample_count)
    solution = solve_ivp(
        lambda phi, state: fieldline_rhs_phi(
            phi,
            state,
            field=field,
            min_bphi_over_b=options.min_bphi_over_b,
        ),
        (phi_start, phi_stop),
        np.asarray(start_state, dtype=float),
        method=options.method,
        rtol=options.rtol,
        atol=options.atol,
        max_step=options.max_step,
        t_eval=phi_grid,
    )
    if not solution.success:
        raise RuntimeError(
            f"Field-line return-map integration failed: {solution.message}"
        )
    states = np.asarray(solution.y.T, dtype=float)
    if states.shape != (sample_count, 2):
        raise RuntimeError(
            f"Field-line return-map integration returned shape {states.shape}"
        )
    bphi_ratios = [
        cylindrical_field_components(
            field,
            radius=float(state[0]),
            phi=float(phi),
            z=float(state[1]),
        ).bphi_over_b
        for phi, state in zip(phi_grid, states, strict=True)
    ]
    min_bphi_over_b = float(np.min(np.asarray(bphi_ratios, dtype=float)))
    unwrapped_theta = chart.unwrapped_thetas(states)
    return FieldlineReturnResult(
        initial_state=start_state,
        final_state=(float(states[-1, 0]), float(states[-1, 1])),
        phi_grid=phi_grid,
        states=states,
        unwrapped_theta=unwrapped_theta,
        winding=float((unwrapped_theta[-1] - unwrapped_theta[0]) / (2.0 * pi)),
        min_bphi_over_b=min_bphi_over_b,
    )


def _fieldline_tangent_rhs_phi(
    phi: float,
    augmented_state: Sequence[float],
    *,
    field: DifferentiableMagneticFieldLike,
    min_bphi_over_b: float,
) -> np.ndarray:
    state = np.asarray(augmented_state[:2], dtype=float)
    tangent_matrix = np.asarray(augmented_state[2:], dtype=float).reshape((2, 2))
    velocity, jacobian = fieldline_rhs_and_jacobian_phi(
        phi,
        state,
        field=field,
        min_bphi_over_b=min_bphi_over_b,
    )
    tangent_rhs = jacobian @ tangent_matrix
    return np.concatenate([velocity, tangent_rhs.reshape(4)])


def integrate_tangent_full_torus_return_map(
    field: DifferentiableMagneticFieldLike,
    initial_state: Sequence[float],
    *,
    chart: PoincareChart,
    phi0: float = 0.0,
    torus_turns: int = 1,
    options: FieldlineIntegratorOptions = DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
) -> FieldlineTangentReturnResult:
    start_state = _normalize_initial_state(initial_state)
    turns = int(torus_turns)
    phi_start = float(phi0)
    if turns <= 0:
        raise ValueError("torus_turns must be positive")
    if not isfinite(phi_start):
        raise ValueError("phi0 must be finite")
    sample_count = int(options.samples_per_full_torus) * turns + 1
    phi_stop = phi_start + 2.0 * pi * float(turns)
    phi_grid = np.linspace(phi_start, phi_stop, sample_count)
    augmented_initial_state = np.concatenate(
        [np.asarray(start_state, dtype=float), np.eye(2, dtype=float).reshape(4)]
    )
    solution = solve_ivp(
        lambda phi, state: _fieldline_tangent_rhs_phi(
            phi,
            state,
            field=field,
            min_bphi_over_b=options.min_bphi_over_b,
        ),
        (phi_start, phi_stop),
        augmented_initial_state,
        method=options.method,
        rtol=options.rtol,
        atol=options.atol,
        max_step=options.max_step,
        t_eval=phi_grid,
    )
    if not solution.success:
        raise RuntimeError(f"Field-line tangent integration failed: {solution.message}")
    states = np.asarray(solution.y[:2, :].T, dtype=float)
    if states.shape != (sample_count, 2):
        raise RuntimeError(
            f"Field-line tangent integration returned shape {states.shape}"
        )
    monodromy = np.asarray(solution.y[2:, -1], dtype=float).reshape((2, 2))
    bphi_ratios = [
        cylindrical_field_components(
            field,
            radius=float(state[0]),
            phi=float(phi),
            z=float(state[1]),
        ).bphi_over_b
        for phi, state in zip(phi_grid, states, strict=True)
    ]
    unwrapped_theta = chart.unwrapped_thetas(states)
    return_map = FieldlineReturnResult(
        initial_state=start_state,
        final_state=(float(states[-1, 0]), float(states[-1, 1])),
        phi_grid=phi_grid,
        states=states,
        unwrapped_theta=unwrapped_theta,
        winding=float((unwrapped_theta[-1] - unwrapped_theta[0]) / (2.0 * pi)),
        min_bphi_over_b=float(np.min(np.asarray(bphi_ratios, dtype=float))),
    )
    return FieldlineTangentReturnResult(
        return_map=return_map,
        monodromy=monodromy,
        trace_m=float(np.trace(monodromy)),
        det_m=float(np.linalg.det(monodromy)),
    )


def integrate_target_return_map(
    field: MagneticFieldLike,
    initial_state: Sequence[float],
    *,
    target: RationalTarget,
    chart: PoincareChart,
    options: FieldlineIntegratorOptions = DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
) -> FieldlineReturnResult:
    return integrate_full_torus_return_map(
        field,
        initial_state,
        chart=chart,
        phi0=target.phi0,
        torus_turns=target.q,
        options=options,
    )


def integrate_tangent_target_return_map(
    field: DifferentiableMagneticFieldLike,
    initial_state: Sequence[float],
    *,
    target: RationalTarget,
    chart: PoincareChart,
    options: FieldlineIntegratorOptions = DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
) -> FieldlineTangentReturnResult:
    return integrate_tangent_full_torus_return_map(
        field,
        initial_state,
        chart=chart,
        phi0=target.phi0,
        torus_turns=target.q,
        options=options,
    )


def target_winding_residual(
    result: FieldlineReturnResult,
    target: RationalTarget,
) -> float:
    return float(result.winding - float(target.expected_winding()))
