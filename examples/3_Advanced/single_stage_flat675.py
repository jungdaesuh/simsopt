#!/usr/bin/env python3
"""Native C++/simsoptpp twin of the flat coupled single-stage example.

Coil, vessel, and Boozer-surface degrees of freedom are one state vector.
There is no nested equilibrium solve: the rotational transform and the net
poloidal current are the two-column least-squares solution of the Boozer
residual, and the eight production terms are public simsopt objectives on a
native ``BiotSavart``.  The outer loop is host SciPy L-BFGS-B at the archived
genuine-675 policy (``maxcor=300``, ``maxls=8``, ``ftol=0``, ``gtol=1e-3``).

This is the published form of the native lane that the JAX fused example
``examples/jax/3_Advanced/single_stage_flat675.py`` was measured against.
The CLI, budget semantics, and JSON keys match that example.  ``--smoke``
runs the same production objective on a deliberately small quadrature for a
couple of iterations.  ``--bundle`` selects the host-local frozen campaign
input; without it the script is clone-runnable from repository test-file
geometry and makes no timing claim of its own.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.optimize import minimize
from simsopt import load
from simsopt._core.derivative import Derivative
from simsopt._core.optimizable import Optimizable
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (
    BoozerResidual,
    CurveCurveDistance,
    CurveLength,
    CurveSurfaceDistance,
    LpCurveCurvature,
    NonQuasiSymmetricRatio,
    Surface,
    SurfaceRZFourier,
    SurfaceXYZTensorFourier,
    Volume,
    create_equally_spaced_curves,
)
from simsopt_jax.core.specs import surface_rz_fourier_dofs_from_spec
from simsopt_jax.examples.single_stage_flat675 import (
    FLAT675_LBFGS_HISTORY,
    FLAT675_LBFGS_MAXLS,
)
from simsopt_jax_adapters.geo import CurveCWSFourier
from simsopt_jax_adapters.geo.flat675 import (
    CERTIFIED_MPOL,
    CERTIFIED_NPHI,
    CERTIFIED_NTHETA,
    CERTIFIED_NTOR,
    CERTIFIED_STELLSYM,
    DEFAULT_FLAT675_BOOZER_POLICY,
    FLAT675_OBJECTIVE_TERM_KEYS,
    FLAT675_OUTER_DOF_COUNT,
    Flat675ContractError,
    Flat675ObjectivePolicy,
    default_flat675_objective_policy,
    fit_flat675_boundary,
    load_flat675_input_manifest,
    load_flat675_vessel_template,
    surface_quadrature_range,
    synthesize_flat675_vessel,
)
from simsopt_jax_adapters.geo.surface_objectives import SurfaceSurfaceDistance

EXAMPLE_ID = "flat675-single-stage-coupled-optimization"

TEST_DATA = Path(__file__).resolve().parents[2] / "tests" / "test_files"
BOUNDARY_INPUT = TEST_DATA / "input.LandremanPaul2021_QA_lowres"

BUNDLE_ROOT = (
    Path.home() / "simsopt_mixed_artifacts" / "genuine675-r3-input-1c23f6c5-20260721-r1"
)
BUNDLE_FLAG = "--bundle"
NATIVE_BIOT_SAVART_FILENAME = "native_biot_savart.json"

BOUNDED_STEPS = 2
NATIVE_DEFAULT_STEPS = 20

BOUNDED_GRID = 6
NATIVE_DEFAULT_GRID = 16
BOUNDED_CURVE_QUADPOINTS = 24
NATIVE_DEFAULT_CURVE_QUADPOINTS = 64

WINDING_SURFACE_FACTOR = 2.2
TF_COIL_RADIUS_FACTOR = 2.6
TF_BASE_COIL_COUNT = 3
TF_COIL_CURRENT_A = 1.0e5
WINDING_COIL_CURRENT_A = 1.5e5
CURVE_ORDER = 2

WINDING_COIL_DOFS = (
    0.119251,
    0.012469,
    -0.017700,
    -0.068677,
    -0.014250,
    0.430936,
    0.082708,
    -0.015905,
    -0.113852,
    -0.025142,
)

LBFGS_GTOL = 1.0e-3
LBFGS_FTOL = 0.0


class _FixedSurfaceBoozerState(Optimizable):
    """Minimal Boozer-surface stand-in for the fixed-surface production terms."""

    def __init__(
        self,
        surface: SurfaceXYZTensorFourier,
        biot_savart: BiotSavart,
        label: Optimizable,
        policy: Flat675ObjectivePolicy,
    ) -> None:
        self.surface = surface
        self.biotsavart = biot_savart
        self.label = label
        self.constraint_weight = policy.boozer_constraint_weight
        self.targetlabel = policy.boozer_target_label
        super().__init__(
            x0=np.empty((0,), dtype=np.float64),
            depends_on=[surface, biot_savart],
        )


class _NativeTerm:
    __slots__ = ("value", "derivative")

    def __init__(self, value: float, derivative: Derivative) -> None:
        self.value = value
        self.derivative = derivative


def _host_float64(value: object) -> np.ndarray:
    return np.array(value, dtype=np.float64, copy=True)


def _xyz_surface_from_spec(spec) -> SurfaceXYZTensorFourier:
    surface = SurfaceXYZTensorFourier(
        mpol=int(spec.mpol),
        ntor=int(spec.ntor),
        nfp=int(spec.nfp),
        stellsym=bool(spec.stellsym),
        quadpoints_phi=_host_float64(spec.quadpoints_phi),
        quadpoints_theta=_host_float64(spec.quadpoints_theta),
    )
    surface.set_dofs(_host_float64(spec.dofs))
    return surface


def _rz_surface_from_spec(spec) -> SurfaceRZFourier:
    vessel = SurfaceRZFourier(
        nfp=int(spec.nfp),
        stellsym=bool(spec.stellsym),
        mpol=int(spec.mpol),
        ntor=int(spec.ntor),
        quadpoints_phi=_host_float64(spec.quadpoints_phi),
        quadpoints_theta=_host_float64(spec.quadpoints_theta),
    )
    vessel.set_dofs(_host_float64(surface_rz_fourier_dofs_from_spec(spec)))
    return vessel


def _certified_plasma_surface(
    *,
    nfp: int,
    surface_coordinates: np.ndarray,
) -> SurfaceXYZTensorFourier:
    quadpoints_phi, quadpoints_theta = Surface.get_quadpoints(
        nphi=CERTIFIED_NPHI,
        ntheta=CERTIFIED_NTHETA,
        nfp=int(nfp),
        range=surface_quadrature_range(stellsym=CERTIFIED_STELLSYM),
    )
    surface = SurfaceXYZTensorFourier(
        mpol=CERTIFIED_MPOL,
        ntor=CERTIFIED_NTOR,
        nfp=int(nfp),
        stellsym=CERTIFIED_STELLSYM,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
    )
    surface.set_dofs(_host_float64(surface_coordinates))
    return surface


def _optimized_coil_index(coils: list) -> int:
    for index, coil in enumerate(coils):
        if coil.curve.x.size > 0 or coil.current.x.size > 0:
            return index
    raise Flat675ContractError(
        "the coil set declares no free coil, so no coil can carry the length "
        "and curvature penalties."
    )


def _boozer_design_matrix(
    magnetic_field: np.ndarray,
    toroidal_tangent: np.ndarray,
    poloidal_tangent: np.ndarray,
    *,
    weight_by_inverse_field_magnitude: bool,
) -> tuple[np.ndarray, np.ndarray]:
    field_squared = np.sum(magnetic_field * magnetic_field, axis=-1)
    if weight_by_inverse_field_magnitude:
        point_weight = 1.0 / np.sqrt(field_squared)
    else:
        point_weight = np.ones_like(field_squared)
    weighted_field_squared = point_weight * field_squared
    iota_column = -(weighted_field_squared[..., None] * poloidal_tangent).reshape(-1)
    current_column = (point_weight[..., None] * magnetic_field).reshape(-1)
    right_hand_side = (weighted_field_squared[..., None] * toroidal_tangent).reshape(-1)
    normalization = np.sqrt(np.float64(magnetic_field.size))
    design_matrix = np.stack((iota_column, current_column), axis=1) / normalization
    return design_matrix, right_hand_side / normalization


def _solve_y(design_matrix: np.ndarray, right_hand_side: np.ndarray) -> np.ndarray:
    orthogonal, triangular = np.linalg.qr(design_matrix, mode="reduced")
    return np.linalg.solve(triangular, orthogonal.T @ right_hand_side)


def _boozer_linear_residual_field_partial(
    magnetic_field: np.ndarray,
    tangent: np.ndarray,
    G: float,
    *,
    weight_inv_modB: bool,
) -> tuple[np.ndarray, np.ndarray]:
    field_squared = np.sum(magnetic_field * magnetic_field, axis=2)
    residual = G * magnetic_field - field_squared[..., None] * tangent
    residual_by_field = (
        G * np.eye(3, dtype=np.float64)[None, None, :, :]
        - 2.0 * tangent[:, :, :, None] * magnetic_field[:, :, None, :]
    )
    if not weight_inv_modB:
        return residual, residual_by_field
    field_magnitude = np.sqrt(field_squared)
    inverse_field_magnitude = 1.0 / field_magnitude
    inverse_field_magnitude_by_field = (
        -magnetic_field / field_squared[:, :, None] ** 1.5
    )
    weighted_residual = inverse_field_magnitude[:, :, None] * residual
    weighted_residual_by_field = (
        residual[:, :, :, None] * inverse_field_magnitude_by_field[:, :, None, :]
        + residual_by_field * inverse_field_magnitude[:, :, None, None]
    )
    return weighted_residual, weighted_residual_by_field


def _boozer_linear_residual_surface_partial(
    magnetic_field: np.ndarray,
    magnetic_field_by_surface: np.ndarray,
    tangent: np.ndarray,
    tangent_by_surface: np.ndarray,
    G: float,
    *,
    weight_inv_modB: bool,
) -> tuple[np.ndarray, np.ndarray]:
    field_squared = np.sum(magnetic_field * magnetic_field, axis=2)
    field_squared_by_surface = 2.0 * np.einsum(
        "ijk,ijkl->ijl",
        magnetic_field,
        magnetic_field_by_surface,
        optimize=True,
    )
    residual = G * magnetic_field - field_squared[..., None] * tangent
    residual_by_surface = (
        G * magnetic_field_by_surface
        - field_squared_by_surface[:, :, None, :] * tangent[..., None]
        - field_squared[:, :, None, None] * tangent_by_surface
    )
    if not weight_inv_modB:
        return residual, residual_by_surface
    field_magnitude = np.sqrt(field_squared)
    inverse_field_magnitude = 1.0 / field_magnitude
    field_magnitude_by_surface = (
        0.5 * field_squared_by_surface / field_magnitude[:, :, None]
    )
    inverse_field_magnitude_by_surface = (
        -field_magnitude_by_surface / field_squared[:, :, None]
    )
    weighted_residual = inverse_field_magnitude[:, :, None] * residual
    weighted_residual_by_surface = (
        residual[..., None] * inverse_field_magnitude_by_surface[:, :, None, :]
        + inverse_field_magnitude[:, :, None, None] * residual_by_surface
    )
    return weighted_residual, weighted_residual_by_surface


def _y_stationarity_outer_vjp(
    surface: SurfaceXYZTensorFourier,
    iota: float,
    G: float,
    biot_savart: BiotSavart,
    y_cotangent: np.ndarray,
    *,
    weight_inv_modB: bool,
) -> Derivative:
    cotangent = _host_float64(y_cotangent)
    surface_gamma = _host_float64(surface.gamma())
    toroidal_tangent = _host_float64(surface.gammadash1())
    poloidal_tangent = _host_float64(surface.gammadash2())
    surface_gamma_by_dofs = _host_float64(surface.dgamma_by_dcoeff())
    toroidal_tangent_by_dofs = _host_float64(surface.dgammadash1_by_dcoeff())
    poloidal_tangent_by_dofs = _host_float64(surface.dgammadash2_by_dcoeff())
    biot_savart.set_points(surface_gamma.reshape((-1, 3)))
    magnetic_field = _host_float64(biot_savart.B()).reshape(surface_gamma.shape)
    magnetic_field_by_position = _host_float64(biot_savart.dB_by_dX()).reshape(
        surface_gamma.shape + (3,)
    )
    magnetic_field_by_surface = np.einsum(
        "ijkl,ijkm->ijlm",
        magnetic_field_by_position,
        surface_gamma_by_dofs,
        optimize=True,
    )
    solved_tangent = toroidal_tangent + iota * poloidal_tangent
    solved_tangent_by_surface = (
        toroidal_tangent_by_dofs + iota * poloidal_tangent_by_dofs
    )
    solved_residual, solved_residual_by_surface = (
        _boozer_linear_residual_surface_partial(
            magnetic_field,
            magnetic_field_by_surface,
            solved_tangent,
            solved_tangent_by_surface,
            G,
            weight_inv_modB=weight_inv_modB,
        )
    )
    cotangent_tangent = cotangent[0] * poloidal_tangent
    cotangent_tangent_by_surface = cotangent[0] * poloidal_tangent_by_dofs
    cotangent_residual, cotangent_residual_by_surface = (
        _boozer_linear_residual_surface_partial(
            magnetic_field,
            magnetic_field_by_surface,
            cotangent_tangent,
            cotangent_tangent_by_surface,
            float(cotangent[1]),
            weight_inv_modB=weight_inv_modB,
        )
    )
    residual_component_count = np.float64(magnetic_field.size)
    surface_partial = (
        np.einsum(
            "ijk,ijkl->l",
            solved_residual,
            cotangent_residual_by_surface,
            optimize=True,
        )
        + np.einsum(
            "ijk,ijkl->l",
            cotangent_residual,
            solved_residual_by_surface,
            optimize=True,
        )
    ) / residual_component_count
    solved_residual_field, solved_residual_by_field = (
        _boozer_linear_residual_field_partial(
            magnetic_field,
            solved_tangent,
            G,
            weight_inv_modB=weight_inv_modB,
        )
    )
    cotangent_residual_field, cotangent_residual_by_field = (
        _boozer_linear_residual_field_partial(
            magnetic_field,
            cotangent_tangent,
            float(cotangent[1]),
            weight_inv_modB=weight_inv_modB,
        )
    )
    field_partial = (
        np.einsum(
            "ijk,ijkl->ijl",
            solved_residual_field,
            cotangent_residual_by_field,
            optimize=True,
        )
        + np.einsum(
            "ijk,ijkl->ijl",
            cotangent_residual_field,
            solved_residual_by_field,
            optimize=True,
        )
    ) / residual_component_count
    return biot_savart.B_vjp(field_partial.reshape((-1, 3))) + Derivative(
        {surface: surface_partial}
    )


def _implicit_y_derivative(
    design_matrix: np.ndarray,
    surface: SurfaceXYZTensorFourier,
    iota: float,
    G: float,
    biot_savart: BiotSavart,
    y_partial: np.ndarray,
    *,
    weight_inv_modB: bool,
) -> Derivative:
    normal_operator = design_matrix.T @ design_matrix
    adjoint = np.linalg.solve(normal_operator.T, y_partial)
    return -1.0 * _y_stationarity_outer_vjp(
        surface,
        iota,
        G,
        biot_savart,
        adjoint,
        weight_inv_modB=weight_inv_modB,
    )


def _quadratic_penalty(
    raw: _NativeTerm, target: float, *, upper_only: bool
) -> _NativeTerm:
    delta = raw.value - target
    active_delta = max(delta, 0.0) if upper_only else delta
    return _NativeTerm(
        value=0.5 * active_delta**2,
        derivative=active_delta * raw.derivative,
    )


def _weighted_term(weight: float, raw: _NativeTerm) -> _NativeTerm:
    if weight == 0.0:
        return _NativeTerm(value=0.0, derivative=Derivative())
    return _NativeTerm(value=weight * raw.value, derivative=weight * raw.derivative)


class NativeFlat675Problem:
    """Native Biot-Savart evaluation of the certified 11+3+661 objective."""

    def __init__(
        self,
        *,
        field: BiotSavart,
        surface: SurfaceXYZTensorFourier,
        vessel: SurfaceRZFourier,
        policy: Flat675ObjectivePolicy,
        weight_by_inverse_field_magnitude: bool,
    ) -> None:
        self.field = field
        self.surface = surface
        self.vessel = vessel
        self.policy = policy
        self.weight_by_inverse_field_magnitude = weight_by_inverse_field_magnitude
        coils = list(field.coils)
        banana = coils[policy.optimized_coil_index].curve
        curves = [coil.curve for coil in coils]
        label = Volume(surface)
        fixed_state = _FixedSurfaceBoozerState(surface, field, label, policy)
        self._non_qs = NonQuasiSymmetricRatio(
            fixed_state,
            field,
            sDIM=policy.non_qs_grid_size // 2,
        )
        self._residual = BoozerResidual(fixed_state, field)
        self._length = CurveLength(banana)
        self._curve_curve = CurveCurveDistance(curves, policy.curve_curve_threshold_m)
        self._curve_surface = CurveSurfaceDistance(
            curves, surface, policy.curve_surface_threshold_m
        )
        self._surface_vessel = SurfaceSurfaceDistance(
            surface, vessel, policy.surface_vessel_threshold_m
        )
        self._curvature = LpCurveCurvature(
            banana,
            policy.hardware_soft_penalty_policy.penalty_exponent,
            policy.curvature_threshold_inverse_m,
        )
        packed = self.pack()
        if packed.shape != (FLAT675_OUTER_DOF_COUNT,):
            raise Flat675ContractError(
                "native flat-675 state must have length "
                f"{FLAT675_OUTER_DOF_COUNT}; got {packed.shape}."
            )

    def pack(self) -> np.ndarray:
        return np.concatenate(
            (
                _host_float64(self.field.x),
                _host_float64(self.vessel.x),
                _host_float64(self.surface.x),
            )
        )

    def unpack(self, coordinates: np.ndarray) -> None:
        vector = _host_float64(coordinates)
        coil_count = int(self.field.x.size)
        vessel_count = int(self.vessel.x.size)
        self.field.x = np.array(vector[:coil_count], dtype=np.float64, copy=True)
        self.vessel.x = np.array(
            vector[coil_count : coil_count + vessel_count], dtype=np.float64, copy=True
        )
        self.surface.x = np.array(
            vector[coil_count + vessel_count :], dtype=np.float64, copy=True
        )

    def _design_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        gamma = _host_float64(self.surface.gamma())
        self.field.set_points(gamma.reshape((-1, 3)))
        magnetic_field = _host_float64(self.field.B()).reshape(gamma.shape)
        return _boozer_design_matrix(
            magnetic_field,
            _host_float64(self.surface.gammadash1()),
            _host_float64(self.surface.gammadash2()),
            weight_by_inverse_field_magnitude=self.weight_by_inverse_field_magnitude,
        )

    def evaluate_terms(self) -> tuple[_NativeTerm, ...]:
        design_matrix, right_hand_side = self._design_matrix()
        iota, G = (float(value) for value in _solve_y(design_matrix, right_hand_side))
        weight_inv = self.weight_by_inverse_field_magnitude
        policy = self.policy

        non_qs_value, non_qs_derivative = (
            self._non_qs.fixed_surface_value_and_derivative()
        )
        residual_value, residual_direct, y_partial = (
            self._residual.fixed_surface_value_derivative_and_y_partial(
                iota, G, weight_inv_modB=weight_inv
            )
        )
        residual_term = _NativeTerm(
            value=residual_value,
            derivative=residual_direct
            + _implicit_y_derivative(
                design_matrix,
                self.surface,
                iota,
                G,
                self.field,
                _host_float64(y_partial),
                weight_inv_modB=weight_inv,
            ),
        )
        iota_term = _quadratic_penalty(
            _NativeTerm(
                value=iota,
                derivative=_implicit_y_derivative(
                    design_matrix,
                    self.surface,
                    iota,
                    G,
                    self.field,
                    np.asarray((1.0, 0.0), dtype=np.float64),
                    weight_inv_modB=weight_inv,
                ),
            ),
            policy.iota_target,
            upper_only=False,
        )
        length_term = _quadratic_penalty(
            _NativeTerm(
                value=float(self._length.J()),
                derivative=self._length.dJ(partials=True),
            ),
            policy.length_target_m,
            upper_only=True,
        )
        raw_terms = (
            _NativeTerm(value=non_qs_value, derivative=non_qs_derivative),
            residual_term,
            iota_term,
            length_term,
            _NativeTerm(
                value=float(self._curve_curve.J()),
                derivative=self._curve_curve.dJ(partials=True),
            ),
            _NativeTerm(
                value=float(self._curve_surface.J()),
                derivative=self._curve_surface.dJ(partials=True),
            ),
            _NativeTerm(
                value=float(self._surface_vessel.J()),
                derivative=self._surface_vessel.dJ(partials=True),
            ),
            _NativeTerm(
                value=float(self._curvature.J()),
                derivative=self._curvature.dJ(partials=True),
            ),
        )
        weights = (
            policy.non_qs_weight,
            policy.residual_weight,
            policy.iota_weight,
            policy.length_weight,
            policy.curve_curve_weight,
            policy.curve_surface_weight,
            policy.surface_vessel_weight,
            policy.curvature_weight,
        )
        return tuple(
            _weighted_term(weight, raw)
            for weight, raw in zip(weights, raw_terms, strict=True)
        )

    def value_and_gradient(self, coordinates: np.ndarray) -> tuple[float, np.ndarray]:
        self.unpack(coordinates)
        terms = self.evaluate_terms()
        total = 0.0
        derivative = Derivative()
        for term in terms:
            total = total + term.value
            derivative = derivative + term.derivative
        gradient = np.concatenate(
            (
                _host_float64(derivative(self.field)),
                _host_float64(derivative(self.vessel)),
                _host_float64(derivative(self.surface)),
            )
        )
        return float(total), gradient

    def weighted_term_values(self) -> dict[str, float]:
        terms = self.evaluate_terms()
        return {
            name: float(term.value)
            for name, term in zip(FLAT675_OBJECTIVE_TERM_KEYS, terms)
        }


def _repository_problem(native_scale: bool) -> NativeFlat675Problem:
    grid = NATIVE_DEFAULT_GRID if native_scale else BOUNDED_GRID
    quadpoints = (
        NATIVE_DEFAULT_CURVE_QUADPOINTS if native_scale else BOUNDED_CURVE_QUADPOINTS
    )
    boundary = SurfaceRZFourier.from_vmec_input(
        str(BOUNDARY_INPUT), range="half period", nphi=grid, ntheta=grid
    )
    points = _host_float64(boundary.gamma()).reshape((-1, 3))
    radius = np.hypot(points[:, 0], points[:, 1])
    major_radius = 0.5 * (float(radius.max()) + float(radius.min()))
    minor_radius = float(np.max(np.hypot(radius - major_radius, points[:, 2])))

    winding_surface = SurfaceRZFourier(
        nfp=boundary.nfp,
        stellsym=True,
        mpol=1,
        ntor=0,
        quadpoints_phi=np.linspace(0.0, 1.0, 16, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 16, endpoint=False),
    )
    winding_surface.set_rc(0, 0, major_radius)
    winding_surface.set_rc(1, 0, minor_radius * WINDING_SURFACE_FACTOR)
    winding_surface.set_zs(1, 0, minor_radius * WINDING_SURFACE_FACTOR)
    winding_surface.fix_all()

    base_curve = CurveCWSFourier(
        quadpoints=quadpoints, order=CURVE_ORDER, surf=winding_surface
    )
    base_curve.x = np.asarray(WINDING_COIL_DOFS, dtype=np.float64)
    winding_coils = coils_via_symmetries(
        [base_curve], [Current(WINDING_COIL_CURRENT_A)], boundary.nfp, True
    )

    tf_curves = create_equally_spaced_curves(
        TF_BASE_COIL_COUNT,
        boundary.nfp,
        stellsym=True,
        R0=major_radius,
        R1=minor_radius * TF_COIL_RADIUS_FACTOR,
        order=CURVE_ORDER,
        numquadpoints=32,
        use_jax_curve=False,
    )
    tf_currents = []
    for curve in tf_curves:
        curve.fix_all()
        current = Current(TF_COIL_CURRENT_A)
        current.fix_all()
        tf_currents.append(current)
    tf_coils = coils_via_symmetries(tf_curves, tf_currents, boundary.nfp, True)
    field = BiotSavart(list(tf_coils) + list(winding_coils))

    surface_spec = fit_flat675_boundary(boundary, nphi=grid, ntheta=grid)
    surface = _xyz_surface_from_spec(surface_spec)
    policy = default_flat675_objective_policy(
        optimized_coil_index=_optimized_coil_index(list(field.coils))
    )
    vessel = _rz_surface_from_spec(
        synthesize_flat675_vessel(
            surface_spec,
            hinge_threshold_m=policy.surface_vessel_threshold_m,
        )
    )
    return NativeFlat675Problem(
        field=field,
        surface=surface,
        vessel=vessel,
        policy=policy,
        weight_by_inverse_field_magnitude=(
            DEFAULT_FLAT675_BOOZER_POLICY.weight_by_inverse_field_magnitude
        ),
    )


def _bundle_problem() -> NativeFlat675Problem:
    if not BUNDLE_ROOT.is_dir():
        raise Flat675ContractError(
            f"{BUNDLE_FLAG} runs the certified frozen-bundle configuration, "
            f"whose input bundle is host-local and was not found at "
            f"{BUNDLE_ROOT}. Run without {BUNDLE_FLAG} to use the repository "
            "geometry this example ships with."
        )
    manifest = load_flat675_input_manifest(BUNDLE_ROOT)
    field = load(str(BUNDLE_ROOT / NATIVE_BIOT_SAVART_FILENAME))
    if not isinstance(field, BiotSavart):
        raise Flat675ContractError(
            f"{NATIVE_BIOT_SAVART_FILENAME} must deserialize to BiotSavart."
        )
    vessel_spec = load_flat675_vessel_template(BUNDLE_ROOT)
    vessel = _rz_surface_from_spec(vessel_spec)
    surface = _certified_plasma_surface(
        nfp=int(vessel_spec.nfp),
        surface_coordinates=np.asarray(
            manifest.candidate.surface_coordinates, dtype=np.float64
        ),
    )
    field.x = np.asarray(manifest.candidate.coil_coordinates, dtype=np.float64)
    vessel.x = np.asarray(manifest.candidate.vessel_coordinates, dtype=np.float64)
    return NativeFlat675Problem(
        field=field,
        surface=surface,
        vessel=vessel,
        policy=manifest.objective_policy,
        weight_by_inverse_field_magnitude=(
            manifest.boozer_construction_policy.weight_by_inverse_field_magnitude
        ),
    )


def _result_payload(
    *,
    scale: str,
    configuration: str,
    max_steps: int,
    optimizer_result,
    final_objective: float,
    finite: bool,
    initial_weighted_terms: dict[str, float],
    outer_dof_count: int,
) -> dict[str, object]:
    status = "ok" if finite and outer_dof_count == FLAT675_OUTER_DOF_COUNT else "failed"
    return {
        "example_id": EXAMPLE_ID,
        "backend_mode": "native_cpu",
        "platform": "cpu",
        "precision": "fp64",
        "scale": scale,
        "status": status,
        "observables": {
            "scale": scale,
            "configuration": configuration,
            "formulation": "flat-coupled-single-stage",
            "outer_dof_count": outer_dof_count,
            "lbfgs_history": FLAT675_LBFGS_HISTORY,
            "lbfgs_max_line_search_steps": FLAT675_LBFGS_MAXLS,
            "max_steps": max_steps,
            "iterations_run": int(optimizer_result.nit),
            "objective_evaluations": int(optimizer_result.nfev),
            "final_objective": final_objective,
            "endpoint_finite": finite,
            "host_step_transfers": int(optimizer_result.nit),
            "host_callback_transfers": 0,
            "host_unclassified_transfers": 0,
            "host_endpoint_transfers": 1,
            "initial_weighted_terms": initial_weighted_terms,
        },
    }


def _solve(
    problem: NativeFlat675Problem,
    *,
    max_steps: int,
    scale: str,
    configuration: str,
) -> dict[str, object]:
    initial = problem.pack()
    initial_weighted_terms = problem.weighted_term_values()
    optimizer_result = minimize(
        problem.value_and_gradient,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": max_steps,
            "maxcor": FLAT675_LBFGS_HISTORY,
            "maxls": FLAT675_LBFGS_MAXLS,
            "ftol": LBFGS_FTOL,
            "gtol": LBFGS_GTOL,
        },
    )
    problem.unpack(np.asarray(optimizer_result.x, dtype=np.float64))
    final_objective, _gradient = problem.value_and_gradient(problem.pack())
    solution = np.asarray(optimizer_result.x, dtype=np.float64)
    finite = bool(np.all(np.isfinite(solution)) and np.isfinite(final_objective))
    return _result_payload(
        scale=scale,
        configuration=configuration,
        max_steps=max_steps,
        optimizer_result=optimizer_result,
        final_objective=final_objective,
        finite=finite,
        initial_weighted_terms=initial_weighted_terms,
        outer_dof_count=int(solution.shape[0]),
    )


def solve(_output_dir: Path, max_steps: int, scale: str) -> dict[str, object]:
    return _solve(
        _repository_problem(scale == "native_default"),
        max_steps=max_steps,
        scale=scale,
        configuration="repository-geometry",
    )


def solve_bundle(_output_dir: Path, max_steps: int, scale: str) -> dict[str, object]:
    return _solve(
        _bundle_problem(),
        max_steps=max_steps,
        scale=scale,
        configuration="certified-frozen-bundle",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(BUNDLE_FLAG, action="store_true")
    return parser


def main(arguments: list[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    scale = "bounded" if options.smoke else "native_default"
    max_steps = options.max_steps or (
        BOUNDED_STEPS if options.smoke else NATIVE_DEFAULT_STEPS
    )
    selected = solve_bundle if options.bundle else solve
    if options.output_dir is not None:
        options.output_dir.mkdir(parents=True, exist_ok=True)
        result = selected(options.output_dir, max_steps, scale)
    elif options.smoke:
        with TemporaryDirectory(prefix="simsopt-native-flat675-single-stage-") as tmp:
            result = selected(Path(tmp), max_steps, scale)
    else:
        result = selected(Path.cwd(), max_steps, scale)
    if options.json:
        print(json.dumps(result, sort_keys=True))
    else:
        observables = result["observables"]
        print(f"example={result['example_id']}")
        print(f"status={result['status']}")
        for name, value in observables.items():
            print(f"{name}={value}")
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
