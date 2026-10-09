"""Float64 QFM kernels with the quadrature and singularities of native simsopt.

Public kernels snapshot caller-owned host arrays before dispatch and retain
device arrays. Points are explicit because native
QfmResidual and ToroidalFlux use their fields' current point buffers, including externally
changed points. Surface derivatives use the native position/normal VJP
formulation, rather than differentiating through the field's optimizer graph.
Float64 subnormal/overflow limits are those documented in surface_geometry;
XLA reductions and derivative arithmetic need not be bitwise native.
"""

from __future__ import annotations

from typing import Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np

from simsopt_jax.backend import register_backend_cache_clear
from simsopt_jax.pytree import pytree_dataclass
from simsopt_jax.runtime.host_boundary import snapshot_host_tree

from .field import (
    grouped_biot_savart_A_from_spec,
    grouped_biot_savart_B_and_dB_from_spec,
    grouped_biot_savart_B_from_spec,
    grouped_biot_savart_dA_by_dX_from_spec,
)
from .specs import GroupedCoilSetSpec, SurfaceSpec
from .surface_fourier_series import surface_get_dofs, surface_spec_with_dofs
from .surface_geometry import (
    surface_area,
    surface_gamma,
    surface_gammadash2,
    surface_normal,
    surface_volume,
)

__all__ = [
    "QfmSpec", "QfmLabelSpec", "qfm_residual", "qfm_residual_value_and_grad",
    "qfm_label", "qfm_label_constraint", "qfm_label_constraint_value_and_grad",
    "qfm_penalty_constraints", "qfm_penalty_constraints_value_and_grad",
]


@pytree_dataclass(data=("surface", "coils", "points"))
class QfmSpec:
    """Immutable QFM geometry and field evaluation operands.

    Args:
        surface (SurfaceSpec): Native Fourier coefficient snapshot and toroidal/poloidal grids in turns.
        coils (GroupedCoilSetSpec): Coil positions/tangents in meters and currents in amperes.
        points (jax.Array | numpy.ndarray): Cartesian field points in meters, shape (nphi * ntheta, 3). Host arrays are owned before evaluation."""

    surface: SurfaceSpec
    coils: GroupedCoilSetSpec
    points: jax.Array | np.ndarray


@pytree_dataclass(data=("surface", "coils", "idx", "points"), meta=("kind",))
class QfmLabelSpec:
    """Immutable volume, area or toroidal-flux label on its own grid.

    Args:
        surface (SurfaceSpec): Label surface sharing the optimized DOFs, with its own quadrature in turns.
        coils (GroupedCoilSetSpec | None): Flux field coils in meters/amperes; None for area and volume.
        idx (jax.Array): Scalar shape () int32 flux row; native negative indices are supported.
        kind (str): One of volume, area or toroidal_flux; part of the static pytree metadata.
        points (jax.Array | numpy.ndarray | None): Flux field points in meters, shape (ntheta, 3); None uses the surface row at idx."""

    surface: SurfaceSpec
    coils: GroupedCoilSetSpec | None
    idx: jax.Array
    kind: str
    points: jax.Array | np.ndarray | None = None


def _norm(vectors: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.sum(vectors * vectors, axis=-1))


def _residual(normal: jax.Array, field: jax.Array) -> jax.Array:
    norm_normal = _norm(normal)
    unit_normal = normal / norm_normal[..., None]
    field_normal = jnp.sum(field * unit_normal, axis=-1)
    norm_field = _norm(field)
    return jnp.sum(field_normal**2 * norm_normal) / jnp.sum(norm_field**2 * norm_normal)


def _residual_normal(spec: QfmSpec) -> jax.Array:
    """Reject incompatible field buffers with native NumPy's exception class."""
    normal = surface_normal(spec.surface)
    if spec.points.size != normal.size:
        raise ValueError(f"cannot reshape array of size {spec.points.size} into shape {normal.shape}")
    return normal


@jax.jit
def _jitted_qfm_residual(spec: QfmSpec) -> jax.Array:
    """Native QfmResidual.J; zero field/normal remains non-finite."""
    normal = _residual_normal(spec)
    field = cast(jax.Array, grouped_biot_savart_B_from_spec(spec.points, spec.coils)).reshape(normal.shape)
    return _residual(normal, field)


@jax.jit
def _jitted_qfm_residual_value_and_grad(spec: QfmSpec) -> tuple[jax.Array, jax.Array]:
    """Value and native full-coefficient gradient, including fixed DOFs."""
    normal = _residual_normal(spec)
    field, field_derivative = grouped_biot_savart_B_and_dB_from_spec(spec.points, spec.coils)
    field = field.reshape(normal.shape)
    field_derivative = field_derivative.reshape((*normal.shape, 3))
    norm_normal = _norm(normal)
    field_normal = jnp.sum(field * normal, axis=-1)
    numerator = jnp.sum(field_normal**2 / norm_normal)
    denominator = jnp.sum(field**2 * norm_normal[..., None])
    numerator_position = (2 * field_normal / norm_normal)[..., None] * jnp.sum(
        field_derivative * normal[..., None, :], axis=-1
    )
    numerator_normal = (
        (2 * field_normal / norm_normal)[..., None] * field
        - (field_normal**2 / norm_normal**3)[..., None] * normal
    )
    denominator_position = 2 * jnp.sum(
        field_derivative * field[..., None, :], axis=-1
    ) * norm_normal[..., None]
    denominator_normal = (jnp.sum(field * field, axis=-1) / norm_normal)[..., None] * normal

    def geometry(dofs: jax.Array) -> tuple[jax.Array, jax.Array]:
        surface = surface_spec_with_dofs(spec.surface, dofs)
        return surface_normal(surface), surface_gamma(surface)

    _, pullback = jax.vjp(geometry, surface_get_dofs(spec.surface))
    gradient, = pullback((
        numerator_normal / denominator - denominator_normal * numerator / denominator**2,
        numerator_position / denominator - denominator_position * numerator / denominator**2,
    ))
    return _residual(normal, field), gradient


@jax.jit
def _jitted_qfm_label(spec: QfmLabelSpec) -> jax.Array:
    """Native label on its own surface grid and current flux-field points."""
    if spec.kind == "volume":
        return surface_volume(spec.surface)
    if spec.kind == "area":
        return surface_area(spec.surface)
    # Construction admits flux only when its coils exist. Index bounds are
    # checked at the native adapter boundary (JAX gather would clamp them).
    assert spec.coils is not None
    points = surface_gamma(spec.surface)[spec.idx] if spec.points is None else spec.points
    tangent = surface_gammadash2(spec.surface)[spec.idx]
    potential = grouped_biot_savart_A_from_spec(points, spec.coils)
    return jnp.sum(potential * tangent) / tangent.shape[0]


def _label_value_and_grad(spec: QfmLabelSpec) -> tuple[jax.Array, jax.Array]:
    """Native label derivative, including spatial derivatives at flux points.

    Native holds the field buffer independently of the label's surface. Its
    spatial derivative is pulled back through the surface at the current idx,
    even when those points differ from the field buffer after copying/editing.
    """
    if spec.kind != "toroidal_flux":
        def label(dofs: jax.Array) -> jax.Array:
            moved = QfmLabelSpec(
                surface_spec_with_dofs(spec.surface, dofs), spec.coils, spec.idx, spec.kind, spec.points,
            )
            return _jitted_qfm_label(moved)

        return jax.value_and_grad(label)(surface_get_dofs(spec.surface))

    assert spec.coils is not None
    points = surface_gamma(spec.surface)[spec.idx] if spec.points is None else spec.points
    potential = cast(jax.Array, grouped_biot_savart_A_from_spec(points, spec.coils))
    derivative = cast(jax.Array, grouped_biot_savart_dA_by_dX_from_spec(points, spec.coils))

    def geometry(dofs: jax.Array) -> tuple[jax.Array, jax.Array]:
        surface = surface_spec_with_dofs(spec.surface, dofs)
        return surface_gamma(surface)[spec.idx], surface_gammadash2(surface)[spec.idx]

    (_, tangent), pullback = jax.vjp(geometry, surface_get_dofs(spec.surface))
    ntheta = tangent.shape[0]
    gradient, = pullback((
        jnp.sum(derivative * tangent[:, None, :], axis=-1) / ntheta,
        jnp.broadcast_to(potential, tangent.shape) / ntheta,
    ))
    return jnp.sum(potential * tangent) / ntheta, gradient


@jax.jit
def _jitted_qfm_label_constraint(
    spec: QfmLabelSpec, targetlabel: jax.Array | np.ndarray, label_value: jax.Array | None = None,
) -> jax.Array:
    """Squared label error; optional host value preserves native rounding.

    Native SLSQP's zero-Jacobian failure depends on exact label subtraction.
    The adapter supplies that value explicitly; standalone kernels compute it.
    """
    value = _jitted_qfm_label(spec) if label_value is None else label_value
    return 0.5 * (value - targetlabel)**2


@jax.jit
def _jitted_qfm_label_constraint_value_and_grad(
    spec: QfmLabelSpec, targetlabel: jax.Array | np.ndarray, label_value: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    value, gradient = _label_value_and_grad(spec)
    value = value if label_value is None else label_value
    residual = value - targetlabel
    return 0.5 * residual**2, residual * gradient


@jax.jit
def _jitted_qfm_penalty_constraints(
    spec: QfmSpec, label: QfmLabelSpec, targetlabel: jax.Array | np.ndarray, constraint_weight: jax.Array | np.ndarray,
    label_value: jax.Array | None = None,
) -> jax.Array:
    return _jitted_qfm_residual(spec) + constraint_weight * _jitted_qfm_label_constraint(label, targetlabel, label_value)


@jax.jit
def _jitted_qfm_penalty_constraints_value_and_grad(
    spec: QfmSpec, label: QfmLabelSpec, targetlabel: jax.Array | np.ndarray, constraint_weight: jax.Array | np.ndarray,
    label_value: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    value, gradient = _jitted_qfm_residual_value_and_grad(spec)
    constraint, label_gradient = _jitted_qfm_label_constraint_value_and_grad(label, targetlabel, label_value)
    return value + constraint_weight * constraint, gradient + constraint_weight * label_gradient


def qfm_residual(spec: QfmSpec) -> jax.Array:
    """Evaluate the native quadratic-flux ratio without changing field points.

    Args:
        spec (QfmSpec): Surface geometry and coil snapshot with shape (nphi * ntheta, 3) Cartesian field points in meters. Host point buffers are copied before dispatch.

    Returns:
        jax.Array: Dimensionless scalar shape () ratio; zero fields or normals remain nonfinite."""
    inputs = snapshot_host_tree((spec,))
    return _jitted_qfm_residual(*inputs)


def qfm_residual_value_and_grad(spec: QfmSpec) -> tuple[jax.Array, jax.Array]:
    """Evaluate the native quadratic-flux ratio without changing field points.

    Args:
        spec (QfmSpec): Surface geometry and coil snapshot with shape (nphi * ntheta, 3) Cartesian field points in meters. Host point buffers are copied before dispatch.

    Returns:
        tuple[jax.Array, jax.Array]: Dimensionless value shape () and full surface-coefficient gradient shape (ndofs,), in inverse meters, including fixed DOFs."""
    inputs = snapshot_host_tree((spec,))
    return _jitted_qfm_residual_value_and_grad(*inputs)


def qfm_label(spec: QfmLabelSpec) -> jax.Array:
    """Evaluate the native label on its own surface grid and current flux points.

    Args:
        spec (QfmLabelSpec): Independent native label grid, flux-coil state and optional Cartesian field points in meters.

    Returns:
        jax.Array: Scalar shape () volume in cubic meters, area in square meters, or toroidal flux in webers."""
    inputs = snapshot_host_tree((spec,))
    return _jitted_qfm_label(*inputs)


def qfm_label_constraint(
    spec: QfmLabelSpec, targetlabel: jax.Array | np.ndarray, label_value: jax.Array | None = None,
) -> jax.Array:
    """Evaluate half the squared native label error with owned host operands.

    Args:
        spec (QfmLabelSpec): Independent native label grid, flux-coil state and optional Cartesian field points in meters.
        targetlabel (jax.Array | numpy.ndarray): Scalar shape () targetlabel in cubic meters for volume, square meters for area, or webers for flux.
        label_value (jax.Array | None): Optional scalar shape () native label value in the label units, preserving exact host subtraction; default None evaluates the immutable label.

    Returns:
        jax.Array: Scalar shape () squared error in squared label units."""
    inputs = snapshot_host_tree((spec, targetlabel, label_value))
    return _jitted_qfm_label_constraint(*inputs)


def qfm_label_constraint_value_and_grad(
    spec: QfmLabelSpec, targetlabel: jax.Array | np.ndarray, label_value: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate half the squared native label error with owned host operands.

    Args:
        spec (QfmLabelSpec): Independent native label grid, flux-coil state and optional Cartesian field points in meters.
        targetlabel (jax.Array | numpy.ndarray): Scalar shape () targetlabel in cubic meters for volume, square meters for area, or webers for flux.
        label_value (jax.Array | None): Optional scalar shape () native label value in the label units, preserving exact host subtraction; default None evaluates the immutable label.

    Returns:
        tuple[jax.Array, jax.Array]: Scalar squared error shape () and full coefficient gradient shape (ndofs,), in squared label units per meter."""
    inputs = snapshot_host_tree((spec, targetlabel, label_value))
    return _jitted_qfm_label_constraint_value_and_grad(*inputs)


def qfm_penalty_constraints(
    spec: QfmSpec, label: QfmLabelSpec, targetlabel: jax.Array | np.ndarray, constraint_weight: jax.Array | np.ndarray,
    label_value: jax.Array | None = None,
) -> jax.Array:
    """Evaluate the native QFM ratio plus the weighted squared label error.

    Args:
        spec (QfmSpec): Surface geometry and coil snapshot with shape (nphi * ntheta, 3) Cartesian field points in meters. Host point buffers are copied before dispatch.
        label (QfmLabelSpec): Independent native label grid, flux-coil state and optional Cartesian field points in meters.
        targetlabel (jax.Array | numpy.ndarray): Scalar shape () targetlabel in cubic meters for volume, square meters for area, or webers for flux.
        constraint_weight (jax.Array | numpy.ndarray): Scalar shape () coefficient multiplying the squared label error, in the native constraint units.
        label_value (jax.Array | None): Optional scalar shape () native label value in the label units, preserving exact host subtraction; default None evaluates the immutable label.

    Returns:
        jax.Array: Scalar shape () native scalarized objective."""
    inputs = snapshot_host_tree((spec, label, targetlabel, constraint_weight, label_value))
    return _jitted_qfm_penalty_constraints(*inputs)


def qfm_penalty_constraints_value_and_grad(
    spec: QfmSpec, label: QfmLabelSpec, targetlabel: jax.Array | np.ndarray, constraint_weight: jax.Array | np.ndarray,
    label_value: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate the native QFM ratio plus the weighted squared label error.

    Args:
        spec (QfmSpec): Surface geometry and coil snapshot with shape (nphi * ntheta, 3) Cartesian field points in meters. Host point buffers are copied before dispatch.
        label (QfmLabelSpec): Independent native label grid, flux-coil state and optional Cartesian field points in meters.
        targetlabel (jax.Array | numpy.ndarray): Scalar shape () targetlabel in cubic meters for volume, square meters for area, or webers for flux.
        constraint_weight (jax.Array | numpy.ndarray): Scalar shape () coefficient multiplying the squared label error, in the native constraint units.
        label_value (jax.Array | None): Optional scalar shape () native label value in the label units, preserving exact host subtraction; default None evaluates the immutable label.

    Returns:
        tuple[jax.Array, jax.Array]: Objective shape () and full surface-coefficient gradient shape (ndofs,), including fixed DOFs; units follow the native scalarization."""
    inputs = snapshot_host_tree((spec, label, targetlabel, constraint_weight, label_value))
    return _jitted_qfm_penalty_constraints_value_and_grad(*inputs)


class _CompiledFunctionCache(Protocol):
    def clear_cache(self) -> None: ...


def _clear_compiled_qfm() -> None:
    # Field kernel tuning is read while tracing: invalidate enclosing QFM
    # programs too when the backend changes its settings.
    for function in (
        _jitted_qfm_residual, _jitted_qfm_residual_value_and_grad, _jitted_qfm_label,
        _jitted_qfm_label_constraint, _jitted_qfm_label_constraint_value_and_grad,
        _jitted_qfm_penalty_constraints, _jitted_qfm_penalty_constraints_value_and_grad,
    ):
        cast(_CompiledFunctionCache, function).clear_cache()


register_backend_cache_clear(_clear_compiled_qfm)
