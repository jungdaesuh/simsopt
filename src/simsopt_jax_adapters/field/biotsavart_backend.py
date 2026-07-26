"""JAX-backed Biot-Savart adapter and coil-tree helpers.

``BiotSavartJAX`` participates in the ``Optimizable`` dependency graph
through its coil list while computing the magnetic field via the pure
JAX kernels in :mod:`simsopt_jax.field.biotsavart`.

This module does **not** inherit from ``sopp.BiotSavart`` or
``sopp.MagneticField`` — it is a parallel JAX-native class per the
M0 rewrite contract (adapter pattern, §5).
"""

from dataclasses import dataclass, fields, is_dataclass
from functools import partial
import time

import jax
import jax.numpy as jnp
import numpy as np

from simsopt._core.derivative import Derivative
from simsopt.field.coil import Current
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt_jax.runtime.host_boundary import block_until_ready, host_array, host_float
from simsopt._core.optimizable import Optimizable
from simsopt_jax.backend import get_field_kernel_tuning
from simsopt_jax.core.state_tokens import make_state_token_factory
from simsopt_jax.backend.dtypes import (
    explicit_device_array,
    runtime_device_put,
    runtime_device_put_tree,
)
from simsopt_jax.core import (
    coil_set_spec_from_dof_extraction_spec,
    coil_specs_from_dof_extraction_spec,
    curve_gamma_and_dash_from_dofs as curve_gamma_and_dash_from_spec_dofs,
    curve_gamma_and_dash_from_spec,
    curve_geometry_from_spec,
    curve_pullback_from_dofs,
    curve_spec_kind,
    curve_spec_with_dofs,
    make_coil_dof_extraction_spec,
    make_biot_savart_spec,
    make_coil_set_dof_extraction_spec,
    make_optimizable_dof_map_spec,
)
from simsopt_jax.core.curve_xyz_fourier import jaxfouriercurve_geometry_pure
from simsopt_jax.core.biotsavart import (
    biot_savart_A,
    biot_savart_B,
    biot_savart_d2A_by_dXdX,
    biot_savart_d2B_by_dXdX,
    biot_savart_dA_by_dX,
    biot_savart_dB_by_dX,
)
from simsopt_jax.core.field import (
    biot_savart_B_vjp_maybe_collective,
    grouped_biot_savart_A_from_inputs,
    grouped_biot_savart_A_from_spec,
    grouped_biot_savart_B_and_dB_from_spec,
    grouped_biot_savart_B_from_spec,
    grouped_biot_savart_d2A_by_dXdX_from_spec,
    grouped_biot_savart_d2B_by_dXdX_from_spec,
    grouped_biot_savart_dA_by_dX_from_inputs,
    grouped_biot_savart_dA_by_dX_from_spec,
    grouped_biot_savart_dB_by_dX_from_inputs,
    grouped_biot_savart_dB_by_dX_from_spec,
    grouped_coil_set_spec_from_coil_specs,
    grouped_coil_set_spec_from_lists,
    grouped_field_data_from_spec,
    grouped_field_inputs_from_spec,
)
from simsopt_jax.core._math_utils import (
    as_compute_array as _as_compute_array,
    as_jax_float64 as _as_jax_float64,
)
from simsopt_jax.core.specs import (
    BiotSavartSpec,
    CoilSetDofExtractionSpec,
    CoilDofExtractionSpec,
    CoilSpec,
    CoilSymmetrySpec,
    CurveSpec,
    FieldEvalSpec,
    GroupedCoilSetSpec,
    SingleStageRuntimeSpec,
    make_field_eval_spec,
    make_grouped_coil_set_spec,
)
from simsopt_jax_adapters.field._coil_graph import (
    _unwrap_coil_curve_and_current_objects,
)
from simsopt_jax_adapters.geo.curve_specs import (
    adapter_curve_dof_mode,
    curve_spec_from_adapter_curve,
    supports_adapter_curve_spec,
)

_new_coil_dof_state_token = make_state_token_factory()
_SPEC_CACHE_KEY_SCALAR_TYPES = (str, int, float, bool, type(None))


def _device_zero_like(value: object) -> jax.Array:
    array = _as_jax_float64(value)
    return array - array


def _spec_cache_key(value: object) -> object:
    if isinstance(value, _SPEC_CACHE_KEY_SCALAR_TYPES):
        return value
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        return (
            "numpy_array",
            str(array.dtype),
            tuple(int(axis) for axis in array.shape),
            array.tobytes(),
        )
    if isinstance(value, jax.Array):
        return (
            "jax_array",
            str(value.dtype),
            tuple(int(axis) for axis in value.shape),
            id(value),
        )
    if is_dataclass(value):
        return (
            type(value).__module__,
            type(value).__name__,
            tuple(
                (field.name, _spec_cache_key(getattr(value, field.name)))
                for field in fields(value)
            ),
        )
    if isinstance(value, (tuple, list)):
        return tuple(_spec_cache_key(item) for item in value)
    raise TypeError(f"Unsupported spec cache key value: {type(value).__name__}")


def _place_array_tree_on_device(tree, device):
    """Place dynamic array leaves on one device while preserving static metadata."""
    return runtime_device_put_tree(tree, device=device)


__all__ = [
    "BiotSavartJAX",
    "BiotSavartFieldPullback",
    "SingleStageRuntimeSpecBiotSavartJAX",
    "SpecBackedBiotSavartJAX",
    "SpecBackedCoil",
    "SpecBackedCurve",
    "SpecBackedCurrent",
]


def _time_call_result(callback):
    start = time.perf_counter()
    result = callback()
    _block_until_ready(result)
    return float(time.perf_counter() - start), result


def _block_until_ready(value):
    if isinstance(value, Derivative):
        block_until_ready(value.data)
        return
    block_until_ready(value)


def _cyl_points_to_cart(points_cyl):
    points = _as_jax_float64(points_cyl)
    r = points[:, 0]
    phi = points[:, 1]
    z = points[:, 2]
    return jnp.stack((r * jnp.cos(phi), r * jnp.sin(phi), z), axis=1)


def _canonical_set_points_cyl(points_cyl):
    points = _as_jax_float64(points_cyl)
    return points.at[:, 1].set(jnp.fmod(points[:, 1], 2.0 * jnp.pi))


def _cart_points_to_cyl(points_cart):
    points = _as_jax_float64(points_cart)
    x = points[:, 0]
    y = points[:, 1]
    phi = jnp.arctan2(y, x)
    phi = jnp.where(phi < 0.0, phi + 2.0 * jnp.pi, phi)
    return jnp.stack(
        (
            jnp.sqrt(x * x + y * y),
            phi,
            points[:, 2],
        ),
        axis=1,
    )


def _points_cyl_for_basis(points_cart, points_cyl):
    if points_cyl is not None:
        return points_cyl
    return _cart_points_to_cyl(points_cart)


def _cart_vectors_to_cyl(vectors_cart, points_cyl):
    vectors = _as_jax_float64(vectors_cart)
    points = _as_jax_float64(points_cyl)
    phi = points[:, 1]
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    return jnp.stack(
        (
            cos_phi * vectors[:, 0] + sin_phi * vectors[:, 1],
            -sin_phi * vectors[:, 0] + cos_phi * vectors[:, 1],
            vectors[:, 2],
        ),
        axis=1,
    )


def _grad_absB_from_B_and_dB(B, dB_by_dX):
    B_jax = _as_jax_float64(B)
    dB_jax = _as_jax_float64(dB_by_dX)
    absB = jnp.linalg.norm(B_jax, axis=1)
    return jnp.sum(dB_jax * B_jax[:, None, :], axis=2) / absB[:, None]


def _per_coil_unit_field_with_batch_size(points, coil_set_spec, kernel, *, batch_size):
    """Per-coil unit-current field as a list of JAX arrays.

    For each coil ``k`` in the public coil ordering, evaluates ``kernel``
    on the single-coil view with ``currents = [1.0]``. Biot-Savart is
    exactly linear in ``I``, so the resulting array equals
    ``∂F/∂I_k`` for the matching spatial-derivative kernel.

    The output is a list of ``ncoils`` separate JAX arrays — one per
    coil — so coil-axis collective reduction does not apply. The
    ``SIMSOPT_JAX_SHARDING=coil_groups`` collective path (used by
    ``grouped_biot_savart_*_from_spec``) is bypassed here by design,
    relying instead on the JAX kernel cache for compile-time reuse
    within a quadrature group.
    """
    compute_points = _as_compute_array(points)
    ncoils = sum(len(group.coil_indices) for group in coil_set_spec.groups)
    result_by_index: dict[int, jax.Array] = {}
    for group in coil_set_spec.groups:
        compute_gammas = _as_compute_array(group.gammas)
        compute_gammadashs = _as_compute_array(group.gammadashs)
        unit_current = _as_compute_array(jnp.ones((1,), dtype=group.currents.dtype))

        def evaluate_single(coil_geometry):
            gamma, gammadash = coil_geometry
            return kernel(
                compute_points,
                gamma[jnp.newaxis, ...],
                gammadash[jnp.newaxis, ...],
                unit_current,
            )

        if batch_size <= 0:
            group_results = jax.vmap(
                lambda gamma, gammadash: evaluate_single((gamma, gammadash))
            )(compute_gammas, compute_gammadashs)
        else:
            group_results = jax.lax.map(
                evaluate_single,
                (compute_gammas, compute_gammadashs),
                batch_size=batch_size,
            )
        for position, coil_index in enumerate(group.coil_indices):
            result_by_index[int(coil_index)] = group_results[position]
    return [result_by_index[index] for index in range(ncoils)]


def _per_coil_unit_field(points, coil_set_spec, kernel):
    return _per_coil_unit_field_with_batch_size(
        points,
        coil_set_spec,
        kernel,
        batch_size=get_field_kernel_tuning().coil_chunk_size,
    )


def _build_profile_breakdown(timings):
    total_s = float(sum(timings.values()))
    if total_s <= 0.0:
        return []
    ranked = sorted(
        timings.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {
            "name": name,
            "elapsed_s": float(elapsed_s),
            "share": float(elapsed_s / total_s),
        }
        for name, elapsed_s in ranked
    ]


def _build_coil_profile_breakdown(per_coil_timings):
    total_s = float(sum(entry["total_s"] for entry in per_coil_timings))
    if total_s <= 0.0:
        return []
    ranked = sorted(
        per_coil_timings,
        key=lambda entry: entry["total_s"],
        reverse=True,
    )
    return [
        {
            "coil_index": int(entry["coil_index"]),
            "elapsed_s": float(entry["total_s"]),
            "share": float(entry["total_s"] / total_s),
        }
        for entry in ranked
    ]


def _zero_profile_component_timings(component_totals):
    return {name: 0.0 for name in component_totals}


def _build_coil_profile_entry(coil_index, coil_timings):
    return {
        "coil_index": int(coil_index),
        "component_timings_s": {
            name: float(elapsed_s) for name, elapsed_s in coil_timings.items()
        },
        "total_s": float(sum(coil_timings.values())),
    }


def _build_pullback_group_profile_entry(*, kind, coil_indices, elapsed_s, native_curve):
    return {
        "kind": kind,
        "coil_indices": [int(coil_index) for coil_index in coil_indices],
        "elapsed_s": float(elapsed_s),
        "native_curve": bool(native_curve),
    }


def _build_pullback_group_profile_breakdown(entries):
    total_s = float(sum(entry["elapsed_s"] for entry in entries))
    if total_s <= 0.0:
        return []
    ranked = sorted(entries, key=lambda entry: entry["elapsed_s"], reverse=True)
    return [
        {
            **entry,
            "share": float(entry["elapsed_s"] / total_s),
        }
        for entry in ranked
    ]


@dataclass(frozen=True)
class _CoilVJPInfo:
    coil_index: int
    coil: object
    curve: object
    rotmat: object
    current: object
    scale: float
    gamma: object
    gammadash: object
    current_value: object
    native_curve: bool
    timings: dict[str, float] | None = None


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BiotSavartFieldPullback:
    """Native grouped cotangent payload for ``BiotSavartJAX`` fields.

    ``d_coil_arrays`` mirrors the grouped field-input structure:
    one ``(d_gammas, d_gammadashs, d_currents)`` tuple per quadrature group.
    ``coil_indices`` maps each group row back to the public coil list.
    """

    d_coil_arrays: tuple[tuple[jax.Array, jax.Array, jax.Array], ...]
    coil_indices: tuple[tuple[int, ...], ...]

    def tree_flatten(self):
        return (self.d_coil_arrays,), self.coil_indices

    @classmethod
    def tree_unflatten(cls, coil_indices, children):
        (d_coil_arrays,) = children
        return cls(d_coil_arrays=d_coil_arrays, coil_indices=coil_indices)


class SpecBackedCurrent:
    """Minimal current view reconstructed from the runtime seed owner state."""

    def __init__(
        self,
        *,
        owner,
        coil_index: int,
    ) -> None:
        self._owner = owner
        self._coil_index = int(coil_index)
        self.local_lower_bounds = np.asarray([-np.inf], dtype=np.float64)
        self.local_upper_bounds = np.asarray([np.inf], dtype=np.float64)

    def get_value(self) -> float:
        coil_spec = coil_specs_from_dof_extraction_spec(
            self._owner.coil_dof_extraction_spec(),
            self._owner.x,
        )[self._coil_index]
        return host_float(coil_spec.current.value[0])


class SpecBackedScaledCurrent:
    """Scaled-current view that preserves the source dependency graph."""

    def __init__(self, current_to_scale: SpecBackedCurrent, scale: float) -> None:
        self.current_to_scale = current_to_scale
        self.scale = float(scale)
        lower = np.asarray(current_to_scale.local_lower_bounds, dtype=np.float64)
        upper = np.asarray(current_to_scale.local_upper_bounds, dtype=np.float64)
        scaled_lower = lower * self.scale
        scaled_upper = upper * self.scale
        self.local_lower_bounds = np.minimum(scaled_lower, scaled_upper)
        self.local_upper_bounds = np.maximum(scaled_lower, scaled_upper)

    def get_value(self) -> float:
        return float(self.current_to_scale.get_value()) * self.scale


class SpecBackedCurve(Optimizable):
    """Read-only curve view backed by an immutable curve spec."""

    return_fn_map = {}

    def __init__(
        self,
        curve_spec: CurveSpec,
        symmetry: CoilSymmetrySpec,
        *,
        owner,
        coil_index: int,
        extraction_spec: CoilDofExtractionSpec,
    ) -> None:
        self._symmetry = symmetry
        self._owner = owner
        self._coil_index = int(coil_index)
        self._extraction_spec = extraction_spec
        self.quadpoints = host_array(curve_spec.quadpoints, dtype=np.float64)
        Optimizable.__init__(
            self,
            x0=np.asarray([], dtype=np.float64),
            depends_on=[owner],
        )

    def _current_curve_spec(self) -> CurveSpec:
        return coil_specs_from_dof_extraction_spec(
            self._owner.coil_dof_extraction_spec(),
            self._owner.x,
        )[self._coil_index].curve

    def to_spec(self) -> CurveSpec:
        return self._current_curve_spec()

    def get_dofs(self) -> jax.Array:
        return self._current_curve_spec().dofs

    def _apply_symmetry(
        self,
        gamma: jax.Array,
        *derivatives: jax.Array,
    ) -> tuple[jax.Array, ...]:
        if not self._symmetry.has_rotation:
            return (gamma, *derivatives)
        rotmat = self._symmetry.rotmat
        return (gamma @ rotmat, *(derivative @ rotmat for derivative in derivatives))

    def _geometry(self) -> tuple[jax.Array, ...]:
        return self._apply_symmetry(
            *curve_geometry_from_spec(self._current_curve_spec())
        )

    def gamma(self) -> np.ndarray:
        gamma, _gammadash = curve_gamma_and_dash_from_spec(self._current_curve_spec())
        return host_array(self._apply_symmetry(gamma)[0], dtype=np.float64)

    def gammadash(self) -> np.ndarray:
        gamma, gammadash = curve_gamma_and_dash_from_spec(self._current_curve_spec())
        return host_array(self._apply_symmetry(gamma, gammadash)[1], dtype=np.float64)

    def gammadashdash(self) -> np.ndarray:
        _gamma, _gammadash, gammadashdash = self._geometry()
        return host_array(gammadashdash, dtype=np.float64)

    def incremental_arclength(self) -> np.ndarray:
        return np.linalg.norm(self.gammadash(), axis=1)

    def kappa(self) -> np.ndarray:
        gammadash = self.gammadash()
        gammadashdash = self.gammadashdash()
        numerator = np.linalg.norm(np.cross(gammadash, gammadashdash), axis=1)
        denominator = np.linalg.norm(gammadash, axis=1) ** 3
        return numerator / denominator

    def _owner_derivative_from_curve_cotangent(
        self,
        coeff_cotangent: object,
    ) -> Derivative:
        coeff_cotangent = _as_jax_float64(coeff_cotangent)
        curve_map = _place_array_tree_on_device(
            self._extraction_spec.curve_map,
            coeff_cotangent.device,
        )
        owner_dofs = explicit_device_array(
            self._owner.x,
            dtype=np.float64,
            reference=coeff_cotangent,
        )
        owner_gradient = _dof_map_cotangent_to_owner_gradient(
            curve_map,
            coeff_cotangent,
            owner_dofs,
        )
        return Derivative({self._owner: owner_gradient})

    def _rotate_cotangent_to_base_frame(self, values: object) -> jax.Array:
        cotangent = _as_jax_float64(values)
        rotmat = explicit_device_array(
            self._symmetry.rotmat,
            dtype=np.float64,
            reference=cotangent,
        )
        return cotangent @ rotmat.T

    def _curve_pullback_derivative(self, dg: object, dgd: object) -> Derivative:
        dg = _as_jax_float64(dg)
        dgd = _as_jax_float64(dgd)
        if self._symmetry.has_rotation:
            dg = self._rotate_cotangent_to_base_frame(dg)
            dgd = self._rotate_cotangent_to_base_frame(dgd)
        curve_spec = _place_array_tree_on_device(
            self._current_curve_spec(),
            dg.device,
        )
        coeff_cotangent, _surface_cotangent = curve_pullback_from_dofs(
            curve_spec,
            curve_spec.dofs,
            dg,
            dgd,
        )
        return self._owner_derivative_from_curve_cotangent(coeff_cotangent)

    def dgamma_by_dcoeff_vjp(self, v: object) -> Derivative:
        cotangent = _as_jax_float64(v)
        return self._curve_pullback_derivative(
            cotangent,
            cotangent - cotangent,
        )

    def dgammadash_by_dcoeff_vjp(self, v: object) -> Derivative:
        cotangent = _as_jax_float64(v)
        return self._curve_pullback_derivative(
            cotangent - cotangent,
            cotangent,
        )

    def dincremental_arclength_by_dcoeff_vjp(self, v: object) -> Derivative:
        cotangent = _as_jax_float64(v)
        gammadash = explicit_device_array(
            self.gammadash(),
            dtype=np.float64,
            reference=cotangent,
        )
        incremental_arclength = jnp.linalg.norm(gammadash, axis=1)
        dgd = cotangent[:, None] * gammadash / incremental_arclength[:, None]
        return self.dgammadash_by_dcoeff_vjp(dgd)

    def dkappa_by_dcoeff_vjp(self, v: object) -> Derivative:
        cotangent = _as_jax_float64(v)
        curve_spec = _place_array_tree_on_device(
            self._current_curve_spec(),
            cotangent.device,
        )

        def kappa_from_dofs(curve_dofs):
            spec = curve_spec_with_dofs(curve_spec, curve_dofs)
            _gamma, gammadash, gammadashdash = curve_geometry_from_spec(spec)
            numerator = jnp.linalg.norm(jnp.cross(gammadash, gammadashdash), axis=1)
            speed = jnp.linalg.norm(gammadash, axis=1)
            denominator = speed * speed * speed
            return numerator / denominator

        _kappa, pullback = jax.vjp(kappa_from_dofs, curve_spec.dofs)
        (coeff_cotangent,) = pullback(cotangent)
        return self._owner_derivative_from_curve_cotangent(coeff_cotangent)


class SpecBackedRotatedCurve(Optimizable):
    """Rotated-curve view that preserves the source dependency graph."""

    return_fn_map = {}

    def __init__(self, curve: SpecBackedCurve, rotmat: object) -> None:
        self.curve = curve
        self.rotmat = host_array(rotmat, dtype=np.float64)
        self.quadpoints = curve.quadpoints
        Optimizable.__init__(
            self,
            x0=np.asarray([], dtype=np.float64),
            depends_on=[curve],
        )

    def _rotate(self, values: object) -> np.ndarray:
        return host_array(
            _as_jax_float64(values) @ _as_jax_float64(self.rotmat),
            dtype=np.float64,
        )

    def _rotate_cotangent(self, values: object) -> jax.Array:
        cotangent = _as_jax_float64(values)
        rotmat = explicit_device_array(
            self.rotmat,
            dtype=np.float64,
            reference=cotangent,
        )
        return cotangent @ rotmat.T

    def gamma(self) -> np.ndarray:
        return self._rotate(self.curve.gamma())

    def gammadash(self) -> np.ndarray:
        return self._rotate(self.curve.gammadash())

    def gammadashdash(self) -> np.ndarray:
        return self._rotate(self.curve.gammadashdash())

    def incremental_arclength(self) -> np.ndarray:
        return self.curve.incremental_arclength()

    def kappa(self) -> np.ndarray:
        return self.curve.kappa()

    def get_dofs(self) -> jax.Array:
        return self.curve.get_dofs()

    def to_spec(self) -> CurveSpec:
        return self.curve.to_spec()

    def dgamma_by_dcoeff_vjp(self, v: object) -> Derivative:
        return self.curve.dgamma_by_dcoeff_vjp(self._rotate_cotangent(v))

    def dgammadash_by_dcoeff_vjp(self, v: object) -> Derivative:
        return self.curve.dgammadash_by_dcoeff_vjp(self._rotate_cotangent(v))

    def dincremental_arclength_by_dcoeff_vjp(self, v: object) -> Derivative:
        return self.curve.dincremental_arclength_by_dcoeff_vjp(v)

    def dkappa_by_dcoeff_vjp(self, v: object) -> Derivative:
        return self.curve.dkappa_by_dcoeff_vjp(v)


def _set_biot_savart_points(field, points):
    field._points_jax = _as_jax_float64(points)
    field._points_cyl_jax = None
    field._points_version += 1
    return field


def _set_biot_savart_points_cyl(field, points_cyl):
    field._points_cyl_jax = _canonical_set_points_cyl(points_cyl)
    field._points_jax = _cyl_points_to_cart(field._points_cyl_jax)
    field._points_version += 1
    return field


def _get_biot_savart_points_cyl(field):
    if field._points_cyl_jax is not None:
        return host_array(field._points_cyl_jax, dtype=np.float64)
    return host_array(_cart_points_to_cyl(field._points_jax), dtype=np.float64)


class _BiotSavartFieldEvaluationMixin:
    """Shared grouped-kernel field API for graph-backed and spec-backed fields."""

    def clear_points(self) -> None:
        """Clear mutable point buffers without changing source geometry."""
        self._points_jax = None
        self._points_cyl_jax = None
        self._points_version += 1

    def _per_coil_unit_current_derivative(self, kernel):
        """Evaluate a unit-current derivative kernel for this field state."""
        return _per_coil_unit_field(
            self._points_jax,
            self.coil_set_spec(),
            kernel,
        )

    def B(self):
        """Magnetic field B at the evaluation points."""
        return grouped_biot_savart_B_from_spec(self._points_jax, self.coil_set_spec())

    def A(self):
        """Vector potential A at the evaluation points."""
        return grouped_biot_savart_A_from_spec(self._points_jax, self.coil_set_spec())

    def dA_by_dX(self):
        """Spatial Jacobian dA/dX at the evaluation points."""
        return grouped_biot_savart_dA_by_dX_from_spec(
            self._points_jax,
            self.coil_set_spec(),
        )

    def d2A_by_dXdX(self):
        """Spatial Hessian d2A/dXdX at the evaluation points."""
        return grouped_biot_savart_d2A_by_dXdX_from_spec(
            self._points_jax,
            self.coil_set_spec(),
        )

    def dB_by_dX(self):
        """Spatial Jacobian dB/dX at the evaluation points."""
        return grouped_biot_savart_dB_by_dX_from_spec(
            self._points_jax,
            self.coil_set_spec(),
        )

    def d2B_by_dXdX(self):
        """Spatial Hessian d2B/dXdX at the evaluation points."""
        return grouped_biot_savart_d2B_by_dXdX_from_spec(
            self._points_jax,
            self.coil_set_spec(),
        )

    def B_and_dB(self):
        """Combined B and dB/dX."""
        return grouped_biot_savart_B_and_dB_from_spec(
            self._points_jax,
            self.coil_set_spec(),
        )

    def AbsB(self):
        """Magnetic-field magnitude at the evaluation points."""
        return jnp.linalg.norm(self.B(), axis=1)[:, None]

    def GradAbsB(self):
        """Cartesian gradient of ``|B|`` at the evaluation points."""
        return _grad_absB_from_B_and_dB(*self.B_and_dB())

    def B_cyl(self):
        """Magnetic field components in the cylindrical basis."""
        return _cart_vectors_to_cyl(
            self.B(),
            _points_cyl_for_basis(self._points_jax, self._points_cyl_jax),
        )

    def A_cyl(self):
        """Vector potential components in the cylindrical basis."""
        return _cart_vectors_to_cyl(
            self.A(),
            _points_cyl_for_basis(self._points_jax, self._points_cyl_jax),
        )

    def GradAbsB_cyl(self):
        """``GradAbsB`` components in the cylindrical basis."""
        return _cart_vectors_to_cyl(
            self.GradAbsB(),
            _points_cyl_for_basis(self._points_jax, self._points_cyl_jax),
        )

    def dB_by_dcoilcurrents(self, compute_derivatives=0):
        """Per-coil B at unit current."""
        return self._per_coil_unit_current_derivative(biot_savart_B)

    def d2B_by_dXdcoilcurrents(self, compute_derivatives=1):
        """Per-coil ``dB/dX`` at unit current."""
        return self._per_coil_unit_current_derivative(biot_savart_dB_by_dX)

    def d3B_by_dXdXdcoilcurrents(self, compute_derivatives=2):
        """Per-coil ``d2B/dXdX`` at unit current."""
        return self._per_coil_unit_current_derivative(biot_savart_d2B_by_dXdX)

    def dA_by_dcoilcurrents(self, compute_derivatives=0):
        """Per-coil A at unit current."""
        return self._per_coil_unit_current_derivative(biot_savart_A)

    def d2A_by_dXdcoilcurrents(self, compute_derivatives=1):
        """Per-coil ``dA/dX`` at unit current."""
        return self._per_coil_unit_current_derivative(biot_savart_dA_by_dX)

    def d3A_by_dXdXdcoilcurrents(self, compute_derivatives=2):
        """Per-coil ``d2A/dXdX`` at unit current."""
        return self._per_coil_unit_current_derivative(biot_savart_d2A_by_dXdX)


class SpecBackedCoil:
    """Read-only coil view reconstructed from a serialized JAX seed spec."""

    def __init__(
        self,
        coil_spec: CoilSpec,
        *,
        owner,
        coil_index: int,
        extraction_spec: CoilDofExtractionSpec,
        base_curve: SpecBackedCurve,
        base_current: SpecBackedCurrent,
    ) -> None:
        self._owner = owner
        self._coil_index = int(coil_index)
        del extraction_spec
        self.curve = (
            SpecBackedRotatedCurve(base_curve, coil_spec.symmetry.rotmat)
            if coil_spec.symmetry.has_rotation
            else base_curve
        )
        self.current = (
            SpecBackedScaledCurrent(base_current, coil_spec.symmetry.scale)
            if float(coil_spec.symmetry.scale) != 1.0
            else base_current
        )

    def to_spec(self) -> CoilSpec:
        return coil_specs_from_dof_extraction_spec(
            self._owner.coil_dof_extraction_spec(),
            self._owner.x,
        )[self._coil_index]


class SpecBackedBiotSavartJAX(_BiotSavartFieldEvaluationMixin, Optimizable):
    """Biot-Savart adapter whose source of truth is an immutable spec."""

    return_fn_map = {}

    def __init__(self, biot_savart_spec: BiotSavartSpec) -> None:
        self.biot_savart_spec = biot_savart_spec
        self._coil_dof_extraction_spec = biot_savart_spec.coil_dof_extraction
        self._x = _as_jax_float64(biot_savart_spec.coil_dofs)
        self._coil_dofs_generation = 0
        self._coil_dof_state_token = _new_coil_dof_state_token()
        self._points_jax: jax.Array | None = None
        self._points_cyl_jax: jax.Array | None = None
        self._points_version = 0
        self._dof_layout_version = 0
        self._free_dof_layout_ready = False
        self._uses_uniform_curve_xyz_fourier_fastpath = False
        Optimizable.__init__(self, x0=host_array(self._x, dtype=np.float64))
        self._coils = self._coils_from_dofs(self._x)
        self._free_dof_layout_ready = True

    def update_free_dof_size_indices(self) -> None:
        super().update_free_dof_size_indices()
        if self._free_dof_layout_ready:
            self._dof_layout_version += 1

    def _coils_from_dofs(
        self,
        coil_dofs: object,
    ) -> tuple[SpecBackedCoil, ...]:
        coil_specs = coil_specs_from_dof_extraction_spec(
            self._coil_dof_extraction_spec,
            coil_dofs,
        )
        curve_views: dict[object, SpecBackedCurve] = {}
        current_views: dict[object, SpecBackedCurrent] = {}
        coils = []
        for coil_index, (extraction_spec, coil_spec) in enumerate(
            zip(
                self._coil_dof_extraction_spec.coils,
                coil_specs,
                strict=True,
            )
        ):
            curve_key = (
                _spec_cache_key(extraction_spec.curve),
                _spec_cache_key(extraction_spec.curve_map),
                _spec_cache_key(extraction_spec.surface_map),
                extraction_spec.surface_output_index,
            )
            if curve_key not in curve_views:
                curve_views[curve_key] = SpecBackedCurve(
                    coil_spec.curve,
                    CoilSymmetrySpec(
                        rotmat=coil_spec.symmetry.rotmat,
                        scale=1.0,
                        has_rotation=False,
                    ),
                    owner=self,
                    coil_index=coil_index,
                    extraction_spec=extraction_spec,
                )
            current_key = (
                (
                    "owner_segments",
                    extraction_spec.current_map.owner_segments,
                )
                if extraction_spec.current_map.owner_segments
                else (
                    "fixed_current",
                    _spec_cache_key(coil_spec.current),
                    coil_index,
                )
            )
            if current_key not in current_views:
                current_views[current_key] = SpecBackedCurrent(
                    owner=self,
                    coil_index=coil_index,
                )
            coils.append(
                SpecBackedCoil(
                    coil_spec,
                    owner=self,
                    coil_index=coil_index,
                    extraction_spec=extraction_spec,
                    base_curve=curve_views[curve_key],
                    base_current=current_views[current_key],
                )
            )
        return tuple(coils)

    @property
    def x(self) -> jax.Array:
        return self._x

    def _set_coil_dofs(self, coil_dofs: object) -> None:
        self._x = _as_jax_float64(coil_dofs)
        assert self._x.shape[0] == self.dof_size, (
            f"BiotSavartSpec DOF count {self._x.shape[0]} != "
            f"Optimizable.dof_size {self.dof_size}"
        )
        self._coil_dofs_generation += 1
        self._coil_dof_state_token = _new_coil_dof_state_token()

    @x.setter
    def x(self, coil_dofs: object) -> None:
        self._set_coil_dofs(coil_dofs)
        if hasattr(self, "_dofs"):
            self._dofs.free_x = host_array(self._x, dtype=np.float64)

    @property
    def local_x(self) -> np.ndarray:
        return host_array(self._x, dtype=np.float64)

    @local_x.setter
    def local_x(self, coil_dofs: object) -> None:
        self._set_coil_dofs(coil_dofs)
        self._dofs.free_x = host_array(self._x, dtype=np.float64)

    @property
    def local_full_x(self) -> np.ndarray:
        return host_array(self._x, dtype=np.float64)

    @local_full_x.setter
    def local_full_x(self, coil_dofs: object) -> None:
        self._set_coil_dofs(coil_dofs)
        self._dofs.full_x = host_array(self._x, dtype=np.float64)

    @property
    def coils(self) -> tuple[SpecBackedCoil, ...]:
        return self._coils

    @property
    def dof_layout_version(self) -> int:
        """Return the monotonic free/fixed DOF-layout version."""
        return self._dof_layout_version

    def coil_dof_extraction_spec(self) -> CoilSetDofExtractionSpec:
        return self._coil_dof_extraction_spec

    def coil_set_spec_from_dofs(self, coil_dofs: object) -> GroupedCoilSetSpec:
        return coil_set_spec_from_dof_extraction_spec(
            self._coil_dof_extraction_spec,
            _as_jax_float64(coil_dofs),
        )

    def coil_set_spec(self) -> GroupedCoilSetSpec:
        return self.coil_set_spec_from_dofs(self._x)

    def grouped_coil_arrays_from_dofs(
        self,
        coil_dofs: object,
    ) -> list[tuple[jax.Array, jax.Array, jax.Array]]:
        return list(self.coil_set_spec_from_dofs(coil_dofs).field_inputs())

    def set_points(self, points: object):
        return _set_biot_savart_points(self, points)

    def set_points_cart(self, points: object):
        return _set_biot_savart_points(self, points)

    def set_points_cyl(self, points_cyl: object):
        return _set_biot_savart_points_cyl(self, points_cyl)

    def set_points_from_spec(self, field_eval_spec: FieldEvalSpec):
        return _set_biot_savart_points(self, field_eval_spec.points)

    def get_points_cart_ref(self):
        return self._points_jax

    def get_points_cart(self):
        return host_array(self._points_jax, dtype=np.float64)

    def get_points_cyl(self):
        return _get_biot_savart_points_cyl(self)

    def field_eval_spec(self) -> FieldEvalSpec:
        return make_field_eval_spec(self._points_jax)

    def coil_cotangents_to_dofs_gradient(
        self,
        d_coil_arrays,
        coil_indices,
        *,
        coil_dofs=None,
    ):
        """Project grouped coil cotangents to the runtime-spec flat DOF order."""
        if coil_dofs is None:
            coil_dofs = self.x
        coil_dofs = _as_jax_float64(coil_dofs)
        extraction_spec = _place_array_tree_on_device(
            self._coil_dof_extraction_spec,
            coil_dofs.device,
        )
        return _jitted_coil_cotangents_to_dofs_gradient(
            extraction_spec,
            d_coil_arrays,
            _canonical_coil_indices(coil_indices),
            coil_dofs,
        )

    def save(self, _path: object) -> None:
        raise RuntimeError("JAX runtime seed specs split runtime from host export.")


class SingleStageRuntimeSpecBiotSavartJAX(SpecBackedBiotSavartJAX):
    """Biot-Savart adapter whose source of truth is a runtime seed spec."""

    def __init__(self, runtime_spec: SingleStageRuntimeSpec) -> None:
        self.runtime_spec = runtime_spec
        super().__init__(
            make_biot_savart_spec(
                coil_dof_extraction=runtime_spec.seed.coil_dof_extraction,
                coil_dofs=runtime_spec.seed.coil_dofs,
            )
        )


def _supports_native_curve_geometry(curve):
    return supports_adapter_curve_spec(curve)


def _require_native_curve_geometry(curve):
    if not _supports_native_curve_geometry(curve):
        raise TypeError(
            "BiotSavartJAX coil cotangent projection requires immutable JAX "
            f"curve specs; unsupported type {type(curve).__name__}. "
            "Provide a native curve spec."
        )


def _require_native_coil_projection_specs(coils, coil_indices):
    for indices in coil_indices:
        for global_i in indices:
            curve, _rotmat, _current, _scale = _unwrap_coil_curve_and_current(
                coils[int(global_i)]
            )
            _require_native_curve_geometry(curve)


def _curve_dof_mode(curve):
    return adapter_curve_dof_mode(curve)


def _curve_live_dofs(curve):
    if _curve_dof_mode(curve) == "full":
        return _as_jax_float64(curve.full_x)
    return _as_jax_float64(curve.get_dofs())


def _curve_quadpoints_jax(curve):
    return _as_jax_float64(curve.quadpoints)


def _merge_derivative_data(target, derivative_like):
    items = (
        derivative_like.data.items()
        if isinstance(derivative_like, Derivative)
        else derivative_like.items()
    )
    for opt, block in items:
        if opt in target:
            target[opt] = target[opt] + block
        else:
            target[opt] = block.copy() if hasattr(block, "copy") else block


def _full_curve_cotangent_to_derivative(curve, full_cotangent):
    full_cotangent = _as_jax_float64(full_cotangent)
    deriv_data = {}
    for opt, (start, end) in curve._full_dof_indices.items():
        if opt.local_full_dof_size == 0:
            continue
        deriv_data[opt] = _slice_1d(full_cotangent, start, end)
    return deriv_data


def _slice_1d(array: jax.Array, start: int, end: int) -> jax.Array:
    return jax.lax.slice_in_dim(array, int(start), int(end), axis=0)


def _update_1d(array: jax.Array, start: int, values: jax.Array) -> jax.Array:
    start = int(start)
    stop = start + int(values.shape[0])
    return jnp.concatenate(
        (
            _slice_1d(array, 0, start),
            values,
            _slice_1d(array, stop, int(array.shape[0])),
        )
    )


def _add_update_1d(array: jax.Array, start: int, values: jax.Array) -> jax.Array:
    start = int(start)
    stop = start + int(values.shape[0])
    return _update_1d(array, start, _slice_1d(array, start, stop) + values)


def _take_positions_1d(array: jax.Array, positions) -> jax.Array:
    indexer = runtime_device_put(positions, dtype=np.int32)
    return jnp.take(_as_jax_float64(array), indexer, axis=0)


def _scatter_free_values(template: jax.Array, free_positions, free_values: jax.Array):
    free_positions = np.asarray(free_positions, dtype=np.int64)
    if np.array_equal(free_positions, np.arange(int(template.shape[0]))):
        return free_values
    mask = np.ones(int(template.shape[0]), dtype=np.float64)
    mask[free_positions] = 0.0
    indexer = runtime_device_put(free_positions[:, None], dtype=np.int32)
    dnums = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    return jax.lax.scatter(
        template * runtime_device_put(mask, dtype=np.float64),
        indexer,
        free_values,
        dnums,
        indices_are_sorted=True,
        unique_indices=True,
    )


def _dof_map_cotangent_to_owner_gradient(map_spec, input_cotangent, owner_dofs):
    owner_dofs = _as_jax_float64(owner_dofs)
    input_cotangent = _as_jax_float64(input_cotangent)
    if map_spec.input_mode == "full":
        full_cotangent = input_cotangent
    else:
        zero = jnp.sum(input_cotangent, dtype=input_cotangent.dtype)
        full_cotangent = jnp.broadcast_to(
            zero - zero,
            map_spec.template_full_dofs.shape,
        )
        full_cotangent = _add_update_1d(
            full_cotangent,
            map_spec.input_start,
            input_cotangent,
        )

    owner_gradient = owner_dofs - owner_dofs
    for owner_start, _owner_end, target_start, target_end in map_spec.owner_segments:
        owner_gradient = _add_update_1d(
            owner_gradient,
            owner_start,
            _slice_1d(full_cotangent, target_start, target_end),
        )
    return owner_gradient


def _add_extraction_cotangent_to_dofs_gradient(
    dofs_gradient,
    extraction_spec,
    coil_spec,
    dg,
    dgd,
    dc,
    coil_dofs,
):
    if extraction_spec.symmetry.has_rotation:
        rotmat_t = _as_jax_float64(extraction_spec.symmetry.rotmat).T
        dg = _as_jax_float64(dg) @ rotmat_t
        dgd = _as_jax_float64(dgd) @ rotmat_t

    coeff_cotangent, surface_cotangent = curve_pullback_from_dofs(
        coil_spec.curve,
        coil_spec.curve.dofs,
        dg,
        dgd,
    )
    current_cotangent = jnp.atleast_1d(
        _as_jax_float64(extraction_spec.symmetry.scale) * _as_jax_float64(dc)
    )
    dofs_gradient = dofs_gradient + _dof_map_cotangent_to_owner_gradient(
        extraction_spec.curve_map,
        coeff_cotangent,
        coil_dofs,
    )
    if extraction_spec.surface_map is not None and surface_cotangent is not None:
        dofs_gradient = dofs_gradient + _dof_map_cotangent_to_owner_gradient(
            extraction_spec.surface_map,
            surface_cotangent,
            coil_dofs,
        )
    return dofs_gradient + _dof_map_cotangent_to_owner_gradient(
        extraction_spec.current_map,
        current_cotangent,
        coil_dofs,
    )


def _empty_external_surface_cotangents(coil_dof_extraction_spec):
    output_count = 1 + max(
        (
            extraction_spec.surface_output_index
            for extraction_spec in coil_dof_extraction_spec.coils
            if extraction_spec.surface_output_index is not None
        ),
        default=-1,
    )
    surface_cotangents = [None] * output_count
    for extraction_spec in coil_dof_extraction_spec.coils:
        output_index = extraction_spec.surface_output_index
        if output_index is None or surface_cotangents[output_index] is not None:
            continue
        surface_dofs = extraction_spec.curve.surface_dofs()
        surface_cotangents[output_index] = surface_dofs - surface_dofs
    return tuple(surface_cotangents)


def _add_external_surface_cotangent(
    surface_cotangents,
    output_index,
    surface_cotangent,
):
    if output_index is None or surface_cotangent is None:
        return surface_cotangents
    updated = list(surface_cotangents)
    updated[output_index] = updated[output_index] + surface_cotangent
    return tuple(updated)


def _coil_cotangents_to_dofs_gradient_from_extraction_spec(
    coil_dof_extraction_spec,
    d_coil_arrays,
    coil_indices,
    coil_dofs,
):
    coil_dofs = _as_jax_float64(coil_dofs)
    dofs_gradient = coil_dofs - coil_dofs
    coil_specs = coil_specs_from_dof_extraction_spec(
        coil_dof_extraction_spec,
        coil_dofs,
    )
    extraction_specs = coil_dof_extraction_spec.coils
    for (d_g, d_gd, d_c), indices in zip(d_coil_arrays, coil_indices):
        for local_i, global_i in enumerate(indices):
            dofs_gradient = _add_extraction_cotangent_to_dofs_gradient(
                dofs_gradient,
                extraction_specs[global_i],
                coil_specs[global_i],
                jax.lax.index_in_dim(d_g, local_i, axis=0, keepdims=False),
                jax.lax.index_in_dim(d_gd, local_i, axis=0, keepdims=False),
                jax.lax.index_in_dim(d_c, local_i, axis=0, keepdims=False),
                coil_dofs,
            )
    return dofs_gradient


def _external_surface_cotangents_from_extraction_spec(
    coil_dof_extraction_spec,
    d_coil_arrays,
    coil_indices,
    coil_dofs,
):
    coil_dofs = _as_jax_float64(coil_dofs)
    surface_cotangents = _empty_external_surface_cotangents(coil_dof_extraction_spec)
    if not surface_cotangents:
        return surface_cotangents

    coil_specs = coil_specs_from_dof_extraction_spec(
        coil_dof_extraction_spec,
        coil_dofs,
    )
    extraction_specs = coil_dof_extraction_spec.coils
    for (d_g, d_gd, _d_c), indices in zip(d_coil_arrays, coil_indices):
        for local_i, global_i in enumerate(indices):
            extraction_spec = extraction_specs[global_i]
            if extraction_spec.surface_output_index is None:
                continue
            dg = jax.lax.index_in_dim(d_g, local_i, axis=0, keepdims=False)
            dgd = jax.lax.index_in_dim(d_gd, local_i, axis=0, keepdims=False)
            if extraction_spec.symmetry.has_rotation:
                rotmat_t = _as_jax_float64(extraction_spec.symmetry.rotmat).T
                dg = _as_jax_float64(dg) @ rotmat_t
                dgd = _as_jax_float64(dgd) @ rotmat_t

            _coeff_cotangent, surface_cotangent = curve_pullback_from_dofs(
                coil_specs[global_i].curve,
                coil_specs[global_i].curve.dofs,
                dg,
                dgd,
            )
            surface_cotangents = _add_external_surface_cotangent(
                surface_cotangents,
                extraction_spec.surface_output_index,
                surface_cotangent,
            )
    return surface_cotangents


@partial(jax.jit, static_argnames=("coil_indices",))
def _jitted_coil_cotangents_to_dofs_gradient(
    coil_dof_extraction_spec,
    d_coil_arrays,
    coil_indices,
    coil_dofs,
):
    return _coil_cotangents_to_dofs_gradient_from_extraction_spec(
        coil_dof_extraction_spec,
        d_coil_arrays,
        coil_indices,
        coil_dofs,
    )


@partial(jax.jit, static_argnames=("coil_indices",))
def _jitted_external_surface_cotangents(
    coil_dof_extraction_spec,
    d_coil_arrays,
    coil_indices,
    coil_dofs,
):
    return _external_surface_cotangents_from_extraction_spec(
        coil_dof_extraction_spec,
        d_coil_arrays,
        coil_indices,
        coil_dofs,
    )


def _canonical_coil_indices(coil_indices):
    return tuple(tuple(int(index) for index in indices) for indices in coil_indices)


def _coil_cotangent_arrays_are_jax_compatible(d_coil_arrays):
    leaves = jax.tree.leaves(d_coil_arrays)
    return all(
        isinstance(leaf, (jax.Array, np.ndarray)) or hasattr(leaf, "aval")
        for leaf in leaves
    )


def _add_local_cotangent_to_dofs_gradient(
    dofs_gradient: jax.Array,
    opt,
    full_cotangent,
    dof_indices,
    *,
    free_positions,
):
    if opt.local_dof_size == 0:
        return dofs_gradient
    start, end = dof_indices[opt]
    free_cotangent = _take_positions_1d(full_cotangent, free_positions)
    return _add_update_1d(dofs_gradient, start, free_cotangent)


def _add_full_curve_cotangent_to_dofs_gradient(
    dofs_gradient: jax.Array,
    curve,
    full_cotangent,
    dof_indices,
    *,
    free_positions_for_opt,
):
    full_cotangent = _as_jax_float64(full_cotangent)
    for opt, (start, end) in curve._full_dof_indices.items():
        dofs_gradient = _add_local_cotangent_to_dofs_gradient(
            dofs_gradient,
            opt,
            _slice_1d(full_cotangent, start, end),
            dof_indices,
            free_positions=free_positions_for_opt(opt),
        )
    return dofs_gradient


def _curve_pullback_data_from_spec(curve, dg, dgd):
    spec = curve_spec_from_adapter_curve(curve)
    coeff_cotangent, surface_cotangent = curve_pullback_from_dofs(
        spec,
        _curve_live_dofs(curve),
        dg,
        dgd,
    )
    if _curve_dof_mode(curve) == "full":
        return _full_curve_cotangent_to_derivative(curve, coeff_cotangent)
    deriv_data = {curve: coeff_cotangent}
    if surface_cotangent is not None:
        deriv_data[curve.surf] = surface_cotangent
    return deriv_data


def _merge_curve_pullback_data(deriv_data, curve, dg, dgd):
    _merge_derivative_data(
        deriv_data,
        _curve_pullback_data_from_spec(curve, dg, dgd),
    )


def _curve_gamma_and_dash_from_dofs(curve, curve_dofs):
    return curve_gamma_and_dash_from_spec_dofs(
        curve_spec_from_adapter_curve(curve),
        curve_dofs,
    )


def _project_single_coil_cotangent_data(coil, dg, dgd, dc):
    curve, rotmat, current, scale = _unwrap_coil_curve_and_current(coil)
    _require_native_curve_geometry(curve)

    if rotmat is not None:
        rotmat_t = _as_jax_float64(rotmat).T
        dg = _as_jax_float64(dg) @ rotmat_t
        dgd = _as_jax_float64(dgd) @ rotmat_t

    deriv_data = {}
    _merge_curve_pullback_data(deriv_data, curve, dg, dgd)
    if current.dof_size > 0:
        current_cotangent = jnp.atleast_1d(_as_jax_float64(scale) * _as_jax_float64(dc))
        _merge_derivative_data(deriv_data, current.vjp(current_cotangent))
    return deriv_data


def project_coil_cotangents_to_derivative(coils, d_coil_arrays, coil_indices):
    """Project grouped coil cotangents to a single public ``Derivative``."""
    _require_native_coil_projection_specs(coils, coil_indices)
    deriv_data = {}
    for (d_g, d_gd, d_c), indices in zip(d_coil_arrays, coil_indices):
        for local_i, global_i in enumerate(indices):
            _merge_derivative_data(
                deriv_data,
                _project_single_coil_cotangent_data(
                    coils[global_i],
                    d_g[local_i],
                    d_gd[local_i],
                    d_c[local_i],
                ),
            )
    return Derivative(deriv_data)


def dofs_gradient_to_derivative(unique_dof_lineage, dofs_gradient):
    """Convert a flat free-DOF gradient into the public ``Derivative`` contract."""
    dofs_gradient = host_array(dofs_gradient, dtype=np.float64)
    deriv_data = {}
    start = 0
    for lineage_opt in unique_dof_lineage:
        width = lineage_opt.local_dof_size
        if width == 0:
            continue

        stop = start + width
        block = np.zeros(lineage_opt.local_full_dof_size)
        block[lineage_opt.local_dofs_free_status] = dofs_gradient[start:stop]
        start = stop

        dep_opts = tuple(lineage_opt.dofs.dep_opts())
        block_share = block / len(dep_opts)
        for dep_opt in dep_opts:
            if dep_opt in deriv_data:
                deriv_data[dep_opt] = deriv_data[dep_opt] + block_share
            else:
                deriv_data[dep_opt] = block_share.copy()

    return Derivative(deriv_data)


def _rotate_curve_geometry(gamma, gammadash, rotmat):
    if rotmat is None:
        return gamma, gammadash
    return gamma @ rotmat, gammadash @ rotmat


def _unwrap_coil_curve_and_current(coil):
    curve, rotmat, current, scale = _unwrap_coil_curve_and_current_objects(
        coil.curve,
        coil.current,
    )
    return (
        curve,
        (None if rotmat is None else _as_jax_float64(rotmat)),
        current,
        scale,
    )


class BiotSavartJAX(_BiotSavartFieldEvaluationMixin, Optimizable):
    r"""JAX-backed Biot-Savart magnetic field evaluation.

    Drop-in replacement for :class:`BiotSavart` in workflows where the
    field is consumed by a JAX-backed objective (e.g. ``SquaredFluxJAX``).

    The class holds no DOFs of its own.  Its ``Optimizable`` dependency
    chain runs through the coil list so that the outer framework
    correctly composes DOFs and derivatives.

    The immutable spec layer is already the hot-path contract here:
    ``coil_specs()`` / ``coil_set_spec()`` expose grouped pytree payloads
    consumed by the pure JAX field kernels. The wrapper remains a mutable
    ``Optimizable`` adapter so legacy flat-vector workflows, caching, and
    derivative plumbing continue to compose around that spec boundary.

    The wrapper itself is mutable and not safe for concurrent use from
    multiple threads. ``set_points()`` / ``set_points_from_spec()`` update the
    cached evaluation points in-place, so shared instances should be treated as
    thread-confined object adapters around the immutable grouped-coil specs.

    When all coils use ``CurveXYZFourier`` (possibly wrapped in
    ``RotatedCurve``), the JAX-native path is enabled: coil geometry
    is evaluated from DOFs via a precomputed Fourier basis matrix
    entirely inside the JIT boundary, eliminating CPU round-trips.
    More general curve families can still participate when they expose
    immutable JAX specs used below. Unsupported curves are rejected explicitly;
    CPU geometry/pullback code is not used by this adapter.

    Args:
        coils: list of :class:`simsopt.field.coil.Coil` objects.
    """

    _simsopt_jax_native_field = True

    def __init__(self, coils):
        self._coils = list(coils)
        self._points_jax = None
        self._points_cyl_jax = None
        self._points_version = 0
        self._dof_layout_version = 0
        self._coil_dofs_generation = 0
        self._coil_dof_state_token = _new_coil_dof_state_token()
        self._free_dof_layout_ready = False
        self._suppress_dependency_coil_dof_state = False
        self._local_free_positions_by_opt = {}
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=self._coils)

        # Uniform CurveXYZFourier fast-path metadata (populated by _introspect_coils)
        self._uses_uniform_curve_xyz_fourier_fastpath = False
        self._unique_base_curves = []
        self._unique_base_currents = []
        self._coil_descs = []  # list of (curve_idx, current_idx, rotmat_jax, scale)
        self._curve_order = 0
        self._curve_quadpoints_jax = None
        self._introspect_coils()
        self._free_dof_layout_ready = True
        self._coil_dof_extraction_spec = self._build_coil_dof_extraction_spec()
        self._fixed_coil_dof_template_fingerprint = (
            self._current_fixed_coil_dof_template_fingerprint()
        )

    def update_free_dof_size_indices(self) -> None:
        super().update_free_dof_size_indices()
        self._local_free_positions_by_opt.clear()
        if self._free_dof_layout_ready:
            self._dof_layout_version += 1
            self._coil_dof_extraction_spec = self._build_coil_dof_extraction_spec()
            self._fixed_coil_dof_template_fingerprint = (
                self._current_fixed_coil_dof_template_fingerprint()
            )

    def _current_fixed_coil_dof_template_fingerprint(self) -> tuple[bytes, ...]:
        return tuple(
            np.ascontiguousarray(
                np.asarray(opt.local_full_x, dtype=np.float64)[
                    ~np.asarray(opt.local_dofs_free_status, dtype=bool)
                ]
            ).tobytes()
            for opt in self.unique_dof_lineage
        )

    def _refresh_fixed_coil_dof_template(self) -> None:
        fingerprint = self._current_fixed_coil_dof_template_fingerprint()
        if fingerprint == self._fixed_coil_dof_template_fingerprint:
            return
        self._coil_dof_extraction_spec = self._build_coil_dof_extraction_spec()
        self._fixed_coil_dof_template_fingerprint = fingerprint

    def _advance_coil_dof_state(self) -> None:
        self._coil_dofs_generation += 1
        self._coil_dof_state_token = _new_coil_dof_state_token()

    def set_recompute_flag(self, parent=None):
        if (
            parent is not None
            and self._free_dof_layout_ready
            and not self._suppress_dependency_coil_dof_state
        ):
            self._advance_coil_dof_state()
            self._refresh_fixed_coil_dof_template()
        super().set_recompute_flag(parent=parent)

    def _set_global_coil_dofs(
        self,
        optimizable_setter,
        coil_dofs,
        *,
        rebuild_extraction_spec,
    ):
        self._suppress_dependency_coil_dof_state = True
        try:
            optimizable_setter(self, coil_dofs)
        finally:
            self._suppress_dependency_coil_dof_state = False
        self._advance_coil_dof_state()
        if rebuild_extraction_spec:
            self._refresh_fixed_coil_dof_template()

    @property
    def x(self):
        return Optimizable.x.fget(self)

    @x.setter
    def x(self, coil_dofs):
        self._set_global_coil_dofs(
            Optimizable.x.fset,
            coil_dofs,
            rebuild_extraction_spec=False,
        )

    @property
    def full_x(self):
        return Optimizable.full_x.fget(self)

    @full_x.setter
    def full_x(self, coil_dofs):
        self._set_global_coil_dofs(
            Optimizable.full_x.fset,
            coil_dofs,
            rebuild_extraction_spec=True,
        )

    def _local_free_positions(self, opt):
        cached = self._local_free_positions_by_opt.get(opt)
        if cached is None:
            cached = np.flatnonzero(opt.local_dofs_free_status)
            self._local_free_positions_by_opt[opt] = cached
        return cached

    def _introspect_coils(self):
        """Walk coil tree to identify unique base curves/currents.

        Enables the JAX-native path when all curves are
        ``CurveXYZFourier`` (possibly wrapped in ``RotatedCurve``)
        with uniform Fourier order and quadrature point count.
        """
        base_curve_ids = {}  # id(obj) → index
        base_current_ids = {}
        base_curves = []
        base_currents = []
        descs = []

        for coil in self._coils:
            curve, rotmat, current, scale = _unwrap_coil_curve_and_current(coil)

            if not isinstance(curve, CurveXYZFourier):
                return

            cid = id(curve)
            if cid not in base_curve_ids:
                base_curve_ids[cid] = len(base_curves)
                base_curves.append(curve)

            # Must resolve to a single-DOF Current (not CurrentSum etc.)
            if not isinstance(current, Current):
                return

            kid = id(current)
            if kid not in base_current_ids:
                base_current_ids[kid] = len(base_currents)
                base_currents.append(current)

            descs.append(
                (
                    base_curve_ids[cid],
                    base_current_ids[kid],
                    _as_jax_float64(rotmat) if rotmat is not None else None,
                    scale,
                )
            )

        # All curves must share the same Fourier order and quadrature grid
        orders = {c.order for c in base_curves}
        if len(orders) != 1:
            return
        ref_qp = np.asarray(base_curves[0].quadpoints)
        for c in base_curves[1:]:
            if not np.array_equal(ref_qp, np.asarray(c.quadpoints)):
                return

        self._uses_uniform_curve_xyz_fourier_fastpath = True
        self._unique_base_curves = base_curves
        self._unique_base_currents = base_currents
        self._coil_descs = descs
        self._curve_order = orders.pop()
        self._curve_quadpoints_jax = _curve_quadpoints_jax(base_curves[0])

    def _build_coil_dof_extraction_spec(self):
        external_surface_ids = {}
        external_surfaces = []

        def coil_extraction_spec(coil):
            curve, rotmat, current, scale = _unwrap_coil_curve_and_current(coil)
            curve_spec = curve_spec_from_adapter_curve(curve)
            surface = getattr(curve, "surf", None)
            is_cws_curve = curve_spec_kind(curve_spec) == "cws_fourier_rz"
            surface_map = (
                self._free_vector_dof_map_spec(surface, full_graph=False)
                if surface is not None and is_cws_curve and surface in self.dof_indices
                else None
            )
            surface_output_index = None
            if surface is not None and is_cws_curve and surface_map is None:
                surface_id = id(surface)
                if surface_id not in external_surface_ids:
                    external_surface_ids[surface_id] = len(external_surfaces)
                    external_surfaces.append(surface)
                surface_output_index = external_surface_ids[surface_id]

            return make_coil_dof_extraction_spec(
                curve=curve_spec,
                curve_map=self._free_vector_dof_map_spec(
                    curve,
                    full_graph=_curve_dof_mode(curve) == "full",
                ),
                current_map=self._free_vector_dof_map_spec(
                    current,
                    full_graph=False,
                ),
                surface_map=surface_map,
                surface_output_index=surface_output_index,
                rotmat=rotmat,
                scale=scale,
            )

        extraction_spec = make_coil_set_dof_extraction_spec(
            coil_extraction_spec(coil) for coil in self._coils
        )
        self._external_cotangent_surfaces = tuple(external_surfaces)
        return extraction_spec

    def coil_dof_extraction_spec(self):
        """Return the cached immutable owner-DOF reconstruction contract."""
        return self._coil_dof_extraction_spec

    @property
    def dof_layout_version(self) -> int:
        """Return the monotonic free/fixed DOF-layout version."""
        return self._dof_layout_version

    def _coil_arrays_in_order_from_dofs_generic_jax(self, coil_dofs):
        """Rebuild per-coil arrays for curve sets with immutable JAX specs."""
        coil_dofs = self._normalize_explicit_coil_dofs(coil_dofs)

        coil_gammas = []
        coil_gammadashs = []
        coil_currents = []
        for coil in self._coils:
            curve, rotmat, current, scale = _unwrap_coil_curve_and_current(coil)
            if not _supports_native_curve_geometry(curve):
                raise RuntimeError(
                    "grouped_coil_arrays_from_dofs() requires immutable JAX specs "
                    f"for every base curve; unsupported type {type(curve).__name__}."
                )
            if not isinstance(current, Current):
                raise RuntimeError(
                    "grouped_coil_arrays_from_dofs() only supports scalar Current "
                    "degrees of freedom on the JAX geometry lane."
                )

            curve_dofs = self._curve_dofs_from_free_vector(curve, coil_dofs)
            gamma, gammadash = _curve_gamma_and_dash_from_dofs(curve, curve_dofs)
            if rotmat is not None:
                gamma = gamma @ rotmat
                gammadash = gammadash @ rotmat

            coil_gammas.append(gamma)
            coil_gammadashs.append(gammadash)
            coil_currents.append(
                _as_jax_float64(scale)
                * self._scalar_current_value_from_dofs(
                    current,
                    coil_dofs,
                    "JAX geometry lane",
                )
            )

        return coil_gammas, coil_gammadashs, coil_currents

    def _local_full_dofs_from_free_vector(self, opt, coil_dofs):
        """Rebuild one Optimizable's full local DOF vector from ``coil_dofs``.

        ``Optimizable.x`` is ordered by unique ancestor name, not by the
        JAX-native coil grouping used below. Reconstruct each curve/current
        block from its own free-DOF slice so mixed free-current / free-curve
        graphs decode correctly.
        """
        full_x = _as_jax_float64(opt.local_full_x)
        if opt.local_dof_size == 0:
            return full_x

        start, end = self.dof_indices[opt]
        free_positions = self._local_free_positions(opt)
        coil_slice = _slice_1d(coil_dofs, start, end)
        return _scatter_free_values(full_x, free_positions, coil_slice)

    def _full_dofs_from_free_vector(self, opt, coil_dofs):
        """Rebuild one Optimizable graph's full DOF vector from ``coil_dofs``."""
        full_x = _as_jax_float64(opt.full_x)
        for dep_opt, (start, end) in opt._full_dof_indices.items():
            dep_full_x = _as_jax_float64(dep_opt.local_full_x)
            if dep_opt.local_dof_size > 0:
                dep_start, dep_end = self.dof_indices[dep_opt]
                free_positions = self._local_free_positions(dep_opt)
                dep_slice = _slice_1d(coil_dofs, dep_start, dep_end)
                dep_full_x = _scatter_free_values(
                    dep_full_x,
                    free_positions,
                    dep_slice,
                )
            full_x = _update_1d(full_x, start, dep_full_x)
        return full_x

    def _curve_dofs_from_free_vector(self, curve, coil_dofs):
        if _curve_dof_mode(curve) == "full":
            return self._full_dofs_from_free_vector(curve, coil_dofs)
        return self._local_full_dofs_from_free_vector(curve, coil_dofs)

    def _free_vector_dof_map_spec(self, opt, *, full_graph):
        if full_graph:
            owner_segments = tuple(
                (
                    owner_start,
                    owner_end,
                    int(target_start),
                    int(target_end),
                )
                for dep_opt, (target_start, target_end) in opt._full_dof_indices.items()
                if dep_opt.local_dof_size > 0
                for owner_start, owner_end in (self.dof_indices[dep_opt],)
            )
            template_full_dofs = _as_jax_float64(opt.full_x)
            return self._full_input_dof_map_spec(template_full_dofs, owner_segments)

        template_full_dofs = _as_jax_float64(opt.local_full_x)
        if opt.local_dof_size == 0:
            return self._full_input_dof_map_spec(template_full_dofs, ())

        owner_start, _owner_end = self.dof_indices[opt]
        owner_segments = tuple(
            (
                int(owner_start + source_offset),
                int(owner_start + source_offset + 1),
                int(target_position),
                int(target_position + 1),
            )
            for source_offset, target_position in enumerate(
                self._local_free_positions(opt)
            )
        )
        return self._full_input_dof_map_spec(template_full_dofs, owner_segments)

    def _full_input_dof_map_spec(self, template_full_dofs, owner_segments):
        return make_optimizable_dof_map_spec(
            template_full_dofs=template_full_dofs,
            owner_segments=owner_segments,
            input_mode="full",
            input_start=0,
            input_end=int(template_full_dofs.shape[0]),
        )

    def _normalize_explicit_coil_dofs(self, coil_dofs):
        coil_dofs = _as_jax_float64(coil_dofs)
        expected_dofs = self.dof_size
        if coil_dofs.shape[0] != expected_dofs:
            raise ValueError(
                f"Expected {expected_dofs} coil DOFs, got {coil_dofs.shape[0]}."
            )
        return coil_dofs

    def coil_specs_from_dofs(self, coil_dofs):
        """Build immutable per-coil specs from an explicit flat DOF vector."""
        coil_dofs = self._normalize_explicit_coil_dofs(coil_dofs)
        return coil_specs_from_dof_extraction_spec(
            self.coil_dof_extraction_spec(),
            coil_dofs,
        )

    def _coil_set_spec_from_dofs_immutable_specs(self, coil_dofs):
        coil_dofs = self._normalize_explicit_coil_dofs(coil_dofs)
        return grouped_coil_set_spec_from_coil_specs(
            self.coil_specs_from_dofs(coil_dofs),
        )

    def _scalar_current_value_from_dofs(self, current, coil_dofs, lane_label):
        current_full_x = self._local_full_dofs_from_free_vector(current, coil_dofs)
        if current_full_x.shape[0] != 1:
            raise RuntimeError(
                "grouped_coil_arrays_from_dofs() only supports scalar Current "
                f"degrees of freedom on the {lane_label}."
            )
        return current_full_x[0]

    def _coil_arrays_in_order_from_dofs(self, coil_dofs):
        """Build per-coil ``(gamma, gammadash, current)`` arrays from DOFs.

        This is the pure-array counterpart to reading geometry from the live
        ``Optimizable`` graph: it reconstructs coil data from the explicit
        flat ``coil_dofs`` vector without assigning ``self.x``.

        The fast path uses the JAX-native uniform-``CurveXYZFourier`` lane.
        Other curves use their immutable specs to rebuild per-coil arrays from DOFs.
        """
        if not self._uses_uniform_curve_xyz_fourier_fastpath:
            return self._coil_arrays_in_order_from_dofs_generic_jax(coil_dofs)

        coil_dofs = self._normalize_explicit_coil_dofs(coil_dofs)

        quadpoints = self._curve_quadpoints_jax

        curve_dofs = []
        for curve in self._unique_base_curves:
            curve_dofs.append(self._local_full_dofs_from_free_vector(curve, coil_dofs))

        current_values = []
        for current in self._unique_base_currents:
            current_values.append(
                self._scalar_current_value_from_dofs(
                    current,
                    coil_dofs,
                    "JAX-native lane",
                )
            )

        base_gammas = []
        base_gammadashs = []
        for curve_x in curve_dofs:
            gamma, gammadash, _, _ = jaxfouriercurve_geometry_pure(
                curve_x,
                quadpoints,
                self._curve_order,
            )
            base_gammas.append(gamma)
            base_gammadashs.append(gammadash)

        coil_gammas = []
        coil_gammadashs = []
        coil_currents = []
        for curve_idx, current_idx, rotmat, scale in self._coil_descs:
            gamma = base_gammas[curve_idx]
            gammadash = base_gammadashs[curve_idx]
            if rotmat is not None:
                gamma = gamma @ rotmat
                gammadash = gammadash @ rotmat
            coil_gammas.append(gamma)
            coil_gammadashs.append(gammadash)
            coil_currents.append(_as_jax_float64(scale) * current_values[current_idx])

        return coil_gammas, coil_gammadashs, coil_currents

    def grouped_coil_arrays_from_dofs(self, coil_dofs):
        """Build grouped coil arrays from an explicit flat DOF vector."""
        return list(
            grouped_field_inputs_from_spec(self.coil_set_spec_from_dofs(coil_dofs))
        )

    def coil_set_spec_from_dofs(self, coil_dofs):
        """Build an immutable grouped coil spec from an explicit flat DOF vector."""
        return self._coil_set_spec_from_dofs_immutable_specs(coil_dofs)

    @property
    def coils(self):
        return self._coils

    def set_points(self, points):
        """Set evaluation points (converted to a JAX array once).

        Accepts both NumPy and JAX arrays.  JAX arrays stay on device
        without a host round-trip. Mutates the cached point buffer on this
        instance, so callers should not share one ``BiotSavartJAX`` across
        concurrent evaluation threads.
        """
        points_array = (
            points if isinstance(points, jax.Array) else np.ascontiguousarray(points)
        )
        return _set_biot_savart_points(self, points_array)

    def set_points_cart(self, points):
        return self.set_points(points)

    def set_points_cyl(self, points_cyl):
        return _set_biot_savart_points_cyl(self, points_cyl)

    def get_points_cart_ref(self):
        """Return the current JAX point buffer for point-preserving callers."""
        return self._points_jax

    def get_points_cart(self):
        return host_array(self._points_jax, dtype=np.float64)

    def get_points_cyl(self):
        return _get_biot_savart_points_cyl(self)

    def set_points_from_spec(self, field_eval_spec):
        """Set evaluation points from an immutable field-evaluation spec.

        This still mutates the receiving ``BiotSavartJAX`` instance.
        """
        return _set_biot_savart_points(self, field_eval_spec.points)

    def field_eval_spec(self):
        """Build the immutable field-evaluation spec for the current points."""
        return make_field_eval_spec(self._points_jax)

    def _base_curve_geometry_with_timings(self, curve, geometry_cache=None):
        cache_key = id(curve)
        if geometry_cache is not None and cache_key in geometry_cache:
            base_gamma, base_gammadash = geometry_cache[cache_key]
            return base_gamma, base_gammadash, 0.0

        if _supports_native_curve_geometry(curve):
            curve_dofs = _curve_live_dofs(curve)
            geometry_s, (base_gamma, base_gammadash) = _time_call_result(
                lambda: _curve_gamma_and_dash_from_dofs(curve, curve_dofs)
            )
        else:
            raise TypeError(
                "BiotSavartJAX requires curves that expose immutable JAX specs; "
                f"unsupported type {type(curve).__name__}. "
                "Provide a native curve spec."
            )

        if geometry_cache is not None:
            geometry_cache[cache_key] = (base_gamma, base_gammadash)
        return base_gamma, base_gammadash, geometry_s

    def _coil_has_free_dofs(self, coil):
        return coil.curve.dof_size > 0 or coil.current.dof_size > 0

    def _build_coil_vjp_info(self, coil_index, coil, inputs, *, timings=None):
        curve, rotmat, current, scale, gamma, gammadash, current_value = inputs
        return _CoilVJPInfo(
            coil_index=int(coil_index),
            coil=coil,
            curve=curve,
            rotmat=rotmat,
            current=current,
            scale=scale,
            gamma=gamma,
            gammadash=gammadash,
            current_value=current_value,
            native_curve=_supports_native_curve_geometry(curve),
            timings=timings,
        )

    def _group_coil_vjp_infos(self, coil_infos):
        grouped = {}
        for info in coil_infos:
            key = int(info.gamma.shape[0])
            grouped.setdefault(key, []).append(info)
        group_infos = []
        for infos in grouped.values():
            group_infos.append(
                {
                    "infos": infos,
                    "gammas": jnp.stack([info.gamma for info in infos]),
                    "gammadashs": jnp.stack([info.gammadash for info in infos]),
                    "currents": jnp.stack([info.current_value for info in infos]),
                    "native_curve": all(info.native_curve for info in infos),
                }
            )
        return group_infos

    def _collect_profiled_free_coil_vjp_infos(self, geometry_cache=None):
        coil_infos = []
        for coil_index, coil in enumerate(self._coils):
            if not self._coil_has_free_dofs(coil):
                continue
            *inputs, timings = self._coil_b_vjp_inputs(coil, geometry_cache)
            coil_infos.append(
                self._build_coil_vjp_info(
                    coil_index,
                    coil,
                    inputs,
                    timings=timings,
                )
            )
        return coil_infos

    def _extract_coil_data_grouped(self):
        """Read coil geometry grouped by quadrature point count.

        Read-only view over the explicit immutable grouped-coil state.

        Returns:
            list of ``(gammas, gammadashs, currents, coil_indices)``
            tuples, one per distinct quadrature count.
        """
        return list(grouped_field_data_from_spec(self.coil_set_spec()))

    def _coil_set_spec_from_explicit_state(self):
        if self._uses_uniform_curve_xyz_fourier_fastpath:
            return grouped_coil_set_spec_from_lists(
                *self._coil_arrays_in_order_from_dofs(_as_jax_float64(self.x))
            )
        return self.coil_set_spec_from_dofs(_as_jax_float64(self.x))

    def _free_coil_set_spec_from_explicit_state(self):
        groups = []
        for group in self._coil_set_spec_from_explicit_state().groups:
            free_positions = tuple(
                position
                for position, coil_index in enumerate(group.coil_indices)
                if self._coil_has_free_dofs(self._coils[coil_index])
            )
            if not free_positions:
                continue
            if len(free_positions) == len(group.coil_indices):
                groups.append(group)
                continue
            indexer = jnp.asarray(free_positions, dtype=jnp.int32)
            groups.append(
                (
                    group.gammas[indexer],
                    group.gammadashs[indexer],
                    group.currents[indexer],
                    tuple(group.coil_indices[position] for position in free_positions),
                )
            )
        return make_grouped_coil_set_spec(groups)

    def coil_set_spec(self):
        """Build the grouped coil spec for the current coil graph.

        The path stays in immutable-spec space: reconstruct from the live
        free-DOF vector with the cached explicit grouped-spec contract.
        """
        return self._coil_set_spec_from_explicit_state()

    def coil_specs(self):
        """Build immutable per-coil specs from the live coil graph."""
        return self.coil_specs_from_dofs(_as_jax_float64(self.x))

    # ------------------------------------------------------------------
    # VJP (reverse-mode gradient w.r.t. coil DOFs)
    # ------------------------------------------------------------------

    def _coil_b_vjp_inputs(self, coil, geometry_cache=None):
        curve, rotmat, current, scale = _unwrap_coil_curve_and_current(coil)
        base_gamma, base_gammadash, geometry_s = self._base_curve_geometry_with_timings(
            curve, geometry_cache
        )
        gamma, gammadash = _rotate_curve_geometry(base_gamma, base_gammadash, rotmat)
        current_s, current_value = _time_call_result(
            lambda: _as_jax_float64(current.get_value() * scale)
        )
        timings = {
            "curve_geometry_s": geometry_s,
            "current_value_s": current_s,
        }
        return curve, rotmat, current, scale, gamma, gammadash, current_value, timings

    def B_pullback_native(self, v):
        r"""Return the native grouped cotangents for ``B``.

        This is the JAX-native pullback boundary. It returns cotangents with
        respect to grouped coil geometry/current arrays, without projecting
        them into SIMSOPT's public :class:`Derivative` object graph.
        """
        points = self._points_jax
        v_jax = _as_jax_float64(v)
        free_coil_set_spec = self._free_coil_set_spec_from_explicit_state()
        d_coil_arrays = tuple(
            biot_savart_B_vjp_maybe_collective(
                points,
                v_jax,
                group.gammas,
                group.gammadashs,
                group.currents,
            )
            for group in free_coil_set_spec.groups
        )
        return BiotSavartFieldPullback(
            d_coil_arrays=d_coil_arrays,
            coil_indices=free_coil_set_spec.coil_index_lists(),
        )

    def _pullback_to_derivative(self, pullback):
        return self.coil_cotangents_to_derivative(
            pullback.d_coil_arrays,
            pullback.coil_indices,
        )

    def B_vjp(self, v):
        r"""Vector-Jacobian product of B w.r.t. coil DOFs.

        Given a cotangent vector ``v`` (typically ``dJ/dB``), returns
        a :class:`Derivative` mapping every free coil DOF to its
        contribution to the scalar objective.

        Uses ``jax.vjp`` through the pure Biot-Savart kernel, then
        projects each coil's geometry/current cotangents through immutable
        curve specs. Unsupported curves are rejected explicitly.

        Args:
            v: (npoints, 3) cotangent, same shape as ``B()``.

        Returns:
            :class:`Derivative` (sum over all coils).
        """
        return self._pullback_to_derivative(self.B_pullback_native(v))

    def _field_pullback_native(
        self,
        grouped_forward,
        cotangent,
    ):
        free_coil_set_spec = self._free_coil_set_spec_from_explicit_state()
        coil_arrays = free_coil_set_spec.field_inputs()
        if not coil_arrays:
            return BiotSavartFieldPullback((), ())

        _, pullback = jax.vjp(
            lambda grouped_inputs: grouped_forward(self._points_jax, grouped_inputs),
            coil_arrays,
        )
        d_coil_arrays = pullback(_as_jax_float64(cotangent))[0]
        return BiotSavartFieldPullback(
            d_coil_arrays=tuple(d_coil_arrays),
            coil_indices=free_coil_set_spec.coil_index_lists(),
        )

    def A_pullback_native(self, v):
        r"""Return native grouped cotangents for ``A``."""
        return self._field_pullback_native(grouped_biot_savart_A_from_inputs, v)

    def dA_by_dX_pullback_native(self, vgrad):
        r"""Return native grouped cotangents for ``dA/dX``."""
        return self._field_pullback_native(
            grouped_biot_savart_dA_by_dX_from_inputs,
            vgrad,
        )

    def dB_by_dX_pullback_native(self, vgrad):
        r"""Return native grouped cotangents for ``dB/dX``."""
        return self._field_pullback_native(
            grouped_biot_savart_dB_by_dX_from_inputs,
            vgrad,
        )

    def A_and_dA_pullback_native(self, v, vgrad):
        r"""Return separate native grouped cotangents for ``A`` and ``dA/dX``."""
        return (
            self.A_pullback_native(v),
            self.dA_by_dX_pullback_native(vgrad),
        )

    def B_and_dB_pullback_native(self, v, vgrad):
        r"""Return separate native grouped cotangents for ``B`` and ``dB/dX``."""
        return (
            self.B_pullback_native(v),
            self.dB_by_dX_pullback_native(vgrad),
        )

    def A_vjp(self, v):
        r"""Vector-Jacobian product of A w.r.t. coil DOFs."""
        return self._pullback_to_derivative(self.A_pullback_native(v))

    def A_and_dA_vjp(self, v, vgrad):
        r"""Separate vector-Jacobian products for A and dA/dX."""
        a_pullback, da_pullback = self.A_and_dA_pullback_native(v, vgrad)
        return (
            self._pullback_to_derivative(a_pullback),
            self._pullback_to_derivative(da_pullback),
        )

    def B_and_dB_vjp(self, v, vgrad):
        r"""Separate vector-Jacobian products for B and dB/dX."""
        b_pullback, db_pullback = self.B_and_dB_pullback_native(v, vgrad)
        return (
            self._pullback_to_derivative(b_pullback),
            self._pullback_to_derivative(db_pullback),
        )

    def _add_single_coil_cotangent_to_dofs_gradient(
        self,
        dofs_gradient,
        coil,
        dg,
        dgd,
        dc,
        coil_dofs,
    ):
        curve, rotmat, current, scale = _unwrap_coil_curve_and_current(coil)
        _require_native_curve_geometry(curve)

        if rotmat is not None:
            rotmat_t = _as_jax_float64(rotmat).T
            dg = _as_jax_float64(dg) @ rotmat_t
            dgd = _as_jax_float64(dgd) @ rotmat_t

        coeff_cotangent, surface_cotangent = curve_pullback_from_dofs(
            curve_spec_from_adapter_curve(curve),
            self._curve_dofs_from_free_vector(curve, coil_dofs),
            dg,
            dgd,
        )
        if _curve_dof_mode(curve) == "full":
            dofs_gradient = _add_full_curve_cotangent_to_dofs_gradient(
                dofs_gradient,
                curve,
                coeff_cotangent,
                self.dof_indices,
                free_positions_for_opt=self._local_free_positions,
            )
        else:
            dofs_gradient = _add_local_cotangent_to_dofs_gradient(
                dofs_gradient,
                curve,
                coeff_cotangent,
                self.dof_indices,
                free_positions=self._local_free_positions(curve),
            )
            if surface_cotangent is not None and curve.surf in self.dof_indices:
                dofs_gradient = _add_local_cotangent_to_dofs_gradient(
                    dofs_gradient,
                    curve.surf,
                    surface_cotangent,
                    self.dof_indices,
                    free_positions=self._local_free_positions(curve.surf),
                )

        if current.dof_size > 0:
            dofs_gradient = _add_local_cotangent_to_dofs_gradient(
                dofs_gradient,
                current,
                jnp.atleast_1d(_as_jax_float64(scale) * _as_jax_float64(dc)),
                self.dof_indices,
                free_positions=self._local_free_positions(current),
            )
        return dofs_gradient

    def coil_cotangents_to_dofs_gradient(
        self,
        d_coil_arrays,
        coil_indices,
        *,
        coil_dofs=None,
    ):
        """Project grouped coil cotangents to the flat free-DOF gradient."""
        if coil_dofs is None:
            coil_dofs = self.x.copy()
        coil_dofs = self._normalize_explicit_coil_dofs(coil_dofs)
        if _coil_cotangent_arrays_are_jax_compatible(d_coil_arrays):
            extraction_spec = _place_array_tree_on_device(
                self._coil_dof_extraction_spec,
                coil_dofs.device,
            )
            return _jitted_coil_cotangents_to_dofs_gradient(
                extraction_spec,
                d_coil_arrays,
                _canonical_coil_indices(coil_indices),
                coil_dofs,
            )

        dofs_gradient = coil_dofs - coil_dofs
        for (d_g, d_gd, d_c), indices in zip(d_coil_arrays, coil_indices):
            for local_i, global_i in enumerate(indices):
                dofs_gradient = self._add_single_coil_cotangent_to_dofs_gradient(
                    dofs_gradient,
                    self._coils[global_i],
                    jax.lax.index_in_dim(d_g, local_i, axis=0, keepdims=False),
                    jax.lax.index_in_dim(d_gd, local_i, axis=0, keepdims=False),
                    jax.lax.index_in_dim(d_c, local_i, axis=0, keepdims=False),
                    coil_dofs,
                )
        return dofs_gradient

    def coil_cotangents_to_derivative(self, d_coil_arrays, coil_indices):
        """Project grouped coil cotangent arrays to a :class:`Derivative`.

        Curves are projected through immutable specs into the flat field
        free-DOF layout, then converted once into the public ``Derivative``.

        Args:
            d_coil_arrays: list of ``(d_gammas, d_gammadashs, d_currents)``
                cotangent tuples, one per quadrature group.
            coil_indices: list of index lists, one per group, mapping
                local position to global coil index.

        Returns:
            :class:`Derivative` over all coil DOFs.
        """
        if not _coil_cotangent_arrays_are_jax_compatible(d_coil_arrays) or not hasattr(
            self, "_coil_dof_extraction_spec"
        ):
            return project_coil_cotangents_to_derivative(
                self._coils,
                d_coil_arrays,
                coil_indices,
            )

        canonical_indices = _canonical_coil_indices(coil_indices)
        dofs_gradient = self.coil_cotangents_to_dofs_gradient(
            d_coil_arrays,
            canonical_indices,
        )
        derivative = self.dofs_gradient_to_derivative(dofs_gradient)
        external_surfaces = getattr(self, "_external_cotangent_surfaces", ())
        if not external_surfaces:
            return derivative

        surface_cotangents = _jitted_external_surface_cotangents(
            self._coil_dof_extraction_spec,
            d_coil_arrays,
            canonical_indices,
            self._normalize_explicit_coil_dofs(self.x.copy()),
        )
        derivative_data = dict(derivative.data)
        for surface, surface_cotangent in zip(
            external_surfaces,
            surface_cotangents,
            strict=True,
        ):
            if surface in derivative_data:
                derivative_data[surface] = derivative_data[surface] + surface_cotangent
            else:
                derivative_data[surface] = surface_cotangent
        return Derivative(derivative_data)

    def dofs_gradient_to_derivative(self, dofs_gradient):
        return dofs_gradient_to_derivative(self.unique_dof_lineage, dofs_gradient)

    def profile_B_vjp(self, v):
        """Return a timing breakdown for ``B_vjp`` at the current points."""
        points = self._points_jax
        v_jax = _as_jax_float64(v)
        geometry_cache = {}
        component_totals = {
            "curve_geometry_s": 0.0,
            "current_value_s": 0.0,
            "single_coil_pullback_s": 0.0,
            "coil_projection_s": 0.0,
        }
        pullback_group_timings = []
        per_coil_timings = [
            _build_coil_profile_entry(
                coil_index,
                _zero_profile_component_timings(component_totals),
            )
            for coil_index, coil in enumerate(self._coils)
            if not self._coil_has_free_dofs(coil)
        ]
        wall_start = time.perf_counter()
        prep_start = time.perf_counter()
        coil_infos = self._collect_profiled_free_coil_vjp_infos(geometry_cache)
        grouped_infos = self._group_coil_vjp_infos(coil_infos)
        coil_dofs = self._normalize_explicit_coil_dofs(self.x.copy())
        dofs_gradient = coil_dofs - coil_dofs
        prep_s = float(time.perf_counter() - prep_start)
        free_coil_indices = [info.coil_index for info in coil_infos]
        component_totals["single_coil_pullback_s"] += prep_s
        if prep_s > 0.0 and free_coil_indices:
            pullback_group_timings.append(
                _build_pullback_group_profile_entry(
                    kind="prep",
                    coil_indices=free_coil_indices,
                    elapsed_s=prep_s,
                    native_curve=False,
                )
            )
        for group in grouped_infos:
            pullback_s, (dg_group, dgd_group, dc_group) = _time_call_result(
                lambda: biot_savart_B_vjp_maybe_collective(
                    points,
                    v_jax,
                    group["gammas"],
                    group["gammadashs"],
                    group["currents"],
                )
            )
            component_totals["single_coil_pullback_s"] += pullback_s
            pullback_group_timings.append(
                _build_pullback_group_profile_entry(
                    kind="group_pullback",
                    coil_indices=[info.coil_index for info in group["infos"]],
                    elapsed_s=pullback_s,
                    native_curve=group["native_curve"],
                )
            )
            for local_i, info in enumerate(group["infos"]):
                coil_projection_s, dofs_gradient = _time_call_result(
                    lambda: self._add_single_coil_cotangent_to_dofs_gradient(
                        dofs_gradient,
                        info.coil,
                        dg_group[local_i],
                        dgd_group[local_i],
                        dc_group[local_i],
                        coil_dofs,
                    )
                )
                component_totals["coil_projection_s"] += coil_projection_s
                coil_timings = dict(info.timings)
                coil_timings.update(
                    {
                        "single_coil_pullback_s": 0.0,
                        "coil_projection_s": coil_projection_s,
                    }
                )
                for name in ("curve_geometry_s", "current_value_s"):
                    component_totals[name] += coil_timings[name]
                per_coil_timings.append(
                    _build_coil_profile_entry(info.coil_index, coil_timings)
                )
        wall_time_s = float(time.perf_counter() - wall_start)
        return {
            "wall_time_s": wall_time_s,
            "component_timings_s": {
                name: float(elapsed_s) for name, elapsed_s in component_totals.items()
            },
            "dominant_components": _build_profile_breakdown(component_totals),
            "per_coil_timings_s": per_coil_timings,
            "dominant_coils": _build_coil_profile_breakdown(per_coil_timings),
            "pullback_group_timings_s": pullback_group_timings,
            "dominant_pullback_groups": _build_pullback_group_profile_breakdown(
                pullback_group_timings
            ),
        }
