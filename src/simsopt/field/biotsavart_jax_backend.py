"""JAX-backed Biot-Savart adapter and coil-tree helpers.

``BiotSavartJAX`` participates in the ``Optimizable`` dependency graph
through its coil list while computing the magnetic field via the pure
JAX kernels in :mod:`simsopt.field.biotsavart_jax`.

This module does **not** inherit from ``sopp.BiotSavart`` or
``sopp.MagneticField`` — it is a parallel JAX-native class per the
M0 rewrite contract (adapter pattern, §5).
"""

from dataclasses import dataclass
import time

import jax
import jax.numpy as jnp
import numpy as np

from .._core.derivative import Derivative
from .._core.jax_host_boundary import host_array, host_float
from .._core.optimizable import Optimizable
from ..jax_core import (
    coil_set_spec_from_dof_extraction_spec,
    coil_specs_from_dof_extraction_spec,
    curve_gamma_and_dash_from_dofs as curve_gamma_and_dash_from_spec_dofs,
    curve_gamma_and_dash_from_spec,
    curve_geometry_from_spec,
    curve_pullback_from_dofs,
    curve_spec_from_curve,
    make_coil_dof_extraction_spec,
    make_coil_set_dof_extraction_spec,
    make_optimizable_dof_map_spec,
)
from ..jax_core.field import (
    biot_savart_B_vjp_maybe_collective,
    grouped_biot_savart_A_from_inputs,
    grouped_biot_savart_A_from_spec,
    grouped_biot_savart_B_and_dB_from_spec,
    grouped_biot_savart_B_from_spec,
    grouped_biot_savart_d2A_by_dXdX_from_spec,
    grouped_biot_savart_dA_by_dX_from_inputs,
    grouped_biot_savart_dA_by_dX_from_spec,
    grouped_biot_savart_dB_by_dX_from_inputs,
    grouped_biot_savart_dB_by_dX_from_spec,
    grouped_coil_set_spec_from_coil_specs,
    grouped_field_data_from_spec,
    grouped_field_inputs_from_spec,
)
from ..jax_core.specs import (
    CoilSetDofExtractionSpec,
    CoilSpec,
    CoilSymmetrySpec,
    CurveSpec,
    FieldEvalSpec,
    GroupedCoilSetSpec,
    SingleStageRuntimeSpec,
    make_field_eval_spec,
)
from ._coil_graph import _unwrap_coil_curve_and_current_objects

__all__ = [
    "BiotSavartJAX",
    "BiotSavartBPullback",
    "BiotSavartFieldPullback",
    "SingleStageRuntimeSpecBiotSavartJAX",
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
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return
    if isinstance(value, Derivative):
        _block_until_ready(value.data)
        return
    if isinstance(value, dict):
        for dict_value in value.values():
            _block_until_ready(dict_value)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _block_until_ready(item)
        return
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


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


BiotSavartBPullback = BiotSavartFieldPullback


class SpecBackedCurrent:
    """Minimal current view reconstructed from a serialized JAX seed spec."""

    def __init__(self, value: float) -> None:
        self._value = float(value)
        self.local_lower_bounds = np.asarray([-np.inf], dtype=np.float64)
        self.local_upper_bounds = np.asarray([np.inf], dtype=np.float64)

    def get_value(self) -> float:
        return self._value


class SpecBackedCurve(Optimizable):
    """Read-only curve view backed by an immutable curve spec."""

    return_fn_map = {}

    def __init__(
        self,
        curve_spec: CurveSpec,
        symmetry: CoilSymmetrySpec,
    ) -> None:
        self._curve_spec = curve_spec
        self._symmetry = symmetry
        self.quadpoints = host_array(curve_spec.quadpoints, dtype=np.float64)
        Optimizable.__init__(self, x0=np.asarray([], dtype=np.float64))

    def to_spec(self) -> CurveSpec:
        return self._curve_spec

    def get_dofs(self) -> jax.Array:
        return self._curve_spec.dofs

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
        return self._apply_symmetry(*curve_geometry_from_spec(self._curve_spec))

    def gamma(self) -> np.ndarray:
        gamma, _gammadash = curve_gamma_and_dash_from_spec(self._curve_spec)
        return host_array(self._apply_symmetry(gamma)[0], dtype=np.float64)

    def gammadash(self) -> np.ndarray:
        gamma, gammadash = curve_gamma_and_dash_from_spec(self._curve_spec)
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

    def dgamma_by_dcoeff_vjp(self, _v: object) -> Derivative:
        return Derivative({})

    def dgammadash_by_dcoeff_vjp(self, _v: object) -> Derivative:
        return Derivative({})

    def dincremental_arclength_by_dcoeff_vjp(self, _v: object) -> Derivative:
        return Derivative({})

    def dkappa_by_dcoeff_vjp(self, _v: object) -> Derivative:
        return Derivative({})


class SpecBackedCoil:
    """Read-only coil view reconstructed from a serialized JAX seed spec."""

    def __init__(self, coil_spec: CoilSpec) -> None:
        self.curve = SpecBackedCurve(coil_spec.curve, coil_spec.symmetry)
        self.current = SpecBackedCurrent(
            host_float(coil_spec.current.value[0]) * float(coil_spec.symmetry.scale)
        )


class SingleStageRuntimeSpecBiotSavartJAX:
    """Biot-Savart adapter whose source of truth is a runtime seed spec."""

    def __init__(self, runtime_spec: SingleStageRuntimeSpec) -> None:
        self.runtime_spec = runtime_spec
        self._coil_dof_extraction_spec = runtime_spec.seed.coil_dof_extraction
        self._x = _as_jax_float64(runtime_spec.seed.coil_dofs)
        self._points_jax: jax.Array | None = None
        self._points_version = 0
        self._coils = self._coils_from_dofs(self._x)

    def _coils_from_dofs(
        self,
        coil_dofs: object,
    ) -> tuple[SpecBackedCoil, ...]:
        return tuple(
            SpecBackedCoil(coil_spec)
            for coil_spec in coil_specs_from_dof_extraction_spec(
                self._coil_dof_extraction_spec,
                coil_dofs,
            )
        )

    @property
    def x(self) -> jax.Array:
        return self._x

    @x.setter
    def x(self, coil_dofs: object) -> None:
        self._x = _as_jax_float64(coil_dofs)
        self._coils = self._coils_from_dofs(self._x)

    @property
    def coils(self) -> tuple[SpecBackedCoil, ...]:
        return self._coils

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

    def set_points(self, points: object) -> None:
        self._points_jax = _as_jax_float64(points)
        self._points_version += 1

    def set_points_from_spec(self, field_eval_spec: FieldEvalSpec) -> None:
        self._points_jax = _as_jax_float64(field_eval_spec.points)
        self._points_version += 1

    def field_eval_spec(self) -> FieldEvalSpec:
        return make_field_eval_spec(self._points_jax)

    def B(self) -> jax.Array:
        return grouped_biot_savart_B_from_spec(
            self._points_jax,
            self.coil_set_spec(),
        )

    def A(self) -> jax.Array:
        return grouped_biot_savart_A_from_spec(
            self._points_jax,
            self.coil_set_spec(),
        )

    def dA_by_dX(self) -> jax.Array:
        return grouped_biot_savart_dA_by_dX_from_spec(
            self._points_jax,
            self.coil_set_spec(),
        )

    def d2A_by_dXdX(self) -> jax.Array:
        return grouped_biot_savart_d2A_by_dXdX_from_spec(
            self._points_jax,
            self.coil_set_spec(),
        )

    def dB_by_dX(self) -> jax.Array:
        return grouped_biot_savart_dB_by_dX_from_spec(
            self._points_jax,
            self.coil_set_spec(),
        )

    def B_and_dB(self) -> tuple[jax.Array, jax.Array]:
        return grouped_biot_savart_B_and_dB_from_spec(
            self._points_jax,
            self.coil_set_spec(),
        )

    def save(self, _path: object) -> None:
        raise RuntimeError("JAX runtime seed specs split runtime from host export.")


def _supports_native_curve_geometry(curve):
    if callable(getattr(curve, "to_spec", None)):
        return True

    surface = getattr(curve, "surf", None)
    return (
        surface is not None
        and getattr(curve, "surf_type", None) == "RZ_Fourier"
        and callable(getattr(surface, "surface_spec", None))
    )


def _curve_dof_mode(curve):
    return getattr(curve, "_jax_curve_dof_mode", "local")


def _as_jax_float64(value) -> jax.Array:
    if isinstance(value, jax.Array):
        return jnp.asarray(value, dtype=jnp.float64)
    if isinstance(value, (np.ndarray, np.generic, list, tuple)) or np.isscalar(value):
        return jax.device_put(np.asarray(value, dtype=np.float64))
    return jnp.asarray(value, dtype=jnp.float64)


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
    return array[int(start) : int(end)]


def _axis0_entries(array):
    """Yield axis-0 slices without materializing the whole grouped object."""
    shape = getattr(array, "shape", None)
    if shape is not None:
        for index in range(int(shape[0])):
            yield array[index]
        return

    index = 0
    while True:
        try:
            yield array[index]
        except IndexError:
            return
        index += 1


def _update_1d(array: jax.Array, start: int, values: jax.Array) -> jax.Array:
    return array.at[int(start) : int(start) + values.shape[0]].set(values)


def _ones_like_float64(array: jax.Array) -> jax.Array:
    return jnp.broadcast_to(
        jax.device_put(np.array(1.0, dtype=np.float64)),
        array.shape,
    )


def _scatter_free_values(template: jax.Array, free_positions, free_values: jax.Array):
    free_positions = np.asarray(free_positions, dtype=np.int64)
    return template.at[free_positions].set(free_values)


def _add_local_cotangent_to_dofs_gradient(
    dofs_gradient: jax.Array,
    opt,
    full_cotangent,
    dof_indices,
):
    if opt.local_dof_size == 0:
        return dofs_gradient
    start, end = dof_indices[opt]
    free_positions = np.flatnonzero(opt.local_dofs_free_status)
    free_cotangent = _as_jax_float64(full_cotangent)[free_positions]
    return dofs_gradient.at[int(start) : int(end)].add(free_cotangent)


def _add_full_curve_cotangent_to_dofs_gradient(
    dofs_gradient: jax.Array,
    curve,
    full_cotangent,
    dof_indices,
):
    full_cotangent = _as_jax_float64(full_cotangent)
    for opt, (start, end) in curve._full_dof_indices.items():
        dofs_gradient = _add_local_cotangent_to_dofs_gradient(
            dofs_gradient,
            opt,
            _slice_1d(full_cotangent, start, end),
            dof_indices,
        )
    return dofs_gradient


def _curve_pullback_data_from_spec(curve, dg, dgd):
    spec = curve_spec_from_curve(curve)
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
        curve_spec_from_curve(curve),
        curve_dofs,
    )


def _project_single_coil_cotangent_data(coil, dg, dgd, dc):
    curve, rotmat, current, scale = _unwrap_coil_curve_and_current(coil)
    if not _supports_native_curve_geometry(curve):
        raise TypeError(
            "BiotSavartJAX coil cotangent projection requires immutable JAX "
            f"curve specs; unsupported type {type(curve).__name__}. "
            "Provide a native curve spec."
        )

    if rotmat is not None:
        rotmat_t = _as_jax_float64(rotmat).T
        dg = _as_jax_float64(dg) @ rotmat_t
        dgd = _as_jax_float64(dgd) @ rotmat_t

    deriv_data = {}
    _merge_curve_pullback_data(deriv_data, curve, dg, dgd)
    if current.dof_size > 0:
        current_cotangent = jnp.atleast_1d(
            _as_jax_float64(scale) * _as_jax_float64(dc)
        )
        _merge_derivative_data(deriv_data, current.vjp(current_cotangent))
    return deriv_data


def project_coil_cotangents_to_derivative(coils, d_coil_arrays, coil_indices):
    """Project grouped coil cotangents to a single public ``Derivative``."""
    deriv_data = {}
    for (d_g, d_gd, d_c), indices in zip(d_coil_arrays, coil_indices):
        for dg_i, dgd_i, dc_i, global_i in zip(
            _axis0_entries(d_g),
            _axis0_entries(d_gd),
            _axis0_entries(d_c),
            indices,
        ):
            _merge_derivative_data(
                deriv_data,
                _project_single_coil_cotangent_data(
                    coils[global_i],
                    dg_i,
                    dgd_i,
                    dc_i,
                ),
            )
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


class BiotSavartJAX(Optimizable):
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

    def __init__(self, coils):
        self._coils = list(coils)
        self._points_jax = None
        self._points_version = 0
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=self._coils)

        # Uniform CurveXYZFourier fast-path metadata (populated by _introspect_coils)
        self._uses_uniform_curve_xyz_fourier_fastpath = False
        self._unique_base_curves = []
        self._unique_base_currents = []
        self._coil_descs = []  # list of (curve_idx, current_idx, rotmat_jax, scale)
        self._curve_order = 0
        self._curve_dof_size = 0
        self._curve_quadpoints_jax = None
        self._introspect_coils()
        self._coil_dof_extraction_spec = self._build_coil_dof_extraction_spec()

    def _introspect_coils(self):
        """Walk coil tree to identify unique base curves/currents.

        Enables the JAX-native path when all curves are
        ``CurveXYZFourier`` (possibly wrapped in ``RotatedCurve``)
        with uniform Fourier order and quadrature point count.
        """
        from ..geo.curvexyzfourier import CurveXYZFourier
        from .coil import Current

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
        self._curve_dof_size = 3 * (2 * self._curve_order + 1)
        self._curve_quadpoints_jax = _curve_quadpoints_jax(base_curves[0])

    def _build_coil_dof_extraction_spec(self):
        return make_coil_set_dof_extraction_spec(
            make_coil_dof_extraction_spec(
                curve=curve_spec_from_curve(curve),
                curve_map=self._free_vector_dof_map_spec(
                    curve,
                    full_graph=_curve_dof_mode(curve) == "full",
                ),
                current_map=self._free_vector_dof_map_spec(
                    current,
                    full_graph=False,
                ),
                rotmat=rotmat,
                scale=scale,
            )
            for curve, rotmat, current, scale in (
                _unwrap_coil_curve_and_current(coil) for coil in self._coils
            )
        )

    def coil_dof_extraction_spec(self):
        """Return the cached immutable owner-DOF reconstruction contract."""
        return self._coil_dof_extraction_spec

    def _coil_arrays_in_order_from_dofs_generic_jax(self, coil_dofs):
        """Rebuild per-coil arrays for curve sets with immutable JAX specs."""
        from .coil import Current

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
        free_positions = np.flatnonzero(opt.local_dofs_free_status)
        coil_slice = _slice_1d(coil_dofs, start, end)
        return _scatter_free_values(full_x, free_positions, coil_slice)

    def _full_dofs_from_free_vector(self, opt, coil_dofs):
        """Rebuild one Optimizable graph's full DOF vector from ``coil_dofs``."""
        full_x = _as_jax_float64(opt.full_x)
        for dep_opt, (start, end) in opt._full_dof_indices.items():
            dep_full_x = _as_jax_float64(dep_opt.local_full_x)
            if dep_opt.local_dof_size > 0:
                dep_start, dep_end = self.dof_indices[dep_opt]
                free_positions = np.flatnonzero(dep_opt.local_dofs_free_status)
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
                np.flatnonzero(opt.local_dofs_free_status)
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
        from ..geo.curvexyzfourier import jaxfouriercurve_pure

        coil_dofs = self._normalize_explicit_coil_dofs(coil_dofs)

        quadpoints = self._curve_quadpoints_jax
        ones = _ones_like_float64(quadpoints)

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
            base_gammas.append(
                jaxfouriercurve_pure(curve_x, quadpoints, self._curve_order)
            )
            base_gammadashs.append(
                jax.jvp(
                    lambda qpts, cx=curve_x: jaxfouriercurve_pure(
                        cx,
                        qpts,
                        self._curve_order,
                    ),
                    (quadpoints,),
                    (ones,),
                )[1]
            )

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
        self._points_jax = _as_jax_float64(points_array)
        self._points_version += 1

    def set_points_from_spec(self, field_eval_spec):
        """Set evaluation points from an immutable field-evaluation spec.

        This still mutates the receiving ``BiotSavartJAX`` instance.
        """
        self._points_jax = _as_jax_float64(field_eval_spec.points)
        self._points_version += 1

    def field_eval_spec(self):
        """Build the immutable field-evaluation spec for the current points."""
        return make_field_eval_spec(self._points_jax)

    def _base_curve_geometry(self, curve, geometry_cache=None):
        gamma, gammadash, _ = self._base_curve_geometry_with_timings(
            curve,
            geometry_cache,
        )
        return gamma, gammadash

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

    def _coil_geometry_inputs(self, coil, geometry_cache=None):
        curve, rotmat, current, scale = _unwrap_coil_curve_and_current(coil)
        gamma, gammadash = self._base_curve_geometry(curve, geometry_cache)
        gamma, gammadash = _rotate_curve_geometry(gamma, gammadash, rotmat)
        current_value = _as_jax_float64(current.get_value() * scale)
        return curve, rotmat, current, scale, gamma, gammadash, current_value

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

    def _collect_free_coil_vjp_infos(self, geometry_cache=None):
        coil_infos = []
        for coil_index, coil in enumerate(self._coils):
            if not self._coil_has_free_dofs(coil):
                continue
            inputs = self._coil_geometry_inputs(coil, geometry_cache)
            coil_infos.append(
                self._build_coil_vjp_info(
                    coil_index,
                    coil,
                    inputs,
                )
            )
        return coil_infos

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
        return self._coil_set_spec_from_dofs_immutable_specs(_as_jax_float64(self.x))

    def coil_set_spec(self):
        """Build the grouped coil spec for the current coil graph.

        The path stays in immutable-spec space: reconstruct from the live
        free-DOF vector with the cached explicit grouped-spec contract.
        """
        return self._coil_set_spec_from_explicit_state()

    def coil_specs(self):
        """Build immutable per-coil specs from the live coil graph."""
        return tuple(coil.to_spec() for coil in self._coils)

    # ------------------------------------------------------------------
    # Forward field evaluation
    # ------------------------------------------------------------------

    def B(self):
        """Magnetic field B at the evaluation points.

        Returns:
            (npoints, 3) JAX array.
        """
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
        """Spatial Jacobian dB/dX at the evaluation points.

        Returns:
            (npoints, 3, 3) JAX array where ``[p, j, l] = ∂_j B_l``.
        """
        return grouped_biot_savart_dB_by_dX_from_spec(
            self._points_jax,
            self.coil_set_spec(),
        )

    def B_and_dB(self):
        """Combined B and dB/dX (single JIT compilation).

        Returns:
            (B, dB_dX) with shapes (npoints, 3) and (npoints, 3, 3).
        """
        return grouped_biot_savart_B_and_dB_from_spec(
            self._points_jax,
            self.coil_set_spec(),
        )

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
        geometry_cache = {}
        d_coil_arrays = []
        coil_indices = []
        coil_infos = self._collect_free_coil_vjp_infos(geometry_cache)
        for group in self._group_coil_vjp_infos(coil_infos):
            d_coil_arrays.append(
                biot_savart_B_vjp_maybe_collective(
                    points,
                    v_jax,
                    group["gammas"],
                    group["gammadashs"],
                    group["currents"],
                )
            )
            coil_indices.append(tuple(info.coil_index for info in group["infos"]))
        return BiotSavartFieldPullback(
            d_coil_arrays=tuple(d_coil_arrays),
            coil_indices=tuple(coil_indices),
        )

    B_cotangents = B_pullback_native

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
        free_groups = self._group_coil_vjp_infos(
            self._collect_free_coil_vjp_infos(geometry_cache={})
        )
        coil_arrays = tuple(
            (group["gammas"], group["gammadashs"], group["currents"])
            for group in free_groups
        )
        if not coil_arrays:
            return BiotSavartFieldPullback((), ())
        coil_indices = tuple(
            tuple(info.coil_index for info in group["infos"]) for group in free_groups
        )

        _, pullback = jax.vjp(
            lambda grouped_inputs: grouped_forward(self._points_jax, grouped_inputs),
            coil_arrays,
        )
        d_coil_arrays = pullback(_as_jax_float64(cotangent))[0]
        return BiotSavartFieldPullback(
            d_coil_arrays=tuple(d_coil_arrays),
            coil_indices=coil_indices,
        )

    def A_pullback_native(self, v):
        r"""Return native grouped cotangents for ``A``."""
        return self._field_pullback_native(grouped_biot_savart_A_from_inputs, v)

    A_cotangents = A_pullback_native

    def dA_by_dX_pullback_native(self, vgrad):
        r"""Return native grouped cotangents for ``dA/dX``."""
        return self._field_pullback_native(
            grouped_biot_savart_dA_by_dX_from_inputs,
            vgrad,
        )

    dA_by_dX_cotangents = dA_by_dX_pullback_native

    def dB_by_dX_pullback_native(self, vgrad):
        r"""Return native grouped cotangents for ``dB/dX``."""
        return self._field_pullback_native(
            grouped_biot_savart_dB_by_dX_from_inputs,
            vgrad,
        )

    dB_by_dX_cotangents = dB_by_dX_pullback_native

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
        if not _supports_native_curve_geometry(curve):
            raise TypeError(
                "BiotSavartJAX coil cotangent projection requires immutable JAX "
                f"curve specs; unsupported type {type(curve).__name__}. "
                "Provide a native curve spec."
            )

        if rotmat is not None:
            rotmat_t = _as_jax_float64(rotmat).T
            dg = _as_jax_float64(dg) @ rotmat_t
            dgd = _as_jax_float64(dgd) @ rotmat_t

        coeff_cotangent, surface_cotangent = curve_pullback_from_dofs(
            curve_spec_from_curve(curve),
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
            )
        else:
            dofs_gradient = _add_local_cotangent_to_dofs_gradient(
                dofs_gradient,
                curve,
                coeff_cotangent,
                self.dof_indices,
            )
            if surface_cotangent is not None and curve.surf in self.dof_indices:
                dofs_gradient = _add_local_cotangent_to_dofs_gradient(
                    dofs_gradient,
                    curve.surf,
                    surface_cotangent,
                    self.dof_indices,
                )

        if current.dof_size > 0:
            dofs_gradient = _add_local_cotangent_to_dofs_gradient(
                dofs_gradient,
                current,
                jnp.atleast_1d(_as_jax_float64(scale) * _as_jax_float64(dc)),
                self.dof_indices,
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
        dofs_gradient = jnp.zeros_like(coil_dofs)
        for (d_g, d_gd, d_c), indices in zip(d_coil_arrays, coil_indices):
            for dg_i, dgd_i, dc_i, global_i in zip(
                _axis0_entries(d_g),
                _axis0_entries(d_gd),
                _axis0_entries(d_c),
                indices,
            ):
                dofs_gradient = self._add_single_coil_cotangent_to_dofs_gradient(
                    dofs_gradient,
                    self._coils[global_i],
                    dg_i,
                    dgd_i,
                    dc_i,
                    coil_dofs,
                )
        return dofs_gradient

    def coil_cotangents_to_derivative(self, d_coil_arrays, coil_indices):
        """Project grouped coil cotangent arrays to a :class:`Derivative`.

        Curves are projected through immutable specs; unsupported curves are
        rejected explicitly.

        Args:
            d_coil_arrays: list of ``(d_gammas, d_gammadashs, d_currents)``
                cotangent tuples, one per quadrature group.
            coil_indices: list of index lists, one per group, mapping
                local position to global coil index.

        Returns:
            :class:`Derivative` over all coil DOFs.
        """
        return project_coil_cotangents_to_derivative(
            self._coils,
            d_coil_arrays,
            coil_indices,
        )

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
            for dg_i, dgd_i, dc_i, info in zip(
                _axis0_entries(dg_group),
                _axis0_entries(dgd_group),
                _axis0_entries(dc_group),
                group["infos"],
            ):
                coil_projection_s, _ = _time_call_result(
                    lambda: _project_single_coil_cotangent_data(
                        info.coil,
                        dg_i,
                        dgd_i,
                        dc_i,
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
