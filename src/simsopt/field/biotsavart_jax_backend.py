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

import numpy as np
import jax
import jax.numpy as jnp

from ..backend import raise_if_strict_jax_fallback
from .._core.derivative import Derivative
from .._core.optimizable import Optimizable
from ..jax_core.field import (
    grouped_biot_savart_B_and_dB_from_spec,
    grouped_biot_savart_B_from_spec,
    grouped_biot_savart_dB_by_dX_from_spec,
    grouped_field_data_from_spec,
    grouped_field_inputs_from_spec,
    grouped_coil_set_spec_from_lists,
)
from .biotsavart_jax import (
    biot_savart_B,
    biot_savart_B_vjp,
)

__all__ = ["BiotSavartJAX"]


@jax.jit
def _single_coil_b_vjp(points, v, gamma, gammadash, current):
    """Reverse-mode pullback for one coil at a time.

    Keeping the reverse pass at single-coil granularity avoids materializing
    grouped multi-coil intermediates in device memory while still letting JAX
    cache one compiled VJP per distinct quadrature shape.
    """

    def fwd(g, gd, c):
        return biot_savart_B(points, g[None, ...], gd[None, ...], c[None])

    _, pullback = jax.vjp(fwd, gamma, gammadash, current)
    return pullback(v)


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


def _raise_if_strict_biot_savart_fallback(detail: str) -> None:
    raise_if_strict_jax_fallback(
        component="BiotSavartJAX",
        detail=detail,
    )


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


def _supports_native_curve_geometry(curve):
    from ..geo.curvexyzfourier import CurveXYZFourier

    return isinstance(curve, CurveXYZFourier) or (
        hasattr(curve, "gamma_jax") and hasattr(curve, "gammadash_jax")
    )


def _curve_dof_mode(curve):
    return getattr(curve, "_jax_curve_dof_mode", "local")


def _curve_live_dofs(curve):
    if _curve_dof_mode(curve) == "full":
        return jnp.asarray(curve.full_x, dtype=jnp.float64)
    return jnp.asarray(curve.get_dofs(), dtype=jnp.float64)


def _curve_quadpoints_jax(curve):
    return jnp.asarray(curve.quadpoints, dtype=jnp.float64)


def _curve_surface_dofs(curve):
    surf = getattr(curve, "surf", None)
    if surf is None or surf.dof_size == 0:
        return None
    return jnp.asarray(surf.get_dofs(), dtype=jnp.float64)


def _supports_jax_curve_pullback(curve):
    from ..geo.curvexyzfourier import CurveXYZFourier

    if isinstance(curve, CurveXYZFourier):
        return True

    if not (
        hasattr(curve, "dgamma_by_dcoeff_vjp_jax")
        and hasattr(curve, "dgammadash_by_dcoeff_vjp_jax")
    ):
        return False

    if _curve_surface_dofs(curve) is None:
        return True

    return hasattr(curve, "dgamma_by_dsurf_vjp_jax") and hasattr(
        curve, "dgammadash_by_dsurf_vjp_jax"
    )


def _supports_cpu_curve_pullback(curve):
    return hasattr(curve, "dgamma_by_dcoeff_vjp") and hasattr(
        curve,
        "dgammadash_by_dcoeff_vjp",
    )


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
    full_cotangent = jnp.asarray(full_cotangent, dtype=jnp.float64)
    deriv_data = {}
    for opt, (start, end) in curve._full_dof_indices.items():
        if opt.local_full_dof_size == 0:
            continue
        deriv_data[opt] = full_cotangent[start:end]
    return deriv_data


def _curve_coeff_pullback_data(curve, dg, dgd):
    from ..geo.curvexyzfourier import CurveXYZFourier, jaxfouriercurve_pure

    if curve.dof_size == 0:
        return {}

    if isinstance(curve, CurveXYZFourier):
        curve_dofs = jnp.asarray(curve.get_dofs(), dtype=jnp.float64)
        quadpoints = _curve_quadpoints_jax(curve)
        ones = jnp.ones_like(quadpoints)

        def gamma_of_dofs(dofs):
            return jaxfouriercurve_pure(dofs, quadpoints, curve.order)

        def gammadash_of_dofs(dofs):
            return jax.jvp(
                lambda qpts: jaxfouriercurve_pure(dofs, qpts, curve.order),
                (quadpoints,),
                (ones,),
            )[1]

        _, gamma_pullback = jax.vjp(gamma_of_dofs, curve_dofs)
        _, gammadash_pullback = jax.vjp(gammadash_of_dofs, curve_dofs)
        return {curve: gamma_pullback(dg)[0] + gammadash_pullback(dgd)[0]}

    curve_dofs = _curve_live_dofs(curve)
    coeff_cotangent = curve.dgamma_by_dcoeff_vjp_jax(
        curve_dofs,
        dg,
    ) + curve.dgammadash_by_dcoeff_vjp_jax(curve_dofs, dgd)
    if _curve_dof_mode(curve) == "full":
        return _full_curve_cotangent_to_derivative(curve, coeff_cotangent)
    return {curve: coeff_cotangent}


def _curve_gamma_from_dofs(curve, curve_dofs):
    from ..geo.curvexyzfourier import CurveXYZFourier, jaxfouriercurve_pure

    if isinstance(curve, CurveXYZFourier):
        quadpoints = _curve_quadpoints_jax(curve)
        return jaxfouriercurve_pure(curve_dofs, quadpoints, curve.order)

    if _curve_dof_mode(curve) == "full":
        return curve.gamma_jax(curve_dofs)

    surf_dofs = _curve_surface_dofs(curve)
    if surf_dofs is not None:
        return curve.gamma_jax(curve_dofs, surf_dofs)

    return curve.gamma_jax(curve_dofs)


def _curve_gammadash_from_dofs(curve, curve_dofs):
    from ..geo.curvexyzfourier import CurveXYZFourier, jaxfouriercurve_pure

    if isinstance(curve, CurveXYZFourier):
        quadpoints = _curve_quadpoints_jax(curve)
        ones = jnp.ones_like(quadpoints)
        return jax.jvp(
            lambda qpts: jaxfouriercurve_pure(curve_dofs, qpts, curve.order),
            (quadpoints,),
            (ones,),
        )[1]

    if _curve_dof_mode(curve) == "full":
        return curve.gammadash_jax(curve_dofs)

    surf_dofs = _curve_surface_dofs(curve)
    if surf_dofs is not None:
        return curve.gammadash_jax(curve_dofs, surf_dofs)

    return curve.gammadash_jax(curve_dofs)


def _curve_surface_pullback_data(curve, dg, dgd):
    surf_dofs = _curve_surface_dofs(curve)
    if surf_dofs is None:
        return {}
    if not (
        hasattr(curve, "dgamma_by_dsurf_vjp_jax")
        and hasattr(curve, "dgammadash_by_dsurf_vjp_jax")
    ):
        return {}

    return {
        curve.surf: curve.dgamma_by_dsurf_vjp_jax(surf_dofs, dg)
        + curve.dgammadash_by_dsurf_vjp_jax(surf_dofs, dgd)
    }


def _curve_coeff_pullback_data_cpu(curve, dg, dgd):
    deriv_data = {}
    _merge_derivative_data(deriv_data, curve.dgamma_by_dcoeff_vjp(np.asarray(dg)))
    _merge_derivative_data(
        deriv_data,
        curve.dgammadash_by_dcoeff_vjp(np.asarray(dgd)),
    )
    return deriv_data


def _project_single_coil_cotangent_data(coil, dg, dgd, dc):
    curve, rotmat, current, scale = _unwrap_coil_curve_and_current(coil)
    supports_jax_pullback = _supports_jax_curve_pullback(curve)
    supports_cpu_pullback = _supports_cpu_curve_pullback(curve)

    if rotmat is not None and (supports_jax_pullback or supports_cpu_pullback):
        rotmat_t = jnp.asarray(rotmat, dtype=jnp.float64).T
        dg = dg @ rotmat_t
        dgd = dgd @ rotmat_t

    if supports_jax_pullback:
        deriv_data = {}
        _merge_derivative_data(deriv_data, _curve_coeff_pullback_data(curve, dg, dgd))
        _merge_derivative_data(deriv_data, _curve_surface_pullback_data(curve, dg, dgd))
        if current.dof_size > 0:
            current_cotangent = jnp.atleast_1d(
                jnp.asarray(scale, dtype=jnp.float64)
                * jnp.asarray(dc, dtype=jnp.float64)
            )
            _merge_derivative_data(deriv_data, current.vjp(current_cotangent))
        return deriv_data

    if supports_cpu_pullback:
        _raise_if_strict_biot_savart_fallback(
            (f"the CPU coil-pullback fallback for curve type {type(curve).__name__}"),
        )
        deriv_data = _curve_coeff_pullback_data_cpu(curve, dg, dgd)
        if current.dof_size > 0:
            current_cotangent = np.atleast_1d(
                np.asarray(scale, dtype=float) * np.asarray(dc, dtype=float)
            )
            _merge_derivative_data(deriv_data, current.vjp(current_cotangent))
        return deriv_data

    raise TypeError(
        "Curve does not expose a supported JAX or CPU pullback contract for "
        "BiotSavartJAX coil cotangent projection."
    )


def project_coil_cotangents_to_derivative(coils, d_coil_arrays, coil_indices):
    """Project grouped coil cotangents to a single public ``Derivative``."""
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


def _rotate_curve_geometry(gamma, gammadash, rotmat):
    if rotmat is None:
        return gamma, gammadash
    return gamma @ rotmat, gammadash @ rotmat


def _unwrap_coil_curve_and_current(coil):
    from ..geo.curve import RotatedCurve
    from .coil import ScaledCurrent

    curve = coil.curve
    rotmat = None
    while isinstance(curve, RotatedCurve):
        next_rotmat = jnp.asarray(curve.rotmat)
        rotmat = next_rotmat if rotmat is None else next_rotmat @ rotmat
        curve = curve.curve

    current = coil.current
    scale = 1.0
    while isinstance(current, ScaledCurrent):
        scale *= float(current.scale)
        current = current.current_to_scale

    return curve, rotmat, current, scale


class BiotSavartJAX(Optimizable):
    r"""JAX-backed Biot-Savart magnetic field evaluation.

    Drop-in replacement for :class:`BiotSavart` in workflows where the
    field is consumed by a JAX-backed objective (e.g. ``SquaredFluxJAX``).

    The class holds no DOFs of its own.  Its ``Optimizable`` dependency
    chain runs through the coil list so that the outer framework
    correctly composes DOFs and derivatives.

    When all coils use ``CurveXYZFourier`` (possibly wrapped in
    ``RotatedCurve``), the JAX-native path is enabled: coil geometry
    is evaluated from DOFs via a precomputed Fourier basis matrix
    entirely inside the JIT boundary, eliminating CPU round-trips.

    Args:
        coils: list of :class:`simsopt.field.coil.Coil` objects.
    """

    def __init__(self, coils):
        self._coils = list(coils)
        self._points_jax = None
        self._points_version = 0
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=self._coils)

        # JAX-native path metadata (populated by _introspect_coils)
        self._jax_native = False
        self._unique_base_curves = []
        self._unique_base_currents = []
        self._coil_descs = []  # list of (curve_idx, current_idx, rotmat_jax, scale)
        self._curve_order = 0
        self._curve_dof_size = 0
        self._curve_quadpoints_jax = None
        self._introspect_coils()

    def _introspect_coils(self):
        """Walk coil tree to identify unique base curves/currents.

        Enables the JAX-native path when all curves are
        ``CurveXYZFourier`` (possibly wrapped in ``RotatedCurve``)
        with uniform Fourier order and quadrature point count.
        """
        try:
            from ..geo.curvexyzfourier import CurveXYZFourier
            from .coil import Current
        except ImportError:
            return

        base_curve_ids = {}  # id(obj) → index
        base_current_ids = {}
        base_curves = []
        base_currents = []
        descs = []

        for coil in self._coils:
            curve, rotmat, current, scale = _unwrap_coil_curve_and_current(coil)

            if not isinstance(curve, CurveXYZFourier):
                return  # unsupported curve type → stay on fallback path

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
                    jnp.asarray(rotmat) if rotmat is not None else None,
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

        self._jax_native = True
        self._unique_base_curves = base_curves
        self._unique_base_currents = base_currents
        self._coil_descs = descs
        self._curve_order = orders.pop()
        self._curve_dof_size = 3 * (2 * self._curve_order + 1)
        self._curve_quadpoints_jax = _curve_quadpoints_jax(base_curves[0])

    def _coil_arrays_in_order_from_dofs_generic_jax(self, coil_dofs):
        """Rebuild per-coil arrays for any JAX-geometry-capable curve set."""
        from .coil import Current

        coil_dofs = self._normalize_explicit_coil_dofs(coil_dofs)

        coil_gammas = []
        coil_gammadashs = []
        coil_currents = []
        for coil in self._coils:
            curve, rotmat, current, scale = _unwrap_coil_curve_and_current(coil)
            if not _supports_native_curve_geometry(curve):
                raise RuntimeError(
                    "grouped_coil_arrays_from_dofs() requires JAX geometry support "
                    f"for every base curve; unsupported type {type(curve).__name__}."
                )
            if not isinstance(current, Current):
                raise RuntimeError(
                    "grouped_coil_arrays_from_dofs() only supports scalar Current "
                    "degrees of freedom on the JAX geometry lane."
                )

            curve_dofs = self._curve_dofs_from_free_vector(curve, coil_dofs)
            gamma = _curve_gamma_from_dofs(curve, curve_dofs)
            gammadash = _curve_gammadash_from_dofs(curve, curve_dofs)
            if rotmat is not None:
                gamma = gamma @ rotmat
                gammadash = gammadash @ rotmat

            coil_gammas.append(gamma)
            coil_gammadashs.append(gammadash)
            coil_currents.append(
                jnp.asarray(scale, dtype=jnp.float64)
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
        full_x = jnp.asarray(opt.local_full_x, dtype=jnp.float64)
        if opt.local_dof_size == 0:
            return full_x

        start, end = self.dof_indices[opt]
        free_indices = np.flatnonzero(opt.local_dofs_free_status)
        return full_x.at[free_indices].set(coil_dofs[start:end])

    def _full_dofs_from_free_vector(self, opt, coil_dofs):
        """Rebuild one Optimizable graph's full DOF vector from ``coil_dofs``."""
        full_x = jnp.asarray(opt.full_x, dtype=jnp.float64)
        for dep_opt, (start, end) in opt._full_dof_indices.items():
            dep_full_x = jnp.asarray(dep_opt.local_full_x, dtype=jnp.float64)
            if dep_opt.local_dof_size > 0:
                dep_start, dep_end = self.dof_indices[dep_opt]
                free_indices = np.flatnonzero(dep_opt.local_dofs_free_status)
                dep_full_x = dep_full_x.at[free_indices].set(
                    coil_dofs[dep_start:dep_end]
                )
            full_x = full_x.at[start:end].set(dep_full_x)
        return full_x

    def _curve_dofs_from_free_vector(self, curve, coil_dofs):
        if _curve_dof_mode(curve) == "full":
            return self._full_dofs_from_free_vector(curve, coil_dofs)
        return self._local_full_dofs_from_free_vector(curve, coil_dofs)

    def _normalize_explicit_coil_dofs(self, coil_dofs):
        coil_dofs = jnp.asarray(coil_dofs, dtype=jnp.float64)
        expected_dofs = self.dof_size
        if coil_dofs.shape[0] != expected_dofs:
            raise ValueError(
                f"Expected {expected_dofs} coil DOFs, got {coil_dofs.shape[0]}."
            )
        return coil_dofs

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
        Other curves can still use this helper when they expose the
        JAX geometry hooks needed to rebuild per-coil arrays from DOFs.
        """
        if not self._jax_native:
            return self._coil_arrays_in_order_from_dofs_generic_jax(coil_dofs)
        from ..geo.curvexyzfourier import jaxfouriercurve_pure

        coil_dofs = self._normalize_explicit_coil_dofs(coil_dofs)

        quadpoints = self._curve_quadpoints_jax
        ones = jnp.ones_like(quadpoints)

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
                    lambda qpts: jaxfouriercurve_pure(
                        curve_x,
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
            coil_currents.append(
                jnp.asarray(scale, dtype=jnp.float64) * current_values[current_idx]
            )

        return coil_gammas, coil_gammadashs, coil_currents

    def grouped_coil_arrays_from_dofs(self, coil_dofs):
        """Build grouped coil arrays from an explicit flat DOF vector."""
        return list(
            grouped_field_inputs_from_spec(self.coil_set_spec_from_dofs(coil_dofs))
        )

    def coil_set_spec_from_dofs(self, coil_dofs):
        """Build an immutable grouped coil spec from an explicit flat DOF vector."""
        gammas, gammadashs, currents = self._coil_arrays_in_order_from_dofs(coil_dofs)
        return grouped_coil_set_spec_from_lists(gammas, gammadashs, currents)

    @property
    def coils(self):
        return self._coils

    def set_points(self, points):
        """Set evaluation points (converted to a JAX array once).

        Accepts both NumPy and JAX arrays.  JAX arrays stay on device
        without a host round-trip.
        """
        if isinstance(points, jax.Array):
            self._points_jax = points
        else:
            self._points_jax = jnp.asarray(np.ascontiguousarray(points))
        self._points_version += 1

    def _base_curve_geometry(self, curve, geometry_cache=None):
        gamma, gammadash, _, _ = self._base_curve_geometry_with_timings(
            curve,
            geometry_cache,
        )
        return gamma, gammadash

    def _base_curve_geometry_with_timings(self, curve, geometry_cache=None):
        cache_key = id(curve)
        if geometry_cache is not None and cache_key in geometry_cache:
            base_gamma, base_gammadash = geometry_cache[cache_key]
            return base_gamma, base_gammadash, 0.0, 0.0

        if _supports_native_curve_geometry(curve):
            curve_dofs = _curve_live_dofs(curve)
            gamma_s, base_gamma = _time_call_result(
                lambda: _curve_gamma_from_dofs(curve, curve_dofs)
            )
            gammadash_s, base_gammadash = _time_call_result(
                lambda: _curve_gammadash_from_dofs(curve, curve_dofs)
            )
        else:
            _raise_if_strict_biot_savart_fallback(
                (
                    f"the CPU curve-geometry fallback for curve type "
                    f"{type(curve).__name__}"
                ),
            )
            gamma_s, base_gamma = _time_call_result(
                lambda: jnp.asarray(curve.gamma(), dtype=jnp.float64)
            )
            gammadash_s, base_gammadash = _time_call_result(
                lambda: jnp.asarray(curve.gammadash(), dtype=jnp.float64)
            )

        if geometry_cache is not None:
            geometry_cache[cache_key] = (base_gamma, base_gammadash)
        return base_gamma, base_gammadash, gamma_s, gammadash_s

    def _coil_geometry_inputs(self, coil, geometry_cache=None):
        curve, rotmat, current, scale = _unwrap_coil_curve_and_current(coil)
        gamma, gammadash = self._base_curve_geometry(curve, geometry_cache)
        gamma, gammadash = _rotate_curve_geometry(gamma, gammadash, rotmat)
        current_value = jnp.asarray(current.get_value() * scale, dtype=jnp.float64)
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

        Compatibility wrapper over the immutable grouped-coil spec.

        Returns:
            list of ``(gammas, gammadashs, currents, coil_indices)``
            tuples, one per distinct quadrature count.
        """
        return list(grouped_field_data_from_spec(self.coil_set_spec()))

    def coil_set_spec(self):
        """Build the immutable grouped coil spec from the live coil graph."""
        if self._jax_native:
            return self.coil_set_spec_from_dofs(jnp.asarray(self.x, dtype=jnp.float64))

        geometry_cache = {}
        gammas = []
        gammadashs = []
        currents = []
        for coil in self._coils:
            *_prefix, gamma, gammadash, current_value = self._coil_geometry_inputs(
                coil,
                geometry_cache,
            )
            gammas.append(gamma)
            gammadashs.append(gammadash)
            currents.append(current_value)
        return grouped_coil_set_spec_from_lists(gammas, gammadashs, currents)

    # ------------------------------------------------------------------
    # Forward field evaluation
    # ------------------------------------------------------------------

    def B(self):
        """Magnetic field B at the evaluation points.

        Returns:
            (npoints, 3) JAX array.
        """
        return grouped_biot_savart_B_from_spec(self._points_jax, self.coil_set_spec())

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
        base_gamma, base_gammadash, gamma_s, gammadash_s = (
            self._base_curve_geometry_with_timings(curve, geometry_cache)
        )
        gamma, gammadash = _rotate_curve_geometry(base_gamma, base_gammadash, rotmat)
        current_s, current_value = _time_call_result(
            lambda: jnp.asarray(current.get_value() * scale, dtype=jnp.float64)
        )
        timings = {
            "curve_gamma_s": gamma_s,
            "curve_gammadash_s": gammadash_s,
            "current_value_s": current_s,
        }
        return curve, rotmat, current, scale, gamma, gammadash, current_value, timings

    def B_vjp(self, v):
        r"""Vector-Jacobian product of B w.r.t. coil DOFs.

        Given a cotangent vector ``v`` (typically ``dJ/dB``), returns
        a :class:`Derivative` mapping every free coil DOF to its
        contribution to the scalar objective.

        Uses ``jax.vjp`` through the pure Biot-Savart kernel, then
        projects each coil's geometry/current cotangents back to coil
        DOFs. Curves that expose JAX pullback methods stay on the JAX
        path through the projection step; only legacy curves that do
        not expose the JAX geometry/pullback interface fall back to
        ``Coil.vjp()``.

        Args:
            v: (npoints, 3) cotangent, same shape as ``B()``.

        Returns:
            :class:`Derivative` (sum over all coils).
        """
        deriv_data = {}
        points = self._points_jax
        v_jax = jnp.asarray(v)
        geometry_cache = {}
        coil_infos = self._collect_free_coil_vjp_infos(geometry_cache)
        for group in self._group_coil_vjp_infos(coil_infos):
            dg_group, dgd_group, dc_group = biot_savart_B_vjp(
                points,
                v_jax,
                group["gammas"],
                group["gammadashs"],
                group["currents"],
            )
            for group_index, info in enumerate(group["infos"]):
                _merge_derivative_data(
                    deriv_data,
                    _project_single_coil_cotangent_data(
                        info.coil,
                        dg_group[group_index],
                        dgd_group[group_index],
                        dc_group[group_index],
                    ),
                )
        return Derivative(deriv_data)

    def coil_cotangents_to_derivative(self, d_coil_arrays, coil_indices):
        """Project grouped coil cotangent arrays to a :class:`Derivative`.

        This is the JAX-native replacement for the standalone
        ``_coil_cotangents_to_derivative()`` helper. Curves that
        expose JAX pullback methods stay on the JAX path through the
        projection step. Legacy curves that do not expose that
        interface fall back to CPU ``Coil.vjp()`` slice by slice.

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
        v_jax = jnp.asarray(v)
        geometry_cache = {}
        component_totals = {
            "curve_gamma_s": 0.0,
            "curve_gammadash_s": 0.0,
            "current_value_s": 0.0,
            "single_coil_pullback_s": 0.0,
            "coil_vjp_s": 0.0,
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
                lambda: biot_savart_B_vjp(
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
            for group_index, info in enumerate(group["infos"]):
                coil_vjp_s, _ = _time_call_result(
                    lambda: _project_single_coil_cotangent_data(
                        info.coil,
                        dg_group[group_index],
                        dgd_group[group_index],
                        dc_group[group_index],
                    )
                )
                component_totals["coil_vjp_s"] += coil_vjp_s
                coil_timings = dict(info.timings)
                coil_timings.update(
                    {
                        "single_coil_pullback_s": 0.0,
                        "coil_vjp_s": coil_vjp_s,
                    }
                )
                for name in ("curve_gamma_s", "curve_gammadash_s", "current_value_s"):
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
