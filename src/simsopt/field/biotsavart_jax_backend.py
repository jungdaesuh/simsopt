"""JAX-backed Biot-Savart adapter and coil-tree helpers.

``BiotSavartJAX`` participates in the ``Optimizable`` dependency graph
through its coil list while computing the magnetic field via the pure
JAX kernels in :mod:`simsopt.field.biotsavart_jax`.

This module does **not** inherit from ``sopp.BiotSavart`` or
``sopp.MagneticField`` — it is a parallel JAX-native class per the
M0 rewrite contract (adapter pattern, §5).
"""

import time

import numpy as np
import jax
import jax.numpy as jnp

from .._core.derivative import Derivative
from .._core.optimizable import Optimizable
from .biotsavart_jax import (
    biot_savart_B,
    biot_savart_dB_by_dX,
    biot_savart_B_and_dB,
    group_coil_data,
    grouped_biot_savart_B,
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
    for leaf in jax.tree_util.tree_leaves(result):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return float(time.perf_counter() - start), result


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
            name: float(elapsed_s)
            for name, elapsed_s in coil_timings.items()
        },
        "total_s": float(sum(coil_timings.values())),
    }


def _supports_native_curve_geometry(curve):
    from ..geo.curve import CurveCWSFourier

    return isinstance(curve, CurveCWSFourier)


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
        self._curve_quadpoints_jax = jnp.asarray(np.asarray(base_curves[0].quadpoints))

    @property
    def coils(self):
        return self._coils

    def set_points(self, points):
        """Set evaluation points (converted to a JAX array once)."""
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
            curve_dofs = jnp.asarray(curve.get_dofs(), dtype=jnp.float64)
            surf_dofs = jnp.asarray(curve.surf.get_dofs(), dtype=jnp.float64)
            gamma_s, base_gamma = _time_call_result(
                lambda: curve.gamma_jax(curve_dofs, surf_dofs)
            )
            gammadash_s, base_gammadash = _time_call_result(
                lambda: curve.gammadash_jax(curve_dofs, surf_dofs)
            )
        else:
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

    def _extract_coil_data_grouped(self):
        """Read coil geometry grouped by quadrature point count.

        Delegates to :func:`group_coil_data` in ``biotsavart_jax.py``.

        Returns:
            list of ``(gammas, gammadashs, currents, coil_indices)``
            tuples, one per distinct quadrature count.
        """
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
            currents.append(float(current_value))
        return group_coil_data(gammas, gammadashs, currents)

    # ------------------------------------------------------------------
    # Forward field evaluation
    # ------------------------------------------------------------------

    def B(self):
        """Magnetic field B at the evaluation points.

        Returns:
            (npoints, 3) JAX array.
        """
        coil_arrays = [(g, gd, c) for g, gd, c, _ in self._extract_coil_data_grouped()]
        return grouped_biot_savart_B(self._points_jax, coil_arrays)

    def dB_by_dX(self):
        """Spatial Jacobian dB/dX at the evaluation points.

        Returns:
            (npoints, 3, 3) JAX array where ``[p, j, l] = ∂_j B_l``.
        """
        groups = self._extract_coil_data_grouped()
        result = biot_savart_dB_by_dX(self._points_jax, *groups[0][:3])
        for gammas, gammadashs, currents, _ in groups[1:]:
            result = result + biot_savart_dB_by_dX(
                self._points_jax, gammas, gammadashs, currents
            )
        return result

    def B_and_dB(self):
        """Combined B and dB/dX (single JIT compilation).

        Returns:
            (B, dB_dX) with shapes (npoints, 3) and (npoints, 3, 3).
        """
        groups = self._extract_coil_data_grouped()
        B, dB = biot_savart_B_and_dB(self._points_jax, *groups[0][:3])
        for gammas, gammadashs, currents, _ in groups[1:]:
            Bi, dBi = biot_savart_B_and_dB(
                self._points_jax, gammas, gammadashs, currents
            )
            B = B + Bi
            dB = dB + dBi
        return B, dB

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

    def _direct_curve_current_vjp(self, curve, rotmat, current, scale, dg, dgd, dc):
        """Map native JAX pullbacks back to curve, surface, and current DOFs."""
        if rotmat is not None:
            rotmat_t = np.asarray(rotmat).T
            dg = dg @ rotmat_t
            dgd = dgd @ rotmat_t

        deriv_data = {}
        if curve.dof_size > 0:
            curve_dofs = jnp.asarray(curve.get_dofs(), dtype=jnp.float64)
            deriv_data[curve] = (
                np.asarray(curve.dgamma_by_dcoeff_vjp_jax(curve_dofs, dg))
                + np.asarray(curve.dgammadash_by_dcoeff_vjp_jax(curve_dofs, dgd))
            )
        if curve.surf.dof_size > 0:
            surf_dofs = curve.surf.get_dofs()
            deriv_data[curve.surf] = (
                np.asarray(curve.dgamma_by_dsurf_vjp_jax(surf_dofs, dg))
                + np.asarray(
                    curve.dgammadash_by_dsurf_vjp_jax(surf_dofs, dgd)
                )
            )
        if current.dof_size > 0:
            deriv_data[current] = np.asarray([float(scale) * float(dc)], dtype=float)
        return Derivative(deriv_data)

    def _coil_b_vjp_derivative(self, coil, points, v_jax, geometry_cache=None):
        curve, rotmat, current, scale, gamma, gammadash, current_value = (
            self._coil_geometry_inputs(coil, geometry_cache)
        )
        dg, dgd, dc = jax.device_get(
            _single_coil_b_vjp(points, v_jax, gamma, gammadash, current_value)
        )
        if _supports_native_curve_geometry(curve):
            return self._direct_curve_current_vjp(
                curve,
                rotmat,
                current,
                scale,
                dg,
                dgd,
                dc,
            )
        return coil.vjp(dg, dgd, np.asarray([dc]))

    def B_vjp(self, v):
        r"""Vector-Jacobian product of B w.r.t. coil DOFs.

        Given a cotangent vector ``v`` (typically ``dJ/dB``), returns
        a :class:`Derivative` mapping every free coil DOF to its
        contribution to the scalar objective.

        Uses ``jax.vjp`` through the pure Biot-Savart kernel, then
        maps the per-coil geometry/current gradients back to DOFs via
        the existing ``Coil.vjp()`` machinery.

        Args:
            v: (npoints, 3) cotangent, same shape as ``B()``.

        Returns:
            :class:`Derivative` (sum over all coils).
        """
        all_derivs = []
        points = self._points_jax
        v_jax = jnp.asarray(v)
        geometry_cache = {}
        for coil in self._coils:
            if not self._coil_has_free_dofs(coil):
                continue
            all_derivs.append(
                self._coil_b_vjp_derivative(coil, points, v_jax, geometry_cache)
            )
        if not all_derivs:
            return Derivative({})
        return sum(all_derivs)

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
        per_coil_timings = []
        wall_start = time.perf_counter()
        for coil_index, coil in enumerate(self._coils):
            if not self._coil_has_free_dofs(coil):
                per_coil_timings.append(
                    _build_coil_profile_entry(
                        coil_index,
                        _zero_profile_component_timings(component_totals),
                    )
                )
                continue
            curve, rotmat, current, scale, gamma, gammadash, current_value, coil_timings = (
                self._coil_b_vjp_inputs(coil, geometry_cache)
            )
            pullback_s, (dg, dgd, dc) = _time_call_result(
                lambda: jax.device_get(
                    _single_coil_b_vjp(points, v_jax, gamma, gammadash, current_value)
                )
            )
            coil_vjp_s, _ = _time_call_result(
                lambda: self._direct_curve_current_vjp(
                    curve,
                    rotmat,
                    current,
                    scale,
                    dg,
                    dgd,
                    dc,
                )
                if _supports_native_curve_geometry(curve)
                else coil.vjp(dg, dgd, np.asarray([dc]))
            )
            coil_timings["single_coil_pullback_s"] = pullback_s
            coil_timings["coil_vjp_s"] = coil_vjp_s
            for name, elapsed_s in coil_timings.items():
                component_totals[name] += elapsed_s
            per_coil_timings.append(_build_coil_profile_entry(coil_index, coil_timings))
        wall_time_s = float(time.perf_counter() - wall_start)
        return {
            "wall_time_s": wall_time_s,
            "component_timings_s": {
                name: float(elapsed_s)
                for name, elapsed_s in component_totals.items()
            },
            "dominant_components": _build_profile_breakdown(component_totals),
            "per_coil_timings_s": per_coil_timings,
            "dominant_coils": _build_coil_profile_breakdown(per_coil_timings),
        }
