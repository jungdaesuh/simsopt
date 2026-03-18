"""
Optimizable adapter wrapping the pure JAX Biot-Savart functions.

``BiotSavartJAX`` participates in the ``Optimizable`` dependency graph
through its coil list while computing the magnetic field via the pure
JAX kernels in :mod:`simsopt.field.biotsavart_jax`.

This module does **not** inherit from ``sopp.BiotSavart`` or
``sopp.MagneticField`` — it is a parallel JAX-native class per the
M0 rewrite contract (adapter pattern, §5).
"""

import numpy as np
import jax
import jax.numpy as jnp

from .._core.optimizable import Optimizable
from .biotsavart_jax import (
    biot_savart_B,
    biot_savart_dB_by_dX,
    biot_savart_B_and_dB,
    group_coil_data,
    grouped_biot_savart_B,
)

__all__ = ["BiotSavartJAX"]


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
            from ..geo.curve import RotatedCurve
            from .coil import ScaledCurrent, Current
        except ImportError:
            return

        base_curve_ids = {}  # id(obj) → index
        base_current_ids = {}
        base_curves = []
        base_currents = []
        descs = []

        for coil in self._coils:
            curve = coil.curve
            rotmat = None

            # Unwrap nested RotatedCurve layers (accumulate rotation).
            # Outer wraps inner: gamma = base.gamma() @ R_inner @ R_outer,
            # so we pre-multiply each inner rotation found while unwrapping.
            while isinstance(curve, RotatedCurve):
                rm = curve.rotmat
                rotmat = rm if rotmat is None else rm @ rotmat
                curve = curve.curve

            if not isinstance(curve, CurveXYZFourier):
                return  # unsupported curve type → stay on fallback path

            cid = id(curve)
            if cid not in base_curve_ids:
                base_curve_ids[cid] = len(base_curves)
                base_curves.append(curve)

            # Unwrap ScaledCurrent chain
            current = coil.current
            scale = 1.0
            while isinstance(current, ScaledCurrent):
                scale *= current.scale
                current = current.current_to_scale

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

    def _extract_coil_data_grouped(self):
        """Read coil geometry grouped by quadrature point count.

        Delegates to :func:`group_coil_data` in ``biotsavart_jax.py``.

        Returns:
            list of ``(gammas, gammadashs, currents, coil_indices)``
            tuples, one per distinct quadrature count.
        """
        return group_coil_data(
            [c.curve.gamma() for c in self._coils],
            [c.curve.gammadash() for c in self._coils],
            [c.current.get_value() for c in self._coils],
        )

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
        groups = self._extract_coil_data_grouped()
        points = self._points_jax
        v_jax = jnp.asarray(v)

        all_derivs = []
        for gammas, gammadashs, currents, indices in groups:

            def fwd(g, gd, c):
                return biot_savart_B(points, g, gd, c)

            _, vjp_fn = jax.vjp(fwd, gammas, gammadashs, currents)
            dg, dgd, dc = vjp_fn(v_jax)

            dg_np = np.asarray(dg)
            dgd_np = np.asarray(dgd)
            dc_np = np.asarray(dc)
            for local_i, global_i in enumerate(indices):
                all_derivs.append(
                    self._coils[global_i].vjp(
                        dg_np[local_i], dgd_np[local_i], np.asarray([dc_np[local_i]])
                    )
                )
        return sum(all_derivs)
