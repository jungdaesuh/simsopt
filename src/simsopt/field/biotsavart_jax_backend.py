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
)

__all__ = ["BiotSavartJAX"]


class BiotSavartJAX(Optimizable):
    r"""JAX-backed Biot-Savart magnetic field evaluation.

    Drop-in replacement for :class:`BiotSavart` in workflows where the
    field is consumed by a JAX-backed objective (e.g. ``SquaredFluxJAX``).

    The class holds no DOFs of its own.  Its ``Optimizable`` dependency
    chain runs through the coil list so that the outer framework
    correctly composes DOFs and derivatives.

    Args:
        coils: list of :class:`simsopt.field.coil.Coil` objects.
    """

    def __init__(self, coils):
        self._coils = list(coils)
        self._points_jax = None
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=self._coils)

    @property
    def coils(self):
        return self._coils

    def set_points(self, points):
        """Set evaluation points (converted to a JAX array once)."""
        self._points_jax = jnp.asarray(np.ascontiguousarray(points))

    def _extract_coil_data(self):
        """Read current coil geometry and currents as stacked JAX arrays.

        Returns:
            gammas:     (ncoils, nquad, 3)
            gammadashs: (ncoils, nquad, 3)
            currents:   (ncoils,)
        """
        gammas = jnp.asarray(np.stack([c.curve.gamma() for c in self._coils]))
        gammadashs = jnp.asarray(np.stack([c.curve.gammadash() for c in self._coils]))
        currents = jnp.asarray(np.array([c.current.get_value() for c in self._coils]))
        return gammas, gammadashs, currents

    # ------------------------------------------------------------------
    # Forward field evaluation
    # ------------------------------------------------------------------

    def B(self):
        """Magnetic field B at the evaluation points.

        Returns:
            (npoints, 3) JAX array.
        """
        gammas, gammadashs, currents = self._extract_coil_data()
        return biot_savart_B(self._points_jax, gammas, gammadashs, currents)

    def dB_by_dX(self):
        """Spatial Jacobian dB/dX at the evaluation points.

        Returns:
            (npoints, 3, 3) JAX array where ``[p, j, l] = ∂_j B_l``.
        """
        gammas, gammadashs, currents = self._extract_coil_data()
        return biot_savart_dB_by_dX(self._points_jax, gammas, gammadashs, currents)

    def B_and_dB(self):
        """Combined B and dB/dX (single JIT compilation).

        Returns:
            (B, dB_dX) with shapes (npoints, 3) and (npoints, 3, 3).
        """
        gammas, gammadashs, currents = self._extract_coil_data()
        return biot_savart_B_and_dB(self._points_jax, gammas, gammadashs, currents)

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
        gammas, gammadashs, currents = self._extract_coil_data()
        points = self._points_jax

        def fwd(g, gd, c):
            return biot_savart_B(points, g, gd, c)

        _, vjp_fn = jax.vjp(fwd, gammas, gammadashs, currents)
        dg, dgd, dc = vjp_fn(jnp.asarray(v))

        return sum(
            coil.vjp(
                np.asarray(dg[i]),
                np.asarray(dgd[i]),
                np.asarray([float(dc[i])]),
            )
            for i, coil in enumerate(self._coils)
        )
