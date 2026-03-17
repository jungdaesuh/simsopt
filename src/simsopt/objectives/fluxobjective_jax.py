"""
JAX-backed SquaredFlux objective for Stage 2 coil optimization.

``SquaredFluxJAX`` is a drop-in replacement for :class:`SquaredFlux` that
uses JAX for both forward evaluation and gradient computation.  The
gradient is obtained by differentiating the full
``BiotSavart → integral_BdotN`` chain via ``jax.value_and_grad``,
then mapping the per-coil geometry/current gradients to DOFs through
the existing ``Coil.vjp()`` machinery.

The surface geometry (``gamma``, ``normal``) is evaluated once at
construction time and kept on JAX arrays for the lifetime of the
objective.  This is correct for Stage 2 where the plasma surface
is fixed.
"""

import numpy as np
import jax
import jax.numpy as jnp

from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec
from ..field.biotsavart_jax import biot_savart_B
from .integral_bdotn_jax import integral_BdotN as integral_BdotN_jax

__all__ = ["SquaredFluxJAX"]


class SquaredFluxJAX(Optimizable):
    r"""JAX-backed quadratic-flux objective for Stage 2.

    Computes the same quantity as :class:`SquaredFlux` but replaces
    ``sopp.integral_BdotN`` and ``sopp.biot_savart_vjp_graph`` with
    pure JAX functions.  The gradient uses ``jax.value_and_grad`` on the
    composed ``BiotSavart → integral_BdotN`` chain.

    .. note::
        The plasma surface must be fixed during the optimization.
        Surface geometry arrays are captured at construction time.

    Args:
        surface: a :class:`Surface` providing ``gamma()`` and ``normal()``.
        field: a :class:`BiotSavartJAX` instance.
        target: optional ``(nphi, ntheta)`` target normal field (default 0).
        definition: ``"quadratic flux"`` | ``"normalized"`` | ``"local"``.
    """

    def __init__(self, surface, field, target=None, definition="quadratic flux"):
        if definition not in ("quadratic flux", "normalized", "local"):
            raise ValueError(f"Unknown definition: {definition!r}")

        self.surface = surface
        self.field = field
        self.definition = definition

        # Freeze surface geometry on JAX arrays (fixed for Stage 2).
        self._normal_jax = jnp.asarray(surface.normal())
        nphi, ntheta = self._normal_jax.shape[:2]

        if target is not None:
            self._target_jax = jnp.asarray(np.ascontiguousarray(target))
        else:
            self._target_jax = jnp.zeros((nphi, ntheta), dtype=jnp.float64)

        # Set evaluation points on the field adapter.
        xyz = surface.gamma()
        field.set_points(xyz.reshape((-1, 3)))

        # Build JIT-compiled forward and value_and_grad functions.
        # The surface arrays are captured in the closure and baked into the
        # XLA program as constants (they never change during Stage 2).
        normal_jax = self._normal_jax
        target_jax = self._target_jax
        points_jax = field._points_jax

        def _raw_forward(gammas, gammadashs, currents):
            B = biot_savart_B(points_jax, gammas, gammadashs, currents)
            Bcoil = B.reshape((nphi, ntheta, 3))
            return integral_BdotN_jax(Bcoil, target_jax, normal_jax, definition)

        self._jit_forward = jax.jit(_raw_forward)
        self._jit_val_grad = jax.jit(
            jax.value_and_grad(_raw_forward, argnums=(0, 1, 2))
        )

        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[field])

    def J(self):
        gammas, gammadashs, currents = self.field._extract_coil_data()
        return float(self._jit_forward(gammas, gammadashs, currents))

    @derivative_dec
    def dJ(self):
        gammas, gammadashs, currents = self.field._extract_coil_data()
        _, (dg, dgd, dc) = self._jit_val_grad(gammas, gammadashs, currents)

        dg_np = np.asarray(dg)
        dgd_np = np.asarray(dgd)
        dc_np = np.asarray(dc)
        return sum(
            coil.vjp(dg_np[i], dgd_np[i], np.asarray([dc_np[i]]))
            for i, coil in enumerate(self.field.coils)
        )
