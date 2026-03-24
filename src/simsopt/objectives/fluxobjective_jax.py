"""
JAX-backed SquaredFlux objective for Stage 2 coil optimization.

``SquaredFluxJAX`` is a drop-in replacement for :class:`SquaredFlux` that
uses JAX for both forward evaluation and gradient computation.

When the field adapter supports the JAX-native path (all coils are
``CurveXYZFourier`` or ``RotatedCurve``), the full
``DOFs → Fourier basis → BiotSavart → integral_BdotN`` chain is
traced end-to-end by ``jax.value_and_grad``, eliminating CPU
round-trips for coil geometry evaluation and ``Coil.vjp()`` calls.

Otherwise, falls back to computing geometry via C++ ``gamma()``
and mapping gradients through ``Coil.vjp()``.

The surface geometry (``gamma``, ``normal``) is evaluated once at
construction time and kept on JAX arrays for the lifetime of the
objective.  This is correct for Stage 2 where the plasma surface
is fixed.
"""

import numpy as np
import jax
import jax.numpy as jnp

from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec, Derivative
from ..field.biotsavart_jax import biot_savart_B
from .integral_bdotn_jax import integral_BdotN as integral_BdotN_jax

__all__ = ["SquaredFluxJAX"]


# -----------------------------------------------------------------------
# Fourier basis precomputation
# -----------------------------------------------------------------------


def _build_fourier_basis(quadpoints_jax, order):
    """Precompute the CurveXYZFourier basis matrix and its derivative.

    The DOF layout per coordinate is ``[c₀, s₁, c₁, s₂, c₂, …]``
    where ``cⱼ`` multiplies ``cos(2πjθ)`` and ``sⱼ`` multiplies
    ``sin(2πjθ)``.

    Args:
        quadpoints_jax: (npts,) quadrature angles in [0, 1).
        order: Fourier truncation order.

    Returns:
        basis:  (npts, 2*order+1) — ``gamma  = basis  @ coeffs.T``
        dbasis: (npts, 2*order+1) — ``gammadash = dbasis @ coeffs.T``
    """
    k = 2 * order + 1
    npts = quadpoints_jax.shape[0]
    basis = jnp.zeros((npts, k))
    dbasis = jnp.zeros((npts, k))

    basis = basis.at[:, 0].set(1.0)
    # dbasis[:, 0] stays zero (constant term has zero derivative)

    for j in range(1, order + 1):
        arg = 2.0 * jnp.pi * j * quadpoints_jax
        s = jnp.sin(arg)
        c = jnp.cos(arg)
        basis = basis.at[:, 2 * j - 1].set(s)
        basis = basis.at[:, 2 * j].set(c)
        dbasis = dbasis.at[:, 2 * j - 1].set(2.0 * jnp.pi * j * c)
        dbasis = dbasis.at[:, 2 * j].set(-2.0 * jnp.pi * j * s)

    return basis, dbasis


# -----------------------------------------------------------------------
# SquaredFluxJAX
# -----------------------------------------------------------------------


class SquaredFluxJAX(Optimizable):
    r"""JAX-backed quadratic-flux objective for Stage 2.

    Computes the same quantity as :class:`SquaredFlux` but replaces
    ``sopp.integral_BdotN`` and ``sopp.biot_savart_vjp_graph`` with
    pure JAX functions.

    When the JAX-native path is active, ``value_and_grad`` traces the
    full ``DOFs → coil geometry → B → integral`` chain so that no
    CPU round-trip or ``Coil.vjp()`` call is needed during
    optimization.

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

        self._clear_cached_results()

        # Choose path: JAX-native (end-to-end) or fallback (via Coil.vjp).
        self._use_jax_native = field._jax_native
        if self._use_jax_native:
            self._init_jax_native(field, nphi, ntheta, definition)
        else:
            self._init_fallback(field, nphi, ntheta, definition)

        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[field])

    # ------------------------------------------------------------------
    # JAX-native path: DOFs → Fourier basis → B → J (single JIT program)
    # ------------------------------------------------------------------

    def _init_jax_native(self, field, nphi, ntheta, definition):
        """Build end-to-end JIT functions from flat DOFs to scalar J."""
        order = field._curve_order
        basis, dbasis = _build_fourier_basis(
            field._curve_quadpoints_jax,
            order,
        )

        n_base_curves = len(field._unique_base_curves)
        curve_dof_size = field._curve_dof_size
        total_curve_dofs = n_base_curves * curve_dof_size
        k = 2 * order + 1

        # Static coil descriptors (unrolled by JIT tracer)
        base_curve_idxs = tuple(d[0] for d in field._coil_descs)
        base_current_idxs = tuple(d[1] for d in field._coil_descs)
        rotmats = tuple(d[2] for d in field._coil_descs)
        current_scales = tuple(d[3] for d in field._coil_descs)
        n_coils = len(field._coil_descs)

        # Closure-captured constants
        points_jax = field._points_jax
        normal_jax = self._normal_jax
        target_jax = self._target_jax

        def forward(flat_dofs):
            # Per-base-curve DOF slices
            curve_dofs = [
                flat_dofs[i * curve_dof_size : (i + 1) * curve_dof_size]
                for i in range(n_base_curves)
            ]
            # Per-base-current scalar values
            current_vals = flat_dofs[total_curve_dofs:]

            gammas = []
            gammadashs = []
            currents = []

            for ci in range(n_coils):
                coeffs = curve_dofs[base_curve_idxs[ci]].reshape(3, k)
                g = basis @ coeffs.T
                gd = dbasis @ coeffs.T

                rm = rotmats[ci]
                if rm is not None:
                    g = g @ rm
                    gd = gd @ rm

                gammas.append(g)
                gammadashs.append(gd)
                currents.append(
                    current_vals[base_current_idxs[ci]] * current_scales[ci]
                )

            B = biot_savart_B(
                points_jax,
                jnp.stack(gammas),
                jnp.stack(gammadashs),
                jnp.array(currents),
            )
            Bcoil = B.reshape((nphi, ntheta, 3))
            return integral_BdotN_jax(Bcoil, target_jax, normal_jax, definition)

        self._jit_forward_dofs = jax.jit(forward)
        self._jit_val_grad_dofs = jax.jit(jax.value_and_grad(forward))

    # ------------------------------------------------------------------
    # Fallback path: geometry via C++ gamma(), gradient via Coil.vjp()
    # ------------------------------------------------------------------

    def _init_fallback(self, field, nphi, ntheta, definition):
        """Build JIT functions for the integral evaluation.

        The Biot-Savart evaluation is delegated to ``field.B()`` which
        handles mixed quadrature counts.  The JIT boundary covers only
        the integral computation and its gradient w.r.t. B.
        """
        normal_jax = self._normal_jax
        target_jax = self._target_jax

        def _integral_from_B(B):
            Bcoil = B.reshape((nphi, ntheta, 3))
            return integral_BdotN_jax(Bcoil, target_jax, normal_jax, definition)

        self._jit_integral = jax.jit(_integral_from_B)
        self._jit_integral_value_grad = jax.jit(jax.value_and_grad(_integral_from_B))

    # ------------------------------------------------------------------
    # DOF gathering (JAX-native path)
    # ------------------------------------------------------------------

    def _gather_unique_full_dofs(self):
        """Concatenate all DOFs from unique base curves and currents.

        Layout: ``[curve_0_dofs, curve_1_dofs, …, current_0, current_1, …]``
        where each ``curve_i_dofs`` has length ``curve_dof_size`` and
        each ``current_j`` is a single scalar.

        Returns:
            (n_total_dofs,) JAX float64 array.
        """
        parts = []
        for curve in self.field._unique_base_curves:
            parts.append(np.asarray(curve.get_dofs()))
        for current in self.field._unique_base_currents:
            parts.append(np.array([current.get_value()]))
        return jnp.asarray(np.concatenate(parts), dtype=jnp.float64)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _clear_cached_results(self):
        self._cached_value = None
        self._cached_partials = None

    def recompute_bell(self, parent=None):
        self._clear_cached_results()

    def J(self):
        if self._use_jax_native and not self.new_x and self._cached_value is not None:
            return self._cached_value

        if self._use_jax_native:
            self._cached_value = float(
                self._jit_forward_dofs(self._gather_unique_full_dofs())
            )
            value = self._cached_value
        else:
            # Mixed-quadrature fallback can change through field.set_points(...)
            # without marking the Optimizable DOFs dirty, so do not reuse cache.
            value = float(self._jit_integral(self.field.B()))
            self._clear_cached_results()

        self._cached_partials = None
        self.new_x = False
        return value

    @derivative_dec
    def dJ(self):
        if self._use_jax_native and not self.new_x and self._cached_partials is not None:
            return self._cached_partials

        if self._use_jax_native:
            self._cached_value, self._cached_partials = self._value_and_dJ_jax_native()
            partials = self._cached_partials
        else:
            _value, partials = self._value_and_dJ_fallback()
            self._clear_cached_results()

        self.new_x = False
        return partials

    def _value_and_dJ_jax_native(self):
        """Combined value and gradient via end-to-end JAX value_and_grad."""
        flat_dofs = self._gather_unique_full_dofs()
        value, grad = self._jit_val_grad_dofs(flat_dofs)
        grad_np = np.asarray(grad)

        # Map the flat gradient back to per-Optimizable Derivative entries.
        deriv_data = {}
        cds = self.field._curve_dof_size
        for i, curve in enumerate(self.field._unique_base_curves):
            deriv_data[curve] = grad_np[i * cds : (i + 1) * cds]

        current_start = len(self.field._unique_base_curves) * cds
        for i, current in enumerate(self.field._unique_base_currents):
            deriv_data[current] = grad_np[current_start + i : current_start + i + 1]

        return float(value), Derivative(deriv_data)

    def _value_and_dJ_fallback(self):
        """Combined value and gradient via field.B_vjp() (mixed quadrature)."""
        B = self.field.B()
        value, dJ_dB = self._jit_integral_value_grad(B)
        return float(value), self.field.B_vjp(np.asarray(dJ_dB))
