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

from ..backend import raise_if_strict_jax_fallback, warn_if_jax_fallback
from .._core.jax_host_boundary import (
    host_array as _host_array,
    host_scalar as _host_scalar,
)
from .._core.util import ObjectiveFailure
from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec, Derivative
from ..jax_core.biotsavart import biot_savart_B
from ..jax_core.objectives_flux import (
    build_fourier_basis,
    fixed_surface_flux_specs_from_surface,
    fixed_surface_flux_integral_from_B,
)

__all__ = ["SquaredFluxJAX"]


# -----------------------------------------------------------------------
# Fallback guardrails
# -----------------------------------------------------------------------


def _handle_squared_flux_fallback(detail: str) -> None:
    raise_if_strict_jax_fallback(component="SquaredFluxJAX", detail=detail)
    warn_if_jax_fallback(component="SquaredFluxJAX", detail=detail)


def _supports_jax_objective_fallback(field) -> bool:
    supports_fallback = getattr(field, "supports_jax_objective_fallback", None)
    if not callable(supports_fallback):
        return False
    return bool(supports_fallback())


def _validate_squared_flux_parity_mode(parity_mode: str) -> str:
    if parity_mode not in {"standard", "native_only"}:
        raise ValueError(f"Unknown SquaredFluxJAX parity_mode={parity_mode!r}")
    return parity_mode


def _raise_native_only_squared_flux_error(detail: str) -> None:
    raise RuntimeError(
        "SquaredFluxJAX parity_mode='native_only' requires the end-to-end "
        f"JAX-native path; fallback seam via {detail} is not allowed."
    )


def _raise_if_nonfinite_squared_flux_gradient(*, definition: str, value, grad) -> None:
    value_array = np.asarray(value, dtype=np.float64)
    grad_array = np.asarray(grad, dtype=np.float64)
    if np.all(np.isfinite(value_array)) and np.all(np.isfinite(grad_array)):
        return
    raise ObjectiveFailure(
        "SquaredFluxJAX "
        f"{definition} gradient is singular because the objective or its "
        "derivative is non-finite."
    )


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

    Degenerate quadrature points follow the same contract as
    :class:`SquaredFlux`: zero-area elements contribute zero, while
    invalid ``"normalized"`` and ``"local"`` singular configurations
    evaluate to ``inf``.

    Args:
        surface: a :class:`Surface` providing ``gamma()`` and ``normal()``.
        field: a :class:`BiotSavartJAX` instance.
        target: optional ``(nphi, ntheta)`` target normal field (default 0).
        definition: ``"quadratic flux"`` | ``"normalized"`` | ``"local"``.
        parity_mode: ``"standard"`` allows the documented fallback path,
            while ``"native_only"`` rejects fallback seams during parity tests.
    """

    def __init__(
        self,
        surface,
        field,
        target=None,
        definition="quadratic flux",
        parity_mode="standard",
    ):
        if definition not in ("quadratic flux", "normalized", "local"):
            raise ValueError(f"Unknown definition: {definition!r}")
        parity_mode = _validate_squared_flux_parity_mode(parity_mode)

        self.surface = surface
        self.field = field
        self.definition = definition
        self.parity_mode = parity_mode

        target_array = None if target is None else np.ascontiguousarray(target)
        field_eval_spec, self._flux_spec = fixed_surface_flux_specs_from_surface(
            surface,
            target=target_array,
            definition=definition,
        )
        self._normal_jax = self._flux_spec.normal
        self._target_jax = self._flux_spec.target

        # Set evaluation points on the field adapter from the immutable spec.
        field.set_points_from_spec(field_eval_spec)

        self._clear_cached_results()

        # Choose path: JAX-native (end-to-end) or fallback (via Coil.vjp).
        self._use_jax_native = field._jax_native
        self._uses_jax_objective_fallback = False
        if self._use_jax_native:
            self._init_jax_native(field, definition)
        elif parity_mode == "native_only":
            _raise_native_only_squared_flux_error(
                "the mixed-quadrature/objective fallback path"
            )
        elif _supports_jax_objective_fallback(field):
            self._uses_jax_objective_fallback = True
            self._init_fallback(field, definition)
        else:
            _handle_squared_flux_fallback(
                "the CPU fallback objective path for non-JAX-native coils"
            )
            self._init_fallback(field, definition)

        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[field])

    # ------------------------------------------------------------------
    # JAX-native path: DOFs → Fourier basis → B → J (single JIT program)
    # ------------------------------------------------------------------

    def _init_jax_native(self, field, definition):
        """Build end-to-end JIT functions from flat DOFs to scalar J."""
        del definition
        order = field._curve_order
        basis, dbasis = build_fourier_basis(
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

        def forward(flat_dofs, flux_spec):
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
                flux_spec.points,
                jnp.stack(gammas),
                jnp.stack(gammadashs),
                jnp.array(currents),
            )
            return fixed_surface_flux_integral_from_B(B, flux_spec)

        jit_forward = jax.jit(forward)
        jit_val_grad = jax.jit(jax.value_and_grad(forward, argnums=0))

        def _jit_forward_dofs(flat_dofs):
            return jit_forward(flat_dofs, self._flux_spec)

        def _jit_val_grad_dofs(flat_dofs):
            return jit_val_grad(flat_dofs, self._flux_spec)

        self._jit_forward_dofs = _jit_forward_dofs
        self._jit_val_grad_dofs = _jit_val_grad_dofs

    # ------------------------------------------------------------------
    # Fallback path: geometry via C++ gamma(), gradient via Coil.vjp()
    # ------------------------------------------------------------------

    def _init_fallback(self, field, definition):
        """Build JIT functions for the integral evaluation.

        The Biot-Savart evaluation is delegated to ``field.B()`` which
        handles mixed quadrature counts.  The JIT boundary covers only
        the integral computation and its gradient w.r.t. B.
        """
        del field, definition

        def _integral_from_B(B, flux_spec):
            # Reduction-order stabilization belongs in integral_bdotn_jax via
            # the immutable-spec helper, not in this wrapper layer.
            return fixed_surface_flux_integral_from_B(B, flux_spec)

        jit_integral = jax.jit(_integral_from_B)
        jit_integral_value_grad = jax.jit(
            jax.value_and_grad(_integral_from_B, argnums=0)
        )

        def _jit_integral(B):
            return jit_integral(B, self._flux_spec)

        def _jit_integral_value_grad(B):
            return jit_integral_value_grad(B, self._flux_spec)

        self._jit_integral = _jit_integral
        self._jit_integral_value_grad = _jit_integral_value_grad

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
        self._cached_points_version = None

    def recompute_bell(self, parent=None):
        self._clear_cached_results()

    def J(self):
        current_points_version = getattr(self.field, "_points_version", None)
        cache_valid = (
            not self.new_x
            and self._cached_value is not None
            and self._cached_points_version == current_points_version
        )
        if cache_valid:
            return self._cached_value

        if self._use_jax_native:
            self._cached_value = float(
                _host_scalar(
                    self._jit_forward_dofs(self._gather_unique_full_dofs()),
                    dtype=np.float64,
                )
            )
            value = self._cached_value
        else:
            value = float(
                _host_scalar(self._jit_integral(self.field.B()), dtype=np.float64)
            )
            self._cached_value = value

        self._cached_partials = None
        self._cached_points_version = current_points_version
        self.new_x = False
        return value

    @derivative_dec
    def dJ(self):
        current_points_version = getattr(self.field, "_points_version", None)
        cache_valid = (
            not self.new_x
            and self._cached_partials is not None
            and self._cached_points_version == current_points_version
        )
        if cache_valid:
            return self._cached_partials

        if self._use_jax_native:
            self._cached_value, self._cached_partials = self._value_and_dJ_jax_native()
            partials = self._cached_partials
        else:
            self._cached_value, partials = self._value_and_dJ_fallback()
            self._cached_partials = partials

        self._cached_points_version = current_points_version
        self.new_x = False
        return partials

    def _value_and_dJ_jax_native(self):
        """Combined value and gradient via end-to-end JAX value_and_grad."""
        flat_dofs = self._gather_unique_full_dofs()
        value, grad = self._jit_val_grad_dofs(flat_dofs)
        value_float = float(_host_scalar(value, dtype=np.float64))
        grad_np = _host_array(grad, dtype=np.float64)
        _raise_if_nonfinite_squared_flux_gradient(
            definition=self.definition,
            value=value_float,
            grad=grad_np,
        )

        # Map the flat gradient back to per-Optimizable Derivative entries.
        deriv_data = {}
        cds = self.field._curve_dof_size
        for i, curve in enumerate(self.field._unique_base_curves):
            deriv_data[curve] = grad_np[i * cds : (i + 1) * cds]

        current_start = len(self.field._unique_base_curves) * cds
        for i, current in enumerate(self.field._unique_base_currents):
            deriv_data[current] = grad_np[current_start + i : current_start + i + 1]

        return value_float, Derivative(deriv_data)

    def _value_and_dJ_fallback(self):
        """Combined value and gradient via field.B_vjp() (mixed quadrature)."""
        B = self.field.B()
        value, dJ_dB = self._jit_integral_value_grad(B)
        value_float = float(_host_scalar(value, dtype=np.float64))
        dJ_dB_host = _host_array(dJ_dB, dtype=np.float64)
        _raise_if_nonfinite_squared_flux_gradient(
            definition=self.definition,
            value=value_float,
            grad=dJ_dB_host,
        )
        return value_float, self.field.B_vjp(dJ_dB_host)
