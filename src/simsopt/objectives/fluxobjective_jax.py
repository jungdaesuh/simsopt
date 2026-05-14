"""
JAX-backed SquaredFlux objective for Stage 2 coil optimization.

``SquaredFluxJAX`` is a drop-in replacement for :class:`SquaredFlux` that
uses JAX for both forward evaluation and gradient computation.

Supported fields stay on a native JAX lane for the full
``DOFs → coil geometry/specs → BiotSavart → integral_BdotN`` chain.
Uniform ``CurveXYZFourier`` coil sets use the existing direct Fourier
fast path; more general JAX-capable coil families run through immutable
coil specs on the same JAX-native contract.

Unsupported fields are rejected by the native contract;
``field.B()`` / ``field.B_vjp()`` compatibility seams are not used.

The fixed surface is captured from its immutable ``surface_spec()``
once at construction time and kept on JAX arrays for the lifetime of
the objective. This is correct for Stage 2 where the plasma surface is
fixed.
"""

import numpy as np
import jax
import jax.numpy as jnp

from .._core.jax_host_boundary import (
    host_array as _host_array,
    host_scalar as _host_scalar,
)
from .._core.util import ObjectiveFailure
from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec, Derivative
from ..jax_core.biotsavart import biot_savart_B
from ..jax_core._math_utils import as_jax_float64 as _as_jax_float64
from ..jax_core.objectives_flux import (
    build_fourier_basis,
    fixed_surface_flux_integral,
    fixed_surface_flux_specs_from_surface,
    fixed_surface_flux_integral_from_B,
)
from ..jax_core import coil_specs_from_dof_extraction_spec
from ..jax_core.field import grouped_coil_set_spec_from_coil_specs

__all__ = [
    "SquaredFluxJAX",
    "coil_current_fixed_geometry_flux_jax",
    "coil_current_fixed_geometry_value_and_grad_jax",
]


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


def _strict_field_coil_dof_extraction_spec(field):
    coil_dof_extraction_spec = getattr(field, "coil_dof_extraction_spec", None)
    if not callable(coil_dof_extraction_spec):
        raise NotImplementedError(
            "SquaredFluxJAX requires a field exposing coil_dof_extraction_spec() "
            "for native JAX execution."
        )
    return coil_dof_extraction_spec()


def _strict_field_dof_layout_version(field) -> int:
    """Return the field's DOF-layout drift counter; field must expose it."""
    layout_version = getattr(field, "_dof_layout_version", None)
    if not isinstance(layout_version, int):
        raise NotImplementedError(
            "SquaredFluxJAX requires a field exposing integer "
            "_dof_layout_version for drift detection."
        )
    return layout_version


def coil_current_fixed_geometry_flux_jax(
    points,
    gammas,
    gammadashs,
    currents,
    flux_spec,
):
    """Return fixed-geometry coil-current normal-field flux in pure JAX."""
    B = biot_savart_B(
        _as_jax_float64(points),
        _as_jax_float64(gammas),
        _as_jax_float64(gammadashs),
        _as_jax_float64(currents),
    )
    return fixed_surface_flux_integral_from_B(B, flux_spec)


def coil_current_fixed_geometry_value_and_grad_jax(
    points,
    gammas,
    gammadashs,
    currents,
    flux_spec,
):
    """Return value and gradient with respect to fixed-geometry coil currents."""
    return jax.value_and_grad(
        lambda current_values: coil_current_fixed_geometry_flux_jax(
            points,
            gammas,
            gammadashs,
            current_values,
            flux_spec,
        )
    )(_as_jax_float64(currents))


def _field_dofs_gradient_to_derivative(field, field_dofs_gradient):
    field_dofs_gradient = np.asarray(field_dofs_gradient, dtype=np.float64)
    deriv_data = {}
    start = 0
    for lineage_opt in field.unique_dof_lineage:
        width = lineage_opt.local_dof_size
        if width == 0:
            continue

        stop = start + width
        block = np.zeros(lineage_opt.local_full_dof_size)
        block[lineage_opt.local_dofs_free_status] = field_dofs_gradient[start:stop]
        start = stop

        dep_opts = tuple(lineage_opt.dofs.dep_opts())
        block_share = block / len(dep_opts)
        for dep_opt in dep_opts:
            if dep_opt in deriv_data:
                deriv_data[dep_opt] = deriv_data[dep_opt] + block_share
            else:
                deriv_data[dep_opt] = block_share.copy()

    return Derivative(deriv_data)


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
    CPU round-trip or ``field.B_vjp()`` compatibility call is needed during
    optimization.

    .. note::
        The plasma surface must be fixed during the optimization.
        Surface geometry arrays are captured at construction time.

    Degenerate quadrature points follow the same contract as
    :class:`SquaredFlux`: zero-area elements contribute zero, while
    invalid ``"normalized"`` and ``"local"`` singular configurations
    evaluate to ``inf``.

    Args:
        surface: a fixed :class:`Surface` exposing ``surface_spec()``.
        field: a :class:`BiotSavartJAX` instance.
        target: optional ``(nphi, ntheta)`` target normal field (default 0).
        definition: ``"quadratic flux"`` | ``"normalized"`` | ``"local"``.
    """

    def __init__(
        self,
        surface,
        field,
        target=None,
        definition="quadratic flux",
    ):
        if definition not in ("quadratic flux", "normalized", "local"):
            raise ValueError(f"Unknown definition: {definition!r}")

        self.surface = surface
        self.field = field
        self.definition = definition

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
        self._field_points_version = field._points_version
        self._field_dof_layout_version = _strict_field_dof_layout_version(field)

        self._clear_cached_results()
        self._init_native_program(field)

        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[field])

    # ------------------------------------------------------------------
    # JAX-native path: DOFs → coil geometry/specs → B → J (single JIT program)
    # ------------------------------------------------------------------

    def _init_native_program(self, field):
        """Build the native JAX objective program for the current field contract."""
        if field._uses_uniform_curve_xyz_fourier_fastpath:
            self._init_curve_xyz_fourier_fastpath(field)
            return
        self._init_spec_native(field)

    def _bind_native_forward(self, forward):
        jit_forward = jax.jit(forward)
        jit_val_grad = jax.jit(jax.value_and_grad(forward, argnums=0))

        def _jit_forward_dofs(flat_dofs):
            return jit_forward(flat_dofs, self._flux_spec)

        def _jit_val_grad_dofs(flat_dofs):
            return jit_val_grad(flat_dofs, self._flux_spec)

        self._jit_forward_dofs = _jit_forward_dofs
        self._jit_val_grad_dofs = _jit_val_grad_dofs

    def _init_curve_xyz_fourier_fastpath(self, field):
        """Build the uniform-``CurveXYZFourier`` fast path."""
        order = field._curve_order
        basis, dbasis = build_fourier_basis(
            field._curve_quadpoints_jax,
            order,
        )

        k = 2 * order + 1

        # Static coil descriptors (unrolled by JIT tracer)
        base_curves = tuple(field._unique_base_curves)
        base_curve_idxs = tuple(d[0] for d in field._coil_descs)
        base_current_idxs = tuple(d[1] for d in field._coil_descs)
        rotmats = tuple(d[2] for d in field._coil_descs)
        current_scales = tuple(d[3] for d in field._coil_descs)
        n_coils = len(field._coil_descs)

        def forward(flat_dofs, flux_spec):
            curve_dofs = [
                field._local_full_dofs_from_free_vector(curve, flat_dofs)
                for curve in base_curves
            ]
            current_vals = [
                field._scalar_current_value_from_dofs(
                    current,
                    flat_dofs,
                    "uniform CurveXYZFourier fast path",
                )
                for current in field._unique_base_currents
            ]

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
                jnp.array(currents, dtype=jnp.float64),
            )
            return fixed_surface_flux_integral_from_B(B, flux_spec)

        self._bind_native_forward(forward)

    def _init_spec_native(self, field):
        """Build the immutable-spec native path for general JAX-capable fields."""
        coil_dof_extraction_spec = _strict_field_coil_dof_extraction_spec(field)

        def forward(flat_dofs, flux_spec):
            coil_specs = coil_specs_from_dof_extraction_spec(
                coil_dof_extraction_spec,
                flat_dofs,
            )
            coil_set_spec = grouped_coil_set_spec_from_coil_specs(coil_specs)
            return fixed_surface_flux_integral(coil_set_spec, flux_spec)

        self._bind_native_forward(forward)

    # ------------------------------------------------------------------
    # DOF gathering
    # ------------------------------------------------------------------

    def _gather_field_free_dofs(self):
        """Read the current flat free-DOF vector from the field dependency."""
        return _as_jax_float64(self.field.x)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _clear_cached_results(self):
        self._cached_value = None
        self._cached_partials = None

    def recompute_bell(self, parent=None):
        self._clear_cached_results()

    def _raise_if_field_points_drifted(self):
        if self.field._points_version == self._field_points_version:
            return
        raise RuntimeError(
            "SquaredFluxJAX captures fixed field-evaluation points at "
            "construction. Do not call field.set_points() after constructing "
            "SquaredFluxJAX; rebuild the objective for a new point set."
        )

    def _raise_if_field_dof_layout_drifted(self):
        layout_version = _strict_field_dof_layout_version(self.field)
        if layout_version == self._field_dof_layout_version:
            return
        raise RuntimeError(
            "SquaredFluxJAX captures the field free/fixed DOF layout at "
            "construction. Do not change coil/current free or fixed status "
            "after constructing SquaredFluxJAX; rebuild the objective for a "
            "new DOF layout."
        )

    def _raise_if_field_contract_drifted(self):
        self._raise_if_field_points_drifted()
        self._raise_if_field_dof_layout_drifted()

    def J(self):
        self._raise_if_field_contract_drifted()
        cache_valid = not self.new_x and self._cached_value is not None
        if cache_valid:
            return self._cached_value

        self._cached_value = float(
            _host_scalar(
                self._jit_forward_dofs(self._gather_field_free_dofs()),
                dtype=np.float64,
            )
        )
        value = self._cached_value
        self._cached_partials = None
        self.new_x = False
        return value

    @derivative_dec
    def dJ(self):
        self._raise_if_field_contract_drifted()
        cache_valid = not self.new_x and self._cached_partials is not None
        if cache_valid:
            return self._cached_partials

        self._cached_value, self._cached_partials = self._value_and_dJ_native()
        partials = self._cached_partials
        self.new_x = False
        return partials

    def _value_and_dJ_native(self):
        """Combined value and gradient via end-to-end JAX value_and_grad."""
        flat_dofs = self._gather_field_free_dofs()
        value, grad = self._jit_val_grad_dofs(flat_dofs)
        value_float = float(_host_scalar(value, dtype=np.float64))
        grad_np = _host_array(grad, dtype=np.float64)
        _raise_if_nonfinite_squared_flux_gradient(
            definition=self.definition,
            value=value_float,
            grad=grad_np,
        )
        return value_float, _field_dofs_gradient_to_derivative(self.field, grad_np)
