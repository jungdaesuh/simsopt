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
fixed; mutating the surface's free DOFs after construction raises a
``RuntimeError`` because the cached geometry would be stale.
"""

import hashlib
from functools import lru_cache

import numpy as np
import jax
import jax.numpy as jnp

from simsopt_jax.runtime.host_boundary import (
    host_array as _host_array,
    host_scalar as _host_scalar,
)
from simsopt._core.util import ObjectiveFailure
from simsopt._core.optimizable import Optimizable
from simsopt._core.derivative import derivative_dec
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt_jax.core.biotsavart import biot_savart_B
from simsopt_jax.core._math_utils import (
    as_jax_float64 as _as_jax_float64,
    as_jax_int32 as _as_jax_int32,
)
from simsopt_jax.core.objectives_flux import (
    build_fourier_basis,
    fixed_surface_flux_integral,
    fixed_surface_flux_specs_from_surface,
    fixed_surface_flux_integral_from_B,
)
from simsopt_jax.core.curve_geometry import optimizable_input_dofs_from_map_spec
from simsopt_jax.core.field import coil_set_spec_from_dof_extraction_spec
from simsopt_jax.core.surface_rzfourier import surface_rz_fourier_spec_from_dofs
from simsopt_jax_adapters.geo.curve_specs import adapter_curve_dof_mode

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
    layout_version = getattr(field, "dof_layout_version", None)
    if not isinstance(layout_version, int):
        raise NotImplementedError(
            "SquaredFluxJAX requires a field exposing integer "
            "dof_layout_version for drift detection."
        )
    return layout_version


def _surface_dofs_fingerprint(surface) -> bytes:
    """Return a fixed-width content fingerprint of a surface's full DOF state.

    Uses ``local_full_x`` (the surface's complete DOF vector including
    fixed entries) rather than ``surface.x`` (free DOFs only). A surface
    that is fully fixed at construction would otherwise hash an empty
    array, and a downstream caller that mutates ``surface.local_full_x``
    or rebinds the underlying values would not be detected.

    blake2b over the contiguous float64 byte view is collision-resistant
    and fast enough that the per-call overhead is negligible compared to
    the JAX forward pass. Using bytes (not the array itself) keeps the
    comparison in the host scope and avoids accidentally pulling JAX
    device buffers across a transfer guard.
    """
    dofs = np.ascontiguousarray(surface.local_full_x, dtype=np.float64)
    return hashlib.blake2b(dofs.tobytes(), digest_size=16).digest()


class _SurfaceSpecProvider:
    def __init__(self, surface_spec) -> None:
        self._surface_spec = surface_spec

    def surface_spec(self):
        return self._surface_spec


def _surface_spec_from_adapter_surface(surface):
    surface_spec_fn = getattr(surface, "surface_spec", None)
    if callable(surface_spec_fn):
        return surface_spec_fn()

    if isinstance(surface, SurfaceRZFourier):
        return surface_rz_fourier_spec_from_dofs(
            _as_jax_float64(surface.get_dofs()),
            quadpoints_phi=_as_jax_float64(surface.quadpoints_phi),
            quadpoints_theta=_as_jax_float64(surface.quadpoints_theta),
            mpol=surface.mpol,
            ntor=surface.ntor,
            nfp=surface.nfp,
            stellsym=surface.stellsym,
        )

    raise NotImplementedError(
        "SquaredFluxJAX fixed-surface setup requires a surface exposing "
        f"surface_spec(); unsupported adapter surface {type(surface).__name__}."
    )


def _fixed_surface_flux_specs_from_adapter_surface(
    surface,
    *,
    target,
    definition: str,
):
    return fixed_surface_flux_specs_from_surface(
        _SurfaceSpecProvider(_surface_spec_from_adapter_surface(surface)),
        target=target,
        definition=definition,
    )


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


@lru_cache(maxsize=2)
def _cached_squared_flux_native_program(forward):
    return jax.jit(forward), jax.jit(jax.value_and_grad(forward, argnums=0))


def _squared_flux_curve_xyz_fourier_forward(
    flat_dofs,
    flux_spec,
    basis,
    dbasis,
    curve_dof_maps,
    current_dof_maps,
    base_curve_idxs,
    base_current_idxs,
    rotmats,
    current_scales,
):
    k = basis.shape[1]
    curve_dofs = jnp.stack(
        [
            optimizable_input_dofs_from_map_spec(curve_map, flat_dofs).reshape(3, k)
            for curve_map in curve_dof_maps
        ]
    )
    current_vals = jnp.stack(
        [
            optimizable_input_dofs_from_map_spec(current_map, flat_dofs)[0]
            for current_map in current_dof_maps
        ]
    )

    coeffs = curve_dofs[base_curve_idxs]
    gammas = jnp.einsum("qk,nck->nqc", basis, coeffs)
    gammadashs = jnp.einsum("qk,nck->nqc", dbasis, coeffs)
    gammas = jnp.einsum("nqi,nij->nqj", gammas, rotmats)
    gammadashs = jnp.einsum("nqi,nij->nqj", gammadashs, rotmats)
    currents = current_vals[base_current_idxs] * current_scales

    B = biot_savart_B(
        flux_spec.points,
        gammas,
        gammadashs,
        currents,
    )
    return fixed_surface_flux_integral_from_B(B, flux_spec)


def _squared_flux_spec_native_forward(
    flat_dofs,
    flux_spec,
    coil_dof_extraction_spec,
):
    coil_set_spec = coil_set_spec_from_dof_extraction_spec(
        coil_dof_extraction_spec,
        flat_dofs,
    )
    return fixed_surface_flux_integral(coil_set_spec, flux_spec)


def _field_dofs_gradient_to_derivative(field, field_dofs_gradient):
    return field.dofs_gradient_to_derivative(field_dofs_gradient)


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
        field_eval_spec, self._flux_spec = (
            _fixed_surface_flux_specs_from_adapter_surface(
                surface,
                target=target_array,
                definition=definition,
            )
        )

        # Set evaluation points on the field adapter from the immutable spec.
        field.set_points_from_spec(field_eval_spec)
        self._field_points_version = field._points_version
        self._field_dof_layout_version = _strict_field_dof_layout_version(field)
        self._field_coil_dof_extraction_spec = (
            _strict_field_coil_dof_extraction_spec(field)
        )
        # Surface DOFs are baked into the JIT-captured ``_flux_spec``.
        # We capture a fingerprint here
        # so that any post-construction mutation of ``surface.x`` is detected
        # at the next ``J()`` / ``dJ()`` call instead of silently producing
        # stale results.
        self._surface_dofs_fingerprint = _surface_dofs_fingerprint(surface)

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

    def _bind_native_forward(self, forward, *forward_args):
        jit_forward, jit_val_grad = _cached_squared_flux_native_program(forward)

        def _jit_forward_dofs(flat_dofs):
            return jit_forward(flat_dofs, self._flux_spec, *forward_args)

        def _jit_val_grad_dofs(flat_dofs):
            return jit_val_grad(flat_dofs, self._flux_spec, *forward_args)

        self._jit_forward_dofs = _jit_forward_dofs
        self._jit_val_grad_dofs = _jit_val_grad_dofs

    def _init_curve_xyz_fourier_fastpath(self, field):
        """Build the uniform-``CurveXYZFourier`` fast path."""
        order = field._curve_order
        basis, dbasis = build_fourier_basis(
            field._curve_quadpoints_jax,
            order,
        )

        # Static coil descriptors.
        base_curves = tuple(field._unique_base_curves)
        curve_dof_maps = tuple(
            field._free_vector_dof_map_spec(
                curve,
                full_graph=adapter_curve_dof_mode(curve) == "full",
            )
            for curve in base_curves
        )
        current_dof_maps = tuple(
            field._free_vector_dof_map_spec(current, full_graph=False)
            for current in field._unique_base_currents
        )
        coil_descs = tuple(field._coil_descs)
        base_curve_idxs = _as_jax_int32([d[0] for d in coil_descs])
        base_current_idxs = _as_jax_int32([d[1] for d in coil_descs])
        rotmats = _as_jax_float64(
            np.stack(
                [
                    np.eye(3, dtype=np.float64)
                    if d[2] is None
                    else _host_array(d[2], dtype=np.float64)
                    for d in coil_descs
                ]
            )
        )
        current_scales = _as_jax_float64([d[3] for d in coil_descs])

        self._bind_native_forward(
            _squared_flux_curve_xyz_fourier_forward,
            basis,
            dbasis,
            curve_dof_maps,
            current_dof_maps,
            base_curve_idxs,
            base_current_idxs,
            rotmats,
            current_scales,
        )

    def _init_spec_native(self, field):
        """Build the immutable-spec native path for general JAX-capable fields."""
        coil_dof_extraction_spec = _strict_field_coil_dof_extraction_spec(field)
        self._bind_native_forward(
            _squared_flux_spec_native_forward,
            coil_dof_extraction_spec,
        )

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

    def _raise_if_field_coil_dof_extraction_drifted(self):
        extraction_spec = _strict_field_coil_dof_extraction_spec(self.field)
        if extraction_spec is self._field_coil_dof_extraction_spec:
            return
        raise RuntimeError(
            "SquaredFluxJAX captures the immutable coil DOF extraction contract "
            "at construction. Fixed coil/current DOF values have changed; "
            "rebuild the objective for the new fixed state."
        )

    def _raise_if_surface_dofs_drifted(self):
        if _surface_dofs_fingerprint(self.surface) == self._surface_dofs_fingerprint:
            return
        raise RuntimeError(
            "SquaredFluxJAX captures fixed surface geometry at construction "
            "(gamma, normal, target are baked into the JIT closure). The "
            "surface free DOFs have changed since construction; rebuild the "
            "objective for the new surface state."
        )

    def _raise_if_field_contract_drifted(self):
        self._raise_if_field_points_drifted()
        self._raise_if_field_dof_layout_drifted()
        self._raise_if_field_coil_dof_extraction_drifted()
        self._raise_if_surface_dofs_drifted()

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

    def value_and_dJ(self, *, partials=False):
        """Return ``(J, dJ)`` through one native value-and-gradient program."""
        self._raise_if_field_contract_drifted()
        if (
            not self.new_x
            and self._cached_value is not None
            and self._cached_partials is not None
        ):
            value, partials_derivative = self._cached_value, self._cached_partials
        else:
            value, partials_derivative = self._value_and_dJ_native()
            self._cached_value = value
            self._cached_partials = partials_derivative
            self.new_x = False

        if partials:
            return value, partials_derivative
        return value, partials_derivative(self)

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
