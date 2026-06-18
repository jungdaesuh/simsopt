"""
JAX-backed Optimizable wrappers for single-stage objectives.

These wrappers mirror the CPU ``BoozerResidual``, ``Iotas``, and
``NonQuasiSymmetricRatio`` classes but use JAX for field evaluation
and gradient computation.

Architecture (implicit differentiation):

  For any outer objective J that depends on the inner Boozer solution
  x*(coils), the total derivative is:

  .. math::

      \\frac{dJ}{d\\text{coils}} = \\frac{\\partial J}{\\partial \\text{coils}}
      - \\text{adj}^T \\frac{\\partial g}{\\partial \\text{coils}}

  where adj solves the inner transposed linearization system
  ``dg/dx_inner^T adj = ∂J/∂x_inner`` and g is the stationarity condition
  of the inner solve.

  Wrapper consumers now obtain solved/adjoint state through explicit
  runtime-summary accessors on ``BoozerSurfaceJAX``. The mutable
  ``run_code()`` result dict remains the compatibility lane owned by
  ``BoozerSurfaceJAX`` itself.
"""

import hashlib
import logging
import os
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from simsopt._core.derivative import Derivative, derivative_dec
from simsopt_jax.runtime.host_boundary import (
    explicit_cotangent_basis as _explicit_cotangent_basis,
    host_array as _host_array,
    host_bool as _host_bool,
    host_inf_norm as _host_inf_norm,
    host_scalar as _host_scalar,
    scalar_pullback_seed as _explicit_scalar_pullback_seed,
)
from simsopt._core.optimizable import Optimizable
from simsopt_jax.core._math_utils import (
    as_jax_float64 as _as_jax_float64,
    as_runtime_float64 as _as_runtime_float64,
    runtime_device_put,
    zeros as _zeros,
)
from simsopt_jax.core.curve_geometry import curve_geometry_from_spec
from simsopt_jax.core.field import (
    coil_set_spec_from_dof_extraction_spec,
    coil_specs_from_dof_extraction_spec,
    grouped_biot_savart_B_from_spec,
    grouped_coil_currents_from_spec,
)
from simsopt_jax.core.qfm_solver import (
    qfm_label_jax_from_dofs,
    qfm_penalty_jax_from_dofs,
    qfm_penalty_value_and_grad_jax_from_dofs,
    qfm_residual_jax_from_dofs,
)
from simsopt_jax.core._surface_dofs_dispatch import (
    surface_gamma_tangents_from_dofs as _surface_gamma_tangents_from_dofs,
    surface_spec_with_dofs as _surface_spec_with_dofs,
    surface_volume_from_dofs as _surface_volume_from_dofs,
)
from simsopt_jax.core._device_scalars import (
    device_one as _device_one,
    two_pi as _two_pi,
)
from simsopt_jax.core.specs import (
    make_surface_xyz_fourier_spec,
    make_surface_xyz_tensor_fourier_spec,
    surface_spec_kind,
)
from simsopt_jax.core.surface_fourier import (
    surface_xyz_fourier_gamma_from_spec,
    surface_xyz_fourier_gammadash1_from_spec,
    surface_xyz_fourier_gammadash1dash1_from_spec,
    surface_xyz_fourier_gammadash1dash2_from_spec,
    surface_xyz_fourier_gammadash2_from_spec,
    surface_xyz_fourier_gammadash2dash2_from_spec,
    surface_xyz_tensor_fourier_gamma_from_spec,
    surface_xyz_tensor_fourier_gammadash1_from_spec,
    surface_xyz_tensor_fourier_gammadash1dash1_from_spec,
    surface_xyz_tensor_fourier_gammadash1dash2_from_spec,
    surface_xyz_tensor_fourier_gammadash2_from_spec,
    surface_xyz_tensor_fourier_gammadash2dash2_from_spec,
)
from simsopt_jax.core.surface_rzfourier import (
    surface_rz_fourier_spec_from_dofs,
    surface_rz_fourier_gamma_from_dofs,
    surface_rz_fourier_gammadash1_from_dofs,
    surface_rz_fourier_gammadash1dash1_from_dofs,
    surface_rz_fourier_gammadash1dash2_from_dofs,
    surface_rz_fourier_gammadash2_from_dofs,
    surface_rz_fourier_gammadash2dash2_from_dofs,
)
from simsopt.geo.curve import incremental_arclength_pure, kappa_pure
from simsopt_jax.geo._pairwise_reductions import (
    _resolve_pairwise_penalty_chunk_size,
    pairwise_min_distance_pure,
    pairwise_selected_smoothmin_distance_batched_pure,
    pairwise_selected_smoothmin_distance_pure,
    pairwise_thresholded_mean_square_distance_pure,
)
from simsopt_jax_adapters.geo.curve_objectives import (
    Lp_curvature_pure,
    cc_distance_pure,
    cs_distance_pure,
    curve_length_pure,
)
from simsopt_jax.geo.boozer_residual import (
    boozer_residual_scalar,
    _surface_geometry_from_dofs,
)
from .boozer_surface import (
    _BoozerPenaltyGeometry,
    _compute_label,
)
from simsopt_jax.geo.optimizers import optimizer as _optimizer_jax
from simsopt_jax.geo.label_constraints import compute_G_from_currents
from simsopt_jax.geo._surface_stellsym import (
    compute_stellsym_mask_indices_for_grid as _compute_stellsym_mask_indices_for_grid,
)
from simsopt.geo.surface import Surface


def _surface_spec_from_surface(surface):
    surface_type = type(surface).__name__
    if surface_type == "SurfaceRZFourier":
        return surface_rz_fourier_spec_from_dofs(
            _as_jax_float64(surface.get_dofs()),
            quadpoints_phi=_as_jax_float64(surface.quadpoints_phi),
            quadpoints_theta=_as_jax_float64(surface.quadpoints_theta),
            mpol=surface.mpol,
            ntor=surface.ntor,
            nfp=surface.nfp,
            stellsym=surface.stellsym,
        )
    if surface_type == "SurfaceXYZFourier":
        return make_surface_xyz_fourier_spec(
            dofs=_as_jax_float64(surface.get_dofs()),
            quadpoints_phi=_as_jax_float64(surface.quadpoints_phi),
            quadpoints_theta=_as_jax_float64(surface.quadpoints_theta),
            nfp=surface.nfp,
            stellsym=surface.stellsym,
            mpol=surface.mpol,
            ntor=surface.ntor,
        )
    if surface_type == "SurfaceXYZTensorFourier":
        return make_surface_xyz_tensor_fourier_spec(
            dofs=_as_jax_float64(surface.get_dofs()),
            quadpoints_phi=_as_jax_float64(surface.quadpoints_phi),
            quadpoints_theta=_as_jax_float64(surface.quadpoints_theta),
            nfp=surface.nfp,
            stellsym=surface.stellsym,
            mpol=surface.mpol,
            ntor=surface.ntor,
            clamped_dims=tuple(
                getattr(surface, "clamped_dims", (False, False, False))
            ),
        )
    raise NotImplementedError(
        "JAX surface objective adapters require an explicit spec builder for "
        f"{surface_type}."
    )


def surface_to_surface_distance_pure(gamma1, gamma2, mdist):
    gamma1 = _as_jax_float64(gamma1)
    gamma2 = _as_jax_float64(gamma2)
    mdist = _as_runtime_float64(mdist, reference=gamma1)
    gamma1 = gamma1.reshape((-1, 3))
    gamma2 = gamma2.reshape((-1, 3))
    return pairwise_thresholded_mean_square_distance_pure(gamma1, gamma2, mdist)

__all__ = [
    "AreaJAX",
    "AspectRatioJAX",
    "BoozerResidualJAX",
    "IotasJAX",
    "MajorRadiusJAX",
    "NonQuasiSymmetricRatioJAX",
    "PrincipalCurvatureJAX",
    "QfmResidualJAX",
    "SurfaceSurfaceDistance",
    "VolumeJAX",
    "coil_dofs_gradient_to_derivative",
    "compute_standard_surface_objective_gradients",
    "make_traceable_single_stage_alm_runtime_bundle",
    "make_traceable_objective",
    "make_traceable_objective_runtime_bundle",
    "make_traceable_objective_seeded_value_and_grad",
    "make_traceable_objective_value_and_grad",
    "make_traceable_solved_state_value_and_grad",
    "make_traceable_objective_profile_suite",
    "surface_area_jax_from_dofs",
    "surface_aspect_ratio_jax_from_dofs",
    "surface_curvatures_jax_from_dofs",
    "surface_d2area_jax_from_dofs",
    "surface_d2aspect_ratio_jax_from_dofs",
    "surface_d2major_radius_jax_from_dofs",
    "surface_d2mean_cross_sectional_area_jax_from_dofs",
    "surface_d2minor_radius_jax_from_dofs",
    "surface_d2volume_jax_from_dofs",
    "surface_darea_jax_from_dofs",
    "surface_daspect_ratio_jax_from_dofs",
    "surface_dmajor_radius_jax_from_dofs",
    "surface_dmean_cross_sectional_area_jax_from_dofs",
    "surface_dminor_radius_jax_from_dofs",
    "surface_dsurface_curvatures_jax_from_dofs",
    "surface_dvolume_jax_from_dofs",
    "surface_major_radius_jax_from_dofs",
    "surface_mean_cross_sectional_area_jax_from_dofs",
    "surface_minor_radius_jax_from_dofs",
    "surface_principal_curvature_jax_from_dofs",
    "surface_qfm_residual_jax_from_dofs",
    "surface_qfm_label_jax_from_dofs",
    "surface_qfm_penalty_jax_from_dofs",
    "surface_qfm_penalty_value_and_grad_jax_from_dofs",
    "surface_volume_jax_from_dofs",
]


_MISSING_STREAMING_GROUP_VJP_ERROR = (
    "BoozerSurfaceJAX objective wrappers require a streaming grouped-adjoint "
    "callback; the legacy full-pytree adjoint fallback is no longer supported."
)


class SurfaceSurfaceDistance(Optimizable):
    """JAX-backed thresholded distance penalty between two sampled surfaces."""

    def __init__(self, surf1, surf2, minimum_distance, *, chunk_size=None):
        self.surf1 = surf1
        self.surf2 = surf2
        self.minimum_distance = float(minimum_distance)
        self.chunk_size = (
            None
            if chunk_size is None
            else _resolve_pairwise_penalty_chunk_size(chunk_size)
        )
        self._value_and_grad = jax.jit(
            jax.value_and_grad(
                lambda gamma1, gamma2: pairwise_thresholded_mean_square_distance_pure(
                    gamma1,
                    gamma2,
                    self.minimum_distance,
                    chunk_size=self.chunk_size,
                ),
                argnums=(0, 1),
            )
        )
        self._shortest_distance = jax.jit(
            lambda gamma1, gamma2: pairwise_min_distance_pure(
                gamma1,
                gamma2,
                chunk_size=self.chunk_size,
            )
        )
        super().__init__(depends_on=[surf1, surf2])

    def _flattened_surface_gammas(self):
        gamma1 = jnp.asarray(self.surf1.gamma(), dtype=jnp.float64)
        gamma2 = jnp.asarray(self.surf2.gamma(), dtype=jnp.float64)
        return (
            jnp.reshape(gamma1, (-1, 3)),
            jnp.reshape(gamma2, (-1, 3)),
            gamma1.shape,
            gamma2.shape,
        )

    def J(self):
        gamma1, gamma2, _gamma1_shape, _gamma2_shape = self._flattened_surface_gammas()
        value, _gradients = self._value_and_grad(gamma1, gamma2)
        return _host_scalar(value)

    def shortest_distance(self):
        gamma1, gamma2, _gamma1_shape, _gamma2_shape = self._flattened_surface_gammas()
        return _host_scalar(self._shortest_distance(gamma1, gamma2))

    @derivative_dec
    def dJ(self):
        gamma1, gamma2, gamma1_shape, gamma2_shape = self._flattened_surface_gammas()
        _value, (grad1, grad2) = self._value_and_grad(gamma1, gamma2)
        return Derivative(
            {
                self.surf1: self.surf1.dgamma_by_dcoeff_vjp(
                    _host_array(jnp.reshape(grad1, gamma1_shape), dtype=np.float64)
                ),
                self.surf2: self.surf2.dgamma_by_dcoeff_vjp(
                    _host_array(jnp.reshape(grad2, gamma2_shape), dtype=np.float64)
                ),
            }
        )

_TRACEABLE_RUNTIME_OPTION_KEYS = (
    "optimizer_backend",
    "least_squares_algorithm",
    "limited_memory",
    "force_ondevice_limited_memory",
    "weight_inv_modB",
    "bfgs_maxiter",
    "bfgs_tol",
    "newton_maxiter",
    "newton_tol",
    "newton_stab",
    "materialize_dense_linearization",
    "max_dense_linearization_bytes",
)

_TRACEABLE_ADJOINT_FAIL_GRAD_SENTINEL = np.nan


def _traceable_diag_progress(message):
    """Emit optional progress logs for the target-lane baseline diagnosis."""
    raw_value = os.environ.get("SIMSOPT_TRACEABLE_DIAG_PROGRESS")
    if raw_value is None:
        return
    if raw_value.strip().lower() in {"", "0", "false", "no", "off"}:
        return
    print(f"[traceable-runtime-diagnose] {message}", flush=True)


logger = logging.getLogger(__name__)

_TRACEABLE_SINGLE_STAGE_OUTER_TERM_SPECS = (
    ("non_qs", "non_qs_weight"),
    ("residual", "residual_weight"),
    ("iota", "iota_weight"),
    ("length", "length_weight"),
    ("curvature", "curvature_weight"),
    ("curve_curve", "curve_curve_weight"),
    ("curve_surface", "curve_surface_weight"),
    ("surface_vessel", "surface_vessel_weight"),
)

_TRACEABLE_SINGLE_STAGE_OUTER_TERM_WEIGHT_KEYS = {
    term_name: weight_key
    for term_name, weight_key in _TRACEABLE_SINGLE_STAGE_OUTER_TERM_SPECS
}

_TRACEABLE_SINGLE_STAGE_OUTER_TERM_DEPENDENCY_FLAGS = {
    "non_qs": (True, True),
    "residual": (True, True),
    "iota": (True, False),
    "length": (False, True),
    "curvature": (False, True),
    "curve_curve": (False, True),
    "curve_surface": (True, True),
    "surface_vessel": (True, False),
}


def _traceable_single_stage_outer_term_dependency_flags(term_name):
    """Return which state families a diagnostic outer term depends on."""
    if term_name is None:
        return True, True
    try:
        return _TRACEABLE_SINGLE_STAGE_OUTER_TERM_DEPENDENCY_FLAGS[term_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown traceable single-stage outer term {term_name!r}."
        ) from exc


def _traceable_single_stage_weight_is_active(weight):
    return float(_host_scalar(weight)) != 0.0


def _traceable_single_stage_effective_dependency_flags(
    term_name,
    *,
    objective_kwargs,
):
    """Resolve effective dependencies after applying configured outer weights."""
    outer_objective_config = objective_kwargs.get("outer_objective_config")
    if outer_objective_config is None:
        return _traceable_single_stage_outer_term_dependency_flags(term_name)

    if term_name is not None:
        weight_key = _TRACEABLE_SINGLE_STAGE_OUTER_TERM_WEIGHT_KEYS[term_name]
        if not _traceable_single_stage_weight_is_active(
            outer_objective_config.get(weight_key, 0.0)
        ):
            return False, False
        return _traceable_single_stage_outer_term_dependency_flags(term_name)

    depends_on_x_inner = False
    depends_on_coil_dofs = False
    for candidate_term_name, weight_key in _TRACEABLE_SINGLE_STAGE_OUTER_TERM_SPECS:
        if not _traceable_single_stage_weight_is_active(
            outer_objective_config.get(weight_key, 0.0)
        ):
            continue
        (
            candidate_depends_on_x_inner,
            candidate_depends_on_coil_dofs,
        ) = _traceable_single_stage_outer_term_dependency_flags(candidate_term_name)
        depends_on_x_inner = depends_on_x_inner or candidate_depends_on_x_inner
        depends_on_coil_dofs = depends_on_coil_dofs or candidate_depends_on_coil_dofs
    return depends_on_x_inner, depends_on_coil_dofs


def _strict_scalar_grad(fun, arg):
    value, pullback = jax.vjp(fun, arg)
    (gradient,) = pullback(_explicit_scalar_pullback_seed(value))
    return gradient


def _strict_scalar_value_and_grad(fun, arg, *args):
    def _objective(first_arg):
        return fun(first_arg, *args)

    value, pullback = jax.vjp(_objective, arg)
    (gradient,) = pullback(_explicit_scalar_pullback_seed(value))
    return value, gradient


def _surface_geometry_second_derivatives_from_dofs(spec, dofs):
    dofs = _as_jax_float64(dofs)
    kind = surface_spec_kind(spec)
    if kind == "rz_fourier":
        return (
            surface_rz_fourier_gamma_from_dofs(spec, dofs),
            surface_rz_fourier_gammadash1_from_dofs(spec, dofs),
            surface_rz_fourier_gammadash2_from_dofs(spec, dofs),
            surface_rz_fourier_gammadash1dash1_from_dofs(spec, dofs),
            surface_rz_fourier_gammadash1dash2_from_dofs(spec, dofs),
            surface_rz_fourier_gammadash2dash2_from_dofs(spec, dofs),
        )
    if kind == "xyz_fourier":
        spec_with_dofs = _surface_spec_with_dofs(spec, dofs)
        return (
            surface_xyz_fourier_gamma_from_spec(spec_with_dofs),
            surface_xyz_fourier_gammadash1_from_spec(spec_with_dofs),
            surface_xyz_fourier_gammadash2_from_spec(spec_with_dofs),
            surface_xyz_fourier_gammadash1dash1_from_spec(spec_with_dofs),
            surface_xyz_fourier_gammadash1dash2_from_spec(spec_with_dofs),
            surface_xyz_fourier_gammadash2dash2_from_spec(spec_with_dofs),
        )
    if kind == "xyz_tensor_fourier":
        spec_with_dofs = _surface_spec_with_dofs(spec, dofs)
        return (
            surface_xyz_tensor_fourier_gamma_from_spec(spec_with_dofs),
            surface_xyz_tensor_fourier_gammadash1_from_spec(spec_with_dofs),
            surface_xyz_tensor_fourier_gammadash2_from_spec(spec_with_dofs),
            surface_xyz_tensor_fourier_gammadash1dash1_from_spec(spec_with_dofs),
            surface_xyz_tensor_fourier_gammadash1dash2_from_spec(spec_with_dofs),
            surface_xyz_tensor_fourier_gammadash2dash2_from_spec(spec_with_dofs),
        )
    raise TypeError(f"Unsupported surface spec kind {kind!r}.")


def _surface_normal_from_tangents(gammadash1, gammadash2):
    return jnp.cross(gammadash1, gammadash2)


def _surface_norm(normal):
    return jnp.sqrt(jnp.sum(normal * normal, axis=-1))


def _surface_curvatures_from_derivatives(
    gammadash1,
    gammadash2,
    gammadash1dash1,
    gammadash1dash2,
    gammadash2dash2,
):
    normal = _surface_normal_from_tangents(gammadash1, gammadash2)
    unitnormal = normal / _surface_norm(normal)[:, :, None]
    e = jnp.sum(gammadash1 * gammadash1, axis=-1)
    f = jnp.sum(gammadash1 * gammadash2, axis=-1)
    g = jnp.sum(gammadash2 * gammadash2, axis=-1)
    ell = jnp.sum(unitnormal * gammadash1dash1, axis=-1)
    m = jnp.sum(unitnormal * gammadash1dash2, axis=-1)
    n = jnp.sum(unitnormal * gammadash2dash2, axis=-1)
    denom = e * g - f * f
    mean_curvature = (ell * g - 2.0 * f * m + n * e) / (2.0 * denom)
    gaussian_curvature = (ell * n - m * m) / denom
    principal_offset = jnp.sqrt(mean_curvature * mean_curvature - gaussian_curvature)
    return jnp.stack(
        [
            mean_curvature,
            gaussian_curvature,
            mean_curvature + principal_offset,
            mean_curvature - principal_offset,
        ],
        axis=-1,
    )


def _surface_normal_norm_and_curvatures_jax_from_dofs(spec, dofs):
    (
        _gamma,
        gammadash1,
        gammadash2,
        gammadash1dash1,
        gammadash1dash2,
        gammadash2dash2,
    ) = _surface_geometry_second_derivatives_from_dofs(spec, dofs)
    normal = _surface_normal_from_tangents(gammadash1, gammadash2)
    return _surface_norm(normal), _surface_curvatures_from_derivatives(
        gammadash1,
        gammadash2,
        gammadash1dash1,
        gammadash1dash2,
        gammadash2dash2,
    )


def _surface_normal_norm_jax_from_dofs(spec, dofs):
    norm_normal, _curvature = _surface_normal_norm_and_curvatures_jax_from_dofs(
        spec,
        dofs,
    )
    return norm_normal


def surface_area_jax_from_dofs(spec, dofs):
    _gamma, gammadash1, gammadash2 = _surface_gamma_tangents_from_dofs(spec, dofs)
    normal = _surface_normal_from_tangents(gammadash1, gammadash2)
    return jnp.mean(_surface_norm(normal))


def surface_volume_jax_from_dofs(spec, dofs):
    return _surface_volume_from_dofs(spec, dofs)


def _surface_scalar_grad_jax_from_dofs(surface_scalar_fn, spec, dofs):
    return jax.jit(
        lambda surface_spec, x: jax.grad(lambda y: surface_scalar_fn(surface_spec, y))(
            x
        )
    )(spec, _as_jax_float64(dofs))


def surface_darea_jax_from_dofs(spec, dofs):
    return _surface_scalar_grad_jax_from_dofs(surface_area_jax_from_dofs, spec, dofs)


def surface_dvolume_jax_from_dofs(spec, dofs):
    return _surface_scalar_grad_jax_from_dofs(surface_volume_jax_from_dofs, spec, dofs)


def surface_d2area_jax_from_dofs(spec, dofs):
    return jax.hessian(lambda x: surface_area_jax_from_dofs(spec, x))(
        _as_jax_float64(dofs)
    )


def surface_d2volume_jax_from_dofs(spec, dofs):
    return jax.hessian(lambda x: surface_volume_jax_from_dofs(spec, x))(
        _as_jax_float64(dofs)
    )


def surface_curvatures_jax_from_dofs(spec, dofs):
    _norm_normal, curvature = _surface_normal_norm_and_curvatures_jax_from_dofs(
        spec,
        dofs,
    )
    return curvature


def surface_dsurface_curvatures_jax_from_dofs(spec, dofs):
    return jax.jacobian(lambda x: surface_curvatures_jax_from_dofs(spec, x))(
        _as_jax_float64(dofs)
    )


def _surface_dnormal_norm_jax_from_dofs(spec, dofs):
    return jax.jacobian(lambda x: _surface_normal_norm_jax_from_dofs(spec, x))(
        _as_jax_float64(dofs)
    )


def surface_mean_cross_sectional_area_jax_from_dofs(spec, dofs):
    if surface_spec_kind(spec) == "xyz_tensor_fourier" and any(spec.clamped_dims):
        raise NotImplementedError(
            "SurfaceXYZTensorFourier clamped_dims are not supported by "
            "surface_mean_cross_sectional_area_jax_from_dofs."
        )
    gamma, gammadash1, gammadash2 = _surface_gamma_tangents_from_dofs(
        spec,
        dofs,
    )
    x, y, _z = jnp.split(gamma, [1, 2], axis=2)
    gammadash1_x, gammadash1_y, gammadash1_z = jnp.split(gammadash1, [1, 2], axis=2)
    gammadash2_x, gammadash2_y, gammadash2_z = jnp.split(gammadash2, [1, 2], axis=2)
    radius_squared = x * x + y * y
    jacobian_00 = (x * gammadash1_y - y * gammadash1_x) / radius_squared
    jacobian_01 = (x * gammadash2_y - y * gammadash2_x) / radius_squared
    dz_dtheta = gammadash2_z - (gammadash1_z * jacobian_01 / jacobian_00)
    signed_area = jnp.mean(
        jnp.sqrt(radius_squared) * dz_dtheta * jacobian_00
    ) / _two_pi(radius_squared)
    return jnp.abs(signed_area)


def surface_minor_radius_jax_from_dofs(spec, dofs):
    mean_area = surface_mean_cross_sectional_area_jax_from_dofs(spec, dofs)
    one = _device_one(mean_area)
    pi = _two_pi(mean_area) / (one + one)
    return jnp.sqrt(mean_area / pi)


def surface_major_radius_jax_from_dofs(spec, dofs):
    volume = _surface_volume_from_dofs(spec, dofs)
    minor_radius = surface_minor_radius_jax_from_dofs(spec, dofs)
    one = _device_one(minor_radius)
    two_pi = _two_pi(minor_radius)
    pi = two_pi / (one + one)
    return jnp.abs(volume) / (two_pi * pi * minor_radius * minor_radius)


def surface_aspect_ratio_jax_from_dofs(spec, dofs):
    return surface_major_radius_jax_from_dofs(
        spec, dofs
    ) / surface_minor_radius_jax_from_dofs(spec, dofs)


def surface_dmean_cross_sectional_area_jax_from_dofs(spec, dofs):
    return jax.grad(lambda x: surface_mean_cross_sectional_area_jax_from_dofs(spec, x))(
        _as_jax_float64(dofs)
    )


def surface_dminor_radius_jax_from_dofs(spec, dofs):
    return jax.grad(lambda x: surface_minor_radius_jax_from_dofs(spec, x))(
        _as_jax_float64(dofs)
    )


def surface_dmajor_radius_jax_from_dofs(spec, dofs):
    return jax.grad(lambda x: surface_major_radius_jax_from_dofs(spec, x))(
        _as_jax_float64(dofs)
    )


def surface_daspect_ratio_jax_from_dofs(spec, dofs):
    return jax.grad(lambda x: surface_aspect_ratio_jax_from_dofs(spec, x))(
        _as_jax_float64(dofs)
    )


def surface_d2mean_cross_sectional_area_jax_from_dofs(spec, dofs):
    return jax.hessian(
        lambda x: surface_mean_cross_sectional_area_jax_from_dofs(spec, x)
    )(_as_jax_float64(dofs))


def surface_d2minor_radius_jax_from_dofs(spec, dofs):
    return jax.hessian(lambda x: surface_minor_radius_jax_from_dofs(spec, x))(
        _as_jax_float64(dofs)
    )


def surface_d2major_radius_jax_from_dofs(spec, dofs):
    return jax.hessian(lambda x: surface_major_radius_jax_from_dofs(spec, x))(
        _as_jax_float64(dofs)
    )


def surface_d2aspect_ratio_jax_from_dofs(spec, dofs):
    return jax.hessian(lambda x: surface_aspect_ratio_jax_from_dofs(spec, x))(
        _as_jax_float64(dofs)
    )


_surface_dmajor_radius_jax_from_dofs = surface_dmajor_radius_jax_from_dofs

_surface_daspect_ratio_jax_from_dofs = surface_daspect_ratio_jax_from_dofs

_surface_d2aspect_ratio_jax_from_dofs = surface_d2aspect_ratio_jax_from_dofs


def surface_principal_curvature_jax_from_dofs(
    spec,
    dofs,
    *,
    kappamax1=1,
    kappamax2=1,
    weight1=0.05,
    weight2=0.05,
):
    norm_normal, curvature = _surface_normal_norm_and_curvatures_jax_from_dofs(
        spec,
        dofs,
    )
    _mean_curvature, _gaussian_curvature, k1, k2 = jnp.split(
        curvature,
        [1, 2, 3],
        axis=2,
    )
    k1 = jnp.reshape(k1, curvature.shape[:2])
    k2 = jnp.reshape(k2, curvature.shape[:2])
    return jnp.sum(norm_normal * jnp.exp(-(k1 - kappamax1) / weight1)) + jnp.sum(
        norm_normal * jnp.exp(-(-k2 - kappamax2) / weight2)
    )


def _surface_dprincipal_curvature_jax_from_dofs(
    spec,
    dofs,
    *,
    kappamax1,
    kappamax2,
    weight1,
    weight2,
):
    dofs = _as_jax_float64(dofs)
    norm_normal = _surface_normal_norm_jax_from_dofs(spec, dofs)
    curvature = surface_curvatures_jax_from_dofs(spec, dofs)
    dnorm_normal = _surface_dnormal_norm_jax_from_dofs(spec, dofs)
    dcurvature = surface_dsurface_curvatures_jax_from_dofs(spec, dofs)
    _mean_curvature, _gaussian_curvature, k1, k2 = jnp.split(
        curvature,
        [1, 2, 3],
        axis=2,
    )
    _dmean_curvature, _dgaussian_curvature, dk1, dk2 = jnp.split(
        dcurvature,
        [1, 2, 3],
        axis=2,
    )
    k1 = jnp.reshape(k1, curvature.shape[:2])
    k2 = jnp.reshape(k2, curvature.shape[:2])
    dk1 = jnp.reshape(dk1, dcurvature.shape[:2] + dcurvature.shape[3:])
    dk2 = jnp.reshape(dk2, dcurvature.shape[:2] + dcurvature.shape[3:])
    exp1 = jnp.exp(-(k1 - kappamax1) / weight1)
    exp2 = jnp.exp((k2 + kappamax2) / weight2)
    dterm1 = (
        exp1[:, :, None] * dnorm_normal
        - (norm_normal * exp1 / weight1)[:, :, None] * dk1
    )
    dterm2 = (
        exp2[:, :, None] * dnorm_normal
        + (norm_normal * exp2 / weight2)[:, :, None] * dk2
    )
    return jnp.sum(dterm1 + dterm2, axis=(0, 1))


def surface_qfm_residual_jax_from_dofs(spec, dofs, coil_set_spec):
    return qfm_residual_jax_from_dofs(spec, dofs, coil_set_spec)


def surface_qfm_label_jax_from_dofs(
    spec,
    dofs,
    coil_set_spec,
    *,
    label: str,
    toroidal_flux_idx: int = 0,
):
    """Return the QFM constraint label using the JAX surface/field lane."""
    return qfm_label_jax_from_dofs(
        spec,
        dofs,
        coil_set_spec,
        label=label,
        toroidal_flux_idx=toroidal_flux_idx,
    )


def surface_qfm_penalty_jax_from_dofs(
    spec,
    dofs,
    coil_set_spec,
    *,
    label: str,
    targetlabel,
    constraint_weight=1.0,
    toroidal_flux_idx: int = 0,
):
    """Return ``QFM + 0.5 * weight * (label - target)^2`` in pure JAX."""
    return qfm_penalty_jax_from_dofs(
        spec,
        dofs,
        coil_set_spec,
        label=label,
        label_spec=spec,
        label_coil_set_spec=coil_set_spec,
        targetlabel=targetlabel,
        constraint_weight=constraint_weight,
        toroidal_flux_idx=toroidal_flux_idx,
    )


def surface_qfm_penalty_value_and_grad_jax_from_dofs(
    spec,
    dofs,
    coil_set_spec,
    *,
    label: str,
    targetlabel,
    constraint_weight=1.0,
    toroidal_flux_idx: int = 0,
):
    """Return QFM penalty value and gradient with respect to surface dofs."""
    return qfm_penalty_value_and_grad_jax_from_dofs(
        spec,
        dofs,
        coil_set_spec,
        label=label,
        label_spec=spec,
        label_coil_set_spec=coil_set_spec,
        targetlabel=targetlabel,
        constraint_weight=constraint_weight,
        toroidal_flux_idx=toroidal_flux_idx,
    )


def _surface_dqfm_residual_jax_from_dofs(spec, dofs, coil_set_spec):
    return _strict_scalar_grad(
        lambda x: surface_qfm_residual_jax_from_dofs(spec, x, coil_set_spec),
        _as_jax_float64(dofs),
    )


def _surface_objective_surface_view(surface, *, range, nphi, ntheta):
    if range is None and nphi is None and ntheta is None:
        return surface
    resolved_range = range
    if resolved_range is None:
        if surface.stellsym:
            resolved_range = Surface.RANGE_HALF_PERIOD
        else:
            resolved_range = Surface.RANGE_FIELD_PERIOD
    if nphi is None:
        nphi = len(surface.quadpoints_phi)
    if ntheta is None:
        ntheta = len(surface.quadpoints_theta)
    regrid_kwargs = {
        "nphi": nphi,
        "ntheta": ntheta,
        "range": resolved_range,
        "nfp": surface.nfp,
        "stellsym": surface.stellsym,
        "mpol": surface.mpol,
        "ntor": surface.ntor,
        "dofs": surface.dofs,
    }
    if type(surface).__name__ == "SurfaceXYZTensorFourier":
        regrid_kwargs["clamped_dims"] = list(surface.clamped_dims)
    return surface.__class__.from_nphi_ntheta(**regrid_kwargs)


class _SurfaceScalarMetricJAX(Optimizable):
    def __init__(self, surface, range=None, nphi=None, ntheta=None):
        self.surface = _surface_objective_surface_view(
            surface,
            range=range,
            nphi=nphi,
            ntheta=ntheta,
        )
        self.range = range
        self.nphi = nphi
        self.ntheta = ntheta
        super().__init__(depends_on=[self.surface])

    def _surface_spec_and_dofs(self):
        return _surface_spec_from_surface(self.surface), _as_jax_float64(
            self.surface.get_dofs()
        )

    def J(self):
        spec, dofs = self._surface_spec_and_dofs()
        return _host_scalar(self._value_fn(spec, dofs))

    @derivative_dec
    def dJ(self):
        return Derivative({self.surface: self.dJ_by_dsurfacecoefficients()})

    def dJ_by_dsurfacecoefficients(self):
        spec, dofs = self._surface_spec_and_dofs()
        return _host_array(self._gradient_fn(spec, dofs))

    def d2J_by_dsurfacecoefficientsdsurfacecoefficients(self):
        spec, dofs = self._surface_spec_and_dofs()
        return _host_array(self._hessian_fn(spec, dofs))


class AspectRatioJAX(_SurfaceScalarMetricJAX):
    """JAX-backed wrapper class for surface aspect ratio."""

    _value_fn = staticmethod(surface_aspect_ratio_jax_from_dofs)
    _gradient_fn = staticmethod(_surface_daspect_ratio_jax_from_dofs)
    _hessian_fn = staticmethod(_surface_d2aspect_ratio_jax_from_dofs)


class AreaJAX(_SurfaceScalarMetricJAX):
    """JAX-backed wrapper class for surface area."""

    _value_fn = staticmethod(surface_area_jax_from_dofs)
    _gradient_fn = staticmethod(surface_darea_jax_from_dofs)
    _hessian_fn = staticmethod(surface_d2area_jax_from_dofs)


class VolumeJAX(_SurfaceScalarMetricJAX):
    """JAX-backed wrapper class for enclosed surface volume."""

    _value_fn = staticmethod(surface_volume_jax_from_dofs)
    _gradient_fn = staticmethod(surface_dvolume_jax_from_dofs)
    _hessian_fn = staticmethod(surface_d2volume_jax_from_dofs)


class PrincipalCurvatureJAX(Optimizable):
    """JAX-backed wrapper for the upstream principal-curvature penalty."""

    def __init__(self, surface, kappamax1=1, kappamax2=1, weight1=0.05, weight2=0.05):
        self.surface = surface
        self.kappamax1 = kappamax1
        self.kappamax2 = kappamax2
        self.weight1 = weight1
        self.weight2 = weight2
        super().__init__(depends_on=[surface])

    def _surface_spec_and_dofs(self):
        return _surface_spec_from_surface(self.surface), _as_jax_float64(
            self.surface.get_dofs()
        )

    def _objective_kwargs(self):
        return dict(
            kappamax1=self.kappamax1,
            kappamax2=self.kappamax2,
            weight1=self.weight1,
            weight2=self.weight2,
        )

    def J(self):
        spec, dofs = self._surface_spec_and_dofs()
        return _host_scalar(
            surface_principal_curvature_jax_from_dofs(
                spec,
                dofs,
                **self._objective_kwargs(),
            )
        )

    @derivative_dec
    def dJ(self):
        return Derivative({self.surface: self.dJ_by_dsurfacecoefficients()})

    def dJ_by_dsurfacecoefficients(self):
        spec, dofs = self._surface_spec_and_dofs()
        return _host_array(
            _surface_dprincipal_curvature_jax_from_dofs(
                spec,
                dofs,
                **self._objective_kwargs(),
            )
        )


class QfmResidualJAX(Optimizable):
    """JAX-backed wrapper for fixed-surface QFM residuals."""

    def __init__(self, surface, biotsavart):
        self.surface = surface
        self.biotsavart = biotsavart
        super().__init__(depends_on=[surface, biotsavart])

    def recompute_bell(self, parent=None):
        self.invalidate_cache()

    def invalidate_cache(self):
        return None

    def _surface_spec_dofs_and_coil_spec(self):
        _current_coil_dofs, coil_set_spec = _current_coil_dofs_and_spec(self.biotsavart)
        return (
            _surface_spec_from_surface(self.surface),
            _as_jax_float64(self.surface.get_dofs()),
            coil_set_spec,
        )

    def J(self):
        spec, dofs, coil_set_spec = self._surface_spec_dofs_and_coil_spec()
        return _host_scalar(
            surface_qfm_residual_jax_from_dofs(
                spec,
                dofs,
                coil_set_spec,
            )
        )

    def dJ_by_dsurfacecoefficients(self):
        spec, dofs, coil_set_spec = self._surface_spec_dofs_and_coil_spec()
        return _host_array(
            _surface_dqfm_residual_jax_from_dofs(spec, dofs, coil_set_spec)
        )


def _explicit_index_array(indices):
    return runtime_device_put(indices, dtype=np.int32)


def _take_runtime_entries(array, indices):
    indices = np.asarray(indices, dtype=np.int32)
    if indices.size == 0:
        return _zeros(0, dtype=array.dtype)
    return jnp.take(array, _explicit_index_array(indices), axis=0)


def _take_runtime_scalar(array, index):
    return jnp.reshape(
        _take_runtime_entries(array, np.array([int(index)], dtype=np.int32)),
        (),
    )


def _take_runtime_row(array, index):
    return jnp.reshape(
        _take_runtime_entries(array, np.array([int(index)], dtype=np.int32)),
        array.shape[1:],
    )


def _split_x_inner_runtime(x_inner, optimize_G):
    length = int(x_inner.shape[0])
    sdof_count = length - (2 if optimize_G else 1)
    sdofs = _take_runtime_entries(x_inner, np.arange(sdof_count, dtype=np.int32))
    iota = _take_runtime_scalar(x_inner, sdof_count)
    if optimize_G:
        return sdofs, iota, _take_runtime_scalar(x_inner, sdof_count + 1)
    return sdofs, iota, None


def _runtime_float64_scalar(value, *, reference):
    return _as_runtime_float64(value, reference=reference)


def _traceable_adjoint_fail_gradient_like(gradient):
    return jnp.full_like(gradient, _TRACEABLE_ADJOINT_FAIL_GRAD_SENTINEL)


def _traceable_rejected_objective_value(value, reference_value):
    reference = lax.stop_gradient(
        _as_runtime_float64(reference_value, reference=reference_value)
    )
    candidate = lax.stop_gradient(_as_runtime_float64(value, reference=reference))
    finite_candidate = jnp.where(jnp.isfinite(candidate), candidate, reference)
    penalty = jnp.maximum(
        jnp.abs(reference),
        _runtime_float64_scalar(1.0, reference=reference),
    )
    return jnp.maximum(finite_candidate, reference + penalty) + penalty


def _runtime_float64_array(value, *, reference):
    return _as_runtime_float64(value, reference=reference)


def _runtime_bool(value):
    return _traceable_runtime_deviceify_tree(np.asarray(bool(value), dtype=bool))


def _runtime_int32_scalar(value):
    return runtime_device_put(value, dtype=np.int32)


def _runtime_zeros_like(value):
    zero = _runtime_float64_scalar(0.0, reference=value)
    return jnp.broadcast_to(zero, value.shape)


def _static_upper_triangle_pair_indices(size: int) -> tuple[np.ndarray, np.ndarray]:
    left, right = np.triu_indices(int(size), k=1)
    return left.astype(np.int32), right.astype(np.int32)


def _static_cross_pair_indices(
    left_size: int,
    right_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    left = np.repeat(np.arange(int(left_size), dtype=np.int32), int(right_size))
    right = np.tile(np.arange(int(right_size), dtype=np.int32), int(left_size))
    return left, right


def _take_static_rows(array, indices: np.ndarray):
    return jnp.take(array, jnp.asarray(indices, dtype=jnp.int32), axis=0)


def _curve_geometry_stacks_from_grouped_spec(coil_set_spec):
    stacks = []
    for group in coil_set_spec.groups:
        gammas = _as_jax_float64(group.gammas)
        gammadashs = _as_jax_float64(group.gammadashs)
        stacks.append(
            (
                gammas.reshape((int(gammas.shape[0]), -1, 3)),
                gammadashs.reshape((int(gammadashs.shape[0]), -1, 3)),
            )
        )
    return tuple(stacks)


def _curve_stacks_from_grouped_spec(coil_set_spec):
    return tuple(
        gammas
        for gammas, _gammadashs in _curve_geometry_stacks_from_grouped_spec(
            coil_set_spec
        )
    )


def _curve_stacks_from_curve_tuple(coil_gammas):
    grouped: dict[tuple[tuple[int, ...], type], list[jax.Array]] = {}
    for gamma in coil_gammas:
        gamma_arr = _as_jax_float64(gamma).reshape((-1, 3))
        shape = tuple(int(dim) for dim in gamma_arr.shape)
        # Do not stack traced primals with closed-over device arrays: JAX treats
        # that as staging a constant and transfer_guard=disallow rejects it.
        grouped.setdefault((shape, type(gamma_arr)), []).append(gamma_arr)
    return tuple(jnp.stack(gammas, axis=0) for gammas in grouped.values())


def _curve_curve_point_pair_batches_from_stacks(curve_stacks):
    batches = []
    for stack_index, curve_stack in enumerate(curve_stacks):
        curve_count = int(curve_stack.shape[0])
        left_indices, right_indices = _static_upper_triangle_pair_indices(curve_count)
        if left_indices.size > 0:
            batches.append(
                (
                    _take_static_rows(curve_stack, left_indices),
                    _take_static_rows(curve_stack, right_indices),
                )
            )
        for previous_stack in curve_stacks[:stack_index]:
            left_indices, right_indices = _static_cross_pair_indices(
                curve_count,
                int(previous_stack.shape[0]),
            )
            if left_indices.size > 0:
                batches.append(
                    (
                        _take_static_rows(curve_stack, left_indices),
                        _take_static_rows(previous_stack, right_indices),
                    )
                )
    return tuple(batches)


def _curve_surface_point_pair_batches_from_stacks(curve_stacks, surface_gamma):
    flat_surface = _as_jax_float64(surface_gamma).reshape((-1, 3))
    batches = []
    for curve_stack in curve_stacks:
        curve_count = int(curve_stack.shape[0])
        if curve_count > 0:
            batches.append((curve_stack, flat_surface))
    return tuple(batches)


def _curve_curve_distance_batch(
    gammas_i,
    gammadashs_i,
    gammas_j,
    gammadashs_j,
    minimum_distance,
):
    if int(gammas_i.shape[0]) == 0:
        return _runtime_float64_scalar(0.0, reference=minimum_distance)
    distances = jax.vmap(
        lambda gamma_i, gammadash_i, gamma_j, gammadash_j: cc_distance_pure(
            gamma_i,
            gammadash_i,
            gamma_j,
            gammadash_j,
            minimum_distance,
        )
    )(gammas_i, gammadashs_i, gammas_j, gammadashs_j)
    return jnp.sum(distances)


@partial(jax.jit, static_argnames=("chunk_size_token",))
def _curve_surface_distance_batch_compiled(
    gammas,
    gammadashs,
    surface_gamma,
    surface_normal,
    minimum_distance,
    *,
    chunk_size_token,
):
    del chunk_size_token
    flat_gammas = gammas.reshape((-1, gammas.shape[-1]))
    flat_gammadashs = gammadashs.reshape((-1, gammadashs.shape[-1]))
    curve_count = int(gammas.shape[0])
    # cs_distance_pure normalizes by (#curve_pts * #surface_pts); flattening N
    # curves divides by (N*R*C) so we multiply by N to recover sum_i sum_{q,c}.
    return _runtime_float64_scalar(curve_count, reference=minimum_distance) * (
        cs_distance_pure(
            flat_gammas,
            flat_gammadashs,
            surface_gamma,
            surface_normal,
            minimum_distance,
        )
    )


def _curve_surface_distance_batch(
    gammas,
    gammadashs,
    surface_gamma,
    surface_normal,
    minimum_distance,
):
    curve_count = int(gammas.shape[0])
    if curve_count == 0:
        return _runtime_float64_scalar(0.0, reference=minimum_distance)
    return _curve_surface_distance_batch_compiled(
        gammas,
        gammadashs,
        surface_gamma,
        surface_normal,
        minimum_distance,
        chunk_size_token=_resolve_pairwise_penalty_chunk_size(),
    )


def _curve_curve_penalty_from_grouped_spec(coil_set_spec, minimum_distance):
    total = _runtime_float64_scalar(0.0, reference=minimum_distance)
    curve_stacks = _curve_geometry_stacks_from_grouped_spec(coil_set_spec)
    for stack_index, (gammas, gammadashs) in enumerate(curve_stacks):
        curve_count = int(gammas.shape[0])
        left_indices, right_indices = _static_upper_triangle_pair_indices(curve_count)
        if left_indices.size > 0:
            total = total + _curve_curve_distance_batch(
                _take_static_rows(gammas, left_indices),
                _take_static_rows(gammadashs, left_indices),
                _take_static_rows(gammas, right_indices),
                _take_static_rows(gammadashs, right_indices),
                minimum_distance,
            )
        for previous_gammas, previous_gammadashs in curve_stacks[:stack_index]:
            left_indices, right_indices = _static_cross_pair_indices(
                curve_count,
                int(previous_gammas.shape[0]),
            )
            if left_indices.size > 0:
                total = total + _curve_curve_distance_batch(
                    _take_static_rows(gammas, left_indices),
                    _take_static_rows(gammadashs, left_indices),
                    _take_static_rows(previous_gammas, right_indices),
                    _take_static_rows(previous_gammadashs, right_indices),
                    minimum_distance,
                )
    return total


def _curve_surface_penalty_from_grouped_spec(
    coil_set_spec,
    surface_gamma,
    surface_normal,
    minimum_distance,
):
    total = _runtime_float64_scalar(0.0, reference=minimum_distance)
    surface_gamma = surface_gamma.reshape((-1, 3))
    surface_normal = surface_normal.reshape((-1, 3))
    for gammas, gammadashs in _curve_geometry_stacks_from_grouped_spec(coil_set_spec):
        total = total + _curve_surface_distance_batch(
            gammas,
            gammadashs,
            surface_gamma,
            surface_normal,
            minimum_distance,
        )
    return total


def _banana_curve_penalties_from_coil_dofs(
    coil_dofs,
    coil_dof_extraction_spec,
    *,
    banana_curve_index,
    length_target,
    curvature_threshold,
    curvature_p_norm,
):
    coil_specs = coil_specs_from_dof_extraction_spec(
        coil_dof_extraction_spec, coil_dofs
    )
    banana_curve_spec = coil_specs[int(banana_curve_index)].curve
    _gamma, banana_gammadash, banana_gammadashdash = curve_geometry_from_spec(
        banana_curve_spec
    )
    banana_curve_length = curve_length_pure(
        incremental_arclength_pure(banana_gammadash)
    )
    zero = _runtime_float64_scalar(0.0, reference=banana_curve_length)
    half = _runtime_float64_scalar(0.5, reference=banana_curve_length)
    length_target_jax = _runtime_float64_scalar(
        length_target, reference=banana_curve_length
    )
    curvature_threshold_jax = _runtime_float64_scalar(
        curvature_threshold,
        reference=banana_curve_length,
    )
    curvature_p_norm_jax = _runtime_float64_scalar(
        curvature_p_norm,
        reference=banana_curve_length,
    )
    length_delta = jnp.maximum(banana_curve_length - length_target_jax, zero)
    length_penalty = half * (length_delta * length_delta)
    curvature_penalty = Lp_curvature_pure(
        kappa_pure(banana_gammadash, banana_gammadashdash),
        banana_gammadash,
        curvature_p_norm_jax,
        curvature_threshold_jax,
    )
    return length_penalty, curvature_penalty


def _traceable_single_stage_outer_term_values(
    x_inner,
    coil_dofs,
    coil_set_spec,
    *,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    surface_kind,
    label_quadpoints_phi,
    label_quadpoints_theta,
    label_mpol,
    label_ntor,
    label_nfp,
    label_stellsym,
    label_scatter_indices,
    label_surface_kind,
    optimize_G,
    weight_inv_modB,
    constraint_weight,
    targetlabel,
    label_type,
    phi_idx,
    iota_target,
    surface_quadpoints_phi,
    surface_quadpoints_theta,
    coil_dof_extraction_spec,
    outer_objective_config,
):
    """Return the raw single-stage outer-objective term values at one state."""
    J_boozer = _boozer_residual_J_of_x_inner(
        x_inner,
        coil_set_spec=coil_set_spec,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
        scatter_indices=scatter_indices,
        surface_kind=surface_kind,
        label_quadpoints_phi=label_quadpoints_phi,
        label_quadpoints_theta=label_quadpoints_theta,
        label_mpol=label_mpol,
        label_ntor=label_ntor,
        label_nfp=label_nfp,
        label_stellsym=label_stellsym,
        label_scatter_indices=label_scatter_indices,
        label_surface_kind=label_surface_kind,
        optimize_G=optimize_G,
        weight_inv_modB=weight_inv_modB,
        constraint_weight=constraint_weight,
        targetlabel=targetlabel,
        label_type=label_type,
        phi_idx=phi_idx,
    )
    iota_penalty = _traceable_iota_target_penalty(
        x_inner,
        optimize_G=optimize_G,
        iota_target=iota_target,
    )
    sdofs, _iota, _G = _split_x_inner_runtime(x_inner, optimize_G)
    surface_gamma, xphi, xtheta = _surface_geometry_from_dofs(
        sdofs,
        surface_quadpoints_phi,
        surface_quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
        surface_kind=surface_kind,
    )
    surface_normal = jnp.cross(xphi, xtheta)
    non_qs_penalty = _qs_ratio_pure(
        sdofs,
        coil_set_spec,
        quadpoints_phi=_runtime_float64_array(
            outer_objective_config["non_qs_quadpoints_phi"],
            reference=sdofs,
        ),
        quadpoints_theta=_runtime_float64_array(
            outer_objective_config["non_qs_quadpoints_theta"],
            reference=sdofs,
        ),
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
        scatter_indices=scatter_indices,
        surface_kind=surface_kind,
        axis=int(outer_objective_config["non_qs_axis"]),
    )

    length_penalty, curvature_penalty = _banana_curve_penalties_from_coil_dofs(
        coil_dofs,
        coil_dof_extraction_spec,
        banana_curve_index=int(outer_objective_config["banana_curve_index"]),
        length_target=outer_objective_config["length_target"],
        curvature_threshold=outer_objective_config["curvature_threshold"],
        curvature_p_norm=outer_objective_config["curvature_p_norm"],
    )

    curve_curve_penalty = _curve_curve_penalty_from_grouped_spec(
        coil_set_spec,
        _runtime_float64_scalar(
            outer_objective_config["curve_curve_threshold"],
            reference=surface_gamma,
        ),
    )

    curve_surface_penalty = _curve_surface_penalty_from_grouped_spec(
        coil_set_spec,
        surface_gamma,
        surface_normal,
        _runtime_float64_scalar(
            outer_objective_config["curve_surface_threshold"],
            reference=surface_gamma,
        ),
    )

    vessel_gamma = _runtime_float64_array(
        outer_objective_config["vessel_gamma"],
        reference=surface_gamma,
    ).reshape((-1, 3))
    surface_vessel_penalty = surface_to_surface_distance_pure(
        surface_gamma,
        vessel_gamma,
        _runtime_float64_scalar(
            outer_objective_config["surface_vessel_threshold"],
            reference=surface_gamma,
        ),
    )
    return {
        "non_qs": non_qs_penalty,
        "residual": J_boozer,
        "iota": iota_penalty,
        "length": length_penalty,
        "curvature": curvature_penalty,
        "curve_curve": curve_curve_penalty,
        "curve_surface": curve_surface_penalty,
        "surface_vessel": surface_vessel_penalty,
    }


def _traceable_weighted_single_stage_outer_term_values(
    term_values,
    *,
    outer_objective_config,
):
    """Apply configured weights to raw single-stage outer-objective terms."""
    weighted_terms = {}
    for term_name, weight_key in _TRACEABLE_SINGLE_STAGE_OUTER_TERM_SPECS:
        term_value = term_values[term_name]
        weight = outer_objective_config.get(weight_key, 0.0)
        if not _traceable_single_stage_weight_is_active(weight):
            weighted_terms[term_name] = _runtime_float64_scalar(
                0.0,
                reference=term_value,
            )
            continue
        weighted_terms[term_name] = (
            _runtime_float64_scalar(weight, reference=term_value) * term_value
        )
    return weighted_terms


def _traceable_smoothmax_selected(values, temperature):
    values = _as_jax_float64(values).reshape((-1,))
    if int(values.shape[0]) == 0:
        return _runtime_float64_scalar(-np.inf, reference=values)
    bounded_temperature = jnp.maximum(
        _runtime_float64_scalar(temperature, reference=values),
        _runtime_float64_scalar(np.finfo(np.float64).eps, reference=values),
    )
    hard_max = jnp.max(values)
    logits = (values - hard_max) / bounded_temperature
    selection_mask = values >= (
        hard_max - _runtime_float64_scalar(4.0, reference=values) * bounded_temperature
    )
    masked_logits = jnp.where(selection_mask, logits, -jnp.inf)
    return hard_max + bounded_temperature * jax.nn.logsumexp(masked_logits)


def _traceable_single_stage_banana_curve_runtime_metrics(
    coil_dofs,
    coil_dof_extraction_spec,
    *,
    banana_curve_index,
    curvature_smoothing,
):
    coil_specs = coil_specs_from_dof_extraction_spec(
        coil_dof_extraction_spec, coil_dofs
    )
    banana_curve_spec = coil_specs[int(banana_curve_index)].curve
    _gamma, banana_gammadash, banana_gammadashdash = curve_geometry_from_spec(
        banana_curve_spec
    )
    coil_length = curve_length_pure(incremental_arclength_pure(banana_gammadash))
    max_curvature = _traceable_smoothmax_selected(
        kappa_pure(banana_gammadash, banana_gammadashdash),
        temperature=curvature_smoothing,
    )
    banana_current = jnp.abs(
        _take_runtime_scalar(coil_specs[int(banana_curve_index)].current.value, 0)
    )
    return coil_length, max_curvature, banana_current


def _traceable_single_stage_coil_gammas(
    coil_dofs,
    coil_dof_extraction_spec,
):
    coil_specs = coil_specs_from_dof_extraction_spec(
        coil_dof_extraction_spec, coil_dofs
    )
    coil_gammas = []
    for coil_spec in coil_specs:
        gamma, _gammadash, _gammadashdash = curve_geometry_from_spec(coil_spec.curve)
        if coil_spec.symmetry.has_rotation:
            gamma = gamma @ coil_spec.symmetry.rotmat
        coil_gammas.append(gamma.reshape((-1, 3)))
    return tuple(coil_gammas)


def _traceable_single_stage_curve_curve_signed_constraint(
    coil_gammas,
    *,
    minimum_distance,
    distance_smoothing,
):
    curve_stacks = _curve_stacks_from_curve_tuple(coil_gammas)
    point_pair_batches = _curve_curve_point_pair_batches_from_stacks(curve_stacks)
    if len(point_pair_batches) == 0:
        return _runtime_float64_scalar(minimum_distance, reference=minimum_distance)
    smooth_min = pairwise_selected_smoothmin_distance_batched_pure(
        point_pair_batches,
        temperature=distance_smoothing,
    )
    return _runtime_float64_scalar(minimum_distance, reference=smooth_min) - smooth_min


def _traceable_single_stage_curve_surface_signed_constraint(
    coil_gammas,
    surface_gamma,
    *,
    minimum_distance,
    distance_smoothing,
):
    curve_stacks = _curve_stacks_from_curve_tuple(coil_gammas)
    point_pair_batches = _curve_surface_point_pair_batches_from_stacks(
        curve_stacks,
        surface_gamma,
    )
    if len(point_pair_batches) == 0:
        return _runtime_float64_scalar(minimum_distance, reference=surface_gamma)
    smooth_min = pairwise_selected_smoothmin_distance_batched_pure(
        point_pair_batches,
        temperature=distance_smoothing,
    )
    return _runtime_float64_scalar(minimum_distance, reference=smooth_min) - smooth_min


def _traceable_single_stage_surface_surface_signed_constraint(
    surface_gamma,
    vessel_gamma,
    *,
    minimum_distance,
    distance_smoothing,
):
    flat_surface = surface_gamma.reshape((-1, 3))
    flat_vessel = _runtime_float64_array(vessel_gamma, reference=surface_gamma).reshape(
        (-1, 3)
    )
    smooth_min = pairwise_selected_smoothmin_distance_pure(
        ((flat_surface, flat_vessel),),
        temperature=distance_smoothing,
    )
    return _runtime_float64_scalar(minimum_distance, reference=smooth_min) - smooth_min


def _traceable_single_stage_hardware_constraint_values(
    x_inner,
    coil_dofs,
    *,
    objective_kwargs,
    alm_config,
):
    outer_objective_config = objective_kwargs["outer_objective_config"]
    if outer_objective_config is None:
        raise RuntimeError(
            "Traceable single-stage ALM runtime requires outer_objective_config."
        )
    optimize_G = bool(objective_kwargs["optimize_G"])
    sdofs, _iota, _G = _split_x_inner_runtime(x_inner, optimize_G)
    surface_gamma, _xphi, _xtheta = _surface_geometry_from_dofs(
        sdofs,
        objective_kwargs["surface_quadpoints_phi"],
        objective_kwargs["surface_quadpoints_theta"],
        objective_kwargs["mpol"],
        objective_kwargs["ntor"],
        objective_kwargs["nfp"],
        objective_kwargs["stellsym"],
        objective_kwargs["scatter_indices"],
        surface_kind=objective_kwargs["surface_kind"],
    )
    coil_gammas = _traceable_single_stage_coil_gammas(
        coil_dofs,
        objective_kwargs["coil_dof_extraction_spec"],
    )
    coil_length, max_curvature, banana_current = (
        _traceable_single_stage_banana_curve_runtime_metrics(
            coil_dofs,
            objective_kwargs["coil_dof_extraction_spec"],
            banana_curve_index=int(outer_objective_config["banana_curve_index"]),
            curvature_smoothing=alm_config["curvature_smoothing"],
        )
    )
    return {
        "coil_coil_spacing": _traceable_single_stage_curve_curve_signed_constraint(
            coil_gammas,
            minimum_distance=outer_objective_config["curve_curve_threshold"],
            distance_smoothing=alm_config["distance_smoothing"],
        ),
        "coil_surface_spacing": _traceable_single_stage_curve_surface_signed_constraint(
            coil_gammas,
            surface_gamma.reshape((-1, 3)),
            minimum_distance=outer_objective_config["curve_surface_threshold"],
            distance_smoothing=alm_config["distance_smoothing"],
        ),
        "surface_vessel_spacing": _traceable_single_stage_surface_surface_signed_constraint(
            surface_gamma,
            outer_objective_config["vessel_gamma"],
            minimum_distance=outer_objective_config["surface_vessel_threshold"],
            distance_smoothing=alm_config["distance_smoothing"],
        ),
        "max_curvature": max_curvature
        - _runtime_float64_scalar(
            outer_objective_config["curvature_threshold"],
            reference=max_curvature,
        ),
        "coil_length_upper_bound": coil_length
        - _runtime_float64_scalar(
            outer_objective_config["length_target"],
            reference=coil_length,
        ),
        "banana_current_upper_bound": banana_current
        - _runtime_float64_scalar(
            alm_config["banana_current_threshold"],
            reference=banana_current,
        ),
    }


def _traceable_single_stage_alm_constraint_values(
    raw_terms,
    x_inner,
    coil_dofs,
    *,
    objective_kwargs,
    alm_config,
):
    outer_objective_config = objective_kwargs["outer_objective_config"]
    if outer_objective_config is None:
        raise RuntimeError(
            "Traceable single-stage ALM runtime requires outer_objective_config."
        )
    hardware_constraints = _traceable_single_stage_hardware_constraint_values(
        x_inner,
        coil_dofs,
        objective_kwargs=objective_kwargs,
        alm_config=alm_config,
    )
    named_constraints = dict(hardware_constraints)
    if alm_config["alm_formulation"] == "thresholded_physics":
        named_constraints.update(
            {
                "qs_error": raw_terms["non_qs"]
                - _runtime_float64_scalar(
                    alm_config["qs_threshold"],
                    reference=raw_terms["non_qs"],
                ),
                "boozer_residual": raw_terms["residual"]
                - _runtime_float64_scalar(
                    alm_config["boozer_threshold"],
                    reference=raw_terms["residual"],
                ),
                "iota_penalty": raw_terms["iota"]
                - _runtime_float64_scalar(
                    alm_config["iota_penalty_threshold"],
                    reference=raw_terms["iota"],
                ),
                "length_penalty": raw_terms["length"]
                - _runtime_float64_scalar(
                    alm_config["length_penalty_threshold"],
                    reference=raw_terms["length"],
                ),
            }
        )
    return jnp.stack(
        [
            named_constraints[constraint_name]
            for constraint_name in alm_config["constraint_names"]
        ]
    )


def _traceable_single_stage_alm_physics_total(raw_terms, *, outer_objective_config):
    return (
        raw_terms["non_qs"]
        + _runtime_float64_scalar(
            outer_objective_config["residual_weight"],
            reference=raw_terms["residual"],
        )
        * raw_terms["residual"]
        + _runtime_float64_scalar(
            outer_objective_config["iota_weight"],
            reference=raw_terms["iota"],
        )
        * raw_terms["iota"]
        + _runtime_float64_scalar(
            outer_objective_config["length_weight"],
            reference=raw_terms["length"],
        )
        * raw_terms["length"]
    )


def _traceable_single_stage_alm_base_total(
    raw_terms,
    *,
    outer_objective_config,
    alm_formulation,
):
    physics_total = _traceable_single_stage_alm_physics_total(
        raw_terms,
        outer_objective_config=outer_objective_config,
    )
    if alm_formulation == "weighted_sum":
        return physics_total, physics_total
    if alm_formulation == "thresholded_physics":
        return _runtime_float64_scalar(0.0, reference=physics_total), physics_total
    raise ValueError(f"Unsupported ALM formulation {alm_formulation!r}.")


def _traceable_augmented_inequality_total(
    base_total,
    constraint_values,
    multipliers,
    penalty,
):
    constraint_values = _as_jax_float64(constraint_values).reshape((-1,))
    multipliers = _runtime_float64_array(
        multipliers, reference=constraint_values
    ).reshape(constraint_values.shape)
    penalty_jax = _runtime_float64_scalar(penalty, reference=constraint_values)
    positive_shift = jnp.maximum(
        _runtime_float64_scalar(0.0, reference=constraint_values),
        multipliers + penalty_jax * constraint_values,
    )
    return base_total + (
        _runtime_float64_scalar(0.5, reference=constraint_values) / penalty_jax
    ) * (
        jnp.dot(positive_shift, positive_shift, precision=lax.Precision.HIGHEST)
        - jnp.dot(multipliers, multipliers, precision=lax.Precision.HIGHEST)
    )


def _traceable_single_stage_alm_evaluation(
    x_inner,
    coil_dofs,
    coil_set_spec,
    *,
    objective_kwargs,
    alm_config,
    multipliers,
    penalty,
):
    del coil_set_spec
    outer_objective_config = objective_kwargs["outer_objective_config"]
    if outer_objective_config is None:
        raise RuntimeError(
            "Traceable single-stage ALM runtime requires outer_objective_config."
        )
    raw_terms = _traceable_single_stage_outer_term_values(
        x_inner,
        coil_dofs,
        coil_set_spec_from_dof_extraction_spec(
            objective_kwargs["coil_dof_extraction_spec"],
            coil_dofs,
        ),
        **_traceable_total_objective_kwargs(objective_kwargs),
    )
    objective_total, physics_total = _traceable_single_stage_alm_base_total(
        raw_terms,
        outer_objective_config=outer_objective_config,
        alm_formulation=alm_config["alm_formulation"],
    )
    constraint_values = _traceable_single_stage_alm_constraint_values(
        raw_terms,
        x_inner,
        coil_dofs,
        objective_kwargs=objective_kwargs,
        alm_config=alm_config,
    )
    feasibility_values = jnp.maximum(
        constraint_values,
        _runtime_float64_scalar(0.0, reference=constraint_values),
    )
    return {
        "total": _traceable_augmented_inequality_total(
            objective_total,
            constraint_values,
            multipliers,
            penalty,
        ),
        "base_total": physics_total,
        "physics_total": physics_total,
        "constraint_values": constraint_values,
        "feasibility_values": feasibility_values,
    }


def _evaluate_traceable_weighted_single_stage_outer_term(
    term_name,
    x_inner,
    coil_dofs,
    coil_set_spec,
    objective_kwargs,
):
    """Evaluate one weighted single-stage outer-objective term."""
    outer_objective_config = objective_kwargs["outer_objective_config"]
    if outer_objective_config is None:
        raise RuntimeError(
            "Weighted single-stage term diagnostics require outer_objective_config."
        )
    term_values = _traceable_single_stage_outer_term_values(
        x_inner,
        coil_dofs,
        coil_set_spec,
        **_traceable_total_objective_kwargs(objective_kwargs),
    )
    return _traceable_weighted_single_stage_outer_term_values(
        term_values,
        outer_objective_config=outer_objective_config,
    )[term_name]


def _traceable_full_single_stage_outer_objective(
    x_inner,
    coil_dofs,
    coil_set_spec,
    *,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    surface_kind,
    label_quadpoints_phi,
    label_quadpoints_theta,
    label_mpol,
    label_ntor,
    label_nfp,
    label_stellsym,
    label_scatter_indices,
    label_surface_kind,
    optimize_G,
    weight_inv_modB,
    constraint_weight,
    targetlabel,
    label_type,
    phi_idx,
    iota_target,
    surface_quadpoints_phi,
    surface_quadpoints_theta,
    coil_dof_extraction_spec,
    outer_objective_config,
):
    raw_terms = _traceable_single_stage_outer_term_values(
        x_inner,
        coil_dofs,
        coil_set_spec,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
        scatter_indices=scatter_indices,
        surface_kind=surface_kind,
        label_quadpoints_phi=label_quadpoints_phi,
        label_quadpoints_theta=label_quadpoints_theta,
        label_mpol=label_mpol,
        label_ntor=label_ntor,
        label_nfp=label_nfp,
        label_stellsym=label_stellsym,
        label_scatter_indices=label_scatter_indices,
        label_surface_kind=label_surface_kind,
        optimize_G=optimize_G,
        weight_inv_modB=weight_inv_modB,
        constraint_weight=constraint_weight,
        targetlabel=targetlabel,
        label_type=label_type,
        phi_idx=phi_idx,
        iota_target=iota_target,
        surface_quadpoints_phi=surface_quadpoints_phi,
        surface_quadpoints_theta=surface_quadpoints_theta,
        coil_dof_extraction_spec=coil_dof_extraction_spec,
        outer_objective_config=outer_objective_config,
    )
    weighted_terms = _traceable_weighted_single_stage_outer_term_values(
        raw_terms,
        outer_objective_config=outer_objective_config,
    )
    total = _runtime_float64_scalar(0.0, reference=next(iter(weighted_terms.values())))
    for term_name, _weight_key in _TRACEABLE_SINGLE_STAGE_OUTER_TERM_SPECS:
        total = total + weighted_terms[term_name]
    return total


def _canonicalize_traceable_exact_quadrature(booz_jax):
    """Return exact-compatible quadrature for the traceable scalar objective.

    VMEC half-period integration grids are half-cell shifted for spectral
    quadrature. BoozerExact's stellsym mask accepts unshifted exact grids, so
    the traceable scalar objective canonicalizes that known integration family
    before building its fixed residual mask.
    """
    quadpoints_phi = np.asarray(booz_jax.quadpoints_phi, dtype=float)
    quadpoints_theta = np.asarray(booz_jax.quadpoints_theta, dtype=float)

    mpol = int(booz_jax.mpol)
    ntor = int(booz_jax.ntor)
    nfp = float(booz_jax.nfp)
    if booz_jax.stellsym and quadpoints_phi.size > 1:
        shifted_half_period_phi = np.asarray(
            Surface.get_phi_quadpoints(
                nphi=quadpoints_phi.size,
                range=Surface.RANGE_HALF_PERIOD,
                nfp=booz_jax.nfp,
            ),
            dtype=float,
        )
        if np.allclose(quadpoints_phi, shifted_half_period_phi):
            quadpoints_phi = np.linspace(
                0.0,
                0.5 / nfp,
                ntor + 1,
                endpoint=False,
            )
            quadpoints_theta = np.linspace(
                0.0,
                1.0,
                2 * mpol + 1,
                endpoint=False,
            )

    mask_indices = _compute_stellsym_mask_indices_for_grid(
        mpol=mpol,
        ntor=ntor,
        nfp=booz_jax.nfp,
        stellsym=booz_jax.stellsym,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
    )
    return (
        _as_jax_float64(quadpoints_phi),
        _as_jax_float64(quadpoints_theta),
        mask_indices,
    )


def _solve_boozer_adjoint(adjoint_state, rhs):
    """Solve the transposed inner linearization for one adjoint runtime state.

    The exact-adjoint runtime uses the operator-backed solve callbacks, whose
    square-system path performs a residual refinement pass by default. Dense PLU
    linearizations can be ill-conditioned enough that CPU LAPACK and JAX/XLA
    triangular solves are not a direct vector-parity contract; parity checks
    should compare residual success and objective behavior, not byte-identical
    CPU/JAX adjoint vectors.
    """
    return _checked_boozer_linear_solve(adjoint_state, rhs, transpose=True)


def _solve_boozer_forward(adjoint_state, rhs):
    """Solve the forward inner linearization for one adjoint runtime state."""
    return _checked_boozer_linear_solve(adjoint_state, rhs, transpose=False)


def _checked_boozer_linear_solve(adjoint_state, rhs, *, transpose):
    direction = "transpose" if transpose else "forward"
    solve_with_status = getattr(
        adjoint_state,
        f"solve_{direction}_with_status",
        None,
    )
    if not callable(solve_with_status):
        raise RuntimeError(
            "Boozer adjoint state exposes no "
            f"solve_{direction}_with_status; "
            "cannot solve the inner linearization."
        )
    solution, status = solve_with_status(rhs)
    if not _host_bool(_optimizer_jax._linear_solve_status_success(status)):
        raise RuntimeError(
            "Boozer adjoint linear solve failed on the JAX runtime-state path "
            f"({adjoint_state.linearization_kind})."
        )
    return solution


def _solve_boozer_adjoint_batch(adjoint_state, rhs_batch):
    """Solve RHS rows as one column-batched runtime-state linear solve."""
    rhs_batch = jnp.asarray(rhs_batch)
    if rhs_batch.ndim != 2:
        raise ValueError(
            "_solve_boozer_adjoint_batch expects a rank-2 array with shape "
            "(num_rhs, decision_size)."
        )
    solved_columns = _checked_boozer_linear_solve(
        adjoint_state,
        rhs_batch.T,
        transpose=True,
    )
    return solved_columns.T


def _adjoint_state_decision_size(adjoint_state):
    return int(adjoint_state.decision_size)


def _adjoint_state_dtype(adjoint_state):
    return adjoint_state.dtype


def _iter_adjoint_coil_cotangents(stream_group_vjps, adjoint):
    """Yield grouped coil cotangents from the streaming adjoint callback."""
    if stream_group_vjps is None:
        raise RuntimeError(_MISSING_STREAMING_GROUP_VJP_ERROR)
    yield from stream_group_vjps(adjoint)


def _adjoint_coil_dofs_gradient(stream_group_vjps, adjoint, biotsavart, coil_dofs):
    """Project streamed adjoint cotangents to flat BiotSavart free-DOF order."""
    coil_dofs = _as_jax_float64(coil_dofs)
    total_gradient = coil_dofs - coil_dofs
    for d_coil_array, coil_group_indices in _iter_adjoint_coil_cotangents(
        stream_group_vjps, adjoint
    ):
        total_gradient = total_gradient + biotsavart.coil_cotangents_to_dofs_gradient(
            [d_coil_array],
            [coil_group_indices],
            coil_dofs=coil_dofs,
        )
    return total_gradient


def coil_dofs_gradient_to_derivative(biotsavart, coil_dofs_gradient):
    """Convert a flat free-DOF gradient into the public ``Derivative`` contract."""
    coil_dofs_gradient = _host_array(coil_dofs_gradient, dtype=np.float64)
    deriv_data = {}
    start = 0
    for lineage_opt in biotsavart.unique_dof_lineage:
        width = lineage_opt.local_dof_size
        if width == 0:
            continue

        block = np.zeros(lineage_opt.local_full_dof_size)
        stop = start + width
        block[lineage_opt.local_dofs_free_status] = coil_dofs_gradient[start:stop]
        start = stop

        dep_opts = tuple(lineage_opt.dofs.dep_opts())
        block_share = block / len(dep_opts)
        for dep_opt in dep_opts:
            if dep_opt in deriv_data:
                deriv_data[dep_opt] = deriv_data[dep_opt] + block_share
            else:
                deriv_data[dep_opt] = block_share.copy()

    return Derivative(deriv_data)


def _project_native_dJ_by_dcoil_dofs(surface_objective):
    return coil_dofs_gradient_to_derivative(
        surface_objective.biotsavart,
        surface_objective._dJ_by_dcoil_dofs,
    )


def _public_scalar_value(value):
    return float(_host_scalar(value, dtype=np.float64))


def _public_dJ_from_native_cache(surface_objective):
    if surface_objective._dJ is None:
        if surface_objective._dJ_by_dcoil_dofs is None:
            surface_objective.compute(compute_gradient=True)
        else:
            surface_objective._dJ = _project_native_dJ_by_dcoil_dofs(surface_objective)
    return surface_objective._dJ


def _make_cached_strict_scalar_value_and_grad(fun):
    """Compile the strict direct-objective value/grad callable once per owner."""

    def value_and_grad(arg, *args):
        return _strict_scalar_value_and_grad(fun, arg, *args)

    compiled_value_and_grad = jax.jit(value_and_grad, static_argnums=(2, 3))
    compiled_value_and_grad._simsopt_value_and_grad = True
    return compiled_value_and_grad


def _traceable_cache_leaf_signature(leaf):
    """Build a deterministic cache signature for one traceable-runtime leaf."""
    if isinstance(leaf, (jax.Array, np.ndarray)):
        array = _host_array(leaf)
        return (
            "array",
            str(array.dtype),
            tuple(array.shape),
            hashlib.blake2b(array.tobytes(), digest_size=16).hexdigest(),
        )
    if isinstance(leaf, np.generic):
        return ("numpy_scalar", str(leaf.dtype), leaf.item())
    if isinstance(leaf, (str, int, float, bool, type(None))):
        return ("scalar", leaf)
    return ("repr", type(leaf).__qualname__, repr(leaf))


def _traceable_cache_tree_signature(tree):
    """Build a deterministic cache signature for a pytree-like runtime object."""
    leaves, treedef = jax.tree.flatten(tree)
    return (
        "tree",
        repr(treedef),
        tuple(_traceable_cache_leaf_signature(leaf) for leaf in leaves),
    )


def _traceable_contract_leaf_signature(leaf):
    """Build a cheap immutable-contract signature for one runtime leaf.

    The traceable runtime-entry cache lives only within one Python process and
    the runtime-bundle contract already requires callers not to mutate captured
    geometry/runtime arrays in place. For cache reuse, scalar values still need
    exact matching, but large array leaves only need structural matching once
    the active Boozer solve generation and object identities are part of the
    cache key.
    """
    if isinstance(leaf, jax.Array):
        if leaf.ndim == 0 or leaf.size == 1:
            return (
                "device_array_scalar",
                str(leaf.dtype),
                np.asarray(jax.device_get(leaf)).reshape(()).item(),
            )
        return (
            "device_array_meta",
            str(leaf.dtype),
            tuple(int(dim) for dim in leaf.shape),
        )
    if isinstance(leaf, np.ndarray):
        array = np.asarray(leaf)
        if array.ndim == 0 or array.size == 1:
            return ("array_scalar", str(array.dtype), array.reshape(()).item())
        return ("array_meta", str(array.dtype), tuple(int(dim) for dim in array.shape))
    if isinstance(leaf, np.generic):
        return ("numpy_scalar", str(leaf.dtype), leaf.item())
    if isinstance(leaf, (str, int, float, bool, type(None))):
        return ("scalar", leaf)
    return ("repr", type(leaf).__qualname__, repr(leaf))


def _traceable_contract_tree_signature(tree):
    """Build a cheap cache signature for immutable runtime contracts."""
    leaves, treedef = jax.tree.flatten(tree)
    return (
        "tree",
        repr(treedef),
        tuple(_traceable_contract_leaf_signature(leaf) for leaf in leaves),
    )


def _traceable_runtime_hostify_leaf(leaf):
    """Explicitly materialize JAX runtime constants on the host once.

    JAX transfer guard permits explicit host/device boundaries but rejects
    implicit transfers. The traceable runtime bundle captures solved baseline
    arrays in closures, so those leaves must be converted to host-backed
    NumPy values before compilation rather than being captured as device
    constants.
    """
    if isinstance(leaf, jax.Array):
        return _host_array(leaf)
    if isinstance(leaf, np.ndarray):
        return np.asarray(leaf)
    return leaf


def _traceable_runtime_hostify_tree(tree):
    """Recursively hostify runtime constants used by traceable closures."""
    return jax.tree.map(_traceable_runtime_hostify_leaf, tree)


def _traceable_runtime_deviceify_leaf(leaf, device):
    """Explicitly place cached runtime arrays back onto the active device."""
    if isinstance(leaf, jax.Array):
        return runtime_device_put(leaf, device=device)
    if isinstance(leaf, float):
        return runtime_device_put(leaf, dtype=np.float64, device=device)
    if isinstance(leaf, (np.ndarray, np.generic)):
        return runtime_device_put(leaf, device=device)
    return leaf


def _traceable_runtime_deviceify_tree(tree):
    """Recursively device-place cached runtime arrays for strict diagnostics."""
    device = jax.local_devices()[0]
    return jax.tree.map(
        lambda leaf: _traceable_runtime_deviceify_leaf(leaf, device),
        tree,
    )


def _evaluate_scalar_or_value_and_grad(
    objective_or_value_and_grad,
    coil_dofs,
    *objective_args,
):
    """Evaluate either a cached value/grad callable or a scalar objective."""
    if getattr(objective_or_value_and_grad, "_simsopt_value_and_grad", False):
        return objective_or_value_and_grad(coil_dofs, *objective_args)
    return _strict_scalar_value_and_grad(
        objective_or_value_and_grad,
        coil_dofs,
        *objective_args,
    )


def _evaluate_direct_coil_objective_value(
    objective,
    coil_dofs,
    *objective_args,
):
    """Evaluate a direct coil objective value without building its gradient."""
    return objective(coil_dofs, *objective_args)


def _current_coil_dofs_and_spec(biotsavart):
    """Return the current free coil DOFs and their immutable grouped spec."""
    current_coil_dofs = _current_coil_dofs(biotsavart)
    return current_coil_dofs, biotsavart.coil_set_spec_from_dofs(current_coil_dofs)


def _current_coil_dofs(biotsavart):
    return _as_jax_float64(biotsavart.x.copy())


def _value_and_direct_coil_gradient(
    objective_or_value_and_grad,
    coil_dofs,
    *objective_args,
):
    """Evaluate a cached coil-DOF objective/gradient pair."""
    objective_value, coil_dofs_gradient = _evaluate_scalar_or_value_and_grad(
        objective_or_value_and_grad,
        coil_dofs,
        *objective_args,
    )
    return objective_value, coil_dofs_gradient


def _qs_ratio_from_coil_dofs(sdofs, coil_dofs, biotsavart, **qs_kwargs):
    """Evaluate the QS-ratio objective from explicit coil DOFs via immutable specs."""
    return _qs_ratio_pure(
        sdofs,
        biotsavart.coil_set_spec_from_dofs(coil_dofs),
        **qs_kwargs,
    )


def _booz_solve_observer_active(result):
    return (
        not bool(result.get("success", False))
        or logger.isEnabledFor(logging.DEBUG)
        or os.environ.get("SIMSOPT_BOOZER_OBSERVABILITY") == "1"
    )


def _boozer_solve_observability_payload(result):
    gradient = result.get("gradient")
    grad_inf = None if gradient is None else float(_host_inf_norm(gradient))
    residual = result.get("residual")
    residual_inf = None if residual is None else float(_host_inf_norm(residual))
    return {
        "solve_type": result.get("type", "unknown"),
        "success": bool(result.get("success", False)),
        "grad_inf": grad_inf,
        "residual_inf": residual_inf,
    }


def _log_boozer_solve_state(booz_surf):
    if booz_surf.res is None:
        logger.warning("BoozerSurfaceJAX solve state unavailable: res=None")
        return
    if not _booz_solve_observer_active(booz_surf.res):
        return
    payload = _boozer_solve_observability_payload(booz_surf.res)
    log_fn = logger.debug if payload["success"] else logger.warning
    log_fn(
        "BoozerSurfaceJAX cached solve: type=%s success=%s grad_inf=%s residual_inf=%s",
        payload["solve_type"],
        payload["success"],
        payload["grad_inf"],
        payload["residual_inf"],
    )


def _ensure_solved(booz_surf):
    """Ensure an adjoint-capable solved state exists for legacy callers."""
    _resolved_boozer_adjoint_runtime_state(booz_surf)
    return None


def _ensure_solved_value_state(booz_surf):
    """Ensure a successful solved state exists without requiring adjoint artifacts."""
    if booz_surf.need_to_run_code:
        if booz_surf.res is None:
            raise RuntimeError(
                "BoozerSurfaceJAX has not been solved yet. "
                "Call boozer_surface.run_code(iota, G=G) before "
                "accessing objective values."
            )
        booz_surf.run_code(booz_surf.res["iota"], G=booz_surf.res["G"])
    _log_boozer_solve_state(booz_surf)
    if booz_surf.res is None or not booz_surf.res["primal_success"]:
        raise RuntimeError(
            "BoozerSurfaceJAX has not been solved yet or the last solve failed "
            "to produce a valid solved state."
        )


def _resolved_boozer_solved_runtime_state(booz_surf):
    """Return the solved-state runtime summary for value-path consumers."""
    _ensure_solved_value_state(booz_surf)
    return _require_boozer_runtime_state_method(booz_surf, "get_solved_runtime_state")()


def _resolved_boozer_adjoint_runtime_state(booz_surf):
    """Return the adjoint-state runtime summary for gradient-path consumers."""
    _ensure_solved_value_state(booz_surf)
    return _require_boozer_runtime_state_method(
        booz_surf, "get_adjoint_runtime_state"
    )()


def _require_boozer_runtime_state_method(booz_surf, method_name):
    method = getattr(booz_surf, method_name, None)
    if not callable(method):
        raise TypeError(
            "JAX Boozer objective wrappers require a BoozerSurfaceJAX runtime "
            f"object with {method_name}()."
        )
    return method


def _qs_ratio_pure(
    sdofs,
    coil_set_spec,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    surface_kind,
    axis,
):
    """Pure JAX QS ratio: ``mean(dS * B_nonQS^2) / mean(dS * B_QS^2)``.

    Fully traceable by ``jax.grad`` / ``jax.vjp``.
    """

    gamma, xphi, xtheta = _surface_geometry_from_dofs(
        sdofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
        surface_kind=surface_kind,
    )
    normal = jnp.cross(xphi, xtheta)
    dS = jnp.sqrt(jnp.sum(normal * normal, axis=-1))

    nphi, ntheta = gamma.shape[:2]
    points = gamma.reshape(-1, 3)
    B = grouped_biot_savart_B_from_spec(points, coil_set_spec)
    B = B.reshape(nphi, ntheta, 3)
    modB = jnp.sqrt(jnp.sum(B * B, axis=-1))

    B_QS = jnp.sum(modB * dS, axis=axis) / jnp.sum(dS, axis=axis)

    # Broadcast back to (nphi, ntheta)
    B_QS = jnp.expand_dims(B_QS, axis=axis)

    B_nonQS = modB - B_QS
    return jnp.sum(dS * (B_nonQS * B_nonQS)) / jnp.sum(dS * (B_QS * B_QS))


def _boozer_residual_J_of_x_inner(
    x_inner,
    coil_set_spec,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    surface_kind,
    label_quadpoints_phi,
    label_quadpoints_theta,
    label_mpol,
    label_ntor,
    label_nfp,
    label_stellsym,
    label_scatter_indices,
    label_surface_kind,
    optimize_G,
    weight_inv_modB,
    constraint_weight,
    targetlabel,
    label_type,
    phi_idx,
):
    """BoozerResidual outer objective as a function of inner DOFs.

    Used to compute ``∂J_BR/∂x_inner`` via ``jax.grad`` for the
    adjoint system.

    Args:
        coil_set_spec: immutable grouped-coil geometry/current payload.
    """
    sdofs, iota, G = _split_x_inner_runtime(x_inner, optimize_G)
    if not optimize_G:
        G = compute_G_from_currents(grouped_coil_currents_from_spec(coil_set_spec))

    gamma, xphi, xtheta = _surface_geometry_from_dofs(
        sdofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
        surface_kind=surface_kind,
    )
    nphi, ntheta = gamma.shape[:2]

    points = gamma.reshape(-1, 3)
    B = grouped_biot_savart_B_from_spec(points, coil_set_spec).reshape(
        nphi,
        ntheta,
        3,
    )

    J_boozer = boozer_residual_scalar(
        G,
        iota,
        B,
        xphi,
        xtheta,
        weight_inv_modB=weight_inv_modB,
    )

    label_gamma, label_xphi, label_xtheta = _surface_geometry_from_dofs(
        sdofs,
        label_quadpoints_phi,
        label_quadpoints_theta,
        label_mpol,
        label_ntor,
        label_nfp,
        label_stellsym,
        label_scatter_indices,
        surface_kind=label_surface_kind,
    )
    label_points = label_gamma.reshape(-1, 3)
    label_val = _compute_label(
        label_type,
        _BoozerPenaltyGeometry(
            gamma=label_gamma,
            xphi=label_xphi,
            xtheta=label_xtheta,
        ),
        phi_idx,
        label_points,
        coil_set_spec=coil_set_spec,
    )
    targetlabel_jax = _runtime_float64_scalar(targetlabel, reference=label_val)
    constraint_weight_jax = _runtime_float64_scalar(
        constraint_weight,
        reference=label_val,
    )
    half = _runtime_float64_scalar(0.5, reference=label_val)
    label_delta = label_val - targetlabel_jax
    J_label = half * constraint_weight_jax * (label_delta * label_delta)
    return J_boozer + J_label


class _BoozerObjectiveBase(Optimizable):
    """Shared Optimizable cache/projector shell for Boozer surface objectives."""

    def _init_boozer_objective(self, boozer_surface, biotsavart, *, x0=None):
        if x0 is None:
            Optimizable.__init__(self, depends_on=[boozer_surface])
        else:
            Optimizable.__init__(self, x0=x0, depends_on=[boozer_surface])
        self.boozer_surface = boozer_surface
        self.biotsavart = biotsavart
        self.in_surface = boozer_surface.surface
        self.surface = self.in_surface
        self.recompute_bell()

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None
        self._dJ_by_dcoil_dofs = None

    def J(self):
        if self._J is None:
            self.compute(compute_gradient=False)
        return self._J

    @derivative_dec
    def dJ(self):
        return _public_dJ_from_native_cache(self)

    def dJ_by_dcoil_dofs(self):
        """Return the native flat free-coil-DOF gradient as a JAX array."""
        if self._dJ_by_dcoil_dofs is None:
            solved_state = _resolved_boozer_solved_runtime_state(self.boozer_surface)
            value, self._dJ_by_dcoil_dofs = (
                self._value_and_dJ_by_dcoil_dofs_from_solved_state(solved_state)
            )
            self._J = _public_scalar_value(value)
        return self._dJ_by_dcoil_dofs

    def compute(self, *, compute_gradient=True):
        solved_state = _resolved_boozer_solved_runtime_state(self.boozer_surface)
        if not compute_gradient:
            self._J = _public_scalar_value(
                self._compute_value_from_solved_state(solved_state)
            )
            return
        value, self._dJ_by_dcoil_dofs = (
            self._value_and_dJ_by_dcoil_dofs_from_solved_state(solved_state)
        )
        self._J = _public_scalar_value(value)
        self._dJ = _project_native_dJ_by_dcoil_dofs(self)


class BoozerResidualJAX(_BoozerObjectiveBase):
    r"""JAX equivalent of ``BoozerResidual``.

    Computes

    .. math::

        J = \frac{1}{2N}\|\mathbf r\|^2
            + \frac{w}{2}(\text{label} - \text{target})^2

    and the gradient w.r.t. coil DOFs via implicit differentiation.

    Args:
        boozer_surface: ``BoozerSurfaceJAX`` instance.
        biotsavart: ``BiotSavartJAX`` instance.
    """

    def __init__(self, boozer_surface, biotsavart, *, constraint_weight=None):
        if boozer_surface.boozer_type != "ls":
            raise ValueError(
                "BoozerResidualJAX requires a least-squares BoozerSurfaceJAX "
                "(constraint_weight must be set)."
            )
        self.constraint_weight = (
            float(boozer_surface.constraint_weight)
            if constraint_weight is None
            else float(constraint_weight)
        )
        self._direct_objective_value_and_grad = (
            _make_cached_strict_scalar_value_and_grad(self._direct_objective_of_coils)
        )
        self._init_boozer_objective(boozer_surface, biotsavart)

    def _direct_objective_of_coils(
        self,
        coil_dofs,
        x_inner,
        optimize_G,
        weight_inv_modB,
    ):
        """Pure direct BoozerResidual objective evaluated from explicit coil DOFs."""
        return _boozer_residual_J_of_x_inner(
            x_inner,
            coil_set_spec=self.biotsavart.coil_set_spec_from_dofs(coil_dofs),
            **self._residual_objective_kwargs(
                optimize_G=optimize_G,
                weight_inv_modB=weight_inv_modB,
            ),
        )

    def _inner_objective_state(self, iota, G, *, sdofs=None):
        """Return the packed inner decision vector and optimize-G flag."""
        surface_dofs = (
            self.boozer_surface._get_surface_dofs() if sdofs is None else sdofs
        )
        optimize_G = G is not None
        return (
            self.boozer_surface._pack_decision_vector(iota, G, sdofs=surface_dofs),
            optimize_G,
        )

    def _value_and_dJ_by_dcoil_dofs(
        self,
        solved_state,
        current_coil_dofs,
        coil_set_spec,
    ):
        iota = solved_state.iota
        G = solved_state.G
        weight_inv_modB = solved_state.weight_inv_modB
        x_inner, optimize_G = self._inner_objective_state(
            iota,
            G,
            sdofs=solved_state.sdofs,
        )
        value, direct_gradient = _value_and_direct_coil_gradient(
            self._direct_objective_value_and_grad,
            current_coil_dofs,
            x_inner,
            optimize_G,
            weight_inv_modB,
        )
        adjoint_state = _resolved_boozer_adjoint_runtime_state(self.boozer_surface)
        dJ_ds = self._compute_dJ_ds(
            coil_set_spec,
            iota,
            G,
            weight_inv_modB,
            sdofs=solved_state.sdofs,
        )
        adjoint = _solve_boozer_adjoint(adjoint_state, dJ_ds)
        adjoint_gradient = _adjoint_coil_dofs_gradient(
            adjoint_state.stream_group_vjps,
            adjoint,
            self.biotsavart,
            current_coil_dofs,
        )
        return value, direct_gradient - adjoint_gradient

    def _compute_value_from_solved_state(self, solved_state):
        iota = solved_state.iota
        G = solved_state.G
        weight_inv_modB = solved_state.weight_inv_modB
        current_coil_dofs = _current_coil_dofs(self.biotsavart)
        x_inner, optimize_G = self._inner_objective_state(
            iota,
            G,
            sdofs=solved_state.sdofs,
        )
        return _evaluate_direct_coil_objective_value(
            self._direct_objective_of_coils,
            current_coil_dofs,
            x_inner,
            optimize_G,
            weight_inv_modB,
        )

    def _value_and_dJ_by_dcoil_dofs_from_solved_state(self, solved_state):
        current_coil_dofs, coil_set_spec = _current_coil_dofs_and_spec(self.biotsavart)
        return self._value_and_dJ_by_dcoil_dofs(
            solved_state,
            current_coil_dofs,
            coil_set_spec,
        )

    def _compute_dJ_ds(self, coil_set_spec, iota, G, weight_inv_modB, *, sdofs):
        """Compute ∂J_BR/∂[surface_dofs, iota, G] via JAX autodiff."""
        x_inner, optimize_G = self._inner_objective_state(iota, G, sdofs=sdofs)

        def objective(x):
            return _boozer_residual_J_of_x_inner(
                x,
                coil_set_spec=coil_set_spec,
                **self._residual_objective_kwargs(
                    optimize_G=optimize_G,
                    weight_inv_modB=weight_inv_modB,
                ),
            )

        dJ_ds_jax = _strict_scalar_grad(
            objective,
            x_inner,
        )
        return dJ_ds_jax

    def _residual_objective_kwargs(self, *, optimize_G, weight_inv_modB):
        booz_surf = self.boozer_surface
        return dict(
            quadpoints_phi=booz_surf.quadpoints_phi,
            quadpoints_theta=booz_surf.quadpoints_theta,
            mpol=booz_surf.mpol,
            ntor=booz_surf.ntor,
            nfp=booz_surf.nfp,
            stellsym=booz_surf.stellsym,
            scatter_indices=booz_surf.scatter_indices,
            surface_kind=booz_surf._surface_geometry_kind,
            label_quadpoints_phi=booz_surf.label_quadpoints_phi,
            label_quadpoints_theta=booz_surf.label_quadpoints_theta,
            label_mpol=booz_surf.label_mpol,
            label_ntor=booz_surf.label_ntor,
            label_nfp=booz_surf.label_nfp,
            label_stellsym=booz_surf.label_stellsym,
            label_scatter_indices=booz_surf.label_scatter_indices,
            label_surface_kind=booz_surf._label_surface_geometry_kind,
            optimize_G=optimize_G,
            weight_inv_modB=weight_inv_modB,
            targetlabel=booz_surf.targetlabel,
            constraint_weight=self.constraint_weight,
            label_type=booz_surf.label_type,
            phi_idx=booz_surf.phi_idx,
        )


class IotasJAX(_BoozerObjectiveBase):
    """JAX equivalent of ``Iotas``.

    Returns the rotational transform on the Boozer surface and its
    gradient w.r.t. coil DOFs via the adjoint (no direct B term).

    Args:
        boozer_surface: ``BoozerSurfaceJAX`` instance.
    """

    def __init__(self, boozer_surface):
        self._init_boozer_objective(
            boozer_surface,
            boozer_surface.biotsavart,
            x0=np.asarray([]),
        )

    def _value_and_dJ_by_dcoil_dofs(self, solved_state, current_coil_dofs):
        adjoint_state = _resolved_boozer_adjoint_runtime_state(self.boozer_surface)
        lhs_dtype = _adjoint_state_dtype(adjoint_state)
        n = _adjoint_state_decision_size(adjoint_state)
        if solved_state.G is not None:
            dJ_ds = _explicit_cotangent_basis(n, n - 2, dtype=lhs_dtype)
        else:
            dJ_ds = _explicit_cotangent_basis(n, n - 1, dtype=lhs_dtype)
        adjoint = _solve_boozer_adjoint(adjoint_state, dJ_ds)
        adjoint_gradient = _adjoint_coil_dofs_gradient(
            adjoint_state.stream_group_vjps,
            adjoint,
            self.biotsavart,
            current_coil_dofs,
        )
        return solved_state.iota, -adjoint_gradient

    def _compute_value_from_solved_state(self, solved_state):
        return solved_state.iota

    def _value_and_dJ_by_dcoil_dofs_from_solved_state(self, solved_state):
        current_coil_dofs = _current_coil_dofs(self.biotsavart)
        return self._value_and_dJ_by_dcoil_dofs(
            solved_state,
            current_coil_dofs,
        )


class MajorRadiusJAX(_BoozerObjectiveBase):
    """JAX equivalent of ``MajorRadius`` for solved Boozer surfaces."""

    def __init__(self, boozer_surface):
        self._init_boozer_objective(
            boozer_surface,
            boozer_surface.biotsavart,
            x0=np.asarray([]),
        )

    def _surface_spec(self):
        return _surface_spec_from_surface(self.surface)

    def _compute_value(self, sdofs):
        return surface_major_radius_jax_from_dofs(self._surface_spec(), sdofs)

    def _compute_dJ_ds(self, sdofs, decision_size, dtype):
        dJ_ds_surface = _surface_dmajor_radius_jax_from_dofs(
            self._surface_spec(),
            sdofs,
        )
        return jnp.concatenate(
            (
                dJ_ds_surface,
                _zeros(
                    decision_size - dJ_ds_surface.size,
                    dtype=dtype,
                ),
            )
        )

    def _value_and_dJ_by_dcoil_dofs(self, solved_state, current_coil_dofs):
        value = self._compute_value(solved_state.sdofs)
        adjoint_state = _resolved_boozer_adjoint_runtime_state(self.boozer_surface)
        dJ_ds = self._compute_dJ_ds(
            solved_state.sdofs,
            _adjoint_state_decision_size(adjoint_state),
            _adjoint_state_dtype(adjoint_state),
        )
        adjoint = _solve_boozer_adjoint(adjoint_state, dJ_ds)
        adjoint_gradient = _adjoint_coil_dofs_gradient(
            adjoint_state.stream_group_vjps,
            adjoint,
            self.biotsavart,
            current_coil_dofs,
        )
        return value, -adjoint_gradient

    def _compute_value_from_solved_state(self, solved_state):
        return self._compute_value(solved_state.sdofs)

    def _value_and_dJ_by_dcoil_dofs_from_solved_state(self, solved_state):
        current_coil_dofs = _current_coil_dofs(self.biotsavart)
        return self._value_and_dJ_by_dcoil_dofs(
            solved_state,
            current_coil_dofs,
        )


class NonQuasiSymmetricRatioJAX(_BoozerObjectiveBase):
    r"""JAX equivalent of ``NonQuasiSymmetricRatio``.

    Computes

    .. math::

        J = \frac{\langle dS\, B_{\text{nonQS}}^2 \rangle}
                 {\langle dS\, B_{\text{QS}}^2 \rangle}

    on an auxiliary surface with finer quadrature, and the gradient
    w.r.t. coil DOFs via implicit differentiation.

    Args:
        boozer_surface: ``BoozerSurfaceJAX`` instance.
        biotsavart: ``BiotSavartJAX`` instance.
        sDIM: half-resolution of auxiliary quadrature grid.
        quasi_poloidal: ``True`` for quasi-poloidal, ``False`` for
            quasi-axisymmetric.
    """

    def __init__(self, boozer_surface, biotsavart, sDIM=20, quasi_poloidal=False):
        self.axis = 1 if quasi_poloidal else 0
        s = boozer_surface.surface
        aux_phi = np.linspace(0, 1 / s.nfp, 2 * sDIM, endpoint=False)
        aux_theta = np.linspace(0, 1.0, 2 * sDIM, endpoint=False)
        self._aux_phi_jax = _as_jax_float64(aux_phi)
        self._aux_theta_jax = _as_jax_float64(aux_theta)
        self._init_boozer_objective(boozer_surface, biotsavart)

    def _qs_objective_kwargs(self):
        booz_surf = self.boozer_surface
        return dict(
            quadpoints_phi=self._aux_phi_jax,
            quadpoints_theta=self._aux_theta_jax,
            mpol=booz_surf.mpol,
            ntor=booz_surf.ntor,
            nfp=booz_surf.nfp,
            stellsym=booz_surf.stellsym,
            scatter_indices=booz_surf.scatter_indices,
            surface_kind=booz_surf._surface_geometry_kind,
            axis=self.axis,
        )

    def _compute_value(self, sdofs, coil_set_spec):
        return _qs_ratio_pure(sdofs, coil_set_spec, **self._qs_objective_kwargs())

    def _direct_coil_gradient(self, current_coil_dofs, sdofs):
        qs_kwargs = self._qs_objective_kwargs()

        def J_of_coils(coil_dofs):
            return _qs_ratio_from_coil_dofs(
                sdofs,
                coil_dofs,
                self.biotsavart,
                **qs_kwargs,
            )

        return _strict_scalar_grad(J_of_coils, current_coil_dofs)

    def _compute_dJ_ds(self, coil_set_spec, sdofs, decision_size):
        qs_kwargs = self._qs_objective_kwargs()

        def J_of_sdofs(surface_dofs):
            return _qs_ratio_pure(surface_dofs, coil_set_spec, **qs_kwargs)

        dJ_ds_surface = _strict_scalar_grad(J_of_sdofs, sdofs)
        return jnp.concatenate(
            (
                dJ_ds_surface,
                _zeros(
                    decision_size - dJ_ds_surface.size,
                    dtype=dJ_ds_surface.dtype,
                ),
            )
        )

    def _value_and_dJ_by_dcoil_dofs(
        self,
        solved_state,
        current_coil_dofs,
        coil_set_spec,
    ):
        sdofs = solved_state.sdofs
        value = self._compute_value(sdofs, coil_set_spec)
        direct_gradient = self._direct_coil_gradient(current_coil_dofs, sdofs)
        adjoint_state = _resolved_boozer_adjoint_runtime_state(self.boozer_surface)
        dJ_ds = self._compute_dJ_ds(
            coil_set_spec,
            sdofs,
            _adjoint_state_decision_size(adjoint_state),
        )
        adjoint = _solve_boozer_adjoint(adjoint_state, dJ_ds)
        adjoint_gradient = _adjoint_coil_dofs_gradient(
            adjoint_state.stream_group_vjps,
            adjoint,
            self.biotsavart,
            current_coil_dofs,
        )
        return value, direct_gradient - adjoint_gradient

    def _compute_value_from_solved_state(self, solved_state):
        _, coil_set_spec = _current_coil_dofs_and_spec(self.biotsavart)
        return self._compute_value(solved_state.sdofs, coil_set_spec)

    def _value_and_dJ_by_dcoil_dofs_from_solved_state(self, solved_state):
        current_coil_dofs, coil_set_spec = _current_coil_dofs_and_spec(self.biotsavart)
        return self._value_and_dJ_by_dcoil_dofs(
            solved_state,
            current_coil_dofs,
            coil_set_spec,
        )


def compute_standard_surface_objective_gradients(
    boozer_residual,
    iotas,
    non_qs_ratio,
):
    """Compute the standard LS wrapper gradients with one shared adjoint solve.

    The three wrapper instances must share the same solved
    ``BoozerSurfaceJAX`` result. The function updates each instance's cached
    ``_J`` and ``_dJ`` values and returns the three public gradients in wrapper
    order: ``(BoozerResidualJAX, IotasJAX, NonQuasiSymmetricRatioJAX)``.
    """
    booz_surf = boozer_residual.boozer_surface
    if (
        iotas.boozer_surface is not booz_surf
        or non_qs_ratio.boozer_surface is not booz_surf
    ):
        raise ValueError(
            "Standard surface-objective batching requires all wrappers to share one BoozerSurfaceJAX."
        )
    if non_qs_ratio.biotsavart is not boozer_residual.biotsavart:
        raise ValueError(
            "Standard surface-objective batching requires BoozerResidualJAX and "
            "NonQuasiSymmetricRatioJAX to share one BiotSavartJAX."
        )

    solved_state = _resolved_boozer_solved_runtime_state(booz_surf)
    sdofs = solved_state.sdofs
    iota_value = solved_state.iota
    G = solved_state.G
    weight_inv_modB = solved_state.weight_inv_modB
    adjoint_state = _resolved_boozer_adjoint_runtime_state(booz_surf)
    current_coil_dofs, coil_set_spec = _current_coil_dofs_and_spec(
        boozer_residual.biotsavart
    )

    x_inner, optimize_G = boozer_residual._inner_objective_state(
        iota_value,
        G,
        sdofs=sdofs,
    )
    direct_objective_args = (x_inner, optimize_G, weight_inv_modB)
    residual_value, residual_direct_gradient = _value_and_direct_coil_gradient(
        boozer_residual._direct_objective_value_and_grad,
        current_coil_dofs,
        *direct_objective_args,
    )
    residual_rhs = boozer_residual._compute_dJ_ds(
        coil_set_spec,
        iota_value,
        G,
        weight_inv_modB,
        sdofs=sdofs,
    )

    lhs_dtype = _adjoint_state_dtype(adjoint_state)
    n = _adjoint_state_decision_size(adjoint_state)
    iota_rhs_index = n - 2 if G is not None else n - 1
    iota_rhs = _explicit_cotangent_basis(n, iota_rhs_index, dtype=lhs_dtype)

    non_qs_value = non_qs_ratio._compute_value(sdofs, coil_set_spec)
    non_qs_direct_gradient = non_qs_ratio._direct_coil_gradient(
        current_coil_dofs,
        sdofs,
    )
    non_qs_rhs = non_qs_ratio._compute_dJ_ds(
        coil_set_spec,
        sdofs,
        n,
    )

    def _project_adjoint_gradient(adjoint, biotsavart):
        return _adjoint_coil_dofs_gradient(
            adjoint_state.stream_group_vjps,
            adjoint,
            biotsavart,
            current_coil_dofs,
        )

    rhs_batch = jnp.stack((residual_rhs, iota_rhs, non_qs_rhs), axis=0)
    adjoint_batch = _solve_boozer_adjoint_batch(adjoint_state, rhs_batch)

    # Keep adjoint extraction on the JAX side so strict transfer_guard mode
    # does not materialize Python scalar indices against device-resident state.
    residual_batch, iota_batch, non_qs_batch = tuple(
        jnp.squeeze(chunk, axis=0)
        for chunk in jnp.split(adjoint_batch, rhs_batch.shape[0], axis=0)
    )

    residual_adjoint_gradient = _project_adjoint_gradient(
        residual_batch,
        boozer_residual.biotsavart,
    )
    iota_adjoint_gradient = _project_adjoint_gradient(iota_batch, iotas.biotsavart)
    non_qs_adjoint_gradient = _project_adjoint_gradient(
        non_qs_batch,
        non_qs_ratio.biotsavart,
    )

    residual_gradient = residual_direct_gradient - residual_adjoint_gradient
    iota_gradient = -iota_adjoint_gradient
    non_qs_gradient = non_qs_direct_gradient - non_qs_adjoint_gradient

    boozer_residual._J = residual_value
    boozer_residual._dJ_by_dcoil_dofs = residual_gradient
    boozer_residual._dJ = _project_native_dJ_by_dcoil_dofs(boozer_residual)
    iotas._J = iota_value
    iotas._dJ_by_dcoil_dofs = iota_gradient
    iotas._dJ = _project_native_dJ_by_dcoil_dofs(iotas)
    non_qs_ratio._J = non_qs_value
    non_qs_ratio._dJ_by_dcoil_dofs = non_qs_gradient
    non_qs_ratio._dJ = _project_native_dJ_by_dcoil_dofs(non_qs_ratio)

    return boozer_residual.dJ(), iotas.dJ(), non_qs_ratio.dJ()


# Traceable runtime/cache/custom-VJP builders live in the sibling module. Keep
# the historical ``simsopt_jax_adapters.geo.surface_objectives`` import path stable for
# public builders and direct private helper imports used by downstream tests.
from .surface_objectives_traceable import (
    TraceableObjectiveSeededValueAndGrad as TraceableObjectiveSeededValueAndGrad,
    _TRACEABLE_EXACT_RESIDUAL_KEYS as _TRACEABLE_EXACT_RESIDUAL_KEYS,
    _TRACEABLE_INNER_OBJECTIVE_KEYS as _TRACEABLE_INNER_OBJECTIVE_KEYS,
    _TRACEABLE_LABEL_GEOMETRY_KEYS as _TRACEABLE_LABEL_GEOMETRY_KEYS,
    _TRACEABLE_LABEL_OBJECTIVE_KEYS as _TRACEABLE_LABEL_OBJECTIVE_KEYS,
    _TRACEABLE_SURFACE_GEOMETRY_KEYS as _TRACEABLE_SURFACE_GEOMETRY_KEYS,
    _TRACEABLE_TOTAL_OBJECTIVE_KEYS as _TRACEABLE_TOTAL_OBJECTIVE_KEYS,
    _TraceableCallableSignature as _TraceableCallableSignature,
    _TraceableRuntimeCacheKey as _TraceableRuntimeCacheKey,
    _build_linear_solve_factors_from_res as _build_linear_solve_factors_from_res,
    _build_traceable_objective_cache_state as _build_traceable_objective_cache_state,
    _build_traceable_objective_compiled_bundle_from_state as _build_traceable_objective_compiled_bundle_from_state,
    _build_traceable_objective_state as _build_traceable_objective_state,
    _classify_nonfinite_scalar as _classify_nonfinite_scalar,
    _ensure_traceable_runtime_host_wrappers as _ensure_traceable_runtime_host_wrappers,
    _ensure_traceable_runtime_optimizer_compiled_bundle as _ensure_traceable_runtime_optimizer_compiled_bundle,
    _ensure_traceable_runtime_optimizer_value_and_grad as _ensure_traceable_runtime_optimizer_value_and_grad,
    _ensure_traceable_runtime_public_boundaries as _ensure_traceable_runtime_public_boundaries,
    _ensure_traceable_runtime_reporting_metrics as _ensure_traceable_runtime_reporting_metrics,
    _ensure_traceable_runtime_seeded_value_and_grad as _ensure_traceable_runtime_seeded_value_and_grad,
    _evaluate_traceable_total_objective as _evaluate_traceable_total_objective,
    _get_cached_traceable_runtime_entry as _get_cached_traceable_runtime_entry,
    _host_boundary_with_baseline_peel as _host_boundary_with_baseline_peel,
    _host_input_matches_baseline as _host_input_matches_baseline,
    _hostify_traceable_reporting_metrics as _hostify_traceable_reporting_metrics,
    _make_traceable_batched_value_and_grad_boundary as _make_traceable_batched_value_and_grad_boundary,
    _make_traceable_batched_value_and_grad_pipeline as _make_traceable_batched_value_and_grad_pipeline,
    _make_traceable_field_eval_sharding_pipeline as _make_traceable_field_eval_sharding_pipeline,
    _make_traceable_forward_result_boundary as _make_traceable_forward_result_boundary,
    _make_traceable_forward_value_pipeline as _make_traceable_forward_value_pipeline,
    _make_traceable_host_objective as _make_traceable_host_objective,
    _make_traceable_host_reporting_metrics as _make_traceable_host_reporting_metrics,
    _make_traceable_host_value_and_grad as _make_traceable_host_value_and_grad,
    _make_traceable_lazy_host_reporting_metrics as _make_traceable_lazy_host_reporting_metrics,
    _make_traceable_lazy_reporting_metrics_boundary as _make_traceable_lazy_reporting_metrics_boundary,
    _make_traceable_objective_boundary as _make_traceable_objective_boundary,
    _make_traceable_objective_from_compiled_bundle as _make_traceable_objective_from_compiled_bundle,
    _make_traceable_objective_profile_suite_from_compiled_bundle as _make_traceable_objective_profile_suite_from_compiled_bundle,
    _make_traceable_reporting_metrics as _make_traceable_reporting_metrics,
    _make_traceable_reporting_metrics_bundle as _make_traceable_reporting_metrics_bundle,
    _make_traceable_runtime_jax_array_boundary as _make_traceable_runtime_jax_array_boundary,
    _make_traceable_value_and_grad_boundary as _make_traceable_value_and_grad_boundary,
    _materialize_traceable_objective_state as _materialize_traceable_objective_state,
    _pack_traceable_forward_result as _pack_traceable_forward_result,
    _resolve_traceable_solved_state as _resolve_traceable_solved_state,
    _summarize_traceable_gradient as _summarize_traceable_gradient,
    _summarize_traceable_linear_solve_status as _summarize_traceable_linear_solve_status,
    _summarize_traceable_scalar as _summarize_traceable_scalar,
    _traceable_adjoint_gradient_or_nan as _traceable_adjoint_gradient_or_nan,
    _traceable_directional_inner_objective as _traceable_directional_inner_objective,
    _traceable_directional_inner_stationarity as _traceable_directional_inner_stationarity,
    _traceable_exact_residual_kwargs as _traceable_exact_residual_kwargs,
    _traceable_forward_result as _traceable_forward_result,
    _traceable_general_forward_result as _traceable_general_forward_result,
    _traceable_inner_objective_kwargs as _traceable_inner_objective_kwargs,
    _traceable_inner_stationarity_coil_jvp as _traceable_inner_stationarity_coil_jvp,
    _traceable_iota_from_x_inner as _traceable_iota_from_x_inner,
    _traceable_iota_target_penalty as _traceable_iota_target_penalty,
    _traceable_objective_gradient_parts as _traceable_objective_gradient_parts,
    _traceable_plu_matrix as _traceable_plu_matrix,
    _traceable_plu_matvec as _traceable_plu_matvec,
    _traceable_plu_unpack_lu_piv as _traceable_plu_unpack_lu_piv,
    _traceable_plu_unpack_triple as _traceable_plu_unpack_triple,
    _traceable_predict_warmstart_x as _traceable_predict_warmstart_x,
    _traceable_reporting_metrics_from_solution as _traceable_reporting_metrics_from_solution,
    _traceable_result_linear_solve_factors as _traceable_result_linear_solve_factors,
    _traceable_runtime_cache_key as _traceable_runtime_cache_key,
    _traceable_runtime_option_signature as _traceable_runtime_option_signature,
    _traceable_runtime_reject_host_input as _traceable_runtime_reject_host_input,
    _traceable_solve_exact_linearization as _traceable_solve_exact_linearization,
    _traceable_solve_hessian_linearization as _traceable_solve_hessian_linearization,
    _traceable_solve_linearization as _traceable_solve_linearization,
    _traceable_solve_plu_linearization as _traceable_solve_plu_linearization,
    _traceable_success_filter_signature as _traceable_success_filter_signature,
    _traceable_term_adjoint_solve_report as _traceable_term_adjoint_solve_report,
    _traceable_total_gradient as _traceable_total_gradient,
    _traceable_total_gradient_with_status as _traceable_total_gradient_with_status,
    _traceable_total_objective as _traceable_total_objective,
    _traceable_total_objective_kwargs as _traceable_total_objective_kwargs,
    diagnose_traceable_objective_runtime as diagnose_traceable_objective_runtime,
    make_traceable_objective as make_traceable_objective,
    make_traceable_objective_profile_suite as make_traceable_objective_profile_suite,
    make_traceable_objective_runtime_bundle as make_traceable_objective_runtime_bundle,
    make_traceable_objective_seeded_value_and_grad as make_traceable_objective_seeded_value_and_grad,
    make_traceable_objective_value_and_grad as make_traceable_objective_value_and_grad,
    make_traceable_solved_state_value_and_grad as make_traceable_solved_state_value_and_grad,
    make_traceable_single_stage_alm_runtime_bundle as make_traceable_single_stage_alm_runtime_bundle,
)
