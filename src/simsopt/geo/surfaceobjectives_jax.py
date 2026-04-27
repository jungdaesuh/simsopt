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
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax import lax

from .._core.derivative import Derivative, derivative_dec
from .._core.jax_host_boundary import (
    explicit_cotangent_basis as _explicit_cotangent_basis,
    host_array as _host_array,
    host_bool as _host_bool,
    host_inf_norm as _host_inf_norm,
    host_scalar as _host_scalar,
    scalar_pullback_seed as _explicit_scalar_pullback_seed,
)
from .._core.optimizable import Optimizable
from ..jax_core._math_utils import (
    as_jax_float64 as _as_jax_float64,
    as_runtime_float64 as _as_runtime_float64,
    zeros as _zeros,
)
from ..jax_core.curve_geometry import curve_geometry_from_spec
from ..jax_core.field import (
    grouped_biot_savart_B_from_spec,
    coil_set_spec_from_dof_extraction_spec,
    coil_specs_from_dof_extraction_spec,
    grouped_coil_currents_from_spec,
)
from ..jax_core.sharding import inspect_array_sharding_summary
from .curve import incremental_arclength_pure, kappa_pure
from ._pairwise_reductions import (
    pairwise_min_distance_pure,
    pairwise_selected_smoothmin_distance_pure,
)
from .curveobjectives import (
    Lp_curvature_pure,
    cc_distance_pure,
    cs_distance_pure,
    curve_length_pure,
)
from .boozer_residual_jax import (
    boozer_residual_scalar,
    _surface_geometry_from_dofs,
)
from .boozersurface_jax import (
    _boozer_exact_residual,
    _compute_label,
    _make_boozer_penalty_objective_closure,
)
from . import optimizer_jax as _optimizer_jax
from .label_constraints_jax import compute_G_from_currents
from ._surface_stellsym import (
    compute_stellsym_mask_indices_for_grid as _compute_stellsym_mask_indices_for_grid,
)
from .surface_fourier_jax import surface_volume
from .surfaceobjectives import (
    surface_to_surface_distance_pure,
    surface_to_surface_shortest_distance_pure,
)

__all__ = [
    "BoozerResidualJAX",
    "IotasJAX",
    "NonQuasiSymmetricRatioJAX",
    "compute_standard_surface_objective_gradients",
    "make_traceable_single_stage_alm_runtime_bundle",
    "make_traceable_objective",
    "make_traceable_objective_runtime_bundle",
    "make_traceable_objective_seeded_value_and_grad",
    "make_traceable_objective_value_and_grad",
    "make_traceable_objective_profile_suite",
]

_MISSING_STREAMING_GROUP_VJP_ERROR = (
    "BoozerSurfaceJAX objective wrappers require a streaming grouped-adjoint "
    "callback; the legacy full-pytree adjoint fallback is no longer supported."
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
        raise ValueError(f"Unknown traceable single-stage outer term {term_name!r}.") from exc


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
        ) = _traceable_single_stage_outer_term_dependency_flags(
            candidate_term_name
        )
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


def _explicit_index_array(indices):
    return jax.device_put(np.asarray(indices, dtype=np.int32))


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


def _runtime_float64_array(value, *, reference):
    return _as_runtime_float64(value, reference=reference)


def _runtime_bool(value):
    return _traceable_runtime_deviceify_tree(np.asarray(bool(value), dtype=bool))


def _runtime_zeros_like(value):
    zero = _runtime_float64_scalar(0.0, reference=value)
    return jnp.broadcast_to(zero, value.shape)


def _curve_curve_penalty_from_grouped_spec(coil_set_spec, minimum_distance):
    total = _runtime_float64_scalar(0.0, reference=minimum_distance)
    curve_terms = []
    for group in coil_set_spec.groups:
        gammas = _as_jax_float64(group.gammas)
        gammadashs = _as_jax_float64(group.gammadashs)
        for coil_index in range(int(gammas.shape[0])):
            curve_terms.append(
                (
                    _take_runtime_row(gammas, coil_index),
                    _take_runtime_row(gammadashs, coil_index),
                )
            )
    for curve_index, (gamma_i, gammadash_i) in enumerate(curve_terms):
        for gamma_j, gammadash_j in curve_terms[:curve_index]:
            total = total + cc_distance_pure(
                gamma_i,
                gammadash_i,
                gamma_j,
                gammadash_j,
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
    for group in coil_set_spec.groups:
        gammas = _as_jax_float64(group.gammas)
        gammadashs = _as_jax_float64(group.gammadashs)
        for coil_index in range(int(gammas.shape[0])):
            total = total + cs_distance_pure(
                _take_runtime_row(gammas, coil_index),
                _take_runtime_row(gammadashs, coil_index),
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
        if weight:
            weighted_terms[term_name] = (
                _runtime_float64_scalar(weight, reference=term_value) * term_value
            )
        else:
            weighted_terms[term_name] = _runtime_float64_scalar(
                0.0, reference=term_value
            )
    return weighted_terms


def _traceable_smoothmin_selected(values, temperature):
    values = _as_jax_float64(values).reshape((-1,))
    if int(values.shape[0]) == 0:
        return _runtime_float64_scalar(np.inf, reference=values)
    bounded_temperature = jnp.maximum(
        _runtime_float64_scalar(temperature, reference=values),
        _runtime_float64_scalar(np.finfo(np.float64).eps, reference=values),
    )
    hard_min = jnp.min(values)
    logits = -(values - hard_min) / bounded_temperature
    selection_mask = values <= (
        hard_min + _runtime_float64_scalar(4.0, reference=values) * bounded_temperature
    )
    masked_logits = jnp.where(selection_mask, logits, -jnp.inf)
    return hard_min - bounded_temperature * jax.nn.logsumexp(masked_logits)


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
    if len(coil_gammas) < 2:
        return _runtime_float64_scalar(minimum_distance, reference=minimum_distance)
    point_pairs = []
    for curve_index, gamma_i in enumerate(coil_gammas):
        for gamma_j in coil_gammas[:curve_index]:
            point_pairs.append((gamma_i, gamma_j))
    smooth_min = pairwise_selected_smoothmin_distance_pure(
        tuple(point_pairs),
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
    if len(coil_gammas) == 0:
        return _runtime_float64_scalar(minimum_distance, reference=surface_gamma)
    flat_surface = surface_gamma.reshape((-1, 3))
    point_pairs = tuple((gamma, flat_surface) for gamma in coil_gammas)
    smooth_min = pairwise_selected_smoothmin_distance_pure(
        point_pairs,
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

    Real single-stage fixtures often initialize Boozer least-squares surfaces
    on the VMEC half-period integration grid. That grid uses half-cell-shifted
    phi points for spectral quadrature, so it is valid for the solve but does
    not match the unshifted quadrature families accepted by
    ``SurfaceXYZTensorFourier.get_stellsym_mask()``. The traceable objective is
    evaluated from surface DOFs, so it can safely canonicalize to an exact
    quadrature family when the input surface uses a shifted integration grid.
    """
    quadpoints_phi = np.asarray(booz_jax.quadpoints_phi, dtype=float)
    quadpoints_theta = np.asarray(booz_jax.quadpoints_theta, dtype=float)

    def _mask_indices_for(phi_grid, theta_grid):
        return _compute_stellsym_mask_indices_for_grid(
            mpol=booz_jax.mpol,
            ntor=booz_jax.ntor,
            nfp=booz_jax.nfp,
            stellsym=booz_jax.stellsym,
            quadpoints_phi=phi_grid,
            quadpoints_theta=theta_grid,
        )

    try:
        mask_indices = _mask_indices_for(quadpoints_phi, quadpoints_theta)
    except ValueError:
        phi_max = float(np.max(quadpoints_phi)) if quadpoints_phi.size else 0.0
        half_period_upper = 0.5 / float(booz_jax.nfp)
        if phi_max <= half_period_upper + 1e-12:
            quadpoints_phi = np.linspace(
                0.0,
                half_period_upper,
                int(booz_jax.ntor) + 1,
                endpoint=False,
            )
        else:
            quadpoints_phi = np.linspace(
                0.0,
                1.0 / float(booz_jax.nfp),
                2 * int(booz_jax.ntor) + 1,
                endpoint=False,
            )
        quadpoints_theta = np.linspace(
            0.0,
            1.0,
            2 * int(booz_jax.mpol) + 1,
            endpoint=False,
        )
        mask_indices = _mask_indices_for(quadpoints_phi, quadpoints_theta)

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
    if callable(solve_with_status):
        solution, success = solve_with_status(rhs)
        if not _host_bool(success):
            raise RuntimeError(
                "Boozer adjoint linear solve failed on the operator-backed runtime "
                f"path ({adjoint_state.linearization_kind}, {direction})."
            )
        return solution
    solver = getattr(adjoint_state, f"solve_{direction}", None)
    if not callable(solver):
        raise RuntimeError(
            "Boozer adjoint state exposes no "
            f"solve_{direction}_with_status or solve_{direction}; "
            "cannot solve the inner linearization."
        )
    return solver(rhs)


def _solve_boozer_adjoint_batch(adjoint_state, rhs_batch):
    """Solve several adjoint right-hand sides while preserving vector semantics."""
    rhs_batch = jnp.asarray(rhs_batch)
    if rhs_batch.ndim != 2:
        raise ValueError(
            "_solve_boozer_adjoint_batch expects a rank-2 array with shape "
            "(num_rhs, decision_size)."
        )
    return jnp.stack(
        [_solve_boozer_adjoint(adjoint_state, rhs) for rhs in rhs_batch],
        axis=0,
    )


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
    total_gradient = jnp.zeros_like(coil_dofs)
    for d_coil_array, coil_group_indices in _iter_adjoint_coil_cotangents(
        stream_group_vjps, adjoint
    ):
        total_gradient = total_gradient + biotsavart.coil_cotangents_to_dofs_gradient(
            [d_coil_array],
            [coil_group_indices],
            coil_dofs=coil_dofs,
        )
    return total_gradient


def _coil_dofs_gradient_to_derivative(biotsavart, coil_dofs_gradient):
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
    return _coil_dofs_gradient_to_derivative(
        surface_objective.biotsavart,
        surface_objective._dJ_by_dcoil_dofs,
    )


def _public_dJ_from_native_cache(surface_objective):
    if surface_objective._dJ is None:
        if surface_objective._dJ_by_dcoil_dofs is None:
            surface_objective.compute(compute_gradient=True)
        else:
            surface_objective._dJ = _project_native_dJ_by_dcoil_dofs(
                surface_objective
            )
    return surface_objective._dJ


def _make_cached_strict_scalar_value_and_grad(fun):
    """Cache a strict scalar value/grad callable behind a stable helper contract."""

    def value_and_grad(arg, *args):
        return _strict_scalar_value_and_grad(fun, arg, *args)

    value_and_grad._simsopt_value_and_grad = True
    return value_and_grad


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
    try:
        leaves, treedef = jax.tree_util.tree_flatten(tree)
    except TypeError:
        return _traceable_cache_leaf_signature(tree)
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
    try:
        leaves, treedef = jax.tree_util.tree_flatten(tree)
    except TypeError:
        return _traceable_contract_leaf_signature(tree)
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
    try:
        return jax.tree_util.tree_map(_traceable_runtime_hostify_leaf, tree)
    except TypeError:
        return _traceable_runtime_hostify_leaf(tree)


def _traceable_runtime_deviceify_leaf(leaf):
    """Explicitly place cached runtime arrays back onto the active device."""
    if isinstance(leaf, jax.Array):
        return leaf
    if isinstance(leaf, float):
        return jax.device_put(np.asarray(leaf, dtype=np.float64))
    if isinstance(leaf, (np.ndarray, np.generic)):
        return jax.device_put(np.asarray(leaf))
    return leaf


def _traceable_runtime_deviceify_tree(tree):
    """Recursively device-place cached runtime arrays for strict diagnostics."""
    try:
        return jax.tree_util.tree_map(_traceable_runtime_deviceify_leaf, tree)
    except TypeError:
        return _traceable_runtime_deviceify_leaf(tree)


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
    return _host_scalar(objective(coil_dofs, *objective_args))


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
    return _host_scalar(objective_value), coil_dofs_gradient


def _qs_ratio_from_coil_dofs(sdofs, coil_dofs, biotsavart, **qs_kwargs):
    """Evaluate the QS-ratio objective from explicit coil DOFs via immutable specs."""
    return _qs_ratio_pure(
        sdofs,
        biotsavart.coil_set_spec_from_dofs(coil_dofs),
        **qs_kwargs,
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
    if booz_surf.res is None or not booz_surf.res.get(
        "primal_success",
        booz_surf.res.get("success"),
    ):
        raise RuntimeError(
            "BoozerSurfaceJAX has not been solved yet or the last solve failed "
            "to produce a valid solved state."
        )


def _resolved_boozer_solved_runtime_state(booz_surf):
    """Return the solved-state runtime summary for value-path consumers."""
    _ensure_solved_value_state(booz_surf)
    return _require_boozer_runtime_state_method(
        booz_surf, "get_solved_runtime_state"
    )()


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
        weight_inv_modB,
    )

    label_val = _compute_label(
        label_type,
        gamma,
        xphi,
        xtheta,
        phi_idx,
        points,
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
            self._J, self._dJ_by_dcoil_dofs = (
                self._value_and_dJ_by_dcoil_dofs_from_solved_state(solved_state)
            )
        return self._dJ_by_dcoil_dofs

    def compute(self, *, compute_gradient=True):
        solved_state = _resolved_boozer_solved_runtime_state(self.boozer_surface)
        if not compute_gradient:
            self._J = self._compute_value_from_solved_state(solved_state)
            return
        self._J, self._dJ_by_dcoil_dofs = (
            self._value_and_dJ_by_dcoil_dofs_from_solved_state(solved_state)
        )
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

    def __init__(self, boozer_surface, biotsavart):
        if boozer_surface.boozer_type != "ls":
            raise ValueError(
                "BoozerResidualJAX requires a least-squares BoozerSurfaceJAX "
                "(constraint_weight must be set)."
            )
        self.constraint_weight = float(boozer_surface.constraint_weight)
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
        dJ_ds = self._compute_dJ_ds(coil_set_spec, iota, G, weight_inv_modB)
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

    def _compute_dJ_ds(self, coil_set_spec, iota, G, weight_inv_modB):
        """Compute ∂J_BR/∂[surface_dofs, iota, G] via JAX autodiff."""
        x_inner, optimize_G = self._inner_objective_state(iota, G)

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
        return float(
            _host_scalar(
                _qs_ratio_pure(sdofs, coil_set_spec, **self._qs_objective_kwargs())
            )
        )

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


def _traceable_iota_from_x_inner(x_inner, optimize_G):
    """Extract iota from the inner decision vector."""
    _, iota, _ = _split_x_inner_runtime(x_inner, optimize_G)
    return iota


def _traceable_iota_target_penalty(x_inner, *, optimize_G, iota_target):
    """Quadratic iota-target penalty at an explicit inner state."""
    iota = _traceable_iota_from_x_inner(x_inner, optimize_G)
    half = _runtime_float64_scalar(0.5, reference=iota)
    iota_target_jax = _runtime_float64_scalar(iota_target, reference=iota)
    delta = iota - iota_target_jax
    return half * (delta * delta)


_TRACEABLE_INNER_OBJECTIVE_KEYS = (
    "quadpoints_phi",
    "quadpoints_theta",
    "mpol",
    "ntor",
    "nfp",
    "stellsym",
    "scatter_indices",
    "surface_kind",
    "targetlabel",
    "constraint_weight",
    "label_type",
    "phi_idx",
    "optimize_G",
    "weight_inv_modB",
)

_TRACEABLE_TOTAL_OBJECTIVE_KEYS = (
    "quadpoints_phi",
    "quadpoints_theta",
    "mpol",
    "ntor",
    "nfp",
    "stellsym",
    "scatter_indices",
    "surface_kind",
    "optimize_G",
    "weight_inv_modB",
    "constraint_weight",
    "targetlabel",
    "label_type",
    "phi_idx",
    "iota_target",
    "surface_quadpoints_phi",
    "surface_quadpoints_theta",
    "coil_dof_extraction_spec",
    "outer_objective_config",
)

_TRACEABLE_EXACT_RESIDUAL_KEYS = (
    "exact_quadpoints_phi",
    "exact_quadpoints_theta",
    "mpol",
    "ntor",
    "nfp",
    "stellsym",
    "scatter_indices",
    "surface_kind",
    "targetlabel",
    "label_type",
    "phi_idx",
    "mask_indices",
    "stellsym_surface",
    "weight_inv_modB",
)


def _traceable_inner_objective_kwargs(objective_kwargs):
    """Select the LS inner-objective kwargs from the full traceable contract."""
    return {key: objective_kwargs[key] for key in _TRACEABLE_INNER_OBJECTIVE_KEYS}


def _traceable_total_objective_kwargs(objective_kwargs):
    """Select the scalar total-objective kwargs from the full traceable contract."""
    return {key: objective_kwargs[key] for key in _TRACEABLE_TOTAL_OBJECTIVE_KEYS}


def _traceable_exact_residual_kwargs(objective_kwargs):
    """Select the exact-residual kwargs from the full traceable contract."""
    exact_kwargs = {
        key: objective_kwargs[key] for key in _TRACEABLE_EXACT_RESIDUAL_KEYS
    }
    exact_kwargs["quadpoints_phi"] = exact_kwargs.pop("exact_quadpoints_phi")
    exact_kwargs["quadpoints_theta"] = exact_kwargs.pop("exact_quadpoints_theta")
    return exact_kwargs


def _traceable_total_objective(
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
    """Pure single-stage objective evaluated at an explicit inner state."""
    if outer_objective_config is not None:
        return _traceable_full_single_stage_outer_objective(
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
        optimize_G=optimize_G,
        weight_inv_modB=weight_inv_modB,
        constraint_weight=constraint_weight,
        targetlabel=targetlabel,
        label_type=label_type,
        phi_idx=phi_idx,
    )
    return J_boozer + _traceable_iota_target_penalty(
        x_inner,
        optimize_G=optimize_G,
        iota_target=iota_target,
    )


def _evaluate_traceable_total_objective(
    x_inner,
    coil_dofs,
    coil_set_spec,
    objective_kwargs,
):
    """Evaluate the full traceable scalar objective from packed kwargs."""
    return _traceable_total_objective(
        x_inner,
        coil_dofs,
        coil_set_spec,
        **_traceable_total_objective_kwargs(objective_kwargs),
    )


def _traceable_directional_inner_stationarity(
    x_inner,
    tangent,
    coil_set_spec,
    **objective_kwargs,
):
    """Directional inner stationarity without materializing the full gradient."""
    runtime_objective_kwargs = _traceable_runtime_deviceify_tree(objective_kwargs)
    inner_objective = _make_boozer_penalty_objective_closure(
        coil_set_spec=coil_set_spec,
        **runtime_objective_kwargs,
    )
    return jax.jvp(inner_objective, (x_inner,), (tangent,))[1]


def _traceable_inner_stationarity_coil_jvp(
    x_inner,
    coil_dofs,
    coil_dofs_tangent,
    coil_set_spec_from_dofs,
    **objective_kwargs,
):
    """Directional coil derivative of inner stationarity without a basis map."""
    runtime_objective_kwargs = _traceable_runtime_deviceify_tree(objective_kwargs)

    def coil_directional_inner_objective(current_x_inner):
        def inner_objective_of_coils(current_coil_dofs):
            inner_objective = _make_boozer_penalty_objective_closure(
                coil_set_spec=coil_set_spec_from_dofs(current_coil_dofs),
                **runtime_objective_kwargs,
            )
            return inner_objective(current_x_inner)

        return jax.jvp(
            inner_objective_of_coils,
            (coil_dofs,),
            (coil_dofs_tangent,),
        )[1]

    return _strict_scalar_grad(coil_directional_inner_objective, x_inner)


def _traceable_directional_inner_objective(
    x_inner,
    tangent,
    coil_set_spec,
    **objective_kwargs,
):
    """Directional derivative of the LS inner objective at an explicit state."""
    return _traceable_directional_inner_stationarity(
        x_inner,
        tangent,
        coil_set_spec,
        **objective_kwargs,
    )


def _traceable_solve_hessian_linearization(
    booz_jax,
    solved_x,
    rhs,
    coil_set_spec,
    objective_kwargs,
    *,
    linear_solve_factors,
    linear_solve_tol,
    linear_solve_stab,
    transpose,
):
    if linear_solve_factors is not None:
        return _traceable_solve_plu_linearization(
            linear_solve_factors,
            rhs,
            linear_solve_tol=linear_solve_tol,
            transpose=transpose,
        )

    objective_fn = _make_boozer_penalty_objective_closure(
        coil_set_spec=coil_set_spec,
        **_traceable_inner_objective_kwargs(objective_kwargs),
    )
    candidate_stabs = tuple(
        float(candidate)
        for candidate in booz_jax._adjoint_hessian_stabilization_schedule()
    ) or (float(linear_solve_stab),)

    def _solve_for_stab(candidate_stab):
        return _optimizer_jax._solve_hessian_system_with_status(
            objective_fn,
            solved_x,
            rhs,
            stab=candidate_stab,
            tol=linear_solve_tol,
        )

    if not isinstance(rhs, jax.core.Tracer):
        solution, success = _solve_for_stab(candidate_stabs[0])
        for candidate_stab in candidate_stabs[1:]:
            if _host_bool(success):
                break
            solution, success = _solve_for_stab(candidate_stab)
        return solution, success

    candidate_stabs_array = jnp.asarray(candidate_stabs, dtype=jnp.asarray(rhs).dtype)
    next_index = jnp.asarray(1, dtype=jnp.int32)
    candidate_count = jnp.asarray(candidate_stabs_array.shape[0], dtype=jnp.int32)
    solution0, success0 = _solve_for_stab(candidate_stabs_array[0])

    def cond_fun(state):
        index, _solution, success = state
        return jnp.logical_and(
            jnp.logical_not(success),
            index < candidate_count,
        )

    def body_fun(state):
        index, _solution, _success = state
        solution, success = _solve_for_stab(candidate_stabs_array[index])
        return index + next_index, solution, success

    _, solution, success = lax.while_loop(
        cond_fun,
        body_fun,
        (next_index, solution0, success0),
    )
    return solution, success


def _traceable_plu_matvec(linear_solve_factors, vector, *, transpose):
    P, L, U = linear_solve_factors
    if transpose:
        return U.T @ (L.T @ (P.T @ vector))
    return P @ (L @ (U @ vector))


def _traceable_plu_matrix(linear_solve_factors):
    P, L, U = linear_solve_factors
    return P @ (L @ U)


def _traceable_plu_residual_tolerance(
    linear_solve_factors,
    solution,
    rhs,
    residual_tol,
):
    matrix = _traceable_plu_matrix(linear_solve_factors)
    dtype = rhs.dtype
    eps = _optimizer_jax._device_scalar(jnp.finfo(dtype).eps, dtype=dtype)
    dimension = _optimizer_jax._device_scalar(matrix.shape[0], dtype=dtype)
    safety = _optimizer_jax._device_scalar(100.0, dtype=dtype)
    backward_error = (
        safety
        * dimension
        * eps
        * (
            jnp.linalg.norm(matrix) * jnp.linalg.norm(solution)
            + jnp.linalg.norm(rhs)
        )
    )
    return jnp.maximum(residual_tol, backward_error)


def _traceable_solve_plu_linearization(
    linear_solve_factors,
    rhs,
    *,
    linear_solve_tol,
    transpose,
):
    """Solve a dense PLU snapshot with a residual-quality success contract."""
    P, L, U = linear_solve_factors
    if transpose:
        y = jsp_linalg.solve_triangular(U.T, rhs, lower=True)
        z = jsp_linalg.solve_triangular(L.T, y, lower=False)
        solution = P @ z
    else:
        y = jsp_linalg.solve_triangular(L, P.T @ rhs, lower=True)
        solution = jsp_linalg.solve_triangular(U, y, lower=False)
    residual = rhs - _traceable_plu_matvec(
        linear_solve_factors,
        solution,
        transpose=transpose,
    )
    residual_norm = jnp.linalg.norm(residual)
    residual_tol = _optimizer_jax._linear_solve_residual_tolerance(
        rhs,
        linear_solve_tol,
    )
    residual_tol = _traceable_plu_residual_tolerance(
        linear_solve_factors,
        solution,
        rhs,
        residual_tol,
    )
    success = (
        jnp.all(jnp.isfinite(solution))
        & jnp.all(jnp.isfinite(residual))
        & jnp.isfinite(residual_norm)
        & (residual_norm <= residual_tol)
    )
    return solution, success


def _traceable_solve_exact_linearization(
    solved_x,
    rhs,
    coil_set_spec,
    objective_kwargs,
    *,
    linear_solve_tol,
    transpose,
):
    def residual_fn(x_inner):
        return _boozer_exact_residual(
            x_inner,
            coil_set_spec=coil_set_spec,
            **_traceable_exact_residual_kwargs(objective_kwargs),
        )

    return _optimizer_jax._solve_jacobian_system_with_status(
        residual_fn,
        solved_x,
        rhs,
        transpose=transpose,
        tol=linear_solve_tol,
    )


def _traceable_solve_linearization(
    booz_jax,
    solved_x,
    rhs,
    coil_set_spec,
    objective_kwargs,
    *,
    linear_solve_factors,
    linearization_kind,
    linear_solve_tol,
    linear_solve_stab,
    transpose,
):
    if linearization_kind == "hessian":
        return _traceable_solve_hessian_linearization(
            booz_jax,
            solved_x,
            rhs,
            coil_set_spec,
            objective_kwargs,
            linear_solve_factors=linear_solve_factors,
            linear_solve_tol=linear_solve_tol,
            linear_solve_stab=linear_solve_stab,
            transpose=transpose,
        )
    if linearization_kind == "exact_jacobian":
        return _traceable_solve_exact_linearization(
            solved_x,
            rhs,
            coil_set_spec,
            objective_kwargs,
            linear_solve_tol=linear_solve_tol,
            transpose=transpose,
        )
    raise ValueError(
        f"Unsupported traceable linearization kind {linearization_kind!r}."
    )


def _pack_traceable_forward_result(
    *,
    value,
    x,
    sdofs,
    iota,
    G,
    linear_solve_factors,
    success,
    primal_success,
    adjoint_linear_solve_available,
):
    """Return the normalized traceable forward-result contract."""
    return {
        "value": value,
        "x": x,
        "sdofs": sdofs,
        "iota": iota,
        "G": G,
        "linear_solve_factors": linear_solve_factors,
        "success": success,
        "primal_success": primal_success,
        "adjoint_linear_solve_available": adjoint_linear_solve_available,
    }


def _traceable_result_linear_solve_factors(solve_result, linearization_kind):
    """Return factors carried by traceable autodiff state, if this kind uses them."""
    if linearization_kind == "exact_jacobian":
        return None
    return solve_result["plu"]


def _resolve_traceable_solved_state(
    booz_jax,
    solve_result,
    *,
    optimize_G,
    coil_set_spec,
):
    """Return solved ``(sdofs, iota, G)`` even when the solve only returns ``x``."""
    if (
        "sdofs" in solve_result
        and "iota" in solve_result
        and ("G" in solve_result or not optimize_G)
    ):
        return (
            solve_result["sdofs"],
            solve_result["iota"],
            solve_result.get("G"),
        )
    return booz_jax._unpack_decision_vector_jax(
        solve_result["x"],
        optimize_G,
        coil_set_spec=coil_set_spec,
    )


def _traceable_general_forward_result(
    booz_jax,
    coil_set_spec_from_dofs,
    *,
    coil_dofs,
    baseline_x,
    baseline_linear_solve_factors,
    linearization_kind,
    linear_solve_tol,
    linear_solve_stab,
    optimize_G,
    baseline_coil_dofs,
    predictor_kind,
    objective_kwargs,
    success_filter,
):
    """Run the general traceable inner solve without the baseline fast path."""
    coil_set_spec = coil_set_spec_from_dofs(coil_dofs)
    warmstart_x, warmstart_linear_solve_success = _traceable_predict_warmstart_x(
        booz_jax,
        coil_set_spec_from_dofs,
        coil_dofs=coil_dofs,
        baseline_coil_dofs=baseline_coil_dofs,
        baseline_x=baseline_x,
        baseline_linear_solve_factors=baseline_linear_solve_factors,
        linearization_kind=linearization_kind,
        linear_solve_tol=linear_solve_tol,
        linear_solve_stab=linear_solve_stab,
        predictor_kind=predictor_kind,
        objective_kwargs=objective_kwargs,
    )

    def _run_traceable_solve(_):
        warmstart_sdofs, warmstart_iota, warmstart_G = booz_jax._unpack_decision_vector_jax(
            warmstart_x,
            optimize_G,
            coil_set_spec=coil_set_spec,
        )
        solve_result = booz_jax.run_code_traceable(
            coil_set_spec,
            warmstart_sdofs,
            warmstart_iota,
            warmstart_G,
        )
        solved_sdofs, solved_iota, solved_G = _resolve_traceable_solved_state(
            booz_jax,
            solve_result,
            optimize_G=optimize_G,
            coil_set_spec=coil_set_spec,
        )
        primal_success = solve_result.get("primal_success", solve_result["success"])
        adjoint_linear_solve_available = solve_result.get(
            "adjoint_linear_solve_available",
            solve_result["success"],
        )
        success = primal_success
        if success_filter is not None:
            success = success & jax.lax.cond(
                primal_success,
                lambda _: success_filter(coil_dofs, solve_result["x"]),
                lambda _: _runtime_bool(False),
                operand=None,
            )
        objective_value = _evaluate_traceable_total_objective(
            solve_result["x"],
            coil_dofs,
            coil_set_spec,
            objective_kwargs,
        )
        return _pack_traceable_forward_result(
            value=objective_value,
            x=solve_result["x"],
            sdofs=solved_sdofs,
            iota=solved_iota,
            G=solved_G,
            linear_solve_factors=_traceable_result_linear_solve_factors(
                solve_result,
                linearization_kind,
            ),
            success=success,
            primal_success=primal_success,
            adjoint_linear_solve_available=adjoint_linear_solve_available,
        )

    if linearization_kind != "exact_jacobian":
        return _run_traceable_solve(None)

    def _warmstart_failure(_):
        warmstart_sdofs, warmstart_iota, warmstart_G = booz_jax._unpack_decision_vector_jax(
            warmstart_x,
            optimize_G,
            coil_set_spec=coil_set_spec,
        )
        failure_value = _evaluate_traceable_total_objective(
            warmstart_x,
            coil_dofs,
            coil_set_spec,
            objective_kwargs,
        )
        failure = _runtime_bool(False)
        return _pack_traceable_forward_result(
            value=failure_value,
            x=warmstart_x,
            sdofs=warmstart_sdofs,
            iota=warmstart_iota,
            G=warmstart_G,
            linear_solve_factors=None,
            success=failure,
            primal_success=failure,
            adjoint_linear_solve_available=failure,
        )

    return lax.cond(
        warmstart_linear_solve_success,
        _run_traceable_solve,
        _warmstart_failure,
        operand=None,
    )


def _traceable_forward_result(
    booz_jax,
    coil_set_spec_from_dofs,
    *,
    coil_dofs,
    baseline_x,
    baseline_value,
    baseline_linear_solve_factors,
    linearization_kind,
    linear_solve_tol,
    linear_solve_stab,
    optimize_G,
    baseline_coil_dofs,
    predictor_kind,
    objective_kwargs,
    success_filter,
):
    """Run the pure traceable inner solve and return value plus solver data."""
    same_coils = jnp.all(coil_dofs == baseline_coil_dofs)

    def baseline_case(_):
        baseline_sdofs, baseline_iota, baseline_G = _split_x_inner_runtime(
            baseline_x,
            optimize_G,
        )
        return _pack_traceable_forward_result(
            # The exact baseline state must return the solved reference objective so
            # the outer optimizer can obtain a real descent direction even when the
            # seed is hardware-invalid. Candidate (non-baseline) states remain
            # subject to the hard success filter below.
            value=baseline_value,
            x=baseline_x,
            sdofs=baseline_sdofs,
            iota=baseline_iota,
            G=baseline_G,
            linear_solve_factors=baseline_linear_solve_factors,
            success=_runtime_bool(True),
            primal_success=_runtime_bool(True),
            adjoint_linear_solve_available=_runtime_bool(False),
        )

    def general_case(_):
        return _traceable_general_forward_result(
            booz_jax,
            coil_set_spec_from_dofs,
            coil_dofs=coil_dofs,
            baseline_x=baseline_x,
            baseline_linear_solve_factors=baseline_linear_solve_factors,
            linearization_kind=linearization_kind,
            linear_solve_tol=linear_solve_tol,
            linear_solve_stab=linear_solve_stab,
            optimize_G=optimize_G,
            baseline_coil_dofs=baseline_coil_dofs,
            predictor_kind=predictor_kind,
            objective_kwargs=objective_kwargs,
            success_filter=success_filter,
        )

    return jax.lax.cond(same_coils, baseline_case, general_case, operand=None)


def _traceable_total_gradient(
    booz_jax,
    coil_set_spec_from_dofs,
    *,
    coil_dofs,
    solved_x,
    solved_linear_solve_factors,
    linearization_kind,
    linear_solve_tol,
    linear_solve_stab,
    objective_kwargs,
):
    """Implicit total derivative of the pure traceable objective."""
    return _traceable_objective_gradient_parts(
        booz_jax,
        coil_set_spec_from_dofs,
        coil_dofs=coil_dofs,
        solved_x=solved_x,
        solved_linear_solve_factors=solved_linear_solve_factors,
        linearization_kind=linearization_kind,
        linear_solve_tol=linear_solve_tol,
        linear_solve_stab=linear_solve_stab,
        objective_kwargs=objective_kwargs,
    )[2]


def _traceable_adjoint_gradient_or_nan(gradient, linear_solve_success):
    """Surface adjoint-solve failures as non-finite gradients, not fallbacks."""
    return lax.cond(
        linear_solve_success,
        lambda _: gradient,
        lambda _: _traceable_adjoint_fail_gradient_like(gradient),
        operand=None,
    )


def _traceable_total_gradient_with_status(
    booz_jax,
    coil_set_spec_from_dofs,
    *,
    coil_dofs,
    solved_x,
    solved_linear_solve_factors,
    linearization_kind,
    linear_solve_tol,
    linear_solve_stab,
    objective_kwargs,
    scalar_objective_fn=None,
):
    _, _, total_grad, linear_solve_success = _traceable_objective_gradient_parts(
        booz_jax,
        coil_set_spec_from_dofs,
        coil_dofs=coil_dofs,
        solved_x=solved_x,
        solved_linear_solve_factors=solved_linear_solve_factors,
        linearization_kind=linearization_kind,
        linear_solve_tol=linear_solve_tol,
        linear_solve_stab=linear_solve_stab,
        objective_kwargs=objective_kwargs,
        scalar_objective_fn=scalar_objective_fn,
    )
    return total_grad, linear_solve_success


def _traceable_objective_gradient_parts(
    booz_jax,
    coil_set_spec_from_dofs,
    *,
    coil_dofs,
    solved_x,
    solved_linear_solve_factors,
    linearization_kind,
    linear_solve_tol,
    linear_solve_stab,
    objective_kwargs,
    term_name=None,
    scalar_objective_fn=None,
):
    """Return direct, implicit, and total gradients for one traceable objective."""
    if scalar_objective_fn is not None and term_name is not None:
        raise ValueError(
            "scalar_objective_fn and term_name are mutually exclusive traceable "
            "gradient selectors."
        )

    def _evaluate_objective(x_inner, current_coil_dofs, coil_set_spec):
        if scalar_objective_fn is not None:
            return scalar_objective_fn(
                x_inner,
                current_coil_dofs,
                coil_set_spec,
                objective_kwargs=objective_kwargs,
            )
        if term_name is None:
            return _evaluate_traceable_total_objective(
                x_inner,
                current_coil_dofs,
                coil_set_spec,
                objective_kwargs,
            )
        return _evaluate_traceable_weighted_single_stage_outer_term(
            term_name,
            x_inner,
            current_coil_dofs,
            coil_set_spec,
            objective_kwargs,
        )

    def _evaluate_objective_of_coils(current_coil_dofs):
        return _evaluate_objective(
            solved_x,
            current_coil_dofs,
            coil_set_spec_from_dofs(current_coil_dofs),
        )

    coil_set_spec = coil_set_spec_from_dofs(coil_dofs)
    depends_on_x_inner = True
    depends_on_coil_dofs = True
    if scalar_objective_fn is None:
        depends_on_x_inner, depends_on_coil_dofs = (
            _traceable_single_stage_effective_dependency_flags(
                term_name,
                objective_kwargs=objective_kwargs,
            )
        )

    if not depends_on_x_inner:
        dJ_dx = _runtime_zeros_like(solved_x)
        adjoint = _runtime_zeros_like(solved_x)
        linear_solve_success = _runtime_bool(True)
    else:
        dJ_dx = _strict_scalar_grad(
            lambda x: _evaluate_objective(x, coil_dofs, coil_set_spec),
            solved_x,
        )
        adjoint, linear_solve_success = _traceable_solve_linearization(
            booz_jax,
            solved_x,
            dJ_dx,
            coil_set_spec,
            objective_kwargs,
            linear_solve_factors=solved_linear_solve_factors,
            linearization_kind=linearization_kind,
            linear_solve_tol=linear_solve_tol,
            linear_solve_stab=linear_solve_stab,
            transpose=True,
        )

    if not depends_on_coil_dofs:
        # Some diagnostic terms depend only on the solved inner state, so
        # their explicit coil derivative is exactly zero. Avoid autodiff on
        # these constant-in-coils scalars under strict transfer guard because
        # JAX 0.9.2 instantiates host scalar zeros for null tangent paths.
        direct_grad = _runtime_zeros_like(coil_dofs)
    else:
        direct_grad = _strict_scalar_grad(_evaluate_objective_of_coils, coil_dofs)

    if not depends_on_x_inner:
        implicit_grad = _runtime_zeros_like(coil_dofs)
        return direct_grad, implicit_grad, direct_grad, linear_solve_success

    inner_objective_kwargs = _traceable_inner_objective_kwargs(objective_kwargs)

    def directional_stationarity_of_coils(current_coil_dofs):
        return _traceable_directional_inner_stationarity(
            solved_x,
            adjoint,
            coil_set_spec_from_dofs(current_coil_dofs),
            **inner_objective_kwargs,
        )

    implicit_grad = _strict_scalar_grad(
        directional_stationarity_of_coils,
        coil_dofs,
    )
    total_grad = _traceable_adjoint_gradient_or_nan(
        direct_grad - implicit_grad,
        linear_solve_success,
    )
    return direct_grad, implicit_grad, total_grad, linear_solve_success


def _traceable_predict_warmstart_x(
    booz_jax,
    coil_set_spec_from_dofs,
    *,
    coil_dofs,
    baseline_coil_dofs,
    baseline_x,
    baseline_linear_solve_factors,
    linearization_kind,
    linear_solve_tol,
    linear_solve_stab,
    predictor_kind,
    objective_kwargs,
):
    """Predict a coil-dependent warm start via a first-order implicit step."""
    delta = coil_dofs - baseline_coil_dofs

    if predictor_kind == "exact":
        exact_residual_kwargs = _traceable_exact_residual_kwargs(objective_kwargs)

        def baseline_residual_of_coils(cd):
            return _boozer_exact_residual(
                baseline_x,
                coil_set_spec=coil_set_spec_from_dofs(cd),
                **exact_residual_kwargs,
            )

        forcing = jax.jvp(
            baseline_residual_of_coils,
            (baseline_coil_dofs,),
            (delta,),
        )[1]
    else:
        inner_objective_kwargs = _traceable_inner_objective_kwargs(objective_kwargs)
        forcing = _traceable_inner_stationarity_coil_jvp(
            baseline_x,
            baseline_coil_dofs,
            delta,
            coil_set_spec_from_dofs,
            **inner_objective_kwargs,
        )

    dx, linear_solve_success = _traceable_solve_linearization(
        booz_jax,
        baseline_x,
        -forcing,
        coil_set_spec_from_dofs(baseline_coil_dofs),
        objective_kwargs,
        linear_solve_factors=baseline_linear_solve_factors,
        linearization_kind=linearization_kind,
        linear_solve_tol=linear_solve_tol,
        linear_solve_stab=linear_solve_stab,
        transpose=False,
    )
    predicted_x = baseline_x + dx
    preserve_failed_predictor = (
        predictor_kind == "exact" or linearization_kind == "exact_jacobian"
    )
    if preserve_failed_predictor:
        return predicted_x, linear_solve_success
    return (
        lax.cond(
            linear_solve_success,
            lambda _: predicted_x,
            lambda _: baseline_x,
            operand=None,
        ),
        linear_solve_success,
    )


def _build_traceable_objective_state(
    booz_jax,
    bs_jax,
    iota_target,
    *,
    outer_objective_config=None,
):
    """Return the shared state used by the traceable objective builders.

    This setup reads the solved mutable object state once, computes the solved
    baseline objective in JAX, then explicitly hostifies the captured runtime
    constants before building the compiled target-lane closures. The resulting
    closures stay pure in the hot path without capturing device-backed arrays
    that would trip strict transfer-guard lowering.
    """
    if booz_jax.boozer_type == "ls":
        objective_method = booz_jax._resolve_optimizer_method()
        if objective_method not in {"bfgs-ondevice", "lbfgs-ondevice", "lm-ondevice"}:
            raise RuntimeError(
                "make_traceable_objective() requires optimizer_backend='ondevice'."
            )

    solved_state = _resolved_boozer_solved_runtime_state(booz_jax)
    warmstart_sdofs = solved_state.sdofs
    warmstart_iota = solved_state.iota
    warmstart_G = solved_state.G

    baseline_coil_dofs = _as_jax_float64(bs_jax.x.copy())
    coil_dof_extraction_spec = _traceable_runtime_hostify_tree(
        bs_jax.coil_dof_extraction_spec()
    )
    coil_set_spec_from_dofs = lambda coil_dofs: coil_set_spec_from_dof_extraction_spec(
        coil_dof_extraction_spec,
        coil_dofs,
    )
    optimize_G = warmstart_G is not None
    predictor_kind = booz_jax.boozer_type
    solve_quadpoints_phi = _as_jax_float64(
        np.asarray(booz_jax.quadpoints_phi, dtype=float)
    )
    solve_quadpoints_theta = _as_jax_float64(
        np.asarray(booz_jax.quadpoints_theta, dtype=float)
    )
    exact_quadpoints_phi, exact_quadpoints_theta, mask_indices = (
        _canonicalize_traceable_exact_quadrature(booz_jax)
    )
    objective_kwargs = {
        "quadpoints_phi": solve_quadpoints_phi,
        "quadpoints_theta": solve_quadpoints_theta,
        "mpol": booz_jax.mpol,
        "ntor": booz_jax.ntor,
        "nfp": booz_jax.nfp,
        "stellsym": booz_jax.stellsym,
        "scatter_indices": booz_jax.scatter_indices,
        "surface_kind": booz_jax._surface_geometry_kind,
        "optimize_G": optimize_G,
        "weight_inv_modB": solved_state.weight_inv_modB,
        "constraint_weight": booz_jax.constraint_weight,
        "targetlabel": booz_jax.targetlabel,
        "label_type": booz_jax.label_type,
        "phi_idx": booz_jax.phi_idx,
        "iota_target": _as_jax_float64(iota_target),
        "exact_quadpoints_phi": exact_quadpoints_phi,
        "exact_quadpoints_theta": exact_quadpoints_theta,
        "surface_quadpoints_phi": _as_jax_float64(
            np.asarray(booz_jax.surface.quadpoints_phi, dtype=float)
        ),
        "surface_quadpoints_theta": _as_jax_float64(
            np.asarray(booz_jax.surface.quadpoints_theta, dtype=float)
        ),
        "coil_dof_extraction_spec": coil_dof_extraction_spec,
        "outer_objective_config": outer_objective_config,
        "mask_indices": mask_indices,
        "stellsym_surface": booz_jax.stellsym,
    }
    linearization_kind = booz_jax.res.get(
        "linearization_kind",
        "exact_jacobian" if booz_jax.boozer_type == "exact" else "hessian",
    )
    baseline_linear_solve_factors = (
        None if linearization_kind == "exact_jacobian" else booz_jax.res["PLU"]
    )
    linear_solve_tol = booz_jax._linear_solve_tolerance()
    linear_solve_stab = float(booz_jax.options.get("newton_stab", 0.0))

    baseline_x = booz_jax._pack_decision_vector(
        warmstart_iota,
        warmstart_G,
        sdofs=warmstart_sdofs,
    )

    baseline_value = _evaluate_traceable_total_objective(
        baseline_x,
        baseline_coil_dofs,
        bs_jax.coil_set_spec_from_dofs(baseline_coil_dofs),
        objective_kwargs,
    )
    return {
        "objective_kwargs": _traceable_runtime_hostify_tree(objective_kwargs),
        "baseline_x": _traceable_runtime_hostify_tree(baseline_x),
        "baseline_value": _traceable_runtime_hostify_tree(baseline_value),
        "baseline_linear_solve_factors": _traceable_runtime_hostify_tree(
            baseline_linear_solve_factors
        ),
        "baseline_coil_dofs": _traceable_runtime_hostify_tree(baseline_coil_dofs),
        "coil_dof_extraction_spec": coil_dof_extraction_spec,
        "coil_set_spec_from_dofs": coil_set_spec_from_dofs,
        "optimize_G": optimize_G,
        "predictor_kind": predictor_kind,
        "linearization_kind": linearization_kind,
        "linear_solve_tol": linear_solve_tol,
        "linear_solve_stab": linear_solve_stab,
    }


def _traceable_runtime_reject_host_input(coil_dofs, entrypoint_name):
    if isinstance(coil_dofs, (np.ndarray, np.generic, list, tuple, float, int)):
        raise RuntimeError(
            f"{entrypoint_name} requires a JAX array. Host inputs must enter "
            "through an explicit staging boundary; transfer_guard=disallow "
            "rejects implicit host-to-device transfer."
        )
    return coil_dofs


def _make_traceable_runtime_jax_array_boundary(compiled_callable, entrypoint_name):
    def boundary(coil_dofs):
        return compiled_callable(
            _traceable_runtime_reject_host_input(coil_dofs, entrypoint_name)
        )

    return boundary


def _build_traceable_objective_compiled_bundle_from_state(
    booz_jax,
    state,
    *,
    success_filter=None,
    general_only_forward=False,
):
    """Build shared compiled closures for one traceable single-stage state."""
    from .optimizer_jax import _mark_cacheable_jit_value_and_grad

    objective_kwargs = state["objective_kwargs"]
    baseline_x = state["baseline_x"]
    baseline_value = state["baseline_value"]
    baseline_linear_solve_factors = state["baseline_linear_solve_factors"]
    baseline_coil_dofs = state["baseline_coil_dofs"]
    optimize_G = state["optimize_G"]
    predictor_kind = state["predictor_kind"]
    linearization_kind = state["linearization_kind"]
    linear_solve_tol = state["linear_solve_tol"]
    linear_solve_stab = state["linear_solve_stab"]
    coil_set_spec_from_dofs = state["coil_set_spec_from_dofs"]

    def _forward_result_for(coil_dofs):
        if general_only_forward:
            return _traceable_general_forward_result(
                booz_jax,
                coil_set_spec_from_dofs,
                coil_dofs=coil_dofs,
                baseline_x=baseline_x,
                baseline_linear_solve_factors=baseline_linear_solve_factors,
                linearization_kind=linearization_kind,
                linear_solve_tol=linear_solve_tol,
                linear_solve_stab=linear_solve_stab,
                optimize_G=optimize_G,
                baseline_coil_dofs=baseline_coil_dofs,
                predictor_kind=predictor_kind,
                objective_kwargs=objective_kwargs,
                success_filter=success_filter,
            )
        return _traceable_forward_result(
            booz_jax,
            coil_set_spec_from_dofs,
            coil_dofs=coil_dofs,
            baseline_x=baseline_x,
            baseline_value=_as_jax_float64(baseline_value),
            baseline_linear_solve_factors=baseline_linear_solve_factors,
            linearization_kind=linearization_kind,
            linear_solve_tol=linear_solve_tol,
            linear_solve_stab=linear_solve_stab,
            optimize_G=optimize_G,
            baseline_coil_dofs=baseline_coil_dofs,
            predictor_kind=predictor_kind,
            objective_kwargs=objective_kwargs,
            success_filter=success_filter,
        )

    jitted_forward_result_for = jax.jit(_forward_result_for)

    def _total_gradient_for(coil_dofs, solved_x, solved_linear_solve_factors):
        return _traceable_total_gradient_with_status(
            booz_jax,
            coil_set_spec_from_dofs,
            coil_dofs=coil_dofs,
            solved_x=solved_x,
            solved_linear_solve_factors=solved_linear_solve_factors,
            linearization_kind=linearization_kind,
            linear_solve_tol=linear_solve_tol,
            linear_solve_stab=linear_solve_stab,
            objective_kwargs=objective_kwargs,
        )

    compiled_total_gradient_for = jax.jit(_total_gradient_for)

    def _value_and_grad_for(coil_dofs):
        result = jitted_forward_result_for(coil_dofs)

        def _success(_):
            return compiled_total_gradient_for(
                coil_dofs,
                result["x"],
                result["linear_solve_factors"],
            )

        grad, linear_solve_success = jax.lax.cond(
            result["success"],
            _success,
            lambda _: (
                _traceable_adjoint_fail_gradient_like(coil_dofs),
                _runtime_bool(False),
            ),
            operand=None,
        )
        return result["value"], _traceable_adjoint_gradient_or_nan(
            grad,
            linear_solve_success,
        )

    jitted_value_and_grad_for = jax.jit(_value_and_grad_for)
    compiled_forward_result_for = _make_traceable_runtime_jax_array_boundary(
        jitted_forward_result_for,
        "compiled_forward_result_for",
    )
    compiled_value_and_grad_for = _mark_cacheable_jit_value_and_grad(
        _make_traceable_runtime_jax_array_boundary(
            jitted_value_and_grad_for,
            "compiled_value_and_grad_for",
        )
    )

    return {
        "state": state,
        "compiled_forward_result_for": compiled_forward_result_for,
        "compiled_total_gradient_for": compiled_total_gradient_for,
        "compiled_value_and_grad_for": compiled_value_and_grad_for,
    }


def _traceable_runtime_option_signature(booz_jax):
    """Capture the solver options that affect traceable runtime compilation."""
    option_state = {
        key: booz_jax.options.get(key) for key in _TRACEABLE_RUNTIME_OPTION_KEYS
    }
    option_state["optimizer_options"] = booz_jax._collect_optimizer_options()
    return _traceable_cache_tree_signature(option_state)


def _traceable_success_filter_signature(success_filter):
    """Return the runtime-cache signature for one optional success filter."""

    if success_filter is None:
        return None
    signature = getattr(success_filter, "_traceable_runtime_cache_signature", None)
    if signature is not None:
        return ("structural", signature)
    return ("callable", id(success_filter))


def _traceable_runtime_cache_key(booz_jax, bs_jax, state, *, success_filter=None):
    """Return a stable cache key for one compiled traceable runtime state.

    The expensive solved baseline arrays are represented by the active Boozer
    solve generation instead of value-hashing their full contents on every
    lookup. This keeps repeated warm-start runtime-bundle construction from
    spending minutes in CPU-side array hashing before the target lane even
    starts compiling or running.
    """
    objective_kwargs = state["objective_kwargs"]
    return (
        # Object identity is part of the contract: callers must rebuild the
        # wrapper instead of mutating booz_jax/bs_jax in place; solved-state
        # freshness is represented separately by _solver_generation.
        id(booz_jax),
        id(bs_jax),
        getattr(booz_jax, "_solver_generation", None),
        state["optimize_G"],
        state["predictor_kind"],
        _traceable_contract_tree_signature(objective_kwargs),
        _traceable_runtime_option_signature(booz_jax),
        _traceable_success_filter_signature(success_filter),
    )


def _get_cached_traceable_runtime_entry(
    booz_jax,
    bs_jax,
    iota_target,
    *,
    outer_objective_config=None,
    success_filter=None,
):
    """Reuse compiled traceable runtime callables while the solved state is unchanged."""
    state = _build_traceable_objective_state(
        booz_jax,
        bs_jax,
        iota_target,
        outer_objective_config=outer_objective_config,
    )
    cache_key = _traceable_runtime_cache_key(
        booz_jax,
        bs_jax,
        state,
        success_filter=success_filter,
    )
    cached_entry = getattr(booz_jax, "_traceable_runtime_entry_cache", None)
    if cached_entry is not None and cached_entry["cache_key"] == cache_key:
        return cached_entry

    compiled_bundle = _build_traceable_objective_compiled_bundle_from_state(
        booz_jax,
        state,
        success_filter=success_filter,
    )
    objective = _make_traceable_objective_from_compiled_bundle(compiled_bundle)
    cached_entry = {
        "cache_key": cache_key,
        "success_filter": success_filter,
        "compiled_bundle": compiled_bundle,
        "objective": objective,
        "batched_value_and_grad": _make_traceable_batched_value_and_grad_pipeline(
            compiled_bundle["compiled_value_and_grad_for"]
        ),
        "reporting_metrics": None,
        "public_objective": None,
        "public_value_and_grad": None,
        "public_batched_value_and_grad": None,
        "public_forward_result": None,
        "public_reporting_metrics": None,
        "host_objective": None,
        "host_value_and_grad": None,
        "host_reporting_metrics": None,
        "profile_suite": None,
        "seeded_compiled_bundle": None,
        "seeded_value_and_grad": None,
        "alm_runtime_bundles": {},
    }
    booz_jax._traceable_runtime_entry_cache = cached_entry
    return cached_entry


def _ensure_traceable_runtime_reporting_metrics(runtime_entry):
    """Materialize the pure reporting-metrics selector on demand."""
    if runtime_entry["reporting_metrics"] is None:
        runtime_entry["reporting_metrics"] = _make_traceable_reporting_metrics_bundle(
            runtime_entry["compiled_bundle"]
        )
    return runtime_entry


def _make_traceable_lazy_reporting_metrics_boundary(runtime_entry):
    """Build a public reporting-metrics boundary that resolves lazily."""

    def reporting_metrics_for(coil_dofs, *, include_distance_metrics=True):
        reporting_metrics = _ensure_traceable_runtime_reporting_metrics(runtime_entry)[
            "reporting_metrics"
        ]
        return reporting_metrics(
            _as_jax_float64(coil_dofs),
            include_distance_metrics=include_distance_metrics,
        )

    return reporting_metrics_for


def _ensure_traceable_runtime_public_boundaries(runtime_entry):
    """Materialize stable public runtime-bundle boundaries on demand."""
    if runtime_entry["public_objective"] is None:
        runtime_entry["public_objective"] = _make_traceable_objective_boundary(
            runtime_entry["objective"]
        )
    if runtime_entry["public_value_and_grad"] is None:
        runtime_entry["public_value_and_grad"] = (
            _make_traceable_value_and_grad_boundary(
                runtime_entry["compiled_bundle"]["compiled_value_and_grad_for"]
            )
        )
    if runtime_entry["public_batched_value_and_grad"] is None:
        runtime_entry["public_batched_value_and_grad"] = (
            _make_traceable_batched_value_and_grad_boundary(
                runtime_entry["batched_value_and_grad"]
            )
        )
    if runtime_entry["public_forward_result"] is None:
        runtime_entry["public_forward_result"] = (
            _make_traceable_forward_result_boundary(
                runtime_entry["compiled_bundle"]["compiled_forward_result_for"],
            )
        )
    if runtime_entry["public_reporting_metrics"] is None:
        runtime_entry["public_reporting_metrics"] = (
            _make_traceable_lazy_reporting_metrics_boundary(runtime_entry)
        )
    return runtime_entry


def _ensure_traceable_runtime_seeded_value_and_grad(runtime_entry, booz_jax):
    """Materialize the seeded optimizer-only value/grad path on demand."""
    seeded_value_and_grad = runtime_entry.get("seeded_value_and_grad")
    if seeded_value_and_grad is not None:
        return seeded_value_and_grad

    state = runtime_entry["compiled_bundle"]["state"]
    seeded_compiled_bundle = _build_traceable_objective_compiled_bundle_from_state(
        booz_jax,
        state,
        success_filter=runtime_entry.get("success_filter"),
        general_only_forward=True,
    )
    baseline_coil_dofs = _traceable_runtime_deviceify_tree(state["baseline_coil_dofs"])
    baseline_x = _traceable_runtime_deviceify_tree(state["baseline_x"])
    baseline_value = _traceable_runtime_deviceify_tree(state["baseline_value"])
    baseline_linear_solve_factors = _traceable_runtime_deviceify_tree(
        state["baseline_linear_solve_factors"]
    )
    with jax.transfer_guard("allow"):
        baseline_gradient, baseline_linear_solve_success = seeded_compiled_bundle[
            "compiled_total_gradient_for"
        ](
            baseline_coil_dofs,
            baseline_x,
            baseline_linear_solve_factors,
        )
        baseline_gradient = _traceable_adjoint_gradient_or_nan(
            baseline_gradient,
            baseline_linear_solve_success,
        )
    seeded_value_and_grad = TraceableObjectiveSeededValueAndGrad(
        value_and_grad=_make_traceable_value_and_grad_boundary(
            seeded_compiled_bundle["compiled_value_and_grad_for"]
        ),
        optimizer_initial_value_and_grad=(
            baseline_value,
            baseline_gradient,
        ),
    )
    runtime_entry["seeded_compiled_bundle"] = seeded_compiled_bundle
    runtime_entry["seeded_value_and_grad"] = seeded_value_and_grad
    return seeded_value_and_grad


def _make_traceable_lazy_host_reporting_metrics(runtime_entry):
    """Build a host-normalized reporting wrapper that resolves lazily."""
    compiled_bundle = runtime_entry["compiled_bundle"]
    state = compiled_bundle["state"]
    baseline_coil_dofs = np.asarray(state["baseline_coil_dofs"], dtype=np.float64)
    baseline_coil_dofs_jax = _traceable_runtime_deviceify_tree(
        state["baseline_coil_dofs"]
    )
    baseline_x = _traceable_runtime_deviceify_tree(state["baseline_x"])
    baseline_host_metrics = {}
    resolved_host_reporting_metrics = None

    def _baseline_reporting_metrics(*, include_distance_metrics):
        include_distances = bool(include_distance_metrics)
        cached_metrics = baseline_host_metrics.get(include_distances)
        if cached_metrics is None:
            with jax.transfer_guard("allow"):
                cached_metrics = _hostify_traceable_reporting_metrics(
                    _traceable_reporting_metrics_from_solution(
                        state["objective_kwargs"],
                        state["coil_set_spec_from_dofs"],
                        coil_dofs=baseline_coil_dofs_jax,
                        solved_x=baseline_x,
                        solver_success=_runtime_bool(True),
                        optimize_G=bool(state["optimize_G"]),
                        include_distance_metrics=include_distances,
                    ),
                    include_distance_metrics=include_distances,
                )
            baseline_host_metrics[include_distances] = cached_metrics
        return dict(cached_metrics)

    def host_reporting_metrics(coil_dofs, *, include_distance_metrics=True):
        nonlocal resolved_host_reporting_metrics
        if _host_input_matches_baseline(coil_dofs, baseline_coil_dofs):
            return _baseline_reporting_metrics(
                include_distance_metrics=include_distance_metrics
            )
        if resolved_host_reporting_metrics is None:
            reporting_metrics = _ensure_traceable_runtime_reporting_metrics(
                runtime_entry
            )["reporting_metrics"]
            resolved_host_reporting_metrics = _make_traceable_host_reporting_metrics(
                reporting_metrics
            )
        return resolved_host_reporting_metrics(
            coil_dofs,
            include_distance_metrics=include_distance_metrics,
        )

    return host_reporting_metrics


def _ensure_traceable_runtime_host_wrappers(runtime_entry, booz_jax):
    """Materialize host-boundary wrappers for one cached runtime entry on demand."""
    if (
        runtime_entry["host_objective"] is None
        or runtime_entry["host_value_and_grad"] is None
        or runtime_entry["host_reporting_metrics"] is None
    ):
        compiled_bundle = runtime_entry["compiled_bundle"]
        state = compiled_bundle["state"]
        baseline_coil_dofs = np.asarray(state["baseline_coil_dofs"], dtype=np.float64)
        baseline_value = float(np.asarray(state["baseline_value"], dtype=np.float64))
        runtime_entry["host_objective"] = _make_traceable_host_objective(
            runtime_entry["objective"],
            baseline_coil_dofs=baseline_coil_dofs,
            baseline_return=baseline_value,
        )
        baseline_coil_dofs_jax = _traceable_runtime_deviceify_tree(
            state["baseline_coil_dofs"]
        )
        baseline_x = _traceable_runtime_deviceify_tree(state["baseline_x"])
        baseline_linear_solve_factors = _traceable_runtime_deviceify_tree(
            state["baseline_linear_solve_factors"]
        )
        baseline_value_jax = _traceable_runtime_deviceify_tree(state["baseline_value"])
        with jax.transfer_guard("allow"):
            baseline_gradient, baseline_linear_solve_success = (
                _traceable_total_gradient_with_status(
                    booz_jax,
                    state["coil_set_spec_from_dofs"],
                    coil_dofs=baseline_coil_dofs_jax,
                    solved_x=baseline_x,
                    solved_linear_solve_factors=baseline_linear_solve_factors,
                    linearization_kind=state["linearization_kind"],
                    linear_solve_tol=state["linear_solve_tol"],
                    linear_solve_stab=state["linear_solve_stab"],
                    objective_kwargs=state["objective_kwargs"],
                )
            )
            baseline_gradient = _traceable_adjoint_gradient_or_nan(
                baseline_gradient,
                baseline_linear_solve_success,
            )
            baseline_gradient = _host_array(
                baseline_gradient,
                dtype=np.float64,
            )
            baseline_value_for_value_and_grad = float(
                _host_scalar(baseline_value_jax, dtype=np.float64)
            )
        runtime_entry["host_value_and_grad"] = _make_traceable_host_value_and_grad(
            compiled_bundle["compiled_value_and_grad_for"],
            baseline_coil_dofs=baseline_coil_dofs,
            baseline_return=lambda: (
                baseline_value_for_value_and_grad,
                baseline_gradient.copy(),
            ),
        )
        runtime_entry["host_reporting_metrics"] = (
            _make_traceable_lazy_host_reporting_metrics(runtime_entry)
        )
    return runtime_entry


def _make_traceable_objective_from_compiled_bundle(compiled_bundle):
    """Build the scalar custom-VJP target-lane objective from one compiled bundle."""
    compiled_forward_result_for = compiled_bundle["compiled_forward_result_for"]
    compiled_total_gradient_for = compiled_bundle["compiled_total_gradient_for"]

    @jax.custom_vjp
    def f(coil_dofs):
        coil_dofs = _as_jax_float64(coil_dofs)
        return compiled_forward_result_for(coil_dofs)["value"]

    def f_fwd(coil_dofs):
        coil_dofs = _as_jax_float64(coil_dofs)
        result = compiled_forward_result_for(coil_dofs)
        return result["value"], (
            coil_dofs,
            result["x"],
            result["linear_solve_factors"],
            result["success"],
        )

    def f_bwd(saved_state, cotangent):
        coil_dofs, solved_x, solved_linear_solve_factors, success = saved_state

        def _success(_):
            grad, linear_solve_success = compiled_total_gradient_for(
                coil_dofs,
                solved_x,
                solved_linear_solve_factors,
            )
            return _traceable_adjoint_gradient_or_nan(grad, linear_solve_success)

        def _failure(_):
            return _traceable_adjoint_fail_gradient_like(coil_dofs)

        grad = jax.lax.cond(success, _success, _failure, operand=None)
        return (_as_runtime_float64(cotangent, reference=grad) * grad,)

    f.defvjp(f_fwd, f_bwd)

    # Keep the pure runtime entrypoint on a real JIT boundary so transfer_guard
    # rejects implicit host inputs consistently with the other runtime-bundle
    # callables. Explicit host materialization belongs on the host wrapper.
    return jax.jit(f)


def _host_input_matches_baseline(coil_dofs, baseline_coil_dofs):
    """Return whether a host input exactly matches the cached baseline state."""
    if coil_dofs is baseline_coil_dofs:
        return True
    if not isinstance(
        coil_dofs,
        (np.ndarray, np.generic, list, tuple, float, int),
    ):
        return False
    host_coil_dofs = np.asarray(coil_dofs, dtype=np.float64)
    return host_coil_dofs.shape == baseline_coil_dofs.shape and np.array_equal(
        host_coil_dofs, baseline_coil_dofs
    )


def _host_boundary_with_baseline_peel(
    host_callable,
    baseline_coil_dofs,
    baseline_return,
):
    """Skip the traced host boundary when host inputs equal the solved baseline."""
    baseline_host = np.asarray(baseline_coil_dofs, dtype=np.float64)

    def wrapped(coil_dofs, *args, **kwargs):
        if _host_input_matches_baseline(coil_dofs, baseline_host):
            if callable(baseline_return):
                return baseline_return(*args, **kwargs)
            return baseline_return
        return host_callable(coil_dofs, *args, **kwargs)

    return wrapped


def _make_traceable_host_objective(
    pure_objective,
    *,
    baseline_coil_dofs=None,
    baseline_return=None,
):
    """Build a host-normalized scalar wrapper around the pure JAX objective."""

    def host_objective(coil_dofs):
        return float(
            _host_scalar(
                pure_objective(_as_jax_float64(coil_dofs)),
                dtype=np.float64,
            )
        )

    if baseline_coil_dofs is None:
        return host_objective
    return _host_boundary_with_baseline_peel(
        host_objective,
        baseline_coil_dofs,
        baseline_return,
    )


def _make_traceable_objective_boundary(pure_objective):
    """Build the public pure-JAX scalar entrypoint for one runtime bundle."""

    def objective(coil_dofs):
        return pure_objective(_as_jax_float64(coil_dofs))

    return objective


def _make_traceable_forward_result_boundary(compiled_forward_result_for):
    """Build the public pure-JAX forward-result entrypoint for one runtime bundle."""

    def forward_result(coil_dofs):
        result = compiled_forward_result_for(_as_jax_float64(coil_dofs))
        if "dense_plu" in result:
            return result
        linear_solve_factors = result["linear_solve_factors"]
        return dict(
            result,
            dense_plu=linear_solve_factors,
            linear_solve_backend="operator",
            dense_linear_solve_factors_available=linear_solve_factors is not None,
        )

    return forward_result


def _make_traceable_host_value_and_grad(
    compiled_value_and_grad_for,
    *,
    baseline_coil_dofs=None,
    baseline_return=None,
):
    """Build a host-normalized wrapper around the fused JAX value/grad callable."""

    def host_value_and_grad(coil_dofs):
        value, grad = compiled_value_and_grad_for(_as_jax_float64(coil_dofs))
        return (
            float(_host_scalar(value, dtype=np.float64)),
            _host_array(grad, dtype=np.float64),
        )

    if baseline_coil_dofs is None:
        return host_value_and_grad
    return _host_boundary_with_baseline_peel(
        host_value_and_grad,
        baseline_coil_dofs,
        baseline_return,
    )


def _make_traceable_value_and_grad_boundary(compiled_value_and_grad_for):
    """Build the public pure-JAX value/grad entrypoint for one runtime bundle.

    This is the explicit host-to-device staging seam for callers that still
    hold coil DOFs as NumPy arrays during setup or test harness construction.
    Under JAX transfer-guard ``disallow``, explicit staging is allowed while
    implicit transfers are not, so keep this entrypoint aligned with the scalar
    objective and reporting-metrics boundaries.
    """
    from .optimizer_jax import _mark_cacheable_jit_value_and_grad

    def value_and_grad(coil_dofs):
        return compiled_value_and_grad_for(_as_jax_float64(coil_dofs))

    return _mark_cacheable_jit_value_and_grad(value_and_grad)


def _traceable_reporting_metrics_from_solution(
    objective_kwargs,
    coil_set_spec_from_dofs,
    *,
    coil_dofs,
    solved_x,
    solver_success,
    optimize_G,
    include_distance_metrics,
):
    """Compute reporting metrics for one explicit solved state."""
    outer_objective_config = objective_kwargs["outer_objective_config"]
    if outer_objective_config is None:
        raise RuntimeError(
            "Traceable reporting metrics require the full single-stage outer objective."
        )

    coil_dof_extraction_spec = objective_kwargs["coil_dof_extraction_spec"]
    banana_curve_index = int(outer_objective_config["banana_curve_index"])
    coil_set_spec = coil_set_spec_from_dofs(coil_dofs)
    raw_terms = _traceable_single_stage_outer_term_values(
        solved_x,
        coil_dofs,
        coil_set_spec,
        **_traceable_total_objective_kwargs(objective_kwargs),
    )
    weighted_terms = _traceable_weighted_single_stage_outer_term_values(
        raw_terms,
        outer_objective_config=outer_objective_config,
    )
    sdofs, iota, G = _split_x_inner_runtime(solved_x, optimize_G)
    surface_gamma, xphi, xtheta = _surface_geometry_from_dofs(
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
    surface_normal = jnp.cross(xphi, xtheta)
    nphi, ntheta = surface_gamma.shape[:2]
    surface_points = surface_gamma.reshape(-1, 3)
    surface_B = grouped_biot_savart_B_from_spec(
        surface_points,
        coil_set_spec,
    ).reshape(nphi, ntheta, 3)
    surface_normal_norm = jnp.sqrt(jnp.sum(surface_normal * surface_normal, axis=-1))
    surface_unit_normal = surface_normal / surface_normal_norm[:, :, None]
    surface_B_normal = jnp.sum(surface_B * surface_unit_normal, axis=-1)
    surface_B_norm = jnp.sqrt(jnp.sum(surface_B * surface_B, axis=-1))
    surface_area = surface_normal_norm / surface_normal_norm.size
    field_error = (
        jnp.sum(jnp.abs(surface_B_normal / surface_B_norm) * surface_area)
        / jnp.sum(surface_area)
    )
    coil_specs = coil_specs_from_dof_extraction_spec(
        coil_dof_extraction_spec,
        coil_dofs,
    )
    banana_curve_spec = coil_specs[banana_curve_index].curve
    banana_current = jnp.abs(
        _take_runtime_scalar(coil_specs[banana_curve_index].current.value, 0)
    )
    _gamma, banana_gammadash, banana_gammadashdash = curve_geometry_from_spec(
        banana_curve_spec
    )
    coil_length = curve_length_pure(incremental_arclength_pure(banana_gammadash))
    max_curvature = jnp.max(kappa_pure(banana_gammadash, banana_gammadashdash))
    inf = _runtime_float64_scalar(np.inf, reference=surface_gamma)
    curve_curve_min_dist = inf
    curve_surface_min_dist = inf
    surface_vessel_min_dist = inf
    if include_distance_metrics:
        vessel_gamma = _runtime_float64_array(
            outer_objective_config["vessel_gamma"],
            reference=surface_gamma,
        ).reshape((-1, 3))
        surface_gamma_flat = surface_gamma.reshape((-1, 3))
        coil_gammas = []
        for group in coil_set_spec.groups:
            gammas = _as_jax_float64(group.gammas)
            for coil_index in range(int(gammas.shape[0])):
                coil_gamma = _take_runtime_row(gammas, coil_index)
                coil_gammas.append(coil_gamma)
                curve_surface_min_dist = jnp.minimum(
                    curve_surface_min_dist,
                    pairwise_min_distance_pure(coil_gamma, surface_gamma_flat),
                )
        for curve_index, gamma_i in enumerate(coil_gammas):
            for gamma_j in coil_gammas[:curve_index]:
                curve_curve_min_dist = jnp.minimum(
                    curve_curve_min_dist,
                    pairwise_min_distance_pure(gamma_i, gamma_j),
                )
        surface_vessel_min_dist = surface_to_surface_shortest_distance_pure(
            surface_gamma,
            vessel_gamma,
        )
    return {
        "solver_success": solver_success,
        "has_G": jnp.asarray(optimize_G, dtype=bool),
        "final_G": G if G is not None else _runtime_float64_scalar(0.0, reference=iota),
        "final_non_qs": weighted_terms["non_qs"],
        "final_boozer_residual": weighted_terms["residual"],
        "final_iota_penalty": weighted_terms["iota"],
        "final_length_penalty": weighted_terms["length"],
        "final_curve_curve_penalty": weighted_terms["curve_curve"],
        "final_curve_surface_penalty": weighted_terms["curve_surface"],
        "final_surface_vessel_penalty": weighted_terms["surface_vessel"],
        "final_curvature_penalty": weighted_terms["curvature"],
        "coil_length": coil_length,
        "max_curvature": max_curvature,
        "banana_current_A": banana_current,
        "field_error": field_error,
        "curve_curve_min_dist": curve_curve_min_dist,
        "curve_surface_min_dist": curve_surface_min_dist,
        "surface_vessel_min_dist": surface_vessel_min_dist,
        "final_volume": surface_volume(surface_gamma, surface_normal),
        "final_iota": iota,
    }


def _make_traceable_reporting_metrics(compiled_bundle, *, include_distance_metrics):
    """Build a pure solved-state reporting summary for one compiled runtime bundle."""
    compiled_forward_result_for = compiled_bundle["compiled_forward_result_for"]
    state = compiled_bundle["state"]
    objective_kwargs = state["objective_kwargs"]
    optimize_G = bool(state["optimize_G"])
    coil_set_spec_from_dofs = state["coil_set_spec_from_dofs"]

    def reporting_metrics(coil_dofs):
        coil_dofs = _as_jax_float64(coil_dofs)
        forward_result = compiled_forward_result_for(coil_dofs)
        return _traceable_reporting_metrics_from_solution(
            objective_kwargs,
            coil_set_spec_from_dofs,
            coil_dofs=coil_dofs,
            solved_x=forward_result["x"],
            solver_success=forward_result["success"],
            optimize_G=optimize_G,
            include_distance_metrics=include_distance_metrics,
        )

    return jax.jit(reporting_metrics)


def _make_traceable_reporting_metrics_bundle(compiled_bundle):
    """Build the pure reporting-metrics selector for one compiled bundle."""
    reporting_metrics = _make_traceable_reporting_metrics(
        compiled_bundle,
        include_distance_metrics=True,
    )
    reporting_metrics_without_distances = _make_traceable_reporting_metrics(
        compiled_bundle,
        include_distance_metrics=False,
    )

    def reporting_metrics_for(coil_dofs, *, include_distance_metrics=True):
        selected_reporting_metrics = (
            reporting_metrics
            if include_distance_metrics
            else reporting_metrics_without_distances
        )
        return selected_reporting_metrics(coil_dofs)

    return reporting_metrics_for


def _hostify_traceable_reporting_metrics(metrics, *, include_distance_metrics):
    """Materialize one traceable reporting-metrics dict on the host."""
    float_metric_names = (
        "final_non_qs",
        "final_boozer_residual",
        "final_iota_penalty",
        "final_length_penalty",
        "final_curve_curve_penalty",
        "final_curve_surface_penalty",
        "final_surface_vessel_penalty",
        "final_curvature_penalty",
        "coil_length",
        "max_curvature",
        "banana_current_A",
        "field_error",
        "final_volume",
        "final_iota",
    )
    distance_metric_names = (
        "curve_curve_min_dist",
        "curve_surface_min_dist",
        "surface_vessel_min_dist",
    )
    has_G = bool(np.asarray(jax.device_get(metrics["has_G"])))
    host_metrics = {
        "solver_success": bool(np.asarray(jax.device_get(metrics["solver_success"]))),
        "final_G": None
        if not has_G
        else float(_host_scalar(metrics["final_G"], dtype=np.float64)),
    }
    for metric_name in float_metric_names:
        host_metrics[metric_name] = float(
            _host_scalar(metrics[metric_name], dtype=np.float64)
        )
    for metric_name in distance_metric_names:
        host_metrics[metric_name] = (
            None
            if not include_distance_metrics
            else float(_host_scalar(metrics[metric_name], dtype=np.float64))
        )
    return host_metrics


def _make_traceable_host_reporting_metrics(reporting_metrics):
    """Build a host-normalized solved-state reporting summary wrapper."""

    def host_reporting_metrics(coil_dofs, *, include_distance_metrics=True):
        metrics = reporting_metrics(
            _as_jax_float64(coil_dofs),
            include_distance_metrics=include_distance_metrics,
        )
        return _hostify_traceable_reporting_metrics(
            metrics,
            include_distance_metrics=include_distance_metrics,
        )

    return host_reporting_metrics


def _make_traceable_batched_value_and_grad_pipeline(compiled_value_and_grad_for):
    """Build a scalar-equivalent batched ``(value, grad)`` pipeline for seed scoring."""
    from .optimizer_jax import _mark_cacheable_jit_value_and_grad

    def _batched_value_and_grad_for(coil_dofs_batch):
        coil_dofs_batch = _as_jax_float64(coil_dofs_batch)
        return lax.map(compiled_value_and_grad_for, coil_dofs_batch)

    return _mark_cacheable_jit_value_and_grad(jax.jit(_batched_value_and_grad_for))


def _make_traceable_batched_value_and_grad_boundary(batched_value_and_grad):
    """Build the public pure-JAX batched value/grad entrypoint."""

    def batched_value_and_grad_for(coil_dofs_batch):
        return batched_value_and_grad(_as_jax_float64(coil_dofs_batch))

    return batched_value_and_grad_for


def _classify_nonfinite_scalar(host_value):
    """Classify one non-finite scalar for compact diagnostics."""
    if np.isnan(host_value):
        return "nan"
    if np.isposinf(host_value):
        return "+inf"
    if np.isneginf(host_value):
        return "-inf"
    return None


def _summarize_traceable_scalar(value):
    """Return a compact host summary for one scalar JAX value."""
    host_value = float(_host_scalar(value, dtype=np.float64))
    finite = bool(np.isfinite(host_value))
    return {
        "value": host_value if finite else None,
        "finite": finite,
        "classification": None if finite else _classify_nonfinite_scalar(host_value),
    }


def _summarize_traceable_gradient(gradient):
    """Return a compact host summary for one gradient vector."""
    host_gradient = np.asarray(jax.device_get(gradient), dtype=np.float64).reshape(-1)
    finite_mask = np.isfinite(host_gradient)
    all_finite = bool(np.all(finite_mask))
    first_nonfinite_index = None
    if not all_finite:
        first_nonfinite_index = int(np.flatnonzero(~finite_mask)[0])
    return {
        "all_finite": all_finite,
        "inf_norm": float(_host_inf_norm(gradient)) if all_finite else None,
        "size": int(host_gradient.size),
        "nonfinite_count": int(host_gradient.size - int(np.count_nonzero(finite_mask))),
        "first_nonfinite_index": first_nonfinite_index,
    }


def _traceable_term_adjoint_solve_report(
    booz_jax,
    coil_set_spec_from_dofs,
    *,
    coil_dofs,
    solved_x,
    solved_linear_solve_factors,
    linearization_kind,
    linear_solve_tol,
    linear_solve_stab,
    objective_kwargs,
    term_name,
):
    depends_on_x_inner, _ = _traceable_single_stage_effective_dependency_flags(
        term_name,
        objective_kwargs=objective_kwargs,
    )
    if not depends_on_x_inner:
        return None

    coil_set_spec = coil_set_spec_from_dofs(coil_dofs)

    def objective_of_x(current_x):
        return _evaluate_traceable_weighted_single_stage_outer_term(
            term_name,
            current_x,
            coil_dofs,
            coil_set_spec,
            objective_kwargs,
        )

    rhs = _strict_scalar_grad(objective_of_x, solved_x)
    adjoint, success = _traceable_solve_linearization(
        booz_jax,
        solved_x,
        rhs,
        coil_set_spec,
        objective_kwargs,
        linear_solve_factors=solved_linear_solve_factors,
        linearization_kind=linearization_kind,
        linear_solve_tol=linear_solve_tol,
        linear_solve_stab=linear_solve_stab,
        transpose=True,
    )
    report = {
        "success": bool(np.asarray(jax.device_get(success))),
        "rhs_norm": _summarize_traceable_scalar(jnp.linalg.norm(rhs)),
        "solution_norm": _summarize_traceable_scalar(jnp.linalg.norm(adjoint)),
        "solution": _summarize_traceable_gradient(adjoint),
    }
    if solved_linear_solve_factors is not None:
        residual = rhs - _traceable_plu_matvec(
            solved_linear_solve_factors,
            adjoint,
            transpose=True,
        )
        residual_norm = jnp.linalg.norm(residual)
        base_residual_tol = _optimizer_jax._linear_solve_residual_tolerance(
            rhs,
            linear_solve_tol,
        )
        residual_tol = _traceable_plu_residual_tolerance(
            solved_linear_solve_factors,
            adjoint,
            rhs,
            base_residual_tol,
        )
        matrix = _traceable_plu_matrix(solved_linear_solve_factors)
        report["plu"] = {
            "matrix_norm": _summarize_traceable_scalar(jnp.linalg.norm(matrix)),
            "base_residual_tolerance": _summarize_traceable_scalar(
                base_residual_tol
            ),
            "residual_tolerance": _summarize_traceable_scalar(residual_tol),
            "residual_norm": _summarize_traceable_scalar(residual_norm),
            "relative_residual": _summarize_traceable_scalar(
                residual_norm / jnp.linalg.norm(rhs)
            ),
        }
    elif linearization_kind == "hessian":
        objective_fn = _make_boozer_penalty_objective_closure(
            coil_set_spec=coil_set_spec,
            **_traceable_inner_objective_kwargs(objective_kwargs),
        )
        hvp_fn = _optimizer_jax._hessian_vector_product_fn(objective_fn)
        attempts = []
        for candidate_stab in booz_jax._adjoint_hessian_stabilization_schedule():
            solution, attempt_success = (
                _optimizer_jax._solve_hessian_system_with_status(
                    objective_fn,
                    solved_x,
                    rhs,
                    stab=float(candidate_stab),
                    tol=linear_solve_tol,
                )
            )
            residual_tol = _optimizer_jax._linear_solve_residual_tolerance(
                rhs,
                linear_solve_tol,
            )
            attempt = {
                "stab": float(candidate_stab),
                "success": bool(np.asarray(jax.device_get(attempt_success))),
                "solution": _summarize_traceable_gradient(solution),
                "solution_norm": _summarize_traceable_scalar(
                    jnp.linalg.norm(solution)
                ),
                "residual_tolerance": _summarize_traceable_scalar(residual_tol),
            }
            if float(candidate_stab) == 0.0:
                residual = rhs - hvp_fn(solved_x, solution)
                residual_norm = jnp.linalg.norm(residual)
                rhs_norm = jnp.linalg.norm(rhs)
                attempt["residual_norm"] = _summarize_traceable_scalar(
                    residual_norm
                )
                attempt["relative_residual"] = _summarize_traceable_scalar(
                    residual_norm / rhs_norm
                )
            attempts.append(attempt)
        report["hessian_operator"] = {"attempts": attempts}
    return report


def diagnose_traceable_objective_runtime(
    booz_jax,
    bs_jax,
    iota_target,
    *,
    outer_objective_config=None,
    success_filter=None,
):
    """Return a compact baseline diagnostic report for the target-lane runtime."""
    _traceable_diag_progress("resolve_runtime_entry")
    runtime_entry = _get_cached_traceable_runtime_entry(
        booz_jax,
        bs_jax,
        iota_target,
        outer_objective_config=outer_objective_config,
        success_filter=success_filter,
    )
    compiled_bundle = runtime_entry["compiled_bundle"]
    state = compiled_bundle["state"]
    objective_kwargs = state["objective_kwargs"]
    if objective_kwargs["outer_objective_config"] is None:
        raise RuntimeError(
            "Traceable runtime diagnosis requires the full single-stage outer objective."
        )

    _traceable_diag_progress("deviceify_baseline_state")
    baseline_coil_dofs = _traceable_runtime_deviceify_tree(state["baseline_coil_dofs"])
    baseline_x = _traceable_runtime_deviceify_tree(state["baseline_x"])
    baseline_value = _traceable_runtime_deviceify_tree(state["baseline_value"])
    baseline_linear_solve_factors = _traceable_runtime_deviceify_tree(
        state["baseline_linear_solve_factors"]
    )
    coil_set_spec_from_dofs = state["coil_set_spec_from_dofs"]
    baseline_coil_set_spec = coil_set_spec_from_dofs(baseline_coil_dofs)
    optimize_G = bool(state["optimize_G"])
    baseline_sdofs, baseline_iota, baseline_G = _split_x_inner_runtime(
        baseline_x,
        optimize_G,
    )
    baseline_success = _traceable_runtime_deviceify_tree(
        np.asarray(True, dtype=bool)
    )
    # The gradient diagnosis always evaluates the cached solved baseline. Peel
    # that state directly here so this host-side diagnostic does not spend
    # minutes compiling the full coil-dependent forward-result JIT before it
    # even reaches the actual baseline objective/gradient checks.
    forward_result = _pack_traceable_forward_result(
        value=baseline_value,
        x=baseline_x,
        sdofs=baseline_sdofs,
        iota=baseline_iota,
        G=baseline_G,
        linear_solve_factors=baseline_linear_solve_factors,
        success=baseline_success,
        primal_success=baseline_success,
        adjoint_linear_solve_available=baseline_success,
    )
    _traceable_diag_progress("baseline_total_gradient")
    total_value = baseline_value
    total_gradient, total_linear_solve_success = _traceable_total_gradient_with_status(
        booz_jax,
        coil_set_spec_from_dofs,
        coil_dofs=baseline_coil_dofs,
        solved_x=baseline_x,
        solved_linear_solve_factors=baseline_linear_solve_factors,
        linearization_kind=state["linearization_kind"],
        linear_solve_tol=state["linear_solve_tol"],
        linear_solve_stab=state["linear_solve_stab"],
        objective_kwargs=objective_kwargs,
    )
    del total_linear_solve_success
    _traceable_diag_progress("raw_term_values")
    raw_terms = _traceable_single_stage_outer_term_values(
        baseline_x,
        baseline_coil_dofs,
        baseline_coil_set_spec,
        **_traceable_total_objective_kwargs(objective_kwargs),
    )
    weighted_terms = _traceable_weighted_single_stage_outer_term_values(
        raw_terms,
        outer_objective_config=objective_kwargs["outer_objective_config"],
    )
    report = {
        "baseline_success": bool(np.asarray(jax.device_get(forward_result["success"]))),
        "total": {
            "value": _summarize_traceable_scalar(total_value),
            "grad": _summarize_traceable_gradient(total_gradient),
        },
        "terms": {},
    }
    nonfinite_terms = []
    for term_name, weight_key in _TRACEABLE_SINGLE_STAGE_OUTER_TERM_SPECS:
        _traceable_diag_progress(f"term_gradient:{term_name}")
        direct_grad, implicit_grad, term_total_grad, linear_solve_success = (
            _traceable_objective_gradient_parts(
                booz_jax,
                coil_set_spec_from_dofs,
                coil_dofs=baseline_coil_dofs,
                solved_x=baseline_x,
                solved_linear_solve_factors=baseline_linear_solve_factors,
                linearization_kind=state["linearization_kind"],
                linear_solve_tol=state["linear_solve_tol"],
                linear_solve_stab=state["linear_solve_stab"],
                objective_kwargs=objective_kwargs,
                term_name=term_name,
            )
        )
        term_report = {
            "weight": float(
                objective_kwargs["outer_objective_config"].get(weight_key, 0.0)
            ),
            "raw_value": _summarize_traceable_scalar(raw_terms[term_name]),
            "weighted_value": _summarize_traceable_scalar(weighted_terms[term_name]),
            "direct_grad": _summarize_traceable_gradient(direct_grad),
            "implicit_grad": _summarize_traceable_gradient(implicit_grad),
            "total_grad": _summarize_traceable_gradient(term_total_grad),
            "linear_solve_success": bool(np.asarray(jax.device_get(linear_solve_success))),
        }
        issues = []
        if not term_report["raw_value"]["finite"]:
            issues.append("raw_value")
        if not term_report["weighted_value"]["finite"]:
            issues.append("weighted_value")
        if not term_report["direct_grad"]["all_finite"]:
            issues.append("direct_grad")
        if not term_report["implicit_grad"]["all_finite"]:
            issues.append("implicit_grad")
        if not term_report["total_grad"]["all_finite"]:
            issues.append("total_grad")
        if not term_report["linear_solve_success"]:
            term_report["adjoint_solve"] = _traceable_term_adjoint_solve_report(
                booz_jax,
                coil_set_spec_from_dofs,
                coil_dofs=baseline_coil_dofs,
                solved_x=baseline_x,
                solved_linear_solve_factors=baseline_linear_solve_factors,
                linearization_kind=state["linearization_kind"],
                linear_solve_tol=state["linear_solve_tol"],
                linear_solve_stab=state["linear_solve_stab"],
                objective_kwargs=objective_kwargs,
                term_name=term_name,
            )
        term_report["issues"] = issues
        report["terms"][term_name] = term_report
        if issues:
            nonfinite_terms.append(term_name)
    report["nonfinite_terms"] = nonfinite_terms
    report["first_nonfinite_term"] = nonfinite_terms[0] if nonfinite_terms else None
    report["all_finite"] = bool(
        report["total"]["value"]["finite"]
        and report["total"]["grad"]["all_finite"]
        and not nonfinite_terms
    )
    _traceable_diag_progress("report_complete")
    return report


def make_traceable_objective(
    booz_jax,
    bs_jax,
    iota_target,
    *,
    outer_objective_config=None,
    success_filter=None,
):
    """Build a pure function ``f(coil_dofs) -> scalar`` for single-stage optimization.

    The returned closure:

    * **Forward**: re-solves the inner Boozer problem from a coil-dependent
      linearized warm-start predictor and returns the exact
      single-stage scalar objective
      ``BoozerResidualJAX + 0.5 * (iota - iota_target)^2``.
    * **No object mutation**: coil geometry is reconstructed directly from
      the explicit ``coil_dofs`` vector, so the traced objective does not
      touch ``bs_jax.x``, ``booz_jax.res``, or descendant Optimizable caches.
    * **No callback seam**: the traced path stays inside JAX primitives;
      there is no ``jax.pure_callback`` bridge back into the stateful
      ``run_code()`` implementation.
    * **Backward**: uses the same implicit-differentiation structure as the
      validated object path, but expressed entirely with pure JAX arrays.

    Args:
        booz_jax: solved :class:`BoozerSurfaceJAX`.
        bs_jax:   :class:`BiotSavartJAX` providing coil geometry.
        iota_target: scalar target iota for the quadratic penalty.
        outer_objective_config: optional structured config enabling the full
            single-stage outer objective. When omitted, the historical traced
            objective remains ``BoozerResidualJAX + 0.5 * (iota-iota_target)^2``.

    Returns:
        ``f(coil_dofs) -> jax.Array`` — traceable scalar objective.

        This is the pure-JAX optimizer contract used by the single-stage
        ondevice lane. Callers that need Python/NumPy materialization should
        use :func:`make_traceable_objective_runtime_bundle` with
        ``include_host_wrappers=True`` and the returned ``host_objective`` /
        ``host_value_and_grad`` wrappers instead of coercing this traced scalar
        directly.
    """
    return _get_cached_traceable_runtime_entry(
        booz_jax,
        bs_jax,
        iota_target,
        outer_objective_config=outer_objective_config,
        success_filter=success_filter,
    )["objective"]


class TraceableObjectiveSeededValueAndGrad(NamedTuple):
    """Optimizer-only seeded value/gradient contract for target-lane L-BFGS."""

    value_and_grad: callable
    optimizer_initial_value_and_grad: tuple[jax.Array, jax.Array]


def make_traceable_objective_seeded_value_and_grad(
    booz_jax,
    bs_jax,
    iota_target,
    *,
    outer_objective_config=None,
    success_filter=None,
):
    """Build the optimizer-only seeded value/gradient lane for ondevice L-BFGS.

    Unlike :func:`make_traceable_objective_runtime_bundle`, this helper keeps the
    seeded optimizer contract separate from the public runtime-bundle dict so the
    hot value/gradient path can lower through the general forward helper only,
    while callers still reuse the cached baseline value/gradient seed for the
    first optimizer evaluation.
    """
    runtime_entry = _get_cached_traceable_runtime_entry(
        booz_jax,
        bs_jax,
        iota_target,
        outer_objective_config=outer_objective_config,
        success_filter=success_filter,
    )
    return _ensure_traceable_runtime_seeded_value_and_grad(runtime_entry, booz_jax)


def make_traceable_objective_value_and_grad(
    booz_jax,
    bs_jax,
    iota_target,
    *,
    outer_objective_config=None,
    success_filter=None,
):
    """Build a pure-JAX function ``f(coil_dofs) -> (value, grad)`` for ondevice L-BFGS.

    This is the fused outer-optimizer objective contract for the single-stage
    ondevice target lane. It shares the exact forward and implicit-gradient
    implementation used by :func:`make_traceable_objective`, but returns both
    outputs from one compiled entrypoint so the outer optimizer can avoid
    rebuilding autodiff transforms around a scalar objective.

    For host-normalized outputs, use
    ``make_traceable_objective_runtime_bundle(include_host_wrappers=True)``
    and call ``runtime_bundle["host_value_and_grad"]``.
    """
    return make_traceable_objective_runtime_bundle(
        booz_jax,
        bs_jax,
        iota_target,
        outer_objective_config=outer_objective_config,
        success_filter=success_filter,
    )["value_and_grad"]


def _make_traceable_forward_value_pipeline(compiled_forward_result_for):
    def _forward_value_for(coil_dofs):
        return compiled_forward_result_for(coil_dofs)["value"]

    return jax.jit(_forward_value_for)


def _make_traceable_field_eval_sharding_pipeline(field_at_solution_for):
    compiled_field_at_solution_for = jax.jit(field_at_solution_for)

    def _field_eval_sharding(coil_dofs):
        return inspect_array_sharding_summary(compiled_field_at_solution_for(coil_dofs))

    return _field_eval_sharding


def _make_traceable_objective_profile_suite_from_compiled_bundle(
    compiled_bundle,
    booz_jax,
    bs_jax,
    *,
    value_and_grad_pipeline=None,
    batched_value_and_grad_pipeline=None,
):
    """Build profiling closures from the shared traceable runtime bundle."""
    state = compiled_bundle["state"]
    objective_kwargs = state["objective_kwargs"]
    baseline_coil_dofs = state["baseline_coil_dofs"]
    baseline_x = state["baseline_x"]
    baseline_linear_solve_factors = state["baseline_linear_solve_factors"]
    optimize_G = state["optimize_G"]
    predictor_kind = state["predictor_kind"]
    linearization_kind = state["linearization_kind"]
    linear_solve_tol = state["linear_solve_tol"]
    linear_solve_stab = state["linear_solve_stab"]
    coil_set_spec_from_dofs = state["coil_set_spec_from_dofs"]
    compiled_forward_result_for = compiled_bundle["compiled_forward_result_for"]
    resolved_value_and_grad_pipeline = (
        compiled_bundle["compiled_value_and_grad_for"]
        if value_and_grad_pipeline is None
        else value_and_grad_pipeline
    )
    resolved_batched_value_and_grad_pipeline = (
        _make_traceable_batched_value_and_grad_pipeline(
            compiled_bundle["compiled_value_and_grad_for"]
        )
        if batched_value_and_grad_pipeline is None
        else batched_value_and_grad_pipeline
    )

    def _warmstart_for(coil_dofs):
        warmstart_x, warmstart_linear_solve_success = _traceable_predict_warmstart_x(
            booz_jax,
            coil_set_spec_from_dofs,
            coil_dofs=coil_dofs,
            baseline_coil_dofs=baseline_coil_dofs,
            baseline_x=baseline_x,
            baseline_linear_solve_factors=baseline_linear_solve_factors,
            linearization_kind=linearization_kind,
            linear_solve_tol=linear_solve_tol,
            linear_solve_stab=linear_solve_stab,
            predictor_kind=predictor_kind,
            objective_kwargs=objective_kwargs,
        )
        return {
            "x": warmstart_x,
            "success": warmstart_linear_solve_success,
        }

    def _solve_for(coil_dofs):
        coil_set_spec = coil_set_spec_from_dofs(coil_dofs)
        warmstart = _warmstart_for(coil_dofs)
        warmstart_x = warmstart["x"]
        warmstart_linear_solve_success = warmstart["success"]

        def _run_traceable_solve(_):
            warmstart_sdofs, warmstart_iota, warmstart_G = (
                booz_jax._unpack_decision_vector_jax(
                    warmstart_x,
                    optimize_G,
                    coil_set_spec=coil_set_spec,
                )
            )
            solve_result = booz_jax.run_code_traceable(
                coil_set_spec,
                warmstart_sdofs,
                warmstart_iota,
                warmstart_G,
            )
            solved_sdofs, solved_iota, solved_G = _resolve_traceable_solved_state(
                booz_jax,
                solve_result,
                optimize_G=optimize_G,
                coil_set_spec=coil_set_spec,
            )
            return {
                "x": solve_result["x"],
                "sdofs": solved_sdofs,
                "iota": solved_iota,
                "G": solved_G,
                "fun": solve_result["fun"],
                "linear_solve_factors": _traceable_result_linear_solve_factors(
                    solve_result,
                    linearization_kind,
                ),
                "success": solve_result["success"],
                "nit": jnp.asarray(solve_result["nit"], dtype=jnp.int64),
            }

        if linearization_kind != "exact_jacobian":
            return _run_traceable_solve(None)

        def _warmstart_failure(_):
            warmstart_sdofs, warmstart_iota, warmstart_G = (
                booz_jax._unpack_decision_vector_jax(
                    warmstart_x,
                    optimize_G,
                    coil_set_spec=coil_set_spec,
                )
            )
            warmstart_fun = _evaluate_traceable_total_objective(
                warmstart_x,
                coil_dofs,
                coil_set_spec,
                objective_kwargs,
            )
            return {
                "x": warmstart_x,
                "sdofs": warmstart_sdofs,
                "iota": warmstart_iota,
                "G": warmstart_G,
                "fun": warmstart_fun,
                "linear_solve_factors": None,
                "success": _runtime_bool(False),
                "nit": jnp.asarray(0, dtype=jnp.int64),
            }

        return lax.cond(
            warmstart_linear_solve_success,
            _run_traceable_solve,
            _warmstart_failure,
            operand=None,
        )

    def _surface_geometry_for(solved_x):
        sdofs, _, _ = _split_x_inner_runtime(solved_x, optimize_G)
        return _surface_geometry_from_dofs(
            sdofs,
            objective_kwargs["quadpoints_phi"],
            objective_kwargs["quadpoints_theta"],
            objective_kwargs["mpol"],
            objective_kwargs["ntor"],
            objective_kwargs["nfp"],
            objective_kwargs["stellsym"],
            objective_kwargs["scatter_indices"],
            surface_kind=objective_kwargs["surface_kind"],
        )

    def _field_for(coil_dofs, solved_x):
        coil_set_spec = coil_set_spec_from_dofs(coil_dofs)
        gamma, _, _ = _surface_geometry_for(solved_x)
        points = gamma.reshape(-1, 3)
        return grouped_biot_savart_B_from_spec(points, coil_set_spec)

    def _field_at_solution_for(coil_dofs):
        return _field_for(coil_dofs, _solve_for(coil_dofs)["x"])

    def _solved_total_objective_for(coil_dofs, solved_x):
        return _evaluate_traceable_total_objective(
            solved_x,
            coil_dofs,
            coil_set_spec_from_dofs(coil_dofs),
            objective_kwargs,
        )

    def _total_gradient_for(coil_dofs, solved_x, solved_linear_solve_factors):
        return _traceable_total_gradient(
            booz_jax,
            coil_set_spec_from_dofs,
            coil_dofs=coil_dofs,
            solved_x=solved_x,
            solved_linear_solve_factors=solved_linear_solve_factors,
            linearization_kind=linearization_kind,
            linear_solve_tol=linear_solve_tol,
            linear_solve_stab=linear_solve_stab,
            objective_kwargs=objective_kwargs,
        )

    compiled_forward_value_for = _make_traceable_forward_value_pipeline(
        compiled_forward_result_for
    )
    compiled_warmstart_for = jax.jit(_warmstart_for)
    compiled_inner_solve_for = jax.jit(_solve_for)
    compiled_surface_geometry_for = jax.jit(_surface_geometry_for)
    compiled_field_for = jax.jit(_field_for)
    compiled_field_eval_sharding = _make_traceable_field_eval_sharding_pipeline(
        _field_at_solution_for
    )
    compiled_solved_total_objective_for = jax.jit(_solved_total_objective_for)
    compiled_solved_total_gradient_for = jax.jit(_total_gradient_for)

    return {
        "forward_result": compiled_forward_result_for,
        "forward_value": compiled_forward_value_for,
        "warmstart_predict": compiled_warmstart_for,
        "inner_solve": compiled_inner_solve_for,
        "surface_geometry": compiled_surface_geometry_for,
        "field_eval": compiled_field_for,
        "field_eval_sharding": compiled_field_eval_sharding,
        "solved_total_objective": compiled_solved_total_objective_for,
        "solved_total_gradient": compiled_solved_total_gradient_for,
        "value_and_grad_pipeline": resolved_value_and_grad_pipeline,
        "batched_value_and_grad_pipeline": resolved_batched_value_and_grad_pipeline,
    }


def make_traceable_objective_runtime_bundle(
    booz_jax,
    bs_jax,
    iota_target,
    *,
    include_profile_suite=False,
    include_host_wrappers=False,
    outer_objective_config=None,
    success_filter=None,
):
    """Build the shared runtime bundle for the target single-stage objective path.

    The returned entrypoints are cached against deterministic signatures of the
    solved baseline state, objective kwargs, and coil extraction/runtime specs.
    Rebuild the bundle after changing those inputs; do not mutate captured
    objects and expect an existing runtime bundle to retarget itself.

    Returned keys:

    ``objective``
        Pure JAX scalar callable returning a 0-d ``jax.Array``.
    ``value_and_grad``
        Pure JAX callable returning ``(0-d jax.Array, grad jax.Array)``.
    ``batched_value_and_grad``
        Pure JAX callable returning batched ``(value, grad)`` outputs for a
        ``(batch, dof)`` seed array.
    ``forward_result``
        Pure JAX callable returning the traceable inner-solve result used by
        the target-lane objective, including the solved decision vector,
        unpacked Boozer state, and success flags.
    ``reporting_metrics``
        Pure JAX callable returning the solved-state reporting scalars used by
        the single-stage example. Callers that need Python/NumPy materialization
        can host-normalize this explicit boundary themselves, or request the
        companion ``host_reporting_metrics`` wrapper. This entrypoint resolves
        lazily and requires ``outer_objective_config`` when invoked.
    ``host_objective``
        Optional host-normalized callable returning a Python ``float`` when
        ``include_host_wrappers=True``.
    ``host_value_and_grad``
        Optional host-normalized callable returning ``(float, np.ndarray)``
        when ``include_host_wrappers=True``.
    ``host_reporting_metrics``
        Optional host-normalized callable returning the final solved-state
        reporting scalars used by the single-stage example when
        ``include_host_wrappers=True``.
    ``profile_suite``
        Optional profiled pure-JAX closures when ``include_profile_suite=True``.
    """
    runtime_entry = _get_cached_traceable_runtime_entry(
        booz_jax,
        bs_jax,
        iota_target,
        outer_objective_config=outer_objective_config,
        success_filter=success_filter,
    )
    _ensure_traceable_runtime_public_boundaries(runtime_entry)
    runtime_bundle = {
        "objective": runtime_entry["public_objective"],
        "value_and_grad": runtime_entry["public_value_and_grad"],
        "batched_value_and_grad": runtime_entry["public_batched_value_and_grad"],
        "forward_result": runtime_entry["public_forward_result"],
        "reporting_metrics": runtime_entry["public_reporting_metrics"],
    }
    if include_host_wrappers:
        _ensure_traceable_runtime_host_wrappers(runtime_entry, booz_jax)
        runtime_bundle.update(
            {
                "host_objective": runtime_entry["host_objective"],
                "host_value_and_grad": runtime_entry["host_value_and_grad"],
                "host_reporting_metrics": runtime_entry["host_reporting_metrics"],
            }
        )
    if not include_profile_suite:
        return runtime_bundle
    compiled_bundle = runtime_entry["compiled_bundle"]
    if runtime_entry["profile_suite"] is None:
        runtime_entry["profile_suite"] = (
            _make_traceable_objective_profile_suite_from_compiled_bundle(
                compiled_bundle,
                booz_jax,
                bs_jax,
                value_and_grad_pipeline=runtime_entry["public_value_and_grad"],
                batched_value_and_grad_pipeline=runtime_entry[
                    "public_batched_value_and_grad"
                ],
            )
        )
    runtime_bundle["profile_suite"] = runtime_entry["profile_suite"]
    return runtime_bundle


def make_traceable_single_stage_alm_runtime_bundle(
    booz_jax,
    bs_jax,
    iota_target,
    *,
    outer_objective_config,
    alm_config,
    success_filter=None,
):
    """Build the pure-JAX single-stage ALM runtime bundle for the inner solve.

    The returned bundle keeps the hot ALM subproblem entirely in JAX and is
    intended for ``backend='jax', optimizer_backend='ondevice'`` single-stage
    ALM inner solves. Host-side accepted-step reporting and artifact shaping
    remain outside this bundle on explicit Python boundaries. ``success_filter``
    is optional and can reject solved states, for example when the host ALM
    reference lane would treat a self-intersecting surface as an immediate
    failure.
    """
    from .optimizer_jax import _mark_cacheable_jit_value_and_grad

    if outer_objective_config is None:
        raise ValueError(
            "make_traceable_single_stage_alm_runtime_bundle() requires "
            "outer_objective_config."
        )
    normalized_alm_config = _traceable_runtime_hostify_tree(dict(alm_config))
    runtime_entry = _get_cached_traceable_runtime_entry(
        booz_jax,
        bs_jax,
        iota_target,
        outer_objective_config=outer_objective_config,
        success_filter=success_filter,
    )
    alm_cache_key = _traceable_contract_tree_signature(normalized_alm_config)
    cached_bundle = runtime_entry["alm_runtime_bundles"].get(alm_cache_key)
    if cached_bundle is not None:
        return cached_bundle

    compiled_bundle = runtime_entry["compiled_bundle"]
    state = compiled_bundle["state"]
    objective_kwargs = dict(
        state["objective_kwargs"],
        outer_objective_config=_traceable_runtime_hostify_tree(outer_objective_config),
    )
    coil_set_spec_from_dofs = state["coil_set_spec_from_dofs"]
    compiled_forward_result_for = compiled_bundle["compiled_forward_result_for"]
    linearization_kind = state["linearization_kind"]
    linear_solve_tol = state["linear_solve_tol"]
    linear_solve_stab = state["linear_solve_stab"]
    constraint_names = tuple(
        str(name) for name in normalized_alm_config["constraint_names"]
    )
    normalized_alm_config["constraint_names"] = constraint_names

    def _failure_evaluation(forward_result):
        failure_total = forward_result["value"]
        constraint_values = jnp.broadcast_to(
            failure_total,
            (len(constraint_names),),
        )
        return {
            "total": failure_total,
            "base_total": failure_total,
            "physics_total": failure_total,
            "constraint_values": constraint_values,
            "feasibility_values": constraint_values,
            "x": forward_result["x"],
            "linear_solve_factors": forward_result["linear_solve_factors"],
            "success": forward_result["success"],
        }

    def _normalize_runtime_inputs(coil_dofs, multipliers, penalty):
        return (
            _as_jax_float64(coil_dofs),
            _as_jax_float64(multipliers),
            _as_jax_float64(penalty),
        )

    def _alm_evaluation_for(coil_dofs, multipliers, penalty):
        coil_dofs, multipliers, penalty = _normalize_runtime_inputs(
            coil_dofs,
            multipliers,
            penalty,
        )
        forward_result = compiled_forward_result_for(coil_dofs)

        def _success(_):
            evaluation = _traceable_single_stage_alm_evaluation(
                forward_result["x"],
                coil_dofs,
                coil_set_spec_from_dofs(coil_dofs),
                objective_kwargs=objective_kwargs,
                alm_config=normalized_alm_config,
                multipliers=multipliers,
                penalty=penalty,
            )
            evaluation["x"] = forward_result["x"]
            evaluation["linear_solve_factors"] = forward_result[
                "linear_solve_factors"
            ]
            evaluation["success"] = forward_result["success"]
            return evaluation

        return jax.lax.cond(
            forward_result["success"],
            _success,
            lambda _: _failure_evaluation(forward_result),
            operand=None,
        )

    compiled_evaluation_for = jax.jit(_alm_evaluation_for)

    def _alm_total_gradient_for(
        coil_dofs,
        solved_x,
        solved_linear_solve_factors,
        multipliers,
        penalty,
    ):
        def _scalar_objective_fn(
            x_inner,
            current_coil_dofs,
            coil_set_spec,
            *,
            objective_kwargs,
        ):
            return _traceable_single_stage_alm_evaluation(
                x_inner,
                current_coil_dofs,
                coil_set_spec,
                objective_kwargs=objective_kwargs,
                alm_config=normalized_alm_config,
                multipliers=multipliers,
                penalty=penalty,
            )["total"]

        return _traceable_total_gradient_with_status(
            booz_jax,
            coil_set_spec_from_dofs,
            coil_dofs=coil_dofs,
            solved_x=solved_x,
            solved_linear_solve_factors=solved_linear_solve_factors,
            linearization_kind=linearization_kind,
            linear_solve_tol=linear_solve_tol,
            linear_solve_stab=linear_solve_stab,
            objective_kwargs=objective_kwargs,
            scalar_objective_fn=_scalar_objective_fn,
        )

    compiled_total_gradient_for = jax.jit(_alm_total_gradient_for)

    @jax.custom_vjp
    def _objective(coil_dofs, multipliers, penalty):
        return compiled_evaluation_for(coil_dofs, multipliers, penalty)["total"]

    def _objective_fwd(coil_dofs, multipliers, penalty):
        evaluation = compiled_evaluation_for(coil_dofs, multipliers, penalty)
        return evaluation["total"], (
            coil_dofs,
            evaluation["x"],
            evaluation["linear_solve_factors"],
            evaluation["success"],
            multipliers,
            penalty,
        )

    def _objective_bwd(saved_state, cotangent):
        (
            coil_dofs,
            solved_x,
            solved_linear_solve_factors,
            success,
            multipliers,
            penalty,
        ) = saved_state

        def _success(_):
            grad, linear_solve_success = compiled_total_gradient_for(
                coil_dofs,
                solved_x,
                solved_linear_solve_factors,
                multipliers,
                penalty,
            )
            return _traceable_adjoint_gradient_or_nan(grad, linear_solve_success)

        grad = jax.lax.cond(
            success,
            _success,
            lambda _: _traceable_adjoint_fail_gradient_like(coil_dofs),
            operand=None,
        )
        multipliers_bar = _runtime_zeros_like(multipliers)
        penalty_bar = _runtime_float64_scalar(0.0, reference=grad)
        return (
            _as_runtime_float64(cotangent, reference=grad) * grad,
            multipliers_bar,
            penalty_bar,
        )

    _objective.defvjp(_objective_fwd, _objective_bwd)
    compiled_objective = jax.jit(_objective)

    def objective(coil_dofs, multipliers, penalty):
        coil_dofs, multipliers, penalty = _normalize_runtime_inputs(
            coil_dofs,
            multipliers,
            penalty,
        )
        return compiled_objective(
            coil_dofs,
            multipliers,
            penalty,
        )

    def evaluate(coil_dofs, multipliers, penalty):
        coil_dofs, multipliers, penalty = _normalize_runtime_inputs(
            coil_dofs,
            multipliers,
            penalty,
        )
        return compiled_evaluation_for(
            coil_dofs,
            multipliers,
            penalty,
        )

    @_mark_cacheable_jit_value_and_grad
    @jax.jit
    def value_and_grad(coil_dofs, multipliers, penalty):
        return jax.value_and_grad(_objective, argnums=0)(
            coil_dofs,
            multipliers,
            penalty,
        )

    def public_value_and_grad(coil_dofs, multipliers, penalty):
        coil_dofs, multipliers, penalty = _normalize_runtime_inputs(
            coil_dofs,
            multipliers,
            penalty,
        )
        return value_and_grad(
            coil_dofs,
            multipliers,
            penalty,
        )

    alm_runtime_bundle = {
        "objective": objective,
        "evaluate": evaluate,
        "value_and_grad": public_value_and_grad,
        "constraint_names": constraint_names,
    }
    runtime_entry["alm_runtime_bundles"][alm_cache_key] = alm_runtime_bundle
    return alm_runtime_bundle


def make_traceable_objective_profile_suite(
    booz_jax,
    bs_jax,
    iota_target,
    *,
    outer_objective_config=None,
):
    """Build profiled pure-JAX closures for the target single-stage objective path."""
    return make_traceable_objective_runtime_bundle(
        booz_jax,
        bs_jax,
        iota_target,
        include_profile_suite=True,
        outer_objective_config=outer_objective_config,
    )["profile_suite"]
