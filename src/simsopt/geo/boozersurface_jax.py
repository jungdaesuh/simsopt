"""
Lane-aware JAX Boozer surface solver.

The public/reference lane still permits host-side SciPy minimization via
``optimizer_backend="scipy"``. The target lane uses
``optimizer_backend="ondevice"`` for JAX-resident execution.

This module owns the LS/exact solver routing contract. Only the
``ondevice`` backend is intended to represent the eventual target optimizer
lane, not a claim that the full workflow is already production-complete.

Architecture (per M0 contract §5-§6):
  - Adapter pattern: ``BoozerSurfaceJAX`` inherits ``Optimizable`` and
    mirrors the CPU ``BoozerSurface`` public API.
  - The outer ``Optimizable`` dependency graph and ``need_to_run_code``
    dirty-flag semantics are preserved.
  - The reference lane may still cross the host/device boundary inside the LS
    optimizer loop; that boundary is isolated to the reference-only optimizer
    module.

Builds on M3's composed derivative path:
  - ``_surface_geometry_from_dofs()`` for surface DOFs → geometry (SSOT)
  - ``boozer_residual_scalar()`` for the forward residual
  - ``boozer_residual_vector()`` for the exact Newton residual vector
  - ``boozer_residual_coil_vjp()`` for outer-path coil sensitivities
  - ``jax.grad`` / ``jax.hessian`` / ``jax.jacfwd`` for all derivatives
"""

import hashlib
import inspect
from dataclasses import dataclass
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg

from ..backend import (
    get_backend_config,
    raise_if_strict_jax_fallback,
    warn_if_jax_fallback,
)
from .._core.jax_host_boundary import (
    host_all_finite as _host_all_finite,
    host_array as _host_numpy,
    host_inf_norm as _host_inf_norm,
    host_scalar as _host_scalar,
    host_tree as _hostify_tree,
)
from ..jax_core._math_utils import (
    as_jax_float64 as _as_jax_float64,
    as_jax_int32 as _as_jax_int32,
    as_runtime_float64 as _as_runtime_float64,
    concat_jax_float64 as _concat_jax_float64,
)

try:
    from simsopt._core.optimizable import Optimizable
except (ImportError, ModuleNotFoundError):
    # Fallback when simsoptpp is unavailable (standalone JAX tests).
    # In production with simsopt fully installed, the real Optimizable is used.
    class Optimizable:  # type: ignore[no-redef]
        def __init__(self, *args, depends_on=None, **kwargs):
            pass


from .surface_fourier_jax import (
    stellsym_scatter_indices,
)
from ._boozersurface_current_guard import (
    guard_none_G_coil_gradient_callback as _guard_none_G_coil_gradient_callback,
    require_fixed_currents_for_none_G as _require_fixed_currents_for_none_G,
)
from ..jax_core.field import (
    grouped_biot_savart_A_from_inputs,
    grouped_biot_savart_A_from_spec,
    grouped_biot_savart_B_from_inputs,
    grouped_biot_savart_B_from_spec,
    grouped_coil_currents_from_inputs,
    grouped_coil_currents_from_spec,
    grouped_coil_index_lists_from_spec,
    grouped_coil_set_spec_from_source,
    grouped_field_data_from_spec,
    grouped_field_inputs_from_spec,
)
from ..jax_core.specs import CoilGroupSpec, GroupedCoilSetSpec
from .boozer_residual_jax import (
    _split_decision_vector as _split_boozer_decision_vector,
    boozer_residual_scalar,
    boozer_residual_vector,
    _surface_geometry_from_dofs,
)
from .label_constraints_jax import (
    area_jax,
    volume_jax,
    toroidal_flux_jax,
    compute_G_from_currents,
)
from . import optimizer_jax as _optimizer_jax
from .optimizer_jax import (
    VALID_LEAST_SQUARES_ALGORITHMS,
    VALID_OPTIMIZER_BACKENDS,
    levenberg_marquardt_traceable,
    newton_exact,
    newton_exact_traceable,
    newton_polish,
    newton_polish_traceable,
    reference_least_squares,
    reference_minimize,
    require_target_backend_x64,
    resolve_reference_least_squares_optimizer_method,
    resolve_target_least_squares_optimizer_method,
    target_least_squares,
    target_minimize,
)

__all__ = ["BoozerSurfaceJAX"]


@dataclass(frozen=True)
class _BoozerPenaltyOptimizerState:
    surface_dofs: jax.Array
    iota: jax.Array


@dataclass(frozen=True)
class _BoozerPenaltyOptimizerStateWithG:
    surface_dofs: jax.Array
    iota: jax.Array
    G: jax.Array


jax.tree_util.register_dataclass(
    _BoozerPenaltyOptimizerState,
    data_fields=["surface_dofs", "iota"],
    meta_fields=[],
)
jax.tree_util.register_dataclass(
    _BoozerPenaltyOptimizerStateWithG,
    data_fields=["surface_dofs", "iota", "G"],
    meta_fields=[],
)


def _require_boozer_vjp_callback_signature(callback, *, callback_name: str):
    """Fail fast when a result-dict VJP hook cannot accept the public contract."""
    if callback is None:
        return None
    try:
        inspect.signature(callback).bind(object(), object(), object(), object())
    except TypeError as exc:
        raise TypeError(
            f"BoozerSurfaceJAX result callback {callback_name!r} must accept "
            "(lm, booz_surf, iota, G)."
        ) from exc
    return callback


def _guard_solver_callback_freshness(
    callback,
    *,
    booz_surf,
    solve_generation: int,
    callback_name: str,
):
    """Reject stale result callbacks after the Boozer solve state changes."""
    if callback is None:
        return None

    def guarded(*args, **kwargs):
        current_generation = getattr(booz_surf, "_solver_generation", None)
        if booz_surf.need_to_run_code or current_generation != solve_generation:
            raise RuntimeError(
                f"BoozerSurfaceJAX result callback {callback_name!r} is stale "
                f"(expected generation {solve_generation}, got {current_generation}). "
                "Re-run boozer_surface.run_code(...) before requesting adjoints."
            )
        return callback(*args, **kwargs)

    return guarded


def _advance_solver_generation(booz_surf) -> int:
    solve_generation = booz_surf._solver_generation + 1
    booz_surf._solver_generation = solve_generation
    return solve_generation


def _prepare_result_callback(
    callback,
    *,
    booz_surf,
    solve_generation: int,
    callback_name: str,
    G_provided: bool,
    freshness_guard: bool,
):
    callback = _guard_none_G_coil_gradient_callback(
        callback,
        biotsavart=booz_surf.biotsavart,
        component="BoozerSurfaceJAX",
        coil_attrs=("_coils",),
        G_provided=G_provided,
    )
    callback = _require_boozer_vjp_callback_signature(
        callback,
        callback_name=callback_name,
    )
    if freshness_guard:
        callback = _guard_solver_callback_freshness(
            callback,
            booz_surf=booz_surf,
            solve_generation=solve_generation,
            callback_name=callback_name,
        )
    return callback


def _as_boozer_penalty_optimizer_state(x, *, optimize_G):
    if optimize_G:
        if isinstance(x, _BoozerPenaltyOptimizerStateWithG):
            return _BoozerPenaltyOptimizerStateWithG(
                surface_dofs=_as_jax_float64(x.surface_dofs),
                iota=_as_jax_float64(x.iota),
                G=_as_jax_float64(x.G),
            )
    elif isinstance(x, _BoozerPenaltyOptimizerState):
        return _BoozerPenaltyOptimizerState(
            surface_dofs=_as_jax_float64(x.surface_dofs),
            iota=_as_jax_float64(x.iota),
        )

    x_jax = _as_jax_float64(x)
    sdofs, iota, G = _split_decision_vector_jax(x_jax, optimize_G=optimize_G)
    if optimize_G:
        return _BoozerPenaltyOptimizerStateWithG(
            surface_dofs=sdofs,
            iota=iota,
            G=G,
        )
    return _BoozerPenaltyOptimizerState(
        surface_dofs=sdofs,
        iota=iota,
    )


def _traceable_array_signature(array):
    """Return a value-based signature for traced static array-like inputs."""
    if array is None:
        return None
    if isinstance(array, jax.Array):
        array_np = np.asarray(jax.device_get(array))
    else:
        array_np = np.asarray(array)
    return (
        str(array_np.dtype),
        tuple(int(dim) for dim in array_np.shape),
        array_np.tobytes(),
    )


def _runtime_cache_leaf_signature(leaf):
    if isinstance(leaf, (jax.Array, np.ndarray)):
        array = np.asarray(jax.device_get(leaf))
        return (
            "array",
            str(array.dtype),
            tuple(int(dim) for dim in array.shape),
            hashlib.blake2b(array.tobytes(), digest_size=16).hexdigest(),
        )
    if isinstance(leaf, np.generic):
        return ("numpy_scalar", str(leaf.dtype), leaf.item())
    if isinstance(leaf, (str, int, float, bool, type(None))):
        return ("scalar", leaf)
    return ("repr", type(leaf).__qualname__, repr(leaf))


def _runtime_cache_tree_signature(tree):
    try:
        leaves, treedef = jax.tree_util.tree_flatten(tree)
    except TypeError:
        return _runtime_cache_leaf_signature(tree)
    return (
        "tree",
        repr(treedef),
        tuple(_runtime_cache_leaf_signature(leaf) for leaf in leaves),
    )


def _boozer_penalty_optimizer_state_to_vector(x, *, optimize_G):
    optimizer_state = _as_boozer_penalty_optimizer_state(x, optimize_G=optimize_G)
    if optimize_G:
        return _concat_jax_float64(
            optimizer_state.surface_dofs,
            [optimizer_state.iota, optimizer_state.G],
        )
    return _concat_jax_float64(
        optimizer_state.surface_dofs,
        [optimizer_state.iota],
    )


def _split_decision_vector_jax(x, *, optimize_G):
    return _split_boozer_decision_vector(x, optimize_G=optimize_G)


def _generic_surface_scatter_operator(mpol: int, ntor: int):
    positions = np.asarray(stellsym_scatter_indices(mpol, ntor), dtype=np.int32)
    n_per_coord = int((2 * mpol + 1) * (2 * ntor + 1))
    operator = np.zeros((3 * n_per_coord, positions.size), dtype=np.float64)
    operator[positions, np.arange(positions.size)] = 1.0
    return _as_jax_float64(operator)


def _cross_product(left, right):
    return jnp.cross(left, right, axis=-1)


def _select_axis0(array, index: int):
    selector = np.zeros(int(array.shape[0]), dtype=np.float64)
    selector[int(index)] = 1.0
    return jnp.tensordot(
        _as_runtime_float64(selector, reference=array),
        jnp.asarray(array),
        axes=((0,), (0,)),
    )


def _surface_sample_z(gamma):
    sample = _select_axis0(_select_axis0(gamma, 0), 0)
    return _select_axis0(sample, 2)


def _replace_group_coil_array(coil_arrays, group_index, group_array):
    grouped_arrays = list(coil_arrays)
    grouped_arrays[group_index] = group_array
    return grouped_arrays


def _replace_group_coil_set_spec(
    coil_set_spec: GroupedCoilSetSpec,
    group_index: int,
    group_array,
) -> GroupedCoilSetSpec:
    gammas, gammadashs, currents = group_array
    groups = list(coil_set_spec.groups)
    group = groups[group_index]
    groups[group_index] = CoilGroupSpec(
        gammas=gammas,
        gammadashs=gammadashs,
        currents=currents,
        coil_indices=group.coil_indices,
    )
    return GroupedCoilSetSpec(groups=tuple(groups))


def _yield_group_vjps(lm, group_runners, coil_arrays, coil_indices):
    for group_runner, group_array, group_index_list in zip(
        group_runners,
        coil_arrays,
        coil_indices,
    ):
        _, vjp_fn = jax.vjp(group_runner, group_array)
        yield vjp_fn(lm)[0], group_index_list


def _extract_grouped_coil_set_spec(biotsavart):
    """Return the immutable grouped-coil spec for a biotsavart-like object.

    ``BoozerSurfaceJAX`` now requires its field adapter to expose explicit
    immutable grouped-coil state through ``coil_set_spec()``. Hidden grouped
    extractors and raw ``_coils`` snapshots are no longer accepted here.
    """
    coil_set_spec = getattr(biotsavart, "coil_set_spec", None)
    if coil_set_spec is None or not callable(coil_set_spec):
        raise AttributeError(
            "BoozerSurfaceJAX requires a biotsavart object that provides "
            "coil_set_spec() for explicit immutable grouped-coil state. "
            "Hidden _extract_coil_data_grouped() and _coils compatibility seams "
            "are no longer supported."
        )
    return grouped_coil_set_spec_from_source(coil_set_spec())


def _grouped_coil_currents(*, coil_arrays=None, coil_set_spec=None):
    if coil_set_spec is not None:
        return grouped_coil_currents_from_spec(coil_set_spec)
    return grouped_coil_currents_from_inputs(coil_arrays)


def _resolved_coil_set_spec(default_spec, *, coil_arrays=None, coil_set_spec=None):
    return default_spec if coil_set_spec is None else coil_set_spec


def _grouped_biot_savart_B_points(points, *, coil_arrays=None, coil_set_spec=None):
    if coil_set_spec is not None:
        return grouped_biot_savart_B_from_spec(points, coil_set_spec)
    return grouped_biot_savart_B_from_inputs(points, coil_arrays)


def _grouped_biot_savart_A_points(points, *, coil_arrays=None, coil_set_spec=None):
    if coil_set_spec is not None:
        return grouped_biot_savart_A_from_spec(points, coil_set_spec)
    return grouped_biot_savart_A_from_inputs(points, coil_arrays)


def _compute_label(
    label_type,
    gamma,
    xphi,
    xtheta,
    phi_idx,
    points,
    coil_arrays=None,
    coil_set_spec=None,
):
    """Compute the label value (volume, area, or toroidal flux).

    Shared by penalty objective, exact residual, and residual vector.
    """
    normal = _cross_product(xphi, xtheta)
    if label_type == "volume":
        return volume_jax(gamma, normal)
    if label_type == "area":
        return area_jax(normal)
    ntheta = gamma.shape[1]
    A = _grouped_biot_savart_A_points(
        points,
        coil_arrays=coil_arrays,
        coil_set_spec=coil_set_spec,
    )
    A = A.reshape(gamma.shape)
    return toroidal_flux_jax(
        _select_axis0(A, phi_idx),
        _select_axis0(xtheta, phi_idx),
        ntheta,
    )


def _compute_label_and_axis_z(
    *,
    gamma,
    xphi,
    xtheta,
    points,
    label_type,
    phi_idx,
    coil_arrays=None,
    coil_set_spec=None,
):
    label_value = _compute_label(
        label_type,
        gamma,
        xphi,
        xtheta,
        phi_idx,
        points,
        coil_arrays=coil_arrays,
        coil_set_spec=coil_set_spec,
    )
    return label_value, _surface_sample_z(gamma)


def _boozer_penalty_objective(
    x,
    *,
    coil_arrays=None,
    coil_set_spec=None,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    surface_kind,
    targetlabel,
    constraint_weight,
    label_type,
    phi_idx,
    optimize_G,
    weight_inv_modB,
):
    """Scalarized penalty objective for the BoozerLS inner solve.

    Extends M3's ``boozer_penalty_composed`` with label and z-constraints.

    Pure function: ``x → scalar``.  JAX autodiff gives gradient and
    Hessian for free.

    The optimizer state may be either the historical flat decision vector
    ``[surface_dofs, iota]`` / ``[surface_dofs, iota, G]`` or the structured
    Boozer penalty optimizer pytree that carries the same fields explicitly.
    """
    optimizer_state = _as_boozer_penalty_optimizer_state(x, optimize_G=optimize_G)
    sdofs = optimizer_state.surface_dofs
    iota = optimizer_state.iota
    G = optimizer_state.G if optimize_G else None
    if not optimize_G:
        G = compute_G_from_currents(
            _grouped_coil_currents(coil_arrays=coil_arrays, coil_set_spec=coil_set_spec)
        )

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
    B = _grouped_biot_savart_B_points(
        points,
        coil_arrays=coil_arrays,
        coil_set_spec=coil_set_spec,
    )
    B = B.reshape(nphi, ntheta, 3)

    J_boozer = boozer_residual_scalar(G, iota, B, xphi, xtheta, weight_inv_modB)

    label_val, gamma_axis_z = _compute_label_and_axis_z(
        gamma=gamma,
        xphi=xphi,
        xtheta=xtheta,
        points=points,
        label_type=label_type,
        phi_idx=phi_idx,
        coil_arrays=coil_arrays,
        coil_set_spec=coil_set_spec,
    )

    half = _as_jax_float64(0.5)
    constraint_weight_jax = _as_jax_float64(constraint_weight)
    targetlabel_jax = _as_jax_float64(targetlabel)
    label_delta = label_val - targetlabel_jax
    J_label = half * constraint_weight_jax * label_delta * label_delta
    J_z = half * constraint_weight_jax * gamma_axis_z * gamma_axis_z

    return J_boozer + J_label + J_z


def _boozer_penalty_residual_vector(
    x,
    *,
    coil_arrays=None,
    coil_set_spec=None,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    surface_kind,
    targetlabel,
    constraint_weight,
    label_type,
    phi_idx,
    optimize_G,
    weight_inv_modB,
):
    optimizer_state = _as_boozer_penalty_optimizer_state(x, optimize_G=optimize_G)
    G_value = (
        optimizer_state.G
        if optimize_G
        else compute_G_from_currents(
            _grouped_coil_currents(coil_arrays=coil_arrays, coil_set_spec=coil_set_spec)
        )
    )
    gamma, xphi, xtheta = _surface_geometry_from_dofs(
        optimizer_state.surface_dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
        surface_kind=surface_kind,
    )
    nphi, ntheta = int(gamma.shape[0]), int(gamma.shape[1])
    points = gamma.reshape(-1, 3)
    B = _grouped_biot_savart_B_points(
        points,
        coil_arrays=coil_arrays,
        coil_set_spec=coil_set_spec,
    ).reshape(nphi, ntheta, 3)

    r_boozer_raw = boozer_residual_vector(
        G_value,
        optimizer_state.iota,
        B,
        xphi,
        xtheta,
        weight_inv_modB,
    )
    num_res = _as_jax_float64(3 * nphi * ntheta)
    r_boozer = r_boozer_raw / jnp.sqrt(num_res)

    constraint_weight = constraint_weight if constraint_weight is not None else 1.0
    constraint_weight = _as_jax_float64(constraint_weight)
    label_value, gamma_axis_z = _compute_label_and_axis_z(
        gamma=gamma,
        xphi=xphi,
        xtheta=xtheta,
        points=points,
        label_type=label_type,
        phi_idx=phi_idx,
        coil_arrays=coil_arrays,
        coil_set_spec=coil_set_spec,
    )
    weight_sqrt = jnp.sqrt(constraint_weight)
    rl = weight_sqrt * (label_value - _as_jax_float64(targetlabel))
    rz = weight_sqrt * gamma_axis_z

    return _concat_jax_float64(r_boozer, [rl, rz])


def _boozer_exact_residual(
    x,
    *,
    coil_arrays=None,
    coil_set_spec=None,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    surface_kind,
    targetlabel,
    label_type,
    phi_idx,
    mask_indices,
    stellsym_surface,
    weight_inv_modB,
):
    """Route to the stellsym-specialized exact residual implementation.

    ``stellsym_surface`` changes the residual length because the axis
    constraint is only present on the non-stellsym branch. Callers must bind
    that flag at closure-construction time so each compiled trace sees one
    fixed output shape.
    """
    residual_fn = _select_exact_residual_fn(stellsym_surface)
    return residual_fn(
        x,
        coil_arrays=coil_arrays,
        coil_set_spec=coil_set_spec,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
        scatter_indices=scatter_indices,
        surface_kind=surface_kind,
        targetlabel=targetlabel,
        label_type=label_type,
        phi_idx=phi_idx,
        mask_indices=mask_indices,
        weight_inv_modB=weight_inv_modB,
    )


def _select_exact_residual_fn(stellsym_surface):
    """Select the exact-residual implementation for a fixed surface symmetry.

    The selected callable becomes part of the surrounding compiled closure, so
    ``stellsym_surface`` is a compile-time specialization choice rather than a
    dynamic traced branch.
    """
    if stellsym_surface:
        return _boozer_exact_residual_stellsym
    return _boozer_exact_residual_nonstellsym


def _boozer_exact_residual_impl(
    x,
    *,
    coil_arrays=None,
    coil_set_spec=None,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    surface_kind,
    targetlabel,
    label_type,
    phi_idx,
    mask_indices,
    weight_inv_modB,
    include_axis_constraint,
):
    """Residual vector for the BoozerExact Newton system.

    Extends M3's ``boozer_residual_vector`` with masking and constraint
    equations (label, z-coordinate).

    Returns: (n_eq,) residual vector where ``r(x) = 0`` at the solution.
    The decision vector is always ``x = [surface_dofs, iota, G]``.
    """
    sdofs, iota, G = _split_decision_vector_jax(x, optimize_G=True)

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
    B = _grouped_biot_savart_B_points(
        points,
        coil_arrays=coil_arrays,
        coil_set_spec=coil_set_spec,
    )
    B = B.reshape(nphi, ntheta, 3)

    r_flat = boozer_residual_vector(G, iota, B, xphi, xtheta, weight_inv_modB)
    r_masked = r_flat[mask_indices]

    label_val, gamma_axis_z = _compute_label_and_axis_z(
        gamma=gamma,
        xphi=xphi,
        xtheta=xtheta,
        points=points,
        label_type=label_type,
        phi_idx=phi_idx,
        coil_arrays=coil_arrays,
        coil_set_spec=coil_set_spec,
    )
    r_label = label_val - _as_jax_float64(targetlabel)

    if include_axis_constraint:
        residual_tail = _as_jax_float64([r_label, gamma_axis_z])
    else:
        residual_tail = _as_jax_float64([r_label])
    return _concat_jax_float64(r_masked, residual_tail)


def _boozer_exact_residual_stellsym(
    x,
    *,
    coil_arrays=None,
    coil_set_spec=None,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    surface_kind,
    targetlabel,
    label_type,
    phi_idx,
    mask_indices,
    weight_inv_modB,
):
    return _boozer_exact_residual_impl(
        x,
        coil_arrays=coil_arrays,
        coil_set_spec=coil_set_spec,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
        scatter_indices=scatter_indices,
        surface_kind=surface_kind,
        targetlabel=targetlabel,
        label_type=label_type,
        phi_idx=phi_idx,
        mask_indices=mask_indices,
        weight_inv_modB=weight_inv_modB,
        include_axis_constraint=False,
    )


def _boozer_exact_residual_nonstellsym(
    x,
    *,
    coil_arrays=None,
    coil_set_spec=None,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    surface_kind,
    targetlabel,
    label_type,
    phi_idx,
    mask_indices,
    weight_inv_modB,
):
    return _boozer_exact_residual_impl(
        x,
        coil_arrays=coil_arrays,
        coil_set_spec=coil_set_spec,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
        scatter_indices=scatter_indices,
        surface_kind=surface_kind,
        targetlabel=targetlabel,
        label_type=label_type,
        phi_idx=phi_idx,
        mask_indices=mask_indices,
        weight_inv_modB=weight_inv_modB,
        include_axis_constraint=True,
    )


def _boozer_exact_coil_vjp(lm, booz_surf, iota, G):
    """JAX VJP for the exact path.

    Replaces CPU ``boozer_surface_dexactresidual_dcoils_dcurrents_vjp``.

    Differentiates the FULL exact residual vector (Boozer + label + z)
    w.r.t. coil geometry and currents via ``jax.vjp``.  This correctly
    includes the label derivative term that the CPU code adds explicitly.

    Args:
        lm: (n_eq,) adjoint vector from the outer implicit-function solve.
        booz_surf: ``BoozerSurfaceJAX`` instance.
        iota: rotational transform at the solution.
        G: Boozer G at the solution.

    Returns:
        (d_coil_arrays,), coil_indices — grouped cotangents and index list.
        ``d_coil_arrays`` is a list of ``(d_g, d_gd, d_c)`` tuples matching
        the coil_arrays pytree structure.
    """
    sdofs = booz_surf._get_surface_dofs()
    x = _concat_jax_float64(sdofs, [iota, G])
    mask_indices = booz_surf._compute_stellsym_mask_indices()

    coil_arrays = booz_surf._coil_arrays
    coil_indices = booz_surf._coil_index_lists

    residual_fn = _select_exact_residual_fn(booz_surf.stellsym)

    def residual_of_coils(ca):
        return residual_fn(
            x,
            coil_arrays=ca,
            quadpoints_phi=booz_surf.quadpoints_phi,
            quadpoints_theta=booz_surf.quadpoints_theta,
            mpol=booz_surf.mpol,
            ntor=booz_surf.ntor,
            nfp=booz_surf.nfp,
            stellsym=booz_surf.stellsym,
            scatter_indices=booz_surf.scatter_indices,
            surface_kind=booz_surf._surface_geometry_kind,
            targetlabel=booz_surf.targetlabel,
            label_type=booz_surf.label_type,
            phi_idx=booz_surf.phi_idx,
            mask_indices=mask_indices,
            weight_inv_modB=booz_surf.options["weight_inv_modB"],
        )

    _, vjp_fn = jax.vjp(residual_of_coils, coil_arrays)
    (d_coil_arrays,) = vjp_fn(lm)
    return d_coil_arrays, coil_indices


def _boozer_exact_coil_vjp_groups(lm, booz_surf, iota, G):
    """Yield exact-solve coil VJPs one grouped coil block at a time."""
    yield from _build_exact_group_vjp_callback(booz_surf, iota, G)(
        lm,
        booz_surf,
        iota,
        G,
    )


def _build_exact_group_vjp_callback(booz_surf, iota, G):
    """Build stable exact-solve group runners for repeated streaming VJPs."""
    sdofs = booz_surf._get_surface_dofs()
    x = _concat_jax_float64(sdofs, [iota, G])
    mask_indices = booz_surf._compute_stellsym_mask_indices()

    coil_arrays = booz_surf._coil_arrays
    coil_indices = booz_surf._coil_index_lists

    group_runners = tuple(
        _make_exact_group_runner(
            x,
            coil_arrays,
            booz_surf,
            mask_indices,
            group_index,
        )
        for group_index in range(len(coil_arrays))
    )

    def vjp_groups(lm, _booz_surf, _iota, _G):
        yield from _yield_group_vjps(lm, group_runners, coil_arrays, coil_indices)

    return vjp_groups


def _make_exact_group_runner(x, coil_arrays, booz_surf, mask_indices, group_index):
    residual_fn = _select_exact_residual_fn(booz_surf.stellsym)

    def residual_of_group(group_array):
        return residual_fn(
            x,
            coil_arrays=_replace_group_coil_array(
                coil_arrays,
                group_index,
                group_array,
            ),
            quadpoints_phi=booz_surf.quadpoints_phi,
            quadpoints_theta=booz_surf.quadpoints_theta,
            mpol=booz_surf.mpol,
            ntor=booz_surf.ntor,
            nfp=booz_surf.nfp,
            stellsym=booz_surf.stellsym,
            scatter_indices=booz_surf.scatter_indices,
            surface_kind=booz_surf._surface_geometry_kind,
            targetlabel=booz_surf.targetlabel,
            label_type=booz_surf.label_type,
            phi_idx=booz_surf.phi_idx,
            mask_indices=mask_indices,
            weight_inv_modB=booz_surf.options["weight_inv_modB"],
        )

    return residual_of_group


def _boozer_ls_coil_vjp(lm, booz_surf, iota, G, weight_inv_modB=True):
    """JAX VJP for the LS penalty path.

    Replaces CPU ``boozer_surface_dlsqgrad_dcoils_vjp``.

    Differentiates the penalty objective GRADIENT w.r.t. coil geometry
    and currents.  This captures all terms (Boozer residual + label +
    z-constraint) because the composed objective includes them.

    Args:
        lm: (n,) adjoint vector (same shape as decision vector).
        booz_surf: ``BoozerSurfaceJAX`` instance.
        iota: rotational transform at the solution.
        G: Boozer G at the solution.
        weight_inv_modB: residual weighting flag.

    Returns:
        (d_coil_arrays,), coil_indices — grouped cotangents and index list.
        ``d_coil_arrays`` is a list of ``(d_g, d_gd, d_c)`` tuples matching
        the coil_arrays pytree structure.
    """
    x, optimize_G = _ls_decision_vector(booz_surf, iota, G)

    coil_arrays = booz_surf._coil_arrays
    coil_indices = booz_surf._coil_index_lists

    def directional_objective_of_coils(ca):
        return _ls_penalty_directional_objective(
            x,
            lm,
            ca,
            booz_surf,
            optimize_G,
            weight_inv_modB,
        )

    d_coil_arrays = jax.grad(directional_objective_of_coils)(coil_arrays)
    return d_coil_arrays, coil_indices


def _boozer_ls_coil_vjp_groups(lm, booz_surf, iota, G, weight_inv_modB=True):
    """Yield LS-path coil VJPs one grouped coil block at a time."""
    yield from _build_ls_group_vjp_callback(
        booz_surf,
        iota,
        G,
        solve_generation=getattr(booz_surf, "_solver_generation", 0),
        weight_inv_modB=weight_inv_modB,
    )(
        lm,
        booz_surf,
        iota,
        G,
    )


def _build_ls_group_vjp_callback(
    booz_surf,
    iota,
    G,
    *,
    solve_generation: int,
    weight_inv_modB=True,
):
    """Build stable LS group runners for repeated streaming VJPs."""
    x, optimize_G = _ls_decision_vector(booz_surf, iota, G)

    coil_set_spec = booz_surf.coil_set_spec
    coil_arrays = grouped_field_inputs_from_spec(coil_set_spec)
    coil_indices = booz_surf._coil_index_lists

    group_runners = tuple(
        _make_ls_group_runner(
            x,
            coil_set_spec,
            booz_surf,
            optimize_G,
            weight_inv_modB,
            group_index,
        )
        for group_index in range(len(coil_arrays))
    )

    def vjp_groups(lm, _booz_surf, _iota, _G):
        current_generation = getattr(booz_surf, "_solver_generation", None)
        if booz_surf.need_to_run_code or current_generation != solve_generation:
            raise RuntimeError(
                "BoozerSurfaceJAX LS grouped VJP callback is stale; "
                "re-run boozer_surface.run_code(...) before requesting adjoints."
            )
        for group_runner, group_array, group_index_list in zip(
            group_runners,
            coil_arrays,
            coil_indices,
        ):
            _, pullback = jax.vjp(group_runner, group_array, lm)
            yield pullback(_as_jax_float64(1.0))[0], group_index_list

    return vjp_groups


def _make_ls_group_runner(
    x,
    coil_set_spec,
    booz_surf,
    optimize_G,
    weight_inv_modB,
    group_index,
):
    def directional_of_group(group_array, tangent):
        return _group_penalty_directional_objective(
            x,
            tangent,
            _replace_group_coil_set_spec(
                coil_set_spec,
                group_index,
                group_array,
            ),
            booz_surf.quadpoints_phi,
            booz_surf.quadpoints_theta,
            booz_surf.mpol,
            booz_surf.ntor,
            booz_surf.nfp,
            booz_surf.stellsym,
            booz_surf.scatter_indices,
            booz_surf._surface_geometry_kind,
            booz_surf.targetlabel,
            booz_surf.constraint_weight,
            booz_surf.label_type,
            booz_surf.phi_idx,
            optimize_G,
            weight_inv_modB,
        )

    return directional_of_group


def _ls_decision_vector(booz_surf, iota, G):
    optimize_G = G is not None
    sdofs = booz_surf._get_surface_dofs()
    if optimize_G:
        x = _concat_jax_float64(sdofs, [iota, G])
    else:
        x = _concat_jax_float64(sdofs, [iota])
    return x, optimize_G


def _ls_penalty_directional_objective(
    x,
    tangent,
    coil_arrays,
    booz_surf,
    optimize_G,
    weight_inv_modB,
):
    return _directional_derivative(
        _make_ls_penalty_objective(
            booz_surf,
            coil_arrays,
            optimize_G,
            weight_inv_modB,
        ),
        x,
        tangent,
    )


def _group_penalty_directional_objective(
    x,
    tangent,
    coil_set_spec,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    surface_kind,
    targetlabel,
    constraint_weight,
    label_type,
    phi_idx,
    optimize_G,
    weight_inv_modB,
):
    return _directional_derivative(
        _make_boozer_penalty_objective_closure(
            coil_set_spec=coil_set_spec,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
            mpol=mpol,
            ntor=ntor,
            nfp=nfp,
            stellsym=stellsym,
            scatter_indices=scatter_indices,
            surface_kind=surface_kind,
            targetlabel=targetlabel,
            constraint_weight=constraint_weight,
            label_type=label_type,
            phi_idx=phi_idx,
            optimize_G=optimize_G,
            weight_inv_modB=weight_inv_modB,
        ),
        x,
        tangent,
    )


def _make_ls_penalty_objective(
    booz_surf,
    coil_arrays,
    optimize_G,
    weight_inv_modB,
):
    return _make_boozer_penalty_objective_closure(
        coil_arrays=coil_arrays,
        quadpoints_phi=booz_surf.quadpoints_phi,
        quadpoints_theta=booz_surf.quadpoints_theta,
        mpol=booz_surf.mpol,
        ntor=booz_surf.ntor,
        nfp=booz_surf.nfp,
        stellsym=booz_surf.stellsym,
        scatter_indices=booz_surf.scatter_indices,
        surface_kind=booz_surf._surface_geometry_kind,
        targetlabel=booz_surf.targetlabel,
        constraint_weight=booz_surf.constraint_weight,
        label_type=booz_surf.label_type,
        phi_idx=booz_surf.phi_idx,
        optimize_G=optimize_G,
        weight_inv_modB=weight_inv_modB,
    )


def _make_boozer_penalty_closure(fn, **kwargs):
    """Generic closure builder for Boozer penalty functions.

    Captures all surface/coil keyword arguments and returns a unary
    ``fn(xx, **kwargs)`` closure suitable for JIT tracing.
    """

    def _closure(xx):
        return fn(xx, **kwargs)

    return _closure


def _make_boozer_penalty_objective_closure(**kwargs):
    return _make_boozer_penalty_closure(_boozer_penalty_objective, **kwargs)


def _make_boozer_penalty_residual_closure(**kwargs):
    return _make_boozer_penalty_closure(_boozer_penalty_residual_vector, **kwargs)


def _directional_derivative(objective, x, tangent):
    _, directional = jax.jvp(objective, (x,), (tangent,))
    return directional


def _traceable_plu_or_dummy(matrix, *, finite):
    """Build PLU factors only for finite matrices on traceable paths."""

    def compute_plu(mat):
        return jax.scipy.linalg.lu(mat)

    def dummy_plu(mat):
        zeros = jnp.zeros_like(mat)
        return zeros, zeros, zeros

    if isinstance(finite, (bool, np.bool_)):
        return compute_plu(matrix) if bool(finite) else dummy_plu(matrix)
    if (
        isinstance(finite, jax.Array)
        and finite.shape == ()
        and not isinstance(finite, jax.core.Tracer)
    ):
        finite_value = bool(np.asarray(jax.device_get(finite)))
        return compute_plu(matrix) if finite_value else dummy_plu(matrix)

    return jax.lax.cond(
        jnp.asarray(finite, dtype=jnp.bool_), compute_plu, dummy_plu, matrix
    )


def _exact_newton_reporting_fields(result):
    return {
        "message": result.get("message"),
        "failure_category": result.get("failure_category"),
        "failure_stage": result.get("failure_stage"),
        "jacobian_materialized": result.get("jacobian_materialized"),
        "dense_jacobian_shape": result.get("dense_jacobian_shape"),
        "dense_jacobian_bytes": result.get("dense_jacobian_bytes"),
        "max_dense_jacobian_bytes": result.get("max_dense_jacobian_bytes"),
    }


_DEFAULT_MAX_DENSE_JACOBIAN_BYTES = 512 * 1024 * 1024


_DEFAULT_OPTIONS_LS = {
    "verbose": True,
    "bfgs_tol": 1e-10,
    "bfgs_maxiter": 1500,
    "optimizer_backend": "scipy",
    "limited_memory": False,
    "newton_tol": 1e-11,
    "newton_maxiter": 40,
    "newton_stab": 0.0,
    "weight_inv_modB": True,
    "max_dense_jacobian_bytes": _DEFAULT_MAX_DENSE_JACOBIAN_BYTES,
}

_DEFAULT_OPTIONS_EXACT = {
    "verbose": True,
    "newton_tol": 1e-13,
    "newton_maxiter": 40,
    "weight_inv_modB": False,
    "max_dense_jacobian_bytes": _DEFAULT_MAX_DENSE_JACOBIAN_BYTES,
}

# Options only meaningful for the target/private optimizer backend.
_PRIVATE_OPTIMIZER_OPTIONS = frozenset(
    {
        "force_ondevice_limited_memory",
        "line_search_maxiter",
        "maxgrad",
    }
)

# Options shared by the public SciPy L-BFGS lane and the private L-BFGS lanes.
_LBFGS_TUNING_OPTIONS = frozenset({"maxcor", "ftol", "maxfun", "maxls"})

# Callback options accepted by all backends.
_CALLBACK_OPTIONS = frozenset({"stage_callback", "progress_callback"})
_ONDEVICE_OPTIMIZER_METHODS = frozenset(
    {"bfgs-ondevice", "lbfgs-ondevice", "lm-ondevice"}
)
_LS_DYNAMIC_OPTION_KEYS = frozenset({"least_squares_algorithm"})

_ALLOWED_OPTIONS_LS = (
    frozenset(_DEFAULT_OPTIONS_LS)
    | _LS_DYNAMIC_OPTION_KEYS
    | _PRIVATE_OPTIMIZER_OPTIONS
    | _LBFGS_TUNING_OPTIONS
    | _CALLBACK_OPTIONS
)
_ALLOWED_OPTIONS_EXACT = frozenset(_DEFAULT_OPTIONS_EXACT) | {
    "optimizer_backend",
    "stage_callback",
}


def default_least_squares_algorithm_for_backend(optimizer_backend):
    if optimizer_backend == "ondevice":
        return "lm"
    return "quasi-newton"


def _default_ls_optimizer_backend() -> str:
    if get_backend_config().backend == "jax":
        return "ondevice"
    return "scipy"


def _normalize_solver_options(raw_options, boozer_type):
    """Validate and normalize constructor options for a Boozer solve mode."""
    if "bfgs_method" in raw_options:
        raise ValueError(
            "BoozerSurfaceJAX option 'bfgs_method' was removed. "
            "Use 'optimizer_backend' with one of: scipy, ondevice."
        )

    allowed_option_keys = (
        _ALLOWED_OPTIONS_LS if boozer_type == "ls" else _ALLOWED_OPTIONS_EXACT
    )
    unknown_option_keys = sorted(set(raw_options) - allowed_option_keys)
    if unknown_option_keys:
        unknown_keys = ", ".join(repr(key) for key in unknown_option_keys)
        raise ValueError(f"Unknown BoozerSurfaceJAX option(s): {unknown_keys}.")

    optimizer_backend = raw_options.get("optimizer_backend")
    if (
        optimizer_backend is not None
        and optimizer_backend not in VALID_OPTIMIZER_BACKENDS
    ):
        raise ValueError("optimizer_backend must be one of: scipy, ondevice.")
    least_squares_algorithm = raw_options.get("least_squares_algorithm")
    if (
        least_squares_algorithm is not None
        and least_squares_algorithm not in VALID_LEAST_SQUARES_ALGORITHMS
    ):
        allowed = ", ".join(sorted(VALID_LEAST_SQUARES_ALGORITHMS))
        raise ValueError(f"least_squares_algorithm must be one of: {allowed}.")

    if boozer_type == "ls":
        effective_backend = raw_options.get(
            "optimizer_backend", _default_ls_optimizer_backend()
        )
        private_keys = sorted(set(raw_options) & _PRIVATE_OPTIMIZER_OPTIONS)
        if private_keys and effective_backend == "scipy":
            keys_str = ", ".join(repr(k) for k in private_keys)
            raise ValueError(
                f"Private optimizer option(s) {keys_str} require "
                "optimizer_backend='ondevice'."
            )

    normalized_options = dict(raw_options)
    if boozer_type == "ls":
        if "optimizer_backend" not in normalized_options:
            normalized_options["optimizer_backend"] = _default_ls_optimizer_backend()
        if "least_squares_algorithm" not in normalized_options:
            normalized_options["least_squares_algorithm"] = (
                default_least_squares_algorithm_for_backend(
                    normalized_options["optimizer_backend"]
                )
            )
    if boozer_type == "exact":
        normalized_options.pop("optimizer_backend", None)
    return normalized_options


class BoozerSurfaceJAX(Optimizable):
    """JAX-native Boozer surface solver.

    Mirrors the CPU ``BoozerSurface`` API — inherits ``Optimizable``,
    carries ``self.label``, and returns result dicts with ``vjp`` hooks.
    The object wrapper is intentionally stateful and should be treated as
    thread-confined: ``run_code()``, ``recompute_bell()``, and related helpers
    mutate ``self.res``, ``self.surface``, ``self.need_to_run_code``, and the
    cached grouped-coil data. Use ``run_code_traceable()`` plus immutable coil
    specs/arrays when you need a pure array contract for the target ondevice
    lane.

    This class therefore sits at the current architecture boundary:
    immutable grouped-coil specs feed the traceable JAX kernels, while the
    public wrapper still owns mutable solve state and flat decision-vector
    orchestration for compatibility with the existing outer optimizer stack.

    Args:
        biotsavart: ``BiotSavartJAX`` instance (or any adapter exposing
            ``coil_set_spec()`` for explicit immutable grouped-coil state).
        surface: CPU ``SurfaceXYZTensorFourier`` instance.
        label: An ``Optimizable`` that computes a flux surface label
            (e.g. ``Volume``, ``ToroidalFlux``).  Stored as ``self.label``
            for downstream consumers that call ``boozer_surface.label.J()``.
        targetlabel: target value for the label constraint.
        constraint_weight: penalty weight.  If ``None``, BoozerExact
            path is used; otherwise BoozerLS.
        options: dict of solver options (see ``_DEFAULT_OPTIONS_*``).
            For LS solves, the omitted ``optimizer_backend`` default follows the
            active simsopt backend contract: ``"scipy"`` on CPU/reference and
            ``"ondevice"`` on JAX backend modes. ``optimizer_backend="scipy"``
            remains the trusted CPU/reference lane and
            ``"ondevice"`` is the target on-device lane.
            ``least_squares_algorithm="quasi-newton"``
            preserves the historical BFGS/L-BFGS route; ``"lm"`` enables the
            residual-vector Levenberg-Marquardt route on supported backends.
    """

    supports_explicit_surface_warm_start = True

    def __init__(
        self,
        biotsavart,
        surface,
        label,
        targetlabel,
        constraint_weight=None,
        options=None,
    ):
        super().__init__(depends_on=[biotsavart])

        self.biotsavart = biotsavart
        self.surface = surface
        self.label = label
        self.targetlabel = float(targetlabel)
        self.constraint_weight = constraint_weight
        self.need_to_run_code = True
        self.res = None
        self._solver_generation = 0

        # Determine solver type
        self.boozer_type = "ls" if constraint_weight is not None else "exact"

        # Infer label_type from the label object.
        # Only Volume, Area, and ToroidalFlux have JAX-native implementations.
        label_cls = type(label).__name__
        if "Volume" in label_cls:
            self.label_type = "volume"
        elif "Area" in label_cls:
            self.label_type = "area"
        elif "ToroidalFlux" in label_cls:
            self.label_type = "toroidal_flux"
        else:
            raise ValueError(
                f"Unsupported label type {label_cls!r} for BoozerSurfaceJAX. "
                "Supported: Volume, Area, ToroidalFlux."
            )

        raw_options = _normalize_solver_options(
            dict(options or {}),
            self.boozer_type,
        )
        defaults = (
            _DEFAULT_OPTIONS_LS if self.boozer_type == "ls" else _DEFAULT_OPTIONS_EXACT
        )
        self.options = {**defaults, **raw_options}
        if self.boozer_type == "ls":
            if self.options["optimizer_backend"] not in VALID_OPTIMIZER_BACKENDS:
                raise ValueError(
                    "optimizer_backend must be one of: scipy, ondevice."
                )

        # --- Extract static data from CPU objects (one-time) ---
        s = surface
        self.mpol = s.mpol
        self.ntor = s.ntor
        self.nfp = s.nfp
        self.stellsym = s.stellsym
        self.quadpoints_phi = _as_jax_float64(s.quadpoints_phi)
        self.quadpoints_theta = _as_jax_float64(s.quadpoints_theta)
        surface_type_name = type(s).__name__
        if surface_type_name == "SurfaceRZFourier":
            self._surface_geometry_kind = "rzfourier"
        elif surface_type_name == "SurfaceXYZFourier":
            self._surface_geometry_kind = "xyzfourier"
        else:
            self._surface_geometry_kind = "generic"

        # Stellsym DOF scatter indices
        if self.stellsym:
            if self._surface_geometry_kind == "generic":
                self.scatter_indices = _generic_surface_scatter_operator(
                    self.mpol,
                    self.ntor,
                )
            else:
                self.scatter_indices = _as_jax_int32(
                    stellsym_scatter_indices(self.mpol, self.ntor)
                )
        else:
            self.scatter_indices = None

        # Toroidal flux phi index (first phi point by default)
        self.phi_idx = 0

        self._traceable_penalty_objective_cache = {}
        self._traceable_penalty_residual_cache = {}
        self._traceable_exact_residual_cache = {}
        self._reference_penalty_objective_cache = {}
        self._reference_penalty_residual_cache = {}

        # Coil data (extracted once, updated via _refresh_coil_data)
        self._refresh_coil_data()

    @property
    def _coil_arrays(self):
        """Coil geometry tuples ``(gammas, gammadashs, currents)`` without index lists."""
        return list(grouped_field_inputs_from_spec(self.coil_set_spec))

    @property
    def _coil_index_lists(self):
        """Per-group coil index lists from ``coil_groups``."""
        return list(grouped_coil_index_lists_from_spec(self.coil_set_spec))

    def recompute_bell(self, parent=None):
        """Mark solver as needing re-execution (dirty flag)."""
        self.need_to_run_code = True

    def _validate_none_G_precondition(self, G):
        if G is not None:
            return
        _require_fixed_currents_for_none_G(
            self.biotsavart,
            component="BoozerSurfaceJAX",
            coil_attrs=("_coils",),
        )

    def _refresh_coil_data(self):
        """Extract coil geometry and currents as JAX arrays.

        Groups coils by quadrature point count so that coils with
        different ``num_quad_points`` can coexist without crashing
        on array stacking.
        """
        self.coil_set_spec = _extract_grouped_coil_set_spec(self.biotsavart)
        self.coil_groups = list(grouped_field_data_from_spec(self.coil_set_spec))
        self.coil_currents = grouped_coil_currents_from_spec(self.coil_set_spec)
        self._reference_penalty_objective_cache.clear()
        self._reference_penalty_residual_cache.clear()

    def _emit_stage_callback(
        self,
        label: str,
        **extra: float | str | None,
    ) -> None:
        callback = self.options.get("stage_callback")
        if callback is not None:
            callback(label, **extra)

    def _solver_diagnostics_payload(
        self,
        result,
        *,
        gradient_key: str,
        residual_key: str | None = None,
    ) -> dict[str, float]:
        payload = {
            "objective": float(_host_scalar(result["fun"])),
        }
        gradient = result.get(gradient_key)
        if gradient is not None:
            payload["grad_inf"] = float(_host_inf_norm(gradient))
        if residual_key is not None:
            residual = result.get(residual_key)
            if residual is not None:
                payload["residual_inf"] = float(_host_inf_norm(residual))
        return payload

    def _make_solver_progress_callback(self, method: str):
        stage_callback = self.options.get("stage_callback")
        if stage_callback is None:
            return None

        def emit_progress(iteration: int, fun_value: float, grad_inf: float) -> None:
            if iteration <= 5 or iteration % 25 == 0:
                stage_callback(
                    "boozer_ls_progress",
                    iteration=float(iteration),
                    objective=float(fun_value),
                    grad_inf=float(grad_inf),
                    method=method,
                )

        return emit_progress

    def _make_newton_progress_callback(self):
        stage_callback = self.options.get("stage_callback")
        if stage_callback is None:
            return None

        def emit_progress(iteration: int, fun_value: float, grad_norm: float) -> None:
            stage_callback(
                "boozer_newton_progress",
                iteration=float(iteration),
                objective=float(fun_value),
                grad_norm=float(grad_norm),
            )

        return emit_progress

    def _resolve_newton_progress_callback(self, method: str):
        if method in _ONDEVICE_OPTIMIZER_METHODS:
            return None
        return self._make_newton_progress_callback()

    def _get_surface_dofs(self):
        """Get current surface DOFs as JAX array."""
        return _as_jax_float64(self.surface.get_dofs())

    def _set_surface_dofs(self, dofs_jax):
        """Write JAX DOFs back to CPU surface."""
        self.surface.set_dofs(_host_numpy(dofs_jax))

    def _pack_decision_vector(self, iota, G, sdofs=None):
        """Pack [surface_dofs, iota] or [surface_dofs, iota, G]."""
        if sdofs is None:
            sdofs = self._get_surface_dofs()
        if G is not None:
            return _concat_jax_float64(sdofs, [iota, G])
        return _concat_jax_float64(sdofs, [iota])

    def _make_penalty_optimizer_state(self, iota, G, *, sdofs=None):
        if sdofs is None:
            sdofs = self._get_surface_dofs()
        if G is None:
            return _BoozerPenaltyOptimizerState(
                surface_dofs=_as_jax_float64(sdofs),
                iota=_as_jax_float64(iota),
            )
        return _BoozerPenaltyOptimizerStateWithG(
            surface_dofs=_as_jax_float64(sdofs),
            iota=_as_jax_float64(iota),
            G=_as_jax_float64(G),
        )

    def _unpack_decision_vector(self, x, optimize_G):
        """Unpack decision vector → (sdofs, iota, G_or_None)."""
        sdofs, iota, G = _split_decision_vector_jax(x, optimize_G=optimize_G)
        if optimize_G:
            return _host_numpy(sdofs), float(_host_scalar(iota)), float(_host_scalar(G))
        return _host_numpy(sdofs), float(_host_scalar(iota)), None

    def _unpack_penalty_optimizer_state(self, x, optimize_G):
        optimizer_state = _as_boozer_penalty_optimizer_state(x, optimize_G=optimize_G)
        if optimize_G:
            return (
                _host_numpy(optimizer_state.surface_dofs),
                float(_host_scalar(optimizer_state.iota)),
                float(_host_scalar(optimizer_state.G)),
            )
        return (
            _host_numpy(optimizer_state.surface_dofs),
            float(_host_scalar(optimizer_state.iota)),
            None,
        )

    def _unpack_decision_vector_jax(
        self,
        x,
        optimize_G,
        coil_set_spec=None,
        coil_arrays=None,
    ):
        """JAX-array version of ``_unpack_decision_vector``."""
        sdofs, iota, G = _split_decision_vector_jax(x, optimize_G=optimize_G)
        if optimize_G:
            return sdofs, iota, G
        G = compute_G_from_currents(
            _grouped_coil_currents(
                coil_arrays=coil_arrays,
                coil_set_spec=_resolved_coil_set_spec(
                    self.coil_set_spec,
                    coil_arrays=coil_arrays,
                    coil_set_spec=coil_set_spec,
                ),
            )
        )
        return sdofs, iota, G

    def _make_penalty_objective_with(
        self,
        optimize_G,
        weight_inv_modB,
        constraint_weight=None,
        coil_set_spec=None,
        coil_arrays=None,
        *,
        hostify_inputs=True,
    ):
        """Build penalty objective with explicit overrides."""
        resolved_coil_set_spec = _resolved_coil_set_spec(
            self.coil_set_spec,
            coil_arrays=coil_arrays,
            coil_set_spec=coil_set_spec,
        )
        resolved_constraint_weight = self._resolve_constraint_weight(constraint_weight)
        if hostify_inputs:
            resolved_coil_set_spec = _hostify_tree(resolved_coil_set_spec)
            key = self._reference_penalty_cache_key(
                optimize_G,
                weight_inv_modB,
                resolved_constraint_weight,
                resolved_coil_set_spec,
            )
            objective_fn = self._reference_penalty_objective_cache.get(key)
            if objective_fn is None:
                objective_fn = _make_boozer_penalty_objective_closure(
                    coil_arrays=coil_arrays,
                    coil_set_spec=resolved_coil_set_spec,
                    quadpoints_phi=_hostify_tree(self.quadpoints_phi),
                    quadpoints_theta=_hostify_tree(self.quadpoints_theta),
                    mpol=self.mpol,
                    ntor=self.ntor,
                    nfp=self.nfp,
                    stellsym=self.stellsym,
                    scatter_indices=_hostify_tree(self.scatter_indices),
                    surface_kind=self._surface_geometry_kind,
                    targetlabel=self.targetlabel,
                    constraint_weight=resolved_constraint_weight,
                    label_type=self.label_type,
                    phi_idx=self.phi_idx,
                    optimize_G=optimize_G,
                    weight_inv_modB=weight_inv_modB,
                )
                objective_fn = _optimizer_jax._mark_cacheable_jit_value_and_grad(
                    objective_fn
                )
                self._reference_penalty_objective_cache[key] = objective_fn
            return objective_fn
        return _make_boozer_penalty_objective_closure(
            coil_arrays=coil_arrays,
            coil_set_spec=resolved_coil_set_spec,
            quadpoints_phi=self.quadpoints_phi,
            quadpoints_theta=self.quadpoints_theta,
            mpol=self.mpol,
            ntor=self.ntor,
            nfp=self.nfp,
            stellsym=self.stellsym,
            scatter_indices=self.scatter_indices,
            surface_kind=self._surface_geometry_kind,
            targetlabel=self.targetlabel,
            constraint_weight=resolved_constraint_weight,
            label_type=self.label_type,
            phi_idx=self.phi_idx,
            optimize_G=optimize_G,
            weight_inv_modB=weight_inv_modB,
        )

    def _make_penalty_residual_with(
        self,
        optimize_G,
        weight_inv_modB,
        constraint_weight=None,
        coil_set_spec=None,
        coil_arrays=None,
        *,
        hostify_inputs=True,
    ):
        """Build the LS residual-vector closure with explicit grouped-field inputs."""
        resolved_coil_set_spec = _resolved_coil_set_spec(
            self.coil_set_spec,
            coil_arrays=coil_arrays,
            coil_set_spec=coil_set_spec,
        )
        resolved_constraint_weight = self._resolve_constraint_weight(constraint_weight)
        if hostify_inputs:
            resolved_coil_set_spec = _hostify_tree(resolved_coil_set_spec)
            key = self._reference_penalty_cache_key(
                optimize_G,
                weight_inv_modB,
                resolved_constraint_weight,
                resolved_coil_set_spec,
            )
            residual_fn = self._reference_penalty_residual_cache.get(key)
            if residual_fn is None:
                residual_fn = _make_boozer_penalty_residual_closure(
                    coil_arrays=coil_arrays,
                    coil_set_spec=resolved_coil_set_spec,
                    constraint_weight=resolved_constraint_weight,
                    optimize_G=optimize_G,
                    weight_inv_modB=weight_inv_modB,
                    **self._traceable_surface_runtime_args(),
                )
                self._reference_penalty_residual_cache[key] = residual_fn
            return residual_fn
        return _make_boozer_penalty_residual_closure(
            coil_arrays=coil_arrays,
            coil_set_spec=resolved_coil_set_spec,
            quadpoints_phi=self.quadpoints_phi,
            quadpoints_theta=self.quadpoints_theta,
            mpol=self.mpol,
            ntor=self.ntor,
            nfp=self.nfp,
            stellsym=self.stellsym,
            scatter_indices=self.scatter_indices,
            surface_kind=self._surface_geometry_kind,
            targetlabel=self.targetlabel,
            constraint_weight=resolved_constraint_weight,
            label_type=self.label_type,
            phi_idx=self.phi_idx,
            optimize_G=optimize_G,
            weight_inv_modB=weight_inv_modB,
        )

    def _traceable_surface_signature(self):
        """Signature for metadata that becomes a traced constant in JAX closures."""
        return (
            int(self.mpol),
            int(self.ntor),
            int(self.nfp),
            bool(self.stellsym),
            str(self._surface_geometry_kind),
            float(self.targetlabel),
            str(self.label_type),
            int(self.phi_idx),
            _traceable_array_signature(self.quadpoints_phi),
            _traceable_array_signature(self.quadpoints_theta),
            _traceable_array_signature(self.scatter_indices),
        )

    def _traceable_surface_runtime_args(self):
        return {
            "quadpoints_phi": _hostify_tree(self.quadpoints_phi),
            "quadpoints_theta": _hostify_tree(self.quadpoints_theta),
            "mpol": self.mpol,
            "ntor": self.ntor,
            "nfp": self.nfp,
            "stellsym": self.stellsym,
            "scatter_indices": _hostify_tree(self.scatter_indices),
            "surface_kind": self._surface_geometry_kind,
            "targetlabel": self.targetlabel,
            "label_type": self.label_type,
            "phi_idx": self.phi_idx,
        }

    def _resolve_constraint_weight(self, constraint_weight):
        return (
            self.constraint_weight if constraint_weight is None else constraint_weight
        )

    def _reference_penalty_cache_key(
        self,
        optimize_G,
        weight_inv_modB,
        constraint_weight,
        coil_set_spec,
    ):
        return (
            bool(optimize_G),
            bool(weight_inv_modB),
            float(constraint_weight),
            self._traceable_surface_signature(),
            _runtime_cache_tree_signature(coil_set_spec),
        )

    def _traceable_penalty_cache_key(
        self,
        optimize_G,
        weight_inv_modB,
        constraint_weight=None,
    ):
        return (
            bool(optimize_G),
            bool(weight_inv_modB),
            float(self._resolve_constraint_weight(constraint_weight)),
            self._traceable_surface_signature(),
        )

    def _traceable_exact_cache_key(self, weight_inv_modB, mask_indices):
        return (
            bool(weight_inv_modB),
            self._traceable_surface_signature(),
            _traceable_array_signature(mask_indices),
        )

    def _get_traceable_penalty_objective(
        self,
        optimize_G,
        weight_inv_modB,
        constraint_weight=None,
    ):
        resolved_constraint_weight = self._resolve_constraint_weight(constraint_weight)
        key = self._traceable_penalty_cache_key(
            optimize_G,
            weight_inv_modB,
            resolved_constraint_weight,
        )
        objective_fn = self._traceable_penalty_objective_cache.get(key)
        if objective_fn is None:
            surface_args = self._traceable_surface_runtime_args()

            def objective_fn(x, coil_set_spec):
                return _boozer_penalty_objective(
                    x,
                    coil_set_spec=coil_set_spec,
                    constraint_weight=resolved_constraint_weight,
                    optimize_G=optimize_G,
                    weight_inv_modB=weight_inv_modB,
                    **surface_args,
                )

            self._traceable_penalty_objective_cache[key] = objective_fn
        return self._traceable_penalty_objective_cache[key]

    def _get_traceable_penalty_residual(
        self,
        optimize_G,
        weight_inv_modB,
        constraint_weight=None,
    ):
        resolved_constraint_weight = self._resolve_constraint_weight(constraint_weight)
        key = self._traceable_penalty_cache_key(
            optimize_G,
            weight_inv_modB,
            resolved_constraint_weight,
        )
        residual_fn = self._traceable_penalty_residual_cache.get(key)
        if residual_fn is None:
            surface_args = self._traceable_surface_runtime_args()

            def residual_fn(x, coil_set_spec):
                return _boozer_penalty_residual_vector(
                    x,
                    coil_set_spec=coil_set_spec,
                    constraint_weight=resolved_constraint_weight,
                    optimize_G=optimize_G,
                    weight_inv_modB=weight_inv_modB,
                    **surface_args,
                )

            self._traceable_penalty_residual_cache[key] = residual_fn
        return self._traceable_penalty_residual_cache[key]

    def _get_traceable_exact_residual(self, weight_inv_modB):
        mask_indices = self._compute_stellsym_mask_indices()
        key = self._traceable_exact_cache_key(weight_inv_modB, mask_indices)
        residual_fn = self._traceable_exact_residual_cache.get(key)
        if residual_fn is None:
            exact_residual = _select_exact_residual_fn(self.stellsym)
            surface_args = self._traceable_surface_runtime_args()
            host_mask_indices = _hostify_tree(mask_indices)

            def residual_fn(x, coil_set_spec):
                return exact_residual(
                    x,
                    coil_set_spec=coil_set_spec,
                    mask_indices=host_mask_indices,
                    weight_inv_modB=weight_inv_modB,
                    **surface_args,
                )

            self._traceable_exact_residual_cache[key] = residual_fn
        return self._traceable_exact_residual_cache[key]

    def run_code_traceable(self, coil_source, sdofs, iota, G):
        """Trace-safe pure-array inner solve for the ondevice target lane.

        Accepts a preferred immutable ``GroupedCoilSetSpec`` or the legacy
        grouped-array payload plus warm-start state, returns only JAX arrays /
        scalars, and never reads or writes ``self.res``, ``self.surface``, or
        ``self.need_to_run_code``.

        Supported modes:
        - LS Boozer solve on the on-device optimizer lane.
        - Exact Boozer Newton solve (backend-independent).
        """
        weight_inv_modB = self.options["weight_inv_modB"]
        coil_set_spec = grouped_coil_set_spec_from_source(coil_source)

        if self.boozer_type == "exact":
            G_exact = (
                G
                if G is not None
                else compute_G_from_currents(
                    grouped_coil_currents_from_spec(coil_set_spec)
                )
            )
            x0 = _concat_jax_float64(sdofs, [iota, G_exact])
            res_fn = self._get_traceable_exact_residual(weight_inv_modB)
            result = newton_exact_traceable(
                res_fn,
                x0,
                maxiter=self.options["newton_maxiter"],
                tol=self.options["newton_tol"],
                args=(coil_set_spec,),
                max_dense_jacobian_bytes=self.options["max_dense_jacobian_bytes"],
            )
            jacobian = result["jacobian"]
            jacobian_available = jacobian is not None
            finite = jnp.all(jnp.isfinite(result["x"])) & jnp.all(
                jnp.isfinite(result["residual"])
            )
            if jacobian_available:
                finite = finite & jnp.all(jnp.isfinite(jacobian))
                P, L, U = _traceable_plu_or_dummy(
                    jacobian,
                    finite=finite,
                )
                plu = (P, L, U)
            else:
                plu = None
            sdofs_exact, iota_exact, G_exact = self._unpack_decision_vector_jax(
                result["x"],
                True,
            )
            half = _as_runtime_float64(0.5, reference=result["residual"])
            jacobian_available_jax = jax.device_put(
                np.asarray(jacobian_available, dtype=np.bool_)
            )
            return {
                "x": result["x"],
                "sdofs": sdofs_exact,
                "iota": iota_exact,
                "G": G_exact,
                "fun": half * jnp.mean(jnp.square(result["residual"])),
                "residual": result["residual"],
                "jacobian": jacobian,
                "plu": plu,
                "nit": result["nit"],
                "success": result["success"] & finite & jacobian_available_jax,
                "type": "exact",
                "weight_inv_modB": weight_inv_modB,
                **_exact_newton_reporting_fields(result),
            }

        optimize_G = G is not None
        method = self._resolve_optimizer_method(optimize_G=optimize_G)
        if method not in _ONDEVICE_OPTIMIZER_METHODS:
            raise RuntimeError(
                "run_code_traceable() requires optimizer_backend='ondevice' for LS solves."
            )

        x0 = self._pack_decision_vector(iota, G, sdofs=_as_jax_float64(sdofs))
        if method == "lm-ondevice":
            residual_fn = self._get_traceable_penalty_residual(
                optimize_G,
                weight_inv_modB,
            )
            ls_state = levenberg_marquardt_traceable(
                residual_fn,
                x0,
                maxiter=self.options["bfgs_maxiter"],
                tol=self.options["bfgs_tol"],
                args=(coil_set_spec,),
            )
            x_ls = ls_state["x"]
        else:
            ls_obj_fn = self._make_penalty_objective_with(
                optimize_G,
                weight_inv_modB,
                coil_set_spec=coil_set_spec,
                hostify_inputs=False,
            )
            optimizer_options = self._collect_optimizer_options()

            if method == "bfgs-ondevice":
                ls_state = _optimizer_jax._minimize_bfgs_private(
                    ls_obj_fn,
                    x0,
                    maxiter=self.options["bfgs_maxiter"],
                    gtol=self.options["bfgs_tol"],
                    line_search_maxiter=int(
                        optimizer_options.get("line_search_maxiter", 10)
                    ),
                )
                x_ls = ls_state.x_k
            else:
                ls_state = _optimizer_jax._minimize_lbfgs_private(
                    ls_obj_fn,
                    x0,
                    maxiter=self.options["bfgs_maxiter"],
                    gtol=self.options["bfgs_tol"],
                    maxcor=int(optimizer_options.get("maxcor", 200)),
                    ftol=float(optimizer_options.get("ftol", 0.0)),
                    maxfun=optimizer_options.get("maxfun"),
                    maxgrad=optimizer_options.get("maxgrad"),
                    maxls=int(optimizer_options.get("maxls", 20)),
                )
                x_ls = ls_state.x_k

        obj_fn = self._get_traceable_penalty_objective(
            optimize_G,
            weight_inv_modB,
        )

        newton_result = self._run_newton_polish_for_method(
            method,
            obj_fn,
            x_ls,
            maxiter=self.options["newton_maxiter"],
            tol=self.options["newton_tol"],
            stab=self.options["newton_stab"],
            objective_args=(coil_set_spec,),
        )
        sdofs_out, iota_out, G_out = self._unpack_decision_vector_jax(
            newton_result["x"],
            optimize_G,
            coil_set_spec=coil_set_spec,
        )
        finite = (
            jnp.all(jnp.isfinite(newton_result["x"]))
            & jnp.all(jnp.isfinite(newton_result["grad"]))
            & jnp.all(jnp.isfinite(newton_result["hessian"]))
        )
        P, L, U = _traceable_plu_or_dummy(
            newton_result["hessian"],
            finite=finite,
        )
        return {
            "x": newton_result["x"],
            "sdofs": sdofs_out,
            "iota": iota_out,
            "G": G_out,
            "fun": newton_result["fun"],
            "grad": newton_result["grad"],
            "hessian": newton_result["hessian"],
            "plu": (P, L, U),
            "nit": newton_result["nit"],
            "success": newton_result["success"] & finite,
            "optimizer_method": method,
            "type": "ls",
            "weight_inv_modB": weight_inv_modB,
        }

    def _compute_residual_vector(
        self,
        sdofs,
        iota,
        G,
        weight_inv_modB,
        constraint_weight=None,
        coil_set_spec=None,
        coil_arrays=None,
    ):
        """Compute unscalarized penalty residual vector at given state.

        Reuses M3's ``boozer_residual_vector`` for the Boozer part,
        appends label and z-constraint residuals.

        Returns a JAX array matching CPU
        ``boozer_penalty_constraints(..., scalarize=False)``.
        """
        coil_set_spec = _resolved_coil_set_spec(
            self.coil_set_spec,
            coil_arrays=coil_arrays,
            coil_set_spec=coil_set_spec,
        )
        gamma, xphi, xtheta = _surface_geometry_from_dofs(
            sdofs,
            self.quadpoints_phi,
            self.quadpoints_theta,
            self.mpol,
            self.ntor,
            self.nfp,
            self.stellsym,
            self.scatter_indices,
            surface_kind=self._surface_geometry_kind,
        )
        nphi, ntheta = int(gamma.shape[0]), int(gamma.shape[1])
        points = gamma.reshape(-1, 3)
        B = _grouped_biot_savart_B_points(
            points,
            coil_arrays=coil_arrays,
            coil_set_spec=coil_set_spec,
        ).reshape(nphi, ntheta, 3)

        r_boozer_raw = boozer_residual_vector(G, iota, B, xphi, xtheta, weight_inv_modB)
        num_res = _as_jax_float64(3 * nphi * ntheta)
        r_boozer = r_boozer_raw / jnp.sqrt(num_res)

        constraint_weight = (
            self.constraint_weight if constraint_weight is None else constraint_weight
        )
        constraint_weight = constraint_weight if constraint_weight is not None else 1.0
        constraint_weight = _as_jax_float64(constraint_weight)
        label_value, gamma_axis_z = _compute_label_and_axis_z(
            gamma=gamma,
            xphi=xphi,
            xtheta=xtheta,
            points=points,
            label_type=self.label_type,
            phi_idx=self.phi_idx,
            coil_arrays=coil_arrays,
            coil_set_spec=coil_set_spec,
        )
        weight_sqrt = jnp.sqrt(constraint_weight)
        rl = weight_sqrt * (label_value - _as_jax_float64(self.targetlabel))
        rz = weight_sqrt * gamma_axis_z

        return _concat_jax_float64(r_boozer, [rl, rz])

    def _resolve_optimizer_method(self, limited_memory=None, *, optimize_G=True):
        """Resolve optimizer method string from options."""
        optimizer_backend = self.options["optimizer_backend"]
        if optimizer_backend not in VALID_OPTIMIZER_BACKENDS:
            raise ValueError("optimizer_backend must be one of: scipy, ondevice.")
        require_target_backend_x64(optimizer_backend)
        if optimizer_backend != "ondevice":
            backend_config = get_backend_config()
            if backend_config.backend == "jax":
                raise RuntimeError(
                    "BoozerSurfaceJAX cannot use "
                    f"optimizer_backend={optimizer_backend!r} on the LS "
                    "reference solver lane while simsopt backend mode "
                    f"{backend_config.mode!r} requires optimizer_backend='ondevice'. "
                    "Select optimizer_backend='ondevice' or switch to the "
                    "native_cpu reference backend."
                )
            raise_if_strict_jax_fallback(
                component="BoozerSurfaceJAX",
                detail=(
                    f"optimizer_backend={optimizer_backend!r} on the LS "
                    "reference solver lane"
                ),
            )
            warn_if_jax_fallback(
                component="BoozerSurfaceJAX",
                detail=(
                    f"optimizer_backend={optimizer_backend!r} on the LS "
                    "reference solver lane"
                ),
            )
        if limited_memory is None:
            limited_memory = self.options["limited_memory"]
        effective_limited_memory = bool(limited_memory)
        if optimizer_backend == "ondevice" and self.options.get(
            "force_ondevice_limited_memory", False
        ):
            effective_limited_memory = True
        least_squares_algorithm = self.options["least_squares_algorithm"]
        if (
            optimizer_backend == "ondevice"
            and least_squares_algorithm == "lm"
            and not optimize_G
        ):
            # The explicit-G full-state path is the on-device LM target lane.
            # The reduced fixed-G compatibility path remains more reliable on
            # the historical quasi-Newton formulation.
            least_squares_algorithm = "quasi-newton"
        if optimizer_backend == "ondevice":
            return resolve_target_least_squares_optimizer_method(
                limited_memory=effective_limited_memory,
                least_squares_algorithm=least_squares_algorithm,
            )
        return resolve_reference_least_squares_optimizer_method(
            limited_memory=effective_limited_memory,
            least_squares_algorithm=least_squares_algorithm,
        )

    def _collect_optimizer_options(self):
        """Gather optimizer-specific options from self.options."""
        return {
            k: self.options[k]
            for k in (
                "line_search_maxiter",
                "maxcor",
                "ftol",
                "maxfun",
                "maxgrad",
                "maxls",
            )
            if k in self.options
        }

    def _run_newton_polish_for_method(
        self,
        method,
        obj_fn,
        x0,
        *,
        maxiter,
        tol,
        stab,
        progress_callback=None,
        objective_args=(),
    ):
        """Run the Newton polish implementation for a resolved optimizer method."""
        if method in _ONDEVICE_OPTIMIZER_METHODS:
            return newton_polish_traceable(
                obj_fn,
                x0,
                maxiter=maxiter,
                tol=tol,
                stab=stab,
                progress_callback=progress_callback,
                args=objective_args,
            )
        if objective_args:
            raise ValueError(
                "Newton objective args are only supported on the ondevice traceable path."
            )
        return newton_polish(
            obj_fn,
            x0,
            maxiter=maxiter,
            tol=tol,
            stab=stab,
            progress_callback=progress_callback,
        )

    def minimize_boozer_penalty_constraints_LBFGS(
        self,
        constraint_weight=1.0,
        iota=0.0,
        G=None,
        tol=None,
        maxiter=None,
        verbose=None,
        limited_memory=False,
        weight_inv_modB=None,
    ):
        """Least-squares first stage of the LS solve. Matches CPU public API."""
        if not self.need_to_run_code:
            return self.res
        tol = tol if tol is not None else self.options["bfgs_tol"]
        maxiter = maxiter if maxiter is not None else self.options["bfgs_maxiter"]
        verbose = verbose if verbose is not None else self.options["verbose"]
        weight_inv_modB = (
            weight_inv_modB
            if weight_inv_modB is not None
            else self.options["weight_inv_modB"]
        )

        optimize_G = G is not None
        s = self.surface
        x0 = self._pack_decision_vector(iota, G)
        method = self._resolve_optimizer_method(
            limited_memory=limited_memory,
            optimize_G=optimize_G,
        )
        progress_callback = self._make_solver_progress_callback(method)
        if method in {"lm", "lm-ondevice"}:
            residual_fn = self._make_penalty_residual_with(
                optimize_G,
                weight_inv_modB,
                constraint_weight,
            )
            least_squares_runner = (
                target_least_squares if method == "lm-ondevice" else reference_least_squares
            )
            result = least_squares_runner(
                residual_fn,
                x0,
                method=method,
                tol=tol,
                maxiter=maxiter,
                progress_callback=progress_callback,
            )
        else:
            obj_fn = self._make_penalty_objective_with(
                optimize_G,
                weight_inv_modB,
                constraint_weight,
            )
            minimize_runner = (
                target_minimize if method.endswith("-ondevice") else reference_minimize
            )
            result = minimize_runner(
                obj_fn,
                x0,
                method=method,
                tol=tol,
                maxiter=maxiter,
                options=self._collect_optimizer_options(),
                progress_callback=progress_callback,
            )

        sdofs_final, iota_out, G_out = self._unpack_penalty_optimizer_state(
            result.x, optimize_G
        )
        self._set_surface_dofs(sdofs_final)

        gradient = _host_numpy(
            _boozer_penalty_optimizer_state_to_vector(
                result.jac,
                optimize_G=optimize_G,
            )
        )

        resdict = {
            "fun": float(_host_scalar(result.fun)),
            "gradient": gradient,
            "iter": int(_host_scalar(result.nit)),
            "info": result,
            "success": bool(_host_scalar(result.success)),
            "G": G_out,
            "s": s,
            "iota": iota_out,
            "optimizer_method": method,
            "weight_inv_modB": weight_inv_modB,
            "type": "ls",
        }
        self.res = resdict
        self.need_to_run_code = False

        if verbose:
            print(
                f"{method} solve - "
                f"success={resdict['success']}  iter={resdict['iter']}, "
                f"iota={iota_out:.16f}, ||grad||_inf="
                f"{_host_inf_norm(resdict['gradient']):.3e}",
                flush=True,
            )
        return resdict

    def minimize_boozer_penalty_constraints_newton(
        self,
        constraint_weight=1.0,
        iota=0.0,
        G=None,
        tol=None,
        maxiter=None,
        stab=0.0,
        verbose=None,
        weight_inv_modB=None,
    ):
        """Newton polish stage of the LS solve. Matches CPU public API."""
        if not self.need_to_run_code:
            return self.res
        tol = tol if tol is not None else self.options["newton_tol"]
        maxiter = maxiter if maxiter is not None else self.options["newton_maxiter"]
        verbose = verbose if verbose is not None else self.options["verbose"]
        weight_inv_modB = (
            weight_inv_modB
            if weight_inv_modB is not None
            else self.options["weight_inv_modB"]
        )

        optimize_G = G is not None
        G_provided = optimize_G
        s = self.surface
        x0 = self._pack_decision_vector(iota, G)
        obj_fn = self._make_penalty_objective_with(
            optimize_G, weight_inv_modB, constraint_weight
        )

        method = self._resolve_optimizer_method(optimize_G=optimize_G)
        result = self._run_newton_polish_for_method(
            method,
            obj_fn,
            x0,
            maxiter=maxiter,
            tol=tol,
            stab=stab,
            progress_callback=self._resolve_newton_progress_callback(method),
        )

        sdofs_final, iota_out, G_out = self._unpack_decision_vector(
            result["x"], optimize_G
        )

        if (
            not _host_all_finite(result["x"])
            or not _host_all_finite(result["grad"])
            or not _host_all_finite(result["hessian"])
        ):
            solve_generation = _advance_solver_generation(self)
            res = {
                "residual": None,
                "jacobian": None,
                "hessian": None,
                "iter": int(_host_scalar(result["nit"], dtype=np.int64)),
                "success": False,
                "G": G_out,
                "s": s,
                "iota": iota_out,
                "PLU": None,
                "vjp": None,
                "vjp_groups": None,
                "type": "ls",
                "solve_generation": solve_generation,
                "weight_inv_modB": weight_inv_modB,
                "fun": float(_host_scalar(result["fun"])),
            }
            self.res = res
            self.need_to_run_code = False
            return res

        self._set_surface_dofs(sdofs_final)
        H = result["hessian"]
        P, L, U = jax.scipy.linalg.lu(H)

        G_for_res = (
            G_out
            if G_out is not None
            else float(compute_G_from_currents(self.coil_currents))
        )
        residual_vec = self._compute_residual_vector(
            sdofs_final,
            iota_out,
            G_for_res,
            weight_inv_modB=weight_inv_modB,
        )
        solve_generation = _advance_solver_generation(self)
        vjp_callback = _prepare_result_callback(
            partial(_boozer_ls_coil_vjp, weight_inv_modB=weight_inv_modB),
            booz_surf=self,
            solve_generation=solve_generation,
            callback_name="vjp",
            G_provided=G_provided,
            freshness_guard=True,
        )
        vjp_groups_callback = _prepare_result_callback(
            _build_ls_group_vjp_callback(
                self,
                iota_out,
                G_out,
                solve_generation=solve_generation,
                weight_inv_modB=weight_inv_modB,
            ),
            booz_surf=self,
            solve_generation=solve_generation,
            callback_name="vjp_groups",
            G_provided=G_provided,
            freshness_guard=True,
        )

        res = {
            "residual": residual_vec,
            "jacobian": result["grad"],
            "hessian": H,
            "iter": int(_host_scalar(result["nit"], dtype=np.int64)),
            "success": bool(_host_scalar(result["success"])),
            "G": G_out,
            "s": s,
            "iota": iota_out,
            "PLU": (P, L, U),
            "vjp": vjp_callback,
            "vjp_groups": vjp_groups_callback,
            "type": "ls",
            "solve_generation": solve_generation,
            "weight_inv_modB": weight_inv_modB,
            "fun": float(_host_scalar(result["fun"])),
        }
        self.res = res
        self.need_to_run_code = False

        if verbose:
            grad_norm = float(np.linalg.norm(_host_numpy(res["jacobian"])))
            print(
                f"NEWTON solve - success={res['success']}  "
                f"iter={res['iter']}, iota={iota_out:.16f}, "
                f"||grad||={grad_norm:.3e}",
                flush=True,
            )
        return res

    def _make_exact_residual_with(
        self,
        mask_indices,
        coil_arrays=None,
        coil_set_spec=None,
        *,
        hostify_inputs=True,
    ):
        """Build the exact residual function with explicit grouped-field inputs."""
        residual_fn = _select_exact_residual_fn(self.stellsym)
        resolved_coil_set_spec = _resolved_coil_set_spec(
            self.coil_set_spec,
            coil_arrays=coil_arrays,
            coil_set_spec=coil_set_spec,
        )
        if hostify_inputs:
            resolved_coil_set_spec = _hostify_tree(resolved_coil_set_spec)
        return partial(
            residual_fn,
            coil_arrays=coil_arrays,
            coil_set_spec=resolved_coil_set_spec,
            quadpoints_phi=(
                _hostify_tree(self.quadpoints_phi)
                if hostify_inputs
                else self.quadpoints_phi
            ),
            quadpoints_theta=(
                _hostify_tree(self.quadpoints_theta)
                if hostify_inputs
                else self.quadpoints_theta
            ),
            mpol=self.mpol,
            ntor=self.ntor,
            nfp=self.nfp,
            stellsym=self.stellsym,
            scatter_indices=(
                _hostify_tree(self.scatter_indices)
                if hostify_inputs
                else self.scatter_indices
            ),
            surface_kind=self._surface_geometry_kind,
            targetlabel=self.targetlabel,
            label_type=self.label_type,
            phi_idx=self.phi_idx,
            mask_indices=mask_indices,
            weight_inv_modB=self.options["weight_inv_modB"],
        )

    def _make_exact_residual(self, mask_indices):
        """Build the JIT-compiled exact residual function."""
        return self._make_exact_residual_with(mask_indices)

    def _compute_stellsym_mask_indices(self):
        """Compute the integer mask indices for the exact residual.

        Extracts the boolean stellsym mask from the CPU surface object
        and converts to integer indices for JAX fancy indexing.
        """
        s = self.surface
        m = s.get_stellsym_mask()
        mask = np.repeat(m[..., None], 3, axis=2)
        if s.stellsym:
            mask[0, 0, 0] = False
        return _as_jax_int32(np.flatnonzero(mask))

    def solve_residual_equation_exactly_newton(
        self,
        tol=None,
        maxiter=None,
        iota=0.0,
        G=None,
        verbose=None,
    ):
        """Solve the Boozer residual system exactly via Newton's method.

        Public API matching CPU ``BoozerSurface.solve_residual_equation_exactly_newton()``.

        Args:
            tol: residual norm tolerance. Defaults to options['newton_tol'].
            maxiter: maximum Newton iterations. Defaults to options['newton_maxiter'].
            iota: initial guess for rotational transform.
            G: initial guess for G (None → compute from coil currents).
            verbose: print convergence info.

        Returns:
            dict with 'residual', 'fun', 'jacobian', 'iter', 'success', 'G',
            's', 'iota', 'PLU', 'mask', 'type', 'vjp', 'weight_inv_modB',
            'message', 'failure_category', 'failure_stage',
            'jacobian_materialized',
            'dense_jacobian_shape', 'dense_jacobian_bytes',
            'max_dense_jacobian_bytes'.
            Exact mode enforces options['max_dense_jacobian_bytes'] before the
            final dense Jacobian/PLU materialization step.
        """
        if not self.need_to_run_code:
            return self.res

        s = self.surface
        G_provided = G is not None
        try:
            from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier

            if not isinstance(s, SurfaceXYZTensorFourier):
                raise RuntimeError(
                    "Exact solution of Boozer Surfaces only supported for "
                    "SurfaceXYZTensorFourier"
                )
        except (ImportError, ModuleNotFoundError):
            # simsoptpp unavailable — skip type check (tests with mock surfaces)
            pass

        tol = tol if tol is not None else self.options["newton_tol"]
        maxiter = maxiter if maxiter is not None else self.options["newton_maxiter"]
        verbose = verbose if verbose is not None else self.options["verbose"]

        if G is None:
            G = float(compute_G_from_currents(self.coil_currents))

        sdofs = self._get_surface_dofs()
        x0 = _concat_jax_float64(sdofs, [iota, G])

        mask_indices = self._compute_stellsym_mask_indices()
        res_fn = self._make_exact_residual(mask_indices)

        result = newton_exact(
            res_fn,
            x0,
            maxiter=maxiter,
            tol=tol,
            max_dense_jacobian_bytes=self.options["max_dense_jacobian_bytes"],
        )

        x_final = result["x"]
        exact_residual = res_fn(x_final)
        sdofs_final, iota_final_jax, G_final_jax = _split_decision_vector_jax(
            x_final,
            optimize_G=True,
        )
        iota_final = float(_host_scalar(iota_final_jax))
        G_final = float(_host_scalar(G_final_jax))
        jacobian = result["jacobian"]
        jacobian_available = jacobian is not None
        exact_reporting = _exact_newton_reporting_fields(result)
        materialization_message = exact_reporting["message"]

        if (
            not bool(_host_scalar(result["success"]))
            or not _host_all_finite(x_final)
            or not _host_all_finite(exact_residual)
            or not jacobian_available
            or not _host_all_finite(jacobian)
        ):
            solve_generation = _advance_solver_generation(self)
            res = {
                "residual": None,
                "fun": float(0.5 * np.mean(np.square(_host_numpy(exact_residual)))),
                "jacobian": None,
                "iter": int(_host_scalar(result["nit"], dtype=np.int64)),
                "success": False,
                "G": G_final,
                "s": s,
                "iota": iota_final,
                "PLU": None,
                "mask": None,
                "type": "exact",
                "vjp": None,
                "vjp_groups": None,
                "solve_generation": solve_generation,
                "weight_inv_modB": self.options["weight_inv_modB"],
                **exact_reporting,
            }
            self.res = res
            self.need_to_run_code = False
            if verbose and materialization_message is not None:
                print(materialization_message, flush=True)
            return res

        self._set_surface_dofs(sdofs_final)
        J = jacobian
        P, L, U = jax.scipy.linalg.lu(J)

        nphi = len(self.quadpoints_phi)
        ntheta = len(self.quadpoints_theta)

        # Reconstruct raw (unmasked) Boozer residual for CPU-contract parity.
        gamma_final, xphi_final, xtheta_final = _surface_geometry_from_dofs(
            sdofs_final,
            self.quadpoints_phi,
            self.quadpoints_theta,
            self.mpol,
            self.ntor,
            self.nfp,
            self.stellsym,
            self.scatter_indices,
            surface_kind=self._surface_geometry_kind,
        )
        B_final = grouped_biot_savart_B_from_spec(
            gamma_final.reshape(-1, 3),
            self.coil_set_spec,
        ).reshape(nphi, ntheta, 3)
        r_raw = boozer_residual_vector(
            G_final,
            iota_final,
            B_final,
            xphi_final,
            xtheta_final,
            self.options["weight_inv_modB"],
        )

        bool_mask = np.zeros(3 * nphi * ntheta, dtype=bool)
        bool_mask[np.asarray(mask_indices)] = True
        solve_generation = _advance_solver_generation(self)
        vjp_callback = _prepare_result_callback(
            _boozer_exact_coil_vjp,
            booz_surf=self,
            solve_generation=solve_generation,
            callback_name="vjp",
            G_provided=G_provided,
            freshness_guard=False,
        )
        vjp_groups_callback = _prepare_result_callback(
            _build_exact_group_vjp_callback(
                self,
                iota_final,
                G_final,
            ),
            booz_surf=self,
            solve_generation=solve_generation,
            callback_name="vjp_groups",
            G_provided=G_provided,
            freshness_guard=False,
        )

        res = {
            "residual": r_raw,
            "fun": float(0.5 * np.mean(np.square(_host_numpy(exact_residual)))),
            "jacobian": J,
            "iter": int(_host_scalar(result["nit"], dtype=np.int64)),
            "success": bool(_host_scalar(result["success"])),
            "G": G_final,
            "s": s,
            "iota": iota_final,
            "PLU": (P, L, U),
            "mask": bool_mask,
            "type": "exact",
            "vjp": vjp_callback,
            "vjp_groups": vjp_groups_callback,
            "solve_generation": solve_generation,
            "weight_inv_modB": self.options["weight_inv_modB"],
            **exact_reporting,
        }
        self.res = res
        self.need_to_run_code = False

        if verbose:
            res_norm = _host_inf_norm(res["residual"])
            print(
                f"NEWTON solve - success={res['success']}  "
                f"iter={res['iter']}, iota={iota_final:.16f}, "
                f"||residual||_inf={res_norm:.3e}",
                flush=True,
            )
        return res

    def run_code(self, iota, G=None, *, sdofs=None):
        """Run the Boozer surface solver (LS or exact depending on config).

        Mirrors ``BoozerSurface.run_code()`` API.

        Args:
            iota: initial guess for rotational transform.
            G: initial guess for G (None → compute from coil currents,
               and coil currents must be fixed).
            sdofs: explicit surface DOFs for the initial guess. If None,
                reads from ``self.surface``.  When provided, syncs
                ``self.surface`` to ``sdofs`` before the solve so that
                failure paths leave the surface in a consistent state.

        Returns:
            dict with solver results, or None if solver was not dirty.
        """
        if not self.need_to_run_code:
            return

        # Sync surface DOFs when caller provides explicit warm-start.
        # This ensures failure paths (which skip _set_surface_dofs) leave
        # self.surface in a state consistent with the warm-start DOFs,
        # matching the old pre-solve ``surface.x = sdofs`` behavior.
        if sdofs is not None:
            self._set_surface_dofs(sdofs)

        # When G=None the gradient treats currents as constants,
        # so coil currents must be fixed to avoid silent gradient errors.
        self._validate_none_G_precondition(G)

        # Refresh coil data in case coils changed
        self._refresh_coil_data()

        if self.boozer_type == "exact":
            res = self.solve_residual_equation_exactly_newton(
                iota=iota,
                G=G,
                tol=self.options["newton_tol"],
                maxiter=self.options["newton_maxiter"],
                verbose=self.options["verbose"],
            )
            return res

        # BoozerLS: BFGS + Newton polish
        assert self.constraint_weight is not None
        first_stage_method = self._resolve_optimizer_method(optimize_G=G is not None)
        self._emit_stage_callback(
            "before_boozer_lbfgs",
            method=str(first_stage_method),
        )
        ls_res = self.minimize_boozer_penalty_constraints_LBFGS(
            constraint_weight=self.constraint_weight,
            iota=iota,
            G=G,
            tol=self.options["bfgs_tol"],
            maxiter=self.options["bfgs_maxiter"],
            verbose=self.options["verbose"],
            limited_memory=self.options["limited_memory"],
            weight_inv_modB=self.options["weight_inv_modB"],
        )
        self._emit_stage_callback(
            "after_boozer_lbfgs",
            solve_success=("true" if bool(ls_res["success"]) else "false"),
            iterations=float(ls_res["iter"]),
            method=str(ls_res["optimizer_method"]),
            **self._solver_diagnostics_payload(
                ls_res,
                gradient_key="gradient",
            ),
        )
        iota_out, G_out = ls_res["iota"], ls_res["G"]

        # Polish with Newton
        self.need_to_run_code = True
        self._emit_stage_callback(
            "before_boozer_newton",
            method="newton-polish",
            ls_method=str(ls_res["optimizer_method"]),
        )
        res = self.minimize_boozer_penalty_constraints_newton(
            constraint_weight=self.constraint_weight,
            iota=iota_out,
            G=G_out,
            verbose=self.options["verbose"],
            tol=self.options["newton_tol"],
            maxiter=self.options["newton_maxiter"],
            stab=self.options["newton_stab"],
            weight_inv_modB=self.options["weight_inv_modB"],
        )
        res["optimizer_method"] = ls_res["optimizer_method"]
        self._emit_stage_callback(
            "after_boozer_newton",
            solve_success=("true" if bool(res["success"]) else "false"),
            iterations=float(res["iter"]),
            **self._solver_diagnostics_payload(
                res,
                gradient_key="jacobian",
                residual_key="residual",
            ),
        )
        return res

    def run_code_functional(self, coil_arrays, sdofs, iota, G):
        """Pure functional form of ``run_code()`` — no self mutation.

        Accepts explicit arguments instead of reading from self state.
        Does NOT set ``self.res``, ``self.need_to_run_code``, or
        ``self.surface`` DOFs.

        This is the legacy-result compatibility wrapper over the trace-safe
        array solve in ``run_code_traceable()``. The inner computation stays on
        the pure-array target lane; this wrapper only repackages that result
        into the historical ``run_code()``-shaped dict with ``s=None`` and
        CPU-only adjoint hooks omitted.

        Differences from the stateful ``run_code()`` result dict:

        * ``sdofs`` — solved surface DOFs as a JAX array (new key).
        * ``s`` — ``None``.  The functional path does not produce a
          CPU surface object; use ``sdofs`` instead.
        * ``vjp``, ``vjp_groups`` — ``None``.  The CPU VJP callbacks
          read from ``self`` state at call/construction time and are
          structurally incompatible with the functional contract.
          Downstream traceable consumers should use JAX autodiff
          through ``coil_arrays → objective`` instead.

        Args:
            coil_arrays: list of ``(gammas, gammadashs, currents)`` tuples.
            sdofs: surface DOFs as a 1-D array.
            iota: initial guess for rotational transform.
            G: initial guess for G.

        Returns:
            dict with solver results.  See docstring for keys that
            differ from the stateful ``run_code()`` path.
        """
        traceable_result = self.run_code_traceable(
            coil_arrays,
            _as_jax_float64(sdofs),
            iota,
            G,
        )
        success = bool(np.asarray(traceable_result["success"]))
        result_type = traceable_result["type"]
        legacy_result = {
            "iter": int(np.asarray(traceable_result["nit"])),
            "success": success,
            "G": traceable_result["G"],
            "s": None,
            "sdofs": traceable_result["sdofs"],
            "iota": traceable_result["iota"],
            "PLU": None,
            "vjp": None,
            "vjp_groups": None,
            "type": result_type,
            "weight_inv_modB": traceable_result["weight_inv_modB"],
            "fun": float(_host_scalar(traceable_result["fun"])),
            "message": traceable_result.get("message"),
        }

        if result_type == "exact":
            mask = None
            if success:
                mask_indices = np.asarray(self._compute_stellsym_mask_indices())
                mask = np.zeros(
                    3 * len(self.quadpoints_phi) * len(self.quadpoints_theta),
                    dtype=bool,
                )
                mask[mask_indices] = True
            legacy_result.update(
                {
                    "residual": None if not success else traceable_result["residual"],
                    "jacobian": None if not success else traceable_result["jacobian"],
                    "mask": mask,
                }
            )
            if success:
                legacy_result["PLU"] = tuple(traceable_result["plu"])
            return legacy_result

        legacy_result["optimizer_method"] = traceable_result["optimizer_method"]
        if not success:
            legacy_result.update(
                {
                    "residual": None,
                    "jacobian": None,
                    "hessian": None,
                }
            )
            return legacy_result

        legacy_result.update(
            {
                "residual": self._compute_residual_vector(
                    traceable_result["sdofs"],
                    traceable_result["iota"],
                    traceable_result["G"],
                    weight_inv_modB=traceable_result["weight_inv_modB"],
                    coil_arrays=coil_arrays,
                ),
                "jacobian": traceable_result["grad"],
                "hessian": traceable_result["hessian"],
                "PLU": tuple(traceable_result["plu"]),
            }
        )
        return legacy_result
