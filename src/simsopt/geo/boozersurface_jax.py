"""
Lane-aware JAX Boozer surface solver.

The public/reference lane still permits host-side SciPy minimization via
``optimizer_backend="scipy"``. The private optimizer lane adds two more
roles:

- ``optimizer_backend="hybrid"``: transitional migration path
- ``optimizer_backend="ondevice"``: target on-device backend for the eventual
  full-GPU workflow

This module owns the LS/exact solver routing contract. Only the
``ondevice`` backend is intended to represent the eventual target optimizer
lane, not a claim that the full workflow is already production-complete.

Architecture (per M0 contract §5-§6):
  - Adapter pattern: ``BoozerSurfaceJAX`` inherits ``Optimizable`` and
    mirrors the CPU ``BoozerSurface`` public API.
  - The outer ``Optimizable`` dependency graph and ``need_to_run_code``
    dirty-flag semantics are preserved.
  - The reference lane may still cross the host/device boundary inside the LS
    optimizer loop; removing that is part of the on-device migration.

Builds on M3's composed derivative path:
  - ``_surface_geometry_from_dofs()`` for surface DOFs → geometry (SSOT)
  - ``boozer_residual_scalar()`` for the forward residual
  - ``boozer_residual_vector()`` for the exact Newton residual vector
  - ``boozer_residual_coil_vjp()`` for outer-path coil sensitivities
  - ``jax.grad`` / ``jax.hessian`` / ``jax.jacfwd`` for all derivatives
"""

from functools import partial
import warnings

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg

from ..backend import raise_if_strict_jax_fallback, warn_if_jax_fallback
from ..backend.runtime import register_backend_cache_clear
from ..jax_core.specs import make_coil_spec

try:
    from simsopt._core.optimizable import Optimizable
except (ImportError, ModuleNotFoundError):
    # Fallback when simsoptpp is unavailable (standalone JAX tests).
    # In production with simsopt fully installed, the real Optimizable is used.
    class Optimizable:  # type: ignore[no-redef]
        def __init__(self, *args, depends_on=None, **kwargs):
            pass


from .surface_fourier_jax import stellsym_scatter_indices
from ..jax_core.field import (
    grouped_biot_savart_A_from_inputs,
    grouped_biot_savart_A_from_spec,
    grouped_biot_savart_B_from_inputs,
    grouped_biot_savart_B_from_spec,
    grouped_coil_currents_from_inputs,
    grouped_coil_currents_from_spec,
    grouped_coil_index_lists_from_spec,
    grouped_coil_set_spec_from_coil_specs,
    grouped_coil_set_spec_from_grouped_data,
    grouped_coil_set_spec_from_source,
    grouped_field_data_from_spec,
    grouped_field_inputs_from_spec,
)
from .boozer_residual_jax import (
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
    VALID_OPTIMIZER_BACKENDS,
    jax_minimize,
    newton_exact,
    newton_exact_traceable,
    newton_polish,
    newton_polish_traceable,
    require_target_backend_x64,
    resolve_optimizer_backend_method,
)

__all__ = ["BoozerSurfaceJAX"]

_GROUPED_EXTRACTOR_FALLBACK_DETAIL = (
    "_extract_coil_data_grouped() in _refresh_coil_data()"
)
_COILS_LIST_FALLBACK_DETAIL = "_coils list extraction in _refresh_coil_data()"
_WARNED_HIDDEN_GROUPED_FALLBACK_DETAILS: set[str] = set()


def _clear_hidden_grouped_fallback_warning_cache() -> None:
    _WARNED_HIDDEN_GROUPED_FALLBACK_DETAILS.clear()


register_backend_cache_clear(_clear_hidden_grouped_fallback_warning_cache)


def _raise_if_strict_hidden_grouped_coil_spec_fallback(detail: str) -> None:
    raise_if_strict_jax_fallback(
        component="BoozerSurfaceJAX",
        detail=f"the hidden grouped-coil spec compatibility fallback via {detail}",
    )


def _warn_hidden_grouped_coil_spec_fallback(detail: str) -> None:
    if detail in _WARNED_HIDDEN_GROUPED_FALLBACK_DETAILS:
        return
    _WARNED_HIDDEN_GROUPED_FALLBACK_DETAILS.add(detail)
    warnings.warn(
        "BoozerSurfaceJAX is using a hidden grouped-coil compatibility fallback "
        f"via {detail}. This path snapshots compatibility data from the live "
        "coil graph and should be treated as a legacy adapter seam.",
        RuntimeWarning,
        stacklevel=3,
    )


def _replace_group_coil_array(coil_arrays, group_index, group_array):
    grouped_arrays = list(coil_arrays)
    grouped_arrays[group_index] = group_array
    return grouped_arrays


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

    ``BiotSavartJAX`` provides a dedicated grouped extractor. Fallback callers,
    including compatibility shims, may only expose a hidden ``_coils`` list.
    """
    coil_set_spec = getattr(biotsavart, "coil_set_spec", None)
    if coil_set_spec is not None:
        return coil_set_spec()

    grouped_extractor = getattr(biotsavart, "_extract_coil_data_grouped", None)
    if grouped_extractor is not None:
        _raise_if_strict_hidden_grouped_coil_spec_fallback(
            _GROUPED_EXTRACTOR_FALLBACK_DETAIL
        )
        _warn_hidden_grouped_coil_spec_fallback(_GROUPED_EXTRACTOR_FALLBACK_DETAIL)
        return grouped_coil_set_spec_from_grouped_data(grouped_extractor())

    coils = getattr(biotsavart, "_coils", None)
    if coils is None:
        raise AttributeError(
            "BoozerSurfaceJAX requires a biotsavart object that provides either "
            "coil_set_spec(), _extract_coil_data_grouped(), or a _coils list."
        )

    _raise_if_strict_hidden_grouped_coil_spec_fallback(_COILS_LIST_FALLBACK_DETAIL)
    _warn_hidden_grouped_coil_spec_fallback(_COILS_LIST_FALLBACK_DETAIL)
    return grouped_coil_set_spec_from_coil_specs(
        tuple(_coil_spec_from_hidden_fallback_coil(coil) for coil in coils)
    )


def _coil_spec_from_hidden_fallback_coil(coil):
    coil_to_spec = getattr(coil, "to_spec", None)
    if coil_to_spec is not None:
        return coil_to_spec()

    curve = getattr(coil, "curve", None)
    current = getattr(coil, "current", None)
    curve_to_spec = getattr(curve, "to_spec", None) if curve is not None else None
    current_to_spec = getattr(current, "to_spec", None) if current is not None else None
    if curve_to_spec is None or current_to_spec is None:
        raise AttributeError(
            "BoozerSurfaceJAX hidden _coils compatibility fallback requires "
            "coils that expose immutable spec builders via coil.to_spec() "
            "or curve.to_spec()/current.to_spec()."
        )

    return make_coil_spec(
        curve=curve_to_spec(),
        current=current_to_spec(),
    )


def _coil_count_from_spec_or_coils(biotsavart, coil_set_spec):
    coils = getattr(biotsavart, "_coils", None)
    if coils is not None:
        return len(coils)
    return (
        max(
            (max(indices) for indices in coil_set_spec.coil_index_lists()),
            default=-1,
        )
        + 1
    )


def _coil_currents_are_fixed(biotsavart):
    coils = getattr(biotsavart, "_coils", None)
    if coils is None:
        return True
    return all(coil.current.dofs.all_fixed() for coil in coils)


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
    normal = jnp.cross(xphi, xtheta)
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
    return toroidal_flux_jax(A[phi_idx], xtheta[phi_idx], ntheta)


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

    The decision vector is ``x = [surface_dofs, iota]`` (optimize_G=False)
    or ``x = [surface_dofs, iota, G]`` (optimize_G=True).
    """
    if optimize_G:
        sdofs, iota, G = x[:-2], x[-2], x[-1]
    else:
        sdofs, iota = x[:-1], x[-1]
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

    label_val = _compute_label(
        label_type,
        gamma,
        xphi,
        xtheta,
        phi_idx,
        points,
        coil_arrays=coil_arrays,
        coil_set_spec=coil_set_spec,
    )

    J_label = 0.5 * constraint_weight * (label_val - targetlabel) ** 2
    J_z = 0.5 * constraint_weight * gamma[0, 0, 2] ** 2

    return J_boozer + J_label + J_z


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
    sdofs, iota, G = x[:-2], x[-2], x[-1]

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

    label_val = _compute_label(
        label_type,
        gamma,
        xphi,
        xtheta,
        phi_idx,
        points,
        coil_arrays=coil_arrays,
        coil_set_spec=coil_set_spec,
    )
    r_label = label_val - targetlabel

    if include_axis_constraint:
        residual_tail = jnp.array([r_label, gamma[0, 0, 2]])
    else:
        residual_tail = jnp.array([r_label])
    return jnp.concatenate([r_masked, residual_tail])


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
    x = jnp.concatenate([sdofs, jnp.array([iota, G])])
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
    x = jnp.concatenate([sdofs, jnp.array([iota, G])])
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
        weight_inv_modB=weight_inv_modB,
    )(
        lm,
        booz_surf,
        iota,
        G,
    )


def _build_ls_group_vjp_callback(booz_surf, iota, G, weight_inv_modB=True):
    """Build stable LS group runners for repeated streaming VJPs."""
    x, optimize_G = _ls_decision_vector(booz_surf, iota, G)

    coil_arrays = booz_surf._coil_arrays
    coil_indices = booz_surf._coil_index_lists

    group_runners = tuple(
        _make_ls_group_runner(
            x,
            coil_arrays,
            booz_surf,
            optimize_G,
            weight_inv_modB,
            group_index,
        )
        for group_index in range(len(coil_arrays))
    )

    def vjp_groups(lm, _booz_surf, _iota, _G):
        for group_runner, group_array, group_index_list in zip(
            group_runners,
            coil_arrays,
            coil_indices,
        ):

            def directional_of_group(updated_group_array):
                return group_runner(updated_group_array, lm)

            yield jax.grad(directional_of_group)(group_array), group_index_list

    return vjp_groups


def _make_ls_group_runner(
    x,
    coil_arrays,
    booz_surf,
    optimize_G,
    weight_inv_modB,
    group_index,
):
    def directional_of_group(group_array, tangent):
        return _group_penalty_directional_objective(
            x,
            tangent,
            _replace_group_coil_array(
                coil_arrays,
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
        x = jnp.concatenate([sdofs, jnp.array([iota, G])])
    else:
        x = jnp.concatenate([sdofs, jnp.array([iota])])
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
    coil_arrays,
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
            coil_arrays=coil_arrays,
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


def _make_boozer_penalty_objective_closure(
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
    return lambda xx: _boozer_penalty_objective(
        xx,
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
        constraint_weight=constraint_weight,
        label_type=label_type,
        phi_idx=phi_idx,
        optimize_G=optimize_G,
        weight_inv_modB=weight_inv_modB,
    )


def _directional_derivative(objective, x, tangent):
    _, directional = jax.jvp(objective, (x,), (tangent,))
    return directional


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
}

_DEFAULT_OPTIONS_EXACT = {
    "verbose": True,
    "newton_tol": 1e-13,
    "newton_maxiter": 40,
    "weight_inv_modB": False,
}

# Options only meaningful for private optimizer backends (hybrid, ondevice).
_PRIVATE_OPTIMIZER_OPTIONS = frozenset(
    {
        "force_ondevice_limited_memory",
        "hybrid_scipy_maxiter",
        "line_search_maxiter",
        "maxgrad",
    }
)

# Options shared by the public SciPy L-BFGS lane and the private L-BFGS lanes.
_LBFGS_TUNING_OPTIONS = frozenset({"maxcor", "ftol", "maxfun", "maxls"})

# Callback options accepted by all backends.
_CALLBACK_OPTIONS = frozenset({"stage_callback", "progress_callback"})
_ONDEVICE_OPTIMIZER_METHODS = frozenset({"bfgs-ondevice", "lbfgs-ondevice"})

_ALLOWED_OPTIONS_LS = (
    frozenset(_DEFAULT_OPTIONS_LS)
    | _PRIVATE_OPTIMIZER_OPTIONS
    | _LBFGS_TUNING_OPTIONS
    | _CALLBACK_OPTIONS
)
_ALLOWED_OPTIONS_EXACT = frozenset(_DEFAULT_OPTIONS_EXACT) | {
    "optimizer_backend",
    "stage_callback",
}


def _normalize_solver_options(raw_options, boozer_type):
    """Validate and normalize constructor options for a Boozer solve mode."""
    if "bfgs_method" in raw_options:
        raise ValueError(
            "BoozerSurfaceJAX option 'bfgs_method' was removed. "
            "Use 'optimizer_backend' with one of: scipy, hybrid, ondevice."
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
        raise ValueError("optimizer_backend must be one of: scipy, hybrid, ondevice.")

    if boozer_type == "ls":
        effective_backend = raw_options.get("optimizer_backend", "scipy")
        private_keys = sorted(set(raw_options) & _PRIVATE_OPTIMIZER_OPTIONS)
        if private_keys and effective_backend == "scipy":
            keys_str = ", ".join(repr(k) for k in private_keys)
            raise ValueError(
                f"Private optimizer option(s) {keys_str} require "
                "optimizer_backend='hybrid' or 'ondevice'."
            )

        lbfgs_keys = sorted(set(raw_options) & _LBFGS_TUNING_OPTIONS)
        if lbfgs_keys and effective_backend == "hybrid":
            keys_str = ", ".join(repr(k) for k in lbfgs_keys)
            raise ValueError(
                f"L-BFGS tuning option(s) {keys_str} are unsupported for "
                "optimizer_backend='hybrid'."
            )

    normalized_options = dict(raw_options)
    if boozer_type == "exact":
        normalized_options.pop("optimizer_backend", None)
    return normalized_options


class BoozerSurfaceJAX(Optimizable):
    """JAX-native Boozer surface solver.

    Mirrors the CPU ``BoozerSurface`` API — inherits ``Optimizable``,
    carries ``self.label``, and returns result dicts with ``vjp`` hooks.

    Args:
        biotsavart: ``BiotSavartJAX`` instance (or any object with
            ``_coils`` attribute providing curve geometry and currents).
        surface: CPU ``SurfaceXYZTensorFourier`` instance.
        label: An ``Optimizable`` that computes a flux surface label
            (e.g. ``Volume``, ``ToroidalFlux``).  Stored as ``self.label``
            for downstream consumers that call ``boozer_surface.label.J()``.
        targetlabel: target value for the label constraint.
        constraint_weight: penalty weight.  If ``None``, BoozerExact
            path is used; otherwise BoozerLS.
        options: dict of solver options (see ``_DEFAULT_OPTIONS_*``).
            For LS solves, ``optimizer_backend="scipy"`` is the trusted
            reference lane, ``"hybrid"`` is the transitional migration lane,
            and ``"ondevice"`` is the target on-device lane for the eventual
            full-GPU workflow.
    """

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
                    "optimizer_backend must be one of: scipy, hybrid, ondevice."
                )

        # --- Extract static data from CPU objects (one-time) ---
        s = surface
        self.mpol = s.mpol
        self.ntor = s.ntor
        self.nfp = s.nfp
        self.stellsym = s.stellsym
        self.quadpoints_phi = jnp.asarray(s.quadpoints_phi, dtype=jnp.float64)
        self.quadpoints_theta = jnp.asarray(s.quadpoints_theta, dtype=jnp.float64)
        surface_type_name = type(s).__name__
        if surface_type_name == "SurfaceRZFourier":
            self._surface_geometry_kind = "rzfourier"
        elif surface_type_name == "SurfaceXYZFourier":
            self._surface_geometry_kind = "xyzfourier"
        else:
            self._surface_geometry_kind = "generic"

        # Stellsym DOF scatter indices
        if self.stellsym:
            self.scatter_indices = jnp.asarray(
                stellsym_scatter_indices(self.mpol, self.ntor)
            )
        else:
            self.scatter_indices = None

        # Toroidal flux phi index (first phi point by default)
        self.phi_idx = 0

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

    def _refresh_coil_data(self):
        """Extract coil geometry and currents as JAX arrays.

        Groups coils by quadrature point count so that coils with
        different ``num_quad_points`` can coexist without crashing
        on array stacking.
        """
        self.coil_set_spec = _extract_grouped_coil_set_spec(self.biotsavart)
        self.coil_groups = list(grouped_field_data_from_spec(self.coil_set_spec))
        self.coil_currents = grouped_coil_currents_from_spec(
            self.coil_set_spec,
            coil_count=_coil_count_from_spec_or_coils(
                self.biotsavart,
                self.coil_set_spec,
            ),
        )

    def _emit_stage_callback(
        self,
        label: str,
        **extra: float | str | None,
    ) -> None:
        callback = self.options.get("stage_callback")
        if callback is not None:
            callback(label, **extra)

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
        return jnp.asarray(self.surface.get_dofs(), dtype=jnp.float64)

    def _set_surface_dofs(self, dofs_jax):
        """Write JAX DOFs back to CPU surface."""
        self.surface.set_dofs(np.asarray(dofs_jax))

    def _pack_decision_vector(self, iota, G, sdofs=None):
        """Pack [surface_dofs, iota] or [surface_dofs, iota, G]."""
        if sdofs is None:
            sdofs = self._get_surface_dofs()
        if G is not None:
            return jnp.concatenate([sdofs, jnp.array([iota, G])])
        return jnp.concatenate([sdofs, jnp.array([iota])])

    def _unpack_decision_vector(self, x, optimize_G):
        """Unpack decision vector → (sdofs, iota, G_or_None)."""
        if optimize_G:
            return x[:-2], float(x[-2]), float(x[-1])
        return x[:-1], float(x[-1]), None

    def _unpack_decision_vector_jax(
        self,
        x,
        optimize_G,
        coil_set_spec=None,
        coil_arrays=None,
    ):
        """JAX-array version of ``_unpack_decision_vector``."""
        if optimize_G:
            return x[:-2], x[-2], x[-1]
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
        return x[:-1], x[-1], G

    def _make_penalty_objective_with(
        self,
        optimize_G,
        weight_inv_modB,
        constraint_weight=None,
        coil_set_spec=None,
        coil_arrays=None,
    ):
        """Build penalty objective with explicit overrides."""
        return partial(
            _boozer_penalty_objective,
            coil_arrays=coil_arrays,
            coil_set_spec=_resolved_coil_set_spec(
                self.coil_set_spec,
                coil_arrays=coil_arrays,
                coil_set_spec=coil_set_spec,
            ),
            quadpoints_phi=self.quadpoints_phi,
            quadpoints_theta=self.quadpoints_theta,
            mpol=self.mpol,
            ntor=self.ntor,
            nfp=self.nfp,
            stellsym=self.stellsym,
            scatter_indices=self.scatter_indices,
            surface_kind=self._surface_geometry_kind,
            targetlabel=self.targetlabel,
            constraint_weight=constraint_weight
            if constraint_weight is not None
            else self.constraint_weight,
            label_type=self.label_type,
            phi_idx=self.phi_idx,
            optimize_G=optimize_G,
            weight_inv_modB=weight_inv_modB,
        )

    def run_code_traceable(self, coil_source, sdofs, iota, G):
        """Trace-safe pure-array inner solve for the Section 3 target path.

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
            x0 = jnp.concatenate(
                [
                    jnp.asarray(sdofs, dtype=jnp.float64),
                    jnp.array([iota, G_exact], dtype=jnp.float64),
                ]
            )
            mask_indices = self._compute_stellsym_mask_indices()
            res_fn = self._make_exact_residual_with(
                mask_indices,
                coil_set_spec=coil_set_spec,
            )
            result = newton_exact_traceable(
                res_fn,
                x0,
                maxiter=self.options["newton_maxiter"],
                tol=self.options["newton_tol"],
            )
            P, L, U = jax.scipy.linalg.lu(result["jacobian"])
            finite = (
                jnp.all(jnp.isfinite(result["x"]))
                & jnp.all(jnp.isfinite(result["residual"]))
                & jnp.all(jnp.isfinite(result["jacobian"]))
            )
            return {
                "x": result["x"],
                "sdofs": result["x"][:-2],
                "iota": result["x"][-2],
                "G": result["x"][-1],
                "fun": 0.5 * jnp.mean(jnp.square(result["residual"])),
                "residual": result["residual"],
                "jacobian": result["jacobian"],
                "plu": (P, L, U),
                "nit": result["nit"],
                "success": result["success"] & finite,
                "type": "exact",
                "weight_inv_modB": weight_inv_modB,
            }

        method = self._resolve_optimizer_method()
        if method not in {"bfgs-ondevice", "lbfgs-ondevice"}:
            raise RuntimeError(
                "run_code_traceable() requires optimizer_backend='ondevice' for LS solves."
            )

        optimize_G = G is not None
        obj_fn = self._make_penalty_objective_with(
            optimize_G,
            weight_inv_modB,
            coil_set_spec=coil_set_spec,
        )
        x0 = self._pack_decision_vector(
            iota, G, sdofs=jnp.asarray(sdofs, dtype=jnp.float64)
        )
        optimizer_options = self._collect_optimizer_options()

        if method == "bfgs-ondevice":
            ls_state = _optimizer_jax._minimize_bfgs_private(
                obj_fn,
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
                obj_fn,
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

        newton_result = self._run_newton_polish_for_method(
            method,
            obj_fn,
            x_ls,
            maxiter=self.options["newton_maxiter"],
            tol=self.options["newton_tol"],
            stab=self.options["newton_stab"],
        )
        P, L, U = jax.scipy.linalg.lu(newton_result["hessian"])
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

        r_boozer_raw = boozer_residual_vector(
            G, iota, B, xphi, xtheta, self.options["weight_inv_modB"]
        )
        num_res = 3 * nphi * ntheta
        r_boozer = r_boozer_raw / jnp.sqrt(num_res)

        cw = self.constraint_weight if self.constraint_weight is not None else 1.0
        lab = float(
            _compute_label(
                self.label_type,
                gamma,
                xphi,
                xtheta,
                self.phi_idx,
                points,
                coil_arrays=coil_arrays,
                coil_set_spec=coil_set_spec,
            )
        )
        rl = jnp.sqrt(cw) * (lab - self.targetlabel)
        rz = jnp.sqrt(cw) * gamma[0, 0, 2]

        return jnp.concatenate([r_boozer, jnp.array([rl, rz])])

    def _resolve_optimizer_method(self, limited_memory=None):
        """Resolve optimizer method string from options."""
        optimizer_backend = self.options["optimizer_backend"]
        require_target_backend_x64(optimizer_backend)
        if optimizer_backend != "ondevice":
            raise_if_strict_jax_fallback(
                component="BoozerSurfaceJAX",
                detail=(
                    f"optimizer_backend={optimizer_backend!r} on the LS "
                    "reference/transitional solver lane"
                ),
            )
            warn_if_jax_fallback(
                component="BoozerSurfaceJAX",
                detail=(
                    f"optimizer_backend={optimizer_backend!r} on the LS "
                    "reference/transitional solver lane"
                ),
            )
        if limited_memory is None:
            limited_memory = self.options["limited_memory"]
        effective_limited_memory = bool(limited_memory)
        if optimizer_backend == "ondevice" and self.options.get(
            "force_ondevice_limited_memory", False
        ):
            effective_limited_memory = True
        return resolve_optimizer_backend_method(
            optimizer_backend,
            limited_memory=effective_limited_memory,
        )

    def _collect_optimizer_options(self):
        """Gather optimizer-specific options from self.options."""
        return {
            k: self.options[k]
            for k in (
                "hybrid_scipy_maxiter",
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
        """BFGS/L-BFGS stage of the LS solve. Matches CPU public API."""
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
        obj_fn = self._make_penalty_objective_with(
            optimize_G, weight_inv_modB, constraint_weight
        )

        method = self._resolve_optimizer_method(limited_memory=limited_memory)
        optimizer_options = self._collect_optimizer_options()

        result = jax_minimize(
            obj_fn,
            x0,
            method=method,
            tol=tol,
            maxiter=maxiter,
            options=optimizer_options,
            progress_callback=self._make_solver_progress_callback(method),
        )

        sdofs_final, iota_out, G_out = self._unpack_decision_vector(
            result.x, optimize_G
        )
        self._set_surface_dofs(sdofs_final)

        resdict = {
            "fun": float(result.fun),
            "gradient": result.jac,
            "iter": int(result.nit),
            "info": result,
            "success": bool(result.success),
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
                f"{float(jnp.max(jnp.abs(resdict['gradient']))):.3e}",
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
        s = self.surface
        x0 = self._pack_decision_vector(iota, G)
        obj_fn = self._make_penalty_objective_with(
            optimize_G, weight_inv_modB, constraint_weight
        )

        method = self._resolve_optimizer_method()
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
            not np.all(np.isfinite(np.asarray(result["x"])))
            or not np.all(np.isfinite(np.asarray(result["grad"])))
            or not np.all(np.isfinite(np.asarray(result["hessian"])))
        ):
            res = {
                "residual": None,
                "jacobian": None,
                "hessian": None,
                "iter": result["nit"],
                "success": False,
                "G": G_out,
                "s": s,
                "iota": iota_out,
                "PLU": None,
                "vjp": None,
                "vjp_groups": None,
                "type": "ls",
                "weight_inv_modB": weight_inv_modB,
                "fun": float(np.asarray(result["fun"])),
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
        residual_vec = self._compute_residual_vector(sdofs_final, iota_out, G_for_res)

        res = {
            "residual": residual_vec,
            "jacobian": result["grad"],
            "hessian": H,
            "iter": result["nit"],
            "success": bool(result["success"]),
            "G": G_out,
            "s": s,
            "iota": iota_out,
            "PLU": (P, L, U),
            "vjp": partial(_boozer_ls_coil_vjp, weight_inv_modB=weight_inv_modB),
            "vjp_groups": _build_ls_group_vjp_callback(
                self,
                iota_out,
                G_out,
                weight_inv_modB=weight_inv_modB,
            ),
            "type": "ls",
            "weight_inv_modB": weight_inv_modB,
            "fun": float(result["fun"]),
        }
        self.res = res
        self.need_to_run_code = False

        if verbose:
            grad_norm = float(jnp.linalg.norm(res["jacobian"]))
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
    ):
        """Build the exact residual function with explicit grouped-field inputs."""
        residual_fn = _select_exact_residual_fn(self.stellsym)
        return partial(
            residual_fn,
            coil_arrays=coil_arrays,
            coil_set_spec=_resolved_coil_set_spec(
                self.coil_set_spec,
                coil_arrays=coil_arrays,
                coil_set_spec=coil_set_spec,
            ),
            quadpoints_phi=self.quadpoints_phi,
            quadpoints_theta=self.quadpoints_theta,
            mpol=self.mpol,
            ntor=self.ntor,
            nfp=self.nfp,
            stellsym=self.stellsym,
            scatter_indices=self.scatter_indices,
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
        return jnp.asarray(np.flatnonzero(mask), dtype=jnp.int32)

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
            's', 'iota', 'PLU', 'mask', 'type', 'vjp', 'weight_inv_modB'.
        """
        if not self.need_to_run_code:
            return self.res

        s = self.surface
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
        x0 = jnp.concatenate([sdofs, jnp.array([iota, G])])

        mask_indices = self._compute_stellsym_mask_indices()
        res_fn = self._make_exact_residual(mask_indices)

        result = newton_exact(res_fn, x0, maxiter=maxiter, tol=tol)

        x_final = result["x"]
        exact_residual = res_fn(x_final)
        sdofs_final = x_final[:-2]
        iota_final = float(x_final[-2])
        G_final = float(x_final[-1])

        if (
            not bool(result["success"])
            or not np.all(np.isfinite(np.asarray(x_final)))
            or not np.all(np.isfinite(np.asarray(exact_residual)))
            or not np.all(np.isfinite(np.asarray(result["jacobian"])))
        ):
            res = {
                "residual": None,
                "fun": float(0.5 * np.mean(np.square(np.asarray(exact_residual)))),
                "jacobian": None,
                "iter": result["nit"],
                "success": False,
                "G": G_final,
                "s": s,
                "iota": iota_final,
                "PLU": None,
                "mask": None,
                "type": "exact",
                "vjp": None,
                "vjp_groups": None,
                "weight_inv_modB": self.options["weight_inv_modB"],
            }
            self.res = res
            self.need_to_run_code = False
            return res

        self._set_surface_dofs(sdofs_final)
        J = result["jacobian"]
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

        res = {
            "residual": r_raw,
            "fun": float(0.5 * jnp.mean(jnp.square(exact_residual))),
            "jacobian": J,
            "iter": result["nit"],
            "success": bool(result["success"]),
            "G": G_final,
            "s": s,
            "iota": iota_final,
            "PLU": (P, L, U),
            "mask": bool_mask,
            "type": "exact",
            "vjp": _boozer_exact_coil_vjp,
            "vjp_groups": _build_exact_group_vjp_callback(
                self,
                iota_final,
                G_final,
            ),
            "weight_inv_modB": self.options["weight_inv_modB"],
        }
        self.res = res
        self.need_to_run_code = False

        if verbose:
            res_norm = float(jnp.max(jnp.abs(res["residual"])))
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
        if G is None:
            assert _coil_currents_are_fixed(self.biotsavart), (
                "Coil currents must be fixed when G=None"
            )

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
        )
        return res

    def run_code_functional(self, coil_arrays, sdofs, iota, G):
        """Pure functional form of ``run_code()`` — no self mutation.

        Accepts explicit arguments instead of reading from self state.
        Does NOT set ``self.res``, ``self.need_to_run_code``, or
        ``self.surface`` DOFs.

        This is the pure-functional prerequisite for a JAX-traceable
        outer objective.  It eliminates self-mutation so that a future
        ``make_traceable_objective`` wrapper can close over this method
        and differentiate through the solve.  **This method is not
        itself JIT/grad-traceable** — it still uses ``float()`` casts,
        ``np.asarray`` conversions, and Python ``if`` on solver outputs.
        Making those trace-safe is deferred to the traceable-objective
        implementation (Section 3 items 2-7).

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
        optimize_G = G is not None
        weight_inv_modB = self.options["weight_inv_modB"]

        # Pack from explicit sdofs (not self.surface)
        sdofs_jax = jnp.asarray(sdofs, dtype=jnp.float64)
        x0 = self._pack_decision_vector(iota, G, sdofs=sdofs_jax)
        obj_fn = self._make_penalty_objective_with(
            optimize_G,
            weight_inv_modB,
            coil_arrays=coil_arrays,
        )
        method = self._resolve_optimizer_method()
        optimizer_options = self._collect_optimizer_options()

        # LBFGS → Newton polish
        ls_result = jax_minimize(
            obj_fn,
            x0,
            method=method,
            tol=self.options["bfgs_tol"],
            maxiter=self.options["bfgs_maxiter"],
            options=optimizer_options,
        )
        newton_result = self._run_newton_polish_for_method(
            method,
            obj_fn,
            ls_result.x,
            maxiter=self.options["newton_maxiter"],
            tol=self.options["newton_tol"],
            stab=self.options["newton_stab"],
        )

        sdofs_final, iota_out, G_out = self._unpack_decision_vector(
            newton_result["x"], optimize_G
        )

        if (
            not np.all(np.isfinite(np.asarray(newton_result["x"])))
            or not np.all(np.isfinite(np.asarray(newton_result["grad"])))
            or not np.all(np.isfinite(np.asarray(newton_result["hessian"])))
        ):
            return {
                "residual": None,
                "jacobian": None,
                "hessian": None,
                "iter": newton_result["nit"],
                "success": False,
                "G": G_out,
                "s": None,
                "sdofs": sdofs_final,
                "iota": iota_out,
                "PLU": None,
                "vjp": None,
                "vjp_groups": None,
                "type": "ls",
                "weight_inv_modB": weight_inv_modB,
                "fun": float(np.asarray(newton_result["fun"])),
                "optimizer_method": method,
            }

        H = newton_result["hessian"]
        P, L, U = jax.scipy.linalg.lu(H)

        G_for_res = (
            G_out
            if G_out is not None
            else float(
                compute_G_from_currents(jnp.concatenate([c for _, _, c in coil_arrays]))
            )
        )
        residual_vec = self._compute_residual_vector(
            sdofs_final,
            iota_out,
            G_for_res,
            coil_arrays=coil_arrays,
        )

        return {
            "residual": residual_vec,
            "jacobian": newton_result["grad"],
            "hessian": H,
            "iter": newton_result["nit"],
            "success": bool(newton_result["success"]),
            "G": G_out,
            "s": None,
            "sdofs": sdofs_final,
            "iota": iota_out,
            "PLU": (P, L, U),
            "vjp": None,
            "vjp_groups": None,
            "type": "ls",
            "weight_inv_modB": weight_inv_modB,
            "fun": float(newton_result["fun"]),
            "optimizer_method": method,
        }
