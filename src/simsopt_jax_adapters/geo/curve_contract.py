"""Shared JAX curve-method contract helpers."""

import numpy as np
from jax import jacfwd, jvp, vjp
import jax.numpy as jnp

from simsopt.geo.jit import jit
from simsopt_jax.core._math_utils import as_jax_float64 as _as_jax_float64
from simsopt_jax.core.specs import make_optimizable_dof_map_spec


def _as_runtime_jax_float64(value):
    return _as_jax_float64(value)


def _quadpoint_jvp(geometry_fn):
    return lambda dofs, quadpoints: jvp(
        lambda qpts: geometry_fn(dofs, qpts),
        (quadpoints,),
        (jnp.ones_like(quadpoints),),
    )[1]


def _curve_uses_full_dofs(curve):
    return getattr(curve, "_jax_curve_dof_mode", "local") == "full"


def _install_curve_jax_contract(curve, gamma_pure, *, dof_mode="local"):
    points = np.asarray(curve.quadpoints)

    curve._jax_curve_dof_mode = dof_mode
    curve.gamma_pure = gamma_pure
    curve.gamma_jax = jit(lambda dofs: gamma_pure(dofs, points))
    curve.gamma_impl_jax = jit(lambda dofs, quadpoints: gamma_pure(dofs, quadpoints))

    curve.gammadash_pure = _quadpoint_jvp(gamma_pure)
    curve.gammadash_jax = jit(lambda dofs: curve.gammadash_pure(dofs, points))

    curve.gammadashdash_pure = _quadpoint_jvp(curve.gammadash_pure)
    curve.gammadashdash_jax = jit(lambda dofs: curve.gammadashdash_pure(dofs, points))

    curve.gammadashdashdash_pure = _quadpoint_jvp(curve.gammadashdash_pure)
    curve.gammadashdashdash_jax = jit(
        lambda dofs: curve.gammadashdashdash_pure(dofs, points)
    )

    curve.dgamma_by_dcoeff_jax = jit(jacfwd(curve.gamma_jax))
    curve.dgamma_by_dcoeff_vjp_jax = jit(
        lambda dofs, cotangent: vjp(curve.gamma_jax, dofs)[1](cotangent)[0]
    )
    curve.dgammadash_by_dcoeff_jax = jit(jacfwd(curve.gammadash_jax))
    curve.dgammadash_by_dcoeff_vjp_jax = jit(
        lambda dofs, cotangent: vjp(curve.gammadash_jax, dofs)[1](cotangent)[0]
    )
    curve.dgammadashdash_by_dcoeff_jax = jit(jacfwd(curve.gammadashdash_jax))
    curve.dgammadashdash_by_dcoeff_vjp_jax = jit(
        lambda dofs, cotangent: vjp(curve.gammadashdash_jax, dofs)[1](cotangent)[0]
    )
    curve.dgammadashdashdash_by_dcoeff_jax = jit(
        jacfwd(curve.gammadashdashdash_jax)
    )
    curve.dgammadashdashdash_by_dcoeff_vjp_jax = jit(
        lambda dofs, cotangent: vjp(curve.gammadashdashdash_jax, dofs)[1](
            cotangent
        )[0]
    )


def _extract_full_subgraph_dofs(owner, subcurve, owner_dofs):
    owner_dofs = jnp.asarray(owner_dofs, dtype=jnp.float64)
    if owner is subcurve:
        return owner_dofs

    subcurve_full_dofs = jnp.asarray(subcurve.full_x, dtype=jnp.float64)
    for opt, (sub_start, sub_end) in subcurve._full_dof_indices.items():
        owner_start, owner_end = owner._full_dof_indices[opt]
        subcurve_full_dofs = subcurve_full_dofs.at[sub_start:sub_end].set(
            owner_dofs[owner_start:owner_end]
        )
    return subcurve_full_dofs


def _curve_jax_arg_from_full_dofs(owner, subcurve, owner_dofs):
    subcurve_full_dofs = _extract_full_subgraph_dofs(owner, subcurve, owner_dofs)
    if _curve_uses_full_dofs(subcurve):
        return subcurve_full_dofs

    start, end = subcurve._full_dof_indices[subcurve]
    return subcurve_full_dofs[start:end]


def _optimizable_local_full_dofs_from_full_dofs(owner, opt, owner_dofs):
    owner_dofs = jnp.asarray(owner_dofs, dtype=jnp.float64)
    start, end = owner._full_dof_indices[opt]
    return owner_dofs[start:end]


def _optimizable_dof_map_components(owner, opt):
    template_full_dofs = jnp.asarray(opt.full_x, dtype=jnp.float64)
    owner_segments = tuple(
        (
            int(owner._full_dof_indices[dep_opt][0]),
            int(owner._full_dof_indices[dep_opt][1]),
            int(sub_start),
            int(sub_end),
        )
        for dep_opt, (sub_start, sub_end) in opt._full_dof_indices.items()
    )
    if _curve_uses_full_dofs(opt):
        input_mode = "full"
        input_start = 0
        input_end = int(template_full_dofs.shape[0])
    else:
        input_mode = "local"
        input_start, input_end = opt._full_dof_indices[opt]
    return (
        template_full_dofs,
        owner_segments,
        input_mode,
        int(input_start),
        int(input_end),
    )


def _optimizable_dof_map_spec(owner, opt):
    (
        template_full_dofs,
        owner_segments,
        input_mode,
        input_start,
        input_end,
    ) = _optimizable_dof_map_components(owner, opt)
    return make_optimizable_dof_map_spec(
        template_full_dofs=template_full_dofs,
        owner_segments=owner_segments,
        input_mode=input_mode,
        input_start=input_start,
        input_end=input_end,
    )


def _curve_jax_eval_from_arg(curve, method_name, curve_dofs, surf_dofs=None):
    curve_dofs = jnp.asarray(curve_dofs, dtype=jnp.float64)
    method = getattr(curve, method_name)
    if _curve_uses_full_dofs(curve):
        return method(curve_dofs)

    surf = getattr(curve, "surf", None)
    if surf is not None and surf.dof_size > 0:
        if surf_dofs is None:
            surf_dofs = surf.get_dofs()
        surf_dofs = jnp.asarray(surf_dofs, dtype=jnp.float64)
        return method(curve_dofs, surf_dofs)
    return method(curve_dofs)
