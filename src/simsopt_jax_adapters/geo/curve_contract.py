"""Shared JAX curve-method contract helpers."""

import jax.numpy as jnp

from simsopt_jax.core.specs import make_optimizable_dof_map_spec


def _curve_uses_full_dofs(curve):
    return getattr(curve, "_jax_curve_dof_mode", "local") == "full"


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
