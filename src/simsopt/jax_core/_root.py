"""Small JAX root-finding kernels shared by MHD and geometry ports."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

__all__ = [
    "newton_scan_fixed_iters",
    "newton_with_implicit_vjp",
]


def _newton_step(
    residual: Callable[[jax.Array], jax.Array],
    jacobian_fn: Callable[[jax.Array], jax.Array],
    x: jax.Array,
) -> jax.Array:
    residual_value = residual(x)
    jacobian = jacobian_fn(x)
    jacobian_matrix = jnp.reshape(jacobian, (residual_value.size, x.size))
    step = jnp.linalg.solve(jacobian_matrix, -jnp.ravel(residual_value))
    return x + jnp.reshape(step, x.shape)


def newton_scan_fixed_iters(
    residual: Callable[[jax.Array], jax.Array],
    x0: jax.Array,
    *,
    max_iter: int,
    jac: Callable[[jax.Array], jax.Array] | None = None,
) -> jax.Array:
    """Run a fixed number of Newton iterations through ``jax.lax.scan``."""
    jacobian_fn = jax.jacfwd(residual) if jac is None else jac

    def body(x, _):
        return _newton_step(residual, jacobian_fn, x), None

    root, _ = jax.lax.scan(body, jnp.asarray(x0), None, length=int(max_iter))
    return root


def _newton_while_solve(
    residual: Callable[[jax.Array, object], jax.Array],
    x0: jax.Array,
    params: object,
    *,
    max_iter: int,
    tol: float,
    jac: Callable[[jax.Array, object], jax.Array] | None,
) -> jax.Array:
    def residual_at_params(x):
        return residual(x, params)

    jacobian_fn = (
        jax.jacfwd(residual_at_params) if jac is None else lambda x: jac(x, params)
    )

    def residual_norm(x):
        return jnp.linalg.norm(jnp.ravel(residual_at_params(x)), ord=jnp.inf)

    initial_x = jnp.asarray(x0)
    initial_norm = residual_norm(initial_x)
    initial_state = (jnp.asarray(0), initial_x, initial_norm)

    def should_continue(state):
        iteration, _, norm = state
        return (iteration < int(max_iter)) & (norm > jnp.asarray(tol, dtype=norm.dtype))

    def body(state):
        iteration, x, _ = state
        next_x = _newton_step(residual_at_params, jacobian_fn, x)
        return iteration + 1, next_x, residual_norm(next_x)

    _, root, _ = jax.lax.while_loop(should_continue, body, initial_state)
    return root


def _implicit_jacobian(
    residual: Callable[[jax.Array, object], jax.Array],
    root: jax.Array,
    params: object,
    jac: Callable[[jax.Array, object], jax.Array] | None,
) -> jax.Array:
    if jac is None:
        return jax.jacfwd(lambda x: residual(x, params))(root)
    return jac(root, params)


def newton_with_implicit_vjp(
    residual: Callable[[jax.Array, object], jax.Array],
    x0: jax.Array,
    params: object,
    *,
    max_iter: int,
    tol: float,
    jac: Callable[[jax.Array, object], jax.Array] | None = None,
) -> jax.Array:
    """Solve ``residual(x, params) = 0`` with an implicit reverse-mode rule."""

    @jax.custom_vjp
    def solve(params_arg, x0_arg):
        return _newton_while_solve(
            residual,
            x0_arg,
            params_arg,
            max_iter=max_iter,
            tol=tol,
            jac=jac,
        )

    def solve_fwd(params_arg, x0_arg):
        root = solve(params_arg, x0_arg)
        return root, (root, params_arg, x0_arg)

    def solve_bwd(saved, root_cotangent):
        root, params_arg, x0_arg = saved
        residual_value = residual(root, params_arg)
        jacobian = _implicit_jacobian(residual, root, params_arg, jac)
        jacobian_matrix = jnp.reshape(jacobian, (residual_value.size, root.size))
        adjoint = jnp.linalg.solve(
            jnp.transpose(jacobian_matrix),
            jnp.ravel(root_cotangent),
        )
        adjoint_residual = jnp.reshape(adjoint, residual_value.shape)
        _, params_pullback = jax.vjp(lambda p: residual(root, p), params_arg)
        (params_cotangent,) = params_pullback(adjoint_residual)
        params_cotangent = jax.tree_util.tree_map(lambda leaf: -leaf, params_cotangent)
        return params_cotangent, jnp.zeros_like(x0_arg)

    solve.defvjp(solve_fwd, solve_bwd)
    return solve(params, x0)
