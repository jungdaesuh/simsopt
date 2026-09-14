"""Dense JAX kernel for the native Boozer least-squares Newton policy."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import lu_factor, lu_solve


class _NativeLsNewtonState(NamedTuple):
    """Current native-order iterate and its full scalar-objective derivatives."""

    x: jax.Array
    fun: jax.Array
    grad: jax.Array
    hessian: jax.Array
    grad_norm: jax.Array
    nit: jax.Array


def newton_ls_native_dense(
    value_grad_hessian_fn: Callable[..., tuple[jax.Array, jax.Array, jax.Array]],
    x0: jax.Array,
    *,
    maxiter: int,
    tol: float,
    stab: float = 0.0,
    args: tuple[object, ...] = (),
) -> dict[str, jax.Array]:
    """Run native-order full-Hessian LS Newton and return its reusable endpoint."""

    initial_x = jnp.asarray(x0)
    initial_fun, initial_grad, initial_hessian = value_grad_hessian_fn(initial_x, *args)
    initial_norm = jnp.linalg.norm(initial_grad)
    initial_state = _NativeLsNewtonState(
        x=initial_x,
        fun=initial_fun,
        grad=initial_grad,
        hessian=initial_hessian,
        grad_norm=initial_norm,
        nit=jnp.asarray(0, dtype=jnp.int32),
    )
    maximum_iterations = jnp.asarray(maxiter, dtype=jnp.int32)
    tolerance = jnp.asarray(tol, dtype=initial_norm.dtype)
    stabilization = jnp.asarray(stab, dtype=initial_hessian.dtype)
    identity = jnp.eye(initial_x.shape[0], dtype=initial_hessian.dtype)

    def continue_iteration(state: _NativeLsNewtonState) -> jax.Array:
        return jnp.logical_and(
            state.nit < maximum_iterations,
            state.grad_norm > tolerance,
        )

    def take_newton_step(state: _NativeLsNewtonState) -> _NativeLsNewtonState:
        stabilized_hessian = state.hessian + stabilization * identity
        factors = lu_factor(stabilized_hessian)
        initial_direction = lu_solve(factors, state.grad)

        def refine_direction(_: None) -> jax.Array:
            correction_rhs = state.grad - stabilized_hessian @ initial_direction
            return initial_direction + lu_solve(factors, correction_rhs)

        direction = lax.cond(
            state.grad_norm < jnp.asarray(1.0e-9, dtype=state.grad_norm.dtype),
            refine_direction,
            lambda _: initial_direction,
            operand=None,
        )
        next_x = state.x - direction
        next_fun, next_grad, next_hessian = value_grad_hessian_fn(next_x, *args)
        return _NativeLsNewtonState(
            x=next_x,
            fun=next_fun,
            grad=next_grad,
            hessian=next_hessian,
            grad_norm=jnp.linalg.norm(next_grad),
            nit=state.nit + jnp.asarray(1, dtype=state.nit.dtype),
        )

    attempted_state = lax.while_loop(
        continue_iteration,
        take_newton_step,
        initial_state,
    )
    success = attempted_state.grad_norm <= tolerance
    persist_solved_state = jnp.logical_or(
        success,
        jnp.logical_and(
            jnp.isfinite(attempted_state.grad_norm),
            attempted_state.grad_norm <= initial_norm,
        ),
    )

    def retain_attempted_state(_: None) -> _NativeLsNewtonState:
        return attempted_state

    def rebuild_initial_state(_: None) -> _NativeLsNewtonState:
        fun, grad, hessian = value_grad_hessian_fn(initial_x, *args)
        return _NativeLsNewtonState(
            x=initial_x,
            fun=fun,
            grad=grad,
            hessian=hessian,
            grad_norm=jnp.linalg.norm(grad),
            nit=attempted_state.nit,
        )

    returned_state = lax.cond(
        persist_solved_state,
        retain_attempted_state,
        rebuild_initial_state,
        operand=None,
    )
    return {
        "x": returned_state.x,
        "fun": returned_state.fun,
        "grad": returned_state.grad,
        "hessian": returned_state.hessian,
        "nit": returned_state.nit,
        "success": success,
        "initial_fun": initial_fun,
        "initial_grad": initial_grad,
        "initial_norm": initial_norm,
        "attempted_norm": attempted_state.grad_norm,
        "final_norm": returned_state.grad_norm,
        "persist_solved_state": persist_solved_state,
    }
