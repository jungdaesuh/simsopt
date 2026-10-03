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
    divergence_factor: float | None = 1e3,
    args: tuple[object, ...] = (),
) -> dict[str, jax.Array]:
    """Run native-order full-Hessian LS Newton and return its reusable endpoint.

    ``divergence_factor`` is a blow-up detector: stop once ``||grad||``
    exceeds that multiple of the residual at Newton entry. Default
    ``1e3``. ``None`` or ``0`` disables the guard. A guarded stop is a
    failed solve: ``success`` is false and the persist/rollback rule is
    unchanged. Twin of the native LS-Newton loop in
    ``simsopt.geo.boozersurface``.

    The reference is the entry residual, not the running best. Undamped
    Newton on Boozer surfaces is non-monotone: an accepted walk can
    excursion by 1e2–1e5 relative to a transient best and still
    converge. Comparing to the best treats those spikes as divergence
    and aborts a successful solve. Blow-up versus the entry residual is
    what distinguishes a rejected line-search trial (313 → 1e10 at
    step 1) from a convergent non-monotone walk.
    """

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
    growth_limit = jnp.asarray(
        0.0 if divergence_factor is None else divergence_factor,
        dtype=initial_norm.dtype,
    )

    def continue_iteration(state: _NativeLsNewtonState) -> jax.Array:
        keep_going = jnp.logical_and(
            state.nit < maximum_iterations,
            state.grad_norm > tolerance,
        )
        if divergence_factor is None or divergence_factor == 0:
            return keep_going
        return jnp.logical_and(
            keep_going,
            state.grad_norm <= growth_limit * initial_norm,
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
