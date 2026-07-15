"""Runtime implementations for public JAX scalar minimization drivers."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import optimistix as optx
from jax.extend import core as jax_core
from scipy.optimize import OptimizeResult

from .contracts import (
    Callback,
    Driver,
    OptaxAdamCallbackEvent,
    OptaxLBFGSCallbackEvent,
    OptimistixLBFGSCallbackEvent,
    ValueAndGradFn,
)
from .optax.contracts import OptaxAdamOptions, OptaxLBFGSOptions, OptaxLineSearch
from .optimistix.contracts import OptimistixLBFGSOptions


def host_array(value) -> np.ndarray:
    return np.asarray(jax.device_get(jax.block_until_ready(value)))


def host_float(value) -> float:
    return float(np.asarray(jax.device_get(jax.block_until_ready(value))).reshape(()))


def block_jax_leaves(value) -> None:
    for leaf in jax.tree.leaves(value):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def optimistix_result_metadata(result) -> tuple[bool, str, str]:
    host_result = jax.device_get(result)
    result_text = str(host_result)
    result_message = str(optx.RESULTS[host_result])
    successful = result_text == str(optx.RESULTS.successful)
    return successful, result_text, result_message


def _closure_converted_scalar_value(
    value_and_grad_fn: ValueAndGradFn,
    example_params,
):
    converted = jax.make_jaxpr(value_and_grad_fn)(example_params)
    value_and_grad_jaxpr = converted.jaxpr
    value_and_grad_consts = tuple(converted.consts)
    const_count = len(value_and_grad_consts)

    def value_and_grad_from_jaxpr(x, consts):
        closed_jaxpr = jax_core.ClosedJaxpr(value_and_grad_jaxpr, consts)
        return jax_core.jaxpr_as_fun(closed_jaxpr)(x)

    @jax.custom_vjp
    def scalar_value(x, consts):
        value, _grad = value_and_grad_from_jaxpr(x, consts)
        return jnp.asarray(value)

    def scalar_value_fwd(x, consts):
        value, grad = value_and_grad_from_jaxpr(x, consts)
        return jnp.asarray(value), jnp.asarray(grad)

    def scalar_value_bwd(explicit_grad, cotangent):
        return (jnp.asarray(cotangent) * explicit_grad, (None,) * const_count)

    scalar_value.defvjp(scalar_value_fwd, scalar_value_bwd)
    return scalar_value, value_and_grad_consts


def run_optax_minimize(
    value_and_grad_fn: ValueAndGradFn,
    x0,
    *,
    driver: Driver,
    options: OptaxLBFGSOptions | OptaxAdamOptions,
    callback: Callback | None,
) -> OptimizeResult:
    start = time.perf_counter()
    params = jnp.asarray(jax.device_put(x0))
    params_sharding = params.sharding
    maxiter = options.maxiter
    gtol = options.gtol
    if isinstance(options, OptaxLBFGSOptions):
        scalar_value, scalar_value_consts = _closure_converted_scalar_value(
            value_and_grad_fn,
            params,
        )
        scalar_value_consts = jax.device_put(scalar_value_consts, params_sharding)

        def value_fn(current_params, *, scalar_value_consts):
            return scalar_value(current_params, scalar_value_consts)

        if options.line_search != OptaxLineSearch.ZOOM:
            raise ValueError(
                "OptaxLBFGSOptions.line_search must be OptaxLineSearch.ZOOM."
            )
        transform = optax.lbfgs(
            memory_size=options.memory_size,
            scale_init_precond=options.scale_init_precond,
            linesearch=optax.scale_by_zoom_linesearch(
                max_linesearch_steps=options.max_linesearch_steps,
                initial_guess_strategy="one",
            ),
        )

        @jax.jit
        def apply_optax_step(
            current_params,
            current_state,
            current_grad,
            current_value,
            scalar_value_consts,
        ):
            updates, next_state = transform.update(
                current_grad,
                current_state,
                current_params,
                value=current_value,
                grad=current_grad,
                value_fn=value_fn,
                scalar_value_consts=scalar_value_consts,
            )
            return optax.apply_updates(current_params, updates), next_state

        optax_step_args = (scalar_value_consts,)
    else:
        transform = (
            optax.adamw(
                learning_rate=options.learning_rate,
                b1=options.b1,
                b2=options.b2,
                eps=options.eps,
                weight_decay=options.weight_decay,
            )
            if options.weight_decay > 0.0
            else optax.adam(
                learning_rate=options.learning_rate,
                b1=options.b1,
                b2=options.b2,
                eps=options.eps,
            )
        )

        @jax.jit
        def apply_optax_step(
            current_params,
            current_state,
            current_grad,
            current_value,
        ):
            del current_value
            updates, next_state = transform.update(
                current_grad,
                current_state,
                current_params,
            )
            return optax.apply_updates(current_params, updates), next_state

        optax_step_args = ()

    state = jax.device_put(jax.jit(transform.init)(params), params_sharding)

    nfev = 0
    njev = 0
    status = 1
    message = "Maximum number of iterations reached."
    success = False
    value, grad = value_and_grad_fn(params)
    value = jnp.asarray(value)
    grad = jnp.asarray(grad)
    block_jax_leaves((value, grad))
    nfev += 1
    njev += 1
    completed_iterations = 0
    for iteration in range(maxiter):
        grad_norm_inf = host_float(jnp.max(jnp.abs(grad)))
        if gtol is not None and grad_norm_inf <= float(gtol):
            status = 0
            message = "Optimization terminated successfully."
            success = True
            break
        params, state = apply_optax_step(params, state, grad, value, *optax_step_args)
        block_jax_leaves((params, state))
        completed_iterations = iteration + 1
        value, grad = value_and_grad_fn(params)
        value = jnp.asarray(value)
        grad = jnp.asarray(grad)
        block_jax_leaves((value, grad))
        nfev += 1
        njev += 1
        grad_norm_inf = host_float(jnp.max(jnp.abs(grad)))
        if callback is not None:
            _emit_optax_callback(
                callback,
                driver=driver,
                iteration=completed_iterations,
                params=params,
                value=value,
                grad_norm_inf=grad_norm_inf,
                state=state,
                options=options,
                wallclock_s=time.perf_counter() - start,
            )
        if gtol is not None and grad_norm_inf <= float(gtol):
            status = 0
            message = "Optimization terminated successfully."
            success = True
            break
    else:
        if gtol is None:
            status = 0
            message = "Optimization terminated after the requested Adam iterations."
            success = driver == Driver.OPTAX_ADAM

    return OptimizeResult(
        x=host_array(params),
        fun=host_float(value),
        jac=host_array(grad),
        nit=completed_iterations,
        nfev=nfev,
        njev=njev,
        status=status,
        success=success,
        message=message,
    )


def _emit_optax_callback(
    callback: Callback,
    *,
    driver: Driver,
    iteration: int,
    params,
    value,
    grad_norm_inf: float,
    state,
    options: OptaxLBFGSOptions | OptaxAdamOptions,
    wallclock_s: float,
) -> None:
    if isinstance(options, OptaxAdamOptions):
        callback(
            OptaxAdamCallbackEvent(
                iteration=iteration,
                x=host_array(params),
                fun=host_float(value),
                grad_norm_inf=grad_norm_inf,
                wallclock_s=wallclock_s,
                learning_rate=options.learning_rate,
            )
        )
        return
    line_search_state = state[-1]
    callback(
        OptaxLBFGSCallbackEvent(
            iteration=iteration,
            x=host_array(params),
            fun=host_float(value),
            grad_norm_inf=grad_norm_inf,
            wallclock_s=wallclock_s,
            learning_rate=host_float(line_search_state.learning_rate),
            num_linesearch_steps=int(
                np.asarray(jax.device_get(line_search_state.info.num_linesearch_steps))
            ),
            decrease_error=host_float(line_search_state.info.decrease_error),
            curvature_error=host_float(line_search_state.info.curvature_error),
        )
    )


def run_optimistix_minimize(
    value_and_grad_fn: ValueAndGradFn,
    x0,
    *,
    options: OptimistixLBFGSOptions,
    callback: Callback | None,
) -> OptimizeResult:
    start = time.perf_counter()
    params = jnp.asarray(jax.device_put(x0))
    solver_tol = float(options.tol)
    scalar_value_fn, scalar_value_consts = _closure_converted_scalar_value(
        value_and_grad_fn,
        params,
    )

    def scalar_value(current, scalar_value_consts):
        return scalar_value_fn(current, scalar_value_consts)

    solver = optx.LBFGS(
        rtol=solver_tol,
        atol=solver_tol,
        history_length=options.history_length,
    )
    with jax.transfer_guard_host_to_device("allow"):
        solution = optx.minimise(
            scalar_value,
            solver,
            params,
            args=scalar_value_consts,
            max_steps=options.maxiter,
            throw=False,
        )
    solution.value.block_until_ready()
    value, grad = value_and_grad_fn(solution.value)
    value = jnp.asarray(value)
    grad = jnp.asarray(grad)
    block_jax_leaves((value, grad))
    x_host = host_array(solution.value)
    fun_host = host_float(value)
    jac_host = host_array(grad)
    grad_norm_inf = float(np.max(np.abs(jac_host))) if jac_host.size else 0.0
    finite = (
        np.all(np.isfinite(x_host))
        and np.isfinite(fun_host)
        and np.all(np.isfinite(jac_host))
    )
    (
        optimistix_success,
        optimistix_result,
        optimistix_result_message,
    ) = optimistix_result_metadata(solution.result)
    success = optimistix_success and bool(finite)
    status = 0 if success else 1 if finite else 2
    num_steps = int(np.asarray(jax.device_get(solution.stats["num_steps"])))
    wallclock_s = time.perf_counter() - start
    if callback is not None:
        callback(
            OptimistixLBFGSCallbackEvent(
                iteration=num_steps,
                x=x_host,
                fun=fun_host,
                grad_norm_inf=grad_norm_inf,
                wallclock_s=wallclock_s,
                history_length=options.history_length,
                optimistix_result=optimistix_result,
            )
        )
    return OptimizeResult(
        x=x_host,
        fun=fun_host,
        jac=jac_host,
        nit=num_steps,
        nfev=num_steps + 1,
        njev=num_steps + 1,
        status=status,
        success=success,
        message=optimistix_result,
        optimistix_result=optimistix_result,
        optimistix_result_message=optimistix_result_message,
    )
