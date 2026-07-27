"""Backend-neutral serial JAX solves for explicit immutable problems."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import time

import jax
import jax.numpy as jnp
import numpy as np

from simsopt_jax.core._finite_difference import (
    forward_jacobian_shard_map,
    forward_jacobian_vmap,
)
from simsopt_jax.runtime.host_boundary import host_array, host_float
from simsopt_jax.runtime.jaxpr_closure import (
    closure_converted_value_and_grad,
    device_put_closure_consts,
)
from simsopt_jax.solve.contracts import OptimizerResult
from simsopt_jax.solve.dispatch import least_squares, minimize
from simsopt_jax.solve.driver import Driver
from simsopt_jax.solve.optax.contracts import OptaxLBFGSOptions
from simsopt_jax.solve.optimistix.contracts import (
    OptimistixLBFGSOptions,
    OptimistixLMOptions,
)
from simsopt_jax.solve.simsopt.contracts import (
    SimsoptBFGSOptions,
    SimsoptLBFGSBOptions,
    SimsoptLMGMRESOptions,
    SimsoptLMQROptions,
)

__all__ = [
    "TraceableEqualityConstrainedProblem",
    "TraceableLeastSquaresProblem",
    "TraceableScalarProblem",
    "constrained_serial_solve_jax",
    "least_squares_serial_solve_jax",
    "serial_solve_jax",
    "traceable_least_squares_jacobian",
]


@dataclass
class TraceableLeastSquaresProblem:
    """Least-squares residual with explicit JAX state and no host graph wrapping."""

    residual_fn: Callable[[jax.Array], jax.Array]
    x: jax.Array
    _solver_residual_fn: Callable[[jax.Array], jax.Array] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        residual_fn = self.residual_fn
        self._solver_residual_fn = jax.jit(
            lambda x: jnp.ravel(residual_fn(jnp.asarray(x)))
        )

    @property
    def dof_size(self) -> int:
        return int(jnp.ravel(self.x).size)

    def residuals(self, x: jax.Array | None = None) -> jax.Array:
        return self._solver_residual_fn(self.x if x is None else x)

    def objective(self, x: jax.Array | None = None) -> jax.Array:
        residuals = self.residuals(x)
        return jnp.sum(residuals * residuals)


@dataclass
class TraceableScalarProblem:
    """Scalar objective with explicit JAX state and no host graph wrapping."""

    objective_fn: Callable[[jax.Array], jax.Array]
    x: jax.Array
    _solver_objective_fn: Callable[[jax.Array], jax.Array] = field(
        init=False,
        repr=False,
    )
    _solver_value_and_grad_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array]] = (
        field(init=False, repr=False)
    )

    def __post_init__(self) -> None:
        objective_fn = self.objective_fn
        x = jnp.asarray(self.x)

        def normalized_objective(current_x):
            return jnp.asarray(objective_fn(jnp.asarray(current_x)))

        converted_value_and_grad, value_and_grad_consts = (
            closure_converted_value_and_grad(
                jax.value_and_grad(normalized_objective),
                x,
            )
        )
        value_and_grad_consts = device_put_closure_consts(value_and_grad_consts, x)
        compiled_value_and_grad = jax.jit(converted_value_and_grad)

        def solver_value_and_grad(current_x):
            return compiled_value_and_grad(current_x, value_and_grad_consts)

        def solver_objective(current_x):
            value, _gradient = solver_value_and_grad(current_x)
            return value

        self._solver_objective_fn = solver_objective
        self._solver_value_and_grad_fn = solver_value_and_grad

    @property
    def dof_size(self) -> int:
        return int(jnp.ravel(self.x).size)

    def objective(self, x: jax.Array | None = None) -> jax.Array:
        return self._solver_objective_fn(self.x if x is None else x)

    def value_and_grad(
        self,
        x: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Evaluate the compiled objective and gradient on the active device."""
        return self._solver_value_and_grad_fn(self.x if x is None else x)


@dataclass
class TraceableEqualityConstrainedProblem:
    """Equality-constrained state awaiting a SIMSOPT-owned JAX solver."""

    objective_fn: Callable[[jax.Array], jax.Array]
    equality_constraint_fn: Callable[[jax.Array], jax.Array]
    x: jax.Array

    @property
    def dof_size(self) -> int:
        return int(jnp.ravel(self.x).size)

    def objective(self, x: jax.Array | None = None) -> jax.Array:
        return jnp.asarray(self.objective_fn(self.x if x is None else jnp.asarray(x)))

    def equality_constraints(self, x: jax.Array | None = None) -> jax.Array:
        return jnp.ravel(
            self.equality_constraint_fn(self.x if x is None else jnp.asarray(x))
        )


def traceable_least_squares_jacobian(
    prob: TraceableLeastSquaresProblem,
    x: jax.Array,
    *,
    method: str,
    abs_step: float = 1.0e-7,
    rel_step: float = 0.0,
    diff_method: str = "forward",
    mesh=None,
) -> jax.Array:
    """Return the traceable residual Jacobian using the caller-selected route."""
    residuals = prob.residuals
    if method == "jacfwd":
        return jax.jacfwd(residuals)(x)
    if method == "vmap":
        return forward_jacobian_vmap(residuals, x, abs_step, rel_step, diff_method)
    if method == "shard_map":
        return forward_jacobian_shard_map(
            residuals,
            x,
            abs_step,
            rel_step,
            diff_method,
            mesh=mesh,
        )
    raise ValueError(f"Unsupported JAX least-squares Jacobian method {method!r}.")


def _write_log_header(objective_file, *, problem_type: str, ndofs: int) -> None:
    objective_file.write(f"Problem type:\n{problem_type}\nnparams:\n{ndofs}\n")
    objective_file.write("function_evaluation,seconds")
    for index in range(ndofs):
        objective_file.write(f",x({index})")
    objective_file.write(",objective_function\n")


def _write_log_row(
    objective_file,
    *,
    eval_index: int,
    elapsed_seconds: float,
    x: jax.Array,
    objective_value: jax.Array,
) -> None:
    objective_file.write(f"{eval_index:6d},{elapsed_seconds:12.4e}")
    x_host = np.ravel(host_array(x))
    objective_host = host_float(objective_value, dtype=objective_value.dtype)
    for value in x_host:
        objective_file.write(f",{value:24.16e}")
    objective_file.write(f",{objective_host:24.16e}\n")
    objective_file.flush()


def _least_squares_options(
    driver: Driver,
    *,
    rtol: float,
    atol: float,
    max_steps: int,
) -> SimsoptLMGMRESOptions | SimsoptLMQROptions | OptimistixLMOptions:
    if driver == Driver.SIMSOPT_LM_GMRES:
        return SimsoptLMGMRESOptions(
            maxiter=max_steps,
            ftol=rtol,
            xtol=atol,
            gtol=min(rtol, atol),
            materialize_dense_linearization=False,
        )
    if driver == Driver.SIMSOPT_LM_QR:
        return SimsoptLMQROptions(
            maxiter=max_steps,
            ftol=rtol,
            xtol=atol,
            gtol=min(rtol, atol),
        )
    if driver == Driver.OPTIMISTIX_LM:
        return OptimistixLMOptions(maxiter=max_steps, tol=min(rtol, atol))
    raise ValueError(
        "least_squares_serial_solve_jax driver must be simsopt_lm_gmres, "
        "simsopt_lm_qr, or the explicit optional optimistix_lm driver."
    )


def _scalar_options(
    driver: Driver,
    *,
    rtol: float,
    atol: float,
    max_steps: int,
    maxcor: int | None,
) -> (
    SimsoptBFGSOptions
    | SimsoptLBFGSBOptions
    | OptaxLBFGSOptions
    | OptimistixLBFGSOptions
):
    if driver == Driver.SIMSOPT_BFGS:
        return SimsoptBFGSOptions(maxiter=max_steps, gtol=atol, xrtol=rtol)
    if driver == Driver.SIMSOPT_LBFGSB:
        return SimsoptLBFGSBOptions(
            maxiter=max_steps,
            maxfun=max_steps * 20,
            gtol=atol,
            ftol=rtol,
            maxcor=10 if maxcor is None else maxcor,
        )
    if driver == Driver.OPTAX_LBFGS:
        return OptaxLBFGSOptions(maxiter=max_steps, gtol=atol)
    if driver == Driver.OPTIMISTIX_LBFGS:
        return OptimistixLBFGSOptions(maxiter=max_steps, tol=min(rtol, atol))
    raise ValueError(
        "serial_solve_jax driver must be simsopt_bfgs, simsopt_lbfgsb, or an "
        "explicitly selected optional optax_lbfgs or optimistix_lbfgs driver."
    )


def _write_bounded_objective_log(
    *,
    problem_type: str,
    ndofs: int,
    initial_x: jax.Array,
    initial_objective: jax.Array,
    final_x: jax.Array,
    final_objective: jax.Array,
    start_time: float,
) -> None:
    with jax.transfer_guard("allow"):
        datestr = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
        with open(f"simsopt_{datestr}.dat", "w") as objective_file:
            _write_log_header(
                objective_file,
                problem_type=problem_type,
                ndofs=ndofs,
            )
            _write_log_row(
                objective_file,
                eval_index=0,
                elapsed_seconds=0.0,
                x=initial_x,
                objective_value=initial_objective,
            )
            _write_log_row(
                objective_file,
                eval_index=1,
                elapsed_seconds=time() - start_time,
                x=final_x,
                objective_value=final_objective,
            )


def _require_success(result: OptimizerResult, *, operation: str) -> None:
    if not result.success:
        raise RuntimeError(
            f"{operation} failed with driver={result.driver.value}, "
            f"status={result.status}: {result.message}"
        )


def least_squares_serial_solve_jax(
    prob: TraceableLeastSquaresProblem,
    *,
    driver: Driver = Driver.SIMSOPT_LM_GMRES,
    optimizer: str | None = None,
    rtol: float = 1.0e-8,
    atol: float = 1.0e-8,
    max_steps: int = 256,
    **kwargs,
) -> OptimizerResult:
    """Solve on the active JAX device and publish the completed state."""
    if not isinstance(prob, TraceableLeastSquaresProblem):
        raise TypeError(
            "least_squares_serial_solve_jax requires TraceableLeastSquaresProblem."
        )
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported JAX least-squares options: {unsupported}")
    if optimizer is not None:
        if driver != Driver.SIMSOPT_LM_GMRES:
            raise TypeError("Specify only driver or the deprecated optimizer keyword.")
        if optimizer == "gauss_newton":
            raise NotImplementedError(
                "optimizer='gauss_newton' has no typed backend-neutral driver; "
                "use driver=Driver.OPTIMISTIX_LM for the explicit optional LM path."
            )
        if optimizer != "lm":
            raise ValueError(f"Unsupported JAX least-squares optimizer {optimizer!r}.")
        warnings.warn(
            "optimizer='lm' is deprecated; use driver=Driver.OPTIMISTIX_LM.",
            DeprecationWarning,
            stacklevel=2,
        )
        driver = Driver.OPTIMISTIX_LM

    initial_x = jnp.asarray(prob.x)
    initial_objective = prob.objective(initial_x)
    start_time = time()
    result = least_squares(
        prob._solver_residual_fn,
        initial_x,
        driver=driver,
        options=_least_squares_options(
            driver,
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
        ),
    )
    _require_success(result, operation="JAX least-squares solve")
    final_x = jax.device_put(result.x, initial_x.sharding)
    final_objective = prob.objective(final_x)
    _write_bounded_objective_log(
        problem_type="least_squares",
        ndofs=prob.dof_size,
        initial_x=initial_x,
        initial_objective=initial_objective,
        final_x=final_x,
        final_objective=final_objective,
        start_time=start_time,
    )
    prob.x = final_x
    return result


def serial_solve_jax(
    prob: TraceableScalarProblem,
    *,
    driver: Driver = Driver.SIMSOPT_BFGS,
    optimizer: str | None = None,
    rtol: float = 1.0e-8,
    atol: float = 1.0e-8,
    max_steps: int = 256,
    maxcor: int | None = None,
    require_success: bool = True,
    **kwargs,
) -> OptimizerResult:
    """Minimize on device and optionally publish a nonconverged bounded state."""
    if not isinstance(prob, TraceableScalarProblem):
        raise TypeError("serial_solve_jax requires TraceableScalarProblem.")
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported JAX scalar solve options: {unsupported}")
    if maxcor is not None:
        if driver != Driver.SIMSOPT_LBFGSB:
            raise TypeError("maxcor is only supported by Driver.SIMSOPT_LBFGSB")
        if maxcor < 1:
            raise ValueError("maxcor must be positive")
    if optimizer is not None:
        if driver != Driver.SIMSOPT_BFGS:
            raise TypeError("Specify only driver or the deprecated optimizer keyword.")
        if optimizer != "bfgs":
            raise ValueError(f"Unsupported JAX scalar optimizer {optimizer!r}.")
        raise NotImplementedError(
            "optimizer='bfgs' selected Optimistix BFGS and has no "
            "behavior-equivalent typed driver. Choose driver=Driver.SIMSOPT_BFGS "
            "for the backend-neutral default or explicitly opt into the distinct "
            "driver=Driver.OPTIMISTIX_LBFGS algorithm."
        )

    initial_x = jnp.asarray(prob.x)
    initial_objective = prob.objective(initial_x)
    start_time = time()
    result = minimize(
        prob._solver_value_and_grad_fn,
        initial_x,
        driver=driver,
        options=_scalar_options(
            driver,
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
            maxcor=maxcor,
        ),
    )
    if require_success:
        _require_success(result, operation="JAX scalar solve")
    final_x = jax.device_put(result.x, initial_x.sharding)
    final_objective = prob.objective(final_x)
    _write_bounded_objective_log(
        problem_type="general",
        ndofs=prob.dof_size,
        initial_x=initial_x,
        initial_objective=initial_objective,
        final_x=final_x,
        final_objective=final_objective,
        start_time=start_time,
    )
    prob.x = final_x
    return result


def constrained_serial_solve_jax(
    prob: TraceableEqualityConstrainedProblem,
    *,
    optimizer: str = "bfgs",
    rtol: float = 1.0e-8,
    atol: float = 1.0e-8,
    max_outer: int = 8,
    inner_max_steps: int = 256,
    initial_penalty_weight: float = 10.0,
    penalty_growth: float = 10.0,
    max_penalty_weight: float = 1.0e8,
    **kwargs,
) -> None:
    """Reject until SIMSOPT owns a backend-neutral constrained JAX solver."""
    if not isinstance(prob, TraceableEqualityConstrainedProblem):
        raise TypeError(
            "constrained_serial_solve_jax requires TraceableEqualityConstrainedProblem."
        )
    raise NotImplementedError(
        "constrained_serial_solve_jax has no SIMSOPT-owned constrained solver "
        "with one backend-neutral CPU/GPU contract. Use the native CPU "
        "constrained_serial_solve reference until that contract is implemented."
    )
