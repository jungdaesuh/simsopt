"""JAX-aware serial least-squares solve for explicit residual problems."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from time import time
from typing import TextIO

import jax
import jax.numpy as jnp
import numpy as np

from simsopt.jax_core._finite_difference import (
    forward_jacobian_shard_map,
    forward_jacobian_vmap,
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

    @property
    def dof_size(self) -> int:
        return int(jnp.ravel(self.x).size)

    def residuals(self, x: jax.Array | None = None) -> jax.Array:
        return jnp.ravel(self.residual_fn(self.x if x is None else jnp.asarray(x)))

    def objective(self, x: jax.Array | None = None) -> jax.Array:
        residuals = self.residuals(x)
        return jnp.sum(residuals * residuals)


@dataclass
class TraceableScalarProblem:
    """Scalar objective with explicit JAX state and no host graph wrapping."""

    objective_fn: Callable[[jax.Array], jax.Array]
    x: jax.Array

    @property
    def dof_size(self) -> int:
        return int(jnp.ravel(self.x).size)

    def objective(self, x: jax.Array | None = None) -> jax.Array:
        return jnp.asarray(self.objective_fn(self.x if x is None else jnp.asarray(x)))


@dataclass
class TraceableEqualityConstrainedProblem:
    """Scalar objective plus equality constraints for the JAX AL solve."""

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


def _optimistix_solver(name: str, *, rtol: float, atol: float):
    import optimistix as optx

    if name == "lm":
        return optx.LevenbergMarquardt(rtol=rtol, atol=atol)
    if name == "gauss_newton":
        return optx.GaussNewton(rtol=rtol, atol=atol)
    raise ValueError(f"Unsupported JAX least-squares optimizer {name!r}.")


def _optimistix_minimizer(name: str, *, rtol: float, atol: float):
    import optimistix as optx

    if name == "bfgs":
        return optx.BFGS(rtol=rtol, atol=atol)
    raise ValueError(f"Unsupported JAX scalar optimizer {name!r}.")


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
    with jax.transfer_guard_device_to_host("allow"):
        x_host = np.ravel(np.asarray(jax.device_get(x)))
        objective_host = float(jax.device_get(objective_value))
    for value in x_host:
        objective_file.write(f",{value:24.16e}")
    objective_file.write(f",{objective_host:24.16e}\n")
    objective_file.flush()


@dataclass
class _ObjectiveEvaluationLogger:
    objective_file: TextIO
    start_time: float
    next_eval_index: int = 0

    def __call__(self, x: jax.Array, objective_value: jax.Array) -> None:
        _write_log_row(
            self.objective_file,
            eval_index=self.next_eval_index,
            elapsed_seconds=time() - self.start_time,
            x=x,
            objective_value=objective_value,
        )
        self.next_eval_index += 1


def _write_constraint_header(
    constraint_file,
    *,
    ndofs: int,
    constraint_count: int,
) -> None:
    constraint_file.write(f"Problem type:\nconstrained\nnparams:\n{ndofs}\n")
    constraint_file.write("function_evaluation,seconds")
    for index in range(ndofs):
        constraint_file.write(f",x({index})")
    constraint_file.write(",constraint_function\n")
    for index in range(constraint_count):
        constraint_file.write(f",F({index})")
    constraint_file.write("\n")


def _write_constraint_row(
    constraint_file,
    *,
    eval_index: int,
    elapsed_seconds: float,
    x: jax.Array,
    constraint_value: jax.Array,
) -> None:
    constraint_file.write(f"{eval_index:6d},{elapsed_seconds:12.4e}")
    with jax.transfer_guard_device_to_host("allow"):
        x_host = np.ravel(np.asarray(jax.device_get(x)))
        constraint_host = np.ravel(np.asarray(jax.device_get(constraint_value)))
    for value in x_host:
        constraint_file.write(f",{value:24.16e}")
    for value in constraint_host:
        constraint_file.write(f",{value:24.16e}")
    constraint_file.write("\n")
    constraint_file.flush()


@dataclass
class _ConstraintEvaluationLogger:
    constraint_file: TextIO
    start_time: float
    next_eval_index: int = 0

    def __call__(self, x: jax.Array, constraint_value: jax.Array) -> None:
        _write_constraint_row(
            self.constraint_file,
            eval_index=self.next_eval_index,
            elapsed_seconds=time() - self.start_time,
            x=x,
            constraint_value=constraint_value,
        )
        self.next_eval_index += 1


def least_squares_serial_solve_jax(
    prob: TraceableLeastSquaresProblem,
    *,
    optimizer: str = "lm",
    rtol: float = 1.0e-8,
    atol: float = 1.0e-8,
    max_steps: int = 256,
    **kwargs,
) -> None:
    """Solve a traceable JAX least-squares problem and update ``prob.x``."""
    if not isinstance(prob, TraceableLeastSquaresProblem):
        raise TypeError(
            "least_squares_serial_solve_jax requires TraceableLeastSquaresProblem."
        )
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported JAX least-squares options: {unsupported}")

    x0 = jnp.asarray(prob.x)
    import optimistix as optx

    solver = _optimistix_solver(optimizer, rtol=rtol, atol=atol)

    datestr = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    start_time = time()
    with open(f"simsopt_{datestr}.dat", "w") as objective_file:
        _write_log_header(
            objective_file,
            problem_type="least_squares",
            ndofs=prob.dof_size,
        )
        evaluation_logger = _ObjectiveEvaluationLogger(objective_file, start_time)

        def residuals_for_solver(x, _args):
            residual_values = prob.residuals(x)
            objective_value = jnp.sum(residual_values * residual_values)
            jax.debug.callback(evaluation_logger, x, objective_value, ordered=True)
            return residual_values

        solution = optx.least_squares(
            residuals_for_solver,
            solver,
            x0,
            options={"jac": "fwd"},
            max_steps=max_steps,
            throw=True,
        )
        final_x = solution.value
        jax.block_until_ready(final_x)
        jax.effects_barrier()

    prob.x = final_x


def serial_solve_jax(
    prob: TraceableScalarProblem,
    *,
    optimizer: str = "bfgs",
    rtol: float = 1.0e-8,
    atol: float = 1.0e-8,
    max_steps: int = 256,
    **kwargs,
) -> None:
    """Solve a traceable JAX scalar objective and update ``prob.x``."""
    if not isinstance(prob, TraceableScalarProblem):
        raise TypeError("serial_solve_jax requires TraceableScalarProblem.")
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported JAX scalar solve options: {unsupported}")

    x0 = jnp.asarray(prob.x)
    import optimistix as optx

    solver = _optimistix_minimizer(optimizer, rtol=rtol, atol=atol)

    datestr = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    start_time = time()
    with open(f"simsopt_{datestr}.dat", "w") as objective_file:
        _write_log_header(
            objective_file,
            problem_type="general",
            ndofs=prob.dof_size,
        )
        evaluation_logger = _ObjectiveEvaluationLogger(objective_file, start_time)

        def objective_for_solver(x, _args):
            objective_value = prob.objective(x)
            jax.debug.callback(evaluation_logger, x, objective_value, ordered=True)
            return objective_value

        solution = optx.minimise(
            objective_for_solver,
            solver,
            x0,
            max_steps=max_steps,
            throw=True,
        )
        final_x = solution.value
        jax.block_until_ready(final_x)
        jax.effects_barrier()

    prob.x = final_x


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
    """Solve a traceable equality-constrained objective by augmented Lagrangian."""
    if not isinstance(prob, TraceableEqualityConstrainedProblem):
        raise TypeError(
            "constrained_serial_solve_jax requires TraceableEqualityConstrainedProblem."
        )
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported JAX constrained solve options: {unsupported}")
    max_outer_value = int(max_outer)
    inner_max_steps_value = int(inner_max_steps)
    if max_outer_value < 1:
        raise ValueError("max_outer must be positive.")
    if inner_max_steps_value < 1:
        raise ValueError("inner_max_steps must be positive.")

    x = jnp.asarray(prob.x)
    objective = jax.jit(prob.objective)
    equality_constraints = jax.jit(prob.equality_constraints)
    constraints0 = equality_constraints(x)
    multipliers = jnp.zeros_like(constraints0)
    penalty_weight = jnp.asarray(initial_penalty_weight, dtype=x.dtype)
    solver = _optimistix_minimizer(optimizer, rtol=rtol, atol=atol)
    import optimistix as optx

    datestr = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    start_time = time()
    with open(f"simsopt_{datestr}.dat", "w") as objective_file:
        with open(f"constraints_{datestr}.dat", "w") as constraint_file:
            current_x = x
            current_multipliers = multipliers
            current_penalty_weight = penalty_weight

            _write_log_header(
                objective_file,
                problem_type="constrained",
                ndofs=prob.dof_size,
            )
            _write_constraint_header(
                constraint_file,
                ndofs=prob.dof_size,
                constraint_count=int(constraints0.size),
            )
            objective_logger = _ObjectiveEvaluationLogger(objective_file, start_time)
            constraint_logger = _ConstraintEvaluationLogger(constraint_file, start_time)

            for _outer_index in range(max_outer_value):

                def augmented_objective(candidate_x, _args):
                    constraints = equality_constraints(candidate_x)
                    objective_value = objective(candidate_x)
                    jax.debug.callback(
                        objective_logger,
                        candidate_x,
                        objective_value,
                        ordered=True,
                    )
                    jax.debug.callback(
                        constraint_logger,
                        candidate_x,
                        constraints,
                        ordered=True,
                    )
                    return (
                        objective_value
                        + jnp.dot(current_multipliers, constraints)
                        + 0.5
                        * current_penalty_weight
                        * jnp.sum(constraints * constraints)
                    )

                solution = optx.minimise(
                    augmented_objective,
                    solver,
                    current_x,
                    max_steps=inner_max_steps_value,
                    throw=True,
                )
                current_x = solution.value
                constraints = equality_constraints(current_x)
                current_multipliers = (
                    current_multipliers + current_penalty_weight * constraints
                )
                current_penalty_weight = jnp.minimum(
                    current_penalty_weight * float(penalty_growth),
                    jnp.asarray(
                        max_penalty_weight,
                        dtype=current_penalty_weight.dtype,
                    ),
                )

            jax.block_until_ready(current_x)
            jax.effects_barrier()

    prob.x = current_x
