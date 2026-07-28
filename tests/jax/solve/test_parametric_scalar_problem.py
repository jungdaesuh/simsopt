from __future__ import annotations

import jax
import jax.numpy as jnp
from simsopt_jax.solve.driver import Driver
from simsopt_jax.solve.serial import (
    TraceableParametricScalarProblem,
    serial_solve_jax,
)


def test_parametric_scalar_problem_reuses_one_traced_program(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    trace_count = 0

    def objective(parameters: jax.Array, target: jax.Array) -> jax.Array:
        nonlocal trace_count
        trace_count += 1
        residual = parameters - target
        return jnp.vdot(residual, residual).real

    problem = TraceableParametricScalarProblem(
        objective_fn=objective,
        objective_parameter=jnp.asarray([1.0, -2.0], dtype=jnp.float64),
        x=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
    )
    first_solver_function = problem._solver_value_and_grad_fn
    first = serial_solve_jax(
        problem,
        driver=Driver.SIMSOPT_LBFGSB,
        max_steps=8,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    problem.set_objective_parameter(jnp.asarray([-3.0, 4.0], dtype=jnp.float64))
    second = serial_solve_jax(
        problem,
        driver=Driver.SIMSOPT_LBFGSB,
        max_steps=8,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    assert trace_count == 1
    assert problem._solver_value_and_grad_fn is first_solver_function
    assert first.success
    assert second.success
    assert jnp.allclose(first.x, jnp.asarray([1.0, -2.0]), atol=1.0e-10)
    assert jnp.allclose(second.x, jnp.asarray([-3.0, 4.0]), atol=1.0e-10)
