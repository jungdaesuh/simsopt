"""Public parametric fused L-BFGS composition and contract tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt_jax.geo.optimizers.fused_lbfgs import (
    FusedLBFGSOptions,
    FusedLBFGSResult,
    PreparedParametricFusedLBFGS,
    prepare_fused_lbfgs,
    prepare_parametric_fused_lbfgs,
)

pytestmark = pytest.mark.skipif(
    jax.__version__ != "0.10.0",
    reason="Fused L-BFGS is validated on the pinned JAX runtime.",
)


def _rosenbrock(parameters: jax.Array) -> jax.Array:
    return (
        100.0 * (parameters[1] - parameters[0] ** 2) ** 2 + (1.0 - parameters[0]) ** 2
    )


def test_dynamic_parameter_runs_inside_one_jitted_scan_without_retracing() -> None:
    trace_count = 0

    def objective(parameters: jax.Array, target: jax.Array) -> jax.Array:
        nonlocal trace_count
        trace_count += 1
        residual = parameters - target
        return 0.5 * jnp.vdot(residual, residual)

    x0 = jnp.zeros((2,), dtype=jnp.float64)
    target0 = jnp.asarray((1.0, -2.0), dtype=jnp.float64)
    prepared = prepare_parametric_fused_lbfgs(
        objective,
        x0,
        target0,
        maximum_iterations=12,
        maximum_function_evaluations=60,
        options=FusedLBFGSOptions(gradient_tolerance=1.0e-12),
    )
    targets = jnp.asarray(
        ((1.0, -2.0), (-3.0, 4.0), (0.25, -0.75)),
        dtype=jnp.float64,
    )

    @jax.jit
    def run_stages(
        initial_parameters: jax.Array,
        stage_targets: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        def stage(
            parameters: jax.Array,
            target: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            result = prepared.run_staged(parameters, target)
            return result.state.parameters, result.state.parameters

        return jax.lax.scan(stage, initial_parameters, stage_targets)

    final_parameters, stage_parameters = run_stages(x0, targets)

    assert isinstance(prepared, PreparedParametricFusedLBFGS)
    assert trace_count == 1
    np.testing.assert_allclose(stage_parameters, targets, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(final_parameters, targets[-1], rtol=0.0, atol=1.0e-12)


def test_fixed_parameter_matches_nonparametric_fused_solver() -> None:
    x0 = jnp.asarray((-1.2, 1.0), dtype=jnp.float64)
    shift = jnp.asarray((0.07, -0.03), dtype=jnp.float64)
    options = FusedLBFGSOptions(
        history_size=7,
        function_tolerance=0.0,
        gradient_tolerance=1.0e-10,
        maximum_line_search_steps=30,
    )

    def objective(parameters: jax.Array, offset: jax.Array) -> jax.Array:
        return _rosenbrock(parameters - offset)

    prepared_parametric = prepare_parametric_fused_lbfgs(
        objective,
        x0,
        shift,
        maximum_iterations=100,
        maximum_function_evaluations=15000,
        options=options,
    )
    parametric = prepared_parametric.run_staged(x0, shift)
    fixed = prepare_fused_lbfgs(
        lambda parameters: objective(parameters, shift),
        x0,
        options=options,
    ).run(x0, maxiter=100, maxfun=15000)

    assert isinstance(parametric, FusedLBFGSResult)
    assert prepared_parametric.maximum_iterations == 100
    assert prepared_parametric.maximum_function_evaluations == 15000
    assert all(
        isinstance(leaf, jax.Array) for leaf in jax.tree_util.tree_leaves(parametric)
    )
    np.testing.assert_allclose(
        parametric.state.parameters,
        fixed.state.parameters,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        parametric.state.objective_value,
        fixed.state.objective_value,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        parametric.state.gradient,
        fixed.state.gradient,
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_array_equal(parametric.status, fixed.status)
    np.testing.assert_array_equal(parametric.iterations, fixed.iterations)
    np.testing.assert_array_equal(
        parametric.function_evaluations,
        fixed.function_evaluations,
    )
    np.testing.assert_array_equal(
        parametric.evaluated_nonfinite_count,
        fixed.evaluated_nonfinite_count,
    )


def test_parameter_shape_and_dtype_are_frozen_at_preparation() -> None:
    x0 = jnp.zeros((2,), dtype=jnp.float64)
    target = jnp.ones((2,), dtype=jnp.float64)

    def objective(parameters: jax.Array, current_target: jax.Array) -> jax.Array:
        residual = parameters - current_target
        return jnp.vdot(residual, residual)

    prepared = prepare_parametric_fused_lbfgs(
        objective,
        x0,
        target,
        maximum_iterations=4,
        maximum_function_evaluations=20,
    )

    with pytest.raises(ValueError, match="shape must remain"):
        prepared.run_staged(x0, jnp.ones((3,), dtype=jnp.float64))
    with pytest.raises(TypeError, match="dtype must remain"):
        prepared.run_staged(x0, jnp.ones((2,), dtype=jnp.float32))
    with pytest.raises(TypeError, match="dtype must match x dtype"):
        prepare_parametric_fused_lbfgs(
            objective,
            x0,
            jnp.ones((2,), dtype=jnp.float32),
            maximum_iterations=4,
            maximum_function_evaluations=20,
        )
