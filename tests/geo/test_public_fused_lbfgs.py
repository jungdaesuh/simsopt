"""Public contracts for the prepared fused L-BFGS seam."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt_jax.geo.optimizers.fused_lbfgs import (
    FusedLBFGSOptions,
    FusedLBFGSResult,
    PreparedFusedLBFGS,
    prepare_fused_lbfgs,
)

pytestmark = pytest.mark.skipif(
    jax.__version__ != "0.10.0",
    reason="Fused L-BFGS is validated on the pinned JAX runtime.",
)


def _rosenbrock(parameters: jax.Array) -> jax.Array:
    return (
        100.0 * (parameters[1] - parameters[0] ** 2) ** 2 + (1.0 - parameters[0]) ** 2
    )


def test_public_api_prepares_once_and_reuses_dynamic_run_budgets() -> None:
    x0 = jnp.asarray([-1.2, 1.0], dtype=jnp.float64)
    options = FusedLBFGSOptions(
        history_size=7,
        function_tolerance=0.0,
        gradient_tolerance=1.0e-7,
        maximum_line_search_steps=30,
    )

    prepared = prepare_fused_lbfgs(_rosenbrock, x0, options=options)
    one_iteration = prepared.run(x0, maxiter=1, maxfun=15000)
    one_evaluation_budget = prepared.run(x0, maxiter=60, maxfun=1)
    complete = prepared.run(x0, maxiter=60, maxfun=15000)

    assert isinstance(prepared, PreparedFusedLBFGS)
    assert prepared.history_size == 7
    assert int(np.asarray(one_iteration.iterations)) == 1
    assert int(np.asarray(one_evaluation_budget.function_evaluations)) < int(
        np.asarray(complete.function_evaluations)
    )
    assert int(np.asarray(complete.iterations)) > 1
    assert float(np.asarray(complete.state.objective_value)) < float(
        np.asarray(one_iteration.state.objective_value)
    )
    np.testing.assert_allclose(
        np.asarray(complete.state.parameters),
        np.ones(2),
        rtol=0.0,
        atol=1.0e-5,
    )


def test_public_result_contains_only_public_jax_array_state() -> None:
    x0 = jnp.asarray([-1.2, 1.0], dtype=jnp.float64)
    result = prepare_fused_lbfgs(_rosenbrock, x0).run(
        x0,
        maxiter=3,
        maxfun=8,
    )

    assert isinstance(result, FusedLBFGSResult)
    assert result.__class__.__module__ == ("simsopt_jax.geo.optimizers.fused_lbfgs")
    assert result.state.__class__.__module__ == (
        "simsopt_jax.geo.optimizers.fused_lbfgs"
    )
    assert result.invalid_step_record.__class__.__module__ == (
        "simsopt_jax.geo.optimizers.fused_lbfgs"
    )
    assert all(
        isinstance(leaf, jax.Array) for leaf in jax.tree_util.tree_leaves(result)
    )
    assert int(np.asarray(result.evaluated_nonfinite_count)) == 0
    assert bool(np.asarray(result.all_accepted_states_finite))


def test_fused_solver_counts_rejected_recovered_nonfinite_trial_on_device() -> None:
    x0 = jnp.asarray([-1.2, 1.0], dtype=jnp.float64)

    def objective_with_nonfinite_trial(parameters: jax.Array) -> jax.Array:
        value = _rosenbrock(parameters)
        rejected_trial = (parameters[0] > -0.4) & (parameters[0] < -0.2)
        return jnp.where(
            rejected_trial,
            jnp.asarray(jnp.nan, dtype=parameters.dtype),
            value,
        )

    result = prepare_fused_lbfgs(
        objective_with_nonfinite_trial,
        x0,
        options=FusedLBFGSOptions(maximum_line_search_steps=30),
    ).run(x0, maxiter=1, maxfun=100)

    assert int(np.asarray(result.evaluated_nonfinite_count)) == 1
    assert bool(np.asarray(result.all_accepted_states_finite))
    assert int(np.asarray(result.iterations)) == 1
    assert np.isfinite(np.asarray(result.state.objective_value))
    assert np.all(np.isfinite(np.asarray(result.state.gradient)))
    assert np.all(np.isfinite(np.asarray(result.state.parameters)))
    assert not bool(np.asarray(result.invalid_step_record.nonfinite_step))


def test_public_contract_has_no_private_mode_or_callback_controls() -> None:
    prepare_parameters = inspect.signature(prepare_fused_lbfgs).parameters
    run_parameters = inspect.signature(PreparedFusedLBFGS.run).parameters

    assert "run_mode" not in prepare_parameters
    assert "cache_owner" not in prepare_parameters
    assert "callback" not in prepare_parameters
    assert "callback" not in run_parameters
    with pytest.raises(FrozenInstanceError):
        FusedLBFGSOptions().__setattr__("history_size", 4)
