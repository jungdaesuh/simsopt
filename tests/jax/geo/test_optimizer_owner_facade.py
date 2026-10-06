"""Optimizer-facade contracts for the linear-solve and Newton-merit owners."""

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import jax.numpy as jnp
from simsopt_jax.geo.optimizers import linear_solve as _linear_solve
from simsopt_jax.geo.optimizers import optimizer as _optimizer


def test_dense_square_operator_solves_are_reexported_from_optimizer():
    """R07 regression: LU/LSQ dense solves must remain on the optimizer facade."""
    for name in (
        "_solve_dense_square_operator_lu_system_with_status",
        "_solve_dense_square_operator_least_squares_system_with_status",
    ):
        assert hasattr(_optimizer, name), name
        assert getattr(_optimizer, name) is getattr(_linear_solve, name)


def test_newton_candidate_acceptance_uses_the_shared_armijo_merit_owner():
    accepted, candidate_norm = _optimizer._newton_candidate_status(
        jnp.asarray((0.5,), dtype=jnp.float64),
        jnp.asarray(0.5, dtype=jnp.float64),
        jnp.asarray((2.0,), dtype=jnp.float64),
        alpha=jnp.asarray(1.0, dtype=jnp.float64),
        current_val=jnp.asarray(1.0, dtype=jnp.float64),
        current_grad=jnp.asarray((1.0,), dtype=jnp.float64),
        current_norm=jnp.asarray(1.0, dtype=jnp.float64),
        dx=jnp.asarray((1.0,), dtype=jnp.float64),
    )

    assert float(candidate_norm) > 1.0
    assert bool(accepted)
