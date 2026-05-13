"""Package export checks for JAX solve helper modules."""

from __future__ import annotations

import simsopt.solve as solve
from simsopt.solve import permanent_magnet_optimization_jax as pm_jax
from simsopt.solve import wireframe_optimization_jax as wireframe_jax


def test_permanent_magnet_jax_solve_helpers_are_public_exports():
    assert solve.relax_and_split_jax is pm_jax.relax_and_split_jax
    assert solve.GPMO_ArbVec_backtracking_jax is pm_jax.GPMO_ArbVec_backtracking_jax
    assert solve.setup_initial_condition_jax is pm_jax.setup_initial_condition_jax


def test_wireframe_jax_solve_helpers_are_public_exports():
    assert solve.optimize_wireframe_jax is wireframe_jax.optimize_wireframe_jax
    assert solve.greedy_stellarator_coil_optimization_jax is (
        wireframe_jax.greedy_stellarator_coil_optimization_jax
    )
    assert solve.regularized_constrained_least_squares_jax is (
        wireframe_jax.regularized_constrained_least_squares_jax
    )
