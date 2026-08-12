from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from simsopt_jax.geo.optimizers.projected_hvp_trust_region import (
    ProjectedSteihaugTermination,
    certified_minimum_norm_correction,
    exact_hvp_bilinear_symmetry_relative_defect,
    factor_certified_gram_projector,
    project_with_certified_gram,
    run_projected_curvature_canary,
    run_projected_hvp_canary,
    solve_projected_steihaug,
)

jax.config.update("jax_enable_x64", True)


def test_certified_gram_projection_is_tangent() -> None:
    jacobian = jnp.asarray([[1.0, 2.0, -1.0], [0.0, 1.0, 1.0]], dtype=jnp.float64)
    vector = jnp.asarray([2.0, -3.0, 0.5], dtype=jnp.float64)

    projector = factor_certified_gram_projector(jacobian)
    projection = project_with_certified_gram(projector, vector)

    np.testing.assert_allclose(jacobian @ projection.projected, 0.0, atol=1.0e-14)
    assert bool(projector.all_finite)
    assert bool(projection.all_finite)
    assert float(projection.solve_relative_residual) <= 1.0e-14
    assert float(projection.solve_forward_error_bound) <= 1.0e-13


def test_spd_projected_steihaug_converges_interior() -> None:
    curvature = jnp.diag(jnp.asarray([2.0, 5.0], dtype=jnp.float64))
    jacobian = jnp.asarray([[0.0, 1.0]], dtype=jnp.float64)
    projector = factor_certified_gram_projector(jacobian)

    result = solve_projected_steihaug(
        lambda vector: curvature @ vector,
        jnp.asarray([2.0, 0.0], dtype=jnp.float64),
        jnp.zeros(2, dtype=jnp.float64),
        projector,
        trust_radius=2.0,
        maximum_iterations=8,
        projected_residual_tolerance=1.0e-12,
    )

    np.testing.assert_allclose(result.combined_step, [-1.0, 0.0], atol=1.0e-14)
    assert result.termination == int(ProjectedSteihaugTermination.INTERIOR_CONVERGED)
    assert result.iterations == 1
    assert result.hvp_evaluations == 1
    assert not result.hit_boundary
    assert result.final_projected_residual_norm <= result.projected_residual_target
    assert result.predicted_reduction == 1.0


def test_nonpositive_curvature_terminates_on_boundary() -> None:
    curvature = jnp.diag(jnp.asarray([-1.0, 3.0], dtype=jnp.float64))
    jacobian = jnp.asarray([[0.0, 1.0]], dtype=jnp.float64)
    projector = factor_certified_gram_projector(jacobian)
    radius = 0.25

    result = solve_projected_steihaug(
        lambda vector: curvature @ vector,
        jnp.asarray([1.0, 0.0], dtype=jnp.float64),
        jnp.zeros(2, dtype=jnp.float64),
        projector,
        trust_radius=radius,
        maximum_iterations=8,
        projected_residual_tolerance=1.0e-12,
    )

    np.testing.assert_allclose(result.combined_step, [-radius, 0.0], atol=1.0e-14)
    assert result.termination == int(ProjectedSteihaugTermination.NONPOSITIVE_CURVATURE)
    assert result.encountered_nonpositive_curvature
    assert result.hit_boundary
    assert result.terminal_curvature < 0.0
    assert result.combined_step_norm == radius
    assert result.tangency_relative_residual <= 1.0e-14


def test_inactive_iterations_do_not_execute_hvp() -> None:
    calls: list[int] = []

    def counted_hvp(vector: jax.Array) -> jax.Array:
        jax.debug.callback(lambda _: calls.append(1), vector[0])
        return 2.0 * vector

    projector = factor_certified_gram_projector(
        jnp.asarray([[0.0, 1.0]], dtype=jnp.float64)
    )

    compiled = jax.jit(
        lambda gradient: solve_projected_steihaug(
            counted_hvp,
            gradient,
            jnp.zeros(2, dtype=jnp.float64),
            projector,
            trust_radius=2.0,
            maximum_iterations=8,
            projected_residual_tolerance=1.0e-12,
        )
    )
    result = compiled(jnp.asarray([2.0, 0.0], dtype=jnp.float64))
    result.combined_step.block_until_ready()

    assert result.iterations == 1
    assert result.hvp_evaluations == 1
    assert len(calls) == 1


def test_minimum_norm_correction_solves_linearized_constraints() -> None:
    point = jnp.asarray([0.8, 0.7], dtype=jnp.float64)
    constraints = jnp.asarray([point @ point - 1.0], dtype=jnp.float64)
    jacobian = jnp.reshape(2.0 * point, (1, 2))

    result = certified_minimum_norm_correction(jacobian, constraints)

    np.testing.assert_allclose(jacobian @ result.correction, -constraints, atol=1.0e-14)
    expected = -(constraints[0] / (jacobian @ jacobian.T)[0, 0]) * jacobian[0]
    np.testing.assert_allclose(result.correction, expected, atol=1.0e-14)
    assert result.relative_residual <= 1.0e-14
    assert result.forward_error_bound <= 1.0e-14
    assert result.all_finite


def test_exact_hvp_matches_finite_difference_and_is_bilinear_symmetric() -> None:
    coordinates = jnp.asarray([0.4, -0.3, 0.2], dtype=jnp.float64)

    def scalar(values: jax.Array) -> jax.Array:
        return (
            values[0] ** 2 * values[1]
            + jnp.sin(values[1] * values[2])
            + 0.25 * values[2] ** 4
        )

    _gradient, exact_hvp = jax.linearize(jax.grad(scalar), coordinates)
    direction = jnp.asarray([0.2, -0.5, 0.7], dtype=jnp.float64)
    epsilon = 1.0e-5
    finite_difference = (
        jax.grad(scalar)(coordinates + epsilon * direction)
        - jax.grad(scalar)(coordinates - epsilon * direction)
    ) / (2.0 * epsilon)

    np.testing.assert_allclose(
        exact_hvp(direction), finite_difference, rtol=1.0e-9, atol=1.0e-10
    )
    assert (
        exact_hvp_bilinear_symmetry_relative_defect(exact_hvp, coordinates) <= 1.0e-14
    )


def test_complete_canary_compiles_and_exact_hvp_beats_identity() -> None:
    curvature = jnp.diag(jnp.asarray([1.0, 100.0, 4.0], dtype=jnp.float64))

    def run(initial_coordinates: jax.Array):
        def joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
            return (
                0.5 * values @ curvature @ values,
                jnp.reshape(values[2], (1,)),
            )

        return run_projected_hvp_canary(
            joint,
            initial_coordinates,
            trust_radius=0.02,
            maximum_iterations=8,
        )

    initial = jnp.asarray([0.01, 0.01, 0.0], dtype=jnp.float64)
    executable = jax.jit(run).lower(initial).compile()
    result = executable(initial)

    assert result.all_finite
    assert result.both_variants_usable
    assert result.exact_hvp_supported
    assert (
        result.exact.scaled_stationarity_inf < result.identity.scaled_stationarity_inf
    )
    assert result.exact.scaled_feasibility_inf <= 1.0e-14
    assert result.exact_hvp_bilinear_symmetry_relative_defect <= 1.0e-14


def test_generic_candidate_matches_exact_hvp_wrapper() -> None:
    curvature = jnp.diag(jnp.asarray([1.0, 100.0, 4.0], dtype=jnp.float64))

    def run(initial_coordinates: jax.Array):
        def joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
            return (
                0.5 * values @ curvature @ values,
                jnp.reshape(values[2], (1,)),
            )

        generic = run_projected_curvature_canary(
            joint,
            initial_coordinates,
            lambda vector: curvature @ vector,
            candidate_valid=jnp.asarray(True),
            trust_radius=0.02,
            maximum_iterations=8,
        )
        exact = run_projected_hvp_canary(
            joint,
            initial_coordinates,
            trust_radius=0.02,
            maximum_iterations=8,
        )
        return generic, exact

    initial = jnp.asarray([0.01, 0.01, 0.0], dtype=jnp.float64)
    generic, exact = jax.jit(run).lower(initial).compile()(initial)

    np.testing.assert_allclose(
        generic.candidate.coordinates, exact.exact.coordinates, atol=1.0e-15
    )
    np.testing.assert_allclose(
        generic.candidate.stationarity, exact.exact.stationarity, atol=1.0e-15
    )
    np.testing.assert_allclose(
        generic.identity.coordinates, exact.identity.coordinates, atol=1.0e-15
    )
    assert generic.candidate_valid
    assert generic.both_variants_usable == exact.both_variants_usable
    assert generic.candidate_supported == exact.exact_hvp_supported
    assert (
        generic.candidate_hvp_bilinear_symmetry_relative_defect
        == exact.exact_hvp_bilinear_symmetry_relative_defect
    )
    assert exact._fields == (
        "initial",
        "identity",
        "exact",
        "exact_hvp_bilinear_symmetry_relative_defect",
        "both_variants_usable",
        "exact_hvp_supported",
        "all_finite",
    )


def test_invalid_candidate_fails_closed_without_invalidating_finite_evidence() -> None:
    curvature = jnp.diag(jnp.asarray([1.0, 100.0, 4.0], dtype=jnp.float64))

    def joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        return (
            0.5 * values @ curvature @ values,
            jnp.reshape(values[2], (1,)),
        )

    result = run_projected_curvature_canary(
        joint,
        jnp.asarray([0.01, 0.01, 0.0], dtype=jnp.float64),
        lambda vector: curvature @ vector,
        candidate_valid=jnp.asarray(False),
        trust_radius=0.02,
        maximum_iterations=8,
    )

    assert result.all_finite
    assert result.identity.usable
    assert result.candidate.usable
    assert not result.candidate_valid
    assert not result.both_variants_usable
    assert not result.candidate_supported


def test_generic_candidate_exposes_actual_negative_search_curvature() -> None:
    candidate_curvature = jnp.diag(jnp.asarray([-1.0, 100.0, 1.0], dtype=jnp.float64))

    def joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        return 0.5 * jnp.vdot(values, values), jnp.reshape(values[2], (1,))

    result = run_projected_curvature_canary(
        joint,
        jnp.asarray([0.1, 0.0, 0.0], dtype=jnp.float64),
        lambda vector: candidate_curvature @ vector,
        candidate_valid=jnp.asarray(True),
        trust_radius=0.02,
        maximum_iterations=8,
    )

    assert result.candidate.cg_negative_curvature
    assert result.candidate_terminal_normalized_curvature < -0.99
