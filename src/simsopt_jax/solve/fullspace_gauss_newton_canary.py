"""Fullspace adapter for the matrix-free Gauss--Newton curvature canary."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from simsopt_jax.geo.optimizers.projected_hvp_trust_region import (
    ProjectedCurvatureCanaryResult,
    ProjectedHvpCanaryEndpoint,
    run_projected_curvature_canary,
)
from simsopt_jax.objectives.single_stage_fullspace import (
    FullSpaceProblem,
    fullspace_objective_residual_vector,
)
from simsopt_jax.objectives.single_stage_fullspace_residuals import (
    ObjectiveResidualReconstruction,
    certify_fullspace_objective_residuals,
)

from .fullspace import (
    FullSpaceScaling,
    fullspace_physical_coordinates,
    fullspace_scaling_from_bootstrap,
)
from .fullspace_sqp import (
    CfsSqp1EndpointDiagnostics,
    cfs_sqp1_endpoint_diagnostics,
    cfs_sqp1_joint_value_constraints,
)


class FullSpaceGaussNewtonCanaryEndpoint(NamedTuple):
    optimizer: ProjectedHvpCanaryEndpoint
    physical: CfsSqp1EndpointDiagnostics


class FullSpaceGaussNewtonCanaryResult(NamedTuple):
    scaling: FullSpaceScaling
    residual_reconstruction: ObjectiveResidualReconstruction
    objective_residual_size: jax.Array
    initial: FullSpaceGaussNewtonCanaryEndpoint
    identity: FullSpaceGaussNewtonCanaryEndpoint
    gauss_newton: FullSpaceGaussNewtonCanaryEndpoint
    gauss_newton_hvp_bilinear_symmetry_relative_defect: jax.Array
    gauss_newton_probe_normalized_curvature: jax.Array
    gauss_newton_terminal_normalized_curvature: jax.Array
    both_variants_usable: jax.Array
    gauss_newton_supported: jax.Array
    all_finite: jax.Array


def run_fullspace_gauss_newton_canary(
    problem: FullSpaceProblem,
    bootstrap_state: jax.Array,
    *,
    trust_radius: float = 2.0**-10,
    maximum_iterations: int = 32,
) -> FullSpaceGaussNewtonCanaryResult:
    """Compare identity and matrix-free GN projected trust-region steps."""

    scaling = fullspace_scaling_from_bootstrap(bootstrap_state, problem)
    initial_coordinates = jnp.zeros_like(bootstrap_state)
    residual_reconstruction = certify_fullspace_objective_residuals(
        bootstrap_state,
        problem,
    )

    def residual_vector(optimizer_coordinates: jax.Array) -> jax.Array:
        physical_state = fullspace_physical_coordinates(
            optimizer_coordinates,
            scaling,
        )
        return fullspace_objective_residual_vector(physical_state, problem)

    initial_residual, residual_pushforward = jax.linearize(
        residual_vector,
        initial_coordinates,
    )
    residual_pullback = jax.linear_transpose(
        residual_pushforward,
        initial_coordinates,
    )

    def gauss_newton_hvp(vector: jax.Array) -> jax.Array:
        return residual_pullback(residual_pushforward(vector))[0]

    indices = jnp.arange(
        1,
        initial_coordinates.size + 1,
        dtype=initial_coordinates.dtype,
    )
    probe = jnp.sin(indices)
    probe = probe / jnp.linalg.norm(probe)
    probe_hvp = gauss_newton_hvp(probe)
    tiny = jnp.asarray(jnp.finfo(initial_coordinates.dtype).tiny)
    curvature_scale = jnp.maximum(
        tiny,
        jnp.linalg.norm(probe) * jnp.linalg.norm(probe_hvp),
    )
    probe_normalized_curvature = jnp.vdot(probe, probe_hvp) / curvature_scale
    candidate_valid = (
        residual_reconstruction.all_finite
        & residual_reconstruction.residual_valid
        & (residual_reconstruction.value_scaled_defect <= 1.0e-12)
        & (residual_reconstruction.gradient_scaled_defect <= 1.0e-10)
        & jnp.all(jnp.isfinite(initial_residual))
        & jnp.all(jnp.isfinite(probe_hvp))
        & jnp.isfinite(probe_normalized_curvature)
        & (probe_normalized_curvature >= -1.0e-10)
    )

    def joint_value_constraints(
        optimizer_coordinates: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return cfs_sqp1_joint_value_constraints(
            optimizer_coordinates,
            problem,
            scaling,
        )

    optimizer_result: ProjectedCurvatureCanaryResult = run_projected_curvature_canary(
        joint_value_constraints,
        initial_coordinates,
        gauss_newton_hvp,
        candidate_valid=candidate_valid,
        trust_radius=trust_radius,
        maximum_iterations=maximum_iterations,
    )

    def endpoint(
        optimizer_endpoint: ProjectedHvpCanaryEndpoint,
    ) -> FullSpaceGaussNewtonCanaryEndpoint:
        physical = cfs_sqp1_endpoint_diagnostics(
            optimizer_endpoint.coordinates,
            optimizer_endpoint.multipliers,
            problem,
            scaling,
        )
        return FullSpaceGaussNewtonCanaryEndpoint(
            optimizer=optimizer_endpoint,
            physical=physical,
        )

    initial = endpoint(optimizer_result.initial)
    identity = endpoint(optimizer_result.identity)
    gauss_newton = endpoint(optimizer_result.candidate)
    both_variants_usable = (
        optimizer_result.both_variants_usable
        & jnp.isfinite(optimizer_result.candidate_terminal_normalized_curvature)
        & (optimizer_result.candidate_terminal_normalized_curvature >= -1.0e-10)
        & initial.physical.all_finite
        & identity.physical.all_finite
        & gauss_newton.physical.all_finite
    )
    gauss_newton_supported = (
        both_variants_usable
        & (
            gauss_newton.physical.physical_objective
            <= initial.physical.physical_objective
        )
        & (
            gauss_newton.physical.raw_kkt_stationarity_infinity_norm
            <= 0.5
            * jnp.minimum(
                initial.physical.raw_kkt_stationarity_infinity_norm,
                identity.physical.raw_kkt_stationarity_infinity_norm,
            )
        )
    )
    all_finite = (
        optimizer_result.all_finite
        & residual_reconstruction.all_finite
        & jnp.isfinite(probe_normalized_curvature)
        & initial.physical.all_finite
        & identity.physical.all_finite
        & gauss_newton.physical.all_finite
    )
    return FullSpaceGaussNewtonCanaryResult(
        scaling=scaling,
        residual_reconstruction=residual_reconstruction,
        objective_residual_size=jnp.asarray(initial_residual.size, dtype=jnp.int32),
        initial=initial,
        identity=identity,
        gauss_newton=gauss_newton,
        gauss_newton_hvp_bilinear_symmetry_relative_defect=(
            optimizer_result.candidate_hvp_bilinear_symmetry_relative_defect
        ),
        gauss_newton_probe_normalized_curvature=probe_normalized_curvature,
        gauss_newton_terminal_normalized_curvature=(
            optimizer_result.candidate_terminal_normalized_curvature
        ),
        both_variants_usable=both_variants_usable,
        gauss_newton_supported=gauss_newton_supported,
        all_finite=all_finite,
    )


__all__ = [
    "FullSpaceGaussNewtonCanaryEndpoint",
    "FullSpaceGaussNewtonCanaryResult",
    "run_fullspace_gauss_newton_canary",
]
