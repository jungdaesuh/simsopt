"""Fullspace adapter for the diagnostic exact-curvature A/B canary."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from simsopt_jax.geo.optimizers.curvature_canary import (
    CurvatureCanaryEndpoint,
    DenseCurvatureCanaryResult,
    run_dense_curvature_canary,
)
from simsopt_jax.objectives.single_stage_fullspace import FullSpaceProblem

from .fullspace import (
    FullSpaceScaling,
    fullspace_scaling_from_bootstrap,
)
from .fullspace_sqp import (
    CfsSqp1EndpointDiagnostics,
    cfs_sqp1_endpoint_diagnostics,
    cfs_sqp1_joint_value_constraints,
)


class FullSpaceCurvatureCanaryEndpoint(NamedTuple):
    optimizer: CurvatureCanaryEndpoint
    physical: CfsSqp1EndpointDiagnostics


class FullSpaceCurvatureCanaryResult(NamedTuple):
    scaling: FullSpaceScaling
    initial: FullSpaceCurvatureCanaryEndpoint
    identity: FullSpaceCurvatureCanaryEndpoint
    exact: FullSpaceCurvatureCanaryEndpoint
    exact_hessian: jax.Array
    exact_hessian_symmetry_relative_defect: jax.Array
    exact_hessian_action_relative_defect: jax.Array
    both_variants_usable: jax.Array
    curvature_hypothesis_supported: jax.Array
    all_finite: jax.Array


def run_fullspace_curvature_canary(
    problem: FullSpaceProblem,
    bootstrap_state: jax.Array,
    *,
    trust_radius: float = 1.0 / 64.0,
    hessian_batch_width: int = 1,
) -> FullSpaceCurvatureCanaryResult:
    """Run the one-step identity-versus-exact curvature comparison."""

    scaling = fullspace_scaling_from_bootstrap(bootstrap_state, problem)
    initial_coordinates = jnp.zeros_like(bootstrap_state)

    def joint_value_constraints(
        optimizer_coordinates: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return cfs_sqp1_joint_value_constraints(
            optimizer_coordinates,
            problem,
            scaling,
        )

    optimizer_result: DenseCurvatureCanaryResult = run_dense_curvature_canary(
        joint_value_constraints,
        initial_coordinates,
        trust_radius=trust_radius,
        hessian_batch_width=hessian_batch_width,
    )

    def endpoint(
        optimizer_endpoint: CurvatureCanaryEndpoint,
    ) -> FullSpaceCurvatureCanaryEndpoint:
        physical = cfs_sqp1_endpoint_diagnostics(
            optimizer_endpoint.coordinates,
            optimizer_endpoint.multipliers,
            problem,
            scaling,
        )
        return FullSpaceCurvatureCanaryEndpoint(
            optimizer=optimizer_endpoint,
            physical=physical,
        )

    initial = endpoint(optimizer_result.initial)
    identity = endpoint(optimizer_result.identity)
    exact = endpoint(optimizer_result.exact)
    curvature_hypothesis_supported = (
        optimizer_result.both_variants_usable
        & exact.physical.all_finite
        & identity.physical.all_finite
        & (
            exact.physical.raw_kkt_stationarity_infinity_norm
            <= 0.5
            * jnp.minimum(
                initial.physical.raw_kkt_stationarity_infinity_norm,
                identity.physical.raw_kkt_stationarity_infinity_norm,
            )
        )
    )
    all_finite = (
        optimizer_result.all_finite
        & initial.physical.all_finite
        & identity.physical.all_finite
        & exact.physical.all_finite
    )
    return FullSpaceCurvatureCanaryResult(
        scaling=scaling,
        initial=initial,
        identity=identity,
        exact=exact,
        exact_hessian=optimizer_result.exact_hessian,
        exact_hessian_symmetry_relative_defect=(
            optimizer_result.exact_hessian_symmetry_relative_defect
        ),
        exact_hessian_action_relative_defect=(
            optimizer_result.exact_hessian_action_relative_defect
        ),
        both_variants_usable=optimizer_result.both_variants_usable,
        curvature_hypothesis_supported=curvature_hypothesis_supported,
        all_finite=all_finite,
    )


__all__ = [
    "FullSpaceCurvatureCanaryEndpoint",
    "FullSpaceCurvatureCanaryResult",
    "run_fullspace_curvature_canary",
]
