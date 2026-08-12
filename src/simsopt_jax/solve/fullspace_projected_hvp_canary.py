"""Fullspace adapter for the projected exact-HVP trust-region canary."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from simsopt_jax.geo.optimizers.projected_hvp_trust_region import (
    ProjectedHvpCanaryEndpoint,
    ProjectedHvpCanaryResult,
    run_projected_hvp_canary,
)
from simsopt_jax.objectives.single_stage_fullspace import FullSpaceProblem

from .fullspace import FullSpaceScaling, fullspace_scaling_from_bootstrap
from .fullspace_sqp import (
    CfsSqp1EndpointDiagnostics,
    cfs_sqp1_endpoint_diagnostics,
    cfs_sqp1_joint_value_constraints,
)


class FullSpaceProjectedHvpCanaryEndpoint(NamedTuple):
    optimizer: ProjectedHvpCanaryEndpoint
    physical: CfsSqp1EndpointDiagnostics


class FullSpaceProjectedHvpCanaryResult(NamedTuple):
    scaling: FullSpaceScaling
    initial: FullSpaceProjectedHvpCanaryEndpoint
    identity: FullSpaceProjectedHvpCanaryEndpoint
    exact: FullSpaceProjectedHvpCanaryEndpoint
    exact_hvp_bilinear_symmetry_relative_defect: jax.Array
    both_variants_usable: jax.Array
    exact_hvp_supported: jax.Array
    all_finite: jax.Array


def run_fullspace_projected_hvp_canary(
    problem: FullSpaceProblem,
    bootstrap_state: jax.Array,
    *,
    trust_radius: float = 2.0**-10,
    maximum_iterations: int = 32,
) -> FullSpaceProjectedHvpCanaryResult:
    """Compare identity and exact-HVP projected trust-region steps."""

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

    optimizer_result: ProjectedHvpCanaryResult = run_projected_hvp_canary(
        joint_value_constraints,
        initial_coordinates,
        trust_radius=trust_radius,
        maximum_iterations=maximum_iterations,
    )

    def endpoint(
        optimizer_endpoint: ProjectedHvpCanaryEndpoint,
    ) -> FullSpaceProjectedHvpCanaryEndpoint:
        physical = cfs_sqp1_endpoint_diagnostics(
            optimizer_endpoint.coordinates,
            optimizer_endpoint.multipliers,
            problem,
            scaling,
        )
        return FullSpaceProjectedHvpCanaryEndpoint(
            optimizer=optimizer_endpoint,
            physical=physical,
        )

    initial = endpoint(optimizer_result.initial)
    identity = endpoint(optimizer_result.identity)
    exact = endpoint(optimizer_result.exact)
    both_variants_usable = (
        optimizer_result.both_variants_usable
        & initial.physical.all_finite
        & identity.physical.all_finite
        & exact.physical.all_finite
    )
    exact_hvp_supported = (
        both_variants_usable
        & (exact.physical.physical_objective <= initial.physical.physical_objective)
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
    return FullSpaceProjectedHvpCanaryResult(
        scaling=scaling,
        initial=initial,
        identity=identity,
        exact=exact,
        exact_hvp_bilinear_symmetry_relative_defect=(
            optimizer_result.exact_hvp_bilinear_symmetry_relative_defect
        ),
        both_variants_usable=both_variants_usable,
        exact_hvp_supported=exact_hvp_supported,
        all_finite=all_finite,
    )


__all__ = [
    "FullSpaceProjectedHvpCanaryEndpoint",
    "FullSpaceProjectedHvpCanaryResult",
    "run_fullspace_projected_hvp_canary",
]
