from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import simsopt_jax.solve.fullspace_gauss_newton_canary as fullspace_module
from simsopt_jax.objectives.single_stage_fullspace_residuals import (
    ObjectiveResidualReconstruction,
)
from simsopt_jax.solve.fullspace import FullSpaceScaling
from simsopt_jax.solve.fullspace_sqp import CfsSqp1EndpointDiagnostics

jax.config.update("jax_enable_x64", True)


def test_fullspace_gauss_newton_adapter_compiles_and_recomputes_endpoints(
    monkeypatch,
) -> None:
    anchor = jnp.asarray([0.01, -0.02, 0.0], dtype=jnp.float64)
    scaling = FullSpaceScaling(
        bootstrap_anchor=anchor,
        variable_scale=jnp.asarray([2.0, 0.5, 1.0], dtype=jnp.float64),
        constraint_inverse_scale=jnp.asarray([3.0], dtype=jnp.float64),
    )
    monkeypatch.setattr(
        fullspace_module,
        "fullspace_scaling_from_bootstrap",
        lambda _state, _problem: scaling,
    )
    certificate = ObjectiveResidualReconstruction(
        reconstructed_value=jnp.asarray(0.0, dtype=jnp.float64),
        authoritative_value=jnp.asarray(0.0, dtype=jnp.float64),
        value_scaled_defect=jnp.asarray(0.0, dtype=jnp.float64),
        gradient_scaled_defect=jnp.asarray(0.0, dtype=jnp.float64),
        residual_valid=jnp.asarray(True),
        all_finite=jnp.asarray(True),
    )
    monkeypatch.setattr(
        fullspace_module,
        "certify_fullspace_objective_residuals",
        lambda _state, _problem: certificate,
    )
    residual_matrix = jnp.asarray(
        [[1.0, 0.0, 0.0], [0.0, 3.0, 0.0], [1.0, -1.0, 0.0]],
        dtype=jnp.float64,
    )
    monkeypatch.setattr(
        fullspace_module,
        "fullspace_objective_residual_vector",
        lambda physical, _problem: residual_matrix @ physical,
    )

    def joint(
        optimizer_coordinates: jax.Array,
        _problem: object,
        _scaling: FullSpaceScaling,
    ) -> tuple[jax.Array, jax.Array]:
        physical = anchor + scaling.variable_scale * optimizer_coordinates
        residual = residual_matrix @ physical
        return 0.5 * jnp.vdot(residual, residual), jnp.reshape(physical[2], (1,))

    monkeypatch.setattr(fullspace_module, "cfs_sqp1_joint_value_constraints", joint)
    endpoint_calls: list[tuple[int, ...]] = []

    def diagnostics(
        optimizer_coordinates: jax.Array,
        multipliers: jax.Array,
        _problem: object,
        active_scaling: FullSpaceScaling,
    ) -> CfsSqp1EndpointDiagnostics:
        physical = (
            active_scaling.bootstrap_anchor
            + active_scaling.variable_scale * optimizer_coordinates
        )
        endpoint_calls.append(physical.shape)
        raw_constraints = jnp.reshape(physical[2], (1,))
        residual = residual_matrix @ physical
        raw_gradient = residual_matrix.T @ residual
        raw_stationarity = raw_gradient.at[2].add(3.0 * multipliers[0])
        return CfsSqp1EndpointDiagnostics(
            physical_state=physical,
            physical_objective=0.5 * jnp.vdot(residual, residual),
            raw_constraints=raw_constraints,
            scaled_constraints=3.0 * raw_constraints,
            scaled_multipliers=multipliers,
            raw_multipliers=3.0 * multipliers,
            raw_stationarity_residual=raw_stationarity,
            raw_constraint_infinity_norm=jnp.linalg.norm(raw_constraints, ord=jnp.inf),
            scaled_constraint_infinity_norm=jnp.linalg.norm(
                3.0 * raw_constraints,
                ord=jnp.inf,
            ),
            raw_kkt_stationarity_infinity_norm=jnp.linalg.norm(
                raw_stationarity,
                ord=jnp.inf,
            ),
            all_finite=jnp.asarray(True),
        )

    monkeypatch.setattr(fullspace_module, "cfs_sqp1_endpoint_diagnostics", diagnostics)

    def run(state: jax.Array):
        return fullspace_module.run_fullspace_gauss_newton_canary(
            object(),
            state,
            trust_radius=0.25,
            maximum_iterations=8,
        )

    executable = jax.jit(run).lower(anchor).compile()
    result = executable(anchor)

    np.testing.assert_array_equal(result.scaling.bootstrap_anchor, anchor)
    assert endpoint_calls == [anchor.shape, anchor.shape, anchor.shape]
    assert bool(result.all_finite)
    assert bool(result.both_variants_usable)
    assert float(result.gauss_newton_hvp_bilinear_symmetry_relative_defect) <= 1.0e-13
    assert float(result.gauss_newton_probe_normalized_curvature) >= -1.0e-13
