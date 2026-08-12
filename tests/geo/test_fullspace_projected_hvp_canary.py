from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import simsopt_jax.solve.fullspace_projected_hvp_canary as fullspace_module
from simsopt_jax.solve.fullspace import FullSpaceScaling
from simsopt_jax.solve.fullspace_sqp import CfsSqp1EndpointDiagnostics

jax.config.update("jax_enable_x64", True)


def test_fullspace_adapter_preserves_scaling_and_recomputes_raw_endpoints(
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

    curvature = jnp.diag(jnp.asarray([1.0, 8.0, 2.0], dtype=jnp.float64))

    def joint(
        optimizer_coordinates: jax.Array,
        _problem: object,
        _scaling: FullSpaceScaling,
    ) -> tuple[jax.Array, jax.Array]:
        objective = 0.5 * optimizer_coordinates @ curvature @ optimizer_coordinates
        objective += jnp.asarray([0.2, -0.1, 0.0]) @ optimizer_coordinates
        return objective, jnp.reshape(optimizer_coordinates[2], (1,))

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
        raw_stationarity = physical.at[2].add(3.0 * multipliers[0])
        objective = 0.5 * jnp.vdot(physical, physical)
        return CfsSqp1EndpointDiagnostics(
            physical_state=physical,
            physical_objective=objective,
            raw_constraints=raw_constraints,
            scaled_constraints=3.0 * raw_constraints,
            scaled_multipliers=multipliers,
            raw_multipliers=3.0 * multipliers,
            raw_stationarity_residual=raw_stationarity,
            raw_constraint_infinity_norm=jnp.linalg.norm(raw_constraints, ord=jnp.inf),
            scaled_constraint_infinity_norm=jnp.linalg.norm(
                3.0 * raw_constraints, ord=jnp.inf
            ),
            raw_kkt_stationarity_infinity_norm=jnp.linalg.norm(
                raw_stationarity, ord=jnp.inf
            ),
            all_finite=jnp.asarray(True),
        )

    monkeypatch.setattr(fullspace_module, "cfs_sqp1_joint_value_constraints", joint)
    monkeypatch.setattr(fullspace_module, "cfs_sqp1_endpoint_diagnostics", diagnostics)

    def run(state: jax.Array):
        return fullspace_module.run_fullspace_projected_hvp_canary(
            object(),
            state,
            trust_radius=0.25,
            maximum_iterations=8,
        )

    executable = jax.jit(run).lower(anchor).compile()
    result = executable(anchor)

    np.testing.assert_array_equal(result.scaling.bootstrap_anchor, anchor)
    assert len(endpoint_calls) == 3
    assert endpoint_calls == [anchor.shape, anchor.shape, anchor.shape]
    np.testing.assert_allclose(result.initial.physical.physical_state, anchor)
    assert bool(result.all_finite)
    assert float(result.exact_hvp_bilinear_symmetry_relative_defect) <= 1.0e-13


def test_fullspace_adapter_lowers_and_compiles() -> None:
    curvature = jnp.diag(jnp.asarray([1.0, 4.0, 2.0], dtype=jnp.float64))

    def run(initial: jax.Array):
        def joint(x: jax.Array) -> tuple[jax.Array, jax.Array]:
            objective = 0.5 * x @ curvature @ x
            return objective, jnp.reshape(x[2], (1,))

        return fullspace_module.run_projected_hvp_canary(
            joint,
            initial,
            trust_radius=0.5,
            maximum_iterations=8,
        )

    executable = (
        jax.jit(run).lower(jnp.asarray([0.1, -0.2, 0.0], dtype=jnp.float64)).compile()
    )
    result = executable(jnp.asarray([0.1, -0.2, 0.0], dtype=jnp.float64))

    assert bool(result.all_finite)
    assert bool(result.both_variants_usable)
