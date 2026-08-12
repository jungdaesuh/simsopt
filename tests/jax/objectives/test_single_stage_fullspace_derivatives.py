"""Matrix-free derivative tests for the coupled single-stage full-space core."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt_jax.core.specs import (
    make_coil_dof_extraction_spec,
    make_coil_set_dof_extraction_spec,
    make_curve_xyzfourier_spec,
    make_optimizable_dof_map_spec,
    make_surface_xyz_tensor_fourier_spec,
)
from simsopt_jax.core.surface_fourier_indices import stellsym_scatter_indices
from simsopt_jax.objectives.single_stage_fullspace import (
    FullSpaceLayout,
    FullSpaceObjectiveConfig,
    FullSpaceProblem,
    FullSpaceState,
    fullspace_constraint_jvp,
    fullspace_constraint_vector,
    fullspace_constraint_vjp,
    fullspace_feasibility_primitives,
    fullspace_kkt_primitives,
    fullspace_value,
    fullspace_value_and_grad,
)
from simsopt_jax.solve.fullspace import (
    FullSpaceRoute,
    cfs_p0_diagnostics,
    cfs_p0_value_and_grad,
    fullspace_optimizer_coordinates,
    fullspace_physical_coordinates,
    fullspace_scaling_from_bootstrap,
    prepare_cfs_p0,
    route_policy,
)

jax.config.update("jax_enable_x64", True)


def _toroidal_surface_dofs() -> np.ndarray:
    coefficients = np.zeros((3, 3, 1), dtype=np.float64)
    coefficients[0, 0, 0] = 1.0
    coefficients[0, 1, 0] = 0.2
    coefficients[2, 2, 0] = -0.2
    return coefficients.reshape(-1)[stellsym_scatter_indices(1, 0)]


def _surface_spec(dofs: np.ndarray, *, nphi: int, ntheta: int):
    return make_surface_xyz_tensor_fourier_spec(
        dofs=dofs,
        quadpoints_phi=np.linspace(0.0, 1.0, nphi, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, ntheta, endpoint=False),
        nfp=1,
        stellsym=True,
        mpol=1,
        ntor=0,
    )


def _problem_and_z() -> tuple[FullSpaceProblem, jax.Array]:
    surface_dofs = _toroidal_surface_dofs()
    curve_dofs = np.asarray(
        (1.55, 0.0, 0.0, 0.0, 0.0, 0.32, 0.0, 0.32, 0.0),
        dtype=np.float64,
    )
    curve_spec = make_curve_xyzfourier_spec(
        dofs=curve_dofs,
        quadpoints=np.linspace(0.0, 1.0, 12, endpoint=False),
        order=1,
    )
    curve_map = make_optimizable_dof_map_spec(
        template_full_dofs=np.zeros(curve_dofs.size),
        owner_segments=((0, curve_dofs.size, 0, curve_dofs.size),),
        input_mode="full",
        input_start=0,
        input_end=curve_dofs.size,
    )
    fixed_current_map = make_optimizable_dof_map_spec(
        template_full_dofs=np.asarray((1.0e5,)),
        owner_segments=(),
        input_mode="full",
        input_start=0,
        input_end=1,
    )
    extraction = make_coil_set_dof_extraction_spec(
        (
            make_coil_dof_extraction_spec(
                curve=curve_spec,
                curve_map=curve_map,
                current_map=fixed_current_map,
            ),
        )
    )
    exact_surface = _surface_spec(surface_dofs, nphi=3, ntheta=4)
    layout = FullSpaceLayout(curve_dofs.size, surface_dofs.size)
    config = FullSpaceObjectiveConfig(
        iota_target=jnp.asarray(0.37),
        major_radius_target=jnp.asarray(0.96),
        length_target=jnp.asarray(1.8),
        volume_target=jnp.asarray(0.78),
        non_qs_weight=jnp.asarray(1.3),
        residual_weight=jnp.asarray(0.7),
        iota_weight=jnp.asarray(1.1),
        major_radius_weight=jnp.asarray(0.9),
        length_weight=jnp.asarray(1.2),
        non_qs_axis=1,
        weight_inv_modB=False,
        length_coil_indices=(0,),
    )
    residual_size = 3 * 3 * 4
    problem = FullSpaceProblem(
        coil_dof_extraction=extraction,
        exact_surface_template=exact_surface,
        label_surface_template=_surface_spec(surface_dofs, nphi=4, ntheta=5),
        non_qs_surface_template=_surface_spec(surface_dofs, nphi=4, ntheta=5),
        exact_mask_indices=jnp.asarray((1, 7, residual_size - 2), dtype=jnp.int32),
        config=config,
        layout=layout,
    )
    z = layout.pack(
        FullSpaceState(
            coil_dofs=jnp.asarray(curve_dofs),
            surface_dofs=jnp.asarray(surface_dofs),
            iota=jnp.asarray(0.31),
            G=jnp.asarray(0.85),
        )
    )
    changed_z = z.at[0].add(0.017).at[-2].add(-0.023).at[-1].add(0.041)
    return problem, changed_z


def _cfs_p0_problem_and_z() -> tuple[FullSpaceProblem, jax.Array]:
    problem, z = _problem_and_z()
    canonical_mask = jnp.resize(problem.exact_mask_indices, (254,))
    return replace(problem, exact_mask_indices=canonical_mask), z


def test_joint_gradient_matches_changed_state_directional_difference() -> None:
    problem, z = _problem_and_z()
    direction = jnp.linspace(-0.4, 0.5, z.size, dtype=jnp.float64)
    direction /= jnp.linalg.norm(direction)
    value, gradient = jax.jit(fullspace_value_and_grad)(z, problem)
    step = jnp.asarray(2.0e-5, dtype=jnp.float64)
    finite_difference = (
        fullspace_value(z + step * direction, problem)
        - fullspace_value(z - step * direction, problem)
    ) / (2.0 * step)

    assert np.isfinite(np.asarray(value))
    assert np.all(np.isfinite(np.asarray(gradient)))
    np.testing.assert_allclose(
        jnp.vdot(gradient, direction),
        finite_difference,
        rtol=3.0e-5,
        atol=3.0e-7,
    )


def test_matrix_free_jvp_vjp_obey_nonsymmetric_transpose_identity() -> None:
    problem, z = _problem_and_z()
    tangent = jnp.linspace(0.2, -0.3, z.size, dtype=jnp.float64)
    cotangent = jnp.asarray((0.7, -0.4, 0.2, -0.6), dtype=jnp.float64)
    jvp = jax.jit(fullspace_constraint_jvp)(z, tangent, problem)
    vjp = jax.jit(fullspace_constraint_vjp)(z, cotangent, problem)

    assert jvp.shape != tangent.shape
    np.testing.assert_allclose(
        jnp.vdot(cotangent, jvp),
        jnp.vdot(vjp, tangent),
        rtol=2.0e-12,
        atol=2.0e-10,
    )


def test_feasibility_and_kkt_primitives_are_finite_and_jittable() -> None:
    problem, z = _problem_and_z()
    constraint_residual = fullspace_constraint_vector(z, problem)
    multipliers = jnp.linspace(
        -0.2,
        0.3,
        constraint_residual.size,
        dtype=jnp.float64,
    )
    feasibility = jax.jit(fullspace_feasibility_primitives)(z, problem)
    kkt = jax.jit(fullspace_kkt_primitives)(z, multipliers, problem)
    _, objective_gradient = fullspace_value_and_grad(z, problem)
    expected_stationarity = objective_gradient + fullspace_constraint_vjp(
        z,
        multipliers,
        problem,
    )

    assert bool(feasibility.all_finite)
    assert bool(kkt.all_finite)
    np.testing.assert_allclose(
        feasibility.residual,
        constraint_residual,
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        feasibility.infinity_norm,
        jnp.max(jnp.abs(constraint_residual)),
    )
    np.testing.assert_allclose(
        kkt.stationarity_residual,
        expected_stationarity,
        rtol=2.0e-12,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        kkt.stationarity_inf,
        jnp.max(jnp.abs(expected_stationarity)),
    )


def test_joint_gradient_does_not_execute_legacy_implicit_adjoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from simsopt_jax_adapters.geo import surface_objectives

    problem, z = _problem_and_z()

    def forbidden_implicit_adjoint(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("legacy nested implicit adjoint was executed")

    monkeypatch.setattr(
        surface_objectives,
        "_solve_boozer_adjoint",
        forbidden_implicit_adjoint,
    )
    value, gradient = fullspace_value_and_grad(z, problem)

    assert np.isfinite(np.asarray(value))
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_cfs_p0_uses_centered_scaled_coordinates_with_exact_round_trip() -> None:
    problem, bootstrap = _cfs_p0_problem_and_z()
    scaling = fullspace_scaling_from_bootstrap(bootstrap, problem)
    optimizer_bootstrap = fullspace_optimizer_coordinates(bootstrap, scaling)
    changed_state = bootstrap + jnp.linspace(
        -0.02,
        0.03,
        bootstrap.size,
        dtype=jnp.float64,
    )
    optimizer_changed = fullspace_optimizer_coordinates(changed_state, scaling)

    np.testing.assert_array_equal(optimizer_bootstrap, jnp.zeros_like(bootstrap))
    np.testing.assert_allclose(
        fullspace_physical_coordinates(optimizer_changed, scaling),
        changed_state,
        rtol=0.0,
        atol=2.0e-16,
    )


def test_cfs_p0_rejects_noncanonical_boozer_constraint_count() -> None:
    problem, bootstrap = _problem_and_z()

    with pytest.raises(
        ValueError,
        match="frozen 254-component Boozer equality mask",
    ):
        fullspace_scaling_from_bootstrap(bootstrap, problem)


def test_cfs_p0_fused_value_gradient_matches_frozen_merit_formula() -> None:
    problem, bootstrap = _cfs_p0_problem_and_z()
    scaling = fullspace_scaling_from_bootstrap(bootstrap, problem)
    optimizer_coordinates = jnp.linspace(
        -0.01,
        0.015,
        bootstrap.size,
        dtype=jnp.float64,
    )
    (value, diagnostics), gradient = jax.jit(cfs_p0_value_and_grad)(
        optimizer_coordinates,
        problem,
        scaling,
    )
    physical_state = fullspace_physical_coordinates(
        optimizer_coordinates,
        scaling,
    )
    raw_constraints = fullspace_constraint_vector(physical_state, problem)
    scaled_constraints = raw_constraints * scaling.constraint_inverse_scale
    expected = fullspace_value(physical_state, problem) + 5.0 * jnp.vdot(
        scaled_constraints,
        scaled_constraints,
    )

    np.testing.assert_allclose(value, expected, rtol=2.0e-15, atol=2.0e-15)
    np.testing.assert_allclose(
        diagnostics.scaled_penalty_value,
        expected,
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    assert bool(diagnostics.all_finite)
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_cfs_p0_scaled_gradient_matches_directional_difference() -> None:
    problem, bootstrap = _cfs_p0_problem_and_z()
    scaling = fullspace_scaling_from_bootstrap(bootstrap, problem)
    optimizer_coordinates = jnp.linspace(
        -0.008,
        0.012,
        bootstrap.size,
        dtype=jnp.float64,
    )
    direction = jnp.linspace(
        0.4,
        -0.3,
        bootstrap.size,
        dtype=jnp.float64,
    )
    direction /= jnp.linalg.norm(direction)
    (_, _), gradient = cfs_p0_value_and_grad(
        optimizer_coordinates,
        problem,
        scaling,
    )
    step = jnp.asarray(1.0e-5, dtype=jnp.float64)
    plus = cfs_p0_diagnostics(
        optimizer_coordinates + step * direction,
        problem,
        scaling,
    ).scaled_penalty_value
    minus = cfs_p0_diagnostics(
        optimizer_coordinates - step * direction,
        problem,
        scaling,
    ).scaled_penalty_value
    finite_difference = (plus - minus) / (2.0 * step)

    np.testing.assert_allclose(
        jnp.vdot(gradient, direction),
        finite_difference,
        rtol=4.0e-5,
        atol=5.0e-7,
    )


def test_prepared_cfs_p0_executes_ten_step_public_fused_state_machine() -> None:
    problem, bootstrap = _cfs_p0_problem_and_z()
    changed_state = bootstrap.at[0].add(2.0e-3).at[-2].add(-1.0e-3)
    prepared = prepare_cfs_p0(problem, bootstrap, changed_state)
    result = prepared.run(maximum_iterations=10)

    assert int(np.asarray(result.optimizer.iterations)) == 10
    assert bool(np.asarray(result.all_finite))
    assert bool(np.asarray(result.made_progress))
    assert int(np.asarray(result.nonfinite_evaluation_count)) == 0
    np.testing.assert_allclose(
        result.optimizer.state.objective_value,
        result.final_diagnostics.scaled_penalty_value,
        rtol=2.0e-13,
        atol=2.0e-12,
    )
    assert all(
        isinstance(leaf, jax.Array) for leaf in jax.tree_util.tree_leaves(result)
    )


def test_cfs_al2_policy_only_escalates_the_inner_accuracy_budget() -> None:
    al1 = route_policy(FullSpaceRoute.CFS_AL1)
    al2 = route_policy(FullSpaceRoute.CFS_AL2)

    assert al1.inner_iterations_per_stage == 100
    assert al1.maximum_total_inner_iterations == 1000
    assert al2.inner_iterations_per_stage == 1000
    assert al2.maximum_total_inner_iterations == 10000
    assert (
        replace(
            al2,
            route=al1.route,
            inner_iterations_per_stage=al1.inner_iterations_per_stage,
            maximum_total_inner_iterations=al1.maximum_total_inner_iterations,
            selection_rule=al1.selection_rule,
        )
        == al1
    )
