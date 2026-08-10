"""Pure-kernel correctness tests for the single-stage full-space problem."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from simsopt_jax.core.surface_dofs import (
    surface_gamma_tangents_from_dofs,
)
from simsopt_jax.core.curve_geometry import curve_length_from_spec
from simsopt_jax.core.field import (
    coil_set_spec_from_dof_extraction_spec,
    coil_specs_from_dof_extraction_spec,
    grouped_biot_savart_B_from_spec,
)
from simsopt_jax.core.specs import (
    make_coil_dof_extraction_spec,
    make_coil_set_dof_extraction_spec,
    make_curve_xyzfourier_spec,
    make_optimizable_dof_map_spec,
    make_surface_xyz_tensor_fourier_spec,
)
from simsopt_jax.core.surface_fourier_indices import stellsym_scatter_indices
from simsopt_jax.core.surface_integrals import surface_major_radius, surface_volume
from simsopt_jax.geo.boozer_residual import boozer_residual_vector
from simsopt_jax.objectives.single_stage_fullspace import (
    FullSpaceLayout,
    FullSpaceObjectiveConfig,
    FullSpaceProblem,
    FullSpaceState,
    evaluate_fullspace,
    flatten_fullspace_constraints,
    fullspace_constraints,
    fullspace_value,
    fullspace_value_and_grad,
)

jax.config.update("jax_enable_x64", True)


def _toroidal_surface_dofs(*, major_radius: float, minor_radius: float) -> np.ndarray:
    """Return stellsym tensor coefficients for a circular nfp=1 torus."""

    mpol = 1
    ntor = 0
    coefficient_count = (2 * mpol + 1) * (2 * ntor + 1)
    coefficients = np.zeros((3, 2 * mpol + 1, 2 * ntor + 1))
    coefficients[0, 0, 0] = major_radius
    coefficients[0, 1, 0] = minor_radius
    coefficients[2, 2, 0] = -minor_radius
    return coefficients.reshape(3 * coefficient_count)[
        stellsym_scatter_indices(mpol, ntor)
    ]


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


def _fixed_current_coil_extraction():
    curve_dofs = np.asarray(
        (
            1.55,
            0.0,
            0.0,
            0.0,
            0.0,
            0.32,
            0.0,
            0.32,
            0.0,
        ),
        dtype=np.float64,
    )
    curve_spec = make_curve_xyzfourier_spec(
        dofs=curve_dofs,
        quadpoints=np.linspace(0.0, 1.0, 16, endpoint=False),
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
    extraction = make_coil_dof_extraction_spec(
        curve=curve_spec,
        curve_map=curve_map,
        current_map=fixed_current_map,
    )
    return make_coil_set_dof_extraction_spec((extraction,)), curve_dofs


def _problem_and_state() -> tuple[FullSpaceProblem, FullSpaceState]:
    surface_dofs = _toroidal_surface_dofs(major_radius=1.0, minor_radius=0.2)
    extraction, coil_dofs = _fixed_current_coil_extraction()
    exact_surface = _surface_spec(surface_dofs, nphi=4, ntheta=5)
    label_surface = _surface_spec(surface_dofs, nphi=6, ntheta=7)
    non_qs_surface = _surface_spec(surface_dofs, nphi=5, ntheta=6)
    layout = FullSpaceLayout(
        coil_dof_count=coil_dofs.size,
        surface_dof_count=surface_dofs.size,
    )
    config = FullSpaceObjectiveConfig(
        iota_target=jnp.asarray(0.37, dtype=jnp.float64),
        major_radius_target=jnp.asarray(0.96, dtype=jnp.float64),
        length_target=jnp.asarray(1.8, dtype=jnp.float64),
        volume_target=jnp.asarray(0.78, dtype=jnp.float64),
        non_qs_weight=jnp.asarray(1.3, dtype=jnp.float64),
        residual_weight=jnp.asarray(0.7, dtype=jnp.float64),
        iota_weight=jnp.asarray(1.1, dtype=jnp.float64),
        major_radius_weight=jnp.asarray(0.9, dtype=jnp.float64),
        length_weight=jnp.asarray(1.2, dtype=jnp.float64),
        non_qs_axis=1,
        weight_inv_modB=False,
        length_coil_indices=(0,),
    )
    residual_size = (
        3 * exact_surface.quadpoints_phi.size * (exact_surface.quadpoints_theta.size)
    )
    mask = jnp.asarray((1, 7, residual_size - 2), dtype=jnp.int32)
    problem = FullSpaceProblem(
        coil_dof_extraction=extraction,
        exact_surface_template=exact_surface,
        label_surface_template=label_surface,
        non_qs_surface_template=non_qs_surface,
        exact_mask_indices=mask,
        config=config,
        layout=layout,
    )
    state = FullSpaceState(
        coil_dofs=jnp.asarray(coil_dofs),
        surface_dofs=jnp.asarray(surface_dofs),
        iota=jnp.asarray(0.31, dtype=jnp.float64),
        G=jnp.asarray(0.85, dtype=jnp.float64),
    )
    return problem, state


def _exact_residual(problem: FullSpaceProblem, state: FullSpaceState) -> jax.Array:
    coil_set = coil_set_spec_from_dof_extraction_spec(
        problem.coil_dof_extraction,
        state.coil_dofs,
    )
    gamma, xphi, xtheta = surface_gamma_tangents_from_dofs(
        problem.exact_surface_template,
        state.surface_dofs,
    )
    magnetic_field = grouped_biot_savart_B_from_spec(
        gamma.reshape((-1, 3)), coil_set
    ).reshape(gamma.shape)
    return boozer_residual_vector(
        state.G,
        state.iota,
        magnetic_field,
        xphi,
        xtheta,
        weight_inv_modB=problem.config.weight_inv_modB,
    )


def _non_qs_ratio(problem: FullSpaceProblem, state: FullSpaceState) -> jax.Array:
    coil_set = coil_set_spec_from_dof_extraction_spec(
        problem.coil_dof_extraction,
        state.coil_dofs,
    )
    gamma, xphi, xtheta = surface_gamma_tangents_from_dofs(
        problem.non_qs_surface_template,
        state.surface_dofs,
    )
    magnetic_field = grouped_biot_savart_B_from_spec(
        gamma.reshape((-1, 3)), coil_set
    ).reshape(gamma.shape)
    differential_area = jnp.linalg.norm(jnp.cross(xphi, xtheta), axis=-1)
    mod_B = jnp.linalg.norm(magnetic_field, axis=-1)
    quasi_symmetric_B = jnp.sum(mod_B * differential_area, axis=1) / jnp.sum(
        differential_area, axis=1
    )
    non_quasi_symmetric_B = mod_B - quasi_symmetric_B[:, None]
    return jnp.sum(differential_area * non_quasi_symmetric_B**2) / jnp.sum(
        differential_area * quasi_symmetric_B[:, None] ** 2
    )


def test_raw_terms_follow_frozen_scalar_formulas() -> None:
    problem, state = _problem_and_state()
    z = problem.layout.pack(state)
    evaluation = evaluate_fullspace(z, problem)

    residual = _exact_residual(problem, state)
    expected_iota = 0.5 * (state.iota - problem.config.iota_target) ** 2
    expected_major_radius = (
        0.5
        * (evaluation.observables.major_radius - problem.config.major_radius_target)
        ** 2
    )
    expected_length = (
        0.5
        * jnp.maximum(
            evaluation.observables.total_length - problem.config.length_target,
            0.0,
        )
        ** 2
    )
    expected_weighted = (
        problem.config.non_qs_weight * evaluation.raw_terms.non_qs
        + problem.config.residual_weight * evaluation.raw_terms.residual
        + problem.config.iota_weight * expected_iota
        + problem.config.major_radius_weight * expected_major_radius
        + problem.config.length_weight * expected_length
    )

    np.testing.assert_allclose(
        evaluation.raw_terms.residual,
        0.5 * jnp.mean(residual**2),
        rtol=1.0e-13,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(evaluation.raw_terms.iota, expected_iota)
    np.testing.assert_allclose(evaluation.raw_terms.major_radius, expected_major_radius)
    np.testing.assert_allclose(evaluation.raw_terms.length, expected_length)
    np.testing.assert_allclose(
        evaluation.raw_terms.non_qs,
        _non_qs_ratio(problem, state),
        rtol=1.0e-13,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(evaluation.weighted_total, expected_weighted)


def test_equalities_preserve_signed_volume_and_exact_mask_order() -> None:
    problem, state = _problem_and_state()
    z = problem.layout.pack(state)
    constraints = fullspace_constraints(z, problem)
    residual = _exact_residual(problem, state)
    gamma, xphi, xtheta = surface_gamma_tangents_from_dofs(
        problem.label_surface_template,
        state.surface_dofs,
    )
    signed_volume = surface_volume(gamma, jnp.cross(xphi, xtheta))

    assert float(signed_volume) < 0.0
    np.testing.assert_allclose(
        constraints.boozer,
        residual[problem.exact_mask_indices],
        rtol=1.0e-13,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        constraints.volume,
        signed_volume - problem.config.volume_target,
        rtol=1.0e-13,
        atol=1.0e-15,
    )
    flattened = flatten_fullspace_constraints(constraints)
    np.testing.assert_array_equal(flattened[:-1], constraints.boozer)
    np.testing.assert_array_equal(flattened[-1], constraints.volume)


def test_length_penalty_is_one_sided_and_uses_selected_coil_length() -> None:
    problem, state = _problem_and_state()
    z = problem.layout.pack(state)
    coil_spec = coil_specs_from_dof_extraction_spec(
        problem.coil_dof_extraction,
        state.coil_dofs,
    )[0]
    expected_length = curve_length_from_spec(coil_spec.curve)
    below = replace(
        problem,
        config=replace(problem.config, length_target=expected_length - 0.25),
    )
    above = replace(
        problem,
        config=replace(problem.config, length_target=expected_length + 0.25),
    )

    np.testing.assert_allclose(
        evaluate_fullspace(z, below).raw_terms.length,
        0.5 * 0.25**2,
        rtol=1.0e-13,
        atol=1.0e-15,
    )
    np.testing.assert_array_equal(
        evaluate_fullspace(z, above).raw_terms.length,
        jnp.asarray(0.0, dtype=jnp.float64),
    )


def test_G_is_a_live_joint_coordinate_in_objective_and_equalities() -> None:
    problem, state = _problem_and_state()
    z = problem.layout.pack(state)
    changed = z.at[-1].add(0.07)
    baseline = evaluate_fullspace(z, problem)
    perturbed = evaluate_fullspace(changed, problem)

    assert not np.array_equal(
        np.asarray(baseline.constraints.boozer),
        np.asarray(perturbed.constraints.boozer),
    )
    assert not np.isclose(
        np.asarray(baseline.raw_terms.residual),
        np.asarray(perturbed.raw_terms.residual),
    )
    np.testing.assert_allclose(perturbed.observables.G, state.G + 0.07)


def test_pack_and_complete_evaluation_run_under_one_jit() -> None:
    problem, state = _problem_and_state()

    @jax.jit
    def packed_evaluation(candidate: FullSpaceState, specification: FullSpaceProblem):
        z = specification.layout.pack(candidate)
        evaluation = evaluate_fullspace(z, specification)
        return (
            z,
            evaluation.weighted_total,
            flatten_fullspace_constraints(evaluation.constraints),
        )

    z, value, constraints = packed_evaluation(state, problem)
    expected_z = problem.layout.pack(state)

    np.testing.assert_array_equal(z, expected_z)
    np.testing.assert_allclose(value, fullspace_value(expected_z, problem))
    assert constraints.shape == (problem.exact_mask_indices.size + 1,)
    assert np.all(np.isfinite(np.asarray(constraints)))


def test_value_gradient_matches_independent_central_directional_difference() -> None:
    problem, state = _problem_and_state()
    z = problem.layout.pack(state)
    direction = jnp.linspace(-0.3, 0.4, z.size, dtype=jnp.float64)
    direction = direction / jnp.linalg.norm(direction)
    value, gradient = jax.jit(fullspace_value_and_grad)(z, problem)
    step = 2.0e-5
    finite_difference = (
        fullspace_value(z + step * direction, problem)
        - fullspace_value(z - step * direction, problem)
    ) / (2.0 * step)

    assert np.isfinite(np.asarray(value))
    np.testing.assert_allclose(
        jnp.vdot(gradient, direction),
        finite_difference,
        rtol=2.0e-5,
        atol=2.0e-7,
    )


def test_surface_major_radius_matches_existing_adapter_oracle_formula() -> None:
    from simsopt_jax_adapters.geo.surface_objectives import (
        _surface_major_radius_from_geometry,
    )

    surface_dofs = _toroidal_surface_dofs(major_radius=1.2, minor_radius=0.24)
    surface = _surface_spec(surface_dofs, nphi=8, ntheta=9)
    gamma, xphi, xtheta = surface_gamma_tangents_from_dofs(surface, surface_dofs)

    promoted = surface_major_radius(gamma, xphi, xtheta)
    adapter_oracle = _surface_major_radius_from_geometry(gamma, xphi, xtheta)

    np.testing.assert_allclose(promoted, adapter_oracle, rtol=1.0e-13, atol=1.0e-15)
