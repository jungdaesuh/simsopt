from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from simsopt_jax.backend import invalidate_backend_cache
from simsopt_jax.core import make_coil_symmetry_spec, make_curve_xyzfourier_spec
from simsopt_jax.core.specs import (
    gson_decode_spec_value,
    make_fixed_surface_flux_spec,
    make_grouped_coil_set_spec,
)
from simsopt_jax.core.banana import (
    banana_current_magnitude_penalty,
    banana_coil_coil_distance_pure,
    banana_coil_surface_distance_pure,
    banana_decision_from_dofs,
    banana_geometry_from_dofs,
    banana_hardware_keepout_point_cloud_curve_pure,
    banana_hardware_keepout_point_cloud_pure,
    banana_local_terms,
    banana_local_value,
    banana_local_value_and_grad,
    banana_pack_rotation_fold_pure,
    banana_pack_twist_strain_pure,
    banana_projected_reach_pure,
    banana_quadratic_penalty,
    banana_rotation_aware_curvature_excess_pure,
    banana_self_distance_mask,
    banana_self_distance_pure,
    banana_stage2_value,
    banana_symmetry_geometry_from_arrays,
    banana_swept_channel_surface_points,
    banana_torsional_strain_pure,
    make_banana_objective_spec,
    make_banana_system_spec,
)


jax.config.update("jax_enable_x64", True)


def _set_pairwise_penalty_chunk_size(monkeypatch, chunk_size: int) -> None:
    monkeypatch.setenv("SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE", str(chunk_size))
    invalidate_backend_cache()


def _make_curve_spec():
    return make_curve_xyzfourier_spec(
        dofs=np.array(
            [0.0, 0.7, 0.1, 0.0, -0.2, 0.9, 0.1, 0.2, -0.1],
            dtype=np.float64,
        ),
        quadpoints=np.linspace(0.0, 1.0, 8, endpoint=False, dtype=np.float64),
        order=1,
    )


def _make_system_spec(
    *,
    current_dof_count=1,
    fixed_current_dofs=(0.0,),
    tf_current_dof_count=0,
    fixed_tf_current_dofs=(),
    tf_current_scale=0.0,
):
    return make_banana_system_spec(
        curve=_make_curve_spec(),
        symmetries=(make_coil_symmetry_spec(scale=2.0),),
        fixed_current_dofs=np.asarray(fixed_current_dofs, dtype=np.float64),
        current_dof_count=current_dof_count,
        fixed_tf_current_dofs=np.asarray(fixed_tf_current_dofs, dtype=np.float64),
        tf_current_dof_count=tf_current_dof_count,
        tf_current_scale=tf_current_scale,
    )


def test_banana_system_spec_rejects_unsupported_multi_current_layout():
    with pytest.raises(ValueError, match="0 or 1 active current DOF"):
        _make_system_spec(current_dof_count=2)
    with pytest.raises(ValueError, match="0 or 1 active TF current DOF"):
        _make_system_spec(tf_current_dof_count=2)


def test_banana_system_spec_carries_auxiliary_coils_and_boozer_config():
    auxiliary_coils = make_grouped_coil_set_spec(
        (
            (
                np.asarray(
                    [
                        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
                        [[0.0, 0.2, 0.0], [0.5, 0.2, 0.0]],
                    ],
                    dtype=np.float64,
                ),
                np.asarray(
                    [
                        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    ],
                    dtype=np.float64,
                ),
                np.asarray([100.0, -20.0], dtype=np.float64),
                np.asarray([0, 1], dtype=np.int64),
            ),
        )
    )

    system = make_banana_system_spec(
        curve=_make_curve_spec(),
        symmetries=(make_coil_symmetry_spec(scale=1.0),),
        fixed_auxiliary_coil_set=auxiliary_coils,
        fixed_auxiliary_coil_roles=("proxy", "vf"),
        boozer_iota=np.asarray([0.31], dtype=np.float64),
        boozer_G=np.asarray([1.7], dtype=np.float64),
        boozer_enabled=True,
        boozer_solver="ls",
        boozer_mpol=3,
        boozer_ntor=2,
        boozer_nfp=4,
        boozer_nphi=10,
        boozer_ntheta=12,
    )

    assert len(system.fixed_auxiliary_coil_set.groups) == 1
    np.testing.assert_allclose(
        system.fixed_auxiliary_coil_set.groups[0].currents,
        np.asarray([100.0, -20.0], dtype=np.float64),
    )
    assert system.fixed_auxiliary_coil_roles == ("proxy", "vf")
    assert system.boozer.enabled is True
    assert system.boozer.solver == "ls"
    assert system.boozer.mpol == 3
    assert system.boozer.ntor == 2
    assert system.boozer.nfp == 4
    assert system.boozer.nphi == 10
    assert system.boozer.ntheta == 12
    np.testing.assert_allclose(system.boozer.iota, np.asarray([0.31]))
    np.testing.assert_allclose(system.boozer.G, np.asarray([1.7]))


def _make_objective_spec(system):
    return make_banana_objective_spec(
        system=system,
        stage="stage2",
        length_max_threshold=2.0,
        length_min_threshold=0.5,
        curvature_threshold=100.0,
        curvature_p=4,
        poloidal_major_radius=1.0,
        poloidal_theta_target=np.deg2rad(70.0),
        width_major_radius=1.0,
        width_minor_radius=0.25,
        width_max_threshold=0.17,
        width_min_threshold=0.10,
        self_distance_minimum=0.01,
        coil_coil_distance_minimum=0.10,
        banana_current_max_threshold=2.0,
        length_weight=0.0,
        curvature_weight=0.0,
        poloidal_weight=0.0,
        width_weight=0.0,
        self_distance_weight=0.0,
        coil_coil_distance_weight=0.0,
        banana_current_weight=1.0,
        self_distance_neighbor_skip=1,
    )


def test_banana_objective_spec_carries_surface_and_flux_quadrature_counts():
    system = _make_system_spec()
    surface_gamma = np.arange(18, dtype=np.float64).reshape((2, 3, 3))
    surface_normal = np.ones((2, 3, 3), dtype=np.float64)
    flux = make_fixed_surface_flux_spec(
        points=surface_gamma.reshape((-1, 3)),
        normal=surface_normal,
        target=np.zeros((2, 3, 3), dtype=np.float64),
        definition="normalized",
    )

    spec = make_banana_objective_spec(
        system=system,
        stage="stage2",
        length_max_threshold=2.0,
        length_min_threshold=0.5,
        curvature_threshold=100.0,
        curvature_p=4,
        poloidal_major_radius=1.0,
        poloidal_theta_target=np.deg2rad(70.0),
        width_major_radius=1.0,
        width_minor_radius=0.25,
        width_max_threshold=0.17,
        width_min_threshold=0.10,
        self_distance_minimum=0.01,
        coil_coil_distance_minimum=0.10,
        banana_current_max_threshold=2.0,
        length_weight=0.0,
        curvature_weight=0.0,
        poloidal_weight=0.0,
        width_weight=0.0,
        self_distance_weight=0.0,
        coil_coil_distance_weight=0.0,
        banana_current_weight=1.0,
        surface_gamma=surface_gamma,
        surface_normal=surface_normal,
        surface_nphi=2,
        surface_ntheta=3,
        flux=flux,
    )

    assert spec.quadrature.curve_point_count == 8
    assert spec.quadrature.surface_sample_count == 6
    assert spec.quadrature.surface_nphi == 2
    assert spec.quadrature.surface_ntheta == 3
    assert spec.quadrature.flux_point_count == 6
    assert spec.quadrature.flux_nphi == 2
    assert spec.quadrature.flux_ntheta == 3


def test_banana_decision_split_uses_current_first_stage2_layout():
    system = _make_system_spec()
    decision = banana_decision_from_dofs(
        system.decision,
        jnp.arange(1 + system.decision.curve_dof_count, dtype=jnp.float64),
    )

    assert decision.tf_current_dofs.shape == (0,)
    np.testing.assert_allclose(decision.current_dofs, np.array([0.0]))
    np.testing.assert_allclose(decision.curve_dofs, np.arange(1.0, 10.0))
    np.testing.assert_allclose(decision.flat_dofs, np.arange(10.0))


def test_banana_decision_split_uses_tf_banana_curve_layout():
    system = _make_system_spec(tf_current_dof_count=1)
    decision = banana_decision_from_dofs(
        system.decision,
        jnp.arange(2 + system.decision.curve_dof_count, dtype=jnp.float64),
    )

    np.testing.assert_allclose(decision.tf_current_dofs, np.array([0.0]))
    np.testing.assert_allclose(decision.current_dofs, np.array([1.0]))
    np.testing.assert_allclose(decision.curve_dofs, np.arange(2.0, 11.0))
    np.testing.assert_allclose(decision.flat_dofs, np.arange(11.0))


def test_banana_decision_rejects_mismatched_vector_length():
    system = _make_system_spec()
    expected_count = 1 + system.decision.curve_dof_count

    with pytest.raises(ValueError, match="decision vector length"):
        banana_decision_from_dofs(
            system.decision,
            jnp.arange(expected_count + 1, dtype=jnp.float64),
        )
    with pytest.raises(ValueError, match="decision vector length"):
        banana_decision_from_dofs(
            system.decision,
            jnp.arange(expected_count - 1, dtype=jnp.float64),
        )


def test_banana_geometry_applies_symmetry_current_scale():
    system = _make_system_spec(tf_current_dof_count=1, tf_current_scale=-4.0)
    dofs = jnp.concatenate(
        (
            jnp.asarray([5.0], dtype=jnp.float64),
            jnp.asarray([3.0], dtype=jnp.float64),
            jnp.asarray(system.curve.dofs, dtype=jnp.float64),
        )
    )

    geometry = banana_geometry_from_dofs(system, dofs)

    assert geometry.gammas.shape == (1, 8, 3)
    assert geometry.gammadashs.shape == (1, 8, 3)
    np.testing.assert_allclose(geometry.tf_current_dof, np.array(5.0))
    np.testing.assert_allclose(geometry.tf_current, np.array(-20.0))
    np.testing.assert_allclose(geometry.base_current_dof, np.array(3.0))
    np.testing.assert_allclose(geometry.currents, np.array([6.0]))


def test_array_symmetry_helper_matches_dense_rotation_and_current_scale():
    base_gamma = jnp.asarray(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        dtype=jnp.float64,
    )
    base_gammadash = jnp.asarray(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype=jnp.float64,
    )
    rotmats = jnp.asarray(
        [
            np.eye(3, dtype=np.float64),
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=jnp.float64,
    )
    current_scales = jnp.asarray([2.0, -0.5], dtype=jnp.float64)

    gammas, gammadashs, currents = banana_symmetry_geometry_from_arrays(
        base_gamma,
        base_gammadash,
        rotmats,
        current_scales,
        jnp.asarray(4.0, dtype=jnp.float64),
    )

    np.testing.assert_allclose(gammas[0], base_gamma)
    np.testing.assert_allclose(gammas[1], base_gamma @ rotmats[1])
    np.testing.assert_allclose(gammadashs[1], base_gammadash @ rotmats[1])
    np.testing.assert_allclose(currents, np.array([8.0, -2.0]))


def test_quadratic_penalty_modes_match_simsopt_semantics():
    value = jnp.asarray(3.0, dtype=jnp.float64)

    np.testing.assert_allclose(
        banana_quadratic_penalty(value, 2.0, "max"),
        np.array(0.5),
    )
    np.testing.assert_allclose(
        banana_quadratic_penalty(value, 4.0, "max"),
        np.array(0.0),
    )
    np.testing.assert_allclose(
        banana_quadratic_penalty(value, 4.0, "min"),
        np.array(0.5),
    )
    np.testing.assert_allclose(
        banana_quadratic_penalty(value, 2.0, "identity"),
        np.array(0.5),
    )
    np.testing.assert_allclose(
        banana_quadratic_penalty(value, 2.0, "two-sided"),
        np.array(0.5),
    )


def test_quadratic_penalty_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unsupported banana penalty mode"):
        banana_quadratic_penalty(
            jnp.asarray(3.0, dtype=jnp.float64),
            2.0,
            "maximum",
        )


def test_current_magnitude_penalty_has_sign_aware_gradient():
    value, grad = jax.value_and_grad(
        lambda current: banana_current_magnitude_penalty(current, 2.0)
    )(jnp.asarray(-3.0, dtype=jnp.float64))

    np.testing.assert_allclose(value, np.array(0.5))
    np.testing.assert_allclose(grad, np.array(-1.0))


def test_pack_rotation_fold_matches_simsopt_lp_abs_hinge_scale():
    frame_binormal_curvature = jnp.asarray([1.0, 3.0, -4.0], dtype=jnp.float64)
    gammadash = jnp.asarray(
        [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=jnp.float64,
    )

    value = banana_pack_rotation_fold_pure(
        frame_binormal_curvature,
        gammadash,
        p=2,
        threshold=2.0,
    )

    np.testing.assert_allclose(value, np.array(13.0 / 6.0))


def test_pack_twist_strain_matches_torsional_strain_lp_scale():
    frame_torsion = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)
    gammadash = jnp.asarray(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=jnp.float64,
    )

    strain = banana_torsional_strain_pure(frame_torsion, width=2.0)
    value, grad = jax.value_and_grad(
        lambda torsion: banana_pack_twist_strain_pure(
            torsion,
            gammadash,
            width=2.0,
            p=2,
            threshold=1.0,
        )
    )(frame_torsion)

    np.testing.assert_allclose(strain, np.array([1.0 / 3.0, 4.0 / 3.0, 3.0]))
    np.testing.assert_allclose(value, np.array(55.0 / 27.0))
    assert grad.shape == frame_torsion.shape
    assert np.all(np.isfinite(np.asarray(grad)))


def test_rotation_aware_pack_reach_projects_rotated_channel_axes():
    gammadash = jnp.asarray(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=jnp.float64,
    )
    gammadashdash = jnp.asarray(
        [[0.0, 4.0, 0.0], [0.0, 0.0, -5.0]],
        dtype=jnp.float64,
    )
    frame_normal = jnp.asarray(
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=jnp.float64,
    )
    frame_binormal = jnp.asarray(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=jnp.float64,
    )

    reach = banana_projected_reach_pure(
        gammadash,
        gammadashdash,
        frame_normal,
        frame_binormal,
        half_normal_width=0.2,
        half_binormal_width=0.3,
    )

    np.testing.assert_allclose(reach, np.array([0.2, 0.3]))


def test_rotation_aware_curvature_excess_matches_pack_formula():
    kappa = jnp.asarray([4.0, 3.0], dtype=jnp.float64)
    gammadash = jnp.asarray(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=jnp.float64,
    )
    gammadashdash = jnp.asarray(
        [[0.0, 4.0, 0.0], [0.0, 0.0, -5.0]],
        dtype=jnp.float64,
    )
    frame_normal = jnp.asarray(
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=jnp.float64,
    )
    frame_binormal = jnp.asarray(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=jnp.float64,
    )

    value = jax.jit(banana_rotation_aware_curvature_excess_pure)(
        kappa,
        gammadash,
        gammadashdash,
        frame_normal,
        frame_binormal,
        2,
        0.1,
        0.2,
        0.3,
    )

    expected = ((2.0 / 3.0) ** 2 + 2.0 * 0.5**2) / 3.0
    np.testing.assert_allclose(value, np.array(expected))


def _make_keepout_curve():
    return jnp.asarray(
        [
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, -2.0, 0.0],
        ],
        dtype=jnp.float64,
    )


def test_swept_channel_surface_points_match_viewer_frame_coefficients():
    gamma = _make_keepout_curve()

    samples = banana_swept_channel_surface_points(
        gamma,
        half_binormal_width=0.1,
        half_radial_depth=0.2,
        winding_major_radius=1.0,
    )

    assert samples.shape == (104, 3)
    np.testing.assert_allclose(samples[0], np.array([1.8, -1.0, 0.1]))


def test_hardware_keepout_point_cloud_curve_matches_margin_normalized_box_distance():
    gamma = _make_keepout_curve()
    hardware_points = jnp.asarray(
        [
            [2.0, 0.0, 0.15],
            [5.0, 5.0, 5.0],
        ],
        dtype=jnp.float64,
    )

    value, grad = jax.value_and_grad(
        lambda curve_points: banana_hardware_keepout_point_cloud_curve_pure(
            curve_points,
            hardware_points,
            point_weight=0.04,
            half_binormal_width=0.1,
            half_radial_depth=0.1,
            margin=0.2,
            winding_major_radius=1.0,
        )
    )(gamma)

    np.testing.assert_allclose(value, np.array(0.5625))
    assert grad.shape == gamma.shape
    assert np.all(np.isfinite(np.asarray(grad)))


def test_hardware_keepout_point_cloud_dense_and_chunked_paths_match(monkeypatch):
    gamma = _make_keepout_curve()
    gammas = jnp.stack(
        (
            gamma,
            gamma + jnp.asarray([0.0, 0.0, 10.0], dtype=jnp.float64),
        ),
        axis=0,
    )
    hardware_points = jnp.asarray(
        [
            [2.0, 0.0, 0.15],
            [5.0, 5.0, 5.0],
        ],
        dtype=jnp.float64,
    )

    try:
        _set_pairwise_penalty_chunk_size(monkeypatch, 0)
        dense_value = banana_hardware_keepout_point_cloud_pure(
            gammas,
            hardware_points,
            point_weight=0.04,
            half_binormal_width=0.1,
            half_radial_depth=0.1,
            margin=0.2,
            winding_major_radius=1.0,
        )

        _set_pairwise_penalty_chunk_size(monkeypatch, 1)
        chunked_value = banana_hardware_keepout_point_cloud_pure(
            gammas,
            hardware_points,
            point_weight=0.04,
            half_binormal_width=0.1,
            half_radial_depth=0.1,
            margin=0.2,
            winding_major_radius=1.0,
        )
    finally:
        monkeypatch.delenv("SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE", raising=False)
        invalidate_backend_cache()

    np.testing.assert_allclose(dense_value, chunked_value, rtol=1e-12, atol=1e-12)


def test_local_terms_use_spec_threshold_modes():
    system = _make_system_spec()
    spec = replace(
        _make_objective_spec(system),
        banana_current_max_threshold=10.0,
        banana_current_max_mode="identity",
    )
    dofs = jnp.concatenate(
        (
            jnp.asarray([1.0], dtype=jnp.float64),
            jnp.asarray(system.curve.dofs, dtype=jnp.float64),
        )
    )

    terms = banana_local_terms(spec, dofs)

    np.testing.assert_allclose(terms.banana_current_max, np.array(32.0))


def test_self_distance_mask_and_penalty_match_dense_pair_formula():
    gamma = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=jnp.float64,
    )
    gammadash = jnp.asarray(
        np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), (4, 1))
    )
    mask = banana_self_distance_mask(4, 0)

    penalty = banana_self_distance_pure(gamma, gammadash, 0.1, mask, False)

    np.testing.assert_allclose(np.diag(np.asarray(mask)), np.zeros(4))
    np.testing.assert_allclose(penalty, np.array(0.0025))


def test_self_distance_dense_and_chunked_paths_match(monkeypatch):
    gamma = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.20, 0.0, 0.0],
            [1.00, 0.0, 0.0],
            [0.00, 1.0, 0.0],
        ],
        dtype=jnp.float64,
    )
    gammadash = jnp.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.1, 0.1, 0.0],
            [0.9, 0.0, 0.1],
            [1.0, 0.2, 0.0],
            [1.0, 0.0, 0.2],
        ],
        dtype=jnp.float64,
    )
    mask = banana_self_distance_mask(5, 0)

    try:
        _set_pairwise_penalty_chunk_size(monkeypatch, 0)
        dense_value, dense_grad = jax.value_and_grad(
            lambda curve_points: banana_self_distance_pure(
                curve_points,
                gammadash,
                0.25,
                mask,
                False,
            )
        )(gamma)

        _set_pairwise_penalty_chunk_size(monkeypatch, 2)
        chunked_value, chunked_grad = jax.value_and_grad(
            lambda curve_points: banana_self_distance_pure(
                curve_points,
                gammadash,
                0.25,
                mask,
                False,
            )
        )(gamma)
    finally:
        monkeypatch.delenv("SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE", raising=False)
        invalidate_backend_cache()

    np.testing.assert_allclose(dense_value, chunked_value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dense_grad, chunked_grad, rtol=1e-12, atol=1e-12)


def test_coil_coil_distance_sums_unique_symmetry_pairs():
    gammas = jnp.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.05, 0.0, 0.0], [1.05, 0.0, 0.0]],
            [[0.20, 0.0, 0.0], [1.20, 0.0, 0.0]],
        ],
        dtype=jnp.float64,
    )
    gammadashs = jnp.asarray(
        np.tile(np.array([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]), (3, 1, 1)),
        dtype=jnp.float64,
    )

    penalty = banana_coil_coil_distance_pure(gammas, gammadashs, 0.10)

    np.testing.assert_allclose(penalty, np.array(0.00125))


def test_coil_coil_distance_dense_and_chunked_paths_match(monkeypatch):
    gammas = jnp.asarray(
        [
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.05, 0.0, 0.0], [0.55, 0.0, 0.1], [1.05, 0.0, 0.0]],
            [[0.20, 0.1, 0.0], [0.70, 0.1, 0.0], [1.20, 0.1, 0.0]],
        ],
        dtype=jnp.float64,
    )
    gammadashs = jnp.asarray(
        [
            [[1.0, 0.0, 0.0], [1.0, 0.1, 0.0], [1.0, 0.0, 0.1]],
            [[1.1, 0.0, 0.0], [1.0, 0.0, 0.2], [0.9, 0.0, 0.0]],
            [[0.8, 0.1, 0.0], [1.0, 0.0, 0.1], [1.2, 0.0, 0.0]],
        ],
        dtype=jnp.float64,
    )

    try:
        _set_pairwise_penalty_chunk_size(monkeypatch, 0)
        dense_value, dense_grad = jax.value_and_grad(
            lambda coil_points: banana_coil_coil_distance_pure(
                coil_points,
                gammadashs,
                0.15,
            )
        )(gammas)

        _set_pairwise_penalty_chunk_size(monkeypatch, 2)
        chunked_value, chunked_grad = jax.value_and_grad(
            lambda coil_points: banana_coil_coil_distance_pure(
                coil_points,
                gammadashs,
                0.15,
            )
        )(gammas)
    finally:
        monkeypatch.delenv("SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE", raising=False)
        invalidate_backend_cache()

    np.testing.assert_allclose(dense_value, chunked_value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dense_grad, chunked_grad, rtol=1e-12, atol=1e-12)


def test_coil_surface_distance_sums_all_banana_coils():
    gammas = jnp.asarray(
        [
            [[0.0, 0.0, 0.05], [1.0, 0.0, 0.05]],
            [[0.0, 0.0, 0.05], [1.0, 0.0, 0.05]],
        ],
        dtype=jnp.float64,
    )
    gammadashs = jnp.asarray(
        np.tile(np.array([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]), (2, 1, 1)),
        dtype=jnp.float64,
    )
    surface_gamma = jnp.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=jnp.float64,
    )
    surface_normal = jnp.asarray(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=jnp.float64,
    )

    penalty = banana_coil_surface_distance_pure(
        gammas,
        gammadashs,
        surface_gamma,
        surface_normal,
        0.10,
    )

    np.testing.assert_allclose(penalty, np.array(0.0025))


def test_coil_surface_distance_dense_and_chunked_paths_match(monkeypatch):
    gammas = jnp.asarray(
        [
            [[0.0, 0.0, 0.05], [0.5, 0.0, 0.07], [1.0, 0.0, 0.05]],
            [[0.0, 0.1, 0.06], [0.5, 0.1, 0.08], [1.0, 0.1, 0.06]],
        ],
        dtype=jnp.float64,
    )
    gammadashs = jnp.asarray(
        [
            [[1.0, 0.0, 0.0], [1.0, 0.2, 0.0], [1.0, 0.0, 0.1]],
            [[0.9, 0.0, 0.0], [1.0, 0.1, 0.0], [1.1, 0.0, 0.2]],
        ],
        dtype=jnp.float64,
    )
    surface_gamma = jnp.asarray(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=jnp.float64,
    )
    surface_normal = jnp.asarray(
        [[0.0, 0.0, 1.0], [0.0, 0.1, 1.0], [0.1, 0.0, 1.0]],
        dtype=jnp.float64,
    )

    try:
        _set_pairwise_penalty_chunk_size(monkeypatch, 0)
        dense_value, dense_grad = jax.value_and_grad(
            lambda coil_points: banana_coil_surface_distance_pure(
                coil_points,
                gammadashs,
                surface_gamma,
                surface_normal,
                0.1,
            )
        )(gammas)

        _set_pairwise_penalty_chunk_size(monkeypatch, 2)
        chunked_value, chunked_grad = jax.value_and_grad(
            lambda coil_points: banana_coil_surface_distance_pure(
                coil_points,
                gammadashs,
                surface_gamma,
                surface_normal,
                0.1,
            )
        )(gammas)
    finally:
        monkeypatch.delenv("SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE", raising=False)
        invalidate_backend_cache()

    np.testing.assert_allclose(dense_value, chunked_value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dense_grad, chunked_grad, rtol=1e-12, atol=1e-12)


def test_local_terms_include_weighted_coil_coil_distance():
    system = make_banana_system_spec(
        curve=_make_curve_spec(),
        symmetries=(
            make_coil_symmetry_spec(scale=1.0),
            make_coil_symmetry_spec(scale=1.0),
        ),
        current_dof_count=1,
    )
    spec = replace(
        _make_objective_spec(system),
        coil_coil_distance_minimum=1.0,
        coil_coil_distance_weight=3.0,
        banana_current_weight=0.0,
    )
    dofs = jnp.concatenate(
        (
            jnp.asarray([1.0], dtype=jnp.float64),
            jnp.asarray(system.curve.dofs, dtype=jnp.float64),
        )
    )

    terms = banana_local_terms(spec, dofs)
    value = banana_local_value(spec, dofs)

    assert terms.coil_coil_distance > 0.0
    np.testing.assert_allclose(value, 3.0 * terms.coil_coil_distance)


def test_local_terms_include_weighted_coil_surface_distance():
    system = make_banana_system_spec(
        curve=_make_curve_spec(),
        symmetries=(make_coil_symmetry_spec(scale=1.0),),
        current_dof_count=1,
    )
    spec = replace(
        _make_objective_spec(system),
        surface_gamma=jnp.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=jnp.float64,
        ),
        surface_normal=jnp.asarray(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            dtype=jnp.float64,
        ),
        coil_surface_distance_minimum=1.0,
        include_coil_surface_distance=True,
        coil_surface_distance_weight=5.0,
        banana_current_weight=0.0,
    )
    dofs = jnp.concatenate(
        (
            jnp.asarray([1.0], dtype=jnp.float64),
            jnp.asarray(system.curve.dofs, dtype=jnp.float64),
        )
    )

    terms = banana_local_terms(spec, dofs)
    value = banana_local_value(spec, dofs)

    assert terms.coil_surface_distance > 0.0
    np.testing.assert_allclose(value, 5.0 * terms.coil_surface_distance)


def test_local_terms_include_weighted_tf_current_penalty():
    system = _make_system_spec(
        current_dof_count=0,
        fixed_current_dofs=(0.0,),
        tf_current_dof_count=1,
        tf_current_scale=-4.0,
    )
    spec = replace(
        _make_objective_spec(system),
        include_current_penalty=False,
        include_tf_current_penalty=True,
        tf_current_max_threshold=10.0,
        tf_current_weight=7.0,
        banana_current_weight=0.0,
    )
    dofs = jnp.concatenate(
        (
            jnp.asarray([3.0], dtype=jnp.float64),
            jnp.asarray(system.curve.dofs, dtype=jnp.float64),
        )
    )

    terms = banana_local_terms(spec, dofs)
    value = banana_local_value(spec, dofs)

    np.testing.assert_allclose(terms.tf_current_max, np.array(2.0))
    np.testing.assert_allclose(value, 14.0)


def test_objective_spec_requires_surface_samples_when_coil_surface_term_is_enabled():
    system = _make_system_spec()

    with pytest.raises(ValueError, match="require frozen surface samples"):
        make_banana_objective_spec(
            system=system,
            stage="stage2",
            length_max_threshold=2.0,
            length_min_threshold=0.5,
            curvature_threshold=100.0,
            curvature_p=4,
            poloidal_major_radius=1.0,
            poloidal_theta_target=np.deg2rad(70.0),
            width_major_radius=1.0,
            width_minor_radius=0.25,
            width_max_threshold=0.17,
            width_min_threshold=0.10,
            self_distance_minimum=0.01,
            coil_coil_distance_minimum=0.10,
            banana_current_max_threshold=2.0,
            length_weight=0.0,
            curvature_weight=0.0,
            poloidal_weight=0.0,
            width_weight=0.0,
            self_distance_weight=0.0,
            coil_coil_distance_weight=0.0,
            banana_current_weight=1.0,
            include_coil_surface_distance=True,
        )


def test_local_value_and_grad_exercises_free_current_dof():
    system = _make_system_spec()
    spec = _make_objective_spec(system)
    dofs = jnp.concatenate(
        (
            jnp.asarray([-3.0], dtype=jnp.float64),
            jnp.asarray(system.curve.dofs, dtype=jnp.float64),
        )
    )

    value, grad = banana_local_value_and_grad(spec, dofs)
    terms = banana_local_terms(spec, dofs)

    np.testing.assert_allclose(terms.banana_current_max, np.array(8.0))
    np.testing.assert_allclose(value, np.array(8.0))
    np.testing.assert_allclose(grad[0], np.array(-8.0))
    assert np.all(np.isfinite(np.asarray(grad)))


def test_local_value_and_grad_runs_under_strict_transfer_guard():
    system = _make_system_spec()
    spec = _make_objective_spec(system)
    dofs = jnp.concatenate(
        (
            jnp.asarray([-3.0], dtype=jnp.float64),
            jnp.asarray(system.curve.dofs, dtype=jnp.float64),
        )
    )
    warm_value, warm_grad = banana_local_value_and_grad(spec, dofs)
    warm_value.block_until_ready()
    warm_grad.block_until_ready()

    with jax.transfer_guard("disallow"):
        value, grad = banana_local_value_and_grad(spec, dofs)
        value.block_until_ready()
        grad.block_until_ready()

    np.testing.assert_allclose(value, np.array(8.0))
    assert np.all(np.isfinite(np.asarray(grad)))


def test_stage2_value_jit_and_value_and_grad_static_shape_under_transfer_guard():
    system = _make_system_spec()
    spec = _make_objective_spec(system)
    dofs = jnp.concatenate(
        (
            jnp.asarray([-3.0], dtype=jnp.float64),
            jnp.asarray(system.curve.dofs, dtype=jnp.float64),
        )
    )
    perturbed_dofs = dofs.at[0].set(-2.5)

    value_fn = jax.jit(banana_stage2_value)
    value_and_grad_fn = jax.jit(jax.value_and_grad(banana_stage2_value, argnums=1))

    value = value_fn(spec, dofs)
    value.block_until_ready()
    assert value_fn._cache_size() == 1

    perturbed_value = value_fn(spec, perturbed_dofs)
    perturbed_value.block_until_ready()
    assert value_fn._cache_size() == 1

    stage2_value, stage2_grad = value_and_grad_fn(spec, dofs)
    stage2_value.block_until_ready()
    stage2_grad.block_until_ready()
    assert value_and_grad_fn._cache_size() == 1

    with jax.transfer_guard("disallow"):
        guarded_value = value_fn(spec, dofs)
        guarded_stage2_value, guarded_stage2_grad = value_and_grad_fn(spec, dofs)
        guarded_value.block_until_ready()
        guarded_stage2_value.block_until_ready()
        guarded_stage2_grad.block_until_ready()

    np.testing.assert_allclose(value, np.array(8.0))
    np.testing.assert_allclose(perturbed_value, np.array(4.5))
    np.testing.assert_allclose(stage2_value, value)
    np.testing.assert_allclose(guarded_value, value)
    np.testing.assert_allclose(guarded_stage2_value, value)
    assert stage2_grad.shape == dofs.shape
    assert guarded_stage2_grad.shape == dofs.shape
    assert np.all(np.isfinite(np.asarray(stage2_grad)))
    assert np.all(np.isfinite(np.asarray(guarded_stage2_grad)))


def test_banana_spec_data_fields_do_not_recompile_but_meta_fields_do():
    system = _make_system_spec(current_dof_count=0, fixed_current_dofs=(3.0,))
    spec = replace(
        _make_objective_spec(system),
        surface_gamma=jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float64),
        surface_normal=jnp.asarray([[0.0, 0.0, 1.0]], dtype=jnp.float64),
    )
    dofs = jnp.asarray(system.curve.dofs, dtype=jnp.float64)

    @jax.jit
    def objective_value(objective_spec, decision_vector):
        return banana_local_value(objective_spec, decision_vector)

    assert objective_value._cache_size() == 0

    np.testing.assert_allclose(objective_value(spec, dofs), np.array(8.0))
    assert objective_value._cache_size() == 1

    dynamic_update = replace(
        spec,
        system=replace(
            spec.system,
            fixed_current_dofs=jnp.asarray([4.0], dtype=jnp.float64),
            fixed_tf_current_dofs=jnp.asarray((), dtype=jnp.float64),
        ),
        surface_gamma=jnp.asarray([[1.0, 0.0, 0.0]], dtype=jnp.float64),
        surface_normal=jnp.asarray([[0.0, 0.0, 1.0]], dtype=jnp.float64),
    )
    np.testing.assert_allclose(objective_value(dynamic_update, dofs), np.array(18.0))
    assert objective_value._cache_size() == 1

    meta_update = replace(spec, banana_current_max_threshold=6.0)
    np.testing.assert_allclose(objective_value(meta_update, dofs), np.array(0.0))
    assert objective_value._cache_size() == 2

    _leaves, treedef = jax.tree.flatten(spec)
    _dynamic_leaves, dynamic_treedef = jax.tree.flatten(dynamic_update)
    _meta_leaves, meta_treedef = jax.tree.flatten(meta_update)

    assert dynamic_treedef == treedef
    assert meta_treedef != treedef


def test_banana_specs_round_trip_through_registered_gson_payloads():
    system = _make_system_spec()
    spec = replace(
        _make_objective_spec(system),
        surface_gamma=jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float64),
        surface_normal=jnp.asarray([[0.0, 0.0, 1.0]], dtype=jnp.float64),
    )

    payload = spec.as_dict()
    restored = type(spec).from_dict(payload)
    decoded = gson_decode_spec_value(payload)

    assert isinstance(payload, dict)
    assert payload["@module"] == "simsopt_jax.core.banana"
    assert payload["@class"] == "BananaObjectiveSpec"
    assert payload["system"]["@class"] == "BananaSystemSpec"
    np.testing.assert_allclose(restored.system.curve.dofs, spec.system.curve.dofs)
    np.testing.assert_allclose(restored.self_distance_mask, spec.self_distance_mask)
    np.testing.assert_allclose(restored.surface_gamma, spec.surface_gamma)
    np.testing.assert_allclose(restored.surface_normal, spec.surface_normal)
    assert restored.system.decision.current_dof_count == 1
    assert restored.system.decision.tf_current_dof_count == 0
    assert restored.system.fixed_auxiliary_coil_roles == ()
    assert restored.system.boozer.enabled is False
    assert restored.quadrature.curve_point_count == spec.quadrature.curve_point_count
    assert (
        restored.quadrature.surface_sample_count
        == spec.quadrature.surface_sample_count
    )
    assert isinstance(decoded, type(spec))
    np.testing.assert_allclose(decoded.system.curve.dofs, spec.system.curve.dofs)
    np.testing.assert_allclose(decoded.surface_gamma, spec.surface_gamma)
    np.testing.assert_allclose(decoded.surface_normal, spec.surface_normal)
    assert decoded.system.boozer.enabled is False
    assert decoded.quadrature.curve_point_count == spec.quadrature.curve_point_count
