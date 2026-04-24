import numpy as np
import jax
import jax.numpy as jnp
import pytest

from simsopt.jax_core import (
    apply_coil_symmetry,
    curve_spec_kind,
    make_coil_dof_extraction_spec,
    make_coil_group_spec,
    make_coil_spec,
    make_coil_symmetry_spec,
    make_coil_set_dof_extraction_spec,
    make_current_value_spec,
    make_curve_cwsfourier_rz_spec,
    make_curve_filament_spec,
    make_curve_helical_spec,
    make_curve_perturbed_spec,
    make_curve_planarfourier_spec,
    make_curve_rzfourier_spec,
    make_curve_xyzfourier_spec,
    make_field_eval_spec,
    make_frame_rotation_spec,
    make_fixed_surface_flux_spec,
    make_grouped_coil_set_spec,
    make_optimizable_dof_map_spec,
    make_single_stage_runtime_spec,
    make_single_stage_seed_spec,
    make_surface_rzfourier_spec,
    make_surface_xyz_fourier_spec,
    make_surface_xyz_tensor_fourier_spec,
    surface_spec_kind,
    surface_xyz_fourier_gamma_from_spec,
    surface_xyz_fourier_normal_from_spec,
    surface_xyz_tensor_fourier_gamma_from_spec,
    surface_xyz_tensor_fourier_normal_from_spec,
)


jax.config.update("jax_enable_x64", True)


def _make_curve_spec():
    return make_curve_xyzfourier_spec(
        dofs=np.arange(9, dtype=np.float32),
        quadpoints=np.linspace(0.0, 1.0, 4, endpoint=False, dtype=np.float32),
        order=1,
    )


def _assert_identity_symmetry(symmetry, *, scale):
    assert symmetry.scale == scale
    assert symmetry.has_rotation is False
    np.testing.assert_array_equal(symmetry.rotmat, np.eye(3))


def _assert_is_float64_array(value):
    assert value.dtype == jnp.float64


def _make_identity_dof_map(num_dofs):
    return make_optimizable_dof_map_spec(
        template_full_dofs=np.zeros(num_dofs, dtype=np.float64),
        owner_segments=[(0, num_dofs, 0, num_dofs)],
        input_mode="full",
        input_start=0,
        input_end=num_dofs,
    )


def _make_curve_quadpoints():
    return np.linspace(0.0, 1.0, 4, endpoint=False, dtype=np.float64)


def _make_surface_spec():
    return make_surface_rzfourier_spec(
        rc=np.array([[1.0, 0.0, 0.0], [0.2, 0.1, -0.1]], dtype=np.float64),
        zs=np.array([[0.0, 0.0, 0.0], [0.3, 0.0, -0.2]], dtype=np.float64),
        quadpoints_phi=np.linspace(0.0, 0.5, 4, endpoint=False, dtype=np.float64),
        quadpoints_theta=np.linspace(0.0, 1.0, 5, endpoint=False, dtype=np.float64),
        nfp=2,
        stellsym=True,
        rs=None,
        zc=None,
    )


def _make_surface_xyz_spec():
    return make_surface_xyz_fourier_spec(
        dofs=np.array([1.0, 0.1, 0.0, 0.1], dtype=np.float64),
        quadpoints_phi=np.linspace(0.0, 1.0, 4, endpoint=False, dtype=np.float64),
        quadpoints_theta=np.linspace(0.0, 1.0, 5, endpoint=False, dtype=np.float64),
        nfp=1,
        stellsym=True,
        mpol=1,
        ntor=0,
    )


def _make_surface_xyztensor_spec():
    return make_surface_xyz_tensor_fourier_spec(
        dofs=np.array([1.0, 0.1, 0.0, 0.1], dtype=np.float64),
        quadpoints_phi=np.linspace(0.0, 1.0, 4, endpoint=False, dtype=np.float64),
        quadpoints_theta=np.linspace(0.0, 1.0, 5, endpoint=False, dtype=np.float64),
        nfp=1,
        stellsym=True,
        mpol=1,
        ntor=0,
    )


def _make_curve_spec_kind_samples():
    quadpoints = _make_curve_quadpoints()
    base_curve = _make_curve_spec()
    base_curve_map = _make_identity_dof_map(base_curve.dofs.shape[0])
    rotation = make_frame_rotation_spec(
        dofs=np.zeros(3, dtype=np.float64),
        quadpoints=quadpoints,
        order=1,
        scale=1.0,
    )
    return {
        "xyz_fourier": base_curve,
        "rz_fourier": make_curve_rzfourier_spec(
            dofs=np.arange(6, dtype=np.float64),
            quadpoints=quadpoints,
            order=1,
            nfp=2,
            stellsym=True,
        ),
        "planar_fourier": make_curve_planarfourier_spec(
            dofs=np.arange(8, dtype=np.float64),
            quadpoints=quadpoints,
            order=1,
        ),
        "helical": make_curve_helical_spec(
            dofs=np.arange(6, dtype=np.float64),
            quadpoints=quadpoints,
            order=1,
            m=2,
            ell=3,
            R0=1.5,
            r=0.2,
        ),
        "cws_fourier_rz": make_curve_cwsfourier_rz_spec(
            dofs=np.arange(8, dtype=np.float64),
            quadpoints=quadpoints,
            surface=_make_surface_spec(),
            order=1,
            G=0.1,
            H=-0.2,
        ),
        "perturbed": make_curve_perturbed_spec(
            dofs=np.array(base_curve.dofs),
            quadpoints=quadpoints,
            base_curve=base_curve,
            base_curve_map=base_curve_map,
            sample_gamma=np.zeros((quadpoints.shape[0], 3), dtype=np.float64),
            sample_gammadash=np.zeros((quadpoints.shape[0], 3), dtype=np.float64),
            sample_gammadashdash=np.zeros((quadpoints.shape[0], 3), dtype=np.float64),
            sample_gammadashdashdash=np.zeros(
                (quadpoints.shape[0], 3),
                dtype=np.float64,
            ),
        ),
        "filament": make_curve_filament_spec(
            dofs=np.concatenate(
                [np.array(base_curve.dofs), np.array(rotation.dofs)],
                axis=0,
            ),
            quadpoints=quadpoints,
            base_curve=base_curve,
            base_curve_map=base_curve_map,
            rotation=rotation,
            rotation_map=_make_identity_dof_map(rotation.dofs.shape[0]),
            frame_kind="centroid",
            dn=0.1,
            db=-0.2,
        ),
    }


def test_make_coil_symmetry_spec_defaults_to_identity_without_rotation():
    symmetry = make_coil_symmetry_spec(scale=2.5)

    _assert_identity_symmetry(symmetry, scale=2.5)
    _assert_is_float64_array(symmetry.rotmat)


def test_make_coil_spec_uses_default_symmetry_contract():
    coil = make_coil_spec(
        curve=_make_curve_spec(), current=make_current_value_spec(3.0)
    )

    _assert_identity_symmetry(coil.symmetry, scale=1.0)


def test_curve_spec_kind_covers_all_supported_curve_variants():
    for expected_kind, curve_spec in _make_curve_spec_kind_samples().items():
        assert curve_spec_kind(curve_spec) == expected_kind


def test_curve_spec_kind_rejects_unrelated_same_name_lookalike():
    impostor = type("CurveXYZFourierSpec", (), {})()

    with pytest.raises(
        TypeError, match="Unsupported curve spec type: CurveXYZFourierSpec"
    ):
        curve_spec_kind(impostor)


def test_surface_spec_kind_covers_supported_fixed_surface_variants():
    rz_spec = _make_surface_spec()
    xyz_spec = _make_surface_xyz_spec()
    xyztensor_spec = _make_surface_xyztensor_spec()

    assert surface_spec_kind(rz_spec) == "rz_fourier"
    assert surface_spec_kind(xyz_spec) == "xyz_fourier"
    assert surface_spec_kind(xyztensor_spec) == "xyz_tensor_fourier"


def test_apply_coil_symmetry_rotates_geometry_and_scales_current():
    phi = np.pi / 2.0
    symmetry = make_coil_symmetry_spec(
        rotmat=np.array(
            [
                [np.cos(phi), -np.sin(phi), 0.0],
                [np.sin(phi), np.cos(phi), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ).T,
        scale=-2.0,
    )
    gamma = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    gammadash = jnp.asarray([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    current = jnp.asarray([3.0], dtype=jnp.float64)

    gamma_sym, gammadash_sym, current_sym = apply_coil_symmetry(
        gamma, gammadash, current, symmetry
    )

    np.testing.assert_allclose(gamma_sym, gamma @ symmetry.rotmat)
    np.testing.assert_allclose(gammadash_sym, gammadash @ symmetry.rotmat)
    np.testing.assert_allclose(current_sym, np.array([-6.0]))


def test_make_field_eval_and_fixed_surface_flux_specs_preserve_shapes_and_float64():
    points = make_field_eval_spec(np.ones((5, 3), dtype=np.float32))
    flux = make_fixed_surface_flux_spec(
        points=np.arange(18, dtype=np.float32).reshape(6, 3),
        normal=np.ones((2, 3, 3), dtype=np.float32),
        target=np.zeros((2, 3), dtype=np.float32),
        definition="quadratic flux",
    )

    assert points.points.shape == (5, 3)
    _assert_is_float64_array(points.points)
    _assert_is_float64_array(flux.points)
    _assert_is_float64_array(flux.normal)
    _assert_is_float64_array(flux.target)
    assert flux.nphi == 2
    assert flux.ntheta == 3


def test_make_grouped_coil_set_spec_accepts_group_specs_and_raw_group_tuples():
    existing_group = make_coil_group_spec(
        gammas=np.zeros((1, 4, 3), dtype=np.float32),
        gammadashs=np.ones((1, 4, 3), dtype=np.float32),
        currents=np.array([1.5], dtype=np.float32),
        coil_indices=[7],
    )

    grouped = make_grouped_coil_set_spec(
        [
            existing_group,
            (
                np.full((2, 4, 3), 2.0, dtype=np.float32),
                np.full((2, 4, 3), -1.0, dtype=np.float32),
                np.array([3.0, -3.0], dtype=np.float32),
                [1, 4],
            ),
        ]
    )

    assert grouped.groups[0] is existing_group
    assert grouped.groups[1].coil_indices == (1, 4)
    _assert_is_float64_array(grouped.groups[1].gammas)
    _assert_is_float64_array(grouped.groups[1].currents)


def test_grouped_coil_set_spec_is_a_real_jittable_pytree():
    grouped = make_grouped_coil_set_spec(
        [
            (
                np.full((1, 4, 3), 2.0, dtype=np.float32),
                np.full((1, 4, 3), -1.0, dtype=np.float32),
                np.array([3.0], dtype=np.float32),
                [1],
            ),
            (
                np.full((2, 4, 3), 4.0, dtype=np.float32),
                np.full((2, 4, 3), -2.0, dtype=np.float32),
                np.array([5.0, -7.0], dtype=np.float32),
                [2, 9],
            ),
        ]
    )

    leaves, treedef = jax.tree_util.tree_flatten(grouped)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert treedef == jax.tree_util.tree_structure(rebuilt)
    assert rebuilt.groups[0].coil_indices == (1,)
    assert rebuilt.groups[1].coil_indices == (2, 9)
    np.testing.assert_allclose(rebuilt.groups[0].gammas, grouped.groups[0].gammas)
    np.testing.assert_allclose(rebuilt.groups[1].currents, grouped.groups[1].currents)

    @jax.jit
    def current_sum(spec):
        total = jnp.asarray(0.0, dtype=jnp.float64)
        for group in spec.groups:
            total = total + jnp.sum(group.currents)
        return total

    np.testing.assert_allclose(current_sum(grouped), np.array(1.0))


def test_single_stage_runtime_spec_is_a_real_jittable_pytree():
    surface = _make_surface_xyztensor_spec()
    coil_set = make_grouped_coil_set_spec(
        [
            (
                np.full((1, 4, 3), 2.0, dtype=np.float32),
                np.full((1, 4, 3), -1.0, dtype=np.float32),
                np.array([3.0], dtype=np.float32),
                [1],
            )
        ]
    )
    seed = make_single_stage_seed_spec(
        surface=surface,
        coil_set=coil_set,
        coil_dof_extraction=make_coil_set_dof_extraction_spec(()),
        coil_dofs=np.array([0.5, -0.25], dtype=np.float64),
        boozer_iota=0.123,
        boozer_G=4.5,
        target_labels=("qs_error", "boozer_residual"),
        hardware_constants=(("tf_current_hard_limit_A", 2.0),),
        self_intersection_mode="supported-surface-jax",
        schema_version=1,
        num_tf_coils=1,
        banana_curve_index=0,
        tf_current_A=80000.0,
        banana_current_A=123.0,
    )
    runtime = make_single_stage_runtime_spec(
        seed=seed,
        mpol=surface.mpol,
        ntor=surface.ntor,
        nfp=surface.nfp,
        nphi=surface.quadpoints_phi.shape[0],
        ntheta=surface.quadpoints_theta.shape[0],
    )

    leaves, treedef = jax.tree_util.tree_flatten(runtime)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert rebuilt.mpol == surface.mpol
    assert rebuilt.seed.target_labels == ("qs_error", "boozer_residual")
    assert rebuilt.seed.hardware_constants == (("tf_current_hard_limit_A", 2.0),)
    _assert_is_float64_array(rebuilt.seed.boozer_iota)
    _assert_is_float64_array(rebuilt.seed.boozer_G)
    _assert_is_float64_array(rebuilt.seed.coil_dofs)
    assert rebuilt.seed.num_tf_coils == 1
    assert rebuilt.seed.banana_curve_index == 0
    assert rebuilt.seed.tf_current_A == 80000.0
    assert rebuilt.seed.banana_current_A == 123.0

    @jax.jit
    def seed_scalar(spec):
        return spec.seed.boozer_iota[0] + spec.seed.boozer_G[0] + spec.seed.surface.dofs[0]

    np.testing.assert_allclose(seed_scalar(runtime), np.array(5.623))


def test_coil_set_dof_extraction_spec_is_a_real_jittable_pytree():
    extraction_spec = make_coil_set_dof_extraction_spec(
        [
            make_coil_dof_extraction_spec(
                curve=_make_curve_spec(),
                curve_map=_make_identity_dof_map(9),
                current_map=_make_identity_dof_map(1),
                scale=2.0,
            )
        ]
    )

    leaves, treedef = jax.tree_util.tree_flatten(extraction_spec)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert treedef == jax.tree_util.tree_structure(rebuilt)
    assert len(rebuilt.coils) == 1
    assert rebuilt.coils[0].symmetry.scale == 2.0

    @jax.jit
    def current_value(spec, owner_dofs):
        return spec.coils[0].current_map.template_full_dofs[0] + owner_dofs[0]

    np.testing.assert_allclose(
        current_value(extraction_spec, jnp.asarray([3.0], dtype=jnp.float64)),
        np.array(3.0),
    )


def test_make_surface_rzfourier_spec_fills_rs_zc_defaults_from_rc_shape():
    spec = make_surface_rzfourier_spec(
        rc=np.array([[1.0, 0.0, 0.0], [0.2, 0.1, -0.1]], dtype=np.float32),
        zs=np.array([[0.0, 0.0, 0.0], [0.3, 0.0, -0.2]], dtype=np.float32),
        quadpoints_phi=np.linspace(0.0, 0.5, 4, endpoint=False, dtype=np.float32),
        quadpoints_theta=np.linspace(0.0, 1.0, 5, endpoint=False, dtype=np.float32),
        nfp=2,
        stellsym=True,
        rs=None,
        zc=None,
    )

    assert spec.mpol == 1
    assert spec.ntor == 1
    assert spec.nfp == 2
    assert spec.stellsym is True
    _assert_is_float64_array(spec.rc)
    _assert_is_float64_array(spec.zs)
    _assert_is_float64_array(spec.rs)
    _assert_is_float64_array(spec.zc)
    np.testing.assert_array_equal(spec.rs, np.zeros_like(spec.rc))
    np.testing.assert_array_equal(spec.zc, np.zeros_like(spec.rc))


def test_non_rz_surface_specs_are_real_jittable_pytrees():
    xyz_spec = _make_surface_xyz_spec()
    xyztensor_spec = _make_surface_xyztensor_spec()

    xyz_gamma = jax.jit(surface_xyz_fourier_gamma_from_spec)(xyz_spec)
    xyz_normal = jax.jit(surface_xyz_fourier_normal_from_spec)(xyz_spec)
    xyztensor_gamma = jax.jit(surface_xyz_tensor_fourier_gamma_from_spec)(
        xyztensor_spec
    )
    xyztensor_normal = jax.jit(surface_xyz_tensor_fourier_normal_from_spec)(
        xyztensor_spec
    )

    assert xyz_gamma.shape == (4, 5, 3)
    assert xyz_normal.shape == (4, 5, 3)
    assert xyztensor_gamma.shape == (4, 5, 3)
    assert xyztensor_normal.shape == (4, 5, 3)
    assert xyz_spec.scatter_indices.dtype == jnp.int32
    assert xyz_spec.coeff_template.shape == (12,)
    assert xyztensor_spec.scatter_indices.dtype == jnp.int32
