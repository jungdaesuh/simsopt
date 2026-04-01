import numpy as np
import jax
import jax.numpy as jnp

from simsopt.jax_core import (
    apply_coil_symmetry,
    curve_spec_kind,
    make_coil_group_spec,
    make_coil_spec,
    make_coil_symmetry_spec,
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
    make_surface_rzfourier_spec,
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
