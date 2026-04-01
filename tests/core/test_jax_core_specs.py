import numpy as np
import jax
import jax.numpy as jnp

from simsopt.jax_core import (
    apply_coil_symmetry,
    make_coil_group_spec,
    make_coil_spec,
    make_coil_symmetry_spec,
    make_current_value_spec,
    make_curve_xyzfourier_spec,
    make_field_eval_spec,
    make_fixed_surface_flux_spec,
    make_grouped_coil_set_spec,
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


def test_make_coil_symmetry_spec_defaults_to_identity_without_rotation():
    symmetry = make_coil_symmetry_spec(scale=2.5)

    _assert_identity_symmetry(symmetry, scale=2.5)
    _assert_is_float64_array(symmetry.rotmat)


def test_make_coil_spec_uses_default_symmetry_contract():
    coil = make_coil_spec(
        curve=_make_curve_spec(), current=make_current_value_spec(3.0)
    )

    _assert_identity_symmetry(coil.symmetry, scale=1.0)


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
