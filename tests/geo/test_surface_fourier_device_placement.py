"""Device-placement regressions for SurfaceXYZFourier DOF scattering."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from simsopt_jax.core.surface_fourier_kernels import dofs_to_xyzc


_SCATTER_INDICES = np.asarray((0, 8, 9, 17, 18, 26), dtype=np.int32)
_SURFACE_DOFS = np.asarray((1.0, -2.0, 3.0, -4.0, 5.0, -6.0), dtype=np.float64)
_SURFACE_TANGENT = np.asarray(
    (0.5, -0.25, 0.125, -0.0625, 0.03125, -0.015625),
    dtype=np.float64,
)


def _expected_flat(values: np.ndarray) -> np.ndarray:
    expected = np.zeros(27, dtype=np.float64)
    expected[_SCATTER_INDICES] = values
    return expected


def _flatten_coordinates(coordinates: tuple[jax.Array, ...]) -> np.ndarray:
    return np.concatenate(
        tuple(np.asarray(coordinate).reshape(-1) for coordinate in coordinates)
    )


def test_dofs_to_xyzc_scatter_matches_host_reference() -> None:
    coordinates = dofs_to_xyzc(_SURFACE_DOFS, _SCATTER_INDICES, 1, 1)

    np.testing.assert_array_equal(
        _flatten_coordinates(coordinates),
        _expected_flat(_SURFACE_DOFS),
    )


def test_dofs_to_xyzc_jvp_preserves_committed_device_under_strict_transfer() -> None:
    device = jax.devices()[0]
    surface_dofs = jax.device_put(_SURFACE_DOFS, device=device)
    surface_tangent = jax.device_put(_SURFACE_TANGENT, device=device)
    scatter_indices = jax.device_put(_SCATTER_INDICES, device=device)

    with jax.transfer_guard("disallow"):
        coordinates, coordinate_tangents = jax.jvp(
            lambda values: dofs_to_xyzc(values, scatter_indices, 1, 1),
            (surface_dofs,),
            (surface_tangent,),
        )
        jax.block_until_ready((coordinates, coordinate_tangents))

    arrays = (*coordinates, *coordinate_tangents)
    assert all(array.committed for array in arrays)
    assert {array.device for array in arrays} == {device}
    assert {array.dtype for array in arrays} == {jnp.dtype(jnp.float64)}
    np.testing.assert_array_equal(
        _flatten_coordinates(coordinates),
        _expected_flat(_SURFACE_DOFS),
    )
    np.testing.assert_array_equal(
        _flatten_coordinates(coordinate_tangents),
        _expected_flat(_SURFACE_TANGENT),
    )


def test_dofs_to_xyzc_vjp_preserves_committed_device_under_strict_transfer() -> None:
    device = jax.devices()[0]
    surface_dofs = jax.device_put(_SURFACE_DOFS, device=device)
    scatter_indices = jax.device_put(_SCATTER_INDICES, device=device)
    flat_cotangent = np.arange(27, dtype=np.float64)
    coordinate_cotangents = tuple(
        jax.device_put(values.reshape(3, 3), device=device)
        for values in np.split(flat_cotangent, 3)
    )

    def weighted_surface_sum(
        values: jax.Array,
        indices: jax.Array,
        x_cotangent: jax.Array,
        y_cotangent: jax.Array,
        z_cotangent: jax.Array,
    ) -> jax.Array:
        coordinates = dofs_to_xyzc(values, indices, 1, 1)
        return (
            jnp.vdot(coordinates[0], x_cotangent)
            + jnp.vdot(coordinates[1], y_cotangent)
            + jnp.vdot(coordinates[2], z_cotangent)
        )

    gradient = jax.jit(
        jax.grad(weighted_surface_sum),
        out_shardings=surface_dofs.sharding,
    )

    with jax.transfer_guard("disallow"):
        surface_cotangent = gradient(
            surface_dofs,
            scatter_indices,
            *coordinate_cotangents,
        )
        surface_cotangent.block_until_ready()

    assert surface_cotangent.committed
    assert surface_cotangent.device == device
    assert surface_cotangent.dtype == jnp.dtype(jnp.float64)
    np.testing.assert_array_equal(
        np.asarray(surface_cotangent),
        flat_cotangent[_SCATTER_INDICES],
    )
