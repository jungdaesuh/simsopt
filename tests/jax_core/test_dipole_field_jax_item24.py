"""Item 24 parity tests for ``simsopt.jax_core.dipole_field``.

The tests exercise the JAX dipole-field kernels (``B``, ``A``, ``dB``, ``dA``)
against the upstream C++ oracle ``simsoptpp.dipole_field_{B,A,dB,dA}`` at the
``direct_kernel`` parity-ladder lane. All tolerances are imported from
``benchmarks.validation_ladder_contract.parity_ladder_tolerances`` -- no
``rtol`` / ``atol`` literals are inlined in the test body.

The same fixtures are then replayed under ``jax.transfer_guard("disallow")``
to prove the JAX kernels do not trigger implicit host transfers when
consuming device-resident input arrays.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import simsoptpp as sopp

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.jax_core.dipole_field import (
    DipoleFieldSpec,
    dipole_field_A_from_spec as dipole_field_A,
    dipole_field_B_from_spec as dipole_field_B,
    dipole_field_dA_from_spec as dipole_field_dA,
    dipole_field_dB_from_spec as dipole_field_dB,
)


_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]


def _seeded_dipoles(seed: int, num_dipoles: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """Generate dipole moments and positions away from a unit shell.

    Positions are inside a ``[-0.3, 0.3]`` cube so the field-evaluation
    points (sampled on a ``[0.6, 2.0]`` shell) never coincide with a dipole
    site. Moments are drawn from a Gaussian and scaled to physical units.
    """
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-0.3, 0.3, size=(num_dipoles, 3)).astype(
        np.float64, copy=False
    )
    moments = rng.normal(loc=0.0, scale=1.0, size=(num_dipoles, 3)).astype(
        np.float64, copy=False
    )
    return moments, positions


def _seeded_points(seed: int, count: int = 60) -> np.ndarray:
    """Generate evaluation points on a far-field shell.

    All components are bounded away from ``0`` so the points cannot land on
    the dipole grid in ``_seeded_dipoles`` (which sits inside ``|x| < 0.3``).
    """
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.6, 2.0, size=(count, 3))
    flips = rng.choice([-1.0, 1.0], size=base.shape)
    return (base * flips).astype(np.float64, copy=False)


# ── CPU-oracle parity ────────────────────────────────────────────────


def test_dipole_field_jax_vs_cpp_direct_kernel():
    """B, A, dB, dA parity vs ``simsoptpp.dipole_field_*`` C++ oracle.

    Oracle: ``simsoptpp.dipole_field_B / _A / _dB / _dA``. Tolerance lane:
    ``direct_kernel`` (``rtol=1e-10``, ``atol=1e-12``). Same-state parity at
    machine precision is expected because the JAX kernels mirror the CPU
    scalar arithmetic literally (the XSIMD path in ``dipole_field.cpp`` is a
    SIMD repack of the same scalar formula).
    """
    moments_np, positions_np = _seeded_dipoles(seed=42, num_dipoles=14)
    points_np = _seeded_points(seed=4242, count=60)

    # C++ oracle outputs.
    B_cpp = np.asarray(
        sopp.dipole_field_B(points_np, positions_np, moments_np), dtype=np.float64
    )
    A_cpp = np.asarray(
        sopp.dipole_field_A(points_np, positions_np, moments_np), dtype=np.float64
    )
    dB_cpp = np.asarray(
        sopp.dipole_field_dB(points_np, positions_np, moments_np), dtype=np.float64
    )
    dA_cpp = np.asarray(
        sopp.dipole_field_dA(points_np, positions_np, moments_np), dtype=np.float64
    )

    spec = DipoleFieldSpec(
        dipole_moments=jnp.asarray(moments_np, dtype=jnp.float64),
        dipole_points=jnp.asarray(positions_np, dtype=jnp.float64),
    )
    B_jax = np.asarray(dipole_field_B(points_np, spec), dtype=np.float64)
    A_jax = np.asarray(dipole_field_A(points_np, spec), dtype=np.float64)
    dB_jax = np.asarray(dipole_field_dB(points_np, spec), dtype=np.float64)
    dA_jax = np.asarray(dipole_field_dA(points_np, spec), dtype=np.float64)

    np.testing.assert_allclose(B_jax, B_cpp, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(A_jax, A_cpp, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(dB_jax, dB_cpp, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(dA_jax, dA_cpp, rtol=_RTOL, atol=_ATOL)


def test_dipole_field_convention_dB_symmetric():
    """``dB`` from a dipole is symmetric in the last two axes (``∂_j B_l = ∂_l B_j``)."""
    moments_np, positions_np = _seeded_dipoles(seed=7, num_dipoles=11)
    points_np = _seeded_points(seed=77, count=50)
    spec = DipoleFieldSpec(
        dipole_moments=jnp.asarray(moments_np, dtype=jnp.float64),
        dipole_points=jnp.asarray(positions_np, dtype=jnp.float64),
    )
    dB = np.asarray(dipole_field_dB(points_np, spec), dtype=np.float64)
    np.testing.assert_allclose(dB, np.transpose(dB, (0, 2, 1)), rtol=_RTOL, atol=_ATOL)


def test_dipole_field_dA_antisymmetric_part_matches_B():
    """The antisymmetric part of ``∂A`` equals the magnetic field ``B``.

    For a magnetic vector potential, ``B = ∇ × A`` so
    ``B_i = ε_ijk ∂_j A_k = (1/2) ε_ijk (∂_j A_k - ∂_k A_j)``. Our ``dA``
    is stored as ``dA[p, l, j] = ∂_j A_l``, so
    ``(∂_j A_l - ∂_l A_j)[p, l, j] = dA - dA^T`` (over the last two axes).
    The expected ``B`` component map from the curl is:
        ``B_x = ∂_y A_z - ∂_z A_y = dA[p, 2, 1] - dA[p, 1, 2]``
        ``B_y = ∂_z A_x - ∂_x A_z = dA[p, 0, 2] - dA[p, 2, 0]``
        ``B_z = ∂_x A_y - ∂_y A_x = dA[p, 1, 0] - dA[p, 0, 1]``.
    This is the physics consistency check that ties ``dA`` and ``B`` together
    via the standard cartesian curl identity.
    """
    moments_np, positions_np = _seeded_dipoles(seed=13, num_dipoles=10)
    points_np = _seeded_points(seed=1313, count=50)
    spec = DipoleFieldSpec(
        dipole_moments=jnp.asarray(moments_np, dtype=jnp.float64),
        dipole_points=jnp.asarray(positions_np, dtype=jnp.float64),
    )
    B = np.asarray(dipole_field_B(points_np, spec), dtype=np.float64)
    dA = np.asarray(dipole_field_dA(points_np, spec), dtype=np.float64)
    Bx_from_curl = dA[:, 2, 1] - dA[:, 1, 2]
    By_from_curl = dA[:, 0, 2] - dA[:, 2, 0]
    Bz_from_curl = dA[:, 1, 0] - dA[:, 0, 1]
    np.testing.assert_allclose(Bx_from_curl, B[:, 0], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(By_from_curl, B[:, 1], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(Bz_from_curl, B[:, 2], rtol=_RTOL, atol=_ATOL)


def test_dipole_field_output_shapes_and_dtypes():
    """Output shapes and float64 dtype contract for all four kernels."""
    moments_np, positions_np = _seeded_dipoles(seed=99, num_dipoles=12)
    points_np = _seeded_points(seed=999, count=51)
    spec = DipoleFieldSpec(
        dipole_moments=jnp.asarray(moments_np, dtype=jnp.float64),
        dipole_points=jnp.asarray(positions_np, dtype=jnp.float64),
    )
    B = dipole_field_B(points_np, spec)
    A = dipole_field_A(points_np, spec)
    dB = dipole_field_dB(points_np, spec)
    dA = dipole_field_dA(points_np, spec)
    assert B.shape == (51, 3), B.shape
    assert A.shape == (51, 3), A.shape
    assert dB.shape == (51, 3, 3), dB.shape
    assert dA.shape == (51, 3, 3), dA.shape
    assert B.dtype == jnp.float64
    assert A.dtype == jnp.float64
    assert dB.dtype == jnp.float64
    assert dA.dtype == jnp.float64


# ── Transfer-guard discipline ────────────────────────────────────────


def _device_points(points: np.ndarray) -> jax.Array:
    return jnp.asarray(points, dtype=jnp.float64)


def test_dipole_field_strict_transfer_guard():
    """All four kernels run cleanly under ``transfer_guard('disallow')``.

    Spec arrays and evaluation points are placed on the device under the
    default (``allow``) guard so the strict-guard region only measures the
    compiled kernels. Any implicit host transfer inside the compiled path
    would raise ``jax.errors.JaxRuntimeError``.
    """
    moments_np, positions_np = _seeded_dipoles(seed=24, num_dipoles=14)
    points_np = _seeded_points(seed=2424, count=60)

    spec = DipoleFieldSpec(
        dipole_moments=jnp.asarray(moments_np, dtype=jnp.float64),
        dipole_points=jnp.asarray(positions_np, dtype=jnp.float64),
    )
    points_dev = _device_points(points_np)

    # Drain pending compilation / device transfers under the default guard.
    spec.dipole_moments.block_until_ready()
    spec.dipole_points.block_until_ready()
    points_dev.block_until_ready()
    dipole_field_B(points_dev, spec).block_until_ready()
    dipole_field_A(points_dev, spec).block_until_ready()
    dipole_field_dB(points_dev, spec).block_until_ready()
    dipole_field_dA(points_dev, spec).block_until_ready()

    with jax.transfer_guard("disallow"):
        dipole_field_B(points_dev, spec).block_until_ready()
        dipole_field_A(points_dev, spec).block_until_ready()
        dipole_field_dB(points_dev, spec).block_until_ready()
        dipole_field_dA(points_dev, spec).block_until_ready()


# ── Input validation ─────────────────────────────────────────────────


def test_dipole_field_rejects_malformed_points():
    """``_validate_points`` rejects shapes that are not ``(N, 3)``."""
    spec = DipoleFieldSpec(
        dipole_moments=jnp.ones((2, 3), dtype=jnp.float64),
        dipole_points=jnp.zeros((2, 3), dtype=jnp.float64),
    )
    with pytest.raises(ValueError):
        dipole_field_B(np.zeros((5,)), spec)
    with pytest.raises(ValueError):
        dipole_field_B(np.zeros((5, 2)), spec)


def test_dipole_field_rejects_malformed_dipole_arrays():
    """``_validate_dipole_arrays`` rejects mismatched leading axes."""
    points = np.zeros((3, 3), dtype=np.float64)
    spec_bad_m = DipoleFieldSpec(
        dipole_moments=jnp.ones((3, 3), dtype=jnp.float64),
        dipole_points=jnp.zeros((2, 3), dtype=jnp.float64),
    )
    spec_bad_shape = DipoleFieldSpec(
        dipole_moments=jnp.ones((2, 2), dtype=jnp.float64),
        dipole_points=jnp.zeros((2, 3), dtype=jnp.float64),
    )
    with pytest.raises(ValueError):
        dipole_field_B(points, spec_bad_m)
    with pytest.raises(ValueError):
        dipole_field_B(points, spec_bad_shape)
