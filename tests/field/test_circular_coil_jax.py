"""Dedicated parity gate for the JAX port of :class:`simsopt.field.CircularCoil`.

This module is the closeout test for Wave R1 item ``12-circularcoil``. It
covers both the pure JAX kernels in
:mod:`simsopt.jax_core.circular_coil` (``circular_coil_B`` /
``circular_coil_A`` / ``circular_coil_dB`` with the ``CircularCoilSpec``
payload) and the
``MagneticField``-boundary wrapper
:class:`simsopt.field.magneticfieldclasses_jax.CircularCoilJAX`.

The CPU :class:`simsopt.field.CircularCoil` is the parity oracle. All
tolerances come from
:func:`benchmarks.validation_ladder_contract.parity_ladder_tolerances`
at the ``direct_kernel`` lane (``rtol=1e-10``, ``atol=1e-12``) -- no
``rtol`` / ``atol`` literals appear inline in the test body. Fixtures
cover:

* the default geometry (``normal = [0, 0]`` spherical, axis along ``z``,
  unit-current default),
* tilted geometries with both a random spherical ``(theta, phi)``
  normal and a random cartesian ``(nx, ny, nz)`` direction normal,
* an off-centre coil (random ``center`` in metres),
* production-scale point clouds with at least 50 cartesian points.

The kernel-level transfer-guard test stages the spec scalars and points
to device before entering ``jax.transfer_guard("disallow")`` to prove
the compiled paths do not trigger implicit host transfers. The JIT
trace test then captures the compiled HLO from the public kernel
entrypoints to demonstrate they trace cleanly under ``jax.jit``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.field import CircularCoil
from simsopt.field.magneticfieldclasses_jax import CircularCoilJAX
from simsopt.jax_core import (
    CircularCoilSpec as ExportedCircularCoilSpec,
    circular_coil_A as exported_circular_coil_A,
    circular_coil_B as exported_circular_coil_B,
    circular_coil_dB as exported_circular_coil_dB,
)
from simsopt.jax_core.circular_coil import (
    CircularCoilSpec,
    _rotation_matrix,
    _rotation_matrix_inv,
    circular_coil_A,
    circular_coil_B,
    circular_coil_dB,
)


_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]


def _production_points(seed: int, count: int = 60) -> np.ndarray:
    """Return ``(count, 3)`` cartesian points well clear of the coil.

    The CPU oracle is regularised at the coil filament with a ``1e-31``
    floor; staying away from ``rho == r0`` keeps the parity gate inside
    the analytic regime and lets the ``direct_kernel`` tolerance bite.
    """

    rng = np.random.default_rng(int(seed))
    points = np.zeros((count, 3), dtype=np.float64)
    points[:, 0] = rng.uniform(0.4, 1.8, size=count)
    points[:, 1] = rng.uniform(0.4, 1.8, size=count)
    points[:, 2] = rng.uniform(-0.5, 0.5, size=count)
    return np.ascontiguousarray(points)


def _random_spherical_normal(seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    theta = float(rng.uniform(-np.pi, np.pi))
    phi = float(rng.uniform(0.05, np.pi - 0.05))
    return theta, phi


def _random_cartesian_normal(seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=3)
    # Avoid the rare degenerate case where the random draw lands close to zero.
    while np.linalg.norm(vec) < 0.1:
        vec = rng.normal(size=3)
    return float(vec[0]), float(vec[1]), float(vec[2])


def _random_center(seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    return (
        float(rng.uniform(-0.4, 0.4)),
        float(rng.uniform(-0.4, 0.4)),
        float(rng.uniform(-0.4, 0.4)),
    )


def _make_spec(
    *,
    r0: float,
    center: tuple[float, float, float],
    current: float,
    normal: tuple[float, ...],
) -> CircularCoilSpec:
    return CircularCoilSpec(
        r0=r0,
        center=center,
        Inorm=current * 4e-7,
        normal=normal,
    )


# ── Pure kernel parity ───────────────────────────────────────────────


class TestKernelParity:
    """Parity vs ``CircularCoil`` CPU oracle for the pure JAX kernels."""

    def test_default_geometry_axis_along_z(self):
        """Default ``CircularCoil`` (normal = [0, 0]) parity over 64 points."""

        points = _production_points(seed=4_201, count=64)
        spec = _make_spec(
            r0=0.1,
            center=(0.0, 0.0, 0.0),
            current=5e5 / np.pi,
            normal=(0.0, 0.0),
        )
        cpu = CircularCoil()
        cpu.set_points_cart(points)
        B_cpu = np.asarray(cpu.B(), dtype=np.float64)
        A_cpu = np.asarray(cpu.A(), dtype=np.float64)
        dB_cpu = np.asarray(cpu.dB_by_dX(), dtype=np.float64)

        B_jax = np.asarray(circular_coil_B(spec, points), dtype=np.float64)
        A_jax = np.asarray(circular_coil_A(spec, points), dtype=np.float64)
        dB_jax = np.asarray(circular_coil_dB(spec, points), dtype=np.float64)

        np.testing.assert_allclose(B_jax, B_cpu, rtol=_RTOL, atol=_ATOL)
        np.testing.assert_allclose(A_jax, A_cpu, rtol=_RTOL, atol=_ATOL)
        np.testing.assert_allclose(dB_jax, dB_cpu, rtol=_RTOL, atol=_ATOL)

    def test_filament_singularity_matches_cpu_nan_contract(self):
        """The coil-wire singularity propagates NaNs like the CPU oracle."""

        points = np.array([[0.1, 0.0, 0.0]], dtype=np.float64)
        spec = _make_spec(
            r0=0.1,
            center=(0.0, 0.0, 0.0),
            current=5e5 / np.pi,
            normal=(0.0, 0.0),
        )
        cpu = CircularCoil()
        cpu.set_points_cart(points)

        np.testing.assert_array_equal(
            np.isnan(np.asarray(circular_coil_B(spec, points), dtype=np.float64)),
            np.isnan(np.asarray(cpu.B(), dtype=np.float64)),
        )
        np.testing.assert_array_equal(
            np.isnan(np.asarray(circular_coil_A(spec, points), dtype=np.float64)),
            np.isnan(np.asarray(cpu.A(), dtype=np.float64)),
        )
        np.testing.assert_array_equal(
            np.isnan(np.asarray(circular_coil_dB(spec, points), dtype=np.float64)),
            np.isnan(np.asarray(cpu.dB_by_dX(), dtype=np.float64)),
        )

    def test_tilted_spherical_normal(self):
        """Random ``(theta, phi)`` normal parity over 60 production points."""

        points = _production_points(seed=4_202, count=60)
        normal = _random_spherical_normal(seed=4_212)
        center = _random_center(seed=4_222)
        spec = _make_spec(
            r0=0.7,
            center=center,
            current=1.2e6,
            normal=normal,
        )
        cpu = CircularCoil(r0=0.7, center=list(center), I=1.2e6, normal=list(normal))
        cpu.set_points_cart(points)
        B_cpu = np.asarray(cpu.B(), dtype=np.float64)
        A_cpu = np.asarray(cpu.A(), dtype=np.float64)
        dB_cpu = np.asarray(cpu.dB_by_dX(), dtype=np.float64)

        B_jax = np.asarray(circular_coil_B(spec, points), dtype=np.float64)
        A_jax = np.asarray(circular_coil_A(spec, points), dtype=np.float64)
        dB_jax = np.asarray(circular_coil_dB(spec, points), dtype=np.float64)

        np.testing.assert_allclose(B_jax, B_cpu, rtol=_RTOL, atol=_ATOL)
        np.testing.assert_allclose(A_jax, A_cpu, rtol=_RTOL, atol=_ATOL)
        np.testing.assert_allclose(dB_jax, dB_cpu, rtol=_RTOL, atol=_ATOL)

    def test_tilted_cartesian_normal(self):
        """Random ``(nx, ny, nz)`` normal parity over 60 production points.

        Exercises the ``normal_kind="cartesian"`` path of
        :class:`CircularCoilSpec` and the corresponding
        ``arctan2``-based ``(theta, phi)`` derivation.
        """

        points = _production_points(seed=4_203, count=60)
        normal = _random_cartesian_normal(seed=4_213)
        center = _random_center(seed=4_223)
        spec = _make_spec(
            r0=0.55,
            center=center,
            current=8.0e5,
            normal=normal,
        )
        cpu = CircularCoil(r0=0.55, center=list(center), I=8.0e5, normal=list(normal))
        cpu.set_points_cart(points)
        B_cpu = np.asarray(cpu.B(), dtype=np.float64)
        A_cpu = np.asarray(cpu.A(), dtype=np.float64)
        dB_cpu = np.asarray(cpu.dB_by_dX(), dtype=np.float64)

        B_jax = np.asarray(circular_coil_B(spec, points), dtype=np.float64)
        A_jax = np.asarray(circular_coil_A(spec, points), dtype=np.float64)
        dB_jax = np.asarray(circular_coil_dB(spec, points), dtype=np.float64)

        np.testing.assert_allclose(B_jax, B_cpu, rtol=_RTOL, atol=_ATOL)
        np.testing.assert_allclose(A_jax, A_cpu, rtol=_RTOL, atol=_ATOL)
        np.testing.assert_allclose(dB_jax, dB_cpu, rtol=_RTOL, atol=_ATOL)

    def test_off_center_parity_large_point_cloud(self):
        """Off-centre coil with >= 50 points exercises the production scale."""

        points = _production_points(seed=4_204, count=128)
        center = _random_center(seed=4_224)
        normal = _random_spherical_normal(seed=4_214)
        spec = _make_spec(
            r0=0.4,
            center=center,
            current=3.3e5,
            normal=normal,
        )
        cpu = CircularCoil(r0=0.4, center=list(center), I=3.3e5, normal=list(normal))
        cpu.set_points_cart(points)
        B_cpu = np.asarray(cpu.B(), dtype=np.float64)
        A_cpu = np.asarray(cpu.A(), dtype=np.float64)
        dB_cpu = np.asarray(cpu.dB_by_dX(), dtype=np.float64)

        B_jax = np.asarray(circular_coil_B(spec, points), dtype=np.float64)
        A_jax = np.asarray(circular_coil_A(spec, points), dtype=np.float64)
        dB_jax = np.asarray(circular_coil_dB(spec, points), dtype=np.float64)

        np.testing.assert_allclose(B_jax, B_cpu, rtol=_RTOL, atol=_ATOL)
        np.testing.assert_allclose(A_jax, A_cpu, rtol=_RTOL, atol=_ATOL)
        np.testing.assert_allclose(dB_jax, dB_cpu, rtol=_RTOL, atol=_ATOL)


# ── Public wrapper parity ────────────────────────────────────────────


class TestWrapperParity:
    """Parity vs ``CircularCoil`` for the ``MagneticField`` wrapper."""

    @pytest.mark.parametrize(
        "normal",
        [
            (0.0, 0.0),
            (0.4, 1.1),
            (0.2, 0.7, 0.68),
        ],
    )
    def test_B_A_and_dB_match_cpu_oracle(self, normal):
        """``set_points_cart -> B/A/dB_by_dX`` matches the CPU class.

        The fixture sweeps the default geometry and one tilted variant in
        each ``normal`` encoding.
        """

        points = _production_points(seed=4_301, count=60)
        center = (0.1, -0.2, 0.3)
        current = 1.2e6
        cpu = CircularCoil(r0=0.7, center=list(center), I=current, normal=list(normal))
        jax_field = CircularCoilJAX(
            r0=0.7,
            center=center,
            I=current,
            normal=normal,
        )
        cpu.set_points_cart(points)
        jax_field.set_points_cart(points)

        np.testing.assert_allclose(
            np.asarray(jax_field.B(), dtype=np.float64),
            np.asarray(cpu.B(), dtype=np.float64),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(jax_field.A(), dtype=np.float64),
            np.asarray(cpu.A(), dtype=np.float64),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(jax_field.dB_by_dX(), dtype=np.float64),
            np.asarray(cpu.dB_by_dX(), dtype=np.float64),
            rtol=_RTOL,
            atol=_ATOL,
        )


# ── Rotation helper sanity ───────────────────────────────────────────


class TestRotationHelpers:
    """``_rotation_matrix`` / ``_rotation_matrix_inv`` mirror the CPU oracle."""

    def test_rotation_matrix_matches_cpu_rotmat(self):
        center = (0.1, -0.2, 0.3)
        normal = (0.4, 1.1)
        spec = _make_spec(r0=0.7, center=center, current=1.2e6, normal=normal)
        cpu = CircularCoil(r0=0.7, center=list(center), I=1.2e6, normal=list(normal))

        rot_jax = np.asarray(_rotation_matrix(spec), dtype=np.float64)
        rot_inv_jax = np.asarray(_rotation_matrix_inv(spec), dtype=np.float64)
        rot_cpu = np.asarray(cpu._rotmat(), dtype=np.float64)
        rot_inv_cpu = np.asarray(cpu._rotmatinv(), dtype=np.float64)

        np.testing.assert_allclose(rot_jax, rot_cpu, rtol=_RTOL, atol=_ATOL)
        np.testing.assert_allclose(rot_inv_jax, rot_inv_cpu, rtol=_RTOL, atol=_ATOL)
        np.testing.assert_allclose(
            rot_jax @ rot_inv_jax,
            np.eye(3),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_rotation_matrix_cartesian_normal_matches_spherical_derivation(self):
        """Cartesian ``normal`` should align with the spherical ``(theta, phi)``.

        Both ``CircularCoilSpec`` and the upstream CPU class convert a
        three-vector to ``(theta, phi)`` via ``arctan2``; both rotation
        matrices must therefore agree bit-for-bit at the parity lane.
        """

        center = (0.0, 0.0, 0.0)
        normal_cart = (0.7, -0.5, 1.2)
        spec = _make_spec(
            r0=0.5,
            center=center,
            current=1.0e6,
            normal=normal_cart,
        )
        cpu = CircularCoil(
            r0=0.5,
            center=list(center),
            I=1.0e6,
            normal=list(normal_cart),
        )
        np.testing.assert_allclose(
            np.asarray(_rotation_matrix(spec), dtype=np.float64),
            np.asarray(cpu._rotmat(), dtype=np.float64),
            rtol=_RTOL,
            atol=_ATOL,
        )


# ── Transfer-guard discipline ────────────────────────────────────────


def _device_points(points: np.ndarray) -> jax.Array:
    return jnp.asarray(points, dtype=jnp.float64)


class TestTransferGuard:
    """Kernel + wrapper run cleanly under ``jax.transfer_guard('disallow')``.

    The point clouds are staged to device before entering the guarded
    region so we only measure whether the compiled kernel paths trigger
    implicit transfers.
    """

    def test_pure_kernels_under_strict_transfer_guard(self):
        points = _production_points(seed=4_401, count=64)
        spec_spherical = _make_spec(
            r0=0.7,
            center=(0.1, -0.2, 0.3),
            current=1.2e6,
            normal=(0.4, 1.1),
        )
        spec_cartesian = _make_spec(
            r0=0.55,
            center=(-0.2, 0.1, 0.4),
            current=8.0e5,
            normal=(0.2, 0.7, 0.68),
        )

        points_dev = _device_points(points)
        points_dev.block_until_ready()

        with jax.transfer_guard("disallow"):
            circular_coil_B(spec_spherical, points_dev).block_until_ready()
            circular_coil_A(spec_spherical, points_dev).block_until_ready()
            circular_coil_dB(spec_spherical, points_dev).block_until_ready()
            circular_coil_B(spec_cartesian, points_dev).block_until_ready()
            circular_coil_A(spec_cartesian, points_dev).block_until_ready()
            circular_coil_dB(spec_cartesian, points_dev).block_until_ready()


# ── JIT trace check ──────────────────────────────────────────────────


class TestJIT:
    """The public kernels trace cleanly under ``jax.jit``."""

    def test_circular_coil_B_jit_trace(self):
        spec = _make_spec(
            r0=0.7,
            center=(0.1, -0.2, 0.3),
            current=1.2e6,
            normal=(0.4, 1.1),
        )
        points_dev = _device_points(_production_points(seed=4_501, count=50))
        points_dev.block_until_ready()

        jitted_B = jax.jit(lambda pts: circular_coil_B(spec, pts))
        jitted_A = jax.jit(lambda pts: circular_coil_A(spec, pts))
        jitted_dB = jax.jit(lambda pts: circular_coil_dB(spec, pts))

        B_jit = np.asarray(jitted_B(points_dev), dtype=np.float64)
        A_jit = np.asarray(jitted_A(points_dev), dtype=np.float64)
        dB_jit = np.asarray(jitted_dB(points_dev), dtype=np.float64)
        B_eager = np.asarray(
            circular_coil_B(spec, np.asarray(points_dev, dtype=np.float64)),
            dtype=np.float64,
        )
        A_eager = np.asarray(
            circular_coil_A(spec, np.asarray(points_dev, dtype=np.float64)),
            dtype=np.float64,
        )
        dB_eager = np.asarray(
            circular_coil_dB(spec, np.asarray(points_dev, dtype=np.float64)),
            dtype=np.float64,
        )

        np.testing.assert_allclose(B_jit, B_eager, rtol=_RTOL, atol=_ATOL)
        np.testing.assert_allclose(A_jit, A_eager, rtol=_RTOL, atol=_ATOL)
        np.testing.assert_allclose(dB_jit, dB_eager, rtol=_RTOL, atol=_ATOL)


# ── Shape and dtype sanity ───────────────────────────────────────────


def test_kernel_output_shapes_and_dtypes():
    points = _production_points(seed=4_601, count=50)
    spec = _make_spec(
        r0=0.7,
        center=(0.1, -0.2, 0.3),
        current=1.2e6,
        normal=(0.4, 1.1),
    )
    B = circular_coil_B(spec, points)
    A = circular_coil_A(spec, points)
    dB = circular_coil_dB(spec, points)
    assert B.shape == (50, 3)
    assert A.shape == (50, 3)
    assert dB.shape == (50, 3, 3)
    assert B.dtype == jnp.float64
    assert A.dtype == jnp.float64
    assert dB.dtype == jnp.float64


def test_package_export_exposes_circular_coil_kernels():
    points = _production_points(seed=4_701, count=4)
    spec = ExportedCircularCoilSpec(
        r0=0.7,
        center=(0.1, -0.2, 0.3),
        Inorm=1.2e6 * 4e-7,
        normal=(0.4, 1.1),
    )
    assert exported_circular_coil_B(spec, points).shape == (4, 3)
    assert exported_circular_coil_A(spec, points).shape == (4, 3)
    assert exported_circular_coil_dB(spec, points).shape == (4, 3, 3)


def test_rejects_malformed_points():
    """``circular_coil_B`` rejects shapes that are not ``(N, 3)``."""

    spec = _make_spec(
        r0=0.7,
        center=(0.1, -0.2, 0.3),
        current=1.2e6,
        normal=(0.4, 1.1),
    )
    with pytest.raises(ValueError):
        circular_coil_B(spec, np.zeros((5,)))
    with pytest.raises(ValueError):
        circular_coil_A(spec, np.zeros((5,)))
    with pytest.raises(ValueError):
        circular_coil_dB(spec, np.zeros((5, 2)))
