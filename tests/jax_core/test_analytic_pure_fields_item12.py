"""Item 12 (partial) parity tests for ``simsopt.jax_core.analytic_pure_fields``.

The tests exercise the new JAX kernels for ``ToroidalField``,
``PoloidalField``, and ``MirrorModel`` against the upstream CPU classes at
the ``direct_kernel`` parity-ladder lane. All tolerances are imported from
``benchmarks.validation_ladder_contract.parity_ladder_tolerances`` -- no
``rtol`` / ``atol`` literals are inlined in the test body.

The same fixtures are then replayed under
``jax.transfer_guard("disallow")`` to prove the JAX kernels do not trigger
implicit host transfers when consuming device-resident point arrays.

``CircularCoil`` is explicitly deferred (see
``.artifacts/jax_port_goal/blockers/12-circularcoil-debug.md``) because
``jax.scipy.special.ellipk`` / ``ellipe`` are not available in the repo's
``jaxlib`` 0.10.0 runtime.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.field import MirrorModel, PoloidalField, ToroidalField
from simsopt.jax_core.analytic_pure_fields import (
    MirrorModelSpec,
    PoloidalFieldSpec,
    ToroidalFieldSpec,
    mirror_B,
    mirror_dB,
    poloidal_B,
    poloidal_dB,
    toroidal_A,
    toroidal_B,
    toroidal_d2B,
    toroidal_dA,
    toroidal_dB,
)

_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]


def _seeded_points(seed: int, count: int = 50) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.4, 2.0, size=(count, 3))
    base[:, 2] = rng.uniform(-0.6, 0.6, size=count)
    return base.astype(np.float64, copy=False)


def _filter_away_from_axis(
    points: np.ndarray, *, R0: float, margin: float
) -> np.ndarray:
    R_xy = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    mask = np.abs(R_xy - R0) > margin
    return points[mask]


# ── ToroidalField parity ─────────────────────────────────────────────


def test_toroidal_field_jax_vs_cpu():
    """B, dB, d2B, A, dA parity vs ``ToroidalField`` CPU class.

    Oracle: ``simsopt.field.ToroidalField`` (``magneticfieldclasses.py``).
    Tolerance lane: ``direct_kernel``. Same-state parity at machine
    precision is expected because the JAX kernel mirrors the CPU class
    arithmetic literally.
    """
    points = _seeded_points(seed=11, count=50)
    spec = ToroidalFieldSpec(R0=1.3, B0=0.8)
    cpu = ToroidalField(R0=spec.R0, B0=spec.B0)
    cpu.set_points(points)

    B_cpu = np.asarray(cpu.B(), dtype=np.float64)
    dB_cpu = np.asarray(cpu.dB_by_dX(), dtype=np.float64)
    d2B_cpu = np.asarray(cpu.d2B_by_dXdX(), dtype=np.float64)
    A_cpu = np.asarray(cpu.A(), dtype=np.float64)
    dA_cpu = np.asarray(cpu.dA_by_dX(), dtype=np.float64)

    B_jax = np.asarray(toroidal_B(spec, points), dtype=np.float64)
    dB_jax = np.asarray(toroidal_dB(spec, points), dtype=np.float64)
    d2B_jax = np.asarray(toroidal_d2B(spec, points), dtype=np.float64)
    A_jax = np.asarray(toroidal_A(spec, points), dtype=np.float64)
    dA_jax = np.asarray(toroidal_dA(spec, points), dtype=np.float64)

    np.testing.assert_allclose(B_jax, B_cpu, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(dB_jax, dB_cpu, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(d2B_jax, d2B_cpu, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(A_jax, A_cpu, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(dA_jax, dA_cpu, rtol=_RTOL, atol=_ATOL)


# ── PoloidalField parity ─────────────────────────────────────────────


def test_poloidal_field_jax_vs_cpu():
    """B and dB parity vs ``PoloidalField`` CPU class.

    Oracle: ``simsopt.field.PoloidalField``. Points are sampled away from
    the magnetic axis ``sqrt(x**2 + y**2) == R0`` where the CPU class is
    singular (the JAX kernel matches the upstream NaN behaviour on the
    axis but the parity gate would compare ``NaN`` to ``NaN``).
    Tolerance lane: ``direct_kernel``.
    """
    R0 = 1.0
    points = _filter_away_from_axis(
        _seeded_points(seed=23, count=80), R0=R0, margin=0.2
    )
    assert points.shape[0] >= 50, "Need >= 50 production-floor points after filtering"

    spec = PoloidalFieldSpec(R0=R0, B0=1.1, q=1.3)
    cpu = PoloidalField(R0=spec.R0, B0=spec.B0, q=spec.q)
    cpu.set_points(points)

    B_cpu = np.asarray(cpu.B(), dtype=np.float64)
    dB_cpu = np.asarray(cpu.dB_by_dX(), dtype=np.float64)

    B_jax = np.asarray(poloidal_B(spec, points), dtype=np.float64)
    dB_jax = np.asarray(poloidal_dB(spec, points), dtype=np.float64)

    np.testing.assert_allclose(B_jax, B_cpu, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(dB_jax, dB_cpu, rtol=_RTOL, atol=_ATOL)


# ── MirrorModel parity ───────────────────────────────────────────────


def test_mirror_field_jax_vs_cpu():
    """B and dB parity vs ``MirrorModel`` CPU class.

    Oracle: ``simsopt.field.MirrorModel``. Points are kept away from the
    axis (``R = sqrt(x**2 + y**2) == 0``) where the CPU class divides by
    zero. Tolerance lane: ``direct_kernel``.
    """
    points = _filter_away_from_axis(
        _seeded_points(seed=37, count=80), R0=0.0, margin=0.2
    )
    assert points.shape[0] >= 50, "Need >= 50 production-floor points after filtering"

    spec = MirrorModelSpec(B0=6.51292, gamma=0.124904, Z_m=0.98)
    cpu = MirrorModel(B0=spec.B0, gamma=spec.gamma, Z_m=spec.Z_m)
    cpu.set_points(points)

    B_cpu = np.asarray(cpu.B(), dtype=np.float64)
    dB_cpu = np.asarray(cpu.dB_by_dX(), dtype=np.float64)

    B_jax = np.asarray(mirror_B(spec, points), dtype=np.float64)
    dB_jax = np.asarray(mirror_dB(spec, points), dtype=np.float64)

    np.testing.assert_allclose(B_jax, B_cpu, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(dB_jax, dB_cpu, rtol=_RTOL, atol=_ATOL)


# ── Transfer-guard discipline ────────────────────────────────────────


def _device_points(points: np.ndarray) -> jax.Array:
    return jnp.asarray(points, dtype=jnp.float64)


def test_jax_paths_under_strict_transfer_guard():
    """All three kernels run cleanly under ``transfer_guard('disallow')``.

    The kernels consume device-resident ``points`` arrays (placed on the
    device before entering the guard scope) and produce device arrays.
    Any implicit host transfer inside the compiled paths would raise
    ``jax.errors.JaxRuntimeError``.
    """
    points_toroidal = _seeded_points(seed=51, count=50)
    points_poloidal = _filter_away_from_axis(
        _seeded_points(seed=52, count=80), R0=1.0, margin=0.2
    )[:50]
    points_mirror = _filter_away_from_axis(
        _seeded_points(seed=53, count=80), R0=0.0, margin=0.2
    )[:50]

    spec_t = ToroidalFieldSpec(R0=1.3, B0=0.8)
    spec_p = PoloidalFieldSpec(R0=1.0, B0=1.1, q=1.3)
    spec_m = MirrorModelSpec(B0=6.51292, gamma=0.124904, Z_m=0.98)

    # Place the JAX inputs on the device under the default (allow) guard so
    # the strict-guard region below only measures the compiled kernels.
    points_t_dev = _device_points(points_toroidal)
    points_p_dev = _device_points(points_poloidal)
    points_m_dev = _device_points(points_mirror)
    points_t_dev.block_until_ready()
    points_p_dev.block_until_ready()
    points_m_dev.block_until_ready()

    with jax.transfer_guard("disallow"):
        toroidal_B(spec_t, points_t_dev).block_until_ready()
        toroidal_dB(spec_t, points_t_dev).block_until_ready()
        toroidal_d2B(spec_t, points_t_dev).block_until_ready()
        toroidal_A(spec_t, points_t_dev).block_until_ready()
        toroidal_dA(spec_t, points_t_dev).block_until_ready()
        poloidal_B(spec_p, points_p_dev).block_until_ready()
        poloidal_dB(spec_p, points_p_dev).block_until_ready()
        mirror_B(spec_m, points_m_dev).block_until_ready()
        mirror_dB(spec_m, points_m_dev).block_until_ready()


# ── Shape and basic sanity ───────────────────────────────────────────


def test_kernel_output_shapes_and_dtypes():
    """Quick contract: kernel output shapes and float64 dtype.

    Catches accidental dtype demotions or shape regressions in any of the
    nine public kernels.
    """
    points = _seeded_points(seed=61, count=8)
    spec_t = ToroidalFieldSpec(R0=1.0, B0=1.0)
    spec_p = PoloidalFieldSpec(R0=1.0, B0=1.0, q=1.0)
    spec_m = MirrorModelSpec(B0=1.0, gamma=0.1, Z_m=0.5)

    outputs = {
        "toroidal_B": (toroidal_B(spec_t, points), (8, 3)),
        "toroidal_dB": (toroidal_dB(spec_t, points), (8, 3, 3)),
        "toroidal_d2B": (toroidal_d2B(spec_t, points), (8, 3, 3, 3)),
        "toroidal_A": (toroidal_A(spec_t, points), (8, 3)),
        "toroidal_dA": (toroidal_dA(spec_t, points), (8, 3, 3)),
        "poloidal_B": (poloidal_B(spec_p, points), (8, 3)),
        "poloidal_dB": (poloidal_dB(spec_p, points), (8, 3, 3)),
        "mirror_B": (mirror_B(spec_m, points), (8, 3)),
        "mirror_dB": (mirror_dB(spec_m, points), (8, 3, 3)),
    }
    for name, (value, expected_shape) in outputs.items():
        assert value.shape == expected_shape, f"{name} shape={value.shape}"
        assert value.dtype == jnp.float64, f"{name} dtype={value.dtype}"


def test_rejects_malformed_points():
    """``_validate_points`` rejects shapes that are not ``(N, 3)``."""
    spec = ToroidalFieldSpec(R0=1.0, B0=1.0)
    with pytest.raises(ValueError):
        toroidal_B(spec, np.zeros((5,)))
    with pytest.raises(ValueError):
        toroidal_B(spec, np.zeros((5, 2)))
