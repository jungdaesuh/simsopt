"""Item 23 parity tests for ``ScalarPotentialRZMagneticFieldJAX``."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("sympy")

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt._core.json import GSONDecoder, GSONEncoder, SIMSON
from simsopt.field import (
    ScalarPotentialRZMagneticField,
    ScalarPotentialRZMagneticFieldJAX,
)


_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]

_POTENTIALS = (
    "2*phi",
    "0.1*phi+0.2*R*Z+0.3*Z*phi+0.4*R**2+0.5*Z**2",
    "Z/(R*R + Z*Z)**(3/2)",
    "R*cos(phi)",
    "Z**2 + R*sin(phi)",
    "sin(R) * cos(Z) * sin(phi)",
    "Piecewise((R**2 + 0.1*phi, R > 1), (Z**2 + 0.2*phi, True))",
    "sec(R + 2) + csc(Z + 2) + cot(phi + 2)",
)


def _points(seed: int = 23, count: int = 48) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.65, 1.8, size=count)
    y = rng.uniform(-0.7, 0.7, size=count)
    z = rng.uniform(-0.45, 0.55, size=count)
    return np.ascontiguousarray(np.stack((x, y, z), axis=1))


@pytest.mark.parametrize("phi_str", _POTENTIALS)
def test_scalar_potential_rz_jax_matches_cpu_B_and_dB(phi_str: str) -> None:
    points = _points(seed=len(phi_str))
    cpu = ScalarPotentialRZMagneticField(phi_str)
    jax_field = ScalarPotentialRZMagneticFieldJAX(phi_str)
    cpu.set_points_cart(points)
    jax_field.set_points_cart(points)

    np.testing.assert_allclose(
        np.asarray(jax_field.B()),
        np.asarray(cpu.B()),
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(jax_field.dB_by_dX()),
        np.asarray(cpu.dB_by_dX()),
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_scalar_potential_rz_jax_serialization_roundtrip() -> None:
    points = _points(seed=42, count=12)
    field = ScalarPotentialRZMagneticFieldJAX("Z**2 + R*sin(phi)")
    field.set_points_cart(points)
    field_json = json.dumps(SIMSON(field), cls=GSONEncoder)
    regenerated = json.loads(field_json, cls=GSONDecoder)

    assert isinstance(regenerated, ScalarPotentialRZMagneticFieldJAX)
    np.testing.assert_allclose(
        np.asarray(regenerated.B()),
        np.asarray(field.B()),
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_scalar_potential_rz_jax_kernels_run_under_strict_transfer_guard() -> None:
    device_points = jnp.asarray(_points(seed=71, count=20), dtype=jnp.float64)
    device_points.block_until_ready()
    fields = tuple(
        ScalarPotentialRZMagneticFieldJAX(phi_str)
        for phi_str in (
            "sin(R) * cos(Z) * sin(phi)",
            "Piecewise((R**2 + 0.1*phi, R > 1), (Z**2 + 0.2*phi, True))",
        )
    )
    for field in fields:
        field._B_kernel(device_points).block_until_ready()
        field._dB_kernel(device_points).block_until_ready()

    with jax.transfer_guard("disallow"):
        for field in fields:
            field._B_kernel(device_points).block_until_ready()
            field._dB_kernel(device_points).block_until_ready()


@pytest.mark.parametrize(
    "phi_str",
    (
        "R + q_unknown",
        "Piecewise((R, R > 1))",
        "Abs(R)",
        "Max(R, Z)",
        "Min(R, Z)",
        "Heaviside(R)",
        "sign(R)",
        "floor(R)",
        "ceiling(R)",
        "Mod(R, Z)",
        "I*R",
    ),
)
def test_scalar_potential_rz_jax_rejects_unsupported_expressions(phi_str: str) -> None:
    with pytest.raises(NotImplementedError):
        ScalarPotentialRZMagneticFieldJAX(phi_str)
