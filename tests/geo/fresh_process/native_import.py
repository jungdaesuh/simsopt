"""Fresh-process child: what importing ``simsopt.geo`` does to JAX.

Usage: ``native_import.py SCENARIO``. The parent exports the scenario's JAX
variables before this process starts; the ``simsopt.geo`` import below is the
event under test, so this script imports nothing from ``simsopt_jax``.
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

# The state the ``simsopt.geo`` import has to change, read before that import,
# which therefore follows it at module level.
_X64_BEFORE_NATIVE_IMPORT = jax.config.values["jax_enable_x64"]

from simsopt.geo import CurveLength, create_equally_spaced_curves


def platform_selection() -> None:
    """``JAX_PLATFORMS=cpu``, ``JAX_PLATFORM_NAME=cpu`` or neither: the import
    keeps the CPU platform name, default backend and x64."""
    explicit_platform = (
        "JAX_PLATFORMS" in os.environ or "JAX_PLATFORM_NAME" in os.environ
    )
    assert jax.config.values["jax_platform_name"] == (
        "cpu" if not explicit_platform or "JAX_PLATFORM_NAME" in os.environ else ""
    )
    assert jax.default_backend() == "cpu"
    assert jax.config.values["jax_enable_x64"]


def x64() -> None:
    """``JAX_PLATFORMS=cpu`` and a disabling ``JAX_ENABLE_X64``: the import enables x64."""
    assert not _X64_BEFORE_NATIVE_IMPORT, "JAX_ENABLE_X64 did not disable x64"
    assert jax.default_backend() == "cpu"
    assert jax.config.values["jax_enable_x64"]


def jit_device_policy() -> None:
    """``JAX_PLATFORMS=cuda,cpu``: CurveLength's jitted value and gradient land on the CPU device."""
    assert jax.default_backend() == "gpu"
    cpu = {jax.devices("cpu")[0]}
    curve = create_equally_spaced_curves(
        1, 1, False, R0=1.0, R1=0.5, order=2, numquadpoints=32
    )[0]
    length = CurveLength(curve)
    arclength = curve.incremental_arclength()

    value = length.J()
    arclength_gradient = length.dJ_dl(arclength)
    assert value.devices() == cpu
    assert arclength_gradient.devices() == cpu
    np.testing.assert_allclose(value, np.mean(arclength), rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(
        arclength_gradient, np.full_like(arclength, 1.0 / arclength.size)
    )
    assert {d.platform for d in jnp.ones(2).devices()} == {"gpu"}


_SCENARIOS = {
    "platform_selection": platform_selection,
    "x64": x64,
    "jit_device_policy": jit_device_policy,
}


if __name__ == "__main__":
    _SCENARIOS[sys.argv[1]]()
