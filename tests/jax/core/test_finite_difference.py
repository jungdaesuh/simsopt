"""N6 finite-difference tests.

Lane key: derivative_heavy via explicit finite-difference/JAX AD parity.
"""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from simsopt._core.util import finite_difference_steps
from simsopt_jax.core._finite_difference import (
    forward_jacobian_shard_map_columns,
    forward_jacobian_shard_map,
    forward_jacobian_vmap,
)

REPO_SRC = str(Path(__file__).resolve().parents[3] / "src")


def test_forward_jacobian_vmap_matches_jacfwd_for_linear_residual():
    """Oracle: exact Jacobian of a linear residual is its coefficient matrix."""
    matrix = jnp.array(
        (
            (1.5, -2.0, 0.25),
            (0.0, 3.0, -4.0),
        ),
        dtype=jnp.float64,
    )

    def residual(x):
        return matrix @ x

    x0 = jnp.array([0.4, -1.2, 2.0], dtype=jnp.float64)
    jacobian = forward_jacobian_vmap(residual, x0, abs_step=2.0**-30)

    np.testing.assert_allclose(np.asarray(jacobian), np.asarray(matrix), rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(jacobian),
        np.asarray(jax.jacfwd(residual)(x0)),
        rtol=1e-12,
    )


def test_forward_jacobian_vmap_centered_matches_nonlinear_oracle():
    """Oracle: analytic Jacobian of a coupled nonlinear residual."""

    def residual(x):
        return jnp.array(
            (
                x[0] * x[1] + jnp.sin(x[2]),
                x[0] ** 2 - 0.5 * x[1] + x[2],
            ),
            dtype=x.dtype,
        )

    x0 = jnp.array([0.7, -1.3, 0.2], dtype=jnp.float64)
    expected = jax.jacfwd(residual)(x0)
    jacobian = forward_jacobian_vmap(
        residual,
        x0,
        abs_step=2.0**-18,
        diff_method="centered",
    )

    np.testing.assert_allclose(
        np.asarray(jacobian),
        np.asarray(expected),
        rtol=1e-10,
        atol=1e-10,
    )


def test_forward_jacobian_vmap_uses_simsopt_finite_difference_step_contract():
    """Observable nonlinear derivative reflects SIMSOPT's max-step contract."""
    x0 = jnp.array([0.1165368, -0.92130271, 1.09012617], dtype=jnp.float64)
    abs_step = 0.08
    rel_step = 0.1
    steps = finite_difference_steps(
        np.asarray(x0), abs_step=abs_step, rel_step=rel_step
    )

    jacobian = forward_jacobian_vmap(
        lambda x: x * x,
        x0,
        abs_step=abs_step,
        rel_step=rel_step,
    )
    expected = np.diag(2.0 * np.asarray(x0) + steps)

    np.testing.assert_allclose(np.asarray(jacobian), expected, rtol=1e-12)


def test_forward_jacobian_vmap_rejects_zero_finite_difference_step():
    """Zero finite-difference steps fail loudly instead of producing NaNs."""
    x0 = jnp.array([1.0, 0.0], dtype=jnp.float64)

    with np.testing.assert_raises(ValueError):
        forward_jacobian_vmap(lambda x: x * x, x0, abs_step=0.0, rel_step=1.0e-3)


def test_forward_jacobian_vmap_rejects_materialized_zero_step():
    """Positive inputs that underflow to zero still follow the SIMSOPT guard."""
    x0 = jnp.array([1.0], dtype=jnp.float32)

    with np.testing.assert_raises(ValueError):
        forward_jacobian_vmap(lambda x: x * x, x0, abs_step=1.0e-46, rel_step=0.0)


def test_forward_jacobian_vmap_jit_rejects_materialized_zero_abs_step():
    """Compiled finite differences fail closed when zero steps cannot be excluded."""
    compiled = jax.jit(
        lambda x: forward_jacobian_vmap(
            lambda y: y * y,
            x,
            abs_step=1.0e-46,
            rel_step=1.0e-3,
        )
    )

    with np.testing.assert_raises(ValueError):
        compiled(jnp.array([0.0], dtype=jnp.float32)).block_until_ready()


def test_forward_jacobian_shard_map_matches_vmap_with_explicit_mesh():
    """Oracle: explicit-mesh shard_map route equals the single-device vmap route."""
    matrix = jnp.array(
        (
            (2.0, -1.0, 0.5),
            (0.25, 1.5, -3.0),
        ),
        dtype=jnp.float64,
    )

    def residual(x):
        return matrix @ x

    x0 = jnp.array([1.0, -0.5, 0.25], dtype=jnp.float64)
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("dof",))
    expected = forward_jacobian_vmap(residual, x0, abs_step=2.0**-30)
    actual = forward_jacobian_shard_map(
        residual,
        x0,
        abs_step=2.0**-30,
        mesh=mesh,
    )

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-12)


def test_forward_jacobian_shard_map_columns_matches_selected_vmap_columns():
    """Explicit column selection computes only the requested Jacobian columns."""
    matrix = jnp.array(
        (
            (2.0, -1.0, 0.5),
            (0.25, 1.5, -3.0),
        ),
        dtype=jnp.float64,
    )

    def residual(x):
        return matrix @ x

    x0 = jnp.array([1.0, -0.5, 0.25], dtype=jnp.float64)
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("dof",))
    columns = jnp.array([0, 2])
    actual = forward_jacobian_shard_map_columns(
        residual,
        x0,
        columns,
        abs_step=2.0**-30,
        mesh=mesh,
    )

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(matrix[:, np.asarray(columns)]),
        rtol=1e-12,
    )


def test_forward_jacobian_shard_map_transfer_guard_clean():
    """Compiled sharded finite-difference route has no implicit host transfers."""
    matrix = jnp.array(((1.0, 2.0), (-3.0, 0.5)), dtype=jnp.float64)
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("dof",))
    replicated = NamedSharding(mesh, P())
    matrix = jax.device_put(matrix, replicated)

    def residual(x):
        return matrix @ x

    compiled = jax.jit(
        lambda x: forward_jacobian_shard_map(
            residual,
            x,
            abs_step=2.0**-30,
            mesh=mesh,
        )
    )
    x0 = jax.device_put(jnp.array([0.2, -0.3], dtype=jnp.float64), replicated)
    compiled(x0).block_until_ready()

    with jax.transfer_guard("disallow"):
        jacobian = compiled(x0)
        jacobian.block_until_ready()

    np.testing.assert_allclose(np.asarray(jacobian), np.asarray(matrix), rtol=1e-12)


def test_forward_jacobian_shard_map_jit_rejects_materialized_zero_abs_step():
    """Compiled sharded finite differences reject zero materialized abs steps."""
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("dof",))
    compiled = jax.jit(
        lambda x: forward_jacobian_shard_map(
            lambda y: y * y,
            x,
            abs_step=1.0e-46,
            rel_step=1.0e-3,
            mesh=mesh,
        )
    )

    with np.testing.assert_raises(ValueError):
        compiled(jnp.array([0.0], dtype=jnp.float32)).block_until_ready()


def test_forward_jacobian_shard_map_fake_two_device_transfer_guard_clean():
    """Fake-CPU two-device mesh proves no implicit mesh transfers in shard_map."""
    code = r"""
import sys
sys.path.insert(0, REPO_SRC)

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from simsopt_jax.core._finite_difference import forward_jacobian_shard_map

mesh = Mesh(np.asarray(jax.devices()[:2]), ("dof",))
replicated = NamedSharding(mesh, P())
matrix = jax.device_put(
    jnp.array(((1.0, 2.0, 3.0), (-1.0, 0.5, 4.0)), dtype=jnp.float64),
    replicated,
)

def residual(x):
    return matrix @ x

compiled = jax.jit(
    lambda x: forward_jacobian_shard_map(
        residual,
        x,
        abs_step=2.0**-30,
        mesh=mesh,
    )
)
x0 = jax.device_put(jnp.array([0.2, -0.3, 0.4], dtype=jnp.float64), replicated)
compiled(x0).block_until_ready()

with jax.transfer_guard("disallow"):
    jacobian = compiled(x0)
    jacobian.block_until_ready()

np.testing.assert_allclose(
    np.asarray(jacobian),
    np.asarray(matrix),
    rtol=5e-7,
    atol=5e-7,
)
"""
    env = {
        "HOME": os.environ["HOME"],
        "JAX_ENABLE_X64": "True",
        "JAX_PLATFORM_NAME": "cpu",
        "PATH": os.environ["PATH"],
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": REPO_SRC,
        "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
    }
    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(code).replace("REPO_SRC", repr(REPO_SRC)),
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
