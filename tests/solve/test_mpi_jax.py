"""N6 MPI/JAX solve tests.

Lane key: derivative_heavy for leader-owned finite-difference Jacobian blocks.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from monty.tempfile import ScratchDir
import pytest

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from simsopt_jax_adapters.solve.mpi import (
    least_squares_mpi_solve_jax,
    traceable_least_squares_mpi_jacobian,
)
from simsopt_jax.solve.serial import (
    TraceableLeastSquaresProblem,
)

if MPI is not None:
    from simsopt.util.mpi import MpiPartition


def _quadratic_residual(x):
    return jnp.array((x[0] - 1.0, x[0] * x[0] - x[1]), dtype=x.dtype)


def _linear_residual(matrix):
    def residual(x):
        return matrix @ x

    return residual


@pytest.mark.skipif(MPI is None, reason="Requires mpi4py")
def test_traceable_least_squares_mpi_jacobian_matches_jacfwd():
    """Leader-owned JAX column blocks assemble the same Jacobian as AD."""
    mpi = MpiPartition(ngroups=min(2, MPI.COMM_WORLD.Get_size()))
    matrix = jnp.array(
        (
            (1.0, 2.0, -0.5),
            (-3.0, 0.25, 4.0),
        ),
        dtype=jnp.float64,
    )
    prob = TraceableLeastSquaresProblem(
        residual_fn=_linear_residual(matrix),
        x=jnp.array([0.5, -0.25, 1.5], dtype=jnp.float64),
    )
    mesh = Mesh(np.asarray(jax.local_devices()[:1]), ("dof",))

    actual = traceable_least_squares_mpi_jacobian(
        prob, mpi, prob.x, mesh=mesh, abs_step=2.0**-30
    )
    expected = jax.jacfwd(prob.residuals)(prob.x)

    if mpi.proc0_world:
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-10)
    else:
        assert actual.shape == (0, 0)


@pytest.mark.skipif(MPI is None, reason="Requires mpi4py")
def test_traceable_least_squares_mpi_jacobian_transfer_guard_clean():
    """MPI column assembly has no implicit host-to-device transfers."""
    mpi = MpiPartition(ngroups=1)
    matrix = jax.device_put(
        np.asarray(
            (
                (1.0, 2.0, -0.5),
                (-3.0, 0.25, 4.0),
            ),
            dtype=np.float64,
        )
    )
    prob = TraceableLeastSquaresProblem(
        residual_fn=_linear_residual(matrix),
        x=jax.device_put(np.asarray([0.5, -0.25, 1.5], dtype=np.float64)),
    )
    mesh = Mesh(np.asarray(jax.local_devices()[:1]), ("dof",))

    with jax.transfer_guard_host_to_device("disallow"):
        actual = traceable_least_squares_mpi_jacobian(
            prob,
            mpi,
            prob.x,
            mesh=mesh,
            abs_step=2.0**-30,
        )
        actual.block_until_ready()

    if mpi.proc0_world:
        np.testing.assert_allclose(np.asarray(actual), np.asarray(matrix), rtol=1e-10)
    else:
        assert actual.shape == (0, 0)


@pytest.mark.skipif(MPI is None, reason="Requires mpi4py")
def test_least_squares_mpi_solve_jax_reaches_traceable_quadratic_optimum():
    """MPI solve uses its JAX Jacobian path and broadcasts the optimum."""
    mpi = MpiPartition(ngroups=min(2, MPI.COMM_WORLD.Get_size()))
    with ScratchDir("."):
        mpi_prob = TraceableLeastSquaresProblem(
            residual_fn=_quadratic_residual,
            x=jnp.array([0.0, 0.0], dtype=jnp.float64),
        )

        least_squares_mpi_solve_jax(mpi_prob, mpi, max_steps=64)

        np.testing.assert_allclose(
            np.asarray(mpi_prob.x), np.array([1.0, 1.0]), rtol=1e-10
        )
        np.testing.assert_allclose(
            np.asarray(mpi_prob.residuals()), np.zeros(2), atol=1e-10
        )
