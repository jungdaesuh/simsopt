"""MPI assembly helpers for traceable JAX least-squares problems."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from scipy.optimize import least_squares

from simsopt.jax_core._finite_difference import forward_jacobian_shard_map_columns
from simsopt.jax_core._math_utils import runtime_device_put as _runtime_device_put
from simsopt.solve.serial_jax import TraceableLeastSquaresProblem
from simsopt.util.mpi import MPI, MpiPartition

__all__ = [
    "least_squares_mpi_solve_jax",
    "traceable_least_squares_mpi_jacobian",
]

CALCULATE_JAX_JAC = 1
STOP = 0


def _owned_columns(dof_count: int, mpi: MpiPartition) -> np.ndarray:
    return np.arange(mpi.rank_leaders, dof_count, mpi.nprocs_leaders, dtype=np.int32)


def traceable_least_squares_mpi_jacobian(
    prob: TraceableLeastSquaresProblem,
    mpi: MpiPartition,
    x: jax.Array,
    *,
    mesh: Mesh,
    abs_step: float = 1.0e-7,
    rel_step: float = 0.0,
    diff_method: str = "forward",
) -> jax.Array:
    """Assemble a traceable residual Jacobian from leader-owned JAX column blocks."""
    if MPI is None:
        raise RuntimeError("traceable_least_squares_mpi_jacobian requires mpi4py.")
    if not isinstance(prob, TraceableLeastSquaresProblem):
        raise TypeError(
            "traceable_least_squares_mpi_jacobian requires TraceableLeastSquaresProblem."
        )

    leader_x = (
        np.asarray(x, dtype=np.asarray(prob.x).dtype) if mpi.proc0_world else None
    )
    if mpi.proc0_groups:
        if not mpi.proc0_world:
            leader_x = np.empty_like(np.asarray(prob.x))
        mpi.comm_leaders.Bcast(leader_x, root=0)
    group_x = mpi.comm_groups.bcast(leader_x, root=0)
    prob.x = _runtime_device_put(group_x)

    local_columns_packet = None
    if mpi.proc0_groups:
        dof_count = int(jnp.ravel(prob.x).size)
        columns = _owned_columns(dof_count, mpi)
        if columns.size:
            local_columns = forward_jacobian_shard_map_columns(
                prob.residuals,
                prob.x,
                _runtime_device_put(columns),
                abs_step,
                rel_step,
                diff_method,
                mesh=mesh,
            )
            with jax.transfer_guard_device_to_host("allow"):
                local_columns_host = np.asarray(jax.device_get(local_columns))
            local_columns_packet = (columns, local_columns_host)
        else:
            local_columns_packet = (
                columns,
                np.empty((int(prob.residuals(prob.x).size), 0)),
            )

        gathered_columns = mpi.comm_leaders.gather(local_columns_packet, root=0)
        if mpi.proc0_world:
            residual_count = int(prob.residuals(prob.x).size)
            assembled = np.empty((residual_count, dof_count))
            for columns, column_values in gathered_columns:
                assembled[:, columns] = column_values
        else:
            assembled = None
    else:
        assembled = None

    if mpi.proc0_world:
        return _runtime_device_put(assembled)
    return jnp.empty((0, 0), dtype=prob.x.dtype)


def least_squares_mpi_solve_jax(
    prob: TraceableLeastSquaresProblem,
    mpi: MpiPartition,
    *,
    mesh: Mesh | None = None,
    abs_step: float = 1.0e-7,
    rel_step: float = 0.0,
    diff_method: str = "forward",
    max_steps: int = 256,
    **kwargs,
) -> None:
    """Solve with SciPy on rank 0 and MPI/JAX finite-difference Jacobians."""
    if MPI is None:
        raise RuntimeError("least_squares_mpi_solve_jax requires mpi4py.")
    if not isinstance(prob, TraceableLeastSquaresProblem):
        raise TypeError(
            "least_squares_mpi_solve_jax requires TraceableLeastSquaresProblem."
        )
    mesh = Mesh(np.asarray(jax.local_devices()[:1]), ("dof",)) if mesh is None else mesh

    if not mpi.proc0_world:
        while True:
            command = mpi.comm_world.bcast(None, root=0)
            if command is None or command == STOP:
                break
            if command == CALCULATE_JAX_JAC:
                traceable_least_squares_mpi_jacobian(
                    prob,
                    mpi,
                    prob.x,
                    mesh=mesh,
                    abs_step=abs_step,
                    rel_step=rel_step,
                    diff_method=diff_method,
                )
        final_x = np.empty_like(np.asarray(prob.x))
        mpi.comm_world.Bcast(final_x, root=0)
        prob.x = _runtime_device_put(final_x)
        return

    def residuals(x):
        with jax.transfer_guard_device_to_host("allow"):
            return np.asarray(jax.device_get(prob.residuals(_runtime_device_put(x))))

    def jacobian(x):
        mpi.comm_world.bcast(CALCULATE_JAX_JAC, root=0)
        jac = traceable_least_squares_mpi_jacobian(
            prob,
            mpi,
            _runtime_device_put(x),
            mesh=mesh,
            abs_step=abs_step,
            rel_step=rel_step,
            diff_method=diff_method,
        )
        with jax.transfer_guard_device_to_host("allow"):
            return np.asarray(jax.device_get(jac))

    result = least_squares(
        residuals,
        np.asarray(prob.x),
        jac=jacobian,
        max_nfev=max_steps,
        **kwargs,
    )
    final_x = np.asarray(result.x)
    mpi.comm_world.bcast(STOP, root=0)
    mpi.comm_world.Bcast(final_x, root=0)
    prob.x = _runtime_device_put(final_x)
