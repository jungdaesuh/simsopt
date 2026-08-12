"""Parity and device-telemetry gates for source-owned incremental GMRES."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from simsopt_jax.geo.optimizers import linear_solve as _linear_solve


def _run_pair(matrix, rhs, *, restart, maxiter, tol):
    def matvec(vector):
        return matrix @ vector

    reference, reference_info = _linear_solve._run_operator_gmres(
        matvec,
        rhs,
        tol=tol,
        restart=restart,
        maxiter=maxiter,
    )
    counted, counted_info, telemetry = (
        _linear_solve._run_operator_gmres_counted_incremental(
            matvec,
            rhs,
            tol=tol,
            restart=restart,
            maxiter=maxiter,
        )
    )
    np.testing.assert_array_equal(np.asarray(counted), np.asarray(reference))
    np.testing.assert_array_equal(np.asarray(counted_info), np.asarray(reference_info))
    assert telemetry.linear_operator_application_count.dtype == jnp.int32
    return telemetry


def test_counted_incremental_gmres_matches_early_stop() -> None:
    telemetry = _run_pair(
        jnp.eye(3, dtype=jnp.float64),
        jnp.asarray([1.0, -2.0, 3.0], dtype=jnp.float64),
        restart=3,
        maxiter=2,
        tol=1.0e-12,
    )

    assert int(telemetry.linear_operator_application_count) == 3


def test_counted_incremental_gmres_matches_immediate_zero_rhs() -> None:
    telemetry = _run_pair(
        jnp.eye(3, dtype=jnp.float64),
        jnp.zeros(3, dtype=jnp.float64),
        restart=3,
        maxiter=2,
        tol=1.0e-12,
    )

    assert int(telemetry.linear_operator_application_count) == 1


def test_counted_incremental_gmres_matches_nonsymmetric_operator() -> None:
    telemetry = _run_pair(
        jnp.asarray(
            [[4.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 2.0]],
            dtype=jnp.float64,
        ),
        jnp.asarray([1.0, -2.0, 3.0], dtype=jnp.float64),
        restart=2,
        maxiter=3,
        tol=1.0e-12,
    )

    assert int(telemetry.linear_operator_application_count) == 10


def test_counted_incremental_gmres_matches_restart_nonconvergence() -> None:
    telemetry = _run_pair(
        jnp.diag(jnp.arange(1.0, 7.0, dtype=jnp.float64)),
        jnp.ones(6, dtype=jnp.float64),
        restart=2,
        maxiter=2,
        tol=1.0e-16,
    )

    assert int(telemetry.linear_operator_application_count) == 7


def test_counted_incremental_gmres_matches_ill_scaled_fp64_operator() -> None:
    _run_pair(
        jnp.asarray(
            [
                [1.0e-8, 1.0, 0.0],
                [0.0, 1.0e8, 1.0],
                [1.0, 0.0, 3.0],
            ],
            dtype=jnp.float64,
        ),
        jnp.asarray([1.0e-8, -1.0e8, 3.0], dtype=jnp.float64),
        restart=3,
        maxiter=2,
        tol=1.0e-12,
    )


def test_counted_incremental_gmres_matches_complex_operator() -> None:
    _run_pair(
        jnp.asarray(
            [[3.0 + 1.0j, 2.0 - 0.5j], [-1.0 + 0.25j, 4.0 - 2.0j]],
            dtype=jnp.complex128,
        ),
        jnp.asarray([1.0 - 2.0j, 3.0 + 0.5j], dtype=jnp.complex128),
        restart=2,
        maxiter=2,
        tol=1.0e-12,
    )


def test_counted_incremental_gmres_returns_counts_through_outer_jit() -> None:
    matrix = jnp.asarray(
        [[3.0, 2.0], [-1.0, 4.0]],
        dtype=jnp.float64,
    )

    @jax.jit
    def solve(rhs):
        return _linear_solve._run_operator_gmres_counted_incremental(
            lambda vector: matrix @ vector,
            rhs,
            tol=1.0e-12,
            restart=2,
            maxiter=2,
        )

    _, _, telemetry = solve(jnp.asarray([1.0, 2.0], dtype=jnp.float64))
    assert telemetry.linear_operator_application_count.shape == ()
    assert telemetry.linear_operator_application_count.dtype == jnp.int32
