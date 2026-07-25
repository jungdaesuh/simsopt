"""Accuracy and fallback contracts for mixed dense iterative refinement."""

from typing import Literal

import numpy as np
import pytest

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from simsopt_jax.backend import set_backend
from simsopt_jax.geo.optimizers import optimizer as _optimizer


def _certificate_key(seed: int) -> jax.Array:
    return jax.random.key(seed)


def _set_test_precision(precision: Literal["mixed", "fp64"]) -> None:
    mode = "jax_gpu_parity" if jax.default_backend() == "gpu" else "jax_cpu_parity"
    set_backend(mode, precision=precision, configure_runtime=False)


def test_mixed_dense_ir_certifies_widened_factors_against_live_fp64_operator():
    certificate_matrix = jnp.asarray(
        ((4.1, 0.7), (0.7, 3.2)),
        dtype=jnp.float64,
    )
    proposal_matrix = jnp.asarray(certificate_matrix, dtype=jnp.float32)
    right_hand_side = jnp.asarray((1.23456789, 2.34567891), dtype=jnp.float64)

    solution, status, telemetry = (
        _optimizer._solve_mixed_dense_ir_operator_with_telemetry(
            lambda vector: proposal_matrix @ vector,
            lambda vector: certificate_matrix @ vector,
            right_hand_side,
            tol=1.0e-12,
            proposal_dtype=np.float32,
            certificate_sweep_dtype=np.float64,
            certificate_probe_key=_certificate_key(11),
        )
    )

    assert bool(status.success)
    assert bool(telemetry.proposal_trusted)
    assert int(telemetry.fp64_rebuild_count) == 0
    assert int(telemetry.proposal_factorization_dtype_bits) == 32
    assert int(telemetry.factor_application_dtype_bits) == 64
    assert int(telemetry.residual_dtype_bits) == 64
    assert int(telemetry.certificate_sweep_dtype_bits) == 64
    assert solution.dtype == jnp.dtype(np.float64)
    np.testing.assert_allclose(
        np.asarray(solution),
        np.asarray(jnp.linalg.solve(certificate_matrix, right_hand_side)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_mixed_dense_ir_rebuilds_the_live_fp64_operator_once_on_proposal_miss():
    certificate_matrix = jnp.asarray(
        ((4.0, 1.0), (1.0, 3.0)),
        dtype=jnp.float64,
    )
    singular_proposal_matrix = jnp.ones((2, 2), dtype=jnp.float32)
    right_hand_side = jnp.asarray((1.0, 2.0), dtype=jnp.float64)

    solution, status, telemetry = (
        _optimizer._solve_mixed_dense_ir_operator_with_telemetry(
            lambda vector: singular_proposal_matrix @ vector,
            lambda vector: certificate_matrix @ vector,
            right_hand_side,
            tol=1.0e-12,
            proposal_dtype=np.float32,
            certificate_sweep_dtype=np.float64,
            certificate_probe_key=_certificate_key(12),
        )
    )

    assert bool(status.success)
    assert not bool(telemetry.proposal_trusted)
    assert int(telemetry.fp64_rebuild_count) == 1
    assert bool(telemetry.fallback.attempted)
    assert bool(telemetry.fallback.success)
    assert int(telemetry.fallback.factorization_dtype_bits) == 64
    np.testing.assert_allclose(
        np.asarray(solution),
        np.asarray(jnp.linalg.solve(certificate_matrix, right_hand_side)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_mixed_dense_ir_jit_keeps_fallback_branch_shapes_identical():
    certificate_matrix = jnp.eye(2, dtype=jnp.float64)
    right_hand_side = jnp.asarray((1.0, 2.0), dtype=jnp.float64)

    @jax.jit
    def solve(proposal_scale: jax.Array):
        proposal_matrix = proposal_scale * jnp.eye(2, dtype=jnp.float32)
        _, _, telemetry = _optimizer._solve_mixed_dense_ir_operator_with_telemetry(
            lambda vector: proposal_matrix @ vector,
            lambda vector: certificate_matrix @ vector,
            right_hand_side,
            tol=1.0e-12,
            proposal_dtype=np.float32,
            certificate_sweep_dtype=np.float64,
            certificate_probe_key=_certificate_key(14),
        )
        return telemetry.fallback

    inactive = solve(jnp.asarray(1.0, dtype=jnp.float32))
    active = solve(jnp.asarray(0.0, dtype=jnp.float32))
    inactive_leaves, inactive_tree = jax.tree.flatten(inactive)
    active_leaves, active_tree = jax.tree.flatten(active)

    assert inactive_tree == active_tree
    assert [leaf.shape for leaf in inactive_leaves] == [
        leaf.shape for leaf in active_leaves
    ]
    assert not bool(inactive.attempted)
    assert bool(active.attempted)
    assert bool(active.success)


def test_mixed_dense_ir_fails_closed_for_catastrophic_forward_error():
    certificate_matrix = jnp.diag(
        jnp.asarray((1.0 - 5.0e-7, 1.0e-30), dtype=jnp.float64)
    )
    proposal_matrix = jnp.diag(jnp.asarray((1.0, 1.0e-7), dtype=jnp.float32))
    right_hand_side = jnp.asarray((1.0, 1.0e-20), dtype=jnp.float64)

    solution, status, telemetry = (
        _optimizer._solve_mixed_dense_ir_operator_with_telemetry(
            lambda vector: proposal_matrix @ vector,
            lambda vector: certificate_matrix @ vector,
            right_hand_side,
            tol=1.0e-12,
            proposal_dtype=np.float32,
            certificate_sweep_dtype=np.float64,
            certificate_probe_key=_certificate_key(13),
        )
    )

    assert not bool(status.success)
    assert not bool(telemetry.proposal_trusted)
    assert int(telemetry.fp64_rebuild_count) == 1
    assert np.all(np.isfinite(np.asarray(solution)))


def test_dense_ir_contraction_certificate_uses_the_caller_key():
    dimension = _optimizer._MIXED_DENSE_IR_CONTRACTION_PROBE_COUNT + 1
    blind_key = _certificate_key(15)
    blind_probes = jax.random.normal(
        blind_key,
        shape=(_optimizer._MIXED_DENSE_IR_CONTRACTION_PROBE_COUNT, dimension),
        dtype=jnp.float64,
    )
    _, _, right_vectors = np.linalg.svd(
        np.asarray(blind_probes),
        full_matrices=True,
    )
    hidden_direction = jnp.asarray(right_vectors[-1], dtype=jnp.float64)
    contraction = (1.0 - 1.0e-14) * jnp.outer(
        hidden_direction,
        hidden_direction,
    )
    certificate_matrix = jnp.eye(dimension, dtype=jnp.float64) - contraction
    certificate_apply_factors = jsp_linalg.lu_factor(
        jnp.eye(dimension, dtype=jnp.float64)
    )
    right_hand_side = jnp.ones((dimension,), dtype=jnp.float64)

    def certificate_matvec(vector: jax.Array) -> jax.Array:
        return certificate_matrix @ vector

    blind_upper, _ = _optimizer._mixed_dense_ir_contraction_operator_norm_upper(
        certificate_matvec,
        certificate_apply_factors,
        right_hand_side,
        certificate_probe_key=blind_key,
    )
    fresh_upper, _ = _optimizer._mixed_dense_ir_contraction_operator_norm_upper(
        certificate_matvec,
        certificate_apply_factors,
        right_hand_side,
        certificate_probe_key=_certificate_key(16),
    )

    assert float(blind_upper) < 1.0e-10
    assert float(fresh_upper) >= 1.0


def test_bounded_mixed_newton_materializes_live_fp64_public_hessian():
    _set_test_precision("mixed")
    matrix = jnp.asarray(
        ((2.00000003, 0.50000007), (0.50000007, 3.00000011)),
        dtype=jnp.float64,
    )
    right_hand_side = jnp.asarray((1.0, 2.0), dtype=jnp.float64)

    def objective(state: jax.Array) -> jax.Array:
        typed_matrix = jnp.asarray(matrix, dtype=state.dtype)
        typed_right_hand_side = jnp.asarray(right_hand_side, dtype=state.dtype)
        return 0.5 * state @ typed_matrix @ state - typed_right_hand_side @ state

    result = _optimizer.bounded_mixed_newton_polish_traceable(
        objective,
        jnp.zeros(2, dtype=jnp.float64),
        tol=1.0e-12,
        stab=0.0,
        materialize_hessian=True,
    )

    assert bool(result["success"])
    assert bool(result["bounded_newton_fp64_certification_covered"])
    assert int(result["bounded_newton_fp64_value_dtype_bits"]) == 64
    assert int(result["bounded_newton_fp64_gradient_dtype_bits"]) == 64
    assert result["grad"].dtype == jnp.dtype(np.float64)
    assert result["hessian"].dtype == jnp.dtype(np.float64)
    np.testing.assert_allclose(
        np.asarray(result["x"]),
        np.asarray(jnp.linalg.solve(matrix, right_hand_side)),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        np.asarray(result["hessian"]),
        np.asarray(matrix),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_default_fp64_bounded_newton_is_independent_of_the_general_runner(
    monkeypatch: pytest.MonkeyPatch,
):
    _set_test_precision("fp64")
    monkeypatch.setattr(
        _optimizer,
        "_make_traceable_newton_polish_runner",
        lambda *args, **kwargs: pytest.fail("bounded solver called the general runner"),
    )

    def objective(state: jax.Array) -> jax.Array:
        return jnp.vdot(state, state).real

    result = _optimizer.bounded_mixed_newton_polish_traceable(
        objective,
        jnp.asarray((1.0, -1.0), dtype=jnp.float64),
        tol=1.0e-12,
    )

    assert bool(result["success"])
    assert not bool(result["bounded_newton_proposal_attempted"])
    np.testing.assert_array_equal(
        np.asarray(result["bounded_newton_factorization_dtype_bits_trace"]),
        np.asarray((64, 0, 0), dtype=np.int32),
    )


def test_newton_candidate_acceptance_uses_the_shared_armijo_merit_owner():
    accepted, candidate_norm = _optimizer._newton_candidate_status(
        jnp.asarray((0.5,), dtype=jnp.float64),
        jnp.asarray(0.5, dtype=jnp.float64),
        jnp.asarray((2.0,), dtype=jnp.float64),
        alpha=jnp.asarray(1.0, dtype=jnp.float64),
        current_val=jnp.asarray(1.0, dtype=jnp.float64),
        current_grad=jnp.asarray((1.0,), dtype=jnp.float64),
        current_norm=jnp.asarray(1.0, dtype=jnp.float64),
        dx=jnp.asarray((1.0,), dtype=jnp.float64),
    )

    assert float(candidate_norm) > 1.0
    assert bool(accepted)
