from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt_jax.geo.optimizers import optimizer as _optimizer


def _affine_residual(values: jax.Array) -> jax.Array:
    matrix = jnp.asarray([[1.5, -0.25], [0.75, 2.0]], dtype=values.dtype)
    target = jnp.asarray([0.5, -1.25], dtype=values.dtype)
    return matrix @ values - target


def test_c0_contract_preserves_the_production_solver_identity() -> None:
    contract = _optimizer.make_traceable_exact_newton_variant_contract("C0")

    assert contract.variant == "C0"
    assert contract.solver is _optimizer.newton_exact_traceable
    assert contract.factorization_backend == "operator-gmres"


def test_c2_cached_maker_builds_once_per_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c2_builds: list[tuple[int, float]] = []
    c2_runner = object()

    def build_c2(_residual_ref, maxiter: int, tol: float):
        c2_builds.append((maxiter, tol))
        return c2_runner

    monkeypatch.setattr(
        _optimizer,
        "_build_traceable_dense_direct_exact_newton_c2_runner",
        build_c2,
    )
    _optimizer._TRACEABLE_DENSE_EXACT_NEWTON_C2_RUNNER_CACHE.clear()

    def residual(values: jax.Array) -> jax.Array:
        return values

    assert (
        _optimizer._make_traceable_dense_direct_exact_newton_c2_runner(
            residual, 4, 2.0e-12
        )
        is c2_runner
    )
    assert (
        _optimizer._make_traceable_dense_direct_exact_newton_c2_runner(
            residual, 4, 2.0e-12
        )
        is c2_runner
    )
    assert c2_builds == [(4, 2.0e-12)]


def test_dense_variant_contract_is_strict_transfer_clean() -> None:
    contract = _optimizer.make_traceable_exact_newton_variant_contract("C2")
    initial = jax.device_put(np.asarray([1.25, 0.5], dtype=np.float64))

    with jax.transfer_guard("disallow"):
        result = contract.solver(
            _affine_residual,
            initial,
            maxiter=2,
            tol=1.0e-12,
        )
        jax.block_until_ready(result)

    assert bool(result["success"])
    assert "jacobian" in result
    assert bool(result["exact_newton_variant_dense_linearization_used"])
    assert int(result["exact_newton_variant_dense_materialization_count"]) > 0
    assert int(result["exact_newton_variant_lu_factorization_count"]) > 0
    assert int(result["exact_newton_variant_lu_solve_count"]) > 0
    assert int(result["exact_newton_variant_refinement_correction_count"]) > 0
    assert int(result["exact_newton_variant_stop_reason_code"]) >= 0
    assert not bool(result["exact_newton_variant_numerical_failure"])
    assert int(result["exact_newton_variant_rollback_recompute_count"]) >= 0


@pytest.mark.parametrize("variant", [None, "", "C1", "c2", "C3", 1])
def test_invalid_variant_fails_during_contract_construction(variant: object) -> None:
    with pytest.raises(ValueError, match="must be one of: C0, C2"):
        _optimizer.make_traceable_exact_newton_variant_contract(variant)


def test_c2_contract_preserves_native_rollback_and_stop_telemetry() -> None:
    def residual(values: jax.Array) -> jax.Array:
        return values**3 - 1.0

    contract = _optimizer.make_traceable_exact_newton_variant_contract("C2")
    result = contract.solver(
        residual,
        jnp.asarray([0.1], dtype=jnp.float64),
        maxiter=2,
        tol=1.0e-12,
    )

    assert bool(result["exact_newton_variant_rollback_branch_taken"])
    assert int(result["exact_newton_variant_rollback_recompute_count"]) == 1
    assert int(result["exact_newton_variant_stop_reason_code"]) == (
        _optimizer._C2_STOP_REASON_MAXITER
    )
    assert not bool(result["exact_newton_variant_numerical_failure"])
