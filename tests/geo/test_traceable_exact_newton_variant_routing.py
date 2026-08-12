from __future__ import annotations

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
    contract = _optimizer._make_traceable_exact_newton_variant_contract("C0")

    assert contract.variant == "C0"
    assert contract.solver is _optimizer.newton_exact_traceable
    assert contract.factorization_backend == "operator-gmres"


def test_c1_and_c2_cached_makers_invoke_only_their_distinct_builders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c1_builds: list[tuple[int, float]] = []
    c2_builds: list[tuple[int, float]] = []
    c1_runner = object()
    c2_runner = object()

    def build_c1(_residual_ref, maxiter: int, tol: float):
        c1_builds.append((maxiter, tol))
        return c1_runner

    def build_c2(_residual_ref, maxiter: int, tol: float):
        c2_builds.append((maxiter, tol))
        return c2_runner

    monkeypatch.setattr(
        _optimizer,
        "_build_traceable_dense_direct_exact_newton_c1_runner",
        build_c1,
    )
    monkeypatch.setattr(
        _optimizer,
        "_build_traceable_dense_direct_exact_newton_c2_runner",
        build_c2,
    )
    _optimizer._TRACEABLE_DENSE_EXACT_NEWTON_C1_RUNNER_CACHE.clear()
    _optimizer._TRACEABLE_DENSE_EXACT_NEWTON_C2_RUNNER_CACHE.clear()

    def residual(values: jax.Array) -> jax.Array:
        return values

    assert (
        _optimizer._make_traceable_dense_direct_exact_newton_c1_runner(
            residual, 3, 1.0e-12
        )
        is c1_runner
    )
    assert (
        _optimizer._make_traceable_dense_direct_exact_newton_c1_runner(
            residual, 3, 1.0e-12
        )
        is c1_runner
    )
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
    assert c1_builds == [(3, 1.0e-12)]
    assert c2_builds == [(4, 2.0e-12)]


@pytest.mark.parametrize("variant", ["C1", "C2"])
def test_dense_variant_contract_is_strict_transfer_clean(variant: str) -> None:
    contract = _optimizer._make_traceable_exact_newton_variant_contract(variant)
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
    assert ("jacobian" in result) is (variant == "C2")
    assert bool(result["exact_newton_variant_dense_linearization_used"])
    assert int(result["exact_newton_variant_dense_materialization_count"]) > 0
    assert int(result["exact_newton_variant_lu_factorization_count"]) > 0
    assert int(result["exact_newton_variant_lu_solve_count"]) > 0
    assert int(result["exact_newton_variant_refinement_correction_count"]) > 0
    if variant == "C1":
        assert int(result["exact_newton_variant_stop_reason_code"]) == 0
        assert not bool(result["exact_newton_variant_numerical_failure"])
        assert not bool(result["exact_newton_variant_stalled"])
        assert not bool(result["exact_newton_variant_retry_linear_solve_at_strict_cap"])
    else:
        assert int(result["exact_newton_variant_stop_reason_code"]) >= 0
        assert not bool(result["exact_newton_variant_numerical_failure"])
        assert int(result["exact_newton_variant_rollback_recompute_count"]) >= 0
    assert "exact_newton_execution_observer_bearing" not in result
    assert "exact_newton_residual_evaluation_count" not in result
    assert "exact_newton_linear_operator_application_count" not in result


@pytest.mark.parametrize("variant", [None, "", "c1", "C3", 1])
def test_invalid_variant_fails_during_contract_construction(variant: object) -> None:
    with pytest.raises(ValueError, match="must be one of: C0, C1, C2"):
        _optimizer._make_traceable_exact_newton_variant_contract(variant)


def test_c2_contract_preserves_native_rollback_and_stop_telemetry() -> None:
    def residual(values: jax.Array) -> jax.Array:
        return values**3 - 1.0

    contract = _optimizer._make_traceable_exact_newton_variant_contract("C2")
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


def test_c1_dense_usage_flag_is_false_when_no_dense_work_runs() -> None:
    def converged_residual(values: jax.Array) -> jax.Array:
        return values - 1.0

    contract = _optimizer._make_traceable_exact_newton_variant_contract("C1")
    result = contract.solver(
        converged_residual,
        jnp.ones(2, dtype=jnp.float64),
        maxiter=2,
        tol=1.0e-12,
    )

    assert bool(result["success"])
    assert not bool(result["exact_newton_variant_dense_linearization_used"])
    assert int(result["exact_newton_variant_dense_materialization_count"]) == 0


@pytest.mark.parametrize(
    ("residual", "initial", "expected_reason", "stalled", "numerical_failure"),
    [
        (
            lambda x: jnp.full_like(x, jnp.nan),
            np.asarray([1.0], dtype=np.float64),
            4,
            False,
            True,
        ),
        (
            lambda x: jnp.asarray([x[0], 1.0], dtype=x.dtype),
            np.asarray([1.0, 0.0], dtype=np.float64),
            3,
            True,
            True,
        ),
        (
            lambda x: jnp.asarray([x[0] ** 3 - 2.0 * x[0] + 2.0]),
            np.asarray([0.0], dtype=np.float64),
            2,
            True,
            False,
        ),
    ],
)
def test_c1_projects_evidence_based_failure_reason(
    residual,
    initial: np.ndarray,
    expected_reason: int,
    stalled: bool,
    numerical_failure: bool,
) -> None:
    contract = _optimizer._make_traceable_exact_newton_variant_contract("C1")

    result = contract.solver(
        residual,
        jax.device_put(initial),
        maxiter=8,
        tol=1.0e-12,
    )
    jax.block_until_ready(result)

    assert int(result["exact_newton_variant_stop_reason_code"]) == expected_reason
    assert bool(result["exact_newton_variant_stalled"]) is stalled
    assert bool(result["exact_newton_variant_numerical_failure"]) is numerical_failure
