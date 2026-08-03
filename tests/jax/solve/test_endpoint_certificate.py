"""Fail-closed endpoint certification contracts for host-driven solvers."""

from __future__ import annotations

import pytest
from simsopt_jax.solve.endpoint_certificate import (
    StatusConvention,
    StoppingReason,
    certify_optimization_endpoint,
    status_convention_for,
)


def test_failed_outer_solve_cannot_pass_on_finite_decreasing_values() -> None:
    certificate = certify_optimization_endpoint(
        provider_success=False,
        provider_status=2,
        status_convention="bfgs",
        iterations=0,
        max_iterations=1000,
        initial_gradient_inf_norm=1.0e-3,
        final_gradient_inf_norm=1.0e-4,
        parameters_finite=True,
        observables_finite=True,
        inner_success=True,
    )

    assert certificate.stopping_reason == "line-search-failed"
    assert certificate.success is False


def test_stationary_provider_success_can_pass_without_an_outer_step() -> None:
    certificate = certify_optimization_endpoint(
        provider_success=True,
        provider_status=0,
        status_convention="bfgs",
        iterations=0,
        max_iterations=1000,
        initial_gradient_inf_norm=5.0e-8,
        final_gradient_inf_norm=5.0e-8,
        parameters_finite=True,
        observables_finite=True,
        inner_success=True,
    )

    assert certificate.initial_stationary is True
    assert certificate.success is True


def test_nonstationary_zero_step_is_rejected_even_with_provider_success() -> None:
    certificate = certify_optimization_endpoint(
        provider_success=True,
        provider_status=0,
        status_convention="bfgs",
        iterations=0,
        max_iterations=1000,
        initial_gradient_inf_norm=1.0e-3,
        final_gradient_inf_norm=1.0e-3,
        parameters_finite=True,
        observables_finite=True,
        inner_success=True,
    )

    assert certificate.initial_stationary is False
    assert certificate.success is False


def test_nonstationary_terminal_gradient_is_rejected_after_steps() -> None:
    certificate = certify_optimization_endpoint(
        provider_success=True,
        provider_status=0,
        status_convention="bfgs",
        iterations=4,
        max_iterations=1000,
        initial_gradient_inf_norm=1.0e-3,
        final_gradient_inf_norm=2.0e-7,
        parameters_finite=True,
        observables_finite=True,
        inner_success=True,
    )

    assert certificate.terminal_stationary is False
    assert certificate.success is False


def test_nonfinite_or_failed_inner_state_is_rejected() -> None:
    nonfinite = certify_optimization_endpoint(
        provider_success=True,
        provider_status=0,
        status_convention="bfgs",
        iterations=4,
        max_iterations=1000,
        initial_gradient_inf_norm=1.0e-3,
        final_gradient_inf_norm=1.0e-8,
        parameters_finite=False,
        observables_finite=True,
        inner_success=True,
    )
    failed_inner = certify_optimization_endpoint(
        provider_success=True,
        provider_status=0,
        status_convention="bfgs",
        iterations=4,
        max_iterations=1000,
        initial_gradient_inf_norm=1.0e-3,
        final_gradient_inf_norm=1.0e-8,
        parameters_finite=True,
        observables_finite=True,
        inner_success=False,
    )

    assert nonfinite.stopping_reason == "nonfinite"
    assert nonfinite.success is False
    assert failed_inner.success is False


def test_constraint_norm_is_checked_fail_closed() -> None:
    violated = certify_optimization_endpoint(
        provider_success=True,
        provider_status=0,
        status_convention="bfgs",
        iterations=4,
        max_iterations=1000,
        initial_gradient_inf_norm=1.0e-3,
        final_gradient_inf_norm=1.0e-8,
        parameters_finite=True,
        observables_finite=True,
        inner_success=True,
        constraint_norm=2.0e-10,
    )
    nonfinite = certify_optimization_endpoint(
        provider_success=True,
        provider_status=0,
        status_convention="bfgs",
        iterations=4,
        max_iterations=1000,
        initial_gradient_inf_norm=1.0e-3,
        final_gradient_inf_norm=1.0e-8,
        parameters_finite=True,
        observables_finite=True,
        inner_success=True,
        constraint_norm=float("nan"),
    )

    assert violated.constraints_satisfied is False
    assert violated.success is False
    assert nonfinite.constraints_satisfied is False
    assert nonfinite.success is False


def test_negative_norms_are_rejected_fail_closed() -> None:
    negative_gradient = certify_optimization_endpoint(
        provider_success=True,
        provider_status=0,
        status_convention="bfgs",
        iterations=4,
        max_iterations=1000,
        initial_gradient_inf_norm=-1.0,
        final_gradient_inf_norm=1.0e-8,
        parameters_finite=True,
        observables_finite=True,
        inner_success=True,
    )
    negative_constraint = certify_optimization_endpoint(
        provider_success=True,
        provider_status=0,
        status_convention="bfgs",
        iterations=4,
        max_iterations=1000,
        initial_gradient_inf_norm=1.0e-3,
        final_gradient_inf_norm=1.0e-8,
        parameters_finite=True,
        observables_finite=True,
        inner_success=True,
        constraint_norm=-1.0,
    )

    assert negative_gradient.success is False
    assert negative_constraint.constraints_satisfied is False
    assert negative_constraint.success is False


@pytest.mark.parametrize(
    ("iterations", "max_iterations"),
    ((-1, 1000), (0, 0), (-1, -1), (1001, 1000)),
)
def test_invalid_iteration_budgets_are_rejected_fail_closed(
    iterations: int,
    max_iterations: int,
) -> None:
    certificate = certify_optimization_endpoint(
        provider_success=True,
        provider_status=0,
        status_convention="bfgs",
        iterations=iterations,
        max_iterations=max_iterations,
        initial_gradient_inf_norm=1.0e-8,
        final_gradient_inf_norm=1.0e-8,
        parameters_finite=True,
        observables_finite=True,
        inner_success=True,
    )

    assert certificate.stopping_reason == "failed"
    assert certificate.success is False


@pytest.mark.parametrize(
    "status_convention",
    ("bfgs", "host-lbfgsb", "scipy-lbfgsb", "optax-lbfgs"),
)
@pytest.mark.parametrize("provider_status", (2, 5, 6, 99, None, 42))
def test_provider_success_with_non_success_status_fails_closed(
    status_convention: StatusConvention,
    provider_status: int | None,
) -> None:
    certificate = certify_optimization_endpoint(
        provider_success=True,
        provider_status=provider_status,
        status_convention=status_convention,
        iterations=1,
        max_iterations=1000,
        initial_gradient_inf_norm=1.0e-3,
        final_gradient_inf_norm=1.0e-8,
        parameters_finite=True,
        observables_finite=True,
        inner_success=True,
    )

    assert certificate.stopping_reason == "failed"
    assert certificate.success is False


def test_contradictory_success_status_precedes_nonfinite_reason() -> None:
    certificate = certify_optimization_endpoint(
        provider_success=True,
        provider_status=6,
        status_convention="bfgs",
        iterations=1,
        max_iterations=1000,
        initial_gradient_inf_norm=1.0e-3,
        final_gradient_inf_norm=1.0e-8,
        parameters_finite=False,
        observables_finite=True,
        inner_success=True,
    )

    assert certificate.stopping_reason == "failed"
    assert certificate.success is False


@pytest.mark.parametrize(
    ("status_convention", "provider_status"),
    (
        ("bfgs", 0),
        ("host-lbfgsb", 0),
        ("host-lbfgsb", 4),
        ("scipy-lbfgsb", 0),
        ("optax-lbfgs", 0),
    ),
)
def test_provider_success_with_convention_success_status_can_certify(
    status_convention: StatusConvention,
    provider_status: int,
) -> None:
    certificate = certify_optimization_endpoint(
        provider_success=True,
        provider_status=provider_status,
        status_convention=status_convention,
        iterations=1,
        max_iterations=1000,
        initial_gradient_inf_norm=1.0e-3,
        final_gradient_inf_norm=1.0e-8,
        parameters_finite=True,
        observables_finite=True,
        inner_success=True,
    )

    assert certificate.stopping_reason == "converged"
    assert certificate.success is True


@pytest.mark.parametrize(
    ("status_convention", "provider_status", "expected_reason"),
    (
        ("bfgs", 1, "iteration-limit"),
        ("bfgs", 2, "line-search-failed"),
        ("bfgs", 3, "nonfinite"),
        ("bfgs", 6, "nonfinite"),
        ("bfgs", 99, "callback-stopped"),
        ("host-lbfgsb", 1, "iteration-limit"),
        ("host-lbfgsb", 2, "evaluation-limit"),
        ("host-lbfgsb", 3, "evaluation-limit"),
        ("host-lbfgsb", 5, "line-search-failed"),
        ("host-lbfgsb", 6, "nonfinite"),
        ("host-lbfgsb", 99, "callback-stopped"),
        ("scipy-lbfgsb", 1, "iteration-limit"),
        ("scipy-lbfgsb", 2, "line-search-failed"),
        ("optax-lbfgs", 1, "iteration-limit"),
        ("optax-lbfgs", 2, "line-search-failed"),
        ("optax-lbfgs", 6, "nonfinite"),
    ),
)
def test_provider_failure_uses_convention_specific_reason(
    status_convention: StatusConvention,
    provider_status: int,
    expected_reason: StoppingReason,
) -> None:
    certificate = certify_optimization_endpoint(
        provider_success=False,
        provider_status=provider_status,
        status_convention=status_convention,
        iterations=1,
        max_iterations=1000,
        initial_gradient_inf_norm=1.0e-3,
        final_gradient_inf_norm=1.0e-8,
        parameters_finite=True,
        observables_finite=True,
        inner_success=True,
    )

    assert certificate.stopping_reason == expected_reason
    assert certificate.success is False


@pytest.mark.parametrize(
    ("iterations", "expected_reason"),
    ((1, "failed"), (1000, "iteration-limit")),
)
def test_unknown_failure_status_uses_iteration_budget_fallback(
    iterations: int,
    expected_reason: StoppingReason,
) -> None:
    certificate = certify_optimization_endpoint(
        provider_success=False,
        provider_status=42,
        status_convention="host-lbfgsb",
        iterations=iterations,
        max_iterations=1000,
        initial_gradient_inf_norm=1.0e-3,
        final_gradient_inf_norm=1.0e-8,
        parameters_finite=True,
        observables_finite=True,
        inner_success=True,
    )

    assert certificate.stopping_reason == expected_reason
    assert certificate.success is False


@pytest.mark.parametrize(
    ("provider", "method", "expected_convention"),
    (
        ("custom", "bfgs", "bfgs"),
        ("native", "bfgs", "bfgs"),
        ("custom", "lbfgs", "host-lbfgsb"),
        ("optax", "lbfgs", "optax-lbfgs"),
        ("native", "lbfgs", "scipy-lbfgsb"),
    ),
)
def test_status_convention_follows_provider_and_method(
    provider: str,
    method: str,
    expected_convention: StatusConvention,
) -> None:
    assert status_convention_for(provider, method) == expected_convention


def test_optax_success_claim_with_host_lbfgsb_success_status_fails_closed() -> None:
    certificate = certify_optimization_endpoint(
        provider_success=True,
        provider_status=4,
        status_convention="optax-lbfgs",
        iterations=1,
        max_iterations=1000,
        initial_gradient_inf_norm=1.0e-3,
        final_gradient_inf_norm=1.0e-8,
        parameters_finite=True,
        observables_finite=True,
        inner_success=True,
    )

    assert certificate.stopping_reason == "failed"
    assert certificate.success is False


def test_unsupported_provider_method_pair_has_no_status_fallback() -> None:
    with pytest.raises(
        ValueError,
        match="provider='native', method='newton-cg'",
    ):
        status_convention_for("native", "newton-cg")


@pytest.mark.parametrize(
    ("provider", "provider_status", "expected_reason"),
    (
        ("custom", 1, "iteration-limit"),
        ("custom", 2, "line-search-failed"),
        ("native", 1, "iteration-limit"),
    ),
)
def test_published_bfgs_lane_stopping_reasons_remain_recomputable(
    provider: str,
    provider_status: int,
    expected_reason: StoppingReason,
) -> None:
    certificate = certify_optimization_endpoint(
        provider_success=False,
        provider_status=provider_status,
        status_convention=status_convention_for(provider, "bfgs"),
        iterations=1,
        max_iterations=1000,
        initial_gradient_inf_norm=1.0e-3,
        final_gradient_inf_norm=1.0e-8,
        parameters_finite=True,
        observables_finite=True,
        inner_success=True,
    )

    assert certificate.stopping_reason == expected_reason
    assert certificate.success is False
