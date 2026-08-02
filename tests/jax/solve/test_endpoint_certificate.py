"""Fail-closed endpoint certification contracts for host-driven solvers."""

from __future__ import annotations

import pytest
from simsopt_jax.solve.endpoint_certificate import certify_optimization_endpoint


def test_failed_outer_solve_cannot_pass_on_finite_decreasing_values() -> None:
    certificate = certify_optimization_endpoint(
        provider_success=False,
        provider_status=2,
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
