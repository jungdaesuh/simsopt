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
        status_convention="scipy-bfgs",
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
        status_convention="scipy-bfgs",
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
        status_convention="scipy-bfgs",
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
        status_convention="scipy-bfgs",
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
        status_convention="scipy-bfgs",
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
        status_convention="scipy-bfgs",
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
        status_convention="scipy-bfgs",
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
        status_convention="scipy-bfgs",
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
        status_convention="scipy-bfgs",
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
        status_convention="scipy-bfgs",
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
        status_convention="scipy-bfgs",
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
    (
        "scipy-bfgs",
        "private-bfgs",
        "host-bfgs",
        "scipy-lbfgsb",
        "private-lbfgsb",
        "host-lbfgsb",
        "optax-lbfgs",
    ),
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
        status_convention="scipy-bfgs",
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
        ("scipy-bfgs", 0),
        ("private-bfgs", 0),
        ("host-bfgs", 0),
        ("host-lbfgsb", 0),
        ("host-lbfgsb", 4),
        ("scipy-lbfgsb", 0),
        ("private-lbfgsb", 0),
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
        ("scipy-bfgs", 1, "iteration-limit"),
        ("scipy-bfgs", 2, "line-search-failed"),
        ("scipy-bfgs", 3, "nonfinite"),
        # Private BFGS reverses SciPy's 2/3: outer 2 is the nonfinite
        # trial (inner line-search status -1), outer 3 = 2 + ls-failed.
        ("private-bfgs", 1, "iteration-limit"),
        ("private-bfgs", 2, "nonfinite"),
        ("private-bfgs", 3, "line-search-failed"),
        ("private-bfgs", 5, "line-search-failed"),
        ("private-bfgs", 99, "callback-stopped"),
        ("host-bfgs", 1, "iteration-limit"),
        ("host-bfgs", 2, "line-search-failed"),
        ("host-lbfgsb", 1, "iteration-limit"),
        ("host-lbfgsb", 2, "evaluation-limit"),
        ("host-lbfgsb", 3, "evaluation-limit"),
        ("host-lbfgsb", 5, "line-search-failed"),
        ("host-lbfgsb", 6, "nonfinite"),
        ("scipy-lbfgsb", 2, "line-search-failed"),
        ("private-lbfgsb", 2, "line-search-failed"),
        ("private-lbfgsb", 6, "nonfinite"),
        ("private-lbfgsb", 99, "callback-stopped"),
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
    ("provider", "method", "accepted_incumbent", "expected_convention"),
    (
        ("custom", "bfgs", True, "host-bfgs"),
        ("custom", "bfgs", False, "private-bfgs"),
        ("native", "bfgs", False, "scipy-bfgs"),
        ("custom", "lbfgs", False, "private-lbfgsb"),
        ("optax", "lbfgs", False, "optax-lbfgs"),
        ("native", "lbfgs", False, "scipy-lbfgsb"),
    ),
)
def test_status_convention_follows_the_emitting_solver_route(
    provider: str,
    method: str,
    accepted_incumbent: bool,
    expected_convention: StatusConvention,
) -> None:
    assert (
        status_convention_for(
            provider,
            method,
            accepted_incumbent=accepted_incumbent,
        )
        == expected_convention
    )


@pytest.mark.parametrize(
    ("provider", "method"),
    (("native", "bfgs"), ("custom", "lbfgs"), ("optax", "lbfgs"), ("native", "lbfgs")),
)
def test_accepted_incumbent_outside_custom_bfgs_is_rejected(
    provider: str,
    method: str,
) -> None:
    with pytest.raises(ValueError, match="accepted-incumbent"):
        status_convention_for(provider, method, accepted_incumbent=True)


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
        status_convention_for("native", "newton-cg", accepted_incumbent=False)


@pytest.mark.parametrize(
    ("provider", "accepted_incumbent", "provider_status", "expected_reason"),
    (
        # The published Boozer lanes run custom BFGS under
        # accepted-incumbent continuation (host core emitter).
        ("custom", True, 1, "iteration-limit"),
        ("custom", True, 2, "line-search-failed"),
        ("native", False, 1, "iteration-limit"),
    ),
)
def test_published_bfgs_lane_stopping_reasons_remain_recomputable(
    provider: str,
    accepted_incumbent: bool,
    provider_status: int,
    expected_reason: StoppingReason,
) -> None:
    certificate = certify_optimization_endpoint(
        provider_success=False,
        provider_status=provider_status,
        status_convention=status_convention_for(
            provider,
            "bfgs",
            accepted_incumbent=accepted_incumbent,
        ),
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


# ---------------------------------------------------------------------------
# Real-emitter certification: drive the actual solvers into their failure
# paths and certify the raw result, so the convention tables cannot drift
# from the emitters they transcribe.
# ---------------------------------------------------------------------------


def _certify_emitter_result(
    *,
    convention: StatusConvention,
    success: bool,
    status: int,
    iterations: int,
    max_iterations: int,
    finite: bool = True,
) -> StoppingReason:
    certificate = certify_optimization_endpoint(
        provider_success=success,
        provider_status=status,
        status_convention=convention,
        iterations=iterations,
        max_iterations=max_iterations,
        initial_gradient_inf_norm=1.0 if finite else float("nan"),
        final_gradient_inf_norm=1.0e-1 if finite else float("nan"),
        parameters_finite=finite,
        observables_finite=finite,
        inner_success=True,
    )
    return certificate.stopping_reason


def _constant_value_unit_gradient(x):
    import jax
    import jax.numpy as jnp

    return jnp.sum(x) - jnp.sum(jax.lax.stop_gradient(x)) + 1.0


def _nan_off_the_start(x):
    import jax.numpy as jnp

    return jnp.sum(x) + jnp.sqrt(1.0e-30 - jnp.sum((x - 1.0) ** 2))


def test_private_lbfgsb_line_search_failure_certifies_line_search_failed() -> None:
    import jax.numpy as jnp
    from simsopt_jax.geo.optimizers.private._lbfgs import _minimize_lbfgs_private

    result = _minimize_lbfgs_private(
        _constant_value_unit_gradient,
        jnp.zeros(2, dtype=jnp.float64),
        maxiter=50,
        maxcor=5,
        gtol=1.0e-12,
        maxls=5,
        x_dtype=jnp.float64,
    )

    assert int(result.status) == 2
    assert not bool(result.converged)
    assert (
        _certify_emitter_result(
            convention="private-lbfgsb",
            success=bool(result.converged),
            status=int(result.status),
            iterations=int(result.k),
            max_iterations=50,
        )
        == "line-search-failed"
    )


def test_private_lbfgsb_nonfinite_trial_certifies_nonfinite() -> None:
    import jax.numpy as jnp
    from simsopt_jax.geo.optimizers.private._lbfgs import _minimize_lbfgs_private

    result = _minimize_lbfgs_private(
        _nan_off_the_start,
        jnp.ones(2, dtype=jnp.float64),
        maxiter=50,
        maxcor=5,
        gtol=1.0e-12,
        maxls=5,
        x_dtype=jnp.float64,
    )

    assert int(result.status) == 6
    assert (
        _certify_emitter_result(
            convention="private-lbfgsb",
            success=bool(result.converged),
            status=int(result.status),
            iterations=int(result.k),
            max_iterations=50,
        )
        == "nonfinite"
    )


def test_private_lbfgsb_evaluation_budget_certifies_evaluation_limit() -> None:
    import jax.numpy as jnp
    from simsopt_jax.geo.optimizers.private._lbfgs import _minimize_lbfgs_private

    def rosenbrock(x):
        return jnp.sum(
            100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2
        )

    result = _minimize_lbfgs_private(
        rosenbrock,
        jnp.asarray([-1.2, 1.0], dtype=jnp.float64),
        maxiter=1000,
        maxcor=5,
        gtol=1.0e-14,
        maxfun=3,
        x_dtype=jnp.float64,
    )

    assert int(result.status) == 1
    assert int(result.k) < 1000
    assert (
        _certify_emitter_result(
            convention="private-lbfgsb",
            success=bool(result.converged),
            status=int(result.status),
            iterations=int(result.k),
            max_iterations=1000,
        )
        == "evaluation-limit"
    )


def test_private_bfgs_nan_trials_certify_fail_closed() -> None:
    # The hardened More-Thuente port refuses to RETURN a nonfinite point,
    # so the outer status-2 branch (ls -1, mapped nonfinite in the table)
    # is a structural backstop: NaN trials surface as a line-search
    # failure (outer 3) and an initially-NaN state halts pre-step with
    # the -1 sentinel, which certification leaves fail-closed.
    import jax.numpy as jnp
    from simsopt_jax.geo.optimizers.private._bfgs import _minimize_bfgs_private

    def nan_gradient_after_moving(x):
        moved = jnp.sum((x - 1.0) ** 2) > 0.0
        return jnp.sum(x**2), jnp.where(moved, jnp.nan, 2.0 * x)

    trial_nan = _minimize_bfgs_private(
        nan_gradient_after_moving,
        jnp.ones(2, dtype=jnp.float64),
        maxiter=50,
        gtol=1.0e-12,
        value_and_grad=True,
        x_dtype=jnp.float64,
    )
    assert int(trial_nan.status) == 3
    assert (
        _certify_emitter_result(
            convention="private-bfgs",
            success=bool(trial_nan.converged),
            status=int(trial_nan.status),
            iterations=int(trial_nan.k),
            max_iterations=50,
        )
        == "line-search-failed"
    )

    def nan_gradient_everywhere(x):
        return jnp.sum(x**2), jnp.full_like(x, jnp.nan)

    initial_nan = _minimize_bfgs_private(
        nan_gradient_everywhere,
        jnp.ones(2, dtype=jnp.float64),
        maxiter=50,
        gtol=1.0e-12,
        value_and_grad=True,
        x_dtype=jnp.float64,
    )
    assert int(initial_nan.status) == -1
    assert int(initial_nan.k) == 0
    assert (
        _certify_emitter_result(
            convention="private-bfgs",
            success=bool(initial_nan.converged),
            status=int(initial_nan.status),
            iterations=int(initial_nan.k),
            max_iterations=50,
            finite=False,
        )
        == "nonfinite"
    )


def test_private_bfgs_line_search_failure_certifies_line_search_failed() -> None:
    import jax.numpy as jnp
    from simsopt_jax.geo.optimizers.private._bfgs import _minimize_bfgs_private

    result = _minimize_bfgs_private(
        _constant_value_unit_gradient,
        jnp.zeros(2, dtype=jnp.float64),
        maxiter=50,
        gtol=1.0e-12,
        x_dtype=jnp.float64,
    )

    assert int(result.status) in (3, 5)
    assert (
        _certify_emitter_result(
            convention="private-bfgs",
            success=bool(result.converged),
            status=int(result.status),
            iterations=int(result.k),
            max_iterations=50,
        )
        == "line-search-failed"
    )


def test_host_bfgs_line_search_failure_certifies_line_search_failed() -> None:
    import numpy as np
    from simsopt_jax.geo.optimizer_host_lbfgs import minimize_bfgs_host_core

    def constant_value_unit_gradient(x):
        return 1.0, np.ones_like(x)

    result = minimize_bfgs_host_core(
        constant_value_unit_gradient,
        np.zeros(2, dtype=np.float64),
        maxiter=50,
        gtol=1.0e-12,
    )

    assert int(result.status) == 2
    assert result.failed
    assert (
        _certify_emitter_result(
            convention="host-bfgs",
            success=bool(result.converged),
            status=int(result.status),
            iterations=int(result.k),
            max_iterations=50,
        )
        == "line-search-failed"
    )


@pytest.mark.parametrize("convention", ("private-lbfgsb", "scipy-lbfgsb"))
def test_merged_budget_status_discriminates_on_iteration_evidence(
    convention: StatusConvention,
) -> None:
    assert (
        _certify_emitter_result(
            convention=convention,
            success=False,
            status=1,
            iterations=7,
            max_iterations=1000,
        )
        == "evaluation-limit"
    )
    assert (
        _certify_emitter_result(
            convention=convention,
            success=False,
            status=1,
            iterations=1000,
            max_iterations=1000,
        )
        == "iteration-limit"
    )


def test_scipy_bfgs_failure_statuses_certify_from_the_real_emitter() -> None:
    import numpy as np
    from scipy import optimize

    def nan_after_moving(x):
        moved = float(np.sum((x - 1.0) ** 2)) > 0.0
        return float("nan") if moved else float(np.sum(x**2))

    nan_result = optimize.minimize(
        nan_after_moving,
        np.ones(2, dtype=np.float64),
        method="BFGS",
        options={"maxiter": 50},
    )
    assert int(nan_result.status) == 3
    assert (
        _certify_emitter_result(
            convention="scipy-bfgs",
            success=bool(nan_result.success),
            status=int(nan_result.status),
            iterations=int(nan_result.nit),
            max_iterations=50,
        )
        == "nonfinite"
    )

    def rosenbrock(x):
        return float(
            np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)
        )

    limited = optimize.minimize(
        rosenbrock,
        np.asarray([-1.2, 1.0], dtype=np.float64),
        method="BFGS",
        options={"maxiter": 2},
    )
    assert int(limited.status) == 1
    assert (
        _certify_emitter_result(
            convention="scipy-bfgs",
            success=bool(limited.success),
            status=int(limited.status),
            iterations=int(limited.nit),
            max_iterations=2,
        )
        == "iteration-limit"
    )

    def nonsmooth(x):
        return float(np.sum(np.abs(x)))

    precision_loss = optimize.minimize(
        nonsmooth,
        np.ones(2, dtype=np.float64) * 0.3,
        method="BFGS",
        options={"maxiter": 50},
    )
    assert int(precision_loss.status) == 2
    assert (
        _certify_emitter_result(
            convention="scipy-bfgs",
            success=bool(precision_loss.success),
            status=int(precision_loss.status),
            iterations=int(precision_loss.nit),
            max_iterations=50,
        )
        == "line-search-failed"
    )
