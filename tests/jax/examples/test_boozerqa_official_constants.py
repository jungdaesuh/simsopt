"""The official BoozerQA constants and outer-endpoint admissibility."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from simsopt_contracts.optimization_endpoint import (
    OptimizationEndpointCertificate,
    certify_optimization_endpoint,
)
from simsopt_jax.examples.boozer_official import (
    OFFICIAL_QA_ADMISSIBLE_OUTER_STOPPING_REASONS,
    OFFICIAL_QA_NEWTON_MAXITER,
    OFFICIAL_QA_NEWTON_TOLERANCE,
    OFFICIAL_QA_NON_QS_RESOLUTION,
    OFFICIAL_QA_OUTER_MAXITER,
    OFFICIAL_QA_PENALTY_WEIGHT,
    boozer_qa_outer_objective_config,
)
from simsopt_jax.examples.single_stage_boozer_vacuum import OUTER_GRADIENT_TOLERANCE


def test_official_constants_match_upstream_boozerqa() -> None:
    """Upstream boozerQA.py: Newton tol 1e-13 cap 20, sDIM 20, BFGS tol 1e-15."""
    assert OFFICIAL_QA_NEWTON_TOLERANCE == 1.0e-13
    assert OFFICIAL_QA_NEWTON_MAXITER == 20
    assert OFFICIAL_QA_NON_QS_RESOLUTION == 20
    assert OFFICIAL_QA_OUTER_MAXITER == 1000
    assert OUTER_GRADIENT_TOLERANCE == 1.0e-15
    configuration = boozer_qa_outer_objective_config(
        nfp=3,
        non_qs_resolution=OFFICIAL_QA_NON_QS_RESOLUTION,
        length_target=1.0,
        major_radius_target=1.0,
        vessel_gamma=[[[0.0, 0.0, 0.0]]],
    )
    assert configuration["non_qs_weight"] == OFFICIAL_QA_PENALTY_WEIGHT == 1.0
    assert configuration["iota_weight"] == OFFICIAL_QA_PENALTY_WEIGHT
    assert configuration["major_radius_weight"] == OFFICIAL_QA_PENALTY_WEIGHT
    assert configuration["length_weight"] == OFFICIAL_QA_PENALTY_WEIGHT
    assert configuration["residual_weight"] == 0.0


#: The shipped script's ``OUTER_STATUS_CONVENTION``; a numbered example
#: directory is not an importable package, so it is written out here.
_HOST_BFGS = "host-bfgs"


def _outer_endpoint(
    *,
    provider_success: bool,
    provider_status: int,
    iterations: int,
    observables_finite: bool = True,
) -> OptimizationEndpointCertificate:
    """Certify one outer endpoint exactly as the shipped script does."""
    return certify_optimization_endpoint(
        status_convention=_HOST_BFGS,
        provider_success=provider_success,
        provider_status=provider_status,
        iterations=iterations,
        max_iterations=OFFICIAL_QA_OUTER_MAXITER,
        initial_gradient_inf_norm=1.0,
        final_gradient_inf_norm=0.5,
        parameters_finite=True,
        observables_finite=observables_finite,
        inner_success=True,
    )


def test_only_the_official_budget_exit_and_convergence_are_admissible() -> None:
    """The endpoint certificate decides whether the run may report success.

    Upstream's BoozerQA ends on its 1000-iteration budget, so a budget exit
    stays admissible and ``solver_success`` false is not a failure.  A failed
    line search or a non-finite endpoint is a different mode, and the delta
    that only reported the stopping reason let those report ``ok``.
    """
    budget = _outer_endpoint(
        provider_success=False,
        provider_status=1,
        iterations=OFFICIAL_QA_OUTER_MAXITER,
    )
    assert budget.stopping_reason == "iteration-limit"
    assert budget.stopping_reason in OFFICIAL_QA_ADMISSIBLE_OUTER_STOPPING_REASONS
    # Upstream's own endpoint is not certified successful; admissible is not
    # the same claim as converged.
    assert budget.success is False

    converged = _outer_endpoint(
        provider_success=True,
        provider_status=0,
        iterations=12,
    )
    assert converged.stopping_reason == "converged"
    assert converged.stopping_reason in OFFICIAL_QA_ADMISSIBLE_OUTER_STOPPING_REASONS

    line_search = _outer_endpoint(
        provider_success=False,
        provider_status=2,
        iterations=3,
    )
    assert line_search.stopping_reason == "line-search-failed"
    assert (
        line_search.stopping_reason not in OFFICIAL_QA_ADMISSIBLE_OUTER_STOPPING_REASONS
    )

    nonfinite = _outer_endpoint(
        provider_success=False,
        provider_status=1,
        iterations=3,
        observables_finite=False,
    )
    assert nonfinite.stopping_reason == "nonfinite"
    assert (
        nonfinite.stopping_reason not in OFFICIAL_QA_ADMISSIBLE_OUTER_STOPPING_REASONS
    )
