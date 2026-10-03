"""The ``2_Intermediate/boozer.py`` first stage is classified, not compared by raw status.

The native library's first stage is ``scipy.optimize.minimize(method='L-BFGS-B')``
(``simsopt/geo/boozersurface.py``) and the JAX mirror's is the private
on-device L-BFGS-B port (``boozer_official_options`` selects
``Driver.SIMSOPT_LBFGSB`` -> method ``lbfgs-ondevice``), and
``simsopt_contracts.optimization_endpoint`` keeps their status tables apart
because the same integer means different things in the two.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import numpy as np
from simsopt_contracts.optimization_endpoint import (
    StatusConvention,
    StoppingReason,
    certify_optimization_endpoint,
)
from simsopt_jax.examples.boozer_official import (
    OFFICIAL_LBFGS_MAXITER,
    BoozerStageOutcome,
    BoozerStageState,
)

#: The status vocabulary each implementation's first stage emits.
NATIVE_CONVENTION: StatusConvention = "scipy-lbfgsb"
JAX_CONVENTION: StatusConvention = "private-lbfgsb"


def _first_stage_outcome(
    status: int,
    *,
    iterations: int,
    endpoint_finite: bool,
) -> BoozerStageOutcome:
    """One L-BFGS first-stage outcome with a chosen provider report."""
    objective = 0.25 if endpoint_finite else float("nan")
    return BoozerStageOutcome(
        state=BoozerStageState(
            surface_dofs=np.asarray([1.0, 2.0]),
            iota=-0.4,
            G=1.0,
        ),
        provider_persisted_iterate=True,
        success=False,
        status=status,
        message="",
        nit=iterations,
        nfev=iterations,
        njev=iterations,
        objective=objective,
        gradient_norm=1.0,
        penalty_residual_norm=None,
    )


def _stopping_reason(
    convention: StatusConvention,
    outcome: BoozerStageOutcome,
    *,
    max_iterations: int,
) -> StoppingReason:
    """The contract's stopping reason for one first-stage outcome.

    The gradient norms :func:`certify_optimization_endpoint` takes feed its
    stationarity fields, which a stopping reason does not read; the endpoint's
    finiteness is passed explicitly.
    """
    endpoint_finite = bool(
        np.all(np.isfinite(outcome.state.surface_dofs))
        and np.isfinite(outcome.state.iota)
        and np.isfinite(outcome.state.G)
        and np.isfinite(float(outcome.objective))
        and np.isfinite(outcome.gradient_norm)
    )
    return certify_optimization_endpoint(
        status_convention=convention,
        provider_success=outcome.success,
        provider_status=outcome.status,
        iterations=int(outcome.nit),
        max_iterations=max_iterations,
        initial_gradient_inf_norm=0.0,
        final_gradient_inf_norm=0.0,
        parameters_finite=endpoint_finite,
        observables_finite=endpoint_finite,
        inner_success=True,
    ).stopping_reason


def test_first_stage_stopping_reason_normalizes_the_two_solver_vocabularies() -> None:
    """The comparable fact is the classification, not the raw provider integer.

    The two cases below are the two ways a raw comparison gets the answer
    wrong, and each one also asserts what the RAW integers do, so this test
    fails if the compared value were the raw status again.
    """
    maxiter = OFFICIAL_LBFGS_MAXITER

    # Same stop, different integers: L-BFGS-B reports its budget code 1 with a
    # non-finite endpoint, the port reports its own non-finite code 6. Both are
    # the same endpoint, so the classification is equal and the raw route would
    # have failed.
    native_nonfinite = _first_stage_outcome(
        1, iterations=maxiter, endpoint_finite=False
    )
    jax_nonfinite = _first_stage_outcome(6, iterations=maxiter, endpoint_finite=False)
    assert native_nonfinite.status != jax_nonfinite.status
    assert _stopping_reason(
        NATIVE_CONVENTION, native_nonfinite, max_iterations=maxiter
    ) == _stopping_reason(JAX_CONVENTION, jax_nonfinite, max_iterations=maxiter)
    assert (
        _stopping_reason(JAX_CONVENTION, jax_nonfinite, max_iterations=maxiter)
        == "nonfinite"
    )

    # Different stops, the same integer: 99 is the port's callback stop and is
    # outside SciPy's vocabulary, where the endpoint is simply the exhausted
    # iteration budget. The raw route would have PASSED this pair.
    native_99 = _first_stage_outcome(99, iterations=maxiter, endpoint_finite=True)
    jax_99 = _first_stage_outcome(99, iterations=maxiter, endpoint_finite=True)
    assert native_99.status == jax_99.status
    assert _stopping_reason(
        NATIVE_CONVENTION, native_99, max_iterations=maxiter
    ) != _stopping_reason(JAX_CONVENTION, jax_99, max_iterations=maxiter)
    assert (
        _stopping_reason(NATIVE_CONVENTION, native_99, max_iterations=maxiter)
        == "iteration-limit"
    )
    assert (
        _stopping_reason(JAX_CONVENTION, jax_99, max_iterations=maxiter)
        == "callback-stopped"
    )

    # Upstream's own first-stage outcome: status 1 at the exhausted budget
    # classifies identically under the two conventions.
    budget_exit = _first_stage_outcome(1, iterations=maxiter, endpoint_finite=True)
    assert _stopping_reason(
        NATIVE_CONVENTION, budget_exit, max_iterations=maxiter
    ) == _stopping_reason(JAX_CONVENTION, budget_exit, max_iterations=maxiter)
    assert (
        _stopping_reason(NATIVE_CONVENTION, budget_exit, max_iterations=maxiter)
        == "iteration-limit"
    )
