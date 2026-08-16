"""Shared contract for the VMEC-free Boozer single-stage example pair."""

from __future__ import annotations

from typing import Final

from simsopt.optimization_endpoint import OptimizationEndpointCertificate

NATIVE_ITERATIONS: Final[int] = 1_000
OUTER_GRADIENT_TOLERANCE: Final[float] = 1.0e-15
JAX_FAST_DRIVER_ID: Final[str] = "simsopt_jax_host_lbfgsb_with_traceable_boozer_newton"
JAX_OPTAX_DRIVER_ID: Final[str] = "simsopt_jax_optax_lbfgs_with_traceable_boozer_newton"
JAX_PARITY_DRIVER_ID: Final[str] = "simsopt_jax_host_bfgs_with_traceable_boozer_newton"


# Terminal states a SOUND bounded run can carry.  This follows the scale-aware
# convention of ``_SOUND_TERMINAL_STATES`` in the projected-route lesson
# (examples/jax/3_Advanced/single_stage_boozer_vacuum_projected_route.py); the
# members differ because the status vocabularies differ.  The bounded lane
# accepts exactly the draws
# its truncated budget produces by construction -- ``converged`` and
# ``iteration-limit``.  Everything else stays failed: ``evaluation-limit`` at
# this budget indicates line-search churn rather than a spent outer budget,
# and a nonfinite state, a collapsed line search, or a provider error is never
# sound at any scale.
SOUND_BOUNDED_STOPPING_REASONS: Final = ("converged", "iteration-limit")


def scientific_success_for_scale(
    *,
    native_scale: bool,
    certificate: OptimizationEndpointCertificate,
    accepted_step: bool,
    inner_success: bool,
    parameters_finite: bool,
    observables_finite: bool,
    monotone_descent: bool,
) -> bool:
    """Judge one endpoint against the contract its own scale is held to.

    ``native_default`` is the lane that claims convergence, so it keeps the full
    endpoint certificate: a converged, stationary, inner-successful endpoint
    that did not raise the objective.

    ``bounded`` is the diagnostic and parity lane the smoke entry point
    selects; its default two-step budget cannot reach stationarity by
    construction, so certifying it against convergence would report every
    honest run as a failure.  It instead claims only what a truncated run can
    show -- the run stopped for a reason that is not a defect, at least one
    step was accepted (or the start was already stationary), the inner Boozer
    solves held, every published observable is finite, and the objective never
    rose.  The published observables are unchanged either way:
    ``outer_stopping_reason`` still reports ``iteration-limit`` and the
    certificate's stationarity fields still report ``False``.
    """
    if native_scale:
        return bool(certificate.success and monotone_descent)
    return bool(
        certificate.stopping_reason in SOUND_BOUNDED_STOPPING_REASONS
        and accepted_step
        and inner_success
        and parameters_finite
        and observables_finite
        and monotone_descent
    )


__all__ = (
    "JAX_FAST_DRIVER_ID",
    "JAX_OPTAX_DRIVER_ID",
    "JAX_PARITY_DRIVER_ID",
    "NATIVE_ITERATIONS",
    "OUTER_GRADIENT_TOLERANCE",
    "SOUND_BOUNDED_STOPPING_REASONS",
    "scientific_success_for_scale",
)
