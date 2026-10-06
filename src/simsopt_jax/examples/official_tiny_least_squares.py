"""Host SciPy TRF policy for the three official tiny least-squares examples.

The residual is evaluated by JAX. SciPy owns finite-difference Jacobian
sampling and stopping, as in ``least_squares_serial_solve(grad=None)``.

It lives in the installed package because the three examples
(``examples/jax/1_Simple/just_a_quadratic.py``, ``minimize_curve_length.py``,
``surf_vol_area.py``) are executed as standalone scripts, with no repository
root on ``sys.path``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Final, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import OptimizeResult, least_squares
from simsopt_contracts.optimization_endpoint import (
    NormalizedTerminalStatus,
    StatusConvention,
    StoppingReason,
    normalized_terminal_status,
    stopping_reason_for_status,
)
from simsopt_jax_adapters.geo.curve_objectives import curve_length_pure

from simsopt_jax.backend.dtypes import explicit_device_array
from simsopt_jax.core.curve_geometry import (
    curve_incremental_arclength_from_spec,
    curve_spec_with_dofs,
)
from simsopt_jax.core.specs import make_curve_rzfourier_spec
from simsopt_jax.core.surface_rzfourier import (
    surface_rz_fourier_area_from_dofs,
    surface_rz_fourier_spec_from_dofs,
    surface_rz_fourier_volume_from_dofs,
)
from simsopt_jax.examples.weighted_quadratic import weighted_quadratic_residuals
from simsopt_jax.runtime.host_boundary import allow_host_transfers, host_array

DRIVER_QUADRATIC = "scipy_least_squares_trf_jax_quadratic"
DRIVER_CURVE = "scipy_least_squares_trf_jax_curve_length"
DRIVER_SURFACE = "scipy_least_squares_trf_jax_surf_vol_area"
# scipy.optimize.least_squares computes its own default evaluation budget as
# ``max_nfev = x0.size * 100`` (scipy/optimize/_lsq/trf.py:250 and :452). An
# example that must publish an integer budget derives it from this factor so the
# declared budget is SciPy's own and can never decide an official outcome.
OFFICIAL_MAX_NFEV_PER_PARAMETER = 100
#: The official body is unseeded (``x0 = np.random.rand(curve.dof_size) - 0.5``
#: with ``x0[0] = 3.0``), so the capture seeds the legacy global generator and
#: this constant is that realization. Regeneration recipe, reproduced bit for
#: bit by ``test_controlled_curve_start_regenerates_from_its_seed`` and by
#: replaying the official body:
#:
#:     np.random.seed(CONTROLLED_CURVE_REPLAY_SEED)
#:     np.random.rand(CONTROLLED_CURVE_DRAWS_BEFORE_START)   # discarded
#:     x0 = np.random.rand(9) - 0.5
#:     x0[0] = 3.0
#:
#: The discarded draws are what importing simsopt costs: the capture runner
#: (``A/reference-simple/generated/examples/1_Simple/``
#: ``capture_minimize_curve_length_controlled.py``) seeds on the line before
#: the verbatim official body, whose own ``import`` statements run next.
#: Evidence: ``A/reference-simple/runs/native-minimize-curve-length/``
#: ``natural-draw-record.json`` (``controlled_replay_seed`` 20260919,
#: ``controlled_exact_initial_full_parameters_for_jax``).
CONTROLLED_CURVE_REPLAY_SEED = 20260919
CONTROLLED_CURVE_DRAWS_BEFORE_START = 5
CONTROLLED_CURVE_INITIAL_FULL = (
    3.0,
    0.2811508886084869,
    -0.3029382519787718,
    0.0810994782579404,
    -0.017172649927526984,
    -0.4931910426039152,
    -0.02585553840931798,
    -0.4283948617087845,
    0.1282806388973341,
)


TrfNormalizedStatus = NormalizedTerminalStatus

#: Emitter convention of every solve in this module: host
#: ``scipy.optimize.least_squares(method="trf")``, which is what
#: ``least_squares_serial_solve(grad=None)`` runs. The status vocabulary itself
#: is owned by ``simsopt_contracts.optimization_endpoint``; this module names
#: its emitter and reads nothing else.
TRF_STATUS_CONVENTION: Final[StatusConvention] = "scipy-trf"


#: The inverse map, for folding outcomes that are already normalized (including
#: one demoted by :func:`guard_finite_endpoint`) back through the shared fold.
_STOPPING_REASON_BY_NORMALIZED_STATUS: Final[
    Mapping[TrfNormalizedStatus, StoppingReason]
] = MappingProxyType(
    {
        "converged": "converged",
        "budget_exhausted": "evaluation-limit",
        "failed": "failed",
    }
)


@dataclass(frozen=True)
class TrfOutcome:
    """What SciPy's trust-region provider reported about one workflow.

    ``normalized_status`` folds the provider's own status into the arbiter's
    vocabulary; ``raw_status`` keeps the status number and SciPy's message so the
    published receipt can be compared with an official capture.
    """

    normalized_status: TrfNormalizedStatus
    raw_status: str
    success: bool
    nfev: int
    njev: int


def official_default_max_nfev(parameter_count: int) -> int:
    """Return SciPy's own default ``least_squares`` budget for this problem size."""
    return OFFICIAL_MAX_NFEV_PER_PARAMETER * parameter_count


def trf_outcome(result: OptimizeResult) -> TrfOutcome:
    """Normalize one ``scipy.optimize.least_squares`` result.

    The status is classified by the contract's own ``scipy-trf`` table and the
    label is the one ``optimization_endpoint.normalized_terminal_status`` gives
    every other lane; nothing about either vocabulary is restated here.

    ``least_squares`` reports NO iteration count -- its budget is ``max_nfev``
    and its budget signal is status 0 -- so this call has no iteration evidence
    to offer and does not invent any: it asks the contract for the
    status-only classification. Passing ``nfev`` against itself, as this
    function used to, made ``certify_optimization_endpoint``'s terminal arm
    ``iterations >= max_iterations`` always true, so ANY status outside the
    ``scipy-trf`` table became ``iteration-limit`` and folded to
    ``budget_exhausted``: a fabricated budget stop for an emitter that has no
    iteration limit. An unknown status is a failure.
    """
    status = int(result.status)
    terminal = normalized_terminal_status(
        scientific_predicate=True,
        stage_stopping_reasons=(
            stopping_reason_for_status(
                status_convention=TRF_STATUS_CONVENTION,
                provider_success=bool(result.success),
                provider_status=status,
                finite=bool(
                    np.all(np.isfinite(result.x)) and np.all(np.isfinite(result.fun))
                ),
            ),
        ),
    )
    return TrfOutcome(
        normalized_status=terminal.normalized_status,
        raw_status=f"{status} {result.message}",
        success=terminal.success,
        nfev=int(result.nfev),
        njev=int(result.njev),
    )


def combine_trf_outcomes(first: TrfOutcome, *rest: TrfOutcome) -> TrfOutcome:
    """Fold one outcome per solve into the outcome of a multi-solve workflow.

    The fold itself is not restated here: it is
    ``optimization_endpoint.normalized_terminal_status``, the rule every other
    optimizer-backed lane folds with.
    """
    outcomes = (first, *rest)
    terminal = normalized_terminal_status(
        # Every condition this workflow has is already in the stopping reasons
        # (a nonfinite endpoint arrives as ``failed``), so there is no separate
        # scientific predicate to fold in.
        scientific_predicate=True,
        stage_stopping_reasons=tuple(
            _STOPPING_REASON_BY_NORMALIZED_STATUS[outcome.normalized_status]
            for outcome in outcomes
        ),
    )
    return TrfOutcome(
        normalized_status=terminal.normalized_status,
        raw_status=" | ".join(outcome.raw_status for outcome in outcomes),
        success=terminal.success,
        nfev=sum(outcome.nfev for outcome in outcomes),
        njev=sum(outcome.njev for outcome in outcomes),
    )


def guard_finite_endpoint(
    outcome: TrfOutcome, values: Iterable[np.ndarray]
) -> TrfOutcome:
    """Demote a provider outcome whose published endpoint is not finite."""
    if all(bool(np.all(np.isfinite(value))) for value in values):
        return outcome
    return replace(
        outcome,
        normalized_status="failed",
        raw_status=f"nonfinite endpoint after {outcome.raw_status}",
        success=False,
    )


def _placed(value: np.ndarray) -> jax.Array:
    """Place a host array on the runtime device with its own dtype (H2D owner)."""
    return explicit_device_array(value, dtype=value.dtype)


def solve_jax_residual(
    residual: Callable[[jax.Array], jax.Array],
    initial: np.ndarray,
    *,
    max_nfev: int | None,
) -> OptimizeResult:
    """Run SciPy's default TRF/2-point policy through explicit JAX transfers."""
    compiled = jax.jit(residual)

    def host_residual(parameters: np.ndarray) -> np.ndarray:
        with allow_host_transfers():
            return host_array(compiled(_placed(parameters)))

    if max_nfev is None:
        return least_squares(host_residual, initial)
    return least_squares(host_residual, initial, max_nfev=max_nfev)


def value_and_jacobian(
    residual: Callable[[jax.Array], jax.Array], parameters: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Measure an endpoint's JAX residual and exact observable Jacobian."""
    with allow_host_transfers():
        value, jacobian = jax.jit(lambda x: (residual(x), jax.jacfwd(residual)(x)))(
            _placed(parameters)
        )
        return (
            host_array(value, dtype=np.float64),
            host_array(jacobian, dtype=np.float64),
        )


def quadratic_residual(
    targets: np.ndarray, weights: np.ndarray
) -> Callable[[jax.Array], jax.Array]:
    with allow_host_transfers():
        targets_device = _placed(targets)
        weights_device = _placed(weights)

    def residual(parameters: jax.Array) -> jax.Array:
        return weighted_quadratic_residuals(parameters, targets_device, weights_device)

    return residual


def curve_length_residual(
    full_dofs: np.ndarray,
    quadpoints: np.ndarray,
    free_positions: np.ndarray,
    *,
    order: int,
    nfp: int,
    stellsym: bool,
) -> Callable[[jax.Array], jax.Array]:
    with allow_host_transfers():
        full_device = _placed(full_dofs)
        positions_device = _placed(free_positions)
        spec = make_curve_rzfourier_spec(
            dofs=full_device,
            quadpoints=_placed(quadpoints),
            order=order,
            nfp=nfp,
            stellsym=stellsym,
        )
        fixed_device = full_device.at[positions_device].set(0.0)

    def residual(parameters: jax.Array) -> jax.Array:
        current = fixed_device.at[positions_device].set(parameters)
        current_spec = curve_spec_with_dofs(spec, current)
        length = curve_length_pure(curve_incremental_arclength_from_spec(current_spec))
        return jnp.reshape(length, (1,))

    return residual


def surface_area_volume_residual(
    full_dofs: np.ndarray,
    quadpoints_phi: np.ndarray,
    quadpoints_theta: np.ndarray,
    free_positions: np.ndarray,
    targets: np.ndarray,
    *,
    mpol: int,
    ntor: int,
    nfp: int,
    stellsym: bool,
) -> Callable[[jax.Array], jax.Array]:
    with allow_host_transfers():
        full_device = _placed(full_dofs)
        positions_device = _placed(free_positions)
        targets_device = _placed(targets)
        spec = surface_rz_fourier_spec_from_dofs(
            full_device,
            quadpoints_phi=_placed(quadpoints_phi),
            quadpoints_theta=_placed(quadpoints_theta),
            mpol=mpol,
            ntor=ntor,
            nfp=nfp,
            stellsym=stellsym,
        )
        fixed_device = full_device.at[positions_device].set(0.0)

    def residual(parameters: jax.Array) -> jax.Array:
        current = fixed_device.at[positions_device].set(parameters)
        return (
            jnp.stack(
                (
                    surface_rz_fourier_area_from_dofs(spec, current),
                    surface_rz_fourier_volume_from_dofs(spec, current),
                )
            )
            - targets_device
        )

    return residual
