"""Official Boozer example settings and single-stage entries.

``2_Intermediate/boozer.py`` runs three stages on one surface: an L-BFGS-B
reduction of the area-constrained Boozer penalty, a manual Levenberg-Marquardt
polish of the same problem, and a manual Levenberg-Marquardt solve after the
label is switched to toroidal flux with three times the converged flux.  Every
stage is reached through the public ``BoozerSurface`` API, which
``BoozerSurfaceJAX`` mirrors, so one entry per stage serves both the native
library route and the JAX route and the two can never drift apart here.

The stage entries take an already-constructed solver (its label and target are
the caller's choice, exactly as in the official script) and a start state, so a
comparator can replay one stage from a recorded state without re-running the
stages before it.

``2_Intermediate/boozerQA.py`` has no stage entry -- it is a single outer solve
-- but its official settings and its outer objective's weights live here for the
same reason: every caller reads one copy.

This module is importable from a plain source checkout (only ``src`` is on the
path when an example script runs), which the ``examples`` package is not.
"""

from __future__ import annotations

from typing import Final, NamedTuple

import numpy as np

from simsopt_contracts.optimization_endpoint import StoppingReason
from simsopt_jax.runtime.host_boundary import host_array, host_bool, host_float
from simsopt_jax.solve import Driver

# Official settings of ``examples/2_Intermediate/boozer.py`` at upstream
# 9e027eac38028d57aa23777be52a781aa860e347 (lines 27-63 there).
OFFICIAL_SURFACE_RESOLUTION = 5
OFFICIAL_SURFACE_DISTANCE = 0.10
OFFICIAL_INITIAL_IOTA = -0.4
OFFICIAL_SOLVER_TOLERANCE = 1.0e-10
OFFICIAL_CONSTRAINT_WEIGHT = 100.0
OFFICIAL_LBFGS_MAXITER = 300
OFFICIAL_LS_MAXITER = 100
OFFICIAL_FLUX_MULTIPLIER = 3.0

# Official settings of ``examples/2_Intermediate/boozerQA.py`` at the same
# upstream commit (lines 49-60, 68, 77-83 and 143-145 there).  The four penalty
# terms are summed with unit weight (``JF = J_nonQSRatio + J_iotas +
# J_major_radius + Jls``) and the Boozer residual is not a term of the official
# objective, hence weight 0.  ``sDIM=20`` is the upstream default of
# ``NonQuasiSymmetricRatio`` (geo/surfaceobjectives.py:711).  The outer solve is
# ``minimize(..., method='BFGS', options={'maxiter': 1e3}, tol=1e-15)``; its
# gradient tolerance is owned by ``simsopt_jax.examples.single_stage_boozer_vacuum``
# (``OUTER_GRADIENT_TOLERANCE``) and is not restated here.
OFFICIAL_QA_SURFACE_RESOLUTION = 6
OFFICIAL_QA_SURFACE_DISTANCE = 0.10
OFFICIAL_QA_INITIAL_IOTA = -0.406
OFFICIAL_QA_NEWTON_TOLERANCE = 1.0e-13
OFFICIAL_QA_NEWTON_MAXITER = 20
OFFICIAL_QA_OUTER_MAXITER = 1000
OFFICIAL_QA_LINE_SEARCH_MAXITER = 20
OFFICIAL_QA_NON_QS_RESOLUTION = 20
OFFICIAL_QA_PENALTY_WEIGHT = 1.0
OFFICIAL_QA_RESIDUAL_WEIGHT = 0.0
#: Upstream drives the outer solve with SciPy's dense BFGS, so the mirror's
#: outer method is fixed by the official workflow and never follows the
#: execution mode's default driver.  ``minimize_bfgs_host_core`` is this
#: driver's host core.
OFFICIAL_QA_OUTER_DRIVER = Driver.SIMSOPT_BFGS
#: Upstream's own BoozerQA outer solve ends on its iteration budget
#: (``MAXITER = 1e3`` with ``tol=1e-15``, boozerQA.py:143-145; the official
#: capture records status 1 after 1000 iterations, "Maximum number of
#: iterations has been exceeded."), so a budget exit is the OFFICIAL endpoint
#: mode and a mirror that reaches it is healthy.  Every other non-converged
#: mode -- a failed line search, a non-finite endpoint, a status outside the
#: emitter's contract -- is not, and must not be reported as a success.
OFFICIAL_QA_ADMISSIBLE_OUTER_STOPPING_REASONS: Final[frozenset[StoppingReason]] = (
    frozenset({"converged", "iteration-limit"})
)


class BoozerStageState(NamedTuple):
    """Surface degrees of freedom, rotational transform and poloidal current."""

    surface_dofs: np.ndarray
    iota: float
    G: float


class BoozerStageOutcome(NamedTuple):
    """One stage's end state and the provider's own report of that stage.

    ``state`` is what the provider actually left behind -- the surface's own
    degrees of freedom plus the ``iota`` and ``G`` of the provider's result --
    and it is what the next stage is chained from, whatever the provider chose
    to persist.

    ``provider_persisted_iterate`` says whether the provider moved the state at
    all: it is ``True`` exactly when the end state differs from the stage start.
    Upstream ALWAYS persists the last iterate (``simsopt/geo/boozersurface.py``
    at 9e027eac3: :603-618 for ``method='manual'`` and :449-456 for the L-BFGS
    branch write the surface dofs, ``iota`` and ``G`` unconditionally), so on
    upstream this is ``False`` only for a stage that took no step.  The
    repository's native library adds ``_boozer_iterate_is_persistable``
    (src/simsopt/geo/boozersurface.py:36-42) and restores the stage start when a
    stage failed without reducing the residual norm; ``BoozerSurfaceJAX`` has no
    such guard and commits as upstream does.  This flag is how a failed stage's
    two possible end states are told apart instead of being silently compared.

    ``status``, ``message``, ``nit``, ``nfev``, ``njev`` and ``objective`` are
    ``None`` on the manual Levenberg-Marquardt route because upstream's loop
    reports none of them; the official capture records ``iter: null`` for the
    same reason.  ``gradient_norm`` is the norm of the penalty gradient at the
    end state on both routes, and ``penalty_residual_norm`` is the norm of the
    solver's own residual vector (with its constraint rows), which the manual
    route alone returns.
    """

    state: BoozerStageState
    provider_persisted_iterate: bool
    success: bool
    status: int | None
    message: str | None
    nit: int | None
    nfev: int | None
    njev: int | None
    objective: float | None
    gradient_norm: float
    penalty_residual_norm: float | None


def boozer_first_stage_budget(*, least_squares_steps: int, native_default: bool) -> int:
    """Return the first stage's L-BFGS-B budget for one execution scale.

    At ``native_default`` the stage runs upstream's own budget,
    ``OFFICIAL_LBFGS_MAXITER``, exactly as stages two and three run
    ``OFFICIAL_LS_MAXITER`` there: the runner's step count does not move an
    official budget.  At a reduced scale the stage keeps the official ratio
    between the two budgets applied to the reduced least-squares step count.
    Every caller takes the first-stage budget from here, so none can drift off
    upstream's 300 while another keeps it.
    """
    if native_default:
        return OFFICIAL_LBFGS_MAXITER
    return least_squares_steps * OFFICIAL_LBFGS_MAXITER // OFFICIAL_LS_MAXITER


def boozer_official_options(
    *, rough_maxiter: int, ls_maxiter: int, tolerance: float
) -> dict[str, object]:
    """Select the typed on-device L-BFGS-B analogue and official budgets."""
    return {
        "inner_driver": Driver.SIMSOPT_LBFGSB,
        "bfgs_maxiter": rough_maxiter,
        "bfgs_tol": tolerance,
        "maxcor": 200,
        "ftol": tolerance,
        "maxfun": 15000,
        "maxls": 20,
        "newton_maxiter": ls_maxiter,
        "newton_tol": tolerance,
        "verbose": False,
    }


def _apply_stage_start(solver, start: BoozerStageState) -> None:
    solver.surface.set_dofs(np.asarray(start.surface_dofs, dtype=np.float64))
    solver.need_to_run_code = True


def _end_state(solver, result) -> BoozerStageState:
    """Read back the state the provider left behind, persisted or reverted."""
    return BoozerStageState(
        surface_dofs=np.asarray(solver.surface.get_dofs(), dtype=np.float64),
        iota=host_float(result["iota"]),
        G=host_float(result["G"]),
    )


def _persisted_iterate(end: BoozerStageState, start: BoozerStageState) -> bool:
    """True when the provider left a state other than the stage start."""
    return not (
        end.iota == start.iota
        and end.G == start.G
        and np.array_equal(
            end.surface_dofs,
            np.asarray(start.surface_dofs, dtype=np.float64),
        )
    )


def _host_norm(vector) -> float:
    return float(np.linalg.norm(host_array(vector, dtype=np.float64)))


def run_boozer_lbfgs_stage(
    solver,
    start: BoozerStageState,
    *,
    tol: float = OFFICIAL_SOLVER_TOLERANCE,
    maxiter: int = OFFICIAL_LBFGS_MAXITER,
    constraint_weight: float = OFFICIAL_CONSTRAINT_WEIGHT,
) -> BoozerStageOutcome:
    """Run the official first stage from ``start`` on ``solver``'s label."""
    _apply_stage_start(solver, start)
    result = solver.minimize_boozer_penalty_constraints_LBFGS(
        tol=tol,
        maxiter=maxiter,
        constraint_weight=constraint_weight,
        iota=start.iota,
        G=start.G,
    )
    provider = result["info"]
    end = _end_state(solver, result)
    return BoozerStageOutcome(
        state=end,
        provider_persisted_iterate=_persisted_iterate(end, start),
        success=host_bool(result["success"]),
        status=int(provider.status),
        message=str(provider.message),
        nit=int(result["iter"]),
        nfev=int(provider.nfev),
        njev=int(provider.njev),
        objective=host_float(result["fun"]),
        gradient_norm=_host_norm(result["gradient"]),
        penalty_residual_norm=None,
    )


def run_boozer_manual_stage(
    solver,
    start: BoozerStageState,
    *,
    tol: float = OFFICIAL_SOLVER_TOLERANCE,
    maxiter: int = OFFICIAL_LS_MAXITER,
    constraint_weight: float = OFFICIAL_CONSTRAINT_WEIGHT,
) -> BoozerStageOutcome:
    """Run one official ``method='manual'`` stage from ``start``."""
    _apply_stage_start(solver, start)
    result = solver.minimize_boozer_penalty_constraints_ls(
        tol=tol,
        maxiter=maxiter,
        constraint_weight=constraint_weight,
        iota=start.iota,
        G=start.G,
        method="manual",
    )
    end = _end_state(solver, result)
    return BoozerStageOutcome(
        state=end,
        provider_persisted_iterate=_persisted_iterate(end, start),
        success=host_bool(result["success"]),
        status=None,
        message=None,
        nit=None,
        nfev=None,
        njev=None,
        objective=None,
        gradient_norm=_host_norm(result["gradient"]),
        penalty_residual_norm=_host_norm(result["residual"]),
    )


def boozer_qa_outer_objective_config(
    *,
    nfp: int,
    non_qs_resolution: int,
    length_target: float,
    major_radius_target: float,
    vessel_gamma: np.ndarray,
    residual_weight: float = OFFICIAL_QA_RESIDUAL_WEIGHT,
) -> dict[str, object]:
    """Build the official BoozerQA outer objective: four unit-weight terms.

    Upstream sums ``NonQuasiSymmetricRatio``, the ``Iotas`` penalty, the
    ``MajorRadius`` penalty and the total-base-coil-length penalty with unit
    weight and nothing else (boozerQA.py:77-83), so every other term the
    traceable objective supports carries weight 0 and its threshold is inert.
    ``residual_weight`` is a parameter because the single-stage variants of this
    objective do report the Boozer residual; the official BoozerQA value is 0.
    """
    return {
        "non_qs_weight": OFFICIAL_QA_PENALTY_WEIGHT,
        "residual_weight": residual_weight,
        "iota_weight": OFFICIAL_QA_PENALTY_WEIGHT,
        "major_radius_weight": OFFICIAL_QA_PENALTY_WEIGHT,
        "length_weight": OFFICIAL_QA_PENALTY_WEIGHT,
        "curvature_weight": 0.0,
        "curve_curve_weight": 0.0,
        "curve_surface_weight": 0.0,
        "surface_vessel_weight": 0.0,
        "non_qs_quadpoints_phi": np.linspace(
            0.0,
            1.0 / nfp,
            2 * non_qs_resolution,
            endpoint=False,
            dtype=np.float64,
        ),
        "non_qs_quadpoints_theta": np.linspace(
            0.0,
            1.0,
            2 * non_qs_resolution,
            endpoint=False,
            dtype=np.float64,
        ),
        "non_qs_axis": 0,
        "optimized_coil_index": 0,
        "length_coil_indices": (0, 1, 2),
        "length_target": length_target,
        "curvature_threshold": 0.0,
        "curvature_p_norm": 2.0,
        "major_radius_target": major_radius_target,
        "curve_curve_threshold": 0.0,
        "curve_surface_threshold": 0.0,
        "vessel_gamma": np.asarray(vessel_gamma, dtype=np.float64),
        "surface_vessel_threshold": 0.0,
    }
