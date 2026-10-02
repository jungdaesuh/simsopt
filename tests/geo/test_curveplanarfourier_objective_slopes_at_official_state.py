"""Every curve-dependent term's analytic slope must describe its own objective.

Upstream ``9e027eac3`` stores the four ``CurvePlanarFourier`` Jacobians in the
persistent cache (``src/simsoptpp/curveplanarfourier.h:94-105``), which survives
``invalidate_cache()``.  That is sound only for a curve linear in its dofs, and
the planar curve rotates by a normalized quaternion, so upstream reuses the
Jacobian of the first evaluated state for the whole run.  Branch commit
``92ba74788`` moved those four blocks to the ordinary cache.

``tests/geo/test_curveplanarfourier_jacobian_cache.py`` checks the four Jacobians
at the curve level.  This file is the objective-level regression, at a state
away from the start, where the defect is visible in the science: the stage-1
end state of ``examples/2_Intermediate/stage_two_optimization_planar_coils.py``.
That state is computed live, by a native SIMSOPT run of the script's own
stage-1 objective and its own L-BFGS-B options (``maxcor=300``, ``tol=1e-15``),
capped at :data:`LIVE_STAGE_ONE_ITERATIONS` iterations so the run takes seconds
instead of the script's 400 iterations.  For each curve-dependent term of that
script's objective the analytic directional slope is compared with a central
difference of the term's own value over the step ladder 1e-4 .. 1e-7, on a
freshly built objective that is handed the end state.

Acceptance is derived, not chosen.  A central difference obeys
``D(h) - s = T h^2 + O(h^4)`` plus rounding: the truncation coefficient ``T`` is
estimated from the two coarsest steps, where truncation dominates, and the
rounding floor is the WORST-CASE summation model ``2 (N - 1) u S`` for the two
evaluations of one central difference (Higham, *Accuracy and Stability of
Numerical Algorithms*, 2nd ed., section 4.2), divided by the ``2 h`` that
difference divides by, i.e. ``(N - 1) u S / h``.  The worst-case model, and not
a probabilistic one, because this floor has to hold at EVERY step of a ladder
and for every one of its evaluations.  ``N`` is the number of float64
quadrature accumulations the objective performs and ``S`` the objective SCALE
at the state (the sum of the absolute term values) -- ``S`` rather than the
term's own value because a penalty term that sits near zero is a difference of
scale-sized intermediates, so its rounding floor is set by the scale.  A term
passes if SOME step of the ladder satisfies
``|s - D(h)| <= T h^2 + (N - 1) u S / h``.

A term that is exactly inactive at the state -- value, slope and every central
difference exactly 0.0 -- passes trivially and carries no evidence.

This file is the KEEP-PATH regression: it holds if and only if the branch keeps
``92ba74788``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize
from simsopt._core.optimizable import Optimizable
from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
from simsopt.geo import (
    CurveCurveDistance,
    CurveLength,
    CurveSurfaceDistance,
    LinkingNumber,
    LpCurveCurvature,
    MeanSquaredCurvature,
    SurfaceRZFourier,
    create_equally_spaced_planar_curves,
)
from simsopt.objectives import QuadraticPenalty, SquaredFlux

SURFACE_INPUT = (
    Path(__file__).resolve().parents[1] / "test_files" / "input.LandremanPaul2021_QA"
)
#: The script's own construction (``stage_two_optimization_planar_coils.py``).
SURFACE_RESOLUTION = 32
NUM_BASE_CURVES = 4
CURVE_ORDER = 5
CURVE_QUADRATURE = 75
LENGTH_TARGET = 10.4
FIRST_LENGTH_WEIGHT = 10.0
CURVE_CURVE_THRESHOLD = 0.08
CURVE_CURVE_WEIGHT = 1000.0
CURVE_SURFACE_THRESHOLD = 0.12
CURVE_SURFACE_WEIGHT = 10.0
CURVATURE_THRESHOLD = 10.0
CURVATURE_WEIGHT = 1.0e-6
MEAN_SQUARED_CURVATURE_THRESHOLD = 10.0
MEAN_SQUARED_CURVATURE_WEIGHT = 1.0e-6
LINKING_NUMBER_WEIGHT = 1.0
#: The script's stage-1 ``minimize(..., options={"maxcor": 300}, tol=1e-15)``.
STAGE_ONE_MAXCOR = 300
STAGE_ONE_TOL = 1.0e-15
#: The script runs 400; the cap keeps the live state computable in seconds.
LIVE_STAGE_ONE_ITERATIONS = 20
#: float64 unit roundoff, HALF the machine epsilon.
UNIT_ROUNDOFF = float(np.finfo(np.float64).eps) / 2.0

DIRECTION_SEED = 20260920
FINITE_DIFFERENCE_STEPS = (1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7)
TERM_NAMES = (
    "squared_flux",
    "length_penalty",
    "curve_curve_distance",
    "curve_surface_distance",
    "lp_curvature",
    "mean_squared_curvature_penalty",
    "linking_number",
)


@dataclass(frozen=True)
class _PlanarStageOne:
    """The script's stage-1 objective and the terms it sums, in script order."""

    surface: SurfaceRZFourier
    coils: tuple[Coil, ...]
    objective: Optimizable
    terms: dict[str, Optimizable]

    @property
    def quadrature_summand_count(self) -> int:
        """Float64 accumulations this objective's quadratures perform.

        Surface quadrature points plus every coil's quadrature points, three
        Cartesian components each: the length of the longest sum any term forms.
        """
        return int(
            np.asarray(self.surface.gamma()).size
            + sum(np.asarray(coil.curve.gamma()).size for coil in self.coils)
        )


def _planar_stage_one() -> _PlanarStageOne:
    surface = SurfaceRZFourier.from_vmec_input(
        str(SURFACE_INPUT),
        range="half period",
        nphi=SURFACE_RESOLUTION,
        ntheta=SURFACE_RESOLUTION,
    )
    surface.fix_all()
    base_curves = create_equally_spaced_planar_curves(
        NUM_BASE_CURVES,
        surface.nfp,
        stellsym=True,
        R0=1.0,
        R1=0.5,
        order=CURVE_ORDER,
        numquadpoints=CURVE_QUADRATURE,
    )
    base_currents = [Current(1.0e5) for _ in base_curves]
    base_currents[0].fix_all()
    coils = coils_via_symmetries(base_curves, base_currents, surface.nfp, True)
    field = BiotSavart(coils)
    field.set_points(surface.gamma().reshape((-1, 3)))
    curves = [coil.curve for coil in coils]
    terms: dict[str, Optimizable] = {
        "squared_flux": SquaredFlux(surface, field),
        "length_penalty": QuadraticPenalty(
            sum(CurveLength(curve) for curve in base_curves), LENGTH_TARGET
        ),
        "curve_curve_distance": CurveCurveDistance(
            curves, CURVE_CURVE_THRESHOLD, num_basecurves=NUM_BASE_CURVES
        ),
        "curve_surface_distance": CurveSurfaceDistance(
            curves, surface, CURVE_SURFACE_THRESHOLD
        ),
        "lp_curvature": sum(
            LpCurveCurvature(curve, 2, CURVATURE_THRESHOLD) for curve in base_curves
        ),
        "mean_squared_curvature_penalty": sum(
            QuadraticPenalty(
                MeanSquaredCurvature(curve), MEAN_SQUARED_CURVATURE_THRESHOLD
            )
            for curve in base_curves
        ),
        "linking_number": LinkingNumber(curves),
    }
    objective = (
        terms["squared_flux"]
        + FIRST_LENGTH_WEIGHT * terms["length_penalty"]
        + CURVE_CURVE_WEIGHT * terms["curve_curve_distance"]
        + CURVE_SURFACE_WEIGHT * terms["curve_surface_distance"]
        + CURVATURE_WEIGHT * terms["lp_curvature"]
        + MEAN_SQUARED_CURVATURE_WEIGHT * terms["mean_squared_curvature_penalty"]
        + LINKING_NUMBER_WEIGHT * terms["linking_number"]
    )
    return _PlanarStageOne(
        surface=surface, coils=tuple(coils), objective=objective, terms=terms
    )


def _native_stage_one_end_state() -> np.ndarray:
    """Run the script's stage-1 L-BFGS-B natively and return where it stopped."""
    objective = _planar_stage_one().objective

    def value_and_gradient(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        objective.x = parameters
        return float(objective.J()), np.asarray(objective.dJ(), dtype=np.float64)

    result = minimize(
        value_and_gradient,
        np.asarray(objective.x, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": LIVE_STAGE_ONE_ITERATIONS, "maxcor": STAGE_ONE_MAXCOR},
        tol=STAGE_ONE_TOL,
    )
    return np.asarray(result.x, dtype=np.float64)


def _worst_case_summation_budget(term_count: int) -> float:
    """Relative budget ``2 (n - 1) u`` for two float64 reductions over ``n`` terms."""
    if term_count < 2:
        raise ValueError("a summation budget needs at least two terms")
    return 2.0 * float(term_count - 1) * UNIT_ROUNDOFF


@dataclass(frozen=True)
class TermSlopeCheck:
    """One term's best step on the ladder and the bound derived for that step."""

    name: str
    value: float
    analytic_slope: float
    truncation_coefficient: float
    step: float
    central_difference: float
    gap: float
    acceptance: float

    @property
    def margin(self) -> float:
        """``gap / acceptance``; a term passes when this is at most one."""
        if self.acceptance == 0.0:
            return 0.0 if self.gap == 0.0 else float("inf")
        return self.gap / self.acceptance


def slope_checks(
    evaluator: _PlanarStageOne,
    parameters: np.ndarray,
    direction: np.ndarray,
) -> dict[str, TermSlopeCheck]:
    """Ladder every curve-dependent term of the script's own stage-1 objective."""
    owner = evaluator.objective
    terms = evaluator.terms
    summand_count = evaluator.quadrature_summand_count

    owner.x = parameters
    values = {name: float(term.J()) for name, term in terms.items()}
    slopes = {
        name: float(
            np.asarray(term.dJ(partials=True)(owner), dtype=np.float64) @ direction
        )
        for name, term in terms.items()
    }
    scale = sum(abs(value) for value in values.values())

    ladder: dict[float, dict[str, float]] = {}
    for step in FINITE_DIFFERENCE_STEPS:
        owner.x = parameters + step * direction
        plus = {name: float(term.J()) for name, term in terms.items()}
        owner.x = parameters - step * direction
        minus = {name: float(term.J()) for name, term in terms.items()}
        ladder[step] = {
            name: (plus[name] - minus[name]) / (2.0 * step) for name in terms
        }

    return {
        name: check_from_ladder(
            name=name,
            value=values[name],
            analytic_slope=slopes[name],
            central_differences={
                step: ladder[step][name] for step in FINITE_DIFFERENCE_STEPS
            },
            summand_count=summand_count,
            scale=scale,
        )
        for name in terms
    }


def check_from_ladder(
    *,
    name: str,
    value: float,
    analytic_slope: float,
    central_differences: dict[float, float],
    summand_count: int,
    scale: float,
) -> TermSlopeCheck:
    """The derived acceptance rule, applied to one term's finished ladder.

    Kept separate from the evaluation so that the same arithmetic can be replayed
    against a ladder recorded on another build.
    """
    coarse, finer = FINITE_DIFFERENCE_STEPS[0], FINITE_DIFFERENCE_STEPS[1]
    truncation = abs(
        (central_differences[coarse] - central_differences[finer])
        / (coarse**2 - finer**2)
    )
    candidates = [
        TermSlopeCheck(
            name=name,
            value=value,
            analytic_slope=analytic_slope,
            truncation_coefficient=truncation,
            step=step,
            central_difference=central_differences[step],
            gap=abs(analytic_slope - central_differences[step]),
            acceptance=truncation * step * step
            + _worst_case_summation_budget(summand_count) * scale / (2.0 * step),
        )
        for step in FINITE_DIFFERENCE_STEPS
    ]
    return min(candidates, key=lambda candidate: candidate.margin)


@pytest.fixture(scope="module")
def checks() -> dict[str, TermSlopeCheck]:
    parameters = _native_stage_one_end_state()
    evaluator = _planar_stage_one()
    direction = np.random.default_rng(DIRECTION_SEED).standard_normal(parameters.size)
    return slope_checks(evaluator, parameters, direction / np.linalg.norm(direction))


def test_the_term_inventory_is_the_objective_s_own(
    checks: dict[str, TermSlopeCheck],
) -> None:
    """The ladder covers exactly the terms the script's objective sums."""
    assert tuple(checks) == TERM_NAMES


@pytest.mark.parametrize("name", TERM_NAMES)
def test_term_slope_matches_central_differences_at_the_native_stage1_end(
    checks: dict[str, TermSlopeCheck],
    name: str,
) -> None:
    check = checks[name]

    assert check.margin <= 1.0, (
        f"{name}: analytic slope {check.analytic_slope:+.10e} against its own "
        f"central difference {check.central_difference:+.10e} at step "
        f"{check.step:.0e} differs by {check.gap:.3e}, over the derived bound "
        f"{check.acceptance:.3e} (truncation coefficient "
        f"{check.truncation_coefficient:.3e}, term value {check.value:.6e}); the "
        "CurvePlanarFourier Jacobians are not following the dofs"
    )
