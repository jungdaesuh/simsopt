"""Every curve-dependent term's analytic slope must describe its own objective.

Upstream ``9e027eac3`` stores the four ``CurvePlanarFourier`` Jacobians in the
persistent cache (``src/simsoptpp/curveplanarfourier.h:94-105``), which survives
``invalidate_cache()``.  That is sound only for a curve linear in its dofs, and
the planar curve rotates by a normalized quaternion, so upstream reuses the
Jacobian of the first evaluated state for the whole run.  Branch commit
``92ba74788`` moved those four blocks to the ordinary cache.

``tests/geo/test_curveplanarfourier_jacobian_cache.py`` checks the four Jacobians
at the curve level.  This file is the objective-level regression, at the state
where the defect is visible in the science: the OFFICIAL stage-1 end state of
``examples/2_Intermediate/stage_two_optimization_planar_coils.py``, read from the
tracked fixture ``examples/jax/parity/official_reference``.  For each
curve-dependent term of that script's objective the analytic directional slope is
compared with a central difference of the term's own value over the step ladder
1e-4 .. 1e-7.

Acceptance is derived, not chosen.  A central difference obeys
``D(h) - s = T h^2 + O(h^4)`` plus rounding: the truncation coefficient ``T`` is
estimated from the two coarsest steps, where truncation dominates, and the
rounding floor is the WORST-CASE summation model of
``tests/official_state_budget.py`` -- ``2 (N - 1) u S`` for the two evaluations
of one central difference, divided by the ``2 h`` that difference divides by,
i.e. ``(N - 1) u S / h``.  The worst-case model, and not the probabilistic one
the planar replay uses for a single agreement check, because this floor has to
hold at EVERY step of a ladder and for every one of its evaluations, not with
probability ``1 - 1e-9`` for one of them.  ``N`` is the number of float64
quadrature accumulations the objective performs and ``S`` the objective SCALE
at the state (the sum of the absolute term values) -- ``S`` rather than the
term's own value because a penalty term that sits near zero is a difference of
scale-sized intermediates, so its rounding floor is set by the scale.  A term
passes if SOME step of the ladder satisfies
``|s - D(h)| <= T h^2 + (N - 1) u S / h``.

Measured separation at this state: the branch passes every term with at least a
five-hundredfold margin (worst 2.0e-3, on the mean-squared-curvature penalty),
while the pre-fix library misses the bound by 34x, 341x and 1152x on squared
flux, the mean-squared-curvature penalty and the Lp curvature penalty
(re-measured under this file's acceptance by replaying the recorded ladders of
both builds through :func:`check_from_ladder`).  ``curve_curve_distance``, ``curve_surface_distance`` and
``linking_number`` are exactly inactive here -- value, slope and every central
difference are exactly 0.0 -- so they are asserted for completeness and carry no
evidence.

This file is the KEEP-PATH regression: it holds if and only if the branch keeps
``92ba74788``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from examples.jax.parity.cases.native_stage_two_optimization_planar_coils import (
    NativePlanarEvaluator,
    _scale_configuration,
    build_native_evaluator,
)
from examples.jax.parity.official_reference import load_official_reference

# venv site-packages/tests shadows the repo tests package, so the helper is
# imported as a top-level module from the tests/ directory.
_TESTS_ROOT = str(Path(__file__).resolve().parents[1])
if _TESTS_ROOT not in sys.path:
    sys.path.append(_TESTS_ROOT)
from official_state_budget import worst_case_summation_budget  # noqa: E402

CASE_ID = "native-stage-two-optimization-planar-coils"
OFFICIAL_STATE_KEY = "stage1:parameters"
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
    evaluator: NativePlanarEvaluator,
    parameters: np.ndarray,
    direction: np.ndarray,
) -> dict[str, TermSlopeCheck]:
    """Ladder every curve-dependent term of the case's own stage-1 objective."""
    owner = evaluator.first_stage_objective()
    terms = evaluator.curve_dependent_terms()
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
            + worst_case_summation_budget(summand_count) * scale / (2.0 * step),
        )
        for step in FINITE_DIFFERENCE_STEPS
    ]
    return min(candidates, key=lambda candidate: candidate.margin)


@pytest.fixture(scope="module")
def checks() -> dict[str, TermSlopeCheck]:
    evaluator = build_native_evaluator(_scale_configuration("native_default"))
    reference = load_official_reference(CASE_ID)
    parameters = reference.array(OFFICIAL_STATE_KEY)
    direction = np.random.default_rng(DIRECTION_SEED).standard_normal(parameters.size)
    return slope_checks(evaluator, parameters, direction / np.linalg.norm(direction))


def test_the_term_inventory_is_the_objective_s_own(
    checks: dict[str, TermSlopeCheck],
) -> None:
    """The ladder covers exactly the terms the case's objective sums."""
    assert tuple(checks) == TERM_NAMES


@pytest.mark.parametrize("name", TERM_NAMES)
def test_term_slope_matches_central_differences_at_the_official_stage1_end(
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
