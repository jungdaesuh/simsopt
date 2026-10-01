"""The planar-coils mirror claim against upstream, stated as what is provable.

Upstream ``9e027eac3`` keeps the four ``CurvePlanarFourier`` Jacobians in the
PERSISTENT cache (``src/simsoptpp/curveplanarfourier.h:94-105``), which
``invalidate_cache()`` never clears.  That cache is sound only for a curve whose
position is linear in its dofs; the planar curve rotates by a normalized
quaternion, so upstream's gradient is the gradient of the FIRST evaluated state
for the whole run.  Upstream's L-BFGS-B consequently stops by line-search
stagnation -- status 0, ``RELATIVE REDUCTION OF F <= FACTR*EPSMCH``, at 135 and
69 iterations of a 400-iteration cap -- and its end point cannot be reached by a
correct gradient.  No assertion here compares an end point with upstream's.

What is asserted is what upstream's own capture supports:

* the case starts at upstream's own start state;
* the branch native objective takes upstream's VALUE at each of upstream's three
  recorded states (start, stage-1 end, stage-2 end);
* the branch native GRADIENT equals upstream's at the start state, and only
  there -- the one state at which upstream's gradient is right.

All three hold whether or not the branch keeps the library fix ``92ba74788``,
because the gradient is compared only at the start state.  Official numbers come
from the tracked fixture ``examples/jax/parity/official_reference``; none of them
is written here as a literal.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import sys
from pathlib import Path

import numpy as np
import pytest
from examples.jax.parity.cases.native_stage_two_optimization_planar_coils import (
    NativePlanarEvaluator,
    _scale_configuration,
    build_native_evaluator,
)
from examples.jax.parity.official_reference import load_official_reference
from simsopt._core.optimizable import Optimizable

# venv site-packages/tests shadows the repo tests package, so the helper is
# imported as a top-level module from the tests/ directory.
_TESTS_ROOT = str(Path(__file__).resolve().parents[1])
if _TESTS_ROOT not in sys.path:
    sys.path.append(_TESTS_ROOT)
from official_state_budget import probabilistic_summation_budget  # noqa: E402

CASE_ID = "native-stage-two-optimization-planar-coils"

REFERENCE = load_official_reference(CASE_ID)
OFFICIAL_START = REFERENCE.array("initial:parameters")


def surface_quadrature_points(evaluator: NativePlanarEvaluator) -> int:
    """Contributions the flux quadrature reduces: one per surface point."""
    return int(np.asarray(evaluator.surface.gamma()).size // 3)


def value_relative_tolerance(evaluator: NativePlanarEvaluator) -> float:
    """Relative allowance for the objective VALUE at one official state.

    The PROBABILISTIC model of ``tests/official_state_budget.py`` -- the ``n``
    roundings of one reduction taken as independent, bounded and mean-zero --
    because ``n`` here is a reduction this file can name and ``S`` is paired
    with it from the state itself.

    Both sides run the same simsoptpp code -- the branch differs from upstream
    only in where the ``CurvePlanarFourier`` Jacobians are cached
    (``92ba74788``), which no VALUE reads -- and the Biot-Savart sum for one
    surface point is sequential in both.  The only freedom left is the ORDER of
    the flux quadrature, which ``integral_BdotN`` reduces under
    ``#pragma omp parallel for reduction(+:...)``
    (``src/simsoptpp/integral_BdotN.cpp:62``), so its partition follows the
    thread count.  So ``n`` is the number of surface quadrature points, 1024.

    ``S`` is taken as the objective VALUE, and that step is a conservative
    over-statement rather than an identity.  The flux quadrature's ``n``
    contributions are non-negative and sum to the SQUARED-FLUX TERM, not to the
    objective; the objective adds the length, clearance, curvature and topology
    terms, which are non-negative too.  Measured at the three official states,
    the flux term is 0.14 %, 99.81 % and 99.84 % of the value, so using the
    value over-states the flux sum's own ``S`` by 710x at the start state and
    by 1.002x at both end states.  That over-statement is also what covers the
    other terms' own reductions: each of them runs over at most ``n`` summands
    (75 quadrature points per curve against the surface's 1024), and the
    objective's seven-term outer sum adds those terms in the same order on both
    sides, so it contributes no error of its own -- it propagates the term
    differences, magnified by at most ``1 + 7 u``.  The one reduction that can exceed ``n`` is the
    curve-surface pair sum, and it is 0.0 at both end states; at the start
    state it is 1.3e-8 of the value, so even at its full pair count it would
    ask for 4.5e-7 of this bound.

    Measured over the three official states: bitwise equality at
    ``OMP_NUM_THREADS=1``, and at most 1.3e-18 absolute (4.4e-16 relative) at
    2, 4 and 8 threads, against this bound of 4.7e-14.
    """
    return probabilistic_summation_budget(surface_quadrature_points(evaluator))


# (state name, optimization stage, official parameter key, official value key)
OFFICIAL_STATES = (
    ("start", 1, "initial:parameters", "taylor:objective"),
    ("stage1_end", 1, "stage1:parameters", "stage1:objective"),
    ("stage2_end", 2, "stage2:parameters", "stage2:objective"),
)


@pytest.fixture(scope="module")
def evaluator() -> NativePlanarEvaluator:
    """The case's own native construction at the scale upstream itself ran."""
    return build_native_evaluator(_scale_configuration("native_default"))


def _stage_objective(evaluator: NativePlanarEvaluator, stage: int) -> Optimizable:
    if stage == 1:
        return evaluator.first_stage_objective()
    return evaluator.second_stage_objective()


def test_the_case_starts_at_the_official_start_state(
    evaluator: NativePlanarEvaluator,
) -> None:
    """Same start state, bit for bit -- otherwise nothing below compares."""
    start = np.asarray(evaluator.field.x, dtype=np.float64)

    assert start.shape == OFFICIAL_START.shape
    assert np.array_equal(start, OFFICIAL_START)


@pytest.mark.parametrize(
    ("state", "stage", "parameter_key", "value_key"),
    OFFICIAL_STATES,
    ids=[row[0] for row in OFFICIAL_STATES],
)
def test_objective_value_equals_upstream_at_upstream_states(
    evaluator: NativePlanarEvaluator,
    state: str,
    stage: int,
    parameter_key: str,
    value_key: str,
) -> None:
    objective = _stage_objective(evaluator, stage)
    objective.x = REFERENCE.array(parameter_key)
    value = float(objective.J())
    official = float(REFERENCE.scalar(value_key))
    difference = abs(value - official)
    tolerance = value_relative_tolerance(evaluator)

    assert difference <= tolerance * abs(official), (
        f"branch objective at the official {state} state is {value!r} against "
        f"upstream's {official!r}: {difference:.3e} absolute, "
        f"{difference / abs(official):.3e} relative, over the derived bound "
        f"{tolerance:.3e}"
    )


def test_objective_gradient_equals_upstream_at_the_start_state(
    evaluator: NativePlanarEvaluator,
) -> None:
    """The start state is the only state at which upstream's gradient is right.

    Upstream's stored gradients at its two end states are the start-state
    Jacobian's, so they are not compared here -- by either lane, under either
    decision about ``92ba74788``.
    """
    official_gradient = REFERENCE.array("taylor:gradient")
    objective = evaluator.first_stage_objective()
    objective.x = OFFICIAL_START
    gradient = np.asarray(objective.dJ(), dtype=np.float64)

    # This is a GATE, not a derived rounding bound. A gradient component reduces
    # SIGNED contributions over `quadrature_summand_count` float64 accumulations
    # and their `S = sum |term|` is not measured here, so the probabilistic model
    # (valid against the S of the same sum, see `official_state_budget`) does not
    # bound it; the value bound above is different, its 1024 flux contributions
    # are non-negative. The NUMBER is kept because no bound in this campaign may
    # loosen: `2 lambda sqrt(N) u` of the largest gradient entry, 12x tighter than
    # the `N eps` this line used to spell. It errs toward a false failure, and the
    # measured agreement uses 1/463 of it (3.49e-14 against 1.62e-11).
    tolerance = probabilistic_summation_budget(
        evaluator.quadrature_summand_count
    ) * float(np.max(np.abs(official_gradient)))
    difference = float(np.max(np.abs(gradient - official_gradient)))

    assert gradient.shape == official_gradient.shape
    assert difference <= tolerance, (
        f"branch gradient at the official start state differs from upstream's by "
        f"{difference:.3e} absolute on a gradient of "
        f"{float(np.max(np.abs(official_gradient))):.6e}, over the derived bound "
        f"{tolerance:.3e}"
    )
