"""Float64 rounding models for the official-state replays, in ONE place.

A same-state replay compares two independent implementations of ONE function at
ONE state, so the only difference it may tolerate is the rounding of the
reductions each implementation performs.  This module owns that arithmetic for
every replay test, so no test carries a hand-entered tolerance and no two tests
carry two copies -- or two versions -- of the rule.

Two models ship here, because the replays need both and they are not
interchangeable.  For a float64 reduction over ``n`` contributions whose
absolute values sum to ``S``:

``worst_case_summation_budget``
    ``(n - 1) u S`` per evaluation (Higham, *Accuracy and Stability of Numerical
    Algorithms*, 2nd ed., section 4.2).  It assumes nothing beyond float64
    arithmetic, and it is the bound to use when the reduction's own ``S`` is not
    measured: there the factor ``(n - 1)`` stands in for ``lambda sqrt(n)``
    times the cancellation amplification ``S / |value|`` the caller cannot
    compute.  It is also the model a finite-difference rounding floor needs,
    since that floor must hold for every evaluation of a ladder.

``probabilistic_summation_budget``
    ``lambda sqrt(n) u S`` per evaluation, with probability at least
    ``1 - EXCEEDANCE_PROBABILITY``, by Hoeffding on the ``n`` roundings modelled
    as independent, bounded and mean-zero (Higham and Mary, *SIAM J. Sci.
    Comput.* 41(5), 2019).  It is the bound to use when the caller pairs it with
    the ``S`` of the SAME sum, measured at the state.

Both are returned already doubled: a replay compares two independently rounded
evaluations -- two lanes at one state, or the two evaluations of a central
difference -- and each carries its own error.  Both are RELATIVE: the caller
multiplies by the ``S`` of the reduction it is bounding.
"""

from __future__ import annotations

import math

import numpy as np

#: float64 unit roundoff.
# Higham's unit roundoff u is HALF the machine epsilon (review finding r3-lens2-s2w-4).
UNIT_ROUNDOFF = float(np.finfo(np.float64).eps) / 2.0

#: Two independently rounded evaluations are compared, each with its own error.
#: Two lanes at one state, or the plus and the minus evaluation of a central
#: difference; the arithmetic is the same and the factor is the same.
COMPARED_EVALUATIONS = 2.0

#: Probability the probabilistic model below is allowed to exceed.  Fixed here
#: once, for every test, before any headroom was computed.
EXCEEDANCE_PROBABILITY = 1.0e-9

#: ``lambda`` of ``lambda sqrt(n) u S``: Hoeffding at that exceedance
#: probability, i.e. ``sqrt(2 ln(2 / p))``.
ROUNDING_FACTOR = math.sqrt(2.0 * math.log(2.0 / EXCEEDANCE_PROBABILITY))


def worst_case_summation_budget(term_count: int) -> float:
    """Relative budget for two float64 reductions over ``term_count`` terms.

    The deterministic model ``2 (n - 1) u``: it holds for every input, and it is
    the one to use when the reduction's own ``S`` is unmeasured, because the
    factor it carries over ``lambda sqrt(n)`` -- 256x at ``n = 2.8e6`` -- is
    what stands in for the unmeasured cancellation amplification.
    """
    if term_count < 2:
        raise ValueError("a summation budget needs at least two terms")
    return COMPARED_EVALUATIONS * float(term_count - 1) * UNIT_ROUNDOFF


def probabilistic_summation_budget(term_count: int) -> float:
    """Relative budget ``2 lambda sqrt(n) u`` for two float64 reductions.

    Valid with probability at least ``1 - EXCEEDANCE_PROBABILITY`` under the
    assumption that the ``n`` roundings of one reduction are independent,
    bounded and mean-zero.  Use it only where the caller supplies the ``S`` of
    the SAME sum, measured at the state; against a value that has cancelled,
    this model is not a bound.
    """
    if term_count < 1:
        raise ValueError("a reduction has at least one contribution")
    return (
        COMPARED_EVALUATIONS
        * ROUNDING_FACTOR
        * math.sqrt(float(term_count))
        * UNIT_ROUNDOFF
    )


def biot_savart_term_count(
    *,
    surface_points: int,
    coils: int,
    curve_quadrature: int,
) -> int:
    """Float64 contributions ONE coil-field reduction runs over at one state.

    One per (surface quadrature point, coil, curve quadrature point).  That
    product is the reduction behind both kinds of quantity these replays
    compare: a coil-field objective's value and one component of its gradient
    each accumulate exactly those contributions, which is why one count serves
    both and why no caller needs a second definition.
    """
    if min(surface_points, coils, curve_quadrature) < 1:
        raise ValueError("a term count needs at least one term per factor")
    return surface_points * coils * curve_quadrature
