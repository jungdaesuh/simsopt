"""CurvePlanarFourier Jacobians must follow the dofs, not the first evaluation.

``Curve<Array>`` in ``src/simsoptpp/curve.h`` keeps two caches: ``cache``, which
``invalidate_cache()`` clears on every dof change, and ``cache_persistent``,
which it never clears.  The persistent one is only sound for a curve whose
position is *linear* in its dofs, where the Jacobian is a constant --
``CurveXYZFourier`` and ``CurveRZFourier``.  ``CurvePlanarFourier`` applies a
quaternion rotation normalized by ``inv_magnitude()``, so all four of its
``d*_by_dcoeff`` blocks are functions of the quaternion dofs and must live in
the ordinary cache.

These tests compare each Jacobian, read back *after* a dof change, against
centered finite differences of the corresponding position derivative.  A kernel
built from a header that routes them through the persistent cache reuses the
Jacobian of the first dof vector ever queried and is wrong by 62-89 % of the
Jacobian's own magnitude; a kernel built from the ordinary cache agrees with
finite differences to ~1e-10 relative.  The tolerance below sits three decades
above that finite-difference floor and seven below the defect.
"""

from __future__ import annotations

import numpy as np
import pytest

from simsopt.geo.curveplanarfourier import CurvePlanarFourier

ORDER = 3
NUMQUADPOINTS = 40
STEP = 1.0e-6
RELATIVE_TOLERANCE = 1.0e-7

FIRST_DOFS = np.array(
    [1.0, 0.1, -0.05, 0.02, 0.03, -0.01, 0.004, 0.9, 0.2, -0.3, 0.1, 0.5, -0.2, 0.3]
)
SECOND_DOFS = FIRST_DOFS + 0.3 * np.random.default_rng(0).standard_normal(FIRST_DOFS.size)

JACOBIAN_OF = {
    "gamma": "dgamma_by_dcoeff",
    "gammadash": "dgammadash_by_dcoeff",
    "gammadashdash": "dgammadashdash_by_dcoeff",
    "gammadashdashdash": "dgammadashdashdash_by_dcoeff",
}


def _curve_at(dofs: np.ndarray) -> CurvePlanarFourier:
    curve = CurvePlanarFourier(NUMQUADPOINTS, ORDER)
    curve.x = np.asarray(dofs, dtype=np.float64)
    return curve


def _finite_difference_jacobian(value_name: str, dofs: np.ndarray) -> np.ndarray:
    """Centered differences of ``value_name`` with respect to every dof."""
    jacobian = None
    for index in range(dofs.size):
        forward, backward = dofs.copy(), dofs.copy()
        forward[index] += STEP
        backward[index] -= STEP
        derivative = (
            np.asarray(getattr(_curve_at(forward), value_name)())
            - np.asarray(getattr(_curve_at(backward), value_name)())
        ) / (2.0 * STEP)
        if jacobian is None:
            jacobian = np.zeros((*derivative.shape, dofs.size))
        jacobian[..., index] = derivative
    return jacobian


@pytest.mark.parametrize(("value_name", "jacobian_name"), sorted(JACOBIAN_OF.items()))
def test_jacobian_read_after_a_dof_change_matches_finite_differences(
    value_name: str, jacobian_name: str
) -> None:
    curve = _curve_at(FIRST_DOFS)
    getattr(curve, jacobian_name)()  # first query fills the cache at FIRST_DOFS
    curve.x = SECOND_DOFS
    reused = np.asarray(getattr(curve, jacobian_name)(), dtype=np.float64)

    reference = _finite_difference_jacobian(value_name, SECOND_DOFS)
    relative_error = float(
        np.max(np.abs(reused - reference)) / np.max(np.abs(reference))
    )

    assert relative_error < RELATIVE_TOLERANCE, (
        f"{jacobian_name} read back after the dofs moved differs from centered "
        f"differences of {value_name} by {relative_error:.3e} of its own "
        "magnitude; the kernel is reusing the Jacobian of the dof vector that "
        "first filled the cache"
    )


@pytest.mark.parametrize(("value_name", "jacobian_name"), sorted(JACOBIAN_OF.items()))
def test_jacobian_depends_on_the_quaternion_dofs(
    value_name: str, jacobian_name: str
) -> None:
    """A curve built straight at the second dofs and one moved there agree."""
    del value_name
    moved = _curve_at(FIRST_DOFS)
    getattr(moved, jacobian_name)()
    moved.x = SECOND_DOFS

    built_there = _curve_at(SECOND_DOFS)
    difference = float(
        np.max(
            np.abs(
                np.asarray(getattr(moved, jacobian_name)(), dtype=np.float64)
                - np.asarray(getattr(built_there, jacobian_name)(), dtype=np.float64)
            )
        )
    )

    assert difference == 0.0, (
        f"{jacobian_name} depends on the path taken to the dofs: a curve moved to "
        f"them disagrees with one built at them by {difference:.6e}"
    )
