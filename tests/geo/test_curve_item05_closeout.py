"""Item 05 closeout: curve spec / pullback parity coverage at production scale.

This module closes the two coverage gaps identified in the item 05 audit of
``src/simsopt/geo/{curve,curvexyzfourier,curverzfourier,
curvexyzfouriersymmetries,curveplanarfourier,curvehelical,curvecwsfourier,
curveperturbed}.py`` and ``src/simsopt/jax_core/curve_geometry.py``:

1. ``CurveXYZFourierSymmetries`` had no JAX-spec parity test row. Because
   the class does not implement ``to_spec()`` and ``curve_spec_from_curve``
   raises ``NotImplementedError`` for it on parent commit ``a9da18fac``,
   this file documents the architecture limitation explicitly through
   ``test_curvexyzfouriersymmetries_spec_routing_is_documented_blocker``
   rather than silently skipping or speculatively passing.
2. No existing curve-class parity fixture co-asserted ``ncoils >= 4`` AND
   ``nquadpoints >= 64`` against the spec-driven ``curve_geometry_from_dofs``
   path. The parametrized ``test_curve_spec_pullback_production_scale_parity``
   case adds the floor for ``CurveXYZFourier``, ``CurveRZFourier``,
   ``CurvePlanarFourier``, and ``CurveHelical``.

Tolerances come from the validation-ladder ``direct_kernel`` lane via
``parity_ladder_tolerances`` so no atol/rtol numeric literals appear in this
file.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.geo.curvehelical import CurveHelical
from simsopt.geo.curveplanarfourier import CurvePlanarFourier
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.curvexyzfouriersymmetries import CurveXYZFourierSymmetries
from simsopt.jax_core import curve_spec_from_curve
from simsopt.jax_core.curve_geometry import curve_geometry_from_dofs


_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]


_PRODUCTION_NCOILS = 4
_PRODUCTION_NQUADPOINTS = 64
_PRODUCTION_ORDER = 2
_PRODUCTION_RAND_SCALE = 0.01
_PRODUCTION_RNG_SEED = 7


def _make_curve_xyzfourier(order: int, nquad: int, dofs: np.ndarray):
    curve = CurveXYZFourier(nquad, order)
    curve.x = np.asarray(dofs, dtype=np.float64)
    return curve


def _make_curve_rzfourier(order: int, nquad: int, dofs: np.ndarray):
    curve = CurveRZFourier(nquad, order, 2, True)
    curve.x = np.asarray(dofs, dtype=np.float64)
    return curve


def _make_curve_planarfourier(order: int, nquad: int, dofs: np.ndarray):
    curve = CurvePlanarFourier(nquad, order)
    curve.x = np.asarray(dofs, dtype=np.float64)
    return curve


def _make_curve_helical(order: int, nquad: int, dofs: np.ndarray):
    curve = CurveHelical(nquad, order, 5, 2, 1.0, 0.3)
    curve.x = np.asarray(dofs, dtype=np.float64)
    return curve


def _seed_dofs_xyzfourier(order: int, rng: np.random.Generator) -> np.ndarray:
    ndofs = 3 * (2 * order + 1)
    dofs = np.zeros(ndofs, dtype=np.float64)
    dofs[1] = 1.0
    dofs[2 * order + 3] = 1.0
    dofs[4 * order + 3] = 1.0
    return dofs + _PRODUCTION_RAND_SCALE * rng.random(ndofs)


def _seed_dofs_rzfourier(order: int, rng: np.random.Generator) -> np.ndarray:
    ndofs = (order + 1) + order
    dofs = np.zeros(ndofs, dtype=np.float64)
    dofs[0] = 1.0
    dofs[1] = 0.1
    dofs[order + 1] = 0.1
    return dofs + _PRODUCTION_RAND_SCALE * rng.random(ndofs)


def _seed_dofs_planarfourier(order: int, rng: np.random.Generator) -> np.ndarray:
    ndofs = (order + 1) + order + 4 + 3
    dofs = np.zeros(ndofs, dtype=np.float64)
    dofs[0] = 1.0
    dofs[1] = 0.1
    dofs[order + 1] = 0.1
    q_start = (order + 1) + order
    dofs[q_start] = 1.0
    return dofs + _PRODUCTION_RAND_SCALE * rng.random(ndofs)


def _seed_dofs_helical(order: int, rng: np.random.Generator) -> np.ndarray:
    ndofs = 1 + 2 * order
    dofs = np.zeros(ndofs, dtype=np.float64)
    dofs[0] = np.pi / 2
    return dofs + _PRODUCTION_RAND_SCALE * rng.random(ndofs)


_PRODUCTION_CURVE_FACTORIES = (
    ("CurveXYZFourier", _make_curve_xyzfourier, _seed_dofs_xyzfourier),
    ("CurveRZFourier", _make_curve_rzfourier, _seed_dofs_rzfourier),
    ("CurvePlanarFourier", _make_curve_planarfourier, _seed_dofs_planarfourier),
    ("CurveHelical", _make_curve_helical, _seed_dofs_helical),
)


@pytest.mark.parametrize(
    ("curve_name", "curve_factory", "seed_factory"),
    _PRODUCTION_CURVE_FACTORIES,
    ids=[name for name, _factory, _seed in _PRODUCTION_CURVE_FACTORIES],
)
def test_curve_spec_pullback_production_scale_parity(
    curve_name: str,
    curve_factory,
    seed_factory,
):
    """Production-scale floor: ncoils=4, nquadpoints=64 per curve class.

    Compares ``curve.gamma()`` (CPU oracle) against
    ``curve_geometry_from_dofs(curve_spec_from_curve(curve), spec.dofs)[0]``
    at the ``direct_kernel`` tolerance lane. The existing
    ``_CURVE_SPEC_FACTORIES`` row in
    ``tests/field/test_biotsavart_jax_parity.py`` exercises the same kernels
    at ``ncoils=1, nquadpoints=100``; this fixture lifts the floor to a
    Stage-2-realistic per-coil quadpoint count while iterating over four
    independently seeded coils per class.
    """
    rng = np.random.default_rng(_PRODUCTION_RNG_SEED)
    for coil_index in range(_PRODUCTION_NCOILS):
        dofs = seed_factory(_PRODUCTION_ORDER, rng)
        curve = curve_factory(_PRODUCTION_ORDER, _PRODUCTION_NQUADPOINTS, dofs)

        spec = curve_spec_from_curve(curve)
        gamma_cpu = np.asarray(curve.gamma(), dtype=np.float64)
        gamma_jax = np.asarray(
            curve_geometry_from_dofs(spec, spec.dofs)[0],
            dtype=np.float64,
        )

        assert gamma_cpu.shape == (_PRODUCTION_NQUADPOINTS, 3), (
            f"{curve_name} coil {coil_index}: CPU gamma shape {gamma_cpu.shape}"
        )
        assert gamma_jax.shape == (_PRODUCTION_NQUADPOINTS, 3), (
            f"{curve_name} coil {coil_index}: JAX gamma shape {gamma_jax.shape}"
        )
        np.testing.assert_allclose(
            gamma_jax,
            gamma_cpu,
            rtol=_RTOL,
            atol=_ATOL,
            err_msg=(
                f"{curve_name} coil {coil_index}: "
                "JAX spec geometry diverges from CPU oracle at production "
                "scale (ncoils=4, nquadpoints=64)."
            ),
        )


def test_curvexyzfouriersymmetries_spec_routing_is_documented_blocker():
    """Document that ``CurveXYZFourierSymmetries`` has no immutable JAX spec.

    ``CurveXYZFourierSymmetries`` (``src/simsopt/geo/curvexyzfouriersymmetries.py``)
    is a ``JaxCurve`` subclass: its forward geometry ``curve.gamma()`` already
    runs through ``jit``-compiled ``gamma_pure`` -> ``gamma_jax``. However,
    the class does NOT implement ``to_spec()`` and it does NOT carry a
    ``surf`` attribute, so the surface-fallback branch of
    ``curve_spec_from_curve`` (``src/simsopt/jax_core/curve_geometry.py:104-129``)
    is unreachable. Calling ``curve_spec_from_curve(curve)`` raises
    ``NotImplementedError("Curve type CurveXYZFourierSymmetries does not
    expose an immutable JAX spec.")``.

    The architecture limitation is recorded as a section-5 ``architecture``
    candidate in the item 05 plan
    (``.artifacts/jax_port_goal/plans/05.md``). Routing this class through
    ``curve_spec_from_curve`` requires adding a new ``CurveXYZFourierSymmetriesSpec``
    plus a ``to_spec`` method on the class; per the item 05 prompt, the
    closeout MAY NOT modify source classes, so the parity row is skipped
    with an explicit blocker rather than silently passing.

    Indirect coverage of the underlying kernel exists in
    ``tests/geo/test_curve.py`` (``CurveXYZFourierSymmetries{1,2,3}`` rows in
    ``Testing.curvetypes``), which exercises ``curve.gamma()`` -- itself a
    ``jit``-compiled JAX evaluation -- against the upstream oracle.
    """
    curve = CurveXYZFourierSymmetries(
        _PRODUCTION_NQUADPOINTS,
        _PRODUCTION_ORDER,
        nfp=3,
        stellsym=True,
        ntor=1,
    )
    curve.set("xc(0)", 1.0)
    curve.set("xc(1)", -0.3)
    curve.set("zs(1)", -0.3)

    # Forward parity over the existing pure-JAX hot path remains valid: the
    # class is a JaxCurve subclass, so ``curve.gamma()`` is a JAX evaluation
    # at any production-scale quadpoint count. Smoke that path so the
    # blocker test still proves the geometry is reachable.
    gamma = np.asarray(curve.gamma(), dtype=np.float64)
    assert gamma.shape == (_PRODUCTION_NQUADPOINTS, 3)
    assert np.isfinite(gamma).all()

    with pytest.raises(NotImplementedError, match="CurveXYZFourierSymmetries"):
        curve_spec_from_curve(curve)

    pytest.skip(
        "CurveXYZFourierSymmetries does not expose an immutable JAX spec on "
        "parent commit a9da18fac; routing it through curve_spec_from_curve "
        "requires a source-side CurveXYZFourierSymmetriesSpec + to_spec() "
        "which is out of scope for item 05 closeout (architecture blocker)."
    )
