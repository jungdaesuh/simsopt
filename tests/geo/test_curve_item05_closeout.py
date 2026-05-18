"""Item 05 closeout: curve spec / pullback parity coverage at production scale.

This module closes the two coverage gaps identified in the item 05 audit of
``src/simsopt/geo/{curve,curvexyzfourier,curverzfourier,
curvexyzfouriersymmetries,curveplanarfourier,curvehelical,curvecwsfourier,
curveperturbed}.py`` and ``src/simsopt/jax_core/curve_geometry.py``:

1. ``CurveXYZFourierSymmetries`` had no JAX-spec parity test row. The
   architecture blocker noted in the item 05 plan has since been lifted:
   the class now exposes ``to_spec()`` returning a
   ``CurveXYZFourierSymmetriesSpec``, and ``curve_spec_from_curve``
   dispatches to it. The positive parity row is pinned in
   ``test_curvexyzfouriersymmetries_exposes_immutable_spec_with_geometry_parity``
   at the ``direct_kernel`` lane tolerance, mirroring the production-scale
   parametrized parity test below.
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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.geo.curve import RotatedCurve
from simsopt.geo.curvehelical import CurveHelical
from simsopt.geo.curveperturbed import CurvePerturbed, GaussianSampler, PerturbationSample
from simsopt.geo.curveplanarfourier import CurvePlanarFourier
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.curvexyzfouriersymmetries import CurveXYZFourierSymmetries
from simsopt.jax_core import (
    curve_dincremental_arclength_by_dcoeff_from_dofs,
    curve_dincremental_arclength_by_dcoeff_vjp_from_dofs,
    curve_dkappa_by_dcoeff_from_dofs,
    curve_dkappa_by_dcoeff_vjp_from_dofs,
    curve_dtorsion_by_dcoeff_from_dofs,
    curve_dtorsion_by_dcoeff_vjp_from_dofs,
    curve_gamma_vjp_from_dofs,
    curve_gammadash_vjp_from_dofs,
    curve_gammadashdash_vjp_from_dofs,
    curve_gammadashdashdash_vjp_from_dofs,
    curve_spec_from_curve,
)
from simsopt.jax_core.curve_planar_fourier import curveplanarfourier_pure
from simsopt.jax_core.curve_geometry import (
    _curve_geometry_with_third_derivative_from_dofs,
    curve_geometry_from_dofs,
)


_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]
_DERIVATIVE_HEAVY = parity_ladder_tolerances("derivative_heavy")
_DERIV_RTOL = _DERIVATIVE_HEAVY["first_derivative_rtol"]
_DERIV_ATOL = _DERIVATIVE_HEAVY["first_derivative_atol"]


_PRODUCTION_NCOILS = 4
_PRODUCTION_NQUADPOINTS = 64
_PRODUCTION_ORDER = 2
_PRODUCTION_RAND_SCALE = 0.01
_PRODUCTION_RNG_SEED = 7


def _planarfourier_num_dofs(order: int) -> int:
    return (order + 1) + order + 4 + 3


def _planarfourier_quaternion_slice(order: int) -> slice:
    start = (order + 1) + order
    return slice(start, start + 4)


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
    ndofs = _planarfourier_num_dofs(order)
    dofs = np.zeros(ndofs, dtype=np.float64)
    dofs[0] = 1.0
    dofs[1] = 0.1
    dofs[order + 1] = 0.1
    dofs = dofs + _PRODUCTION_RAND_SCALE * rng.random(ndofs)
    dofs[_planarfourier_quaternion_slice(order)] = np.asarray([0.5, 0.5, 0.5, 0.5])
    return dofs


def test_curve_planarfourier_zero_quaternion_gradient_matches_cpp():
    """Zero quaternion has identity forward value and zero quaternion gradient."""

    order = 1
    nquad = 16
    curve = CurvePlanarFourier(nquad, order)
    dofs = np.asarray(curve.x, dtype=np.float64)
    quaternion_slice = _planarfourier_quaternion_slice(order)
    dofs[quaternion_slice] = 0.0
    curve.x = dofs

    quadpoints = jnp.asarray(curve.quadpoints, dtype=jnp.float64)
    jax_jac = jax.jacfwd(
        lambda local_dofs: curveplanarfourier_pure(local_dofs, quadpoints, order)
    )(jnp.asarray(dofs, dtype=jnp.float64))
    cpp_jac = curve.dgamma_by_dcoeff()

    jax_quat = np.asarray(jax_jac[:, :, quaternion_slice])
    cpp_quat = np.asarray(cpp_jac[:, :, quaternion_slice])
    assert np.isfinite(jax_quat).all()
    np.testing.assert_allclose(jax_quat, cpp_quat, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(
        jax_quat, np.zeros_like(jax_quat), rtol=_RTOL, atol=_ATOL
    )


def test_curve_planarfourier_derivative_cache_tracks_quaternion_dof_updates():
    """Planar Fourier derivative caches depend on the normalized quaternion."""

    order = 2
    nquad = 16
    quaternion_slice = _planarfourier_quaternion_slice(order)

    initial_dofs = np.zeros(_planarfourier_num_dofs(order), dtype=np.float64)
    initial_dofs[0] = 1.0
    initial_dofs[1] = 0.17
    initial_dofs[order + 1] = 0.11
    initial_dofs[quaternion_slice] = np.asarray([0.7, -0.2, 0.5, 0.3])
    initial_dofs[-3:] = np.asarray([0.1, -0.2, 0.3])

    updated_dofs = initial_dofs.copy()
    updated_dofs[quaternion_slice] = np.asarray([0.2, 0.6, -0.4, 0.5])

    curve = CurvePlanarFourier(nquad, order)
    curve.x = initial_dofs
    seeded_derivatives = {
        method_name: np.asarray(getattr(curve, method_name)(), dtype=np.float64).copy()
        for method_name in (
            "dgamma_by_dcoeff",
            "dgammadash_by_dcoeff",
            "dgammadashdash_by_dcoeff",
            "dgammadashdashdash_by_dcoeff",
        )
    }

    curve.x = updated_dofs
    fresh_curve = CurvePlanarFourier(nquad, order)
    fresh_curve.x = updated_dofs

    for method_name, seeded in seeded_derivatives.items():
        updated = np.asarray(getattr(curve, method_name)(), dtype=np.float64)
        fresh = np.asarray(getattr(fresh_curve, method_name)(), dtype=np.float64)

        assert np.max(np.abs(fresh - seeded)) > _ATOL
        np.testing.assert_allclose(
            updated,
            fresh,
            rtol=_RTOL,
            atol=_ATOL,
            err_msg=f"{method_name} stayed stale after quaternion DOF update.",
        )


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

_CURVE_VJP_CASES = (
    ("gamma", curve_gamma_vjp_from_dofs, "dgamma_by_dcoeff"),
    ("gammadash", curve_gammadash_vjp_from_dofs, "dgammadash_by_dcoeff"),
    (
        "gammadashdash",
        curve_gammadashdash_vjp_from_dofs,
        "dgammadashdash_by_dcoeff",
    ),
    (
        "gammadashdashdash",
        curve_gammadashdashdash_vjp_from_dofs,
        "dgammadashdashdash_by_dcoeff",
    ),
)

_CURVE_SCALAR_DERIVATIVE_CASES = (
    (
        "incremental_arclength",
        curve_dincremental_arclength_by_dcoeff_from_dofs,
        curve_dincremental_arclength_by_dcoeff_vjp_from_dofs,
        "dincremental_arclength_by_dcoeff",
    ),
    (
        "kappa",
        curve_dkappa_by_dcoeff_from_dofs,
        curve_dkappa_by_dcoeff_vjp_from_dofs,
        "dkappa_by_dcoeff",
    ),
    (
        "torsion",
        curve_dtorsion_by_dcoeff_from_dofs,
        curve_dtorsion_by_dcoeff_vjp_from_dofs,
        "dtorsion_by_dcoeff",
    ),
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
        geometry_jax = curve_geometry_from_dofs(spec, spec.dofs)
        gamma_jax = np.asarray(geometry_jax[0], dtype=np.float64)

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
        if curve_name in {"CurveRZFourier", "CurvePlanarFourier"}:
            derivative_oracles = (
                ("gammadash", curve.gammadash(), geometry_jax[1]),
                ("gammadashdash", curve.gammadashdash(), geometry_jax[2]),
            )
            for label, cpu_value, jax_value in derivative_oracles:
                np.testing.assert_allclose(
                    np.asarray(jax_value, dtype=np.float64),
                    np.asarray(cpu_value, dtype=np.float64),
                    rtol=_RTOL,
                    atol=_ATOL,
                    err_msg=f"{curve_name} coil {coil_index}: {label} parity failed.",
                )

            third_geometry = _curve_geometry_with_third_derivative_from_dofs(
                spec, spec.dofs
            )
            np.testing.assert_allclose(
                np.asarray(third_geometry[3], dtype=np.float64),
                np.asarray(curve.gammadashdashdash(), dtype=np.float64),
                rtol=_RTOL,
                atol=_ATOL,
                err_msg=(
                    f"{curve_name} coil {coil_index}: gammadashdashdash parity failed."
                ),
            )


@pytest.mark.parametrize(
    ("curve_name", "curve_factory", "seed_factory"),
    _PRODUCTION_CURVE_FACTORIES,
    ids=[name for name, _factory, _seed in _PRODUCTION_CURVE_FACTORIES],
)
def test_curve_least_squares_fit_cpu_boundary_materializes_jax_spec(
    curve_name: str,
    curve_factory,
    seed_factory,
):
    """CPU curve fit mutation remains a valid JAX-spec materialization boundary."""

    rng = np.random.default_rng(_PRODUCTION_RNG_SEED + 29)
    source_dofs = seed_factory(_PRODUCTION_ORDER, rng)
    target_dofs = seed_factory(_PRODUCTION_ORDER, rng)
    source_curve = curve_factory(
        _PRODUCTION_ORDER,
        _PRODUCTION_NQUADPOINTS,
        source_dofs,
    )
    target_curve = curve_factory(
        _PRODUCTION_ORDER,
        _PRODUCTION_NQUADPOINTS,
        target_dofs,
    )
    target_gamma = np.asarray(target_curve.gamma(), dtype=np.float64)

    source_curve.least_squares_fit(target_gamma.copy())
    spec = curve_spec_from_curve(source_curve)
    fitted_gamma_cpu = np.asarray(source_curve.gamma(), dtype=np.float64)
    fitted_gamma_jax = np.asarray(
        curve_geometry_from_dofs(spec, spec.dofs)[0],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        fitted_gamma_jax,
        fitted_gamma_cpu,
        rtol=_RTOL,
        atol=_ATOL,
        err_msg=(
            f"{curve_name} CPU-fitted curve did not materialize into an equivalent "
            "JAX CurveSpec."
        ),
    )


@pytest.mark.parametrize(
    ("curve_name", "curve_factory", "seed_factory"),
    _PRODUCTION_CURVE_FACTORIES,
    ids=[name for name, _factory, _seed in _PRODUCTION_CURVE_FACTORIES],
)
@pytest.mark.parametrize(
    ("term_name", "vjp_fn", "derivative_method"),
    _CURVE_VJP_CASES,
    ids=[name for name, _fn, _method in _CURVE_VJP_CASES],
)
def test_curve_named_geometry_vjp_wrappers_match_cpu_derivatives(
    curve_name: str,
    curve_factory,
    seed_factory,
    term_name: str,
    vjp_fn,
    derivative_method: str,
):
    """Named JAX VJP wrappers match CPU derivative tensor contractions."""

    rng = np.random.default_rng(_PRODUCTION_RNG_SEED + 31)
    dofs = seed_factory(_PRODUCTION_ORDER, rng)
    curve = curve_factory(_PRODUCTION_ORDER, _PRODUCTION_NQUADPOINTS, dofs)
    spec = curve_spec_from_curve(curve)
    derivative_cpu = np.asarray(
        getattr(curve, derivative_method)(),
        dtype=np.float64,
    )
    cotangent = rng.normal(size=derivative_cpu.shape[:2])

    vjp_jax = np.asarray(
        vjp_fn(spec, spec.dofs, jnp.asarray(cotangent, dtype=jnp.float64)),
        dtype=np.float64,
    )
    vjp_cpu = np.einsum("ij,ijk->k", cotangent, derivative_cpu)

    np.testing.assert_allclose(
        vjp_jax,
        vjp_cpu,
        rtol=_DERIV_RTOL,
        atol=_DERIV_ATOL,
        err_msg=f"{curve_name} {term_name} named VJP diverges from CPU tensor contraction.",
    )


@pytest.mark.parametrize(
    ("curve_name", "curve_factory", "seed_factory"),
    _PRODUCTION_CURVE_FACTORIES,
    ids=[name for name, _factory, _seed in _PRODUCTION_CURVE_FACTORIES],
)
@pytest.mark.parametrize(
    ("term_name", "derivative_fn", "vjp_fn", "derivative_method"),
    _CURVE_SCALAR_DERIVATIVE_CASES,
    ids=[name for name, _derivative_fn, _vjp_fn, _method in _CURVE_SCALAR_DERIVATIVE_CASES],
)
def test_curve_scalar_derivative_wrappers_match_cpu_derivatives(
    curve_name: str,
    curve_factory,
    seed_factory,
    term_name: str,
    derivative_fn,
    vjp_fn,
    derivative_method: str,
):
    """Scalar geometry derivative wrappers match CPU Jacobians and pullbacks."""

    rng = np.random.default_rng(_PRODUCTION_RNG_SEED + 37)
    dofs = seed_factory(_PRODUCTION_ORDER, rng)
    curve = curve_factory(_PRODUCTION_ORDER, _PRODUCTION_NQUADPOINTS, dofs)
    spec = curve_spec_from_curve(curve)
    derivative_cpu = np.asarray(
        getattr(curve, derivative_method)(),
        dtype=np.float64,
    )
    derivative_jax = np.asarray(
        derivative_fn(spec, spec.dofs),
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        derivative_jax,
        derivative_cpu,
        rtol=_DERIV_RTOL,
        atol=_DERIV_ATOL,
        err_msg=f"{curve_name} {term_name} derivative wrapper diverges from CPU.",
    )

    cotangent = rng.normal(size=derivative_cpu.shape[0])
    vjp_jax = np.asarray(
        vjp_fn(spec, spec.dofs, jnp.asarray(cotangent, dtype=jnp.float64)),
        dtype=np.float64,
    )
    vjp_cpu = np.einsum("i,ik->k", cotangent, derivative_cpu)

    np.testing.assert_allclose(
        vjp_jax,
        vjp_cpu,
        rtol=_DERIV_RTOL,
        atol=_DERIV_ATOL,
        err_msg=f"{curve_name} {term_name} scalar VJP diverges from CPU.",
    )


def test_curve_perturbed_named_geometry_vjp_wrappers_match_base_cpu_derivatives():
    """Perturbed specs route named VJPs through base curve DOFs."""

    rng = np.random.default_rng(_PRODUCTION_RNG_SEED + 41)
    dofs = _seed_dofs_xyzfourier(_PRODUCTION_ORDER, rng)
    base_curve = _make_curve_xyzfourier(
        _PRODUCTION_ORDER,
        _PRODUCTION_NQUADPOINTS,
        dofs,
    )
    sampler = GaussianSampler(
        base_curve.quadpoints,
        sigma=0.1,
        length_scale=0.5,
        n_derivs=3,
    )
    sample = PerturbationSample(sampler, randomgen=rng)
    perturbed_curve = CurvePerturbed(base_curve, sample)
    spec = curve_spec_from_curve(perturbed_curve)

    for term_name, vjp_fn, derivative_method in _CURVE_VJP_CASES:
        derivative_cpu = np.asarray(
            getattr(base_curve, derivative_method)(),
            dtype=np.float64,
        )
        cotangent = rng.normal(size=derivative_cpu.shape[:2])
        vjp_jax = np.asarray(
            vjp_fn(spec, spec.dofs, jnp.asarray(cotangent, dtype=jnp.float64)),
            dtype=np.float64,
        )
        vjp_cpu = np.einsum("ij,ijk->k", cotangent, derivative_cpu)

        np.testing.assert_allclose(
            vjp_jax,
            vjp_cpu,
            rtol=_DERIV_RTOL,
            atol=_DERIV_ATOL,
            err_msg=f"CurvePerturbed {term_name} named VJP mismatch.",
        )


def test_curvexyzfouriersymmetries_exposes_immutable_spec_with_geometry_parity():
    """Pin ``CurveXYZFourierSymmetries`` -> spec -> geometry parity at production scale.

    ``CurveXYZFourierSymmetries`` (``src/simsopt/geo/curvexyzfouriersymmetries.py``)
    is a ``JaxCurve`` subclass. The blocker recorded in the item 05 plan
    (``.artifacts/jax_port_goal/plans/05.md`` section-5 architecture candidate)
    has been lifted: the class now exposes ``to_spec()`` returning a
    ``CurveXYZFourierSymmetriesSpec``, and ``curve_spec_from_curve`` dispatches
    to it. This test pins the positive parity row in place of the prior
    blocker assertion (``pytest.raises(NotImplementedError)``).

    Oracle: CPU ``curve.gamma()`` at production scale (nquadpoints=64).
    Lane: ``direct_kernel`` (rtol/atol from ``parity_ladder_tolerances``).
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

    spec = curve_spec_from_curve(curve)
    assert type(spec).__name__ == "CurveXYZFourierSymmetriesSpec"

    gamma_cpu = np.asarray(curve.gamma(), dtype=np.float64)
    gamma_jax = np.asarray(
        curve_geometry_from_dofs(spec, spec.dofs)[0],
        dtype=np.float64,
    )

    assert gamma_cpu.shape == (_PRODUCTION_NQUADPOINTS, 3)
    assert gamma_jax.shape == (_PRODUCTION_NQUADPOINTS, 3)
    np.testing.assert_allclose(
        gamma_jax,
        gamma_cpu,
        rtol=_RTOL,
        atol=_ATOL,
        err_msg=(
            "CurveXYZFourierSymmetries: JAX spec geometry diverges from CPU "
            "oracle at production scale (nquadpoints=64)."
        ),
    )


def test_rotated_curve_spec_dispatcher_documents_cpu_only_wrapper():
    """``RotatedCurve`` placement is not a standalone JAX ``CurveSpec``."""
    base = CurveXYZFourier(_PRODUCTION_NQUADPOINTS, _PRODUCTION_ORDER)
    rotated = RotatedCurve(base, phi=0.37, flip=True)

    with pytest.raises(NotImplementedError, match="CoilSymmetrySpec"):
        curve_spec_from_curve(rotated)
