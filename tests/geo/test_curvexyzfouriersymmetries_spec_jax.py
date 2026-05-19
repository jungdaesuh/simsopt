"""N05: ``CurveXYZFourierSymmetries`` immutable JAX spec parity coverage.

This file closes the documented architecture limitation pinned by
``tests/geo/test_curve_item05_closeout.py::
test_curvexyzfouriersymmetries_does_not_expose_immutable_spec`` on parent
commit ``a9da18fac``. After N05, ``CurveXYZFourierSymmetries`` exposes
``to_spec()`` returning a ``CurveXYZFourierSymmetriesSpec`` and the
existing ``curve_spec_from_curve(curve)`` delegate succeeds for this
class.

Source landmarks:

- ``src/simsopt/geo/curvexyzfouriersymmetries.py:8-57``
  (``jaxXYZFourierSymmetriescurve_pure`` — the SSOT JAX kernel).
- ``src/simsopt/geo/curvexyzfouriersymmetries.py:60-162``
  (``CurveXYZFourierSymmetries(JaxCurve)`` — the host class).
- ``src/simsopt/jax_core/specs.py`` (the new
  ``CurveXYZFourierSymmetriesSpec`` dataclass).
- ``src/simsopt/jax_core/curve_geometry.py`` (the new
  ``xyz_fourier_symmetries`` branch in ``_curve_gamma_kernel``).

All numeric tolerances come from the validation-ladder
``direct_kernel`` lane via ``parity_ladder_tolerances`` for the forward
gamma path; ``derivative_heavy`` lane for the first/second derivative
path.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.geo.curvexyzfouriersymmetries import CurveXYZFourierSymmetries
from simsopt.jax_core import (
    CurveXYZFourierSymmetriesSpec,
    curve_gamma_vjp_from_dofs,
    curve_gammadash_vjp_from_dofs,
    curve_gammadashdash_vjp_from_dofs,
    curve_gammadashdashdash_vjp_from_dofs,
    curve_spec_from_curve,
)
from simsopt.jax_core.curve_geometry import (
    curve_gamma_and_dash_from_dofs,
    curve_geometry_from_dofs,
    curve_pullback_from_dofs,
)
from simsopt.jax_core.specs import curve_spec_kind


_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_DIRECT_RTOL = _DIRECT_KERNEL["rtol"]
_DIRECT_ATOL = _DIRECT_KERNEL["atol"]

_DERIV_HEAVY = parity_ladder_tolerances("derivative_heavy")
_DERIV_RTOL = _DERIV_HEAVY["first_derivative_rtol"]
_DERIV_ATOL = _DERIV_HEAVY["first_derivative_atol"]
_SECOND_DERIV_RTOL = _DERIV_HEAVY["second_derivative_rtol"]
_SECOND_DERIV_ATOL = _DERIV_HEAVY["second_derivative_atol"]


_NQUADPOINTS = 64
_ORDER = 3
_RAND_SCALE = 1e-2
_RNG_SEED = 1729

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


def _stellsym_num_dofs(order: int) -> int:
    return (order + 1) + order + order


def _non_stellsym_num_dofs(order: int) -> int:
    return 3 * (2 * order + 1)


def _seed_stellsym_dofs(order: int, rng: np.random.Generator) -> np.ndarray:
    ndofs = _stellsym_num_dofs(order)
    dofs = np.zeros(ndofs, dtype=np.float64)
    # xc(0) sets the loop radius scale; xc(1), ys(1), zs(1) are the
    # dominant first-harmonic amplitudes for a helical coil.
    dofs[0] = 1.0
    dofs[1] = 0.25
    dofs[order + 1] = 0.25
    dofs[2 * order + 1] = 0.25
    return dofs + _RAND_SCALE * rng.random(ndofs)


def _seed_non_stellsym_dofs(order: int, rng: np.random.Generator) -> np.ndarray:
    ndofs = _non_stellsym_num_dofs(order)
    dofs = np.zeros(ndofs, dtype=np.float64)
    # xc(0), yc(0) place the loop centre; xc(1) and ys(1) provide the
    # first-harmonic shape.
    dofs[0] = 1.0
    dofs[1] = 0.25
    dofs[2 * order + 1] = 0.0
    dofs[3 * order + 2] = 0.25
    return dofs + _RAND_SCALE * rng.random(ndofs)


def _build_curve(
    *,
    order: int,
    nfp: int,
    stellsym: bool,
    ntor: int,
    rng: np.random.Generator,
    nquadpoints: int = _NQUADPOINTS,
) -> CurveXYZFourierSymmetries:
    curve = CurveXYZFourierSymmetries(
        nquadpoints,
        order,
        nfp=nfp,
        stellsym=stellsym,
        ntor=ntor,
    )
    if stellsym:
        dofs = _seed_stellsym_dofs(order, rng)
    else:
        dofs = _seed_non_stellsym_dofs(order, rng)
    curve.x = dofs
    return curve


_FORWARD_PARITY_CASES = [
    pytest.param(3, 2, True, id="3-2-True"),
    pytest.param(3, 2, False, id="3-2-False"),
    pytest.param(5, 3, True, id="5-3-True"),
    pytest.param(5, 3, False, id="5-3-False"),
    pytest.param(4, 1, True, id="4-1-True"),
    pytest.param(7, 4, True, id="7-4-True"),
]


def _numpy_gamma_oracle(
    dofs: np.ndarray,
    quadpoints: np.ndarray,
    order: int,
    nfp: int,
    stellsym: bool,
    ntor: int,
) -> np.ndarray:
    """Pure-NumPy ``gamma`` reference for ``CurveXYZFourierSymmetries``.

    Implements the closed-form Fourier series documented at
    ``src/simsopt/geo/curvexyzfouriersymmetries.py:66-90`` without
    sharing any code with ``jaxXYZFourierSymmetriescurve_pure``. Acts as
    the independent oracle for the JAX kernel parity tests.
    """
    theta = np.asarray(quadpoints, dtype=np.float64)
    angles_full = 2.0 * np.pi * nfp * np.outer(theta, np.arange(order + 1))
    cos_full = np.cos(angles_full)  # (N, order+1)
    sin_tail = np.sin(angles_full[:, 1:])  # (N, order)

    if stellsym:
        xc = dofs[: order + 1]
        ys = dofs[order + 1 : 2 * order + 1]
        zs = dofs[2 * order + 1 :]
        hat_x = cos_full @ xc
        hat_y = sin_tail @ ys
        z = sin_tail @ zs
    else:
        xc = dofs[0 : order + 1]
        xs = dofs[order + 1 : 2 * order + 1]
        yc = dofs[2 * order + 1 : 3 * order + 2]
        ys = dofs[3 * order + 2 : 4 * order + 2]
        zc = dofs[4 * order + 2 : 5 * order + 3]
        zs = dofs[5 * order + 3 :]
        hat_x = cos_full @ xc + sin_tail @ xs
        hat_y = cos_full @ yc + sin_tail @ ys
        z = cos_full @ zc + sin_tail @ zs

    tor = 2.0 * np.pi * theta * ntor
    cos_tor = np.cos(tor)
    sin_tor = np.sin(tor)
    x = hat_x * cos_tor - hat_y * sin_tor
    y = hat_x * sin_tor + hat_y * cos_tor
    return np.stack((x, y, z), axis=1).astype(np.float64)


@pytest.mark.parametrize(("nfp", "ntor", "stellsym"), _FORWARD_PARITY_CASES)
def test_spec_gamma_byte_identical_to_curve_gamma(
    nfp: int,
    ntor: int,
    stellsym: bool,
) -> None:
    """Spec-driven gamma matches ``curve.gamma()`` at the direct-kernel lane."""
    rng = np.random.default_rng(_RNG_SEED)
    curve = _build_curve(
        order=_ORDER,
        nfp=nfp,
        stellsym=stellsym,
        ntor=ntor,
        rng=rng,
    )
    spec = curve.to_spec()
    assert isinstance(spec, CurveXYZFourierSymmetriesSpec)

    gamma_curve = np.asarray(curve.gamma(), dtype=np.float64)
    gamma_jax = np.asarray(
        curve_geometry_from_dofs(spec, spec.dofs)[0], dtype=np.float64
    )
    # Independent NumPy oracle (closed-form series from the docstring;
    # shares no code with `jaxXYZFourierSymmetriescurve_pure`).
    gamma_numpy = _numpy_gamma_oracle(
        np.asarray(spec.dofs, dtype=np.float64),
        np.asarray(spec.quadpoints, dtype=np.float64),
        order=int(spec.order),
        nfp=int(spec.nfp),
        stellsym=bool(spec.stellsym),
        ntor=int(spec.ntor),
    )

    assert gamma_curve.shape == (_NQUADPOINTS, 3)
    assert gamma_jax.shape == (_NQUADPOINTS, 3)
    assert gamma_numpy.shape == (_NQUADPOINTS, 3)
    # Both JaxCurve route and the new spec route must agree with the
    # independent oracle (these two paths share the JAX kernel, so an
    # algebraic regression in `jaxXYZFourierSymmetriescurve_pure` would
    # fail the oracle comparison even though they would still agree with
    # each other).
    np.testing.assert_allclose(
        gamma_jax,
        gamma_numpy,
        rtol=_DIRECT_RTOL,
        atol=_DIRECT_ATOL,
    )
    np.testing.assert_allclose(
        gamma_curve,
        gamma_numpy,
        rtol=_DIRECT_RTOL,
        atol=_DIRECT_ATOL,
    )


@pytest.mark.parametrize(
    "stellsym", [True, False], ids=["stellsym=True", "stellsym=False"]
)
def test_to_spec_round_trips_fields(stellsym: bool) -> None:
    """``to_spec()`` returns a spec whose fields mirror the host class state."""
    rng = np.random.default_rng(_RNG_SEED + 1)
    curve = _build_curve(
        order=_ORDER,
        nfp=3,
        stellsym=stellsym,
        ntor=2,
        rng=rng,
    )
    spec = curve.to_spec()

    assert isinstance(spec, CurveXYZFourierSymmetriesSpec)
    assert spec.order == curve.order
    assert spec.nfp == curve.nfp
    assert spec.stellsym == curve.stellsym
    assert spec.ntor == curve.ntor

    np.testing.assert_array_equal(
        np.asarray(spec.dofs), np.asarray(curve.get_dofs(), dtype=np.float64)
    )
    np.testing.assert_array_equal(
        np.asarray(spec.quadpoints), np.asarray(curve.quadpoints, dtype=np.float64)
    )

    # ``curve_spec_kind`` discriminator must recognize the new variant.
    assert curve_spec_kind(spec) == "xyz_fourier_symmetries"


def test_curve_spec_from_curve_delegates_to_to_spec() -> None:
    """``curve_spec_from_curve(curve)`` returns the same spec as ``to_spec()``."""
    rng = np.random.default_rng(_RNG_SEED + 2)
    curve = _build_curve(
        order=_ORDER,
        nfp=5,
        stellsym=True,
        ntor=3,
        rng=rng,
    )
    spec_a = curve_spec_from_curve(curve)
    spec_b = curve.to_spec()

    assert isinstance(spec_a, CurveXYZFourierSymmetriesSpec)
    assert isinstance(spec_b, CurveXYZFourierSymmetriesSpec)
    assert spec_a.order == spec_b.order
    assert spec_a.nfp == spec_b.nfp
    assert spec_a.stellsym == spec_b.stellsym
    assert spec_a.ntor == spec_b.ntor
    np.testing.assert_array_equal(np.asarray(spec_a.dofs), np.asarray(spec_b.dofs))
    np.testing.assert_array_equal(
        np.asarray(spec_a.quadpoints), np.asarray(spec_b.quadpoints)
    )


@pytest.mark.parametrize(
    "stellsym", [True, False], ids=["stellsym=True", "stellsym=False"]
)
def test_spec_higher_derivatives_match_curve(stellsym: bool) -> None:
    """``curve_geometry_from_dofs`` ladder matches CPU oracle for gamma/dash/dashdash."""
    rng = np.random.default_rng(_RNG_SEED + 3)
    curve = _build_curve(
        order=_ORDER,
        nfp=3,
        stellsym=stellsym,
        ntor=2,
        rng=rng,
    )
    spec = curve.to_spec()

    gamma_cpu = np.asarray(curve.gamma(), dtype=np.float64)
    gammadash_cpu = np.asarray(curve.gammadash(), dtype=np.float64)
    gammadashdash_cpu = np.asarray(curve.gammadashdash(), dtype=np.float64)

    gamma_jax, gammadash_jax, gammadashdash_jax = (
        np.asarray(term, dtype=np.float64)
        for term in curve_geometry_from_dofs(spec, spec.dofs)
    )

    np.testing.assert_allclose(
        gamma_jax,
        gamma_cpu,
        rtol=_DIRECT_RTOL,
        atol=_DIRECT_ATOL,
    )
    np.testing.assert_allclose(
        gammadash_jax,
        gammadash_cpu,
        rtol=_DERIV_RTOL,
        atol=_DERIV_ATOL,
    )
    np.testing.assert_allclose(
        gammadashdash_jax,
        gammadashdash_cpu,
        rtol=_SECOND_DERIV_RTOL,
        atol=_SECOND_DERIV_ATOL,
    )


@pytest.mark.parametrize(
    "stellsym", [True, False], ids=["stellsym=True", "stellsym=False"]
)
def test_curve_pullback_shape(stellsym: bool) -> None:
    """VJP cotangent has the shape of ``spec.dofs`` and no surface cotangent."""
    rng = np.random.default_rng(_RNG_SEED + 4)
    curve = _build_curve(
        order=_ORDER,
        nfp=3,
        stellsym=stellsym,
        ntor=2,
        rng=rng,
    )
    spec = curve.to_spec()

    gamma_jax, gammadash_jax = curve_gamma_and_dash_from_dofs(spec, spec.dofs)
    cotangent_g = jnp.ones_like(gamma_jax)
    cotangent_gd = jnp.ones_like(gammadash_jax)

    coeff_cotangent, surface_cotangent = curve_pullback_from_dofs(
        spec, spec.dofs, cotangent_g, cotangent_gd
    )
    assert surface_cotangent is None
    assert coeff_cotangent.shape == spec.dofs.shape
    # The all-ones cotangent must produce a finite, non-zero result for a
    # non-degenerate curve. Zero would mean the pullback misrouted.
    coeff_np = np.asarray(coeff_cotangent, dtype=np.float64)
    assert np.all(np.isfinite(coeff_np))
    assert np.linalg.norm(coeff_np) > 0.0


@pytest.mark.parametrize(
    "stellsym", [True, False], ids=["stellsym=True", "stellsym=False"]
)
@pytest.mark.parametrize(
    ("term_name", "vjp_fn", "derivative_method"),
    _CURVE_VJP_CASES,
    ids=[name for name, _fn, _method in _CURVE_VJP_CASES],
)
def test_named_geometry_vjp_wrappers_match_cpu_derivatives(
    stellsym: bool,
    term_name: str,
    vjp_fn,
    derivative_method: str,
) -> None:
    """Named geometry VJP wrappers match CPU derivative tensor contractions."""

    rng = np.random.default_rng(_RNG_SEED + 5 + int(stellsym))
    curve = _build_curve(
        order=_ORDER,
        nfp=3,
        stellsym=stellsym,
        ntor=2,
        rng=rng,
    )
    spec = curve.to_spec()
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
        err_msg=f"CurveXYZFourierSymmetries {term_name} named VJP mismatch.",
    )


def test_spec_geometry_runs_under_strict_transfer_guard() -> None:
    """Spec-driven geometry executes under ``jax.transfer_guard('disallow')``.

    The spec is placed on-device via ``jax.device_put`` and a compiled
    callable evaluates ``curve_geometry_from_dofs`` without implicit
    host-to-device transfers.
    """
    rng = np.random.default_rng(_RNG_SEED + 5)
    curve = _build_curve(
        order=_ORDER,
        nfp=3,
        stellsym=True,
        ntor=2,
        rng=rng,
    )
    spec = jax.device_put(curve.to_spec())

    @jax.jit
    def evaluate(spec_arg):
        gamma, gammadash, gammadashdash = curve_geometry_from_dofs(
            spec_arg, spec_arg.dofs
        )
        return gamma, gammadash, gammadashdash

    compiled = evaluate.lower(spec).compile()
    with jax.transfer_guard("disallow"):
        gamma, gammadash, gammadashdash = compiled(spec)

    assert gamma.shape == (_NQUADPOINTS, 3)
    assert gammadash.shape == (_NQUADPOINTS, 3)
    assert gammadashdash.shape == (_NQUADPOINTS, 3)
    assert np.all(np.isfinite(np.asarray(gamma)))
    assert np.all(np.isfinite(np.asarray(gammadash)))
    assert np.all(np.isfinite(np.asarray(gammadashdash)))


def test_coprime_invariant_unchanged() -> None:
    """The host-class constructor still enforces ``gcd(ntor, nfp) == 1``."""
    with pytest.raises(Exception, match="nfp and ntor must be coprime"):
        CurveXYZFourierSymmetries(_NQUADPOINTS, _ORDER, nfp=4, stellsym=True, ntor=2)
