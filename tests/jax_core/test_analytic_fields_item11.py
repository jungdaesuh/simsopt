"""Parity tests for the JAX ``analytic_fields`` port (item 11).

Each test imports tolerances from
``benchmarks.validation_ladder_contract.parity_ladder_tolerances`` so the
lane contract is preserved end-to-end. Two oracles cover the new port:

* ``test_dommaschk_paper_fixtures`` -- closed-form oracle from
  ``tests/field/test_magneticfields.py::test_Dommaschk`` (the published
  Dommaschk paper, hard-coded references). Inherits the historical
  precision of those printed values via ``np.allclose`` defaults.
* ``test_dommaschk_cpp_cross_oracle`` and ``test_reiman_cpp_cross_oracle``
  -- direct cross-oracle parity against the C++ kernels
  ``sopp.DommaschkB`` / ``sopp.DommaschkdB`` / ``sopp.ReimanB`` /
  ``sopp.ReimandB``, at ``direct_kernel`` lane tolerance.
* ``test_reiman_closed_form`` -- closed-form sympy-style expression from
  the existing ``tests/field/test_magneticfields.py::test_Reiman``
  fixture.
* ``test_reiman_dB_taylor`` -- finite-difference Taylor test that
  mirrors the existing CPU regression.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp
import simsoptpp as sopp

from benchmarks.validation_ladder_contract import parity_ladder_tolerances

from simsopt.field.magneticfieldclasses import ToroidalField
from simsopt.jax_core.analytic_fields import (
    DommaschkSpec,
    ReimanSpec,
    _dommaschk_B_multimode_kernel,
    _dommaschk_dB_multimode_kernel,
    _dommaschk_term_bundle,
    _reiman_B_kernel,
    _reiman_dB_kernel,
    clear_dommaschk_caches,
    clear_reiman_caches,
    dommaschk_B,
    dommaschk_dB,
    reiman_B,
    reiman_dB,
)


_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_DIRECT_RTOL = _DIRECT_KERNEL["rtol"]
_DIRECT_ATOL = _DIRECT_KERNEL["atol"]

_RELAXED_KERNEL = parity_ladder_tolerances("relaxed_kernel")
_RELAXED_RTOL = _RELAXED_KERNEL["rtol"]
_RELAXED_ATOL = _RELAXED_KERNEL["atol"]

_DERIVATIVE_HEAVY = parity_ladder_tolerances("derivative_heavy")
_DERIVATIVE_RTOL = _DERIVATIVE_HEAVY["first_derivative_rtol"]
_DERIVATIVE_ATOL = _DERIVATIVE_HEAVY["first_derivative_atol"]

_FD_GRADIENT = parity_ladder_tolerances("fd_gradient")
_FD_ERROR_RATE = _FD_GRADIENT["central_fd_error_rate"]
_FD_RTOL = _FD_GRADIENT["directional_fd_rtol"]
_FD_ATOL = _FD_GRADIENT["directional_fd_atol"]


# Hard-coded Dommaschk fixtures from
# ``tests/field/test_magneticfields.py::test_Dommaschk`` (lines 642-708).
# Layout per fixture:
#   (mn_pairs, coeff_pairs, point, expected_B_wrapper, expected_grad_wrapper)
_DOMMASCHK_PAPER_FIXTURES = (
    (
        ((10, 2), (15, 3)),
        ((-2.18, -2.18), (25.8, -25.8)),
        ((0.9231, 0.8423, -0.1123),),
        ((-1.72696, 3.26173, -2.22013),),
        (
            (
                (-59.9602, 8.96793, -24.8844),
                (8.96793, 49.0327, -18.4131),
                (-24.8844, -18.4131, 10.9275),
            ),
        ),
    ),
    (
        ((5, 2), (5, 4), (5, 10)),
        ((1.4, 1.4), (19.25, 0), (5.10e10, 5.10e10)),
        ((0.71879008, 0.76265643, 0.0745),),
        ((-0.7094243, 0.65632967, -0.125321),),
        (
            (
                (0.90663628, 0.5078183, -0.55436901),
                (0.5078183, 0.27261978, -0.66073972),
                (-0.55436901, -0.66073972, -1.17925605),
            ),
        ),
    ),
    (
        ((3, 2), (6, 4), (2, 11)),
        ((1.4, 1.4), (19.25, 0), (5.10e10, 5.10e10)),
        ((0.77066908, -0.61182119, 0.1057),),
        ((0.55674279, 0.83401312, -0.121491),),
        (
            (
                (0.11538721234011184, -0.7518405857812525, -0.6107605261251816),
                (-0.7518410735861303, 1.0695191900989125, 0.14110885184619465),
                (-0.6107606676662055, 0.1411086735566982, -1.18491),
            ),
        ),
    ),
    (
        ((5, 0), (10, 10), (15, 19)),
        ((1.4, 1.4), (5.10e10, 5.10e10), (9e20, 0)),
        ((0.06660615, -0.93924128, 0.16),),
        ((3.90161959, -1.87151853, 0.0119783),),
        (
            (
                (39.394312086253024, 14.061725133810995, 0.1684479703125076),
                (14.061729381899355, -40.23304445668633, -0.40810476986895994),
                (0.16844815337021118, -0.4081047568874514, 0.838733),
            ),
        ),
    ),
)


def _build_dommaschk_spec(mn_pairs, coeff_pairs):
    m_tuple = tuple(int(pair[0]) for pair in mn_pairs)
    n_tuple = tuple(int(pair[1]) for pair in mn_pairs)
    coeffs = np.asarray(coeff_pairs, dtype=np.float64)
    return DommaschkSpec(m=m_tuple, n=n_tuple, coeffs=coeffs)


def _toroidal_baseline(points: np.ndarray):
    """Return the ``ToroidalField(R0=1, B0=1)`` B and dB at ``points``."""
    tf = ToroidalField(1, 1)
    tf.set_points(points)
    return np.array(tf.B()), np.array(tf.dB_by_dX())


def test_dommaschk_paper_fixtures():
    """Replicate the four printed fixtures from the existing C++ test.

    Each Dommaschk fixture in ``test_Dommaschk`` compares the wrapper
    output (raw kernel sum + ``ToroidalField(1, 1)`` baseline) against
    a printed reference. The new JAX kernel reproduces the *raw* sum
    only, so we re-add the same baseline before comparing to the
    printed wrapper reference. ``np.allclose`` defaults mirror the
    historical precision of the printed values.
    """

    for fixture in _DOMMASCHK_PAPER_FIXTURES:
        mn_pairs, coeff_pairs, point_tuple, B_ref, grad_ref = fixture
        point = np.asarray(point_tuple, dtype=np.float64)
        spec = _build_dommaschk_spec(mn_pairs, coeff_pairs)

        B_raw_sum = np.asarray(dommaschk_B(spec, point)).sum(axis=0)
        dB_raw_sum = np.asarray(dommaschk_dB(spec, point)).sum(axis=0)

        tf_B, tf_dB = _toroidal_baseline(point)
        B_with_baseline = B_raw_sum + tf_B
        dB_with_baseline = dB_raw_sum + tf_dB

        # Historical reference precision: ``np.allclose`` defaults match
        # the existing ``test_Dommaschk`` comparison.
        assert np.allclose(B_with_baseline, np.asarray(B_ref))
        assert np.allclose(dB_with_baseline, np.asarray(grad_ref))


def test_dommaschk_cpp_cross_oracle():
    """Direct ``direct_kernel`` parity vs ``sopp.DommaschkB`` /
    ``sopp.DommaschkdB`` at well-conditioned inputs.

    Uses small coefficients so neither kernel exhibits catastrophic
    cancellation; both kernels compute the same scalar formula, so
    parity is bounded by floating-point ULP times the term magnitudes.
    """

    rng = np.random.default_rng(seed=1729)
    points = np.column_stack(
        [
            0.6 + 0.4 * rng.standard_normal(8),
            -0.2 + 0.4 * rng.standard_normal(8),
            rng.standard_normal(8) * 0.2,
        ]
    )
    points[:, :2] = np.where(
        np.linalg.norm(points[:, :2], axis=1, keepdims=True) > 0.5,
        points[:, :2],
        points[:, :2] + 1.0,
    )
    mn_pairs = ((5, 3), (4, 2), (3, 1), (2, 4))
    coeff_pairs = tuple(
        (float(rng.uniform(-1.0, 1.0)), float(rng.uniform(-1.0, 1.0)))
        for _ in range(len(mn_pairs))
    )
    spec = _build_dommaschk_spec(mn_pairs, coeff_pairs)

    m_int16 = np.array([pair[0] for pair in mn_pairs], dtype=np.int16)
    n_int16 = np.array([pair[1] for pair in mn_pairs], dtype=np.int16)
    coeffs_cpp = np.asarray(coeff_pairs, dtype=np.float64)
    B_cpp = np.asarray(sopp.DommaschkB(m_int16, n_int16, coeffs_cpp, points))
    dB_cpp = np.asarray(sopp.DommaschkdB(m_int16, n_int16, coeffs_cpp, points))

    B_jax = np.asarray(dommaschk_B(spec, points))
    dB_jax = np.asarray(dommaschk_dB(spec, points))

    assert B_jax.shape == B_cpp.shape
    assert dB_jax.shape == dB_cpp.shape
    np.testing.assert_allclose(B_jax, B_cpp, rtol=_DIRECT_RTOL, atol=_DIRECT_ATOL)
    np.testing.assert_allclose(dB_jax, dB_cpp, rtol=_DIRECT_RTOL, atol=_DIRECT_ATOL)


def test_dommaschk_large_coefficients_use_relaxed_kernel_lane():
    """Large coefficients exercise the documented monomial-merge ULP drift."""
    mn_pairs = ((5, 2), (5, 4), (5, 10))
    coeff_pairs = ((1.4, 1.4), (19.25, 0.0), (5.10e10, 5.10e10))
    points = np.asarray(((0.71879008, 0.76265643, 0.0745),), dtype=np.float64)
    spec = _build_dommaschk_spec(mn_pairs, coeff_pairs)

    m_int16 = np.asarray([pair[0] for pair in mn_pairs], dtype=np.int16)
    n_int16 = np.asarray([pair[1] for pair in mn_pairs], dtype=np.int16)
    coeffs_cpp = np.asarray(coeff_pairs, dtype=np.float64)
    B_cpp = np.asarray(sopp.DommaschkB(m_int16, n_int16, coeffs_cpp, points))
    dB_cpp = np.asarray(sopp.DommaschkdB(m_int16, n_int16, coeffs_cpp, points))

    B_jax = np.asarray(dommaschk_B(spec, points))
    dB_jax = np.asarray(dommaschk_dB(spec, points))

    assert np.max(np.abs(coeffs_cpp)) >= 1.0e10
    np.testing.assert_allclose(B_jax, B_cpp, rtol=_RELAXED_RTOL, atol=_RELAXED_ATOL)
    np.testing.assert_allclose(
        dB_jax,
        dB_cpp,
        rtol=_RELAXED_RTOL,
        atol=_RELAXED_ATOL,
    )


def test_dommaschk_caches_are_bounded_and_clearable():
    clear_dommaschk_caches()
    spec = _build_dommaschk_spec(((5, 3), (4, 2)), ((0.75, -0.5), (0.25, 0.125)))
    points = np.asarray([[0.72, -0.41, 0.18]], dtype=np.float64)

    np.asarray(dommaschk_B(spec, points))
    np.asarray(dommaschk_dB(spec, points))

    assert _dommaschk_term_bundle.cache_info().maxsize == 256
    assert _dommaschk_B_multimode_kernel.cache_info().maxsize == 128
    assert _dommaschk_dB_multimode_kernel.cache_info().maxsize == 128
    assert _dommaschk_term_bundle.cache_info().currsize > 0

    clear_dommaschk_caches()

    assert _dommaschk_term_bundle.cache_info().currsize == 0
    assert _dommaschk_B_multimode_kernel.cache_info().currsize == 0
    assert _dommaschk_dB_multimode_kernel.cache_info().currsize == 0


def test_reiman_caches_are_bounded_and_clearable():
    clear_reiman_caches()
    spec = ReimanSpec(
        iota0=0.15,
        iota1=0.38,
        k_theta=(5, 7),
        epsilon=np.asarray((0.01, 0.005), dtype=np.float64),
        m0_symmetry=1,
    )
    points = np.asarray([[0.72, 0.31, 0.22]], dtype=np.float64)

    np.asarray(reiman_B(spec, points))
    np.asarray(reiman_dB(spec, points))

    assert _reiman_B_kernel.cache_info().maxsize == 128
    assert _reiman_dB_kernel.cache_info().maxsize == 128
    assert _reiman_B_kernel.cache_info().currsize > 0
    assert _reiman_dB_kernel.cache_info().currsize > 0

    clear_reiman_caches()

    assert _reiman_B_kernel.cache_info().currsize == 0
    assert _reiman_dB_kernel.cache_info().currsize == 0


def test_dommaschk_default_n_zero_raw_kernel_is_finite_zero():
    """The JAX ``Nmn(m, -1)`` empty expansion pins the raw zero-field result."""
    spec = _build_dommaschk_spec(((0, 0),), ((0.0, 0.0),))
    points = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)

    np.testing.assert_array_equal(
        np.asarray(dommaschk_B(spec, points)),
        np.zeros((1, 1, 3), dtype=np.float64),
    )
    np.testing.assert_array_equal(
        np.asarray(dommaschk_dB(spec, points)),
        np.zeros((1, 1, 3, 3), dtype=np.float64),
    )


def test_dommaschk_grad_symmetric():
    """Vacuum-field consistency check: ``dB`` is symmetric and divergence-free.

    The Dommaschk field derives from a scalar potential, so its
    Cartesian gradient ``dB[p, i, j] = d B_j / d x_i`` must be
    symmetric in ``(i, j)`` for every evaluation point and every
    mode contribution. As a source-free vacuum field, ``trace(dB)``
    must also vanish for every point and mode.
    """

    fixture = _DOMMASCHK_PAPER_FIXTURES[0]
    mn_pairs, coeff_pairs, point_tuple, _, _ = fixture
    point = np.asarray(point_tuple, dtype=np.float64)
    spec = _build_dommaschk_spec(mn_pairs, coeff_pairs)
    dB = np.asarray(dommaschk_dB(spec, point))
    dB_T = np.swapaxes(dB, -1, -2)
    np.testing.assert_allclose(dB, dB_T, rtol=_DIRECT_RTOL, atol=_DIRECT_ATOL)
    np.testing.assert_allclose(
        np.trace(dB, axis1=-2, axis2=-1),
        0.0,
        rtol=_DIRECT_RTOL,
        atol=_DIRECT_ATOL,
    )


def test_dommaschk_dB_taylor():
    """Central finite-difference Taylor test for ``dommaschk_dB``."""

    mn_pairs, coeff_pairs, point_tuple, _, _ = _DOMMASCHK_PAPER_FIXTURES[0]
    points = np.asarray(point_tuple, dtype=np.float64)
    spec = _build_dommaschk_spec(mn_pairs, coeff_pairs)
    mode_idx = 0
    point_idx = 0
    dB = np.asarray(dommaschk_dB(spec, points))[mode_idx, point_idx]

    for direction in (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ):
        deriv = dB.T @ direction
        errors = []
        for i in range(3, 8):
            eps = 0.5**i
            Bplus = np.asarray(dommaschk_B(spec, points + eps * direction))[
                mode_idx, point_idx
            ]
            Bminus = np.asarray(dommaschk_B(spec, points - eps * direction))[
                mode_idx, point_idx
            ]
            deriv_est = (Bplus - Bminus) / (2.0 * eps)
            errors.append(float(np.linalg.norm(deriv - deriv_est)))
        assert min(errors) < max(1.0e-10, 0.01 * errors[0])
        if errors[0] > 1.0e-10:
            for previous, current in zip(errors, errors[1:]):
                assert current < _FD_ERROR_RATE * previous + np.finfo(float).eps


def _central_fd_gradient(objective, x):
    eps = 2.0 ** -18
    grad = np.zeros_like(x, dtype=np.float64)
    for index in np.ndindex(x.shape):
        step = np.zeros_like(x, dtype=np.float64)
        step[index] = eps
        grad[index] = (objective(x + step) - objective(x - step)) / (2.0 * eps)
    return grad


def test_dommaschk_grad_over_coefficients_matches_central_fd():
    mn_pairs, coeff_pairs, point_tuple, _, _ = _DOMMASCHK_PAPER_FIXTURES[0]
    points = jnp.asarray(point_tuple, dtype=jnp.float64)
    coeffs = np.asarray(coeff_pairs, dtype=np.float64)
    m_tuple = tuple(pair[0] for pair in mn_pairs)
    n_tuple = tuple(pair[1] for pair in mn_pairs)

    def objective(coeff_array):
        spec = DommaschkSpec(m=m_tuple, n=n_tuple, coeffs=coeff_array)
        B = dommaschk_B(spec, points)
        return jnp.sum(B * B)

    actual = np.asarray(jax.grad(objective)(jnp.asarray(coeffs)))
    expected = _central_fd_gradient(
        lambda c: float(objective(jnp.asarray(c, dtype=jnp.float64))), coeffs
    )

    np.testing.assert_allclose(actual, expected, rtol=_FD_RTOL, atol=_FD_ATOL)


def test_reiman_grad_over_coefficients_matches_central_fd():
    points = jnp.asarray(
        np.array(
            [
                [0.72, 0.31, 0.22],
                [0.63, -0.42, -0.18],
                [1.24, 0.27, 0.19],
            ],
            dtype=np.float64,
        )
    )
    epsilon = np.asarray((0.01, 0.005), dtype=np.float64)

    def objective(epsilon_array):
        spec = ReimanSpec(
            iota0=0.15,
            iota1=0.38,
            k_theta=(5, 7),
            epsilon=epsilon_array,
            m0_symmetry=1,
        )
        B = reiman_B(spec, points)
        return jnp.sum(B * B)

    actual = np.asarray(jax.grad(objective)(jnp.asarray(epsilon)))
    expected = _central_fd_gradient(
        lambda e: float(objective(jnp.asarray(e, dtype=jnp.float64))), epsilon
    )

    np.testing.assert_allclose(actual, expected, rtol=_FD_RTOL, atol=_FD_ATOL)


# Reiman fixture parameters from
# ``tests/field/test_magneticfields.py::test_Reiman`` (lines 958-1003).
_REIMAN_IOTA0 = 0.15
_REIMAN_IOTA1 = 0.38
_REIMAN_K = (6,)
_REIMAN_EPSILONK = (0.01,)
_REIMAN_M0 = 1


def _build_reiman_spec():
    return ReimanSpec(
        iota0=_REIMAN_IOTA0,
        iota1=_REIMAN_IOTA1,
        k_theta=_REIMAN_K,
        epsilon=np.asarray(_REIMAN_EPSILONK, dtype=np.float64),
        m0_symmetry=_REIMAN_M0,
    )


@pytest.mark.parametrize("kernel", [reiman_B, reiman_dB])
def test_reiman_rejects_nonpositive_k_theta(kernel):
    spec = ReimanSpec(
        iota0=_REIMAN_IOTA0,
        iota1=_REIMAN_IOTA1,
        k_theta=(0,),
        epsilon=np.asarray(_REIMAN_EPSILONK, dtype=np.float64),
        m0_symmetry=_REIMAN_M0,
    )
    points = np.asarray([[1.1, 0.0, 0.1]], dtype=np.float64)

    with pytest.raises(ValueError, match="k_theta"):
        kernel(spec, points)


@pytest.mark.parametrize(
    "bad_points", [np.zeros(3), np.zeros((2, 2)), np.zeros((1, 3, 1))]
)
@pytest.mark.parametrize(
    "kernel,spec",
    [
        (dommaschk_B, _build_dommaschk_spec(((5, 3),), ((0.75, -0.5),))),
        (dommaschk_dB, _build_dommaschk_spec(((5, 3),), ((0.75, -0.5),))),
        (reiman_B, _build_reiman_spec()),
        (reiman_dB, _build_reiman_spec()),
    ],
)
def test_analytic_field_kernels_reject_bad_point_shape(kernel, spec, bad_points):
    with pytest.raises(ValueError, match=r"shape \[N, 3\]"):
        kernel(spec, bad_points)


def test_reiman_closed_form():
    """Replicate the closed-form expression from ``test_Reiman``.

    Mirrors the upstream Bx/By/Bz expressions for the single ``k=6``,
    ``epsilon=0.01`` Reiman case at the same set of evaluation points
    used in the existing CPU test.
    """

    rng = np.random.default_rng(seed=20260512)
    npoints = 20
    base = np.asarray(npoints * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
    pointVar = 1e-1
    points = base + pointVar * (rng.random(base.shape) - 0.5)

    spec = _build_reiman_spec()
    B_jax = np.asarray(reiman_B(spec, points))

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    sqxy = np.sqrt(x**2 + y**2)
    rmin_sq = (-1 + sqxy) ** 2 + z**2
    rmin_4 = rmin_sq**2
    phi = np.arctan2(y, x)
    inner = phi - 6 * np.arctan(z / (-1 + sqxy))
    Bx = (
        y * sqxy
        + x * z * (0.15 + 0.38 * rmin_sq - 0.06 * rmin_4 * np.cos(inner))
        + 0.06 * x * (1 - sqxy) * rmin_4 * np.sin(inner)
    ) / (x**2 + y**2)
    By = (
        -x * sqxy
        + y * z * (0.15 + 0.38 * rmin_sq - 0.06 * rmin_4 * np.cos(inner))
        + 0.06 * y * (1 - sqxy) * rmin_4 * np.sin(inner)
    ) / (x**2 + y**2)
    Bz = (
        -((-1 + sqxy) * (0.15 + 0.38 * rmin_sq - 0.06 * rmin_4 * np.cos(inner)))
        - 0.06 * z * rmin_4 * np.sin(inner)
    ) / sqxy
    B_closed = np.column_stack([Bx, By, Bz])

    np.testing.assert_allclose(B_jax, B_closed, rtol=_DIRECT_RTOL, atol=_DIRECT_ATOL)


def test_reiman_cpp_cross_oracle():
    """Direct parity vs ``sopp.ReimanB`` and ``sopp.ReimandB``."""

    rng = np.random.default_rng(seed=314159)
    npoints = 20
    base = np.asarray(npoints * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
    pointVar = 1e-1
    points = base + pointVar * (rng.random(base.shape) - 0.5)

    spec = _build_reiman_spec()
    B_jax = np.asarray(reiman_B(spec, points))
    dB_jax = np.asarray(reiman_dB(spec, points))

    iota0 = float(_REIMAN_IOTA0)
    iota1 = float(_REIMAN_IOTA1)
    k_arr = np.asarray(_REIMAN_K, dtype=np.int32)
    eps_arr = np.asarray(_REIMAN_EPSILONK, dtype=np.float64)
    m0 = int(_REIMAN_M0)
    B_cpp = np.asarray(sopp.ReimanB(iota0, iota1, k_arr, eps_arr, m0, points))
    dB_cpp = np.asarray(sopp.ReimandB(iota0, iota1, k_arr, eps_arr, m0, points))

    assert B_jax.shape == B_cpp.shape
    assert dB_jax.shape == dB_cpp.shape
    np.testing.assert_allclose(B_jax, B_cpp, rtol=_DIRECT_RTOL, atol=_DIRECT_ATOL)
    np.testing.assert_allclose(dB_jax, dB_cpp, rtol=_DIRECT_RTOL, atol=_DIRECT_ATOL)


def test_reiman_odd_k_quadrant_parity_vs_cpp():
    """Odd ``k`` pins the arctan2 quadrant convention against the C++ oracle."""

    spec = ReimanSpec(
        iota0=0.15,
        iota1=0.38,
        k_theta=(5,),
        epsilon=np.asarray((0.01,), dtype=np.float64),
        m0_symmetry=1,
    )
    points = np.ascontiguousarray(
        np.array(
            [
                [0.72, 0.31, 0.22],
                [0.63, -0.42, -0.18],
                [1.24, 0.27, 0.19],
            ],
            dtype=np.float64,
        )
    )

    B_jax = np.asarray(reiman_B(spec, points))
    dB_jax = np.asarray(reiman_dB(spec, points))
    B_cpp = np.asarray(
        sopp.ReimanB(0.15, 0.38, np.asarray([5], dtype=np.int32), spec.epsilon, 1, points)
    )
    dB_cpp = np.asarray(
        sopp.ReimandB(
            0.15, 0.38, np.asarray([5], dtype=np.int32), spec.epsilon, 1, points
        )
    )

    np.testing.assert_allclose(B_jax, B_cpp, rtol=_DIRECT_RTOL, atol=_DIRECT_ATOL)
    np.testing.assert_allclose(dB_jax, dB_cpp, rtol=_DIRECT_RTOL, atol=_DIRECT_ATOL)


def test_reiman_cylindrical_B_phi_is_minus_one():
    spec = ReimanSpec(
        iota0=0.15,
        iota1=0.38,
        k_theta=(5,),
        epsilon=np.asarray((0.01,), dtype=np.float64),
        m0_symmetry=1,
    )
    points = np.ascontiguousarray(
        np.array(
            [
                [0.72, 0.31, 0.22],
                [0.63, -0.42, -0.18],
                [1.24, 0.27, 0.19],
            ],
            dtype=np.float64,
        )
    )

    B = np.asarray(reiman_B(spec, points))
    phi = np.arctan2(points[:, 1], points[:, 0])
    B_phi = -np.sin(phi) * B[:, 0] + np.cos(phi) * B[:, 1]

    np.testing.assert_allclose(
        B_phi,
        -np.ones(points.shape[0], dtype=np.float64),
        rtol=_DIRECT_RTOL,
        atol=_DIRECT_ATOL,
    )


def test_reiman_axis_ring_grad_nan_contract_and_near_axis_tolerance():
    spec = ReimanSpec(
        iota0=0.15,
        iota1=0.38,
        k_theta=(5,),
        epsilon=np.asarray((0.01,), dtype=np.float64),
        m0_symmetry=1,
    )

    def scalar(point):
        return jnp.sum(reiman_B(spec, point[None, :])[0])

    axis_ring_point = jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64)
    axis_grad = np.asarray(jax.grad(scalar)(axis_ring_point))
    assert not np.all(np.isfinite(axis_grad))

    near_axis_point = jnp.asarray([1.0 + 1.0e-4, 2.0e-4, -1.5e-4], dtype=jnp.float64)
    jac = jax.jacfwd(lambda point: reiman_B(spec, point[None, :])[0])(near_axis_point)
    dB = reiman_dB(spec, near_axis_point[None, :])[0]
    np.testing.assert_allclose(
        np.asarray(jac).T,
        np.asarray(dB),
        rtol=_DERIVATIVE_RTOL,
        atol=_DERIVATIVE_ATOL,
    )


@pytest.mark.parametrize("idx", [0, 16])
def test_reiman_dB_taylor(idx):
    """Central finite-difference Taylor test for ``reiman_dB``.

    Mirrors ``subtest_reiman_dBdX_taylortest`` from
    ``tests/field/test_magneticfields.py`` (lines 1005-1027), upgraded
    to central differences so the convergence-rate floor matches the
    ``fd_gradient`` lane's ``central_fd_error_rate``. Central FD has an
    ``O(eps^2)`` truncation error, so successive eps-halvings shrink
    the error by a factor of ``1/4`` -- comfortably under the lane's
    ``0.4`` ceiling.
    """

    rng = np.random.default_rng(seed=11)
    npoints = 17
    base = np.asarray(npoints * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
    points = base + 0.001 * (rng.random(base.shape) - 0.5)
    spec = _build_reiman_spec()

    dB = np.asarray(reiman_dB(spec, points))[idx]

    for direction in (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ):
        # ``dB[i, j] = d B_j / d x_i``; directional derivative along
        # ``direction`` is ``direction @ dB``.
        deriv = dB.T @ direction
        err = 1e6
        for i in range(5, 10):
            eps = 0.5**i
            Bplus = np.asarray(reiman_B(spec, points + eps * direction))[idx]
            Bminus = np.asarray(reiman_B(spec, points - eps * direction))[idx]
            deriv_est = (Bplus - Bminus) / (2.0 * eps)
            new_err = float(np.linalg.norm(deriv - deriv_est))
            assert new_err < _FD_ERROR_RATE * err + np.finfo(float).eps
            err = new_err
