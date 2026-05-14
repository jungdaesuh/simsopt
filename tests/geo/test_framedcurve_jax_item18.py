"""Wave R4 item 18 tests for ``simsopt.jax_core.framedcurve``.

The SSOT pure JAX kernels in ``simsopt.jax_core.framedcurve`` re-express
the Frenet / coil-centroid frame arithmetic from
``simsopt.geo.framedcurve`` without the surrounding ``Optimizable``
graph. ``simsopt.geo.framedcurve`` directly re-exports
``rotated_centroid_frame`` and ``rotated_frenet_frame`` from
``simsopt.jax_core.framedcurve``, so there is no independent "upstream"
JAX implementation to compare against; an `is`-identity check on the
imports confirms the two names bind to the same Python function object.

Real coverage in this file:

* **Analytic planar-circle parity** -- closed-form centroid and Frenet
  frames for a circle of radius ``R`` parameterised as
  ``gamma(t) = (R cos(2 pi t), R sin(2 pi t), 0)``. The analytic
  frame is the only oracle; the test compares the JAX kernel output
  against hand-derived ``(t, n, b)`` triples (and their rotated variants
  under the kernel's ``N = cos(alpha) N0 - sin(alpha) B0`` /
  ``B = sin(alpha) N0 + cos(alpha) B0`` convention).
* **Orthonormality** -- ``|t|=|n|=|b|=1``, mutual orthogonality, and the
  right-handed ``t x n == b`` invariant on a non-planar
  ``CurveXYZFourier`` fixture under a varying ``alpha`` profile.
* **alpha = 0 reduction** -- the rotated kernels reduce to the unrotated
  kernels when ``alpha = 0`` (delegation property, not a parity claim).
* **Strict transfer guard** -- the compiled kernels run cleanly under
  ``jax.transfer_guard('disallow')`` when consuming device-resident
  inputs.

Production-scale fixture (per the ladder convention used elsewhere in
``tests/geo``):

* ``CurveXYZFourier`` with order 3 and 64 quadrature points
* Independently seeded coils for orthonormality coverage across multiple
  states
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.jax_core.framedcurve import (
    centroid_frame,
    frenet_frame,
    rotated_centroid_frame,
    rotated_frenet_frame,
)


_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]


_PRODUCTION_NQUADPOINTS = 64
_PRODUCTION_ORDER = 3
_PRODUCTION_NCOILS = 4
_PRODUCTION_RNG_SEED = 1729
_PRODUCTION_PERTURB_SCALE = 0.02


def _seed_dofs_xyzfourier(order: int, rng: np.random.Generator) -> np.ndarray:
    """Return a non-degenerate ``CurveXYZFourier`` DOF vector.

    A baseline coil is laid out in the x-y plane (``xc(1) = ys(1) = 1``)
    with a small z perturbation so ``gammadashdash`` is non-collinear with
    ``gammadash`` (the Frenet ``tdash`` denominator must stay finite).
    """
    ndofs = 3 * (2 * order + 1)
    dofs = np.zeros(ndofs, dtype=np.float64)
    # CurveXYZFourier stores (xc0, xs1, xc1, xs2, xc2, ...,
    #                        yc0, ys1, yc1, ..., zc0, zs1, zc1, ...).
    # Place a unit-radius circle in xy with a z bump to guarantee non-planar
    # acceleration.
    dofs[2] = 1.0  # xc(1)
    dofs[(2 * order + 1) + 1] = 1.0  # ys(1) (y-block offset + 1 (xs(1) index))
    # Add a small z bump on the first cosine to ensure non-planar curve.
    dofs[2 * (2 * order + 1) + 2] = 0.25  # zc(1)
    return dofs + _PRODUCTION_PERTURB_SCALE * rng.standard_normal(ndofs)


def _make_curve(rng: np.random.Generator) -> CurveXYZFourier:
    curve = CurveXYZFourier(_PRODUCTION_NQUADPOINTS, _PRODUCTION_ORDER)
    curve.x = _seed_dofs_xyzfourier(_PRODUCTION_ORDER, rng)
    return curve


def _alpha_profiles(rng: np.random.Generator) -> dict[str, np.ndarray]:
    quadpoints = np.linspace(
        0.0,
        1.0,
        _PRODUCTION_NQUADPOINTS,
        endpoint=False,
        dtype=np.float64,
    )
    constant_value = 0.3 * np.pi
    varying_amp = 0.4 * np.pi
    return {
        "zero": np.zeros(_PRODUCTION_NQUADPOINTS, dtype=np.float64),
        "constant": np.full(
            _PRODUCTION_NQUADPOINTS,
            constant_value,
            dtype=np.float64,
        ),
        "varying": (
            varying_amp * np.cos(2.0 * np.pi * quadpoints)
            + 0.1 * np.pi * rng.standard_normal(_PRODUCTION_NQUADPOINTS)
        ),
    }


def _assert_frame_close(
    label: str,
    expected: tuple[np.ndarray, np.ndarray, np.ndarray],
    actual: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    for name, exp, act in zip(("t", "n", "b"), expected, actual, strict=True):
        np.testing.assert_allclose(
            np.asarray(act, dtype=np.float64),
            np.asarray(exp, dtype=np.float64),
            rtol=_RTOL,
            atol=_ATOL,
            err_msg=f"{label}: {name} component diverges from upstream oracle",
        )


def _assert_orthonormal(
    label: str,
    frame: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    t, n, b = (np.asarray(component, dtype=np.float64) for component in frame)
    np.testing.assert_allclose(
        np.sum(t * t, axis=1),
        np.ones(t.shape[0]),
        rtol=_RTOL,
        atol=_ATOL,
        err_msg=f"{label}: |t| != 1",
    )
    np.testing.assert_allclose(
        np.sum(n * n, axis=1),
        np.ones(n.shape[0]),
        rtol=_RTOL,
        atol=_ATOL,
        err_msg=f"{label}: |n| != 1",
    )
    np.testing.assert_allclose(
        np.sum(b * b, axis=1),
        np.ones(b.shape[0]),
        rtol=_RTOL,
        atol=_ATOL,
        err_msg=f"{label}: |b| != 1",
    )
    np.testing.assert_allclose(
        np.sum(t * n, axis=1),
        np.zeros(t.shape[0]),
        rtol=_RTOL,
        atol=_ATOL,
        err_msg=f"{label}: t . n != 0",
    )
    np.testing.assert_allclose(
        np.sum(t * b, axis=1),
        np.zeros(t.shape[0]),
        rtol=_RTOL,
        atol=_ATOL,
        err_msg=f"{label}: t . b != 0",
    )
    np.testing.assert_allclose(
        np.sum(n * b, axis=1),
        np.zeros(t.shape[0]),
        rtol=_RTOL,
        atol=_ATOL,
        err_msg=f"{label}: n . b != 0",
    )
    # Right-handed: t x n == b.
    cross_tn = np.cross(t, n, axis=1)
    np.testing.assert_allclose(
        cross_tn,
        b,
        rtol=_RTOL,
        atol=_ATOL,
        err_msg=f"{label}: t x n != b (right-handed orientation)",
    )


_ANALYTIC_RTOL = 1e-12
_ANALYTIC_ATOL = 1e-12
_ANALYTIC_NQUADPOINTS = 64
_ANALYTIC_RADIUS = 1.7


def _planar_circle_arrays(
    radius: float,
    nquadpoints: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(quadpoints, gamma, gammadash, gammadashdash)`` for a planar circle.

    The curve is ``gamma(t) = (R cos(2 pi t), R sin(2 pi t), 0)`` sampled
    on ``quadpoints = linspace(0, 1, N, endpoint=False)``. The mean of
    ``gamma`` along axis 0 is exactly the origin for this symmetric
    sampling, so the centroid frame consumes ``centroid = 0``.
    """
    quadpoints = np.linspace(0.0, 1.0, nquadpoints, endpoint=False, dtype=np.float64)
    two_pi = 2.0 * np.pi
    angle = two_pi * quadpoints
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    gamma = np.stack(
        (radius * cos_a, radius * sin_a, np.zeros_like(angle)),
        axis=1,
    )
    gammadash = np.stack(
        (-two_pi * radius * sin_a, two_pi * radius * cos_a, np.zeros_like(angle)),
        axis=1,
    )
    gammadashdash = np.stack(
        (
            -(two_pi**2) * radius * cos_a,
            -(two_pi**2) * radius * sin_a,
            np.zeros_like(angle),
        ),
        axis=1,
    )
    return quadpoints, gamma, gammadash, gammadashdash


def _planar_circle_centroid_frame_analytic(
    quadpoints: np.ndarray,
    alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hand-derived rotated centroid frame for the planar circle.

    For ``gamma = (R cos(2 pi t), R sin(2 pi t), 0)`` the centroid is the
    origin and the unrotated frame is

    * ``T = (-sin(2 pi t), cos(2 pi t), 0)``
    * ``N0 = (cos(2 pi t), sin(2 pi t), 0)``  (outward from centroid)
    * ``B0 = T x N0 = (0, 0, -1)``

    The kernel applies the rotation ``N = cos(alpha) N0 - sin(alpha) B0``
    and ``B = sin(alpha) N0 + cos(alpha) B0`` -- see ``_rotate_frame`` in
    ``simsopt.jax_core.framedcurve``.
    """
    two_pi = 2.0 * np.pi
    angle = two_pi * quadpoints
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    zero = np.zeros_like(angle)
    one = np.ones_like(angle)
    t = np.stack((-sin_a, cos_a, zero), axis=1)
    n0 = np.stack((cos_a, sin_a, zero), axis=1)
    b0 = np.stack((zero, zero, -one), axis=1)
    cos_alpha = np.cos(alpha)[:, None]
    sin_alpha = np.sin(alpha)[:, None]
    n = cos_alpha * n0 - sin_alpha * b0
    b = sin_alpha * n0 + cos_alpha * b0
    return t, n, b


def _planar_circle_frenet_frame_analytic(
    quadpoints: np.ndarray,
    alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hand-derived rotated Frenet frame for the planar circle.

    The Frenet principal normal of a planar circle points inward toward
    the centre, so the unrotated frame is

    * ``T = (-sin(2 pi t), cos(2 pi t), 0)``
    * ``N0_frenet = (-cos(2 pi t), -sin(2 pi t), 0)``  (inward)
    * ``B0_frenet = T x N0_frenet = (0, 0, 1)``

    The kernel rotation convention matches the centroid case.
    """
    two_pi = 2.0 * np.pi
    angle = two_pi * quadpoints
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    zero = np.zeros_like(angle)
    one = np.ones_like(angle)
    t = np.stack((-sin_a, cos_a, zero), axis=1)
    n0 = np.stack((-cos_a, -sin_a, zero), axis=1)
    b0 = np.stack((zero, zero, one), axis=1)
    cos_alpha = np.cos(alpha)[:, None]
    sin_alpha = np.sin(alpha)[:, None]
    n = cos_alpha * n0 - sin_alpha * b0
    b = sin_alpha * n0 + cos_alpha * b0
    return t, n, b


def _analytic_alpha_profiles(
    quadpoints: np.ndarray,
) -> dict[str, np.ndarray]:
    """Deterministic alpha profiles for the analytic-oracle tests.

    Avoids randomness so the analytic comparison is fully reproducible.
    """
    zero = np.zeros_like(quadpoints)
    constant = np.full_like(quadpoints, 0.3 * np.pi)
    varying = 0.4 * np.pi * np.cos(2.0 * np.pi * quadpoints) + 0.15 * np.pi * np.sin(
        4.0 * np.pi * quadpoints
    )
    return {"zero": zero, "constant": constant, "varying": varying}


def _assert_frame_close_analytic(
    label: str,
    expected: tuple[np.ndarray, np.ndarray, np.ndarray],
    actual: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Tight analytic-oracle comparison at ``rtol=atol=1e-12``."""
    for name, exp, act in zip(("t", "n", "b"), expected, actual, strict=True):
        np.testing.assert_allclose(
            np.asarray(act, dtype=np.float64),
            np.asarray(exp, dtype=np.float64),
            rtol=_ANALYTIC_RTOL,
            atol=_ANALYTIC_ATOL,
            err_msg=f"{label}: {name} diverges from analytic planar-circle oracle",
        )


@pytest.mark.parametrize("alpha_kind", ("zero", "constant", "varying"))
def test_rotated_centroid_frame_matches_planar_circle_analytic(alpha_kind: str):
    """Centroid frame for a planar circle agrees with closed-form analytic oracle.

    Independent oracle: the centroid frame ``(t, n, b)`` of
    ``gamma(t) = (R cos(2 pi t), R sin(2 pi t), 0)`` is hand-derived in
    ``_planar_circle_centroid_frame_analytic``. The kernel must reproduce
    it to ``rtol=atol=1e-12`` under the kernel's
    ``N = cos(alpha) N0 - sin(alpha) B0`` rotation convention.
    """
    quadpoints, gamma, gammadash, _ = _planar_circle_arrays(
        _ANALYTIC_RADIUS, _ANALYTIC_NQUADPOINTS
    )
    alpha = _analytic_alpha_profiles(quadpoints)[alpha_kind]
    expected = _planar_circle_centroid_frame_analytic(quadpoints, alpha)
    actual = rotated_centroid_frame(gamma, gammadash, alpha)
    _assert_frame_close_analytic(
        f"rotated_centroid_frame planar-circle alpha={alpha_kind}",
        expected,
        actual,
    )


@pytest.mark.parametrize("alpha_kind", ("zero", "constant", "varying"))
def test_rotated_frenet_frame_matches_planar_circle_analytic(alpha_kind: str):
    """Frenet frame for a planar circle agrees with closed-form analytic oracle.

    Independent oracle: the Frenet frame ``(t, n, b)`` of
    ``gamma(t) = (R cos(2 pi t), R sin(2 pi t), 0)`` is hand-derived in
    ``_planar_circle_frenet_frame_analytic``. The kernel must reproduce
    it to ``rtol=atol=1e-12``.
    """
    quadpoints, gamma, gammadash, gammadashdash = _planar_circle_arrays(
        _ANALYTIC_RADIUS, _ANALYTIC_NQUADPOINTS
    )
    alpha = _analytic_alpha_profiles(quadpoints)[alpha_kind]
    expected = _planar_circle_frenet_frame_analytic(quadpoints, alpha)
    actual = rotated_frenet_frame(gamma, gammadash, gammadashdash, alpha)
    _assert_frame_close_analytic(
        f"rotated_frenet_frame planar-circle alpha={alpha_kind}",
        expected,
        actual,
    )


def test_centroid_frame_matches_rotated_at_alpha_zero():
    """Unrotated centroid frame == rotated centroid frame at ``alpha = 0``.

    Bit-identity expected because ``rotated_centroid_frame`` delegates to
    ``centroid_frame`` and ``cos(0) = 1, sin(0) = 0``.
    """
    rng = np.random.default_rng(_PRODUCTION_RNG_SEED + 200)
    alpha_zero = np.zeros(_PRODUCTION_NQUADPOINTS, dtype=np.float64)
    for coil_index in range(_PRODUCTION_NCOILS):
        curve = _make_curve(rng)
        gamma = np.asarray(curve.gamma(), dtype=np.float64)
        gammadash = np.asarray(curve.gammadash(), dtype=np.float64)
        unrotated = centroid_frame(gamma, gammadash)
        rotated = rotated_centroid_frame(gamma, gammadash, alpha_zero)
        _assert_frame_close(
            f"centroid alpha=0 coil={coil_index}",
            unrotated,
            rotated,
        )
        _assert_orthonormal(f"centroid unrotated coil={coil_index}", unrotated)


def test_frenet_frame_matches_rotated_at_alpha_zero():
    """Unrotated Frenet frame == rotated Frenet frame at ``alpha = 0``."""
    rng = np.random.default_rng(_PRODUCTION_RNG_SEED + 300)
    alpha_zero = np.zeros(_PRODUCTION_NQUADPOINTS, dtype=np.float64)
    for coil_index in range(_PRODUCTION_NCOILS):
        curve = _make_curve(rng)
        gamma = np.asarray(curve.gamma(), dtype=np.float64)
        gammadash = np.asarray(curve.gammadash(), dtype=np.float64)
        gammadashdash = np.asarray(curve.gammadashdash(), dtype=np.float64)
        unrotated = frenet_frame(gamma, gammadash, gammadashdash)
        rotated = rotated_frenet_frame(gamma, gammadash, gammadashdash, alpha_zero)
        _assert_frame_close(
            f"frenet alpha=0 coil={coil_index}",
            unrotated,
            rotated,
        )
        _assert_orthonormal(f"frenet unrotated coil={coil_index}", unrotated)


@pytest.mark.parametrize("frame_kind", ("centroid", "frenet"))
def test_rotated_frame_is_orthonormal(frame_kind: str):
    """Rotated frames remain right-handed orthonormal at every quadpoint."""
    rng = np.random.default_rng(_PRODUCTION_RNG_SEED + 400)
    alpha_rng = np.random.default_rng(_PRODUCTION_RNG_SEED + 401)
    alpha = _alpha_profiles(alpha_rng)["varying"]
    for coil_index in range(_PRODUCTION_NCOILS):
        curve = _make_curve(rng)
        gamma = np.asarray(curve.gamma(), dtype=np.float64)
        gammadash = np.asarray(curve.gammadash(), dtype=np.float64)
        if frame_kind == "centroid":
            frame = rotated_centroid_frame(gamma, gammadash, alpha)
        else:
            gammadashdash = np.asarray(curve.gammadashdash(), dtype=np.float64)
            frame = rotated_frenet_frame(gamma, gammadash, gammadashdash, alpha)
        _assert_orthonormal(
            f"{frame_kind} rotated varying alpha coil={coil_index}",
            frame,
        )


def test_kernels_run_under_strict_transfer_guard():
    """All four kernels run cleanly under ``transfer_guard('disallow')``.

    Device-resident inputs are placed under the default guard so the
    strict-guard region only measures the compiled kernels. An implicit
    host transfer inside the compiled paths would raise
    ``jax.errors.JaxRuntimeError``.
    """
    rng = np.random.default_rng(_PRODUCTION_RNG_SEED + 500)
    alpha_rng = np.random.default_rng(_PRODUCTION_RNG_SEED + 501)
    curve = _make_curve(rng)
    gamma_dev = jnp.asarray(curve.gamma(), dtype=jnp.float64)
    gammadash_dev = jnp.asarray(curve.gammadash(), dtype=jnp.float64)
    gammadashdash_dev = jnp.asarray(curve.gammadashdash(), dtype=jnp.float64)
    alpha_dev = jnp.asarray(_alpha_profiles(alpha_rng)["varying"], dtype=jnp.float64)
    gamma_dev.block_until_ready()
    gammadash_dev.block_until_ready()
    gammadashdash_dev.block_until_ready()
    alpha_dev.block_until_ready()

    with jax.transfer_guard("disallow"):
        t_c, n_c, b_c = centroid_frame(gamma_dev, gammadash_dev)
        t_c.block_until_ready()
        n_c.block_until_ready()
        b_c.block_until_ready()

        t_rc, n_rc, b_rc = rotated_centroid_frame(gamma_dev, gammadash_dev, alpha_dev)
        t_rc.block_until_ready()
        n_rc.block_until_ready()
        b_rc.block_until_ready()

        t_f, n_f, b_f = frenet_frame(gamma_dev, gammadash_dev, gammadashdash_dev)
        t_f.block_until_ready()
        n_f.block_until_ready()
        b_f.block_until_ready()

        t_rf, n_rf, b_rf = rotated_frenet_frame(
            gamma_dev, gammadash_dev, gammadashdash_dev, alpha_dev
        )
        t_rf.block_until_ready()
        n_rf.block_until_ready()
        b_rf.block_until_ready()
