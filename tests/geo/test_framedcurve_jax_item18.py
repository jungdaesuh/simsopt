"""Wave R4 item 18 parity tests for ``simsopt.jax_core.framedcurve``.

The SSOT pure JAX kernels in ``simsopt.jax_core.framedcurve`` re-express
the Frenet / coil-centroid frame arithmetic from
``simsopt.geo.framedcurve`` without the surrounding ``Optimizable``
graph. These tests pin the new kernels to bit-identity with the upstream
JAX evaluation at the ``direct_kernel`` parity-ladder lane (no inline
``rtol`` / ``atol`` literals), and verify orthonormality plus the
``alpha = 0`` reduction to the unrotated variant.

Production-scale fixture (per the ladder convention used elsewhere in
``tests/geo``):

* ``CurveXYZFourier`` with order 3 and 64 quadrature points
* Independently seeded coils for parity coverage across multiple states
* Each kernel exercised under ``alpha = 0``, ``alpha = const != 0``, and
  ``alpha = varying`` profiles

A separate test replays the same fixtures inside
``jax.transfer_guard("disallow")`` to confirm the compiled kernels do
not trigger implicit host transfers when consuming device-resident
inputs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.framedcurve import (
    rotated_centroid_frame as upstream_rotated_centroid_frame,
    rotated_frenet_frame as upstream_rotated_frenet_frame,
)
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


@pytest.mark.parametrize("alpha_kind", ("zero", "constant", "varying"))
def test_rotated_centroid_frame_matches_upstream(alpha_kind: str):
    """Per-coil centroid-frame parity vs upstream JAX oracle.

    Direct kernel-level parity at ``direct_kernel`` tolerance over all
    three alpha profiles, four independently seeded coils.
    """
    rng = np.random.default_rng(_PRODUCTION_RNG_SEED)
    alpha_rng = np.random.default_rng(_PRODUCTION_RNG_SEED + 1)
    alphas = _alpha_profiles(alpha_rng)
    alpha = alphas[alpha_kind]
    for coil_index in range(_PRODUCTION_NCOILS):
        curve = _make_curve(rng)
        gamma = np.asarray(curve.gamma(), dtype=np.float64)
        gammadash = np.asarray(curve.gammadash(), dtype=np.float64)
        expected = upstream_rotated_centroid_frame(gamma, gammadash, alpha)
        actual = rotated_centroid_frame(gamma, gammadash, alpha)
        _assert_frame_close(
            f"rotated_centroid_frame coil={coil_index} alpha={alpha_kind}",
            expected,
            actual,
        )


@pytest.mark.parametrize("alpha_kind", ("zero", "constant", "varying"))
def test_rotated_frenet_frame_matches_upstream(alpha_kind: str):
    """Per-coil Frenet-frame parity vs upstream JAX oracle.

    Direct kernel-level parity at ``direct_kernel`` tolerance over all
    three alpha profiles, four independently seeded coils.
    """
    rng = np.random.default_rng(_PRODUCTION_RNG_SEED + 100)
    alpha_rng = np.random.default_rng(_PRODUCTION_RNG_SEED + 101)
    alphas = _alpha_profiles(alpha_rng)
    alpha = alphas[alpha_kind]
    for coil_index in range(_PRODUCTION_NCOILS):
        curve = _make_curve(rng)
        gamma = np.asarray(curve.gamma(), dtype=np.float64)
        gammadash = np.asarray(curve.gammadash(), dtype=np.float64)
        gammadashdash = np.asarray(curve.gammadashdash(), dtype=np.float64)
        expected = upstream_rotated_frenet_frame(gamma, gammadash, gammadashdash, alpha)
        actual = rotated_frenet_frame(gamma, gammadash, gammadashdash, alpha)
        _assert_frame_close(
            f"rotated_frenet_frame coil={coil_index} alpha={alpha_kind}",
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
