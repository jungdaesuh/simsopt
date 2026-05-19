"""Parity tests for the JAX ``boozer_radial_interp`` port (item 32).

The cross-oracle tests compare the JAX kernels against the C++
``simsoptpp.compute_kmnc_kmns``, ``simsoptpp.compute_kmns``,
``simsoptpp.fourier_transform_{odd,even}`` and
``simsoptpp.inverse_fourier_transform_{odd,even}`` bindings.

Tolerances come from
``benchmarks.validation_ladder_contract.parity_ladder_tolerances`` so the
parity-ladder contract is preserved end-to-end. All kernels are
closed-form same-state evaluations and route through the
``direct_kernel`` lane (``rtol=1e-10``, ``atol=1e-12``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import simsoptpp as sopp

from benchmarks.validation_ladder_contract import parity_ladder_tolerances

from simsopt.jax_core.boozer_radial_interp import (
    _build_angle_basis,
    _compute_K_per_point,
    compute_kmnc_kmns,
    compute_kmns,
    fourier_transform_even,
    fourier_transform_odd,
    inverse_fourier_transform_even,
    inverse_fourier_transform_even_1d,
    inverse_fourier_transform_even_2d,
    inverse_fourier_transform_odd,
    inverse_fourier_transform_odd_1d,
    inverse_fourier_transform_odd_2d,
)
from .jaxpr_utils import count_jaxpr_primitives


_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = float(_DIRECT_KERNEL["rtol"])
_ATOL = float(_DIRECT_KERNEL["atol"])


# ----------------------------------------------------------------------
# Synthetic input builders
# ----------------------------------------------------------------------


def _make_modes(num_modes: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Reproducible (xm, xn) generator. ``xm[0]==xn[0]==0`` is the DC mode."""
    rng = np.random.RandomState(seed)
    xm = np.zeros(num_modes, dtype=np.float64)
    xn = np.zeros(num_modes, dtype=np.float64)
    # First mode is the DC mode; remaining modes pick poloidal/toroidal
    # mode numbers in a representative range.
    if num_modes > 1:
        xm[1:] = rng.choice(np.arange(0, 5), size=num_modes - 1)
        xn[1:] = rng.choice(np.arange(-3, 4), size=num_modes - 1)
    return xm, xn


def _make_quad_points(num_points: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a structured (theta, zeta) grid scaled to (num_points,)."""
    rng = np.random.RandomState(seed)
    thetas = rng.uniform(0.0, 2.0 * np.pi, size=num_points)
    zetas = rng.uniform(0.0, 2.0 * np.pi, size=num_points)
    return thetas, zetas


def _make_half_grid_fields(
    *,
    num_modes: int,
    num_surf: int,
    seed: int,
    stellsym: bool,
) -> dict[str, np.ndarray]:
    """Build a representative half-grid Fourier coefficient bundle.

    The coefficient magnitudes are small enough to keep ``B(theta,zeta) > 0``
    almost surely (DC bias ~ 1.0, fluctuation magnitude ~ 0.05).
    """
    rng = np.random.RandomState(seed)
    bundle: dict[str, np.ndarray] = {}

    # Cos-symmetric (stellsym) coefficients.
    bundle["rmnc"] = 0.3 * rng.randn(num_modes, num_surf)
    bundle["rmnc"][0, :] += 2.0  # mean major radius ~ 2.0
    bundle["drmncds"] = 0.1 * rng.randn(num_modes, num_surf)
    bundle["zmns"] = 0.3 * rng.randn(num_modes, num_surf)
    bundle["dzmnsds"] = 0.1 * rng.randn(num_modes, num_surf)
    bundle["numns"] = 0.1 * rng.randn(num_modes, num_surf)
    bundle["dnumnsds"] = 0.05 * rng.randn(num_modes, num_surf)
    bundle["bmnc"] = 0.05 * rng.randn(num_modes, num_surf)
    bundle["bmnc"][0, :] += 1.0  # mean B0 ~ 1.0

    if not stellsym:
        bundle["rmns"] = 0.1 * rng.randn(num_modes, num_surf)
        bundle["drmnsds"] = 0.05 * rng.randn(num_modes, num_surf)
        bundle["zmnc"] = 0.1 * rng.randn(num_modes, num_surf)
        bundle["dzmncds"] = 0.05 * rng.randn(num_modes, num_surf)
        bundle["numnc"] = 0.05 * rng.randn(num_modes, num_surf)
        bundle["dnumncds"] = 0.02 * rng.randn(num_modes, num_surf)
        bundle["bmns"] = 0.02 * rng.randn(num_modes, num_surf)

    bundle["iota"] = 0.5 + 0.05 * rng.randn(num_surf)
    bundle["G"] = 2.0 + 0.05 * rng.randn(num_surf)
    bundle["I"] = 0.1 + 0.02 * rng.randn(num_surf)
    return bundle


def _jax_compute_K_per_point(
    *,
    xm: np.ndarray,
    xn: np.ndarray,
    thetas: np.ndarray,
    zetas: np.ndarray,
    bundle: dict[str, np.ndarray],
    stellsym: bool,
) -> jax.Array:
    cos_a, sin_a = _build_angle_basis(
        jnp.asarray(xm),
        jnp.asarray(xn),
        jnp.asarray(thetas),
        jnp.asarray(zetas),
    )
    kwargs = {
        "cos_a": cos_a,
        "sin_a": sin_a,
        "xm": jnp.asarray(xm),
        "xn": jnp.asarray(xn),
        "rmnc": jnp.asarray(bundle["rmnc"][:, 0]),
        "drmncds": jnp.asarray(bundle["drmncds"][:, 0]),
        "zmns": jnp.asarray(bundle["zmns"][:, 0]),
        "dzmnsds": jnp.asarray(bundle["dzmnsds"][:, 0]),
        "numns": jnp.asarray(bundle["numns"][:, 0]),
        "dnumnsds": jnp.asarray(bundle["dnumnsds"][:, 0]),
        "bmnc": jnp.asarray(bundle["bmnc"][:, 0]),
        "zetas": jnp.asarray(zetas),
        "iota_isurf": jnp.asarray(bundle["iota"][0]),
        "G_isurf": jnp.asarray(bundle["G"][0]),
        "I_isurf": jnp.asarray(bundle["I"][0]),
    }
    if not stellsym:
        kwargs |= {
            "rmns": jnp.asarray(bundle["rmns"][:, 0]),
            "drmnsds": jnp.asarray(bundle["drmnsds"][:, 0]),
            "zmnc": jnp.asarray(bundle["zmnc"][:, 0]),
            "dzmncds": jnp.asarray(bundle["dzmncds"][:, 0]),
            "numnc": jnp.asarray(bundle["numnc"][:, 0]),
            "dnumncds": jnp.asarray(bundle["dnumncds"][:, 0]),
            "bmns": jnp.asarray(bundle["bmns"][:, 0]),
        }
    return _compute_K_per_point(**kwargs)


def _k_per_point_jaxpr_dot_count(
    *,
    num_modes: int,
    num_points: int,
    modes_seed: int,
    points_seed: int,
    fields_seed: int,
    stellsym: bool,
) -> int:
    xm, xn = _make_modes(num_modes, seed=modes_seed)
    thetas, zetas = _make_quad_points(num_points, seed=points_seed)
    bundle = _make_half_grid_fields(
        num_modes=num_modes,
        num_surf=1,
        seed=fields_seed,
        stellsym=stellsym,
    )
    return count_jaxpr_primitives(
        jax.make_jaxpr(
            lambda: _jax_compute_K_per_point(
                xm=xm,
                xn=xn,
                thetas=thetas,
                zetas=zetas,
                bundle=bundle,
                stellsym=stellsym,
            )
        )(),
        "dot_general",
    )


def _numpy_compute_K_per_point(
    *,
    xm: np.ndarray,
    xn: np.ndarray,
    thetas: np.ndarray,
    zetas: np.ndarray,
    rmnc: np.ndarray,
    drmncds: np.ndarray,
    zmns: np.ndarray,
    dzmnsds: np.ndarray,
    numns: np.ndarray,
    dnumnsds: np.ndarray,
    bmnc: np.ndarray,
    iota_isurf: float,
    G_isurf: float,
    I_isurf: float,
    rmns: np.ndarray | None = None,
    drmnsds: np.ndarray | None = None,
    zmnc: np.ndarray | None = None,
    dzmncds: np.ndarray | None = None,
    numnc: np.ndarray | None = None,
    dnumncds: np.ndarray | None = None,
    bmns: np.ndarray | None = None,
) -> np.ndarray:
    angles = thetas[:, None] * xm[None, :] - zetas[:, None] * xn[None, :]
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    if rmns is None:
        B = cos_a @ bmnc
        R = cos_a @ rmnc
        dRdtheta = sin_a @ (-xm * rmnc)
        dRdzeta = sin_a @ (xn * rmnc)
        dRds = cos_a @ drmncds
        dZdtheta = cos_a @ (xm * zmns)
        dZdzeta = cos_a @ (-xn * zmns)
        dZds = sin_a @ dzmnsds
        nu = sin_a @ numns
        dnuds = sin_a @ dnumnsds
        dnudtheta = cos_a @ (xm * numns)
        dnudzeta = cos_a @ (-xn * numns)
    else:
        B = cos_a @ bmnc + sin_a @ bmns
        R = cos_a @ rmnc + sin_a @ rmns
        dRdtheta = sin_a @ (-xm * rmnc) + cos_a @ (xm * rmns)
        dRdzeta = sin_a @ (xn * rmnc) + cos_a @ (-xn * rmns)
        dRds = cos_a @ drmncds + sin_a @ drmnsds
        dZdtheta = cos_a @ (xm * zmns) + sin_a @ (-xm * zmnc)
        dZdzeta = cos_a @ (-xn * zmns) + sin_a @ (xn * zmnc)
        dZds = sin_a @ dzmnsds + cos_a @ dzmncds
        nu = sin_a @ numns + cos_a @ numnc
        dnuds = sin_a @ dnumnsds + cos_a @ dnumncds
        dnudtheta = cos_a @ (xm * numns) + sin_a @ (-xm * numnc)
        dnudzeta = cos_a @ (-xn * numns) + sin_a @ (xn * numnc)

    phi = zetas - nu
    dphids = -dnuds
    dphidtheta = -dnudtheta
    dphidzeta = 1.0 - dnudzeta
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    dXdtheta = dRdtheta * cos_phi - R * sin_phi * dphidtheta
    dYdtheta = dRdtheta * sin_phi + R * cos_phi * dphidtheta
    dXds = dRds * cos_phi - R * sin_phi * dphids
    dYds = dRds * sin_phi + R * cos_phi * dphids
    dXdzeta = dRdzeta * cos_phi - R * sin_phi * dphidzeta
    dYdzeta = dRdzeta * sin_phi + R * cos_phi * dphidzeta

    gstheta = dXdtheta * dXds + dYdtheta * dYds + dZdtheta * dZds
    gszeta = dXdzeta * dXds + dYdzeta * dYds + dZdzeta * dZds
    sqrtg = (G_isurf + iota_isurf * I_isurf) / (B * B)
    return (gszeta + iota_isurf * gstheta) / sqrtg


def test_compute_K_per_point_batches_stellsym_fourier_sums() -> None:
    num_modes = 8
    num_points = 16
    assert (
        _k_per_point_jaxpr_dot_count(
            num_modes=num_modes,
            num_points=num_points,
            modes_seed=3201,
            points_seed=3202,
            fields_seed=3203,
            stellsym=True,
        )
        == 2
    )


def test_compute_K_per_point_batches_asym_fourier_sums() -> None:
    num_modes = 8
    num_points = 16
    assert (
        _k_per_point_jaxpr_dot_count(
            num_modes=num_modes,
            num_points=num_points,
            modes_seed=3211,
            points_seed=3212,
            fields_seed=3213,
            stellsym=False,
        )
        == 2
    )


def test_compute_K_per_point_matches_closed_form_numpy_stellsym() -> None:
    num_modes = 9
    num_points = 21
    xm, xn = _make_modes(num_modes, seed=3221)
    thetas, zetas = _make_quad_points(num_points, seed=3222)
    bundle = _make_half_grid_fields(
        num_modes=num_modes,
        num_surf=1,
        seed=3223,
        stellsym=True,
    )

    actual = np.asarray(
        _jax_compute_K_per_point(
            xm=xm,
            xn=xn,
            thetas=thetas,
            zetas=zetas,
            bundle=bundle,
            stellsym=True,
        )
    )
    expected = _numpy_compute_K_per_point(
        xm=xm,
        xn=xn,
        thetas=thetas,
        zetas=zetas,
        rmnc=bundle["rmnc"][:, 0],
        drmncds=bundle["drmncds"][:, 0],
        zmns=bundle["zmns"][:, 0],
        dzmnsds=bundle["dzmnsds"][:, 0],
        numns=bundle["numns"][:, 0],
        dnumnsds=bundle["dnumnsds"][:, 0],
        bmnc=bundle["bmnc"][:, 0],
        iota_isurf=bundle["iota"][0],
        G_isurf=bundle["G"][0],
        I_isurf=bundle["I"][0],
    )

    np.testing.assert_allclose(actual, expected, rtol=_RTOL, atol=_ATOL)


def test_compute_K_per_point_matches_closed_form_numpy_asym() -> None:
    num_modes = 9
    num_points = 21
    xm, xn = _make_modes(num_modes, seed=3231)
    thetas, zetas = _make_quad_points(num_points, seed=3232)
    bundle = _make_half_grid_fields(
        num_modes=num_modes,
        num_surf=1,
        seed=3233,
        stellsym=False,
    )

    actual = np.asarray(
        _jax_compute_K_per_point(
            xm=xm,
            xn=xn,
            thetas=thetas,
            zetas=zetas,
            bundle=bundle,
            stellsym=False,
        )
    )
    expected = _numpy_compute_K_per_point(
        xm=xm,
        xn=xn,
        thetas=thetas,
        zetas=zetas,
        rmnc=bundle["rmnc"][:, 0],
        drmncds=bundle["drmncds"][:, 0],
        zmns=bundle["zmns"][:, 0],
        dzmnsds=bundle["dzmnsds"][:, 0],
        numns=bundle["numns"][:, 0],
        dnumnsds=bundle["dnumnsds"][:, 0],
        bmnc=bundle["bmnc"][:, 0],
        iota_isurf=bundle["iota"][0],
        G_isurf=bundle["G"][0],
        I_isurf=bundle["I"][0],
        rmns=bundle["rmns"][:, 0],
        drmnsds=bundle["drmnsds"][:, 0],
        zmnc=bundle["zmnc"][:, 0],
        dzmncds=bundle["dzmncds"][:, 0],
        numnc=bundle["numnc"][:, 0],
        dnumncds=bundle["dnumncds"][:, 0],
        bmns=bundle["bmns"][:, 0],
    )

    np.testing.assert_allclose(actual, expected, rtol=_RTOL, atol=_ATOL)


# ----------------------------------------------------------------------
# compute_kmns (stellsym) parity
# ----------------------------------------------------------------------


@pytest.mark.parametrize("num_points", [50, 128, 256])
@pytest.mark.parametrize("num_modes,num_surf", [(8, 4), (16, 6), (24, 3)])
def test_compute_kmns_matches_cpp(
    num_modes: int, num_surf: int, num_points: int
) -> None:
    """``compute_kmns`` reproduces the C++ kernel at ``direct_kernel`` tolerance."""
    xm, xn = _make_modes(num_modes, seed=10 * num_modes + num_surf)
    thetas, zetas = _make_quad_points(num_points, seed=num_points)
    bundle = _make_half_grid_fields(
        num_modes=num_modes,
        num_surf=num_surf,
        seed=37 + num_modes,
        stellsym=True,
    )
    args = (
        bundle["rmnc"],
        bundle["drmncds"],
        bundle["zmns"],
        bundle["dzmnsds"],
        bundle["numns"],
        bundle["dnumnsds"],
        bundle["bmnc"],
        bundle["iota"],
        bundle["G"],
        bundle["I"],
        xm,
        xn,
        thetas,
        zetas,
    )
    expected = sopp.compute_kmns(*args)
    actual = np.asarray(compute_kmns(*args))

    assert actual.shape == expected.shape == (num_modes, num_surf)
    np.testing.assert_allclose(actual, expected, rtol=_RTOL, atol=_ATOL)
    # The DC row is identically zero in both kernels.
    np.testing.assert_array_equal(actual[0, :], np.zeros(num_surf))


# ----------------------------------------------------------------------
# compute_kmnc_kmns (non-stellsym) parity
# ----------------------------------------------------------------------


@pytest.mark.parametrize("num_points", [50, 128])
@pytest.mark.parametrize("num_modes,num_surf", [(6, 4), (12, 5)])
def test_compute_kmnc_kmns_matches_cpp(
    num_modes: int, num_surf: int, num_points: int
) -> None:
    """``compute_kmnc_kmns`` reproduces the C++ kernel at ``direct_kernel`` tolerance.

    Both the cos and sin output components are checked. The DC sin row
    is required to be exactly zero (matches the ``if (im > 0)`` guard).
    """
    xm, xn = _make_modes(num_modes, seed=20 * num_modes + num_surf)
    thetas, zetas = _make_quad_points(num_points, seed=2 * num_points + 1)
    bundle = _make_half_grid_fields(
        num_modes=num_modes,
        num_surf=num_surf,
        seed=53 + num_modes,
        stellsym=False,
    )
    args = (
        bundle["rmnc"],
        bundle["drmncds"],
        bundle["zmns"],
        bundle["dzmnsds"],
        bundle["numns"],
        bundle["dnumnsds"],
        bundle["bmnc"],
        bundle["rmns"],
        bundle["drmnsds"],
        bundle["zmnc"],
        bundle["dzmncds"],
        bundle["numnc"],
        bundle["dnumncds"],
        bundle["bmns"],
        bundle["iota"],
        bundle["G"],
        bundle["I"],
        xm,
        xn,
        thetas,
        zetas,
    )
    expected = sopp.compute_kmnc_kmns(*args)
    actual = np.asarray(compute_kmnc_kmns(*args))

    assert actual.shape == expected.shape == (2, num_modes, num_surf)
    np.testing.assert_allclose(actual, expected, rtol=_RTOL, atol=_ATOL)
    # DC sin row is identically zero.
    np.testing.assert_array_equal(actual[1, 0, :], np.zeros(num_surf))


# ----------------------------------------------------------------------
# fourier_transform_{odd,even} parity and closed-form orthogonality
# ----------------------------------------------------------------------


@pytest.mark.parametrize("num_points", [64, 128, 256])
@pytest.mark.parametrize("num_modes", [6, 12])
def test_fourier_transform_odd_matches_cpp(num_modes: int, num_points: int) -> None:
    xm, xn = _make_modes(num_modes, seed=7 * num_modes)
    thetas, zetas = _make_quad_points(num_points, seed=3 * num_points)
    rng = np.random.RandomState(11 + num_points)
    K = rng.randn(num_points)

    expected = sopp.fourier_transform_odd(K, xm, xn, thetas, zetas)
    actual = np.asarray(fourier_transform_odd(K, xm, xn, thetas, zetas))

    assert actual.shape == (num_modes,)
    np.testing.assert_allclose(actual, expected, rtol=_RTOL, atol=_ATOL)
    # The DC row is zero in both kernels (C++ skips ``im=0``).
    np.testing.assert_array_equal(actual[0], 0.0)


@pytest.mark.parametrize("num_points", [64, 128, 256])
@pytest.mark.parametrize("num_modes", [6, 12])
def test_fourier_transform_even_matches_cpp(num_modes: int, num_points: int) -> None:
    xm, xn = _make_modes(num_modes, seed=8 * num_modes)
    thetas, zetas = _make_quad_points(num_points, seed=5 * num_points)
    rng = np.random.RandomState(13 + num_points)
    K = rng.randn(num_points)

    expected = sopp.fourier_transform_even(K, xm, xn, thetas, zetas)
    actual = np.asarray(fourier_transform_even(K, xm, xn, thetas, zetas))

    assert actual.shape == (num_modes,)
    np.testing.assert_allclose(actual, expected, rtol=_RTOL, atol=_ATOL)


def _numpy_fourier_transform_odd(K, xm, xn, thetas, zetas):
    angle = thetas[:, None] * xm[None, :] - zetas[:, None] * xn[None, :]
    basis = np.sin(angle)
    numer = basis.T @ K
    denom = np.sum(basis * basis, axis=0)
    out = np.divide(numer, denom, out=np.zeros_like(numer), where=denom != 0.0)
    out[0] = 0.0
    return out


def _numpy_fourier_transform_even(K, xm, xn, thetas, zetas):
    angle = thetas[:, None] * xm[None, :] - zetas[:, None] * xn[None, :]
    basis = np.cos(angle)
    return (basis.T @ K) / np.sum(basis * basis, axis=0)


def test_fourier_transform_odd_matches_closed_form_numpy_oracle() -> None:
    xm, xn = _make_modes(10, seed=109)
    thetas, zetas = _make_quad_points(113, seed=111)
    rng = np.random.RandomState(113)
    K = rng.randn(thetas.size)

    actual = np.asarray(fourier_transform_odd(K, xm, xn, thetas, zetas))
    expected = _numpy_fourier_transform_odd(K, xm, xn, thetas, zetas)

    np.testing.assert_allclose(actual, expected, rtol=_RTOL, atol=_ATOL)


def test_fourier_transform_even_matches_closed_form_numpy_oracle() -> None:
    xm, xn = _make_modes(10, seed=115)
    thetas, zetas = _make_quad_points(117, seed=117)
    rng = np.random.RandomState(119)
    K = rng.randn(thetas.size)

    actual = np.asarray(fourier_transform_even(K, xm, xn, thetas, zetas))
    expected = _numpy_fourier_transform_even(K, xm, xn, thetas, zetas)

    np.testing.assert_allclose(actual, expected, rtol=_RTOL, atol=_ATOL)


# ----------------------------------------------------------------------
# inverse_fourier_transform parity (1D and 2D variants)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("num_points", [50, 100, 200])
@pytest.mark.parametrize("num_modes", [6, 12, 24])
def test_inverse_fourier_transform_odd_1d_matches_cpp(
    num_modes: int, num_points: int
) -> None:
    xm, xn = _make_modes(num_modes, seed=9 * num_modes)
    thetas, zetas = _make_quad_points(num_points, seed=7 * num_points)
    rng = np.random.RandomState(17 + num_points)
    kmns = rng.randn(num_modes)

    expected = np.zeros(num_points, dtype=np.float64)
    sopp.inverse_fourier_transform_odd(expected, kmns, xm, xn, thetas, zetas)

    actual = np.asarray(inverse_fourier_transform_odd(kmns, xm, xn, thetas, zetas))
    np.testing.assert_allclose(actual, expected, rtol=_RTOL, atol=_ATOL)

    # The explicit 1D variant must give the same result.
    actual_1d = np.asarray(
        inverse_fourier_transform_odd_1d(kmns, xm, xn, thetas, zetas)
    )
    np.testing.assert_allclose(actual_1d, expected, rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize("num_points", [50, 100, 200])
@pytest.mark.parametrize("num_modes", [6, 12, 24])
def test_inverse_fourier_transform_even_1d_matches_cpp(
    num_modes: int, num_points: int
) -> None:
    xm, xn = _make_modes(num_modes, seed=11 * num_modes)
    thetas, zetas = _make_quad_points(num_points, seed=13 * num_points)
    rng = np.random.RandomState(19 + num_points)
    kmnc = rng.randn(num_modes)

    expected = np.zeros(num_points, dtype=np.float64)
    sopp.inverse_fourier_transform_even(expected, kmnc, xm, xn, thetas, zetas)

    actual = np.asarray(inverse_fourier_transform_even(kmnc, xm, xn, thetas, zetas))
    np.testing.assert_allclose(actual, expected, rtol=_RTOL, atol=_ATOL)

    actual_1d = np.asarray(
        inverse_fourier_transform_even_1d(kmnc, xm, xn, thetas, zetas)
    )
    np.testing.assert_allclose(actual_1d, expected, rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize("num_points", [50, 100, 200])
@pytest.mark.parametrize("num_modes", [6, 12, 24])
def test_inverse_fourier_transform_odd_2d_matches_cpp(
    num_modes: int, num_points: int
) -> None:
    """The 2D variant uses ``kmns(im, ip)`` per-point indexing (diagonal broadcast).

    Mirrors the ``BoozerRadialInterpolant._K_impl`` calling convention.
    """
    xm, xn = _make_modes(num_modes, seed=23 * num_modes)
    thetas, zetas = _make_quad_points(num_points, seed=29 * num_points)
    rng = np.random.RandomState(31 + num_points)
    kmns = rng.randn(num_modes, num_points)

    expected = np.zeros(num_points, dtype=np.float64)
    sopp.inverse_fourier_transform_odd(expected, kmns, xm, xn, thetas, zetas)

    actual = np.asarray(inverse_fourier_transform_odd(kmns, xm, xn, thetas, zetas))
    np.testing.assert_allclose(actual, expected, rtol=_RTOL, atol=_ATOL)

    actual_2d = np.asarray(
        inverse_fourier_transform_odd_2d(kmns, xm, xn, thetas, zetas)
    )
    np.testing.assert_allclose(actual_2d, expected, rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize("num_points", [50, 100, 200])
@pytest.mark.parametrize("num_modes", [6, 12, 24])
def test_inverse_fourier_transform_even_2d_matches_cpp(
    num_modes: int, num_points: int
) -> None:
    xm, xn = _make_modes(num_modes, seed=37 * num_modes)
    thetas, zetas = _make_quad_points(num_points, seed=41 * num_points)
    rng = np.random.RandomState(43 + num_points)
    kmnc = rng.randn(num_modes, num_points)

    expected = np.zeros(num_points, dtype=np.float64)
    sopp.inverse_fourier_transform_even(expected, kmnc, xm, xn, thetas, zetas)

    actual = np.asarray(inverse_fourier_transform_even(kmnc, xm, xn, thetas, zetas))
    np.testing.assert_allclose(actual, expected, rtol=_RTOL, atol=_ATOL)

    actual_2d = np.asarray(
        inverse_fourier_transform_even_2d(kmnc, xm, xn, thetas, zetas)
    )
    np.testing.assert_allclose(actual_2d, expected, rtol=_RTOL, atol=_ATOL)


def test_inverse_fourier_transform_empty_modes_match_cpp_noop() -> None:
    num_points = 17
    xm = np.empty((0,), dtype=np.float64)
    xn = np.empty((0,), dtype=np.float64)
    thetas, zetas = _make_quad_points(num_points, seed=47)

    for coeffs, odd_variant, even_variant in (
        (
            np.empty((0,), dtype=np.float64),
            inverse_fourier_transform_odd_1d,
            inverse_fourier_transform_even_1d,
        ),
        (
            np.empty((0, num_points), dtype=np.float64),
            inverse_fourier_transform_odd_2d,
            inverse_fourier_transform_even_2d,
        ),
    ):
        expected_odd = np.zeros(num_points, dtype=np.float64)
        sopp.inverse_fourier_transform_odd(expected_odd, coeffs, xm, xn, thetas, zetas)
        actual_odd = np.asarray(
            inverse_fourier_transform_odd(coeffs, xm, xn, thetas, zetas)
        )
        np.testing.assert_array_equal(actual_odd, expected_odd)
        np.testing.assert_array_equal(
            np.asarray(odd_variant(coeffs, xm, xn, thetas, zetas)), expected_odd
        )

        expected_even = np.zeros(num_points, dtype=np.float64)
        sopp.inverse_fourier_transform_even(
            expected_even, coeffs, xm, xn, thetas, zetas
        )
        actual_even = np.asarray(
            inverse_fourier_transform_even(coeffs, xm, xn, thetas, zetas)
        )
        np.testing.assert_array_equal(actual_even, expected_even)
        np.testing.assert_array_equal(
            np.asarray(even_variant(coeffs, xm, xn, thetas, zetas)), expected_even
        )


# ----------------------------------------------------------------------
# Forward/inverse round-trip closed-form oracle
# ----------------------------------------------------------------------


def test_inverse_fourier_transform_reconstructs_pure_modes() -> None:
    """Closed-form: a pure sin mode evaluated through the inverse transform
    reproduces ``K[ip] = sin(angle[ip, target_mode])``.

    This is an empty-oracle parity check that does not depend on the C++
    binding — it tests the kernel against a closed-form analytic answer.
    """
    num_modes = 8
    num_points = 128
    xm, xn = _make_modes(num_modes, seed=101)
    thetas, zetas = _make_quad_points(num_points, seed=103)

    for target in range(1, num_modes):
        kmns_unit = np.zeros(num_modes)
        kmns_unit[target] = 1.0
        actual = np.asarray(
            inverse_fourier_transform_odd_1d(kmns_unit, xm, xn, thetas, zetas)
        )
        expected = np.sin(xm[target] * thetas - xn[target] * zetas)
        np.testing.assert_allclose(actual, expected, rtol=_RTOL, atol=_ATOL)


def test_inverse_fourier_transform_even_reconstructs_pure_cos_mode() -> None:
    """Closed-form check for the cos branch (includes the DC mode)."""
    num_modes = 8
    num_points = 128
    xm, xn = _make_modes(num_modes, seed=107)
    thetas, zetas = _make_quad_points(num_points, seed=109)

    for target in range(num_modes):
        kmnc_unit = np.zeros(num_modes)
        kmnc_unit[target] = 1.0
        actual = np.asarray(
            inverse_fourier_transform_even_1d(kmnc_unit, xm, xn, thetas, zetas)
        )
        expected = np.cos(xm[target] * thetas - xn[target] * zetas)
        np.testing.assert_allclose(actual, expected, rtol=_RTOL, atol=_ATOL)


# ----------------------------------------------------------------------
# Shape and DC handling sanity checks
# ----------------------------------------------------------------------


def test_inverse_fourier_transform_rejects_unsupported_rank() -> None:
    """``ndim != 1, 2`` is a contract violation that surfaces as ValueError."""
    xm, xn = _make_modes(4, seed=0)
    thetas, zetas = _make_quad_points(16, seed=0)
    bad = np.zeros((4, 16, 2))
    with pytest.raises(ValueError, match="ndim 1 or 2"):
        inverse_fourier_transform_odd(bad, xm, xn, thetas, zetas)
    with pytest.raises(ValueError, match="ndim 1 or 2"):
        inverse_fourier_transform_even(bad, xm, xn, thetas, zetas)


def test_compute_kmns_zero_dc_row() -> None:
    """``compute_kmns`` always returns a zero DC row, matching C++ ``im>=1``."""
    xm, xn = _make_modes(8, seed=0)
    thetas, zetas = _make_quad_points(64, seed=0)
    bundle = _make_half_grid_fields(num_modes=8, num_surf=3, seed=0, stellsym=True)
    actual = np.asarray(
        compute_kmns(
            bundle["rmnc"],
            bundle["drmncds"],
            bundle["zmns"],
            bundle["dzmnsds"],
            bundle["numns"],
            bundle["dnumnsds"],
            bundle["bmnc"],
            bundle["iota"],
            bundle["G"],
            bundle["I"],
            xm,
            xn,
            thetas,
            zetas,
        )
    )
    np.testing.assert_array_equal(actual[0, :], np.zeros(3))


def test_compute_kmnc_kmns_zero_dc_sin_row() -> None:
    """``compute_kmnc_kmns`` DC sin row is identically zero."""
    xm, xn = _make_modes(6, seed=0)
    thetas, zetas = _make_quad_points(64, seed=0)
    bundle = _make_half_grid_fields(num_modes=6, num_surf=3, seed=0, stellsym=False)
    actual = np.asarray(
        compute_kmnc_kmns(
            bundle["rmnc"],
            bundle["drmncds"],
            bundle["zmns"],
            bundle["dzmnsds"],
            bundle["numns"],
            bundle["dnumnsds"],
            bundle["bmnc"],
            bundle["rmns"],
            bundle["drmnsds"],
            bundle["zmnc"],
            bundle["dzmncds"],
            bundle["numnc"],
            bundle["dnumncds"],
            bundle["bmns"],
            bundle["iota"],
            bundle["G"],
            bundle["I"],
            xm,
            xn,
            thetas,
            zetas,
        )
    )
    np.testing.assert_array_equal(actual[1, 0, :], np.zeros(3))


def test_compute_kernels_cover_single_mode_and_empty_axes() -> None:
    """Edge shapes stay aligned with the C++ projection kernels."""
    xm, xn = _make_modes(1, seed=3241)
    thetas, zetas = _make_quad_points(19, seed=3242)
    bundle = _make_half_grid_fields(num_modes=1, num_surf=2, seed=3243, stellsym=True)
    args = (
        bundle["rmnc"],
        bundle["drmncds"],
        bundle["zmns"],
        bundle["dzmnsds"],
        bundle["numns"],
        bundle["dnumnsds"],
        bundle["bmnc"],
        bundle["iota"],
        bundle["G"],
        bundle["I"],
        xm,
        xn,
        thetas,
        zetas,
    )

    np.testing.assert_allclose(
        np.asarray(compute_kmns(*args)),
        sopp.compute_kmns(*args),
        rtol=_RTOL,
        atol=_ATOL,
    )

    empty_modes = (np.empty((0, 2), dtype=np.float64),) * 7
    empty_mode_args = (
        *empty_modes,
        bundle["iota"],
        bundle["G"],
        bundle["I"],
        np.empty((0,), dtype=np.float64),
        np.empty((0,), dtype=np.float64),
        thetas,
        zetas,
    )
    assert compute_kmns(*empty_mode_args).shape == (0, 2)

    empty_points = (np.empty((0,), dtype=np.float64),) * 2
    actual_empty_points = np.asarray(compute_kmns(*args[:-2], *empty_points))
    expected_empty_points = sopp.compute_kmns(*args[:-2], *empty_points)
    np.testing.assert_array_equal(actual_empty_points, expected_empty_points)


def test_mixed_precision_matches_float64_reference() -> None:
    """Mixed-precision calls stay finite and close to the float64 lane."""
    xm, xn = _make_modes(7, seed=3251)
    thetas, zetas = _make_quad_points(31, seed=3252)
    bundle = _make_half_grid_fields(num_modes=7, num_surf=3, seed=3253, stellsym=True)
    float64_args = (
        bundle["rmnc"],
        bundle["drmncds"],
        bundle["zmns"],
        bundle["dzmnsds"],
        bundle["numns"],
        bundle["dnumnsds"],
        bundle["bmnc"],
        bundle["iota"],
        bundle["G"],
        bundle["I"],
        xm,
        xn,
        thetas,
        zetas,
    )
    float32_args = tuple(np.asarray(arg, dtype=np.float32) for arg in float64_args)

    expected = np.asarray(compute_kmns(*float64_args), dtype=np.float64)
    actual = np.asarray(compute_kmns(*float32_args), dtype=np.float64)

    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected, rtol=2.0e-5, atol=2.0e-6)


def test_mode_table_permutation_preserves_positional_pairing() -> None:
    """Mode numbers and coefficients must be permuted together."""
    xm, xn = _make_modes(9, seed=3261)
    thetas, zetas = _make_quad_points(37, seed=3262)
    rng = np.random.RandomState(3263)
    K = rng.randn(thetas.size)
    kmns = rng.randn(xm.size)
    permutation = np.array([0, 3, 7, 2, 8, 1, 4, 6, 5])

    odd_original = np.asarray(fourier_transform_odd(K, xm, xn, thetas, zetas))
    odd_permuted = np.asarray(
        fourier_transform_odd(K, xm[permutation], xn[permutation], thetas, zetas)
    )
    np.testing.assert_allclose(
        odd_permuted,
        odd_original[permutation],
        rtol=_RTOL,
        atol=_ATOL,
    )

    inverse_original = np.asarray(
        inverse_fourier_transform_odd_1d(kmns, xm, xn, thetas, zetas)
    )
    inverse_permuted = np.asarray(
        inverse_fourier_transform_odd_1d(
            kmns[permutation], xm[permutation], xn[permutation], thetas, zetas
        )
    )
    np.testing.assert_allclose(
        inverse_permuted,
        inverse_original,
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_jitted_kernels_run_without_host_roundtrip() -> None:
    """Device-resident inputs execute under strict transfer guard."""
    xm, xn = _make_modes(5, seed=3271)
    thetas, zetas = _make_quad_points(29, seed=3272)
    bundle = _make_half_grid_fields(num_modes=5, num_surf=2, seed=3273, stellsym=True)
    arrays = tuple(
        jax.device_put(np.asarray(arg, dtype=np.float64))
        for arg in (
            bundle["rmnc"],
            bundle["drmncds"],
            bundle["zmns"],
            bundle["dzmnsds"],
            bundle["numns"],
            bundle["dnumnsds"],
            bundle["bmnc"],
            bundle["iota"],
            bundle["G"],
            bundle["I"],
            xm,
            xn,
            thetas,
            zetas,
        )
    )
    for array in arrays:
        array.block_until_ready()

    with jax.transfer_guard("disallow"):
        result = jax.jit(compute_kmns)(*arrays)
        result.block_until_ready()


@pytest.mark.skipif(
    not any(device.platform == "gpu" for device in jax.devices()),
    reason="GPU deterministic lane requires a local GPU backend.",
)
def test_gpu_deterministic_lane_repeats_same_state() -> None:
    """Same-state GPU execution is deterministic for the Boozer-radial kernel."""
    gpu = next(device for device in jax.devices() if device.platform == "gpu")
    xm, xn = _make_modes(6, seed=3281)
    thetas, zetas = _make_quad_points(41, seed=3282)
    bundle = _make_half_grid_fields(num_modes=6, num_surf=2, seed=3283, stellsym=True)
    arrays = tuple(
        jax.device_put(np.asarray(arg, dtype=np.float64), device=gpu)
        for arg in (
            bundle["rmnc"],
            bundle["drmncds"],
            bundle["zmns"],
            bundle["dzmnsds"],
            bundle["numns"],
            bundle["dnumnsds"],
            bundle["bmnc"],
            bundle["iota"],
            bundle["G"],
            bundle["I"],
            xm,
            xn,
            thetas,
            zetas,
        )
    )

    result_a = np.asarray(compute_kmns(*arrays))
    result_b = np.asarray(compute_kmns(*arrays))
    np.testing.assert_array_equal(result_b, result_a)


# ----------------------------------------------------------------------
# JIT-compatibility regression
# ----------------------------------------------------------------------


def test_kernels_are_jit_compatible() -> None:
    """All public kernels are wrapped in ``jax.jit`` and trace successfully.

    The kernels are decorated at module scope; calling them once forces a
    trace. We check that the result is a JAX device array and that
    repeated calls produce identical output (the JIT cache hit).
    """
    import jax

    xm, xn = _make_modes(4, seed=0)
    thetas, zetas = _make_quad_points(32, seed=0)
    rng = np.random.RandomState(0)
    K = rng.randn(32)
    bundle = _make_half_grid_fields(num_modes=4, num_surf=2, seed=0, stellsym=True)

    result_a = compute_kmns(
        bundle["rmnc"],
        bundle["drmncds"],
        bundle["zmns"],
        bundle["dzmnsds"],
        bundle["numns"],
        bundle["dnumnsds"],
        bundle["bmnc"],
        bundle["iota"],
        bundle["G"],
        bundle["I"],
        xm,
        xn,
        thetas,
        zetas,
    )
    assert isinstance(result_a, jax.Array)

    result_b = fourier_transform_odd(K, xm, xn, thetas, zetas)
    assert isinstance(result_b, jax.Array)

    result_c = inverse_fourier_transform_odd_1d(
        np.asarray(result_b), xm, xn, thetas, zetas
    )
    assert isinstance(result_c, jax.Array)
