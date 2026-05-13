"""Item 33 private fixed-state Boozer radial evaluator tests."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import simsoptpp as sopp
from scipy.interpolate import PPoly

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.jax_core.boozer_fixed_state import (
    BoozerRadialFixedState,
    PiecewisePolynomial1D,
    boozer_radial_fixed_state_from_host,
    boozer_radial_fixed_state_to_host,
    evaluate_boozer_radial_fixed_state,
    ppoly_eval,
)


_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = float(_DIRECT_KERNEL["rtol"])
_ATOL = float(_DIRECT_KERNEL["atol"])


_PROFILE_NAMES = (
    "psip",
    "G",
    "I",
    "iota",
    "dGds",
    "dIds",
    "diotads",
    "K_sin",
    "K_cos",
    "nu_sin",
    "nu_cos",
    "dnuds_sin",
    "dnuds_cos",
    "R_cos",
    "R_sin",
    "dRds_cos",
    "dRds_sin",
    "Z_sin",
    "Z_cos",
    "dZds_sin",
    "dZds_cos",
    "modB_cos",
    "modB_sin",
    "dmodBds_cos",
    "dmodBds_sin",
)


def _make_modes(num_modes: int) -> tuple[np.ndarray, np.ndarray]:
    xm = np.array([0, 1, 2, 1, 3, 2], dtype=np.float64)[:num_modes]
    xn = np.array([0, -1, 1, 2, -2, 3], dtype=np.float64)[:num_modes]
    return xm, xn


def _profile(
    rng: np.random.Generator,
    breaks: np.ndarray,
    *,
    num_modes: int | None,
    scale: float,
) -> tuple[PiecewisePolynomial1D, np.ndarray]:
    n_segments = breaks.size - 1
    degree = 3
    if num_modes is None:
        coeffs = scale * rng.standard_normal(size=(degree + 1, n_segments))
    else:
        coeffs = scale * rng.standard_normal(size=(num_modes, degree + 1, n_segments))
    return (
        PiecewisePolynomial1D(
            breaks=jnp.asarray(breaks, dtype=jnp.float64),
            coeffs=jnp.asarray(coeffs, dtype=jnp.float64),
        ),
        coeffs,
    )


def _zero_profile(
    breaks: np.ndarray, *, num_modes: int
) -> tuple[PiecewisePolynomial1D, np.ndarray]:
    coeffs = np.zeros((num_modes, 4, breaks.size - 1), dtype=np.float64)
    return (
        PiecewisePolynomial1D(
            breaks=jnp.asarray(breaks, dtype=jnp.float64),
            coeffs=jnp.asarray(coeffs, dtype=jnp.float64),
        ),
        coeffs,
    )


def _make_spec(*, stellsym: bool, no_K: bool = False):
    rng = np.random.default_rng(3300 + int(stellsym) + 10 * int(no_K))
    breaks = np.array([0.0, 0.3, 0.65, 1.0], dtype=np.float64)
    num_modes = 6
    xm, xn = _make_modes(num_modes)
    profiles: dict[str, PiecewisePolynomial1D] = {}
    coeffs: dict[str, np.ndarray] = {}

    for name in _PROFILE_NAMES:
        is_mode = name not in {"psip", "G", "I", "iota", "dGds", "dIds", "diotads"}
        asymmetric = (
            name.endswith("_cos")
            and name not in {"R_cos", "modB_cos", "dRds_cos", "dmodBds_cos"}
        ) or name in {"R_sin", "dRds_sin", "modB_sin", "dmodBds_sin"}
        if stellsym and asymmetric:
            profiles[name], coeffs[name] = _zero_profile(breaks, num_modes=num_modes)
        else:
            profiles[name], coeffs[name] = _profile(
                rng,
                breaks,
                num_modes=num_modes if is_mode else None,
                scale=0.05 if is_mode else 0.2,
            )

    coeffs["modB_cos"][0, -1, :] += 1.5
    coeffs["R_cos"][0, -1, :] += 2.0
    profiles["modB_cos"] = PiecewisePolynomial1D(
        breaks=jnp.asarray(breaks),
        coeffs=jnp.asarray(coeffs["modB_cos"]),
    )
    profiles["R_cos"] = PiecewisePolynomial1D(
        breaks=jnp.asarray(breaks),
        coeffs=jnp.asarray(coeffs["R_cos"]),
    )

    spec = BoozerRadialFixedState(
        xm=jnp.asarray(xm, dtype=jnp.float64),
        xn=jnp.asarray(xn, dtype=jnp.float64),
        no_K=no_K,
        **profiles,
    )
    return spec, coeffs, breaks, xm, xn


def _eval_profile(coeffs: np.ndarray, breaks: np.ndarray, s: np.ndarray) -> np.ndarray:
    if coeffs.ndim == 2:
        return PPoly(coeffs, breaks)(s)
    values = PPoly(np.transpose(coeffs, (1, 2, 0)), breaks)(s)
    return np.asarray(values).T


def _odd(coeffs: np.ndarray, xm: np.ndarray, xn: np.ndarray, points: np.ndarray):
    out = np.zeros(points.shape[0], dtype=np.float64)
    sopp.inverse_fourier_transform_odd(out, coeffs, xm, xn, points[:, 1], points[:, 2])
    return out


def _even(coeffs: np.ndarray, xm: np.ndarray, xn: np.ndarray, points: np.ndarray):
    out = np.zeros(points.shape[0], dtype=np.float64)
    sopp.inverse_fourier_transform_even(out, coeffs, xm, xn, points[:, 1], points[:, 2])
    return out


def _oracle(coeffs, breaks, xm, xn, points, *, no_K: bool):
    s = points[:, 0]
    mode = {name: _eval_profile(values, breaks, s) for name, values in coeffs.items()}
    xm_col = xm[:, None]
    xn_col = xn[:, None]
    K_value = _odd(mode["K_sin"], xm, xn, points) + _even(mode["K_cos"], xm, xn, points)
    return {
        "K": np.zeros_like(K_value) if no_K else K_value,
        "dKdtheta": np.zeros_like(K_value)
        if no_K
        else _even(mode["K_sin"] * xm_col, xm, xn, points)
        + _odd(-mode["K_cos"] * xm_col, xm, xn, points),
        "dKdzeta": np.zeros_like(K_value)
        if no_K
        else _even(-mode["K_sin"] * xn_col, xm, xn, points)
        + _odd(mode["K_cos"] * xn_col, xm, xn, points),
        "nu": _odd(mode["nu_sin"], xm, xn, points)
        + _even(mode["nu_cos"], xm, xn, points),
        "dnuds": _odd(mode["dnuds_sin"], xm, xn, points)
        + _even(mode["dnuds_cos"], xm, xn, points),
        "dnudtheta": _even(mode["nu_sin"] * xm_col, xm, xn, points)
        + _odd(-mode["nu_cos"] * xm_col, xm, xn, points),
        "dnudzeta": _even(-mode["nu_sin"] * xn_col, xm, xn, points)
        + _odd(mode["nu_cos"] * xn_col, xm, xn, points),
        "R": _even(mode["R_cos"], xm, xn, points) + _odd(mode["R_sin"], xm, xn, points),
        "dRds": _even(mode["dRds_cos"], xm, xn, points)
        + _odd(mode["dRds_sin"], xm, xn, points),
        "dRdtheta": _odd(-mode["R_cos"] * xm_col, xm, xn, points)
        + _even(mode["R_sin"] * xm_col, xm, xn, points),
        "dRdzeta": _odd(mode["R_cos"] * xn_col, xm, xn, points)
        + _even(-mode["R_sin"] * xn_col, xm, xn, points),
        "Z": _odd(mode["Z_sin"], xm, xn, points) + _even(mode["Z_cos"], xm, xn, points),
        "dZds": _odd(mode["dZds_sin"], xm, xn, points)
        + _even(mode["dZds_cos"], xm, xn, points),
        "dZdtheta": _even(mode["Z_sin"] * xm_col, xm, xn, points)
        + _odd(-mode["Z_cos"] * xm_col, xm, xn, points),
        "dZdzeta": _even(-mode["Z_sin"] * xn_col, xm, xn, points)
        + _odd(mode["Z_cos"] * xn_col, xm, xn, points),
        "modB": _even(mode["modB_cos"], xm, xn, points)
        + _odd(mode["modB_sin"], xm, xn, points),
        "dmodBds": _even(mode["dmodBds_cos"], xm, xn, points)
        + _odd(mode["dmodBds_sin"], xm, xn, points),
        "dmodBdtheta": _odd(-mode["modB_cos"] * xm_col, xm, xn, points)
        + _even(mode["modB_sin"] * xm_col, xm, xn, points),
        "dmodBdzeta": _odd(mode["modB_cos"] * xn_col, xm, xn, points)
        + _even(-mode["modB_sin"] * xn_col, xm, xn, points),
        "psip": _eval_profile(coeffs["psip"], breaks, s),
        "G": _eval_profile(coeffs["G"], breaks, s),
        "I": _eval_profile(coeffs["I"], breaks, s),
        "iota": _eval_profile(coeffs["iota"], breaks, s),
        "dGds": _eval_profile(coeffs["dGds"], breaks, s),
        "dIds": _eval_profile(coeffs["dIds"], breaks, s),
        "diotads": _eval_profile(coeffs["diotads"], breaks, s),
    }


def _points() -> np.ndarray:
    s = np.array([0.05, 0.22, 0.49, 0.72, 0.96], dtype=np.float64)
    theta = np.array([0.1, 1.2, 2.4, 3.3, 5.1], dtype=np.float64)
    zeta = np.array([0.2, 0.7, 1.5, 2.8, 5.5], dtype=np.float64)
    return np.stack([s, theta, zeta], axis=1)


def test_ppoly_eval_matches_scipy_ppoly_values_and_derivative():
    rng = np.random.default_rng(3390)
    breaks = np.array([0.0, 0.25, 0.7, 1.0], dtype=np.float64)
    poly, coeffs = _profile(rng, breaks, num_modes=4, scale=0.2)
    s = np.array([0.1, 0.32, 0.9], dtype=np.float64)
    scipy_poly = PPoly(np.transpose(coeffs, (1, 2, 0)), breaks)

    np.testing.assert_allclose(
        np.asarray(ppoly_eval(poly, jnp.asarray(s))).T,
        scipy_poly(s),
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(ppoly_eval(poly, jnp.asarray(s), derivative=1)).T,
        scipy_poly.derivative()(s),
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_fixed_state_evaluator_matches_cpu_oracle_for_stellsym_and_nonstellsym():
    for stellsym in (True, False):
        spec, coeffs, breaks, xm, xn = _make_spec(stellsym=stellsym)
        points = _points()
        actual = evaluate_boozer_radial_fixed_state(spec, jnp.asarray(points))
        expected = _oracle(coeffs, breaks, xm, xn, points, no_K=False)
        for name, expected_value in expected.items():
            np.testing.assert_allclose(
                np.asarray(getattr(actual, name)),
                expected_value,
                rtol=_RTOL,
                atol=_ATOL,
                err_msg=name,
            )


def test_fixed_state_evaluator_honors_no_k_without_affecting_other_quantities():
    spec, coeffs, breaks, xm, xn = _make_spec(stellsym=False, no_K=True)
    points = _points()
    actual = evaluate_boozer_radial_fixed_state(spec, jnp.asarray(points))
    expected = _oracle(coeffs, breaks, xm, xn, points, no_K=True)

    np.testing.assert_array_equal(np.asarray(actual.K), np.zeros(points.shape[0]))
    np.testing.assert_array_equal(
        np.asarray(actual.dKdtheta), np.zeros(points.shape[0])
    )
    np.testing.assert_array_equal(np.asarray(actual.dKdzeta), np.zeros(points.shape[0]))
    np.testing.assert_allclose(
        np.asarray(actual.modB), expected["modB"], rtol=_RTOL, atol=_ATOL
    )


def test_fixed_state_evaluator_jits_under_strict_transfer_guard():
    spec, _, _, _, _ = _make_spec(stellsym=False)
    points = jax.device_put(jnp.asarray(_points()))

    @jax.jit
    def _run(spec_data: BoozerRadialFixedState, point_data: jax.Array):
        return evaluate_boozer_radial_fixed_state(spec_data, point_data).modB

    _run(spec, points).block_until_ready()
    with jax.transfer_guard("disallow"):
        out = _run(spec, points)
        out.block_until_ready()

    assert out.shape == (points.shape[0],)
    assert np.all(np.isfinite(np.asarray(out)))


def test_fixed_state_restart_payload_round_trips():
    spec, _, _, _, _ = _make_spec(stellsym=True)
    points = jnp.asarray(_points())
    payload = boozer_radial_fixed_state_to_host(spec)
    restored = boozer_radial_fixed_state_from_host(payload)

    actual = evaluate_boozer_radial_fixed_state(restored, points)
    expected = evaluate_boozer_radial_fixed_state(spec, points)
    np.testing.assert_allclose(
        np.asarray(actual.R),
        np.asarray(expected.R),
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(actual.modB),
        np.asarray(expected.modB),
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_public_boozer_jax_wrapper_module_exists_but_is_opt_in():
    # Item 33 ships ``simsopt.field.boozermagneticfield_jax`` as an
    # explicit-import module. The class is deliberately NOT registered
    # in the lazy-export map for ``simsopt.field``; users must import
    # the module path directly.
    import simsopt.field as field_pkg

    repo_root = Path(__file__).resolve().parents[2]
    assert (
        repo_root / "src" / "simsopt" / "field" / "boozermagneticfield_jax.py"
    ).exists()
    assert "BoozerRadialInterpolantJAX" not in getattr(field_pkg, "__all__", ())
