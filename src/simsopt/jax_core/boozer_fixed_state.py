"""Private fixed-state Boozer radial evaluator for item 33.

This module intentionally does not expose a public ``simsopt.field`` wrapper.
It evaluates a frozen Boozer radial/Fourier payload using JAX arrays only. The
payload stores already-normalized radial polynomial profiles, so VMEC,
BOOZXFORM, SciPy spline construction, and ``BoozerRadialInterpolant`` object
mutation stay outside the compiled path.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ._math_utils import as_jax_float64 as _as_jax_float64
from .boozer_radial_interp import (
    inverse_fourier_transform_even,
    inverse_fourier_transform_odd,
)


@dataclass(frozen=True)
class PiecewisePolynomial1D:
    """Piecewise polynomial profile in SciPy ``PPoly`` coefficient order."""

    breaks: jax.Array
    coeffs: jax.Array


jax.tree_util.register_dataclass(
    PiecewisePolynomial1D,
    data_fields=["breaks", "coeffs"],
    meta_fields=[],
)


@dataclass(frozen=True)
class BoozerRadialFixedState:
    """Frozen Boozer radial payload for private fixed-state evaluation.

    Fourier coefficient profiles use normalized ``BoozerRadialInterpolant``
    convention: mode coefficients have already been divided by ``mn_factor``.
    Radial derivative profiles are stored explicitly because the CPU wrapper
    keeps derivative splines as separate state.
    """

    xm: jax.Array
    xn: jax.Array
    psip: PiecewisePolynomial1D
    G: PiecewisePolynomial1D
    I: PiecewisePolynomial1D
    iota: PiecewisePolynomial1D
    dGds: PiecewisePolynomial1D
    dIds: PiecewisePolynomial1D
    diotads: PiecewisePolynomial1D
    K_sin: PiecewisePolynomial1D
    K_cos: PiecewisePolynomial1D
    nu_sin: PiecewisePolynomial1D
    nu_cos: PiecewisePolynomial1D
    dnuds_sin: PiecewisePolynomial1D
    dnuds_cos: PiecewisePolynomial1D
    R_cos: PiecewisePolynomial1D
    R_sin: PiecewisePolynomial1D
    dRds_cos: PiecewisePolynomial1D
    dRds_sin: PiecewisePolynomial1D
    Z_sin: PiecewisePolynomial1D
    Z_cos: PiecewisePolynomial1D
    dZds_sin: PiecewisePolynomial1D
    dZds_cos: PiecewisePolynomial1D
    modB_cos: PiecewisePolynomial1D
    modB_sin: PiecewisePolynomial1D
    dmodBds_cos: PiecewisePolynomial1D
    dmodBds_sin: PiecewisePolynomial1D
    no_K: bool = False


jax.tree_util.register_dataclass(
    BoozerRadialFixedState,
    data_fields=[
        "xm",
        "xn",
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
    ],
    meta_fields=["no_K"],
)


@dataclass(frozen=True)
class BoozerRadialEvaluation:
    """Evaluated Boozer public quantities at fixed ``(s, theta, zeta)`` points."""

    K: jax.Array
    dKdtheta: jax.Array
    dKdzeta: jax.Array
    nu: jax.Array
    dnuds: jax.Array
    dnudtheta: jax.Array
    dnudzeta: jax.Array
    R: jax.Array
    dRds: jax.Array
    dRdtheta: jax.Array
    dRdzeta: jax.Array
    Z: jax.Array
    dZds: jax.Array
    dZdtheta: jax.Array
    dZdzeta: jax.Array
    modB: jax.Array
    dmodBds: jax.Array
    dmodBdtheta: jax.Array
    dmodBdzeta: jax.Array
    psip: jax.Array
    G: jax.Array
    I: jax.Array
    iota: jax.Array
    dGds: jax.Array
    dIds: jax.Array
    diotads: jax.Array


jax.tree_util.register_dataclass(
    BoozerRadialEvaluation,
    data_fields=[
        "K",
        "dKdtheta",
        "dKdzeta",
        "nu",
        "dnuds",
        "dnudtheta",
        "dnudzeta",
        "R",
        "dRds",
        "dRdtheta",
        "dRdzeta",
        "Z",
        "dZds",
        "dZdtheta",
        "dZdzeta",
        "modB",
        "dmodBds",
        "dmodBdtheta",
        "dmodBdzeta",
        "psip",
        "G",
        "I",
        "iota",
        "dGds",
        "dIds",
        "diotads",
    ],
    meta_fields=[],
)


def _poly_derivative_coeffs(coeffs: jax.Array, order: int) -> jax.Array:
    out = coeffs
    for _ in range(order):
        degree = out.shape[-2] - 1
        if degree < 1:
            return jnp.zeros_like(out[..., :1, :])
        powers = jnp.arange(degree, 0, -1, dtype=out.dtype)
        shape = (1,) * (out.ndim - 2) + (degree, 1)
        out = out[..., :-1, :] * jnp.reshape(powers, shape)
    return out


def ppoly_eval(
    poly: PiecewisePolynomial1D, s: jax.Array, *, derivative: int = 0
) -> jax.Array:
    """Evaluate a frozen piecewise polynomial at radial points ``s``."""

    breaks = _as_jax_float64(poly.breaks)
    coeffs = _poly_derivative_coeffs(_as_jax_float64(poly.coeffs), derivative)
    segment = jnp.searchsorted(breaks, s, side="right") - 1
    segment = jnp.clip(segment, 0, breaks.shape[0] - 2)
    local = s - breaks[segment]
    gathered = jnp.take(coeffs, segment, axis=-1)
    acc = gathered[..., 0, :]
    for icoeff in range(1, gathered.shape[-2]):
        acc = acc * local + gathered[..., icoeff, :]
    return acc


def _even(
    spec: BoozerRadialFixedState, coeffs: jax.Array, points: jax.Array
) -> jax.Array:
    return inverse_fourier_transform_even(
        coeffs, spec.xm, spec.xn, points[:, 1], points[:, 2]
    )


def _odd(
    spec: BoozerRadialFixedState, coeffs: jax.Array, points: jax.Array
) -> jax.Array:
    return inverse_fourier_transform_odd(
        coeffs, spec.xm, spec.xn, points[:, 1], points[:, 2]
    )


def _modes(poly: PiecewisePolynomial1D, s: jax.Array) -> jax.Array:
    return ppoly_eval(poly, s)


def _scalar(poly: PiecewisePolynomial1D, s: jax.Array) -> jax.Array:
    return jnp.ravel(ppoly_eval(poly, s))


def evaluate_boozer_radial_fixed_state(
    spec: BoozerRadialFixedState, points: jax.Array
) -> BoozerRadialEvaluation:
    """Evaluate frozen Boozer radial quantities at ``(s, theta, zeta)`` points."""

    pts = _as_jax_float64(points)
    s = pts[:, 0]
    xm = spec.xm[:, None]
    xn = spec.xn[:, None]

    K_sin = _modes(spec.K_sin, s)
    K_cos = _modes(spec.K_cos, s)
    K_value = _odd(spec, K_sin, pts) + _even(spec, K_cos, pts)
    K = jnp.where(spec.no_K, jnp.zeros_like(K_value), K_value)
    dKdtheta = jnp.where(
        spec.no_K,
        jnp.zeros_like(K_value),
        _even(spec, K_sin * xm, pts) + _odd(spec, -K_cos * xm, pts),
    )
    dKdzeta = jnp.where(
        spec.no_K,
        jnp.zeros_like(K_value),
        _even(spec, -K_sin * xn, pts) + _odd(spec, K_cos * xn, pts),
    )

    nu_sin = _modes(spec.nu_sin, s)
    nu_cos = _modes(spec.nu_cos, s)
    R_cos = _modes(spec.R_cos, s)
    R_sin = _modes(spec.R_sin, s)
    Z_sin = _modes(spec.Z_sin, s)
    Z_cos = _modes(spec.Z_cos, s)
    B_cos = _modes(spec.modB_cos, s)
    B_sin = _modes(spec.modB_sin, s)

    return BoozerRadialEvaluation(
        K=K,
        dKdtheta=dKdtheta,
        dKdzeta=dKdzeta,
        nu=_odd(spec, nu_sin, pts) + _even(spec, nu_cos, pts),
        dnuds=_odd(spec, _modes(spec.dnuds_sin, s), pts)
        + _even(spec, _modes(spec.dnuds_cos, s), pts),
        dnudtheta=_even(spec, nu_sin * xm, pts) + _odd(spec, -nu_cos * xm, pts),
        dnudzeta=_even(spec, -nu_sin * xn, pts) + _odd(spec, nu_cos * xn, pts),
        R=_even(spec, R_cos, pts) + _odd(spec, R_sin, pts),
        dRds=_even(spec, _modes(spec.dRds_cos, s), pts)
        + _odd(spec, _modes(spec.dRds_sin, s), pts),
        dRdtheta=_odd(spec, -R_cos * xm, pts) + _even(spec, R_sin * xm, pts),
        dRdzeta=_odd(spec, R_cos * xn, pts) + _even(spec, -R_sin * xn, pts),
        Z=_odd(spec, Z_sin, pts) + _even(spec, Z_cos, pts),
        dZds=_odd(spec, _modes(spec.dZds_sin, s), pts)
        + _even(spec, _modes(spec.dZds_cos, s), pts),
        dZdtheta=_even(spec, Z_sin * xm, pts) + _odd(spec, -Z_cos * xm, pts),
        dZdzeta=_even(spec, -Z_sin * xn, pts) + _odd(spec, Z_cos * xn, pts),
        modB=_even(spec, B_cos, pts) + _odd(spec, B_sin, pts),
        dmodBds=_even(spec, _modes(spec.dmodBds_cos, s), pts)
        + _odd(spec, _modes(spec.dmodBds_sin, s), pts),
        dmodBdtheta=_odd(spec, -B_cos * xm, pts) + _even(spec, B_sin * xm, pts),
        dmodBdzeta=_odd(spec, B_cos * xn, pts) + _even(spec, -B_sin * xn, pts),
        psip=_scalar(spec.psip, s),
        G=_scalar(spec.G, s),
        I=_scalar(spec.I, s),
        iota=_scalar(spec.iota, s),
        dGds=_scalar(spec.dGds, s),
        dIds=_scalar(spec.dIds, s),
        diotads=_scalar(spec.diotads, s),
    )


def boozer_radial_fixed_state_to_host(
    spec: BoozerRadialFixedState,
) -> dict[str, object]:
    """Return a host-array restart payload for the private fixed-state spec."""

    leaves, treedef = jax.tree_util.tree_flatten(spec)
    host_leaves = [jax.device_get(leaf) for leaf in leaves]
    return {"treedef": treedef, "leaves": host_leaves}


def boozer_radial_fixed_state_from_host(
    payload: dict[str, object],
) -> BoozerRadialFixedState:
    """Rebuild a private fixed-state spec from ``boozer_radial_fixed_state_to_host``."""

    return jax.tree_util.tree_unflatten(payload["treedef"], payload["leaves"])


__all__ = [
    "BoozerRadialEvaluation",
    "BoozerRadialFixedState",
    "PiecewisePolynomial1D",
    "boozer_radial_fixed_state_from_host",
    "boozer_radial_fixed_state_to_host",
    "evaluate_boozer_radial_fixed_state",
    "ppoly_eval",
]
