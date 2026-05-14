"""JAX-backed wrapper that freezes an upstream ``BoozerRadialInterpolant``.

``BoozerRadialInterpolantJAX`` mirrors the public ``BoozerMagneticField``
surface (``set_points``, ``modB``, ``K``, ``nu``, ``R``, ``Z``, ``iota``,
``G``, ``I``, ``psip`` and first-derivative bundles) while routing the
field-evaluation hot path through immutable JAX state captured at
construction time from an existing CPU ``BoozerRadialInterpolant``.

Architectural notes (item 33):

- This wrapper does **not** inherit from ``sopp.BoozerMagneticField``.
- It does **not** rewrite the upstream class. Construction reads the
  already-built ``InterpolatedUnivariateSpline`` objects and translates
  them into ``scipy.interpolate.PPoly`` coefficients so the JAX
  evaluator can compute spline values without leaving the compiled
  path.
- Frozen state semantics: mutating the wrapped CPU instance after
  construction does not propagate to the JAX wrapper. The wrapper
  exposes a fresh ``Optimizable`` node with no DOFs of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import PPoly

from .._core.optimizable import Optimizable
from .._core.json import GSONDecoder
from ..jax_core._math_utils import as_jax_float64 as _as_jax_float64
from ..jax_core.boozer_analytic import (
    BoozerAnalyticFrozenState,
    _eval_G as _eval_analytic_G,
    _eval_I as _eval_analytic_I,
    _eval_K as _eval_analytic_K,
    _eval_dGds as _eval_analytic_dGds,
    _eval_dIds as _eval_analytic_dIds,
    _eval_dKdtheta as _eval_analytic_dKdtheta,
    _eval_dKdzeta as _eval_analytic_dKdzeta,
    _eval_diotads as _eval_analytic_diotads,
    _eval_dmodBds as _eval_analytic_dmodBds,
    _eval_dmodBdtheta as _eval_analytic_dmodBdtheta,
    _eval_dmodBdzeta as _eval_analytic_dmodBdzeta,
    _eval_iota as _eval_analytic_iota,
    _eval_modB as _eval_analytic_modB,
    _eval_psip as _eval_analytic_psip,
    freeze_boozer_analytic_state,
)
from ..jax_core.boozer_fixed_state import (
    PiecewisePolynomial1D,
    ppoly_eval,
)
from ..jax_core.boozer_radial_interp import (
    inverse_fourier_transform_even,
    inverse_fourier_transform_odd,
)
from ..jax_core.interpolated_boozer_field import (
    FLUX_FUNCTION_SCALARS,
    InterpolatedBoozerFieldFrozenState,
    SYMMETRY_EXPLOIT_SCALARS,
    build_spec_for_scalar as _interp_build_spec_for_scalar,
    evaluate_scalar as _eval_interp_scalar,
    freeze_interpolated_boozer_field_state,
)
from ..jax_core.regular_grid_interp import (
    UniformInterpolationRule as _jax_core_uniform_rule,
)

__all__ = [
    "BoozerAnalyticFrozenState",
    "BoozerAnalyticJAX",
    "BoozerRadialInterpolantFrozenState",
    "BoozerRadialInterpolantJAX",
    "InterpolatedBoozerFieldFrozenState",
    "InterpolatedBoozerFieldJAX",
    "freeze_boozer_analytic_state",
    "freeze_boozer_radial_state",
    "freeze_interpolated_boozer_field_state",
]


# ----------------------------------------------------------------------
# Frozen state pytree
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BoozerRadialInterpolantFrozenState:
    """Immutable PPoly payload captured from a CPU ``BoozerRadialInterpolant``.

    Per-mode profiles store the spline coefficients **as built by the
    upstream class**, i.e. with ``mn_factor`` multiplied in. Evaluation
    divides by ``mn_factor`` at the sample location to recover the
    physical Fourier coefficient (matches ``_*_impl`` semantics in
    ``simsopt.field.boozermagneticfield``).

    Scalar profiles (``psip``, ``G``, ``I``, ``iota``, ``dGds``, ``dIds``,
    ``diotads``) are stored as ``(degree+1, n_segments)`` PPoly coeffs.
    Mode-tabled profiles are stored as ``(num_modes, degree+1, n_segments)``.

    ``no_K`` is a meta-field — the K evaluators return zeros when True.
    ``stellsym`` is also a meta-field — non-stellsym profiles are stored
    but evaluated only when ``stellsym`` is False.
    """

    xm: jax.Array
    xn: jax.Array

    # Scalar radial profiles.
    psip: PiecewisePolynomial1D
    G: PiecewisePolynomial1D
    I: PiecewisePolynomial1D
    iota: PiecewisePolynomial1D
    dGds: PiecewisePolynomial1D
    dIds: PiecewisePolynomial1D
    diotads: PiecewisePolynomial1D

    # Per-mode spline-baked profiles (numerator: spline = mn_factor * physical_coeff).
    bmnc: PiecewisePolynomial1D
    dbmncds: PiecewisePolynomial1D
    rmnc: PiecewisePolynomial1D
    drmncds: PiecewisePolynomial1D
    zmns: PiecewisePolynomial1D
    dzmnsds: PiecewisePolynomial1D
    numns: PiecewisePolynomial1D
    dnumnsds: PiecewisePolynomial1D

    # Asym (non-stellsym) per-mode profiles. Always present; populated
    # with zeros for stellsym fields so the pytree shape is stable.
    bmns: PiecewisePolynomial1D
    dbmnsds: PiecewisePolynomial1D
    rmns: PiecewisePolynomial1D
    drmnsds: PiecewisePolynomial1D
    zmnc: PiecewisePolynomial1D
    dzmncds: PiecewisePolynomial1D
    numnc: PiecewisePolynomial1D
    dnumncds: PiecewisePolynomial1D

    # Per-mode normalization factor profiles.
    mn_factor: PiecewisePolynomial1D
    d_mn_factor: PiecewisePolynomial1D

    # K Fourier coefficients (scaled by mn_factor). Always present;
    # populated with zeros when ``no_K`` is True so the pytree shape is
    # stable.
    kmns: PiecewisePolynomial1D
    kmnc: PiecewisePolynomial1D

    # Meta-fields.
    stellsym: bool = True
    no_K: bool = False


jax.tree_util.register_dataclass(
    BoozerRadialInterpolantFrozenState,
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
        "bmnc",
        "dbmncds",
        "rmnc",
        "drmncds",
        "zmns",
        "dzmnsds",
        "numns",
        "dnumnsds",
        "bmns",
        "dbmnsds",
        "rmns",
        "drmnsds",
        "zmnc",
        "dzmncds",
        "numnc",
        "dnumncds",
        "mn_factor",
        "d_mn_factor",
        "kmns",
        "kmnc",
    ],
    meta_fields=["stellsym", "no_K"],
)

_FROZEN_ARRAY_FIELDS = ("xm", "xn")
_FROZEN_META_FIELDS = ("stellsym", "no_K")
_FROZEN_PROFILE_FIELDS = tuple(
    field.name
    for field in fields(BoozerRadialInterpolantFrozenState)
    if field.name not in (*_FROZEN_ARRAY_FIELDS, *_FROZEN_META_FIELDS)
)


def _profile_to_host(profile: PiecewisePolynomial1D) -> dict[str, np.ndarray]:
    return {
        "breaks": np.asarray(profile.breaks, dtype=np.float64),
        "coeffs": np.asarray(profile.coeffs, dtype=np.float64),
    }


def _profile_from_host(payload: dict) -> PiecewisePolynomial1D:
    return PiecewisePolynomial1D(
        breaks=jnp.asarray(payload["breaks"], dtype=jnp.float64),
        coeffs=jnp.asarray(payload["coeffs"], dtype=jnp.float64),
    )


def _frozen_state_to_host(state: BoozerRadialInterpolantFrozenState) -> dict:
    payload = {
        "xm": np.asarray(state.xm, dtype=np.float64),
        "xn": np.asarray(state.xn, dtype=np.float64),
        "stellsym": bool(state.stellsym),
        "no_K": bool(state.no_K),
    }
    for name in _FROZEN_PROFILE_FIELDS:
        payload[name] = _profile_to_host(getattr(state, name))
    return payload


def _frozen_state_from_host(payload: dict) -> BoozerRadialInterpolantFrozenState:
    profiles = {
        name: _profile_from_host(payload[name]) for name in _FROZEN_PROFILE_FIELDS
    }
    return BoozerRadialInterpolantFrozenState(
        xm=jnp.asarray(payload["xm"], dtype=jnp.float64),
        xn=jnp.asarray(payload["xn"], dtype=jnp.float64),
        stellsym=bool(payload["stellsym"]),
        no_K=bool(payload["no_K"]),
        **profiles,
    )


# ----------------------------------------------------------------------
# Freeze helper (CPU → JAX)
# ----------------------------------------------------------------------


def _ppoly_from_spline(spline) -> PPoly:
    """Convert an ``InterpolatedUnivariateSpline`` to ``scipy.PPoly``."""
    return PPoly.from_spline(spline._eval_args)


def _scalar_profile(spline) -> PiecewisePolynomial1D:
    pp = _ppoly_from_spline(spline)
    return PiecewisePolynomial1D(
        breaks=jnp.asarray(np.asarray(pp.x), dtype=jnp.float64),
        coeffs=jnp.asarray(np.asarray(pp.c), dtype=jnp.float64),
    )


def _mode_profile_stack(splines) -> PiecewisePolynomial1D:
    """Stack a list of ``InterpolatedUnivariateSpline`` into a mode-tabled PPoly.

    All splines must share identical breakpoints, which is the case for
    every spline built inside ``BoozerRadialInterpolant.init_splines`` /
    ``BoozerRadialInterpolant.compute_K``.
    """
    pps = [_ppoly_from_spline(spline) for spline in splines]
    ref_x = pps[0].x
    for pp in pps[1:]:
        if not np.allclose(pp.x, ref_x):
            raise ValueError(
                "Per-mode splines have mismatched breakpoints; cannot stack."
            )
    stacked = np.stack([pp.c for pp in pps], axis=0)
    return PiecewisePolynomial1D(
        breaks=jnp.asarray(ref_x, dtype=jnp.float64),
        coeffs=jnp.asarray(stacked, dtype=jnp.float64),
    )


def _zeros_like_profile(reference: PiecewisePolynomial1D) -> PiecewisePolynomial1D:
    return PiecewisePolynomial1D(
        breaks=reference.breaks,
        coeffs=jnp.zeros_like(reference.coeffs),
    )


def freeze_boozer_radial_state(upstream) -> BoozerRadialInterpolantFrozenState:
    """Extract a JAX-friendly frozen state from a CPU ``BoozerRadialInterpolant``.

    The wrapped instance must already have its splines built — i.e. it
    must have run ``init_splines`` (and ``compute_K`` unless
    ``no_K`` is True). This is the case for any production
    ``BoozerRadialInterpolant`` instance after ``__init__``.
    """

    stellsym = bool(upstream.stellsym)
    no_K = bool(upstream.no_K)

    xm = jnp.asarray(np.asarray(upstream.xm_b, dtype=np.float64), dtype=jnp.float64)
    xn = jnp.asarray(np.asarray(upstream.xn_b, dtype=np.float64), dtype=jnp.float64)

    bmnc = _mode_profile_stack(upstream.bmnc_splines)
    dbmncds = _mode_profile_stack(upstream.dbmncds_splines)
    rmnc = _mode_profile_stack(upstream.rmnc_splines)
    drmncds = _mode_profile_stack(upstream.drmncds_splines)
    zmns = _mode_profile_stack(upstream.zmns_splines)
    dzmnsds = _mode_profile_stack(upstream.dzmnsds_splines)
    numns = _mode_profile_stack(upstream.numns_splines)
    dnumnsds = _mode_profile_stack(upstream.dnumnsds_splines)
    mn_factor = _mode_profile_stack(upstream.mn_factor_splines)
    d_mn_factor = _mode_profile_stack(upstream.d_mn_factor_splines)

    if stellsym:
        bmns = _zeros_like_profile(bmnc)
        dbmnsds = _zeros_like_profile(dbmncds)
        rmns = _zeros_like_profile(rmnc)
        drmnsds = _zeros_like_profile(drmncds)
        zmnc = _zeros_like_profile(zmns)
        dzmncds = _zeros_like_profile(dzmnsds)
        numnc = _zeros_like_profile(numns)
        dnumncds = _zeros_like_profile(dnumnsds)
    else:
        bmns = _mode_profile_stack(upstream.bmns_splines)
        dbmnsds = _mode_profile_stack(upstream.dbmnsds_splines)
        rmns = _mode_profile_stack(upstream.rmns_splines)
        drmnsds = _mode_profile_stack(upstream.drmnsds_splines)
        zmnc = _mode_profile_stack(upstream.zmnc_splines)
        dzmncds = _mode_profile_stack(upstream.dzmncds_splines)
        numnc = _mode_profile_stack(upstream.numnc_splines)
        dnumncds = _mode_profile_stack(upstream.dnumncds_splines)

    if no_K:
        kmns = _zeros_like_profile(bmnc)
        kmnc = _zeros_like_profile(bmnc)
    else:
        kmns = _mode_profile_stack(upstream.kmns_splines)
        if stellsym:
            kmnc = _zeros_like_profile(kmns)
        else:
            kmnc = _mode_profile_stack(upstream.kmnc_splines)

    return BoozerRadialInterpolantFrozenState(
        xm=xm,
        xn=xn,
        psip=_scalar_profile(upstream.psip_spline),
        G=_scalar_profile(upstream.G_spline),
        I=_scalar_profile(upstream.I_spline),
        iota=_scalar_profile(upstream.iota_spline),
        dGds=_scalar_profile(upstream.dGds_spline),
        dIds=_scalar_profile(upstream.dIds_spline),
        diotads=_scalar_profile(upstream.diotads_spline),
        bmnc=bmnc,
        dbmncds=dbmncds,
        rmnc=rmnc,
        drmncds=drmncds,
        zmns=zmns,
        dzmnsds=dzmnsds,
        numns=numns,
        dnumnsds=dnumnsds,
        bmns=bmns,
        dbmnsds=dbmnsds,
        rmns=rmns,
        drmnsds=drmnsds,
        zmnc=zmnc,
        dzmncds=dzmncds,
        numnc=numnc,
        dnumncds=dnumncds,
        mn_factor=mn_factor,
        d_mn_factor=d_mn_factor,
        kmns=kmns,
        kmnc=kmnc,
        stellsym=stellsym,
        no_K=no_K,
    )


# ----------------------------------------------------------------------
# JAX evaluators (pure functions on the frozen state)
# ----------------------------------------------------------------------


def _column_at(s: jax.Array, profile: PiecewisePolynomial1D) -> jax.Array:
    """Evaluate a mode-tabled PPoly at points ``s``.

    Returns shape ``(num_modes, num_points)``.
    """
    return ppoly_eval(profile, s)


def _scalar_at(s: jax.Array, profile: PiecewisePolynomial1D) -> jax.Array:
    """Evaluate a scalar PPoly profile at ``s`` and ravel to ``(num_points,)``."""
    return jnp.ravel(ppoly_eval(profile, s))


def _normalize(values: jax.Array, mn_factor: jax.Array) -> jax.Array:
    """Divide a mode-tabled spline column-stack by per-(mode, point) mn_factor."""
    return values / mn_factor


def _radial_normalized(
    spline_vals: jax.Array,
    dspline_vals: jax.Array,
    mn_factor: jax.Array,
    d_mn_factor: jax.Array,
) -> jax.Array:
    """Apply the radial-derivative quotient used by ``_*ds_impl`` methods."""
    return (dspline_vals - spline_vals * d_mn_factor / mn_factor) / mn_factor


def _eval_modB(
    state: BoozerRadialInterpolantFrozenState, points: jax.Array
) -> jax.Array:
    s = points[:, 0]
    thetas = points[:, 1]
    zetas = points[:, 2]
    mn = _column_at(s, state.mn_factor)
    bmnc = _normalize(_column_at(s, state.bmnc), mn)
    result = inverse_fourier_transform_even(bmnc, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        bmns = _normalize(_column_at(s, state.bmns), mn)
        result = result + inverse_fourier_transform_odd(
            bmns, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_dmodBdtheta(
    state: BoozerRadialInterpolantFrozenState, points: jax.Array
) -> jax.Array:
    s = points[:, 0]
    thetas = points[:, 1]
    zetas = points[:, 2]
    mn = _column_at(s, state.mn_factor)
    xm_col = state.xm[:, None]
    bmnc = -xm_col * _normalize(_column_at(s, state.bmnc), mn)
    result = inverse_fourier_transform_odd(bmnc, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        bmns = xm_col * _normalize(_column_at(s, state.bmns), mn)
        result = result + inverse_fourier_transform_even(
            bmns, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_dmodBdzeta(
    state: BoozerRadialInterpolantFrozenState, points: jax.Array
) -> jax.Array:
    s = points[:, 0]
    thetas = points[:, 1]
    zetas = points[:, 2]
    mn = _column_at(s, state.mn_factor)
    xn_col = state.xn[:, None]
    bmnc = xn_col * _normalize(_column_at(s, state.bmnc), mn)
    result = inverse_fourier_transform_odd(bmnc, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        bmns = -xn_col * _normalize(_column_at(s, state.bmns), mn)
        result = result + inverse_fourier_transform_even(
            bmns, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_dmodBds(
    state: BoozerRadialInterpolantFrozenState, points: jax.Array
) -> jax.Array:
    s = points[:, 0]
    thetas = points[:, 1]
    zetas = points[:, 2]
    mn = _column_at(s, state.mn_factor)
    d_mn = _column_at(s, state.d_mn_factor)
    bmnc = _radial_normalized(
        _column_at(s, state.bmnc), _column_at(s, state.dbmncds), mn, d_mn
    )
    result = inverse_fourier_transform_even(bmnc, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        bmns = _radial_normalized(
            _column_at(s, state.bmns), _column_at(s, state.dbmnsds), mn, d_mn
        )
        result = result + inverse_fourier_transform_odd(
            bmns, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_R(state: BoozerRadialInterpolantFrozenState, points: jax.Array) -> jax.Array:
    s = points[:, 0]
    thetas = points[:, 1]
    zetas = points[:, 2]
    mn = _column_at(s, state.mn_factor)
    rmnc = _normalize(_column_at(s, state.rmnc), mn)
    result = inverse_fourier_transform_even(rmnc, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        rmns = _normalize(_column_at(s, state.rmns), mn)
        result = result + inverse_fourier_transform_odd(
            rmns, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_dRdtheta(
    state: BoozerRadialInterpolantFrozenState, points: jax.Array
) -> jax.Array:
    s = points[:, 0]
    thetas = points[:, 1]
    zetas = points[:, 2]
    mn = _column_at(s, state.mn_factor)
    xm_col = state.xm[:, None]
    rmnc = -xm_col * _normalize(_column_at(s, state.rmnc), mn)
    result = inverse_fourier_transform_odd(rmnc, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        rmns = xm_col * _normalize(_column_at(s, state.rmns), mn)
        result = result + inverse_fourier_transform_even(
            rmns, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_dRdzeta(
    state: BoozerRadialInterpolantFrozenState, points: jax.Array
) -> jax.Array:
    s = points[:, 0]
    thetas = points[:, 1]
    zetas = points[:, 2]
    mn = _column_at(s, state.mn_factor)
    xn_col = state.xn[:, None]
    rmnc = xn_col * _normalize(_column_at(s, state.rmnc), mn)
    result = inverse_fourier_transform_odd(rmnc, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        rmns = -xn_col * _normalize(_column_at(s, state.rmns), mn)
        result = result + inverse_fourier_transform_even(
            rmns, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_dRds(
    state: BoozerRadialInterpolantFrozenState, points: jax.Array
) -> jax.Array:
    s = points[:, 0]
    thetas = points[:, 1]
    zetas = points[:, 2]
    mn = _column_at(s, state.mn_factor)
    d_mn = _column_at(s, state.d_mn_factor)
    rmnc = _radial_normalized(
        _column_at(s, state.rmnc), _column_at(s, state.drmncds), mn, d_mn
    )
    result = inverse_fourier_transform_even(rmnc, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        rmns = _radial_normalized(
            _column_at(s, state.rmns), _column_at(s, state.drmnsds), mn, d_mn
        )
        result = result + inverse_fourier_transform_odd(
            rmns, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_Z(state: BoozerRadialInterpolantFrozenState, points: jax.Array) -> jax.Array:
    s = points[:, 0]
    thetas = points[:, 1]
    zetas = points[:, 2]
    mn = _column_at(s, state.mn_factor)
    zmns = _normalize(_column_at(s, state.zmns), mn)
    result = inverse_fourier_transform_odd(zmns, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        zmnc = _normalize(_column_at(s, state.zmnc), mn)
        result = result + inverse_fourier_transform_even(
            zmnc, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_dZdtheta(
    state: BoozerRadialInterpolantFrozenState, points: jax.Array
) -> jax.Array:
    s = points[:, 0]
    thetas = points[:, 1]
    zetas = points[:, 2]
    mn = _column_at(s, state.mn_factor)
    xm_col = state.xm[:, None]
    zmns = xm_col * _normalize(_column_at(s, state.zmns), mn)
    result = inverse_fourier_transform_even(zmns, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        zmnc = -xm_col * _normalize(_column_at(s, state.zmnc), mn)
        result = result + inverse_fourier_transform_odd(
            zmnc, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_dZdzeta(
    state: BoozerRadialInterpolantFrozenState, points: jax.Array
) -> jax.Array:
    s = points[:, 0]
    thetas = points[:, 1]
    zetas = points[:, 2]
    mn = _column_at(s, state.mn_factor)
    xn_col = state.xn[:, None]
    zmns = -xn_col * _normalize(_column_at(s, state.zmns), mn)
    result = inverse_fourier_transform_even(zmns, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        zmnc = xn_col * _normalize(_column_at(s, state.zmnc), mn)
        result = result + inverse_fourier_transform_odd(
            zmnc, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_dZds(
    state: BoozerRadialInterpolantFrozenState, points: jax.Array
) -> jax.Array:
    s = points[:, 0]
    thetas = points[:, 1]
    zetas = points[:, 2]
    mn = _column_at(s, state.mn_factor)
    d_mn = _column_at(s, state.d_mn_factor)
    zmns = _radial_normalized(
        _column_at(s, state.zmns), _column_at(s, state.dzmnsds), mn, d_mn
    )
    result = inverse_fourier_transform_odd(zmns, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        zmnc = _radial_normalized(
            _column_at(s, state.zmnc), _column_at(s, state.dzmncds), mn, d_mn
        )
        result = result + inverse_fourier_transform_even(
            zmnc, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_nu(state: BoozerRadialInterpolantFrozenState, points: jax.Array) -> jax.Array:
    s = points[:, 0]
    thetas = points[:, 1]
    zetas = points[:, 2]
    mn = _column_at(s, state.mn_factor)
    numns = _normalize(_column_at(s, state.numns), mn)
    result = inverse_fourier_transform_odd(numns, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        numnc = _normalize(_column_at(s, state.numnc), mn)
        result = result + inverse_fourier_transform_even(
            numnc, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_dnudtheta(
    state: BoozerRadialInterpolantFrozenState, points: jax.Array
) -> jax.Array:
    s = points[:, 0]
    thetas = points[:, 1]
    zetas = points[:, 2]
    mn = _column_at(s, state.mn_factor)
    xm_col = state.xm[:, None]
    numns = xm_col * _normalize(_column_at(s, state.numns), mn)
    result = inverse_fourier_transform_even(numns, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        numnc = -xm_col * _normalize(_column_at(s, state.numnc), mn)
        result = result + inverse_fourier_transform_odd(
            numnc, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_dnudzeta(
    state: BoozerRadialInterpolantFrozenState, points: jax.Array
) -> jax.Array:
    s = points[:, 0]
    thetas = points[:, 1]
    zetas = points[:, 2]
    mn = _column_at(s, state.mn_factor)
    xn_col = state.xn[:, None]
    numns = -xn_col * _normalize(_column_at(s, state.numns), mn)
    result = inverse_fourier_transform_even(numns, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        numnc = xn_col * _normalize(_column_at(s, state.numnc), mn)
        result = result + inverse_fourier_transform_odd(
            numnc, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_dnuds(
    state: BoozerRadialInterpolantFrozenState, points: jax.Array
) -> jax.Array:
    s = points[:, 0]
    thetas = points[:, 1]
    zetas = points[:, 2]
    mn = _column_at(s, state.mn_factor)
    d_mn = _column_at(s, state.d_mn_factor)
    numns = _radial_normalized(
        _column_at(s, state.numns), _column_at(s, state.dnumnsds), mn, d_mn
    )
    result = inverse_fourier_transform_odd(numns, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        numnc = _radial_normalized(
            _column_at(s, state.numnc), _column_at(s, state.dnumncds), mn, d_mn
        )
        result = result + inverse_fourier_transform_even(
            numnc, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_K(state: BoozerRadialInterpolantFrozenState, points: jax.Array) -> jax.Array:
    if state.no_K:
        return jnp.zeros(points.shape[0], dtype=jnp.float64)
    s = points[:, 0]
    thetas = points[:, 1]
    zetas = points[:, 2]
    mn = _column_at(s, state.mn_factor)
    kmns = _normalize(_column_at(s, state.kmns), mn)
    result = inverse_fourier_transform_odd(kmns, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        kmnc = _normalize(_column_at(s, state.kmnc), mn)
        result = result + inverse_fourier_transform_even(
            kmnc, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_dKdtheta(
    state: BoozerRadialInterpolantFrozenState, points: jax.Array
) -> jax.Array:
    if state.no_K:
        return jnp.zeros(points.shape[0], dtype=jnp.float64)
    s = points[:, 0]
    thetas = points[:, 1]
    zetas = points[:, 2]
    mn = _column_at(s, state.mn_factor)
    xm_col = state.xm[:, None]
    kmns = xm_col * _normalize(_column_at(s, state.kmns), mn)
    result = inverse_fourier_transform_even(kmns, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        kmnc = -xm_col * _normalize(_column_at(s, state.kmnc), mn)
        result = result + inverse_fourier_transform_odd(
            kmnc, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_dKdzeta(
    state: BoozerRadialInterpolantFrozenState, points: jax.Array
) -> jax.Array:
    if state.no_K:
        return jnp.zeros(points.shape[0], dtype=jnp.float64)
    s = points[:, 0]
    thetas = points[:, 1]
    zetas = points[:, 2]
    mn = _column_at(s, state.mn_factor)
    xn_col = state.xn[:, None]
    kmns = -xn_col * _normalize(_column_at(s, state.kmns), mn)
    result = inverse_fourier_transform_even(kmns, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        kmnc = xn_col * _normalize(_column_at(s, state.kmnc), mn)
        result = result + inverse_fourier_transform_odd(
            kmnc, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_psip(
    state: BoozerRadialInterpolantFrozenState, points: jax.Array
) -> jax.Array:
    return _scalar_at(points[:, 0], state.psip)


def _eval_G(state: BoozerRadialInterpolantFrozenState, points: jax.Array) -> jax.Array:
    return _scalar_at(points[:, 0], state.G)


def _eval_I(state: BoozerRadialInterpolantFrozenState, points: jax.Array) -> jax.Array:
    return _scalar_at(points[:, 0], state.I)


def _eval_iota(
    state: BoozerRadialInterpolantFrozenState, points: jax.Array
) -> jax.Array:
    return _scalar_at(points[:, 0], state.iota)


def _eval_dGds(
    state: BoozerRadialInterpolantFrozenState, points: jax.Array
) -> jax.Array:
    return _scalar_at(points[:, 0], state.dGds)


def _eval_dIds(
    state: BoozerRadialInterpolantFrozenState, points: jax.Array
) -> jax.Array:
    return _scalar_at(points[:, 0], state.dIds)


def _eval_diotads(
    state: BoozerRadialInterpolantFrozenState, points: jax.Array
) -> jax.Array:
    return _scalar_at(points[:, 0], state.diotads)


# ----------------------------------------------------------------------
# Public wrapper
# ----------------------------------------------------------------------


def _as_column(values: jax.Array) -> jax.Array:
    """Match the upstream ``_*_impl`` shape convention of ``(n, 1)``."""
    return values[:, None]


class BoozerRadialInterpolantJAX(Optimizable):
    """JAX-backed wrapper that freezes an upstream ``BoozerRadialInterpolant``.

    Architectural note: this wrapper does **not** inherit from
    ``sopp.BoozerMagneticField``. It exposes the same public surface
    (``modB``, ``K``, ``nu``, ``R``, ``Z``, ``iota``, ``G``, ``I``,
    ``psip``, ``set_points``, ``get_points``, plus first-derivative
    bundles) while routing the field-evaluation hot path through
    immutable JAX state captured at construction time. State is frozen
    — modifying the wrapped CPU instance after construction does not
    propagate.

    Args:
        upstream: an instance of
            :class:`simsopt.field.boozermagneticfield.BoozerRadialInterpolant`
            with splines already built (i.e. ``init_splines`` has run).
            ``compute_K`` must also have run unless ``upstream.no_K`` is
            True.
    """

    def __init__(self, upstream):
        Optimizable.__init__(self, x0=np.asarray([]))
        self._frozen_state = freeze_boozer_radial_state(upstream)
        self._psi0 = float(upstream.psi0)
        self._nfp = int(getattr(upstream.booz.bx, "nfp", 1))
        self._points = jnp.zeros((0, 3), dtype=jnp.float64)
        self._cache: dict[str, jax.Array] = {}

    @classmethod
    def from_frozen_state(
        cls,
        frozen_state: BoozerRadialInterpolantFrozenState,
        *,
        psi0: float,
        nfp: int,
    ):
        wrapper = cls.__new__(cls)
        Optimizable.__init__(wrapper, x0=np.asarray([]))
        wrapper._frozen_state = frozen_state
        wrapper._psi0 = float(psi0)
        wrapper._nfp = int(nfp)
        wrapper._points = jnp.zeros((0, 3), dtype=jnp.float64)
        wrapper._cache = {}
        return wrapper

    # ------------------------------------------------------------------
    # Points / cache management
    # ------------------------------------------------------------------

    @property
    def psi0(self) -> float:
        return self._psi0

    @property
    def stellsym(self) -> bool:
        return bool(self._frozen_state.stellsym)

    @property
    def nfp(self) -> int:
        return self._nfp

    @property
    def no_K(self) -> bool:
        return bool(self._frozen_state.no_K)

    @property
    def frozen_state(self) -> BoozerRadialInterpolantFrozenState:
        return self._frozen_state

    def set_points(self, points):
        """Set the Boozer ``(s, theta, zeta)`` evaluation points.

        Returns ``self`` to match the upstream ``set_points`` signature.
        """
        arr = _as_jax_float64(np.asarray(points, dtype=np.float64))
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"points must have shape (n, 3); got shape={tuple(arr.shape)!r}"
            )
        self._points = arr
        self._cache.clear()
        return self

    def get_points(self) -> np.ndarray:
        return np.asarray(self._points)

    def get_points_ref(self) -> jax.Array:
        return self._points

    def clear_cached_properties(self):
        self._cache.clear()

    def as_dict(self, serial_objs_dict) -> dict:
        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "@name": getattr(self, "name", str(id(self))),
            "@version": None,
        }
        d["frozen_state"] = _frozen_state_to_host(self._frozen_state)
        d["psi0"] = self._psi0
        d["nfp"] = self._nfp
        d["points"] = self.get_points()
        return d

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        decoder = GSONDecoder()
        frozen_payload = decoder.process_decoded(
            d["frozen_state"], serial_objs_dict, recon_objs
        )
        points = decoder.process_decoded(d["points"], serial_objs_dict, recon_objs)
        wrapper = cls.from_frozen_state(
            _frozen_state_from_host(frozen_payload),
            psi0=float(d["psi0"]),
            nfp=int(d["nfp"]),
        )
        wrapper.set_points(points)
        return wrapper

    # ------------------------------------------------------------------
    # Field evaluators
    # ------------------------------------------------------------------

    def _cached(self, name: str, fn) -> jax.Array:
        cached = self._cache.get(name)
        if cached is None:
            cached = fn(self._frozen_state, self._points)
            self._cache[name] = cached
        return cached

    def modB(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("modB", _eval_modB)))

    def dmodBdtheta(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dmodBdtheta", _eval_dmodBdtheta)))

    def dmodBdzeta(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dmodBdzeta", _eval_dmodBdzeta)))

    def dmodBds(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dmodBds", _eval_dmodBds)))

    def modB_derivs(self) -> np.ndarray:
        ds = np.asarray(self._cached("dmodBds", _eval_dmodBds))
        dtheta = np.asarray(self._cached("dmodBdtheta", _eval_dmodBdtheta))
        dzeta = np.asarray(self._cached("dmodBdzeta", _eval_dmodBdzeta))
        return np.stack([ds, dtheta, dzeta], axis=1)

    def K(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("K", _eval_K)))

    def dKdtheta(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dKdtheta", _eval_dKdtheta)))

    def dKdzeta(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dKdzeta", _eval_dKdzeta)))

    def K_derivs(self) -> np.ndarray:
        dtheta = np.asarray(self._cached("dKdtheta", _eval_dKdtheta))
        dzeta = np.asarray(self._cached("dKdzeta", _eval_dKdzeta))
        return np.stack([dtheta, dzeta], axis=1)

    def nu(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("nu", _eval_nu)))

    def dnudtheta(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dnudtheta", _eval_dnudtheta)))

    def dnudzeta(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dnudzeta", _eval_dnudzeta)))

    def dnuds(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dnuds", _eval_dnuds)))

    def nu_derivs(self) -> np.ndarray:
        ds = np.asarray(self._cached("dnuds", _eval_dnuds))
        dtheta = np.asarray(self._cached("dnudtheta", _eval_dnudtheta))
        dzeta = np.asarray(self._cached("dnudzeta", _eval_dnudzeta))
        return np.stack([ds, dtheta, dzeta], axis=1)

    def R(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("R", _eval_R)))

    def dRdtheta(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dRdtheta", _eval_dRdtheta)))

    def dRdzeta(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dRdzeta", _eval_dRdzeta)))

    def dRds(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dRds", _eval_dRds)))

    def R_derivs(self) -> np.ndarray:
        ds = np.asarray(self._cached("dRds", _eval_dRds))
        dtheta = np.asarray(self._cached("dRdtheta", _eval_dRdtheta))
        dzeta = np.asarray(self._cached("dRdzeta", _eval_dRdzeta))
        return np.stack([ds, dtheta, dzeta], axis=1)

    def Z(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("Z", _eval_Z)))

    def dZdtheta(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dZdtheta", _eval_dZdtheta)))

    def dZdzeta(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dZdzeta", _eval_dZdzeta)))

    def dZds(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dZds", _eval_dZds)))

    def Z_derivs(self) -> np.ndarray:
        ds = np.asarray(self._cached("dZds", _eval_dZds))
        dtheta = np.asarray(self._cached("dZdtheta", _eval_dZdtheta))
        dzeta = np.asarray(self._cached("dZdzeta", _eval_dZdzeta))
        return np.stack([ds, dtheta, dzeta], axis=1)

    def psip(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("psip", _eval_psip)))

    def G(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("G", _eval_G)))

    def I(self) -> np.ndarray:  # noqa: E743 — matches upstream API name
        return np.asarray(_as_column(self._cached("I", _eval_I)))

    def iota(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("iota", _eval_iota)))

    def dGds(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dGds", _eval_dGds)))

    def dIds(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dIds", _eval_dIds)))

    def diotads(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("diotads", _eval_diotads)))


# ----------------------------------------------------------------------
# BoozerAnalyticJAX — analytic Landreman-Sengupta field, pure JAX kernels
# ----------------------------------------------------------------------


class BoozerAnalyticJAX(Optimizable):
    """JAX-backed analytic Boozer field (Landreman & Sengupta, JPP 2018).

    Mirrors the public surface of
    :class:`simsopt.field.boozermagneticfield.BoozerAnalytic` (``set_points``,
    ``modB``, ``K``, ``G``, ``I``, ``iota``, ``psip``, derivative bundles)
    while routing the field-evaluation hot path through pure JAX kernels
    on an immutable frozen-state pytree.  This class does **not** inherit
    from ``sopp.BoozerMagneticField`` — it is a pure JAX-native sibling.

    Construction signature matches the CPU oracle exactly: ``(etabar, B0,
    N, G0, psi0, iota0, Bbar=1.0, I0=0.0, G1=0.0, I1=0.0, K1=0.0)``.

    Frozen-state semantics: the eleven scalar parameters are captured at
    construction time into an immutable ``BoozerAnalyticFrozenState``
    pytree.  Mutation requires constructing a new ``BoozerAnalyticJAX`` —
    there are no setters.
    """

    def __init__(
        self,
        etabar,
        B0,
        N,
        G0,
        psi0,
        iota0,
        Bbar=1.0,
        I0=0.0,
        G1=0.0,
        I1=0.0,
        K1=0.0,
    ):
        Optimizable.__init__(self, x0=np.asarray([]))
        self._frozen_state = freeze_boozer_analytic_state(
            etabar=etabar,
            B0=B0,
            N=N,
            G0=G0,
            psi0=psi0,
            iota0=iota0,
            Bbar=Bbar,
            I0=I0,
            G1=G1,
            I1=I1,
            K1=K1,
        )
        self._N_int = int(N)
        self._psi0_host = float(psi0)
        self._points = _as_jax_float64(np.zeros((0, 3), dtype=np.float64))
        self._cache: dict[str, jax.Array] = {}

    @classmethod
    def from_frozen_state(
        cls,
        frozen_state: BoozerAnalyticFrozenState,
        *,
        N: int,
        psi0: float,
    ):
        """Build a wrapper directly from a pre-built frozen state.

        This bypasses scalar re-coercion and is useful for tests or
        downstream consumers that want to mutate one parameter without
        going through the full constructor.
        """
        wrapper = cls.__new__(cls)
        Optimizable.__init__(wrapper, x0=np.asarray([]))
        wrapper._frozen_state = frozen_state
        wrapper._N_int = int(N)
        wrapper._psi0_host = float(psi0)
        wrapper._points = _as_jax_float64(np.zeros((0, 3), dtype=np.float64))
        wrapper._cache = {}
        return wrapper

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def psi0(self) -> float:
        return self._psi0_host

    @property
    def N(self) -> int:  # noqa: N802 — mirror CPU API
        return self._N_int

    @property
    def frozen_state(self) -> BoozerAnalyticFrozenState:
        return self._frozen_state

    # ------------------------------------------------------------------
    # Points / cache management
    # ------------------------------------------------------------------

    def set_points(self, points):
        """Set the Boozer ``(s, theta, zeta)`` evaluation points.

        Returns ``self`` to match the CPU ``set_points`` contract.
        """
        arr = _as_jax_float64(np.asarray(points, dtype=np.float64))
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"points must have shape (n, 3); got shape={tuple(arr.shape)!r}"
            )
        self._points = arr
        self._cache.clear()
        return self

    def get_points(self) -> np.ndarray:
        return np.asarray(self._points)

    def get_points_ref(self) -> jax.Array:
        return self._points

    def clear_cached_properties(self):
        self._cache.clear()

    # ------------------------------------------------------------------
    # Field evaluators
    # ------------------------------------------------------------------

    def _cached(self, name: str, fn) -> jax.Array:
        cached = self._cache.get(name)
        if cached is None:
            cached = fn(self._frozen_state, self._points)
            self._cache[name] = cached
        return cached

    def modB(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("modB", _eval_analytic_modB)))

    def dmodBds(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dmodBds", _eval_analytic_dmodBds)))

    def dmodBdtheta(self) -> np.ndarray:
        return np.asarray(
            _as_column(self._cached("dmodBdtheta", _eval_analytic_dmodBdtheta))
        )

    def dmodBdzeta(self) -> np.ndarray:
        return np.asarray(
            _as_column(self._cached("dmodBdzeta", _eval_analytic_dmodBdzeta))
        )

    def modB_derivs(self) -> np.ndarray:
        ds = np.asarray(self._cached("dmodBds", _eval_analytic_dmodBds))
        dtheta = np.asarray(self._cached("dmodBdtheta", _eval_analytic_dmodBdtheta))
        dzeta = np.asarray(self._cached("dmodBdzeta", _eval_analytic_dmodBdzeta))
        return np.stack([ds, dtheta, dzeta], axis=1)

    def K(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("K", _eval_analytic_K)))

    def dKdtheta(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dKdtheta", _eval_analytic_dKdtheta)))

    def dKdzeta(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dKdzeta", _eval_analytic_dKdzeta)))

    def K_derivs(self) -> np.ndarray:
        dtheta = np.asarray(self._cached("dKdtheta", _eval_analytic_dKdtheta))
        dzeta = np.asarray(self._cached("dKdzeta", _eval_analytic_dKdzeta))
        return np.stack([dtheta, dzeta], axis=1)

    def G(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("G", _eval_analytic_G)))

    def dGds(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dGds", _eval_analytic_dGds)))

    def I(self) -> np.ndarray:  # noqa: E743 — matches upstream API name
        return np.asarray(_as_column(self._cached("I", _eval_analytic_I)))

    def dIds(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dIds", _eval_analytic_dIds)))

    def iota(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("iota", _eval_analytic_iota)))

    def diotads(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("diotads", _eval_analytic_diotads)))

    def psip(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("psip", _eval_analytic_psip)))


# ----------------------------------------------------------------------
# InterpolatedBoozerFieldJAX (item N02)
# ----------------------------------------------------------------------


def _eval_scalar_factory(scalar_name: str):
    """Return a closed-over callable matching the ``(state, points)`` signature.

    The wrapper's ``_cached`` helper expects a single function reference
    per cached entry. This factory turns the per-scalar dispatch into a
    stable callable identity so the cache key is the scalar name.
    """

    def _eval(
        state: InterpolatedBoozerFieldFrozenState, points: jax.Array
    ) -> jax.Array:
        return _eval_interp_scalar(state, scalar_name, points)

    _eval.__name__ = f"_eval_interp_{scalar_name}"
    return _eval


_INTERP_EVALUATORS: dict[
    str, Callable[[InterpolatedBoozerFieldFrozenState, jax.Array], jax.Array]
] = {
    name: _eval_scalar_factory(name)
    for name in (*FLUX_FUNCTION_SCALARS, *tuple(SYMMETRY_EXPLOIT_SCALARS))
}


class InterpolatedBoozerFieldJAX(Optimizable):
    """JAX-native re-fit of a CPU :class:`BoozerMagneticField` on a regular grid.

    Mirrors the public surface of
    :class:`simsopt.field.boozermagneticfield.InterpolatedBoozerField`
    (``set_points``, ``modB``, ``K``, ``nu``, ``R``, ``Z``, ``G``, ``I``,
    ``iota``, ``psip``, and all first / second derivative bundles) while
    routing the field-evaluation hot path through pure JAX kernels on
    pre-fit :class:`InterpolatedBoozerFieldFrozenState` payloads.

    Construction signature exactly matches the CPU oracle:
    ``(field, degree, srange, thetarange, zetarange, extrapolate=True,
    nfp=1, stellsym=True)``.

    Architectural notes:

    - This wrapper does **not** inherit from ``sopp.BoozerMagneticField``
      or call into the C++ ``InterpolatedBoozerField`` class. It builds
      its own per-scalar interpolant set by sampling ``field``'s scalar
      getters on the regular grid.
    - Per-scalar interpolants are built **lazily** on the first call to
      each method, exactly mirroring the C++ template behaviour. The
      same base field may therefore be passed even if it only
      implements a subset of the 34 scalars: as long as the methods
      called on the wrapper map onto implemented getters on the base
      field, construction succeeds.
    - The wrapper exposes ``Optimizable`` with no DOFs of its own —
      mutating the wrapped CPU field after construction does NOT
      propagate to specs that have already been built. Newly-requested
      specs do sample the (possibly mutated) field state at the time of
      first request, just as the C++ template does.
    - The ``_simsopt_jax_native_field = True`` marker registers this
      class with the composition-strict-mode guard in
      :mod:`simsopt.field.magneticfield`.
    """

    _simsopt_jax_native_field = True

    def __init__(
        self,
        field,
        degree,
        srange,
        thetarange,
        zetarange,
        extrapolate: bool = True,
        nfp: int = 1,
        stellsym: bool = True,
        *,
        scalars: tuple[str, ...] | None = None,
    ):
        Optimizable.__init__(self, x0=np.asarray([]))
        self._field = field
        # Eagerly build the specified scalars at construction time.
        # ``scalars=None`` builds the full 34-scalar set (matches the
        # ``BoozerRadialInterpolant``-driven canonical use case). Pass a
        # tuple subset to match a base field that does not implement
        # every getter (e.g. ``BoozerAnalytic`` exposes only the 14
        # closed-form scalars).
        self._frozen_state = freeze_interpolated_boozer_field_state(
            field,
            degree=degree,
            srange=srange,
            thetarange=thetarange,
            zetarange=zetarange,
            extrapolate=extrapolate,
            nfp=nfp,
            stellsym=stellsym,
            scalars=scalars,
        )
        self._psi0_host = float(field.psi0)
        self._nfp = int(nfp)
        self._stellsym = bool(stellsym)
        self._extrapolate = bool(extrapolate)
        self._rule = _jax_core_uniform_rule(self._frozen_state.degree)
        self._points = _as_jax_float64(np.zeros((0, 3), dtype=np.float64))
        self._cache: dict[str, jax.Array] = {}

    @classmethod
    def from_frozen_state(
        cls,
        frozen_state: InterpolatedBoozerFieldFrozenState,
        *,
        psi0: float,
        nfp: int | None = None,
    ):
        """Build a wrapper directly from a pre-built frozen state.

        ``nfp`` is read from ``frozen_state.nfp`` unless an explicit
        override is supplied. This keeps the metadata consistent with
        the underlying interpolant geometry. The resulting wrapper has
        no reference to a source ``field`` and therefore cannot build
        additional specs on demand — any scalar method that was not
        pre-fit at freeze time will raise ``KeyError``.
        """
        wrapper = cls.__new__(cls)
        Optimizable.__init__(wrapper, x0=np.asarray([]))
        wrapper._field = None
        wrapper._frozen_state = frozen_state
        wrapper._psi0_host = float(psi0)
        wrapper._nfp = int(frozen_state.nfp if nfp is None else nfp)
        wrapper._stellsym = bool(frozen_state.stellsym)
        wrapper._extrapolate = bool(frozen_state.extrapolate)
        wrapper._rule = _jax_core_uniform_rule(frozen_state.degree)
        wrapper._points = _as_jax_float64(np.zeros((0, 3), dtype=np.float64))
        wrapper._cache = {}
        return wrapper

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def psi0(self) -> float:
        return self._psi0_host

    @property
    def nfp(self) -> int:
        return self._nfp

    @property
    def stellsym(self) -> bool:
        return self._stellsym

    @property
    def extrapolate(self) -> bool:
        return self._extrapolate

    @property
    def frozen_state(self) -> InterpolatedBoozerFieldFrozenState:
        return self._frozen_state

    # ------------------------------------------------------------------
    # Points / cache management
    # ------------------------------------------------------------------

    def set_points(self, points):
        """Set the Boozer ``(s, theta, zeta)`` evaluation points.

        Returns ``self`` to match the CPU ``set_points`` contract.
        """
        arr = _as_jax_float64(np.asarray(points, dtype=np.float64))
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"points must have shape (n, 3); got shape={tuple(arr.shape)!r}"
            )
        self._points = arr
        self._cache.clear()
        return self

    def get_points(self) -> np.ndarray:
        return np.asarray(self._points)

    def get_points_ref(self) -> jax.Array:
        return self._points

    def clear_cached_properties(self):
        self._cache.clear()

    # ------------------------------------------------------------------
    # Field evaluators
    # ------------------------------------------------------------------

    def _ensure_spec(self, name: str) -> None:
        """Lazy-build the per-scalar interpolant the first time it is read.

        Mirrors the C++ ``InterpolatedBoozerField`` lazy-build at header
        line 41-50 etc.: the interpolant is constructed and the base
        field is sampled only on the first call to the corresponding
        impl method. If the wrapper was built via
        :meth:`from_frozen_state` no base field reference is available,
        so unbuilt specs surface as ``KeyError``.
        """
        if self._frozen_state.has(name):
            return
        if self._field is None:
            raise KeyError(
                f"spec for scalar {name!r} was not pre-fit and the wrapper "
                f"has no base field to lazy-fit against (likely built via "
                f"from_frozen_state). Available scalars: "
                f"{sorted(self._frozen_state.specs)}"
            )
        spec = _interp_build_spec_for_scalar(
            self._field,
            scalar_name=name,
            rule=self._rule,
            s_range=self._frozen_state.s_range,
            theta_range=self._frozen_state.theta_range,
            zeta_range=self._frozen_state.zeta_range,
            extrapolate=self._frozen_state.extrapolate,
        )
        # The frozen state is a frozen dataclass; ``specs`` is a regular
        # dict held inside it, and we mutate that dict in place so the
        # add is observable through the public ``specs`` attribute.
        self._frozen_state.specs[name] = spec

    def _cached(self, name: str) -> jax.Array:
        cached = self._cache.get(name)
        if cached is None:
            self._ensure_spec(name)
            cached = _INTERP_EVALUATORS[name](self._frozen_state, self._points)
            self._cache[name] = cached
        return cached

    # Flux-function scalars — all (N, 1) shape
    def psip(self) -> np.ndarray:
        return np.asarray(self._cached("psip"))

    def G(self) -> np.ndarray:
        return np.asarray(self._cached("G"))

    def I(self) -> np.ndarray:  # noqa: E743 — matches CPU API name
        return np.asarray(self._cached("I"))

    def iota(self) -> np.ndarray:
        return np.asarray(self._cached("iota"))

    def dGds(self) -> np.ndarray:
        return np.asarray(self._cached("dGds"))

    def dIds(self) -> np.ndarray:
        return np.asarray(self._cached("dIds"))

    def diotads(self) -> np.ndarray:
        return np.asarray(self._cached("diotads"))

    # modB family
    def modB(self) -> np.ndarray:
        return np.asarray(self._cached("modB"))

    def dmodBdtheta(self) -> np.ndarray:
        return np.asarray(self._cached("dmodBdtheta"))

    def dmodBdzeta(self) -> np.ndarray:
        return np.asarray(self._cached("dmodBdzeta"))

    def dmodBds(self) -> np.ndarray:
        return np.asarray(self._cached("dmodBds"))

    def modB_derivs(self) -> np.ndarray:
        return np.asarray(self._cached("modB_derivs"))

    def d2modBdtheta2(self) -> np.ndarray:
        return np.asarray(self._cached("d2modBdtheta2"))

    def d2modBdzeta2(self) -> np.ndarray:
        return np.asarray(self._cached("d2modBdzeta2"))

    def d2modBdthetadzeta(self) -> np.ndarray:
        return np.asarray(self._cached("d2modBdthetadzeta"))

    # K family
    def K(self) -> np.ndarray:
        return np.asarray(self._cached("K"))

    def dKdtheta(self) -> np.ndarray:
        return np.asarray(self._cached("dKdtheta"))

    def dKdzeta(self) -> np.ndarray:
        return np.asarray(self._cached("dKdzeta"))

    def K_derivs(self) -> np.ndarray:
        return np.asarray(self._cached("K_derivs"))

    # nu family
    def nu(self) -> np.ndarray:
        return np.asarray(self._cached("nu"))

    def dnudtheta(self) -> np.ndarray:
        return np.asarray(self._cached("dnudtheta"))

    def dnudzeta(self) -> np.ndarray:
        return np.asarray(self._cached("dnudzeta"))

    def dnuds(self) -> np.ndarray:
        return np.asarray(self._cached("dnuds"))

    def nu_derivs(self) -> np.ndarray:
        return np.asarray(self._cached("nu_derivs"))

    # R family
    def R(self) -> np.ndarray:
        return np.asarray(self._cached("R"))

    def dRdtheta(self) -> np.ndarray:
        return np.asarray(self._cached("dRdtheta"))

    def dRdzeta(self) -> np.ndarray:
        return np.asarray(self._cached("dRdzeta"))

    def dRds(self) -> np.ndarray:
        return np.asarray(self._cached("dRds"))

    def R_derivs(self) -> np.ndarray:
        return np.asarray(self._cached("R_derivs"))

    # Z family
    def Z(self) -> np.ndarray:
        return np.asarray(self._cached("Z"))

    def dZdtheta(self) -> np.ndarray:
        return np.asarray(self._cached("dZdtheta"))

    def dZdzeta(self) -> np.ndarray:
        return np.asarray(self._cached("dZdzeta"))

    def dZds(self) -> np.ndarray:
        return np.asarray(self._cached("dZds"))

    def Z_derivs(self) -> np.ndarray:
        return np.asarray(self._cached("Z_derivs"))
