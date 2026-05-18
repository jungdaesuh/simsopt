"""Pure JAX frozen-state kernels for ``BoozerRadialInterpolant``."""

from __future__ import annotations

from dataclasses import dataclass, fields

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import PPoly

from .boozer_fixed_state import PiecewisePolynomial1D, ppoly_eval
from .boozer_radial_interp import (
    inverse_fourier_transform_even,
    inverse_fourier_transform_odd,
)

__all__ = [
    "BoozerRadialColumnBundle",
    "BoozerRadialInterpolantFrozenState",
    "_eval_G",
    "_eval_I",
    "_eval_K",
    "_eval_R",
    "_eval_Z",
    "_eval_dGds",
    "_eval_dIds",
    "_eval_dKdtheta",
    "_eval_dKdzeta",
    "_eval_dRdtheta",
    "_eval_dRds",
    "_eval_dRdzeta",
    "_eval_dZdtheta",
    "_eval_dZds",
    "_eval_dZdzeta",
    "_eval_diotads",
    "_eval_dmodBds",
    "_eval_dmodBdtheta",
    "_eval_dmodBdzeta",
    "_eval_dnuds",
    "_eval_dnudtheta",
    "_eval_dnudzeta",
    "_eval_radial_columns",
    "_eval_iota",
    "_eval_modB",
    "_eval_nu",
    "_eval_psip",
    "_frozen_state_from_host",
    "_frozen_state_to_host",
    "freeze_boozer_radial_state",
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
    ``simsopt.field.boozermagneticfield``). For ``rescale=True``, this
    is the upstream inverse-power normalization: ``m=1`` modes use
    ``s^{-1/2}``, odd ``m>1`` modes use ``s^{-3/2}``, and even ``m>1``
    modes use ``s^{-1}``. Samples below the first retained half-grid
    knot follow the frozen CPU spline extrapolation for that inverse
    factor; the JAX wrapper does not recompute a separate closed-form
    positive-power factor.

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

@dataclass(frozen=True)
class BoozerRadialColumnBundle:
    """Per-points radial profile evaluations shared by scalar siblings."""

    psip: jax.Array
    G: jax.Array
    I: jax.Array
    iota: jax.Array
    dGds: jax.Array
    dIds: jax.Array
    diotads: jax.Array
    bmnc: jax.Array
    dbmncds: jax.Array
    rmnc: jax.Array
    drmncds: jax.Array
    zmns: jax.Array
    dzmnsds: jax.Array
    numns: jax.Array
    dnumnsds: jax.Array
    bmns: jax.Array
    dbmnsds: jax.Array
    rmns: jax.Array
    drmnsds: jax.Array
    zmnc: jax.Array
    dzmncds: jax.Array
    numnc: jax.Array
    dnumncds: jax.Array
    mn_factor: jax.Array
    d_mn_factor: jax.Array
    kmns: jax.Array
    kmnc: jax.Array


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


def _eval_radial_columns(
    state: BoozerRadialInterpolantFrozenState, s: jax.Array
) -> BoozerRadialColumnBundle:
    """Evaluate every radial profile once for one points/state cycle."""
    return BoozerRadialColumnBundle(
        psip=_scalar_at(s, state.psip),
        G=_scalar_at(s, state.G),
        I=_scalar_at(s, state.I),
        iota=_scalar_at(s, state.iota),
        dGds=_scalar_at(s, state.dGds),
        dIds=_scalar_at(s, state.dIds),
        diotads=_scalar_at(s, state.diotads),
        bmnc=_column_at(s, state.bmnc),
        dbmncds=_column_at(s, state.dbmncds),
        rmnc=_column_at(s, state.rmnc),
        drmncds=_column_at(s, state.drmncds),
        zmns=_column_at(s, state.zmns),
        dzmnsds=_column_at(s, state.dzmnsds),
        numns=_column_at(s, state.numns),
        dnumnsds=_column_at(s, state.dnumnsds),
        bmns=_column_at(s, state.bmns),
        dbmnsds=_column_at(s, state.dbmnsds),
        rmns=_column_at(s, state.rmns),
        drmnsds=_column_at(s, state.drmnsds),
        zmnc=_column_at(s, state.zmnc),
        dzmncds=_column_at(s, state.dzmncds),
        numnc=_column_at(s, state.numnc),
        dnumncds=_column_at(s, state.dnumncds),
        mn_factor=_column_at(s, state.mn_factor),
        d_mn_factor=_column_at(s, state.d_mn_factor),
        kmns=_column_at(s, state.kmns),
        kmnc=_column_at(s, state.kmnc),
    )


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


def _eval_modB_from_columns(
    state: BoozerRadialInterpolantFrozenState,
    columns: BoozerRadialColumnBundle,
    points: jax.Array,
) -> jax.Array:
    thetas = points[:, 1]
    zetas = points[:, 2]
    bmnc = _normalize(columns.bmnc, columns.mn_factor)
    result = inverse_fourier_transform_even(bmnc, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        bmns = _normalize(columns.bmns, columns.mn_factor)
        result = result + inverse_fourier_transform_odd(
            bmns, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_dmodBdtheta_from_columns(
    state: BoozerRadialInterpolantFrozenState,
    columns: BoozerRadialColumnBundle,
    points: jax.Array,
) -> jax.Array:
    thetas = points[:, 1]
    zetas = points[:, 2]
    xm_col = state.xm[:, None]
    bmnc = -xm_col * _normalize(columns.bmnc, columns.mn_factor)
    result = inverse_fourier_transform_odd(bmnc, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        bmns = xm_col * _normalize(columns.bmns, columns.mn_factor)
        result = result + inverse_fourier_transform_even(
            bmns, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_dmodBdzeta_from_columns(
    state: BoozerRadialInterpolantFrozenState,
    columns: BoozerRadialColumnBundle,
    points: jax.Array,
) -> jax.Array:
    thetas = points[:, 1]
    zetas = points[:, 2]
    xn_col = state.xn[:, None]
    bmnc = xn_col * _normalize(columns.bmnc, columns.mn_factor)
    result = inverse_fourier_transform_odd(bmnc, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        bmns = -xn_col * _normalize(columns.bmns, columns.mn_factor)
        result = result + inverse_fourier_transform_even(
            bmns, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_dmodBds_from_columns(
    state: BoozerRadialInterpolantFrozenState,
    columns: BoozerRadialColumnBundle,
    points: jax.Array,
) -> jax.Array:
    thetas = points[:, 1]
    zetas = points[:, 2]
    bmnc = _radial_normalized(
        columns.bmnc, columns.dbmncds, columns.mn_factor, columns.d_mn_factor
    )
    result = inverse_fourier_transform_even(bmnc, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        bmns = _radial_normalized(
            columns.bmns, columns.dbmnsds, columns.mn_factor, columns.d_mn_factor
        )
        result = result + inverse_fourier_transform_odd(
            bmns, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_R_from_columns(
    state: BoozerRadialInterpolantFrozenState,
    columns: BoozerRadialColumnBundle,
    points: jax.Array,
) -> jax.Array:
    thetas = points[:, 1]
    zetas = points[:, 2]
    rmnc = _normalize(columns.rmnc, columns.mn_factor)
    result = inverse_fourier_transform_even(rmnc, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        rmns = _normalize(columns.rmns, columns.mn_factor)
        result = result + inverse_fourier_transform_odd(
            rmns, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_dRdtheta_from_columns(
    state: BoozerRadialInterpolantFrozenState,
    columns: BoozerRadialColumnBundle,
    points: jax.Array,
) -> jax.Array:
    thetas = points[:, 1]
    zetas = points[:, 2]
    xm_col = state.xm[:, None]
    rmnc = -xm_col * _normalize(columns.rmnc, columns.mn_factor)
    result = inverse_fourier_transform_odd(rmnc, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        rmns = xm_col * _normalize(columns.rmns, columns.mn_factor)
        result = result + inverse_fourier_transform_even(
            rmns, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_dRdzeta_from_columns(
    state: BoozerRadialInterpolantFrozenState,
    columns: BoozerRadialColumnBundle,
    points: jax.Array,
) -> jax.Array:
    thetas = points[:, 1]
    zetas = points[:, 2]
    xn_col = state.xn[:, None]
    rmnc = xn_col * _normalize(columns.rmnc, columns.mn_factor)
    result = inverse_fourier_transform_odd(rmnc, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        rmns = -xn_col * _normalize(columns.rmns, columns.mn_factor)
        result = result + inverse_fourier_transform_even(
            rmns, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_dRds_from_columns(
    state: BoozerRadialInterpolantFrozenState,
    columns: BoozerRadialColumnBundle,
    points: jax.Array,
) -> jax.Array:
    thetas = points[:, 1]
    zetas = points[:, 2]
    rmnc = _radial_normalized(
        columns.rmnc, columns.drmncds, columns.mn_factor, columns.d_mn_factor
    )
    result = inverse_fourier_transform_even(rmnc, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        rmns = _radial_normalized(
            columns.rmns, columns.drmnsds, columns.mn_factor, columns.d_mn_factor
        )
        result = result + inverse_fourier_transform_odd(
            rmns, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_Z_from_columns(
    state: BoozerRadialInterpolantFrozenState,
    columns: BoozerRadialColumnBundle,
    points: jax.Array,
) -> jax.Array:
    thetas = points[:, 1]
    zetas = points[:, 2]
    zmns = _normalize(columns.zmns, columns.mn_factor)
    result = inverse_fourier_transform_odd(zmns, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        zmnc = _normalize(columns.zmnc, columns.mn_factor)
        result = result + inverse_fourier_transform_even(
            zmnc, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_dZdtheta_from_columns(
    state: BoozerRadialInterpolantFrozenState,
    columns: BoozerRadialColumnBundle,
    points: jax.Array,
) -> jax.Array:
    thetas = points[:, 1]
    zetas = points[:, 2]
    xm_col = state.xm[:, None]
    zmns = xm_col * _normalize(columns.zmns, columns.mn_factor)
    result = inverse_fourier_transform_even(zmns, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        zmnc = -xm_col * _normalize(columns.zmnc, columns.mn_factor)
        result = result + inverse_fourier_transform_odd(
            zmnc, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_dZdzeta_from_columns(
    state: BoozerRadialInterpolantFrozenState,
    columns: BoozerRadialColumnBundle,
    points: jax.Array,
) -> jax.Array:
    thetas = points[:, 1]
    zetas = points[:, 2]
    xn_col = state.xn[:, None]
    zmns = -xn_col * _normalize(columns.zmns, columns.mn_factor)
    result = inverse_fourier_transform_even(zmns, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        zmnc = xn_col * _normalize(columns.zmnc, columns.mn_factor)
        result = result + inverse_fourier_transform_odd(
            zmnc, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_dZds_from_columns(
    state: BoozerRadialInterpolantFrozenState,
    columns: BoozerRadialColumnBundle,
    points: jax.Array,
) -> jax.Array:
    thetas = points[:, 1]
    zetas = points[:, 2]
    zmns = _radial_normalized(
        columns.zmns, columns.dzmnsds, columns.mn_factor, columns.d_mn_factor
    )
    result = inverse_fourier_transform_odd(zmns, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        zmnc = _radial_normalized(
            columns.zmnc, columns.dzmncds, columns.mn_factor, columns.d_mn_factor
        )
        result = result + inverse_fourier_transform_even(
            zmnc, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_nu_from_columns(
    state: BoozerRadialInterpolantFrozenState,
    columns: BoozerRadialColumnBundle,
    points: jax.Array,
) -> jax.Array:
    thetas = points[:, 1]
    zetas = points[:, 2]
    numns = _normalize(columns.numns, columns.mn_factor)
    result = inverse_fourier_transform_odd(numns, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        numnc = _normalize(columns.numnc, columns.mn_factor)
        result = result + inverse_fourier_transform_even(
            numnc, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_dnudtheta_from_columns(
    state: BoozerRadialInterpolantFrozenState,
    columns: BoozerRadialColumnBundle,
    points: jax.Array,
) -> jax.Array:
    thetas = points[:, 1]
    zetas = points[:, 2]
    xm_col = state.xm[:, None]
    numns = xm_col * _normalize(columns.numns, columns.mn_factor)
    result = inverse_fourier_transform_even(numns, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        numnc = -xm_col * _normalize(columns.numnc, columns.mn_factor)
        result = result + inverse_fourier_transform_odd(
            numnc, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_dnudzeta_from_columns(
    state: BoozerRadialInterpolantFrozenState,
    columns: BoozerRadialColumnBundle,
    points: jax.Array,
) -> jax.Array:
    thetas = points[:, 1]
    zetas = points[:, 2]
    xn_col = state.xn[:, None]
    numns = -xn_col * _normalize(columns.numns, columns.mn_factor)
    result = inverse_fourier_transform_even(numns, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        numnc = xn_col * _normalize(columns.numnc, columns.mn_factor)
        result = result + inverse_fourier_transform_odd(
            numnc, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_dnuds_from_columns(
    state: BoozerRadialInterpolantFrozenState,
    columns: BoozerRadialColumnBundle,
    points: jax.Array,
) -> jax.Array:
    thetas = points[:, 1]
    zetas = points[:, 2]
    numns = _radial_normalized(
        columns.numns, columns.dnumnsds, columns.mn_factor, columns.d_mn_factor
    )
    result = inverse_fourier_transform_odd(numns, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        numnc = _radial_normalized(
            columns.numnc, columns.dnumncds, columns.mn_factor, columns.d_mn_factor
        )
        result = result + inverse_fourier_transform_even(
            numnc, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_K_from_columns(
    state: BoozerRadialInterpolantFrozenState,
    columns: BoozerRadialColumnBundle,
    points: jax.Array,
) -> jax.Array:
    if state.no_K:
        return jnp.zeros(points.shape[0], dtype=jnp.float64)
    thetas = points[:, 1]
    zetas = points[:, 2]
    kmns = _normalize(columns.kmns, columns.mn_factor)
    result = inverse_fourier_transform_odd(kmns, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        kmnc = _normalize(columns.kmnc, columns.mn_factor)
        result = result + inverse_fourier_transform_even(
            kmnc, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_dKdtheta_from_columns(
    state: BoozerRadialInterpolantFrozenState,
    columns: BoozerRadialColumnBundle,
    points: jax.Array,
) -> jax.Array:
    if state.no_K:
        return jnp.zeros(points.shape[0], dtype=jnp.float64)
    thetas = points[:, 1]
    zetas = points[:, 2]
    xm_col = state.xm[:, None]
    kmns = xm_col * _normalize(columns.kmns, columns.mn_factor)
    result = inverse_fourier_transform_even(kmns, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        kmnc = -xm_col * _normalize(columns.kmnc, columns.mn_factor)
        result = result + inverse_fourier_transform_odd(
            kmnc, state.xm, state.xn, thetas, zetas
        )
    return result


def _eval_dKdzeta_from_columns(
    state: BoozerRadialInterpolantFrozenState,
    columns: BoozerRadialColumnBundle,
    points: jax.Array,
) -> jax.Array:
    if state.no_K:
        return jnp.zeros(points.shape[0], dtype=jnp.float64)
    thetas = points[:, 1]
    zetas = points[:, 2]
    xn_col = state.xn[:, None]
    kmns = -xn_col * _normalize(columns.kmns, columns.mn_factor)
    result = inverse_fourier_transform_even(kmns, state.xm, state.xn, thetas, zetas)
    if not state.stellsym:
        kmnc = xn_col * _normalize(columns.kmnc, columns.mn_factor)
        result = result + inverse_fourier_transform_odd(
            kmnc, state.xm, state.xn, thetas, zetas
        )
    return result


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
