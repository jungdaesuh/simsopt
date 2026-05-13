"""JAX port of dommaschk.cpp and reiman.cpp.

This module provides pure JAX implementations of the raw analytic field
kernels used by ``simsopt.field.magneticfieldclasses.Dommaschk`` and
``simsopt.field.magneticfieldclasses.Reiman``.

The Dommaschk vacuum field (W. Dommaschk, 1986, *Computer Physics
Communications* **40**, 203-218) is expressed through the scalar potential
helpers ``D_mn`` and ``N_mn`` and their first derivatives. The Reiman
island-model field (Reiman & Greenside, 1986, *Computer Physics
Communications* **43**, 157-167) admits a closed-form series in
``rmin = sqrt((R - R_axis)^2 + Z^2)``.

Public surface
--------------

- :class:`DommaschkSpec` -- immutable container of ``(m, n)`` mode indices
  and per-mode coefficient pairs.
- :func:`dommaschk_B` and :func:`dommaschk_dB` -- raw Dommaschk Cartesian
  field and gradient with shape ``[K, N, 3]`` / ``[K, N, 3, 3]``. The
  baseline ``ToroidalField(R0=1, B0=1)`` contribution is *not* included:
  the public :class:`~simsopt.field.magneticfieldclasses.Dommaschk`
  wrapper adds it explicitly. This matches ``sopp.DommaschkB`` /
  ``sopp.DommaschkdB`` outputs exactly.
- :class:`ReimanSpec` -- immutable container of ``iota0``, ``iota1``,
  Fourier indices ``k_theta``, coefficients ``epsilon``, and the
  toroidal-symmetry parameter ``m0_symmetry``.
- :func:`reiman_B` and :func:`reiman_dB` -- Cartesian field and gradient
  with shape ``[N, 3]`` / ``[N, 3, 3]``.

All kernels require ``R = sqrt(x^2 + y^2) > 0`` at every evaluation
point. Mode indices ``m`` and ``n`` (Dommaschk) and ``k_theta`` (Reiman)
must be Python integers; they are static metadata in the JIT cache key.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "DommaschkSpec",
    "ReimanSpec",
    "dommaschk_B",
    "dommaschk_dB",
    "reiman_B",
    "reiman_dB",
]


# ── Dommaschk scalar helper tables (Python, static at trace time) ─────


def _alpha_py(m: int, l: int) -> float:
    """Equation-(40) ``alpha`` helper from dommaschk.cpp.

    Returns 0 when ``l < 0``; otherwise
    ``(-1)^l / (Gamma(m+l+1) * Gamma(l+1) * 2^(2l+m))``. The integer
    arguments are bounded so ``math.gamma`` evaluates only on positive
    reals (cf. the ``l < 0`` early return in the C++ kernel).
    """

    if l < 0:
        return 0.0
    return ((-1.0) ** l) / (
        math.gamma(m + l + 1) * math.gamma(l + 1) * (2.0 ** (2 * l + m))
    )


def _alphas_py(m: int, l: int) -> float:
    return (2 * l + m) * _alpha_py(m, l)


def _beta_py(m: int, l: int) -> float:
    if l < 0 or l >= m:
        return 0.0
    return math.gamma(m - l) / (math.gamma(l + 1) * (2.0 ** (2 * l - m + 1)))


def _betas_py(m: int, l: int) -> float:
    return (2 * l - m) * _beta_py(m, l)


def _gamma1_py(m: int, l: int) -> float:
    if l <= 0:
        return 0.0
    sum_n = 0.0
    for i in range(1, l + 1):
        sum_n += 1.0 / i + 1.0 / (m + i)
    return (_alpha_py(m, l) / 2.0) * sum_n


def _gammas_py(m: int, l: int) -> float:
    return (2 * l + m) * _gamma1_py(m, l)


# ── Dommaschk D_mn / N_mn coefficient tables ──────────────────────────


class _DommaschkTerm(NamedTuple):
    """One R^p * (log R)^q * Z^s contribution to a Dmn / Nmn series."""

    exp_R: int  # exponent of R
    exp_Z: int  # exponent of Z
    exp_log: int  # 0 or 1
    coeff: float  # scalar combinatorial factor


def _accumulate_terms(
    terms: list[_DommaschkTerm], exp_R: int, exp_Z: int, exp_log: int, coeff: float
) -> None:
    """Append a term to the list; merge identical (exp_R, exp_Z, exp_log)."""

    if coeff == 0.0:
        return
    for idx, term in enumerate(terms):
        if term.exp_R == exp_R and term.exp_Z == exp_Z and term.exp_log == exp_log:
            terms[idx] = _DommaschkTerm(exp_R, exp_Z, exp_log, term.coeff + coeff)
            return
    terms.append(_DommaschkTerm(exp_R, exp_Z, exp_log, coeff))


def _dmn_terms(m: int, n: int) -> list[_DommaschkTerm]:
    """Return polynomial expansion of ``Dmn(m, n, R, Z)``.

    Mirrors ``dommaschk.cpp::Dmn`` exactly. Each term has the form
    ``coeff * R^exp_R * (log R)^exp_log * Z^exp_Z`` so that the JAX
    kernel can sum the terms via dense ``jnp.power`` calls.
    """

    if n < 0:
        return []
    terms: list[_DommaschkTerm] = []
    for k in range(n // 2 + 1):
        z_pow = n - 2 * k
        outer = 1.0 / math.gamma(z_pow + 1)
        for j in range(k + 1):
            inner_log = -_alpha_py(m, j) * _alphas_py(m, k - m - j)
            inner_const = -(
                _alpha_py(m, j) * (_gammas_py(m, k - m - j) - _alpha_py(m, k - m - j))
                - _gamma1_py(m, j) * _alphas_py(m, k - m - j)
                + _alpha_py(m, j) * _betas_py(m, k - j)
            )
            r_pow_pos = 2 * j + m
            _accumulate_terms(terms, r_pow_pos, z_pow, 1, outer * inner_log)
            _accumulate_terms(terms, r_pow_pos, z_pow, 0, outer * inner_const)
            r_pow_neg = 2 * j - m
            inner_neg = _alphas_py(m, k - j) * _beta_py(m, j)
            _accumulate_terms(terms, r_pow_neg, z_pow, 0, outer * inner_neg)
    return [t for t in terms if t.coeff != 0.0]


def _nmn_terms(m: int, n: int) -> list[_DommaschkTerm]:
    """Return polynomial expansion of ``Nmn(m, n, R, Z)`` from C++."""

    if n < 0:
        return []
    terms: list[_DommaschkTerm] = []
    for k in range(n // 2 + 1):
        z_pow = n - 2 * k
        outer = 1.0 / math.gamma(z_pow + 1)
        for j in range(k + 1):
            inner_log = _alpha_py(m, j) * _alpha_py(m, k - m - j)
            inner_const = (
                _alpha_py(m, j) * _gamma1_py(m, k - m - j)
                - _gamma1_py(m, j) * _alpha_py(m, k - m - j)
                + _alpha_py(m, j) * _beta_py(m, k - j)
            )
            r_pow_pos = 2 * j + m
            _accumulate_terms(terms, r_pow_pos, z_pow, 1, outer * inner_log)
            _accumulate_terms(terms, r_pow_pos, z_pow, 0, outer * inner_const)
            r_pow_neg = 2 * j - m
            inner_neg = -_alpha_py(m, k - j) * _beta_py(m, j)
            _accumulate_terms(terms, r_pow_neg, z_pow, 0, outer * inner_neg)
    return [t for t in terms if t.coeff != 0.0]


def _diff_R_terms(terms: list[_DommaschkTerm]) -> list[_DommaschkTerm]:
    """Apply ``d/dR`` to a term list.

    ``d/dR [R^p * (log R)^q] = R^{p-1} * (p * (log R)^q + q * (log R)^{q-1})``.
    ``q in {0, 1}`` covers every term used by the Dommaschk series.
    """

    out: list[_DommaschkTerm] = []
    for term in terms:
        if term.exp_log == 0:
            new_coeff = term.coeff * term.exp_R
            _accumulate_terms(out, term.exp_R - 1, term.exp_Z, 0, new_coeff)
        elif term.exp_log == 1:
            # d/dR [R^p log R] = p R^{p-1} log R + R^{p-1}
            _accumulate_terms(
                out, term.exp_R - 1, term.exp_Z, 1, term.coeff * term.exp_R
            )
            _accumulate_terms(out, term.exp_R - 1, term.exp_Z, 0, term.coeff)
        else:
            raise ValueError(f"Unsupported log exponent: {term.exp_log}")
    return [t for t in out if t.coeff != 0.0]


def _diff_Z_terms(terms: list[_DommaschkTerm]) -> list[_DommaschkTerm]:
    """Apply ``d/dZ`` to a term list."""

    out: list[_DommaschkTerm] = []
    for term in terms:
        if term.exp_Z == 0:
            continue
        _accumulate_terms(
            out,
            term.exp_R,
            term.exp_Z - 1,
            term.exp_log,
            term.coeff * term.exp_Z,
        )
    return [t for t in out if t.coeff != 0.0]


# ── Per-(m, n) cached compiled evaluators ─────────────────────────────


def _eval_terms_dense(
    terms: list[_DommaschkTerm], R: jax.Array, Z: jax.Array, log_R: jax.Array
) -> jax.Array:
    """Evaluate a Dommaschk term list at arrays ``(R, Z, log_R)``.

    Returns a scalar JAX array with the broadcast shape of the inputs.
    """

    if not terms:
        return jnp.zeros_like(R)
    total = jnp.zeros_like(R)
    for term in terms:
        contribution = jnp.asarray(term.coeff, dtype=R.dtype) * jnp.power(
            R, jnp.asarray(term.exp_R, dtype=R.dtype)
        )
        if term.exp_log == 1:
            contribution = contribution * log_R
        if term.exp_Z != 0:
            contribution = contribution * jnp.power(
                Z, jnp.asarray(term.exp_Z, dtype=Z.dtype)
            )
        total = total + contribution
    return total


@lru_cache(maxsize=None)
def _dommaschk_term_bundle(m: int, n: int) -> dict[str, tuple[_DommaschkTerm, ...]]:
    """Cached polynomial expansions for all Dmn/Nmn derivatives at ``(m, n)``."""

    d_terms = _dmn_terms(m, n)
    dr_d_terms = _diff_R_terms(d_terms)
    dz_d_terms = _diff_Z_terms(d_terms)
    drr_d_terms = _diff_R_terms(dr_d_terms)
    drz_d_terms = _diff_Z_terms(dr_d_terms)
    dzz_d_terms = _diff_Z_terms(dz_d_terms)

    n_minus_terms = _nmn_terms(m, n - 1)
    dr_n_terms = _diff_R_terms(n_minus_terms)
    dz_n_terms = _diff_Z_terms(n_minus_terms)
    drr_n_terms = _diff_R_terms(dr_n_terms)
    drz_n_terms = _diff_Z_terms(dr_n_terms)
    dzz_n_terms = _diff_Z_terms(dz_n_terms)

    return {
        "D": tuple(d_terms),
        "dR_D": tuple(dr_d_terms),
        "dZ_D": tuple(dz_d_terms),
        "dRR_D": tuple(drr_d_terms),
        "dRZ_D": tuple(drz_d_terms),
        "dZZ_D": tuple(dzz_d_terms),
        "N": tuple(n_minus_terms),
        "dR_N": tuple(dr_n_terms),
        "dZ_N": tuple(dz_n_terms),
        "dRR_N": tuple(drr_n_terms),
        "dRZ_N": tuple(drz_n_terms),
        "dZZ_N": tuple(dzz_n_terms),
    }


# ── Public spec containers ────────────────────────────────────────────


@dataclass(frozen=True)
class DommaschkSpec:
    """Static (m, n) mode list plus runtime coefficient pairs.

    Attributes
    ----------
    m, n
        Tuples of Python integers with the same length ``K``. These are
        treated as JIT-static metadata; changing them recompiles the
        per-mode kernel.
    coeffs
        ``float64`` JAX array of shape ``[K, 2]`` carrying the two
        Dommaschk coefficients per mode. Treated as runtime data.
    """

    m: tuple[int, ...]
    n: tuple[int, ...]
    coeffs: jax.Array


@dataclass(frozen=True)
class ReimanSpec:
    """Reiman island-model spec.

    ``k_theta`` is a tuple of Python integers (static metadata). The
    remaining attributes are runtime float scalars / arrays. ``epsilon``
    is shape ``[M]`` with ``M = len(k_theta)``.
    """

    iota0: float
    iota1: float
    k_theta: tuple[int, ...]
    epsilon: jax.Array
    m0_symmetry: int


# ── Dommaschk per-mode evaluator ──────────────────────────────────────


def _dommaschk_single_mode_BR_BZ_Bphi(
    m: int,
    n: int,
    R: jax.Array,
    Z: jax.Array,
    phi: jax.Array,
    coeff1: jax.Array,
    coeff2: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return ``(BR, BZ, Bphi)`` from Dommaschk Phi for a single (m, n)."""

    bundle = _dommaschk_term_bundle(m, n)
    log_R = jnp.log(R)
    m_phi = jnp.asarray(m, dtype=phi.dtype) * phi
    sin_m_phi = jnp.sin(m_phi)
    cos_m_phi = jnp.cos(m_phi)
    if n % 2 == 0:
        a = jnp.zeros_like(coeff1)
        d = jnp.zeros_like(coeff1)
        b = coeff1
        c = coeff2
    else:
        a = coeff1
        d = coeff2
        b = jnp.zeros_like(coeff1)
        c = jnp.zeros_like(coeff1)

    angle_d = a * cos_m_phi + b * sin_m_phi
    angle_n = c * cos_m_phi + d * sin_m_phi
    angle_d_phi = -a * sin_m_phi + b * cos_m_phi
    angle_n_phi = -c * sin_m_phi + d * cos_m_phi

    dR_D = _eval_terms_dense(bundle["dR_D"], R, Z, log_R)
    dR_N = _eval_terms_dense(bundle["dR_N"], R, Z, log_R)
    dZ_D = _eval_terms_dense(bundle["dZ_D"], R, Z, log_R)
    dZ_N = _eval_terms_dense(bundle["dZ_N"], R, Z, log_R)
    D = _eval_terms_dense(bundle["D"], R, Z, log_R)
    N = _eval_terms_dense(bundle["N"], R, Z, log_R)

    m_scalar = jnp.asarray(m, dtype=R.dtype)
    BR = angle_d * dR_D + angle_n * dR_N
    BZ = angle_d * dZ_D + angle_n * dZ_N
    Bphi = m_scalar * (angle_d_phi * D + angle_n_phi * N) / R
    return BR, BZ, Bphi


def _dommaschk_single_mode_dB_local(
    m: int,
    n: int,
    R: jax.Array,
    Z: jax.Array,
    phi: jax.Array,
    coeff1: jax.Array,
    coeff2: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Return ``(BR, BZ, Bphi, dRBR, dZBR, dphiBR, dRBZ, dZBZ, dphiBZ,
    dRBphi, dZBphi, dphiBphi)`` for the (m, n) component."""

    bundle = _dommaschk_term_bundle(m, n)
    log_R = jnp.log(R)
    m_phi = jnp.asarray(m, dtype=phi.dtype) * phi
    sin_m_phi = jnp.sin(m_phi)
    cos_m_phi = jnp.cos(m_phi)
    if n % 2 == 0:
        a = jnp.zeros_like(coeff1)
        d = jnp.zeros_like(coeff1)
        b = coeff1
        c = coeff2
    else:
        a = coeff1
        d = coeff2
        b = jnp.zeros_like(coeff1)
        c = jnp.zeros_like(coeff1)

    angle_d = a * cos_m_phi + b * sin_m_phi
    angle_n = c * cos_m_phi + d * sin_m_phi
    angle_d_phi = -a * sin_m_phi + b * cos_m_phi
    angle_n_phi = -c * sin_m_phi + d * cos_m_phi
    angle_d_phi2 = -a * cos_m_phi - b * sin_m_phi
    angle_n_phi2 = -c * cos_m_phi - d * sin_m_phi

    D = _eval_terms_dense(bundle["D"], R, Z, log_R)
    N = _eval_terms_dense(bundle["N"], R, Z, log_R)
    dR_D = _eval_terms_dense(bundle["dR_D"], R, Z, log_R)
    dR_N = _eval_terms_dense(bundle["dR_N"], R, Z, log_R)
    dZ_D = _eval_terms_dense(bundle["dZ_D"], R, Z, log_R)
    dZ_N = _eval_terms_dense(bundle["dZ_N"], R, Z, log_R)
    dRR_D = _eval_terms_dense(bundle["dRR_D"], R, Z, log_R)
    dRR_N = _eval_terms_dense(bundle["dRR_N"], R, Z, log_R)
    dRZ_D = _eval_terms_dense(bundle["dRZ_D"], R, Z, log_R)
    dRZ_N = _eval_terms_dense(bundle["dRZ_N"], R, Z, log_R)
    dZZ_D = _eval_terms_dense(bundle["dZZ_D"], R, Z, log_R)
    dZZ_N = _eval_terms_dense(bundle["dZZ_N"], R, Z, log_R)

    m_scalar = jnp.asarray(m, dtype=R.dtype)
    BR = angle_d * dR_D + angle_n * dR_N
    BZ = angle_d * dZ_D + angle_n * dZ_N
    Bphi = m_scalar * (angle_d_phi * D + angle_n_phi * N) / R

    dRBR = angle_d * dRR_D + angle_n * dRR_N
    dZBZ = angle_d * dZZ_D + angle_n * dZZ_N
    dRBZ = angle_d * dRZ_D + angle_n * dRZ_N
    dZBR = dRBZ  # C++ identity: dZBR(m,n,...) == dRBZ(m,n,...).
    dphiBR = m_scalar * (angle_d_phi * dR_D + angle_n_phi * dR_N)
    dphiBZ = m_scalar * (angle_d_phi * dZ_D + angle_n_phi * dZ_N)
    dphiBphi = m_scalar * (m_scalar * (angle_d_phi2 * D + angle_n_phi2 * N) / R)
    dRBphi = m_scalar * (angle_d_phi * dR_D + angle_n_phi * dR_N) / R - m_scalar * (
        angle_d_phi * D + angle_n_phi * N
    ) / (R * R)
    dZBphi = m_scalar * (angle_d_phi * dZ_D + angle_n_phi * dZ_N) / R
    return (
        BR,
        BZ,
        Bphi,
        dRBR,
        dZBR,
        dphiBR,
        dRBZ,
        dZBZ,
        dphiBZ,
        dRBphi,
        dZBphi,
        dphiBphi,
    )


def _cylindrical_to_cartesian_B(
    BR: jax.Array, Bphi: jax.Array, BZ: jax.Array, cosphi: jax.Array, sinphi: jax.Array
) -> jax.Array:
    bx = BR * cosphi - Bphi * sinphi
    by = BR * sinphi + Bphi * cosphi
    return jnp.stack([bx, by, BZ], axis=-1)


def _cylindrical_to_cartesian_dB(
    R: jax.Array,
    BR: jax.Array,
    BZ: jax.Array,
    Bphi: jax.Array,
    dRBR: jax.Array,
    dZBR: jax.Array,
    dphiBR: jax.Array,
    dRBZ: jax.Array,
    dZBZ: jax.Array,
    dphiBZ: jax.Array,
    dRBphi: jax.Array,
    dZBphi: jax.Array,
    dphiBphi: jax.Array,
    cosphi: jax.Array,
    sinphi: jax.Array,
) -> jax.Array:
    """Map cylindrical-component gradients to a Cartesian (3, 3) tensor.

    Mirrors the dB(j, i, :, :) assembly in ``dommaschk.cpp`` /
    ``reiman.cpp`` exactly.
    """

    dB00 = (
        dRBR * cosphi * cosphi
        - (dphiBR - Bphi + dRBphi * R) * cosphi * sinphi / R
        + sinphi * sinphi * (dphiBphi + BR) / R
    )
    dB01 = (
        sinphi * cosphi * (dRBR * R - dphiBphi - BR) / R
        + sinphi * sinphi * (Bphi - dphiBR) / R
        + cosphi * cosphi * dRBphi
    )
    dB02 = dRBZ * cosphi - dphiBZ * sinphi / R
    dB10 = (
        sinphi * cosphi * (dRBR * R - dphiBphi - BR) / R
        + cosphi * cosphi * (dphiBR - Bphi) / R
        - sinphi * sinphi * dRBphi
    )
    dB11 = (
        dRBR * sinphi * sinphi
        + (dphiBR - Bphi + dRBphi * R) * cosphi * sinphi / R
        + cosphi * cosphi * (dphiBphi + BR) / R
    )
    dB12 = dRBZ * sinphi + dphiBZ * cosphi / R
    dB20 = dZBR * cosphi - dZBphi * sinphi
    dB21 = dZBR * sinphi + dZBphi * cosphi
    dB22 = dZBZ
    row0 = jnp.stack([dB00, dB01, dB02], axis=-1)
    row1 = jnp.stack([dB10, dB11, dB12], axis=-1)
    row2 = jnp.stack([dB20, dB21, dB22], axis=-1)
    return jnp.stack([row0, row1, row2], axis=-2)


@lru_cache(maxsize=None)
def _dommaschk_B_multimode_kernel(m_tuple: tuple[int, ...], n_tuple: tuple[int, ...]):
    """Compiled multi-mode Dommaschk Cartesian-B kernel.

    The ``(m, n)`` tuples are static metadata in the JIT cache key, so
    the per-mode Python ``for`` loop is unrolled at tracing time and
    the loop counter never crosses the host/device boundary.
    """

    def kernel(coeffs: jax.Array, points: jax.Array) -> jax.Array:
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        R = jnp.sqrt(x * x + y * y)
        phi = jnp.arctan2(y, x)
        cosphi = x / R
        sinphi = y / R
        per_mode: list[jax.Array] = []
        for j, (m, n) in enumerate(zip(m_tuple, n_tuple)):
            coeff_pair = coeffs[j]
            coeff1 = coeff_pair[0]
            coeff2 = coeff_pair[1]
            BR, BZ, Bphi = _dommaschk_single_mode_BR_BZ_Bphi(
                m, n, R, z, phi, coeff1, coeff2
            )
            per_mode.append(_cylindrical_to_cartesian_B(BR, Bphi, BZ, cosphi, sinphi))
        return jnp.stack(per_mode, axis=0)

    return jax.jit(kernel)


@lru_cache(maxsize=None)
def _dommaschk_dB_multimode_kernel(m_tuple: tuple[int, ...], n_tuple: tuple[int, ...]):
    """Compiled multi-mode Dommaschk Cartesian-dB kernel.

    See :func:`_dommaschk_B_multimode_kernel` for the static-metadata
    contract.
    """

    def kernel(coeffs: jax.Array, points: jax.Array) -> jax.Array:
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        R = jnp.sqrt(x * x + y * y)
        phi = jnp.arctan2(y, x)
        cosphi = x / R
        sinphi = y / R
        per_mode: list[jax.Array] = []
        for j, (m, n) in enumerate(zip(m_tuple, n_tuple)):
            coeff_pair = coeffs[j]
            coeff1 = coeff_pair[0]
            coeff2 = coeff_pair[1]
            (
                BR,
                BZ,
                Bphi,
                dRBR,
                dZBR,
                dphiBR,
                dRBZ,
                dZBZ,
                dphiBZ,
                dRBphi,
                dZBphi,
                dphiBphi,
            ) = _dommaschk_single_mode_dB_local(m, n, R, z, phi, coeff1, coeff2)
            per_mode.append(
                _cylindrical_to_cartesian_dB(
                    R,
                    BR,
                    BZ,
                    Bphi,
                    dRBR,
                    dZBR,
                    dphiBR,
                    dRBZ,
                    dZBZ,
                    dphiBZ,
                    dRBphi,
                    dZBphi,
                    dphiBphi,
                    cosphi,
                    sinphi,
                )
            )
        return jnp.stack(per_mode, axis=0)

    return jax.jit(kernel)


def _validate_dommaschk_spec(spec: DommaschkSpec) -> int:
    if not isinstance(spec, DommaschkSpec):
        raise TypeError("dommaschk kernel requires a DommaschkSpec")
    if len(spec.m) != len(spec.n):
        raise ValueError("DommaschkSpec.m and .n must have equal length")
    coeffs = jnp.asarray(spec.coeffs)
    if coeffs.ndim != 2 or coeffs.shape[1] != 2:
        raise ValueError("DommaschkSpec.coeffs must have shape [K, 2]")
    if coeffs.shape[0] != len(spec.m):
        raise ValueError("DommaschkSpec.coeffs length does not match mode-index length")
    return int(coeffs.shape[0])


def dommaschk_B(spec: DommaschkSpec, points: jax.Array) -> jax.Array:
    """Raw Dommaschk Cartesian magnetic field per (m, n) mode.

    Parameters
    ----------
    spec
        :class:`DommaschkSpec` carrying the ``(m, n)`` mode indices and
        per-mode coefficient pairs.
    points
        ``float64`` JAX array with shape ``[N, 3]`` of Cartesian
        evaluation points. ``R = sqrt(x^2 + y^2)`` must be positive.

    Returns
    -------
    jax.Array
        Shape ``[K, N, 3]`` Cartesian field, mirroring
        ``sopp.DommaschkB``. The ``ToroidalField(R0=1, B0=1)`` baseline
        is **not** included.
    """

    _validate_dommaschk_spec(spec)
    points_arr = jnp.asarray(points, dtype=jnp.float64)
    coeffs = jnp.asarray(spec.coeffs, dtype=jnp.float64)
    m_tuple = tuple(int(v) for v in spec.m)
    n_tuple = tuple(int(v) for v in spec.n)
    kernel = _dommaschk_B_multimode_kernel(m_tuple, n_tuple)
    return kernel(coeffs, points_arr)


def dommaschk_dB(spec: DommaschkSpec, points: jax.Array) -> jax.Array:
    """Raw Dommaschk Cartesian magnetic-field gradient per mode.

    Returns a shape ``[K, N, 3, 3]`` array with the convention
    ``dB[k, p, i, j] = d B_j(x_p) / d x_i`` for mode index ``k``,
    mirroring ``sopp.DommaschkdB``. The ``ToroidalField(1, 1)`` baseline
    is **not** included.
    """

    _validate_dommaschk_spec(spec)
    points_arr = jnp.asarray(points, dtype=jnp.float64)
    coeffs = jnp.asarray(spec.coeffs, dtype=jnp.float64)
    m_tuple = tuple(int(v) for v in spec.m)
    n_tuple = tuple(int(v) for v in spec.n)
    kernel = _dommaschk_dB_multimode_kernel(m_tuple, n_tuple)
    return kernel(coeffs, points_arr)


# ── Reiman field ──────────────────────────────────────────────────────


def _validate_reiman_spec(spec: ReimanSpec) -> int:
    if not isinstance(spec, ReimanSpec):
        raise TypeError("reiman kernel requires a ReimanSpec")
    eps = jnp.asarray(spec.epsilon)
    if eps.ndim != 1:
        raise ValueError("ReimanSpec.epsilon must be 1-D")
    if eps.shape[0] != len(spec.k_theta):
        raise ValueError("ReimanSpec.epsilon length does not match k_theta length")
    return int(eps.shape[0])


def _reiman_pure_B(
    iota0: jax.Array,
    iota1: jax.Array,
    k_theta_tuple: tuple[int, ...],
    epsilon: jax.Array,
    m0_symmetry: int,
    points: jax.Array,
) -> jax.Array:
    """Pure Cartesian Reiman ``B(x)``.

    ``k_theta`` is unrolled at trace time (Python ``for``). All numeric
    inputs (``iota0``, ``iota1``, ``epsilon``, ``points``) are JAX
    arrays. ``m0_symmetry`` is a Python integer (static).
    """

    R_axis = jnp.asarray(1.0, dtype=points.dtype)
    x = points[:, 0]
    y = points[:, 1]
    Zp = points[:, 2]
    RR = jnp.sqrt(x * x + y * y)
    cosphi = x / RR
    sinphi = y / RR
    varphi = jnp.arctan2(y, x)
    theta = jnp.arctan2(Zp, RR - R_axis)
    rmin = jnp.sqrt(jnp.square(RR - R_axis) + jnp.square(Zp))

    combo = iota0 + iota1 * rmin * rmin
    combo1 = jnp.zeros_like(RR)
    for ind, kth in enumerate(k_theta_tuple):
        kth_f = jnp.asarray(kth, dtype=points.dtype)
        m0_f = jnp.asarray(m0_symmetry, dtype=points.dtype)
        angle = kth_f * theta - m0_f * varphi
        # rmin ** (k_theta - 2) -- finite Python integer exponent.
        rpow = jnp.power(rmin, jnp.asarray(kth - 2, dtype=points.dtype))
        combo = combo - kth_f * epsilon[ind] * rpow * jnp.cos(angle)
        combo1 = combo1 + kth_f * epsilon[ind] * rpow * jnp.sin(angle)

    BR = ((RR - R_axis) / RR) * combo1 + (Zp / RR) * combo
    BZ = -((RR - R_axis) / RR) * combo + (Zp / RR) * combo1
    Bphi = -jnp.ones_like(RR)
    return _cylindrical_to_cartesian_B(BR, Bphi, BZ, cosphi, sinphi)


def _reiman_pure_dB(
    iota0: jax.Array,
    iota1: jax.Array,
    k_theta_tuple: tuple[int, ...],
    epsilon: jax.Array,
    m0_symmetry: int,
    points: jax.Array,
) -> jax.Array:
    """Pure Cartesian Reiman ``dB/dx`` tensor of shape ``[N, 3, 3]``."""

    R_axis = jnp.asarray(1.0, dtype=points.dtype)
    x = points[:, 0]
    y = points[:, 1]
    Zp = points[:, 2]
    RR = jnp.sqrt(x * x + y * y)
    cosphi = x / RR
    sinphi = y / RR
    varphi = jnp.arctan2(y, x)
    theta = jnp.arctan2(Zp, RR - R_axis)
    rmin = jnp.sqrt(jnp.square(RR - R_axis) + jnp.square(Zp))

    combo = iota0 + iota1 * rmin * rmin
    combo1 = jnp.zeros_like(RR)
    dcombodR = 2.0 * iota1 * (RR - R_axis)
    dcombodZ = 2.0 * iota1 * Zp
    dcombodphi = jnp.zeros_like(RR)
    dcombo1dR = jnp.zeros_like(RR)
    dcombo1dZ = jnp.zeros_like(RR)
    dcombo1dphi = jnp.zeros_like(RR)

    m0_f = jnp.asarray(m0_symmetry, dtype=points.dtype)
    for ind, kth in enumerate(k_theta_tuple):
        kth_f = jnp.asarray(kth, dtype=points.dtype)
        angle = kth_f * theta - m0_f * varphi
        cos_a = jnp.cos(angle)
        sin_a = jnp.sin(angle)
        rpow_m2 = jnp.power(rmin, jnp.asarray(kth - 2, dtype=points.dtype))
        rpow_m4 = jnp.power(rmin, jnp.asarray(kth - 4, dtype=points.dtype))
        kth_eps = kth_f * epsilon[ind]
        combo = combo - kth_eps * rpow_m2 * cos_a
        combo1 = combo1 + kth_eps * rpow_m2 * sin_a
        dcombodR = dcombodR - kth_f * rpow_m4 * epsilon[ind] * (
            kth_f * Zp * sin_a + (kth_f - 2.0) * (RR - R_axis) * cos_a
        )
        dcombodZ = dcombodZ + rpow_m4 * epsilon[ind] * kth_f * (
            kth_f * sin_a * (RR - R_axis) - (kth_f - 2.0) * Zp * cos_a
        )
        dcombodphi = dcombodphi - kth_eps * rpow_m2 * sin_a * m0_f
        dcombo1dR = dcombo1dR + kth_f * rpow_m4 * epsilon[ind] * (
            -kth_f * Zp * cos_a + (kth_f - 2.0) * sin_a * (RR - R_axis)
        )
        dcombo1dZ = dcombo1dZ + kth_f * rpow_m4 * epsilon[ind] * (
            kth_f * cos_a * (RR - R_axis) + (kth_f - 2.0) * sin_a * Zp
        )
        dcombo1dphi = dcombo1dphi - kth_eps * rpow_m2 * cos_a * m0_f

    BR = ((RR - R_axis) / RR) * combo1 + (Zp / RR) * combo
    BZ = -((RR - R_axis) / RR) * combo + (Zp / RR) * combo1
    Bphi = -jnp.ones_like(RR)

    inv_RR = 1.0 / RR
    inv_RR2 = inv_RR * inv_RR
    dRBR = (
        -Zp * inv_RR2 * combo
        + (Zp * inv_RR) * dcombodR
        + combo1 * R_axis * inv_RR2
        + dcombo1dR * (RR - R_axis) * inv_RR
    )
    dZBR = (
        inv_RR * combo + (Zp * inv_RR) * dcombodZ + dcombo1dZ * (RR - R_axis) * inv_RR
    )
    dphiBR = ((RR - R_axis) * inv_RR) * dcombo1dphi + (Zp * inv_RR) * dcombodphi
    dRBZ = (
        -R_axis * inv_RR2 * combo
        - ((RR - R_axis) * inv_RR) * dcombodR
        - combo1 * Zp * inv_RR2
        + dcombo1dR * Zp * inv_RR
    )
    dZBZ = (
        -((RR - R_axis) * inv_RR) * dcombodZ + combo1 * inv_RR + dcombo1dZ * Zp * inv_RR
    )
    dphiBZ = -((RR - R_axis) * inv_RR) * dcombodphi + (Zp * inv_RR) * dcombo1dphi
    dRBphi = jnp.zeros_like(RR)
    dZBphi = jnp.zeros_like(RR)
    dphiBphi = jnp.zeros_like(RR)

    return _cylindrical_to_cartesian_dB(
        RR,
        BR,
        BZ,
        Bphi,
        dRBR,
        dZBR,
        dphiBR,
        dRBZ,
        dZBZ,
        dphiBZ,
        dRBphi,
        dZBphi,
        dphiBphi,
        cosphi,
        sinphi,
    )


@lru_cache(maxsize=None)
def _reiman_B_kernel(k_theta_tuple: tuple[int, ...], m0_symmetry: int):
    """Compiled Reiman B kernel keyed on (k_theta, m0_symmetry)."""

    def kernel(
        iota0: jax.Array, iota1: jax.Array, epsilon: jax.Array, points: jax.Array
    ) -> jax.Array:
        return _reiman_pure_B(iota0, iota1, k_theta_tuple, epsilon, m0_symmetry, points)

    return jax.jit(kernel)


@lru_cache(maxsize=None)
def _reiman_dB_kernel(k_theta_tuple: tuple[int, ...], m0_symmetry: int):
    """Compiled Reiman dB kernel keyed on (k_theta, m0_symmetry)."""

    def kernel(
        iota0: jax.Array, iota1: jax.Array, epsilon: jax.Array, points: jax.Array
    ) -> jax.Array:
        return _reiman_pure_dB(
            iota0, iota1, k_theta_tuple, epsilon, m0_symmetry, points
        )

    return jax.jit(kernel)


def reiman_B(spec: ReimanSpec, points: jax.Array) -> jax.Array:
    """Cartesian Reiman island-model B(x) of shape ``[N, 3]``."""

    _validate_reiman_spec(spec)
    points_arr = jnp.asarray(points, dtype=jnp.float64)
    iota0 = jnp.asarray(spec.iota0, dtype=jnp.float64)
    iota1 = jnp.asarray(spec.iota1, dtype=jnp.float64)
    epsilon = jnp.asarray(spec.epsilon, dtype=jnp.float64)
    kernel = _reiman_B_kernel(
        tuple(int(k) for k in spec.k_theta), int(spec.m0_symmetry)
    )
    return kernel(iota0, iota1, epsilon, points_arr)


def reiman_dB(spec: ReimanSpec, points: jax.Array) -> jax.Array:
    """Cartesian Reiman island-model dB(x) of shape ``[N, 3, 3]``.

    Index convention matches ``sopp.ReimandB``: ``dB[p, i, j]`` is the
    partial derivative of the ``j``-th Cartesian B-component with
    respect to the ``i``-th Cartesian coordinate. (The C++ source uses
    ``dB(i, j, k)`` with ``j`` the derivative axis and ``k`` the field
    axis.)
    """

    _validate_reiman_spec(spec)
    points_arr = jnp.asarray(points, dtype=jnp.float64)
    iota0 = jnp.asarray(spec.iota0, dtype=jnp.float64)
    iota1 = jnp.asarray(spec.iota1, dtype=jnp.float64)
    epsilon = jnp.asarray(spec.epsilon, dtype=jnp.float64)
    kernel = _reiman_dB_kernel(
        tuple(int(k) for k in spec.k_theta), int(spec.m0_symmetry)
    )
    return kernel(iota0, iota1, epsilon, points_arr)


# Static-analysis hooks: keep ``np`` imported for downstream consumers
# that may wish to construct specs from numpy arrays.
_ = np
