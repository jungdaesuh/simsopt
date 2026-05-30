"""Pure JAX kernels for the Redl bootstrap-current formula."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from simsopt._constants import ELEMENTARY_CHARGE

from ._math_utils import as_jax_float64

__all__ = [
    "RedlDetailsJAX",
    "j_dot_B_Redl_jax_from_arrays",
]


@dataclass(frozen=True)
class RedlDetailsJAX:
    """JIT-friendly payload matching the CPU ``j_dot_B_Redl`` details fields."""

    s: jax.Array
    ne_s: jax.Array
    ni_s: jax.Array
    Zeff_s: jax.Array
    Te_s: jax.Array
    Ti_s: jax.Array
    d_ne_d_s: jax.Array
    d_Te_d_s: jax.Array
    d_Ti_d_s: jax.Array
    ln_Lambda_e: jax.Array
    ln_Lambda_ii: jax.Array
    nu_e_star: jax.Array
    nu_i_star: jax.Array
    X31: jax.Array
    X32e: jax.Array
    X32ei: jax.Array
    F32ee: jax.Array
    F32ei: jax.Array
    L31: jax.Array
    L32: jax.Array
    L34: jax.Array
    alpha0: jax.Array
    alpha: jax.Array
    dnds_term: jax.Array
    dTeds_term: jax.Array
    dTids_term: jax.Array
    jdotB: jax.Array


_REDL_DETAIL_FIELDS = [
    "s",
    "ne_s",
    "ni_s",
    "Zeff_s",
    "Te_s",
    "Ti_s",
    "d_ne_d_s",
    "d_Te_d_s",
    "d_Ti_d_s",
    "ln_Lambda_e",
    "ln_Lambda_ii",
    "nu_e_star",
    "nu_i_star",
    "X31",
    "X32e",
    "X32ei",
    "F32ee",
    "F32ei",
    "L31",
    "L32",
    "L34",
    "alpha0",
    "alpha",
    "dnds_term",
    "dTeds_term",
    "dTids_term",
    "jdotB",
]


jax.tree_util.register_dataclass(
    RedlDetailsJAX,
    data_fields=_REDL_DETAIL_FIELDS,
    meta_fields=[],
)


def j_dot_B_Redl_jax_from_arrays(
    ne_s,
    Te_s,
    Ti_s,
    Zeff_s,
    d_ne_d_s,
    d_Te_d_s,
    d_Ti_d_s,
    *,
    helicity_n,
    s,
    G,
    R,
    iota,
    epsilon,
    f_t,
    psi_edge,
    nfp,
) -> tuple[jax.Array, RedlDetailsJAX]:
    """Evaluate Redl ``<J dot B>`` from profile values on a shared ``s`` grid.

    This is a faithful array port of the CPU Redl formula. It assumes the
    caller supplies points in the physical domain of that formula: positive
    density and temperature profiles, ``Zeff >= 1``, positive ``epsilon``, and
    ``iota != nfp * helicity_n``. The kernel intentionally does not regularize
    the ``Zeff = 1`` or ``iota = nfp * helicity_n`` boundary for AD.
    """
    s_jax = as_jax_float64(s)
    ne_jax = as_jax_float64(ne_s)
    Te_jax = as_jax_float64(Te_s)
    Ti_jax = as_jax_float64(Ti_s)
    Zeff_jax = as_jax_float64(Zeff_s)
    d_ne_jax = as_jax_float64(d_ne_d_s)
    d_Te_jax = as_jax_float64(d_Te_d_s)
    d_Ti_jax = as_jax_float64(d_Ti_d_s)
    G_jax = as_jax_float64(G)
    R_jax = as_jax_float64(R)
    iota_jax = as_jax_float64(iota)
    epsilon_jax = as_jax_float64(epsilon)
    f_t_jax = as_jax_float64(f_t)
    psi_edge_jax = as_jax_float64(psi_edge)
    helicity_N = as_jax_float64(nfp) * as_jax_float64(int(helicity_n))

    ni_s = ne_jax / Zeff_jax
    pe_s = ne_jax * Te_jax
    pi_s = ni_s * Ti_jax

    ln_Lambda_e = 31.3 - jnp.log(jnp.sqrt(ne_jax) / Te_jax)
    ln_Lambda_ii = 30.0 - jnp.log(Zeff_jax**3 * jnp.sqrt(ni_s) / (Ti_jax**1.5))

    geometry_factor = jnp.abs(R_jax / (iota_jax - helicity_N))
    nu_e = (
        geometry_factor
        * 6.921e-18
        * ne_jax
        * Zeff_jax
        * ln_Lambda_e
        / (Te_jax * Te_jax * epsilon_jax**1.5)
    )
    nu_i = (
        geometry_factor
        * 4.90e-18
        * ni_s
        * Zeff_jax**4
        * ln_Lambda_ii
        / (Ti_jax * Ti_jax * epsilon_jax**1.5)
    )

    sqrt_nu_e = jnp.sqrt(nu_e)
    sqrt_zeff_minus_one = jnp.sqrt(Zeff_jax - 1.0)
    X31 = f_t_jax / (
        1.0
        + 0.67 * (1.0 - 0.7 * f_t_jax) * sqrt_nu_e / (0.56 + 0.44 * Zeff_jax)
        + (0.52 + 0.086 * sqrt_nu_e)
        * (1.0 + 0.87 * f_t_jax)
        * nu_e
        / (1.0 + 1.13 * sqrt_zeff_minus_one)
    )

    Zfac = Zeff_jax**1.2 - 0.71
    L31 = (
        (1.0 + 0.15 / Zfac) * X31
        - 0.22 / Zfac * X31**2
        + 0.01 / Zfac * X31**3
        + 0.06 / Zfac * X31**4
    )

    X32e = f_t_jax / (
        1.0
        + 0.23 * (1.0 - 0.96 * f_t_jax) * sqrt_nu_e / jnp.sqrt(Zeff_jax)
        + 0.13
        * (1.0 - 0.38 * f_t_jax)
        * nu_e
        / (Zeff_jax * Zeff_jax)
        * (
            jnp.sqrt(1.0 + 2.0 * sqrt_zeff_minus_one)
            + f_t_jax
            * f_t_jax
            * jnp.sqrt((0.075 + 0.25 * (Zeff_jax - 1.0) ** 2) * nu_e)
        )
    )

    F32ee = (
        (0.1 + 0.6 * Zeff_jax)
        * (X32e - X32e**4)
        / (Zeff_jax * (0.77 + 0.63 * (1.0 + (Zeff_jax - 1.0) ** 1.1)))
        + 0.7 / (1.0 + 0.2 * Zeff_jax) * (X32e**2 - X32e**4 - 1.2 * (X32e**3 - X32e**4))
        + 1.3 / (1.0 + 0.5 * Zeff_jax) * X32e**4
    )

    X32ei = f_t_jax / (
        1.0
        + 0.87
        * (1.0 + 0.39 * f_t_jax)
        * sqrt_nu_e
        / (1.0 + 2.95 * (Zeff_jax - 1.0) ** 2)
        + 1.53 * (1.0 - 0.37 * f_t_jax) * nu_e * (2.0 + 0.375 * (Zeff_jax - 1.0))
    )

    F32ei = (
        -(0.4 + 1.93 * Zeff_jax)
        / (Zeff_jax * (0.8 + 0.6 * Zeff_jax))
        * (X32ei - X32ei**4)
        + 5.5
        / (1.5 + 2.0 * Zeff_jax)
        * (X32ei**2 - X32ei**4 - 0.8 * (X32ei**3 - X32ei**4))
        - 1.3 / (1.0 + 0.5 * Zeff_jax) * X32ei**4
    )

    L32 = F32ei + F32ee
    L34 = L31

    alpha0 = (
        -(0.62 + 0.055 * (Zeff_jax - 1.0))
        * (1.0 - f_t_jax)
        / (
            (0.53 + 0.17 * (Zeff_jax - 1.0))
            * (
                1.0
                - (0.31 - 0.065 * (Zeff_jax - 1.0)) * f_t_jax
                - 0.25 * f_t_jax * f_t_jax
            )
        )
    )
    alpha = (
        (alpha0 + 0.7 * Zeff_jax * jnp.sqrt(f_t_jax * nu_i))
        / (1.0 + 0.18 * jnp.sqrt(nu_i))
        - 0.002 * nu_i * nu_i * f_t_jax**6
    ) / (1.0 + 0.004 * nu_i * nu_i * f_t_jax**6)

    denominator = psi_edge_jax * (iota_jax - helicity_N)
    charge = as_jax_float64(ELEMENTARY_CHARGE)
    dnds_term = (
        -G_jax
        * charge
        * (ne_jax * Te_jax + ni_s * Ti_jax)
        * L31
        * (d_ne_jax / ne_jax)
        / denominator
    )
    dTeds_term = (
        -G_jax * charge * pe_s * (L31 + L32) * (d_Te_jax / Te_jax) / denominator
    )
    dTids_term = (
        -G_jax * charge * pi_s * (L31 + L34 * alpha) * (d_Ti_jax / Ti_jax) / denominator
    )
    jdotB = dnds_term + dTeds_term + dTids_term

    details = RedlDetailsJAX(
        s=s_jax,
        ne_s=ne_jax,
        ni_s=ni_s,
        Zeff_s=Zeff_jax,
        Te_s=Te_jax,
        Ti_s=Ti_jax,
        d_ne_d_s=d_ne_jax,
        d_Te_d_s=d_Te_jax,
        d_Ti_d_s=d_Ti_jax,
        ln_Lambda_e=ln_Lambda_e,
        ln_Lambda_ii=ln_Lambda_ii,
        nu_e_star=nu_e,
        nu_i_star=nu_i,
        X31=X31,
        X32e=X32e,
        X32ei=X32ei,
        F32ee=F32ee,
        F32ei=F32ei,
        L31=L31,
        L32=L32,
        L34=L34,
        alpha0=alpha0,
        alpha=alpha,
        dnds_term=dnds_term,
        dTeds_term=dTeds_term,
        dTids_term=dTids_term,
        jdotB=jdotB,
    )
    return jdotB, details
