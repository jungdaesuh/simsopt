# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

"""JAX-backed public bootstrap-current entrypoints."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from simsopt._core.optimizable import Optimizable
from simsopt_jax.core.mhd_bootstrap import compute_trapped_fraction_jax
from simsopt_jax.core.redl_current import RedlDetailsJAX, j_dot_B_Redl_jax_from_arrays
from simsopt.mhd.profiles import Profile, ProfilePolynomial
from .profiles import ProfilePolynomialJAX

__all__ = [
    "RedlBootstrapJAX",
    "RedlDetailsJAX",
    "compute_trapped_fraction_jax",
    "j_dot_B_Redl_jax",
    "j_dot_B_Redl_jax_from_arrays",
]


def j_dot_B_Redl_jax(
    ne,
    Te,
    Ti,
    Zeff,
    helicity_n=None,
    s=None,
    G=None,
    R=None,
    iota=None,
    epsilon=None,
    f_t=None,
    psi_edge=None,
    nfp=None,
):
    """Evaluate Redl ``<J dot B>`` through the JAX array kernel.

    Inputs must stay inside the Redl formula's physical domain; the JAX path
    does not smooth the ``Zeff = 1`` or ``iota = nfp * helicity_n`` singular
    boundaries for differentiation.
    """
    if Zeff is None:
        Zeff = ProfilePolynomial(1.0)
    if not isinstance(Zeff, Profile):
        Zeff = ProfilePolynomial([Zeff])

    return j_dot_B_Redl_jax_from_arrays(
        ne(s),
        Te(s),
        Ti(s),
        Zeff(s),
        ne.dfds(s),
        Te.dfds(s),
        Ti.dfds(s),
        helicity_n=helicity_n,
        s=s,
        G=G,
        R=R,
        iota=iota,
        epsilon=epsilon,
        f_t=f_t,
        psi_edge=psi_edge,
        nfp=nfp,
    )


def _profile_from_dofs(profile, dofs, s):
    return profile.f_from_dofs(dofs, s), profile.dfds_from_dofs(dofs, s)


def _profile_derivative_helper(profile):
    if isinstance(profile, ProfilePolynomialJAX):
        return profile
    if isinstance(profile, ProfilePolynomial):
        return ProfilePolynomialJAX(profile.local_full_x)
    return profile


def _constant_profile(value):
    return ProfilePolynomial([value])


class RedlBootstrapJAX(Optimizable):
    """JAX-backed adapter for evaluating Redl ``<J dot B>`` profiles.

    Derivative helpers are supported for profile coefficients away from the
    Redl formula's physical singularities.
    """

    def __init__(self, geom, ne, Te, Ti, Zeff, helicity_n):
        if Zeff is None:
            Zeff = _constant_profile(1.0)
        if not isinstance(Zeff, Profile):
            Zeff = _constant_profile(Zeff)
        self.geom = geom
        self.ne = ne
        self.Te = Te
        self.Ti = Ti
        self.Zeff = Zeff
        self._ne_derivative_profile = _profile_derivative_helper(ne)
        self._Te_derivative_profile = _profile_derivative_helper(Te)
        self._Ti_derivative_profile = _profile_derivative_helper(Ti)
        self._Zeff_derivative_profile = _profile_derivative_helper(Zeff)
        self.helicity_n = helicity_n
        super().__init__(depends_on=[geom, ne, Te, Ti, Zeff])

    def _geom_data(self):
        return self.geom()

    def _array_kwargs(self, geom_data):
        return {
            "s": geom_data.surfaces,
            "G": geom_data.G,
            "R": geom_data.R,
            "iota": geom_data.iota,
            "epsilon": geom_data.epsilon,
            "f_t": geom_data.f_t,
            "psi_edge": geom_data.psi_edge,
            "nfp": geom_data.nfp,
        }

    def J(self):
        """Return the Redl ``<J dot B>`` array on the geometry surfaces."""
        jdotB, _ = self.details()
        return jdotB

    def details(self):
        """Return ``(<J dot B>, RedlDetailsJAX)`` for the current state."""
        geom_data = self._geom_data()
        return j_dot_B_Redl_jax(
            self.ne,
            self.Te,
            self.Ti,
            self.Zeff,
            self.helicity_n,
            s=geom_data.surfaces,
            G=geom_data.G,
            R=geom_data.R,
            iota=geom_data.iota,
            epsilon=geom_data.epsilon,
            f_t=geom_data.f_t,
            psi_edge=geom_data.psi_edge,
            nfp=geom_data.nfp,
        )

    def _jdotB_from_profile_dofs(self, profile_name: str, dofs, geom_data):
        s = jnp.asarray(geom_data.surfaces, dtype=jnp.float64)
        ne_profile = self._ne_derivative_profile
        Te_profile = self._Te_derivative_profile
        Ti_profile = self._Ti_derivative_profile
        Zeff_profile = self._Zeff_derivative_profile
        ne_dofs = dofs if profile_name == "ne" else jnp.asarray(self.ne.local_full_x)
        Te_dofs = dofs if profile_name == "Te" else jnp.asarray(self.Te.local_full_x)
        Ti_dofs = dofs if profile_name == "Ti" else jnp.asarray(self.Ti.local_full_x)
        Zeff_dofs = (
            dofs if profile_name == "Zeff" else jnp.asarray(self.Zeff.local_full_x)
        )
        ne_s, d_ne_d_s = _profile_from_dofs(ne_profile, ne_dofs, s)
        Te_s, d_Te_d_s = _profile_from_dofs(Te_profile, Te_dofs, s)
        Ti_s, d_Ti_d_s = _profile_from_dofs(Ti_profile, Ti_dofs, s)
        Zeff_s, _ = _profile_from_dofs(Zeff_profile, Zeff_dofs, s)
        jdotB, _ = j_dot_B_Redl_jax_from_arrays(
            ne_s,
            Te_s,
            Ti_s,
            Zeff_s,
            d_ne_d_s,
            d_Te_d_s,
            d_Ti_d_s,
            helicity_n=self.helicity_n,
            **self._array_kwargs(geom_data),
        )
        return jdotB

    def _dJ_by_profile_dofs(self, profile_name: str, profile):
        geom_data = self._geom_data()
        dofs = jnp.asarray(profile.local_full_x, dtype=jnp.float64)
        return jax.jacfwd(
            lambda profile_dofs: self._jdotB_from_profile_dofs(
                profile_name, profile_dofs, geom_data
            )
        )(dofs)

    def dJ_by_dne_dofs(self):
        """Return ``d<J dot B>/d(ne dofs)`` for polynomial profiles."""
        return self._dJ_by_profile_dofs("ne", self.ne)

    def dJ_by_dTe_dofs(self):
        """Return ``d<J dot B>/d(Te dofs)`` for polynomial profiles."""
        return self._dJ_by_profile_dofs("Te", self.Te)

    def dJ_by_dTi_dofs(self):
        """Return ``d<J dot B>/d(Ti dofs)`` for polynomial profiles."""
        return self._dJ_by_profile_dofs("Ti", self.Ti)

    def dJ_by_dZeff_dofs(self):
        """Return ``d<J dot B>/d(Zeff dofs)`` for polynomial profiles."""
        return self._dJ_by_profile_dofs("Zeff", self.Zeff)
