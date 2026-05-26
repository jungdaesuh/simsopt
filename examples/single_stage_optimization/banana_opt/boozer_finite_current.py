r"""
Finite-enclosed-current Boozer-surface support, kept out of ``src/`` so the
upstream simsopt code stays mergeable.

Physics
-------
The Boozer-surface residual on a flux surface is

    r = alpha * B  -  |B|^2 * (x_phi + iota * x_theta)

In vacuum this is the upstream form ``alpha = G``. With a finite enclosed
plasma current ``I`` (Boozer's other surface invariant), the correct form is

    alpha = G + iota * I.

The correction is **linear and separable**: passing
``G_effective = G + iota * I`` to the existing (vacuum) residual implementation
reproduces the correct ``r``.

For first and second derivatives, we treat ``alpha`` as a function of
``(iota, G; I)`` with ``I`` as an external parameter.  The upstream code
returns ``d/diota`` and ``d/dG`` while *holding alpha fixed*; with finite
``I`` the "true" parameters are ``(iota, G)`` with the chain-rule contribution
``d alpha / d iota = I``.  Concretely, if ``g_alpha`` is the gradient row in
the (..., iota_alpha, G_alpha) basis returned by the vacuum kernel, the
gradient row in the (..., iota_true, G_true) basis is

    g_true[iota] = g_alpha[iota] + I * g_alpha[G]
    g_true[G]    = g_alpha[G]
    g_true[surface_dofs] unchanged.

This is exactly ``T^T g`` for the (N x N) elementary transform

    T = I_N
    T[-1, -2] = I              # (last_row=G, second_to_last_col=iota)

and similarly ``H_true = T^T H_alpha T`` for Hessians.  ``T`` is identity plus
a single off-diagonal scalar, so these transforms reduce to direct rank-one
slice updates and we never materialize ``T``.

Design rationale
----------------
* All the I-aware logic lives here, in ``examples/``.  ``src/simsopt`` stays a
  clean copy of upstream simsopt so the fork can be rebased / cleanly merged.
* No private helpers from ``src/`` are imported.  The wrapper depends only on
  the public upstream API: ``boozer_surface_residual``,
  ``boozer_surface_residual_dB``, the ``BoozerSurface`` class, and the
  ``simsoptpp`` upstream kernel signatures.
* The four ``BoozerSurface`` numerical methods that *consume* ``alpha`` are
  overridden in :class:`BoozerSurfaceFiniteI` rather than monkey-patching: the
  upstream methods are transcribed closely, with the substitution
  ``G -> G + iota * self.I`` at the residual call site and the ``T^T g`` /
  ``T^T H T`` transform applied to gradients/Hessians where the upstream
  kernel returns derivatives in the alpha-fixed basis.  The exact residual
  Newton path additionally backtracks residual-worsening finite-I steps while
  keeping the upstream full Newton step as the first trial.
"""

from functools import partial

import numpy as np
from scipy.linalg import lu

import simsoptpp as sopp

from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.surfaceobjectives import (
    boozer_surface_dexactresidual_dcoils_dcurrents_vjp,
    boozer_surface_dlsqgrad_dcoils_vjp,
    boozer_surface_residual,
    boozer_surface_residual_dB,
)
from simsopt.objectives.utilities import forward_solve

__all__ = [
    "boozer_surface_residual_finite_I",
    "boozer_surface_residual_dB_finite_I",
    "BoozerSurfaceFiniteI",
    "derive_signed_G_from_field",
]


_EXACT_NEWTON_BACKTRACKING_STEPS = tuple(0.5 ** i for i in range(8))

# SIMSOPT writes the Boozer residual in normalized angles (theta, phi in
# [0, 1]), so the ``G`` invariant on a flux surface equals
# ``mu0 * sum_signed(I_TF_linked)`` for a vacuum field — the factor of 2*pi
# from the angle change of variables cancels the 1/(2*pi) in the
# Biot-Savart prefactor. The upstream ``boozer_surface_residual`` default
# uses the *unsigned* sum (``np.abs``), which silently flips the sign of
# the Newton seed when the linked TF current is negative (e.g., CW HBT
# TF at -80 kA). This module provides the signed SSOT seed.
_MU0_TM_PER_A = 4.0e-7 * np.pi


def _require_explicit_G(G):
    if G is None:
        raise ValueError("finite-current Boozer paths require an explicit signed G")
    return G


def _signed_G_from_tf_currents(tf_coils) -> float:
    """Internal SSOT formula: ``G = mu0 * sum_signed(I_TF)``.

    Kept private so the public ``derive_signed_G_from_field`` is the single
    sanctioned entry point that callers declare a field dependency through;
    the legacy ``compute_tf_G0(tf_coils)`` helper in
    ``stage2_single_stage_handoff`` delegates here for backward-compatible
    sites that only carry the TF coil list.

    Duplicate coils in ``tf_coils`` are rejected: the toroidal-current
    linkage on the Boozer surface counts each *physical* TF coil exactly
    once, so a repeated coil reference would silently double-count its
    current in ``G`` and seed Newton with a magnitude inflated by the
    duplication factor.
    """
    if not tf_coils:
        raise ValueError(
            "Signed Boozer G derivation requires a non-empty TF coil bundle; "
            "vacuum-only fields with no TF coils have no toroidal-current "
            "linkage and cannot seed the Newton solve."
        )
    if len({id(c) for c in tf_coils}) != len(tf_coils):
        raise ValueError(
            "derive_signed_G_from_field: tf_coils contains duplicate coils"
        )
    signed_current_sum = float(
        sum(coil.current.get_value() for coil in tf_coils)
    )
    return _MU0_TM_PER_A * signed_current_sum


def derive_signed_G_from_field(biotsavart, *, tf_coils) -> float:
    """Return the signed Boozer ``G`` seed for a TF bundle in a BS field.

    Computes ``G = mu0 * sum_signed(I_TF)`` in SIMSOPT's normalized-angle
    convention (same magnitude as the upstream sign-blind default, but
    signed). ``tf_coils`` is the explicit TF subset of ``biotsavart.coils``
    -- the function intentionally does *not* try to discover the TF subset
    from the field, because the banana / proxy / VF coils share a
    ``BiotSavart`` instance and only the TF bundle contributes to the
    surface's poloidal-current linkage. ``biotsavart`` is accepted so that
    callers declare the field dependency explicitly and so the helper can
    verify that the TF subset really belongs to the field that will feed
    the Boozer residual.

    Proxy / VF currents intentionally do not enter ``G``: in the
    finite-enclosed-current Boozer wrapper the proxy plasma current enters
    via the separate ``I`` invariant (``alpha = G + iota * I``); folding it
    into ``G`` here would double-count its effect once the proxy field is
    already part of ``biotsavart``.
    """
    if biotsavart is None:
        raise ValueError("derive_signed_G_from_field requires a BiotSavart field.")
    field_coil_ids = {id(coil) for coil in biotsavart.coils}
    for coil in tf_coils:
        if id(coil) not in field_coil_ids:
            raise ValueError(
                "TF coil is not part of the supplied BiotSavart field; signed "
                "G derivation requires the TF subset to be drawn from the same "
                "field that feeds the Boozer residual."
            )
    return _signed_G_from_tf_currents(tf_coils)


def _to_explicit_current_basis(I_value, tensor):
    """Apply ``T^T`` along the trailing basis axis: rank-one update
    ``out[..., -2] += I_value * tensor[..., -1]``.

    ``T = I_N + I_value * e_{-1} e_{-2}^T``.  ``T^T @ x`` (or, equivalently,
    ``x @ T`` when the basis is on the trailing axis) is identity except for
    a single shifted slice; we apply the rank-one update directly instead of
    materializing ``T``.  Supports any shape with the basis on the last axis
    (1-D vector, 2-D Jacobian rows, 3-D / 4-D mixed-derivative tensors).
    """
    if I_value == 0.0:
        return tensor
    out = tensor.copy()
    out[..., -2] += I_value * tensor[..., -1]
    return out


def _to_explicit_current_basis_hessian(I_value, hessian):
    """Apply ``T^T H T`` along the trailing two axes: column update then row
    update.  Supports a 2-D Hessian or a 3-D residual-stack ``(M, N, N)``.
    """
    if I_value == 0.0:
        return hessian
    out = hessian.copy()
    out[..., -2] += I_value * hessian[..., -1]
    out[..., -2, :] += I_value * out[..., -1, :]
    return out


def _T_apply_to_lm(I_value, lm):
    """Apply ``T`` (the forward transform) to a Lagrange-multiplier vector:
    ``out[-1] += I_value * lm[-2]``.

    Used by the lsq adjoint via the identity
    ``vjp(lm . J_new^T . r) = vjp((T @ lm) . J_upstream^T . r)``.
    """
    if I_value == 0.0:
        return lm
    out = lm.copy()
    out[-1] += I_value * lm[-2]
    return out


def boozer_surface_residual_finite_I(
    surface,
    iota,
    G,
    biotsavart,
    derivatives=0,
    weight_inv_modB=False,
    I=0.0,
):
    """Finite-current ``boozer_surface_residual`` variant.

    The finite-current path requires an explicit signed ``G``.  When
    ``derivatives >= 1`` the returned Jacobian is in the
    (surface_dofs, iota, [G]) basis with ``I`` as an external parameter; this
    differs from the upstream alpha-fixed basis by the elementary basis
    transform implemented in this module.
    """

    G = _require_explicit_G(G)
    G_effective = G + iota * I

    # Always pass G_effective (never None) so the upstream kernel populates
    # the alpha-derivative column the basis transform needs.
    boozer = boozer_surface_residual(
        surface,
        iota,
        G_effective,
        biotsavart,
        derivatives=derivatives,
        weight_inv_modB=weight_inv_modB,
    )

    if derivatives == 0:
        return boozer

    if derivatives == 1:
        r, J = boozer
        J_new = _to_explicit_current_basis(I, J)
        return r, J_new

    r, J, H = boozer
    J_new = _to_explicit_current_basis(I, J)
    H_new = _to_explicit_current_basis_hessian(I, H)
    return r, J_new, H_new


def boozer_surface_residual_dB_finite_I(
    surface,
    iota,
    G,
    biotsavart,
    derivatives=0,
    weight_inv_modB=False,
    I=0.0,
    include_mixed_derivatives=True,
):
    """Finite-current ``boozer_surface_residual_dB`` variant.

    Requires an explicit signed ``G``.  Only the ``dr/ds`` (Jacobian) output
    needs basis correction --- ``dr/dB`` and the higher-order mixed derivatives
    are partials with respect to the field components alone, which depend on
    ``alpha`` only.  Substituting ``alpha = G + iota * I`` upstream therefore
    captures those terms exactly.
    """

    G = _require_explicit_G(G)
    G_effective = G + iota * I

    out = boozer_surface_residual_dB(
        surface,
        iota,
        G_effective,
        biotsavart,
        derivatives=derivatives,
        weight_inv_modB=weight_inv_modB,
        include_mixed_derivatives=include_mixed_derivatives,
    )

    if derivatives == 0:
        return out

    if include_mixed_derivatives:
        rtil, drtil_dB, J, d2_dsdB, d2_dsdgradB = out
        J_new = _to_explicit_current_basis(I, J)
        d2_dsdB_new = _to_explicit_current_basis(I, d2_dsdB)
        d2_dsdgradB_new = _to_explicit_current_basis(I, d2_dsdgradB)
        return rtil, drtil_dB, J_new, d2_dsdB_new, d2_dsdgradB_new

    rtil, drtil_dB, J = out
    J_new = _to_explicit_current_basis(I, J)
    return rtil, drtil_dB, J_new


def _exact_vjp_finite_I(I_value, lm, booz_surf, iota, G):
    """``boozer_surface_dexactresidual_dcoils_dcurrents_vjp`` with the I
    substitution ``G -> G + iota * I`` applied to the upstream alpha-only vjp.
    """
    G_effective = G + iota * I_value
    return boozer_surface_dexactresidual_dcoils_dcurrents_vjp(
        lm, booz_surf, iota, G_effective
    )


def _lsqgrad_vjp_finite_I(I_value, lm, booz_surf, iota, G, weight_inv_modB=True):
    """``boozer_surface_dlsqgrad_dcoils_vjp`` with the I substitution applied.

    The upstream lsq adjoint computes ``vjp(lm . J_upstream^T . residual)``,
    where ``J_upstream`` is the Jacobian in the alpha-only basis.  We want
    ``vjp(lm . J_new^T . residual)`` where ``J_new = J_upstream @ T``.
    Substituting: ``lm @ J_new^T @ r = lm @ T^T @ J_upstream^T @ r =
    (T @ lm) @ J_upstream^T @ r``.  So we pass ``T @ lm`` to the upstream vjp
    plus ``G_effective = G + iota * I`` for the residual.
    """
    G_effective = G + iota * I_value
    lm_upstream = _T_apply_to_lm(I_value, lm)
    return boozer_surface_dlsqgrad_dcoils_vjp(
        lm_upstream, booz_surf, iota, G_effective, weight_inv_modB=weight_inv_modB
    )


class BoozerSurfaceFiniteI(BoozerSurface):
    """A :class:`BoozerSurface` that solves the finite-enclosed-current
    Boozer-residual equations.

    The constructor mirrors :class:`BoozerSurface`'s but adds an ``I``
    keyword (the enclosed toroidal plasma current).  The four
    ``BoozerSurface`` methods that build alpha and apply derivatives
    are overridden so that ``alpha = G + iota * self.I`` is used and the
    derivative outputs are reported in the (..., iota, G; I-external) basis.
    """

    def __init__(
        self,
        biotsavart,
        surface,
        label,
        targetlabel,
        constraint_weight=None,
        options=None,
        I=0.0,
    ):
        super().__init__(
            biotsavart,
            surface,
            label,
            targetlabel,
            constraint_weight=constraint_weight,
            options=options,
        )
        self.I = float(I)

    def _annotate_current(self, payload):
        """Attach ``self.I`` to a result dict and, for LS-type results, swap
        the upstream alpha-only adjoint vjp for the I-aware variant.  The
        exact-Newton path installs the I-aware vjp directly inside
        :meth:`solve_residual_equation_exactly_newton`; idempotent on re-runs.
        """
        if payload is None:
            return None
        payload["I"] = self.I
        if payload.get("type") == "ls":
            weight_inv_modB = payload.get("weight_inv_modB", True)
            payload["vjp"] = partial(
                _lsqgrad_vjp_finite_I,
                self.I,
                weight_inv_modB=weight_inv_modB,
            )
        return payload

    def run_code(self, iota, G):
        _require_explicit_G(G)
        result = super().run_code(iota, G=G)
        return self._annotate_current_result(result)

    def _annotate_current_result(self, result):
        self._annotate_current(self.res)
        return self._annotate_current(result)

    def minimize_boozer_penalty_constraints_LBFGS(
        self,
        tol=1e-3,
        maxiter=1000,
        constraint_weight=1.0,
        iota=0.0,
        *,
        G,
        vectorize=True,
        limited_memory=True,
        weight_inv_modB=True,
        verbose=False,
    ):
        G = _require_explicit_G(G)
        result = super().minimize_boozer_penalty_constraints_LBFGS(
            tol=tol,
            maxiter=maxiter,
            constraint_weight=constraint_weight,
            iota=iota,
            G=G,
            vectorize=vectorize,
            limited_memory=limited_memory,
            weight_inv_modB=weight_inv_modB,
            verbose=verbose,
        )
        return self._annotate_current_result(result)

    def minimize_boozer_penalty_constraints_newton(
        self,
        tol=1e-12,
        maxiter=10,
        constraint_weight=1.0,
        iota=0.0,
        *,
        G,
        stab=0.0,
        vectorize=True,
        weight_inv_modB=True,
        verbose=False,
    ):
        G = _require_explicit_G(G)
        result = super().minimize_boozer_penalty_constraints_newton(
            tol=tol,
            maxiter=maxiter,
            constraint_weight=constraint_weight,
            iota=iota,
            G=G,
            stab=stab,
            vectorize=vectorize,
            weight_inv_modB=weight_inv_modB,
            verbose=verbose,
        )
        return self._annotate_current_result(result)

    def minimize_boozer_penalty_constraints_ls(
        self,
        tol=1e-12,
        maxiter=10,
        constraint_weight=1.0,
        iota=0.0,
        *,
        G,
        method="lm",
    ):
        G = _require_explicit_G(G)
        result = super().minimize_boozer_penalty_constraints_ls(
            tol=tol,
            maxiter=maxiter,
            constraint_weight=constraint_weight,
            iota=iota,
            G=G,
            method=method,
        )
        return self._annotate_current_result(result)

    def minimize_boozer_exact_constraints_newton(
        self,
        tol=1e-12,
        maxiter=10,
        iota=0.0,
        *,
        G,
        lm=(0.0, 0.0),
    ):
        G = _require_explicit_G(G)
        result = super().minimize_boozer_exact_constraints_newton(
            tol=tol,
            maxiter=maxiter,
            iota=iota,
            G=G,
            lm=lm,
        )
        return self._annotate_current_result(result)

    def boozer_penalty_constraints(
        self,
        x,
        derivatives=0,
        constraint_weight=1.0,
        scalarize=True,
        optimize_G=True,
        weight_inv_modB=True,
    ):
        r"""Finite-I version of :meth:`BoozerSurface.boozer_penalty_constraints`.

        Finite-current Boozer paths require an explicit signed ``G``; this
        method therefore supports only the ``optimize_G=True`` degree-of-freedom
        layout ``(surface_dofs, iota, G)``.
        """

        assert derivatives in [0, 1, 2]
        if not optimize_G:
            raise ValueError("finite-current Boozer paths require optimize_G=True")
        sdofs = x[:-2]
        iota = x[-2]
        G = x[-1]
        nsurfdofs = sdofs.size
        s = self.surface
        num_res = 3 * s.quadpoints_phi.size * s.quadpoints_theta.size
        biotsavart = self.biotsavart

        s.set_dofs(sdofs)

        boozer = boozer_surface_residual_finite_I(
            s,
            iota,
            G,
            biotsavart,
            derivatives=derivatives,
            weight_inv_modB=weight_inv_modB,
            I=self.I,
        )
        # normalizing the residuals here
        boozer = tuple([b / np.sqrt(num_res) for b in boozer])

        r = boozer[0]
        l = self.label.J()
        rl = l - self.targetlabel
        rz = s.gamma()[0, 0, 2] - 0.0
        r = np.concatenate(
            (
                r,
                [
                    np.sqrt(constraint_weight) * rl,
                    np.sqrt(constraint_weight) * rz,
                ],
            )
        )

        val = 0.5 * np.sum(r ** 2)
        if derivatives == 0:
            if scalarize:
                return val
            else:
                return r

        J = boozer[1]

        dl = np.zeros(x.shape)
        drz = np.zeros(x.shape)

        dl[:nsurfdofs] = self.label.dJ(partials=True)(s)

        drz[:nsurfdofs] = s.dgamma_by_dcoeff()[0, 0, 2, :]
        J = np.concatenate(
            (
                J,
                np.sqrt(constraint_weight) * dl[None, :],
                np.sqrt(constraint_weight) * drz[None, :],
            ),
            axis=0,
        )
        dval = np.sum(r[:, None] * J, axis=0)
        if derivatives == 1:
            if scalarize:
                return val, dval
            else:
                return r, J
        if not scalarize:
            raise NotImplementedError("Can only return Hessian for scalarized version.")

        H = boozer[2]

        d2l = np.zeros((x.shape[0], x.shape[0]))
        d2l[:nsurfdofs, :nsurfdofs] = self.label.d2J_by_dsurfacecoefficientsdsurfacecoefficients()

        H_full = np.concatenate(
            (
                H,
                np.sqrt(constraint_weight) * d2l[None, :, :],
                np.zeros((1, x.size, x.size)),
            ),
            axis=0,
        )
        d2val = J.T @ J + np.sum(r[:, None, None] * H_full, axis=0)
        return val, dval, d2val

    def boozer_penalty_constraints_vectorized(
        self,
        dofs,
        derivatives=0,
        constraint_weight=1.0,
        optimize_G=True,
        weight_inv_modB=True,
    ):
        """Vectorized finite-I implementation for explicit signed ``G``.

        Calls the upstream-only ``sopp.boozer_residual*`` kernels with
        ``alpha = G + iota * self.I``; the gradient/Hessian in the alpha-fixed
        basis are then transformed into the (..., iota, G) basis here.
        """

        assert derivatives in [0, 1, 2]
        if optimize_G:
            sdofs = dofs[:-2]
            iota = dofs[-2]
            G = dofs[-1]
        else:
            raise ValueError("finite-current Boozer paths require optimize_G=True")

        s = self.surface
        nphi = s.quadpoints_phi.size
        ntheta = s.quadpoints_theta.size
        nsurfdofs = sdofs.size

        s.set_dofs(sdofs)

        surface = self.surface
        biotsavart = self.biotsavart
        x = surface.gamma()
        xphi = surface.gammadash1()
        xtheta = surface.gammadash2()
        nphi = x.shape[0]
        ntheta = x.shape[1]

        xsemiflat = x.reshape((x.size // 3, 3)).copy()
        biotsavart.set_points(xsemiflat)
        biotsavart.compute(derivatives)
        B = biotsavart.B().reshape((nphi, ntheta, 3))

        if derivatives >= 1:
            dx_dc = surface.dgamma_by_dcoeff()
            dxphi_dc = surface.dgammadash1_by_dcoeff()
            dxtheta_dc = surface.dgammadash2_by_dcoeff()
            dB_dx = biotsavart.dB_by_dX().reshape((nphi, ntheta, 3, 3))

        if derivatives == 2:
            d2B_by_dXdX = biotsavart.d2B_by_dXdX().reshape((nphi, ntheta, 3, 3, 3))

        num_res = 3 * s.quadpoints_phi.size * s.quadpoints_theta.size
        alpha = G + iota * self.I

        if derivatives == 0:
            val = sopp.boozer_residual(alpha, iota, xphi, xtheta, B, weight_inv_modB)
            boozer = (val,)
        elif derivatives == 1:
            val, dval = sopp.boozer_residual_ds(
                alpha,
                iota,
                B,
                dB_dx,
                xphi,
                xtheta,
                dx_dc,
                dxphi_dc,
                dxtheta_dc,
                weight_inv_modB,
            )
            dval = _to_explicit_current_basis(self.I, dval)
            boozer = val, dval
        elif derivatives == 2:
            val, dval, d2val = sopp.boozer_residual_ds2(
                alpha,
                iota,
                B,
                dB_dx,
                d2B_by_dXdX,
                xphi,
                xtheta,
                dx_dc,
                dxphi_dc,
                dxtheta_dc,
                weight_inv_modB,
            )
            dval = _to_explicit_current_basis(self.I, dval)
            d2val = _to_explicit_current_basis_hessian(self.I, d2val)
            boozer = val, dval, d2val

        # normalizing the residuals here
        boozer = tuple([b / num_res for b in boozer])

        lab = self.label.J()

        rnl = boozer[0]
        rl = np.sqrt(constraint_weight) * (lab - self.targetlabel)
        rz = np.sqrt(constraint_weight) * (s.gamma()[0, 0, 2] - 0.0)
        r = rnl + 0.5 * rl ** 2 + 0.5 * rz ** 2

        if derivatives == 0:
            return r

        dl = np.zeros(dofs.shape)
        drz = np.zeros(dofs.shape)
        dl[:nsurfdofs] = self.label.dJ(partials=True)(s)
        drz[:nsurfdofs] = s.dgamma_by_dcoeff()[0, 0, 2, :]

        Jnl = boozer[1]
        if not optimize_G:
            Jnl = Jnl[:-1]

        drl = np.sqrt(constraint_weight) * dl
        drz = np.sqrt(constraint_weight) * drz
        J = Jnl + rl * drl + rz * drz

        if derivatives == 1:
            return r, J

        Hnl = boozer[2]
        if not optimize_G:
            Hnl = Hnl[:-1, :-1]

        d2rl = np.zeros((dofs.shape[0], dofs.shape[0]))
        d2rl[:nsurfdofs, :nsurfdofs] = (
            np.sqrt(constraint_weight)
            * self.label.d2J_by_dsurfacecoefficientsdsurfacecoefficients()
        )
        H = (
            Hnl
            + drl[:, None] @ drl[None, :]
            + drz[:, None] @ drz[None, :]
            + rl * d2rl
        )

        return r, J, H

    def boozer_exact_constraints(self, xl, derivatives=0, optimize_G=True):
        r"""Finite-I version of :meth:`BoozerSurface.boozer_exact_constraints`."""

        assert derivatives in [0, 1]
        if optimize_G:
            sdofs = xl[:-4]
            iota = xl[-4]
            G = xl[-3]
        else:
            raise ValueError("finite-current Boozer paths require optimize_G=True")
        lm = xl[-2:]
        s = self.surface
        biotsavart = self.biotsavart
        s.set_dofs(sdofs)
        nsurfdofs = sdofs.size

        boozer = boozer_surface_residual_finite_I(
            s,
            iota,
            G,
            biotsavart,
            derivatives=derivatives + 1,
            I=self.I,
        )
        r, J = boozer[0:2]

        dl = np.zeros((xl.shape[0] - 2,))

        l = self.label.J()
        dl[:nsurfdofs] = self.label.dJ(partials=True)(s)
        drz = np.zeros((xl.shape[0] - 2,))
        g = [l - self.targetlabel]
        rz = s.gamma()[0, 0, 2] - 0.0
        drz[:nsurfdofs] = s.dgamma_by_dcoeff()[0, 0, 2, :]

        res = np.zeros(xl.shape)
        res[:-2] = np.sum(r[:, None] * J, axis=0) - lm[-2] * dl - lm[-1] * drz
        res[-2] = g[0]
        res[-1] = rz
        if derivatives == 0:
            return res

        H = boozer[2]

        d2l = np.zeros((xl.shape[0] - 2, xl.shape[0] - 2))
        d2l[:nsurfdofs, :nsurfdofs] = self.label.d2J_by_dsurfacecoefficientsdsurfacecoefficients()

        dres = np.zeros((xl.shape[0], xl.shape[0]))
        dres[:-2, :-2] = J.T @ J + np.sum(r[:, None, None] * H, axis=0) - lm[-2] * d2l
        dres[:-2, -2] = -dl
        dres[:-2, -1] = -drz

        dres[-2, :-2] = dl
        dres[-1, :-2] = drz
        return res, dres

    def solve_residual_equation_exactly_newton(
        self, tol=1e-10, maxiter=10, iota=0.0, *, G, verbose=False
    ):
        """Finite-I version of
        :meth:`BoozerSurface.solve_residual_equation_exactly_newton`.

        Logic follows upstream except the residual is computed via
        :func:`boozer_surface_residual_finite_I`, each Newton direction is
        accepted only at a residual-decreasing backtracking step, the result
        dict is annotated with ``"I"``, and the stored vjp callback is wrapped
        to substitute ``G -> G + iota * I`` before invoking the upstream vjp.
        """

        G = _require_explicit_G(G)
        if not self.need_to_run_code:
            return self.res

        from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier

        s = self.surface
        if not isinstance(s, SurfaceXYZTensorFourier):
            raise RuntimeError(
                "Exact solution of Boozer Surfaces only supported for SurfaceXYZTensorFourier"
            )

        m = s.get_stellsym_mask()
        mask = np.concatenate((m[..., None], m[..., None], m[..., None]), axis=2)
        if s.stellsym:
            mask[0, 0, 0] = False
        mask = mask.flatten()

        label = self.label

        def residual_vector(residual):
            label_constraints = [label.J() - self.targetlabel]
            if not s.stellsym:
                label_constraints.append(s.gamma()[0, 0, 2])
            return np.concatenate((residual[mask], label_constraints))

        def constrained_jacobian(residual_jacobian):
            rows = [
                residual_jacobian[mask, :],
                np.concatenate((label.dJ(partials=True)(s), [0.0, 0.0])),
            ]
            if not s.stellsym:
                rows.append(np.concatenate((s.dgamma_by_dcoeff()[0, 0, 2, :], [0.0, 0.0])))
            return np.vstack(rows)

        def result_from_state(residual, jacobian, iteration, success, plu, message=None):
            res = {
                "residual": residual,
                "jacobian": jacobian,
                "iter": iteration,
                "success": success,
                "G": G,
                "s": s,
                "iota": iota,
                "PLU": plu,
                "mask": mask,
                "type": "exact",
                "weight_inv_modB": False,
                # Wrap the upstream alpha-only vjp so it sees alpha = G + iota * I.
                "vjp": partial(_exact_vjp_finite_I, self.I),
            }
            if message is not None:
                res["message"] = message
            res = self._annotate_current(res)

            if verbose:
                print(
                    f"NEWTON solve - {res['success']}  iter={res['iter']}, iota={res['iota']:.16f}, "
                    f"||residual||_inf = {np.linalg.norm(res['residual'], ord=np.inf):.3e}",
                    flush=True,
                )

            self.res = res
            self.need_to_run_code = False
            return res

        x = np.concatenate((s.get_dofs(), [iota, G]))
        i = 0
        r, J = boozer_surface_residual_finite_I(
            s, iota, G, self.biotsavart, derivatives=1, I=self.I
        )
        norm = 1e6
        while i < maxiter:
            b = residual_vector(r)
            norm = np.linalg.norm(b)
            if norm <= tol:
                break
            J_augmented = constrained_jacobian(J)
            P, L, U = lu(J_augmented)
            dx = forward_solve(P, L, U, b)
            dx += forward_solve(P, L, U, b - J_augmented @ dx)

            if self.I == 0.0:
                x -= dx
                s.set_dofs(x[:-2])
                iota = x[-2]
                G = x[-1]
                i += 1
                r, J = boozer_surface_residual_finite_I(
                    s, iota, G, self.biotsavart, derivatives=1, I=self.I
                )
                continue

            step_accepted = False
            for step_scale in _EXACT_NEWTON_BACKTRACKING_STEPS:
                x_trial = x - step_scale * dx
                s.set_dofs(x_trial[:-2])
                iota_trial = x_trial[-2]
                G_trial = x_trial[-1]
                r_trial, J_trial = boozer_surface_residual_finite_I(
                    s, iota_trial, G_trial, self.biotsavart, derivatives=1, I=self.I
                )
                trial_norm = np.linalg.norm(residual_vector(r_trial))
                if trial_norm < norm:
                    x = x_trial
                    iota = iota_trial
                    G = G_trial
                    r = r_trial
                    J = J_trial
                    norm = trial_norm
                    step_accepted = True
                    break

            if not step_accepted:
                s.set_dofs(x[:-2])
                break

            i += 1

        J = constrained_jacobian(J)

        P, L, U = lu(J)
        return result_from_state(r, J, i, norm <= tol, (P, L, U))
