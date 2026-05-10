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

and similarly ``H_true = T^T H_alpha T`` for Hessians.

Design rationale
----------------
* All the I-aware logic lives here, in ``examples/``.  ``src/simsopt`` stays a
  clean copy of upstream simsopt so the fork can be rebased / cleanly merged.
* No private helpers from ``src/`` are imported.  The wrapper depends only on
  the public upstream API: ``boozer_surface_residual``,
  ``boozer_surface_residual_dB``, the ``BoozerSurface`` and ``BoozerResidual``
  classes, and the ``simsoptpp`` upstream kernel signatures.
* The four ``BoozerSurface`` numerical methods that *consume* ``alpha`` are
  overridden in :class:`BoozerSurfaceFiniteI` rather than monkey-patching: the
  upstream methods are transcribed verbatim, with the substitution
  ``G -> G + iota * self.I`` at the residual call site and the ``T^T g`` /
  ``T^T H T`` transform applied to gradients/Hessians where the upstream
  kernel returns derivatives in the alpha-fixed basis.
* :class:`BoozerResidualFiniteI` reads ``self.boozer_surface.I`` and routes
  through the I-aware ``boozer_surface_residual_dB_finite_I`` wrapper so that
  ``J`` and ``dJ/dB`` are computed against the correct residual.
"""

from functools import partial

import numpy as np
from scipy.linalg import lu

import simsoptpp as sopp

from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.surfaceobjectives import (
    BoozerResidual,
    boozer_surface_dexactresidual_dcoils_dcurrents_vjp,
    boozer_surface_dlsqgrad_dcoils_vjp,
    boozer_surface_residual,
    boozer_surface_residual_dB,
)
from simsopt.objectives.utilities import forward_backward, forward_solve

__all__ = [
    "boozer_surface_residual_finite_I",
    "boozer_surface_residual_dB_finite_I",
    "BoozerSurfaceFiniteI",
    "BoozerResidualFiniteI",
]


# ---------------------------------------------------------------------------
# Basis-transform helpers (moved from src/).  These convert derivative tensors
# returned in the alpha-only basis (where alpha = G_eff is held fixed) into
# the (iota, G; I-external) basis used by the optimizer.
# ---------------------------------------------------------------------------


def _explicit_current_transform(I, size):
    """Elementary (size x size) transform mapping the alpha-fixed basis to
    the (iota, G) basis when ``alpha = G + iota * I``.

    The last two coordinates of the basis are assumed to be ``[..., iota, G]``;
    ``size`` therefore must be at least 2.
    """

    transform = np.eye(size)
    transform[-1, -2] = I
    return transform


def _alpha_only_to_explicit_current_gradient(I, gradient):
    """Apply ``T^T`` to a gradient (1-D vector or 2-D Jacobian-row stack)."""

    return _explicit_current_transform(I, gradient.shape[0]).T @ gradient


def _alpha_only_to_explicit_current_hessian(I, hessian):
    """Apply ``T^T H T`` to a Hessian (2-D matrix)."""

    transform = _explicit_current_transform(I, hessian.shape[0])
    return transform.T @ hessian @ transform


# ---------------------------------------------------------------------------
# Module-level (function-style) wrappers.
# ---------------------------------------------------------------------------


def boozer_surface_residual_finite_I(
    surface,
    iota,
    G,
    biotsavart,
    derivatives=0,
    weight_inv_modB=False,
    I=0.0,
):
    """Drop-in replacement for ``boozer_surface_residual`` that supports a
    finite enclosed toroidal current ``I``.

    Parameters mirror the upstream signature.  When ``derivatives >= 1`` the
    returned Jacobian is in the (surface_dofs, iota, [G]) basis with ``I`` as
    an external parameter; this differs from the upstream alpha-fixed basis
    by the elementary basis transform implemented in this module.
    """

    user_provided_G = G is not None
    if not user_provided_G:
        # Same convention as upstream: deduce G from coil currents in the
        # vacuum limit so the caller doesn't have to.
        G = (
            2.0
            * np.pi
            * np.sum(np.abs([c.current.get_value() for c in biotsavart.coils]))
            * (4 * np.pi * 1e-7 / (2 * np.pi))
        )

    G_effective = G + iota * I

    # Always pass G_effective (never None) to upstream so that the upstream
    # kernel can populate the alpha-derivative column we need for the basis
    # transform.  When the outer caller passed ``G=None`` we drop that column
    # (and the corresponding Hessian row/col) from the returned tensors so
    # the upstream public contract is preserved.
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
        # J is shape (num_residuals, nsurfdofs + 2): ..., iota, G_effective.
        J_new = _alpha_only_to_explicit_current_gradient(I, J.T).T
        if not user_provided_G:
            J_new = J_new[:, :-1]
        return r, J_new

    r, J, H = boozer
    J_new = _alpha_only_to_explicit_current_gradient(I, J.T).T
    transform = _explicit_current_transform(I, H.shape[1])
    H_new = np.einsum("ji,kjl,lm->kim", transform, H, transform, optimize=True)
    if not user_provided_G:
        J_new = J_new[:, :-1]
        H_new = H_new[:, :-1, :-1]
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
    """Drop-in replacement for ``boozer_surface_residual_dB``.

    Only the ``dr/ds`` (Jacobian) output needs basis correction --- ``dr/dB``
    and the higher-order mixed derivatives are partials with respect to the
    field components alone, which depend on ``alpha`` only.  Substituting
    ``alpha = G + iota * I`` upstream therefore captures those terms exactly.
    """

    user_provided_G = G is not None
    if not user_provided_G:
        G = (
            2.0
            * np.pi
            * np.sum(np.abs([c.current.get_value() for c in biotsavart.coils]))
            * (4 * np.pi * 1e-7 / (2 * np.pi))
        )

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
        # (rtil_flat, drtil_dB_flat); neither depends on the (iota, G) basis.
        return out

    # derivatives == 1: structure depends on include_mixed_derivatives
    if include_mixed_derivatives:
        rtil, drtil_dB, J, d2_dsdB, d2_dsdgradB = out
    else:
        rtil, drtil_dB, J = out

    # Always pre-transform; the upstream tensors are sized for G_effective
    # (since we always pass it).  When the outer caller passed ``G=None`` we
    # drop the trailing G column from every transformed tensor afterwards.
    J_new = _alpha_only_to_explicit_current_gradient(I, J.T).T

    if include_mixed_derivatives:
        # d2r/dsdB has shape (num_components, 3, N) and d2r/dsdgradB has shape
        # (num_components, 3, 3, N), with the surface-state basis on the
        # trailing axis. Apply the elementary transform to that last axis.
        transform = _explicit_current_transform(I, J_new.shape[1])
        d2_dsdB_new = np.einsum("ijk,kl->ijl", d2_dsdB, transform, optimize=True)
        d2_dsdgradB_new = np.einsum(
            "ijkl,lm->ijkm", d2_dsdgradB, transform, optimize=True
        )
        if not user_provided_G:
            J_new = J_new[:, :-1]
            d2_dsdB_new = d2_dsdB_new[..., :-1]
            d2_dsdgradB_new = d2_dsdgradB_new[..., :-1]
        return rtil, drtil_dB, J_new, d2_dsdB_new, d2_dsdgradB_new

    if not user_provided_G:
        J_new = J_new[:, :-1]
    return rtil, drtil_dB, J_new


# ---------------------------------------------------------------------------
# Adjoint (vjp) wrappers.  The upstream vjp callbacks compute their internal
# alpha as ``G`` and so silently drop the iota * I contribution.  We wrap them
# to substitute G_effective = G + iota * I before delegating.
# ---------------------------------------------------------------------------


def _exact_vjp_finite_I(I_value, lm, booz_surf, iota, G):
    """``boozer_surface_dexactresidual_dcoils_dcurrents_vjp`` with the I
    substitution applied via ``G -> G + iota * I``.

    The upstream vjp uses ``alpha = G`` internally; passing ``G + iota * I``
    yields the correct ``alpha`` for the finite-I residual.
    """

    G_effective = G + iota * I_value
    return boozer_surface_dexactresidual_dcoils_dcurrents_vjp(
        lm, booz_surf, iota, G_effective
    )


def _lsqgrad_vjp_finite_I(I_value, lm, booz_surf, iota, G, weight_inv_modB=True):
    """``boozer_surface_dlsqgrad_dcoils_vjp`` with the I substitution applied.

    The upstream lsq adjoint computes ``vjp(lm @ J_upstream^T @ residual)``,
    where ``J_upstream`` is the Jacobian in the alpha-only basis. We want
    ``vjp(lm @ J_new^T @ residual)`` where ``J_new = J_upstream @ T`` is the
    Jacobian in the explicit-current basis (T is the elementary transform).
    Substituting: ``lm @ J_new^T @ r = lm @ T^T @ J_upstream^T @ r =
    (T @ lm) @ J_upstream^T @ r``. So we pass ``T @ lm`` (not ``lm``) to the
    upstream vjp, plus ``G_effective = G + iota * I`` for the residual.
    """
    G_effective = G + iota * I_value
    transform = _explicit_current_transform(I_value, np.asarray(lm).shape[0])
    lm_upstream = transform @ lm
    return boozer_surface_dlsqgrad_dcoils_vjp(
        lm_upstream, booz_surf, iota, G_effective, weight_inv_modB=weight_inv_modB
    )


# ---------------------------------------------------------------------------
# BoozerSurface override that carries an enclosed current ``I``.
# ---------------------------------------------------------------------------


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
        """Attach the enclosed-current value to a result dict in place and,
        for LS-type solver outputs, replace the upstream alpha-only adjoint
        VJP with the I-aware variant. The exact-Newton path already installs
        an I-aware VJP via ``solve_residual_equation_exactly_newton``.
        """
        if not isinstance(payload, dict):
            return payload
        payload["I"] = self.I
        if payload.get("type") == "ls":
            weight_inv_modB = payload.get("weight_inv_modB", True)
            payload["vjp"] = partial(
                _lsqgrad_vjp_finite_I,
                self.I,
                weight_inv_modB=weight_inv_modB,
            )
        return payload

    def run_code(self, iota, G=None):
        result = super().run_code(iota, G=G)
        # ``run_code`` returns either the inner solver's result dict or
        # nothing (when ``need_to_run_code`` is already False).  In every
        # case, propagate ``I`` (and patch the LS adjoint VJP if relevant)
        # into ``self.res`` for downstream consumers.
        self._annotate_current(self.res)
        return self._annotate_current(result)

    # ------------------------------------------------------------------
    # boozer_penalty_constraints --- python (non-vectorized) version.
    # ------------------------------------------------------------------
    def boozer_penalty_constraints(
        self,
        x,
        derivatives=0,
        constraint_weight=1.0,
        scalarize=True,
        optimize_G=False,
        weight_inv_modB=True,
    ):
        r"""Finite-I version of :meth:`BoozerSurface.boozer_penalty_constraints`.

        Behavior matches upstream byte-for-byte except the residual call is
        routed through :func:`boozer_surface_residual_finite_I` so the
        gradient/Hessian come out in the (surface_dofs, iota, [G]) basis with
        ``I`` external.
        """

        assert derivatives in [0, 1, 2]
        if optimize_G:
            sdofs = x[:-2]
            iota = x[-2]
            G = x[-1]
        else:
            sdofs = x[:-1]
            iota = x[-1]
            G = None
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

        # ``H`` is a (num_residuals, N, N) tensor.  We must aggregate it via
        # ``sum(r * H, axis=0)`` *before* adding it to a 2-D matrix; trying to
        # add the 3-D tensor directly would be a dimension mismatch.
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

    # ------------------------------------------------------------------
    # boozer_penalty_constraints_vectorized --- C++ kernel version.
    # ------------------------------------------------------------------
    def boozer_penalty_constraints_vectorized(
        self,
        dofs,
        derivatives=0,
        constraint_weight=1.0,
        optimize_G=False,
        weight_inv_modB=True,
    ):
        """Vectorized finite-I implementation matching upstream behavior.

        We call the upstream-only (post-revert) ``sopp.boozer_residual*``
        kernels with ``alpha = G + iota * self.I``.  When derivatives are
        requested, the kernel returns a gradient/Hessian in the alpha-fixed
        basis which we transform into the (..., iota, G) basis here.
        """

        assert derivatives in [0, 1, 2]
        if optimize_G:
            sdofs = dofs[:-2]
            iota = dofs[-2]
            G = dofs[-1]
        else:
            sdofs = dofs[:-1]
            iota = dofs[-1]
            # Match upstream: derive G from coil currents using the
            # underscore-prefixed ``_coils`` attribute (consistent with the
            # upstream pattern at the same point in this method).
            G = (
                2.0
                * np.pi
                * np.sum(
                    np.abs(
                        [coil.current.get_value() for coil in self.biotsavart._coils]
                    )
                )
                * (4 * np.pi * 1e-7 / (2 * np.pi))
            )

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
            dval = _alpha_only_to_explicit_current_gradient(self.I, dval)
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
            dval = _alpha_only_to_explicit_current_gradient(self.I, dval)
            d2val = _alpha_only_to_explicit_current_hessian(self.I, d2val)
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

    # ------------------------------------------------------------------
    # boozer_exact_constraints --- Lagrange-multiplier formulation.
    # ------------------------------------------------------------------
    def boozer_exact_constraints(self, xl, derivatives=0, optimize_G=True):
        r"""Finite-I version of :meth:`BoozerSurface.boozer_exact_constraints`."""

        assert derivatives in [0, 1]
        if optimize_G:
            sdofs = xl[:-4]
            iota = xl[-4]
            G = xl[-3]
        else:
            sdofs = xl[:-3]
            iota = xl[-3]
            G = None
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

    # ------------------------------------------------------------------
    # solve_residual_equation_exactly_newton --- the Newton solver
    # used by the BoozerExact path.
    # ------------------------------------------------------------------
    def solve_residual_equation_exactly_newton(
        self, tol=1e-10, maxiter=10, iota=0.0, G=None, verbose=False
    ):
        """Finite-I version of
        :meth:`BoozerSurface.solve_residual_equation_exactly_newton`.

        Logic is byte-for-byte upstream except the residual is computed via
        :func:`boozer_surface_residual_finite_I`, the resulting result dict
        is annotated with ``"I"``, and the stored vjp callback is wrapped to
        substitute ``G -> G + iota * I`` before invoking the upstream vjp.
        """

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
        if G is None:
            G = (
                2.0
                * np.pi
                * np.sum(
                    np.abs(
                        [c.current.get_value() for c in self.biotsavart.coils]
                    )
                )
                * (4 * np.pi * 1e-7 / (2 * np.pi))
            )
        x = np.concatenate((s.get_dofs(), [iota, G]))
        i = 0
        r, J = boozer_surface_residual_finite_I(
            s, iota, G, self.biotsavart, derivatives=1, I=self.I
        )
        norm = 1e6
        while i < maxiter:
            if s.stellsym:
                b = np.concatenate((r[mask], [(label.J() - self.targetlabel)]))
            else:
                b = np.concatenate(
                    (
                        r[mask],
                        [(label.J() - self.targetlabel), s.gamma()[0, 0, 2]],
                    )
                )
            norm = np.linalg.norm(b)
            if norm <= tol:
                break
            if s.stellsym:
                J = np.vstack(
                    (
                        J[mask, :],
                        np.concatenate((label.dJ(partials=True)(s), [0.0, 0.0])),
                    )
                )
            else:
                J = np.vstack(
                    (
                        J[mask, :],
                        np.concatenate((label.dJ(partials=True)(s), [0.0, 0.0])),
                        np.concatenate((s.dgamma_by_dcoeff()[0, 0, 2, :], [0.0, 0.0])),
                    )
                )
            P, L, U = lu(J)
            dx = forward_solve(P, L, U, b)
            dx += forward_solve(P, L, U, b - J @ dx)
            x -= dx
            s.set_dofs(x[:-2])
            iota = x[-2]
            G = x[-1]
            i += 1
            r, J = boozer_surface_residual_finite_I(
                s, iota, G, self.biotsavart, derivatives=1, I=self.I
            )

        if s.stellsym:
            J = np.vstack(
                (
                    J[mask, :],
                    np.concatenate((label.dJ(partials=True)(s), [0.0, 0.0])),
                )
            )
        else:
            J = np.vstack(
                (
                    J[mask, :],
                    np.concatenate((label.dJ(partials=True)(s), [0.0, 0.0])),
                    np.concatenate((s.dgamma_by_dcoeff()[0, 0, 2, :], [0.0, 0.0])),
                )
            )

        P, L, U = lu(J)
        res = {
            "residual": r,
            "jacobian": J,
            "iter": i,
            "success": norm <= tol,
            "G": G,
            "s": s,
            "iota": iota,
            "PLU": (P, L, U),
            "mask": mask,
            "type": "exact",
            "weight_inv_modB": False,
            # Wrap the upstream alpha-only vjp so it sees alpha = G + iota * I.
            "vjp": partial(_exact_vjp_finite_I, self.I),
        }
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


# ---------------------------------------------------------------------------
# BoozerResidual override that respects the enclosed-current value carried by
# its associated BoozerSurface.
# ---------------------------------------------------------------------------


class BoozerResidualFiniteI(BoozerResidual):
    """A :class:`BoozerResidual` that reads ``self.boozer_surface.I`` and
    evaluates the Boozer residual through :func:`boozer_surface_residual_dB_finite_I`.

    The class structure (inputs, outputs, gradient assembly) is identical to
    upstream; the only differences are the residual call site and the use of
    the I-aware least-squares-vjp wrapper for the BoozerLS branch.
    """

    def _enclosed_current(self):
        return float(getattr(self.boozer_surface, "I", 0.0))

    def compute(self):
        self.boozer_surface.run_code_from_last_solution()

        self.surface.set_dofs(self.in_surface.get_dofs())

        nphi = self.surface.quadpoints_phi.size
        ntheta = self.surface.quadpoints_theta.size
        num_points = 3 * nphi * ntheta
        sqrt_num_points = np.sqrt(num_points)

        surface = self.surface
        booz_surf = self.boozer_surface
        iota = booz_surf.res["iota"]
        G = booz_surf.res["G"]
        I_value = self._enclosed_current()
        weight_inv_modB = booz_surf.res["weight_inv_modB"]
        if booz_surf.res["type"] == "ls":
            r, r_dB, J, d2r_dsdB, d2r_dsdgradB = boozer_surface_residual_dB_finite_I(
                surface,
                iota,
                G,
                self.biotsavart,
                derivatives=1,
                weight_inv_modB=weight_inv_modB,
                I=I_value,
            )
        else:
            r, r_dB, J = boozer_surface_residual_dB_finite_I(
                surface,
                iota,
                G,
                self.biotsavart,
                derivatives=1,
                weight_inv_modB=weight_inv_modB,
                I=I_value,
                include_mixed_derivatives=False,
            )
        constraint_scale = np.sqrt(self.constraint_weight)
        label_residual = self.boozer_surface.label.J() - self.boozer_surface.targetlabel
        rtil = np.concatenate((r / sqrt_num_points, [constraint_scale * label_residual]))
        self._J = 0.5 * np.sum(rtil ** 2)

        P, L, U = booz_surf.res["PLU"]

        dJ_by_dB = self._scaled_dJ_by_dB(r, r_dB, sqrt_num_points)
        dJ_by_dcoils = self.biotsavart.B_vjp(dJ_by_dB)

        dl = np.zeros((J.shape[1],))
        dlabel_dsurface = self.boozer_surface.label.dJ_by_dsurfacecoefficients()
        dl[: dlabel_dsurface.size] = dlabel_dsurface
        Jtil = np.concatenate(
            (J / sqrt_num_points, constraint_scale * dl[None, :]), axis=0
        )
        dJ_ds = Jtil.T @ rtil

        adj = forward_backward(P, L, U, dJ_ds)

        if booz_surf.res["type"] == "ls":
            adj_times_dg_dcoil = self._lsq_vjp_finite_I(
                adj,
                booz_surf,
                iota,
                G,
                I_value,
                weight_inv_modB,
                r,
                r_dB,
                J,
                d2r_dsdB,
                d2r_dsdgradB,
                sqrt_num_points,
            )
        else:
            adj_times_dg_dcoil = booz_surf.res["vjp"](adj, booz_surf, iota, G)
        self._dJ = dJ_by_dcoils - adj_times_dg_dcoil

    def dJ_by_dB(self):
        """Return the partial derivative of the objective with respect to B."""

        res = self.boozer_surface.run_code_from_last_solution()
        self.surface.set_dofs(self.in_surface.get_dofs())
        surface = self.surface
        nphi = self.surface.quadpoints_phi.size
        ntheta = self.surface.quadpoints_theta.size
        num_points = 3 * nphi * ntheta
        I_value = self._enclosed_current()
        r, r_dB = boozer_surface_residual_dB_finite_I(
            surface,
            res["iota"],
            res["G"],
            self.biotsavart,
            derivatives=0,
            weight_inv_modB=res["weight_inv_modB"],
            I=I_value,
        )
        return self._scaled_dJ_by_dB(r, r_dB, np.sqrt(num_points))

    # ------------------------------------------------------------------
    # Local helpers -- functional, no state.
    # ------------------------------------------------------------------
    @staticmethod
    def _scaled_dJ_by_dB(r, r_dB, sqrt_num_components):
        """Mirror of upstream ``_boozer_residual_dJ_by_dB`` so we don't depend
        on the private symbol."""
        scaled_r = r / sqrt_num_components
        scaled_r_dB = r_dB / sqrt_num_components
        dJ_by_dB = scaled_r[:, None] * scaled_r_dB
        return np.sum(dJ_by_dB.reshape((-1, 3, 3)), axis=1)

    @staticmethod
    def _lsq_vjp_finite_I(
        lm,
        booz_surf,
        iota,
        G,
        I_value,
        weight_inv_modB,
        r,
        r_dB,
        J,
        d2r_dsdB,
        d2r_dsdgradB,
        sqrt_num_components,
    ):
        """Inlined I-aware version of ``_boozer_lsqgrad_vjp_from_residual_state``.

        The upstream helper is private; rather than import it (which would
        couple us to a src/-internal symbol) we re-implement the same
        contraction here.  All inputs are already in the I-aware basis.
        """
        r_scaled = r / sqrt_num_components
        dr_dB = r_dB.reshape((-1, 3, 3)) / sqrt_num_components
        dr_ds = J / sqrt_num_components
        d2r_dsdB_scaled = d2r_dsdB / sqrt_num_components
        d2r_dsdgradB_scaled = d2r_dsdgradB / sqrt_num_components

        v1 = np.sum(
            np.sum(lm[:, None] * dr_ds.T, axis=0).reshape((-1, 3, 1)) * dr_dB,
            axis=1,
        )
        v2 = np.sum(
            r_scaled.reshape((-1, 3, 1))
            * np.sum(lm[None, None, :] * d2r_dsdB_scaled, axis=-1).reshape(
                (-1, 3, 3)
            ),
            axis=1,
        )
        v3 = np.sum(
            r_scaled.reshape((-1, 3, 1, 1))
            * np.sum(lm[None, None, None, :] * d2r_dsdgradB_scaled, axis=-1).reshape(
                (-1, 3, 3, 3)
            ),
            axis=1,
        )
        dres_dcoils = booz_surf.biotsavart.B_and_dB_vjp(v1 + v2, v3)
        return dres_dcoils[0] + dres_dcoils[1]
