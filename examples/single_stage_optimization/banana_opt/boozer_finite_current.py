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
  upstream methods are transcribed verbatim, with the substitution
  ``G -> G + iota * self.I`` at the residual call site and the ``T^T g`` /
  ``T^T H T`` transform applied to gradients/Hessians where the upstream
  kernel returns derivatives in the alpha-fixed basis.
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
]


_MU0_OVER_2PI = 4 * np.pi * 1e-7 / (2 * np.pi)


def _default_G_from_coils(biotsavart):
    """Upstream default: ``G = sum(|coil currents|) * mu_0``.  Used when the
    caller does not pass an explicit ``G``.
    """
    return (
        2.0
        * np.pi
        * np.sum(np.abs([c.current.get_value() for c in biotsavart.coils]))
        * _MU0_OVER_2PI
    )


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
    """Drop-in replacement for ``boozer_surface_residual`` that supports a
    finite enclosed toroidal current ``I``.

    Parameters mirror the upstream signature.  When ``derivatives >= 1`` the
    returned Jacobian is in the (surface_dofs, iota, [G]) basis with ``I`` as
    an external parameter; this differs from the upstream alpha-fixed basis
    by the elementary basis transform implemented in this module.
    """

    user_provided_G = G is not None
    if not user_provided_G:
        G = _default_G_from_coils(biotsavart)

    G_effective = G + iota * I

    # Always pass G_effective (never None) so the upstream kernel populates
    # the alpha-derivative column the basis transform needs.  When the outer
    # caller passed G=None we drop that column from the returned tensors so
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
        J_new = _to_explicit_current_basis(I, J)
        if not user_provided_G:
            J_new = J_new[:, :-1]
        return r, J_new

    r, J, H = boozer
    J_new = _to_explicit_current_basis(I, J)
    H_new = _to_explicit_current_basis_hessian(I, H)
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
        G = _default_G_from_coils(biotsavart)

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
        if not user_provided_G:
            J_new = J_new[:, :-1]
            d2_dsdB_new = d2_dsdB_new[..., :-1]
            d2_dsdgradB_new = d2_dsdgradB_new[..., :-1]
        return rtil, drtil_dB, J_new, d2_dsdB_new, d2_dsdgradB_new

    rtil, drtil_dB, J = out
    J_new = _to_explicit_current_basis(I, J)
    if not user_provided_G:
        J_new = J_new[:, :-1]
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

    def run_code(self, iota, G=None):
        result = super().run_code(iota, G=G)
        self._annotate_current(self.res)
        return self._annotate_current(result)

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
        optimize_G=False,
        weight_inv_modB=True,
    ):
        """Vectorized finite-I implementation matching upstream behavior.

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
            sdofs = dofs[:-1]
            iota = dofs[-1]
            G = _default_G_from_coils(self.biotsavart)

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
            G = _default_G_from_coils(self.biotsavart)
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
