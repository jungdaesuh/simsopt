from __future__ import annotations

import numpy as np

from simsopt._core.derivative import derivative_dec
from simsopt._core.optimizable import Optimizable
from simsopt.geo import SurfaceXYZTensorFourier
from simsopt.geo.surfaceobjectives import (
    _boozer_lsqgrad_vjp_from_residual_state,
    _boozer_residual_dJ_by_dB,
    _resolve_boozer_current_I,
    boozer_surface_residual_dB,
)
from simsopt.objectives.utilities import forward_backward

__all__ = ["BoozerResidualExact", "RefinedBoozerResidual"]


def _quadpoints_for_multiplier(in_surface, grid_multiplier):
    if grid_multiplier == 1:
        return in_surface.quadpoints_phi, in_surface.quadpoints_theta

    nphis = in_surface.quadpoints_phi.size
    nthetas = in_surface.quadpoints_theta.size
    return (
        np.linspace(
            0,
            1.0 / in_surface.nfp,
            nphis * grid_multiplier,
            endpoint=False,
        ),
        np.linspace(
            0,
            1,
            nthetas * grid_multiplier,
            endpoint=False,
        ),
    )


def _num_boozer_components(surface):
    return 3 * surface.quadpoints_phi.size * surface.quadpoints_theta.size


class RefinedBoozerResidual(Optimizable):
    r"""
    Configurable Boozer residual used by the banana single-stage optimizer.

    ``grid_multiplier=1`` preserves the same quadrature grid as
    :class:`simsopt.geo.surfaceobjectives.BoozerResidual`. Larger multipliers
    evaluate the same solved surface on a fresh uniform ``linspace`` grid over
    one field period, not by subdividing the input quadrature. The label
    constraint path follows ``BoozerResidual`` and expects a numeric
    ``constraint_weight``.
    """

    def __init__(
        self,
        boozer_surface,
        bs,
        *,
        grid_multiplier: int = 1,
        include_label_constraint: bool = True,
        weight_inv_modB: bool | None = None,
    ):
        if grid_multiplier < 1:
            raise ValueError("grid_multiplier must be >= 1")

        Optimizable.__init__(self, depends_on=[boozer_surface])
        in_surface = boozer_surface.surface
        self.boozer_surface = boozer_surface
        self.grid_multiplier = grid_multiplier
        self.include_label_constraint = include_label_constraint
        self.weight_inv_modB = weight_inv_modB

        phis, thetas = _quadpoints_for_multiplier(in_surface, grid_multiplier)

        s = SurfaceXYZTensorFourier(
            mpol=in_surface.mpol,
            ntor=in_surface.ntor,
            stellsym=in_surface.stellsym,
            nfp=in_surface.nfp,
            quadpoints_phi=phis,
            quadpoints_theta=thetas,
        )
        s.set_dofs(in_surface.get_dofs())

        if include_label_constraint:
            self.constraint_weight = boozer_surface.constraint_weight
        self.in_surface = in_surface
        self.surface = s
        self.biotsavart = bs
        self.recompute_bell()

    def J(self):
        """
        Return the value of the penalty function.
        """

        if self._J is None:
            self.compute()
        return self._J

    @derivative_dec
    def dJ(self):
        """
        Return the derivative of the penalty function with respect to coil DOFs.
        """

        if self._dJ is None:
            self.compute()
        return self._dJ

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    def _weight_inv_modB(self):
        if self.weight_inv_modB is None:
            return self.boozer_surface.res["weight_inv_modB"]
        return self.weight_inv_modB

    def compute(self):
        if self.boozer_surface.need_to_run_code:
            res = self.boozer_surface.res
            self.boozer_surface.run_code(res["iota"], G=res["G"])

        self.surface.set_dofs(self.in_surface.get_dofs())

        surface = self.surface
        sqrt_n = np.sqrt(_num_boozer_components(surface))

        iota = self.boozer_surface.res["iota"]
        G = self.boozer_surface.res["G"]
        I = _resolve_boozer_current_I(self.boozer_surface)
        weight_inv_modB = self._weight_inv_modB()

        if self.boozer_surface.res["type"] == "ls":
            r, r_dB, J, d2r_dsdB, d2r_dsdgradB = boozer_surface_residual_dB(
                surface,
                iota,
                G,
                self.biotsavart,
                derivatives=1,
                weight_inv_modB=weight_inv_modB,
                I=I,
            )
        else:
            r, r_dB, J = boozer_surface_residual_dB(
                surface,
                iota,
                G,
                self.biotsavart,
                derivatives=1,
                weight_inv_modB=weight_inv_modB,
                I=I,
                include_mixed_derivatives=False,
            )
        rtil = r / sqrt_n
        Jtil = J / sqrt_n
        if self.include_label_constraint:
            constraint_scale = np.sqrt(self.constraint_weight)
            label_residual = self.boozer_surface.label.J() - self.boozer_surface.targetlabel
            rtil = np.concatenate(
                (
                    rtil,
                    [constraint_scale * label_residual],
                )
            )
            dl = np.zeros((J.shape[1],))
            dlabel_dsurface = self.boozer_surface.label.dJ_by_dsurfacecoefficients()
            dl[: dlabel_dsurface.size] = dlabel_dsurface
            Jtil = np.concatenate(
                (Jtil, constraint_scale * dl[None, :]),
                axis=0,
            )
        self._J = 0.5 * np.sum(rtil**2)

        dJ_by_dB = _boozer_residual_dJ_by_dB(r, r_dB, sqrt_n)
        dJ_by_dcoils = self.biotsavart.B_vjp(dJ_by_dB)

        booz_surf = self.boozer_surface
        P, L, U = booz_surf.res["PLU"]
        dJ_ds = Jtil.T @ rtil
        adj = forward_backward(P, L, U, dJ_ds)
        if booz_surf.res["type"] == "ls":
            adj_times_dg_dcoil = _boozer_lsqgrad_vjp_from_residual_state(
                adj,
                self.biotsavart,
                r,
                r_dB,
                J,
                d2r_dsdB,
                d2r_dsdgradB,
                sqrt_n,
            )
        else:
            adj_times_dg_dcoil = booz_surf.res["vjp"](adj, booz_surf, iota, G)
        self._dJ = dJ_by_dcoils - adj_times_dg_dcoil

    def dJ_by_dB(self):
        """
        Return the partial derivative of the objective with respect to B.
        """

        surface = self.surface
        sqrt_n = np.sqrt(_num_boozer_components(surface))
        I = _resolve_boozer_current_I(self.boozer_surface)
        r, r_dB = boozer_surface_residual_dB(
            surface,
            self.boozer_surface.res["iota"],
            self.boozer_surface.res["G"],
            self.biotsavart,
            derivatives=0,
            weight_inv_modB=self._weight_inv_modB(),
            I=I,
        )

        return _boozer_residual_dJ_by_dB(r, r_dB, sqrt_n)


class BoozerResidualExact(RefinedBoozerResidual):
    def __init__(self, boozer_surface, bs):
        super().__init__(
            boozer_surface,
            bs,
            grid_multiplier=4,
            include_label_constraint=False,
            weight_inv_modB=True,
        )
