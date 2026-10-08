"""Native ``NonQuasiSymmetricRatio`` with its math in JAX.

:class:`JaxNonQuasiSymmetricRatio` takes what native
``NonQuasiSymmetricRatio`` takes, with a
:class:`~simsopt_jax_adapters.field.JaxBiotSavart` field, and gives native's
value and coil gradient, so ``examples/2_Intermediate/boozerQA.py`` runs in
JAX by swapping two constructors::

    boozer_surface = JaxBoozerSurface(JaxBiotSavart(bs.coils), s, vol, vol_target)
    J_nonQSRatio = JaxNonQuasiSymmetricRatio(boozer_surface, JaxBiotSavart(bs.coils))

Native ``Iotas`` and ``MajorRadius`` need no JAX version: their own math is a
few host operations, and their solve and adjoint (``res['PLU']``,
``res['vjp']``) are already those of
:class:`~simsopt_jax_adapters.geo.boozer_surface.JaxBoozerSurface`. For
``BoozerResidual`` use that module's ``JaxBoozerResidual``.

This module imports the field adapters, so it is not re-exported from
:mod:`simsopt_jax_adapters.geo`. Like native, each objective instance owns
mutable caches and belongs to one thread.
"""

from __future__ import annotations

import numpy as np

from simsopt._core.derivative import derivative_dec
from simsopt._core.optimizable import Optimizable
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.objectives.utilities import forward_backward
from simsopt_jax.core.quasisymmetry import non_quasi_symmetric_ratio
from simsopt_jax.runtime.host_boundary import host_tree
from simsopt_jax_adapters.field.biotsavart_backend import JaxBiotSavart

from .boozer_surface import JaxBoozerSurface
from .surface_specs import surface_spec_from_surface

__all__ = ["JaxNonQuasiSymmetricRatio"]


class JaxNonQuasiSymmetricRatio(Optimizable):
    """Native ``NonQuasiSymmetricRatio(boozer_surface, bs, sDIM,
    quasi_poloidal)`` with ``bs`` a ``JaxBiotSavart``.

    As natively, the ratio is evaluated on ``surface``, a
    ``SurfaceXYZTensorFourier`` sharing the Boozer surface's DOFs on a
    ``2 sDIM x 2 sDIM`` grid over one field period, after re-solving when
    ``boozer_surface`` needs it; ``dJ`` subtracts ``res['vjp']`` of the
    ``res['PLU']`` adjoint from the direct coil derivative. ``surface``,
    ``biotsavart``, ``axis`` and ``boozer_surface`` are read at every
    evaluation; ``boozer_surface`` may be a ``JaxBoozerSurface`` or a native
    ``BoozerSurface``. Unlike native, ``bs``'s evaluation points are left as
    they were, and ``bs`` is a parent alongside ``boozer_surface``: coils
    that only ``bs`` holds are part of ``x`` and ``dJ`` and invalidate ``J``
    when they change.

    Args:
        boozer_surface (JaxBoozerSurface | BoozerSurface): Mutable tensor-surface solver
            with geometry shape (nphi, ntheta, 3); its solve/adjoint supplies implicit
            coil dependence.
        bs (JaxBiotSavart): Field for the ratio; also a parent so its exclusive coils
            enter x and invalidate caches.
        sDIM (int): Half of each auxiliary grid dimension; grid spans one field period.
        quasi_poloidal (bool): Average over theta when true, otherwise over phi.
    """

    def __init__(
        self,
        boozer_surface: JaxBoozerSurface | BoozerSurface,
        bs: JaxBiotSavart,
        sDIM: int = 20,
        quasi_poloidal: bool = False,
    ):
        # only SurfaceXYZTensorFourier, as natively
        assert type(boozer_surface.surface) is SurfaceXYZTensorFourier
        if not isinstance(bs, JaxBiotSavart):
            raise TypeError(f"JaxNonQuasiSymmetricRatio needs a JaxBiotSavart field, got {type(bs).__name__}.")
        Optimizable.__init__(self, depends_on=[boozer_surface, bs])
        in_surface = boozer_surface.surface
        self.boozer_surface = boozer_surface
        self.surface = SurfaceXYZTensorFourier(
            mpol=in_surface.mpol,
            ntor=in_surface.ntor,
            stellsym=in_surface.stellsym,
            nfp=in_surface.nfp,
            quadpoints_phi=np.linspace(0, 1 / in_surface.nfp, 2 * sDIM, endpoint=False),
            quadpoints_theta=np.linspace(0, 1.0, 2 * sDIM, endpoint=False),
            dofs=in_surface.dofs,
        )
        self.axis = 1 if quasi_poloidal else 0
        self.in_surface = in_surface
        self.biotsavart = bs
        self.recompute_bell()

    def recompute_bell(self, parent=None):
        """Invalidate the cached computation after a parent changes.

        Args:
            parent (Optimizable | None): Parent notifying this objective of changed DOFs;
                unused.

        Returns:
            None: Clears the cached value and derivative.
        """
        self._J = None
        self._dJ = None

    def J(self):
        """Return the cached objective value, computing it when needed.


        Returns:
            float: Cached scalar objective, recomputed through the surface solve when
                invalidated; dimensionless for the non-QS ratio, native penalty units for
                BoozerResidual.
        """
        if self._J is None:
            self.compute()
        return self._J

    @derivative_dec
    def dJ(self):
        """Return the total coil derivative through the surface solve.


        Returns:
            numpy.ndarray | Derivative: Total coil derivative including the surface-solve
                adjoint. By default the decorator projects to shape (nfree,) in the
                objective's free DOF order; partials=True returns Derivative.
        """
        if self._dJ is None:
            self.compute()
        return self._dJ

    def compute(self):
        """Compute the objective and its direct-minus-adjoint coil derivative.


        Returns:
            None: Populates the objective value and total coil derivative caches, re-solving
                the Boozer surface when required.
        """
        booz_surf = self.boozer_surface
        if booz_surf.need_to_run_code:
            res = booz_surf.res
            booz_surf.run_code(res["iota"], G=res["G"])

        value, dJ_by_dsurface, dJ_by_dcoils = non_quasi_symmetric_ratio(
            surface_spec_from_surface(self.surface), self.biotsavart.coil_set_spec(), axis=self.axis
        )
        value, dJ_by_dsurface = host_tree((value, dJ_by_dsurface))
        self._J = value[()]

        # Native's adjoint: dJ/diota = dJ/dG = 0 after the surface DOFs.
        res = booz_surf.res
        P, L, U = res["PLU"]
        direct = self.biotsavart.coil_cotangents_to_derivative(
            dJ_by_dcoils.field_inputs(), dJ_by_dcoils.coil_index_lists()
        )
        dJ_ds = np.zeros(L.shape[0])
        dJ_ds[: dJ_by_dsurface.size] = dJ_by_dsurface
        adj = forward_backward(P, L, U, dJ_ds)
        self._dJ = direct - res["vjp"](adj, booz_surf, res["iota"], res["G"])
