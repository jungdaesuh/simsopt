"""
JAX-backed Optimizable wrappers for single-stage objectives.

These wrappers mirror the CPU ``BoozerResidual``, ``Iotas``, and
``NonQuasiSymmetricRatio`` classes but use JAX for field evaluation
and gradient computation.

Architecture (implicit differentiation):

  For any outer objective J that depends on the inner Boozer solution
  x*(coils), the total derivative is:

  .. math::

      \\frac{dJ}{d\\text{coils}} = \\frac{\\partial J}{\\partial \\text{coils}}
      - \\text{adj}^T \\frac{\\partial g}{\\partial \\text{coils}}

  where adj solves ``(PLU)^T adj = ∂J/∂x_inner`` and g is the
  stationarity condition of the inner solve.

  The PLU factorization and VJP hooks come from ``BoozerSurfaceJAX``'s
  ``run_code()`` result dict (Milestone 4).
"""

import numpy as np
import jax
import jax.numpy as jnp

from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec
from ..objectives.utilities import forward_backward
from ..field.biotsavart_jax import grouped_biot_savart_B
from .boozer_residual_jax import (
    boozer_residual_vector,
    _surface_geometry_from_dofs,
)
from .boozersurface_jax import _compute_label
from .label_constraints_jax import compute_G_from_currents
from .surfacexyztensorfourier import SurfaceXYZTensorFourier

__all__ = [
    "BoozerResidualJAX",
    "IotasJAX",
    "NonQuasiSymmetricRatioJAX",
]


def _coil_cotangents_to_derivative(coils, d_coil_arrays, coil_indices):
    """Convert grouped coil cotangent arrays to a ``Derivative``.

    Maps per-group cotangent tuples back to individual coil DOFs
    via ``Coil.vjp()``, using ``coil_indices`` to recover the
    original coil ordering.

    Args:
        coils: list of ``Coil`` objects.
        d_coil_arrays: list of ``(d_gammas, d_gammadashs, d_currents)``
            cotangent tuples, one per quadrature group.
        coil_indices: list of index lists, one per group, mapping
            local position to global coil index.

    Returns:
        ``Derivative`` over all coil DOFs.
    """
    all_derivs = []
    for (d_g, d_gd, d_c), indices in zip(d_coil_arrays, coil_indices):
        dg = np.asarray(d_g)
        dgd = np.asarray(d_gd)
        dc = np.asarray(d_c)
        for local_i, global_i in enumerate(indices):
            all_derivs.append(
                coils[global_i].vjp(
                    dg[local_i], dgd[local_i], np.asarray([dc[local_i]])
                )
            )
    return sum(all_derivs)


def _ensure_solved(booz_surf):
    """Re-run the Boozer inner solve if the surface is dirty."""
    if booz_surf.need_to_run_code:
        if booz_surf.res is None:
            raise RuntimeError(
                "BoozerSurfaceJAX has not been solved yet. "
                "Call boozer_surface.run_code(iota, G=G) before "
                "accessing objective values."
            )
        booz_surf.run_code(booz_surf.res["iota"], G=booz_surf.res["G"])


def _qs_ratio_pure(
    sdofs,
    coil_arrays,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    axis,
):
    """Pure JAX QS ratio: ``mean(dS * B_nonQS^2) / mean(dS * B_QS^2)``.

    Fully traceable by ``jax.grad`` / ``jax.vjp``.

    Args:
        coil_arrays: list of ``(gammas, gammadashs, currents)`` tuples.
    """
    gamma, xphi, xtheta = _surface_geometry_from_dofs(
        sdofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
    )
    normal = jnp.cross(xphi, xtheta)
    dS = jnp.sqrt(jnp.sum(normal**2, axis=-1))

    nphi, ntheta = gamma.shape[:2]
    points = gamma.reshape(-1, 3)
    B = grouped_biot_savart_B(points, coil_arrays)
    B = B.reshape(nphi, ntheta, 3)
    modB = jnp.sqrt(jnp.sum(B**2, axis=-1))

    B_QS = jnp.mean(modB * dS, axis=axis) / jnp.mean(dS, axis=axis)

    # Broadcast back to (nphi, ntheta)
    B_QS = jnp.expand_dims(B_QS, axis=axis)

    B_nonQS = modB - B_QS
    return jnp.mean(dS * B_nonQS**2) / jnp.mean(dS * B_QS**2)


def _boozer_residual_J_of_x_inner(
    x_inner,
    coil_arrays,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    optimize_G,
    weight_inv_modB,
    constraint_weight,
    targetlabel,
    label_type,
    phi_idx,
):
    """BoozerResidual outer objective as a function of inner DOFs.

    Used to compute ``∂J_BR/∂x_inner`` via ``jax.grad`` for the
    adjoint system.

    Args:
        coil_arrays: list of ``(gammas, gammadashs, currents)`` tuples.
    """
    if optimize_G:
        sdofs, iota, G = x_inner[:-2], x_inner[-2], x_inner[-1]
    else:
        sdofs, iota = x_inner[:-1], x_inner[-1]
        G = compute_G_from_currents(jnp.concatenate([c for _, _, c in coil_arrays]))

    gamma, xphi, xtheta = _surface_geometry_from_dofs(
        sdofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
    )
    nphi, ntheta = gamma.shape[:2]
    num_points = 3 * nphi * ntheta

    points = gamma.reshape(-1, 3)
    B = grouped_biot_savart_B(points, coil_arrays).reshape(nphi, ntheta, 3)

    r_flat = boozer_residual_vector(G, iota, B, xphi, xtheta, weight_inv_modB)
    J_boozer = 0.5 * jnp.sum(r_flat**2) / num_points

    label_val = _compute_label(
        label_type,
        gamma,
        xphi,
        xtheta,
        phi_idx,
        points,
        coil_arrays,
    )
    J_label = 0.5 * constraint_weight * (label_val - targetlabel) ** 2
    return J_boozer + J_label


class BoozerResidualJAX(Optimizable):
    r"""JAX equivalent of ``BoozerResidual``.

    Computes

    .. math::

        J = \frac{1}{2N}\|\mathbf r\|^2
            + \frac{w}{2}(\text{label} - \text{target})^2

    and the gradient w.r.t. coil DOFs via implicit differentiation.

    Args:
        boozer_surface: ``BoozerSurfaceJAX`` instance.
        biotsavart: ``BiotSavartJAX`` instance.
    """

    def __init__(self, boozer_surface, biotsavart):
        Optimizable.__init__(self, depends_on=[boozer_surface])
        self.boozer_surface = boozer_surface
        self.biotsavart = biotsavart
        self.in_surface = boozer_surface.surface

        # Auxiliary surface (same quadrature, independent DOF copy)
        s = self.in_surface
        self.surface = SurfaceXYZTensorFourier(
            mpol=s.mpol,
            ntor=s.ntor,
            stellsym=s.stellsym,
            nfp=s.nfp,
            quadpoints_phi=s.quadpoints_phi,
            quadpoints_theta=s.quadpoints_theta,
        )
        self.surface.set_dofs(s.get_dofs())

        self.constraint_weight = boozer_surface.constraint_weight
        self.recompute_bell()

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    def J(self):
        if self._J is None:
            self.compute()
        return self._J

    @derivative_dec
    def dJ(self):
        if self._dJ is None:
            self.compute()
        return self._dJ

    def compute(self):
        booz_surf = self.boozer_surface
        _ensure_solved(booz_surf)

        self.surface.set_dofs(self.in_surface.get_dofs())
        self.biotsavart.set_points(self.surface.gamma().reshape((-1, 3)))

        nphi = self.surface.quadpoints_phi.size
        ntheta = self.surface.quadpoints_theta.size
        num_points = 3 * nphi * ntheta

        iota = booz_surf.res["iota"]
        G = booz_surf.res["G"]
        weight_inv_modB = booz_surf.res.get("weight_inv_modB", True)
        cw = self.constraint_weight if self.constraint_weight is not None else 1.0

        xphi_jax = jnp.asarray(self.surface.gammadash1())
        xtheta_jax = jnp.asarray(self.surface.gammadash2())

        B = self.biotsavart.B()
        B_3d = B.reshape(nphi, ntheta, 3)

        r_flat = boozer_residual_vector(
            G, iota, B_3d, xphi_jax, xtheta_jax, weight_inv_modB
        )
        r = np.asarray(r_flat) / np.sqrt(num_points)

        label_val = float(booz_surf.label.J())
        rl = np.sqrt(cw) * (label_val - booz_surf.targetlabel)

        rtil = np.concatenate([r, [rl]])
        self._J = 0.5 * np.sum(rtil**2)

        P, L, U = booz_surf.res["PLU"]
        vjp_fn = booz_surf.res["vjp"]

        dJ_by_dB = self._compute_dJ_by_dB(
            B_3d,
            xphi_jax,
            xtheta_jax,
            iota,
            G,
            weight_inv_modB,
            nphi,
            ntheta,
            num_points,
        )
        dJ_by_dcoils = self.biotsavart.B_vjp(dJ_by_dB)

        dJ_ds = self._compute_dJ_ds(iota, G, weight_inv_modB, cw, nphi, ntheta)
        adj = forward_backward(P, L, U, dJ_ds)

        d_coil_arrays, coil_indices = vjp_fn(adj, booz_surf, iota, G)
        adj_derivative = _coil_cotangents_to_derivative(
            self.biotsavart.coils, d_coil_arrays, coil_indices
        )

        self._dJ = dJ_by_dcoils - adj_derivative

    def _compute_dJ_by_dB(
        self,
        B_3d,
        xphi,
        xtheta,
        iota,
        G,
        weight_inv_modB,
        nphi,
        ntheta,
        num_points,
    ):
        """Compute ∂J_boozer/∂B via JAX autodiff."""

        def J_of_B_flat(B_flat):
            Bv = B_flat.reshape(nphi, ntheta, 3)
            rv = boozer_residual_vector(G, iota, Bv, xphi, xtheta, weight_inv_modB)
            return 0.5 * jnp.sum(rv**2) / num_points

        B_flat = B_3d.reshape(-1)
        dJ_dB = jax.grad(J_of_B_flat)(B_flat)
        return np.asarray(dJ_dB).reshape(-1, 3)

    def _compute_dJ_ds(self, iota, G, weight_inv_modB, cw, nphi, ntheta):
        """Compute ∂J_BR/∂[surface_dofs, iota, G] via JAX autodiff."""
        booz_surf = self.boozer_surface
        sdofs = booz_surf._get_surface_dofs()
        optimize_G = G is not None

        if optimize_G:
            x_inner = jnp.concatenate([sdofs, jnp.array([iota, G])])
        else:
            x_inner = jnp.concatenate([sdofs, jnp.array([iota])])

        coil_arrays = booz_surf._coil_arrays
        dJ_ds_jax = jax.grad(_boozer_residual_J_of_x_inner)(
            x_inner,
            coil_arrays=coil_arrays,
            quadpoints_phi=booz_surf.quadpoints_phi,
            quadpoints_theta=booz_surf.quadpoints_theta,
            mpol=booz_surf.mpol,
            ntor=booz_surf.ntor,
            nfp=booz_surf.nfp,
            stellsym=booz_surf.stellsym,
            scatter_indices=booz_surf.scatter_indices,
            optimize_G=optimize_G,
            weight_inv_modB=weight_inv_modB,
            constraint_weight=cw,
            targetlabel=booz_surf.targetlabel,
            label_type=booz_surf.label_type,
            phi_idx=booz_surf.phi_idx,
        )
        return np.asarray(dJ_ds_jax)


class IotasJAX(Optimizable):
    """JAX equivalent of ``Iotas``.

    Returns the rotational transform on the Boozer surface and its
    gradient w.r.t. coil DOFs via the adjoint (no direct B term).

    Args:
        boozer_surface: ``BoozerSurfaceJAX`` instance.
    """

    def __init__(self, boozer_surface):
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[boozer_surface])
        self.boozer_surface = boozer_surface
        self.biotsavart = boozer_surface.biotsavart
        self.recompute_bell()

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    def J(self):
        if self._J is None:
            self.compute()
        return self._J

    @derivative_dec
    def dJ(self):
        if self._dJ is None:
            self.compute()
        return self._dJ

    def compute(self):
        booz_surf = self.boozer_surface
        _ensure_solved(booz_surf)

        iota = booz_surf.res["iota"]
        G = booz_surf.res["G"]
        self._J = iota
        P, L, U = booz_surf.res["PLU"]
        vjp_fn = booz_surf.res["vjp"]

        # dJ/dx_inner for iota: unit vector at the iota position
        n = L.shape[0]
        dJ_ds = np.zeros(n)
        if G is not None:
            dJ_ds[-2] = 1.0  # [surface_dofs..., iota, G]
        else:
            dJ_ds[-1] = 1.0  # [surface_dofs..., iota]

        adj = forward_backward(P, L, U, dJ_ds)

        d_coil_arrays, coil_indices = vjp_fn(adj, booz_surf, iota, G)
        adj_derivative = _coil_cotangents_to_derivative(
            self.biotsavart.coils, d_coil_arrays, coil_indices
        )

        self._dJ = -1.0 * adj_derivative


class NonQuasiSymmetricRatioJAX(Optimizable):
    r"""JAX equivalent of ``NonQuasiSymmetricRatio``.

    Computes

    .. math::

        J = \frac{\langle dS\, B_{\text{nonQS}}^2 \rangle}
                 {\langle dS\, B_{\text{QS}}^2 \rangle}

    on an auxiliary surface with finer quadrature, and the gradient
    w.r.t. coil DOFs via implicit differentiation.

    Args:
        boozer_surface: ``BoozerSurfaceJAX`` instance.
        biotsavart: ``BiotSavartJAX`` instance.
        sDIM: half-resolution of auxiliary quadrature grid.
        quasi_poloidal: ``True`` for quasi-poloidal, ``False`` for
            quasi-axisymmetric.
    """

    def __init__(self, boozer_surface, biotsavart, sDIM=20, quasi_poloidal=False):
        Optimizable.__init__(self, depends_on=[boozer_surface])
        self.boozer_surface = boozer_surface
        self.biotsavart = biotsavart
        self.axis = 1 if quasi_poloidal else 0
        self.in_surface = boozer_surface.surface

        # Auxiliary surface with finer quadrature (matches CPU)
        s = self.in_surface
        aux_phi = np.linspace(0, 1 / s.nfp, 2 * sDIM, endpoint=False)
        aux_theta = np.linspace(0, 1.0, 2 * sDIM, endpoint=False)
        self.surface = SurfaceXYZTensorFourier(
            mpol=s.mpol,
            ntor=s.ntor,
            stellsym=s.stellsym,
            nfp=s.nfp,
            quadpoints_phi=aux_phi,
            quadpoints_theta=aux_theta,
            dofs=s.dofs,
        )
        self._aux_phi_jax = jnp.asarray(aux_phi)
        self._aux_theta_jax = jnp.asarray(aux_theta)

        self.recompute_bell()

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    def J(self):
        if self._J is None:
            self.compute()
        return self._J

    @derivative_dec
    def dJ(self):
        if self._dJ is None:
            self.compute()
        return self._dJ

    def compute(self):
        booz_surf = self.boozer_surface
        _ensure_solved(booz_surf)

        self.surface.set_dofs(self.in_surface.get_dofs())

        iota = booz_surf.res["iota"]
        G = booz_surf.res["G"]
        P, L, U = booz_surf.res["PLU"]
        vjp_fn = booz_surf.res["vjp"]

        sdofs = booz_surf._get_surface_dofs()
        coil_arrays = booz_surf._coil_arrays
        coil_indices = booz_surf._coil_index_lists

        qs_kwargs = dict(
            quadpoints_phi=self._aux_phi_jax,
            quadpoints_theta=self._aux_theta_jax,
            mpol=booz_surf.mpol,
            ntor=booz_surf.ntor,
            nfp=booz_surf.nfp,
            stellsym=booz_surf.stellsym,
            scatter_indices=booz_surf.scatter_indices,
            axis=self.axis,
        )

        self._J = float(_qs_ratio_pure(sdofs, coil_arrays, **qs_kwargs))

        def J_of_coils(ca):
            return _qs_ratio_pure(sdofs, ca, **qs_kwargs)

        d_coil_arrays_direct = jax.grad(J_of_coils)(coil_arrays)
        dJ_by_dcoils = _coil_cotangents_to_derivative(
            self.biotsavart.coils,
            d_coil_arrays_direct,
            coil_indices,
        )

        def J_of_sdofs(s):
            return _qs_ratio_pure(s, coil_arrays, **qs_kwargs)

        dJ_ds_surface = np.asarray(jax.grad(J_of_sdofs)(sdofs))

        n = L.shape[0]
        dJ_ds = np.zeros(n)
        dJ_ds[: dJ_ds_surface.size] = dJ_ds_surface

        adj = forward_backward(P, L, U, dJ_ds)

        d_coil_arrays_adj, coil_indices_adj = vjp_fn(adj, booz_surf, iota, G)
        adj_derivative = _coil_cotangents_to_derivative(
            self.biotsavart.coils, d_coil_arrays_adj, coil_indices_adj
        )

        self._dJ = dJ_by_dcoils - adj_derivative
