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

from .._core.derivative import Derivative, derivative_dec
from .._core.optimizable import Optimizable
from ..jax_core.field import (
    grouped_biot_savart_B_from_spec,
    grouped_coil_currents_from_spec,
)
from ..objectives.utilities import forward_backward_jax, plu_solve_jax
from .boozer_residual_jax import (
    boozer_residual_vector,
    _surface_geometry_from_dofs,
)
from .boozersurface_jax import (
    _boozer_penalty_objective,
    _boozer_exact_residual,
    _compute_label,
    _make_boozer_penalty_objective_closure,
)
from .label_constraints_jax import compute_G_from_currents

__all__ = [
    "BoozerResidualJAX",
    "IotasJAX",
    "NonQuasiSymmetricRatioJAX",
    "make_traceable_objective",
]

_MISSING_STREAMING_GROUP_VJP_ERROR = (
    "BoozerSurfaceJAX objective wrappers require res['vjp_groups']; "
    "the legacy full-pytree adjoint fallback is no longer supported."
)
_LEGACY_PROJECTION_HELPER_ERROR = (
    "surfaceobjectives_jax._coil_cotangents_to_derivative() is no longer "
    "supported; use BiotSavartJAX.coil_cotangents_to_derivative()."
)


def _solve_boozer_adjoint(booz_surf, rhs):
    """Solve the transposed PLU adjoint system for a BoozerSurfaceJAX result."""
    P, L, U = booz_surf.res["PLU"]
    return forward_backward_jax(P, L, U, rhs, iterative_refinement=True)


def _iter_adjoint_coil_cotangents(vjp_groups_fn, booz_surf, iota, G, adjoint):
    """Yield grouped coil cotangents from the streaming adjoint callback."""
    if vjp_groups_fn is None:
        raise RuntimeError(_MISSING_STREAMING_GROUP_VJP_ERROR)
    yield from vjp_groups_fn(adjoint, booz_surf, iota, G)


def _coil_cotangents_to_derivative(coils, d_coil_arrays, coil_indices):
    """Deprecated compatibility helper kept only as an explicit hard-fail seam."""
    del coils, d_coil_arrays, coil_indices
    raise RuntimeError(_LEGACY_PROJECTION_HELPER_ERROR)


def _adjoint_coil_derivative(vjp_groups_fn, booz_surf, iota, G, adjoint, biotsavart):
    """Project grouped adjoint cotangents to a coil ``Derivative``.

    Uses ``BiotSavartJAX.coil_cotangents_to_derivative()`` for
    shared coil DOF projection. JAX-capable curves stay on the JAX
    projection path; unsupported curves fall back to ``Coil.vjp()``
    slice by slice.
    """
    total_derivative = Derivative({})
    for d_coil_array, coil_group_indices in _iter_adjoint_coil_cotangents(
        vjp_groups_fn, booz_surf, iota, G, adjoint
    ):
        total_derivative += biotsavart.coil_cotangents_to_derivative(
            [d_coil_array],
            [coil_group_indices],
        )
    return total_derivative


def _coil_dofs_gradient_to_derivative(biotsavart, coil_dofs_gradient):
    """Convert a flat free-DOF gradient into the public ``Derivative`` contract."""
    coil_dofs_gradient = np.asarray(coil_dofs_gradient, dtype=float)
    deriv_data = {}
    start = 0
    for lineage_opt in biotsavart.unique_dof_lineage:
        width = lineage_opt.local_dof_size
        if width == 0:
            continue

        block = np.zeros(lineage_opt.local_full_dof_size)
        stop = start + width
        block[lineage_opt.local_dofs_free_status] = coil_dofs_gradient[start:stop]
        start = stop

        dep_opts = tuple(lineage_opt.dofs.dep_opts())
        block_share = block / len(dep_opts)
        for dep_opt in dep_opts:
            if dep_opt in deriv_data:
                deriv_data[dep_opt] = deriv_data[dep_opt] + block_share
            else:
                deriv_data[dep_opt] = block_share.copy()

    return Derivative(deriv_data)


def _current_coil_dofs_and_spec(biotsavart):
    """Return the current free coil DOFs and their immutable grouped spec."""
    current_coil_dofs = jnp.asarray(biotsavart.x.copy(), dtype=jnp.float64)
    return current_coil_dofs, biotsavart.coil_set_spec_from_dofs(current_coil_dofs)


def _value_and_direct_coil_derivative(biotsavart, objective_of_coils, coil_dofs):
    """Evaluate a coil-DOF objective and map its direct gradient to Derivative."""
    objective_value, coil_dofs_gradient = jax.value_and_grad(objective_of_coils)(
        coil_dofs
    )
    direct_derivative = _coil_dofs_gradient_to_derivative(
        biotsavart,
        coil_dofs_gradient,
    )
    return float(objective_value), direct_derivative


def _qs_ratio_from_coil_dofs(sdofs, coil_dofs, biotsavart, **qs_kwargs):
    """Evaluate the QS-ratio objective from explicit coil DOFs via immutable specs."""
    return _qs_ratio_pure(
        sdofs,
        biotsavart.coil_set_spec_from_dofs(coil_dofs),
        **qs_kwargs,
    )


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
    if (
        booz_surf.res is None
        or not booz_surf.res.get("success")
        or "PLU" not in booz_surf.res
        or booz_surf.res["PLU"] is None
        or "vjp" not in booz_surf.res
        or booz_surf.res["vjp"] is None
    ):
        raise RuntimeError(
            "BoozerSurfaceJAX has not been solved yet or the last solve failed "
            "to produce valid adjoint state."
        )


def _qs_ratio_pure(
    sdofs,
    coil_set_spec,
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
    B = grouped_biot_savart_B_from_spec(points, coil_set_spec)
    B = B.reshape(nphi, ntheta, 3)
    modB = jnp.sqrt(jnp.sum(B**2, axis=-1))

    B_QS = jnp.mean(modB * dS, axis=axis) / jnp.mean(dS, axis=axis)

    # Broadcast back to (nphi, ntheta)
    B_QS = jnp.expand_dims(B_QS, axis=axis)

    B_nonQS = modB - B_QS
    return jnp.mean(dS * B_nonQS**2) / jnp.mean(dS * B_QS**2)


def _boozer_residual_J_of_x_inner(
    x_inner,
    coil_set_spec,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    surface_kind,
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
        coil_set_spec: immutable grouped-coil geometry/current payload.
    """
    if optimize_G:
        sdofs, iota, G = x_inner[:-2], x_inner[-2], x_inner[-1]
    else:
        sdofs, iota = x_inner[:-1], x_inner[-1]
        G = compute_G_from_currents(grouped_coil_currents_from_spec(coil_set_spec))

    gamma, xphi, xtheta = _surface_geometry_from_dofs(
        sdofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
        surface_kind=surface_kind,
    )
    nphi, ntheta = gamma.shape[:2]
    num_points = 3 * nphi * ntheta

    points = gamma.reshape(-1, 3)
    B = grouped_biot_savart_B_from_spec(points, coil_set_spec).reshape(
        nphi,
        ntheta,
        3,
    )

    r_flat = boozer_residual_vector(G, iota, B, xphi, xtheta, weight_inv_modB)
    J_boozer = 0.5 * jnp.sum(r_flat**2) / num_points

    label_val = _compute_label(
        label_type,
        gamma,
        xphi,
        xtheta,
        phi_idx,
        points,
        coil_set_spec=coil_set_spec,
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
        if boozer_surface.boozer_type != "ls":
            raise ValueError(
                "BoozerResidualJAX requires a least-squares BoozerSurfaceJAX "
                "(constraint_weight must be set)."
            )
        self.boozer_surface = boozer_surface
        self.biotsavart = biotsavart
        self.in_surface = boozer_surface.surface

        # Auxiliary surface (same quadrature, independent DOF copy)
        from .surfacexyztensorfourier import SurfaceXYZTensorFourier

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

        sdofs = booz_surf._get_surface_dofs()
        iota = booz_surf.res["iota"]
        G = booz_surf.res["G"]
        weight_inv_modB = booz_surf.res.get("weight_inv_modB", True)
        x_inner = booz_surf._pack_decision_vector(iota, G, sdofs=sdofs)
        current_coil_dofs, coil_set_spec = _current_coil_dofs_and_spec(self.biotsavart)

        def objective_of_coils(coil_dofs):
            return _boozer_penalty_objective(
                x_inner,
                coil_set_spec=self.biotsavart.coil_set_spec_from_dofs(coil_dofs),
                quadpoints_phi=booz_surf.quadpoints_phi,
                quadpoints_theta=booz_surf.quadpoints_theta,
                mpol=booz_surf.mpol,
                ntor=booz_surf.ntor,
                nfp=booz_surf.nfp,
                stellsym=booz_surf.stellsym,
                scatter_indices=booz_surf.scatter_indices,
                surface_kind=booz_surf._surface_geometry_kind,
                targetlabel=booz_surf.targetlabel,
                constraint_weight=self.constraint_weight,
                label_type=booz_surf.label_type,
                phi_idx=booz_surf.phi_idx,
                optimize_G=G is not None,
                weight_inv_modB=weight_inv_modB,
            )

        self._J, dJ_by_dcoils = _value_and_direct_coil_derivative(
            self.biotsavart,
            objective_of_coils,
            current_coil_dofs,
        )
        vjp_groups_fn = booz_surf.res.get("vjp_groups")

        dJ_ds = self._compute_dJ_ds(coil_set_spec, iota, G, weight_inv_modB)
        adj = _solve_boozer_adjoint(booz_surf, dJ_ds)

        adj_derivative = _adjoint_coil_derivative(
            vjp_groups_fn,
            booz_surf,
            iota,
            G,
            adj,
            self.biotsavart,
        )

        self._dJ = dJ_by_dcoils - adj_derivative

    def _compute_dJ_ds(self, coil_set_spec, iota, G, weight_inv_modB):
        """Compute ∂J_BR/∂[surface_dofs, iota, G] via JAX autodiff."""
        booz_surf = self.boozer_surface
        sdofs = booz_surf._get_surface_dofs()
        optimize_G = G is not None
        constraint_weight = (
            self.constraint_weight if self.constraint_weight is not None else 1.0
        )

        if optimize_G:
            x_inner = jnp.concatenate([sdofs, jnp.array([iota, G])])
        else:
            x_inner = jnp.concatenate([sdofs, jnp.array([iota])])

        dJ_ds_jax = jax.grad(_boozer_residual_J_of_x_inner)(
            x_inner,
            coil_set_spec=coil_set_spec,
            quadpoints_phi=booz_surf.quadpoints_phi,
            quadpoints_theta=booz_surf.quadpoints_theta,
            mpol=booz_surf.mpol,
            ntor=booz_surf.ntor,
            nfp=booz_surf.nfp,
            stellsym=booz_surf.stellsym,
            scatter_indices=booz_surf.scatter_indices,
            surface_kind=booz_surf._surface_geometry_kind,
            optimize_G=optimize_G,
            weight_inv_modB=weight_inv_modB,
            constraint_weight=constraint_weight,
            targetlabel=booz_surf.targetlabel,
            label_type=booz_surf.label_type,
            phi_idx=booz_surf.phi_idx,
        )
        return dJ_ds_jax


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
        vjp_groups_fn = booz_surf.res.get("vjp_groups")

        # dJ/dx_inner for iota: unit vector at the iota position
        L = booz_surf.res["PLU"][1]
        n = L.shape[0]
        dJ_ds = jnp.zeros(n, dtype=jnp.asarray(L).dtype)
        if G is not None:
            dJ_ds = dJ_ds.at[-2].set(1.0)  # [surface_dofs..., iota, G]
        else:
            dJ_ds = dJ_ds.at[-1].set(1.0)  # [surface_dofs..., iota]

        adj = _solve_boozer_adjoint(booz_surf, dJ_ds)

        adj_derivative = _adjoint_coil_derivative(
            vjp_groups_fn,
            booz_surf,
            iota,
            G,
            adj,
            self.biotsavart,
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
        from .surfacexyztensorfourier import SurfaceXYZTensorFourier

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
        vjp_groups_fn = booz_surf.res.get("vjp_groups")

        sdofs = booz_surf._get_surface_dofs()
        current_coil_dofs, coil_set_spec = _current_coil_dofs_and_spec(self.biotsavart)

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

        self._J = float(_qs_ratio_pure(sdofs, coil_set_spec, **qs_kwargs))

        def J_of_coils(coil_dofs):
            return _qs_ratio_from_coil_dofs(
                sdofs,
                coil_dofs,
                self.biotsavart,
                **qs_kwargs,
            )

        dJ_by_dcoils = _coil_dofs_gradient_to_derivative(
            self.biotsavart,
            jax.grad(J_of_coils)(current_coil_dofs),
        )

        def J_of_sdofs(s):
            return _qs_ratio_pure(s, coil_set_spec, **qs_kwargs)

        dJ_ds_surface = jax.grad(J_of_sdofs)(sdofs)

        n = booz_surf.res["PLU"][1].shape[0]
        dJ_ds = jnp.zeros(n, dtype=dJ_ds_surface.dtype)
        dJ_ds = dJ_ds.at[: dJ_ds_surface.size].set(dJ_ds_surface)

        adj = _solve_boozer_adjoint(booz_surf, dJ_ds)

        adj_derivative = _adjoint_coil_derivative(
            vjp_groups_fn,
            booz_surf,
            iota,
            G,
            adj,
            self.biotsavart,
        )

        self._dJ = dJ_by_dcoils - adj_derivative


def _traceable_iota_from_x_inner(x_inner, optimize_G):
    """Extract iota from the inner decision vector."""
    return x_inner[-2] if optimize_G else x_inner[-1]


def _traceable_iota_target_penalty(x_inner, *, optimize_G, iota_target):
    """Quadratic iota-target penalty at an explicit inner state."""
    iota = _traceable_iota_from_x_inner(x_inner, optimize_G)
    return 0.5 * (iota - iota_target) ** 2


_TRACEABLE_INNER_OBJECTIVE_KEYS = (
    "quadpoints_phi",
    "quadpoints_theta",
    "mpol",
    "ntor",
    "nfp",
    "stellsym",
    "scatter_indices",
    "surface_kind",
    "targetlabel",
    "constraint_weight",
    "label_type",
    "phi_idx",
    "optimize_G",
    "weight_inv_modB",
)

_TRACEABLE_TOTAL_OBJECTIVE_KEYS = (
    "quadpoints_phi",
    "quadpoints_theta",
    "mpol",
    "ntor",
    "nfp",
    "stellsym",
    "scatter_indices",
    "surface_kind",
    "optimize_G",
    "weight_inv_modB",
    "constraint_weight",
    "targetlabel",
    "label_type",
    "phi_idx",
    "iota_target",
)

_TRACEABLE_EXACT_RESIDUAL_KEYS = (
    "quadpoints_phi",
    "quadpoints_theta",
    "mpol",
    "ntor",
    "nfp",
    "stellsym",
    "scatter_indices",
    "surface_kind",
    "targetlabel",
    "label_type",
    "phi_idx",
    "mask_indices",
    "stellsym_surface",
    "weight_inv_modB",
)


def _traceable_inner_objective_kwargs(objective_kwargs):
    """Select the LS inner-objective kwargs from the full traceable contract."""
    return {key: objective_kwargs[key] for key in _TRACEABLE_INNER_OBJECTIVE_KEYS}


def _traceable_total_objective_kwargs(objective_kwargs):
    """Select the scalar total-objective kwargs from the full traceable contract."""
    return {key: objective_kwargs[key] for key in _TRACEABLE_TOTAL_OBJECTIVE_KEYS}


def _traceable_exact_residual_kwargs(objective_kwargs):
    """Select the exact-residual kwargs from the full traceable contract."""
    return {key: objective_kwargs[key] for key in _TRACEABLE_EXACT_RESIDUAL_KEYS}


def _traceable_total_objective(
    x_inner,
    coil_set_spec,
    *,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    surface_kind,
    optimize_G,
    weight_inv_modB,
    constraint_weight,
    targetlabel,
    label_type,
    phi_idx,
    iota_target,
):
    """Pure single-stage objective evaluated at an explicit inner state."""
    J_boozer = _boozer_residual_J_of_x_inner(
        x_inner,
        coil_set_spec=coil_set_spec,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
        scatter_indices=scatter_indices,
        surface_kind=surface_kind,
        optimize_G=optimize_G,
        weight_inv_modB=weight_inv_modB,
        constraint_weight=constraint_weight,
        targetlabel=targetlabel,
        label_type=label_type,
        phi_idx=phi_idx,
    )
    return J_boozer + _traceable_iota_target_penalty(
        x_inner,
        optimize_G=optimize_G,
        iota_target=iota_target,
    )


def _evaluate_traceable_total_objective(x_inner, coil_set_spec, objective_kwargs):
    """Evaluate the full traceable scalar objective from packed kwargs."""
    return _traceable_total_objective(
        x_inner,
        coil_set_spec,
        **_traceable_total_objective_kwargs(objective_kwargs),
    )


def _traceable_directional_inner_objective(
    x_inner,
    tangent,
    coil_set_spec,
    **objective_kwargs,
):
    """Directional derivative of the LS inner objective at an explicit state."""
    inner_objective = _make_boozer_penalty_objective_closure(
        coil_set_spec=coil_set_spec,
        **objective_kwargs,
    )
    return jax.jvp(inner_objective, (x_inner,), (tangent,))[1]


def _traceable_forward_result(
    booz_jax,
    bs_jax,
    *,
    coil_dofs,
    baseline_x,
    baseline_value,
    baseline_plu,
    optimize_G,
    baseline_coil_dofs,
    failure_value,
    failure_scale,
    predictor_kind,
    objective_kwargs,
):
    """Run the pure traceable inner solve and return value plus solver data."""
    same_coils = jnp.all(coil_dofs == baseline_coil_dofs)

    def baseline_case(_):
        return {
            "value": baseline_value,
            "x": baseline_x,
            "plu": baseline_plu,
            "success": jnp.array(True, dtype=bool),
        }

    def general_case(_):
        coil_set_spec = bs_jax.coil_set_spec_from_dofs(coil_dofs)
        warmstart_x = _traceable_predict_warmstart_x(
            booz_jax,
            bs_jax,
            coil_dofs=coil_dofs,
            baseline_coil_dofs=baseline_coil_dofs,
            baseline_x=baseline_x,
            baseline_plu=baseline_plu,
            predictor_kind=predictor_kind,
            objective_kwargs=objective_kwargs,
        )
        warmstart_sdofs, warmstart_iota, warmstart_G = (
            booz_jax._unpack_decision_vector_jax(
                warmstart_x,
                optimize_G,
                coil_set_spec=coil_set_spec,
            )
        )
        solve_result = booz_jax.run_code_traceable(
            coil_set_spec,
            warmstart_sdofs,
            warmstart_iota,
            warmstart_G,
        )
        delta = coil_dofs - baseline_coil_dofs
        failure_penalty = failure_value + 0.5 * failure_scale * jnp.dot(delta, delta)
        return {
            "value": jnp.where(
                solve_result["success"],
                _evaluate_traceable_total_objective(
                    solve_result["x"],
                    coil_set_spec,
                    objective_kwargs,
                ),
                failure_penalty,
            ),
            "x": solve_result["x"],
            "plu": solve_result["plu"],
            "success": solve_result["success"],
        }

    return jax.lax.cond(same_coils, baseline_case, general_case, operand=None)


def _traceable_total_gradient(
    booz_jax,
    bs_jax,
    *,
    coil_dofs,
    solved_x,
    solved_plu,
    objective_kwargs,
):
    """Implicit total derivative of the pure traceable objective."""
    inner_objective_kwargs = _traceable_inner_objective_kwargs(objective_kwargs)

    def total_of_coils(cd):
        return _evaluate_traceable_total_objective(
            solved_x,
            bs_jax.coil_set_spec_from_dofs(cd),
            objective_kwargs,
        )

    coil_set_spec = bs_jax.coil_set_spec_from_dofs(coil_dofs)
    dJ_dx = jax.grad(
        lambda x: _evaluate_traceable_total_objective(
            x,
            coil_set_spec,
            objective_kwargs,
        )
    )(solved_x)
    adjoint = forward_backward_jax(*solved_plu, dJ_dx, iterative_refinement=True)

    def directional_of_coils(cd):
        return _traceable_directional_inner_objective(
            solved_x,
            adjoint,
            bs_jax.coil_set_spec_from_dofs(cd),
            **inner_objective_kwargs,
        )

    direct_grad = jax.grad(total_of_coils)(coil_dofs)
    implicit_grad = jax.grad(directional_of_coils)(coil_dofs)
    return direct_grad - implicit_grad


def _traceable_predict_warmstart_x(
    booz_jax,
    bs_jax,
    *,
    coil_dofs,
    baseline_coil_dofs,
    baseline_x,
    baseline_plu,
    predictor_kind,
    objective_kwargs,
):
    """Predict a coil-dependent warm start via a first-order implicit step."""
    delta = coil_dofs - baseline_coil_dofs

    if predictor_kind == "exact":
        exact_residual_kwargs = _traceable_exact_residual_kwargs(objective_kwargs)

        def baseline_residual_of_coils(cd):
            return _boozer_exact_residual(
                baseline_x,
                coil_set_spec=bs_jax.coil_set_spec_from_dofs(cd),
                **exact_residual_kwargs,
            )

        forcing = jax.jvp(
            baseline_residual_of_coils,
            (baseline_coil_dofs,),
            (delta,),
        )[1]
    else:
        inner_objective_kwargs = _traceable_inner_objective_kwargs(objective_kwargs)

        def baseline_stationarity_of_coils(cd):
            inner_objective = _make_boozer_penalty_objective_closure(
                coil_set_spec=bs_jax.coil_set_spec_from_dofs(cd),
                **inner_objective_kwargs,
            )
            return jax.grad(inner_objective)(baseline_x)

        forcing = jax.jvp(
            baseline_stationarity_of_coils,
            (baseline_coil_dofs,),
            (delta,),
        )[1]

    dx = plu_solve_jax(*baseline_plu, -forcing, iterative_refinement=True)
    return baseline_x + dx


def make_traceable_objective(booz_jax, bs_jax, iota_target):
    """Build a pure function ``f(coil_dofs) -> scalar`` for single-stage optimization.

    The returned closure:

    * **Forward**: re-solves the inner Boozer problem from a coil-dependent
      linearized warm-start predictor and returns the exact
      single-stage scalar objective
      ``BoozerResidualJAX + 0.5 * (iota - iota_target)^2``.
    * **No object mutation**: coil geometry is reconstructed directly from
      the explicit ``coil_dofs`` vector, so the traced objective does not
      touch ``bs_jax.x``, ``booz_jax.res``, or descendant Optimizable caches.
    * **No callback seam**: the traced path stays inside JAX primitives;
      there is no ``jax.pure_callback`` bridge back into the stateful
      ``run_code()`` implementation.
    * **Backward**: uses the same implicit-differentiation structure as the
      validated object path, but expressed entirely with pure JAX arrays.

    Args:
        booz_jax: solved :class:`BoozerSurfaceJAX`.
        bs_jax:   :class:`BiotSavartJAX` providing coil geometry.
        iota_target: scalar target iota for the quadratic penalty.

    Returns:
        ``f(coil_dofs) -> jax.Array`` — traceable scalar objective.
    """
    _ensure_solved(booz_jax)

    if booz_jax.boozer_type == "ls":
        objective_method = booz_jax._resolve_optimizer_method()
        if objective_method not in {"bfgs-ondevice", "lbfgs-ondevice"}:
            raise RuntimeError(
                "make_traceable_objective() requires optimizer_backend='ondevice'."
            )

    warmstart_sdofs = jnp.asarray(booz_jax.surface.get_dofs(), dtype=jnp.float64)
    warmstart_iota = jnp.asarray(booz_jax.res["iota"], dtype=jnp.float64)
    warmstart_G = booz_jax.res["G"]
    if warmstart_G is not None:
        warmstart_G = jnp.asarray(warmstart_G, dtype=jnp.float64)

    baseline_coil_dofs = jnp.asarray(bs_jax.x.copy(), dtype=jnp.float64)
    optimize_G = warmstart_G is not None
    predictor_kind = booz_jax.boozer_type
    mask_indices = booz_jax._compute_stellsym_mask_indices()
    objective_kwargs = {
        "quadpoints_phi": booz_jax.quadpoints_phi,
        "quadpoints_theta": booz_jax.quadpoints_theta,
        "mpol": booz_jax.mpol,
        "ntor": booz_jax.ntor,
        "nfp": booz_jax.nfp,
        "stellsym": booz_jax.stellsym,
        "scatter_indices": booz_jax.scatter_indices,
        "surface_kind": booz_jax._surface_geometry_kind,
        "optimize_G": optimize_G,
        "weight_inv_modB": booz_jax.options["weight_inv_modB"],
        "constraint_weight": booz_jax.constraint_weight,
        "targetlabel": booz_jax.targetlabel,
        "label_type": booz_jax.label_type,
        "phi_idx": booz_jax.phi_idx,
        "iota_target": jnp.asarray(iota_target, dtype=jnp.float64),
        "mask_indices": mask_indices,
        "stellsym_surface": booz_jax.stellsym,
    }
    baseline_plu = booz_jax.res["PLU"]

    baseline_x = booz_jax._pack_decision_vector(
        float(warmstart_iota),
        None if warmstart_G is None else float(warmstart_G),
        sdofs=warmstart_sdofs,
    )

    baseline_value = float(
        jax.jit(
            lambda x, coil_set_spec: _evaluate_traceable_total_objective(
                x,
                coil_set_spec,
                objective_kwargs,
            )
        )(
            baseline_x,
            bs_jax.coil_set_spec_from_dofs(baseline_coil_dofs),
        )
    )
    failure_value = jnp.asarray(
        baseline_value + max(abs(baseline_value), 1.0),
        dtype=jnp.float64,
    )
    failure_scale = jnp.asarray(1.0, dtype=jnp.float64)

    def _forward_result_for(coil_dofs):
        return _traceable_forward_result(
            booz_jax,
            bs_jax,
            coil_dofs=coil_dofs,
            baseline_x=baseline_x,
            baseline_value=jnp.asarray(baseline_value, dtype=jnp.float64),
            baseline_plu=baseline_plu,
            optimize_G=optimize_G,
            baseline_coil_dofs=baseline_coil_dofs,
            failure_value=failure_value,
            failure_scale=failure_scale,
            predictor_kind=predictor_kind,
            objective_kwargs=objective_kwargs,
        )

    compiled_forward_result_for = jax.jit(_forward_result_for)

    def _total_gradient_for(coil_dofs, solved_x, solved_plu):
        return _traceable_total_gradient(
            booz_jax,
            bs_jax,
            coil_dofs=coil_dofs,
            solved_x=solved_x,
            solved_plu=solved_plu,
            objective_kwargs=objective_kwargs,
        )

    compiled_total_gradient_for = jax.jit(_total_gradient_for)

    @jax.custom_vjp
    def f(coil_dofs):
        coil_dofs = jnp.asarray(coil_dofs, dtype=jnp.float64)
        return compiled_forward_result_for(coil_dofs)["value"]

    def f_fwd(coil_dofs):
        coil_dofs = jnp.asarray(coil_dofs, dtype=jnp.float64)
        result = compiled_forward_result_for(coil_dofs)
        return result["value"], (
            coil_dofs,
            result["x"],
            result["plu"],
            result["success"],
        )

    def f_bwd(saved_state, cotangent):
        coil_dofs, solved_x, solved_plu, success = saved_state

        def _success(_):
            return compiled_total_gradient_for(coil_dofs, solved_x, solved_plu)

        def _failure(_):
            return failure_scale * (coil_dofs - baseline_coil_dofs)

        grad = jax.lax.cond(success, _success, _failure, operand=None)
        return (jnp.asarray(cotangent, dtype=grad.dtype) * grad,)

    f.defvjp(f_fwd, f_bwd)

    return f
