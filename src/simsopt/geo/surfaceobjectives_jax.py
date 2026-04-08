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

import hashlib
import numpy as np
import jax
import jax.numpy as jnp

from .._core.derivative import Derivative, derivative_dec
from .._core.jax_host_boundary import (
    explicit_cotangent_basis as _explicit_cotangent_basis,
    host_scalar as _host_scalar,
    scalar_pullback_seed as _explicit_scalar_pullback_seed,
)
from .._core.optimizable import Optimizable
from ..jax_core._math_utils import (
    as_runtime_float64 as _as_runtime_float64,
    zeros as _zeros,
)
from ..jax_core.field import (
    grouped_biot_savart_B_from_spec,
    coil_set_spec_from_dof_extraction_spec,
    grouped_coil_currents_from_spec,
)
from ..jax_core.sharding import inspect_array_sharding_summary
from ..objectives.utilities import forward_backward_jax, plu_solve_jax
from .boozer_residual_jax import (
    boozer_residual_vector,
    _surface_geometry_from_dofs,
)
from .boozersurface_jax import (
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
    "make_traceable_objective_runtime_bundle",
    "make_traceable_objective_value_and_grad",
    "make_traceable_objective_profile_suite",
]

_MISSING_STREAMING_GROUP_VJP_ERROR = (
    "BoozerSurfaceJAX objective wrappers require res['vjp_groups']; "
    "the legacy full-pytree adjoint fallback is no longer supported."
)
_LEGACY_PROJECTION_HELPER_ERROR = (
    "surfaceobjectives_jax._coil_cotangents_to_derivative() is no longer "
    "supported; use BiotSavartJAX.coil_cotangents_to_derivative()."
)
_TRACEABLE_RUNTIME_OPTION_KEYS = (
    "optimizer_backend",
    "least_squares_algorithm",
    "limited_memory",
    "force_ondevice_limited_memory",
    "weight_inv_modB",
    "bfgs_maxiter",
    "bfgs_tol",
    "newton_maxiter",
    "newton_tol",
    "newton_stab",
)


def _strict_scalar_grad(fun, arg):
    value, pullback = jax.vjp(fun, arg)
    (gradient,) = pullback(_explicit_scalar_pullback_seed(value))
    return gradient


def _strict_scalar_value_and_grad(fun, arg, *args):
    def _objective(first_arg):
        return fun(first_arg, *args)

    value, pullback = jax.vjp(_objective, arg)
    (gradient,) = pullback(_explicit_scalar_pullback_seed(value))
    return value, gradient


def _explicit_index_array(indices):
    return jax.device_put(np.asarray(indices, dtype=np.int32))


def _take_runtime_entries(array, indices):
    indices = np.asarray(indices, dtype=np.int32)
    if indices.size == 0:
        return _zeros(0, dtype=array.dtype)
    return jnp.take(array, _explicit_index_array(indices), axis=0)


def _take_runtime_scalar(array, index):
    return jnp.reshape(
        _take_runtime_entries(array, np.array([int(index)], dtype=np.int32)),
        (),
    )


def _split_x_inner_runtime(x_inner, optimize_G):
    length = int(x_inner.shape[0])
    sdof_count = length - (2 if optimize_G else 1)
    sdofs = _take_runtime_entries(x_inner, np.arange(sdof_count, dtype=np.int32))
    iota = _take_runtime_scalar(x_inner, sdof_count)
    if optimize_G:
        return sdofs, iota, _take_runtime_scalar(x_inner, sdof_count + 1)
    return sdofs, iota, None


def _runtime_float64_scalar(value, *, reference):
    return _as_runtime_float64(value, reference=reference)


def _surface_stellsym_mask_for_grid(
    *,
    ntor,
    mpol,
    nfp,
    stellsym,
    quadpoints_phi,
    quadpoints_theta,
):
    phis = np.asarray(quadpoints_phi, dtype=float)
    thetas = np.asarray(quadpoints_theta, dtype=float)
    mask = np.ones((phis.size, thetas.size), dtype=bool)
    if not stellsym:
        return mask

    def _same_grid(lhs, rhs):
        return lhs.shape == rhs.shape and np.allclose(lhs, rhs)

    full_phi = np.linspace(0.0, 1.0 / nfp, 2 * ntor + 1, endpoint=False)
    full_theta = np.linspace(0.0, 1.0, 2 * mpol + 1, endpoint=False)
    half_theta = np.linspace(0.0, 0.5, mpol + 1, endpoint=False)
    half_phi = np.linspace(0.0, 1.0 / (2.0 * nfp), ntor + 1, endpoint=False)

    if _same_grid(phis, full_phi) and _same_grid(thetas, full_theta):
        mask[:, mpol + 1 :] = False
        mask[ntor + 1 :, 0] = False
        return mask
    if _same_grid(phis, full_phi) and _same_grid(thetas, half_theta):
        mask[ntor + 1 :, 0] = False
        return mask
    if _same_grid(phis, half_phi) and _same_grid(thetas, full_theta):
        mask[0, mpol + 1 :] = False
        return mask
    raise Exception(
        "Stellarator symmetric BoozerExact surfaces require a specific set of "
        "quadrature points on the surface. See the "
        "SurfaceXYZTensorFourier.get_stellsym_mask() docstring for more "
        "information."
    )


def _compute_stellsym_mask_indices_for_grid(
    *,
    mpol,
    ntor,
    nfp,
    stellsym,
    quadpoints_phi,
    quadpoints_theta,
):
    """Return exact-residual mask indices for a specific surface quadrature."""
    mask = np.repeat(
        _surface_stellsym_mask_for_grid(
            mpol=mpol,
            ntor=ntor,
            nfp=nfp,
            stellsym=stellsym,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )[..., None],
        3,
        axis=2,
    )
    if stellsym:
        mask[0, 0, 0] = False
    return jnp.asarray(np.flatnonzero(mask), dtype=jnp.int32)


def _canonicalize_traceable_exact_quadrature(booz_jax):
    """Return exact-compatible quadrature for the traceable scalar objective.

    Real single-stage fixtures often initialize Boozer least-squares surfaces
    on the VMEC half-period integration grid. That grid uses half-cell-shifted
    phi points for spectral quadrature, so it is valid for the solve but does
    not match the unshifted quadrature families accepted by
    ``SurfaceXYZTensorFourier.get_stellsym_mask()``. The traceable objective is
    evaluated from surface DOFs, so it can safely canonicalize to an exact
    quadrature family when the input surface uses a shifted integration grid.
    """
    quadpoints_phi = np.asarray(booz_jax.quadpoints_phi, dtype=float)
    quadpoints_theta = np.asarray(booz_jax.quadpoints_theta, dtype=float)

    def _mask_indices_for(phi_grid, theta_grid):
        return _compute_stellsym_mask_indices_for_grid(
            mpol=booz_jax.mpol,
            ntor=booz_jax.ntor,
            nfp=booz_jax.nfp,
            stellsym=booz_jax.stellsym,
            quadpoints_phi=phi_grid,
            quadpoints_theta=theta_grid,
        )

    try:
        mask_indices = _mask_indices_for(quadpoints_phi, quadpoints_theta)
    except Exception:
        phi_max = float(np.max(quadpoints_phi)) if quadpoints_phi.size else 0.0
        half_period_upper = 0.5 / float(booz_jax.nfp)
        if phi_max <= half_period_upper + 1e-12:
            quadpoints_phi = np.linspace(
                0.0,
                half_period_upper,
                int(booz_jax.ntor) + 1,
                endpoint=False,
            )
        else:
            quadpoints_phi = np.linspace(
                0.0,
                1.0 / float(booz_jax.nfp),
                2 * int(booz_jax.ntor) + 1,
                endpoint=False,
            )
        quadpoints_theta = np.linspace(
            0.0,
            1.0,
            2 * int(booz_jax.mpol) + 1,
            endpoint=False,
        )
        mask_indices = _mask_indices_for(quadpoints_phi, quadpoints_theta)

    return (
        jnp.asarray(quadpoints_phi, dtype=jnp.float64),
        jnp.asarray(quadpoints_theta, dtype=jnp.float64),
        mask_indices,
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
    coil_dofs_gradient = np.asarray(jax.device_get(coil_dofs_gradient), dtype=float)
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


def _make_cached_strict_scalar_value_and_grad(fun):
    """Cache a strict scalar value/grad callable behind a stable helper contract."""

    def value_and_grad(arg, *args):
        return _strict_scalar_value_and_grad(fun, arg, *args)

    value_and_grad._simsopt_value_and_grad = True
    return value_and_grad


def _traceable_cache_leaf_signature(leaf):
    """Build a deterministic cache signature for one traceable-runtime leaf."""
    if isinstance(leaf, (jax.Array, np.ndarray)):
        array = np.asarray(jax.device_get(leaf))
        return (
            "array",
            str(array.dtype),
            tuple(array.shape),
            hashlib.blake2b(array.tobytes(), digest_size=16).hexdigest(),
        )
    if isinstance(leaf, np.generic):
        return ("numpy_scalar", str(leaf.dtype), leaf.item())
    if isinstance(leaf, (str, int, float, bool, type(None))):
        return ("scalar", leaf)
    return ("repr", type(leaf).__qualname__, repr(leaf))


def _traceable_cache_tree_signature(tree):
    """Build a deterministic cache signature for a pytree-like runtime object."""
    try:
        leaves, treedef = jax.tree_util.tree_flatten(tree)
    except TypeError:
        return _traceable_cache_leaf_signature(tree)
    return (
        "tree",
        repr(treedef),
        tuple(_traceable_cache_leaf_signature(leaf) for leaf in leaves),
    )


def _evaluate_scalar_or_value_and_grad(
    objective_or_value_and_grad,
    coil_dofs,
    *objective_args,
):
    """Evaluate either a cached value/grad callable or a scalar objective."""
    if getattr(objective_or_value_and_grad, "_simsopt_value_and_grad", False):
        return objective_or_value_and_grad(coil_dofs, *objective_args)
    return _strict_scalar_value_and_grad(
        objective_or_value_and_grad,
        coil_dofs,
        *objective_args,
    )


def _current_coil_dofs_and_spec(biotsavart):
    """Return the current free coil DOFs and their immutable grouped spec."""
    current_coil_dofs = jnp.asarray(biotsavart.x.copy(), dtype=jnp.float64)
    return current_coil_dofs, biotsavart.coil_set_spec_from_dofs(current_coil_dofs)


def _value_and_direct_coil_derivative(
    biotsavart,
    objective_or_value_and_grad,
    coil_dofs,
    *objective_args,
):
    """Evaluate a cached coil-DOF objective/gradient pair and map its gradient."""
    objective_value, coil_dofs_gradient = _evaluate_scalar_or_value_and_grad(
        objective_or_value_and_grad,
        coil_dofs,
        *objective_args,
    )
    direct_derivative = _coil_dofs_gradient_to_derivative(
        biotsavart,
        coil_dofs_gradient,
    )
    return _host_scalar(objective_value), direct_derivative


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
    surface_kind,
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
        surface_kind=surface_kind,
    )
    normal = jnp.cross(xphi, xtheta)
    dS = jnp.sqrt(jnp.sum(normal * normal, axis=-1))

    nphi, ntheta = gamma.shape[:2]
    points = gamma.reshape(-1, 3)
    B = grouped_biot_savart_B_from_spec(points, coil_set_spec)
    B = B.reshape(nphi, ntheta, 3)
    modB = jnp.sqrt(jnp.sum(B * B, axis=-1))

    B_QS = jnp.sum(modB * dS, axis=axis) / jnp.sum(dS, axis=axis)

    # Broadcast back to (nphi, ntheta)
    B_QS = jnp.expand_dims(B_QS, axis=axis)

    B_nonQS = modB - B_QS
    return jnp.sum(dS * (B_nonQS * B_nonQS)) / jnp.sum(dS * (B_QS * B_QS))


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
    sdofs, iota, G = _split_x_inner_runtime(x_inner, optimize_G)
    if not optimize_G:
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
    half = _runtime_float64_scalar(0.5, reference=r_flat)
    num_points_jax = _runtime_float64_scalar(float(num_points), reference=r_flat)
    J_boozer = half * jnp.sum(r_flat * r_flat) / num_points_jax

    label_val = _compute_label(
        label_type,
        gamma,
        xphi,
        xtheta,
        phi_idx,
        points,
        coil_set_spec=coil_set_spec,
    )
    targetlabel_jax = _runtime_float64_scalar(targetlabel, reference=label_val)
    constraint_weight_jax = _runtime_float64_scalar(
        constraint_weight,
        reference=label_val,
    )
    label_delta = label_val - targetlabel_jax
    J_label = half * constraint_weight_jax * (label_delta * label_delta)
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
        self.surface = self.in_surface

        self.constraint_weight = float(boozer_surface.constraint_weight)
        self._direct_objective_value_and_grad = _make_cached_strict_scalar_value_and_grad(
            self._direct_objective_of_coils
        )
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

    def _direct_objective_of_coils(
        self,
        coil_dofs,
        x_inner,
        optimize_G,
        weight_inv_modB,
    ):
        """Pure direct BoozerResidual objective evaluated from explicit coil DOFs."""
        return _boozer_residual_J_of_x_inner(
            x_inner,
            coil_set_spec=self.biotsavart.coil_set_spec_from_dofs(coil_dofs),
            **self._residual_objective_kwargs(
                optimize_G=optimize_G,
                weight_inv_modB=weight_inv_modB,
            ),
        )

    def _inner_objective_state(self, iota, G, *, sdofs=None):
        """Return the packed inner decision vector and optimize-G flag."""
        surface_dofs = (
            self.boozer_surface._get_surface_dofs() if sdofs is None else sdofs
        )
        optimize_G = G is not None
        return (
            self.boozer_surface._pack_decision_vector(iota, G, sdofs=surface_dofs),
            optimize_G,
        )

    def compute(self):
        booz_surf = self.boozer_surface
        _ensure_solved(booz_surf)

        sdofs = booz_surf._get_surface_dofs()
        iota = booz_surf.res["iota"]
        G = booz_surf.res["G"]
        weight_inv_modB = booz_surf.res.get("weight_inv_modB", True)
        x_inner, optimize_G = self._inner_objective_state(iota, G, sdofs=sdofs)
        current_coil_dofs, coil_set_spec = _current_coil_dofs_and_spec(self.biotsavart)

        self._J, dJ_by_dcoils = _value_and_direct_coil_derivative(
            self.biotsavart,
            self._direct_objective_value_and_grad,
            current_coil_dofs,
            x_inner,
            optimize_G,
            weight_inv_modB,
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
        x_inner, optimize_G = self._inner_objective_state(iota, G)

        def objective(x):
            return _boozer_residual_J_of_x_inner(
                x,
                coil_set_spec=coil_set_spec,
                **self._residual_objective_kwargs(
                    optimize_G=optimize_G,
                    weight_inv_modB=weight_inv_modB,
                ),
            )

        dJ_ds_jax = _strict_scalar_grad(
            objective,
            x_inner,
        )
        return dJ_ds_jax

    def _residual_objective_kwargs(self, *, optimize_G, weight_inv_modB):
        booz_surf = self.boozer_surface
        return dict(
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
            targetlabel=booz_surf.targetlabel,
            constraint_weight=self.constraint_weight,
            label_type=booz_surf.label_type,
            phi_idx=booz_surf.phi_idx,
        )


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
        if G is not None:
            dJ_ds = _explicit_cotangent_basis(n, n - 2, dtype=L.dtype)
        else:
            dJ_ds = _explicit_cotangent_basis(n, n - 1, dtype=L.dtype)

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

        s = self.in_surface
        aux_phi = np.linspace(0, 1 / s.nfp, 2 * sDIM, endpoint=False)
        aux_theta = np.linspace(0, 1.0, 2 * sDIM, endpoint=False)
        self.surface = self.in_surface
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
            surface_kind=booz_surf._surface_geometry_kind,
            axis=self.axis,
        )

        self._J = float(_host_scalar(_qs_ratio_pure(sdofs, coil_set_spec, **qs_kwargs)))

        def J_of_coils(coil_dofs):
            return _qs_ratio_from_coil_dofs(
                sdofs,
                coil_dofs,
                self.biotsavart,
                **qs_kwargs,
            )

        dJ_by_dcoils = _coil_dofs_gradient_to_derivative(
            self.biotsavart,
            _strict_scalar_grad(J_of_coils, current_coil_dofs),
        )

        def J_of_sdofs(s):
            return _qs_ratio_pure(s, coil_set_spec, **qs_kwargs)

        dJ_ds_surface = _strict_scalar_grad(J_of_sdofs, sdofs)

        n = booz_surf.res["PLU"][1].shape[0]
        dJ_ds = jnp.concatenate(
            (
                dJ_ds_surface,
                _zeros(n - dJ_ds_surface.size, dtype=dJ_ds_surface.dtype),
            )
        )

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
    _, iota, _ = _split_x_inner_runtime(x_inner, optimize_G)
    return iota


def _traceable_iota_target_penalty(x_inner, *, optimize_G, iota_target):
    """Quadratic iota-target penalty at an explicit inner state."""
    iota = _traceable_iota_from_x_inner(x_inner, optimize_G)
    half = _runtime_float64_scalar(0.5, reference=iota)
    iota_target_jax = _runtime_float64_scalar(iota_target, reference=iota)
    delta = iota - iota_target_jax
    return half * (delta * delta)


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
    coil_set_spec_from_dofs,
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
    success_filter,
):
    """Run the pure traceable inner solve and return value plus solver data."""
    same_coils = jnp.all(coil_dofs == baseline_coil_dofs)

    def baseline_case(_):
        success = jnp.array(True, dtype=bool)
        if success_filter is not None:
            success = success_filter(coil_dofs, baseline_x)
        return {
            "value": jnp.where(success, baseline_value, failure_value),
            "x": baseline_x,
            "plu": baseline_plu,
            "success": success,
        }

    def general_case(_):
        coil_set_spec = coil_set_spec_from_dofs(coil_dofs)
        warmstart_x = _traceable_predict_warmstart_x(
            booz_jax,
            coil_set_spec_from_dofs,
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
        success = solve_result["success"]
        if success_filter is not None:
            success = jax.lax.cond(
                solve_result["success"],
                lambda _: success_filter(coil_dofs, solve_result["x"]),
                lambda _: jnp.array(False, dtype=bool),
                operand=None,
            )
        delta = coil_dofs - baseline_coil_dofs
        failure_half = _runtime_float64_scalar(0.5, reference=delta)
        failure_value_jax = _runtime_float64_scalar(failure_value, reference=delta)
        failure_scale_jax = _runtime_float64_scalar(failure_scale, reference=delta)
        failure_penalty = (
            failure_value_jax
            + failure_half * failure_scale_jax * jnp.dot(delta, delta)
        )
        return {
            "value": jnp.where(
                success,
                _evaluate_traceable_total_objective(
                    solve_result["x"],
                    coil_set_spec,
                    objective_kwargs,
                ),
                failure_penalty,
            ),
            "x": solve_result["x"],
            "plu": solve_result["plu"],
            "success": success,
        }

    return jax.lax.cond(same_coils, baseline_case, general_case, operand=None)


def _traceable_total_gradient(
    booz_jax,
    coil_set_spec_from_dofs,
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
            coil_set_spec_from_dofs(cd),
            objective_kwargs,
        )

    coil_set_spec = coil_set_spec_from_dofs(coil_dofs)
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
            coil_set_spec_from_dofs(cd),
            **inner_objective_kwargs,
        )

    direct_grad = jax.grad(total_of_coils)(coil_dofs)
    implicit_grad = jax.grad(directional_of_coils)(coil_dofs)
    return direct_grad - implicit_grad


def _traceable_predict_warmstart_x(
    booz_jax,
    coil_set_spec_from_dofs,
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
                coil_set_spec=coil_set_spec_from_dofs(cd),
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
                coil_set_spec=coil_set_spec_from_dofs(cd),
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


def _build_traceable_objective_state(booz_jax, bs_jax, iota_target):
    """Return the shared state used by the traceable objective builders.

    This setup reads the solved mutable object state once, then keeps the
    warm-start and baseline objective data in explicit JAX arrays before
    building the compiled target-lane closures. The resulting closures are the
    trace-safe hot path; this helper itself is bootstrap code, not the compiled
    optimization loop.
    """
    _ensure_solved(booz_jax)

    if booz_jax.boozer_type == "ls":
        objective_method = booz_jax._resolve_optimizer_method()
        if objective_method not in {"bfgs-ondevice", "lbfgs-ondevice", "lm-ondevice"}:
            raise RuntimeError(
                "make_traceable_objective() requires optimizer_backend='ondevice'."
            )

    warmstart_sdofs = jnp.asarray(booz_jax.surface.get_dofs(), dtype=jnp.float64)
    warmstart_iota = jnp.asarray(booz_jax.res["iota"], dtype=jnp.float64)
    warmstart_G = booz_jax.res["G"]
    if warmstart_G is not None:
        warmstart_G = jnp.asarray(warmstart_G, dtype=jnp.float64)

    baseline_coil_dofs = jnp.asarray(bs_jax.x.copy(), dtype=jnp.float64)
    coil_dof_extraction_spec = bs_jax.coil_dof_extraction_spec()
    coil_set_spec_from_dofs = lambda coil_dofs: coil_set_spec_from_dof_extraction_spec(
        coil_dof_extraction_spec,
        coil_dofs,
    )
    optimize_G = warmstart_G is not None
    predictor_kind = booz_jax.boozer_type
    quadpoints_phi, quadpoints_theta, mask_indices = (
        _canonicalize_traceable_exact_quadrature(booz_jax)
    )
    objective_kwargs = {
        "quadpoints_phi": quadpoints_phi,
        "quadpoints_theta": quadpoints_theta,
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
        warmstart_iota,
        warmstart_G,
        sdofs=warmstart_sdofs,
    )

    baseline_value = _evaluate_traceable_total_objective(
        baseline_x,
        bs_jax.coil_set_spec_from_dofs(baseline_coil_dofs),
        objective_kwargs,
    )
    failure_value = jnp.asarray(
        baseline_value + jnp.maximum(jnp.abs(baseline_value), 1.0),
        dtype=jnp.float64,
    )
    failure_scale = jnp.asarray(1.0, dtype=jnp.float64)
    return {
        "objective_kwargs": objective_kwargs,
        "baseline_x": baseline_x,
        "baseline_value": baseline_value,
        "baseline_plu": baseline_plu,
        "baseline_coil_dofs": baseline_coil_dofs,
        "coil_dof_extraction_spec": coil_dof_extraction_spec,
        "coil_set_spec_from_dofs": coil_set_spec_from_dofs,
        "optimize_G": optimize_G,
        "predictor_kind": predictor_kind,
        "failure_value": failure_value,
        "failure_scale": failure_scale,
    }


def _build_traceable_objective_compiled_bundle_from_state(
    booz_jax,
    state,
    *,
    success_filter=None,
):
    """Build shared compiled closures for one traceable single-stage state."""
    objective_kwargs = state["objective_kwargs"]
    baseline_x = state["baseline_x"]
    baseline_value = state["baseline_value"]
    baseline_plu = state["baseline_plu"]
    baseline_coil_dofs = state["baseline_coil_dofs"]
    optimize_G = state["optimize_G"]
    predictor_kind = state["predictor_kind"]
    coil_set_spec_from_dofs = state["coil_set_spec_from_dofs"]
    failure_value = state["failure_value"]
    failure_scale = state["failure_scale"]

    def _forward_result_for(coil_dofs):
        return _traceable_forward_result(
            booz_jax,
            coil_set_spec_from_dofs,
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
            success_filter=success_filter,
        )

    compiled_forward_result_for = jax.jit(_forward_result_for)

    def _total_gradient_for(coil_dofs, solved_x, solved_plu):
        return _traceable_total_gradient(
            booz_jax,
            coil_set_spec_from_dofs,
            coil_dofs=coil_dofs,
            solved_x=solved_x,
            solved_plu=solved_plu,
            objective_kwargs=objective_kwargs,
        )

    compiled_total_gradient_for = jax.jit(_total_gradient_for)

    def _failure_gradient_for(coil_dofs):
        return failure_scale * (coil_dofs - baseline_coil_dofs)

    def _value_and_grad_for(coil_dofs):
        result = compiled_forward_result_for(coil_dofs)

        def _success(_):
            return compiled_total_gradient_for(
                coil_dofs,
                result["x"],
                result["plu"],
            )

        grad = jax.lax.cond(
            result["success"],
            _success,
            lambda _: _failure_gradient_for(coil_dofs),
            operand=None,
        )
        return result["value"], grad

    compiled_value_and_grad_for = jax.jit(_value_and_grad_for)

    return {
        "state": state,
        "compiled_forward_result_for": compiled_forward_result_for,
        "compiled_total_gradient_for": compiled_total_gradient_for,
        "compiled_value_and_grad_for": compiled_value_and_grad_for,
        "failure_gradient_for": _failure_gradient_for,
    }


def _traceable_runtime_option_signature(booz_jax):
    """Capture the solver options that affect traceable runtime compilation."""
    option_state = {
        key: booz_jax.options.get(key)
        for key in _TRACEABLE_RUNTIME_OPTION_KEYS
    }
    option_state["optimizer_options"] = booz_jax._collect_optimizer_options()
    return _traceable_cache_tree_signature(option_state)


def _traceable_runtime_cache_key(booz_jax, bs_jax, state, *, success_filter=None):
    """Return a stable cache key for one compiled traceable runtime state."""
    return (
        id(booz_jax),
        id(bs_jax),
        state["optimize_G"],
        state["predictor_kind"],
        _traceable_cache_tree_signature(state["coil_dof_extraction_spec"]),
        _traceable_cache_tree_signature(state["objective_kwargs"]),
        _traceable_cache_tree_signature(state["baseline_x"]),
        _traceable_cache_tree_signature(state["baseline_value"]),
        _traceable_cache_tree_signature(state["baseline_plu"]),
        _traceable_cache_tree_signature(state["baseline_coil_dofs"]),
        _traceable_cache_tree_signature(state["failure_value"]),
        _traceable_cache_tree_signature(state["failure_scale"]),
        _traceable_runtime_option_signature(booz_jax),
        None if success_filter is None else ("callable", id(success_filter)),
    )


def _get_cached_traceable_runtime_entry(
    booz_jax,
    bs_jax,
    iota_target,
    *,
    success_filter=None,
):
    """Reuse compiled traceable runtime callables while the solved state is unchanged."""
    state = _build_traceable_objective_state(booz_jax, bs_jax, iota_target)
    cache_key = _traceable_runtime_cache_key(
        booz_jax,
        bs_jax,
        state,
        success_filter=success_filter,
    )
    cached_entry = getattr(booz_jax, "_traceable_runtime_entry_cache", None)
    if cached_entry is not None and cached_entry["cache_key"] == cache_key:
        return cached_entry

    compiled_bundle = _build_traceable_objective_compiled_bundle_from_state(
        booz_jax,
        state,
        success_filter=success_filter,
    )
    cached_entry = {
        "cache_key": cache_key,
        "compiled_bundle": compiled_bundle,
        "objective": _make_traceable_objective_from_compiled_bundle(compiled_bundle),
        "profile_suite": None,
    }
    booz_jax._traceable_runtime_entry_cache = cached_entry
    return cached_entry


def _make_traceable_objective_from_compiled_bundle(compiled_bundle):
    """Build the scalar custom-VJP target-lane objective from one compiled bundle."""
    compiled_forward_result_for = compiled_bundle["compiled_forward_result_for"]
    compiled_total_gradient_for = compiled_bundle["compiled_total_gradient_for"]
    failure_gradient_for = compiled_bundle["failure_gradient_for"]

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
            return failure_gradient_for(coil_dofs)

        grad = jax.lax.cond(success, _success, _failure, operand=None)
        return (jnp.asarray(cotangent, dtype=grad.dtype) * grad,)

    f.defvjp(f_fwd, f_bwd)

    return f


def make_traceable_objective(
    booz_jax,
    bs_jax,
    iota_target,
    *,
    success_filter=None,
):
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
    return _get_cached_traceable_runtime_entry(
        booz_jax,
        bs_jax,
        iota_target,
        success_filter=success_filter,
    )["objective"]


def make_traceable_objective_value_and_grad(
    booz_jax,
    bs_jax,
    iota_target,
    *,
    success_filter=None,
):
    """Build a pure function ``f(coil_dofs) -> (value, grad)`` for ondevice L-BFGS.

    This is the fused outer-optimizer objective contract for the single-stage
    ondevice target lane. It shares the exact forward and implicit-gradient
    implementation used by :func:`make_traceable_objective`, but returns both
    outputs from one compiled entrypoint so the outer optimizer can avoid
    rebuilding autodiff transforms around a scalar objective.
    """
    return make_traceable_objective_runtime_bundle(
        booz_jax,
        bs_jax,
        iota_target,
        success_filter=success_filter,
    )["value_and_grad"]


def _make_traceable_forward_value_pipeline(compiled_forward_result_for):
    def _forward_value_for(coil_dofs):
        return compiled_forward_result_for(coil_dofs)["value"]

    return jax.jit(_forward_value_for)


def _make_traceable_field_eval_sharding_pipeline(field_at_solution_for):
    compiled_field_at_solution_for = jax.jit(field_at_solution_for)

    def _field_eval_sharding(coil_dofs):
        return inspect_array_sharding_summary(
            compiled_field_at_solution_for(coil_dofs)
        )

    return _field_eval_sharding


def _make_traceable_objective_profile_suite_from_compiled_bundle(
    compiled_bundle,
    booz_jax,
    bs_jax,
    *,
    value_and_grad_pipeline=None,
):
    """Build profiling closures from the shared traceable runtime bundle."""
    state = compiled_bundle["state"]
    objective_kwargs = state["objective_kwargs"]
    baseline_coil_dofs = state["baseline_coil_dofs"]
    baseline_x = state["baseline_x"]
    baseline_plu = state["baseline_plu"]
    optimize_G = state["optimize_G"]
    predictor_kind = state["predictor_kind"]
    coil_set_spec_from_dofs = state["coil_set_spec_from_dofs"]
    compiled_forward_result_for = compiled_bundle["compiled_forward_result_for"]
    resolved_value_and_grad_pipeline = (
        compiled_bundle["compiled_value_and_grad_for"]
        if value_and_grad_pipeline is None
        else value_and_grad_pipeline
    )

    def _warmstart_for(coil_dofs):
        return _traceable_predict_warmstart_x(
            booz_jax,
            coil_set_spec_from_dofs,
            coil_dofs=coil_dofs,
            baseline_coil_dofs=baseline_coil_dofs,
            baseline_x=baseline_x,
            baseline_plu=baseline_plu,
            predictor_kind=predictor_kind,
            objective_kwargs=objective_kwargs,
        )

    def _solve_for(coil_dofs):
        coil_set_spec = coil_set_spec_from_dofs(coil_dofs)
        warmstart_x = _warmstart_for(coil_dofs)
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
        return {
            "x": solve_result["x"],
            "sdofs": solve_result["sdofs"],
            "iota": solve_result["iota"],
            "G": solve_result["G"],
            "fun": solve_result["fun"],
            "plu": solve_result["plu"],
            "success": solve_result["success"],
            "nit": solve_result["nit"],
        }

    def _surface_geometry_for(solved_x):
        sdofs, _, _ = _split_x_inner_runtime(solved_x, optimize_G)
        return _surface_geometry_from_dofs(
            sdofs,
            objective_kwargs["quadpoints_phi"],
            objective_kwargs["quadpoints_theta"],
            objective_kwargs["mpol"],
            objective_kwargs["ntor"],
            objective_kwargs["nfp"],
            objective_kwargs["stellsym"],
            objective_kwargs["scatter_indices"],
            surface_kind=objective_kwargs["surface_kind"],
        )

    def _field_for(coil_dofs, solved_x):
        coil_set_spec = coil_set_spec_from_dofs(coil_dofs)
        gamma, _, _ = _surface_geometry_for(solved_x)
        points = gamma.reshape(-1, 3)
        return grouped_biot_savart_B_from_spec(points, coil_set_spec)

    def _field_at_solution_for(coil_dofs):
        return _field_for(coil_dofs, _solve_for(coil_dofs)["x"])

    def _solved_total_objective_for(coil_dofs, solved_x):
        return _evaluate_traceable_total_objective(
            solved_x,
            coil_set_spec_from_dofs(coil_dofs),
            objective_kwargs,
        )

    def _total_gradient_for(coil_dofs, solved_x, solved_plu):
        return _traceable_total_gradient(
            booz_jax,
            coil_set_spec_from_dofs,
            coil_dofs=coil_dofs,
            solved_x=solved_x,
            solved_plu=solved_plu,
            objective_kwargs=objective_kwargs,
        )

    compiled_forward_value_for = _make_traceable_forward_value_pipeline(
        compiled_forward_result_for
    )
    compiled_warmstart_for = jax.jit(_warmstart_for)
    compiled_inner_solve_for = jax.jit(_solve_for)
    compiled_surface_geometry_for = jax.jit(_surface_geometry_for)
    compiled_field_for = jax.jit(_field_for)
    compiled_field_eval_sharding = _make_traceable_field_eval_sharding_pipeline(
        _field_at_solution_for
    )
    compiled_solved_total_objective_for = jax.jit(_solved_total_objective_for)
    compiled_solved_total_gradient_for = jax.jit(_total_gradient_for)

    return {
        "forward_result": compiled_forward_result_for,
        "forward_value": compiled_forward_value_for,
        "warmstart_predict": compiled_warmstart_for,
        "inner_solve": compiled_inner_solve_for,
        "surface_geometry": compiled_surface_geometry_for,
        "field_eval": compiled_field_for,
        "field_eval_sharding": compiled_field_eval_sharding,
        "solved_total_objective": compiled_solved_total_objective_for,
        "solved_total_gradient": compiled_solved_total_gradient_for,
        "value_and_grad_pipeline": resolved_value_and_grad_pipeline,
    }


def make_traceable_objective_runtime_bundle(
    booz_jax,
    bs_jax,
    iota_target,
    *,
    include_profile_suite=False,
    success_filter=None,
):
    """Build the shared runtime bundle for the target single-stage objective path."""
    runtime_entry = _get_cached_traceable_runtime_entry(
        booz_jax,
        bs_jax,
        iota_target,
        success_filter=success_filter,
    )
    compiled_bundle = runtime_entry["compiled_bundle"]
    compiled_value_and_grad_for = compiled_bundle["compiled_value_and_grad_for"]
    if not include_profile_suite:
        return {
            "objective": runtime_entry["objective"],
            "value_and_grad": compiled_value_and_grad_for,
        }
    if runtime_entry["profile_suite"] is None:
        runtime_entry["profile_suite"] = (
            _make_traceable_objective_profile_suite_from_compiled_bundle(
                compiled_bundle,
                booz_jax,
                bs_jax,
                value_and_grad_pipeline=compiled_value_and_grad_for,
            )
        )
    return {
        "objective": runtime_entry["objective"],
        "value_and_grad": compiled_value_and_grad_for,
        "profile_suite": runtime_entry["profile_suite"],
    }


def make_traceable_objective_profile_suite(booz_jax, bs_jax, iota_target):
    """Build profiled pure-JAX closures for the target single-stage objective path."""
    return make_traceable_objective_runtime_bundle(
        booz_jax,
        bs_jax,
        iota_target,
        include_profile_suite=True,
    )["profile_suite"]
