"""
Lane-aware JAX Boozer surface solver.

The public/reference lane still permits host-side SciPy minimization via
``optimizer_backend="scipy"``. The private optimizer lane adds two more
roles:

- ``optimizer_backend="hybrid"``: transitional migration path
- ``optimizer_backend="ondevice"``: target full-GPU backend

This module owns the LS/exact solver routing contract. Only the
``ondevice`` backend is intended to represent the final full-GPU optimizer
state.

Architecture (per M0 contract §5-§6):
  - Adapter pattern: ``BoozerSurfaceJAX`` inherits ``Optimizable`` and
    mirrors the CPU ``BoozerSurface`` public API.
  - The outer ``Optimizable`` dependency graph and ``need_to_run_code``
    dirty-flag semantics are preserved.
  - The reference lane may still cross the host/device boundary inside the LS
    optimizer loop; removing that is part of the on-device migration.

Builds on M3's composed derivative path:
  - ``_surface_geometry_from_dofs()`` for surface DOFs → geometry (SSOT)
  - ``boozer_residual_scalar()`` for the forward residual
  - ``boozer_residual_vector()`` for the exact Newton residual vector
  - ``boozer_residual_coil_vjp()`` for outer-path coil sensitivities
  - ``jax.grad`` / ``jax.hessian`` / ``jax.jacfwd`` for all derivatives
"""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg

try:
    from simsopt._core.optimizable import Optimizable
except (ImportError, ModuleNotFoundError):
    # Fallback when simsoptpp is unavailable (standalone JAX tests).
    # In production with simsopt fully installed, the real Optimizable is used.
    class Optimizable:  # type: ignore[no-redef]
        def __init__(self, *args, depends_on=None, **kwargs):
            pass


from .surface_fourier_jax import stellsym_scatter_indices
from ..field.biotsavart_jax import (
    group_coil_data,
    grouped_biot_savart_B,
    grouped_biot_savart_A,
)
from .boozer_residual_jax import (
    boozer_residual_scalar,
    boozer_residual_vector,
    _surface_geometry_from_dofs,
)
from .label_constraints_jax import (
    area_jax,
    volume_jax,
    toroidal_flux_jax,
    compute_G_from_currents,
)
from .optimizer_jax import (
    VALID_OPTIMIZER_BACKENDS,
    jax_minimize,
    newton_exact,
    newton_polish,
    require_target_backend_x64,
    resolve_optimizer_backend_method,
)

__all__ = ["BoozerSurfaceJAX"]


def _replace_group_coil_array(coil_arrays, group_index, group_array):
    grouped_arrays = list(coil_arrays)
    grouped_arrays[group_index] = group_array
    return grouped_arrays


def _yield_group_vjps(lm, group_runners, coil_arrays, coil_indices):
    for group_runner, group_array, group_index_list in zip(
        group_runners,
        coil_arrays,
        coil_indices,
    ):
        _, vjp_fn = jax.vjp(group_runner, group_array)
        yield vjp_fn(lm)[0], group_index_list


def _compute_label(
    label_type,
    gamma,
    xphi,
    xtheta,
    phi_idx,
    points,
    coil_arrays,
):
    """Compute the label value (volume, area, or toroidal flux).

    Shared by penalty objective, exact residual, and residual vector.
    """
    normal = jnp.cross(xphi, xtheta)
    if label_type == "volume":
        return volume_jax(gamma, normal)
    if label_type == "area":
        return area_jax(normal)
    ntheta = gamma.shape[1]
    A = grouped_biot_savart_A(points, coil_arrays)
    A = A.reshape(gamma.shape)
    return toroidal_flux_jax(A[phi_idx], xtheta[phi_idx], ntheta)


def _boozer_penalty_objective(
    x,
    coil_arrays,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    targetlabel,
    constraint_weight,
    label_type,
    phi_idx,
    optimize_G,
    weight_inv_modB,
):
    """Scalarized penalty objective for the BoozerLS inner solve.

    Extends M3's ``boozer_penalty_composed`` with label and z-constraints.

    Pure function: ``x → scalar``.  JAX autodiff gives gradient and
    Hessian for free.

    The decision vector is ``x = [surface_dofs, iota]`` (optimize_G=False)
    or ``x = [surface_dofs, iota, G]`` (optimize_G=True).
    """
    if optimize_G:
        sdofs, iota, G = x[:-2], x[-2], x[-1]
    else:
        sdofs, iota = x[:-1], x[-1]
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

    points = gamma.reshape(-1, 3)
    B = grouped_biot_savart_B(points, coil_arrays)
    B = B.reshape(nphi, ntheta, 3)

    J_boozer = boozer_residual_scalar(G, iota, B, xphi, xtheta, weight_inv_modB)

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
    J_z = 0.5 * constraint_weight * gamma[0, 0, 2] ** 2

    return J_boozer + J_label + J_z


def _boozer_exact_residual(
    x,
    coil_arrays,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    targetlabel,
    label_type,
    phi_idx,
    mask_indices,
    stellsym_surface,
    weight_inv_modB,
):
    """Residual vector for the BoozerExact Newton system.

    Extends M3's ``boozer_residual_vector`` with masking and constraint
    equations (label, z-coordinate).

    Returns: (n_eq,) residual vector where ``r(x) = 0`` at the solution.
    The decision vector is always ``x = [surface_dofs, iota, G]``.
    """
    sdofs, iota, G = x[:-2], x[-2], x[-1]

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

    points = gamma.reshape(-1, 3)
    B = grouped_biot_savart_B(points, coil_arrays)
    B = B.reshape(nphi, ntheta, 3)

    r_flat = boozer_residual_vector(G, iota, B, xphi, xtheta, weight_inv_modB)
    r_masked = r_flat[mask_indices]

    label_val = _compute_label(
        label_type,
        gamma,
        xphi,
        xtheta,
        phi_idx,
        points,
        coil_arrays,
    )
    r_label = label_val - targetlabel

    if stellsym_surface:
        return jnp.concatenate([r_masked, jnp.array([r_label])])
    else:
        r_z = gamma[0, 0, 2]
        return jnp.concatenate([r_masked, jnp.array([r_label, r_z])])


def _boozer_exact_coil_vjp(lm, booz_surf, iota, G):
    """JAX VJP for the exact path.

    Replaces CPU ``boozer_surface_dexactresidual_dcoils_dcurrents_vjp``.

    Differentiates the FULL exact residual vector (Boozer + label + z)
    w.r.t. coil geometry and currents via ``jax.vjp``.  This correctly
    includes the label derivative term that the CPU code adds explicitly.

    Args:
        lm: (n_eq,) adjoint vector from the outer implicit-function solve.
        booz_surf: ``BoozerSurfaceJAX`` instance.
        iota: rotational transform at the solution.
        G: Boozer G at the solution.

    Returns:
        (d_coil_arrays,), coil_indices — grouped cotangents and index list.
        ``d_coil_arrays`` is a list of ``(d_g, d_gd, d_c)`` tuples matching
        the coil_arrays pytree structure.
    """
    sdofs = booz_surf._get_surface_dofs()
    x = jnp.concatenate([sdofs, jnp.array([iota, G])])
    mask_indices = booz_surf._compute_stellsym_mask_indices()

    coil_arrays = booz_surf._coil_arrays
    coil_indices = booz_surf._coil_index_lists

    def residual_of_coils(ca):
        return _boozer_exact_residual(
            x,
            coil_arrays=ca,
            quadpoints_phi=booz_surf.quadpoints_phi,
            quadpoints_theta=booz_surf.quadpoints_theta,
            mpol=booz_surf.mpol,
            ntor=booz_surf.ntor,
            nfp=booz_surf.nfp,
            stellsym=booz_surf.stellsym,
            scatter_indices=booz_surf.scatter_indices,
            targetlabel=booz_surf.targetlabel,
            label_type=booz_surf.label_type,
            phi_idx=booz_surf.phi_idx,
            mask_indices=mask_indices,
            stellsym_surface=booz_surf.stellsym,
            weight_inv_modB=booz_surf.options["weight_inv_modB"],
        )

    _, vjp_fn = jax.vjp(residual_of_coils, coil_arrays)
    (d_coil_arrays,) = vjp_fn(lm)
    return d_coil_arrays, coil_indices


def _boozer_exact_coil_vjp_groups(lm, booz_surf, iota, G):
    """Yield exact-solve coil VJPs one grouped coil block at a time."""
    yield from _build_exact_group_vjp_callback(booz_surf, iota, G)(
        lm,
        booz_surf,
        iota,
        G,
    )


def _build_exact_group_vjp_callback(booz_surf, iota, G):
    """Build stable exact-solve group runners for repeated streaming VJPs."""
    sdofs = booz_surf._get_surface_dofs()
    x = jnp.concatenate([sdofs, jnp.array([iota, G])])
    mask_indices = booz_surf._compute_stellsym_mask_indices()

    coil_arrays = booz_surf._coil_arrays
    coil_indices = booz_surf._coil_index_lists

    group_runners = tuple(
        _make_exact_group_runner(
            x,
            coil_arrays,
            booz_surf,
            mask_indices,
            group_index,
        )
        for group_index in range(len(coil_arrays))
    )

    def vjp_groups(lm, _booz_surf, _iota, _G):
        yield from _yield_group_vjps(lm, group_runners, coil_arrays, coil_indices)

    return vjp_groups


def _make_exact_group_runner(x, coil_arrays, booz_surf, mask_indices, group_index):
    def residual_of_group(group_array):
        return _boozer_exact_residual(
            x,
            coil_arrays=_replace_group_coil_array(
                coil_arrays,
                group_index,
                group_array,
            ),
            quadpoints_phi=booz_surf.quadpoints_phi,
            quadpoints_theta=booz_surf.quadpoints_theta,
            mpol=booz_surf.mpol,
            ntor=booz_surf.ntor,
            nfp=booz_surf.nfp,
            stellsym=booz_surf.stellsym,
            scatter_indices=booz_surf.scatter_indices,
            targetlabel=booz_surf.targetlabel,
            label_type=booz_surf.label_type,
            phi_idx=booz_surf.phi_idx,
            mask_indices=mask_indices,
            stellsym_surface=booz_surf.stellsym,
            weight_inv_modB=booz_surf.options["weight_inv_modB"],
        )
    return residual_of_group


def _boozer_ls_coil_vjp(lm, booz_surf, iota, G, weight_inv_modB=True):
    """JAX VJP for the LS penalty path.

    Replaces CPU ``boozer_surface_dlsqgrad_dcoils_vjp``.

    Differentiates the penalty objective GRADIENT w.r.t. coil geometry
    and currents.  This captures all terms (Boozer residual + label +
    z-constraint) because the composed objective includes them.

    Args:
        lm: (n,) adjoint vector (same shape as decision vector).
        booz_surf: ``BoozerSurfaceJAX`` instance.
        iota: rotational transform at the solution.
        G: Boozer G at the solution.
        weight_inv_modB: residual weighting flag.

    Returns:
        (d_coil_arrays,), coil_indices — grouped cotangents and index list.
        ``d_coil_arrays`` is a list of ``(d_g, d_gd, d_c)`` tuples matching
        the coil_arrays pytree structure.
    """
    optimize_G = G is not None
    sdofs = booz_surf._get_surface_dofs()
    if optimize_G:
        x = jnp.concatenate([sdofs, jnp.array([iota, G])])
    else:
        x = jnp.concatenate([sdofs, jnp.array([iota])])

    coil_arrays = booz_surf._coil_arrays
    coil_indices = booz_surf._coil_index_lists

    def grad_of_coils(ca):
        obj = lambda xx: _boozer_penalty_objective(
            xx,
            coil_arrays=ca,
            quadpoints_phi=booz_surf.quadpoints_phi,
            quadpoints_theta=booz_surf.quadpoints_theta,
            mpol=booz_surf.mpol,
            ntor=booz_surf.ntor,
            nfp=booz_surf.nfp,
            stellsym=booz_surf.stellsym,
            scatter_indices=booz_surf.scatter_indices,
            targetlabel=booz_surf.targetlabel,
            constraint_weight=booz_surf.constraint_weight,
            label_type=booz_surf.label_type,
            phi_idx=booz_surf.phi_idx,
            optimize_G=optimize_G,
            weight_inv_modB=weight_inv_modB,
        )
        return jax.grad(obj)(x)

    _, vjp_fn = jax.vjp(grad_of_coils, coil_arrays)
    (d_coil_arrays,) = vjp_fn(lm)
    return d_coil_arrays, coil_indices


def _boozer_ls_coil_vjp_groups(lm, booz_surf, iota, G, weight_inv_modB=True):
    """Yield LS-path coil VJPs one grouped coil block at a time."""
    yield from _build_ls_group_vjp_callback(
        booz_surf,
        iota,
        G,
        weight_inv_modB=weight_inv_modB,
    )(
        lm,
        booz_surf,
        iota,
        G,
    )


def _build_ls_group_vjp_callback(booz_surf, iota, G, weight_inv_modB=True):
    """Build stable LS group runners for repeated streaming VJPs."""
    optimize_G = G is not None
    sdofs = booz_surf._get_surface_dofs()
    if optimize_G:
        x = jnp.concatenate([sdofs, jnp.array([iota, G])])
    else:
        x = jnp.concatenate([sdofs, jnp.array([iota])])

    coil_arrays = booz_surf._coil_arrays
    coil_indices = booz_surf._coil_index_lists

    group_runners = tuple(
        _make_ls_group_runner(
            x,
            coil_arrays,
            booz_surf,
            optimize_G,
            weight_inv_modB,
            group_index,
        )
        for group_index in range(len(coil_arrays))
    )

    def vjp_groups(lm, _booz_surf, _iota, _G):
        yield from _yield_group_vjps(lm, group_runners, coil_arrays, coil_indices)

    return vjp_groups


def _make_ls_group_runner(
    x,
    coil_arrays,
    booz_surf,
    optimize_G,
    weight_inv_modB,
    group_index,
):
    def grad_of_group(group_array):
        return _group_penalty_gradient(
            x,
            _replace_group_coil_array(
                coil_arrays,
                group_index,
                group_array,
            ),
            booz_surf.quadpoints_phi,
            booz_surf.quadpoints_theta,
            booz_surf.mpol,
            booz_surf.ntor,
            booz_surf.nfp,
            booz_surf.stellsym,
            booz_surf.scatter_indices,
            booz_surf.targetlabel,
            booz_surf.constraint_weight,
            booz_surf.label_type,
            booz_surf.phi_idx,
            optimize_G,
            weight_inv_modB,
        )

    return grad_of_group


def _group_penalty_gradient(
    x,
    coil_arrays,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    targetlabel,
    constraint_weight,
    label_type,
    phi_idx,
    optimize_G,
    weight_inv_modB,
):
    obj = lambda xx: _boozer_penalty_objective(
        xx,
        coil_arrays=coil_arrays,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
        scatter_indices=scatter_indices,
        targetlabel=targetlabel,
        constraint_weight=constraint_weight,
        label_type=label_type,
        phi_idx=phi_idx,
        optimize_G=optimize_G,
        weight_inv_modB=weight_inv_modB,
    )
    return jax.grad(obj)(x)


_DEFAULT_OPTIONS_LS = {
    "verbose": True,
    "bfgs_tol": 1e-10,
    "bfgs_maxiter": 1500,
    "optimizer_backend": "scipy",
    "limited_memory": False,
    "force_ondevice_limited_memory": False,
    "newton_tol": 1e-11,
    "newton_maxiter": 40,
    "newton_stab": 0.0,
    "weight_inv_modB": True,
}

_DEFAULT_OPTIONS_EXACT = {
    "verbose": True,
    "newton_tol": 1e-13,
    "newton_maxiter": 40,
    "weight_inv_modB": False,
}

_INTERNAL_OPTIMIZER_OPTIONS = frozenset(
    {
        "hybrid_scipy_maxiter",
        "line_search_maxiter",
        "maxcor",
        "ftol",
        "maxfun",
        "maxgrad",
        "maxls",
        "stage_callback",
        "progress_callback",
    }
)
_ALLOWED_OPTIONS_LS = frozenset(_DEFAULT_OPTIONS_LS) | _INTERNAL_OPTIMIZER_OPTIONS
_ALLOWED_OPTIONS_EXACT = frozenset(_DEFAULT_OPTIONS_EXACT) | {
    "optimizer_backend",
    "stage_callback",
}


def _normalize_solver_options(raw_options, boozer_type):
    """Validate and normalize constructor options for a Boozer solve mode."""
    if "bfgs_method" in raw_options:
        raise ValueError(
            "BoozerSurfaceJAX option 'bfgs_method' was removed. "
            "Use 'optimizer_backend' with one of: scipy, hybrid, ondevice."
        )

    allowed_option_keys = (
        _ALLOWED_OPTIONS_LS if boozer_type == "ls" else _ALLOWED_OPTIONS_EXACT
    )
    unknown_option_keys = sorted(set(raw_options) - allowed_option_keys)
    if unknown_option_keys:
        unknown_keys = ", ".join(repr(key) for key in unknown_option_keys)
        raise ValueError(f"Unknown BoozerSurfaceJAX option(s): {unknown_keys}.")

    optimizer_backend = raw_options.get("optimizer_backend")
    if (
        optimizer_backend is not None
        and optimizer_backend not in VALID_OPTIMIZER_BACKENDS
    ):
        raise ValueError(
            "optimizer_backend must be one of: scipy, hybrid, ondevice."
        )

    normalized_options = dict(raw_options)
    if boozer_type == "exact":
        normalized_options.pop("optimizer_backend", None)
    return normalized_options


class BoozerSurfaceJAX(Optimizable):
    """JAX-native Boozer surface solver.

    Mirrors the CPU ``BoozerSurface`` API — inherits ``Optimizable``,
    carries ``self.label``, and returns result dicts with ``vjp`` hooks.

    Args:
        biotsavart: ``BiotSavartJAX`` instance (or any object with
            ``_coils`` attribute providing curve geometry and currents).
        surface: CPU ``SurfaceXYZTensorFourier`` instance.
        label: An ``Optimizable`` that computes a flux surface label
            (e.g. ``Volume``, ``ToroidalFlux``).  Stored as ``self.label``
            for downstream consumers that call ``boozer_surface.label.J()``.
        targetlabel: target value for the label constraint.
        constraint_weight: penalty weight.  If ``None``, BoozerExact
            path is used; otherwise BoozerLS.
        options: dict of solver options (see ``_DEFAULT_OPTIONS_*``).
            For LS solves, ``optimizer_backend="scipy"`` is the trusted
            reference lane, ``"hybrid"`` is the transitional migration lane,
            and ``"ondevice"`` is the full-GPU target backend.
    """

    def __init__(
        self,
        biotsavart,
        surface,
        label,
        targetlabel,
        constraint_weight=None,
        options=None,
    ):
        super().__init__(depends_on=[biotsavart])

        self.biotsavart = biotsavart
        self.surface = surface
        self.label = label
        self.targetlabel = float(targetlabel)
        self.constraint_weight = constraint_weight
        self.need_to_run_code = True
        self.res = None

        # Determine solver type
        self.boozer_type = "ls" if constraint_weight is not None else "exact"

        # Infer label_type from the label object.
        # Only Volume, Area, and ToroidalFlux have JAX-native implementations.
        label_cls = type(label).__name__
        if "Volume" in label_cls:
            self.label_type = "volume"
        elif "Area" in label_cls:
            self.label_type = "area"
        elif "ToroidalFlux" in label_cls:
            self.label_type = "toroidal_flux"
        else:
            raise ValueError(
                f"Unsupported label type {label_cls!r} for BoozerSurfaceJAX. "
                "Supported: Volume, Area, ToroidalFlux."
            )

        raw_options = _normalize_solver_options(
            dict(options or {}),
            self.boozer_type,
        )
        defaults = (
            _DEFAULT_OPTIONS_LS if self.boozer_type == "ls" else _DEFAULT_OPTIONS_EXACT
        )
        self.options = {**defaults, **raw_options}
        if self.boozer_type == "ls":
            if self.options["optimizer_backend"] not in VALID_OPTIMIZER_BACKENDS:
                raise ValueError(
                    "optimizer_backend must be one of: scipy, hybrid, ondevice."
                )

        # --- Extract static data from CPU objects (one-time) ---
        s = surface
        self.mpol = s.mpol
        self.ntor = s.ntor
        self.nfp = s.nfp
        self.stellsym = s.stellsym
        self.quadpoints_phi = jnp.asarray(s.quadpoints_phi, dtype=jnp.float64)
        self.quadpoints_theta = jnp.asarray(s.quadpoints_theta, dtype=jnp.float64)

        # Stellsym DOF scatter indices
        if self.stellsym:
            self.scatter_indices = jnp.asarray(
                stellsym_scatter_indices(self.mpol, self.ntor)
            )
        else:
            self.scatter_indices = None

        # Toroidal flux phi index (first phi point by default)
        self.phi_idx = 0

        # Coil data (extracted once, updated via _refresh_coil_data)
        self._refresh_coil_data()

    @property
    def _coil_arrays(self):
        """Coil geometry tuples ``(gammas, gammadashs, currents)`` without index lists."""
        return [(g, gd, c) for g, gd, c, _ in self.coil_groups]

    @property
    def _coil_index_lists(self):
        """Per-group coil index lists from ``coil_groups``."""
        return [idx for _, _, _, idx in self.coil_groups]

    def recompute_bell(self, parent=None):
        """Mark solver as needing re-execution (dirty flag)."""
        self.need_to_run_code = True

    def _refresh_coil_data(self):
        """Extract coil geometry and currents as JAX arrays.

        Groups coils by quadrature point count so that coils with
        different ``num_quad_points`` can coexist without crashing
        on array stacking.
        """
        coils = self.biotsavart._coils
        gammas = []
        gammadashs = []
        currents = []
        for c in coils:
            gammas.append(c.curve.gamma())
            gammadashs.append(c.curve.gammadash())
            currents.append(c.current.get_value())
        self.coil_groups = group_coil_data(gammas, gammadashs, currents)
        self.coil_currents = jnp.asarray(np.array(currents))

    def _emit_stage_callback(
        self,
        label: str,
        **extra: float | str | None,
    ) -> None:
        callback = self.options.get("stage_callback")
        if callback is not None:
            callback(label, **extra)

    def _make_solver_progress_callback(self, method: str):
        stage_callback = self.options.get("stage_callback")
        if stage_callback is None:
            return None

        def emit_progress(iteration: int, fun_value: float, grad_inf: float) -> None:
            if iteration <= 5 or iteration % 25 == 0:
                stage_callback(
                    "boozer_ls_progress",
                    iteration=float(iteration),
                    objective=float(fun_value),
                    grad_inf=float(grad_inf),
                    method=method,
                )

        return emit_progress

    def _get_surface_dofs(self):
        """Get current surface DOFs as JAX array."""
        return jnp.asarray(self.surface.get_dofs(), dtype=jnp.float64)

    def _set_surface_dofs(self, dofs_jax):
        """Write JAX DOFs back to CPU surface."""
        self.surface.set_dofs(np.asarray(dofs_jax))

    def _pack_decision_vector(self, iota, G):
        """Pack [surface_dofs, iota] or [surface_dofs, iota, G]."""
        sdofs = self._get_surface_dofs()
        if G is not None:
            return jnp.concatenate([sdofs, jnp.array([iota, G])])
        return jnp.concatenate([sdofs, jnp.array([iota])])

    def _unpack_decision_vector(self, x, optimize_G):
        """Unpack decision vector → (sdofs, iota, G_or_None)."""
        if optimize_G:
            return x[:-2], float(x[-2]), float(x[-1])
        return x[:-1], float(x[-1]), None

    def _make_penalty_objective_with(
        self, optimize_G, weight_inv_modB, constraint_weight=None
    ):
        """Build penalty objective with explicit overrides."""
        return partial(
            _boozer_penalty_objective,
            coil_arrays=self._coil_arrays,
            quadpoints_phi=self.quadpoints_phi,
            quadpoints_theta=self.quadpoints_theta,
            mpol=self.mpol,
            ntor=self.ntor,
            nfp=self.nfp,
            stellsym=self.stellsym,
            scatter_indices=self.scatter_indices,
            targetlabel=self.targetlabel,
            constraint_weight=constraint_weight
            if constraint_weight is not None
            else self.constraint_weight,
            label_type=self.label_type,
            phi_idx=self.phi_idx,
            optimize_G=optimize_G,
            weight_inv_modB=weight_inv_modB,
        )

    def _compute_residual_vector(self, sdofs, iota, G):
        """Compute unscalarized penalty residual vector at given state.

        Reuses M3's ``boozer_residual_vector`` for the Boozer part,
        appends label and z-constraint residuals.

        Returns a NumPy array matching CPU
        ``boozer_penalty_constraints(..., scalarize=False)``.
        """
        gamma, xphi, xtheta = _surface_geometry_from_dofs(
            sdofs,
            self.quadpoints_phi,
            self.quadpoints_theta,
            self.mpol,
            self.ntor,
            self.nfp,
            self.stellsym,
            self.scatter_indices,
        )
        nphi, ntheta = int(gamma.shape[0]), int(gamma.shape[1])
        points = gamma.reshape(-1, 3)
        B = grouped_biot_savart_B(points, self._coil_arrays).reshape(nphi, ntheta, 3)

        r_boozer_raw = boozer_residual_vector(
            G, iota, B, xphi, xtheta, self.options["weight_inv_modB"]
        )
        num_res = 3 * nphi * ntheta
        r_boozer = r_boozer_raw / jnp.sqrt(num_res)

        cw = self.constraint_weight if self.constraint_weight is not None else 1.0
        lab = float(
            _compute_label(
                self.label_type,
                gamma,
                xphi,
                xtheta,
                self.phi_idx,
                points,
                self._coil_arrays,
            )
        )
        rl = jnp.sqrt(cw) * (lab - self.targetlabel)
        rz = jnp.sqrt(cw) * gamma[0, 0, 2]

        return np.asarray(jnp.concatenate([r_boozer, jnp.array([rl, rz])]))

    def minimize_boozer_penalty_constraints_LBFGS(
        self,
        constraint_weight=1.0,
        iota=0.0,
        G=None,
        tol=None,
        maxiter=None,
        verbose=None,
        limited_memory=False,
        weight_inv_modB=None,
    ):
        """BFGS/L-BFGS stage of the LS solve. Matches CPU public API."""
        if not self.need_to_run_code:
            return self.res
        tol = tol if tol is not None else self.options["bfgs_tol"]
        maxiter = maxiter if maxiter is not None else self.options["bfgs_maxiter"]
        verbose = verbose if verbose is not None else self.options["verbose"]
        weight_inv_modB = (
            weight_inv_modB
            if weight_inv_modB is not None
            else self.options["weight_inv_modB"]
        )

        optimize_G = G is not None
        s = self.surface
        x0 = self._pack_decision_vector(iota, G)
        obj_fn = self._make_penalty_objective_with(
            optimize_G, weight_inv_modB, constraint_weight
        )

        optimizer_backend = self.options["optimizer_backend"]
        require_target_backend_x64(optimizer_backend)
        effective_limited_memory = bool(limited_memory)
        if optimizer_backend == "ondevice" and self.options[
            "force_ondevice_limited_memory"
        ]:
            effective_limited_memory = True
        method = resolve_optimizer_backend_method(
            optimizer_backend,
            limited_memory=effective_limited_memory,
        )

        optimizer_options = {}
        for key in (
            "hybrid_scipy_maxiter",
            "line_search_maxiter",
            "maxcor",
            "ftol",
            "maxfun",
            "maxgrad",
            "maxls",
        ):
            if key in self.options:
                optimizer_options[key] = self.options[key]

        result = jax_minimize(
            obj_fn,
            x0,
            method=method,
            tol=tol,
            maxiter=maxiter,
            options=optimizer_options,
            progress_callback=self._make_solver_progress_callback(method),
        )

        sdofs_final, iota_out, G_out = self._unpack_decision_vector(
            result.x, optimize_G
        )
        self._set_surface_dofs(sdofs_final)

        resdict = {
            "fun": float(result.fun),
            "gradient": np.asarray(result.jac),
            "iter": int(result.nit),
            "info": result,
            "success": bool(result.success),
            "G": G_out,
            "s": s,
            "iota": iota_out,
            "optimizer_method": method,
            "weight_inv_modB": weight_inv_modB,
            "type": "ls",
        }
        self.res = resdict
        self.need_to_run_code = False

        if verbose:
            print(
                f"{'L-BFGS-B' if effective_limited_memory else 'BFGS'} solve - "
                f"success={resdict['success']}  iter={resdict['iter']}, "
                f"iota={iota_out:.16f}, ||grad||_inf="
                f"{np.linalg.norm(resdict['gradient'], ord=np.inf):.3e}",
                flush=True,
            )
        return resdict

    def minimize_boozer_penalty_constraints_newton(
        self,
        constraint_weight=1.0,
        iota=0.0,
        G=None,
        tol=None,
        maxiter=None,
        stab=0.0,
        verbose=None,
        weight_inv_modB=None,
    ):
        """Newton polish stage of the LS solve. Matches CPU public API."""
        if not self.need_to_run_code:
            return self.res
        tol = tol if tol is not None else self.options["newton_tol"]
        maxiter = maxiter if maxiter is not None else self.options["newton_maxiter"]
        verbose = verbose if verbose is not None else self.options["verbose"]
        weight_inv_modB = (
            weight_inv_modB
            if weight_inv_modB is not None
            else self.options["weight_inv_modB"]
        )

        optimize_G = G is not None
        s = self.surface
        x0 = self._pack_decision_vector(iota, G)
        obj_fn = self._make_penalty_objective_with(
            optimize_G, weight_inv_modB, constraint_weight
        )

        result = newton_polish(obj_fn, x0, maxiter=maxiter, tol=tol, stab=stab)

        sdofs_final, iota_out, G_out = self._unpack_decision_vector(
            result["x"], optimize_G
        )

        if (
            not np.all(np.isfinite(np.asarray(result["x"])))
            or not np.all(np.isfinite(np.asarray(result["grad"])))
            or not np.all(np.isfinite(np.asarray(result["hessian"])))
        ):
            res = {
                "residual": None,
                "jacobian": None,
                "hessian": None,
                "iter": result["nit"],
                "success": False,
                "G": G_out,
                "s": s,
                "iota": iota_out,
                "PLU": None,
                "vjp": None,
                "vjp_groups": None,
                "type": "ls",
                "weight_inv_modB": weight_inv_modB,
                "fun": float(np.asarray(result["fun"])),
            }
            self.res = res
            self.need_to_run_code = False
            return res

        self._set_surface_dofs(sdofs_final)
        H = result["hessian"]
        P, L, U = jax.scipy.linalg.lu(H)

        G_for_res = (
            G_out
            if G_out is not None
            else float(compute_G_from_currents(self.coil_currents))
        )
        residual_vec = self._compute_residual_vector(sdofs_final, iota_out, G_for_res)

        res = {
            "residual": residual_vec,
            "jacobian": np.asarray(result["grad"]),
            "hessian": H,
            "iter": result["nit"],
            "success": result["success"],
            "G": G_out,
            "s": s,
            "iota": iota_out,
            "PLU": (np.asarray(P), np.asarray(L), np.asarray(U)),
            "vjp": partial(_boozer_ls_coil_vjp, weight_inv_modB=weight_inv_modB),
            "vjp_groups": _build_ls_group_vjp_callback(
                self,
                iota_out,
                G_out,
                weight_inv_modB=weight_inv_modB,
            ),
            "type": "ls",
            "weight_inv_modB": weight_inv_modB,
            "fun": float(result["fun"]),
        }
        self.res = res
        self.need_to_run_code = False

        if verbose:
            grad_norm = float(np.linalg.norm(res["jacobian"]))
            print(
                f"NEWTON solve - success={res['success']}  "
                f"iter={res['iter']}, iota={iota_out:.16f}, "
                f"||grad||={grad_norm:.3e}",
                flush=True,
            )
        return res

    def _make_exact_residual(self, mask_indices):
        """Build the JIT-compiled exact residual function."""
        return partial(
            _boozer_exact_residual,
            coil_arrays=self._coil_arrays,
            quadpoints_phi=self.quadpoints_phi,
            quadpoints_theta=self.quadpoints_theta,
            mpol=self.mpol,
            ntor=self.ntor,
            nfp=self.nfp,
            stellsym=self.stellsym,
            scatter_indices=self.scatter_indices,
            targetlabel=self.targetlabel,
            label_type=self.label_type,
            phi_idx=self.phi_idx,
            mask_indices=mask_indices,
            stellsym_surface=self.stellsym,
            weight_inv_modB=self.options["weight_inv_modB"],
        )

    def _compute_stellsym_mask_indices(self):
        """Compute the integer mask indices for the exact residual.

        Extracts the boolean stellsym mask from the CPU surface object
        and converts to integer indices for JAX fancy indexing.
        """
        s = self.surface
        m = s.get_stellsym_mask()
        mask = np.repeat(m[..., None], 3, axis=2)
        if s.stellsym:
            mask[0, 0, 0] = False
        return jnp.asarray(np.flatnonzero(mask), dtype=jnp.int32)

    def solve_residual_equation_exactly_newton(
        self,
        tol=None,
        maxiter=None,
        iota=0.0,
        G=None,
        verbose=None,
    ):
        """Solve the Boozer residual system exactly via Newton's method.

        Public API matching CPU ``BoozerSurface.solve_residual_equation_exactly_newton()``.

        Args:
            tol: residual norm tolerance. Defaults to options['newton_tol'].
            maxiter: maximum Newton iterations. Defaults to options['newton_maxiter'].
            iota: initial guess for rotational transform.
            G: initial guess for G (None → compute from coil currents).
            verbose: print convergence info.

        Returns:
            dict with 'residual', 'fun', 'jacobian', 'iter', 'success', 'G',
            's', 'iota', 'PLU', 'mask', 'type', 'vjp', 'weight_inv_modB'.
        """
        if not self.need_to_run_code:
            return self.res

        s = self.surface
        try:
            from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier

            if not isinstance(s, SurfaceXYZTensorFourier):
                raise RuntimeError(
                    "Exact solution of Boozer Surfaces only supported for "
                    "SurfaceXYZTensorFourier"
                )
        except (ImportError, ModuleNotFoundError):
            # simsoptpp unavailable — skip type check (tests with mock surfaces)
            pass

        tol = tol if tol is not None else self.options["newton_tol"]
        maxiter = maxiter if maxiter is not None else self.options["newton_maxiter"]
        verbose = verbose if verbose is not None else self.options["verbose"]

        if G is None:
            G = float(compute_G_from_currents(self.coil_currents))

        sdofs = self._get_surface_dofs()
        x0 = jnp.concatenate([sdofs, jnp.array([iota, G])])

        mask_indices = self._compute_stellsym_mask_indices()
        res_fn = self._make_exact_residual(mask_indices)

        result = newton_exact(res_fn, x0, maxiter=maxiter, tol=tol)

        x_final = result["x"]
        exact_residual = res_fn(x_final)
        sdofs_final = x_final[:-2]
        iota_final = float(x_final[-2])
        G_final = float(x_final[-1])

        if (
            not bool(result["success"])
            or not np.all(np.isfinite(np.asarray(x_final)))
            or not np.all(np.isfinite(np.asarray(exact_residual)))
            or not np.all(np.isfinite(np.asarray(result["jacobian"])))
        ):
            res = {
                "residual": None,
                "fun": float(0.5 * np.mean(np.square(np.asarray(exact_residual)))),
                "jacobian": None,
                "iter": result["nit"],
                "success": False,
                "G": G_final,
                "s": s,
                "iota": iota_final,
                "PLU": None,
                "mask": None,
                "type": "exact",
                "vjp": None,
                "vjp_groups": None,
                "weight_inv_modB": self.options["weight_inv_modB"],
            }
            self.res = res
            self.need_to_run_code = False
            return res

        self._set_surface_dofs(sdofs_final)
        J = result["jacobian"]
        P, L, U = jax.scipy.linalg.lu(J)

        nphi = len(self.quadpoints_phi)
        ntheta = len(self.quadpoints_theta)

        # Reconstruct raw (unmasked) Boozer residual for CPU-contract parity.
        gamma_final, xphi_final, xtheta_final = _surface_geometry_from_dofs(
            sdofs_final,
            self.quadpoints_phi,
            self.quadpoints_theta,
            self.mpol,
            self.ntor,
            self.nfp,
            self.stellsym,
            self.scatter_indices,
        )
        B_final = grouped_biot_savart_B(
            gamma_final.reshape(-1, 3),
            self._coil_arrays,
        ).reshape(nphi, ntheta, 3)
        r_raw = boozer_residual_vector(
            G_final,
            iota_final,
            B_final,
            xphi_final,
            xtheta_final,
            self.options["weight_inv_modB"],
        )

        bool_mask = np.zeros(3 * nphi * ntheta, dtype=bool)
        bool_mask[np.asarray(mask_indices)] = True

        res = {
            "residual": np.asarray(r_raw),
            "fun": float(0.5 * np.mean(np.square(np.asarray(exact_residual)))),
            "jacobian": np.asarray(J),
            "iter": result["nit"],
            "success": bool(result["success"]),
            "G": G_final,
            "s": s,
            "iota": iota_final,
            "PLU": (np.asarray(P), np.asarray(L), np.asarray(U)),
            "mask": bool_mask,
            "type": "exact",
            "vjp": _boozer_exact_coil_vjp,
            "vjp_groups": _build_exact_group_vjp_callback(
                self,
                iota_final,
                G_final,
            ),
            "weight_inv_modB": self.options["weight_inv_modB"],
        }
        self.res = res
        self.need_to_run_code = False

        if verbose:
            res_norm = float(np.linalg.norm(res["residual"], ord=np.inf))
            print(
                f"NEWTON solve - success={res['success']}  "
                f"iter={res['iter']}, iota={iota_final:.16f}, "
                f"||residual||_inf={res_norm:.3e}",
                flush=True,
            )
        return res

    def run_code(self, iota, G=None):
        """Run the Boozer surface solver (LS or exact depending on config).

        Mirrors ``BoozerSurface.run_code()`` API.

        Args:
            iota: initial guess for rotational transform.
            G: initial guess for G (None → compute from coil currents,
               and coil currents must be fixed).

        Returns:
            dict with solver results, or None if solver was not dirty.
        """
        if not self.need_to_run_code:
            return

        # When G=None the gradient treats currents as constants,
        # so coil currents must be fixed to avoid silent gradient errors.
        if G is None:
            assert all(c.current.dofs.all_fixed() for c in self.biotsavart._coils), (
                "Coil currents must be fixed when G=None"
            )

        # Refresh coil data in case coils changed
        self._refresh_coil_data()

        if self.boozer_type == "exact":
            res = self.solve_residual_equation_exactly_newton(
                iota=iota,
                G=G,
                tol=self.options["newton_tol"],
                maxiter=self.options["newton_maxiter"],
                verbose=self.options["verbose"],
            )
            return res

        # BoozerLS: BFGS + Newton polish
        assert self.constraint_weight is not None
        ls_res = self.minimize_boozer_penalty_constraints_LBFGS(
            constraint_weight=self.constraint_weight,
            iota=iota,
            G=G,
            tol=self.options["bfgs_tol"],
            maxiter=self.options["bfgs_maxiter"],
            verbose=self.options["verbose"],
            limited_memory=self.options["limited_memory"],
            weight_inv_modB=self.options["weight_inv_modB"],
        )
        self._emit_stage_callback(
            "after_boozer_lbfgs",
            solve_success=("true" if bool(ls_res["success"]) else "false"),
            iterations=float(ls_res["iter"]),
            method=str(ls_res["optimizer_method"]),
        )
        iota_out, G_out = ls_res["iota"], ls_res["G"]

        # Polish with Newton
        self.need_to_run_code = True
        res = self.minimize_boozer_penalty_constraints_newton(
            constraint_weight=self.constraint_weight,
            iota=iota_out,
            G=G_out,
            verbose=self.options["verbose"],
            tol=self.options["newton_tol"],
            maxiter=self.options["newton_maxiter"],
            stab=self.options["newton_stab"],
            weight_inv_modB=self.options["weight_inv_modB"],
        )
        res["optimizer_method"] = ls_res["optimizer_method"]
        self._emit_stage_callback(
            "after_boozer_newton",
            solve_success=("true" if bool(res["success"]) else "false"),
            iterations=float(res["iter"]),
        )
        return res
