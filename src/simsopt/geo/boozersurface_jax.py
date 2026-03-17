"""
JAX-native Boozer surface solver.

Replaces the CPU ``BoozerSurface.run_code()`` with a pure JAX inner solve
that keeps all computation on-device between entry and exit.

Architecture (per M0 contract §5-§6):
  - Adapter pattern: ``BoozerSurfaceJAX`` inherits ``Optimizable`` and
    mirrors the CPU ``BoozerSurface`` public API.
  - The outer ``Optimizable`` dependency graph and ``need_to_run_code``
    dirty-flag semantics are preserved.
  - Host↔device transfers happen only at the ``run_code()`` boundary.

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


from simsopt.geo.surface_fourier_jax import stellsym_scatter_indices
from simsopt.field.biotsavart_jax import biot_savart_B, biot_savart_A
from simsopt.geo.boozer_residual_jax import (
    boozer_residual_scalar,
    boozer_residual_vector,
    _surface_geometry_from_dofs,
)
from simsopt.geo.label_constraints_jax import (
    area_jax,
    volume_jax,
    toroidal_flux_jax,
    compute_G_from_currents,
)
from simsopt.geo.optimizer_jax import jax_minimize, newton_polish, newton_exact

__all__ = ["BoozerSurfaceJAX"]


# ---------------------------------------------------------------------------
# Pure JAX objective functions (no Python side effects, fully JIT-able)
#
# These extend M3's composed pipeline with label constraints and
# z-coordinate penalty for the full Boozer inner solve.
# ---------------------------------------------------------------------------


def _boozer_penalty_objective(
    x,
    # --- static / closed-over data ---
    coil_gammas,
    coil_gammadashs,
    coil_currents,
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
    # 1. Unpack decision vector
    if optimize_G:
        sdofs, iota, G = x[:-2], x[-2], x[-1]
    else:
        sdofs, iota = x[:-1], x[-1]
        G = compute_G_from_currents(coil_currents)

    # 2. Surface geometry from DOFs (reuses M3's SSOT helper)
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

    # 3. Magnetic field on surface
    points = gamma.reshape(-1, 3)
    B = biot_savart_B(points, coil_gammas, coil_gammadashs, coil_currents)
    B = B.reshape(nphi, ntheta, 3)

    # 4. Boozer residual scalar (M3 forward kernel)
    J_boozer = boozer_residual_scalar(G, iota, B, xphi, xtheta, weight_inv_modB)

    # 5. Label constraint (M4 addition)
    normal = jnp.cross(xphi, xtheta)
    if label_type == "volume":
        label_val = volume_jax(gamma, normal)
    elif label_type == "area":
        label_val = area_jax(normal)
    else:  # "toroidal_flux"
        A = biot_savart_A(points, coil_gammas, coil_gammadashs, coil_currents)
        A = A.reshape(nphi, ntheta, 3)
        label_val = toroidal_flux_jax(A[phi_idx], xtheta[phi_idx], ntheta)

    # 6. Penalty terms
    J_label = 0.5 * constraint_weight * (label_val - targetlabel) ** 2
    J_z = 0.5 * constraint_weight * gamma[0, 0, 2] ** 2

    return J_boozer + J_label + J_z


def _boozer_exact_residual(
    x,
    # --- static / closed-over data ---
    coil_gammas,
    coil_gammadashs,
    coil_currents,
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

    # Surface geometry (M3 SSOT helper)
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

    # Magnetic field
    points = gamma.reshape(-1, 3)
    B = biot_savart_B(points, coil_gammas, coil_gammadashs, coil_currents)
    B = B.reshape(nphi, ntheta, 3)

    # Boozer residual vector (M3 kernel, reused)
    r_flat = boozer_residual_vector(G, iota, B, xphi, xtheta, weight_inv_modB)

    # Select masked equations (M4: colocation grid filtering)
    r_masked = r_flat[mask_indices]

    # Label constraint (M4 addition)
    normal = jnp.cross(xphi, xtheta)
    if label_type == "volume":
        label_val = volume_jax(gamma, normal)
    elif label_type == "area":
        label_val = area_jax(normal)
    else:
        A = biot_savart_A(points, coil_gammas, coil_gammadashs, coil_currents)
        A = A.reshape(nphi, ntheta, 3)
        label_val = toroidal_flux_jax(A[phi_idx], xtheta[phi_idx], ntheta)

    r_label = label_val - targetlabel

    if stellsym_surface:
        return jnp.concatenate([r_masked, jnp.array([r_label])])
    else:
        r_z = gamma[0, 0, 2]
        return jnp.concatenate([r_masked, jnp.array([r_label, r_z])])


# ---------------------------------------------------------------------------
# JAX VJP wrappers for outer-path coil sensitivities
# ---------------------------------------------------------------------------


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
        (d_coil_gammas, d_coil_gammadashs, d_coil_currents) cotangent arrays.
    """
    sdofs = jnp.asarray(booz_surf.surface.get_dofs())
    x = jnp.concatenate([sdofs, jnp.array([iota, G])])
    mask_indices = booz_surf._compute_stellsym_mask_indices()

    def residual_of_coils(cg, cgd, ci):
        return _boozer_exact_residual(
            x,
            coil_gammas=cg,
            coil_gammadashs=cgd,
            coil_currents=ci,
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

    _, vjp_fn = jax.vjp(
        residual_of_coils,
        booz_surf.coil_gammas,
        booz_surf.coil_gammadashs,
        booz_surf.coil_currents,
    )
    return vjp_fn(lm)


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
        (d_coil_gammas, d_coil_gammadashs, d_coil_currents) cotangent arrays.
    """
    optimize_G = G is not None
    sdofs = jnp.asarray(booz_surf.surface.get_dofs())
    if optimize_G:
        x = jnp.concatenate([sdofs, jnp.array([iota, G])])
    else:
        x = jnp.concatenate([sdofs, jnp.array([iota])])

    def grad_of_coils(cg, cgd, ci):
        """Gradient of the penalty objective w.r.t. decision vector x,
        as a function of coil params."""
        obj = lambda xx: _boozer_penalty_objective(
            xx,
            coil_gammas=cg,
            coil_gammadashs=cgd,
            coil_currents=ci,
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

    _, vjp_fn = jax.vjp(
        grad_of_coils,
        booz_surf.coil_gammas,
        booz_surf.coil_gammadashs,
        booz_surf.coil_currents,
    )
    return vjp_fn(lm)


# ---------------------------------------------------------------------------
# BoozerSurfaceJAX class (adapter around pure functions)
# ---------------------------------------------------------------------------


_DEFAULT_OPTIONS_LS = {
    "verbose": True,
    "bfgs_tol": 1e-10,
    "bfgs_maxiter": 1500,
    "bfgs_method": "bfgs",
    "limited_memory": False,
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

        defaults = (
            _DEFAULT_OPTIONS_LS if self.boozer_type == "ls" else _DEFAULT_OPTIONS_EXACT
        )
        self.options = {**defaults, **(options or {})}

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

    def recompute_bell(self, parent=None):
        """Mark solver as needing re-execution (dirty flag)."""
        self.need_to_run_code = True

    def _refresh_coil_data(self):
        """Extract coil geometry and currents as JAX arrays."""
        coils = self.biotsavart._coils
        gammas = []
        gammadashs = []
        currents = []
        for c in coils:
            gammas.append(c.curve.gamma())
            gammadashs.append(c.curve.gammadash())
            currents.append(c.current.get_value())
        self.coil_gammas = jnp.asarray(np.array(gammas))
        self.coil_gammadashs = jnp.asarray(np.array(gammadashs))
        self.coil_currents = jnp.asarray(np.array(currents))

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
            coil_gammas=self.coil_gammas,
            coil_gammadashs=self.coil_gammadashs,
            coil_currents=self.coil_currents,
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
        B = biot_savart_B(
            points, self.coil_gammas, self.coil_gammadashs, self.coil_currents
        ).reshape(nphi, ntheta, 3)

        # Boozer residual vector (reuse M3 kernel)
        r_boozer_raw = boozer_residual_vector(
            G, iota, B, xphi, xtheta, self.options["weight_inv_modB"]
        )
        num_res = 3 * nphi * ntheta
        r_boozer = r_boozer_raw / jnp.sqrt(num_res)

        # Label and z-constraint residuals
        cw = self.constraint_weight if self.constraint_weight is not None else 1.0
        normal = jnp.cross(xphi, xtheta)
        if self.label_type == "volume":
            lab = float(volume_jax(gamma, normal))
        elif self.label_type == "area":
            lab = float(area_jax(normal))
        else:
            A = biot_savart_A(
                points, self.coil_gammas, self.coil_gammadashs, self.coil_currents
            ).reshape(nphi, ntheta, 3)
            lab = float(
                toroidal_flux_jax(A[self.phi_idx], xtheta[self.phi_idx], ntheta)
            )
        rl = jnp.sqrt(cw) * (lab - self.targetlabel)
        rz = jnp.sqrt(cw) * gamma[0, 0, 2]

        return np.asarray(jnp.concatenate([r_boozer, jnp.array([rl, rz])]))

    # ------------------------------------------------------------------
    # LS (penalty) path
    # ------------------------------------------------------------------

    def _make_penalty_objective(self, optimize_G):
        """Build penalty objective using default weight_inv_modB."""
        return self._make_penalty_objective_with(
            optimize_G, self.options["weight_inv_modB"]
        )

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

        method = "lbfgs" if limited_memory else "bfgs"
        result = jax_minimize(obj_fn, x0, method=method, tol=tol, maxiter=maxiter)

        sdofs_final, iota_out, G_out = self._unpack_decision_vector(
            result.x, optimize_G
        )
        s.set_dofs(np.asarray(sdofs_final))

        resdict = {
            "fun": float(result.fun),
            "gradient": np.asarray(result.jac),
            "iter": result.nit,
            "info": result,
            "success": result.success,
            "G": G_out,
            "s": s,
            "iota": iota_out,
            "weight_inv_modB": weight_inv_modB,
            "type": "ls",
        }
        self.res = resdict
        self.need_to_run_code = False

        if verbose:
            print(
                f"{'L-BFGS-B' if limited_memory else 'BFGS'} solve - "
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

    # ------------------------------------------------------------------
    # Exact (Newton) path
    # ------------------------------------------------------------------

    def _make_exact_residual(self, mask_indices):
        """Build the JIT-compiled exact residual function."""
        return partial(
            _boozer_exact_residual,
            coil_gammas=self.coil_gammas,
            coil_gammadashs=self.coil_gammadashs,
            coil_currents=self.coil_currents,
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
        mask = np.concatenate((m[..., None], m[..., None], m[..., None]), axis=2)
        if s.stellsym:
            mask[0, 0, 0] = False
        mask = mask.flatten()
        return jnp.asarray(np.where(mask)[0], dtype=jnp.int32)

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
            dict with 'residual', 'jacobian', 'iter', 'success', 'G',
            's', 'iota', 'PLU', 'mask', 'type', 'vjp'.
        """
        if not self.need_to_run_code:
            return self.res

        # Preflight: exact Newton requires SurfaceXYZTensorFourier
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
        sdofs_final = x_final[:-2]
        iota_final = float(x_final[-2])
        G_final = float(x_final[-1])

        self._set_surface_dofs(sdofs_final)

        J = result["jacobian"]
        P, L, U = jax.scipy.linalg.lu(J)

        res = {
            "residual": np.asarray(result["residual"]),
            "jacobian": np.asarray(J),
            "iter": result["nit"],
            "success": result["success"],
            "G": G_final,
            "s": s,
            "iota": iota_final,
            "PLU": (np.asarray(P), np.asarray(L), np.asarray(U)),
            "mask": np.asarray(mask_indices),
            "type": "exact",
            "vjp": _boozer_exact_coil_vjp,
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        res = self.minimize_boozer_penalty_constraints_LBFGS(
            constraint_weight=self.constraint_weight,
            iota=iota,
            G=G,
            tol=self.options["bfgs_tol"],
            maxiter=self.options["bfgs_maxiter"],
            verbose=self.options["verbose"],
            limited_memory=self.options["limited_memory"],
            weight_inv_modB=self.options["weight_inv_modB"],
        )
        iota_out, G_out = res["iota"], res["G"]

        # Polish with Newton
        self.need_to_run_code = True
        res = self.minimize_boozer_penalty_constraints_newton(
            constraint_weight=self.constraint_weight,
            iota=iota_out,
            G=G_out,
            verbose=self.options["verbose"],
            tol=self.options["newton_tol"],
            maxiter=self.options["newton_maxiter"],
            weight_inv_modB=self.options["weight_inv_modB"],
        )
        return res
