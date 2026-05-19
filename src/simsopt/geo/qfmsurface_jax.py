"""JAX-backed QFM surface solver orchestration."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .._core.jax_host_boundary import (
    host_array as _host_array,
    host_bool as _host_bool,
    host_scalar as _host_scalar,
)
from ..backend import is_jax_backend
from ..jax_core._math_utils import as_jax_float64
from ..jax_core.qfm_solver import (
    qfm_augmented_lagrangian_solve_jax,
    qfm_label_jax_from_dofs,
    qfm_penalty_jax_from_dofs,
    qfm_penalty_solve_jax,
    qfm_residual_jax_from_dofs,
)
from .qfmsurface import QfmSurface
from .surfaceobjectives import Area, ToroidalFlux, Volume
from .surfaceobjectives_jax import QfmResidualJAX

__all__ = ["QfmSurfaceJAX"]


def _host_int(value: object) -> int:
    return int(np.asarray(jax.device_get(value)).reshape(()).item())


def _write_surface_dofs(surface, dofs: object) -> None:
    surface.x = np.asarray(jax.device_get(dofs), dtype=np.float64)


def _host_value_and_optional_gradient(value_fn, dofs: object, derivatives: int):
    if derivatives == 0:
        return _host_scalar(value_fn(dofs))
    value, gradient = jax.value_and_grad(value_fn)(dofs)
    return _host_scalar(value), _host_array(gradient)


def _augmented_lagrangian_iteration_budget(maxiter: int) -> tuple[int, int]:
    maxiter_value = int(maxiter)
    if maxiter_value < 1:
        raise ValueError("maxiter must be positive.")
    max_outer = min(10, maxiter_value)
    return max_outer, max(1, maxiter_value // max_outer)


class QfmSurfaceJAX:
    """JAX-backed QFM surface adapter with explicit final DOF write-back."""

    def __init__(self, biotsavart, surface, label, targetlabel):
        self.biotsavart = biotsavart
        self.surface = surface
        self.label = label
        self.targetlabel = targetlabel
        self.qfm = QfmResidualJAX(surface, biotsavart)
        self.name = str(id(self))

    def _label_contract(self) -> tuple[str, int]:
        if isinstance(self.label, Area):
            return "area", 0
        if isinstance(self.label, Volume):
            return "volume", 0
        if isinstance(self.label, ToroidalFlux):
            return "toroidal_flux", int(self.label.idx)
        raise TypeError(f"Unsupported QFM label object {type(self.label).__name__!r}.")

    def _coil_set_spec(self):
        return self.biotsavart.coil_set_spec_from_dofs(
            jnp.asarray(self.biotsavart.x, dtype=jnp.float64)
        )

    def _solve_inputs(self):
        label, toroidal_flux_idx = self._label_contract()
        return (
            self.surface.surface_spec(),
            as_jax_float64(self.surface.get_dofs()),
            self._coil_set_spec(),
            label,
            toroidal_flux_idx,
        )

    def qfm_label_constraint(self, x, derivatives=0):
        """Return the label residual objective and optional JAX gradient."""
        if derivatives not in (0, 1):
            raise ValueError("derivatives must be 0 or 1.")
        spec, _dofs, coil_set_spec, label, toroidal_flux_idx = self._solve_inputs()
        dofs = as_jax_float64(x)

        def value_fn(surface_dofs):
            label_value = qfm_label_jax_from_dofs(
                spec,
                surface_dofs,
                coil_set_spec,
                label=label,
                toroidal_flux_idx=toroidal_flux_idx,
            )
            residual = label_value - jnp.asarray(
                self.targetlabel, dtype=label_value.dtype
            )
            return 0.5 * residual * residual

        return _host_value_and_optional_gradient(value_fn, dofs, derivatives)

    def qfm_objective(self, x, derivatives=0):
        """Return the QFM residual and optional JAX gradient."""
        if derivatives not in (0, 1):
            raise ValueError("derivatives must be 0 or 1.")
        spec, _dofs, coil_set_spec, _label, _toroidal_flux_idx = self._solve_inputs()
        dofs = as_jax_float64(x)

        def value_fn(surface_dofs):
            return qfm_residual_jax_from_dofs(
                spec,
                surface_dofs,
                coil_set_spec,
            )

        return _host_value_and_optional_gradient(value_fn, dofs, derivatives)

    def qfm_penalty_constraints(self, x, derivatives=0, constraint_weight=1.0):
        """Return the QFM penalty objective and optional JAX gradient."""
        if derivatives not in (0, 1):
            raise ValueError("derivatives must be 0 or 1.")
        spec, _dofs, coil_set_spec, label, toroidal_flux_idx = self._solve_inputs()
        dofs = as_jax_float64(x)

        def value_fn(surface_dofs):
            return qfm_penalty_jax_from_dofs(
                spec,
                surface_dofs,
                coil_set_spec,
                label=label,
                targetlabel=self.targetlabel,
                constraint_weight=constraint_weight,
                toroidal_flux_idx=toroidal_flux_idx,
            )

        return _host_value_and_optional_gradient(value_fn, dofs, derivatives)

    def _penalty_result_dict(self, final_dofs: object, info) -> dict[str, object]:
        _write_surface_dofs(self.surface, final_dofs)
        return {
            "fun": _host_scalar(info.penalty_value),
            "gradient": _host_array(info.gradient),
            "iter": _host_int(info.nit),
            "info": info,
            "success": _host_bool(info.success),
            "s": self.surface,
        }

    def _augmented_result_dict(self, final_dofs: object, info) -> dict[str, object]:
        _write_surface_dofs(self.surface, final_dofs)
        return {
            "fun": _host_scalar(info.qfm_value),
            "gradient": _host_array(info.gradient),
            "iter": _host_int(info.nit),
            "info": info,
            "success": _host_bool(info.success),
            "s": self.surface,
        }

    def minimize_qfm_penalty_jax(
        self,
        tol: float = 1e-3,
        maxiter: int = 1000,
        constraint_weight: float = 1.0,
        optimizer: str = "bfgs",
    ) -> dict[str, object]:
        """Run the pure JAX QFM penalty solve and write final DOFs once."""
        spec, dofs, coil_set_spec, label, toroidal_flux_idx = self._solve_inputs()
        final_dofs, info = qfm_penalty_solve_jax(
            spec,
            coil_set_spec,
            label,
            self.targetlabel,
            constraint_weight,
            dofs,
            max_iter=maxiter,
            tol=tol,
            optimizer=optimizer,
            toroidal_flux_idx=toroidal_flux_idx,
        )
        return self._penalty_result_dict(final_dofs, info)

    def minimize_qfm_penalty_constraints_LBFGS(
        self,
        tol: float = 1e-3,
        maxiter: int = 1000,
        constraint_weight: float = 1.0,
    ) -> dict[str, object]:
        """Compatibility alias for the JAX BFGS penalty solve."""
        return self.minimize_qfm_penalty_jax(
            tol=tol,
            maxiter=maxiter,
            constraint_weight=constraint_weight,
            optimizer="bfgs",
        )

    def minimize_qfm_exact_jax(
        self,
        tol: float = 1e-3,
        maxiter: int = 1000,
        optimizer: str = "bfgs",
    ) -> dict[str, object]:
        """Run the pure JAX augmented-Lagrangian QFM solve."""
        spec, dofs, coil_set_spec, label, toroidal_flux_idx = self._solve_inputs()
        max_outer, inner_max_iter = _augmented_lagrangian_iteration_budget(maxiter)
        final_dofs, info = qfm_augmented_lagrangian_solve_jax(
            spec,
            coil_set_spec,
            label,
            self.targetlabel,
            dofs,
            max_outer=max_outer,
            inner_max_iter=inner_max_iter,
            tol=tol,
            optimizer=optimizer,
            toroidal_flux_idx=toroidal_flux_idx,
        )
        return self._augmented_result_dict(final_dofs, info)

    def minimize_qfm_exact_constraints_SLSQP(
        self,
        tol: float = 1e-3,
        maxiter: int = 1000,
    ) -> dict[str, object]:
        """Compatibility alias for the JAX augmented-Lagrangian exact path."""
        return self.minimize_qfm_exact_jax(tol=tol, maxiter=maxiter, optimizer="bfgs")

    def minimize_qfm(
        self,
        tol: float = 1e-3,
        maxiter: int = 1000,
        method: str = "AL",
        constraint_weight: float = 1.0,
    ) -> dict[str, object]:
        """Dispatch QFM minimization by explicit backend mode and method."""
        if not is_jax_backend():
            if method == "BFGS":
                native_method = "LBFGS"
            elif method == "AL":
                native_method = "SLSQP"
            else:
                raise ValueError("method must be one of 'BFGS' or 'AL'.")
            return QfmSurface(
                self.biotsavart,
                self.surface,
                self.label,
                self.targetlabel,
            ).minimize_qfm(
                tol=tol,
                maxiter=maxiter,
                method=native_method,
                constraint_weight=constraint_weight,
            )
        if method == "BFGS":
            return self.minimize_qfm_penalty_jax(
                tol=tol,
                maxiter=maxiter,
                constraint_weight=constraint_weight,
                optimizer="bfgs",
            )
        if method == "AL":
            return self.minimize_qfm_exact_jax(
                tol=tol, maxiter=maxiter, optimizer="bfgs"
            )
        if method == "LM":
            return self.minimize_qfm_exact_jax(tol=tol, maxiter=maxiter, optimizer="lm")
        raise ValueError("method must be one of 'BFGS', 'LM', or 'AL'.")
