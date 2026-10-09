"""Native QFM interfaces with single-device float64 JAX evaluation.

SciPy owns the host optimization and native stopping/failure semantics.
The mutable adapters, like native QFM and BiotSavart, belong to one evaluation
thread. Immutable input snapshots and kernels may be shared across threads.
Configure simsopt_jax.backend before constructing the field. Numeric targetlabel
and constraint_weight values are explicit traced operands; new values do not recompile.
"""

from __future__ import annotations

from typing import Literal, cast, overload

import jax
import numpy as np
from scipy.optimize import minimize

from simsopt._core.json import GSONable
from simsopt._core.optimizable import Optimizable
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo.surfaceobjectives import Area, ToroidalFlux, Volume
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt_jax.backend.dtypes import explicit_device_array
from simsopt_jax.core.qfm import (
    QfmLabelSpec,
    QfmSpec,
    qfm_label_constraint,
    qfm_label_constraint_value_and_grad,
    qfm_penalty_constraints,
    qfm_penalty_constraints_value_and_grad,
    qfm_residual,
    qfm_residual_value_and_grad,
)
from simsopt_jax.runtime.host_boundary import host_array, host_tree
from simsopt_jax_adapters.field.biotsavart_backend import JaxBiotSavart

from .surface_specs import surface_spec_from_surface

__all__ = ["JaxQfmResidual", "JaxQfmSurface"]

_Surface = SurfaceRZFourier | SurfaceXYZFourier | SurfaceXYZTensorFourier
_Label = Volume | Area | ToroidalFlux


@overload
def _host_result(result: jax.Array) -> np.float64: ...


@overload
def _host_result(result: tuple[jax.Array, jax.Array]) -> tuple[np.float64, np.ndarray]: ...


def _host_result(result: jax.Array | tuple[jax.Array, jax.Array]):
    materialized = host_tree(result)
    if isinstance(materialized, tuple):
        value, gradient = materialized
        return np.float64(value), host_array(gradient)
    return np.float64(materialized)


class JaxQfmResidual(Optimizable):
    """Native QfmResidual evaluation and full surface gradient with refreshed JAX snapshots.
    
    Args:
        surface (SurfaceRZFourier | SurfaceXYZFourier | SurfaceXYZTensorFourier): Mutable native surface of exactly one supported class; subclasses are rejected.
        biotsavart (JaxBiotSavart): Mutable single-device field providing coils and current Cartesian point buffers."""

    def __init__(self, surface: _Surface, biotsavart: JaxBiotSavart):
        self.surface = surface
        self.biotsavart = biotsavart
        self.biotsavart.append_parent(self.surface)
        super().__init__(depends_on=[surface, biotsavart])

    def recompute_bell(self, parent=None):
        """Reset field points when a parent invalidates this objective.
        
        Args:
            parent (Optimizable | None): Changed parent supplied by native notification; unused.
        
        Returns:
            None: The field receives a snapshot of the current surface positions."""
        self.invalidate_cache()

    def invalidate_cache(self):
        """Refresh the field point buffer from the current surface.
        
        Returns:
            None: Points are replaced with Cartesian surface positions in meters."""
        self.biotsavart.set_points(np.array(self.surface.gamma().reshape(-1, 3), copy=True))

    def _spec(self) -> QfmSpec:
        return QfmSpec(
            surface_spec_from_surface(self.surface),
            self.biotsavart.coil_set_spec(),
            cast(jax.Array, self.biotsavart.get_points_cart_ref()),
        )

    def J(self):
        """Evaluate the dimensionless native QFM ratio from live geometry and field points.
        
        Returns:
            numpy.float64: Scalar ratio; native nonfinite behavior is retained."""
        return _host_result(qfm_residual(self._spec()))

    def dJ_by_dsurfacecoefficients(self):
        """Evaluate the native gradient with respect to every surface coefficient.
        
        Returns:
            numpy.ndarray: Shape (ndofs,) full gradient in inverse meters, including fixed DOFs."""
        return _host_result(qfm_residual_value_and_grad(self._spec()))[1]


class JaxQfmSurface(GSONable):
    """Native QfmSurface host solves using immutable single-device QFM snapshots.
    
    Args:
        biotsavart (JaxBiotSavart): Mutable single-device field providing coils and current Cartesian point buffers.
        surface (SurfaceRZFourier | SurfaceXYZFourier | SurfaceXYZTensorFourier): Mutable native surface of exactly one supported class; subclasses are rejected.
        label (Volume | Area | ToroidalFlux): Live native label, optionally on a separate grid sharing the surface DOFs. Flux field points stay independent of QFM points.
        targetlabel (float | jax.Array): Scalar target in the native label units, read afresh for each evaluation."""

    def __init__(self, biotsavart: JaxBiotSavart, surface: _Surface, label: _Label, targetlabel):
        self.biotsavart = biotsavart
        self.surface = surface
        self.label = label
        self.targetlabel = targetlabel
        self.qfm = JaxQfmResidual(surface, biotsavart)
        self.name = str(id(self))
        self._native_label_field: tuple[BiotSavart, tuple[int, ...], JaxBiotSavart] | None = None

    def _label_spec(self) -> QfmLabelSpec:
        label = self.label
        kinds = {Volume: "volume", Area: "area", ToroidalFlux: "toroidal_flux"}
        if type(label) not in kinds:
            raise TypeError("JaxQfmSurface supports native Volume, Area and ToroidalFlux labels.")
        surface = surface_spec_from_surface(label.surface)
        coils = None
        points = None
        idx = 0
        if isinstance(label, ToroidalFlux):
            idx = label.idx
            # Match NumPy's negative-index and out-of-bounds semantics before
            # handing the dynamic index to JAX's clamping gather.
            label.surface.quadpoints_phi[idx]
            field = label.biotsavart
            points = explicit_device_array(
                field.get_points_cart() if isinstance(field, BiotSavart) else field.get_points_cart_ref(),
                dtype=np.float64, reference=surface.quadpoints_phi,
            )
            if isinstance(field, BiotSavart):
                cached = self._native_label_field
                coil_ids = tuple(id(coil) for coil in field.coils)
                if cached is None or cached[0] is not field or cached[1] != coil_ids:
                    cached = (field, coil_ids, JaxBiotSavart(field.coils))
                    self._native_label_field = cached
                field = cached[2]
            coils = field.coil_set_spec()
        return QfmLabelSpec(
            surface, coils,
            explicit_device_array(idx, dtype=np.int32, reference=surface.quadpoints_phi),
            kinds[type(label)], points,
        )

    def _target(self, label: QfmLabelSpec) -> jax.Array:
        return explicit_device_array(self.targetlabel, dtype=np.float64, reference=label.surface.quadpoints_phi)

    def _label_value(self, spec: QfmLabelSpec) -> jax.Array:
        # Preserve native's exact floating label subtraction: the squared
        # SLSQP equality has a singular Jacobian precisely at a zero residual.
        # An epsilon threshold would change native's optimization problem.
        label = self.label
        if isinstance(label, ToroidalFlux) and isinstance(label.biotsavart, JaxBiotSavart):
            # ToroidalFlux.J dispatches NumPy's sum to JAX for a JAX field.
            # Keep those device arithmetic boundaries, with explicit operands.
            potential = cast(jax.Array, label.biotsavart.A())
            tangent = explicit_device_array(
                label.surface.gammadash2()[label.idx], dtype=np.float64, reference=potential,
            )
            ntheta = explicit_device_array(tangent.shape[0], dtype=np.int64, reference=potential)
            value = host_array(np.sum(potential * tangent) / ntheta)[()]
        else:
            value = label.J()
        # Native Python-float labels raise OverflowError in this power before
        # evaluating derivatives. Evaluating it before tracing keeps that
        # contract; NumPy scalar labels retain their inf/NaN behavior instead.
        _ = (value - host_tree(self.targetlabel))**2
        return explicit_device_array(value, dtype=np.float64, reference=spec.surface.quadpoints_phi)

    @overload
    def qfm_label_constraint(self, x, derivatives: Literal[0] = 0) -> np.float64: ...

    @overload
    def qfm_label_constraint(self, x, derivatives: Literal[1]) -> tuple[np.float64, np.ndarray]: ...

    @overload
    def qfm_label_constraint(self, x, derivatives: int = 0) -> np.float64 | tuple[np.float64, np.ndarray]: ...

    def qfm_label_constraint(self, x, derivatives=0):
        """Evaluate the native qfm label constraint from the current public inputs.
        
        Args:
            x (numpy.ndarray): Shape (nfree,) free native surface DOFs in meters; evaluation updates the mutable surface.
            derivatives (int): 0 returns the value; 1 returns value and the full surface gradient. Other orders raise the native assertion.
        
        Returns:
            numpy.float64 | tuple[numpy.float64, numpy.ndarray]: Scalar native value, or value and shape (ndofs,) full coefficient gradient, including fixed DOFs. Value uses squared label units; the gradient uses squared label units per meter."""
        assert derivatives in [0, 1]
        self.surface.x = x
        label = self._label_spec()
        function = qfm_label_constraint_value_and_grad if derivatives else qfm_label_constraint
        return _host_result(function(label, self._target(label), self._label_value(label)))

    @overload
    def qfm_objective(self, x, derivatives: Literal[0] = 0) -> np.float64: ...

    @overload
    def qfm_objective(self, x, derivatives: Literal[1]) -> tuple[np.float64, np.ndarray]: ...

    @overload
    def qfm_objective(self, x, derivatives: int = 0) -> np.float64 | tuple[np.float64, np.ndarray]: ...

    def qfm_objective(self, x, derivatives=0):
        """Evaluate the native qfm objective from the current public inputs.
        
        Args:
            x (numpy.ndarray): Shape (nfree,) free native surface DOFs in meters; evaluation updates the mutable surface.
            derivatives (int): 0 returns the value; 1 returns value and the full surface gradient. Other orders raise the native assertion.
        
        Returns:
            numpy.float64 | tuple[numpy.float64, numpy.ndarray]: Scalar native value, or value and shape (ndofs,) full coefficient gradient, including fixed DOFs. Value is dimensionless; the gradient uses inverse meters."""
        assert derivatives in [0, 1]
        self.surface.x = x
        function = qfm_residual_value_and_grad if derivatives else qfm_residual
        return _host_result(function(self.qfm._spec()))

    @overload
    def qfm_penalty_constraints(self, x, derivatives: Literal[0] = 0, constraint_weight: float = 1) -> np.float64: ...

    @overload
    def qfm_penalty_constraints(self, x, derivatives: Literal[1], constraint_weight: float = 1) -> tuple[np.float64, np.ndarray]: ...

    @overload
    def qfm_penalty_constraints(self, x, derivatives: int = 0, constraint_weight: float = 1) -> np.float64 | tuple[np.float64, np.ndarray]: ...

    def qfm_penalty_constraints(self, x, derivatives=0, constraint_weight: float = 1):
        """Evaluate the native qfm penalty constraints from the current public inputs.
        
        Args:
            x (numpy.ndarray): Shape (nfree,) free native surface DOFs in meters; evaluation updates the mutable surface.
            derivatives (int): 0 returns the value; 1 returns value and the full surface gradient. Other orders raise the native assertion.
            constraint_weight (float): Scalar native coefficient multiplying half the squared label error.
        
        Returns:
            numpy.float64 | tuple[numpy.float64, numpy.ndarray]: Scalar native value, or value and shape (ndofs,) full coefficient gradient, including fixed DOFs. Units follow the native scalarization."""
        assert derivatives in [0, 1]
        self.surface.x = x
        spec = self.qfm._spec()
        label = self._label_spec()
        placed_constraint_weight = explicit_device_array(constraint_weight, dtype=np.float64, reference=label.surface.quadpoints_phi)
        function = qfm_penalty_constraints_value_and_grad if derivatives else qfm_penalty_constraints
        return _host_result(function(spec, label, self._target(label), placed_constraint_weight, self._label_value(label)))

    def minimize_qfm_penalty_constraints_LBFGS(self, tol=1e-3, maxiter=1000, constraint_weight=1.):
        """Run the native host optimizer and store its final surface iterate.
        
        Args:
            tol (float): Native SciPy stopping tolerance; passed as ftol and also gtol for L-BFGS-B.
            maxiter (int): Maximum native SciPy iterations; default 1000.
            constraint_weight (float): Native penalty coefficient, used only for L-BFGS-B.
        
        Returns:
            dict: Ordered fun, gradient, iter, info, success and s entries, with scalar objective, shape (nfree,) optimizer gradient, SciPy result and mutable surface. Exceptions preserve the last callback iterate."""
        def objective(x):
            return self.qfm_penalty_constraints(x, derivatives=1, constraint_weight=constraint_weight)

        result = minimize(
            objective, self.surface.x, jac=True, method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": tol, "gtol": tol, "maxcor": 200},
        )
        return self._store(result)

    def minimize_qfm_exact_constraints_SLSQP(self, tol=1e-3, maxiter=1000):
        """Run the native host optimizer and store its final surface iterate.
        
        Args:
            tol (float): Native SciPy stopping tolerance; passed as ftol and also gtol for L-BFGS-B.
            maxiter (int): Maximum native SciPy iterations; default 1000.
        
        Returns:
            dict: Ordered fun, gradient, iter, info, success and s entries, with scalar objective, shape (nfree,) optimizer gradient, SciPy result and mutable surface. Exceptions preserve the last callback iterate."""
        def objective(x):
            return self.qfm_objective(x, derivatives=1)

        def constraint(x):
            return self.qfm_label_constraint(x, derivatives=1)[0]

        def constraint_gradient(x):
            return self.qfm_label_constraint(x, derivatives=1)[1]

        result = minimize(
            objective, self.surface.x, jac=True, method="SLSQP",
            constraints=[{"type": "eq", "fun": constraint, "jac": constraint_gradient}],
            options={"maxiter": maxiter, "ftol": tol},
        )
        return self._store(result)

    def _store(self, result):
        result_dict = {
            "fun": result.fun, "gradient": result.jac, "iter": result.nit,
            "info": result, "success": result.success,
        }
        self.surface.x = result.x
        result_dict["s"] = self.surface
        return result_dict

    def minimize_qfm(self, tol=1e-3, maxiter=1000, method="SLSQP", constraint_weight=1.):
        """Run the native host optimizer and store its final surface iterate.
        
        Args:
            tol (float): Native SciPy stopping tolerance; passed as ftol and also gtol for L-BFGS-B.
            maxiter (int): Maximum native SciPy iterations; default 1000.
            method (str): LBFGS selects the penalty solve; SLSQP selects the squared-label equality solve. Other names raise ValueError.
            constraint_weight (float): Native penalty coefficient, used only for L-BFGS-B.
        
        Returns:
            dict: Ordered fun, gradient, iter, info, success and s entries, with scalar objective, shape (nfree,) optimizer gradient, SciPy result and mutable surface. Exceptions preserve the last callback iterate."""
        if method == "SLSQP":
            return self.minimize_qfm_exact_constraints_SLSQP(tol=tol, maxiter=maxiter)
        if method == "LBFGS":
            return self.minimize_qfm_penalty_constraints_LBFGS(tol=tol, maxiter=maxiter, constraint_weight=constraint_weight)
        raise ValueError
