"""Immutable opt-in exact single-stage evaluator for host SciPy optimizers.

Build from a solved native/JAX BoozerSurface after fixing the coil layout.
The object is a snapshot, not an Optimizable or a session. Pass its initial
state into evaluate and keep the returned state in the caller's optimizer
closure; independent closures/copies can share the same frozen object::

    evaluator = ExactSingleStageJAX.from_boozer_surface(
        boozer_surface, field, base_curves,
        iota_target=iota_target, major_radius_target=radius_target,
        length_target=length_target,
    )
    state = evaluator.initial_state

    def fun(parameters):
        global state
        evaluation = evaluator.evaluate(parameters, state)
        state = evaluation.state
        return evaluation.value, evaluation.gradient

    result = minimize(fun, host_array(evaluator.initial_parameters), jac=True,
                      method="BFGS")

Numeric problem settings may be replaced through dataclasses.replace; they
remain traced operands. Rebuild after changes to the coil graph, free/fixed
layout or fixed values, surface/label grids and structural settings. Native
objects are never changed by construction or evaluation. A failed evaluation
does not carry its failed inner iterate into the next warm start.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import numpy as np
from numpy.typing import NDArray

from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.curve import Curve
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt_jax.backend.dtypes import commit_in_place, explicit_device_array
from simsopt_jax.core.single_stage_exact import (
    ExactSingleStageProblem,
    ExactSingleStageState,
    exact_single_stage_evaluate,
)
from simsopt_jax.pytree import pytree_dataclass
from simsopt_jax.runtime.host_boundary import host_value
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

from .boozer_problem import boozer_exact_residual_rows, boozer_problem
from .boozer_surface import BoozerSurfaceJAX
from .surface_specs import surface_spec_from_surface

__all__ = ["ExactSingleStageEvaluation", "ExactSingleStageJAX"]


@dataclass(frozen=True)
class ExactSingleStageEvaluation:
    """Host value/gradient and solve facts; warm start stays device-resident.

    The gradient is evaluated at solved_x even on failure, while state keeps
    the incoming successful state. Pass value/gradient to minimize(jac=True).
    """

    value: float
    gradient: NDArray[np.float64]
    state: ExactSingleStageState
    solved_x: jax.Array
    success: bool
    iterations: int
    norm: float
    terms: NDArray[np.float64]


@pytree_dataclass(data=("problem", "initial_state", "initial_parameters"))
class ExactSingleStageJAX:
    """Frozen exact boozerQA snapshot with explicit state in/out.

    Construction captures a solved surface, coil layout, label and targets.
    evaluate(parameters, state) returns the value/gradient and next state;
    no native object or hidden warm-start/cache is mutated.
    """

    problem: ExactSingleStageProblem
    initial_state: ExactSingleStageState
    initial_parameters: jax.Array

    @classmethod
    def from_boozer_surface(
        cls,
        boozer_surface: BoozerSurface | BoozerSurfaceJAX,
        field: BiotSavartJAX,
        length_curves: Sequence[Curve],
        *,
        iota_target: float,
        major_radius_target: float,
        length_target: float,
        sDIM: int = 20,
        quasi_poloidal: bool = False,
        tol: float = 1e-13,
        maxiter: float = 40,
    ) -> ExactSingleStageJAX:
        """Snapshot a solved exact surface and its field's free DOF layout.

        length_curves must occur directly in field.coils (not unlisted curves).
        Targets have native QuadraticPenalty semantics and unit weights.
        Infinite targets retain native inf values and nonfinite gradients;
        only nonfinite component adjoints cause the native ValueError.
        The inner vector includes G; current-derived G is not this route.
        """
        if boozer_surface.boozer_type != "exact":
            raise ValueError("ExactSingleStageJAX requires an exact BoozerSurface.")
        if not boozer_surface.res["success"]:
            raise ValueError("the initial exact BoozerSurface solve must have succeeded.")
        if boozer_surface.res["G"] is None:
            raise ValueError("ExactSingleStageJAX requires G as an explicit inner variable.")
        coils = tuple(field.coils)
        surface_coils = tuple(boozer_surface.biotsavart.coils)
        if len(coils) != len(surface_coils) or any(a is not b for a, b in zip(coils, surface_coils)):
            raise ValueError("the evaluator field must use the BoozerSurface's coils.")
        surface = boozer_surface.surface
        assert type(surface) is SurfaceXYZTensorFourier
        length_indices = []
        for curve in length_curves:
            matches = [index for index, coil in enumerate(coils) if coil.curve is curve]
            if not matches:
                raise ValueError("length_curves must occur directly in field.coils.")
            length_indices.append(matches[0])
        problem = boozer_problem(field, surface, boozer_surface.label, boozer_surface.targetlabel)
        reference = commit_in_place(problem.target_label)
        auxiliary = SurfaceXYZTensorFourier(
            mpol=surface.mpol, ntor=surface.ntor, stellsym=surface.stellsym, nfp=surface.nfp,
            quadpoints_phi=np.linspace(0, 1 / surface.nfp, 2 * sDIM, endpoint=False),
            quadpoints_theta=np.linspace(0, 1, 2 * sDIM, endpoint=False),
            dofs=surface.dofs,
        )
        # Commit all inputs like the warm-start outputs to avoid a second
        # executable when the first evaluation's state is fed back.
        problem = jax.tree.map(commit_in_place, problem)
        extraction = jax.tree.map(commit_in_place, field.coil_dof_extraction_spec())
        nonqs_surface = jax.tree.map(commit_in_place, surface_spec_from_surface(auxiliary))
        initial_x = np.concatenate((
            surface.get_dofs(), [boozer_surface.res["iota"], boozer_surface.res["G"]],
        ))
        return cls(
            problem=ExactSingleStageProblem(
                boozer=problem,
                extraction=extraction,
                nonqs_surface=nonqs_surface,
                residual_rows=boozer_exact_residual_rows(surface, reference),
                targets=explicit_device_array(
                    [iota_target, major_radius_target, length_target], dtype=np.float64, reference=reference,
                ),
                tol=explicit_device_array(tol, dtype=np.float64, reference=reference),
                maxiter=explicit_device_array(maxiter, dtype=np.float64, reference=reference),
                length_indices=tuple(length_indices),
                axis=1 if quasi_poloidal else 0,
            ),
            initial_state=ExactSingleStageState(
                explicit_device_array(initial_x, dtype=np.float64, reference=reference),
            ),
            initial_parameters=explicit_device_array(field.x, dtype=np.float64, reference=reference),
        )

    def evaluate(
        self, parameters: NDArray[np.float64] | jax.Array, state: ExactSingleStageState,
    ) -> ExactSingleStageEvaluation:
        """One fused dispatch and explicit host read, with native solve errors.

        The caller owns state. A raised error returns no state, so retrying
        with the same state is safe. No compilation cache/session is owned here.
        """
        parameters = explicit_device_array(parameters, dtype=np.float64, reference=state.x)
        if parameters.shape != self.initial_parameters.shape:
            raise ValueError(f"expected coil DOFs of shape {self.initial_parameters.shape}, got {parameters.shape}.")
        result = exact_single_stage_evaluate(self.problem, parameters, state)
        value, gradient, success, iterations, norm, singular, finite, radius_singular, terms = host_value((
            result.value, result.gradient, result.success, result.iterations,
            result.norm, result.singular, result.adjoint_finite, result.radius_singular, result.terms,
        ))
        if singular:
            raise np.linalg.LinAlgError("Singular matrix")
        if not np.isfinite(terms[0]):
            raise ValueError("array must not contain infs or NaNs")
        if radius_singular:
            raise np.linalg.LinAlgError("Singular matrix")
        if not finite:
            raise ValueError("array must not contain infs or NaNs")
        return ExactSingleStageEvaluation(
            value=float(value), gradient=np.asarray(gradient, dtype=np.float64),
            state=result.state, solved_x=result.solved_x, success=bool(success),
            iterations=int(iterations), norm=float(norm), terms=np.asarray(terms, dtype=np.float64),
        )
