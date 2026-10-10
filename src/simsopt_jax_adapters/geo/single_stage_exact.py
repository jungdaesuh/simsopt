"""Immutable opt-in exact single-stage evaluator for host SciPy optimizers.

Build from a solved native/JAX BoozerSurface after fixing the coil layout.
The object is a snapshot, not an Optimizable or a session. Pass its initial
state into evaluate and keep the returned state in the caller's optimizer
closure; independent closures/copies can share the same frozen object::

    evaluator = JaxExactSingleStage.from_boozer_surface(
        boozer_surface, biotsavart, base_curves,
        iota_target=iota_target, major_radius_target=radius_target,
        length_target=length_target,
    )
    state = evaluator.initial_state

    def fun(dofs):
        global state
        evaluation = evaluator.evaluate(dofs, state)
        state = evaluation.state
        return evaluation.value, evaluation.gradient

    result = minimize(fun, host_array(evaluator.initial_dofs), jac=True,
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
from simsopt_jax_adapters.field.biotsavart_backend import JaxBiotSavart

from .boozer_problem import boozer_exact_residual_rows, boozer_problem
from .boozer_surface import JaxBoozerSurface
from .surface_specs import surface_spec_from_surface

__all__ = ["ExactSingleStageEvaluation", "JaxExactSingleStage"]


@dataclass(frozen=True)
class ExactSingleStageEvaluation:
    """Host value/gradient and solve facts; warm start stays device-resident.

    The gradient is evaluated at solved_x even on failure, while state keeps
    the incoming successful state. Pass value/gradient to minimize(jac=True).

    Args:
        value (float): Objective value, or 1e3 on failed convergence.
        gradient (numpy.ndarray): Float64 shape (nfree,) outer free coil DOF gradient at
            solved_x, also on failure.
        state (ExactSingleStageState): Caller-owned warm start for the inner exact
            solve. Returned next warm start; unchanged on failed convergence.
        solved_x (jax.Array): Shape (nsurface + 2,) final inner [surface DOFs, iota, G],
            including a failed iterate.
        success (bool): Whether the inner residual meets the tolerance.
        iterations (int): Completed inner Newton step count.
        norm (float): Native exact-solve stopping residual norm.
        terms (numpy.ndarray): Shape (4,) raw [dimensionless non-QS ratio, dimensionless
            iota, major radius in meters, selected total length in meters].
    """

    value: float
    gradient: NDArray[np.float64]
    state: ExactSingleStageState
    solved_x: jax.Array
    success: bool
    iterations: int
    norm: float
    terms: NDArray[np.float64]


@pytree_dataclass(data=("problem", "initial_state", "initial_dofs"))
class JaxExactSingleStage:
    r"""Exact-Boozer single-stage objective of ``examples/2_Intermediate/boozerQA.py`` and its
    gradient with respect to the free coil DOFs, evaluated in JAX. There is no single native
    class for it; native builds it from :class:`simsopt.geo.NonQuasiSymmetricRatio`,
    :class:`simsopt.geo.Iotas`, :class:`simsopt.geo.MajorRadius`,
    :class:`simsopt.geo.CurveLength` and :class:`simsopt.objectives.QuadraticPenalty`.

    For coil DOFs :math:`c`, the exact Boozer solve gives the surface, :math:`\iota` and
    :math:`G`, and the objective is

    .. math::
        J(c) = J_{\text{non-QS}} + \tfrac{1}{2}(\iota - \iota^*)^2 + \tfrac{1}{2}(R - R^*)^2
            + \tfrac{1}{2}\max\Big(\sum_{k} L_k - L^*, 0\Big)^2,

    where :math:`J_{\text{non-QS}}` is the ratio of :class:`simsopt.geo.NonQuasiSymmetricRatio`,
    :math:`R` is the major radius of :class:`simsopt.geo.MajorRadius`, :math:`L_k` are the
    lengths of the selected curves, and :math:`\iota^*`, :math:`R^*`, :math:`L^*` are the
    targets. If the inner solve fails, the value is :math:`10^3`, as in ``boozerQA.py``. The
    gradient includes the dependence of the surface, :math:`\iota` and :math:`G` on the
    coils through the solve.

    Build it with :meth:`from_boozer_surface`, then call ``evaluate(dofs, state)``, which
    returns the value, gradient and next warm-start state; no native object or hidden
    state is changed. It is not an Optimizable: rebuild it after changing the coil graph,
    the free/fixed layout, fixed values or grids.

    Args:
        problem (ExactSingleStageProblem): Frozen geometry, coil layout, labels and
            numeric targets.
        initial_state (ExactSingleStageState): Caller-owned warm start for the inner
            exact solve. Captured successful seed.
        initial_dofs (jax.Array): Shape (nfree,) free coil DOFs in the extraction
            snapshot's native order, including geometry coefficients and current DOFs.
            Captured initial values.
    """

    problem: ExactSingleStageProblem
    initial_state: ExactSingleStageState
    initial_dofs: jax.Array

    @classmethod
    def from_boozer_surface(
        cls,
        boozer_surface: BoozerSurface | JaxBoozerSurface,
        biotsavart: JaxBiotSavart,
        length_curves: Sequence[Curve],
        *,
        iota_target: float,
        major_radius_target: float,
        length_target: float,
        sDIM: int = 20,
        quasi_poloidal: bool = False,
        tol: float = 1e-13,
        maxiter: float = 40,
    ) -> JaxExactSingleStage:
        """Snapshot a solved exact surface and its field's free DOF layout.

        length_curves must occur directly in biotsavart.coils (not unlisted curves).
        Targets have native QuadraticPenalty semantics and unit weights.
        Iota and major radius use equality penalties; length contributes
        ``0.5 * max(total_length - length_target, 0)**2``.
        Infinite targets retain native inf values and nonfinite gradients;
        only nonfinite component adjoints cause the native ValueError.
        The inner vector includes G; current-derived G is not this route.

        Args:
            boozer_surface (BoozerSurface | JaxBoozerSurface): Successfully solved exact
                tensor surface with explicit G and geometry shape (nphi, ntheta, 3).
            biotsavart (JaxBiotSavart): Field using exactly the solved surface's coil
                objects.
            length_curves (Sequence[Curve]): Selected curves occurring directly in
                biotsavart.coils; not restricted to base curves.
            iota_target (float): Dimensionless rotational-transform target.
            major_radius_target (float): Major-radius target in meters.
            length_target (float): Upper bound on total selected curve length in meters.
            sDIM (int): Half of each auxiliary non-QS grid dimension; grid spans one field
                period; default 20.
            quasi_poloidal (bool): Average over theta when true, otherwise over phi;
                default False.
            tol (float): Euclidean norm threshold for the native masked residual system,
                including label and nonsymmetric z constraints; default 1e-13.
                This norm combines residual and constraint units. At the step cap,
                success uses the norm before the last step. A nonpositive maxiter
                returns the native 1e6 sentinel; initial convergence with a
                positive cap returns the computed initial system norm.
            maxiter (float): Inner Newton step cap; default 40, which may be infinite.
                Positive fractional caps admit steps while the integer count is
                below the cap.

        Returns:
            JaxExactSingleStage: Frozen snapshot evaluator object with explicit state in/out,
                rather than an Optimizable dependency graph. Rebuild after structural or
                fixed-value changes; construction does not mutate native objects.
        """
        if boozer_surface.boozer_type != "exact":
            raise ValueError("JaxExactSingleStage requires an exact BoozerSurface.")
        if not boozer_surface.res["success"]:
            raise ValueError("the initial exact BoozerSurface solve must have succeeded.")
        if boozer_surface.res["G"] is None:
            raise ValueError("JaxExactSingleStage requires G as an explicit inner variable.")
        coils = tuple(biotsavart.coils)
        surface_coils = tuple(boozer_surface.biotsavart.coils)
        if len(coils) != len(surface_coils) or any(a is not b for a, b in zip(coils, surface_coils)):
            raise ValueError("the evaluator field must use the BoozerSurface's coils.")
        surface = boozer_surface.surface
        assert type(surface) is SurfaceXYZTensorFourier
        length_indices = []
        for curve in length_curves:
            index = next((index for index, coil in enumerate(coils) if coil.curve is curve), None)
            if index is None:
                raise ValueError("length_curves must occur directly in biotsavart.coils.")
            length_indices.append(index)
        problem = boozer_problem(biotsavart, surface, boozer_surface.label, boozer_surface.targetlabel)
        reference = commit_in_place(problem.targetlabel)
        auxiliary = SurfaceXYZTensorFourier(
            mpol=surface.mpol, ntor=surface.ntor, stellsym=surface.stellsym, nfp=surface.nfp,
            quadpoints_phi=np.linspace(0, 1 / surface.nfp, 2 * sDIM, endpoint=False),
            quadpoints_theta=np.linspace(0, 1, 2 * sDIM, endpoint=False),
            dofs=surface.dofs,
        )
        # Commit all inputs like the warm-start outputs to avoid a second
        # executable when the first evaluation's state is fed back.
        problem = jax.tree.map(commit_in_place, problem)
        extraction = jax.tree.map(commit_in_place, biotsavart.coil_dof_extraction_spec())
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
            initial_dofs=explicit_device_array(biotsavart.x, dtype=np.float64, reference=reference),
        )

    def evaluate(
        self, dofs: NDArray[np.float64] | jax.Array, state: ExactSingleStageState,
    ) -> ExactSingleStageEvaluation:
        """One fused dispatch and explicit host read, with native solve errors.

        The caller owns state. A raised error returns no state, so retrying
        with the same state is safe. No compilation cache/session is owned here.

        Args:
            dofs (numpy.ndarray | jax.Array): Float64 shape (nfree,) outer free coil DOFs in
                the captured native order.
            state (ExactSingleStageState): Caller-owned warm start for the inner exact
                solve.

        Returns:
            ExactSingleStageEvaluation: Host value/gradient and solve facts, with device-
                resident next state. Singular matrices and nonfinite component adjoints
                raise native errors; no hidden state or native object is changed.
        """
        dofs = explicit_device_array(dofs, dtype=np.float64, reference=state.x)
        if dofs.shape != self.initial_dofs.shape:
            raise ValueError(f"expected coil DOFs of shape {self.initial_dofs.shape}, got {dofs.shape}.")
        result = exact_single_stage_evaluate(self.problem, dofs, state)
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
