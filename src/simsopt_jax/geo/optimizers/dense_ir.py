"""Dense iterative refinement of cached LU factors against a live operator.

Owns the FP64 adaptive refinement used by the ``hybrid_final_dense_ir``
traceable Newton linear solver. Generic linear algebra comes from the acyclic
:mod:`linear_solve` leaf.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax

import jax.numpy as jnp

import jax.scipy.linalg as jsp_linalg

from jax import lax

from simsopt_jax.geo.optimizers.linear_solve import (
    _LinearSolveStatus,
    _device_int32,
    _linear_solve_effective_tolerance_reached,
    _linear_solve_finite,
    _linear_solve_status,
    _terminal_linear_solve_status,
)


_DENSE_IR_NEWTON_REFINEMENT_STEPS = 2

_DENSE_IR_NEWTON_MATVEC_BUDGET = _DENSE_IR_NEWTON_REFINEMENT_STEPS + 1


class _DenseIrContractionTelemetry(NamedTuple):
    residual_relatives: jax.Array
    contraction_ratios: jax.Array
    residual_relative_trace_length: jax.Array
    contraction_finite: jax.Array
    contraction_monotone: jax.Array
    stagnated: jax.Array


class _DenseIrRefinementState(NamedTuple):
    solution: jax.Array
    residual: jax.Array
    status: _LinearSolveStatus
    telemetry: _DenseIrContractionTelemetry


def _run_dense_ir_refinement(
    matvec: Callable[[jax.Array], jax.Array],
    lu_piv: tuple[jax.Array, jax.Array],
    rhs: jax.Array,
    *,
    tol: float | jax.Array,
    max_refinement_corrections: int = _DENSE_IR_NEWTON_REFINEMENT_STEPS,
) -> _DenseIrRefinementState:
    """Adaptively refine cached factors against the live operator.

    The correction budget remains static for JAX compilation, but execution
    stops as soon as the live FP64 residual certifies, ceases to contract, or
    reaches the unit-roundoff stagnation floor. A rejected correction is
    recorded for diagnosis but never replaces the last contracting solution.
    """
    rhs = jnp.asarray(rhs)
    factor_dtype = jnp.asarray(lu_piv[0]).dtype
    solution = jnp.asarray(
        jsp_linalg.lu_solve(lu_piv, jnp.asarray(rhs, dtype=factor_dtype)),
        dtype=rhs.dtype,
    )
    residual = rhs - matvec(solution)
    status = _linear_solve_status(
        solution,
        residual,
        rhs,
        tol=tol,
        iterations=_device_int32(0, like=rhs),
    )
    trace_dtype = rhs.dtype
    residual_relatives = (
        jnp.full(
            (max_refinement_corrections + 1,),
            jnp.asarray(jnp.nan, dtype=trace_dtype),
            dtype=trace_dtype,
        )
        .at[0]
        .set(status.residual_relative)
    )
    contraction_ratios = jnp.full(
        (max_refinement_corrections,),
        jnp.asarray(jnp.nan, dtype=trace_dtype),
        dtype=trace_dtype,
    )
    initial_finite = _linear_solve_finite(solution, residual) & jnp.isfinite(
        status.residual_relative
    )
    initial_state = _DenseIrRefinementState(
        solution=solution,
        residual=residual,
        status=status,
        telemetry=_DenseIrContractionTelemetry(
            residual_relatives=residual_relatives,
            contraction_ratios=contraction_ratios,
            residual_relative_trace_length=_device_int32(1, like=rhs),
            contraction_finite=initial_finite,
            contraction_monotone=jnp.asarray(True, dtype=jnp.bool_),
            stagnated=jnp.asarray(False, dtype=jnp.bool_),
        ),
    )
    unit_roundoff = jnp.asarray(
        jnp.finfo(trace_dtype).eps / 2.0,
        dtype=trace_dtype,
    )

    def refinement_active(state: _DenseIrRefinementState) -> jax.Array:
        correction_count = state.telemetry.residual_relative_trace_length - 1
        return (
            state.telemetry.contraction_finite
            & state.telemetry.contraction_monotone
            & ~state.telemetry.stagnated
            & ~_linear_solve_effective_tolerance_reached(state.status)
            & (correction_count < max_refinement_corrections)
        )

    def refine_once(state: _DenseIrRefinementState) -> _DenseIrRefinementState:
        correction = jsp_linalg.lu_solve(
            lu_piv,
            jnp.asarray(state.residual, dtype=factor_dtype),
        )
        candidate_solution = state.solution + jnp.asarray(
            correction,
            dtype=rhs.dtype,
        )
        candidate_residual = rhs - matvec(candidate_solution)
        trace_index = state.telemetry.residual_relative_trace_length
        candidate_status = _linear_solve_status(
            candidate_solution,
            candidate_residual,
            rhs,
            tol=tol,
            iterations=trace_index,
        )
        previous_relative = state.status.residual_relative
        candidate_relative = candidate_status.residual_relative
        ratio = candidate_relative / jnp.maximum(previous_relative, unit_roundoff)
        candidate_finite = (
            _linear_solve_finite(candidate_solution, candidate_residual)
            & jnp.isfinite(candidate_relative)
            & jnp.isfinite(ratio)
        )
        monotone = candidate_relative < previous_relative
        improvement = previous_relative - candidate_relative
        stagnation_floor = unit_roundoff * jnp.maximum(
            previous_relative,
            jnp.asarray(1.0, dtype=trace_dtype),
        )
        stagnated = candidate_finite & monotone & (improvement <= stagnation_floor)
        accept_candidate = candidate_finite & monotone
        accepted_solution, accepted_residual, accepted_status = lax.cond(
            accept_candidate,
            lambda _: (candidate_solution, candidate_residual, candidate_status),
            lambda _: (state.solution, state.residual, state.status),
            operand=None,
        )
        ratio_index = trace_index - 1
        return _DenseIrRefinementState(
            solution=accepted_solution,
            residual=accepted_residual,
            status=accepted_status,
            telemetry=_DenseIrContractionTelemetry(
                residual_relatives=state.telemetry.residual_relatives.at[
                    trace_index
                ].set(candidate_relative),
                contraction_ratios=state.telemetry.contraction_ratios.at[
                    ratio_index
                ].set(ratio),
                residual_relative_trace_length=trace_index + 1,
                contraction_finite=(
                    state.telemetry.contraction_finite & candidate_finite
                ),
                contraction_monotone=(state.telemetry.contraction_monotone & monotone),
                stagnated=state.telemetry.stagnated | stagnated,
            ),
        )

    refined = (
        initial_state
        if max_refinement_corrections == 0
        else lax.while_loop(refinement_active, refine_once, initial_state)
    )
    correction_count = refined.telemetry.residual_relative_trace_length - 1
    return refined._replace(
        status=refined.status._replace(iterations=correction_count),
    )


def _solve_dense_ir_system_with_status(
    matvec: Callable[[jax.Array], jax.Array],
    lu_piv: tuple[jax.Array, jax.Array],
    rhs: jax.Array,
    *,
    tol: float | jax.Array,
    max_refinement_corrections: int = _DENSE_IR_NEWTON_REFINEMENT_STEPS,
) -> tuple[jax.Array, _LinearSolveStatus]:
    """Refine cached factors adaptively against the current live operator."""
    rhs = jnp.asarray(rhs)
    refined = _run_dense_ir_refinement(
        matvec,
        lu_piv,
        rhs,
        tol=tol,
        max_refinement_corrections=max_refinement_corrections,
    )
    return (
        refined.solution,
        _terminal_linear_solve_status(refined.status, rhs, tol=tol),
    )


__all__ = (
    "_DENSE_IR_NEWTON_MATVEC_BUDGET",
    "_DENSE_IR_NEWTON_REFINEMENT_STEPS",
    "_DenseIrContractionTelemetry",
    "_DenseIrRefinementState",
    "_run_dense_ir_refinement",
    "_solve_dense_ir_system_with_status",
)
