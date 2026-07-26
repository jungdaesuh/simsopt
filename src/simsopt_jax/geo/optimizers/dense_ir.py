"""Dense iterative-refinement policy, telemetry, and refinement owner.

Owns mixed FP32-factor / FP64-certificate policy, adaptive refinement, HMT
contraction certificates, and production mixed / FP64 rebuild paths. Generic
linear algebra comes from the acyclic :mod:`linear_solve` leaf.
"""

from __future__ import annotations

from typing import Callable, Final, NamedTuple

import jax

import jax.numpy as jnp

import jax.scipy.linalg as jsp_linalg

import numpy as np

from jax import lax

from numpy.typing import DTypeLike

from simsopt_jax.backend import get_backend_policy

from simsopt_jax.core._device_scalars import staged_like as _staged_like

from simsopt_jax.numerical_policy import (
    DENSE_IR_HISTORY_CONTRACTION_RATIO_CAPACITY,
    DENSE_IR_HISTORY_RESIDUAL_RELATIVE_CAPACITY,
    MIXED_DENSE_IR_ACCURACY_POLICY,
    MIXED_DENSE_IR_CONTRACTION_IDEAL_GAUSSIAN_FAILURE_PROBABILITY_BOUND,
    MIXED_DENSE_IR_CONTRACTION_NORM_UPPER_LIMIT,
    MIXED_DENSE_IR_CONTRACTION_PROBE_ALPHA,
    MIXED_DENSE_IR_CONTRACTION_PROBE_COUNT,
    MIXED_DENSE_IR_MAX_REFINEMENT_CORRECTIONS,
    DenseIrHistorySource,
    mixed_dense_ir_certificate_dtype_name,
)
from simsopt_jax.runtime.host_boundary import host_array

from simsopt_jax.geo.optimizers.linear_solve import (
    _DENSE_OPERATOR_CHUNK_BATCH_SIZE,
    _LINEAR_SOLVE_ITERATIONS_UNKNOWN,
    _LinearSolveStatus,
    _dense_matrix_backward_error_success,
    _dense_matrix_solve_numerically_safe,
    _dense_matrix_solve_small_solution_success,
    _dense_square_operator_matrix,
    _device_int32,
    _device_scalar,
    _forward_error_tolerance,
    _linear_solve_effective_tolerance_reached,
    _linear_solve_finite,
    _linear_solve_status,
    _solve_dense_square_operator_least_squares_system_with_status,
    _solve_dense_square_operator_lu_system_with_status,
    _terminal_linear_solve_status,
)


_DENSE_IR_NEWTON_REFINEMENT_STEPS = 2

_DENSE_IR_NEWTON_MATVEC_BUDGET = _DENSE_IR_NEWTON_REFINEMENT_STEPS + 1

DENSE_IR_HISTORY_MAX_CORRECTIONS: Final[int] = MIXED_DENSE_IR_MAX_REFINEMENT_CORRECTIONS

_MIXED_DENSE_IR_CONTRACTION_PROBE_COUNT: Final[int] = (
    MIXED_DENSE_IR_CONTRACTION_PROBE_COUNT
)

_MIXED_DENSE_IR_CONTRACTION_PROBE_ALPHA: Final[float] = (
    MIXED_DENSE_IR_CONTRACTION_PROBE_ALPHA
)

_MIXED_DENSE_IR_CONTRACTION_NORM_UPPER_LIMIT: Final[float] = (
    MIXED_DENSE_IR_CONTRACTION_NORM_UPPER_LIMIT
)

_MIXED_DENSE_IR_CONTRACTION_IDEAL_GAUSSIAN_FAILURE_PROBABILITY_BOUND: Final[float] = (
    MIXED_DENSE_IR_CONTRACTION_IDEAL_GAUSSIAN_FAILURE_PROBABILITY_BOUND
)


class _MixedDenseIrResolvedPolicy(NamedTuple):
    """Immutable dense-IR kernel inputs resolved from the accuracy policy."""

    certificate_dtype: np.dtype
    certificate_dtype_name: str
    linear_solve_tolerance_floor: float
    linear_solve_tolerance_cap: float
    forward_error_tolerance_multiplier: float
    max_refinement_corrections: int
    contraction_probe_count: int
    contraction_probe_alpha: float
    contraction_norm_upper_limit: float
    contraction_ideal_gaussian_failure_probability_bound: float


def resolve_mixed_dense_ir_policy() -> _MixedDenseIrResolvedPolicy:
    """Resolve certificate dtype and dense-IR constants from the immutable policy."""
    accuracy = MIXED_DENSE_IR_ACCURACY_POLICY
    certificate_dtype_name = mixed_dense_ir_certificate_dtype_name()
    return _MixedDenseIrResolvedPolicy(
        certificate_dtype=np.dtype(certificate_dtype_name),
        certificate_dtype_name=certificate_dtype_name,
        linear_solve_tolerance_floor=float(accuracy.linear_solve_tolerance_floor),
        linear_solve_tolerance_cap=float(accuracy.linear_solve_tolerance_cap),
        forward_error_tolerance_multiplier=float(
            accuracy.forward_error_tolerance_multiplier
        ),
        max_refinement_corrections=int(MIXED_DENSE_IR_MAX_REFINEMENT_CORRECTIONS),
        contraction_probe_count=int(MIXED_DENSE_IR_CONTRACTION_PROBE_COUNT),
        contraction_probe_alpha=float(MIXED_DENSE_IR_CONTRACTION_PROBE_ALPHA),
        contraction_norm_upper_limit=float(MIXED_DENSE_IR_CONTRACTION_NORM_UPPER_LIMIT),
        contraction_ideal_gaussian_failure_probability_bound=float(
            MIXED_DENSE_IR_CONTRACTION_IDEAL_GAUSSIAN_FAILURE_PROBABILITY_BOUND
        ),
    )


def _require_policy_certificate_dtype(rhs_dtype, *, detail: str) -> np.dtype:
    """Fail closed when a certificate array dtype disagrees with the policy SSOT."""
    resolved = resolve_mixed_dense_ir_policy()
    actual = np.dtype(rhs_dtype)
    if actual != resolved.certificate_dtype:
        raise ValueError(
            f"{detail} requires policy certificate dtype "
            f"{resolved.certificate_dtype_name}; got {actual.name}."
        )
    return resolved.certificate_dtype


class _DenseIrContractionTelemetry(NamedTuple):
    residual_relatives: jax.Array
    contraction_ratios: jax.Array
    residual_relative_trace_length: jax.Array
    contraction_finite: jax.Array
    contraction_monotone: jax.Array
    stagnated: jax.Array


class _DenseIrHistory(NamedTuple):
    """Fixed-shape dense-IR correction history for one selected linear solve."""

    residual_relative_trace: jax.Array
    contraction_ratio_trace: jax.Array
    trace_length: jax.Array
    source_code: jax.Array


class _DenseIrRefinementState(NamedTuple):
    solution: jax.Array
    residual: jax.Array
    status: _LinearSolveStatus
    telemetry: _DenseIrContractionTelemetry


class _MixedDenseIrFallbackTelemetry(NamedTuple):
    """Independent FP64 rebuild evidence kept separate from proposal trust."""

    attempted: jax.Array
    success: jax.Array
    status: _LinearSolveStatus
    factorization_dtype_bits: jax.Array
    factor_application_dtype_bits: jax.Array
    residual_dtype_bits: jax.Array
    refinement: _DenseIrContractionTelemetry


class _MixedDenseIrTrustTelemetry(NamedTuple):
    """Probabilistic forward-accuracy evidence for FP32-origin dense factors."""

    active: jax.Array
    certificate_probe_key_data: jax.Array
    contraction_probe_count: jax.Array
    contraction_probe_alpha: jax.Array
    contraction_operator_norm_upper: jax.Array
    contraction_operator_norm_upper_limit: jax.Array
    contraction_ideal_gaussian_failure_probability_bound: jax.Array
    correction_tail_relative_bound: jax.Array
    forward_error_tolerance: jax.Array
    proposal_factorization_dtype_bits: jax.Array
    factor_application_dtype_bits: jax.Array
    residual_dtype_bits: jax.Array
    certificate_sweep_dtype_bits: jax.Array
    refinement: _DenseIrContractionTelemetry
    proposal_trusted: jax.Array
    fp64_rebuild_count: jax.Array
    fallback: _MixedDenseIrFallbackTelemetry


class _MixedDenseIrSolveStatus(NamedTuple):
    success: jax.Array
    residual: jax.Array
    residual_relative: jax.Array
    iterations: jax.Array
    trust: _MixedDenseIrTrustTelemetry

    def __array__(self, dtype=None):
        return np.asarray(host_array(self.success, dtype=dtype))

    def __bool__(self):
        return bool(np.asarray(self))


def _inactive_mixed_dense_ir_fallback_telemetry(
    reference: jax.Array,
) -> _MixedDenseIrFallbackTelemetry:
    reference = jnp.asarray(reference)
    dtype = reference.dtype
    nan = _device_scalar(np.nan, dtype=dtype)
    false = _staged_like(reference, False, dtype=jnp.bool_)
    residual_relatives = jnp.full(
        (MIXED_DENSE_IR_MAX_REFINEMENT_CORRECTIONS + 1,),
        nan,
        dtype=dtype,
    )
    contraction_ratios = jnp.full(
        (MIXED_DENSE_IR_MAX_REFINEMENT_CORRECTIONS,),
        nan,
        dtype=dtype,
    )
    inactive_refinement = _DenseIrContractionTelemetry(
        residual_relatives=residual_relatives,
        contraction_ratios=contraction_ratios,
        residual_relative_trace_length=_device_int32(0, like=reference),
        contraction_finite=false,
        contraction_monotone=false,
        stagnated=false,
    )
    return _MixedDenseIrFallbackTelemetry(
        attempted=false,
        success=false,
        status=_LinearSolveStatus(
            success=false,
            residual=jnp.full_like(reference, nan),
            residual_relative=nan,
            iterations=_device_int32(
                _LINEAR_SOLVE_ITERATIONS_UNKNOWN,
                like=reference,
            ),
            residual_scale=nan,
            requested_tolerance=nan,
            effective_tolerance=nan,
        ),
        factorization_dtype_bits=_device_int32(0, like=reference),
        factor_application_dtype_bits=_device_int32(0, like=reference),
        residual_dtype_bits=_device_int32(0, like=reference),
        refinement=inactive_refinement,
    )


def _inactive_mixed_dense_ir_trust_telemetry(
    reference: jax.Array,
) -> _MixedDenseIrTrustTelemetry:
    reference = jnp.asarray(reference)
    dtype = reference.dtype
    nan = _device_scalar(np.nan, dtype=dtype)
    false = _staged_like(reference, False, dtype=jnp.bool_)
    inactive_fallback = _inactive_mixed_dense_ir_fallback_telemetry(reference)
    return _MixedDenseIrTrustTelemetry(
        active=false,
        certificate_probe_key_data=_staged_like(
            reference,
            np.zeros((2,), dtype=np.uint32),
            dtype=jnp.uint32,
        ),
        contraction_probe_count=_device_int32(0, like=reference),
        contraction_probe_alpha=nan,
        contraction_operator_norm_upper=nan,
        contraction_operator_norm_upper_limit=nan,
        contraction_ideal_gaussian_failure_probability_bound=_device_scalar(
            np.nan,
            dtype=dtype,
        ),
        correction_tail_relative_bound=nan,
        forward_error_tolerance=nan,
        proposal_factorization_dtype_bits=_device_int32(0, like=reference),
        factor_application_dtype_bits=_device_int32(0, like=reference),
        residual_dtype_bits=_device_int32(0, like=reference),
        certificate_sweep_dtype_bits=_device_int32(0, like=reference),
        refinement=inactive_fallback.refinement,
        proposal_trusted=false,
        fp64_rebuild_count=_device_int32(0, like=reference),
        fallback=inactive_fallback,
    )


def _materialize_dense_ir_proposal_matrix(hvp_fn, x, rhs, *, stab):
    """Build the policy compute-dtype Newton-chord proposal operator."""
    proposal_dtype = np.dtype(get_backend_policy().compute_dtype)
    proposal_x = jnp.asarray(x, dtype=proposal_dtype)
    proposal_stab = _staged_like(proposal_x, stab, dtype=proposal_dtype)

    def proposal_matvec(vector):
        return hvp_fn(proposal_x, vector) + proposal_stab * vector

    return _dense_square_operator_matrix(
        proposal_matvec,
        rhs,
        matrix_dtype=proposal_dtype,
        sweep_dtype=proposal_x.dtype,
    )


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


def _fixed_dense_ir_history(
    telemetry: _DenseIrContractionTelemetry,
    *,
    source: DenseIrHistorySource,
) -> _DenseIrHistory:
    """Pad one dense-IR trace without changing its valid prefix or work budget."""
    residual_count = int(telemetry.residual_relatives.shape[0])
    contraction_count = int(telemetry.contraction_ratios.shape[0])
    if residual_count > DENSE_IR_HISTORY_RESIDUAL_RELATIVE_CAPACITY:
        raise ValueError("Dense-IR residual history exceeds its fixed capacity.")
    if contraction_count > DENSE_IR_HISTORY_CONTRACTION_RATIO_CAPACITY:
        raise ValueError("Dense-IR contraction history exceeds its fixed capacity.")
    trace_dtype = telemetry.residual_relatives.dtype
    residual_relative_trace = (
        _staged_like(
            telemetry.residual_relatives,
            np.full(
                (DENSE_IR_HISTORY_RESIDUAL_RELATIVE_CAPACITY,),
                np.nan,
                dtype=np.dtype(trace_dtype),
            ),
            dtype=trace_dtype,
        )
        .at[:residual_count]
        .set(telemetry.residual_relatives)
    )
    contraction_ratio_trace = (
        _staged_like(
            telemetry.contraction_ratios,
            np.full(
                (DENSE_IR_HISTORY_CONTRACTION_RATIO_CAPACITY,),
                np.nan,
                dtype=np.dtype(trace_dtype),
            ),
            dtype=trace_dtype,
        )
        .at[:contraction_count]
        .set(telemetry.contraction_ratios)
    )
    return _DenseIrHistory(
        residual_relative_trace=residual_relative_trace,
        contraction_ratio_trace=contraction_ratio_trace,
        trace_length=telemetry.residual_relative_trace_length,
        source_code=_device_int32(
            source.value,
            like=telemetry.residual_relatives,
        ),
    )


def _inactive_dense_ir_history(reference: jax.Array) -> _DenseIrHistory:
    """Return the fixed-shape marker for a solve that did not use dense IR."""
    reference = jnp.asarray(reference)
    trace_dtype = reference.dtype
    return _DenseIrHistory(
        residual_relative_trace=_staged_like(
            reference,
            np.full(
                (DENSE_IR_HISTORY_RESIDUAL_RELATIVE_CAPACITY,),
                np.nan,
                dtype=np.dtype(trace_dtype),
            ),
            dtype=trace_dtype,
        ),
        contraction_ratio_trace=_staged_like(
            reference,
            np.full(
                (DENSE_IR_HISTORY_CONTRACTION_RATIO_CAPACITY,),
                np.nan,
                dtype=np.dtype(trace_dtype),
            ),
            dtype=trace_dtype,
        ),
        trace_length=_device_int32(0, like=reference),
        source_code=_device_int32(
            DenseIrHistorySource.NONE.value,
            like=reference,
        ),
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


def _mixed_dense_ir_contraction_operator_norm_upper(
    certificate_matvec: Callable[[jax.Array], jax.Array],
    certificate_apply_factors: tuple[jax.Array, jax.Array],
    rhs: jax.Array,
    *,
    certificate_probe_key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Estimate an upper bound for ``||I - B A||_2`` from runtime probes.

    ``A`` is the live FP64 operator and ``B`` applies the widened FP32-origin
    factors. Under the independent ideal standard Gaussian theorem model,
    Halko-Martinsson-Tropp Lemma 4.1 gives the separately returned exceptional-
    probability bound. Execution uses finite Threefry2x32 pseudorandom normals,
    so that number is not an unconditional PRNG failure probability. The caller
    supplies a runtime key drawn after freezing ``A``; embedding a fixed seed
    here would permit a dangerous direction orthogonal to the fixed probe span.
    No certificate matrix is materialized.
    """
    rhs = jnp.asarray(rhs)
    certificate_dtype = rhs.dtype
    gaussian_probes = jax.random.normal(
        certificate_probe_key,
        shape=(_MIXED_DENSE_IR_CONTRACTION_PROBE_COUNT, int(rhs.shape[0])),
        dtype=certificate_dtype,
    )

    def contraction_image(probe: jax.Array) -> jax.Array:
        live_image = certificate_matvec(probe)
        preconditioned_live_image = jsp_linalg.lu_solve(
            certificate_apply_factors,
            jnp.asarray(live_image, dtype=certificate_dtype),
        )
        return probe - preconditioned_live_image

    contraction_images = lax.map(
        contraction_image,
        gaussian_probes,
        batch_size=_DENSE_OPERATOR_CHUNK_BATCH_SIZE,
    )
    sampled_norm_max = jnp.max(jnp.linalg.norm(contraction_images, axis=1))
    alpha = _device_scalar(
        _MIXED_DENSE_IR_CONTRACTION_PROBE_ALPHA,
        dtype=certificate_dtype,
    )
    gaussian_scale = jnp.sqrt(_device_scalar(2.0 / np.pi, dtype=certificate_dtype))
    norm_upper = alpha * gaussian_scale * sampled_norm_max
    ideal_gaussian_failure_probability_bound = _device_scalar(
        _MIXED_DENSE_IR_CONTRACTION_IDEAL_GAUSSIAN_FAILURE_PROBABILITY_BOUND,
        dtype=certificate_dtype,
    )
    return norm_upper, ideal_gaussian_failure_probability_bound


def _mixed_dense_ir_correction_tail_relative_bound(
    refined: _DenseIrRefinementState,
    lu_piv: tuple[jax.Array, jax.Array],
    rhs: jax.Array,
    *,
    contraction_operator_norm_upper: jax.Array,
) -> jax.Array:
    """Bound unresolved error relative to the unknown exact solution."""
    rhs = jnp.asarray(rhs)
    certificate_dtype = rhs.dtype
    next_correction = jsp_linalg.lu_solve(
        lu_piv,
        jnp.asarray(refined.residual, dtype=certificate_dtype),
    )
    next_correction_norm = jnp.linalg.norm(
        jnp.asarray(next_correction, dtype=certificate_dtype)
    )
    tiny = _device_scalar(jnp.finfo(certificate_dtype).tiny, dtype=certificate_dtype)
    solution_scale = jnp.maximum(jnp.linalg.norm(refined.solution), tiny)
    contraction_denominator = (
        _device_scalar(1.0, dtype=certificate_dtype) - contraction_operator_norm_upper
    )
    correction_relative_to_candidate = jnp.where(
        contraction_denominator > _device_scalar(0.0, dtype=certificate_dtype),
        next_correction_norm / (contraction_denominator * solution_scale),
        _device_scalar(jnp.inf, dtype=certificate_dtype),
    )
    exact_solution_denominator = (
        _device_scalar(1.0, dtype=certificate_dtype) - correction_relative_to_candidate
    )
    return jnp.where(
        exact_solution_denominator > _device_scalar(0.0, dtype=certificate_dtype),
        correction_relative_to_candidate / exact_solution_denominator,
        _device_scalar(jnp.inf, dtype=certificate_dtype),
    )


def _solve_fp64_dense_ir_rebuild_with_telemetry(
    certificate_matvec: Callable[[jax.Array], jax.Array],
    rhs: jax.Array,
    *,
    tol: float | jax.Array,
    certificate_sweep_dtype: DTypeLike,
) -> tuple[jax.Array, _LinearSolveStatus, _MixedDenseIrFallbackTelemetry]:
    """Factor one live certificate-dtype operator and refine both tolerance gates."""
    rhs = jnp.asarray(rhs)
    resolved_policy = resolve_mixed_dense_ir_policy()
    certificate_dtype = _require_policy_certificate_dtype(
        rhs.dtype,
        detail="FP64 dense-IR rebuild",
    )
    certificate_matrix = _dense_square_operator_matrix(
        certificate_matvec,
        rhs,
        matrix_dtype=certificate_dtype,
        sweep_dtype=certificate_sweep_dtype,
    )
    certificate_lu_piv = jsp_linalg.lu_factor(certificate_matrix)
    refined = _run_dense_ir_refinement(
        certificate_matvec,
        certificate_lu_piv,
        rhs,
        tol=tol,
        max_refinement_corrections=resolved_policy.max_refinement_corrections,
    )
    backward_error_success = _dense_matrix_backward_error_success(
        certificate_matrix,
        refined.solution,
        rhs,
        tol=tol,
    )
    solve_safe = _dense_matrix_solve_numerically_safe(
        certificate_matrix,
        refined.solution,
        rhs,
        tol=tol,
        lu_piv=certificate_lu_piv,
        solve_dtype=certificate_dtype,
    )
    small_solution_success = _dense_matrix_solve_small_solution_success(
        refined.solution,
        rhs,
        tol=tol,
    )
    guarded_status = refined.status._replace(
        success=(
            ((refined.status.success | backward_error_success) & solve_safe)
            | (refined.status.success & small_solution_success)
        )
    )
    terminal_status = _terminal_linear_solve_status(guarded_status, rhs, tol=tol)
    certificate_bits = _device_int32(
        certificate_dtype.itemsize * 8,
        like=rhs,
    )
    fallback = _MixedDenseIrFallbackTelemetry(
        attempted=jnp.asarray(True, dtype=jnp.bool_),
        success=terminal_status.success,
        status=terminal_status._replace(residual=refined.residual),
        factorization_dtype_bits=certificate_bits,
        factor_application_dtype_bits=certificate_bits,
        residual_dtype_bits=certificate_bits,
        refinement=refined.telemetry,
    )
    return refined.solution, terminal_status, fallback


def _solve_mixed_dense_ir_operator_with_telemetry(
    proposal_matvec: Callable[[jax.Array], jax.Array],
    certificate_matvec: Callable[[jax.Array], jax.Array],
    rhs: jax.Array,
    *,
    tol: float | jax.Array,
    proposal_dtype: DTypeLike,
    certificate_sweep_dtype: DTypeLike,
    certificate_probe_key: jax.Array,
) -> tuple[jax.Array, _LinearSolveStatus, _MixedDenseIrTrustTelemetry]:
    """Solve with FP32-origin factors and certify the live FP64 operator.

    The proposal operator is materialized and factorized at ``proposal_dtype``.
    Its packed LU values are then widened exactly to the RHS dtype so triangular
    application, solution accumulation, and every residual calculation run at
    certificate precision.  Adaptive refinement always evaluates residuals
    through ``certificate_matvec``. Acceptance requires a probabilistic global
    contraction certificate for ``I - B A`` and the resulting correction-tail
    forward-error bound at the established FP64 tolerance. If either gate fails,
    one unchanged certificate-dtype dense least-squares solve is the sole fallback.

    This is the standard five-precision separation used by the no-factor dense
    Hessian consumers: factor construction may be low precision, while factor
    application, working updates, residuals, and the acceptance certificate stay
    at runtime precision.  It deliberately does not cast a low-precision matrix
    back to FP64 and call that a certificate operator.
    """
    rhs = jnp.asarray(rhs)
    proposal_dtype = np.dtype(proposal_dtype)
    certificate_dtype = _require_policy_certificate_dtype(
        rhs.dtype,
        detail="Mixed dense IR",
    )
    certificate_sweep_dtype = np.dtype(certificate_sweep_dtype)
    if proposal_dtype != np.dtype(np.float32):
        raise ValueError("Mixed dense IR requires FP32 proposal factorization.")
    if certificate_sweep_dtype != certificate_dtype:
        raise ValueError(
            "Mixed dense IR requires certificate-dtype HVP sweeps matching the "
            f"policy certificate dtype ({certificate_dtype.name})."
        )

    proposal_matrix = _dense_square_operator_matrix(
        proposal_matvec,
        rhs,
        matrix_dtype=proposal_dtype,
        sweep_dtype=proposal_dtype,
    )
    proposal_lu, proposal_piv = jsp_linalg.lu_factor(proposal_matrix)
    certificate_apply_factors = (
        jnp.asarray(proposal_lu, dtype=certificate_dtype),
        proposal_piv,
    )
    return _certify_mixed_dense_ir_factors_with_telemetry(
        certificate_matvec,
        certificate_apply_factors,
        rhs,
        tol=tol,
        proposal_factorization_dtype=proposal_dtype,
        certificate_sweep_dtype=certificate_sweep_dtype,
        certificate_probe_key=certificate_probe_key,
    )


def _certify_mixed_dense_ir_factors_with_telemetry(
    certificate_matvec: Callable[[jax.Array], jax.Array],
    certificate_apply_factors: tuple[jax.Array, jax.Array],
    rhs: jax.Array,
    *,
    tol: float | jax.Array,
    proposal_factorization_dtype: DTypeLike,
    certificate_sweep_dtype: DTypeLike,
    certificate_probe_key: jax.Array,
) -> tuple[jax.Array, _LinearSolveStatus, _MixedDenseIrTrustTelemetry]:
    """Certify one frozen FP32-origin factor tuple against a live FP64 operator.

    Factor construction stays outside this seam. The same widened factors feed
    refinement, HMT contraction probes, the correction-tail gate, and the
    optional single FP64 rebuild, making this the production SSOT for both the
    solver and same-factor certificate attribution.
    """
    rhs = jnp.asarray(rhs)
    resolved_policy = resolve_mixed_dense_ir_policy()
    proposal_factorization_dtype = np.dtype(proposal_factorization_dtype)
    certificate_dtype = _require_policy_certificate_dtype(
        rhs.dtype,
        detail="Mixed dense IR",
    )
    certificate_sweep_dtype = np.dtype(certificate_sweep_dtype)
    factor_application_dtype = np.dtype(jnp.asarray(certificate_apply_factors[0]).dtype)
    if proposal_factorization_dtype != np.dtype(np.float32):
        raise ValueError("Mixed dense IR requires FP32 proposal factorization.")
    if certificate_sweep_dtype != certificate_dtype:
        raise ValueError(
            "Mixed dense IR requires certificate-dtype HVP sweeps matching the "
            f"policy certificate dtype ({certificate_dtype.name})."
        )
    if factor_application_dtype != certificate_dtype:
        raise ValueError(
            "Mixed dense IR requires widened factors at certificate precision."
        )

    refined = _run_dense_ir_refinement(
        certificate_matvec,
        certificate_apply_factors,
        rhs,
        tol=tol,
        max_refinement_corrections=resolved_policy.max_refinement_corrections,
    )
    certified_proposal_status = _terminal_linear_solve_status(
        refined.status,
        rhs,
        tol=tol,
    )
    contraction_operator_norm_upper_limit = _device_scalar(
        resolved_policy.contraction_norm_upper_limit,
        dtype=certificate_dtype,
    )
    forward_error_tolerance = _forward_error_tolerance(
        tol=tol,
        dtype=certificate_dtype,
    )

    def run_forward_error_certificate(_: None):
        (
            contraction_operator_norm_upper,
            contraction_ideal_gaussian_failure_probability_bound,
        ) = _mixed_dense_ir_contraction_operator_norm_upper(
            certificate_matvec,
            certificate_apply_factors,
            rhs,
            certificate_probe_key=certificate_probe_key,
        )
        correction_tail_relative_bound = _mixed_dense_ir_correction_tail_relative_bound(
            refined,
            certificate_apply_factors,
            rhs,
            contraction_operator_norm_upper=(contraction_operator_norm_upper),
        )
        contraction_certificate_safe = (
            refined.telemetry.contraction_finite
            & refined.telemetry.contraction_monotone
            & ~refined.telemetry.stagnated
            & jnp.isfinite(contraction_operator_norm_upper)
            & (contraction_operator_norm_upper <= contraction_operator_norm_upper_limit)
            & jnp.isfinite(correction_tail_relative_bound)
            & (correction_tail_relative_bound <= forward_error_tolerance)
        )
        return (
            contraction_operator_norm_upper,
            contraction_ideal_gaussian_failure_probability_bound,
            correction_tail_relative_bound,
            contraction_certificate_safe,
        )

    def skip_forward_error_certificate(_: None):
        nan = _device_scalar(np.nan, dtype=certificate_dtype)
        return nan, nan, nan, jnp.asarray(False, dtype=jnp.bool_)

    forward_error_certificate_active = certified_proposal_status.success
    (
        contraction_operator_norm_upper,
        contraction_ideal_gaussian_failure_probability_bound,
        correction_tail_relative_bound,
        contraction_certificate_safe,
    ) = lax.cond(
        forward_error_certificate_active,
        run_forward_error_certificate,
        skip_forward_error_certificate,
        operand=None,
    )
    proposal_trusted = contraction_certificate_safe & certified_proposal_status.success
    certified_proposal_status = certified_proposal_status._replace(
        success=proposal_trusted
    )

    def keep_certified_proposal(
        _: None,
    ) -> tuple[
        jax.Array,
        _LinearSolveStatus,
        _MixedDenseIrFallbackTelemetry,
    ]:
        return (
            refined.solution,
            certified_proposal_status,
            _inactive_mixed_dense_ir_fallback_telemetry(rhs),
        )

    def rebuild_certificate(
        _: None,
    ) -> tuple[
        jax.Array,
        _LinearSolveStatus,
        _MixedDenseIrFallbackTelemetry,
    ]:
        return _solve_fp64_dense_ir_rebuild_with_telemetry(
            certificate_matvec,
            rhs,
            tol=tol,
            certificate_sweep_dtype=certificate_sweep_dtype,
        )

    solution, status, fallback = lax.cond(
        proposal_trusted,
        keep_certified_proposal,
        rebuild_certificate,
        operand=None,
    )
    telemetry = _MixedDenseIrTrustTelemetry(
        active=jnp.asarray(True, dtype=jnp.bool_),
        certificate_probe_key_data=jax.random.key_data(certificate_probe_key),
        contraction_probe_count=lax.select(
            forward_error_certificate_active,
            _device_int32(
                _MIXED_DENSE_IR_CONTRACTION_PROBE_COUNT,
                like=rhs,
            ),
            _device_int32(0, like=rhs),
        ),
        contraction_probe_alpha=lax.select(
            forward_error_certificate_active,
            _device_scalar(
                _MIXED_DENSE_IR_CONTRACTION_PROBE_ALPHA,
                dtype=certificate_dtype,
            ),
            _device_scalar(np.nan, dtype=certificate_dtype),
        ),
        contraction_operator_norm_upper=contraction_operator_norm_upper,
        contraction_operator_norm_upper_limit=lax.select(
            forward_error_certificate_active,
            contraction_operator_norm_upper_limit,
            _device_scalar(np.nan, dtype=certificate_dtype),
        ),
        contraction_ideal_gaussian_failure_probability_bound=(
            contraction_ideal_gaussian_failure_probability_bound
        ),
        correction_tail_relative_bound=correction_tail_relative_bound,
        forward_error_tolerance=lax.select(
            forward_error_certificate_active,
            forward_error_tolerance,
            _device_scalar(np.nan, dtype=certificate_dtype),
        ),
        proposal_factorization_dtype_bits=_device_int32(
            proposal_factorization_dtype.itemsize * 8,
            like=rhs,
        ),
        factor_application_dtype_bits=_device_int32(
            factor_application_dtype.itemsize * 8,
            like=rhs,
        ),
        residual_dtype_bits=_device_int32(
            np.dtype(certificate_dtype).itemsize * 8,
            like=rhs,
        ),
        certificate_sweep_dtype_bits=_device_int32(
            np.dtype(certificate_sweep_dtype).itemsize * 8,
            like=rhs,
        ),
        refinement=refined.telemetry,
        proposal_trusted=proposal_trusted,
        fp64_rebuild_count=fallback.attempted.astype(jnp.int32),
        fallback=fallback,
    )
    return solution, status, telemetry


def _solve_mixed_dense_ir_operator_with_status(
    proposal_matvec: Callable[[jax.Array], jax.Array],
    certificate_matvec: Callable[[jax.Array], jax.Array],
    rhs: jax.Array,
    *,
    tol: float | jax.Array,
    proposal_dtype: DTypeLike,
    certificate_sweep_dtype: DTypeLike,
    certificate_probe_key: jax.Array,
) -> tuple[jax.Array, _MixedDenseIrSolveStatus]:
    """Return the mixed dense-IR solution/status while retaining one SSOT."""
    solution, status, telemetry = _solve_mixed_dense_ir_operator_with_telemetry(
        proposal_matvec,
        certificate_matvec,
        rhs,
        tol=tol,
        proposal_dtype=proposal_dtype,
        certificate_sweep_dtype=certificate_sweep_dtype,
        certificate_probe_key=certificate_probe_key,
    )
    return solution, _MixedDenseIrSolveStatus(
        success=status.success,
        residual=status.residual,
        residual_relative=status.residual_relative,
        iterations=status.iterations,
        trust=telemetry,
    )


def _solve_dense_ir_system_with_contraction_telemetry(
    matvec: Callable[[jax.Array], jax.Array],
    lu_piv: tuple[jax.Array, jax.Array],
    rhs: jax.Array,
    *,
    tol: float | jax.Array,
    max_refinement_corrections: int = _DENSE_IR_NEWTON_REFINEMENT_STEPS,
) -> tuple[jax.Array, _LinearSolveStatus, _DenseIrContractionTelemetry]:
    """Run adaptive dense IR and retain every attempted live residual ratio."""
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
        refined.telemetry,
    )


__all__ = (
    "DENSE_IR_HISTORY_MAX_CORRECTIONS",
    "_DENSE_IR_NEWTON_MATVEC_BUDGET",
    "_DENSE_IR_NEWTON_REFINEMENT_STEPS",
    "_DenseIrContractionTelemetry",
    "_DenseIrHistory",
    "_DenseIrRefinementState",
    "_MIXED_DENSE_IR_CONTRACTION_IDEAL_GAUSSIAN_FAILURE_PROBABILITY_BOUND",
    "_MIXED_DENSE_IR_CONTRACTION_NORM_UPPER_LIMIT",
    "_MIXED_DENSE_IR_CONTRACTION_PROBE_ALPHA",
    "_MIXED_DENSE_IR_CONTRACTION_PROBE_COUNT",
    "_MixedDenseIrFallbackTelemetry",
    "_MixedDenseIrResolvedPolicy",
    "_MixedDenseIrSolveStatus",
    "_MixedDenseIrTrustTelemetry",
    "_certify_mixed_dense_ir_factors_with_telemetry",
    "_fixed_dense_ir_history",
    "_inactive_dense_ir_history",
    "_inactive_mixed_dense_ir_fallback_telemetry",
    "_inactive_mixed_dense_ir_trust_telemetry",
    "_materialize_dense_ir_proposal_matrix",
    "_mixed_dense_ir_contraction_operator_norm_upper",
    "_mixed_dense_ir_correction_tail_relative_bound",
    "_run_dense_ir_refinement",
    "_solve_dense_ir_system_with_contraction_telemetry",
    "_solve_dense_ir_system_with_status",
    "_solve_dense_square_operator_least_squares_system_with_status",
    "_solve_dense_square_operator_lu_system_with_status",
    "_solve_fp64_dense_ir_rebuild_with_telemetry",
    "_solve_mixed_dense_ir_operator_with_status",
    "_solve_mixed_dense_ir_operator_with_telemetry",
    "resolve_mixed_dense_ir_policy",
)
