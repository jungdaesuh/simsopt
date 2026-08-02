"""
JAX optimizer adapter for the Boozer inner solve.

Reference/oracle methods:
  - ``method="bfgs"``: host-driven SciPy BFGS loop with JAX value/grad.
  - ``method="lbfgs"``: host-driven SciPy L-BFGS-B loop with JAX value/grad.
  - ``method="adam"``: host-driven Adam for noisy/stochastic scalar objectives.

Least-squares methods:
  - ``method="lm"``: host-driven Levenberg-Marquardt for residual-vector
    objectives on the reference lane.
  - ``method="lm-ondevice"``: trace-safe Levenberg-Marquardt for
    residual-vector objectives on the target lane.
  - ``method="lm-minpack-ondevice"``: trace-safe dense-QR
    Levenberg-Marquardt for residual-vector objectives on the target lane.
  - ``method="optimistix-lm-ondevice"``: Optimistix Levenberg-Marquardt lane
    with a Lineax LSMR inner solve.

LM family note:
  Neither ``"lm"`` nor ``"lm-ondevice"`` is a port of MINPACK ``lmder``
  (the algorithm behind ``scipy.optimize.least_squares(method="lm")``).
  Both methods route through ``levenberg_marquardt`` /
  ``levenberg_marquardt_traceable`` (host-driven and trace-safe variants
  of the same JAX LM loop). They are **algorithmically distinct** from
  MINPACK along three load-bearing axes:

  - **Inner solve.** MINPACK uses a pivoted-QR factorization of the
    Jacobian; the JAX LM uses matrix-free GMRES against the
    regularized Gauss-Newton operator ``J^T J + λI`` (no QR pivoting,
    no dense Jacobian factorization in the inner step). See
    ``_lm_iteration`` and ``_gmres_solve_least_squares_system``.
  - **Termination.** MINPACK terminates on independent ``ftol``, ``xtol``,
    and ``gtol`` criteria. The JAX LM now surfaces the matrix-free-computable
    subset as ``info`` codes 1, 2, 3, 5, 6, and 7. It also supports ``gtol``
    as the matrix-free infinity-norm gradient gate when callers explicitly
    provide it; otherwise the legacy ``tol`` gradient gate is preserved.
    MINPACK ``info`` codes 4 and 8 both require the pivoted-QR scaled gradient
    norm and remain outside this matrix-free lane.
  - **Damping update.** MINPACK uses Marquardt's classic
    expand/contract scaling; the JAX LM uses the same symmetric damping
    factors for this matrix-free lane — decrease ``× 0.5`` on
    ``ratio > 0.75`` and increase ``× 2.0`` on ``ratio < 0.25`` or rejected
    steps (see ``_lm_iteration`` and ``_lm_defaults``).

  The opt-in ``"lm-minpack-ondevice"`` lane uses a dense pivoted-QR
  augmented-system step, so it matches MINPACK's QR conditioning model at
  tolerance level without claiming MINPACK's packed-QR byte identity.

  The opt-in ``"optimistix-lm-ondevice"`` lane delegates the nonlinear
  least-squares loop to Optimistix and the inner linear solves to Lineax LSMR.
  It is tolerance-equivalent to the in-tree JAX LM family, not a MINPACK
  parity lane, and requires the Optimistix/Lineax runtime dependencies from
  the ``JAX`` or ``JAX_GPU`` extra.

  Consequence: the JAX LM lanes are **tolerance-equivalent** to MINPACK
  ``lmder`` on well-conditioned fixtures but **not byte-equivalent**;
  ``"lm"`` (reference, host-driven) and ``"lm-ondevice"`` (target,
  trace-safe) are each other's byte-equality oracle, not MINPACK.
  Callers needing MINPACK byte-equality must invoke
  ``scipy.optimize.least_squares(method="lm")`` directly. Use
  ``optimizer_backend="ondevice"`` + ``least_squares_algorithm="lm"``
  to engage the matrix-free on-device LM lane, or
  ``optimizer_backend="ondevice"`` + ``least_squares_algorithm="lm-minpack"``
  to engage the dense pivoted-QR LM lane, or
  ``optimizer_backend="ondevice"`` +
  ``least_squares_algorithm="optimistix-lm"`` to engage the optional
  Optimistix/Lineax LSMR lane.

Target private methods (maintained for the pinned JAX 0.10.0 runtime after the
initial port from the upstream JAX optimizer sources):
  - ``method="bfgs-ondevice"``: JAX on-device BFGS.
  - ``method="lbfgs-ondevice"``: in-tree SciPy-compatible L-BFGS-B state
    machine on the target lane. It uses a host stepwise loop over explicit
    macro-step observables rather than reverse-communication task reads,
    specializes the public no-bounds lane to an Optax-style two-loop L-BFGS
    direction, and preserves SciPy-style counters, statuses, callbacks, and
    inverse-Hessian history. The full L-BFGS-B compact subspace path remains
    available for bounded states and generic private kernels.

Target SciPy-control method:
  - ``method="lbfgs-scipy-jax"``: host SciPy L-BFGS-B control with JAX
    target-lane value/grad evaluations.
  - ``method="lbfgs-scipy-jax-fullgraph"``: host SciPy L-BFGS-B control with
    JAX value/grad evaluations over a caller-owned full Optimizable graph.

Target public stochastic method:
  - ``method="adam-ondevice"``: trace-safe Adam for noisy/stochastic scalar
    objectives on the target lane.

Target public quasi-Newton methods:
  - ``method="optax-lbfgs-ondevice"``: Optax gradient-transformation L-BFGS on
    the target lane. It is not a SciPy L-BFGS-B parity lane.
  - ``method="optimistix-lbfgs-ondevice"``: Optimistix L-BFGS on the target
    lane.

The private methods live in ``optimizer_jax_private/`` and are derived from the
upstream JAX optimizer implementation pinned by this port, so line-search and
iteration behavior stay stable across runtime upgrades. High-level JAX backend
flows route through the target lane only; the host SciPy adapter lives in the
separate ``optimizer_jax_reference`` module. The provenance source is the
upstream ``jax-v0.9.2`` tag (``a659757d768587a81d095a9fab5f0c36f8beb218``);
the supported runtime documented by ``CLAUDE.md`` is the checked local
JAX/JAXLIB 0.10.0 environment.

This module contains zero ``jax._src`` imports. The private package now does as
well; both paths use public JAX APIs.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache, wraps
from itertools import count
from threading import Lock
from typing import (
    Callable,
    Generic,
    NamedTuple,
    Protocol,
    TypeVar,
    cast,
)
from weakref import ref

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import optimistix as optx
from jax import lax
from jax.extend import core as jax_core
from jax.flatten_util import ravel_pytree
from jax.scipy.sparse.linalg import gmres
from scipy.optimize import OptimizeResult

from simsopt_jax.backend import (
    get_backend_config,
    get_backend_policy,
    is_float32_smoke_policy,
    raise_if_strict_jax_fallback,
    strict_target_lane_purity,
    target_lane_purity_requested,
)
from simsopt_jax.core._device_scalars import staged_like as _staged_like
from simsopt_jax.geo._optimizer_backend_choices import (
    CONCRETE_OPTIMIZER_BACKENDS,
    HOST_JAX_OUTER_OPTIMIZER_BACKEND,
    OUTER_OPTIMIZER_BACKEND_MESSAGE,
    RESOLVABLE_OPTIMIZER_BACKEND_MESSAGE,
    TARGET_OUTER_OPTIMIZER_BACKENDS,
    TARGET_PUBLIC_LBFGS_OPTIMIZER_BACKENDS,
    TARGET_SCIPY_CONTROL_OPTIMIZER_BACKENDS,
    VALID_OPTIMIZER_BACKENDS,
    VALID_OUTER_OPTIMIZER_BACKENDS,
    render_invalid_optimizer_backend_message,
)
from simsopt_jax.geo.optimizers._shared import (
    _CACHEABLE_VALUE_AND_GRAD_ATTR,
    _STRUCTURED_SOLVER_CACHE_TOKEN_ATTR,
    PRIVATE_OPTIMIZER_JAX_VERSION,
    _hostify_optimizer_tree,
    _is_flat_optimizer_vector,
    _optimizer_scalar,
    _prepare_optimizer_callable_inputs,
    _prepare_optimizer_pytree_adapter,
    _x64_enabled,
    private_optimizer_runtime_is_supported,
)
from simsopt_jax.geo.optimizers._shared import (
    cast_floating_tree as _cast_floating_tree,
)
from simsopt_jax.geo.optimizers.linear_solve import (
    _CACHEABLE_LINEAR_OPERATOR_ATTR as _CACHEABLE_LINEAR_OPERATOR_ATTR,
    _CACHED_HVP_ATTR as _CACHED_HVP_ATTR,
    _CACHED_JVP_ATTR as _CACHED_JVP_ATTR,
    _DENSE_LINEAR_SOLVE_RESIDUAL_DIMENSION_FACTOR as _DENSE_LINEAR_SOLVE_RESIDUAL_DIMENSION_FACTOR,
    _DENSE_LINEAR_SOLVE_SMALL_SOLUTION_FACTOR as _DENSE_LINEAR_SOLVE_SMALL_SOLUTION_FACTOR,
    _DENSE_OPERATOR_ACTIVATION_BYTES_PER_PARALLEL_COLUMN as _DENSE_OPERATOR_ACTIVATION_BYTES_PER_PARALLEL_COLUMN,
    _DENSE_OPERATOR_CHUNK_BATCH_SIZE as _DENSE_OPERATOR_CHUNK_BATCH_SIZE,
    _DENSE_OPERATOR_CHUNK_BATCH_SIZE_ENV as _DENSE_OPERATOR_CHUNK_BATCH_SIZE_ENV,
    _DENSE_OPERATOR_CHUNK_BATCH_SIZE_FALLBACK as _DENSE_OPERATOR_CHUNK_BATCH_SIZE_FALLBACK,
    _DENSE_OPERATOR_CHUNK_BATCH_SIZE_MAX as _DENSE_OPERATOR_CHUNK_BATCH_SIZE_MAX,
    _DENSE_OPERATOR_DEFAULT_BUDGET_BYTES as _DENSE_OPERATOR_DEFAULT_BUDGET_BYTES,
    _DENSE_OPERATOR_LEGACY_BYTES_PER_PARALLEL_COLUMN as _DENSE_OPERATOR_LEGACY_BYTES_PER_PARALLEL_COLUMN,
    _EXACT_ADJOINT_DENSE_LU as _EXACT_ADJOINT_DENSE_LU,
    _FLOAT64_DENSE_MATRIX_MAX_CONDITION_ESTIMATE as _FLOAT64_DENSE_MATRIX_MAX_CONDITION_ESTIMATE,
    _HAGER_HIGHAM_CONDITION_ITERATIONS as _HAGER_HIGHAM_CONDITION_ITERATIONS,
    _JIT_LINEAR_OPERATOR_CACHE_LOCK as _JIT_LINEAR_OPERATOR_CACHE_LOCK,
    _LINEAR_SOLVE_ITERATIONS_UNKNOWN as _LINEAR_SOLVE_ITERATIONS_UNKNOWN,
    _LinearSolveStatus as _LinearSolveStatus,
    _SQUARE_OPERATOR_GMRES_REFINEMENT_STEPS as _SQUARE_OPERATOR_GMRES_REFINEMENT_STEPS,
    _apply_column_batched_operator as _apply_column_batched_operator,
    _cached_jit_linear_operator as _cached_jit_linear_operator,
    _combine_linear_solve_iteration_counts as _combine_linear_solve_iteration_counts,
    _complete_linear_solve_status as _complete_linear_solve_status,
    _dense_linear_solve_status as _dense_linear_solve_status,
    _dense_matrix_backward_error_success as _dense_matrix_backward_error_success,
    _dense_matrix_condition_estimate as _dense_matrix_condition_estimate,
    _dense_matrix_condition_estimate_numerically_safe as _dense_matrix_condition_estimate_numerically_safe,
    _dense_matrix_nonsingular_threshold as _dense_matrix_nonsingular_threshold,
    _dense_matrix_solve_forward_error_success as _dense_matrix_solve_forward_error_success,
    _dense_matrix_solve_numerically_safe as _dense_matrix_solve_numerically_safe,
    _dense_matrix_solve_small_solution_success as _dense_matrix_solve_small_solution_success,
    _dense_operator_chunk_batch_size_from_budget as _dense_operator_chunk_batch_size_from_budget,
    _dense_square_operator_lu_materialization_allowed as _dense_square_operator_lu_materialization_allowed,
    _dense_square_operator_materialization_allowed as _dense_square_operator_materialization_allowed,
    _dense_square_operator_matrix as _dense_square_operator_matrix,
    _dense_square_operator_matrix_bytes_allowed as _dense_square_operator_matrix_bytes_allowed,
    _dense_square_operator_matrix_dtype as _dense_square_operator_matrix_dtype,
    _device_int32 as _device_int32,
    _device_scalar as _device_scalar,
    _effective_dense_backward_error_tolerance as _effective_dense_backward_error_tolerance,
    _effective_linear_solve_tolerance as _effective_linear_solve_tolerance,
    _factor_dense_hessian as _factor_dense_hessian,
    _forward_error_bound as _forward_error_bound,
    _forward_error_success as _forward_error_success,
    _forward_error_tolerance as _forward_error_tolerance,
    _gmres_iteration_limits as _gmres_iteration_limits,
    _gmres_solve_array_system as _gmres_solve_array_system,
    _hager_higham_inverse_1_norm_estimate as _hager_higham_inverse_1_norm_estimate,
    _hessian_vector_product_fn as _hessian_vector_product_fn,
    _jacobian_linear_operator as _jacobian_linear_operator,
    _jacobian_vector_product_fn as _jacobian_vector_product_fn,
    _linear_solve_effective_tolerance_reached as _linear_solve_effective_tolerance_reached,
    _linear_solve_finite as _linear_solve_finite,
    _linear_solve_iteration_count as _linear_solve_iteration_count,
    _linear_solve_iterations_host_value as _linear_solve_iterations_host_value,
    _linear_solve_residual_scale as _linear_solve_residual_scale,
    _linear_solve_residual_tolerance as _linear_solve_residual_tolerance,
    _linear_solve_solution_or_nan as _linear_solve_solution_or_nan,
    _linear_solve_status as _linear_solve_status,
    _linear_solve_status_iterations as _linear_solve_status_iterations,
    _linear_solve_status_success as _linear_solve_status_success,
    _lu_solve_dense_hessian as _lu_solve_dense_hessian,
    _materialize_dense_hessian as _materialize_dense_hessian,
    _materialize_dense_hessian_host as _materialize_dense_hessian_host,
    _materialize_dense_jacobian as _materialize_dense_jacobian,
    _materialize_dense_linear_operator as _materialize_dense_linear_operator,
    _matrix_one_norm as _matrix_one_norm,
    _place_like_concrete_array as _place_like_concrete_array,
    _place_like_concrete_scalar as _place_like_concrete_scalar,
    _plu_from_lu_piv as _plu_from_lu_piv,
    _relative_residual_1_norm as _relative_residual_1_norm,
    _relative_residual_norm as _relative_residual_norm,
    _resolve_dense_operator_chunk_batch_size as _resolve_dense_operator_chunk_batch_size,
    _run_operator_gmres as _run_operator_gmres,
    _solve_dense_square_operator_least_squares_system_with_status as _solve_dense_square_operator_least_squares_system_with_status,
    _solve_dense_square_operator_lu_system_with_status as _solve_dense_square_operator_lu_system_with_status,
    _solve_jacobian_operator as _solve_jacobian_operator,
    _solve_jacobian_operator_with_status as _solve_jacobian_operator_with_status,
    _solve_jacobian_system_with_status as _solve_jacobian_system_with_status,
    _solve_square_array_system_operator_only as _solve_square_array_system_operator_only,
    _solve_square_vector_system_operator_only as _solve_square_vector_system_operator_only,
    _solve_square_vector_system_operator_only_nonzero_rhs as _solve_square_vector_system_operator_only_nonzero_rhs,
    _symmetrize_dense_hessian as _symmetrize_dense_hessian,
    _terminal_linear_solve_status as _terminal_linear_solve_status,
)
from simsopt_jax.geo.optimizers.dense_ir import (
    DENSE_IR_HISTORY_MAX_CORRECTIONS as DENSE_IR_HISTORY_MAX_CORRECTIONS,
    _DENSE_IR_NEWTON_MATVEC_BUDGET as _DENSE_IR_NEWTON_MATVEC_BUDGET,
    _DENSE_IR_NEWTON_REFINEMENT_STEPS as _DENSE_IR_NEWTON_REFINEMENT_STEPS,
    _DenseIrContractionTelemetry as _DenseIrContractionTelemetry,
    _DenseIrHistory as _DenseIrHistory,
    _DenseIrRefinementState as _DenseIrRefinementState,
    _MIXED_DENSE_IR_CONTRACTION_IDEAL_GAUSSIAN_FAILURE_PROBABILITY_BOUND as _MIXED_DENSE_IR_CONTRACTION_IDEAL_GAUSSIAN_FAILURE_PROBABILITY_BOUND,
    _MIXED_DENSE_IR_CONTRACTION_NORM_UPPER_LIMIT as _MIXED_DENSE_IR_CONTRACTION_NORM_UPPER_LIMIT,
    _MIXED_DENSE_IR_CONTRACTION_PROBE_ALPHA as _MIXED_DENSE_IR_CONTRACTION_PROBE_ALPHA,
    _MIXED_DENSE_IR_CONTRACTION_PROBE_COUNT as _MIXED_DENSE_IR_CONTRACTION_PROBE_COUNT,
    _MixedDenseIrFallbackTelemetry as _MixedDenseIrFallbackTelemetry,
    _MixedDenseIrResolvedPolicy as _MixedDenseIrResolvedPolicy,
    _MixedDenseIrSolveStatus as _MixedDenseIrSolveStatus,
    _MixedDenseIrTrustTelemetry as _MixedDenseIrTrustTelemetry,
    _certify_mixed_dense_ir_factors_with_telemetry as _certify_mixed_dense_ir_factors_with_telemetry,
    _fixed_dense_ir_history as _fixed_dense_ir_history,
    _inactive_dense_ir_history as _inactive_dense_ir_history,
    _inactive_mixed_dense_ir_fallback_telemetry as _inactive_mixed_dense_ir_fallback_telemetry,
    _inactive_mixed_dense_ir_trust_telemetry as _inactive_mixed_dense_ir_trust_telemetry,
    _materialize_dense_ir_proposal_matrix as _materialize_dense_ir_proposal_matrix,
    _mixed_dense_ir_contraction_operator_norm_upper as _mixed_dense_ir_contraction_operator_norm_upper,
    _mixed_dense_ir_correction_tail_relative_bound as _mixed_dense_ir_correction_tail_relative_bound,
    _run_dense_ir_refinement as _run_dense_ir_refinement,
    _solve_dense_ir_system_with_contraction_telemetry as _solve_dense_ir_system_with_contraction_telemetry,
    _solve_dense_ir_system_with_status as _solve_dense_ir_system_with_status,
    _solve_fp64_dense_ir_rebuild_with_telemetry as _solve_fp64_dense_ir_rebuild_with_telemetry,
    _solve_mixed_dense_ir_operator_with_status as _solve_mixed_dense_ir_operator_with_status,
    _solve_mixed_dense_ir_operator_with_telemetry as _solve_mixed_dense_ir_operator_with_telemetry,
    resolve_mixed_dense_ir_policy as resolve_mixed_dense_ir_policy,
)
from simsopt_jax.geo.optimizers.adjoint_linear_solve import (
    _ADJOINT_LINEAR_SOLVER as _ADJOINT_LINEAR_SOLVER,
    _AdjointHessianLinearSolver as _AdjointHessianLinearSolver,
    _EXACT_JACOBIAN_OPERATOR_GMRES_REFINEMENT_STEPS as _EXACT_JACOBIAN_OPERATOR_GMRES_REFINEMENT_STEPS,
    _hessian_linear_operator as _hessian_linear_operator,
    _lineax_lsmr_solver as _lineax_lsmr_solver,
    _require_tree_first_leaf as _require_tree_first_leaf,
    _solve_hessian_least_squares_system_with_status as _solve_hessian_least_squares_system_with_status,
    _solve_hessian_system as _solve_hessian_system,
    _solve_hessian_system_with_status as _solve_hessian_system_with_status,
    _solve_regularized_normal_system_lsmr_j_with_status as _solve_regularized_normal_system_lsmr_j_with_status,
    _solve_symmetric_operator_cg_with_status as _solve_symmetric_operator_cg_with_status,
    adjoint_hessian_stabilization as adjoint_hessian_stabilization,
)
from simsopt_jax.geo.optimizers._evaluation_provider import (
    TargetScipyDeviceEvaluation,
    TargetScipyEvaluationProvider,
)
from simsopt_jax.geo.optimizers.policy import TraceableNewtonLinearSolver
from simsopt_jax.geo.optimizers.private import (
    _minimize_bfgs_private as _private_minimize_bfgs,
)
from simsopt_jax.geo.optimizers.private import (
    _minimize_lbfgs_private as _private_minimize_lbfgs,
)
from simsopt_jax.geo.optimizers.private import (
    _minimize_lbfgs_private_value_and_grad as _private_minimize_lbfgs_value_and_grad,
)
from simsopt_jax.geo.optimizers.private import (
    _private_bfgs_result_to_optimize_result as _private_bfgs_result_to_optimize_result_impl,
)
from simsopt_jax.geo.optimizers.private import (
    _private_lbfgs_result_to_optimize_result as _private_lbfgs_result_to_optimize_result_impl,
)
from simsopt_jax.numerical_policy import (
    MIXED_DENSE_IR_MAX_REFINEMENT_CORRECTIONS,
    NEWTON_ARMIJO_C1,
    PRODUCTION_HYBRID_FINAL_DENSE_IR_BACKEND_CODE,
    DenseIrHistorySource,
    mixed_dense_ir_accuracy_policy,
)
from simsopt_jax.runtime.host_boundary import (
    host_array as _host_array,
    host_bool as _host_bool,
    host_int as _host_int,
    host_scalar as _host_scalar,
)
from simsopt_jax.solve.driver import (
    Driver,
    legacy_reference_least_squares_method,
    legacy_reference_minimize_method,
    legacy_target_method,
    legacy_target_scipy_control_method,
)
from simsopt_jax.solve.minimize_runtime import (
    run_optax_minimize,
    run_optimistix_minimize,
)
from simsopt_jax.solve.optax import OptaxLBFGSOptions
from simsopt_jax.solve.optimistix import OptimistixLBFGSOptions


class _PrivateOptimizerRuntime(NamedTuple):
    minimize_bfgs: Callable
    minimize_lbfgs: Callable
    minimize_lbfgs_value_and_grad: Callable
    bfgs_result_to_optimize_result: Callable
    lbfgs_result_to_optimize_result: Callable


@lru_cache(maxsize=1)
def _private_optimizer_runtime() -> _PrivateOptimizerRuntime:
    return _PrivateOptimizerRuntime(
        minimize_bfgs=_private_minimize_bfgs,
        minimize_lbfgs=_private_minimize_lbfgs,
        minimize_lbfgs_value_and_grad=_private_minimize_lbfgs_value_and_grad,
        bfgs_result_to_optimize_result=_private_bfgs_result_to_optimize_result_impl,
        lbfgs_result_to_optimize_result=_private_lbfgs_result_to_optimize_result_impl,
    )


def _minimize_bfgs_private(*args, **kwargs):
    return _private_optimizer_runtime().minimize_bfgs(*args, **kwargs)


def _minimize_lbfgs_private(*args, **kwargs):
    return _private_optimizer_runtime().minimize_lbfgs(*args, **kwargs)


def _minimize_lbfgs_private_value_and_grad(*args, **kwargs):
    return _private_optimizer_runtime().minimize_lbfgs_value_and_grad(*args, **kwargs)


def _private_bfgs_result_to_optimize_result(*args, **kwargs):
    return _private_optimizer_runtime().bfgs_result_to_optimize_result(*args, **kwargs)


def _private_lbfgs_result_to_optimize_result(*args, **kwargs):
    return _private_optimizer_runtime().lbfgs_result_to_optimize_result(*args, **kwargs)


__all__ = [
    "BoozerInnerDriverOptions",
    "Driver",
    "PRIVATE_OPTIMIZER_JAX_VERSION",
    "ReferenceOptimizerContract",
    "TargetObjectiveRoute",
    "TargetOptimizerContract",
    "TraceableNewtonLinearSolver",
    "adjoint_hessian_stabilization",
    "adam_optimize",
    "adam_optimize_traceable",
    "private_optimizer_runtime_is_supported",
    "VALID_LEAST_SQUARES_ALGORITHMS",
    "VALID_OPTIMIZER_BACKENDS",
    "VALID_OUTER_OPTIMIZER_BACKENDS",
    "CONCRETE_OPTIMIZER_BACKENDS",
    "BOOZER_INNER_OPTIMIZER_BACKENDS",
    "VALID_BOOZER_INNER_OPTIMIZER_BACKENDS",
    "TARGET_OUTER_OPTIMIZER_BACKENDS",
    "TARGET_PUBLIC_LBFGS_OPTIMIZER_BACKENDS",
    "TARGET_SCIPY_CONTROL_OPTIMIZER_BACKENDS",
    "TARGET_X64_REQUIRED_OPTIMIZER_BACKENDS",
    "BOOZER_INNER_X64_REQUIRED_OPTIMIZER_BACKENDS",
    "render_invalid_optimizer_backend_message",
    "render_invalid_boozer_inner_optimizer_backend_message",
    "jax_least_squares",
    "jax_least_squares_optimistix",
    "jax_minimize",
    "levenberg_marquardt",
    "levenberg_marquardt_minpack_traceable",
    "levenberg_marquardt_traceable",
    "newton_polish",
    "newton_polish_traceable",
    "newton_exact",
    "newton_exact_traceable",
    "reference_least_squares",
    "reference_minimize",
    "require_target_backend_x64",
    "require_boozer_inner_backend_x64",
    "resolve_optimizer_backend",
    "resolve_boozer_inner_optimizer_backend",
    "resolve_optimizer_backend_driver",
    "resolve_boozer_inner_driver",
    "resolve_boozer_inner_optimizer_method",
    "resolve_least_squares_optimizer_method",
    "resolve_least_squares_optimizer_driver",
    "resolve_reference_least_squares_optimizer_method",
    "resolve_reference_least_squares_optimizer_driver",
    "resolve_reference_optimizer_contract",
    "resolve_reference_optimizer_driver",
    "resolve_reference_optimizer_method",
    "resolve_target_least_squares_optimizer_method",
    "resolve_target_least_squares_optimizer_driver",
    "resolve_target_optimizer_contract",
    "resolve_target_optimizer_driver",
    "resolve_target_optimizer_method",
    "resolve_optimizer_backend_method",
    "host_jax_least_squares",
    "host_jax_minimize_value_and_grad",
    "reference_driver_method",
    "resolve_reference_outer_loop_optimizer_contract",
    "resolve_target_outer_loop_optimizer_contract",
    "target_driver_method",
    "boozer_inner_driver_legacy_options",
    "wrap_strict_target_lane_value_and_grad",
    "_is_flat_optimizer_vector",
    "target_least_squares",
    "target_minimize",
    "target_optimizer_diagnostic_events",
]


_OUTER_OPTIMIZER_BACKEND_MESSAGE = OUTER_OPTIMIZER_BACKEND_MESSAGE
_RESOLVABLE_OPTIMIZER_BACKEND_MESSAGE = RESOLVABLE_OPTIMIZER_BACKEND_MESSAGE
OPTIMIZER_BACKEND_ROLE = {
    "scipy": "reference",
    "ondevice": "target",
    "scipy-jax": "target-scipy-control",
    "scipy-jax-decomposed": "target-scipy-control",
    HOST_JAX_OUTER_OPTIMIZER_BACKEND: "target-host-control",
    "scipy-jax-fullgraph": "target-scipy-control-fullgraph",
    "optax-lbfgs": "target-optax-lbfgs",
    "optimistix-lbfgs": "target-optimistix-lbfgs",
}
TARGET_X64_REQUIRED_OPTIMIZER_BACKENDS = TARGET_OUTER_OPTIMIZER_BACKENDS | frozenset(
    {HOST_JAX_OUTER_OPTIMIZER_BACKEND}
)
VALID_LEAST_SQUARES_ALGORITHMS = frozenset(
    {"quasi-newton", "lm", "lm-minpack", "optimistix-lm"}
)
_SUPPORTED_METHODS = {
    "adam",
    "adam-ondevice",
    "bfgs",
    "lbfgs",
    "lbfgs-scipy-jax",
    "lbfgs-scipy-jax-decomposed",
    "lbfgs-scipy-jax-fullgraph",
    "optax-lbfgs-ondevice",
    "optimistix-lbfgs-ondevice",
    "lbfgs-trace",
    "bfgs-ondevice",
    "lbfgs-ondevice",
}
_TARGET_LEAST_SQUARES_METHODS = frozenset(
    {"lm-ondevice", "lm-minpack-ondevice", "optimistix-lm-ondevice"}
)
_SUPPORTED_LEAST_SQUARES_METHODS = frozenset({"lm"}) | _TARGET_LEAST_SQUARES_METHODS
_RESIDUAL_LEAST_SQUARES_ALGORITHMS = frozenset({"lm", "lm-minpack", "optimistix-lm"})
_DEFAULT_LM_FTOL = 1e-8
_DEFAULT_LM_XTOL = 1e-8
_OPTIMISTIX_LM_DEFAULT_FTOL = _DEFAULT_LM_FTOL
_OPTIMISTIX_LM_DEFAULT_XTOL = _DEFAULT_LM_XTOL
_OPTIMISTIX_LM_DEFAULT_GTOL = None
_REFERENCE_METHODS = frozenset({"bfgs", "lbfgs"})
_REFERENCE_TRACE_METHODS = frozenset({"lbfgs-trace"})
_REFERENCE_JAX_METHODS = frozenset({"adam"})
_TARGET_PRIVATE_METHODS = frozenset({"bfgs-ondevice", "lbfgs-ondevice"})
_TARGET_SCIPY_CONTROL_METHODS = frozenset(
    {"lbfgs-scipy-jax", "lbfgs-scipy-jax-decomposed", "lbfgs-scipy-jax-fullgraph"}
)
_TARGET_PUBLIC_LBFGS_METHODS = frozenset(
    {"optax-lbfgs-ondevice", "optimistix-lbfgs-ondevice"}
)
_TARGET_PUBLIC_METHODS = frozenset({"adam-ondevice"}) | _TARGET_PUBLIC_LBFGS_METHODS
_TARGET_METHODS = (
    _TARGET_PRIVATE_METHODS | _TARGET_PUBLIC_METHODS | _TARGET_SCIPY_CONTROL_METHODS
)
_TARGET_LBFGSB_METHODS = frozenset({"lbfgs-ondevice"}) | _TARGET_SCIPY_CONTROL_METHODS
_TARGET_PUBLIC_LBFGS_BACKEND_BY_METHOD = {
    "optax-lbfgs-ondevice": "optax-lbfgs",
    "optimistix-lbfgs-ondevice": "optimistix-lbfgs",
}
_UNSUPPORTED_TARGET_LBFGSB_OPTIONS = frozenset({"initial_step_size", "maxgrad"})
_STRICT_REFERENCE_OPTIMIZER_DETAIL = "the host-side SciPy reference optimizer lane"
_STRICT_REFERENCE_JAX_OPTIMIZER_DETAIL = "the host-side JAX reference optimizer lane"
_STRICT_REFERENCE_LEAST_SQUARES_DETAIL = (
    "the host-side reference least-squares optimizer lane"
)
_EISENSTAT_WALKER_GAMMA = 0.9
# α=2 is inlined as ``ratio * ratio`` inside
# ``_eisenstat_walker_choice2_tolerance`` for bit-stable evaluation; see
# Eisenstat & Walker (1996) eq. (2.6).
_EISENSTAT_WALKER_MIN_ETA = 1.0e-12
_EISENSTAT_WALKER_MAX_ETA = 0.5
_EISENSTAT_WALKER_STRICT_CAP_NEAR_TARGET_FACTOR = 100.0
_NEWTON_BACKTRACKING_MAX_STEPS = 8
_SCALAR_VALUE_AND_GRAD_CACHE_LOCK = Lock()
_CACHED_VALUE_AND_GRAD_ATTR = "_simsopt_cached_jit_value_and_grad"
_TARGET_OPTIMIZER_DIAGNOSTIC_EVENT_CALLBACK = ContextVar(
    "simsopt_target_optimizer_diagnostic_event_callback",
    default=None,
)
_TRACEABLE_RUNNER_CACHE_TOKEN_ATTR = "_simsopt_traceable_runner_cache_token"
_TRACEABLE_CALLBACK_LOCK = Lock()
_TRACEABLE_CALLBACK_IDS = count(1)
_TRACEABLE_CALLBACKS: dict[int, Callable[..., object]] = {}
_TRACEABLE_MATVEC_COUNTER_IDS = count(1)
_TRACEABLE_MATVEC_COUNTERS: dict[int, list[int]] = {}
_TRACEABLE_RUNNER_CACHE_LOCK = Lock()
# Explicit traceable cache tokens own semantic reuse; bare callables stay
# isolated by object identity because their closure state is not comparable.
_TRACEABLE_LM_RUNNER_CACHE = {}
_TRACEABLE_NEWTON_POLISH_RUNNER_CACHE = {}
_TRACEABLE_EXACT_NEWTON_RUNNER_CACHE = {}
_TRACEABLE_NEWTON_MATVEC_COUNT_ENV = "SIMSOPT_TRACEABLE_NEWTON_MATVEC_COUNTS"
_TRACEABLE_NEWTON_LINEAR_SOLVER_OPERATOR_GMRES: TraceableNewtonLinearSolver = (
    "operator_gmres"
)
_TRACEABLE_NEWTON_LINEAR_SOLVER_DENSE_LU: TraceableNewtonLinearSolver = "dense_lu"
_TRACEABLE_NEWTON_LINEAR_SOLVER_HYBRID_FINAL_DENSE_LU: TraceableNewtonLinearSolver = (
    "hybrid_final_dense_lu"
)
_TRACEABLE_NEWTON_LINEAR_SOLVER_HYBRID_FINAL_DENSE_IR: TraceableNewtonLinearSolver = (
    "hybrid_final_dense_ir"
)
_TRACEABLE_NEWTON_LINEAR_SOLVERS = frozenset(
    {
        _TRACEABLE_NEWTON_LINEAR_SOLVER_OPERATOR_GMRES,
        _TRACEABLE_NEWTON_LINEAR_SOLVER_DENSE_LU,
        _TRACEABLE_NEWTON_LINEAR_SOLVER_HYBRID_FINAL_DENSE_LU,
        _TRACEABLE_NEWTON_LINEAR_SOLVER_HYBRID_FINAL_DENSE_IR,
    }
)
_TRACEABLE_NEWTON_LINEAR_SOLVER_CODES = {
    _TRACEABLE_NEWTON_LINEAR_SOLVER_OPERATOR_GMRES: 1,
    _TRACEABLE_NEWTON_LINEAR_SOLVER_DENSE_LU: 2,
    _TRACEABLE_NEWTON_LINEAR_SOLVER_HYBRID_FINAL_DENSE_LU: 3,
    _TRACEABLE_NEWTON_LINEAR_SOLVER_HYBRID_FINAL_DENSE_IR: (
        PRODUCTION_HYBRID_FINAL_DENSE_IR_BACKEND_CODE
    ),
}


def _resolve_traceable_newton_linear_solver(
    value: object,
) -> TraceableNewtonLinearSolver:
    """Validate one exact traceable Newton linear-solver selector."""
    if not isinstance(value, str) or value not in _TRACEABLE_NEWTON_LINEAR_SOLVERS:
        names = ", ".join(sorted(_TRACEABLE_NEWTON_LINEAR_SOLVERS))
        raise ValueError(f"linear_solver must be one of: {names}; got {value!r}.")
    return cast(TraceableNewtonLinearSolver, value)


_DEPRECATION_LOGGER = logging.getLogger("simsopt_jax.solve.deprecation")
_DEPRECATED_SOLVE_JAX_CALLSITE_LOCK = Lock()
_DEPRECATED_SOLVE_JAX_CALLSITES: set["_DeprecationCallSite"] = set()
_DEPRECATED_MINIMIZE_METHOD_TO_DRIVER = {
    "adam": "simsopt_adam_host",
    "adam-ondevice": "simsopt_adam",
    "bfgs": "scipy_bfgs",
    "bfgs-ondevice": "simsopt_bfgs",
    "lbfgs": "scipy_lbfgsb",
    "lbfgs-ondevice": "simsopt_lbfgsb",
    "lbfgs-scipy-jax": "scipy_lbfgsb",
    "lbfgs-scipy-jax-decomposed": "scipy_lbfgsb",
    "lbfgs-scipy-jax-fullgraph": "scipy_lbfgsb",
    "optax-lbfgs-ondevice": "optax_lbfgs",
    "optimistix-lbfgs-ondevice": "optimistix_lbfgs",
    "lbfgs-trace": "simsopt_trace_lbfgs",
}
_DEPRECATED_LEAST_SQUARES_METHOD_TO_DRIVER = {
    "lm": "simsopt_lm_gmres_host",
    "lm-minpack-ondevice": "simsopt_lm_qr",
    "lm-ondevice": "simsopt_lm_gmres",
    "optimistix-lm-ondevice": "optimistix_lm",
}


@contextmanager
def target_optimizer_diagnostic_events(callback):
    """Route target optimizer diagnostic events to a stack-scoped callback."""
    token = _TARGET_OPTIMIZER_DIAGNOSTIC_EVENT_CALLBACK.set(callback)
    try:
        yield
    finally:
        _TARGET_OPTIMIZER_DIAGNOSTIC_EVENT_CALLBACK.reset(token)


def _target_optimizer_diagnostic_event_callback():
    return _TARGET_OPTIMIZER_DIAGNOSTIC_EVENT_CALLBACK.get()


def _record_target_optimizer_diagnostic_event(callback, label, **fields):
    if callback is not None:
        callback(label, **fields)


@dataclass(frozen=True)
class _DeprecationCallSite:
    api: str
    filename: str
    lineno: int
    function: str


class _ArrayObjectiveWithArgs(Protocol):
    def __call__(self, x: jax.Array, *args: object) -> jax.Array: ...


class _TraceableNewtonLinearTelemetry(NamedTuple):
    """Typed provenance for one traceable Newton linear solve."""

    ir_correction_count: jax.Array
    ir_residual_relative_trace: jax.Array
    ir_contraction_ratio_trace: jax.Array
    ir_trace_length: jax.Array
    ir_history_source_code: jax.Array
    live_operator_certificate: jax.Array
    factorization_dtype_bits: jax.Array
    factor_application_dtype_bits: jax.Array
    residual_dtype_bits: jax.Array
    factorization_event_increment: jax.Array
    fp64_refactor_retry: jax.Array


class _BoundedMixedCertificateAttempts(NamedTuple):
    """Selected and physical attempts for one bounded second correction."""

    selected: dict[str, jax.Array]
    primary: dict[str, jax.Array]
    retry: dict[str, jax.Array]
    fp64_refactor_retry: jax.Array
    factorization_dtype_bits: jax.Array
    factorization_count: jax.Array


class _BoundedNewtonAttemptTrace(NamedTuple):
    """Fixed-shape device telemetry for ordered physical or logical attempts."""

    active: jax.Array
    accepted: jax.Array
    linear_success: jax.Array
    linear_iterations: jax.Array
    linear_ir_correction_count: jax.Array
    linear_ir_residual_relative: jax.Array
    linear_ir_contraction_ratio: jax.Array
    linear_ir_length: jax.Array
    linear_ir_source_code: jax.Array
    linear_residual: jax.Array
    linear_residual_norm: jax.Array
    linear_residual_scale: jax.Array
    linear_requested_tolerance: jax.Array
    linear_effective_tolerance: jax.Array
    live_operator_certificate: jax.Array
    factorization_dtype_bits: jax.Array
    factor_application_dtype_bits: jax.Array
    residual_dtype_bits: jax.Array
    certificate_value_dtype_bits: jax.Array
    certificate_gradient_dtype_bits: jax.Array
    step_norm: jax.Array
    step_finite: jax.Array
    backtracking_attempted: jax.Array
    backtracking_iterations: jax.Array
    accepted_alpha: jax.Array
    fp64_refactor_retry: jax.Array


def resolve_optimizer_backend(optimizer_backend: str | None) -> str:
    if optimizer_backend is None or optimizer_backend == "auto":
        return get_backend_policy().default_optimizer_backend
    if optimizer_backend not in VALID_OUTER_OPTIMIZER_BACKENDS:
        raise ValueError(_RESOLVABLE_OPTIMIZER_BACKEND_MESSAGE)
    return optimizer_backend


HOST_JAX_BOOZER_OPTIMIZER_BACKEND = "host-jax"
BOOZER_INNER_OPTIMIZER_BACKENDS = CONCRETE_OPTIMIZER_BACKENDS | frozenset(
    {HOST_JAX_BOOZER_OPTIMIZER_BACKEND}
)
VALID_BOOZER_INNER_OPTIMIZER_BACKENDS = (
    frozenset({"auto"}) | BOOZER_INNER_OPTIMIZER_BACKENDS
)
_BOOZER_INNER_OPTIMIZER_BACKEND_DISPLAY_ORDER = (
    "auto",
    "scipy",
    HOST_JAX_BOOZER_OPTIMIZER_BACKEND,
    "ondevice",
)
assert (
    frozenset(_BOOZER_INNER_OPTIMIZER_BACKEND_DISPLAY_ORDER)
    == VALID_BOOZER_INNER_OPTIMIZER_BACKENDS
)
BOOZER_INNER_X64_REQUIRED_OPTIMIZER_BACKENDS = frozenset(
    {HOST_JAX_BOOZER_OPTIMIZER_BACKEND, "ondevice"}
)


def render_invalid_boozer_inner_optimizer_backend_message() -> str:
    names = ", ".join(_BOOZER_INNER_OPTIMIZER_BACKEND_DISPLAY_ORDER)
    return f"optimizer_backend must be one of: {names}."


_BOOZER_INNER_OPTIMIZER_BACKEND_MESSAGE = (
    render_invalid_boozer_inner_optimizer_backend_message()
)


def resolve_boozer_inner_optimizer_backend(optimizer_backend: str | None) -> str:
    if optimizer_backend is None or optimizer_backend == "auto":
        return get_backend_policy().default_optimizer_backend
    if optimizer_backend not in BOOZER_INNER_OPTIMIZER_BACKENDS:
        raise ValueError(_BOOZER_INNER_OPTIMIZER_BACKEND_MESSAGE)
    return optimizer_backend


def _register_traceable_callback(callback: Callable[..., object] | None) -> int:
    if callback is None:
        return 0
    with _TRACEABLE_CALLBACK_LOCK:
        token = next(_TRACEABLE_CALLBACK_IDS)
        _TRACEABLE_CALLBACKS[token] = callback
    return token


def _unregister_traceable_callback(token: int) -> None:
    if token == 0:
        return
    with _TRACEABLE_CALLBACK_LOCK:
        del _TRACEABLE_CALLBACKS[token]


def _lookup_traceable_callback(token, kind: str) -> Callable[..., object]:
    token_value = int(np.asarray(token).reshape(()).item())
    with _TRACEABLE_CALLBACK_LOCK:
        callback = _TRACEABLE_CALLBACKS.get(token_value)
    if callback is None:
        raise RuntimeError(f"Missing active traceable {kind} callback token.")
    return callback


def _lookup_traceable_runner_callable(callable_ref, kind: str):
    callable_fn = callable_ref()
    if callable_fn is None:
        raise RuntimeError(f"Traceable {kind} callable has been released.")
    return callable_fn


def _traceable_newton_matvec_counts_requested() -> bool:
    value = os.environ.get(_TRACEABLE_NEWTON_MATVEC_COUNT_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _register_traceable_matvec_counter(maxiter: int) -> int:
    if maxiter <= 0:
        return 0
    with _TRACEABLE_CALLBACK_LOCK:
        token = next(_TRACEABLE_MATVEC_COUNTER_IDS)
        _TRACEABLE_MATVEC_COUNTERS[token] = [0] * int(maxiter)
    return token


def _unregister_traceable_matvec_counter(token: int) -> None:
    if token == 0:
        return
    with _TRACEABLE_CALLBACK_LOCK:
        _TRACEABLE_MATVEC_COUNTERS.pop(token, None)


def _drain_traceable_matvec_counter(
    token: int, *, rearm: bool = False
) -> tuple[int, ...] | None:
    """Read a counter, optionally resetting its persistent compiled window."""
    if token == 0:
        return None
    with _TRACEABLE_CALLBACK_LOCK:
        if rearm:
            values = _TRACEABLE_MATVEC_COUNTERS.get(token)
            if values is not None:
                _TRACEABLE_MATVEC_COUNTERS[token] = [0] * len(values)
        else:
            values = _TRACEABLE_MATVEC_COUNTERS.pop(token, None)
    if values is None:
        return None
    return tuple(values)


def _is_jax_tracer(value) -> bool:
    return isinstance(value, jax_core.Tracer)


def traceable_newton_matvec_counts_from_token(token: int) -> tuple[int, ...] | None:
    """Read and rearm one opt-in compiled traceable Newton counter."""
    return _drain_traceable_matvec_counter(token, rearm=True)


class _StrongTraceableCallableRef:
    __slots__ = ("_callable_fn",)

    def __init__(self, callable_fn):
        self._callable_fn = callable_fn

    def __call__(self):
        return self._callable_fn


class _TraceableRunnerCallableCell:
    __slots__ = ("_callable_ref",)

    def __init__(self, callable_ref):
        self._callable_ref = callable_ref

    def __call__(self):
        return self._callable_ref()

    def replace_ref(self, callable_ref):
        self._callable_ref = callable_ref

    def owns_ref(self, callable_ref):
        return self._callable_ref is callable_ref


def _traceable_runner_cache_entry_key(callable_fn):
    traceable_token = getattr(callable_fn, _TRACEABLE_RUNNER_CACHE_TOKEN_ATTR, None)
    if traceable_token is not None:
        return ("traceable-token", traceable_token), True
    return ("callable-identity", id(callable_fn)), False


def _traceable_runner_cache_entry_dead(cache, cache_entry_key, callable_ref):
    with _TRACEABLE_RUNNER_CACHE_LOCK:
        cache_entry = cache.get(cache_entry_key)
        if cache_entry is not None and cache_entry[0].owns_ref(callable_ref):
            cache.pop(cache_entry_key, None)


def _traceable_runner_callable_ref(callable_fn, cache=None, cache_entry_key=None):
    if cache is None:
        try:
            return ref(callable_fn)
        except TypeError:
            return None

    def remove_callable_ref(callable_ref):
        _traceable_runner_cache_entry_dead(cache, cache_entry_key, callable_ref)

    try:
        return ref(callable_fn, remove_callable_ref)
    except TypeError:
        return None


def _cached_traceable_runner(cache, callable_fn, cache_key, build_runner):
    callable_ref = _traceable_runner_callable_ref(callable_fn)
    if callable_ref is None:
        return build_runner(_StrongTraceableCallableRef(callable_fn))

    cache_entry_key, is_token_keyed = _traceable_runner_cache_entry_key(callable_fn)
    with _TRACEABLE_RUNNER_CACHE_LOCK:
        cache_entry = cache.get(cache_entry_key)
        if cache_entry is None or (
            not is_token_keyed and cache_entry[0]() is not callable_fn
        ):
            callable_ref = _traceable_runner_callable_ref(
                callable_fn,
                # Token-keyed runners are owned by the semantic token, not by a
                # transient closure object. Identity-keyed runners still clean
                # up with the callable weakref.
                None if is_token_keyed else cache,
                cache_entry_key,
            )
            if callable_ref is None:
                return build_runner(_StrongTraceableCallableRef(callable_fn))
            callable_cell = _TraceableRunnerCallableCell(callable_ref)
            callable_cache = {}
            cache[cache_entry_key] = (callable_cell, callable_cache)
        else:
            callable_cell, callable_cache = cache_entry
            if is_token_keyed and callable_cell() is not callable_fn:
                callable_ref = _traceable_runner_callable_ref(callable_fn)
                callable_cell.replace_ref(callable_ref)
        runner = callable_cache.get(cache_key)
        if runner is None:
            runner = build_runner(callable_cell)
            callable_cache[cache_key] = runner
        return runner


def _shim_caller_stack(caller_frame) -> str:
    code = caller_frame.f_code
    return f"{code.co_filename}:{caller_frame.f_lineno}:{code.co_name}"


def _warn_deprecated_solve_jax_call(
    *,
    api: str,
    method: str,
    translated_driver: str,
    caller_frame,
) -> None:
    callsite = _DeprecationCallSite(
        api=api,
        filename=caller_frame.f_code.co_filename,
        lineno=caller_frame.f_lineno,
        function=caller_frame.f_code.co_name,
    )
    with _DEPRECATED_SOLVE_JAX_CALLSITE_LOCK:
        should_warn = callsite not in _DEPRECATED_SOLVE_JAX_CALLSITES
        if should_warn:
            _DEPRECATED_SOLVE_JAX_CALLSITES.add(callsite)
    if should_warn:
        warnings.warn(
            f"simsopt_jax.geo.optimizers.optimizer.{api} is deprecated; use "
            "simsopt_jax.solve instead. Translation: "
            f"method={method!r} -> driver={translated_driver!r}.",
            DeprecationWarning,
            stacklevel=3,
        )
    _DEPRECATION_LOGGER.info(
        "deprecated_solve_jax_call",
        extra={
            "old_api": api,
            "old_method": method,
            "translated_driver": translated_driver,
            "stack": _shim_caller_stack(caller_frame),
        },
    )


def _invoke_traceable_lm_callback(token, x) -> None:
    callback = _lookup_traceable_callback(token, "LM step")
    callback(_hostify_optimizer_tree(x))


def _invoke_traceable_progress_callback(token, nit, fun, grad_norm) -> None:
    callback = _lookup_traceable_callback(token, "progress")
    callback(nit, fun, grad_norm)


def _invoke_traceable_matvec_counter(token, iteration) -> None:
    token_value = int(np.asarray(token).reshape(()).item())
    iteration_index = int(np.asarray(iteration).reshape(()).item())
    with _TRACEABLE_CALLBACK_LOCK:
        counter = _TRACEABLE_MATVEC_COUNTERS.get(token_value)
        if counter is not None and 0 <= iteration_index < len(counter):
            counter[iteration_index] += 1


@dataclass(frozen=True)
class ReferenceOptimizerContract:
    driver: Driver


class TargetObjectiveRoute(str, Enum):
    ARRAY_NATIVE = "array_native"
    SCIPY_JAX = "scipy_jax"
    SCIPY_JAX_FULLGRAPH = "scipy_jax_fullgraph"
    SCIPY_JAX_DECOMPOSED = "scipy_jax_decomposed"


@dataclass(frozen=True)
class TargetOptimizerContract:
    driver: Driver
    use_least_squares_objective: bool = False
    objective_route: TargetObjectiveRoute = TargetObjectiveRoute.ARRAY_NATIVE


@dataclass(frozen=True)
class BoozerInnerDriverOptions:
    optimizer_backend: str
    limited_memory: bool
    least_squares_algorithm: str


_TARGET_LEAST_SQUARES_DRIVERS = frozenset(
    {Driver.SIMSOPT_LM_GMRES, Driver.SIMSOPT_LM_QR, Driver.OPTIMISTIX_LM}
)
_BOOZER_INNER_DRIVER_OPTIONS = {
    Driver.SCIPY_BFGS: BoozerInnerDriverOptions(
        optimizer_backend="scipy",
        limited_memory=False,
        least_squares_algorithm="quasi-newton",
    ),
    Driver.SCIPY_LBFGSB: BoozerInnerDriverOptions(
        optimizer_backend="scipy",
        limited_memory=True,
        least_squares_algorithm="quasi-newton",
    ),
    Driver.SIMSOPT_BFGS: BoozerInnerDriverOptions(
        optimizer_backend="ondevice",
        limited_memory=False,
        least_squares_algorithm="quasi-newton",
    ),
    Driver.SIMSOPT_LBFGSB: BoozerInnerDriverOptions(
        optimizer_backend="ondevice",
        limited_memory=True,
        least_squares_algorithm="quasi-newton",
    ),
    Driver.SIMSOPT_LM_GMRES_HOST: BoozerInnerDriverOptions(
        optimizer_backend="scipy",
        limited_memory=False,
        least_squares_algorithm="lm",
    ),
    Driver.SIMSOPT_LM_GMRES: BoozerInnerDriverOptions(
        optimizer_backend="ondevice",
        limited_memory=False,
        least_squares_algorithm="lm",
    ),
    Driver.SIMSOPT_LM_QR: BoozerInnerDriverOptions(
        optimizer_backend="ondevice",
        limited_memory=False,
        least_squares_algorithm="lm-minpack",
    ),
    Driver.OPTIMISTIX_LM: BoozerInnerDriverOptions(
        optimizer_backend="ondevice",
        limited_memory=False,
        least_squares_algorithm="optimistix-lm",
    ),
}
# Inverse of ``_BOOZER_INNER_DRIVER_OPTIONS`` keyed on the option triple. Derived
# from the forward table so the option/driver mapping stays single-sourced; every
# resolver that turns an ``(optimizer_backend, limited_memory,
# least_squares_algorithm)`` contract into a typed Boozer inner driver looks this
# up instead of re-deriving the enum by hand.
_BOOZER_INNER_DRIVER_BY_OPTIONS = {
    (
        options.optimizer_backend,
        options.limited_memory,
        options.least_squares_algorithm,
    ): driver
    for driver, options in _BOOZER_INNER_DRIVER_OPTIONS.items()
}
assert len(_BOOZER_INNER_DRIVER_BY_OPTIONS) == len(_BOOZER_INNER_DRIVER_OPTIONS)
_BOOZER_INNER_DRIVER_BY_OPTIONS.update(
    {
        (
            HOST_JAX_BOOZER_OPTIMIZER_BACKEND,
            False,
            "quasi-newton",
        ): Driver.SCIPY_BFGS,
        (
            HOST_JAX_BOOZER_OPTIMIZER_BACKEND,
            True,
            "quasi-newton",
        ): Driver.SCIPY_LBFGSB,
        (
            HOST_JAX_BOOZER_OPTIMIZER_BACKEND,
            False,
            "lm",
        ): Driver.SIMSOPT_LM_GMRES_HOST,
    }
)
# The reference lane exposes a single residual least-squares driver, so every
# residual algorithm coalesces onto the one ``"lm"`` table row for that lane.
_REFERENCE_RESIDUAL_LEAST_SQUARES_OPTION = "lm"


def _boozer_inner_driver_for_options(
    optimizer_backend: str,
    *,
    limited_memory: bool,
    least_squares_algorithm: str,
) -> Driver:
    """Look up the typed Boozer inner driver for a concrete option triple."""
    return _BOOZER_INNER_DRIVER_BY_OPTIONS[
        (optimizer_backend, limited_memory, least_squares_algorithm)
    ]


def reference_driver_method(driver: Driver) -> str:
    """Translate a reference contract driver at the legacy optimizer boundary."""
    return legacy_reference_minimize_method(driver)


def _reference_least_squares_driver_method(driver: Driver) -> str:
    return legacy_reference_least_squares_method(driver)


def target_driver_method(contract: TargetOptimizerContract) -> str:
    """Translate a target contract driver at the legacy optimizer boundary."""
    if contract.driver == Driver.SCIPY_LBFGSB:
        return legacy_target_scipy_control_method(contract.objective_route.value)
    if contract.objective_route != TargetObjectiveRoute.ARRAY_NATIVE:
        raise ValueError(
            "Target objective_route is only valid for Driver.SCIPY_LBFGSB."
        )
    return legacy_target_method(contract.driver)


def _resolved_public_optimizer_backend(optimizer_backend):
    return (
        resolve_optimizer_backend(optimizer_backend)
        if optimizer_backend is None or optimizer_backend == "auto"
        else optimizer_backend
    )


def _target_objective_route_for_optimizer_backend(optimizer_backend):
    if optimizer_backend == "scipy-jax-fullgraph":
        return TargetObjectiveRoute.SCIPY_JAX_FULLGRAPH
    if optimizer_backend == "scipy-jax":
        return TargetObjectiveRoute.SCIPY_JAX
    if optimizer_backend == "scipy-jax-decomposed":
        return TargetObjectiveRoute.SCIPY_JAX_DECOMPOSED
    return TargetObjectiveRoute.ARRAY_NATIVE


def _target_optimizer_contract_for_backend_driver(
    optimizer_backend,
    driver,
    *,
    use_least_squares_objective=False,
):
    return TargetOptimizerContract(
        driver=driver,
        use_least_squares_objective=use_least_squares_objective,
        objective_route=_target_objective_route_for_optimizer_backend(
            optimizer_backend
        ),
    )


def _optimizer_method_for_backend_driver(
    optimizer_backend,
    driver,
    *,
    reference_method,
):
    if optimizer_backend in {"scipy", HOST_JAX_BOOZER_OPTIMIZER_BACKEND}:
        return reference_method(driver)
    return target_driver_method(
        _target_optimizer_contract_for_backend_driver(optimizer_backend, driver)
    )


def boozer_inner_driver_legacy_options(driver: Driver) -> BoozerInnerDriverOptions:
    """Translate a typed Boozer inner driver to the legacy option tuple."""
    if not isinstance(driver, Driver):
        raise TypeError("BoozerSurfaceJAX option 'inner_driver' must be a Driver.")
    try:
        return _BOOZER_INNER_DRIVER_OPTIONS[driver]
    except KeyError as exc:
        allowed = ", ".join(sorted(item.value for item in _BOOZER_INNER_DRIVER_OPTIONS))
        raise ValueError(
            f"BoozerSurfaceJAX inner_driver must be one of: {allowed}. "
            f"Got {driver.value!r}."
        ) from exc


def _raise_if_strict_optimizer_fallback(
    *,
    component: str,
    method: str,
    detail: str,
) -> None:
    raise_if_strict_jax_fallback(
        component=component,
        detail=f"{detail} for method={method!r}",
    )


def _raise_if_target_lane_required(
    *,
    component: str,
    method: str,
    detail: str,
) -> None:
    backend_config = get_backend_config()
    if backend_config.backend != "jax":
        return
    raise RuntimeError(
        f"{component} cannot use {detail} for method={method!r} while simsopt "
        f"backend mode {backend_config.mode!r} requires an ondevice optimizer "
        "method. Select an ondevice optimizer method or switch to the "
        "native_cpu reference backend."
    )


def _require_native_cpu_reference_backend_for_scipy_adapter(
    *,
    component: str,
    method: str,
) -> None:
    backend_config = get_backend_config()
    if backend_config.backend != "jax":
        return
    raise RuntimeError(
        f"{component} cannot use the host SciPy adapter for method={method!r} "
        f"while simsopt backend mode {backend_config.mode!r} requires an "
        "ondevice optimizer method. Select an ondevice optimizer method or "
        "switch to the native_cpu reference backend."
    )


def _require_native_cpu_reference_backend_for_trace_adapter(
    *,
    component: str,
    method: str,
) -> None:
    backend_config = get_backend_config()
    if backend_config.backend != "jax":
        return
    raise RuntimeError(
        f"{component} cannot use the CPU/C++ trace adapter for method={method!r} "
        f"while simsopt backend mode {backend_config.mode!r} requires an "
        "ondevice optimizer method. Select an ondevice optimizer method or "
        "switch to the native_cpu reference backend."
    )


def _mark_cacheable_jit_linear_operator(fun):
    # Same callable mutability contract as ``mark_cacheable_jit_value_and_grad``.
    setattr(fun, _CACHEABLE_LINEAR_OPERATOR_ATTR, True)
    return fun


def _mark_traceable_runner_cacheable(fun, *, cache_token):
    # Same contract as ``mark_cacheable_jit_value_and_grad``.
    setattr(fun, _TRACEABLE_RUNNER_CACHE_TOKEN_ATTR, cache_token)
    return fun


def _mark_structured_private_solver_cacheable(fun, *, cache_token):
    # Same contract as ``mark_cacheable_jit_value_and_grad``.
    setattr(fun, _STRUCTURED_SOLVER_CACHE_TOKEN_ATTR, cache_token)
    return _mark_traceable_runner_cacheable(fun, cache_token=cache_token)


_StrictDecisionT = TypeVar("_StrictDecisionT")
_StrictDevicePacketT = TypeVar("_StrictDevicePacketT")
_StrictHostPacketT = TypeVar("_StrictHostPacketT")
_StrictDeviceValueT = TypeVar("_StrictDeviceValueT")
_StrictDeviceGradientT = TypeVar("_StrictDeviceGradientT")
_StrictHostValueT = TypeVar("_StrictHostValueT")
_StrictHostGradientT = TypeVar("_StrictHostGradientT")


@dataclass(frozen=True)
class _StrictTargetScipyEvaluationProvider(
    Generic[
        _StrictDecisionT,
        _StrictDevicePacketT,
        _StrictHostPacketT,
        _StrictDeviceValueT,
        _StrictDeviceGradientT,
        _StrictHostValueT,
        _StrictHostGradientT,
    ]
):
    provider: TargetScipyEvaluationProvider[
        _StrictDecisionT,
        _StrictDevicePacketT,
        _StrictHostPacketT,
        _StrictDeviceValueT,
        _StrictDeviceGradientT,
        _StrictHostValueT,
        _StrictHostGradientT,
    ]

    def __call__(
        self,
        decision_vector: _StrictDecisionT,
        /,
    ) -> tuple[_StrictDeviceValueT, _StrictDeviceGradientT]:
        with strict_target_lane_purity():
            return self.provider(decision_vector)

    def evaluate_target_scipy(
        self,
        decision_vector: _StrictDecisionT,
        host_decision_vector: np.ndarray,
        /,
    ) -> TargetScipyDeviceEvaluation[
        _StrictDevicePacketT,
        _StrictHostPacketT,
        _StrictHostValueT,
        _StrictHostGradientT,
    ]:
        with strict_target_lane_purity():
            return self.provider.evaluate_target_scipy(
                decision_vector,
                host_decision_vector,
            )


def wrap_strict_target_lane_value_and_grad(fun):
    """Wrap target-lane value/grad calls in the stack-scoped purity guard."""
    if not target_lane_purity_requested():
        return fun
    if isinstance(fun, TargetScipyEvaluationProvider):
        return _StrictTargetScipyEvaluationProvider(provider=fun)

    @wraps(fun)
    def wrapped(*args, **kwargs):
        with strict_target_lane_purity():
            return fun(*args, **kwargs)

    return wrapped


def _cached_jit_value_and_grad(fun):
    if not getattr(fun, _CACHEABLE_VALUE_AND_GRAD_ATTR, False):
        return jax.jit(jax.value_and_grad(fun, argnums=0))
    cached = getattr(fun, _CACHED_VALUE_AND_GRAD_ATTR, None)
    if cached is not None:
        return cached
    compiled = jax.jit(jax.value_and_grad(fun, argnums=0))
    # Double-checked install under the cache lock. ``fun`` has already
    # been marked via ``mark_cacheable_jit_value_and_grad`` (the marker
    # check above gated this branch), so ``setattr`` cannot raise.
    with _SCALAR_VALUE_AND_GRAD_CACHE_LOCK:
        cached = getattr(fun, _CACHED_VALUE_AND_GRAD_ATTR, None)
        if cached is not None:
            return cached
        setattr(fun, _CACHED_VALUE_AND_GRAD_ATTR, compiled)
        return compiled


def _finalize_optimizer_result(result, adapter):
    if adapter is None:
        return result
    return adapter.finalize_result(result)


def resolve_optimizer_backend_driver(optimizer_backend, *, limited_memory):
    """Map the public backend contract to the typed optimizer driver."""
    if optimizer_backend is None or optimizer_backend == "auto":
        optimizer_backend = resolve_optimizer_backend(optimizer_backend)
    if optimizer_backend not in VALID_OUTER_OPTIMIZER_BACKENDS:
        raise ValueError(_OUTER_OPTIMIZER_BACKEND_MESSAGE)
    if optimizer_backend in {"scipy", HOST_JAX_OUTER_OPTIMIZER_BACKEND}:
        return resolve_reference_optimizer_driver(limited_memory=limited_memory)
    if optimizer_backend in TARGET_SCIPY_CONTROL_OPTIMIZER_BACKENDS:
        return Driver.SCIPY_LBFGSB
    if optimizer_backend == "optax-lbfgs":
        return Driver.OPTAX_LBFGS
    if optimizer_backend == "optimistix-lbfgs":
        return Driver.OPTIMISTIX_LBFGS
    return resolve_target_optimizer_driver(limited_memory=limited_memory)


def resolve_optimizer_backend_method(optimizer_backend, *, limited_memory):
    """Map the public backend contract to the concrete optimizer method."""
    resolved_backend = _resolved_public_optimizer_backend(optimizer_backend)
    driver = resolve_optimizer_backend_driver(
        resolved_backend,
        limited_memory=limited_memory,
    )
    return _optimizer_method_for_backend_driver(
        resolved_backend,
        driver,
        reference_method=reference_driver_method,
    )


def resolve_reference_optimizer_driver(*, limited_memory):
    """Resolve the CPU/reference scalar optimizer driver."""
    return _boozer_inner_driver_for_options(
        "scipy",
        limited_memory=limited_memory,
        least_squares_algorithm="quasi-newton",
    )


def resolve_reference_optimizer_method(*, limited_memory):
    """Resolve the CPU/reference scalar optimizer method."""
    return reference_driver_method(
        resolve_reference_optimizer_driver(limited_memory=limited_memory)
    )


def resolve_target_optimizer_driver(*, limited_memory):
    """Resolve the JAX target scalar optimizer driver."""
    return _boozer_inner_driver_for_options(
        "ondevice",
        limited_memory=limited_memory,
        least_squares_algorithm="quasi-newton",
    )


def resolve_target_optimizer_method(*, limited_memory):
    """Resolve the JAX target scalar optimizer method."""
    return target_driver_method(
        _target_optimizer_contract_for_backend_driver(
            "ondevice", resolve_target_optimizer_driver(limited_memory=limited_memory)
        )
    )


def _scipy_control_least_squares_algorithm_message(optimizer_backend):
    return (
        f"optimizer_backend={optimizer_backend!r} only supports "
        "least_squares_algorithm='quasi-newton'."
    )


def _validate_least_squares_algorithm(least_squares_algorithm):
    if least_squares_algorithm not in VALID_LEAST_SQUARES_ALGORITHMS:
        allowed = ", ".join(sorted(VALID_LEAST_SQUARES_ALGORITHMS))
        raise ValueError(f"least_squares_algorithm must be one of: {allowed}.")


def _resolve_concrete_least_squares_optimizer_driver(
    optimizer_backend,
    *,
    limited_memory,
    least_squares_algorithm,
):
    _validate_least_squares_algorithm(least_squares_algorithm)
    if least_squares_algorithm == "quasi-newton":
        return _boozer_inner_driver_for_options(
            optimizer_backend,
            limited_memory=limited_memory,
            least_squares_algorithm="quasi-newton",
        )
    if limited_memory:
        raise ValueError(
            f"least_squares_algorithm={least_squares_algorithm!r} is incompatible "
            "with limited_memory=True."
        )
    if optimizer_backend in {"scipy", HOST_JAX_BOOZER_OPTIMIZER_BACKEND}:
        least_squares_algorithm = _REFERENCE_RESIDUAL_LEAST_SQUARES_OPTION
    return _boozer_inner_driver_for_options(
        optimizer_backend,
        limited_memory=limited_memory,
        least_squares_algorithm=least_squares_algorithm,
    )


def resolve_boozer_inner_driver(
    optimizer_backend,
    *,
    limited_memory,
    least_squares_algorithm,
):
    """Map the Boozer LS option contract to the typed inner driver."""
    resolved_optimizer_backend = resolve_boozer_inner_optimizer_backend(
        optimizer_backend
    )
    return _resolve_concrete_least_squares_optimizer_driver(
        resolved_optimizer_backend,
        limited_memory=limited_memory,
        least_squares_algorithm=least_squares_algorithm,
    )


def resolve_boozer_inner_optimizer_method(
    optimizer_backend,
    *,
    limited_memory,
    least_squares_algorithm,
):
    resolved_optimizer_backend = resolve_boozer_inner_optimizer_backend(
        optimizer_backend
    )
    driver = _resolve_concrete_least_squares_optimizer_driver(
        resolved_optimizer_backend,
        limited_memory=limited_memory,
        least_squares_algorithm=least_squares_algorithm,
    )
    return _optimizer_method_for_backend_driver(
        resolved_optimizer_backend,
        driver,
        reference_method=_reference_least_squares_driver_method,
    )


def resolve_least_squares_optimizer_driver(
    optimizer_backend,
    *,
    limited_memory,
    least_squares_algorithm,
):
    """Map the LS backend contract to the typed least-squares driver."""
    optimizer_backend = resolve_optimizer_backend(optimizer_backend)
    _validate_least_squares_algorithm(least_squares_algorithm)
    if least_squares_algorithm == "quasi-newton":
        return resolve_optimizer_backend_driver(
            optimizer_backend,
            limited_memory=limited_memory,
        )
    if (
        optimizer_backend in TARGET_SCIPY_CONTROL_OPTIMIZER_BACKENDS
        or optimizer_backend in TARGET_PUBLIC_LBFGS_OPTIMIZER_BACKENDS
    ):
        raise ValueError(
            _scipy_control_least_squares_algorithm_message(optimizer_backend)
        )
    return _resolve_concrete_least_squares_optimizer_driver(
        optimizer_backend,
        limited_memory=limited_memory,
        least_squares_algorithm=least_squares_algorithm,
    )


def resolve_least_squares_optimizer_method(
    optimizer_backend,
    *,
    limited_memory,
    least_squares_algorithm,
):
    """Map the LS backend contract to the concrete least-squares method."""
    optimizer_backend = resolve_optimizer_backend(optimizer_backend)
    driver = resolve_least_squares_optimizer_driver(
        optimizer_backend,
        limited_memory=limited_memory,
        least_squares_algorithm=least_squares_algorithm,
    )
    return _optimizer_method_for_backend_driver(
        optimizer_backend,
        driver,
        reference_method=_reference_least_squares_driver_method,
    )


def resolve_reference_least_squares_optimizer_driver(
    *,
    limited_memory,
    least_squares_algorithm,
):
    """Resolve the CPU/reference least-squares optimizer driver."""
    return _resolve_concrete_least_squares_optimizer_driver(
        "scipy",
        limited_memory=limited_memory,
        least_squares_algorithm=least_squares_algorithm,
    )


def resolve_reference_least_squares_optimizer_method(
    *,
    limited_memory,
    least_squares_algorithm,
):
    """Resolve the CPU/reference least-squares optimizer method."""
    return _reference_least_squares_driver_method(
        resolve_reference_least_squares_optimizer_driver(
            limited_memory=limited_memory,
            least_squares_algorithm=least_squares_algorithm,
        )
    )


def resolve_target_least_squares_optimizer_driver(
    *,
    limited_memory,
    least_squares_algorithm,
):
    """Resolve the JAX target least-squares optimizer driver."""
    return _resolve_concrete_least_squares_optimizer_driver(
        "ondevice",
        limited_memory=limited_memory,
        least_squares_algorithm=least_squares_algorithm,
    )


def resolve_target_least_squares_optimizer_method(
    *,
    limited_memory,
    least_squares_algorithm,
):
    """Resolve the JAX target least-squares optimizer method."""
    return target_driver_method(
        _target_optimizer_contract_for_backend_driver(
            "ondevice",
            resolve_target_least_squares_optimizer_driver(
                limited_memory=limited_memory,
                least_squares_algorithm=least_squares_algorithm,
            ),
        )
    )


def require_target_backend_x64(optimizer_backend):
    """Fail fast when a target-lane backend is requested without float64."""
    optimizer_backend = resolve_optimizer_backend(optimizer_backend)
    if optimizer_backend not in TARGET_X64_REQUIRED_OPTIMIZER_BACKENDS:
        return
    if _x64_enabled():
        return
    if is_float32_smoke_policy(get_backend_policy()):
        return
    role = OPTIMIZER_BACKEND_ROLE[optimizer_backend]
    raise RuntimeError(
        f"optimizer_backend='{optimizer_backend}' ({role}) requires "
        "jax_enable_x64=True before import/use."
    )


def require_boozer_inner_backend_x64(optimizer_backend):
    """Fail fast when a Boozer inner target-kernel backend lacks float64."""
    optimizer_backend = resolve_boozer_inner_optimizer_backend(optimizer_backend)
    if optimizer_backend not in BOOZER_INNER_X64_REQUIRED_OPTIMIZER_BACKENDS:
        return
    if _x64_enabled():
        return
    if is_float32_smoke_policy(get_backend_policy()):
        return
    role = (
        "target-host-control"
        if optimizer_backend == HOST_JAX_BOOZER_OPTIMIZER_BACKEND
        else OPTIMIZER_BACKEND_ROLE[optimizer_backend]
    )
    raise RuntimeError(
        f"optimizer_backend='{optimizer_backend}' ({role}) requires "
        "jax_enable_x64=True before import/use."
    )


def resolve_reference_optimizer_contract(
    field_backend,
    optimizer_backend,
    *,
    limited_memory,
    component_label,
):
    """Resolve the explicit CPU/reference optimizer contract."""
    if optimizer_backend not in VALID_OUTER_OPTIMIZER_BACKENDS:
        raise ValueError(_OUTER_OPTIMIZER_BACKEND_MESSAGE)
    if field_backend == "jax":
        raise ValueError(
            f"{component_label} with backend='jax' requires "
            "optimizer_backend='ondevice', optimizer_backend='scipy-jax', "
            "optimizer_backend='scipy-jax-decomposed', "
            "optimizer_backend='host-jax', "
            "optimizer_backend='scipy-jax-fullgraph', "
            "optimizer_backend='optax-lbfgs', or "
            "optimizer_backend='optimistix-lbfgs'. "
            "The SciPy/reference optimizer lane is CPU/reference-only."
        )
    if field_backend != "jax" and optimizer_backend != "scipy":
        raise ValueError(
            f"{component_label} CPU/reference lane only supports "
            "optimizer_backend='scipy'."
        )
    return ReferenceOptimizerContract(
        driver=resolve_reference_optimizer_driver(
            limited_memory=limited_memory,
        ),
    )


def resolve_target_optimizer_contract(
    field_backend,
    optimizer_backend,
    *,
    limited_memory,
    component_label,
    least_squares_algorithm="quasi-newton",
):
    """Resolve the explicit JAX target optimizer contract."""
    if optimizer_backend not in VALID_OUTER_OPTIMIZER_BACKENDS:
        raise ValueError(_OUTER_OPTIMIZER_BACKEND_MESSAGE)
    if (
        field_backend != "jax"
        or optimizer_backend not in TARGET_OUTER_OPTIMIZER_BACKENDS
    ):
        raise ValueError(
            f"{component_label} with backend='jax' requires "
            "optimizer_backend='ondevice', optimizer_backend='scipy-jax', "
            "optimizer_backend='scipy-jax-decomposed', "
            "optimizer_backend='host-jax', "
            "optimizer_backend='scipy-jax-fullgraph', "
            "optimizer_backend='optax-lbfgs', or "
            "optimizer_backend='optimistix-lbfgs'. "
            "The SciPy/reference optimizer lane is CPU/reference-only."
        )
    require_target_backend_x64(optimizer_backend)
    if optimizer_backend in TARGET_SCIPY_CONTROL_OPTIMIZER_BACKENDS:
        if least_squares_algorithm != "quasi-newton":
            raise ValueError(
                _scipy_control_least_squares_algorithm_message(optimizer_backend)
            )
        return _target_optimizer_contract_for_backend_driver(
            optimizer_backend,
            Driver.SCIPY_LBFGSB,
        )
    if optimizer_backend in TARGET_PUBLIC_LBFGS_OPTIMIZER_BACKENDS:
        if least_squares_algorithm != "quasi-newton":
            raise ValueError(
                _scipy_control_least_squares_algorithm_message(optimizer_backend)
            )
        return _target_optimizer_contract_for_backend_driver(
            optimizer_backend,
            resolve_optimizer_backend_driver(
                optimizer_backend,
                limited_memory=limited_memory,
            ),
        )
    driver = resolve_target_least_squares_optimizer_driver(
        limited_memory=limited_memory,
        least_squares_algorithm=least_squares_algorithm,
    )
    return _target_optimizer_contract_for_backend_driver(
        optimizer_backend,
        driver=driver,
        use_least_squares_objective=driver in _TARGET_LEAST_SQUARES_DRIVERS,
    )


def resolve_reference_outer_loop_optimizer_contract(
    field_backend,
    optimizer_backend,
    *,
    component_label,
):
    """Resolve the CPU/reference outer-loop contract."""
    return resolve_reference_optimizer_contract(
        field_backend,
        optimizer_backend,
        limited_memory=True,
        component_label=component_label,
    )


def resolve_target_outer_loop_optimizer_contract(
    field_backend,
    optimizer_backend,
    *,
    component_label,
    least_squares_algorithm="quasi-newton",
):
    """Resolve the JAX target outer-loop contract."""
    limited_memory = least_squares_algorithm not in _RESIDUAL_LEAST_SQUARES_ALGORITHMS
    return resolve_target_optimizer_contract(
        field_backend,
        optimizer_backend,
        limited_memory=limited_memory,
        component_label=component_label,
        least_squares_algorithm=least_squares_algorithm,
    )


def _least_squares_cost(residual):
    residual = jnp.ravel(jnp.asarray(residual))
    return _device_scalar(0.5, dtype=residual.dtype) * jnp.vdot(residual, residual).real


def _least_squares_linearization_from_jacobian(residual, jacobian):
    residual = jnp.ravel(jnp.asarray(residual))
    jacobian = jnp.asarray(jacobian)
    gradient = jacobian.T @ residual
    hessian = jacobian.T @ jacobian
    return gradient, hessian


def _dense_lm_state_from_residual_jacobian(residual, jacobian):
    residual = jnp.ravel(jnp.asarray(residual))
    jacobian = jnp.asarray(jacobian)
    gradient, hessian = _least_squares_linearization_from_jacobian(
        residual,
        jacobian,
    )
    return {
        "residual": residual,
        "residual_jacobian": jacobian,
        "grad": gradient,
        "hessian": hessian,
        "fun": _least_squares_cost(residual),
        "grad_norm_inf": _tree_inf_norm(gradient),
    }


def _tree_zeros_like(tree):
    return jax.tree.map(
        lambda leaf: jnp.zeros_like(jnp.asarray(leaf)),
        tree,
    )


def _tree_scalar_mul(tree, scalar):
    scalar = jnp.asarray(scalar)
    return jax.tree.map(lambda leaf: scalar * jnp.asarray(leaf), tree)


def _tree_add(lhs, rhs):
    return jax.tree.map(
        lambda lhs_leaf, rhs_leaf: jnp.asarray(lhs_leaf) + jnp.asarray(rhs_leaf),
        lhs,
        rhs,
    )


def _tree_sub(lhs, rhs):
    return jax.tree.map(
        lambda lhs_leaf, rhs_leaf: jnp.asarray(lhs_leaf) - jnp.asarray(rhs_leaf),
        lhs,
        rhs,
    )


def _tree_square(tree):
    return jax.tree.map(
        lambda leaf: jnp.square(jnp.asarray(leaf)),
        tree,
    )


def _tree_bias_correction(tree, correction):
    correction = jnp.asarray(correction)
    return jax.tree.map(
        lambda leaf: jnp.asarray(leaf) / correction,
        tree,
    )


def _tree_adam_step(mean, variance, *, step_size, eps):
    step_size = jnp.asarray(step_size)
    eps = jnp.asarray(eps)
    return jax.tree.map(
        lambda mean_leaf, variance_leaf: (
            step_size
            * jnp.asarray(mean_leaf)
            / (jnp.sqrt(jnp.asarray(variance_leaf)) + eps)
        ),
        mean,
        variance,
    )


def _tree_vdot_real(lhs, rhs):
    lhs_leaves, lhs_tree = jax.tree.flatten(lhs)
    rhs_leaves, rhs_tree = jax.tree.flatten(rhs)
    if lhs_tree != rhs_tree:
        raise ValueError("Tree dot products require matching pytree structures.")
    if not lhs_leaves:
        return _device_scalar(0.0)
    dtype = jnp.result_type(
        *[jnp.asarray(leaf).dtype for leaf in lhs_leaves + rhs_leaves]
    )
    total = jnp.asarray(0.0, dtype=dtype)
    for lhs_leaf, rhs_leaf in zip(lhs_leaves, rhs_leaves):
        total = total + jnp.vdot(
            jnp.ravel(jnp.asarray(lhs_leaf)),
            jnp.ravel(jnp.asarray(rhs_leaf)),
        ).real.astype(dtype)
    return total


def _tree_inf_norm(tree):
    leaves = jax.tree.leaves(tree)
    if not leaves:
        return _device_scalar(0.0)
    dtype = jnp.result_type(*[jnp.asarray(leaf).dtype for leaf in leaves])
    max_value = jnp.asarray(0.0, dtype=dtype)
    for leaf in leaves:
        leaf = jnp.ravel(jnp.asarray(leaf))
        leaf_norm = jnp.asarray(0.0, dtype=dtype)
        if leaf.size:
            leaf_norm = jnp.max(jnp.abs(leaf)).astype(dtype)
        max_value = jnp.maximum(max_value, leaf_norm)
    return max_value


def _tree_l2_norm(tree):
    return jnp.sqrt(jnp.maximum(_tree_vdot_real(tree, tree), _device_scalar(0.0)))


def _tree_all_finite(tree):
    leaves = jax.tree.leaves(tree)
    finite = jnp.asarray(True)
    for leaf in leaves:
        finite = finite & jnp.all(jnp.isfinite(jnp.asarray(leaf)))
    return finite


def _tree_select(pred, candidate, current):
    return jax.tree.map(
        lambda cand, curr: lax.select(pred, jnp.asarray(cand), jnp.asarray(curr)),
        candidate,
        current,
    )


def _flattened_residual_output(residual_fn):
    def wrapped(x):
        return jnp.ravel(jnp.asarray(residual_fn(x)))

    return wrapped


def _normalize_solver_args(args):
    if args is None:
        return ()
    if isinstance(args, tuple):
        return args
    return (args,)


def _wrap_value_and_grad_fun(fun, x0, *, host_inputs):
    expected_tree = jax.tree.structure(x0)

    def wrapped(x):
        call_x = _hostify_optimizer_tree(x) if host_inputs else x
        value, grad = fun(call_x)
        if jax.tree.structure(grad) != expected_tree:
            raise ValueError(
                "Explicit value-and-gradient objectives must return a gradient "
                "with the same pytree structure as x0."
            )
        return jnp.asarray(value), jax.tree.map(jnp.asarray, grad)

    return wrapped


def _prepare_adam_eval_fn(fun, x0, *, value_and_grad, host_inputs):
    if value_and_grad:
        return _wrap_value_and_grad_fun(fun, x0, host_inputs=host_inputs)
    return _cached_jit_value_and_grad(fun)


def _adam_defaults(dtype):
    return {
        "step_size": _device_scalar(1.0e-2, dtype=dtype),
        "beta1": _device_scalar(0.9, dtype=dtype),
        "beta2": _device_scalar(0.999, dtype=dtype),
        "eps": _device_scalar(1.0e-8, dtype=dtype),
    }


def _adam_hyperparameters(options, *, dtype):
    defaults = _adam_defaults(dtype)
    options = options or {}
    return {
        "step_size": _device_scalar(
            options.get("step_size", defaults["step_size"]), dtype=dtype
        ),
        "beta1": _device_scalar(options.get("beta1", defaults["beta1"]), dtype=dtype),
        "beta2": _device_scalar(options.get("beta2", defaults["beta2"]), dtype=dtype),
        "eps": _device_scalar(options.get("eps", defaults["eps"]), dtype=dtype),
    }


def _adam_result_message(status, success):
    if _host_bool(success):
        return "converged"
    if int(_host_scalar(status, dtype=np.int64)) == 2:
        return "non-finite objective, gradient, or step encountered"
    return "maximum iterations reached"


def _adam_result_to_optimize_result(result):
    nit = int(_host_scalar(result["nit"], dtype=np.int64))
    status = int(_host_scalar(result["status"], dtype=np.int64))
    success = _host_bool(result["success"])
    return OptimizeResult(
        x=result["x"],
        fun=result["fun"],
        jac=result["grad"],
        nit=nit,
        nfev=nit + 1,
        njev=nit + 1,
        status=status,
        success=success,
        mean=result["mean"],
        variance=result["variance"],
        message=_adam_result_message(status, success),
    )


def _adam_iteration(eval_fn, state, *, hyperparameters, tol):
    step_number = state["nit"] + 1
    beta1 = hyperparameters["beta1"]
    beta2 = hyperparameters["beta2"]
    one_minus_beta1 = jnp.asarray(1.0, dtype=beta1.dtype) - beta1
    one_minus_beta2 = jnp.asarray(1.0, dtype=beta2.dtype) - beta2
    mean = _tree_add(
        _tree_scalar_mul(state["mean"], beta1),
        _tree_scalar_mul(state["grad"], one_minus_beta1),
    )
    variance = _tree_add(
        _tree_scalar_mul(state["variance"], beta2),
        _tree_scalar_mul(_tree_square(state["grad"]), one_minus_beta2),
    )
    step_exponent = jnp.asarray(step_number, dtype=beta1.dtype)
    mean_hat = _tree_bias_correction(mean, 1.0 - jnp.power(beta1, step_exponent))
    variance_hat = _tree_bias_correction(
        variance,
        1.0 - jnp.power(beta2, step_exponent),
    )
    step = _tree_adam_step(
        mean_hat,
        variance_hat,
        step_size=hyperparameters["step_size"],
        eps=hyperparameters["eps"],
    )
    x_candidate = _tree_sub(state["x"], step)
    fun_candidate, grad_candidate = eval_fn(x_candidate)
    grad_norm_inf = _tree_inf_norm(grad_candidate)
    finite_candidate = (
        _tree_all_finite(x_candidate)
        & jnp.isfinite(fun_candidate)
        & _tree_all_finite(grad_candidate)
        & _tree_all_finite(step)
    )
    return {
        "x": _tree_select(finite_candidate, x_candidate, state["x"]),
        "fun": lax.select(finite_candidate, fun_candidate, state["fun"]),
        "grad": _tree_select(finite_candidate, grad_candidate, state["grad"]),
        "grad_norm_inf": lax.select(
            finite_candidate,
            grad_norm_inf,
            state["grad_norm_inf"],
        ),
        "mean": _tree_select(finite_candidate, mean, state["mean"]),
        "variance": _tree_select(finite_candidate, variance, state["variance"]),
        "nit": step_number,
        "status": lax.select(
            finite_candidate,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(2, dtype=jnp.int32),
        ),
        "success": finite_candidate & (grad_norm_inf <= tol),
    }


def adam_optimize(
    fun,
    x0,
    *,
    value_and_grad=False,
    maxiter=1500,
    tol=1e-10,
    options=None,
    callback=None,
    progress_callback=None,
):
    """Host-driven Adam optimizer for noisy/stochastic scalar objectives."""
    x = jax.tree.map(jnp.asarray, x0)
    x_dtype = _require_tree_first_leaf(
        x,
        detail="Adam initial state must contain at least one leaf.",
    ).dtype
    eval_fn = _prepare_adam_eval_fn(
        fun, x, value_and_grad=value_and_grad, host_inputs=True
    )
    hyperparameters = _adam_hyperparameters(options, dtype=x_dtype)
    fun_value, grad = eval_fn(x)
    grad_norm_inf = _tree_inf_norm(grad)
    mean = _tree_zeros_like(x)
    variance = _tree_zeros_like(x)
    nit = 0
    status = 1
    success = bool(grad_norm_inf <= tol)

    while nit < maxiter and not success:
        state = _adam_iteration(
            eval_fn,
            {
                "x": x,
                "fun": fun_value,
                "grad": grad,
                "grad_norm_inf": grad_norm_inf,
                "mean": mean,
                "variance": variance,
                "nit": jnp.asarray(nit, dtype=jnp.int32),
            },
            hyperparameters=hyperparameters,
            tol=_device_scalar(tol, dtype=x_dtype),
        )
        nit = int(state["nit"])
        status = int(state["status"])
        x = state["x"]
        fun_value = state["fun"]
        grad = state["grad"]
        grad_norm_inf = state["grad_norm_inf"]
        mean = state["mean"]
        variance = state["variance"]
        if callback is not None:
            callback(_hostify_optimizer_tree(x))
        if progress_callback is not None:
            progress_callback(nit, float(fun_value), float(grad_norm_inf))
        success = bool(state["success"])
        if status == 2:
            break

    return {
        "x": x,
        "fun": fun_value,
        "grad": grad,
        "mean": mean,
        "variance": variance,
        "nit": nit,
        "status": status,
        "success": success,
    }


def adam_optimize_traceable(
    fun,
    x0,
    *,
    value_and_grad=False,
    maxiter=1500,
    tol=1e-10,
    options=None,
    callback=None,
    progress_callback=None,
):
    """Trace-safe Adam optimizer for noisy/stochastic scalar objectives."""
    x = jax.tree.map(jnp.asarray, x0)
    x_dtype = _require_tree_first_leaf(
        x,
        detail="Adam initial state must contain at least one leaf.",
    ).dtype
    eval_fn = _prepare_adam_eval_fn(
        fun, x, value_and_grad=value_and_grad, host_inputs=False
    )
    hyperparameters = _adam_hyperparameters(options, dtype=x_dtype)
    tol_value = _device_scalar(tol, dtype=x_dtype)

    def run_solver(x_init):
        fun0, grad0 = eval_fn(x_init)
        state0 = {
            "x": x_init,
            "fun": fun0,
            "grad": grad0,
            "grad_norm_inf": _tree_inf_norm(grad0),
            "mean": _tree_zeros_like(x_init),
            "variance": _tree_zeros_like(x_init),
            "nit": jnp.asarray(0, dtype=jnp.int32),
            "status": jnp.asarray(1, dtype=jnp.int32),
            "success": _tree_inf_norm(grad0) <= tol_value,
        }

        def cond_fun(state):
            return (
                (state["nit"] < maxiter) & (~state["success"]) & (state["status"] != 2)
            )

        def body_fun(state):
            next_state = _adam_iteration(
                eval_fn,
                state,
                hyperparameters=hyperparameters,
                tol=tol_value,
            )
            if callback is not None:
                jax.debug.callback(
                    lambda current_x: callback(_hostify_optimizer_tree(current_x)),
                    next_state["x"],
                    ordered=False,
                )
            if progress_callback is not None:
                jax.debug.callback(
                    progress_callback,
                    next_state["nit"],
                    next_state["fun"],
                    next_state["grad_norm_inf"],
                    ordered=False,
                )
            return next_state

        return lax.while_loop(cond_fun, body_fun, state0)

    run_solver.__name__ = "adam_traceable_run_solver"
    return jax.jit(run_solver)(x)


def _least_squares_gradient_state(flat_residual_fn, x):
    residual, pullback = jax.vjp(flat_residual_fn, x)
    grad = pullback(residual)[0]
    cost = _least_squares_cost(residual)
    grad_norm_inf = _tree_inf_norm(grad)
    return residual, cost, grad, grad_norm_inf, pullback


def _make_traceable_levenberg_marquardt_runner(
    residual_fn,
    maxiter,
    tol,
    ftol,
    xtol,
    gtol,
    materialize_dense_linearization,
    max_dense_linearization_bytes,
    callback_enabled,
    progress_callback_enabled,
):
    cache_key = (
        int(maxiter),
        float(tol),
        float(ftol),
        float(xtol),
        None if gtol is None else float(gtol),
        bool(materialize_dense_linearization),
        max_dense_linearization_bytes,
        bool(callback_enabled),
        bool(progress_callback_enabled),
    )
    return _cached_traceable_runner(
        _TRACEABLE_LM_RUNNER_CACHE,
        residual_fn,
        cache_key,
        lambda residual_fn_ref: _build_traceable_levenberg_marquardt_runner(
            residual_fn_ref,
            int(maxiter),
            float(tol),
            float(ftol),
            float(xtol),
            None if gtol is None else float(gtol),
            bool(materialize_dense_linearization),
            max_dense_linearization_bytes,
            bool(callback_enabled),
            bool(progress_callback_enabled),
        ),
    )


def _build_traceable_levenberg_marquardt_runner(
    residual_fn_ref,
    maxiter,
    tol,
    ftol,
    xtol,
    gtol,
    materialize_dense_linearization,
    max_dense_linearization_bytes,
    callback_enabled,
    progress_callback_enabled,
):
    def run_solver(x_init, fn_args, callback_token, progress_callback_token):
        residual_fn = _lookup_traceable_runner_callable(residual_fn_ref, "LM residual")

        def residual_eval(x):
            return jnp.ravel(jnp.asarray(residual_fn(x, *fn_args)))

        x_dtype = _require_tree_first_leaf(
            x_init,
            detail="Least-squares initial state must contain at least one leaf.",
        ).dtype
        tol_value = _device_scalar(tol, dtype=x_dtype)
        gradient_tol = _lm_gradient_tol(tol, gtol, dtype=x_dtype)
        residual0, cost0, grad0, grad_norm_inf0, _ = _least_squares_gradient_state(
            residual_eval,
            x_init,
        )
        state0 = {
            "x": x_init,
            "residual": residual0,
            "cost": cost0,
            "grad": grad0,
            "grad_norm_inf": grad_norm_inf0,
            "damping": _lm_defaults(x_dtype)["initial_damping"],
            "delta": _lm_initial_delta(x_init, dtype=x_dtype),
            "nit": jnp.asarray(0, dtype=jnp.int32),
            "status": jnp.asarray(0, dtype=jnp.int32),
            "info": jnp.asarray(0, dtype=jnp.int32),
            "accepted": jnp.asarray(False),
            "success": grad_norm_inf0 <= gradient_tol,
        }

        def cond_fun(state):
            return (
                (state["nit"] < maxiter)
                & (~state["success"])
                & (state["status"] != 2)
                & (state["info"] == 0)
            )

        def body_fun(state):
            next_state = _lm_iteration(
                residual_eval,
                state,
                tol=tol_value,
                gradient_tol=gradient_tol,
                ftol=_device_scalar(ftol, dtype=x_dtype),
                xtol=_device_scalar(xtol, dtype=x_dtype),
                maxiter=maxiter,
            )
            if callback_enabled:
                lax.cond(
                    next_state["accepted"],
                    lambda _: jax.debug.callback(
                        _invoke_traceable_lm_callback,
                        callback_token,
                        next_state["x"],
                        ordered=False,
                    ),
                    lambda _: None,
                    operand=None,
                )
            if progress_callback_enabled:
                lax.cond(
                    next_state["accepted"],
                    lambda _: jax.debug.callback(
                        _invoke_traceable_progress_callback,
                        progress_callback_token,
                        next_state["nit"],
                        next_state["cost"],
                        next_state["grad_norm_inf"],
                        ordered=False,
                    ),
                    lambda _: None,
                    operand=None,
                )
            return next_state

        state = lax.while_loop(cond_fun, body_fun, state0)
        residual_final = residual_eval(state["x"])
        linearization_rows = int(np.asarray(jnp.asarray(residual_final).size))
        linearization_cols = sum(
            int(np.asarray(jnp.asarray(leaf).size))
            for leaf in jax.tree.leaves(state["x"])
        )
        materialize_linearization = bool(materialize_dense_linearization)
        dense_report = _least_squares_dense_linearization_report(
            linearization_rows,
            linearization_cols,
            x_dtype,
            max_dense_linearization_bytes,
        )
        dense_report["failure_category"] = None
        dense_report["failure_stage"] = None
        dense_report["message"] = None
        residual_jacobian = None
        hessian = None
        if materialize_linearization:
            materialize_linearization, dense_report = (
                _least_squares_dense_linearization_policy(
                    linearization_rows,
                    linearization_cols,
                    x_dtype,
                    max_dense_linearization_bytes,
                )
            )
            if materialize_linearization:
                residual_final, residual_jacobian, _flat_grad, hessian = (
                    _materialize_dense_least_squares_linearization(
                        residual_eval,
                        state["x"],
                    )
                )
        return {
            "x": state["x"],
            "residual": residual_final,
            "residual_jacobian": residual_jacobian,
            "fun": state["cost"],
            "grad": state["grad"],
            "hessian": hessian,
            "damping": state["damping"],
            "nit": state["nit"],
            "status": state["status"],
            "info": state["info"],
            "success": state["success"],
            "dense_linearization_materialized": materialize_linearization,
            **dense_report,
        }

    run_solver.__name__ = "traceable_levenberg_marquardt_run_solver"
    if not callback_enabled and not progress_callback_enabled:

        def run_solver_without_callbacks(x_init, fn_args):
            return run_solver(x_init, fn_args, 0, 0)

        run_solver_without_callbacks.__name__ = run_solver.__name__
        return jax.jit(run_solver_without_callbacks)
    return jax.jit(run_solver, static_argnums=(2, 3))


def _least_squares_matvec(flat_residual_fn, x, pullback, tangent):
    jvp_residual = jax.jvp(flat_residual_fn, (x,), (tangent,))[1]
    return pullback(jvp_residual)[0]


def _gmres_solve_least_squares_system(
    flat_residual_fn,
    x,
    grad,
    pullback,
    *,
    damping,
    tol,
):
    grad_leaves = jax.tree.leaves(grad)
    first_grad_leaf = _require_tree_first_leaf(
        grad,
        detail="Least-squares gradients must contain at least one leaf.",
    )
    dtype = first_grad_leaf.dtype
    n = sum(int(np.asarray(jnp.asarray(leaf).size)) for leaf in grad_leaves)
    restart = max(5, min(n, 50))
    maxiter = max(10, min(4 * n, 200))
    damping_value = jnp.asarray(damping, dtype=dtype)

    def matvec(v):
        jt_j_v = _least_squares_matvec(flat_residual_fn, x, pullback, v)
        return jax.tree.map(
            lambda jt_j_leaf, v_leaf: jt_j_leaf + damping_value * v_leaf,
            jt_j_v,
            v,
        )

    with jax.transfer_guard_host_to_device("allow"):
        step, _ = gmres(
            matvec,
            grad,
            tol=tol,
            atol=0.0,
            restart=restart,
            maxiter=maxiter,
            solve_method="incremental",
        )
    residual = jax.tree.map(
        lambda grad_leaf, matvec_leaf: grad_leaf - matvec_leaf,
        grad,
        matvec(step),
    )
    return step, residual, matvec


def _materialize_dense_least_squares_linearization(flat_residual_fn, x):
    flat_x, unravel = ravel_pytree(x)
    flat_x = jnp.asarray(flat_x)
    jvp_fn = _jacobian_vector_product_fn(lambda flat: flat_residual_fn(unravel(flat)))
    residual = flat_residual_fn(x)
    jacobian = _materialize_dense_jacobian(jvp_fn, flat_x)
    gradient, hessian = _least_squares_linearization_from_jacobian(
        residual,
        jacobian,
    )
    return residual, jacobian, gradient, hessian


def _clip_lm_damping(damping, *, dtype):
    minimum = _device_scalar(1.0e-12, dtype=dtype)
    maximum = _device_scalar(1.0e12, dtype=dtype)
    return jnp.clip(jnp.asarray(damping, dtype=dtype), minimum, maximum)


def _lm_defaults(dtype):
    return {
        "initial_damping": _device_scalar(1.0e-3, dtype=dtype),
        "initial_delta_factor": _device_scalar(100.0, dtype=dtype),
        "accept_threshold": _device_scalar(1.0e-4, dtype=dtype),
        "increase_factor": _device_scalar(2.0, dtype=dtype),
        "decrease_factor": _device_scalar(0.5, dtype=dtype),
        "minimum_delta_update": _device_scalar(0.1, dtype=dtype),
        "ratio_low": _device_scalar(0.25, dtype=dtype),
        "ratio_high": _device_scalar(0.75, dtype=dtype),
        "predicted_floor": _device_scalar(1.0e-18, dtype=dtype),
    }


def _lm_initial_delta(x, *, dtype):
    x_norm = _tree_l2_norm(x)
    return _lm_defaults(dtype)["initial_delta_factor"] * jnp.maximum(
        x_norm,
        _device_scalar(1.0, dtype=dtype),
    )


def _lm_gradient_tol(tol, gtol, *, dtype):
    if gtol is None:
        return _optimizer_scalar(tol, dtype=dtype)
    return _optimizer_scalar(gtol, dtype=dtype)


def _optimistix_lm_nondefault_tuning_options(ftol, xtol, gtol):
    unsupported = []
    if ftol is None or float(ftol) != _OPTIMISTIX_LM_DEFAULT_FTOL:
        unsupported.append("ftol")
    if xtol is None or float(xtol) != _OPTIMISTIX_LM_DEFAULT_XTOL:
        unsupported.append("xtol")
    if gtol is not None:
        unsupported.append("gtol")
    return tuple(unsupported)


def _require_optimistix_lm_contract_options(
    *,
    ftol,
    xtol,
    gtol,
    callback,
    progress_callback,
):
    if callback is not None or progress_callback is not None:
        raise ValueError(
            "optimistix-lm-ondevice does not support solver callbacks. "
            "Use method='lm-ondevice' for callback-instrumented LM runs."
        )
    unsupported = _optimistix_lm_nondefault_tuning_options(ftol, xtol, gtol)
    if unsupported:
        unsupported_options = ", ".join(unsupported)
        raise ValueError(
            "optimistix-lm-ondevice uses a single tol value for Optimistix "
            "and Lineax convergence. Non-default LM tuning option(s) are not "
            f"supported: {unsupported_options}."
        )


def _matrix_free_lm_info(
    *,
    actual_reduction,
    predicted_reduction,
    cost,
    delta,
    x_norm,
    nit,
    maxiter,
    ftol,
    xtol,
    epsmch,
):
    """Return MINPACK-style info for the matrix-free-computable LM subset."""
    one = jnp.asarray(1.0, dtype=cost.dtype)
    half = jnp.asarray(0.5, dtype=cost.dtype)
    cost_floor = jnp.maximum(cost, jnp.finfo(cost.dtype).tiny)
    nonnegative_reduction = actual_reduction >= jnp.asarray(0.0, dtype=cost.dtype)
    relative_actual = actual_reduction / cost_floor
    relative_predicted = predicted_reduction / cost_floor
    ratio = actual_reduction / jnp.maximum(predicted_reduction, cost_floor * epsmch)
    ftol_met = (
        nonnegative_reduction
        & (relative_actual <= ftol)
        & (relative_predicted <= ftol)
        & (half * ratio <= one)
    )
    xtol_met = delta <= xtol * x_norm

    info = jnp.asarray(0, dtype=jnp.int32)
    info = jnp.where(ftol_met, jnp.asarray(1, dtype=jnp.int32), info)
    info = jnp.where(xtol_met, jnp.asarray(2, dtype=jnp.int32), info)
    info = jnp.where(
        ftol_met & xtol_met,
        jnp.asarray(3, dtype=jnp.int32),
        info,
    )
    info = jnp.where(
        (info == 0) & (nit >= maxiter),
        jnp.asarray(5, dtype=jnp.int32),
        info,
    )
    info = jnp.where(
        (info == 0)
        & nonnegative_reduction
        & (relative_actual <= epsmch)
        & (relative_predicted <= epsmch)
        & (half * ratio <= one),
        jnp.asarray(6, dtype=jnp.int32),
        info,
    )
    info = jnp.where(
        (info == 0) & (delta <= epsmch * x_norm),
        jnp.asarray(7, dtype=jnp.int32),
        info,
    )
    return info


def _lm_delta_after_step(delta, step_norm, ratio, actual_reduction, *, defaults):
    low_ratio_base = jnp.minimum(
        delta,
        step_norm / defaults["minimum_delta_update"],
    )
    low_ratio_scale = jnp.where(
        actual_reduction >= jnp.asarray(0.0, dtype=delta.dtype),
        defaults["decrease_factor"],
        defaults["minimum_delta_update"],
    )
    low_ratio_delta = low_ratio_scale * low_ratio_base
    updated_delta = jnp.where(
        ratio >= defaults["ratio_high"],
        step_norm / defaults["decrease_factor"],
        delta,
    )
    return jnp.where(
        ratio <= defaults["ratio_low"],
        low_ratio_delta,
        updated_delta,
    )


def _lm_iteration(flat_residual_fn, state, *, tol, gradient_tol, ftol, xtol, maxiter):
    state_dtype = _require_tree_first_leaf(
        state["x"],
        detail="Least-squares state x must contain at least one leaf.",
    ).dtype
    defaults = _lm_defaults(state_dtype)
    damping = _clip_lm_damping(state["damping"], dtype=state_dtype)
    linear_tol = jnp.minimum(
        _device_scalar(1.0e-10, dtype=state["cost"].dtype),
        jnp.maximum(
            _optimizer_scalar(tol, dtype=state["cost"].dtype)
            * _device_scalar(0.1, dtype=state["cost"].dtype),
            _device_scalar(1.0e-14, dtype=state["cost"].dtype),
        ),
    )
    _, current_pullback = jax.vjp(flat_residual_fn, state["x"])
    step, linear_residual, _ = _gmres_solve_least_squares_system(
        flat_residual_fn,
        state["x"],
        state["grad"],
        current_pullback,
        damping=damping,
        tol=linear_tol,
    )
    x_candidate = jax.tree.map(
        lambda x_leaf, step_leaf: x_leaf - step_leaf,
        state["x"],
        step,
    )
    residual_candidate, cost_candidate, grad_candidate, grad_norm_candidate, _ = (
        _least_squares_gradient_state(flat_residual_fn, x_candidate)
    )

    predicted_reduction = _device_scalar(
        0.5,
        dtype=state["cost"].dtype,
    ) * (
        jnp.asarray(damping, dtype=state["cost"].dtype) * _tree_vdot_real(step, step)
        + _tree_vdot_real(step, state["grad"])
    )
    actual_reduction = state["cost"] - cost_candidate
    ratio = actual_reduction / jnp.maximum(
        predicted_reduction,
        defaults["predicted_floor"],
    )
    finite_candidate = (
        _tree_all_finite(x_candidate)
        & jnp.all(jnp.isfinite(residual_candidate))
        & jnp.isfinite(cost_candidate)
        & _tree_all_finite(grad_candidate)
        & _tree_all_finite(linear_residual)
    )
    accepted = finite_candidate & (ratio >= defaults["accept_threshold"])
    step_norm = _tree_l2_norm(step)
    delta_after_step = _lm_delta_after_step(
        state["delta"],
        step_norm,
        jnp.asarray(ratio, dtype=state["delta"].dtype),
        jnp.asarray(actual_reduction, dtype=state["delta"].dtype),
        defaults=defaults,
    )

    damping_after_accept = lax.cond(
        ratio > defaults["ratio_high"],
        lambda _: damping * defaults["decrease_factor"],
        lambda _: lax.cond(
            ratio < defaults["ratio_low"],
            lambda __: damping * defaults["increase_factor"],
            lambda __: damping,
            operand=None,
        ),
        operand=None,
    )
    next_damping = lax.cond(
        accepted,
        lambda _: _clip_lm_damping(damping_after_accept, dtype=state_dtype),
        lambda _: _clip_lm_damping(
            damping * defaults["increase_factor"],
            dtype=state_dtype,
        ),
        operand=None,
    )
    x_next = _tree_select(accepted, x_candidate, state["x"])
    residual_next = lax.select(accepted, residual_candidate, state["residual"])
    cost_next = lax.select(accepted, cost_candidate, state["cost"])
    grad_next = _tree_select(accepted, grad_candidate, state["grad"])
    grad_norm_next = lax.select(
        accepted,
        grad_norm_candidate,
        state["grad_norm_inf"],
    )
    x_norm = _tree_l2_norm(x_next)
    next_nit = state["nit"] + 1
    info_candidate = _matrix_free_lm_info(
        actual_reduction=actual_reduction,
        predicted_reduction=predicted_reduction,
        cost=state["cost"],
        delta=delta_after_step,
        x_norm=x_norm,
        nit=next_nit,
        maxiter=jnp.asarray(maxiter, dtype=jnp.int32),
        ftol=jnp.asarray(ftol, dtype=state["cost"].dtype),
        xtol=jnp.asarray(xtol, dtype=state["cost"].dtype),
        epsmch=jnp.asarray(
            jnp.finfo(state["cost"].dtype).eps, dtype=state["cost"].dtype
        ),
    )
    info_next = lax.select(
        finite_candidate,
        info_candidate,
        jnp.asarray(0, dtype=jnp.int32),
    )
    legacy_success = grad_norm_next <= gradient_tol
    info_success = (info_next == 1) | (info_next == 2) | (info_next == 3)

    return {
        "x": x_next,
        "residual": residual_next,
        "cost": cost_next,
        "grad": grad_next,
        "grad_norm_inf": grad_norm_next,
        "damping": next_damping,
        "delta": delta_after_step,
        "nit": next_nit,
        "status": lax.select(
            finite_candidate,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(2, dtype=jnp.int32),
        ),
        "info": info_next,
        "accepted": accepted,
        "success": finite_candidate & (legacy_success | info_success),
    }


@jax.jit
def _dense_lm_propose_step(hessian, gradient, damping):
    hessian = jnp.asarray(hessian)
    gradient = jnp.asarray(gradient)
    dtype = hessian.dtype
    cols = hessian.shape[1]
    damped_hessian = hessian.at[jnp.diag_indices(cols)].add(
        jnp.asarray(damping, dtype=dtype)
    )
    return jnp.linalg.solve(damped_hessian, gradient)


@jax.jit
def _dense_lm_accept_state(
    x,
    residual,
    jacobian,
    gradient,
    hessian,
    cost,
    grad_norm_inf,
    candidate_residual,
    candidate_jacobian,
    candidate_gradient,
    candidate_hessian,
    candidate_cost,
    candidate_grad_norm_inf,
    step,
    damping,
    delta,
    nit,
    status,
    info,
    gradient_tol,
    ftol,
    xtol,
    maxiter,
):
    del status, info
    dtype = jnp.asarray(cost).dtype
    defaults = _lm_defaults(dtype)
    damping = _clip_lm_damping(damping, dtype=dtype)
    x_candidate = jnp.asarray(x) - jnp.asarray(step)
    predicted_reduction = _device_scalar(0.5, dtype=dtype) * (
        jnp.asarray(damping, dtype=dtype) * jnp.vdot(step, step).real.astype(dtype)
        + jnp.vdot(step, gradient).real.astype(dtype)
    )
    actual_reduction = jnp.asarray(cost) - jnp.asarray(candidate_cost)
    ratio = actual_reduction / jnp.maximum(
        predicted_reduction,
        defaults["predicted_floor"],
    )
    finite_candidate = (
        jnp.all(jnp.isfinite(x_candidate))
        & jnp.all(jnp.isfinite(step))
        & jnp.all(jnp.isfinite(candidate_residual))
        & jnp.all(jnp.isfinite(candidate_jacobian))
        & jnp.all(jnp.isfinite(candidate_gradient))
        & jnp.all(jnp.isfinite(candidate_hessian))
        & jnp.isfinite(candidate_cost)
        & jnp.isfinite(predicted_reduction)
    )
    accepted = finite_candidate & (ratio >= defaults["accept_threshold"])
    step_norm = jnp.linalg.norm(step)
    delta_after_step = _lm_delta_after_step(
        delta,
        step_norm,
        jnp.asarray(ratio, dtype=jnp.asarray(delta).dtype),
        jnp.asarray(actual_reduction, dtype=jnp.asarray(delta).dtype),
        defaults=defaults,
    )
    damping_after_accept = lax.cond(
        ratio > defaults["ratio_high"],
        lambda _: damping * defaults["decrease_factor"],
        lambda _: lax.cond(
            ratio < defaults["ratio_low"],
            lambda __: damping * defaults["increase_factor"],
            lambda __: damping,
            operand=None,
        ),
        operand=None,
    )
    next_damping = lax.cond(
        accepted,
        lambda _: _clip_lm_damping(damping_after_accept, dtype=dtype),
        lambda _: _clip_lm_damping(
            damping * defaults["increase_factor"],
            dtype=dtype,
        ),
        operand=None,
    )
    x_next = lax.select(accepted, x_candidate, jnp.asarray(x))
    residual_next = lax.select(accepted, candidate_residual, residual)
    jacobian_next = lax.select(accepted, candidate_jacobian, jacobian)
    gradient_next = lax.select(accepted, candidate_gradient, gradient)
    hessian_next = lax.select(accepted, candidate_hessian, hessian)
    cost_next = lax.select(accepted, candidate_cost, cost)
    grad_norm_next = lax.select(
        accepted,
        candidate_grad_norm_inf,
        grad_norm_inf,
    )
    x_norm = jnp.linalg.norm(x_next)
    next_nit = jnp.asarray(nit, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    info_candidate = _matrix_free_lm_info(
        actual_reduction=actual_reduction,
        predicted_reduction=predicted_reduction,
        cost=cost,
        delta=delta_after_step,
        x_norm=x_norm,
        nit=next_nit,
        maxiter=jnp.asarray(maxiter, dtype=jnp.int32),
        ftol=jnp.asarray(ftol, dtype=dtype),
        xtol=jnp.asarray(xtol, dtype=dtype),
        epsmch=jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype),
    )
    info_next = lax.select(
        finite_candidate,
        info_candidate,
        jnp.asarray(0, dtype=jnp.int32),
    )
    legacy_success = grad_norm_next <= jnp.asarray(gradient_tol, dtype=dtype)
    info_success = (info_next == 1) | (info_next == 2) | (info_next == 3)
    status_next = lax.select(
        finite_candidate,
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(2, dtype=jnp.int32),
    )
    return {
        "x": x_next,
        "residual": residual_next,
        "residual_jacobian": jacobian_next,
        "grad": gradient_next,
        "hessian": hessian_next,
        "fun": cost_next,
        "grad_norm_inf": grad_norm_next,
        "damping": next_damping,
        "delta": delta_after_step,
        "nit": next_nit,
        "status": status_next,
        "info": info_next,
        "accepted": accepted,
        "success": finite_candidate & (legacy_success | info_success),
    }


def _least_squares_result_message(status, success, info=0):
    info_value = int(_host_scalar(info, dtype=np.int64))
    if info_value == 1:
        return "converged: ftol termination condition is satisfied"
    if info_value == 2:
        return "converged: xtol termination condition is satisfied"
    if info_value == 3:
        return "converged: both ftol and xtol termination conditions are satisfied"
    if info_value == 4:
        return "converged: gtol termination condition is satisfied"
    if info_value == 5:
        return "maximum iterations reached"
    if info_value == 6:
        return "ftol is too small; no further cost reduction is possible"
    if info_value == 7:
        return "xtol is too small; no further step reduction is possible"
    if info_value == 8:
        return "gtol is too small; residual is orthogonal to Jacobian columns"
    if _host_bool(success):
        return "converged"
    if int(_host_scalar(status, dtype=np.int64)) == 2:
        return "non-finite residual, gradient, or linear solve encountered"
    return "maximum iterations reached"


def _normalize_dense_lm_state(state):
    return {
        "residual": jnp.ravel(jnp.asarray(state["residual"])),
        "residual_jacobian": jnp.asarray(state["residual_jacobian"]),
        "grad": jnp.asarray(state["grad"]),
        "hessian": jnp.asarray(state["hessian"]),
        "fun": jnp.asarray(state["fun"]),
        "grad_norm_inf": jnp.asarray(state["grad_norm_inf"]),
    }


def _dense_jacobian_basis_block(start, cols, chunk_size, dtype):
    valid = min(int(chunk_size), int(cols) - int(start))
    basis = np.zeros((int(chunk_size), int(cols)), dtype=np.dtype(dtype))
    if valid > 0:
        rows = np.arange(valid)
        basis[rows, int(start) + rows] = 1.0
    return jnp.asarray(basis)


def _materialize_dense_jacobian_blocks(
    jacobian_block_fn,
    x,
    args,
    *,
    chunk_size,
):
    x_array = jnp.ravel(jnp.asarray(x))
    cols = int(x_array.size)
    blocks = []
    for start in range(0, cols, int(chunk_size)):
        valid = min(int(chunk_size), cols - start)
        basis = _dense_jacobian_basis_block(
            start,
            cols,
            int(chunk_size),
            x_array.dtype,
        )
        block = jnp.asarray(jacobian_block_fn(x, basis, *args))
        blocks.append(block[:, :valid])
    return jnp.concatenate(blocks, axis=1)


def _levenberg_marquardt_dense_loop(
    evaluate_state,
    x0,
    *,
    maxiter=1500,
    tol=1e-10,
    ftol=1e-8,
    xtol=1e-8,
    gtol=None,
    materialize_dense_linearization=True,
    max_dense_linearization_bytes=None,
    callback=None,
    progress_callback=None,
):
    x = jnp.asarray(x0)
    state = _normalize_dense_lm_state(evaluate_state(x))
    x_dtype = x.dtype
    gradient_tol = _lm_gradient_tol(tol, gtol, dtype=x_dtype)
    damping = _lm_defaults(x_dtype)["initial_damping"]
    delta = _lm_initial_delta(x, dtype=x_dtype)
    status = 1
    info = 0
    nit = 0
    success = bool(state["grad_norm_inf"] <= gradient_tol)

    while nit < maxiter and not success and info == 0:
        step = _dense_lm_propose_step(
            state["hessian"],
            state["grad"],
            damping,
        )
        candidate_x = x - step
        candidate = _normalize_dense_lm_state(evaluate_state(candidate_x))
        step_state = _dense_lm_accept_state(
            x,
            state["residual"],
            state["residual_jacobian"],
            state["grad"],
            state["hessian"],
            state["fun"],
            state["grad_norm_inf"],
            candidate["residual"],
            candidate["residual_jacobian"],
            candidate["grad"],
            candidate["hessian"],
            candidate["fun"],
            candidate["grad_norm_inf"],
            step,
            damping,
            delta,
            jnp.asarray(nit, dtype=jnp.int32),
            jnp.asarray(status, dtype=jnp.int32),
            jnp.asarray(info, dtype=jnp.int32),
            gradient_tol,
            _optimizer_scalar(ftol, dtype=x_dtype),
            _optimizer_scalar(xtol, dtype=x_dtype),
            jnp.asarray(maxiter, dtype=jnp.int32),
        )
        nit = int(_host_scalar(step_state["nit"], dtype=np.int64))
        status = int(_host_scalar(step_state["status"], dtype=np.int64))
        info = int(_host_scalar(step_state["info"], dtype=np.int64))
        damping = step_state["damping"]
        delta = step_state["delta"]
        if bool(_host_bool(step_state["accepted"])):
            x = step_state["x"]
            state = _normalize_dense_lm_state(step_state)
            if callback is not None:
                callback(_hostify_optimizer_tree(x))
            if progress_callback is not None:
                progress_callback(
                    nit,
                    float(_host_scalar(state["fun"])),
                    float(_host_scalar(state["grad_norm_inf"])),
                )
        success = bool(_host_bool(step_state["success"]))
        if status == 2:
            break

    linearization_rows = int(state["residual"].size)
    linearization_cols = int(x.size)
    dense_report = _least_squares_dense_linearization_report(
        linearization_rows,
        linearization_cols,
        x_dtype,
        max_dense_linearization_bytes,
    )
    dense_report["failure_category"] = None
    dense_report["failure_stage"] = None
    dense_report["message"] = None
    dense_linearization_materialized = bool(materialize_dense_linearization)
    if dense_linearization_materialized:
        dense_linearization_materialized, dense_report = (
            _least_squares_dense_linearization_policy(
                linearization_rows,
                linearization_cols,
                x_dtype,
                max_dense_linearization_bytes,
            )
        )
    residual_jacobian = (
        state["residual_jacobian"] if dense_linearization_materialized else None
    )
    hessian = state["hessian"] if dense_linearization_materialized else None
    return {
        "x": x,
        "residual": state["residual"],
        "residual_jacobian": residual_jacobian,
        "fun": state["fun"],
        "grad": state["grad"],
        "hessian": hessian,
        "damping": damping,
        "nit": nit,
        "status": status,
        "info": info,
        "success": success,
        "dense_linearization_materialized": dense_linearization_materialized,
        "dense_linearization_kind": (
            "in_loop" if dense_linearization_materialized else None
        ),
        **dense_report,
    }


def levenberg_marquardt_dense_state(
    state_fn,
    x0,
    *,
    maxiter=1500,
    tol=1e-10,
    ftol=1e-8,
    xtol=1e-8,
    gtol=None,
    materialize_dense_linearization=True,
    max_dense_linearization_bytes=None,
    callback=None,
    progress_callback=None,
    args=(),
):
    """Host-driven LM over caller-owned residual/Jacobian state kernels."""
    normalized_args = _normalize_solver_args(args)

    def evaluate_state(x):
        return state_fn(x, *normalized_args)

    return _levenberg_marquardt_dense_loop(
        evaluate_state,
        x0,
        maxiter=maxiter,
        tol=tol,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        materialize_dense_linearization=materialize_dense_linearization,
        max_dense_linearization_bytes=max_dense_linearization_bytes,
        callback=callback,
        progress_callback=progress_callback,
    )


def levenberg_marquardt_block_jacobian(
    residual_fn,
    jacobian_block_fn,
    x0,
    *,
    maxiter=1500,
    tol=1e-10,
    ftol=1e-8,
    xtol=1e-8,
    gtol=None,
    materialize_dense_linearization=True,
    max_dense_linearization_bytes=None,
    callback=None,
    progress_callback=None,
    args=(),
    jacobian_chunk_size=32,
):
    """Host-driven LM with a fixed-shape Jacobian-column-block kernel."""
    normalized_args = _normalize_solver_args(args)

    def evaluate_state(x):
        residual = jnp.ravel(jnp.asarray(residual_fn(x, *normalized_args)))
        jacobian = _materialize_dense_jacobian_blocks(
            jacobian_block_fn,
            x,
            normalized_args,
            chunk_size=jacobian_chunk_size,
        )
        return _dense_lm_state_from_residual_jacobian(residual, jacobian)

    return _levenberg_marquardt_dense_loop(
        evaluate_state,
        x0,
        maxiter=maxiter,
        tol=tol,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        materialize_dense_linearization=materialize_dense_linearization,
        max_dense_linearization_bytes=max_dense_linearization_bytes,
        callback=callback,
        progress_callback=progress_callback,
    )


def levenberg_marquardt(
    residual_fn,
    x0,
    *,
    maxiter=1500,
    tol=1e-10,
    ftol=1e-8,
    xtol=1e-8,
    gtol=None,
    materialize_dense_linearization=True,
    max_dense_linearization_bytes=None,
    callback=None,
    progress_callback=None,
):
    """Host-driven Levenberg-Marquardt solver for least-squares residuals.

    The LM loop is matrix-free: it uses ``jvp``/``vjp`` products inside GMRES
    and only rebuilds the dense residual Jacobian/Hessian once at the final
    iterate so existing Boozer adjoint consumers retain their contract.

    ``ftol`` and ``xtol`` feed the matrix-free MINPACK-style ``info`` subset.
    Explicit ``gtol`` values replace the legacy ``tol`` gradient gate with a
    matrix-free infinity-norm gradient gate. MINPACK's QR-scaled ``gtol``
    ``info`` code still requires pivoted-QR data and is therefore not emitted
    by this matrix-free solver.
    """
    residual_eval = jax.jit(_flattened_residual_output(residual_fn))

    x = jax.tree.map(jnp.asarray, x0)
    residual, cost, grad, grad_norm_inf, _ = _least_squares_gradient_state(
        residual_eval,
        x,
    )
    x_dtype = _require_tree_first_leaf(
        x,
        detail="Least-squares initial state must contain at least one leaf.",
    ).dtype
    gradient_tol = _lm_gradient_tol(tol, gtol, dtype=x_dtype)
    damping = _lm_defaults(x_dtype)["initial_damping"]
    delta = _lm_initial_delta(x, dtype=x_dtype)
    status = 1
    success = bool(grad_norm_inf <= gradient_tol)
    info = 0
    nit = 0

    while nit < maxiter and not success and info == 0:
        step_state = _lm_iteration(
            residual_eval,
            {
                "x": x,
                "residual": residual,
                "cost": cost,
                "grad": grad,
                "grad_norm_inf": grad_norm_inf,
                "damping": damping,
                "delta": delta,
                "nit": jnp.asarray(nit, dtype=jnp.int32),
                "status": jnp.asarray(status, dtype=jnp.int32),
                "info": jnp.asarray(info, dtype=jnp.int32),
                "accepted": jnp.asarray(False),
                "success": jnp.asarray(False),
            },
            tol=_optimizer_scalar(tol, dtype=x_dtype),
            gradient_tol=gradient_tol,
            ftol=_optimizer_scalar(ftol, dtype=x_dtype),
            xtol=_optimizer_scalar(xtol, dtype=x_dtype),
            maxiter=int(maxiter),
        )
        nit = int(step_state["nit"])
        status = int(step_state["status"])
        info = int(step_state["info"])
        damping = step_state["damping"]
        delta = step_state["delta"]
        if bool(step_state["accepted"]):
            x = step_state["x"]
            residual = step_state["residual"]
            cost = step_state["cost"]
            grad = step_state["grad"]
            grad_norm_inf = step_state["grad_norm_inf"]
            if callback is not None:
                callback(_hostify_optimizer_tree(x))
            if progress_callback is not None:
                progress_callback(nit, float(cost), float(grad_norm_inf))
        success = bool(step_state["success"])
        if status == 2:
            break

    residual = residual_eval(x)
    linearization_rows = int(np.asarray(jnp.asarray(residual).size))
    linearization_cols = sum(
        int(np.asarray(jnp.asarray(leaf).size)) for leaf in jax.tree.leaves(x)
    )
    dense_report = _least_squares_dense_linearization_report(
        linearization_rows,
        linearization_cols,
        x_dtype,
        max_dense_linearization_bytes,
    )
    dense_report["failure_category"] = None
    dense_report["failure_stage"] = None
    dense_report["message"] = None
    residual_jacobian = None
    hessian = None
    dense_linearization_materialized = bool(materialize_dense_linearization)
    if dense_linearization_materialized:
        dense_linearization_materialized, dense_report = (
            _least_squares_dense_linearization_policy(
                linearization_rows,
                linearization_cols,
                x_dtype,
                max_dense_linearization_bytes,
            )
        )
        if dense_linearization_materialized:
            residual, residual_jacobian, _flat_grad, hessian = (
                _materialize_dense_least_squares_linearization(residual_eval, x)
            )

    return {
        "x": x,
        "residual": residual,
        "residual_jacobian": residual_jacobian,
        "fun": cost,
        "grad": grad,
        "hessian": hessian,
        "damping": damping,
        "nit": nit,
        "status": status,
        "info": info,
        "success": success,
        "dense_linearization_materialized": dense_linearization_materialized,
        "dense_linearization_kind": (
            "post_hoc" if dense_linearization_materialized else None
        ),
        **dense_report,
    }


def levenberg_marquardt_traceable(
    residual_fn,
    x0,
    *,
    maxiter=1500,
    tol=1e-10,
    ftol=1e-8,
    xtol=1e-8,
    gtol=None,
    materialize_dense_linearization=True,
    max_dense_linearization_bytes=None,
    callback=None,
    progress_callback=None,
    args=(),
):
    """Trace-safe Levenberg-Marquardt solver for least-squares residuals.

    ``ftol`` and ``xtol`` feed the matrix-free MINPACK-style ``info`` subset.
    Explicit ``gtol`` values replace the legacy ``tol`` gradient gate with a
    matrix-free infinity-norm gradient gate. QR-scaled ``gtol`` ``info`` codes
    remain reserved for a future pivoted-QR MINPACK lane.
    """
    runner = _make_traceable_levenberg_marquardt_runner(
        residual_fn,
        int(maxiter),
        float(tol),
        float(ftol),
        float(xtol),
        None if gtol is None else float(gtol),
        bool(materialize_dense_linearization),
        max_dense_linearization_bytes,
        callback is not None,
        progress_callback is not None,
    )
    callback_token = _register_traceable_callback(callback)
    progress_callback_token = _register_traceable_callback(progress_callback)
    normalized_args = _normalize_solver_args(args)
    try:
        if callback_token == 0 and progress_callback_token == 0:
            result = runner(x0, normalized_args)
        else:
            result = runner(
                x0,
                normalized_args,
                callback_token,
                progress_callback_token,
            )
        if callback_token != 0 or progress_callback_token != 0:
            jax.effects_barrier()
        return result
    finally:
        _unregister_traceable_callback(callback_token)
        _unregister_traceable_callback(progress_callback_token)


def _qr_lm_dense_state(flat_residual_fn, flat_x):
    residual = flat_residual_fn(flat_x)

    def jvp_fn(x, v):
        return jax.jvp(flat_residual_fn, (x,), (v,))[1]

    jacobian = _materialize_dense_jacobian(jvp_fn, flat_x)
    gradient, hessian = _least_squares_linearization_from_jacobian(
        residual,
        jacobian,
    )
    return {
        "residual": residual,
        "jacobian": jacobian,
        "gradient": gradient,
        "hessian": hessian,
        "cost": _least_squares_cost(residual),
        "grad_norm_inf": _tree_inf_norm(gradient),
    }


def _qr_scaled_gradient_norm(residual, jacobian):
    residual = jnp.ravel(jnp.asarray(residual))
    jacobian = jnp.asarray(jacobian)
    dtype = residual.dtype
    residual_norm = jnp.linalg.norm(residual)
    column_norms = jnp.linalg.norm(jacobian, axis=0)
    numerator = jnp.abs(jacobian.T @ residual)
    denominator = column_norms * residual_norm
    cosines = jnp.where(
        denominator > _device_scalar(0.0, dtype=dtype),
        numerator / denominator,
        _device_scalar(0.0, dtype=dtype),
    )
    return jnp.max(cosines)


def _qr_lm_info(
    *,
    actual_reduction,
    predicted_reduction,
    cost,
    delta,
    x_norm,
    nit,
    maxiter,
    ftol,
    xtol,
    gtol,
    epsmch,
    qr_gnorm,
):
    info = _matrix_free_lm_info(
        actual_reduction=actual_reduction,
        predicted_reduction=predicted_reduction,
        cost=cost,
        delta=delta,
        x_norm=x_norm,
        nit=nit,
        maxiter=maxiter,
        ftol=ftol,
        xtol=xtol,
        epsmch=epsmch,
    )
    info = jnp.where(
        (info == 0) & (qr_gnorm <= gtol),
        jnp.asarray(4, dtype=jnp.int32),
        info,
    )
    info = jnp.where(
        (info == 0) & (qr_gnorm <= epsmch),
        jnp.asarray(8, dtype=jnp.int32),
        info,
    )
    return info


def _qr_lm_step(jacobian, residual, damping):
    jacobian = jnp.asarray(jacobian)
    residual = jnp.ravel(jnp.asarray(residual))
    cols = jacobian.shape[1]
    dtype = jacobian.dtype
    damping_sqrt = jnp.sqrt(jnp.asarray(damping, dtype=dtype))
    augmented_jacobian = jnp.concatenate(
        (jacobian, damping_sqrt * jnp.eye(cols, dtype=dtype)),
        axis=0,
    )
    augmented_rhs = jnp.concatenate(
        (-residual, jnp.zeros(cols, dtype=dtype)),
        axis=0,
    )
    q_matrix, r_matrix, pivots = jsp_linalg.qr(
        augmented_jacobian,
        pivoting=True,
        mode="economic",
    )
    pivoted_step = jsp_linalg.solve_triangular(
        r_matrix,
        q_matrix.T @ augmented_rhs,
        lower=False,
    )
    return jnp.zeros_like(pivoted_step).at[pivots].set(pivoted_step)


def _qr_lm_iteration(
    flat_residual_fn,
    state,
    *,
    gradient_tol,
    ftol,
    xtol,
    gtol,
    maxiter,
):
    dtype = state["x"].dtype
    defaults = _lm_defaults(dtype)
    damping = _clip_lm_damping(state["damping"], dtype=dtype)
    step = _qr_lm_step(state["jacobian"], state["residual"], damping)
    x_candidate = state["x"] + step
    candidate = _qr_lm_dense_state(flat_residual_fn, x_candidate)

    predicted_residual = state["residual"] + state["jacobian"] @ step
    predicted_reduction = state["cost"] - _least_squares_cost(predicted_residual)
    actual_reduction = state["cost"] - candidate["cost"]
    ratio = actual_reduction / jnp.maximum(
        predicted_reduction,
        defaults["predicted_floor"],
    )
    finite_candidate = (
        jnp.all(jnp.isfinite(x_candidate))
        & jnp.all(jnp.isfinite(step))
        & jnp.all(jnp.isfinite(candidate["residual"]))
        & jnp.all(jnp.isfinite(candidate["jacobian"]))
        & jnp.all(jnp.isfinite(candidate["gradient"]))
        & jnp.all(jnp.isfinite(candidate["hessian"]))
        & jnp.isfinite(candidate["cost"])
        & jnp.isfinite(predicted_reduction)
    )
    accepted = finite_candidate & (ratio >= defaults["accept_threshold"])
    step_norm = jnp.linalg.norm(step)
    delta_after_step = _lm_delta_after_step(
        state["delta"],
        step_norm,
        jnp.asarray(ratio, dtype=state["delta"].dtype),
        jnp.asarray(actual_reduction, dtype=state["delta"].dtype),
        defaults=defaults,
    )
    damping_after_accept = lax.cond(
        ratio > defaults["ratio_high"],
        lambda _: damping * defaults["decrease_factor"],
        lambda _: lax.cond(
            ratio < defaults["ratio_low"],
            lambda __: damping * defaults["increase_factor"],
            lambda __: damping,
            operand=None,
        ),
        operand=None,
    )
    next_damping = lax.cond(
        accepted,
        lambda _: _clip_lm_damping(damping_after_accept, dtype=dtype),
        lambda _: _clip_lm_damping(
            damping * defaults["increase_factor"],
            dtype=dtype,
        ),
        operand=None,
    )
    x_next = lax.select(accepted, x_candidate, state["x"])
    residual_next = lax.select(accepted, candidate["residual"], state["residual"])
    jacobian_next = lax.select(accepted, candidate["jacobian"], state["jacobian"])
    gradient_next = lax.select(accepted, candidate["gradient"], state["gradient"])
    hessian_next = lax.select(accepted, candidate["hessian"], state["hessian"])
    cost_next = lax.select(accepted, candidate["cost"], state["cost"])
    grad_norm_next = lax.select(
        accepted,
        candidate["grad_norm_inf"],
        state["grad_norm_inf"],
    )
    x_norm = jnp.linalg.norm(x_next)
    next_nit = state["nit"] + 1
    qr_gnorm = _qr_scaled_gradient_norm(residual_next, jacobian_next)
    epsmch = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
    info_candidate = _qr_lm_info(
        actual_reduction=actual_reduction,
        predicted_reduction=predicted_reduction,
        cost=state["cost"],
        delta=delta_after_step,
        x_norm=x_norm,
        nit=next_nit,
        maxiter=jnp.asarray(maxiter, dtype=jnp.int32),
        ftol=jnp.asarray(ftol, dtype=dtype),
        xtol=jnp.asarray(xtol, dtype=dtype),
        gtol=jnp.asarray(gtol, dtype=dtype),
        epsmch=epsmch,
        qr_gnorm=qr_gnorm,
    )
    info_next = lax.select(
        finite_candidate,
        info_candidate,
        jnp.asarray(0, dtype=jnp.int32),
    )
    legacy_success = grad_norm_next <= gradient_tol
    info_success = (
        (info_next == 1) | (info_next == 2) | (info_next == 3) | (info_next == 4)
    )

    return {
        "x": x_next,
        "residual": residual_next,
        "jacobian": jacobian_next,
        "gradient": gradient_next,
        "hessian": hessian_next,
        "cost": cost_next,
        "grad_norm_inf": grad_norm_next,
        "damping": next_damping,
        "delta": delta_after_step,
        "nit": next_nit,
        "status": lax.select(
            finite_candidate,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(2, dtype=jnp.int32),
        ),
        "info": info_next,
        "accepted": accepted,
        "success": finite_candidate & (legacy_success | info_success),
    }


def levenberg_marquardt_minpack_traceable(
    residual_fn,
    x0,
    *,
    maxiter=1500,
    tol=1e-10,
    ftol=1e-8,
    xtol=1e-8,
    gtol=1e-8,
    materialize_dense_linearization=True,
    max_dense_linearization_bytes=None,
    callback=None,
    progress_callback=None,
    args=(),
):
    """Trace-safe QR Levenberg-Marquardt solver for least-squares residuals.

    This opt-in lane materializes the dense Jacobian each iteration and solves
    the Marquardt augmented least-squares system with column-pivoted QR. It is
    MINPACK-style and tolerance-equivalent; it does not claim MINPACK packed-QR
    byte identity.
    """
    x = jax.tree.map(jnp.asarray, x0)
    flat_x0, unravel = ravel_pytree(x)
    normalized_args = _normalize_solver_args(args)
    dtype = flat_x0.dtype

    def residual_eval(flat_x):
        return jnp.ravel(jnp.asarray(residual_fn(unravel(flat_x), *normalized_args)))

    # Probe the residual row count by abstract shape inference on the original
    # pytree (passed as a traced arg). jax.eval_shape traces without executing,
    # so it avoids the OLD eager `residual0 = residual_eval(flat_x0)`, which
    # tripped transfer_guard("disallow"): evaluating the residual materializes
    # its weak host scalars (e.g. int64/float64 literals stacked by jnp.asarray)
    # as an implicit host->device transfer. Routing x (not unravel(flat_x)) keeps
    # the probe off ravel_pytree's unravel path as well.
    residual_shape = jax.eval_shape(
        lambda probe_x: jnp.ravel(jnp.asarray(residual_fn(probe_x, *normalized_args))),
        x,
    )
    linearization_rows = int(np.prod(residual_shape.shape))
    linearization_cols = int(flat_x0.size)
    dense_linearization_within_budget, dense_report = (
        _least_squares_dense_linearization_policy(
            linearization_rows,
            linearization_cols,
            dtype,
            max_dense_linearization_bytes,
        )
    )
    if not dense_linearization_within_budget:
        raise MemoryError(
            _least_squares_required_dense_linearization_message(
                linearization_rows,
                linearization_cols,
                dtype,
                max_dense_linearization_bytes,
            )
        )

    def run_solver(flat_x_init):
        # Build tol scalars inside the trace so they are staged as constants
        # rather than closed-over concrete device arrays (which JAX bakes via
        # mlir.ir_constant -> a device->host copy, tripping transfer_guard).
        gradient_tol = _lm_gradient_tol(tol, gtol, dtype=dtype)
        gtol_value = _device_scalar(gtol, dtype=dtype)
        initial = _qr_lm_dense_state(residual_eval, flat_x_init)
        initial_qr_gnorm = _qr_scaled_gradient_norm(
            initial["residual"],
            initial["jacobian"],
        )
        initial_info = jnp.where(
            initial_qr_gnorm <= gtol_value,
            jnp.asarray(4, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )
        state0 = {
            "x": flat_x_init,
            "residual": initial["residual"],
            "jacobian": initial["jacobian"],
            "gradient": initial["gradient"],
            "hessian": initial["hessian"],
            "cost": initial["cost"],
            "grad_norm_inf": initial["grad_norm_inf"],
            "damping": _lm_defaults(dtype)["initial_damping"],
            "delta": _lm_defaults(dtype)["initial_delta_factor"]
            * jnp.maximum(
                jnp.linalg.norm(flat_x_init),
                _device_scalar(1.0, dtype=dtype),
            ),
            "nit": jnp.asarray(0, dtype=jnp.int32),
            "status": jnp.asarray(0, dtype=jnp.int32),
            "info": initial_info,
            "accepted": jnp.asarray(False),
            "success": (initial["grad_norm_inf"] <= gradient_tol) | (initial_info == 4),
        }

        def cond_fun(state):
            return (
                (state["nit"] < maxiter)
                & (~state["success"])
                & (state["status"] != 2)
                & (state["info"] == 0)
            )

        def body_fun(state):
            next_state = _qr_lm_iteration(
                residual_eval,
                state,
                gradient_tol=gradient_tol,
                ftol=_device_scalar(ftol, dtype=dtype),
                xtol=_device_scalar(xtol, dtype=dtype),
                gtol=gtol_value,
                maxiter=maxiter,
            )
            if callback is not None:
                lax.cond(
                    next_state["accepted"],
                    lambda _: jax.debug.callback(
                        lambda flat_x: callback(
                            _hostify_optimizer_tree(unravel(flat_x))
                        ),
                        next_state["x"],
                        ordered=False,
                    ),
                    lambda _: None,
                    operand=None,
                )
            if progress_callback is not None:
                lax.cond(
                    next_state["accepted"],
                    lambda _: jax.debug.callback(
                        progress_callback,
                        next_state["nit"],
                        next_state["cost"],
                        next_state["grad_norm_inf"],
                        ordered=False,
                    ),
                    lambda _: None,
                    operand=None,
                )
            return next_state

        return lax.while_loop(cond_fun, body_fun, state0)

    state = jax.jit(run_solver)(flat_x0)
    return {
        "x": unravel(state["x"]),
        "residual": state["residual"],
        "residual_jacobian": state["jacobian"],
        "fun": state["cost"],
        "grad": unravel(state["gradient"]),
        "hessian": state["hessian"],
        "damping": state["damping"],
        "nit": state["nit"],
        "status": state["status"],
        "info": state["info"],
        "success": state["success"],
        "dense_linearization_materialized": True,
        "dense_linearization_kind": "in_loop",
        **dense_report,
    }


def jax_least_squares_optimistix(
    residual_fn,
    x0,
    *,
    maxiter=1500,
    tol=1e-10,
    ftol=_OPTIMISTIX_LM_DEFAULT_FTOL,
    xtol=_OPTIMISTIX_LM_DEFAULT_XTOL,
    gtol=_OPTIMISTIX_LM_DEFAULT_GTOL,
    materialize_dense_linearization=True,
    max_dense_linearization_bytes=None,
    callback=None,
    progress_callback=None,
    args=(),
):
    """Optional Optimistix/Lineax LSMR least-squares target lane.

    Uses ``lineax.LSMR`` rather than the Optimistix default ``lineax.QR()`` so
    the inner solve stays matrix-free on oversampled fixtures rather than
    materializing a dense Jacobian factorization per LM step.

    ``tol`` drives both the outer LM and inner LSMR (``rtol=atol=tol``).
    ``ftol``/``xtol``/``gtol``, ``callback``, and ``progress_callback`` raise
    ``ValueError``; use ``method="lm-ondevice"`` for MINPACK-style
    three-criterion termination or callback-instrumented runs.

    ``max_dense_linearization_bytes`` gates only the post-hoc Jacobian/Hessian
    materialization at the converged ``x``; LSMR is matrix-free, so it does
    not affect inner-solve memory. ``materialize_dense_linearization=False``
    skips the post-hoc step and returns ``residual_jacobian`` and ``hessian``
    as ``None``.

    Requires the Optimistix/Lineax runtime dependencies from the ``JAX`` or
    ``JAX_GPU`` extra.
    """
    _require_optimistix_lm_contract_options(
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        callback=callback,
        progress_callback=progress_callback,
    )

    x = jax.tree.map(jnp.asarray, x0)
    normalized_args = _normalize_solver_args(args)
    dtype = _require_tree_first_leaf(
        x,
        detail="Least-squares initial state must contain at least one leaf.",
    ).dtype
    tol_value = float(tol)

    def residual_eval(x_current):
        return jnp.ravel(jnp.asarray(residual_fn(x_current, *normalized_args)))

    def optx_residual(x_current, fn_args):
        return jnp.ravel(jnp.asarray(residual_fn(x_current, *fn_args)))

    solver = optx.LevenbergMarquardt(
        rtol=tol_value,
        atol=tol_value,
        linear_solver=_lineax_lsmr_solver(rtol=tol_value, atol=tol_value),
    )
    solution = optx.least_squares(
        optx_residual,
        solver,
        x,
        args=normalized_args,
        max_steps=int(maxiter),
        throw=False,
    )

    residual, cost, grad, grad_norm_inf, _ = _least_squares_gradient_state(
        residual_eval,
        solution.value,
    )
    linearization_rows = int(np.asarray(jnp.asarray(residual).size))
    linearization_cols = sum(
        int(np.asarray(jnp.asarray(leaf).size))
        for leaf in jax.tree.leaves(solution.value)
    )
    residual_jacobian = None
    hessian = None
    dense_linearization_materialized = bool(materialize_dense_linearization)
    if dense_linearization_materialized:
        dense_linearization_materialized, dense_report = (
            _least_squares_dense_linearization_policy(
                linearization_rows,
                linearization_cols,
                dtype,
                max_dense_linearization_bytes,
            )
        )
        if dense_linearization_materialized:
            residual, residual_jacobian, _flat_grad, hessian = (
                _materialize_dense_least_squares_linearization(
                    residual_eval,
                    solution.value,
                )
            )
    else:
        dense_report = _least_squares_dense_linearization_report(
            linearization_rows,
            linearization_cols,
            dtype,
            max_dense_linearization_bytes,
        )
        dense_report["failure_category"] = None
        dense_report["failure_stage"] = None
        dense_report["message"] = None

    finite = (
        _tree_all_finite(solution.value)
        & jnp.all(jnp.isfinite(residual))
        & _tree_all_finite(grad)
        & jnp.isfinite(cost)
    )
    if hessian is not None:
        finite = finite & jnp.all(jnp.isfinite(hessian))
    solution_success = jnp.asarray(solution.result == optx.RESULTS.successful)
    max_steps_reached = jnp.asarray(
        (solution.result == optx.RESULTS.nonlinear_max_steps_reached)
        | (solution.result == optx.RESULTS.max_steps_reached)
    )
    status = jnp.where(
        solution_success & finite,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.where(
            finite,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(2, dtype=jnp.int32),
        ),
    )
    info = jnp.where(
        max_steps_reached,
        jnp.asarray(5, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    return {
        "x": solution.value,
        "residual": residual,
        "residual_jacobian": residual_jacobian,
        "fun": cost,
        "grad": grad,
        "grad_norm_inf": grad_norm_inf,
        "hessian": hessian,
        "damping": None,
        "nit": jnp.asarray(solution.stats["num_steps"], dtype=jnp.int32),
        "status": status,
        "info": info,
        "success": solution_success & finite,
        "dense_linearization_materialized": dense_linearization_materialized,
        "dense_linearization_kind": (
            "post_hoc" if dense_linearization_materialized else None
        ),
        "optimistix_result": str(solution.result),
        "optimistix_result_message": str(optx.RESULTS[solution.result]),
        **dense_report,
    }


# ---------------------------------------------------------------------------
# Newton solvers (public path, no jax._src)
# ---------------------------------------------------------------------------


# Static column batch for dense Jacobian/Hessian materialization. An explicit
# override wins; otherwise CUDA derives a conservative value from the backend
# dense-operator budget while non-CUDA lanes retain the historical batch of 8.


def dense_operator_chunk_batch_size():
    """Return the static dense-operator column batch used by JAX kernels."""
    return int(_DENSE_OPERATOR_CHUNK_BATCH_SIZE)


# Solver for the inner-Boozer Gauss-Newton adjoint system (``J^T J + stab I``,
# symmetric positive-(semi)definite).  Any value other than ``"cg"`` or
# ``"lsmr_j"`` (default ``"dense"``) keeps the established path: a dense
# ``lstsq`` solve when the N x N operator fits ``max_dense_jacobian_bytes``, else
# an operator-only GMRES refinement.  ``"cg"`` solves the same square system
# matrix-free with ``lineax`` CG.  ``"lsmr_j"`` is an explicit experimental
# comparator: it requires a residual-vector closure and positive ``stab`` so the
# system is solved as a regularized least-squares problem on the unsquared
# residual Jacobian ``[J; sqrt(stab) I]``.  The unstabilized production
# ``stab=0`` case needs a KKT/two-solve formulation, not a disguised normal-
# equation solve.  Read once at import (selects a static trace-time branch).


# Operator-only square solves historically performed one residual-correction
# solve. Keep that default for LS/Hessian fallback callers; exact-jacobian
# adjoints opt into the smallest Track-A extension that clears one additional
# residual floor without exposing a public algorithm knob.

# Opt-in dense direct factorization for the EXACT-Jacobian adjoint transpose
# solve ``J^T λ = g``.  At high mode counts (m18: ``J`` is 2055x2055) the
# UNPRECONDITIONED operator-GMRES path (``restart=64``/``maxiter=10`` = 640
# matvecs) stagnates at ``residual_relative ~ 1`` because the Krylov subspace
# never resolves the spectrum, and the monotonic-rejection refinement loop then
# correctly rolls every non-improving correction back.  The un-squared exact
# Boozer Jacobian is well-conditioned (kappa(J) ~ 5.6e3), so the lit-endorsed
# fix for a small square system is a direct LU factorization plus one step of
# iterative refinement (GMRES-IR with a direct preconditioner): it solves to
# machine precision in O(n^3) where the matrix-free Krylov method stalls.  The
# 34 MB dense ``J^T`` is far under the ``max_dense_jacobian_bytes`` policy at
# m18, but the materialization stays guarded by that policy.  Read once at
# import (selects a static trace-time branch); default OFF so the operator-GMRES
# path remains the baseline for A/B comparison.


def _dense_operator_nbytes(rows, cols, dtype):
    return int(rows) * int(cols) * np.dtype(dtype).itemsize


def _dense_operator_exceeds_bytes_limit(rows, cols, dtype, max_dense_bytes):
    if max_dense_bytes is None:
        return False
    return _dense_operator_nbytes(rows, cols, dtype) > int(max_dense_bytes)


def _dense_square_operator_report(name, size, dtype, max_dense_bytes):
    return {
        f"dense_{name}_shape": (int(size), int(size)),
        f"dense_{name}_bytes": _dense_operator_nbytes(size, size, dtype),
        f"max_dense_{name}_bytes": (
            None if max_dense_bytes is None else int(max_dense_bytes)
        ),
    }


def _dense_square_operator_message(
    *,
    solver_name,
    artifact_name,
    size,
    dtype,
    max_dense_bytes,
):
    required_bytes = _dense_operator_nbytes(size, size, dtype)
    return (
        f"{solver_name} skipped dense {artifact_name} materialization because "
        f"the final {int(size)}x{int(size)} matrix in dtype {np.dtype(dtype)} "
        f"would require {required_bytes} bytes, exceeding "
        f"max_dense_{artifact_name}_bytes={int(max_dense_bytes)}."
    )


def _exact_newton_dense_jacobian_report(rows, cols, dtype, max_dense_bytes):
    return {
        "dense_jacobian_shape": (int(rows), int(cols)),
        "dense_jacobian_bytes": _dense_operator_nbytes(rows, cols, dtype),
        "max_dense_jacobian_bytes": (
            None if max_dense_bytes is None else int(max_dense_bytes)
        ),
    }


def _exact_newton_dense_jacobian_message(rows, cols, dtype, max_dense_bytes):
    required_bytes = _dense_operator_nbytes(rows, cols, dtype)
    return (
        "Exact Newton skipped dense Jacobian materialization because "
        f"the final {int(rows)}x{int(cols)} Jacobian in dtype {np.dtype(dtype)} "
        f"would require {required_bytes} bytes, exceeding "
        f"max_dense_jacobian_bytes={int(max_dense_bytes)}."
    )


def _exact_newton_dense_jacobian_policy(rows, cols, dtype, max_dense_bytes):
    report = _exact_newton_dense_jacobian_report(
        rows,
        cols,
        dtype,
        max_dense_bytes,
    )
    materialize_jacobian = not _dense_operator_exceeds_bytes_limit(
        rows,
        cols,
        dtype,
        max_dense_bytes,
    )
    report["failure_category"] = None
    report["failure_stage"] = None
    report["message"] = None
    if not materialize_jacobian:
        report["failure_category"] = "scaling_limit"
        report["failure_stage"] = "dense_jacobian_finalization"
        report["message"] = _exact_newton_dense_jacobian_message(
            rows,
            cols,
            dtype,
            max_dense_bytes,
        )
    return materialize_jacobian, report


def _stabilize_dense_hessian(H, stab):
    stab_value = _optimizer_scalar(stab, dtype=H.dtype)
    return H.at[jnp.diag_indices(H.shape[0])].add(stab_value)


def _solve_dense_newton_step(H, grad, *, refine):
    H_host = np.asarray(H, dtype=np.float64)
    grad_host = np.asarray(grad, dtype=np.float64)
    dx = np.linalg.solve(H_host, grad_host)
    if refine:
        dx = dx + np.linalg.solve(H_host, grad_host - H_host @ dx)
    return jnp.asarray(dx, dtype=jnp.asarray(grad).dtype)


def _least_squares_dense_hessian_report(size, dtype, max_dense_bytes):
    return _dense_square_operator_report(
        "hessian",
        size,
        dtype,
        max_dense_bytes,
    )


def _least_squares_dense_hessian_message(size, dtype, max_dense_bytes):
    return _dense_square_operator_message(
        solver_name="Newton polish",
        artifact_name="hessian",
        size=size,
        dtype=dtype,
        max_dense_bytes=max_dense_bytes,
    )


def _least_squares_dense_hessian_policy(size, dtype, max_dense_bytes):
    report = _least_squares_dense_hessian_report(size, dtype, max_dense_bytes)
    materialize_hessian = not _dense_operator_exceeds_bytes_limit(
        size,
        size,
        dtype,
        max_dense_bytes,
    )
    report["failure_category"] = None
    report["failure_stage"] = None
    report["message"] = None
    if not materialize_hessian:
        report["failure_category"] = "scaling_limit"
        report["failure_stage"] = "dense_hessian_finalization"
        report["message"] = _least_squares_dense_hessian_message(
            size,
            dtype,
            max_dense_bytes,
        )
    return materialize_hessian, report


def _resolve_dense_hessian_materialization(
    requested,
    size,
    dtype,
    max_dense_bytes,
):
    if not requested:
        report = _least_squares_dense_hessian_report(size, dtype, max_dense_bytes)
        report["failure_category"] = None
        report["failure_stage"] = None
        report["message"] = None
        return False, report
    return _least_squares_dense_hessian_policy(size, dtype, max_dense_bytes)


def _least_squares_dense_linearization_report(rows, cols, dtype, max_dense_bytes):
    jacobian_bytes = _dense_operator_nbytes(rows, cols, dtype)
    hessian_bytes = _dense_operator_nbytes(cols, cols, dtype)
    return {
        "dense_residual_jacobian_shape": (int(rows), int(cols)),
        "dense_residual_jacobian_bytes": jacobian_bytes,
        "dense_hessian_shape": (int(cols), int(cols)),
        "dense_hessian_bytes": hessian_bytes,
        "dense_linearization_bytes": jacobian_bytes + hessian_bytes,
        "max_dense_linearization_bytes": (
            None if max_dense_bytes is None else int(max_dense_bytes)
        ),
    }


def _least_squares_dense_linearization_message(rows, cols, dtype, max_dense_bytes):
    report = _least_squares_dense_linearization_report(
        rows,
        cols,
        dtype,
        max_dense_bytes,
    )
    return (
        "Levenberg-Marquardt skipped dense linearization materialization because "
        f"the final residual Jacobian/Hessian compatibility artifacts would "
        f"require {report['dense_linearization_bytes']} bytes in dtype "
        f"{np.dtype(dtype)}, exceeding "
        f"max_dense_linearization_bytes={int(max_dense_bytes)}."
    )


def _least_squares_required_dense_linearization_message(
    rows,
    cols,
    dtype,
    max_dense_bytes,
):
    report = _least_squares_dense_linearization_report(
        rows,
        cols,
        dtype,
        max_dense_bytes,
    )
    return (
        "Levenberg-Marquardt dense QR solve requires residual Jacobian/Hessian "
        f"artifacts totaling {report['dense_linearization_bytes']} bytes in "
        f"dtype {np.dtype(dtype)}, exceeding "
        f"max_dense_linearization_bytes={int(max_dense_bytes)}."
    )


def _least_squares_dense_linearization_policy(rows, cols, dtype, max_dense_bytes):
    report = _least_squares_dense_linearization_report(
        rows,
        cols,
        dtype,
        max_dense_bytes,
    )
    materialize_linearization = max_dense_bytes is None or report[
        "dense_linearization_bytes"
    ] <= int(max_dense_bytes)
    report["failure_category"] = None
    report["failure_stage"] = None
    report["message"] = None
    if not materialize_linearization:
        report["failure_category"] = "scaling_limit"
        report["failure_stage"] = "dense_linearization_finalization"
        report["message"] = _least_squares_dense_linearization_message(
            rows,
            cols,
            dtype,
            max_dense_bytes,
        )
    return materialize_linearization, report


def _newton_step_finite(x_next, grad_next):
    return jnp.all(jnp.isfinite(x_next)) & jnp.all(jnp.isfinite(grad_next))


def _newton_candidate_status(
    x_next,
    val_next,
    grad_next,
    *,
    alpha,
    current_val,
    current_grad,
    current_norm,
    dx,
):
    """Accept a finite Newton candidate by stationarity or Armijo merit."""
    candidate_norm = jnp.linalg.norm(grad_next)
    finite = (
        _newton_step_finite(x_next, grad_next)
        & jnp.isfinite(val_next)
        & jnp.isfinite(candidate_norm)
    )
    descent_measure = jnp.real(jnp.vdot(current_grad, dx))
    armijo_bound = current_val - (
        _device_scalar(NEWTON_ARMIJO_C1, dtype=jnp.asarray(x_next).dtype)
        * alpha
        * descent_measure
    )
    objective_decrease = (descent_measure > 0) & (val_next <= armijo_bound)
    accepted = finite & ((candidate_norm <= current_norm) | objective_decrease)
    return accepted, candidate_norm


_NEWTON_STOP_SUCCESS = 0
_NEWTON_STOP_MAXITER = 1
_NEWTON_STOP_STALLED = 2
_NEWTON_STOP_NONFINITE = 3
_NEWTON_STOP_UNKNOWN = 4

_BOUNDED_MIXED_NEWTON_ATTEMPT_LIMIT = 2
_BOUNDED_MIXED_NEWTON_ACTUAL_CORRECTION_LIMIT = 3
_BOUNDED_MIXED_NEWTON_FACTORIZATION_EVENT_LIMIT = 3
_BOUNDED_NEWTON_ATTEMPT_NONE = 0
_BOUNDED_NEWTON_ATTEMPT_FP32_PROPOSAL = 1
_BOUNDED_NEWTON_ATTEMPT_FP64_CERTIFICATE = 2

_DENSE_HESSIAN_MATERIALIZATION_NOT_REQUESTED = 0
_DENSE_HESSIAN_MATERIALIZATION_COMPLETE = 1
_DENSE_HESSIAN_MATERIALIZATION_BYTE_LIMIT = 2


def _newton_backtracking_continue(state):
    return (state["iteration"] < _NEWTON_BACKTRACKING_MAX_STEPS) & (~state["accepted"])


def _backtracking_value_grad_step(
    val_and_grad_fn,
    x,
    dx,
    current_val,
    current_grad,
    current_norm,
):
    dtype = jnp.asarray(x).dtype
    one = _device_scalar(1.0, dtype=dtype)
    half = _device_scalar(0.5, dtype=dtype)
    state0 = {
        "iteration": jnp.asarray(0, dtype=jnp.int32),
        "alpha": one,
        "x": x,
        "val": current_val,
        "grad": current_grad,
        "norm": current_norm,
        "accepted": jnp.asarray(False),
    }

    def body_fun(state):
        candidate_x = x - state["alpha"] * dx
        candidate_val, candidate_grad = val_and_grad_fn(candidate_x)
        candidate_accepted, candidate_norm = _newton_candidate_status(
            candidate_x,
            candidate_val,
            candidate_grad,
            alpha=state["alpha"],
            current_val=current_val,
            current_grad=current_grad,
            current_norm=current_norm,
            dx=dx,
        )
        return {
            "iteration": state["iteration"] + 1,
            "alpha": state["alpha"] * half,
            "x": lax.select(candidate_accepted, candidate_x, state["x"]),
            "val": lax.select(candidate_accepted, candidate_val, state["val"]),
            "grad": lax.select(candidate_accepted, candidate_grad, state["grad"]),
            "norm": lax.select(candidate_accepted, candidate_norm, state["norm"]),
            "accepted": candidate_accepted,
        }

    return lax.while_loop(_newton_backtracking_continue, body_fun, state0)


def _host_backtracking_value_grad_step(
    val_and_grad_fn,
    x,
    dx,
    current_val,
    current_grad,
    current_norm,
):
    dtype = jnp.asarray(x).dtype
    alpha = _device_scalar(1.0, dtype=dtype)
    half = _device_scalar(0.5, dtype=dtype)
    state = {
        "iteration": jnp.asarray(0, dtype=jnp.int32),
        "alpha": alpha,
        "x": x,
        "val": current_val,
        "grad": current_grad,
        "norm": current_norm,
        "accepted": jnp.asarray(False),
    }
    for iteration in range(_NEWTON_BACKTRACKING_MAX_STEPS):
        candidate_x = x - alpha * dx
        candidate_val, candidate_grad = val_and_grad_fn(candidate_x)
        candidate_accepted, candidate_norm = _newton_candidate_status(
            candidate_x,
            candidate_val,
            candidate_grad,
            alpha=alpha,
            current_val=current_val,
            current_grad=current_grad,
            current_norm=current_norm,
            dx=dx,
        )
        next_alpha = alpha * half
        if _host_bool(candidate_accepted):
            return {
                "iteration": jnp.asarray(iteration + 1, dtype=jnp.int32),
                "alpha": next_alpha,
                "x": candidate_x,
                "val": candidate_val,
                "grad": candidate_grad,
                "norm": candidate_norm,
                "accepted": jnp.asarray(True),
            }
        state = {
            **state,
            "iteration": jnp.asarray(iteration + 1, dtype=jnp.int32),
            "alpha": next_alpha,
        }
        alpha = next_alpha
    return state


def _backtracking_residual_step(residual_eval, x, dx, residual, current_norm):
    dtype = jnp.asarray(x).dtype
    one = _device_scalar(1.0, dtype=dtype)
    half = _device_scalar(0.5, dtype=dtype)
    state0 = {
        "iteration": jnp.asarray(0, dtype=jnp.int32),
        "alpha": one,
        "x": x,
        "residual": residual,
        "norm": current_norm,
        "accepted": jnp.asarray(False),
    }

    def body_fun(state):
        candidate_x = x - state["alpha"] * dx
        candidate_residual = residual_eval(candidate_x)
        candidate_norm = jnp.linalg.norm(candidate_residual)
        candidate_accepted = (
            jnp.all(jnp.isfinite(candidate_x))
            & jnp.all(jnp.isfinite(candidate_residual))
            & jnp.isfinite(candidate_norm)
            & (candidate_norm <= current_norm)
        )
        return {
            "iteration": state["iteration"] + 1,
            "alpha": state["alpha"] * half,
            "x": lax.select(candidate_accepted, candidate_x, state["x"]),
            "residual": lax.select(
                candidate_accepted,
                candidate_residual,
                state["residual"],
            ),
            "norm": lax.select(candidate_accepted, candidate_norm, state["norm"]),
            "accepted": candidate_accepted,
        }

    return lax.while_loop(_newton_backtracking_continue, body_fun, state0)


def _operator_gmres_matvec_budget(n, *, max_refinement_steps):
    """Return the worst-case operator matvec budget for the current GMRES path."""
    restart, maxiter = _gmres_iteration_limits(n)
    per_solve_budget = 1 + maxiter * (restart + 1)
    return int(per_solve_budget * (1 + int(max_refinement_steps)))


def _gmres_solve_newton_system(hvp_fn, x, rhs, *, stab, tol):
    stab_value = _optimizer_scalar(stab, dtype=rhs.dtype)

    def matvec(v):
        return hvp_fn(x, v) + stab_value * v

    dx, _ = _run_operator_gmres(matvec, rhs, tol=tol)
    residual = rhs - matvec(dx)
    return dx, residual, matvec


def _gmres_solve_exact_newton_system(jvp_fn, x, rhs, *, tol):
    def matvec(v):
        return jvp_fn(x, v)

    dx, _ = _run_operator_gmres(matvec, rhs, tol=tol)
    residual = rhs - matvec(dx)
    return dx, residual, matvec


# Growth-factor margin on the ``n * eps`` LU backward-error bound used by
# ``_effective_dense_backward_error_tolerance``. Keeps the acceptance gate a tiny
# multiple of attainable backward-stability (so degenerate solves with
# eta ~ O(1) still fail closed by ~10 orders) while admitting backward-stable
# solves of large, moderately ill-conditioned systems.
# Float64 dense solves still need a condition cap below the theoretical
# ``1 / (n * eps)`` rank threshold at small n: a consistent near-singular
# system can have machine-small residual but a 1e-4-scale wrong solution at
# condition estimates around 2e13. Production Boozer adjoint estimates are
# documented at least two orders below this cap.


def _linear_solve_requested_tolerance_reached(status):
    """Return whether the diagnostic caller target was reached."""
    return (
        status.success
        & jnp.isfinite(status.residual_relative)
        & jnp.isfinite(status.requested_tolerance)
        & (status.residual_relative <= status.requested_tolerance)
    )


def _linear_solve_status_with_relative_tolerance(
    solution,
    residual,
    rhs,
    *,
    tolerance,
    iterations,
):
    residual_norm = jnp.linalg.norm(residual)
    residual_scale = _linear_solve_residual_scale(rhs)
    residual_relative = residual_norm / residual_scale
    tolerance_value = _optimizer_scalar(tolerance, dtype=rhs.dtype)
    success = (
        _linear_solve_finite(solution, residual)
        & jnp.isfinite(residual_norm)
        & jnp.isfinite(residual_relative)
        & (residual_relative <= tolerance_value)
    )
    return _LinearSolveStatus(
        success=success,
        residual=residual_norm,
        residual_scale=residual_scale,
        residual_relative=residual_relative,
        requested_tolerance=tolerance_value,
        effective_tolerance=tolerance_value,
        iterations=_linear_solve_status_iterations(iterations),
    )


def _normalize_traceable_linear_solve_status(status, *, rhs, tolerance):
    """Materialize optional status fields before a JAX control-flow join."""
    tolerance_value = _optimizer_scalar(tolerance, dtype=rhs.dtype)
    return _LinearSolveStatus(
        success=jnp.asarray(status.success, dtype=jnp.bool_),
        residual=jnp.asarray(status.residual, dtype=rhs.dtype),
        residual_relative=jnp.asarray(status.residual_relative, dtype=rhs.dtype),
        iterations=_linear_solve_status_iterations(status.iterations),
        residual_scale=jnp.asarray(
            _linear_solve_residual_scale(rhs)
            if status.residual_scale is None
            else status.residual_scale,
            dtype=rhs.dtype,
        ),
        requested_tolerance=jnp.asarray(
            tolerance_value
            if status.requested_tolerance is None
            else status.requested_tolerance,
            dtype=rhs.dtype,
        ),
        effective_tolerance=jnp.asarray(
            tolerance_value
            if status.effective_tolerance is None
            else status.effective_tolerance,
            dtype=rhs.dtype,
        ),
    )


def _eisenstat_walker_strict_cap(tol_value, *, dtype):
    # Floor/cap match the mixed dense-IR accuracy policy SSOT so Newton forcing
    # terms and certified dense-IR tolerances share one production band.
    accuracy_policy = mixed_dense_ir_accuracy_policy()
    return jnp.minimum(
        _device_scalar(
            accuracy_policy.linear_solve_tolerance_cap,
            dtype=dtype,
        ),
        jnp.maximum(
            tol_value * _device_scalar(0.1, dtype=dtype),
            _device_scalar(
                accuracy_policy.linear_solve_tolerance_floor,
                dtype=dtype,
            ),
        ),
    )


def _eisenstat_walker_strict_cap_applies(norm, tol_value, *, dtype):
    return (
        norm
        <= _device_scalar(
            _EISENSTAT_WALKER_STRICT_CAP_NEAR_TARGET_FACTOR,
            dtype=dtype,
        )
        * tol_value
    )


def _mixed_loose_fp32_handoff_applies(norm, tol_value, *, dtype):
    handoff_floor = jnp.maximum(
        _device_scalar(
            _EISENSTAT_WALKER_STRICT_CAP_NEAR_TARGET_FACTOR,
            dtype=dtype,
        )
        * tol_value,
        _device_scalar(jnp.finfo(dtype).eps, dtype=dtype),
    )
    return norm <= handoff_floor


def _eisenstat_walker_choice2_tolerance(norm, previous_norm, *, tol):
    """Return the Eisenstat-Walker Choice-2 relative linear-solve tolerance.

    Eisenstat & Walker, "Choosing the Forcing Terms in an Inexact Newton
    Method," SIAM J. Sci. Comput. 17(1):16-32 (1996), eq. (2.6) with
    γ=0.9, α=2. The returned value is the **relative** linear residual
    tolerance (`||A·dx + r_k|| ≤ η · ||r_k||`) consumed directly as
    `tol=` by `jax.scipy.sparse.linalg.gmres`, which interprets `tol` as
    relative to `||rhs||`. The strict cap applies near the Newton target;
    earlier iterations retain the looser E-W forcing term.
    """
    dtype = norm.dtype
    accuracy_policy = mixed_dense_ir_accuracy_policy()
    tol_value = _optimizer_scalar(tol, dtype=dtype)
    strict_cap = _eisenstat_walker_strict_cap(tol_value, dtype=dtype)
    gamma = _device_scalar(_EISENSTAT_WALKER_GAMMA, dtype=dtype)
    eta_min = _device_scalar(_EISENSTAT_WALKER_MIN_ETA, dtype=dtype)
    eta_max = _device_scalar(_EISENSTAT_WALKER_MAX_ETA, dtype=dtype)
    denominator = jnp.maximum(
        previous_norm,
        _device_scalar(jnp.finfo(dtype).tiny, dtype=dtype),
    )
    ratio = norm / denominator
    eta = gamma * (ratio * ratio)
    eta = jnp.clip(eta, eta_min, eta_max)
    near_convergence = _eisenstat_walker_strict_cap_applies(
        norm,
        tol_value,
        dtype=dtype,
    )
    capped_eta = jnp.where(
        near_convergence,
        jnp.minimum(strict_cap, eta),
        eta,
    )
    return jnp.maximum(
        _device_scalar(
            accuracy_policy.linear_solve_tolerance_floor,
            dtype=dtype,
        ),
        capped_eta,
    )


def _dense_matrix_condition_numerically_safe(
    condition_estimate,
    *,
    size,
    solve_dtype,
):
    """Apply the shared dtype-specific numerical-singularity condition screen."""
    return _dense_matrix_condition_estimate_numerically_safe(
        condition_estimate,
        size=int(size),
        dtype=np.dtype(solve_dtype),
    )


# ---------------------------------------------------------------------------
# Linear-solve, dense-IR, and adjoint-solve implementations live in their
# dedicated owner modules. The explicit aliases above are compatibility
# reexports only; mutable selectors must be read and patched on their owner.
# ---------------------------------------------------------------------------
# Explicit ``import X as X`` aliases keep the declared compatibility surface
# visible and make the intentional reexports F401-safe for Ruff.


def _solve_traceable_newton_operator_gmres_with_status(matvec, rhs, *, tol):
    solution, residual, info = _gmres_solve_array_system(matvec, rhs, tol=tol)
    return solution, _linear_solve_status_with_relative_tolerance(
        solution,
        residual,
        rhs,
        tolerance=tol,
        iterations=_linear_solve_iteration_count(info),
    )


def _refine_traceable_newton_operator_gmres_solution(
    matvec,
    rhs,
    solution,
    status,
    *,
    tol,
):
    """Run one bounded refinement solve for an unconverged Newton correction."""
    status = _normalize_traceable_linear_solve_status(
        status,
        rhs=rhs,
        tolerance=tol,
    )
    residual = rhs - matvec(solution)
    correction, correction_residual, correction_info = _gmres_solve_array_system(
        matvec,
        residual,
        tol=tol,
    )
    correction_finite = _linear_solve_finite(correction, correction_residual)
    refined_solution = lax.cond(
        correction_finite,
        lambda _: solution + correction,
        lambda _: solution,
        operand=None,
    )
    refined_residual = rhs - matvec(refined_solution)
    refined_iterations = _combine_linear_solve_iteration_counts(
        status.iterations,
        _linear_solve_iteration_count(correction_info),
    )
    refined_status = _linear_solve_status_with_relative_tolerance(
        refined_solution,
        refined_residual,
        rhs,
        tolerance=tol,
        iterations=refined_iterations,
    )
    fallback_status = status._replace(
        iterations=_linear_solve_status_iterations(refined_iterations)
    )
    return lax.cond(
        correction_finite,
        lambda _: (refined_solution, refined_status),
        lambda _: (solution, fallback_status),
        operand=None,
    )


def newton_polish(
    objective_fn,
    x0,
    *,
    maxiter=40,
    tol=1e-11,
    stab=0.0,
    materialize_hessian=True,
    max_dense_hessian_bytes=None,
    dense_newton_steps=False,
    progress_callback=None,
    allow_host_control=False,
    args=(),
):
    """Newton polish using exact Hessian-vector products.

    Iterations solve the Newton system with GMRES against the exact
    Hessian linear operator, avoiding the peak memory cost of
    ``jax.hessian(objective_fn)`` on large Boozer LS problems.

    The dense Hessian is still materialized once at the final iterate so
    callers retain the existing adjoint/PLU contract.
    """
    if not allow_host_control:
        raise_if_strict_jax_fallback(
            component="newton_polish",
            detail="host-controlled Newton polish loop",
        )
    val_and_grad_fn = _cached_jit_value_and_grad(objective_fn)
    hvp_fn = _hessian_vector_product_fn(objective_fn)
    normalized_args = _normalize_solver_args(args)

    def value_and_grad_eval(x_value):
        return val_and_grad_fn(x_value, *normalized_args)

    def hvp_eval(x_value, vector):
        return hvp_fn(x_value, vector, *normalized_args)

    x = x0
    val, grad = value_and_grad_eval(x)
    norm = jnp.linalg.norm(grad)

    hessian_size = int(np.asarray(jnp.asarray(x).size))
    dense_step_materialized, dense_step_report = _resolve_dense_hessian_materialization(
        bool(dense_newton_steps),
        hessian_size,
        x.dtype,
        max_dense_hessian_bytes,
    )
    backtracking_step = (
        _host_backtracking_value_grad_step
        if allow_host_control
        else _backtracking_value_grad_step
    )
    materialize_dense_hessian_fn = (
        _materialize_dense_hessian_host
        if allow_host_control
        else _materialize_dense_hessian
    )

    nit = 0
    iterative_refinement_ran = False
    final_step_iterative_refinement_ran = False
    dense_refinement_ran = False
    final_step_dense_refinement_ran = False
    accuracy_policy = mixed_dense_ir_accuracy_policy()
    while nit < maxiter and float(norm) > tol:
        linear_tol = min(
            accuracy_policy.linear_solve_tolerance_cap,
            max(
                float(tol) * 0.1,
                accuracy_policy.linear_solve_tolerance_floor,
            ),
        )
        dense_refine_step = False
        if dense_step_materialized:
            refine_step = float(norm) < 1e-9
            dense_refine_step = refine_step
            H_step = _stabilize_dense_hessian(
                materialize_dense_hessian_fn(
                    hvp_eval,
                    x,
                    symmetrize=False,
                ),
                stab,
            )
            dx = _solve_dense_newton_step(H_step, grad, refine=refine_step)
            dense_refinement_ran = dense_refinement_ran or refine_step
            iterative_refinement_ran = iterative_refinement_ran or refine_step
        else:
            refine_step = False
            dx, linear_residual, _ = _gmres_solve_newton_system(
                hvp_eval,
                x,
                grad,
                stab=stab,
                tol=linear_tol,
            )
            linear_residual_norm = float(np.linalg.norm(np.asarray(linear_residual)))
            if (
                np.all(np.isfinite(np.asarray(dx)))
                and linear_residual_norm > linear_tol
            ):
                correction, _, _ = _gmres_solve_newton_system(
                    hvp_eval,
                    x,
                    linear_residual,
                    stab=stab,
                    tol=linear_tol,
                )
                if np.all(np.isfinite(np.asarray(correction))):
                    dx = dx + correction
                    iterative_refinement_ran = True
                    refine_step = True
        candidate = backtracking_step(
            value_and_grad_eval,
            x,
            dx,
            val,
            grad,
            norm,
        )
        if not bool(candidate["accepted"]):
            break
        x = candidate["x"]
        val = candidate["val"]
        grad = candidate["grad"]
        norm = candidate["norm"]
        nit += 1
        final_step_iterative_refinement_ran = bool(refine_step)
        final_step_dense_refinement_ran = bool(dense_refine_step)
        if progress_callback is not None:
            progress_callback(nit, float(val), float(norm))

    materialize_hessian, dense_report = _resolve_dense_hessian_materialization(
        materialize_hessian,
        hessian_size,
        x.dtype,
        max_dense_hessian_bytes,
    )
    if bool(dense_newton_steps):
        dense_report["dense_newton_steps_materialized"] = dense_step_materialized
        dense_report["dense_newton_steps_message"] = dense_step_report["message"]
    H = None
    if materialize_hessian:
        H = _stabilize_dense_hessian(
            materialize_dense_hessian_fn(
                hvp_eval,
                x,
                symmetrize=True,
            ),
            adjoint_hessian_stabilization(stab),
        )

    return {
        "x": x,
        "fun": val,
        "grad": grad,
        "hessian": H,
        "nit": nit,
        "newton_iter": nit,
        "success": bool(float(norm) <= tol),
        "final_gradient_norm": float(norm),
        "final_gradient_inf_norm": float(jnp.linalg.norm(grad, ord=jnp.inf)),
        "iterative_refinement_ran": bool(iterative_refinement_ran),
        "final_step_iterative_refinement_ran": bool(
            final_step_iterative_refinement_ran
        ),
        "dense_refinement_ran": bool(dense_refinement_ran),
        "final_step_dense_refinement_ran": bool(final_step_dense_refinement_ran),
        "hessian_materialized": materialize_hessian,
        **dense_report,
    }


def _bounded_dense_ir_newton_attempt(
    *,
    certificate_val_and_grad_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    certificate_hvp_fn: Callable[[jax.Array, jax.Array], jax.Array],
    factor_hvp_fn: Callable[[jax.Array, jax.Array], jax.Array],
    x: jax.Array,
    value: jax.Array,
    gradient: jax.Array,
    gradient_norm: jax.Array,
    linear_tol: float | jax.Array,
    stab: float | jax.Array,
    proposal: bool,
    max_refinement_corrections: int = _DENSE_IR_NEWTON_REFINEMENT_STEPS,
) -> dict[str, jax.Array]:
    """Take one live-certified dense-IR direction without low-precision apply.

    Proposal matrices may be factorized in FP32, but packed factor values are
    widened before triangular application, refinement updates, and live-operator
    residuals. The incumbent is retained when the certified direction or FP64
    candidate evaluation fails.
    """
    dtype = jnp.asarray(x).dtype
    stab_value = _optimizer_scalar(stab, dtype=dtype)

    def live_matvec(vector: jax.Array) -> jax.Array:
        return certificate_hvp_fn(x, vector) + stab_value * vector

    matrix = (
        _materialize_dense_ir_proposal_matrix(
            factor_hvp_fn,
            x,
            gradient,
            stab=stab,
        )
        if proposal
        else _dense_square_operator_matrix(
            live_matvec,
            gradient,
            matrix_dtype=dtype,
            sweep_dtype=dtype,
        )
    )
    factorization_lu, factorization_piv = jsp_linalg.lu_factor(matrix)
    application_lu_piv = (
        jnp.asarray(factorization_lu, dtype=dtype),
        factorization_piv,
    )
    (
        direction,
        linear_status,
        refinement_telemetry,
    ) = _solve_dense_ir_system_with_contraction_telemetry(
        live_matvec,
        application_lu_piv,
        gradient,
        tol=linear_tol,
        max_refinement_corrections=max_refinement_corrections,
    )
    ir_history = _fixed_dense_ir_history(
        refinement_telemetry,
        source=(
            DenseIrHistorySource.FP32_PROPOSAL_FACTORS
            if proposal
            else DenseIrHistorySource.FP64_CERTIFICATE_FACTORS
        ),
    )
    direction_finite = jnp.all(jnp.isfinite(direction))

    def run_backtracking(_: None) -> dict[str, jax.Array]:
        return _backtracking_value_grad_step(
            certificate_val_and_grad_fn,
            x,
            direction,
            value,
            gradient,
            gradient_norm,
        )

    def reject_without_backtracking(_: None) -> dict[str, jax.Array]:
        return {
            "iteration": _device_int32(0, like=x),
            "alpha": _staged_like(x, 0.0, dtype=dtype),
            "x": x,
            "val": value,
            "grad": gradient,
            "norm": gradient_norm,
            "accepted": jnp.asarray(False, dtype=jnp.bool_),
        }

    candidate = lax.cond(
        linear_status.success & direction_finite,
        run_backtracking,
        reject_without_backtracking,
        operand=None,
    )
    accepted = linear_status.success & direction_finite & candidate["accepted"]
    accepted_alpha = lax.select(
        accepted,
        candidate["alpha"] * _staged_like(x, 2.0, dtype=dtype),
        _staged_like(x, 0.0, dtype=dtype),
    )
    return {
        "x": lax.select(accepted, candidate["x"], x),
        "value": lax.select(accepted, candidate["val"], value),
        "gradient": lax.select(accepted, candidate["grad"], gradient),
        "gradient_norm": lax.select(
            accepted,
            candidate["norm"],
            gradient_norm,
        ),
        "attempted": jnp.asarray(True, dtype=jnp.bool_),
        "accepted": accepted,
        "direction_norm": jnp.linalg.norm(direction),
        "direction_finite": direction_finite,
        "linear_success": linear_status.success,
        "linear_iterations": linear_status.iterations,
        "linear_ir_correction_count": linear_status.iterations,
        "linear_ir_residual_relative_trace": ir_history.residual_relative_trace,
        "linear_ir_contraction_ratio_trace": ir_history.contraction_ratio_trace,
        "linear_ir_trace_length": ir_history.trace_length,
        "linear_ir_history_source_code": ir_history.source_code,
        "linear_residual_norm": linear_status.residual,
        "linear_residual_scale": linear_status.residual_scale,
        "linear_residual_relative": linear_status.residual_relative,
        "linear_requested_tolerance": linear_status.requested_tolerance,
        "linear_effective_tolerance": linear_status.effective_tolerance,
        # Compatibility alias retained for readers of the historical flat
        # result. Its value has always been the requested Eisenstat-Walker input.
        "linear_tolerance": _staged_like(x, linear_tol, dtype=dtype),
        "live_operator_certificate": jnp.asarray(True, dtype=jnp.bool_),
        "backtracking_iterations": candidate["iteration"],
        "accepted_alpha": accepted_alpha,
        "factorization_dtype_bits": _device_int32(
            np.dtype(matrix.dtype).itemsize * 8,
            like=x,
        ),
        "factor_application_dtype_bits": _device_int32(
            np.dtype(application_lu_piv[0].dtype).itemsize * 8,
            like=x,
        ),
        "residual_dtype_bits": _device_int32(
            np.dtype(linear_status.residual.dtype).itemsize * 8,
            like=x,
        ),
        "certificate_value_dtype_bits": _device_int32(
            np.dtype(candidate["val"].dtype).itemsize * 8,
            like=x,
        ),
        "certificate_gradient_dtype_bits": _device_int32(
            np.dtype(candidate["grad"].dtype).itemsize * 8,
            like=x,
        ),
    }


def _inactive_bounded_newton_attempt(
    x: jax.Array,
    value: jax.Array,
    gradient: jax.Array,
    gradient_norm: jax.Array,
) -> dict[str, jax.Array]:
    dtype = jnp.asarray(x).dtype
    ir_history = _inactive_dense_ir_history(x)
    return {
        "x": x,
        "value": value,
        "gradient": gradient,
        "gradient_norm": gradient_norm,
        "attempted": jnp.asarray(False, dtype=jnp.bool_),
        "accepted": jnp.asarray(False, dtype=jnp.bool_),
        "direction_norm": _staged_like(x, np.nan, dtype=dtype),
        "direction_finite": jnp.asarray(False, dtype=jnp.bool_),
        "linear_success": jnp.asarray(False, dtype=jnp.bool_),
        "linear_iterations": _device_int32(
            _LINEAR_SOLVE_ITERATIONS_UNKNOWN,
            like=x,
        ),
        "linear_ir_correction_count": _device_int32(
            _LINEAR_SOLVE_ITERATIONS_UNKNOWN,
            like=x,
        ),
        "linear_ir_residual_relative_trace": ir_history.residual_relative_trace,
        "linear_ir_contraction_ratio_trace": ir_history.contraction_ratio_trace,
        "linear_ir_trace_length": ir_history.trace_length,
        "linear_ir_history_source_code": ir_history.source_code,
        "linear_residual_norm": _staged_like(x, np.nan, dtype=dtype),
        "linear_residual_scale": _staged_like(x, np.nan, dtype=dtype),
        "linear_residual_relative": _staged_like(x, np.nan, dtype=dtype),
        "linear_requested_tolerance": _staged_like(x, np.nan, dtype=dtype),
        "linear_effective_tolerance": _staged_like(x, np.nan, dtype=dtype),
        "linear_tolerance": _staged_like(x, np.nan, dtype=dtype),
        "live_operator_certificate": jnp.asarray(False, dtype=jnp.bool_),
        "backtracking_iterations": _device_int32(
            _LINEAR_SOLVE_ITERATIONS_UNKNOWN,
            like=x,
        ),
        "accepted_alpha": _staged_like(x, np.nan, dtype=dtype),
        "factorization_dtype_bits": _device_int32(0, like=x),
        "factor_application_dtype_bits": _device_int32(0, like=x),
        "residual_dtype_bits": _device_int32(0, like=x),
        "certificate_value_dtype_bits": _device_int32(0, like=x),
        "certificate_gradient_dtype_bits": _device_int32(0, like=x),
    }


def _bounded_newton_attempt_trace(
    attempts: tuple[dict[str, jax.Array], ...],
    *,
    fp64_refactor_retry: tuple[jax.Array, ...],
) -> _BoundedNewtonAttemptTrace:
    """Stack one ordered attempt sequence without changing its semantics."""
    if len(attempts) != len(fp64_refactor_retry):
        raise ValueError("Bounded Newton attempt and retry trace lengths differ")
    linear_success = jnp.stack([attempt["linear_success"] for attempt in attempts])
    step_finite = jnp.stack([attempt["direction_finite"] for attempt in attempts])
    return _BoundedNewtonAttemptTrace(
        active=jnp.stack([attempt["attempted"] for attempt in attempts]),
        accepted=jnp.stack([attempt["accepted"] for attempt in attempts]),
        linear_success=linear_success,
        linear_iterations=jnp.stack(
            [attempt["linear_iterations"] for attempt in attempts]
        ),
        linear_ir_correction_count=jnp.stack(
            [attempt["linear_ir_correction_count"] for attempt in attempts]
        ),
        linear_ir_residual_relative=jnp.stack(
            [attempt["linear_ir_residual_relative_trace"] for attempt in attempts]
        ),
        linear_ir_contraction_ratio=jnp.stack(
            [attempt["linear_ir_contraction_ratio_trace"] for attempt in attempts]
        ),
        linear_ir_length=jnp.stack(
            [attempt["linear_ir_trace_length"] for attempt in attempts]
        ),
        linear_ir_source_code=jnp.stack(
            [attempt["linear_ir_history_source_code"] for attempt in attempts]
        ),
        linear_residual=jnp.stack(
            [attempt["linear_residual_relative"] for attempt in attempts]
        ),
        linear_residual_norm=jnp.stack(
            [attempt["linear_residual_norm"] for attempt in attempts]
        ),
        linear_residual_scale=jnp.stack(
            [attempt["linear_residual_scale"] for attempt in attempts]
        ),
        linear_requested_tolerance=jnp.stack(
            [attempt["linear_requested_tolerance"] for attempt in attempts]
        ),
        linear_effective_tolerance=jnp.stack(
            [attempt["linear_effective_tolerance"] for attempt in attempts]
        ),
        live_operator_certificate=jnp.stack(
            [attempt["live_operator_certificate"] for attempt in attempts]
        ),
        factorization_dtype_bits=jnp.stack(
            [attempt["factorization_dtype_bits"] for attempt in attempts]
        ),
        factor_application_dtype_bits=jnp.stack(
            [attempt["factor_application_dtype_bits"] for attempt in attempts]
        ),
        residual_dtype_bits=jnp.stack(
            [attempt["residual_dtype_bits"] for attempt in attempts]
        ),
        certificate_value_dtype_bits=jnp.stack(
            [attempt["certificate_value_dtype_bits"] for attempt in attempts]
        ),
        certificate_gradient_dtype_bits=jnp.stack(
            [attempt["certificate_gradient_dtype_bits"] for attempt in attempts]
        ),
        step_norm=jnp.stack([attempt["direction_norm"] for attempt in attempts]),
        step_finite=step_finite,
        backtracking_attempted=linear_success & step_finite,
        backtracking_iterations=jnp.stack(
            [attempt["backtracking_iterations"] for attempt in attempts]
        ),
        accepted_alpha=jnp.stack([attempt["accepted_alpha"] for attempt in attempts]),
        fp64_refactor_retry=jnp.stack(fp64_refactor_retry),
    )


def _bounded_newton_certificate_attempt_required(
    certificate_gradient_norm: jax.Array,
    tolerance: jax.Array,
) -> jax.Array:
    """Return whether the certified incumbent requires the second attempt."""
    return certificate_gradient_norm > tolerance


def _bounded_mixed_certificate_attempt(
    *,
    certificate_val_and_grad_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    certificate_hvp_fn: Callable[[jax.Array, jax.Array], jax.Array],
    proposal_hvp_fn: Callable[[jax.Array, jax.Array], jax.Array],
    x: jax.Array,
    value: jax.Array,
    gradient: jax.Array,
    gradient_norm: jax.Array,
    linear_tol: float | jax.Array,
    stab: float | jax.Array,
) -> _BoundedMixedCertificateAttempts:
    """Try FP32 factors against the FP64 operator, then one FP64 refactor."""
    proposal_factor_attempt = _bounded_dense_ir_newton_attempt(
        certificate_val_and_grad_fn=certificate_val_and_grad_fn,
        certificate_hvp_fn=certificate_hvp_fn,
        factor_hvp_fn=proposal_hvp_fn,
        x=x,
        value=value,
        gradient=gradient,
        gradient_norm=gradient_norm,
        linear_tol=linear_tol,
        stab=stab,
        proposal=True,
        max_refinement_corrections=MIXED_DENSE_IR_MAX_REFINEMENT_CORRECTIONS,
    )
    fp64_refactor_retry = ~proposal_factor_attempt["accepted"]

    def run_fp64_refactor(_: None) -> dict[str, jax.Array]:
        attempt = _bounded_fp64_certificate_attempt(
            certificate_val_and_grad_fn=certificate_val_and_grad_fn,
            certificate_hvp_fn=certificate_hvp_fn,
            x=x,
            value=value,
            gradient=gradient,
            gradient_norm=gradient_norm,
            linear_tol=linear_tol,
            stab=stab,
        )
        return {
            **attempt,
            "linear_ir_history_source_code": _device_int32(
                DenseIrHistorySource.FP64_REFACTOR_RETRY.value,
                like=x,
            ),
        }

    def skip_fp64_refactor(_: None) -> dict[str, jax.Array]:
        return _inactive_bounded_newton_attempt(
            x,
            value,
            gradient,
            gradient_norm,
        )

    fp64_factor_attempt = lax.cond(
        fp64_refactor_retry,
        run_fp64_refactor,
        skip_fp64_refactor,
        operand=None,
    )
    selected_attempt = {
        key: lax.select(
            fp64_refactor_retry,
            fp64_factor_attempt[key],
            proposal_factor_attempt[key],
        )
        for key in proposal_factor_attempt
    }
    factorization_dtype_bits = jnp.stack(
        (
            proposal_factor_attempt["factorization_dtype_bits"],
            fp64_factor_attempt["factorization_dtype_bits"],
        )
    )
    factorization_count = proposal_factor_attempt["attempted"].astype(
        jnp.int32
    ) + fp64_factor_attempt["attempted"].astype(jnp.int32)
    return _BoundedMixedCertificateAttempts(
        selected=selected_attempt,
        primary=proposal_factor_attempt,
        retry=fp64_factor_attempt,
        fp64_refactor_retry=fp64_refactor_retry,
        factorization_dtype_bits=factorization_dtype_bits,
        factorization_count=factorization_count,
    )


def _bounded_fp64_certificate_attempt(
    *,
    certificate_val_and_grad_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    certificate_hvp_fn: Callable[[jax.Array, jax.Array], jax.Array],
    x: jax.Array,
    value: jax.Array,
    gradient: jax.Array,
    gradient_norm: jax.Array,
    linear_tol: float | jax.Array,
    stab: float | jax.Array,
) -> dict[str, jax.Array]:
    """Take one canonical FP64-factor certificate attempt."""
    return _bounded_dense_ir_newton_attempt(
        certificate_val_and_grad_fn=certificate_val_and_grad_fn,
        certificate_hvp_fn=certificate_hvp_fn,
        factor_hvp_fn=certificate_hvp_fn,
        x=x,
        value=value,
        gradient=gradient,
        gradient_norm=gradient_norm,
        linear_tol=linear_tol,
        stab=stab,
        proposal=False,
    )


def bounded_mixed_newton_polish_traceable(
    objective_fn: _ArrayObjectiveWithArgs,
    x0: jax.Array,
    *,
    tol: float | jax.Array = 1e-11,
    stab: float | jax.Array = 0.0,
    args: tuple[object, ...] = (),
    materialize_hessian: bool = False,
    max_dense_hessian_bytes: int | None = None,
) -> dict[str, object]:
    """Run the V5 stage-local Newton composition with at most two attempts.

    Mixed mode takes one FP32 dense-IR proposal from the FP64 incumbent. After
    an accepted step that remains above tolerance, its second logical attempt
    uses fresh FP32 factors at the new state, with exactly one FP64 refactor
    retry after rejection. A rejected first proposal goes directly to the FP64
    certificate attempt at the same state instead of rebuilding identical FP32
    factors. Pure FP64 takes only the canonical FP64 attempt. This bounded
    composition does not call the general runner; both use the shared
    Armijo-aware candidate acceptance policy.
    """
    policy = get_backend_policy()
    dense_ir_policy = resolve_mixed_dense_ir_policy()
    compute_dtype = np.dtype(policy.compute_dtype)
    certificate_dtype = np.dtype(policy.runtime_dtype)
    if certificate_dtype != dense_ir_policy.certificate_dtype:
        raise RuntimeError(
            "bounded V5 Newton requires a runtime certificate dtype matching "
            f"mixed dense-IR policy ({dense_ir_policy.certificate_dtype_name})"
        )

    x_initial = jnp.asarray(x0, dtype=certificate_dtype)
    if not _dense_square_operator_lu_materialization_allowed(x_initial):
        raise RuntimeError("bounded V5 Newton requires an allowed dense LU operator")
    normalized_args = _normalize_solver_args(args)
    certificate_args = _cast_floating_tree(normalized_args, certificate_dtype)

    def certificate_objective(x: jax.Array) -> jax.Array:
        return objective_fn(x, *certificate_args)

    certificate_val_and_grad_fn = jax.value_and_grad(certificate_objective)
    certificate_grad_fn = jax.grad(certificate_objective)

    def certificate_hvp_fn(x: jax.Array, vector: jax.Array) -> jax.Array:
        return jax.jvp(certificate_grad_fn, (x,), (vector,))[1]

    value_initial, gradient_initial = certificate_val_and_grad_fn(x_initial)
    gradient_norm_initial = jnp.linalg.norm(gradient_initial)
    tol_value = _staged_like(x_initial, tol, dtype=certificate_dtype)
    mixed_proposal_enabled = (
        compute_dtype == np.dtype(np.float32)
        and certificate_dtype == dense_ir_policy.certificate_dtype
    )

    if mixed_proposal_enabled:
        proposal_args = _cast_floating_tree(normalized_args, compute_dtype)

        def proposal_objective(x: jax.Array) -> jax.Array:
            return objective_fn(x, *proposal_args)

        proposal_grad_fn = jax.grad(proposal_objective)

        def proposal_hvp_fn(x: jax.Array, vector: jax.Array) -> jax.Array:
            return jax.jvp(proposal_grad_fn, (x,), (vector,))[1]

        proposal_linear_tol = _eisenstat_walker_choice2_tolerance(
            gradient_norm_initial,
            gradient_norm_initial,
            tol=tol_value,
        )

        def run_initial_proposal(_: None) -> dict[str, jax.Array]:
            return _bounded_dense_ir_newton_attempt(
                certificate_val_and_grad_fn=certificate_val_and_grad_fn,
                certificate_hvp_fn=certificate_hvp_fn,
                factor_hvp_fn=proposal_hvp_fn,
                x=x_initial,
                value=value_initial,
                gradient=gradient_initial,
                gradient_norm=gradient_norm_initial,
                linear_tol=proposal_linear_tol,
                stab=stab,
                proposal=True,
            )

        def skip_initial_proposal(_: None) -> dict[str, jax.Array]:
            return _inactive_bounded_newton_attempt(
                x_initial,
                value_initial,
                gradient_initial,
                gradient_norm_initial,
            )

        proposal_attempt = lax.cond(
            gradient_norm_initial > tol_value,
            run_initial_proposal,
            skip_initial_proposal,
            operand=None,
        )
        certificate_x = proposal_attempt["x"]
        certificate_value = proposal_attempt["value"]
        certificate_gradient = proposal_attempt["gradient"]
        certificate_gradient_norm = proposal_attempt["gradient_norm"]
    else:
        proposal_attempt = _inactive_bounded_newton_attempt(
            x_initial,
            value_initial,
            gradient_initial,
            gradient_norm_initial,
        )
        certificate_x = x_initial
        certificate_value = value_initial
        certificate_gradient = gradient_initial
        certificate_gradient_norm = gradient_norm_initial

    certificate_linear_tol = _eisenstat_walker_choice2_tolerance(
        certificate_gradient_norm,
        gradient_norm_initial,
        tol=tol_value,
    )
    if mixed_proposal_enabled:

        def direct_fp64_certificate_attempt(
            _: None,
        ) -> _BoundedMixedCertificateAttempts:
            attempt = _bounded_fp64_certificate_attempt(
                certificate_val_and_grad_fn=certificate_val_and_grad_fn,
                certificate_hvp_fn=certificate_hvp_fn,
                x=certificate_x,
                value=certificate_value,
                gradient=certificate_gradient,
                gradient_norm=certificate_gradient_norm,
                linear_tol=certificate_linear_tol,
                stab=stab,
            )
            retry = _inactive_bounded_newton_attempt(
                certificate_x,
                certificate_value,
                certificate_gradient,
                certificate_gradient_norm,
            )
            return _BoundedMixedCertificateAttempts(
                selected=attempt,
                primary=attempt,
                retry=retry,
                fp64_refactor_retry=jnp.asarray(False, dtype=jnp.bool_),
                factorization_dtype_bits=jnp.stack(
                    (
                        attempt["factorization_dtype_bits"],
                        _device_int32(0, like=certificate_x),
                    )
                ),
                factorization_count=attempt["attempted"].astype(jnp.int32),
            )

        def inactive_certificate_attempt(
            _: None,
        ) -> _BoundedMixedCertificateAttempts:
            attempt = _inactive_bounded_newton_attempt(
                certificate_x,
                certificate_value,
                certificate_gradient,
                certificate_gradient_norm,
            )
            return _BoundedMixedCertificateAttempts(
                selected=attempt,
                primary=attempt,
                retry=attempt,
                fp64_refactor_retry=jnp.asarray(False, dtype=jnp.bool_),
                factorization_dtype_bits=jnp.zeros((2,), dtype=jnp.int32),
                factorization_count=_device_int32(0, like=certificate_x),
            )

        def run_mixed_certificate_attempt(
            _: None,
        ) -> _BoundedMixedCertificateAttempts:
            return _bounded_mixed_certificate_attempt(
                certificate_val_and_grad_fn=certificate_val_and_grad_fn,
                certificate_hvp_fn=certificate_hvp_fn,
                proposal_hvp_fn=proposal_hvp_fn,
                x=certificate_x,
                value=certificate_value,
                gradient=certificate_gradient,
                gradient_norm=certificate_gradient_norm,
                linear_tol=certificate_linear_tol,
                stab=stab,
            )

        def run_active_certificate_attempt(
            _: None,
        ) -> _BoundedMixedCertificateAttempts:
            return lax.cond(
                proposal_attempt["accepted"],
                run_mixed_certificate_attempt,
                direct_fp64_certificate_attempt,
                operand=None,
            )

        certificate_attempts = lax.cond(
            _bounded_newton_certificate_attempt_required(
                certificate_gradient_norm,
                tol_value,
            ),
            run_active_certificate_attempt,
            inactive_certificate_attempt,
            operand=None,
        )
        certificate_attempt = certificate_attempts.selected
        certificate_primary_attempt = certificate_attempts.primary
        certificate_retry_attempt = certificate_attempts.retry
        certificate_fp64_refactor_retry = certificate_attempts.fp64_refactor_retry
        certificate_factorization_dtype_bits = (
            certificate_attempts.factorization_dtype_bits
        )
        certificate_factorization_count = certificate_attempts.factorization_count
    else:

        def run_fp64_certificate_attempt(_: None) -> dict[str, jax.Array]:
            return _bounded_fp64_certificate_attempt(
                certificate_val_and_grad_fn=certificate_val_and_grad_fn,
                certificate_hvp_fn=certificate_hvp_fn,
                x=certificate_x,
                value=certificate_value,
                gradient=certificate_gradient,
                gradient_norm=certificate_gradient_norm,
                linear_tol=certificate_linear_tol,
                stab=stab,
            )

        def skip_fp64_certificate_attempt(_: None) -> dict[str, jax.Array]:
            return _inactive_bounded_newton_attempt(
                certificate_x,
                certificate_value,
                certificate_gradient,
                certificate_gradient_norm,
            )

        certificate_attempt = lax.cond(
            _bounded_newton_certificate_attempt_required(
                certificate_gradient_norm,
                tol_value,
            ),
            run_fp64_certificate_attempt,
            skip_fp64_certificate_attempt,
            operand=None,
        )
        certificate_primary_attempt = certificate_attempt
        certificate_retry_attempt = _inactive_bounded_newton_attempt(
            certificate_x,
            certificate_value,
            certificate_gradient,
            certificate_gradient_norm,
        )
        certificate_fp64_refactor_retry = jnp.asarray(False, dtype=jnp.bool_)
        certificate_factorization_dtype_bits = jnp.zeros((2,), dtype=jnp.int32)
        certificate_factorization_count = _device_int32(0, like=certificate_x)

    final_x = certificate_attempt["x"]
    final_value = certificate_attempt["value"]
    final_gradient = certificate_attempt["gradient"]
    final_gradient_norm = certificate_attempt["gradient_norm"]
    final_hessian_dtype = certificate_dtype
    materialize_final_hessian, final_hessian_report = (
        _resolve_dense_hessian_materialization(
            materialize_hessian,
            int(final_x.shape[0]),
            final_hessian_dtype,
            max_dense_hessian_bytes,
        )
    )
    final_hessian_materialization_status_code = (
        _DENSE_HESSIAN_MATERIALIZATION_COMPLETE
        if materialize_final_hessian
        else (
            _DENSE_HESSIAN_MATERIALIZATION_BYTE_LIMIT
            if materialize_hessian
            else _DENSE_HESSIAN_MATERIALIZATION_NOT_REQUESTED
        )
    )
    # This solver is itself a JAX transform boundary. Human-readable report
    # strings are not valid JAX leaves, so expose a typed status code here and
    # leave text rendering to host-side reporting.
    final_hessian_report = {
        **final_hessian_report,
        "failure_category": None,
        "failure_stage": None,
        "message": None,
    }
    final_hessian = None
    if materialize_final_hessian:
        final_hessian_stabilization = adjoint_hessian_stabilization(stab)
        certificate_stab = _staged_like(
            final_x,
            final_hessian_stabilization,
            dtype=certificate_dtype,
        )

        def final_certificate_matvec(vector):
            return certificate_hvp_fn(final_x, vector) + certificate_stab * vector

        final_hessian = _dense_square_operator_matrix(
            final_certificate_matvec,
            final_gradient,
            matrix_dtype=certificate_dtype,
            sweep_dtype=certificate_dtype,
        )
        final_hessian = _symmetrize_dense_hessian(final_hessian)
    success = (
        jnp.all(jnp.isfinite(final_x))
        & jnp.isfinite(final_value)
        & jnp.all(jnp.isfinite(final_gradient))
        & (final_gradient_norm <= tol_value)
    )
    proposal_attempted = proposal_attempt["attempted"]
    certificate_attempted = certificate_attempt["attempted"]
    attempted_iterations = proposal_attempted.astype(
        jnp.int32
    ) + certificate_attempted.astype(jnp.int32)
    accepted_iterations = proposal_attempt["accepted"].astype(
        jnp.int32
    ) + certificate_attempt["accepted"].astype(jnp.int32)
    certificate_fallback = (
        proposal_attempted & (~proposal_attempt["accepted"]) & certificate_attempted
    )
    certificate_started_from_proposal = (
        proposal_attempt["accepted"] & certificate_attempted
    )
    if mixed_proposal_enabled:
        factorization_dtype_bits_trace = jnp.concatenate(
            (
                jnp.reshape(proposal_attempt["factorization_dtype_bits"], (1,)),
                certificate_factorization_dtype_bits,
            )
        )
        factorization_event_count = (
            proposal_attempted.astype(jnp.int32) + certificate_factorization_count
        )
    else:
        factorization_dtype_bits_trace = jnp.concatenate(
            (
                jnp.reshape(certificate_attempt["factorization_dtype_bits"], (1,)),
                jnp.zeros((2,), dtype=jnp.int32),
            )
        )
        factorization_event_count = certificate_attempted.astype(jnp.int32)

    if mixed_proposal_enabled:
        trace_attempts = (proposal_attempt, certificate_attempt)
        attempt_kind_codes = (
            _BOUNDED_NEWTON_ATTEMPT_FP32_PROPOSAL,
            _BOUNDED_NEWTON_ATTEMPT_FP64_CERTIFICATE,
        )
        trace_value_before = (value_initial, certificate_value)
        trace_norm_before = (gradient_norm_initial, certificate_gradient_norm)
        trace_linear_tolerances = (proposal_linear_tol, certificate_linear_tol)
    else:
        inactive = _inactive_bounded_newton_attempt(
            final_x,
            final_value,
            final_gradient,
            final_gradient_norm,
        )
        trace_attempts = (certificate_attempt, inactive)
        attempt_kind_codes = (
            _BOUNDED_NEWTON_ATTEMPT_FP64_CERTIFICATE,
            _BOUNDED_NEWTON_ATTEMPT_NONE,
        )
        trace_value_before = (value_initial, final_value)
        trace_norm_before = (gradient_norm_initial, final_gradient_norm)
        trace_linear_tolerances = (certificate_linear_tol, certificate_linear_tol)

    false_retry = jnp.asarray(False, dtype=jnp.bool_)
    trace = _bounded_newton_attempt_trace(
        trace_attempts,
        fp64_refactor_retry=(
            (false_retry, certificate_fp64_refactor_retry)
            if mixed_proposal_enabled
            else (false_retry, false_retry)
        ),
    )
    trace_active = trace.active
    trace_accepted = trace.accepted
    trace_linear_success = trace.linear_success
    trace_linear_iterations = trace.linear_iterations
    trace_linear_ir_correction_count = trace.linear_ir_correction_count
    trace_linear_ir_residual_relative = trace.linear_ir_residual_relative
    trace_linear_ir_contraction_ratio = trace.linear_ir_contraction_ratio
    trace_linear_ir_length = trace.linear_ir_length
    trace_linear_ir_source_code = trace.linear_ir_source_code
    trace_linear_residual = trace.linear_residual
    trace_linear_residual_norm = trace.linear_residual_norm
    trace_linear_residual_scale = trace.linear_residual_scale
    trace_linear_requested_tolerance = trace.linear_requested_tolerance
    trace_linear_effective_tolerance = trace.linear_effective_tolerance
    trace_live_operator_certificate = trace.live_operator_certificate
    trace_factorization_dtype_bits = trace.factorization_dtype_bits
    trace_factor_application_dtype_bits = trace.factor_application_dtype_bits
    trace_residual_dtype_bits = trace.residual_dtype_bits
    trace_certificate_value_dtype_bits = trace.certificate_value_dtype_bits
    trace_certificate_gradient_dtype_bits = trace.certificate_gradient_dtype_bits
    trace_step_norm = trace.step_norm
    trace_step_finite = trace.step_finite
    trace_backtracking_attempted = trace.backtracking_attempted
    trace_backtracking = trace.backtracking_iterations
    trace_alpha = trace.accepted_alpha
    trace_fp64_refactor_retry = trace.fp64_refactor_retry

    actual_trace_attempts = (
        proposal_attempt,
        certificate_primary_attempt,
        certificate_retry_attempt,
    )
    actual_trace = _bounded_newton_attempt_trace(
        actual_trace_attempts,
        fp64_refactor_retry=(
            false_retry,
            false_retry,
            certificate_fp64_refactor_retry,
        ),
    )
    trace_backend_code = jnp.full(
        (_BOUNDED_MIXED_NEWTON_ATTEMPT_LIMIT,),
        _TRACEABLE_NEWTON_LINEAR_SOLVER_CODES[
            _TRACEABLE_NEWTON_LINEAR_SOLVER_HYBRID_FINAL_DENSE_IR
        ],
        dtype=jnp.int32,
    )
    nominal_dense_ir_budget = _device_int32(
        _DENSE_IR_NEWTON_MATVEC_BUDGET,
        like=final_x,
    )
    mixed_certificate_dense_ir_budget = _device_int32(
        MIXED_DENSE_IR_MAX_REFINEMENT_CORRECTIONS + 1,
        like=final_x,
    )
    certificate_dense_ir_budget = lax.select(
        proposal_attempt["accepted"],
        mixed_certificate_dense_ir_budget
        + nominal_dense_ir_budget * certificate_fp64_refactor_retry.astype(jnp.int32),
        nominal_dense_ir_budget,
    )
    trace_budget = jnp.stack(
        (
            nominal_dense_ir_budget,
            certificate_dense_ir_budget,
        )
        if mixed_proposal_enabled
        else (
            nominal_dense_ir_budget,
            nominal_dense_ir_budget,
        )
    )
    if mixed_proposal_enabled:
        last_attempt = {
            key: lax.select(
                certificate_attempted,
                certificate_attempt[key],
                proposal_attempt[key],
            )
            for key in proposal_attempt
        }
    else:
        last_attempt = certificate_attempt
    stalled = (~success) & (attempted_iterations > 0) & (~last_attempt["accepted"])
    selected_attempt_kind_code = jnp.select(
        [certificate_attempt["accepted"], proposal_attempt["accepted"]],
        [
            _device_int32(
                _BOUNDED_NEWTON_ATTEMPT_FP64_CERTIFICATE,
                like=final_x,
            ),
            _device_int32(
                _BOUNDED_NEWTON_ATTEMPT_FP32_PROPOSAL,
                like=final_x,
            ),
        ],
        default=_device_int32(_BOUNDED_NEWTON_ATTEMPT_NONE, like=final_x),
    ).astype(jnp.int32)
    stop_reason_code = jnp.select(
        [success, stalled, ~jnp.isfinite(final_gradient_norm)],
        [_NEWTON_STOP_SUCCESS, _NEWTON_STOP_STALLED, _NEWTON_STOP_NONFINITE],
        default=_NEWTON_STOP_MAXITER,
    ).astype(jnp.int32)
    return {
        "x": final_x,
        "fun": final_value,
        "grad": final_gradient,
        "hessian": final_hessian,
        "nit": accepted_iterations,
        "newton_iter": accepted_iterations,
        "success": success,
        "initial_gradient_norm": gradient_norm_initial,
        "final_gradient_norm": final_gradient_norm,
        "final_gradient_inf_norm": jnp.linalg.norm(final_gradient, ord=jnp.inf),
        "newton_attempted_iterations": attempted_iterations,
        "newton_stalled": stalled,
        "newton_stop_reason_code": stop_reason_code,
        "newton_last_step_accepted": last_attempt["accepted"],
        "newton_last_step_norm": last_attempt["direction_norm"],
        "newton_last_step_finite": last_attempt["direction_finite"],
        "newton_last_linear_solve_success": last_attempt["linear_success"],
        "newton_last_linear_solve_iterations": last_attempt["linear_iterations"],
        "newton_last_linear_ir_correction_count": last_attempt[
            "linear_ir_correction_count"
        ],
        "newton_last_linear_ir_residual_relative_trace": last_attempt[
            "linear_ir_residual_relative_trace"
        ],
        "newton_last_linear_ir_contraction_ratio_trace": last_attempt[
            "linear_ir_contraction_ratio_trace"
        ],
        "newton_last_linear_ir_trace_length": last_attempt["linear_ir_trace_length"],
        "newton_last_linear_ir_history_source_code": last_attempt[
            "linear_ir_history_source_code"
        ],
        "newton_last_linear_solve_matvec_budget": lax.select(
            certificate_attempted,
            certificate_dense_ir_budget,
            nominal_dense_ir_budget,
        ),
        "newton_last_linear_residual_relative": last_attempt[
            "linear_residual_relative"
        ],
        "newton_last_linear_residual_norm": last_attempt["linear_residual_norm"],
        "newton_last_linear_residual_scale": last_attempt["linear_residual_scale"],
        "newton_last_linear_requested_tolerance": last_attempt[
            "linear_requested_tolerance"
        ],
        "newton_last_linear_effective_tolerance": last_attempt[
            "linear_effective_tolerance"
        ],
        # Legacy requested-tolerance alias. New readers must not use it as the
        # adjusted acceptance threshold.
        "newton_last_linear_tolerance": last_attempt["linear_tolerance"],
        "newton_last_linear_live_operator_certificate": last_attempt[
            "live_operator_certificate"
        ],
        "newton_last_linear_factorization_dtype_bits": last_attempt[
            "factorization_dtype_bits"
        ],
        "newton_last_linear_factor_application_dtype_bits": last_attempt[
            "factor_application_dtype_bits"
        ],
        "newton_last_linear_residual_dtype_bits": last_attempt["residual_dtype_bits"],
        "newton_last_certificate_value_dtype_bits": last_attempt[
            "certificate_value_dtype_bits"
        ],
        "newton_last_certificate_gradient_dtype_bits": last_attempt[
            "certificate_gradient_dtype_bits"
        ],
        "newton_last_backtracking_iterations": last_attempt["backtracking_iterations"],
        "newton_last_accepted_alpha": last_attempt["accepted_alpha"],
        "newton_linear_solve_backend_code": _device_int32(
            _TRACEABLE_NEWTON_LINEAR_SOLVER_CODES[
                _TRACEABLE_NEWTON_LINEAR_SOLVER_HYBRID_FINAL_DENSE_IR
            ],
            like=final_x,
        ),
        "newton_trace_active": trace_active,
        "newton_trace_step_accepted": trace_accepted,
        "newton_trace_value_before": jnp.stack(trace_value_before),
        "newton_trace_gradient_norm_before": jnp.stack(trace_norm_before),
        "newton_trace_linear_tol": jnp.stack(trace_linear_tolerances),
        "newton_trace_step_norm": trace_step_norm,
        "newton_trace_step_finite": trace_step_finite,
        "newton_trace_linear_solve_success": trace_linear_success,
        "newton_trace_linear_solve_iterations": trace_linear_iterations,
        "newton_trace_linear_ir_correction_count": (trace_linear_ir_correction_count),
        "newton_trace_linear_ir_residual_relative": (trace_linear_ir_residual_relative),
        "newton_trace_linear_ir_contraction_ratio": (trace_linear_ir_contraction_ratio),
        "newton_trace_linear_ir_trace_length": trace_linear_ir_length,
        "newton_trace_linear_ir_history_source_code": (trace_linear_ir_source_code),
        "newton_trace_linear_solve_backend_code": trace_backend_code,
        "newton_trace_linear_solve_matvec_budget": trace_budget,
        "newton_trace_linear_residual_relative": trace_linear_residual,
        "newton_trace_linear_residual_norm": trace_linear_residual_norm,
        "newton_trace_linear_residual_scale": trace_linear_residual_scale,
        "newton_trace_linear_requested_tolerance": trace_linear_requested_tolerance,
        "newton_trace_linear_effective_tolerance": trace_linear_effective_tolerance,
        "newton_trace_linear_live_operator_certificate": (
            trace_live_operator_certificate
        ),
        "newton_trace_linear_factorization_dtype_bits": (
            trace_factorization_dtype_bits
        ),
        "newton_trace_linear_factor_application_dtype_bits": (
            trace_factor_application_dtype_bits
        ),
        "newton_trace_linear_residual_dtype_bits": trace_residual_dtype_bits,
        "newton_trace_certificate_value_dtype_bits": (
            trace_certificate_value_dtype_bits
        ),
        "newton_trace_certificate_gradient_dtype_bits": (
            trace_certificate_gradient_dtype_bits
        ),
        "newton_trace_backtracking_attempted": trace_backtracking_attempted,
        "newton_trace_backtracking_accepted": trace_accepted,
        "newton_trace_backtracking_iterations": trace_backtracking,
        "newton_trace_accepted_alpha": trace_alpha,
        "newton_trace_fp64_refactor_retry": trace_fp64_refactor_retry,
        "bounded_newton_actual_correction_limit": _device_int32(
            _BOUNDED_MIXED_NEWTON_ACTUAL_CORRECTION_LIMIT,
            like=final_x,
        ),
        "bounded_newton_actual_trace_active": actual_trace.active,
        "bounded_newton_actual_trace_accepted": actual_trace.accepted,
        "bounded_newton_actual_trace_linear_success": actual_trace.linear_success,
        "bounded_newton_actual_trace_linear_iterations": (
            actual_trace.linear_iterations
        ),
        "bounded_newton_actual_trace_ir_correction_count": (
            actual_trace.linear_ir_correction_count
        ),
        "bounded_newton_actual_trace_ir_residual_relative": (
            actual_trace.linear_ir_residual_relative
        ),
        "bounded_newton_actual_trace_ir_contraction_ratio": (
            actual_trace.linear_ir_contraction_ratio
        ),
        "bounded_newton_actual_trace_ir_length": actual_trace.linear_ir_length,
        "bounded_newton_actual_trace_ir_source_code": (
            actual_trace.linear_ir_source_code
        ),
        "bounded_newton_actual_trace_residual_relative": (actual_trace.linear_residual),
        "bounded_newton_actual_trace_residual_norm": (
            actual_trace.linear_residual_norm
        ),
        "bounded_newton_actual_trace_residual_scale": (
            actual_trace.linear_residual_scale
        ),
        "bounded_newton_actual_trace_requested_tolerance": (
            actual_trace.linear_requested_tolerance
        ),
        "bounded_newton_actual_trace_effective_tolerance": (
            actual_trace.linear_effective_tolerance
        ),
        "bounded_newton_actual_trace_live_operator_certificate": (
            actual_trace.live_operator_certificate
        ),
        "bounded_newton_actual_trace_factorization_dtype_bits": (
            actual_trace.factorization_dtype_bits
        ),
        "bounded_newton_actual_trace_factor_application_dtype_bits": (
            actual_trace.factor_application_dtype_bits
        ),
        "bounded_newton_actual_trace_residual_dtype_bits": (
            actual_trace.residual_dtype_bits
        ),
        "bounded_newton_actual_trace_certificate_value_dtype_bits": (
            actual_trace.certificate_value_dtype_bits
        ),
        "bounded_newton_actual_trace_certificate_gradient_dtype_bits": (
            actual_trace.certificate_gradient_dtype_bits
        ),
        "bounded_newton_actual_trace_direction_norm": actual_trace.step_norm,
        "bounded_newton_actual_trace_direction_finite": actual_trace.step_finite,
        "bounded_newton_actual_trace_backtracking_attempted": (
            actual_trace.backtracking_attempted
        ),
        "bounded_newton_actual_trace_backtracking_accepted": (actual_trace.accepted),
        "bounded_newton_actual_trace_backtracking_iterations": (
            actual_trace.backtracking_iterations
        ),
        "bounded_newton_actual_trace_accepted_alpha": actual_trace.accepted_alpha,
        "bounded_newton_actual_trace_fp64_refactor_retry": (
            actual_trace.fp64_refactor_retry
        ),
        "bounded_newton_attempt_limit": _device_int32(
            _BOUNDED_MIXED_NEWTON_ATTEMPT_LIMIT,
            like=final_x,
        ),
        "bounded_newton_factorization_event_limit": _device_int32(
            _BOUNDED_MIXED_NEWTON_FACTORIZATION_EVENT_LIMIT,
            like=final_x,
        ),
        "bounded_newton_factorization_event_count": factorization_event_count,
        "newton_factorization_event_count": factorization_event_count,
        "newton_fp64_refactor_retry_count": (
            certificate_fp64_refactor_retry.astype(jnp.int32)
        ),
        "bounded_newton_factorization_dtype_bits_trace": (
            factorization_dtype_bits_trace
        ),
        "bounded_newton_attempt_kind_codes": jnp.asarray(
            attempt_kind_codes,
            dtype=jnp.int32,
        ),
        "bounded_newton_proposal_attempted": proposal_attempted,
        "bounded_newton_proposal_accepted": proposal_attempt["accepted"],
        "bounded_newton_proposal_attempt_identity_code": lax.select(
            proposal_attempted,
            _device_int32(
                _BOUNDED_NEWTON_ATTEMPT_FP32_PROPOSAL,
                like=final_x,
            ),
            _device_int32(_BOUNDED_NEWTON_ATTEMPT_NONE, like=final_x),
        ),
        "bounded_newton_selected_attempt_identity_code": selected_attempt_kind_code,
        "bounded_newton_final_accepted": selected_attempt_kind_code
        != _device_int32(_BOUNDED_NEWTON_ATTEMPT_NONE, like=final_x),
        "bounded_newton_second_correction_attempted": certificate_attempted,
        "bounded_newton_second_correction_accepted": certificate_attempt["accepted"],
        "bounded_newton_fp64_certification_covered": jnp.asarray(
            np.dtype(value_initial.dtype) == np.dtype(np.float64)
            and np.dtype(gradient_initial.dtype) == np.dtype(np.float64)
            and np.dtype(final_gradient_norm.dtype) == np.dtype(np.float64),
            dtype=jnp.bool_,
        ),
        "bounded_newton_fp64_value_dtype_bits": _device_int32(
            np.dtype(value_initial.dtype).itemsize * 8,
            like=final_x,
        ),
        "bounded_newton_fp64_gradient_dtype_bits": _device_int32(
            np.dtype(gradient_initial.dtype).itemsize * 8,
            like=final_x,
        ),
        "bounded_newton_final_gradient_norm_dtype_bits": _device_int32(
            np.dtype(final_gradient_norm.dtype).itemsize * 8,
            like=final_x,
        ),
        "bounded_newton_final_gradient_tolerance": tol_value,
        # Historical aliases retained for the v1 reader. They describe the
        # conditional second correction, never overall certification coverage.
        "bounded_newton_certificate_attempted": certificate_attempted,
        "bounded_newton_certificate_accepted": certificate_attempt["accepted"],
        "bounded_newton_certificate_fallback": certificate_fallback,
        "bounded_newton_certificate_fp64_refactor_retry": (
            certificate_fp64_refactor_retry
        ),
        "bounded_newton_certificate_started_from_proposal": (
            certificate_started_from_proposal
        ),
        "hessian_materialized": bool(materialize_final_hessian),
        "dense_hessian_materialization_status_code": _device_int32(
            final_hessian_materialization_status_code,
            like=final_x,
        ),
        **final_hessian_report,
    }


def _make_traceable_newton_polish_runner(
    objective_fn,
    maxiter,
    tol,
    stab,
    materialize_hessian,
    max_dense_hessian_bytes,
    progress_callback_enabled,
    matvec_count_enabled,
    linear_solver: TraceableNewtonLinearSolver,
):
    cache_key = (
        int(maxiter),
        float(tol),
        float(stab),
        bool(materialize_hessian),
        max_dense_hessian_bytes,
        bool(progress_callback_enabled),
        bool(matvec_count_enabled),
        linear_solver,
    )
    return _cached_traceable_runner(
        _TRACEABLE_NEWTON_POLISH_RUNNER_CACHE,
        objective_fn,
        cache_key,
        lambda objective_fn_ref: _build_traceable_newton_polish_runner(
            objective_fn_ref,
            int(maxiter),
            float(tol),
            float(stab),
            bool(materialize_hessian),
            max_dense_hessian_bytes,
            bool(progress_callback_enabled),
            bool(matvec_count_enabled),
            linear_solver,
        ),
    )


def _build_traceable_newton_polish_runner(
    objective_fn_ref,
    maxiter,
    tol,
    stab,
    materialize_hessian,
    max_dense_hessian_bytes,
    progress_callback_enabled,
    matvec_count_enabled,
    linear_solver,
):
    requested_materialize_hessian = materialize_hessian

    def run_solver(
        x_init,
        fn_args,
        progress_callback_token,
        matvec_counter_token,
    ):
        objective_fn = _lookup_traceable_runner_callable(
            objective_fn_ref,
            "Newton objective",
        )

        def objective_eval(x):
            return objective_fn(x, *fn_args)

        grad_fn = jax.grad(objective_eval)
        val_and_grad_fn = jax.value_and_grad(objective_eval)

        def hvp_fn(x, v):
            return jax.jvp(grad_fn, (x,), (v,))[1]

        dtype = jnp.asarray(x_init).dtype
        tol_value = _optimizer_scalar(tol, dtype=dtype)
        val0, grad0 = val_and_grad_fn(x_init)
        norm0 = jnp.linalg.norm(grad0)
        trace_shape = (maxiter,)
        trace_false = jnp.zeros(trace_shape, dtype=jnp.bool_)
        trace_nan = jnp.full(trace_shape, jnp.nan, dtype=dtype)
        trace_unknown_int = jnp.full(
            trace_shape,
            _LINEAR_SOLVE_ITERATIONS_UNKNOWN,
            dtype=jnp.int32,
        )
        hessian_size = int(np.asarray(jnp.asarray(x_init).size))
        dense_lu_materialization_allowed = (
            _dense_square_operator_lu_materialization_allowed(x_init)
        )
        traceable_dense_lu_enabled = (
            linear_solver == _TRACEABLE_NEWTON_LINEAR_SOLVER_DENSE_LU
            and dense_lu_materialization_allowed
        )
        traceable_hybrid_dense_lu_enabled = (
            linear_solver == _TRACEABLE_NEWTON_LINEAR_SOLVER_HYBRID_FINAL_DENSE_LU
            and dense_lu_materialization_allowed
        )
        traceable_dense_ir_enabled = (
            linear_solver == _TRACEABLE_NEWTON_LINEAR_SOLVER_HYBRID_FINAL_DENSE_IR
            and dense_lu_materialization_allowed
        )
        operator_linear_solver_code = _device_int32(
            _TRACEABLE_NEWTON_LINEAR_SOLVER_CODES[
                _TRACEABLE_NEWTON_LINEAR_SOLVER_OPERATOR_GMRES
            ]
        )
        dense_linear_solver_code = _device_int32(
            _TRACEABLE_NEWTON_LINEAR_SOLVER_CODES[
                _TRACEABLE_NEWTON_LINEAR_SOLVER_DENSE_LU
            ]
        )
        hybrid_linear_solver_code = _device_int32(
            _TRACEABLE_NEWTON_LINEAR_SOLVER_CODES[
                _TRACEABLE_NEWTON_LINEAR_SOLVER_HYBRID_FINAL_DENSE_LU
            ]
        )
        dense_ir_linear_solver_code = _device_int32(
            _TRACEABLE_NEWTON_LINEAR_SOLVER_CODES[
                _TRACEABLE_NEWTON_LINEAR_SOLVER_HYBRID_FINAL_DENSE_IR
            ]
        )
        initial_linear_solver_code = (
            dense_linear_solver_code
            if traceable_dense_lu_enabled
            else (
                hybrid_linear_solver_code
                if traceable_hybrid_dense_lu_enabled
                else (
                    dense_ir_linear_solver_code
                    if traceable_dense_ir_enabled
                    else operator_linear_solver_code
                )
            )
        )
        dense_ir_linear_solve_matvec_budget = _device_int32(
            _DENSE_IR_NEWTON_MATVEC_BUDGET
        )
        dense_linear_solve_matvec_budget = _device_int32(hessian_size)
        operator_linear_solve_matvec_budget = _device_int32(
            _operator_gmres_matvec_budget(
                hessian_size,
                max_refinement_steps=0,
            )
        )
        # Near-target refinement ceiling: one extra residual matvec plus one
        # more full GMRES solve on top of the single-pass budget.
        operator_refined_linear_solve_matvec_budget = _device_int32(
            _operator_gmres_matvec_budget(
                hessian_size,
                max_refinement_steps=_SQUARE_OPERATOR_GMRES_REFINEMENT_STEPS,
            )
            + 1
        )
        initial_linear_solve_matvec_budget = (
            dense_linear_solve_matvec_budget
            if traceable_dense_lu_enabled
            else operator_linear_solve_matvec_budget
        )
        materialize_final_hessian, dense_report = (
            _resolve_dense_hessian_materialization(
                requested_materialize_hessian,
                hessian_size,
                x_init.dtype,
                max_dense_hessian_bytes,
            )
        )

        if traceable_dense_ir_enabled:
            # v2 lazy chord: the LU factors are carried in the loop state and
            # materialized on the FIRST near-target iteration (a warm chord),
            # not at the runner's entry iterate.  A chord factored at a cold
            # entry point (an x_init still far from target) yields stale
            # directions whose <= _DENSE_IR_NEWTON_MATVEC_BUDGET IR steps
            # cannot reach the tight near-target tolerance -- an ondevice
            # cold-start polish measured a plateau at ||grad|| ~4e-11 versus
            # the operator path's ~1e-13.  Factoring at the near-target entry
            # keeps the chord warm on both the warm-restart decomposed K1 path
            # and cold ondevice starts.  These placeholders are never the
            # active factors: on the first near-target iteration the loop
            # rematerializes before the dense-IR solve reads them, and
            # far-from-target iterations take the operator branch.
            # Never-read carry placeholders: the first near-target iteration
            # rematerializes before any dense-IR solve reads them.  Use cheap
            # broadcast zeros rather than an embedded ``n x n`` identity
            # constant -- smaller HLO in this compile-graph-sensitive file, and
            # the same transfer-guard-clean form as the ``trace_*`` carry seeds.
            dense_ir_matrix_dtype = _dense_square_operator_matrix_dtype(grad0)
            dense_ir_placeholder_lu = jnp.zeros(
                (hessian_size, hessian_size), dtype=dense_ir_matrix_dtype
            )
            dense_ir_placeholder_piv = jnp.zeros(hessian_size, dtype=jnp.int32)

        def cond_fun(state):
            return (
                (state["attempted_iterations"] < maxiter)
                & (state["norm"] > tol_value)
                & (~state["stalled"])
            )

        def body_fun(state):
            stab_value = _optimizer_scalar(stab, dtype=state["x"].dtype)
            strict_cap_tol = _eisenstat_walker_strict_cap(
                tol_value, dtype=state["x"].dtype
            )
            eisenstat_walker_tol = _eisenstat_walker_choice2_tolerance(
                state["norm"],
                state["previous_norm"],
                tol=tol_value,
            )
            # Inexact-Newton safeguard: after a backtracking failure on a
            # loose Eisenstat-Walker direction, the retry iteration solves at
            # the strict cap before the loop may declare a stall.  A crude
            # loose-tolerance direction failing the line search is not
            # evidence of a true stall (the pre-uncap clamped solver would
            # have converged); only a tight-direction failure is.
            linear_tol = jnp.where(
                state["retry_linear_solve_at_strict_cap"],
                jnp.minimum(eisenstat_walker_tol, strict_cap_tol),
                eisenstat_walker_tol,
            )

            def matvec(v):
                result = hvp_fn(state["x"], v) + stab_value * v
                if matvec_count_enabled:
                    jax.debug.callback(
                        _invoke_traceable_matvec_counter,
                        matvec_counter_token,
                        state["attempted_iterations"],
                        ordered=False,
                    )
                return result

            def dense_lu_solve(_):
                dx, linear_status = _solve_dense_square_operator_lu_system_with_status(
                    matvec,
                    state["grad"],
                    tol=linear_tol,
                )
                return (
                    dx,
                    linear_status,
                    dense_linear_solver_code,
                    dense_linear_solve_matvec_budget,
                )

            def operator_gmres_solve(_):
                dx, linear_status = _solve_traceable_newton_operator_gmres_with_status(
                    matvec,
                    state["grad"],
                    tol=linear_tol,
                )
                linear_status = _normalize_traceable_linear_solve_status(
                    linear_status,
                    rhs=state["grad"],
                    tolerance=linear_tol,
                )
                # In the Eisenstat-Walker strict-cap region an unconverged
                # single-pass GMRES correction gets one bounded refinement
                # pass, so the tight final tolerance is actually achieved
                # (restarted GMRES plateaus on round-off above it).  Away
                # from the target the loose E-W tolerance is met in one
                # pass and no refinement cost is paid.
                near_target_refinement = (
                    _eisenstat_walker_strict_cap_applies(
                        state["norm"],
                        tol_value,
                        dtype=state["x"].dtype,
                    )
                    | state["retry_linear_solve_at_strict_cap"]
                ) & (~linear_status.success)

                def refined_solve(_inner):
                    refined_dx, refined_status = (
                        _refine_traceable_newton_operator_gmres_solution(
                            matvec,
                            state["grad"],
                            dx,
                            linear_status,
                            tol=linear_tol,
                        )
                    )
                    return (
                        refined_dx,
                        refined_status,
                        operator_refined_linear_solve_matvec_budget,
                    )

                def single_pass_solve(_inner):
                    return (
                        dx,
                        linear_status,
                        operator_linear_solve_matvec_budget,
                    )

                (
                    dx_final,
                    linear_status_final,
                    active_matvec_budget,
                ) = lax.cond(
                    near_target_refinement,
                    refined_solve,
                    single_pass_solve,
                    operand=None,
                )
                return (
                    dx_final,
                    linear_status_final,
                    operator_linear_solver_code,
                    active_matvec_budget,
                )

            if traceable_dense_lu_enabled:
                (
                    dx,
                    linear_status,
                    active_linear_solver_code,
                    active_linear_solve_matvec_budget,
                ) = dense_lu_solve(None)
            elif traceable_hybrid_dense_lu_enabled:
                # The strict-cap retry exists to hand the line search one
                # quality direction before the loop may stall; strict-cap
                # operator GMRES runs to essentially the full Krylov
                # dimension on the squared-conditioned Hessian, while the
                # dense direction is tolerance-exact at about half that
                # matvec cost.  Routing the retry through dense-LU leaves
                # hybrid mode with no strict-tolerance GMRES entry point,
                # and a rejected dense retry still stalls immediately.
                use_dense_lu_iteration = (
                    _eisenstat_walker_strict_cap_applies(
                        state["norm"],
                        tol_value,
                        dtype=state["x"].dtype,
                    )
                    | state["retry_linear_solve_at_strict_cap"]
                )
                (
                    dx,
                    linear_status,
                    active_linear_solver_code,
                    active_linear_solve_matvec_budget,
                ) = lax.cond(
                    use_dense_lu_iteration,
                    dense_lu_solve,
                    operator_gmres_solve,
                    operand=None,
                )
            elif traceable_dense_ir_enabled:
                # Same routing as hybrid: exact-by-refinement directions for
                # near-target iterations AND for the strict-cap retry; loose
                # far-from-target iterations stay on single-pass operator
                # GMRES.  A rejected dense-IR direction stalls immediately
                # (active code is not the operator code), so the mode has no
                # strict-tolerance GMRES entry point and no retry churn.
                near_target_now = _eisenstat_walker_strict_cap_applies(
                    state["norm"],
                    tol_value,
                    dtype=state["x"].dtype,
                )
                use_dense_ir_iteration = (
                    near_target_now | state["retry_linear_solve_at_strict_cap"]
                )

                # Materialize the chord factors at the current iterate the
                # first time the polish enters the near-target region, then
                # reuse those warm factors for the remaining near-target
                # iterations.  The build's HVPs go through ``entry_matvec``
                # (uncounted), so only the <=3 IR matvecs register in the
                # per-iteration telemetry, exactly as the v1 pre-loop build
                # did.  ``factors_ready`` latches only for a NEAR-TARGET build:
                # a strict-cap retry can fire while still far from target, and
                # latching a far chord would let later near-target iterations
                # reuse stale factors (the plateau v2 fixes).  A far retry thus
                # re-materializes an exact direction at its own iterate without
                # persisting it.  This trades v1's factor-once amortization for
                # a fresh uncounted n-HVP build + LU per far retry (not in the
                # matvec budget) -- the deliberate price of an exact direction
                # over a stale chord; retries are rare, maxiter-bounded, and one
                # such build is cheaper than the strict-cap operator grind it
                # replaces.
                def materialize_dense_ir_factors(_):
                    def entry_matvec(vector):
                        return hvp_fn(state["x"], vector) + stab_value * vector

                    lu_new, piv_new = jsp_linalg.lu_factor(
                        _dense_square_operator_matrix(entry_matvec, state["grad"])
                    )
                    return lu_new, piv_new, near_target_now

                def keep_dense_ir_factors(_):
                    return (
                        state["dense_ir_hessian_lu"],
                        state["dense_ir_hessian_piv"],
                        state["dense_ir_factors_ready"],
                    )

                needs_dense_ir_factors = use_dense_ir_iteration & (
                    ~state["dense_ir_factors_ready"]
                )
                (
                    dense_ir_hessian_lu,
                    dense_ir_hessian_piv,
                    dense_ir_factors_ready,
                ) = lax.cond(
                    needs_dense_ir_factors,
                    materialize_dense_ir_factors,
                    keep_dense_ir_factors,
                    operand=None,
                )

                def dense_ir_solve(_):
                    dx_ir, ir_status = _solve_dense_ir_system_with_status(
                        matvec,
                        (dense_ir_hessian_lu, dense_ir_hessian_piv),
                        state["grad"],
                        tol=linear_tol,
                    )
                    return (
                        dx_ir,
                        ir_status,
                        dense_ir_linear_solver_code,
                        dense_ir_linear_solve_matvec_budget,
                    )

                (
                    dx,
                    linear_status,
                    active_linear_solver_code,
                    active_linear_solve_matvec_budget,
                ) = lax.cond(
                    use_dense_ir_iteration,
                    dense_ir_solve,
                    operator_gmres_solve,
                    operand=None,
                )
            else:
                (
                    dx,
                    linear_status,
                    active_linear_solver_code,
                    active_linear_solve_matvec_budget,
                ) = operator_gmres_solve(None)
            step_norm = jnp.linalg.norm(dx)
            step_finite = jnp.all(jnp.isfinite(dx))
            candidate = _backtracking_value_grad_step(
                val_and_grad_fn,
                state["x"],
                dx,
                state["val"],
                state["grad"],
                state["norm"],
            )
            accepted = candidate["accepted"]
            # A rejected step only stalls the loop when the direction was
            # already solved at (or below) the strict cap; a rejected loose
            # operator-GMRES direction schedules one strict-cap retry
            # iteration instead.  Dense-LU directions are tolerance-exact, so
            # their rejections stall immediately as before.
            # Non-finite directions keep the immediate fail-loud stall; only a
            # finite crude direction earns the strict-cap retry.
            loose_operator_direction = (
                (linear_tol > strict_cap_tol)
                & (active_linear_solver_code == operator_linear_solver_code)
                & step_finite
            )
            retry_at_strict_cap = (~accepted) & loose_operator_direction
            stalled = (~accepted) & (~loose_operator_direction)
            next_nit = state["nit"] + 1
            next_attempted_iterations = state["attempted_iterations"] + 1
            accepted_alpha = lax.select(
                accepted,
                candidate["alpha"] * _optimizer_scalar(2.0, dtype=dtype),
                _optimizer_scalar(0.0, dtype=dtype),
            )
            trace_index = state["attempted_iterations"]
            if progress_callback_enabled:
                lax.cond(
                    accepted,
                    lambda _: jax.debug.callback(
                        _invoke_traceable_progress_callback,
                        progress_callback_token,
                        next_nit,
                        candidate["val"],
                        candidate["norm"],
                        ordered=False,
                    ),
                    lambda _: None,
                    operand=None,
                )
            next_state = {
                "x": lax.select(accepted, candidate["x"], state["x"]),
                "val": lax.select(accepted, candidate["val"], state["val"]),
                "grad": lax.select(accepted, candidate["grad"], state["grad"]),
                "norm": lax.select(accepted, candidate["norm"], state["norm"]),
                "previous_norm": lax.select(
                    accepted,
                    state["norm"],
                    state["previous_norm"],
                ),
                "nit": lax.select(accepted, next_nit, state["nit"]),
                "stalled": stalled,
                "retry_linear_solve_at_strict_cap": retry_at_strict_cap,
                "attempted_iterations": next_attempted_iterations,
                "last_step_accepted": accepted,
                "last_step_norm": step_norm,
                "last_step_finite": step_finite,
                "last_linear_solve_success": linear_status.success,
                "last_linear_solve_iterations": linear_status.iterations,
                "last_linear_solve_matvec_budget": active_linear_solve_matvec_budget,
                "last_linear_residual_relative": linear_status.residual_relative,
                "last_backtracking_iterations": candidate["iteration"],
                "last_accepted_alpha": accepted_alpha,
                "newton_trace_active": state["newton_trace_active"]
                .at[trace_index]
                .set(True),
                "newton_trace_step_accepted": state["newton_trace_step_accepted"]
                .at[trace_index]
                .set(accepted),
                "newton_trace_value_before": state["newton_trace_value_before"]
                .at[trace_index]
                .set(state["val"]),
                "newton_trace_gradient_norm_before": state[
                    "newton_trace_gradient_norm_before"
                ]
                .at[trace_index]
                .set(state["norm"]),
                "newton_trace_linear_tol": state["newton_trace_linear_tol"]
                .at[trace_index]
                .set(linear_tol),
                "newton_trace_step_norm": state["newton_trace_step_norm"]
                .at[trace_index]
                .set(step_norm),
                "newton_trace_step_finite": state["newton_trace_step_finite"]
                .at[trace_index]
                .set(step_finite),
                "newton_trace_linear_solve_success": state[
                    "newton_trace_linear_solve_success"
                ]
                .at[trace_index]
                .set(linear_status.success),
                "newton_trace_linear_solve_iterations": state[
                    "newton_trace_linear_solve_iterations"
                ]
                .at[trace_index]
                .set(linear_status.iterations),
                "newton_trace_linear_solve_backend_code": state[
                    "newton_trace_linear_solve_backend_code"
                ]
                .at[trace_index]
                .set(active_linear_solver_code),
                "newton_trace_linear_solve_matvec_budget": state[
                    "newton_trace_linear_solve_matvec_budget"
                ]
                .at[trace_index]
                .set(active_linear_solve_matvec_budget),
                "newton_trace_linear_residual_relative": state[
                    "newton_trace_linear_residual_relative"
                ]
                .at[trace_index]
                .set(linear_status.residual_relative),
                "newton_trace_backtracking_iterations": state[
                    "newton_trace_backtracking_iterations"
                ]
                .at[trace_index]
                .set(candidate["iteration"]),
                "newton_trace_accepted_alpha": state["newton_trace_accepted_alpha"]
                .at[trace_index]
                .set(accepted_alpha),
                "newton_linear_solve_backend_code": active_linear_solver_code,
            }
            if traceable_dense_ir_enabled:
                # Thread the lazily materialized chord factors and the
                # "factors ready" flag through the loop carry so the first
                # near-target iteration factors once and later iterations
                # reuse it.
                next_state["dense_ir_hessian_lu"] = dense_ir_hessian_lu
                next_state["dense_ir_hessian_piv"] = dense_ir_hessian_piv
                next_state["dense_ir_factors_ready"] = dense_ir_factors_ready
            return next_state

        initial_state = {
            "x": x_init,
            "val": val0,
            "grad": grad0,
            "norm": norm0,
            "previous_norm": norm0,
            "nit": jnp.asarray(0, dtype=jnp.int32),
            "stalled": jnp.asarray(False),
            "retry_linear_solve_at_strict_cap": jnp.asarray(False),
            "attempted_iterations": jnp.asarray(0, dtype=jnp.int32),
            "last_step_accepted": jnp.asarray(False),
            "last_step_norm": _optimizer_scalar(jnp.nan, dtype=dtype),
            "last_step_finite": jnp.asarray(False),
            "last_linear_solve_success": jnp.asarray(False),
            "last_linear_solve_iterations": jnp.asarray(
                _LINEAR_SOLVE_ITERATIONS_UNKNOWN,
                dtype=jnp.int32,
            ),
            "last_linear_solve_matvec_budget": (initial_linear_solve_matvec_budget),
            "last_linear_residual_relative": _optimizer_scalar(jnp.nan, dtype=dtype),
            "last_backtracking_iterations": jnp.asarray(
                _LINEAR_SOLVE_ITERATIONS_UNKNOWN,
                dtype=jnp.int32,
            ),
            "last_accepted_alpha": _optimizer_scalar(jnp.nan, dtype=dtype),
            "newton_trace_active": trace_false,
            "newton_trace_step_accepted": trace_false,
            "newton_trace_value_before": trace_nan,
            "newton_trace_gradient_norm_before": trace_nan,
            "newton_trace_linear_tol": trace_nan,
            "newton_trace_step_norm": trace_nan,
            "newton_trace_step_finite": trace_false,
            "newton_trace_linear_solve_success": trace_false,
            "newton_trace_linear_solve_iterations": trace_unknown_int,
            "newton_trace_linear_solve_backend_code": trace_unknown_int,
            "newton_trace_linear_solve_matvec_budget": trace_unknown_int,
            "newton_trace_linear_residual_relative": trace_nan,
            "newton_trace_backtracking_iterations": trace_unknown_int,
            "newton_trace_accepted_alpha": trace_nan,
            "newton_linear_solve_backend_code": initial_linear_solver_code,
        }
        if traceable_dense_ir_enabled:
            initial_state["dense_ir_hessian_lu"] = dense_ir_placeholder_lu
            initial_state["dense_ir_hessian_piv"] = dense_ir_placeholder_piv
            initial_state["dense_ir_factors_ready"] = jnp.asarray(False)
        state = lax.while_loop(cond_fun, body_fun, initial_state)

        val_final, grad_final = val_and_grad_fn(state["x"])
        norm_final = jnp.linalg.norm(grad_final)
        norm_final_finite = jnp.isfinite(norm_final)
        success = norm_final <= tol_value
        stop_reason_code = jnp.select(
            [
                success,
                state["stalled"],
                ~norm_final_finite,
                state["attempted_iterations"] >= jnp.asarray(maxiter, dtype=jnp.int32),
            ],
            [
                jnp.asarray(_NEWTON_STOP_SUCCESS, dtype=jnp.int32),
                jnp.asarray(_NEWTON_STOP_STALLED, dtype=jnp.int32),
                jnp.asarray(_NEWTON_STOP_NONFINITE, dtype=jnp.int32),
                jnp.asarray(_NEWTON_STOP_MAXITER, dtype=jnp.int32),
            ],
            default=jnp.asarray(_NEWTON_STOP_UNKNOWN, dtype=jnp.int32),
        )
        H = None
        if materialize_final_hessian:
            H = _stabilize_dense_hessian(
                _materialize_dense_hessian(hvp_fn, state["x"]),
                adjoint_hessian_stabilization(stab),
            )

        return {
            "x": state["x"],
            "fun": val_final,
            "grad": grad_final,
            "hessian": H,
            "nit": state["nit"],
            "newton_iter": state["nit"],
            "success": success,
            "initial_gradient_norm": norm0,
            "final_gradient_norm": norm_final,
            "final_gradient_inf_norm": jnp.linalg.norm(grad_final, ord=jnp.inf),
            "newton_attempted_iterations": state["attempted_iterations"],
            "newton_stalled": state["stalled"],
            "newton_stop_reason_code": stop_reason_code,
            "newton_last_step_accepted": state["last_step_accepted"],
            "newton_last_step_norm": state["last_step_norm"],
            "newton_last_step_finite": state["last_step_finite"],
            "newton_last_linear_solve_success": state["last_linear_solve_success"],
            "newton_last_linear_solve_iterations": state[
                "last_linear_solve_iterations"
            ],
            "newton_last_linear_solve_matvec_budget": state[
                "last_linear_solve_matvec_budget"
            ],
            "newton_last_linear_residual_relative": state[
                "last_linear_residual_relative"
            ],
            "newton_last_backtracking_iterations": state[
                "last_backtracking_iterations"
            ],
            "newton_last_accepted_alpha": state["last_accepted_alpha"],
            "newton_linear_solve_backend_code": state[
                "newton_linear_solve_backend_code"
            ],
            "newton_trace_active": state["newton_trace_active"],
            "newton_trace_step_accepted": state["newton_trace_step_accepted"],
            "newton_trace_value_before": state["newton_trace_value_before"],
            "newton_trace_gradient_norm_before": state[
                "newton_trace_gradient_norm_before"
            ],
            "newton_trace_linear_tol": state["newton_trace_linear_tol"],
            "newton_trace_step_norm": state["newton_trace_step_norm"],
            "newton_trace_step_finite": state["newton_trace_step_finite"],
            "newton_trace_linear_solve_success": state[
                "newton_trace_linear_solve_success"
            ],
            "newton_trace_linear_solve_iterations": state[
                "newton_trace_linear_solve_iterations"
            ],
            "newton_trace_linear_solve_backend_code": state[
                "newton_trace_linear_solve_backend_code"
            ],
            "newton_trace_linear_solve_matvec_budget": state[
                "newton_trace_linear_solve_matvec_budget"
            ],
            "newton_trace_linear_residual_relative": state[
                "newton_trace_linear_residual_relative"
            ],
            "newton_trace_backtracking_iterations": state[
                "newton_trace_backtracking_iterations"
            ],
            "newton_trace_accepted_alpha": state["newton_trace_accepted_alpha"],
            "hessian_materialized": materialize_final_hessian,
            **dense_report,
        }

    run_solver.__name__ = "traceable_newton_polish_run_solver"
    if not progress_callback_enabled and not matvec_count_enabled:

        def run_solver_without_callback(x_init, fn_args):
            return run_solver(x_init, fn_args, 0, 0)

        run_solver_without_callback.__name__ = run_solver.__name__
        return jax.jit(run_solver_without_callback)
    if not progress_callback_enabled:

        def run_solver_with_matvec_counter(
            x_init,
            fn_args,
            matvec_counter_token,
        ):
            return run_solver(x_init, fn_args, 0, matvec_counter_token)

        run_solver_with_matvec_counter.__name__ = run_solver.__name__
        return jax.jit(run_solver_with_matvec_counter)
    if not matvec_count_enabled:

        def run_solver_with_progress_callback(
            x_init,
            fn_args,
            progress_callback_token,
        ):
            return run_solver(x_init, fn_args, progress_callback_token, 0)

        run_solver_with_progress_callback.__name__ = run_solver.__name__
        return jax.jit(run_solver_with_progress_callback, static_argnums=(2,))
    return jax.jit(run_solver, static_argnums=(2,))


def newton_polish_traceable(
    objective_fn,
    x0,
    *,
    maxiter=40,
    tol=1e-11,
    stab=0.0,
    materialize_hessian=True,
    max_dense_hessian_bytes=None,
    linear_solver: TraceableNewtonLinearSolver = (
        _TRACEABLE_NEWTON_LINEAR_SOLVER_OPERATOR_GMRES
    ),
    progress_callback=None,
    args=(),
):
    """Trace-safe Newton polish for JAX-traceable objective paths.

    This variant keeps all loop state and step decisions inside JAX control
    flow so higher-level traced objectives can invoke the Newton stage without
    crossing back into Python. Newton corrections default to operator GMRES;
    in the Eisenstat-Walker strict-cap region an unconverged single-pass
    correction receives one bounded iterative-refinement pass so the tight
    final tolerance is actually achieved. The explicit ``linear_solver``
    selector enables dense comparator and factor-reuse modes without changing
    the operator-GMRES default.
    The dense Hessian policy only controls final compatibility metadata.
    """
    linear_solver = _resolve_traceable_newton_linear_solver(linear_solver)
    matvec_count_enabled = _traceable_newton_matvec_counts_requested()
    runner = _make_traceable_newton_polish_runner(
        objective_fn,
        int(maxiter),
        float(tol),
        float(stab),
        bool(materialize_hessian),
        max_dense_hessian_bytes,
        progress_callback is not None,
        matvec_count_enabled,
        linear_solver,
    )
    progress_callback_token = _register_traceable_callback(progress_callback)
    matvec_counter_token = (
        _register_traceable_matvec_counter(int(maxiter)) if matvec_count_enabled else 0
    )
    normalized_args = _normalize_solver_args(args)
    try:
        if progress_callback_token == 0 and matvec_counter_token == 0:
            result = runner(x0, normalized_args)
        elif progress_callback_token == 0:
            result = runner(x0, normalized_args, matvec_counter_token)
        elif matvec_counter_token == 0:
            result = runner(
                x0,
                normalized_args,
                progress_callback_token,
            )
        else:
            result = runner(
                x0,
                normalized_args,
                progress_callback_token,
                matvec_counter_token,
            )
        if progress_callback_token != 0 or matvec_counter_token != 0:
            jax.effects_barrier()
        if matvec_counter_token != 0 and _is_jax_tracer(result["newton_trace_active"]):
            result = dict(result)
            result["newton_matvec_counter_token"] = _device_int32(matvec_counter_token)
            matvec_counter_token = 0
        else:
            matvec_counts = _drain_traceable_matvec_counter(matvec_counter_token)
            matvec_counter_token = 0
            if matvec_counts is not None:
                result = dict(result)
                active = _host_array(result["newton_trace_active"], dtype=bool)
                actual = np.full(
                    (int(maxiter),),
                    _LINEAR_SOLVE_ITERATIONS_UNKNOWN,
                    dtype=np.int32,
                )
                actual[active] = np.asarray(matvec_counts, dtype=np.int32)[active]
                attempted = _host_int(result["newton_attempted_iterations"])
                last_actual = (
                    _LINEAR_SOLVE_ITERATIONS_UNKNOWN
                    if attempted <= 0
                    else int(actual[attempted - 1])
                )
                result["newton_trace_linear_solve_matvec_actual"] = jnp.asarray(
                    actual,
                    dtype=jnp.int32,
                )
                result["newton_last_linear_solve_matvec_actual"] = _device_int32(
                    last_actual
                )
        return result
    finally:
        _unregister_traceable_callback(progress_callback_token)
        _unregister_traceable_matvec_counter(matvec_counter_token)


def newton_exact(
    residual_fn,
    x0,
    *,
    maxiter=40,
    tol=1e-13,
    max_dense_jacobian_bytes=None,
):
    """Newton solver for the exact Boozer residual system ``r(x) = 0``.

    Iterations solve the linearized system with GMRES against exact
    Jacobian-vector products, avoiding dense Jacobian materialization in the
    hot loop. The dense Jacobian is rebuilt once at the final iterate only for
    public compatibility metadata and diagnostics.
    """
    raise_if_strict_jax_fallback(
        component="newton_exact",
        detail="host-controlled exact Newton loop",
    )
    res_fn = jax.jit(residual_fn)
    jvp_fn = _jacobian_vector_product_fn(residual_fn)

    x = x0
    r = res_fn(x)
    norm = jnp.linalg.norm(r)
    accuracy_policy = mixed_dense_ir_accuracy_policy()
    linear_tol = min(
        accuracy_policy.linear_solve_tolerance_cap,
        max(
            float(tol) * 0.1,
            accuracy_policy.linear_solve_tolerance_floor,
        ),
    )

    nit = 0
    exact_newton_linear_residual_rel = None
    exact_refinement_correction_rel = None
    while nit < maxiter and float(norm) > tol:
        dx, linear_residual, _ = _gmres_solve_exact_newton_system(
            jvp_fn,
            x,
            r,
            tol=linear_tol,
        )
        dx_before_refinement = dx
        exact_newton_linear_residual_rel = float(
            _relative_residual_norm(linear_residual, r)
        )
        linear_residual_norm = float(np.linalg.norm(np.asarray(linear_residual)))
        if not np.all(np.isfinite(np.asarray(dx))):
            break
        if linear_residual_norm > linear_tol:
            correction, _, _ = _gmres_solve_exact_newton_system(
                jvp_fn,
                x,
                linear_residual,
                tol=linear_tol,
            )
            if np.all(np.isfinite(np.asarray(correction))):
                dx = dx + correction
                denominator = np.linalg.norm(np.asarray(dx_before_refinement))
                exact_refinement_correction_rel = float(
                    np.linalg.norm(np.asarray(correction)) / max(denominator, 1e-30)
                )
        x_candidate = x - dx
        r_candidate = res_fn(x_candidate)
        norm_candidate = jnp.linalg.norm(r_candidate)
        if float(norm_candidate) <= float(norm):
            x = x_candidate
            r = r_candidate
            norm = norm_candidate
        else:
            break
        nit += 1

    rows = int(np.prod(np.shape(r)))
    cols = int(np.prod(np.shape(x)))
    materialize_jacobian, report = _exact_newton_dense_jacobian_policy(
        rows,
        cols,
        x.dtype,
        max_dense_jacobian_bytes,
    )
    if not materialize_jacobian:
        return {
            "x": x,
            "residual": r,
            "jacobian": None,
            "nit": nit,
            "success": bool(float(norm) <= tol),
            "jacobian_materialized": False,
            "exact_newton_linear_residual_rel": exact_newton_linear_residual_rel,
            "exact_refinement_correction_rel": exact_refinement_correction_rel,
            **report,
        }

    J = _materialize_dense_jacobian(jvp_fn, x)

    return {
        "x": x,
        "residual": r,
        "jacobian": J,
        "nit": nit,
        "success": bool(float(norm) <= tol),
        "jacobian_materialized": True,
        "exact_newton_linear_residual_rel": exact_newton_linear_residual_rel,
        "exact_refinement_correction_rel": exact_refinement_correction_rel,
        **report,
    }


def _make_traceable_exact_newton_runner(
    residual_fn,
    maxiter,
    tol,
):
    cache_key = (int(maxiter), float(tol))
    return _cached_traceable_runner(
        _TRACEABLE_EXACT_NEWTON_RUNNER_CACHE,
        residual_fn,
        cache_key,
        lambda residual_fn_ref: _build_traceable_exact_newton_runner(
            residual_fn_ref,
            int(maxiter),
            float(tol),
        ),
    )


def _build_traceable_exact_newton_runner(
    residual_fn_ref,
    maxiter,
    tol,
):
    def run_solver(x_init, fn_args):
        residual_fn = _lookup_traceable_runner_callable(
            residual_fn_ref,
            "exact Newton residual",
        )

        # The exact Newton body differentiates the Boozer residual through
        # repeated GMRES matvecs. Rematerializing the residual keeps the JVP
        # intermediates out of the compiled loop's live set; this trades a
        # bounded amount of recomputation for materially lower compile/RSS
        # pressure on GPU backends.
        residual_eval = jax.checkpoint(
            jax.jit(lambda x: residual_fn(x, *fn_args)),
            policy=jax.checkpoint_policies.nothing_saveable,
            prevent_cse=False,
        )

        def jvp_fn(x, v):
            return jax.jvp(residual_eval, (x,), (v,))[1]

        dtype = jnp.asarray(x_init).dtype
        tol_value = _optimizer_scalar(tol, dtype=dtype)
        r0 = residual_eval(x_init)
        norm0 = jnp.linalg.norm(r0)

        def cond_fun(state):
            return (
                (state["nit"] < maxiter)
                & (state["norm"] > tol_value)
                & (~state["stalled"])
            )

        def body_fun(state):
            strict_cap_tol = _eisenstat_walker_strict_cap(
                tol_value,
                dtype=state["x"].dtype,
            )
            eisenstat_walker_tol = _eisenstat_walker_choice2_tolerance(
                state["norm"],
                state["previous_norm"],
                tol=tol_value,
            )
            linear_tol_iteration = jnp.where(
                state["retry_linear_solve_at_strict_cap"],
                jnp.minimum(eisenstat_walker_tol, strict_cap_tol),
                eisenstat_walker_tol,
            )
            dx, linear_residual, _ = _gmres_solve_exact_newton_system(
                jvp_fn,
                state["x"],
                state["residual"],
                tol=linear_tol_iteration,
            )
            linear_residual_norm = jnp.linalg.norm(linear_residual)
            linear_residual_rel = _relative_residual_norm(
                linear_residual,
                state["residual"],
            )

            def add_correction(current_dx):
                correction, _, _ = _gmres_solve_exact_newton_system(
                    jvp_fn,
                    state["x"],
                    linear_residual,
                    tol=linear_tol_iteration,
                )
                correction_rel = jnp.linalg.norm(correction) / jnp.maximum(
                    jnp.linalg.norm(current_dx),
                    _device_scalar(
                        jnp.finfo(current_dx.dtype).tiny,
                        dtype=current_dx.dtype,
                    ),
                )
                correction_finite = jnp.all(jnp.isfinite(correction))
                return (
                    lax.cond(
                        correction_finite,
                        lambda corr: current_dx + corr,
                        lambda _corr: current_dx,
                        correction,
                    ),
                    lax.select(
                        correction_finite,
                        correction_rel,
                        _device_scalar(jnp.nan, dtype=current_dx.dtype),
                    ),
                )

            dx, correction_rel = lax.cond(
                jnp.all(jnp.isfinite(dx))
                & (linear_residual_norm > linear_tol_iteration),
                add_correction,
                lambda current_dx: (
                    current_dx,
                    _device_scalar(0.0, dtype=current_dx.dtype),
                ),
                dx,
            )
            candidate = _backtracking_residual_step(
                residual_eval,
                state["x"],
                dx,
                state["residual"],
                state["norm"],
            )
            accepted = candidate["accepted"]
            loose_finite_direction = (linear_tol_iteration > strict_cap_tol) & jnp.all(
                jnp.isfinite(dx)
            )
            retry_at_strict_cap = (~accepted) & loose_finite_direction
            stalled = (~accepted) & (~loose_finite_direction)
            return {
                "x": lax.select(accepted, candidate["x"], state["x"]),
                "residual": lax.select(
                    accepted,
                    candidate["residual"],
                    state["residual"],
                ),
                "norm": lax.select(accepted, candidate["norm"], state["norm"]),
                "previous_norm": lax.select(
                    accepted,
                    state["norm"],
                    state["previous_norm"],
                ),
                "nit": lax.select(accepted, state["nit"] + 1, state["nit"]),
                "stalled": stalled,
                "retry_linear_solve_at_strict_cap": retry_at_strict_cap,
                "exact_newton_linear_residual_rel": lax.select(
                    accepted,
                    linear_residual_rel,
                    state["exact_newton_linear_residual_rel"],
                ),
                "exact_refinement_correction_rel": lax.select(
                    accepted,
                    correction_rel,
                    state["exact_refinement_correction_rel"],
                ),
            }

        state = lax.while_loop(
            cond_fun,
            body_fun,
            {
                "x": x_init,
                "residual": r0,
                "norm": norm0,
                "previous_norm": norm0,
                "nit": jnp.asarray(0, dtype=jnp.int32),
                "stalled": jnp.asarray(False),
                "retry_linear_solve_at_strict_cap": jnp.asarray(False),
                "exact_newton_linear_residual_rel": jnp.asarray(
                    jnp.nan,
                    dtype=dtype,
                ),
                "exact_refinement_correction_rel": jnp.asarray(
                    jnp.nan,
                    dtype=dtype,
                ),
            },
        )
        return {
            "x": state["x"],
            "residual": state["residual"],
            "nit": state["nit"],
            "success": state["norm"] <= tol_value,
            "exact_newton_linear_residual_rel": state[
                "exact_newton_linear_residual_rel"
            ],
            "exact_refinement_correction_rel": state["exact_refinement_correction_rel"],
        }

    run_solver.__name__ = "traceable_exact_newton_run_solver"
    return jax.jit(run_solver)


def newton_exact_traceable(
    residual_fn,
    x0,
    *,
    maxiter=40,
    tol=1e-13,
    args=(),
):
    """Trace-safe Newton solver for the exact Boozer residual system.

    The loop keeps Jacobian application matrix-free via JVPs and does not
    materialize dense Jacobians. Public dense metadata belongs to
    ``newton_exact(...)`` / ``BoozerSurfaceJAX.run_code()``.
    """
    normalized_args = _normalize_solver_args(args)
    runner = _make_traceable_exact_newton_runner(
        residual_fn,
        int(maxiter),
        float(tol),
    )
    result = runner(x0, normalized_args)
    result["jacobian"] = None
    result["jacobian_materialized"] = False
    result["failure_category"] = None
    result["failure_stage"] = None
    result["message"] = None
    return result


# ---------------------------------------------------------------------------
# Dispatcher — shared hub for all optimizer methods
# ---------------------------------------------------------------------------


def reference_least_squares(
    residual_fn,
    x0,
    *,
    method="lm",
    tol=1e-10,
    maxiter=1500,
    options=None,
    callback=None,
    progress_callback=None,
):
    """Explicit CPU/reference least-squares entrypoint."""
    return optimizer_jax_reference.reference_least_squares(
        residual_fn,
        x0,
        method=method,
        tol=tol,
        maxiter=maxiter,
        options=options,
        callback=callback,
        progress_callback=progress_callback,
    )


def _least_squares_state_to_optimize_result(result):
    nit = int(_host_scalar(result["nit"], dtype=np.int64))
    status = int(_host_scalar(result["status"], dtype=np.int64))
    info = int(_host_scalar(result["info"], dtype=np.int64))
    success = _host_bool(result["success"])
    return OptimizeResult(
        x=result["x"],
        fun=result["fun"],
        jac=result["grad"],
        residual=result["residual"],
        residual_jacobian=result["residual_jacobian"],
        hessian=result["hessian"],
        damping=result["damping"],
        nit=nit,
        nfev=nit + 1,
        njev=nit + 1,
        status=status,
        info=info,
        success=success,
        message=_least_squares_result_message(
            status,
            success,
            info=info,
        ),
        dense_linearization_materialized=result["dense_linearization_materialized"],
        dense_linearization_kind=result.get("dense_linearization_kind"),
        dense_residual_jacobian_shape=result.get("dense_residual_jacobian_shape"),
        dense_residual_jacobian_bytes=result.get("dense_residual_jacobian_bytes"),
        dense_hessian_shape=result.get("dense_hessian_shape"),
        dense_hessian_bytes=result.get("dense_hessian_bytes"),
        dense_linearization_bytes=result.get("dense_linearization_bytes"),
        max_dense_linearization_bytes=result.get("max_dense_linearization_bytes"),
        failure_category=result.get("failure_category"),
        failure_stage=result.get("failure_stage"),
        optimistix_result=result.get("optimistix_result"),
        optimistix_result_message=result.get("optimistix_result_message"),
    )


def host_jax_least_squares(
    residual_fn,
    x0,
    *,
    method="lm",
    tol=1e-10,
    maxiter=1500,
    options=None,
    callback=None,
    progress_callback=None,
    args=(),
    state_fn=None,
    jacobian_block_fn=None,
    jacobian_chunk_size=32,
):
    """Host LM control over a compiled JAX residual evaluator."""
    if method != "lm":
        raise ValueError(
            f"host_jax_least_squares() only supports method='lm'. Got {method!r}."
        )
    options = dict(options or {})
    if callback is not None:
        options["callback"] = callback
    if progress_callback is not None:
        options["progress_callback"] = progress_callback
    require_boozer_inner_backend_x64(HOST_JAX_BOOZER_OPTIMIZER_BACKEND)
    if jacobian_block_fn is not None:
        return _least_squares_state_to_optimize_result(
            levenberg_marquardt_block_jacobian(
                residual_fn,
                jacobian_block_fn,
                x0,
                maxiter=maxiter,
                tol=tol,
                ftol=options.get("ftol", 1e-8),
                xtol=options.get("xtol", 1e-8),
                gtol=options.get("gtol"),
                materialize_dense_linearization=bool(
                    options.get("materialize_dense_linearization", True)
                ),
                max_dense_linearization_bytes=options.get(
                    "max_dense_linearization_bytes"
                ),
                callback=options.get("callback"),
                progress_callback=options.get("progress_callback"),
                args=args,
                jacobian_chunk_size=jacobian_chunk_size,
            )
        )
    if state_fn is not None:
        return _least_squares_state_to_optimize_result(
            levenberg_marquardt_dense_state(
                state_fn,
                x0,
                maxiter=maxiter,
                tol=tol,
                ftol=options.get("ftol", 1e-8),
                xtol=options.get("xtol", 1e-8),
                gtol=options.get("gtol"),
                materialize_dense_linearization=bool(
                    options.get("materialize_dense_linearization", True)
                ),
                max_dense_linearization_bytes=options.get(
                    "max_dense_linearization_bytes"
                ),
                callback=options.get("callback"),
                progress_callback=options.get("progress_callback"),
                args=args,
            )
        )
    return _least_squares_state_to_optimize_result(
        levenberg_marquardt(
            residual_fn,
            x0,
            maxiter=maxiter,
            tol=tol,
            ftol=options.get("ftol", 1e-8),
            xtol=options.get("xtol", 1e-8),
            gtol=options.get("gtol"),
            materialize_dense_linearization=bool(
                options.get("materialize_dense_linearization", True)
            ),
            max_dense_linearization_bytes=options.get("max_dense_linearization_bytes"),
            callback=options.get("callback"),
            progress_callback=options.get("progress_callback"),
        )
    )


def target_least_squares(
    residual_fn,
    x0,
    *,
    method="lm-ondevice",
    tol=1e-10,
    maxiter=1500,
    options=None,
    callback=None,
    progress_callback=None,
    args=(),
):
    """Explicit JAX target least-squares entrypoint."""
    if method not in _TARGET_LEAST_SQUARES_METHODS:
        raise ValueError(
            "target_least_squares() only supports method='lm-ondevice', "
            "method='lm-minpack-ondevice', or method='optimistix-lm-ondevice'. "
            f"Got {method!r}."
        )

    options = dict(options or {})
    if callback is not None:
        options["callback"] = callback
    if progress_callback is not None:
        options["progress_callback"] = progress_callback

    require_target_backend_x64("ondevice")
    ftol = options.get("ftol", _DEFAULT_LM_FTOL)
    xtol = options.get("xtol", _DEFAULT_LM_XTOL)
    materialize_dense_linearization = bool(
        options.get("materialize_dense_linearization", True)
    )
    max_dense_linearization_bytes = options.get("max_dense_linearization_bytes")
    callback = options.get("callback")
    progress_callback = options.get("progress_callback")
    if method == "optimistix-lm-ondevice":
        result = jax_least_squares_optimistix(
            residual_fn,
            x0,
            maxiter=maxiter,
            tol=tol,
            ftol=ftol,
            xtol=xtol,
            gtol=options.get("gtol"),
            materialize_dense_linearization=materialize_dense_linearization,
            max_dense_linearization_bytes=max_dense_linearization_bytes,
            callback=callback,
            progress_callback=progress_callback,
            args=args,
        )
        return _least_squares_state_to_optimize_result(result)

    solver = (
        levenberg_marquardt_minpack_traceable
        if method == "lm-minpack-ondevice"
        else levenberg_marquardt_traceable
    )
    gtol = options.get("gtol")
    if method == "lm-minpack-ondevice" and gtol is None:
        gtol = 1e-8
    result = solver(
        residual_fn,
        x0,
        maxiter=maxiter,
        tol=tol,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        materialize_dense_linearization=materialize_dense_linearization,
        max_dense_linearization_bytes=max_dense_linearization_bytes,
        callback=callback,
        progress_callback=progress_callback,
        args=args,
    )

    return _least_squares_state_to_optimize_result(result)


def _jax_least_squares_legacy(
    residual_fn,
    x0,
    *,
    method="lm",
    tol=1e-10,
    maxiter=1500,
    options=None,
    callback=None,
    progress_callback=None,
):
    """Compatibility least-squares entrypoint that dispatches by lane."""
    if method not in _SUPPORTED_LEAST_SQUARES_METHODS:
        raise ValueError(
            "Unknown least-squares method "
            f"{method!r}. Supported: {sorted(_SUPPORTED_LEAST_SQUARES_METHODS)}."
        )
    if method == "lm":
        _raise_if_target_lane_required(
            component="optimizer_jax.jax_least_squares",
            method=method,
            detail=_STRICT_REFERENCE_LEAST_SQUARES_DETAIL,
        )
    if method == "lm":
        return reference_least_squares(
            residual_fn,
            x0,
            method=method,
            tol=tol,
            maxiter=maxiter,
            options=options,
            callback=callback,
            progress_callback=progress_callback,
        )
    return target_least_squares(
        residual_fn,
        x0,
        method=method,
        tol=tol,
        maxiter=maxiter,
        options=options,
        callback=callback,
        progress_callback=progress_callback,
    )


def reference_minimize(
    fun,
    x0,
    *,
    method="bfgs",
    tol=1e-10,
    maxiter=1500,
    options=None,
    value_and_grad=False,
    callback=None,
    progress_callback=None,
    failure_callback=None,
    initial_value_and_grad=None,
    allow_jax_host_control=False,
):
    """Explicit CPU/reference scalar optimizer entrypoint."""
    if failure_callback is not None and method not in _REFERENCE_TRACE_METHODS:
        raise ValueError(
            "reference_minimize() only supports failure_callback for "
            "method='lbfgs-trace'."
        )
    if initial_value_and_grad is not None and (
        method not in _REFERENCE_TRACE_METHODS or not value_and_grad
    ):
        raise ValueError(
            "reference_minimize() only supports initial_value_and_grad for "
            "explicit value-and-gradient objectives with method='lbfgs-trace'."
        )
    if method in _REFERENCE_JAX_METHODS:
        _raise_if_target_lane_required(
            component="optimizer_jax.reference_minimize",
            method=method,
            detail=_STRICT_REFERENCE_JAX_OPTIMIZER_DETAIL,
        )
        _raise_if_strict_optimizer_fallback(
            component="optimizer_jax.reference_minimize",
            method=method,
            detail=_STRICT_REFERENCE_JAX_OPTIMIZER_DETAIL,
        )
        result = adam_optimize(
            fun,
            x0,
            value_and_grad=value_and_grad,
            maxiter=maxiter,
            tol=tol,
            options=options,
            callback=callback,
            progress_callback=progress_callback,
        )
        return _adam_result_to_optimize_result(result)
    return optimizer_jax_reference.reference_minimize(
        fun,
        x0,
        method=method,
        tol=tol,
        maxiter=maxiter,
        options=options,
        value_and_grad=value_and_grad,
        callback=callback,
        progress_callback=progress_callback,
        failure_callback=failure_callback,
        initial_value_and_grad=initial_value_and_grad,
        allow_jax_host_control=allow_jax_host_control,
    )


def host_jax_minimize_value_and_grad(
    fun,
    x0,
    *,
    method="bfgs",
    tol=1e-10,
    maxiter=1500,
    options=None,
    value_and_grad=True,
    callback=None,
    progress_callback=None,
):
    """Host SciPy control over a compiled JAX value/gradient evaluator."""
    if method not in {"bfgs", "lbfgs"}:
        raise ValueError(
            "host_jax_minimize_value_and_grad() only supports "
            "method='bfgs' or method='lbfgs'."
        )
    if not value_and_grad:
        raise ValueError(
            "host_jax_minimize_value_and_grad() requires value_and_grad=True."
        )
    options = dict(options or {})
    fun = wrap_strict_target_lane_value_and_grad(fun)
    fun, x0, callback, pytree_adapter = _prepare_optimizer_callable_inputs(
        fun,
        x0,
        value_and_grad=True,
        callback=callback,
    )
    if callback is not None:
        options["callback"] = callback
    if progress_callback is not None:
        options["progress_callback"] = progress_callback
    require_boozer_inner_backend_x64(HOST_JAX_BOOZER_OPTIMIZER_BACKEND)
    result = optimizer_jax_reference.target_scipy_minimize_value_and_grad(
        fun,
        x0,
        method=method,
        tol=tol,
        maxiter=maxiter,
        options=options,
    )
    return _finalize_optimizer_result(result, pytree_adapter)


def target_minimize(
    fun,
    x0,
    *,
    method="bfgs-ondevice",
    tol=1e-10,
    maxiter=1500,
    options=None,
    value_and_grad=False,
    callback=None,
    progress_callback=None,
    failure_callback=None,
    initial_value_and_grad=None,
):
    """Explicit JAX target scalar optimizer entrypoint."""
    options = dict(options or {})
    if failure_callback is not None:
        raise ValueError(
            "target_minimize() does not support failure_callback. "
            "Use reference_minimize(method='lbfgs-trace') for host-side "
            "L-BFGS rejection diagnostics."
        )
    if initial_value_and_grad is not None and (
        method != "lbfgs-ondevice" or not value_and_grad
    ):
        raise ValueError(
            "target_minimize() only supports initial_value_and_grad for "
            "explicit value-and-gradient objectives with method='lbfgs-ondevice'."
        )
    if method in _TARGET_LBFGSB_METHODS:
        unsupported_options = _UNSUPPORTED_TARGET_LBFGSB_OPTIONS.intersection(options)
        if unsupported_options:
            raise ValueError(
                "target L-BFGS-B methods follow SciPy L-BFGS-B options and do "
                f"not support {sorted(unsupported_options)}."
            )
    if method in _TARGET_SCIPY_CONTROL_METHODS:
        if not value_and_grad:
            raise RuntimeError(
                f"target_minimize() requires value_and_grad=True for method={method!r}."
            )
        fun = wrap_strict_target_lane_value_and_grad(fun)
        fun, x0, callback, pytree_adapter = _prepare_optimizer_callable_inputs(
            fun,
            x0,
            value_and_grad=True,
            callback=callback,
        )
        if callback is not None:
            options["callback"] = callback
        if progress_callback is not None:
            options["progress_callback"] = progress_callback
        required_backend = (
            "scipy-jax-fullgraph"
            if method == "lbfgs-scipy-jax-fullgraph"
            else "scipy-jax-decomposed"
            if method == "lbfgs-scipy-jax-decomposed"
            else "scipy-jax"
        )
        require_target_backend_x64(required_backend)
        result = optimizer_jax_reference.target_scipy_minimize_value_and_grad(
            fun,
            x0,
            method="lbfgs",
            tol=tol,
            maxiter=maxiter,
            options=options,
        )
        return _finalize_optimizer_result(result, pytree_adapter)
    if method in _TARGET_PUBLIC_METHODS:
        if method == "adam-ondevice":
            require_target_backend_x64("ondevice")
            result = adam_optimize_traceable(
                fun,
                x0,
                value_and_grad=value_and_grad,
                maxiter=maxiter,
                tol=tol,
                options=options,
                callback=callback,
                progress_callback=progress_callback,
            )
            return _adam_result_to_optimize_result(result)

        required_backend = _TARGET_PUBLIC_LBFGS_BACKEND_BY_METHOD[method]
        require_target_backend_x64(required_backend)
        fun, x0, _unused_callback, pytree_adapter = _prepare_optimizer_callable_inputs(
            fun,
            x0,
            value_and_grad=value_and_grad,
            callback=None,
        )
        if value_and_grad:
            value_and_grad_fun = wrap_strict_target_lane_value_and_grad(fun)
        else:
            scalar_value_and_grad = jax.value_and_grad(fun)

            def value_and_grad_fun(flat_x):
                return scalar_value_and_grad(flat_x)

            value_and_grad_fun = wrap_strict_target_lane_value_and_grad(
                value_and_grad_fun
            )

        if callback is None and progress_callback is None:
            public_callback = None
        else:

            def public_callback(event):
                if callback is not None:
                    callback(
                        event.x
                        if pytree_adapter is None
                        else pytree_adapter._hostify_flat(
                            event.x,
                            dtype=pytree_adapter.flat_dtype,
                        )
                    )
                if progress_callback is not None:
                    progress_callback(
                        event.iteration,
                        event.fun,
                        event.grad_norm_inf,
                    )

        if method == "optax-lbfgs-ondevice":
            public_options = OptaxLBFGSOptions(
                maxiter=maxiter,
                gtol=tol,
                memory_size=int(options.get("maxcor", OptaxLBFGSOptions().memory_size)),
                scale_init_precond=bool(options.get("scale_init_precond", True)),
                max_linesearch_steps=int(options.get("maxls", 20)),
            )
            driver = Driver.OPTAX_LBFGS
        else:
            public_options = OptimistixLBFGSOptions(
                maxiter=maxiter,
                tol=tol,
                history_length=int(options.get("maxcor", 200)),
            )
            driver = Driver.OPTIMISTIX_LBFGS
        result = (
            run_optax_minimize(
                value_and_grad_fun,
                x0,
                driver=driver,
                options=public_options,
                callback=public_callback,
            )
            if method == "optax-lbfgs-ondevice"
            else run_optimistix_minimize(
                value_and_grad_fun,
                x0,
                options=public_options,
                callback=public_callback,
            )
        )
        return _finalize_optimizer_result(result, pytree_adapter)

    if method not in _TARGET_PRIVATE_METHODS:
        raise ValueError(
            "target_minimize() only supports target-lane methods "
            f"{sorted(_TARGET_METHODS)}. Got {method!r}."
        )

    pytree_adapter = _prepare_optimizer_pytree_adapter(x0)

    def finalize(result):
        return _finalize_optimizer_result(result, pytree_adapter)

    if callback is not None:
        options["callback"] = callback
    if progress_callback is not None:
        options["progress_callback"] = progress_callback

    require_target_backend_x64("ondevice")

    diagnostic_event_callback = _target_optimizer_diagnostic_event_callback()
    if method == "lbfgs-ondevice":
        lbfgs_ftol = float(options.get("ftol", tol))

    if value_and_grad:
        if method == "bfgs-ondevice":
            fun = wrap_strict_target_lane_value_and_grad(fun)
            state = _minimize_bfgs_private(
                fun,
                x0,
                maxiter=maxiter,
                gtol=tol,
                xrtol=float(options.get("xrtol", 0.0)),
                line_search_maxiter=int(options.get("line_search_maxiter", 10)),
                callback=options.get("callback"),
                progress_callback=options.get("progress_callback"),
                value_and_grad=True,
            )
            return finalize(_private_bfgs_result_to_optimize_result(state))
        if method != "lbfgs-ondevice":
            raise RuntimeError(
                "Explicit value-and-gradient objectives are only supported on the "
                "trusted SciPy reference methods, bfgs-ondevice, and "
                "lbfgs-ondevice today."
            )
        fun = wrap_strict_target_lane_value_and_grad(fun)
        state = _minimize_lbfgs_private_value_and_grad(
            fun,
            x0,
            maxiter=maxiter,
            gtol=tol,
            maxcor=int(options.get("maxcor", 10)),
            ftol=lbfgs_ftol,
            maxfun=options.get("maxfun"),
            maxls=int(options.get("maxls", 20)),
            callback=options.get("callback"),
            progress_callback=options.get("progress_callback"),
            initial_value_and_grad=initial_value_and_grad,
            record_optimizer_state_trace=bool(
                options.get("record_optimizer_state_trace", False)
            ),
            max_optimizer_state_trace_bytes=options.get(
                "max_optimizer_state_trace_bytes"
            ),
            diagnostic_event_callback=diagnostic_event_callback,
            run_mode=options.get("lbfgs_run_mode", "stepwise"),
        )
        _record_target_optimizer_diagnostic_event(
            diagnostic_event_callback,
            "lbfgs_result_conversion_started",
        )
        result = _private_lbfgs_result_to_optimize_result(state)
        _record_target_optimizer_diagnostic_event(
            diagnostic_event_callback,
            "lbfgs_result_conversion_returned",
        )
        return finalize(result)

    if method == "bfgs-ondevice":
        state = _minimize_bfgs_private(
            fun,
            x0,
            maxiter=maxiter,
            gtol=tol,
            xrtol=float(options.get("xrtol", 0.0)),
            line_search_maxiter=int(options.get("line_search_maxiter", 10)),
            callback=options.get("callback"),
            progress_callback=options.get("progress_callback"),
        )
        return finalize(_private_bfgs_result_to_optimize_result(state))

    if method == "lbfgs-ondevice":
        state = _minimize_lbfgs_private(
            fun,
            x0,
            maxiter=maxiter,
            gtol=tol,
            maxcor=int(options.get("maxcor", 10)),
            ftol=lbfgs_ftol,
            maxfun=options.get("maxfun"),
            maxls=int(options.get("maxls", 20)),
            callback=options.get("callback"),
            progress_callback=options.get("progress_callback"),
            record_optimizer_state_trace=bool(
                options.get("record_optimizer_state_trace", False)
            ),
            max_optimizer_state_trace_bytes=options.get(
                "max_optimizer_state_trace_bytes"
            ),
            diagnostic_event_callback=diagnostic_event_callback,
            run_mode=options.get("lbfgs_run_mode", "stepwise"),
        )
        _record_target_optimizer_diagnostic_event(
            diagnostic_event_callback,
            "lbfgs_result_conversion_started",
        )
        result = _private_lbfgs_result_to_optimize_result(state)
        _record_target_optimizer_diagnostic_event(
            diagnostic_event_callback,
            "lbfgs_result_conversion_returned",
        )
        return finalize(result)
    raise ValueError(f"Unknown target optimizer method {method!r}.")


def _jax_minimize_legacy(
    fun,
    x0,
    *,
    method="bfgs",
    tol=1e-10,
    maxiter=1500,
    options=None,
    value_and_grad=False,
    callback=None,
    progress_callback=None,
):
    """Compatibility scalar optimizer entrypoint that dispatches by lane."""
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Unknown method {method!r}. Supported: {sorted(_SUPPORTED_METHODS)}."
        )

    if method in _REFERENCE_METHODS | _REFERENCE_TRACE_METHODS | _REFERENCE_JAX_METHODS:
        detail = (
            _STRICT_REFERENCE_JAX_OPTIMIZER_DETAIL
            if method in _REFERENCE_JAX_METHODS
            else _STRICT_REFERENCE_OPTIMIZER_DETAIL
        )
        _raise_if_target_lane_required(
            component="optimizer_jax.jax_minimize",
            method=method,
            detail=detail,
        )
        return reference_minimize(
            fun,
            x0,
            method=method,
            tol=tol,
            maxiter=maxiter,
            options=options,
            value_and_grad=value_and_grad,
            callback=callback,
            progress_callback=progress_callback,
        )
    return target_minimize(
        fun,
        x0,
        method=method,
        tol=tol,
        maxiter=maxiter,
        options=options,
        value_and_grad=value_and_grad,
        callback=callback,
        progress_callback=progress_callback,
    )


def jax_least_squares(
    residual_fn,
    x0,
    *,
    method="lm",
    tol=1e-10,
    maxiter=1500,
    options=None,
    callback=None,
    progress_callback=None,
):
    """Deprecated compatibility least-squares entrypoint."""
    if method not in _SUPPORTED_LEAST_SQUARES_METHODS:
        raise ValueError(
            "Unknown least-squares method "
            f"{method!r}. Supported: {sorted(_SUPPORTED_LEAST_SQUARES_METHODS)}."
        )
    _warn_deprecated_solve_jax_call(
        api="jax_least_squares",
        method=method,
        translated_driver=_DEPRECATED_LEAST_SQUARES_METHOD_TO_DRIVER[method],
        caller_frame=sys._getframe(1),
    )
    return _jax_least_squares_legacy(
        residual_fn,
        x0,
        method=method,
        tol=tol,
        maxiter=maxiter,
        options=options,
        callback=callback,
        progress_callback=progress_callback,
    )


def jax_minimize(
    fun,
    x0,
    *,
    method="bfgs",
    tol=1e-10,
    maxiter=1500,
    options=None,
    value_and_grad=False,
    callback=None,
    progress_callback=None,
):
    """Deprecated compatibility scalar optimizer entrypoint."""
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Unknown method {method!r}. Supported: {sorted(_SUPPORTED_METHODS)}."
        )
    _warn_deprecated_solve_jax_call(
        api="jax_minimize",
        method=method,
        translated_driver=_DEPRECATED_MINIMIZE_METHOD_TO_DRIVER[method],
        caller_frame=sys._getframe(1),
    )
    return _jax_minimize_legacy(
        fun,
        x0,
        method=method,
        tol=tol,
        maxiter=maxiter,
        options=options,
        value_and_grad=value_and_grad,
        callback=callback,
        progress_callback=progress_callback,
    )


from . import reference as optimizer_jax_reference  # noqa: E402
