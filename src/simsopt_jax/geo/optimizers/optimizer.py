"""
JAX optimizer adapter for the Boozer inner solve.

Reference/oracle methods:
  - ``method="bfgs"``: host-driven SciPy BFGS loop with JAX value/grad.
  - ``method="lbfgs"``: host-driven SciPy L-BFGS-B loop with JAX value/grad.

Least-squares method:
  - ``method="lm-minpack-ondevice"``: trace-safe dense pivoted-QR
    Levenberg-Marquardt for residual-vector objectives on the target lane,
    ported from MINPACK ``lmder``/``lmpar`` (the algorithm behind
    ``scipy.optimize.least_squares(method="lm")``, here with
    ``x_scale=1.0``).  It takes MINPACK's steps and stops on MINPACK's tests;
    dense Householder QR stands in for MINPACK's packed QR and Givens sweeps,
    so iterates agree to rounding, not bytewise.

Target private methods (maintained for the pinned JAX 0.10.0 runtime after the
initial port from the upstream JAX optimizer sources):
  - ``method="bfgs-ondevice"``: JAX on-device BFGS.
  - ``method="lbfgs-ondevice"``: in-tree SciPy-compatible L-BFGS-B state
    machine on the target lane. It uses a host stepwise loop over explicit
    macro-step observables rather than reverse-communication task reads,
    specializes the public no-bounds lane to a two-loop L-BFGS
    direction, and preserves SciPy-style counters, statuses, callbacks, and
    inverse-Hessian history. The full L-BFGS-B compact subspace path remains
    available for bounded states and generic private kernels.

Target SciPy-control method:
  - ``method="lbfgs-scipy-jax"``: host SciPy L-BFGS-B control with JAX
    target-lane value/grad evaluations.
  - ``method="lbfgs-scipy-jax-fullgraph"``: host SciPy L-BFGS-B control with
    JAX value/grad evaluations over a caller-owned full Optimizable graph.

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
    Literal,
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
from jax import lax
from jax.flatten_util import ravel_pytree
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
    TARGET_SCIPY_CONTROL_OPTIMIZER_BACKENDS,
    VALID_OPTIMIZER_BACKENDS,
    VALID_OUTER_OPTIMIZER_BACKENDS,
    render_invalid_optimizer_backend_message,
)
from simsopt_jax.geo.optimizers._shared import (
    _CACHEABLE_VALUE_AND_GRAD_ATTR,
    PRIVATE_OPTIMIZER_JAX_VERSION,
    _hostify_optimizer_tree,
    _is_flat_optimizer_vector,
    _optimizer_scalar,
    _prepare_optimizer_callable_inputs,
    _prepare_optimizer_pytree_adapter,
    _x64_enabled,
    private_optimizer_runtime_is_supported,
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
    _DenseJacobianAssembler as _DenseJacobianAssembler,
    _DenseJacobianMaterialization as _DenseJacobianMaterialization,
    _DenseJacobianMaterializationTelemetry as _DenseJacobianMaterializationTelemetry,
    _LinearSolveStatus as _LinearSolveStatus,
    _SQUARE_OPERATOR_GMRES_REFINEMENT_STEPS as _SQUARE_OPERATOR_GMRES_REFINEMENT_STEPS,
    _apply_column_batched_operator as _apply_column_batched_operator,
    _cached_jit_linear_operator as _cached_jit_linear_operator,
    _combine_linear_solve_iteration_counts as _combine_linear_solve_iteration_counts,
    _complete_linear_solve_status as _complete_linear_solve_status,
    _dense_linear_solve_status as _dense_linear_solve_status,
    _dense_matrix_backward_error_success as _dense_matrix_backward_error_success,
    _dense_matrix_condition_estimate as _dense_matrix_condition_estimate,
    _dense_matrix_condition_estimate_with_telemetry as _dense_matrix_condition_estimate_with_telemetry,
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
    _exact_newton_gmres_iteration_limits as _exact_newton_gmres_iteration_limits,
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
    _linearize_and_materialize_dense_square_jacobian as _linearize_and_materialize_dense_square_jacobian,
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
    _terminal_linear_solve_status as _terminal_linear_solve_status,
)
from simsopt_jax.geo.optimizers.adjoint_linear_solve import (
    _EXACT_JACOBIAN_OPERATOR_GMRES_REFINEMENT_STEPS as _EXACT_JACOBIAN_OPERATOR_GMRES_REFINEMENT_STEPS,
    _hessian_linear_operator as _hessian_linear_operator,
    _require_tree_first_leaf as _require_tree_first_leaf,
    _solve_hessian_least_squares_system_with_status as _solve_hessian_least_squares_system_with_status,
    _solve_hessian_system as _solve_hessian_system,
    _solve_hessian_system_with_status as _solve_hessian_system_with_status,
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
    NEWTON_ARMIJO_C1,
    mixed_dense_ir_accuracy_policy,
)
from simsopt_jax.runtime.host_boundary import (
    host_bool as _host_bool,
    host_scalar as _host_scalar,
)
from simsopt_jax.solve.driver import (
    Driver,
    legacy_reference_least_squares_method,
    legacy_reference_minimize_method,
    legacy_target_method,
    legacy_target_scipy_control_method,
)

# ---------------------------------------------------------------------------
# Linear-solve, dense-IR, and adjoint-solve implementations live in their
# dedicated owner modules. The explicit aliases above are compatibility
# reexports only; mutable selectors must be read and patched on their owner.
# ---------------------------------------------------------------------------
# Explicit ``import X as X`` aliases keep the declared compatibility surface
# visible and make the intentional reexports F401-safe for Ruff.


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
    "TraceableExactNewtonVariantContract",
    "TraceableNewtonLinearSolver",
    "adjoint_hessian_stabilization",
    "private_optimizer_runtime_is_supported",
    "VALID_LEAST_SQUARES_ALGORITHMS",
    "VALID_OPTIMIZER_BACKENDS",
    "VALID_OUTER_OPTIMIZER_BACKENDS",
    "CONCRETE_OPTIMIZER_BACKENDS",
    "BOOZER_INNER_OPTIMIZER_BACKENDS",
    "VALID_BOOZER_INNER_OPTIMIZER_BACKENDS",
    "TARGET_OUTER_OPTIMIZER_BACKENDS",
    "TARGET_SCIPY_CONTROL_OPTIMIZER_BACKENDS",
    "TARGET_X64_REQUIRED_OPTIMIZER_BACKENDS",
    "BOOZER_INNER_X64_REQUIRED_OPTIMIZER_BACKENDS",
    "render_invalid_optimizer_backend_message",
    "render_invalid_boozer_inner_optimizer_backend_message",
    "levenberg_marquardt_minpack_traceable",
    "make_traceable_exact_newton_variant_contract",
    "newton_polish",
    "newton_polish_traceable",
    "newton_exact",
    "newton_exact_traceable",
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
}
TARGET_X64_REQUIRED_OPTIMIZER_BACKENDS = TARGET_OUTER_OPTIMIZER_BACKENDS | frozenset(
    {HOST_JAX_OUTER_OPTIMIZER_BACKEND}
)
VALID_LEAST_SQUARES_ALGORITHMS = frozenset({"quasi-newton", "lm-minpack"})
_TARGET_LEAST_SQUARES_METHODS = frozenset({"lm-minpack-ondevice"})
_RESIDUAL_LEAST_SQUARES_ALGORITHMS = frozenset({"lm-minpack"})
_REFERENCE_METHODS = frozenset({"bfgs", "lbfgs"})
_TARGET_PRIVATE_METHODS = frozenset({"bfgs-ondevice", "lbfgs-ondevice"})
_TARGET_SCIPY_CONTROL_METHODS = frozenset(
    {"lbfgs-scipy-jax", "lbfgs-scipy-jax-decomposed", "lbfgs-scipy-jax-fullgraph"}
)
_TARGET_METHODS = _TARGET_PRIVATE_METHODS | _TARGET_SCIPY_CONTROL_METHODS
_TARGET_LBFGSB_METHODS = frozenset({"lbfgs-ondevice"}) | _TARGET_SCIPY_CONTROL_METHODS
_UNSUPPORTED_TARGET_LBFGSB_OPTIONS = frozenset({"initial_step_size", "maxgrad"})
_STRICT_REFERENCE_OPTIMIZER_DETAIL = "the host-side SciPy reference optimizer lane"
_STRICT_HOST_SCIPY_ADAPTER_DETAIL = "the host SciPy adapter"
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
_TRACEABLE_RUNNER_CACHE_LOCK = Lock()
# Explicit traceable cache tokens own semantic reuse; bare callables stay
# isolated by object identity because their closure state is not comparable.
_TRACEABLE_LM_QR_RUNNER_CACHE = {}
_TRACEABLE_NEWTON_POLISH_RUNNER_CACHE = {}
_TRACEABLE_EXACT_NEWTON_RUNNER_CACHE = {}
_TRACEABLE_DENSE_EXACT_NEWTON_C2_RUNNER_CACHE = {}
_TRACEABLE_NEWTON_LINEAR_SOLVER_OPERATOR_GMRES: TraceableNewtonLinearSolver = (
    "operator_gmres"
)
_TRACEABLE_NEWTON_LINEAR_SOLVER_DENSE_LU: TraceableNewtonLinearSolver = "dense_lu"
_TRACEABLE_NEWTON_LINEAR_SOLVER_HYBRID_FINAL_DENSE_LU: TraceableNewtonLinearSolver = (
    "hybrid_final_dense_lu"
)
_TRACEABLE_NEWTON_LINEAR_SOLVERS = frozenset(
    {
        _TRACEABLE_NEWTON_LINEAR_SOLVER_OPERATOR_GMRES,
        _TRACEABLE_NEWTON_LINEAR_SOLVER_DENSE_LU,
        _TRACEABLE_NEWTON_LINEAR_SOLVER_HYBRID_FINAL_DENSE_LU,
    }
)
_TRACEABLE_NEWTON_LINEAR_SOLVER_CODES = {
    _TRACEABLE_NEWTON_LINEAR_SOLVER_OPERATOR_GMRES: 1,
    _TRACEABLE_NEWTON_LINEAR_SOLVER_DENSE_LU: 2,
    _TRACEABLE_NEWTON_LINEAR_SOLVER_HYBRID_FINAL_DENSE_LU: 3,
}


def _resolve_traceable_newton_linear_solver(
    value: object,
) -> TraceableNewtonLinearSolver:
    """Validate one exact traceable Newton linear-solver selector."""
    if not isinstance(value, str) or value not in _TRACEABLE_NEWTON_LINEAR_SOLVERS:
        names = ", ".join(sorted(_TRACEABLE_NEWTON_LINEAR_SOLVERS))
        raise ValueError(f"linear_solver must be one of: {names}; got {value!r}.")
    return cast(TraceableNewtonLinearSolver, value)


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


class _ArrayValueAndJacobianWithArgs(Protocol):
    def __call__(self, x: jax.Array, *args: object) -> tuple[jax.Array, jax.Array]: ...


class _DenseExactNewtonDirection(NamedTuple):
    """Current-state dense linearization, factors, and certified direction."""

    residual: jax.Array
    jacobian: jax.Array
    lu: jax.Array
    pivots: jax.Array
    initial_solve: jax.Array
    refinement_rhs: jax.Array
    direction: jax.Array
    correction: jax.Array
    linear_residual: jax.Array
    condition_estimate: jax.Array
    status: _LinearSolveStatus


class _NativeDenseExactNewtonC2Result(NamedTuple):
    """Native-order C2 result with fixed-shape applied-state telemetry."""

    x: jax.Array
    residual: jax.Array
    returned_jacobian: jax.Array
    iteration_count: jax.Array
    applied_update_count: jax.Array
    success: jax.Array
    numerical_failure: jax.Array
    stop_reason_code: jax.Array
    rollback_branch_taken: jax.Array
    native_persist_predicate: jax.Array
    persist_solved_state: jax.Array
    initial_norm: jax.Array
    assessed_norm: jax.Array
    returned_norm: jax.Array
    applied_state_trace: jax.Array
    applied_state_trace_active: jax.Array
    assessed_norm_trace: jax.Array
    assessed_norm_trace_active: jax.Array
    exact_newton_linear_residual_rel: jax.Array
    exact_refinement_correction_rel: jax.Array
    linear_solve_attempt_count: jax.Array
    dense_materialization_count: jax.Array
    lu_factorization_count: jax.Array
    lu_solve_count: jax.Array
    refinement_correction_count: jax.Array
    rollback_recompute_count: jax.Array


_TraceableExactNewtonVariant = Literal["C0", "C2"]


class _TraceableExactNewtonSolver(Protocol):
    """Construction-time exact-Newton solver used by the Boozer adapter."""

    def __call__(
        self,
        residual_fn: Callable[..., jax.Array],
        x0: jax.Array,
        *,
        maxiter: int,
        tol: float,
        args: tuple[object, ...] = (),
    ) -> dict[str, object]: ...


@dataclass(frozen=True)
class TraceableExactNewtonVariantContract:
    """Immutable construction choice; no variant branch enters staged code.

    Public because the Boozer adapter selects and carries the variant it
    stages. Build one with
    :func:`make_traceable_exact_newton_variant_contract`, or construct
    directly when the caller stages a module-local solver seam (the Boozer
    adapter's monkeypatchable C0 path does).
    """

    variant: _TraceableExactNewtonVariant
    solver: _TraceableExactNewtonSolver
    factorization_backend: Literal["operator-gmres", "dense-lu"]
    returns_jacobian: bool


_C2_STOP_REASON_CONVERGED = np.int32(0)
_C2_STOP_REASON_MAXITER = np.int32(1)
_C2_STOP_REASON_NUMERICAL_FAILURE = np.int32(2)


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


def _traceable_callback_token_operand(token: int) -> jax.Array:
    """Stage a callback-registry token as a traced int32 solver operand.

    Tokens are minted per call, so a token declared ``static_argnums`` compiles
    one executable per instrumented solve — and, once the runner itself is
    memoized, retains every one of them. Passed as a traced operand the token
    only routes the host-side registry lookup in ``_lookup_traceable_callback``
    (which already accepts arrays), so the solver numerics are unchanged and
    every instrumented solve shares one executable.
    """
    return _device_scalar(token, dtype=jnp.int32)


def _invoke_traceable_lm_callback(token, x) -> None:
    callback = _lookup_traceable_callback(token, "LM step")
    callback(_hostify_optimizer_tree(x))


def _invoke_traceable_progress_callback(token, nit, fun, grad_norm) -> None:
    callback = _lookup_traceable_callback(token, "progress")
    callback(nit, fun, grad_norm)


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


_TARGET_LEAST_SQUARES_DRIVERS = frozenset({Driver.SIMSOPT_LM_QR})
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
    Driver.SIMSOPT_LM_QR: BoozerInnerDriverOptions(
        optimizer_backend="ondevice",
        limited_memory=False,
        least_squares_algorithm="lm-minpack",
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
        # The host-control lanes have no least-squares solver of their own:
        # their residual solves run the one Levenberg-Marquardt, the
        # on-device MINPACK-style lane.
        (
            HOST_JAX_BOOZER_OPTIMIZER_BACKEND,
            False,
            "lm-minpack",
        ): Driver.SIMSOPT_LM_QR,
        ("scipy", False, "lm-minpack"): Driver.SIMSOPT_LM_QR,
    }
)


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
    if (
        optimizer_backend in {"scipy", HOST_JAX_BOOZER_OPTIMIZER_BACKEND}
        and driver not in _TARGET_LEAST_SQUARES_DRIVERS
    ):
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
    _raise_if_target_lane_required(
        component=component,
        method=method,
        detail=_STRICT_HOST_SCIPY_ADAPTER_DETAIL,
    )


def _mark_cacheable_jit_linear_operator(fun):
    # Same callable mutability contract as ``mark_cacheable_jit_value_and_grad``.
    setattr(fun, _CACHEABLE_LINEAR_OPERATOR_ATTR, True)
    return fun


def _mark_traceable_runner_cacheable(fun, *, cache_token):
    # Same contract as ``mark_cacheable_jit_value_and_grad``.
    setattr(fun, _TRACEABLE_RUNNER_CACHE_TOKEN_ATTR, cache_token)
    return fun


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


_STRICT_TARGET_LANE_WRAPPER_LOCK = Lock()
_STRICT_TARGET_LANE_WRAPPER_ATTR = "_simsopt_strict_target_lane_wrapper"


def wrap_strict_target_lane_value_and_grad(fun):
    """Wrap target-lane value/grad calls in the stack-scoped purity guard.

    ``functools.wraps`` copies the cacheable marker onto the wrapper, so the
    wrapper — not ``fun`` — is what the solvers install their compiled-executable
    caches on. A cacheable ``fun`` therefore memoizes its one guard wrapper, or a
    per-solve wrapper would drop every compiled solver at the end of each solve.
    Unmarked callables get a fresh wrapper: ``_cached_private_solver`` never
    caches for them anyway, and only a marked callable is known to accept the
    ``setattr`` (same contract as ``mark_cacheable_jit_value_and_grad``).
    """
    if not target_lane_purity_requested():
        return fun
    if isinstance(fun, TargetScipyEvaluationProvider):
        return _StrictTargetScipyEvaluationProvider(provider=fun)
    cacheable = bool(getattr(fun, _CACHEABLE_VALUE_AND_GRAD_ATTR, False))
    if cacheable:
        memoized = _memoized_strict_target_lane_wrapper(fun)
        if memoized is not None:
            return memoized

    @wraps(fun)
    def wrapped(*args, **kwargs):
        with strict_target_lane_purity():
            return fun(*args, **kwargs)

    if not cacheable:
        return wrapped
    # Double-checked install under the wrapper lock. ``fun`` carries the
    # cacheable marker, and some in-repo callers write that marker with
    # ``object.__setattr__`` because plain ``setattr`` is overridden on their
    # callables — use the same spelling so every marked callable is writable.
    with _STRICT_TARGET_LANE_WRAPPER_LOCK:
        memoized = _memoized_strict_target_lane_wrapper(fun)
        if memoized is not None:
            return memoized
        object.__setattr__(fun, _STRICT_TARGET_LANE_WRAPPER_ATTR, wrapped)
        return wrapped


def _memoized_strict_target_lane_wrapper(fun):
    """Return ``fun``'s own memoized guard wrapper, or ``None``.

    ``functools.wraps`` copies the attribute dictionary, so a wrapper of a
    *different* callable can arrive here through ``fun.__dict__``; reusing it
    would run the wrong objective. ``__wrapped__`` identity is the discriminator.
    """
    memoized = getattr(fun, _STRICT_TARGET_LANE_WRAPPER_ATTR, None)
    if memoized is None or getattr(memoized, "__wrapped__", None) is not fun:
        return None
    return memoized


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
    if optimizer_backend in TARGET_SCIPY_CONTROL_OPTIMIZER_BACKENDS:
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
    return _optimizer_method_for_backend_driver(
        "scipy",
        resolve_reference_least_squares_optimizer_driver(
            limited_memory=limited_memory,
            least_squares_algorithm=least_squares_algorithm,
        ),
        reference_method=_reference_least_squares_driver_method,
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
            "optimizer_backend='host-jax', or "
            "optimizer_backend='scipy-jax-fullgraph'. "
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
            "optimizer_backend='host-jax', or "
            "optimizer_backend='scipy-jax-fullgraph'. "
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
    """``0.5 * r.r``, SciPy ``least_squares``' cost, without spurious overflow.

    The plain ``0.5 * vdot(r, r)`` is returned wherever it is finite, so the
    value is unchanged on every normal path.  ``r.r`` can overflow although
    the cost is representable (``r = [1.5e154]``: ``r.r`` is inf, the cost
    1.125e308 is not); there, and only while MINPACK's overflow-safe norm
    ``enorm(r)`` is finite, the cost is ``(enorm(r) * sqrt(0.5))**2``.  A
    non-finite residual keeps the plain, non-finite value.
    """
    residual = jnp.ravel(jnp.asarray(residual))
    half = _device_scalar(0.5, dtype=residual.dtype)
    plain = half * jnp.vdot(residual, residual).real
    norm = _minpack_enorm(residual)
    # XLA's algebraic simplifier reassociates (n * c) * (n * c) with a
    # constant c into (n * n) * (c * c), which overflows again; the barrier
    # keeps the scaled norm as formed before it is squared.
    scaled_norm = lax.optimization_barrier(norm * jnp.sqrt(half))
    return jnp.where(
        ~jnp.isfinite(plain) & jnp.isfinite(norm),
        scaled_norm * scaled_norm,
        plain,
    )


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


def _normalize_solver_args(args):
    if args is None:
        return ()
    if isinstance(args, tuple):
        return args
    return (args,)


# MINPACK ``lmder`` constants (netlib MINPACK; More 1978) as SciPy's
# ``least_squares(method="lm")`` passes them: step-bound factor 100, and
# ``lmpar``'s cap of ten secular-equation iterations.
_MINPACK_STEP_BOUND_FACTOR = 100.0
_MINPACK_LMPAR_MAX_ITERATIONS = 10
# enorm's range limits: squares of components above ``rdwarf`` and below
# ``rgiant / n`` neither underflow nor overflow when summed in double precision.
_MINPACK_ENORM_RDWARF = 3.834e-20
_MINPACK_ENORM_RGIANT = 1.304e19


def _minpack_enorm(vector):
    """MINPACK ``enorm``: the Euclidean norm without overflow or underflow.

    Components are split into small (``<= rdwarf``), intermediate and large
    (``>= rgiant / n``) ranges; the small and large sums are scaled by their
    range maximum before squaring, so finite vectors never overflow to inf.
    The scaled sums are formed at once rather than by enorm's running
    rescale, which changes only rounding.
    """
    vector = jnp.ravel(vector)
    dtype = vector.dtype
    zero = jnp.zeros((), dtype=dtype)
    one = jnp.ones((), dtype=dtype)
    rdwarf = jnp.asarray(_MINPACK_ENORM_RDWARF, dtype=dtype)
    agiant = jnp.asarray(_MINPACK_ENORM_RGIANT / vector.size, dtype=dtype)
    magnitude = jnp.abs(vector)
    intermediate = (magnitude > rdwarf) & (magnitude < agiant)
    small = magnitude <= rdwarf
    # Everything else is large, NaN included, as in enorm's branch order.
    large = ~(intermediate | small)

    def scaled_sum(mask):
        range_max = jnp.max(jnp.where(mask, magnitude, zero))
        ratio = jnp.where(
            magnitude == range_max,
            one,
            magnitude / jnp.where(range_max == zero, one, range_max),
        )
        return range_max, jnp.sum(jnp.where(mask & (magnitude != zero), ratio**2, zero))

    x1max, s1 = scaled_sum(large)
    x3max, s3 = scaled_sum(small)
    s2 = jnp.sum(jnp.where(intermediate, magnitude**2, zero))
    safe_x1max = jnp.where(x1max == zero, one, x1max)
    safe_x3max = jnp.where(x3max == zero, one, x3max)
    safe_s2 = jnp.where(s2 == zero, one, s2)
    mixed = jnp.where(
        s2 >= x3max,
        jnp.sqrt(s2 * (one + (x3max / safe_s2) * (x3max * s3))),
        jnp.sqrt(x3max * ((s2 / safe_x3max) + (x3max * s3))),
    )
    return jnp.where(
        s1 != zero,
        x1max * jnp.sqrt(s1 + (s2 / safe_x1max) / safe_x1max),
        jnp.where(s2 != zero, mixed, x3max * jnp.sqrt(s3)),
    )


def _minpack_scaled_gradient_cosine(r_matrix, pivots, qtf, column_norms, fnorm):
    """Return MINPACK ``lmder``'s ``gnorm``, the largest |cos(f, J[:, j])|.

    ``r_matrix``, ``pivots`` and ``qtf`` are J's column-pivoted QR and the
    leading ``Q^T f``, so ``(R^T qtf)_j`` is ``J[:, pivots[j]]^T f``. Zero
    Jacobian columns contribute nothing, and ``fnorm == 0`` gives zero.
    """
    dtype = qtf.dtype
    zero = jnp.zeros((), dtype=dtype)
    one = jnp.ones((), dtype=dtype)
    residual_is_zero = fnorm == zero
    projected = r_matrix.T @ (qtf / jnp.where(residual_is_zero, one, fnorm))
    pivoted_norms = column_norms[pivots]
    nonzero_column = pivoted_norms != zero
    cosines = jnp.where(
        nonzero_column,
        jnp.abs(projected) / jnp.where(nonzero_column, pivoted_norms, one),
        zero,
    )
    return jnp.where(residual_is_zero, zero, jnp.max(cosines))


def _minpack_qrsolv(r_matrix, pivots, scaled_diag, qtb):
    """Solve MINPACK ``qrsolv``'s damped system for one ``sqrt(par) * diag``.

    Minimizes ``||[R; S] z - [qtb; 0]||`` with ``S = diag(scaled_diag[pivots])``
    and returns ``(x, s_matrix)``: ``x[pivots] = z`` and the triangular factor
    with ``s_matrix^T s_matrix = R^T R + S^2``. A dense Householder QR replaces
    MINPACK's Givens sweep; the factors agree up to row signs.
    """
    n = r_matrix.shape[0]
    dtype = r_matrix.dtype
    augmented = jnp.concatenate((r_matrix, jnp.diag(scaled_diag[pivots])), axis=0)
    q_matrix, s_matrix = jnp.linalg.qr(augmented, mode="reduced")
    solution = jsp_linalg.solve_triangular(
        s_matrix,
        q_matrix[:n].T @ qtb,
        lower=False,
    )
    return jnp.zeros(n, dtype=dtype).at[pivots].set(solution), s_matrix


def _minpack_lmpar(r_matrix, pivots, diag, qtb, delta, par):
    """MINPACK ``lmpar``: the Levenberg-Marquardt parameter for step bound ``delta``.

    Returns ``(x, par)`` with ``x`` solving ``(J^T J + par D^2) x = J^T f`` in
    J's pivoted-QR form (the step is ``-x``). Either ``par = 0`` and the
    Gauss-Newton ``x`` has ``||D x|| <= 1.1 delta``, or ``par > 0`` solves the
    secular equation ``||D x(par)|| = delta`` to 10% within ten iterations.
    """
    n = r_matrix.shape[0]
    dtype = r_matrix.dtype
    zero = jnp.zeros((), dtype=dtype)
    p1 = jnp.asarray(0.1, dtype=dtype)
    p001 = jnp.asarray(0.001, dtype=dtype)
    dwarf = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    pivoted_diag = diag[pivots]

    # Gauss-Newton direction. The first zero on R's diagonal truncates the
    # solve to the leading nonsingular block (``nsing``); later entries are 0.
    nonsingular = jnp.cumsum(jnp.diagonal(r_matrix) == zero) == 0
    full_rank = nonsingular[-1]
    leading_block = jnp.where(
        nonsingular[:, None] & nonsingular[None, :],
        r_matrix,
        jnp.eye(n, dtype=dtype),
    )
    gauss_newton = jsp_linalg.solve_triangular(
        leading_block,
        jnp.where(nonsingular, qtb, zero),
        lower=False,
    )
    x_gauss_newton = jnp.zeros(n, dtype=dtype).at[pivots].set(gauss_newton)
    scaled_x = diag * x_gauss_newton
    dxnorm = _minpack_enorm(scaled_x)
    fp = dxnorm - delta
    gauss_newton_accepted = fp <= p1 * delta

    # Bracket the root: the Newton step gives the lower bound ``parl`` (full
    # rank only), the scaled gradient the upper bound ``paru``.
    newton_rhs = pivoted_diag * (scaled_x[pivots] / dxnorm)
    newton_w = jsp_linalg.solve_triangular(
        leading_block,
        newton_rhs,
        trans="T",
        lower=False,
    )
    newton_w_norm = _minpack_enorm(newton_w)
    parl = jnp.where(full_rank, fp / delta / newton_w_norm / newton_w_norm, zero)
    gnorm = _minpack_enorm((r_matrix.T @ qtb) / pivoted_diag)
    paru = gnorm / delta
    paru = jnp.where(paru == zero, dwarf / jnp.minimum(delta, p1), paru)
    par = jnp.minimum(jnp.maximum(par, parl), paru)
    par = jnp.where(par == zero, gnorm / dxnorm, par)

    def secular_cond(carry):
        return ~carry["done"]

    def secular_body(carry):
        trial_par = jnp.where(
            carry["par"] == zero,
            jnp.maximum(dwarf, p001 * carry["paru"]),
            carry["par"],
        )
        x, s_matrix = _minpack_qrsolv(
            r_matrix,
            pivots,
            jnp.sqrt(trial_par) * diag,
            qtb,
        )
        scaled = diag * x
        trial_dxnorm = _minpack_enorm(scaled)
        previous_fp = carry["fp"]
        trial_fp = trial_dxnorm - delta
        iteration = carry["iteration"] + 1
        done = (
            (jnp.abs(trial_fp) <= p1 * delta)
            | (
                (carry["parl"] == zero)
                & (trial_fp <= previous_fp)
                & (previous_fp < zero)
            )
            | (iteration == _MINPACK_LMPAR_MAX_ITERATIONS)
        )
        correction = jsp_linalg.solve_triangular(
            s_matrix,
            pivoted_diag * (scaled[pivots] / trial_dxnorm),
            trans="T",
            lower=False,
        )
        correction_norm = _minpack_enorm(correction)
        parc = trial_fp / delta / correction_norm / correction_norm
        parl_next = jnp.where(
            trial_fp > zero,
            jnp.maximum(carry["parl"], trial_par),
            carry["parl"],
        )
        paru_next = jnp.where(
            trial_fp < zero,
            jnp.minimum(carry["paru"], trial_par),
            carry["paru"],
        )
        return {
            "iteration": iteration,
            "x": x,
            "fp": trial_fp,
            "par": jnp.where(done, trial_par, jnp.maximum(parl_next, trial_par + parc)),
            "parl": parl_next,
            "paru": paru_next,
            "done": done,
        }

    final = lax.while_loop(
        secular_cond,
        secular_body,
        {
            "iteration": jnp.asarray(0, dtype=jnp.int32),
            "x": x_gauss_newton,
            "fp": fp,
            "par": par,
            "parl": parl,
            "paru": paru,
            "done": gauss_newton_accepted,
        },
    )
    return final["x"], jnp.where(gauss_newton_accepted, zero, final["par"])


def _least_squares_result_message(status, success, info=0):
    # A non-finite state is a failure even when that step also met a
    # MINPACK stop test, so it is reported before the stop code.
    if int(_host_scalar(status, dtype=np.int64)) == 2:
        return "non-finite residual, gradient, or linear solve encountered"
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
    return "maximum iterations reached"


def _make_traceable_levenberg_marquardt_minpack_runner(
    residual_fn,
    maxiter,
    ftol,
    xtol,
    gtol,
    callback_enabled,
    progress_callback_enabled,
):
    """Return the memoized compiled QR-LM runner for one residual callable.

    ``residual_fn`` identity owns the cache entry; the cache key is exactly the
    constant set the builder bakes into the trace. The decision vector, the
    residual ``args``, and the callback tokens stay runtime arguments, so
    repeated solves reuse one executable instead of retracing per call.
    """
    cache_key = (
        int(maxiter),
        float(ftol),
        float(xtol),
        float(gtol),
        bool(callback_enabled),
        bool(progress_callback_enabled),
    )
    return _cached_traceable_runner(
        _TRACEABLE_LM_QR_RUNNER_CACHE,
        residual_fn,
        cache_key,
        lambda residual_fn_ref: _build_traceable_levenberg_marquardt_minpack_runner(
            residual_fn_ref,
            int(maxiter),
            float(ftol),
            float(xtol),
            float(gtol),
            bool(callback_enabled),
            bool(progress_callback_enabled),
        ),
    )


def _build_traceable_levenberg_marquardt_minpack_runner(
    residual_fn_ref,
    maxiter,
    ftol,
    xtol,
    gtol,
    callback_enabled,
    progress_callback_enabled,
):
    def run_solver(x_init, fn_args, callback_token, progress_callback_token):
        residual_fn = _lookup_traceable_runner_callable(
            residual_fn_ref,
            "LM QR residual",
        )
        # Flatten inside the trace: the dense QR step needs one flat column
        # space, and keeping ``unravel`` a traced-local keeps the decision
        # pytree structure out of the runner cache key (``jax.jit`` already
        # discriminates it through the argument signature).
        flat_x_init, unravel = ravel_pytree(x_init)
        dtype = flat_x_init.dtype

        def residual_eval(flat_x):
            return jnp.ravel(jnp.asarray(residual_fn(unravel(flat_x), *fn_args)))

        def jacobian_eval(flat_x):
            def jvp_fn(x, v):
                return jax.jvp(residual_eval, (x,), (v,))[1]

            return _materialize_dense_jacobian(jvp_fn, flat_x)

        # Build the scalars inside the trace so they are staged as constants
        # rather than closed-over concrete device arrays (which JAX bakes via
        # mlir.ir_constant -> a device->host copy, tripping transfer_guard).
        zero = _device_scalar(0.0, dtype=dtype)
        one = _device_scalar(1.0, dtype=dtype)
        p1 = _device_scalar(0.1, dtype=dtype)
        p5 = _device_scalar(0.5, dtype=dtype)
        p25 = _device_scalar(0.25, dtype=dtype)
        p75 = _device_scalar(0.75, dtype=dtype)
        p0001 = _device_scalar(1.0e-4, dtype=dtype)
        epsmch = _device_scalar(np.finfo(np.dtype(dtype)).eps, dtype=dtype)
        ftol_value = _device_scalar(ftol, dtype=dtype)
        xtol_value = _device_scalar(xtol, dtype=dtype)
        gtol_value = _device_scalar(gtol, dtype=dtype)
        factor = _device_scalar(_MINPACK_STEP_BOUND_FACTOR, dtype=dtype)
        info_none = jnp.asarray(0, dtype=jnp.int32)

        # Upstream passes x_scale=1.0, which SciPy maps to MINPACK mode=2 with
        # diag = 1: the scaling stays fixed instead of tracking column norms.
        diag = jnp.ones_like(flat_x_init)
        residual0 = residual_eval(flat_x_init)
        jacobian0 = jacobian_eval(flat_x_init)
        xnorm0 = _minpack_enorm(diag * flat_x_init)
        delta0 = factor * xnorm0
        state0 = {
            "x": flat_x_init,
            "residual": residual0,
            "jacobian": jacobian0,
            "fnorm": _minpack_enorm(residual0),
            "xnorm": xnorm0,
            "delta": jnp.where(delta0 == zero, factor, delta0),
            "par": zero,
            "first_iteration": jnp.asarray(True),
            "gnorm": zero,
            "ratio": zero,
            "nfev": jnp.asarray(1, dtype=jnp.int32),
            # njev counts the Jacobians lmder factors (one per outer
            # iteration); jacobian_evaluations also counts the one evaluated at
            # an accepted end point for the result, as SciPy's wrapper does.
            "njev": jnp.asarray(0, dtype=jnp.int32),
            "jacobian_evaluations": jnp.asarray(1, dtype=jnp.int32),
            "info": info_none,
            "nonfinite": ~(
                jnp.all(jnp.isfinite(flat_x_init))
                & jnp.all(jnp.isfinite(residual0))
                & jnp.all(jnp.isfinite(jacobian0))
            ),
        }

        def trial_step(state, r_matrix, pivots, qtf):
            """One pass of ``lmder``'s inner loop: a step, its test, the tests."""
            lm_x, par = _minpack_lmpar(
                r_matrix,
                pivots,
                diag,
                qtf,
                state["delta"],
                state["par"],
            )
            step = -lm_x
            x_trial = state["x"] + step
            pnorm = _minpack_enorm(diag * step)
            # Until the first successful step, the bound never exceeds the step.
            delta = jnp.where(
                state["first_iteration"],
                jnp.minimum(state["delta"], pnorm),
                state["delta"],
            )
            residual_trial = residual_eval(x_trial)
            nfev = state["nfev"] + 1
            fnorm = state["fnorm"]
            fnorm1 = _minpack_enorm(residual_trial)

            # Scaled actual and predicted reductions and the directional
            # derivative; a NaN or overflowing trial compares False and falls
            # to actred = -1, a rejected step.
            actred = jnp.where(p1 * fnorm1 < fnorm, one - (fnorm1 / fnorm) ** 2, -one)
            temp1 = _minpack_enorm(r_matrix @ step[pivots]) / fnorm
            temp2 = jnp.sqrt(par) * pnorm / fnorm
            prered = temp1**2 + temp2**2 / p5
            dirder = -(temp1**2 + temp2**2)
            ratio = jnp.where(prered != zero, actred / prered, zero)

            # Step-bound update: shrink by ``temp`` on a poor ratio (``temp`` by
            # the quadratic model when actred < 0), expand to 2 ||D p|| on a good
            # ratio or a Gauss-Newton step.
            shrink = jnp.where(
                actred >= zero,
                p5,
                p5 * dirder / (dirder + p5 * actred),
            )
            shrink = jnp.where((p1 * fnorm1 >= fnorm) | (shrink < p1), p1, shrink)
            poor_ratio = ratio <= p25
            expand = (par == zero) | (ratio >= p75)
            delta = jnp.where(
                poor_ratio,
                shrink * jnp.minimum(delta, pnorm / p1),
                jnp.where(expand, pnorm / p5, delta),
            )
            par = jnp.where(
                poor_ratio,
                par / shrink,
                jnp.where(expand, p5 * par, par),
            )

            # A rejected step keeps x, f and J; only delta and par moved.
            accepted = ratio >= p0001
            x_next = jnp.where(accepted, x_trial, state["x"])
            residual_next = jnp.where(accepted, residual_trial, state["residual"])
            xnorm = jnp.where(accepted, _minpack_enorm(diag * x_trial), state["xnorm"])
            fnorm_next = jnp.where(accepted, fnorm1, fnorm)
            jacobian_next = lax.cond(
                accepted,
                jacobian_eval,
                lambda _x: state["jacobian"],
                x_trial,
            )

            # Convergence tests (a later match overrides an earlier one, as in
            # lmder.f), then the stringent-tolerance and budget tests.
            ftol_met = (
                (jnp.abs(actred) <= ftol_value)
                & (prered <= ftol_value)
                & (p5 * ratio <= one)
            )
            xtol_met = delta <= xtol_value * xnorm
            info = jnp.where(
                ftol_met & xtol_met,
                3,
                jnp.where(xtol_met, 2, jnp.where(ftol_met, 1, 0)),
            )
            stringent = jnp.where(
                state["gnorm"] <= epsmch,
                8,
                jnp.where(
                    delta <= epsmch * xnorm,
                    7,
                    jnp.where(
                        (jnp.abs(actred) <= epsmch)
                        & (prered <= epsmch)
                        & (p5 * ratio <= one),
                        6,
                        jnp.where(nfev >= maxiter, 5, 0),
                    ),
                ),
            )
            info = jnp.where(info == 0, stringent, info).astype(jnp.int32)
            # A non-finite accepted x or Jacobian ends the solve as a failure,
            # whatever stop code this step set: lmder would factor that
            # Jacobian next, and the result must not report success with it.
            nonfinite = accepted & ~(
                jnp.all(jnp.isfinite(x_trial)) & jnp.all(jnp.isfinite(jacobian_next))
            )

            if callback_enabled:
                lax.cond(
                    accepted,
                    lambda _: jax.debug.callback(
                        _invoke_traceable_lm_callback,
                        callback_token,
                        x_next,
                        ordered=False,
                    ),
                    lambda _: None,
                    operand=None,
                )
            if progress_callback_enabled:
                lax.cond(
                    accepted,
                    lambda _: jax.debug.callback(
                        _invoke_traceable_progress_callback,
                        progress_callback_token,
                        nfev - 1,
                        _least_squares_cost(residual_next),
                        jnp.max(jnp.abs(jacobian_next.T @ residual_next)),
                        ordered=False,
                    ),
                    lambda _: None,
                    operand=None,
                )
            return {
                "x": x_next,
                "residual": residual_next,
                "jacobian": jacobian_next,
                "fnorm": fnorm_next,
                "xnorm": xnorm,
                "delta": delta,
                "par": par,
                "first_iteration": state["first_iteration"] & ~accepted,
                "gnorm": state["gnorm"],
                "ratio": ratio,
                "nfev": nfev,
                "njev": state["njev"],
                "jacobian_evaluations": (
                    state["jacobian_evaluations"] + accepted.astype(jnp.int32)
                ),
                "info": info,
                "nonfinite": nonfinite,
            }

        def outer_cond(state):
            return (state["info"] == 0) & ~state["nonfinite"]

        def outer_body(state):
            """One ``lmder`` outer iteration: factor J, gtol test, inner loop."""
            q_matrix, r_matrix, pivots = jsp_linalg.qr(
                state["jacobian"],
                pivoting=True,
                mode="economic",
            )
            qtf = q_matrix.T @ state["residual"]
            gnorm = _minpack_scaled_gradient_cosine(
                r_matrix,
                pivots,
                qtf,
                jax.vmap(_minpack_enorm, in_axes=1)(state["jacobian"]),
                state["fnorm"],
            )

            def inner_cond(inner):
                return (
                    (inner["info"] == 0)
                    & ~inner["nonfinite"]
                    & (inner["ratio"] < p0001)
                )

            return lax.while_loop(
                inner_cond,
                lambda inner: trial_step(inner, r_matrix, pivots, qtf),
                {
                    **state,
                    "njev": state["njev"] + 1,
                    "gnorm": gnorm,
                    "ratio": zero,
                    "info": jnp.where(
                        gnorm <= gtol_value,
                        jnp.asarray(4, dtype=jnp.int32),
                        info_none,
                    ),
                },
            )

        final = lax.while_loop(outer_cond, outer_body, state0)
        gradient, hessian = _least_squares_linearization_from_jacobian(
            final["residual"],
            final["jacobian"],
        )
        converged = (final["info"] >= 1) & (final["info"] <= 4)
        cost = _least_squares_cost(final["residual"])
        # A finite residual and Jacobian can still overflow in J^T r, J^T J
        # or the cost; a result carrying any non-finite value is a failure.
        nonfinite = (
            final["nonfinite"]
            | ~jnp.all(jnp.isfinite(gradient))
            | ~jnp.all(jnp.isfinite(hessian))
            | ~jnp.isfinite(cost)
        )
        return {
            "x": final["x"],
            "residual": final["residual"],
            "jacobian": final["jacobian"],
            "gradient": gradient,
            "hessian": hessian,
            "cost": cost,
            "par": final["par"],
            "nit": final["nfev"] - 1,
            "njev": final["njev"],
            "jacobian_evaluations": final["jacobian_evaluations"],
            "status": jnp.where(
                nonfinite,
                jnp.asarray(2, dtype=jnp.int32),
                jnp.where(
                    final["nfev"] > 1,
                    jnp.asarray(1, dtype=jnp.int32),
                    jnp.asarray(0, dtype=jnp.int32),
                ),
            ),
            "info": final["info"],
            "success": converged & ~nonfinite,
        }

    run_solver.__name__ = "traceable_levenberg_marquardt_minpack_run_solver"
    if not callback_enabled and not progress_callback_enabled:

        def run_solver_without_callbacks(x_init, fn_args):
            return run_solver(x_init, fn_args, 0, 0)

        run_solver_without_callbacks.__name__ = run_solver.__name__
        return jax.jit(run_solver_without_callbacks)
    # Callback tokens stay traced operands, not static arguments; see
    # ``_traceable_callback_token_operand``.
    return jax.jit(run_solver)


def levenberg_marquardt_minpack_traceable(
    residual_fn,
    x0,
    *,
    maxiter=1500,
    ftol=1e-8,
    xtol=1e-8,
    gtol=1e-8,
    max_dense_linearization_bytes=None,
    callback=None,
    progress_callback=None,
    args=(),
):
    """Trace-safe MINPACK ``lmder`` Levenberg-Marquardt for residual vectors.

    Mirrors ``scipy.optimize.least_squares(method="lm", x_scale=1.0)``: the
    trust-region bound ``delta`` sets the Marquardt parameter through
    ``lmpar``'s secular equation, and the solve stops only on MINPACK's
    ``info`` tests, with ``maxiter`` as ``max_nfev``. Dense Householder QR
    replaces MINPACK's packed QR and Givens sweeps, so iterates agree to
    rounding rather than bytewise.
    """
    x = jax.tree.map(jnp.asarray, x0)
    flat_x0, unravel = ravel_pytree(x)
    normalized_args = _normalize_solver_args(args)
    dtype = flat_x0.dtype
    if int(maxiter) < 1:
        raise ValueError(
            f"maxiter is MINPACK's max_nfev and must be positive; got {maxiter!r}."
        )

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
    if linearization_rows < linearization_cols:
        raise ValueError(
            "MINPACK Levenberg-Marquardt needs at least as many residuals as "
            f"variables; got {linearization_rows} residuals for "
            f"{linearization_cols} variables."
        )
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

    runner = _make_traceable_levenberg_marquardt_minpack_runner(
        residual_fn,
        maxiter,
        ftol,
        xtol,
        gtol,
        callback is not None,
        progress_callback is not None,
    )
    # The compiled runner carries the flat iterate, so the registered step
    # callback is the flat-vector adapter that restores the caller's pytree.
    callback_token = _register_traceable_callback(
        None
        if callback is None
        else lambda flat_x: callback(_hostify_optimizer_tree(unravel(flat_x)))
    )
    progress_callback_token = _register_traceable_callback(progress_callback)
    try:
        if callback_token == 0 and progress_callback_token == 0:
            state = runner(x, normalized_args)
        else:
            state = runner(
                x,
                normalized_args,
                _traceable_callback_token_operand(callback_token),
                _traceable_callback_token_operand(progress_callback_token),
            )
            jax.effects_barrier()
    finally:
        _unregister_traceable_callback(callback_token)
        _unregister_traceable_callback(progress_callback_token)
    return {
        "x": unravel(state["x"]),
        "residual": state["residual"],
        "residual_jacobian": state["jacobian"],
        "fun": state["cost"],
        "grad": unravel(state["gradient"]),
        "hessian": state["hessian"],
        "damping": state["par"],
        "nit": state["nit"],
        "njev": state["njev"],
        "jacobian_evaluations": state["jacobian_evaluations"],
        "status": state["status"],
        "info": state["info"],
        "success": state["success"],
        "dense_linearization_materialized": True,
        "dense_linearization_kind": "in_loop",
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

    restart, maxiter = _exact_newton_gmres_iteration_limits(rhs.shape[0])
    dx, _ = _run_operator_gmres(
        matvec,
        rhs,
        tol=tol,
        restart=restart,
        maxiter=maxiter,
    )
    residual = rhs - matvec(dx)
    return dx, residual, matvec


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
        dense_materialization_count=_device_int32(
            status.dense_materialization_count,
            like=rhs,
        ),
        lu_factorization_count=_device_int32(
            status.lu_factorization_count,
            like=rhs,
        ),
        lu_solve_count=_device_int32(status.lu_solve_count, like=rhs),
        refinement_correction_count=_device_int32(
            status.refinement_correction_count,
            like=rhs,
        ),
    )


def _eisenstat_walker_strict_cap(tol_value, *, dtype):
    # Floor/cap match the dense-IR accuracy policy SSOT so Newton forcing terms
    # and dense linear-solve tolerances share one production band.
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


def _make_traceable_newton_polish_runner(
    objective_fn,
    maxiter,
    tol,
    stab,
    materialize_hessian,
    max_dense_hessian_bytes,
    progress_callback_enabled,
    linear_solver: TraceableNewtonLinearSolver,
):
    cache_key = (
        int(maxiter),
        float(tol),
        float(stab),
        bool(materialize_hessian),
        max_dense_hessian_bytes,
        bool(progress_callback_enabled),
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
    linear_solver,
):
    requested_materialize_hessian = materialize_hessian

    def run_solver(
        x_init,
        fn_args,
        progress_callback_token,
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
        initial_linear_solver_code = (
            dense_linear_solver_code
            if traceable_dense_lu_enabled
            else (
                hybrid_linear_solver_code
                if traceable_hybrid_dense_lu_enabled
                else operator_linear_solver_code
            )
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
                return hvp_fn(state["x"], v) + stab_value * v

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
    if not progress_callback_enabled:

        def run_solver_without_callback(x_init, fn_args):
            return run_solver(x_init, fn_args, 0)

        run_solver_without_callback.__name__ = run_solver.__name__
        return jax.jit(run_solver_without_callback)
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
    runner = _make_traceable_newton_polish_runner(
        objective_fn,
        int(maxiter),
        float(tol),
        float(stab),
        bool(materialize_hessian),
        max_dense_hessian_bytes,
        progress_callback is not None,
        linear_solver,
    )
    progress_callback_token = _register_traceable_callback(progress_callback)
    normalized_args = _normalize_solver_args(args)
    try:
        if progress_callback_token == 0:
            return runner(x0, normalized_args)
        result = runner(
            x0,
            normalized_args,
            progress_callback_token,
        )
        jax.effects_barrier()
        return result
    finally:
        _unregister_traceable_callback(progress_callback_token)


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


def _materialize_traceable_dense_exact_newton_c2_state(
    residual_fn: Callable[[jax.Array], jax.Array],
    x: jax.Array,
    fn_args: tuple[object, ...],
    *,
    value_jacobian_fn: _ArrayValueAndJacobianWithArgs | None,
) -> _DenseJacobianMaterialization:
    """Materialize C2's state pair from a provider or the legacy AD path."""

    x = jnp.asarray(x)
    if value_jacobian_fn is None:
        return _linearize_and_materialize_dense_square_jacobian(
            residual_fn,
            x,
        )

    residual, jacobian = value_jacobian_fn(x, *fn_args)
    residual = jnp.asarray(residual)
    jacobian = jnp.asarray(jacobian)
    return _DenseJacobianMaterialization(
        residual=residual,
        jacobian=jacobian,
        telemetry=_DenseJacobianMaterializationTelemetry(
            assembler_code=_staged_like(
                residual,
                int(_DenseJacobianAssembler.VALUE_AND_JACOBIAN),
                dtype=jnp.int32,
            ),
            residual_evaluation_count=_staged_like(
                residual,
                1,
                dtype=jnp.int32,
            ),
            primal_traversal_count=_staged_like(
                residual,
                1,
                dtype=jnp.int32,
            ),
            tangent_batch_count=_staged_like(
                residual,
                0,
                dtype=jnp.int32,
            ),
            tangent_direction_count=_staged_like(
                residual,
                0,
                dtype=jnp.int32,
            ),
            batch_width=_staged_like(
                residual,
                0,
                dtype=jnp.int32,
            ),
            tail_width=_staged_like(
                residual,
                0,
                dtype=jnp.int32,
            ),
        ),
    )


def _make_traceable_dense_direct_exact_newton_c2_runner(
    residual_fn: Callable[..., jax.Array],
    maxiter: int,
    tol: float,
    *,
    value_jacobian_fn: _ArrayValueAndJacobianWithArgs | None = None,
):
    """Return a C2 runner keyed by residual and pair-provider identity."""

    pair_provider_key = (
        None
        if value_jacobian_fn is None
        else ("value-jacobian-callable", id(value_jacobian_fn))
    )
    cache_key = (int(maxiter), float(tol), pair_provider_key)

    def build_runner(residual_fn_ref):
        if value_jacobian_fn is None:
            return _build_traceable_dense_direct_exact_newton_c2_runner(
                residual_fn_ref,
                int(maxiter),
                float(tol),
            )
        return _build_traceable_dense_direct_exact_newton_c2_runner(
            residual_fn_ref,
            int(maxiter),
            float(tol),
            value_jacobian_fn=value_jacobian_fn,
        )

    return _cached_traceable_runner(
        _TRACEABLE_DENSE_EXACT_NEWTON_C2_RUNNER_CACHE,
        residual_fn,
        cache_key,
        build_runner,
    )


def _dense_direct_exact_newton_direction_from_jacobian(
    residual: jax.Array,
    jacobian: jax.Array,
    *,
    tol: float | jax.Array,
) -> _DenseExactNewtonDirection:
    """Factor and certify one residual/Jacobian pair bound to the same state."""

    residual = jnp.asarray(residual)
    jacobian = jnp.asarray(jacobian)
    rhs_dtype = residual.dtype
    lu, pivots = _factor_dense_hessian(
        jacobian,
        optimizer_backend="ondevice",
    )
    lu_piv = (lu, pivots)
    initial_solve = _lu_solve_dense_hessian(
        lu_piv,
        residual,
        transpose=False,
    )
    refinement_rhs = residual - jacobian @ initial_solve
    correction = _lu_solve_dense_hessian(
        lu_piv,
        refinement_rhs,
        transpose=False,
    )
    direction = initial_solve + correction
    linear_residual = residual - jacobian @ direction
    status = _linear_solve_status(
        direction,
        linear_residual,
        residual,
        tol=tol,
        iterations=_device_int32(0, like=residual),
    )
    backward_error_success = _dense_matrix_backward_error_success(
        jacobian,
        direction,
        residual,
        tol=tol,
    )
    (
        condition_estimate,
        condition_factorizations,
        condition_lu_solves,
    ) = _dense_matrix_condition_estimate_with_telemetry(
        jacobian,
        lu_piv=lu_piv,
    )
    solve_safe = _dense_matrix_solve_numerically_safe(
        jacobian,
        direction,
        residual,
        tol=tol,
        lu_piv=lu_piv,
        solve_dtype=rhs_dtype,
        condition_estimate=condition_estimate,
    )
    status = status._replace(
        success=(status.success | backward_error_success) & solve_safe,
        lu_factorization_count=(
            _device_int32(1, like=residual) + condition_factorizations
        ),
        lu_solve_count=(_device_int32(2, like=residual) + condition_lu_solves),
        refinement_correction_count=_device_int32(1, like=residual),
    )
    direction = _linear_solve_solution_or_nan(direction, status)
    return _DenseExactNewtonDirection(
        residual=residual,
        jacobian=jacobian,
        lu=lu,
        pivots=pivots,
        initial_solve=initial_solve,
        refinement_rhs=refinement_rhs,
        direction=direction,
        correction=correction,
        linear_residual=linear_residual,
        condition_estimate=condition_estimate,
        status=status,
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
        zero_count = _device_int32(0, like=x_init)

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
            next_state = {
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
            return next_state

        state = lax.while_loop(
            cond_fun,
            body_fun,
            {
                "x": x_init,
                "residual": r0,
                "norm": norm0,
                "previous_norm": norm0,
                "nit": zero_count,
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
        result = {
            "x": state["x"],
            "residual": state["residual"],
            "nit": state["nit"],
            "success": state["norm"] <= tol_value,
            "exact_newton_linear_residual_rel": state[
                "exact_newton_linear_residual_rel"
            ],
            "exact_refinement_correction_rel": state["exact_refinement_correction_rel"],
        }
        return result

    run_solver.__name__ = "traceable_exact_newton_run_solver"
    return jax.jit(run_solver)


def _build_traceable_dense_direct_exact_newton_c2_runner(
    residual_fn_ref,
    maxiter: int,
    tol: float,
    *,
    value_jacobian_fn: _ArrayValueAndJacobianWithArgs | None = None,
):
    """Build native finite-path C2 with optional exact pair assembly."""

    def run_solver(x_init, fn_args):
        residual_fn = _lookup_traceable_runner_callable(
            residual_fn_ref,
            "native-order dense exact Newton residual",
        )
        residual_eval = jax.jit(lambda x: residual_fn(x, *fn_args))
        x_init_array = jnp.asarray(x_init)
        dtype = x_init_array.dtype
        tol_value = _optimizer_scalar(tol, dtype=dtype)
        strict_cap_tol = _eisenstat_walker_strict_cap(
            tol_value,
            dtype=dtype,
        )
        initial_materialization = _materialize_traceable_dense_exact_newton_c2_state(
            residual_eval,
            x_init_array,
            fn_args,
            value_jacobian_fn=value_jacobian_fn,
        )
        initial_residual = initial_materialization.residual
        zero_count = _device_int32(0, like=x_init_array)
        one_count = _device_int32(1, like=x_init_array)
        trace_length = maxiter + 1
        applied_state_trace = jnp.broadcast_to(
            x_init_array,
            (trace_length,) + x_init_array.shape,
        )
        applied_state_trace_active = (
            jnp.zeros(
                (trace_length,),
                dtype=jnp.bool_,
            )
            .at[0]
            .set(True)
        )
        assessed_norm_trace = jnp.full(
            (trace_length,),
            jnp.nan,
            dtype=dtype,
        )
        assessed_norm_trace_active = jnp.zeros(
            (trace_length,),
            dtype=jnp.bool_,
        )

        def cond_fun(state):
            return (
                (state["iteration_count"] < maxiter)
                & (~state["finished"])
                & (~state["numerical_failure"])
            )

        def stop_at_converged(state):
            return {**state, "finished": jnp.asarray(True)}

        def apply_native_update(state):
            current = _dense_direct_exact_newton_direction_from_jacobian(
                state["residual"],
                state["jacobian"],
                tol=strict_cap_tol,
            )
            linear_residual_rel = _relative_residual_norm(
                current.linear_residual,
                current.residual,
            )
            correction_rel = jnp.linalg.norm(current.correction) / jnp.maximum(
                jnp.linalg.norm(current.direction - current.correction),
                _device_scalar(
                    jnp.finfo(current.direction.dtype).tiny,
                    dtype=current.direction.dtype,
                ),
            )
            attempted = {
                **state,
                "linear_solve_attempt_count": (state["linear_solve_attempt_count"] + 1),
                "lu_factorization_count": (
                    state["lu_factorization_count"]
                    + current.status.lu_factorization_count
                ),
                "lu_solve_count": (
                    state["lu_solve_count"] + current.status.lu_solve_count
                ),
                "refinement_correction_count": (
                    state["refinement_correction_count"]
                    + current.status.refinement_correction_count
                ),
                "exact_newton_linear_residual_rel": linear_residual_rel,
                "exact_refinement_correction_rel": correction_rel,
            }

            def materialize_updated_state(current_state):
                candidate_x = current_state["x"] - current.direction
                candidate_materialization = (
                    _materialize_traceable_dense_exact_newton_c2_state(
                        residual_eval,
                        candidate_x,
                        fn_args,
                        value_jacobian_fn=value_jacobian_fn,
                    )
                )
                candidate_residual = candidate_materialization.residual
                candidate_jacobian = candidate_materialization.jacobian
                candidate_finite = (
                    jnp.all(jnp.isfinite(candidate_x))
                    & jnp.all(jnp.isfinite(candidate_residual))
                    & jnp.all(jnp.isfinite(candidate_jacobian))
                )
                next_iteration = current_state["iteration_count"] + 1
                next_state = {
                    **current_state,
                    "x": candidate_x,
                    "residual": candidate_residual,
                    "jacobian": candidate_jacobian,
                    "iteration_count": next_iteration,
                    "applied_update_count": (current_state["applied_update_count"] + 1),
                    "applied_state_trace": current_state["applied_state_trace"]
                    .at[next_iteration]
                    .set(candidate_x),
                    "applied_state_trace_active": current_state[
                        "applied_state_trace_active"
                    ]
                    .at[next_iteration]
                    .set(True),
                    "numerical_failure": ~candidate_finite,
                    "dense_materialization_count": (
                        current_state["dense_materialization_count"] + 1
                    ),
                }
                return next_state

            return lax.cond(
                current.status.success,
                materialize_updated_state,
                lambda current_state: {
                    **current_state,
                    "numerical_failure": jnp.asarray(True),
                },
                attempted,
            )

        def body_fun(state):
            assessed_norm = jnp.linalg.norm(state["residual"])
            has_initial_norm = state["has_initial_norm"]
            initial_norm = lax.select(
                has_initial_norm,
                state["initial_norm"],
                assessed_norm,
            )
            assessment_index = state["iteration_count"]
            assessed_state = {
                **state,
                "initial_norm": initial_norm,
                "has_initial_norm": jnp.asarray(True),
                "assessed_norm": assessed_norm,
                "assessed_norm_trace": state["assessed_norm_trace"]
                .at[assessment_index]
                .set(assessed_norm),
                "assessed_norm_trace_active": state["assessed_norm_trace_active"]
                .at[assessment_index]
                .set(True),
            }
            return lax.cond(
                assessed_norm <= tol_value,
                stop_at_converged,
                apply_native_update,
                assessed_state,
            )

        state = lax.while_loop(
            cond_fun,
            body_fun,
            {
                "x": x_init_array,
                "residual": initial_residual,
                "jacobian": initial_materialization.jacobian,
                "iteration_count": zero_count,
                "applied_update_count": zero_count,
                "finished": jnp.asarray(False),
                "numerical_failure": jnp.asarray(False),
                "has_initial_norm": jnp.asarray(False),
                "initial_norm": jnp.asarray(jnp.nan, dtype=dtype),
                "assessed_norm": jnp.asarray(jnp.nan, dtype=dtype),
                "applied_state_trace": applied_state_trace,
                "applied_state_trace_active": applied_state_trace_active,
                "assessed_norm_trace": assessed_norm_trace,
                "assessed_norm_trace_active": assessed_norm_trace_active,
                "exact_newton_linear_residual_rel": jnp.asarray(
                    jnp.nan,
                    dtype=dtype,
                ),
                "exact_refinement_correction_rel": jnp.asarray(
                    jnp.nan,
                    dtype=dtype,
                ),
                "linear_solve_attempt_count": zero_count,
                "dense_materialization_count": one_count,
                "lu_factorization_count": zero_count,
                "lu_solve_count": zero_count,
                "refinement_correction_count": zero_count,
            },
        )
        success = state["has_initial_norm"] & (state["assessed_norm"] <= tol_value)
        improved_assessed_state = (
            state["has_initial_norm"]
            & jnp.isfinite(state["assessed_norm"])
            & (state["assessed_norm"] <= state["initial_norm"])
        )
        native_persist_predicate = success | improved_assessed_state
        persist_solved_state = native_persist_predicate & (~state["numerical_failure"])
        returned_x = lax.select(
            persist_solved_state,
            state["x"],
            x_init_array,
        )
        rollback_branch_taken = ~persist_solved_state

        def rebuild_initial_residual(_operand):
            rebuilt = _materialize_traceable_dense_exact_newton_c2_state(
                residual_eval,
                x_init_array,
                fn_args,
                value_jacobian_fn=value_jacobian_fn,
            )
            return rebuilt.residual, rebuilt.jacobian, one_count

        (
            returned_residual,
            returned_jacobian,
            rollback_recompute_count,
        ) = lax.cond(
            rollback_branch_taken,
            rebuild_initial_residual,
            lambda _operand: (
                state["residual"],
                state["jacobian"],
                zero_count,
            ),
            operand=None,
        )
        stop_reason_code = jnp.where(
            state["numerical_failure"],
            _device_int32(
                _C2_STOP_REASON_NUMERICAL_FAILURE,
                like=x_init_array,
            ),
            jnp.where(
                success,
                _device_int32(_C2_STOP_REASON_CONVERGED, like=x_init_array),
                _device_int32(_C2_STOP_REASON_MAXITER, like=x_init_array),
            ),
        )
        native_result = _NativeDenseExactNewtonC2Result(
            x=returned_x,
            residual=returned_residual,
            returned_jacobian=returned_jacobian,
            iteration_count=state["iteration_count"],
            applied_update_count=state["applied_update_count"],
            success=success,
            numerical_failure=state["numerical_failure"],
            stop_reason_code=stop_reason_code,
            rollback_branch_taken=rollback_branch_taken,
            native_persist_predicate=native_persist_predicate,
            persist_solved_state=persist_solved_state,
            initial_norm=state["initial_norm"],
            assessed_norm=state["assessed_norm"],
            returned_norm=jnp.linalg.norm(returned_residual),
            applied_state_trace=state["applied_state_trace"],
            applied_state_trace_active=state["applied_state_trace_active"],
            assessed_norm_trace=state["assessed_norm_trace"],
            assessed_norm_trace_active=state["assessed_norm_trace_active"],
            exact_newton_linear_residual_rel=state["exact_newton_linear_residual_rel"],
            exact_refinement_correction_rel=state["exact_refinement_correction_rel"],
            linear_solve_attempt_count=state["linear_solve_attempt_count"],
            dense_materialization_count=(
                state["dense_materialization_count"] + rollback_recompute_count
            ),
            lu_factorization_count=state["lu_factorization_count"],
            lu_solve_count=state["lu_solve_count"],
            refinement_correction_count=state["refinement_correction_count"],
            rollback_recompute_count=rollback_recompute_count,
        )
        return native_result

    run_solver.__name__ = "traceable_dense_direct_exact_newton_c2_run_solver"
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


def _newton_exact_traceable_c2(
    residual_fn,
    x0,
    *,
    maxiter: int,
    tol: float,
    args: tuple[object, ...] = (),
    value_jacobian_fn: _ArrayValueAndJacobianWithArgs | None = None,
) -> dict[str, object]:
    """Run C2 and normalize its returned-state linearization for the adapter."""

    runner = _make_traceable_dense_direct_exact_newton_c2_runner(
        residual_fn,
        maxiter,
        tol,
        value_jacobian_fn=value_jacobian_fn,
    )
    result = runner(x0, _normalize_solver_args(args))
    jacobian = result.returned_jacobian
    return {
        "x": result.x,
        "residual": result.residual,
        "jacobian": jacobian,
        "nit": result.iteration_count,
        "success": result.success,
        "exact_newton_linear_residual_rel": (result.exact_newton_linear_residual_rel),
        "exact_refinement_correction_rel": (result.exact_refinement_correction_rel),
        "exact_newton_variant_dense_linearization_used": (
            result.dense_materialization_count > _device_int32(0, like=result.x)
        ),
        "exact_newton_variant_linear_solve_attempt_count": (
            result.linear_solve_attempt_count
        ),
        "exact_newton_variant_dense_materialization_count": (
            result.dense_materialization_count
        ),
        "exact_newton_variant_lu_factorization_count": (result.lu_factorization_count),
        "exact_newton_variant_lu_solve_count": result.lu_solve_count,
        "exact_newton_variant_refinement_correction_count": (
            result.refinement_correction_count
        ),
        "exact_newton_variant_applied_update_count": result.applied_update_count,
        "exact_newton_variant_stop_reason_code": result.stop_reason_code,
        "exact_newton_variant_numerical_failure": result.numerical_failure,
        "exact_newton_variant_rollback_branch_taken": (result.rollback_branch_taken),
        "exact_newton_variant_rollback_recompute_count": (
            result.rollback_recompute_count
        ),
        "exact_newton_variant_native_persist_predicate": (
            result.native_persist_predicate
        ),
        "exact_newton_variant_persist_solved_state": result.persist_solved_state,
        "exact_newton_variant_initial_norm": result.initial_norm,
        "exact_newton_variant_assessed_norm": result.assessed_norm,
        "exact_newton_variant_returned_norm": result.returned_norm,
    }


@lru_cache(maxsize=2)
def make_traceable_exact_newton_variant_contract(
    variant: object,
) -> TraceableExactNewtonVariantContract:
    """Resolve C0/C2 once, before callers stage the selected solver."""

    if variant == "C0":
        return TraceableExactNewtonVariantContract(
            variant="C0",
            solver=newton_exact_traceable,
            factorization_backend="operator-gmres",
            returns_jacobian=False,
        )
    if variant == "C2":
        return TraceableExactNewtonVariantContract(
            variant="C2",
            solver=_newton_exact_traceable_c2,
            factorization_backend="dense-lu",
            returns_jacobian=True,
        )
    raise ValueError(f"exact Newton variant must be one of: C0, C2; got {variant!r}.")


# ---------------------------------------------------------------------------
# Dispatcher — shared hub for all optimizer methods
# ---------------------------------------------------------------------------


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
        njev=int(_host_scalar(result["njev"], dtype=np.int64)),
        jacobian_evaluations=int(
            _host_scalar(result["jacobian_evaluations"], dtype=np.int64)
        ),
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
    )


def _least_squares_tolerances(tol, options):
    """Return MINPACK's ``(ftol, xtol, gtol)``: each explicit option, else ``tol``.

    Upstream's Boozer least squares passes ``ftol = xtol = gtol = tol`` to
    ``least_squares(method="lm")``, so a caller's single ``tol`` gates all
    three tests; an option set to ``None`` counts as omitted.
    """
    return tuple(
        tol if options.get(name) is None else options[name]
        for name in ("ftol", "xtol", "gtol")
    )


def _require_dense_least_squares_linearization(materialize):
    """Refuse an effective ``materialize_dense_linearization=False`` for lm-minpack.

    ``materialize`` is the caller's resolved setting (``None`` when unset).
    MINPACK's LM factors the dense residual Jacobian every iteration; the
    matrix-free option belonged to the removed GMRES Levenberg-Marquardt.
    ``max_dense_linearization_bytes`` caps the bytes of the dense residual
    Jacobian and Hessian instead; it gates those two arrays, not the solver's
    peak workspace (QR factors and ``lmpar`` temporaries come on top).
    """
    if materialize is not None and not materialize:
        raise ValueError(
            "lm-minpack always materializes the dense residual Jacobian and "
            "cannot honour materialize_dense_linearization=False (the "
            "matrix-free Levenberg-Marquardt was removed); cap its dense "
            "residual Jacobian and Hessian bytes with "
            "max_dense_linearization_bytes instead."
        )


def target_least_squares(
    residual_fn,
    x0,
    *,
    method="lm-minpack-ondevice",
    tol=1e-10,
    maxiter=1500,
    options=None,
    callback=None,
    progress_callback=None,
    args=(),
):
    """Explicit JAX target least-squares entrypoint.

    ``tol`` is the default for MINPACK's ``ftol``, ``xtol`` and ``gtol`` (see
    ``_least_squares_tolerances``); ``options`` may set any of them.
    """
    if method not in _TARGET_LEAST_SQUARES_METHODS:
        raise ValueError(
            "target_least_squares() only supports method='lm-minpack-ondevice'. "
            f"Got {method!r}."
        )

    options = dict(options or {})
    if callback is not None:
        options["callback"] = callback
    if progress_callback is not None:
        options["progress_callback"] = progress_callback

    require_target_backend_x64("ondevice")
    ftol, xtol, gtol = _least_squares_tolerances(tol, options)
    _require_dense_least_squares_linearization(
        options.get("materialize_dense_linearization")
    )
    max_dense_linearization_bytes = options.get("max_dense_linearization_bytes")
    callback = options.get("callback")
    progress_callback = options.get("progress_callback")
    result = levenberg_marquardt_minpack_traceable(
        residual_fn,
        x0,
        maxiter=maxiter,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        max_dense_linearization_bytes=max_dense_linearization_bytes,
        callback=callback,
        progress_callback=progress_callback,
        args=args,
    )

    return _least_squares_state_to_optimize_result(result)


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
    allow_jax_host_control=False,
):
    """Explicit CPU/reference scalar optimizer entrypoint."""
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
    initial_value_and_grad=None,
):
    """Explicit JAX target scalar optimizer entrypoint."""
    options = dict(options or {})
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


from . import reference as optimizer_jax_reference  # noqa: E402
