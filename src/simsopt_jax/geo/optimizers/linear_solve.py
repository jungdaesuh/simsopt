"""Generic traceable linear-solve primitives for geometry optimizers.

This dependency owner contains status and tolerance contracts, dense/operator
materialization, HVP/JVP construction, direct LU/least-squares solves, and
bounded operator refinement. It does not import the optimizer facade.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import nullcontext
from enum import IntEnum
from functools import partial
from threading import Lock
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import scipy.linalg
from jax import lax
from jax.extend import core as jax_core
from jax.scipy.sparse.linalg import gmres

from simsopt_jax.backend import get_backend_policy
from simsopt_jax.backend.dtypes import (
    explicit_device_array as _explicit_device_array,
)
from simsopt_jax.backend.dtypes import (
    runtime_device_put,
)
from simsopt_jax.core._device_scalars import staged_like as _staged_like
from simsopt_jax.geo.optimizers._shared import (
    _CACHEABLE_LINEAR_OPERATOR_ATTR,
    _optimizer_scalar,
)
from simsopt_jax.geo.optimizers.exact_final_linearization import (
    _ExactFinalLinearization,
    _ExactFinalLinearizationInputs,
    _mint_exact_final_linearization,
    _mint_exact_final_linearization_identity,
    _mint_exact_final_linearization_inputs,
)
from simsopt_jax.numerical_policy import mixed_dense_ir_accuracy_policy
from simsopt_jax.runtime.host_boundary import host_array, host_int
from simsopt_jax.runtime.trace_annotations import PhaseId, device_scope

_HAGER_HIGHAM_CONDITION_ITERATIONS = 5

_LINEAR_SOLVE_ITERATIONS_UNKNOWN = -1

_JIT_LINEAR_OPERATOR_CACHE_LOCK = Lock()

_CACHED_HVP_ATTR = "_simsopt_cached_jit_hvp"

_CACHED_JVP_ATTR = "_simsopt_cached_jit_jvp"


class _LinearSolveStatus(NamedTuple):
    success: jax.Array
    residual: jax.Array
    residual_relative: jax.Array
    iterations: jax.Array
    residual_scale: jax.Array | None = None
    requested_tolerance: jax.Array | None = None
    effective_tolerance: jax.Array | None = None
    dense_materialization_count: jax.Array | np.int32 = np.int32(0)
    lu_factorization_count: jax.Array | np.int32 = np.int32(0)
    lu_solve_count: jax.Array | np.int32 = np.int32(0)
    refinement_correction_count: jax.Array | np.int32 = np.int32(0)

    def __array__(self, dtype=None):
        return np.asarray(host_array(self.success, dtype=dtype))

    def __bool__(self):
        return bool(np.asarray(self))


class _DenseJacobianAssembler(IntEnum):
    """Private, statically selected dense-Jacobian assembly strategies."""

    LINEARIZE_ONCE = 0
    CHECKPOINTED_BATCHED_JVP = 1


class _DenseJacobianMaterializationTelemetry(NamedTuple):
    """Static-shape device telemetry for one dense-Jacobian assembly."""

    assembler_code: jax.Array
    residual_evaluation_count: jax.Array
    primal_traversal_count: jax.Array
    tangent_batch_count: jax.Array
    tangent_direction_count: jax.Array
    batch_width: jax.Array
    tail_width: jax.Array


class _DenseJacobianMaterialization(NamedTuple):
    """One fixed-state residual and its column-oriented dense Jacobian."""

    residual: jax.Array
    jacobian: jax.Array
    telemetry: _DenseJacobianMaterializationTelemetry


class _CountedIncrementalGmresTelemetry(NamedTuple):
    """Fixed-shape device telemetry from one incremental GMRES solve."""

    linear_operator_application_count: jax.Array


class _ExactFinalLinearizationValidation(NamedTuple):
    success: jax.Array
    identity_valid: jax.Array
    orientation_valid: jax.Array
    factorization_residual: jax.Array
    factor_metadata_valid: jax.Array
    factorization_valid: jax.Array
    factorization_reconstruction_count: jax.Array


class _RetainedJacobianTransposeSolve(NamedTuple):
    solution: jax.Array
    correction: jax.Array
    residual: jax.Array
    condition_estimate: jax.Array
    payload_validation: _ExactFinalLinearizationValidation
    status: _LinearSolveStatus


_EXACT_FINAL_LINEARIZATION_ORIENTATION_J = np.int32(0)


def _device_scalar(value, *, dtype=jnp.float64):
    if isinstance(value, jax.Array) or hasattr(value, "aval"):
        return jnp.asarray(value, dtype=dtype)
    return jnp.asarray(np.asarray(value, dtype=np.dtype(dtype)))


def _device_int32(value, *, like=None):
    if like is not None:
        return _staged_like(like, value, dtype=jnp.int32)
    return runtime_device_put(value, dtype=jnp.int32)


def _cached_jit_linear_operator(fun, cache_attr, build_compiled):
    if not getattr(fun, _CACHEABLE_LINEAR_OPERATOR_ATTR, False):
        return build_compiled(fun)
    cached = getattr(fun, cache_attr, None)
    if cached is not None:
        return cached
    compiled = build_compiled(fun)
    with _JIT_LINEAR_OPERATOR_CACHE_LOCK:
        cached = getattr(fun, cache_attr, None)
        if cached is not None:
            return cached
        setattr(fun, cache_attr, compiled)
        return compiled


_DENSE_OPERATOR_CHUNK_BATCH_SIZE_ENV = "SIMSOPT_DENSE_OPERATOR_CHUNK_BATCH_SIZE"

_DENSE_OPERATOR_CHUNK_BATCH_SIZE_FALLBACK = 8

_DENSE_OPERATOR_CHUNK_BATCH_SIZE_MAX = 64

_DENSE_OPERATOR_DEFAULT_BUDGET_BYTES = 256 * 1024 * 1024

_DENSE_OPERATOR_LEGACY_BYTES_PER_PARALLEL_COLUMN = 32 * 1024 * 1024

_DENSE_OPERATOR_ACTIVATION_BYTES_PER_PARALLEL_COLUMN = 3072 * 1024 * 1024


def _dense_operator_chunk_batch_size_from_budget(max_dense_operator_bytes):
    if max_dense_operator_bytes is None:
        return _DENSE_OPERATOR_CHUNK_BATCH_SIZE_FALLBACK
    byte_budget = int(max_dense_operator_bytes)
    if byte_budget <= _DENSE_OPERATOR_DEFAULT_BUDGET_BYTES:
        return max(
            1,
            min(
                _DENSE_OPERATOR_CHUNK_BATCH_SIZE_FALLBACK,
                byte_budget // _DENSE_OPERATOR_LEGACY_BYTES_PER_PARALLEL_COLUMN,
            ),
        )
    return max(
        _DENSE_OPERATOR_CHUNK_BATCH_SIZE_FALLBACK,
        min(
            _DENSE_OPERATOR_CHUNK_BATCH_SIZE_MAX,
            byte_budget // _DENSE_OPERATOR_ACTIVATION_BYTES_PER_PARALLEL_COLUMN,
        ),
    )


def _resolve_dense_operator_chunk_batch_size():
    env_value = os.environ.get(_DENSE_OPERATOR_CHUNK_BATCH_SIZE_ENV)
    if env_value is not None:
        return max(1, int(env_value))
    policy = get_backend_policy()
    if policy.jax_platform not in {"cuda", "gpu"}:
        return _DENSE_OPERATOR_CHUNK_BATCH_SIZE_FALLBACK
    return _dense_operator_chunk_batch_size_from_budget(policy.max_dense_jacobian_bytes)


_DENSE_OPERATOR_CHUNK_BATCH_SIZE = _resolve_dense_operator_chunk_batch_size()

_SQUARE_OPERATOR_GMRES_REFINEMENT_STEPS = 1

# Exact Boozer Newton solves use the unsquared residual Jacobian. The general
# square-system path intentionally keeps a small restarted Krylov budget, but
# that restart is too short for the 255-variable native-scale Boozer state:
# the solve can stagnate even though the operator is finite and nonsingular.
# Permit up to two full Krylov cycles for small exact systems while keeping a
# fixed 256-vector cap for larger systems so the exact path remains matrix-free
# and memory-bounded.
_EXACT_NEWTON_GMRES_RESTART_CAP = 256
_EXACT_NEWTON_GMRES_MAXITER = 10

_EXACT_ADJOINT_DENSE_LU = os.environ.get(
    "SIMSOPT_EXACT_ADJOINT_DENSE_LU", "0"
).strip().lower() not in ("", "0", "false", "off", "no")


def _materialize_dense_linear_operator(linear_operator_fn, x):
    eye = jnp.eye(x.shape[0], dtype=x.dtype)
    # Assemble the dense operator in column batches rather than mapping all N basis
    # columns in parallel: numerically identical up to floating-point reduction
    # order (bit-exact for a linear Jacobian column; the Hessian's reducing HVP can
    # differ by ~1e-16 because batching reorders the reduction), with peak memory
    # bounded to batch_size parallel JVP/HVPs instead of N (each column is a full
    # BiotSavart JVP). Mirrors the chunked dense Boozer-Jacobian fix in
    # simsopt_jax_adapters/geo/boozer_surface.py (commit dcd70a2ae); without it the
    # dense linearization OOMs under XLA preallocation.
    cols = lax.map(
        lambda basis: linear_operator_fn(x, basis),
        eye,
        batch_size=_DENSE_OPERATOR_CHUNK_BATCH_SIZE,
    )
    return jnp.swapaxes(cols, 0, 1)


def _dense_jacobian_materialization_telemetry(
    residual: jax.Array,
    *,
    assembler: _DenseJacobianAssembler,
    dimension: int,
    batch_width: int,
) -> _DenseJacobianMaterializationTelemetry:
    tangent_batch_count = (dimension + batch_width - 1) // batch_width
    primal_traversal_count = (
        1
        if assembler is _DenseJacobianAssembler.LINEARIZE_ONCE
        else 1 + tangent_batch_count
    )
    tail_width = dimension % batch_width
    return _DenseJacobianMaterializationTelemetry(
        assembler_code=_staged_like(residual, int(assembler), dtype=jnp.int32),
        residual_evaluation_count=_staged_like(
            residual,
            1,
            dtype=jnp.int32,
        ),
        primal_traversal_count=_staged_like(
            residual,
            primal_traversal_count,
            dtype=jnp.int32,
        ),
        tangent_batch_count=_staged_like(
            residual,
            tangent_batch_count,
            dtype=jnp.int32,
        ),
        tangent_direction_count=_staged_like(
            residual,
            dimension,
            dtype=jnp.int32,
        ),
        batch_width=_staged_like(residual, batch_width, dtype=jnp.int32),
        tail_width=_staged_like(residual, tail_width, dtype=jnp.int32),
    )


def _linearize_and_materialize_dense_square_jacobian(
    residual_fn: Callable[[jax.Array], jax.Array],
    x: jax.Array,
    *,
    assembler: _DenseJacobianAssembler = _DenseJacobianAssembler.LINEARIZE_ONCE,
    batch_width: int | None = None,
    jacobian_construction_phase: PhaseId | None = None,
    dense_materialization_phase: PhaseId | None = None,
) -> _DenseJacobianMaterialization:
    """Materialize one square Jacobian using a statically selected assembler."""

    x = jnp.asarray(x)
    if not isinstance(assembler, _DenseJacobianAssembler):
        raise TypeError(
            "assembler must be a _DenseJacobianAssembler, got "
            f"{type(assembler).__name__}"
        )
    configured_batch_width = _resolve_dense_operator_batch_width(
        batch_width,
        dimension=int(x.shape[0]),
    )
    if assembler is _DenseJacobianAssembler.LINEARIZE_ONCE:
        with (
            device_scope(jacobian_construction_phase)
            if jacobian_construction_phase is not None
            else nullcontext()
        ):
            residual, linearized_fn = jax.linearize(residual_fn, x)
        with (
            device_scope(dense_materialization_phase)
            if dense_materialization_phase is not None
            else nullcontext()
        ):
            jacobian = _dense_square_operator_matrix(
                linearized_fn,
                residual,
                batch_width=configured_batch_width,
            )
    else:
        with (
            device_scope(jacobian_construction_phase)
            if jacobian_construction_phase is not None
            else nullcontext()
        ):
            residual = residual_fn(x)
        checkpointed_residual_fn = jax.checkpoint(
            residual_fn,
            policy=jax.checkpoint_policies.nothing_saveable,
            prevent_cse=False,
        )

        def checkpointed_jvp(tangent: jax.Array) -> jax.Array:
            return jax.jvp(
                checkpointed_residual_fn,
                (x,),
                (tangent,),
            )[1]

        with (
            device_scope(dense_materialization_phase)
            if dense_materialization_phase is not None
            else nullcontext()
        ):
            jacobian = _dense_square_operator_matrix(
                checkpointed_jvp,
                residual,
                batch_width=configured_batch_width,
            )
    return _DenseJacobianMaterialization(
        residual=residual,
        jacobian=jacobian,
        telemetry=_dense_jacobian_materialization_telemetry(
            residual,
            assembler=assembler,
            dimension=int(x.shape[0]),
            batch_width=configured_batch_width,
        ),
    )


def _hessian_vector_product_fn(objective_fn):
    def build_compiled(fn):
        grad_fn = jax.grad(fn, argnums=0)

        def hvp(x, v, *fn_args):
            def grad_for_x(x_inner):
                return grad_fn(x_inner, *fn_args)

            return jax.jvp(grad_for_x, (x,), (v,))[1]

        # Dense assembly stages this HVP inside an outer scan. Inlining exposes
        # objective closure arrays to that trace so they can remain operands,
        # rather than becoming device-backed constants in a nested call.
        return jax.jit(hvp, inline=True)

    return _cached_jit_linear_operator(objective_fn, _CACHED_HVP_ATTR, build_compiled)


def _jacobian_vector_product_fn(residual_fn):
    def build_compiled(fn):
        return jax.jit(lambda x, v: jax.jvp(fn, (x,), (v,))[1])

    return _cached_jit_linear_operator(residual_fn, _CACHED_JVP_ATTR, build_compiled)


def _materialize_dense_hessian(hvp_fn, x, *, symmetrize=True):
    dense = _materialize_dense_linear_operator(hvp_fn, x)
    if not bool(symmetrize):
        return dense
    upper = jnp.triu(dense)
    return upper + jnp.triu(dense, 1).T


def _symmetrize_dense_hessian(dense: jax.Array) -> jax.Array:
    upper = jnp.triu(dense)
    return upper + jnp.triu(dense, 1).T


def _materialize_dense_hessian_host(hvp_fn, x, *, symmetrize=True):
    x_array = jnp.asarray(x)
    dtype = np.dtype(x_array.dtype)
    dimension = int(x_array.shape[0])
    dense_host = np.empty((dimension, dimension), dtype=dtype)
    basis = np.zeros(dimension, dtype=dtype)
    for column_index in range(dimension):
        basis[column_index] = 1
        basis_vector = jnp.asarray(basis, dtype=x_array.dtype)
        column = host_array(hvp_fn(x_array, basis_vector), dtype=dtype)
        dense_host[:, column_index] = column
        basis[column_index] = 0
    if not bool(symmetrize):
        return jnp.asarray(dense_host, dtype=x_array.dtype)
    upper = np.triu(dense_host)
    return jnp.asarray(upper + np.triu(dense_host, 1).T, dtype=x_array.dtype)


def _materialize_dense_jacobian(jvp_fn, x):
    return _materialize_dense_linear_operator(jvp_fn, x)


def _factor_dense_hessian(H, *, optimizer_backend):
    """Factor a dense LS Hessian once and return packed ``(lu, piv)``.

    Per ``docs/parity_scientific_equivalence_contract_2026-05-09.md`` §5.3
    (Phase 2 adjoint factor-once hybrid). The resulting factors are reused
    for both forward and adjoint solves so the bytes are bit-identical by
    construction.

    The ``optimizer_backend == "scipy"`` branch routes through host LAPACK
    ``dgetrf`` via ``scipy.linalg.lu_factor`` so the LS reference lane keeps
    matching CPU pivot tie-breaks. All other backends call
    ``jax.scipy.linalg.lu_factor`` on ``H``'s device, which dispatches to
    LAPACK on CPU and cuSOLVER ``getrf`` on CUDA. Both APIs use the same
    0-indexed packed pivot semantics, so the returned ``(lu, piv)`` is a
    drop-in to ``jax.scipy.linalg.lu_solve``.
    """
    if H is None:
        return None
    if optimizer_backend == "scipy":
        H_host = np.asarray(H, dtype=np.float64)
        lu_host, piv_host = scipy.linalg.lu_factor(H_host)
        lu = jnp.asarray(lu_host, dtype=H.dtype)
        piv = jnp.asarray(piv_host, dtype=jnp.int32)
        return lu, piv
    return jsp_linalg.lu_factor(H)


def _lu_solve_dense_hessian(lu_piv, rhs, *, transpose):
    """Solve a dense LS Hessian system from packed ``(lu, piv)`` factors.

    Routes through ``jax.scipy.linalg.lu_solve`` with ``trans=1`` for the
    transpose path so adjoint and forward solves consume the same packed
    factor bytes. Pivot reconstruction stays inside the LAPACK/cuSOLVER
    contract; no manual ``_piv_from(P)`` rebuilding happens at the call
    site.
    """
    lu, piv = lu_piv
    trans = 1 if transpose else 0
    return jsp_linalg.lu_solve((lu, piv), rhs, trans=trans)


@jax.jit
def _plu_from_lu_piv(lu_piv):
    """Derive ``(P, L, U)`` matrices from packed ``(lu, piv)`` factors.

    Used for backward-compatible reporting under the
    ``"dense-plu-shared"`` factorization backend: the ``res["PLU"]`` slot
    keeps surfacing the public triple while the runtime forward and
    adjoint solves consume the same ``(lu, piv)`` factor bytes. The
    permutation array is built with ``lax.fori_loop`` so the helper is
    JIT-traceable; the ``jax.jit`` wrapper hoists the static-shape
    ``jnp.eye`` / ``jnp.zeros`` constructors inside the trace so callers
    in strict transfer-guard contexts do not pay a host roundtrip per
    invocation.
    """
    lu, piv = lu_piv
    n = lu.shape[0]
    eye = jnp.eye(n, dtype=lu.dtype)
    L = jnp.tril(lu, k=-1) + eye
    U = jnp.triu(lu)

    def body(i, perm):
        a = perm[i]
        b = perm[piv[i]]
        perm = perm.at[i].set(b)
        perm = perm.at[piv[i]].set(a)
        return perm

    perm = lax.fori_loop(0, n, body, jnp.arange(n, dtype=jnp.int32))
    columns = jnp.arange(n, dtype=jnp.int32)
    P = (
        jnp.zeros((n, n), dtype=lu.dtype)
        .at[perm, columns]
        .set(jnp.asarray(1.0, dtype=lu.dtype))
    )
    return P, L, U


def _normalized_plu_factorization_residual(jacobian, lu_piv):
    """Return ``||J - P L U||_F / max(||J||_F, tiny)``."""

    jacobian = jnp.asarray(jacobian)
    P, L, U = _plu_from_lu_piv(lu_piv)
    reconstructed = P @ L @ U
    denominator = jnp.maximum(
        jnp.linalg.norm(jacobian),
        _device_scalar(jnp.finfo(jacobian.dtype).tiny, dtype=jacobian.dtype),
    )
    return jnp.linalg.norm(jacobian - reconstructed) / denominator


def _plu_factorization_residual_success(jacobian, residual):
    jacobian = jnp.asarray(jacobian)
    residual = jnp.asarray(residual, dtype=jacobian.dtype)
    dimension = int(jacobian.shape[0])
    threshold = _device_scalar(
        64.0 * dimension * jnp.finfo(jacobian.dtype).eps,
        dtype=jacobian.dtype,
    )
    return jnp.isfinite(residual) & (residual <= threshold)


def _build_exact_final_linearization_inputs(
    *,
    solved_state,
    coil_dofs,
    coil_dynamic_inputs,
    residual_configuration,
) -> _ExactFinalLinearizationInputs:
    """Mint the only atomic state/coil input accepted by the payload producer."""

    return _mint_exact_final_linearization_inputs(
        solved_state=jnp.asarray(solved_state),
        coil_dofs=jnp.asarray(coil_dofs),
        coil_dynamic_inputs=tuple(jnp.asarray(v) for v in coil_dynamic_inputs),
        residual_configuration=tuple(jnp.asarray(v) for v in residual_configuration),
    )


def _build_exact_final_linearization(
    inputs: _ExactFinalLinearizationInputs,
    *,
    residual,
    jacobian,
    lu_piv,
    producer_solve_success,
) -> _ExactFinalLinearization:
    """Bind one final-state ``J`` and its packed factors to atomic inputs."""

    residual = jnp.asarray(residual)
    jacobian = jnp.asarray(jacobian)
    lu, pivots = lu_piv
    orientation_code = _device_int32(
        _EXACT_FINAL_LINEARIZATION_ORIENTATION_J,
        like=jacobian,
    )
    producer_solve_success = jnp.asarray(producer_solve_success, dtype=jnp.bool_)
    identity = _mint_exact_final_linearization_identity(
        inputs=inputs,
        residual=residual,
        jacobian=jacobian,
        orientation_code=orientation_code,
        producer_solve_success=producer_solve_success,
    )
    return _mint_exact_final_linearization(
        inputs=inputs,
        residual=residual,
        jacobian=jacobian,
        lu=jnp.asarray(lu),
        pivots=jnp.asarray(pivots, dtype=jnp.int32),
        orientation_code=orientation_code,
        producer_solve_success=producer_solve_success,
        identity=identity,
    )


def _validate_exact_final_linearization(
    payload: _ExactFinalLinearization,
) -> _ExactFinalLinearizationValidation:
    """Recompute device certificates from the payload's live fields."""

    def exact_tree_equal(left, right):
        left_leaves, left_structure = jax.tree_util.tree_flatten(left)
        right_leaves, right_structure = jax.tree_util.tree_flatten(right)
        if left_structure != right_structure or len(left_leaves) != len(right_leaves):
            return jnp.asarray(False)
        equal = jnp.asarray(True)
        for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
            if (
                left_leaf.shape != right_leaf.shape
                or left_leaf.dtype != right_leaf.dtype
            ):
                return jnp.asarray(False)
            equal = equal & jnp.array_equal(left_leaf, right_leaf)
        return equal

    identity_valid = (
        exact_tree_equal(payload.inputs, payload.identity.inputs)
        & exact_tree_equal(payload.residual, payload.identity.residual)
        & exact_tree_equal(payload.jacobian, payload.identity.jacobian)
        & exact_tree_equal(
            payload.orientation_code,
            payload.identity.orientation_code,
        )
        & exact_tree_equal(
            payload.producer_solve_success,
            payload.identity.producer_solve_success,
        )
    )
    orientation_valid = payload.orientation_code == _device_int32(
        _EXACT_FINAL_LINEARIZATION_ORIENTATION_J,
        like=payload.jacobian,
    )
    factorization_residual = _normalized_plu_factorization_residual(
        payload.jacobian,
        (payload.lu, payload.pivots),
    )
    factor_metadata_valid = jnp.asarray(
        payload.lu.shape == payload.jacobian.shape
        and payload.pivots.shape == (payload.jacobian.shape[0],)
        and payload.lu.dtype == payload.jacobian.dtype
        and payload.pivots.dtype == np.dtype(np.int32)
    )
    factorization_valid = factor_metadata_valid & _plu_factorization_residual_success(
        payload.jacobian,
        factorization_residual,
    )
    success = (
        payload.producer_solve_success
        & identity_valid
        & orientation_valid
        & factorization_valid
    )
    return _ExactFinalLinearizationValidation(
        success=success,
        identity_valid=identity_valid,
        orientation_valid=orientation_valid,
        factorization_residual=factorization_residual,
        factor_metadata_valid=factor_metadata_valid,
        factorization_valid=factorization_valid,
        factorization_reconstruction_count=_device_int32(
            1,
            like=payload.jacobian,
        ),
    )


def _gmres_iteration_limits(n):
    restart = max(5, min(n, 64))
    maxiter = 10
    return restart, maxiter


def _exact_newton_gmres_iteration_limits(n):
    dimension = int(n)
    restart = max(5, min(dimension, _EXACT_NEWTON_GMRES_RESTART_CAP))
    maxiter = (
        2
        if dimension <= _EXACT_NEWTON_GMRES_RESTART_CAP
        else _EXACT_NEWTON_GMRES_MAXITER
    )
    return restart, maxiter


def _run_operator_gmres(matvec, rhs, *, tol, restart=None, maxiter=None):
    n = rhs.shape[0]
    default_restart, default_maxiter = _gmres_iteration_limits(n)
    restart = default_restart if restart is None else int(restart)
    maxiter = default_maxiter if maxiter is None else int(maxiter)
    # JAX's gmres implementation currently lowers a few scalar literals through
    # host-to-device conversions even when the caller provides fully device-
    # resident operands. Keep the allowance scoped to the library call so the
    # surrounding operator path remains strict-transfer clean.
    with jax.transfer_guard_host_to_device("allow"):
        return gmres(
            matvec,
            rhs,
            tol=tol,
            atol=0.0,
            restart=restart,
            maxiter=maxiter,
            # JAX documents the incremental method as numerically stabler than the
            # default batched variant, which matters more than lower GPU overhead
            # on the checked operator-only runtime path.
            solve_method="incremental",
        )


def _counted_gmres_norm(vector):
    result = jnp.vdot(
        vector.real,
        vector.real,
        precision=lax.Precision.HIGHEST,
    )
    if jnp.iscomplexobj(vector):
        result = result + jnp.vdot(
            vector.imag,
            vector.imag,
            precision=lax.Precision.HIGHEST,
        )
    return jnp.sqrt(result)


def _counted_gmres_safe_normalize(vector, *, threshold=None):
    norm = _counted_gmres_norm(vector)
    if threshold is None:
        threshold = jnp.asarray(jnp.finfo(norm.dtype).eps, dtype=norm.dtype)
    threshold = jnp.asarray(threshold, dtype=vector.dtype).real
    use_norm = norm > threshold
    norm_cast = lax.convert_element_type(norm, vector.dtype)
    normalized = jnp.where(use_norm, vector / norm_cast, 0.0)
    return normalized, jnp.where(use_norm, norm, 0.0)


def _counted_gmres_iterative_classical_gram_schmidt(
    basis,
    vector,
    vector_norm,
):
    """Match JAX 0.10's two-pass iterative classical Gram-Schmidt."""

    overlaps = jnp.zeros(basis.shape[-1], dtype=basis.dtype)
    scaled_vector_norm = vector_norm / jnp.sqrt(2.0)

    def body(carry):
        iteration, current, accumulated, scaled_current_norm = carry
        projection = jnp.einsum(
            "...n,...->n",
            basis.conj(),
            current,
            precision=lax.Precision.HIGHEST,
        )
        current = current - jnp.dot(
            basis,
            projection,
            precision=lax.Precision.HIGHEST,
        )
        accumulated = accumulated + projection

        def calculate_norm(norm_carry):
            norm_iteration, _, norm_vector, _ = norm_carry
            _, current_norm = _counted_gmres_safe_normalize(norm_vector)
            return (
                norm_iteration,
                jnp.asarray(False),
                norm_vector,
                current_norm / jnp.sqrt(2.0),
            )

        _, _, current, scaled_current_norm = lax.while_loop(
            lambda norm_carry: norm_carry[1] & (norm_carry[0] < 1),
            calculate_norm,
            (iteration, jnp.asarray(True), current, scaled_current_norm),
        )
        return iteration + 1, current, accumulated, scaled_current_norm

    def continue_orthogonalization(carry):
        iteration, _, accumulated, scaled_current_norm = carry
        _, overlap_norm = _counted_gmres_safe_normalize(accumulated)
        return (iteration < 1) & (overlap_norm < scaled_current_norm)

    state = body((0, vector, overlaps, scaled_vector_norm))
    _, vector, overlaps, _ = lax.while_loop(
        continue_orthogonalization,
        body,
        state,
    )
    return vector, overlaps


def _counted_gmres_rotate(vector, index, cosine, sine):
    first = vector[index]
    second = vector[index + 1]
    vector = vector.at[index].set(cosine.conj() * first - sine.conj() * second)
    return vector.at[index + 1].set(sine * first + cosine * second)


def _counted_gmres_givens_rotation(first, second):
    second_zero = abs(second) == 0
    first_lt_second = abs(first) < abs(second)
    ratio = -jnp.where(first_lt_second, first, second) / jnp.where(
        first_lt_second,
        second,
        first,
    )
    scale = lax.rsqrt(1 + abs(ratio) ** 2).astype(ratio.dtype)
    cosine = jnp.where(
        second_zero,
        1,
        jnp.where(first_lt_second, scale * ratio, scale),
    )
    sine = jnp.where(
        second_zero,
        0,
        jnp.where(first_lt_second, scale, scale * ratio),
    )
    return cosine, sine


def _counted_gmres_apply_givens_rotations(hessenberg_row, givens, index):
    rotated = lax.fori_loop(
        0,
        index,
        lambda rotation_index, row: _counted_gmres_rotate(
            row,
            rotation_index,
            *givens[rotation_index, :],
        ),
        hessenberg_row,
    )
    factors = _counted_gmres_givens_rotation(
        rotated[index],
        rotated[index + 1],
    )
    givens = givens.at[index, :].set(jnp.asarray(factors))
    return _counted_gmres_rotate(rotated, index, *factors), givens


def _counted_gmres_restart(
    matvec,
    rhs,
    x0,
    unit_residual,
    residual_norm,
    ptol,
    restart,
):
    """Match one JAX 0.10 incremental-GMRES restart and return its work."""

    basis = jnp.pad(unit_residual[..., None], ((0, 0), (0, restart)))
    dtype = rhs.dtype
    hessenberg = jnp.eye(restart, restart + 1, dtype=dtype)
    givens = jnp.zeros((restart, 2), dtype=dtype)
    beta = jnp.zeros((restart + 1), dtype=dtype).at[0].set(residual_norm.astype(dtype))

    def arnoldi_qr_step(carry):
        index, _, current_basis, current_hessenberg, current_beta, rotations = carry
        vector = matvec(current_basis[..., index])
        _, initial_norm = _counted_gmres_safe_normalize(vector)
        vector, overlaps = _counted_gmres_iterative_classical_gram_schmidt(
            current_basis,
            vector,
            initial_norm,
        )
        unit_vector, final_norm = _counted_gmres_safe_normalize(
            vector,
            threshold=jnp.finfo(dtype).eps * initial_norm,
        )
        current_basis = current_basis.at[..., index + 1].set(unit_vector)
        overlaps = overlaps.at[index + 1].set(final_norm.astype(dtype))
        current_hessenberg = current_hessenberg.at[index, :].set(overlaps)
        row, rotations = _counted_gmres_apply_givens_rotations(
            current_hessenberg[index, :],
            rotations,
            index,
        )
        current_hessenberg = current_hessenberg.at[index, :].set(row)
        current_beta = _counted_gmres_rotate(
            current_beta,
            index,
            *rotations[index, :],
        )
        return (
            index + 1,
            abs(current_beta[index + 1]),
            current_basis,
            current_hessenberg,
            current_beta,
            rotations,
        )

    state = lax.while_loop(
        lambda carry: (carry[0] < restart) & (carry[1] > ptol),
        arnoldi_qr_step,
        (0, residual_norm, basis, hessenberg, beta, givens),
    )
    arnoldi_iterations, _, basis, hessenberg, beta, _ = state
    coefficients = jsp_linalg.solve_triangular(
        hessenberg[:, :-1].T,
        beta[:-1],
    )
    update = jnp.dot(
        basis[..., :-1],
        coefficients,
        precision=lax.Precision.HIGHEST,
    )
    solution = x0 + update
    residual = rhs - matvec(solution)
    unit_residual, residual_norm = _counted_gmres_safe_normalize(residual)
    return solution, unit_residual, residual_norm, arnoldi_iterations


def _run_operator_gmres_counted_incremental(
    matvec,
    rhs,
    *,
    tol,
    restart=None,
    maxiter=None,
):
    """Run the pinned incremental GMRES algorithm with device-only counts."""

    dimension = rhs.shape[0]
    default_restart, default_maxiter = _gmres_iteration_limits(dimension)
    restart = min(
        default_restart if restart is None else int(restart),
        int(rhs.size),
    )
    maxiter = default_maxiter if maxiter is None else int(maxiter)
    rhs_norm = _counted_gmres_norm(rhs)
    absolute_tolerance = jnp.maximum(tol * rhs_norm, 0.0)
    preconditioned_tolerance = rhs_norm * jnp.minimum(
        1.0,
        absolute_tolerance / rhs_norm,
    )
    x0 = jnp.zeros_like(rhs)

    def solve(operator, solve_rhs):
        residual = solve_rhs - operator(x0)
        unit_residual, residual_norm = _counted_gmres_safe_normalize(residual)
        zero = _staged_like(solve_rhs, 0, dtype=jnp.int32)
        one = _staged_like(solve_rhs, 1, dtype=jnp.int32)

        def restart_body(state):
            solution, cycle, current_unit_residual, current_norm, applications = state
            solution, current_unit_residual, current_norm, arnoldi_iterations = (
                _counted_gmres_restart(
                    operator,
                    solve_rhs,
                    solution,
                    current_unit_residual,
                    current_norm,
                    preconditioned_tolerance,
                    restart,
                )
            )
            return (
                solution,
                cycle + one,
                current_unit_residual,
                current_norm,
                applications + arnoldi_iterations.astype(jnp.int32) + one,
            )

        solution, _, _, _, applications = lax.while_loop(
            lambda state: (state[1] < maxiter) & (state[3] > absolute_tolerance),
            restart_body,
            (x0, zero, unit_residual, residual_norm, one),
        )
        return solution, _CountedIncrementalGmresTelemetry(
            linear_operator_application_count=applications,
        )

    with jax.transfer_guard_host_to_device("allow"):
        solution, telemetry = lax.custom_linear_solve(
            matvec,
            rhs,
            solve=solve,
            transpose_solve=solve,
            has_aux=True,
        )
    failed = jnp.isnan(_counted_gmres_norm(solution))
    info = jnp.where(failed, -1, 0)
    return solution, info, telemetry


def _gmres_solve_array_system(matvec, rhs, *, tol):
    solution, info = _run_operator_gmres(matvec, rhs, tol=tol)
    residual = rhs - matvec(solution)
    return solution, residual, info


def _linear_solve_finite(solution, residual):
    return jnp.all(jnp.isfinite(solution)) & jnp.all(jnp.isfinite(residual))


_DENSE_LINEAR_SOLVE_RESIDUAL_DIMENSION_FACTOR = 64.0

_DENSE_LINEAR_SOLVE_SMALL_SOLUTION_FACTOR = 100.0

_FLOAT64_DENSE_MATRIX_MAX_CONDITION_ESTIMATE = 1.0e12


def _effective_linear_solve_tolerance(rhs, tol):
    dtype = rhs.dtype
    policy = get_backend_policy()
    tol_value = _optimizer_scalar(tol, dtype=dtype)
    tolerance_floor = _device_scalar(
        policy.linear_solve_tolerance_floor,
        dtype=dtype,
    )
    tolerance_cap = (
        _device_scalar(jnp.inf, dtype=dtype)
        if policy.linear_solve_tolerance_cap is None
        else _device_scalar(policy.linear_solve_tolerance_cap, dtype=dtype)
    )
    return jnp.minimum(tolerance_cap, jnp.maximum(tolerance_floor, tol_value))


def _effective_dense_backward_error_tolerance(rhs, tol):
    """Return the dense-LU backward-error tolerance, including its ``n * eps`` floor."""
    dtype = rhs.dtype
    policy = get_backend_policy()
    effective_tolerance = _effective_linear_solve_tolerance(rhs, tol)
    dimension_floor = (
        _device_scalar(_DENSE_LINEAR_SOLVE_RESIDUAL_DIMENSION_FACTOR, dtype=dtype)
        * _device_scalar(rhs.shape[0], dtype=dtype)
        * _device_scalar(jnp.finfo(dtype).eps, dtype=dtype)
    )
    tolerance_cap = (
        _device_scalar(jnp.inf, dtype=dtype)
        if policy.linear_solve_tolerance_cap is None
        else _device_scalar(policy.linear_solve_tolerance_cap, dtype=dtype)
    )
    return jnp.minimum(
        tolerance_cap,
        jnp.maximum(effective_tolerance, dimension_floor),
    )


def _linear_solve_residual_scale(rhs):
    dtype = rhs.dtype
    rhs_norm = jnp.linalg.norm(rhs)
    eps = _device_scalar(jnp.finfo(dtype).eps, dtype=dtype)
    return jnp.maximum(rhs_norm, eps)


def _linear_solve_residual_tolerance(rhs, tol):
    return _effective_linear_solve_tolerance(rhs, tol) * _linear_solve_residual_scale(
        rhs
    )


def _linear_solve_status_success(status):
    return status.success


def _linear_solve_effective_tolerance_reached(status):
    """Return whether a complete status meets the attainable policy gate."""
    return (
        status.success
        & jnp.isfinite(status.residual_relative)
        & jnp.isfinite(status.effective_tolerance)
        & (status.residual_relative <= status.effective_tolerance)
    )


def _linear_solve_solution_or_nan(solution, status):
    return jax.lax.cond(
        jnp.asarray(_linear_solve_status_success(status), dtype=jnp.bool_),
        lambda value: value,
        lambda value: jax.tree.map(lambda leaf: jnp.full_like(leaf, jnp.nan), value),
        solution,
    )


def _linear_solve_iteration_count(info):
    if info is None:
        return _device_int32(_LINEAR_SOLVE_ITERATIONS_UNKNOWN)
    return _device_int32(info)


def _linear_solve_status_iterations(iterations):
    if isinstance(iterations, jax.Array) or hasattr(iterations, "aval"):
        return jnp.asarray(iterations, dtype=jnp.int32)
    return _device_int32(iterations)


def _combine_linear_solve_iteration_counts(*counts):
    counts = tuple(_linear_solve_status_iterations(count) for count in counts)
    all_known = counts[0] >= _device_int32(0)
    for iteration_count in counts[1:]:
        all_known = all_known & (iteration_count >= _device_int32(0))
    total = sum(counts, _device_int32(0))
    return lax.cond(
        all_known,
        lambda _: total,
        lambda _: _device_int32(_LINEAR_SOLVE_ITERATIONS_UNKNOWN),
        operand=None,
    )


def _linear_solve_iterations_host_value(iterations):
    value = host_int(iterations)
    if value == _LINEAR_SOLVE_ITERATIONS_UNKNOWN:
        return None
    return value


def _linear_solve_status(solution, residual, rhs, *, tol, iterations):
    residual_norm = jnp.linalg.norm(residual)
    residual_scale = _linear_solve_residual_scale(rhs)
    residual_relative = residual_norm / residual_scale
    requested_tolerance = _optimizer_scalar(tol, dtype=rhs.dtype)
    effective_tolerance = _effective_linear_solve_tolerance(rhs, tol)
    zero_count = _device_int32(0, like=rhs)
    success = (
        _linear_solve_finite(solution, residual)
        & jnp.isfinite(residual_norm)
        & jnp.isfinite(residual_relative)
        & (residual_relative <= effective_tolerance)
    )
    return _LinearSolveStatus(
        success=success,
        residual=residual_norm,
        residual_scale=residual_scale,
        residual_relative=residual_relative,
        requested_tolerance=requested_tolerance,
        effective_tolerance=effective_tolerance,
        iterations=_linear_solve_status_iterations(iterations),
        dense_materialization_count=zero_count,
        lu_factorization_count=zero_count,
        lu_solve_count=zero_count,
        refinement_correction_count=zero_count,
    )


def _complete_linear_solve_status(status, rhs, *, tol):
    """Complete legacy status metrics before entering a staged branch."""
    rhs = jnp.asarray(rhs)
    residual_scale = (
        _linear_solve_residual_scale(rhs)
        if status.residual_scale is None
        else status.residual_scale
    )
    requested_tolerance = (
        _optimizer_scalar(tol, dtype=rhs.dtype)
        if status.requested_tolerance is None
        else status.requested_tolerance
    )
    effective_tolerance = (
        _effective_linear_solve_tolerance(rhs, tol)
        if status.effective_tolerance is None
        else status.effective_tolerance
    )
    return status._replace(
        residual_scale=residual_scale,
        requested_tolerance=requested_tolerance,
        effective_tolerance=effective_tolerance,
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


def _terminal_linear_solve_status(status, rhs, *, tol):
    """Complete status and apply the attainable solver authority."""
    completed = _complete_linear_solve_status(status, rhs, tol=tol)
    return completed._replace(
        success=_linear_solve_effective_tolerance_reached(completed)
    )


def _dense_linear_solve_status(matvec, solution, rhs, *, tol):
    residual = rhs - matvec(solution)
    return _linear_solve_status(
        solution,
        residual,
        rhs,
        tol=tol,
        iterations=_device_int32(0),
    )


def _dense_matrix_backward_error_success(matrix, solution, rhs, *, tol):
    residual = rhs - matrix @ solution
    dtype = rhs.dtype
    residual_norm = jnp.linalg.norm(residual)
    scale = jnp.linalg.norm(matrix) * jnp.linalg.norm(solution) + jnp.linalg.norm(rhs)
    eps = _device_scalar(jnp.finfo(dtype).eps, dtype=dtype)
    threshold = _effective_dense_backward_error_tolerance(rhs, tol) * jnp.maximum(
        scale,
        eps,
    )
    return (
        _linear_solve_finite(solution, residual)
        & jnp.isfinite(residual_norm)
        & jnp.isfinite(scale)
        & (residual_norm <= threshold)
    )


def _relative_residual_norm(residual, rhs, *, ord=None):
    """Return ``||residual|| / max(||rhs||, eps_runtime)``."""
    dtype = rhs.dtype
    residual_norm = jnp.linalg.norm(residual, ord=ord)
    rhs_norm = jnp.linalg.norm(rhs, ord=ord)
    eps = _device_scalar(jnp.finfo(dtype).eps, dtype=dtype)
    return residual_norm / jnp.maximum(rhs_norm, eps)


def _relative_residual_1_norm(residual, rhs):
    return _relative_residual_norm(residual, rhs, ord=1)


def _forward_error_bound(residual_rel, condition_estimate):
    dtype = residual_rel.dtype
    one = _device_scalar(1.0, dtype=dtype)
    inf_value = _device_scalar(jnp.inf, dtype=dtype)
    scaled = condition_estimate * residual_rel
    denominator = one - scaled
    return jnp.where(
        denominator > _device_scalar(0.0, dtype=dtype),
        scaled / denominator,
        inf_value,
    )


def _forward_error_tolerance(*, tol, dtype):
    """Return the shared relative forward-accuracy acceptance threshold."""
    dtype = np.dtype(dtype)
    accuracy_policy = mixed_dense_ir_accuracy_policy()
    tol_value = _optimizer_scalar(tol, dtype=dtype)
    floor = jnp.sqrt(_device_scalar(jnp.finfo(dtype).eps, dtype=dtype))
    return jnp.maximum(
        floor,
        _device_scalar(
            accuracy_policy.forward_error_tolerance_multiplier,
            dtype=dtype,
        )
        * tol_value,
    )


def _forward_error_success(residual_rel, condition_estimate, *, tol):
    dtype = residual_rel.dtype
    gate = _forward_error_tolerance(tol=tol, dtype=dtype)
    ferr = _forward_error_bound(residual_rel, condition_estimate)
    return jnp.isfinite(ferr) & (ferr <= gate)


def _matrix_one_norm(matrix):
    return jnp.max(jnp.sum(jnp.abs(matrix), axis=0))


def _place_like_concrete_array(value, reference, *, dtype=None):
    if isinstance(value, jax.core.Tracer) or isinstance(reference, jax.core.Tracer):
        return jnp.asarray(value, dtype=dtype)
    resolved_dtype = getattr(value, "dtype", None) if dtype is None else dtype
    if resolved_dtype is None:
        resolved_dtype = np.asarray(value).dtype
    return _explicit_device_array(
        value,
        dtype=resolved_dtype,
        reference=reference,
    )


def _place_like_concrete_scalar(value, reference):
    return _place_like_concrete_array(value, reference, dtype=reference.dtype)


def _hager_higham_inverse_1_norm_estimate(
    solve,
    transpose_solve,
    *,
    size,
    dtype,
    placement_reference,
    iterations=_HAGER_HIGHAM_CONDITION_ITERATIONS,
    return_solve_count=False,
):
    """Estimate ``||A^-1||_1`` and optionally return factor applications."""
    one = _staged_like(placement_reference, 1.0, dtype=dtype)
    zero = _staged_like(placement_reference, 0.0, dtype=dtype)
    inf_value = _staged_like(placement_reference, np.inf, dtype=dtype)
    size_scalar = _staged_like(placement_reference, size, dtype=dtype)
    indices = _staged_like(
        placement_reference,
        np.arange(size, dtype=np.int32),
        dtype=np.int32,
    )
    x0 = _staged_like(
        placement_reference,
        np.ones((size,), dtype=np.dtype(dtype)),
        dtype=dtype,
    )
    x0 = x0 / size_scalar

    def unit_vector(index):
        return jnp.where(indices == index, one, zero)

    def body_fun(_iteration, state):
        x, best_estimate, solve_count = state
        y = solve(x)
        estimate = jnp.sum(jnp.abs(y))
        signs = jnp.where(y >= zero, one, -one)
        z = transpose_solve(signs)
        next_index = jnp.argmax(jnp.abs(z))
        next_x = unit_vector(next_index)
        finite = jnp.all(jnp.isfinite(y)) & jnp.all(jnp.isfinite(z))
        next_estimate = jnp.maximum(best_estimate, estimate)
        return (
            next_x,
            jnp.where(finite, next_estimate, inf_value),
            solve_count + _device_int32(2, like=placement_reference),
        )

    # ``lax.fori_loop`` lowers the Python integer bounds through a weakly
    # typed host-to-device conversion that strict transfer-guard contexts
    # flag as a violation. Mirror the ``_run_operator_gmres`` allowance:
    # scope the relaxation to the library call so the surrounding solve
    # path stays strict-transfer clean.
    with jax.transfer_guard_host_to_device("allow"):
        _, estimate, solve_count = lax.fori_loop(
            0,
            int(iterations),
            body_fun,
            (x0, zero, _device_int32(0, like=placement_reference)),
        )
    if return_solve_count:
        return estimate, solve_count
    return estimate


def _dense_matrix_condition_estimate_with_telemetry(
    matrix,
    *,
    lu_piv=None,
    transpose_operator=False,
):
    """Estimate ``cond_1(J)`` or ``cond_1(J^T)`` from factors of ``J``."""

    matrix = jnp.asarray(matrix)
    size = int(matrix.shape[0])
    factorization_count = _device_int32(0, like=matrix)

    if lu_piv is None:
        lu_piv = jsp_linalg.lu_factor(matrix)
        factorization_count = _device_int32(1, like=matrix)
    lu, piv = lu_piv

    if transpose_operator:

        def solve(rhs):
            return jsp_linalg.lu_solve((lu, piv), rhs, trans=1)

        def transpose_solve(rhs):
            return jsp_linalg.lu_solve((lu, piv), rhs, trans=0)

        matrix_norm = jnp.max(jnp.sum(jnp.abs(matrix), axis=1))
    else:

        def solve(rhs):
            return jsp_linalg.lu_solve((lu, piv), rhs, trans=0)

        def transpose_solve(rhs):
            return jsp_linalg.lu_solve((lu, piv), rhs, trans=1)

        matrix_norm = _matrix_one_norm(matrix)
    inverse_norm, solve_count = _hager_higham_inverse_1_norm_estimate(
        solve,
        transpose_solve,
        size=size,
        dtype=matrix.dtype,
        placement_reference=lu,
        return_solve_count=True,
    )
    inverse_norm = _place_like_concrete_scalar(inverse_norm, matrix_norm)
    return matrix_norm * inverse_norm, factorization_count, solve_count


def _dense_matrix_condition_estimate(
    matrix,
    *,
    lu_piv=None,
    transpose_operator=False,
):
    """Return a JAX-native Hager-Higham 1-norm condition estimate.

    The Hager-Higham iteration evaluates ``A⁻¹`` and ``A⁻ᵀ`` repeatedly,
    so the inner solves consume cached ``(lu, piv)`` factors via
    ``jsp_linalg.lu_solve``. When ``lu_piv`` is supplied it must be exactly
    the packed two-tuple returned by ``jsp_linalg.lu_factor``; callers holding
    public ``(P, L, U, lu, piv)`` factors should pass ``factors[3:5]``. With
    cached factors no factorization runs at all; otherwise the helper
    factorizes ``matrix`` once and shares those bytes across every inner solve.
    The naïve ``jnp.linalg.solve`` form re-factorized for every call, costing
    10 × O(n³) per estimate instead of the present O(n³) + 10 × O(n²).
    """
    condition_estimate, _factorization_count, _solve_count = (
        _dense_matrix_condition_estimate_with_telemetry(
            matrix,
            lu_piv=lu_piv,
            transpose_operator=transpose_operator,
        )
    )
    return condition_estimate


def _dense_matrix_solve_forward_error_success(
    matrix,
    solution,
    rhs,
    *,
    tol,
    condition_estimate=None,
):
    residual = rhs - matrix @ solution
    residual_rel = _relative_residual_1_norm(residual, rhs)
    if condition_estimate is None:
        condition_estimate = _dense_matrix_condition_estimate(matrix)
    return _forward_error_success(residual_rel, condition_estimate, tol=tol)


def _dense_matrix_solve_small_solution_success(solution, rhs, *, tol):
    solution = jnp.asarray(solution)
    rhs = jnp.asarray(rhs)
    solution_inf_norm = jnp.linalg.norm(solution, ord=np.inf)
    threshold = _device_scalar(
        _DENSE_LINEAR_SOLVE_SMALL_SOLUTION_FACTOR,
        dtype=rhs.dtype,
    ) * _effective_linear_solve_tolerance(rhs, tol)
    return jnp.all(jnp.isfinite(solution)) & (solution_inf_norm <= threshold)


def _dense_matrix_nonsingular_threshold(size, dtype):
    dtype = np.dtype(dtype)
    eps = float(np.finfo(dtype).eps)
    dimension_factor = np.sqrt(float(size)) if dtype == np.dtype(np.float32) else size
    threshold = 1.0 / (dimension_factor * eps)
    if dtype == np.dtype(np.float64):
        threshold = min(threshold, _FLOAT64_DENSE_MATRIX_MAX_CONDITION_ESTIMATE)
    return _device_scalar(threshold, dtype=dtype)


@partial(jax.jit, static_argnames=("size", "dtype"))
def _dense_matrix_condition_estimate_numerically_safe(
    condition_estimate,
    *,
    size,
    dtype,
):
    condition_estimate = jnp.asarray(condition_estimate, dtype=dtype)
    threshold = _dense_matrix_nonsingular_threshold(size, dtype)
    return jnp.isfinite(condition_estimate) & (condition_estimate < threshold)


def _dense_matrix_solve_numerically_safe(
    matrix,
    solution,
    rhs,
    *,
    tol,
    lu_piv=None,
    solve_dtype=None,
    condition_estimate=None,
):
    """Whether a dense adjoint solve is numerically trustworthy at ``solve_dtype``.

    A backward-error gate cannot distinguish a well-conditioned solve from a
    backward-stable-but-forward-garbage solve of a (near-)singular operator
    (``lu_factor`` of a rank-deficient matrix yields a tiny-but-finite pivot, so
    the solve returns a finite wrong answer with a small residual).  The
    condition screen ``isfinite(cond) & (cond < threshold)`` lets a degenerate
    operator fail closed: float64 production uses the smaller of the LAPACK
    rank-tolerance reciprocal ``1 / (n * eps)`` and an explicit 1e12 cap,
    cleanly separating the well-conditioned production ``J^T`` (cond ~ 1e3-1e6,
    with 1-norm estimates still far below the cap) from numerically singular
    systems where residual-only success hides a wrong forward solution.

    The forward-error *bound* ``cond * residual_rel`` is deliberately applied to
    float32 only.  At large ``n`` (the float64 production regime) the 1-norm
    condition estimate inflates ~``n``-fold over the 2-norm conditioning,
    tripping the bound's ``sqrt(eps)`` gate even on an accurate solve, so it
    would false-reject production.  Float32 smoke solves instead clear the
    broader ``1 / (sqrt(n) * eps)`` condition screen, which admits moderately
    conditioned operators that float32 precision cannot resolve to smoke
    tolerance; those must additionally satisfy the forward-error bound before the
    solve is accepted.  ``solve_dtype`` (the caller's rhs dtype) selects the lane
    so the gate keys on the intended working precision even when the operator is
    materialized at the runtime float64 policy dtype.

    ``condition_estimate`` reuses an existing certificate when the caller
    already computed one from the same matrix or factors.
    """
    if solve_dtype is None:
        solve_dtype = matrix.dtype
    solve_dtype = np.dtype(solve_dtype)
    size = int(matrix.shape[0])
    if condition_estimate is None:
        condition_estimate = _dense_matrix_condition_estimate(
            jnp.asarray(matrix),
            lu_piv=lu_piv,
        )
    nonsingular = _dense_matrix_condition_estimate_numerically_safe(
        condition_estimate,
        size=size,
        dtype=solve_dtype,
    )
    if solve_dtype != np.dtype(np.float32):
        return nonsingular
    matrix = jnp.asarray(matrix, dtype=solve_dtype)
    solution = jnp.asarray(solution, dtype=solve_dtype)
    rhs = jnp.asarray(rhs, dtype=solve_dtype)
    forward_error_safe = _dense_matrix_solve_forward_error_success(
        matrix,
        solution,
        rhs,
        tol=tol,
        condition_estimate=condition_estimate,
    )
    return nonsingular & forward_error_safe


def _solve_retained_jacobian_transpose_adjoint(
    payload: _ExactFinalLinearization,
    rhs,
    *,
    tol,
) -> _RetainedJacobianTransposeSolve:
    """Solve ``J^T lambda = rhs`` from one validated final-state payload."""

    validation = _validate_exact_final_linearization(payload)
    jacobian = jnp.asarray(payload.jacobian)
    rhs_dtype = jnp.asarray(rhs).dtype
    rhs = jnp.asarray(rhs, dtype=jacobian.dtype)
    nan_vector = jnp.full_like(rhs, jnp.nan)
    zero_vector = jnp.zeros_like(rhs)

    def invalid_payload(_operand):
        status = _linear_solve_status(
            nan_vector,
            nan_vector,
            rhs,
            tol=tol,
            iterations=_device_int32(0, like=rhs),
        )._replace(success=jnp.asarray(False))
        return _RetainedJacobianTransposeSolve(
            solution=nan_vector,
            correction=nan_vector,
            residual=nan_vector,
            condition_estimate=_device_scalar(jnp.inf, dtype=jacobian.dtype),
            payload_validation=validation,
            status=status,
        )

    def valid_payload(_operand):
        def zero_rhs(_zero_operand):
            status = _linear_solve_status(
                zero_vector,
                zero_vector,
                rhs,
                tol=tol,
                iterations=_device_int32(0, like=rhs),
            )._replace(success=jnp.asarray(True))
            return _RetainedJacobianTransposeSolve(
                solution=zero_vector,
                correction=zero_vector,
                residual=zero_vector,
                condition_estimate=_device_scalar(
                    jnp.nan,
                    dtype=jacobian.dtype,
                ),
                payload_validation=validation,
                status=status,
            )

        def nonzero_rhs(_nonzero_operand):
            lu_piv = (payload.lu, payload.pivots)
            with device_scope(PhaseId.ADJOINT_LU_SOLVE):
                solution = _lu_solve_dense_hessian(
                    lu_piv,
                    rhs,
                    transpose=True,
                )
            with device_scope(PhaseId.ADJOINT_REFINEMENT):
                correction = _lu_solve_dense_hessian(
                    lu_piv,
                    rhs - jacobian.T @ solution,
                    transpose=True,
                )
                solution = solution + correction
                residual = rhs - jacobian.T @ solution
            status = _linear_solve_status(
                solution,
                residual,
                rhs,
                tol=tol,
                iterations=_device_int32(0, like=rhs),
            )
            backward_error_success = _dense_matrix_backward_error_success(
                jacobian.T,
                solution,
                rhs,
                tol=tol,
            )
            condition_estimate, factorization_count, condition_solve_count = (
                _dense_matrix_condition_estimate_with_telemetry(
                    jacobian,
                    lu_piv=lu_piv,
                    transpose_operator=True,
                )
            )
            solve_safe = _dense_matrix_solve_numerically_safe(
                jacobian.T,
                solution,
                rhs,
                tol=tol,
                solve_dtype=rhs_dtype,
                condition_estimate=condition_estimate,
            )
            status = status._replace(
                success=(status.success | backward_error_success) & solve_safe,
                lu_factorization_count=factorization_count,
                lu_solve_count=(_device_int32(2, like=rhs) + condition_solve_count),
                refinement_correction_count=_device_int32(1, like=rhs),
            )
            solution = _linear_solve_solution_or_nan(solution, status)
            correction = jnp.where(status.success, correction, nan_vector)
            return _RetainedJacobianTransposeSolve(
                solution=solution,
                correction=correction,
                residual=residual,
                condition_estimate=condition_estimate,
                payload_validation=validation,
                status=status,
            )

        return lax.cond(
            jnp.all(rhs == zero_vector),
            zero_rhs,
            nonzero_rhs,
            operand=None,
        )

    return lax.cond(
        validation.success,
        valid_payload,
        invalid_payload,
        operand=None,
    )


def _solve_square_vector_system_operator_only(
    matvec,
    rhs,
    *,
    tol,
    max_refinement_steps=_SQUARE_OPERATOR_GMRES_REFINEMENT_STEPS,
):
    """Solve one square linear system with bounded operator-GMRES refinement."""
    rhs = jnp.asarray(rhs)

    def zero_rhs_solution(_unused):
        solution = jnp.zeros_like(rhs)
        residual = jnp.zeros_like(rhs)
        status = _linear_solve_status(
            solution,
            residual,
            rhs,
            tol=tol,
            iterations=_device_int32(0),
        )
        return solution, status

    def nonzero_rhs_solution(_unused):
        return _solve_square_vector_system_operator_only_nonzero_rhs(
            matvec,
            rhs,
            tol=tol,
            max_refinement_steps=max_refinement_steps,
        )

    return lax.cond(
        jnp.all(rhs == jnp.zeros((), dtype=rhs.dtype)),
        zero_rhs_solution,
        nonzero_rhs_solution,
        operand=None,
    )


def _solve_square_vector_system_operator_only_nonzero_rhs(
    matvec,
    rhs,
    *,
    tol,
    max_refinement_steps,
):
    effective_tol = _effective_linear_solve_tolerance(rhs, tol)
    solution, residual, info = _gmres_solve_array_system(
        matvec,
        rhs,
        tol=effective_tol,
    )
    status = _linear_solve_status(
        solution,
        residual,
        rhs,
        tol=tol,
        iterations=_linear_solve_iteration_count(info),
    )

    def refinement_step(carry, _unused):
        solution, residual, status, can_refine, accept_first_correction = carry

        def refine(_):
            correction, correction_residual, correction_info = (
                _gmres_solve_array_system(
                    matvec,
                    residual,
                    tol=effective_tol,
                )
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
            refined_status = _linear_solve_status(
                refined_solution,
                refined_residual,
                rhs,
                tol=tol,
                iterations=refined_iterations,
            )
            residual_improved = (
                refined_status.residual_relative <= status.residual_relative
            )
            accept_correction = correction_finite & (
                accept_first_correction | residual_improved | refined_status.success
            )
            accepted_can_refine = (
                accept_correction
                & _linear_solve_finite(refined_solution, refined_residual)
                & (~refined_status.success)
            )
            rejected_status = status._replace(iterations=refined_iterations)
            return lax.cond(
                accept_correction,
                lambda _: (
                    refined_solution,
                    refined_residual,
                    refined_status,
                    accepted_can_refine,
                    jnp.asarray(False),
                ),
                lambda _: (
                    solution,
                    residual,
                    rejected_status,
                    jnp.asarray(False),
                    jnp.asarray(False),
                ),
                operand=None,
            )

        return lax.cond(
            can_refine,
            refine,
            lambda _: (
                solution,
                residual,
                status,
                can_refine,
                accept_first_correction,
            ),
            operand=None,
        ), None

    initial_can_refine = _linear_solve_finite(solution, residual) & (~status.success)
    (solution, residual, status, _can_refine, _accept_first_correction), _ = lax.scan(
        refinement_step,
        (
            solution,
            residual,
            status,
            initial_can_refine,
            jnp.asarray(True),
        ),
        xs=None,
        length=int(max_refinement_steps),
    )
    return solution, status


def _apply_column_batched_operator(matvec, rhs):
    rhs = jnp.asarray(rhs)
    if rhs.ndim == 1:
        return matvec(rhs)
    return jax.vmap(matvec, in_axes=1, out_axes=1)(rhs)


def _dense_square_operator_matrix_dtype(rhs):
    rhs_dtype = np.dtype(jnp.asarray(rhs).dtype)
    if rhs_dtype.kind == "f":
        return np.dtype(get_backend_policy().runtime_dtype)
    return rhs_dtype


def _dense_square_operator_matrix_bytes_allowed(rhs):
    """Whether the ``n x n`` dense materialization (``n = rhs.shape[0]``) fits the
    dense-Jacobian byte cap.  The operator dimension is ``rhs.shape[0]`` whether
    ``rhs`` is a single vector or a column-batched ``(n, k)`` right-hand side."""
    rhs = jnp.asarray(rhs)
    dimension = int(rhs.shape[0])
    matrix_bytes = (
        dimension * dimension * _dense_square_operator_matrix_dtype(rhs).itemsize
    )
    return matrix_bytes <= int(get_backend_policy().max_dense_jacobian_bytes)


def _dense_square_operator_materialization_allowed(rhs):
    rhs = jnp.asarray(rhs)
    if rhs.ndim != 1:
        return False
    return _dense_square_operator_matrix_bytes_allowed(rhs)


def _dense_square_operator_lu_materialization_allowed(rhs):
    """Dense-LU exact-adjoint gate: accept a single vector OR a column-batched
    ``(n, k)`` right-hand side.  The dense ``J^T`` is ``n x n`` regardless of the
    number of columns, so one factorization serves all ``k`` columns and
    ``lu_solve`` solves the batched RHS in a single call.  This is what lets the
    dense-LU path reach the production single-stage adjoint, whose fused
    residual+iota+non-QS gradient solves a 2-D batched RHS (the per-objective
    ``dJ()`` solves a single vector)."""
    rhs = jnp.asarray(rhs)
    if rhs.ndim not in (1, 2):
        return False
    return _dense_square_operator_matrix_bytes_allowed(rhs)


def _resolve_dense_operator_batch_width(
    batch_width: int | None,
    *,
    dimension: int,
) -> int:
    configured_width = (
        _DENSE_OPERATOR_CHUNK_BATCH_SIZE if batch_width is None else batch_width
    )
    if isinstance(configured_width, bool) or not isinstance(configured_width, int):
        raise TypeError("batch_width must be an integer")
    if configured_width < 1:
        raise ValueError("batch_width must be positive")
    return configured_width


def _dense_square_operator_matrix(
    matvec,
    rhs,
    *,
    matrix_dtype=None,
    sweep_dtype=None,
    batch_width: int | None = None,
):
    rhs = jnp.asarray(rhs)
    dimension = int(rhs.shape[0])
    if sweep_dtype is None:
        sweep_dtype = _dense_square_operator_matrix_dtype(rhs)
    else:
        sweep_dtype = np.dtype(sweep_dtype)
    if matrix_dtype is None:
        matrix_dtype = _dense_square_operator_matrix_dtype(rhs)
    else:
        matrix_dtype = np.dtype(matrix_dtype)

    column_indices = _staged_like(
        rhs,
        np.arange(dimension, dtype=np.int32),
        dtype=np.int32,
    )
    chunk_size = min(
        _resolve_dense_operator_batch_width(
            batch_width,
            dimension=dimension,
        ),
        dimension,
    )
    chunk_count = dimension // chunk_size
    chunked_dimension = chunk_count * chunk_size
    column_index_chunks = _staged_like(
        rhs,
        np.arange(chunked_dimension, dtype=np.int32).reshape(
            chunk_count,
            chunk_size,
        ),
        dtype=np.int32,
    )
    example_basis = _staged_like(
        rhs,
        np.zeros(dimension, dtype=sweep_dtype),
        dtype=sweep_dtype,
    )
    matvec_closed_jaxpr = jax.make_jaxpr(matvec)(example_basis)
    matvec_jaxpr = matvec_closed_jaxpr.jaxpr
    matvec_closure_args = tuple(
        _place_like_concrete_array(constant, rhs)
        for constant in matvec_closed_jaxpr.consts
    )
    scan_carry = (column_indices, matvec_closure_args)

    def converted_matvec(basis_vector, *closure_args):
        closed_jaxpr = jax_core.ClosedJaxpr(matvec_jaxpr, closure_args)
        return jax_core.jaxpr_as_fun(closed_jaxpr)(basis_vector)[0]

    def materialize_column(column_index, basis_indices, closure_args):
        basis_vector = jnp.asarray(
            basis_indices == column_index,
            dtype=sweep_dtype,
        )
        return converted_matvec(basis_vector, *closure_args)

    def materialize_chunk(carry, chunk_column_indices):
        basis_indices, closure_args = carry
        columns = jax.vmap(materialize_column, in_axes=(0, None, None))(
            chunk_column_indices,
            basis_indices,
            closure_args,
        )
        return carry, columns

    _, chunked_columns = lax.scan(
        materialize_chunk,
        scan_carry,
        column_index_chunks,
    )
    columns = jnp.reshape(
        chunked_columns,
        (chunked_dimension, dimension),
    )
    if chunked_dimension < dimension:
        remainder_indices = _staged_like(
            rhs,
            np.arange(chunked_dimension, dimension, dtype=np.int32),
            dtype=np.int32,
        )
        _, remainder_batches = lax.scan(
            materialize_chunk,
            scan_carry,
            remainder_indices[None, :],
        )
        remainder_columns = remainder_batches[0]
        columns = jnp.concatenate((columns, remainder_columns), axis=0)
    return jnp.asarray(jnp.swapaxes(columns, 0, 1), dtype=matrix_dtype)


def _solve_square_array_system_operator_only(
    matvec,
    rhs,
    *,
    tol,
    max_refinement_steps=_SQUARE_OPERATOR_GMRES_REFINEMENT_STEPS,
):
    """Solve vector or column-batched square systems with operator-only GMRES."""
    rhs = jnp.asarray(rhs)
    if rhs.ndim == 1:
        return _solve_square_vector_system_operator_only(
            matvec,
            rhs,
            tol=tol,
            max_refinement_steps=max_refinement_steps,
        )

    def solve_column(column):
        return _solve_square_vector_system_operator_only(
            matvec,
            column,
            tol=tol,
            max_refinement_steps=max_refinement_steps,
        )

    solutions, column_statuses = jax.vmap(
        solve_column,
        in_axes=1,
        out_axes=(1, 0),
    )(rhs)
    return solutions, _LinearSolveStatus(
        success=jnp.all(column_statuses.success),
        residual=jnp.max(column_statuses.residual),
        residual_relative=jnp.max(column_statuses.residual_relative),
        iterations=jnp.max(column_statuses.iterations),
        dense_materialization_count=jnp.sum(
            column_statuses.dense_materialization_count
        ),
        lu_factorization_count=jnp.sum(column_statuses.lu_factorization_count),
        lu_solve_count=jnp.sum(column_statuses.lu_solve_count),
        refinement_correction_count=jnp.sum(
            column_statuses.refinement_correction_count
        ),
    )


def _jacobian_linear_operator(residual_fn, x):
    jvp_fn = _jacobian_vector_product_fn(residual_fn)
    residual_x, pullback = jax.vjp(residual_fn, x)
    residual_size = int(np.asarray(jnp.asarray(residual_x).size))
    decision_size = int(np.asarray(jnp.asarray(x).size))
    dtype = jnp.asarray(x).dtype

    def matvec_column(v):
        return jvp_fn(x, v)

    def transpose_matvec_column(v):
        return pullback(v)[0]

    def matvec(v):
        return _apply_column_batched_operator(matvec_column, v)

    def transpose_matvec(v):
        return _apply_column_batched_operator(transpose_matvec_column, v)

    return {
        "kind": "jacobian",
        "shape": (residual_size, decision_size),
        "dtype": dtype,
        "matvec": matvec,
        "transpose_matvec": transpose_matvec,
    }


def _solve_jacobian_operator(
    operator,
    rhs,
    *,
    transpose,
    tol,
    max_refinement_steps=_SQUARE_OPERATOR_GMRES_REFINEMENT_STEPS,
):
    solution, status = _solve_jacobian_operator_with_status(
        operator,
        rhs,
        transpose=transpose,
        tol=tol,
        max_refinement_steps=max_refinement_steps,
    )
    return _linear_solve_solution_or_nan(solution, status)


def _solve_jacobian_system_with_status(
    residual_fn,
    x,
    rhs,
    *,
    transpose,
    tol,
    max_refinement_steps=_SQUARE_OPERATOR_GMRES_REFINEMENT_STEPS,
):
    operator = _jacobian_linear_operator(residual_fn, x)
    return _solve_jacobian_operator_with_status(
        operator,
        rhs,
        transpose=transpose,
        tol=tol,
        max_refinement_steps=max_refinement_steps,
    )


def _solve_jacobian_operator_with_status(
    operator,
    rhs,
    *,
    transpose,
    tol,
    max_refinement_steps=_SQUARE_OPERATOR_GMRES_REFINEMENT_STEPS,
):
    matvec = operator["transpose_matvec"] if transpose else operator["matvec"]
    # Exact-Jacobian adjoint: parity profiles require the deterministic direct
    # solve, while fast profiles retain operator GMRES unless the explicit
    # diagnostic opt-in is set. The dense route remains scoped to transpose
    # solves and to matrices within the backend's byte budget.
    exact_adjoint_dense_lu = _EXACT_ADJOINT_DENSE_LU or get_backend_policy().parity_mode
    if (
        exact_adjoint_dense_lu
        and transpose
        and _dense_square_operator_lu_materialization_allowed(rhs)
    ):
        return _solve_dense_square_operator_lu_system_with_status(
            matvec,
            rhs,
            tol=tol,
        )
    return _solve_square_array_system_operator_only(
        matvec,
        rhs,
        tol=tol,
        max_refinement_steps=max_refinement_steps,
    )


def _solve_dense_square_operator_least_squares_system_with_status(
    matvec,
    rhs,
    *,
    tol,
    sweep_dtype=None,
):
    rhs_dtype = jnp.asarray(rhs).dtype
    matrix = _dense_square_operator_matrix(matvec, rhs, sweep_dtype=sweep_dtype)
    rhs = jnp.asarray(rhs, dtype=matrix.dtype)
    solution = jnp.linalg.lstsq(matrix, rhs, rcond=None)[0]
    residual = rhs - matrix @ solution
    status = _linear_solve_status(
        solution,
        residual,
        rhs,
        tol=tol,
        iterations=_device_int32(0, like=rhs),
    )
    backward_error_success = _dense_matrix_backward_error_success(
        matrix,
        solution,
        rhs,
        tol=tol,
    )
    # Numerical-safety guard -- see the LU sibling: a backward-stable solve
    # of a (near-)singular operator is still forward-garbage, which the
    # backward-error gate cannot detect.
    condition_estimate, condition_factorizations, condition_lu_solves = (
        _dense_matrix_condition_estimate_with_telemetry(matrix)
    )
    solve_safe = _dense_matrix_solve_numerically_safe(
        matrix,
        solution,
        rhs,
        tol=tol,
        solve_dtype=rhs_dtype,
        condition_estimate=condition_estimate,
    )
    return solution, status._replace(
        success=(status.success | backward_error_success) & solve_safe,
        dense_materialization_count=_device_int32(1, like=rhs),
        lu_factorization_count=condition_factorizations,
        lu_solve_count=condition_lu_solves,
    )


def _solve_dense_square_operator_lu_system_with_status(
    matvec,
    rhs,
    *,
    tol,
    sweep_dtype=None,
):
    """Direct LU solve of one square system from an operator matvec.

    Materializes the dense square operator implied by ``matvec`` (reusing the
    chunked, transfer-guard-clean ``_dense_square_operator_matrix`` assembler),
    factorizes it once with ``lu_factor``, solves ``M x = rhs`` with ``lu_solve``
    and then applies a single step of iterative refinement (resolve the residual
    against the same factors, add the correction).  Returns the established
    ``_LinearSolveStatus`` via ``_linear_solve_status`` so the gate logic is
    unchanged.  Used for the exact-Jacobian adjoint transpose solve, where the
    matrix is the small, well-conditioned ``J^T`` that operator-GMRES cannot
    resolve.
    """
    rhs_dtype = jnp.asarray(rhs).dtype
    with device_scope(PhaseId.ADJOINT_DENSE_MATRIX):
        matrix = _dense_square_operator_matrix(
            matvec,
            rhs,
            sweep_dtype=sweep_dtype,
        )
    rhs = jnp.asarray(rhs, dtype=matrix.dtype)
    with device_scope(PhaseId.ADJOINT_LU_FACTOR):
        lu_piv = jsp_linalg.lu_factor(matrix)
    with device_scope(PhaseId.ADJOINT_LU_SOLVE):
        solution = jsp_linalg.lu_solve(lu_piv, rhs)
    # One step of iterative refinement against the cached factors: resolves the
    # rounding error of the direct solve back to the matrix's backward-error
    # floor without a second factorization (O(n^2) per step).
    with device_scope(PhaseId.ADJOINT_REFINEMENT):
        correction = jsp_linalg.lu_solve(lu_piv, rhs - matrix @ solution)
        solution = solution + correction
        residual = rhs - matrix @ solution
        status = _linear_solve_status(
            solution,
            residual,
            rhs,
            tol=tol,
            iterations=_device_int32(0, like=rhs),
        )
        backward_error_success = _dense_matrix_backward_error_success(
            matrix,
            solution,
            rhs,
            tol=tol,
        )
        # Numerical-safety guard: a backward-stable solve of a singular or
        # near-singular operator still yields a forward-garbage solution that the
        # backward-error gate above cannot detect.  Fail closed when the Hager-Higham
        # condition estimate exceeds the dtype-specific degeneracy threshold; float32
        # smoke solves that pass the broader threshold must also satisfy the forward
        # error bound.  A degenerate J^T then fails closed instead of silently
        # returning a wrong adjoint, while the well-conditioned production J^T
        # (cond ~ 1e3-1e6) passes with many orders of margin.  The cached ``lu_piv``
        # is reused by the Hager-Higham inner solves (O(n^2)), avoiding a second
        # factorization and keeping the strict transfer-guard path on device.
        condition_estimate, condition_factorizations, condition_lu_solves = (
            _dense_matrix_condition_estimate_with_telemetry(
                matrix,
                lu_piv=lu_piv,
            )
        )
        solve_safe = _dense_matrix_solve_numerically_safe(
            matrix,
            solution,
            rhs,
            tol=tol,
            lu_piv=lu_piv,
            solve_dtype=rhs_dtype,
            condition_estimate=condition_estimate,
        )
        return solution, status._replace(
            success=(status.success | backward_error_success) & solve_safe,
            dense_materialization_count=_device_int32(1, like=rhs),
            lu_factorization_count=(
                _device_int32(1, like=rhs) + condition_factorizations
            ),
            lu_solve_count=(_device_int32(2, like=rhs) + condition_lu_solves),
            refinement_correction_count=_device_int32(1, like=rhs),
        )
