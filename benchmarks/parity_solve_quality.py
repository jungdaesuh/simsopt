"""Deterministic operator-action probe spec for the scientific-equivalence ladder.

Implements §4 of ``docs/parity_scientific_equivalence_contract_2026-05-09.md``:
sha256-seeded Gaussian probes plus a standard-basis pin, orthonormalized via QR.
The output is a fixed probe set used by the parity arbiter to populate the
``ls_hessian_action_max_rel`` and ``exact_jacobian_action_max_rel`` reporting
fields without depending on Python's randomized ``hash()``.

Standard-basis-only probe sets are explicitly forbidden because they collapse
the operator-action gate to dense-bytes parity.

Phase 1.5 adds ``compute_dense_operator_action_max_rel_error`` — the
arbiter-side composition that materializes both probe construction and
maximum-relative-error reduction in one call so the parity benchmark only
imports a single SSOT entrypoint per gate. Both the LS gate L4
(``ls_hessian_action_max_rel``) and the Exact gate E3
(``exact_jacobian_action_max_rel``) consume this composition.
"""

from __future__ import annotations

import hashlib
from typing import Final

import numpy as np


_PROBE_COUNT_LIMIT: Final[int] = 8


def _process_stable_seed(artifact_name: str) -> int:
    digest = hashlib.sha256(artifact_name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little")


def _gaussian_probes(
    *,
    decision_size: int,
    seed: int,
    probe_count: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(size=(decision_size, probe_count))
    q, _ = np.linalg.qr(raw)
    return q


def _standard_basis_probe(
    *,
    decision_size: int,
    basis_index: int,
) -> np.ndarray:
    if not 0 <= basis_index < decision_size:
        raise ValueError(
            "basis_index must lie in [0, decision_size); "
            f"got basis_index={basis_index} and decision_size={decision_size}."
        )
    probe = np.zeros((decision_size, 1), dtype=np.float64)
    probe[basis_index, 0] = 1.0
    return probe


def construct_operator_action_probes(
    *,
    decision_size: int,
    artifact_name: str,
    extra_basis_index: int = 0,
) -> np.ndarray:
    """Return the fixed probe set for the LS L4 / Exact E3 gates.

    Args:
        decision_size: Hessian / Jacobian dimension ``n``.
        artifact_name: Stable identifier of the parity fixture; used to seed
            the probe RNG via sha256 so probes match across runs and machines.
        extra_basis_index: Index of the standard-basis vector pinned as the
            ``k+1``-th probe per §4.

    Returns:
        ``(decision_size, k+1)`` probe matrix with ``k = min(8, n)``
        QR-orthonormalized Gaussian columns followed by ``e_extra_basis_index``.
    """
    if decision_size <= 0:
        raise ValueError(f"decision_size must be positive; got {decision_size}.")
    seed = _process_stable_seed(artifact_name)
    probe_count = min(_PROBE_COUNT_LIMIT, decision_size)
    gaussian = _gaussian_probes(
        decision_size=decision_size,
        seed=seed,
        probe_count=probe_count,
    )
    basis = _standard_basis_probe(
        decision_size=decision_size,
        basis_index=extra_basis_index,
    )
    return np.concatenate([gaussian, basis], axis=1)


def operator_action_max_relative_error(
    op_jax_action: np.ndarray,
    op_cpp_action: np.ndarray,
    *,
    eps: float = 1e-30,
) -> float:
    """Return ``max_i ‖op_jax v_i − op_cpp v_i‖ / max(‖op_cpp v_i‖, eps)``.

    ``op_jax_action`` and ``op_cpp_action`` carry the operator action on the
    probe set, with shape ``(decision_size, num_probes)``. ``eps`` guards
    against zero-norm divisor when the C++ oracle returns the trivial action.
    """
    op_jax = np.asarray(op_jax_action, dtype=np.float64)
    op_cpp = np.asarray(op_cpp_action, dtype=np.float64)
    if op_jax.shape != op_cpp.shape:
        raise ValueError(
            "JAX and C++ probe-action arrays must share shape; "
            f"got JAX={op_jax.shape}, C++={op_cpp.shape}."
        )
    if op_jax.ndim != 2:
        raise ValueError(f"Probe-action arrays must be 2D; got ndim={op_jax.ndim}.")
    diff = np.linalg.norm(op_jax - op_cpp, axis=0)
    ref = np.linalg.norm(op_cpp, axis=0)
    rel = diff / np.maximum(ref, float(eps))
    return float(np.max(rel))


def compute_dense_operator_action_max_rel_error(
    op_jax_dense: np.ndarray,
    op_cpp_dense: np.ndarray,
    *,
    artifact_name: str,
    extra_basis_index: int = 0,
    eps: float = 1e-30,
) -> float:
    """Return the LS L4 / Exact E3 operator-action max relative error.

    SSOT composition for the scientific-equivalence ladder gates L4
    (``ls_hessian_action_max_rel``) and E3
    (``exact_jacobian_action_max_rel``) per
    ``docs/parity_scientific_equivalence_contract_2026-05-09.md`` §4.

    The function performs two steps the arbiter would otherwise duplicate:

    1. Build the deterministic probe set using
       ``construct_operator_action_probes(decision_size, artifact_name)``.
    2. Apply each dense operator to the probe set and reduce via
       ``operator_action_max_relative_error``.

    Operator dimensionality requirements:

    - For LS (gate L4) the operators are ``(n, n)`` symmetric Hessians whose
      action ``H @ v`` lives in ``R^n``. ``decision_size = n``.
    - For Exact (gate E3) the Jacobian is ``(m, n)`` where ``m >= n`` because
      the augmented residual stacks the constraint rows. The probe set must
      be applied as ``J @ v`` with ``v in R^n``; ``decision_size = n``.

    Both ``op_jax_dense`` and ``op_cpp_dense`` must share shape and have at
    least 2 dimensions. ``decision_size`` is inferred from the second axis.

    Standard-basis-only probe sets remain forbidden — gate L4 / E3 are smoke
    diagnostics, not proofs of operator equality. The rigorous proof method
    is the existing ``direct-hessian-oracle`` lane in
    ``benchmarks/validation_ladder_contract.py``.
    """
    op_jax = np.asarray(op_jax_dense, dtype=np.float64)
    op_cpp = np.asarray(op_cpp_dense, dtype=np.float64)
    if op_jax.shape != op_cpp.shape:
        raise ValueError(
            "JAX and C++ operators must share shape; "
            f"got JAX={op_jax.shape}, C++={op_cpp.shape}."
        )
    if op_jax.ndim != 2:
        raise ValueError(f"Operators must be 2D; got ndim={op_jax.ndim}.")
    decision_size = int(op_cpp.shape[1])
    probes = construct_operator_action_probes(
        decision_size=decision_size,
        artifact_name=artifact_name,
        extra_basis_index=extra_basis_index,
    )
    op_jax_action = op_jax @ probes
    op_cpp_action = op_cpp @ probes
    return operator_action_max_relative_error(
        op_jax_action,
        op_cpp_action,
        eps=eps,
    )
