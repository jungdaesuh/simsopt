"""Field-kernel chunk tuning for the JAX backend.

Owns the chunk contract plus the pure builder that resolves it from policy and
environment. Process-global caches and locks live in
:mod:`simsopt_jax.backend.runtime`.
"""

from __future__ import annotations

from dataclasses import dataclass

from simsopt_jax.backend._runtime_policy import (
    BackendPolicy,
    _COIL_CHUNK_SIZE_ENV,
    _HESSIAN_VJP_PAIR_BUDGET_ENV,
    _POINT_CHUNK_SIZE_ENV,
    _QUADRATURE_BLOCK_SIZE_ENV,
    _optional_nonneg_int_env,
)

_FIELD_KERNEL_DEFAULTS = {
    "native_cpu": {"coil_chunk_size": 0, "quadrature_block_size": 0, "hessian_vjp_pair_budget": 0},
    "jax_cpu_fast": {"coil_chunk_size": 64, "quadrature_block_size": 64, "hessian_vjp_pair_budget": 262_144},
    "jax_cpu_parity": {"coil_chunk_size": 16, "quadrature_block_size": 0, "hessian_vjp_pair_budget": 262_144},
    "jax_gpu_parity": {"coil_chunk_size": 16, "quadrature_block_size": 0, "hessian_vjp_pair_budget": 1_048_576},
    "jax_gpu_fast": {"coil_chunk_size": 64, "quadrature_block_size": 64, "hessian_vjp_pair_budget": 1_048_576},
}
# Closed-form gradgradB VJP budgets, in point-node pairs per point tile (XLA
# temp grows about 64-100 bytes per pair). Measured at 16 coils x 128 nodes,
# where the budgets give 512-point GPU and 128-point CPU tiles. RTX 5090,
# 16384 points: tile 512 takes 18 ms with 68 MB temp, versus 19 ms/34 MB for
# 256 and 14 ms/807 MB for 4096, so 512 keeps the tile well below the B/dB
# VJP's peak. CPU, 4096 points: tile 128 takes 27 ms with 15 MB temp, versus
# 65 ms for 8 and 48 ms for 32. Forward point tiling is independent.
_POINT_CHUNK_SIZE_BY_POLICY = {
    "host_reference": 0,
    "stable_default": 256,
    "performance_tuned": 1024,
}
_FIELD_KERNEL_ENV_BY_KEY = {
    "coil_chunk_size": _COIL_CHUNK_SIZE_ENV,
    "quadrature_block_size": _QUADRATURE_BLOCK_SIZE_ENV,
    "point_chunk_size": _POINT_CHUNK_SIZE_ENV,
    "hessian_vjp_pair_budget": _HESSIAN_VJP_PAIR_BUDGET_ENV,
}


@dataclass(frozen=True)
class FieldKernelTuning:
    """Immutable field-kernel tiles; zero disables tiling on the selected axis.

    The gradgradB VJP reads only ``hessian_vjp_pair_budget``; the coil,
    quadrature and point tiles apply to the other field kernels.

    Args:
        mode (str): Resolved backend mode, such as ``jax_cpu_parity``.
        chunk_policy (str): Point-tiling policy, with ``_dense_audit`` appended
            when the transfer guard disables all tiling.
        coil_chunk_size (int): Number of coils per reduction tile.
        quadrature_block_size (int): Number of quadrature nodes per tile.
        point_chunk_size (int): Number of evaluation points per forward tile.
        hessian_vjp_pair_budget (int): Point-node pairs per gradgradB VJP
            tile. A coil set of ``ncoils`` coils with ``nquad`` nodes each gets
            ``max(1, hessian_vjp_pair_budget // (ncoils * nquad))`` points per
            tile, so its working memory stays bounded as the coil set grows.
    """
    mode: str
    chunk_policy: str
    coil_chunk_size: int
    quadrature_block_size: int
    point_chunk_size: int
    hessian_vjp_pair_budget: int


def _point_chunk_size_default(chunk_policy: str) -> int:
    return _POINT_CHUNK_SIZE_BY_POLICY.get(chunk_policy, 0)


def _static_chunk_sizes(mode: str, chunk_policy: str) -> dict[str, int]:
    return {
        **_FIELD_KERNEL_DEFAULTS[mode],
        "point_chunk_size": _point_chunk_size_default(chunk_policy),
    }


def _apply_chunk_env_overrides(chunk_sizes: dict[str, int]) -> dict[str, int]:
    resolved = dict(chunk_sizes)
    for key, env_name in _FIELD_KERNEL_ENV_BY_KEY.items():
        value = _optional_nonneg_int_env(env_name)
        if value is not None:
            resolved[key] = value
    return resolved


def _build_field_kernel_tuning(
    mode: str,
    policy: BackendPolicy,
) -> FieldKernelTuning:
    chunk_sizes = _apply_chunk_env_overrides(
        _static_chunk_sizes(mode, policy.chunk_policy)
    )
    effective_chunk_policy = policy.chunk_policy
    if policy.transfer_guard == "disallow":
        effective_chunk_policy = f"{policy.chunk_policy}_dense_audit"
        chunk_sizes = dict.fromkeys(chunk_sizes, 0)
    return FieldKernelTuning(
        mode=mode,
        chunk_policy=effective_chunk_policy,
        coil_chunk_size=chunk_sizes["coil_chunk_size"],
        quadrature_block_size=chunk_sizes["quadrature_block_size"],
        point_chunk_size=chunk_sizes["point_chunk_size"],
        hessian_vjp_pair_budget=chunk_sizes["hessian_vjp_pair_budget"],
    )
