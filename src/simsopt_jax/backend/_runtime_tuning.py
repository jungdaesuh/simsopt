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
    _HESSIAN_VJP_POINT_CHUNK_SIZE_ENV,
    _POINT_CHUNK_SIZE_ENV,
    _QUADRATURE_BLOCK_SIZE_ENV,
    _optional_nonneg_int_env,
)

_FIELD_KERNEL_DEFAULTS = {
    "native_cpu": {"coil_chunk_size": 0, "quadrature_block_size": 0, "hessian_vjp_point_chunk_size": 0},
    "jax_cpu_fast": {"coil_chunk_size": 64, "quadrature_block_size": 64, "hessian_vjp_point_chunk_size": 128},
    "jax_cpu_parity": {"coil_chunk_size": 16, "quadrature_block_size": 0, "hessian_vjp_point_chunk_size": 128},
    "jax_gpu_parity": {"coil_chunk_size": 16, "quadrature_block_size": 0, "hessian_vjp_point_chunk_size": 2048},
    "jax_gpu_fast": {"coil_chunk_size": 64, "quadrature_block_size": 64, "hessian_vjp_point_chunk_size": 2048},
}
# gradgradB VJP point tile, measured with 16 coils x 128 nodes. GPU 2048 (RTX
# 5090, 16384 points): d2B pullback 74 ms at 0.91 GB peak, versus 84 ms at
# 0.54 GB for 512 and 70 ms at 1.82 GB for 4096. CPU 128 (4096 points, 4
# threads): 115 ms, versus 360 ms for 8, 253 ms for 32 and 240 ms for 512.
# Live memory is about 220 bytes per (point, coil node) pair in a tile.
# Forward point tiling is independent.
_POINT_CHUNK_SIZE_BY_POLICY = {
    "host_reference": 0,
    "stable_default": 256,
    "performance_tuned": 1024,
}
_FIELD_KERNEL_ENV_BY_KEY = {
    "coil_chunk_size": _COIL_CHUNK_SIZE_ENV,
    "quadrature_block_size": _QUADRATURE_BLOCK_SIZE_ENV,
    "point_chunk_size": _POINT_CHUNK_SIZE_ENV,
    "hessian_vjp_point_chunk_size": _HESSIAN_VJP_POINT_CHUNK_SIZE_ENV,
}


@dataclass(frozen=True)
class FieldKernelTuning:
    """Immutable field-kernel tiles; zero disables tiling on the selected axis.

    Args:
        mode (str): Resolved backend mode, such as ``jax_cpu_parity``.
        chunk_policy (str): Point-tiling policy, with ``_dense_audit`` appended
            when the transfer guard disables all tiling.
        coil_chunk_size (int): Number of coils per reduction tile.
        quadrature_block_size (int): Number of quadrature nodes per tile.
        point_chunk_size (int): Number of evaluation points per forward tile.
        hessian_vjp_point_chunk_size (int): Number of evaluation points per
            gradgradB VJP tile, independent of ``point_chunk_size``.
    """
    mode: str
    chunk_policy: str
    coil_chunk_size: int
    quadrature_block_size: int
    point_chunk_size: int
    hessian_vjp_point_chunk_size: int


def _point_chunk_size_default(chunk_policy: str) -> int:
    return _POINT_CHUNK_SIZE_BY_POLICY[chunk_policy]


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


def _build_field_kernel_tuning(policy: BackendPolicy) -> FieldKernelTuning:
    chunk_sizes = _apply_chunk_env_overrides(
        _static_chunk_sizes(policy.mode, policy.chunk_policy)
    )
    effective_chunk_policy = policy.chunk_policy
    if policy.transfer_guard == "disallow":
        effective_chunk_policy = f"{policy.chunk_policy}_dense_audit"
        chunk_sizes = dict.fromkeys(chunk_sizes, 0)
    return FieldKernelTuning(
        mode=policy.mode,
        chunk_policy=effective_chunk_policy,
        coil_chunk_size=chunk_sizes["coil_chunk_size"],
        quadrature_block_size=chunk_sizes["quadrature_block_size"],
        point_chunk_size=chunk_sizes["point_chunk_size"],
        hessian_vjp_point_chunk_size=chunk_sizes["hessian_vjp_point_chunk_size"],
    )
