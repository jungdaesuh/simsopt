from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def _restore_backend_config(config) -> None:
    from simsopt.backend import set_backend

    set_backend(
        config.mode,
        strict=config.strict,
        debug_nans=config.debug_nans,
        transfer_guard=config.transfer_guard,
        compilation_cache_dir=config.compilation_cache_dir,
        xla_gpu_preallocate=config.xla_gpu_preallocate,
        xla_gpu_mem_fraction=config.xla_gpu_mem_fraction,
        xla_gpu_allocator=config.xla_gpu_allocator,
        tf_gpu_allocator=config.tf_gpu_allocator,
        configure_runtime=False,
    )


def test_jax_metal_smoke_policy_runtime_dtype():
    from simsopt.backend import get_backend_config, get_backend_policy, set_backend

    previous = get_backend_config()
    try:
        set_backend("jax_metal_smoke", configure_runtime=False)

        policy = get_backend_policy()

        assert policy.runtime_dtype == "float32"
        assert policy.host_dtype == "float32"
    finally:
        _restore_backend_config(previous)


def test_non_metal_modes_keep_float64_policy_dtype():
    from simsopt.backend import get_backend_policy

    for mode in (
        "native_cpu",
        "jax_cpu_fast",
        "jax_cpu_parity",
        "jax_gpu_fast",
        "jax_gpu_parity",
    ):
        policy = get_backend_policy(mode)

        assert policy.runtime_dtype == "float64"
        assert policy.host_dtype == "float64"


def test_require_runtime_dtype_follows_backend_policy():
    from simsopt.backend import get_backend_config, set_backend
    from simsopt.jax_core._math_utils import require_runtime_dtype

    previous = get_backend_config()
    try:
        set_backend("jax_metal_smoke", configure_runtime=False)
        require_runtime_dtype("x", jnp.asarray([1.0], dtype=jnp.float32))
        with pytest.raises(TypeError, match="x must have runtime dtype float32"):
            require_runtime_dtype("x", jnp.asarray([1.0], dtype=jnp.float64))

        set_backend("jax_cpu_parity", configure_runtime=False)
        require_runtime_dtype("x", jnp.asarray([1.0], dtype=jnp.float64))
        with pytest.raises(TypeError, match="x must have runtime dtype float64"):
            require_runtime_dtype("x", jnp.asarray([1.0], dtype=jnp.float32))
    finally:
        _restore_backend_config(previous)


def test_as_runtime_value_uses_policy_dtype_for_host_values():
    from simsopt.backend import get_backend_config, set_backend
    from simsopt.jax_core._math_utils import as_runtime_value

    previous = get_backend_config()
    try:
        set_backend("jax_metal_smoke", configure_runtime=False)
        reference32 = jnp.asarray([0.0], dtype=jnp.float32)
        value32 = as_runtime_value(
            np.asarray([1.0, 2.0], dtype=np.float64),
            reference=reference32,
        )
        assert value32.dtype == jnp.float32

        set_backend("jax_cpu_parity", configure_runtime=False)
        reference64 = jnp.asarray([0.0], dtype=jnp.float64)
        value64 = as_runtime_value(
            np.asarray([1.0, 2.0], dtype=np.float32),
            reference=reference64,
        )
        assert value64.dtype == jnp.float64
    finally:
        _restore_backend_config(previous)


def test_boozer_residual_accepts_float32_under_metal_policy():
    from simsopt.backend import get_backend_config, set_backend
    from simsopt.geo.boozer_residual_jax import (
        boozer_residual_scalar,
        boozer_residual_vector,
    )

    previous = get_backend_config()
    try:
        set_backend("jax_metal_smoke", configure_runtime=False)
        B = jnp.ones((2, 3, 3), dtype=jnp.float32)
        xphi = jnp.full((2, 3, 3), 2.0, dtype=jnp.float32)
        xtheta = jnp.full((2, 3, 3), -0.5, dtype=jnp.float32)

        scalar_value = boozer_residual_scalar(
            np.float32(1.25),
            np.float32(-0.2),
            B,
            xphi,
            xtheta,
            weight_inv_modB=True,
        )
        vector_value = boozer_residual_vector(
            np.float32(1.25),
            np.float32(-0.2),
            B,
            xphi,
            xtheta,
            weight_inv_modB=True,
        )

        assert scalar_value.dtype == jnp.float32
        assert vector_value.dtype == jnp.float32
        assert bool(jnp.isfinite(scalar_value))
        assert bool(jnp.all(jnp.isfinite(vector_value)))
    finally:
        _restore_backend_config(previous)
