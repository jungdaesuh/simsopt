from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


@pytest.mark.mps
def test_jax_mps_smoke_runtime_dtype_on_real_mps_device():
    mps_devices = tuple(
        device for device in jax.devices() if device.platform.lower() == "mps"
    )
    if not mps_devices:
        pytest.skip("requires a JAX MPS device")

    from simsopt.backend import get_backend_config, get_backend_policy, set_backend
    from simsopt.backend.dtypes import runtime_device_put

    previous = get_backend_config()
    try:
        set_backend("jax_mps_smoke", configure_runtime=False)
        policy = get_backend_policy()

        value = runtime_device_put([1.0, 2.0], target=mps_devices[0])

        assert policy.runtime_dtype == "float32"
        assert value.dtype == jnp.float32
        assert value.device.platform.lower() == "mps"
    finally:
        set_backend(
            previous.mode,
            strict=previous.strict,
            debug_nans=previous.debug_nans,
            disable_jit=previous.disable_jit,
            transfer_guard=previous.transfer_guard,
            compilation_cache_dir=previous.compilation_cache_dir,
            xla_gpu_preallocate=previous.xla_gpu_preallocate,
            xla_gpu_mem_fraction=previous.xla_gpu_mem_fraction,
            xla_gpu_allocator=previous.xla_gpu_allocator,
            tf_gpu_allocator=previous.tf_gpu_allocator,
            configure_runtime=False,
        )
