import os

import jax

if "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
    jax.config.update("jax_platform_name", "cpu")
if (
    "JAX_ENABLE_X64" not in os.environ
    and os.environ.get("SIMSOPT_BACKEND_MODE", "native_cpu") == "native_cpu"
):
    jax.config.update("jax_enable_x64", True)

from jax import jit as jaxjit
from .config import parameters


def native_jax_device():
    """The device the native objects evaluate upstream's jitted JAX kernels on.

    The host CPU device, reached by explicit ``device_put``/``device_get``, so a
    native objective stays host-owned under ``jax.transfer_guard("disallow")``
    and never lands on a GPU in a process whose default backend is CUDA. A
    process whose JAX initialized CUDA alone (``JAX_PLATFORMS=cuda``, the strict
    jax-gpu parity lane) has no host JAX device; there the kernels run on that
    process's default device through the same explicit transfers.
    """
    platforms = jax.config.jax_platforms
    if platforms and "cpu" not in platforms.split(","):
        return jax.devices()[0]
    return jax.devices("cpu")[0]


def jit(fun, **args):
    if parameters['jit']:
        return jaxjit(fun, **args)
    else:
        return fun
