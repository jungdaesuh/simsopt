import os
from functools import wraps

import jax
from jax import jit as jaxjit

from .config import parameters

if "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
    jax.config.update("jax_platform_name", "cpu")


def native_jax_device():
    """Device on which native simsopt runs the functions compiled by :func:`jit`.

    Returns:
        jax.Device object: First CPU device when CPU is enabled, otherwise the
            first default device (for example with ``JAX_PLATFORMS=cuda``).
    """
    platforms = jax.config.jax_platforms
    if platforms and "cpu" not in platforms.split(","):
        return jax.devices()[0]
    return jax.devices("cpu")[0]


def jit(fun, **args):
    """Compile ``fun`` with :func:`jax.jit` for native simsopt.

    Each call runs under ``jax.default_device(native_jax_device())``, so NumPy
    and other uncommitted inputs, and the results, stay on the CPU even when
    another platform such as CUDA is the default backend. Inputs committed to a
    device keep their placement. When ``parameters['jit']`` is false, ``fun`` is
    returned unchanged.

    Args:
        fun (callable): Function to compile.
        **args: Keyword arguments forwarded to :func:`jax.jit`.

    Returns:
        callable: The compiled function, or ``fun`` when jit is disabled.
    """
    if not parameters['jit']:
        return fun
    compiled = jaxjit(fun, **args)

    @wraps(fun)
    def on_native_device(*fun_args, **fun_kwargs):
        with jax.default_device(native_jax_device()):
            return compiled(*fun_args, **fun_kwargs)

    return on_native_device
