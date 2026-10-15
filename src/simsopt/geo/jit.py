import os
from functools import wraps

import jax
import numpy as np
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


def _owned(value):
    # Native caches can overwrite NumPy views while JAX still reads them.
    return np.array(value, copy=True) if isinstance(value, np.ndarray) else value


def jit(fun, **args):
    """Compile ``fun`` with :func:`jax.jit` for native simsopt.

    Each call snapshots NumPy pytree leaves, including when jit is disabled.
    Compiled calls run under ``jax.default_device(native_jax_device())``, so NumPy
    and other uncommitted inputs, and the results, stay on the CPU even when
    another platform such as CUDA is the default backend. Inputs committed to a
    device keep their placement. When ``parameters['jit']`` is false, only the
    input snapshots are applied before calling ``fun``.

    Args:
        fun (callable): Function to compile.
        **args: Keyword arguments forwarded to :func:`jax.jit`.

    Returns:
        callable: The input-owning wrapper, compiled when jit is enabled.
    """
    use_jit = parameters['jit']
    run = jaxjit(fun, **args) if use_jit else fun

    @wraps(fun)
    def on_native_device(*fun_args, **fun_kwargs):
        fun_args, fun_kwargs = jax.tree_util.tree_map(_owned, (fun_args, fun_kwargs))
        if not use_jit:
            return run(*fun_args, **fun_kwargs)
        with jax.default_device(native_jax_device()):
            return run(*fun_args, **fun_kwargs)

    return on_native_device
