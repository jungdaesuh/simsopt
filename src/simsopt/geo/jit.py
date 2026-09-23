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


def jit(fun, **args):
    if parameters["jit"]:
        return jaxjit(fun, **args)
    else:
        return fun
