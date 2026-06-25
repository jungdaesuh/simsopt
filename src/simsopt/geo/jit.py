import os

import jax

if "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
    jax.config.update("jax_platform_name", "cpu")

from jax import jit as jaxjit
from .config import parameters


def jit(fun, **args):
    if parameters['jit']:
        return jaxjit(fun, **args)
    else:
        return fun
