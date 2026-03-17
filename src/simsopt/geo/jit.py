from jax import jit as jaxjit
from .config import parameters


def jit(fun, **kwargs):
    if parameters["jit"]:
        return jaxjit(fun, **kwargs)
    else:
        return fun
