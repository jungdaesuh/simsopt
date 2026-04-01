try:
    from jax import jit as jaxjit
except ImportError:
    jaxjit = None
from .config import parameters


def jit(fun, **kwargs):
    if parameters["jit"] and jaxjit is not None:
        return jaxjit(fun, **kwargs)
    return fun
