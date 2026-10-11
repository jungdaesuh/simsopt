from functools import wraps

import jax
import numpy as np
jax.config.update('jax_platform_name', 'cpu')
from jax import jit as jaxjit
from .config import parameters


def _owned(value):
    # A NumPy argument may be a view of a native cache that the next DOF change
    # overwrites; JAX can read it after dispatch returns, so pass a private copy.
    return np.array(value) if isinstance(value, np.ndarray) else value


def jit(fun, **args):
    if not parameters['jit']:
        return fun
    compiled = jaxjit(fun, **args)

    @wraps(fun)
    def call(*fargs, **fkwargs):
        return compiled(*map(_owned, fargs), **{k: _owned(v) for k, v in fkwargs.items()})

    return call
