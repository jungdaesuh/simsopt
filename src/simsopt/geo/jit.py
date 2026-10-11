from functools import wraps

import jax
import numpy as np
jax.config.update('jax_platform_name', 'cpu')
from jax import jit as jaxjit
from .config import parameters


def _owned(value):
    # A NumPy argument may be a view of a native cache that the next DOF change
    # overwrites; JAX can read it after the call returns, so pass a private copy.
    return np.array(value) if isinstance(value, np.ndarray) else value


def jit(fun, **args):
    run = jaxjit(fun, **args) if parameters['jit'] else fun

    @wraps(fun)
    def call(*fargs, **fkwargs):
        fargs, fkwargs = jax.tree_util.tree_map(_owned, (fargs, fkwargs))
        return run(*fargs, **fkwargs)

    return call
