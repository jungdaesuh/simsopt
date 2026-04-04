"""Shared utilities for private optimizer submodules.

PRIVATE_OPTIMIZER_JAX_VERSION and _x64_enabled are imported from the public
module (``optimizer_jax``).  This is safe because both are defined *before*
the public module's ``try: from .optimizer_jax_private import ...`` block,
so they exist on the partially-initialized module object when this file loads.
"""

from __future__ import annotations

from functools import partial

import jax.numpy as jnp
from jax import lax

from ..optimizer_jax import PRIVATE_OPTIMIZER_JAX_VERSION, _x64_enabled

_dot = partial(jnp.dot, precision=lax.Precision.HIGHEST)
_einsum = partial(jnp.einsum, precision=lax.Precision.HIGHEST)


def _norm(x, *, ord=None):
    return jnp.linalg.norm(x, ord=ord)


def _promote_dtypes_inexact(*args):
    """Promote arguments to a shared inexact dtype using public JAX APIs."""
    dtype = jnp.result_type(*args)
    if not jnp.issubdtype(dtype, jnp.inexact):
        dtype = jnp.promote_types(dtype, jnp.float32)
    return tuple(jnp.asarray(arg, dtype=dtype) for arg in args)


def _require_private_optimizer_runtime(x0):
    import jax

    if jax.__version__ != PRIVATE_OPTIMIZER_JAX_VERSION:
        raise RuntimeError(
            f"On-device optimizer is validated on JAX "
            f"{PRIVATE_OPTIMIZER_JAX_VERSION}; found {jax.__version__}. "
            "Use envs/jax-0.9.2.yml for the supported runtime or "
            "fall back to optimizer_backend='scipy'."
        )
    if not _x64_enabled():
        raise RuntimeError(
            "On-device optimizer requires jax_enable_x64=True before import/use."
        )

    x0 = jnp.asarray(x0, dtype=jnp.float64)
    if x0.ndim != 1:
        raise ValueError(
            f"On-device optimizer expects a flat 1-D decision vector, got {x0.shape}."
        )
    return x0


def _resolve_lbfgs_limits(d, maxiter, maxfun, maxgrad):
    """Resolve None defaults for L-BFGS iteration/eval/grad limits."""
    import numpy as np

    if (maxiter is None) and (maxfun is None) and (maxgrad is None):
        maxiter = d * 200
    if maxiter is None:
        maxiter = np.inf
    if maxfun is None:
        maxfun = np.inf
    if maxgrad is None:
        maxgrad = np.inf
    return maxiter, maxfun, maxgrad


def _emit_iteration_callbacks(callback, progress_callback, x_kp1, next_k, f_kp1, g_kp1):
    """Dispatch callback/progress_callback via jax.debug.callback.

    Shared by the BFGS and L-BFGS on-device body functions.
    """
    import jax
    import numpy as np

    if callback is not None:
        jax.debug.callback(
            lambda x: callback(np.asarray(x, dtype=float)),
            x_kp1,
            ordered=True,
        )
    if progress_callback is not None:
        grad_inf = _norm(g_kp1, ord=jnp.inf)
        jax.debug.callback(
            lambda iteration, fun_value, grad_inf_value: progress_callback(
                int(iteration),
                float(fun_value),
                float(grad_inf_value),
            ),
            next_k,
            f_kp1,
            grad_inf,
            ordered=True,
        )
