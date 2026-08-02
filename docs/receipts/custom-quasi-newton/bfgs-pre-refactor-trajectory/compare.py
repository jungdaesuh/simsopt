"""Emit a deterministic FP64 BFGS accepted-state trajectory."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
from simsopt_jax.geo.optimizers.private._bfgs import _minimize_bfgs_private


def value_and_grad(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    value = 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2
    gradient = jnp.asarray(
        [
            -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0]),
            200.0 * (x[1] - x[0] ** 2),
        ],
        dtype=x.dtype,
    )
    return value, gradient


accepted: list[np.ndarray] = []


def callback(x: np.ndarray) -> None:
    accepted.append(np.asarray(x, dtype=np.float64).copy())


result = _minimize_bfgs_private(
    value_and_grad,
    jnp.asarray((-1.2, 1.0), dtype=jnp.float64),
    maxiter=3,
    gtol=1.0e-12,
    xrtol=0.0,
    line_search_maxiter=20,
    callback=callback,
    value_and_grad=True,
)
jax.effects_barrier()

trace = []
for x in accepted:
    value, gradient = value_and_grad(jnp.asarray(x, dtype=jnp.float64))
    trace.append(
        {
            "fun": float(value),
            "jac": np.asarray(gradient).tolist(),
            "x": x.tolist(),
        }
    )

print(
    json.dumps(
        {
            "f": float(result.f_k),
            "g": np.asarray(result.g_k).tolist(),
            "k": int(result.k),
            "nfev": int(result.nfev),
            "ngev": int(result.ngev),
            "status": int(result.status),
            "trace": trace,
            "x": np.asarray(result.x_k).tolist(),
        },
        sort_keys=True,
    )
)
