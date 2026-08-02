"""Emit one deterministic FP64 accepted-state trajectory for this receipt."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
from simsopt_jax.geo.optimizers.private._lbfgs import (
    _minimize_lbfgs_private_value_and_grad,
)


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


result = _minimize_lbfgs_private_value_and_grad(
    value_and_grad,
    jnp.asarray([-1.2, 1.0], dtype=jnp.float64),
    maxiter=3,
    maxcor=3,
    ftol=0.0,
    gtol=1.0e-12,
    maxls=20,
    record_optimizer_state_trace=True,
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
            "trace": [
                {
                    "fun": float(entry["fun"]),
                    "iteration": int(entry["iteration"]),
                    "jac": np.asarray(entry["jac"]).tolist(),
                    "nfev": int(entry["nfev"]),
                    "njev": int(entry["njev"]),
                    "x": np.asarray(entry["x"]).tolist(),
                }
                for entry in result.optimizer_state_trace
            ],
            "x": np.asarray(result.x_k).tolist(),
        },
        sort_keys=True,
    )
)
