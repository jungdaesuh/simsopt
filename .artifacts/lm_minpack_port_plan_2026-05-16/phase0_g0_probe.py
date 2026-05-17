"""Probe Track 1 Gate G0 packed-QR feasibility.

This is an execution artifact for PLAN.md Phase 0, not production code. It
compares JAX's strongest currently reachable packed pivoted-QR path against
SciPy's LAPACK-backed oracle on the shapes required by the plan.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla
from scipy.linalg import lapack

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax._src.lax import linalg as lax_linalg


SHAPE_SEEDS = (
    ((40, 40), (0,)),
    ((75, 39), (0,)),
    ((100, 50), (7,)),
    ((384, 40), tuple(range(100))),
    ((2000, 80), (0,)),
)


def _scipy_geqp3(a):
    geqp3 = lapack.get_lapack_funcs("geqp3", (a,))
    _, _, _, work_query, info_query = geqp3(a.copy(), lwork=-1, overwrite_a=False)
    if info_query != 0:
        raise RuntimeError(f"SciPy geqp3 lwork query failed with info={info_query}")
    lwork = int(work_query[0])
    packed, pivots, taus, _work, info = geqp3(
        a.copy(),
        lwork=lwork,
        overwrite_a=False,
    )
    if info != 0:
        raise RuntimeError(f"SciPy geqp3 failed with info={info}")
    return packed, pivots, taus


def _probe_case(shape, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(shape)
    fvec = rng.standard_normal(shape[0])

    jax_packed, jax_pivots, jax_taus = lax_linalg.geqp3(
        jnp.asarray(a),
        jnp.zeros(shape[1], dtype=jnp.int32),
    )
    jax_packed = np.asarray(jax_packed)
    jax_pivots = np.asarray(jax_pivots)
    jax_taus = np.asarray(jax_taus)

    scipy_packed, scipy_pivots, scipy_taus = _scipy_geqp3(a)
    qtb_jax_full = np.asarray(
        lax_linalg.ormqr(
            jnp.asarray(jax_packed),
            jnp.asarray(jax_taus),
            jnp.asarray(fvec[:, None]),
            left=True,
            transpose=True,
        )
    )[:, 0]
    qtb_scipy, _r, scipy_qr_pivots = sla.qr_multiply(
        a,
        fvec[None, :],
        mode="right",
        pivoting=True,
        conjugate=True,
    )
    qtb_scipy = qtb_scipy[0]

    return {
        "shape": shape,
        "seed": seed,
        "packed_bit_equal": np.array_equal(jax_packed, scipy_packed),
        "packed_max_abs": float(np.max(np.abs(jax_packed - scipy_packed))),
        "pivots_bit_equal": np.array_equal(jax_pivots, scipy_pivots),
        "taus_bit_equal": np.array_equal(jax_taus, scipy_taus),
        "taus_max_abs": float(np.max(np.abs(jax_taus - scipy_taus))),
        "qtb_bit_equal": np.array_equal(qtb_jax_full[: qtb_scipy.size], qtb_scipy),
        "qtb_max_abs": float(
            np.max(np.abs(qtb_jax_full[: qtb_scipy.size] - qtb_scipy))
        ),
        "qr_pivots_bit_equal": np.array_equal(jax_pivots - 1, scipy_qr_pivots),
    }


def main():
    rows = []
    for shape, seeds in SHAPE_SEEDS:
        for seed in seeds:
            rows.append(_probe_case(shape, seed))

    failures = [
        row
        for row in rows
        if not (
            row["packed_bit_equal"]
            and row["pivots_bit_equal"]
            and row["taus_bit_equal"]
            and row["qtb_bit_equal"]
            and row["qr_pivots_bit_equal"]
        )
    ]
    print(
        "shape,seed,packed_eq,packed_max_abs,pivots_eq,taus_eq,taus_max_abs,qtb_eq,qtb_max_abs,qr_pivots_eq"
    )
    for row in rows:
        print(
            "{shape},{seed},{packed_bit_equal},{packed_max_abs:.17e},"
            "{pivots_bit_equal},{taus_bit_equal},{taus_max_abs:.17e},"
            "{qtb_bit_equal},{qtb_max_abs:.17e},{qr_pivots_bit_equal}".format(**row)
        )
    print(f"G0_PASS={not failures}")
    print(f"G0_FAILURE_COUNT={len(failures)}")


if __name__ == "__main__":
    main()
