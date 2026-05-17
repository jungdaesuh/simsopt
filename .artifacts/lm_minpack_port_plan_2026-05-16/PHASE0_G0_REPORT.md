# Phase 0 G0 Feasibility Report

Date: 2026-05-16

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .conda/jax/bin/python .artifacts/lm_minpack_port_plan_2026-05-16/phase0_g0_probe.py
```

## Result

`G0_PASS=False`

The Track 1 CPU-byte-exact MINPACK spike fails at the Phase 0 gate on the
production BoozerSurface shape `(384, 40)`.

## Evidence

Official docs/upstream checked before running the probe:

- JAX docs via Context7 for `jax.scipy.linalg.qr`, `jax.lax.linalg.qr`,
  `ormqr`, `householder_product`, and `jax.ffi.ffi_call`.
- Installed JAX 0.10.0 upstream source for `jax._src.lax.linalg.geqp3`,
  `ormqr`, `householder_product`, and the `geqp3_ffi` / `hybrid_geqp3`
  lowering.
- SciPy docs via Context7 for `qr_multiply` and `least_squares(method="lm")`.
- Installed SciPy 1.17.1 docstring/source for `qr_multiply` and
  LAPACK `dgeqp3`.
- Netlib MINPACK `qrfac.f` and `lmder.f`.

The probe compares JAX's strongest currently reachable packed pivoted-QR path:

- `jax._src.lax.linalg.geqp3`
- `jax._src.lax.linalg.ormqr`

against SciPy's LAPACK-backed oracle:

- `scipy.linalg.lapack.dgeqp3`
- `scipy.linalg.qr_multiply(..., mode="right", pivoting=True, conjugate=True)`

Observed outcomes:

| Shape | Seeds | Packed factor | Pivots | Taus | `Q^T f` |
|---|---:|---|---|---|---|
| `(40, 40)` | 1 | bit-equal | bit-equal | bit-equal | bit-equal |
| `(75, 39)` | 1 | bit-equal | bit-equal | bit-equal | bit-equal |
| `(100, 50)` | 1 | bit-equal | bit-equal | bit-equal | bit-equal |
| `(384, 40)` | 100 | not bit-equal | bit-equal | mixed | not bit-equal |
| `(2000, 80)` | 1 | not bit-equal | bit-equal | not bit-equal | not bit-equal |

Representative max absolute differences:

- `(384, 40)` packed factor: `3.55e-15` typical, `7.11e-15` worst observed
- `(384, 40)` taus: `0` or `2.22e-16`
- `(384, 40)` `Q^T f`: `8.88e-16` to `2.66e-15`
- `(2000, 80)` packed factor: `7.11e-15`
- `(2000, 80)` `Q^T f`: `1.33e-15`

## Gate Decision

Gate G0 requires bit-equal packed `fjac` and `qtb` on `(384, 40)` for all
100 random seeds and conjunctive agreement across at least two attempted paths.
The strongest reachable path already fails the required production shape.

Per `PLAN.md` Phase 0, Track 1 is abandoned at production scope. Re-scoping to
`m ~= n` only is not a default fallback and would require owner sign-off.

Track 2 remains valid and is unaffected by this gate failure.
