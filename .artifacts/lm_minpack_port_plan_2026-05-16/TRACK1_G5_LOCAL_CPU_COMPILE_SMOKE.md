# Track 1 G5 Local CPU Compile Smoke

Date: 2026-05-17

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .conda/jax/bin/python - <<'PY'
...
PY
```

The measured call prepared the current oversampled Boozer fixture from
`build_ls_parity_problem(ncoils=4, nphi=16, ntheta=8)`, then timed the first
local CPU call to `target_least_squares(..., method="lm-minpack-ondevice")`
after `jax.clear_caches()`. The measured result was synchronized with
`jax.block_until_ready((result.x, result.fun, result.jac, result.residual, result.residual_jacobian, result.hessian))`.
This is a local CPU cold compile smoke, not the CUDA performance gate in
`docs/source/jax_acceptance.rst`.

Result:

| Field | Value |
|---|---:|
| JAX backend | `cpu` |
| Python | `3.11.15` |
| JAX | `0.10.0` |
| jaxlib | `0.10.0` |
| NumPy | `2.4.3` |
| OS | `macOS-26.2-arm64-arm-64bit` |
| machine | `arm64` |
| processor | `arm` |
| x64 enabled | `true` |
| residual shape | `(386,)` |
| state shape | `(39,)` |
| elapsed seconds | `3.6744802079629153` |
| success | `true` |
| iterations | `124` |
| objective | `3.9276553235153005e-06` |
| residual inf norm | `3.0146161663506586e-04` |
| repeat count | `1` |
| warmup runs | `0` |
| cache reset | `jax.clear_caches() before measured call` |
| synchronization | `jax.block_until_ready((result.x, result.fun, result.jac, result.residual, result.residual_jacobian, result.hessian))` |
| timing method | `time.perf_counter around first call after jax.clear_caches with explicit JAX result synchronization` |
| random seed | `n/a` |

G5 decision: PASS as a local CPU smoke for the revised Track 1 dense-QR lane.
This artifact does not certify CUDA first-compile performance.
