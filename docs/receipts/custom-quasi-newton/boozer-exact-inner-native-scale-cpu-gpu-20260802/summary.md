# Native-scale exact Boozer inner-solve diagnostic

Verdict: `diagnostic-pass-not-promotion`

Source commit: `19194b957db00d78ab6062a9685c4de5c7941ecc`

## Commands

Both runs used the public example with `--max-steps 2 --json`, FP64, strict
backend selection, and `JAX_TRANSFER_GUARD=disallow`. The exact Newton path
contains the documented scoped transfer allowance around JAX GMRES.

CPU:

```text
env JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
  SIMSOPT_BACKEND_MODE=jax_cpu_parity SIMSOPT_BACKEND_STRICT=1 \
  SIMSOPT_JAX_TRANSFER_GUARD=disallow SIMSOPT_PRECISION=fp64 \
  JAX_TRANSFER_GUARD=disallow \
  ./.venv-qn-cpu/bin/python \
  examples/jax/3_Advanced/single_stage_boozer_vacuum_optimization.py \
  --max-steps 2 --json
```

GPU:

```text
env JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
  SIMSOPT_BACKEND_MODE=jax_gpu_parity SIMSOPT_BACKEND_STRICT=1 \
  SIMSOPT_JAX_TRANSFER_GUARD=disallow SIMSOPT_PRECISION=fp64 \
  JAX_TRANSFER_GUARD=disallow XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ./.venv-qn-gpu/bin/python \
  examples/jax/3_Advanced/single_stage_boozer_vacuum_optimization.py \
  --max-steps 2 --json
```

## Results

| lane | inner success | Boozer residual | residual RMS | outer status | outer iterations |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU | true | 1.9847460904955183e-29 | 6.300390607725077e-15 | 2 | 0 |
| RTX 5090 GPU | true | 2.516488847956131e-29 | 7.094348240615386e-15 | 2 | 0 |

The exact inner solve succeeds at native resolution (`resolution=6`, 255
state variables). The outer result is not a convergence result: both parity
runs accepted zero outer steps and are retained only as inner-solve evidence.
The selected JSON observables are in `raw/cpu.json` and `raw/gpu.json`.

Runtime versions: JAX `0.10.0`, NumPy `2.4.6`, SciPy `1.17.1`.
