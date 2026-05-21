# VMEC Compute Geometry JAX CUDA Bench Note — 2026-05-20

## Scope

This note records the N3.6 benchmark artifact for
`vmec_compute_geometry_jax` on the pinned low-resolution VMEC fixture.

## Environment

- Host: NERSC Perlmutter GPU debug queue
- Job: `53204761` (`ljax-n3bench`)
- Submit line: `sbatch -A m4680_g -q debug -C gpu -N 1 -c 32 --gpus-per-node=1 -t 00:30:00 -J ljax-n3bench`
- Runtime: `jax==0.10.0`, CUDA backend
- Environment: `JAX_ENABLE_X64=True`, `JAX_PLATFORM_NAME=cuda`,
  `JAX_PLATFORMS=cuda,cpu`,
  `XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true`

## Method

The benchmark lowered a `jax.jit`-compiled scalar token over
`vmec_compute_geometry_jax(frozen, s, theta, phi)` using JAX's AOT lowering
surface and counted StableHLO operation lines from
`lowered.as_text(dialect="stablehlo")`.

Input shape:

- `s`: `[2]`
- `theta`: `[4]`
- `phi`: `[5]`

Fixture:

- `tests/test_files/wout_li383_low_res_reference.nc`

## Result

- Backend: `gpu`
- Devices visible to JAX: `cuda:0`, `cuda:1`, `cuda:2`, `cuda:3`
- StableHLO node count: `6236`
- One-call wall time after compile/warmup: `0.001767153007676825` seconds

The raw JSON artifact is:

```json
{"devices": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"], "hlo_dialect": "stablehlo", "hlo_node_count": 6236, "jax_version": "0.10.0", "one_call_wall_time_seconds": 0.001767153007676825, "platform": "gpu", "shape": {"phi": [5], "s": [2], "theta": [4]}}
```
