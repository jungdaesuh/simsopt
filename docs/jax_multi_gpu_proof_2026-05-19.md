# JAX Multi-GPU Sharding Proof — 2026-05-19

This artifact records the round-3 N30 proof for real Perlmutter multi-GPU
sharding. It is intentionally separate from the full Tier 5 performance
characterization because the Tier 5 GPU run hit host-memory/walltime pressure
before producing a usable multi-GPU sharding result.

## Official Documentation Checked

- JAX documents `Mesh`, `NamedSharding`, `PartitionSpec`, `jax.device_put`, and
  `jax.shard_map` as the explicit multi-device sharding path. Computations on
  sharded arrays are automatically parallelized across the mesh devices.
- JAX GPU memory documentation states that
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` disables default GPU preallocation, and
  `XLA_PYTHON_CLIENT_MEM_FRACTION` controls the preallocation fraction when
  preallocation is enabled.
- NERSC Perlmutter documentation describes GPU nodes as four NVIDIA A100 GPUs
  per node and recommends `shared` for one- or two-GPU jobs; this proof uses a
  `regular` full-node job because it must compare 1 / 2 / 4 visible GPUs in the
  same hardware allocation.

Sources:

- JAX sharding docs via Context7 `/google/jax`:
  `https://context7.com/google/jax/llms.txt`
- JAX GPU memory allocation:
  `https://github.com/google/jax/blob/main/docs/gpu_memory_allocation.rst`
- NERSC Perlmutter architecture:
  `https://docs.nersc.gov/systems/perlmutter/architecture/`
- NERSC running jobs and queue policy:
  `https://docs.nersc.gov/systems/perlmutter/running-jobs/`,
  `https://docs.nersc.gov/jobs/policy/`

## Probe

Script:

- `benchmarks/jax_multi_gpu_sharding_probe.py`

Perlmutter wrapper:

- `.artifacts/perlmutter_setup/perlmutter_multi_gpu_sharding_probe.slurm`

Environment:

```bash
JAX_ENABLE_X64=True
JAX_PLATFORMS=cuda,cpu
SIMSOPT_JAX_PLATFORM=cuda
SIMSOPT_BACKEND_MODE=jax_gpu_parity
SIMSOPT_JAX_SHARDING=points
SIMSOPT_JAX_MIN_POINTS_TO_SHARD=1
SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

The wrapper runs each probe in a fresh Python process with
`CUDA_VISIBLE_DEVICES` set to `0`, `0,1`, and `0,1,2,3`. This matters because
JAX device enumeration and backend initialization happen at process import time.

## Results

Artifact root:

`/pscratch/sd/j/jungdae/simsopt-jax-results/jax-0.10.0-e2e-memperf-customvjp-20260519T095319Z/wave8_multi_gpu_sharding`

Primary Slurm job:

- `53168132` (`regular`, full GPU node, completed)

Corroborating debug job:

- `53168131` (`debug`, full GPU node, completed)

Both jobs wrote `aggregate.json` with `"passed": true` and an empty
`failures` list.

Important diagnostic: earlier non-pre-sharded debug probes (`53167952` and
`53168089`) failed the surface-integral speedup acceptance even though sharding
was active. Those failures measured repeated input placement from a single GPU
into a multi-device mesh. The passing runs below pre-place the timed inputs on
the mesh once, matching JAX's documented `device_put` + `NamedSharding` usage.

### `integral_BdotN_surface_sharded`

| GPUs | Median wall time (s) | Speedup vs 1 GPU | Peak GPU memory | Sharding | HLO collectives |
| --- | ---: | ---: | --- | --- | --- |
| 1 | 0.010654 | 1.00x | max `{"0": 16831}` MiB; delta `{"0": 16396}` MiB | single device baseline | none |
| 2 | 0.005247 | 2.03x | max `{"0": 17147, "1": 8941}` MiB; delta `{"0": 16712, "1": 8512}` MiB | `NamedSharding`, mesh `{"d": 2}`, `P('d', None, None)` | `all_reduce_count=1` |
| 4 | 0.002756 | 3.87x | max `{"0": 17483, "1": 5181, "2": 5181, "3": 5181}` MiB; delta `{"0": 17048, "1": 4752, "2": 4752, "3": 4752}` MiB | `NamedSharding`, mesh `{"d": 4}`, `P('d', None, None)` | `all_reduce_count=1` |

### Seed-Batch Scoring

| GPUs | Median wall time (s) | Speedup vs 1 GPU | Peak GPU memory | Sharding | HLO collectives |
| --- | ---: | ---: | --- | --- | --- |
| 1 | 0.539941 | 1.00x | max `{"0": 2507}` MiB; delta `{"0": 2072}` MiB | single device baseline | none |
| 2 | 0.280070 | 1.93x | max `{"0": 2627, "1": 585}` MiB; delta `{"0": 2192, "1": 156}` MiB | `NamedSharding`, mesh `{"d": 2}`, `P('d',)` | none expected |
| 4 | 0.142815 | 3.78x | max `{"0": 3159, "1": 1099, "2": 1099, "3": 1099}` MiB; delta `{"0": 2724, "1": 670, "2": 670, "3": 670}` MiB | `NamedSharding`, mesh `{"d": 4}`, `P('d',)` | none expected |

## Acceptance

- [x] `integral_BdotN_surface_sharded` runs on JAX CUDA/GPU for 1 / 2 / 4
  visible GPUs.
- [x] `integral_BdotN_surface_sharded` reports active `NamedSharding` for 2 and
  4 visible GPUs.
- [x] `integral_BdotN_surface_sharded` preserves parity with the unsharded
  scalar reference at `rtol=1e-12`, `atol=1e-12`.
- [x] `integral_BdotN_surface_sharded` exceeds 1.5x speedup at 2 GPUs and 2.5x
  speedup at 4 GPUs against the same-node 1-GPU baseline.
- [x] Seed-batch scoring runs on JAX CUDA/GPU for 1 / 2 / 4 visible GPUs.
- [x] Seed-batch scoring reports active `NamedSharding` for 2 and 4 visible
  GPUs.
- [x] Seed-batch scoring preserves value/gradient parity with the `vmap`
  reference at `rtol=1e-12`, `atol=1e-12`.
- [x] Seed-batch scoring exceeds 1.5x speedup at 2 GPUs and 2.5x speedup at 4
  GPUs against the same-node 1-GPU baseline.
- [x] Peak GPU memory is recorded for every probe row.

## Single-Stage Init Parity Follow-Up

The steady-state proof above does not exercise the private optimizer adapter
and Boozer penalty geometry construction path used by the real single-stage
initialization. A follow-up hardware run closed that gap after adding explicit
active replicated placement for private optimizer arrays and Boozer penalty
geometry leaves.

Artifact:

`/pscratch/sd/j/jungdae/simsopt-jax-results/jax-0.10.0-e2e-memperf-customvjp-20260519T095319Z/wave8_multi_gpu_sharding_presharded_regular/single_stage_init_parity_post_review_debug20_allow`

Job:

- `53170493` (`debug`, 4 A100 GPUs, completed)

Result:

- `single_stage_cuda_init.json` has `"passed": true`.
- `stdout` ends with `SINGLE-STAGE INIT PARITY PASSED`.
- JAX/JAXLIB `0.10.0`; backend `gpu`; devices `cuda:0`, `cuda:1`,
  `cuda:2`, `cuda:3`; x64 enabled.
- `SIMSOPT_BACKEND_MODE=jax_gpu_parity`,
  `SIMSOPT_JAX_SHARDING=points`, `SIMSOPT_JAX_MIN_POINTS_TO_SHARD=1`.
- CPU vs JAX: `|iota diff|=0.00e+00`, volume relative difference
  `0.00e+00`, field-error relative difference `2.51e-16`, surface relative
  difference `0.00e+00`.
- Slurm elapsed `7:34`; `/usr/bin/time` wall `7:27.85`.
- `/usr/bin/time` MaxRSS `6285132` KB; Slurm batch MaxRSS `7905296K`.
- Early stdout memory snapshot: peak RSS `1077.7 MB`, GPU memory `435.0 MB`.
- `nvidia_smi_before.csv` and `nvidia_smi_after.csv` show all four
  `NVIDIA A100-SXM4-80GB` devices at `0 MiB` before and after the command.

Diagnostic:

- Job `53169133` reached `SINGLE-STAGE INIT PARITY PASSED` and wrote the same
  kind of JSON/time/memory artifacts, but Slurm marked it `TIMEOUT` because the
  10-minute debug walltime was shorter than the command's `10:33.41` runtime.
  Job `53170493` is the clean post-review completion artifact.

## Verdict

Passed for the steady-state pre-sharded multi-GPU contract. This closes the
hardware-gated N30 proof for the surface-quadrature and seed-batch sharding
paths. The follow-up single-stage init parity job also passed on four A100s
with active point sharding, covering the private optimizer and Boozer penalty
geometry adapter path that the steady-state probe does not execute.
