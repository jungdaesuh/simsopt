# Beta Quickstart — Banana Coil Optimization on JAX

Minimal recipes for running banana coil optimization with the JAX backend
(on-device L-BFGS-B). Pick a device, copy the matching block, run.

For the full workflow (Stage 2 seed handoff, runtime specs, parity gates,
production artifacts), see [`README.md`](./README.md) in this folder.

---

## What gets run

Both scripts default to the **JAX on-device L-BFGS-B** optimizer
(`lbfgs-ondevice`) when you pass `--backend jax` — on every device.
Everything compiles to XLA and executes on whatever device
`SIMSOPT_BACKEND_MODE` selects: CPU, CUDA, or Apple Silicon (MPS).

| Script | What it optimizes |
|---|---|
| `STAGE_2/banana_coil_solver.py` | Coil geometry against a target plasma surface. Outputs `biot_savart_opt.json`. |
| `SINGLE_STAGE/single_stage_banana_example.py` | Quasi-symmetry + Boozer coordinates using the Stage 2 coils. |

---

## Install (one-time)

From the repo root:

```bash
# CPU-only / Apple Silicon
python -m pip install -e ".[deploy]"

# CUDA (Linux + NVIDIA GPU)
python -m pip install -e ".[deploy_gpu]"

# Apple Silicon GPU (MPS) — Python >= 3.13 on darwin
python -m pip install -e ".[JAX,JAX_MPS]"
```

---

## Stage 2 — banana coil solver

### NVIDIA GPU (CUDA)

```bash
SIMSOPT_BACKEND_MODE=jax_gpu_fast \
SIMSOPT_JAX_PLATFORM=cuda \
SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled \
JAX_ENABLE_X64=1 \
JAX_PLATFORMS=cuda,cpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
env -u LD_LIBRARY_PATH \
python examples/single_stage_optimization/STAGE_2/banana_coil_solver.py \
  --backend jax \
  --plasma-surf-filename wout_nfp22ginsburg_000_014417_iota15.nc \
  --major-radius 0.915 \
  --toroidal-flux 0.24 \
  --banana-surf-radius 0.22
```

### CPU (JAX/XLA, not C++)

```bash
SIMSOPT_BACKEND_MODE=jax_cpu_fast \
JAX_ENABLE_X64=1 \
JAX_PLATFORMS=cpu \
python examples/single_stage_optimization/STAGE_2/banana_coil_solver.py \
  --backend jax \
  --plasma-surf-filename wout_nfp22ginsburg_000_014417_iota15.nc \
  --major-radius 0.915 \
  --toroidal-flux 0.24 \
  --banana-surf-radius 0.22
```

### Apple Silicon GPU (MPS)

```bash
SIMSOPT_BACKEND_MODE=jax_mps_smoke \
SIMSOPT_JAX_PLATFORM=mps \
JAX_ENABLE_X64=0 \
JAX_PLATFORMS=mps \
python examples/single_stage_optimization/STAGE_2/banana_coil_solver.py \
  --backend jax \
  --plasma-surf-filename wout_nfp22ginsburg_000_014417_iota15.nc \
  --major-radius 0.915 \
  --toroidal-flux 0.24 \
  --banana-surf-radius 0.22
```

MPS is float32-only by policy (`runtime_dtype="float32"`,
`tolerance_tier="float32_smoke"`). `JAX_ENABLE_X64=0` is required — x64 on
MPS will fail. The Python interpreter must be ≥3.13 on darwin with `jax-mps`
installed (see install section).

---

## Single-stage — quasi-symmetry + Boozer

Needs a Stage 2 seed first. Either:

- a real Stage 2 run output: `STAGE2_BS_PATH=/path/to/biot_savart_opt.json`, or
- the checked-in fixture:
  `benchmarks/fixtures/single_stage_seed_iota15/biot_savart_opt.json`

### NVIDIA GPU (CUDA)

```bash
SIMSOPT_BACKEND_MODE=jax_gpu_fast \
SIMSOPT_JAX_PLATFORM=cuda \
SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled \
JAX_ENABLE_X64=1 \
JAX_PLATFORMS=cuda,cpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
env -u LD_LIBRARY_PATH \
python examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
  --backend jax \
  --stage2-bs-path benchmarks/fixtures/single_stage_seed_iota15/biot_savart_opt.json \
  --iota-target 0.17 \
  --vol-target 0.10 \
  --cc-dist 0.07 \
  --mpol 10 \
  --ntor 10
```

### CPU (JAX/XLA)

```bash
SIMSOPT_BACKEND_MODE=jax_cpu_fast \
JAX_ENABLE_X64=1 \
JAX_PLATFORMS=cpu \
python examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
  --backend jax \
  --stage2-bs-path benchmarks/fixtures/single_stage_seed_iota15/biot_savart_opt.json \
  --iota-target 0.17 \
  --vol-target 0.10 \
  --cc-dist 0.07 \
  --mpol 10 \
  --ntor 10
```

### Apple Silicon GPU (MPS)

```bash
SIMSOPT_BACKEND_MODE=jax_mps_smoke \
SIMSOPT_JAX_PLATFORM=mps \
JAX_ENABLE_X64=0 \
JAX_PLATFORMS=mps \
python examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
  --backend jax \
  --stage2-bs-path benchmarks/fixtures/single_stage_seed_iota15/biot_savart_opt.json \
  --iota-target 0.17 \
  --vol-target 0.10 \
  --cc-dist 0.07 \
  --mpol 10 \
  --ntor 10
```

---

## What you get on each device

| Device | Dtype | Optimizer | Notes |
|---|---|---|---|
| CUDA (`jax_gpu_fast`) | float64 | on-device L-BFGS-B (`lbfgs-ondevice`) | Production-grade speed lane. Default sharding=`hybrid` (multi-device-capable). The companion `jax_gpu_parity` mode pins sharding=`none` (single-device) until a multi-GPU parity/speedup proof is recorded. |
| CPU (`jax_cpu_fast`) | float64 | on-device L-BFGS-B (`lbfgs-ondevice`) | Same JAX SETULB on XLA-CPU. Slower than CUDA, no infra friction. |
| MPS (`jax_mps_smoke`) | float32 | on-device L-BFGS-B (`lbfgs-ondevice`) | Apple Silicon GPU. Experimental — float32 only, runtime relies on the third-party `jax-mps` PJRT plugin. Treat results as a smoke run, not a production claim. |

---

## Verifying the lane

After launch, confirm the backend was actually selected:

```python
from simsopt.backend import get_backend_config, get_jax_platform
cfg = get_backend_config()
print(cfg.mode, get_jax_platform())
# Expect e.g. "jax_gpu_fast cuda"
```

The run's `results.json` also records `optimizer_backend="ondevice"` and
`outer_optimizer_method="lbfgs-ondevice"` when the on-device lane was used.

---

## When to switch off `_fast`

The `_fast` modes (`jax_cpu_fast`, `jax_gpu_fast`) skip the byte-identity
gate against the C++ reference: `matmul_precision="default"` and
`chunk_policy="performance_tuned"`. That is fine for production banana
runs — your acceptance criteria are the physics objectives, not exact
match to C++.

Switch to `jax_cpu_parity` / `jax_gpu_parity` if you need to publish a
result that claims algorithmic parity to the C++ oracle. Those modes pin
`matmul_precision="highest"` and `chunk_policy="stable_default"`, and run
5–20× slower; see the main [`README.md`](./README.md) for the full
parity-mode invocation.

---

## Known issues (current as of 2026-05-20)

- **CUDA on Runpod-style hosts:** jaxlib cubin v12.9 vs system nvlink
  mismatch has been observed. Local launcher patches exist but are not
  upstream yet. If you hit `cublasLt` errors, prefer a host with NVIDIA
  driver ≥ 555 and avoid mixing system CUDA libs (`unset LD_LIBRARY_PATH` /
  `env -u LD_LIBRARY_PATH`).
- **MPS plugin:** the runtime depends on the third-party
  `tillahoffmann/jax-mps` PJRT plugin, which replaced the unmaintained
  Apple `jax-metal` plugin (see `runtime.py:43-56`). The mode policy carries
  `default_optimizer_backend="scipy"` to signal the lane is not validated for
  production accuracy; the example scripts still default to `lbfgs-ondevice`
  for any `--backend jax`. Use MPS for laptop dev only.

---

## Where to look next

- Full production workflow + parity recipes: [`README.md`](./README.md)
- GPU runbook + ship gates:
  [`../../docs/source/jax_gpu_setup.rst`](../../docs/source/jax_gpu_setup.rst)
- Mode-by-mode policy details:
  [`../../docs/using_jax_backend.md`](../../docs/using_jax_backend.md)
