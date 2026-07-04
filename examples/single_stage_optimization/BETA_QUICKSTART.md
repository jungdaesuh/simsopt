# Beta Quickstart — Banana Coil Optimization on JAX

Minimal recipes for running banana coil optimization with the JAX backend.
Bare `--backend jax` uses the regular `scipy-jax` optimizer lane in Stage 2
and the decomposed `scipy-jax-decomposed` optimizer lane in single-stage by default.
The plain single-stage `scipy-jax` lane is deprecated; keep it only for legacy
reduced-lane diagnostics.
Add `--optimizer-backend ondevice` only when you explicitly want the compiled
on-device L-BFGS-B stress lane. Pick a device, copy the matching block, run.

For the full workflow (Stage 2 seed handoff, runtime specs, parity gates,
production artifacts), see [`README.md`](./README.md) in this folder.

---

## What gets run

Both scripts keep the outer optimizer on the host while evaluating the objective
and derivatives through JAX/XLA on the selected backend. Stage 2 defaults to
`scipy-jax` (`lbfgs-scipy-jax`); single-stage defaults to
`scipy-jax-decomposed` (`lbfgs-scipy-jax-decomposed`). Explicit
`--optimizer-backend ondevice` remains available for the compiled
`lbfgs-ondevice` stress lane; explicit single-stage `scipy-jax` is deprecated.

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

## What you get on each device

| Device | Dtype | Optimizer | Notes |
|---|---|---|---|
| CUDA (`jax_gpu_fast`) | float64 | Stage 2 default `scipy-jax`; single-stage default `scipy-jax-decomposed`; explicit `ondevice` = `lbfgs-ondevice` | Production speed lane for objective/gradient evaluation. Default sharding=`hybrid` (multi-device-capable). The companion `jax_gpu_parity` mode pins sharding=`none` (single-device) until a multi-GPU parity/speedup proof is recorded. |
| CPU (`jax_cpu_fast`) | float64 | Stage 2 default `scipy-jax`; single-stage default `scipy-jax-decomposed`; explicit `ondevice` = `lbfgs-ondevice` | Host SciPy control with XLA-CPU objective evaluation by default. The explicit on-device lane is available but compile-heavy on CPU. |

---

## Verifying the lane

After launch, confirm the backend was actually selected:

```python
from simsopt_jax.backend import get_backend_config, get_jax_platform
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
---

## Where to look next

- Full production workflow + parity recipes: [`README.md`](./README.md)
- GPU runbook + ship gates:
  [`../../docs/source/jax_gpu_setup.rst`](../../docs/source/jax_gpu_setup.rst)
- Mode-by-mode policy details:
  [`../../docs/using_jax_backend.md`](../../docs/using_jax_backend.md)
