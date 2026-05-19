# Perlmutter GPU Test Plan

Date: 2026-05-19

Scope: run the simsopt-jax GPU proof ladder on NERSC Perlmutter using a
commit-accurate source snapshot and Slurm jobs that record enough provenance to
make the result usable for release-grade GPU signoff.

## Direct Answers

- Project/account name: yes, required. Fill `GPU_ACCOUNT` with the NERSC GPU
  allocation account exactly as `iris` reports it. Some NERSC GPU account names
  are project names ending in `_g`; use the value assigned to this user/project.
- External sample scripts: not required. The repo already has the proof commands;
  this plan uses small repo-specific Slurm wrappers around them.
- `debug` QOS: do not use it as the main lane. Use `debug` only for a tiny
  batch canary that is guaranteed to finish under 30 minutes. Use `shared` for
  the normal 1-GPU smoke/proof jobs, and use `interactive` only for manual live
  debugging.

## Current Local Preconditions

At the time this plan was written:

- Local branch: `gpu-purity-stage2-20260405`
- Local HEAD: `8418fabf728d25354cb29b303ea8419b79a6333f`
- Upstream tracking branch: `fork/gpu-purity-stage2-20260405`
- Local branch was ahead of upstream and the working tree was dirty.

Do not rsync the whole local working tree to Perlmutter. The ignored
`.artifacts/` directory is large and not part of the proof source. Before
running on Perlmutter, choose one source mode:

1. Preferred: commit the intended source, push the branch, and test that exact
   SHA.
2. Alternative: create a git bundle/archive for a clean committed SHA.
3. If uncommitted changes must be tested, create an explicit patch or filtered
   transfer with a recorded file list. Treat that as a dirty-tree proof, not a
   clean release proof.

## Queue Recommendation

Use this queue order:

1. `shared`: default lane for 1-GPU smoke and normal proof steps.
2. `interactive`: only when a human is actively debugging on the allocation.
3. `debug`: optional tiny canary only, not the proof ladder.
4. `regular`: only if a later run needs full-node or multi-GPU behavior.

Reason: JAX cold compilation, repo import, and native extension checks can burn
most of a 30 minute `debug` window. NERSC recommends `shared` QOS for jobs using
1 or 2 GPUs.

## Required Inputs

Set these before writing/submitting Slurm scripts:

```bash
export GPU_ACCOUNT="<gpu_account_from_iris>"
export REPO_SHA="<exact_sha_to_test>"
export REPO_URL="git@github.com:jungdaesuh/simsopt.git"
export REPO_REF="gpu-purity-stage2-20260405"
export SCRATCH_ROOT="${SCRATCH}/simsopt-jax-${REPO_SHA}"
```

For the production proof wrapper, also provide one of:

```bash
export SINGLE_STAGE_WARM_START_RUN_DIR="<path_to_warm_start_run_dir>"
# or
export SINGLE_STAGE_JAX_RUNTIME_SEED_SPEC="<path_to_runtime_seed_spec_json>"
```

The production proof requires `--single-stage-warm-start-run-dir` or
`--single-stage-jax-runtime-seed-spec`.

## Source Setup On Perlmutter Login Node

Run setup on a login node, not inside a GPU allocation:

```bash
set -euo pipefail

: "${GPU_ACCOUNT:?set GPU_ACCOUNT}"
: "${REPO_SHA:?set REPO_SHA}"
: "${REPO_URL:?set REPO_URL}"
: "${REPO_REF:?set REPO_REF}"
: "${SCRATCH_ROOT:?set SCRATCH_ROOT}"
JAX_GPU_WHEEL_SPEC="${JAX_GPU_WHEEL_SPEC:-jax[cuda12]==0.10.0}"

mkdir -p "${SCRATCH_ROOT}"
cd "${SCRATCH_ROOT}"

if [ ! -d repo ]; then
  git clone --recursive --branch "${REPO_REF}" --single-branch "${REPO_URL}" repo
fi

cd repo
git fetch origin "${REPO_REF}"
git checkout "${REPO_SHA}"
git submodule update --init --recursive

python3 -m venv "${SCRATCH_ROOT}/venv"
. "${SCRATCH_ROOT}/venv/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade "${JAX_GPU_WHEEL_SPEC}"
python -m pip install -e ".[test,ALGS]" "shapely>=2.1,<3" "numba>=0.64,<0.66"
```

The virtualenv above is the pip-wheel GPU proof runtime. It is not GPU signoff
by itself until the Slurm GPU smoke records a CUDA/GPU backend. The current
candidate is `jax[cuda12]==0.10.0`; a 2026-05-19 dry-run resolved that exact
wheel set for Linux `manylinux_2_27_x86_64` / Python 3.11. Record a fresh
dry-run before launch, or use the NERSC-recommended NVIDIA JAX container lane.
Do not launch the legacy `jax[cuda12]==0.9.2` lane; its required
`jax-cuda12-plugin==0.9.2` wheel was not available for the target. Do not reuse
a CPU-only `jax` / `jaxlib` virtualenv for the canary, shared smoke, or proof
jobs below.

Verify the local extension resolves from this checkout:

```bash
cd "${SCRATCH_ROOT}/repo"
. "${SCRATCH_ROOT}/venv/bin/activate"
python - <<'PY'
from repo_bootstrap import bootstrap_local_simsopt
bootstrap_local_simsopt("src")
import simsoptpp
print(simsoptpp.__file__)
PY
```

## Optional Tiny Debug Canary

Use this only to prove that the allocation/account and CUDA visibility work.
Skip it if the `shared` smoke is already easy to submit.

`perlmutter_gpu_debug_canary.slurm`:

```bash
#!/usr/bin/env bash
#SBATCH -A <gpu_account>
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J simsopt-jax-gpu-canary
#SBATCH -o logs/gpu_canary_%j.out
#SBATCH -e logs/gpu_canary_%j.err

set -euo pipefail

cd "<scratch_root>/repo"
. "<scratch_root>/venv/bin/activate"

export SLURM_CPU_BIND="cores"
export PYTHONPATH="$PWD:$PWD/src"
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

srun -n 1 -c 32 --gpus-per-task=1 bash -lc '
  nvidia-smi
  python - <<PY
import jax
import jaxlib
import importlib.metadata as metadata
print("jax", jax.__version__)
print("jaxlib", jaxlib.__version__)
print("jax-cuda12-plugin", metadata.version("jax-cuda12-plugin"))
print("jax-cuda12-pjrt", metadata.version("jax-cuda12-pjrt"))
print("backend", jax.default_backend())
print("devices", jax.devices())
print("x64", jax.config.read("jax_enable_x64"))
assert jax.default_backend() in {"cuda", "gpu"}
assert jax.config.read("jax_enable_x64") is True
PY
'
```

Submit:

```bash
mkdir -p logs
sbatch perlmutter_gpu_debug_canary.slurm
```

## Main 1-GPU Shared Smoke

Use `shared` for the first real check.

`perlmutter_gpu_shared_smoke.slurm`:

```bash
#!/usr/bin/env bash
#SBATCH -A <gpu_account>
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J simsopt-jax-gpu-smoke
#SBATCH -o logs/gpu_smoke_%j.out
#SBATCH -e logs/gpu_smoke_%j.err

set -euo pipefail

cd "<scratch_root>/repo"
. "<scratch_root>/venv/bin/activate"

export SLURM_CPU_BIND="cores"
export PYTHONPATH="$PWD:$PWD/src"
export JAX_ENABLE_X64=1
export SIMSOPT_BACKEND_MODE=jax_gpu_parity
export SIMSOPT_JAX_PLATFORM=cuda
export SIMSOPT_EXAMPLE_PARITY_JAX_PLATFORM=cuda
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

RESULTS_DIR="${SCRATCH}/simsopt-jax-results/${SLURM_JOB_ID}_smoke"
mkdir -p "${RESULTS_DIR}"

srun -n 1 -c 32 --gpus-per-task=1 bash -lc '
  set -euo pipefail
  nvidia-smi | tee "'"${RESULTS_DIR}"'/nvidia-smi.txt"
  python - <<PY | tee "'"${RESULTS_DIR}"'/jax_smoke.txt"
import json
import importlib.metadata as metadata
import jax
import jaxlib
from repo_bootstrap import bootstrap_local_simsopt

bootstrap_local_simsopt("src")
import simsopt
import simsoptpp

payload = {
    "slurm_job_id": "'"${SLURM_JOB_ID}"'",
    "jax": jax.__version__,
    "jaxlib": jaxlib.__version__,
    "jax_cuda12_plugin": metadata.version("jax-cuda12-plugin"),
    "jax_cuda12_pjrt": metadata.version("jax-cuda12-pjrt"),
    "backend": jax.default_backend(),
    "devices": [str(device) for device in jax.devices()],
    "x64": bool(jax.config.read("jax_enable_x64")),
    "simsopt": getattr(simsopt, "__version__", None),
    "simsoptpp": simsoptpp.__file__,
}
print(json.dumps(payload, indent=2, sort_keys=True))
assert payload["backend"] in {"cuda", "gpu"}
assert payload["x64"] is True
PY
'
```

## Proof Ladder

Run these only after the shared smoke passes.

### 1. CPU Baseline For Non-Banana

The CPU baseline is the oracle artifact for the GPU follow-up. It can run as a
CPU-only process inside the same prepared checkout, but do not run expensive CPU
work on a login node.

```bash
export PYTHONPATH="$PWD:$PWD/src"
export JAX_ENABLE_X64=1
export SIMSOPT_EXAMPLE_PARITY_JAX_PLATFORM=cpu
export JAX_PLATFORMS=cpu

python benchmarks/non_banana_example_cpp_jax_cpu_parity.py \
  --lanes cpu_cpp,jax_cpu \
  --output-json "${RESULTS_DIR}/non_banana_cpu_baseline.json"
```

### 2. Non-Banana GPU Follow-Up

This lane intentionally requires exact CUDA selector environment:

```bash
export PYTHONPATH="$PWD:$PWD/src"
export JAX_ENABLE_X64=1
export SIMSOPT_BACKEND_MODE=jax_gpu_parity
export SIMSOPT_JAX_PLATFORM=cuda
export SIMSOPT_EXAMPLE_PARITY_JAX_PLATFORM=cuda
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python benchmarks/non_banana_example_cpp_jax_cpu_parity.py \
  --lanes cpu_cpp,jax_gpu \
  --baseline-json "${RESULTS_DIR}/non_banana_cpu_baseline.json" \
  --output-json "${RESULTS_DIR}/non_banana_gpu_followup.json"
```

### 3. Stage 2 CUDA Proof

Use the repo helper behavior for this script. It requests CUDA and may append
CPU for callback fallback while keeping CUDA as the default backend.

```bash
export PYTHONPATH="$PWD:$PWD/src"
export JAX_ENABLE_X64=1
export SIMSOPT_BACKEND_MODE=jax_gpu_parity
export SIMSOPT_JAX_PLATFORM=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python benchmarks/stage2_e2e_comparison.py \
  --platform cuda \
  --equilibria-dir examples/single_stage_optimization/equilibria \
  --output-json "${RESULTS_DIR}/stage2_cuda.json"
```

### 4. Single-Stage CUDA Proof

```bash
export PYTHONPATH="$PWD:$PWD/src"
export JAX_ENABLE_X64=1
export SIMSOPT_BACKEND_MODE=jax_gpu_parity
export SIMSOPT_JAX_PLATFORM=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python benchmarks/single_stage_init_parity.py \
  --platform cuda \
  --equilibria-dir examples/single_stage_optimization/equilibria \
  --case-artifacts-dir "${RESULTS_DIR}/artifacts/single_stage_cuda" \
  --output-json "${RESULTS_DIR}/single_stage_cuda.json"
```

### 5. Production Proof Body

Use the existing repo proof body instead of inventing a separate contract:

```bash
python -m pip install -v -e .

python benchmarks/stage2_e2e_comparison.py \
  --results-dir "${RESULTS_DIR}/stage2" \
  --stage2-bs-path benchmarks/fixtures/single_stage_seed_iota15/biot_savart_opt.json \
  --stage2-platform cuda

python benchmarks/single_stage_outer_loop_probe.py \
  --results-dir "${RESULTS_DIR}/single_stage" \
  --platform cuda \
  --warm-start-run-dir "${SINGLE_STAGE_WARM_START_RUN_DIR}"
```

If using a runtime seed spec instead of a warm-start run directory:

```bash
python benchmarks/stage2_e2e_comparison.py \
  --results-dir "${RESULTS_DIR}/stage2" \
  --stage2-bs-path benchmarks/fixtures/single_stage_seed_iota15/biot_savart_opt.json \
  --stage2-platform cuda

python benchmarks/single_stage_outer_loop_probe.py \
  --results-dir "${RESULTS_DIR}/single_stage" \
  --platform cuda \
  --jax-runtime-seed-spec "${SINGLE_STAGE_JAX_RUNTIME_SEED_SPEC}"
```

## Acceptance Criteria

A GPU artifact is valid only if it records:

- exact repo SHA
- dirty tree status
- Slurm job id
- `nvidia-smi` output
- NVIDIA driver and CUDA runtime visible to the job
- `jax` and `jaxlib` versions
- `jax-cuda12-plugin` and `jax-cuda12-pjrt` versions for pip-wheel GPU runs
- `jax.default_backend()` as CUDA/GPU
- `jax.devices()`
- `JAX_ENABLE_X64=1` and JAX x64 enabled
- CUDA visibility and platform selector env
- peak RSS and peak GPU memory where available
- pass/fail plus failure list

Reject the GPU signoff if:

- backend is CPU
- x64 is false
- SHA differs from the intended proof SHA
- CUDA provenance is missing
- the artifact only proves CPU or JAX CPU behavior

Keep the trust chain explicit:

```text
Existing SIMSOPT C++/SciPy behavior
-> JAX CPU matches
-> JAX CUDA/GPU matches
-> JAX CPU and GPU match each other
```

JAX-vs-JAX agreement alone is not enough.

## Useful NERSC Commands

Discover account/project:

```bash
iris
```

Check scratch:

```bash
printf 'SCRATCH=%s\nPSCRATCH=%s\n' "$SCRATCH" "$PSCRATCH"
```

Submit and watch:

```bash
sbatch perlmutter_gpu_shared_smoke.slurm
squeue -u "$USER"
sacct -j <job_id> --format=JobID,JobName,State,Elapsed,AllocTRES%50
```

Manual interactive allocation when needed:

```bash
salloc -N 1 -C gpu -q interactive -t 00:30:00 --gpus=4 -A <gpu_account>
```

## Sources

- NERSC Perlmutter running jobs: https://docs.nersc.gov/systems/perlmutter/running-jobs/
- NERSC interactive jobs: https://docs.nersc.gov/jobs/interactive/
- NERSC queue policy: https://docs.nersc.gov/jobs/policy/
- JAX installation: https://docs.jax.dev/en/latest/installation.html
- JAX GPU memory allocation: https://docs.jax.dev/en/latest/gpu_memory_allocation.html
