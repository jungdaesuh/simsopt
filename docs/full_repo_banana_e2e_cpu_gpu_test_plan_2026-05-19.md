# Full Repo, Banana E2E, CPU/GPU Parity, And Performance Test Plan

Date: 2026-05-19

Purpose: define a release-grade test campaign for this repository that covers
the full Python test suite, the banana Stage 2 and single-stage end-to-end
contracts, CPU/GPU parity, and performance characterization on Perlmutter.

This is a plan and execution checklist. It does not claim the tests have been
run.

## Goals

- [ ] Test one exact repo snapshot, not an implicit local working directory.
- [ ] Run the full repo test suite on CPU.
- [ ] Run the focused banana CPU/JAX correctness suite.
- [ ] Run banana Stage 2 and single-stage end-to-end CPU proof artifacts.
- [ ] Run real CUDA/GPU hardware preflight on Perlmutter.
- [ ] Run hardware-gated GPU tests and proof scripts on Perlmutter.
- [ ] Establish the trust chain:

  ```text
  Existing SIMSOPT C++/SciPy behavior
  -> JAX CPU matches
  -> JAX CUDA/GPU matches
  -> JAX CPU and GPU match each other
  ```

- [ ] Capture timing and memory data separately from correctness pass/fail.
- [ ] Produce a final report with exact SHA, dirty-tree status, Slurm job ids,
  hardware facts, artifacts, failures, and accepted residual risk.

## Non-Goals

- [ ] Do not treat CPU-only artifacts as GPU signoff.
- [ ] Do not treat JAX CPU vs JAX GPU agreement as enough without the
  C++/SciPy CPU oracle.
- [ ] Do not loosen tolerances to hide drift.
- [ ] Do not run production proof from a login node.
- [ ] Do not rsync `.artifacts/`, local virtualenvs, `.conda`, or an arbitrary
  dirty working tree to Perlmutter.
- [ ] Do not use Perlmutter `debug` QOS as the main proof lane. It is only for
  tiny canaries.

## Rationale

The repo has several different proof surfaces:

- Standard `pytest` coverage catches broad regressions across the package.
- Marker-targeted tests (`stage2`, `single_stage`, `boozer`, `integration`,
  `slow`) cover the banana and parity-specific contracts more directly.
- Scripted proof artifacts under `benchmarks/` record provenance and structured
  pass/fail data that normal pytest output does not capture.
- Performance characterization is meaningful only after correctness passes and
  only when the artifact records backend, devices, x64, CUDA visibility, memory,
  and exact source provenance.

The test campaign therefore runs in waves. Each wave either produces a durable
artifact or blocks the next wave with a concrete failure.

## Required Inputs

- [x] Perlmutter GPU account: `m4680_g`
  - `m4680_g` is the GPU allocation account reported by NERSC. The `_g`
    suffix is part of the GPU project/account name, not an arbitrary local
    convention.
- [ ] Perlmutter CPU account for CPU-only jobs: `<cpu_account_from_iris>`
- [ ] Exact repo SHA to test: `<repo_sha>`
- [ ] Source mode:
  - [ ] clean committed SHA pushed to `fork/gpu-purity-stage2-20260405`
  - [ ] git bundle/archive for a clean committed SHA
  - [ ] explicit patch/file-list transfer for a dirty-tree proof
- [ ] Single-stage production proof seed:
  - [ ] `SINGLE_STAGE_WARM_START_RUN_DIR=<path>`
  - [ ] or `SINGLE_STAGE_JAX_RUNTIME_SEED_SPEC=<path>`
- [ ] Output root:

  ```bash
  export GPU_ACCOUNT="m4680_g"
  export REPO_SHA="<repo_sha>"
  export REPO_REF="gpu-purity-stage2-20260405"
  export REPO_URL="git@github.com:jungdaesuh/simsopt.git"
  export SCRATCH_ROOT="${SCRATCH}/simsopt-jax-${REPO_SHA}"
  export ENV_ROOT="${SCRATCH_ROOT}/conda-env"
  export RESULTS_ROOT="${SCRATCH}/simsopt-jax-results/${REPO_SHA}"
  export STAGE2_GEOMETRY_REPRO_MAXITER=21
  export STAGE2_GEOMETRY_REL_TOL=1e-6
  ```

## Source And Environment Setup

### Local Source Freeze

- [ ] Check local state:

  ```bash
  git status --short --branch
  git rev-parse HEAD
  git log --oneline --decorate -5
  ```

- [ ] Decide whether the proof target is clean committed HEAD or dirty-tree
  bytes.
- [ ] If clean proof: commit intended changes, push the branch, and record the
  exact SHA.
- [ ] If dirty proof: generate a patch and file manifest, and mark all
  artifacts as dirty-tree evidence.

### Perlmutter Login-Node Setup

Run this on a login node, not inside a GPU allocation.

```bash
set -euo pipefail

: "${GPU_ACCOUNT:?set GPU_ACCOUNT}"
: "${REPO_SHA:?set REPO_SHA}"
: "${REPO_REF:?set REPO_REF}"
: "${REPO_URL:?set REPO_URL}"
: "${SCRATCH_ROOT:?set SCRATCH_ROOT}"
: "${ENV_ROOT:?set ENV_ROOT}"
: "${RESULTS_ROOT:?set RESULTS_ROOT}"

mkdir -p "${SCRATCH_ROOT}" "${RESULTS_ROOT}"
cd "${SCRATCH_ROOT}"

if [ ! -d repo ]; then
  git clone --recursive --branch "${REPO_REF}" --single-branch "${REPO_URL}" repo
fi

cd repo
git fetch origin "${REPO_REF}"
git checkout "${REPO_SHA}"
git submodule update --init --recursive

module load python
conda create -y -p "${ENV_ROOT}" python=3.11 pip numpy scipy
conda activate "${ENV_ROOT}"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade "jax[cuda12]==0.9.2"
python -m pip install -e ".[deploy_gpu]"

export SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled

python - <<'PY'
import jax
import jaxlib

if jax.__version__ != "0.9.2" or jaxlib.__version__ != "0.9.2":
    raise SystemExit(
        f"expected jax/jaxlib 0.9.2, got {jax.__version__}/{jaxlib.__version__}"
    )
PY

python - <<'PY'
from repo_bootstrap import bootstrap_local_simsopt
bootstrap_local_simsopt("src")
import simsoptpp
print(simsoptpp.__file__)
PY
```

The first GPU proof intentionally pins `jax[cuda12]==0.9.2` even though local
CPU development environments may move faster. The repo's production GPU proof
image and `SIMSOPT_BENCHMARK_JAX_VERSION` performance contract currently use
0.9.2, so this plan pins the hardware proof to that known runtime before
comparing or benchmarking.

Container environment lane: NERSC's Python/JAX guidance recommends NVIDIA JAX
containers through Shifter or Podman-HPC as the most reliable GPU setup on
Perlmutter. Treat the conda/pip lane above and a container lane as separate
environment choices made before Wave 0. Do not switch environment lanes within a
single proof bundle; if the environment lane changes, rerun all waves from the
same source snapshot and label the artifacts with that lane.

The pip-wheel lane uses JAX's bundled CUDA userspace libraries. Do not load a
separate `cudatoolkit` module for this lane, and keep
`SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled` in GPU jobs so repo subprocess helpers
do not prepend a local CUDA toolkit or `LD_LIBRARY_PATH` over the wheel stack.

Record setup provenance:

- [ ] `git rev-parse HEAD`
- [ ] `git status --short --untracked-files=no`
- [ ] `python --version`
- [ ] `python -m pip freeze`
- [ ] `python -c 'import jax, jaxlib; print(jax.__version__, jaxlib.__version__)'`
- [ ] `python -c 'import simsopt, simsoptpp; print(simsopt.__version__, simsoptpp.__file__)'`

### Common Slurm Job Prologue

Every batch script that uses the piped commands below starts with this prologue
so `pytest | tee ...` and proof-script pipelines preserve the failing command's
exit status.

```bash
#!/usr/bin/env bash
set -euo pipefail

: "${SCRATCH_ROOT:?set SCRATCH_ROOT}"
: "${ENV_ROOT:?set ENV_ROOT}"
: "${RESULTS_ROOT:?set RESULTS_ROOT}"

module load python
conda activate "${ENV_ROOT}"

cd "${SCRATCH_ROOT}/repo"
export PYTHONPATH="$PWD:$PWD/src"
```

## Wave 0: CPU Import And Full Test Baseline

Purpose: establish that the repo passes its full CPU-side suite before GPU
hardware is involved.

Run Wave 0 on a CPU compute node, not a login node. NERSC login nodes are
resource-limited and are not intended for significant full-suite pytest runs.
Use an interactive CPU allocation for manual debugging or a batch script for the
actual baseline.

Example CPU allocation:

```bash
salloc -A <cpu_account_from_iris> -C cpu -q interactive -t 02:00:00 -N 1
```

Example CPU batch header:

```bash
#SBATCH -A <cpu_account_from_iris>
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -t 02:00:00
#SBATCH -n 1
#SBATCH -c 32
```

Use the CPU project account reported by `iris`; do not assume the GPU account
`m4680_g` is accepted for CPU-only jobs. The batch header above intentionally
uses CPU `shared` QOS for a 32-logical-CPU pytest lane. If running under
whole-node CPU `regular` QOS with one task, use NERSC's CPU-node affinity
formula and request `-c 256` instead.

Environment:

```bash
cd "${SCRATCH_ROOT}/repo"
export PYTHONPATH="$PWD:$PWD/src"
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu
export SIMSOPT_JAX_PLATFORM=cpu
export SIMSOPT_EXAMPLE_PARITY_JAX_PLATFORM=cpu
mkdir -p "${RESULTS_ROOT}/wave0_cpu_full"
```

Checklist:

- [ ] Import smoke:

  ```bash
  python -m pytest tests/test_jax_import_smoke.py \
    -ra --tb=short --durations=50 \
    --junitxml="${RESULTS_ROOT}/wave0_cpu_full/import_smoke.xml" \
    | tee "${RESULTS_ROOT}/wave0_cpu_full/import_smoke.log"
  ```

- [ ] Full repo test suite:

  ```bash
  python -m pytest tests \
    -ra --tb=short --durations=100 \
    --junitxml="${RESULTS_ROOT}/wave0_cpu_full/full_tests.xml" \
    | tee "${RESULTS_ROOT}/wave0_cpu_full/full_tests.log"
  ```

- [ ] Marker-level reruns for summary clarity:

  ```bash
  python -m pytest tests -m "integration or stage2 or single_stage or boozer" \
    -ra --tb=short --durations=100 \
    --junitxml="${RESULTS_ROOT}/wave0_cpu_full/focused_markers.xml" \
    | tee "${RESULTS_ROOT}/wave0_cpu_full/focused_markers.log"
  ```

Acceptance:

- [ ] Full CPU suite passes, or every failure is categorized as known,
  non-regression, environment-only, or blocker.
- [ ] No GPU signoff is inferred from this wave.

## Wave 1: Focused CPU Banana Correctness

Purpose: isolate the banana-specific correctness surface before CUDA.

Commands:

```bash
mkdir -p "${RESULTS_ROOT}/wave1_cpu_banana"

python -m pytest \
  tests/integration/test_stage2_jax.py \
  tests/integration/test_stage2_target_lane_purity.py \
  tests/integration/test_single_stage_jax.py \
  tests/integration/test_single_stage_jax_cpu_reference.py \
  tests/integration/test_single_stage_physics_parity.py \
  tests/geo/test_single_stage_example.py \
  tests/geo/test_single_stage_continuation.py \
  tests/geo/test_boozersurface_jax.py \
  tests/geo/test_boozer_derivatives_jax.py \
  -ra --tb=short --durations=100 \
  --junitxml="${RESULTS_ROOT}/wave1_cpu_banana/banana_cpu_focused.xml" \
  | tee "${RESULTS_ROOT}/wave1_cpu_banana/banana_cpu_focused.log"
```

Structured CPU proof artifacts:

```bash
python benchmarks/non_banana_example_cpp_jax_cpu_parity.py \
  --lanes cpu_cpp,jax_cpu \
  --output-json "${RESULTS_ROOT}/wave1_cpu_banana/non_banana_cpu_baseline.json"

python benchmarks/stage2_e2e_comparison.py \
  --platform cpu \
  --equilibria-dir examples/single_stage_optimization/equilibria \
  --output-json "${RESULTS_ROOT}/wave1_cpu_banana/stage2_cpu_e2e.json"

python benchmarks/stage2_e2e_comparison.py \
  --platform cpu \
  --maxiter "${STAGE2_GEOMETRY_REPRO_MAXITER}" \
  --geometry-rel-tol "${STAGE2_GEOMETRY_REL_TOL}" \
  --equilibria-dir examples/single_stage_optimization/equilibria \
  --output-json "${RESULTS_ROOT}/wave1_cpu_banana/stage2_cpu_e2e_geometry_repro.json"

python benchmarks/single_stage_init_parity.py \
  --platform cpu \
  --equilibria-dir examples/single_stage_optimization/equilibria \
  --case-artifacts-dir "${RESULTS_ROOT}/wave1_cpu_banana/artifacts/single_stage_cpu" \
  --output-json "${RESULTS_ROOT}/wave1_cpu_banana/single_stage_cpu_init.json"
```

Optional CPU outer-loop artifact:

```bash
python benchmarks/single_stage_outer_loop_probe.py \
  --platform cpu \
  --equilibria-dir examples/single_stage_optimization/equilibria \
  --output-json "${RESULTS_ROOT}/wave1_cpu_banana/single_stage_cpu_outer_loop.json"
```

Acceptance:

- [ ] `stage2_cpu_e2e.json` has `passed: true`.
- [ ] `stage2_cpu_e2e_geometry_repro.json` has `passed: true` and gates final
  banana-coil geometry through `geometry_rel_tol`.
- [ ] `single_stage_cpu_init.json` has `passed: true`.
- [ ] CPU/C++/SciPy oracle and JAX CPU candidate are both represented.
- [ ] Any CPU banana failure blocks GPU correctness interpretation.

## Wave 2: Perlmutter GPU Preflight

Purpose: prove that the Slurm allocation sees an NVIDIA GPU and that JAX
initializes CUDA with x64.

Queue: `shared`, not `debug`, for the normal run.

Slurm header:

```bash
#SBATCH -A m4680_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
```

Preflight body:

```bash
cd "${SCRATCH_ROOT}/repo"
conda activate "${ENV_ROOT}"

export PYTHONPATH="$PWD:$PWD/src"
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cuda,cpu
export SIMSOPT_JAX_PLATFORM=cuda
export SIMSOPT_BACKEND_MODE=jax_gpu_parity
export SIMSOPT_EXAMPLE_PARITY_JAX_PLATFORM=cuda
export SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_exclude_nondeterministic_ops=true"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "${RESULTS_ROOT}/wave2_gpu_preflight"

srun -n 1 -c 32 --cpu-bind=cores --gpus-per-task=1 bash -lc '
  set -euo pipefail
  nvidia-smi | tee "'"${RESULTS_ROOT}"'/wave2_gpu_preflight/nvidia-smi.txt"
  python - <<PY | tee "'"${RESULTS_ROOT}"'/wave2_gpu_preflight/jax_gpu_preflight.json"
import json
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
    "backend": jax.default_backend(),
    "devices": [str(device) for device in jax.devices()],
    "jax_platforms": "'"${JAX_PLATFORMS}"'",
    "cuda_library_mode": "'"${SIMSOPT_JAX_CUDA_LIBRARY_MODE}"'",
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

Acceptance:

- [ ] `nvidia-smi` reports an A100 GPU.
- [ ] JAX default backend is CUDA/GPU.
- [ ] `JAX_PLATFORMS=cuda,cpu`; CUDA is first and remains the default backend.
- [ ] JAX x64 is true.
- [ ] `simsoptpp` imports from the prepared checkout.
- [ ] Slurm job id and hardware facts are saved.

## Wave 3: GPU Hardware-Gated Pytest Slices

Purpose: run the tests that exercise CUDA-specific runtime boundaries before
the heavier proof scripts.

Environment inside the GPU job:

```bash
export PYTHONPATH="$PWD:$PWD/src"
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cuda,cpu
export SIMSOPT_JAX_PLATFORM=cuda
export SIMSOPT_BACKEND_MODE=jax_gpu_parity
export SIMSOPT_EXAMPLE_PARITY_JAX_PLATFORM=cuda
export SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_exclude_nondeterministic_ops=true"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

Commands:

```bash
mkdir -p "${RESULTS_ROOT}/wave3_gpu_pytest"

python -m pytest \
  tests/test_jax_import_smoke.py::test_transfer_guard_disallow_allows_gpu_ondevice_loops_with_host_constants \
  tests/test_jax_import_smoke.py::test_transfer_guard_disallow_allows_grouped_biot_savart_gpu_spec_eval \
  tests/test_jax_import_smoke.py::test_transfer_guard_disallow_allows_grouped_biot_savart_gpu_current_arrays \
  tests/test_jax_import_smoke.py::test_transfer_guard_disallow_allows_stage2_target_objective_host_closure_constants \
  tests/test_jax_import_smoke.py::test_transfer_guard_disallow_allows_stage2_target_objective_ondevice_entry \
  tests/test_jax_import_smoke.py::test_transfer_guard_disallow_allows_single_stage_surface_self_intersection \
  -ra --tb=short --durations=50 \
  --junitxml="${RESULTS_ROOT}/wave3_gpu_pytest/gpu_runtime_smoke.xml" \
  | tee "${RESULTS_ROOT}/wave3_gpu_pytest/gpu_runtime_smoke.log"
```

Run the grouped coil collective lowering control separately. That test forces a
CPU host-platform sharding setup inside its subprocess and is useful regression
coverage, but it is not GPU signoff.

```bash
python -m pytest \
  tests/test_jax_import_smoke.py::test_grouped_biot_savart_coil_collective_parity_and_lowering \
  -ra --tb=short --durations=50 \
  --junitxml="${RESULTS_ROOT}/wave3_gpu_pytest/grouped_collective_cpu_lowering.xml" \
  | tee "${RESULTS_ROOT}/wave3_gpu_pytest/grouped_collective_cpu_lowering.log"
```

Then run the real-fixture GPU M5 parity class:

```bash
python -m pytest \
  tests/integration/test_single_stage_jax_cpu_reference.py::TestRealFixtureGpuM5Parity \
  -ra --tb=short --durations=50 \
  --junitxml="${RESULTS_ROOT}/wave3_gpu_pytest/single_stage_gpu_m5.xml" \
  | tee "${RESULTS_ROOT}/wave3_gpu_pytest/single_stage_gpu_m5.log"
```

Acceptance:

- [ ] No CUDA runtime boundary smoke fails.
- [ ] Grouped coil collective lowering control passes, but is not counted as
  CUDA proof.
- [ ] Real-fixture GPU M5 parity class passes or produces a concrete failure
  artifact.
- [ ] Any skip must be justified by environment facts, not by missing CUDA.

## Wave 4: CPU/GPU Parity Proof Artifacts

Purpose: run the structured CPU/GPU proof ladder.

### 4A. Non-Banana GPU Follow-Up

This script intentionally requires exact CUDA platform env.

```bash
mkdir -p "${RESULTS_ROOT}/wave4_gpu_parity"

export PYTHONPATH="$PWD:$PWD/src"
export JAX_ENABLE_X64=1
export SIMSOPT_BACKEND_MODE=jax_gpu_parity
export SIMSOPT_JAX_PLATFORM=cuda
export SIMSOPT_EXAMPLE_PARITY_JAX_PLATFORM=cuda
export SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled
export JAX_PLATFORMS=cuda,cpu
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_exclude_nondeterministic_ops=true"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python benchmarks/non_banana_example_cpp_jax_cpu_parity.py \
  --lanes cpu_cpp,jax_gpu \
  --baseline-json "${RESULTS_ROOT}/wave1_cpu_banana/non_banana_cpu_baseline.json" \
  --output-json "${RESULTS_ROOT}/wave4_gpu_parity/non_banana_gpu_followup.json"
```

### 4B. Banana Stage 2 CUDA E2E

```bash
python benchmarks/stage2_e2e_comparison.py \
  --platform cuda \
  --equilibria-dir examples/single_stage_optimization/equilibria \
  --output-json "${RESULTS_ROOT}/wave4_gpu_parity/stage2_cuda_e2e.json"
```

### 4C. Banana Stage 2 CUDA Geometry Repro

The default 20-iteration Stage 2 rung is a smoke budget whose geometry gate is
report-only in the repo ladder contract. Release-grade signoff also runs an
explicit geometry-repro rung.

```bash
python benchmarks/stage2_e2e_comparison.py \
  --platform cuda \
  --maxiter "${STAGE2_GEOMETRY_REPRO_MAXITER}" \
  --geometry-rel-tol "${STAGE2_GEOMETRY_REL_TOL}" \
  --equilibria-dir examples/single_stage_optimization/equilibria \
  --output-json "${RESULTS_ROOT}/wave4_gpu_parity/stage2_cuda_e2e_geometry_repro.json"
```

### 4D. Banana Single-Stage CUDA Init

```bash
python benchmarks/single_stage_init_parity.py \
  --platform cuda \
  --equilibria-dir examples/single_stage_optimization/equilibria \
  --case-artifacts-dir "${RESULTS_ROOT}/wave4_gpu_parity/artifacts/single_stage_cuda" \
  --output-json "${RESULTS_ROOT}/wave4_gpu_parity/single_stage_cuda_init.json"
```

Acceptance:

- [ ] Non-banana follow-up has real `jax_gpu` runtime metadata and passes.
- [ ] Stage 2 CUDA artifact has `passed: true`.
- [ ] Stage 2 CUDA geometry-repro artifact has `passed: true` and gates final
  banana-coil geometry through `geometry_rel_tol`.
- [ ] Single-stage CUDA artifact has `passed: true`.
- [ ] Each CUDA artifact records CUDA backend, devices, x64, `nvidia-smi`,
  driver/runtime, repo SHA, dirty status, and memory.
- [ ] Any artifact with CPU backend is invalid for GPU signoff.

## Wave 5: Production Banana GPU Proof Body

Purpose: run the repo's current production GPU proof contract instead of a
one-off proof command.

Preconditions:

- [ ] Wave 4 passed.
- [ ] `SINGLE_STAGE_WARM_START_RUN_DIR` or `SINGLE_STAGE_JAX_RUNTIME_SEED_SPEC`
  is set.

Command with warm-start run directory:

```bash
mkdir -p "${RESULTS_ROOT}/wave5_production_gpu"

export PYTHONPATH="$PWD:$PWD/src"
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cuda,cpu
export SIMSOPT_JAX_PLATFORM=cuda
export SIMSOPT_BACKEND_MODE=jax_gpu_parity
export SIMSOPT_EXAMPLE_PARITY_JAX_PLATFORM=cuda
export SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_exclude_nondeterministic_ops=true"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

bash benchmarks/hf_jobs/run_production_gpu_proof.sh \
  --results-dir "${RESULTS_ROOT}/wave5_production_gpu/production_gpu_proof" \
  --equilibria-dir examples/single_stage_optimization/equilibria \
  --stage2-bs-path benchmarks/fixtures/single_stage_seed_iota15/biot_savart_opt.json \
  --stage2-platform cuda \
  --stage2-maxiter "${STAGE2_GEOMETRY_REPRO_MAXITER}" \
  --geometry-rel-tol "${STAGE2_GEOMETRY_REL_TOL}" \
  --single-stage-platform cuda \
  --single-stage-warm-start-run-dir "${SINGLE_STAGE_WARM_START_RUN_DIR}"
```

Command with runtime seed spec:

```bash
bash benchmarks/hf_jobs/run_production_gpu_proof.sh \
  --results-dir "${RESULTS_ROOT}/wave5_production_gpu/production_gpu_proof" \
  --equilibria-dir examples/single_stage_optimization/equilibria \
  --stage2-bs-path benchmarks/fixtures/single_stage_seed_iota15/biot_savart_opt.json \
  --stage2-platform cuda \
  --stage2-maxiter "${STAGE2_GEOMETRY_REPRO_MAXITER}" \
  --geometry-rel-tol "${STAGE2_GEOMETRY_REL_TOL}" \
  --single-stage-platform cuda \
  --single-stage-jax-runtime-seed-spec "${SINGLE_STAGE_JAX_RUNTIME_SEED_SPEC}"
```

Acceptance:

- [ ] CUDA PTX and CUBIN canaries pass.
- [ ] `stage2_cold.json` exists and passes.
- [ ] `stage2_warm.json` exists and passes.
- [ ] `stage2_warm_repro.json` exists and passes.
- [ ] `single_stage_cold.json` exists and passes.
- [ ] `single_stage_warm.json` exists and passes.
- [ ] `boozer_well_conditioned_adjoint.json` exists and passes.
- [ ] `reduction_cancellation_stress.json` exists and passes.
- [ ] Proof summary reports no validation failures.

## Wave 6: Performance Characterization

Purpose: measure performance only after correctness passes.

### 6A. Tier 5 Trusted Fixture Performance

Run GPU phase in a GPU job:

```bash
mkdir -p "${RESULTS_ROOT}/wave6_performance"

export PYTHONPATH="$PWD:$PWD/src"
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cuda,cpu
export SIMSOPT_JAX_PLATFORM=cuda
export SIMSOPT_BACKEND_MODE=jax_gpu_parity
export SIMSOPT_EXAMPLE_PARITY_JAX_PLATFORM=cuda
export SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_exclude_nondeterministic_ops=true"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python benchmarks/tier5_performance_characterization.py \
  --platform cuda \
  --phase gpu \
  --benchmark-mode \
  --output-json "${RESULTS_ROOT}/wave6_performance/tier5_gpu.json"
```

Run CPU phase in a CPU job:

```bash
export PYTHONPATH="$PWD:$PWD/src"
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu
export SIMSOPT_JAX_PLATFORM=cpu

python benchmarks/tier5_performance_characterization.py \
  --platform cpu \
  --phase cpu \
  --benchmark-mode \
  --output-json "${RESULTS_ROOT}/wave6_performance/tier5_cpu.json"
```

Aggregate:

```bash
python benchmarks/tier5_performance_characterization.py \
  --phase aggregate \
  --gpu-input-json "${RESULTS_ROOT}/wave6_performance/tier5_gpu.json" \
  --cpu-input-json "${RESULTS_ROOT}/wave6_performance/tier5_cpu.json" \
  --output-json "${RESULTS_ROOT}/wave6_performance/tier5_aggregate.json"
```

### 6B. Boozer run_code CPU/GPU Benchmarks

Run CPU:

```bash
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
export SIMSOPT_BENCHMARK_JAX_VERSION=0.9.2

python benchmarks/cpu_run_code_benchmark.py \
  --backend ondevice \
  --repeats 3 \
  | tee "${RESULTS_ROOT}/wave6_performance/cpu_run_code_benchmark.log"
```

Run GPU:

```bash
export JAX_PLATFORMS=cuda,cpu
export JAX_ENABLE_X64=1
export SIMSOPT_JAX_PLATFORM=cuda
export SIMSOPT_BACKEND_MODE=jax_gpu_parity
export SIMSOPT_BENCHMARK_JAX_VERSION=0.9.2
export SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_exclude_nondeterministic_ops=true"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python benchmarks/gpu_run_code_benchmark.py \
  --backend ondevice \
  --repeats 3 \
  | tee "${RESULTS_ROOT}/wave6_performance/gpu_run_code_benchmark.log"
```

Start with the smaller benchmark configs if queue time is tight:

```bash
python benchmarks/gpu_run_code_benchmark.py \
  --config "Small (4 coils, 15x15)" \
  --config "Medium (6 coils, 15x15)" \
  --backend ondevice \
  --repeats 3
```

Then run the full config matrix if the small/medium pass.

Performance acceptance:

- [ ] Artifacts separate cold compile time from warm steady-state time where
  the benchmark exposes both.
- [ ] CPU and GPU timings use the same repo SHA and comparable fixture config.
- [ ] GPU timing artifact records CUDA backend and device provenance.
- [ ] Peak GPU memory is recorded for proof scripts that sample it.
- [ ] A slowdown is not automatically a correctness failure, but it blocks a
  performance win claim until explained.

## Wave 7: Reporting And Signoff

Create:

- [ ] `REPORT.md` under the run artifact directory.
- [ ] Artifact index with path, command, SHA, Slurm job id, backend, pass/fail.
- [ ] Failure table with owner, blocker/non-blocker classification, and rerun
  command.
- [ ] CPU/GPU parity table:

  | Area | CPU oracle | JAX CPU | JAX CUDA | CPU/GPU agreement | Status |
  | --- | --- | --- | --- | --- | --- |
  | Full pytest |  |  |  |  |  |
  | Non-banana examples |  |  |  |  |  |
  | Banana Stage 2 E2E |  |  |  |  |  |
  | Banana single-stage init |  |  |  |  |  |
  | Production GPU proof |  |  |  |  |  |
  | Tier 5 performance |  |  |  |  |  |
  | Boozer run_code performance |  |  |  |  |  |

- [ ] Final verdict:
  - [ ] release-grade CPU correctness passed
  - [ ] release-grade GPU correctness passed
  - [ ] CPU/GPU parity passed
  - [ ] performance claim accepted
  - [ ] performance data collected but no win claimed
  - [ ] blocked, with exact blocking artifact

## Slurm Execution Policy

- [ ] CPU setup and environment build happen on login nodes.
- [ ] CPU full tests run on CPU compute nodes, not login nodes.
- [ ] Every Slurm job script uses `set -euo pipefail` before any command that
  pipes test/proof output through `tee`.
- [ ] GPU preflight/proofs run under `shared` QOS with `--gpus-per-task=1`.
- [ ] GPU jobs use `JAX_PLATFORMS=cuda,cpu`; CUDA must stay first and must be
  the recorded default backend.
- [ ] GPU `srun` commands use `--cpu-bind=cores` instead of relying on
  `SLURM_CPU_BIND`.
- [ ] Use `interactive` only for manual diagnosis.
- [ ] Use `debug` only for tiny canaries.
- [ ] Record all Slurm job ids in the final report.

Recommended 1-GPU header:

```bash
#SBATCH -A m4680_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
```

Use longer time limits for Wave 5 and Wave 6 if the allocation policy allows it.

## Required Artifact Metadata

Every structured proof artifact must include or be accompanied by:

- [ ] repo SHA
- [ ] git dirty status
- [ ] command argv
- [ ] Slurm job id
- [ ] host name
- [ ] `nvidia-smi` output for GPU jobs
- [ ] NVIDIA driver version
- [ ] CUDA runtime visible to JAX
- [ ] `jax` and `jaxlib` versions
- [ ] JAX default backend
- [ ] JAX devices
- [ ] x64 enabled
- [ ] CUDA visibility env
- [ ] CUDA library mode
- [ ] XLA flags
- [ ] Stage 2 geometry policy and `proof_parity` block where the runner emits
  them
- [ ] peak RSS
- [ ] peak GPU memory where available
- [ ] pass/fail and failure list

## Blocker Rules

- [ ] Any CPU full-suite failure is a blocker unless explicitly classified and
  justified.
- [ ] Any CPU banana E2E failure blocks GPU interpretation.
- [ ] Any GPU proof artifact with CPU backend is invalid.
- [ ] Any missing CUDA provenance blocks GPU signoff.
- [ ] Any tolerance relaxation requires a separate review and cannot be folded
  into this run silently.
- [ ] Performance results are advisory until all correctness waves pass.

## Official Docs Checked

- JAX installation: `https://docs.jax.dev/en/latest/installation.html`
- JAX configuration options: `https://docs.jax.dev/en/latest/config_options.html`
- JAX default dtypes and x64: `https://docs.jax.dev/en/latest/default_dtypes.html`
- JAX GPU memory allocation:
  `https://docs.jax.dev/en/latest/gpu_memory_allocation.html`
- NERSC Python on Perlmutter:
  `https://docs.nersc.gov/development/languages/python/using-python-perlmutter/`
- NERSC Perlmutter running jobs:
  `https://docs.nersc.gov/systems/perlmutter/running-jobs/`
- NERSC affinity: `https://docs.nersc.gov/jobs/affinity/`
- NERSC resource usage policy:
  `https://docs.nersc.gov/policies/resource-usage/`

## Related Repo Files

- `docs/perlmutter_gpu_test_plan_2026-05-19.md`
- `docs/jax_parity_manifest.md`
- `docs/banana_jax_full_test_parity_coverage_impl_plan_2026-05-06.md`
- `benchmarks/non_banana_example_cpp_jax_cpu_parity.py`
- `benchmarks/stage2_e2e_comparison.py`
- `benchmarks/single_stage_init_parity.py`
- `benchmarks/single_stage_outer_loop_probe.py`
- `benchmarks/tier5_performance_characterization.py`
- `benchmarks/cpu_run_code_benchmark.py`
- `benchmarks/gpu_run_code_benchmark.py`
- `benchmarks/hf_jobs/run_production_gpu_proof.sh`
- `benchmarks/fixtures/single_stage_seed_iota15/`
- `scripts/run_gpu_parity.sh`
