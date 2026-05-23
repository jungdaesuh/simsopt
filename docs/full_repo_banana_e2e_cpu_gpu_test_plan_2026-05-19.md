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

### Perlmutter Source Setup And Environment Lane

Run the source checkout and modest environment setup on a login node, not
inside a GPU allocation. Do not run heavyweight pytest/proof workloads there.
If dependency solving or editable builds become compute- or memory-intensive,
move that step into an interactive or batch allocation before continuing.

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
JAX_GPU_WHEEL_SPEC="${JAX_GPU_WHEEL_SPEC:-jax[cuda12]==0.10.0}"
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SIMSOPT="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SIMSOPT:-1.9.4.dev0}"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SIMSOPT

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade "${JAX_GPU_WHEEL_SPEC}"
python -m pip install -e ".[JAX_GPU,test,ALGS]" "shapely>=2.1,<3" "numba>=0.64,<0.66"

export SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled

python - <<'PY'
import jax
import jaxlib

if (jax.__version__, jaxlib.__version__) != ("0.10.0", "0.10.0"):
    raise SystemExit(
        f"expected JAX/JAXLIB 0.10.0 for the on-device optimizer lane, "
        f"got {jax.__version__}/{jaxlib.__version__}"
    )
PY

python - <<'PY'
from repo_bootstrap import bootstrap_local_simsopt
bootstrap_local_simsopt("src")
import simsoptpp
print(simsoptpp.__file__)
PY
```

The split install is intentional: the first command fixes the CUDA JAX wheel
lane under test, and the editable repo install repeats the repo `JAX_GPU`
requirement to add the typed optimizer runtime dependencies without changing
that pinned lane. Do not replace it with a monolithic `.[deploy_gpu]` install in
the Slurm proof runner.

Environment lane decision:

- [ ] The conda environment above is the pip-wheel GPU proof runtime and can
  also run the CPU/reference waves. It is not GPU signoff by itself until Wave
  2 records a CUDA/GPU backend from a Slurm GPU job.
- [ ] Preferred Perlmutter GPU lane: run the proof inside a NERSC-supported
  NVIDIA JAX container through Shifter or Podman-HPC, then install the repo
  proof dependencies into that runtime without replacing the container's JAX
  wheels.
- [ ] Proven pip-wheel candidate: `python -m pip install "jax[cuda12]==0.10.0"`
  resolves for Linux `manylinux_2_27_x86_64` / Python 3.11 in a 2026-05-19
  dry-run. Record a fresh dry-run in the proof bundle before launch.
- [ ] Do not reuse a CPU-only `jax` / `jaxlib` environment or a package-rich
  environment whose installed dependencies constrain JAX below `0.10.0` for the
  Wave 2+ GPU preflight or proof waves.
- [ ] Blocked legacy pip-wheel lane: do not launch a proof with any stale
  dependency spec that resolves to an unavailable CUDA plugin wheel for the
  target.

The benchmark scripts default `SIMSOPT_BENCHMARK_JAX_VERSION` to `0.10.0`.
For a container lane, set `SIMSOPT_BENCHMARK_JAX_VERSION` to the JAX version
recorded by Wave 2 only after confirming that the on-device optimizer runtime
accepts that version.
Do not switch environment lanes within a single proof bundle; if the environment
lane changes, rerun all waves from the same source snapshot and label the
artifacts with that lane.

For a future proven pip-wheel lane, use JAX's bundled CUDA userspace libraries.
Do not load a separate `cudatoolkit` module for that lane, and keep
`SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled` in GPU jobs so repo subprocess helpers
do not prepend a local CUDA toolkit or `LD_LIBRARY_PATH` over the wheel stack.
If using the container lane, replace `module load python` / `conda activate`
prologue commands in the GPU job bodies with the selected Shifter or Podman-HPC
runtime invocation and record the image digest in the final report.

Record setup provenance:

- [ ] `git rev-parse HEAD`
- [ ] `git status --short --untracked-files=no`
- [ ] `python --version`
- [ ] `python -m pip freeze`
- [ ] `python -c 'import jax, jaxlib; print(jax.__version__, jaxlib.__version__)'`
- [ ] `python -c 'import importlib.metadata as m; print(m.version("jax-cuda12-plugin"), m.version("jax-cuda12-pjrt"))'`
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
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export SIMSOPT_JAX_GPU_PREALLOCATE=true

mkdir -p "${RESULTS_ROOT}/wave2_gpu_preflight"

srun -n 1 -c 32 --cpu-bind=cores --gpus-per-task=1 bash -lc '
  set -euo pipefail
  nvidia-smi | tee "'"${RESULTS_ROOT}"'/wave2_gpu_preflight/nvidia-smi.txt"
  python - <<PY | tee "'"${RESULTS_ROOT}"'/wave2_gpu_preflight/jax_gpu_preflight.json"
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
    "jax_platforms": "'"${JAX_PLATFORMS}"'",
    "cuda_library_mode": "'"${SIMSOPT_JAX_CUDA_LIBRARY_MODE}"'",
    "x64": bool(jax.config.read("jax_enable_x64")),
    "simsopt": getattr(simsopt, "__version__", None),
    "simsoptpp": simsoptpp.__file__,
}
print(json.dumps(payload, indent=2, sort_keys=True))
assert (payload["jax"], payload["jaxlib"]) == ("0.10.0", "0.10.0")
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
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export SIMSOPT_JAX_GPU_PREALLOCATE=true
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
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export SIMSOPT_JAX_GPU_PREALLOCATE=true

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

### 4E. Banana Single-Stage Outer-Iteration Parity Ladder

Run this ladder after the init-only probe passes. Do not jump straight to
large mode counts or a 20-iteration run: the first useful optimizer-path signal
is `mpol=2, ntor=2, maxiter=3`, followed by `4/4/5`, then `6/6/10`, then
`8/8/20`.

Official-doc timing rule: JAX dispatch is asynchronous, so performance numbers
must come from artifacts that wait for the computation to finish. For this
runner, use the structured JSON written after both subprocess lanes finish plus
`/usr/bin/time -v`; lower-level microbenchmarks must call
`block_until_ready()` before stopping the timer.

Rungs:

| Rung | `mpol` | `ntor` | `nphi` | `ntheta` | outer `maxiter` | Queue | Seed requirement | Purpose |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `m02n02-i00-init` | 2 | 2 | 31 | 16 | 0 | `debug` or `shared` | none | CUDA/runtime/init parity canary. |
| `m02n02-i03-smoke` | 2 | 2 | 31 | 16 | 3 | `debug` or `shared` | none | First real optimizer-path parity check. |
| `m04n04-i05-useful` | 4 | 4 | 63 | 32 | 5 | `shared` | none | Useful small parity signal without queue waste. |
| `m06n06-i10-serious` | 6 | 6 | 127 | 48 | 10 | `shared` | `SINGLE_STAGE_JAX_RUNTIME_SEED_SPEC_M06N06` | Serious optimizer-path parity evidence. |
| `m08n08-i20-release-small` | 8 | 8 | 255 | 64 | 20 | `shared` | `SINGLE_STAGE_JAX_RUNTIME_SEED_SPEC_M08N08` | Small release-grade parity/performance check. |

The `m06n06` and `m08n08` rungs must use a prebuilt runtime seed spec; the
seed's `mpol`, `ntor`, `nphi`, and `ntheta` must match the rung. The runner
intentionally rejects cold high-resolution outer runs without
`--warm-start-run-dir` or `--jax-runtime-seed-spec`.

```bash
mkdir -p "${RESULTS_ROOT}/wave4_gpu_parity/single_stage_ladder"

export PYTHONPATH="$PWD:$PWD/src"
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cuda,cpu
export SIMSOPT_JAX_PLATFORM=cuda
export SIMSOPT_BACKEND_MODE=jax_gpu_parity
export SIMSOPT_EXAMPLE_PARITY_JAX_PLATFORM=cuda
export SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_exclude_nondeterministic_ops=true --xla_gpu_enable_command_buffer="
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export SIMSOPT_JAX_GPU_PREALLOCATE=true

run_single_stage_ladder_rung() {
  local rung="$1"
  local mpol="$2"
  local ntor="$3"
  local nphi="$4"
  local ntheta="$5"
  local maxiter="$6"
  local seed_spec="${7:-}"
  local rung_dir="${RESULTS_ROOT}/wave4_gpu_parity/single_stage_ladder/${rung}"
  mkdir -p "${rung_dir}/cases"

  nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total \
    --format=csv > "${rung_dir}/nvidia_smi_before.csv"

  local seed_args=()
  if [[ -n "${seed_spec}" ]]; then
    seed_args=(--jax-runtime-seed-spec "${seed_spec}")
  fi

  /usr/bin/time -v -o "${rung_dir}/time.txt" \
    python benchmarks/single_stage_init_parity.py \
      --platform cuda \
      --equilibria-dir examples/single_stage_optimization/equilibria \
      --mpol "${mpol}" \
      --ntor "${ntor}" \
      --nphi "${nphi}" \
      --ntheta "${ntheta}" \
      --maxiter "${maxiter}" \
      --case-artifacts-dir "${rung_dir}/cases" \
      --output-json "${rung_dir}/single_stage_cuda.json" \
      "${seed_args[@]}" \
      > "${rung_dir}/stdout.log" \
      2> "${rung_dir}/stderr.log"

  nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total \
    --format=csv > "${rung_dir}/nvidia_smi_after.csv"
}

run_single_stage_ladder_rung m02n02-i00-init 2 2 31 16 0
run_single_stage_ladder_rung m02n02-i03-smoke 2 2 31 16 3
run_single_stage_ladder_rung m04n04-i05-useful 4 4 63 32 5
run_single_stage_ladder_rung \
  m06n06-i10-serious 6 6 127 48 10 \
  "${SINGLE_STAGE_JAX_RUNTIME_SEED_SPEC_M06N06:?set m06n06 runtime seed spec}"
run_single_stage_ladder_rung \
  m08n08-i20-release-small 8 8 255 64 20 \
  "${SINGLE_STAGE_JAX_RUNTIME_SEED_SPEC_M08N08:?set m08n08 runtime seed spec}"
```

Per-rung required records:

- [ ] `${rung}/single_stage_cuda.json` has `passed: true`.
- [ ] `${rung}/single_stage_cuda.json` records `provenance.backend`,
  `provenance.devices`, `provenance.peak_rss_mb`, and
  `provenance.gpu_memory_mb`.
- [ ] `${rung}/single_stage_cuda.json` records `timings.cpu_elapsed_s`,
  `timings.jax_elapsed_s`, and phase timing keys for both lanes.
- [ ] `${rung}/single_stage_cuda.json` records `comparison` parity deltas
  including iota, volume, field error, and surface geometry where available.
- [ ] `${rung}/time.txt` records `/usr/bin/time -v` wall time and maximum
  resident set size for the full rung command.
- [ ] `${rung}/nvidia_smi_before.csv` and `${rung}/nvidia_smi_after.csv`
  record per-GPU memory before and after the rung.
- [ ] Rung summary table reports, for each rung: `mpol`, `ntor`, `maxiter`,
  `nphi`, `ntheta`, Slurm job id, queue, backend, devices, pass/fail, CPU wall
  time, JAX wall time, peak RSS, peak GPU memory, and all parity deltas.

Acceptance:

- [ ] Non-banana follow-up has real `jax_gpu` runtime metadata and passes.
- [ ] Stage 2 CUDA artifact has `passed: true`.
- [ ] Stage 2 CUDA geometry-repro artifact has `passed: true` and gates final
  banana-coil geometry through `geometry_rel_tol`.
- [ ] Single-stage CUDA artifact has `passed: true`.
- [ ] Single-stage outer-iteration parity ladder passes through at least
  `m04n04-i05-useful` before any `m06n06` or larger run is submitted.
- [ ] Single-stage outer-iteration parity ladder records performance and memory
  for every rung, including CPU/JAX elapsed time, peak RSS, peak GPU memory,
  and per-rung parity deltas.
- [ ] Single-stage outer-iteration parity ladder records
  `--xla_gpu_enable_command_buffer=` in `XLA_FLAGS`; this disables CUDA command
  buffers for the multi-rung ladder after the regular-GPU run exhausted VRAM
  while instantiating hundreds of alive CUDA graphs.
- [ ] Each CUDA artifact records CUDA backend, devices, x64, `nvidia-smi`,
  driver/runtime, repo SHA, dirty status, and memory.
- [ ] Any artifact with CPU backend is invalid for GPU signoff.

### 4F. Optimizer Backend E2E Matrix

Purpose: compare the supported L-BFGS control surfaces on the same reduced
single-stage fixture before interpreting larger single-stage outer-loop timing:

- SciPy L-BFGS CPU reference lane
- private on-device L-BFGS target lane
- SciPy-controlled full JAX graph L-BFGS target lane
- Optax L-BFGS target lane
- Optimistix L-BFGS diagnostic lane

This matrix has two different contracts. The private on-device and
SciPy-controlled full-graph lanes are SciPy/SIMSOPT trajectory-parity lanes:
the final single-stage metrics must match the CPU reference after the same
outer-iteration budget. The Optax lane is the public optimizer comparison lane:
its fixed-candidate objective/gradient contract, strict transfer behavior,
performance, and memory are recorded, but its free-running accepted-step
trajectory is not assumed to be identical to SciPy L-BFGS-B unless a
same-candidate replay proves that the remaining split is optimizer control
rather than objective/gradient math. Optimistix L-BFGS remains diagnostic on
JAX `0.10.0` / Optimistix `0.1.0`: the focused GPU probe below records an
upstream full-`jax.transfer_guard("disallow")` failure independent of SIMSOPT,
so it is not part of the strict signoff matrix until that upstream behavior is
clean.

Use a single Slurm job for the matrix. Do not submit one job per backend.
Each backend rung must run `benchmarks/single_stage_init_parity.py` in
`--benchmark-mode`, sample `nvidia-smi`, and write a structured summary.
When the command is launched through `srun`, use Slurm step `MaxRSS` as the
CPU-memory record; `/usr/bin/time -v` around `srun` records launcher RSS.
This runner owns the relevant oracle contract: its reference lane is the
SciPy/C++ CPU path, and its target lane is selected by `--optimizer-backend`.
The target backends are:

```bash
for backend in ondevice scipy-jax-fullgraph optax-lbfgs; do
  rung_dir="${RESULTS_ROOT}/wave4_gpu_parity/optimizer_matrix/${backend}"
  mkdir -p "${rung_dir}"
  trace_args=()
  if [[ "${backend}" == "optax-lbfgs" ]]; then
    trace_args+=(--record-objective-evaluation-trace)
  fi
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
    --format=csv -l 5 > "${rung_dir}/nvidia_smi_monitor.csv" &
  monitor_pid="$!"

  /usr/bin/time -v -o "${rung_dir}/time.txt" \
    python benchmarks/single_stage_init_parity.py \
      --platform cuda \
      --stage2-bs-path benchmarks/fixtures/single_stage_seed_iota15/biot_savart_opt.json \
      --nphi 31 \
      --ntheta 16 \
      --mpol 2 \
      --ntor 2 \
      --maxiter 1 \
      --benchmark-mode \
      --optimizer-backend "${backend}" \
      "${trace_args[@]}" \
      --equilibria-dir examples/single_stage_optimization/equilibria \
      --case-artifacts-dir "${rung_dir}/cases" \
      --output-json "${rung_dir}/single_stage_optimizer.json"

  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
done
```

Acceptance:

- [ ] The SciPy-compatible target rungs, `ondevice` and
  `scipy-jax-fullgraph`, exit zero and have `passed: true`.
- [ ] The SciPy L-BFGS CPU reference lane is recorded in each rung.
- [ ] The target backend's recorded optimizer method matches the requested
  public backend contract.
- [ ] Public optimizer comparison rung `optax-lbfgs` records strict-transfer
  status, same-candidate objective/gradient replay when available, final metric
  deltas, timing, RSS, and GPU memory. A final-metric split is accepted only as
  an optimizer-control split after same-candidate objective/gradient parity is
  proven.
- [ ] Optimistix L-BFGS full-strict status is recorded as a separate diagnostic
  against the current Optimistix/Equinox/JAX versions; it is not a production
  GPU signoff blocker until the independent upstream strict-transfer probe is
  clean.
- [ ] Each rung records fixed-state precision metrics, final metric deltas,
  CPU wall time, target wall time, Slurm step `MaxRSS`, `/usr/bin/time -v`
  launcher/process RSS as applicable, and sampled peak GPU memory.
- [ ] No production GPU signoff depends on a public optimizer backend that
  fails full strict-transfer execution.
- [ ] This matrix is a reduced single-stage E2E optimizer-control comparison.
  It is not a substitute for the single-stage `m04n04-i05-useful` and larger
  rungs.

## Wave 5: Production Banana GPU Proof Body

Purpose: run the repo's current production GPU proof contract instead of a
one-off proof command.

Preconditions:

- [ ] Wave 4 passed.
- [ ] `m04n04-i05-useful` has passed before any larger single-stage run is
  submitted.

Command:

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
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export SIMSOPT_JAX_GPU_PREALLOCATE=true

python benchmarks/stage2_e2e_comparison.py \
  --platform cuda \
  --maxiter "${STAGE2_GEOMETRY_REPRO_MAXITER}" \
  --geometry-rel-tol "${STAGE2_GEOMETRY_REL_TOL}" \
  --equilibria-dir examples/single_stage_optimization/equilibria \
  --output-json "${RESULTS_ROOT}/wave5_production_gpu/stage2_geometry_repro.json"

python benchmarks/single_stage_outer_loop_probe.py \
  --platform cuda \
  --stage2-bs-path benchmarks/fixtures/single_stage_seed_iota15/biot_savart_opt.json \
  --output-json "${RESULTS_ROOT}/wave5_production_gpu/single_stage_outer_loop.json"

python benchmarks/production_boozer_parity_probe.py \
  --platform cuda \
  --output-json "${RESULTS_ROOT}/wave5_production_gpu/boozer_production_grid.json"

python benchmarks/adjoint_fd_validation.py \
  --platform cuda \
  --stage2-bs-path benchmarks/fixtures/single_stage_seed_iota15/biot_savart_opt.json \
  --output-json "${RESULTS_ROOT}/wave5_production_gpu/adjoint_fd_validation.json"
```

Acceptance:

- [ ] CUDA PTX and CUBIN canaries pass.
- [ ] `stage2_geometry_repro.json` exists and has `passed: true`.
- [ ] `single_stage_outer_loop.json` exists and has `passed: true`.
- [ ] `boozer_production_grid.json` exists and has `passed: true`.
- [ ] `adjoint_fd_validation.json` exists and has `passed: true`.
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
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export SIMSOPT_JAX_GPU_PREALLOCATE=true

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
export SIMSOPT_BENCHMARK_JAX_VERSION=0.10.0

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
export SIMSOPT_BENCHMARK_JAX_VERSION=0.10.0
export SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_exclude_nondeterministic_ops=true"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export SIMSOPT_JAX_GPU_PREALLOCATE=true

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

## Execution Snapshot: 2026-05-19 Perlmutter JAX 0.10.0 Dirty-Tree Run

Artifact root:
`/pscratch/sd/j/jungdae/simsopt-jax-results/jax-0.10.0-e2e-memperf-customvjp-20260519T095319Z`

Source/runtime:

- Remote worktree: `/pscratch/sd/j/jungdae/simsopt-jax-worktree`
- Remote base SHA recorded by artifacts: `d03699dd398cc898212b10daefd03d5a4d7f1676`
- Runtime: Python 3.11, `jax==0.10.0`, `jaxlib==0.10.0`
- Evidence class: dirty-tree hardware proof; do not treat as clean-release
  signoff until the intended patch slice is committed and rerun from that SHA.

Correctness and fix-validation results:

| Area | Job / command | Result | Time / memory |
| --- | --- | --- | --- |
| Native `simsoptpp` rebuild after PM print-interval fix | Slurm `53165120` | PASS | 2:10.59 wall by `/usr/bin/time`; 967824 KB MaxRSS by `/usr/bin/time`; Slurm batch MaxRSS 13387572K |
| PM QA reduced fixed-state fixture | Slurm `53165165` | PASS | 0:13.18 wall; 934088 KB MaxRSS |
| GPU runtime smoke and grouped lowering | Slurm `53164465` first two packets | PASS | runtime smoke 3:04.11 / 3317048 KB; grouped lowering 0:38.15 / 1072004 KB |
| GPU M5 public wrapper rerun after transfer-clean CWS pullback fix | Slurm `53164835` | PASS | 1:53.04 wall; 3033300 KB MaxRSS; `3 passed in 99.81s` |
| 4-GPU single-stage init parity after active replicated-placement fixes | Slurm `53170493` | PASS | Slurm elapsed 7:34; `/usr/bin/time` wall 7:27.85; `/usr/bin/time` MaxRSS 6285132 KB; Slurm batch MaxRSS 7905296K; 4x A100 before/after memory 0 MiB |

Performance and memory-pressure results:

| Area | Job | Result | Time / memory | Notes |
| --- | --- | --- | --- | --- |
| Tier 5 CPU performance characterization | `53164679` first phase | PASS | 3:31.33 wall; 4897820 KB MaxRSS | CPU phase artifact completed before the later run-code benchmark failure. |
| CPU run-code benchmark, all configs in one process | `53164679` second phase | FAIL | 1:19:47 wall; 11775832 KB MaxRSS; exit 1 | Failed during Full-HBT repeat 3/3 with JAX/XLA CPU `LLVM compilation error: Cannot allocate memory` and `Failed to materialize symbols`. Treat as accumulated CPU compile-memory pressure in the benchmark process. |
| CPU run-code benchmark, isolated Full-HBT only | `53165744` | PASS | 36:36.59 wall; 6939428 KB MaxRSS; Slurm batch MaxRSS 7453984K | Full-HBT repeat median 403451.6 ms; first call 499.291 s; LS 8204.9 ms; Newton 378095.5 ms. |
| GPU Tier 5 high-memory retry | `53164760` | TIMEOUT | Slurm timeout at 2:00:16; step MaxRSS 76182456K under `--mem-per-gpu=80G`; sampled GPU memory peak 2347 MiB | Passed Stage 2 CPU-vs-JAX value/gradient parity (`J` rel_err `3.95e-16`, grad L2 rel_err `1.56e-14`) but did not complete performance characterization before walltime. |
| GPU Tier 5 original 57 GB run | `53164210` | FAIL / OOM | 50:05.20 wall; 58045712 KB MaxRSS; exit 1 | Establishes that the pressure is host memory, not GPU VRAM. |
| Multi-GPU sharding proof, pre-sharded steady state | `53168132` regular; corroborated by `53168131` debug | PASS | `53168132` batch MaxRSS 2860224K; `53168131` batch MaxRSS 2962280K; per-probe GPU memory recorded in `docs/jax_multi_gpu_proof_2026-05-19.md` | `integral_BdotN_surface_sharded`: 2.03x at 2 GPUs, 3.87x at 4 GPUs with `NamedSharding` and all-reduce. Seed-batch scoring: 1.93x at 2 GPUs, 3.78x at 4 GPUs. |
| Single-stage init parity with active point sharding, post-review exact bytes | `53170493` debug | PASS | Slurm elapsed 7:34; `/usr/bin/time` wall 7:27.85; MaxRSS 6285132 KB; Slurm batch MaxRSS 7905296K; early stdout peak RSS 1077.7 MB; early GPU memory 435.0 MB; before/after GPU memory 0 MiB on all four A100s | `single_stage_cuda_init.json` has `passed: true`; JAX/JAXLIB 0.10.0; backend `gpu`; devices `cuda:0..3`; `SIMSOPT_JAX_SHARDING=points`; CPU vs JAX field-error rel diff `2.51e-16`; iota and volume diffs `0.00e+00`. |

Current verdict for this run:

- [x] Local transfer-clean and scatter/CWS regression packet passed.
- [x] PM native SIGFPE fixture passed after rebuilding the exact C++ bytes.
- [x] GPU M5 public-wrapper transfer-clean test packet passed on real CUDA.
- [x] CPU performance data collected for Tier 5 and run-code Full-HBT.
- [x] GPU memory-pressure data collected.
- [x] Multi-GPU sharding performance and memory data collected.
- [x] Multi-GPU sharding proof passed for the pre-sharded steady-state
  contract.
- [x] 4-GPU single-stage init parity passed after active replicated-placement
  fixes for the private optimizer and Boozer penalty geometry.
- [ ] GPU performance characterization completed.
- [ ] Release-grade GPU performance claim accepted.

Blocking interpretation:

- The GPU high-memory run survived past the prior 57 GB OOM point but timed out
  at the two-hour Slurm limit while still inside Tier 5 GPU characterization.
  This blocks any GPU performance win claim.
- The CPU all-config benchmark failure is reproducible evidence that running
  every large config in one JAX process accumulates enough CPU/XLA compile
  pressure to fail. The isolated Full-HBT rerun completed, so the Full-HBT
  timing itself is available and the failure is a benchmark-process memory
  pressure issue rather than a solver correctness failure.
- The N30 multi-GPU sharding proof passed only after the timed inputs were
  pre-placed on the JAX mesh. Earlier non-pre-sharded debug probes showed active
  sharding but failed the speedup gate because repeated input placement
  dominated the surface-integral timing. This matches the official JAX
  `device_put` + `NamedSharding` model and is recorded in
  `docs/jax_multi_gpu_proof_2026-05-19.md`.
- The first 4-GPU single-stage init retry reached `SINGLE-STAGE INIT PARITY
  PASSED` but exceeded the 10-minute debug walltime by 33.41 s (`53169133`).
  The post-review rerun with transfer guard `allow` completed cleanly as
  `53170493`, proving the exact current active replicated-placement bytes
  within the 20-minute debug allocation.

## Current Execution Update: Clean Committed SHA `1345ea9e0`

This update records the post-cleanup committed SHA currently under E2E test:

- Local branch: `gpu-purity-stage2-20260405`
- Current committed SHA:
  `1345ea9e081a5ebaa0727cf4ae51ebce6bead3a1`
- Remote clean source checkout:
  `/pscratch/sd/j/jungdae/simsopt-jax-clean-1345ea9e0-20260521T143208Z-src`
- Remote runtime:
  `/pscratch/sd/j/jungdae/simsopt-jax-runtimes/jax-0.10.0`
- JAX/JAXLIB: `0.10.0` / `0.10.0`
- Local working tree still has unrelated dirt:
  `conda.recipe/meta.yaml` and `.conda/`.

Focused local regression packet for this SHA:

| Command | Result |
| --- | --- |
| `python -m pytest tests/geo/test_single_stage_example.py -k "runtime_spec_biotsavart_projects_cotangents_to_owner_dofs or runtime_spec_biotsavart_full_artifact_curves_follow_updated_dofs or host_curve_max_curvature_allows_strict_transfer_guard" -q` | PASS: `3 passed, 346 deselected` |
| `python -m pytest tests/field/test_biotsavart_jax.py -q` | PASS: `52 passed` |
| `python -m pytest tests/integration/test_stage2_jax.py -k b_pullback_native_projects_to_free_dof_gradient -q` | PASS: `1 passed, 183 deselected` |
| `python -m pytest tests/test_jax_import_smoke.py -k "transfer_guard_disallow_enforces_single_stage_target_runtime_boundaries or import_biotsavart_jax" -q` | PASS: `2 passed, 119 deselected` |

Single-stage CUDA `m02n02-i03-smoke` debug ladder on Perlmutter:

| Job | SHA | Result | Evidence |
| --- | --- | --- | --- |
| `53241854` / `ss-m02-i03-33cb87225-r2` | `33cb87225` | FAIL | Strict transfer guard caught scalar Python indexing in `SpecBackedBiotSavartJAX.coil_cotangents_to_dofs_gradient`; step MaxRSS `22085192K`. |
| `53242990` / `ss-m02-i03-1d430a699` | `1d430a699` | FAIL | Strict transfer guard caught host-staged zero construction in `SpecBackedCurve.dgammadash_by_dcoeff_vjp`; step MaxRSS `22108988K`. |
| `53243425` / `ss-m02-i03-7d23c241b` | `7d23c241b` | FAIL | Strict transfer guard caught JAX `integer_pow` JVP in the curvature VJP path; step MaxRSS `22177380K`. |
| `53244319` / `ss-m02-i03-1345ea9e0` | `1345ea9e0` | TIMEOUT | No strict-transfer traceback before the 30-minute debug walltime. The target lane reached three optimizer iterations and final Boozer solve. Step MaxRSS `22167980K`; sampled peak GPU memory `3153 MiB`; peak GPU utilization `10%`; backend `gpu`; devices `cuda:0..3`; runtime JSON records `simsopt==0.1.dev1885+g1345ea9e0`. |

Open jobs for this SHA:

| Job | Purpose | QOS / partition | State at submission audit |
| --- | --- | --- | --- |
| `53245257` / `ss-m02-i03-1345ea9e0-r1h` | Superseded regular-QOS repeat of `m02n02-i03-smoke`. | `gpu_regular` / `gpu_ss11` | CANCELED before proof-body evidence; regular reserves an exclusive GPU node and was replaced by shared-QOS job `53245542`. |
| `53245542` / `ss-m02-i03-1345ea9e0-shared` | Repeat `m02n02-i03-smoke` with one-hour walltime to distinguish finalization runtime from a new transfer bug. | `gpu_shared` / `shared_gpu_ss11`; live scheduler requires `32` CPU cores for `1` GPU on this partition. | `RUNNING` since `2026-05-21 08:54 PDT` on `nid003192`; reached three optimizer iterations and final Boozer solve; no `single_stage_cuda.json` yet. |
| `53245429` / `cpu-w0w1-1345ea9e0` | Wave 0 import smoke plus Wave 1 focused CPU banana/proof artifacts, with per-step `/usr/bin/time -v` MaxRSS. | CPU `shared` / `shared_milan_ss11` | `PENDING (Resources)` |
| `53246693` / `opt-matrix-1345ea9e0` | Superseded Stage 2 optimizer matrix attempt. | `gpu_shared` / `shared_gpu_ss11` | CANCELED after reviewer audit showed this did not exercise the requested SciPy CPU reference lane; the corrected plan uses `benchmarks/single_stage_init_parity.py` instead. |

The current clean-SHA state does not satisfy the full plan. It only proves that
the latest strict-transfer issues found by the debug ladder have been fixed
locally and that the latest GPU debug canary progressed past all previously
observed transfer failures before timing out.

## Current Execution Update: Optimizer Strict-Transfer Docfix at `710903cfd0`

This update records the optimizer-layer transfer-guard check after official-doc
cross-checking and the Perlmutter debug proof:

- Local branch: `gpu-purity-stage2-20260405`
- Base committed SHA:
  `710903cfd0a2a7ac73d857bc255c675646cf52ce`
- Remote source checkout patched with the five current-session files:
  `/pscratch/sd/j/jungdae/simsopt-jax-clean-710903cfd0-docfix-20260521T202439Z-src`
- Remote runtime:
  `/pscratch/sd/j/jungdae/simsopt-jax-runtimes/jax-0.10.0`
- JAX/JAXLIB/CUDA plugin/PJRT: `0.10.0` / `0.10.0` / `0.10.0` / `0.10.0`

Official-doc decision boundary:

- JAX transfer guards distinguish explicit `jax.device_put` / `jax.device_get`
  from implicit transfers; `disallow` rejects implicit transfers while allowing
  explicit transfers.
- JAX closed-over-constant, `make_jaxpr`, and `jax.extend.core` documentation
  support extracting a `ClosedJaxpr` and passing its constants as explicit
  dynamic arguments. The documented `closure_convert` API was checked, but
  rejected for this path after Perlmutter job `53266068` showed it still lowered
  the GPU `target` array as an implicit device-to-host constant inside the Optax
  line-search JIT.
- CUDA best-practices documentation treats host-device transfers as costly and
  recommends keeping intermediate data on device.
- Optimistix `minimise` officially passes `args` through to `fn(y, args)`, so
  the wrapper now passes JAXPR closure constants through `args`.
- Equinox enumeration docs expose `.where(pred, a, b)` over scalar boolean
  arrays; the live Perlmutter repro showed current Optimistix/Equinox LBFGS
  result-enum handling is not full `jax.transfer_guard("disallow")` clean on
  GPU independent of SIMSOPT. Therefore only Optax LBFGS is used for the
  full-strict GPU closure-constant proof in this slice; Optimistix LBFGS keeps
  host-to-device transfer hygiene coverage but is not claimed as a full-strict
  GPU lane until upstream behavior changes.
- Post-review correction: the JAXPR closure conversion is limited to Optax
  LBFGS, because only that branch needs a line-search `value_fn`. Optax Adam
  keeps the direct eager value/grad contract and now has a regression test for a
  host-materializing value/grad callback.

Focused local regression packet:

| Command | Result |
| --- | --- |
| `python -m pytest -q tests/solve/jax/test_value_grad_contract.py -k 'optax or optimistix'` | PASS: `15 passed, 1 skipped, 2 deselected` |
| `python -m pytest -q tests/solve/jax/test_driver_dispatch.py tests/solve/jax/test_import_boundaries.py` | PASS: `11 passed` |
| `python -m pytest -q tests/geo/test_single_stage_example.py -k 'resolve_target_lane_boozer_init_base_overrides'` | PASS: `6 passed, 344 deselected` |
| `python -m py_compile src/simsopt/solve/jax/_dispatch.py tests/solve/jax/test_value_grad_contract.py examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py tests/geo/test_single_stage_example.py` | PASS |
| `git diff --check -- docs/full_repo_banana_e2e_cpu_gpu_test_plan_2026-05-19.md src/simsopt/solve/jax/_dispatch.py tests/solve/jax/test_value_grad_contract.py examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py tests/geo/test_single_stage_example.py` | PASS |

Perlmutter debug proof:

| Job | Result | Evidence |
| --- | --- | --- |
| `53265848` / `strict-710903` | PASS | JAXPR-const implementation passed `tests/solve/jax/test_value_grad_contract.py -k gpu_closure_constants_run_under_strict_transfer_guard`: `1 passed, 16 deselected in 6.66s`; backend `gpu`; device `cuda:0`; x64 `True`; elapsed `29s`; pytest wall time `11.78s`; process MaxRSS `999476 KB`; sampled peak GPU memory `435 MiB` on A100-40GB; exit `0`. |
| `53266068` / `strict-710903` | EXPECTED FAIL | Rejected the attempted `jax.closure_convert` implementation: Optax GPU strict-transfer test failed with `Disallowed device-to-host transfer: shape=(2), dtype=F64, device=cuda:0` while lowering the closed-over `target` array. Pytest wall time `12.24s`; process MaxRSS `990660 KB`; exit `1`. |
| `53266208` / `strict-710903` | PASS | Current synced JAXPR-const implementation passed `tests/solve/jax/test_value_grad_contract.py -k gpu_closure_constants_run_under_strict_transfer_guard`: `1 passed, 16 deselected in 3.14s`; backend `gpu`; device `cuda:0`; x64 `True`; elapsed `21s`; pytest wall time `5.40s`; process MaxRSS `1002364 KB`; sampled peak GPU memory `435 MiB` on A100-40GB; exit `0`. |
| `53266572` / `strict-710903` | PASS | Public `jax.extend.core` JAXPR execution implementation passed `tests/solve/jax/test_value_grad_contract.py -k gpu_closure_constants_run_under_strict_transfer_guard`: `1 passed, 16 deselected in 3.18s`; backend `gpu`; device `cuda:0`; x64 `True`; elapsed `19s`; pytest wall time `5.32s`; process MaxRSS `1002304 KB`; sampled peak GPU memory `435 MiB` on A100-40GB; exit `0`. |

## Current Execution Update: Committed Strict-Transfer Fix at `ebca962d5`

This update records the committed version of the optimizer strict-transfer fix:

- Committed SHA:
  `ebca962d5d8bb757d1187aa565469c62672acb2f`
- Commit: `fix: preserve optimizer strict-transfer constants`
- Remote source archive:
  `/pscratch/sd/j/jungdae/simsopt-jax-clean-ebca962d5-strict-20260521T215326Z-src`
- Remote runtime:
  `/pscratch/sd/j/jungdae/simsopt-jax-runtimes/jax-0.10.0`
- Local working tree still has unrelated dirt:
  `conda.recipe/meta.yaml` and `.conda/`.

Current-SHA Perlmutter debug proof:

| Job | Result | Evidence |
| --- | --- | --- |
| `53267612` / `strict-ebca962d5` | PASS | Committed Optax LBFGS closure-constant implementation passed `tests/solve/jax/test_value_grad_contract.py -k gpu_closure_constants_run_under_strict_transfer_guard`: `1 passed, 17 deselected in 3.35s`; JAX/JAXLIB/CUDA plugin/PJRT `0.10.0`; backend `gpu`; device `cuda:0`; x64 `True`; Slurm elapsed `20s`; pytest wall time `5.64s`; process MaxRSS `995616 KB`; exit `0`. |

This is a current-SHA GPU strict-transfer proof for the optimizer-layer
constant residency fix. It is not a replacement for the Wave 4 optimizer
backend matrix or the larger single-stage GPU proof rungs.

Current-SHA Wave 4 optimizer backend matrix:

| Job | State | Evidence |
| --- | --- | --- |
| `53268207` / `optbench-ebca962d5` | FAILED | Single shared-GPU job ran against source archive `/pscratch/sd/j/jungdae/simsopt-jax-clean-ebca962d5-strict-20260521T215326Z-src` and runtime `/pscratch/sd/j/jungdae/simsopt-jax-runtimes/jax-0.10.0`. Environment recorded JAX/JAXLIB/CUDA plugin/PJRT `0.10.0`, Optax `0.2.8`, Optimistix `0.1.0`, Equinox `0.13.8`, backend `gpu`, device `cuda:0`, x64 `True`, NVIDIA driver `580.105.08`, CUDA `13.0`. The Slurm job failed overall because the public optimizer comparison rungs did not satisfy their current gates, but both SciPy-compatible target lanes passed production trajectory parity. |

Per-rung Wave 4 matrix results:

| Backend | Exit | Pass | Final metric parity | Slurm step RSS | GPU peak | Timing |
| --- | ---: | --- | --- | ---: | ---: | --- |
| `ondevice` | `0` | PASS | `iota=2.54e-17`, `volume=1.11e-15`, `field=5.06e-15`, `curvature=1.51e-14` | `15082756K` | `1693 MiB / 40960 MiB` | CPU `13.75s`; JAX `1413.17s`; optimizer `1239.85s`; Boozer solve `39.22s`; Slurm step `24:14` |
| `scipy-jax-fullgraph` | `0` | PASS | `iota=4.66e-17`, `volume=5.56e-16`, `field=7.88e-16`, `curvature=1.47e-14` | `4164872K` | `1483 MiB / 40960 MiB` | CPU `13.66s`; JAX `622.23s`; optimizer `383.20s`; Boozer solve `24.97s`; Slurm step `10:58` |
| `optimistix-lbfgs` | `1` | FAIL | no final JSON; failed before artifact write | `4083744K` | `1471 MiB / 40960 MiB` | Slurm step `7:33` |
| `optax-lbfgs` | `1` | FAIL | `iota=3.55e-05`, `volume=1.98e-04`, `field=1.56e-01`, `curvature=3.47e-01` | `5376264K` | `1497 MiB / 40960 MiB` | CPU `13.55s`; JAX `1023.69s`; optimizer `860.96s`; Boozer solve `27.94s`; Slurm step `17:39` |

The `optax-lbfgs` failed final-state parity after one outer iteration:
CPU/SIMSOPT L-BFGS-B produced `FINAL_IOTA=0.001445580130964724`,
`FINAL_VOLUME=0.09989701364124072`,
`FIELD_ERROR=0.0019267414500637196`, and
`MAX_CURVATURE=15.67283188282475`; Optax produced
`FINAL_IOTA=0.0014811217638137767`,
`FINAL_VOLUME=0.0998772011774527`,
`FIELD_ERROR=0.0022273458279302534`, and
`MAX_CURVATURE=21.114297209987548`. This is not, by itself, evidence of a
physics or objective-gradient bug; it is a free-running optimizer-control split
until a same-candidate replay proves or falsifies the fixed-candidate math.

Focused optimizer-library strict-transfer probes:

| Job | Result | Evidence |
| --- | --- | --- |
| `53270528` / `optlib2-ebca962d5` | COMPLETED | Corrected current-source probe removed the stale editable-install redirect and imported `/pscratch/sd/j/jungdae/simsopt-jax-clean-ebca962d5-strict-20260521T215326Z-src/src/simsopt/__init__.py`. Backend `gpu`, device `cuda:0`, JAX `0.10.0`, Optax `0.2.8`, Optimistix `0.1.0`, Equinox `0.13.8`. Repo Optax supplied-gradient LBFGS under full `jax.transfer_guard("disallow")` completed one step without transfer failure (`ok=True`, `nit=1`, `fun=1.527864045000421`). Repo Optimistix supplied-gradient LBFGS failed full strict transfer with `Disallowed device-to-host transfer: shape=(), dtype=PRED, device=cuda:0`. A direct Optimistix AD-valued minimize also failed full strict transfer with `Disallowed host-to-device transfer: aval=ShapedArray(bool[])`. |

Focused Optax same-candidate replay:

| Job | State | Evidence |
| --- | --- | --- |
| `53270745` / `optax-replay-ebca962d5` | HARNESS FAIL | First replay harness ran against the clean source archive but omitted `SIMSOPT_REPO_SHA` and `SIMSOPT_GIT_STATUS_SHORT`. Because the archive is not a Git checkout, `benchmarks/validation_ladder_common.py` failed at provenance collection before physics execution: `git -C ... rev-parse HEAD` returned `128`. Result root: `/pscratch/sd/j/jungdae/simsopt-jax-results/ebca962d5-optax-same-candidate-replay-20260521T234331Z`; Slurm step `00:02`; no benchmark JSON. |
| `53270856` / `optax-replay-ebca962d5` | INSTRUMENTATION FAIL | Corrected provenance and reached the target Optax replay path, but failed while writing `outer_optimizer_progress.json`: the objective-evaluation trace payload contained a JAX `ArrayImpl`, and `json.dump` rejected it as not JSON serializable. Result root: `/pscratch/sd/j/jungdae/simsopt-jax-results/ebca962d5-optax-same-candidate-replay-20260521T234726Z-provenance`; Slurm step `7:12`; Python MaxRSS `4293920K`; sampled GPU peak `1471 MiB / 40960 MiB`; no benchmark JSON. Local fix: `examples/single_stage_optimization/hardware_constraints.py` now materializes JAX arrays through the explicit host boundary in `sanitize_diagnostic_payload`; focused regression `python -m pytest -q tests/geo/test_single_stage_example.py -k 'sanitize_diagnostic_payload'` passes (`2 passed, 349 deselected`). |
| `53271080` / `optax-replay-jsonfix` | HARNESS FAIL | JSON serialization fix worked past the previous callback failure, then the benchmark tried to force exact `--replay-objective-evaluation-trace` through the native Optax target-lane path. The single-stage script correctly rejected that unsupported mode with `--replay-objective-evaluation-trace requires the host-dispatched adapter objective path.` Result root: `/pscratch/sd/j/jungdae/simsopt-jax-results/ebca962d5-optax-same-candidate-replay-jsonfix-20260522T000154Z`; Slurm step `7:08`; Python MaxRSS `4552576K`; sampled GPU peak `1471 MiB / 40960 MiB`; no benchmark JSON. Local fix: `benchmarks/single_stage_init_parity.py` now runs exact replay only for `scipy-jax-fullgraph`; public L-BFGS target lanes use the existing trace-vs-trace comparison path. Focused regression `python -m pytest -q tests/test_benchmark_helpers.py -k 'case_pair_replays_reference_trace_before_jax_fullgraph or case_pair_skips_exact_replay_for_public_lbfgs or same_candidate'` passes (`16 passed, 267 deselected`). |
| `53272435` / `optax-trace-harnessfix` | OLD-GATE FAIL / TRACE PASS | Replay/trace rerun against patched source copy `/pscratch/sd/j/jungdae/simsopt-jax-clean-ebca962d5-optax-trace-harnessfix-20260522T001400Z-src` completed with Slurm elapsed `22:10`; Python step failed in `21:54` with step MaxRSS `5592436K` and sampled GPU peak `1495 MiB / 81920 MiB`. The benchmark JSON had `passed: false` only because the old final-metric gate still hard-failed `iota=3.55e-05`, `volume=1.98e-04`, and `field=1.56e-01`. The trace evidence passed: `same_candidate_replay.status=pass`, `failures=[]`, `max_objective_abs_diff=0.0`, `max_optimizer_gradient_abs_diff=9.59e-14`, and `comparison.optimizer_path_split_kind=optimizer_acceptance_split_after_same_candidate_parity`. Timings: CPU `15.81s`, JAX `1272.40s`, JAX optimizer `1103.99s`, optimizer main `826.10s`, Boozer total `44.32s`. Result root: `/pscratch/sd/j/jungdae/simsopt-jax-results/ebca962d5-optax-same-candidate-trace-harnessfix-20260522T004729Z`. This run predates the final public-optimizer acceptance-contract patch, so it is evidence for trace execution but not final evidence for the updated pass/fail semantics. |
| `53273813` / `optax-public-accept` | LAUNCHER FAIL | First debug rerun failed in `00:06` before environment capture or physics execution. Root cause was sbatch-generation quoting: `SIMSOPT_GIT_STATUS_SHORT=$'...'` was expanded while generating the script, so `set -u` saw an unbound `$M` on line 25. This is not a JAX/SIMSOPT/physics failure. Result root: `/pscratch/sd/j/jungdae/simsopt-jax-results/ebca962d5-optax-public-acceptance-20260522T012600Z`. |
| `53273985` / `optax-public-accept` | PASS | Corrected debug rerun with literal heredoc quoting for `SIMSOPT_GIT_STATUS_SHORT`. Current local patched files were synced to `/pscratch/sd/j/jungdae/simsopt-jax-clean-ebca962d5-optax-public-acceptance-20260522T011500Z-src`; SHA-256 hashes match local for `benchmarks/single_stage_init_parity.py`, `examples/single_stage_optimization/hardware_constraints.py`, `tests/test_benchmark_helpers.py`, `tests/geo/test_single_stage_example.py`, and this plan doc. Result root: `/pscratch/sd/j/jungdae/simsopt-jax-results/ebca962d5-optax-public-acceptance-20260522T013000Z`. Slurm elapsed `21:08`; Python step elapsed `20:53`, MaxRSS `5671632K`; sampled GPU peak `1497 MiB / 40960 MiB`; `/usr/bin/time` launcher wall `20:52.85`. Benchmark summary: `run_exit=0`, `passed=true`, `failures=[]`, `same_candidate_replay.status=pass`, `same_candidate_replay.failures=[]`, `max_objective_abs_diff=0.0`, `max_optimizer_gradient_abs_diff=9.59e-14`, `optimizer_path_objective_evaluations.status=split`, `comparison.final_metric_split_accepted=true`, and `comparison.final_metric_parity_required=false`. Timings: CPU `15.15s`, JAX `1213.61s`, JAX optimizer `1058.05s`, optimizer main `781.31s`, Boozer total `37.65s`. |

Local review and validation for the current public-optimizer acceptance patch:

| Check | Result |
| --- | --- |
| `python -m pytest -q tests/test_benchmark_helpers.py -k 'single_stage_init_parity_can_defer_public_optimizer_final_metrics or single_stage_init_public_optimizer_final_metric_drift_needs_path_split or single_stage_init_public_optimizer_trace_required_for_outer_loop or case_pair_replays_reference_trace_before_jax_fullgraph or case_pair_skips_exact_replay_for_public_lbfgs or same_candidate'` | PASS: `19 passed, 267 deselected in 2.89s` |
| `python -m pytest -q tests/geo/test_single_stage_example.py -k 'sanitize_diagnostic_payload'` | PASS: `2 passed, 349 deselected in 2.99s` |
| `python -m py_compile benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py examples/single_stage_optimization/hardware_constraints.py tests/geo/test_single_stage_example.py examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py` | PASS |
| `git diff --check -- benchmarks/single_stage_init_parity.py docs/full_repo_banana_e2e_cpu_gpu_test_plan_2026-05-19.md examples/single_stage_optimization/hardware_constraints.py tests/geo/test_single_stage_example.py tests/test_benchmark_helpers.py` | PASS |
| Reviewer subagent `019e4d39-640e-7db3-951e-77a3678e8167` | PASS: prior findings fixed; public Optax/Optimistix rungs require trace evidence and only accept final metric drift after same-candidate objective/gradient parity plus optimizer-path split; Wave 4 command passes `--record-objective-evaluation-trace` for public optimizer rungs. |

Official-docs check for the optimizer backend interpretation:

- Context7 Optax docs for `/google-deepmind/optax` describe `optax.lbfgs`
  as a quasi-Newton optimizer whose default line search is
  `optax.scale_by_zoom_linesearch`; that line search performs extra objective
  calls and has independent line-search parameters. Therefore Optax L-BFGS is
  not an exact SciPy L-BFGS-B trajectory oracle.
- Context7 Optimistix docs for `/patrick-kidger/optimistix` describe
  `optimistix.minimise` as automatically JIT-compiled through
  `eqx.filter_jit`, matching the observed strict-transfer failure path through
  Optimistix/Equinox/JAX lowering.
- Context7 JAX docs for `/google/jax` confirm transfer guards disallow
  implicit host/device transfers and that closed-over JAX arrays participate in
  tracing/lowering as constants, matching the closure-constant residency work
  already proven by job `53267612`.

## Current Execution Update: Clean Committed SHA `7750e34d8`

This update records the clean current-SHA rerun after committing the public
optimizer parity contract.

- Committed SHA:
  `7750e34d824a72858e1c4d52efcc379d40b1f9de`
- Commit: `fix: align public optimizer parity contract`
- Remote source archive:
  `/pscratch/sd/j/jungdae/simsopt-jax-clean-7750e34d82-20260522T020608Z-src`
- Remote runtime:
  `/pscratch/sd/j/jungdae/simsopt-jax-runtimes/jax-0.10.0`
- Local working tree still has unrelated dirt:
  `conda.recipe/meta.yaml` and `.conda/`.

Runtime install/provenance correction:

| Job | Result | Evidence |
| --- | --- | --- |
| `53275710` / `inst2-7750` | PASS | Installed missing build-system packages `scikit-build-core` and `setuptools_scm`, then rebuilt the runtime editable install as `simsopt-0.1.dev1886+g7750e34d82` from the clean source archive. Import verification records `simsopt.__file__ == /pscratch/sd/j/jungdae/simsopt-jax-clean-7750e34d82-20260522T020608Z-src/src/simsopt/__init__.py`, `jax==0.10.0`, `jaxlib==0.10.0`, backend `cpu`, and `simsoptpp` from the runtime extension path. Slurm elapsed `3:18`; batch MaxRSS `10179604K`. |

Current-SHA Wave 2 GPU preflight:

| Job | Result | Evidence |
| --- | --- | --- |
| `53276043` / `gpu-pre-7750` | PASS | JAX/JAXLIB/CUDA plugin/PJRT all `0.10.0`; backend `gpu`; device `cuda:0`; x64 `true`; `SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled`; `XLA_FLAGS` includes `--xla_gpu_exclude_nondeterministic_ops=true`; `simsopt` imports from the clean `7750e34d8` source archive; `nvidia-smi` records an A100-SXM4-40GB with NVIDIA driver `580.105.08` and visible CUDA `13.0`. Slurm elapsed `15s`; timed preflight wall `0:08.32`; process MaxRSS `749332 KB`; GPU memory before probe `0 MiB / 40960 MiB`. |

Current-SHA Wave 3 GPU pytest slices:

| Job | Result | Evidence |
| --- | --- | --- |
| `53276235` / `gpu-w3-7750` | PASS | Ran against the clean `7750e34d8` source archive and current `jax-0.10.0` runtime. GPU runtime smoke passed `6 passed in 197.47s`; grouped collective lowering control passed `1 passed in 32.90s`; real-fixture GPU M5 parity class passed `3 passed in 92.91s`. Slurm elapsed `5:48`; overall exit `0`; Slurm step MaxRSS values were `4658772K`, `1190036K`, and `2397356K`; per-step `/usr/bin/time -v` MaxRSS values were `3499948 KB`, `1079592 KB`, and `2793800 KB`. Before/after `nvidia-smi` records show 4x A100-SXM4-40GB visible with `0 MiB / 40960 MiB` retained memory after teardown. |

Superseded `7750e34d8` jobs before the clean `c2f59c427f` rerun:

| Job | Purpose | QOS / partition | State at latest audit |
| --- | --- | --- | --- |
| `53275983` / `cpu-w0w1-7750` | Wave 0 import smoke, full CPU tests, focused marker reruns, Wave 1 focused CPU banana tests, and structured CPU parity/proof JSONs, with per-step `/usr/bin/time -v` records. | CPU `shared` / `shared_milan_ss11` | FAILED `import_smoke` because the source archive included macOS AppleDouble metadata; superseded by the clean `c2f59c427f` rerun. |
| `53276949` / `gpu-w4core-7750` | Wave 4 core GPU proof job: Stage 2 CUDA, Stage 2 geometry repro, single-stage CUDA init, and single-stage ladder rungs through `m04n04-i05-useful`, with per-step timing, RSS, and `nvidia-smi` monitor records. Non-banana GPU follow-up is deferred until the CPU baseline JSON exists. | GPU `shared` / `shared_gpu_ss11` | FAILED `stage2_cuda_e2e` with Slurm `OUT_OF_MEMORY` under the no-preallocation policy; superseded by the preallocated `c2f59c427f` rerun. |

Next GPU submissions after `53276949` must run with
`SIMSOPT_JAX_GPU_PREALLOCATE=true` and
`XLA_PYTHON_CLIENT_PREALLOCATE=true`. Job `53276949` was launched with
`XLA_PYTHON_CLIENT_PREALLOCATE=false`, so its timing and memory records are
interpreted as the no-preallocation baseline rather than the
preallocated-memory rerun.

Official-docs check for this current-SHA run:

- Context7 JAX docs for `/google/jax` state that JAX dispatch is asynchronous
  and performance timing must wait for results with `.block_until_ready()` or
  an equivalent synchronization. The current plan therefore treats benchmark
  script timings and Slurm `/usr/bin/time -v` records as process-level evidence,
  while lower-level JAX microbenchmarks must synchronize explicitly.
- NERSC job-policy documentation was checked for QOS selection. The current
  run uses `debug` only for short GPU preflight/pytest canaries and CPU
  `shared` for the longer Wave 0/1 baseline.
- NVIDIA CUDA best-practices documentation was checked for host/device transfer
  interpretation; unnecessary host-device transfers remain a performance smell,
  and strict-transfer failures are treated as correctness issues for GPU-pure
  proof lanes.

## Current Rerun Update: Clean Committed SHA `c2f59c427f`

This update supersedes the open `7750e34d8` jobs after two concrete run issues:

- CPU `53275983` failed `import_smoke` because the previous source archive
  contained macOS AppleDouble metadata files such as `.__bfgs.py`; the private
  optimizer source-scan test treated those binary metadata files as Python
  source. Commit `c2f59c427f` fixes the scanner to inspect visible Python
  source files only and adds a binary hidden-metadata regression.
- GPU `53276949` failed the first required Wave 4 step,
  `stage2_cuda_e2e`, with Slurm `OUT_OF_MEMORY` at MaxRSS `58474700K`
  against a `57472M` allocation. That run used
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`; it is retained as the no-preallocation
  host-memory baseline, not as a failed physics/parity artifact.

The replacement source archive was built with `git archive` for the main repo
and each submodule to avoid carrying local filesystem metadata:

- Committed SHA:
  `c2f59c427fc5489a396d61141ffe95ec860df6e3`
- Commit: `test: ignore hidden metadata in source scan`
- Remote source archive:
  `/pscratch/sd/j/jungdae/simsopt-jax-clean-c2f59c427f-20260522T033739Z-src`
- Archive verification: `src/simsopt/__init__.py` exists,
  `thirdparty/pybind11/CMakeLists.txt` exists, and no `.__*` or `._*`
  files are present.
- Local working tree still has unrelated dirt:
  `conda.recipe/meta.yaml` and `.conda/`.

Submitted replacement jobs:

| Job | Purpose | QOS / partition | State at submission |
| --- | --- | --- | --- |
| `53279137` / `inst-c2f59` | Superseded install/provenance job. | CPU `shared` / `shared_milan_ss11` | CANCELED before start; short install work belongs in debug. |
| `53279138` / `cpu-w0w1-c2f59` | Superseded CPU rerun dependent on `53279137`. | CPU `shared` / `shared_milan_ss11` | CANCELED before start. |
| `53279139` / `gpu-w4pre-c2f59` | Superseded GPU rerun dependent on `53279137`. | GPU `shared` / `shared_gpu_ss11` | CANCELED before start. |
| `53279392` / `instdbg-c2f59` | Superseded debug install/provenance job. | CPU `debug` / `regular_milan_ss11` | FAILED in `00:00:04`: editable metadata generation could not infer a `setuptools-scm` version from the clean source archive because it has no `.git` directory. |
| `53279393` / `cpu-w0w1-c2f59` | Superseded CPU rerun dependent on `53279392`. | CPU `shared` / `shared_milan_ss11` | CANCELED after `53279392` failed. |
| `53279394` / `gpu-w4pre-c2f59` | Superseded GPU rerun dependent on `53279392`. | GPU `shared` / `shared_gpu_ss11` | CANCELED after `53279392` failed. |
| `53279659` / `instscm-c2f59` | Reinstall runtime editable package from the clean `c2f59c427f` archive with `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SIMSOPT=0.1.dev1888+gc2f59c427f`, then verify `simsopt.__file__` points at that archive. | CPU `debug` / `regular_milan_ss11` | COMPLETED in `00:03:04`; batch MaxRSS `14165100K`; installed `simsopt-0.1.dev1888+gc2f59c427f`; runtime provenance shows JAX/JAXLIB/CUDA plugin/PJRT `0.10.0`, CPU backend for install, x64 enabled, and `simsopt.__file__` under `/pscratch/sd/j/jungdae/simsopt-jax-clean-c2f59c427f-20260522T033739Z-src/src/simsopt/__init__.py`. |
| `53279660` / `cpu-w0w1-c2f59` | Rerun Wave 0 import smoke, full CPU tests, focused marker reruns, Wave 1 focused CPU banana tests, and structured CPU parity/proof JSONs. | CPU `shared` / `shared_milan_ss11` | `PENDING`; requested `06:00:00` and `60960M`. |
| `53279661` / `gpu-w4pre-c2f59` | Superseded GPU rerun with `XLA_PYTHON_CLIENT_PREALLOCATE=true`. | GPU `shared` / `shared_gpu_ss11` | FAILED in `00:00:07` before any benchmark step: the provenance Python block imported `jax` before `simsopt`, so `simsopt` correctly rejected late GPU-memory environment resolution. This is a run-script ordering issue, not a physics/parity result. |
| `53279966` / `gpu-w4pre2-c2f59` | Superseded GPU rerun with fixed provenance import ordering. | GPU `shared` / `shared_gpu_ss11` | CANCELED after `00:02:04`: runtime provenance proved backend `gpu` and JAX/JAXLIB/CUDA plugin/PJRT `0.10.0`, but reported `preallocate=false` even though the raw Slurm env had `XLA_PYTHON_CLIENT_PREALLOCATE=true`. Root cause: SIMSOPT runtime owns the memory-policy env and rewrote the downstream JAX env from its `jax_gpu_parity` default because `SIMSOPT_JAX_GPU_PREALLOCATE` was not set. This is a run-script policy issue, not a physics/parity result. |
| `53280092` / `gpu-w4pre3-c2f59` | Corrected Wave 4 core GPU proof with both `SIMSOPT_JAX_GPU_PREALLOCATE=true` and `XLA_PYTHON_CLIENT_PREALLOCATE=true`, plus the `simsopt`-before-`jax` provenance import order. | GPU `shared` / `shared_gpu_ss11` | RUNNING on `nid008528`; runtime provenance now proves backend `gpu`, device `cuda:0`, JAX/JAXLIB/CUDA plugin/PJRT `0.10.0`, x64 enabled, and `preallocate=true`. The first `stage2_cuda_e2e` monitor samples show expected preallocation behavior: GPU memory rose from `0 MiB / 81920 MiB` to about `61757 MiB / 81920 MiB` before solver output. |

Official-docs check for the rerun:

- Context7 JAX docs for `/google/jax` state that
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` disables JAX's default GPU memory
  preallocation and allocates GPU memory on demand. The replacement GPU run
  intentionally uses SIMSOPT's `SIMSOPT_JAX_GPU_PREALLOCATE=true` policy knob
  so the SIMSOPT runtime writes `XLA_PYTHON_CLIENT_PREALLOCATE=true` before JAX
  imports and initializes the CUDA client.
- Context7 `setuptools-scm` docs for `/pypa/setuptools-scm` state that
  source archives without `.git` need archive metadata or an explicit pretend
  version override. The replacement debug install uses the package-specific
  `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SIMSOPT` override, following the documented
  PEP 503-normalized `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_${DIST_NAME}` form.
- NERSC Perlmutter running-jobs documentation says 1-GPU jobs should use
  `shared` QOS, shows the 1-node/1-task/1-GPU header with `-c 32` and
  `--gpus-per-task=1`, and gives `--mem-per-gpu` as the way to size memory
  for single-GPU shared steps. A Slurm `sbatch --test-only` probe accepted the
  replacement GPU header with `--mem-per-gpu=110G`.

## Current Execution Update: Clean Committed SHA `7e297e94b`

Commit `7e297e94b291cab397a13d82d9d489055724e3f9`
(`fix: restore clean-source parity validation`) supersedes the `c2f59c427f`
evidence for final signoff. The older `c2f59c427f` GPU preallocation job is
still useful as a preallocation/resource probe, but it is no longer the current
release candidate.

Current source and local validation:

- Remote clean source:
  `/pscratch/sd/j/jungdae/simsopt-jax-clean-7e297e94b-20260522T0542Z-src`
- Source transfer archive:
  `/pscratch/sd/j/jungdae/simsopt-jax-slurm/simsopt-jax-7e297e94b.tar.gz`
- Local focused validation before commit:
  `48 passed in 10.37s`; `ruff check` passed; `py_compile` passed;
  `git diff --check` passed.
- Unrelated local dirt remains outside this proof slice:
  `conda.recipe/meta.yaml` and `.conda/`.

Current `7e297e94b` Slurm evidence and queued work:

| Job | Purpose | QOS / partition | State |
| --- | --- | --- | --- |
| `53283836` / `focus-7e297` | Clean-source focused regression validation for the committed import/parity fixes. | CPU `debug` / `regular_milan_ss11` | PASS: `48 passed in 17.51s`; `/usr/bin/time` wall `0:20.38`; process MaxRSS `1607184 KB`; Slurm batch MaxRSS `2238844K`; elapsed `00:00:42`; exit `0`. |
| `53284028` / `cpu-w0w1-7e297` | Current-HEAD Wave 0/Wave 1 CPU baseline: import smoke, full CPU tests, focused marker reruns, focused banana CPU tests, and structured CPU proof JSONs with per-step time/RSS. | CPU `shared` / `shared_milan_ss11` | PENDING as of `2026-05-22T05:49Z`; requested `06:00:00`, `32` CPUs, and `64G`; scheduler reason changed from `Priority` to `Resources`. Result root: `/pscratch/sd/j/jungdae/simsopt-jax-results/7e297e94b-wave0-wave1-cpu-20260522T0546Z`. |
| `53284383` / `gpu-w4pre-7e297` | Current-HEAD Wave 4 GPU preallocation proof: Stage 2 CUDA, Stage 2 geometry repro, single-stage CUDA init, and ladder rungs through `m04n04-i05-useful`, with per-step `/usr/bin/time -v` and `nvidia-smi` monitors. | GPU `shared` / `shared_gpu_ss11` | PENDING as of submission; requested `04:00:00`, `1` GPU, `32` CPUs, and `--mem-per-gpu=110G`. Result root: `/pscratch/sd/j/jungdae/simsopt-jax-results/7e297e94b-wave4-gpu-prealloc-simsoptprealloc-20260522T054815Z`. |

The current-HEAD GPU script exports both memory-policy knobs before any JAX
client initialization:

```bash
export JAX_PLATFORMS=cuda,cpu
export SIMSOPT_JAX_PLATFORM=cuda
export SIMSOPT_BACKEND_MODE=jax_gpu_parity
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export SIMSOPT_JAX_GPU_PREALLOCATE=true
export SIMSOPT_REPO_SHA=7e297e94b291cab397a13d82d9d489055724e3f9
```

Official-docs check for the current-HEAD preallocation run:

- Context7 JAX docs for `/google/jax` were rechecked on `2026-05-22`: JAX
  preallocates GPU memory by default, `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  disables preallocation and allocates on demand, and
  `XLA_PYTHON_CLIENT_MEM_FRACTION` controls the preallocated fraction. The
  current-HEAD GPU script therefore keeps `XLA_PYTHON_CLIENT_PREALLOCATE=true`
  explicit and also sets SIMSOPT's upstream policy knob
  `SIMSOPT_JAX_GPU_PREALLOCATE=true`.

Live old-source resource probe retained for context:

- `53280092` / `gpu-w4pre3-c2f59` remains RUNNING on `nid008528`.
  It is still in `stage2_cuda_e2e` with no exit artifact yet. Runtime
  provenance already proves `preallocate=true`; sampled GPU memory is
  `61757 MiB / 81920 MiB`; Slurm step MaxRSS is `76101844K`. This is
  preallocation/resource evidence only until the current-HEAD `53284383` run
  completes.

## Slurm Execution Policy

- [ ] Source checkout and modest environment setup may happen on login nodes;
  heavyweight dependency builds, full tests, and proof workloads run on compute
  nodes.
- [ ] CPU full tests run on CPU compute nodes, not login nodes.
- [ ] Every Slurm job script uses `set -euo pipefail` before any command that
  pipes test/proof output through `tee`.
- [ ] Jobs launched from a clean source archive without `.git` export
  `SIMSOPT_REPO_SHA` and `SIMSOPT_GIT_STATUS_SHORT` so benchmark provenance
  does not call `git rev-parse` inside a non-Git archive.
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
- [ ] `jax-cuda12-plugin` and `jax-cuda12-pjrt` versions for pip-wheel GPU runs
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
- JAX transfer guard: `https://docs.jax.dev/en/latest/transfer_guard.html`
- JAX closed-over constants:
  `https://docs.jax.dev/en/latest/internals/constants.html`
- JAX `closure_convert`:
  `https://docs.jax.dev/en/latest/_autosummary/jax.closure_convert.html`
- JAX `device_put`:
  `https://docs.jax.dev/en/latest/_autosummary/jax.device_put.html`
- JAX `make_jaxpr`:
  `https://docs.jax.dev/en/latest/_autosummary/jax.make_jaxpr.html`
- JAX `jax.extend.core`:
  `https://docs.jax.dev/en/latest/jax.extend.core.html`
- JAX asynchronous dispatch:
  `https://docs.jax.dev/en/latest/async_dispatch.html`
- JAX benchmarking:
  `https://docs.jax.dev/en/latest/benchmarking.html`
- CUDA C++ Best Practices, host-device transfer:
  `https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#data-transfer-between-host-and-device`
- Optax L-BFGS:
  `https://optax.readthedocs.io/en/latest/_collections/examples/lbfgs.html`
- Optimistix L-BFGS:
  `https://docs.kidger.site/optimistix/api/minimise/`
- Equinox enumerations:
  `https://docs.kidger.site/equinox/api/enumerations/`
- SIMSOPT geo BoozerSurface:
  `https://simsopt.readthedocs.io/v1.8.3/simsopt_user.geo.html`
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
- `benchmarks/non_banana_example_cpp_jax_cpu_parity.py`
- `benchmarks/stage2_e2e_comparison.py`
- `benchmarks/single_stage_init_parity.py`
- `benchmarks/single_stage_outer_loop_probe.py`
- `benchmarks/tier5_performance_characterization.py`
- `benchmarks/cpu_run_code_benchmark.py`
- `benchmarks/gpu_run_code_benchmark.py`
- `benchmarks/fixtures/single_stage_seed_iota15/`
- `scripts/run_gpu_parity.sh`
