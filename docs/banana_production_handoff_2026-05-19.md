# Banana Stage 2 And Single-Stage Production Handoff

Date: 2026-05-19

Scope: install, run, validate, and reproduce the banana Stage 2 plus
single-stage workflow from explicit artifacts. This document does not change
solver behavior, optimizer tolerances, or parity thresholds.

## Install

Run from the repository root.

CPU/JAX development lane:

```bash
python -m pip install -e ".[deploy]"
```

CUDA lane:

```bash
python -m pip install -e ".[deploy_gpu]"
```

The repository extras are the SSOT. `deploy` includes the CPU/JAX development
dependencies used by the banana workflow. `deploy_gpu` routes through the
repo `JAX_GPU` extra, which currently installs `jax[cuda12]`. Current JAX docs
also publish `jax[cuda13]`; changing CUDA wheel families is an explicit
environment-lane decision, not a banana workflow change.

Runtime selectors:

```bash
# CPU reference oracle
export SIMSOPT_BACKEND_MODE=native_cpu
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

```bash
# JAX CPU parity
export SIMSOPT_BACKEND_MODE=jax_cpu_parity
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

```bash
# JAX GPU parity
export SIMSOPT_BACKEND_MODE=jax_gpu_parity
export SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled
export SIMSOPT_JAX_PLATFORM=cuda
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset LD_LIBRARY_PATH
```

JAX treats `JAX_PLATFORMS` as an ordered list: every listed platform must
initialize, and the first platform is the default. JAX also preallocates GPU
memory by default, so production proof jobs set
`XLA_PYTHON_CLIENT_PREALLOCATE=false` before import. For pip-installed CUDA
wheels, do not let `LD_LIBRARY_PATH` override the bundled NVIDIA libraries.

## Stage 2

CPU reference run:

```bash
SIMSOPT_BACKEND_MODE=native_cpu \
JAX_ENABLE_X64=1 \
JAX_PLATFORMS=cpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python examples/single_stage_optimization/STAGE_2/banana_coil_solver.py \
  --backend cpu \
  --optimizer-backend scipy \
  --plasma-surf-filename wout_nfp22ginsburg_000_014417_iota15.nc \
  --major-radius 0.915 \
  --toroidal-flux 0.24 \
  --banana-surf-radius 0.22 \
  --skip-postprocess
```

JAX CPU parity run:

```bash
SIMSOPT_BACKEND_MODE=jax_cpu_parity \
JAX_ENABLE_X64=1 \
JAX_PLATFORMS=cpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python examples/single_stage_optimization/STAGE_2/banana_coil_solver.py \
  --backend jax \
  --optimizer-backend ondevice \
  --plasma-surf-filename wout_nfp22ginsburg_000_014417_iota15.nc \
  --major-radius 0.915 \
  --toroidal-flux 0.24 \
  --banana-surf-radius 0.22 \
  --skip-postprocess
```

The external handoff is an explicit artifact directory. A restartable Stage 2
handoff must contain:

- `results.json`
- `biot_savart_opt.json`
- `surf_opt.json`

Generate a seed catalog:

```bash
python examples/single_stage_optimization/STAGE_2/stage2_seed_report.py \
  --scan-root examples/single_stage_optimization/STAGE_2 \
  --output-json .artifacts/stage2_seed_catalog.json \
  --require-pass
```

Use `--stage2-bs-path /path/to/stage2/run/biot_savart_opt.json` for external
handoff. `--stage2-source database` is an internal/archive resolver for old
Columbia paths, not the production default.

## Seed Catalog

The checked-in reduced fixture is:

```text
benchmarks/fixtures/single_stage_seed_iota15/
```

It contains:

- `results.json`
- `biot_savart_opt.json`
- `single_stage_jax_runtime_spec.json`

It is the canonical small proof fixture for copy-paste startup commands. It is
not a complete Stage 2 seed-catalog candidate because it intentionally does not
contain `surf_opt.json`; `stage2_seed_report.py` should report that limitation
when run directly against the fixture directory.

## Runtime Seed Specs

`single_stage_jax_runtime_spec.json` is the production JAX startup artifact. It
freezes the immutable seed surface, coil spec, coil dofs, Boozer initialization
scalars, hardware constants, and Stage 2 seed metadata.

Compile a new spec from a warm-start run:

```bash
python examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
  --warm-start-run-dir /path/to/single_stage/warm_start_run \
  --compile-jax-runtime-seed-spec \
  --jax-runtime-seed-spec /path/to/single_stage_jax_runtime_spec.json
```

Use the existing fixture spec for reduced proof commands:

```text
benchmarks/fixtures/single_stage_seed_iota15/single_stage_jax_runtime_spec.json
```

## Single Stage

CPU reference init proof:

```bash
SIMSOPT_BACKEND_MODE=native_cpu \
JAX_ENABLE_X64=1 \
JAX_PLATFORMS=cpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
  --backend cpu \
  --optimizer-backend scipy \
  --stage2-bs-path benchmarks/fixtures/single_stage_seed_iota15/biot_savart_opt.json \
  --iota-target 0.17 \
  --vol-target 0.10 \
  --cc-dist 0.07 \
  --mpol 15 \
  --ntor 6 \
  --init-only \
  --minimal-artifacts \
  --output-root .artifacts/single_stage_cpu_init
```

JAX CPU parity init proof:

```bash
SIMSOPT_BACKEND_MODE=jax_cpu_parity \
JAX_ENABLE_X64=1 \
JAX_PLATFORMS=cpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
  --backend jax \
  --optimizer-backend ondevice \
  --boozer-optimizer-backend ondevice \
  --jax-runtime-seed-spec benchmarks/fixtures/single_stage_seed_iota15/single_stage_jax_runtime_spec.json \
  --iota-target 0.17 \
  --vol-target 0.10 \
  --cc-dist 0.07 \
  --mpol 15 \
  --ntor 6 \
  --init-only \
  --minimal-artifacts \
  --output-root .artifacts/single_stage_jax_cpu_init
```

JAX GPU parity init proof:

```bash
SIMSOPT_BACKEND_MODE=jax_gpu_parity \
SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled \
SIMSOPT_JAX_PLATFORM=cuda \
JAX_ENABLE_X64=1 \
JAX_PLATFORMS=cuda,cpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
env -u LD_LIBRARY_PATH \
python examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
  --backend jax \
  --optimizer-backend ondevice \
  --boozer-optimizer-backend ondevice \
  --jax-runtime-seed-spec benchmarks/fixtures/single_stage_seed_iota15/single_stage_jax_runtime_spec.json \
  --iota-target 0.17 \
  --vol-target 0.10 \
  --cc-dist 0.07 \
  --mpol 15 \
  --ntor 6 \
  --init-only \
  --minimal-artifacts \
  --output-root .artifacts/single_stage_jax_gpu_init
```

## Validation Ladder

Run proof in this order:

1. Stage 2 CPU reference command.
2. Stage 2 JAX CPU parity command.
3. Single-stage CPU reference init proof.
4. Single-stage JAX CPU parity init proof.
5. Single-stage JAX GPU parity init proof on a CUDA node.
6. Structured benchmark artifacts:

```bash
python benchmarks/stage2_e2e_comparison.py \
  --platform cpu \
  --output-json .artifacts/stage2_e2e_cpu.json

python benchmarks/stage2_e2e_comparison.py \
  --platform cuda \
  --output-json .artifacts/stage2_e2e_cuda.json

python benchmarks/single_stage_init_parity.py \
  --platform cpu \
  --stage2-bs-path benchmarks/fixtures/single_stage_seed_iota15/biot_savart_opt.json \
  --jax-runtime-seed-spec benchmarks/fixtures/single_stage_seed_iota15/single_stage_jax_runtime_spec.json \
  --case-artifacts-dir .artifacts/single_stage_init_cpu \
  --output-json .artifacts/single_stage_init_cpu.json

python benchmarks/single_stage_init_parity.py \
  --platform cuda \
  --stage2-bs-path benchmarks/fixtures/single_stage_seed_iota15/biot_savart_opt.json \
  --jax-runtime-seed-spec benchmarks/fixtures/single_stage_seed_iota15/single_stage_jax_runtime_spec.json \
  --case-artifacts-dir .artifacts/single_stage_init_cuda \
  --output-json .artifacts/single_stage_init_cuda.json

python benchmarks/tier5_performance_characterization.py \
  --platform cuda \
  --stage2-bs-path benchmarks/fixtures/single_stage_seed_iota15/biot_savart_opt.json \
  --output-json .artifacts/tier5_cuda.json \
  --phase gpu
```

The trust chain is:

```text
Existing SIMSOPT C++/SciPy behavior
-> JAX CPU matches
-> JAX CUDA/GPU matches
-> JAX CPU and GPU match each other
```

JAX CPU versus JAX GPU agreement alone is not enough.

## Expected Artifacts

Stage 2:

- `results.json`
- `biot_savart_opt.json`
- `surf_opt.json`
- optional VTK and PNG diagnostics when post-processing is enabled

Single stage:

- `results.json`
- `biot_savart_opt.json`
- `surf_opt.json`
- `single_stage_jax_runtime_spec.json`
- optional VTK, PNG, log, Poincare, and profiler artifacts

Structured proof:

- JSON output from each benchmark runner
- exact repo SHA and dirty-tree status
- JAX/JAXLIB and CUDA plugin versions
- backend, device list, and x64 state
- RSS and GPU memory observations for hardware runs

## Memory And Performance Interpretation

Separate correctness from performance. A cold JAX run includes compilation and
can be slower than the CPU oracle while still passing parity. Interpret warm
run timings separately from cold compile overhead.

For CPU runs, use maximum resident set size and wall time. For GPU runs, record
`nvidia-smi`, JAX device list, `jax.default_backend()`, x64 status, and sampled
GPU memory. With `XLA_PYTHON_CLIENT_PREALLOCATE=false`, GPU memory traces are
more interpretable, but disabling preallocation can increase fragmentation risk
for jobs that nearly fill the device.

## Perlmutter

Use the durable runner:

```bash
export RESULTS_ROOT="${SCRATCH}/simsopt-jax-results/$(git rev-parse --short HEAD)"
export JAX_GPU_WHEEL_SPEC="jax[cuda12]"
sbatch -A <gpu_account> benchmarks/perlmutter/banana_e2e_cpu_gpu.slurm
```

If the account has a default GPU project, omit `-A`. The script uses Perlmutter
`shared` QOS for the 1-GPU smoke/proof path. Four-GPU comparisons require a
full-node `regular` job; do not repurpose the shared 1-GPU script for that.
`JAX_GPU_WHEEL_SPEC` is for pinning the repo CUDA 12 wheel lane, for example
`jax[cuda12]==<version>`. Switching to CUDA 13 requires changing the repo
`JAX_GPU` extra and the runner's CUDA plugin provenance probe together.

## Sources

- JAX installation: https://docs.jax.dev/en/latest/installation.html
- JAX configuration options: https://docs.jax.dev/en/latest/config_options.html
- JAX GPU memory allocation: https://docs.jax.dev/en/latest/gpu_memory_allocation.html
- SIMSOPT installation: https://simsopt.readthedocs.io/latest/installation.html
- NERSC running jobs on Perlmutter: https://docs.nersc.gov/systems/perlmutter/running-jobs/
- NERSC Slurm job basics: https://docs.nersc.gov/jobs/
- NERSC queues and charges: https://docs.nersc.gov/jobs/policy/
