# P8 External Signoff Gates

These items are not local checklist TODOs. They require hardware, wall-clock, or
multi-platform evidence that is outside this CPU-only workstation. Local device
probe on this pass returned `jax.devices() == [CpuDevice(id=0)]` and
`jax.default_backend() == "cpu"`.

## Required Evidence

1. Integration-level testing: run the existing CUDA e2e workflow
   `.github/workflows/jax_smoke.yml::jax-gpu-e2e`, which emits
   `benchmark_artifacts/stage2_e2e_cuda.json` and
   `benchmark_artifacts/single_stage_init_cuda.json`.
2. CUDA/GPU determinism: run `.github/workflows/jax_gpu_parity.yml` with
   `XLA_FLAGS=--xla_gpu_deterministic_ops=true`, including
   `scripts/jax_ci_contract.py --platform cuda`.
3. Production-scale benchmarks: run
   `.github/workflows/jax_h200_production_proof.yml` or the equivalent
   `benchmarks/hf_jobs/launch_production_gpu_proof.py --hardware h200
   --platform cuda` path with a current SHA and archived JSON proof.
4. Cross-platform parity: compare current-SHA artifacts from macOS Apple
   Silicon CPU, Linux x86_64 CPU, and CUDA GPU. The LS path remains
   non-portable unless the artifact explicitly documents the platform lane.
5. Long-trajectory robustness: run a dedicated tracing soak with
   `tmax ~= 1 sec` scale and preserve trajectory status, accepted-step counts,
   stop reasons, and non-finite checks. This is intentionally not a unit test.
6. Concurrent-call safety: run a multiprocessing or dask stress harness that
   exercises the public JAX wrappers concurrently with isolated runtime state
   and records process count, backend, transfer guard, and failure policy.

## Checklist Policy

Do not mark these gates as complete from local CPU tests, stale CUDA artifacts,
documentation-only evidence, or JAX CPU/GPU agreement without the upstream
CPU/C++/SciPy oracle leg. Completion requires current-SHA artifact provenance:
git SHA, platform, JAX/JAXLIB version, XLA flags, device facts, command line,
output JSON, and pass/fail summary.
