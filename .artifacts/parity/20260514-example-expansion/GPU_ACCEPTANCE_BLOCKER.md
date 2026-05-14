# GPU Acceptance Blocker

Date: 2026-05-14

This artifact records why the remaining GPU-later checklist items in
`docs/example_cpp_jax_cpu_gpu_parity_expansion_plan_2026-05-14.md` are not
closed by the current local evidence.

## Remaining Plan Items

- Lines 542-544: permanent-magnet JAX GPU fixture rerun and JAX GPU vs JAX CPU
  payload-preserving comparison.
- Lines 634-636: wireframe JAX GPU fixed-state/matrix/solve comparison.
- Lines 713-715: tracing JAX GPU native-field endpoint/event comparison.

## Current Local Evidence

- `nvidia-smi` is not installed on this host.
- Repo-local JAX reports backend `cpu` with devices `[('cpu', 'cpu', 0)]`.
- `runpodctl pod list` returns `[]`.
- `runpodctl user` reports a negative client balance and `currentSpendPerHr: 0`.
- The current 2026-05-14 artifacts select only `cpu_cpp,jax_cpu`.
- All current 2026-05-14 artifacts record `jax_gpu.status = runtime_required`.

## Existing Artifact Boundary

Older GPU/H100 artifacts under `.artifacts/parity/` do not satisfy these
checklist items. They are earlier single-stage or fixed-state proof artifacts
and do not contain the 2026-05-14 example fixture IDs or fixture input hashes.

The exact current PM/wireframe/tracing example rows are only present in:

- `.artifacts/parity/20260514-example-expansion/all-supported-cpu.json`
- `.artifacts/parity/20260514-example-expansion/all-fixtures.json`

Those artifacts are CPU artifacts and are the baseline inputs for the later GPU
run.

## Command Required To Close

Run this on a CUDA-enabled host with the same checked-out code and a valid GPU
runtime:

```bash
SIMSOPT_BACKEND_MODE=jax_gpu_parity \
SIMSOPT_JAX_PLATFORM=cuda \
SIMSOPT_EXAMPLE_PARITY_JAX_PLATFORM=cuda \
JAX_PLATFORMS=cuda \
JAX_ENABLE_X64=1 \
conda run -n jax-0.9.2 python benchmarks/non_banana_example_cpp_jax_cpu_parity.py \
    --fixtures all-supported \
    --lanes cpu_cpp,jax_gpu \
    --baseline-json .artifacts/parity/20260514-example-expansion/all-supported-cpu.json \
    --git-sha "$(git rev-parse HEAD)" \
    --dirty-policy record \
    --output-json .artifacts/parity/20260514-example-expansion/all-supported-gpu.json
```

After that run, the GPU-later checklist can be closed only if the artifact
contains matching fixture input hashes, emits `jax_cpu_vs_jax_gpu` comparisons,
records CUDA provenance, and leaves CPU/C++ vs JAX CPU as the oracle.
