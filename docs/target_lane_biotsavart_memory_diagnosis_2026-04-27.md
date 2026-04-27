# Target Lane Biot-Savart Memory Diagnosis - 2026-04-27

## Status

Resolved for the default supported JAX/ondevice target-lane path.

The remaining 30 GB-class footprint after the pairwise distance fixes was not
from surface-surface or curve-surface pairwise reductions. Compiled XLA memory
analysis isolated the cliff to reverse-mode autodiff through point-chunked
Biot-Savart field evaluation inside the traceable Boozer inner solve.

## Review Inputs

Oracle GPT-5.5 Pro was asked for an independent code review of the memory
diagnosis and candidate optimizations. The response agreed that rematerializing
the Biot-Savart point-chunk kernel is the correct first-order fix, recommended
using `compiled.memory_analysis()` as the primary memory gate, and rejected
opaque runtime fallbacks for the unsupported limited-memory target-lane path.

Independent documentation checks matched that direction:

- JAX checkpoint/remat docs: `jax.checkpoint()` reduces reverse-mode saved
  residuals by recomputing checkpointed internals in the backward pass.
- JAX memory profiling docs: compiled `memory_analysis()` is the right local
  XLA workspace signal; process RSS includes allocator, executable, and runtime
  high-water state.
- JAX GPU memory allocation docs: raw GPU allocation can be dominated by XLA
  preallocation on GPU. This note does not explain the current macOS CPU RSS
  result, but it matters for later CUDA validation.
- SIMSOPT user docs do not define this private JAX target-lane profiling
  contract; current source and tests are the SSOT for this optimization path.

References:

- https://docs.jax.dev/en/latest/gradient-checkpointing.html
- https://docs.jax.dev/en/latest/notebooks/host-offloading.html
- https://docs.jax.dev/en/latest/gpu_memory_allocation.html

## Root Cause

At `mpol=10`, `ntor=10`, `nphi=255`, `ntheta=64`, the traceable target lane
evaluates the Boozer inner solve under the outer optimizer's
`value_and_grad` pipeline. Before the fix, `_point_chunk_reduce()` chunked the
forward Biot-Savart point evaluation, but reverse-mode autodiff still retained
large per-chunk intermediates across the differentiated solve.

The exact high-memory compiled closures were:

| Closure | Before temp memory |
| --- | ---: |
| `optimizer_value_and_grad` | 21945.74 MiB |
| `value_and_grad_pipeline` | 21945.74 MiB |
| `forward_result` | 21936.51 MiB |
| `inner_solve` | 21936.35 MiB |
| `solved_total_gradient` | 11447.12 MiB |

Standalone `field_eval` was not the dominant issue. The high footprint came
from differentiating the full target-lane solve that contains the field path.

## Fix

Patch:

- `src/simsopt/jax_core/biotsavart.py`
- Function: `_point_chunk_reduce()`

The chunk kernel is now wrapped once with `jax.checkpoint()` and used for the
first chunk and every loop chunk. This keeps the forward chunking contract but
tells JAX reverse-mode AD to rematerialize chunk internals during the backward
pass instead of saving them as residuals.


This is a structural memory fix, not a fallback or alternate runtime path. The
mathematical field evaluation is unchanged.

Official JAX documentation basis:

- `jax.checkpoint`, also known as `jax.remat`, controls autodiff saved values.
- With checkpointing, JAX saves the checkpointed function inputs and recomputes
  needed residuals during the backward pass.
- This is the documented tradeoff for lower reverse-mode memory at the cost of
  extra recomputation.

Reference: https://docs.jax.dev/en/latest/gradient-checkpointing.html

## Measurement Evidence

### Compiled XLA Memory Analysis

Before artifact:

`.artifacts/parity/20260427-target-lane-memory-analysis-m10/mpol=10-ntor=10-91df9d3c/results.json`

After artifact:

`.artifacts/parity/20260427-target-lane-memory-analysis-m10-remat1/mpol=10-ntor=10-91df9d3c/results.json`

| Closure | Before temp memory | After temp memory |
| --- | ---: | ---: |
| `optimizer_value_and_grad` | 21945.74 MiB | 305.48 MiB |
| `value_and_grad_pipeline` | 21945.74 MiB | 305.48 MiB |
| `forward_result` | 21936.51 MiB | 296.24 MiB |
| `inner_solve` | 21936.35 MiB | 296.08 MiB |
| `solved_total_gradient` | 11447.12 MiB | 154.28 MiB |

### Process High-Water

Profile run before:

- Log: `.artifacts/parity/20260427-target-lane-memory-analysis-m10.log`
- Wall time: 2863.72 s
- Max RSS: 15979692032 bytes
- Peak memory footprint: 31932797328 bytes

Profile run after:

- Log: `.artifacts/parity/20260427-target-lane-memory-analysis-m10-remat1.log`
- Wall time: 3277.09 s
- Max RSS: 10585063424 bytes
- Peak memory footprint: 20690643296 bytes

Normal `maxiter=1` CPU smoke after:

- Log: `.artifacts/parity/20260427-jax-cpu-memory-smoke-m1-remat1.log`
- Results: `.artifacts/parity/20260427-jax-cpu-memory-smoke-m1-remat1/mpol=10-ntor=10-5264a5d6/results.json`
- Wall time: 2617.65 s
- Max RSS: 9769877504 bytes
- Peak memory footprint: 21490742856 bytes

The process high-water remains much larger than the compiled temporary buffer
because it includes JAX/XLA runtime, compilation, executable storage, and other
allocator high-water state. The actual differentiated target-lane temp workspace
cliff dropped from about 21.9 GiB to about 305 MiB.

## Numerical Outcome Check

The normal CPU smoke run after the fix completed one outer iteration as
expected for `--maxiter 1`.

| Quantity | After remat smoke |
| --- | ---: |
| Final objective | 0.0008323486959847221 |
| Objective decrease | 1.8037581967075486e-7 |
| Final volume | 0.03996450369955034 |
| Final iota | 0.24994449458770404 |
| Max curvature | 94.09060749581626 |

Compared to the earlier CPU log:

- Volume changed from `0.039964503699568193` to `0.03996450369955034`.
- Iota changed from `0.2499444945641246` to `0.24994449458770404`.
- Max curvature changed from `94.09060749137089` to `94.09060749581626`.

These differences are tiny floating-point reduction/order differences from the
same mathematical computation. No math or physics regression was observed in
this smoke.

## Validation Commands Run

```bash
python -m py_compile \
  src/simsopt/jax_core/biotsavart.py \
  examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
  tests/geo/test_single_stage_example.py

pytest tests/geo/test_single_stage_example.py \
  -k "profile_target_lane_memory_analysis or profile_traceable_target_lane_objective_records_memory_analysis or build_target_lane_outer_objectives_profiles or prepare_target_lane_outer_objectives" \
  -q

pytest tests/geo/test_boozersurface_jax.py \
  -k "biot or traceable or field" \
  -q

pytest tests/geo/test_surface_objectives_jax.py \
  -k "surface_to_surface or signed_constraint or alm_smoothmin or selected_smoothmin" \
  -q

pytest tests/integration/test_stage2_jax.py::TestStage2OptimizerContract::test_stage2_target_alm_value_and_grad_matches_host_evaluation \
  -q

git diff --check
```

Observed results:

- `6 passed, 293 deselected`
- `20 passed, 2 skipped, 269 deselected`
- `12 passed, 1 skipped, 121 deselected`
- `1 passed`
- `git diff --check` clean

## Added Diagnostic Surface

Patch:

- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
- `tests/geo/test_single_stage_example.py`

New flag:

```bash
--profile-target-lane-memory-analysis
```

This flag forces target-lane profiling and records XLA
`compiled.memory_analysis()` byte counts for the profiled target-lane closures
and the actual optimizer `value_and_grad` closure. It is a measurement hook only.
It does not change optimizer behavior.

## Added Tuning Surface

Patch:

- `src/simsopt/backend/runtime.py`
- `tests/test_backend.py`
- `tests/conftest.py`
- `tests/field/test_biotsavart_jax.py`
- `tests/integration/test_stage2_jax.py`

New environment override:

```bash
SIMSOPT_JAX_POINT_CHUNK_SIZE=<nonnegative integer>
```

This override controls the Biot-Savart point-axis chunk size directly.
`FieldKernelTuning` carries this value with the coil and quadrature chunk sizes,
so the Biot-Savart kernel factory reads one immutable tuning snapshot. It is
separate from:

- `SIMSOPT_JAX_COIL_CHUNK_SIZE`
- `SIMSOPT_JAX_QUADRATURE_BLOCK_SIZE`
- `SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE`

After rematerialization, point chunk size is the main explicit tuning knob for
trading memory pressure against recomputation overhead. Larger point chunks can
reduce loop/recompute overhead while increasing compiled temporary memory. A
small CPU Biot-Savart gradient microbench at `255 * 64` points, 2 coils, and 96
quadrature samples showed identical values across chunk sizes and this rough
trend:

| Point chunk size | Temp memory with remat | Runtime |
| ---: | ---: | ---: |
| 128 | 3.56 MiB | 0.039 s |
| 256 | 5.63 MiB | 0.026 s |
| 512 | 10.13 MiB | 0.023 s |
| 1024 | 19.13 MiB | 0.021 s |
| 2048 | 37.13 MiB | 0.020 s |
| 4096 | 73.13 MiB | 0.024 s |

`prevent_cse=False` was tested as an experiment because JAX's loop/remat
examples use it for scanned rematerialized functions. The small Biot-Savart
microbench showed slightly higher temp memory and a small speed improvement near
the larger chunks:

| Point chunk size | Temp memory with remat, `prevent_cse=False` | Runtime |
| ---: | ---: | ---: |
| 128 | 4.12 MiB | 0.036 s |
| 256 | 6.75 MiB | 0.027 s |
| 512 | 12.38 MiB | 0.021 s |
| 1024 | 23.63 MiB | 0.019 s |
| 2048 | 46.13 MiB | 0.021 s |
| 4096 | 91.13 MiB | 0.023 s |

The production target-lane memory-analysis gate remains the source of truth for
selecting a default. The new environment variable exists so that gate can test
`512`, `1024`, and `2048` without editing code.

A production-shape target-lane run with `SIMSOPT_JAX_POINT_CHUNK_SIZE=1024` and
the experimental `prevent_cse=False` setting did not beat the point-256
baseline:

- Artifact: `.artifacts/parity/20260427-target-lane-memory-analysis-m10-remat-point1024/mpol=10-ntor=10-1cea19e5/results.json`
- `optimizer_value_and_grad` temp memory: 1098.01 MiB
- `value_and_grad_pipeline` warm time: 376.89 s
- Profile process peak footprint: 20586866632 bytes

For comparison, the point-256 remat baseline with the shipped
`jax.checkpoint(chunk_kernel)` setting stayed at 305.48 MiB
`optimizer_value_and_grad` temp memory and 322.62 s warm
`value_and_grad_pipeline` time. The point-1024 experiment is enough to reject a
blind default increase, but not an apples-to-apples replacement for a future
current-code chunk sweep. Keep point 256 as the current default for this CPU
m10/n10 lane.

## Separate Limited-Memory Contract

`--boozer-limited-memory` is a separate pre-existing target-lane incompatibility.
It routes the traceable Boozer inner solve into the host-loop private L-BFGS
implementation, which calls host array conversion during tracing and cannot be
lowered through `jax.lax.cond` / JAX tracing.

The unsupported JAX/ondevice single-stage contract now fails early when
`boozer_limited_memory=True` is explicitly requested. No defensive fallback was
added. A future limited-memory target-lane solve requires a real trace-safe
L-BFGS implementation.

## Rejected Follow-Ups

### Remove private-optimizer final value-and-gradient refresh

Oracle suggested checking whether the private BFGS/L-BFGS final
value-and-gradient evaluations could be removed. Local tests showed this is not
safe in the current optimizer contract:

```bash
pytest tests/geo/test_boozersurface_jax_private.py \
  -k "minimize_bfgs_private_failed_state_does_not_flip_to_converged or minimize_lbfgs_private_skips_degenerate_curvature_update" \
  -q
```

Removing the final refresh broke existing tests because failed and
degenerate-curvature states rely on that final evaluation to report the true
objective and gradient at the stored iterate. The optimizer final refresh should
stay unless the failure-state reporting contract is redesigned separately.
