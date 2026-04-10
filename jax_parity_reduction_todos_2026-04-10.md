# simsopt-jax Parity Tightening Plan

**Date:** 2026-04-10
**Scope:** CPU -> JAX CPU -> JAX GPU parity for reduction-heavy kernels and solver-adjacent objective paths
**Status:** Planning document

## Context

This repo already uses the right high-level parity architecture:

- `native_cpu` is the reference lane
- `jax_cpu_parity` is the algorithmic parity lane
- `jax_gpu_parity` is the device parity lane
- `strict=True` should reject fallback seams instead of silently mixing JAX and CPU/C++

Relevant repo docs:

- [docs/using_jax_backend.md](docs/using_jax_backend.md)
- [docs/source/jax_acceptance.rst](docs/source/jax_acceptance.rst)
- [docs/source/jax_gpu_setup.rst](docs/source/jax_gpu_setup.rst)

External primary-source context:

- JAX x64 defaults and enabling `jax_enable_x64`: https://docs.jax.dev/en/latest/default_dtypes.html
- JAX async dispatch: https://docs.jax.dev/en/latest/async_dispatch.html
- JAX transfer guard: https://docs.jax.dev/en/latest/transfer_guard.html
- NVIDIA CUDA floating-point behavior: https://docs.nvidia.com/cuda/archive/13.0.0/floating-point/index.html
- PyTorch reproducibility notes: https://docs.pytorch.org/docs/stable/notes/randomness.html
- TensorFlow op determinism: https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism
- ReproBLAS: https://bebop.cs.berkeley.edu/reproblas/
- ExBLAS: https://github.com/riakymch/exblas
- OzBLAS: https://www.r-ccs.riken.jp/labs/lpnctrt/projects/ozblas/

## Decision Record

- [x] Do **not** require bitwise CPU/GPU identity for the full stack.
- [x] Treat exact parity as a mirrored test contract under fixed fixtures, `x64`, strict mode, and explicit sync.
- [x] Keep solver parity contract-based: final objective, residual norm, gradients, and physics quantities.
- [x] Only consider stronger reproducible arithmetic for reduction-dominated kernels.

## Priority Summary

1. Biot-Savart reductions
2. `integral_BdotN` reductions
3. Boozer residual scalar reductions
4. Tighten object-level parity tests around those kernels
5. Leave `optimizer_jax.py` on a solver-contract parity model

## Shared Harness TODOs

- [x] Add or update a parity manifest mapping `upstream test -> JAX test -> exact / partial / missing`
- [x] Standardize `jax_enable_x64=True` before any test arrays are created
- [x] Standardize `strict=True` for parity lanes
- [x] Standardize explicit host materialization or `block_until_ready()` before timing and cross-device assertions
- [x] Standardize seeded fixtures across CPU/JAX lanes
- [x] Separate CPU parity and GPU parity assertions in every mirrored test

## Kernel Matrix

### 1. Biot-Savart

**Files**

- `src/simsopt/jax_core/biotsavart.py`
- `src/simsopt/field/biotsavart_jax.py`
- `tests/field/test_biotsavart_jax_parity.py`

**Risk**

- Drift risk: medium-high
- Best candidate for stronger reproducible reduction work

**Reduction sites**

- Quadrature reductions in `_quadrature_block_integral()`
- Coil-chunk accumulation in `_coil_chunk_reduce()`
- Final current contraction is lower priority

**TODOs**

- [x] Add a reusable pairwise/tree reduction helper for axis reductions in `src/simsopt/jax_core/biotsavart.py`
- [x] Replace `jnp.sum(values, axis=1)` in `_quadrature_block_integral()` with the pairwise helper
- [x] Replace `jnp.sum(block_integrand, axis=1)` in `_quadrature_block_integral()` with the pairwise helper
- [x] Evaluate whether coil-chunk accumulation should use a fixed binary reduction tree instead of serial `acc + reduce_chunk(...)`
  - Current decision: keep the outer coil-chunk accumulation serial until parity data points there; the new many-coil / many-quadrature stress lane compares dense, quadrature-only chunking, and fully chunked accumulation and did not justify widening the hot-path reduction change further.
- [x] Keep the final `jnp.einsum("c,cj->j", ...)` unchanged unless parity data shows it is a real error source
- [x] Add a parity regression test that stresses many-coil / many-quadrature accumulation order
- [x] Document the expected performance cost of pairwise reductions before merging

### 2. Integral BdotN

**Files**

- `src/simsopt/objectives/integral_bdotn_jax.py`
- `src/simsopt/objectives/fluxobjective_jax.py`
- `tests/objectives/test_integral_bdotn_jax.py`
- `tests/integration/test_stage2_jax.py`

**Risk**

- Drift risk: medium
- Best second target after Biot-Savart

**Reduction sites**

- Global sums of `|n|`
- Global sums of `|B|^2 |n|`
- Final scalar objective reduction via `vdot(residual, residual)`

**TODOs**

- [x] Add a reusable pairwise/tree reduction helper for global quadrature sums in `src/simsopt/objectives/integral_bdotn_jax.py`
- [x] Replace denominator accumulation in `"normalized"` mode with the pairwise helper
- [x] Evaluate whether final scalar objective accumulation should use compensated summation in strict-oracle mode
  - Current decision: keep the final `jnp.vdot(residual, residual)` contraction unchanged for now. The reduction probe and mirrored parity tests pointed at the normalized denominator as the active drift site; a stricter compensated scalar path remains deferred until parity data shows the final residual norm is the real bottleneck.
- [x] Keep `fluxobjective_jax.py` as a wrapper around the stabilized kernel instead of adding arithmetic complexity there
- [x] Add a dedicated mirrored `test_fluxobjective_jax_parity.py` instead of relying only on split integration coverage
- [x] Add parity cases for `quadratic flux`, `normalized`, and `local` with degenerate normals and singular-field behavior

### 3. Boozer Residual

**Files**

- `src/simsopt/geo/boozer_residual_jax.py`
- `tests/geo/test_boozer_residual_jax.py`
- `tests/integration/test_single_stage_jax_cpu_reference.py`

**Risk**

- Drift risk: medium-high
- Arithmetic stabilization is useful only after the composed derivative path is complete

**Reduction sites**

- `B2 = jnp.sum(B * B, axis=-1)`
- Final scalar objective `jnp.sum(rtil * rtil)`

**TODOs**

- [ ] Finish the composed derivative path first; do not optimize arithmetic around the M1 limitation
- [ ] Add a reusable pairwise/tree reduction helper for `B2` and scalar penalty accumulation
- [ ] Evaluate compensated summation for the final scalar objective if this objective is the observed parity bottleneck
- [ ] Add a stress parity test where the residual norm is near the current tolerance floor
- [ ] Keep vector-level parity and scalar-level parity as separate checks

### 4. FluxObjective Wrapper

**Files**

- `src/simsopt/objectives/fluxobjective_jax.py`
- `tests/objectives/test_fluxobjective.py`
- `tests/integration/test_stage2_jax.py`

**Risk**

- Drift risk: inherited from underlying kernels, not the wrapper itself

**TODOs**

- [x] Add a native-only parity mode so wrapper tests fail on fallback seams
- [x] Create mirrored object-level tests for definitions, derivatives, and edge-case contracts
- [ ] Keep arithmetic stabilization work in `integral_bdotn_jax.py`, not here

### 5. Surface Objectives Family

**Files**

- `src/simsopt/geo/surfaceobjectives_jax.py`
- `src/simsopt/geo/label_constraints_jax.py`
- `src/simsopt/geo/surface_fourier_jax.py`
- `tests/geo/test_surface_objectives_jax.py`
- `tests/geo/test_label_constraints_jax.py`

**Risk**

- Main blocker is API completeness, not arithmetic reproducibility

**TODOs**

- [x] Complete missing `BiotSavartJAX` object-level methods needed for literal upstream test mirroring
- [x] Finish composed derivative plumbing before adding reproducible-arithmetic experiments here
- [x] Mirror upstream ToroidalFlux and related objective tests more literally once the API is complete
- [ ] Keep this family on tolerance-based parity for now

### 6. BoozerSurface and Optimizer

**Files**

- `src/simsopt/geo/boozersurface_jax.py`
- `src/simsopt/geo/optimizer_jax.py`
- `tests/geo/test_boozersurface_jax.py`
- `tests/integration/test_single_stage_jax_cpu_reference.py`

**Risk**

- Drift risk: high
- Wrong place to start with reproducible summation

**Why**

Solver drift is dominated by:

- GMRES iteration behavior
- dense fallback decisions
- branch sensitivity
- backend linear algebra differences
- nonlinear acceptance logic

**TODOs**

- [ ] Keep solver parity defined by end-state quality, not identical iterates
- [ ] Add explicit mirrored tests for residual norm, convergence success, final objective, and final physics quantities
- [ ] Do not attempt ExBLAS/ReproBLAS-style reproducibility inside `optimizer_jax.py` as a first-line fix
- [ ] Add solver logs or diagnostics to explain accepted drift envelopes when parity tests fail

### 7. SurfaceRZFourier

**Files**

- `src/simsopt/jax_core/surface_rzfourier.py`
- `tests/geo/test_surface_rzfourier_jax.py`

**Risk**

- Drift risk: low

**TODOs**

- [ ] Keep this path on strict tolerance-based parity
- [ ] Do not add reproducible summation complexity here unless a concrete failure appears
- [ ] Focus only on missing object/API parity if full-class parity is required

## Reproducible Arithmetic Escalation Ladder

- [ ] Tier 1: pairwise/tree reductions in hot scalar and vector reductions
- [ ] Tier 2: compensated summation for especially unstable scalar objectives
- [ ] Tier 3: dedicated strict-oracle reduction mode for selected kernels
- [ ] Tier 4: evaluate whether ExBLAS/ReproBLAS/OzBLAS-style exact reproducibility is worth the complexity
- [ ] Gate Tier 4 behind demonstrated need, because throughput and implementation complexity will rise sharply

## Test TODOs

- [ ] Add a parity test matrix document for:
  - Biot-Savart
  - FluxObjective
  - SurfaceObjectives family
  - BoozerSurface
  - SurfaceRZFourier
- [ ] Add one stress test per hot reduction kernel where accumulation order is likely to matter
- [ ] Add one CPU-parity lane and one GPU-parity lane for every new mirrored test
- [ ] Keep tolerances tied to documented acceptance tiers, not ad hoc values

## Exit Criteria

- [ ] Biot-Savart parity remains green after pairwise reduction changes
- [ ] `integral_BdotN` parity remains green after reduction changes
- [ ] Stage 2 parity remains green with no hidden fallback seams
- [ ] Single-stage parity remains green under the documented solver-contract acceptance model
- [ ] No new reduction-stability work is attempted in `optimizer_jax.py` unless a concrete failure points there

## Notes

- Most successful CPU -> GPU ports do not guarantee full CPU/GPU bitwise identity.
- The standard success model is: deterministic settings where possible, tolerance-based parity against a CPU oracle, and stronger reproducible arithmetic only in selected numerically sensitive kernels.
- This repo should follow that model unless a publication or customer requirement forces stronger reproducibility.
