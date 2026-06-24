# BiotSavart JAX Performance and Memory Optimization Implementation Plan

## Summary

This plan turns the read-only BiotSavart JAX performance and memory audit into an
implementation track with no-regression gates. It is BiotSavart-specific and
does not widen `docs/jax_perf_mem_optimization_implementation_plan.md`, which
explicitly excludes production BiotSavart `B/dB` kernel changes.

Source validation was performed against HEAD `0b6ccf70c` on 2026-06-24. The
audit findings are source-backed but not benchmark-backed yet. The first task is
therefore to establish a runnable baseline and measurement harness before
changing production kernel behavior.

## Goals

- Reduce BiotSavart JAX compile pressure and peak memory without weakening
  existing C++ parity, chunked self-consistency, or analytical tests.
- Land the lowest-risk memory win first: bounded per-coil unit-field batching in
  the adapter path.
- Evaluate quadrature chunking defaults using measured GPU/CPU evidence rather
  than treating a default change as automatically safe.
- Add a fast-backend-only analytic `B+dB` kernel path if it passes the existing
  derivative parity ladder.
- Keep the byte-oriented `cpu_ordered` parity twin untouched unless a separate
  parity-preserving fix explicitly targets it.

## Non-Goals

- No float32 production downcast.
- No epsilon regularization of BiotSavart singularities.
- No removal of per-test `jax.clear_caches()` in `tests/conftest.py`; cache
  pressure should be handled by the runtime compilation-cache policy instead.
- No change to `biotsavart_cpu_ordered.py` while implementing the fast backend
  analytic derivative path.
- No broad rewrite of `d2B` or coil-current derivative APIs before the smaller
  memory fixes and parity gates are green.

## Current State

- `src/simsopt_jax/core/biotsavart.py:451-484` computes standalone `dB` through
  `jax.jacfwd(one_point)`, `d2B` through nested `jax.jacfwd`, and fused `B+dB`
  through `jax.linearize` plus a 3-vector basis `vmap`.
- `src/simsopt_jax/core/biotsavart.py:589-601` documents dense Hessian
  intermediates of roughly 226 MB for `P_chunk=512, C=16, Q=128` before
  quadrature reduction.
- `src/simsopt_jax_adapters/field/biotsavart_backend.py:194-225` evaluates
  per-coil unit-current fields with an unbounded `jax.vmap` over each coil group.
  The output is a Python list by public coil order, so group collective reduction
  is bypassed by design.
- `src/simsopt_jax/core/biotsavart.py:247-313` already has quadrature-block
  integration, with comments warning that single-block, exact two-block, and
  padded multi-block paths use different reduction trees.
- `src/simsopt_jax/backend/runtime.py:371-378` currently sets
  `jax_gpu_parity` quadrature blocks to `0` and `jax_gpu_fast` to `64`.
  `runtime.py:410-487` also autotunes fast/performance policies, including
  larger quadrature block sizes.
- `src/simsopt_jax/core/biotsavart.py:316-351` applies `jax.checkpoint` only
  inside active point chunking. There is no standalone `point_remat` knob.
- `src/simsopt_jax/core/biotsavart.py:357-363` computes `r_inv3` as
  `r_inv * _explicit_inv(r2)`. `src/simsopt_jax/core/biotsavart_cpu_ordered.py:45-131`
  uses the C++-ordered `r_inv * r_inv * r_inv` algebra and also contains the
  existing analytic `B+dB` formula.
- `tests/field/test_biotsavart_jax.py:626`, `:935`, and `:1171` provide C++
  parity and chunked self-consistency classes. The chunked class explicitly uses
  dense JAX as its oracle, not C++.
- `benchmarks/validation_ladder_contract.py:68-110` sets direct kernel
  tolerances at `rtol=1e-10, atol=1e-12`, first derivative tolerances at
  `rtol=1e-8, atol=1e-10`, and second derivative tolerances at
  `rtol=1e-6, atol=1e-8`.
- `tests/conftest.py:104-112` sets chunked-vs-dense acceptance tiers at
  `1e-12` relative with tight CPU/GPU absolute tolerances.
- Current ambient-shell validation is blocked before tests are collected:
  `ImportError: cannot import name 'Curve' from 'simsoptpp'`. The ambient
  `python` resolves through `/Users/suhjungdae/.local/bin/python` to Homebrew
  Python 3.14.3 and cannot import `jax`. No repo-local interpreter exists at
  `./.conda-env/bin/python` or `./.conda/jax/bin/python` in this checkout, so
  Phase 0 must provision or select a real Python 3.11 JAX+simsoptpp environment
  and bind it explicitly before any PASS claim.

## Risks and Assumptions

- The analytic `B+dB` rewrite is not byte-preserving. It changes the FP operation
  graph, so it is allowed only in the fast backend after derivative parity gates
  pass.
- Quadrature chunking changes reduction-tree shape. Any default change must be
  validated against chunked-vs-dense tests and C++ parity tests, with CPU and GPU
  lanes considered separately.
- The per-coil unit-field batching task is expected to be numerically safest
  because it changes mapping structure over independent coils and does not change
  reduction order inside a coil.
- Performance impact claims remain assumptions until measured on the intended
  CPU/GPU runtime. This plan does not mark an optimization complete based on
  static source inspection alone.

## Implementation Plan

### Phase 0 - Baseline and Environment Repair

- [ ] Set `PYTHON=/absolute/path/to/python` to a Python 3.11 environment that
      can import `jax` and exposes `simsoptpp.Curve`; do not use ambient
      `python` for validation.
- [ ] Run the validation-environment smoke in the Validation section before any
      collect-only, parity, or benchmark command.
- [ ] Record baseline runtime config for each tested lane:
      `native_cpu`, `jax_cpu_fast`, `jax_cpu_parity`, `jax_gpu_parity`, and
      `jax_gpu_fast`.
- [ ] Capture baseline compile counts and peak memory for:
      `biot_savart_B`, `biot_savart_dB_by_dX`, `biot_savart_B_and_dB`, and the
      per-coil unit-field derivative path.
- [ ] Run the no-change baseline gates listed in the Validation section before
      checking off any implementation phase.

### Phase 1 - Bound Per-Coil Unit-Field Batching

Tier: local implementation change, low API risk.

Files:

- `src/simsopt_jax_adapters/field/biotsavart_backend.py`
- `tests/field/test_biotsavart_jax.py`

Tasks:

- [ ] Replace the unbounded group-level `jax.vmap(evaluate_single)` at
      `biotsavart_backend.py:222` with a bounded mapping strategy over coils,
      using `lax.map(..., batch_size=...)` or an equivalent static batching
      helper that preserves public coil order.
- [ ] Keep `result_by_index` output ordering unchanged.
- [ ] Do not change the per-coil single-coil kernel call shape:
      `gamma[jnp.newaxis, ...]`, `gammadash[jnp.newaxis, ...]`,
      `unit_current`.
- [ ] Add a focused regression that compares batched and unbatched per-coil
      outputs for `dB_by_dcoilcurrents`, `d2B_by_dXdcoilcurrents`, and the
      `d3B` path if available in the current fixture.
- [ ] Measure peak memory on a larger coil group before and after the change.

Exit criteria:

- [ ] Existing C++ coil-current parity class passes.
- [ ] New batched-vs-unbatched regression passes at existing derivative-heavy
      tolerances.
- [ ] Measurement artifact reports peak-memory delta and compile count delta.

### Phase 2 - Quadrature Chunking Policy Evaluation

Tier: runtime policy change, reduction-order sensitive.

Files:

- `src/simsopt_jax/backend/runtime.py`
- `src/simsopt_jax/core/biotsavart.py`
- `tests/field/test_biotsavart_jax.py`

Tasks:

- [ ] Treat `jax_gpu_parity` and `jax_gpu_fast` separately. `jax_gpu_fast`
      already has quadrature block size `64`; `jax_gpu_parity` is the lane with
      default `0`.
- [ ] Benchmark dense, block size `32`, block size `64`, and block size `128`
      for representative point/coil/quadrature shapes.
- [ ] Run chunked self-consistency at every proposed block size.
- [ ] If changing a parity-lane default, document the exact reduction drift and
      keep it inside `tests/conftest.py` acceptance tiers without loosening
      tolerances.
- [ ] Keep environment overrides and autotuned fast policies as the SSOT for
      experiments; do not add a second tuning surface.

Exit criteria:

- [ ] CPU and GPU chunked-vs-dense gates pass for the proposed policy.
- [ ] C++ direct and derivative parity gates still pass.
- [ ] A before/after table justifies any default change with measured memory,
      compile time, and runtime.

### Phase 3 - Fast-Backend Analytic B+dB Kernel

Tier: production physics fast path, behavior-sensitive.

Files:

- `src/simsopt_jax/core/biotsavart.py`
- `src/simsopt_jax/core/biotsavart_cpu_ordered.py` as reference only
- `tests/field/test_biotsavart_jax.py`

Tasks:

- [ ] Implement a separate analytic `B+dB` fast-kernel path behind
      `_DiffMode.VALUE_AND_JACOBIAN`.
- [ ] Reuse the source algebra from `biotsavart_cpu_ordered.py:74-131` and the
      C++ reference formula, but do not route the fast backend through the
      `cpu_ordered` parity implementation.
- [ ] Preserve existing public return shape and axis convention:
      `dB[p, j, l] = d_j B_l(x_p)`.
- [ ] Keep standalone `dB` and `d2B` AD paths unchanged until the fused path is
      green and measured.
- [ ] Add an internal side-by-side test comparing the analytic fused path
      against the existing `linearize` fused path on deterministic fixtures.
- [ ] Re-run derivative-heavy C++ parity gates before enabling the path by
      default for fast mode.

Exit criteria:

- [ ] `B` parity remains inside direct-kernel tolerances.
- [ ] `dB` parity remains inside derivative-heavy first-derivative tolerances.
- [ ] Compile count and runtime improve on the Boozer/local-label hot path or the
      change stays disabled behind an internal experiment switch.

### Phase 4 - Remat and r_inv3 Micro-Experiments

Tier: optional measured experiments, ULP-sensitive for algebra changes.

Files:

- `src/simsopt_jax/core/biotsavart.py`

Tasks:

- [ ] Measure point chunking with and without `jax.checkpoint(chunk_kernel)` for
      small, medium, and large point chunks.
- [ ] If remat is not beneficial below a size threshold, replace the unconditional
      point-chunk remat with a size-derived policy. Do not add an externally
      visible knob unless a caller has to own the choice.
- [ ] Test `r_inv3 = r_inv * r_inv * r_inv` only in the fast backend. Treat it
      as ULP-changing and rerun direct parity before keeping it.

Exit criteria:

- [ ] Any remat policy change has compile-time, peak-memory, and runtime
      evidence.
- [ ] Any `r_inv3` algebra change passes direct-kernel C++ parity and chunked
      self-consistency without tolerance changes.

### Phase 5 - Matrix-Free d2B Contraction Feasibility

Tier: larger API/internal-structure change, deferred.

Files:

- `src/simsopt_jax/core/biotsavart.py`
- current consumers that immediately contract `d2B` with coil-geometry
  sensitivities

Tasks:

- [ ] Inventory all production consumers of `biot_savart_d2B_by_dXdX` and any
      coil-current derivative variants.
- [ ] Identify call sites that immediately contract dense Hessian output.
- [ ] Prototype an internal contracted helper without changing the public dense
      Hessian API.
- [ ] Compare against the dense result at derivative-heavy second-derivative
      tolerances.

Exit criteria:

- [ ] No public API change is required for the first implementation.
- [ ] Dense and contracted paths agree inside existing second-derivative
      tolerances.
- [ ] Peak-memory savings are large enough to justify the larger surface area.

## Validation

All commands below use an explicit interpreter. The local checkout inspected on
2026-06-24 does not contain `./.conda-env/bin/python` or
`./.conda/jax/bin/python`, and ambient `python` is Homebrew Python 3.14.3 without
JAX. Export `PYTHON` to the provisioned Python 3.11 JAX+simsoptpp interpreter
for the machine running the gates.

Validation-environment smoke:

```bash
PYTHONNOUSERSITE=1 "${PYTHON}" - <<'PY'
import sys
import jax
import simsoptpp

assert sys.version_info[:2] == (3, 11), sys.version
assert hasattr(simsoptpp, "Curve")
print(sys.executable)
print(sys.version.split()[0])
print(jax.__version__)
print(simsoptpp.__file__)
PY
```

Prerequisite collection gate:

```bash
PYTHONNOUSERSITE=1 "${PYTHON}" -m pytest --collect-only -q \
  tests/field/test_biotsavart_jax.py::TestBiotSavartJaxCppParity \
  tests/field/test_biotsavart_jax.py::TestBiotSavartJaxCppCoilCurrentParity \
  tests/field/test_biotsavart_jax.py::TestBiotSavartJaxChunkedSelfConsistency \
  tests/field/test_biotsavart_jax.py::TestBiotSavartJaxAnalytical
```

CPU focused gates:

```bash
PYTHONNOUSERSITE=1 JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu "${PYTHON}" -m pytest -q \
  tests/field/test_biotsavart_jax.py::TestBiotSavartJaxAnalytical \
  tests/field/test_biotsavart_jax.py::TestBiotSavartJaxCppParity \
  tests/field/test_biotsavart_jax.py::TestBiotSavartJaxCppCoilCurrentParity \
  tests/field/test_biotsavart_jax.py::TestBiotSavartJaxChunkedSelfConsistency
```

CPU ordered parity guard:

```bash
PYTHONNOUSERSITE=1 JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu "${PYTHON}" -m pytest -q \
  tests/field/test_biotsavart_jax_cpu_ordered.py
```

Static checks after any code edit:

```bash
PYTHONNOUSERSITE=1 "${PYTHON}" -m compileall -q \
  src/simsopt_jax/core/biotsavart.py \
  src/simsopt_jax/core/biotsavart_cpu_ordered.py \
  src/simsopt_jax_adapters/field/biotsavart_backend.py \
  src/simsopt_jax/backend/runtime.py
git diff --check
```

GPU parity gates should run under the repo's intended CUDA/JAX environment with
the runtime's nondeterministic GPU reduction exclusion enforced for strict or
parity modes. Do not substitute chunked JAX self-consistency for C++ parity when
certifying analytic derivative or algebra changes.

## Progress

- [x] Source audit reconciled into this implementation plan.
- [x] Existing broad JAX performance plan inspected and left unchanged because
      it excludes production BiotSavart kernels.
- [x] Local collection blocker reproduced:
      `ImportError: cannot import name 'Curve' from 'simsoptpp'`.
- [x] Default-shell JAX import blocker reproduced:
      `ModuleNotFoundError: No module named 'jax'`.
- [x] Repo-local interpreter paths checked: `./.conda-env/bin/python` and
      `./.conda/jax/bin/python` are absent in this checkout.
- [ ] Phase 0 baseline runnable.
- [ ] Phase 1 per-coil batching implemented.
- [ ] Phase 2 quadrature policy measured.
- [ ] Phase 3 analytic `B+dB` fast path implemented and gated.
- [ ] Phase 4 micro-experiments measured.
- [ ] Phase 5 matrix-free `d2B` feasibility decided.
