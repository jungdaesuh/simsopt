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
- A local CPU validation recipe was later found for this checkout:
  `/Users/suhjungdae/code/columbia/simsopt-jax/.conda/jax-0.10.0/bin/python`
  with `python -S` and `PYTHONPATH` explicitly pointing at this checkout's
  `build/cp311-cp311-macosx_26_0_arm64`, `src`, `tests`, and that environment's
  `site-packages`. This avoids the sibling checkout's scikit-build editable
  redirect and loads this checkout's `simsopt` and `simsoptpp` artifacts.

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

- [x] Set `PYTHON=/absolute/path/to/python` to a Python 3.11 environment that
      can import `jax` and exposes `simsoptpp.Curve`; do not use ambient
      `python` for validation.
- [x] Run the validation-environment smoke in the Validation section before any
      collect-only, parity, or benchmark command.
- [x] Record baseline runtime config for each tested lane:
      `native_cpu`, `jax_cpu_fast`, `jax_cpu_parity`, `jax_gpu_parity`, and
      `jax_gpu_fast`.
- [x] Capture baseline compile counts and peak memory for:
      `biot_savart_B`, `biot_savart_dB_by_dX`, `biot_savart_B_and_dB`, and the
      per-coil unit-field derivative path.
- [x] Run the no-change baseline gates listed in the Validation section before
      checking off any implementation phase.

### Phase 1 - Bound Per-Coil Unit-Field Batching

Tier: local implementation change, low API risk.

Files:

- `src/simsopt_jax_adapters/field/biotsavart_backend.py`
- `tests/field/test_biotsavart_jax.py`

Tasks:

- [x] Replace the unbounded group-level `jax.vmap(evaluate_single)` at
      `biotsavart_backend.py:222` with a bounded mapping strategy over coils,
      using `lax.map(..., batch_size=...)` or an equivalent static batching
      helper that preserves public coil order.
- [x] Keep `result_by_index` output ordering unchanged.
- [x] Do not change the per-coil single-coil kernel call shape:
      `gamma[jnp.newaxis, ...]`, `gammadash[jnp.newaxis, ...]`,
      `unit_current`.
- [x] Add a focused regression that compares batched and unbatched per-coil
      outputs for `dB_by_dcoilcurrents`, `d2B_by_dXdcoilcurrents`, and the
      `d3B` path if available in the current fixture.
- [x] Measure peak memory on a larger coil group before and after the change.

Exit criteria:

- [x] Existing C++ coil-current parity class passes.
- [x] New batched-vs-unbatched regression passes at existing derivative-heavy
      tolerances.
- [x] Measurement artifact reports peak-memory delta and compile count delta.

### Phase 2 - Quadrature Chunking Policy Evaluation

Tier: runtime policy change, reduction-order sensitive.

Files:

- `src/simsopt_jax/backend/runtime.py`
- `src/simsopt_jax/core/biotsavart.py`
- `tests/field/test_biotsavart_jax.py`

Tasks:

- [x] Treat `jax_gpu_parity` and `jax_gpu_fast` separately. `jax_gpu_fast`
      already has quadrature block size `64`; `jax_gpu_parity` is the lane with
      default `0`.
- [x] Benchmark dense, block size `32`, block size `64`, and block size `128`
      for representative point/coil/quadrature shapes.
- [x] Run chunked self-consistency at every proposed block size.
- [x] If changing a parity-lane default, document the exact reduction drift and
      keep it inside `tests/conftest.py` acceptance tiers without loosening
      tolerances. No parity-lane default change is promoted by the local CPU
      evidence.
- [x] Keep environment overrides and autotuned fast policies as the SSOT for
      experiments; do not add a second tuning surface.

Exit criteria:

- [x] CPU chunked-vs-dense gates pass for the measured no-default-change policy.
- [x] GPU chunked-vs-dense gates passed on Perlmutter debug GPU job `54985084`
      with one visible A100 device. The gate produced both `jax_gpu_parity` and
      `jax_gpu_fast` quadrature JSON artifacts without loosening tolerances.
- [x] C++ direct and derivative parity gates still pass on the local CPU lane.
- [x] A before/after table justifies not changing a default with only local CPU
      evidence: memory improves, but runtime is mixed and GPU evidence is
      unavailable.

### Phase 3 - Fast-Backend Analytic B+dB Kernel

Tier: production physics fast path, behavior-sensitive.

Files:

- `src/simsopt_jax/core/biotsavart.py`
- `src/simsopt_jax/core/biotsavart_cpu_ordered.py` as reference only
- `tests/field/test_biotsavart_jax.py`

Tasks:

- [x] Implement a separate analytic `B+dB` fast-kernel path behind
      `_DiffMode.VALUE_AND_JACOBIAN`.
- [x] Reuse the source algebra from `biotsavart_cpu_ordered.py:74-131` and the
      C++ reference formula, but do not route the fast backend through the
      `cpu_ordered` parity implementation.
- [x] Preserve existing public return shape and axis convention:
      `dB[p, j, l] = d_j B_l(x_p)`.
- [x] Keep standalone `dB` and `d2B` AD paths unchanged until the fused path is
      green and measured.
- [x] Add an internal side-by-side test comparing the analytic fused path
      against the existing `linearize` fused path on deterministic fixtures.
- [x] Re-run derivative-heavy C++ parity gates before enabling the path by
      default for fast mode.

Exit criteria:

- [x] `B` parity remains inside direct-kernel tolerances.
- [x] `dB` parity remains inside derivative-heavy first-derivative tolerances.
- [x] Compile count and runtime improve on the measured local fused `B+dB` probe.

### Phase 4 - Remat and r_inv3 Micro-Experiments

Tier: optional measured experiments, ULP-sensitive for algebra changes.

Files:

- `src/simsopt_jax/core/biotsavart.py`

Tasks:

- [x] Measure point chunking with and without `jax.checkpoint(chunk_kernel)` for
      small, medium, and large point chunks.
- [x] If remat is not beneficial below a size threshold, replace the unconditional
      point-chunk remat with a size-derived policy. Do not add an externally
      visible knob unless a caller has to own the choice. The measurement showed
      remat was helpful or neutral, so no production policy change was made.
- [x] Test `r_inv3 = r_inv * r_inv * r_inv` only in the fast backend. Treat it
      as ULP-changing and rerun direct parity before keeping it.

Exit criteria:

- [x] Any remat policy change has compile-time, peak-memory, and runtime
      evidence. No policy change was justified by the evidence.
- [x] Any `r_inv3` algebra change passes direct-kernel C++ parity and chunked
      self-consistency without tolerance changes. No algebra change was kept.

### Phase 5 - Matrix-Free d2B Contraction Feasibility

Tier: private internal helper; public dense-Hessian API unchanged.

Files:

- `src/simsopt_jax/core/biotsavart.py`
- current consumers that immediately contract `d2B` with coil-geometry
  sensitivities

Tasks:

- [x] Inventory all production consumers of `biot_savart_d2B_by_dXdX` and any
      coil-current derivative variants.
- [x] Identify call sites that immediately contract dense Hessian output.
- [x] Prototype an internal contracted helper without changing the public dense
      Hessian API: `_biot_savart_d2B_by_dXdX_contract(...)` contracts
      point-aligned left/right directions through a private JVP-based kernel and
      is not exported from `__all__`.
- [x] Compare against the dense result at derivative-heavy second-derivative
      tolerances with
      `TestBiotSavartJaxChunkedSelfConsistency::test_d2B_contracted_helper_matches_dense_hessian_contraction`.

Exit criteria:

- [x] No public API change is required for the first implementation.
- [x] Dense and contracted paths agree inside existing second-derivative
      tolerances for dense and point-chunked tuning.
- [x] Peak-memory savings are large enough to justify the private helper surface
      for low-rank contractions: the local CPU probe measured a 414064-byte
      XLA temp reduction, 20.6% versus dense Hessian materialization, at
      64 points / 12 coils / 48 quadrature samples / 2 left directions /
      1 right direction.

Decision:

- Implement the matrix-free `d2B` contraction as a private helper, with the
  dense public/API-oracle consumers intentionally unchanged. The live inventory
  remains: `BiotSavartJAX.d2B_by_dXdX()` returns a dense grouped Hessian in
  `src/simsopt_jax_adapters/field/biotsavart_backend.py`, `BoozerSurface`
  reshapes dense `d2B_by_dXdX` and passes it to the C++ oracle
  `sopp.boozer_residual_ds2`, and `src/simsopt/geo/surfaceobjectives.py`
  immediately contracts dense `d2B_by_dXdX` with `dx_dc`. This phase provides
  the validated low-rank contraction primitive without migrating those public
  or C++-oracle contracts.

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
  benchmarks/per_coil_unit_field_vmap_probe.py \
  benchmarks/biotsavart_quadrature_chunking_probe.py \
  benchmarks/biotsavart_phase4_micro_probe.py \
  benchmarks/biotsavart_phase5_d2B_contract_probe.py \
  src/simsopt_jax/core/biotsavart.py \
  src/simsopt_jax/core/biotsavart_cpu_ordered.py \
  src/simsopt_jax_adapters/field/biotsavart_backend.py \
  src/simsopt_jax/backend/runtime.py \
  tests/field/test_biotsavart_jax.py
git diff --check -- \
  benchmarks/per_coil_unit_field_vmap_probe.py \
  benchmarks/biotsavart_quadrature_chunking_probe.py \
  benchmarks/biotsavart_phase4_micro_probe.py \
  benchmarks/biotsavart_phase5_d2B_contract_probe.py \
  docs/biotsavart_jax_perf_memory_optimization_implementation_plan.md \
  src/simsopt_jax/core/biotsavart.py \
  src/simsopt_jax_adapters/field/biotsavart_backend.py \
  tests/field/test_biotsavart_jax.py
```

GPU parity gates should run under the repo's intended CUDA/JAX environment with
the runtime's nondeterministic GPU reduction exclusion enforced for strict or
parity modes. Do not substitute chunked JAX self-consistency for C++ parity when
certifying analytic derivative or algebra changes.

Focused BiotSavart CUDA gate:

```bash
PYTHON_BIN=/absolute/path/to/python \
RESULTS_DIR=.artifacts/biotsavart_gpu_gate \
bash scripts/run_biotsavart_gpu_gate.sh
```

This wrapper fails early unless `nvidia-smi` is available, the selected Python is
3.11, `simsoptpp.Curve` imports, and JAX selects a CUDA/GPU backend. It then runs
the focused BiotSavart pytest parity/self-consistency/analytical gates plus the
quadrature block-size probe for both `jax_gpu_parity` and `jax_gpu_fast`.
On Perlmutter debug GPU nodes, make only one GPU visible for this focused
single-GPU gate, for example with `CUDA_VISIBLE_DEVICES=0`; Slurm can allocate a
full four-GPU node even when the script requests `--gpus=1`, and the explicit
point-sharding regression is intentionally shape-sensitive to the visible device
mesh.

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
- [x] Phase 0 CPU validation recipe found and smoke-tested against this checkout:
      Python 3.11.15, JAX 0.10.0, current `src/simsopt`, and current
      `build/cp311-cp311-macosx_26_0_arm64/simsoptpp`.
- [x] Phase 0 collect gate runnable: 47 BiotSavart tests collected after the
      Phase 1 and Phase 3 regressions were added.
- [x] Phase 0 CPU focused baseline passed before implementation:
      42 passed, 3 skipped.
- [x] Phase 0 CPU ordered parity guard passed before implementation:
      4 passed.
- [x] Phase 1 bounded per-coil unit-field mapping implemented with
      `jax.lax.map(batch_size=get_field_kernel_tuning().coil_chunk_size)`.
- [x] Phase 1 functional regression added for bounded batch size versus the old
      unbounded `jax.vmap` reference on the `dB_by_dcoilcurrents`,
      `d2B_by_dXdcoilcurrents`, and `d3B_by_dXdXdcoilcurrents` kernels.
- [x] Phase 1/3 post-change CPU focused gate passed after the analytic fused
      path landed: 52 passed, 3 skipped.
- [x] Phase 1 peak-memory and compile-count measurement artifact captured:
      `.artifacts/biotsavart_phase1_per_coil_batching_20260624.json`.
      Shape: 64 coils, 32 points, 32 quadrature samples, batch size 16,
      kernels `B`, `dB`, `d2B`. Compile-log delta was 0 for every kernel.
      XLA temp-size deltas were -786392 B (`B`), -3784656 B (`dB`), and
      -8785848 B (`d2B`). Output max-absolute diffs were 7.94e-23, 9.26e-23,
      and 0.0 respectively. CPU RSS high-water was mixed at this small local
      shape (+3.84 MB, -15.72 MB, +5.03 MB), and post-compile CPU runtime was
      slower for the bounded `d2B` microcase, so this artifact certifies the
      compiler temp-memory reduction rather than a blanket CPU walltime win.
- [x] Phase 2 CPU quadrature block-size matrix captured:
      `.artifacts/biotsavart_phase2_quadrature_chunking_20260624.json`.
      Shape: 32 coils, 128 quadrature samples, 64 points, coil chunk size 16,
      blocks 0/32/64/128, kernels `B`, `dB`, `B_and_dB`. Block size 32 gave
      the clearest XLA temp-size reduction on CPU (`B`: 8388680 -> 770408 B,
      `dB`: 27263048 -> 2966064 B, `B_and_dB`: 35749960 -> 24585008 B), with
      max absolute drift versus dense <= 5.56e-17. Runtime was mixed and
      sometimes slower, and no GPU lane was available locally, so no
      `jax_gpu_parity` default change is promoted from this evidence.
- [x] Phase 2 GPU parity/fast confirmation completed on Perlmutter debug GPU
      job `54985084` (`gpu_debug`, `m4680_g`, node `nid001301`,
      `CUDA_VISIBLE_DEVICES=0`). The gate ran against the staged contents of
      local source commit `6dc71710ab5825ee1200fbd6f6763abb24ad4384` with
      Python 3.11.15 and JAX 0.10.0. Focused pytest passed
      `49 passed, 2 skipped` in 168.25 s. The quadrature probe produced
      `.artifacts/biotsavart_gpu_gate_perlmutter_54985084/biotsavart_quadrature_chunking_jax_gpu_parity.json`
      and
      `.artifacts/biotsavart_gpu_gate_perlmutter_54985084/biotsavart_quadrature_chunking_jax_gpu_fast.json`.
      For both GPU backend modes, block size 32 gave the lowest XLA temp size
      for `B`, `dB`, and `B_and_dB`, and max absolute drift versus dense was
      <= 1.11e-16.
- [x] Phase 3 analytic `B+dB` fast path implemented for
      `_DiffMode.VALUE_AND_JACOBIAN`; standalone `dB` and `d2B` AD paths were
      left unchanged. Added an analytic-vs-old-linearized regression across
      dense, coil-chunked, quadrature-chunked, and point-chunked settings.
- [x] Phase 3 CPU measurement artifact captured:
      `.artifacts/biotsavart_phase3_analytic_B_and_dB_20260624.json`.
      Against the pre-change Phase 2 artifact on the same 32-coil/128-quad/
      64-point shape, `B_and_dB` post-compile median improved at every tested
      quadrature block (dense 0.00175858 -> 0.00109950 s; block 32
      0.00200546 -> 0.00141354 s; block 64 0.00173471 -> 0.00093800 s;
      block 128 0.00198108 -> 0.00136333 s). XLA temp size improved for every
      block, most strongly at block 32 (24585008 -> 1966384 B). Max absolute
      drift versus dense in the post-change artifact was <= 5.56e-17.
- [x] Phase 3 validation passed locally: full
      `tests/field/test_biotsavart_jax.py` CPU file `52 passed, 3 skipped`;
      CPU-ordered parity guard `tests/field/test_biotsavart_jax_cpu_ordered.py`
      `4 passed`.
- [x] Phase 4 micro-experiments measured:
      `.artifacts/biotsavart_phase4_micro_probe_20260624.json`. Reverse-mode
      point-gradient probe over `sum(B) + 0.01 sum(dB)` showed remat is helpful
      or neutral on the tested CPU shape (16 coils, 64 quadrature samples,
      256 points): point chunk 16 temp 4122848 B with remat vs 60130664 B
      without; point chunk 64 temp 16311392 B vs 57771368 B; point chunk 128
      temp identical at 45091848 B. No remat policy change is justified.
      The local `r_inv3 = r_inv*r_inv*r_inv` experiment reduced temp size by
      2097152 B but was slightly slower post-compile and ULP-changing
      (max abs output drift 5.56e-17), so production algebra was left unchanged.
- [x] Phase 5 matrix-free `d2B` contraction helper implemented privately in
      `src/simsopt_jax/core/biotsavart.py` as
      `_biot_savart_d2B_by_dXdX_contract(...)`. The helper keeps the dense
      public Hessian API unchanged, is cache-keyed on the existing
      coil/quadrature/point tuning tuple, and is cleared by
      `invalidate_kernel_cache()`.
- [x] Phase 5 dense-oracle regression added and passed:
      `TestBiotSavartJaxChunkedSelfConsistency::test_d2B_contracted_helper_matches_dense_hessian_contraction`.
      It compares the helper against
      `einsum("pjkl,paj,pbk->pabl", dense_d2B, left, right)` under dense and
      point-chunked tuning, using the validation-ladder second-derivative
      tolerances.
- [x] Phase 5 measurement artifact captured:
      `.artifacts/biotsavart_phase5_d2B_contract_probe_20260624.json`. Shape:
      64 points, 12 coils, 48 quadrature samples, 2 left directions, 1 right
      direction, coil chunk 8, quadrature block 24, point chunk 16. Dense
      Hessian compiled temp size was 2014336 B; contracted helper compiled temp
      size was 1600272 B; savings were 414064 B (20.6%). The contracted helper
      also compiled 0.043 s faster and ran 0.0013 s faster post-compile on this
      CPU microcase. Dense-contraction delta was max abs 8.88e-16 and max rel
      1.93e-12.
- [x] Final local validation rerun completed on 2026-06-24 with the explicit
      Python 3.11/JAX 0.10.0 CPU environment: validation smoke passed and showed
      `[CpuDevice(id=0)]`; collect-only found 47 tests before Phase 5 and the
      final focused BiotSavart suite passed `57 passed, 3 skipped` across
      `tests/field/test_biotsavart_jax.py` and
      `tests/field/test_biotsavart_jax_cpu_ordered.py`; scoped
      `git diff --check` and `compileall` passed.
- [x] GPU parity/fast confirmation is no longer external. The successful
      Perlmutter debug run is archived locally under
      `.artifacts/biotsavart_gpu_gate_perlmutter_54985084/`. A first Perlmutter
      attempt (`54984765`) intentionally remains non-promotional evidence: it
      exposed all four A100s to JAX and failed only the explicit point-sharding
      regression with a 3-coil/4-device indivisibility error. The promotional
      rerun constrained the visible CUDA mesh to one GPU and passed. This GPU
      gate covers the public BiotSavart paths in commit `6dc71710`; the private
      Phase 5 helper added after that gate is locally CPU-validated and has no
      production caller.
- [x] Focused BiotSavart CUDA gate wrapper added at
      `scripts/run_biotsavart_gpu_gate.sh` and syntax-checked locally with
      `bash -n`. It composes the focused pytest gates and CUDA quadrature probe
      into one command for a provisioned CUDA host; it was not executed locally
      because no CUDA JAX backend is visible.
