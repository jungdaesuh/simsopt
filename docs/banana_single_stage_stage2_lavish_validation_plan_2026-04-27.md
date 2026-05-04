# Banana Single-Stage and Stage 2 Lavish Validation Plan

Status: draft implementation plan as of 2026-04-27.

Current tree inspected at `dfd94f583521`. The working tree was dirty when this
plan was written; this document is intentionally additive and does not judge
those uncommitted changes.

## Context

The current JAX validation surface is already strong for the banana optimization
product path:

- Stage 2 and single-stage banana entrypoints both have CPU/JAX routing.
- Core kernels have direct CPU/C++ or CPU-reference parity coverage.
- GPU parity tests exist, but they are hardware-gated.
- The validation ladder already defines named tolerance lanes in
  `benchmarks/validation_ladder_contract.py`.
- Existing parity docs include `docs/jax_parity_manifest.md`, but that manifest
  is broad JAX coverage. It is not a dedicated, machine-checkable banana product
  readiness map.

The remaining problem is not "there are no tests." The problem is that the
banana product contract is spread across entrypoints, benchmarks, pytest files,
GPU artifact runs, and documented carve-outs. If we can spend lavishly, the
right move is to turn that scattered evidence into a first-class validation
system.

## Product Surface

This plan tracks only the files and features required for single-stage and
Stage 2 banana coil optimization. It does not attempt to mirror every upstream
`simsoptpp` API.

| Surface | Primary files | Validation class |
| --- | --- | --- |
| Stage 2 entrypoint and objective assembly | `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py` | CPU/JAX value, gradient, e2e, GPU, artifact |
| Single-stage entrypoint and outer ALM/L-BFGS flow | `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py` | CPU/JAX init, wrapper gradient, GPU, artifact |
| Shared banana contracts | `examples/single_stage_optimization/banana_opt/*.py`, `alm_utils.py`, `hardware_constraints.py`, `jax_host_boundary.py` | contract/unit/integration |
| JAX Biot-Savart field path | `src/simsopt/jax_core/biotsavart.py`, `src/simsopt/field/biotsavart_jax_backend.py` | direct CPU/C++ parity |
| Stage 2 flux and target objectives | `src/simsopt/objectives/fluxobjective_jax.py`, `src/simsopt/objectives/stage2_target_objective_jax.py` | CPU/JAX value and gradient parity |
| Integral BdotN kernel | `src/simsopt/objectives/integral_bdotn_jax.py` | direct `simsoptpp.integral_BdotN` parity |
| Boozer residual/surface/objective wrappers | `src/simsopt/geo/boozer_residual_jax.py`, `src/simsopt/geo/boozersurface_jax.py`, `src/simsopt/geo/surfaceobjectives_jax.py` | CPU/C++ parity, dense/PLU reference, operator-backed adjoint |
| JAX geometry/spec layer | `src/simsopt/jax_core/specs.py`, `src/simsopt/jax_core/curve_geometry.py`, `src/simsopt/jax_core/surface_rzfourier.py`, `src/simsopt/geo/surface_fourier_jax.py` | CPU/JAX geometry and contract coverage |
| Optimizer backend contract | `src/simsopt/geo/optimizer_jax.py`, `src/simsopt/geo/optimizer_jax_private/*` | target-lane routing, trace, memory, performance |

## Current Coverage Baseline

| Coverage area | Existing evidence |
| --- | --- |
| Biot-Savart JAX vs CPU/C++ | `tests/field/test_biotsavart_jax.py`, `tests/field/test_biotsavart_jax_parity.py` |
| Integral BdotN JAX vs `simsoptpp` | `tests/objectives/test_integral_bdotn_jax.py` |
| SquaredFlux / Stage 2 value-gradient parity | `tests/objectives/test_fluxobjective_jax_parity.py`, `tests/integration/test_stage2_jax.py`, `benchmarks/stage2_value_gradient_parity.py` |
| Stage 2 e2e parity | `benchmarks/stage2_e2e_comparison.py`, `tests/integration/test_stage2_jax.py` |
| Stage 2 mixed quadrature parity | `tests/integration/test_stage2_jax.py::TestMixedQuadratureParity` |
| CurveCWS banana path parity | `tests/integration/test_stage2_jax.py::TestCurveCWSFourierCPPParity` |
| Boozer residual and wrapper parity | `tests/integration/test_single_stage_jax_cpu_reference.py`, `tests/geo/test_boozersurface_jax.py` |
| Operator-backed adjoint, no silent dense fallback | `tests/integration/test_single_stage_jax_cpu_reference.py`, `tests/geo/test_boozersurface_jax.py` |
| Single-stage init / outer proof contract | `benchmarks/single_stage_init_parity.py`, `tests/geo/test_single_stage_example.py`, `tests/geo/test_single_stage_alm_integration.py` |
| GPU parity | hardware-gated tests in `tests/integration/test_single_stage_jax_cpu_reference.py`, `tests/geo/test_boozersurface_jax.py`, `tests/core/test_reductions.py` |

## Known Carve-Outs

- Hessians and second derivatives are validated by FD/Taylor/symmetry, not by a
  direct C++ second-derivative parity oracle.
- Ill-conditioned exact adjoints intentionally use residual and failure-category
  coverage rather than vector parity.
- Long L-BFGS / ALM trajectories are branch-sensitive. They should not require
  per-step identity, but they should emit enough trace data to explain the first
  accepted divergence.
- Mixed quadrature has Stage 2 value/field/gradient coverage. Exact Boozer
  mixed-quadrature per-coil-group inner-solve vector parity is still a luxury
  gap.
- GPU proof exists but requires actual CUDA hardware. CPU-only pytest runs
  cannot prove CUDA parity.
- There are no standalone native C++ tests in `tests/`. The current "C++ parity"
  tests are Python pytest/benchmark tests that call compiled `simsoptpp`.

## Target End State

The lavish validation system should answer these questions without manual
forensics:

- Which banana product feature is this?
- Which implementation files own it?
- Which CPU/C++ or CPU-reference oracle validates it?
- Which pytest or benchmark enforces it?
- Which tolerance lane applies?
- Does it require CUDA hardware?
- Is the proof same-state vector parity, whole-solve parity, FD/Taylor, or
  documented carve-out?
- Which JSON artifact proves the last GPU run?

## Implementation Plan

### P0: Banana Validation Manifest

- [ ] Add `benchmarks/banana_validation_manifest.py` as the SSOT for the banana
  product validation surface.
- [ ] Represent each feature as a typed record with:
  - `feature_id`
  - `stage`: `stage2`, `single_stage`, or `shared`
  - `implementation_paths`
  - `oracle_paths`
  - `pytest_paths`
  - `benchmark_paths`
  - `tolerance_lane`
  - `proof_type`
  - `requires_cuda`
  - `known_carve_out`
- [ ] Include all Stage 2 features:
  - Biot-Savart `B`, `A`, `dB`, VJP, grouped field evaluation.
  - Integral BdotN scalar kernel.
  - SquaredFlux value and gradient.
  - CurveCWS banana geometry and current path.
  - Mixed quadrature TF plus banana field path.
  - Stage 2 target objective and optimizer backend routing.
  - Stage 2 artifact envelope.
- [ ] Include all single-stage features:
  - Boozer residual scalar/vector/full penalty.
  - Boozer LS and exact surface solve.
  - Iotas and non-QS ratio wrappers.
  - Operator-backed adjoint state.
  - Single-stage runtime bundle.
  - ALM smoothing and target-lane optimizer routing.
  - Warm-start / Stage 2 seed artifact ingestion.
  - Hardware constraints and result metadata.
- [ ] Add `tests/validation/test_banana_validation_manifest.py`.
- [ ] Make the manifest test fail when a required feature has no test or
  benchmark pointer.
- [ ] Make the manifest test fail when a feature claims direct C++ parity but
  has no `simsoptpp` or CPU-reference oracle path.
- [ ] Make the manifest test fail when a CUDA feature has no hardware-gated
  proof lane.

### P1: Contract Hygiene

- [ ] Split `derivative_heavy` in `benchmarks/validation_ladder_contract.py`
  into narrower lanes:
  - `first_derivative_cpp_parity`
  - `vjp_cpp_parity`
  - `composed_jacobian_cpp_parity`
  - `second_derivative_fd_taylor_validation`
- [ ] Update tests and benchmarks to request the narrower lane names.
- [ ] Document that second derivatives are not direct C++ parity unless a new
  oracle lands.
- [ ] Add a manifest assertion that every tolerance lane used by the banana
  manifest exists in `validation_ladder_contract.py`.
- [ ] Add a manifest assertion that every known carve-out is explicit, not
  inferred from a missing test.

### P2: Promote Benchmarks to First-Class Validation Lanes

- [ ] Add pytest wrappers or marker-driven tests for:
  - `benchmarks/stage2_value_gradient_parity.py`
  - `benchmarks/stage2_e2e_comparison.py`
  - `benchmarks/single_stage_init_parity.py`
- [ ] Add markers:
  - `banana_tier1`
  - `banana_tier2`
  - `banana_tier3`
  - `banana_gpu`
  - `banana_perf`
- [ ] Ensure benchmark failures return pytest failures with the JSON artifact
  path in the assertion message.
- [ ] Standardize output JSON keys across all banana validation scripts.
- [ ] Add a JSON schema or typed validator for validation artifacts.
- [ ] Emit the exact command, git SHA, dirty-tree status, platform, device,
  x64 status, fixture hash, tolerance lane, and pass/fail reason.

### P3: Expand Luxury Parity Coverage

- [ ] Add a deterministic high-precision FD oracle for second derivatives around
  existing CPU/C++ kernels.
- [ ] Add Hessian validation records to the manifest only after the oracle is
  implemented.
- [ ] Add exact mixed-quadrature Boozer per-coil-group vector parity:
  - grouped TF contribution
  - grouped banana contribution
  - total field contribution
  - objective component contribution
- [ ] Add first-divergence trace parity for branch-sensitive optimizers:
  - initial same-state value and gradient
  - accepted/rejected step metadata
  - line-search status
  - trust-radius / penalty state
  - first divergence point
  - final state summary
- [ ] Add trace comparison rules that do not require per-step identity after a
  documented branch divergence.
- [ ] Add regression fixtures for explicit Stage 2 seed paths and warm-start
  donor run directories so they bypass derived-archive hardware seed validation
  by contract.

### P4: GPU CI and Runtime Budgets

- [ ] Add a real CUDA validation lane through self-hosted GPU CI or a Runpod
  CI job.
- [ ] Require these CUDA lanes before release sign-off:
  - Stage 2 value/gradient CUDA parity.
  - Stage 2 short e2e CUDA parity.
  - Single-stage init CUDA parity.
  - Exact well-conditioned adjoint CPU/GPU parity.
  - Reduction CPU/GPU cancellation-stress parity.
- [ ] Upload validation JSON artifacts from GPU jobs.
- [ ] Record device model, driver, CUDA runtime, JAX version, XLA flags, and
  peak memory.
- [ ] Add compile and runtime budgets:
  - XLA compile time.
  - first-call runtime.
  - steady-state iteration time.
  - peak host RSS.
  - peak GPU memory.
  - no accidental dense materialization on target lanes.
- [ ] Add a weekly stable-hardware performance ratchet.

### P5: Documentation and Release Gate

- [ ] Generate or maintain
  `docs/banana_single_stage_stage2_validation_matrix.md` from the manifest.
- [ ] Include one row per banana feature with proof type and carve-out status.
- [ ] Add a "release readiness" section:
  - required local CPU commands
  - required CUDA commands
  - required artifact files
  - allowed skips
  - disallowed skips
- [ ] Link the new matrix from `docs/jax_parity_manifest.md`.
- [ ] Link the matrix from `docs/source/jax_acceptance.rst` if it should appear
  in rendered docs.

## Proposed Validation Commands

CPU kernel and objective parity:

```bash
python -m pytest -q \
  tests/field/test_biotsavart_jax.py \
  tests/objectives/test_integral_bdotn_jax.py \
  tests/objectives/test_fluxobjective_jax_parity.py \
  tests/integration/test_stage2_jax.py::TestMixedQuadratureParity \
  tests/integration/test_stage2_jax.py::TestCurveCWSFourierCPPParity
```

Single-stage CPU-reference parity:

```bash
python -m pytest -q tests/integration/test_single_stage_jax_cpu_reference.py
```

Single-stage contract surface:

```bash
python -m pytest -q \
  tests/geo/test_single_stage_example.py \
  tests/geo/test_single_stage_alm_integration.py
```

GPU release proof, on CUDA hardware:

```bash
python benchmarks/stage2_value_gradient_parity.py \
  --platform cuda \
  --fixture real \
  --output-json .artifacts/parity/stage2_tier1_cuda.json

python benchmarks/stage2_e2e_comparison.py \
  --platform cuda \
  --optimizer-backend ondevice \
  --output-json .artifacts/parity/stage2_tier2_cuda.json

python benchmarks/single_stage_init_parity.py \
  --platform cuda \
  --optimizer-backend ondevice \
  --output-json .artifacts/parity/single_stage_tier3_cuda.json
```

Future manifest gate:

```bash
python -m pytest -q tests/validation/test_banana_validation_manifest.py
```

## Definition of Done

- [ ] Every single-stage and Stage 2 banana feature has a manifest row.
- [ ] Every manifest row points to at least one enforcement test or benchmark.
- [ ] Every direct C++ parity claim names the CPU/C++ oracle path.
- [ ] Every FD/Taylor-only claim is labeled as such.
- [ ] Every GPU claim is backed by a hardware-gated test or artifact-producing
  benchmark.
- [ ] Every known carve-out is explicit in the manifest and generated matrix.
- [ ] The validation matrix can be reviewed without searching the full repo.
- [ ] The release gate can be run from documented commands.

## Non-Goals

- Do not mirror every original upstream `simsoptpp` test unless it is needed by
  banana single-stage or Stage 2 optimization.
- Do not require optimizer iterate-by-iterate identity after a legitimate branch
  divergence.
- Do not claim direct C++ second-derivative parity until there is a real
  second-derivative oracle.
- Do not treat Runpod artifacts as equivalent to local pytest unless they are
  wired into an explicit pass/fail validation lane.

## Recommended First Patch

Start with P0 and P1 only:

- [ ] Add the manifest.
- [ ] Add the manifest integrity test.
- [ ] Split the derivative-heavy contract.
- [ ] Update the broad JAX parity doc to point to the banana-specific matrix.

That gives the largest immediate gain: the codebase will stop relying on manual
audit memory to know what banana parity is supposed to mean.
