# Release-Grade C++ Behavior Preservation Gate for JAX - 2026-05-02

## Goal

JAX is releasable only if it preserves existing SIMSOPT CPU/C++ behavior.

The required trust chain is:

```text
existing SIMSOPT CPU/C++ behavior
  -> JAX CPU matches
  -> JAX GPU matches
  -> JAX CPU and JAX GPU match each other
```

JAX CPU vs GPU agreement alone is not a release proof. It only proves that two
JAX lanes agree. If both JAX lanes share the same drift from SIMSOPT CPU/C++,
that still fails the release contract.

## Current Context

This plan extends the existing trajectory-specific plan:

- `docs/cpu_cpp_jax_cpu_full_trajectory_parity_plan_2026-04-28.md`
- `docs/single_stage_parity_revalidation_2026-04-28.md`

Validated current-tree facts:

- `benchmarks/single_stage_parity_matrix.py` already has lane labels for
  `cpu_scipy`, `cpu_cpp_trace`, `jax_cpu`, and `h100_gpu`.
- `benchmarks/single_stage_parity_matrix.py` currently consumes
  `jax_cpu_vs_h100_value_grad` for same-state value/gradient parity.
- The matrix does not yet consume a production fixed-state `cpp_cpu` artifact.
- The matrix does not yet emit direct `cpp_cpu_vs_jax_cpu` or
  `cpp_cpu_vs_jax_gpu` fixed-state comparisons.
- `benchmarks/single_stage_init_parity.py` pins the JAX target lane to the
  current `ondevice` optimizer backend.
- CPU/reference parity defaults to SciPy `L-BFGS-B` over the legacy `JF.x`
  Optimizable vector.
- The JAX target lane optimizes the `bs.x` coordinate contract.
- Existing tests assert the selector behavior for `JF.x` vs `bs.x`, but they do
  not prove the full coordinate/gradient projection contract.
- GPU runtime provenance helpers already record JAX, jaxlib, devices, x64,
  CUDA/XLA env, NVIDIA GPU facts for CUDA runs, sharding, and memory metadata.

## Existing Evidence To Preserve

- JAX CPU vs H100 same-state value/gradient parity has strong evidence for
  matched inputs.
- JAX CPU vs H100 matched full trajectory parity has evidence when the exact
  H100 runtime seed spec and equilibrium file are reused.
- CPU/C++ vs JAX CPU same-seed metrics still show small derived-metric drift in
  the existing report.
- The CPU/C++ full-trajectory parity proof is still incomplete because the
  current CPU/C++ artifact is not yet the fixed-state production oracle and not
  a matching full optimizer run under the release contract.

## Oracle Definitions

### Fixed-State CPU/C++ Kernel Oracle

This is the first release gate.

Properties:

- CPU backend.
- Legacy Optimizable graph.
- Existing SIMSOPT CPU/C++ kernels where SIMSOPT currently uses them.
- No outer optimizer steps.
- Same physical state and same active variables as the JAX lanes.
- Emits objective, objective components, full gradient, field metrics, Boozer
  residual/operator checks, hashes, versions, and runtime metadata.

Purpose:

- Prove the physics/operator boundary before optimizer path sensitivity enters.

### CPU/SciPy Full-Run Reference

This is the public behavior reference for optimizer-equipped CPU behavior.

Properties:

- CPU backend.
- Legacy Optimizable graph.
- Optimizer vector is `JF.x`.
- Outer optimizer is SciPy `L-BFGS-B`.
- Starts from the same init-only seed as the JAX lanes.

Purpose:

- Compare public optimizer behavior: initial values, accepted state where
  visible, termination, final objective, final gradient norm, final mapped
  state, final physics metrics, constraints, runtime, and memory.

### CPU Host-Core Trace Diagnostic

This is not the CPU/SciPy oracle.

Properties:

- Uses the shared host L-BFGS/Wolfe core.
- Can emit target-style `optimizer_state_trace`.
- Bypasses SciPy.

Purpose:

- Debug target-lane optimizer control and compare trace mechanics.
- Must stay labeled as diagnostic and excluded from CPU/SciPy release claims.

## Non-Goals

- [ ] Do not replace the CPU/SciPy reference lane with the host-core diagnostic.
- [ ] Do not project the CPU/SciPy reference lane into `bs.x`.
- [ ] Do not make CPU/SciPy use JAX optimizer internals.
- [ ] Do not treat `lbfgs-trace` as the CPU/C++ oracle.
- [ ] Do not loosen tolerances to hide drift.
- [ ] Do not infer C++ behavior preservation from JAX CPU vs GPU agreement.
- [ ] Do not claim SciPy Wolfe-internal trace parity in this release gate.

## Release Gate Buckets

The top-level release result should be:

```json
{
  "release_gate_passed": false,
  "buckets": {
    "fixed_state_physics_parity": {"status": "blocked"},
    "coordinate_mapping_parity": {"status": "blocked"},
    "optimizer_public_behavior_parity": {"status": "blocked"},
    "final_metric_envelope": {"status": "blocked"},
    "performance_memory_report": {"status": "blocked"}
  }
}
```

Bucket meanings:

- `fixed_state_physics_parity`: Direct same-state CPU/C++ vs JAX CPU/GPU
  value, gradient, components, and operator checks.
- `coordinate_mapping_parity`: `JF.x` and `bs.x` active-variable, frozen-mask,
  state reconstruction, and gradient projection proof.
- `optimizer_public_behavior_parity`: CPU/SciPy full-run public behavior
  compared to JAX CPU and JAX GPU.
- `final_metric_envelope`: Final objectives, constraints, physical metrics, and
  mapped final state are inside the accepted envelope.
- `performance_memory_report`: Runtime, compile time, memory high-water, device,
  driver, CUDA/XLA/JAX metadata are recorded and satisfy the checked-in memory
  and performance budgets.

## Tolerance Policy

Use the existing tolerance lanes from `benchmarks/validation_ladder_contract.py`.

- [ ] Use `direct_kernel` only for kernel-level forward quantities with direct
  C++ oracle coverage, such as Biot-Savart `B`, surface `gamma`,
  `integral_BdotN`, and raw Boozer residual vectors/norms driven by identical
  inputs.
- [ ] Use `gpu_runtime` same-state forward/gradient tolerances for assembled
  single-stage objective scalars and full optimizer-basis gradients. Do not use
  `direct_kernel` for the composed objective or composed full gradient.
- [ ] Use `derivative_heavy` for first-derivative kernels with direct C++
  oracle coverage, such as `dB/dX`, Biot-Savart VJP, surface coefficient
  Jacobians, and the composed Boozer residual Jacobian/JVP/VJP.
- [ ] Use `exact_well_conditioned_adjoint` for well-conditioned Boozer adjoint
  vector parity.
- [ ] Use `exact_ill_conditioned_adjoint` for ill-conditioned exact fixtures,
  residual/failure-only checks, and no vector parity claim.
- [ ] Do not introduce one global tolerance for all quantities.

## Reused Existing Contracts

Do not parallel-implement proof semantics that already exist in the validation
ladder contract module.

- [ ] Reuse
  `single_stage_proof_contract(TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG)` as the SSOT
  for target-lane single-stage outer-loop requirements, including
  `required_outer_optimizer_method="lbfgs-ondevice"` and required final metric
  keys.
- [ ] Reuse `gpu_proof_parity_contract("single_stage")` as the SSOT for
  single-stage GPU proof value/gradient tolerance wiring.
- [ ] Reuse `grouped_adjoint_memory_budget(...)` plus
  `evaluate_grouped_adjoint_memory_budget(...)` for grouped-adjoint memory
  budgets. For `real_single_stage_init`, the current checked-in ceilings are
  8192 MB peak RSS on CPU/CUDA and 12288 MB peak GPU memory on CUDA.
- [ ] Reuse `tier5_performance_budget("stable_hardware_weekly")` for checked-in
  performance floors from the checked-in warm/cold CPU baseline artifacts.
- [ ] Bucket evaluators in `benchmarks/single_stage_parity_matrix.py` must call
  these helpers instead of duplicating constants.

## Workstream A: Fixed-State C++/JAX Artifact Producer

Target file:

- [ ] Add `benchmarks/single_stage_cpp_jax_state_parity.py`.

Inputs:

- [ ] Existing Stage 2 seed Biot-Savart JSON.
- [ ] Equilibrium file path.
- [ ] Resolution: `mpol`, `ntor`, `nphi`, `ntheta`.
- [ ] Canonical runtime seed spec.
- [ ] Platform selector: exactly `cpu` or `cuda`; every other value is rejected
  in release proof commands.
- [ ] Output JSON path.

Lanes:

- [ ] `cpp_cpu`: legacy CPU/C++ fixed-state evaluator.
- [ ] `jax_cpu`: JAX target evaluator pinned to CPU.
- [ ] `jax_gpu`: JAX target evaluator pinned to CUDA.

Artifact schema:

- [ ] `schema_version`.
- [ ] `provenance`.
- [ ] `inputs`.
- [ ] `lanes.cpp_cpu`.
- [ ] `lanes.jax_cpu`.
- [ ] `lanes.jax_gpu`.
- [ ] `comparisons.cpp_cpu_vs_jax_cpu`.
- [ ] `comparisons.cpp_cpu_vs_jax_gpu`.
- [ ] `comparisons.jax_cpu_vs_jax_gpu`.
- [ ] `passed`.
- [ ] `failures`.

Required assembled lane outputs:

- [ ] total objective.
- [ ] objective components.
- [ ] full optimizer-basis gradient.
- [ ] gradient infinity norm.
- [ ] gradient L2 norm.
- [ ] field error.
- [ ] iota.
- [ ] volume.
- [ ] max curvature.
- [ ] coil-coil minimum distance.
- [ ] coil-plasma minimum distance.
- [ ] plasma-vessel minimum distance.
- [ ] self-intersection status and availability.
- [ ] hardware-constraint status for every configured constraint.

Required kernel/operator lane outputs:

- [ ] Biot-Savart `B` on the shared quadrature points.
- [ ] Surface `gamma`.
- [ ] `integral_BdotN`.
- [ ] Raw Boozer residual vector.
- [ ] Boozer residual norm.
- [ ] Boozer residual max norm.
- [ ] First-derivative kernel samples needed by the tolerance contract.
- [ ] Boozer residual Jacobian metadata.
- [ ] Boozer JVP / linearized residual action.
- [ ] Boozer VJP / adjoint transpose-matvec projection.
- [ ] Boozer adjoint solve status and residual.

Required provenance:

- [ ] repo SHA.
- [ ] dirty-worktree status summary.
- [ ] Python version.
- [ ] SIMSOPT import path.
- [ ] JAX version.
- [ ] jaxlib version.
- [ ] backend and devices.
- [ ] `jax_enable_x64`.
- [ ] XLA flags.
- [ ] CUDA env vars.
- [ ] CUDA visible devices.
- [ ] NVIDIA GPU name, driver, and memory total for CUDA lanes.
- [ ] compilation-cache policy and cache dir.
- [ ] sharding metadata.
- [ ] distributed runtime metadata.
- [ ] peak RSS.
- [ ] GPU memory high-water for CUDA lanes.
- [ ] compile time and run time per lane.

Required hashes:

- [ ] Stage 2 seed hash.
- [ ] Biot-Savart JSON hash.
- [ ] runtime seed spec hash.
- [ ] equilibrium file hash.
- [ ] objective configuration hash.
- [ ] active DOF mask hash.
- [ ] fixed/frozen DOF mask hash.

Hash equality gate:

- [ ] Fail producer-side before numerical comparison if `equilibrium_hash`,
  `runtime_seed_spec_hash`, `biot_savart_json_hash`,
  `objective_configuration_hash`, or `active_dof_mask_hash` differs across
  `cpp_cpu`, `jax_cpu`, and `jax_gpu`.
- [ ] Record the mismatched hash names and lane values as artifact failures, not
  as floating-point drift.

CPU/C++ evaluator todos:

- [ ] Reuse the exact legacy single-stage CPU setup.
- [ ] Ensure the evaluator writes no optimizer step.
- [ ] Run `BoozerSurface.run_code(...)` at the restored same state.
- [ ] Evaluate the legacy objective total and components.
- [ ] Evaluate `JF.dJ()` in the legacy `JF.x` basis.
- [ ] Capture Boozer runtime adjoint state through the public runtime seam.
- [ ] For Boozer JVP, materialize the CPU Boozer residual dense Jacobian in
  analytic mode and compute `J @ direction` for the same direction used by the
  JAX lane.
- [ ] For Boozer VJP, compute `J.T @ cotangent` for the same cotangent used by
  the JAX lane.
- [ ] For well-conditioned adjoint vector parity, use a fixture where the dense
  transpose solve is well-conditioned and compare under
  `exact_well_conditioned_adjoint`.
- [ ] For ill-conditioned real-plasma fixtures, record solve residual/status
  under `exact_ill_conditioned_adjoint` and do not claim adjoint vector parity.
- [ ] Emit the full gradient before any projection into `bs.x`.
- [ ] Emit projection metadata needed by Workstream B.

JAX CPU evaluator todos:

- [ ] Reuse the target-lane runtime bundle construction.
- [ ] Pin platform to CPU before JAX import.
- [ ] Require `jax_enable_x64`.
- [ ] Evaluate the same physical state without optimizer steps.
- [ ] Emit total objective, components, and gradient in `bs.x` basis.
- [ ] Emit Boozer residual/JVP/VJP checks using the same operator seam.

JAX GPU evaluator todos:

- [ ] Pin platform to CUDA before JAX import.
- [ ] Require CUDA device execution.
- [ ] Require `jax_enable_x64`.
- [ ] Enable deterministic GPU reductions for proof runs by passing
  `deterministic_gpu_reductions=True` through the subprocess environment builder.
- [ ] Assert provenance includes `--xla_gpu_exclude_nondeterministic_ops=true` in
  `XLA_FLAGS`; missing deterministic flags fail the GPU proof.
- [ ] Record compile time separately from execution time.
- [ ] Record GPU memory high-water.
- [ ] Emit the same schema as `jax_cpu`.

Comparison todos:

- [ ] Compare kernel-level CPU/C++ forward quantities to JAX CPU/GPU with
  `direct_kernel`.
- [ ] Compare assembled objective scalar and full optimizer-basis gradients with
  `gpu_runtime` same-state tolerances.
- [ ] Compare JAX CPU to JAX GPU with `gpu_runtime`.
- [ ] Compare derivative kernels, Boozer JVPs, and Boozer VJPs with
  `derivative_heavy`.
- [ ] Compare well-conditioned adjoint vectors with
  `exact_well_conditioned_adjoint`.
- [ ] For ill-conditioned adjoint cases, record residual/failure-only and do
  not assert vector parity.
- [ ] Fail the artifact if any fixed-state CPU/C++ vs JAX comparison drifts.

Tests:

- [ ] Add unit tests for schema validation.
- [ ] Add unit tests for comparison status aggregation.
- [ ] Add a small fake-artifact test proving `cpp_cpu_vs_jax_gpu` is required.
- [ ] Add a test that missing `cpp_cpu` blocks `fixed_state_physics_parity`.
- [ ] Add a test that JAX CPU vs GPU pass cannot override CPU/C++ drift.

## Workstream B: `JF.x` to `bs.x` Coordinate Mapping Proof

Target file:

- [ ] Add `benchmarks/single_stage_dof_mapping_proof.py`.
- [ ] Add `tests/integration/test_single_stage_dof_mapping.py`.

Required artifact:

- [ ] Write `.artifacts/parity/<date>-coordinate-mapping-proof.json`.
- [ ] Make `--coordinate-mapping-json` required for release-gate matrix runs.
- [ ] Keep the pytest fixture and the producer on the same proof helper so the
  JSON schema and test assertions cannot drift.

Artifact schema:

- [ ] `schema_version`.
- [ ] `status`.
- [ ] `inputs`.
- [ ] `mapping`.
- [ ] `active_indices`.
- [ ] `frozen_indices`.
- [ ] `state_reconstruction`.
- [ ] `gradient_projection`.
- [ ] `finite_difference_checks`.
- [ ] `failures`.

Required fixtures:

- [ ] Small deterministic single-stage setup.
- [ ] CPU/C++ legacy `JF` objective.
- [ ] JAX target `BiotSavart` / `bs` object.
- [ ] Same restored coil/surface/Boozer state.
- [ ] At least one frozen TF group and one active banana-coil group.

Mapping todos:

- [ ] Extract `JF.x`.
- [ ] Extract `JF.dofs_free_status`.
- [ ] Extract `bs.x`.
- [ ] Extract per-coil and per-current local free masks from the JAX target
  lane.
- [ ] Build an explicit mapping object from legacy `JF.x` indices to target
  `bs.x` indices.
- [ ] Assert mapped active variables represent the same physical coil DOFs.
- [ ] Assert fixed TF variables do not appear in the JAX optimizer state.
- [ ] Assert inactive/frozen variables stay unchanged after one target-lane
  optimizer step.

State reconstruction todos:

- [ ] Apply a small `bs.x` perturbation.
- [ ] Reconstruct coil specs from `bs.x`.
- [ ] Apply the mapped perturbation to the legacy CPU/C++ graph.
- [ ] Assert physical coil geometry agrees after reconstruction.
- [ ] Assert current values and fixed-current masks agree.

Gradient projection todos:

- [ ] Evaluate CPU/C++ `JF.dJ()` at the same state.
- [ ] Project CPU/C++ gradient from `JF.x` basis into `bs.x` basis.
- [ ] Evaluate JAX gradient in `bs.x` basis.
- [ ] Assert projected CPU/C++ gradient matches JAX gradient under
  `derivative_heavy`.
- [ ] Include finite-difference directional checks for at least three mapped
  active directions.
- [ ] Include a frozen-variable perturbation check showing no optimizer-state
  leakage.

Failure semantics:

- [ ] Missing `coordinate_mapping_proof.json` fails
  `coordinate_mapping_parity`.
- [ ] Missing mapping metadata fails the test and producer.
- [ ] Shape mismatch should fail with the lane names and vector sizes.
- [ ] Mask mismatch should identify the first mismatched DOF name or index.

## Workstream C: Parity Matrix Release-Gate Upgrade

Target file:

- [ ] Update `benchmarks/single_stage_parity_matrix.py`.

CLI todos:

- [ ] Add `--fixed-state-parity-json`.
- [ ] Add required `--coordinate-mapping-json` for release-gate mode.
- [ ] Keep existing `--parity-report-json`.
- [ ] Keep existing progress JSON options for CPU/SciPy, JAX CPU, and GPU.

Lane todos:

- [ ] Add `LANE_CPP_CPU = "cpp_cpu"`.
- [ ] Keep `LANE_CPU_SCIPY = "cpu_scipy"`.
- [ ] Keep `LANE_CPU_CPP_TRACE = "cpu_cpp_trace"` as diagnostic.
- [ ] Keep `LANE_JAX_CPU = "jax_cpu"`.
- [ ] Add canonical `LANE_JAX_GPU = "jax_gpu"`.
- [ ] Delete `LANE_H100_GPU` from the release-gate matrix contract.
- [ ] Rename producer outputs from `jax_cpu_vs_h100_*` to
  `jax_cpu_vs_jax_gpu_*` in the same implementation PR.
- [ ] Matrix readers must require canonical `jax_gpu` keys and fail on legacy
  `h100` keys in release-gate mode.
- [ ] Update tests to assert legacy `h100` keys are rejected by release-gate
  mode.

Comparison todos:

- [ ] Add `cpp_cpu_vs_jax_cpu_fixed_state`.
- [ ] Add `cpp_cpu_vs_jax_gpu_fixed_state`.
- [ ] Keep `jax_cpu_vs_jax_gpu_fixed_state`.
- [ ] Keep CPU/SciPy vs JAX CPU final/public behavior comparisons separate from
  fixed-state kernel comparisons.
- [ ] Keep target-style optimizer trace comparison only for lanes that actually
  emit target-style traces.
- [ ] Treat CPU/SciPy progress without target-style trace as termination/final
  state evidence, not trace evidence.

Output todos:

- [ ] Emit `buckets.fixed_state_physics_parity`.
- [ ] Emit `buckets.coordinate_mapping_parity`.
- [ ] Emit `buckets.optimizer_public_behavior_parity`.
- [ ] Emit `buckets.final_metric_envelope`.
- [ ] Emit `buckets.performance_memory_report`.
- [ ] Emit `blocking_buckets`.
- [ ] Emit top-level `release_gate_passed`.
- [ ] Emit structured `first_divergence` when full-run public behavior drifts:
  `stage`, `lane_pair`, `metric`, `evidence`, and `explanation`.
- [ ] Restrict `first_divergence.stage` to
  `fixed_state_physics`, `coordinate_mapping`, `initial_gradient`,
  `line_search`, `termination`, or `final_sync`.
- [ ] Fail `performance_memory_report` when required memory or performance
  metrics are missing, over checked-in memory budget, or below checked-in speed
  floors.

Tests:

- [ ] Update `tests/test_benchmark_helpers.py` for the new matrix schema.
- [ ] Add a fixture where JAX CPU vs GPU passes but `cpp_cpu_vs_jax_cpu` drifts;
  assert `release_gate_passed` is false.
- [ ] Add a fixture where fixed-state passes but coordinate mapping is missing;
  assert release is blocked.
- [ ] Add a fixture where CPU/SciPy has no target trace; assert public behavior
  can still be evaluated without falsely requiring internal Wolfe trace parity.
- [ ] Add a fixture where GPU provenance is missing for CUDA; assert
  `performance_memory_report` is blocked.
- [ ] Add a fixture where memory/provenance is recorded but exceeds
  `evaluate_grouped_adjoint_memory_budget`; assert
  `performance_memory_report` fails.
- [ ] Add a fixture where final-state drift emits a valid `first_divergence`
  enum value with supporting evidence.

## Workstream D: Same-Seed Full Runs

Required lanes:

- [ ] CPU/SciPy full run: legacy `JF.x` plus SciPy `L-BFGS-B`.
- [ ] JAX CPU full run: target lane, `ondevice`, `bs.x`.
- [ ] JAX GPU full run: target lane, `ondevice`, `bs.x`.

Seed policy:

- [ ] Start all lanes from the same init-only seed.
- [ ] Do not seed JAX from CPU/SciPy final state.
- [ ] Record seed run dir.
- [ ] Record runtime seed spec hash.
- [ ] Record equilibrium hash.
- [ ] Record Biot-Savart JSON hash.

Public behavior comparisons:

- [ ] Initial objective.
- [ ] Initial gradient norm.
- [ ] Initial physical metrics.
- [ ] Termination status.
- [ ] Termination message.
- [ ] Final objective.
- [ ] Final gradient norm.
- [ ] Final physical metrics.
- [ ] Final constraints.
- [ ] Final mapped coordinate state.
- [ ] Runtime.
- [ ] Memory.

Path-sensitivity policy:

- [ ] Exact optimizer trajectory equality is required only for lanes that share
  the same optimizer contract and trace visibility.
- [ ] JAX CPU vs JAX GPU target-lane trace parity should stay strict.
- [ ] CPU/SciPy vs JAX target-lane parity is public-behavior based in this gate;
  SciPy Wolfe-internal trace parity is not part of the release claim.
- [ ] Any full-run final-state drift must identify the first known divergence:
  fixed-state physics, coordinate mapping, initial gradient, line search,
  termination, or final sync.
- [ ] The matrix must encode that divergence in a structured `first_divergence`
  object for failing reports; free-form reason strings are report text only,
  not gate semantics.
- [ ] `first_divergence` is diagnostic evidence only. It cannot convert a
  failing public-behavior comparison into a release pass.

Acceptance:

- [ ] CPU/SciPy vs JAX CPU public behavior passes the release-gate envelope.
- [ ] CPU/SciPy vs JAX GPU public behavior passes the release-gate envelope.
- [ ] JAX CPU vs JAX GPU passes strict target-lane trajectory parity.
- [ ] Runtime and memory are recorded for all lanes.

## Workstream E: GPU Proof

Runtime requirements:

- [ ] `jax_enable_x64` must be true.
- [ ] CUDA request must initialize a CUDA/GPU backend.
- [ ] CUDA lanes must prove `jax.default_backend()` and selected devices are
  CUDA; CPU execution in a CUDA lane is a hard failure.
- [ ] Device kind must be recorded.
- [ ] NVIDIA driver and total memory must be recorded for CUDA lanes.
- [ ] XLA flags must be recorded.
- [ ] CUDA proof subprocesses must set `--xla_gpu_exclude_nondeterministic_ops=true` through
  the shared environment builder.
- [ ] CUDA proof provenance must prove the deterministic flag was present during
  execution.
- [ ] CUDA proof env vars must be recorded.
- [ ] jaxlib version must be recorded.
- [ ] Compile time and run time must be separated.
- [ ] GPU memory high-water must be recorded.
- [ ] `performance_memory_report` must fail if memory data is missing or exceeds
  checked-in budget.

Proof comparisons:

- [ ] Compare JAX GPU directly to `cpp_cpu`.
- [ ] Compare JAX GPU to JAX CPU.
- [ ] Do not use JAX CPU as the only bridge between C++ and GPU.

H100 run todos:

- [ ] Run fixed-state producer on H100.
- [ ] Run full target-lane JAX GPU artifact on H100.
- [ ] Pull result JSON, progress JSON, memory/provenance JSON, and logs.
- [ ] Run the parity matrix locally or on the GPU host against the full artifact
  set.
- [ ] Preserve exact command lines and environment.

## Workstream F: Reporting

Target report:

- [ ] Add `.artifacts/parity/<date>-release-grade-cpp-jax-gate/report.json`.
- [ ] Add `.artifacts/parity/<date>-release-grade-cpp-jax-gate/report.md`.

Report contents:

- [ ] Top-level yes/no release verdict.
- [ ] Bucket statuses.
- [ ] Blocking comparisons.
- [ ] Fixed-state deltas.
- [ ] Coordinate-mapping proof status.
- [ ] Full-run public behavior deltas.
- [ ] First divergence explanation when public behavior drifts.
- [ ] Structured `first_divergence` enum and evidence object when public
  behavior drifts.
- [ ] Runtime table.
- [ ] Memory table.
- [ ] Memory-budget and performance-budget pass/fail table.
- [ ] Device and version table.
- [ ] Artifact paths.
- [ ] Commands used.
- [ ] Git SHA and dirty status.

Human-readable verdict text:

```text
Release gate: FAIL

Reason:
- fixed_state_physics_parity is blocked because cpp_cpu_vs_jax_gpu is missing.
- coordinate_mapping_parity is blocked because JF.x <-> bs.x gradient projection
  proof is missing.

JAX CPU vs GPU parity is not enough for release.
```

## Workstream G: Validation Commands

After implementation, run targeted checks first:

- [ ] `python -m pytest tests/test_benchmark_helpers.py -k "single_stage_parity_matrix or parity_ladder_tolerances" -q`
- [ ] `python -m pytest tests/geo/test_single_stage_example.py -k "optimizer_initial_dofs or dof_mapping or target_lane" -q`
- [ ] `python -m pytest tests/integration/test_single_stage_dof_mapping.py -q`

Run coordinate mapping proof:

- [ ] `python benchmarks/single_stage_dof_mapping_proof.py --output-json .artifacts/parity/<date>-coordinate-mapping-proof.json`

Run fixed-state CPU/JAX CPU locally:

- [ ] `python benchmarks/single_stage_cpp_jax_state_parity.py --platform cpu --output-json .artifacts/parity/<date>-fixed-state-cpu.json`

Run matrix locally with CPU/JAX CPU evidence:

- [ ] `python benchmarks/single_stage_parity_matrix.py --fixed-state-parity-json .artifacts/parity/<date>-fixed-state-cpu.json --coordinate-mapping-json .artifacts/parity/<date>-coordinate-mapping-proof.json --parity-report-json <report.json> --output-json .artifacts/parity/<date>-matrix-cpu.json`

Run GPU proof on H100:

- [ ] `python benchmarks/single_stage_cpp_jax_state_parity.py --platform cuda --output-json .artifacts/parity/<date>-fixed-state-h100.json`

Run final release matrix:

- [ ] `python benchmarks/single_stage_parity_matrix.py --fixed-state-parity-json <fixed-state-json> --coordinate-mapping-json <coordinate-mapping-json> --parity-report-json <merged-report-json> --cpu-progress-json <cpu-progress-json> --jax-cpu-progress-json <jax-cpu-progress-json> --gpu-progress-json <gpu-progress-json> --output-json <release-matrix-json>`

Static cleanup:

- [ ] `git diff --check`
- [ ] Run the repo's focused lint/format command for touched files.

## Definition of Done

The release-grade gate is complete when all boxes below are checked:

- [ ] Fixed-state CPU/C++ vs JAX CPU passes.
- [ ] Fixed-state CPU/C++ vs JAX GPU passes.
- [ ] Fixed-state JAX CPU vs JAX GPU passes.
- [ ] Equilibrium, runtime seed spec, Biot-Savart JSON, objective
  configuration, and active DOF mask hashes agree across all fixed-state lanes.
- [ ] `JF.x` to `bs.x` active-variable mapping passes.
- [ ] `JF.x` gradient projected to `bs.x` matches JAX `bs.x` gradient.
- [ ] Frozen/inactive variables are proven not to leak into JAX optimizer state.
- [ ] CPU/SciPy full-run public behavior vs JAX CPU passes.
- [ ] CPU/SciPy full-run public behavior vs JAX GPU passes.
- [ ] JAX CPU vs JAX GPU strict target-lane trajectory parity passes.
- [ ] GPU runtime proof records x64, real CUDA device execution, device kind,
  memory, compile time, run time, jaxlib, CUDA env, deterministic XLA flag, and
  XLA flags.
- [ ] Performance and memory bucket records required metrics and passes checked-in
  budgets.
- [ ] Full-run drift emits a structured `first_divergence` with fixed enum stage
  and evidence, and the release gate remains failed until the drift is fixed.
- [ ] Release-gate mode uses only canonical `jax_gpu` keys and rejects legacy
  `h100` keys.
- [ ] Matrix emits bucket-level statuses and top-level `release_gate_passed`.
- [ ] Final JSON and Markdown reports are written.
- [ ] The report says "JAX preserves existing SIMSOPT CPU/C++ behavior" only
  when all required buckets pass.

## Immediate Next Actions

- [ ] Implement `benchmarks/single_stage_cpp_jax_state_parity.py`.
- [ ] Implement `benchmarks/single_stage_dof_mapping_proof.py`.
- [ ] Add fake-artifact matrix tests proving direct `cpp_cpu` comparisons are
  required.
- [ ] Add `tests/integration/test_single_stage_dof_mapping.py`.
- [ ] Extend `benchmarks/single_stage_parity_matrix.py` to consume the fixed-state
  artifact and emit bucketed release-gate output.
- [ ] Run local CPU/JAX CPU fixed-state proof.
- [ ] Run H100 fixed-state proof.
- [ ] Generate one release-gate report with pass/fail, deltas, runtime, memory,
  and artifact paths.
