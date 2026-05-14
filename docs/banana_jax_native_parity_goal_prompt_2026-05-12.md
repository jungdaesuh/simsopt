# Banana JAX Native Parity Goal Prompt

Use this prompt to drive an agent through a comprehensive verification loop for
the banana coil optimization JAX/GPU port.

```text
Goal: Verify, and if needed repair, the simsopt-jax banana coil optimization
JAX/GPU port until its native JAX lanes are scientifically release-grade
against the original SIMSOPT CPU/C++/SciPy lanes.

Repository: /Users/suhjungdae/code/columbia/simsopt-jax

Branch/worktree rule:
- Inspect current HEAD and dirty state first.
- Do not revert unrelated changes.
- Do not claim completion from docs, manifests, fake CUDA tests, dry-run
  launchers, or old artifacts alone.

Core release contract:
1. Existing SIMSOPT CPU/C++/SciPy behavior is the oracle.
2. JAX CPU must match the oracle at identical fixed states.
3. JAX CUDA/GPU must match the same oracle on a real CUDA device when GPU
   parity is claimed.
4. JAX CPU and JAX GPU must match each other.

JAX-vs-JAX agreement alone is not enough.
CPU backend aliases, fake CUDA tests, dry-run launchers, and provenance-only
artifacts do not satisfy the CUDA row.

Scope:
- Stage 2 banana optimizer:
  examples/single_stage_optimization/STAGE_2/banana_coil_solver.py
- Single-stage banana optimizer:
  examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py
- Core JAX target lane:
  BiotSavartJAX, SquaredFluxJAX / integral_BdotN_jax,
  Stage2TargetObjectiveBundle, BoozerSurfaceJAX, surfaceobjectives_jax,
  optimizer_jax / target_minimize, immutable spec and restart artifacts.
- CPU/C++ oracle lane:
  BiotSavart, SquaredFlux, BoozerSurface.run_code /
  solve_residual_equation_exactly_newton, BoozerResidual, Iotas,
  NonQuasiSymmetricRatio, SciPy L-BFGS-B / least_squares, and simsoptpp
  kernels or wrappers they call.

Non-negotiables:
- Do not loosen tolerances to hide drift.
- Keep fixed-state oracle checks separate from optimizer-trace diagnostics.
- Do not use lbfgs-trace or host-core traces as the CPU/C++ oracle.
- Do not route production JAX/GPU execution through host SciPy to pass parity.
- Host boundaries are allowed only when explicit and outside the compute
  contract: startup/setup compatibility, optional diagnostics/logging,
  artifact export, and final JSON writing.
- If a path re-enters legacy Optimizable/C++/SciPy code in the target compute
  lane, classify it as a port gap unless the repo explicitly documents it as
  outside the JAX-native contract.
- Treat benchmarks/validation_ladder_contract.py as the SSOT for parity
  tolerances and GPU proof metadata. Do not duplicate, soften, or reinterpret
  tolerances elsewhere.
- Jitted target compute must be functionally pure: no hidden Python mutation,
  global state dependence, print/debug callbacks, device_get/host conversion,
  or host callback in value/gradient/Boozer solve kernels. Any observability
  callback belongs outside the target compute contract.

Required investigation loop:

1. Read the current docs and treat them as hypotheses, not proof:
   - docs/jax_parity_manifest.md
   - docs/banana_jax_full_test_parity_coverage_impl_plan_2026-05-06.md
   - docs/banana_jax_native_port_todos_2026-05-05.md
   - docs/banana_cpp_cpu_dependency_manifest_2026-05-05.md
   - docs/single_stage_banana_jax_gpu_dependency_trace_2026-04-13.md
   - docs/release_grade_cpp_jax_behavior_preservation_plan_2026-05-02.md
   - docs/parity_scientific_equivalence_contract_2026-05-09.md
   - benchmarks/validation_ladder_contract.py

2. Re-check official docs before judging behavior:
   - JAX docs for jit purity, explicit state threading, transfer_guard,
     device_get, asynchronous dispatch / block_until_ready benchmarking, and
     GPU memory allocation.
   - SIMSOPT docs and current source for BoozerSurface, Boozer residual
     solves, BiotSavart, SquaredFlux, BoozerResidual, Iotas, and
     NonQuasiSymmetricRatio semantics.
   - NVIDIA CUDA docs for CUDA_VISIBLE_DEVICES, CUDA_LAUNCH_BLOCKING, PTX JIT,
     driver/runtime, and GPU provenance fields used by the run.
   Record the doc URLs or versions in the verdict.

3. Run a stale-code and stale-artifact sweep:
   - Every referenced file, class, function, test, benchmark, and artifact
     path must exist in the current tree or be removed from the proof claim.
   - Separate historical artifacts from current-HEAD evidence. Historical
     artifacts may explain context; they cannot prove current CUDA parity.
   - Update stale manifest rows, stale TODO checkboxes, stale command examples,
     and stale proof labels before using them downstream.

4. Build a current code-path map for both banana entrypoints:
   - CPU/C++/SciPy oracle path.
   - JAX CPU target path.
   - JAX CUDA target path.
   - Explicit host setup/reporting/artifact seams.
   - Any accidental host/C++/SciPy re-entry in target-lane value, gradient,
     Boozer solve, hardware constraints, accepted-step reporting, or final
     metrics.
   - Explicit JAX device placement, jit/lower/compile boundaries, and all
     device-to-host or host-to-device transfers.

5. Verify fixed-state math/physics parity before optimizer behavior:
   - Biot-Savart B, derivatives, and VJP.
   - integral_BdotN / SquaredFlux values and gradients.
   - Surface geometry gamma, normals, and derivatives used by banana.
   - Boozer residual vectors, norms, JVP/VJP/adjoints, and solve quality.
   - Assembled single-stage objective components and full optimizer-basis
     gradient.
   - Hardware metrics: iota, G, volume, field error, coil length, curvature,
     coil-coil distance, coil-plasma distance, plasma-vessel distance, and
     self-intersection status.
   - Numerical conditioning and solve-quality metrics for exact/least-squares
     Boozer paths, including original residual norms after any preconditioning
     or basis transform.

6. Verify optimizer behavior after fixed-state parity:
   - CPU/SciPy full-run public behavior vs JAX CPU and JAX GPU.
   - Final objective, gradient norm, final mapped state, final physics metrics,
     constraints, termination, and accepted-state behavior.
   - Treat iterate-by-iterate identity as optional unless a doc or test
     explicitly requires it.
   - Benchmark only after warmup/compilation is separated from execution and
     accelerator work is synchronized with block_until_ready or an equivalent
     explicit host read at the measurement boundary.

7. Verify upstream, downstream, and E2E regression:
   - Upstream: the CPU/C++/SciPy lane remains compatible with public SIMSOPT
     semantics and is not forced through JAX target internals.
   - Downstream: immutable specs, restart artifacts, parity matrix/report
     consumers, docs manifest checks, HF/Runpod proof launchers, and JSON
     schema readers still consume the emitted artifacts.
   - E2E: Stage 2 strict reduced/full run -> saved spec/restart/output
     artifacts -> single-stage init/continuation -> parity matrix/proof report.
   - Report regressions as bugs even when isolated fixed-state checks still
     pass.

8. Run or extend the existing proof surfaces:
   - tests/docs/test_banana_parity_coverage_manifest.py
   - tests/test_state_artifact_merge_logic.py
   - tests/test_hf_production_gpu_proof.py
   - tests/integration/test_stage2_jax.py
   - tests/integration/test_stage2_target_lane_purity.py
   - tests/integration/test_single_stage_jax_cpu_reference.py
   - tests/integration/test_single_stage_physics_parity.py
   - tests/field/test_biotsavart_jax_parity.py
   - tests/objectives/test_fluxobjective_jax_parity.py
   - tests/objectives/test_integral_bdotn_jax.py
   - tests/geo/test_boozersurface_jax.py
   - benchmarks/single_stage_cpp_jax_state_parity.py
   - benchmarks/single_stage_init_parity.py
   - benchmarks/single_stage_parity_matrix.py
   - benchmarks/stage2_value_gradient_parity.py
   - benchmarks/production_boozer_parity_probe.py

9. For GPU claims, require real CUDA artifacts from the current repo state:
   - git SHA and dirty-tree status.
   - exact command.
   - JAX and jaxlib versions, x64 status, and backend mode.
   - CUDA/XLA flags, deterministic GPU reductions flag, and CUDA_VISIBLE_DEVICES.
   - nvidia-smi GPU facts, driver/runtime version.
   - peak RSS and peak GPU memory.
   - pass/fail metadata and parity JSON.
   - proof that target arrays and compiled executable ran on CUDA, with host
     transfers limited to named setup/report/export boundaries.

CPU-only tests, fake CUDA tests, dry-run launchers, or provenance scaffolding do
not close CUDA parity.

Deliverables:
- A concise current-state verdict: PASS, PARTIAL, or FAIL for native JAX port
  status.
- A table of every banana-relevant lane: CPU/C++ oracle, JAX CPU, JAX GPU,
  CPU-vs-JAX CPU, CPU-vs-JAX GPU, and JAX CPU-vs-GPU.
- A list of every host-bound seam, classified as allowed boundary, open port
  gap, or intentional diagnostic/reporting path.
- Exact commands run and artifacts inspected.
- Official docs checked and how they affected the verdict.
- Upstream, downstream, and E2E regression results.
- Stale code, stale artifact, and stale documentation fixes.
- Any fixes implemented, with focused tests.
- Remaining blockers, if any, in priority order.

Completion condition:
Only declare success when the current tree proves the trust chain with real
evidence:

CPU/C++/SciPy oracle -> JAX CPU -> JAX CUDA/GPU -> JAX CPU/GPU agreement

The proof must cover fixed-state physics parity, optimizer public-behavior
parity, final metric envelope, upstream/downstream/E2E regression, stale-proof
cleanup, and GPU provenance. If any CUDA row is still hardware-blocked, any
referenced proof surface is stale, or any target compute path silently re-enters
legacy C++/SciPy, report PARTIAL and continue or produce the next narrow fix.
```
