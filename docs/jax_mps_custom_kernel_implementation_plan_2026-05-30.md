# JAX-MPS SIMSOPT Custom Kernel Implementation Plan

## Purpose

Define an executable implementation plan for the highest-performance Apple MPS path for SIMSOPT's `scipy-jax` Boozer objective. The plan targets a custom SIMSOPT kernel boundary that avoids the current jax-mps dynamic `stablehlo.while` host synchronization bottleneck.

This is a planning artifact, not a claim that the custom kernel already exists.

## Goals

- Add an opt-in MPS custom-kernel path that returns the Boozer objective value and coil-DOF gradient for the single-stage `scipy-jax` target lane.
- Keep the hot Newton/GMRES convergence loop inside one backend/kernel-owned execution boundary instead of exposing each loop predicate through jax-mps `while` evaluation.
- Preserve the current CPU JAX and CUDA-compatible paths as the scientific oracle and default fallback.
- Prove objective, gradient, and solve-state parity against the existing JAX path before using any performance result.
- Produce a completed MPS `maxiter=3` smoke run with improved matched checkpoint timings before declaring the path useful.

## Non-Goals

- Do not replace the existing CPU JAX, CUDA JAX, or public SciPy-L-BFGS-B optimizer paths.
- Do not try to solve general jax-mps `stablehlo.while` execution in this plan.
- Do not add a Python-side MLX sidecar as the production path if it requires JAX-array to MLX-array host copies.
- Do not support every surface, curve, label, or Boozer mode in the first production gate.
- Do not claim production-grade f64 Apple GPU support unless the backend/runtime can actually execute it.

## Current Context

- `src/simsopt/geo/surfaceobjectives_jax.py` builds `BoozerResidualJAX._direct_objective_value_and_grad` with `_make_cached_strict_scalar_value_and_grad(...)`, which wraps a JIT-compiled JAX value-and-gradient callable.
- `src/simsopt/geo/boozersurface_jax.py` routes traceable exact and least-squares Boozer solves through dynamic Newton/least-squares loops, including `jax.lax.while_loop`.
- `src/simsopt/geo/optimizer_jax.py` routes exact Newton linear solves through `_run_operator_gmres(...)`, which calls `jax.scipy.sparse.linalg.gmres(...)`.
- `src/simsopt/jax_core/field.py` already has immutable `GroupedCoilSetSpec` based field kernels such as `grouped_biot_savart_B_from_spec(...)`.
- `src/simsopt/field/biotsavart_jax_backend.py` already exposes explicit `coil_set_spec_from_dofs(...)` and grouped field inputs, which are the right source of flattened kernel inputs.
- `/Users/suhjungdae/code/opensource/jax-mps/src/pjrt_plugin/ops/control_flow.cc` currently executes dynamic `stablehlo.while` loops by running MLX eval and reading `.item<bool>()` predicates on the host.
- The same jax-mps file already handles `stablehlo.custom_call` targets such as `mps.sdpa`, `mps.addmm`, `mps.qr`, and `mps.svd`.
- `/Users/suhjungdae/code/opensource/jax-mps/src/pjrt_plugin/pjrt_executable.cc` marks execution synchronous for now.
- Official MLX docs confirm MLX lazy evaluation, scalar `.item()` forcing evaluation, `mx.fast.metal_kernel(...)` for Python custom Metal kernels, and C++ extension support:
  - https://ml-explore.github.io/mlx/build/html/usage/quick_start.html
  - https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html
  - https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
  - https://ml-explore.github.io/mlx/build/html/dev/extensions.html
- `pyproject.toml` uses `scikit-build-core`; `CMakeLists.txt` already builds a pybind11 C++ extension for `simsoptpp`.
- The local `.conda/jax-mps` environment has JAX `0.10.0`. It exposes `jax.ffi` and `jax.interpreters.mlir.register_lowering`, but local introspection did not find `jax.extend.mlir.custom_call` or `jax.interpreters.mlir.custom_call`. The first implementation gate must therefore prove the exact JAX lowering route before numerical kernel work starts.
- Prior fixed-trip jax-mps chunking improved synthetic loops but did not complete the real SIMSOPT MPS `maxiter=3` smoke within 600 seconds.

## Rationale

The best-performance path is a backend-routed custom call, not a Python-only rewrite.

The current MPS slowdown is caused by too many small dynamic control-flow boundaries. The custom kernel should therefore make the useful abstraction boundary larger: one explicit SIMSOPT operation should own the hot Boozer solve and return final value/gradient data.

Design-it-twice:

- Approach A: building-block custom calls for Biot-Savart `B`, Biot-Savart VJP, Boozer residual, and linearized operator matvec. This is easier to validate incrementally and reuses more of the current solver, but may still leave dynamic Newton/GMRES control flow exposed.
- Approach B: one fused Boozer solve/value-gradient custom call. This is highest performance because it can own Newton/GMRES iteration, convergence checks, residual reductions, and gradient assembly internally, but it duplicates more numerical logic and has higher parity risk.

Decision: implement Approach A as the measurement/parity ladder, but do not stop there. The completion target is Approach B for the narrow production fixture shape. Approach A is useful only if it proves kernel formulas and data layout before fusion.

## Assumptions

- The first target is Apple MPS through local `jax-mps`, not CUDA or CPU.
- The first numerical target can be float32/MPS-compatible, with CPU JAX float32 parity used as the acceptance oracle when f64 is not available on MPS.
- The first supported SIMSOPT shape is the reduced single-stage smoke shape, then `mpol=4`/`ntor=4` if the smoke passes.
- The implementation can modify both `/Users/suhjungdae/code/columbia/simsopt-jax` and `/Users/suhjungdae/code/opensource/jax-mps`.
- The production integration should use `stablehlo.custom_call` handled by jax-mps so data stays inside the JAX/PJRT execution path.
- If MLX C++ does not expose the same custom Metal kernel surface as Python `mx.fast.metal_kernel`, the backend part will need either an MLX C++ extension route or a direct Metal implementation inside the jax-mps plugin.

## Implementation Plan

1. Prove custom-call and kernel feasibility before numerical work
   - [ ] In SIMSOPT, create the smallest possible private JAX primitive or `jax.ffi` wrapper that lowers one scalar array operation to a named `stablehlo.custom_call` target.
   - [ ] In jax-mps, add a temporary custom-call target that accepts the smoke operation and returns a deterministic result.
   - [ ] Verify the lowered IR actually contains the expected `stablehlo.custom_call` target name.
   - [ ] Verify the custom-call smoke executes through `.conda/jax-mps/bin/python` without routing through a Python MLX sidecar.
   - [ ] Verify whether the jax-mps backend can call MLX C++ custom kernels directly; if not, choose direct Metal command encoding or an MLX extension boundary before writing numerical kernels.
   - [ ] Remove the temporary smoke target or keep it only as a backend test fixture after the route is proven.

2. Freeze the target contract and baseline artifacts
   - [ ] Define the first supported fixture: `BoozerResidualJAX`, least-squares Boozer surface, uniform `CurveXYZFourier` coil fast path, reduced smoke resolution.
   - [ ] Record the exact flattened input contract: coil DOFs, grouped coil gammas, grouped coil tangents, currents, surface DOFs, quadrature grids, solver tolerances, warm-start `x_inner`, label configuration, and static shape metadata.
   - [ ] Add a shape/dtype dump helper beside the existing benchmark artifacts, not in the hot path, that serializes the target kernel input schema to `.artifacts/mps_custom_kernel_contract/*.json`.
   - [ ] Capture current CPU JAX and current MPS baseline timings for initial value/grad and `maxiter=3` using the existing `benchmarks/single_stage_init_parity.py` harness.
   - [ ] Capture the current jax-mps while-profile counters for the same run so later validation can prove the custom path reduced dynamic loop host reads.

3. Add a CPU oracle for the custom-kernel boundary
   - [ ] Create a narrow SIMSOPT module for the custom-kernel contract, for example `src/simsopt/jax_core/mps_boozer_kernel_contract.py`.
   - [ ] Implement a pure JAX/CPU oracle function with the same flattened inputs and outputs planned for the custom kernel.
   - [ ] Make the oracle call the existing JAX functions rather than reimplementing formulas in Python.
   - [ ] Add parity tests that compare the flattened-contract oracle with current `BoozerResidualJAX._direct_objective_value_and_grad(...)`.
   - [ ] Keep the flattened contract immutable: runtime arrays are data leaves, shape/policy values are static metadata.

4. Add backend-routed custom-call plumbing
   - [ ] In SIMSOPT, add the private lowering module using the JAX route proven in Phase 1; do not assume a `custom_call` helper exists in `jax.extend.mlir`.
   - [ ] Emit `stablehlo.custom_call` only for the MPS custom-kernel path.
   - [ ] Use explicit target names such as `mps.simsopt_biot_savart_b`, `mps.simsopt_biot_savart_vjp`, and `mps.simsopt_boozer_value_grad`.
   - [ ] Keep CPU/CUDA lowering routed to the oracle or unsupported-path tests; do not silently route non-MPS devices into partial custom behavior.
   - [ ] In jax-mps, move SIMSOPT-specific custom-call handling into a separate handler file instead of growing `control_flow.cc` further, then dispatch from `HandleCustomCall(...)`.
   - [ ] Add structured backend-config parsing for static shape metadata, solver bounds, tolerances, and output layout.

5. Implement Stage 1 kernels: Biot-Savart value and pullback
   - [ ] Implement a custom kernel for grouped Biot-Savart `B(points, coil_spec)` for the supported uniform coil layout.
   - [ ] Implement the matching pullback/VJP for coil DOFs needed by direct objective gradients.
   - [ ] Use row-contiguous flattened buffers for points, gammas, gammadashs, currents, and group offsets.
   - [ ] Compare kernel output against `grouped_biot_savart_B_from_spec(...)` at fixed smoke shapes.
   - [ ] Compare kernel VJP against `jax.vjp(...)` of the existing JAX Biot-Savart path.
   - [ ] Benchmark Stage 1 against the current JAX Biot-Savart path with `block_until_ready()` and a warmed compile cache.

6. Implement Stage 2 kernels: Boozer residual blocks
   - [ ] Implement kernel support for surface geometry outputs consumed by the Boozer residual, or prove the existing surface Fourier JAX code is not the bottleneck at the target shape.
   - [ ] Implement a custom Boozer residual evaluation from `B`, `xphi`, `xtheta`, `iota`, `G`, and label data.
   - [ ] Implement the residual pullback pieces needed by the direct coil gradient.
   - [ ] Add tests comparing residual scalar, residual vector norm, and direct gradient against the current JAX path.
   - [ ] Record whether Stage 2 reduces kernel count and host synchronization in the jax-mps profile.

7. Implement Stage 3 fused solve/value-gradient custom call
   - [ ] Define the fused op output contract: objective value, coil gradient, final packed inner state, residual norm, gradient norm, Newton iteration count, GMRES iteration count, convergence flag, and finite flag.
   - [ ] Implement bounded Newton and GMRES loops inside the custom operation using device-owned convergence state. The host may read convergence only after the fused op completes.
   - [ ] Preserve current solver semantics for tolerances, acceptance, finite checks, and warm-start handling at the supported fixture shape.
   - [ ] Implement explicit gradient assembly inside the fused operation or a paired backward custom call; do not rely on tracing through the internal solver loop.
   - [ ] Add a fail-loud status output for unsupported shape, unsupported dtype, non-finite intermediate state, or convergence failure.
   - [ ] Keep diagnostic status as data returned from the operation; do not add hidden global state to the kernel.

8. Integrate with SIMSOPT objective routing
   - [ ] Add an experimental opt-in resolver for the custom path at the objective/value-gradient boundary used by `BoozerResidualJAX`.
   - [ ] Reuse the existing `_simsopt_value_and_grad` callable marker so `_evaluate_scalar_or_value_and_grad(...)` does not need another dispatch concept.
   - [ ] Gate the custom path on explicit support checks: active jax-mps platform, supported fixture shape, supported dtype, supported Boozer mode, and explicit experimental opt-in.
   - [ ] Leave the existing JAX value/grad callable as the default until parity and smoke gates pass.
   - [ ] Make unsupported custom-kernel requests fail loudly rather than falling back silently.

9. Performance and parity hardening
   - [ ] Add an MPS dynamic-convergence microbenchmark that mimics Newton/GMRES loop structure and measures host predicate reads before and after the fused op.
   - [ ] Add benchmark output fields for compile time, first value/grad time, warmed value/grad time, per-objective trace count, while-profile read count, and peak memory if available.
   - [ ] Compare CPU JAX oracle, current MPS JAX, Stage 1 custom calls, Stage 2 custom calls, and Stage 3 fused custom call at the same fixture shape.
   - [ ] Run numerical parity at multiple tolerances and reject any result where objective or gradient drift exceeds the agreed f32/f64 tolerance contract.
   - [ ] Document all unsupported shapes and precision modes in the plan/report before any code is marked complete.

## Validation Plan

Validation commands below are post-implementation gates. In the current checkout, plain `uv run ...` attempts an editable build and fails before script execution because version metadata parsing rejects the current Git tag. Use the repo-local `.conda/jax-mps/bin/python` environment for current argument/path checks, or fix the editable-build metadata issue before relying on `uv run`.

- [ ] `uv run ruff check src/simsopt tests benchmarks`, or the equivalent repo environment command once the editable-build issue is resolved.
- [ ] `uv run python -m py_compile src/simsopt/jax_core/mps_boozer_kernel_contract.py`, after that module exists.
- [ ] `uv run pytest -q tests/jax_core/test_mps_boozer_kernel_contract.py`, after that test file exists.
- [ ] `uv run pytest -q tests/geo/test_boozer_mps_custom_kernel_parity.py -m mps`, after that MPS parity test exists.
- [ ] In `/Users/suhjungdae/code/opensource/jax-mps`: run the existing control-flow/custom-call tests plus new SIMSOPT custom-call tests.
- [ ] Run a standalone Biot-Savart custom-kernel benchmark with warmed execution and explicit synchronization.
- [ ] Run a fused Boozer value/grad benchmark with warmed execution and explicit synchronization.
- [ ] Run the existing single-stage parity harness for the reduced smoke fixture:

```bash
.conda/jax-mps/bin/python benchmarks/single_stage_init_parity.py \
  --platform auto \
  --output-json .artifacts/mps_custom_kernel/single_stage_maxiter3.json \
  --case-artifacts-dir .artifacts/mps_custom_kernel/cases \
  --optimizer-backend scipy-jax \
  --maxiter 3 \
  --mpol 2 \
  --ntor 2 \
  --benchmark-mode \
  --record-objective-evaluation-trace \
  --single-stage-case-timeout-seconds 1200 \
  --single-stage-target-case-timeout-seconds 1200
```

- [ ] Compare against `.artifacts/jax_mps_scipyjax_maxiter3_patched_withseed_20260530T025720Z/RUNTIME_FIX_DECISION.md`.
- [ ] Require `results.json` or a structured fail-loud rejection artifact from the target lane; a timeout is not acceptable.
- [ ] Verify that initial objective and gradient are finite and match the CPU oracle within the agreed tolerance.
- [ ] Verify that matched checkpoint timings improve for `target_lane_initial_objective_value_and_grad_returned` and `phase1_attempt_0_started`.
- [ ] Verify that jax-mps while-profile scalar predicate reads are removed or materially reduced for the custom path.

## Risks and Mitigations

- Risk: The custom kernel duplicates solver math and silently diverges from the JAX oracle.
  Mitigation: Build the flattened CPU oracle first, add parity tests per stage, and reject performance-only success without objective/gradient parity.

- Risk: MLX C++ does not expose enough custom Metal kernel API for the backend handler.
  Mitigation: Verify the C++ API in Phase 1; if missing, choose between an MLX extension module or direct Metal command encoding in the jax-mps plugin before implementing numerical kernels.

- Risk: A fused op is too large to debug.
  Mitigation: Stage through Biot-Savart and residual block kernels first, then fuse only after formula and layout parity are proven.

- Risk: The first kernel only accelerates Biot-Savart while the remaining Newton/GMRES loop still syncs to host.
  Mitigation: Treat Stage 1 and Stage 2 as validation ladders only; completion requires the Stage 3 fused solve/value-gradient op.

- Risk: Float32 MPS behavior changes convergence relative to CPU f64.
  Mitigation: Use explicit f32 CPU JAX parity for MPS acceptance, keep f64 CPU/CUDA comparisons as scientific reference, and record unsupported f64 MPS status.

- Risk: The custom path grows `control_flow.cc` into an unmaintainable backend module.
  Mitigation: Add a SIMSOPT custom-call handler module in jax-mps and keep `HandleCustomCall(...)` as dispatch glue.

- Risk: Experimental opt-in becomes another permanent backend mode.
  Mitigation: Keep the opt-in named experimental until parity and `maxiter=3` smoke gates pass; document removal or promotion criteria.

## Completion Criteria

- [ ] The plan's supported fixture has a serialized input/output contract artifact.
- [ ] CPU oracle parity passes against current `BoozerResidualJAX` value/grad.
- [ ] Stage 1 Biot-Savart custom kernel value and VJP parity pass.
- [ ] Stage 2 Boozer residual block parity passes.
- [ ] Stage 3 fused solve/value-gradient custom call passes objective and gradient parity.
- [ ] The custom path completes the MPS `scipy-jax` `maxiter=3`, `mpol=2`, `ntor=2` smoke without timeout.
- [ ] Matched MPS checkpoint timing improves against the previous patched artifact.
- [ ] The final report states exactly which shapes, dtypes, Boozer modes, and platforms are supported.
- [ ] The existing CPU JAX and CUDA-compatible paths remain unchanged by default.

## Open Questions

- Does the MLX C++ API expose a production-suitable equivalent of Python `mx.fast.metal_kernel(...)`, or does jax-mps need a direct Metal path for SIMSOPT kernels?
- Should the first fused op compute the full coil gradient internally, or should it expose forward solve state plus a paired backward custom call?
- What f32 objective/gradient tolerance should be accepted for MPS parity against CPU JAX f32?
- Which fixture shape should be the first promotion target after `mpol=2`, `ntor=2`: `mpol=4`, `ntor=4`, or the smallest CUDA-winning production shape?
- Is a custom call accepted as a local jax-mps patch, or should the long-term target be upstream MLX/jax-mps support for user-registered custom calls?
