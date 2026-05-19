# CPU/C++ vs JAX CPU Full Trajectory Parity Plan - 2026-04-28

## Target

Prove full production single-stage trajectory parity between:

- the existing CPU/C++ SciPy reference lane
- the JAX CPU target lane

This is a stricter target than fixed-state value/gradient parity. It requires
both lanes to start from the same production seed and compare the optimizer
behavior that SciPy actually exposes: initial objective, initial gradient,
accepted iterates, termination mode, final optimizer state, and final physics
metrics.

The CPU/C++ parity oracle is the legacy SciPy `L-BFGS-B` path over `JF.x`.
The parity run must not project the CPU lane into the JAX target-lane `bs.x`
coordinate contract. A shared host L-BFGS core is useful for target-lane
debugging, but it is not CPU/SciPy behavior.

## Pre-Implementation State

JAX CPU vs H100 full production trajectory parity was tracked through the
matched target-lane contract.

CPU/C++ vs JAX CPU full production trajectory parity is not proven yet.

The first blocker was the CPU/C++ optimizer contract. The CPU/C++
reference optimizer goes through SciPy:

- `src/simsopt/geo/optimizer_jax_reference.py`

SciPy only provides public callback visibility at accepted iterates. It does
not expose the same Wolfe trial/search-direction/internal line-search trace
that the JAX target L-BFGS path now records.

The second blocker was trace completeness. The JAX target trace recorded only
the first accepted step:

- `src/simsopt/geo/optimizer_jax_private/_lbfgs.py`
  initializes `optimizer_state_trace = ()` and only fills it under
  `if not optimizer_state_trace`.
- `benchmarks/single_stage_parity_matrix.py` compares `lhs_trace[0]` and
  `rhs_trace[0]`.

That meant the prior infrastructure could prove target-lane first-accepted-step
parity plus final metric proximity, not CPU/SciPy trajectory parity.

## Validated Code Review Findings

The plan must account for these current-tree facts before implementation:

- `src/simsopt/geo/optimizer_jax_private/_lbfgs.py` already contains a mostly
  host-dispatched NumPy L-BFGS loop, but it still calls
  `_require_private_optimizer_runtime()` and builds a cached JAX
  `value_and_grad` kernel. The shared host core must lift the JAX runtime gate
  and bypass the JAX kernel cache for CPU/C++ evaluators.
- `_REFERENCE_METHODS = {"bfgs", "lbfgs"}` routes to SciPy. That is the
  required CPU/C++ parity target.
- `lbfgs-trace` is not a CPU/SciPy parity lane if it uses the shared host
  L-BFGS core. It can remain a target-alignment diagnostic, but reports must
  not describe it as matching CPU/SciPy behavior.
- CPU/SciPy parity must use the legacy `JF.x` optimizer vector. The JAX
  target-lane `bs.x` vector is a different coordinate contract.
- `should_record_single_stage_outer_optimizer_progress(use_target_lane)` is
  target-lane-only today. CPU/SciPy parity can still compare final results and
  accepted callback-level data, but SciPy does not expose the same internal
  Wolfe trace as the private JAX L-BFGS path.
- `benchmarks/single_stage_parity_matrix.py` has a CPU progress input, but its
  lane labels and termination checks are still target-lane/SciPy-oriented.
- There is no production-state CPU/C++ vs JAX CPU value/gradient/Boozer
  operator artifact producer yet. Unit-scope CPU/JAX parity tests are not
  enough for the production trajectory claim.
- `_private_lbfgs_result_to_optimize_result()` is evaluator-agnostic at the
  `OptimizeResult` layer, but `_LBFGSResults` is JAX-array oriented. The host
  core should return a host result object; the JAX wrapper can convert through
  `_LBFGSResults`, while the CPU/C++ wrapper should return `OptimizeResult`
  directly with the same public fields.

## Non-Goals

- Do not change the default CPU/reference SciPy lane.
- Do not replace CPU/SciPy with a shared host L-BFGS implementation when the
  claim is CPU/SciPy parity.
- Do not project the CPU/SciPy reference lane from `JF.x` to `bs.x`.
- Do not weaken existing parity tolerances.
- Do not infer trajectory parity from final-state closeness.
- Do not use SciPy callback output as a substitute for the target-lane
  `optimizer_state_trace`.
- Do not add fallback behavior. This parity lane should either emit the full
  trace contract or fail its contract.

## Root Fix

For CPU/SciPy parity, preserve the existing CPU/C++ SciPy `L-BFGS-B` contract
and make the comparison harness line up around it:

- CPU/reference starts from the same init-only seed as JAX CPU.
- CPU/reference optimizes the legacy `JF.x` vector.
- CPU/reference runs `scipy.optimize.minimize(..., method="L-BFGS-B")`.
- JAX CPU is compared against the CPU/SciPy run as an implementation under
  test, not as the definition of the reference behavior.

SciPy does not expose its internal Wolfe trial/search-direction trace through
the public `minimize` API. Therefore the CPU/SciPy parity proof should compare
the strongest public contract available: seed, accepted iterates/callback
state where recorded, termination, final optimizer vector, and final physics
metrics. If exact Wolfe-internal trace is required, that is a separate
SciPy-instrumentation project, not a JAX-port parity shortcut.

The shared host L-BFGS split remains useful for the JAX target crash/memory
fix, but it is not the CPU/SciPy parity oracle.

The shared split should be:

```text
shared host L-BFGS/Wolfe control
  |
  +-- JAX CPU evaluator: cached JIT value_and_grad -> device_get
  |
  +-- optional CPU/C++ diagnostic evaluator, not CPU/SciPy parity
```

For CPU/SciPy parity, the optimizer loop is part of the reference behavior,
not a replaceable harness detail.

## Required Code Shape

1. Keep the CPU/reference optimizer route on SciPy:

   - `ReferenceOptimizerContract(method="lbfgs")`
   - `reference_minimize(..., method="lbfgs", value_and_grad=True)`
   - `_scipy_minimize_value_and_grad(...)`
   - `_scipy_dispatch(..., scipy_method="L-BFGS-B")`

2. Keep CPU/reference optimizer coordinates on the legacy Optimizable graph:

   - initialize with `dofs = JF.x`
   - write trial candidates with `JF.x = x`
   - return `JF.dJ()` in the same coordinate basis

   The CPU/SciPy reference lane must not use `bs.x`, even when the active
   banana-coil subset is 11 coordinates.

3. Use a shared init-only seed for full-run parity:

   - run a CPU init-only seed artifact
   - warm-start the CPU/SciPy full run from that seed
   - warm-start the JAX CPU full run from the same seed
   - never seed JAX CPU from the CPU/SciPy final state

4. Compare the strongest public SciPy contract:

   - initial objective/gradient
   - accepted callback state if recorded
   - final optimizer vector
   - termination status/message
   - final physics metrics and constraints

   Do not claim Wolfe-internal trace parity for SciPy unless SciPy itself is
   instrumented.

5. Keep the shared host L-BFGS core only for JAX target execution and optional
   diagnostics.

   A non-SciPy CPU/C++ evaluator using that host core is a target-alignment
   diagnostic. It must be labeled that way and excluded from CPU/SciPy parity
   claims.

## Implementation Status

Completed in this pass:

- Extracted the shared host L-BFGS/Wolfe control into:

  - `src/simsopt/geo/optimizer_host_lbfgs.py`

- Reduced the JAX target L-BFGS wrapper to:

  - cached JAX value-and-gradient kernel
  - explicit device-to-host evaluator boundary
  - shared host optimizer core
  - existing `_LBFGSResults` conversion

- Added an opt-in CPU/reference host-core trace diagnostic. This is not the
  CPU/SciPy parity oracle because it intentionally bypasses SciPy.
- Corrected the single-stage CPU/SciPy parity path so `lbfgs-trace` no longer
  implies the JAX `bs.x` coordinate contract. CPU/reference parity must use
  `JF.x`.
- Corrected the parity wrapper so any CPU full outer run uses a shared
  init-only seed instead of seeding JAX from the CPU final state.
- Extended private L-BFGS to record every accepted-step trace entry, not just
  the first accepted step.
- Extended `benchmarks/single_stage_parity_matrix.py` so optimizer trace
  comparisons cover every accepted-step entry and include
  `cpu_cpp_trace_vs_jax_cpu` for the diagnostic host-core lane.
- Added focused regression tests for:

  - no SciPy on the host-core diagnostic trace lane
  - initial value/gradient on the host-core diagnostic trace lane
  - full accepted-step trace recording
  - later-entry optimizer trace drift detection

Still open:

- Produce the production fixed-state CPU/C++ vs JAX CPU parity artifact
  described in Acceptance Ladder 3.
- Run matched production CPU/SciPy and JAX CPU artifacts through the parity
  matrix.
- If fixed-state physics parity fails, debug the Boozer/operator evaluator
  boundary before interpreting optimizer trajectory drift.

## Acceptance Ladder

### 1. Shared Host Optimizer Unit Contract

Pass deterministic tests on a quadratic objective:

- scalar objective path
- explicit value-and-gradient path
- zero-iteration budget
- converged seed with exact counters
- full accepted-step trace length for `maxiter > 1`
- first-step and later-step trace fields
- skipped curvature update does not poison history
- non-descent direction fails the L-BFGS contract
- final `OptimizeResult` fields match current target-lane semantics

### 2. Host-Core Diagnostic Wrapper Contract

Run a small CPU/C++ objective through the optional host-core diagnostic method
and verify:

- no SciPy call occurs
- `optimizer_state_trace` is non-empty after an accepted step
- trace entries contain the same fields as JAX target L-BFGS
- callback/progress payloads are host values
- default CPU `method="lbfgs"` still calls SciPy
- `method="lbfgs-trace"` succeeds when `scipy.optimize.minimize` is patched to
  raise
- `failure_callback` and `optimizer_initial_value_and_grad` are accepted only
  for the diagnostic trace lane

### 3. Fixed-State Production Physics Parity

Before trusting a trajectory comparison, compare CPU/C++ and JAX CPU at the
same production single-stage state:

- objective value
- gradient infinity norm
- objective components
- Boozer residual norm
- Boozer JVP or linearized residual action
- Boozer transpose/adjoint solve output

Add a producer script, for example:

```text
benchmarks/single_stage_cpu_cpp_jax_cpu_state_parity.py
```

It must emit a JSON artifact with CPU/C++ and JAX CPU values for the quantities
above, plus seed spec hash, equilibrium hash, git SHA, versions, and thread
settings. The parity matrix must consume this artifact instead of reusing the
JAX CPU vs H100 `jax_cpu_vs_h100_value_grad` field.

If this fails, trajectory parity cannot be expected yet; the root issue is in
the physics evaluator boundary, not the optimizer.

### 4. First Accepted Step Diagnostic Parity

Run the optional CPU/C++ host-core diagnostic and JAX CPU from the same
production seed with:

- `maxiter=1`
- same `ftol`, `gtol`, `maxcor`, `maxls`
- same seed/spec/equilibrium files

Pass criteria:

- initial objective and gradient match the fixed-state envelope
- search direction matches
- selected line-search step scale matches
- trial objective and trial gradient match
- line-search status matches
- line-search fallback/Wolfe classification matches
- accepted trial point matches

### 5. Full Production CPU/SciPy Parity

Run the full production single-stage CPU/SciPy lane and matched JAX CPU lane,
then compare with:

```bash
python benchmarks/single_stage_parity_matrix.py \
  --parity-report-json <matched-report.json> \
  --cpu-progress-json <cpu-scipy-run>/outer_optimizer_progress.json \
  --jax-cpu-progress-json <jax-cpu-run>/outer_optimizer_progress.json \
  --output-json .artifacts/parity/<cpu-cpp-vs-jax-cpu-full-trajectory>.json
```

Pass criteria:

- CPU/SciPy and JAX CPU start from the same init-only seed
- CPU/SciPy uses `JF.x` and SciPy `L-BFGS-B`
- termination status and message are compatible
- final objective and physics metrics are inside the fixed-state parity
  envelope
- runtime and memory are recorded for both lanes

Required parity-matrix updates:

- keep the CPU/SciPy lane label distinct from the host-core diagnostic lane
- keep `LANE_CPU_SCIPY = "cpu_scipy"` only for default SciPy reports
- compare every trace entry only for lanes that actually emit target-style
  optimizer traces
- include CPU/SciPy termination in termination compatibility checks
- reject or mark invalid comparisons when seed spec hash, equilibrium hash, or
  trace schema version differ

## Expected Failure Modes

1. The first frozen-state value/gradient differs.

   Root area: CPU/C++ vs JAX physics evaluator, likely Boozer residual,
   adjoint/operator solve, or objective component assembly.

2. Frozen-state value/gradient matches, but accepted-step trajectory differs.

   Root area: optimizer coordinate contract, SciPy/JAX optimizer policy, or
   initial state construction.

3. CPU/SciPy and JAX CPU terminate differently.

   Root area: line-search policy differences, objective evaluation counts, or
   trial-state mutation order.

4. First accepted step matches, but later trajectory diverges.

   Root area: state update, curvature history accounting, final refresh, or a
   backend-sensitive physics component that only appears after the first move.

## Practical Run Order

1. Keep CPU/SciPy parity on `JF.x` and SciPy `L-BFGS-B`.
2. Run focused optimizer unit tests.
3. Run existing JAX target L-BFGS tests to guard downstream regressions.
4. Generate the production-state CPU/C++ vs JAX CPU fixed-state artifact.
5. Generate CPU/SciPy `maxiter=1` artifact from a shared init-only seed.
6. Generate matched JAX CPU `maxiter=1` artifact.
7. Run the parity matrix on the first-step/final-metric pair.
8. Run full production CPU/SciPy.
9. Run matched JAX CPU.
10. Run the full parity matrix.
11. Record time, memory, artifacts, and parity deltas in a follow-up report.

## Release Meaning

This work makes the existing CPU/SciPy lane the parity oracle for the JAX port.
It does not mean SciPy and JAX expose the same optimizer-internal trace. It
means the migration claim is anchored to the behavior users already run:

- the same production seed
- the same legacy `JF.x` CPU/SciPy reference behavior
- comparable termination/final physics metrics
- clearly separated target-lane diagnostics when optimizer-control drift must
  be investigated

That is the missing piece for a defensible CPU/C++ vs JAX CPU full production
single-stage parity claim.
