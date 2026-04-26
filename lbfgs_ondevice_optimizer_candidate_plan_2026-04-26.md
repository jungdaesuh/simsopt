# L-BFGS On-Device Optimizer Candidate Plan

Date: 2026-04-26

## Context

The production optimizer target is full JAX/GPU execution for Stage 2 and
single-stage optimization while preserving behavioral parity with upstream
SIMSOPT where parity is expected.

Upstream SIMSOPT delegates the relevant limited-memory quasi-Newton behavior to
SciPy `minimize(..., method="L-BFGS-B")`. That makes SciPy L-BFGS-B the
executable parity oracle, not the production GPU implementation.

The local JAX production entrypoint is `target_minimize(...,
method="lbfgs-ondevice")`. This route already owns the JAX-native compiled
optimizer loop, explicit value-and-gradient objectives, pytree support, cached
solver lowering, Stage 2 wiring, and single-stage target-lane wiring.

The only external candidates in scope are Optimistix LBFGS and Optax LBFGS.
Both are maintained JAX-native optimizer libraries, but neither is a direct
SciPy `L-BFGS-B` replacement. They must therefore be evaluated as candidates,
not adopted by default.

No candidate may be added as a silent fallback or a second production behavior.
Each candidate should be evaluated in a private bakeoff harness and either
promoted behind the existing `lbfgs-ondevice` contract or deleted.

## Architecture Decision

- Production SSOT: `method="lbfgs-ondevice"`.
- Production oracle: upstream SIMSOPT plus SciPy L-BFGS-B behavior.
- Candidate experiment: Optimistix LBFGS and Optax LBFGS in a private,
  non-public bakeoff.
- Public API stability: do not expose a second optimizer method for any
  candidate.
- No fallback policy: if a lane fails, it fails. Do not silently reroute to
  SciPy, Optimistix, Optax, or the custom implementation.
- Promotion policy: a candidate can replace the private custom implementation
  only if it matches parity, memory, performance, maintenance, and API
  requirements. If promoted, delete the replaced custom implementation path.

## Risks And Dependencies

1. Optimistix is a maintained solver-style JAX optimization library with an
   LBFGS candidate, but it is not a SciPy `L-BFGS-B` drop-in. It must prove
   status, stopping, line-search, memory, and value-and-gradient compatibility.
2. Optax is maintained and includes `optax.lbfgs`, but not LBFGSB. It is an
   optimizer-transformation API, not a SciPy `OptimizeResult` replacement. Any
   Optax candidate must prove status, stopping, line-search, and result-schema
   parity explicitly.
3. Tolerance norm convention is a silent parity risk. SciPy L-BFGS-B uses the
   projected-gradient infinity norm for `gtol`. Optimistix and Optax candidate
   bakeoff runs must make the norm convention explicit and must not compare
   candidates with the same numeric tolerance unless the norm conversion is part
   of the test.
4. Curvature handling is a production parity risk. The current in-house solver
   accepts finite, non-stalled, line-search-successful steps but skips the
   correction-pair update strictly when curvature is invalid. Optimistix and
   Optax candidates must match this behavior or prove better parity against the
   SciPy oracle. The bakeoff must include indefinite-Hessian and
   invalid-curvature fixtures.
5. Phase 1 contains two audit/cleanup items rather than confirmed behavior
   bugs: `ftol` status 4 is already mapped to success, and the current
   `valid_curvature` retry trigger is dominated by the invalid-step log writer.
   Workers should preserve or simplify these paths, not invent new behavior.

## Source Contracts

- Upstream SIMSOPT Boozer solve:
  `/Users/suhjungdae/code/opensource/simsopt/src/simsopt/geo/boozersurface.py`
  calls SciPy `L-BFGS-B` when `limited_memory=True`.
- Local target optimizer API:
  `src/simsopt/geo/optimizer_jax.py::target_minimize`.
- Local custom implementation:
  `src/simsopt/geo/optimizer_jax_private/_lbfgs.py`.
- Local result conversion:
  `src/simsopt/geo/optimizer_jax_private/_result_converters.py`.
- Reference adapter:
  `src/simsopt/geo/optimizer_jax_reference.py`.
- Stage 2 route:
  `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py`.
- Single-stage route:
  `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`.

## Official Behavior To Match

### SciPy L-BFGS-B

- `maxcor`: number of limited-memory correction pairs.
- `gtol`: projected-gradient convergence threshold.
- `ftol`: relative objective reduction:

  ```text
  (f_k - f_{k+1}) / max(abs(f_k), abs(f_{k+1}), 1) <= ftol
  ```

- `maxls`: maximum line-search steps per iteration.
- `tol` in SciPy `minimize` maps to both `ftol` and `gtol` for L-BFGS-B when
  those options are not explicitly supplied.
- SciPy status/success and messages are the parity target for reference-lane
  behavior where the local JAX API claims SciPy equivalence.

### Original L-BFGS-B Algorithm

- Limited-memory quasi-Newton method with stored correction pairs instead of a
  dense Hessian.
- Line-search-driven updates.
- Bound constraints are core to L-BFGS-B, but the current JAX production lane
  mostly uses unconstrained or transformed coordinates. Native bound support is
  out of scope for this candidate set unless a maintained candidate adds it and
  passes the same parity gates.

### Optimistix And Optax

- Optimistix LBFGS is a maintained JAX-native solver-style candidate.
- Optax LBFGS is a maintained JAX-native candidate but has no LBFGSB variant
  and uses the Optax transformation API.
- Candidates exist only until they prove parity, maintenance viability, and
  performance.
- No production adoption unless a candidate can replace the current
  implementation without adding a permanent dual-path abstraction.

Candidate matrix:

| Candidate | Maintained | LBFGS | Box constraints | JIT-clean | Main parity risk |
| --- | --- | --- | --- | --- | --- |
| Current in-house `lbfgs-ondevice` | yes | yes | transformed/unconstrained only | yes | SciPy-relative `ftol` patch still needed |
| Optimistix `LBFGS` | yes | yes | no | yes | status/stopping semantics and SciPy parity |
| Optax `lbfgs` | yes | yes | no | yes | API/status schema and stopping semantics |

## Non-Goals

- Do not rewrite upstream SIMSOPT.
- Do not route production JAX/GPU solves through SciPy.
- Do not keep custom L-BFGS and any candidate as two permanent production
  solvers.
- Do not add defensive fallbacks.
- Do not add broad optimizer abstractions before a measured winner exists.
- Do not introduce dense-Hessian production behavior for large Stage 2 or
  single-stage lanes.

## Requirements

- [ ] Preserve public production method name: `lbfgs-ondevice`.
- [ ] Preserve upstream SciPy L-BFGS-B as parity oracle.
- [ ] Keep JAX target solves JIT-compatible and GPU-resident.
- [ ] Keep memory complexity O(`maxcor * dim`) for L-BFGS.
- [ ] Support explicit `(value, grad)` objective calls.
- [ ] Support pytree parameters without spreading flatten/unflatten logic.
- [ ] Preserve transfer-guard-clean target execution.
- [ ] Preserve Stage 2 and single-stage optimizer contracts.
- [ ] Use one status/success mapping SSOT.
- [ ] Use the existing parity tolerance ladder where applicable.
- [ ] Define and use an explicit `optimizer_state_parity` lane before accepting
  any replacement candidate.
- [ ] Make tolerance norm conventions explicit in every candidate comparison.
- [ ] Add no runtime fallback path.

## Phase 1: Finish The Custom Production Solver

Goal: make the current `lbfgs-ondevice` path the clean production baseline
before comparing candidates.

### Implementation Tasks

- [ ] Patch `ftol` semantics in
  `src/simsopt/geo/optimizer_jax_private/_lbfgs.py`.
  - Replace absolute `state.f_k - f_kp1 < ftol` with SciPy relative reduction.
  - Use `max(abs(f_k), abs(f_kp1), 1)` as denominator.
  - Use SciPy's `<= ftol` relation, not strict `< ftol`.
  - Keep dtype handling inside JAX arrays.

- [ ] Audit and preserve `ftol` status and success behavior.
  - Status 4 is already included in `_LBFGS_SUCCESS_STATUSES`.
  - Preserve status 4 as successful `ftol` termination after the relative
    reduction patch.
  - Keep the status code/message in one converter table.

- [ ] Keep invalid-curvature step acceptance.
  - Accept finite, non-stalled, line-search-successful steps.
  - Skip only the L-BFGS correction-pair update when curvature is invalid.
  - Do not record invalid curvature alone as a failed step.

- [ ] Remove redundant `valid_curvature` retry trigger in single-stage retry
  logic.
  - Retry only failed line search, nonfinite step, or stalled nonconverged step.
  - `valid_curvature=False` is dominated by the invalid-step log writer today:
    accepted steps do not write invalid-step events.
  - Treat this as cleanup and contract clarification, not a behavior change.

- [ ] Audit dense BFGS only for proven parity bugs.
  - Dense BFGS already keeps the step and skips the Hessian update when
    curvature is invalid.
  - Confirm no status or stopping mismatch before changing code.
  - Do not widen dense BFGS usage.

- [ ] Audit status mapping.
  - Line-search failure.
  - Max iterations.
  - Max function evaluations.
  - Max gradient evaluations.
  - `ftol`.
  - `gtol`.
  - Nonfinite objective or gradient.

### Tests

- [ ] Unit: invalid curvature accepts step and advances `x_k`.
- [ ] Unit: invalid curvature does not update `s/y/rho` history.
- [ ] Unit: nonfinite trial step is rejected.
- [ ] Unit: stalled nonconverged step is rejected.
- [ ] Unit: line-search failure is rejected.
- [ ] Unit: SciPy-relative `ftol` matches a reference calculation.
- [ ] Unit: status 4 maps to success after the `ftol` patch.
- [ ] Unit: retry classification ignores `valid_curvature` when no failed-step
  cause exists.
- [ ] Unit: `tol`/`ftol`/`gtol` option handling matches the target contract.
- [ ] Unit: result converter maps status/success consistently.
- [ ] Integration: Stage 2 short L-BFGS-B parity.
- [ ] Integration: single-stage target-lane short optimizer parity.
- [ ] Runtime: repeated `lbfgs-ondevice` calls reuse compiled solver.
- [ ] Runtime: transfer guard remains clean.

### Acceptance Gate

- [ ] `tests/geo/test_boozersurface_jax_private.py` targeted optimizer tests pass.
- [ ] `tests/geo/test_single_stage_example.py` targeted retry/status tests pass.
- [ ] `tests/integration/test_stage2_jax.py` targeted optimizer parity tests pass.
- [ ] CPU JAX parity passes against SciPy oracle for the bakeoff fixture set.
- [ ] CUDA smoke passes for Stage 2 and single-stage short runs.

## Phase 2: Add Private Candidate Bakeoff Harness

Goal: evaluate Optimistix and Optax without creating a second production path.

### Implementation Tasks

- [ ] Add a private benchmark-only adapter for Optimistix LBFGS.
  - Keep it outside public `target_minimize` dispatch at first.
  - Keep the adapter small and removable.
  - Use typed inputs and outputs matching the existing optimizer result shape.
  - Map Optimistix solver output into the same bakeoff schema as the custom
    solver and SciPy oracle.

- [ ] Add a private benchmark-only adapter for Optax LBFGS.
  - Keep it outside public `target_minimize` dispatch at first.
  - Keep Optax's transformation API localized to the candidate adapter.
  - Map candidate outputs into the same bakeoff schema as other candidates.

- [ ] Evaluate Optimistix `LBFGS`.
  - Match the current unconstrained or transformed-coordinate production path.
  - Use explicit value-and-gradient objective calls.
  - Keep pytree flattening localized.
  - Test its stopping norm against SciPy infinity-norm stopping explicitly.

- [ ] Evaluate Optax `lbfgs` as the maintained candidate.
  - Use `optax.value_and_grad_from_state` or an equivalent local value/grad
    wrapper required by Optax line search.
  - Test status and stopping semantics explicitly because Optax does not return
    a SciPy-style `OptimizeResult`.
  - Reject production adoption if adapter glue becomes wider than the custom
    solver it would replace.

- [ ] Add a candidate-only result normalizer.
  - Map Optimistix and Optax state into the same measurement schema as SciPy
    and custom JAX.
  - Do not expose it as a public production converter until promotion.

- [ ] Add the optimizer parity lane to the ladder SSOT.
  - Add `optimizer_state_parity` to
    `benchmarks/validation_ladder_contract.py::OPTIMIZER_DRIFT_TOLERANCES`.
  - Proposed tolerances: `x_rtol=1e-6`, `x_atol=1e-8`,
    `objective_rel_tol=1e-6`, `gradient_rtol=1e-6`,
    `gradient_atol=1e-8`, `jac_norm_inf_abs_tol=1e-8`.
  - Require fixed seed, fixed initial state, equal `maxiter`, equal `maxcor`,
    explicit norm convention, and SciPy L-BFGS-B oracle output.

- [ ] Add bakeoff runner.
  - Inputs: objective fixture name, method candidate, seed, dtype, maxiter,
    maxcor, ftol, gtol, maxls.
  - Outputs: JSON with final `x`, `fun`, `jac_norm_inf`, `nit`, `nfev`,
    status, success, compile time, warm runtime, peak memory when available.

### Bakeoff Fixture Matrix

- [ ] Quadratic convex objective.
- [ ] Rosenbrock objective.
- [ ] Invalid-curvature synthetic objective.
- [ ] Nonfinite objective case.
- [ ] Stalled nonconverged step case.
- [ ] Line-search failure case.
- [ ] Indefinite-Hessian step at a known iterate.
- [ ] Stage 2 objective short run.
- [ ] Single-stage outer objective short run.

### Bakeoff Metrics

- [ ] Final `x` parity against SciPy/custom JAX.
- [ ] Final objective parity.
- [ ] Final gradient infinity norm parity.
- [ ] Iteration count compatibility.
- [ ] Function/gradient evaluation count compatibility where comparable.
- [ ] Status/success compatibility.
- [ ] Tolerance norm compatibility.
- [ ] Cold compile time.
- [ ] Warm runtime.
- [ ] Peak device memory.
- [ ] Host transfer behavior.
- [ ] CUDA execution success.

### Acceptance Gate

- [ ] Optimistix candidate matches SciPy oracle under `optimizer_state_parity`.
- [ ] Optax candidate matches SciPy oracle under `optimizer_state_parity`.
- [ ] Each candidate's tolerance norm convention is explicit and parity-tested.
- [ ] Winning candidate matches or beats custom JAX warm runtime.
- [ ] Winning candidate matches or beats custom JAX memory behavior.
- [ ] Winning candidate supports the required value-and-gradient API cleanly.
- [ ] Winning candidate satisfies maintenance requirements.
- [ ] Winning candidate does not add permanent abstraction complexity.

## Phase 3: Promotion Or Deletion Decision

Goal: avoid permanent dual implementations.

### Promote A Candidate If All Gates Pass

- [ ] Replace the internals behind `method="lbfgs-ondevice"` with the winning
  candidate.
- [ ] Preserve the public method name and optimizer contract.
- [ ] Delete the custom private L-BFGS solver implementation.
- [ ] Delete tests that only assert custom internal mechanics.
- [ ] Keep parity, status, Stage 2, single-stage, and transfer tests.
- [ ] Update docs to state which implementation backs `lbfgs-ondevice`.

### Delete Candidate Harness If No Candidate Clears All Hard Gates

- [ ] Keep the custom in-house solver as the only production implementation.
- [ ] Remove the candidate adapters.
- [ ] Remove candidate-only tests and bakeoff plumbing.
- [ ] Record the failing gate:
  - parity,
  - memory,
  - warm runtime,
  - compile behavior,
  - API mismatch,
  - status/success mismatch.

## Work Ordering

1. [ ] Patch custom `ftol` semantics.
2. [ ] Remove redundant `valid_curvature` retry trigger.
3. [ ] Add edge-case optimizer parity tests.
4. [ ] Run targeted local optimizer tests.
5. [ ] Run Stage 2 and single-stage short parity tests.
6. [ ] Add private candidate harness for Optimistix and Optax.
7. [ ] Run CPU bakeoff matrix.
8. [ ] Run CUDA bakeoff matrix.
9. [ ] Decide promote or delete.
10. [ ] Commit the chosen final state with no dead candidate path.

## Parallelizable Work

- [ ] Worker A: custom solver parity patch and unit tests.
- [ ] Worker B: single-stage retry/status audit.
- [ ] Worker C: Optimistix and Optax candidate harness.
- [ ] Worker D: bakeoff fixture runner and JSON schema.
- [ ] Worker E: CUDA/Runpod validation once CPU tests are green.

Dependencies:

- Worker B depends on accepted-step semantics from Worker A.
- Worker E depends on green local CPU tests.
- Promotion/deletion decision depends on Worker C and Worker D bakeoff data.

## Success Definition

This plan is complete when there is exactly one production JAX L-BFGS path:
`method="lbfgs-ondevice"`.

That path must:

- [ ] match upstream SIMSOPT/SciPy behavior where parity is required,
- [ ] run on-device for JAX/GPU target workflows,
- [ ] remain memory efficient for large Stage 2 and single-stage problems,
- [ ] preserve explicit value-and-gradient objective support,
- [ ] pass Stage 2 and single-stage e2e validation,
- [ ] avoid fallback paths and permanent duplicate solver behavior.
