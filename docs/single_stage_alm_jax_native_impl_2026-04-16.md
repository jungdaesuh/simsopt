# Single-Stage ALM JAX-Native Implementation Plan

Date: 2026-04-16

Audience: GPT-5.4 xhigh implementing the remaining single-stage ALM JAX-native port in `simsopt-jax`.

## Goal

Make the single-stage ALM lane in
[`examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`](/Users/suhjungdae/code/columbia/simsopt-jax/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:1)
truly JAX-native for `backend=jax` and `optimizer_backend=ondevice`, while preserving:

- the CPU/reference SciPy lane
- the JAX-on-CPU parity lane
- the existing Stage 2 ALM behavior

This is not an ALM algorithm rewrite. The algorithm is already present. The remaining work is backend routing and removal of host-bound inner-loop seams.

## Read First

- [x] The single-stage outer loop already resolves an ondevice optimizer contract via
  [resolve_single_stage_optimizer_contract()](/Users/suhjungdae/code/columbia/simsopt-jax/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:5793).
- [x] The ALM branch currently does **not** pass `inner_optimizer_contract` into
  [minimize_alm(...)](/Users/suhjungdae/code/columbia/simsopt-jax/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:8284).
- [x] `minimize_alm()` already supports `TargetOptimizerContract` for inner solves in
  [alm_utils.py](/Users/suhjungdae/code/columbia/simsopt-jax/examples/single_stage_optimization/alm_utils.py:768).
- [x] The current target inner solve still routes every objective/gradient evaluation through
  [`jax.pure_callback`](/Users/suhjungdae/code/columbia/simsopt-jax/examples/single_stage_optimization/alm_utils.py:758).
- [x] The existing host/object ALM evaluation path is
  [evaluate_single_stage_alm_problem()](/Users/suhjungdae/code/columbia/simsopt-jax/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:4157).
- [x] The CPU/reference lane is intentionally preserved by optimizer contract policy in
  [optimizer_jax.py](/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/optimizer_jax.py:543).
- [x] JAX-on-CPU parity is a supported runtime mode in
  [runtime.py](/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/backend/runtime.py:82).
- [x] The right reuse seam for a pure JAX objective bundle is
  [make_traceable_objective_runtime_bundle()](/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/surfaceobjectives_jax.py:2982).

## External Constraints

These are not optional style opinions. They constrain the implementation strategy.

- JAX external callback docs describe `pure_callback` as a host-side callback mechanism, so a hot ALM inner loop that uses it is not fully native device execution.
  Source: https://docs.jax.dev/en/latest/external-callbacks.html
- JAX `jax.scipy.optimize.minimize` currently supports only `"BFGS"`, so it is not a drop-in replacement for the current bounded `L-BFGS-B` style inner solve.
  Source: https://docs.jax.dev/en/latest/_autosummary/jax.scipy.optimize.minimize.html
- CUDA best-practice guidance is to keep the hot path on device and minimize host/device transfers.
  Source: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- ALM literature says the core algorithm is an outer multiplier/penalty loop around repeated subproblem solves, which means the clean port is to replace the backend of the inner subproblem, not redesign ALM.
  Source: https://pages.cs.wisc.edu/~swright/nd2016/IMA_augmentedLagrangian.pdf
- Open-source JAX optimization ecosystems already distinguish pure-JAX bounded solvers from SciPy-backed wrappers.
  Source: https://jaxopt.github.io/stable/_autosummary/jaxopt.LBFGSB.html

## Scope Checklist

Required for this landing:

- [ ] Single-stage ALM passes the target inner optimizer contract when `backend=jax` and `optimizer_backend=ondevice`.
- [ ] Single-stage ALM has a pure-JAX native evaluation path with no `jax.pure_callback` in the native inner loop.
- [ ] CPU/reference SciPy ALM still works unchanged.
- [ ] JAX CPU parity lane still works with the same native ALM path, just on CPU.
- [ ] Tests explicitly prove the native lane does not hit SciPy minimize.
- [ ] Tests explicitly prove the native lane JAXPR does not contain `pure_callback`.

Not required for this landing:

- [ ] A new ALM algorithm
- [ ] Removing the SciPy fallback globally
- [ ] Real CUDA certification in this environment
- [ ] Reworking Stage 2 unless code reuse requires a small shared helper extraction

## Current Code Facts

### Single-stage ALM entry and gap

- Outer optimizer contract is resolved at
  [single_stage_banana_example.py:7993](/Users/suhjungdae/code/columbia/simsopt-jax/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:7993).
- The ALM branch constructs `evaluate_problem(...)` and calls
  [minimize_alm(...)](/Users/suhjungdae/code/columbia/simsopt-jax/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:8284)
  without `inner_optimizer_contract`.

### Existing ALM backend split

- `minimize_alm()` falls back to SciPy `minimize(..., method="L-BFGS-B")` at
  [alm_utils.py:1893](/Users/suhjungdae/code/columbia/simsopt-jax/examples/single_stage_optimization/alm_utils.py:1893)
  when `target_inner_optimizer is None`.
- `minimize_alm()` already routes to `_run_target_inner_solve(...)` when a valid `TargetOptimizerContract` is present at
  [alm_utils.py:1904](/Users/suhjungdae/code/columbia/simsopt-jax/examples/single_stage_optimization/alm_utils.py:1904).
- The target contract validation and method restriction already exist at
  [alm_utils.py:768](/Users/suhjungdae/code/columbia/simsopt-jax/examples/single_stage_optimization/alm_utils.py:768).

### Native blocker

- `_build_target_inner_value_and_grad(...)` currently wraps host evaluation with
  [`jax.pure_callback`](/Users/suhjungdae/code/columbia/simsopt-jax/examples/single_stage_optimization/alm_utils.py:758),
  so the present target inner solve is only optimizer-ondevice, not objective-native.

### Host-bound ALM objective path

- `evaluate_single_stage_alm_problem(...)` at
  [single_stage_banana_example.py:4157](/Users/suhjungdae/code/columbia/simsopt-jax/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:4157)
  still uses host/object state:
  - `JF.x = candidate_x`
  - `boozer_surface.run_code(...)`
  - `update_self_intersection_status(...)`
  - `JCurveCurve.shortest_distance()`
  - `JCurveSurface.shortest_distance()`
  - `JSurfSurf.shortest_distance()`
  - `banana_curve.kappa()`
  - `curvelength.J()`
  - `banana_current.get_value()`

### Existing target runtime pattern to reuse

- The pure target-lane objective infrastructure already exists in
  [surfaceobjectives_jax.py:2982](/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/surfaceobjectives_jax.py:2982)
  as `make_traceable_objective_runtime_bundle(...)`.

### Stage 2 as the wiring example

- Stage 2 already threads `inner_optimizer_contract` into ALM:
  - wrapper: [banana_coil_solver.py:2142](/Users/suhjungdae/code/columbia/simsopt-jax/examples/single_stage_optimization/STAGE_2/banana_coil_solver.py:2142)
  - dispatch selection: [banana_coil_solver.py:2765](/Users/suhjungdae/code/columbia/simsopt-jax/examples/single_stage_optimization/STAGE_2/banana_coil_solver.py:2765)

## Architectural Decision

Use a two-lane ALM design:

1. Reference lane
   - `backend=cpu`
   - `optimizer_backend=scipy`
   - keep the current host/object evaluation path
   - keep SciPy `L-BFGS-B`

2. Native lane
   - `backend=jax`
   - `optimizer_backend=ondevice`
   - use a pure-JAX ALM runtime bundle
   - use `TargetOptimizerContract`
   - do not allow `pure_callback` in the inner objective path

This preserves the intended contracts encoded in
[optimizer_jax.py:543](/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/optimizer_jax.py:543)
and
[optimizer_jax.py:571](/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/optimizer_jax.py:571).

## Implementation Phases

### Phase 1: Contract plumbing

Objective:

- make single-stage ALM choose the existing ondevice inner optimizer when the lane is JAX/ondevice

Tasks:

- [ ] Add single-stage helper logic mirroring Stage 2:
  - resolve `alm_inner_optimizer_contract`
  - set it only when `args.backend == "jax"` and `args.optimizer_backend == "ondevice"`
  - otherwise leave it `None`
- [ ] Pass `inner_optimizer_contract=alm_inner_optimizer_contract` into
  `minimize_alm(...)`
- [ ] Keep the CPU/reference path untouched

Likely files:

- [ ] [single_stage_banana_example.py](/Users/suhjungdae/code/columbia/simsopt-jax/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:7993)

Acceptance:

- [ ] JAX/ondevice ALM no longer selects SciPy because of missing wiring
- [ ] CPU/reference ALM still uses SciPy

Important note:

This phase alone is not enough for true native execution. It only moves the optimizer backend, not the objective evaluation backend.

### Phase 2: Extract a native single-stage ALM runtime bundle

Objective:

- build a pure-JAX `value_and_grad` path for the ALM subproblem

Tasks:

- [ ] Define a new runtime-bundle builder for single-stage ALM, preferably adjacent to or layered on top of the existing target runtime bundle in
  [surfaceobjectives_jax.py](/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/surfaceobjectives_jax.py:2982)
- [ ] Make the bundle return at minimum:
  - base objective value
  - base objective gradient
  - constraint vector
  - augmented-Lagrangian scalar
  - augmented-Lagrangian gradient
  - reporting metrics needed by accepted-step/final-state logic
- [ ] Keep any host wrappers explicit and optional, matching the current runtime-bundle design pattern

Design rule:

- do not mutate `Optimizable` objects or `run_dict` inside the compiled objective

### Phase 3: Split host/reference ALM evaluation from native ALM evaluation

Objective:

- preserve current behavior for CPU/reference users while enabling a native target lane

Tasks:

- [ ] Keep
  [evaluate_single_stage_alm_problem()](/Users/suhjungdae/code/columbia/simsopt-jax/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:4157)
  as the host/reference path
- [ ] Add a native path builder or dispatcher for the JAX/ondevice lane
- [ ] Make the ALM branch choose:
  - host/reference evaluator for CPU/SciPy
  - native runtime bundle for JAX/ondevice

Implementation constraint:

- do not mix host-only side effects into the native evaluator

### Phase 4: Remove `pure_callback` from the native ALM inner loop

Objective:

- ensure the JAX/ondevice ALM inner loop is actually JAX-native

Tasks:

- [ ] Refactor `_build_target_inner_value_and_grad(...)` in
  [alm_utils.py](/Users/suhjungdae/code/columbia/simsopt-jax/examples/single_stage_optimization/alm_utils.py:719)
  so native callers can provide a pure-JAX `value_and_grad` directly
- [ ] Keep the callback-based path only for the host/reference compatibility lane if still needed
- [ ] Ensure the native path passed into `_run_target_inner_solve(...)` no longer uses
  `jax.pure_callback`

Design options:

- preferred:
  - make `_run_target_inner_solve(...)` accept either:
    - a pure JAX callable, or
    - the old host bridge callable
- acceptable:
  - add a separate native target-inner-solve helper if keeping both code paths in one function becomes too tangled

### Phase 5: Move diagnostics and artifact shaping off the hot path

Objective:

- keep the compiled inner loop pure and move side effects to explicit host boundaries

Tasks:

- [ ] Remove `run_dict` mutation from the native evaluator
- [ ] Do not call file writers, JSON emitters, or artifact helpers from the native objective
- [ ] If per-accept reporting is needed, derive metrics from the runtime bundle result and host-format them outside the compiled solve
- [ ] Reuse the current explicit reporting boundary pattern used in the traceable outer objective lane

Non-goal:

- do not try to make plotting or VTK export JIT-compatible

### Phase 6: Native constraint kernel closeout

Objective:

- ensure all constraint terms used by single-stage ALM are available in the native evaluator

Tasks:

- [ ] Audit which terms already have JAX-traceable kernels and reuse them
- [ ] Port missing pieces for:
  - coil-length upper bound
  - banana-current upper bound
  - curve-curve minimum distance
  - curve-surface minimum distance
  - surface-vessel minimum distance
  - curvature upper bound
- [ ] Keep host-only helpers only in the reference lane

Practical rule:

- if a term only exists today as an object method like `.J()` or `.shortest_distance()`, it is suspect until there is a traceable equivalent

## File-Level Change Checklist

### `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`

- [ ] Add `alm_inner_optimizer_contract` selection near the existing outer contract setup
- [ ] Thread `inner_optimizer_contract` into `minimize_alm(...)`
- [ ] Add native ALM runtime bundle construction for JAX/ondevice mode
- [ ] Split evaluator dispatch between reference and native lanes
- [ ] Keep accepted-step and final-report logic outside the native hot path

### `examples/single_stage_optimization/alm_utils.py`

- [ ] Preserve existing SciPy fallback path
- [ ] Add native path support without `jax.pure_callback`
- [ ] Keep trust-radius and continuation logic shared if possible
- [ ] Avoid regressing Stage 2 ALM

### `src/simsopt/geo/surfaceobjectives_jax.py`

- [ ] Extend or factor the existing runtime bundle to support ALM-native single-stage needs
- [ ] Do not duplicate target-lane geometry/objective machinery if existing traced kernels already provide the same information

### Tests

- [ ] Add or extend single-stage ALM integration tests
- [ ] Add a no-SciPy assertion for JAX/ondevice ALM
- [ ] Add a no-`pure_callback` assertion for the native ALM JAXPR
- [ ] Keep CPU/reference regression coverage

## Validation Plan

Run these after each phase, not only at the end.

### Phase 1 validation

- [ ] Targeted unit/integration test proving `inner_optimizer_contract` is passed on the JAX/ondevice lane
- [ ] Targeted unit/integration test proving SciPy is still used on the CPU/reference lane

### Native path validation

- [ ] Add a test that monkeypatches SciPy `minimize` to fail if called from single-stage ALM in JAX/ondevice mode
- [ ] Add a JAXPR assertion that the native ALM objective path contains no `pure_callback`
- [ ] Reuse the repo standard already enforced in:
  - [test_single_stage_jax_cpu_reference.py:6983](/Users/suhjungdae/code/columbia/simsopt-jax/tests/integration/test_single_stage_jax_cpu_reference.py:6983)
  - [test_stage2_jax.py:4316](/Users/suhjungdae/code/columbia/simsopt-jax/tests/integration/test_stage2_jax.py:4316)

### Regression validation

- [ ] `pytest -q tests/geo/test_single_stage_alm_integration.py`
- [ ] targeted `tests/geo/test_single_stage_example.py` slices covering ALM dispatch/state/reporting
- [ ] targeted `tests/integration/test_single_stage_jax_cpu_reference.py` slices for traceability and accepted-step parity
- [ ] targeted Stage 2 ALM regression slices if `alm_utils.py` changes shared behavior
- [ ] `python -m py_compile` on touched files
- [ ] `git diff --check`

### Final external gate

- [ ] real CUDA smoke run on `backend=jax`, `optimizer_backend=ondevice`, `constraint_method=alm`
- [ ] certify actual run recipe on hardware

## Implementation Order

Do this order unless you find a local blocker:

1. [ ] Wire `inner_optimizer_contract` into single-stage ALM.
2. [ ] Add a regression proving JAX/ondevice ALM no longer falls into SciPy because of missing dispatch.
3. [ ] Build the native ALM runtime bundle.
4. [ ] Switch the JAX/ondevice ALM lane off the host evaluator and off `pure_callback`.
5. [ ] Add no-`pure_callback` JAXPR coverage.
6. [ ] Run targeted single-stage and Stage 2 regressions.
7. [ ] Leave CUDA certification as the last step.

## Failure Modes To Avoid

- [ ] Do not delete the SciPy fallback from `alm_utils.py` globally.
- [ ] Do not break `backend=cpu, optimizer_backend=scipy`.
- [ ] Do not assume GPU-only hardware; `jax_cpu_parity` must still be valid.
- [ ] Do not move file I/O or plotting into traced code.
- [ ] Do not add a second ALM algorithm implementation unless absolutely necessary.
- [ ] Do not introduce a native path that still secretly depends on `pure_callback`.
- [ ] Do not re-enter mutable `Optimizable` state from the native hot path.

## Definition Of Done

All must be true:

- [ ] Single-stage ALM on `backend=jax`, `optimizer_backend=ondevice` uses `TargetOptimizerContract`
- [ ] Single-stage native ALM inner objective path contains no `pure_callback`
- [ ] Single-stage native ALM lane does not call SciPy `minimize`
- [ ] CPU/reference lane still works unchanged
- [ ] JAX CPU parity lane still works
- [ ] Targeted regressions pass
- [ ] Real CUDA validation remains the only open external gate

## Notes For The Implementing Model

- Prefer extracting a small shared helper over duplicating Stage 2 dispatch logic.
- Reuse the existing traceable runtime-bundle pattern instead of inventing a second ad hoc JAX objective contract.
- Keep the write scope focused. This is a backend-boundary repair, not a campaign-policy refactor.
- If a constraint lacks a traceable kernel, isolate that as the next blocker instead of papering over it with another callback seam.
