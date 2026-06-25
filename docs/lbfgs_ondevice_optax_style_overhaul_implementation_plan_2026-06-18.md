# L-BFGS On-Device Optax-Style Overhaul Implementation Plan

## Purpose

Patch `method="lbfgs-ondevice"` so its execution architecture looks like Optax:
explicit optimizer state, bounded history buffers, small compiled update kernels,
and explicit host/device boundaries.

The goal is not to replace `lbfgs-ondevice` with `optax.lbfgs`. The current method
is a SciPy-compatible JAX L-BFGS-B implementation. Optax provides the better JAX
optimizer architecture, but not the same L-BFGS-B public contract.

## Goals

- [ ] Preserve the public SciPy L-BFGS-B behavior of `method="lbfgs-ondevice"`:
  `maxls`, `maxcor`, `ftol`, `gtol`, `maxfun`, status mapping, counters, seeded
  value/grad behavior, callbacks, and `hess_inv`.
- [ ] Replace the current monolithic main-solver compile boundary with
  Optax-style `init -> step -> host driver -> result conversion` structure.
- [ ] Keep L-BFGS history and workspace memory explicit, bounded, and visible in
  state sizing diagnostics.
- [ ] Make every host/device transfer intentional: input staging at the boundary,
  scalar loop/result reads through explicit host-boundary helpers, and no hidden
  `device_get` or host callback in the normal step kernel.
- [ ] Add characterization tests before changing routing, then switch the public
  route only after parity and transfer-boundary checks pass.

## Non-Goals

- [ ] Do not change `lbfgs-ondevice` into plain Optax L-BFGS. Optax does not carry
  the SciPy L-BFGS-B reverse-communication contract or `hess_inv` result surface.
- [ ] Do not change the existing `method="optax-lbfgs-ondevice"` lane except where
  shared tests expose an option/default mismatch.
- [ ] Do not solve unrelated full-objective compile blowups by hiding them behind
  this optimizer refactor. This plan targets the optimizer control boundary.
- [ ] Do not weaken the strict transfer-guard tests by removing them. If a host
  read is required, it must be small, explicit, and covered by tests.

## Current Context

Confirmed facts:

- `lbfgs-ondevice` is documented as a target private JAX L-BFGS method, while
  `lbfgs-scipy-jax` is the host SciPy L-BFGS-B control method and
  `optax-lbfgs-ondevice` is a separate target quasi-Newton lane
  (`src/simsopt_jax/geo/optimizers/optimizer.py:69-86`).
- The public target route maps `lbfgs-ondevice` to private `_minimize_lbfgs_private`
  or `_minimize_lbfgs_private_value_and_grad`, defaulting `maxcor` to `10` and
  `maxls` to `20` (`src/simsopt_jax/geo/optimizers/optimizer.py:6021-6107`).
- The current private implementation builds separate jitted kernels for initial
  state, seeded first value/grad, and a main L-BFGS-B solver kernel. The main
  kernel wraps the whole `lbfgsb_mainlb(...)` run in `jax.jit`
  (`src/simsopt_jax/geo/optimizers/private/_lbfgs.py:145-179`,
  `:413-542`).
- The SciPy-compatible workspace is already bounded by `n` and `m`: floating
  workspace size is `2*m*n + 5*n + 11*m*m + 8*m`, and integer workspace size is
  `3*n` (`src/simsopt_jax/geo/optimizers/private/_lbfgsb_scipy.py:258-363`).
- The low-level L-BFGS-B loop is reverse-communication shaped: `lbfgsb_mainlb`
  repeatedly runs `lbfgsb_setulb`, evaluates value/grad on `FG`, emits
  `NEW_X` callbacks, and stops on convergence/status (`_lbfgsb_scipy.py:1559-1685`).
- Existing tests assert no host-core or SciPy fallback, SciPy-style `maxls`,
  SciPy public-result parity, `hess_inv`, optional optimizer trace allocation,
  strict transfer-guard behavior, seeded value/grad parity, and repeated-call
  stability (`tests/geo/test_boozersurface_jax_private.py:1697-2047`).
- Optax `lbfgs` at local repo `/Users/suhjungdae/code/opensource/optax`, HEAD
  `3205908`, defaults `memory_size=10` and warns that the optimizer is memory
  intensive (`optax/_src/alias.py:2655-2764`).
- Optax's useful design shape is stateful transformation, not magic memory
  elimination: `ScaleByLBFGSState` stores count, params, updates, parameter
  difference memory, update difference memory, and weights memory
  (`optax/_src/transform.py:1533-1555`).
- Optax preallocates stacked history leaves with shape
  `(memory_size,) + leaf.shape` (`optax/_src/transform.py:1720-1738`) and applies
  the two-loop recursion with `jax.lax.scan` over the bounded memory
  (`optax/_src/transform.py:1598-1634`).
- Optax zoom line search is also a bounded state update around
  `jax.value_and_grad(value_fn)` and `jax.lax.while_loop`
  (`optax/_src/linesearch.py:1597-1618`).
- The repo's compile-blowup notes identify the bad class of failures as
  monolithic compilation breadth, not L-BFGS history data alone
  (`docs/single_stage_compile_blowup_fix_implementation_plan.md:45-69`,
  `docs/single_stage_ondevice_compile_blowup_root_cause_2026-06-16.md:28-44`).
- Context7 documentation for `/google-deepmind/optax` describes Optax L-BFGS as
  a gradient transformation with explicit state: initialize state, compute grads,
  call `optimizer.update(...)`, then `optax.apply_updates(...)`.

## Rationale

Optax is the better architectural model for this JAX/GPU lane because it exposes
an optimizer as a small state transition. That can give XLA a compact optimizer
kernel to compile and lets the Python driver keep control-flow, callbacks,
metadata, and stopping checks outside the optimizer graph. This does not by
itself fix a separately monolithic objective graph.

The current `lbfgs-ondevice` problem is not that L-BFGS history is unbounded.
Both Optax and this repo allocate memory proportional to `memory_size * n`. The
problem is the compile boundary: the current main private kernel stages the full
solver run as one jitted loop. Matching Optax means shrinking that boundary while
retaining the existing SciPy-compatible L-BFGS-B state machine.

The safest design is therefore an Optax-style internal API over the existing
SciPy-compatible kernels:

```text
init_state(x0, options) -> LbfgsbState
step(value_and_grad, state, limits) -> LbfgsbState
run_stepwise(step, state, options, observers) -> LbfgsbState
convert_result(final_state, history) -> OptimizeResult
```

## Assumptions

- [ ] `lbfgs-ondevice` should remain the SciPy-compatible L-BFGS-B lane.
- [ ] Scalar host synchronization once per accepted step or terminal check is
  acceptable if it is explicit and tested.
- [ ] Array materialization during callbacks and final result conversion is
  acceptable only at explicit host boundaries.
- [ ] The existing `optax-lbfgs-ondevice` lane remains useful as an architectural
  reference and unconstrained optimizer option, not as a parity oracle for
  L-BFGS-B.
- [ ] GPU promotion requires compile-size and transfer-boundary evidence, not only
  unit-test parity on small CPU quadratics.

## Implementation Plan

### Phase 1: Lock the Existing Contract

- [ ] Add a short design note near the private L-BFGS code explaining that
  `lbfgs-ondevice` is SciPy-compatible L-BFGS-B, while the overhaul only copies
  Optax's state/update architecture.
- [ ] Treat the existing characterization tests as the routing gate before
  changing behavior. Current coverage already includes `maxiter=0` deferred-stop
  parity, SciPy public-result parity, `hess_inv`, trace default/budget behavior,
  strict transfer-guard behavior with callbacks, seeded value/grad parity,
  repeated-call stability, and nonfinite initial value/gradient status behavior
  (`tests/geo/test_boozersurface_jax_private.py:1762-2150`).
- [ ] Add only the missing characterization gaps:
  - [ ] `maxfun` parity on a line-search-heavy function.
  - [ ] callback/progress event count remains tied to accepted `NEW_X` count after
        the macro-step split.
  - [ ] compile/cache diagnostics distinguish old monolithic main-solver compile
        from new macro-step compile.
- [ ] Add an explicit test that `lbfgs-ondevice` and `optax-lbfgs-ondevice` are not
  expected to produce identical trajectories or statuses.

### Phase 2: Introduce an Optax-Style Private API

- [ ] In `src/simsopt_jax/geo/optimizers/private/_lbfgsb_scipy.py`, extract the
  body of `lbfgsb_mainlb` into a pure transition helper:

```text
lbfgsb_transition(value_and_grad, state, *, maxiter, maxfun) -> LbfgsbState
```

- [ ] Add a macro-step helper that advances from the current observable state to
  the next accepted `NEW_X` or terminal state:

```text
lbfgsb_advance_to_next_observable(value_and_grad, state, *, maxiter, maxfun)
```

  This helper should keep line-search reverse communication inside JAX, so the
  host driver does not inspect every `FG` request.

- [ ] Keep the existing `lbfgsb_mainlb(...)` as a compatibility wrapper over the
  new transition helper until the stepwise route is promoted.
- [ ] Do not add new behavior-changing guards to the line search. Use existing
  `maxls`, task/status, `maxiter`, and `maxfun` semantics.

### Phase 3: Split the Compiled Kernels

- [ ] In `src/simsopt_jax/geo/optimizers/private/_lbfgs.py`, replace the single
  `_lbfgsb_mainlb_kernel(...)` call path with cached kernels for:
  - [ ] `init_state`.
  - [ ] optional `start_with_initial_value_and_grad`.
  - [ ] `advance_to_next_observable`.
  - [ ] `extract_history` / result payload preparation.
- [ ] Compile the macro-step kernel with `jax.jit`, but keep the host loop outside
  that kernel:

```text
state = init_state_jit(x0)
while not host_terminal(state) and host_iterations(state) can continue:
    state = step_jit(state)
final = result_payload_jit(state)
```

- [ ] Use `simsopt_jax.runtime.host_boundary.host_bool`, `host_int`, and
  `host_array` for every loop predicate and final materialization instead of raw
  `np.asarray(...)` or `jax.device_get(...)` in the new driver.
- [ ] Keep callback/progress behavior out of the normal no-observer kernel. For
  observer mode, either use an observer-specific step kernel or materialize the
  accepted-step payload through explicit host-boundary helpers after the macro-step
  returns.

### Phase 4: Make Memory Sizing First-Class

- [ ] Add a private utility that reports L-BFGS-B workspace bytes from `n`,
  `maxcor`, dtype, integer workspace, and optional trace budget.
- [ ] Surface a diagnostic event before compilation with:
  - [ ] `n`.
  - [ ] `maxcor`.
  - [ ] `maxls`.
  - [ ] estimated workspace bytes.
  - [ ] whether callbacks or optimizer trace are enabled.
- [ ] Keep `lbfgs-ondevice` default `maxcor=10`; do not inherit the current
  `OptaxLBFGSOptions.memory_size=200` default.
- [ ] Separately review `OptaxLBFGSOptions.memory_size=200`
  (`src/simsopt_jax/solve/optax/contracts.py:15-22`) because it is far larger
  than Optax's local default of `10`.

### Phase 5: Route Public `lbfgs-ondevice` Through the Step Driver

- [ ] Add an internal option, for example `options={"lbfgs_run_mode": "stepwise"}`,
  while the old monolithic wrapper remains available for parity debugging.
- [ ] Run the current `target_minimize(..., method="lbfgs-ondevice")` route through
  the stepwise driver after parity tests pass.
- [ ] Keep result conversion through `_private_lbfgs_result_to_optimize_result`
  so `hess_inv`, counters, status, and messages stay unchanged.
- [ ] Preserve the seeded value/grad fast path. The seed should satisfy only the
  first SciPy `FG_START` request, matching the existing tests.
- [ ] Once production validation passes, invert the option so stepwise is default
  and monolithic is debug-only or removed.

### Phase 6: Transfer-Boundary Hardening

- [ ] Add tests that monkeypatch or grep the new stepwise path to reject raw
  `jax.device_get`, raw `np.asarray(jax.Array)`, and unguarded host conversion.
- [ ] Run no-callback and callback paths under `jax.transfer_guard("disallow")`.
- [ ] For callback paths, assert only explicit callback payload materialization
  crosses to the host.
- [ ] Add a GPU-only transfer-guard test mirroring
  `test_optax_lbfgs_gpu_closure_constants_run_under_strict_transfer_guard`, but
  targeting `method="lbfgs-ondevice"`.

### Phase 7: Compile and Memory Promotion

- [ ] Capture a before/after compile diagnostic on a small deterministic quadratic:
  old monolithic main solver versus new macro-step kernel.
- [ ] Capture StableHLO or jaxpr size for:
  - [ ] init kernel.
  - [ ] step kernel.
  - [ ] result payload kernel.
  - [ ] old monolithic kernel, while it still exists.
- [ ] Run the relevant Boozer limited-memory path with compile logging enabled and
  confirm the optimizer control kernel no longer owns a full-solve compilation.
- [ ] Promote only after CPU and CUDA runs prove bounded host RSS, no hidden
  transfer-guard failures, and stable repeated-call behavior.

## Validation Plan

- [ ] Run these commands from an environment with the compiled native extension
  available as `from simsoptpp import Curve`. The repo test bootstrap imports
  local `simsopt` through `tests/conftest.py`, and collection fails before test
  selection if that symbol is missing.
- [ ] Run focused private L-BFGS tests:

```bash
python -m pytest tests/geo/test_boozersurface_jax_private.py -k "lbfgs_ondevice or minimize_lbfgs_private"
```

- [ ] Run the low-level SciPy L-BFGS-B kernel parity suite:

```bash
python -m pytest tests/geo/test_lbfgsb_scipy_jax_kernels.py
```

- [ ] Run Optax/transfer-boundary tests because the target design intentionally
  copies that architecture:

```bash
python -m pytest tests/jax/solve/test_value_grad_contract.py -k "optax_lbfgs or transfer_guard"
python -m pytest tests/jax/solve/test_optimizer_result_schema.py
```

- [ ] Run a strict-transfer guard smoke for public `lbfgs-ondevice` with:
  - [ ] numpy `x0`.
  - [ ] device `x0`.
  - [ ] explicit value/grad.
  - [ ] callback and no-callback modes.
- [ ] Run a GPU compile-memory diagnostic on the target lane and record:
  - [ ] peak host RSS.
  - [ ] compile wall time.
  - [ ] number of compiled executables.
  - [ ] whether the step kernel recompiles across repeated calls.
- [ ] Compare public outputs against SciPy on pinned toy problems, not against
  Optax:
  - [ ] final `x`.
  - [ ] `fun`.
  - [ ] `jac`.
  - [ ] `nit`.
  - [ ] `nfev`.
  - [ ] `njev`.
  - [ ] `status`.
  - [ ] `success`.

## Risks and Mitigations

- [ ] Risk: splitting the loop changes L-BFGS-B reverse-communication timing.
  Mitigation: macro-step to `NEW_X` or terminal, not per-`FG` host stepping, and
  parity tests for counters/status.
- [ ] Risk: host loop introduces many device-to-host synchronizations.
  Mitigation: synchronize only on scalar observable state per accepted step, and
  keep line-search `FG` cycling inside the jitted macro-step.
- [ ] Risk: callbacks force a separate compile path or hidden transfers.
  Mitigation: keep observer and no-observer kernels separate, with explicit
  transfer-boundary tests.
- [ ] Risk: memory claims regress into overclaims. Optax also stores
  `memory_size * n` history.
  Mitigation: report workspace bytes and compile-memory separately.
- [ ] Risk: default history size accidentally follows the public Optax lane's
  `memory_size=200`.
  Mitigation: keep `lbfgs-ondevice` at `maxcor=10` unless a benchmark proves a
  larger value is worth the memory.
- [ ] Risk: preserving `hess_inv` requires final history extraction that pulls
  arrays to host.
  Mitigation: perform extraction only during result conversion, not inside the
  iterative step kernel.

## Completion Criteria

- [ ] `lbfgs-ondevice` defaults to the stepwise private driver.
- [ ] The old monolithic main-solver kernel is removed or gated behind an explicit
  debug-only option.
- [ ] All existing `lbfgs-ondevice` public behavior tests pass.
- [ ] New tests prove no SciPy fallback, no raw hidden host transfers, and stable
  repeated-call cache behavior.
- [ ] CPU and CUDA diagnostics show the optimizer control boundary no longer
  produces a monolithic full-run compile.
- [ ] Documentation states the exact difference between `lbfgs-ondevice`,
  `lbfgs-scipy-jax`, and `optax-lbfgs-ondevice`.

## Open Questions

- [ ] Should the first merged patch expose `lbfgs_run_mode="stepwise"` only for
  private testing, or switch public `lbfgs-ondevice` immediately after local
  parity passes?
- [ ] Is a scalar host sync once per accepted step acceptable for the target
  production lane, or does the lane require a fully jitted run mode for some
  benchmark?
- [ ] Should `OptaxLBFGSOptions.memory_size` be reduced from `200` to `10`, or is
  that public lane intentionally tuned away from upstream Optax defaults?
- [ ] What is the required GPU promotion artifact: unit GPU test, single-stage
  smoke, or full production-shaped compile diagnostic?
- [ ] Should observer/callback support remain inside JAX via `jax.debug.callback`,
  or move to explicit host-boundary materialization after each macro-step?
