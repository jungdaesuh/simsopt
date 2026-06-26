# L-BFGS-ondevice: Adopt Optax's GPU-Native Design

## Purpose

Make the in-tree `lbfgs-ondevice` lane more GPU-native by adopting the parts of
Optax's L-BFGS design that actually help on accelerators — a compact pure-pytree
optimizer-state contract, host control flow that routes on an explicit observable
(not reverse-communication internals), cached value/grad reuse, and an
unconstrained fast path that shrinks compiled graph breadth — **without** welding
the expensive objective into the optimizer graph (the welded path OOM'd at
~422 GiB compile-time) and **without** regressing SciPy L-BFGS-B `OptimizeResult`
/ callback-stop parity.

This plan supports execution and review of that change set. It supersedes the
informal "Stage 1/3" framing discussed in session and must be kept current
against the live checkout before implementation work is declared complete.

## Goals

- Host driver routes the stepwise loop on an **explicit compact step observable**
  returned by the macro-step kernel (`terminal`, `accepted_new_x`, `status`,
  `nfev`/`njev`, `f`, projected-gradient norm), instead of reading
  `state.workspace.task[0]` internals on the host.
- An **unconstrained two-loop-recursion direction fast path**, selected when
  `bounds is None`, that bypasses the L-BFGS-B Cauchy/`formk`/`subsm`/Cholesky
  machinery and reduces compiled-graph breadth, while producing the same search
  direction as the L-BFGS-B path (numerically verified).
- **Value/grad reuse** across line-search trial points and outer steps (Optax
  `value_and_grad_from_state` analog), cutting redundant objective evaluations.
- SciPy parity preserved: `status` codes (incl. `99` callback-stop, `6`
  nonfinite), `OptimizeResult` fields, and per-iteration callback semantics are
  byte/behavior-identical to the current lane on existing fixtures.

## Non-Goals

- **Not** making the outer loop fully on-device. The stepwise driver remains a
  host `while` loop; Stage 1 makes its hand-off compact and explicit, it does not
  eliminate host involvement. (Verified: `_lbfgsb_stepwise_driver` loops
  `while not host_status.terminal` on the host.)
- **Not** welding the objective (inner Boozer solve + IFT adjoint + polish) into
  the optimizer graph. That is the `monolithic_debug` path and is the documented
  ~422 GiB compile-time OOM cause; it stays gated/debug-only.
- **Not** replacing the host callback with `jax.debug.callback`. A debug callback
  cannot preserve SciPy `StopIteration` control semantics (clean stop + status 99
  + best-x); the real host `try/except` callback stays.
- **Not** touching the separate `optax-lbfgs-ondevice` lane (real `optax.lbfgs`)
  or the host Nocedal core in `optimizer_host_lbfgs.py`.
- **Not** changing bound-constrained behavior: when real `bounds` are passed, the
  L-BFGS-B path is unchanged.

## Current Context

Verified against the current working tree in this review (file:line references
may drift as the plan is implemented):

- **Default run mode is `stepwise`, not monolithic.** `optimizer.py:6067` and
  `:6111` both default `run_mode=options.get("lbfgs_run_mode", "stepwise")`.
  Monolithic is explicitly gated at `_lbfgs.py:736`
  (`if run_mode == _LBFGS_RUN_MODE_MONOLITHIC_DEBUG`).
- **Stepwise host loop now routes on an explicit macro-step observable.**
  `_lbfgsb_stepwise_host_status` reads only `step_result.entry_kind` and
  `step_result.terminal`, and `_lbfgsb_stepwise_driver` is seeded with an explicit
  initial entry route (`START` for normal starts, `SEARCH` for seeded
  `initial_value_and_grad`). The host loop no longer reads
  `state.workspace.task` for routing; `workspace.task` remains kernel/result
  payload state.
- **Macro-step kernels return the compact observable.**
  `LbfgsbMacroStepResult` now carries `state`, `accepted_new_x`, `terminal`,
  `entry_kind`, `status`, `nfev`, `njev`, `f`, and `proj_grad_norm`.
  `lbfgsb_macro_step_result` populates the observable and the macro-step
  `lax.while_loop` conditions use `result.terminal`, not
  `result.state.workspace.task[0]`.
- **On-device convergence math already exists.** `_lbfgsb_setulb_new_x_reentry`
  (`_lbfgsb_scipy.py:1478`): `gradient_converged = sbgnrm <= state.pgtol` (:1482),
  `reduction_converged` (:1484).
- **Public route is unconstrained.** `lbfgs-ondevice` passes `bounds=None`
  (`_lbfgs.py:139`), yet the Cauchy/subspace machinery is still in the compiled
  path: `lbfgsb_formk:2684`, `lbfgsb_cauchy:3069`, `lbfgsb_cmprlb:3371`,
  `lbfgsb_subsm:3452`. The `cnstnd` flag (`LbfgsbActiveResult`,
  `_lbfgsb_scipy.py:105/109`) already gates a `cmprlb` unconstrained `r=−g`
  branch.
- **Callback-stop is a real host exception path.** `_lbfgs.py:623` `except
  StopIteration:` → `callback_stop_state(state)` (:626) →
  `_lbfgsb_callback_stop_state_kernel` (:284) sets `task=(STOP, STOP_CALLB)`;
  `_lbfgsb_public_status` maps to `LBFGS_STATUS_CALLBACK_STOP = 99`
  (`_types.py:15`).
- **Result contract.** `_LBFGSResults` (`_types.py:99`) carries
  `s_history`/`y_history`/`rho_history`/`gamma`/`hess_inv_*`/`task`; converted to
  SciPy `OptimizeResult` downstream.
- **Compile-blowup framing (doc).**
  `docs/single_stage_compile_blowup_fix_implementation_plan.md`: the crash is
  "host-side XLA/LLVM code emission … section-memory allocation failure during
  code emission at ~422 GiB MaxRSS"; root cause = "monolithic `jit(run)` breadth"
  (fused outer + inner-solve + adjoint + polish). This is distinct from the
  separate 50-min construction wall that was an `O(jaxpr²)` host perf bug, already
  fixed in commit `23464a0da`.
- **Optax anchors (read-only reference repo).** `lbfgs(memory_size=10)`
  `alias.py:2655`; `ScaleByLBFGSState` `transform.py:1533`; two-loop `lax.scan`
  direction `transform.py:1558–1634`; `value_and_grad_from_state`
  `linesearch.py:1641`.
- **Existing test surfaces to extend.** `tests/geo/test_lbfgsb_scipy_jax_kernels.py`
  (kernel-level parity), `tests/test_lbfgs_ondevice_compile_shape.py` +
  `benchmarks/lbfgs_ondevice_compile_shape.py` (stepwise vs monolithic compile
  counts / graph-shape), `tests/geo/test_optimizer_jax_reference.py`.

## Requirements Review Status

Reviewed against the current working tree. The document remains a forward plan:
Stage 1's host-routing slice is implemented locally, while Stage 2, Stage 3, full
CPU regression, and CUDA/A100 evidence remain open.

- **Stage 1 host routing is implemented in the current working tree.**
  `LbfgsbMacroStepResult` carries the compact observable; the host driver routes
  on `entry_kind`/`terminal`; and the focused source plus runtime tests cover the
  no-host-`workspace.task` boundary and observable fidelity.
- **Stage 2 is not implemented.** `_lbfgsb_evaluate_value_and_grad` still calls
  `value_and_grad(state.x)` for every `FG` request. The existing
  `initial_value_and_grad` support seeds the first evaluation only; it is not an
  Optax-style reusable line-search/outer-step cache.
- **Stage 3 is not implemented.** There is no `lbfgsb_two_loop_direction`
  function in the private JAX L-BFGS-B path, and the unconstrained public lane
  still compiles through the L-BFGS-B Cauchy/subspace machinery.
- **Donation is an investigation, not a current requirement.** The live test
  `test_lbfgsb_step_kernel_omits_partial_state_donation` asserts
  `donate_argnums is None`; adding donation first requires proving the full
  macro-step carry is single-owner and updating that test deliberately.
- **Current local validation is CPU-only.** Local checks passed for `py_compile`,
  `ruff`, `tests/geo/test_lbfgsb_scipy_jax_kernels.py`, the compile-shape unit,
  and a focused L-BFGS selector covering the explicit observable, host-boundary
  source gate, callback-stop, maxfun, maxls, and hess-inverse rollover. No
  CUDA/A100 artifact is available for this review, so GPU claims remain
  design-level until a CUDA validation artifact is produced.

## Rationale

The GPU-relevant content of Optax's design is (1) control flow over a small pure
pytree state with no host round-trip on *control variables*, and (2) the
expensive objective compiled once and reused, never inlined into the optimizer
graph. The current lane already host-drives (good — avoids the welding OOM) but
hands off through reverse-communication internals and recompiles the full
bound-constrained subspace machinery even though every public call is
unconstrained.

- **Stage 1 (explicit observable)** is low-risk and is the precise, correct
  version of the in-session "Stage 1": it does not claim to remove host looping;
  it makes the hand-off an explicit contract so the host stops reaching into
  `workspace.task`, which also unlocks cleaner status/counters and is a
  prerequisite for any future on-device convergence predicate.
- **Stage 3 (unconstrained two-loop)** is the higher-leverage compile-memory win:
  it directly removes graph **breadth** (Cauchy/`formk`/`subsm`/Cholesky) on the
  only path the public lane uses, and the two representations are mathematically
  dual (`theta = ⟨y,y⟩/⟨s,y⟩ = 1/gamma`), so direction parity is provable and
  testable.
- **Stage 2 (value/grad reuse)** attacks the dominant cost (objective evals) and
  mirrors Optax's `value_and_grad_from_state` exactly.

Welding everything on-device (the naive "adopt Optax = one `lax.while_loop`"
reading) is explicitly rejected: the repo already has that path and it OOMs.

## Assumptions

- The L-BFGS-B `cmprlb` unconstrained branch (`r=−g` when `~cnstnd`) plus `subsm`
  with no active bounds is the exact analog of the Optax two-loop direction on
  identical `(s,y)` history. *(Direction-parity test in Stage 3 must confirm; if
  it fails, Stage 3 is descoped to a graph-shape-only refactor.)*
- No production caller depends on reading `state.workspace.task` *through* the
  stepwise driver's host status beyond terminal/entry-kind routing. *(Grep for
  external `workspace.task` access in Validation.)*
- Tests run under a repo JAX env with `JAX_ENABLE_X64=True` and `simsoptpp`
  importable (float64 required by the L-BFGS-B port). Current local validation
  used `/opt/homebrew/Caskroom/miniforge/base/bin/python` on Python 3.13. Do not
  hard-code a py3.11 path until the CI/runtime convention is confirmed.
- The compile-shape benchmark's stepwise/monolithic counters
  (`optimizer_control_monolithic_full_run_compile`) are the agreed metric for the
  graph-breadth regression check.

## Implementation Plan

1. **Stage 1 — Explicit compact step observable (host routing contract)**
   - [x] Extend `LbfgsbMacroStepResult` (`_lbfgsb_scipy.py:140`) with explicit
         observable fields: `terminal: bool`, `status: int32` (public status, not
         raw warnflag), `entry_kind`/next-routing hint, `nfev`, `njev`, `f`, and
         `proj_grad_norm` (reuse `dsave[12]`/`sbgnrm`). Populate them where the
         macro-step result is constructed (`:1686`, `:1748`, and the
         `continue_condition` sites `:1743`, `:1859`).
   - [x] Add a small pure helper `lbfgsb_macro_step_result(...)` computing
         `terminal = task[0] >= CONVERGENCE`,
         `entry_kind`, and the public status via the same logic as
         `_lbfgsb_public_status` (`_lbfgs.py:404`) so host and kernel agree.
   - [x] Rewrite `_lbfgsb_stepwise_host_status` (`_lbfgs.py:576`) /
         `_lbfgsb_stepwise_driver` (`:592`, loop `:603`) to route on the returned
         observable fields instead of
         `host_array(state.workspace.task, dtype=np.int32)[0]` or any other host
         read of `workspace.task`. The host must read only the explicit
         observable (one small struct), never `workspace.task` directly.
   - [x] Preserve the callback path exactly: keep `accepted_step_callback`,
         `try/except StopIteration` (`:623`), and `callback_stop_state` (`:626`).
         Drive the callback off `step_result.accepted_new_x` as today.
   - [x] Keep `_LBFGSResults` / `OptimizeResult` conversion unchanged; only the
         host↔kernel routing payload changes.

2. **Stage 2 — Value/grad reuse (Optax `value_and_grad_from_state` analog)**
   - [ ] Identify redundant objective evaluations across line-search trial points
         and outer-step reentry in `lbfgsb_reenter_new_x` / the dcsrch loop.
   - [ ] Thread the last accepted `(f, g)` through the macro-step so the next
         step / line-search reuses it instead of recomputing, matching Optax's
         cached-state contract (`linesearch.py:1641`).
   - [ ] Assert eval-count parity-or-reduction: `nfev`/`njev` must not increase
         vs the current lane on the fixtures; reductions are the win.

3. **Stage 3 — Unconstrained two-loop fast path (`bounds is None`)**
   - [ ] Add `lbfgsb_two_loop_direction(state)` implementing the Nocedal two-loop
         recursion over the stored `(ws, wy)` history with `gamma = ⟨s,y⟩/⟨y,y⟩`
         (Optax `transform.py:1558–1634`), returning `d = −H g`.
   - [ ] Gate it: when `~cnstnd` (no active bound; guaranteed under `bounds=None`,
         `_lbfgs.py:139`), compute the direction via the two-loop path and skip
         `lbfgsb_cauchy`/`lbfgsb_formk`/`lbfgsb_subsm`/Cholesky. Keep the full
         L-BFGS-B path when `cnstnd`.
   - [ ] Ensure the gated branch removes those ops from the compiled graph for the
         unconstrained lane (verify via the compile-shape benchmark, not by
         inspection alone).
   - [ ] Keep line search, convergence tests, history update, and result payload
         identical — only the direction computation is swapped.

4. **Cross-cutting**
   - [ ] Investigate `donate_argnums` for the macro-step carry where safe (state
         is single-consumer per host iteration). Promote it to an implementation
         requirement only after a test proves no partial-state aliasing hazard;
         the current test suite intentionally asserts no donation.
   - [ ] Confirm float64 invariants preserved throughout (no silent x32 demotion).
   - [ ] Update `optimizer.py` docstrings (~:71–:89) to describe the explicit
         observable contract and the unconstrained fast path.

## Validation Plan

- [x] **SciPy result parity (Stage 1, gate):** run
      `tests/geo/test_lbfgsb_scipy_jax_kernels.py` and the optimizer reference
      `tests/geo/test_optimizer_jax_reference.py`; `x`, `fun`, `jac`, `nit`,
      `nfev`, `njev`, `status`, `success`, `message` must match the prior lane.
      `PYTHONPATH=src JAX_ENABLE_X64=True <repo-jax-python> -m pytest
      tests/geo/test_lbfgsb_scipy_jax_kernels.py tests/geo/test_optimizer_jax_reference.py -q`
- [x] **Callback-stop parity (Stage 1):** a test where a user callback raises
      `StopIteration` returns `status == 99` with the current best `x` (mirror the
      `a8957b8ea` parity assertion). Confirm against `scipy.optimize.minimize`
      L-BFGS-B with the same callback.
- [x] **Nonfinite parity (Stage 1):** NaN/Inf in `f`/`g` still yields `status == 6`.
- [x] **No host access to `workspace.task` outside the kernel (Stage 1):**
      `rg -n "workspace\\.task" src/simsopt_jax/geo/optimizers/` shows no host
      driver reads after the refactor (only kernel-internal use).
- [x] **Transfer-guard boundary smoke (Stage 1):** run the focused
      `lbfgs_stepwise_driver_host_reads_use_host_boundary_helpers` source gate
      plus the callback-stop fixture under `jax.transfer_guard("disallow")`.
      The same focused selector also covers maxfun, maxls, and hess-inverse
      parity. This catches the rejected `state.workspace.task[0]`
      host-scalar-indexing variant.
- [ ] **Direction parity (Stage 3, gate):** new test asserting
      `lbfgsb_two_loop_direction` and the L-BFGS-B `cmprlb`→`subsm` direction agree
      to ~1e-10 (float64) on randomized unconstrained `(s,y)` histories of varying
      `m`; include the `count==0`/empty-history and single-pair cases.
- [ ] **End-to-end result parity (Stage 3):** unconstrained problems converge to
      the same `x`/`fun`/`status` as the current L-BFGS-B path within tolerance.
- [ ] **Graph-breadth regression (Stage 3):** run
      `benchmarks/lbfgs_ondevice_compile_shape.py`; assert the unconstrained
      stepwise path no longer compiles Cauchy/`formk`/`subsm` fragments and total
      `text_bytes`/`jaxpr_text_bytes` drop vs baseline; `monolithic` count stays 0.
      Extend `tests/test_lbfgs_ondevice_compile_shape.py` with the assertion.
- [ ] **Eval-count check (Stage 2):** `nfev`/`njev` do not regress on the
      reference fixtures.
- [ ] **Full CPU regression:** project unit suite green under `<repo-jax-python>`
      (`JAX_ENABLE_X64=True`, CPU).

Current local validation evidence for the Stage 1 slice:

- `PYTHONPATH=src /opt/homebrew/Caskroom/miniforge/base/bin/python -m py_compile
  src/simsopt_jax/geo/optimizers/private/_lbfgs.py
  src/simsopt_jax/geo/optimizers/private/_lbfgsb_scipy.py
  tests/geo/test_boozersurface_jax_private.py` passed.
- `PYTHONPATH=src /opt/homebrew/Caskroom/miniforge/base/bin/python -m ruff check
  src/simsopt_jax/geo/optimizers/private/_lbfgs.py
  src/simsopt_jax/geo/optimizers/private/_lbfgsb_scipy.py
  tests/geo/test_boozersurface_jax_private.py` passed.
- Focused private L-BFGS selector covering explicit observable, host-boundary
  source gate, seeded transition, callback-stop, maxfun, maxls, and hess-inverse
  rollover: `7 passed, 106 deselected`.
- Nonfinite selector:
  `2 passed, 111 deselected`.
- `tests/geo/test_lbfgsb_scipy_jax_kernels.py`:
  `55 passed`.
- `tests/geo/test_optimizer_jax_reference.py`:
  `4 passed`.
- `tests/test_lbfgs_ondevice_compile_shape.py`:
  `1 passed`.

Additional current validation evidence from the requirements review loop:

- `PYTHONPATH=src /opt/homebrew/Caskroom/miniforge/base/bin/python -m py_compile
  tests/subprocess/jax_runtime_cases.py tests/test_jax_import_smoke.py
  tests/geo/test_boozersurface_jax_private.py` passed.
- `PYTHONPATH=src /opt/homebrew/Caskroom/miniforge/base/bin/python -m ruff check
  tests/subprocess/jax_runtime_cases.py tests/test_jax_import_smoke.py
  tests/geo/test_boozersurface_jax_private.py` passed.
- Focused diagnostic/Optax-distinct selector:
  `2 passed, 111 deselected`.
- Focused subprocess compile-count selector:
  `3 passed, 92 deselected`.
- The compile-count subprocess payload now reports both
  `stepwise_compile_count` and `monolithic_compile_count`, and the smoke asserts
  `monolithic_compile_count == 0` for public and target value/grad
  `lbfgs-ondevice` compile-reuse cases.

## Risks and Mitigations

- Risk: Extending `LbfgsbMacroStepResult` perturbs the cached-kernel signatures
  and forces recompiles / breaks pytree assumptions in `_cached_private_solver`.
  Mitigation: keep the struct flat and statically-shaped; run the compile-shape
  benchmark to confirm stepwise compile count stays at the expected small number
  (per the compile-shape artifact baseline) and monolithic stays 0.
- Risk: Two-loop direction diverges from L-BFGS-B beyond rounding (e.g.
  first-step `gamma` cap, pair-skip threshold mismatch), failing direction parity.
  Mitigation: Stage 3 is gated behind the direction-parity test; if it fails,
  descope to graph-shape-only and keep L-BFGS-B direction, or align the
  first-step/skip heuristics explicitly and document the intended difference.
- Risk: Value/grad reuse changes `nfev` accounting and breaks SciPy parity on
  evaluation counts.
  Mitigation: assert eval-count parity-or-reduction; if SciPy parity requires the
  exact count, make reuse opt-in and off by default in the parity lane.
- Risk: Hidden external dependence on the host driver reading `workspace.task`.
  Mitigation: the grep gate above; if found, expose the needed field on the
  explicit observable rather than reverting.
- Risk: Stage 1 mistaken for a GPU latency win on the single-stage lane.
  Mitigation: documented Non-Goal — for the expensive objective the dominant host
  cost is the objective's own `device_put` volume (Track D), not the macro-step
  hand-off; Stage 1's win is contract clarity + cheap-objective lanes.

## Completion Criteria

- [x] Stepwise host driver routes solely on the explicit step observable; no
      `workspace.task` host reads remain.
- [x] SciPy `OptimizeResult`, callback-stop (99), and nonfinite (6) parity tests
      pass unchanged.
- [ ] Unconstrained lane uses the two-loop direction, passes direction-parity and
      end-to-end result-parity tests, and shows a measured graph-breadth /
      compile-byte reduction in the compile-shape benchmark.
- [ ] Value/grad reuse lands with non-regressing `nfev`/`njev` (or is gated off in
      the strict parity lane).
- [ ] Full CPU regression suite green; `optimizer.py` docstrings updated.
- [ ] CUDA/A100 validation artifact captured before making GPU-performance or
      GPU-memory claims beyond design-level reasoning.

## Open Questions

- Should the strict SciPy-parity lane keep value/grad reuse OFF (to match SciPy
  `nfev` exactly) and enable it only on a fast lane? Owner: maintainer decision.
- Do we want an optional on-device convergence predicate (true Optax
  `continuing_criterion`) as a Stage 4 follow-on for cheap-objective lanes, or is
  the explicit-observable host loop sufficient? Owner: maintainer decision.
- Confirm the canonical test interpreter/env path for CI (`<repo-jax-python>`) to
  lock the Validation commands. Owner: needs repo-convention confirmation.
