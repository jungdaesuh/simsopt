# scipy-jax-decomposed Newton Polish and Final Reporting Reuse Report

Date: 2026-07-01

Repository: `/Users/suhjungdae/code/columbia/simopt-jax-clean-local`

Scope: single-stage banana optimization with JAX target lane, especially
`lbfgs-scipy-jax-decomposed`.

## Executive Summary

The diagnosed performance problem was not primarily that the JAX kernels were
uncompiled, running on CPU, or using the wrong linear solver. The diagnostic
evidence pointed to two higher-level target-lane workflow problems:

1. Full Newton polish was paid on line-search trial candidates, including
   candidates that are rejected or already show failure/stall symptoms.
2. Final/reporting synchronization could recompute target-lane forward state
   instead of reusing the phase-2 K1 forward result that was already computed
   for the final optimizer point.

These are expensive because `scipy-jax-decomposed` is intentionally split into
host SciPy control plus JAX target-lane kernels:

- K1: solve the Boozer surface / produce a forward result.
- K2: evaluate value and gradient from the solved state.
- Reporting sync: produce final metrics, accepted-state data, and artifacts.

The split is good, but it makes reuse boundaries explicit. If K1 is run during
the optimizer and then final reporting runs another K1 for the same final DOFs,
the JAX lane pays duplicate expensive solve work that the native CPU path avoids
structurally through mutable `BoozerSurface` state.

The follow-up patch addresses both workflow boundaries in the default
`scipy-jax-decomposed` path. The focused dense-adjoint Perlmutter smoke at
commit `118803140155` proved final-sync reuse of the exact-DOF solved-state
payload and clean process exit, but it also exposed a missed runtime wiring
boundary: the run was configured with the trial-only `skip` policy, yet K1
events still reported `newton_polish_skipped=false`. A later Perlmutter smoke at
remote commit `b67d15a8` proved the solve-call trial-policy correction: trial K1
returned with `newton_polish_skipped=true`, `newton_iter=0`, and
`pre_newton_iter=0`. That same run exposed one remaining cache boundary in the
trace wrapper: real target-lane inputs arrived as `jax.Array`, while the
decomposed K1 memo only cached NumPy inputs. The current patch fixes that by
host-normalizing all non-tracer array inputs before keying the memo. RunPod H100
smokes on 2026-07-02 now prove both runtime cache boundaries:
trace-enabled trial-`skip` runs reuse the decomposed K1 forward result inside
the objective-evaluation trace lane, and a same-fidelity non-trace `run/run`
smoke reuses the decomposed solved-state payload during final reporting
(`target_lane_reporting_forward_result_* reused=true`). Trial `skip` runs
intentionally do not reuse their lower-fidelity trial solve as the
final/reporting solved state. A later quality A/B did not prove `skip` as a
safe production default, so the current production default keeps trial Newton
polish at `run`. A later cap-300 smoke proved the explicit trial-budget
plumbing but also showed that the current decomposed wrapper applies trial caps
to SciPy's first `x0` objective evaluation. Trial BFGS caps therefore remain
explicit-only until incumbent/full-fidelity evaluations can be excluded.

The right fix is not to default to `lsmr_j`, not to switch to `lbfgs-ondevice`,
and not to copy the CPU path blindly. The right fix is to make the target-lane
contract explicit:

- Current production optimizer candidate evaluations stay same-fidelity by
  default; cheaper/bounded trial policies are explicit experiments until the
  objective path can distinguish true probes from incumbent evaluations.
- Future probe-only trial candidates can use the cheapest solve sufficient for
  line-search decisions once that classification exists.
- Accepted/final candidates get the full high-fidelity solve.
- The final reporting path consumes the already computed accepted/final K1
  forward result when available.

## What Was Observed

### Perlmutter Diagnostic Run

The relevant diagnostic run was:

- Job: `55353209`
- QOS: `gpu_shared_interactive`
- GPU: A100-SXM4-40GB
- Exit: `0:0`
- Walltime: `00:14:05`
- Commit under test: `945a010b2`
- Output root:
  `/pscratch/sd/j/jungdae/k1_matrix_runs/newtondiag-945a010b28ba-20260701T1433Z-direct-srun/55353209/output/mpol=10-ntor=10-a5587f70`

Runtime settings recorded for the run:

- `jax 0.10.0`
- default backend: `gpu`
- device: `cuda:0`
- `SIMSOPT_ADJOINT_LINEAR_SOLVER=dense`
- `SIMSOPT_DENSE_OPERATOR_CHUNK_BATCH_SIZE=16`
- `SIMSOPT_JAX_TRANSFER_GUARD=disallow`
- `XLA_PYTHON_CLIENT_PREALLOCATE=true`

The run intentionally used `--maxiter 1`, so `REJECTED.json` was expected. The
purpose was not final optimizer quality; it was to isolate where time is spent
inside the target-lane solve.

### Diagnostic Timing Signal

The important event-level signal was:

| Eval | Outcome | Main cost | Notes |
| --- | --- | --- | --- |
| 1 | success | moderate | warm start was already good; Newton attempted 0 iterations |
| 2 | failure | very expensive | rejected candidate spent about 175 s in Newton polish |
| 3 | success | expensive | successful candidate still spent about 98 s in Newton polish |

The rejected eval was the key. It showed:

- `success=False`
- `primal_success=False`
- `newton_attempted_iterations=24`
- `newton_stalled=True`
- `newton_stop_reason_code=2`
- last linear solve failed
- last backtracking accepted alpha was `0.0`
- `newton_polish` took about 175 s

That is the problem in one line: a doomed line-search trial can burn minutes in
Newton polish before it is rejected.

Important precision: the expensive trial work is traceable Newton polish using
operator-GMRES solves against Hessian-vector products. The dense adjoint solver
setting controls K2/final compatibility factorization, not the inner trial
Newton correction solve itself.

The final reporting path also reported a forward-result boundary with
`has_runtime_forward_result=True` but `reused=False`, meaning the target-lane
reporting path had the machinery to use runtime forward results but did not
consume the already computed phase-2 result in that run.

## Relevant Code Paths

### Native CPU / C++-Reference Behavior

The native `BoozerSurface.run_code()` path is stateful. For BoozerLS, it runs
BFGS/L-BFGS first and then Newton polish:

- `src/simsopt/geo/boozersurface.py:162` - `run_code(...)`
- `src/simsopt/geo/boozersurface.py:196` - BoozerLS branch
- `src/simsopt/geo/boozersurface.py:203` - BFGS/L-BFGS phase
- `src/simsopt/geo/boozersurface.py:208` - Newton polish phase

The CPU path does not have a separate immutable `forward_result` object. Its
cache is the mutable combination of:

- `boozer_surface.surface`
- `boozer_surface.res`
- `boozer_surface.need_to_run_code`
- accepted state stored in `run_dict`

The single-stage CPU/reference objective restores accepted state before a
candidate solve and rolls back after failed/rejected candidates:

- `single_stage_banana_example.py:11755` - `_restore_cpu_boozer_state`
- `single_stage_banana_example.py:11763` - accepted-state restore helper
- `single_stage_banana_example.py:11798` - run Boozer from `run_dict`
- `single_stage_banana_example.py:12040` - candidate solve starts from
  accepted state
- `single_stage_banana_example.py:12124` - failed/rejected candidate restores
  accepted state

Accepted-step snapshotting then persists the solved mutable state:

- `single_stage_banana_example.py:13213` -
  `snapshot_accepted_step_state_from_values`
- `single_stage_banana_example.py:13245` - `snapshot_accepted_step_state`
- `single_stage_banana_example.py:13287` - `accept_step`

This means the CPU path avoids the exact "duplicate immutable K1 result" problem
by construction: there is no separate K1 artifact to pass around. The solved
state is installed into the mutable Boozer object and then snapshotted.

However, CPU does not solve the performance problem by skipping Newton polish on
rejected line-search trials. If `run_code()` is called for a candidate, BoozerLS
still runs the polish sequence. The difference is that the native path is much
cheaper and the mutable cache model hides some duplication.

### JAX Target-Lane Decomposed Behavior

The decomposed JAX path explicitly separates K1 and K2:

- `single_stage_banana_example.py:7427` - decomposed host `value_and_grad`
- `single_stage_banana_example.py:7437` - K1 `solve_fn(...)`
- `single_stage_banana_example.py:7466` - K2 value/gradient from solved state
- `single_stage_banana_example.py:7479` - success branch
- `single_stage_banana_example.py:7487` - primal-success rejected branch
- `single_stage_banana_example.py:7494` - primal-failure baseline gradient

This split is the correct architecture for avoiding a monolithic on-device
outer optimizer, but it requires explicit reuse.

There is already a local K1 reuse hook for trace-style paths:

- `single_stage_banana_example.py:7420` - `last_solved_forward_result`
- `single_stage_banana_example.py:7496` - `reuse_forward_result_for(...)`
- `single_stage_banana_example.py:8569` -
  `_make_reusing_trace_forward_result(...)`

There is also accepted/final sync cache plumbing:

- `single_stage_banana_example.py:6449` -
  `_cache_target_lane_objective_evaluation_sync_state`
- `single_stage_banana_example.py:6470` -
  `_cached_target_lane_objective_evaluation_sync_state`
- `single_stage_banana_example.py:6488` -
  `_cached_target_lane_objective_evaluation_forward_result`
- `single_stage_banana_example.py:13587` -
  `_cached_target_lane_sync_inputs`
- `single_stage_banana_example.py:13605` -
  `sync_accepted_step_state_from_target_lane_values`
- `single_stage_banana_example.py:13627` -
  `sync_accepted_step_state_from_objective_value_and_grad`

But production `scipy-jax-decomposed` currently does not populate the
objective-evaluation sync cache on the normal non-trace path. The trace wrapper
can populate `_cache_target_lane_objective_evaluation_sync_state(...)`, but
production only wraps the objective when objective-evaluation tracing is enabled:

- `single_stage_banana_example.py:17627` starts from
  `target_value_and_grad_objective`
- `single_stage_banana_example.py:17634` only builds the trace wrapper when
  `objective_evaluation_trace_callback` is present
- `single_stage_banana_example.py:17660` builds that wrapper only in the trace
  branch

The optional `target_lane_objective_evaluation_forward_result` field is not the
required reuse mechanism for final reporting. The required payload is the
exact-DOF `target_lane_objective_evaluation_solve_result`:

- `single_stage_banana_example.py:13170` caches solve state
- `single_stage_banana_example.py:13173` stores `solve_result`
- `single_stage_banana_example.py:13174` stores the optional full
  `forward_result` only in a narrower trace-wrapper mode

The actual recompute site is the reporting sync path:

- `single_stage_banana_example.py:7597` uses `target_lane_solve_result` when it
  is provided
- `single_stage_banana_example.py:7600` calls `accepted_step_solve_result(...)`
  when no cached solve result is provided
- `single_stage_banana_example.py:7574` then calls `forward_result_fn(coil_dofs)`

Final penalty metrics consume cached target-lane reporting metrics:

- `single_stage_banana_example.py:6520` -
  `_require_cached_target_lane_reporting_metrics`
- `single_stage_banana_example.py:6572` -
  `_target_lane_reporting_cache_matches_final_state`
- `single_stage_banana_example.py:6600` -
  `resolve_single_stage_final_penalty_metrics`

If the cache does not match the final state, benchmark mode can build a deferred
reporting snapshot:

- `single_stage_banana_example.py:18279` - benchmark-mode cache check
- `single_stage_banana_example.py:18294` - deferred reporting snapshot

The general final penalty/reporting path should not need a second K1 solve when
the final K1 result was already computed during the phase-2 optimizer
evaluation. The implementation target is therefore the solved-state sync cache,
not the optional full-forward-result cache field.

### JAX Newton Polish Policy

The target-lane Boozer adapter supports a policy knob:

- `src/simsopt_jax_adapters/geo/boozer_surface.py:6324` - skip branch in the
  traceable path
- `src/simsopt_jax_adapters/geo/boozer_surface.py:6382` - otherwise run Newton
  polish
- `src/simsopt_jax_adapters/geo/boozer_surface.py:7840` - skip branch in the
  host-style adapter path
- `src/simsopt_jax_adapters/geo/boozer_surface.py:7865` - otherwise run Newton
  polish

The pre-fix policy was too coarse. It was essentially global `run` or `skip`.
The diagnostics required a context-aware policy:

- Future line-search trial candidate: bounded/cheap solve, early escape on
  failure after the objective path can distinguish true probes from incumbent
  full-fidelity evaluations.
- Accepted/final candidate: full Newton polish.
- Explicit diagnostic/reference run: allow full polish everywhere when requested.

## Root Cause

The root cause is not a single low-level kernel. It is an orchestration mismatch:
the target lane made the inner Boozer solve more explicit and more functional,
but the outer workflow treated every candidate and every reporting sync as if a
native mutable state object were cheap to refresh.

This produces two concrete costs:

1. **Rejected-trial over-polishing**

   The line search probes candidates that are expected to be rejected. Some are
   far from the accepted state or already show failed linear solves /
   backtracking failure. The pre-fix target-lane solve could still pay full
   Newton polish before returning a failure. The diagnostic eval-2 shows this can cost
   about 175 seconds for one rejected trial.

2. **Final-reporting duplicate K1**

   The decomposed optimizer already computed a K1 forward result for the final
   point. If final reporting cannot consume that result, it runs another
   forward/reporting solve for the same DOFs. CPU avoids this class of problem
   through mutable installed state; JAX needs an explicit forward-result reuse
   contract.

## Non-Root Causes

These are real issues in the broader project, but they are not the current main
fix for the observed slowdown.

### Not "JAX Is Running on CPU"

The Perlmutter diagnostic run recorded JAX GPU execution on `cuda:0` with the
transfer guard enabled. The slow path was inside the target-lane solve, not a
CPU fallback misconfiguration.

### Not Primarily `lsmr_j`

`lsmr_j` is useful as an experimental comparator for residual-J adjoints, and
the dependency gate was fixed in commit `66394f101`. But the observed expensive
work was full traceable Newton-polish work on trial candidates: repeated
operator-GMRES solves against Hessian-vector products, plus duplicate
forward/reporting sync. It was not primarily the dense-vs-LSMR choice.

`lsmr_j` should remain experimental/debug until its gradient and optimizer-step
behavior are validated in full single-stage comparisons.

### Not a Need to Switch to `lbfgs-ondevice`

`lbfgs-scipy-jax-decomposed` exists to keep SciPy host control and avoid the
monolithic compile/memory risks of full on-device outer L-BFGS. That design
choice remains sound. The problem is the adapter contract around K1 reuse and
trial solve fidelity.

### Not a Pure Compile-Cache Problem

Persistent cache matters operationally, but the diagnostic run got into real
runtime work. Compile cache will reduce repeated cold starts; it will not fix
one rejected line-search trial spending minutes in Newton polish.

## Required Fixes

### Fix 1: Reuse the Phase-2 Solved-State Payload in Final Reporting

Goal: if the optimizer's last accepted/final candidate already produced a K1
forward result, final sync/reporting must consume the exact-DOF solved-state
payload derived from it rather than recomputing.

Design requirements:

- The cache key must be the exact optimizer DOFs.
- Reuse must be disabled if DOFs differ by even one bit.
- Reuse must carry enough data for reporting and accepted-state sync:
  - `sdofs`
  - `iota`
  - `G`
  - solved `x`
  - success/primal-success/finite flags
  - optional `objective_value`
  - optional `objective_grad`
- The final reporting path should record whether it reused the result.

Likely implementation points:

- Ensure decomposed production `value_and_grad` records the latest exact-DOF
  solved-state payload derived from its K1 `forward_result`.
- Store that payload through `_cache_target_lane_objective_evaluation_sync_state(...)`
  as `target_lane_objective_evaluation_solve_result`; do not make final
  reporting depend on the optional full
  `target_lane_objective_evaluation_forward_result`.
- Keep `resolve_target_lane_post_run_state_sync(...)` using the adapter cache
  path so it passes `target_lane_solve_result` into
  `build_single_stage_target_lane_accepted_step_sync(...)`, avoiding the
  `accepted_step_solve_result(...) -> forward_result_fn(...)` recompute.
- Add a regression proving that final reporting for matching DOFs does not call
  a second K1 solve.

Acceptance criteria:

- Progress events show `target_lane_reporting_forward_result_started` with
  `reused=True` for the final state, proving reporting sync received a cached
  `target_lane_solve_result` and did not call `forward_result_fn(...)` again.
- A unit/integration test fails if final reporting calls a fake second K1 solve.
- Existing CPU/reference reporting remains unchanged.

### Fix 2: Keep Trial Polish Same-Fidelity Until Probe Classification Exists

Goal: avoid shipping a lower-fidelity production default until the decomposed
objective path can distinguish true line-search probes from incumbent full-
fidelity evaluations. Full polish remains the production default; cheaper
candidate policies remain explicit experiments.

Design requirements:

- Preserve high-fidelity accepted/final reporting.
- Keep a reference/debug mode that can force full polish on every trial.
- For normal production target lane:
  - use same-fidelity `run` for optimizer candidate evaluations;
  - keep cheaper or bounded trial policies behind explicit flags until probe
    classification exists;
  - run full Newton polish for accepted/final sync.
- The policy must be explicit in results/progress metadata.

Possible policy shape:

```text
target_lane_newton_polish_context = {
  trial: bounded | skip | cheap,
  accepted: run,
  final: run,
  reference: run
}
```

This should replace the current global-only `run`/`skip` choice with a
context-aware policy. A single global knob is too blunt because it trades
accuracy and speed for all candidates at once.

Acceptance criteria:

- Production defaults do not lower trial fidelity implicitly.
- Accepted/final states still report full-polish metrics.
- Progress events clearly distinguish trial solve policy from final solve
  policy.
- A regression verifies that omitted trial BFGS caps do not become active trial
  overrides via the full target-lane budget.

### Fix 3: Remove Separate Newton-Stall Escape as a Standalone Fix

The traceable Newton runner already exits after a stalled Newton step: its loop
condition includes `~state["stalled"]`, and each failed step sets
`stalled = ~accepted`. The observed `newton_stalled=True` is therefore the
terminal result of the failed step, not a signal the loop ignored for additional
iterations.

Keep the actionable work under Fix 2: make the Newton polish policy
line-search-context-aware so trial candidates use a bounded, cheap, or skipped
policy, while accepted/final/reference candidates can still request full polish.

## Validation Plan

### Static/Unit Tests

1. Add a fake K1 callable that increments a counter.
2. Run a target-lane decomposed objective evaluation.
3. Run final reporting for the same final DOFs.
4. Assert the K1 counter did not increment a second time.

Add a separate test for Newton trial policy:

1. Configure production trial context.
2. Invoke a fake line-search trial candidate.
3. Assert full traceable Newton polish is not called under the trial policy.
4. Assert accepted/final/reference policy still calls full polish.

### Focused GPU Smoke

Run `lbfgs-scipy-jax-decomposed` on the same `iota011_R0935` candidate:

- same seed
- same targets
- `maxiter=1` first
- transfer guard disallow
- preallocation true
- dense-adjoint solver first
- chunk batch 16 if memory-safe, otherwise 8

Expected improvements:

- rejected trial Newton-polish time should drop sharply;
- final reporting should show K1 reuse;
- total walltime should decrease even if compile cache is cold.

### Full Comparison

After the focused smoke:

- dense-adjoint baseline with old policy
- dense-adjoint path with trial-aware polish + final K1 reuse
- optional `lsmr_j` comparator behind experimental flag

Metrics:

- total walltime
- number of K1 solves
- number of K2 value/grad calls
- rejected-trial Newton-polish time
- accepted/final Newton-polish time
- final objective
- iota/volume/residual
- optimizer success status
- GPU memory high-water mark

## Risks and Guardrails

### Risk: Skipping Too Much on Trial Candidates

Line search needs a coherent merit value and gradient. The fix must not return a
cheap value that violates the optimizer contract. The existing decomposed
contract should remain:

| Case | Value | Gradient |
| --- | --- | --- |
| success=True | solved-state objective | candidate gradient |
| success=False, primal_success=True | filtered/rejected value | candidate gradient |
| success=False, primal_success=False | rejected value | baseline gradient |

### Risk: Reusing Stale Solved-State Payloads

Solved-state reuse must be exact-keyed on optimizer DOFs. Approximate matching
is unsafe. If the key differs, fall back to recompute.

### Risk: Benchmark-Only Optimization

The cache/reuse path must work outside `--benchmark-mode`. Benchmark mode is
useful for isolating runtime, but the production final artifact path must also
benefit.

### Risk: Hiding Solver Failures

Trial-aware polish must be loud in diagnostics. It should reduce wasted work,
not silently make failed solves look successful or let low-fidelity trial state
become accepted/final state.

## Recommended Patch Order

1. **Final K1 solved-state reuse**
   - Use existing cache abstractions.
   - Add a counter-based regression.
   - Verify final reporting progress shows `reused=True`.

2. **Trial-aware Newton polish policy**
   - Add a context-aware policy surface.
   - Keep production trial candidates same-fidelity by default.
   - Leave bounded/cheap policies explicit-only until true probe classification
     exists.
   - Keep full polish for accepted/final/reference.

3. **Remove standalone Newton-stall work**
   - Treat stall diagnostics as evidence for Fix 2, not a separate missing
     escape.
   - Do not duplicate the existing traceable Newton stalled-state loop exit.

4. **GPU validation**
   - Run focused maxiter-1 smoke.
   - Then run a longer dense full-path comparison.
   - Keep `lsmr_j` as experimental comparator only.

## Current Status

Already done:

- K1/K2 diagnostic instrumentation landed in commits:
  - `1a9deabac perf(jax): add traceable Newton diagnostics`
  - `945a010b2 perf(jax): propagate traceable Newton progress fields`
- The Perlmutter diagnostic run confirmed the expensive rejected-trial Newton
  polish behavior.
- The codebase already contains partial cache/reuse infrastructure.
- Traceable Newton already exits on a stalled step; the problem is entering full
  trial polish before the outer line search knows whether the candidate will be
  accepted.

Implemented in the follow-up patch:

- Production non-trace `scipy-jax-decomposed` objective evaluations now cache
  the latest exact-DOF solved-state payload into the same sync cache consumed by
  accepted-step/final reporting.
- The target lane now has a separate trial-only Boozer Newton-polish policy.
  The default remains full `run`; explicit trial BFGS budget flags remain
  available for experiments, while explicit `skip` remains available for
  lower-fidelity experiments.
- Trial polish policy is now recorded as an active override only when it differs
  from the full target-lane policy. Same-fidelity `run/run` resolves to no
  active trial polish override, preserving the final-sync solved-state cache.
- The decomposed K1 `solve_fn` is wrapped so trial Boozer overrides are applied
  when SciPy evaluates each candidate, not only while the objective provider is
  constructed.
- The decomposed K1 memo now host-normalizes non-tracer array inputs before
  keying the cache. This covers the real target-lane path where SciPy/JAX
  wrapper calls pass `jax.Array` inputs rather than plain NumPy arrays.
- The traceable runtime cache signature includes `newton_polish_policy`, so
  full-vs-trial K1 runtimes cannot share a stale compiled runtime keyed only on
  the other Boozer options.
- The parity wrapper and Perlmutter launch scripts now pass and record the
  trial-only policy, so GPU validation jobs exercise the intended child
  runtime contract instead of relying on implicit child defaults.
- Focused regressions cover exact-DOF solved-state cache reuse and the
  trial-vs-full polish policy split.
- Focused H100 remote validation after the same-fidelity override patch passed:
  `tests/integration/test_single_stage_newton_polish_policy.py -k "child_trial_override or trial_boozer_overrides_use_trial_policy_not_full_policy or trial_solve_cache or same_fidelity_trial_solve"`
  (`7 passed`) and
  `tests/integration/test_single_stage_jax_cpu_reference.py -k "adapter_final_sync_falls_back_to_decomposed_solve_cache or decomposed_host_objective_feeds_final_reporting_sync_cache"`
  (`2 passed`).

Runtime validation from the first follow-up patch:

- Staged local commit: `118803140155`
- Remote archive commit: `88ffef1e73a2`
- Run root:
  `/pscratch/sd/j/jungdae/scipy_jax_decomp_report_runs/20260701T173241Z-118803140155-iota011-direct`
- Completed fixed-path job: `55358522`
- Case:
  `/pscratch/sd/j/jungdae/scipy_jax_decomp_report_runs/20260701T173241Z-118803140155-iota011-direct/55358152B/task_1_skip_trial_dense`
- GPU/runtime evidence:
  - `default_backend=gpu`
  - device `cuda:0`
  - `jax 0.10.0`
  - x64 enabled
  - `SIMSOPT_JAX_TRANSFER_GUARD=disallow`
  - `XLA_PYTHON_CLIENT_PREALLOCATE=true`
  - `SIMSOPT_DENSE_OPERATOR_CHUNK_BATCH_SIZE=4`
  - `SIMSOPT_ADJOINT_LINEAR_SOLVER=dense`
- Manifest evidence:
  - `case_label=skip_trial_dense`
  - `trial_newton_polish_policy=skip`
  - `local_commit=118803140155`
  - `remote_archive_commit=88ffef1e73a2`
- Progress evidence:
  - startup event recorded
    `target_lane_trial_boozer_newton_polish_policy=skip`
  - bundle setup recorded
    `trial_boozer_overrides.newton_polish_policy=skip`
  - decomposed K1 evals returned in about `44.15 s`, `33.20 s`, and
    `63.30 s`
  - those K1 evals still reported `newton_polish_skipped=false`, with Newton
    iterations observed on the rejected-trial path; this disproved the initial
    assumption that construction-time trial overrides reached the later SciPy
    `solve_fn` calls
  - decomposed K2 value/grad ran for evals 1 and 3
  - final sync recorded
    `target_lane_reporting_forward_result_started reused=true`
  - final sync returned in about `83.00 s`
- Output evidence:
  - `exit_status.txt` is `0`
  - Slurm job `55358522` completed with `ExitCode=0:0`
  - `/usr/bin/time` recorded `20:30.32` for the Python step
  - no `fatal_error.log` content
  - the final artifact is `REJECTED.json`, not `results.json`, because the
    smoke intentionally used `maxiter=1`
  - rejection reasons are `optimizer_success_not_true` and
    `optimizer_status_1`
  - optimizer status is `1`, message
    `STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT`
  - physics state stayed coherent for the smoke:
    `FINAL_IOTA=0.11017506684972597`,
    `TARGET_IOTA=0.11015671329581699`,
    `FINAL_VOLUME=0.04916423194665707`,
    `TARGET_VOLUME=0.04920000000000004`,
    `FINAL_BOOZER_RESIDUAL=4.573133940318431e-07`,
    `MAX_CURVATURE=35.92533799246334`,
    `SELF_INTERSECTING=false`

Runtime boundary:

- This is a completed GPU smoke proving final-sync reuse and clean process exit
  for the dense-adjoint path. It is not a final accepted single-stage production
  result.
- A second focused Perlmutter smoke for the solve-call correction was run as
  job `55363238` against remote source commit `b67d15a8e9c8287e6755948246d42b1b424a4e5d`:
  `/pscratch/sd/j/jungdae/scipy_jax_decomp_report_runs/20260701T190508Z-6e0cb1c337be-iota011-direct`.
  Runtime files are under
  `/pscratch/sd/j/jungdae/scipy_jax_decomp_report_runs/20260701T190508Z-6e0cb1c337be-iota011-direct/55363238/task_1_skip_trial_dense_fast2`.
  This run used `default_backend=gpu`, device `cuda:0`, x64, transfer guard
  `disallow`, preallocation `true`, chunk batch `4`, and dense adjoint solver.
  It proved the trial solve-call policy: eval 1 emitted
  `target_lane_decomposed_k1_forward_returned` with
  `newton_polish_skipped=true`, `newton_iter=0`, `pre_newton_iter=0`,
  `success=true`, and `primal_success=true`.
- The same job exposed a cache boundary bug in the trace wrapper. After K2
  returned, progress recorded
  `target_lane_optimizer_objective_eval_1_forward_result_started` instead of
  `target_lane_optimizer_objective_eval_1_forward_result_reused`. The cause was
  the K1 memo's NumPy-only cache population; the real path passed a `jax.Array`.
  Job `55363238` was intentionally cancelled after this finding to avoid paying
  the duplicate K1 recompute.
- The current patch fixes the JAX-array cache population and strengthens the
  regression so the trace wrapper calls the decomposed objective with a
  `jax.Array` under `jax.transfer_guard("disallow")`. Remote commit
  `425969c7db8a657d96e6d12031c114bcfd4b9162` passed:
  `pytest -q tests/integration/test_single_stage_jax_cpu_reference.py -k "trace_wrapper_uses_decomposed_k1_cache_after_value_grad or decomposed_trace_reuse_hits_through_real_optimizer_to_coil_transform"`.
  RunPod H100 run
  `/workspace/runpod-k1-runs/direct-jax-m10-h100-edea9ba10-20260702T030856Z`
  then proved the runtime trace boundary: it exited `0`, wrote
  `final_artifact_write_returned`, recorded 17 K1 returns and 9 K2 returns, and
  emitted trace-lane forward-result reuse after K2 for same-candidate checks.
  Local copies are under `runpod_artifacts_2026-07-02/trace/`.
- RunPod H100 non-trace/trial-skip run
  `/workspace/runpod-k1-runs/direct-jax-m10-nontrace-h100-edea9ba10-20260702T033009Z`
  reproduced the intended lower-fidelity boundary: active trial overrides
  disable final-sync solved-state reuse (`target_lane_reporting_forward_result_*`
  recorded `reused=false`) because final/reporting must not consume a
  trial-`skip` solved state as a full-fidelity result. Local copies are under
  `runpod_artifacts_2026-07-02/nontrace/`.
- RunPod H100 same-fidelity non-trace run
  `/workspace/runpod-k1-runs/direct-jax-m10-nontrace-trialrun-nobudget-h100-edea9ba10-20260702T041302Z`
  proves final-sync solved-state reuse when no trial override is active. It
  used full `run`, trial `run`, no explicit full BFGS/Newton budget overrides,
  exited `0`, and recorded every `trial_boozer_overrides` field as `null`.
  Final sync recorded
  `target_lane_reporting_forward_result_started reused=true`,
  `target_lane_reporting_forward_result_returned reused=true`,
  `target_lane_reporting_value_and_grad_started reused=true`,
  `target_lane_reporting_value_and_grad_returned reused=true`, and
  `target_lane_final_sync_returned elapsed_s=1.1226`. Local copies are under
  `runpod_artifacts_2026-07-02/nontrace_trialrun_nobudget/`.
- A same-resolution JAX runtime seed-spec projection hang was encountered while
  staging the official warm-start path. The smoke used a manually generated
  direct-DOF runtime seed spec for `iota011_R0935`. That is a separate
  staging-path issue and should be fixed independently by bypassing projection
  when the serialized surface already matches the requested resolution.
- The optional `lsmr_j` comparator was not run as part of this smoke and should
  remain experimental until its optimizer-step behavior is validated.

## Bottom Line

The original target-lane slowdown was mainly a workflow/contract problem:

- JAX made K1/K2/forward-result boundaries explicit.
- The optimizer/reporting workflow had not fully exploited those boundaries.
- Rejected line-search candidates paid full Newton polish before the outer line
  search accepted or rejected them.
- Final reporting recomputed forward state that was already available from the
  optimizer.

The follow-up patch fixes the intended workflow boundaries for
`scipy-jax-decomposed`: trace mode reuses the decomposed K1 forward result for
same-candidate objective-evaluation replay, same-fidelity non-trace final sync
reuses the exact-DOF solved-state payload, and trial probes are wired to use the
trial-only policy at K1 solve-call time. Perlmutter smokes confirmed the
solve-call `skip` path; RunPod H100 smokes now confirm trace reuse and
same-fidelity non-trace final-sync reuse. Trial `skip` remains a lower-fidelity
experimental path, so final reporting correctly reruns/reuses only full-
fidelity solved state rather than consuming the skipped trial solve. The
production default is full trial Newton polish with no implicit trial BFGS cap;
cap-limited trial budgets remain explicit experiments until the objective path
can distinguish true line-search probes from incumbent evaluations. A longer
accepted-result run and an optional `lsmr_j` comparator remain separate
follow-ups.
