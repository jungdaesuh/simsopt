# Strict CUDA E2E Progress Report

Snapshot time: 2026-05-28T18:47:18Z

Local checkout:

- Repository: `/Users/suhjungdae/code/columbia/simsopt-jax`
- Branch: `gpu-purity-stage2-20260405`
- Local HEAD: `13a664f157d4`
- Tracked tree status at snapshot: clean
- Untracked local paths intentionally left alone: `.antigravitycli/`, `.conda/`, `analysis/`, `runs/`

Runpod target:

- SSH: `ssh -i /Users/suhjungdae/.runpod/ssh/RunPod-Key-Go -p 16628 root@154.54.102.24`
- Root: `/root/simsopt_e2e_ea597cc14_20260527T191926Z`
- Python: `/root/simsopt_canary_local/venv/bin/python`
- Current deployed source:
  `/root/simsopt_e2e_ea597cc14_20260527T191926Z/src_13a664f15_20260528T175259Z`
- Current deployed source provenance: local commit
  `13a664f157d4dfac3df5ae16a6b73a1d37020682`, clean tracked git status,
  JAX `0.10.0`, jaxlib `0.10.0`, backend `gpu`, device `cuda:0`

## Executive Status

The strict CUDA E2E effort is not complete yet. The current blocker is still
the production `lbfgs-ondevice` target lane at m04 resolution, but the failure
has been narrowed substantially:

- m02 current-source CUDA `ondevice` passed parity.
- m04 current-source CUDA no longer looks like a physics parity mismatch first.
  The observed failure was a target-lane L-BFGS performance/liveness failure
  after the CPU reference and same-candidate replay had already completed.
- The old unstructured harness crash path was replaced by structured failure
  JSON in the e2cc run.
- A root-cause-adjacent callback issue was found and fixed locally in
  `13a664f15`: objective-evaluation trace recording no longer automatically
  installs target optimizer state callbacks inside compiled target L-BFGS.
- The first patched m04 rerun
  `m04n04_i05_current_13a664f15_20260528T175502Z` was manually aborted at the
  user's request and is not benchmark evidence.
- A fresh patched full m04 rerun
  `m04n04_i05_current_13a664f15_20260528T183941Z` emitted structured JSON, but
  failed in the generated CPU seed lane with child exit `-15` before
  `results.json`; this is a harness/process failure, not parity evidence.
- A small m04 JAX target-lane reproducer
  `repro_m04_target_cuda_13a664f15_20260528T184542Z` is active on Runpod with
  `maxiter=1`, using the valid completed seed from the manually aborted run.

The fastest path remains: finish patched m04, then generate valid m06/m08
warm-start donors, then run the optimizer matrix, Stage 2 CUDA E2E, and the
full ladder as final proof.

## Progress Completed

### Planning and provenance

- Wrote and committed the live recovery plan:
  `docs/strict_cuda_e2e_runpod_recovery_plan_2026-05-28.md`.
- Committed plan as `a94a7643e docs: plan strict CUDA E2E Runpod recovery`.
- Stopped treating stale artifacts as current evidence.
- Deployed source snapshots to Runpod with explicit provenance, instead of
  relying on the root `repo_sha.txt` marker from older runs.
- Confirmed current source copies can carry provenance without a `.git`
  directory by exporting/recording source SHA and git status.

### Current code fix

Committed fix:

- Commit: `13a664f157d4dfac3df5ae16a6b73a1d37020682`
- Subject: `Avoid target optimizer trace callbacks by default`

Files changed:

- `benchmarks/single_stage_init_parity.py`
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
- `tests/geo/test_single_stage_example.py`
- `tests/test_benchmark_helpers.py`

Behavioral change:

- `--record-objective-evaluation-trace` no longer implies target
  optimizer-state tracing for `lbfgs-ondevice`.
- New explicit flag: `--record-target-optimizer-state-trace`.
- The parity wrapper passes target optimizer-state tracing only for the
  `lbfgs-trace` reference diagnostic lane.
- Added an `optimizer_endpoint_trace` fallback after optimizer return so final
  metric drift can still be classified as an optimizer-control split without
  placing `jax.debug.callback` inside the compiled L-BFGS loop.

Local validation for `13a664f15`:

- `tests/test_benchmark_helpers.py`: `321 passed, 2 skipped in 5.64s`
- Focused `tests/geo/test_single_stage_example.py`: `16 passed, 359 deselected in 3.71s`
- Earlier focused new tests:
  - benchmark helper trace-gating tests: `3 passed`
  - single-stage parse test: `1 passed`
- `ruff format --check` on touched files: passed
- `ruff check` on touched files: passed
- `py_compile` on benchmark and example entrypoints: passed
- `git diff --check`: passed

Remote validation for deployed `13a664f15`:

- Focused benchmark helper tests:
  `3 passed, 320 deselected in 3.81s`

## Current Active Run

Active Runpod run:

- Run directory:
  `/root/simsopt_e2e_ea597cc14_20260527T191926Z/artifacts/repro_m04_target_cuda_13a664f15_20260528T184542Z`
- Source:
  `/root/simsopt_e2e_ea597cc14_20260527T191926Z/src_13a664f15_20260528T175259Z`
- Command family: direct
  `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
- Resolution: `mpol=4`, `ntor=4`, `nphi=63`, `ntheta=32`, `maxiter=1`
- Optimizer backend: `ondevice`
- Mode: JAX target lane, benchmark mode, minimal artifacts
- Reproducer iteration budget: `maxiter=1`
- Top-level timeout: 1800 seconds

Status at 2026-05-28T18:47:18Z:

- Final JSON: not present yet
- Active child: JAX target lane
- Last progress event: `initial_hardware_status_started`
- Last progress elapsed: about `64.318 s`
- Process RSS at poll: about `2,034,148 KB`
- GPU memory at poll: about `1529 MiB`, `5%` utilization

Important warm-start detail:

- This target-lane reproducer uses the completed seed donor from:
  `/root/simsopt_e2e_ea597cc14_20260527T191926Z/artifacts/m04n04_i05_current_13a664f15_20260528T175502Z/case_artifacts/seed_outputs/mpol=4-ntor=4-66b0718f`.
- It avoids another full CPU reference spend while checking whether patched
  `lbfgs-ondevice` target execution returns.

The target-lane reproducer result is pending. It is not a full parity verdict,
but it is the current critical-path liveness/performance artifact.

Fresh full patched m04 run:

- Run directory:
  `/root/simsopt_e2e_ea597cc14_20260527T191926Z/artifacts/m04n04_i05_current_13a664f15_20260528T183941Z`
- Final JSON: `single_stage_cuda.json`
- Status: `case-execution-failed`
- Failure: generated CPU seed child exited `-15` at
  `host_artifact_export_started`, before `results.json`
- Wall time: `4:23.85`
- MaxRSS: `1,761,036 KB`
- Evidence use: structured harness/process failure only, not a parity verdict

Manually aborted patched run:

- Run directory:
  `/root/simsopt_e2e_ea597cc14_20260527T191926Z/artifacts/m04n04_i05_current_13a664f15_20260528T175502Z`
- Stop reason: user requested `stop all things`
- Final JSON: absent
- `run.exit_code`: absent after process termination
- Evidence use: operational history only, not a parity, performance, or memory
  verdict

## Completed Evidence So Far

### m02 CUDA ondevice

Run directory:

`/root/simsopt_e2e_ea597cc14_20260527T191926Z/artifacts/ondevice_m02_current_e2cc7394590f_20260528T140953Z`

Result:

- Passed: `true`
- Failures: none
- Backend: CUDA GPU
- Optimizer backend: `ondevice`
- Strict transfer guard: disallow
- X64: true

Parity:

- Same-candidate replay max objective difference:
  `1.0658141036401503e-14`
- Same-candidate replay max optimizer-gradient difference:
  `4.863220937068036e-12`
- Same-candidate replay failures: none
- Final metric drift was accepted as an optimizer acceptance/control split
  after same-candidate parity passed.

Performance and memory:

| Metric | Value |
| --- | ---: |
| Wall time | `37:50.89` |
| MaxRSS | `13,733,416 KB` |
| Sampled peak GPU memory | `3509 MiB` |
| CPU elapsed | `237.8897715397179 s` |
| JAX elapsed | `1191.6138821206987 s` |
| CPU outer optimizer | `184.03278366103768 s` |
| JAX outer optimizer | `1051.1107656676322 s` |
| CPU outer optimizer main | `182.67811040952802 s` |
| JAX outer optimizer main | `680.170995650813 s` |
| JAX initial target objective | `211.37829169258475 s` |
| JAX final sync | `4.579431440681219 s` |

Speed comparison:

- JAX/GPU was about `5.0x` slower than CPU by total elapsed time.
- JAX/GPU was about `5.7x` slower by outer optimizer time.
- JAX/GPU was about `3.72x` slower by outer optimizer main time.

Improvement versus the older f14 source run:

- Wall time improved from `45:10.36` to `37:50.89`, about `16.3%`.
- JAX elapsed improved from `1704.814 s` to `1191.614 s`, about `30.1%`.
- JAX outer optimizer improved from `1555.152 s` to `1051.111 s`, about
  `32.4%`.
- JAX outer optimizer main improved from `1174.931 s` to `680.171 s`, about
  `42.1%`.
- MaxRSS was roughly similar: old `13,895,176 KB`, current `13,733,416 KB`.

### m04 CUDA ondevice, e2cc source

Run directory:

`/root/simsopt_e2e_ea597cc14_20260527T191926Z/artifacts/m04n04_i05_current_e2cc7394590f_20260528T144935Z`

Top-level command included:

- `--warm-start-run-dir .../case_artifacts/seed_outputs/mpol=4-ntor=4-7e603170`
  in the generated child lane commands
- `--jax-runtime-seed-spec .../case_artifacts/single_stage_jax_runtime_seed_spec.json`
- `--benchmark-mode`
- `--record-objective-evaluation-trace`
- `--target-lane-accepted-step-sync final-only`

Seed lane:

- Progress file:
  `case_artifacts/seed_outputs/mpol=4-ntor=4-7e603170/outer_optimizer_progress.json`
- Events: `23`
- Completed event: `final_artifact_write_returned`
- Event elapsed: `256.526 s`
- Wrote `results.json`
- Final objective: `1.1250792012684074`
- Final iota: about `-3.07e-18`
- Field error: about `5.885e-05`

CPU reference lane:

- Progress file:
  `case_artifacts/cpu_outputs/mpol=4-ntor=4-b1d67fa4/outer_optimizer_progress.json`
- Events: `61`
- Completed event: `final_artifact_write_returned`
- Event elapsed: `4573.452467849478 s`, about `76:13`
- Wrote `REJECTED.json`, not `results.json`
- Progress trace is still valid evidence for timing and lane completion.

Same-candidate replay lane:

- Progress file:
  `case_artifacts/target_same_candidate_replay_outputs/mpol=4-ntor=4-d847dfe7/outer_optimizer_progress.json`
- Events: `151`
- Completed event: `final_artifact_write_returned`
- Event elapsed: `1213.5411032103002 s`, about `20:14`
- Wrote `REJECTED.json`
- Same-candidate replay was about `3.77x` faster than CPU reference by lane
  elapsed time.

Actual target lane:

- Progress file:
  `case_artifacts/target_outputs/mpol=4-ntor=4-d847dfe7/outer_optimizer_progress.json`
- Events before termination: `53`
- Last event: `lbfgs_main_kernel_started`
- Last event elapsed: `524.9867575448006 s`
- Last event showed `accepted_step_callback: true`
- Last event showed `record_optimizer_state_trace: true`
- GPU utilization was `0%` when checked during the stall.
- Target child CPU was about `198%`.
- Target lane RSS reached about `69.5 GB`.
- `/usr/bin/time` MaxRSS: `69,821,784 KB`
- Target child exceeded the already-long CPU reference wall clock and was
  terminated deliberately to preserve time.

Classification:

- This was not a first-order physics parity mismatch.
- This was not the old unstructured harness crash.
- The useful failure class is target-lane L-BFGS performance/liveness after
  CPU reference and same-candidate replay completed.
- The structured final JSON classified the case as `case-execution-failed`
  after subprocess termination.

### Optimistix L-BFGS strict transfer row

Run directory:

`/root/simsopt_e2e_ea597cc14_20260527T191926Z/artifacts/optimizer_matrix_optimistix_current_e2cc7394590f_20260528T145914Z`

Result:

- Process exit: `0`
- JSON status: `unsupported`
- JSON passed: `false`
- JSON failures: `[]`
- Strict-transfer supported: `false`
- Reason:
  `Optimistix/Equinox scalar predicate handling performs device-to-host transfer before SIMSOPT can hostify result metadata.`

Performance and memory:

| Metric | Value |
| --- | ---: |
| Wall time | `0:07.95` |
| MaxRSS | `751,336 KB` |
| Sampled peak GPU memory | `474 MiB` |

Classification:

- This is an expected unsupported strict-transfer path unless the upstream
  Optimistix/Equinox scalar predicate path becomes strict-clean.
- It should not be reported as a physics parity failure.

## Discoveries and Findings

### The major m04 cost is not just the full ladder

The m04 rung alone is expensive:

- CPU reference lane took about `76:13`.
- Same-candidate replay took about `20:14`.
- The target lane then stalled past the CPU reference wall in the failed e2cc
  run.

This explains the long turnaround. Running the full ladder as the debug loop
would repeatedly spend hours before exposing the same target-lane blocker.

### Objective trace recording accidentally enabled target optimizer callbacks

The key code finding before `13a664f15`:

- `--record-objective-evaluation-trace` caused the target `lbfgs-ondevice`
  route to set target optimizer-state tracing.
- That made `_lbfgsb_accepted_step_observer` non-null.
- The private L-BFGS path then installed accepted-step callback behavior inside
  the compiled target optimizer loop.
- In the failed m04 target progress trace, this was visible as:
  `accepted_step_callback: true` and `record_optimizer_state_trace: true`.

This was both a strict-transfer concern and a performance/liveness concern.
The fix was to make target optimizer-state tracing explicit and keep ordinary
objective-evaluation trace recording out of the target L-BFGS loop.

### m04 e2cc replay evidence narrows the bug

Same-candidate replay completed for m04 e2cc while actual target L-BFGS did
not. That split matters:

- It suggests target objective/gradient evaluation can complete on CUDA at
  m04.
- The stall is more likely in actual target optimizer control, callback/final
  synchronization, or private L-BFGS state handling.
- It is not enough evidence to rewrite physics math or loosen parity
  tolerances.

### m06 and m08 failures were invalid invocation coverage

The earlier m06/m08 failures should not be treated as physics failures.
Current code requires explicit warm-start run directories for high-resolution
outer runs. A runtime seed spec alone is not a valid continuation donor for
those rungs.

Required continuation chain:

- m06 donor depends on a valid current m04 target run directory.
- m08 donor depends on a valid current m06 donor run directory.

So m06/m08 should wait until patched m04 produces a valid target run directory.

### JAX memory and transfer-guard docs matter here

The relevant official JAX docs checked earlier support the current
interpretation:

- `XLA_PYTHON_CLIENT_PREALLOCATE=false` disables default GPU memory
  preallocation, but can increase fragmentation risk.
- `XLA_PYTHON_CLIENT_MEM_FRACTION` controls preallocated memory fraction when
  preallocation is enabled.
- `transfer_guard=disallow` is designed to catch implicit host/device
  transfers; explicit `device_get` is still allowed when intentionally used.

That makes callback-driven D2H traffic in a strict target optimizer path a real
contract issue, not just a harmless diagnostic detail.

## Attempts and Outcomes

| Attempt | Outcome | Current status |
| --- | --- | --- |
| Plan strict CUDA E2E recovery | Plan doc committed | Done |
| Deploy current source to Runpod | e2cc source deployed and used | Superseded by 13a source |
| Run m02 CUDA ondevice | Passed same-candidate parity | Done |
| Run m04 CUDA ondevice on e2cc | CPU and same-candidate lanes completed; target L-BFGS stalled | Classified |
| Classify m04 failure | Identified target optimizer callback/state-trace involvement | Done |
| Patch target optimizer trace gating | Committed `13a664f15` | Done |
| Validate patch locally | Focused and broad helper tests passed | Done |
| Validate patch remotely | Focused helper tests passed on Runpod | Done |
| Rerun full m04 on patched source | Fresh full run failed in seed artifact export with child exit `-15` | Needs seed-harness triage |
| Run m04 target-lane repro on patched source | Active `maxiter=1` target-lane run | Pending |
| Optimistix strict row | Structured `unsupported` JSON emitted | Done |
| m06/m08 continuation | Blocked on valid current m04 target donor | Pending |
| SciPy JAX fullgraph row | Needs current rerun and census classification | Pending |
| Optax L-BFGS row | Needs trace recording diagnostics and current rerun | Pending |
| Stage 2 CUDA E2E | Waiting for m04/m06/m08 and optimizer matrix | Pending |
| Full ladder | Waiting for smaller blockers to clear | Pending |

## Parity Status

| Item | Current verdict | Evidence |
| --- | --- | --- |
| m02 CUDA ondevice | Pass | Same-candidate objective diff `1.07e-14`, gradient diff `4.86e-12` |
| m04 CUDA ondevice e2cc | Not a parity verdict | Target lane stalled after CPU and replay lanes completed |
| m04 CUDA ondevice 13a full parity | Not validly tested yet | Fresh full run failed in seed artifact export with child exit `-15`; active target repro is not a full parity run |
| m04 CUDA ondevice 13a target repro | Pending | Active `maxiter=1` target-lane run has no final artifact yet |
| m06 CUDA ondevice | Not validly tested yet | Needs explicit warm-start donor |
| m08 CUDA ondevice | Not validly tested yet | Needs explicit warm-start donor |
| Optimistix L-BFGS | Unsupported under strict transfer | Structured `status: unsupported` |
| SciPy JAX fullgraph | Unresolved | Prior strict census drift around `pre_newton_state` needs current classification |
| Optax L-BFGS | Unresolved | Needs replay trace recording fix/diagnostics and current rerun |
| Stage 2 CUDA E2E | Not run as final proof | Waiting on lower-level blockers |
| Full ladder | Not run as final proof | Waiting on lower-level blockers |

## Performance Status

| Item | Current result |
| --- | --- |
| m02 ondevice | Correct but slower than CPU: about `5.0x` slower total elapsed |
| m02 ondevice vs old f14 | Improved: wall down about `16.3%`, JAX elapsed down about `30.1%`, main optimizer down about `42.1%` |
| m04 CPU reference | Very expensive: about `76:13` in the completed e2cc run |
| m04 same-candidate replay | About `20:14`, about `3.77x` faster than CPU reference |
| m04 actual target e2cc | Bad: exceeded CPU reference time and stalled at `lbfgs_main_kernel_started` |
| m04 actual target 13a | Pending target repro |
| Optimistix strict unsupported row | Cheap structured classification: `0:07.95` |

## Memory Status

| Item | Current result |
| --- | --- |
| m02 ondevice | MaxRSS `13,733,416 KB`, peak GPU `3509 MiB` |
| m02 old f14 | MaxRSS `13,895,176 KB` |
| m04 e2cc target stall | MaxRSS `69,821,784 KB`; target process reached about `69.5 GB` RSS |
| m04 fresh full patched run | Seed-phase MaxRSS `1,761,036 KB`; no target memory result |
| m04 active target repro | At poll: process RSS about `2,034,148 KB`, GPU about `1529 MiB` |
| Optimistix strict unsupported row | MaxRSS `751,336 KB`, peak GPU `474 MiB` |

The biggest memory risk remains m04 actual target L-BFGS, not m02 and not the
Optimistix unsupported gate.

## What Is Still Left

1. Let the active m04 target-lane reproducer finish or reach timeout.
   - If it returns, classify target-lane liveness, wall time, MaxRSS, GPU peak,
     and final artifact status.
   - If it stalls again, add only the next private L-BFGS probes:
     state build entered/returned, main kernel entered/returned,
     callback/barrier entered/returned.

2. Fix or isolate the fresh full m04 seed-phase SIGTERM.
   - The failed run died before `results.json`, so the full parity wrapper
     cannot be used as signoff until this harness/process issue is understood.

3. Generate valid continuation donors.
   - Use patched m04 target output as the m06 warm-start donor.
   - Use m06 donor output as the m08 warm-start donor.
   - Do not use a runtime seed spec alone for high-resolution continuation.

4. Rerun m06 and m08 on current source.
   - Require explicit warm-start run directories.
   - Capture parity, timing, MaxRSS, GPU memory, source SHA, transfer guard,
     and XLA memory policy.

5. Finish optimizer matrix.
   - `ondevice`: current-source verdict after m04 patch.
   - `scipy-jax-fullgraph`: classify strict census drift as real
     nondeterminism/source difference or over-asserted internal census.
   - `optax-lbfgs`: fix/verify replay trace recording before classifying
     endpoint drift.
   - `optimistix-lbfgs`: keep as structured unsupported unless strict-clean
     support becomes available.

6. Run Stage 2 CUDA E2E.
   - Only after m04/m06/m08 and optimizer matrix have current artifacts.

7. Run the final full ladder.
   - This should be proof, not the debug loop.

## Current Best Assessment

The work has moved from broad failing-run triage into one narrow blocker:
`lbfgs-ondevice` target-lane behavior at m04 and above. m02 proves the current
CUDA path can achieve strict same-candidate parity at small resolution. m04
proves the replay/objective path can complete, but the actual target optimizer
path was not yet production-clean on e2cc.

The `13a664f15` patch removes a real callback/state-trace source of strict
transfer and timing contamination. The active patched m04 rerun is the next
decisive artifact. Until that finishes, m06, m08, Stage 2, and the full ladder
should remain queued rather than launched speculatively.
