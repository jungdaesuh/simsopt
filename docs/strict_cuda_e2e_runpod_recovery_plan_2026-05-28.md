# Strict CUDA E2E Runpod Recovery Plan

> Supersession note, 2026-05-30: this was a recovery plan for an older Runpod
> source/provenance state (`7496577a82b5` and earlier). Do not use the SHA,
> endpoint, or artifact assumptions below as current remediation signoff
> evidence without revalidating them against the active checkout. The current
> remediation CUDA release gate and fail-closed command packet live in
> `docs/jax_port_review_remediation_plan.md` under "Validation caveats" and
> derive the signoff target from the active checkout's `git rev-parse HEAD`.

## Purpose

This file preserves the execution plan used to recover the strict CUDA
single-stage and Stage 2 E2E ladder on Runpod. It exists to prevent stale
verdicts, invalid-run coverage, and optimizer-path failures from being mixed
together.

## Goals

- Deploy the current local diagnostics commit to Runpod and prove artifact
  provenance before making new parity, performance, or memory claims.
- Resolve the `lbfgs-ondevice` production target-lane blocker with the smallest
  CUDA reproducer before spending time on the full ladder.
- Produce valid m04, m06, and m08 single-stage CUDA artifacts with parity,
  timing, RSS, GPU memory, and trace diagnostics.
- Produce an optimizer matrix where `ondevice` and `scipy-jax-fullgraph` have
  current-source pass/fail verdicts, `optax-lbfgs` has current trace evidence,
  and `optimistix-lbfgs` is either strict-clean or explicitly unsupported under
  strict transfer.
- Run Stage 2 CUDA E2E and the final full ladder only after the smaller
  blockers have current-source artifacts.

## Non-Goals

- Do not loosen tolerances to convert mismatches into passes.
- Do not hide strict-transfer failures with host fallbacks.
- Do not treat a runtime seed spec alone as valid high-resolution continuation
  evidence when the current runner requires a warm-start run directory.
- Do not classify public optimizer trajectory drift as physics drift unless
  same-candidate objective and gradient parity fails.
- Do not use remote artifacts whose recorded source does not match the deployed
  source being tested.

## Current Context

The bullets in this section are historical plan-creation facts for the older
Runpod recovery attempt. They are not current remediation signoff facts unless
re-probed against the active checkout.

- Local checkout: branch `gpu-purity-stage2-20260405`, HEAD
  `7496577a82b5`.
- Local dirty state at plan creation: untracked `.antigravitycli/`, `.conda/`,
  `analysis/`, and `runs/`.
- Runpod SSH access:
  `ssh -i /Users/suhjungdae/.runpod/ssh/RunPod-Key-Go -p 16628 root@154.54.102.24`.
- Runpod root:
  `/root/simsopt_e2e_ea597cc14_20260527T191926Z`.
- Runpod Python:
  `/root/simsopt_canary_local/venv/bin/python`.
- Runpod source copy currently has no `.git` directory at
  `/root/simsopt_e2e_ea597cc14_20260527T191926Z/src`.
- Runpod root provenance currently records `repo_sha.txt=f264e386d`, while the
  most recent local diagnostics commit is `7496577a82b5`. Artifact subfolders
  have their own `provenance.txt`; the root marker is not enough for signoff.
- No active `single_stage`, `validation`, `continuation`, or `python` workload
  was observed on Runpod during plan creation.
- Current repo docs define the single-stage ladder rungs:
  `m04n04-i05-useful`, `m06n06-i10-serious`, and
  `m08n08-i20-release-small`.
- Current code rejects high-resolution outer runs without
  `--warm-start-run-dir`; a `--jax-runtime-seed-spec` alone is only a runtime
  startup guess for those rungs.
- Current code can emit structured `status: unsupported` JSON for
  `optimistix-lbfgs` under CUDA strict-transfer guard.
- `benchmarks/validation_ladder_common.py` supports source copies without
  `.git` when `SIMSOPT_REPO_SHA` and `SIMSOPT_GIT_STATUS_SHORT` are exported.
- The JAX GPU memory policy variables captured by the current runner include
  `XLA_PYTHON_CLIENT_PREALLOCATE`, `XLA_PYTHON_CLIENT_MEM_FRACTION`,
  `XLA_CLIENT_MEM_FRACTION`, `XLA_PYTHON_CLIENT_ALLOCATOR`, `TF_GPU_ALLOCATOR`,
  `SIMSOPT_JAX_GPU_PREALLOCATE`, `SIMSOPT_JAX_GPU_MEM_FRACTION`,
  `SIMSOPT_JAX_GPU_ALLOCATOR`, and `SIMSOPT_TF_GPU_ALLOCATOR`.
- Prior remote artifacts are useful diagnostics, but not final evidence for
  the current local diagnostics commit.

## Evidence Reviewed

- `benchmarks/single_stage_init_parity.py` validates high-resolution
  `--warm-start-run-dir` donors and rejects high-resolution outer runs that
  only provide `--jax-runtime-seed-spec`.
- `benchmarks/single_stage_init_parity.py` writes `status: unsupported` for
  `optimistix-lbfgs` under CUDA strict transfer guard.
- `benchmarks/validation_ladder_common.py` reads `SIMSOPT_REPO_SHA` and
  `SIMSOPT_GIT_STATUS_SHORT` before falling back to `git`, which is required
  for Runpod source directories without `.git`.
- Official JAX GPU memory allocation docs state that JAX preallocates GPU
  memory by default; `XLA_PYTHON_CLIENT_PREALLOCATE=false` disables
  preallocation but can increase fragmentation, and
  `XLA_PYTHON_CLIENT_MEM_FRACTION` controls the preallocated fraction when
  preallocation is enabled.
- Existing repo docs define the m04/m06/m08 ladder, optimizer matrix,
  strict-transfer constraints, and performance/memory evidence requirements in
  `docs/full_repo_banana_e2e_cpu_gpu_test_plan_2026-05-19.md`.

## Rationale

The fastest path is to stop using the full ladder as the debug loop. The
smallest failing target-lane reproducer isolates the production
`lbfgs-ondevice` path, which also affects the single-stage ladder and can affect
Stage 2 through `target_minimize(..., method="lbfgs-ondevice")`. Once that path
has a current-source verdict, donor generation, optimizer matrix classification,
Stage 2 E2E, and the full ladder can run without repeatedly timing out on the
same root blocker.

## Assumptions

- The Runpod endpoint remains reachable at the SSH host and port above.
- The Runpod venv already has the required JAX, CUDA, Optax, Optimistix,
  Equinox, SciPy, and SIMSOPT dependencies installed.
- Local `7496577a82b5` contains the latest target-lane diagnostics that need to
  be deployed before the next CUDA run.
- The current production optimizer method remains `lbfgs-ondevice` unless the
  code proves it unsupported under the strict CUDA contract.
- Current validation commands must force
  `PYTHONPATH=/Users/suhjungdae/code/columbia/simsopt-jax/src` locally to avoid
  importing a sibling checkout.

## Implementation Plan

1. Establish provenance and deploy the current source.
   - [ ] On Runpod, confirm the machine is idle:
     `pgrep -af 'single_stage|validation|continuation|python'`.
   - [ ] Create a new source directory named with the local short SHA, for
     example
     `/root/simsopt_e2e_ea597cc14_20260527T191926Z/src_7496577a82b5`.
   - [ ] Before syncing, verify the deploy set does not include secret env
     files: `git ls-files | rg '(^|/)\.env($|\.)'` may only report
     `.env.example`.
   - [ ] Sync only the intended source tree to that directory. Exclude `.git`,
     `.env`, `.env.*`, `.antigravitycli`, `.conda`, `analysis`, `runs`,
     `__pycache__`, and `.pytest_cache`; keep `.env.example` if it is present.
   - [ ] Write `repo_sha.txt`, `local_git_status_short.txt`, and an artifact
     `provenance.txt` that record `repo_sha=7496577a82b5`, the exact source
     path, Python path, JAX/CUDA package versions, transfer guard, XLA flags,
     preallocation policy, and GPU facts.
   - [ ] Export `SIMSOPT_REPO_SHA=7496577a82b5` and
     `SIMSOPT_GIT_STATUS_SHORT="$(git status --short --untracked-files=no)"`
     for every benchmark command launched from a source copy without `.git`.
   - [ ] Refuse to launch new signoff runs if the remote source provenance does
     not equal local HEAD.

2. Reproduce the production `lbfgs-ondevice` blocker with the smallest CUDA
   target lane.
   - [ ] Run `benchmarks/single_stage_init_parity.py` on CUDA with
     `--mpol 2 --ntor 2 --nphi 31 --ntheta 16 --maxiter 1
     --optimizer-backend ondevice --benchmark-mode
     --record-objective-evaluation-trace`.
   - [ ] Use hard case and target-case timeouts, and capture
     `/usr/bin/time -v`, stdout, stderr, `nvidia-smi` monitor CSV, progress
     JSONs, and final `single_stage_optimizer.json`.
   - [ ] Require the new diagnostics events from `7496577a82b5`, including
     `target_lane_reporting_snapshot_started`,
     `target_lane_reporting_snapshot_returned`,
     `target_lane_initial_objective_started`, and
     `target_lane_init_reporting_snapshot_returned`.
   - [ ] Classify the result as one of: pass, parity mismatch,
     strict-transfer failure, compile stall, optimizer-loop stall, callback or
     final-sync stall, or artifact/provenance failure.

3. Add only targeted `lbfgs-ondevice` probes if the smallest run stalls.
   - [ ] Instrument private L-BFGS phase 2 entry and return.
   - [ ] Instrument state-build entry and return.
   - [ ] Instrument main kernel entry and return.
   - [ ] Instrument callback and final barrier entry and return.
   - [ ] Inspect result conversion and status handling in
     `src/simsopt/geo/optimizer_jax_private/_result_converters.py`.
   - [ ] Inspect optimizer state parity and termination semantics in
     `src/simsopt/geo/optimizer_jax_private/_lbfgs.py`.
   - [ ] Keep instrumentation diagnostic-only unless the root cause requires a
     code fix.

4. Fix the root cause, then validate locally before redeploying.
   - [ ] Add or update focused tests for the exact failing path.
   - [ ] Run focused local validation with
     `PYTHONPATH=/Users/suhjungdae/code/columbia/simsopt-jax/src`.
   - [ ] Run `python -m ruff format --check` and `python -m ruff check` on
     touched files if Ruff is available in the active environment.
   - [ ] Run `git diff --check`.
   - [ ] Commit only the intended source/test/docs slice when requested; keep
     unrelated untracked directories out of the commit.

5. Generate valid m06 and m08 continuation donors.
   - [ ] Stop using the invalid high-resolution invocation that passes only
     `--jax-runtime-seed-spec`.
   - [ ] Produce explicit warm-start run directories for m06 and m08 using
     `examples/single_stage_optimization/SINGLE_STAGE/run_single_stage_continuation.py`.
   - [ ] Prefer target-resolution donor generation from the runtime spec over
     coarse projection if coarse projection collapses or stalls.
   - [ ] Require each donor `results.json` to be non-init-only, finite
     `FINAL_IOTA`, finite `FINAL_G`, `HARDWARE_CONSTRAINTS_OK != false`,
     `SELF_INTERSECTING != true`, and within the seed iota quality gate.
   - [ ] Record donor timing, RSS, GPU memory, source SHA, and command.
   - [ ] Update the runner or docs if the documented seed contract still
     implies that a runtime seed spec alone is sufficient for high-resolution
     outer runs.

6. Rerun single-stage ladder rungs on current source.
   - [ ] Rerun `m04n04-i05-useful` first and require structured
     `single_stage_cuda.json`.
   - [ ] For m04, separately classify target-native replay parity and
     same-candidate replay diagnostics; do not collapse replay mismatch into
     liveness.
   - [ ] Rerun `m06n06-i10-serious` with its validated warm-start run
     directory.
   - [ ] Rerun `m08n08-i20-release-small` with its validated warm-start run
     directory.
   - [ ] For every rung, capture final parity, timing, `/usr/bin/time -v`
     MaxRSS, sampled peak GPU memory, transfer guard, XLA flags, device list,
     and source SHA.

7. Rerun the optimizer matrix on current source.
   - [ ] Run `ondevice` and require a current-source pass or a structured
     failure with L-BFGS diagnostics.
   - [ ] Run `scipy-jax-fullgraph` and classify any strict census drift as
     either real source nondeterminism or an over-asserted pre-Newton internal
     census.
   - [ ] Run `optax-lbfgs` with objective-evaluation trace recording; require
     same-candidate objective and gradient parity before accepting final metric
     drift as an optimizer-control split.
   - [ ] Run `optimistix-lbfgs` under strict transfer and require either
     strict-clean execution or structured `status: unsupported`.
   - [ ] Check the ladder aggregator treats expected `unsupported` Optimistix
     status as unsupported metadata even though the per-rung JSON has
     `passed: false`; it must not be reported as a physics-parity failure.

8. Rerun Stage 2 CUDA E2E with explicit memory policy evidence.
   - [ ] Run the Stage 2 CUDA E2E artifact after `lbfgs-ondevice` has a current
     small-case verdict.
   - [ ] Record `XLA_PYTHON_CLIENT_PREALLOCATE`,
     `XLA_PYTHON_CLIENT_MEM_FRACTION`, `XLA_CLIENT_MEM_FRACTION`,
     `XLA_PYTHON_CLIENT_ALLOCATOR`, `TF_GPU_ALLOCATOR`,
     `SIMSOPT_JAX_GPU_PREALLOCATE`, `SIMSOPT_JAX_GPU_MEM_FRACTION`,
     `SIMSOPT_JAX_GPU_ALLOCATOR`, `SIMSOPT_TF_GPU_ALLOCATOR`,
     `JAX_PLATFORMS`, `SIMSOPT_JAX_TRANSFER_GUARD`, and `XLA_FLAGS`.
   - [ ] Capture `nvidia-smi` monitor CSV, `/usr/bin/time -v`, stdout, stderr,
     and final Stage 2 JSON.
   - [ ] Classify OOM, preallocation, transfer, target-objective, and physics
     parity failures separately.

9. Run the final full ladder.
   - [ ] Launch the full ladder only after m04, m06, m08, optimizer matrix,
     and Stage 2 have current-source artifacts.
   - [ ] Pull the full artifact packet back into `runs/` without staging it for
     commit.
   - [ ] Produce a summary table with pass/fail/unsupported, parity deltas,
     wall time, MaxRSS, sampled peak GPU memory, source SHA, and artifact path
     for every rung.

## Validation Plan

- [ ] Remote source provenance equals local HEAD before every signoff run:
  `7496577a82b5` or a later explicit local commit.
- [ ] Benchmarks launched from source copies without `.git` export
  `SIMSOPT_REPO_SHA` and `SIMSOPT_GIT_STATUS_SHORT`, and the output JSON
  records those exact values.
- [ ] Runpod preflight reports no active workloads before deployment.
- [ ] Small `optimizer_matrix_ondevice` or equivalent m02 target-lane run emits
  final JSON or a structured timeout with the new `7496577a82b5` diagnostics.
- [ ] Local focused tests pass for any code fix before redeploy.
- [ ] m04 emits structured target-native replay and same-candidate diagnostics.
- [ ] m06 and m08 are run with explicit validated warm-start run directories.
- [ ] Optimizer matrix artifacts contain trace evidence for public optimizers
  and strict-transfer classification for Optimistix.
- [ ] Stage 2 CUDA E2E records memory/preallocation policy and does not reuse
  stale artifact provenance.
- [ ] Final full ladder summary has parity, performance, and memory columns for
  every required rung.

## Risks and Mitigations

- Risk: Remote source copy has no `.git`, so `git rev-parse` cannot prove SHA.
  Mitigation: treat `repo_sha.txt`, deployed directory name, rsync manifest,
  and artifact `provenance.txt` as required source identity; fail closed if
  they are absent or mismatched.
- Risk: `lbfgs-ondevice` stalls after high compile cost and burns GPU hours.
  Mitigation: reproduce with the smallest CUDA target-lane run and hard
  timeout before m04, m06, m08, Stage 2, or the full ladder.
- Risk: Optimizer drift is misclassified as objective or physics drift.
  Mitigation: require same-candidate objective and gradient parity before
  accepting or rejecting public optimizer final metric drift.
- Risk: Donor generation produces a bootable but low-quality seed.
  Mitigation: require explicit warm-start run directories and seed quality
  checks before high-resolution rungs.
- Risk: Stage 2 memory failures are hidden by default JAX preallocation.
  Mitigation: record memory policy variables and sampled `nvidia-smi` output
  for every Stage 2 CUDA E2E attempt; when disabling preallocation, record that
  official JAX docs warn this can increase fragmentation.
- Risk: Long-running results are lost between sessions.
  Mitigation: maintain this plan plus per-run artifact directories and summary
  files as the live status ledger.

## Completion Criteria

- [ ] Runpod source and every signoff artifact record the same current source
  SHA.
- [ ] `lbfgs-ondevice` has a current-source pass or a root-caused code fix with
  focused validation.
- [ ] `m04n04-i05-useful` passes or has a structured current-source parity
  failure with replay diagnostics.
- [ ] `m06n06-i10-serious` and `m08n08-i20-release-small` use validated
  warm-start run directories and produce structured current-source artifacts.
- [ ] `scipy-jax-fullgraph` census status is classified as real drift or
  over-assertion with evidence.
- [ ] `optax-lbfgs` current-source trace evidence proves or falsifies
  same-candidate objective and gradient parity.
- [ ] `optimistix-lbfgs` is strict-clean or expected-unsupported under CUDA
  strict transfer.
- [ ] Stage 2 CUDA E2E records pass/fail with memory policy, RSS, and GPU
  memory evidence.
- [ ] Final full ladder summary reports parity, performance, and memory for
  all required rows.

## Open Questions

- Should the high-resolution donor contract require at least one accepted
  optimizer iteration, or are the current `results.json` quality gates
  sufficient?
- Should the docs be updated immediately to make `--warm-start-run-dir`
  mandatory for high-resolution outer runs, or should the runner accept a
  richer runtime seed spec with explicit continuation provenance?
- If `lbfgs-ondevice` remains strict-transfer clean but slow, what runtime
  threshold should distinguish acceptable performance from a release blocker?
- Should `optimistix-lbfgs` remain in the full ladder as expected-unsupported
  metadata, or move to a separate optimizer-library diagnostic report?
