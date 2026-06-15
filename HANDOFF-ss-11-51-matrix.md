# HANDOFF — Single-stage 11-vs-51 matrix → host-driven SciPy + fair comparison

> Last updated: 2026-06-15 07:18 EDT · Status: Migration + inner-Boozer fix + fair-comparison protocol +
> compile-diagnostics wiring + donor trial-policy fix all COMMITTED (code-bearing commit `5f14a1463`; this
> handoff is the operational launch-state source of truth).
> **CORRECTION (06-15 doc-review): the iota15 fixture `benchmarks/fixtures/single_stage_seed_iota15` is
> mpol10/ntor10, nfp5, hardware-clean (vol 0.0937), but it is NOT a usable mpol10 `--warm-start-run-dir`
> donor for `single_stage_init_parity.py`.** It has `biot_savart_opt.json`, `results.json`, and
> `single_stage_jax_runtime_spec.json`, but no `surf_opt.json`; more importantly, the harness rejects
> high-resolution outer runs (`maxiter > 0`, max(mpol, ntor) > 4) unless `--warm-start-run-dir` is supplied
> from a validated continuation donor. A contract-valid donor still gates the mpol10 fair-compare/speed run,
> but it must still be built before the mpol10 fair compare.
> fullgraph-51 advances past iter-4 (slow, not wedged). GPU recompile pathology is measurable via the
> compile diagnostics. Donor `54462557` FAILED after 4m27s because the launcher inherited
> `run_single_stage_continuation.py`'s default `validated-fast` trial policy; committed fix `5f14a1463`
> defaults donor builds to `DONOR_TRIAL_POLICY=none` and passes `--trial-policy "${DONOR_TRIAL_POLICY}"`.
> **NEW 06-15 RUNPOD RESULT:** the native continuation route is now disproven as a fast path: exact documented
> RunPod ladder (`--trial-policy none`) failed at the first coarse rung because wrapper defaults left
> `--coarse-maxiter=1`; the corrected budget probe (`--coarse-maxiter 50 --medium-maxiter 50
> --prefinal-maxiter 50`) also failed with native L-BFGS-B `ABNORMAL` / optimizer status `2`, final iota
> only ~0.0035, and no accepted `results.json`. JAX preserves the iota15 branch, but the current A100 PCIe
> pod has only 117 GB container RAM and the `run` Newton policy hit cgroup OOM after Boozer init. Next viable
> donor launch is high-memory **JAX** donor, not native donor. **Submitted:** corrected Perlmutter high-memory JAX
> donor probe jobs from checkout `/pscratch/sd/j/jungdae/ss-jax-donor-306d53bdc-20260615T074239Z/checkout`:
> primary full-node 4 h / `DONOR_MAXITER=60` job `54483185`, backfill-friendly non-exclusive `mem=180G` 4 h /
> `DONOR_MAXITER=60` job `54483605`, and backup full-node 12 h / `DONOR_MAXITER=300` job `54482812`.
> Added CPU-JAX fallback. First full CPU-JAX job `54484362` failed before physics because Slurm stdout landed
> inside the checkout and tripped the clean-source guard (`?? slurm-54484362.out`); stdout/stderr paths were
> moved out of the checkout and corrected CPU-JAX full job `54485382` ran to Newton completion but FAILED with
> an OOM under the 180G cgroup during XLA constant folding / initial objective work. Debug smoke `54484779`
> proved the CPU-JAX runtime path reaches Boozer Newton setup before its 30 min timeout. Live at 06:29 EDT:
> `54483185` (GPU full-node) and `54485382` (CPU-JAX 180G) both preserved the iota15 branch through
> BFGS to `iota=0.1500517720536309`, `objective=6.075541041282424e-06`, `grad_inf=3.56e-09`, and are waiting at
> `before_boozer_newton` with `newton_polish_policy=run`. `54483605` also reached the same boundary after it
> started, then was cancelled intentionally as duplicate GPU burn (`CANCELLED+`, elapsed 13:41). `54482812`
> remains the pending 12 h GPU backup. New full-memory CPU-JAX fallback `54488062` was submitted at 07:18 EDT
> (`ReqTRES=cpu=32,mem=487802M,node=1`, `DONOR_MAXITER=60`) to test whether the CPU-JAX failure is purely
> memory-bounded.
> Previous job `54477744` was cancelled before start because its submitted
> script still used the bad `--nphi 64 --ntheta 32` runtime-spec mismatch. **RunPod H100 probe harvested, no
> donor:** pod `ecxt9xwaudcejo`; first command
> failed before optimization because the fixture runtime spec expects `quadrature.nphi=255`, not `--nphi 64`;
> corrected relaunch `/workspace/runpod_himem_jax_mpol10_donor_306d53bdc_20260615T083623Z_nphi255b` advanced to
> `optimizer_output_dir_ready` / `before_boozer_newton`, then the pod exited before a clean
> `single_stage_exit_status.txt` or donor artifact. Slim artifacts are local under
> `.artifacts/runpod_himem_jax_mpol10_donor_20260615_h100_ecxt9xwaudcejo/`; pod is stopped.

## 1. Goal
Single-stage parity/performance matrix on the clean JAX-port branch: native cpp/CPU reference vs JAX across
the 11-dim reduced (`scipy-jax`) and 51-dim full-space (`scipy-jax-fullgraph`) formulations, {cpu, gpu},
mpol2 (smoke) + mpol10 (production). `ondevice` removed (OOM monolith). User wants a **defensible
(fair) cpu-vs-gpu comparison**, not confounded numbers, and to push production now.
**Definition of done:** per cell — parity (JAX vs cpp), walltime, RSS/GPU-mem, iters, physics, pass/fail —
with cpu-vs-gpu measured fairly (same node, capped threads, replay parity); docs/memory updated.

## 2. Where we are right now
All code needed before the next donor submission is committed (minimum code-bearing commit `5f14a1463`): the 8-cell host-driven
matrix, the fullgraph inner-Boozer fix, the thread-cap fairness fix, the same-node fair-compare launcher +
tests, the compile-diagnostics wiring (`--record-jax-compile-diagnostics`), and the donor trial-policy fix.
Two background efforts:
the **Perlmutter donor `54462557`** (**FAILED**; see NEXT ACTION #1), the **RunPod mpol2 smokes** (DONE
— see §5), and the **RunPod mpol10 donor probes** (FAILED — see §5). The immediate mpol10 fair-compare/speed
step is still **donor-gated**. The iota15 fixture is a useful mpol10 Stage 2 seed/runtime-spec fixture, but
not a continuation donor accepted by the high-resolution harness contract. Native continuation from this seed
falls to the near-zero-iota branch; JAX preserves the branch but needs more host/container memory than the
current A100 PCIe pod exposed. Current attempts are **Perlmutter job `54483185`** (GPU full-node 4 h,
`DONOR_MAXITER=60`, RUNNING), **full-memory CPU-JAX fallback `54488062`** (`ReqTRES=cpu=32,mem=487802M,node=1`,
12 h, pending), and backup job `54482812` (GPU full-node 12 h / `DONOR_MAXITER=300`, pending backup).
Corrected CPU-JAX 180G job `54485382` completed Newton but failed with an OOM during XLA constant folding /
initial objective work, so the full-memory CPU-JAX fallback is the next CPU path. Midmem GPU job `54483605`
reached the same BFGS/Newton boundary and was cancelled as a duplicate. First CPU-JAX full job `54484362`
failed only because Slurm stdout dirtied the checkout; it did
not reach physics. Debug smoke `54484779` reached `before_boozer_newton` and timed out without OOM. These jobs supersede
cancelled job `54477744`, which had the wrong quadrature flags. The harvested RunPod H100 probe confirms
the RunPod command must match the fixture runtime spec:
`--nphi 255 --ntheta 64` (`single_stage_jax_runtime_spec.json`: `quadrature.nphi=255`,
`quadrature.ntheta=64`); `--nphi 64 --ntheta 32` fails before optimization. The corrected H100 probe was
interrupted before completion at `before_boozer_newton` (no clean exit status, no donor), so the mpol10
fair-compare remains donor-gated on Perlmutter or another bounded high-memory JAX attempt. If a contract-valid
donor completes, run CPU/CUDA fair compare from that donor. If the Perlmutter job fails by memory, use a larger
RunPod class only after confirming container RAM >120 GB, or make a code-level seed-preservation fix.

## 3. NEXT ACTIONS (start here on resume)
0. [ ] **Monitor high-memory JAX donor jobs `54483185`, `54488062`, and `54482812`, then CPU/CUDA compare.** Native donor
       continuation is now a
       dead end for this seed (falls to iota ~0.0035 and writes `REJECTED.json`). Build the donor with the JAX
       path that preserves the iota15 branch, but run it on a node/container with enough host RAM for the dense
       Newton/target-lane graph. Current Perlmutter launch:
       `ssh perlmutter 'squeue -j 54483185,54488062,54482812 -o "%i %j %T %M %L %R"; sacct -j 54483185,54488062,54485382,54484362,54484779,54483605,54482812 -X -o JobID,JobName,State,Elapsed,ExitCode%20,Start,End -n'`.
       Primary full-node: `54483185` (`DONOR_MAXITER=60`, 4 h; RUNNING since 2026-06-15 03:04 PDT).
       Full-memory CPU-JAX fallback: `54488062` (`DONOR_MAXITER=60`, 12 h, script
       `jax_mpol10_donor_cpujax_180g.slurm` with sbatch overrides `--exclusive --mem=0 -t 12:00:00`,
       `ReqTRES=cpu=32,mem=487802M,node=1`; pending `Resources` at submit). Backup full-node:
       `54482812` (`DONOR_MAXITER=300`, 12 h; pending). Cancelled duplicate midmem GPU job:
       `54483605` (`DONOR_MAXITER=60`, 4 h, script `jax_mpol10_donor_midmem_180g.slurm`, `--mem=180G`,
       non-exclusive; reached the same BFGS/Newton boundary, then `CANCELLED+` at 13:41 elapsed).
       Failed CPU-JAX 180G job: `54485382` reached Newton completion (`solve_success=true`, Newton iterations 6,
       `grad_inf=8.34e-12`), then OOM-killed during XLA constant folding / initial objective work
       (`slow_operation_alarm` on `f64[1323,663]` dot; Slurm OOM event; validation `valid_count=0`).
       First CPU-JAX full job `54484362` failed before physics because `slurm-54484362.out` was written inside
       the checkout and tripped the clean-source guard; fixed by moving Slurm stdout/stderr out of the checkout
       and resubmitting as `54485382`. CPU-JAX debug smoke `54484779` timed out at the Newton boundary but
       proved setup + branch preservation path through BFGS.
       Live useful signal from `54483185`, `54485382`, and `54483605`: all reached
       `before_boozer_newton` after BFGS with `iota=0.1500517720536309`, `objective=6.075541041282424e-06`,
       and `grad_inf=3.5611373073486663e-09`. This is not a donor yet; wait for artifacts/exit status.
       If one starts and reaches a useful donor/result first, cancel the others to avoid duplicate GPU burn.
       Script:
       `/pscratch/sd/j/jungdae/ss-jax-donor-306d53bdc-20260615T074239Z/slurm/jax_mpol10_donor.slurm`.
       RunPod H100 probe artifacts:
       `.artifacts/runpod_himem_jax_mpol10_donor_20260615_h100_ecxt9xwaudcejo/` (pod `ecxt9xwaudcejo` stopped).
       Command shape:
       ```bash
       export SIMSOPT_BACKEND_MODE=jax_gpu_parity
       export SIMSOPT_BACKEND_STRICT=1
       export SIMSOPT_JAX_TRANSFER_GUARD=disallow
       export SIMSOPT_JAX_PLATFORM=cuda
       export JAX_ENABLE_X64=1
       export JAX_PLATFORMS=cuda,cpu
       export XLA_PYTHON_CLIENT_PREALLOCATE=false
       unset LD_LIBRARY_PATH

       python examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
         --backend jax --optimizer-backend scipy-jax \
         --record-jax-compile-diagnostics \
         --mpol 10 --ntor 10 --nphi 255 --ntheta 64 \
         --maxiter <bounded donor budget, e.g. 60 first> \
         --target-lane-boozer-bfgs-maxiter 1500 \
         --target-lane-boozer-newton-polish-policy run \
         --target-lane-boozer-newton-maxiter 50 \
         --stage2-bs-path benchmarks/fixtures/single_stage_seed_iota15/biot_savart_opt.json \
         --jax-runtime-seed-spec <compiled iota15 mpol10 runtime spec> \
         --output-root <donor-output-root>
       ```
       Expected donor output: highest-mpol run dir contains `surf_opt.json`, `results.json`,
       `biot_savart_opt.json`; `results.json` has finite `FINAL_IOTA`/`FINAL_G`,
       `HARDWARE_CONSTRAINTS_OK=True`, not self-intersecting, and iota-quality pass. Then run the fair-compare
       harness twice on the same pod with that donor as `--warm-start-run-dir`: first `--platform cpu`
       (`SIMSOPT_BACKEND_MODE=jax_cpu_parity`, `JAX_PLATFORMS=cpu`), then `--platform cuda`
       (`SIMSOPT_BACKEND_MODE=jax_gpu_parity`, `JAX_PLATFORMS=cuda,cpu`, `JAX_ENABLE_X64=1`,
       `--record-jax-compile-diagnostics`). Expected compare output: replay/same-candidate parity,
       per-iter CPU vs GPU timing, compile-count diagnostics, and GPU memory. Full convergence is useful but
       not required for the parity/per-iteration speed answer.
1. [ ] **Do not resubmit the native Perlmutter donor unchanged.** Donor `54462557` failed:
       `sacct` = `FAILED`, `ExitCode=1:0`, elapsed `00:04:27`. It ran only `trial_policy=validated-fast`,
       mpol2 coarse, `maxiter=1`, `--minimal-artifacts`; validation then failed because no `results.json`
       snapshot / finite donor metrics were present. Local fix: `single_stage_continuation_donor.slurm`
       defaults `DONOR_TRIAL_POLICY=none` and passes `--trial-policy "${DONOR_TRIAL_POLICY}"`; regression
       test added in `tests/integration/test_continuation_donor_backend_contract.py` (21-test slice passes).
       A later RunPod replay showed that even with `--trial-policy none` and longer non-final rung budgets,
       native continuation stays on the low-iota branch and writes `REJECTED.json`. So the Slurm donor launcher
       is still useful as setup/reference, but not sufficient as a launch recipe for this iota15 donor unless
       the backend/seed-handling route changes. When a JAX donor completes, find the mpol10 warm-start:
       `ssh perlmutter 'ls -dt /pscratch/sd/j/jungdae/<fixed-run-root>/runs/<new-jobid>/continuation_outputs/*'`
       (the highest-mpol rung dir).
       **Do not use the iota15 fixture as a donor-free mpol10 `--stage2-bs-path` speed probe.** Verified
       2026-06-15: `single_stage_init_parity.py --platform cpu --optimizer-backend scipy-jax --stage2-bs-path
       benchmarks/fixtures/single_stage_seed_iota15/biot_savart_opt.json --mpol 10 --ntor 10 --maxiter 1
       --record-jax-compile-diagnostics --output-json /tmp/ss_iota15_probe_invalid.json` fails before
       optimization with `ValueError: single_stage_init_parity high-resolution outer runs require
       --warm-start-run-dir from a validated continuation donor`.
       **Exact trigger** (`_is_high_resolution_outer_run`, init_parity.py:123): `maxiter > 0 AND
       max(mpol,ntor) > MAX_COLD_SEED_OUTER_RUN_RESOLUTION (=4)`. So donor-free runs are ONLY possible at
       `max(mpol,ntor) ≤ 4` or `maxiter = 0` (init-only) — neither is a production-res optimization speed
       test. There is no donor-free shortcut to the mpol10 GPU-vs-CPU answer short of changing the harness
       contract (which deliberately rejects fixture-only seeds — they don't prove continuation-branch
       preservation). And even WITH a donor, `_require_supported_single_stage_seed_contract` (:1616→:1483)
       rejects it unless HARDWARE_CONSTRAINTS_OK, not SELF_INTERSECTING, and iota-quality pass.
2. [ ] **Use a checkout that includes at least code-bearing commit `5f14a1463` for fair-compare.** The active
       JAX-donor checkout
       `/pscratch/sd/j/jungdae/ss-jax-donor-306d53bdc-20260615T074239Z/checkout` is at `306d53bd` and already
       contains `benchmarks/perlmutter/single_stage_fair_compare_gpu.slurm`,
       `FAIR_WARM_START_RUN_DIR`, and `--record-jax-compile-diagnostics`. It is acceptable for fair-compare
       once a valid donor exists. If staging a fresh checkout instead, use the current branch HEAD (doc-only
       commits are fine) and always run `git submodule update --init --recursive` (MANDATORY — bundle lacks the
       C++ submodules).
3. [ ] **Submit the same-node fair-compare** (the defensible cpu-vs-gpu + replay-parity run):
       `sbatch -A m4680_g --export=ALL,REPO_ROOT=<new-checkout>,RUN_ROOT=<runs-OUTSIDE-checkout>,FAIR_WARM_START_RUN_DIR=<donor-warmstart> <new-checkout>/benchmarks/perlmutter/single_stage_fair_compare_gpu.slurm`
       It runs fullgraph mpol10 under `--platform cpu` then `--platform cuda` on one GPU node.
       ⚠️ Heads-up: fullgraph converges SLOWLY (~4–11 min/outer-iter, slowing) — mpol10 may run many hours
       and may NOT reach `passed=true` in walltime. The **replay parity is convergence-independent** and is
       the robust deliverable; per-iter throughput is the perf metric. Consider looser inner budgets
       (`FAIR_BOOZER_BFGS_MAXITER`, `FAIR_NEWTON_MAXITER`) if a converged end-state is required.
4. [x] **DONE** — corrected memory `project_fullgraph51_gate_outcome.md`: the "hard iter-4 stall" claim was
       WRONG; fullgraph advances past iter-4 (slow line search), proven uninterrupted (iters 1–5, iter-5 J
       matches cpp). Memory now reflects "very slow, not wedged."
5. [x] **DONE (commit `77655d960`)** — wired `--record-jax-compile-diagnostics` into `single_stage_init_parity.py`
       (default-off, gated `backend == "jax"`); the fair launcher passes it on both lanes by default
       (`FAIR_RECORD_COMPILE_DIAGNOSTICS=1`, `=0` to disable). Full-chain test
       `tests/integration/test_single_stage_init_parity_compile_diagnostics.py`. This gives compile-vs-steady-state
       separation + GPU XLA recompile counts for the fair comparison.
6. [ ] **GPU production cells**: deferred until the scipy-jax GPU recompile pathology is addressed.

## 4. Environment & how to run
- cwd/repo/branch: `/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean` / `pr/jax-port-clean`
- **Code-bearing commit `5f14a1463`** (donor trial-policy fix on top of compile-diagnostics wiring). Current
  branch HEAD may include doc-only launch-state commits; preserve unrelated untracked `HANDOFF.md` from the
  parallel session.
- **Perlmutter cert**: `sshproxy -u jungdae` (user-run, interactive password+OTP). Valid until
  **2026-06-15 20:26 EDT**; re-run if `ssh perlmutter` gives "Permission denied (publickey)".
- **Perlmutter staging (donor)**: bundle `/pscratch/sd/j/jungdae/ss-prod-94f6ea838.bundle`; checkout
  `/pscratch/sd/j/jungdae/ss-prod-94f6ea838-20260615T003257Z/checkout` (@94f6ea838); RUN_ROOT
  `.../ss-prod-94f6ea838-20260615T003257Z/runs`. Donor job **54462557** (`-A m4680`, native_cpu) FAILED.
  Background poll IDs are not authoritative; verify live state with direct `sacct` for any new Perlmutter job.
- **Perlmutter active JAX donors**: primary full-node job `54483185` (`-A m4680_g`, 4 h, `DONOR_MAXITER=60`,
  RUNNING), full-memory CPU-JAX fallback `54488062` (`-A m4680`, `-C cpu`, `--exclusive --mem=0`, 12 h,
  `DONOR_MAXITER=60`, pending; `ReqTRES=cpu=32,mem=487802M,node=1`), and backup full-node job `54482812`
  (`-A m4680_g`, 12 h, `DONOR_MAXITER=300`, pending). Corrected CPU-JAX 180G job `54485382` reached Newton
  completion but OOM-killed during XLA constant folding / initial objective work (`valid_count=0`). Midmem job
  `54483605` (`-A m4680_g`, `--mem=180G`, non-exclusive) reached the same BFGS/Newton boundary and was
  cancelled as duplicate GPU burn. First CPU-JAX full job `54484362` failed before physics on the clean-source guard because Slurm
  stdout landed in the checkout; debug smoke `54484779` reached the Newton boundary and timed out. All use checkout
  `/pscratch/sd/j/jungdae/ss-jax-donor-306d53bdc-20260615T074239Z/checkout`, runs
  `/pscratch/sd/j/jungdae/ss-jax-donor-306d53bdc-20260615T074239Z/runs`). They supersede cancelled job
  `54477744` (`--nphi 64 --ntheta 32` mismatch). For the GPU full-node scripts, `scontrol` reported
  `ReqTRES=cpu=32,mem=229902M,node=1,billing=32,gres/gpu=1`; the CPU-JAX fallback requests
  `cpu=32,mem=180G,node=1` and no GPU.
- **Local validation**: `JAX_ENABLE_X64=1 ../simsopt-jax/.miniforge/bin/python3.13 -m pytest tests/integration/test_single_stage_matrix_manifest.py tests/integration/test_fair_compare_launcher_contract.py tests/integration/test_single_stage_init_parity_compile_diagnostics.py tests/integration/test_continuation_donor_backend_contract.py -q` (8+5+6+2 = 21 pass). ruff at `../simsopt-jax/.miniforge/bin/ruff` (note: `ruff format` is NOT enforced repo-wide — only my edited regions are format-clean). Manifest regen: `python benchmarks/perlmutter/build_single_stage_matrix.py --source-sha <sha>`.
- **RunPod**: `runpodctl`. H100 donor pod `ecxt9xwaudcejo` was restarted only to harvest artifacts and is now
  stopped. The corrected `nphi=255`, `ntheta=64` JAX donor advanced to `before_boozer_newton` but was interrupted
  without a clean exit status or donor artifact. Current A100 PCIe donor pod `ibjsq44mxt72lg` was stopped after
  harvesting artifacts.
  It exposed only **117 GB container RAM** (`memoryInGb=117`, cgroup `memory.max=116999999488`), and the JAX
  Newton `run` donor hit cgroup OOM at peak RSS ~113.8 GB after Boozer init. Do not retry the same pod class
  for the JAX donor unless the memory limit changes. GOTCHAS: a restarted pod loses `/usr/bin/time` + apt
  packages (only `/workspace` persists → `apt-get install -y time`); transient MooseFS `OSError: Errno 5`
  write flakes are retryable.

## 5. Done so far (with evidence)
- [x] **Migration + inner-Boozer fix COMMITTED** `0752b18f1` (+ `94f6ea838` source_sha pin): 8 cells
      {scipy-jax(11), scipy-jax-fullgraph(51)} × {cpu,gpu} × {mpol2,mpol10}; ondevice removed; quasi-newton
      inner only; launchers default scipy-jax + pass `--boozer-optimizer-backend`; fullgraph forces
      `--boozer-optimizer-backend ondevice` (fixes the `boozer_surface.py:5659` crash under jax_cpu_parity);
      `tests/integration/test_single_stage_matrix_manifest.py` (8 tests pass).
- [x] **Fair-comparison protocol COMMITTED** `329c92750` (thread caps OMP/OPENBLAS/MKL/NUMEXPR =
      SLURM_CPUS_PER_TASK in both production launchers + protocol in the plan doc) and `9ae9a1f4f`
      (`benchmarks/perlmutter/single_stage_fair_compare_gpu.slurm` same-node cpu+cuda fullgraph mpol10 +
      `tests/integration/test_fair_compare_launcher_contract.py`, 5 tests pass).
- [x] **Compile-diagnostics wiring COMMITTED** `77655d960`: `single_stage_init_parity.py` exposes
      `--record-jax-compile-diagnostics` (store_true default-off), sourced at the single relay call gated
      `backend == "jax"` (relay/resolver already existed). Fair launcher passes it on both lanes by default
      (`FAIR_RECORD_COMPILE_DIAGNOSTICS=1`, `=0` to disable). Full-chain test
      `test_single_stage_init_parity_compile_diagnostics.py` (6 tests, parse_args→child cmd incl. backend
      gate) + flipped the fair-compare "must-not-pass" assertion. **Byte-identical when the flag is absent**
      (`x or (jax and False)` ≡ `x`) → zero regression for existing runs. ruff check clean.
- [x] **fullgraph-51 gate (RunPod b5f97fdf9)**: `free_x` RETIRED; CPU `5659` inner-Boozer crash fixed
      (`--boozer-optimizer-backend ondevice`, proven: stderr empty). Artifacts:
      `.artifacts/clean_reconciliation_benchmarks/gate_fullgraph51_b5f97fdf9_20260614T202101Z/`.
- [x] **RunPod mpol2 smokes DONE** (`.artifacts/clean_reconciliation_benchmarks/mpol2_runpod_20260615T013841Z/`):
      - Cell A scipy-jax (11): ran to completion (153 iters), **the 11-dim lane CONVERGED** (FINAL_IOTA 0.1433
        ≈ 0.15 target); rc=1 = **dim-mismatched parity fail** vs the 51-dim cpp ref (5 iters, iota 0.0036) —
        NOT a code bug. Needs an 11-dim reference or non-parity scoring to be meaningful.
      - Cell B scipy-jax-fullgraph (51): **CORRECTION — advances PAST iter-4** when uninterrupted (iters 1→5:
        J 1.1132→1.1037→1.0964→1.0838→**1.0817**; iter-5 J matches the cpp reference's iter-5). The earlier
        "iter-4 wedge" was a premature SIGKILL of a very slow line search. No `5659`, no `free_x`. Did NOT
        reach passed=true at mpol2 (converges ~4–11 min/iter, slowing) — stopped at the $4 cost guardrail
        (~$2.47), no result JSON for the retry.
- [x] **Existing production runs reconciled** (parallel campaign, SHA `06b7f1a8f`): scipy-jax 11-dim cpu+cuda
      mpol2 exist (`.artifacts/.../runpod_{cpu32,gpu}_prod_sjqn_06b7f1a8f_*/`); both `passed=False` = the same
      dim-mismatch (not crashes). **REUSE for the 11-dim perf story; do not re-run.** The GPU "12× slower"
      number there is an ARTIFACT (throttled CPU slice: cpp ref 44s→2299s + more iters + small-problem
      overhead); per-eval GPU boozer-solve was actually ~2× FASTER (0.17 vs 0.36 s/iter).
- [x] **Stale old-matrix Perlmutter jobs cancelled** (54390918/20, 54401010/17 — old ondevice + LS-axis);
      stale monitor bq28fl125 stopped.
- [x] **Donor 54462557 failed fast** (native_cpu) — root cause identified: inherited `validated-fast`
      trial policy suppressed non-final artifacts; launcher/test fix committed at `5f14a1463`.
- [x] **RunPod mpol10 donor probes FAILED/HARVESTED** (`.artifacts/clean_reconciliation_benchmarks/runpod_mpol10_donor_fail_961df64c3_20260615T0738Z/mpol10_donor_fail_artifacts_20260615T0738Z.tar.gz`):
      - Exact documented native continuation (`--trial-policy none`, default `coarse/medium/prefinal` budgets
        `1/1/2`) failed at stage-01 coarse: no accepted `results.json`, iota ~0.0012.
      - Corrected native budget probe (`--coarse-maxiter 50 --medium-maxiter 50 --prefinal-maxiter 50`) also
        failed at stage-01 coarse: L-BFGS-B `ABNORMAL`, optimizer status `2`, final iota ~0.0035,
        `REJECTED.json`, no accepted progress.
      - JAX runtime-seed donor with Newton `run` preserved iota15 through Boozer init, but was SIGKILLed by the
        pod's 117 GB cgroup limit after target-lane initial value/grad; GPU memory stayed tiny, so host RAM was
        the limiter. JAX `skip` avoided dense Newton but failed initialization (`solve_success=false`).

## 6. Key decisions & rationale (do NOT relitigate)
- **RunPod is allowed only with the right memory envelope** — one GPU pod can build the donor and then run CPU
  and CUDA lanes sequentially on the same host, but the tested A100 PCIe 117 GB container is too small for the
  JAX `run` donor path.
- **Fair comparison = same node + capped threads + replay parity + per-iter throughput** — the only way to
  isolate the device (the old cross-pod numbers were confounded 5 ways).
- **Parity rides on the fullgraph same-candidate replay** (auto for `scipy-jax-fullgraph`,
  `single_stage_init_parity.py:1517/1657`) — convergence-independent, so the slow convergence doesn't block
  the port-correctness result.
- **inner-Boozer fix is fullgraph-only** — the reduced lane uses ondevice inner by default; forcing it would
  flip the harness Newton-polish-policy resolution.
- **Same-pod CPU then GPU for the speed question** (scipy-jax GPU recompile pathology measured via compile
  diagnostics). **ondevice fully dropped.** **Reuse existing scipy-jax perf where applicable; the new work is
  fullgraph-51 + contract-valid mpol10 donor.**

## 7. Dead ends / do NOT retry
- **ondevice 51 production** — 422 GiB XLA compile OOM (architectural).
- **Separate CPU and GPU RunPod pods for the fair speed claim** — confounds host CPU, quota, filesystem, and
  clocks. Use one GPU pod and run CPU then CUDA sequentially instead.
- **(RESOLVED `77655d960`)** `--record-jax-compile-diagnostics` IS now a valid `single_stage_init_parity.py`
  flag (was previously unwired/argparse-error). Do NOT re-add the "un-enableable" caveat anywhere.
- **SIGKILLing fullgraph at "iter-4"** — NOT a wedge; it's a slow line search that breaks through. Let it run.
- **Staging without `git submodule update --init --recursive`** / sharing one checkout across jobs — build fails.
- **Native continuation donor from iota15 Stage 2 seed** — tested on RunPod with exact documented command and
  longer non-final budgets; both fail the stage contract and stay on the near-zero-iota branch.
- **Same A100 PCIe RunPod class for JAX donor** — cgroup RAM is 117 GB and already OOM-killed the dense Newton
  path; use Perlmutter/high-memory or confirm a larger RunPod container RAM limit first.

## 8. Open questions / blockers
- **Will fullgraph-51 mpol10 reach a usable converged state in walltime?** It converges slowly cold at mpol2;
  the donor WARM-START may speed it (better start) — UNVERIFIED. The replay parity is robust regardless;
  end-to-end convergence may need looser inner budgets.
- **Replacement donor success is still UNVERIFIED** — `54462557` itself failed, and native continuation failed
  again on RunPod. The next donor must be JAX/high-memory or a code-level seed-preservation fix.
- **GPU recompile pathology** (scipy-jax GPU) — separate, blocks the GPU column.
- **11-dim parity scoring** — the 11-dim lane converges but can't pass the 51-dim parity gate; needs an
  11-dim reference or non-parity scoring to be a meaningful matrix cell.

## 9. Mental model (hard-won context)
- **Formulation↔backend coupling:** 11=`scipy-jax` (reduced coil-only, surface solved each iter by inner
  Boozer solve), 51=`scipy-jax-fullgraph` (host SciPy over full JF.x; surface re-solved by inner solve).
  ondevice (dropped) compiled the whole outer loop → OOM.
- **Inner-Boozer fix:** `boozer_surface.py:5659` rejects inner `optimizer_backend != ondevice` under
  jax_*_parity; fullgraph defaults inner to `scipy` and the harness only auto-supplies ondevice on cuda
  (`_resolve_target_boozer_optimizer_backend`) — so fullgraph cells must force `--boozer-optimizer-backend ondevice`.
- **The "convergence stall" is SLOWNESS, not a wedge:** fullgraph's outer L-BFGS-B does very long line
  searches near penalty regions (self-intersection / hardware constraints), ~4–11 min/outer-iter and slowing,
  but it DOES progress (iter-5 J matched cpp). Reduced 11-dim re-solves the surface each iter → fast.
- **For this iota15 seed, the viable donor route is JAX/high-memory, not native_cpu.** Native continuation was
  tested with the documented commands and longer non-final budgets and stayed on the near-zero-iota branch.
  JAX runtime-seed preserves the iota15 branch, but the dense Newton/target-lane graph needs enough host RAM.
  A successful donor is still backend-independent geometry → consumable by strict-jax fair-compare cells.
- **GPU "slower" was an artifact**, not real (throttled CPU slice; per-eval GPU compute is fine). Fair perf
  needs the same-node run.

## 10. Pointers
- Commits: `0752b18f1` `94f6ea838` `329c92750` `9ae9a1f4f` `77655d960` (compile-diagnostics wiring),
  `5f14a1463` (donor `--trial-policy none` fix). Launch-state SSOT:
  `HANDOFF-ss-11-51-matrix.md`. Plan/rationale:
  `docs/scipy_jax_11_51_matrix_implementation_plan.md`. Manifest:
  `docs/single_stage_11_51_matrix_2026-06-13.{json,md}`.
- Launchers: `benchmarks/perlmutter/single_stage_production_{cpu,gpu}.slurm` (matrix cells),
  `single_stage_fair_compare_gpu.slurm` (same-node fair compare), `single_stage_continuation_donor.slurm` (donor),
  `submit_single_stage_matrix.py` (refuses mpol10 w/o --warm-start-run-dir), `build_single_stage_matrix.py`.
- Artifacts: gate `.artifacts/clean_reconciliation_benchmarks/gate_fullgraph51_b5f97fdf9_20260614T202101Z/`;
  mpol2 `.../mpol2_runpod_20260615T013841Z/`; existing prod `.../runpod_{cpu32,gpu}_prod_sjqn_06b7f1a8f_*/`.
- Memory (under `/Users/suhjungdae/.claude/projects/-Users-suhjungdae-code-columbia-simsopt/memory/`):
  `project_fullgraph51_gate_outcome.md` (iter-4 correction already applied: slow, not wedged), `project_clean_reconciliation_production_runs.md`,
  `project_scipy_jax_gpu_compile_bound.md`.
- Parallel session handoff (different focus): `HANDOFF.md`.
