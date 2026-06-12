# JAX Clean Branch Reconciliation Implementation Plan

## Purpose

This file defines the recovery plan after benchmark and signoff work was started
from `/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-pure` instead of the
intended clean PR worktree at
`/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean`.

The plan exists to keep `pr/jax-port-clean` as the source of truth for code
edits, commits, validation, and final CPU/GPU evidence, while allowing
`pr/jax-port-pure` to remain a reference and donor branch for selectively
recovering validated work.

## Goals

- Establish `simsopt-pr-jax-port-clean` as the only local worktree used for
  implementation, validation, commits, and final signoff.
- Inventory the commits, docs, and dirty patches present in
  `simsopt-pr-jax-port-pure` but missing from `simsopt-pr-jax-port-clean`.
- Port only clean-branch-relevant work from `pure` into `clean`, preserving the
  clean branch's reconstructed PR boundary and avoiding wholesale merges.
- Re-run CPU/GPU benchmark and signoff evidence from the clean branch's intended
  source state before treating results as final.
- Preserve diagnostic evidence from the mistaken `pure`-based runs without
  presenting it as clean-branch signoff.

## Non-Goals

- Do not merge `pr/jax-port-pure` wholesale into `pr/jax-port-clean`.
- Do not use `pure`-based RunPod or Perlmutter results as final evidence for
  `pr/jax-port-clean` unless the source state is proven identical.
- Do not widen the clean PR to include unrelated legacy, C++, native build, or
  maintenance work already excluded by `docs/jax_clean_pr_reconstruction_audit.md`.
- Do not overwrite or revert existing dirty work in `simsopt-pr-jax-port-clean`
  without first classifying it and confirming it belongs to the current slice.

## Current Context

- Current clean worktree:
  `/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean`.
- Current clean branch:
  `pr/jax-port-clean` at `a4b4a583e`, ahead 11 of `upstream_hss/master`.
- Current clean branch has dirty signoff/benchmark-slice files including:
  `docs/jax_gpu_integration_test_paths_2026-06-05.txt`,
  `docs/jax_gpu_integration_batches_2026-06-05/`,
  `scripts/jax_gpu_failed_stale_tests_signoff.py`,
  `docs/jax_clean_branch_reconciliation_commit_classification_2026-06-11.md`,
  `src/simsopt_jax/core/dipole_field.py`,
  `src/simsopt_jax/core/pm_optimization.py`,
  `src/simsopt_jax/core/surface_fourier_kernels.py`,
  `src/simsopt_jax/solve/permanent_magnet.py`,
  `tests/conftest.py`,
  `tests/jax/core/test_dipole_field_item24.py`,
  `tests/jax/core/test_pm_optimization_jax_item25.py`,
  `tests/solve/test_permanent_magnet_optimization_jax_item28.py`,
  and `tests/test_gpu_transfer_guard_harness.py`.
- Current donor worktree:
  `/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-pure`.
- Current donor branch:
  `pr/jax-port-pure` at `98f3efe03`, ahead 33 of `upstream_hss/master`.
- `pr/jax-port-clean` and `pr/jax-port-pure` diverge at
  `fc28d62f8` (`Merge pull request #632 from itpplasma/fix-docker-pr-login`).
- Live comparison found 33 pure-only commits, 10 clean-only commits, and a
  branch diff of 103 files with 6751 insertions and 1339 deletions from clean to
  pure.
- Pure-only docs include
  `docs/perlmutter_cpu_gpu_e2e_benchmark_implementation_plan.md` and
  `docs/jax_stage2_single_stage_artifacts_2026-06-08.md`.
- The existing clean audit,
  `docs/jax_clean_pr_reconstruction_audit.md`, states that the clean PR should
  contain isolated JAX packages, adapter package, JAX tests/docs/examples,
  packaging support, and only legacy-path changes required for that port.
- Clean contains the benchmark entrypoints
  `benchmarks/stage2_e2e_comparison.py` and
  `benchmarks/single_stage_init_parity.py`, and both currently define
  `--platform {auto,cpu,cuda}` plus required `--output-json` arguments.
- The clean local Miniforge environment at
  `../simsopt-jax/.miniforge/bin/python3.13` can import the native extension:
  `from simsoptpp import Curve` returns `<class 'simsoptpp.Curve'>`. Both
  benchmark entrypoints reach argparse under that Python. The system/default
  `python` path is not the benchmark authority for this plan.
- The local Miniforge environment has JAX/JAXLIB `0.9.2`. Stage 2 CPU
  benchmark execution works with `JAX_ENABLE_X64=1`, but default single-stage
  target-lane execution reaches the JAX private optimizer gate and requires the
  pinned JAX/JAXLIB `0.10.0` runtime used by the Perlmutter/RunPod benchmark
  setup.
- The stale CUDA signoff inventory has been refreshed against the clean
  worktree. A default fail-closed dry-run at
  `/tmp/clean-signoff-dry-run-after-final-review-pass/summary.json` requests
  130 integration paths, records 130 present paths and 0 missing paths, reports
  no missing focused selector paths, preserves 8 focused-selector deselectors,
  and records zero current failed selectors, zero new failed selectors, and zero
  stale-failure hits. This is still dry-run evidence only; final CUDA signoff
  must run on a clean-source CUDA host.
- Recent RunPod and Perlmutter jobs launched from the mistaken source snapshot
  are diagnostic only for the clean branch until rerun from clean or proven
  source-identical.
- Pure-only commit and donor-dirty-patch classification is recorded in
  `docs/jax_clean_branch_reconciliation_commit_classification_2026-06-11.md`.
  That note also records that donor commit `d698d26bc` is rejected for this clean
  source state because its dipole axis-basis convention fails the live native
  oracle in both clean and pure.
- Diagnostic artifact indexing and seed-file verification are recorded in
  `docs/jax_clean_reconciliation_diagnostics_2026-06-11.md`.
  The copied `.artifacts/clean_reconciliation_diagnostics/runpod_full_cpu_gpu_20260611`
  bundle is explicitly marked non-final for clean signoff because its recorded
  source state is a dirty `pr/jax-port-pure` staging tree.

## Diagnostic Run Inventory

These runs are useful debugging and benchmark evidence, but they are not final
`pr/jax-port-clean` signoff because their source states do not match the clean
branch source state recorded above.

- Perlmutter source checkout:
  `/pscratch/sd/j/jungdae/simsopt-pr-jax-port-pure-98f3efe037d6-gpu-signoff-20260611T044104Z/repo`.
  Live check on 2026-06-11 showed branch `pr/jax-port-pure`, HEAD `0e4d10c1`,
  ahead 2 of `origin/pr/jax-port-pure`, with tracked dirty files in
  `scripts/jax_gpu_failed_stale_tests_signoff.py`,
  `src/simsopt_jax/core/pm_optimization.py`,
  `tests/jax/core/test_pm_optimization_jax_item25.py`, and
  `tests/test_gpu_transfer_guard_harness.py`.
- Perlmutter CPU baseline job `54304250`
  (`pure-cpu-cpp-jax-clean`) completed on node `nid007045` in `00:03:27`.
  Its `summary.json` records repo head
  `0e4d10c1b93aed9e1ee554ac97044d5ca91fbdfb`, not the clean branch head.
  Phase evidence:
  `cpu_device_probe` exit 0, wall `0:06.74`, MaxRSS `594044K`;
  `run_code_cpp_vs_jax_cpu_parity` exit 0, wall `0:28.96`, MaxRSS `1742620K`;
  `run_code_cpp_python_cpu` exit 0, wall `0:54.07`, MaxRSS `273264K`;
  `run_code_jax_cpu_ondevice` exit 0, wall `1:52.56`, MaxRSS `2389500K`.
- Perlmutter GPU abort-debug job `54314828`
  (`dbg-b012-vjp-cache`) completed on node `nid003925` in `00:01:22`.
  Both targeted cache modes passed:
  `.artifacts/gpu_abort_debug_54314828/cache_on_fresh/pytest.log.rc=0` and
  `.artifacts/gpu_abort_debug_54314828/cache_off/pytest.log.rc=0`.
  The tested selector was
  `tests/integration/test_single_stage_jax_cpu_reference.py::TestAdjointSolveConsistency::test_streaming_group_vjp_matches_full_vjp_without_inv_modB_weighting`.
- An older Perlmutter stale-signoff artifact
  `.artifacts/jax_gpu_failed_stale_tests_signoff_54290378` is diagnostic only:
  `jax-device-probe.log.rc=0`, `transfer-guard-probe.log.rc=0`, and
  `pure_jax_tests.log.rc=1`.
- The Perlmutter artifacts currently mirrored locally live under the donor
  worktree:
  `/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-pure/.artifacts/runpod_full_cpu_gpu_20260611/perlmutter/`.
  If they are retained for this plan, copy or re-index them into a clean
  diagnostic artifact root with source labels instead of treating the donor
  worktree as an authority.
- RunPod active pod as of 2026-06-11T21:38:39Z:
  `0d2guz9ioc95bb` (`simsopt-a100-full-gpu`), image
  `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, desired status
  `RUNNING`, current spend rate `$1.407/hr`.
- RunPod source checkout:
  `/workspace/simsopt-runpod-clean-gpu`, branch `master`, HEAD `76c2655`, with
  only `.artifacts/` untracked. This source state does not match
  `pr/jax-port-clean` HEAD `56d85b14a`.
- RunPod A100 GPU benchmark artifacts under `/workspace/runpod-a100-full-gpu`
  currently record:
  `stage2_cuda.rc=0`, wall `768.6249721050262` seconds, MaxRSS `4808544K`;
  initial `single_stage_cuda.rc=1`;
  `single_stage_cuda_xla_serial.rc=1`;
  repaired `single_stage_cuda_cuda129.rc=0`, wall `439.8951349258423` seconds,
  MaxRSS `6797372K`.
  The CUDA repair installed and selected the CUDA 12.9 toolchain after the
  initial failures hit a CUDA 12.9 object versus CUDA 12.4 `nvlink` mismatch.
- RunPod 32-core CPU artifacts under
  `/workspace/runpod-a100-cpu32-immediate` are A100-host CPU diagnostics, not a
  dedicated CPU-only RunPod pod. Both lanes passed:
  `stage2_cpu32.rc=0`, wall `540.7914683818817` seconds, MaxRSS `4625448K`;
  `single_stage_cpu32.rc=0`, wall `467.3590750694275` seconds, MaxRSS
  `5656752K`.
- RunPod stale CUDA signoff under
  `/workspace/runpod-a100-full-gpu/stale_signoff_cuda129` was still running on
  2026-06-11T21:39:02Z in integration `batch_010`. Earlier in the same run,
  full `tests/jax` reported two failures in
  `tests/jax/core/test_dipole_field_item24.py::test_dipole_field_Bn_on_axis_noncartesian_matches_cpp`
  for the `cylindrical` and `toroidal` parametrizations. Treat this signoff as
  active/red until the final rc and artifacts are copied.

## Rationale

`pr/jax-port-clean` is a reconstructed, reviewable PR branch. It is not a
scratch branch and should not inherit all of `pr/jax-port-pure`'s history or
legacy footprint. The two branches already contain different fixes for related
problem families, so a wholesale merge would likely reintroduce work that the
clean reconstruction intentionally displaced.

The safer path is to treat `pure` as an evidence-backed donor: inspect each
pure-only commit or dirty patch, decide whether it fits the clean PR boundary,
port it as a small clean-branch change, and validate it in the clean source
state. Benchmark results then need to be regenerated from clean because
performance, precision, and memory evidence only sign off the source tree that
actually produced it.

## Assumptions

- `pr/jax-port-clean` remains the intended PR branch for final review and
  submission.
- `pr/jax-port-pure` remains available locally as a donor/reference branch.
- Existing dirty files in `simsopt-pr-jax-port-clean` are user or prior-agent
  work and must be preserved unless explicitly included in the current slice.
- CPU/GPU benchmark comparison should remain platform-specific: CPU evidence
  from a CPU lane or CPU allocation, GPU evidence from a GPU lane or GPU
  allocation, with JSON/time/memory artifacts preserved.
- Perlmutter and RunPod can be used for execution, but final signoff jobs must
  be launched from the clean branch source state.

## Implementation Plan

1. Freeze the source-of-truth boundary.
   - [x] Run `git -C /Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean status --short --branch`.
   - [x] Run `git -C /Users/suhjungdae/code/columbia/simsopt-pr-jax-port-pure status --short --branch`.
   - [x] Record the clean HEAD, pure HEAD, and dirty file lists in a local
         reconciliation note or commit message before making code changes.
   - [x] Treat all future local code/doc edits as clean-worktree edits unless
         the user explicitly redirects to another worktree.

2. Classify pure-only commits against the clean PR boundary.
   - [x] Generate the pure-only list with
         `git -C /Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean log --oneline --reverse --no-merges pr/jax-port-clean..pr/jax-port-pure`.
   - [x] Group pure-only commits into: clean-required, doc-only useful,
         diagnostic-only, superseded-by-clean, and out-of-scope.
   - [x] For each clean-required commit, identify the exact files and behaviors
         to port; do not rely on subject-line similarity.
   - [x] For each superseded commit, cite the clean commit or dirty clean file
         that already covers the behavior.
   - [x] For each out-of-scope commit, confirm it would violate the clean audit
         boundary before leaving it behind.
   - [x] Do not cherry-pick a commit merely because `git cherry -v` marks it
         patch-unique; inspect the hunk-level behavior against clean's existing
         replacement commits and dirty files.

3. Reconcile docs without importing stale authority.
   - [x] Keep `docs/jax_clean_pr_reconstruction_audit.md` as the clean branch
         boundary document.
   - [x] Port or rewrite the useful parts of
         `docs/perlmutter_cpu_gpu_e2e_benchmark_implementation_plan.md` into a
         clean-branch benchmark plan.
        2026-06-11 rewrite: carried forward the source-manifest,
        hardware-provenance, timing, RSS, GPU-memory, and no-pytest-speedup
        rules in `docs/jax_clean_reconciliation_diagnostics_2026-06-11.md`
        without importing the donor split-API implementation plan as clean
        authority.
   - [x] Port or rewrite the useful parts of
         `docs/jax_stage2_single_stage_artifacts_2026-06-08.md` only after
         verifying that each referenced artifact path still exists and that the
         Stage 2 versus single-stage/BoozerSurface file contracts are preserved.
        2026-06-11 verification: checked all 36 selected endpoint and seed-root
        files on disk with 0 missing; recorded the `surf_opt.json` versus
        `surf_opt_boozer_surface.json` contract distinction in
        `docs/jax_clean_reconciliation_diagnostics_2026-06-11.md`.
   - [x] Refresh the stale CUDA signoff inventory before final signoff:
         either remove/rewrite missing selectors and paths in
         `docs/jax_gpu_failed_selectors_2026-06-05.txt`,
         `docs/jax_gpu_integration_test_paths_2026-06-05.txt`, and
         `docs/jax_gpu_integration_batches_2026-06-05/`, or explicitly classify
         them as out-of-scope in this plan.
        2026-06-11 refresh: removed 34 integration paths that are absent from
        both clean and pure, and removed the stale focused repro selector
        `tests/geo/test_lbfgsb_scipy_parity.py::test_jax_setulb_fg_start_reentry_convergence_matches_scipy`,
        whose file is absent from both clean and pure. The default dry-run now
        records zero missing inventory paths and zero missing focused selector
        paths.
   - [x] Use `--missing-path-policy=record` only as an inventory-audit aid; do
         not treat a final signoff as green until the default fail-closed dry
         run succeeds.
        2026-06-11 evidence: the current accepted dry-run used the default
        fail-closed policy and passed with 0 missing paths.
   - [x] Mark all pure-based RunPod/Perlmutter results as diagnostic unless a
         source-state comparison proves they apply to clean.
        2026-06-11 evidence: the diagnostic index marks the copied Perlmutter
        CPU baseline and GPU abort-debug bundles as final-clean-signoff
        ineligible.

4. Preserve and label diagnostic remote evidence.
   - [x] Copy or index the Perlmutter diagnostic artifacts from the donor
         worktree mirror into a clean diagnostic artifact root such as
         `.artifacts/clean_reconciliation_diagnostics/perlmutter/`.
        2026-06-11 evidence: copied 480 files / 121 MB from the donor
        `runpod_full_cpu_gpu_20260611` mirror into
        `.artifacts/clean_reconciliation_diagnostics/runpod_full_cpu_gpu_20260611`.
   - [ ] Copy or index RunPod artifacts from
         `/workspace/runpod-a100-full-gpu` and
         `/workspace/runpod-a100-cpu32-immediate` into
         `.artifacts/clean_reconciliation_diagnostics/runpod/` after the active
         stale signoff finishes or is intentionally stopped.
   - [x] Store each diagnostic bundle with a source-state manifest that records
         remote path, branch, HEAD, dirty status, job or pod id, and whether it
         is final-clean-signoff eligible.
        2026-06-11 evidence: `docs/jax_clean_reconciliation_diagnostics_2026-06-11.md`
        records source state and final-clean eligibility for the copied
        Perlmutter CPU baseline and GPU abort-debug bundles. RunPod workspace
        artifact indexing remains open.
   - [ ] Mark the RunPod A100 pod for shutdown once all requested artifacts are
         copied and no further diagnostic work is needed.

5. Port clean-required code and tests in small slices.
   - [x] Start from current clean dirty files and classify whether each belongs
         to the current signoff/benchmark slice.
   - [x] For overlapping dirty files between clean and pure, compare actual
         patches before copying any hunk:
         `scripts/jax_gpu_failed_stale_tests_signoff.py`,
         `src/simsopt_jax/core/pm_optimization.py`,
         `src/simsopt_jax/core/surface_fourier_kernels.py`,
         `tests/jax/core/test_pm_optimization_jax_item25.py`, and
         `tests/test_gpu_transfer_guard_harness.py`.
   - [x] For pure-only dirty files, decide whether they fit clean:
         `docs/jax_stage2_single_stage_artifacts_2026-06-08.md`,
         `src/simsopt/field/sampling.py`,
         `src/simsopt/field/tracing.py`, and
         `src/simsopt_jax_adapters/geo/surface_objectives.py`.
   - [x] Port one behavior family at a time using patch-level application or
         hand edits in the clean worktree.
        2026-06-11 code/test slice: ported or clean-adapted the strict transfer
        guard signoff lane, PM projection parity, surface Fourier split, and
        dipole SIMD oracle behaviors. Donor dipole commit `d698d26bc` and the
        donor weighted surface split patch were explicitly rejected after live
        oracle/reviewer evidence contradicted them.
   - [x] After each slice, run `git diff --check` and the focused tests that
         exercise the changed behavior.
        Current evidence is recorded below; final CUDA and benchmark reruns are
        still open.

6. Rebuild clean-source benchmark artifacts.
   - [x] Create a clean-source tarball or remote checkout from
         `/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean`, not from
         `simsopt-pr-jax-port-pure`.
        2026-06-12 evidence: created local clean-source archive
        `.artifacts/clean_reconciliation_source/20de74d8b_20260612T001101Z/simsopt-pr-jax-port-clean-20de74d8b.tgz`
        from
        `20de74d8b5e0563f841bb8f36e242f5675597e63`.
        2026-06-12 update: created refreshed local archive and git bundle at
        `.artifacts/clean_reconciliation_source/2f273bf26_20260612T001956Z/`
        from `2f273bf26e2574eada705f49547881ff3ab66265`, then cloned the
        bundle on Perlmutter at
        `/pscratch/sd/j/jungdae/simsopt-pr-jax-port-clean-2f273bf26-e2e-20260612T001956Z`.
   - [x] Include a source manifest beside the tarball or remote checkout with
         `git rev-parse HEAD`, `git status --short --branch`, and a hash or
         saved copy of any intentionally included dirty patch.
        2026-06-12 evidence: the same archive root contains
        `repo-head.txt`, `git-status.txt`, `dirty.patch`,
        `dirty-diff-stat.txt`, `source-manifest.txt`, and
        `archive-size-bytes.txt`; `dirty.patch` and `dirty-diff-stat.txt` are
        empty.
        2026-06-12 update: the refreshed `2f273bf26` source root adds
        `bundle-size-bytes.txt` and `bundle-verify.txt`; the remote checkout
        verified `git rev-parse HEAD` as
        `2f273bf26e2574eada705f49547881ff3ab66265` and
        `git status --short --branch` as
        `pr/jax-port-clean...origin/pr/jax-port-clean`.
   - [ ] Before running Stage 2 or single-stage benchmarks, verify the execution
         environment can import the native extension with
         `python -c "from simsoptpp import Curve; print(Curve)"`.
   - [ ] Submit or run the clean CPU benchmark lane and preserve JSON,
         `/usr/bin/time -v` output, host RSS, CPU count, node/pod identity, and
         exact git source state.
        Partial local evidence: clean-source Stage 2 CPU passed under
        `JAX_ENABLE_X64=1` at
        `.artifacts/clean_reconciliation_benchmarks/cpu_330925564_x64_20260611T230856Z`.
        Default single-stage CPU remains open locally because JAX/JAXLIB
        `0.9.2` cannot run the required private on-device Boozer optimizer.
        Submitted pending remote evidence: Perlmutter job `54325846` was
        accepted from the clean `2f273bf26` checkout with results root
        `/pscratch/sd/j/jungdae/simsopt-pr-jax-port-clean-2f273bf26-e2e-20260612T001956Z/results`;
        as of submission it was `PENDING` on `shared_gpu_ss11` for priority.
   - [ ] Submit or run the clean GPU benchmark lane and preserve JSON,
         `/usr/bin/time -v` output, host RSS, GPU memory samples, GPU model,
         driver/toolchain details, and exact git source state.
        Submitted pending remote evidence: the same Perlmutter job `54325846`
        will run Stage 2 CUDA and single-stage CUDA after the CPU lanes; no
        final JSON exists until the job reaches `COMPLETED` and artifacts are
        copied or indexed.
   - [ ] Compare CPU and GPU JSON outputs only after both runs are from the
         clean source state.
   - [ ] Keep old pure-based artifacts in a diagnostic directory, clearly
         labeled as non-final for clean.

7. Commit only scoped clean-worktree slices.
   - [ ] Use `git diff --cached --name-only`, `git diff --cached --stat`, and
         `git diff --check` before each commit.
   - [ ] Stage only files belonging to the current slice.
   - [ ] Leave unrelated dirty files unstaged.
   - [ ] In commit messages, state whether the slice ports pure work,
         supersedes pure work, or adds clean-only validation.

## Validation Plan

- [x] `git -C /Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean status --short --branch`
- [x] `git -C /Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean diff --check`
- [x] `git -C /Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean log --oneline --reverse --no-merges pr/jax-port-clean..pr/jax-port-pure`
- [x] `git -C /Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean cherry -v pr/jax-port-clean pr/jax-port-pure`
- [x] `python scripts/jax_gpu_failed_stale_tests_signoff.py --dry-run --repo . --python-bin $(command -v python) --results-dir /tmp/clean-signoff-dry-run` succeeds with the default fail-closed missing-path policy.
      2026-06-11 evidence used
      `../simsopt-jax/.miniforge/bin/python3.13` and
      `/tmp/clean-signoff-dry-run-after-final-review-pass`; the summary records
      130 requested/present integration paths, 0 missing paths, 8 focused
      deselectors, 0 current/new/stale failures, and `failures: []`.
- [x] Focused tests for each ported slice, chosen from the files touched by
      that slice.
      2026-06-11 signoff-slice evidence: `ruff check` on touched Python passed;
      `tests/test_gpu_transfer_guard_harness.py -q` passed `17 passed`;
      `tests/jax/core/test_dipole_field_item24.py -q` passed
      `23 passed, 1 warning`;
      `tests/jax/core/test_pm_optimization_jax_item25.py::TestPMKernelHelpers -q`
      passed `15 passed`;
      `tests/solve/test_permanent_magnet_optimization_jax_item28.py -q` passed
      `48 passed, 9 warnings`;
      `tests/geo/test_surface_fourier_jax.py::test_split_flat_to_xyzc_keeps_nan_blocks_isolated -q`
      passed `1 passed`.
      2026-06-12 surface-vessel adapter evidence:
      `JAX_ENABLE_X64=1 ../simsopt-jax/.miniforge/bin/python3.13 -m pytest tests/geo/test_surface_objectives_jax.py::test_surface_surface_distance_adapter_flattens_surface_grids_for_pairwise_vjp -q`
      passed `1 passed, 1 warning`; `ruff check` on the touched adapter/test
      files passed; `git diff --check` passed.
- [x] Crucible reviewer loop reaches strict `PASS` for reviewed code/test
      slices.
      2026-06-11 signoff-slice evidence: six review agents were run and closed. Initial
      findings on the donor weighted surface split and tracked/untracked doc
      wording were fixed; delta reviewers returned `PASS`.
      2026-06-12 surface-vessel adapter delta review: six reviewers returned
      `PASS` after stale validation wording was fixed.
- [ ] `python -c "from simsoptpp import Curve; print(Curve)"` succeeds in the
      clean-source CPU and GPU benchmark environments.
      Local Miniforge CPU evidence passed, but final CPU/GPU benchmark
      environments still need this gate recorded beside their run artifacts.
- [ ] `python scripts/jax_gpu_failed_stale_tests_signoff.py --repo /path/to/clean/remote/checkout --python-bin /path/to/python --results-dir /path/to/results` on a CUDA host after clean-source staging is complete.
      Submitted pending evidence: Perlmutter job `54325885` was accepted from
      the clean `2f273bf26` checkout using run root
      `/pscratch/sd/j/jungdae/simsopt-pr-jax-port-clean-2f273bf26-stale-signoff-20260612T002410Z`;
      as of submission it was `PENDING` on `shared_gpu_ss11` for priority.
- [x] `ssh perlmutter 'sacct -j 54304250,54314828 --format=JobID,JobName,State,Elapsed,MaxRSS,ReqCPUS,AllocCPUS,NNodes,NodeList -P'` still shows the cited Perlmutter jobs as completed when their diagnostic artifacts are copied or referenced.
      2026-06-12 evidence: `54304250` and `54314828` both returned
      `COMPLETED`; node identities were `nid007045` for the CPU baseline and
      `nid003925` for the GPU abort-debug job.
- [ ] RunPod diagnostic artifact copy includes rc files and metrics JSON from
      `/workspace/runpod-a100-full-gpu/benchmarks/`,
      `/workspace/runpod-a100-full-gpu/stale_signoff_cuda129/`, and
      `/workspace/runpod-a100-cpu32-immediate/benchmarks/`.
- [ ] `python benchmarks/stage2_e2e_comparison.py --platform cpu --output-json <cpu-stage2.json>` from a clean CPU source state.
      Local evidence passed with `JAX_ENABLE_X64=1` and
      `.artifacts/clean_reconciliation_benchmarks/cpu_330925564_x64_20260611T230856Z/stage2_cpu.json`.
- [ ] `python benchmarks/stage2_e2e_comparison.py --platform cuda --output-json <gpu-stage2.json>` from a clean GPU source state.
      Submitted pending under Perlmutter job `54325846`; not final until the
      job completes and the JSON is copied or indexed.
- [ ] `python benchmarks/single_stage_init_parity.py --platform cpu --output-json <cpu-single-stage.json>` from a clean CPU source state.
      Local default-lane attempts are blocked after the surface-vessel adapter
      shape fix by JAX/JAXLIB `0.9.2` lacking the required private optimizer
      runtime. Rerun on the pinned JAX/JAXLIB `0.10.0` benchmark environment.
- [ ] `python benchmarks/single_stage_init_parity.py --platform cuda --output-json <gpu-single-stage.json>` from a clean GPU source state.
      Submitted pending under Perlmutter job `54325846`; not final until the
      job completes and the JSON is copied or indexed.
- [ ] CPU/GPU comparison report records exact source commit, dirty status,
      platform, walltime, host RSS, GPU memory when applicable, precision
      deltas, and pass/fail status.

## Risks and Mitigations

- Risk: A wholesale merge from `pure` reintroduces displaced legacy or C++ work.
  Mitigation: Use patch-level porting only, with each hunk checked against
  `docs/jax_clean_pr_reconstruction_audit.md`.
- Risk: Clean and pure contain similar-looking fixes that are not actually
  patch-equivalent.
  Mitigation: Compare behavior and tests, not commit subjects; use
  `git cherry -v`, `git diff`, and focused test evidence.
- Risk: Dirty clean files are overwritten while porting donor work.
  Mitigation: Capture clean dirty status before each slice and stage only the
  intended files/hunks.
- Risk: Pure-based benchmark evidence is accidentally presented as clean
  signoff.
  Mitigation: Label existing pure-based RunPod/Perlmutter artifacts as
  diagnostic and rerun final evidence from a clean source state.
- Risk: The active RunPod A100 pod keeps accruing cost after diagnostic value is
  exhausted.
  Mitigation: Copy the required artifacts promptly and stop/delete the pod once
  no further RunPod diagnostics are needed.
- Risk: The stale CUDA signoff inventory is reused after paths were deleted or
  renamed on the clean branch.
  Mitigation: Make the signoff dry-run pass with the default missing-path policy
  before launching the expensive CUDA signoff.
- Risk: Benchmark commands are submitted from a source-only checkout without a
  usable native extension.
  Mitigation: Gate CPU/GPU benchmark launch on the `simsoptpp.Curve` import
  smoke check in the same Python environment that will run the benchmark.
- Risk: CPU/GPU benchmark comparison mixes different source states or hardware
  classes.
  Mitigation: Store source state, hardware identity, walltime, memory, and
  precision outputs with every run, then compare only like-source results.
- Risk: The clean branch already has replacement commits that make direct
  cherry-picks conflict-heavy or semantically wrong.
  Mitigation: Prefer manual, behavior-level porting for required pieces and
  cite clean replacements for superseded pure commits.

## Completion Criteria

- [x] Every pure-only commit is classified as clean-required, doc-only useful,
      diagnostic-only, superseded-by-clean, or out-of-scope.
- [x] Every clean-required pure behavior is either ported to
      `pr/jax-port-clean` or explicitly rejected with a boundary reason.
- [x] All currently ported code/test slices have focused validation results
      recorded.
- [ ] Diagnostic Perlmutter and RunPod artifacts are copied or indexed under a
      clean reconciliation artifact root with source-state labels and
      final-signoff eligibility marked false unless proven otherwise.
- [ ] Final CPU Stage 2 and single-stage benchmark artifacts are generated from
      the clean source state.
- [ ] Final GPU Stage 2 and single-stage benchmark artifacts are generated from
      the clean source state.
- [ ] CPU/GPU comparison report is based only on clean-source artifacts.
- [ ] All final commits are made from
      `/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean` with unrelated
      dirty work preserved.

## Open Questions

- Which pure-only docs should be ported as clean docs versus archived as
  diagnostic history?
- Resolved: pure's dipole axis-basis convention fix must not be ported directly.
  The donor commit fails the live native oracle in this environment; clean now
  restores the non-finite SIMD convention and the focused parity test passes.
- Which current clean dirty files are user-authored and should remain untouched
  during the reconciliation?
- Should final benchmark runs be performed on Perlmutter only, RunPod only, or
  both, once the clean source state is ready?
- Should the active RunPod A100 pod be stopped immediately after
  `stale_signoff_cuda129` artifacts are copied, or kept running for a clean
  source rerun?
