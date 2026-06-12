# JAX Clean Reconciliation Diagnostics - 2026-06-11

## Scope

This note indexes diagnostic artifacts recovered while reconciling
`pr/jax-port-clean` against the donor `pr/jax-port-pure` worktree. These
artifacts are useful for debugging and run planning, but they are not final
clean-branch signoff evidence unless the source state is explicitly marked as
`pr/jax-port-clean` and source-identical to the final commit under test.

## Clean Source Archive

- Local archive root:
  `/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean/.artifacts/clean_reconciliation_source/20de74d8b_20260612T001101Z`
- Archive:
  `simsopt-pr-jax-port-clean-20de74d8b.tgz`
- Archive size:
  `55592828` bytes.
- Recorded repo head:
  `20de74d8b5e0563f841bb8f36e242f5675597e63`
- Recorded status:
  `pr/jax-port-clean...upstream_hss/master [ahead 14]`
- Manifest sidecars:
  `repo-head.txt`, `git-status.txt`, `dirty.patch`,
  `dirty-diff-stat.txt`, `source-manifest.txt`, and
  `archive-size-bytes.txt`.
- Dirty patch status:
  `dirty.patch` and `dirty-diff-stat.txt` are empty.

This archive is clean-source staging material for the remaining CPU/GPU reruns.
It is not itself CPU/GPU benchmark evidence.

### Refreshed Source Bundle And Remote Checkout

- Local source root:
  `/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean/.artifacts/clean_reconciliation_source/2f273bf26_20260612T001956Z`
- Archive:
  `simsopt-pr-jax-port-clean-2f273bf26.tgz`
- Archive size:
  `55592844` bytes.
- Git bundle:
  `simsopt-pr-jax-port-clean-2f273bf26.bundle`
- Bundle size:
  `118757095` bytes.
- Recorded repo head:
  `2f273bf26e2574eada705f49547881ff3ab66265`
- Local recorded status:
  `pr/jax-port-clean...upstream_hss/master [ahead 16]`
- Remote checkout:
  `/pscratch/sd/j/jungdae/simsopt-pr-jax-port-clean-2f273bf26-e2e-20260612T001956Z`
- Remote checkout status:
  `pr/jax-port-clean...origin/pr/jax-port-clean`
- Remote stale-CUDA signoff run root:
  `/pscratch/sd/j/jungdae/simsopt-pr-jax-port-clean-2f273bf26-stale-signoff-20260612T002410Z`

Perlmutter accepted two clean-source jobs from this source state:

- Benchmark job `54325846`, job name `banana-e2e-cpu-gpu`, submitted from the
  remote checkout with results root
  `/pscratch/sd/j/jungdae/simsopt-pr-jax-port-clean-2f273bf26-e2e-20260612T001956Z/results`.
- CUDA stale-test signoff job `54325885`, job name `clean-stale-cuda`, submitted
  with run root
  `/pscratch/sd/j/jungdae/simsopt-pr-jax-port-clean-2f273bf26-stale-signoff-20260612T002410Z`.

Scheduler recheck on 2026-06-12T00:56:36Z still showed both GPU/shared jobs as
`PENDING` for priority on `shared_gpu_ss11`:

- `54325846|banana-e2e-cpu-gpu|PENDING|00:00:00|0:0||32|0|1|None assigned`
- `54325885|clean-stale-cuda|PENDING|00:00:00|0:0||32|0|1|None assigned`

Before either job started, the remote checkout was rechecked and had generated
untracked Stage 2 output files:

- `examples/single_stage_optimization/STAGE_2/outputs-wout_nfp22ginsburg_000_014417_iota15.nc/curves_init.vtu`
- `examples/single_stage_optimization/STAGE_2/outputs-wout_nfp22ginsburg_000_014417_iota15.nc/surf_init.vts`

Those files would have failed the signoff scripts' clean-source gate. They were
moved out of the checkout to preserve them under:

`/pscratch/sd/j/jungdae/simsopt-pr-jax-port-clean-2f273bf26-prelaunch-untracked-20260612T010613Z`

After the move, `git status --porcelain=v1 --untracked-files=all` in the remote
checkout was empty and `git status --short --branch` returned
`pr/jax-port-clean...origin/pr/jax-port-clean`.

The original pending jobs were restored to their intended 6 hour limit after a
temporary walltime reduction test. Scheduler recheck on 2026-06-12T01:07:34Z:

- `54325846|banana-e2e-cpu-gpu|PENDING|0:00|(None)|gpu_shared|shared_gpu_ss11`
- `54325885|clean-stale-cuda|PENDING|0:00|(None)|gpu_shared|shared_gpu_ss11`

`scontrol show job` reported both jobs as `PENDING`, `Reason=None`,
`TimeLimit=06:00:00`, QOS `gpu_shared`, and no assigned node.

Alternate batch-QOS checks on 2026-06-12 corrected the scheduler picture:

- `sbatch --test-only -A m4680_g -q debug -t 00:30:00` was accepted, with a
  forecast start on `2026-06-13T08:35`.
- `sbatch --test-only -A m4680_g -q shared -t 04:00:00` was accepted, with a
  forecast start on `2026-06-13T01:31`.
- Batch `-q interactive` was rejected with
  `Cannot submit batch jobs to gpu_interactive_ss11`.
- Batch `-q premium` was rejected with `Invalid qos specification`.
- The earlier `gpu_debug`, `gpu_regular`, and `gpu_shared` QOS-name attempts
  remained rejected by policy.

Two 4-hour shared duplicate jobs were submitted as a possible acceleration path:

- Benchmark duplicate `54327327`, results root
  `/pscratch/sd/j/jungdae/simsopt-pr-jax-port-clean-2f273bf26-e2e-shared4h-20260612T011243Z/results`.
- Stale-signoff duplicate `54327328`, run root
  `/pscratch/sd/j/jungdae/simsopt-pr-jax-port-clean-2f273bf26-stale-signoff-shared4h-20260612T011243Z`.

After submission, both duplicates were normalized to QOS `gpu_shared`, had
`START_TIME=N/A`, and remained pending for priority. They were canceled before
allocation to avoid duplicate GPU consumption. `sacct` reported:

- `54327327|banana-e2e-cpu-gpu|CANCELLED by 114058|00:00:00|0:0|32|0|1|None assigned`
- `54327328|clean-stale-cuda|CANCELLED by 114058|00:00:00|0:0|32|0|1|None assigned`

The original 6-hour jobs `54325846` and `54325885` remain the active queued
routes.

The pending jobs are not final clean-source evidence until Slurm reports
completion and the JSON/log/result artifacts are copied or indexed.

## Final Clean-Source CPU Benchmark Bundle

- Perlmutter job:
  `54326039`, job name `clean-cpu-final`.
- Remote run root:
  `/pscratch/sd/j/jungdae/simsopt-pr-jax-port-clean-2f273bf26-cpu-final-20260612T003430Z`
- Local copied root:
  `/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean/.artifacts/clean_reconciliation_benchmarks/perlmutter_cpu_2f273bf26_54326039`
- Copy result:
  43 files, 3.9 MB, with the remote virtual environment excluded.
- Source commit:
  `2f273bf26e2574eada705f49547881ff3ab66265`
- Source status:
  `pr/jax-port-clean...origin/pr/jax-port-clean`; `git_status_short.txt`,
  `dirty.patch`, and `dirty-diff-stat.txt` are empty.
- Runtime:
  JAX `0.10.0`, JAXLIB `0.10.0`, default backend `cpu`, devices `cpu:0`,
  `JAX_ENABLE_X64=1`, `JAX_PLATFORMS=cpu`,
  `SIMSOPT_BACKEND_MODE=jax_cpu_parity`, `SIMSOPT_BACKEND_STRICT=1`,
  `SIMSOPT_JAX_PLATFORM=cpu`, and
  `SIMSOPT_JAX_TRANSFER_GUARD=disallow`.
- Native extension smoke:
  `simsoptpp_curve_smoke.txt` records `<class 'simsoptpp.Curve'>`.

Slurm accounting for the final CPU job:

- `54326039|clean-cpu-final|COMPLETED|00:15:20|0:0||32|256|1|nid007026`
- `54326039.batch|batch|COMPLETED|00:15:20|0:0|14268364K|256|256|1|nid007026`
- `54326039.extern|extern|COMPLETED|00:15:20|0:0||256|256|1|nid007026`
- `54326039.0|python|COMPLETED|00:07:36|0:0|6559080K|32|32|1|nid007026`
- `54326039.1|python|COMPLETED|00:03:35|0:0|5494412K|32|32|1|nid007026`

Final Stage 2 CPU evidence:

- JSON:
  `artifacts/stage2_cpu.json`
- Result:
  `passed: true`, `failures: []`.
- Provenance:
  clean repo SHA `2f273bf26e2574eada705f49547881ff3ab66265`, JAX/JAXLIB
  `0.10.0`, backend `cpu`, clean git status, `worktree_dirty: false`, and
  `x64_enabled: true`.
- Precision:
  final objective relative difference `0.0`, field-error relative difference
  `0.0`, geometry pointwise differences `0.0`, gradient relative difference
  `0.0`, and 20 iterations in both CPU and JAX lanes.
- Timing sidecar:
  `/usr/bin/time -v` command
  `srun -n 1 -c 32 python benchmarks/stage2_e2e_comparison.py --platform cpu --output-json <run>/artifacts/stage2_cpu.json`;
  walltime `7:35.93`, maximum resident set size `13124` KB, exit status `0`.

Final single-stage CPU evidence:

- JSON:
  `artifacts/single_stage_cpu.json`
- Result:
  `passed: true`, `failures: []`.
- Provenance:
  clean repo SHA `2f273bf26e2574eada705f49547881ff3ab66265`, JAX/JAXLIB
  `0.10.0`, backend `cpu`, clean git status, `worktree_dirty: false`, and
  `x64_enabled: true`.
- Precision:
  final iota absolute difference `0.0`, final volume relative difference
  `1.250125968366028E-15`, field-error relative difference
  `3.637770558581993E-15`, max-curvature relative difference
  `3.714961948211354E-16`, no surface pointwise difference, and no final
  metric parity failures.
- Artifact contract:
  CPU and JAX lanes both accept final artifact JSONs, and both share runtime
  seed spec hash
  `4c4f94d7be6552ee04caf3a366e6e5e209cfe832631c5bda06b9f922b8a67283`.
- Timing sidecar:
  `/usr/bin/time -v` command
  `srun -n 1 -c 32 python benchmarks/single_stage_init_parity.py --platform cpu --stage2-bs-path benchmarks/fixtures/single_stage_seed_iota15/biot_savart_opt.json --case-artifacts-dir <run>/artifacts/single_stage_cpu_cases --output-json <run>/artifacts/single_stage_cpu.json`;
  walltime `3:34.47`, maximum resident set size `10776` KB, exit status `0`.

This CPU bundle is final clean-source CPU benchmark evidence for commit
`2f273bf26e2574eada705f49547881ff3ab66265`. It does not close final GPU
benchmark evidence or the clean CUDA stale-test signoff.

## Copied RunPod Diagnostic Bundle

- Pod:
  `0d2guz9ioc95bb` (`simsopt-a100-full-gpu`)
- Image:
  `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Local copied root:
  `/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean/.artifacts/clean_reconciliation_diagnostics/runpod/0d2guz9ioc95bb`
- Copy result:
  249 files, 13 MB.
- Remote GPU artifact root:
  `/workspace/runpod-a100-full-gpu`
- Remote CPU diagnostic artifact root:
  `/workspace/runpod-a100-cpu32-immediate`
- Recorded RunPod source checkout:
  `/workspace/simsopt-runpod-clean-gpu`, branch `master`, HEAD
  `76c2655a7bb4197897a51e837ed6d325777a016c`.
- Final-clean-signoff eligible:
  false, because the source state does not match current clean branch
  `2f273bf26e2574eada705f49547881ff3ab66265`.

Copied GPU benchmark rc files:

- `stage2_cuda.rc`: `0`
- `single_stage_cuda.rc`: `1`
- `single_stage_cuda_xla_serial.rc`: `1`
- `single_stage_cuda_cuda129.rc`: `0`

Copied CPU diagnostic rc files:

- `stage2_cpu32.rc`: `0`
- `single_stage_cpu32.rc`: `0`

Copied RunPod stale CUDA signoff:

- Run root:
  `runpod-a100-full-gpu/stale_signoff_cuda129`
- Result:
  rc `1`; final-clean-signoff eligible: false.
- Summary:
  `results/summary.json` records 165 requested integration paths, 130 present
  paths, 35 missing paths, 11 current failed/error selectors, 11 new selectors,
  and 0 stale-failure-pattern hits.
- Primary failures:
  one focused abort repro selector missing, 35 integration inventory paths
  missing, `pure-tests-jax` returned `1`, integration batches `003`, `005`,
  `012`, `013`, `018`, `019`, and `020` failed or hard-aborted, and 11
  failed/error selectors remained.

The RunPod pod was stopped after these artifacts were copied:
`runpodctl pod stop 0d2guz9ioc95bb` returned desired status `EXITED`, and a
subsequent `runpodctl pod list` returned an empty list.

## Copied Diagnostic Bundle

- Clean diagnostic root:
  `/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean/.artifacts/clean_reconciliation_diagnostics/runpod_full_cpu_gpu_20260611`
- Donor source:
  `/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-pure/.artifacts/runpod_full_cpu_gpu_20260611`
- Copy result: 480 files, 121 MB.
- Final-clean-signoff eligible: false.

### Perlmutter `sacct` Verification

The copied Perlmutter diagnostic jobs were rechecked on 2026-06-12:

`ssh perlmutter 'sacct -j 54304250,54314828 --format=JobID,JobName,State,Elapsed,MaxRSS,ReqCPUS,AllocCPUS,NNodes,NodeList -P'`

- `54304250|pure-cpu-cpp-jax-clean|COMPLETED|00:03:27||32|256|1|nid007045`
- `54304250.batch|batch|COMPLETED|00:03:27|44888K|256|256|1|nid007045`
- `54304250.0|time|COMPLETED|00:00:07|0|32|32|1|nid007045`
- `54304250.1|time|COMPLETED|00:00:29|0|32|32|1|nid007045`
- `54304250.2|time|COMPLETED|00:00:54|239320K|32|32|1|nid007045`
- `54304250.3|time|COMPLETED|00:01:53|2157680K|32|32|1|nid007045`
- `54314828|dbg-b012-vjp-cache|COMPLETED|00:01:22||32|32|1|nid003925`
- `54314828.batch|batch|COMPLETED|00:01:22|25664K|32|32|1|nid003925`
- `54314828.0|python|COMPLETED|00:00:43|1892208K|32|32|1|nid003925`
- `54314828.1|python|COMPLETED|00:00:35|1606128K|32|32|1|nid003925`

This confirms the diagnostic jobs reached Slurm `COMPLETED` state. It does not
make them final clean-source evidence because their recorded source state is the
dirty `pr/jax-port-pure` staging tree described below.

### Perlmutter CPU Baseline `54304250`

- Local copied root:
  `/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean/.artifacts/clean_reconciliation_diagnostics/runpod_full_cpu_gpu_20260611/perlmutter/cpu_cpp_jax_baseline_54304250`
- Remote artifact root recorded by `summary.json`:
  `/pscratch/sd/j/jungdae/simsopt-pr-jax-port-pure-98f3efe037d6-gpu-signoff-20260611T044104Z/repo/.artifacts/perlmutter_cpu_cpp_jax_baseline_clean_54304250`
- Recorded repo head:
  `0e4d10c1b93aed9e1ee554ac97044d5ca91fbdfb`
- Recorded status:
  `pr/jax-port-pure...origin/pr/jax-port-pure [ahead 2]` with dirty
  `scripts/jax_gpu_failed_stale_tests_signoff.py`,
  `src/simsopt_jax/core/pm_optimization.py`,
  `tests/jax/core/test_pm_optimization_jax_item25.py`, and
  `tests/test_gpu_transfer_guard_harness.py`.
- Useful evidence:
  `summary.json` records exit status 0 for `cpu_device_probe`,
  `run_code_cpp_vs_jax_cpu_parity`, `run_code_cpp_python_cpu`, and
  `run_code_jax_cpu_ondevice`.
- Final-clean-signoff eligible: false, because the source state is a dirty
  `pr/jax-port-pure` staging tree.

### Perlmutter GPU Abort Debug `54314828`

- Local copied root:
  `/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean/.artifacts/clean_reconciliation_diagnostics/runpod_full_cpu_gpu_20260611/perlmutter/gpu_abort_debug_54314828`
- Recorded repo head:
  `0e4d10c1b93aed9e1ee554ac97044d5ca91fbdfb`
- Recorded status:
  `pr/jax-port-pure...origin/pr/jax-port-pure [ahead 2]` with the same dirty
  four-file staging set as the CPU baseline.
- Useful evidence:
  `cache_on_fresh/pytest.log.rc=0` and `cache_off/pytest.log.rc=0`.
- Final-clean-signoff eligible: false, because the source state is a dirty
  `pr/jax-port-pure` staging tree.

## Verified Stage 2 And Single-Stage Seed Inventory

The donor artifact inventory in
`/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-pure/docs/jax_stage2_single_stage_artifacts_2026-06-08.md`
was rechecked from the clean worktree on 2026-06-11.

- Selected Stage 2 and single-stage endpoint files: 30 checked, 0 missing.
- Single-stage seed-root input files: 6 checked, 0 missing.
- Total required files checked: 36.
- Contract distinction preserved:
  `S2-01` and `S2-02` use `surf_opt.json`; `S2-03` through `S2-05` and
  `SS-01` through `SS-05` use `surf_opt_boozer_surface.json`.

These files are valid as seed-load, parser, fixed-state parity, and
deterministic replay inputs. They are not success-performance artifacts and must
not be used as final clean-source CPU/GPU benchmark evidence.

## Clean Benchmark Rerun Requirements

The useful donor benchmark-plan requirements are carried forward as clean
execution rules:

- CPU and GPU benchmark lanes must use the same clean source commit and must
  record `git rev-parse HEAD`, `git status --short --branch`, and any intended
  dirty patch hash or saved patch.
- CPU and GPU lanes must preserve benchmark JSON, command argv, wall time, host
  RSS, hardware identity, Python/JAX/JAXLIB versions, and fixture paths.
- GPU lanes must additionally preserve GPU model, driver/toolchain metadata,
  `nvidia-smi` start/end output, sampled GPU memory, and any XLA/JAX profile
  paths.
- Pytest correctness durations are not CPU/GPU benchmark speedup claims.
- Existing one-shot benchmark entrypoints remain authoritative unless a separate
  split-producer/consumer benchmark API is explicitly implemented and validated.

## Local Clean-Source CPU Benchmark Attempts

All local benchmark attempts below used the clean worktree and recorded source
state, command text, JSON/log sidecars, Python/JAX/JAXLIB versions, and macOS
`/usr/bin/time -l` output under
`.artifacts/clean_reconciliation_benchmarks/`.

### Stage 2 CPU

- Artifact root:
  `/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean/.artifacts/clean_reconciliation_benchmarks/cpu_330925564_x64_20260611T230856Z`
- Source commit:
  `3309255646666a692d2ee4b901de4f2d75ec862a`
- Runtime:
  Python `3.13.12`, JAX `0.9.2`, JAXLIB `0.9.2`, `jax_enable_x64=True`,
  CPU device `TFRT_CPU_0`.
- Command:
  `JAX_ENABLE_X64=1 ../simsopt-jax/.miniforge/bin/python3.13 benchmarks/stage2_e2e_comparison.py --platform cpu --output-json <artifact>/stage2_cpu.json`
- Result:
  rc `0`, `STAGE 2 E2E COMPARISON PASSED`; stdout records final objective,
  field-error, geometry, and matched-gradient relative differences as `0.00e+00`.
- Timing sidecar:
  `132.45 real`, maximum resident set size `6650904576` bytes.

### Single-Stage CPU

- Initial artifact root:
  `/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean/.artifacts/clean_reconciliation_benchmarks/cpu_330925564_single_stage_x64_20260611T231139Z`
- Initial result:
  rc `1`; failed in the CPU/reference surface-vessel penalty path with
  `TypeError: sub got incompatible shapes for broadcasting: (16, 3), (62, 3)`.
- Corrective source delta:
  `src/simsopt_jax_adapters/geo/surface_objectives.py` now flattens
  grid-shaped surface `gamma()` arrays at the pairwise adapter boundary and
  reshapes gradients back before calling each surface VJP. The regression
  `tests/geo/test_surface_objectives_jax.py::test_surface_surface_distance_adapter_flattens_surface_grids_for_pairwise_vjp`
  passed under `JAX_ENABLE_X64=1`.
- Default-lane retry artifact root:
  `/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean/.artifacts/clean_reconciliation_benchmarks/cpu_330925564_single_stage_x64_surfacefix_20260611T235929Z`
- Default-lane retry result:
  rc `1`; the previous surface-vessel shape error was gone. The target JAX lane
  then failed at the private on-device Boozer optimizer gate:
  `On-device optimizer requires JAX >= 0.10.0; found 0.9.2`.
- Fullgraph diagnostic artifact root:
  `/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean/.artifacts/clean_reconciliation_benchmarks/cpu_330925564_single_stage_fullgraph_x64_surfacefix_20260612T000148Z`
- Fullgraph diagnostic result:
  rc `1`; CPU and JAX Boozer initialization completed, then
  `scipy-jax-fullgraph` failed later in the fullgraph outer-optimizer DOF map
  with `AttributeError: 'jaxlib._jax.ArrayImpl' object has no attribute 'free_x'`.

The local single-stage CPU attempts remain non-final. They are superseded for
CPU signoff by Perlmutter job `54326039`, which reran the default single-stage
lane in the pinned JAX/JAXLIB `0.10.0` benchmark environment.
