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

The original 6-hour jobs `54325846` and `54325885` later left the queue and
failed the clean-source gate:

- `54325846|banana-e2e-cpu-gpu|FAILED|00:04:13||32|32|1|nid008229|1:0`
- `54325846.batch|batch|FAILED|00:04:13|18334296K|32|32|1|nid008229|1:0`
- `54325885|clean-stale-cuda|FAILED|00:03:36||32|32|1|nid008229|1:0`
- `54325885.batch|batch|FAILED|00:03:36|18512924K|32|32|1|nid008229|1:0`

The failure logs record the same root cause: run-output files had been written
inside the shared checkout, so the clean-source signoff gate rejected the run.
The locally copied diagnostic bundle is:

`/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean/.artifacts/clean_reconciliation_benchmarks/perlmutter_failed_clean_2f273bf26_54325846_54325885`

These failed jobs are diagnostic-only and are not final clean-source evidence.

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

## Same-SHA Perlmutter CPU Rerun

Same-SHA final CPU evidence was produced after the clean branch advanced to
repo SHA `5572edb9517bcd9c77e79628afb5c45f359e85f4`.

- Job:
  `54331917` (`clean-cpu-final`)
- Local copied root:
  `/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean/.artifacts/clean_reconciliation_benchmarks/perlmutter_5572edb95_54331917`
- Source state:
  clean repo SHA `5572edb9517bcd9c77e79628afb5c45f359e85f4`; `dirty.patch`
  and `git_status_short.txt` are empty.
- Runtime:
  JAX/JAXLIB `0.10.0`, CPU backend, device `cpu:0`, x64 enabled.
- Native extension smoke:
  `simsoptpp_curve_smoke.txt` records `<class 'simsoptpp.Curve'>`.

Slurm accounting:

- `54331917|clean-cpu-final|COMPLETED|00:15:51||32|256|1|nid004332`
- `54331917.0|python|COMPLETED|00:07:39|6634568K|32|32|1|nid004332`
- `54331917.1|python|COMPLETED|00:03:35|5493488K|32|32|1|nid004332`

Stage 2 CPU:

- JSON:
  `artifacts/stage2_cpu.json`
- Result:
  `passed: true`, `failures: []`.
- Precision:
  final objective, value, field-error, and gradient relative differences all
  `0.0`.
- Timing:
  `/usr/bin/time -v` wall `7:38.83`, wrapper MaxRSS `13852` KB, Slurm step
  MaxRSS `6634568K`, exit status `0`.

Single-stage CPU:

- JSON:
  `artifacts/single_stage_cpu.json`
- Result:
  `passed: true`, `failures: []`.
- Precision:
  value relative difference `3.637770558581993E-15`, final iota absolute
  difference `0.0`, final volume relative difference
  `1.250125968366028E-15`, max-curvature relative difference
  `3.714961948211354E-16`, no surface pointwise difference, and no final metric
  parity failures.
- Timing:
  `/usr/bin/time -v` wall `3:34.84`, wrapper MaxRSS `17932` KB, Slurm step
  MaxRSS `5493488K`, exit status `0`.

Perlmutter GPU job `54331914` remains pending as an additional scheduler-backed
GPU route.

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
`runpodctl pod stop 0d2guz9ioc95bb` returned desired status `EXITED`. On
2026-06-12 it was deleted after the local mirror was rechecked, and the final
RunPod cleanup check returned `runpodctl pod list --all` as `[]` with
`currentSpendPerHr: 0`.

## Clean RunPod A100 32-vCPU Rerun

- Pod:
  `3qmh9akb92o9te` (`simsopt-clean-a100-32vcpu-ssh-20260612T033610Z`)
- Image:
  `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Local copied root:
  `/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean/.artifacts/clean_reconciliation_benchmarks/runpod_a100_32vcpu_5572edb95_3qmh9akb92o9te_20260612T033610Z/run`
- Copy result:
  87 files copied after excluding the virtual environment, JAX compilation
  cache, and `__pycache__` directories.
- Source commit:
  `5572edb9517bcd9c77e79628afb5c45f359e85f4`; recorded source status was
  `## HEAD (no branch)` with no tracked or untracked source dirt.
- Hardware:
  `NVIDIA A100-SXM4-80GB`, driver `550.127.05`, `nvidia-smi` CUDA version
  `12.4`, 80 GiB GPU memory, 32-vCPU pod request. The container exposed 256
  logical CPUs; CPU benchmark steps were pinned to CPUs `0-31`, and
  `taskset_cpu_count.txt` records `taskset_cpu_count=32`.
- Runtime contract:
  `runtime_versions_gpu.json` records JAX/JAXLIB/JAX CUDA plugin/PJRT
  `0.10.0`, default backend `gpu`, device `cuda:0`, x64 enabled,
  `LD_LIBRARY_PATH: null`, `SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled`, pip
  `nvidia-cuda-nvcc-cu12==12.9.86`,
  `nvidia-cuda-runtime-cu12==12.9.79`, and
  `nvidia-nvjitlink-cu12==12.9.86`.
- Official-doc runtime checks:
  `artifacts/official_docs_runtime_checks/pre_setup_cuda_tools.txt` records no
  pre-existing `nvcc`, `ptxas`, or `nvlink` on `PATH`. The post-setup check
  records pip `ptxas` CUDA `12.9`, unset `LD_LIBRARY_PATH`, and no host
  `nvlink` on `PATH`, matching the branch's bundled pip-CUDA lane in
  `docs/source/jax_gpu_setup.rst` and `docs/using_jax_backend.md`.

Benchmark exit summary:

- `stage2_cuda`: rc `0`; `stage2_cuda.json` has `passed: true`, empty
  failures, and stdout records final objective, field-error, geometry, and
  matched-gradient relative differences as `0.00e+00`. `/usr/bin/time -v`
  recorded wall `9:15.78` and MaxRSS `6784152` KB.
- `single_stage_cuda`: rc `1`; `single_stage_cuda.json` has
  `passed: false`, status `case-execution-failed`, and records
  `nvlink fatal : Input file ... newer than toolkit (129 vs 124)`. This is
  clean-source failure evidence for this RunPod host, not a green GPU
  single-stage signoff.
- `stage2_cpu32`: rc `0`; `stage2_cpu32.json` has `passed: true`, empty
  failures, and stdout records the same `0.00e+00` parity differences.
  `/usr/bin/time -v` recorded wall `11:58.64` and MaxRSS `6706092` KB.
- `single_stage_cpu32`: rc `0`; `single_stage_cpu32.json` has `passed: true`
  and empty failures. Stdout records `|iota diff|=0.00e+00`,
  volume relative difference `1.39e-16`, field-error relative difference
  `1.63e-15`, and surface relative difference `0.00e+00`. `/usr/bin/time -v`
  recorded wall `5:22.74` and MaxRSS `5293800` KB.

The `single_stage_cuda` failure above is superseded by the fixed CUDA 12.9
local-mode rerun below.

## Fixed Clean RunPod A100 CUDA 12.9 Single-Stage Rerun

- Pod:
  `eyeiml0pmoe135`
- Hardware:
  `NVIDIA A100 80GB PCIe`, driver `565.57.01`, `nvidia-smi` CUDA version
  `12.7`, 80 GiB GPU memory, 32-vCPU affinity `0-31`.
- Local copied root:
  `/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean/.artifacts/clean_reconciliation_benchmarks/runpod_a100_32vcpu_cuda129_single_stage_5572edb95_eyeiml0pmoe135_20260612T050000Z/run`
- Source state:
  clean repo SHA `5572edb9517bcd9c77e79628afb5c45f359e85f4`; `dirty.patch`
  and `git_status_short.txt` are empty.
- Runtime contract:
  JAX/JAXLIB/JAX CUDA plugin/PJRT `0.10.0`, default backend `gpu`, device
  `cuda:0`, x64 enabled, `LD_LIBRARY_PATH` unset,
  `SIMSOPT_JAX_CUDA_LIBRARY_MODE=local`,
  `SIMSOPT_CUDA_TOOLCHAIN_ROOT=/usr/local/cuda`, `XLA_FLAGS` pointing at
  `/usr/local/cuda`, CUDA root `/usr/local/cuda-12.9`, and CUDA 12.9
  `nvcc`/`ptxas`/`nvlink`.

Benchmark result:

- JSON:
  `artifacts/benchmarks/single_stage_cuda129.json`
- Result:
  `passed: true`, `failures: []`.
- Precision:
  value and field-error relative differences
  `1.6307247331574344E-15`, final iota absolute difference `0.0`, final volume
  relative difference `1.3890288537400292E-16`, max-curvature relative
  difference `3.714961948211354E-16`, no surface pointwise difference, and no
  final metric parity failures.
- Timing and memory:
  `/usr/bin/time -v` wall `6:50.00`, MaxRSS `5773804` KB, exit status `0`,
  max GPU memory `883 MiB`.
- Internal performance counters:
  CPU reference elapsed `48.38252680003643` seconds, JAX target elapsed
  `320.1753979101777` seconds, CPU/JAX elapsed ratio
  `0.15111256866028697`, and JAX script total
  `296.27518022060394` seconds.

Follow-up scheduler check:

- `runpodctl` was updated from `2.1.9` to `2.3.0` because the current official
  RunPod CLI docs expose `--min-cuda-version` on `runpodctl pod create`, while
  the older local CLI did not.
- The new CLI still does not expose the 32-vCPU requirement on `pod create`, so
  the official GraphQL create path was used to request 32-vCPU A100 hosts with
  `allowedCudaVersions: ["12.9"]`.
- The first 80 GiB A100 scheduler attempts returned `SUPPLY_CONSTRAINT` for the
  required CUDA 12.9 and 32-vCPU combination. A later `NVIDIA A100 80GB PCIe`
  pod with system CUDA 12.9 became available and produced the fixed
  single-stage CUDA evidence above.
- All RunPod pods used by this reconciliation, including `eyeiml0pmoe135`, were
  deleted after artifacts were copied. Final RunPod checks returned
  `runpodctl pod list --all` as `[]`, account `currentSpendPerHr: 0`, and
  client balance `6.0517982506`.

## Clean Reconciliation Parity Matrix

The current clean-source parity/performance matrix is recorded in both JSON and
Markdown:

- `.artifacts/clean_reconciliation_benchmarks/parity_matrix_5572edb95_20260612T052700Z.json`
- `.artifacts/clean_reconciliation_benchmarks/parity_matrix_5572edb95_20260612T052700Z.md`

Coverage:

- Stage 2 RunPod CPU32 versus JAX/CPU: pass, final objective/value/field-error
  and gradient relative differences all `0.0`, wall `11:58.64`, MaxRSS
  `6706092K`, internal CPU/JAX ratio `1.3658788177575967`.
- Stage 2 RunPod CUDA: pass, final objective/value/field-error and gradient
  relative differences all `0.0`, wall `9:15.78`, MaxRSS `6784152K`, max GPU
  memory `859 MiB`, internal CPU/JAX ratio `1.4309916895395305`.
- Single-stage RunPod CPU32: pass, value and field-error relative differences
  `1.6307247331574344E-15`, final iota absolute difference `0.0`, wall
  `5:22.74`, MaxRSS `5293800K`, internal CPU/JAX ratio
  `0.15100556703887005`.
- Single-stage RunPod CUDA 12.9: pass, value and field-error relative
  differences `1.6307247331574344E-15`, final iota absolute difference `0.0`,
  wall `6:50.00`, MaxRSS `5773804K`, max GPU memory `883 MiB`, internal
  CPU/JAX ratio `0.15111256866028697`.
- Stage 2 Perlmutter CPU job `54331917`: pass, final objective/value/
  field-error and gradient relative differences all `0.0`, wall `7:38.83`,
  Slurm step MaxRSS `6634568K`, internal CPU/JAX ratio
  `1.319938672628129`.
- Single-stage Perlmutter CPU job `54331917`: pass, value and field-error
  relative differences `3.637770558581993E-15`, final iota absolute difference
  `0.0`, wall `3:34.84`, Slurm step MaxRSS `5493488K`, internal CPU/JAX ratio
  `0.11402404300986589`.

CUDA rows use the top-level benchmark runtime/device evidence as the
authoritative GPU runtime signal; nested target result metadata can still carry
CPU labels from the reference/control path.

Performance interpretation:

- Stage 2 CUDA is faster than the RunPod CPU32 lane: wall `9:15.78` versus
  `11:58.64`, and internal JAX target elapsed `122.04172632843256` seconds
  versus `181.46444527059793` seconds.
- Single-stage init CUDA is slower than the RunPod CPU32 lane: wall `6:50.00`
  versus `5:22.74`, and internal JAX target elapsed `320.1753979101777`
  seconds versus `258.07328782975674` seconds.
- The single-stage init benchmark is dominated by setup, JAX compile/prewarm,
  and reporting overhead; it should not be read as a steady-state GPU
  throughput benchmark.
- The final clean-source single-stage rows are init/parity probes with
  `outer_maxiter = 0`, `initial_step_maxiter = 0`, and no recorded
  1500-iteration L-BFGS / 50-Newton production optimization run.

Production single-stage gap:

- Required follow-up: run 1500 outer L-BFGS iterations with
  `--target-lane-boozer-bfgs-maxiter 1500`,
  `--target-lane-boozer-newton-maxiter 50`, and
  `--target-lane-boozer-newton-polish-policy run`.
- Required lanes: cpp/python/cpu reference, JAX CPU target, and JAX GPU target.
- Required metrics: precision parity, final physics metrics, optimizer
  iteration/evaluation counts, walltime, host RSS, GPU memory for CUDA,
  backend/device, compile behavior, and pass/fail status.
- The current parity matrix does not satisfy this requirement because its
  single-stage rows use `outer_maxiter = 0`.

## Production RunPod A100 1500/50 Launch

RunPod pod `vd4ob48umodxpr`
(`simsopt-prod1500-50-a100-20260612T055705Z`) was launched on 2026-06-12
from clean source bundle SHA `5572edb9517bcd9c77e79628afb5c45f359e85f4`.

Launch/runtime facts:

- GPU: `NVIDIA A100-SXM4-80GB`, driver `565.57.01`, memory `81920 MiB`.
- RunPod allocation: 32 vCPU, 251 GB system memory, `$1.49/hr` pod price.
- Account spend check after launch: `$1.532/hr`.
- Auto-termination: `2026-06-12T09:02:01Z`.
- Remote run root:
  `/workspace/runpod_a100_32vcpu_prod1500_50_clean_5572edb95_vd4ob48umodxpr_20260612T060247Z`.
- Local runner artifact:
  `/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean/.artifacts/clean_reconciliation_source/runpod_5572edb95_prod1500_50_20260612T055705Z/runpod_clean_a100_32vcpu_prod1500_50_runner.sh`.

2026-06-12T06:04:08Z live check:

- Remote runner process `131` was active.
- Current setup phase: `apt-get install -y cuda-nvcc-12-9`.
- CPU lane is configured to pin to `0-31`; the host exposes enough CPUs for the
  requested 32-core lane.

2026-06-12T06:08:18Z user-requested stop:

- `runpodctl pod stop vd4ob48umodxpr` set desired status to `EXITED`.
- Follow-up `runpodctl pod list --all` listed pod `vd4ob48umodxpr` as
  `EXITED`.
- Follow-up account check reported `currentSpendPerHr: 0.028`, consistent with
  the stopped pod/volume state rather than an active GPU benchmark.
- The runner had reached Python environment setup before the stop; no final
  production JSONs were copied back.

Pending final artifacts:

- `single_stage_cuda_prod1500_50.json`
- `single_stage_cpu32_prod1500_50.json`
- `production_single_stage_1500_50_summary.json`

This launch does not close the production parity/performance gap until those
artifacts are copied back and verified.

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

## Clean 5572edb95 Submit Checkout Contamination And Rescue

The shared submit checkout
`/pscratch/sd/j/jungdae/simsopt-pr-jax-port-clean-5572edb95-perlmutter-submit-20260612T043227Z/repo`
was rechecked on 2026-06-12 while GPU job `54331914` was still pending. The
completed CPU job `54331917` had written untracked Stage 2 init artifacts into
it:

- `examples/single_stage_optimization/STAGE_2/outputs-wout_nfp22ginsburg_000_014417_iota15.nc/curves_init.vtu`
- `examples/single_stage_optimization/STAGE_2/outputs-wout_nfp22ginsburg_000_014417_iota15.nc/surf_init.vts`
- `examples/single_stage_optimization/STAGE_2/outputs-wout_nfp22ginsburg_000_014417_iota15.nc/R0=0.915-...-backend=jax-cm=penalty/`

Those files would have failed `54331914`'s clean-source gate at job start.
They were moved out of the checkout to
`/pscratch/sd/j/jungdae/simsopt-pr-jax-port-clean-5572edb95-prelaunch-untracked-20260612T060637Z`,
after which `git status --porcelain=v1 --untracked-files=all` was empty and
`54331914` remained pending and gate-safe.

## Stage 2 Probe Output-Root Root Cause And Fix

File mtimes inside the moved-out contamination (21:49:45 through 21:51:10
on 2026-06-11, Pacific) fall inside Slurm step `54331917.0`
(21:43:51 to 21:51:30), which ran `benchmarks/stage2_e2e_comparison.py`, not
the single-stage step. The writer chain is:

- `_run_stage2_probe` in `benchmarks/stage2_e2e_comparison.py` spawned
  `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py` without
  `--output-root`, unlike its sibling `_run_stage2_case`.
- The child defaults `--output-root` to its repo-relative `STAGE_2` script
  directory and creates `outputs-<plasma>/` plus `curves_init`/`surf_init`
  artifacts there.

Fix committed on the clean branch as `090621311`:
`_run_stage2_probe` now threads a per-call tempdir `--output-root`, with the
subprocess regression test
`tests/integration/test_stage2_jax.py::test_stage2_e2e_probe_threads_external_output_root`
pinning that the probe argv carries an `--output-root` outside the repo root.
This closes the contamination class that failed jobs `54325846`/`54325885`.

## Production Wrapper Flag Gap And Fix

The recorded production command shape requires
`--target-lane-boozer-bfgs-maxiter 1500`, and the staged RunPod production
runner passes it, but `benchmarks/single_stage_init_parity.py` had no parser
entry for the flag (only the child-argv threading existed), so the contract
command failed argparse with `unrecognized arguments`. The stopped RunPod
production pod `vd4ob48umodxpr` never reached benchmark execution, so this
would have surfaced only after its full environment setup.

Fix committed on the clean branch as `f17ebc68d`: the wrapper now accepts
`--target-lane-boozer-bfgs-maxiter` (int, default `None` preserves all prior
callers). The full production argv was parse-validated for both `--platform
cpu` and `--platform cuda`. The staged RunPod runner at
`.artifacts/clean_reconciliation_source/runpod_5572edb95_prod1500_50_20260612T055705Z/`
was repointed to expect SHA `c9a09bee5944af3eecd512b5a2e5a533eb6547cd` and
needs the matching bundle
`simsopt-pr-jax-port-clean-c9a09bee5.bundle` if it is relaunched.

## Committed Perlmutter Launchers And Production Submissions

Launcher slice committed as `c9a09bee5944af3eecd512b5a2e5a533eb6547cd`:

- `benchmarks/perlmutter/banana_e2e_cpu_gpu.slurm` now requires
  `RESULTS_ROOT` outside `REPO_ROOT`.
- New `benchmarks/perlmutter/single_stage_production_cpu.slurm` and
  `single_stage_production_gpu.slurm` run the production single-stage matrix
  (1500 outer L-BFGS, 1500 target-lane Boozer BFGS, 50 target-lane Boozer
  Newton, polish-policy `run`) with clean-source gates, the
  `RUN_ROOT`-outside-`REPO_ROOT` guard, pinned JAX/JAXLIB `0.10.0`,
  `/usr/bin/time -v` sidecars, and GPU memory sampling on the CUDA lane.
- New `benchmarks/perlmutter/stale_cuda_signoff.slurm` is the committed
  launcher for `scripts/jax_gpu_failed_stale_tests_signoff.py`.

`sacctmgr show qos` reports MaxWall `2-00:00:00` for both `shared` and
`gpu_shared`, so the earlier 6 hour limits were self-imposed; the production
jobs use 24 hour limits with 10 hour per-case timeouts.

2026-06-12 submissions from submit root
`/pscratch/sd/j/jungdae/simsopt-pr-jax-port-clean-c9a09bee5-prod-20260612T064303Z`,
with one dedicated bundle-cloned checkout per job (all verified at HEAD
`c9a09bee5944af3eecd512b5a2e5a533eb6547cd` with zero tracked or untracked
dirt) and run roots outside the checkouts:

- `54335305` (`prod-ss-cpu`): production single-stage CPU matrix lane
  (cpp/python/cpu reference plus JAX CPU target).
- `54335306` (`prod-ss-gpu`): production single-stage CUDA matrix lane
  (JAX GPU target plus its duplicate CPU reference control).
- `54335307` (`clean-stale-cuda`): clean-source CUDA stale-test signoff,
  replacing failed job `54325885`.

All three were accepted and pending for priority at submission time, with
job stdout and run outputs under `jobs/<lane>/`, outside the checkouts.
Init benchmark job `54331914` (SHA `5572edb95`) remains pending as the
Perlmutter GPU init/parity route and also produces the GPU
`simsoptpp_curve_smoke.txt` artifact. These submissions are not final
evidence until Slurm reports `COMPLETED` and artifacts are copied and
verified.

## Production CPU Job 54335305 Failure And Full-State Lineage Fix

Production CPU matrix job `54335305` started on `nid004332`-class node
`nid004117`, passed all clean-source gates, recorded the
`<class 'simsoptpp.Curve'>` smoke and a valid CPU runtime contract, and then
failed in the JAX target-lane child after 3:17 of benchmark wall time:

- Failing lane: `--backend jax --optimizer-backend ondevice --maxiter 1500`
  with constraint method `penalty`, which is by contract a full-state target
  lane (`single_stage_optimizer_contract_uses_full_state_target_lane`).
- Crash: `single_stage_banana_example.py` outer-optimizer bootstrap
  (`resolve_single_stage_outer_optimizer_initial_dofs`, `JF.x.copy()`) raised
  `AttributeError: 'jaxlib._jax.ArrayImpl' object has no attribute 'free_x'`
  inside native `Optimizable.x` lineage traversal.
- Root cause: `DeferredSurfaceXYZTensorFourier` stored its runtime JAX dof
  array in an instance attribute named `_dofs`, shadowing the `__getattr__`
  delegation to the materialized surface's DOFs object. The native `x` and
  `full_x` setters would likewise have written dead `local_x`/`local_full_x`
  instance attributes, silently freezing surface state.
- Why init evidence never hit this: the default `scipy-jax` backend routes to
  the coil-dofs lane (`bs.x`), and `maxiter = 0` probes do not exercise the
  full-state bootstrap. The earlier local `scipy-jax-fullgraph` failure with
  the same signature was this same defect.

Fix committed as `521fa05f1667fd56629f508d9e2b54a96cf6a31b`: the proxy's
runtime dof attribute is renamed to `_runtime_dofs` so host-only lineage reads
materialize and delegate as designed, and write-through `local_x`/
`local_full_x` properties route native assignments through the materialized
surface while refreshing the runtime dofs. Regression tests
(`tests/integration/test_single_stage_physics_parity.py::TestDeferredSurfaceNativeDofLineage`,
three tests) pin the exact crash line via a real composite round-trip and are
mutation-verified red against the unfixed proxy. An adversarial review pass
returned `PASS` and empirically confirmed free/full dof-space correctness,
materialize-ordering idempotence, and unchanged traced-lane semantics.

Recovery actions:

- Pending GPU production job `54335306` kept its queue position; its
  `checkout_prod_gpu` was swapped in place (old checkout preserved as
  `checkout_prod_gpu_old_c9a09bee5`) and re-verified at
  `521fa05f1667fd56629f508d9e2b54a96cf6a31b` with zero dirt. The launcher
  script content is identical between the two SHAs, so the Slurm-captured
  batch script remains valid.
- Replacement CPU production job `54337350` (`prod-ss-cpu`, 24 hour limit)
  was submitted from new checkout `checkout_prod_cpu2` at the same fixed SHA
  with run root `jobs/prod_cpu2`.
- Failed-job evidence remains under `jobs/prod_cpu/54335305` (gates, runtime
  contract, the structured `passed: false` output JSON, and the traceback).
- The staged RunPod production runner was repointed to expect
  `521fa05f1667fd56629f508d9e2b54a96cf6a31b`.

The production matrix now requires jobs `54337350` (CPU) and `54335306`
(CUDA) to reach `COMPLETED` with passing JSONs from SHA `521fa05f1`.

## Production CPU Job 54337350 OOM And Full-Node Resubmission

Replacement production CPU job `54337350` (SHA
`521fa05f1667fd56629f508d9e2b54a96cf6a31b`, shared QOS, 32 cores) confirmed
the deferred-surface lineage fix: the target lane passed the previously
crashing `JF.x` bootstrap, the scipy reference lane optimized normally within
budget, and the ondevice target lane reached
`Starting target-lane outer optimizer (... boozer_bfgs_maxiter=1500,
boozer_newton_maxiter=50, ... remaining_maxiter=1500)`.

The job then failed with a new, independent cause: Slurm step
`54337350.0` ended `OUT_OF_MEMORY` with MaxRSS `61497548K` (about 58.6 GiB)
against the shared-QOS allocation `ReqMem=60960M` (about 59.5 GiB,
1796M per CPU times 32). The step log records
`Detected 1 oom_kill event in StepId=54337350.0`. The kill landed in the
ondevice target lane shortly after the jitted outer optimizer started, which
matches the known monolithic `jit(run)` compile/warmup memory blowup. This is
an allocation-sizing failure, not a code regression; partial evidence remains
under `jobs/prod_cpu2/54337350`.

Recovery:

- Production CPU matrix job `54341531` (`prod-ss-cpu`) was resubmitted with
  `-q regular --exclusive --mem=0` (full CPU node, about 512 GB) from the
  same clean checkout `checkout_prod_cpu2` at
  `521fa05f1667fd56629f508d9e2b54a96cf6a31b`, run root `jobs/prod_cpu3`.
- Pending GPU production job `54335306` had its host-memory request doubled
  in place via `scontrol update MinMemoryCPU=3592`; `ReqTRES` now records
  `mem=114944M` (about 112 GiB) with the queue position preserved. The GPU
  lane's host-side XLA compile is the same blowup class, so the prior
  57472M ceiling was below the measured 58.6 GiB CPU peak.
- Init benchmark job `54331914` started running on `nid008201` during this
  recovery window.

## Perlmutter GPU Init Evidence Harvested And Tier 5 Seed-Contract Fix

Init benchmark job `54331914` (SHA
`5572edb9517bcd9c77e79628afb5c45f359e85f4`, A100 node `nid008201`) produced
the first final clean-source Perlmutter GPU benchmark evidence:

- `stage2_cpu.json`, `stage2_cuda.json`, `single_stage_cpu.json`, and
  `single_stage_cuda.json` all record `passed: true` with empty failures.
- GPU lanes ran on backend `gpu`; precision: Stage 2 CUDA final objective and
  field-error relative differences `0.0`; single-stage CUDA field-error
  relative difference `3.637770558581993E-15`, final iota absolute difference
  `0.0`, final volume relative difference `1.250125968366028E-15`.
- The GPU-host `simsoptpp_curve_smoke.txt` records
  `<class 'simsoptpp.Curve'>`, closing the GPU native-extension smoke
  checkbox on a Perlmutter CUDA allocation.
- Local copied root:
  `.artifacts/clean_reconciliation_benchmarks/perlmutter_gpu_5572edb95_54331914`
  (85 files, 7.9 MB, venv excluded).

The job's overall Slurm state is `FAILED` only because the auxiliary
`tier5_cuda` performance step exited 1 after 16:20; the four required
benchmark lanes above completed before it. The tier5 failure is a
seed-contract drift, not a GPU fault:

- Tier 5's outer-loop probe uses the default `scipy-jax` backend with
  `maxiter > 0`. The example computes the initial (value, grad) pair
  unconditionally and forwarded it as an optimizer seed because
  `target_lane_contract_supports_optimizer_seed` also claimed the
  SciPy-driven route; `target_minimize()` only consumes
  `initial_value_and_grad` with `method='lbfgs-ondevice'` and raised
  `ValueError`. The defect dates to the branch birth commit and exists in
  the donor branch as well.
- Fix committed as `43513bc52`: the seed contract is narrowed to the
  ondevice route and the redundant raise removed, so seedless lanes simply
  re-evaluate at `x0`. A mutation-verified invariant test pins that every
  seed-supporting contract resolves to `lbfgs-ondevice`. The adversarial
  review pass confirmed no optimization-result change on any lane.

These init rows are at SHA `5572edb95`; production matrix rows continue at
`521fa05f1` via jobs `54341531` (CPU, running on a full `--mem=0` regular
node) and `54335306` (CUDA, pending with `mem=114944M`).
