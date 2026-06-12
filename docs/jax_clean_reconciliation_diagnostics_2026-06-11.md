# JAX Clean Reconciliation Diagnostics - 2026-06-11

## Scope

This note indexes diagnostic artifacts recovered while reconciling
`pr/jax-port-clean` against the donor `pr/jax-port-pure` worktree. These
artifacts are useful for debugging and run planning, but they are not final
clean-branch signoff evidence unless the source state is explicitly marked as
`pr/jax-port-clean` and source-identical to the final commit under test.

## Copied Diagnostic Bundle

- Clean diagnostic root:
  `/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean/.artifacts/clean_reconciliation_diagnostics/runpod_full_cpu_gpu_20260611`
- Donor source:
  `/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-pure/.artifacts/runpod_full_cpu_gpu_20260611`
- Copy result: 480 files, 121 MB.
- Final-clean-signoff eligible: false.

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

The single-stage CPU benchmark remains non-final. Final clean-source evidence
requires rerunning the default single-stage lane in the pinned JAX/JAXLIB
`0.10.0` benchmark environment.
