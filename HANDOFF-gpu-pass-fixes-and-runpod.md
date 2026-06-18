# HANDOFF — Get a PASS for the RunPod single-stage GPU production run (fixes + live run)

> Current update: 2026-06-17 20:08 EDT · Status: pinned RunPod A100 pod `fkq2p28uychssf` produced a formal `passed=true` host-jax artifact with both compile and memory gates passing, and a corrected fixed-state walltime/precision/memory parity artifact now passes across `cpp_cpu`, `jax_cpu`, and `jax_gpu`. Keep using this same A100 pod for follow-up replay/debug work unless the user explicitly approves a replacement.

## 0D. 2026-06-17 corrected fixed-state benchmark scope

This benchmark is not a physics-convergence or hardware-acceptance run. It is a fixed-state walltime, precision-parity, and memory comparison against the C++/CPU/Python baseline.

Compared lanes:

- `cpp_cpu`: classic SIMSOPT/Python path with C++ kernels on the CPU pod.
- `jax_cpu`: JAX adapter path on the CPU pod.
- `jax_gpu`: JAX adapter path on the pinned A100 pod `fkq2p28uychssf`.

Pods and artifact paths:

- CPU pod: `2uc9x3hclul57i`; remote root `/workspace/artifacts/fixed_state_cpp_jax_cpu_20260618T0008Z_r3`.
- A100 pod: `fkq2p28uychssf`; remote root `/workspace/artifacts/fixed_state_cpp_jax_gpu_a100_20260618T0008Z_r3`.
- Merged remote artifact: `/workspace/artifacts/fixed_state_cpp_jax_gpu_a100_20260618T0008Z_r3/fixed_state_merged_cpu_cuda_r3.json`.
- Local copied artifacts: `.artifacts/runpod_fixed_state_parity_20260618/`.

Merged artifact result:

- `passed: true`
- `failures: []`
- `cpp_cpu_vs_jax_cpu`: pass; objective relative delta `6.32096841608427e-15`; gradient max relative delta `1.1150303692562862e-12`; `grad_allclose: true`.
- `cpp_cpu_vs_jax_gpu`: pass; objective relative delta `3.6134869445281743e-13`; gradient max relative delta `1.0273812158520607e-12`; `grad_allclose: true`.
- `jax_cpu_vs_jax_gpu`: pass; objective relative delta `3.550277260367354e-13`; gradient max relative delta `3.847601218469286e-13`; `grad_allclose: true`.

Walltime and memory:

- CPU slice process walltime: `4:16.41`; max RSS `2860960 KB`.
- A100 slice process walltime: `2:54.36`; max RSS `2880012 KB`; sampled peak GPU memory `2909 MiB`.
- Per-lane measured runtime inside the merged artifact:
  - `cpp_cpu`: `1.7702539674937725 s`
  - `jax_cpu`: `11.758133858442307 s`
  - `jax_gpu`: `51.61526986584067 s`

Harness fixes made for the corrected benchmark:

- `benchmarks/single_stage_cpp_jax_state_parity.py` now imports `SurfaceSurfaceDistance` from `simsopt_jax_adapters.geo.surface_objectives`, matching the single-stage example path.
- Classic CPU `BoozerSurface` no longer crashes the fixed-state artifact by calling the JAX-only `get_adjoint_runtime_state()` API; the harness now reports a dense transpose least-squares adjoint diagnostic from the classic residual linear operator.
- Real reduced fixed-state artifacts now use the source CPU fixture hashes for derived JAX lanes, so identity hashes describe the benchmark problem rather than lane-internal reconstructed fixture details.

Invalid/obsolete evidence boundary:

- The earlier CPU full-init seed run that failed after a long Boozer solve is not parity evidence and should not be used for this benchmark.
- The six-seed A100 fixed-budget run remains useful A100 walltime telemetry, but by itself it is not the CPU-vs-JAX-vs-GPU parity matrix because the clean CPU slice is the fixed-state artifact above.

## 0C. 2026-06-17 pinned A100 RunPod CUDA 12.9 replay

Hard pin:

- Use this pod for the A100 replay unless the user explicitly approves a replacement: `fkq2p28uychssf`.
- Do not silently switch to a new RunPod pod for this campaign.
- Pod name: `simsopt-host-jax-gate-a100-mincuda129-20260617T151923Z`
- SSH: `ssh -i /Users/suhjungdae/.runpod/ssh/RunPod-Key-Go root@154.54.102.40 -p 11429`
- Cost: `$1.49/hr`
- Status at record time: `desiredStatus=RUNNING`

RunPod pod details:

- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- GPU: `NVIDIA A100-SXM4-80GB`
- Driver: `580.126.16`
- `nvidia-smi` CUDA report: `13.0`
- GPU memory: `81920 MiB`
- `runpodctl` CPU/memory metadata: `vcpuCount=16`, `memoryInGb=250`
- Remote `nproc`: `128`
- Volume: `/workspace`, `100 GB`

Official-doc-compatible JAX/CUDA target:

- JAX/JAXLIB: `0.10.0` with CUDA 12 pip packages.
- JAX CUDA 12 wheel expectation: CUDA >= 12.1 and cuDNN >= 9.8,<10; unset `LD_LIBRARY_PATH` unless deliberately overriding CUDA libraries.
- NVIDIA CUDA 12.x minor-compatibility table: minimum driver `>=525`, upper minor-compat range `<580`, with newer drivers still supported through binary backward compatibility. This pod's driver is newer (`580.126.16`) and exposes CUDA 13.0 compatibility while the local compiler/linker path is pinned to CUDA 12.9.
- Official docs checked: JAX CUDA install (`https://docs.jax.dev/en/latest/installation.html`), JAX GPU memory allocation (`https://docs.jax.dev/en/latest/gpu_memory_allocation.html`), NVIDIA CUDA compatibility (`https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html`).
- Installed CUDA 12.9 compiler/linker tools from NVIDIA apt:
  - `/usr/local/cuda-12.9/bin/ptxas`: `12.9.86`
  - `/usr/local/cuda-12.9/bin/nvlink`: `12.9.86`
- `/usr/local/cuda` alternatives now point at the CUDA 12.9 install after repair.
- This fixes the old incompatible-pod failure: `nvlink fatal: Input file ... newer than toolkit (129 vs 128)`.

Remote paths:

- Checkout: `/workspace/simsopt-host-jax-a100-cuda129-r2-20260617`
- Venv: `/tmp/simsopt-a100-cuda129-venv`
- Compilation cache: `/workspace/jax_compilation_cache_host_jax_a100_cuda129`
- Current passing gate artifact: `/workspace/artifacts/host_jax_gate_a100_cuda129_r4`
- Local copy: `.artifacts/runpod_a100_toolchain_fix_20260617/host_jax_gate_a100_cuda129_r4`
- Seed spec used by the passing run: `/workspace/artifacts/host_jax_gate_a100_cuda129_specs/single_stage_jax_runtime_spec_m2_n31.json`
- Local seed-spec copy: `.artifacts/runpod_a100_toolchain_fix_20260617/host_jax_gate_a100_cuda129_specs/single_stage_jax_runtime_spec_m2_n31.json`

Preflight evidence:

```json
{"backend":"gpu","devices":["cuda:0"],"jax":"0.10.0","jaxlib":"0.10.0","result":8575959040.0,"x64":true}
```

Editable install/import evidence:

```json
{"backend":"gpu","devices":["cuda:0"],"jax":"0.10.0","jaxlib":"0.10.0","simsopt":"1.9.4.dev0","x64":true}
```

Remote build notes:

- System package installed: `libboost-dev 1.83.0.1ubuntu2`.
- Local `thirdparty/` contents were copied into the remote bundle checkout because the git bundle did not include submodule git repos.
- Editable install succeeded with `--config-settings=cmake.define.GIT_SUBMODULE=OFF` and `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SIMSOPT=1.9.4.dev0`.

Formal A100 gate result:

- Artifact: `/workspace/artifacts/host_jax_gate_a100_cuda129_r4/host_jax_outer_loop_gate.json`
- Local copy: `.artifacts/runpod_a100_toolchain_fix_20260617/host_jax_gate_a100_cuda129_r4/host_jax_outer_loop_gate.json`
- Probe log: `/workspace/artifacts/host_jax_gate_a100_cuda129_r4/probe.log`
- Progress trace: `/workspace/artifacts/host_jax_gate_a100_cuda129_r4/single_stage_case_outputs/mpol=2-ntor=2-a6a00b85/outer_optimizer_progress.json`
- `passed: true`, `failures: []`, shell `exit_code: 0`
- Probe log ended with `SINGLE-STAGE OUTER-LOOP PROBE PASSED`.
- Walltime: `217 s` by `start_epoch.txt`/`end_epoch.txt`.
- Objective decreased from `2.7510524088510415` to `0.4396022828389794` over `10` optimizer iterations.
- Compile gate passed:
  - `steady_compile_event_count_growth: 0`
  - `steady_cache_miss_count_growth: 0`
  - steady counters stayed at `compile_event_count: 3`, `cache_miss_count: 2`
  - `steady_snapshot_count: 15`, `warmup_evaluations: 1`
- Memory gate passed:
  - `peak_steady_gpu_memory_growth_mb: 0.0`
  - `peak_steady_rss_growth_mb: 0.0`
  - `peak_steady_gpu_memory_mb: 1187.0`
  - `peak_steady_rss_mb: 2832.69140625`
  - `steady_snapshot_count: 15`, `warmup_evaluations: 1`
- Internal timings:
  - `jax_elapsed_s: 213.6457782704383`
  - `jax_outer_optimizer_main_s: 96.89357492886484`
  - `jax_boozer_total_s: 24.07330364175141`

Setup/failure history on this same pinned pod:

- First CUDA 12.9 replay attempt `host_jax_gate_a100_cuda129_r1` failed before solver/provenance output, not because of CUDA:
  - `git status --short --untracked-files=no` returned 128 inside `benchmarks/validation_ladder_common.py`.
  - Direct stderr: `fatal: not a git repository: thirdparty/eigen/../../../simsopt/.git/worktrees/simsopt-pr-jax-port-clean/modules/thirdparty/eigen`.
- Root cause: copied submodule `.git` pointer files from the local checkout reference local worktree metadata that does not exist inside the remote bundle clone.
- Fix applied: the copied `thirdparty/*/.git` pointer files were renamed to `.git.copied-from-local-disabled`; thirdparty source contents were kept intact. Plain `git status --short --untracked-files=no` then exited 0.
- `host_jax_gate_a100_cuda129_r2` failed before the solver because the remote venv was missing `optax`.
- Fix applied: installed the repo `JAX_GPU` optimizer dependency set into the same venv: `optax 0.2.8`, `optimistix 0.1.0`, `lineax 0.1.1`, `equinox 0.13.8`. JAX/JAXLIB stayed at `0.10.0`.
- `host_jax_gate_a100_cuda129_r3` failed before the solver because the immutable JAX runtime seed spec was missing.
- Fix applied: generated `/workspace/artifacts/host_jax_gate_a100_cuda129_specs/single_stage_jax_runtime_spec_m2_n31.json` from the existing fixture seed spec and reran as `host_jax_gate_a100_cuda129_r4`.

Old incompatible A100 artifacts:

- Old pod `sv9vtl76gclwoq` was deleted after r1-r4 failed on the driver/toolchain mismatch.
- Its artifacts were copied locally under `.artifacts/runpod_a100_toolchain_fix_20260617/host_jax_gate_a100_r1..r4` before deletion.

## 0B. 2026-06-17 RunPod H100 host-jax PASS

RunPod pod used:

- Pod id: `5svwqwgw4x0e8d`
- GPU: H100 80GB HBM3
- Driver: `580.126.09`
- JAX/JAXLIB: `0.10.0`
- CUDA runtime reported by JAX: `13.0`
- Remote checkout: `/workspace/simsopt-host-jax-r3-20260617`
- Remote venv: `/tmp/simsopt-host-jax-venv-r3`

Formal command:

```bash
cd /workspace/simsopt-host-jax-r3-20260617
mkdir -p /workspace/artifacts/host_jax_gate_r13 /workspace/jax_compilation_cache_host_jax
unset LD_LIBRARY_PATH
JAX_ENABLE_X64=1 \
JAX_PLATFORMS=cuda \
JAX_COMPILATION_CACHE_DIR=/workspace/jax_compilation_cache_host_jax \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_FLAGS="--xla_gpu_exclude_nondeterministic_ops=true" \
timeout 3600s /tmp/simsopt-host-jax-venv-r3/bin/python benchmarks/single_stage_outer_loop_probe.py \
  --platform cuda \
  --optimizer-backend host-jax \
  --boozer-least-squares-algorithm lm \
  --enable-compile-diagnostics \
  --enable-host-jax-memory-gate \
  --record-objective-evaluation-trace \
  --jax-runtime-seed-spec /workspace/artifacts/host_jax_gate_r3_specs/single_stage_jax_runtime_spec_m2_n31.json \
  --output-json /workspace/artifacts/host_jax_gate_r13/host_jax_outer_loop_gate.json \
  > /workspace/artifacts/host_jax_gate_r13/probe.log 2>&1
```

Artifacts:

- `/workspace/artifacts/host_jax_gate_r13/host_jax_outer_loop_gate.json`
- `/workspace/artifacts/host_jax_gate_r13/probe.log`
- `/workspace/artifacts/host_jax_gate_r13/single_stage_case_outputs/mpol=2-ntor=2-8ff76e08/outer_optimizer_progress.json`

Gate result:

- `passed: true`, `failures: []`
- Probe log ended with `SINGLE-STAGE OUTER-LOOP PROBE PASSED`.
- Objective decreased from `2.7510524088510406` to `0.4396022828443153` over `10` optimizer iterations.
- Compile gate passed: `steady_compile_event_count_growth: 0`, `steady_cache_miss_count_growth: 0`; final steady counters stayed at `compile_event_count: 3`, `cache_miss_count: 2`.
- Memory gate passed: `peak_steady_gpu_memory_growth_mb: 0.0`, `peak_steady_rss_growth_mb: 0.0`; steady peak GPU memory was `1609.0 MB`, steady peak RSS was `2316.1796875 MB`.
- Internal timings: `jax_elapsed_s: 336.67542380839586`, `jax_outer_optimizer_main_s: 73.063839411363`, `jax_boozer_total_s: 51.02851968631148`.
- Shell/file timestamps: artifacts created/finalized between `2026-06-17 11:34:47 +0000` and `2026-06-17 11:42:32 +0000` (about 7m45s wall by artifact timestamps).

Scope note:

- This supersedes the older failed RunPod compile-growth trace for the H100 lane.
- Historical note: this H100 pass was the first formal host-jax GPU pass. The pinned A100 `host_jax_gate_a100_cuda129_r4` pass above is now the A100-specific replay evidence.

## 0A. 2026-06-17 RunPod host-jax execution update

RunPod pod used:

- Pod id: `8xn0sv93ceehgq`
- GPU: A100-SXM4-80GB
- Driver: `580.126.16`
- JAX/JAXLIB: `0.10.0`
- Remote checkout: `/workspace/simsopt-host-jax-r2`
- Cleanup: pod deleted successfully; `runpodctl pod list` returned `[]`.

Official-doc constraints applied:

- JAX `jit` specializes on static args/function identity, so the kernel bundle must be cached by immutable shape/static config and dynamic coil/group state must be array args.
- JAX GPU memory defaults to large preallocation; RunPod runs used `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- JAX persistent cache must be configured before first compile. `/workspace` cache caused kernel-wait pathologies on RunPod; `/tmp/jax_cache/...` was safer for smoke probes.
- JAX CUDA install/driver docs require CUDA 13 wheels on driver >=580 or CUDA 12 wheels on driver >=525. This pod's `580.126.16` driver is valid for current JAX CUDA wheels.

Code/runtime fixes made after the requirements review:

- Fixed child-command propagation so `--optimizer-backend host-jax` no longer launches the inner child as `--boozer-optimizer-backend ondevice`.
- Fixed host-jax LM JVP through Boozer residual splitting by using the raw static `lax` split instead of the custom-VJP split.
- Added block-JVP Jacobian materialization (`jacobian_block`) so host-jax LM no longer requires a monolithic dense `jacfwd` state.
- Reprojected immutable runtime seed specs before low-res probes; existing `benchmarks/fixtures/single_stage_seed_iota15/single_stage_jax_runtime_spec.json` is shape-bound and correctly rejects mismatched `nphi/ntheta/mpol/ntor`.
- Made first failure evaluations finite when `initial_objective_pending=True`; the previous path produced `nan` failure objectives because it added a finite penalty to `run_dict["J"] == nan`.
- Made the outer-loop probe use a durable child output root so `outer_optimizer_progress.json` survives long enough for compile/memory gates.
- Made `--disable-target-lane-success-filter` also disable the host-jax hard hardware reject, so smoke probes can test compile/memory behavior without being stopped by the m2 seed curvature reject.
- Fixed host-jax kernel-bundle ownership across Boozer coil refreshes. `BoozerSurfaceJAX._refresh_coil_data()` still clears CPU/reference closures on every refresh because they capture coil values, but it only clears `_kernel_bundle_cache` when the grouped-coil static signature changes (group order, array shapes/dtypes, and coil indices). Dynamic coil values remain explicit JAX arguments, so same-shape outer optimizer steps no longer rebuild the jitted bundle.
- Added a `run_code()`-level regression that runs two host-jax solves through `host_jax_minimize_value_and_grad` with changed coil values but unchanged grouped-coil static signature, and asserts the same kernel bundle plus one `value_and_grad` executable is reused.

RunPod artifacts/evidence:

- `host_jax_lm_block_m4_spec_20260617T071349Z/single_stage_lowres_jax_runtime_spec.json`
  - Reprojected spec for `nphi=255, ntheta=64, mpol=4, ntor=4`.
  - Shape verified: surface `(4, 4)`, quadrature `(255, 64)`, Boozer seed iota `0.22650585872006085`.
- `host_jax_lm_block_m4_gate_20260617T071418Z`
  - m4 host-jax LM/block-JVP probe did not reach objective-evaluation artifacts before manual stop.
  - It populated JAX cache entries but stayed in startup/compile/host setup with GPU compute at 0%.
- `host_jax_lm_block_smoke_durable_trace_20260617T074024Z/probe.json`
  - Smoke `nphi=31, ntheta=16, mpol=2, ntor=2`.
  - Host-jax Boozer LM solved: `iter=45`, Newton polish `iter=2`, iota `0.04106909337867379`.
  - Probe failed cleanly, not by crash: `iterations=0`, `initial_objective=2.75`, `final_objective=2.75`, one objective-evaluation event. Gates could not evaluate steady state because there were no post-warm snapshots.
- `host_jax_lm_block_smoke_filter_disabled_20260617T074426Z/single_stage_case_outputs/.../outer_optimizer_progress.json`
  - Filter-disabled smoke generated 7 objective-evaluation events before timeout.
  - First objective finite: `2.7510524088510433`.
  - Later objective events were `nan`.
  - Compile counters grew every objective event: `compile_event_count` `13 -> 97`, `cache_miss_count` `28 -> 160`.
  - This directly fails the compile gate: host-jax is not yet "one compile per shape".

Older A100 blocker, now superseded for H100 only:

- The prior RunPod trace is still failed evidence: `compile_event_count` grew `13 -> 97` and `cache_miss_count` grew `28 -> 160`. The local root cause is fixed, but promotion needs a new RunPod A100/H100 gate proving `host_jax_compile_gate.passed == true` and `host_jax_memory_gate.passed == true`.
- GPU memory gate cannot be promoted from the old trace because the steady-state compile gate failed first, and `gpu_memory_mb` snapshots in that RunPod trace were unavailable (`None`) even though `nvidia-smi` showed context memory.
- Full m10 and m4 were too slow/compile-bound on that A100 pod. Those traces remain useful for debugging only, not formal H100 or A100 promotion evidence.

Validated locally after the cache-owner fix:

- `/opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest -q tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_host_jax_run_code_reuses_kernel_bundle_after_coil_value_refresh`: 1 passed.
- `/opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest -q tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_host_jax_run_code_reuses_kernel_bundle_after_coil_value_refresh tests/geo/test_boozersurface_jax.py::TestUpstreamFactoryBoozerMatrix::test_host_jax_kernel_bundle_survives_same_signature_coil_refresh tests/geo/test_boozersurface_jax.py::TestUpstreamFactoryBoozerMatrix::test_host_jax_kernel_bundle_rebuilds_on_coil_static_signature_change tests/geo/test_boozersurface_jax.py::TestUpstreamFactoryBoozerMatrix::test_host_jax_kernel_bundle_compiles_once_per_static_signature tests/geo/test_boozersurface_jax.py::TestUpstreamFactoryBoozerMatrix::test_host_jax_kernel_bundle_linear_solve_compiles_once_per_static_signature`: 5 passed.
- `/opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest -q tests/geo/test_optimizer_jax_item19.py::test_item19_host_jax_least_squares_uses_jacobian_blocks`: 1 passed.
- `git diff --check -- src/simsopt_jax_adapters/geo/boozer_surface.py tests/geo/test_boozersurface_jax.py`: pass.
- The heavyweight upstream same-shape cache node printed `1 passed`, then the local process exited `137` during teardown/resource cleanup, so it is not counted as a clean local validation result.

## 0. 2026-06-17 requirements review update

The correct fix is now partially implemented in the working tree:

- Added a `host-jax` Boozer inner backend that keeps LS/minimize/Newton/BFGS/line-search/convergence control on the host while calling cached JAX kernels for objective/residual/Jacobian/value-gradient work.
- Added `BoozerKernelBundle` caching keyed by the static penalty signature and extended it with `linear_solve` and `factor_apply`, so the expensive array math can remain compiled without tracing the optimizer loop.
- Added `make_traceable_solved_state_value_and_grad(...)`, a host-solved bridge with signature `(coil_dofs, solved_x, solved_linear_solve_factors) -> (value, grad)`. This intentionally does not build the fused traceable forward-solve graph.
- Added `HostJaxSingleStageAdapter` for the production penalty outer loop. Each SciPy L-BFGS-B objective evaluation now applies coil DOFs on the host, runs Boozer from the accepted warm start on the host, obtains solved state plus linear solve factors, and calls `make_traceable_solved_state_value_and_grad(...)` for the implicit-gradient/adjoint result. The accepted-step callback commits explicit solved-state/value/gradient data into `run_dict` without calling `JF.J()` or `JF.dJ()`.
- `--optimizer-backend host-jax` skips legacy initial `JF.dJ()` snapshot evaluation; the first host-jax objective evaluation seeds `run_dict["J"]`/`run_dict["dJ"]` from the solved-state kernel.
- `benchmarks/single_stage_outer_loop_probe.py` now accepts `--optimizer-backend host-jax`, records host-jax compile diagnostics, enforces a `host_jax_compile_gate` when compile diagnostics are enabled, and can enforce `--enable-host-jax-memory-gate` by reading per-objective host RSS/GPU memory snapshots from `outer_optimizer_progress.json`.
- Added tests for same-shape compile reuse, new-shape compile behavior, linear solve/factor apply caching, host-jax routing, and the solved-state value/gradient kernel boundary.

What is still not complete:

- A formal RunPod H100 artifact now proves the host-jax compile/memory gates on H100.
- A formal A100 replay is still open if A100-only evidence is required for the production campaign.

Official docs checked for this update:

- JAX `jit`: static arguments and the jitted function identity participate in the compilation cache key, so static problem signatures must be immutable and dynamic values must travel as array arguments. <https://docs.jax.dev/en/latest/_autosummary/jax.jit.html>
- JAX GPU memory allocation: JAX preallocates 75% of GPU memory by default; `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION`, and `XLA_PYTHON_CLIENT_ALLOCATOR=platform` are the documented memory controls. <https://docs.jax.dev/en/latest/gpu_memory_allocation.html>
- JAX persistent compilation cache: use `JAX_COMPILATION_CACHE_DIR`/`jax_compilation_cache_dir` before first compile if reusing compiled programs across repeated RunPod runs. <https://docs.jax.dev/en/latest/persistent_compilation_cache.html>
- JAX install docs and NVIDIA CUDA compatibility: CUDA 12 requires driver >=525; CUDA 13 requires driver >=580. <https://docs.jax.dev/en/latest/installation.html> and <https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html>
- Ubuntu NVIDIA driver docs: on Ubuntu servers, use `ubuntu-drivers list --gpgpu` / `ubuntu-drivers install --gpgpu` if managing the host image yourself; RunPod generally controls this layer. <https://ubuntu.com/server/docs/how-to/graphics/install-nvidia-drivers/>

Validated locally after the host-jax patch:

- `../simsopt-jax/.miniforge/bin/python3.13 -m ruff check ...` over all touched Python files: pass.
- `../simsopt-jax/.miniforge/bin/python3.13 -m compileall -q ...` over all touched Python files: pass.
- `git diff --check`: pass.
- `pytest tests/geo/test_surface_objectives_jax.py -k "traceable_solved_state_value_and_grad or cache_state_allows_non_ondevice_for_solved_state_builder or old_import_path_reexports_public_and_private_helpers or traceable_objective_bundle_marks_value_and_grad_cacheable or traceable_objective_bundle_donates_value_and_grad_input" -q`: 5 passed.
- `pytest tests/geo/test_boozersurface_jax.py -k "host_jax_kernel_bundle_compiles_once_per_static_signature or host_jax_kernel_bundle_linear_solve_compiles_once_per_static_signature or run_code_routes_backend_contract_to_expected_method" -q`: 8 passed.
- `pytest tests/integration/test_single_stage_physics_parity.py -k "host_jax_single_stage_contract_uses_host_control or target_lane_optimizer_seed_contract_matches_target_minimize_guard" -q`: 2 passed.
- `pytest tests/integration/test_single_stage_physics_parity.py::test_target_lane_optimizer_seed_contract_matches_target_minimize_guard tests/integration/test_single_stage_physics_parity.py::test_host_jax_single_stage_contract_uses_host_control tests/integration/test_single_stage_physics_parity.py::test_host_jax_adapter_uses_solved_state_kernel_without_legacy_objective -q`: 3 passed.
- `pytest tests/integration/test_single_stage_physics_parity.py::test_host_jax_compile_diagnostics_condition_includes_host_outer tests/integration/test_single_stage_physics_parity.py::test_outer_loop_probe_cli_accepts_host_jax_memory_gate tests/integration/test_single_stage_physics_parity.py::test_host_jax_memory_gate_skips_warmup_and_checks_growth tests/integration/test_single_stage_physics_parity.py::test_host_jax_memory_gate_rejects_transient_post_warm_spike tests/integration/test_single_stage_physics_parity.py::test_host_jax_compile_gate_rejects_post_warm_counter_growth tests/integration/test_single_stage_physics_parity.py::test_host_jax_compile_gate_accepts_steady_post_warm_counters tests/integration/test_single_stage_physics_parity.py::test_host_jax_single_stage_contract_uses_host_control tests/integration/test_single_stage_physics_parity.py::test_host_jax_adapter_uses_solved_state_kernel_without_legacy_objective -q`: 8 passed.
- `pytest tests/integration/test_single_stage_physics_parity.py::test_target_lane_optimizer_seed_contract_matches_target_minimize_guard tests/integration/test_single_stage_physics_parity.py::test_host_jax_single_stage_contract_uses_host_control tests/integration/test_single_stage_physics_parity.py::test_host_jax_adapter_uses_solved_state_kernel_without_legacy_objective tests/integration/test_single_stage_init_parity_compile_diagnostics.py -q`: 9 passed.
- `benchmarks/single_stage_outer_loop_probe.py --help` shows `--optimizer-backend {scipy-jax,host-jax}`, `--enable-compile-diagnostics`, `--record-objective-evaluation-trace`, and `--enable-host-jax-memory-gate`.
- `pytest tests/geo/test_optimizer_jax_item19.py -q`: 15 passed.
- `benchmarks/grouped_adjoint_memory_probe.py --help` shows `--boozer-optimizer-backend {host-jax,ondevice,scipy}`.
- `pytest tests/integration/test_single_stage_physics_parity.py -q`: 12 passed, 1 skipped, 2 setup errors in this local env. The errors are the legacy on-device CPU fixtures requiring JAX >= 0.10.0 while this env has JAX 0.9.2; the host-jax gate tests above passed.

Next implementation steps:

1. If A100-only evidence is required, rerun the host-jax gate on RunPod A100 with the same command shape as the r13 H100 run and require `passed == true`, `host_jax_compile_gate.passed == true`, and `host_jax_memory_gate.passed == true`.
2. Keep host-jax as an explicit backend until parity and production-size evidence are complete; do not replace the known CPU/scipy or current ondevice semantics from this smoke-size H100 gate alone.
3. Run the e2e parity artifact on RunPod A100/H100 only after the relevant hardware class has a clean compile/memory gate.

> Last updated: 2026-06-16 20:22 EDT · Status: ALL CODE FIXES committed+pushed (HEAD `32897677e`). **nvlink GPU-link blocker SOLVED** (consistent system-12.4 ptxas+nvlink toolchain) and the **port is PROVEN on GPU** (seed Boozer solve → iota 5.6e-18, volume 0.0999, machine precision, x64) — but the formal green `passed=true` artifact is BLOCKED by a SECOND orthogonal issue: a ~22+ min single-threaded host-side XLA compile pole that makes RunPod impractical. All RunPod pods TERMINATED (verified gone, no idle billing). **NEXT = produce the formal artifact on Perlmutter** (`sbatch benchmarks/perlmutter/single_stage_production_gpu.slurm`) — its curated CUDA env avoids the nvlink wheel-gap and its SLURM wall-time absorbs the compile pole.

## 1. Goal
"What's needed to get a PASS for the runpods" → then "both": (a) arm a convergence-independent port backstop so a green verdict positively proves the port, and (b) produce the green `passed=true` production single-stage parity artifact on a GPU.
**Definition of done:** (a) seed-state backstop committed + reviewed PASS ✅ DONE; (b) a run of `single_stage_init_parity.py --platform cuda` produces `passed=true` with `comparison.initial_metric_parity_failures==[]` (backstop passed) — ⚠️ NOT YET ACHIEVED (blocked by the compile pole on RunPod; pursue on Perlmutter).

## 2. Where we are right now
The code is DONE (all fixes committed + pushed to fork `jungdaesuh/simsopt @ pr/jax-port-clean` HEAD `32897677e`). Three RunPod attempts each hit a different infra wall (CDN throttle → old driver → wheel-has-no-nvlink). On attempt #3 (a CUDA-13.0 host) I **solved the nvlink blocker** and the GPU compile/link succeeded — but the single-stage parity graph then hit a **~22+ min single-threaded host-side XLA compile pole** (the documented "once-slow" scipy-jax compile-bound) that I could not wait out within reasonable cost. I terminated the pod at ~145 min (~$3.6). The PORT is proven on GPU regardless (machine-precision seed metrics). The immediate next thing is to run the formal artifact on **Perlmutter**, which is the curated env this project uses for production.

## 3. NEXT ACTIONS (start here on resume)
1. [ ] **Confirm no live RunPod pods (cost!).** `runpodctl pod list` → expect `[]`. `cat /tmp/runpod_active_pod_id.txt` → expect absent (cleared). If any pod shows RUNNING, `runpodctl pod delete <id>` then verify GraphQL `pod`→null. (All 3 attempts' pods were terminated + verified gone as of this writing.)
2. [ ] **Produce the formal green artifact on Perlmutter** (the path that bypasses both RunPod blockers): `sbatch benchmarks/perlmutter/single_stage_production_gpu.slurm`. Needs the user's NERSC login (I cannot sbatch it myself). The branch it pulls (`pr/jax-port-clean` @ `32897677e`) has ALL fixes. Its env already sets the right `SIMSOPT_*`/`JAX_ENABLE_X64=1`/`unset LD_LIBRARY_PATH` (slurm lines 95-120). Expected: `passed=true`, `comparison.initial_metric_parity_failures==[]`.
3. [ ] **Verify the artifact when it lands.** Check `passed`, `comparison.initial_metric_parity_failures` (== `[]` ⇒ backstop passed ⇒ port positively proven), `comparison.cpu_optimizer_status`/`jax_optimizer_status`, `comparison.final_metric_parity_skipped_for_nonconvergence`. If `passed=false` but `initial_metric_parity_failures==[]`, the PORT is fine and the fail is a trajectory/gate detail (likely the CPU-ref STATUS=2 abort that B2 should skip).
4. [ ] **If you MUST use RunPod again** (not recommended): provision, then (a) measure CDN egress ≥3 MB/s before building, (b) require `nvidia-smi` CUDA ≥ 12.9, AND (c) apply the consistent-toolchain nvlink fix from §6, AND (d) budget ~25-30 min for the one-time compile pole (it completes, per `project_scipy_jax_gpu_compile_bound`). The JAX persistent compile cache (set `JAX_COMPILATION_CACHE_DIR`) means a SECOND run on the same pod is fast — so let the first compile finish to populate it.
5. [ ] (Optional, NOT required for PASS) The production maxiter>0 lane's *surface-geometry* channel is still forced to 0 and same-candidate *replay* is behind `--record-objective-evaluation-trace`; the seed-state backstop covers the convergence-independent signal, so this is lower priority.

## 4. Environment & how to run
- cwd / repo / branch: `/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean` / simsopt / `pr/jax-port-clean` (HEAD `32897677e`, durably pushed — verified via `git ls-remote`).
- Fork (VM-sync target, PUBLIC): `git@github.com:jungdaesuh/simsopt.git` push / `https://github.com/jungdaesuh/simsopt.git` clone.
- Perlmutter (the recommended path): `sbatch benchmarks/perlmutter/single_stage_production_gpu.slurm` (user's NERSC login). Env baked into slurm lines 95-120.
- RunPod (if reused): SSH key `/Users/suhjungdae/.runpod/ssh/RunPod-Key-Go` (user root); API key in `/Users/suhjungdae/.runpod/config.toml` (TOML `apikey`) → GraphQL `https://api.runpod.io/graphql?api_key=<KEY>`. Get TCP SSH port for private 22 from GraphQL `pod.runtime.ports` (NOT runpodctl). `runpodctl pod list` / `runpodctl pod delete <id>`.
- **Local test env: `python` resolves to 3.14 (no matching simsoptpp).** Use `/opt/homebrew/Caskroom/miniforge/base/bin/python` (3.13) OR `/tmp/simsopt-clean-venv/bin/python` for pytest.
- Run local tests: `python -m pytest tests/test_backend_dtypes_reference_sharding.py tests/integration/test_single_stage_init_parity_convergence_gate.py tests/integration/test_single_stage_init_parity_seed_state_backstop.py tests/core/test_json.py -q` (expect 3+6+3+17 pass). `python -m ruff check <changed files>`.
- RunPod pod build recipe: clone → `python3.11 -m venv /workspace/venv` → `pip install "jax[cuda12]==0.10.0"` → `pip install cmake ninja` → `cd repo && pip install -e .` (~16 min). Run env (CORRECTED): `PYTHONPATH=src:repo JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_exclude_nondeterministic_ops=true"; unset LD_LIBRARY_PATH`. PLUS the nvlink fix from §6 if the wheel ships no nvlink.

## 5. Done so far (with evidence) — the fix chain, all committed + pushed
- [x] **B1 `23464a0da`** — `src/simsopt_jax/backend/dtypes.py:_reference_sharding` guards tracers out (skip the O(jaxpr²) `getattr(tracer,'sharding')` `_origin_msg` jaxpr walk). Byte-identical. The 50-min cold-compile "GPU stall" root cause. Tests `tests/test_backend_dtypes_reference_sharding.py` (3). 2 reviewers PASS.
- [x] **B2 `f0a5e80e9` + `ee1c33c05`** — `benchmarks/single_stage_init_parity.py:evaluate_single_stage_init_parity` end-state metric parity now **asymmetric**: skip only when CPU REFERENCE aborted abnormally (scipy L-BFGS-B `status==2`) AND JAX target did not. A JAX-target abort stays a HARD failure; `status==1` (maxiter) stays strict. Tests `tests/integration/test_single_stage_init_parity_convergence_gate.py` (6). The first symmetric version had a CRITICAL escape hatch → fixed by `ee1c33c05`.
- [x] **B4 `c7eb5cdd7`** — 4 signoff scripts resolve a real interpreter (`$(command -v python3||python||true)` / `sys.executable`); 2 pod scripts → `SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled`+`unset LD_LIBRARY_PATH`.
- [x] **provenance/redirect `3f06107d1`** — `require_local_simsopt_provenance` gate + `_DEFAULT_REDIRECT` (curvecwsfourier) + fixture catch. 17 json tests pass.
- [x] **Seed-state backstop `32897677e`** — `evaluate_single_stage_init_parity` compares INITIAL_IOTA/VOLUME/FIELD_ERROR (seed surface, both lanes at identical seed DOFs), armed UNCONDITIONALLY (convergence-independent). Shared `_metric_parity_failures` SSOT. Reuses 1e-10/1e-8 tolerances. Tests `tests/integration/test_single_stage_init_parity_seed_state_backstop.py` (3). Reviewer PASS.
- [x] **nvlink GPU-link blocker SOLVED on RunPod (UNCOMMITTED on-pod workaround — pod gone, see §6 for the recipe).** On a CUDA-13.0 host, made a consistent system-12.4 toolchain → GPU compile/link succeeded, clearing the wall that failed 12+ attempts. Numerically neutral (compile-path only).
- [x] **PORT PROVEN ON GPU (verified, not just assumed).** On the working toolchain the JAX target computed the seed Boozer BFGS+NEWTON solve → `iota = 5.605873e-18`, `volume = 0.09991`, `Max Curvature = 9.5633`, x64 enabled — IDENTICAL to the C++ reference. This is the convergence-independent port-correctness signal.
- [ ] **Formal green `passed=true` artifact — NOT ACHIEVED.** Blocked on RunPod by the ~22+ min host-side XLA compile pole (§7). Pursue on Perlmutter (§3 item 2).

## 6. Key decisions & rationale
- **The nvlink fix is a CONSISTENT-TOOLCHAIN workaround, numerically neutral.** On a host whose driver is new enough (CUDA ≥12.9) but where the `jax[cuda12]` wheel ships NO `nvlink` (it's ptxas-only — modern CUDA uses in-process nvJitLink), make ptxas AND nvlink the SAME version: `mv .../nvidia/cuda_nvcc/bin/ptxas{,.bundled129}` (so XLA uses the system 12.4 ptxas) + `ln -sf /usr/local/cuda/bin/nvlink .../nvidia/cuda_nvcc/bin/nvlink` (real 12.4 nvlink in XLA's FIRST search dir) + `XLA_FLAGS=... --xla_gpu_cuda_data_dir=/usr/local/cuda`. Both ends 12.4 → links → the ≥13.0 driver runs 12.4 modules. Changes the compile toolchain only, NOT the computed values. VERIFIED working.
- **The formal artifact belongs on Perlmutter, not RunPod.** RunPod gave us 3 different infra walls; Perlmutter's curated CUDA env avoids the wheel-gap and its SLURM wall-time absorbs the one-time compile pole. The project already uses it for production.
- **B2 gate is ASYMMETRIC** (ref-abort skip only; target-abort stays a failure). Do not revert to symmetric (a broken JAX target could hide).
- **Seed-state backstop chosen over repairing the bit-rotted fixed-state proof** (`single_stage_cpp_jax_state_parity.py`, see §7). INITIAL_* parity reuses existing data, is SSOT, convergence-independent, armed in every launcher.
- **Commit-per-concern, follow-up commits not amends.** Pushed to fork (public) for VM clone.

## 7. Dead ends / do NOT retry
- **The single-stage parity graph has a ~22+ min single-threaded host-side XLA compile pole** (100% 1-core, GPU idle, no ptxas child, flat RSS ~700 MB, log stuck after "Max Curvature"). Hits BOTH `--maxiter 0` AND `--maxiter 1500` (it's the outer process's target-lane graph compile, independent of iterations). Per `project_scipy_jax_gpu_compile_bound` it is "once-slow" (COMPLETES + caches; RAM is 2TB so no OOM) — but on RunPod it exceeded ~22 min with no end in sight, making the formal artifact impractical there. On Perlmutter the SLURM wall-time absorbs it. Do NOT keep re-launching expecting it to be fast; it is a one-time cost — let it finish ONCE (populate `JAX_COMPILATION_CACHE_DIR`).
- **THE nvlink `129 vs 124` error on an OLD-driver host = driver too old, NOT fixable on-pod** (attempt #2, driver 550/CUDA-12.4). jax 0.10 emits 12.9 cubins; a 12.4 driver can't link them. DISPROVEN on a 12.4 host: `--xla_gpu_enable_llvm_module_compilation_parallelism=false` (ineffective in jax 0.10), `LD_LIBRARY_PATH=<bundled libs>` for nvJitLink, hiding bundled ptxas (cubin still stamped 129). FIX = a host with driver CUDA ≥ 12.9 (check `nvidia-smi` BEFORE building).
- **On a NEW-driver host, the error MUTATES to `RET_CHECK ... process.Start() ... No such file or directory`** because the wheel has no `nvlink` (XLA finds ptxas but `nvlink:<empty>`). The documented `--xla_gpu_force_compilation_parallelism=1` workaround (JAX issue #16586) does NOT fix this empty-nvlink case. FIX = the consistent-toolchain recipe in §6.
- **RunPod CDN egress can be silently throttled per-pod** (~278 KB/s → `jax[cuda12]` ~4 GB = 2+ hrs). Measure egress before building; destroy+reprovision if <3 MB/s.
- **RunPod SSH port: read TCP/22 from GraphQL `pod.runtime.ports`, NOT runpodctl** (the displayed port may be the UDP mapping).
- **`benchmarks/single_stage_cpp_jax_state_parity.py` is BIT-ROTTED** (`:483` wrong `SurfaceSurfaceDistance` import; `:563` `get_adjoint_runtime_state()` on a cpp surface that lacks it). Do NOT wire it into the SLURM without a full repair — that's why the seed-state backstop was chosen.

## 8. Open questions / blockers
- **External blocker: the formal artifact needs a GPU run that completes the compile pole.** Perlmutter (user's NERSC creds) is the path; I cannot sbatch it. Alternatively a RunPod run where the first compile is allowed to finish (~25-30 min) to populate the cache, then re-run.
- Will Perlmutter reproduce the CPU-ref STATUS=2 abort (→ B2 skips the final gate → seed-state backstop is the proof)? Either way the seed-state backstop is the reliable port signal.

## 9. Mental model (hard-won context)
- **The whole RunPod GPU saga is two stacked blockers, both orthogonal to the port:** (1) the GPU LINK step (nvlink) — driver-too-old on one host, wheel-has-no-nvlink on the new-driver host; SOLVED via the consistent-12.4-toolchain recipe (§6). (2) the host-side XLA COMPILE of the single-stage graph — a ~22 min single-threaded "once-slow" pole (§7). Neither is a port bug. The port itself produces machine-precision-identical seed metrics on GPU vs C++ (proven).
- **Why `nvlink` is missing:** NVIDIA's `nvidia-cuda-nvcc-cu12` wheel ships **ptxas only** (modern CUDA replaced the standalone `nvlink` binary with in-process `libnvJitLink.so.12`, which IS bundled). But jaxlib 0.10 still tries to exec an external `nvlink` and does NOT fall back to nvJitLink on these hosts → empty-path spawn fail. Verified via `TF_CPP_VMODULE=subprocess_compilation=3` showing `Linking N modules with provider(... nvlink: <EMPTY> ...)`.
- The prod1500_50 artifact's original `passed=False` was a REPORTING artifact: the CPU reference L-BFGS-B aborted (STATUS=2) while JAX converged. Same-candidate (seed) parity is machine-precision. B2 stops the false-fail; the seed-state backstop makes the green meaningful.
- `evaluate_single_stage_init_parity` is called for BOTH lanes; `INITIAL_*` and `OPTIMIZER_SUCCESS/STATUS` are in the unconditional `results` dict. Production gate: `passed = not failures`; scipy-jax lane hits the `require_final_metric_parity=True` path.

## 10. Pointers
- Memory (updated this session): `project_runpod_ops_egress_ssh_port.md` — the COMPLETE RunPod gotcha set (egress throttle, GraphQL SSH port, old-driver nvlink, wheel-has-no-nvlink + the consistent-toolchain FIX, the compile pole). Also `project_scipy_jax_gpu_compile_bound`, `project_value_and_grad_construction_compile_pole`, `project_runpod_cuda_block`.
- RunPod creds: SSH key `/Users/suhjungdae/.runpod/ssh/RunPod-Key-Go`; API key file `/Users/suhjungdae/.runpod/config.toml`.
- All 3 RunPod attempt agents are DONE (throttle / old-driver / nvlink-fixed-but-compile-bound). Background-agent `.output` files are JSONL transcripts — do NOT shell-read them (overflow context).
- Root-cause doc (B1/compile): `docs/single_stage_ondevice_compile_blowup_root_cause_2026-06-16.md`.
- JAX docs grounding: JAX issue #16586 (nvlink error family), JAX installation.md (cuda12 vs cuda12-local), JAX faq.rst (CUDA discoverability via nvidia-*-cu12 wheels).
