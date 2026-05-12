# Banana JAX Native Parity Completion Audit - 2026-05-12

Verdict: PARTIAL.

This audit records the current state for
`docs/banana_jax_native_parity_goal_prompt_2026-05-12.md`. It is not a
release signoff. The local CPU/JAX proof surfaces pass after the current fixes,
but the completion contract still requires current-SHA real CUDA evidence:

`CPU/C++/SciPy oracle -> JAX CPU -> JAX CUDA/GPU -> JAX CPU/GPU agreement`

## Current Tree

- HEAD inspected: `215891905d357ab370b2a5ba21d47eab7efec99d`.
- Dirty tracked files are the intended parity/fix files:
  - `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
  - `src/simsopt/geo/boozersurface_jax.py`
  - `src/simsopt/geo/optimizer_jax.py`
  - `src/simsopt/geo/surfaceobjectives_jax.py`
  - `tests/geo/test_boozersurface_jax_private.py`
  - `tests/geo/test_single_stage_example.py`
  - `tests/geo/test_surface_objectives_jax.py`
  - `tests/integration/test_single_stage_physics_parity.py`
- Many preexisting untracked `.artifacts/`, `.claude/`, `.conda/`, image,
  VTU, and scratch files remain unclaimed and were not treated as proof.
- Local runtime: JAX `0.9.2`, jaxlib `0.9.2`, backend `cpu`, devices
  `[('cpu', 'cpu')]`.
- CUDA proof state: `nvidia-smi` is unavailable locally, and
  `rg -l "215891905d357ab370b2a5ba21d47eab7efec99d" .artifacts/runpod_prod_signoff .artifacts/parity .artifacts/pytest`
  returned no current-SHA artifacts.

## Lane Verdicts

| Lane | Status | Current evidence | Remaining blocker |
| --- | --- | --- | --- |
| CPU/C++/SciPy oracle | PASS for covered local surfaces | CPU oracle/reference tests in the parity suites below pass; upstream CPU lane was not rewritten by the current fix. | Not a full upstream SIMSOPT release-suite proof by itself. |
| JAX CPU | PASS for covered local surfaces | Stage 2, Boozer, field, objective, traceable objective, and single-stage physics parity tests pass on CPU with x64. | None for the covered CPU surfaces. |
| JAX GPU | MISSING | Local runtime has only CPU devices; no current-SHA CUDA artifact exists. | Must run on real CUDA hardware with required provenance. |
| CPU-vs-JAX CPU | PASS for covered surfaces | Fixed-state and optimizer-behavior proof surfaces listed below pass locally. | Broader release matrix still depends on current GPU proof. |
| CPU-vs-JAX GPU | MISSING | No current real CUDA run. | Need current-SHA CUDA parity JSON and artifact bundle. |
| JAX CPU-vs-GPU | MISSING | No current real CUDA run. | Need same-state CPU/GPU JAX comparison with device provenance. |

Overall native JAX port status is PARTIAL because every CUDA row is still open.

## Fixes Included In This State

- Operator-only least-squares Hessian adjoint solve:
  `src/simsopt/geo/optimizer_jax.py:2454` adds
  `_solve_hessian_least_squares_system_with_status`, using the Hessian linear
  operator and normal equations instead of dense PLU materialization for
  singular least-squares adjoint systems.
- Traceable target lane avoids dense linearization materialization:
  `src/simsopt/geo/boozersurface_jax.py:4516` accepts
  `materialize_dense_linearization`, and the target objective path passes
  `False` at `src/simsopt/geo/surfaceobjectives_jax.py:3294`.
- Dense linear-solve factors are now reporting/parity artifacts outside the
  target runtime contract:
  `src/simsopt/geo/surfaceobjectives_jax.py:3195` returns no target-lane
  factors, and `src/simsopt/geo/surfaceobjectives_jax.py:4851` records the
  least-squares operator diagnostics separately.
- Single-stage smoke parity no longer consumes a stale high-resolution JAX
  runtime seed spec. `tests/integration/test_single_stage_physics_parity.py:227`
  compiles a smoke-resolution runtime spec from the CPU Stage 2 init fixture.
- Native JAX output loading now reconstructs from
  `single_stage_jax_runtime_spec.json` when CPU-style `surf_opt.json` and
  `biot_savart_opt.json` are intentionally absent:
  `tests/integration/test_single_stage_physics_parity.py:447`.
- Deferred `XYZTensorFourier` surfaces are reprojected instead of bypassed:
  `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:3473`.

## Prompt Checklist

| Prompt requirement | Status | Evidence |
| --- | --- | --- |
| Inspect HEAD and dirty state before claiming | DONE | HEAD and `git status --short` refreshed in this audit. |
| Do not use docs, fake CUDA, dry-runs, or old artifacts as completion proof | DONE | Historical artifacts were excluded; current-SHA artifact sweep found no matches. |
| CPU/C++/SciPy -> JAX CPU fixed-state proof | PASS for covered surfaces | Field/objective/Boozer/Stage 2/single-stage CPU proof surfaces passed. |
| Real JAX CUDA/GPU parity | BLOCKED | No CUDA device and no current-SHA CUDA artifact. |
| JAX CPU/GPU agreement | BLOCKED | No current GPU run. |
| Keep fixed-state oracle checks separate from optimizer traces | DONE | Fixed-state suites and optimizer/integration suites are listed separately below. |
| Do not route production JAX/GPU through host SciPy | IMPROVED | Target traceable solve path now avoids dense PLU materialization and uses operator-only JAX solves for the fixed least-squares adjoint case. |
| Use `benchmarks/validation_ladder_contract.py` as SSOT | DONE | GPU proof lanes and metadata remain centralized there, including `gpu_runtime`, CPU/GPU reduction lanes, and `GPU_PROOF_PARITY_CONTRACTS`. |
| Stale artifact sweep | DONE | No current-SHA artifact found in `.artifacts/runpod_prod_signoff`, `.artifacts/parity`, or `.artifacts/pytest`. |
| Stale code/test fixture repair | DONE | Single-stage physics parity now derives its smoke runtime spec from the current CPU Stage 2 fixture instead of a stale high-resolution runtime spec. |
| Upstream/downstream/E2E regression | PARTIAL | Local downstream consumers/tests pass; current real GPU E2E remains missing. |
| Completion condition | NOT MET | CUDA/GPU proof and CPU/GPU agreement are still missing. |

## Host-Boundary Inventory

Allowed boundaries:

- CLI parsing, setup compatibility, immutable runtime spec construction, restart
  artifact loading, and final JSON writing.
- Test fixture compilation from CPU Stage 2 init artifacts into a JAX runtime
  seed spec.
- Final report/artifact reads, parity JSON readers, and benchmark measurement
  boundaries that synchronize accelerator work before reading results.

Intentional diagnostic/reporting paths:

- Compile/provenance helpers, including CUDA/JAX metadata collection.
- Dense or structured linearization metadata when explicitly requested for
  diagnostics or parity reporting, not for the target traceable value/gradient
  contract.
- `hessian_least_squares_operator` diagnostics in traceable objective reports.

Open port gaps:

- No current-SHA target-lane execution on CUDA is available.
- No current-SHA GPU proof shows target arrays or compiled executable running
  on CUDA.
- No current-SHA E2E GPU path proves Stage 2 strict reduced/full output into
  single-stage init/continuation into parity matrix/proof report.

## Official Docs Checked

- JAX stateful computations:
  https://docs.jax.dev/en/latest/stateful-computations.html
  - Applied as the purity constraint for jitted target compute and explicit
    state threading.
- JAX transfer guard:
  https://docs.jax.dev/en/latest/transfer_guard.html
  - Applied to host/device transfer classification.
- JAX asynchronous dispatch:
  https://docs.jax.dev/en/latest/async_dispatch.html
  - Applied to benchmark/proof requirements around synchronization at
    measurement boundaries.
- SIMSOPT stable docs:
  https://simsopt.readthedocs.io/v1.8.3/
  https://simsopt.readthedocs.io/v1.8.3/simsopt.geo.html
  https://simsopt.readthedocs.io/v1.8.3/simsopt.field.html
  - Used to keep BoozerSurface, BiotSavart, SquaredFlux, and CPU oracle
    semantics anchored to public SIMSOPT behavior.
- NVIDIA CUDA C Programming Guide:
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
  - Applied to the CUDA provenance fields required before claiming GPU parity.

## Validation Commands

Commands run after the current fixes:

```bash
JAX_ENABLE_X64=True JAX_PLATFORMS=cpu python -m pytest -q tests/integration/test_single_stage_physics_parity.py
```

Result: `4 passed, 1 skipped in 1459.68s`.

```bash
JAX_ENABLE_X64=True JAX_PLATFORMS=cpu python -m pytest -q \
  tests/geo/test_single_stage_example.py::SingleStageExampleTests::test_project_surface_dofs_to_resolution_reprojects_deferred_xyz_surface \
  tests/geo/test_single_stage_example.py::SingleStageExampleTests::test_project_surface_dofs_to_resolution_returns_matching_deferred_xyz_dofs \
  tests/geo/test_surface_objectives_jax.py::test_traceable_custom_vjp_surfaces_adjoint_solve_failure_as_nan_gradient \
  tests/geo/test_surface_objectives_jax.py::test_traceable_objective_gradient_parts_use_strict_vjp_helpers \
  tests/geo/test_surface_objectives_jax.py::test_traceable_hessian_solve_uses_configured_stabilization_once \
  tests/geo/test_surface_objectives_jax.py::test_traceable_hessian_solve_uses_configured_stabilization_under_jit \
  tests/geo/test_boozersurface_jax_private.py::TestBoozerSurfaceJAXClassPrivate::test_hessian_least_squares_system_solves_singular_minimum_residual \
  tests/geo/test_boozersurface_jax_private.py::TestBoozerSurfaceJAXClassPrivate::test_hessian_system_status_jaxpr_stays_operator_only
```

Result: `8 passed in 3.05s`.

```bash
python -m py_compile \
  examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
  src/simsopt/geo/boozersurface_jax.py \
  src/simsopt/geo/optimizer_jax.py \
  src/simsopt/geo/surfaceobjectives_jax.py \
  tests/geo/test_boozersurface_jax_private.py \
  tests/geo/test_single_stage_example.py \
  tests/geo/test_surface_objectives_jax.py \
  tests/integration/test_single_stage_physics_parity.py
python -m ruff check \
  examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
  src/simsopt/geo/boozersurface_jax.py \
  src/simsopt/geo/optimizer_jax.py \
  src/simsopt/geo/surfaceobjectives_jax.py \
  tests/geo/test_boozersurface_jax_private.py \
  tests/geo/test_single_stage_example.py \
  tests/geo/test_surface_objectives_jax.py \
  tests/integration/test_single_stage_physics_parity.py
python -m ruff format --check \
  examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
  src/simsopt/geo/boozersurface_jax.py \
  src/simsopt/geo/optimizer_jax.py \
  src/simsopt/geo/surfaceobjectives_jax.py \
  tests/geo/test_boozersurface_jax_private.py \
  tests/geo/test_single_stage_example.py \
  tests/geo/test_surface_objectives_jax.py \
  tests/integration/test_single_stage_physics_parity.py
git diff --check
```

Result: py_compile passed; `ruff check` passed; `ruff format --check` reported
`8 files already formatted`; `git diff --check` passed.

Earlier proof-surface results from this repair pass:

- Docs/manifest/target-lane proof slice:
  `tests/docs/test_banana_parity_coverage_manifest.py`,
  `tests/test_single_stage_cpp_jax_state_parity.py`,
  `tests/test_hf_production_gpu_proof.py`,
  `tests/integration/test_stage2_target_lane_purity.py`:
  `62 passed in 30.96s`.
- Field/objective fixed-state parity:
  `tests/field/test_biotsavart_jax_parity.py`,
  `tests/objectives/test_fluxobjective_jax_parity.py`,
  `tests/objectives/test_integral_bdotn_jax.py`:
  `98 passed, 58 skipped in 22.19s`.
- Boozer JAX:
  `tests/geo/test_boozersurface_jax.py`:
  `374 passed, 4 skipped in 361.46s`.
- Stage 2 JAX:
  `tests/integration/test_stage2_jax.py`:
  `173 passed in 185.07s`.
- Traceable single-stage objective:
  `tests/integration/test_single_stage_jax_cpu_reference.py::TestTraceableObjective -vv`:
  `37 passed in 810.86s`.
- Full single-stage CPU reference:
  `tests/integration/test_single_stage_jax_cpu_reference.py`:
  `173 passed, 5 skipped in 983.52s`.

Current audit/provenance commands:

```bash
git rev-parse HEAD
git status --short
python -c "import jax, jaxlib; print('jax', jax.__version__); print('jaxlib', jaxlib.__version__); print('backend', jax.default_backend()); print('devices', [(d.platform, d.device_kind) for d in jax.devices()])"
nvidia-smi
rg -l "215891905d357ab370b2a5ba21d47eab7efec99d" .artifacts/runpod_prod_signoff .artifacts/parity .artifacts/pytest
```

Results: current SHA confirmed; local backend is CPU; `nvidia-smi` not found;
current-SHA artifact search returned no files.

## Remote Proof Launch Feasibility

The remote GPU path was checked after the local audit because CUDA proof is the
remaining release blocker.

- Default `hf` on this host currently fails before command dispatch with a
  local Typer/Click import mismatch. The repo-local shim
  `.artifacts/hf-cli-bin/hf` works via `uvx --from huggingface_hub hf`; it
  reports Hugging Face CLI `1.14.0` and can list Jobs hardware.
- `hf jobs hardware` through the shim lists `h200` as available at `$5.00/hour`.
- `runpodctl user` succeeds but reports a negative client balance, and
  `runpodctl pod list` returns `[]`; no existing Runpod pod can run the current
  proof.
- The fork branch `gpu-purity-stage2-20260405` is 21 commits behind local
  `HEAD` for this workspace state. `git ls-remote --heads fork
  gpu-purity-stage2-20260405` reports remote tip
  `7f5e526ef8a12992a22a3b525f04f794e9c1501e`, while local `HEAD` is
  `215891905d357ab370b2a5ba21d47eab7efec99d`.
- The HF production-proof preflight/dry-run against the current SHA failed
  before launching a paid job:

```bash
PATH="$PWD/.artifacts/hf-cli-bin:$PATH" \
SIMSOPT_HF_GPU_IMAGE=ghcr.io/jungdaesuh/simsopt-jax-hf-production-proof:banana-surface-parity-m7-image-r1 \
python benchmarks/hf_jobs/launch_production_gpu_proof.py \
  --dry-run \
  --hardware h200 \
  --platform cuda \
  --repo-url https://github.com/jungdaesuh/simsopt.git \
  --repo-ref gpu-purity-stage2-20260405 \
  --repo-sha 215891905d357ab370b2a5ba21d47eab7efec99d \
  --single-stage-jax-runtime-seed-spec benchmarks/fixtures/single_stage_seed_iota15/single_stage_jax_runtime_spec.json
```

Result:

```text
repo SHA 215891905d357ab370b2a5ba21d47eab7efec99d is not present on https://github.com/jungdaesuh/simsopt.git under refs/heads/gpu-purity-stage2-20260405; HF checkout would fail.
```

- The local tree contains `.github/workflows/jax_h200_production_proof.yml`
  in tracked history, but the fork branch is 21 commits behind local `HEAD`.
  `gh workflow list --repo jungdaesuh/simsopt --all` currently exposes
  `JAX HF CUDA Image` and `JAX Smoke Tests`, not the local
  `JAX H200 Production Proof` workflow.

## Remaining Blockers

1. Make the current proof state remotely executable: commit and push the exact
   proof state to a reachable ref, or use an approved remote path that applies
   the exact local patch and records the dirty-tree status as part of the proof.
2. Add usable GPU credits/capacity: Runpod is currently blocked by negative
   balance/no pods, and an HF H200 run should not be expected to succeed until
   account funding is available.
3. Run the current tree on real CUDA hardware and emit a current-SHA proof
   bundle with exact command, dirty-tree status, JAX/jaxlib versions, x64 mode,
   backend, CUDA/XLA flags, `CUDA_VISIBLE_DEVICES`, `nvidia-smi` facts,
   driver/runtime version, peak RSS, peak GPU memory, pass/fail metadata, and
   parity JSON.
4. Prove CPU/C++/SciPy oracle vs JAX GPU at identical fixed states, not only
   JAX-vs-JAX agreement.
5. Prove JAX CPU vs JAX GPU agreement on the same current states.
6. Run the current-SHA E2E GPU path:
   Stage 2 strict reduced/full output -> saved spec/restart/output artifacts ->
   single-stage init/continuation -> parity matrix/proof report.
7. After current CUDA proof exists, update any manifest rows or TODO checkboxes
   that still point at historical or non-current proof artifacts.
