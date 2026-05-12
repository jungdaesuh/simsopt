# Banana JAX Native Parity Completion Audit - 2026-05-12

Verdict: PARTIAL.

This audit records the current state for
`docs/banana_jax_native_parity_goal_prompt_2026-05-12.md`. It is not a
release signoff. The local CPU/JAX proof surfaces pass after the current fixes,
but the completion contract still requires current-SHA real CUDA evidence:

`CPU/C++/SciPy oracle -> JAX CPU -> JAX CUDA/GPU -> JAX CPU/GPU agreement`

## Current Tree

- Implementation proof-state commit inspected after scoped commit/push:
  `03a3243c76f377c303efdb00d0efbcf12e8d69b5`.
- Later audit-only status commits may advance branch `HEAD`; use
  `git rev-parse HEAD` for the live audit document commit.
- The intended parity/fix files are committed in
  `03a3243c76f377c303efdb00d0efbcf12e8d69b5` and clean after commit:
  - `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
  - `src/simsopt/geo/boozersurface_jax.py`
  - `src/simsopt/geo/optimizer_jax.py`
  - `src/simsopt/geo/surfaceobjectives_jax.py`
  - `tests/geo/test_boozersurface_jax_private.py`
  - `tests/geo/test_single_stage_example.py`
  - `tests/geo/test_surface_objectives_jax.py`
  - `tests/integration/test_single_stage_physics_parity.py`
  - `docs/banana_jax_native_parity_completion_audit_2026-05-12.md`
- Many preexisting untracked `.artifacts/`, `.claude/`, `.conda/`, image,
  VTU, and scratch files remain unclaimed and were not treated as proof.
- Local runtime: JAX `0.9.2`, jaxlib `0.9.2`, backend `cpu`, devices
  `[('cpu', 'cpu')]`.
- CUDA proof state: `nvidia-smi` is unavailable locally, and for the
  implementation proof-state commit
  `rg -l "03a3243c76f377c303efdb00d0efbcf12e8d69b5" .artifacts/runpod_prod_signoff .artifacts/parity .artifacts/pytest`
  returned no proof-state CUDA artifacts.

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
rg -l "03a3243c76f377c303efdb00d0efbcf12e8d69b5" .artifacts/runpod_prod_signoff .artifacts/parity .artifacts/pytest
```

Results: current SHA confirmed; local backend is CPU; `nvidia-smi` not found;
current-SHA artifact search returned no files.

After committing the scoped proof-state slice:

```bash
git rev-parse HEAD
git status --short -- docs/banana_jax_native_parity_completion_audit_2026-05-12.md \
  examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
  src/simsopt/geo/boozersurface_jax.py \
  src/simsopt/geo/optimizer_jax.py \
  src/simsopt/geo/surfaceobjectives_jax.py \
  tests/geo/test_boozersurface_jax_private.py \
  tests/geo/test_single_stage_example.py \
  tests/geo/test_surface_objectives_jax.py \
  tests/integration/test_single_stage_physics_parity.py
git push fork gpu-purity-stage2-20260405
git ls-remote --heads fork gpu-purity-stage2-20260405
git rev-list --left-right --count fork/gpu-purity-stage2-20260405...HEAD
```

Results: implementation proof-state `HEAD` was
`03a3243c76f377c303efdb00d0efbcf12e8d69b5`; the intended proof-state files
were clean; the fork branch resolved to the same SHA; ahead/behind was `0 0`.
Subsequent commits in this file are audit-only status updates, not code/test
changes to the proof path.

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
- The fork branch `gpu-purity-stage2-20260405` now contains the current
  proof-state commit
  `03a3243c76f377c303efdb00d0efbcf12e8d69b5`.
- The HF production-proof preflight/dry-run against the pushed current SHA
  passes and emits the expected H200 launch plan:

```bash
PATH="$PWD/.artifacts/hf-cli-bin:$PATH" \
SIMSOPT_HF_GPU_IMAGE=ghcr.io/jungdaesuh/simsopt-jax-hf-production-proof:banana-surface-parity-m7-image-r1 \
python benchmarks/hf_jobs/launch_production_gpu_proof.py \
  --dry-run \
  --hardware h200 \
  --platform cuda \
  --repo-url https://github.com/jungdaesuh/simsopt.git \
  --repo-ref gpu-purity-stage2-20260405 \
  --repo-sha 03a3243c76f377c303efdb00d0efbcf12e8d69b5 \
  --single-stage-jax-runtime-seed-spec benchmarks/fixtures/single_stage_seed_iota15/single_stage_jax_runtime_spec.json
```

Result: preflight JSON confirms `repo_ref_commit` equals
`03a3243c76f377c303efdb00d0efbcf12e8d69b5`, `platform` is `cuda`, hardware is
`h200`, and the command would clone the pushed branch and check out the exact
current SHA.

- The real foreground H200 proof launch was attempted with the same pushed
  SHA:

```bash
PATH="$PWD/.artifacts/hf-cli-bin:$PATH" \
SIMSOPT_HF_GPU_IMAGE=ghcr.io/jungdaesuh/simsopt-jax-hf-production-proof:banana-surface-parity-m7-image-r1 \
python benchmarks/hf_jobs/launch_production_gpu_proof.py \
  --hardware h200 \
  --platform cuda \
  --no-detach \
  --timeout 8h \
  --repo-url https://github.com/jungdaesuh/simsopt.git \
  --repo-ref gpu-purity-stage2-20260405 \
  --repo-sha 03a3243c76f377c303efdb00d0efbcf12e8d69b5 \
  --single-stage-jax-runtime-seed-spec benchmarks/fixtures/single_stage_seed_iota15/single_stage_jax_runtime_spec.json
```

Result: Hugging Face Jobs rejected the launch before job creation:

```text
Error: Client error '402 Payment Required' for url 'https://huggingface.co/api/jobs/CreativeEngineer'
Pre-paid credit balance is insufficient - add more credits to your account to use Jobs.
```

- After the audit-only GitHub Actions blocker update, the same foreground H200
  launch was retried at `2026-05-12T18:42:53Z` against pushed branch head
  `34b9cdaa748b40cc7ab7a7f98d1ca00512f55a5d`. Preflight resolved
  `repo_ref_commit` to the same SHA, with `platform=cuda`, `hardware=h200`,
  and expected JAX `0.9.2`. Hugging Face Jobs again rejected the launch before
  job creation with `402 Payment Required`, so no current CUDA proof artifact
  was produced.

- The local tree contains `.github/workflows/jax_h200_production_proof.yml`
  in tracked history, but `gh workflow run jax_h200_production_proof.yml
  --repo jungdaesuh/simsopt --ref gpu-purity-stage2-20260405` is blocked with
  HTTP 404 because GitHub Actions dispatch only sees workflows present on the
  repository default branch, currently `master`.
- `gh workflow list --repo jungdaesuh/simsopt --all` currently exposes `JAX HF
  CUDA Image` and `JAX Smoke Tests`, not `JAX H200 Production Proof`. `JAX HF
  CUDA Image` builds/pushes the CUDA image only; it does not execute the parity
  proof. Local `.github/workflows/jax_gpu_parity.yml` also has
  `workflow_dispatch` and self-hosted GPU jobs, but it is not exposed by the
  fork's default-branch workflow list.
- `JAX Smoke Tests` contains self-hosted GPU jobs in its YAML, but it has no
  `workflow_dispatch` trigger for this branch. `gh api
  repos/jungdaesuh/simsopt/actions/runners` returns `total=0`, so there is no
  usable GitHub Actions GPU capacity for the current proof.
- The direct HF launcher no longer depends on that workflow listing because the
  pushed branch/SHA preflight now passes.
- Read-only fallback checks for other authenticated cloud CLIs did not expose a
  usable immediate GPU route: the configured GCP project cannot query Compute
  Engine regions because billing is disabled, and the configured AWS region
  reports `0.0` quota for both "Running On-Demand G and VT instances" and
  "Running On-Demand P instances". The repo also has no AWS/GCP-native proof
  launcher comparable to `benchmarks/hf_jobs/launch_production_gpu_proof.py`.
- Lightning AI CLI lists GPU machines including `H200`, and a tiny
  `CPU_SMALL` smoke job can be submitted, inspected as `Completed`, and
  deleted. That route is not accepted as proof-ready here because
  `lightning inspect job` exposes only command/image/machine/status/cost, not
  stdout or proof artifacts, and the repo has no Lightning-native launcher or
  artifact export contract for the production CUDA proof.

## Remaining Blockers

1. Add usable GPU credits/capacity: HF Jobs H200 launch is currently rejected
   with `402 Payment Required`; Runpod is blocked by negative balance/no pods;
   GitHub Actions has no registered GPU runners; GCP billing is disabled; and
   the configured AWS GPU-family quotas are zero. Lightning needs a
   proof-artifact export path before it can be treated as a valid fallback.
2. Run the current tree on real CUDA hardware and emit a current-SHA proof
   bundle with exact command, dirty-tree status, JAX/jaxlib versions, x64 mode,
   backend, CUDA/XLA flags, `CUDA_VISIBLE_DEVICES`, `nvidia-smi` facts,
   driver/runtime version, peak RSS, peak GPU memory, pass/fail metadata, and
   parity JSON.
3. Prove CPU/C++/SciPy oracle vs JAX GPU at identical fixed states, not only
   JAX-vs-JAX agreement.
4. Prove JAX CPU vs JAX GPU agreement on the same current states.
5. Run the current-SHA E2E GPU path:
   Stage 2 strict reduced/full output -> saved spec/restart/output artifacts ->
   single-stage init/continuation -> parity matrix/proof report.
6. After current CUDA proof exists, update any manifest rows or TODO checkboxes
   that still point at historical or non-current proof artifacts.
