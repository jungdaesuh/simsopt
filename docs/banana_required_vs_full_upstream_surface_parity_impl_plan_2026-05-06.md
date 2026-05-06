# Banana-Required vs Full-Upstream Surface Parity Implementation Plan

Status: local non-CUDA implementation and validation evidence committed; M7 image publication and real-image preflight are complete for current runtime validation tag `banana-surface-parity-m7-unitnormal-r1`, but real CUDA evidence remains open because the H200 launch is blocked by external GPU account credits.
Date: 2026-05-06.
Base commit: `0bb26bb0a`.
Validation basis: committed implementation slice on 2026-05-06. Repo-local interpreter `.conda/jax-0.9.2/bin/python` reports JAX `0.10.0`, CPU backend only, with `JAX_ENABLE_X64=True` and `JAX_PLATFORMS=cpu` for local gates. Unrelated untracked artifact files were left unchanged.

## Verdict

For strict banana-required CPU/C++ precision and function parity, do not require every partial surface-family file to become a full legacy mirror. The banana contract is narrower than full upstream SIMSOPT parity:

- Field and flux kernels used by Stage 2.
- Stage 2 target bundle and reporting path.
- Boozer/single-stage objective path.
- Surface geometry/spec paths actually consumed by Stage 2 and single-stage workflows.

The larger "partial" labels mostly come from upstream surface API breadth: second-order surface tangents, forms/curvatures, scalar metric Hessians, host object utilities, broad I/O/copy behavior, and missing upstream surface objective wrappers. Those are valid full-upstream parity requirements, but they are not all banana ship blockers.

## Source Of Truth Split

- `docs/jax_parity_manifest.md` remains the status SSOT for parity rows.
- This file is the implementation plan for closing the quoted partial-file requirements.
- Banana readiness is judged by Requirement Set A plus current-sha CUDA artifacts where required by P5.
- Full legacy/upstream parity is judged by Requirement Set B.
- Do not mark manifest rows complete until the named tests and artifact evidence exist.

## Local Validation Evidence

CPU-only evidence from the 2026-05-06 committed implementation slice:

- Repo-local interpreter: `.conda/jax-0.9.2/bin/python`, JAX `0.10.0`, backend `cpu`, devices `[CpuDevice(id=0)]`.
- Manifest/harness gate: `52 passed`.
- Backend/smoke/native-path gate: `223 passed, 1 skipped`.
- Banana CPU/JAX parity gate: `589 passed, 134 skipped`.
- Boozer focused wrapper gate: `377 passed, 4 skipped`.
- Single-stage CPU reference closure gate: `173 passed, 5 skipped`.
- Full RZ/non-RZ surface parity gate: `247 passed, 51 skipped, 164 subtests passed`.
- Full non-RZ JAX surface file: `tests/geo/test_surface_fourier_jax.py -q` passed `123 passed`.
- Explicit heavy non-RZ normal Hessian gate: `tests/geo/test_surface_fourier_jax.py -q -k "SecondNormalDerivativeParity"` passed `4 passed, 119 deselected`.
- Full surface Taylor file: `tests/geo/test_surface_taylor.py -q` passed `19 passed, 142 subtests passed`.
- Surface conversion focused gate: `tests/geo/test_surface_taylor.py -q -k "surface_conversion"` passed `1 passed, 18 deselected, 4 subtests passed`.
- Full JAX surface-objective file: `tests/geo/test_surface_objectives_jax.py -q` passed `216 passed, 27 skipped`.
- Non-RZ object API focused gate: `tests/geo/test_surface_fourier_jax.py -q -k "ObjectApiParity and (copy_module_protocol or copy_object_api_variants or fit_to_curve or scale_object_api)"` passed `20 passed, 99 deselected`.
- Surface objective wrapper gate: `92 passed, 26 skipped, 40 subtests passed`.
- PrincipalCurvature helper dependency focused gate: `tests/geo/test_surface_objectives_jax.py -q -k "PrincipalCurvature"` passed `28 passed, 215 deselected`.
- Release-gate unit/schema checks: `28 passed`.
- HF launcher contract focused gate: `tests/test_hf_production_gpu_proof.py -q -k "requires_explicit_image_or_env or rejects_bootstrap_mode_override or rejects_remote_sha_not_on_repo_ref or accepts_matching_remote_repo_ref_and_sha"` passed `4 passed, 44 deselected`.
- Runtime seed fixture gate: `tests/test_benchmark_helpers.py::test_single_stage_init_fixture_runtime_seed_spec_loads` plus the HF runtime-seed accept/reject slices passed `3 passed`.
- `.artifacts/parity/coordinate-mapping-proof.json`: `status=pass`.
- `.artifacts/parity/fixed-state-cpu.json`: generated CPU artifact with `cpp_cpu_vs_jax_cpu` passing; overall `passed=false` because CUDA lanes were not evaluated on this CPU-only machine.

M7 CUDA launch preflight status in this local continuation:

- Implementation commit: `a514fa24c5a9adb849f963a4366cf07b74b37b47`.
- Audit refresh commit: `49c733fa4eef058cddda265228567ce3fc2120bc`.
- Validation-tag preflight refresh commit: `2861936d45fecf8a52aa8d87202281bc`.
- HF CUDA image workflow commit: `c90fac6a3c9e7c866f8e1806f8db5cde1f7c689c`.
- Runtime validation commit after the RZ unitnormal parity fix: `0b6293b075342acc5cf996160ecf4bd87f709610`.
- Pushed validation tag `banana-surface-parity-m7-unitnormal-r1` resolves to `0b6293b075342acc5cf996160ecf4bd87f709610` on `jungdaesuh/simsopt`.
- GHCR image: `ghcr.io/jungdaesuh/simsopt-jax-hf-production-proof:banana-surface-parity-m7-image-r1`.
- GHCR manifest digest: `sha256:eac2e1887eaf08628af62b28e5d7d7141b84afdfdcbfd00179823b1eb8f3df39`.
- Fork branch `jungdaesuh/simsopt:gpu-purity-stage2-20260405` still resolves to `7e3f2eb5e5462c7d3cc989ce8bf1fe010a04f3a2`; it was not moved because the local branch is ahead and the scoped validation tag is sufficient for launcher preflight.
- Local JAX devices are CPU-only: `[CpuDevice(id=0)]`; the unaffiliated default shell has `jax_enable_x64=False` unless the validation gate sets `JAX_ENABLE_X64=1`.
- Docker CLI is not available on this host (`command -v docker` produced no path), so the CUDA image was built by `.github/workflows/jax_hf_cuda_image.yml` instead of locally.
- Local global `hf` is still unusable, but the local shim `.artifacts/hf-cli-bin/hf` runs `uvx --from huggingface_hub hf`; `hf auth whoami` reports user `CreativeEngineer`.
- Runpod H200 stock is available, but `runpodctl user` reports a negative client balance and `runpodctl pod list` returns `[]`; no current-SHA Runpod CUDA proof can be launched without account credit.
- The current launcher has no patch/worktree upload path: it clones the configured repo/ref, checks out the resolved SHA, and validates the repo-relative seed paths at that SHA before launching.
- The old ad hoc bootstrap fallback path is intentionally unavailable: `--bootstrap-mode` is rejected, and `bootstrap_runtime.sh` requires a prebuilt `/opt/venv/bin/python` runtime with a GPU JAX backend.
- HF launcher dry-run preflight with the real GHCR image, `--repo-sha 0b6293b075342acc5cf996160ecf4bd87f709610`, and `--repo-ref banana-surface-parity-m7-unitnormal-r1` passes remote SHA/ref and repo-relative seed-path checks, then prints the H200 command.
- The real H200 launch with the same image, ref, SHA, and seed spec reaches Hugging Face when the local HF CLI shim is first on `PATH`, then fails before a job is created because Hugging Face Jobs returns HTTP 402 Payment Required: pre-paid credit balance is insufficient.

M7 unblock checklist:

- Use a pushed branch or tag that contains the exact runtime commit to be validated. `banana-surface-parity-m7-unitnormal-r1` already satisfies this for `0b6293b075342acc5cf996160ecf4bd87f709610`; create a new validation tag if later runtime code commits must be included in the CUDA proof.
- Use the published CUDA image from `benchmarks/hf_jobs/production_gpu_proof.Dockerfile`: `ghcr.io/jungdaesuh/simsopt-jax-hf-production-proof:banana-surface-parity-m7-image-r1`.
- Keep the launcher dry-run green; it must print a preflight report whose `repo_sha` equals the pushed validation commit and whose `image` is the published CUDA image.
- Add HF Jobs credits or Runpod credits/capacity, then run the H200/CUDA no-detach proof and attach the resulting artifact metadata here or in the manifest-linked proof doc. The manual GitHub Actions path through `.github/workflows/jax_h200_production_proof.yml` is only dispatchable after that workflow file exists on the repository default branch; the fork currently exposes only the HF image workflow on its default branch.

Committed validation-SHA implementation scope from `a514fa24c5a9adb849f963a4366cf07b74b37b47`:

Surface/objective parity slice required for this plan's non-CUDA claims:

- `benchmarks/validation_ladder_contract.py`
- `docs/banana_required_vs_full_upstream_surface_parity_impl_plan_2026-05-06.md`
- `docs/jax_parity_manifest.md`
- `src/simsopt/geo/surface_fourier_jax.py`
- `src/simsopt/geo/surfaceobjectives_jax.py`
- `src/simsopt/geo/surfacerzfourier.py`
- `src/simsopt/geo/surfacexyzfourier.py`
- `src/simsopt/geo/surfacexyztensorfourier.py`
- `src/simsopt/jax_core/__init__.py`
- `src/simsopt/jax_core/surface_fourier.py`
- `src/simsopt/jax_core/surface_rzfourier.py`
- `tests/geo/test_surface_fourier_jax.py`
- `tests/geo/test_surface_objectives_jax.py`
- `tests/geo/test_surface_rzfourier_jax.py`
- `tests/geo/test_surface_taylor.py`

M7/HF proof support slice included in `a514fa24c5a9adb849f963a4366cf07b74b37b47` and therefore required in the pushed target commit used for CUDA proof:

- `benchmarks/fixtures/single_stage_seed_iota15/single_stage_jax_runtime_spec.json` (required when the runtime seed is passed as the directory `benchmarks/fixtures/single_stage_seed_iota15`)
- `benchmarks/hf_jobs/cuda_pytest_probe.py` (required by `run_production_gpu_proof.sh` for the CUDA pytest proof payloads)
- `benchmarks/hf_jobs/launch_production_gpu_proof.py`
- `benchmarks/hf_jobs/run_production_gpu_proof.sh`
- `benchmarks/single_stage_init_parity.py`
- `benchmarks/stage2_e2e_comparison.py`
- `benchmarks/validation_ladder_common.py`
- `tests/subprocess/hf_production_gpu_fake_runner.py`
- `tests/test_benchmark_helpers.py`
- `tests/test_hf_production_gpu_proof.py`

Stage 2 / target-reporting dependency slice:

- `docs/banana_jax_full_test_parity_coverage_impl_plan_2026-05-06.md`
- `docs/banana_jax_native_port_todos_2026-05-05.md`
- `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py`
- `src/simsopt/geo/_distance_jax.py`
- `src/simsopt/geo/curveobjectives.py`
- `src/simsopt/objectives/stage2_target_objective_jax.py`
- `tests/integration/test_stage2_jax.py`
- `tests/objectives/test_integral_bdotn_jax.py`

This slice was not classified as pure surface/objective parity, but it affects
the banana CPU/JAX and Stage 2 reporting gates listed above. If it is excluded
from the validation SHA, rerun those gates against the committed SHA before
using the local pass counts as evidence.

Do not include unrelated `.artifacts/`, local `.conda/`, generated plots, or
older runpod output trees in the validation SHA.

Non-destructive scope audit command:

```bash
git status --short -- \
  benchmarks/validation_ladder_contract.py \
  docs/banana_required_vs_full_upstream_surface_parity_impl_plan_2026-05-06.md \
  docs/jax_parity_manifest.md \
  src/simsopt/geo/surface_fourier_jax.py \
  src/simsopt/geo/surfaceobjectives_jax.py \
  src/simsopt/geo/surfacerzfourier.py \
  src/simsopt/geo/surfacexyzfourier.py \
  src/simsopt/geo/surfacexyztensorfourier.py \
  src/simsopt/jax_core/__init__.py \
  src/simsopt/jax_core/surface_fourier.py \
  src/simsopt/jax_core/surface_rzfourier.py \
  tests/geo/test_surface_fourier_jax.py \
  tests/geo/test_surface_objectives_jax.py \
  tests/geo/test_surface_rzfourier_jax.py \
  tests/geo/test_surface_taylor.py \
  benchmarks/fixtures/single_stage_seed_iota15/single_stage_jax_runtime_spec.json \
  benchmarks/hf_jobs/cuda_pytest_probe.py \
  benchmarks/hf_jobs/launch_production_gpu_proof.py \
  benchmarks/hf_jobs/run_production_gpu_proof.sh \
  benchmarks/single_stage_init_parity.py \
  benchmarks/stage2_e2e_comparison.py \
  benchmarks/validation_ladder_common.py \
  tests/subprocess/hf_production_gpu_fake_runner.py \
  tests/test_benchmark_helpers.py \
  tests/test_hf_production_gpu_proof.py \
  docs/banana_jax_full_test_parity_coverage_impl_plan_2026-05-06.md \
  docs/banana_jax_native_port_todos_2026-05-05.md \
  examples/single_stage_optimization/STAGE_2/banana_coil_solver.py \
  src/simsopt/geo/_distance_jax.py \
  src/simsopt/geo/curveobjectives.py \
  src/simsopt/objectives/stage2_target_objective_jax.py \
  tests/integration/test_stage2_jax.py \
  tests/objectives/test_integral_bdotn_jax.py
```

If the validation-SHA scope is explicitly approved, stage only the same path
list and then verify with `git diff --cached --name-status` before committing.

## Completion Audit

Audit date: 2026-05-06.

Objective: execute and implement this plan file. Completion requires the
local Set A banana-required rows, the implemented Set B full-upstream
surface/objective rows, the manifest/doc status update, and the required M7
current-SHA CUDA evidence if CUDA closure is being claimed.

| Deliverable / gate | Concrete artifact evidence | Audit status |
| --- | --- | --- |
| Set A scope split and manifest reflection | `docs/jax_parity_manifest.md` links this plan, separates CPU/JAX from CUDA, and the manifest guard passes. | complete for non-CUDA scope; CUDA rows remain open |
| Field/flux banana lanes | `tests/objectives/test_fluxobjective_jax_parity.py` and `tests/objectives/test_integral_bdotn_jax.py` are part of the passing banana CPU/JAX gate. | complete for local CPU/JAX |
| RZ banana geometry/spec path | `src/simsopt/jax_core/surface_rzfourier.py`, `src/simsopt/geo/surfacerzfourier.py`, and `tests/geo/test_surface_rzfourier_jax.py`; full RZ/non-RZ surface parity gate passed. | complete for local CPU/JAX |
| `SurfaceXYZTensorFourier` single-stage seed geometry | `src/simsopt/jax_core/surface_fourier.py`, `src/simsopt/geo/surface_fourier_jax.py`, `tests/geo/test_surface_fourier_jax.py`, and benchmark fixture tests; full non-RZ file passed `123 passed`. | complete for local CPU/JAX |
| Surface objective wrappers and helpers | `src/simsopt/geo/surfaceobjectives_jax.py` and `tests/geo/test_surface_objectives_jax.py`; full JAX objective file passed `216 passed, 27 skipped`. | complete for local CPU/JAX |
| Set B RZ/non-RZ geometry, metric, curvature, Hessian, and object API breadth | RZ/non-RZ source and tests listed in the File Ownership Map; focused d2normal gate passed `4 passed, 119 deselected`, surface Taylor passed `19 passed, 142 subtests passed`. | complete for implemented non-CUDA rows |
| Set B conditional I/O/label/higher paired-point rows | VTK/file-output, `aspect_ratio` Boozer label, and higher `*_lin` APIs are explicitly conditional in B4/B5/B7. | not claimed; not a blocker for implemented rows |
| Manifest/doc update | `docs/jax_parity_manifest.md` has a documentary non-CUDA surface/objective section and banana inventory rows with CUDA still open where required. | complete for docs; manifest guard passed |
| Guardrails | `git diff --check` is clean; touched source/test diff grep found no dynamic imports, `typing.cast`, `Any`, or new `try`/`except`. | complete |
| M7 current-SHA CUDA artifact gate | Validation tag `banana-surface-parity-m7-unitnormal-r1` makes `0b6293b075342acc5cf996160ecf4bd87f709610` reachable; the GHCR image `ghcr.io/jungdaesuh/simsopt-jax-hf-production-proof:banana-surface-parity-m7-image-r1` is pullable with digest `sha256:eac2e1887eaf08628af62b28e5d7d7141b84afdfdcbfd00179823b1eb8f3df39`; real-image dry-run preflight passes remote SHA/ref plus seed-path checks. The real HF H200 launch reaches Hugging Face with the local CLI shim and then fails before job creation with HTTP 402 Payment Required, and Runpod has no active pods with a negative client balance. | incomplete and blocked on external GPU credits plus real H200 run |

Audit verdict: not complete for CUDA/P5 closure. The local non-CUDA
implementation and documentation work is complete, but the objective cannot be
marked achieved until the M7 current-SHA CUDA artifacts are produced or the
scope is explicitly changed to non-CUDA only.

## Requirement Set A: Banana-Required Closure

### A0 Scope Lock

- [x] Treat existing SIMSOPT C++/SciPy behavior as the oracle.
- [x] Require same-state C++/SciPy -> JAX CPU parity before interpreting optimizer behavior.
- [x] Require JAX CPU -> JAX CUDA and CPU/GPU agreement before marking CUDA rows complete.
- [x] Keep JAX-vs-JAX agreement insufficient by itself.
- [x] Keep `BiotSavartJAX`, fixed-surface flux, Stage 2 target bundle, Boozer/single-stage objectives, and consumed surface specs as the banana product surface.
- [x] Link this Set A/Set B scope split from the `docs/jax_parity_manifest.md` preamble.
- [x] Keep this scope decision reflected in `docs/jax_parity_manifest.md` after every closure PR.
- [x] Keep conditional full-upstream surface/API rows out of banana blockers unless the banana product path starts loading those APIs directly.

### A1 Field And Flux Lanes

Files:

- `src/simsopt/jax_core/biotsavart.py`
- `src/simsopt/field/biotsavart_jax_backend.py`
- `src/simsopt/objectives/fluxobjective_jax.py`
- `src/simsopt/jax_core/objectives_flux.py`

Required state:

- [x] Keep Biot-Savart value and derivative parity exact for banana-required kernels.
- [x] Keep `BiotSavartJAX`, `SpecBackedBiotSavartJAX`, and `SingleStageRuntimeSpecBiotSavartJAX` contract-complete for required field paths.
- [x] Keep `SquaredFluxJAX` and fixed-surface flux kernels contract-complete for Stage 2.
- [x] Keep CPU/C++ precision checks separate from optimizer trace diagnostics.
- [x] Preserve existing tolerances from the validation ladder; do not loosen tolerances to hide drift.
- [x] Treat JSON/getter breadth and raw-kernel Taylor polish as full-repo parity polish unless the banana manifest promotes them.

Acceptance:

- [x] `tests/objectives/test_fluxobjective_jax_parity.py`
- [x] `tests/objectives/test_integral_bdotn_jax.py`
- [ ] Stage 2 fixed-state value and gradient artifacts when claiming P5 CUDA closure.

### A2 SurfaceRZFourier Banana Geometry/Spec Path

Files:

- `src/simsopt/jax_core/surface_rzfourier.py`
- `src/simsopt/geo/surfacerzfourier.py`
- `src/simsopt/geo/surface.py`
- `src/simsopt/jax_core/specs.py`

Banana-required functions:

- [x] `gamma`
- [x] `gammadash1`
- [x] `gammadash2`
- [x] `normal`
- [x] `unitnormal`
- [x] `area`
- [x] `volume`
- [x] `darea_by_dcoeff`
- [x] `dvolume_by_dcoeff`
- [x] DOF roundtrip from mutable surface object to immutable spec.
- [x] Spec roundtrip from immutable spec to JAX kernels.
- [x] Loaders used by Stage 2 and single-stage workflows.

Required maintenance tasks:

- [x] Preserve CPU/C++ DOF order exactly.
- [x] Keep RZ mutable-wrapper methods thin: snapshot state into a spec and call kernel functions.
- [x] Keep new math in `src/simsopt/jax_core/surface_rzfourier.py`, not in objective wrappers.
- [x] Keep `SurfaceRZFourierSpec` immutable and pytree-compatible in `src/simsopt/jax_core/specs.py`.
- [ ] Add a banana-focused regression test if a Stage 2 or single-stage artifact loader starts consuming any new RZ host utility.

Acceptance:

- [x] `tests/geo/test_surface_rzfourier_jax.py`
- [ ] Any banana Stage 2/single-stage artifact loader tests that consume RZ specs.

### A3 SurfaceXYZTensorFourier Support Consumed By Single-Stage

Files:

- `src/simsopt/geo/surface_fourier_jax.py`
- `src/simsopt/jax_core/surface_fourier.py`
- `src/simsopt/geo/surfacexyztensorfourier.py`
- `src/simsopt/jax_core/specs.py`

Banana-required state:

- [x] `SurfaceXYZTensorFourier.surface_spec()` / `to_spec()` builds `SurfaceXYZTensorFourierSpec` for unclamped tensor surfaces.
- [x] `SurfaceXYZTensorFourier.surface_spec()` rejects `clamped_dims`.
- [x] Single-stage JAX runtime seed payloads use the canonical `"SurfaceXYZTensorFourier"` surface class.
- [x] Single-stage JAX runtime seed loading rejects non-`SurfaceXYZTensorFourier` surface classes.
- [x] JAX spec wrappers cover `gamma`, `gammadash1`, `gammadash2`, and `normal` for `SurfaceXYZTensorFourierSpec`.
- [x] Add explicit acceptance coverage that tensor `clamped_dims=True` remains rejected for JAX specs.
- [x] Strengthen banana-required spec parity for `SurfaceXYZTensorFourierSpec` across:
  - [x] `gamma`
  - [x] `gammadash1`
  - [x] `gammadash2`
  - [x] `normal`
  - [x] stellsym true/false where applicable
  - [x] `nfp > 1`
  - [x] nontrivial `mpol` and `ntor`
  - [x] nondefault quadrature point ranges
- [x] Add `SurfaceXYZTensorFourierSpec` area/volume parity against CPU for banana diagnostics/output.
- [x] Add artifact/load-spec acceptance: a legacy or JAX-emitted single-stage tensor surface artifact loads into immutable specs and evaluates `gamma`/`normal` without compiled target code calling host `surface.gamma()`.
- [x] Add single-stage surface-distance/self-intersection acceptance that the JAX tensor point cloud equals CPU geometry on the exact banana workflow grid.
- [x] Keep `SurfaceXYZFourierSpec` outside banana-required acceptance; full-upstream coverage is tracked in B5.

Acceptance:

- [x] `tests/geo/test_surface_fourier_jax.py`
- [x] Single-stage artifact/spec loader tests.
- [x] Single-stage surface-distance/self-intersection parity tests if those checks are claimed in the product path.

### A4 Single-Stage Objective Wrappers Used By Banana

Files:

- `src/simsopt/geo/surfaceobjectives_jax.py`
- `src/simsopt/geo/boozer_residual_jax.py`
- `src/simsopt/geo/boozersurface_jax.py`

Banana-required wrappers:

- [x] `BoozerResidualJAX`
- [x] `IotasJAX`
- [x] `NonQuasiSymmetricRatioJAX`
- [x] Traceable runtime objective bundle.
- [x] `ToroidalFlux` where used.

Required maintenance tasks:

- [x] Route objective geometry through the existing `surface_kind` dispatch.
- [x] Preserve exact Boozer solve support scope; do not expand `SurfaceXYZTensorFourier` exact-solve support as a side effect of surface parity work.
- [x] Keep host materialization only in named host wrappers.
- [x] Do not add `jax.pure_callback` bridges to compiled objective paths.
- [x] Keep Stage 2 target-lane reporting pure through `Stage2TargetObjectiveBundle.reporting_summary`.

Not banana blockers today:

- [x] `MajorRadiusJAX`
- [x] `PrincipalCurvatureJAX`
- [x] `QfmResidualJAX`
- [x] `AspectRatioJAX`

Acceptance:

- [x] `tests/geo/test_surface_objectives_jax.py`
- [x] `tests/geo/test_boozersurface_jax.py`
- [x] `tests/geo/test_boozer_derivatives_jax.py`
- [x] `tests/integration/test_single_stage_jax_cpu_reference.py`

### A5 Current-SHA CUDA Artifact Gate

This is required for banana P5 CUDA closure, but it is not the same thing as full upstream surface parity.

- [ ] Do not mark CUDA rows complete without real CUDA artifacts from the current pushed SHA.
- [ ] Record git SHA and dirty-tree status.
- [ ] Record command line and environment.
- [ ] Record Python, JAX, CUDA, driver, device, x64, and XLA metadata.
- [ ] Record host RSS and GPU memory telemetry where available.
- [ ] Preserve pass/fail reason and artifact path in the manifest.
- [x] Keep CPU-only local proof labeled as CPU evidence only.
- [x] Publish a CUDA-capable proof image and verify the tag is pullable.
- [x] Run real-image H200 launcher dry-run against the pushed validation SHA.

Required CUDA rows if still open in the manifest:

- [ ] Stage 2 fixed-state value.
- [ ] Stage 2 fixed-state gradient.
- [ ] Stage 2 reduced end-to-end strict run.
- [ ] Single-stage initialization.
- [ ] Boozer well-conditioned adjoint.
- [ ] CPU/GPU reduction stress.

## Requirement Set B: Full Legacy/Upstream Coverage

Set B is the real requirement list if the goal is to remove the "partial" label from the listed surface-family files completely. This is a broader surface/objective port, not a narrow banana validation task.

### B0 Cross-Cutting Architecture Review Checklist

These items are PR review checks unless a later implementation adds an explicit lint, grep, or CI owner for them.

- [x] Keep all pure surface math in `src/simsopt/jax_core/*` or the existing JAX surface modules.
- [x] Keep mutable Python object wrappers thin.
- [x] Preserve CPU/reference behavior unchanged.
- [x] Preserve immutable spec constructors as the SSOT for JAX surface state.
- [x] Preserve DOF ordering, stellsym skipped modes, and coefficient layout exactly.
- [x] Do not introduce dynamic imports.
- [x] Do not introduce `Any` casts or `typing.cast`.
- [x] Do not add defensive try/except fallbacks.
- [x] Do not auto-convert host inputs inside JIT/runtime boundaries.
- [x] Do not introduce callback bridges in compiled paths.
- [x] Avoid naive production Hessians that allocate avoidable `O(ndofs^2)` intermediates outside tests or explicit Hessian APIs.
- [x] Keep singular scalar-metric tests away from zero-area/zero-volume cases unless the CPU oracle explicitly defines those limits.

### B1 RZ Second-Order Geometry Core

Files:

- `src/simsopt/jax_core/surface_rzfourier.py`
- `src/simsopt/geo/surfacerzfourier.py`
- `tests/geo/test_surface_rzfourier_jax.py`
- `tests/geo/test_surface_taylor.py`

Implementation:

- [x] Add direct JAX spec/from-dofs function for `gammadash1dash1`.
- [x] Add direct JAX spec/from-dofs function for `gammadash1dash2`.
- [x] Add direct JAX spec/from-dofs function for `gammadash2dash2`.
- [x] Include Cartesian frame derivative terms, not only Fourier `R/Z` mode derivatives.
- [x] Add optional fused second-geometry output only if it removes real duplicated work. No fused output was added because the scalar APIs avoid product-path duplication.
- [x] Add coefficient Jacobians/VJPs for `gammadash1dash1`.
- [x] Add coefficient Jacobians/VJPs for `gammadash1dash2`.
- [x] Add coefficient Jacobians/VJPs for `gammadash2dash2`.
- [x] Add thin public `_jax` wrappers in `src/simsopt/geo/surfacerzfourier.py` only after kernel tests pass.
- [x] Export new functions from `src/simsopt/jax_core/__init__.py` if existing export conventions require it.

Tests:

- [x] Value parity against legacy `SurfaceRZFourier.gammadash1dash1()`.
- [x] Value parity against legacy `SurfaceRZFourier.gammadash1dash2()`.
- [x] Value parity against legacy `SurfaceRZFourier.gammadash2dash2()`.
- [x] Coefficient-derivative parity against legacy `dgammadash*_by_dcoeff()` methods.
- [x] Taylor tests mirroring `tests/geo/test_surface_taylor.py`.
- [x] stellsym true coverage.
- [x] stellsym false coverage.
- [x] `nfp > 1` coverage.
- [x] Nondefault quadrature point coverage.
- [ ] HLO/transfer-guard smoke if a fused kernel is added.

### B2 RZ Forms And Curvatures

Files:

- `src/simsopt/jax_core/surface_rzfourier.py`
- `src/simsopt/geo/surfacerzfourier.py`
- `src/simsoptpp/surface.h`
- `src/simsoptpp/surface.cpp`
- `src/simsoptpp/python_surfaces.cpp`
- `tests/geo/test_surface_rzfourier_jax.py`
- `tests/geo/test_surface_taylor.py`
- `tests/geo/test_surface.py`

Implementation:

- [x] Add `first_fund_form` with legacy ordering `[E, F, G]`.
- [x] Add `second_fund_form` with legacy ordering `[L, M, N]`.
- [x] Add `surface_curvatures` with legacy ordering `[H, K, kappa1, kappa2]`.
- [x] Add `dfirst_fund_form_by_dcoeff` if parity claim includes form derivatives.
- [x] Add `dsecond_fund_form_by_dcoeff` if parity claim includes form derivatives.
- [x] Add `dsurface_curvatures_by_dcoeff`.
- [x] Preserve normal orientation from `gammadash1 x gammadash2`.
- [x] Document the sign convention in comments only where the formula is otherwise ambiguous.

Tests:

- [x] Value parity for first fundamental form.
- [x] Value parity for second fundamental form.
- [x] Value parity for surface curvatures.
- [x] Derivative parity for form derivatives if implemented.
- [x] Derivative parity for `dsurface_curvatures_by_dcoeff`.
- [x] Taylor finite-difference checks.
- [x] Gauss-Bonnet style coverage against upstream `tests/geo/test_surface.py`.
- [x] Curvature sign regression on at least one nontrivial non-stellsym surface.

### B3 RZ Scalar Metric Hessians

Files:

- `src/simsopt/jax_core/surface_rzfourier.py`
- `src/simsopt/geo/surfacerzfourier.py`
- `src/simsoptpp/surface.h`
- `src/simsoptpp/surface.cpp`
- `src/simsoptpp/python_surfaces.cpp`
- `tests/geo/test_surface_rzfourier_jax.py`
- `tests/geo/test_surface_taylor.py`
- `tests/geo/test_surface_rzfourier.py`

Implementation:

- [x] Add `mean_cross_sectional_area` JAX scalar helper.
- [x] Add `minor_radius` JAX scalar helper.
- [x] Add `major_radius` JAX scalar helper.
- [x] Add `aspect_ratio` JAX scalar helper.
- [x] Add JAX `d2area_by_dcoeffdcoeff` mirroring the C++/pybind CPU oracle.
- [x] Add JAX `d2volume_by_dcoeffdcoeff` mirroring the C++/pybind CPU oracle.
- [x] Add `d2minor_radius_by_dcoeff_dcoeff`.
- [x] Add `d2major_radius_by_dcoeff_dcoeff`.
- [x] Add `d2aspect_ratio_by_dcoeff_dcoeff`.
- [x] Keep Hessian APIs explicit so production paths do not allocate Hessians accidentally.

Tests:

- [x] CPU/JAX value parity for `mean_cross_sectional_area`.
- [x] CPU/JAX value parity for `minor_radius`.
- [x] CPU/JAX value parity for `major_radius`.
- [x] CPU/JAX value parity for `aspect_ratio`.
- [x] Gradient parity for each scalar metric.
- [x] Hessian parity for each scalar metric with upstream tolerances.
- [x] Second-order Taylor tests.
- [x] Area Hessian parity against the C++/pybind CPU oracle.
- [x] Volume Hessian parity against the C++/pybind CPU oracle.
- [x] Tests avoid near-zero singular cases unless explicitly testing CPU-defined behavior.

### B4 Broader SurfaceRZFourier Host API Behavior

Files:

- `src/simsopt/geo/surfacerzfourier.py`
- `src/simsopt/jax_core/surface_rzfourier.py`
- `tests/geo/test_surface_rzfourier_jax.py`
- `tests/geo/test_surface_rzfourier.py`

Implementation and coverage:

- [x] Verify `from_focus` output can produce a JAX spec and match CPU geometry.
- [x] Verify `from_pyQSC` output can produce a JAX spec and match CPU geometry.
- [x] Verify `make_rotating_ellipse` output can produce a JAX spec and match CPU geometry.
- [x] Verify `change_resolution` preserves JAX spec roundtrip.
- [x] Verify `condense_spectrum` preserves JAX spec roundtrip.
- [x] Verify `extend_via_normal` preserves JAX spec roundtrip.
- [x] Verify `copy` and object-independence semantics for JAX spec snapshots.
- [ ] Add serialization/GSON roundtrip tests if I/O parity is claimed.
- [ ] Add `to_vtk` smoke/file-exists coverage only if I/O parity is claimed.
- [x] Treat optional dependency tests as skipped when the upstream CPU tests skip for missing optional dependencies.

Acceptance:

- [x] Existing CPU host tests still pass unchanged.
- [x] JAX spec tests prove the resulting surfaces evaluate the same `gamma`, tangents, normals, area, and volume as CPU.

### B5 Non-RZ Geometry And Derivative Parity

Files:

- `src/simsopt/geo/surface_fourier_jax.py`
- `src/simsopt/jax_core/surface_fourier.py`
- `src/simsopt/geo/surfacexyzfourier.py`
- `src/simsopt/geo/surfacexyztensorfourier.py`
- `tests/geo/test_surface_fourier_jax.py`
- `tests/geo/test_surface_taylor.py`

Implementation:

- [x] Add `SurfaceXYZFourier` coefficient derivative parity for `dgamma_by_dcoeff`.
- [x] Add `SurfaceXYZFourier` coefficient derivative parity for `dgammadash1_by_dcoeff`.
- [x] Add `SurfaceXYZFourier` coefficient derivative parity for `dgammadash2_by_dcoeff`.
- [x] Add `gammadash1dash1` for `SurfaceXYZFourierSpec`.
- [x] Add `gammadash1dash2` for `SurfaceXYZFourierSpec`.
- [x] Add `gammadash2dash2` for `SurfaceXYZFourierSpec`.
- [x] Add `gammadash1dash1` for `SurfaceXYZTensorFourierSpec`.
- [x] Add `gammadash1dash2` for `SurfaceXYZTensorFourierSpec`.
- [x] Add `gammadash2dash2` for `SurfaceXYZTensorFourierSpec`.
- [x] Add coefficient derivatives for second coordinate derivatives where upstream exposes them.
- [x] Add CPU parity tests for existing `gamma_lin` / `surface_gamma_lin_from_dofs` if paired-point APIs are in full-upstream scope.
- [x] Add `gammadash1_lin`.
- [x] Add `gammadash2_lin`.
- [ ] Add higher `*_lin` paired-point APIs only if full legacy parity explicitly includes them.
- [x] Add `unitnormal`.
- [x] Add `dnormal_by_dcoeff`.
- [x] Add `d2normal_by_dcoeffdcoeff` only as an explicit heavy API.
- [x] Add `dunitnormal_by_dcoeff`.
- [x] Add full-upstream non-RZ area/volume value APIs for surface families not owned by A3 tensor diagnostics.
- [x] Add `darea`, `d2area`, `dvolume`, and `d2volume` parity for full-upstream non-RZ scope.

Tests:

- [x] CPU/JAX parity for all new non-RZ coordinate derivatives.
- [x] CPU/JAX parity for coefficient derivatives.
- [x] First- and second-order Taylor tests.
- [x] stellsym true/false coverage where supported.
- [x] `nfp > 1` coverage.
- [x] nondefault quadrature coverage.
- [x] Tensor unclamped coverage.
- [x] Explicit rejection coverage for tensor `clamped_dims` unless full upstream scope decides to support it.

### B6 Non-RZ Object API Breadth

Files:

- `src/simsopt/geo/surface_fourier_jax.py`
- `src/simsopt/jax_core/surface_fourier.py`
- `src/simsopt/geo/surfacexyzfourier.py`
- `src/simsopt/geo/surfacexyztensorfourier.py`
- `tests/geo/test_surface_fourier_jax.py`
- `tests/geo/test_surface_xyzfourier.py`

Implementation and coverage:

- [x] Add copy/object-independence tests for `SurfaceXYZFourier`.
- [x] Add copy/object-independence tests for `SurfaceXYZTensorFourier`.
- [x] Add Python `copy`/`deepcopy` coverage if that object protocol is claimed.
- [x] Add direct JSON/GSON roundtrip tests for `SurfaceXYZFourier`.
- [x] Add direct JSON/GSON roundtrip tests for `SurfaceXYZTensorFourier`.
- [ ] Add VTK smoke/file-exists coverage for tensor surfaces if I/O parity is claimed.
- [x] Add object API tests for `to_RZFourier`.
- [x] Add object API tests for `cross_section`.
- [x] Add object API tests for `least_squares_fit`.
- [x] Add object API tests for `fit_to_curve`.
- [x] Add object API tests for `scale`.
- [x] Add object API tests for `extend_via_normal`.
- [x] Add object API tests for `extend_via_projected_normal`.
- [x] Fix or add the intended `test_surface_conversion` coverage if the current test body is exercising the wrong helper.

Acceptance:

- [x] `tests/geo/test_surface_fourier_jax.py`
- [x] `tests/geo/test_surface_xyzfourier.py`
- [x] `tests/geo/test_surface.py`
- [x] `tests/geo/test_surface_taylor.py`

### B7 Missing Surface Objective Wrappers

File:

- `src/simsopt/geo/surfaceobjectives_jax.py`

Related dependencies:

- `src/simsopt/jax_core/surface_rzfourier.py`
- `src/simsopt/geo/surface_fourier_jax.py`
- `src/simsopt/jax_core/surface_fourier.py`
- `src/simsopt/geo/boozersurface_jax.py`
- `tests/geo/test_surface_objectives_jax.py`
- `tests/geo/test_surface_objectives.py`

Implementation order:

- [x] Add shared pure-JAX scalar helpers first:
  - [x] `mean_cross_sectional_area`
  - [x] `minor_radius`
  - [x] `major_radius`
  - [x] `aspect_ratio`
  - [x] gradients for all four
  - [x] Hessians where upstream exposes them
- [x] Implement `AspectRatioJAX`.
- [x] Implement `MajorRadiusJAX`.
- [x] Implement `QfmResidualJAX`.
- [x] Implement `PrincipalCurvatureJAX` last, after curvature kernels exist.
- [x] Export new wrappers through `surfaceobjectives_jax.__all__`.
- [x] Add import/lazy-access smoke coverage through `simsopt.geo` if existing conventions require it.
- [ ] Add `aspect_ratio` label support to `BoozerSurfaceJAX` only if full label-test parity is in scope.

`AspectRatioJAX` requirements:

- [x] Mirror upstream value behavior.
- [x] Mirror `dJ`.
- [x] Mirror `dJ_by_dsurfacecoefficients`.
- [x] Mirror `d2J_by_dsurfacecoefficientsdsurfacecoefficients`.
- [x] Test CPU/JAX value parity.
- [x] Test surface-gradient parity.
- [x] Test Hessian parity.
- [x] Test first-order Taylor.
- [x] Test second-order Taylor.
- [x] Cover `SurfaceRZFourier`.
- [x] Cover `SurfaceXYZFourier`.
- [x] Cover `SurfaceXYZTensorFourier`.
- [x] Cover stellsym true/false where supported.

`MajorRadiusJAX` requirements:

- [x] Reuse the existing Boozer objective base if it fits the derivative contract.
- [x] Value is solved-surface major radius.
- [x] Direct coil gradient is zero.
- [x] Adjoint RHS is the major-radius surface gradient padded into `[surface_dofs, iota, G]`.
- [x] Test value parity vs CPU `MajorRadius`.
- [x] Test public `Derivative` projection parity.
- [x] Test re-solve directional Taylor/finite difference with respect to coil DOFs for LS where supported.
- [x] Test exact solve only where `BoozerSurfaceJAX` supports the surface family.

`QfmResidualJAX` requirements:

- [x] Implement pure scalar `qfm_residual` from surface DOFs, coil-set spec, and surface metadata.
- [x] Use `jax.grad` with respect to surface DOFs for the surface derivative.
- [x] Use existing `BiotSavartJAX` field and pullback APIs.
- [x] Test value parity vs CPU `QfmResidual`.
- [x] Test surface-gradient parity.
- [x] Test first-order Taylor with respect to surface DOFs.
- [x] Test cache/update behavior when surface DOFs change.

`PrincipalCurvatureJAX` requirements:

- [x] Depend on `surface_curvatures`.
- [x] Depend on `dsurface_curvatures_by_dcoeff`.
- [x] Test value parity vs CPU `PrincipalCurvature`.
- [x] Test surface-gradient parity.
- [x] Test first-order Taylor.
- [x] Do not add Hessian tests unless upstream exposes a Hessian contract.

Banana classification:

- [x] Keep `MajorRadiusJAX` classified as banana-adjacent but not a current banana blocker.
- [x] Keep `AspectRatioJAX` classified as upstream parity backlog unless it becomes a JAX Boozer label or QFM constraint in the product path.
- [x] Keep `QfmResidualJAX` classified as upstream/QFM workflow backlog unless product scope changes.
- [x] Keep `PrincipalCurvatureJAX` classified as upstream parity backlog; banana currently uses curve curvature, not surface principal curvature.

## File Ownership Map

Production files:

- [x] `src/simsopt/jax_core/surface_rzfourier.py`: RZ pure JAX kernels, derivative kernels, forms, curvatures, scalar metrics, explicit Hessians.
- [x] `src/simsopt/geo/surfacerzfourier.py`: thin RZ object wrappers and spec snapshot access only.
- [x] `src/simsopt/geo/surface.py`: CPU/reference base API remains unchanged unless a pure wrapper needs a documented parity hook.
- [x] `src/simsoptpp/surface.h`, `src/simsoptpp/surface.cpp`, `src/simsoptpp/python_surfaces.cpp`: C++/pybind CPU oracle for forms, curvatures, area/volume Hessians, and derivative-heavy surface parity.
- [x] `src/simsopt/geo/surface_fourier_jax.py`: non-RZ pure JAX geometry and derivative primitives.
- [x] `src/simsopt/jax_core/surface_fourier.py`: immutable non-RZ spec wrappers.
- [x] `src/simsopt/jax_core/specs.py`: immutable specs only when new state is actually required.
- [x] `src/simsopt/jax_core/__init__.py`: exports only after kernel APIs are stable.
- [x] `src/simsopt/geo/surfaceobjectives_jax.py`: missing objective wrappers and objective-specific plumbing only.
- [x] `src/simsopt/geo/boozersurface_jax.py`: label support only if full label parity is in scope.

Test files:

- [x] `tests/geo/test_surface_rzfourier_jax.py`: RZ JAX parity, transfer guards, spec tests.
- [x] `tests/geo/test_surface_fourier_jax.py`: non-RZ JAX parity, spec tests.
- [x] `tests/geo/test_surface_objectives_jax.py`: JAX objective wrappers.
- [x] `tests/geo/test_surface_taylor.py`: CPU oracle/Taylor reference stays authoritative.
- [x] `tests/geo/test_surface_rzfourier.py`: CPU host/API oracle stays authoritative.
- [x] `tests/geo/test_surface.py`: base surface oracle and Gauss-Bonnet/form coverage.
- [x] `tests/geo/test_surface_objectives.py`: CPU objective oracle stays authoritative.
- [x] `tests/docs/test_banana_parity_coverage_manifest.py`: manifest status validation after evidence exists.

## Milestone Order

The sequence below is safe for one engineer. For parallel implementation, M4 and M5 can start after M0/M1, M3 starts after M2, and M6 can split by dependency: `QfmResidualJAX` after existing field/pullback contracts, `AspectRatioJAX` / `MajorRadiusJAX` after scalar metric helpers, and `PrincipalCurvatureJAX` after curvature kernels.

### M0 Scope And Baseline

- [x] Freeze the Set A vs Set B scope split in this file.
- [x] Confirm `docs/jax_parity_manifest.md` still reflects banana rows accurately.
- [x] Confirm local interpreter and x64 settings before running parity tests.
- [x] Confirm no source edits are needed for banana Set A unless current tests fail.

### M1 Banana Non-CUDA Acceptance Tightening

- [x] Add tensor `clamped_dims` rejection test.
- [x] Strengthen `SurfaceXYZTensorFourierSpec` parity across banana-relevant grids.
- [x] Add tensor area/volume spec-level parity if banana diagnostics depend on it.
- [x] Add artifact/load-spec acceptance for single-stage tensor surfaces.
- [x] Add self-intersection/surface-distance point-cloud equality if claimed in the product path.
- [x] Run banana CPU/JAX gates.
- [x] Update manifest only for evidence-backed rows.

### M2 RZ Full Legacy Geometry

- [x] Implement RZ second-order coordinate derivatives.
- [x] Implement RZ coefficient derivatives/VJPs for second-order geometry.
- [x] Add RZ Taylor and CPU parity tests.
- [x] Run RZ JAX and CPU oracle tests.

### M3 RZ Forms, Curvatures, Metrics

- [x] Implement fundamental forms.
- [x] Implement surface curvatures.
- [x] Implement curvature derivatives.
- [x] Implement scalar metric helpers.
- [x] Implement explicit scalar metric Hessians.
- [x] Add parity, derivative, Taylor, and Gauss-Bonnet tests.

### M4 Non-RZ Full Geometry

- [x] Implement non-RZ second coordinate derivatives.
- [x] Implement non-RZ second-coordinate coefficient derivatives.
- [x] Implement remaining non-RZ coordinate coefficient derivatives.
- [x] Implement non-RZ normal/unitnormal derivative APIs.
- [x] Implement non-RZ area/volume derivative APIs.
- [x] Add parity tests.
- [x] Add Taylor tests.

### M5 Host/Object API Breadth

- [x] Add RZ host utility/spec roundtrip coverage.
- [x] Add non-RZ copy/I/O/object API breadth coverage for `copy`, `copy.deepcopy`, JSON/GSON, `to_RZFourier`, `cross_section`, `least_squares_fit`, `fit_to_curve`, `scale`, and normal-extension APIs.
- [x] Keep VTK as an explicit unclosed backlog row unless claimed.
- [x] Keep optional dependency skips aligned with CPU tests.
- [x] Keep CPU behavior unchanged.

### M6 Missing Objective Wrappers

- [x] Implement shared scalar helpers.
- [x] Implement `AspectRatioJAX`.
- [x] Implement `MajorRadiusJAX`.
- [x] Implement `QfmResidualJAX`.
- [x] Implement `PrincipalCurvatureJAX`.
- [x] Add CPU/JAX value, derivative, and Taylor tests (`AspectRatioJAX`, `QfmResidualJAX`, `PrincipalCurvatureJAX`, and `MajorRadiusJAX` are covered; `MajorRadiusJAX` includes native adjoint, public projection, LS re-solve, and exact tensor-runtime finite-difference coverage).
- [x] Update exports/import smoke tests.

### M7 CUDA And Documentation Evidence

- [ ] Run current-sha CUDA artifacts for banana P5 rows if banana CUDA closure is the goal.
- [ ] Attach artifact metadata to the manifest or linked proof doc.
- [x] Build and publish the CUDA proof image from the repo-owned Dockerfile.
- [x] Verify launcher preflight with the published image, pushed tag, and repo-owned runtime seed fixture.
- [x] Update `docs/jax_parity_manifest.md`.
- [x] Update the existing banana coverage plan for this local non-CUDA status refresh.
- [x] Keep Set B backlog rows separate from banana blockers.

## Validation Commands

Use the repo-local interpreter when available:

```bash
cd /Users/suhjungdae/code/columbia/simsopt-jax
export PY=/Users/suhjungdae/code/columbia/simsopt-jax/.conda/jax-0.9.2/bin/python
export PYTHONPATH=/Users/suhjungdae/code/columbia/simsopt-jax/src
export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cpu
```

Banana manifest and harness gate:

```bash
$PY -m pytest -q \
  tests/docs/test_banana_parity_coverage_manifest.py \
  tests/test_hf_production_gpu_proof.py \
  tests/test_benchmark_helpers.py::test_single_stage_init_fixture_files_are_vendored \
  tests/test_benchmark_helpers.py::test_single_stage_init_fixture_runtime_seed_spec_loads
```

Backend / smoke / native-path gate:

```bash
$PY -m pytest -q \
  tests/test_backend.py \
  tests/test_jax_import_smoke.py \
  tests/integration/test_jax_native_path.py
```

Banana CPU/JAX parity gate:

```bash
$PY -m pytest -q \
  tests/objectives/test_fluxobjective_jax_parity.py \
  tests/objectives/test_integral_bdotn_jax.py \
  tests/geo/test_surface_rzfourier_jax.py \
  tests/geo/test_surface_fourier_jax.py \
  tests/geo/test_surface_objectives_jax.py \
  tests/integration/test_stage2_target_lane_purity.py \
  tests/integration/test_stage2_jax.py
```

Boozer focused wrapper gate:

```bash
$PY -m pytest -q \
  tests/geo/test_boozersurface_jax.py \
  tests/geo/test_boozer_derivatives_jax.py \
  -k "boozer or Boozer"
```

Single-stage CPU reference closure gate:

```bash
$PY -m pytest -q \
  tests/integration/test_single_stage_jax_cpu_reference.py
```

Full RZ surface parity gate:

```bash
$PY -m pytest -q \
  tests/geo/test_surface_rzfourier_jax.py \
  tests/geo/test_surface_rzfourier.py \
  tests/geo/test_surface_taylor.py \
  tests/geo/test_surface.py
```

Full non-RZ surface parity gate:

```bash
$PY -m pytest -q \
  tests/geo/test_surface_fourier_jax.py \
  tests/geo/test_surface.py \
  tests/geo/test_surface_taylor.py \
  tests/geo/test_surface_xyzfourier.py
```

Surface objective wrapper gate:

```bash
$PY -m pytest -q \
  tests/geo/test_surface_objectives_jax.py \
  tests/geo/test_surface_objectives.py \
  -k "ToroidalFlux or MajorRadius or PrincipalCurvature or QfmResidual or AspectRatio"
```

Release-gate unit/schema checks:

```bash
$PY -m pytest -q \
  tests/test_single_stage_cpp_jax_state_parity.py \
  tests/integration/test_single_stage_dof_mapping.py \
  tests/test_benchmark_helpers.py \
  -k "release_gate or fixed_state or coordinate_mapping or single_stage_parity_matrix"
```

Local CPU/C++ fixed-state and mapping artifacts:

```bash
mkdir -p .artifacts/parity
$PY benchmarks/single_stage_dof_mapping_proof.py \
  --output-json .artifacts/parity/coordinate-mapping-proof.json
$PY benchmarks/single_stage_cpp_jax_state_parity.py \
  --platform cpu \
  --output-json .artifacts/parity/fixed-state-cpu.json
```

Matrix gate after a same-seed run report exists:

```bash
$PY benchmarks/single_stage_parity_matrix.py \
  --fixed-state-parity-json .artifacts/parity/fixed-state-cpu.json \
  --coordinate-mapping-json .artifacts/parity/coordinate-mapping-proof.json \
  --parity-report-json <merged-same-seed-report.json> \
  --output-json .artifacts/parity/release-matrix-cpu.json \
  --output-md .artifacts/parity/release-matrix-cpu.md
```

Optional CUDA/H200 gate only after commit, pushed validation ref, and image:

```bash
SIMSOPT_HF_GPU_IMAGE=<registry>/simsopt-jax:cuda12-jax092 \
$PY benchmarks/hf_jobs/launch_production_gpu_proof.py \
  --repo-url https://github.com/jungdaesuh/simsopt.git \
  --repo-ref <pushed-validation-ref> \
  --repo-sha <pushed-current-sha> \
  --hardware h200 \
  --platform cuda \
  --single-stage-mpol 10 \
  --single-stage-ntor 10 \
  --single-stage-jax-runtime-seed-spec benchmarks/fixtures/single_stage_seed_iota15 \
  --no-detach
```

## Review Checklist And Enforced Gates

Checklist items below are human review requirements unless an automated owner is named.

- [x] Every new JAX value API has CPU oracle parity tests.
- [x] Every new JAX derivative API has CPU oracle derivative parity or a Taylor test.
- [x] Every Hessian API has explicit Hessian parity or second-order Taylor coverage.
- [ ] Every product-path CUDA claim has a current-sha CUDA artifact.
- [x] No manifest row is marked complete from CPU-only evidence when CUDA evidence is required.
- [x] No tolerance changes are made without updating the validation ladder contract and explaining why.
- [x] No broad host API/I/O parity is treated as banana-required unless a banana workflow directly consumes it.
- [x] Dirty unrelated files remain untouched during implementation.
- [x] `tests/docs/test_banana_parity_coverage_manifest.py` is wired into `.github/workflows/jax_smoke.yml` so manifest status edits run the machine-checkable banana inventory guard.

## Recommended Scope Decision

Current local execution followed the full-upstream non-CUDA Set B path. CUDA
P5 closure remains separate and open.

- [ ] If the goal is banana ship readiness, execute Set A and P5 artifact closure only. Do not implement Set B now.
- [x] If the goal is zero-gap JAX-vs-C++/Python surface parity, execute Set B in milestones M2 through M6.
- [ ] Keep Set B as a full-upstream parity backlog until the product requirement changes.
