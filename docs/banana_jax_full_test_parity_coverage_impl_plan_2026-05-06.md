# Banana JAX Parity Coverage Closure Plan

Status: implementation in progress as of 2026-05-06; this is not a full
parity completion record.

Initial tree inspected at `f59a85ab4`. The working tree contained many
untracked artifacts when this plan was written. This document is additive and
does not judge or stage those artifacts. The continuation audit below records
the current implementation state separately.

This file is a scoped closure plan and progress ledger for the banana JAX
parity surface. It records CPU/JAX implementation and test closure where those
lanes are complete, and it keeps the P5 real-CUDA artifact gate separate. Do
not read the title or checked CPU items as a claim that full test parity has
been achieved while the Definition of Done and P5 CUDA checkboxes remain open.

## Scope

This plan defines and tracks the remaining test-parity gaps for the banana
Stage 2 and single-stage JAX path. It is not a general promise to port every
upstream SIMSOPT class or every `simsoptpp` extension surface.

The product scope is:

- Stage 2 banana optimizer:
  `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py`
- Single-stage banana optimizer:
  `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
- Core JAX target lane used by those entrypoints:
  Biot-Savart, SquaredFlux, Stage 2 target bundle, BoozerSurfaceJAX,
  surface objectives, distance objectives, optimizer target wrappers, and
  immutable spec/restart artifacts.

The parity chain for any future full-coverage claim is:

1. Existing SIMSOPT C++/SciPy behavior is the oracle.
2. JAX CPU matches that oracle at the same fixed state.
3. JAX CUDA matches the same oracle when hardware parity is claimed.
4. JAX CPU and JAX CUDA match each other under the named tolerance lane.

JAX-vs-JAX agreement alone is not enough for release-grade parity.

Full-repo parity backlog boundary:

The broader seven-area JAX-vs-C++ audit identifies real full-repo parity gaps
such as unmigrated surface-objective wrappers (`MajorRadius`, `PrincipalCurvature`,
`QfmResidual`, `AspectRatio`), second-order surface tangents/forms/curvatures,
oriented/framed curve direct FD coverage, core helper unit tests, and
BiotSavart/SquaredFlux JSON or legacy getter polish. Those are valid backlog
items for zero-gap upstream parity, but they are not P5 banana CUDA blockers
unless the Stage 2 or single-stage banana product path starts consuming them.
The P5 blocker for this plan remains the real-CUDA artifact gate for the rows
marked `open under P5` in `docs/jax_parity_manifest.md`.

## Source Documents

- `docs/jax_parity_manifest.md`
- `docs/banana_jax_native_port_todos_2026-05-05.md`
- `docs/banana_cpp_cpu_dependency_manifest_2026-05-05.md`
- `docs/boozer_full_parity_plan_2026-05-04.md`
- `docs/using_jax_backend.md`
- `docs/banana_single_stage_stage2_lavish_validation_plan_2026-04-27.md`
- `benchmarks/validation_ladder_contract.py`

`benchmarks/validation_ladder_contract.py` remains the tolerance-lane SSOT.
This plan should reference those lanes instead of inventing informal tolerances.

## Current Coverage Baseline

The banana gradient hot path is already covered strongly enough that the
following are not open blockers for banana fixed-state value-and-gradient
coverage:

- Biot-Savart C++/Python parity.
- `CurveCWSFourierCPP` forward and VJP for the banana JAX path.
- `CurveCurveDistance` and `CurveSurfaceDistance` C++ culler public JAX path.
- Boozer CPU/JAX contract, including wrapper-gradient slices.

CUDA Boozer parity is still a separate hardware gate. The CPU Boozer closure
does not by itself prove GPU parity.

Single-stage parity does not need a separate implementation phase unless a
single-stage-only legacy re-entry site is found. Current single-stage closure is
delivered through the Boozer CPU contract rows, the `ToroidalFlux` value-matrix
completion in P3, the CUDA gate in P5, and the Stage 2 artifact/reporting
contracts in P4 where single-stage consumes Stage 2 outputs.

The old manifest `partial` rows should not be treated as four equal-sized
implementation gaps. Current code inspection narrows them to:

1. Flux / `integral_BdotN` manifest reconciliation. Regular direct C++ parity
   and documented degenerate `0.0` / `inf` contracts are already covered. The
   open issue is whether a real CPU-`nan` reproducer exists for the manifest's
   current wording.
2. Surface geometry scope clarification. `SurfaceRZFourier` object-API parity is
   mostly covered, while `SurfaceXYZFourier` and `SurfaceXYZTensorFourier` cover
   geometry/spec parity but not broad object I/O. That broader XYZ object API is
   not banana-required unless the scope expands to full legacy surface parity.
3. `ToroidalFlux` value matrix symmetry. Derivative, Hessian, coil derivative,
   and Taylor coverage are already parametrized broadly; the small missing test
   is value parity over the same surface-type and `stellsym` matrix.
4. Stage 2 snapshot/callback/reporting re-entry into the legacy `Optimizable`
   graph outside the gradient hot path. This is the only remaining item that is
   both missing implementation work and missing full-run strict-purity coverage.

## Refined Priority

1. Stage 2 snapshot/callback/reporting reroute. This is the real follow-up
   phase for the stronger "no legacy graph anywhere on the JAX lane" claim.
2. Flux / `integral_BdotN` manifest reconciliation. Find a CPU-`nan`
   reproducer or promote the stale wording.
3. `ToroidalFlux` value-parity matrix. Add one parametrized value test and keep
   tolerance-based parity as the contract.
4. Surface scope statement. Keep XYZ object API out of banana scope unless the
   project explicitly chooses full legacy surface parity.
5. CUDA Boozer hardware gate. This remains hardware-bound and separate from CPU
   parity closure.

## Phase Effort And Proof Lanes

These estimates prevent the bounded doc/test reconciliations from being treated
as equivalent to the Stage 2 reporting refactor.

| Phase | Effort | Proof lane / status rule |
| --- | --- | --- |
| P1 flux / `integral_BdotN` | Under 1 day, including at most 0.5 day of targeted edge-case search | `direct_kernel` for `integral_BdotN`; `ls_wrapper_gradient` for wrapper value/grad; row 34 remains `tier2_stage2_e2e` integration scope. |
| P2 surface scope | 0 days if banana scope is held; about 3 days per XYZ family if full legacy object API is later required | Documentation-only for banana; add a new object-API lane before any full-legacy expansion. |
| P3 `ToroidalFlux` value matrix | About 0.5 day | Extend `derivative_heavy` with scalar value tolerances or add a `toroidal_flux_value_matrix` lane in `validation_ladder_contract.py` before the test lands. |
| P4 Stage 2 reporting purity | 5-10 focused days | Add a `reporting_contract` lane for payload parity; use `tier2_stage2_e2e` only for the reduced Stage 2 run shape. |
| P5 CUDA hardware gate | Hardware-gated | `gpu_runtime` and `reduction_cpu_gpu`; CPU-only runs cannot close CUDA rows. |
| P6 coverage inventory | About 1 day after narrow closures | Manifest consistency enforcement; this is closeout infrastructure, not a prerequisite for P1/P3/P4 unless drift recurs. |

## Definition of Done

Current status: the non-CUDA definition-of-done items are closed by P1-P4 and
P6 evidence below. The overall plan remains incomplete until P5 captures real
CUDA artifacts for every `open under P5` row in `docs/jax_parity_manifest.md`.

- [x] Every banana-relevant upstream test has a direct JAX parity test,
  CPU-reference contract test, CUDA hardware-gated parity test, or explicit
  documented exclusion with a strict rejection test.
- [x] Every optimizer-consumed derivative has same-state CPU/JAX coverage:
  value, gradient, VJP, HVP, or documented FD/Taylor contract as appropriate.
- [x] Fixed-state parity tests are separate from optimizer-trace diagnostics.
- [x] Degenerate/singular behavior is either made identical or documented as
  intentional contract divergence with direct tests.
- [x] `docs/jax_parity_manifest.md` has no banana-relevant `partial` row unless
  the row is explicitly out of banana scope.
- [ ] Real CUDA artifacts exist for all `open under P5` rows, with git SHA,
  dirty-tree status, command, runtime, device, memory, and pass/fail metadata.

## P1: FluxObjective And Integral BdotN Manifest Reconciliation

Context:

`tests/objectives/test_fluxobjective.py` is mirrored by
`tests/objectives/test_fluxobjective_jax_parity.py`. Regular same-state
value/gradient parity is strong. The previous plan overstated the gap as a
settled `nan`-vs-stabilized policy decision. Current inspection of
`src/simsoptpp/integral_BdotN.cpp` shows the documented degenerate flow returns
`0.0` for zero-normal quadratic flux and `+inf` for normalized/local zero-field
singularities. The JAX tests already assert those same contracts.

Implementation touch points:

- `src/simsopt/objectives/fluxobjective.py`
- `src/simsopt/objectives/fluxobjective_jax.py`
- `src/simsopt/objectives/integral_bdotn_jax.py`
- `src/simsoptpp/integral_BdotN.cpp`
- `tests/objectives/test_fluxobjective.py`
- `tests/objectives/test_fluxobjective_jax_parity.py`
- `tests/objectives/test_integral_bdotn_jax.py`
- `tests/integration/test_stage2_jax.py`

Validated current coverage:

- [x] Direct `simsoptpp.integral_BdotN` vs JAX kernel parity exists for regular
  inputs across quadratic flux, normalized, and local definitions.
- [x] The C++ kernel's documented degenerate branches return `0.0` or `+inf`,
  not `nan`, for the covered zero-normal and zero-field flows.
- [x] JAX tests assert zero-normal quadratic flux returns `0.0`.
- [x] JAX tests assert normalized zero-field and local zero-field singularities
  return `+inf`.
- [x] CPU/JAX wrapper value and gradient parity exists for all three
  definitions where the wrapper contract is defined.
- [x] Target handling parity is covered for quadratic flux.
- [x] Non-RZ surface coverage exists through `SurfaceXYZFourier` and
  `SurfaceXYZTensorFourier` wrapper cases.
- [x] Native-contract rejection and mutation/layout guards are covered.
- [x] Stage 2 mixed-quadrature coverage lives in `tests/integration/test_stage2_jax.py`.

Execution constraints:

- [x] Spend no more than 0.5 day on targeted fuzzing / edge-case construction
  for a CPU-`nan` reproducer. Search the low-level kernel inputs and
  surface-derived normals/field arrays; do not let an unproductive search keep
  rows 33 and 35 permanently partial.
- [x] Use `direct_kernel` for low-level `integral_BdotN` parity claims.
- [x] Use `ls_wrapper_gradient` for `SquaredFlux` wrapper value/gradient parity
  claims.
- [x] Keep manifest row 34 partial unless Stage 2 integration evidence changes;
  that row is about integration scope, not the low-level `integral_BdotN`
  boundary behavior.
- [x] Manifest rows 33 and 35 may stay `partial` only if the status is qualified
  as pending the CPU-`nan` reproducer search or if a concrete reproducer test is
  added. If the search finds no reproducer within the timebox, promote the rows
  to `exact` or `contract-complete`.

Open work:

- [x] Search for or construct the exact CPU-side pathological input that makes
  the manifest's claimed `nan` boundary reproducible.
- [x] Record that no CPU-`nan` reproducer exists after the targeted search; no
  low-level `simsoptpp` carve-out test is needed.
- [x] Update `docs/jax_parity_manifest.md` rows 33 and 35 to remove the stale
  unpinned `nan` qualifier.
- [x] If no CPU-`nan` reproducer exists after targeted search, promote the
  banana flux/kernel rows to exact or contract-complete status.
- [x] Keep the Stage 2 integration row separate if it remains partial for
  integration-scope reasons unrelated to `integral_BdotN` math.

Acceptance:

- [x] The manifest no longer claims an unpinned CPU-`nan` boundary.
- [x] Either a concrete reproducer test justifies the partial label, or the row
  is promoted.
- [x] No broad new flux test matrix is added unless the reproducer exposes a
  real uncovered path.

P1 implementation note 2026-05-06: repo-local `.conda/jax-0.9.2` simsoptpp
search covered 248 finite low-level cases across all three definitions and
found zero CPU-`nan` reproducers. `tests/objectives/test_integral_bdotn_jax.py`
now pins direct C++/JAX boundary contracts for zero-normal quadratic flux,
normalized zero-field singularity, and local zero-field / zero-normal behavior.

## P2: Surface Geometry Scope Clarification

Context:

The previous plan described surface legacy API coverage too broadly.
`SurfaceRZFourier` already has substantial object-API parity coverage,
including loaders, copy variants, DOF round-trips, area/volume gradients,
Jacobian parity, and spec roundtrip. `SurfaceXYZFourier` and
`SurfaceXYZTensorFourier` have geometry/spec parity, but they do not have the
same broad object I/O parity. Banana does not require that XYZ object I/O
surface today.

Surface C++ sources exist in `src/simsoptpp`, but the remaining surface
`partial` status is not an uncovered required banana C++ oracle lane. It is a
scope marker for broader legacy Python surface object/API mirroring.

Implementation touch points:

- `src/simsopt/geo/surface.py`
- `src/simsopt/geo/surfacerzfourier.py`
- `src/simsopt/geo/surfacexyzfourier.py`
- `src/simsopt/geo/surfacexyztensorfourier.py`
- `src/simsopt/jax_core/specs.py`
- `src/simsopt/jax_core/surface_rzfourier.py`
- `src/simsopt/geo/surface_fourier_jax.py`
- `src/simsoptpp/surface.cpp`
- `src/simsoptpp/surfacerzfourier.cpp`
- `src/simsoptpp/surfacexyzfourier.cpp`
- `src/simsoptpp/python_surfaces.cpp`
- `tests/geo/test_surface_rzfourier.py`
- `tests/geo/test_surface_rzfourier_jax.py`

Validated current coverage:

- [x] `SurfaceRZFourier` stellsym and non-stellsym DOF round-trip coverage.
- [x] `SurfaceRZFourier` area/volume gradient parity.
- [x] `SurfaceRZFourier` Jacobian parity.
- [x] `SurfaceRZFourier` spec roundtrip coverage.
- [x] `SurfaceRZFourier.from_wout` object-API parity.
- [x] `SurfaceRZFourier.from_vmec_input` object-API parity.
- [x] `SurfaceRZFourier.from_nescoil_input` object-API parity.
- [x] `SurfaceRZFourier.copy` object-API parity.
- [x] `SurfaceXYZFourier` geometry/tangent/spec parity.
- [x] `SurfaceXYZTensorFourier` geometry/derivative/normal/spec parity.

Open banana-scope work:

- [x] Update `docs/jax_parity_manifest.md` to say the banana surface contract is
  geometry/spec complete, while broad XYZ object I/O is intentionally
  out-of-scope for banana.
- [x] Do not add broad XYZ object-API mirror tests unless the product claim is
  explicitly expanded from banana parity to full legacy surface parity.

Optional full-legacy work, only if scope expands:

- [ ] Mirror RZ-style `copy` object-API tests for `SurfaceXYZFourier`.
- [ ] Mirror RZ-style object I/O tests for `SurfaceXYZFourier` where upstream
  supports equivalent constructors.
- [ ] Mirror RZ-style `copy` object-API tests for `SurfaceXYZTensorFourier`.
- [ ] Mirror RZ-style object I/O tests for `SurfaceXYZTensorFourier` where
  upstream supports equivalent constructors.

Acceptance:

- [x] Banana parity docs do not imply full XYZ object-API parity.
- [x] XYZ object I/O remains a documented non-banana scope item unless the
  project chooses the larger legacy-surface parity goal.

## P3: ToroidalFlux Value Matrix

Context:

`ToroidalFlux` parity coverage is stronger than the original plan stated.
`tests/geo/test_surface_objectives_jax.py` already has Taylor checks for surface
Hessian and coil-DOF gradients, plus object parity for first derivatives,
second derivatives, and coil derivatives across the surface-type and `stellsym`
matrix. Exact arithmetic parity is not the right target; tolerance-based
same-state parity is the contract.

Implementation touch points:

- `src/simsopt/geo/surfaceobjectives.py`
- `src/simsopt/geo/surfaceobjectives_jax.py`
- `tests/geo/test_surface_objectives.py`
- `tests/geo/test_surface_objectives_jax.py`
- `benchmarks/validation_ladder_contract.py`

Validated current coverage:

- [x] Reference value parity exists for the canonical object fixture.
- [x] Physical constant-in-`idx` sanity coverage exists.
- [x] First derivative parity is parametrized over `_SURFACE_TYPES` and
  `_STELLSYM_OPTIONS`.
- [x] Second derivative parity is parametrized over `_SURFACE_TYPES` and
  `_STELLSYM_OPTIONS`.
- [x] Coil-derivative parity is parametrized over `_SURFACE_TYPES` and
  `_STELLSYM_OPTIONS`.
- [x] Surface Hessian Taylor coverage exists.
- [x] Coil-DOF gradient Taylor coverage exists.

Open work:

- [x] Add one `test_toroidal_flux_value_parity_matrix` parametrized over
  `_SURFACE_TYPES` and `_STELLSYM_OPTIONS`, matching the derivative matrix.
- [x] Before adding that test, update `validation_ladder_contract.py` by either
  adding scalar value tolerances to `derivative_heavy` or introducing a named
  `toroidal_flux_value_matrix` lane. Do not choose ad hoc tolerances inside the
  test file.
- [x] Keep the manifest row tolerance-based; do not force identical reduction
  order unless a future requirement explicitly demands bitwise/exact arithmetic.
- [x] After the value matrix lands, promote the manifest row to complete with
  bounded tolerance semantics.

Acceptance:

- [x] The only current `ToroidalFlux` test delta is closed by the value matrix.
- [x] No doc claims exact arithmetic parity for this objective family.

P3 implementation note 2026-05-06: `derivative_heavy` now owns scalar value
tolerances, `tests/geo/test_surface_objectives_jax.py` includes the full
surface-type / `stellsym` value matrix, and the manifest row is complete with
tolerance-based semantics.

## P4: Stage 2 Snapshot, Callback, And Reporting Re-Entry Removal

Context:

The gradient hot path is guarded and JAX-native. The stronger guarantee
"zero re-entry into the legacy Optimizable graph anywhere on the JAX lane" is
not delivered. Live non-gradient re-entry sites remain in Stage 2 diagnostics:

- `banana_coil_solver.py::_build_stage2_explicit_term_payload`
- `banana_coil_solver.py::capture_stage2_trajectory_snapshot`
- `banana_coil_solver.py::accepted_callback`

Implementation touch points:

- `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py`
- `src/simsopt/objectives/stage2_target_objective_jax.py`
- `tests/integration/test_stage2_target_lane_purity.py`
- `tests/integration/test_stage2_jax.py`
- `docs/banana_jax_native_port_todos_2026-05-05.md`

Effort and lane:

- [x] Treat this as the only large implementation phase in this plan: 5-10
  focused days for reporting-surface design, Stage 2 reroute, tests, and docs.
- [x] Add a `reporting_contract` lane to `validation_ladder_contract.py` before
  writing payload-parity assertions. Current gradient lanes do not describe
  snapshot/report payload equality.
- [x] Use `tier2_stage2_e2e` only for the reduced Stage 2 run-shape check.

Required refactor:

- [x] Add a target-bundle reporting surface that exposes accepted-step summary
  metrics without calling legacy `Jf.J()`, `Jls.J()`, `Jccdist.J()`,
  `Jccdist.shortest_distance()`, or `Jc.J()`.
- [x] Make the JAX target-lane reporting path bypass
  `_build_stage2_explicit_term_payload` and read from the target-bundle term
  summary for target-lane snapshots/probe payloads.
- [x] Make `capture_stage2_trajectory_snapshot` accept a spec/reporting bundle
  path that does not require `JF`, `Jf`, `Jls`, `Jccdist`, or `Jc`.
- [x] Make `accepted_callback` use the cached accepted-step JAX summary instead
  of mutating `JF.x` and re-evaluating legacy objectives.
- [x] Keep CPU/reference reporting behavior intact for the CPU lane.
- [x] Keep VTK/matplotlib/export boundaries documented as allowed host
  post-processing if they are still intentionally outside the target lane.

Required tests:

- [x] Reduced Stage 2 target-lane run under strict purity for value/grad,
  callback, snapshot, and final report.
- [x] The reduced strict-purity run uses the banana input equilibrium
  `examples/single_stage_optimization/equilibria/wout_nfp22ginsburg_000_014417_iota15.nc`,
  `--backend jax`, `--maxiter 2`, `SIMSOPT_TARGET_LANE_STRICT=1`, and the
  reduced coils/surface/quadrature fixture already used by
  `tests/integration/test_stage2_jax.py`; it must not use production resolution
  as a routine CPU CI gate.
- [x] If the 2-iteration smoke does not deterministically accept a step, cover
  accepted-step reporting through a fixed unit fixture and keep the CLI smoke as
  the strict-purity full-run proof.
- [x] Negative test proving legacy `Optimizable.J` calls fail when strict
  full-run target purity is enabled.
- [x] Test proving accepted callback does not call the C++ distance culler.
- [x] Test proving trajectory snapshot does not call the C++ distance culler.
- [x] Fixed-state CPU/JAX report payload comparison for the fields that are
  expected to match.
- [x] Restart artifact rehydration test using specs, not live C++ objects.
- [x] `docs/banana_jax_native_port_todos_2026-05-05.md` updated so the old
  out-of-scope re-entry note is either closed or explicitly narrowed.

Acceptance:

- [x] A full reduced Stage 2 JAX target-lane run can enable strict full-run
  purity without tripping legacy graph entry points.
- [x] The remaining allowed host boundaries are explicit export/reporting
  operations, not objective re-evaluation.
- [x] The manifest distinguishes this stronger claim from the already-complete
  gradient-hot-path claim.

P4 implementation note 2026-05-06: `Stage2TargetObjectiveBundle` now exposes a
JAX reporting summary for objective, field-error, length, sampled
curve-curve/curve-surface distance, curvature, current, distance-gate, and
self-intersection status. `banana_coil_solver.py` uses that surface for
target-lane trajectory snapshots, accepted-step feasible-partial capture, and
target artifact-state capture. Focused tests now cover fixed-state CPU/JAX
reporting parity, target trajectory snapshots without legacy `J*/Jccdist`
calls, and target feasible-partial capture without the C++ distance culler. The
existing spec-restart test covers artifact rehydration, and the reduced strict
CLI proof passed with `SIMSOPT_TARGET_LANE_STRICT=1`, `--backend jax`,
`--optimizer-backend ondevice`, `--skip-postprocess`, `--nphi 31`, `--ntheta 16`,
and `--maxiter 2` against the default banana equilibrium. CUDA hardware proof
remains separate under P5.

## P5: CUDA Hardware Gate

Context:

CPU contract closure is not CUDA closure. GPU parity requires real hardware
artifacts and should stay separate from CPU-only tests.

Implementation touch points:

- `tests/integration/test_single_stage_jax_cpu_reference.py`
- `tests/geo/test_boozersurface_jax.py`
- `tests/core/test_reductions.py`
- `benchmarks/validation_ladder_contract.py`
- `benchmarks/hf_jobs/run_production_gpu_proof.sh`
- `benchmarks/hf_jobs/cuda_pytest_probe.py`
- GPU/Runpod validation scripts and artifact directories.

Harness status:

- [x] Production GPU proof runner emits explicit JSON payloads for the Boozer
  well-conditioned adjoint CUDA lane and the CPU/GPU reduction
  cancellation-stress lane instead of relying on ad hoc pytest logs.
- [x] GPU proof payloads carry command argv, tracked dirty-tree status, JAX/CUDA
  runtime metadata, x64 status, XLA/JAX flags, host RSS, sampled GPU memory, and
  pass/fail reason.
- [x] The P5 handoff uses the repo-owned immutable runtime seed fixture at
  `benchmarks/fixtures/single_stage_seed_iota15/single_stage_jax_runtime_spec.json`
  instead of an untracked `.artifacts` warm-start path.
- [x] The HF launcher preflight validates both repo-owned seed inputs at the
  target SHA: the Stage 2 `biot_savart_opt.json` seed and the single-stage
  runtime seed spec.
- [ ] Real CUDA artifacts from the current repo state have been captured for all
  required lanes below.

Required CUDA lanes:

- [ ] Stage 2 fixed-state value CUDA parity.
- [ ] Stage 2 fixed-state gradient CUDA parity.
- [ ] Stage 2 reduced e2e CUDA parity.
- [ ] Single-stage initialization CUDA parity.
- [ ] Boozer well-conditioned adjoint CUDA parity.
- [ ] CPU/GPU reduction cancellation-stress parity.

Required artifact metadata:

- [ ] git SHA.
- [ ] dirty-tree status.
- [ ] command.
- [ ] Python version.
- [ ] JAX version.
- [ ] CUDA runtime.
- [ ] driver version.
- [ ] device model.
- [ ] x64 status.
- [ ] XLA/JAX environment flags.
- [ ] peak host RSS.
- [ ] peak GPU memory.
- [ ] pass/fail reason.

Acceptance:

- [x] No CUDA row is marked complete without an artifact from a real CUDA run.
- [x] CPU-only local tests remain valid as CPU closure evidence only.

Current blocker audit, 2026-05-06:

- Local `.conda/jax-0.9.2/bin/python` reports JAX 0.10.0 on `cpu:0`
  only, with `jax_enable_x64=False`; this cannot satisfy a CUDA gate.
- `nvidia-smi` is not available on the local host.
- `runpodctl pod list` returns no active pods.
- `SIMSOPT_HF_GPU_IMAGE` is unset, so the Hugging Face Jobs launch handoff is
  not runnable from the current environment.
- `docker` is not available on the local host, so the reusable HF GPU proof
  image cannot be built locally from `benchmarks/hf_jobs/production_gpu_proof.Dockerfile`.
- Hugging Face Jobs reports no running jobs for this account, so there is no
  in-flight remote proof to wait on.
- Remote `fork/gpu-purity-stage2-20260405` resolves to
  `7e3f2eb5e5462c7d3cc989ce8bf1fe010a04f3a2`, not current local SHA
  `0bb26bb0ad9f7492f7df67f1d203c8159b1816dc`; the official P5 launcher cannot
  prove the current dirty tree until the intended implementation slice is
  committed and pushed.
- The repo-owned P5 runtime seed fixture is present locally at
  `benchmarks/fixtures/single_stage_seed_iota15/single_stage_jax_runtime_spec.json`,
  but like the rest of this implementation slice it is not visible to remote
  preflight until committed and pushed.
- No searched P5 artifact under `.artifacts/runpod_prod_signoff`,
  `.artifacts/parity`, or `.artifacts/pytest` contains current git SHA
  `0bb26bb0ad9f7492f7df67f1d203c8159b1816dc`.

P5 launch handoff:

Run this only after the current work is available at a pushed `--repo-sha` and a
CUDA-capable Hugging Face Jobs image is configured. Build and publish the image
from the repo-owned Dockerfile:

```bash
docker build -f benchmarks/hf_jobs/production_gpu_proof.Dockerfile \
  -t <registry>/simsopt-jax:cuda12-jax092 .
docker push <registry>/simsopt-jax:cuda12-jax092
```

Then launch against the exact pushed SHA. The launcher will clone the repo,
check out the exact SHA, validate that the repo-owned runtime seed fixture is
present at that SHA, run the production proof bundle, and fail if the remote job
does not complete successfully.

```bash
SIMSOPT_HF_GPU_IMAGE=<registry>/simsopt-jax:cuda12-jax092 \
.conda/jax-0.9.2/bin/python benchmarks/hf_jobs/launch_production_gpu_proof.py \
  --repo-url https://github.com/jungdaesuh/simsopt.git \
  --repo-ref gpu-purity-stage2-20260405 \
  --repo-sha <pushed-current-sha> \
  --single-stage-mpol 10 \
  --single-stage-ntor 10 \
  --single-stage-jax-runtime-seed-spec benchmarks/fixtures/single_stage_seed_iota15 \
  --no-detach
```

The same H200-only acceptance path can be triggered through
`.github/workflows/jax_h200_production_proof.yml` with the published image and
exactly one repo-relative single-stage seed input. That workflow runs
`benchmarks/hf_jobs/launch_production_gpu_proof.py` with `--hardware h200`,
`--platform cuda`, and `--repo-sha ${{ github.sha }}` so the workflow status
tracks the remote proof result.

## P6: Closeout Coverage Inventory

Goal: turn the current verbal gap list into a machine-checkable banana coverage
inventory after the narrow gap closures land. This should not delay P1, P3, or
P4 unless manifest drift recurs while those phases are underway.

- [x] Add or update a banana parity coverage table that maps:
  - [x] upstream Python test file,
  - [x] upstream C++ implementation file,
  - [x] JAX implementation file,
  - [x] JAX parity test file,
  - [x] tolerance lane,
  - [x] CPU/JAX status,
  - [x] CUDA status,
  - [x] known carve-out.
- [x] Prefer extending the existing manifest machinery from
  `docs/banana_single_stage_stage2_lavish_validation_plan_2026-04-27.md`
  rather than creating a second hidden SSOT.
- [x] Add a lightweight pytest check that fails if a banana coverage row has:
  - [x] no owner test,
  - [x] a nonexistent file path,
  - [x] a tolerance lane missing from `validation_ladder_contract.py`, or
  - [x] a `complete` status with an unresolved carve-out.
- [x] Add a manifest consistency regression test that fails when a row claims a
  specific coverage behavior but the referenced test file does not exist. This
  is intended to catch stale claims like an unpinned CPU-`nan` boundary before
  they become permanent manifest text.
- [x] Keep broad unsupported families separate from banana scope:
  `SurfaceGarabedian`, `SurfaceHenneberg`, `SurfaceRZPseudospectral`,
  clamped `SurfaceXYZTensorFourier`, analytic/interpolated fields,
  wireframe/permanent-magnet fields, field tracing, and broad objective
  wrappers.

Done when:

- [x] One manifest-style table answers which existing C++/Python tests are
  fully covered, partially covered, or intentionally out of banana scope.
- [x] The table is enforced by a low-cost test.

P6 implementation note 2026-05-06: `docs/jax_parity_manifest.md` now contains a
machine-checked banana coverage inventory. `tests/docs/test_banana_parity_coverage_manifest.py`
enforces owner paths, JAX implementation paths, parity test paths, tolerance
lane existence, and the rule that complete CPU/JAX rows cannot carry unresolved
carve-outs.

## Validation Commands

These commands are pre-merge gates for each P-section closure, not evidence
that this docs-only planning pass ran the test suite.

Run local CPU checks with this repository pinned on `PYTHONPATH` so imports do
not resolve to sibling checkouts:

```bash
PYTHONPATH=/Users/suhjungdae/code/columbia/simsopt-jax/src \
JAX_ENABLE_X64=True \
JAX_PLATFORMS=cpu \
pytest -q \
  tests/objectives/test_fluxobjective_jax_parity.py \
  tests/objectives/test_integral_bdotn_jax.py \
  tests/geo/test_surface_rzfourier_jax.py \
  tests/geo/test_surface_objectives_jax.py \
  tests/integration/test_stage2_target_lane_purity.py \
  tests/integration/test_stage2_jax.py
```

Run Boozer CPU closure checks before changing Boozer status claims:

```bash
PYTHONPATH=/Users/suhjungdae/code/columbia/simsopt-jax/src \
JAX_ENABLE_X64=True \
JAX_PLATFORMS=cpu \
pytest -q \
  tests/geo/test_boozersurface_jax.py \
  tests/geo/test_boozer_derivatives_jax.py \
  tests/integration/test_single_stage_jax_cpu_reference.py \
  -k "boozer or Boozer"
```

Run CUDA checks only on a CUDA machine. Do not treat a CPU run as hardware
proof.

Current local audit outputs, 2026-05-06:

```text
.conda/jax-0.9.2/bin/python -m pytest -q \
  tests/docs/test_banana_parity_coverage_manifest.py \
  tests/test_hf_production_gpu_proof.py
48 passed in 18.19s

.conda/jax-0.9.2/bin/python -m pytest -q \
  tests/objectives/test_integral_bdotn_jax.py::TestIntegralBdotNCppParity \
  tests/geo/test_surface_objectives_jax.py::TestToroidalFluxObjectParity \
  tests/integration/test_stage2_target_lane_purity.py
37 passed, 33 skipped in 33.18s

.conda/jax-0.9.2/bin/python -m pytest -q \
  tests/integration/test_stage2_jax.py \
  -k "target_reporting_summary or capture_stage2_trajectory_snapshot_uses_target_reporting or target_feasible_partial_candidate_skips_cpp_distance_culler or strict_mode_allows_target_scalar_objective_evaluation or target_scalar_objective_matches_stage2_composite_contract"
7 passed, 166 deselected in 26.91s

SIMSOPT_TARGET_LANE_STRICT=1 .conda/jax-0.9.2/bin/python \
  examples/single_stage_optimization/STAGE_2/banana_coil_solver.py \
  --backend jax --optimizer-backend ondevice --skip-postprocess \
  --nphi 31 --ntheta 16 --maxiter 2 \
  --trajectory-json /tmp/stage2-strict-audit-YYkMFb/trajectory.json \
  --output-root /tmp/stage2-strict-audit-YYkMFb/outputs
rc=0
trajectory_bytes=4438

.conda/jax-0.9.2/bin/python -m ruff check \
  benchmarks/hf_jobs/cuda_pytest_probe.py \
  benchmarks/validation_ladder_common.py \
  benchmarks/stage2_e2e_comparison.py \
  benchmarks/single_stage_init_parity.py \
  tests/subprocess/hf_production_gpu_fake_runner.py \
  tests/test_hf_production_gpu_proof.py \
  tests/docs/test_banana_parity_coverage_manifest.py
All checks passed!

bash -n benchmarks/hf_jobs/run_production_gpu_proof.sh
passed

git diff --check -- <implementation/docs/test slice>
passed
```

## Completion Audit

Objective audited: execute and implement this plan against
`docs/jax_parity_manifest.md`.

Prompt-to-artifact checklist:

- [x] `docs/jax_parity_manifest.md` distinguishes the manifest/index from this
  detailed closure plan and does not claim CUDA hardware parity.
- [x] Banana Coverage Inventory maps upstream Python tests, required C++ oracle
  files, JAX implementation files, JAX parity tests, tolerance lanes, CPU/JAX
  status, CUDA status, and carve-outs.
- [x] A machine-checkable verifier,
  `tests/docs/test_banana_parity_coverage_manifest.py`, fails on missing paths,
  unknown tolerance lanes, complete rows with carve-outs, or required C++ lanes
  that are not CPU/JAX complete.
- [x] P1 flux / `integral_BdotN` closure is implemented and tested through
  direct C++/JAX boundary checks plus manifest updates.
- [x] P2 surface scope is implemented as a banana-scope clarification:
  `SurfaceRZFourier` object/API parity remains covered, non-RZ geometry/spec
  parity remains covered, and broad XYZ object I/O is optional full-legacy work.
- [x] P3 `ToroidalFlux` value matrix closure is implemented with tolerance-lane
  ownership in `benchmarks/validation_ladder_contract.py`.
- [x] P4 Stage 2 target reporting / strict reduced run closure is implemented
  without treating optimizer traces as fixed-state parity evidence.
- [x] P5 proof harness is implemented for the additional Boozer adjoint and
  CPU/GPU reduction stress lanes, with provenance and fail-closed metadata.
- [ ] P5 real CUDA artifacts from the current repo state exist for every
  `open under P5` manifest row.
- [x] P6 closeout inventory and local CPU/JAX audit outputs are recorded.

Completion verdict: not complete. All repo-local non-CUDA implementation,
documentation, and verifier work is closed; the remaining missing requirement is
P5 real CUDA proof from a pushed current SHA on a configured CUDA image/host.

Continuation audit, 2026-05-06:

```text
git rev-parse HEAD
0bb26bb0ad9f7492f7df67f1d203c8159b1816dc

.conda/jax-0.9.2/bin/python -m pytest -q \
  tests/docs/test_banana_parity_coverage_manifest.py \
  tests/test_hf_production_gpu_proof.py \
  tests/test_benchmark_helpers.py::test_single_stage_init_fixture_files_are_vendored \
  tests/test_benchmark_helpers.py::test_single_stage_init_fixture_runtime_seed_spec_loads
52 passed in 21.64s

.conda/jax-0.9.2/bin/python -m pytest -q \
  tests/test_hf_production_gpu_proof.py::test_launch_production_gpu_proof_accepts_repo_runtime_seed_spec \
  tests/test_hf_production_gpu_proof.py::test_launch_production_gpu_proof_accepts_matching_remote_repo_ref_and_sha \
  tests/test_hf_production_gpu_proof.py::test_launch_production_gpu_proof_rejects_stage2_seed_missing_from_target_sha \
  tests/test_hf_production_gpu_proof.py::test_launch_production_gpu_proof_rejects_runtime_seed_directory_without_spec \
  tests/test_hf_production_gpu_proof.py::test_launch_production_gpu_proof_rejects_runtime_seed_spec_tree
5 passed in 2.18s

.conda/jax-0.9.2/bin/python -m ruff check \
  benchmarks/hf_jobs/cuda_pytest_probe.py \
  benchmarks/hf_jobs/launch_production_gpu_proof.py \
  benchmarks/validation_ladder_common.py \
  benchmarks/stage2_e2e_comparison.py \
  benchmarks/single_stage_init_parity.py \
  tests/subprocess/hf_production_gpu_fake_runner.py \
  tests/test_hf_production_gpu_proof.py \
  tests/docs/test_banana_parity_coverage_manifest.py \
  tests/test_benchmark_helpers.py
All checks passed!

bash -n benchmarks/hf_jobs/run_production_gpu_proof.sh
passed

git diff --check -- <implementation/docs/test slice>
passed

.conda/jax-0.9.2/bin/python -c '<jax runtime probe>'
{"python": "3.11.15", "jax": "0.10.0", "backend": "cpu", "devices": ["cpu:0"], "x64": false}

nvidia-smi
unavailable

docker / podman / nerdctl
unavailable

SIMSOPT_HF_GPU_IMAGE
unset

runpodctl pod list
[]

HF Jobs MCP
No running jobs found.

git ls-remote fork gpu-purity-stage2-20260405
7e3f2eb5e5462c7d3cc989ce8bf1fe010a04f3a2 refs/heads/gpu-purity-stage2-20260405

rg -l 0bb26bb0ad9f7492f7df67f1d203c8159b1816dc \
  .artifacts/runpod_prod_signoff .artifacts/parity .artifacts/pytest
no matching P5 artifacts
```

## Closeout Checklist

- [x] Update `docs/jax_parity_manifest.md` after each completed workstream.
- [x] Update this file's checkboxes as implementation lands.
- [x] Keep old partial rows until their tests and docs both agree.
- [x] Attach pytest command output or artifact paths to the final closeout.
- [ ] If committing, stage only this plan or the exact implementation slice
  requested; leave unrelated `.artifacts/` and generated files untouched.
