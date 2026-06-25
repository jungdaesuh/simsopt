# JAX Port Faithfulness Remediation — Implementation Plan

> Created 2026-06-25. Source: the 15-lane surrogate→JAX faithfulness audit
> (workflow run `wf_482470b2-f5c`, 23 agents, adversarial verify on
> critical/high). See memory `project_jax_port_faithfulness_audit_2026_06_25`.
>
> Execution status updated 2026-06-25 in
> `/Users/suhjungdae/code/columbia/simopt-jax-clean-local`: required CPU/JAX
> remediation items are closed. The finite-I Boozer full port and finite-build
> conductor-curvature diagnostics remain explicitly deferred by the conditional
> gates below.

## Purpose

Close the concrete, evidence-bound gaps the audit confirmed between the custom
reference code in `/Users/suhjungdae/code/columbia/simsopt-surrogate` and its
JAX port in this worktree's
`src/{simsopt_jax,simsopt_jax_adapters}`. The audit
verdict was **faithful at the kernel level** (every differentiable kernel
reproduces reference math; 10/15 lanes have genuine cross-boundary C++/CPU-oracle
parity), with the work below being an *incomplete superset-port* in three places
plus a few overstated "parity" tests. This plan turns the audit's prioritized
actions into executable, trackable work.

## Goals

- The `surface_tangent` framed-curve family (reference default for the banana
  finite-build coil pack) is ported to JAX and cross-validated against the
  reference, OR the JAX path fails loud when it is requested (no silent
  centroid substitution).
- The JAX banana single-stage objective (`banana_local_value`) is either
  completed to match the production Stage-2 objective, or explicitly stamped as a
  named subset everywhere a caller could mistake it for the full objective.
- Finite-enclosed-current Boozer (α = G + iota·I) is either implemented in JAX or
  fail-closed-guarded so the orphaned scaffolding can never silently run a wrong
  residual.
- The tracing within-step Levelset stop-localization and stop-event ordering
  match the custom reference, so lost-line exit points / loss diagnostics are not
  off by an adaptive step.
- The overstated/tautological parity tests (banana driver, qfm degenerate
  boundary, GSCO oscillation, higher-order surface `_lin`) are replaced or
  augmented with genuine checks — no production kernel behavior changes.

## Non-Goals

- Re-validating the 9 FAITHFUL lanes — they are confirmed and out of scope.
- CUDA/GPU hardware parity closure (tracked separately under the P5 hardware
  gate; this plan is CPU/JAX-faithfulness only).
- Refactoring host-side/non-differentiable code that the audit correctly judged
  NON-PORTABLE (BiotSavart cache, GSON/MPI, QFM SLSQP host solver, axis
  fixed-point locator, `hull2D`, `accessibility.py`, SPEC/Fortran, VTK/IO).
- Changing the production GSCO/qfm/banana kernels for the test-integrity items —
  those are test-only changes unless a defect is separately confirmed.

## Current Context

Historical audit anchors (read 2026-06-25 in both trees) and execution status:

- **Source-worktree provenance.** The JAX target is this worktree,
  `/Users/suhjungdae/code/columbia/simopt-jax-clean-local`, originally audited at
  `4fa5e61c2`. Reference anchors below are from the sibling custom reference
  `/Users/suhjungdae/code/columbia/simsopt-surrogate` at `bdc6f375e` with a
  dirty/ahead tree. Do not validate those reference anchors against this checkout's
  bundled `src/simsopt` or `src/simsoptpp`; those lag the custom reference for the
  surface-tangent finite-build path and dense Levelset root-finding.
- **D1/D3 silent frame fallback — closed.** The audit baseline found
  `curve_specs.py` mapping every non-Frenet frame to centroid and
  `finitebuild.py` using a Frenet/centroid binary branch with no validation. The
  current remediation replaces that with explicit
  `{centroid, frenet, surface_tangent}` handling, spec-construction validation,
  and a required `surface_major_radius` for `surface_tangent`.
- **D1 surface-tangent port — closed for current consumers.** The sibling
  reference already had JAX formulas for `surface_tangent_normal_direction`,
  `rotated_surface_tangent_frame*`, `torsion_pure_surface_tangent`, and
  `binormal_curvature_pure_surface_tangent`. The current worktree ports those
  kernels, threads `R0/z0` through `CurveFilamentSpec`, wires finite-build and
  adapter wrappers, and leaves only optional conductor-curvature
  `gammadashdash` consumers deferred until a real consumer exists.
- **D4 finite-I Boozer residual — fail-closed.** The audit baseline found the
  JAX Boozer residual vacuum-only (`G`, not `G + iota*I`). Grep over the JAX tree
  showed no `--boozer-I` / `--plasma-current-A` CLI route and no consumer for the
  finite-current resolver helpers. The current remediation adds the explicit
  zero-enclosed-current guard; the full finite-I residual/adjoint port remains
  deferred until a finite-I banana fixture is scheduled.
- **D5 banana geometry test tautology — closed.** The audit baseline found the
  driver parity test comparing `banana_local_terms` against driver terms that
  shared the same poloidal/width kernels. The current remediation adds closed-form
  kernel tests in `tests/core/test_banana.py` and independent NumPy driver-oracle
  checks in `tests/test_banana_jax_specs.py` before the frozen-spec wiring
  comparison.
- **D2 Levelset endpoint-only stop — closed.** The audit baseline found the
  Levelset classifier evaluated only at the accepted-step endpoint. The current
  remediation selects the earliest localized stopping-criterion event inside the
  accepted step, handles start-outside semantics, and filters phi-plane hits after
  `stop_time`; adapter translation feeds public Levelset criteria into that core
  localizing path.

Test/env note: the JAX repo has **no** local `.conda-env`; the surrogate repo has
`.conda-env/bin/python3.11`. Full GPU suite is XLA-compile-bound and must set
`JAX_COMPILATION_CACHE_DIR` (see `project_full_tests_runpod_compile_cache`).
CPU-lane parity validation is the practical local gate.

## Rationale

At planning time the defects clustered by *reachability*, not by physics
correctness:

- **D3 (silent fallback)** was the actively-dangerous future-caller hazard, so it
  landed first as fail-loud validation before the frame port was wired.
- **D1 (port the frame)** was a transcription job: the reference kernels were
  already JAX, so the work moved them into `simsopt_jax/core/framedcurve.py`,
  threaded `R0/z0` through `CurveFilamentSpec`, and covered the current consumers
  with independent analytic/NumPy and finite-difference checks.
- **D4 (finite-I)** was latent because no JAX path could set `I≠0`; the selected
  remediation is fail-closed now, with the full finite-current banana port deferred
  until it has a fixture and consumer.
- **D5/qfm/GSCO/surface `_lin`** were test-integrity items: they remove false
  confidence without changing production kernel behavior.

## Assumptions

- **A1 (verified/resolved):** the live JAX banana single-stage driver does *not*
  route through finite-build framed curves; D1 was forward-looking rather than an
  actively-miscomputed banana objective path.
- **A2 (verified 2026-06-25):** there is no `--boozer-I` / `--plasma-current-A`
  (or equivalent) plumbing that can reach `_boozer_weighted_residual` with `I≠0`
  at `4fa5e61c2`; the finite-current resolver helpers exist in
  `current_contracts.py` but have no consumers. Re-check argparse + the
  single-stage facade before implementing P1-4a on a later HEAD.
- **A3 (resolved):** `torsion_pure_surface_tangent` and
  `binormal_curvature_pure_surface_tangent` were read from the reference
  `framedcurve.py` and ported alongside the frame kernels.
- **A4:** the reference banana penalty terms are the SSOT (no C++ oracle), so
  "faithful" for banana means matching the reference JAX kernels / closed-form
  hand calculations, not a C++ comparison.

## Implementation Plan

### Phase P0 — Stop the silent-correctness hazard, then port the frame (framedcurve)

1. Fail-loud frame validation (D3) — cheap, lands first.
   - [x] In `src/simsopt_jax/core/finitebuild.py:_frame_geometry_from_spec`,
         validate `spec.frame_kind ∈ {"centroid","frenet","surface_tangent"}`;
         raise `ValueError` on anything else instead of falling through to centroid.
   - [x] Superseded by P0-2 in the same remediation: `surface_tangent` no longer
         needs a long-lived `NotImplementedError` guard because the frame family
         is ported and wired instead of silently computing centroid.
   - [x] In `src/simsopt_jax/core/specs.py` (`CurveFilamentSpec` ctor /
         `make_curve_filament_spec`, ~735-746 & ~1256-1278), validate `frame_kind`
         at spec-construction time with the same allowed set.
   - [x] In `src/simsopt_jax_adapters/geo/curve_specs.py:138`, replace the
         `frenet`-else-`centroid` ternary with an explicit map that detects
         `FramedCurveCentroid`, `FramedCurveFrenet`, and `FramedCurveSurfaceTangent`,
         and raises on an unrecognized framed-curve type.
2. Port the `surface_tangent` frame family (D1).
   - [x] Read and transcribe into `src/simsopt_jax/core/framedcurve.py`:
         `surface_tangent_normal_direction`, `rotated_surface_tangent_frame`,
         `rotated_surface_tangent_frame_dash` (via `jvp`),
         `rotated_surface_tangent_frame_dashdash`, and the
         `torsion_pure_surface_tangent` / `binormal_curvature_pure_surface_tangent`
         kernels (reference `framedcurve.py:865-946` and `1102-1118`). Keep
         formulas byte-identical; do not "clean up" the
         projection/renormalization.
   - [x] Add `R0` (`surface_major_radius`) and `z0` (`surface_midplane_z`) fields to
         `CurveFilamentSpec` (`specs.py`), defaulting to `None`, required only when
         `frame_kind == "surface_tangent"`.
   - [x] Add the `surface_tangent` branch to
         `finitebuild.py:_frame_geometry_from_spec` calling the new frame kernels
         (replace the P0-1 `NotImplementedError`).
   - [x] Thread `R0/z0` through `curve_specs.py:_curve_filament_spec_from_curve`
         from `FramedCurveSurfaceTangent.major_radius` / `.midplane_z`, and import
         `FramedCurveSurfaceTangent`.
   - [x] Add a `FramedCurveSurfaceTangentJAX` adapter wrapper in
         `src/simsopt_jax_adapters/geo/framed_curve.py` (mirroring the centroid/Frenet
         wrappers) and export it in `__all__`.
3. [x] (Optional gate resolved as deferred) port `alphadashdash` /
   `rotated_*_frame_dashdash` consumers and `CurveFilament.gammadashdash` for
   finite-build conductor-curvature diagnostics. No current P0-2 consumer requires
   this path, so it remains deliberately deferred until a real consumer lands.

### Phase P1 — Make the banana JAX objective complete or explicitly bounded

4. Finite-enclosed-current Boozer (D4) — fail-closed now, port later.
   - [x] (4a) Add a guard in the JAX banana single-stage facade (the caller of
         `make_traceable_solved_state_value_and_grad` / `_boozer_weighted_residual`)
         that asserts enclosed current `I == 0`, raising a clear error otherwise.
         Verify A2 first so this is a true fail-closed, not a false barrier.
   - [x] (4b, scheduled when finite-I banana is actually needed) deferred: `I` is
         not threaded into the residual in this CPU-faithfulness remediation because
         no JAX banana route can configure finite enclosed current; the 4a guard is
         the active contract until a finite-I fixture is scheduled. Full task:
         thread `I` into
         `_boozer_weighted_residual` as `G_eff = G + iota*I`, and propagate the
         (iota, G) rank-one chain-rule into the gradient/adjoint path
         (`make_traceable_solved_state_value_and_grad`). Cross-validate against the
         reference `boozer_finite_current.py` on a finite-I fixture.
5. Banana objective contract decision (real gap, not a "defect").
   - [x] Decision: document `banana_local_value` as a named subset.
   - [x] Completing branch deferred by the subset decision: port the always-on
         **self-envelope** term
         (`_self_distance_window_pure` + Gonzalez–Maddocks `_global_radius_pure`,
         reference `self_intersect.py:368,451`), the **fold** term assembly
         (`CurveSurfaceGeodesicCurvature`, reference `banana_coil_solver.py:4472,4713`,
         which needs framed-curve binormal curvature in the frozen-spec geometry),
         and the single-stage **`PoloidalExtentFloor`** smooth-max soft floor
         (reference `poloidal_extent.py:264`).
   - [x] Bounding branch done: stamp `banana_local_value` as a named subset in its
         docstring, in `docs/banana_jax_port_implementation_plan.md`, and in
         `docs/jax_parity_manifest.md`, listing the omitted always-on terms so no
         caller assumes production-objective equivalence.

### Phase P2 — Solver / diagnostic fidelity (bounded, non-objective)

6. Tracing within-step Levelset stop-localization (D2 + stop-event ordering).
   - [x] In `src/simsopt_jax/core/tracing.py`, when a Levelset criterion changes
         sign within the accepted step, root-find via the existing
         `bracket_root_jax` sub-step machinery on `classifier_fn ∘ state_at_fraction`
         and record the interpolated crossing state (reference
         `/Users/suhjungdae/code/columbia/simsopt-surrogate/src/simsoptpp/tracing.cpp:461-518`).
   - [x] Handle the start-of-step `value_last < 0` case.
   - [x] Drop step crossings with `t_hit > stop_time`; select the stop event by
         **minimum time**, not minimum index, when ≥2 criteria fire.
   - [x] Adapter route verified at
         `src/simsopt_jax_adapters/field/tracing.py:689-715`: the public
         `LevelsetStoppingCriterion` is translated to `JaxLevelsetStoppingCriterion`
         and therefore uses the core localizing path; no separate adapter
         interpolation implementation is needed.
   - [x] Fix the cheap convention nit alongside (`iter >= max_iter` boundary,
         tracing-4).
7. iota-collapse early-exit guard (solveopt-1).
   - [x] Documented branch selected: the JAX single-stage route relies on the
         objective-level iota penalty rather than host early `iota_collapsed`
         telemetry. `docs/jax_parity_manifest.md` records that the telemetry is not
         claimed. Deferred alternative: add `iota_collapse_fraction` / `iota_reference` to
         `BoozerSurfaceJAX.run_code` and the LBFGS stage, emitting `iota_collapsed`
         (reference `boozersurface.py:173-225`, `single_stage_geometry.py:1177-1182`,
         env `IOTA_COLLAPSE_REJECT_FRACTION=0.3`), OR document that the JAX
         single-stage relies on the objective-level iota penalty instead.

### Phase P3 — Test integrity (no production-kernel behavior change)

8. Banana geometry-kernel tests (D5).
   - [x] Add hand-computed closed-form unit tests for `banana_poloidal_extent_pure`
         and `banana_projected_ellipse_width_pure` (mirror the self-distance/pack
         closed-form tests in `tests/core/test_banana.py:583-601`).
   - [x] Add an independent oracle side to the driver geometry-term parity:
         `tests/test_banana_jax_specs.py` now checks poloidal extent and projected
         width penalties against local NumPy formulas before comparing the frozen
         spec and driver wiring. This satisfies the allowed branch: stop routing the
         driver geometry-term parity through the shared
         `banana.py` kernels in `tests/test_banana_jax_specs.py:348-466` (or add an
         independent oracle side).
9. qfm degenerate-boundary tests (qfmflux-1).
   - [x] There is no existing `*_matches_cpp_boundary_contract` selector in
         `tests/geo/test_qfmsurface_jax.py` at `4fa5e61c2`; add explicit
         degenerate-boundary coverage for the QFM residual/label cases, or, if the
         audit refers to a different file, first locate that selector and rename it
         there. The test must record any JAX `0.0`/`inf` vs C++ `nan` divergence
         explicitly and must not hide it behind `importorskip`. Do **not** change
         the production kernel.
10. GSCO oscillation parity (magnets-1).
    - [x] Intentional-divergence branch selected: keep the corrected JAX modulo
          opposite-candidate policy and document that raw byte identity with the
          native precedence quirk is not claimed. Either reproduce the compiled C++
          GSCO behavior (`opt_ind+nLoops` literal,
          vs the JAX `(opt_ind+nLoops)%(2*nLoops)`) for byte-parity, or keep the
          corrected index and add a **non-degenerate** GSCO parity test (nonzero
          A,b; multiple loops; run to oscillation) plus an intentional-divergence note.
11. Higher-order surface `_lin` anchor (D6) + Boozer composition (boozerfield-1/2).
    - [x] Add a central-difference anchor of `gammadash1dash1_lin` vs d/dφ of the
          C++-validated `gammadash1_lin` (or compare against the `.def`-bound C++
          `surface.gammadash1dash1dash1_lin` on a paired grid) at
          `tests/geo/test_surface_fourier_jax.py:1334-1343`.
    - [x] Add the `InterpolatedBoozerFieldJAX` composition test vs
          `sopp.InterpolatedBoozerField`.

### Phase P4 — Cosmetic

12. Cosmetic documentation/test-label cleanup.
    - [x] Rename/redocument the mislabeled `tests/field/test_biotsavart_jax_parity.py`
          (the real cross-boundary gate is the sibling
          `test_biotsavart_jax.py::TestBiotSavartJaxCppParity`).
    - [x] Fix the axishull PI-controller docstring overclaim (axishull-2).

## Validation Plan

- [x] **P0-1 (fail-loud):** unknown `frame_kind` values raise at
      spec-construction and finite-build use; `surface_tangent` specs require
      surface metadata instead of falling through to centroid; `centroid`/`frenet`
      paths remain covered.
- [x] **P0-2 (frame port):** port `rotated_surface_tangent_frame`
      (+`_dash`, torsion, binormal-curvature) from the sibling reference source in
      `/Users/suhjungdae/code/columbia/simsopt-surrogate`; cover the current tree
      with the closed-form circular-torus oracle, non-planar orthonormality checks,
      NumPy wrapper value oracle, and finite-difference value/VJP tests in
      `tests/geo/test_framedcurve_jax_item18.py`,
      `tests/geo/test_framedcurve_jax_wrappers_item18.py`, and
      `tests/geo/test_finitebuild_jax_item20.py`. No external sibling-runtime
      regression test is claimed in this branch.
- [x] **P1-4a (fail-closed I):** test that the banana single-stage facade raises on
      `I≠0` and runs unchanged on `I=0`.
- [x] **P1-4b (finite-I, if done):** not applicable to this remediation branch;
      finite-I remains fail-closed until a finite-current banana fixture is
      scheduled, so no finite-I residual/gradient parity gate was run.
- [x] **P1-5 (banana contract):** if bounding, a doc/test that asserts the subset
      manifest lists every always-on reference term not in `banana_local_value`
      (extend `tests/docs/test_banana_parity_coverage_manifest.py`).
- [x] **P2-6 (tracing):** Levelset-on-3D-field loss test asserting the recorded
      loss point lies on the stopping surface (not one step past it); regression
      that phi-plane Poincaré dots are unchanged.
- [x] **P3-8/9/10/11:** tests now use independent or non-tautological anchors:
      banana closed-form and NumPy driver oracles, explicit QFM zero-boundary
      contracts, non-degenerate GSCO opposite-candidate state, surface `_lin`
      central difference, and CPU-vs-`InterpolatedBoozerFieldJAX` composition.
      The Boozer composition test is present but the local focused run skipped it
      when `booz_xform` was unavailable.
      This is the retained mutation-sensitivity evidence; no external mutation
      harness artifact is required for this CPU lane.
- [x] **Suite gate:** `tests/geo/test_framedcurve_jax*.py`,
      `tests/core/test_banana.py`, `tests/test_banana_jax_specs.py`,
      `tests/field/test_*tracing*` green on the CPU lane; full GPU suite only when a
      cache dir is set (`JAX_COMPILATION_CACHE_DIR`). Local focused run result:
      27 passed, 1 skipped (`booz_xform` missing).
- [x] No regression in the existing genuine parity tests for the touched CPU/JAX
      lanes; full GPU parity remains outside this CPU-only plan.

## Risks and Mitigations

- Risk: P0-2 frame port introduces a subtle projection/renormalization difference
  from the reference (e.g. tangent normalization order).
  Mitigation: transcribe from reference `framedcurve.py:865-897`; gate current
  consumers on independent circular-torus analytic checks, wrapper closed-form
  checks, non-planar orthonormality, and finite-difference value/VJP coverage.
- Risk: P1-4a fail-closed guard fires on a legitimate I=0-but-not-exactly-zero
  float path, breaking a working lane.
  Mitigation: verify A2 (no I≠0 plumbing) first; guard on the spec/argparse value,
  not a downstream float, and use an exact `== 0` on the configured current.
- Risk: P2-6 within-step root-find changes converged Poincaré/loss numbers for
  existing runs, masquerading as a regression.
  Mitigation: assert phi-plane crossings unchanged; only lost-line exit point/time
  should move (toward the surface). Document the expected ≤1-step delta.
- Risk: P3 test-integrity changes are written to pass trivially (re-introducing
  tautology).
  Mitigation: anchor each new test to a closed-form, NumPy, CPU, or
  central-difference oracle so shared-kernel tautologies do not pass.
- Risk: scope creep into completing the full banana objective (P1-5 "complete"
  branch) when "bound + document" is sufficient for current needs.
  Mitigation: make the contract decision (P1-5 first checkbox) an explicit gate
  with the user before porting self-envelope/fold/floor.

## Completion Criteria

- [x] No JAX path can silently compute a centroid frame when `surface_tangent` is
      requested (P0-1 merged with tests).
- [x] `surface_tangent` frame is fail-loud for missing metadata and is covered by
      independent analytic/NumPy geometry oracles plus finite-difference gradient
      checks for current consumers (P0-2); no external sibling-runtime regression
      gate is claimed.
- [x] `_boozer_weighted_residual` cannot run with `I≠0` unless the finite-I
      chain-rule path is implemented and parity-checked (P1-4).
- [x] `banana_local_value` is either complete vs the production Stage-2 objective
      or documented as a named subset in code + manifest (P1-5).
- [x] Tracing lost-line exit points match the reference within the documented
      ≤1-step bound; stop-event selection is by minimum time (P2-6).
- [x] The four overstated tests (banana driver, qfm degenerate, GSCO, surface
      `_lin`) are replaced/augmented with genuine independent-oracle checks (P3).
- [x] Audit memory `project_jax_port_faithfulness_audit_2026_06_25` update is
      represented by this execution-status section and the final OpenMemory record
      for the run; confirmed defects are marked resolved or explicitly deferred.

## Open Questions

- Banana objective contract (P1-5): resolved as **bound + document**. Completing
  the always-on self-envelope + fold + floor terms is a future production-objective
  expansion, not part of this CPU-faithfulness closure.
- Finite-I banana (P1-4b): resolved as fail-closed until a finite-enclosed-current
  single-stage fixture is scheduled.
- GSCO (P3-10): resolved as corrected JAX modulo opposite-candidate policy plus an
  intentional-divergence note; raw byte identity with the C++ precedence quirk is
  not claimed for that branch.
- P2-7 iota-collapse: resolved as documented reliance on objective-level iota
  penalty; host early `iota_collapsed` telemetry is not claimed by the JAX
  single-stage route.
