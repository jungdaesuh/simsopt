# JAX Port Faithfulness Remediation — Implementation Plan

> Created 2026-06-25. Source: the 15-lane surrogate→JAX faithfulness audit
> (workflow run `wf_482470b2-f5c`, 23 agents, adversarial verify on
> critical/high). See memory `project_jax_port_faithfulness_audit_2026_06_25`.

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

Confirmed anchors (read 2026-06-25 in both trees):

- **Source-worktree provenance.** The JAX target is this worktree,
  `/Users/suhjungdae/code/columbia/simopt-jax-clean-local`, at `4fa5e61c2`
  during this review. Reference anchors below are from the sibling custom
  reference `/Users/suhjungdae/code/columbia/simsopt-surrogate` at `bdc6f375e`
  with a dirty/ahead tree. Do not validate those reference anchors against this
  checkout's bundled `src/simsopt` or `src/simsoptpp`; those lag the custom
  reference for the surface-tangent finite-build path and dense Levelset
  root-finding.
- **D1/D3 silent frame fallback.**
  `src/simsopt_jax_adapters/geo/curve_specs.py:138` maps
  `frame_kind="frenet" if isinstance(curve.framedcurve, FramedCurveFrenet) else "centroid"`
  (no `surface_tangent`, no `FramedCurveSurfaceTangent` import).
  `src/simsopt_jax/core/finitebuild.py:89-131` (`_frame_geometry_from_spec`) is a
  binary `if spec.frame_kind == "frenet": … else: <centroid>` with no validation.
  Reference
  `/Users/suhjungdae/code/columbia/simsopt-surrogate/src/simsopt/geo/finitebuild.py:118-127`
  raises `ValueError` for any frame outside
  `{centroid, frenet, surface_tangent}` and again when `surface_tangent` is
  requested without `surface_major_radius`.
- **D1 port surface already exists as JAX in the sibling reference** —
  `/Users/suhjungdae/code/columbia/simsopt-surrogate/src/simsopt/geo/framedcurve.py`:
  `surface_tangent_normal_direction` (865), `rotated_surface_tangent_frame` (881),
  `rotated_surface_tangent_frame_dash` (900), `rotated_surface_tangent_frame_dashdash`
  (910), all `*_dcoeff_vjp{0,1,3,4,5}` (916-946), plus
  `torsion_pure_surface_tangent` / `binormal_curvature_pure_surface_tangent`
  (1102-1118; consumed by `FramedCurveSurfaceTangent.__init__`, 545-569) and
  the class `FramedCurveSurfaceTangent` (508-725). It is gradient-wired into
  the banana finite-build path via `finitebuild.py:145-151` and
  `examples/.../banana_opt/finitebuild_export.py` (`default="surface_tangent"`).
- **D4 vacuum-only Boozer residual.**
  `src/simsopt_jax/geo/boozer_residual.py:193-200` (`_boozer_weighted_residual`)
  computes `residual = G * B - B2[…]·(xphi + iota·xtheta)` — α hard-coded to `G`.
  Reference finite-I path: `examples/.../banana_opt/boozer_finite_current.py:228,275`
  (`G_effective = G + iota*I`), `boozer_residuals.py:160-172`,
  `stage2_single_stage_handoff.py:1414`. Grep over the current JAX tree shows
  no `--boozer-I` / `--plasma-current-A` CLI route and no consumer for
  `resolve_plasma_current_settings` / `PlasmaCurrentSettings`; existing imports
  of `examples/single_stage_optimization/banana_opt/current_contracts.py` use
  current-bound helpers, not the finite-current Boozer resolver.
- **D5 tautological banana driver-parity test.**
  `tests/test_banana_jax_specs.py:348-466` compares `banana_local_terms` against
  the driver's `banana_geometry_terms`, but both import/call the identical
  `src/simsopt_jax/core/banana.py:1248-1285` kernels
  (`banana_poloidal_extent_pure`, `banana_projected_ellipse_width_pure`). Those two
  kernels have no independent numeric test (self-distance does:
  `tests/core/test_banana.py:583-601`). No C++ oracle exists (reference banana
  penalties are themselves JAX).
- **D2 tracing endpoint-only stop.**
  `src/simsopt_jax/core/tracing.py:624-630,1178-1205` and adapter
  `src/simsopt_jax_adapters/field/tracing.py:689-715` evaluate the Levelset
  classifier only at the accepted-step endpoint. Reference
  `/Users/suhjungdae/code/columbia/simsopt-surrogate/src/simsoptpp/tracing.cpp:461-518`
  root-finds the crossing inside the step (toms748 on the dense interpolant) and
  handles the start-of-step `value_last<0` case. The JAX phi-plane path already
  has sub-step bracketing (`bracket_root_jax`) that the Levelset path can reuse.

Test/env note: the JAX repo has **no** local `.conda-env`; the surrogate repo has
`.conda-env/bin/python3.11`. Full GPU suite is XLA-compile-bound and must set
`JAX_COMPILATION_CACHE_DIR` (see `project_full_tests_runpod_compile_cache`).
CPU-lane parity validation is the practical local gate.

## Rationale

The defects cluster by *reachability*, not by physics correctness:

- **D3 (silent fallback) is the only actively-dangerous one** for a future caller
  and is a ~10-line fail-loud guard — do it first, independent of everything else.
- **D1 (port the frame)** is a transcription job: the reference kernels are
  already JAX, so the work is moving them into `simsopt_jax/core/framedcurve.py`,
  threading `R0/z0` through `CurveFilamentSpec`, and cross-validating — low
  algorithmic risk, high coverage payoff (unblocks the reference-default coil pack).
- **D4 (finite-I)** is latent because no JAX path can set `I≠0` today; the cheap
  correct move is fail-closed now, full port only when finite-current banana is
  actually scheduled — avoids building unused, untested numerics.
- **D5/qfm/GSCO/surface `_lin`** are test-integrity items: the kernels are
  correct, the tests just don't prove it. They remove false confidence without
  touching production behavior, so they are P3 (do after the correctness items).

## Assumptions

- **A1 (verify cheaply):** the live JAX banana single-stage driver does *not*
  currently route through finite-build framed curves, so D1 is forward-looking
  rather than actively miscomputing. Confirm by grepping the JAX banana driver for
  `CurveFilament`/`finitebuild` usage before sizing P0-2.
- **A2 (verified 2026-06-25):** there is no `--boozer-I` / `--plasma-current-A`
  (or equivalent) plumbing that can reach `_boozer_weighted_residual` with `I≠0`
  at `4fa5e61c2`; the finite-current resolver helpers exist in
  `current_contracts.py` but have no consumers. Re-check argparse + the
  single-stage facade before implementing P1-4a on a later HEAD.
- **A3:** `torsion_pure_surface_tangent` and `binormal_curvature_pure_surface_tangent`
  exist in the reference `framedcurve.py` near 1102-1118 (referenced by the class
  init at 545-569); they must be read and ported alongside the frame kernels.
- **A4:** the reference banana penalty terms are the SSOT (no C++ oracle), so
  "faithful" for banana means matching the reference JAX kernels / closed-form
  hand calculations, not a C++ comparison.

## Implementation Plan

### Phase P0 — Stop the silent-correctness hazard, then port the frame (framedcurve)

1. Fail-loud frame validation (D3) — cheap, lands first.
   - [ ] In `src/simsopt_jax/core/finitebuild.py:_frame_geometry_from_spec`,
         validate `spec.frame_kind ∈ {"centroid","frenet","surface_tangent"}`;
         raise `ValueError` on anything else instead of falling through to centroid.
   - [ ] Until P0-2 lands, raise `NotImplementedError("surface_tangent frame not
         yet ported to JAX")` for `frame_kind == "surface_tangent"` rather than
         computing centroid.
   - [ ] In `src/simsopt_jax/core/specs.py` (`CurveFilamentSpec` ctor /
         `make_curve_filament_spec`, ~735-746 & ~1256-1278), validate `frame_kind`
         at spec-construction time with the same allowed set.
   - [ ] In `src/simsopt_jax_adapters/geo/curve_specs.py:138`, replace the
         `frenet`-else-`centroid` ternary with an explicit map that detects
         `FramedCurveCentroid`, `FramedCurveFrenet`, and `FramedCurveSurfaceTangent`,
         and raises on an unrecognized framed-curve type.
2. Port the `surface_tangent` frame family (D1).
   - [ ] Read and transcribe into `src/simsopt_jax/core/framedcurve.py`:
         `surface_tangent_normal_direction`, `rotated_surface_tangent_frame`,
         `rotated_surface_tangent_frame_dash` (via `jvp`),
         `rotated_surface_tangent_frame_dashdash`, and the
         `torsion_pure_surface_tangent` / `binormal_curvature_pure_surface_tangent`
         kernels (reference `framedcurve.py:865-946` and `1102-1118`). Keep
         formulas byte-identical; do not "clean up" the
         projection/renormalization.
   - [ ] Add `R0` (`surface_major_radius`) and `z0` (`surface_midplane_z`) fields to
         `CurveFilamentSpec` (`specs.py`), defaulting to `None`, required only when
         `frame_kind == "surface_tangent"`.
   - [ ] Add the `surface_tangent` branch to
         `finitebuild.py:_frame_geometry_from_spec` calling the new frame kernels
         (replace the P0-1 `NotImplementedError`).
   - [ ] Thread `R0/z0` through `curve_specs.py:_curve_filament_spec_from_curve`
         from `FramedCurveSurfaceTangent.major_radius` / `.midplane_z`, and import
         `FramedCurveSurfaceTangent`.
   - [ ] Add a `FramedCurveSurfaceTangentJAX` adapter wrapper in
         `src/simsopt_jax_adapters/geo/framed_curve.py` (mirroring the centroid/Frenet
         wrappers) and export it in `__all__`.
3. (Optional, deferred unless P0-2 consumers need it) port `alphadashdash` /
   `rotated_*_frame_dashdash` consumers and `CurveFilament.gammadashdash` for
   finite-build conductor-curvature diagnostics. Gate on a real consumer.

### Phase P1 — Make the banana JAX objective complete or explicitly bounded

4. Finite-enclosed-current Boozer (D4) — fail-closed now, port later.
   - [ ] (4a) Add a guard in the JAX banana single-stage facade (the caller of
         `make_traceable_solved_state_value_and_grad` / `_boozer_weighted_residual`)
         that asserts enclosed current `I == 0`, raising a clear error otherwise.
         Verify A2 first so this is a true fail-closed, not a false barrier.
   - [ ] (4b, scheduled when finite-I banana is actually needed) thread `I` into
         `_boozer_weighted_residual` as `G_eff = G + iota*I`, and propagate the
         (iota, G) rank-one chain-rule into the gradient/adjoint path
         (`make_traceable_solved_state_value_and_grad`). Cross-validate against the
         reference `boozer_finite_current.py` on a finite-I fixture.
5. Banana objective contract decision (real gap, not a "defect").
   - [ ] Decide: complete `banana_local_value` or document it as a subset.
   - [ ] If completing: port the always-on **self-envelope** term
         (`_self_distance_window_pure` + Gonzalez–Maddocks `_global_radius_pure`,
         reference `self_intersect.py:368,451`), the **fold** term assembly
         (`CurveSurfaceGeodesicCurvature`, reference `banana_coil_solver.py:4472,4713`,
         which needs framed-curve binormal curvature in the frozen-spec geometry),
         and the single-stage **`PoloidalExtentFloor`** smooth-max soft floor
         (reference `poloidal_extent.py:264`).
   - [ ] If bounding: stamp `banana_local_value` as a named subset in its
         docstring, in `docs/banana_jax_port_implementation_plan.md`, and in
         `docs/jax_parity_manifest.md`, listing the omitted always-on terms so no
         caller assumes production-objective equivalence.

### Phase P2 — Solver / diagnostic fidelity (bounded, non-objective)

6. Tracing within-step Levelset stop-localization (D2 + stop-event ordering).
   - [ ] In `src/simsopt_jax/core/tracing.py`, when a Levelset criterion changes
         sign within the accepted step, root-find via the existing
         `bracket_root_jax` sub-step machinery on `classifier_fn ∘ state_at_fraction`
         and record the interpolated crossing state (reference
         `/Users/suhjungdae/code/columbia/simsopt-surrogate/src/simsoptpp/tracing.cpp:461-518`).
   - [ ] Handle the start-of-step `value_last < 0` case.
   - [ ] Drop step crossings with `t_hit > stop_time`; select the stop event by
         **minimum time**, not minimum index, when ≥2 criteria fire.
   - [ ] Update adapter `src/simsopt_jax_adapters/field/tracing.py:689-715` to route
         the levelset criterion to the localizing path.
   - [ ] Fix the cheap convention nit alongside (`iter >= max_iter` boundary,
         tracing-4).
7. iota-collapse early-exit guard (solveopt-1).
   - [ ] Add `iota_collapse_fraction` / `iota_reference` to
         `BoozerSurfaceJAX.run_code` and the LBFGS stage, emitting `iota_collapsed`
         (reference `boozersurface.py:173-225`, `single_stage_geometry.py:1177-1182`,
         env `IOTA_COLLAPSE_REJECT_FRACTION=0.3`), OR document that the JAX
         single-stage relies on the objective-level iota penalty instead.

### Phase P3 — Test integrity (no production-kernel behavior change)

8. Banana geometry-kernel tests (D5).
   - [ ] Add hand-computed closed-form unit tests for `banana_poloidal_extent_pure`
         and `banana_projected_ellipse_width_pure` (mirror the self-distance/pack
         closed-form tests in `tests/core/test_banana.py:583-601`).
   - [ ] Stop routing the driver geometry-term parity through the shared
         `banana.py` kernels in `tests/test_banana_jax_specs.py:348-466` (or add an
         independent oracle side).
9. qfm degenerate-boundary tests (qfmflux-1).
   - [ ] There is no existing `*_matches_cpp_boundary_contract` selector in
         `tests/geo/test_qfmsurface_jax.py` at `4fa5e61c2`; add explicit
         degenerate-boundary coverage for the QFM residual/label cases, or, if the
         audit refers to a different file, first locate that selector and rename it
         there. The test must record any JAX `0.0`/`inf` vs C++ `nan` divergence
         explicitly and must not hide it behind `importorskip`. Do **not** change
         the production kernel.
10. GSCO oscillation parity (magnets-1).
    - [ ] Either reproduce the compiled C++ GSCO behavior (`opt_ind+nLoops` literal,
          vs the JAX `(opt_ind+nLoops)%(2*nLoops)`) for byte-parity, or keep the
          corrected index and add a **non-degenerate** GSCO parity test (nonzero
          A,b; multiple loops; run to oscillation) plus an intentional-divergence note.
11. Higher-order surface `_lin` anchor (D6) + Boozer composition (boozerfield-1/2).
    - [ ] Add a central-difference anchor of `gammadash1dash1_lin` vs d/dφ of the
          C++-validated `gammadash1_lin` (or compare against the `.def`-bound C++
          `surface.gammadash1dash1dash1_lin` on a paired grid) at
          `tests/geo/test_surface_fourier_jax.py:1334-1343`.
    - [ ] Add the `InterpolatedBoozerFieldJAX` composition test vs
          `sopp.InterpolatedBoozerField`.

### Phase P4 — Cosmetic

12. - [ ] Rename/redocument the mislabeled `tests/field/test_biotsavart_jax_parity.py`
          (the real cross-boundary gate is the sibling
          `test_biotsavart_jax.py::TestBiotSavartJaxCppParity`).
    - [ ] Fix the axishull PI-controller docstring overclaim (axishull-2).

## Validation Plan

- [ ] **P0-1 (fail-loud):** new unit test — constructing/using a
      `CurveFilamentSpec` with `frame_kind="surface_tangent"` (before P0-2) raises;
      an unknown `frame_kind` raises at spec-construction; `centroid`/`frenet`
      unchanged.
- [ ] **P0-2 (frame port):** cross-validate `rotated_surface_tangent_frame`
      (+`_dash`, torsion, binormal-curvature) on a **non-planar** curve against the
      sibling reference implementation in
      `/Users/suhjungdae/code/columbia/simsopt-surrogate` to ~1e-12 (run the
      reference side under
      `/Users/suhjungdae/code/columbia/simsopt-surrogate/.conda-env/bin/python3.11`).
      Add a `FramedCurveSurfaceTangentJAX` value+VJP parity test to
      `tests/geo/test_framedcurve_jax_wrappers_item18.py`.
- [ ] **P1-4a (fail-closed I):** test that the banana single-stage facade raises on
      `I≠0` and runs unchanged on `I=0`.
- [ ] **P1-4b (finite-I, if done):** finite-I residual + gradient parity vs
      reference `boozer_finite_current.py` on a seeded fixture.
- [ ] **P1-5 (banana contract):** if bounding, a doc/test that asserts the subset
      manifest lists every always-on reference term not in `banana_local_value`
      (extend `tests/docs/test_banana_parity_coverage_manifest.py`).
- [ ] **P2-6 (tracing):** Levelset-on-3D-field loss test asserting the recorded
      loss point lies on the stopping surface (not one step past it); regression
      that phi-plane Poincaré dots are unchanged.
- [ ] **P3-8/9/10/11:** new tests fail against the *current* kernels only where a
      real divergence exists (they should pass for correct kernels, catch a seeded
      sign/formula mutation) — i.e. confirm they are non-tautological by mutation.
- [ ] **Suite gate:** `tests/geo/test_framedcurve_jax*.py`,
      `tests/core/test_banana.py`, `tests/test_banana_jax_specs.py`,
      `tests/field/test_*tracing*` green on the CPU lane; full GPU suite only when a
      cache dir is set (`JAX_COMPILATION_CACHE_DIR`).
- [ ] No regression in the existing genuine parity tests for the 9 FAITHFUL lanes.

## Risks and Mitigations

- Risk: P0-2 frame port introduces a subtle projection/renormalization difference
  from the reference (e.g. tangent normalization order).
  Mitigation: transcribe verbatim from reference `framedcurve.py:865-897`; gate on
  the 1e-12 non-planar cross-validation before wiring into any objective.
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
  Mitigation: prove each new test catches a seeded mutation of the kernel under
  test before merging.
- Risk: scope creep into completing the full banana objective (P1-5 "complete"
  branch) when "bound + document" is sufficient for current needs.
  Mitigation: make the contract decision (P1-5 first checkbox) an explicit gate
  with the user before porting self-envelope/fold/floor.

## Completion Criteria

- [ ] No JAX path can silently compute a centroid frame when `surface_tangent` is
      requested (P0-1 merged with tests).
- [ ] `surface_tangent` frame produces reference-matching geometry+gradients to
      1e-12, or is explicitly NotImplemented-guarded with a tracked follow-up
      (P0-2).
- [ ] `_boozer_weighted_residual` cannot run with `I≠0` unless the finite-I
      chain-rule path is implemented and parity-checked (P1-4).
- [ ] `banana_local_value` is either complete vs the production Stage-2 objective
      or documented as a named subset in code + manifest (P1-5).
- [ ] Tracing lost-line exit points match the reference within the documented
      ≤1-step bound; stop-event selection is by minimum time (P2-6).
- [ ] The four overstated tests (banana driver, qfm degenerate, GSCO, surface
      `_lin`) are replaced/augmented with mutation-proven genuine checks (P3).
- [ ] Audit memory `project_jax_port_faithfulness_audit_2026_06_25` updated to
      mark each confirmed defect resolved or explicitly deferred.

## Open Questions

- Banana objective contract (P1-5): **complete** the always-on self-envelope +
  fold + floor terms now, or **bound + document** as a subset? Needs the user's
  call — depends on whether near-term JAX banana runs need production-equivalent
  objective values or only the vacuum geometry/penalty subset.
- Finite-I banana (P1-4b): is finite-enclosed-current single-stage on the roadmap
  soon enough to justify porting the (iota,G) chain-rule now, or is fail-closed
  sufficient until then?
- GSCO (P3-10): is byte-parity with the C++ `opt_ind+nLoops` precedence quirk
  required, or is the corrected index the intended behavior (keep + document)?
- Does P2-7 (iota-collapse guard) need to be wired into `run_code`, or is the
  objective-level iota penalty already considered sufficient steering for the JAX
  single-stage search?
