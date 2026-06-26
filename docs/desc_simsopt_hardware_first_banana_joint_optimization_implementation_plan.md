# DESC/SIMSOPT Hardware-First Banana Joint Optimization Implementation Plan

## Purpose

This file defines the implementation plan for a full hardware-first banana-coil co-design workflow:

```text
DESC:
  optimize equilibrium + banana coils jointly

SIMSOPT / existing repo:
  seed generation, artifact compatibility, Boozer/Poincare checks, hardware oracle
```

The plan is intended to be executable by another engineer or agent without rediscovering the current repo boundaries. It separates confirmed facts from assumptions and keeps final feasibility tied to direct SIMSOPT/CAD artifact validation rather than DESC objective values alone.

## Goals

- Implement a hardware-first workflow that starts from hardware constraints and jointly solves for banana coil geometry, coil currents, and plasma boundary/equilibrium variables.
- Use DESC as the core joint free-boundary optimizer with `BoundaryError` or `VacuumBoundaryError`, not `QuadraticFlux`, for true equilibrium-plus-coil optimization.
- Preserve the existing SIMSOPT banana artifact ecosystem for seed generation, coil group semantics, export compatibility, Poincare/Boozer validation, and hardware oracle checks.
- Add typed, tested conversion contracts between SIMSOPT banana artifacts and DESC equilibrium/coil objects.
- Produce promotion-ready result payloads that distinguish search-time steering, DESC solve status, SIMSOPT physics validation, and final hardware oracle status.

## Non-Goals

- Do not replace the existing SIMSOPT single-stage banana optimizer in one large rewrite.
- Do not treat DESC coil-distance or keepout objective values as a replacement for final CAD/contact/hardware oracle validation.
- Do not use `QuadraticFlux` for joint equilibrium-plus-coil optimization.
- Do not retarget proxy or VF coils as a side effect of adding DESC.
- Do not upstream HBT-specific hardware policy into generic DESC APIs unless the generic abstraction is proven first.

## Current Context

- Current checkout: `/Users/suhjungdae/code/columbia/simsopt-surrogate` on branch `surrogate-confinement-v2`, with pre-existing dirty source/test files. This plan is an additive docs file.
- Current DESC checkout: `/Users/suhjungdae/code/opensource/DESC`.
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py` currently describes itself as "single-stage Boozer/quasi-symmetry optimization from a Stage 2 seed" and consumes VMEC/wout-style equilibrium inputs plus SIMSOPT artifacts.
- `examples/single_stage_optimization/banana_opt/single_stage_banana_geometry_mode.py` already owns banana geometry modes: `shared_symmetry`, `materialized_cws`, and `free_xyz`; it preserves coil ordering when replacing banana coils in `BiotSavart`.
- `examples/single_stage_optimization/banana_opt/single_stage_geometry.py` already separates `search_hardware_status` from `artifact_hardware_status`, with final artifact status driving the hardware constraint result fields.
- `examples/single_stage_optimization/banana_opt/hardware_keepout.py` contains SIMSOPT-side keepout machinery including `CurveHardwareKeepout`, `CurveVesselEnvelopeKeepout`, hardware metadata, and hardware keepout loading.
- `examples/single_stage_optimization/banana_opt/hardware_contracts.py` owns current HBT hardware constants, while `examples/single_stage_optimization/banana_opt/hardware_constraint_schema.py` owns current hardware constraint names, thresholds, artifact field names, and payload status fields.
- `examples/single_stage_optimization/banana_opt/artifact_contracts.py` and `examples/single_stage_optimization/banana_opt/stage2_single_stage_handoff.py` already enforce Stage 2 artifact metadata such as `STAGE2_BS_SHA256`, coil counts, `COIL_GROUPS`, current metadata, and hardware artifact fields.
- DESC has coil primitives including `FourierRZCoil`, `FourierXYZCoil`, `CoilSet`, and `MixedCoilSet`.
- DESC has joint/free-boundary objective primitives including `BoundaryError`, `VacuumBoundaryError`, `PlasmaCoilSetMinDistance`, and `LinkingCurrentConsistency`.
- DESC `QuadraticFlux` is explicitly a fixed-equilibrium coil optimization objective. DESC's optimizer rejects `QuadraticFlux` when an `Equilibrium` is in the optimized things.
- DESC supports using saved DESC or VMEC equilibrium output as an initial guess through its existing equilibrium initialization path.

## Rationale

The desired workflow is hardware-first co-design, not coil polish against a fixed surface. The optimization target is:

```text
hardware specs -> allowed coils + allowed plasma boundary -> joint equilibrium/coil solution
```

DESC is the right core for the joint solve because its free-boundary objectives are built around an optimized equilibrium and an external magnetic field. SIMSOPT is the right shell for the banana workflow because the current repo already owns banana-specific CWS geometry, current/group contracts, hardware keepout metadata, artifact formats, and final validation.

Design-it-twice comparison:

- Option A: implement everything inside the current SIMSOPT single-stage script. This minimizes short-term conversion work but keeps the architecture tied to a large script and does not produce a clean DESC free-boundary core.
- Option B: implement a DESC-only banana optimizer. This gives a clean mathematical core but loses current artifact compatibility, hardware oracle integration, and SIMSOPT validation paths.
- Selected option: implement an additive bridge. DESC owns the joint optimizer; SIMSOPT owns seed/artifact/hardware validation. This minimizes replacement risk while creating a path to a true hardware-first solver.

Risk tier:

- The bridge and runner are Tier 2 changes because they introduce new module boundaries and cross-repo contracts.
- Any new exported DESC objective, coil class, or optimizer-facing API is Tier 3 and must include caller inventory, compatibility tests, and rollback notes before being treated as complete.

## Assumptions

- The first implementation can use DESC `FourierXYZCoil.from_values` to represent sampled SIMSOPT banana coils before adding a native DESC banana-CWS curve.
- The first full solve should support vacuum or low-beta free-boundary optimization before finite-beta production runs.
- Existing SIMSOPT `hardware_contracts.py` and `hardware_constraint_schema.py` remain the SSOT for hardware thresholds and result field names; new DESC-joint config may reference them but must not copy their constants into a second schema.
- Existing SIMSOPT hardware keepout JSON/SDF artifacts remain the source for hardware geometry, while final promotion remains tied to the CAD/contact oracle.
- Existing SIMSOPT Poincare/Boozer validation scripts can consume exported DESC-optimized coil artifacts after round-trip conversion.
- VMEC/wout-derived seeds are acceptable initial guesses for DESC, but a filename containing `desc` is not proof of DESC runtime optimization.

## Implementation Plan

1. Establish contracts and ownership
   - [ ] Create `examples/single_stage_optimization/DESC_JOINT/README.md` documenting the selected architecture: DESC joint solver, SIMSOPT seed/export/validation shell.
   - [ ] Define a typed result schema for hardware-first joint runs with separate sections for `input_contract`, `desc_solve_status`, `search_hardware_status`, `artifact_hardware_status`, `physics_validation_status`, and `promotion_status`.
   - [ ] Define the failure policy for missing artifact binding fields such as coil group metadata, current signs, NFP, symmetry, source checksums, and hardware keepout provenance.
   - [ ] Add a caller inventory for affected SIMSOPT entrypoints: Stage 2 solver, current single-stage runner, goal-mode comparison runner, frontier trajectory tooling, and Poincare/Boozer validation scripts.
   - [ ] Add a caller inventory for affected DESC surfaces: coil classes, objective assembly, optimizer calls, and free-boundary examples/tests.

2. Extend the existing hardware-contract SSOT for DESC joint runs
   - [ ] Add `examples/single_stage_optimization/DESC_JOINT/hardware_first_schema.md` with required runner inputs: vessel/port keepout sources, coil length limits, coil-coil spacing, coil-plasma spacing, curvature limits, current limits, width limits, TF/proxy/VF policy, and final oracle path.
   - [ ] In `hardware_first_schema.md`, map every threshold and status field back to `banana_opt/hardware_contracts.py` or `banana_opt/hardware_constraint_schema.py`; do not introduce a second set of hardware constants.
   - [ ] Add `examples/single_stage_optimization/banana_opt/desc_joint_hardware_spec.py` with frozen typed dataclasses for DESC-joint runner configuration. The dataclasses should reference the existing hardware schema and carry paths/provenance, not duplicate threshold values.
   - [ ] Add a loader that reads JSON hardware specs and existing `hardware_keepout.json`/SDF metadata without parsing Markdown as configuration.
   - [ ] Add validation that fails loudly when hardware geometry provenance is missing, stale, or not bound to the GLB/SDF artifact used by the run.
   - [ ] Add unit tests proving DESC-joint config resolves the current hard constraints through the existing schema: max coil length, min coil-coil spacing, max curvature, coil-plasma spacing, banana current cap, TF current cap, width bounds, and keepout source identity.

3. Build SIMSOPT to DESC conversion
   - [ ] Add `examples/single_stage_optimization/banana_opt/desc_bridge/coil_export.py` to convert SIMSOPT coils into DESC coil objects.
   - [ ] Preserve coil groups explicitly: TF, banana, proxy, VF, and any loaded auxiliary groups.
   - [ ] Preserve current values in Amperes, current signs, coil names, ordering, NFP, symmetry, and banana pack grouping.
   - [ ] Convert banana CWS curves by sampling dense Cartesian coordinates and fitting DESC `FourierXYZCoil` objects at a configured Fourier order.
   - [ ] Record coordinate basis, sample count, fit residual, length delta, curvature delta, min-distance delta, and field-sample delta in a conversion report.
   - [ ] Add tests that fail if CWS materialization or free-XYZ conversion changes coil ordering or current signs.
   - [ ] Add tests that compare SIMSOPT and DESC magnetic fields at a fixed set of points within a tolerance chosen from observed conversion residuals.

4. Build DESC to SIMSOPT conversion
   - [ ] Add `examples/single_stage_optimization/banana_opt/desc_bridge/coil_import.py` to convert DESC optimized coils back to SIMSOPT-compatible coils or MAKEGRID artifacts.
   - [ ] Preserve the original coil group manifest during import so downstream validation does not infer groups from heuristics.
   - [ ] Add round-trip tests: SIMSOPT -> DESC -> SIMSOPT should preserve sampled coordinates, current signs, lengths, min distances, and group labels within explicit tolerances.
   - [ ] Add artifact metadata fields for DESC optimizer version, DESC commit, conversion residuals, and source artifact checksums.
   - [ ] Add a fail-closed check that prevents validation from treating a DESC-exported artifact as hardware-clean until the existing hardware oracle has run.

5. Seed DESC equilibrium and plasma boundary state
   - [ ] Add `examples/single_stage_optimization/banana_opt/desc_bridge/equilibrium_seed.py` to create or load the DESC equilibrium seed from a DESC file or VMEC/wout output.
   - [ ] Preserve NFP, handedness, stellarator symmetry, angular convention, major/minor scale, and LCFS mode truncation choices in a seed report.
   - [ ] Add a fixed-equilibrium smoke path that evaluates DESC coil objectives against the seeded equilibrium before enabling joint optimization.
   - [ ] Add tests comparing seeded LCFS samples against the source VMEC/SIMSOPT surface at deterministic grid points.
   - [ ] Add a guard that marks the seed provenance as `vmec_wout`, `desc_h5`, or `simsopt_surface`; do not infer DESC provenance from filenames.

6. Implement DESC objective assembly
   - [ ] Add `examples/single_stage_optimization/banana_opt/desc_bridge/objective_factory.py` for assembling DESC objectives from typed config.
   - [ ] Implement a fixed-equilibrium coil-polish objective mode using `QuadraticFlux`, `LinkingCurrentConsistency(eq_fixed=True)`, coil length, coil curvature, coil-coil distance, and plasma-coil distance.
   - [ ] Implement the true joint mode using `BoundaryError` or `VacuumBoundaryError`, `LinkingCurrentConsistency(eq_fixed=False)`, `PlasmaCoilSetMinDistance(eq_fixed=False)`, coil length, coil curvature, coil-coil distance, and hardware keepout terms.
   - [ ] Add a test that joint mode rejects `QuadraticFlux` before reaching the DESC optimizer.
   - [ ] Add objective scaling metadata so reported residuals can be compared across fixed-equilibrium polish, vacuum joint solve, and finite-beta joint solve.
   - [ ] Add chunk-size configuration only for compute/memory infrastructure concerns; do not expose algorithm-choice knobs as user-facing switches unless they are necessary for external operational constraints.

7. Add generic in-loop hardware objectives for production
   - [ ] For the first Lane A smoke test, existing DESC coil/plasma distance objectives plus SIMSOPT post-validation are acceptable because the lane is a conversion and fixed-equilibrium polish check, not a hardware-first production claim.
   - [ ] For Lane B/C production, hardware constraints must influence the search in-loop: either through differentiable DESC objectives/constraints or through a fail-closed constrained outer loop that rejects and records violating candidates before promotion.
   - [ ] If direct hardware keepout must be differentiable inside DESC, add generic objective support in DESC rather than HBT-specific names.
   - [ ] Candidate DESC implementation: `desc/objectives/_hardware.py` with a typed signed-distance source object and coil-to-SDF objective.
   - [ ] Export new objective(s) through `desc/objectives/__init__.py` only after unit tests and API-evolution notes exist.
   - [ ] Add DESC tests with synthetic SDF/point-cloud fixtures proving sign convention, gradient direction, chunking behavior, and fixed-vs-moving coil semantics.
   - [ ] Keep the SIMSOPT CAD/contact oracle as the final promotion gate even if DESC gets differentiable keepout steering.

8. Implement the DESC joint runner
   - [ ] Add `examples/single_stage_optimization/DESC_JOINT/run_desc_joint_banana.py`.
   - [ ] Support modes: `fixed_equilibrium_polish`, `vacuum_joint`, and `finite_beta_joint`.
   - [ ] Load hardware spec, SIMSOPT seed artifact, DESC/VMEC equilibrium seed, and run output root from explicit CLI arguments.
   - [ ] Emit a preflight JSON before optimization with resolved paths, source checksums, coil group counts, current signs, NFP, symmetry, hardware source metadata, and objective stack.
   - [ ] Run fixed-equilibrium polish as the first executable stage and require it to pass round-trip validation before joint mode can be promoted.
   - [ ] Run true joint mode with DESC `BoundaryError` or `VacuumBoundaryError`, never with `QuadraticFlux`.
   - [ ] Write `desc_result.json`, `desc_coils.h5` or equivalent DESC artifact, exported SIMSOPT coil artifact, conversion report, and validation manifest.

9. Integrate SIMSOPT validation and final oracle
   - [ ] Add a validation wrapper that accepts the exported DESC-optimized artifact and runs the existing SIMSOPT Poincare/Boozer validation path.
   - [ ] Reuse existing result fields where possible: `HARDWARE_CONSTRAINTS_OK`, `HARDWARE_CONSTRAINT_VIOLATIONS`, `BEST_FEASIBLE_HARDWARE_CONSTRAINTS_OK`, and final feasibility fields.
   - [ ] Keep `search_hardware_status` and `artifact_hardware_status` separate in DESC joint run payloads.
   - [ ] Add a final oracle gate that requires direct loaded-artifact hardware/contact evidence before setting promotion status to pass.
   - [ ] Add a report generator that lists DESC objective success, SIMSOPT physics checks, final hardware oracle status, and artifact paths separately.

10. Add tests and regression fixtures
   - [ ] Add focused SIMSOPT tests under `tests/geo/test_desc_bridge_conversion.py` for coil conversion, current sign preservation, group preservation, and round-trip metadata.
   - [ ] Add focused SIMSOPT tests under `tests/geo/test_desc_joint_runner_contracts.py` for CLI preflight, mode selection, failure policy, and result schema.
   - [ ] Add DESC-side unit tests for any new generic hardware objective before using it in the joint runner.
   - [ ] Add a small synthetic integration fixture with one simple equilibrium and a small coil set to prove the end-to-end fixed-equilibrium polish path.
   - [ ] Add a gated, non-default HBT smoke fixture for the first low-resolution vacuum joint solve.
   - [ ] Add regression tests that prevent `QuadraticFlux` from appearing in joint mode objective stacks.

11. Add execution lanes
   - [ ] Lane A: fixed-equilibrium DESC coil polish from a known SIMSOPT banana artifact.
   - [ ] Lane B: vacuum joint DESC solve with banana coil geometry and plasma boundary moving together.
   - [ ] Lane C: finite-beta joint DESC solve after Lane B has a passing validation artifact.
   - [ ] Lane D: hardware-space exploration using the same DESC joint runner but multiple hardware spec files.
   - [ ] Record lane outputs in an inventory file with absolute artifact paths, source hashes, run mode, objective stack, validation status, and promotion status.

12. Performance and memory hardening
   - [ ] Add low-resolution defaults for local smoke tests and explicit production-resolution configs for real runs.
   - [ ] Record Biot-Savart chunk sizes, DESC source/eval grids, coil Fourier order, sample counts, and memory-relevant settings in every run payload.
   - [ ] Add timing sections for conversion, objective build, first objective evaluation, first gradient evaluation, optimizer iterations, export, and validation.
   - [ ] Add a benchmark smoke command that runs one objective and gradient evaluation without launching a long optimization.
   - [ ] Keep high-cost JIT/gradient work behind explicit runner flags or production configs, not implicit test defaults.

13. Documentation and handoff
   - [ ] Update `docs/desc_banana_single_stage_feasibility_report_2026-06-26.md` after the first implementation slice lands.
   - [ ] Add `examples/single_stage_optimization/DESC_JOINT/README.md` with a minimal command sequence for Lane A and Lane B.
   - [ ] Add a troubleshooting section for current sign mismatch, NFP/symmetry mismatch, LCFS seed mismatch, failed hardware provenance, and failed oracle status.
   - [ ] Add a result interpretation guide that clearly separates DESC solve success from physics validation success and final hardware promotion.

## Validation Plan

Current-checkout note: `tests/geo/test_desc_bridge_conversion.py` and `tests/geo/test_desc_joint_runner_contracts.py` are planned files and do not exist yet. Run their commands only after Phase 10 adds them.

- [ ] `git -C /Users/suhjungdae/code/columbia/simsopt-surrogate diff --check`
- [ ] `cd /Users/suhjungdae/code/columbia/simsopt-surrogate && PYTHONNOUSERSITE=1 ./.conda-env/bin/python -m pytest tests/geo/test_desc_bridge_conversion.py -q`
- [ ] `cd /Users/suhjungdae/code/columbia/simsopt-surrogate && PYTHONNOUSERSITE=1 ./.conda-env/bin/python -m pytest tests/geo/test_desc_joint_runner_contracts.py -q`
- [ ] `cd /Users/suhjungdae/code/columbia/simsopt-surrogate && PYTHONNOUSERSITE=1 ./.conda-env/bin/python -m pytest tests/geo/test_hardware_keepout_winding_frame.py tests/geo/test_stage2_formulation_audit_fixes.py -q`
- [ ] `cd /Users/suhjungdae/code/opensource/DESC && python -m pytest tests/test_objective_funs.py -k "BoundaryError or VacuumBoundaryError or QuadraticFlux or LinkingCurrentConsistency" -q`
- [ ] `cd /Users/suhjungdae/code/opensource/DESC && python -m pytest tests/test_optimizer.py -k "optimize_coil_currents" -q`
- [ ] Run SIMSOPT -> DESC -> SIMSOPT round-trip on a synthetic fixture and compare coordinate, current, length, distance, and field-sample residuals.
- [ ] Run Lane A fixed-equilibrium polish and confirm exported artifact can be loaded by existing SIMSOPT validation.
- [ ] Run Lane B low-resolution vacuum joint solve and confirm result payload separates DESC success, physics validation, and hardware oracle status.
- [ ] Run final hardware/contact oracle on any candidate before marking it promotion-ready.

## Risks and Mitigations

- Risk: SIMSOPT and DESC coordinate/sign conventions silently diverge.
  Mitigation: Require deterministic surface samples, coil samples, current signs, NFP, symmetry, and field samples in every conversion test and run report.

- Risk: Hardware steering is mistaken for final buildability.
  Mitigation: Preserve `search_hardware_status` versus `artifact_hardware_status` and require direct loaded-artifact oracle evidence for promotion.

- Risk: DESC `QuadraticFlux` is accidentally used in a joint equilibrium-plus-coil solve.
  Mitigation: Add an objective-factory test and runner preflight that reject `QuadraticFlux` in joint mode before optimizer construction.

- Risk: A native DESC banana coil class becomes HBT-specific and shallow.
  Mitigation: Start with `FourierXYZCoil.from_values`; only add a native class if conversion residuals or performance justify it, and keep generic API boundaries in DESC.

- Risk: The current large SIMSOPT script absorbs DESC-specific complexity.
  Mitigation: Add a new `DESC_JOINT` runner and `banana_opt/desc_bridge` package instead of expanding `single_stage_banana_example.py` first.

- Risk: Long joint solves fail late because seed artifacts are malformed.
  Mitigation: Add preflight checksum, group, current, NFP, symmetry, hardware-provenance, and conversion-residual gates before optimization.

- Risk: Differentiable hardware SDF in DESC duplicates SIMSOPT hardware policy.
  Mitigation: Keep DESC objective generic and preserve SIMSOPT/CAD as the final policy layer.

- Risk: Tests become mock-only wiring checks.
  Mitigation: Require observable state assertions: coordinates, currents, group labels, objective stack contents, result payload fields, and loadable exported artifacts.

## Completion Criteria

- [ ] A hardware-first spec can be loaded and validated without reading Markdown or relying on environment variables.
- [ ] A known SIMSOPT banana artifact can be converted to DESC coils and back with explicit residuals and passing tests.
- [ ] A DESC equilibrium seed can be created from a DESC or VMEC/wout source with provenance and LCFS parity checks.
- [ ] Lane A fixed-equilibrium polish runs end-to-end and exports a SIMSOPT-loadable artifact.
- [ ] Lane B low-resolution vacuum joint solve runs end-to-end with `BoundaryError` or `VacuumBoundaryError`.
- [ ] Result payloads separately report DESC solve status, SIMSOPT physics validation, search hardware status, artifact hardware status, and promotion status.
- [ ] No joint-mode objective stack includes `QuadraticFlux`.
- [ ] Final promotion requires existing SIMSOPT Poincare/Boozer validation plus direct hardware oracle evidence.
- [ ] Documentation includes command examples, artifact paths, validation commands, and interpretation rules.

## Open Questions

- Should generic DESC hardware keepout objectives live in DESC proper, or should the first implementation keep them in `simsopt-surrogate` until the abstraction is proven?
- Which seed should be the first Lane A artifact: an existing known-good single-stage Boozer artifact, a Stage 2 finite-build artifact, or a synthetic fixture only?
- Should Lane B start strictly vacuum, or should the first joint solve include finite-beta pressure balance from the beginning?
- What exact artifact format should be the canonical DESC-to-SIMSOPT export: DESC HDF5 plus conversion sidecar, MAKEGRID file, SIMSOPT JSON, or all three?
- What tolerance should define conversion success for magnetic field samples near the plasma boundary?
- Which CAD/contact oracle output should be treated as the promotion SSOT for this runner?
- Should proxy and VF coils be fixed in the first joint solve, or should they be included as separate optimized groups with explicit bounds?
