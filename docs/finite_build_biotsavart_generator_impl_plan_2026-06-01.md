# Finite-Build BiotSavart Generator Implementation Plan

## Purpose

This plan defines the work to add an examples-side utility that converts a single-filament HBT banana `BiotSavart` artifact into a finite-build banana-coil `BiotSavart` artifact. It is intended as a convenience/export tool, not a change to the single-stage optimizer or finite-current physics model.

## Goals

- Add a deterministic CLI/script that loads an existing Stage 2 or single-stage `BiotSavart` JSON and writes a finite-build `BiotSavart` JSON.
- Preserve TF, proxy, and VF coils unchanged while replacing the banana coil bundle with a multi-filament approximation.
- Split the banana current across the generated filaments so total banana current is preserved unless an explicit override is supplied.
- Record enough metadata or naming context that downstream users can tell single-filament and finite-build artifacts apart.
- Add focused tests for coil counts, current conservation, output naming, and compatibility with existing finitebuild primitives.

## Non-Goals

- Do not change `src/simsopt/geo/finitebuild.py` unless a real bug is found in the existing generic multifilament implementation.
- Do not make finite-build coils part of the optimization loop in this task.
- Do not change Boozer finite-current behavior, VMEC current modeling, or confinement scoring.
- Do not port stale `BoozerSurface(..., I=...)` behavior from `banana_drivers-main`.

## Current Context

- `simsopt-surrogate` already has generic finite-build primitives in `src/simsopt/geo/finitebuild.py`, especially `create_multifilament_grid(curve, numfilaments_n, numfilaments_b, gapsize_n, gapsize_b, rotation_order, rotation_scaling, frame)`.
- `tests/geo/test_finitebuild.py` already verifies filament geometry, coefficient derivatives, symmetry interaction, and use in a `BiotSavart` objective.
- A `BiotSavart` owns its coil list as `.coils`/`._coils`; Boozer-surface artifacts own the field as `.biotsavart`. The generator must normalize all supported loaded objects to one `BiotSavart` before partitioning coils.
- `banana_opt.coil_groups` is the current source of truth for `COIL_GROUPS` manifests and legacy manifest resolution. `banana_opt.finite_current_profiles` provides default manifests for the supported vacuum, Wataru proxy/VF, and jhalpern30 proxy/VF profiles.
- `banana_drivers-main` has a narrow utility at `src/banana_drivers/scripts/generate_biotsavart_finitebuild.py` that:
  - loads a `BiotSavart` JSON,
  - partitions coils by fixed HBT indices,
  - replaces the first banana curve with a multi-filament grid,
  - divides banana current by `numfilaments_n * numfilaments_b`,
  - applies symmetries to the finite-build banana filaments,
  - writes a sibling file whose name replaces `biotsavart` with `biotsavart_finitebuild`.
- `simsopt-surrogate` does not define project console scripts in `pyproject.toml`; examples are normally run as direct Python scripts under `examples/single_stage_optimization`.
- Current `simsopt-surrogate` Stage 2/single-stage code carries richer coil-group metadata than `banana_drivers-main`, so hard-coded coil indices should be avoided.

## Rationale

The lowest-risk approach is to add a small examples-side generator that reuses the existing `simsopt.geo.create_multifilament_grid` primitive. This keeps upstream/core finitebuild code untouched and treats finite-build export as a post-processing artifact conversion. The utility should use current `simsopt-surrogate` coil partition metadata when present, with a conservative fallback for legacy HBT artifacts.

Porting the `drivers-main` script literally would work only for the fixed 20 TF / 10 banana / 1 proxy / 20 VF layout and would miss the newer `COIL_GROUPS`, finite-current-profile, strict-vacuum, and jhalpern30 artifact contracts. The implementation should keep the useful conversion behavior but adapt the partitioning and metadata to this repo.

## Assumptions

- The first banana coil in the resolved banana partition is the master curve whose symmetry copies represent the banana bundle.
- The finite-build artifact is primarily for inspection, field evaluation, and downstream export, not for resuming single-stage optimization.
- No current repo source of truth was found for finite-build filament counts or gap sizes in `banana_opt.hardware_contracts`; the utility should require explicit values or define documented local defaults near the CLI parser.
- If loaded artifacts lack `COIL_GROUPS`, fallback partitioning is acceptable only through `FiniteCurrentProfile.build_default_coil_groups_manifest()` and only when the loaded coil count matches the selected profile total.

## Implementation Plan

1. Define the script location and public CLI.
   - [ ] Add `examples/single_stage_optimization/generate_finitebuild_biotsavart.py` or `examples/single_stage_optimization/banana_opt/finitebuild_export.py` plus a thin script wrapper.
   - [ ] Use `import_provenance.configure_local_simsopt_imports(__file__)` so the script reliably imports the local checkout.
   - [ ] Support required input `biotsavart_file`.
   - [ ] Support `--output` and default sibling output naming that replaces `biot_savart` or `biotsavart` with `biot_savart_finitebuild`.
   - [ ] Support explicit finite-build controls: `--numfilaments-n`, `--numfilaments-b`, `--gapsize-n`, `--gapsize-b`, `--rotation-order`, and `--frame`.
   - [ ] Support `--banana-current-A` as an override in amperes; avoid kA-only CLI semantics in new code.

2. Resolve coil partitions from current artifact contracts.
   - [ ] Load the source artifact with `banana_opt.json_compat.load_boozer_finite_i` or `simsopt._core.load` as appropriate.
   - [ ] Normalize supported sources to a `BiotSavart`: raw `BiotSavart`, object carrying `.coils`, or object carrying `.biotsavart.coils`; reject unsupported objects with a clear error.
   - [ ] If a Stage 2 results JSON is supplied, reuse `banana_opt.stage2_single_stage_handoff.partition_loaded_stage2_coils`.
   - [ ] Add optional `--stage2-results` for metadata-backed partitioning.
   - [ ] Add optional `--finite-current-mode` for metadata-free fallback and resolve it through `banana_opt.finite_current_profiles.get_finite_current_profile`.
   - [ ] For no results JSON, build the fallback manifest from the selected `FiniteCurrentProfile` and require the loaded coil count to match `profile.default_total_coils`; do not infer 51-coil Wataru vs jhalpern30 provenance from count alone.
   - [ ] Validate that the banana partition is non-empty and contains a curve compatible with `create_multifilament_grid`.

3. Build the finite-build banana coil bundle.
   - [ ] Compute `nfil = numfilaments_n * numfilaments_b` and reject non-positive counts.
   - [ ] Resolve total banana current from the first banana coil unless `--banana-current-A` is supplied.
   - [ ] Create one base finite-build filament grid from the master banana curve.
   - [ ] Assign each base filament `total_banana_current / nfil`.
   - [ ] Apply stellarator symmetries with `apply_symmetries_to_curves` and `apply_symmetries_to_currents`.
   - [ ] Preserve TF, proxy, and VF coils in their original order.
   - [ ] Save a new `BiotSavart` with `TF + finitebuild banana + proxy + VF`.

4. Add artifact metadata and diagnostics.
   - [ ] Print a concise conversion summary: input path, output path, source coil counts, finite-build filament counts, and total banana current before/after.
   - [ ] If the source has a sidecar/results JSON, write a sibling finite-build metadata JSON with source path, source SHA256, source coil counts, output coil counts, filament settings, and current override status.
   - [ ] Ensure the output filename cannot silently overwrite the input path.
   - [ ] Add `--overwrite` for explicit replacement of an existing finite-build output.

5. Keep the utility isolated from optimization workflows.
   - [ ] Do not import the new generator from `single_stage_banana_example.py`.
   - [ ] Do not alter Stage 2 or single-stage objective construction.
   - [ ] Document that generated finite-build artifacts are export/evaluation artifacts and are not accepted as warm-start optimization seeds unless a future task adds that contract.

## Validation Plan

- [ ] Add `tests/geo/test_finitebuild_biotsavart_generator.py` with synthetic coil partitions and no external VMEC dependency.
- [ ] Test that a 30-coil vacuum-style input produces `20 + 10 * nfil` coils when no proxy/VF coils exist.
- [ ] Test that a 51-coil finite-current-style input produces `20 + 10 * nfil + 1 + 20` coils.
- [ ] Test that total banana current after finite-build conversion matches the source banana current within numerical tolerance.
- [ ] Test that `--banana-current-A` changes only the finite-build banana current total and does not mutate TF/proxy/VF currents.
- [ ] Test output naming for `biot_savart_opt.json`, `biotsavart.json`, and an explicit `--output`.
- [ ] Test that unsupported coil counts fail loudly unless `--stage2-results` supplies a valid manifest or a selected fallback profile matches the loaded coil count exactly.
- [ ] Add a unit/current conservation check: all banana finite-build filament currents are in amperes and sum to the source or override banana current.
- [ ] Add a field smoke check: the saved finite-build `BiotSavart` can be reloaded, `set_points` succeeds, and `B()` returns finite values with the expected shape.
- [ ] Run `python -m pytest tests/geo/test_finitebuild.py tests/geo/test_finitebuild_biotsavart_generator.py`.
- [ ] Run one manual smoke conversion on a small existing Stage 2 artifact and verify the saved JSON loads with the local `simsopt` checkout.

## Risks and Mitigations

- Risk: Hard-coded HBT coil indices could silently corrupt artifacts with nonstandard coil layouts.
  Mitigation: Prefer `COIL_GROUPS` metadata through `partition_loaded_stage2_coils`; allow metadata-free fallback only through an explicit `FiniteCurrentProfile` manifest whose total matches the loaded artifact.

- Risk: Current units could be confused because `drivers-main` used `--banana-current-ka`.
  Mitigation: Use `--banana-current-A` in the new utility and print current values in amperes.

- Risk: Generated finite-build artifacts could be mistaken for optimization warm-start seeds.
  Mitigation: Name outputs with `finitebuild`, write metadata, and document export-only scope.

- Risk: Frame or gap defaults may not match the historical HBT finite-build intent.
  Mitigation: Make finite-build geometry parameters explicit CLI flags; if defaults are added, keep them local and documented because no current hardware-contract constant owns them.

- Risk: Shared current optimizable objects from the source artifact may accidentally remain coupled to output filament currents.
  Mitigation: Build explicit new `Current` or `ScaledCurrent` objects for finite-build filaments and test that output current changes do not mutate the source object.

- Risk: A generated finite-build export could be mistaken for a proxy/VF-backed finite-current optimization seed.
  Mitigation: Preserve source provenance in metadata and keep the generator export/evaluation-only; do not synthesize proxy or VF field sources during conversion.

## Completion Criteria

- [ ] The generator script exists in the examples tree and can be run directly with the local checkout.
- [ ] The utility can convert known HBT vacuum and finite-current coil layouts.
- [ ] The saved finite-build `BiotSavart` loads successfully and has the expected coil order.
- [ ] Banana current is conserved by default and override behavior is tested.
- [ ] Tests cover metadata-backed and fallback partitioning behavior.
- [ ] Documentation in the script or a short README note states that finite-build output is an export/evaluation artifact, not a new physics model.

## Open Questions

- Should finite-build defaults be centralized in a new banana finite-build constants module, or should the CLI require all filament dimensions explicitly?
- Should the generator write only a `BiotSavart` JSON, or also a companion results/metadata JSON that downstream reports can ingest?
- Should finite-build export support jhalpern30 shared/unfixed VF current controls, or should it freeze all non-banana currents in the exported artifact?
- Should 51-coil metadata-free fallback require an explicit `--finite-current-mode`, or should the utility refuse 51-coil fallback entirely unless a results JSON provides provenance?
