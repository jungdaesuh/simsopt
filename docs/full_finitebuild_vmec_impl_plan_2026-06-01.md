# Full Finite-Build VMEC Implementation Plan

## Purpose

This plan defines the follow-on work needed to answer: "What does the fully manufactured machine look like if every coil family is finite-build?" It extends the current banana-only finite-build `BiotSavart` exporter into an all-family finite-build field artifact, then adds an optional free-boundary VMEC handoff through Simsopt/VMEC `mgrid` files.

## Goals

- Add a deterministic all-family finite-build export mode for TF, banana, proxy, and VF coils.
- Preserve the existing banana-only export path for compatibility and for narrow HBT banana finite-build studies.
- Preserve per-coil signed current by splitting each source coil current across its generated filaments.
- Write metadata that distinguishes banana-only, all-family single-group, and all-family grouped MGrid/VMEC artifacts.
- Optionally write a VMEC-compatible `mgrid` file from the full finite-build field.
- Provide enough recommended VMEC namelist values that a downstream free-boundary VMEC run uses the generated field correctly.

## Non-Goals

- Do not change single-stage optimization objectives or make finite-build coils part of the live optimizer in this task.
- Do not add plasma-current physics to Simsopt `BiotSavart`; plasma current remains a VMEC current-profile concern.
- Do not infer new proxy or VF coils that are not present in the source artifact.
- Do not change core Simsopt finite-build primitives unless implementation reveals a real upstream bug.
- Do not run long VMEC production sweeps as part of the implementation; add smoke-level and contract-level validation only.

## Current Context

- The current exporter lives in `examples/single_stage_optimization/banana_opt/finitebuild_export.py`.
- The current CLI wrapper is `examples/single_stage_optimization/generate_finitebuild_biotsavart.py`.
- The current plan is `docs/finite_build_biotsavart_generator_impl_plan_2026-06-01.md`.
- The current exporter partitions source coils into TF, banana, proxy, and VF groups, but only replaces the banana partition with finite-build filaments.
- `simsopt.geo.create_multifilament_grid(curve, numfilaments_n, numfilaments_b, gapsize_n, gapsize_b, rotation_order, rotation_scaling, frame)` is the official Simsopt finite-build primitive for generating a filament pack around one curve.
- `simsopt.field.apply_symmetries_to_curves` and `simsopt.field.apply_symmetries_to_currents` are official Simsopt helpers for symmetry expansion, including stellarator-symmetry current sign flips.
- For loaded Stage 2 artifacts, source coil lists are already physically expanded. An all-family converter should finite-build each loaded physical coil directly unless the source metadata explicitly marks a master-coil representation.
- `simsopt.field.MagneticField.to_mgrid(...)` writes a single-current-group `mgrid` file suitable for free-boundary VMEC with `vmec.indata.extcur[0] = 1.0`.
- `simsopt.field.MGrid.add_field_cylindrical(...)` supports multiple current groups. VMEC scales these groups with the `EXTCUR` array.
- Simsopt's own free-boundary VMEC example sets `lfreeb = True`, `mgrid_file`, `nzeta = nphi`, and `extcur[0] = 1.0` after writing `bs.to_mgrid(...)`.
- Simsopt VMEC docs state that, for free-boundary VMEC, MGrid `nphi` should be an integer multiple of VMEC `NZETA`. The first implementation should recommend the simple equality case, `NZETA = nphi`.
- STELLOPT/VMEC documentation states that free-boundary VMEC uses an `mgrid` file and requires one `EXTCUR` entry per current group.
- VMEC plasma current remains separate from external coil fields: `NCURR = 1`, `CURTOR`, `AC`, or a Simsopt current profile control the toroidal current profile.
- The current banana-only exporter finite-builds one HBT master banana curve and then applies Simsopt symmetries. It does not finite-build each loaded banana symmetry copy directly.

## Rationale

The safest implementation is to generalize the existing exporter instead of creating a second unrelated conversion path. The existing module already owns artifact loading, coil partitioning, metadata, output naming, and tests. Extending it keeps the finite-build artifact contract in one place.

For all-family finite-build, direct per-loaded-coil conversion is safer than reconstructing master coils and reapplying symmetry. The source artifact already encodes signs, ordering, proxy/VF presence, and any nonstandard partition supplied by Stage 2 metadata. Converting each loaded physical coil preserves that state and avoids accidental double symmetry expansion.

This does not mean the current banana-only helper should be refactored away immediately. Its current behavior is finite-build-first, then symmetry expansion. A direct per-symmetry-copy builder would be finite-build-after-symmetry; that must be treated as a distinct geometry path until a field/geometry parity test proves equivalence for the selected frame and rotation settings.

For VMEC, there are two useful export levels:

- A single-group `mgrid` from `BiotSavart.to_mgrid(...)`, where VMEC uses `EXTCUR[0] = 1.0`. This is the simplest faithful free-boundary handoff for the full finite-build field sampled on the MGrid.
- A grouped `mgrid` built with `MGrid.add_field_cylindrical(...)` once per family or circuit. This is needed only if downstream VMEC runs must scale TF, banana, proxy, or VF currents independently through `EXTCUR`.

## Assumptions

- Stage 2/single-stage `BiotSavart` artifacts list already-expanded physical coils, not one master coil per family.
- The existing `COIL_GROUPS` metadata, `partition_loaded_stage2_coils`, and finite-current profile fallback remain the source of truth for TF/banana/proxy/VF partitioning.
- One shared finite-build geometry setting for all families is acceptable for the first all-family export, but the CLI should be designed so per-family settings can be added without breaking existing arguments.
- `BiotSavart.to_mgrid(...)` is sufficient for the first VMEC handoff because it writes the net full finite-build field as one current group.
- Grouped MGrid export should be explicit because independent `EXTCUR` groups change the downstream VMEC control surface.

## Implementation Plan

1. Extend the export configuration and CLI.
   - [ ] Add an export scope option, for example `--finitebuild-scope banana-only|all-families`, defaulting to `banana-only` for compatibility.
   - [ ] Keep existing banana-only arguments unchanged: `--numfilaments-n`, `--numfilaments-b`, `--gapsize-n`, `--gapsize-b`, `--rotation-order`, `--frame`, and `--banana-current-A`.
   - [ ] Add metadata labels that distinguish `banana_only` and `all_families` output.
   - [ ] Add optional MGrid arguments: `--write-mgrid`, `--mgrid-output`, `--mgrid-nr`, `--mgrid-nz`, `--mgrid-nphi`, `--mgrid-rmin`, `--mgrid-rmax`, `--mgrid-zmin`, `--mgrid-zmax`, and `--mgrid-nfp`.
   - [ ] Add `--mgrid-grouping single|family` with `single` as the first supported default.
   - [ ] Reject `--mgrid-grouping family` until grouped field generation is implemented and tested.

2. Introduce a generic finite-build family builder.
   - [ ] Add a generic direct-per-loaded-coil helper that accepts a sequence of `Coil` objects and finite-build settings.
   - [ ] For each input coil, call `create_multifilament_grid` on that coil's curve.
   - [ ] For each input coil, compute `filament_current_A = source_current_A / nfilaments`.
   - [ ] Create new fixed `Current(filament_current_A)` objects for each filament so output currents do not share mutable source current objects.
   - [ ] Return the generated `Coil` objects in source-coil order, with all filaments for one source coil contiguous.
   - [ ] Preserve the current banana-only helper behavior by keeping `_build_finitebuild_banana_coils(...)` as the banana-only path unless a parity test proves the direct helper is equivalent.

3. Build the all-family `BiotSavart` artifact.
   - [ ] In `banana-only` scope, preserve current output ordering: `TF + finitebuild banana + proxy + VF`.
   - [ ] In `all-families` scope, output `finitebuild TF + finitebuild banana + finitebuild proxy + finitebuild VF`.
   - [ ] Preserve empty proxy/VF partitions without special output placeholders.
   - [ ] Compute source and output counts per family.
   - [ ] Compute signed and absolute current totals per family before and after conversion.
   - [ ] Save the full finite-build `BiotSavart` JSON using the existing output path safety checks.

4. Add VMEC/MGrid single-group export.
   - [ ] If `--write-mgrid` is passed with `--mgrid-grouping single`, call `output_biot_savart.to_mgrid(...)`.
   - [ ] Require explicit grid bounds and resolution for MGrid output; do not guess HBT production grid limits inside the exporter.
   - [ ] Write MGrid metadata next to the `BiotSavart` metadata: path, grid bounds, grid resolution, `nfp`, `nphi`, grouping mode, and recommended VMEC values.
   - [ ] Emit recommended VMEC settings in metadata and CLI summary: `LFREEB = T`, `MGRID_FILE = <mgrid path>`, `NZETA = <mgrid nphi>` for the simple equality case, `EXTCUR[0] = 1.0`.
   - [ ] Record that Simsopt permits MGrid `nphi` to be an integer multiple of VMEC `NZETA`; the exporter recommendation uses equality because it is the local example-backed path.
   - [ ] State in metadata that plasma current is not encoded in the external-field MGrid and must be set through VMEC current-profile inputs.

5. Add grouped MGrid support only after the single-group path is validated.
   - [ ] Add a helper that evaluates one family `BiotSavart` at the MGrid cylindrical tensor grid.
   - [ ] Match the grid construction used by `MagneticField.to_mgrid`: `phi` in `[0, 2*pi/nfp)`, `np.meshgrid(..., indexing="ij")`, `set_points_cyl`, `B_cyl`, then reshape to `(nphi, nz, nr)`.
   - [ ] Use `MGrid.add_field_cylindrical(...)` once per requested group.
   - [ ] Store B-field components only in the first grouped implementation; do not add grouped vector-potential export until a separate test covers `ar/ap/az` component order.
   - [ ] Support initial groups `tf`, `banana`, `proxy`, and `vf`.
   - [ ] Write group labels and recommended `EXTCUR` entries in the same order as the MGrid groups.
   - [ ] Verify that summing loaded MGrid groups reproduces the single-group field within numerical tolerance on a small grid.

6. Keep VMEC execution separate from export.
   - [ ] Do not run VMEC inside the finite-build exporter.
   - [ ] Add a small optional helper script or documented command that materializes a free-boundary VMEC input from an existing input file and generated MGrid path.
   - [ ] Keep compatibility with `banana_opt.vmec_seed_loader`, which currently rejects free-boundary seeds for fixed-boundary Stage A workflows.
   - [ ] Document that production VMEC sweeps must set `NCURR`, `CURTOR`, and current-profile coefficients independently when plasma current is part of the question.

## Validation Plan

- [ ] Extend `tests/geo/test_finitebuild_biotsavart_generator.py` with an all-family synthetic artifact containing TF, banana, proxy, and VF partitions.
- [ ] Test that `banana-only` scope keeps current output counts and metadata stable.
- [ ] Test that `all-families` scope multiplies every non-empty family count by `numfilaments_n * numfilaments_b`.
- [ ] Test signed current conservation per source coil: each source coil's generated filament-current sum equals the source coil current.
- [ ] Test signed and absolute current conservation per family in metadata.
- [ ] Test that all generated filament currents are new `Current` objects and source current values are not mutated.
- [ ] Test that a saved all-family finite-build `BiotSavart` reloads, accepts `set_points`, and returns finite `B()` values.
- [ ] Test that banana-only scope still uses the current symmetry-expansion path and preserves current banana-only counts, current totals, and smoke-field behavior.
- [ ] If the generic direct helper ever replaces the banana-only helper, add a parity test comparing finite-build curve coordinates or sampled `B()` values against the current finite-build-first/symmetry-second path before making that refactor.
- [ ] Add an MGrid smoke test that runs all-family export with `--write-mgrid --mgrid-grouping single` on a tiny synthetic grid.
- [ ] Load the written MGrid with `MGrid.from_file(...)` and verify `nextcur == 1`, `coil_names == ["simsopt_coils"]` or the padded equivalent, and field array shapes match `(nphi, nz, nr)`.
- [ ] Compare one or more MGrid grid-point field values against direct `BiotSavart.B_cyl()` evaluation.
- [ ] If grouped MGrid is implemented, test `nextcur == number_of_nonempty_groups` and verify summed group fields match the single-group field.
- [ ] Run focused tests:
  `PYTHONNOUSERSITE=1 PYTHONPATH=examples/single_stage_optimization .conda-env/bin/python -m pytest -q tests/geo/test_finitebuild.py tests/geo/test_finitebuild_biotsavart_generator.py tests/field/test_mgrid.py::Testing::test_write tests/field/test_mgrid.py::Testing::test_from_file`
- [ ] If the local VMEC Python module is available, run the existing free-boundary VMEC smoke:
  `PYTHONNOUSERSITE=1 .conda-env/bin/python -m pytest -q tests/field/test_mgrid.py::VmecTests::test_free_boundary_vmec`

## Risks and Mitigations

- Risk: Double-applying symmetry would duplicate coils and corrupt current signs.
  Mitigation: In all-family mode, finite-build each loaded physical coil directly and do not call `apply_symmetries_to_curves` unless metadata explicitly identifies a master-coil representation.

- Risk: Refactoring banana-only export through the generic direct helper could change finite-build geometry because finite-build-after-symmetry is not proven equivalent to finite-build-before-symmetry.
  Mitigation: Keep the current banana-only helper in place until a dedicated parity test proves equivalence for the selected frame and rotation settings.

- Risk: Grouped MGrid export could imply independent VMEC current knobs that do not match real circuit wiring.
  Mitigation: Make grouping explicit, document group order, and keep single-group export as the default.

- Risk: MGrid bounds may omit part of the plasma or be too coarse for VMEC deformation.
  Mitigation: Require explicit grid bounds and resolution in the CLI; report them in metadata; use the official rule of at least `4 * ntor` toroidal planes as guidance, not as hidden automatic behavior.

- Risk: Users may mistake external coil finite-build for plasma-current modeling.
  Mitigation: Metadata and CLI summary must state that MGrid encodes only external coil fields; plasma current belongs in VMEC `NCURR=1`, `CURTOR`, `AC`, or current-profile settings.

- Risk: Current overrides could accidentally apply to every family.
  Mitigation: Preserve `--banana-current-A` as banana-only, and do not add family current overrides until a separate current-control contract is specified.

- Risk: All-family finite-build can create many coils and make field evaluation slow.
  Mitigation: Keep the exporter explicit, report generated counts before expensive MGrid generation, and keep validation grids small.

## Completion Criteria

- [ ] `banana-only` export behavior remains backward compatible and existing tests pass.
- [ ] `all-families` export mode writes a reloadable `BiotSavart` with finite-build TF, banana, proxy, and VF coils.
- [ ] Metadata records source/output counts and current totals for every family.
- [ ] MGrid single-group export writes a file readable by `MGrid.from_file(...)`.
- [ ] Metadata includes recommended free-boundary VMEC settings for the generated MGrid.
- [ ] Focused finite-build and MGrid tests pass locally.
- [ ] Documentation clearly separates external finite-build coil fields from VMEC plasma-current settings.

## Open Questions

- Should all-family finite-build initially require one shared filament-pack geometry, or should TF/banana/proxy/VF have separate settings from the first implementation?
- Should grouped MGrid use physical family labels only, or circuit labels from future hardware metadata when available?
- Should the exporter support current scaling for TF/proxy/VF, or should all non-banana current scaling remain out of scope until VMEC sweep requirements are finalized?
- Should the VMEC helper materialize a new input file, or only emit metadata and leave namelist editing to the caller?
