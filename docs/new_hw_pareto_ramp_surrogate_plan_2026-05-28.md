# New HW Pareto Ramp Plan For simsopt-surrogate

## Purpose

Plan a surrogate-native HBT banana-coil campaign that uses the signed vacuum-current `nv2_iota298_negTF` package as a diagnostic warm-start while producing new-HW-compliant single-stage candidates with final volume `>= 0.09` and final iota `>= 0.10`.

This plan belongs to:

```text
/Users/suhjungdae/code/columbia/simsopt-surrogate
```

It is not the `baseline-original` registry plan. The baseline plan is only an external comparison reference.

## Goals

- Produce at least one promotable `simsopt-surrogate` candidate with `FINAL_VOLUME >= 0.09` and `FINAL_IOTA >= 0.10`.
- Preserve the strict vacuum-current contract: signed TF current `-80000 A`, signed banana currents from `biot_savart_opt.json`, signed negative Boozer `G`, no `I` field, no `BoozerSurfaceFiniteI`, no proxy/VF/plasma-current finite-current flags.
- Use surrogate-native artifacts and validators: `biot_savart_opt.json`, `results.json`, `surf_*_boozer_surface.json`, Boozer state sidecars, `validate_boozer_surface_json_current_lineage()`, and Poincare sidecars.
- Ramp geometry before volume: first satisfy the new winding surface, poloidal footprint, width, self-intersection, length, distance, curvature, and current constraints; then grow volume in small steps.
- Record each accepted and near-miss candidate with enough metadata to reconstruct command flags, parent artifact, hardware thresholds, realized metrics, Boozer solve state, and topology evidence.

## Non-Goals

- Do not use `baseline-original` local registry IDs (`s01_*`, `s02_*`) as surrogate run identifiers.
- Do not require local `baseline-original` artifact names such as `boozersurface_{id}_opt.json`, `state_{id}_opt.npz`, or `diagnostics_{id}.txt`.
- Do not promote historical `jhalpern30` compatibility imports unless they pass the strict vacuum-current artifact checks after conversion.
- Do not flip, regenerate, or unsigned-normalize the signed seed in place before the first validation run.
- Do not introduce augmented Lagrangian work until measured soft-penalty ramps repeatedly stall on a specific active constraint.
- Do not treat `--iota-target` as a hard iota floor. The acceptance gate remains `FINAL_IOTA >= 0.10`.

## Current Context

- Reviewed against checked HEAD `d476d2ac2`. Refresh this with `git rev-parse --short HEAD` before executing the plan.
- The signed package is:
  `/Users/suhjungdae/code/columbia/simsopt-surrogate/tmp/nv2_iota298_negTF_signed_artifacts_for_review_20260527T113921`
- The package run instructions are:
  `/Users/suhjungdae/code/columbia/simsopt-surrogate/tmp/nv2_iota298_negTF_signed_artifacts_for_review_20260527T113921/AGENT_RUN_INSTRUCTIONS.md`
- The package encodes:
  - TF current: `-80000 A`.
  - signed negative Boozer `G` from `files/stage00/state.json`.
  - handoff iota: `0.2978860258524164`.
  - handoff volume: `0.039929944341681727`.
  - shared alternating banana currents embedded in `files/biot_savart_opt.json`, approximately `[-15830.220858611288, +15830.220858611288, ...] A`.
- The minimal boot command in the package uses:
  - `--single-stage-resume-bs-path "$PKG/files/biot_savart_opt.json"`
  - `--stage2-seed-surf-path "$PKG/files/surf_opt_boozer_surface.json"`
  - `--tf-current-A -80000`
  - `--offspec-replay-debug-only`
  - `--accept-offspec-r0-seed`
  - `--hardware-search-mode warn`
  - `--topology-gate-fieldlines 0`
  - `--topology-scorer-every 0`
  - `--banana-surf-radius 0.21`
- For a real 50-iteration startup budget, pass both `--maxiter 50` and `--multisurface-initial-step-maxiter 50`.
- Surrogate hardware constants live in `examples/single_stage_optimization/banana_opt/hardware_contracts.py`:
  - TF hard limit magnitude: `80000 A`, with clockwise default `-80000 A`.
  - banana-current hard limit: `16000 A`.
  - coil length target: `1.9 m`; hard limit: `2.0 m`.
  - coil-coil minimum distance: `0.0462 m`.
  - coil-plasma minimum distance: `0.010 m`.
  - diagnostic plasma-vessel reference: `0.04 m`.
  - maximum curvature: `100 m^-1`.
  - vacuum vessel: `R0=0.976 m`, `a=0.222 m`.
  - banana winding surface: `R0=0.903 m`, `a=0.142 m`.
  - banana width bounds: `0.05 m <= width <= 0.17 m`.
  - poloidal half-width limit: `70 deg`.
- Surrogate code currently sets target LCFS bounds from the banana winding envelope:
  - `TARGET_LCFS_MAX_MAJOR_RADIUS_M = 0.903`
  - `TARGET_LCFS_MAX_MINOR_RADIUS_M = 0.132`
  These are stricter than the external note `R <= 0.92`, `a <= 0.15`. Do not silently mix those contracts; if the campaign wants the looser external target LCFS bounds, update the surrogate contract explicitly first.
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py` has `--strict-vacuum-current`, which cannot be combined with `--offspec-replay-debug-only` and requires `--strict-vacuum-lineage`.
  - Use `--strict-vacuum-lineage recent_stage1_candidate --stage1-candidate-id <s01_or_campaign_id>` for promotable production candidates.
  - Use `--strict-vacuum-lineage legacy_control` only for control-only replays. Results with this lineage set `STRICT_VACUUM_CONTROL_ONLY=true` and are not promotion candidates.
  - `--accept-offspec-r0-seed` is only valid with `--offspec-replay-debug-only`, so it is not available in production strict-vacuum runs.
- `--stage2-seed-major-radius` is the Stage 2 `MAJOR_RADIUS` / vacuum-vessel major radius field. It defaults to `VACUUM_VESSEL_MAJOR_RADIUS_M=0.976`; it is not the banana winding-surface major radius. The banana winding-surface major radius is recorded separately as `BANANA_WINDING_SURFACE_MAJOR_RADIUS_M=0.903`.
- The strict-vacuum command validator rejects `--boozer-I`, `--plasma-current-A`, `--finite-current-mode`, `--proxy-plasma-current-A`, `--vf-current-A`, and command tokens containing `BoozerSurfaceFiniteI`. This campaign also rejects `--vf-template-path` because VF coils/templates are outside the vacuum-current contract.
- The single-stage save path writes `biot_savart_opt.json` and `surf_opt*` artifacts, and `banana_opt.single_stage_geometry.save_surface_artifacts()` validates saved Boozer-surface current lineage through `validate_boozer_surface_json_current_lineage()`.
- Strict-vacuum runs write `captured_command.txt` and `seed_manifest.json` in the output directory.
- Poincare outputs are mode-specific sidecars such as `PoincarePlot_opt.png`, `PoincareMetrics_opt.json`, and explicit `_validation`, `_diagnostic`, or `_default` metric JSONs.

## Rationale

The signed seed is useful because it already exercises the correct negative-TF/sign-G convention and has enough iota margin. It is not directly promotable because it is a low-volume, old-geometry seed. The most likely failure mode is wasting long single-stage solves on candidates that increase volume while remaining outside the new HBT geometry envelope. The surrogate campaign should therefore use the signed seed as a boot and lineage regression anchor, then run a staged ramp: strict-vacuum boot, footprint repair, winding-surface shrink, volume growth, and finally Pareto/topology certification.

Soft penalties are sufficient for the first campaign because the immediate question is whether BoozerSurface optimization can boot, remain on the correct signed branch, and move toward the new envelope. ALM should be introduced only after soft-penalty evidence identifies a repeated active constraint that needs harder feasibility control.

## Assumptions

- The surrogate conda environment at `.conda-env` can import the compiled `simsoptpp` extension used by the single-stage scripts.
- The signed package remains available at the path above, or is copied without changing its file contents.
- Vacuum-current candidates are production candidates only when saved Boozer JSON lineage is plain upstream `BoozerSurface` with no `I` field and no `BoozerSurfaceFiniteI`.
- The first volume ramp can use target-mode iota control, but candidate promotion still uses a separate hard acceptance gate `FINAL_IOTA >= 0.10`.
- Strict Poincare validation is reserved for final candidates and high-value near-misses, not every exploratory ramp point.

## Implementation Plan

1. Establish a strict-vacuum signed boot baseline
   - [ ] Run the package boot command from `AGENT_RUN_INSTRUCTIONS.md` unchanged except for output root.
   - [ ] Confirm `TF_CURRENT_A == -80000` and `STAGE2_TF_CURRENT_A == -80000` in `results.json`.
   - [ ] Confirm banana currents are loaded from `files/biot_savart_opt.json`; do not rebuild from unsigned source JSON.
   - [ ] Confirm Boozer solve returns near the package iota before optimization changes are trusted.
   - [ ] Run the 50-iteration startup with both `--maxiter 50` and `--multisurface-initial-step-maxiter 50`.
   - [ ] Archive the baseline output root and record the exact command, stdout/stderr log, `results.json`, `biot_savart_opt.json`, and `surf_opt*_boozer_surface.json`.

2. Convert the replay command from offspec diagnostic to production vacuum-current mode
   - [ ] Remove `--offspec-replay-debug-only` and `--accept-offspec-r0-seed`; production strict-vacuum mode rejects that combination.
   - [ ] Add `--strict-vacuum-current`.
   - [ ] For promotable rows, add `--strict-vacuum-lineage recent_stage1_candidate --stage1-candidate-id <id>`.
   - [ ] Use `--strict-vacuum-lineage legacy_control` only for signed-package or `014417` control replays that are not promotable.
   - [ ] Keep forbidden flags absent: `--boozer-I`, `--plasma-current-A`, `--finite-current-mode`, `--proxy-plasma-current-A`, `--vf-current-A`, and `--vf-template-path`.
   - [ ] Use `--tf-current-A -80000`.
   - [ ] Keep `--banana-current-max-A 16000` or the code default hard limit.
   - [ ] Confirm the output records `STRICT_VACUUM_CURRENT=true`, `CURRENT_LINEAGE=strict_vacuum`, and `STRICT_VACUUM_METADATA_VALIDATION.passed=true`.
   - [ ] Save or verify `captured_command.txt` and `seed_manifest.json` in the candidate directory.

3. Repair footprint before growing volume
   - [ ] Keep volume target near the seed volume while reducing the poloidal half-width toward `1.2217304763960306 rad`.
   - [ ] Use `--single-stage-poloidal-threshold-rad 1.2217304763960306`.
   - [ ] Enforce or record banana width bounds `0.05 m <= width <= 0.17 m`.
   - [ ] Require self-intersection penalty/status to be zero or below the campaign's explicit numerical acceptance threshold.
   - [ ] Keep `--length-target 1.9`, `--cc-dist 0.0462`, `--cs-dist 0.01`, and `--curvature-threshold 100`.
   - [ ] Keep the best hardware-clean candidate and the best near-miss at each footprint step.

4. Move onto the new banana winding surface
   - [ ] Ramp `--banana-surf-radius` and `--stage2-seed-banana-surf-radius` in small steps: `0.21`, `0.18`, `0.16`, `0.142`.
   - [ ] Keep `--stage2-seed-major-radius 0.976` when passing it explicitly; do not use this flag to express the `0.903 m` banana winding-surface major radius.
   - [ ] Verify `BANANA_WINDING_SURFACE_MAJOR_RADIUS_M == 0.903` and `banana_surf_radius == 0.142` in `results.json` before promotion.
   - [ ] If using a Stage 2 seed instead of single-stage resume, ensure the Stage 2 `results.json` and `biot_savart_opt.json` digest match the selected parent.

5. Grow volume with iota-floor acceptance
   - [ ] Use the first footprint-clean, new-winding candidate as the parent.
   - [ ] Ramp `--vol-target` in order: `0.055`, `0.070`, `0.085`, `0.090`.
   - [ ] Do not force iota back to `0.298`; sweep lower iota targets if that improves volume.
   - [ ] Use `--iota-target` sweeps around `0.10`, `0.15`, and `0.20`, then accept only rows with `FINAL_IOTA >= 0.10`.
   - [ ] At every volume step, record hardware slack, final volume, final iota, Boozer residual, non-QS metric, optimizer success, and topology status if evaluated.

6. Run Pareto and topology certification
   - [ ] Rank rows by hard-limit pass first, vacuum-current lineage second, strict topology pass third, volume fourth, iota fifth, and Boozer residual sixth.
   - [ ] Run strict Poincare validation on promoted candidates using `POINCARE_OUT_DIR=/path/to/candidate python examples/single_stage_optimization/POINCARE_PLOTTING/poincare_surfaces.py`.
   - [ ] Archive `PoincareMetrics_*_validation.json` rather than relying only on aggregate `PoincareMetrics_*.json`.
   - [ ] Produce a Pareto table with the schema below.

7. Decide whether ALM is required
   - [ ] If the staged penalty ramp reaches a hard-limit-clean candidate, do not add ALM.
   - [ ] If candidates repeatedly hover just outside one hard limit while improving physics metrics, document the active constraint and only then test ALM or a hard-bound formulation.
   - [ ] Keep ALM runs separate from penalty runs in output roots and Pareto rows.

## Validation Plan

- [ ] Run `git status --short` before each edit or commit; preserve unrelated dirty files and untracked run artifacts.
- [ ] Validate every saved Boozer surface JSON with `validate_boozer_surface_json_current_lineage()`.
- [ ] Reject any promoted row whose Boozer JSON contains an `I` field or references `BoozerSurfaceFiniteI`.
- [ ] Reject any promoted row whose command or metadata activates finite-current, proxy-current, VF-current, or plasma-current terms.
- [ ] Reject any promoted row that lacks `STRICT_VACUUM_METADATA_VALIDATION.passed=true`.
- [ ] Reject any promoted row with `STRICT_VACUUM_CONTROL_ONLY=true`; promotable rows must have `STRICT_VACUUM_PRODUCTION_CANDIDATE=true`.
- [ ] Confirm `TF_CURRENT_A == -80000`.
- [ ] Confirm `BANANA_CURRENT_MAX_ABS_A <= 16000`.
- [ ] Confirm `COIL_LENGTH <= 1.9` for target-pass rows and `<= 2.0` absolute.
- [ ] Confirm `CURVE_CURVE_MIN_DIST >= 0.0462`.
- [ ] Confirm `CURVE_SURFACE_MIN_DIST >= 0.01`.
- [ ] Confirm plasma-vessel clearance is recorded or separately checked against `0.04 m`.
- [ ] Confirm `MAX_CURVATURE <= 100`.
- [ ] Confirm poloidal half-width `<= 1.2217304763960306 rad`.
- [ ] Confirm active winding telemetry records `BANANA_WINDING_SURFACE_MAJOR_RADIUS_M=0.903` and `banana_surf_radius=0.142`.
- [ ] Confirm `FINAL_VOLUME >= 0.09`.
- [ ] Confirm `FINAL_IOTA >= 0.10`.
- [ ] Confirm Boozer residual/trust fields are recorded and acceptable for the promoted row.
- [ ] Confirm strict Poincare validation sidecars exist for final candidates.

## Required Surrogate Artifact Package

Each archived surrogate candidate directory must contain or point to:

- exact command file or run script
- `captured_command.txt`
- `seed_manifest.json` for strict-vacuum runs
- stdout/stderr log
- `biot_savart_opt.json`
- `results.json`
- `surf_opt.json` and/or `surf_opt_*` surface JSON artifacts
- `surf_opt*_boozer_surface.json`
- Boozer state sidecar such as `surf_opt*_boozer_state.json` or `.boozer_state.json` when written
- `curves_opt.vtu`
- `surf_opt.vtu` when written
- cross-section and normal-field diagnostics when written
- Poincare validation plot and metric sidecars for promoted rows
- a Pareto row with the schema below

Required Pareto row columns:

| Column | Meaning |
| --- | --- |
| `run_id` | Output directory basename or campaign ID. |
| `parent_artifact` | Signed package path, Stage 2 seed path, or previous single-stage output root. |
| `seed_source` | `signed_neg_tf_package`, `footprint_ramp`, `winding_shrink`, `volume_ramp`, or `fresh_stage2`. |
| `constraint_method` | `penalty` or `alm`. |
| `strict_vacuum_current` | True only when the command and saved artifacts satisfy the strict vacuum contract. |
| `volume` | Final measured volume. |
| `iota` | Final measured iota. |
| `tf_current_A` | Signed TF current; must be `-80000`. |
| `banana_current_max_abs_A` | Maximum absolute banana current; must be `<=16000`. |
| `coil_length_m` | Banana coil length. |
| `coil_coil_min_m` | Minimum coil-coil distance. |
| `coil_plasma_min_m` | Minimum coil-plasma distance. |
| `plasma_vessel_min_m` | Plasma-vessel clearance or separate-check reference. |
| `max_curvature_inv_m` | Maximum curvature. |
| `poloidal_extent_rad` | Realized poloidal half-width. |
| `banana_width_m` | Projected banana width metric. |
| `self_intersect_penalty` | Self-intersection metric or penalty. |
| `winding_R0_m` | Active winding-surface major radius. |
| `winding_a_m` | Active winding-surface minor radius. |
| `boozer_residual` | Recorded Boozer residual or residual norm. |
| `non_qs_metric` | Final non-QS metric. |
| `topology_status` | Strict Poincare/topology status for promoted rows. |
| `promotable` | True only if every hard gate and physics floor passes. |

## Risks and Mitigations

- Risk: The signed package remains useful only as an offspec replay seed.
  Mitigation: Treat it as a sign and Boozer-lineage regression anchor; start a fresh new-HW Stage 2 seed if the strict-vacuum production command cannot leave offspec flags.
- Risk: The external target LCFS note (`R <= 0.92`, `a <= 0.15`) conflicts with surrogate's tighter current constants (`R <= 0.903`, `a <= 0.132`).
  Mitigation: Use the surrogate constants for this plan unless the contract is explicitly changed in code and tests.
- Risk: A compatibility-loaded artifact appears to optimize but is not production vacuum-current.
  Mitigation: Promotion requires saved JSON lineage validation, not just successful loading.
- Risk: Higher volume worsens poloidal footprint or topology.
  Mitigation: Ramp volume in small increments and preserve the last hard-limit-clean parent before each volume increase.
- Risk: ALM adds complexity before the baseline question is answered.
  Mitigation: Start with soft penalties and only test ALM against a documented repeated active constraint.

## Completion Criteria

- [ ] At least one surrogate output root passes all new-HW hard gates with `FINAL_VOLUME >= 0.09` and `FINAL_IOTA >= 0.10`.
- [ ] The promoted candidate is strict vacuum-current: no `I`, no `BoozerSurfaceFiniteI`, no proxy/VF/plasma-current flags, signed TF `-80000 A`, signed banana-current provenance from `biot_savart_opt.json`, signed negative `G`, `STRICT_VACUUM_PRODUCTION_CANDIDATE=true`, and `STRICT_VACUUM_CONTROL_ONLY=false`.
- [ ] The promoted candidate records `BANANA_WINDING_SURFACE_MAJOR_RADIUS_M=0.903` and `banana_surf_radius=0.142`.
- [ ] The promoted candidate has `results.json`, Boozer surface JSON, Boozer state sidecar if written, `biot_savart_opt.json`, and Poincare validation sidecars.
- [ ] The Pareto table includes all required columns and marks near-misses separately from promotable rows.
- [ ] The campaign records whether the signed negative-TF package became a useful parent or only a regression seed.

## Open Questions

- Should the surrogate contract continue using the tighter target LCFS bounds (`0.903`, `0.132`) or should code be updated to represent the external note (`0.92`, `0.15`) separately from the winding-surface envelope?
- What strict Poincare budget should be required for final promotion: prior `50` lines and `tmax=7000`, or a cheaper screening gate followed by the full final run?
- Should the first production strict-vacuum run use single-surface mode only, or should it immediately use the multisurface startup path with the explicit 50-iteration startup budget?
