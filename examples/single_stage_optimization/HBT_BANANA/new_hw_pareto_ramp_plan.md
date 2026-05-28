# New HW Pareto Ramp Plan

## Purpose

Plan a staged optimization campaign that uses the signed vacuum-current `nv2_iota298_negTF` package as a diagnostic seed while moving toward the new HBT hardware limits and the requested Pareto region: volume `>= 0.09` and iota `>= 0.10`.

## Goals

- Produce at least one new-HW-compliant candidate with volume `>= 0.09` and iota `>= 0.10`.
- Preserve the vacuum-current sign contract: TF current `-80 kA`, corresponding signed banana currents from the lane artifact telemetry, signed `G < 0`, no `I` field, no `BoozerSurfaceFiniteI`, and no proxy/VF/plasma-current finite-current terms. For the `simsopt-surrogate` signed package, the current source is `biot_savart_opt.json`.
- Avoid wasting Boozer solves on infeasible geometry by satisfying the coil footprint and winding-surface envelope before growing volume.
- Record enough telemetry to rank candidates by hardware slack, volume, iota, Boozer residual, and topology quality.

## Non-Goals

- Do not promote the zipped signed package as a final new-HW design.
- Do not jump directly from the current `volume ~= 0.04` seed to `volume >= 0.09` under all hard limits in one optimization.
- Do not require iota near `0.298`; the requested constraint is iota `>= 0.10`.
- Do not use augmented Lagrangian work unless the staged soft-penalty ramp proves insufficient.
- Do not promote finite-current compatibility artifacts, proxy-current coils, VF-current coils, or plasma-current flags. A legacy-compatible JSON loader is acceptable only when validation proves it loaded a plain upstream `BoozerSurface` JSON with no `I` field and no `BoozerSurfaceFiniteI` rewrite.

## Current Context

- Authoritative plan path in this checkout:
  `/Users/suhjungdae/code/hbt-compare/repos/baseline-original/examples/single_stage_optimization/HBT_BANANA/new_hw_pareto_ramp_plan.md`
- If a handoff says this named plan path is absent, treat that statement as stale for this checkout. First verify the current working directory and locate the plan with `rg --files | rg 'new_hw_pareto_ramp_plan\.md$'` before assuming the plan moved.
- This repo may already contain unrelated dirty or untracked work. Start every follow-up by running `git status --short`; preserve unrelated changes and outputs. If Stage 2 or single-stage files are already modified, treat them as user/prior-agent changes, inspect them before relying on them, and do not edit, stage, or commit them unless the task explicitly includes those files.
- Baseline hard limits were updated in `examples/single_stage_optimization/HBT_BANANA/config.yaml`.
- New-HW campaign spec from `config.yaml`:
  - TF current: `-80 kA`.
  - Banana current magnitude limit: `16 kA`.
  - Banana coil length: `<= 2.0 m` absolute, `1.9 m` optimization target.
  - Coil-coil minimum distance: `4.62 cm`.
  - Coil-plasma minimum distance: `1.0 cm`.
  - Plasma-vessel minimum distance: `4.0 cm`.
  - Maximum curvature: `100 m^-1`.
  - HBT/vacuum-vessel/TF major radius: `0.976 m`.
  - Vacuum vessel minor radius: `0.222 m`.
  - Banana coil winding surface: `R0=0.903 m`, `a=0.142 m`, concentric with the vacuum vessel.
  - Maximum target LCFS shell limit: major radius `0.92 m`, minor radius `0.15 m`; the smaller banana winding surface can become the active limiter.
  - Poloidal half-width from inboard midplane: `70 deg` maximum (`140 deg` full width).
- There are two relevant execution lanes, and they must not be mixed:
  - `HBT_BANANA/02_stage2_driver.py` + `HBT_BANANA/03_singlestage_driver.py` are the clean local new-HW lane. They read thresholds and targets from `config.yaml`, write registry-named artifacts through `utils/run_registry.py`, and `03_singlestage_driver.py` constructs plain upstream `BoozerSurface`.
  - `HBT_BANANA/jhalpern30/` is a historical smoke/replay lane. Its scripts build proxy and VF coils for finite-current-style diagnostics; those runs are not promotable for this vacuum-current campaign.
- The signed package is:
  `/Users/suhjungdae/code/columbia/simsopt-surrogate/tmp/nv2_iota298_negTF_signed_artifacts_for_review_20260527T113921`
- The signed package is the vacuum-current source of truth. Its Boozer JSON must remain plain `simsopt.geo.boozersurface.BoozerSurface`: no `I` field and no `banana_opt.boozer_finite_current.BoozerSurfaceFiniteI`.
- Candidate boot commands must not pass finite-current controls. In particular, do not pass proxy-plasma-current, VF-current, or finite-current-mode flags; the signed negative TF and banana currents are already encoded in `biot_savart_opt.json`.
- The package records old/offspec scalar telemetry in `files/results.json`:
  - TF current: `-80 kA`
  - banana current max abs: `15.830 kA`
  - coil length: `1.6999 m`
  - coil-coil min distance: `7.455 cm`
  - coil-plasma min distance: `3.661 cm`
  - plasma-vessel min distance: `4.764 cm`
  - max curvature: `79.67 m^-1`
- That package telemetry is not a new-HW pass. It also records `LENGTH_TARGET=1.7`, `banana_surf_radius=0.21`, and `MAJOR_RADIUS=0.976`; new runs must explicitly override the new campaign targets.
- The package is not a final new-HW design because it uses the old banana surface minor radius `0.21 m`, does not prove the new banana winding major radius `0.903 m`, and still has to pass the `70 deg` poloidal extent constraint under the new profile.
- The later 50-iteration signed run reported the active HW failure as poloidal extent:
  - realized half-width: `1.525824 rad` (`87.4 deg`)
  - threshold: `1.221730 rad` (`70.0 deg`)
- The target region is much larger in volume than the seed:
  - seed volume: `0.03993`
  - requested floor: `0.09`
  - required growth: more than `2.25x`
- For the `simsopt-surrogate` CLI lane, verified single-stage controls include `--banana-surf-radius`, `--stage2-seed-banana-surf-radius`, `--stage2-seed-major-radius`, `--single-stage-poloidal-threshold-rad`, `--length-target`, `--cc-dist`, `--cs-dist`, `--curvature-threshold`, `--banana-current-max-A`, `--tf-current-A`, `--vol-target`, `--iota-target`, `--maxiter`, and `--multisurface-initial-step-maxiter`.
- For the local `HBT_BANANA` lane, use `config.yaml` as the SSOT and vary runs by editing/hashing config values or the driver-supported environment overrides. The default local target values are `targets.volume=0.10` and `targets.iota=0.15`, which are above the requested promotion floors.
- The promoted banana winding major radius is now resolved in code, but still needs result telemetry:
  - `baseline-original` `config.yaml` sets `winding_surface.R0=0.903` and `winding_surface.a=0.142`.
  - `baseline-original` `jhalpern30` scripts use `WINDSURF_MAJOR_R=0.903`, but that lane is proxy/VF diagnostic only.
  - Current `simsopt-surrogate` `banana_opt.reference_surfaces.build_banana_reference_surfaces()` sets `coil_winding_surface.rc(0,0)` from `BANANA_WINDING_SURFACE_MAJOR_RADIUS_M=0.903`.
- High-volume artifacts that require `BoozerSurfaceFiniteI`, an `I` field, or proxy/VF/plasma-current metadata are invalid for this campaign even if they can be evaluated by a diagnostic loader.
- Current `simsopt-surrogate` commit `156b7a331` fixes the vacuum Boozer lineage in the shared construction path: zero/no-current saves must be upstream `BoozerSurface` artifacts, and the save path validates against zero-current `BoozerSurfaceFiniteI` JSON. The compatibility loader still exists for legacy diagnostics; promotion evidence must be the saved JSON lineage check, not successful compatibility loading.

## Rationale

The seed already has enough iota and acceptable current, length, curvature, and clearance margins under the old/offspec telemetry. The dominant mismatch is geometric: old surface radius, unverified new banana major radius, and too-wide poloidal footprint. Growing volume before the coil footprint is inside the new envelope will drive the optimizer toward expensive but unpromotable designs. The safer strategy is to first make the coil geometry live inside the new hardware envelope at roughly the existing volume, then grow volume in small increments while checking final iota against the requested floor.

## Seed And Parent Selection

| Role | Artifact or source | Use | Promotion status |
| --- | --- | --- | --- |
| Diagnostic signed seed | `/Users/suhjungdae/code/columbia/simsopt-surrogate/tmp/nv2_iota298_negTF_signed_artifacts_for_review_20260527T113921` | First vacuum-current boot, sign convention check, Boozer JSON lineage check, low-volume footprint-ramp parent only if it boots cleanly. | Not promotable: old `banana_surf_radius=0.21`, old `LENGTH_TARGET=1.7`, old/offspec `MAJOR_RADIUS=0.976`, and measured `70 deg` poloidal extent failure. |
| First ramp parent | Best booted signed-seed continuation that passes vacuum-current JSON lineage and does not introduce new hard-limit failures beyond the known poloidal extent miss. | Parent for footprint ramp at `volume ~= 0.04`. | Not promotable until it passes new-HW footprint and winding-surface checks. |
| Shrink-ramp parent | Best footprint-ramp candidate at the tightest completed poloidal threshold. | Parent for `banana_surf_radius` shrink steps. | Not promotable until active winding-surface telemetry proves `R0=0.903`, `a=0.142`. |
| Volume-ramp parent | First candidate that passes footprint plus winding-surface checks. | Parent for volume targets `0.055 -> 0.070 -> 0.085 -> 0.090`. | Promotion candidate only after final hard-limit, vacuum-current, and topology gates pass. |
| Local registry parent IDs | `config.yaml` `warm_start.stage1_id` and `warm_start.stage2_id`. | Required by local `02_stage2_driver.py` and `03_singlestage_driver.py`. | Currently TBD; fill with concrete run IDs before local registry execution. |

Parent selection rule: never keep using the original signed package once a later candidate has better hardware slack under the same vacuum-current contract. The original package is a sign/Boozer regression anchor, not a privileged optimizer parent.

## Run Book

### Signed package replay in `simsopt-surrogate`

Use the exact command in:

```text
/Users/suhjungdae/code/columbia/simsopt-surrogate/tmp/nv2_iota298_negTF_signed_artifacts_for_review_20260527T113921/AGENT_RUN_INSTRUCTIONS.md
```

Required properties of that command:

- It passes `--single-stage-resume-bs-path "$PKG/files/biot_savart_opt.json"`.
- It passes `--stage2-seed-surf-path "$PKG/files/surf_opt_boozer_surface.json"`.
- It passes `--tf-current-A -80000`.
- It does not pass `--finite-current-mode`, `--proxy-plasma-current-A`, `--vf-current-A`, `--vf-template-path`, or other plasma-current/proxy/VF controls.
- For a 50-iteration startup budget, use both `--maxiter 50` and `--multisurface-initial-step-maxiter 50`.

### Local clean new-HW lane in `baseline-original`

The local lane is registry-based and requires parent IDs; there is no "latest run" shortcut. This directory has Stage 2 and single-stage drivers, so the Stage 2 parent must already exist as a Stage 1 registry artifact before this runbook starts.

```bash
cd /Users/suhjungdae/code/hbt-compare/repos/baseline-original/examples/single_stage_optimization/HBT_BANANA

# Before Stage 2, set warm_start.stage1_id in config.yaml to an existing s01_* parent.
BANANA_OUT_DIR=/path/to/new_hw_runs python 02_stage2_driver.py

# Before single-stage, set warm_start.stage2_id in config.yaml to the emitted s02_* parent.
BANANA_OUT_DIR=/path/to/new_hw_runs python 03_singlestage_driver.py
```

Before running this lane, update or confirm these `config.yaml` values:

- `warm_start.stage1_id`: concrete `s01_*` parent for Stage 2.
- `warm_start.stage2_id`: concrete `s02_*` parent for single-stage.
- `winding_surface.R0: 0.903`.
- `winding_surface.a: 0.142`.
- `thresholds.length_target: 1.9`.
- `thresholds.length_max: 2.0`.
- `thresholds.coil_coil_min: 0.0462`.
- `thresholds.coil_surface_min: 0.01`.
- `thresholds.plasma_vessel_min: 0.04`.
- `thresholds.poloidal_half_width_max_deg: 70.0`.
- `thresholds.curvature_max: 100`.
- `targets.volume` and `targets.iota` for the current ramp step.

Do not use `HBT_BANANA/jhalpern30/` output as a promotion parent. That lane is useful for historical smoke/debugging only because it materializes proxy/VF finite-current-style coils.

Expected local artifacts use the registry patterns in `utils/run_registry.py`, not the `simsopt-surrogate` package names. For this lane, Stage 2 and single-stage Boozer outputs are `boozersurface_{id}_opt.json` or `boozersurface_{id}_failed.json`, diagnostics are `diagnostics_{id}.txt`, and successful single-stage runs also write `state_{id}_opt.npz`.

## Assumptions

- The `simsopt-surrogate` single-stage path can run with explicit thresholds for poloidal extent, coil width, coil-plasma distance, coil-coil distance, curvature, banana current, TF current, and volume/iota targets.
- The signed package remains useful for sign/Boozer regression only if the replay path is vacuum-current: plain `BoozerSurface`, no `I` field, no `BoozerSurfaceFiniteI`, and no proxy/VF/plasma-current flags.
- A candidate is not promotable unless hardware telemetry includes realized values and thresholds for poloidal extent, winding surface, coil length, coil-coil distance, coil-plasma distance, curvature, TF current, banana current, volume, iota, and topology.
- Current target-mode uses `--iota-target` as a target objective, not a verified hard floor. The campaign must either implement/verify true floor semantics or use target sweeps with a separate acceptance gate `FINAL_IOTA >= 0.10`.
- Final new-HW promotion needs proof that the active coil winding surface telemetry is `R0=0.903 m`, `a=0.142 m`; code constants alone are not enough.

## Implementation Plan

1. Establish a signed replay baseline
   - [ ] Run the minimal boot check from the package `AGENT_RUN_INSTRUCTIONS.md`.
   - [ ] Confirm the replay is vacuum-current: plain `BoozerSurface`, no `I` field, no `BoozerSurfaceFiniteI`, no proxy/VF/plasma-current finite-current flags in the command, and no active proxy/VF/plasma-current metadata in the result.
   - [ ] Confirm `TF_CURRENT_A=-80000`, signed negative TF and corresponding signed banana currents are loaded from `biot_savart_opt.json`, signed `G < 0`, and iota near `0.2979`.
   - [ ] Save the boot `results.json` and `results_all_properties.json` under a run directory named for the package timestamp.
   - [ ] Confirm old/offspec scalar telemetry is reproduced before applying new-HW thresholds.
   - [ ] Confirm the expected new-HW failure is poloidal extent unless an additional measured violation appears.

2. Footprint ramp at fixed low volume
   - [ ] Keep volume near the seed value (`--vol-target ~= 0.04`) and use `--iota-target` only as a target-mode input unless a true floor objective is implemented.
   - [ ] Run poloidal half-width thresholds in order with `--single-stage-poloidal-threshold-rad`: `80 deg` (`1.3962634016`), `75 deg` (`1.3089969390`), `70 deg` (`1.2217304764`).
   - [ ] At each target, reject candidates that violate banana current `<=16 kA`, TF current `=-80 kA`, coil length `<=1.9 m` target or `<=2.0 m` absolute, curvature `<=100 m^-1`, coil-coil `>=4.62 cm`, or coil-plasma `>=1.0 cm`.
   - [ ] Pass `--length-target 1.9` in new-HW runs; do not inherit the package's old `1.7 m` target.
   - [ ] Keep the best candidate per target by hardware slack first, then Boozer residual, then volume.

3. Winding-surface shrink ramp
   - [ ] Start from the best footprint-ramp candidate, not necessarily the original package.
   - [ ] Move the banana surface minor radius with `--banana-surf-radius` and `--stage2-seed-banana-surf-radius` in small steps: `0.21`, `0.18`, `0.16`, `0.142`.
   - [ ] For `simsopt-surrogate`, verify the run recorded `BANANA_WINDING_SURFACE_MAJOR_RADIUS_M=0.903` and `banana_surf_radius=0.142`; the current builder already uses the `0.903 m` major-radius constant and `BANANA_WINDING_MINOR_RADIUS_M=0.142`.
   - [ ] For the local `HBT_BANANA` lane, verify the registry config hash records `winding_surface.R0=0.903` and `winding_surface.a=0.142`.
   - [ ] Re-run the `70 deg` poloidal half-width check at every shrink step.
   - [ ] Stop the lane if the optimizer can only pass by violating current, length, curvature, or clearance limits.

4. Volume ramp after new-HW geometry is feasible
   - [ ] Use the first candidate that passes the new footprint and winding-surface checks as the parent.
   - [ ] Ramp volume target in order: `0.055`, `0.070`, `0.085`, `0.090`.
   - [ ] Do not force iota back to `0.298` during the first volume ramp.
   - [ ] If the code still only exposes target-mode iota control, use `--iota-target` sweeps and enforce `FINAL_IOTA >= 0.10` at acceptance.
   - [ ] At each volume step, keep the best hardware-clean candidate and one near-miss candidate for debugging.
   - [ ] If volume growth breaks topology or Boozer solve stability, bisect the last successful volume interval.

5. Pareto sweep after bootability
   - [ ] Sweep iota target values or verified floors: `0.10`, `0.15`, `0.20`.
   - [ ] Sweep volume targets around the front: `0.085`, `0.090`, `0.095`.
   - [ ] Rank by: hardware-clean first, topology pass second, `FINAL_VOLUME` third, `FINAL_IOTA` fourth, Boozer residual fifth.
   - [ ] Archive every promoted candidate with the lane-specific artifact set below, non-null realized metrics, and Poincare diagnostics.

6. Decide whether soft penalties are enough
   - [ ] If the footprint and winding-surface ramps pass with soft penalties, keep the method simple.
   - [ ] If repeated candidates hover just outside hard limits while improving physics metrics, document which constraint is active and only then consider augmented Lagrangian or a hard-bound formulation.
   - [ ] Do not introduce ALM for the first ramp unless soft penalties fail on a specific measured constraint.

## Validation Plan

- [ ] For every run, assert the vacuum-current contract:
  - Boozer JSON uses plain `BoozerSurface`.
  - Boozer JSON has no `I` field.
  - Boozer JSON does not reference `BoozerSurfaceFiniteI`.
  - Boot command contains no finite-current-mode, proxy-plasma-current, VF-current, or plasma-current flags.
  - Result metadata contains no active proxy/VF/plasma-current finite-current terms.
  - For `simsopt-surrogate`, run or reproduce the `validate_boozer_surface_json_current_lineage()` check on each saved Boozer JSON.
  - For local `HBT_BANANA`, reject any artifact produced through `jhalpern30/` proxy/VF paths.
- [ ] For every run, assert signed currents:
  - `TF_CURRENT_A == -80000`
  - `abs(BANANA_CURRENT_MAX_ABS_A) <= 16000`
  - signed banana current values preserve the signed negative-TF/banana-current convention and are backed by the lane artifact telemetry: loaded `biot_savart_opt.json` for `simsopt_surrogate_cli`, registry/config/artifact telemetry for `baseline_original_hbt_banana`.
- [ ] For every promoted candidate, assert hard limits:
  - `LENGTH_TARGET == 1.9` for new-HW campaign rows.
  - `COIL_LENGTH <= 1.9` for target-pass rows and `<= 2.0` absolute.
  - `CURVE_CURVE_MIN_DIST >= 0.0462`.
  - `CURVE_SURFACE_MIN_DIST >= 0.01`.
  - `SURFACE_VESSEL_MIN_DIST >= 0.04` with a non-null measured value, or a documented separate vessel-clearance check if the single-surface results omit it.
  - `MAX_CURVATURE <= 100`.
  - `POLOIDAL_EXTENT_RAD <= 1.2217304763960306` and `POLOIDAL_EXTENT_THRESHOLD_RAD == 1.2217304763960306`.
  - `banana_surf_radius == 0.142` for final candidates.
  - new winding surface `R0=0.903`, `a=0.142` for final candidates, with explicit telemetry for the active coil winding surface, not only seed or clearance-reference values.
  - target LCFS major radius `<=0.92` and minor radius `<=0.15`, unless the smaller banana winding surface is the active limiting envelope and is checked explicitly.
- [ ] For every promoted candidate, assert physics floors:
  - `FINAL_VOLUME >= 0.09`.
  - `FINAL_IOTA >= 0.10`.
  - Boozer solve succeeds and residual is recorded.
- [ ] Run strict Poincare validation on final candidates, not on every exploratory near-miss.
- [ ] Check that result files include both realized values and thresholds for hardware-clean rows.

## Required Artifact Package

Each archived candidate directory must contain or point to the artifact set for its lane.

For `simsopt_surrogate_cli` candidates:

- `biot_savart_opt.json`
- vacuum-current Boozer surface JSON
- Boozer state sidecar if the run path writes one
- surface JSON
- full `results.json`
- all-properties JSON or equivalent non-null property dump
- diagnostics log
- Poincare data and plot for promoted candidates
- a single Pareto row with the schema below

For `baseline_original_hbt_banana` candidates:

- registry row and config hash for the local run ID
- parent `s01_*` or `s02_*` registry ID used for the stage
- `boozersurface_{id}_opt.json` for successful rows, or `boozersurface_{id}_failed.json` for retained near-miss diagnostics
- `state_{id}_opt.npz` for successful single-stage rows
- `diagnostics_{id}.txt`
- explicit realized-metric dump or registry export covering the Pareto schema below
- external Poincare data and plot for promoted candidates
- a single Pareto row with the schema below

Do not require `biot_savart_opt.json` or `results.json` from the local `baseline_original_hbt_banana` lane unless a separate exporter is added; those names belong to the `simsopt-surrogate` artifact convention.

Required Pareto row columns:

| Column | Meaning |
| --- | --- |
| `run_id` | Candidate run identifier or path basename. |
| `parent_id` | Immediate parent candidate, stage ID, or signed package name. |
| `seed_source` | `signed_zip`, `footprint_ramp`, `shrink_ramp`, `volume_ramp`, or `fresh_stage2`. |
| `lane` | `simsopt_surrogate_cli` or `baseline_original_hbt_banana`. |
| `volume` | Final measured volume. |
| `iota` | Final measured iota. |
| `tf_current_A` | Signed TF current; must be `-80000`. |
| `banana_current_max_abs_A` | Maximum absolute banana current; must be `<=16000`. |
| `coil_length_m` | Banana coil length. |
| `coil_coil_min_m` | Minimum coil-coil distance. |
| `coil_plasma_min_m` | Minimum coil-plasma distance. |
| `plasma_vessel_min_m` | Minimum plasma-vessel distance or explicit separate-check reference. |
| `max_curvature_inv_m` | Maximum curvature. |
| `poloidal_extent_rad` | Realized poloidal half-width. |
| `winding_R0_m` | Active winding-surface major radius. |
| `winding_a_m` | Active winding-surface minor radius. |
| `boozer_residual` | Recorded Boozer residual or residual norm. |
| `topology_status` | Strict Poincare/topology result for promoted rows. |
| `vacuum_lineage_ok` | True only if Boozer JSON is plain `BoozerSurface` with no `I` field and no `BoozerSurfaceFiniteI`. |
| `promotable` | True only if every hard gate and physics floor passes. |

## Risks and Mitigations

- Risk: The signed package is too far from the new winding surface to be a useful parent.
  Mitigation: Use it only for diagnostic continuation; start a fresh new-HW Stage 2 lane if the shrink ramp stalls.
- Risk: A finite-current diagnostic artifact appears to satisfy the physics targets but violates the campaign contract.
  Mitigation: Reject any candidate whose Boozer JSON contains an `I` field, references `BoozerSurfaceFiniteI`, or whose boot command/result metadata activates proxy/VF/plasma-current terms.
- Risk: Volume `>=0.09` conflicts with the smaller winding envelope.
  Mitigation: Ramp volume only after footprint feasibility, then bisect the last successful interval to locate the real front.
- Risk: Forcing iota near `0.298` prevents volume growth.
  Mitigation: Use lower `--iota-target` sweeps or a verified floor formulation, then gate accepted candidates on `FINAL_IOTA >=0.10`.
- Risk: Soft penalties allow near-miss candidates that look good but fail final hard checks.
  Mitigation: Gate promotion on realized hard-limit metrics, not objective value.
- Risk: Missing telemetry makes hardware-clean rows ambiguous.
  Mitigation: Require realized value plus threshold for every hardware metric before ranking.
- Risk: The minor-radius flag is mistaken for full winding-surface control.
  Mitigation: Treat final promotion as unresolved until result telemetry proves both the major radius and minor radius used by the active coil winding surface.

## Completion Criteria

- [ ] At least one archived candidate passes all new HW limits with volume `>=0.09` and iota `>=0.10`.
- [ ] The archived candidate is vacuum-current: no `I` field, no `BoozerSurfaceFiniteI`, no proxy/VF/plasma-current finite-current flags, and lane-specific evidence for signed negative TF plus corresponding signed banana currents. For `simsopt_surrogate_cli`, that evidence is `biot_savart_opt.json`; for `baseline_original_hbt_banana`, it is the registry/config/artifact telemetry for the local run.
- [ ] The final candidate has strict Poincare evidence and non-null hardware metrics.
- [ ] The Pareto table includes at least volume, iota, Boozer residual, non-QS metric, coil length, coil-coil distance, coil-plasma distance, curvature, current, poloidal extent, and topology status.
- [ ] The campaign records whether the signed zip was useful as a parent or only as a sign/Boozer regression.

## Open Questions

- Does each promoted artifact record the active winding-surface `R0=0.903` and `a=0.142` values in results/registry telemetry, or do we need to add one more telemetry field before promotion?
- Should the Pareto front implement a hard iota floor formulation, an asymmetric penalty around `0.10`, or target sweeps plus acceptance gating?
- What strict Poincare budget should be required for promotion: the prior `50` lines and `tmax=7000`, or a cheaper gate before the final run?
