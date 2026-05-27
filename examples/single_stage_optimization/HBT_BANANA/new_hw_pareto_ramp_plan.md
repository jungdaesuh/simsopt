# New HW Pareto Ramp Plan

## Purpose

Plan a staged optimization campaign that uses the signed `nv2_iota298_negTF` package as a diagnostic seed while moving toward the new HBT hardware limits and the requested Pareto region: volume `>= 0.09` and iota `>= 0.10`.

## Goals

- Produce at least one new-HW-compliant candidate with volume `>= 0.09` and iota `>= 0.10`.
- Preserve the negative-TF sign contract: TF current `-80 kA`, signed `G < 0`, and no unintended plasma-current/proxy-field terms.
- Avoid wasting Boozer solves on infeasible geometry by satisfying the coil footprint and winding-surface envelope before growing volume.
- Record enough telemetry to rank candidates by hardware slack, volume, iota, Boozer residual, and topology quality.

## Non-Goals

- Do not promote the zipped signed package as a final new-HW design.
- Do not jump directly from the current `volume ~= 0.04` seed to `volume >= 0.09` under all hard limits in one optimization.
- Do not require iota near `0.298`; the requested constraint is iota `>= 0.10`.
- Do not use augmented Lagrangian work unless the staged soft-penalty ramp proves insufficient.

## Current Context

- Baseline hard limits were updated in `examples/single_stage_optimization/HBT_BANANA/config.yaml`.
- The signed package is:
  `/Users/suhjungdae/code/columbia/simsopt-surrogate/tmp/nv2_iota298_negTF_signed_artifacts_for_review_20260527T113921`
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
- Verified `simsopt-surrogate` single-stage CLI controls include `--banana-surf-radius`, `--stage2-seed-banana-surf-radius`, `--stage2-seed-major-radius`, `--single-stage-poloidal-threshold-rad`, `--length-target`, `--cc-dist`, `--cs-dist`, `--curvature-threshold`, `--banana-current-max-A`, `--tf-current-A`, `--vol-target`, `--iota-target`, `--maxiter`, and `--multisurface-initial-step-maxiter`.
- A promoted banana winding major-radius control is still unresolved. Current `simsopt-surrogate` hardware constants set `BANANA_WINDING_SURFACE_MAJOR_RADIUS_M=0.903`, but `banana_opt.reference_surfaces.build_banana_reference_surfaces()` builds `coil_winding_surface` with `VACUUM_VESSEL_MAJOR_RADIUS_M=0.976`; reaching final `R0=0.903` therefore requires reconciling that source path or proving another active path owns the promoted winding surface.

## Rationale

The seed already has enough iota and acceptable current, length, curvature, and clearance margins under the old/offspec telemetry. The dominant mismatch is geometric: old surface radius, unverified new banana major radius, and too-wide poloidal footprint. Growing volume before the coil footprint is inside the new envelope will drive the optimizer toward expensive but unpromotable designs. The safer strategy is to first make the coil geometry live inside the new hardware envelope at roughly the existing volume, then grow volume in small increments while checking final iota against the requested floor.

## Assumptions

- The `simsopt-surrogate` single-stage path can run with explicit thresholds for poloidal extent, coil width, coil-plasma distance, coil-coil distance, curvature, banana current, TF current, and volume/iota targets.
- The signed package remains useful for sign/Boozer regression even though its winding surface is old.
- A candidate is not promotable unless hardware telemetry includes realized values and thresholds for poloidal extent, winding surface, coil length, coil-coil distance, coil-plasma distance, curvature, TF current, banana current, volume, iota, and topology.
- Current target-mode uses `--iota-target` as a target objective, not a verified hard floor. The campaign must either implement/verify true floor semantics or use target sweeps with a separate acceptance gate `FINAL_IOTA >= 0.10`.
- Current verified surface-radius CLI coverage is for the banana surface minor radius and the Stage 2 seed major radius. Final new-HW promotion also needs proof that the active coil winding surface major radius is `0.903 m`, not just that a seed or clearance reference used that value.

## Implementation Plan

1. Establish a signed replay baseline
   - [ ] Run the minimal boot check from the package `AGENT_RUN_INSTRUCTIONS.md`.
   - [ ] Confirm `TF_CURRENT_A=-80000`, signed `G < 0`, no proxy/VF/plasma-current terms, and iota near `0.2979`.
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
   - [ ] Reconcile the `0.903 m` hardware constant with the current reference-surface builder before claiming the final winding surface: `--stage2-seed-major-radius` is not enough if `coil_winding_surface` is still built at `0.976 m`.
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
   - [ ] Archive every promoted candidate with `biot_savart_opt.json`, Boozer surface JSON, surface JSON, full results, non-null properties, and Poincare diagnostics.

6. Decide whether soft penalties are enough
   - [ ] If the footprint and winding-surface ramps pass with soft penalties, keep the method simple.
   - [ ] If repeated candidates hover just outside hard limits while improving physics metrics, document which constraint is active and only then consider augmented Lagrangian or a hard-bound formulation.
   - [ ] Do not introduce ALM for the first ramp unless soft penalties fail on a specific measured constraint.

## Validation Plan

- [ ] For every run, assert signed currents:
  - `TF_CURRENT_A == -80000`
  - `abs(BANANA_CURRENT_MAX_ABS_A) <= 16000`
  - signed banana current list alternates as expected.
- [ ] For every promoted candidate, assert hard limits:
  - `LENGTH_TARGET == 1.9` for new-HW campaign rows.
  - `COIL_LENGTH <= 1.9` for target-pass rows and `<= 2.0` absolute.
  - `CURVE_CURVE_MIN_DIST >= 0.0462`.
  - `CURVE_SURFACE_MIN_DIST >= 0.01`.
  - `SURFACE_VESSEL_MIN_DIST >= 0.04` with a non-null measured value, or a documented separate vessel-clearance check if the single-surface results omit it.
  - `MAX_CURVATURE <= 100`.
  - `POLOIDAL_EXTENT_RAD <= 1.2217304763960306` and `POLOIDAL_EXTENT_THRESHOLD_RAD == 1.2217304763960306`.
  - `banana_surf_radius == 0.142` for final candidates.
  - new winding surface `R0=0.903`, `a=0.142` for final candidates, with explicit telemetry or code evidence for the active coil winding surface, not only seed or clearance-reference values.
- [ ] For every promoted candidate, assert physics floors:
  - `FINAL_VOLUME >= 0.09`.
  - `FINAL_IOTA >= 0.10`.
  - Boozer solve succeeds and residual is recorded.
- [ ] Run strict Poincare validation on final candidates, not on every exploratory near-miss.
- [ ] Check that result files include both realized values and thresholds for hardware-clean rows.

## Risks and Mitigations

- Risk: The signed package is too far from the new winding surface to be a useful parent.
  Mitigation: Use it only for diagnostic continuation; start a fresh new-HW Stage 2 lane if the shrink ramp stalls.
- Risk: Volume `>=0.09` conflicts with the smaller winding envelope.
  Mitigation: Ramp volume only after footprint feasibility, then bisect the last successful interval to locate the real front.
- Risk: Forcing iota near `0.298` prevents volume growth.
  Mitigation: Use lower `--iota-target` sweeps or a verified floor formulation, then gate accepted candidates on `FINAL_IOTA >=0.10`.
- Risk: Soft penalties allow near-miss candidates that look good but fail final hard checks.
  Mitigation: Gate promotion on realized hard-limit metrics, not objective value.
- Risk: Missing telemetry makes hardware-clean rows ambiguous.
  Mitigation: Require realized value plus threshold for every hardware metric before ranking.
- Risk: The minor-radius flag is mistaken for full winding-surface control.
  Mitigation: Treat `R0=0.903` as unresolved until `build_banana_reference_surfaces()` or another active source path and result telemetry prove the coil winding major-radius value.

## Completion Criteria

- [ ] At least one archived candidate passes all new HW limits with volume `>=0.09` and iota `>=0.10`.
- [ ] The final candidate has strict Poincare evidence and non-null hardware metrics.
- [ ] The Pareto table includes at least volume, iota, Boozer residual, non-QS metric, coil length, coil-coil distance, coil-plasma distance, curvature, current, poloidal extent, and topology status.
- [ ] The campaign records whether the signed zip was useful as a parent or only as a sign/Boozer regression.

## Open Questions

- Does the active `simsopt-surrogate` driver use `BANANA_WINDING_SURFACE_MAJOR_RADIUS_M=0.903` for the coil winding surface, or does `build_banana_reference_surfaces()` need to be corrected from `0.976`?
- Should the Pareto front implement a hard iota floor formulation, an asymmetric penalty around `0.10`, or target sweeps plus acceptance gating?
- What strict Poincare budget should be required for promotion: the prior `50` lines and `tmax=7000`, or a cheaper gate before the final run?
