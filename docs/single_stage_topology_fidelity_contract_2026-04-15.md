# Single-Stage Topology Fidelity Contract

Date: 2026-04-15
Updated: 2026-04-18
Status: Implemented for the shared topology scorer used by the cheap gate, the medium callback scorer, and the strict Poincare validator.

## Goal

Keep the topology ladder on one shared tracing contract while varying fidelity
through `nfieldlines`, `tmax`, and `nphis`, not through different seeding
geometries.

## Seed Contract

All tiers now seed inside the traced Boozer surface by:

1. taking the `phi=0` cross-section
2. identifying the inner and outer midplane radii from the near-`Z=0` contour points
3. insetting that radial span by `inset_fraction = 0.05` with `min_inset = 0.01`
4. launching an evenly spaced radial sweep at `Z = 0`

This matches the upstream SIMSOPT `POINCARE_PLOTTING/poincare_surfaces.py`
midplane convention. The effective seed contract is:

- `seed_mode = midplane_radial_sweep`
- `nplanes = 1`
- `phi = 0.0`
- `Z = 0.0`
- `inset_fraction = 0.05`

### Cheap Tier

- `nfieldlines = 4`
- `nphis = 1`
- shared seed contract: midplane radial sweep
- field policy: native field only (`field_policy = never`)
- intended use: fast search-time gate

### Medium Tier

- `nfieldlines = 12`
- `nphis = 4`
- shared seed contract: midplane radial sweep
- field policy: auto-interpolate for `tmax >= 50`
- intended use: callback checkpoint scorer

### Strict Tier

- `nfieldlines = 50`
- `nphis = 4`
- shared seed contract: midplane radial sweep
- field policy: auto-interpolate for `tmax >= 50`
- intended use: expensive Poincare validation

## Interpolation Policy

- Cheap tier stays on the native field.
- Medium and strict tiers use the shared interpolation-preparation helper when `tmax >= 50`.
- The scorer and the Poincare artifact now record interpolation metadata:
  - selected mode
  - selection reason
  - interpolation grid
  - max / mean absolute field error on the traced surface grid
  - max relative field error on the traced surface grid

## Calibration Report

Use:

```bash
python examples/single_stage_optimization/run_topology_fidelity_ladder.py \
  /path/to/output_a \
  /path/to/output_b \
  --report-path /tmp/topology_fidelity_ladder_report.json
```

The report records:

- per-case cheap / medium / strict results
- per-tier fidelity settings, including the shared `seed_mode` and `inset_fraction`
- false-pass counts versus strict validation
- false-reject counts versus strict validation
- Spearman rank correlation of cheap and medium confinement scores against strict
