# Finite-Current Ceiling Plan (2026-04-05)

Status: design note only. No implementation in this document.

## Goal

Define a correct and robust way to add an `8 kA` / `16 kA` style finite-current ceiling to the Columbia single-stage workflow without misinterpreting units or overstating the physics.

## Executive Summary

The current single-stage code already has a finite-current hook through `--boozer-I`, but that parameter is not in physical `A` or `kA`. It is the Boozer/SIMSOPT current quantity `I`, which uses the same convention as `G`.

That means a correct implementation should:

1. Accept a user-facing physical enclosed toroidal current in `kA` or `A`.
2. Convert it internally to the Boozer-unit `I`.
3. Enforce any ceiling on the physical current magnitude, not on the raw internal `boozer_I`.
4. Record both physical current and converted internal current in the results.
5. Label this mode as a project-specific `boozer_surrogate` finite-current model unless and until the current is pushed down into the equilibrium/current-profile layer.

## What The Current Code Actually Does

Unless noted otherwise, the file paths below point at the stable `/Users/suhjungdae/code/columbia/simsopt` tree. At the time of writing, the same finite-current hooks also exist in the active `/Users/suhjungdae/code/columbia/simsopt-surrogate` mirror.

The current single-stage solver exposes:

- `--boozer-I` in `/Users/suhjungdae/code/columbia/simsopt/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
- the Boozer residual
  - `(G + iota I) B - |B|^2 (x_phi + iota x_theta)`
  - in `/Users/suhjungdae/code/columbia/simsopt/src/simsopt/geo/surfaceobjectives.py`

The same value is written to results as:

- `BOOZER_I`

So the code already supports a nonzero finite-current-like parameter, but only at the Boozer-residual level.

## Unit Convention

SIMSOPT's own convention is:

- `G(s)` is `mu0/(2*pi)` times the poloidal current outside the surface
- `I(s)` is `mu0/(2*pi)` times the toroidal current inside the surface

This convention appears in:

- `/Users/suhjungdae/code/columbia/simsopt/src/simsopt/mhd/vmec_diagnostics.py`
- SIMSOPT docs: `https://simsopt.readthedocs.io/latest/example_vmec.html`

Therefore:

- `boozer_I` is not naturally "8" or "16" if the user means `kA`
- the correct conversion is:

```text
boozer_I = (mu0 / (2*pi)) * I_phys_A
         = 2e-7 * I_phys_A
```

Examples:

```text
 8 kA  -> 0.0016
16 kA  -> 0.0032
30 kA  -> 0.0060
35.2 kA -> 0.00704
```

This is the main correctness constraint on any implementation.

## HBT / HBT-Like Current Scales

### Source-backed scales

- Official HBT-EP page:
  - `Ip <= 30 kA`
  - source: `https://fusion.columbia.edu/facilities/hbt-ep-tokamak`

- Columbia HBT-EP slide deck:
  - typical plasma current `10-15 kA`
  - source: `https://apam.columbia.edu/files/seasdepts/applied-physics-and-applied-math/pdf-files/0120_DShiraki.pdf`

- Constructed HBT-EP-derived equilibrium in the literature:
  - `I = -35.2 kA`
  - source: `https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/omnigenous-umbilic-stellarators/9B2DA755935A20E123403AACC45833CA`

### Interpretation

These numbers support the claim that:

- `8 kA` is a conservative HBT-like scenario
- `16 kA` is a plausible nominal HBT-like scenario
- `30 kA` is a Columbia facility-page upper-bound HBT-EP scenario

They do **not** support the claim that one of these is the uniquely correct ceiling for every HBT-derived or stellarator-hybrid use case.

## Why A Naive `--boozer-I <= 16` Plan Is Wrong

If a user types:

```text
--boozer-I 16
```

thinking this means `16 kA`, the physical interpretation would instead be:

```text
I_phys = boozer_I / (2e-7) = 80,000 kA
```

So a direct ceiling on the raw internal `boozer_I` is physically wrong unless the caller is already reasoning in Boozer units.

## Why The Effect May Be Small In The Current Workflow

The current workflow is still mostly a vacuum / coil-field optimization with a finite-current correction inside the Boozer residual.

The codebase currently uses:

- banana seed current scale `1e4`
- TF coils: `20` coils at `1e5 A` each

This means the present `G` scale is much larger than the physical `8-16 kA` Boozer-unit contribution:

```text
G = 2*pi * sum(|I_coil|) * mu0/(2*pi)
  = sum(|I_coil|) * mu0
  = (20 * 1e5 A) * (4*pi*1e-7)
  = 2.513...

G ~ 2.51
I(8 kA) = 0.0016
I(16 kA) = 0.0032
```

Representative scale ratios:

```text
I_phys   I_boozer   iota*I/G (iota=0.15)   iota*I/G (iota=0.25)
8 kA     0.0016     9.5e-5                 1.6e-4
16 kA    0.0032     1.9e-4                 3.2e-4
35.2 kA  0.00704    4.2e-4                 7.0e-4
```

So in the current single-stage residual, a physical `8-16 kA` finite-current term is probably:

- meaningful as provenance and modeling correctness
- but numerically modest relative to the background toroidal-field scale

This is why the short-term implementation should be described as a project-specific `boozer_surrogate` finite-current bound, not as a full finite-current equilibrium capability.

## What The Literature Suggests Is The More Physical Approach

Finite-current stellarator optimization in the literature is usually done through:

- equilibrium/current-profile control
- finite-beta equilibrium + plasma-current field contributions
- free-boundary or fixed-boundary equilibrium tools such as SPEC or VMEC

Key references:

- prescribed toroidal current profile in SPEC:
  - `https://www.cambridge.org/core/services/aop-cambridge-core/content/view/38C1F45C49272E111E28CDD763903BD8/S0022377821000520a.pdf/computation-of-multi-region-relaxed-magnetohydrodynamic-equilibria-with-prescribed-toroidal-current-profile.pdf`

- finite-beta and toroidal-current optimization:
  - `https://arxiv.org/abs/2111.15564`

- compact stellarator-tokamak hybrid:
  - `https://arxiv.org/abs/2406.02353`

This literature weakens any plan that treats a single Boozer residual scalar as the full finite-current story.

## Recommended Plan

## Phase 1: Correct Surrogate Ceiling

Add a user-facing physical current parameter to the single-stage workflow.

Prior implementation context:

- `simsopt` already has branch `finite-current-boozer`
- commit `8efc93ed`: `fix: add finite-current Boozer residual support`

Recommended fields:

- `plasma_current_kA`
- `plasma_current_ceiling_kA`

Internal behavior:

```text
plasma_current_A = 1000 * plasma_current_kA
boozer_I = (mu0 / (2*pi)) * plasma_current_A
```

Enforcement:

```text
abs(plasma_current_kA) <= plasma_current_ceiling_kA
```

Result fields to record:

- `PLASMA_CURRENT_KA`
- `PLASMA_CURRENT_A`
- `PLASMA_CURRENT_CEILING_KA`
- `BOOZER_I`
- `FINITE_CURRENT_MODE = "boozer_surrogate"`

Compatibility rule:

- keep `--boozer-I` only as an expert/internal parameter or deprecate it
- do not allow both raw `--boozer-I` and physical current inputs at the same time
- if these fields are propagated into `autoresearch`, update `registry/registry.py` so the new top-level keys are either mapped or explicitly skipped during ingest

## Phase 2: Scenario-Based Use, Not One Hardcoded Number

Use current scenarios rather than claiming one universal ceiling:

- `0 kA`: vacuum baseline
- `8 kA`: conservative HBT-like scenario
- `16 kA`: nominal HBT-like scenario
- `30 kA`: facility-page upper-bound HBT-EP scenario

This is the strongest scientifically defensible near-term plan.

## Phase 3: Full Finite-Current Upgrade

If the goal is real finite-current physics, move the current constraint to:

- VMEC current profile / `curtor`
- SPEC prescribed toroidal current profile
- or a finite-beta / virtual-casing coupled workflow

Then the ceiling applies to the actual equilibrium/current-profile quantity, not just a Boozer residual correction.

## Strengths Of This Plan

- respects the repo's own unit convention
- prevents catastrophic unit mistakes
- aligns with HBT-scale current magnitudes from primary sources
- preserves backward compatibility if desired
- keeps short-term implementation cheap and auditable
- leaves a clean upgrade path to equilibrium-level finite-current modeling

## Weaknesses / Limits

1. This still does not make the current workflow a full finite-current equilibrium solver.
2. In the present vacuum-style workflow, the physical effect of `8-16 kA` may be small.
3. No source found so far proves that `8 kA` or `16 kA` is the one uniquely correct ceiling.
4. If a stronger finite-current effect is needed, Phase 1 alone will not be sufficient.

## Validation Criteria

Before calling implementation correct, require:

1. `0 kA` reproduces current behavior.
2. `8 kA`, `16 kA`, `30 kA` map to the correct internal `BOOZER_I`.
3. Out-of-range physical currents fail clearly.
4. Sign is preserved and documented explicitly. The code should define what positive and negative toroidal current mean in the chosen coordinate/orientation convention.
5. Result files contain both physical and internal current values.
6. Historical result parsing remains backward compatible.
7. A small sensitivity sweep (`0`, `8`, `16`, `30`, optionally `35.2 kA`) confirms the new term is not being lost in numerical noise and does not silently invert sign.

## Decision

The correct near-term implementation is:

- a physical-current ceiling in `kA`
- converted once into internal Boozer units
- explicitly labeled as a project-specific `boozer_surrogate` finite-current mode

The correct long-term implementation is:

- equilibrium/current-profile-level current control and current ceilings

## References

- HBT-EP Tokamak:
  - `https://fusion.columbia.edu/facilities/hbt-ep-tokamak`

- HBT-EP parameters slide deck:
  - `https://apam.columbia.edu/files/seasdepts/applied-physics-and-applied-math/pdf-files/0120_DShiraki.pdf`

- Omnigenous umbilic stellarators:
  - `https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/omnigenous-umbilic-stellarators/9B2DA755935A20E123403AACC45833CA`

- Prescribed toroidal current profile in SPEC:
  - `https://www.cambridge.org/core/services/aop-cambridge-core/content/view/38C1F45C49272E111E28CDD763903BD8/S0022377821000520a.pdf/computation-of-multi-region-relaxed-magnetohydrodynamic-equilibria-with-prescribed-toroidal-current-profile.pdf`

- Stellarator optimization at finite beta and toroidal current:
  - `https://arxiv.org/abs/2111.15564`

- Compact stellarator-tokamak hybrid:
  - `https://arxiv.org/abs/2406.02353`

- SIMSOPT VMEC/current docs:
  - `https://simsopt.readthedocs.io/latest/example_vmec.html`
