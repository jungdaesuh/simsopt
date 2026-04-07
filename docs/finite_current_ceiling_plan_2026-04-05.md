# Finite-Current / Coil-Baseline Plan (2026-04-05)

Status: partially implemented. Interface, provenance, run-identity, and workflow-driver pieces are live in the repo; the self-consistent finite-current physics upgrade path remains future work.

## Goal

Separate two different tasks that were getting conflated:

1. establish a configurable external-coil baseline for the Columbia single-stage workflow
2. validate the existing finite-plasma-current surrogate in a unit-safe way

This note now treats those as independent tracks. It does not use `vacuum (no current)` as shorthand, because that phrase is ambiguous in this workflow.

## Executive Summary

The current code has two distinct current concepts:

1. external coil currents that generate the Biot-Savart field used by the workflow
2. a separate Boozer-level finite-plasma-current surrogate exposed as `--boozer-I`

Those should not share one combined "current cap" discussion.

The near-term plan should therefore be:

1. create a `coil-only, zero-plasma-current baseline` track
2. run that baseline with an explicit `tf_current_per_coil_A = 8e4` assumption unless the experiment owner says otherwise
3. keep plasma current fixed to `0 A` in that baseline track
4. create a separate finite-plasma-current track that accepts physical current in `A` only
5. convert that physical current once into internal Boozer `I`
6. state in the executive summary that this finite-current feature is only a `boozer_surrogate` model layered on top of a coil-generated Biot-Savart field, not a self-consistent finite-current equilibrium capability

## Implementation Status

Implemented in the repo:

1. user-facing plasma current in `A`
   - `--plasma-current-A`
2. internal conversion to Boozer `I`
3. raw `--boozer-I` retained as expert/internal input
4. Stage 2 TF-current parameterization
   - `--tf-current-A`
5. Stage 2 TF-current provenance in output naming and metadata
6. explicit run-identity hashing from a frozen config object rather than ambient globals
7. shared Stage 2 path-format helpers plus workflow entrypoints for:
   - an `80 kA` coil-only weighted sweep
   - a reduced finite-current smoke harness

Still not implemented:

1. a self-consistent finite-current equilibrium model
2. a plasma-current contribution added directly to the magnetic field source `B`
3. a toroidal-current profile across surfaces
4. a physically meaningful optimizer-controlled plasma-current ceiling

So the current code should be understood as an improved surrogate workflow and experiment contract, not as the final finite-current physics model.

## What The Current Code Actually Does

### Finite plasma current

The single-stage solver now exposes:

- a user-facing physical-current input in `A`
  - `--plasma-current-A`
- a raw expert/internal Boozer-current input
  - `--boozer-I`

The resolved current value is then threaded into the Boozer-surface setup and residual evaluation:

- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
- `src/simsopt/geo/surfaceobjectives.py`

The residual is:

```text
(G + iota I) B - |B|^2 (x_phi + iota x_theta)
```

The run results now record:

- `PLASMA_CURRENT_A`
- `PLASMA_CURRENT_INPUT_SOURCE`
- `BOOZER_I`

This means the repo supports a unit-safe physical interface and still preserves the raw expert path, but the actual physics scope remains only Boozer-residual level.

### External coil baseline

The field source is still the external Biot-Savart coil set.

In the Stage 2 workflow, the TF current baseline is now parameterized:

- `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py`
  - `--tf-current-A`

The banana coils are also external coils with a separate initialization current scale:

- `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py`
  - `ScaledCurrent(Current(1), 1e4)`

Stage 2 seed outputs now also encode the TF current in their directory naming and metadata:

- `TFC=...` in the Stage 2 output directory name
- `TF_CURRENT_A` in `results.json`

So there are already multiple external-coil current scales in the workflow, independent of the Boozer `I` surrogate, and the TF-current baseline now has explicit provenance.

## Core Interpretation Of The Physicist Feedback

The lead physicist's feedback is best interpreted as:

1. stop mixing external-coil baseline work with finite-plasma-current validation
2. use `A` only for any user-facing plasma-current input
3. treat the requested `80 kA` case as a coil-baseline task with plasma current fixed to zero
4. handle the weighted baseline sweep separately before returning to finite-current validation

The clearest replacement for the old ambiguous phrase is:

- `coil-only, zero-plasma-current baseline`

not:

- `vacuum (no current)`

because external coil currents are still on in that scenario.

## Unit Convention For Plasma Current

SIMSOPT's convention is:

- `G(s)` is `mu0/(2*pi)` times the poloidal current outside the surface
- `I(s)` is `mu0/(2*pi)` times the toroidal current inside the surface

Therefore the correct finite-plasma-current conversion is:

```text
boozer_I = (mu0 / (2*pi)) * plasma_current_A
         = 2e-7 * plasma_current_A
```

Examples:

```text
  0 A    -> 0.0
8000 A   -> 0.0016
16000 A  -> 0.0032
30000 A  -> 0.0060
35200 A  -> 0.00704
```

This remains correct and should still govern the finite-current surrogate interface.

## Why `A`-Only Input Is Better

The previous design discussed accepting physical current in `kA` or `A`.

After the physicist feedback, the stricter and safer interface is:

- accept only physical current in SI `A`

Recommended user-facing field:

- `plasma_current_A`

Recommended internal behavior:

```text
boozer_I = (mu0 / (2*pi)) * plasma_current_A
```

Recommended compatibility rule:

- keep raw `--boozer-I` as internal or expert-only
- reject simultaneous use of raw `--boozer-I` and physical-current inputs

This removes an avoidable unit ambiguity and makes the CLI semantics align with the stated physics quantity.

## Why The Old "Effect May Be Small" Argument Is Too Weak As A Design Premise

The old note estimated smallness using the present external-coil baseline, where the TF currents are hardcoded at `1e5 A` per coil.

That argument is too brittle to be a design premise because:

1. it depends directly on the chosen external-coil baseline
2. the requested baseline is now likely changing from `100 kA` to `80 kA`
3. the workflow still does not add a plasma-current field contribution to `B`; it only changes the Boozer residual scalar combination `G + iota I`

The `iota I / G` ratio can still be useful as a run-specific diagnostic. It should not be used as the main rationale for the interface design.

## Physics Scope That The Note Must State Explicitly

The current finite-current path is not a self-consistent finite-current equilibrium model.

It is only:

- a Boozer-residual surrogate layered on top of a coil-only Biot-Savart field

That means:

1. it does not add a plasma-current contribution directly to the magnetic field source model
2. it does not solve a finite-current equilibrium
3. it does not implement a current profile
4. in multi-surface mode, the same scalar `boozer_I` is currently applied to every optimized surface

This is still useful as a bounded surrogate experiment. It should not be described more strongly than that.

## Track 1: Coil-Only, Zero-Plasma-Current Baseline

Track 1 creates a frozen coil artifact at a specified TF current per coil. Track 2 can consume that artifact immediately; it does not need Track 1's full weighted sweep to finish first.

Definition:

- external coil currents on
- plasma current fixed to `0 A`

Immediate implementation contract:

- set `tf_current_per_coil_A = 8e4` for the requested `80 kA` baseline unless the experiment owner confirms a different meaning
- keep the banana-coil initialization current at the existing `1e4 A` scale unless a separate change explicitly redefines it
- regenerate the Stage 2 seed artifact for that TF-current setting rather than assuming old `100 kA` seeds remain valid
- require TF-current provenance in both seed naming and Stage 2 `results.json`

The `8e4 A per TF coil` interpretation is still an experiment-definition assumption, so it should be confirmed before a large batch, but the repo should treat it as a per-coil quantity, not a total-machine shorthand.

Recommended outcome for this track:

1. make TF current configurable
2. produce a working `80 kA` coil-only, zero-plasma-current baseline
3. run a weighted tradeoff sweep over existing optimization weights
4. extract the non-dominated set afterward

This note deliberately says `weighted tradeoff sweep`, not `Pareto engine`, because the workflow currently exposes scalarized objective weights rather than a dedicated Pareto optimizer.

Likely scan knobs:

- `--cc-weight`
- `--curvature-weight`
- `--length-weight`
- `--res-weight`
- `--iotas-weight`
- optionally `--constraint-method`, if fixed carefully for comparability

## Track 2: Finite Plasma-Current Surrogate Validation

Track 2 is independent from the full Track 1 sweep. It only requires a frozen coil artifact whose provenance is explicit.

Definition:

- external-coil baseline fixed
- vary only plasma current

Recommended user-facing fields:

- `plasma_current_A`

This field should be documented as the enclosed toroidal plasma current for the chosen full-torus configuration, in physical SI amperes.

Recommended result fields:

- `PLASMA_CURRENT_A`
- `PLASMA_CURRENT_INPUT_SOURCE`
- `BOOZER_I`
- `FINITE_CURRENT_MODE = "boozer_surrogate"`

Compatibility warning:

- historical results that only record `BOOZER_I` are not safely back-convertible to physical amperes unless provenance explicitly says that value came from the physical-current interface
- backward compatibility should therefore preserve the old raw-unit interpretation when provenance is absent

Recommended baseline scenarios for this surrogate track:

- `0 A`
- `8000 A` as an engineering smoke test, not a canonical HBT operating point
- `10000 A`
- `15000 A`
- `30000 A` as an HBT-EP-style upper-bound case
- optional literature point `-35200 A` as a constructed reference equilibrium, not a facility operating spec

This set is better aligned with the available HBT / HBT-EP references than the earlier `8/16/30 kA` framing.

## Longer-Term Physical Upgrade Path

If the project needs real finite-current physics rather than a Boozer-level surrogate, the current quantity should move to the equilibrium/current-profile layer:

- VMEC current profile / `curtor`
- SPEC prescribed toroidal current profile
- or a finite-beta plus plasma-current-field workflow

That is the level where a true current ceiling belongs.

## Strengths Of The Updated Plan

- separates external-coil baseline work from plasma-current validation
- removes the ambiguous `vacuum (no current)` wording
- removes `kA` versus `A` ambiguity by using `A` only
- matches the actual repo structure more closely
- keeps the finite-current feature described at the correct physics scope
- creates a cleaner path to a baseline tradeoff sweep and then a controlled surrogate-current sensitivity study

## Weaknesses / Limits

1. The current finite-current implementation still does not model a self-consistent plasma-current equilibrium.
2. The requested `80 kA` interpretation is still an assumption until confirmed at the experiment-definition level.
3. The current workflow is a weighted scalarization workflow, not a dedicated Pareto implementation.
4. Multi-surface runs still apply one scalar `boozer_I` across all surfaces rather than a surface-dependent current model.

## Validation Criteria

Before calling implementation correct, require:

1. The configurable TF-current baseline reproduces current behavior when set to the present `1e5 A` value.
2. The new baseline path runs correctly at the requested `80 kA` setting with plasma current fixed to `0 A`.
3. Physical plasma-current inputs in `A` map to the correct internal `BOOZER_I`.
4. Raw `--boozer-I` and physical-current inputs cannot be used together silently.
5. Result files contain physical current, current-source provenance, and internal current when the physical-current interface is used.
6. Sign convention is documented explicitly for positive and negative toroidal current.
7. Historical result handling preserves raw-unit interpretation when provenance is absent.
8. Both Boozer solver branches are checked with `0 A`, positive current, and negative current at fixed geometry and fixed coil seed.
9. A small surrogate-current sensitivity sweep confirms the finite-current term is actually active and not being lost numerically.

## Decision

The correct near-term implementation is not "add an `8/16 kA` ceiling" in isolation.

The correct near-term implementation is:

1. separate the external-coil baseline problem from the plasma-current surrogate problem
2. make the external TF baseline configurable so a coil-only, zero-plasma-current `80 kA` case can be run
3. expose plasma current in `A` only
4. convert it once into internal Boozer units
5. describe the feature as a project-specific `boozer_surrogate` finite-current mode

The correct long-term implementation remains:

- equilibrium/current-profile-level finite-current control

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

- SIMSOPT VMEC/current docs:
  - `https://simsopt.readthedocs.io/latest/example_vmec.html`
