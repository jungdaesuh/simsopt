# DESC banana-coil single-stage feasibility report

Date: 2026-06-26

## Verdict

Banana-coil single-stage optimization is possible in `simsopt-surrogate` today through the existing SIMSOPT/Boozer/CWS lane, but it is not currently possible as a DESC-backed one-flag mode.

A DESC-backed banana single-stage workflow is engineering-feasible as a new bridge/lane. It needs explicit conversion between SIMSOPT coil/surface objects and DESC equilibrium/coil objects, plus validation that keeps the existing SIMSOPT Poincare/Boozer and CAD hardware-contact oracle as final evidence. DESC objective values alone would not prove a banana artifact is confined, buildable, or hardware-clean.

## Live evidence

### Current repository state

- `simsopt-surrogate` checkout: `/Users/suhjungdae/code/columbia/simsopt-surrogate`
- Branch: `surrogate-confinement-v2`
- HEAD: `8d5b54996 topology: add invariant_manifold module (Wu/Ws grower + turnstile flux) for 2/9 X-point tangle analysis`
- Worktree is dirty before this report; source/test edits already existed and were not touched by this report.
- DESC checkout: `/Users/suhjungdae/code/opensource/DESC`
- DESC HEAD: `c119da0f8 Deprecate constants (#1769)`

### Existing SIMSOPT banana single-stage lane

The current banana single-stage entrypoint is:

- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`

Its CLI describes the run as single-stage Boozer/quasi-symmetry optimization from a Stage 2 seed. It consumes VMEC `wout`/equilibrium paths and Stage 2 artifacts, not DESC runtime objects.

Relevant current capabilities in this lane:

- Stage-2 to single-stage handoff from VMEC/wout surfaces.
- Boozer and iota bootability checks before expensive single-stage runs.
- CWS winding-surface shape freedom through `CurveCWSFourierCPP`.
- Banana current modes, including shared finite-current banana pack handling.
- Finite-build pack field swap into `BiotSavart(tf + pack + proxy + vf)`.
- SIMSOPT coil penalties for length, curvature, curve-curve distance, curve-surface distance, force, and related geometry terms.
- Hardware status separation in the docs between search-time steering and artifact-level certification.

A focused source search found no DESC imports or DESC runtime calls in the banana single-stage path or `banana_opt` Python modules. The only relevant `desc` occurrence in the searched banana path was a type literal spelling `desc` for sort direction. Some artifact names contain `desc`, for example a `wout_*_desc_*` file, but those are VMEC/wout inputs rather than DESC optimization calls.

### DESC primitives that make a bridge feasible

DESC has the building blocks for a DESC-native or DESC-assisted lane:

- Coil models: `FourierRZCoil`, `FourierXYZCoil`, `CoilSet`, `MixedCoilSet`.
- Multi-object optimization over equilibrium and coil objects.
- Coil geometry objectives: `CoilLength`, `CoilCurvature`, `CoilSetMinDistance`, `PlasmaCoilSetDistanceBound`, `PlasmaCoilSetMinDistance`.
- Field/objective terms: `QuadraticFlux`, `SurfaceQuadraticFlux`, `BoundaryError`, `VacuumBoundaryError`, `LinkingCurrentConsistency`.
- Test coverage exists for coil optimization with `QuadraticFlux`, and for multi-thing objective wiring with `eq` plus coil objects.

There is an important DESC objective boundary:

- `QuadraticFlux` is appropriate for coil optimization against a fixed equilibrium.
- DESC's optimizer rejects `QuadraticFlux` when an `Equilibrium` is one of the optimized things, because that objective assumes the equilibrium is fixed.
- A true joint equilibrium-plus-coil single-stage DESC lane should use `BoundaryError` or `VacuumBoundaryError` instead of `QuadraticFlux`.

## Feasible implementation lanes

### Lane A: fixed-DESC-equilibrium banana coil polish

This is the lowest-risk DESC integration.

Use a fixed DESC equilibrium or a VMEC/wout-derived DESC equilibrium and optimize only DESC coil objects. The objective stack can include:

- `QuadraticFlux(eq=eq, field=coilset, ...)`
- `LinkingCurrentConsistency(eq=eq, coil=coilset, eq_fixed=True)`
- `CoilLength`
- `CoilCurvature`
- `CoilSetMinDistance`
- `PlasmaCoilSetMinDistance(eq=eq, coil=coilset, eq_fixed=True)`

This would be a coil-polish or cross-check lane, not a full replacement for SIMSOPT's current Boozer single-stage optimizer.

### Lane B: true DESC joint equilibrium-plus-banana-coil single-stage

This is feasible but materially larger.

Optimize both the DESC equilibrium and the DESC coil set. The objective stack should use:

- `BoundaryError` or `VacuumBoundaryError`
- DESC equilibrium force-balance/current/profile objectives appropriate to the chosen model
- iota/profile objectives as available in DESC
- `LinkingCurrentConsistency(eq_fixed=False)`
- `PlasmaCoilSetMinDistance(eq_fixed=False)`
- coil geometry objectives such as length, curvature, coil-coil distance, and coil-plasma distance

This lane is the closest conceptual match to "banana coils single stage in DESC," but it needs careful parity work before any artifact-level claim.

### Lane C: drop-in replacement inside the current SIMSOPT script

This is not currently realistic as a quick switch.

The current script is built around SIMSOPT `BoozerSurface`, `Iotas`, `VolumeBoozer`, `NonQuasiSymmetricRatio`, `BiotSavart`, `CurveCWSFourierCPP`, and SIMSOPT derivative plumbing. DESC does not directly consume those objects. A drop-in replacement would become a hidden rewrite unless the bridge is built and validated first.

## Required bridge

A production-grade DESC bridge needs explicit, tested conversions in both directions:

1. Surface and equilibrium conversion
   - Preserve NFP, stellarator symmetry, handedness, scale, major/minor radius conventions, and angular coordinates.
   - Treat VMEC/wout provenance carefully: a filename containing `desc` is not evidence that DESC runtime optimization was used.

2. Coil conversion
   - Convert SIMSOPT curves to DESC `FourierXYZCoil` or another DESC coil representation through sampled coordinates or coefficient-level mapping.
   - Preserve current signs, current units, coil grouping, banana-pack topology, TF/proxy/VF separation, and CW/CCW handedness.
   - Do not retarget proxy or VF coils as a side effect of enabling DESC.

3. Objective mapping
   - Keep fixed-equilibrium coil polish separate from true joint equilibrium-plus-coil single-stage.
   - Do not use `QuadraticFlux` for joint equilibrium-plus-coil optimization.
   - Keep SIMSOPT hardware SDF/CAD oracle evidence separate from DESC distance penalties.

4. Export and validation
   - Export optimized DESC coils back to SIMSOPT/MAKEGRID-compatible artifacts.
   - Re-run existing SIMSOPT Poincare/Boozer checks.
   - Re-run existing hardware contact / keep-out oracle.
   - Report search-time steering, final artifact hardware status, and direct loaded-artifact evidence as separate fields.

## Main risks

- Object model mismatch: DESC and SIMSOPT coil/surface classes are not interchangeable.
- Sign and coordinate convention drift can silently invert or rotate banana geometry.
- DESC coil-distance terms are not a substitute for the banana hardware SDF/GLB/CAD-contact oracle.
- Fixed-equilibrium coil polish can look useful while failing to constitute a true single-stage equilibrium-plus-coil solve.
- Existing SIMSOPT single-stage and Stage 2 workflows have not been proven to globally squeeze all available 3D vessel space; a DESC bridge should not inherit stronger claims without new evidence.

## Recommended next step

Build Lane A first as an additive experiment:

1. Add a small conversion module that imports one existing SIMSOPT banana artifact into DESC coils and a fixed DESC/VMEC-derived equilibrium.
2. Validate a round-trip by comparing sampled coil coordinates, current signs, coil lengths, minimum distances, and field samples before and after conversion.
3. Run a short fixed-equilibrium DESC coil-polish solve.
4. Export back to SIMSOPT and run the existing Poincare/Boozer and hardware-contact validation stack.

Only after that parity path is stable should the project attempt Lane B, the true DESC joint equilibrium-plus-banana-coil single-stage optimizer.
