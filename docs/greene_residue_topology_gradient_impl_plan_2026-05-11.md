# Greene-Residue Topology-Gradient Implementation Plan

Date: 2026-05-11  
Status: Draft implementation plan, not approved for optimizer use  
Scope: `examples/single_stage_optimization/banana_opt/` first; possible promotion only after convention, branch-tracking, and Taylor-test gates pass

## Goal

Add the first serious gradient-based topology objective for banana coil optimization:

```text
fixed target rational branches
+ periodic field-line solve
+ monodromy / tangent-map integration
+ Greene residue objective
+ branch-tracked implicit differentiation
+ BiotSavart B_and_dB_vjp coil gradient
```

The objective is to reduce selected magnetic-island-producing residues while preserving the existing smooth surface/QS/engineering objective stack and strict Poincare validation.

## Executive Summary

- [ ] Build Greene residue before Cary-Hanson / Geraldini island-width sensitivity.
- [ ] Treat Greene residue as a topology-gradient objective, not a topology certificate.
- [ ] Keep BoozerResidual / NQS / Iotas as the smooth surface and QS backbone.
- [ ] Keep strict Poincare / field-line survival validation as the final judge.
- [ ] Freeze target rationals from the target iota profile; do not choose rationals adaptively from the current iterate.
- [ ] Track both O and X branches for island elimination.
- [ ] Use full-torus return maps in v0 to avoid field-period and `nfp` convention traps.
- [ ] Use `B_and_dB_vjp`, not only `B_vjp`, for production residue gradients because the monodromy map depends on `grad B`.
- [ ] Keep HRCA as a lower-priority auxiliary finite-time leakage-risk experiment.

## Current Code Context

### Already Present

- [x] `examples/single_stage_optimization/topology_scorer.py` computes hard field-line topology diagnostics: survival, line lifetimes, stop-reason counts, surrogate confinement loss, and KAM-like spread metrics.
- [x] `src/simsopt/field/tracing.py::compute_fieldlines()` calls `simsoptpp.fieldline_tracing(...)`.
- [x] `src/simsoptpp/tracing.cpp` implements the current C++/Boost odeint-style tracing path with hard stopping and Poincare-hit logic.
- [x] `src/simsopt/field/biotsavart.py` exposes `B_vjp(v)` and `B_and_dB_vjp(v, vgrad)`.
- [x] `src/simsoptpp/magneticfield.h` and Python bindings expose `d2B_by_dXdX`.
- [x] `src/simsopt/geo/surfaceobjectives.py` contains BoozerResidual, NQS, Iotas, and Boozer-surface derivative plumbing.
- [x] `src/simsopt/mhd/spec.py::Residue` computes Greene residue from a SPEC equilibrium using `pyoculus`, but it is not a direct BiotSavart coil-gradient objective.
- [x] Single-stage objective wiring already includes `JnonQSRatio + RES_WEIGHT * JBoozerResidual + IOTAS_WEIGHT * Jiota`.
- [x] Stage 2 default objective remains mostly `SquaredFlux + engineering` and has no default topology-gradient term.

### Missing For Direct BiotSavart Residue

- [ ] Direct BiotSavart field-line return-map integrator in a fixed convention.
- [ ] Periodic-orbit solver for fixed rational branches.
- [ ] Branch tracking and branch identity metadata.
- [ ] Monodromy / tangent-map integration for the chosen map.
- [ ] Residue convention tests.
- [ ] Direct coil-DOF derivative path through the periodic orbit and tangent map.
- [ ] Taylor tests for the full branch-resolved objective.
- [ ] Optimizer integration as an optional, low-weight term.

## Non-Goals

- [ ] Do not replace strict Poincare validation.
- [ ] Do not differentiate the existing hard-stop field-line scorer.
- [ ] Do not modify `src/simsoptpp/tracing.cpp` in v0.
- [ ] Do not use the SPEC/pyoculus `Residue` object as the production direct-coil gradient implementation.
- [ ] Do not use adaptive rational selection from the current iterate in v0.
- [ ] Do not start with island-width sensitivity before residue value, branch tracking, and residue gradient tests pass.
- [ ] Do not make residue a default Stage 2 objective.
- [ ] Do not promote to `src/simsopt/` until the banana-local prototype is stable.

## Convention Lock

This project must lock conventions before implementation. Most residue bugs will be convention bugs.

### Rotational Transform

Use one of these equivalent conventions, and name it explicitly in code:

```text
radian angles:
    iota = Delta theta_rad / Delta phi_rad

normalized turns:
    iota = (Delta theta_rad / 2pi) / (Delta phi_rad / 2pi)
```

For a map rational:

```text
iota = p / q
```

For a Fourier resonance:

```text
m * iota - n = 0
iota = n / m
```

Do not pass raw `(m, n)` or `(p, q)` integers without declaring which convention they use.

### v0 Map Convention

Use a full-torus return map in v0:

```text
P_c: section phi = phi0 -> section phi = phi0 + 2pi
```

For reduced `iota = p / q`, solve:

```text
F(z, c) = P_c^q(z; c) - z = 0
```

and require winding:

```text
Delta theta_rad = 2pi * p
```

Reason: full-torus maps avoid the common `nfp` field-period trap. One-field-period maps can be added later with explicit tests.

### Residue Definition

Use Greene's residue:

```text
R_G = (2 - trace(M)) / 4
```

where:

```text
M = D_z P_c^q(z*)
```

Classification:

- [ ] `0 < R_G < 1`: elliptic / O-point branch.
- [ ] `R_G < 0` or `R_G > 1`: hyperbolic / X-point branch.
- [ ] `R_G = 0`: parabolic / rational-surface limit; island-elimination target.
- [ ] `R_G = 1`: period-doubling boundary; not an island-elimination target.

For island elimination:

```text
target residue = 0
```

Do not call `R = 1` a good target. It is also marginal, but it is the wrong marginal point for island elimination.

## Mathematical Plan

### Field-Line State

Use toroidal angle `phi` as the independent variable.

```text
x(R, phi, Z) = [R cos(phi), R sin(phi), Z]
y(phi) = [R(phi), Z(phi)]
```

Field components:

```text
B_R   = B(x; c) dot e_R
B_phi = B(x; c) dot e_phi
B_Z   = B(x; c) dot e_Z
```

RHS:

```text
dR/dphi = R * B_R / B_phi
dZ/dphi = R * B_Z / B_phi
```

Gate every evaluation on:

```text
min_orbit |B_phi| / |B| > configured threshold
```

Do not hide low-`B_phi` failures with denominator regularization in v0.

### Periodic Orbit

Let `z = [R, Z]` on section `phi = phi0`.

For a target rational `iota = p / q`:

```text
F(z, c) = P_c^q(z; c) - z = 0
```

Required branch checks:

- [ ] closure residual below tolerance.
- [ ] unwrapped poloidal winding equals `p`.
- [ ] orbit remains inside the target radial window.
- [ ] branch classification matches expected O or X branch.
- [ ] `min |B_phi| / |B|` stays above threshold.

### Tangent Map

Integrate:

```text
dY/dphi = A(phi) * Y
Y(0) = I
A = dV/dz
```

At the periodic orbit:

```text
M = Y(2pi * q)
```

Mandatory diagnostic:

```text
det(M) ~= 1
```

Do not silently normalize the determinant in the objective. If normalized residue is useful for diagnostics, report it separately.

### Objective

For each rational target and branch:

```text
J_branch = rho(R_G / R_scale)
```

v0 robust loss:

```text
rho(x) = 0.5 * x^2
```

Only after Taylor tests pass, consider pseudo-Huber or capped quadratic for wild failed branches.

Total:

```text
J_residue = sum_targets sum_branches weight[target, branch] * rho(R_G / R_scale)
```

## Target-Rational Selection

### Selection Rule

Use the target iota profile, not the current iterate:

- [ ] Define target profile source.
- [ ] Choose low-order rationals `p / q` that lie inside the target confinement domain.
- [ ] Assign each rational a target radial label `s_star`.
- [ ] Assign a radial search window around `s_star`.
- [ ] Freeze this rational set in the run manifest.
- [ ] Keep Iotas penalties active to prevent iota gaming.

### Initial v0 Rational Set

- [ ] Start with a small set, not all rationals.
- [ ] Prefer low-order rationals, for example `q <= 8` or `q <= 12`.
- [ ] Include edge-relevant rationals if strict Poincare validation shows edge islands.
- [ ] Add new rationals only after validation identifies persistent untracked island chains.
- [ ] Do not remove a target because optimization made it inconvenient.

### Iota-Gaming Guard

Reject residue-only improvement if:

- [ ] iota target penalty worsens beyond tolerance.
- [ ] rational branch leaves its radial window.
- [ ] branch winding changes.
- [ ] residue drops because the target resonance moved out of the intended domain.

## Architecture Decision

### First Location

Implement under:

```text
examples/single_stage_optimization/banana_opt/topology/
```

Candidate files:

- [ ] `examples/single_stage_optimization/banana_opt/topology/__init__.py`
- [ ] `examples/single_stage_optimization/banana_opt/topology/rational_target.py`
- [ ] `examples/single_stage_optimization/banana_opt/topology/poincare_chart.py`
- [ ] `examples/single_stage_optimization/banana_opt/topology/fieldline_map.py`
- [ ] `examples/single_stage_optimization/banana_opt/topology/periodic_orbit.py`
- [ ] `examples/single_stage_optimization/banana_opt/topology/greene_residue.py`
- [ ] `examples/single_stage_optimization/banana_opt/topology/residue_objective.py`
- [ ] `examples/single_stage_optimization/banana_opt/topology/diagnostics.py`

Rationale: this is experimental banana optimization work first. Promote to `src/simsopt/` only after the API and tests prove general value.

### Required Data Classes

#### `RationalTarget`

- [ ] `p`
- [ ] `q`
- [ ] `weight`
- [ ] `radial_label`
- [ ] `radial_window`
- [ ] `branches`
- [ ] `phi0`
- [ ] `nfp`
- [ ] `convention`
- [ ] optional Fourier `(m, n)` metadata

#### `PoincareChart`

- [ ] angle units, origin, orientation, and winding sign
- [ ] radial-label definition
- [ ] symmetry-equivalent branch rule
- [ ] circular / analytic fixture test for `[R, Z] -> (radial_label, theta)`

#### `BranchState`

- [ ] target id
- [ ] branch label: `O` or `X`
- [ ] section coordinate `z0`
- [ ] winding
- [ ] radial label
- [ ] previous residue
- [ ] branch status
- [ ] continuation generation / accepted-iteration id

#### `GreeneResidueResult`

- [ ] target
- [ ] branch
- [ ] `z0`
- [ ] `M`
- [ ] `residue`
- [ ] `traceM`
- [ ] `detM`
- [ ] winding
- [ ] radial label
- [ ] Newton residual
- [ ] Newton iterations
- [ ] `min_Bphi_over_B`
- [ ] status
- [ ] diagnostics

### Public Objective Class

```python
class BiotSavartResidue(Optimizable):
    """Experimental direct-coil Greene-residue objective."""
```

The optimizer-facing class must register the field dependency with `Optimizable.__init__(..., depends_on=[biotsavart])` or an equivalent local wrapper. Diagnostic helpers may stay plain Python classes, but objective arithmetic must use the `Optimizable` wrapper.

Initial constructor contract:

- [ ] `biotsavart`: direct BiotSavart field, not a generic field lacking `B_and_dB_vjp`.
- [ ] `targets`: immutable list of `RationalTarget`.
- [ ] `chart`: fixed section/radial/poloidal chart.
- [ ] `orbit_solver`: deterministic periodic-orbit solver.
- [ ] `integrator_options`: fixed tolerances and step/solver contract.
- [ ] `residue_scale`.
- [ ] `branch_cache_policy`.
- [ ] `failure_policy`.

Initial methods:

- [ ] `J() -> float`
- [ ] `dJ() -> Derivative`
- [ ] `residues() -> np.ndarray`
- [ ] `diagnostics() -> list[GreeneResidueResult]`
- [ ] `clear_branch_cache()`

## Derivative Strategy

### v0 Derivative Roadmap

Do not jump straight to production reverse mode.

1. [ ] Value-only residue.
2. [ ] Frozen-orbit finite differences.
3. [ ] Branch-resolved finite differences with orbit re-solve.
4. [ ] Forward directional sensitivity for selected coil directions.
5. [ ] Production adjoint / VJP using `B_and_dB_vjp`.

### Why `B_and_dB_vjp`

Residue depends on:

```text
M = D_z P_c^q(z*)
```

The monodromy map depends on:

```text
A = dV/dz
```

where `V` is the phi-parametrized field-line RHS. Therefore residue gradients require variation of both:

```text
B
grad B
```

Production coil VJP must use:

```text
dB_part, dgradB_part = biotsavart.B_and_dB_vjp(v_B, v_gradB)
dJ_dcoils = dB_part + dgradB_part
```

`B_vjp` alone is insufficient for the full residue gradient.

### Required Field Values

At orbit/tangent samples:

- [ ] `B`
- [ ] `dB_by_dX`
- [ ] `d2B_by_dXdX`

For coil-gradient accumulation:

- [ ] cotangents for `B`
- [ ] cotangents for `dB_by_dX`
- [ ] `B_and_dB_vjp`

### MagneticFieldSum Caveat

The checked local code exposes `B_vjp` on `MagneticFieldSum`, but not a generic `B_and_dB_vjp` aggregator.

- [ ] Require direct `BiotSavart` for v0.
- [ ] Do not assume every `MagneticField` supports `B_and_dB_vjp`.
- [ ] Add an explicit fail-fast interface check.
- [ ] Only add aggregate `B_and_dB_vjp` support if a real non-BiotSavart use case appears.

## Implementation Phases

## Phase 0: Convention And Reference Tests

Objective: make convention drift impossible.

- [ ] Add `tests/geo/test_greene_residue_conventions.py`.
- [ ] Define `RationalTarget` with explicit `p/q` and optional Fourier metadata.
- [ ] Add tests for `iota = p / q`.
- [ ] Add tests for Fourier resonance `m * iota - n = 0`.
- [ ] Add tests for full-torus vs field-period map metadata.
- [ ] Add residue sign/classification unit tests.
- [ ] Add deterministic string representation for target manifests.

Exit criteria:

- [ ] Every target declares its convention.
- [ ] No code path accepts ambiguous bare `(m, n)` or `(p, q)` pairs.

## Phase 1: Value-Only Field-Line Return Map

Objective: compute `P_c^q(z)` in the locked convention.

- [ ] Implement cylindrical basis and phi-parametrized RHS.
- [ ] Implement deterministic full-torus return map.
- [ ] Track unwrapped poloidal angle for winding checks.
- [ ] Report `min_Bphi_over_B`.
- [ ] Add analytic-field sanity tests.
- [ ] Compare section-hit geometry / return-map locations against existing `compute_fieldlines` where comparable; do not compare raw time samples.

Exit criteria:

- [ ] Return map is deterministic.
- [ ] Winding checks are reliable.
- [ ] Low-`B_phi` cases are reported and gated.

## Phase 2: Tangent Map And Residue Value

Objective: compute `M`, `detM`, `traceM`, and `R_G`.

- [ ] Implement `A = dV/dz`.
- [ ] Integrate `dY/dphi = A Y`.
- [ ] Compute `M = Y(2pi * q)`.
- [ ] Compute `R_G = (2 - trace(M)) / 4`.
- [ ] Report `detM`.
- [ ] Add tangent-map finite-difference tests.
- [ ] Add area-preservation tests.
- [ ] Add starting-point invariance tests.

Exit criteria:

- [ ] `M v` matches finite-difference return-map perturbations.
- [ ] `detM` is close to 1 under refinement.
- [ ] residue sign/classification matches known or validated branches.

## Phase 3: Periodic-Orbit Solver

Objective: find and validate fixed branches.

- [ ] Implement damped Newton / trust-region solve for `F(z, c) = P_c^q(z; c) - z`.
- [ ] Use `F_z = M - I`.
- [ ] Add radial-window checks.
- [ ] Add winding checks.
- [ ] Add branch classification checks.
- [ ] Support multistart initial guesses for the initial branch discovery only.
- [ ] Support continuation from prior accepted branch state during optimization.
- [ ] Fail loudly on wrong winding or branch switch.

Initial guess sources:

- [ ] reference surface / target radial label
- [ ] diagnostic Poincare cloud
- [ ] previous accepted branch state
- [ ] symmetry or phase-shift guesses

Exit criteria:

- [ ] periodic-orbit closure residual passes strict tolerance.
- [ ] branch identity is stable under small coil perturbations.
- [ ] O/X branches are both found where expected.

## Phase 4: Value-Only Diagnostics In Banana Runs

Objective: make residue observable before gradients.

- [ ] Add a diagnostic runner, for example `run_residue_probe.py`.
- [ ] Evaluate fixed rational branches on existing initial/optimized artifacts.
- [ ] Write JSON diagnostics:
  - target id
  - branch
  - residue
  - `detM`
  - winding
  - radial label
  - branch status
  - `min_Bphi_over_B`
  - solver iterations
- [ ] Overlay periodic points on strict Poincare plots.
- [ ] Compare residue diagnostics against visible island chains.

Exit criteria:

- [ ] residue diagnostics align with Poincare evidence on known cases.
- [ ] false branch matches are caught by status flags.

## Phase 5: Directional Sensitivity Debugging

Objective: prove derivative formulas before production VJP.

- [ ] Add frozen-orbit finite-difference tests.
- [ ] Add branch-resolved central finite differences with orbit re-solve.
- [ ] Implement forward directional sensitivity for selected coil perturbations.
- [ ] Verify derivative of `P_c^q`.
- [ ] Verify derivative of `M`.
- [ ] Verify derivative of `R_G`.
- [ ] Verify derivative of branch location `z*(c)`.

Exit criteria:

- [ ] directional derivatives match central finite differences over a usable epsilon range.
- [ ] derivative failures can be localized to map, tangent map, IFT solve, or branch tracking.

## Phase 6: Production Adjoint / VJP

Objective: compute gradient with respect to all free coil DOFs.

- [ ] Implement discrete adjoint or custom reverse pass for the selected integrator.
- [ ] Include implicit differentiation through the periodic orbit solve.
- [ ] Accumulate cotangents for `B`.
- [ ] Accumulate cotangents for `dB_by_dX`.
- [ ] Call `B_and_dB_vjp(v_B, v_gradB)`.
- [ ] Sum the returned `B` and `grad B` derivative contributions before returning the SIMSOPT `Derivative`.
- [ ] Convert result into SIMSOPT-compatible derivative format.
- [ ] Add debug mode for adjoint norm and cotangent diagnostics.

Exit criteria:

- [ ] full objective Taylor tests pass.
- [ ] gradients are deterministic.
- [ ] gradients remain stable under integrator refinement.

## Phase 7: Objective Wiring

Objective: expose residue as an optional objective term.

- [ ] Add low-level objective class.
- [ ] Add optional single-stage wiring behind explicit flag.
- [ ] Default residue weight must be zero.
- [ ] Require target-rational manifest for nonzero weight.
- [ ] Require passing validation artifact before optimizer use.
- [ ] Log all residue diagnostics per accepted iteration.
- [ ] Do not change frontier dominance semantics in v0.
- [ ] Do not make residue a default Stage 2 term.

Candidate objective:

```text
J_total =
    J_NQS
  + RES_WEIGHT * J_BoozerResidual
  + IOTAS_WEIGHT * J_iota
  + engineering terms
  + RESIDUE_WEIGHT * J_residue
```

## Validation Matrix

### Required Tests Before Optimizer Use

- [ ] Map convention test: known `iota = p / q` orbit returns after `q` full-torus turns.
- [ ] Winding test: unwrapped poloidal winding equals `p`.
- [ ] Tangent-map finite-difference test: `M v` matches return-map perturbation.
- [ ] Area-preservation test: `det(M) - 1` below tolerance.
- [ ] Residue sign test: known elliptic and hyperbolic branches classify correctly.
- [ ] Starting-point invariance test: different points on same periodic orbit give same residue.
- [ ] Chart invariance test where a smooth chart transform is available.
- [ ] Poincare overlay test: periodic points land on intended O/X structures.
- [ ] Newton residual gate: `|P^q(z*) - z*|` below tolerance.
- [ ] Integrator convergence: `z*`, `M`, `R_G`, and gradient stable under refinement.
- [ ] Frozen-orbit derivative test.
- [ ] IFT orbit derivative test.
- [ ] Full random-direction Taylor test.
- [ ] `B_and_dB_vjp` dot test at sampled orbit points.
- [ ] Branch-stability perturbation test.
- [ ] Banana regression test on fixed known artifacts.

### Taylor-Test Contract

For random coil direction `v`:

```text
J(c + eps v) - J(c) - eps * gradJ dot v = O(eps^2)
```

Required before optimizer use:

- [ ] log-log slope near 2 over a usable epsilon window.
- [ ] central finite-difference directional derivative agrees with `gradJ dot v`.
- [ ] branch id and winding unchanged across the tested epsilon window.

### Strict Validation Contract

Residue improvement is not accepted as sufficient unless:

- [ ] strict Poincare validation does not worsen.
- [ ] survival/lifetime metrics do not worsen.
- [ ] BoozerResidual does not worsen beyond tolerance.
- [ ] NQS / QS metric does not worsen beyond tolerance.
- [ ] Iotas target penalty does not worsen beyond tolerance.
- [ ] engineering constraints remain valid.

## Branch-Tracking Failure Modes

- [ ] Newton converges to wrong winding.
- [ ] Newton converges to a different radial branch.
- [ ] branch switches from O to X or X to O unexpectedly.
- [ ] branch jumps to symmetry-related copy that is not declared equivalent.
- [ ] branch disappears at bifurcation.
- [ ] branch becomes non-isolated near `R_G -> 0`.
- [ ] branch enters low-`B_phi` region.
- [ ] multiple nearby roots make continuation ambiguous.
- [ ] low shear makes radial localization unreliable.
- [ ] island overlap / stochastic layer prevents clean isolated branch tracking.

## Handling Near Success

Near `R_G = 0`, `M - I` becomes ill-conditioned because the isolated periodic point approaches a rational-surface family.

Policy:

- [ ] Define `R_satisfied`.
- [ ] Once `|R_G| < R_satisfied`, mark branch as satisfied.
- [ ] Reduce or freeze that branch's gradient contribution.
- [ ] Keep branch monitored diagnostically.
- [ ] Let BoozerResidual, Iotas, and Poincare validation maintain the achieved surface quality.

Do not continue pushing a branch through a singular IFT solve.

## Kill Criteria

Demote Greene residue to diagnostic-only if any persist:

- [ ] Taylor tests fail.
- [ ] `det(M)` is not close to 1 under refinement.
- [ ] branch tracking succeeds too rarely under optimizer-scale perturbations.
- [ ] optimizer reduces residue by moving iota away from target profile.
- [ ] residue improves while strict Poincare validation worsens.
- [ ] most targets are already near `R_G = 0` and IFT singular from the beginning.
- [ ] low `B_phi` invalidates common branches.
- [ ] runtime is dominated by branch finding before value diagnostics prove useful.
- [ ] residue improvements do not correlate with island reduction or strict topology metrics.

## Minimum Value Experiment

Run matched experiments with identical optimizer budget and initial artifacts.

### Arms

- [ ] A: baseline smooth single-stage stack.
- [ ] B: baseline plus value-only residue diagnostics.
- [ ] C: baseline plus low-weight residue objective after gradients pass.
- [ ] D: baseline plus residue plus HRCA diagnostics only.
- [ ] E: baseline plus residue plus later island-width objective, only after residue is stable.

### Success Metrics

- [ ] target residues move toward zero without iota gaming.
- [ ] O and X branches both improve for selected rational chains.
- [ ] strict Poincare plots show reduced island structures.
- [ ] survival/lifetime metrics improve or do not degrade.
- [ ] BoozerResidual / NQS / Iotas remain within tolerances.
- [ ] engineering constraints remain valid.
- [ ] branch tracking remains stable across accepted optimizer steps.
- [ ] runtime remains acceptable relative to optimization loop budget.

## Priority Relative To Other Work

1. [ ] Preserve existing BoozerResidual / NQS / Iotas / engineering stack.
2. [ ] Implement value-only direct BiotSavart Greene residue.
3. [ ] Validate conventions, branch tracking, tangent map, and residue values.
4. [ ] Implement directional sensitivities.
5. [ ] Implement production `B_and_dB_vjp` adjoint/VJP.
6. [ ] Add optional low-weight residue objective.
7. [ ] Keep strict Poincare validation as final gate.
8. [ ] Add Geraldini / Cary-Hanson island-width sensitivity only after residue is stable.
9. [ ] Keep HRCA as auxiliary finite-time risk experiment.
10. [ ] Consider QFM / ghost / turnstile objectives as longer-term research.

## Literature Anchors

Core:

- [ ] Greene residue criterion: periodic-orbit residue as topology diagnostic.
- [ ] Hanson and Cary / Cary and Hanson: stellarator island and stochasticity control using residue and small-island theory.
- [ ] Geraldini, Landreman, Paul (`arXiv:2102.04497`): adjoint sensitivity of island width and residue with respect to magnetic-field variations.
- [ ] SIMSOPT `example_islands`: practical SPEC residue workflow minimizing O/X residues for selected rationals.
- [ ] SIMSOPT BiotSavart derivative docs: `B_vjp`, `B_and_dB_vjp`, and field spatial derivatives.

Adjacent:

- [ ] Giuliani et al. Boozer-surface optimization: why BoozerResidual / NQS / Iotas remain the smooth backbone.
- [ ] QFM and ghost-surface literature: almost-invariant-surface alternatives.
- [ ] Turnstile-area / lobe-transport work: nonlocal chaotic transport metrics for later phases.
- [ ] Shadowing / chaotic sensitivity literature: caution against interpreting finite local metrics as full transport proof.
- [ ] Particle-confinement metrics: field-line topology is not particle confinement.

## Final Decision Gate

Residue can move from diagnostic to optional optimizer term only after all are true:

- [ ] convention tests pass.
- [ ] value-only residues align with Poincare evidence.
- [ ] branch tracking is stable.
- [ ] tangent-map and determinant tests pass.
- [ ] full Taylor tests pass.
- [ ] iota-gaming guard is active.
- [ ] strict topology validation does not worsen.
- [ ] existing smooth physics and engineering objectives remain within tolerance.
- [ ] residue remains optional and non-default.
