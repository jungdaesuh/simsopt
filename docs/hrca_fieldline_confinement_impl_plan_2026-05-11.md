# HRCA Field-Line Confinement Implementation Plan

Date: 2026-05-11  
Status: Draft implementation plan, not approved for optimizer use  
Scope: `examples/single_stage_optimization/banana_opt/` first; possible later promotion only after validation gates pass

## Goal

Prototype HRCA as a bounded auxiliary objective for banana coil optimization:

```text
fixed-horizon phi-parametrized field lines
+ smooth finite-time wall / radial risk loss
+ manual discrete adjoint of the implemented integrator
+ BiotSavart B_vjp coil gradient
```

The purpose is to test whether a differentiable finite-time field-line risk signal catches leakage modes that are missed by the existing smooth surface proxies.

## Executive Summary

- [ ] Keep Greene-residue / Geraldini-style island sensitivity as the higher-priority serious topology-gradient path.
- [ ] Treat HRCA as an experimental auxiliary finite-time leakage-risk term only.
- [ ] Do not use HRCA as a topology certificate.
- [ ] Do not replace BoozerResidual, NQS, Iotas, Greene residue, QFM / turnstile research, or strict Poincare validation.
- [ ] Build HRCA outside the current C++ tracer path.
- [ ] Require Taylor tests, holdout seeds, horizon convergence, and strict topology-correlation evidence before any optimizer coupling.

## Current Code Context

### Already Present

- [x] `examples/single_stage_optimization/topology_scorer.py` traces field lines and reports survival, lifetime, stop-reason, surrogate confinement loss, and KAM-like spread metrics.
- [x] `src/simsopt/field/tracing.py::compute_fieldlines()` wraps `simsoptpp.fieldline_tracing`.
- [x] `src/simsoptpp/tracing.cpp` provides the current C++/Boost odeint-style tracer with hard stopping and Poincare-hit logic.
- [x] `src/simsopt/field/biotsavart.py` exposes `B_vjp()` and `B_and_dB_vjp()` after `set_points(...)`.
- [x] `src/simsopt/geo/surfaceobjectives.py` contains BoozerResidual, NonQuasiSymmetricRatio, Iotas, and Boozer-surface derivative plumbing.
- [x] `examples/single_stage_optimization/banana_opt/single_stage_objectives.py` wires `JnonQSRatio + RES_WEIGHT * JBoozerResidual + IOTAS_WEIGHT * Jiota` into the single-stage objective.
- [x] `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py` default objective remains `SquaredFlux + engineering` terms, with no default topology/confinement objective.

### Missing For HRCA

- [ ] A differentiable field-line integrator path that avoids hard stops.
- [ ] A smooth signed-distance object with both value and `grad_xyz`.
- [ ] A discrete adjoint / reverse pass for the implemented fixed-step integrator.
- [ ] Batched accumulation of magnetic-field cotangents for `B_vjp`.
- [ ] Validation tests proving the gradient is the derivative of the scalar objective actually optimized.
- [ ] Holdout validation proving HRCA is not just fitting selected seeds.

## Non-Goals

- [ ] Do not modify `src/simsoptpp/tracing.cpp` for HRCA v0.
- [ ] Do not differentiate the existing hard-stop topology scorer.
- [ ] Do not optimize hard survival fraction, hard first-exit time, hard Poincare hit counts, or hard KAM classification.
- [ ] Do not use `InterpolatedField` in the differentiable path.
- [ ] Do not require `B_and_dB_vjp` in HRCA v0.
- [ ] Do not promote HRCA to core `src/simsopt/` until the examples-level prototype passes all gates.
- [ ] Do not let HRCA dominate the objective, even if it passes tests.

## Mathematical Contract

Use toroidal angle `phi` as the independent variable.

State:

```text
y(phi) = [R(phi), Z(phi)]
x(R, phi, Z) = [R cos(phi), R sin(phi), Z]
```

Magnetic-field components:

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

Required assumption:

```text
B_phi keeps the expected sign and stays bounded away from zero
```

Per-seed objective:

```text
J_seed(c) =
    (1 / Phi_H) integral_phi0^(phi0 + Phi_H) [
        q_wall(x(phi; c))
      + radial_weight * q_radial(x(phi; c))
    ] dphi
```

Total objective:

```text
J_HRCA(c) = weighted_mean_seed_loss(J_seed)
```

### Wall-Risk Loss

Let `d_wall(x)` be smooth signed distance, positive inside the safe region.

Use a non-saturating soft barrier:

```text
z_wall(x) = (margin - d_wall(x)) / distance_scale
softplus_eps(z) = eps * log(1 + exp(z / eps))
q_wall(x) = softplus_eps(z_wall(x))^2
```

Do not use a bounded sigmoid as the primary v0 loss. It can saturate and remove useful gradient when a line is already bad.

### Optional Radial-Risk Loss

Only add this if a smooth fixed radial proxy exists:

```text
q_radial(x) = softplus((s_proxy(x) - s_max) / radial_scale)^2
```

Do not derive `s_proxy` from discontinuous field-line classification or a live surface solve in v0.

### Bphi Guard

Track `B_phi / |B|` along every trajectory.

- [ ] Add a report-only `bphi_min_observed` metric in v0.
- [ ] Keep `bphi_weight = 0` in v0; the wall/radial objective must not include `q_bphi` yet.
- [ ] Add a smooth guard loss only after the report-only metric shows failures matter.
- [ ] Hard-fail validation if `B_phi` approaches zero enough to make the phi-parametrized RHS ill-conditioned.

## Architecture Decision

### First Location

Implement under:

```text
examples/single_stage_optimization/banana_opt/hrca/
```

Candidate files:

- [ ] `examples/single_stage_optimization/banana_opt/hrca/__init__.py`
- [ ] `examples/single_stage_optimization/banana_opt/hrca/coordinates.py`
- [ ] `examples/single_stage_optimization/banana_opt/hrca/distance.py`
- [ ] `examples/single_stage_optimization/banana_opt/hrca/integrator.py`
- [ ] `examples/single_stage_optimization/banana_opt/hrca/objective.py`
- [ ] `examples/single_stage_optimization/banana_opt/hrca/validation.py`

Do not add `src/simsopt/field/hrca_*` in v0. Keep the feature experimental and local to banana optimization until it proves value.

### Public Objective Class

```python
class FieldlineConfinement(Optimizable):
    """Experimental fixed-horizon differentiable field-line risk objective."""
```

The optimizer-facing version must implement SIMSOPT-style `J()` and `dJ()` methods and register the field dependency through `Optimizable.__init__(..., depends_on=[field])` or an equivalent local wrapper. Diagnostic-only v0 code may use a plain helper class, but it must not be wired into objective arithmetic until this `Optimizable` contract exists.

Initial constructor contract:

- [ ] `field`: BiotSavart-like field with `set_points`, `B`, `dB_by_dX`, and `B_vjp`.
- [ ] `seeds`: immutable seed bundle with `(R0, Z0, phi0, weight, tag)`.
- [ ] `distance`: smooth signed-distance provider with `value_and_grad_xyz(points)`.
- [ ] `ntransits`: fixed finite horizon.
- [ ] `nsteps_per_transit`: fixed step count.
- [ ] `margin`: wall safety margin.
- [ ] `distance_scale`: wall-distance normalization.
- [ ] `softplus_eps`: smoothing width.
- [ ] `normalize`: normalize loss by horizon and seed weights.
- [ ] `store_trajectories`: optional diagnostic-only trajectory cache.

Return object:

- [ ] scalar `loss`
- [ ] SIMSOPT-compatible coil derivative
- [ ] per-seed loss
- [ ] per-seed max wall risk
- [ ] `bphi_min_observed`
- [ ] SDF domain violation count
- [ ] step size and horizon metadata
- [ ] optional trajectory summary for debugging

### Stateful Field Ownership

SIMSOPT magnetic fields are stateful: `set_points(...)` determines the points used by `B()`, `dB_by_dX()`, and `B_vjp(...)`.

- [ ] HRCA must either own a dedicated field object or save/restore field points around every evaluation.
- [ ] HRCA must call `field.set_points(...)` immediately before every `B()`, `dB_by_dX()`, and `B_vjp(...)` batch.
- [ ] HRCA diagnostics must not leave shared field objects with HRCA stage points installed.
- [ ] Add a regression test proving another objective evaluated before/after HRCA sees its own points, not HRCA's cached points.

## Derivative Contract

### v0 Primitive

- [ ] Use `B_vjp` for coil-gradient accumulation.
- [ ] Use `dB_by_dX` values for state-adjoint propagation.
- [ ] Do not use `B_and_dB_vjp` in v0.

Reason:

```text
The v0 forward RHS depends on B(x; c), not on grad B as a forward input.
The reverse pass needs dB/dX values to move adjoints through x -> B(x),
but the coil VJP is through B values at stage points.
```

### When `B_and_dB_vjp` Becomes Necessary

- [ ] Guiding-center or particle dynamics with grad-B drift terms.
- [ ] Mirror-force / curvature-force losses.
- [ ] Lyapunov / tangent-map losses that evolve variational equations as forward state.
- [ ] Greene-residue or island-width objectives.
- [ ] Any objective that explicitly consumes `grad B` or `dB_by_dX` as part of the forward scalar.

## Integrator Decision

Use fixed-step RK4 for v0.

- [ ] Implement the forward RK4 path first.
- [ ] Store stage states, stage fields, SDF values, and quadrature weights.
- [ ] Implement the reverse pass as a manual discrete adjoint of the same RK4 computation.
- [ ] Confirm the returned gradient is the gradient of the discrete scalar that `J()` returns.
- [ ] Add checkpointing / recomputation only after v0 is correct and runtime/memory justify it.

Do not start with a continuous adjoint. Do not start with adaptive ODE stepping. Do not start with JAX unless the field evaluation is pure JAX or wrapped with exact custom VJPs.

## Smooth Signed-Distance Plan

### Interface

```python
class SmoothSignedDistance:
    def value_and_grad_xyz(self, points):
        """Return (distance, grad_xyz) for shape (n, 3) points."""
```

### v0 Implementation Options

Preferred order:

1. [ ] Analytic test distance for unit tests and Taylor tests.
2. [ ] Smooth spline / periodic grid SDF around a fixed target surface.
3. [ ] SurfaceClassifier-derived distance only after proving gradient quality.

### Required Tests

- [ ] `value_and_grad_xyz` finite-difference test away from known nonsmooth locations.
- [ ] Periodicity test across the `phi` seam.
- [ ] Grid refinement test for SDF value and gradient.
- [ ] Domain coverage test for all training and holdout trajectories.
- [ ] Independent validation distance implementation for final checks.

### Explicit Risks

- [ ] Nearest-triangle switches are nonsmooth.
- [ ] Faceted wall normals are nonsmooth.
- [ ] Hard inside/outside classifiers are nonsmooth.
- [ ] Trilinear grids have gradient discontinuities at cell boundaries.
- [ ] Periodic `phi` wrapping can introduce seam discontinuities.
- [ ] SurfaceClassifier may be good for stopping/scoring but still insufficient for gradient-grade HRCA without extra tests.

## Seed Policy

### v0 Seed Bundle

- [ ] Use fixed physical seeds for all gradient tests.
- [ ] Do not differentiate through seed construction.
- [ ] Include edge-biased seeds because HRCA is meant to sense leakage risk.
- [ ] Include several radial bands, for example approximate `s = 0.4, 0.6, 0.75, 0.85, 0.92, 0.97`.
- [ ] Include multiple poloidal phases per band.
- [ ] Include multiple toroidal starting phases.
- [ ] Store seed metadata in a deterministic JSON-serializable schema.

### Holdout Seeds

- [ ] Define a separate holdout bundle before optimizer experiments.
- [ ] Do not tune HRCA weights using holdout results.
- [ ] Require holdout improvement before treating HRCA as useful.

## Implementation Phases

## Phase 0: Frozen-Point Derivative Audit

Objective: prove local BiotSavart derivative plumbing before tracing.

- [ ] Add `tests/geo/test_hrca_biot_savart_vjp.py`.
- [ ] Pick fixed Cartesian points and random cotangents.
- [ ] Verify `sum_i v_i dot B_i(c)` directional derivatives against finite differences.
- [ ] Verify `dB_by_dX` values against spatial finite differences.
- [ ] Record tolerances and expected finite-difference epsilon range.
- [ ] Fail loudly if the field does not expose `B_vjp`.

Exit criteria:

- [ ] `B_vjp` directional tests pass on representative banana coils.
- [ ] `dB_by_dX` spatial finite differences pass away from coil singularities.

## Phase 1: Forward Phi Integrator

Objective: implement deterministic finite-horizon trajectory evaluation without gradients.

- [ ] Add cylindrical basis helpers in `hrca/coordinates.py`.
- [ ] Add `rhs_phi(R, Z, phi, field)` using direct BiotSavart `B`.
- [ ] Implement fixed-step RK4 in `hrca/integrator.py`.
- [ ] Batch field evaluations per step or per stage where practical.
- [ ] Record `B_phi`, `|B|`, and stage points for diagnostics.
- [ ] Compare short-horizon trajectories against `compute_fieldlines` without stopping criteria.
- [ ] Add analytic-field tests where the expected field-line behavior is known.

Exit criteria:

- [ ] Forward integrator gives stable trajectories under step refinement.
- [ ] Forward integrator agrees with the existing tracer on simple cases where both formulations are comparable.
- [ ] `B_phi` singularity conditions are reported, not hidden.

## Phase 2: Smooth Distance And Loss

Objective: compute stable scalar HRCA loss without gradients.

- [ ] Add analytic `SmoothSignedDistance` test implementation.
- [ ] Add production candidate SDF provider behind a strict interface.
- [ ] Add softplus-squared wall-risk loss.
- [ ] Add optional radial-risk loss only if a smooth radial proxy is available.
- [ ] Normalize by horizon and seed weights.
- [ ] Report per-seed loss and max risk.
- [ ] Add deterministic snapshot tests for fixed seeds and fixed field.

Exit criteria:

- [ ] HRCA `J()` is deterministic.
- [ ] Loss changes smoothly under small coil perturbations in non-chaotic short-horizon cases.
- [ ] SDF value and gradient tests pass.

## Phase 3: Manual Discrete Adjoint

Objective: implement `dJ()` for the exact discrete RK4 objective.

- [ ] Implement reverse kernels for cylindrical projection `B -> (B_R, B_phi, B_Z)`.
- [ ] Implement reverse kernels for `f = [R B_R / B_phi, R B_Z / B_phi]`.
- [ ] Implement reverse RK4 sweep.
- [ ] Accumulate Cartesian magnetic-field cotangents at all RK stages.
- [ ] Reset the field to the exact VJP stage-point batch before calling `field.B_vjp(vB)`.
- [ ] Call `field.B_vjp(vB)` once per batch or per seed batch.
- [ ] Return a derivative object compatible with SIMSOPT objective aggregation.
- [ ] Add optional debug mode returning intermediate adjoint norms.

Exit criteria:

- [ ] State-gradient tests pass against finite differences in `(R0, Z0)`.
- [ ] Coil-gradient central finite differences match `dJ()`.
- [ ] Random-direction Taylor tests show second-order remainder over a usable epsilon window.

## Phase 4: Gradient And Numerical Validation Matrix

Objective: prove HRCA gradients are usable before any optimizer coupling.

- [ ] Run one-seed / one-transit Taylor test.
- [ ] Run multi-seed / one-transit Taylor test.
- [ ] Run multi-seed / 5-transit Taylor test.
- [ ] Run horizon sweep at 5, 10, and 20 transits.
- [ ] Run step refinement with `nsteps_per_transit` doubled.
- [ ] Run SDF grid refinement.
- [ ] Run holdout seed evaluation.
- [ ] Compare HRCA ranking across archived candidates against strict topology scorer outputs.

Pass thresholds to define before execution:

- [ ] Directional derivative relative error target for non-chaotic short tests.
- [ ] Minimum acceptable gradient-direction cosine under step refinement.
- [ ] Minimum acceptable rank correlation against strict topology metrics.
- [ ] Maximum allowed runtime fraction relative to the existing objective stack.

## Phase 5: Diagnostic-Only Repo Integration

Objective: make HRCA observable without changing optimizer behavior.

- [ ] Add CLI flag to compute HRCA diagnostics after accepted iterations only.
- [ ] Write HRCA metrics into existing per-iteration diagnostic artifacts.
- [ ] Keep objective weight fixed at zero.
- [ ] Add result schema fields:
  - `hrca_loss`
  - `hrca_grad_norm`
  - `hrca_bphi_min_observed`
  - `hrca_train_seed_loss_mean`
  - `hrca_holdout_seed_loss_mean`
  - `hrca_step_size`
  - `hrca_ntransits`
- [ ] Add archive/reporting support without changing frontier dominance semantics.

Exit criteria:

- [ ] HRCA diagnostics run without affecting optimizer state.
- [ ] Existing single-stage and Stage 2 outputs remain backward compatible.
- [ ] HRCA diagnostic metrics correlate with at least one strict topology failure mode.

## Phase 6: Low-Weight Auxiliary Trial

Objective: test HRCA as a small objective term only after all gates pass.

- [ ] Add explicit opt-in flag, for example `--experimental-hrca-weight`.
- [ ] Default weight must be zero.
- [ ] Reject nonzero HRCA weight unless validation artifact exists for the chosen seed/SDF/horizon config.
- [ ] Add train/holdout seed reporting for every HRCA-enabled run.
- [ ] Compare against matched baseline with same optimizer budget.
- [ ] Require no degradation in:
  - BoozerResidual
  - NQS / QS metric
  - Iotas
  - Greene residue when available
  - engineering constraints
  - strict Poincare / topology scorer

Exit criteria:

- [ ] HRCA improves holdout finite-time risk at equal or better engineering quality.
- [ ] HRCA does not degrade strict topology validation.
- [ ] HRCA catches at least one leakage mode not already caught by BoozerResidual / NQS / Iotas / residue.

## Kill Criteria

Stop HRCA as an optimization term and keep it diagnostic-only if any condition persists:

- [ ] Taylor tests fail in smooth short-horizon cases.
- [ ] Gradient direction changes materially under step refinement.
- [ ] HRCA improves while strict Poincare / topology scorer worsens.
- [ ] HRCA improves train seeds but not holdout seeds.
- [ ] HRCA gains disappear under SDF refinement or independent distance validation.
- [ ] Optimizer exploits `B_phi`, seed placement, current scaling, or SDF artifacts.
- [ ] Useful HRCA requires hard stops or event-time derivatives.
- [ ] Useful HRCA requires `InterpolatedField` in the differentiable path.
- [ ] Useful HRCA requires modifying the C++ tracer.
- [ ] Runtime dominates the main objective stack without independent validation gain.
- [ ] Greene residue / island metrics explain all observed improvement and HRCA adds no independent signal.

## Optimizer Gaming Checklist

- [ ] Seed overfit: require holdout seeds and phase-shifted seed bundles.
- [ ] Iota gaming: keep Iotas and residue diagnostics active.
- [ ] Current-magnitude scaling: use phi horizon and keep current constraints active.
- [ ] Wall-distance gaming: validate with strict topology scorer and radial spread.
- [ ] Finite-time stickiness: require horizon sweep and long validation traces.
- [ ] `B_phi` singularity: report and gate on `B_phi / |B|`.
- [ ] Step-size gaming: require step convergence.
- [ ] SDF-grid gaming: require SDF refinement and independent distance check.
- [ ] Softplus saturation: use non-saturating softplus-squared, not bounded sigmoid.

## Minimum Value Experiment

Run matched A/B/C experiments on the same seed designs and optimizer budget.

### Arms

- [ ] A: baseline smooth objective stack.
- [ ] B: baseline plus Greene residue if available.
- [ ] C: baseline plus low-weight HRCA.
- [ ] D: baseline plus Greene residue plus low-weight HRCA, only after B and C are interpretable.

### Success Metrics

- [ ] HRCA holdout seed loss improves by a predeclared threshold.
- [ ] Strict topology scorer survival/lifetime metrics improve or do not degrade.
- [ ] Poincare audit does not reveal worse islands or stochastic layers.
- [ ] BoozerResidual, NQS, Iotas, and engineering metrics do not degrade beyond predeclared tolerances.
- [ ] Improvement survives horizon and step refinement.
- [ ] Improvement cannot be explained solely by iota movement or residue improvement.

## Literature Anchors

Core HRCA / ODE-adjoint context:

- [ ] Chen et al., "Neural Ordinary Differential Equations" (`arXiv:1806.07366`): general ODE-adjoint framing.
- [ ] Gholami et al., "ANODE" (`arXiv:1902.10298`): memory / accuracy issues for neural ODE gradients.
- [ ] Diffrax adjoint documentation: practical discrete/checkpointed vs continuous/backsolve adjoint distinction.
- [ ] Wang et al., least-squares shadowing (`arXiv:1204.0159`): chaotic sensitivity caution.
- [ ] Ni / Talnikar NILSAS work (`arXiv:1801.08674`): long-horizon chaotic adjoint caution.

Core magnetic-topology context:

- [ ] Geraldini, Landreman, Paul (`arXiv:2102.04497`): island-width and residue sensitivity reference.
- [ ] Greene residue literature: periodic-orbit residue as local topology diagnostic.
- [ ] Giuliani et al. Boozer-surface optimization: smooth nested-surface / QS objective context.
- [ ] Hudson / Dewar QFM and ghost-surface papers: almost-invariant-surface alternatives.
- [ ] Smiet et al. turnstile-area work: nonlocal chaotic transport metric, longer-term path.

Particle-confinement context:

- [ ] Gamma-c / effective-ripple / direct fast-ion-loss literature: particle confinement is not equivalent to field-line topology.

## Final Decision Gate

HRCA can move from diagnostic to low-weight auxiliary objective only after all are true:

- [ ] v0 implementation passes derivative tests.
- [ ] v0 implementation passes step, horizon, and SDF convergence checks.
- [ ] v0 diagnostic correlates with at least one strict topology failure mode.
- [ ] Holdout seeds improve in a matched experiment.
- [ ] No primary physics objective or engineering constraint degrades beyond predeclared tolerance.
- [ ] Greene residue remains the primary topology-gradient implementation priority.
