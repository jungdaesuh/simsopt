# Edge-Delivered Iota Lane Implementation Plan

## Implementation status (2026-06-27)

Phases 1-3 and the Phase-4 reporting path were implemented in prior work. This
revision completes the Phase-4 **optimizer-facing `soft` mode** as a genuinely
differentiable steering term:

- `banana_opt/edge_iota_proxy.py` — a cheap, analytic-gradient surrogate for the
  field-line-trace oracle, using the exact winding identity
  `iota = 2*pi / oint (B_phi / (R*B_pol)) dl_pol` on fixed EQDSK flux contours;
  gradient via `BiotSavart.B_vjp` (FD-verified to ~1e-10 on signal DOFs; tokamak
  anchor matches `1/q` to ~2e-4; warm cost ~25 ms).
- `Stage2EdgeIotaSteeringObjective` + `_add_stage2_edge_iota_objective` in
  `banana_opt/stage2_objectives.py` — a quadratic hinge toward
  `--stage2-edge-iota-target-min`, wired into the penalty and ALM objective paths
  via `--stage2-edge-iota-weight` (default `0` ⇒ objective byte-identical when off).

**Caveat that frames this as a STEERING signal, not a guarantee:** the trace
oracle is chaotic-unreliable at the edge for the current banana seeds (field lines
don't survive at nonzero banana current; the survival set flips with trace
resolution). The differentiable proxy assumes a nested flux surface *exists*, so it
is a transform-*steering* signal — it tells the optimizer which way grows external
edge transform, but it does **not** detect chaos or guarantee confinement. The p10
promotion gate (trace-oracle profile) remains separate and authoritative; the soft
term steers the edge-band *mean* `delta_abs` as a smooth surrogate and never relaxes
a hardware gate.

## Purpose

Define a production path for adding an `edge_delivered_iota_lane` to
`simsopt-surrogate`: the optimizer should stop rewarding coil sets that make a good
core-only vacuum object and should instead measure, report, and eventually optimize
useful external transform delivered to the HBT-EP plasma edge.

## Goals

- Add a post-run delivered-edge-iota oracle for HBT-EP geometry.
- Report an edge radial profile, not only a scalar iota value.
- Make single-stage runs target outer surfaces with edge-weighted iota objectives.
- Promote Stage 2 from scalar/single-surface iota reporting to edge-profile reporting
  and, after validation, an optimizer-facing constraint or objective.
- Preserve hard buildability gates: hardware keepout, current, coil length, coil
  spacing, curvature, and self-intersection constraints.
- Keep the current core-transform result usable as a diagnostic, but not as the
  primary success metric for disruption-avoidance claims.

## Non-Goals

- Do not implement free-boundary plasma response in this lane. The first oracle is a
  vacuum-superposition estimate on a fixed EQDSK plasma geometry.
- Do not rewrite SIMSOPT, Boozer tracing, or the Stage 2 solver architecture.
- Do not loosen hardware constraints to obtain a visually good magnetic result.
- Do not treat average/core iota as evidence of edge-delivered transform.
- Do not silently infer coil/plasma helicity sign. The sign convention must be an
  explicit input or recorded result.

## Current Context

### Facts

- Single-stage already has a per-surface iota profile objective builder:
  `/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/banana_opt/single_stage_objectives.py:154`.
- Single-stage CLI wiring for iota-profile objectives already exists around:
  `/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:3512`.
- Single-stage topology-gate controls already exist around:
  `/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:2481`.
- Stage 2 iota runtime is currently scalar/single-surface and marked legacy at:
  `/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/banana_opt/stage2_objectives.py:374`.
- Stage 2 scalar iota already has `report` and `soft` modes, with `report` as the
  default and `soft` rejected for ALM hard-constraint paths:
  `/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/STAGE_2/banana_coil_solver.py:839`.
- Stage 2 artifact contracts currently expose scalar `STAGE2_IOTA_*` fields at:
  `/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/banana_opt/artifact_contracts.py:86`.
- Stage 2 solver output currently records scalar iota fields around:
  `/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/STAGE_2/banana_coil_solver.py:2458`.
- The current validated HBT-EP analysis says the existing champion delivers about
  `iota_ext ~= 0.11` only near `r ~= 0.05 m`, while the real HBT-EP edge
  (`a ~= 0.142 m`) receives no coherent useful external transform:
  `/Users/suhjungdae/code/columbia/autoresearch/analysis/reports/iota_ext_on_real_hbtep_plasma_validation_2026-06-25.md`.
- That same validation report says the stable inner-core hybrid traces reduce the
  tokamak-transform magnitude for shot 105995's reconstructed sign convention. The
  regression fixture should therefore call the champion "core-localized vacuum-transform
  positive", not "demonstrated added transform for that shot sign".

### Classification

This is a medium cross-module change, not a rewrite.

- Report-only oracle: Tier 2 style change.
- Optimizer-facing Stage 2 edge constraint plus artifact schema propagation: Tier 2
  bordering Tier 3 inside the local solver/reporting stack.
- Free-boundary plasma response or HPC production batching would be a separate Tier 3
  effort.

## Rationale

The current pipeline can produce a magnetically interesting core object, but the
disruption-avoidance claim needs delivered transform at the plasma edge. Reweighting
the existing vacuum iota objective is not enough, because the optimizer can still win
by making transform where surfaces naturally survive: the core.

The first step should be a post-run oracle. It prevents false promotion by answering
the direct question: when this finite coil set is evaluated on the real HBT-EP plasma
geometry, how much transform reaches `r/a = 0.75-1.0`, and do edge-started field lines
survive?

### Design-It-Twice

Option A: extend the current Stage 2 scalar iota runtime into a list of radial samples.

- Benefit: smallest apparent edit.
- Problem: the runtime is already marked legacy and is shaped around a scalar Boozer
  value. Extending it risks mixing vacuum-object iota with real-plasma delivered edge
  transform.

Option B: add a separate `edge_delivered_iota` oracle and contract.

- Benefit: isolates EQDSK geometry, TF double-counting, co-helicity sign, edge survival,
  and chaos-width assumptions from the existing scalar iota machinery.
- Cost: more code and new artifact fields.

Use Option B. Keep the old scalar `STAGE2_IOTA_*` fields for compatibility, and add new
edge-specific fields with explicit names.

## Assumptions

- HBT-EP EQDSK and LCFS artifacts are passed to the evaluator by CLI/config, not found
  by hidden path magic.
- The initial edge band is `r/a = 0.75-1.0`.
- The primary target is co-helicity delivered external transform at the edge:
  `edge_delta_abs_iota ~= 0.10-0.15`, where the promotion scalar is based on
  `abs(iota_hybrid) - abs(iota_tokamak)` and the raw signed delta is recorded
  separately.
- Co-helicity is required for promotion. Counter-helicity is a diagnostic result, not a
  successful disruption-avoidance configuration.
- The first implementation is report-only. Optimizer-facing use comes only after the
  oracle reproduces the current champion as a core-localized vacuum-transform /
  edge-fail case.
- If an optimizer-facing edge term has no trustworthy gradient, it must be routed as an
  outer-loop ranking metric, derivative-free penalty, or explicitly documented
  non-gradient constraint. It must not masquerade as a differentiable objective.

## Implementation Plan

### Phase 1: Define the edge-delivered-iota contract

- [ ] Add a focused module such as
  `examples/single_stage_optimization/banana_opt/edge_delivered_iota.py`.
- [ ] Define typed records for:
  - [ ] `EdgeIotaSample`: radius label, `r/a`, seed point, baseline tokamak iota,
    hybrid iota, signed delta, magnitude delta, convergence, survival status,
    width/chaos indicator.
  - [ ] `EdgeIotaProfile`: list of samples plus edge-band summary values.
  - [ ] `EdgeIotaConfig`: EQDSK path, LCFS path, edge band, sample count, helicity sign,
    trace turns, tolerances, and coil partition metadata.
- [ ] Define scalar summaries:
  - [ ] `edge_delta_abs_iota_min`.
  - [ ] `edge_delta_abs_iota_p10`.
  - [ ] `edge_delta_abs_iota_mean`.
  - [ ] `edge_delta_signed_iota_min`.
  - [ ] `edge_delta_signed_iota_mean`.
  - [ ] `edge_surface_survival_fraction`.
  - [ ] `edge_width_max`.
  - [ ] `edge_iota_status`.
  - [ ] `edge_helicity_status`.
- [ ] Make active modes fail closed when required inputs are missing.
- [ ] Add unit tests for scalar summary math and status classification.

### Phase 2: Add the post-run HBT-EP evaluator

- [ ] Add a minimal, reviewed EQDSK reader and axisymmetric tokamak-field builder, or
  promote the existing scratch implementation into the new module with tests.
- [ ] Validate tokamak-only field-line iota against the EQDSK q-profile before using a
  hybrid result.
- [ ] Add explicit coil partitioning so the evaluator can add banana coils without
  double-counting TF already present in the EQDSK field.
- [ ] Trace baseline tokamak and hybrid fields on edge-started samples in the configured
  edge band.
- [ ] Record raw `iota_tokamak`, raw `iota_hybrid`, signed
  `delta_iota_signed = iota_hybrid - iota_tokamak`, and promotion-facing
  `delta_abs_iota = abs(iota_hybrid) - abs(iota_tokamak)`.
- [ ] Classify helicity explicitly from the signed delta and configured operating-point
  convention; do not let absolute-value summaries hide counter-helicity.
- [ ] Compute edge surface survival and width/chaos diagnostics.
- [ ] Add limiting-case checks: zero banana current gives zero signed and magnitude
  delta, and tokamak-only traces reproduce the EQDSK q-profile inside the LCFS.
- [ ] Persist a profile JSON artifact and summary fields.
- [ ] Add a regression fixture using the current `iota011_a150_R0935_2026-06-25`
  champion; it must be classified as core-localized vacuum-transform-positive but
  edge-delivered-iota failing.

### Phase 3: Add the single-stage edge lane preset

- [ ] Add an `edge_delivered_iota_lane` preset or config path in
  `SINGLE_STAGE/single_stage_banana_example.py`.
- [ ] Reuse the existing per-surface iota profile objective, but weight outer surfaces
  heavily.
- [ ] Use edge-oriented surface fractions, for example `s = 0.70, 0.85, 0.95, 1.00`.
- [ ] Keep the existing iota-profile objective default-off outside this lane.
- [ ] Make topology gates edge-started for this lane.
- [ ] Promote winding-surface radius, center, and low-order shape controls from cleanup
  knobs to primary search degrees of freedom for this lane.
- [ ] Add CLI/preset tests that assert the final search weights are edge-weighted.

### Phase 4: Promote Stage 2 reporting, then Stage 2 constraints

- [ ] Extend artifact contracts with edge-specific fields such as:
  - [ ] `EDGE_IOTA_STATUS`.
  - [ ] `EDGE_IOTA_PROFILE_JSON`.
  - [ ] `EDGE_DELTA_ABS_IOTA_MIN`.
  - [ ] `EDGE_DELTA_ABS_IOTA_P10`.
  - [ ] `EDGE_DELTA_ABS_IOTA_MEAN`.
  - [ ] `EDGE_DELTA_SIGNED_IOTA_MIN`.
  - [ ] `EDGE_DELTA_SIGNED_IOTA_MEAN`.
  - [ ] `EDGE_SURFACE_SURVIVAL_FRACTION`.
  - [ ] `EDGE_WIDTH_MAX`.
  - [ ] `EDGE_HELICITY_STATUS`.
- [ ] Do not overload existing scalar `STAGE2_IOTA_*` fields.
- [ ] Add Stage 2 CLI/config controls:
  - [ ] `--stage2-edge-iota-mode {off,report,soft}`.
  - [ ] `--stage2-edge-iota-eqdsk`.
  - [ ] `--stage2-edge-iota-lcfs`.
  - [ ] `--stage2-edge-iota-radial-band`.
  - [ ] `--stage2-edge-iota-target-min`.
  - [ ] `--stage2-edge-iota-helicity`.
- [x] Implement `off` and `report` modes first; keep `report` behavior-neutral for
  optimization.
- [x] Add `soft` mode only after report-mode artifacts reproduce the current champion
  failure and at least one synthetic/pass fixture. (Both fixtures exist in
  `tests/geo/test_edge_delivered_iota.py`: champion-fail + synthetic-pass.)
- [x] `soft` mode is implemented as a genuinely DIFFERENTIABLE term, not a fake
  gradient: `banana_opt/edge_iota_proxy.py` is a cheap analytic-gradient surrogate
  (exact field-line-winding identity on fixed EQDSK flux contours, gradient via
  `BiotSavart.B_vjp`), wired as the `Stage2EdgeIotaSteeringObjective` quadratic-hinge
  steering term in both the penalty (`make_stage2_fun`) and ALM
  (`evaluate_stage2_alm_problem`) paths via `--stage2-edge-iota-weight` (default 0 =
  byte-identical when off). See the status note at the top of this plan for the
  oracle-reliability caveat that frames it as a *steering* signal, not a guarantee.

### Phase 5: Update wrappers and reports

- [ ] Update autoresearch launch/report wrappers to pass EQDSK, LCFS, radial band, and
  helicity arguments.
- [ ] Add edge-iota summary columns to run inventories:
  - [ ] `edge_delta_abs_iota_min`.
  - [ ] `edge_delta_abs_iota_p10`.
  - [ ] `edge_delta_signed_iota_min`.
  - [ ] `edge_surface_survival_fraction`.
  - [ ] `edge_width_max`.
  - [ ] `edge_helicity_status`.
  - [ ] `edge_iota_status`.
- [ ] Store raw run payloads under campaign archives.
- [ ] Store reusable derived reports under `analysis/reports/`.
- [ ] Store reusable figures under `analysis/figures/`.

### Phase 6: Define promotion gates

- [ ] Require co-helicity with the selected HBT-EP operating point.
- [ ] Require `edge_delta_abs_iota_p10 >= 0.10` across `r/a = 0.75-1.0`, unless a
  different scalar is explicitly chosen.
- [ ] Require no unacceptable edge island/chaos width.
- [ ] Require edge surface survival above the configured threshold.
- [ ] Require existing hardware gates: keepout, current, curvature, length, spacing, and
  self-intersection.
- [ ] Require direct CAD/contact oracle for hardware promotion; SDF/proxy clearance is
  steering evidence, not final proof.

## Validation Plan

- [ ] Run existing iota-profile tests:

  ```bash
  PYTHONNOUSERSITE=1 /Users/suhjungdae/code/columbia/simsopt-surrogate/.conda-env/bin/python -m pytest \
    /Users/suhjungdae/code/columbia/simsopt-surrogate/tests/geo/test_single_stage_profile_objectives.py -q
  ```

- [ ] Add and run edge oracle tests:

  ```bash
  PYTHONNOUSERSITE=1 /Users/suhjungdae/code/columbia/simsopt-surrogate/.conda-env/bin/python -m pytest \
    /Users/suhjungdae/code/columbia/simsopt-surrogate/tests/geo/test_edge_delivered_iota.py -q
  ```

- [ ] Run focused single-stage wiring tests:

  ```bash
  PYTHONNOUSERSITE=1 /Users/suhjungdae/code/columbia/simsopt-surrogate/.conda-env/bin/python -m pytest \
    /Users/suhjungdae/code/columbia/simsopt-surrogate/tests/geo/test_single_stage_example.py -k edge_iota -q
  ```

- [ ] Run focused artifact-contract tests:

  ```bash
  PYTHONNOUSERSITE=1 /Users/suhjungdae/code/columbia/simsopt-surrogate/.conda-env/bin/python -m pytest \
    /Users/suhjungdae/code/columbia/simsopt-surrogate/tests/geo/test_ishw_deliverables.py -k edge_iota -q
  ```

- [ ] Run focused Stage 2 objective/module tests:

  ```bash
  PYTHONNOUSERSITE=1 /Users/suhjungdae/code/columbia/simsopt-surrogate/.conda-env/bin/python -m pytest \
    /Users/suhjungdae/code/columbia/simsopt-surrogate/tests/geo/test_banana_objective_modules.py -k edge_iota -q
  ```

- [ ] Compile touched Python files:

  ```bash
  PYTHONNOUSERSITE=1 /Users/suhjungdae/code/columbia/simsopt-surrogate/.conda-env/bin/python -m compileall \
    /Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization
  ```

- [ ] Run whitespace/path sanity:

  ```bash
  cd /Users/suhjungdae/code/columbia/simsopt-surrogate && git diff --check
  ```

- [ ] Regression requirement: the current
  `iota011_a150_R0935_2026-06-25` champion must not pass the new edge gate.

## Risks and Mitigations

- Risk: Field-line tracing is too slow for optimizer hot loops.
  - Mitigation: make report mode the first deliverable; cache profiles; only promote to
    optimizer-facing use after runtime measurement.

- Risk: TF double-counting or helicity-sign mistakes produce misleading iota.
  - Mitigation: require explicit coil partition metadata, record helicity sign, and keep
    a tokamak-only q-profile validation gate.

- Risk: The optimizer exploits a scalar edge metric while degrading surface quality.
  - Mitigation: make edge survival and width/chaos hard gates, not only report fields.

- Risk: The vacuum-superposition oracle overstates real plasma penetration.
  - Mitigation: label the metric as fixed-boundary/vacuum-superposition and keep
    free-boundary plasma response as a later validation layer.

- Risk: Schema changes break existing downstream reports.
  - Mitigation: add new `EDGE_*` fields instead of changing `STAGE2_IOTA_*` semantics.

- Risk: The current core-localized vacuum-transform champion accidentally passes.
  - Mitigation: add it as an explicit fail regression fixture.

## Completion Criteria

- [ ] A report-only edge-delivered-iota oracle exists and writes a profile JSON plus
  scalar summaries.
- [ ] The oracle validates tokamak-only traces against the EQDSK q-profile.
- [ ] The current champion is classified as core-localized vacuum-transform-positive /
  edge-fail by the new oracle.
- [ ] Single-stage has an `edge_delivered_iota_lane` preset with edge-weighted surface
  targets.
- [ ] Stage 2 artifacts include `EDGE_*` profile and summary fields without changing the
  meaning of scalar `STAGE2_IOTA_*`.
- [x] Stage 2 `report` mode is tested and behavior-neutral.
- [x] Any optimizer-facing `soft` mode has an explicit gradient or non-gradient routing
  story. (Explicit ANALYTIC gradient via `edge_iota_proxy.py` + `BiotSavart.B_vjp`;
  FD-verified. Default-off ⇒ byte-identical when not requested.)
- [x] Existing hardware gates remain hard. (The steering term is an additive smooth
  nudge only; it never relaxes keepout/current/curvature/length/spacing/self-intersect
  gates, and the p10 promotion gate stays separate and authoritative.)
- [x] Focused tests and `git diff --check` pass. (`tests/geo/test_edge_iota_proxy.py`
  + `test_edge_delivered_iota.py` + `test_ishw_deliverables.py` green; diff --check clean.)

## Open Questions

- Should the promotion scalar be `edge_delta_abs_iota_min`, `edge_delta_abs_iota_p10`,
  or another robust edge-band statistic?
- Is the target strictly `edge_delta_abs_iota = 0.10-0.15`, or should absolute hybrid
  edge iota also be constrained?
- What edge surface survival fraction and width/chaos threshold are acceptable for HBT-EP
  proposal claims?
- Should co-helicity be achieved by coil-current sign, winding convention, or HBT-EP
  plasma-current direction in the planned experiment?
- Does optimizer-facing edge tracing need a derivative-free outer loop, or can a cheaper
  differentiable proxy be validated against the oracle?
- What runtime budget is acceptable before moving this to HPC batching?
