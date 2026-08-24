# JAX Native-Test Mirror Wave Implementation Plan

**Status:** Draft
**Last updated:** 2026-08-23

## Purpose

Close the mirror-test gaps found by the 2026-08-23 coverage audit under the project rule
"every native test capability has JAX mirror coverage unless it is VMEC-, SPEC-, or
third-party-dependent," and record the deliberate non-port rulings so exclusion is a
documented decision, not silence. This plan executes as one parallel subagent wave
(six MECE units, disjoint file ownership, CPU-side; no unit touches the GPU box while the
concurrent nested-LS B37 campaign runs).

This plan does **not** redesign the coverage contract: unit 6 implements the first
executable slice of the pre-existing
`docs/jax_native_unit_test_coverage_implementation_plan.md` (Draft, 2026-07-29) and adopts
its artifact names and disposition vocabulary verbatim.

## Goals

- Native self-field-force behavior (23 funcs, `tests/field/test_selffieldforces.py`) has a
  JAX mirror suite exercising `src/simsopt_jax_adapters/field/force.py`.
- Strain, curve-subclass, and CurvePerturbed-reformulation behavior each gain direct JAX
  unit/parity tests (today: example-level parity only or incidental reach).
- The rcls-with-ports adapter boundary is verified by test (identical constrained-segment
  index set on both lanes), not assumed.
- The coverage-manifest artifacts from the 2026-07-29 plan exist and are drift-gated, with
  rows populated for the domains this wave touches and dated dispositions recorded for
  CoilSet, PortSet, and `fourier_interpolation`.

## Non-Goals

- No JAX port of `CoilSet` (647 LOC), `PortSet`/`ports.py` (1032 LOC), or
  `fourier_interpolation` (54 LOC) — see Decisions.
- No GPU-executed validation in this wave (box fenced by the concurrent B37 campaign);
  new tests use the existing `parity_lane` fixture so GPU lanes activate when CUDA is
  present, but the wave's own validation runs CPU-only.
- No full population of the 71-file native manifest — that remains the 2026-07-29 plan's
  own checklist; this wave builds the artifacts and seeds the audited domains.
- No changes to production `src/` numerics.

## Current Context (confirmed 2026-08-23)

- Audit baseline: upstream native suite = 71 files / 721 test funcs, all preserved in the
  fork; fork total 6,384 funcs; GPU-capable tests ≈ 140 funcs (~2%), skip-silent without
  CUDA; no integration mirror executes the `jax-gpu` lane.
- Force: native source `src/simsopt/field/{selffield,force,coil}.py`; native tests
  `tests/field/test_selffieldforces.py` (23 funcs, 1,603 LOC; classes
  `SpecialFunctionsTests` — k²/δ²/symmetry/limits/regularization circ+rect — and
  `CoilForcesTest` — analytic circular coil, convergence, HSX coil, net force/torque,
  force objectives, Taylor tests, downsample/quadpoint guards, regularized-coil
  subclasses). JAX side: `src/simsopt_jax_adapters/field/force.py` (114 defs) +
  `src/simsopt_jax_adapters/objectives/force_stage_two.py`, tested only by
  `tests/jax/objectives/test_force_stage_two.py` (2 funcs).
- Strain: native `src/simsopt/geo/strain_optimization.py` (192 LOC) with
  `tests/geo/test_strainopt.py` (3 funcs); JAX `src/simsopt_jax/examples/strain_optimization.py`
  covered only by integration mirrors
  (`tests/integration/test_jax_mirror_strain_optimization_parity.py`, strict-transfer).
- Curve subclasses: `src/simsopt_jax/core/{curve_helical,curve_planar_fourier,curve_rz_fourier}.py`
  exist; reached only incidentally (e.g. via `tests/field/test_biotsavart_jax_parity.py`).
- CurvePerturbed: native `src/simsopt/geo/curveperturbed.py` (209 LOC,
  `tests/geo/test_curveperturbed.py` 5 funcs); JAX reformulates as data —
  `src/simsopt_jax/examples/stochastic_samples.py`
  (`GaussianPerturbationSampler`, `StochasticPerturbationBundle`) +
  `src/simsopt_jax/objectives/stochastic_stage_two.py` (`StochasticCoilPerturbations`);
  `tests/jax/examples/test_stochastic_samples.py` has 3 funcs (fingerprints).
- Ports: `examples/jax/parity/cases/native_wireframe_rcls_with_ports.py:42-143` builds
  ports natively and records `constrained_segments`; the JAX adapter consumes
  `wframe.unconstrained_segments()` (`src/simsopt_jax_adapters/solve/wireframe.py:314`).
  Port geometry never executes through JAX.
- Coverage SSOT: all five artifacts required by the 2026-07-29 plan are missing
  (`tests/fixtures/jax_native_unit_coverage_manifest.json`,
  `scripts/jax_native_unit_coverage.py`,
  `tests/jax/test_native_unit_coverage_manifest.py`,
  `docs/jax_native_unit_test_coverage.md`, `tests/jax/native_unit_parity/`).
- Environment: one test file per process;
  `PYTHONPATH=src:build/cp311-cp311-linux_x86_64 JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
  .venv/bin/python -m pytest <file> -q`; lint via `bash scripts/lint.sh check|format`
  (never unpinned ruff). Known conftest defect: `tests/conftest.py:584` lists nonexistent
  `field/test_force_item09_closeout.py` (fix in unit 1, which owns that namespace).

## Decisions (2026-08-23, proposed rulings — frozen into the manifest by unit 6)

- **CoilSet → `native_only`.** Sole in-repo consumer is `normal_field.py` (SPEC family,
  excluded by the project rule); no example uses it; JAX stage-two objectives cover coil
  collections in their own idiom. Porting 647 LOC with zero JAX-side consumers is YAGNI.
  Revisit only if a JAX example adopts the CoilSet API.
- **PortSet → `hybrid_boundary`** (the 2026-07-29 plan's own category: host computation
  intentionally retained, boundary tested). One-shot host-side collision geometry has no
  gradient or GPU value; the correct verification is the unit-5 index-set parity test,
  not a port.
- **`fourier_interpolation` → `native_only`.** 54 LOC, zero in-repo consumers.
- Mirror tests compare against the **native implementation as oracle** (same-process
  native call or frozen fixture), matching the repo's existing `*_jax_*` parity-test
  idiom; bitwise is not required — use the established tolerance-bucket helpers where the
  algebra reassociates.

## Assumptions

- The `parity_lane` / strict-backend fixtures in `tests/conftest.py` (cpu+gpu params,
  CUDA-skip) are the sanctioned way to make new tests GPU-capable without GPU-only CI.
- `src/simsopt_jax_adapters/field/force.py` implements the regularized self-field force
  path the native tests exercise (verified to exist and be substantial; per-function
  coverage is discovered by unit 1 and any *missing* JAX behavior is recorded as
  `jax_partial`/`jax_missing` manifest rows, not papered over with weak tests).
- The concurrent session's B37 campaign owns the GPU box for the duration; all six units
  are CPU-valid.

## Implementation Plan

Orchestration: six units, one parallel wave, disjoint file ownership (unit-owned files
listed per unit; no unit edits another's files). Suggested tiering per
orchestrate-subagents: unit 1 opus/high (critical path), units 2–5 sonnet/high, unit 6
opus/high (contract artifact + gate). Each unit self-validates (its own pytest files,
one per process, plus pinned lint) before reporting; the wave closes with a crucible
review of the combined diff.

New mirror-test files land in `tests/jax/native_unit_parity/` — the 2026-07-29 plan's
required home for "new behavioral parity tests when no suitable existing test owns the
capability" (its Coverage Contract). Unit 6 creates that directory; units 1–3 add only
their own files to it. Unit 4 extends an existing file in place; unit 5's boundary test
follows the existing integration-test convention in `tests/integration/`.

1. **Unit 1 — Force mirror suite** (owns `tests/jax/native_unit_parity/test_force_parity.py`
   [new], `tests/jax/objectives/test_force_stage_two.py`, `tests/conftest.py` [one-line fix])
   - [ ] Map the 23 native funcs to `force.py` symbols; record any native behavior with
         no JAX implementation as manifest-facts for unit 6 (do not fake coverage).
   - [ ] Mirror `SpecialFunctionsTests` (k², δ², symmetry, limits, regularization
         circular + rectangular) against the JAX implementations, native values as oracle.
   - [ ] Mirror the analytic circular-coil force, HSX-coil value, net force/torque
         consistency, force/torque objectives (values + gradients via Taylor tests), and
         the guard behaviors (downsample divisibility, mixed quadpoints, regularized-coil
         requirement) where the JAX API exposes them.
   - [ ] Use `parity_lane` so the suite is GPU-capable; no GPU execution in this wave.
   - [ ] Remove the stale `field/test_force_item09_closeout.py` entry at
         `tests/conftest.py:584`.
2. **Unit 2 — Strain unit tests** (owns `tests/jax/native_unit_parity/test_strain_parity.py` [new])
   - [ ] Mirror the 3 native funcs in `tests/geo/test_strainopt.py` (torsional + binormal
         curvature strain values and gradient/Taylor checks) against
         `src/simsopt_jax/examples/strain_optimization.py`, native as oracle.
3. **Unit 3 — Curve-subclass coverage** (owns
   `tests/jax/native_unit_parity/test_curve_subclasses_parity.py` [new])
   - [ ] Parametrize position/tangent/derivative and objective evaluations over
         `curve_helical`, `curve_planar_fourier`, `curve_rz_fourier` vs their native
         counterparts (values + first derivatives; DOF round-trips).
4. **Unit 4 — CurvePerturbed reformulation parity** (owns
   `tests/jax/examples/test_stochastic_samples.py`)
   - [ ] Extend beyond fingerprints: assert the materialized perturbed gammas equal native
         `CurvePerturbed` evaluations for matched sampler config/seed, for both
         perturbation branches the stochastic example uses (systematic + statistical).
   - [ ] Mirror the five native behaviors (`tests/geo/test_curveperturbed.py`: perturbed
         gammadash, periodicity, torsion and distance objectives through perturbed curves,
         serialization — its distance test also exercises `resample()`) to the extent the
         JAX bundle exposes equivalents; behaviors with no JAX counterpart (e.g. in-place
         `resample`) become `jax_partial` manifest rows for unit 6, not fake tests.
5. **Unit 5 — Ports boundary parity test** (owns
   `tests/integration/test_jax_rcls_ports_boundary.py` [new])
   - [ ] Assert the JAX lane's consumed free-segment index set is the exact complement of
         native `wireframe.constrained_segments()` for the rcls-with-ports case at the
         scale under test, pinning the set's exact indices and size as the case computes
         them (the case's wireframe is 12×22 at both scales — only plasma resolution
         differs — so derive the expected count from the case, never hardcode one; the
         probe's "254 constraints" figure belongs to a different configuration).
6. **Unit 6 — Coverage manifest + drift gate (first slice of the 2026-07-29 plan)**
   (owns the five artifact paths listed in Current Context)
   - [ ] Create `tests/fixtures/jax_native_unit_coverage_manifest.json` seeded with rows
         for: force, strain, curves, curveperturbed, ports, coilset,
         fourier_interpolation, normal_field, spec, vmec-family, virtual_casing, mgrid —
         using the 2026-07-29 plan's disposition vocabulary; encode the three Decisions
         above with dated rationale.
   - [ ] `scripts/jax_native_unit_coverage.py` + `tests/jax/test_native_unit_coverage_manifest.py`:
         fail-closed validator — every file under `tests/` matching the native-surface
         globs must have a manifest row; unknown rows, stale paths, empty reasons fail
         (RED tests first, per the 2026-07-29 plan).
   - [ ] `docs/jax_native_unit_test_coverage.md`: generated summary, marking
         unseeded upstream files as explicit `unclassified` (valid planning state, listed,
         not silent).
7. **Wave close (orchestrator)**
   - [ ] Crucible review of the combined diff (units may not self-certify); fix findings;
         re-run per-file validations.
   - [ ] Commit via commit-only-work, scoped to unit-owned files; never stage the
         concurrent session's `docs/receipts/evidence/nested_ls_*` files.

## Validation Plan

- [ ] Per new/changed test file, one process each:
      `PYTHONPATH=src:build/cp311-cp311-linux_x86_64 JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python -m pytest <file> -q` — green.
- [ ] Native oracles unchanged: `tests/field/test_selffieldforces.py`,
      `tests/geo/test_strainopt.py`, `tests/geo/test_curveperturbed.py` still green
      (one process each) — proves mirrors read, not mutated, native behavior.
- [ ] Manifest gate: `tests/jax/test_native_unit_coverage_manifest.py` green, and the RED
      variants (omitted row / stale path / empty reason) demonstrated failing first.
- [ ] `bash scripts/lint.sh check` and `format` on every touched file (pinned).
- [ ] `tests/test_jax_example_device_assignment.py` still 11/11 (no scoreboard coupling
      expected; guard anyway).
- [ ] Deferred (post-B37, box free): one strict-GPU pass of the new force suite under
      `SIMSOPT_BACKEND_MODE=jax_gpu_parity` on the 5090 — recorded as follow-up, not a
      wave gate.

## Risks and Mitigations

- Risk: `force.py` lacks some native behavior (e.g. a regularized-coil subclass path), and
  a mirror test would fail for implementation reasons, not test reasons.
  Mitigation: unit 1 records such rows as `jax_partial`/`jax_missing` in the manifest and
  skips-with-reason rather than writing tautology tests; the manifest makes the gap
  visible instead of red-washing it.
- Risk: two units touch adjacent conftest/fixture behavior.
  Mitigation: only unit 1 may touch `tests/conftest.py`, and only the one stale-entry
  line; all other units own new files.
- Risk: unit 6 drifts from the 2026-07-29 plan and creates a second coverage authority.
  Mitigation: unit 6 cites that plan per artifact and changes none of its contract text;
  any needed contract deviation is reported to the orchestrator, not improvised.
- Risk: native-oracle comparisons entangle test processes with mixed JAX env mutation.
  Mitigation: one file per process (established repo law); native oracle calls stay
  inside the same single-file process.

## Completion Criteria

- [ ] All six units merged, per-file validations green, crucible PASS on the combined diff.
- [ ] Every Goal bullet satisfied; the three Decisions recorded as dated manifest rows.
- [ ] Coverage answerable by command: the manifest validator enumerates
      mirrored / partial / missing / hybrid / native_only for every seeded domain.
- [ ] Follow-ups filed (not gated): strict-GPU pass of the force suite; full manifest
      population per the 2026-07-29 plan's own checklist.

## Open Questions

- Confirm the three non-port rulings (CoilSet `native_only`, PortSet `hybrid_boundary`,
  `fourier_interpolation` `native_only`) — owner: user; unit 6 freezes them once confirmed.
- Does `force.py` cover the regularized-coil *subclasses* the native suite tests
  (circular/rectangular subclass methods), or only the generic path? Unit 1 discovers;
  answer determines how many `jax_partial` rows the manifest opens with.
- Should the QS-residual frozen-data test (JAX `core/quasisymmetry.py`, currently
  untested) join this wave as a seventh unit or ride the 2026-07-29 plan's later phases?
