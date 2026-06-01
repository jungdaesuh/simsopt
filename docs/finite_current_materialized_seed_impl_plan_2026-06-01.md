# Finite-Current Materialized Seed Implementation Plan

## Purpose

This plan defines the work to create Stage-2-compatible finite-current
single-stage seeds whose `BiotSavart` field actually contains proxy
plasma-current and VF coils. The materializer is an explicit pre-replay step: it
builds a new artifact with current-backed field sources plus a sibling
`results.json`, then lets single-stage continue to act as a strict loader of
persisted fields.

## Goals

- Add a deterministic materializer that reuses the existing Stage 2 proxy/VF
  builders instead of reimplementing current-source geometry.
- Produce reloadable finite-current `BiotSavart` artifacts with TF, banana,
  proxy, and VF coils in the existing Stage 2 order.
- Produce a sibling Stage-2-style `results.json` that passes current
  single-stage seed loading, checksum, metadata, and hardware-contract checks.
- Derive `BOOZER_I` from the realized proxy coil current in the materialized
  `BiotSavart`, not from an independent CLI or manifest scalar.
- Keep single-stage replay fail-loud when a requested non-vacuum current is not
  backed by loaded proxy/VF field sources.
- Add unit, smoke, and science-gate validation that separate coil-space
  FE/Poincare/topology use from trusted iota/confinement claims.

## Non-Goals

- Do not make single-stage synthesize or retarget proxy/VF coils during replay.
- Do not treat the materialized proxy-coil iota as trusted unless the stability
  sweep and VMEC cross-check pass.
- Do not replace VMEC `curtor` for current-profile or confinement claims.
- Do not change upstream SIMSOPT `BoozerSurface` or core `BiotSavart` APIs.
- Do not merge this with the finite-build exporter or all-family MGrid work;
  finite-build geometry remains a separate artifact-conversion concern.

## Current Context

- `examples/single_stage_optimization/banana_opt/stage2_geometry.py` already
  owns the finite-current proxy/VF builders:
  `build_proxy_plasma_current_coils()`, `build_vf_coils_for_profile()`, and the
  finite-current dispatch that assembles `BiotSavart(tf + banana + proxy + vf)`.
- `examples/single_stage_optimization/banana_opt/jhalpern30_compat.py` derives
  `BOOZER_I` from the realized proxy coil current after loading the constructed
  `BiotSavart`: `proxy_current_A = biotsavart.coils[proxy_index].current.get_value()`
  followed by `BOOZER_I = MU0 * proxy_current_A`.
- `banana_drivers-main/src/banana_drivers/utils/boozersurface.py` uses the same
  source-of-truth pattern: `I = proxy_coil.current.get_value() * MU0`.
- `examples/single_stage_optimization/banana_opt/stage2_single_stage_handoff.py`
  enforces the replay contract: explicit current inputs may only select the
  donor's already-materialized proxy/VF current, and non-vacuum
  `boozer_surrogate` replay is rejected.
- `examples/single_stage_optimization/workflow_runner_common.py` loads Stage 2
  seed metadata from a sibling `results.json` and requires
  `STAGE2_BS_SHA256` to match the loaded `BiotSavart` file.
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
  treats `--single-stage-resume-bs-path` as a debug or strict-vacuum lane; normal
  finite-current seed loading should use an explicit `--stage2-bs-path`.
- `examples/single_stage_optimization/banana_opt/boozer_finite_current.py`
  implements the local finite-enclosed-current residual by passing
  `G_effective = G + iota * I` to the existing vacuum residual; proxy/VF field
  sources still enter through the loaded `BiotSavart`.
- Official SIMSOPT field docs define a `Coil` as a `Curve` plus a `Current`, and
  `BiotSavart` as the field induced by the list of current-carrying coils. That
  makes the realized coil current the authority for field strength.
- Official STELLOPT/VMEC docs state that `NCURR=1` selects prescribed toroidal
  current, `AC` supplies the current-profile coefficients, and `CURTOR` supplies
  the current scale in amperes.
- `../autoresearch/scripts/vmec_operating_range_sweep.py` already provides the
  local VMEC `curtor` oracle lane: `ncurr=1`, current-profile shape from `ac`,
  and net current scale from `curtor`.
- `../autoresearch/analysis/hybrid_operating_range_2026-05-30/operating_range_summary.md`
  shows that VMEC `curtor` moves iota in the operating range; that remains the
  source of truth for iota/confinement claims.

## Rationale

The clean design is to materialize finite-current field sources outside
single-stage and then replay the resulting artifact. That keeps the Stage 2
geometry builders as the single source of truth, preserves single-stage as a
strict loader, and avoids hidden retargeting where a CLI current scalar appears
to change physics without changing the field.

`BOOZER_I` should be stored only as derived metadata. The materialized
`BiotSavart` owns the actual proxy current through its `Current` object; deriving
the Boozer covariant-current label from that object keeps the residual label and
field source consistent by construction.

The materializer is valuable for coil-space finite-current field-error,
Poincare, and topology studies. It is not automatically a faithful equilibrium
iota model because a proxy filament plus VF coils is a prescribed-field
superposition, while VMEC solves an equilibrium with the toroidal current
profile. The implementation must encode that boundary in validation and docs.

## Assumptions

- The first materializer target is a loaded Stage 2 or single-stage-compatible
  `BiotSavart` seed whose TF and banana coils should be preserved.
- The materializer may rebuild proxy/VF coils from the selected finite-current
  profile and requested current, but single-stage replay itself must not.
- Existing finite-current profiles remain the contract for Wataru and
  jhalpern30 proxy placement, VF current, and coil-group manifest metadata.
- `BOOZER_I` is allowed in `results.json` only when it is recomputed from the
  materialized proxy coil current.
- The source artifact has a valid Stage 2 or single-stage-compatible results
  dictionary with `MAJOR_RADIUS`, `TOROIDAL_FLUX`, WOUT convention fields, and
  hardware-contract fields needed by the current seed validators.
- VMEC `curtor` comparison can use the existing `autoresearch` sweep script or a
  small wrapper around it; the materializer should not depend on VMEC at import
  time.

## Implementation Plan

1. Define the materialized-seed contract.
   - [ ] Add `examples/single_stage_optimization/banana_opt/finite_current_materializer.py`.
   - [ ] Define one typed request object covering source artifact path, output
     path, finite-current mode, proxy current in amperes, optional VF template,
     optional Stage 2 `results.json` path, and overwrite policy.
   - [ ] Define one result object covering output `BiotSavart` path,
     `results.json` path, coil counts, realized proxy current, realized VF
     current, derived `BOOZER_I`, and source artifact hash.
   - [ ] Keep the module importable without running SIMSOPT solves or VMEC.
   - [ ] Add a short interface comment stating that the module materializes field
     sources and does not validate equilibrium iota.

2. Load and partition the donor artifact without mutating it.
   - [ ] Reuse `banana_opt.json_compat.load_boozer_finite_i` or
     `simsopt._core.load` as appropriate, matching the finite-build exporter
     pattern.
   - [ ] Normalize supported donors to a `BiotSavart`: raw `BiotSavart`, object
     with `.coils`, or object with `.biotsavart.coils`.
   - [ ] Reuse `partition_loaded_stage2_coils()` when `results.json` metadata is
     available.
   - [ ] For metadata-free donors, require an explicit `--finite-current-mode`
     and use `FiniteCurrentProfile.build_default_coil_groups_manifest()` only
     when the loaded coil count exactly matches the selected profile; mark these
     outputs as export/smoke-only until a full Stage-2-compatible `results.json`
     can be provided.
   - [ ] Copy or rebuild output coils so changing proxy/VF currents in the
     materialized artifact cannot mutate the donor object.
   - [ ] Preserve donor TF and banana coil order exactly.

3. Reuse the Stage 2 finite-current builders.
   - [ ] Route Wataru proxy construction through
     `build_proxy_plasma_current_coils()`.
   - [ ] Route jhalpern30 proxy construction through
     `build_jhalpern30_proxy_plasma_current_coils()`.
   - [ ] Route VF construction through `build_vf_coils_for_profile()`.
   - [ ] Do not duplicate proxy placement, VF sign, or current-mutability policy
     constants in the materializer.
   - [ ] Assemble output coils as `tf_coils + banana_coils + proxy_coils + vf_coils`.
   - [ ] Save the output as a new `BiotSavart` artifact.

4. Derive and persist a Stage-2-compatible `results.json`.
   - [ ] Identify the materialized proxy coil by partition metadata, not by a
     hard-coded index unless the selected finite-current profile provides that
     index contract.
   - [ ] Read `realized_proxy_current_A` from
     `proxy_coil.current.get_value()`.
   - [ ] Compute `BOOZER_I = MU0 * realized_proxy_current_A`.
   - [ ] Read or compute realized VF current from the built VF current control or
     first VF coil current according to the selected profile policy.
   - [ ] Start from the donor `results.json` dictionary and update
     only the current-source fields owned by the materializer.
   - [ ] Write `results.json` next to the materialized `BiotSavart`, including
     `STAGE2_BS_SHA256` for the new `BiotSavart` file.
   - [ ] Preserve or populate every key required by
     `validate_stage2_seed_contract()`: WOUT convention fields, `TF_CURRENT_A`,
     `MAJOR_RADIUS`, `TOROIDAL_FLUX`, banana winding-surface radius, curvature
     threshold, and hardware-contract metrics.
   - [ ] Preserve or populate every selected
     `FiniteCurrentProfile.required_artifact_metadata_keys` entry, including
     `NUM_TF_COILS`, `NUM_BANANA_COILS`, `NUM_PROXY_COILS`, `NUM_VF_COILS`,
     `TOTAL_COILS`, and `COIL_GROUPS`.
   - [ ] For `jhalpern30_proxy_field`, preserve required historical replay
     fields such as `FLIP_BANANA`, `BANANA_CURRENT_SIGN`,
     `BANANA_CURRENT_PINNED`, `BANANA_I_FIXED_S2_KA`, `IOTA_TARGET_SIGN`,
     `JHALPERN30_STAGE_NAME`, and `JHALPERN30_STAGE_STATE`; fail loudly if the
     donor metadata cannot supply them.
   - [ ] Reject `results.json` writes where `BOOZER_I` does not match the
     realized proxy current within an explicit numerical tolerance.

5. Add the CLI wrapper.
   - [ ] Add `examples/single_stage_optimization/materialize_finite_current_seed.py`
     as a thin wrapper around the materializer module.
   - [ ] Use `import_provenance.configure_local_simsopt_imports(__file__)` so the
     wrapper uses the local checkout.
   - [ ] Support required `--source-biot-savart`, `--output-root`,
     `--finite-current-mode`, and `--proxy-current-A`.
   - [ ] Support optional `--stage2-results`, `--vf-template-path`,
     `--vf-current-A`, `--toroidal-flux`, `--surface-scale-factor`,
     `--nphi`, `--ntheta`, and `--overwrite` only when required by the reused
     Stage 2 builders.
   - [ ] Write `<output-root>/biot_savart_opt.json` and
     `<output-root>/results.json` so downstream users can pass
     `--stage2-bs-path <output-root>/biot_savart_opt.json`.
   - [ ] Print a concise summary: input path, output paths, source/output coil
     counts, realized proxy/VF currents, derived `BOOZER_I`, checksum status,
     and whether the artifact is eligible for normal single-stage replay.

6. Keep single-stage strict and add integration points only at the boundary.
   - [ ] Do not import the materializer from `single_stage_banana_example.py`.
   - [ ] Ensure the materializer output `results.json` satisfies
     `validate_loaded_seed_current_source_contract()`.
   - [ ] Add docs in the CLI help and generated `results.json` metadata stating
     that retargeting current requires materializing a new seed.
   - [ ] Add a single-stage smoke test that loads a materialized seed through
     `--stage2-bs-path` and passes `load_stage2_artifact_results()`,
     `validate_stage2_seed_contract()`, and
     `validate_loaded_seed_current_source_contract()` before any optimization
     step.

7. Add science gates for iota trust.
   - [ ] Add a small sweep wrapper under
     `examples/single_stage_optimization/run_materialized_current_sweep.py` that
     materializes several currents from the same donor and records Boozer solve
     status, iota, field error, topology metrics, and current metadata.
   - [ ] Require a no-collapse gate: every accepted materialized point has finite
     field values, finite Boozer residual metrics, no self-intersection flag,
     and no failed final iota solve flag.
   - [ ] Require a monotonicity gate over the selected current window unless the
     run explicitly records a physics reason for non-monotonic response.
   - [ ] Compare materialized iota response against the VMEC `curtor` sweep for
     the same current signs and approximate current magnitudes.
   - [ ] Mark materialized iota as untrusted in metadata unless the stability and
     VMEC-cross-check gates pass.

## Validation Plan

- [ ] Unit test donor immutability: after materialization, donor coil currents
  and donor `B(points)` are unchanged.
- [ ] Unit test Wataru materialization: output has TF + banana + proxy + VF
  partitions, `results.json` current metadata, and a finite `B(points)`
  evaluation.
- [ ] Unit test jhalpern30 materialization: output matches jhalpern30 proxy/VF
  counts and derives `BOOZER_I` from the realized proxy coil current.
- [ ] Unit test that changing `--proxy-current-A` changes sampled `B(points)` at
  fixed points, proving current is a field source rather than a label-only input.
- [ ] Unit test that `BOOZER_I` is recomputed from the realized proxy coil and
  fails if `results.json` metadata is inconsistent.
- [ ] Unit test metadata-free fallback: explicit finite-current mode plus exact
  coil-count match succeeds only for export/smoke output; normal single-stage
  replay eligibility remains false without a complete `results.json` contract.
- [ ] Unit test `results.json` checksum binding: `STAGE2_BS_SHA256` matches the
  materialized `BiotSavart`, and a tampered file fails through
  `load_stage2_artifact_results()`.
- [ ] Unit test Stage-2 contract compatibility:
  `validate_stage2_seed_contract()` accepts materialized replay-eligible output.
- [ ] Regression test single-stage guard acceptance for materialized proxy/VF
  donors and rejection for non-vacuum donors without proxy/VF field sources.
- [ ] Smoke test CLI on a small synthetic or fixture artifact and reload the
  saved `BiotSavart`.
- [ ] Run focused tests:
  ```bash
  PYTHONNOUSERSITE=1 PYTHONPATH=examples/single_stage_optimization \
    .conda-env/bin/python -m pytest -q \
    tests/geo/test_stage2_single_stage_handoff.py \
    tests/geo/test_jhalpern30_compat.py \
    tests/geo/test_finite_current_materializer.py
  ```
- [ ] Run format/lint/compile checks on touched files:
  `ruff format --check <touched files>`,
  `ruff check <touched files>`,
  and `.conda-env/bin/python -m py_compile <touched python files>`.
- [ ] Run one materialized-current sweep and store a summary table with iota
  stability, monotonicity, and VMEC comparison status before using materialized
  iota in any science claim.
- [ ] Run or refresh the VMEC oracle comparison with
  ```bash
  cd ../autoresearch
  PYTHONNOUSERSITE=1 python scripts/vmec_operating_range_sweep.py ...
  ```
  for the same current sign convention and current window.

## Risks and Mitigations

- Risk: The materializer duplicates Stage 2 proxy/VF geometry policy.
  Mitigation: Call the existing Stage 2 builders and profile helpers directly;
  add tests that profile metadata is consumed rather than copied.

- Risk: `BOOZER_I` drifts from the actual field source.
  Mitigation: Derive it after materialization from `proxy_coil.current.get_value()`
  and reject inconsistent metadata.

- Risk: The materialized field artifact loads, but its sibling `results.json`
  fails the Stage 2 checksum or seed-contract validators.
  Mitigation: Write `STAGE2_BS_SHA256` for the new artifact and run
  `load_stage2_artifact_results()` plus `validate_stage2_seed_contract()` in
  tests.

- Risk: Donor artifacts are accidentally mutated by shared `Current` objects.
  Mitigation: Build output coils with new current objects where currents are
  changed; add donor-current and donor-field immutability tests.

- Risk: Users mistake materialized proxy-coil iota for a self-consistent
  equilibrium result.
  Mitigation: Metadata must carry an explicit iota-trust flag; default to
  untrusted until no-collapse, monotonicity, and VMEC comparison pass.

- Risk: Metadata-free fallback silently assigns the wrong coil partition.
  Mitigation: Require explicit finite-current mode and exact coil-count match;
  prefer `results.json` metadata whenever available.

- Risk: VMEC and materializer current signs are compared under different
  conventions.
  Mitigation: Record sign convention fields in both summaries and compare only
  after mapping to the same physical current direction.

- Risk: The current-sweep gate becomes too expensive for ordinary development.
  Mitigation: Keep unit and smoke tests small; run the science sweep as an
  explicit acceptance artifact, not as part of default pytest.

## Completion Criteria

- [ ] A materializer module and CLI exist and are documented.
- [ ] The CLI writes a reloadable `BiotSavart` plus sibling `results.json` for
  Wataru and jhalpern30 finite-current modes.
- [ ] The materialized artifact can be passed to single-stage as
  `--stage2-bs-path <output-root>/biot_savart_opt.json` without checksum,
  metadata, or current-source guard failures.
- [ ] `BOOZER_I` is derived from the realized proxy coil current and covered by
  regression tests.
- [ ] Donor immutability and `B(points)` current-response tests pass.
- [ ] Single-stage accepts materialized proxy/VF donors and still rejects
  label-only non-vacuum replay.
- [ ] Focused tests, lint, formatting, py_compile, and `git diff --check` pass
  for touched files.
- [ ] A current-sweep science-gate artifact states whether materialized iota is
  trusted or VMEC-only for the selected donor/current window.
- [ ] Documentation clearly states that VMEC `curtor` remains the oracle for
  current-driven iota/confinement claims.

## Open Questions

- Should the first materializer support only Stage 2 `results.json`-backed
  donors, or also metadata-free donors through explicit finite-current profiles?
- Should Wataru materialization require a VMEC equilibrium file every time, or
  can source `results.json` metadata provide enough axis/proxy placement
  information?
- What exact tolerance should gate `BOOZER_I == MU0 * realized_proxy_current_A`?
- Which donor metadata fields, if any, may be intentionally rewritten besides
  current-source fields and `STAGE2_BS_SHA256`?
- Which donor/current window should be the first binding science-gate fixture for
  the VMEC cross-check?
- Should materialized iota trust metadata be a simple boolean, or a structured
  status such as `unrun|failed|passed` with links to sweep artifacts?
