# Finite-Current Proxy Sign Convention Fix Plan

## Purpose

Make finite-current proxy-current sign semantics explicit, mode-aware, and
fail-loud at the operator boundary without breaking upstream jhalpern30 replay
parity. The immediate issue is not a local proxy-winding defect; it is an
unguarded cross-convention interface where `--proxy-current-A` looks like the
same signed-current input as TF and banana currents, while jhalpern30 follows
the upstream jhalpern30 signed proxy-loop convention.

## Goals

- Provide one source of truth for finite-current proxy sign semantics by mode.
- Surface the resolved sign convention in CLI help, materializer summaries, and
  persisted metadata.
- Preserve jhalpern30 / `banana_drivers` parity by keeping the existing proxy
  winding unchanged.
- Keep Wataru/HBT nonnegative-magnitude current semantics distinct from
  jhalpern30 signed-current semantics.
- Add regression tests that catch accidental local winding flips, missing
  metadata, and stale user-facing sign documentation.
- Keep materialized proxy iota evidence separate from trusted equilibrium
  claims until the no-collapse and VMEC `curtor` gates pass.

## Non-Goals

- Do not flip `ys(1)` in
  `examples/single_stage_optimization/banana_opt/jhalpern30_compat.py` as a
  local fix.
- Do not perform an upstream-wide convention migration in this patch.
- Do not redefine TF, banana, VMEC `curtor`, or hardware sign conventions.
- Do not claim that a materializer iota shift is a validated equilibrium result
  without the existing gated VMEC cross-check.
- Do not rerun the finite-current ladder as part of this documentation and
  interface hardening plan.

## Current Context

- `examples/single_stage_optimization/banana_opt/jhalpern30_compat.py` builds
  the jhalpern30 proxy loop with `xc(1)=R` and `ys(1)=R` in
  `build_jhalpern30_proxy_plasma_current_coils()`.
- `../banana_drivers-main/src/banana_drivers/utils/coils.py` uses the same
  `xc(1)=R`, `ys(1)=R` proxy-loop convention in `generate_proxy_coils()`.
  Local blame points to the replay-path port; upstream blame points to the
  Wataru/jhalpern reference implementation. That makes this a parity
  convention, not a newly introduced local winding bug.
- `../HW_constraint.md` documents the hardware TF convention as
  `TF current per coil = -80 kA` with CW toroidal field, and the jhalpern30
  banana init current as negative in the same operating basin.
- Before this fix,
  `examples/single_stage_optimization/materialize_finite_current_seed.py`
  described `--proxy-current-A` only as a physical proxy plasma current in
  amperes. It did not state that Wataru/HBT uses nonnegative magnitude
  semantics while jhalpern30 uses a signed upstream proxy-loop scalar.
- Before this fix,
  `examples/single_stage_optimization/run_materialized_current_sweep.py`
  recorded the current grid and warned that materializer iota is untrusted
  until no-collapse and VMEC checks pass, but it did not print the resolved
  proxy-current sign convention per point.
- `examples/single_stage_optimization/banana_opt/current_contracts.py` already
  separates Wataru/HBT nonnegative proxy/VF current validation from jhalpern30
  signed proxy-current validation.
- `examples/single_stage_optimization/banana_opt/finite_current_profiles.py`
  already records the profile-level scalar policy:
  `nonnegative_magnitude` for Wataru/HBT and `signed_physical_scalar` for
  jhalpern30.
- `examples/single_stage_optimization/banana_opt/finite_current_materializer.py`
  already persists finite-current mode, proxy current, VF current, scalar
  policy, derived `BOOZER_I`, and an explicit untrusted iota status.
- `../autoresearch/GPD/analysis/derivation-current-reversal-sign-flip.md`
  records the deeper physics boundary: a clean `B -> -B` twin is a vacuum-only
  relabel, while finite enclosed current must be regenerated as a real source.
- Before this fix, the regression surface had tests for finite-current
  profiles, jhalpern30 compatibility, materializer metadata, CLI smoke, and
  sweep wrapper behavior, but lacked coverage for the jhalpern30 proxy-loop
  handedness and mode-specific sign-convention text.

## Rationale

The validated issue is at the interface: one numeric flag,
`--proxy-current-A`, appears to share the same sign convention as TF and banana
currents, while the jhalpern30 materializer intentionally follows the upstream
  proxy-loop replay convention. Flipping the local jhalpern30 proxy winding
  would make the materializer diverge from `banana_drivers` and would turn a
  documentation gap into a replay-parity bug.

The production fix is therefore to make the convention boundary explicit and
machine-checkable. The code should tell operators which sign semantics are in
effect for the selected finite-current mode before a run starts, store the same
semantics in output metadata, and test that future edits cannot silently erase
or invert this convention.

This also keeps the physics boundary clean. Materializing proxy and VF coils is
the required finite-current source construction, but materializer iota remains
proxy evidence until the no-collapse gate and VMEC `curtor` comparison validate
the point.

## Assumptions

- jhalpern30 replay parity with `banana_drivers` remains a hard requirement.
- The initial implementation target is
  `/Users/suhjungdae/code/columbia/simsopt-surrogate`.
- Additive metadata fields are acceptable for downstream readers, but existing
  required keys must remain stable.
- Operator-facing text may state the upstream signed proxy-loop convention
  directly. Any stronger "co-current" label must be tied to committed empirical
  evidence or marked as effect-based.
- A global migration to "negative means operating/co-current everywhere" would
  require coordinated edits in `banana_drivers`, local replay code, docs, and
  parity tests.

## Implementation Status

Completed in this checkout on 2026-06-01:

- The descriptor lives in
  `examples/single_stage_optimization/banana_opt/finite_current_profiles.py`.
- Materialized `results.json` files, materializer JSON output, sweep JSON/CSV
  output, and both CLI help surfaces are populated from that descriptor.
- The jhalpern30 replay geometry is unchanged; `jhalpern30_compat.py` has no
  local winding diff.
- Operator documentation was added in
  `docs/finite_current_proxy_sign_conventions.md`.

## Implementation Plan

1. Add a finite-current sign-convention descriptor.
   - [x] Add a small immutable descriptor or helper in
     `examples/single_stage_optimization/banana_opt/current_contracts.py` or
     `examples/single_stage_optimization/banana_opt/finite_current_profiles.py`.
   - [x] Include the selected mode, current scalar policy, signedness, replay
     reference, short operator warning, and metadata-safe convention key.
   - [x] Define Wataru/HBT as nonnegative proxy/VF magnitude semantics.
   - [x] Define jhalpern30 as the signed upstream proxy-loop replay convention.
   - [x] Keep the descriptor free of runtime solver dependencies.

2. Surface the convention at the CLI boundary.
   - [x] Update `materialize_finite_current_seed.py` help text for
     `--proxy-current-A` to state that semantics are mode-specific.
   - [x] Add an argparse epilog or summary text that names the jhalpern30
     signed upstream proxy-loop convention and the Wataru/HBT nonnegative
     magnitude convention.
   - [x] Update `run_materialized_current_sweep.py` so every sweep summary shows
     the resolved sign convention next to the current value.
   - [x] Keep the existing warning that materializer iota is untrusted until
     no-collapse and VMEC gates pass.

3. Persist sign semantics in materialized artifacts.
   - [x] Add additive `results.json` fields such as
     `PROXY_CURRENT_SIGN_CONVENTION`, `PROXY_CURRENT_SIGN_FRAME`, and
     `PROXY_CURRENT_OPERATOR_WARNING`.
   - [x] Populate those fields from the descriptor, not from duplicated string
     literals in the CLI.
   - [x] Preserve existing fields including `FINITE_CURRENT_MODE`,
     `PROXY_VF_CURRENT_SCALAR_POLICY`, `PROXY_PLASMA_CURRENT_A`, `VF_CURRENT_A`,
     `VF_CURRENT_SIGN_POLICY`, `BOOZER_I`, and
     `MATERIALIZED_IOTA_TRUST_STATUS`.
   - [x] Do not change proxy or VF coil geometry while adding metadata.

4. Add regression tests.
   - [x] Add descriptor tests in `tests/geo/test_finite_current_profiles.py` or
     a focused current-contract test.
   - [x] Add materializer metadata tests in
     `tests/geo/test_finite_current_materializer.py` for jhalpern30 signed
     semantics and Wataru/HBT nonnegative semantics.
   - [x] Add a jhalpern30 compatibility regression in
     `tests/geo/test_jhalpern30_compat.py` that asserts the proxy loop still
     uses the upstream `xc(1)=R`, `ys(1)=R` winding, not just the correct
     radius and `z=0` placement.
   - [x] Extend the existing CLI smoke or sweep-wrapper tests to cover
     mode-specific sign-convention text produced from the shared descriptor.

5. Update docs and operator notes.
   - [x] Add a short sign-convention note near the materializer docs explaining
     why local winding flips are forbidden without upstream migration.
   - [x] Document that Wataru/HBT and jhalpern30 intentionally use different
     current input contracts.
   - [x] Document that device-wide sign unification is a future coordinated
     migration, not a local bug fix.
   - [x] Reference the no-collapse plus VMEC `curtor` gate before any topology
     or confinement interpretation.

## Validation Plan

- [x] Run the focused finite-current test files that own profiles,
  jhalpern30 compatibility, materializer metadata, CLI smoke, and sweep wrapper
  behavior:

  ```bash
  PYTHONNOUSERSITE=1 .conda-env/bin/python -m pytest \
    tests/geo/test_finite_current_materializer.py \
    tests/geo/test_finite_current_profiles.py \
    tests/geo/test_jhalpern30_compat.py \
    -q
  ```

  Result: `38 passed, 2 subtests passed`.

- [x] Re-run the signed-Boozer/current-metadata regression slice:

  ```bash
  PYTHONNOUSERSITE=1 .conda-env/bin/python -m pytest \
    tests/geo/test_boozersurface.py::BoozerSurfaceTests::test_finite_current_requires_explicit_signed_G \
    tests/geo/test_boozersurface.py::DeriveSignedGFromFieldTests::test_excludes_non_tf_coils_from_signed_G \
    tests/geo/test_boozersurface.py::SignedGWireInBoozerNewtonConvergenceTests::test_attempt_initialize_boozer_surface_routes_signed_G_through_run_code \
    -q
  ```

  Result: `3 passed, 4 subtests passed`.

- [x] Check the CLI help includes mode-aware sign wording:

  ```bash
  PYTHONNOUSERSITE=1 .conda-env/bin/python \
    examples/single_stage_optimization/materialize_finite_current_seed.py \
    --help
  ```

- [x] Check formatting and obvious diff errors:

  ```bash
  git diff --check
  ```

- [x] Run a focused ruff check on touched Python files:

  ```bash
  PYTHONNOUSERSITE=1 .conda-env/bin/python -m ruff check \
    examples/single_stage_optimization/banana_opt/finite_current_profiles.py \
    examples/single_stage_optimization/banana_opt/finite_current_materializer.py \
    examples/single_stage_optimization/materialize_finite_current_seed.py \
    examples/single_stage_optimization/run_materialized_current_sweep.py \
    tests/geo/test_finite_current_materializer.py \
    tests/geo/test_finite_current_profiles.py \
    tests/geo/test_jhalpern30_compat.py
  ```

  Result: `All checks passed!`.

- [x] Run a focused ruff format check on touched Python files:

  ```bash
  PYTHONNOUSERSITE=1 .conda-env/bin/python -m ruff format --check \
    examples/single_stage_optimization/banana_opt/finite_current_profiles.py \
    examples/single_stage_optimization/banana_opt/finite_current_materializer.py \
    examples/single_stage_optimization/materialize_finite_current_seed.py \
    examples/single_stage_optimization/run_materialized_current_sweep.py \
    tests/geo/test_finite_current_materializer.py \
    tests/geo/test_finite_current_profiles.py \
    tests/geo/test_jhalpern30_compat.py
  ```

  Result: `7 files already formatted`.

- [x] Manually inspect the output metadata from one Wataru/HBT materialization
  and one jhalpern30 materialization to confirm the new sign fields differ by
  mode and that existing current fields are unchanged.

- [x] Confirm `git diff` contains no `ys(1)` sign flip in
  `jhalpern30_compat.py` unless an explicit upstream-coordinated migration is
  being performed.

## Risks And Mitigations

- Risk: The implementation overstates "co-current" without committed empirical
  evidence.
  Mitigation: Separate the upstream signed proxy-loop convention from
  effect-based co-current labels, and require VMEC/materializer evidence before
  stronger labels are persisted.

- Risk: A local winding flip breaks jhalpern30 replay parity.
  Mitigation: Treat proxy-loop geometry as out of scope for this fix and add a
  regression test that pins the upstream winding.

- Risk: The metadata grows into duplicated string state.
  Mitigation: Generate CLI text, summaries, and metadata from one descriptor.

- Risk: Wataru/HBT nonnegative magnitude semantics are accidentally treated as a
  signed physical-current lane.
  Mitigation: Keep Wataru/HBT validation and jhalpern30 validation mode-specific
  in `current_contracts.py`, and test both.

- Risk: Users interpret materializer iota as final confinement proof.
  Mitigation: Preserve `MATERIALIZED_IOTA_TRUST_STATUS` and repeat the
  no-collapse plus VMEC `curtor` gate in summaries and docs.

## Completion Criteria

- [x] `--proxy-current-A --help` text says that sign semantics are mode-specific.
- [x] Materializer and sweep summaries print the selected sign convention.
- [x] Materialized `results.json` files contain additive sign-convention metadata.
- [x] Tests cover descriptor behavior, mode-specific metadata, and no local
  jhalpern30 proxy-winding flip.
- [x] Focused finite-current tests pass in the repo-local `.conda-env`.
- [x] The patch contains no local `ys(1) -> -R` winding change.
- [x] Documentation clearly says that global sign unification requires coordinated
  upstream migration and parity re-tests.

## Resolved Questions

- Resolved: do not persist effect-based "raises iota" labels in this patch;
  persist only the upstream convention and operator warning.
- Resolved: the descriptor lives with profile metadata in
  `finite_current_profiles.py`.
- Resolved for this patch: display the metadata in materializer JSON output and
  sweep JSON/CSV summaries. Topology archive rows remain a separate downstream
  integration decision.
- If device-wide sign unification is later approved, should it be implemented as
  a new finite-current mode first rather than mutating the existing jhalpern30
  replay mode?
