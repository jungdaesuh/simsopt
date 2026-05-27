# Finite-Current Proxy, Sign, and VF Policy Implementation Plan

## Purpose

This plan turns the three finite-current decisions into executable work:
proxy-coil placement, signed current handling, and optional optimizable VF
current. It is written for this repo's `wataru_proxy_field` and
`jhalpern30_proxy_field` code paths, with the goal of keeping the production
path simple and explicit while retaining faithful historical replay.

## Goals

- Keep the production/default proxy placement tied to VMEC magnetic-axis data,
  not `Surface.major_radius()`.
- Represent proxy and VF currents as signed physical scalars where the sign is
  physically meaningful; use absolute value only for magnitude limits.
- Add a small, explicit VF-current DOF policy so VF current can be optimized
  when requested without making shared mutable current state the default.
- Keep `G0` handling unchanged: signed explicit TF current remains the repo
  policy.
- Preserve `jhalpern30_proxy_field` as the historical replay profile instead of
  silently changing replay semantics.

## Non-Goals

- Do not adopt jhalpern's hardcoded negative fresh-run `G0` policy.
- Do not make 20 independent VF-current DOFs the default.
- Do not route this through `run_stage2_to_single_stage.py` unless that runner
  explicitly supports the selected finite-current profile.
- Do not add fallback template discovery or path guessing.
- Do not relax hardware/current constraints by widening tolerances.

## Current Context

- `examples/single_stage_optimization/banana_opt/finite_current_profiles.py`
  already defines two profiles:
  - `wataru_proxy_field`: `vmec_axis_zeroth_coefficients`,
    `template_sign_vf_current_scalar`, `independent_fixed_current`.
  - `jhalpern30_proxy_field`: `surface_major_radius_z0`,
    `template_sign_abs_proxy_current`, `shared_unfixed_scaled_current`.
- `examples/single_stage_optimization/banana_opt/stage2_geometry.py` builds the
  Wataru proxy coil from `raxis_cc[0]` and `zaxis_cs[0]`, then fixes the proxy
  and VF currents.
- `examples/single_stage_optimization/banana_opt/jhalpern30_compat.py` builds
  the historical proxy coil from `surface.major_radius(), Z=0` and builds VF
  coils from one shared `ScaledCurrent(Current(1.0), vf_current_A)` with
  `unfix_all()`.
- `examples/single_stage_optimization/banana_opt/current_contracts.py` currently
  treats Wataru proxy/VF current as nonnegative and treats jhalpern proxy/VF
  current as signed.
- The upstream SIMSOPT docs/source support the core mechanisms:
  - VMEC `raxis` and `zaxis` are Fourier modes for the magnetic-axis curve
    (`/Users/suhjungdae/code/opensource/simsopt/docs/source/example_vmec.rst`).
    This repo currently reads the wout `raxis_cc[0]` / `zaxis_cs[0]`
    coefficients, so the Wataru proxy is a zeroth-axis circular approximation,
    not the full magnetic-axis curve.
  - `Surface.major_radius()` is a surface-derived scalar computed from volume
    and minor radius, not the magnetic-axis curve
    (`/Users/suhjungdae/code/opensource/simsopt/src/simsopt/geo/surface.py`).
  - `Current` is the optimizable scalar current object. `ScaledCurrent`
    represents fixed scale/sign relationships around a leaf `Current`; the
    scale itself is not an optimizer DOF
    (`/Users/suhjungdae/code/opensource/simsopt/src/simsopt/field/coil.py`).
  - `fix_all()` and `unfix_all()` recursively decide whether `Current` DOFs are
    optimizer variables, so bounds must target the leaf `Current` returned by
    `unwrap_current_optimizable()`.

## Rationale

The clean design is to keep immutable profile metadata as the SSOT and create
mutable SIMSOPT optimizer objects only at the boundary where coils are built.
That matches functional-programming and thread-safety preferences without
fighting SIMSOPT's optimizer API, which is mutable by design.

For proxy placement, VMEC-axis placement is the stronger production source of
truth because the proxy coil represents plasma current and should be tied to the
magnetic axis. jhalpern's `major_radius(), Z=0` placement remains valuable only
as historical replay behavior.

For current signs, signed current is the more correct physical scalar. A
negative current is a direction, not an invalid magnitude. Magnitude constraints
should be enforced with `abs(current_A)`.

For VF current, fixed independent currents remain the KISS/YAGNI default. When
VF optimization is enabled, the first implementation should be explicit and
shared: one global VF current supply DOF with template signs, not 20 independent
DOFs.

## Assumptions

- The default production mode remains `wataru_proxy_field`.
- Historical replay remains `jhalpern30_proxy_field`.
- The first optimizable VF implementation should expose one shared VF-current
  DOF, not per-coil VF-current DOFs.
- A VF-current optimizer bound must be chosen before enabling shared
  optimizable VF in a real run.

## Implementation Plan

1. Make finite-current policy typed and explicit.
   - [ ] Replace string-only policy fields in `FiniteCurrentProfile` with typed
     `Literal` aliases for proxy placement, proxy/VF sign policy, and VF-current
     DOF policy.
   - [ ] Add a profile-level field that states whether proxy/VF scalar currents
     are signed or magnitude-only.
   - [ ] Keep `wataru_proxy_field` and `jhalpern30_proxy_field` as separate
     profiles; do not merge replay behavior into the default profile.
   - [ ] Add tests in `tests/geo/test_finite_current_profiles.py` asserting the
     exact policy values for both profiles.

2. Item 1: keep production proxy placement VMEC-axis based.
   - [ ] Keep `wataru_proxy_field` default proxy placement as
     `vmec_axis_zeroth_coefficients`.
   - [ ] Add a short code-level contract near
     `build_proxy_plasma_current_coils()` stating that the current implementation
     is a circular proxy coil using the zeroth magnetic-axis coefficients read
     from the VMEC wout.
   - [ ] Add a focused unit test that loads fake `raxis_cc` / `zaxis_cs` data and
     verifies `xc(1)`, `ys(1)`, and `zc(0)` are derived from VMEC-axis values,
     not `surface.major_radius()`.
   - [ ] Keep `surface_major_radius_z0` only in `jhalpern30_proxy_field` tests
     and metadata.
   - [ ] Optional follow-up: add `vmec_axis_fourier_curve` as a new explicit
     proxy placement policy if a real run needs the full magnetic-axis curve.
     If implemented, use all available wout axis Fourier coefficients instead
     of `Surface.major_radius()`. Do not silently replace the current
     zeroth-axis placement without a parity/smoke comparison.

3. Item 2: adopt signed scalar current semantics where the mode permits it.
   - [ ] Split the current validation concepts in `current_contracts.py`:
     ratio validation (`VF_CURRENT_A = PROXY_PLASMA_CURRENT_A / 6.5`) should be
     independent from sign policy.
   - [ ] Add a signed proxy/VF validation helper that accepts negative proxy and
     VF scalars and rejects only ratio mismatch or non-finite values.
   - [ ] Keep the existing nonnegative Wataru helper available until the
     production-mode policy decision is made; do not silently reinterpret old
     artifacts.
   - [ ] If `wataru_proxy_field` is moved to signed scalar semantics, update its
     profile metadata, `validate_proxy_vf_current_convention_for_mode()`, and
     tests in one commit.
   - [ ] Ensure hardware or traversal limits use `abs(current_A)` and preserve
     the original current sign in artifact metadata.
   - [ ] Verify Boozer finite-current conversion remains
     `BOOZER_I = mu0 * signed_proxy_current_A`, separate from `G0`.
   - [ ] Keep signed proxy/VF metadata (`PROXY_PLASMA_CURRENT_A`,
     `VF_CURRENT_A`) separate from scalar banana-current summaries; signed banana
     current consumers must read `BANANA_CURRENTS_A` when `BANANA_CURRENT_A` is
     a legacy or max-abs hardware summary.

4. Item 3: add an explicit shared optimizable VF-current policy.
   - [ ] Introduce a small builder result type, for example
     `VFCoilBuildResult(coils, current_control)`, so callers can bound the
     shared VF DOF without inspecting nested `ScaledCurrent` wrappers.
   - [ ] Keep the existing fixed-current builder behavior for
     `independent_fixed_current`.
   - [ ] Add a shared builder for `shared_unfixed_scaled_current`:
     create one `Current(1.0)`, wrap it in
     `ScaledCurrent(shared_current, vf_current_A)`, apply template signs, and
     call `unfix_all()` only for the explicit shared-optimizable policy.
   - [ ] Route VF builder selection from `FiniteCurrentProfile.vf_current_mutability`
     instead of ad hoc mode checks.
   - [ ] Add a VF-current bound handler modeled after
     `apply_banana_current_upper_bound()` before any
     `shared_unfixed_scaled_current` object enters an optimizer objective.
     Target the leaf `Current`, not `ScaledCurrent.scale`.
   - [ ] Do not mutate `ScaledCurrent.scale`; the optimizer variable is the
     underlying shared `Current(1.0)`.
   - [ ] Add a schema or CLI-owned threshold for `vf_current_max_A` before
     enabling shared optimizable VF in production runs.
   - [ ] Keep 20 independent VF-current DOFs out of scope unless a separate
     physics reason and validation plan are written.

5. Thread the policy through Stage 2 and single-stage.
   - [ ] Update `STAGE_2/banana_coil_solver.py` so seeded current traversal does
     not force shared VF currents back to fixed independent currents when the
     profile says shared optimizable.
   - [ ] Update artifact payload construction so `VF_CURRENT_MUTABILITY`,
     `VF_CURRENT_SIGN_POLICY`, `PROXY_PLACEMENT_MODE`, and signed
     `PROXY_PLASMA_CURRENT_A` / `VF_CURRENT_A` are emitted consistently.
   - [ ] Update single-stage replay parsing/reporting so signed proxy/VF
     metadata remains source-owned and signed banana-current consumers use
     `BANANA_CURRENTS_A` rather than scalar `BANANA_CURRENT_A` when the scalar
     is a max-abs hardware summary.
   - [ ] Keep `run_stage2_to_single_stage.py` rejection behavior for unsupported
     profile paths unless it is explicitly upgraded.

6. Documentation and migration.
   - [ ] Update `docs/jhalpern30_replay_compatibility_plan_2026-05-27.md` only
     if this work changes replay facts already documented there.
   - [ ] Add a short user-facing note explaining the three policies:
     proxy placement, signed current scalar, and VF-current DOF policy.
   - [ ] Document that SIMSOPT mutation is isolated to coil/current object
     construction and optimizer execution; profile specs remain immutable.

## Validation Plan

- [ ] Run focused profile tests:
  `PYTHONPATH=build/cp313-cp313-macosx_26_0_arm64:src:examples/single_stage_optimization python3.13 -m pytest tests/geo/test_finite_current_profiles.py -q`
- [ ] Run proxy/VF geometry tests:
  `PYTHONPATH=build/cp313-cp313-macosx_26_0_arm64:src:examples/single_stage_optimization python3.13 -m pytest tests/geo/test_jhalpern30_compat.py tests/geo/test_wataru_vf_template_resolution.py -q`
- [ ] Run seeded-restart current tests:
  `PYTHONPATH=build/cp313-cp313-macosx_26_0_arm64:src:examples/single_stage_optimization python3.13 -m pytest tests/geo/test_stage2_seeded_restart_vf_consistency.py -q`
- [ ] Run Stage 2 handoff tests:
  `PYTHONPATH=build/cp313-cp313-macosx_26_0_arm64:src:examples/single_stage_optimization python3.13 -m pytest tests/geo/test_stage2_single_stage_handoff.py -q`
- [ ] Run single-stage metadata tests:
  `PYTHONPATH=build/cp313-cp313-macosx_26_0_arm64:src:examples/single_stage_optimization python3.13 -m pytest tests/geo/test_single_stage_example.py -q`
- [ ] Run lint and whitespace checks:
  `ruff check examples/single_stage_optimization/banana_opt examples/single_stage_optimization/STAGE_2 tests/geo`
- [ ] Run `git diff --check`.

## Risks and Mitigations

- Risk: Changing Wataru sign validation could reinterpret existing artifacts.
  Mitigation: Keep sign policy profile-driven and update artifacts/tests in one
  scoped change; do not silently change old metadata.

- Risk: Shared optimizable VF current can become an unbounded escape knob.
  Mitigation: Require a concrete `vf_current_max_A` bound before enabling it in
  optimizer runs.

- Risk: Treating `ScaledCurrent.scale` as the optimized variable would make
  bounds and optimizer state inconsistent with SIMSOPT.
  Mitigation: Keep scale/sign as fixed relationships and bound the leaf
  `Current` returned by `unwrap_current_optimizable()`.

- Risk: Returning only `list[Coil]` hides the shared VF DOF and makes bounds
  fragile.
  Mitigation: Return an explicit builder result with the shared control current.

- Risk: Full VMEC-axis proxy placement changes field geometry enough to break
  old Wataru basins.
  Mitigation: Add it as a new explicit proxy placement policy and require
  parity/smoke evidence before making it default.

- Risk: Mutable SIMSOPT `Current` objects conflict with immutable design
  preferences.
  Mitigation: Keep profile/config dataclasses immutable and isolate mutation to
  the SIMSOPT builder/optimizer boundary.

## Completion Criteria

- [ ] `FiniteCurrentProfile` exposes typed, tested policies for proxy
  placement, current sign semantics, and VF-current DOF behavior.
- [ ] Production/default proxy placement remains VMEC-axis based and tests prove
  it does not use `surface.major_radius()`.
- [ ] Signed current semantics are explicit and artifact-visible wherever they
  are enabled.
- [ ] Shared optimizable VF current is available only through an explicit
  profile/policy and has a bound before optimizer use.
- [ ] Any shared VF-current bound is applied to the leaf `Current`; no plan or
  test assumes `ScaledCurrent.scale` is a mutable optimizer variable.
- [ ] `jhalpern30_proxy_field` replay behavior remains unchanged except for
  deliberate metadata/type tightening.
- [ ] Focused tests, ruff, and `git diff --check` pass.

## Open Questions

- Should `wataru_proxy_field` itself move from nonnegative proxy/VF scalars to
  signed scalar semantics, or should signed scalar semantics be introduced in a
  new production profile name?
- What is the authoritative hardware or experiment limit for `vf_current_max_A`
  if VF current becomes an optimizer DOF?
- Is the full VMEC magnetic-axis curve worth implementing now, or is the current
  zeroth-axis circular proxy sufficient until a physics run shows placement is
  limiting?
