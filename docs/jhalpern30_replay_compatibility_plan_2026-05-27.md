# jhalpern30 Replay Compatibility Adoption Plan

**Date:** 2026-05-27
**Status:** Implemented in `simsopt-surrogate`; finite-current profile registry
follow-up implemented with focused verification refreshed after donor-repair
retirement
**Scope:** Add an explicit compatibility path for reproducing historical `jhalpern30`
single-stage and stage-2 behavior in this repo.

**Architecture follow-up:** The jhalpern behavior is implemented, but the mode
policy is still too centered on `banana_opt/jhalpern30_compat.py`. The next
cleanup is to promote finite-current mode policy into a shared
`banana_opt/finite_current_profiles.py` registry and leave
`jhalpern30_compat.py` as the historical adapter/builder implementation.

## Reference Points

- Historical source of truth:
  `/Users/suhjungdae/code/columbia/banana_drivers-main/jhalpern30/singlestage.py`
  and `/Users/suhjungdae/code/columbia/banana_drivers-main/jhalpern30/stage2.py`
- Historical VF template:
  `/Users/suhjungdae/code/columbia/banana_drivers-main/inputs/vf_biotsavart.json`
  - 20 VF coils
  - sha256: `1df87dbe845b014199fb1a4a1a414a2dff922d0ae9da10b1861092d0f326d989`
- Current repo finite-current code:
  - `examples/single_stage_optimization/banana_opt/current_contracts.py`
  - `examples/single_stage_optimization/banana_opt/coil_groups.py`
  - `examples/single_stage_optimization/banana_opt/stage2_objectives.py`
  - `examples/single_stage_optimization/banana_opt/stage2_geometry.py`
  - `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py`
  - `examples/single_stage_optimization/banana_opt/boozer_finite_current.py`
  - `examples/single_stage_optimization/banana_opt/stage2_single_stage_handoff.py`
  - `examples/single_stage_optimization/workflow_helpers.py`
  - `examples/single_stage_optimization/workflow_runner_common.py`
  - `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
  - `examples/single_stage_optimization/run_stage2_to_single_stage.py`

## Implementation Evidence

- Added explicit finite-current mode `jhalpern30_proxy_field` while preserving
  `wataru_proxy_field` as the default.
- Vendored the historical 20-coil VF template at
  `examples/single_stage_optimization/banana_opt/jhalpern30_vf_biotsavart.json`
  with sha256
  `1df87dbe845b014199fb1a4a1a414a2dff922d0ae9da10b1861092d0f326d989`.
- Centralized replay behavior in
  `examples/single_stage_optimization/banana_opt/jhalpern30_compat.py`.
- Threaded jhalpern mode through Stage 2 geometry, current resolution,
  workflow command construction, single-stage finite-current parsing, and the
  StageNN importer.
- Direct single-stage replay now honors imported `_flip` metadata via
  `FLIP_BANANA` / `IOTA_TARGET_SIGN` instead of requiring callers to repeat
  `--flip-banana`.
- The StageNN importer now requires an explicit WOUT path and emits the
  April-plus-WOUT coil-seed metadata required by the active
  `stage2_coil_seed_contract_impl_plan_2026-05-26.md` contract.
- The active coil-seed contract intentionally keeps `validate_stage2_seed_contract`
  narrow: WOUT convention, TF current, banana winding-surface radius, and
  curvature threshold. Proxy/VF mode validation is enforced at Stage 2 finite
  current config/import construction instead of becoming a general seed gate.
- Upstream Simsopt `BoozerSurface` does not expose an `I=` constructor in
  `/Users/suhjungdae/code/opensource/simsopt`; this repo ports the legacy
  finite-current behavior through the examples-side
  `banana_opt.boozer_finite_current.BoozerSurfaceFiniteI` wrapper and
  `banana_opt.json_compat.load_boozer_finite_i`.

Focused verification command set:

```bash
PYTHONPATH=build/cp313-cp313-macosx_26_0_arm64:src:examples/single_stage_optimization python3.13 -m pytest tests/geo/test_jhalpern30_compat.py -q
PYTHONPATH=build/cp313-cp313-macosx_26_0_arm64:src:examples/single_stage_optimization python3.13 -m pytest tests/geo/test_stage2_seeded_restart_vf_consistency.py tests/geo/test_wataru_vf_template_resolution.py -q
PYTHONPATH=build/cp313-cp313-macosx_26_0_arm64:src:examples/single_stage_optimization python3.13 -m pytest tests/geo/test_stage2_single_stage_handoff.py -q
PYTHONPATH=build/cp313-cp313-macosx_26_0_arm64:src:examples/single_stage_optimization python3.13 -m pytest tests/geo/test_single_stage_example.py -q
```

## Purpose

Adopt the parts where `jhalpern30` is superior for replaying a historical
champion without weakening this repo's current general finite-current behavior.
The target is not to make `jhalpern30` the default. The target is an explicit,
testable replay mode that can consume old handoff artifacts and reproduce the
old coil-current, proxy-current, VF-current, proxy-placement, Boozer `I`, banana
sign, and iota-target sign while preserving this repo's explicit signed-G
policy.

## Goals

- [x] Add an explicit finite-current compatibility mode, proposed name:
      `jhalpern30_proxy_field`.
- [x] Preserve current `wataru_proxy_field` behavior as the default finite-current
      mode.
- [x] Bundle or vendor the historical 20-coil VF template into this repo.
- [x] Match `jhalpern30` proxy placement: one circular proxy coil at
      `R = surf.major_radius()` and `Z = 0`.
- [x] Match `jhalpern30` signed proxy-current semantics, including negative proxy
      currents.
- [x] Match `jhalpern30` VF current construction, including the shared mutable
      scaled current object and `unfix_all()` behavior.
- [x] Match `jhalpern30` coil-group cardinality:
      `20 TF + 10 banana + 1 proxy + 20 VF = 51 total coils`.
- [x] Do **not** adopt `jhalpern30` fresh-run G0 policy. Keep this repo's
      explicit signed-G policy: `G0 = mu0 * sum(I_tf)`.
- [x] Match `jhalpern30` Boozer finite-current parameter:
      `I = mu0 * proxy_current_A`, separate from `G0`.
- [x] Match `jhalpern30` banana-current sign controls:
      `--flip-banana`, `_flip` output suffix, `BANANA_CURRENT_SIGN`,
      `BANANA_I_FIXED_S2`, and `IOTA_TARGET_SIGN`.
- [x] Match the exact `BANANA_I_FIXED_S2` contract: unset or empty leaves the
      banana current as a free DOF with initial
      `BANANA_CURRENT_SIGN * -10 kA`; set to numeric kA pins the raw
      `Current(1.0)` with `fix_all()` and scales it by
      `BANANA_CURRENT_SIGN * value_kA`.
- [x] Match the exact flip propagation: Stage 2 `--flip-banana` sets
      `BANANA_CURRENT_SIGN=-1`, routes output to `I{kA}_flip`, and single-stage
      replay infers `_flip` from the parent directory to set
      `IOTA_TARGET_SIGN=-1`.
- [x] Keep resume behavior faithful: if a resumed Boozer surface provides saved
      `G`, use saved `G` rather than recomputing fresh-run G0.
- [x] Add a simple handoff/import path for `stageNN/bsurf_opt.json` plus
      `stageNN/state.json` with `stage00` as the primary handoff case.
- [x] Thread the compatibility mode through supported downstream consumers while
      keeping unsupported Wataru-only wrappers explicitly rejected.
- [x] Record enough artifact metadata to prevent silent mode drift.
- [ ] Add a root-level-in-`banana_opt` finite-current profile registry so
      `wataru_proxy_field` and `jhalpern30_proxy_field` are peers with typed
      policy metadata instead of scattered mode-specific constants.
- [ ] Demote `banana_opt/jhalpern30_compat.py` from policy SSOT to historical
      builders/import adapter after the profile registry owns mode metadata.

## Non-Goals

- Do not replace the current signed-G SSOT for normal workflows.
  `mu0 * sum(I_tf)` remains the correct general policy for signed TF currents.
- Do not make `jhalpern30` compatibility the default mode.
- Do not relax Wataru/HBT validators to accept negative proxy/VF current.
  Signed proxy current belongs only to the explicit `jhalpern30_proxy_field`
  mode.
- Do not add silent fallback discovery for missing templates, missing state, or
  wrong artifact names.
- Do not rewrite the existing config/registry pipeline. Use the explicit
  StageNN adapter path for historical bundles.
- Do not route jhalpern replay through donor-repair wrappers. The separate
  donor-repair retirement plan owns that cleanup.
- Do not symlink the historical bundle at runtime. The supported path ports the
  historical `stageNN/bsurf_opt.json` plus `stageNN/state.json` into this
  repo's artifact contract.
- Do not move jhalpern policy to repository root or make it global default
  state. The registry belongs under `banana_opt`, next to `current_contracts.py`,
  `artifact_contracts.py`, and `coil_groups.py`.
- Do not modify the older implemented objective-parity plan in
  `docs/jhalpern30_external_parity_impl_plan_2026-05-12.md`.

## Requirements Analysis

| jhalpern-superior behavior | Current repo state | Required adoption | Rationale |
| --- | --- | --- | --- |
| Faithful historical reproducer | Current code is a structured Stage 2 pipeline, not an exact replay contract | Add explicit `jhalpern30_proxy_field` mode and tests against historical construction | Replay needs the old behavior as source of truth, not a nearby modern interpretation |
| Correct 20-coil VF template | Default bundled `wataru_vf_template.json` has 2 VF coils | Vendor `jhalpern30_vf_biotsavart.json` and resolve it only for the compatibility mode | Historical runs used the 20-coil input template |
| Historical proxy placement | Wataru proxy uses scaled VMEC axis zeroth coefficients `raxis_cc[0]` and `zaxis_cs[0]` | Add mode-specific builder using `surf.major_radius()` and `Z = 0` | Proxy field geometry is part of the old basin |
| Signed proxy current | Wataru validator rejects negative proxy/VF current | Add a separate jhalpern validator that allows signed proxy current | Current validator is correct for Wataru mode, but not for replay |
| Historical coil cardinality | Current Wataru finite-current path is typically `20 TF + 10 banana + 1 proxy + 2 VF = 33` coils | Add jhalpern mode layout `20 TF + 10 banana + 1 proxy + 20 VF = 51` coils | The saved BiotSavart ordering and group counts are part of replay compatibility |
| VF current mutability and signs | Current repo fixes independent VF currents | Add jhalpern VF builder using shared `ScaledCurrent(Current(1.0), VF_CURRENT_A)` and `unfix_all()`; effective currents are `abs(proxy_current_A) / 6.5 * sign(template_current)` for nonzero proxy | Old optimization allowed VF current DOFs to move, and the sign algebra is part of replay |
| Boozer finite-current `I` | Current repo already supports `mu0 * plasma_current_A` for Wataru mode | Preserve that convention for jhalpern mode and label it separately from `G0` | jhalpern passes `I=mu0 * proxy_current_A` into `BoozerSurface` |
| Banana sign and iota sign | Current repo has its own banana seed and current controls | Add replay mapping for `--flip-banana`, `_flip`, `BANANA_CURRENT_SIGN`, `BANANA_I_FIXED_S2`, and `IOTA_TARGET_SIGN` | Historical `_flip` changes both banana-current sign and the iota target sign |
| Banana current pinning | Current repo uses CLI/config current controls, not jhalpern's env toggle | Preserve jhalpern's env contract: unset or empty `BANANA_I_FIXED_S2` leaves the banana current free; numeric `BANANA_I_FIXED_S2` pins the raw current DOF and applies `BANANA_CURRENT_SIGN` to the scale | This changes whether banana current is an optimizer variable |
| Hardcoded negative G0 | Current repo derives signed G from signed TF currents | Do not adopt jhalpern's `-mu0 * sum(abs(I_tf))`; keep explicit signed-G as the repo policy | The jhalpern value is historical basin behavior, not the correct general current contract |
| Simple handoff path | Current driver expects config, registry, and fixed artifact layout | Add explicit importer for `stageNN/bsurf_opt.json` plus `stageNN/state.json`, with `stage00` as the primary case | Historical handoffs are one script plus per-stage state |
| Downstream consumer gates | Single-stage accepts `jhalpern30_proxy_field`; pre-Boozer repair via `run_stage2_to_single_stage.py` remains Wataru-only | Keep direct single-stage replay supported, and keep unsupported wrappers explicitly rejecting jhalpern mode | A Stage 2-only mode does not satisfy the stated single-stage replay scope, but pre-Boozer repair is not the right replay route |
| Less abstraction drift | Current pipeline has multiple config/artifact layers | Centralize historical replay implementation in one compatibility module | Fewer scattered historical builders/parsers means fewer mode-default mismatches |
| Finite-current policy SSOT | Mode policy is split across `current_contracts.py`, `jhalpern30_compat.py`, `stage2_geometry.py`, `workflow_helpers.py`, `banana_coil_solver.py`, and single-stage parsing | Add `banana_opt/finite_current_profiles.py` as the typed registry for layout, policies, templates, and entrypoint support | Future modes should add one profile rather than coordinated edits across callers |

## Architecture Follow-up: Finite-Current Profiles

Current implementation chose the smallest working replay adapter first:
`jhalpern30_compat.py` owns historical constants, builders, template validation,
and the StageNN importer. That is acceptable for the initial port, but it is not
the right long-term SSOT because generic callers must still know the mode string
and import jhalpern-specific constants.

Two designs were considered:

1. Keep `jhalpern30_compat.py` as the policy owner.
   - Rejected for long-term maintenance: it makes a historical adapter the
     place where generic finite-current policy lives.
2. Add `banana_opt/finite_current_profiles.py`.
   - Chosen: it gives `wataru_proxy_field` and `jhalpern30_proxy_field` equal,
     typed profile records while preserving `jhalpern30_compat.py` for
     implementation details that are truly historical.

Profile fields should include:

- finite-current mode name
- coil-layout counts and `COIL_GROUPS` manifest construction inputs
- Boozer current convention
- G0 policy name
- proxy placement policy
- VF template path and hash
- VF current sign/mutability policy
- banana sign/pinning/replay policy
- supported entrypoints
- explicitly rejected entrypoints
- artifact metadata keys required for this mode

## Current Implementation Status

### Completed port

- [x] Added explicit finite-current mode `jhalpern30_proxy_field` in
      `banana_opt/current_contracts.py` while preserving
      `DEFAULT_FINITE_CURRENT_MODE = "wataru_proxy_field"`.
- [x] Added mode-specific current convention metadata so jhalpern Boozer current
      remains `mu0`-normalized without changing Wataru validation.
- [x] Centralized historical policy in
      `banana_opt/jhalpern30_compat.py` instead of spreading constants through
      Stage 2 and single-stage call sites.
- [x] Vendored the historical 20-coil VF template as
      `banana_opt/jhalpern30_vf_biotsavart.json` and pinned the recorded sha256.
- [x] Implemented `resolve_jhalpern30_vf_template_path` with no absolute runtime
      dependency on `/Users/suhjungdae/code/columbia/banana_drivers-main`.
- [x] Implemented `validate_jhalpern30_vf_template` so the bundled template's
      hash, 20-coil count, and template signs are testable.
- [x] Implemented `build_jhalpern30_proxy_plasma_current_coils(surface,
      proxy_current_A)` using `surface.major_radius()` and `Z = 0`.
- [x] Implemented `build_jhalpern30_vf_coils(proxy_current_A, template_path)`
      using the 20-coil template, shared mutable `ScaledCurrent(Current(1.0),
      VF_CURRENT_A)`, `unfix_all()`, fixed VF curves, and template-sign effective
      currents.
- [x] Implemented `build_jhalpern30_banana_coils(...)` and replay helpers for
      `BANANA_CURRENT_SIGN`, `BANANA_I_FIXED_S2`, `_flip`, and
      `IOTA_TARGET_SIGN`.
- [x] Added mode dispatch in `banana_opt/stage2_geometry.py` so Wataru keeps the
      VMEC-axis proxy plus 2-coil VF behavior while jhalpern gets historical
      proxy/VF/banana construction.
- [x] Added `validate_jhalpern30_proxy_vf_current_convention` and dispatched to
      it only for `jhalpern30_proxy_field`.
- [x] Kept `validate_hbt_proxy_vf_current_convention` unchanged for Wataru mode;
      Wataru still rejects negative proxy/VF current.
- [x] Threaded mode-aware VF template resolution through workflow helpers.
- [x] Added single-stage CLI/current-resolution support for
      `jhalpern30_proxy_field`.
- [x] Preserved signed-G behavior: this repo still treats fresh-run
      `G0 = mu0 * sum(I_tf)` as the general policy and records
      `G0_POLICY = signed_explicit_tf_current` for imported jhalpern artifacts.
- [x] Added Boozer finite-current normalization as a separate contract:
      `BOOZER_I = mu0 * proxy_current_A`.
- [x] Implemented the StageNN adapter in
      `import_jhalpern30_replay.py` and
      `banana_opt/jhalpern30_compat.py::import_jhalpern30_stage_bundle`.
- [x] Made the adapter a port/conversion step, not a symlink: it writes
      `biot_savart_opt.json`, copies the source `bsurf_opt.json` and
      `state.json`, and writes a current-repo `results.json`.
- [x] Adapter output includes `FINITE_CURRENT_MODE`,
      `BOOZER_CURRENT_CONVENTION`, `BOOZER_I`, `PROXY_PLACEMENT_MODE`,
      `G0_POLICY`, `PROXY_PLASMA_CURRENT_A`, `VF_CURRENT_A`,
      `VF_TEMPLATE_PATH`, `VF_TEMPLATE_SHA256`, `VF_CURRENT_SIGN_POLICY`,
      `VF_CURRENT_MUTABILITY`, `FLIP_BANANA`, `BANANA_CURRENT_SIGN`,
      `BANANA_CURRENT_PINNED`, `BANANA_I_FIXED_S2_KA`, `IOTA_TARGET_SIGN`,
      `NUM_TF_COILS`, `NUM_BANANA_COILS`, `NUM_PROXY_COILS`, `NUM_VF_COILS`,
      `TOTAL_COILS`, `COIL_GROUPS`, source hashes, and StageNN state
      provenance.
- [x] Kept `run_stage2_to_single_stage.py` Wataru-only for pre-Boozer repair; it
      explicitly rejects non-Wataru finite-current modes instead of pretending
      to support jhalpern replay.
- [x] Kept `validate_stage2_seed_contract` narrow and aligned with the active
      coil-seed contract. Proxy/VF validation belongs to finite-current mode
      construction and the importer, not the general seed gate.

### Remaining cleanup

1. Documentation and user-facing path cleanup
   - [ ] Update user-facing docs to describe the supported jhalpern path as:
         historical bundle -> `import_jhalpern30_replay.py` -> current-repo
         replay artifact -> direct single-stage replay.
   - [x] Remove or reword stale text that implies jhalpern replay should flow
         through donor repair or `run_stage2_to_single_stage.py`.
   - [x] After donor-repair retirement commit `9676c40f5`, keep only historical
         or explicit rejection-boundary references to
         `run_single_stage_donor_repair.py`.
   - [ ] Add a short "ported, not symlinked" note to the relevant handoff docs
         so future users understand why `results.json` is regenerated.
2. Finite-current profile registry
   - [ ] Add `examples/single_stage_optimization/banana_opt/finite_current_profiles.py`.
   - [ ] Define a frozen `FiniteCurrentProfile` data object with fields for:
         mode, coil counts, Boozer current convention, G0 policy, proxy placement
         policy, VF template path/hash, VF current policy, banana replay policy,
         supported entrypoints, rejected entrypoints, and required artifact
         metadata keys.
   - [ ] Add `get_finite_current_profile(mode)` and fail loudly for unsupported
         modes.
   - [ ] Add profiles for `wataru_proxy_field` and `jhalpern30_proxy_field`.
   - [ ] Keep `DEFAULT_FINITE_CURRENT_MODE = "wataru_proxy_field"` in
         `current_contracts.py`; do not make the registry change the default.
   - [ ] Move shared policy constants out of `jhalpern30_compat.py` into the
         jhalpern profile:
         `JHALPERN30_NUM_TF_COILS`, `JHALPERN30_NUM_BANANA_COILS`,
         `JHALPERN30_NUM_PROXY_COILS`, `JHALPERN30_NUM_VF_COILS`,
         `JHALPERN30_G0_POLICY`, `JHALPERN30_PROXY_PLACEMENT_MODE`,
         `JHALPERN30_VF_CURRENT_SIGN_POLICY`,
         `JHALPERN30_VF_CURRENT_MUTABILITY`, and
         `JHALPERN30_VF_TEMPLATE_SHA256`.
   - [ ] Keep implementation-only helpers in `jhalpern30_compat.py`: historical
         builders, StageNN importer, template validation, stage-state parsing,
         `_flip` parsing, and `BANANA_I_FIXED_S2` replay helper logic.
3. Consumer migration
   - [ ] Replace scattered `finite_current_mode == "jhalpern30_proxy_field"` or
         `JHALPERN30_FINITE_CURRENT_MODE` checks with profile queries when the
         caller only needs policy metadata.
   - [ ] Audit `SINGLE_STAGE/single_stage_banana_example.py`,
         `STAGE_2/banana_coil_solver.py`, `workflow_helpers.py`, and
         `stage2_geometry.py` for copied jhalpern constants that should read
         from `finite_current_profiles.py`.
   - [ ] Keep the 51-coil manifest construction metadata-driven through
         `COIL_GROUPS`; do not add position-based special cases outside the
         importer and geometry builders.
   - [ ] Keep mode-specific builder dispatch in `stage2_geometry.py`; profile
         metadata should select policy, not hide distinct construction logic
         behind a shallow pass-through abstraction.
4. Rejection-boundary cleanup
   - [ ] Keep a test proving `run_stage2_to_single_stage.py` rejects
         `jhalpern30_proxy_field` with a clear Wataru-only message.
   - [ ] Keep a test proving single-stage accepts `jhalpern30_proxy_field` when
         the artifact metadata is present.
   - [ ] Ensure any future wrapper that consumes `FINITE_CURRENT_MODE` either
         handles `jhalpern30_proxy_field` explicitly or rejects it with a
         mode-specific error.
5. Evidence refresh
   - [x] Re-run the focused jhalpern and Wataru regression tests after the
         donor-repair cleanup was committed.
   - [ ] Record the exact passing commands and commit hash in this document if
         this plan is used as handoff evidence.

## Validation Plan

- [ ] Add and run finite-current profile tests:

```bash
PYTHONPATH=build/cp313-cp313-macosx_26_0_arm64:src:examples/single_stage_optimization python3.13 -m pytest tests/geo/test_finite_current_profiles.py -q
```

Required assertions:

- `get_finite_current_profile("wataru_proxy_field")` preserves the Wataru
  default profile and 2-VF-template path.
- `get_finite_current_profile("jhalpern30_proxy_field")` reports
  `20 TF + 10 banana + 1 proxy + 20 VF = 51`.
- the jhalpern profile has `G0_POLICY = signed_explicit_tf_current`.
- the jhalpern profile marks `run_stage2_to_single_stage.py` pre-Boozer repair
  as unsupported.
- the StageNN importer emits metadata matching the jhalpern profile.

- [ ] Run the focused jhalpern compatibility test:

```bash
PYTHONPATH=build/cp313-cp313-macosx_26_0_arm64:src:examples/single_stage_optimization python3.13 -m pytest tests/geo/test_jhalpern30_compat.py -q
```

- [ ] Run the Wataru/VF regression tests:

```bash
PYTHONPATH=build/cp313-cp313-macosx_26_0_arm64:src:examples/single_stage_optimization python3.13 -m pytest tests/geo/test_stage2_seeded_restart_vf_consistency.py tests/geo/test_wataru_vf_template_resolution.py -q
```

- [ ] Run the Stage 2/single-stage handoff regression suite:

```bash
PYTHONPATH=build/cp313-cp313-macosx_26_0_arm64:src:examples/single_stage_optimization python3.13 -m pytest tests/geo/test_stage2_single_stage_handoff.py -q
```

- [ ] Run the single-stage example regression suite:

```bash
PYTHONPATH=build/cp313-cp313-macosx_26_0_arm64:src:examples/single_stage_optimization python3.13 -m pytest tests/geo/test_single_stage_example.py -q
```

- [ ] Run static/doc hygiene:

```bash
python -m ruff check examples/single_stage_optimization tests/geo
git diff --check
```

## Risks and Mitigations

- Risk: VF current sign algebra is easy to simplify incorrectly.
  Mitigation: Keep tests that inspect the shared leaf current, `unfix_all()`,
  fixed VF curves, and template-sign effective currents.
- Risk: Future cleanup may accidentally route jhalpern replay through
  Wataru-only pre-Boozer repair wrappers.
  Mitigation: Keep explicit rejection tests for `run_stage2_to_single_stage.py`
  and document direct single-stage replay as the supported path.
- Risk: The jhalpern negative-abs G0 policy may be reintroduced while copying
  other jhalpern behavior.
  Mitigation: Keep `G0_POLICY = signed_explicit_tf_current` metadata and tests
  that distinguish `G0 = mu0 * sum(I_tf)` from `-mu0 * sum(abs(I_tf))`.
- Risk: Historical Boozer JSON files carrying `BoozerSurface(I=...)` metadata
  will not load through plain upstream Simsopt.
  Mitigation: Use this repo's `load_boozer_finite_i` decoder for the adapter and
  fail loudly on unsupported object/version load errors; do not silently rebuild
  a different seed.
- Risk: 51-coil ordering can drift if code relies on positional assumptions
  instead of metadata.
  Mitigation: Keep `COIL_GROUPS` and `NUM_*` metadata emitted by the adapter and
  consumed by partition/load code.
- Risk: A profile registry can become a shallow pass-through if it only wraps
  existing `if mode == ...` branches.
  Mitigation: Store durable policy data in the profile and leave construction
  logic in builder modules; callers should use the profile only when they need
  policy metadata.
- Risk: Moving constants can create a partial-conversion state where both
  `jhalpern30_compat.py` and `finite_current_profiles.py` claim to own the same
  policy.
  Mitigation: Move profile-owned constants in one scoped change and leave
  adapter-owned functions in `jhalpern30_compat.py`.

## Completion Criteria

- [x] `jhalpern30_proxy_field` builds the historical proxy and VF layout.
- [x] `jhalpern30_proxy_field` emits
      `20 TF + 10 banana + 1 proxy + 20 VF = 51` total coils.
- [x] The bundled VF template has 20 coils and matches the historical template
      hash.
- [x] Negative proxy current is accepted only in jhalpern mode.
- [x] VF currents are mutable only in jhalpern mode.
- [x] Fresh jhalpern-compat paths preserve this repo's explicit signed-G policy:
      `mu0 * sum(I_tf)`.
- [x] Jhalpern Boozer finite-current metadata records
      `I = mu0 * proxy_current_A`.
- [x] Resume/imported jhalpern artifacts preserve saved `G` as StageNN
      provenance.
- [x] Banana sign, `_flip`, `BANANA_I_FIXED_S2`, and iota-target sign semantics
      are explicit and tested.
- [x] StageNN handoff import is explicit, validated, documented, and writes a
      current-repo artifact bundle.
- [x] Importer emits a canonical `results.json` with `COIL_GROUPS`.
- [x] Artifact metadata distinguishes Wataru and jhalpern runs.
- [x] There is no runtime dependency on
      `/Users/suhjungdae/code/columbia/banana_drivers-main`.
- [ ] `banana_opt/finite_current_profiles.py` exists and owns finite-current
      mode policy metadata for Wataru and jhalpern profiles.
- [ ] `jhalpern30_compat.py` no longer owns shared policy constants; it owns only
      historical builders, StageNN import, and parsing/validation helpers.
- [ ] Consumers that need policy metadata use `get_finite_current_profile(...)`
      instead of importing jhalpern-specific constants.
- [ ] Profile tests prove jhalpern remains 51 coils, Wataru remains default, and
      `run_stage2_to_single_stage.py` remains unsupported for jhalpern replay.
- [x] User-facing docs no longer imply jhalpern replay uses donor repair or
      `run_stage2_to_single_stage.py`.
- [x] Focused validation has been re-run after the donor-repair retirement
      landed.

## Open Questions

- Should `run_stage2_to_single_stage.py` remain permanently Wataru-only, or
  should a future implementation add jhalpern support there? Current decision:
  keep it Wataru-only and route jhalpern replay through the StageNN adapter plus
  direct single-stage replay.
- Should profile-owned constants keep their historical `JHALPERN30_*` export
  names for compatibility, or should callers migrate immediately to profile
  fields? Current preference: migrate internal callers to profile fields and
  keep compatibility aliases only if external scripts require them.
- Which README/examples still need the direct StageNN-adapter-to-single-stage
  replay path documented more prominently?
