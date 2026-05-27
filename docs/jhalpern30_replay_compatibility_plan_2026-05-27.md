# jhalpern30 Replay Compatibility Adoption Plan

**Date:** 2026-05-27
**Status:** Draft, not implemented
**Scope:** Add an explicit compatibility path for reproducing historical `jhalpern30`
single-stage and stage-2 behavior in this repo.

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

## Purpose

Adopt the parts where `jhalpern30` is superior for replaying a historical
champion without weakening this repo's current general finite-current behavior.
The target is not to make `jhalpern30` the default. The target is an explicit,
testable replay mode that can consume old handoff artifacts and reproduce the
old coil-current, proxy-current, VF-current, proxy-placement, Boozer `I`, banana
sign, and iota-target sign while preserving this repo's explicit signed-G
policy.

## Goals

- [ ] Add an explicit finite-current compatibility mode, proposed name:
      `jhalpern30_proxy_field`.
- [ ] Preserve current `wataru_proxy_field` behavior as the default finite-current
      mode.
- [ ] Bundle or vendor the historical 20-coil VF template into this repo.
- [ ] Match `jhalpern30` proxy placement: one circular proxy coil at
      `R = surf.major_radius()` and `Z = 0`.
- [ ] Match `jhalpern30` signed proxy-current semantics, including negative proxy
      currents.
- [ ] Match `jhalpern30` VF current construction, including the shared mutable
      scaled current object and `unfix_all()` behavior.
- [ ] Match `jhalpern30` coil-group cardinality:
      `20 TF + 10 banana + 1 proxy + 20 VF = 51 total coils`.
- [ ] Do **not** adopt `jhalpern30` fresh-run G0 policy. Keep this repo's
      explicit signed-G policy: `G0 = mu0 * sum(I_tf)`.
- [ ] Match `jhalpern30` Boozer finite-current parameter:
      `I = mu0 * proxy_current_A`, separate from `G0`.
- [ ] Match `jhalpern30` banana-current sign controls:
      `--flip-banana`, `_flip` output suffix, `BANANA_CURRENT_SIGN`,
      `BANANA_I_FIXED_S2`, and `IOTA_TARGET_SIGN`.
- [ ] Match the exact `BANANA_I_FIXED_S2` contract: unset or empty leaves the
      banana current as a free DOF with initial
      `BANANA_CURRENT_SIGN * -10 kA`; set to numeric kA pins the raw
      `Current(1.0)` with `fix_all()` and scales it by
      `BANANA_CURRENT_SIGN * value_kA`.
- [ ] Match the exact flip propagation: Stage 2 `--flip-banana` sets
      `BANANA_CURRENT_SIGN=-1`, routes output to `I{kA}_flip`, and single-stage
      replay infers `_flip` from the parent directory to set
      `IOTA_TARGET_SIGN=-1`.
- [ ] Keep resume behavior faithful: if a resumed Boozer surface provides saved
      `G`, use saved `G` rather than recomputing fresh-run G0.
- [ ] Add a simple handoff/import path for `stageNN/bsurf_opt.json` plus
      `stageNN/state.json` with `stage00` as the primary handoff case.
- [ ] Thread the compatibility mode through downstream consumers that currently
      accept only `boozer_surrogate` or `wataru_proxy_field`.
- [ ] Record enough artifact metadata to prevent silent mode drift.

## Non-Goals

- [ ] Do not replace the current signed-G SSOT for normal workflows.
      `mu0 * sum(I_tf)` remains the correct general policy for signed TF currents.
- [ ] Do not make `jhalpern30` compatibility the default mode.
- [ ] Do not relax Wataru/HBT validators to accept negative proxy/VF current.
      Signed proxy current belongs only to the explicit `jhalpern30_proxy_field`
      mode.
- [ ] Do not add silent fallback discovery for missing templates, missing state,
      or wrong artifact names.
- [ ] Do not rewrite the existing config/registry pipeline. Add an adapter or
      explicit import path for historical bundles.
- [ ] Do not modify the older implemented objective-parity plan in
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
| Downstream consumer gates | Single-stage CLI and repair wrappers currently reject non-Wataru replay modes | Update or explicitly bypass those consumers | A Stage 2-only mode does not satisfy the stated single-stage replay scope |
| Less abstraction drift | Current pipeline has multiple config/artifact layers | Centralize replay policy in one compatibility module | Fewer scattered conditionals means fewer mode-default mismatches |

## Design Decisions

### 1. Use an explicit mode

Add `jhalpern30_proxy_field` to the finite-current mode model. This avoids
changing the meaning of `wataru_proxy_field` and keeps production behavior
stable.

Expected updates:

- [ ] Extend `FiniteCurrentMode`.
- [ ] Extend `EffectiveCurrentMode` if needed.
- [ ] Add mode-specific current convention metadata.
- [ ] Keep `DEFAULT_FINITE_CURRENT_MODE = "wataru_proxy_field"`.
- [ ] Add CLI choices for the new mode where Stage 2 accepts finite current.

### 2. Centralize compatibility logic

Create one compatibility module rather than scattering historical rules across
the Stage 2 solver, geometry helpers, and handoff code.

Proposed file:

`examples/single_stage_optimization/banana_opt/jhalpern30_compat.py`

Responsibilities:

- [ ] Historical VF template path.
- [ ] Historical VF template hash/count validation helper.
- [ ] Proxy placement builder.
- [ ] VF coil builder.
- [ ] Signed proxy/VF current convention validation.
- [ ] Boozer `I = mu0 * proxy_current_A` policy labeling.
- [ ] Signed-G preservation checks so jhalpern mode cannot silently switch to
      negative-abs G0.
- [ ] Banana sign and iota sign replay helpers.
- [ ] StageNN state parsing helpers.

### 3. Keep signed G explicit

Current repo policy:

```text
G0 = mu0 * sum(I_tf)
```

Do not adopt the jhalpern30 fresh-run policy:

```text
G0 = -mu0 * sum(abs(I_tf))
```

Boozer finite-current policy:

```text
I  = mu0 * proxy_current_A
```

Implementation requirement:

- [ ] Do not change `compute_tf_G0(tf_coils)` to the jhalpern negative-abs
      formula.
- [ ] Existing hot/probe paths call
      `derive_signed_G_from_field(bs, tf_coils=...)`; jhalpern mode must keep
      that BiotSavart/TF-subset validation.
- [ ] If a historical `state.json` provides saved `G`, treat it as replay
      provenance only. Do not use it to redefine the fresh-run G policy unless
      the caller explicitly requests a saved Boozer-surface resume path.
- [ ] Tests must cover positive-TF, negative-TF, and mixed-sign TF inputs so the
      behavior cannot be mistaken for the signed general policy.
- [ ] Tests must prove Boozer `I` is `mu0 * proxy_current_A` and is not folded
      into `G0`.

## Detailed Implementation Plan

### Phase 0: Baseline probes

- [ ] Record the current behavior of `wataru_proxy_field` before edits.
- [ ] Write a small local probe that loads the historical 20-coil VF template and
      records:
      - coil count
      - template current signs
      - object types for current wrappers
      - initial effective currents for positive, negative, and zero proxy current
- [ ] Probe `jhalpern30/singlestage.py` and `jhalpern30/stage2.py` construction
      directly, then use those numbers as expected values for compatibility
      tests.
- [ ] Confirm the exact fresh-run versus resume G0 path:
      - fresh current-repo run: explicit signed TF-current sum
      - historical Boozer-surface resume: saved `G` from state
      - no adoption of jhalpern's fresh-run negative-abs TF-current sum
- [ ] Confirm the exact Boozer `I` path:
      - `BOOZER_I_PARAM = mu0 * PROXY_CURRENT_KA * 1e3`
      - passed into `BoozerSurface(..., I=...)`
- [ ] Confirm banana sign inputs:
      - `--flip-banana` in stage 2
      - `_flip` parent-directory suffix in single-stage replay
      - `BANANA_I_FIXED_S2`
      - `IOTA_TARGET_SIGN`
      - unset or empty `BANANA_I_FIXED_S2` means free banana-current DOF
      - numeric `BANANA_I_FIXED_S2` means fixed banana-current DOF after
        applying `BANANA_CURRENT_SIGN`

### Phase 1: Mode and metadata plumbing

- [ ] Add `jhalpern30_proxy_field` to finite-current mode literals.
- [ ] Add a current-convention descriptor for the mode.
- [ ] Reuse existing artifact metadata and add only missing replay fields:
      - `FINITE_CURRENT_MODE`
      - `BOOZER_CURRENT_CONVENTION`
      - `PROXY_PLACEMENT_MODE`
      - `VF_TEMPLATE_PATH`
      - `VF_TEMPLATE_SHA256`
      - `NUM_PROXY_COILS`
      - `NUM_VF_COILS`
      - `TOTAL_COILS`
      - `PROXY_PLASMA_CURRENT_A`
      - `VF_CURRENT_A`
      - `VF_CURRENT_SIGN_POLICY`
      - `VF_CURRENT_MUTABILITY`
      - `FLIP_BANANA`
      - `BANANA_CURRENT_SIGN`
      - `BANANA_CURRENT_PINNED`
      - `BANANA_I_FIXED_S2_KA`
      - `IOTA_TARGET_SIGN`
      - `G0_POLICY = signed_explicit_tf_current`
- [ ] Ensure the jhalpern importer/writer emits the existing `COIL_GROUPS`
      manifest plus `NUM_*` legacy counts.
- [ ] Add partition tests for
      `20 TF + 10 banana + 1 proxy + 20 VF = 51` total coils; the current
      partition/load path is already metadata-driven, so the task is to
      preserve and extend that path rather than replace it.
- [ ] Add schema/version notes if current artifact metadata has a version field.

### Phase 2: Bundle the historical VF template

- [ ] Copy the historical template to:
      `examples/single_stage_optimization/banana_opt/jhalpern30_vf_biotsavart.json`
- [ ] Add a resolver for the jhalpern template.
- [ ] Add tests that verify:
      - the bundled file loads with Simsopt
      - the file has 20 coils
      - the signs match the historical template
      - the sha256 matches the recorded historical hash unless the file is
        deliberately regenerated and the plan/doc is updated
- [ ] Do not use the user's absolute `banana_drivers-main` path at runtime.

### Phase 3: Proxy and VF builders

- [ ] Add `build_jhalpern30_proxy_plasma_current_coils(surface, proxy_current_A)`.
      - Uses `surface.major_radius()`.
      - Sets `Z = 0`.
      - Builds exactly one proxy coil.
      - Keeps proxy current fixed, matching jhalpern.
- [ ] Add `build_jhalpern30_vf_coils(proxy_current_A, template_path)`.
      - Loads the 20-coil template.
      - Computes the signed scale `VF_CURRENT_A = proxy_current_A / 6.5`.
      - Reproduces jhalpern sign algebra exactly.
      - For nonzero proxy current, effective per-coil current is
        `abs(proxy_current_A) / 6.5 * sign(template_current)`.
      - Uses shared `ScaledCurrent(Current(1.0), VF_CURRENT_A)`.
      - Calls `unfix_all()` on VF currents.
      - Fixes VF curves.
- [ ] Add mode dispatch in coil initialization:
      - Wataru mode keeps VMEC-axis proxy placement and fixed VF currents.
      - jhalpern mode uses historical placement and mutable VF currents.
- [ ] Add tests comparing the generated coil layout against a direct
      jhalpern-style construction.

### Phase 4: Current validation and CLI behavior

- [ ] Keep `validate_hbt_proxy_vf_current_convention` unchanged for Wataru mode.
- [ ] Add `validate_jhalpern30_proxy_vf_current_convention`.
      - Allows signed `proxy_current_A`.
      - Requires stored/scalar `vf_current_A == proxy_current_A / 6.5` when the
        caller supplies a VF current.
      - Documents that effective VF coil currents follow the template signs and
        are not simply the signed scalar repeated across all VF coils.
      - Allows zero proxy current explicitly.
      - Rejects inconsistent proxy/VF signs only in this mode's own terms.
- [ ] Update `_resolve_stage2_finite_current_config` to dispatch by mode.
- [ ] Update `validate_stage2_seed_contract` and related metadata validators to
      dispatch by mode instead of always calling the Wataru nonnegative helper.
- [ ] Add CLI support:
      - `--finite-current-mode jhalpern30_proxy_field`
      - optional jhalpern VF template override for test/probe use
      - no silent fallback if an explicit template path is invalid
- [ ] Make the chosen mode visible in logs and artifact metadata.

### Phase 5: G0 non-adoption guardrails

- [ ] Keep fresh jhalpern-compat runs on the current repo signed-G path:
      `derive_signed_G_from_field(bs, tf_coils=...)`.
- [ ] Keep the field-aware validation that proves the TF subset belongs to the
      same `BiotSavart` field.
- [ ] Route resumed jhalpern Boozer surfaces through saved state `G`.
- [ ] Keep existing signed-G helpers for normal production routes.
- [ ] Thread the non-adoption guardrails through live call sites:
      - `banana_opt.stage2_objectives.build_stage2_iota_runtime`
      - `banana_opt.stage2_single_stage_handoff.probe_stage2_seed_bootability`
        / `_probe_initialization_inputs`
      - the StageNN importer
- [ ] Add tests:
      - signed-G helper returns `mu0 * sum(I_tf)`.
      - jhalpern mode still returns `mu0 * sum(I_tf)` for fresh runs.
      - no fresh-run path computes `-mu0 * sum(abs(I_tf))`.
      - resume path uses saved `G` even if TF coil currents would imply a
        different fresh-run value.
      - Boozer `I` remains `mu0 * proxy_current_A`.

### Phase 6: Downstream consumers

- [ ] Update or explicitly bypass
      `SINGLE_STAGE/single_stage_banana_example.py`.
      - Its CLI currently accepts only `boozer_surrogate` and
        `wataru_proxy_field`.
      - Its single-surface current resolver is locked to the Wataru default.
- [ ] Update or explicitly bypass `run_stage2_to_single_stage.py`.
      - It currently hard-requires `FINITE_CURRENT_MODE='wataru_proxy_field'`
        for pre-Boozer repair.
      - If jhalpern replay should not use this wrapper, add a clear rejection
        test and document the supported path.
      - If it should use this wrapper, update `build_recovery_command` and its
        tests for `jhalpern30_proxy_field`.
- [ ] Make template resolution mode-aware in:
      - `workflow_helpers.py`
      - `workflow_runner_common.py`
      - `Stage2SeedSpec`
      - run-directory suffixing / provenance helpers
- [ ] Add consumer tests proving a jhalpern artifact can reach the intended
      single-stage replay path, or proving that unsupported wrappers reject it
      loudly with a mode-specific message.

### Phase 7: StageNN handoff importer

- [ ] Add an explicit importer for historical bundles:
      - input directory contains `stageNN/bsurf_opt.json`
      - input directory contains `stageNN/state.json`
      - `stage00` is the primary handoff case, but the importer should not bake
        in stage 0 if the state file advertises another `stage_idx`.
- [ ] Validate required state fields:
      - `iota`
      - `G`
      - `volume`
      - `iota_target`
      - `stage_idx`
      - `stage_mpol`
      - `stage_ntor`
      - `stage_order`
      - `stage_qp`
- [ ] Distinguish historical input routes:
      - raw `biot_savart_opt.json` import
      - warm-start reconstruction when proxy/VF must be rebuilt
      - Boozer-surface resume from `stageNN/`
- [ ] Emit one canonical current-repo artifact bundle, not an ambiguous in-memory
      alternative:
      - `biot_savart_opt.json`
      - `results.json`
      - `COIL_GROUPS`
      - `NUM_TF_COILS`
      - `NUM_BANANA_COILS`
      - `NUM_PROXY_COILS`
      - `NUM_VF_COILS`
      - `FINITE_CURRENT_MODE`
      - `BOOZER_CURRENT_CONVENTION`
      - `PROXY_PLASMA_CURRENT_A`
      - `VF_CURRENT_A`
      - `VF_TEMPLATE_PATH`
      - `VF_TEMPLATE_SHA256`
      - `STAGE2_BS_SHA256`
      - resume-surface and state provenance
- [ ] Fail loudly on missing files, unsupported JSON object types, or mode
      mismatch.
- [ ] Document the importer command once implemented.

### Phase 8: Tests and validation

- [ ] Unit tests for finite-current mode parsing and metadata.
- [ ] Unit tests for the bundled VF template.
- [ ] Unit tests for proxy placement.
- [ ] Unit tests for mutable jhalpern VF currents.
      - all 20 VF coils share the same unwrapped leaf `Current`
      - the leaf current is unfixed
      - VF curves are fixed
      - changing the shared leaf DOF updates all effective VF currents with the
        expected template signs
- [ ] Unit tests for signed proxy current validation.
- [ ] Unit tests for G0 non-adoption.
- [ ] Unit tests for Boozer `I = mu0 * proxy_current_A`.
- [ ] Unit tests for banana sign and iota target sign replay.
- [ ] Integration test for jhalpern mode coil layout:
      `20 TF + 10 banana + 1 proxy + 20 VF = 51` total coils.
- [ ] Seeded restart test using StageNN-style state.
- [ ] Importer test that materializes canonical `biot_savart_opt.json` plus
      `results.json`.
- [ ] Regression test that Wataru mode still rejects negative proxy/VF current.
- [ ] Regression test that Wataru mode still uses the 2-coil default VF template.
- [ ] Downstream-consumer tests for single-stage and repair wrapper acceptance or
      explicit rejection.

Suggested focused commands after Phase 8 adds `tests/geo/test_jhalpern30_compat.py`:

```bash
PYTHONPATH=examples/single_stage_optimization python -m pytest \
  tests/geo/test_banana_helper_modules.py \
  tests/geo/test_stage2_single_stage_handoff.py \
  tests/geo/test_stage2_seeded_restart_vf_consistency.py \
  tests/geo/test_wataru_vf_template_resolution.py \
  tests/geo/test_jhalpern30_compat.py \
  -q

PYTHONPATH=examples/single_stage_optimization python -m pytest \
  tests/geo/test_single_stage_example.py \
  tests/geo/test_single_stage_alm_integration.py \
  -q

python -m ruff check examples/single_stage_optimization tests/geo
git diff --check
```

## Done Criteria

- [ ] `wataru_proxy_field` tests pass unchanged.
- [ ] `jhalpern30_proxy_field` builds the historical proxy and VF layout.
- [ ] `jhalpern30_proxy_field` emits
      `20 TF + 10 banana + 1 proxy + 20 VF = 51` total coils.
- [ ] The bundled VF template has 20 coils and matches the historical template.
- [ ] Negative proxy current is accepted only in jhalpern mode.
- [ ] VF currents are mutable only in jhalpern mode.
- [ ] Fresh jhalpern-compat replay uses this repo's explicit signed G:
      `mu0 * sum(I_tf)`.
- [ ] Fresh jhalpern replay passes `BoozerSurface I = mu0 * proxy_current_A`.
- [ ] Resume jhalpern replay uses saved `G`.
- [ ] Banana sign, `_flip`, and iota-target sign semantics are explicit and tested.
- [ ] StageNN handoff import is explicit, validated, and documented.
- [ ] Importer emits a canonical current-repo artifact bundle with `COIL_GROUPS`.
- [ ] Artifact metadata is sufficient to distinguish Wataru and jhalpern runs.
- [ ] No runtime dependency on `/Users/suhjungdae/code/columbia/banana_drivers-main`.

## Risks and Open Questions

- [ ] VF current sign algebra is easy to simplify incorrectly. The implementation
      must match the object graph and effective initial currents from jhalpern,
      not just the apparent formula.
- [ ] Current partition/load code is already metadata-driven via `COIL_GROUPS`
      and `NUM_*`. The risk is failing to emit correct jhalpern metadata, not
      needing to redesign partitioning from scratch.
- [ ] Some historical Boozer JSON files may require Simsopt forks with
      `BoozerSurface(I=...)` compatibility. The importer should report this as
      an unsupported object/version problem, not silently rebuild a different
      seed.
- [ ] The jhalpern negative-abs G0 policy is intentionally not adopted. The
      implementation must not reintroduce it while adding other jhalpern
      compatibility pieces.
- [ ] Pre-Boozer repair and single-stage wrappers currently have Wataru-only
      assumptions. The implementation must choose acceptance or explicit
      rejection; leaving them implicit would produce confusing handoff failures.
- [ ] If the historical 20-coil template is large, keep it as a data artifact
      with a hash test instead of regenerating it from code.
