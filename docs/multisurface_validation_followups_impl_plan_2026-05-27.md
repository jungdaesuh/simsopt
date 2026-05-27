# Multi-Surface Banana-Coil Validation — Follow-up Patch/Update Plan

## Purpose

Capture the actionable patch/update items surfaced by the 2026-05-27 GPD-style
validation of the multi-surface Boozer-surface implementation for banana-coil
single-stage optimization. The validation verdict was: **the `published_multisurface`
path is implemented correctly** (signed-G, vacuum G-constancy, outer→inner
continuation, objective split, weighted-mean aggregation/gradients all verified;
162 targeted tests green). This file tracks the *non-blocking* findings worth
acting on, so they are not lost.

This is a remediation/cleanup plan, **not** a correctness-blocker list. Nothing
here invalidates the published multi-surface path.

## Goals

- Resolve the one latent code-behavior inconsistency (experimental-mode surface
  weights vs. the descended objective) with an explicit decision + patch.
- Make the unconstrained interior-iota/shear behavior at least *observable*.
- Convert two fail-closed runtime postconditions into earlier/explicit guards.
- Clear three minor cleanup items (DRY, typing, dynamic import) if still applicable.

## Non-Goals

- Re-architecting the surface-mode contract or the continuation strategy (they are correct).
- Changing published-mode physics or weights (fixed `(1,1,1)` is intended and correct).
- Touching the signed-G SSOT formula (`G = μ₀·Σ_signed(I_TF)` is verified correct).
- Fixing/serializing the KAM / frontier / topology work being edited concurrently
  by other sessions (out of scope; see Current Context).

## Current Context

- Repo: `/Users/suhjungdae/code/columbia/simsopt-surrogate`, branch `surrogate-confinement-v2`.
- Canonical multi-surface mode is `PUBLISHED_MULTISURFACE` (label fractions `0.6/0.8/1.0`,
  fixed weights `(1,1,1)`), defined in
  `examples/single_stage_optimization/banana_opt/surface_mode_contracts.py`.
- Objective wiring lives in `SINGLE_STAGE/single_stage_banana_example.py`
  (`build_single_stage_objective_bundle`, `build_boozer_derived_objective_terms`,
  `evaluate_total_objective`) + `banana_opt/single_stage_objectives.py`
  (`average_surface_objectives`, `_resolve_surface_objective_terms`, `build_total_objective`).
- Continuation init: `initialize_published_surface_data_from_stage2_seed`,
  `contract_surface_to_target_volume`, `_fit_contracted_surface` in
  `SINGLE_STAGE/single_stage_banana_example.py`.
- Signed-G SSOT: `banana_opt/boozer_finite_current.py::derive_signed_G_from_field`.
- **Canonical interpreter: `.conda-env/bin/python` (Python 3.11.15).** The compiled
  `simsoptpp` extension is built against this env. Ambient `python` is
  non-deterministic across sessions (observed 3.11/3.13/3.14 in different shells) and
  a mismatched ambient interpreter (e.g., 3.14) fails to import `simsoptpp.Curve`.
  Always invoke `.conda-env/bin/python` for compile/test, not bare `python`.
- **⚠ Concurrency hazard (verified 2026-05-27 ~15:13):** ~9 other
  `claude --dangerously-skip-permissions` sessions + an autoresearch stage1 sweep
  are editing this same checkout live. During this session the working tree grew
  from 9 to 23 modified files, an untracked `banana_opt/vf_coils.py` was
  relocated/removed, and new `banana_opt/topology/` + `tests/geo/test_kam_birkhoff.py`
  appeared (mtimes 15:00–15:10). **Line numbers drift; anchor edits to function/symbol
  names and re-confirm before patching.**

## Rationale

The validation found no correctness defect in the published multi-surface path, so
this work is about (a) removing a confusing dead-parameter path that becomes a real
divergence in the *experimental* mode, (b) improving physical observability of an
intentionally-unconstrained quantity, and (c) turning "caught late and aborts" into
"caught early or impossible." These reduce future debugging cost without changing
shipped physics.

## Assumptions

- `published_multisurface` remains vacuum-locked (`boozer_I=0` via
  `current_contracts.py::resolve_plasma_current_settings_for_surface_mode`) and uses
  fixed `(1,1,1)` weights. (Verified true today.)
- The production search path always passes the uniform global `JNonQSObjective`/
  `JBoozerObjective` into `evaluate_total_objective` (verified:
  `resolve_current_surface_objective_terms` → `evaluate_total_objective` call sites).
- `experimental_multisurface` is still considered experimental (lower bar than published).
- A clean baseline of the touched files can be obtained despite concurrent edits
  (may require coordinating with / pausing the other sessions).

## Implementation Plan

1. **[Tier 1 — DECIDED 2026-05-27: option (a), gate/diagnostics-only] Experimental-mode surface-weight ramp**
   - Context: `_resolve_surface_objective_terms` (`single_stage_objectives.py`)
     computes `raw_*` from `surface_weights` but then returns
     `objective_* = raw_* if JxxxObjective is None else JxxxObjective`. The production
     path always supplies the uniform global (`resolve_current_surface_objective_terms`),
     so the optimizer descends a **uniform-weighted** multi-surface objective. The
     continuation ramp (`continuation_inner_surface_weight` →
     `build_surface_search_weights`) reaches only diagnostics + the acceptance gate
     (`build_surface_search_gate`: gap-threshold scaling + `enforce_nesting` toggle).
     Inert for published `(1,1,1)`; a real silent divergence for the experimental ramp.
   - **Decision: (a) gate/diagnostics-only — do NOT thread the ramp into the objective.**
     Rationale: the ramp is a function of `accepted_iterations` (`single_stage_geometry.py:963`),
     so wiring it into the descended objective would minimize a *moving objective*, breaking
     the LBFGS-B/ALM line search + quasi-Newton Hessian and making the solution path-dependent.
     The acceptance gate already handles the real failure mode (a not-yet-resolved inner surface
     rejecting steps); both options converge to the same full-strength (weight→1) objective; and
     `test_single_stage_alm_surface_stack_gate_relaxes_spacing_only_for_solver`
     confirms the relaxation is intended to be solver/gate-scoped. The objective-weight plumbing
     is vestigial (severed by the `JNonQSObjective`/`JBoozerObjective` refactor).
   - [ ] Document `_resolve_surface_objective_terms`, `build_surface_search_weights`, and `evaluate_total_objective`: the objective uses uniform surface weights; the continuation ramp governs the acceptance gate + diagnostics only.
   - [ ] Rename the objective-path `surface_weights` arg to `diagnostic_surface_weights` (or drop it from the objective path) so it no longer implies it weights the descended objective; leave the gate wiring (`build_surface_search_gate`) unchanged.
   - [ ] ~~Option (b): thread ramp weights into the descended objective~~ — **rejected** (moving-objective hazard; see Decision).
   - [ ] Add a regression test asserting experimental non-uniform ramp weights **provably do not** change `evaluate_total_objective["total"]` (they affect only `gate_scale` + diagnostics).
   - [ ] (Deferred — only if profiling shows a poorly-resolved inner surface dominating the gradient) consider a *fixed, bounded* inner weight applied once at bundle build (contract `weights` field): a static reweighting with no moving-objective problem, NOT the per-iteration ramp.

2. **[Tier 2 — physics observability] Interior iota / shear is unconstrained**
   - Context: `Jiota` uses `surface_iota_terms[-1]` (outer only); `JVolume` uses
     `Volume(surface_data[-1])`. QS + Boozer residual aggregate across all surfaces,
     but the iota *profile* / magnetic shear is free, so an interior surface could sit
     on a low-order rational without an explicit penalty.
   - [x] ~~Record per-surface solved iota in the run metadata~~ — **already done** (verified 2026-05-27): `collect_surface_run_metadata` (`single_stage_geometry.py:1329`) serializes `FINAL_SURFACE_IOTAS` (`:1345`) and `INITIAL_SURFACE_IOTAS` (`:1354`). No work needed.
   - [ ] Add a cheap diagnostic flag when any interior surface's solved iota (from the already-serialized `FINAL_SURFACE_IOTAS`) lands within a tolerance of a low-order rational `n/m` (m ≤ configurable bound). **This is the genuinely-new part of Tier 2.**
   - [ ] (Optional, gated decision) add an opt-in interior-iota or shear penalty term to the objective bundle, default-off to preserve current published behavior.

3. **[Tier 3 — robustness] Flux-seed / volume-target ordering + volume sign**
   - Context: `build_surface_configs_for_contract` seeds by flux fraction but solves
     to wout-volume-ratio targets. Volume **ordering is already guarded** — both
     pre-solve (`build_surface_configs_for_contract` derived-target check,
     `single_stage_geometry.py:187-199`) and post-solve (`_require_published_volume_order`,
     `evaluate_surface_stack`), plus `contract_surface_to_target_volume`'s
     `0 < target < previous` bracket guard. The real gap is the volume **sign**:
     `Surface.volume()` is sign-dependent on θ-orientation and the path assumes a
     positive value without an explicit guard.
   - [ ] **Real remaining gap:** add an explicit `volume > 0` assertion (or θ-orientation sign-normalization) at the entry of `contract_surface_to_target_volume` and `_require_published_volume_order` with a clear error message. Ordering is guarded (below) but volume *sign* is not.
   - [x] ~~Pre-solve target-volume ordering check~~ — **already done** (verified 2026-05-27): `build_surface_configs_for_contract` (`single_stage_geometry.py:187-199`) raises if derived target volumes are not strictly ordered inner→outer; `_require_published_volume_order` re-checks solved volumes post-solve. The earlier "verify coil-field V(s) monotonicity" item was redundant with this and is dropped.

4. **[Tier 4 — cleanup] Minor consistency/style items**
   - [ ] DRY: evaluate replacing `_fit_contracted_surface` with the native C++ `Surface.scale()` (`src/simsoptpp/surface.cpp`); **first verify** it does not mutate the source surface and reproduces volumes within the bisection tolerance, then swap.
   - [ ] Tighten `VFCoilBuildResult.coils` typing from `list[object]` to `list[Coil]` **iff** the relocated `vf_coils.py` still exists and the import-light constraint allows it (re-locate the symbol first — the file moved during concurrent edits).
   - [ ] Hoist the function-local `from .finite_current_profiles import FINITE_CURRENT_PROFILES` in `artifact_contracts.py::_upgrade_legacy_finite_current_metadata` to module scope **iff** the `current_contracts ↔ finite_current_profiles` import cycle can be broken at the source; otherwise add a comment justifying the deliberate cycle-break.

## Validation Plan

- [ ] Fast compile loop (AGENTS.md): `.conda-env/bin/python -m py_compile` on every edited module.
- [ ] Targeted regression (must stay green — baseline **162 passed, 260 deselected**; the deselect count drifts as concurrent sessions add tests):
      `.conda-env/bin/python -m pytest tests/geo/test_surface_mode_contracts.py tests/geo/test_single_stage_example.py tests/geo/test_finite_current_profiles.py -k "published or multisurface or continuation or surface_stack or surface_config or surface_mode or single_surface or evaluate_surface_stack or contract" -q`
- [ ] Tier 1: new test pins experimental non-uniform-weight behavior to the documented/implemented semantics.
- [ ] Tier 3: new unit test that a non-positive / mis-ordered seed volume is rejected with the new explicit guard (not the late postcondition).
- [ ] Import smoke test of all touched modules (`.conda-env/bin/python -c "import ..."`).
- [ ] Published-mode parity spot check: confirm `evaluate_total_objective["total"]` for a `published_multisurface` fixture is unchanged by the patches (proves inertness for the shipped path).

## Risks and Mitigations

- Risk: **Concurrent sessions are editing the same checkout**, so edits may clobber or be clobbered, and line anchors drift.
  Mitigation: do this work on a dedicated branch created from the *current* state of each target file; anchor edits to function/symbol names; re-run `git status`/`git diff` immediately before each edit; coordinate timing with the other sessions (esp. the KAM/frontier/VF work).
- Risk: Threading ramp weights into the objective (Tier 1 option b) changes experimental-mode optimization trajectories.
  Mitigation: gate behind the explicit decision; keep published path byte-for-byte unchanged; add the parity spot check.
- Risk: Swapping in native `Surface.scale()` (Tier 4) silently mutates the source surface or changes volumes.
  Mitigation: verify non-mutation + volume parity in a scratch script before replacing; keep the change behind its own commit.
- Risk: `vf_coils.py` typing task targets a file that has already moved/changed.
  Mitigation: re-locate `VFCoilBuildResult` before editing; treat the task as possibly-obsolete.

## Completion Criteria

- [x] Tier 1 decision recorded — **option (a), gate/diagnostics-only** (2026-05-27).
- [ ] Tier 1 implemented: doc + objective-path arg rename + regression test proving ramp weights do not change `evaluate_total_objective["total"]`.
- [ ] Interior-iota diagnostic emitted in run metadata (Tier 2 minimum).
- [ ] Explicit positive/ordered volume guard added with a rejecting test (Tier 3).
- [ ] Tier 4 items either done or explicitly closed as obsolete/not-worth-it with a one-line reason.
- [ ] Targeted test suite still green; published-mode objective parity confirmed.

## Open Questions

- ~~Is the experimental `continuation_inner_surface_weight` ramp objective-biasing or gate/diagnostics-only?~~ **Resolved 2026-05-27: gate/diagnostics-only (Tier 1 option a).** See the Tier 1 Decision; reopen only if profiling shows inner-surface gradient domination (then a *static* weight, not the ramp).
- Should interior iota be an actual **constraint/penalty** or only a **diagnostic**? (Physics-design call; affects whether Tier 2 step 3 is in scope.)
- Did the concurrent VF-current work **rename/relocate `vf_coils.py` / `VFCoilBuildResult`** permanently? (Confirm before Tier 4 typing task.)
