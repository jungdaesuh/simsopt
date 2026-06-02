# Remaining-Items Implementation Plan (2026-06-02)

## Purpose

Close out the unfinished work across the three active planning docs so each
reaches "all items done." This plan is the execution artifact derived from the
2026-06-02 plan-conformance audit of:

- `docs/bloat_reduction_plan_2026-05-20.md` (doc 1)
- `docs/bloat_torax_coherent_execution_plan_2026-05-31.md` (doc 2)
- `docs/torax_jax_porting_patterns_impl_plan_2026-05-27.md` (doc 3)

Audit result (110 items): **78 DONE, 15 PARTIAL, 17 MISSING**;
**19 required items unfinished**. The code refactors landed; what remains is
**release-blocker validation closeout** (tier-exit gates + accelerator-lane
proof), doc bookkeeping, and doc-3 Phases 4–5.

## Goals

- Run and record the three bloat-plan **tier-exit gates** and create the
  `bloat-reduction-T{1,2,3}-complete` tags.
- Produce **accelerator-lane evidence** (strict-transfer CUDA/MPS proof +
  Stage 2 / single-stage parity replay) currently missing from every slice.
- Complete doc-2 **preflight + numerical/parity closeout** bookkeeping items.
- Complete doc-3 **Phase 4 (branch discipline)** and **Phase 5 (numerical
  shape/stability audit)**.
- Update **CLAUDE.md / user docs** for the outer-optimizer lanes and the stale
  backend-mode table.
- Re-run both audits and reach `converged: true` (zero required items unfinished).

## Non-Goals

- No new LOC-reduction refactors beyond what a tier-exit LOC shortfall forces
  (see ASSUMP-1 / OQ-7); T3.3/T3.5/T3.7 stay closed as NO-BANK decisions.
- No new runtime features; this is closeout + proof, not implementation.
- No relaxation of the strict CPU↔C++ byte-identity gate or parity tolerances.
- No editing of `MEMORY.md` unless OQ-6 is explicitly answered "yes."

## Current Context

- Branch: `shared-jax-clean`. Validation lane: in-tree `.conda/jax` (JAX
  0.10.0 / jaxlib 0.10.0) per `CLAUDE.md` → "Validation".
- **No `bloat-reduction-*` git tags exist** (`git tag -l "*bloat*"` empty) —
  all three tier-exit gates are unfired.
- **No accelerator-lane proof exists anywhere**: the only
  `transfer_guard("disallow")` tests run on CPU (host==device); doc 2 line 62
  states "Strict-transfer and full accelerator replay gates are not claimed by
  this pass." The audit host is Apple Silicon (Metal-only, no local CUDA).
- The release-blocker byte-identity gate is
  `_pre_newton_census_gate_failures` at
  `benchmarks/single_stage_init_parity.py:3284` (CLAUDE.md → "Parity modes").
- `CLAUDE.md:242` still says `VALID_OPTIMIZER_BACKENDS = {"scipy", "ondevice"}`
  and contains **no** `scipy-jax` / `scipy-jax-fullgraph` outer-lane text.
- Doc-3 Phases 1–3 are DONE with real tests; Phase 4 is a 2-contract pilot;
  Phase 5 is entirely unstarted (all 11 boxes open, no evidence note).
- `docs/using_jax_backend.md` exists (21 KB, last touched 2026-05-31) and its
  backend-mode table is flagged stale by doc 3 line 238.

## Rationale

The remaining work clusters into four dependency-ordered workstreams. **WS-A
(decisions)** comes first because choosing a CUDA/MPS venue (OQ-3) unblocks the
entire accelerator-lane workstream, and the T1 LOC decision (OQ-7) gates the T1
tag. **WS-B (accelerator proof)** and **WS-E (doc-3 phases)** touch disjoint
code and run in parallel. **WS-C (tier exits)** depends on the LOC decision and
shares the parity-replay artifacts with WS-B. **WS-D (bookkeeping)** is mostly
independent, except the final grouped tier-completion validation
(`CEP-IMPL-7.1`) must run last.

Tags + grouped replays are mechanical; the genuinely new engineering is doc-3
Phase 5 (numerical-stability audit) and the accelerator-lane proof on real
hardware.

## Assumptions

- **ASSUMP-1:** The T1 LOC target (≥600 net) is a hard acceptance criterion.
  Currently ≈400 net LOC is banked (T1.1=217, T1.2≈56, T1.3≈−41, …), so the
  T1 tag cannot be issued without either banking ~200+ more net LOC in T1 scope
  or recording an explicit target revision (→ OQ-7).
- **ASSUMP-2:** A CUDA host with the repo's CUDA/JAX env is reachable for the
  strict-transfer proof. If only Apple MPS is available, the `jax_mps_smoke`
  lane (python≥3.13, `pip install '.[JAX_MPS]'`) is the fallback and must be
  labeled as smoke, not CUDA parity.
- **ASSUMP-3:** `linearization_residency="device"` is required on the strict
  CUDA lane (CLAUDE.md → "Linearization residency / strict-transfer contract");
  `"host"` fails by design under `transfer_guard("disallow")` on an accelerator.
- **ASSUMP-4:** The working-tree docs are the source of truth the audit ran
  against; tasks below edit those same files.

## Implementation Plan

### WS-A — Decisions / unblock (do first)

1. Resolve the open questions that gate downstream work.
   - [ ] **OQ-3 / `BP-OQ-3` / `CEP-OQ-3` (doc1:828, doc2:2078):** choose the
     CUDA/MPS proof venue (self-hosted GitHub CUDA runner, Perlmutter, Runpod,
     or local MPS smoke). Record the choice in doc 1 §12 and doc 2 Open
     Questions. **Blocks WS-B.**
   - [ ] **OQ-7 / T1 LOC (doc1:444):** decide T1-exit LOC posture — bank ≥200
     more net LOC in T1 scope, OR revise/justify the ≥600 target in-doc.
     **Blocks `T1-exit`.**
   - [ ] **`BP-OQ-1` (doc1:826):** answer branch strategy (stay on
     `gpu-purity-stage2-20260405`/current vs. new `bloat-reduction-20260520`).
   - [ ] **`BP-OQ-2` (doc1:827):** answer time horizon (sprint/background/ad hoc).
   - [ ] **`BP-OQ-6` (doc1:831):** decide whether to record an OpenMemory/project
     note for the T1.4 curve↔jax_core import-cycle story (do **not** touch
     `MEMORY.md` unless answered yes).
   - [ ] **`BP-OQ-5` / `T4.2` CLAUDE.md edits (doc1:830, 686):** confirm the 3
     CLAUDE.md edits are in scope now (folded into WS-D task 13) or explicitly
     deferred in-doc.

### WS-B — Accelerator-lane proof (needs OQ-3)

2. Strict-transfer proof on the chosen accelerator lane.
   - [ ] Run a `jax.transfer_guard("disallow")` proof on the affected
     `jax_gpu_*` (CUDA) or `jax_mps_smoke` lane for transfer-sensitive surfaces
     (Biot-Savart, BoozerSurface LS factors, single-stage adjoint). Record the
     exact command + output. Satisfies `CEP-IMPL-3.3` (doc2:142) and
     `CEP-VAL-6` (doc2:176).
   - [ ] Confirm `linearization_residency="device"` on the CUDA lane (ASSUMP-3);
     assert no device→host transfer escapes the guard.
   - [ ] Tick doc2:142 and doc2:176; update the per-slice "transfer-sensitive
     proof remains open" notes that now have evidence.
3. Stage 2 + single-stage parity replay (release blocker).
   - [ ] Replay the Stage 2 / single-stage parity gates named in the bloat plan:
     `tests/integration/test_stage2_jax.py`,
     `tests/integration/test_single_stage_jax_cpu_reference.py`, and the
     `_pre_newton_census_gate_failures` byte-identity gate on pinned input.
     Satisfies `CEP-VAL-7` (doc2:177) and `CEP-IMPL-7.2` (doc2:165).
   - [ ] Record results; tick doc2:165, doc2:177.
   - [ ] Update `BP-SC` sub-criterion #7 (GPU/strict-transfer proof) once WS-B
     tasks 2–3 produce real accelerator evidence (doc1:93–94 / §2).

### WS-C — Tier-exit gates (needs OQ-7 for T1)

4. **`T1-exit`** (doc1:444).
   - [ ] Satisfy ASSUMP-1 per OQ-7 (bank ≥200 net LOC or record target revision).
   - [ ] Run the full T1 suite green; re-affirm the contract checklist.
   - [ ] `git tag bloat-reduction-T1-complete` at the validated commit.
5. **`T2-exit`** (doc1:578).
   - [ ] Re-run the T2 suite + contract checklist + `_pre_newton_census_gate_failures`
     replay gate **as one grouped tier-exit proof**.
   - [ ] `git tag bloat-reduction-T2-complete`.
6. **`T3-exit`** (doc1:667).
   - [ ] Run grouped closure validation + one adversarial review + contract
     checklist re-affirmation against the final tree.
   - [ ] `git tag bloat-reduction-T3-complete`.

### WS-D — Closeout & preflight bookkeeping (doc 2 + doc 1)

7. Preflight evidence lock (doc 2 §1).
   - [ ] `CEP-IMPL-1.1` (doc2:127): paste literal `git status --short` output
     into the preflight evidence block; tick the box.
   - [ ] `CEP-IMPL-1.2` (doc2:128): tick the TORAX checkout/HEAD preflight box
     (HEAD `60190df1` already verified) and record it as a completed step.
   - [ ] `CEP-IMPL-1.3` (doc2:129): add one consolidated path-check
     re-verification covering all referenced paths; tick the box.
8. Branch/optimizer decisions (doc 2 §6).
   - [ ] `CEP-IMPL-6.1` (doc2:158): classify the third category — **explicit
     host-boundary work** — for the branch sites (doc3:182 dependency); extend
     beyond the 2-site pilot.
   - [ ] `CEP-IMPL-6.4` (doc2:161): re-affirm branch-semantics tests pass and
     mark the guardrail satisfied for §6 scope.
9. Numerical/parity closeout (doc 2 §7).
   - [ ] `CEP-IMPL-7.2` (doc2:165): covered by WS-B task 3.
   - [ ] `CEP-IMPL-7.3` (doc2:166): tick the "evidence-backed status changes"
     process box once all status changes above carry evidence.
   - [ ] `CEP-IMPL-7.1` (doc2:164): **run last** — the grouped/full
     tier-completion validation gate; record results; tick the box.
10. `GATE-9.1` (doc1:732).
    - [ ] Run the §9.1 smoke+unit suite (8 files) green and record; confirm CI
      `jax-public-unit` still encodes the same suite contract; close the gate.
11. `T4.2` doc update — see WS-D task 13 (folded with `BP-OQ-5`).

### WS-E — Doc-3 open phases

12. **Phase 4 — branch discipline (`TP-P4-OPEN`, doc3:178,182,185-196).**
    - [ ] doc3:178 — audit expensive `lax.cond` / static-arg sites.
    - [ ] doc3:182 — explicit host-boundary classification (shared with
      `CEP-IMPL-6.1`).
    - [ ] doc3:185 — verify hot paths hide no dense fallbacks / host callbacks /
      unexpected materialization.
    - [ ] doc3:186 — keep CPU proof vs CUDA transfer proof distinct (uses WS-B).
    - [ ] doc3:190-196 — classify/test the **7 recommended targets**:
      `biotsavart.py`, `surfaceobjectives_jax.py`, `optimizer_jax.py`,
      `pm_workflow.py`, `wireframe_workflow.py`,
      `permanent_magnet_optimization_jax.py`, optimizer backend static toggles.
13. **Phase 5 — numerical shape/stability audit (`TP-P5`, doc3:212-223).**
    *(Entirely unstarted — the only required item needing net-new engineering.)*
    - [ ] doc3:216 — audit VMEC geometry divisions/√ in
      `src/simsopt/jax_core/vmec_geometry.py`; document each numerical contract.
    - [ ] doc3:217 — audit surface curvature discriminant √ in
      `src/simsopt/geo/surfaceobjectives_jax.py`.
    - [ ] doc3:218 — document solver status/convergence/residual conventions.
    - [ ] doc3:219 — audit compensated reductions / summation order in
      parity-sensitive paths (`src/simsopt/jax_core/reductions.py`).
    - [ ] doc3:220-223 — document physics/numerical contract per guard; add
      parity tests **before** any semantic change; reject silent clamps; add
      explicit invalid-input tests instead of defensive fallbacks.
14. Doc bookkeeping (doc 3 Phase 6 + `T4.2` + `BP-OQ-5`).
    - [ ] `TP-P6-USINGJAX` (doc3:238): refresh the stale
      `docs/using_jax_backend.md` backend-mode table to match the SSOT
      `VALID_BACKEND_MODES` (`src/simsopt/backend/runtime.py:210-218`) =
      `{native_cpu, jax_cpu_fast, jax_cpu_parity, jax_cpu_float32_smoke,
      jax_gpu_fast, jax_gpu_parity, jax_mps_smoke}`; drop `jax_metal_smoke`
      (deprecated alias → `jax_mps_smoke`, `runtime.py:105`) and refresh
      optimizer-default guidance.
    - [ ] `TP-P6-XLINK` (doc3:236-237): add the missing cross-links to
      `docs/remaining_jax_port_surfaces_impl_plan_2026-05-19.md` and
      `docs/bloat_reduction_plan_2026-05-20.md`.
    - [ ] `T4.2` (doc1:686) + `BP-OQ-5` (doc1:830): add the 3 CLAUDE.md edits —
      (a) `scipy-jax` (default) vs `scipy-jax-fullgraph` (stress/parity) outer
      lanes, (b) `src/simsopt/_core/state_tokens.py` location in the
      token/cache contract, (c) SciPy 1.17.1-compatible-port disclosure — and
      update `CLAUDE.md:242`'s `VALID_OPTIMIZER_BACKENDS` note for the outer
      lanes. Mirror the outer-lane distinction into `docs/using_jax_backend.md`.
    - [ ] `TP-P0-INV` / `TP-P3-CATS` (doc3:92-93,154-155): either run+record the
      two remaining pre-edit `rg` inventories and the while_loop/host-loop
      control-flow policy notes, or mark them N/A for this pass with rationale
      (optional, non-gating).

## Validation Plan

Env preamble for all CPU validation (CLAUDE.md → "Validation"):

```bash
export PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu
```

- [ ] **Tier suites + census gate (WS-C):**
  ```bash
  .conda/jax/bin/python -m pytest tests/test_jax_import_smoke.py \
    tests/field/test_biotsavart_jax.py tests/geo/test_surface_fourier_jax.py \
    tests/geo/test_boozer_residual_jax.py tests/objectives/test_integral_bdotn_jax.py \
    tests/geo/test_boozer_derivatives_jax.py tests/geo/test_boozersurface_jax.py \
    tests/integration/test_jax_native_path.py -m "not private_optimizer_runtime" -q
  .conda/jax/bin/python -m pytest tests/test_benchmark_helpers.py \
    -k 'quantity_parity_tolerance or parity_ladder_tolerances' -q   # T2 contract snapshot
  ```
- [ ] **Stage 2 / single-stage parity replay (WS-B task 3, CEP-VAL-7 / 7.2):**
  ```bash
  # Neither parity gate lives behind the private_optimizer_runtime marker, so run
  # both files UNFILTERED — matching CLAUDE.md "Validation" M2+M5 integration bucket
  # (`tests/integration/ -v`). The single-stage file's CPU-reference parity is its
  # 179 public tests; its 2 ondevice tests carry the marker and skip if the private
  # runtime is absent. (Verified collection: stage2 187 tests; single-stage 181.)
  .conda/jax/bin/python -m pytest tests/integration/test_stage2_jax.py \
    tests/integration/test_single_stage_jax_cpu_reference.py -q
  # plus _pre_newton_census_gate_failures byte-identity on pinned input
  # (private ondevice lane, optional): append -m "private_optimizer_runtime"
  ```
- [ ] **Strict-transfer CUDA proof (WS-B task 2, CEP-IMPL-3.3 / CEP-VAL-6)** — on a CUDA host only:
  ```bash
  PYTHONPATH=src SIMSOPT_BACKEND_MODE=jax_gpu_parity SIMSOPT_BACKEND_STRICT=1 \
    SIMSOPT_JAX_TRANSFER_GUARD=disallow JAX_PLATFORMS=cuda,cpu \
    XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_exclude_nondeterministic_ops=true" \
    .conda/jax/bin/python -m pytest -q tests/test_backend.py -k 'cuda_determinism or gpu_memory'
  ```
- [ ] **Doc-3 Phase 4/5 (WS-E):**
  ```bash
  .conda/jax/bin/python -m pytest tests/geo/test_surface_objectives_jax.py \
    tests/geo/test_boozer_residual_jax.py tests/mhd/test_vmec_compute_geometry_jax.py \
    tests/core/test_reductions.py tests/jax_core -k 'surface or vmec or residual or stability or compensated or branch or cond' -q
  ```
- [ ] **Tags created (WS-C):** `git tag -l "bloat-reduction-T*-complete"` lists all 3.
- [ ] **Doc drift closed (WS-D 13/14):**
  ```bash
  grep -niE "scipy-jax|scipy-jax-fullgraph|state_tokens|1\.17" CLAUDE.md           # >0
  grep -nE "jax_cpu_float32_smoke" docs/using_jax_backend.md \
    && grep -nE "jax_mps_smoke" docs/using_jax_backend.md \
    && ! grep -nq "jax_metal_smoke" docs/using_jax_backend.md
  ```
- [ ] **Lint/format on touched files:** `.conda/jax/bin/python -m ruff check <files> && ruff format --check <files>`; `git diff --check`.
- [ ] **Re-audit to convergence:** re-run the plan-conformance audit over all 3
  docs and confirm `requiredUnfinishedCount == 0` / `converged: true`.

## Risks and Mitigations

- Risk: No CUDA host is actually available, blocking WS-B (CEP-IMPL-3.3,
  CEP-VAL-6/7, BP-SC#7).
  Mitigation: Resolve OQ-3 first; if only MPS, run `jax_mps_smoke` and label it
  smoke (not CUDA parity), leaving the CUDA box explicitly conditional.
- Risk: T1-exit LOC target (≥600) is unreachable without new refactors,
  contradicting the "closeout-only" scope.
  Mitigation: OQ-7 forces an explicit decision (bank vs. revise target) before
  tagging; do not tag against an unmet criterion.
- Risk: Stage 2 / single-stage parity replay surfaces a real byte-identity
  regression (release blocker) after 15+ slices touched parity code.
  Mitigation: Run `_pre_newton_census_gate_failures` early in WS-B task 3;
  treat any non-identity as a stop-ship bug, not a tolerance adjustment.
- Risk: Phase-5 stability audit introduces silent clamps that change physics.
  Mitigation: Per doc3:221-222, add parity tests before any semantic change and
  reject clamps unless the mathematical contract already defines them.
- Risk: Doc edits drift from code (the failure mode this audit caught).
  Mitigation: Every status/checkbox change cites file:line evidence
  (CEP-IMPL-7.3); re-audit at the end.

## Completion Criteria

- [ ] All 19 required items DONE: `T1-exit`, `T2-exit`, `T3-exit`, `GATE-9.1`,
  `T4.2` (doc 1); `CEP-IMPL-3.3`, `CEP-VAL-6`, `CEP-VAL-7`, `CEP-IMPL-1.1/1.2/1.3`,
  `CEP-IMPL-6.1/6.4`, `CEP-IMPL-7.1/7.2/7.3` (doc 2); `TP-P4-OPEN`, `TP-P5`,
  `BP-SC` (doc 3 / cross-doc).
- [ ] `git tag -l "bloat-reduction-T*-complete"` returns all three tags.
- [ ] Accelerator-lane evidence (strict-transfer + Stage 2/single-stage parity)
  recorded with exact commands/outputs, or the CUDA box explicitly conditional
  on venue with MPS-smoke evidence in its place.
- [ ] `CLAUDE.md` + `docs/using_jax_backend.md` updated (outer-optimizer lanes,
  `state_tokens.py`, SciPy 1.17.1 port, current backend-mode table).
- [ ] Doc-3 Phase 4 + Phase 5 checkboxes ticked with evidence notes.
- [ ] Re-run audit: `requiredUnfinishedCount == 0`, `converged: true`.
- [ ] `ruff` clean + `git diff --check` clean on all touched files; no new mypy
  errors on touched files (`BP-SC` #9).

## Open Questions

- **OQ-3** (owner: user) — Which CUDA/MPS venue for the strict-transfer + parity
  proof? (self-hosted CUDA runner / Perlmutter / Runpod / local MPS smoke).
  Unblocks WS-B. (doc1:828, doc2:2078)
- **OQ-7** (owner: user) — T1-exit LOC: bank ≥200 more net LOC, or revise the
  ≥600 target with justification? Gates the T1 tag. (doc1:444)
- **OQ-1 / OQ-2 / OQ-6** (owner: user) — branch strategy, time horizon, and the
  OpenMemory-note decision (doc1:826,827,831). Non-gating but should be recorded
  to close doc 1 §12.
- **OQ-BP-5** (owner: user) — apply the 3 CLAUDE.md edits now (WS-D 13) or defer
  as a separate effort? (doc1:830)
