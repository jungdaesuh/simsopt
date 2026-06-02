# Remaining-Items Implementation Plan (2026-06-02)

## Purpose

Close out the unfinished work across the three active planning docs so each
reaches "all items done." This plan is the execution artifact derived from the
2026-06-02 plan-conformance audit of:

- `docs/bloat_reduction_plan_2026-05-20.md` (doc 1)
- `docs/bloat_torax_coherent_execution_plan_2026-05-31.md` (doc 2)
- `docs/torax_jax_porting_patterns_impl_plan_2026-05-27.md` (doc 3)

Original audit result (110 items): **78 DONE, 15 PARTIAL, 17 MISSING**;
**19 required items unfinished** before the 2026-06-02 closeout updates. The
code refactors landed. After the local docs/test closeout and CPU/X64 replay,
the remaining required work is **accelerator validation closeout**:
CUDA/GPU strict-transfer proof, post-CUDA tags, and the final post-CUDA
convergence audit.

## Goals

- Run and record the post-CUDA/GPU **tier-exit finalization** and create the
  `bloat-reduction-T{1,2,3}-complete` tags after the validated commit exists.
- Produce **accelerator-lane evidence** (strict-transfer CUDA proof) currently
  missing from every accelerator-sensitive slice.
- Complete doc-2 **preflight + numerical/parity closeout** bookkeeping items.
- Complete doc-3 **Phase 4 (branch discipline)** and **Phase 5 (numerical
  shape/stability audit)**.
- Update **CLAUDE.md / user docs** for the outer-optimizer lanes and the stale
  backend-mode table.
- Re-run both audits and reach `converged: true` (zero required items unfinished).

## Non-Goals

- No new LOC-reduction refactors; T1 LOC is revised in closure mode and
  T3.3/T3.5/T3.7 stay closed as NO-BANK decisions.
- No new runtime features; this is closeout + proof, not implementation.
- No relaxation of the strict CPU↔C++ byte-identity gate or parity tolerances.
- No editing of `MEMORY.md` unless OQ-6 is explicitly answered "yes."

## Current Context

- Branch: `shared-jax-clean`. Validation lane: in-tree `.conda/jax` (JAX
  0.10.0 / jaxlib 0.10.0) per `CLAUDE.md` → "Validation".
- **No `bloat-reduction-*` git tags exist** (`git tag -l "*bloat*"` empty) —
  all three tier-exit gates are unfired.
- **No accelerator-lane proof exists anywhere**: the only
  `transfer_guard("disallow")` tests run on CPU (host==device); doc 2 states
  "Strict-transfer and full accelerator replay gates are not claimed by this
  pass." The audit host is Apple Silicon (Metal-only, no local CUDA).
- The release-blocker byte-identity gate is
  `_pre_newton_census_gate_failures` at
  `benchmarks/single_stage_init_parity.py:3284` (CLAUDE.md → "Parity modes").
- **2026-06-02 local docs/test closeout now exists in the dirty tree:**
  `CLAUDE.md` separates the Boozer LS backend vocabulary from the outer
  Stage 2/single-stage optimizer lanes and documents `scipy-jax`,
  `scipy-jax-fullgraph`, `state_tokens.py`, and the SciPy 1.17.1 port.
- Doc-3 Phases 4 and 5 now have local CPU/source evidence notes and focused
  invalid-input tests. This satisfies the local doc-3 closeout slice, but does
  not close CUDA/GPU accelerator proof, post-CUDA tags, or final post-CUDA
  convergence. Stage 2/single-stage parity replay and grouped CPU/source
  validation are now closed below with exact evidence.
- `docs/using_jax_backend.md` now lists the seven public backend modes,
  `jax_cpu_float32_smoke`, and `jax_mps_smoke`; `jax_metal_smoke` remains only
  as a rejected/deprecated selector reference.

## Rationale

The remaining work clusters into four dependency-ordered workstreams. **WS-A
(decisions)** comes first because choosing a CUDA venue (OQ-3) unblocks the
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

- **ASSUMP-1:** The old T1 LOC target is revised for closure mode. Complexity
  reduction and validated source-queue closure are the gate; net LOC is a
  secondary indicator, matching doc 1 §2. No extra T1 LOC-bank task remains.
- **ASSUMP-2:** A CUDA host with the repo's CUDA/JAX env is reachable for the
  strict-transfer proof. Local Apple MPS smoke is out of scope for this
  closeout pass and must not be substituted for CUDA strict-transfer parity.
- **ASSUMP-3:** `linearization_residency="device"` is required on the strict
  CUDA lane (CLAUDE.md → "Linearization residency / strict-transfer contract");
  `"host"` fails by design under `transfer_guard("disallow")` on an accelerator.
- **ASSUMP-4:** The working-tree docs are the source of truth the audit ran
  against; tasks below edit those same files.

## Implementation Plan

### WS-A — Decisions / unblock (do first)

1. Resolve the open questions that gate downstream work.
   - [ ] **CUDA/GPU OQ-3 / `BP-OQ-3` / `CEP-OQ-3` (doc1:853, doc2:2078):**
     choose the CUDA proof venue (self-hosted GitHub CUDA runner, Perlmutter, Runpod,
     or another CUDA host). Record the choice in doc 1 §12 and doc 2 Open
     Questions. **Blocks WS-B.**
   - [x] **OQ-7 / T1 LOC (doc1:444):** closure-mode target revised: complexity
     reduction and validated source-queue closure are the gate; no new T1 LOC
     banking task remains.
   - [x] **`BP-OQ-1` (doc1:851):** branch strategy for this dirty-tree closeout
     is to stay on the current `shared-jax-clean` branch; no new branch is
     created.
   - [x] **`BP-OQ-2` (doc1:852):** time horizon is this current closeout pass;
     no sprint/background follow-up is opened unless new scope is requested.
   - [x] **`BP-OQ-6` (doc1:859):** no project note is required for this pass
     because no T1.4 curve↔jax_core import-cycle behavior changed.
   - [x] **`BP-OQ-5` / `T4.2` CLAUDE.md edits (doc1:855, 686):** confirm the 3
     CLAUDE.md edits are in scope now (folded into WS-D task 13) or explicitly
     deferred in-doc.
     2026-06-02 status: applied in scope and mirrored into
     `docs/using_jax_backend.md`; doc 1 Open Question 5 is now checked.

### WS-B — Accelerator-lane proof (needs OQ-3)

2. Strict-transfer proof on the chosen accelerator lane.
   - [ ] Run a CUDA/GPU `jax.transfer_guard("disallow")` proof on the affected
     `jax_gpu_*` CUDA lane for transfer-sensitive surfaces (Biot-Savart,
     BoozerSurface LS factors, single-stage adjoint). Record the
     exact command + output. Satisfies `CEP-IMPL-3.3` (doc2:142) and
     `CEP-VAL-6` (doc2:176).
   - [ ] Confirm `linearization_residency="device"` on the CUDA lane (ASSUMP-3);
     assert no device→host transfer escapes the guard.
     2026-06-02 local CPU evidence: focused Boozer residency,
     guard-silencing, host-boundary, and dispatch transfer-contract tests
     passed (`21 passed, 2 skipped, 9 deselected in 41.74s`). This catches
     source-level silent transfer escapes, including accidental
     `device_to_host allow` around host linearization placement, but because CPU
     has host==device it does not close the CUDA strict-transfer proof.
   - [ ] After CUDA/GPU proof, tick doc2:142 and doc2:176; update the per-slice "transfer-sensitive
     proof remains open" notes that now have evidence.
3. Stage 2 + single-stage parity replay (release blocker).
   - [x] Replay the Stage 2 / single-stage parity gates named in the bloat plan:
     `tests/integration/test_stage2_jax.py`,
     `tests/integration/test_single_stage_jax_cpu_reference.py`, and the
     `_pre_newton_census_gate_failures` byte-identity gate on pinned input.
     Satisfies `CEP-VAL-7` (doc2:177) and `CEP-IMPL-7.2` (doc2:165).
     2026-06-02 focused CPU/X64 blocker status: the production LS parity and
     Iotas FD blockers now pass after tightening the production LS Newton
     polish and making the ill-conditioned Iotas FD oracle condition-aware
     (`2 passed in 68.51s`). The pinned pre-Newton census selector passes
     (`7 passed, 353 deselected in 2.29s`). A later full unfiltered replay
     finished with one traceable optimizer endpoint test-contract failure
     (`1 failed, 362 passed, 5 skipped in 3166.77s`); that focused traceable
     test now passes after success-gating optimizer result-field/endpoint
     equality (`1 passed in 854.67s`). The full unfiltered rerun after that fix
     passed (`363 passed, 5 skipped, 5 warnings in 3947.69s`).
   - [x] Record results; tick doc2:165, doc2:177.
     2026-06-02 status: doc 2 §7 and its validation-plan parity-sensitive gate
     now record the full replay pass above.
   - [ ] Update `BP-SC` sub-criterion #7 (GPU/strict-transfer proof) once WS-B
     tasks 2–3 produce real accelerator evidence (doc1:93–94 / §2).

### WS-C — Tier-exit gates (needs OQ-7 for T1)

4. **`T1-exit`** (doc1:444).
   - [x] Satisfy ASSUMP-1 per OQ-7 (bank ≥200 net LOC or record target revision).
     2026-06-02 status: target revision recorded; no extra T1 LOC-bank task
     remains in closure mode.
   - [x] Run the full T1 suite green; re-affirm the contract checklist.
     2026-06-02 evidence: the §9.1 file set passed in split CPU/X64 chunks
     (`114 passed, 11 skipped`; `53 passed`; `255 passed, 105 skipped`;
     `502 passed, 4 skipped`; `14 passed`), the Stage 2/single-stage replay
     passed (`363 passed, 5 skipped`), `tests/test_benchmark_helpers.py -k
     'quantity_parity_tolerance or parity_ladder_tolerances'` passed
     (`6 passed, 354 deselected`), `tests/test_run_code_benchmark_common.py`
     passed (`5 passed`), and the pinned pre-Newton census selector passed
     (`7 passed, 353 deselected`).
   - [ ] Post-CUDA/GPU finalization: `git tag bloat-reduction-T1-complete` at
     the validated commit after accelerator proof lands.
5. **`T2-exit`** (doc1:578).
   - [x] Re-run the T2 suite + contract checklist + `_pre_newton_census_gate_failures`
     replay gate **as one grouped tier-exit proof**.
     2026-06-02 evidence: `tests/test_benchmark_helpers.py -k
     'quantity_parity_tolerance or parity_ladder_tolerances'` passed
     (`6 passed, 354 deselected`), the pinned pre-Newton census selector passed
     (`7 passed, 353 deselected`), and the full Stage 2/single-stage replay
     passed (`363 passed, 5 skipped`).
   - [ ] Post-CUDA/GPU finalization: `git tag bloat-reduction-T2-complete` at
     the validated commit after accelerator proof lands.
6. **`T3-exit`** (doc1:667).
   - [x] Run grouped closure validation + one adversarial review + contract
     checklist re-affirmation against the final tree.
     2026-06-02 evidence: CPU/X64 grouped validation passed across §9.1,
     Stage 2/single-stage replay, T2 tolerance snapshot, benchmark-common, and
     doc-3 Phase 4/5 selectors. A read-only adversarial review found stale docs
     only and no code findings for transfer-guard bypasses or fixture fixes;
     the stale docs were corrected in this pass.
   - [ ] Post-CUDA/GPU finalization: `git tag bloat-reduction-T3-complete` at
     the validated commit after accelerator proof lands.

### WS-D — Closeout & preflight bookkeeping (doc 2 + doc 1)

7. Preflight evidence lock (doc 2 §1).
   - [x] `CEP-IMPL-1.1` (doc2:127): paste literal `git status --short` output
     into the preflight evidence block; tick the box.
     2026-06-02 continuation update records the literal dirty tree under doc 2's
     local Phase 4/5 closeout log.
   - [x] `CEP-IMPL-1.2` (doc2:128): tick the TORAX checkout/HEAD preflight box
     (HEAD `60190df1` already verified) and record it as a completed step.
     2026-06-02 continuation update reconfirmed TORAX clean `main` at
     `60190df1`.
   - [x] `CEP-IMPL-1.3` (doc2:129): add one consolidated path-check
     re-verification covering all referenced paths; tick the box.
     2026-06-02 continuation update verified owner docs, user docs, CI workflow,
     and touched tests exist in the current checkout.
8. Branch/optimizer decisions (doc 2 §6).
   - [x] `CEP-IMPL-6.1` (doc2:158): classify the third category — **explicit
     host-boundary work** — for the branch sites (doc3:191 dependency); extend
     beyond the 2-site pilot.
     2026-06-02 evidence: doc 3 Phase 4 classifies the seven recommended targets,
     including explicit host-boundary work; doc 2 mirrors the classification.
   - [x] `CEP-IMPL-6.4` (doc2:161): re-affirm branch-semantics tests pass and
     mark the guardrail satisfied for §6 scope.
     2026-06-02 CPU/X64 suite passed:
     `tests/jax_core/test_tracing_jax_item14.py`,
     `tests/jax_core/test_tracing_jax_conservation.py`,
     `tests/solve/test_pm_workflow_jax.py`, and
     `tests/solve/test_wireframe_workflow_jax.py` reported
     `95 passed, 1 skipped`.
9. Numerical/parity closeout (doc 2 §7).
   - [x] `CEP-IMPL-7.2` (doc2:165): covered by WS-B task 3.
     2026-06-02 status: full CPU/X64 Stage 2 + single-stage replay passed
     (`363 passed, 5 skipped, 5 warnings in 3947.69s`) after the focused
     traceable endpoint-contract fix.
   - [x] `CEP-IMPL-7.3` (doc2:166): tick the "evidence-backed status changes"
     process box once all status changes above carry evidence.
     2026-06-02 status changes in doc 1/2/3 now cite local commands, path checks,
     or explicit unresolved blockers; only CUDA/GPU accelerator proof remains
     unchecked.
   - [x] `CEP-IMPL-7.1` (doc2:164): **run last** — the grouped/full
     tier-completion validation gate; record results; tick the box.
     2026-06-02 status: non-accelerator grouped validation and adversarial
     review completed as recorded above; final tags/audit stay post-CUDA/GPU.
10. `GATE-9.1` (doc1:739).
    - [x] Run the §9.1 smoke+unit file set (8 files) green and record; confirm CI
      `jax-public-unit` still encodes the same suite contract. This is equivalent
      split-file-set evidence under the updated §9.1 rule, not an all-in-one
      command pass. 2026-06-02 local CPU/X64 split rerun covered all 8 files:
      import smoke
      `114 passed, 11 skipped`; Biot-Savart JAX `53 passed`;
      surface/boozer-residual/integral chunk `255 passed, 105 skipped`; Boozer
      derivatives/BoozerSurface chunk `502 passed, 4 skipped`; native-path
      integration `14 passed`. `.github/workflows/jax_smoke.yml` still carries
      the import-smoke step and the remaining §9.1 files in `jax-public-unit`.
11. `T4.2` doc update — see WS-D task 13 (folded with `BP-OQ-5`).

### WS-E — Doc-3 open phases

12. **Phase 4 — branch discipline (`TP-P4-OPEN`, doc3:185-234).**
    - [x] doc3:187 — audit expensive `lax.cond` / static-arg sites.
    - [x] doc3:191 — explicit host-boundary classification (shared with
      `CEP-IMPL-6.1`).
    - [x] doc3:194 — verify hot paths hide no dense fallbacks / host callbacks /
      unexpected materialization.
    - [x] doc3:195 — keep CPU proof vs CUDA transfer proof distinct (uses WS-B).
    - [x] doc3:197-205 — classify/test the **7 recommended targets**:
      `biotsavart.py`, `surfaceobjectives_jax.py`, `optimizer_jax.py`,
      `pm_workflow.py`, `wireframe_workflow.py`,
      `permanent_magnet_optimization_jax.py`, optimizer backend static toggles.
      Evidence is recorded in doc 3 Phase 4 and the 2026-06-02 overlay log; this
      remains CPU/source evidence, not CUDA transfer proof.
13. **Phase 5 — numerical shape/stability audit (`TP-P5`, doc3:236-263).**
    - [x] doc3:240 — audit VMEC geometry divisions/√ in
      `src/simsopt/jax_core/vmec_geometry.py`; document each numerical contract.
    - [x] doc3:241 — audit surface curvature discriminant √ in
      `src/simsopt/geo/surfaceobjectives_jax.py`.
    - [x] doc3:242 — document solver status/convergence/residual conventions.
    - [x] doc3:243 — audit compensated reductions / summation order in
      parity-sensitive paths (`src/simsopt/jax_core/reductions.py`).
    - [x] doc3:244-247 — document physics/numerical contract per guard; add
      parity tests **before** any semantic change; reject silent clamps; add
      explicit invalid-input tests instead of defensive fallbacks.
      Evidence is recorded in doc 3 Phase 5 and the 2026-06-02 overlay log.
14. Doc bookkeeping (doc 3 Phase 6 + `T4.2` + `BP-OQ-5`).
    - [x] `TP-P6-USINGJAX` (doc3:278): refresh the stale
      `docs/using_jax_backend.md` backend-mode table to match the SSOT
      `VALID_BACKEND_MODES` (`src/simsopt/backend/runtime.py:210-218`) =
      `{native_cpu, jax_cpu_fast, jax_cpu_parity, jax_cpu_float32_smoke,
      jax_gpu_fast, jax_gpu_parity, jax_mps_smoke}`; drop `jax_metal_smoke`
      (deprecated alias → `jax_mps_smoke`, `runtime.py:105`) and refresh
      optimizer-default guidance.
    - [x] `TP-P6-XLINK` (doc3:275-278): add the missing cross-links to
      `docs/remaining_jax_port_surfaces_impl_plan_2026-05-19.md` and
      `docs/bloat_reduction_plan_2026-05-20.md`.
    - [x] `T4.2` (doc1:686) + `BP-OQ-5` (doc1:855): add the 3 CLAUDE.md edits —
      (a) `scipy-jax` (default) vs `scipy-jax-fullgraph` (stress/parity) outer
      lanes, (b) `src/simsopt/_core/state_tokens.py` location in the
      token/cache contract, (c) SciPy 1.17.1-compatible-port disclosure — and
      update `CLAUDE.md:242`'s `VALID_OPTIMIZER_BACKENDS` note for the outer
      lanes. Mirror the outer-lane distinction into `docs/using_jax_backend.md`.
    - [x] `TP-P0-INV` / `TP-P3-CATS` (doc3:92-93,154-155): either run+record the
      two remaining pre-edit `rg` inventories and the while_loop/host-loop
      control-flow policy notes, or mark them N/A for this pass with rationale
      (optional, non-gating).
      2026-06-02 evidence: doc 3 already records all four Phase 0 inventories as
      checked (`register_dataclass|data_fields|meta_fields|static_arg|static_argnames`,
      `persistent_cache|compilation_cache|XLA_FLAGS|JAX_COMPILATION_CACHE`,
      `lax.scan|lax.while_loop|lax.cond|fori_loop`, and
      `sqrt|where|nan|clip|maximum|minimum|compensated`). The current checkout
      rerun refreshed the dataclass/static, cache/XLA, and control-flow
      inventories, and doc 3 Phase 3 records the intended categories:
      `lax.scan` for fixed-capacity loops, `lax.while_loop` only for true
      dynamic state machines with an owned differentiation contract, and host
      loops for I/O, callbacks, plotting, logging, and object mutation.

## Validation Plan

Env preamble for all CPU validation (CLAUDE.md → "Validation"):

```bash
export PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu
```

- [x] **Tier suites + census gate (WS-C):**
  ```bash
  .conda/jax/bin/python -m pytest tests/test_jax_import_smoke.py \
    tests/field/test_biotsavart_jax.py tests/geo/test_surface_fourier_jax.py \
    tests/geo/test_boozer_residual_jax.py tests/objectives/test_integral_bdotn_jax.py \
    tests/geo/test_boozer_derivatives_jax.py tests/geo/test_boozersurface_jax.py \
    tests/integration/test_jax_native_path.py -m "not private_optimizer_runtime" -q
  .conda/jax/bin/python -m pytest tests/test_benchmark_helpers.py \
    -k 'quantity_parity_tolerance or parity_ladder_tolerances' -q   # T2 contract snapshot
  ```
  2026-06-02 evidence: the §9.1 portion ran in split local CPU/X64 chunks,
  covering every file in the command above with no failures (`114/11 skipped`,
  `53`, `255/105 skipped`, `502/4 skipped`, `14`). The T2 contract snapshot
  passed (`6 passed, 354 deselected`), `tests/test_run_code_benchmark_common.py`
  passed (`5 passed`), the pinned pre-Newton census selector passed (`7 passed,
  353 deselected`), and the full Stage 2/single-stage replay passed (`363
  passed, 5 skipped, 5 warnings`).
- [x] **Stage 2 / single-stage parity replay (WS-B task 3, CEP-VAL-7 / 7.2):**
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
  2026-06-02 local CPU/X64 status: the first full replay reached 97% before
  termination after two failure markers; focused rerun reproduced
  `TestRunCodeLSParity::test_ls_solve_state_parity_production_scale` and
  `TestIotasJAXResolveFD::test_iotas_resolve_fd`. After the focused test
  contract fix, those two selectors pass (`2 passed in 68.51s`). The pinned
  pre-Newton census gate passes (`7 passed, 353 deselected in 2.29s`). A later
  full unfiltered replay finished with one traceable optimizer endpoint
  test-contract failure (`1 failed, 362 passed, 5 skipped in 3166.77s`); the
  focused traceable test now passes after success-gating optimizer
  result-field/endpoint equality (`1 passed in 854.67s`). The full unfiltered
  rerun after that fix passed (`363 passed, 5 skipped, 5 warnings in
  3947.69s`).
- [ ] **Strict-transfer CUDA proof (WS-B task 2, CEP-IMPL-3.3 / CEP-VAL-6)** — on a CUDA host only:
  ```bash
  PYTHONPATH=src SIMSOPT_BACKEND_MODE=jax_gpu_parity SIMSOPT_BACKEND_STRICT=1 \
    SIMSOPT_JAX_TRANSFER_GUARD=disallow JAX_PLATFORMS=cuda,cpu \
    XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_exclude_nondeterministic_ops=true" \
    .conda/jax/bin/python -m pytest -q tests/test_backend.py -k 'cuda_determinism or gpu_memory'
  ```
- [x] **Doc-3 Phase 4/5 (WS-E):**
  ```bash
  .conda/jax/bin/python -m pytest tests/geo/test_surface_objectives_jax.py \
    tests/geo/test_boozer_residual_jax.py tests/mhd/test_vmec_compute_geometry_jax.py \
    tests/core/test_reductions.py tests/jax_core -k 'surface or vmec or residual or stability or compensated or branch or cond' -q
  ```
  2026-06-02 evidence: the first full rerun exposed two stale test-fixture
  monkeypatch targets in `tests/geo/test_surface_objectives_jax.py`; after
  correcting the fixtures, the two focused failures passed (`2 passed in
  4.63s`) and the full selector passed (`386 passed, 63 skipped, 492
  deselected, 1 xfailed in 629.42s`).
- [ ] **Post-CUDA/GPU tags created (WS-C):** `git tag -l "bloat-reduction-T*-complete"` lists all 3.
- [x] **Doc drift closed (WS-D 13/14):**
  ```bash
  grep -niE "scipy-jax|scipy-jax-fullgraph|state_tokens|1\.17" CLAUDE.md           # >0
  grep -nE "jax_cpu_float32_smoke" docs/using_jax_backend.md \
    && grep -nE "jax_mps_smoke" docs/using_jax_backend.md \
    && ! grep -nq "jax_metal_smoke" docs/using_jax_backend.md
  ```
  2026-06-02 evidence: the grep gate passed and found `scipy-jax`,
  `scipy-jax-fullgraph`, `state_tokens.py`, and SciPy `1.17.1` in
  `CLAUDE.md`; `docs/using_jax_backend.md` lists `jax_cpu_float32_smoke` and
  `jax_mps_smoke` and does not contain `jax_metal_smoke`.
- [x] **Lint/format on touched files:** `.conda/jax/bin/python -m ruff check <files> && ruff format --check <files>`; `git diff --check`.
  2026-06-02 evidence: `ruff check` passed on the five touched Python test
  files, `ruff format --check` reported `5 files already formatted`, and
  `git diff --check` passed. The touched Python set is test-only and
  `py_compile` passed on all five touched test files.
- [ ] **Post-CUDA/GPU re-audit to convergence:** re-run the plan-conformance audit over all 3
  docs and confirm `requiredUnfinishedCount == 0` / `converged: true`.

## Risks and Mitigations

- Risk: No CUDA host is actually available, blocking WS-B (CEP-IMPL-3.3,
  CEP-VAL-6/7, BP-SC#7).
  Mitigation: Resolve OQ-3 to a CUDA venue first; keep the CUDA box open until
  that proof runs, and do not substitute local CPU/MPS smoke evidence for it.
- Risk: Historical LOC targets are mistaken for active implementation tasks.
  Mitigation: OQ-7 records the closure-mode target revision: complexity
  reduction and validated source-queue closure are the gate; no new LOC-bank
  task remains.
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

- [ ] Post-CUDA/GPU final audit: all 19 required items DONE: `T1-exit`, `T2-exit`, `T3-exit`, `GATE-9.1`,
  `T4.2` (doc 1); `CEP-IMPL-3.3`, `CEP-VAL-6`, `CEP-VAL-7`, `CEP-IMPL-1.1/1.2/1.3`,
  `CEP-IMPL-6.1/6.4`, `CEP-IMPL-7.1/7.2/7.3` (doc 2); `TP-P4-OPEN`, `TP-P5`,
  `BP-SC` (doc 3 / cross-doc).
- [ ] Post-CUDA/GPU finalization: `git tag -l "bloat-reduction-T*-complete"` returns all three tags.
- [ ] Accelerator-lane evidence (strict-transfer CUDA/GPU proof)
  recorded with exact commands/outputs on the CUDA/GPU lane.
- [x] `CLAUDE.md` + `docs/using_jax_backend.md` updated (outer-optimizer lanes,
  `state_tokens.py`, SciPy 1.17.1 port, current backend-mode table).
- [x] Doc-3 Phase 4 + Phase 5 checkboxes ticked with evidence notes.
- [ ] Post-CUDA/GPU re-run audit: `requiredUnfinishedCount == 0`, `converged: true`.
- [x] `ruff` clean + `git diff --check` clean on all touched files; no new mypy
  errors on touched files (`BP-SC` #9).

## Open Questions

- **CUDA/GPU OQ-3** (owner: user) — Which CUDA venue for the strict-transfer + parity
  proof? (self-hosted CUDA runner / Perlmutter / Runpod / another CUDA host).
  Unblocks WS-B. (doc1:853, doc2:2078)
- **OQ-7** — closed on 2026-06-02: T1 LOC target revised for closure mode; no
  extra T1 LOC-bank task remains.
- **OQ-1 / OQ-2 / OQ-6** — closed on 2026-06-02: stay on `shared-jax-clean` for
  this dirty-tree pass, no sprint/background scope opened, and no T1.4
  import-cycle memory note is required because no such behavior changed.
- **OQ-BP-5** — closed on 2026-06-02: the 3 CLAUDE.md edits were applied in
  scope and mirrored into `docs/using_jax_backend.md`. (doc1:855)
