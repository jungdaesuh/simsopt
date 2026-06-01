# Coherent Bloat And TORAX JAX Execution Plan

## Review Envelope

- Target repo: `/Users/suhjungdae/code/columbia/simsopt-jax-shared-jax`
- Source-doc review basis: `b267b0d95` on `shared-jax-clean`
- Current execution checkpoint: `8b94c2bbd` on `shared-jax-clean` with a broad dirty implementation tree across `src/`, `tests/`, and `docs/`
- Reference TORAX repo reviewed: `/Users/suhjungdae/code/opensource/torax` at `60190df1` on clean `main`
- Historical local status at source-doc review: the two source docs were modified and this overlay was untracked; no source-code edits were part of that review. This is no longer the current working-tree state.
- Artifact note: this checkout does not contain a repo-local `.artifacts/` tree. Historical code-smell artifacts referenced by the bloat plan were found in sibling checkout `/Users/suhjungdae/code/columbia/simsopt-jax/.artifacts/code_smell_review_2026-05-20/`.

## 2026-06-01 Drift Checkpoint

The current dirty tree is validated as a contract-hardening / complexity-reduction checkpoint, not as a strict LOC-reduction checkpoint. Do not commit the whole tree under a generic "bloat reduction" label.

Drift-checkpoint ledger captured before the v8 doc correction (`git diff --numstat -- src tests docs` plus untracked-file `wc -l`):

- `src/` tracked: `1933 insertions / 1941 deletions`, net `-8`
- Untracked source helpers: `+53`
- Effective source net: `+45`
- `tests/` tracked: `1174 insertions / 54 deletions`, net `+1120`
- Untracked tests: `+224`
- `docs/` tracked: `938 insertions / 161 deletions`, net `+777`
- Total tracked plus untracked over `src/`, `tests/`, and `docs/`: `+2166`

Execution gate for the next pass:

- **Banked-shrink:** source LOC is net-negative in an isolated scoped slice and behavior/API compatibility is validated.
- **Foundation-only:** source LOC is flat or positive, but the slice names the exact deletion it unlocks and the next deletion task is tracked.
- **Not LOC-banked:** complexity or contract quality improved, but no current source shrink can be claimed.
- **Defer/revert-candidate:** source LOC is flat or positive and no immediate deletion payoff is identified.

Before any commit, split the dirty tree by this classification. Salvage the banked-shrink slices first; keep foundation-only slices only with their follow-up deletion target; do not count tests/docs growth as bloat reduction.

## Purpose

Coordinate execution of the refreshed bloat-reduction and TORAX-informed JAX-porting plans without treating them as independent backlogs. This file is an execution overlay for:

- `docs/bloat_reduction_plan_2026-05-20.md`
- `docs/torax_jax_porting_patterns_impl_plan_2026-05-27.md`

The source plans remain the SSOT for detailed item text, line refs, and acceptance gates. This overlay defines the order, dependency boundaries, and validation checkpoints needed to tackle the two plans coherently.

## Goals

- Execute shared JAX contract work once, then reuse it across bloat-reduction and TORAX-pattern tasks.
- Bank low-risk bloat reductions without blocking higher-value correctness gates.
- Keep persistent-cache, transfer-boundary, and MPS smoke-lane evidence separate from CPU-only proof.
- Prevent new TORAX-inspired abstractions from adding more scaffolding than they remove.
- Produce small, reviewable slices with explicit validation and rollback boundaries.

## Non-Goals

- Do not replace either source plan.
- Do not merge unrelated bloat, cache, optimizer, PM, wireframe, and MPS work into one broad refactor.
- Do not relax public APIs, parity tolerances, backend-mode contracts, or host-transfer policy to make refactors easier.
- Do not copy TORAX abstractions verbatim; only adopt patterns that fit current `simsopt-jax` contracts.
- Do not treat LOC reduction as the only success gate. Complexity reduction and preserved behavior are required.

## Current Context

- Source-doc refresh basis: `shared-jax-clean` at `b267b0d95`.
- Current execution checkpoint is `shared-jax-clean` at `8b94c2bbd`; source-doc edit status from the original review envelope is historical only.
- `docs/bloat_reduction_plan_2026-05-20.md` is a tiered reduction plan: T1 mechanical wins, T2 factory introductions, T3 structural consolidations, and T4 contract decisions.
- `docs/torax_jax_porting_patterns_impl_plan_2026-05-27.md` is a pattern-hardening plan: static/dynamic pytree contracts, persistent-cache proof, bounded control flow, branch discipline, and numerical stability.
- Shared dependency surfaces include `jax_core` specs, backend runtime/cache policy, validation ladder helpers, host-boundary helpers, fixed-iteration scan code, PM/wireframe workflows, and GPU/MPS-sensitive runtime paths.

## Rationale

The two plans overlap in the places most likely to create regressions: JAX object contracts, cache/transfer configuration, compiled control flow, and parity validation. Running the bloat plan first without the TORAX contract work risks deleting or folding code before the invariants are well tested. Running the TORAX plan first without bloat discipline risks adding helper abstractions that increase long-term maintenance cost.

The right sequencing is contract-first, then mechanical deletion, then shared factories, then structural folds. Each slice should have a clear owner document, a narrow changed-file set, and validation strong enough for the touched surface.

## Assumptions

- The source docs were refreshed at `b267b0d95`, but execution has advanced to `8b94c2bbd` with a broad dirty tree. Re-run evidence refresh and the drift ledger before selecting or committing any additional implementation slice.
- Code work must load and apply `/Users/suhjungdae/.agent-docs/SOFTWARE_DESIGN.md` before implementation.
- CPU validation is not enough for GPU-sensitive or MPS-sensitive claims.
- `jax_mps_smoke` remains a smoke lane, not a production parity lane.
- TORAX is a reference for useful JAX patterns, not an upstream dependency or architectural template.

## Dependency Map

| Coherent slice | Bloat-plan source | TORAX-plan source | Why they belong together |
| --- | --- | --- | --- |
| Evidence refresh | Section 4.3, Section 9 | Phase 0 | Both plans require fresh path/caller inventories before edits. |
| JAX object contracts | T1.1, T2.4, selected T2/T3 helpers | Phase 1 | Lazy exports, spec registration, and static/dynamic pytree proof all affect import and JIT behavior. |
| Cache and transfer policy | Section 4.1, Section 9.5 | Phase 2 | Persistent-cache proof and strict-transfer proof share runtime/environment boundaries. |
| Mechanical bloat bank | T1.2 through T1.11 | Phase 0 evidence only | Low-risk deletions should proceed after caller inventory, without waiting for broader TORAX work. |
| Done-gated scan dedup | T3.7, related PM/wireframe items | Phase 3 | Bounded-loop helper work should be designed once and piloted on one workflow pair. |
| Branch discipline | T4.1, T4.2, T4.3 | Phase 4 | Static host decisions, traced branches, and optimizer-lane decisions need one classification model. |
| Numerical shape/stability | T2.8, T2.9, T3.5 | Phase 5 | Tolerance helpers, lane artifacts, and replay diagnostics must preserve independent oracles. |
| Closeout docs | Appendix/status sections | Phase 6 | Source docs should be updated only after code and validation evidence exists. |

## Implementation Plan

1. Preflight and evidence lock
   - [ ] Record `git status --short`, `git rev-parse --short HEAD`, and active branch.
   - [ ] Confirm TORAX reference checkout and HEAD if TORAX-derived claims will be touched.
   - [ ] Re-run source-doc path checks for all `path:line` refs used by the chosen slice.
   - [ ] Run caller inventories before deleting or folding any symbol.
   - [ ] Decide one execution slice and explicitly name its owner source doc.

2. Contract-first foundation
   - [ ] Start with TORAX Phase 1 only where it directly supports bloat-plan work.
   - [ ] Prove current pytree data/meta behavior before changing spec helpers.
   - [ ] Keep static metadata explicit and immutable; do not hide `data_fields` / `meta_fields` in a broad abstraction.
   - [ ] If adding a helper, require it to reduce repeated partition declarations and keep fields auditable.

3. Cache, transfer, and runtime proof
   - [ ] Complete persistent-cache write/reuse proof before claiming cache-policy hardening.
   - [ ] Keep process-local JIT cache proof separate from persistent-cache proof.
   - [ ] For transfer-sensitive changes, run strict-transfer proof on the relevant backend lane.
   - [ ] Keep MPS smoke evidence separate from CPU/CUDA parity evidence.

4. Low-risk bloat reduction
   - [ ] Execute Tier 1 bloat items after the preflight caller inventory.
   - [ ] Prefer import/export list consolidation, dead private helper deletion, and already-validated no-op cleanup before factories.
   - [ ] Preserve public compatibility kwargs and probe scripts unless caller migration is proven.
   - [ ] Commit or review each item as a bisectable slice.

5. Shared factory and loop consolidation
   - [ ] Design factory work twice before changing Tier 2 or Tier 3 surfaces.
   - [ ] Pilot done-gated scan deduplication on one PM/wireframe pair before broader rollout.
   - [ ] Keep independent oracle assertions named and separate even when surrounding setup is deduplicated.
   - [ ] Stop any abstraction that adds a second source of truth for tolerances, schemas, or backend modes.

6. Branch and optimizer decisions
   - [ ] Classify each branch as static host decision, traced runtime control flow, or explicit host-boundary work.
   - [ ] Keep `scipy-jax` / `scipy-jax-fullgraph` as a documented lane decision unless new evidence changes the source plan.
   - [ ] Treat QFM BFGS/SLSQP decisions as behavior-contract decisions, not mechanical dedupe.
   - [ ] Require tests that prove branch semantics, not just reduced branch count.

7. Numerical and parity closeout
   - [ ] Run the validation gate named by the bloat-plan tier and the TORAX-plan phase.
   - [ ] Replay parity-sensitive gates when touching Stage 2, single-stage, tolerance, or lane-artifact code.
   - [ ] Update source docs only with evidence-backed status changes.
   - [ ] Leave unresolved work unchecked and explain blockers directly.

## Validation Plan

- [ ] `git diff --check -- docs/bloat_reduction_plan_2026-05-20.md docs/torax_jax_porting_patterns_impl_plan_2026-05-27.md docs/bloat_torax_coherent_execution_plan_2026-05-31.md`
- [ ] For every implementation slice: `git grep` or `rg` all changed/deleted symbols across `src`, `tests`, `benchmarks`, `examples`, `docs`, `.github`, scripts, and artifacts that are part of the repo contract.
- [ ] For import/export or lazy-loading changes: run the relevant import smoke plus `from simsopt.<package> import *` smoke.
- [ ] For field/geometry/backend-sensitive changes: run the bloat-plan Tier 1 or Tier 2 test gate exactly as scoped in the source plan.
- [ ] For persistent-cache work: run a two-process cache reuse proof with a shared temporary cache directory.
- [ ] For transfer-sensitive work: run strict-transfer proof on the backend lane that the changed code affects.
- [ ] For parity-sensitive work: run the Stage 2 and single-stage gates named in the bloat plan before closing the item.

## Risks and Mitigations

- Risk: A TORAX-inspired helper creates another abstraction layer without deleting real complexity.
  Mitigation: Require a before/after caller map and reject helpers that do not remove repeated declarations or enforce a tested invariant.

- Risk: Low-risk bloat work accidentally removes public compatibility or oracle coverage.
  Mitigation: Run full caller inventory and preserve public kwargs, probe scripts, and independent assertions unless migration evidence is explicit.

- Risk: CPU-only validation is mistaken for GPU, CUDA, or MPS proof.
  Mitigation: Label each validation result by backend lane and do not close GPU-sensitive work without the relevant strict-transfer or smoke evidence.

- Risk: Source docs drift again during concurrent commits.
  Mitigation: Treat all line refs as snapshots and re-grep before every implementation slice.

- Risk: Multiple plans become conflicting sources of truth.
  Mitigation: Keep this file as an overlay only; update the source plan that owns the detailed item when status changes.

## Completion Criteria

- [ ] One execution slice is selected with a named source-doc owner and validation gate.
- [ ] All changed symbols in that slice have caller inventories.
- [ ] The implemented dirty-tree slices preserve public APIs, parity tolerances, backend modes, and transfer/cache contracts in their recorded validation scope.
- [ ] The source doc owning the completed work is updated with evidence, not just checkbox changes.
- [ ] Validation output is recorded with backend lane and exact command.
- [ ] Remaining work is still traceable to the source docs and not duplicated into an unsorted backlog.
- [ ] Before the next commit, split the dirty tree into banked-shrink, foundation-only, not-LOC-banked, and defer/revert-candidate slices.

## Open Questions

- Which slice should be executed first: Tier 1 bloat import/export cleanup, TORAX Phase 1 pytree contract tests, or persistent-cache two-process proof?
- Should completed slices be committed one checkbox at a time, or grouped by validation gate when multiple tiny doc-only updates are adjacent?
- What backend lane is available for strict-transfer proof in the current machine context when a GPU-sensitive item is selected?
