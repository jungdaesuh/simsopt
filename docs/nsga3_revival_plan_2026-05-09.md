# NSGA-III Revival Plan

**Date:** 2026-05-09
**Branch:** surrogate-confinement-v2
**Predecessor:** `b85e4f12b frontier: land r1-r50 bloat reduction (~22% LOC)` (2026-05-08) deleted NSGA-III on D1=DROP evidence
**Code restoration ref:** `b85e4f12b^` (parent commit; full deletion diff is `git show b85e4f12b`)
**Owner:** TBD
**Status:** Not started — gated on a trigger below

---

## 1. When to revive (triggers — pick at least one)

- **T1.** `D1.4` returns YES — a teammate produces a retained NSGA-III campaign artifact AND a benchmark showing it reaches a Pareto member that `multilane_local + achievement_chebyshev_full_simplex_v1` cannot reach under equal evaluation budget. (`D1.B` original criteria.)
- **T2.** `1A.10` / `CC.3` / `DC.13` certified-lane smoke remains blocked after the three external-action paths in the bloat-reduction plan's r50 entry, AND a stakeholder argues that population-based search would unblock where `multilane_local` cannot. (Speculative — must come with a concrete failure analysis pointing at the engine, not at fixtures or contracts.)
- **T3.** A new research direction needs a generated 4-D Pareto frontier on objectives that `multilane_local` cannot scalarize (it varies only iota↔volume shares, clamped to [0.2, 0.8]; never touches qa_error / boozer_residual axes).

If none of T1–T3 is met, **do not revive.** This plan is dormant. The deletion was YAGNI-correct.

---

## 2. Pre-revival audit

Run before touching code. If any of these is unsatisfied, the revival does not pay for itself.

- [ ] **A1.** Reproduce the trigger condition. Attach the artifact path (T1), failure log (T2), or research request (T3) to this plan.
- [ ] **A2.** Confirm `multilane_local` cannot reach the missing region. Run an `achievement_chebyshev_full_simplex_v1` 4-axis sweep at the same evaluation budget as the proposed NSGA-III run; confirm it does not produce the missing Pareto member. Attach archive JSON.
- [ ] **A3.** Confirm `pymoo>=0.6` is acceptable as a documented optional dependency. Project-wide policy check.
- [ ] **A4.** Confirm wall-time budget. NSGA-III population × generations × per-eval ALM cost is typically days. Get explicit sign-off on the run-time budget before promising a campaign artifact.

---

## 3. Restoration sequence

### 3.1 Recover deleted code from history

```bash
# Ref: b85e4f12b^ is the parent of the r1-r50 landing commit
PARENT=b85e4f12b^

git checkout "$PARENT" -- \
  examples/single_stage_optimization/banana_opt/frontier_engine_nsga3.py \
  tests/geo/test_frontier_evaluator.py
```

Note: `frontier_evaluator.py` was retired in r26 as orphaned. Do **not** restore the evaluator unless a concrete consumer requires it; see §3.5.

The runner-dispatch hunks, validator support, summary-field plumbing, and other touch points were edited (not file-level deleted) in `b85e4f12b`. They must be restored hunk-by-hunk:

```bash
git show "$PARENT":examples/single_stage_optimization/run_single_stage_frontier_campaign.py > /tmp/runner_pre_drop.py
diff /tmp/runner_pre_drop.py examples/single_stage_optimization/run_single_stage_frontier_campaign.py
# Re-apply the NSGA-III-specific hunks only; do NOT undo r10-r50 hardening
```

Files with mixed hunks to selectively restore:
- `examples/single_stage_optimization/run_single_stage_frontier_campaign.py` (CLI choices, dispatch, summary post-processing)
- `examples/single_stage_optimization/banana_opt/frontier_contracts.py` (engine string in `SUPPORTED_FRONTIER_ENGINES`, optional NSGA-III summary fields validator, removal of those field names from `DELETED_FRONTIER_SUMMARY_FIELDS`)
- Test files with NSGA-III blocks: `tests/geo/test_frontier_evaluator.py` (recover), `test_single_stage_workflow_helpers.py` (`test_frontier_campaign_nsga3_records_generation_summary`, `test_frontier_campaign_nsga3_resume_reuses_saved_engine_artifacts`), `test_frontier_contracts.py` (`test_summary_validator_accepts_optional_nsga3_fields`)

### 3.2 Engine Protocol abstraction (mandatory for revival; not optional)

The original deletion's audit found that the runner used a hard-coded `if args.frontier_engine == "nsga3"` branch (now gone). Reviving without the SOLID fix would re-introduce the same problem.

Define `examples/single_stage_optimization/banana_opt/frontier_engine_protocol.py`:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class FrontierEngine(Protocol):
    name: str
    def run(self, spec, budget, *, hypervolume_reference, ...) -> EngineArtifacts: ...
    def load_artifacts(self, path) -> EngineArtifacts | None: ...
```

Both engines implement it:
- `MultilaneLocalEngine` — wraps the current scalarized-lane execution path
- `NSGA3Engine` — wraps the restored pymoo dispatch

Runner replaces the if/else with a registry:
```python
ENGINE_REGISTRY: Mapping[str, Callable[[], FrontierEngine]] = MappingProxyType({
    "multilane_local": MultilaneLocalEngine,
    "nsga3": NSGA3Engine,
})
engine = ENGINE_REGISTRY[args.frontier_engine]()
```

This is the YAGNI line: a protocol with two implementations is justified; a protocol with one is not (which is why the original 14-module engine layer was deleted).

### 3.3 Pymoo as optional extra (not required runtime dep)

```toml
# pyproject.toml
[project.optional-dependencies]
nsga3 = ["pymoo>=0.6"]
```

Keep the existing module-level `try/except ImportError` lazy-load pattern at `frontier_engine_nsga3.py:33-44` — that *is* the lazy mechanism; do not move class definitions inside functions (the original audit's "Approach A" was rejected as over-engineering when "Approach B" already works). The validator should refuse to dispatch to NSGA-III when pymoo is missing, with a clear error pointing at `pip install simsopt-surrogate[nsga3]`.

### 3.4 Pre-existing issues to fix on revival (from r6.2 audit, not optional)

These were "fell away with deletion" findings. They are blockers for KEEP:

- [ ] **B1. Lane-record divergence.** The original NSGA-III branch did not populate `lane_records_by_id`, so `frontier_lane_records` was whatever stale entries were already there (`run_single_stage_frontier_campaign.py:916-942` in `b85e4f12b^`). Fix: every per-generation NSGA-III evaluation must produce a `FrontierLaneRecord` written into `lane_records_by_id` with the same shape as `multilane_local` lanes.
- [ ] **B2. Double-evaluation in callback.** `_ArchiveTrackingCallback.notify` re-extracts `algorithm.pop` and re-evaluates the entire population every generation (`frontier_engine_nsga3.py:107-109` in `b85e4f12b^`). Pymoo caches results, so this inflates lookups 2× without producing wrong answers. Fix: read population evaluations from cache; do not call `evaluate(...)` again.
- [ ] **B3. Pymoo RNG state.** Currently *not* serialized in `load_nsga3_frontier_campaign_artifacts`. The original resume design is binary (load complete prior run or run fresh — no partial-generation continuation), so this is moot today but becomes a blocker if a partial-generation resume path is added. Decision: keep binary resume; do not add partial-generation continuation. Document this in the engine module docstring.
- [ ] **B4. 6th hypervolume call site.** `_ArchiveTrackingCallback` calls `frontier_archive_hypervolume` per generation (`frontier_engine_nsga3.py:153-156` in `b85e4f12b^`). The lru_cache landed in r7 (8.1) makes this cheap when the input set repeats, but per-generation populations differ, so cache hits will be rare. Consider whether per-generation HV reporting is worth the cost; if yes, document; if no, drop the per-generation HV from the callback and emit only at termination.
- [ ] **B5. Sign-flip-only normalization in `objective_vector_for_minimization`.** Audited as correct in r6.2 but lacks a property test. Add `test_objective_vector_for_minimization_round_trip` asserting that maximize axes are negated and minimize axes are passed through.
- [ ] **B6. `frontier_reference_mode` constraint.** NSGA-III previously implied a Das-Dennis lattice mode; multilane_local implies the legacy share sweep. Document the valid (engine, reference-mode) pairs as a typed mapping or a runtime check; do not encode as a comment.

### 3.5 Evaluator status

`frontier_evaluator.py` was retired in r26 as orphaned. Two options:

- **Option E1 (preferred): keep it retired.** The evaluator was a cache layer that no production code imported by 2026-05-08. NSGA-III's per-evaluation entry point is `_FrontierNSGA3Problem.evaluate`, which calls `single_stage_banana_example.run_alm` directly. Restoring the evaluator means restoring a cache that has no consumer.
- **Option E2: restore if benchmarking shows the cache pays.** Only choose this after running an A2 baseline and finding that re-evaluation cost is the bottleneck. Restore the evaluator + its cache tests in a follow-up commit, not in the revival commit.

Default is E1 unless E2 is justified by data.

---

## 4. Benchmark gate (`D1.B` — must pass before merging the revival)

### 4.1 Required test

```
tests/geo/test_frontier_engine_benchmark.py
```

Asserts: `NSGA3Engine` reaches at least one Pareto member that `MultilaneLocalEngine + achievement_chebyshev_full_simplex_v1` cannot reach under equal evaluation budget. Member-set comparison uses non-dominated front of the union; the NSGA-III-only set must be non-empty.

### 4.2 Fixture

Synthetic 4-D test problem with a known Pareto frontier whose extreme regions are reachable only by population diversity. Use a closed-form objective (no ALM call) — keep test wall-time under 30 seconds. The integration-with-real-ALM benchmark belongs in a separate, manually-triggered campaign artifact (see A1).

### 4.3 If the benchmark fails

Revert the revival. Reverting is single-commit (`git revert <revival-commit>`); leave this plan parked, mark the failed attempt in §6 (revision log), and update the bloat-reduction plan's `D1.4` entry to record that an external NSGA-III claim was provided but did not benchmark above multilane_local.

---

## 5. Done criteria

- [ ] **DC.1** Engine restored: `rg "nsga3" examples/single_stage_optimization/banana_opt/ tests/geo/` returns hits in source code (the inverse of the bloat-reduction plan's DC.2)
- [ ] **DC.2** Real `FrontierEngine` Protocol exists in `frontier_engine_protocol.py` with both engines as conforming implementations
- [ ] **DC.3** Runner dispatches via `ENGINE_REGISTRY`, not `if/else`
- [ ] **DC.4** Benchmark `tests/geo/test_frontier_engine_benchmark.py` passes
- [ ] **DC.5** `pymoo` is a documented optional extra (`pyproject.toml [project.optional-dependencies]`)
- [ ] **DC.6** Validator accepts `nsga3` as a `SUPPORTED_FRONTIER_ENGINES` value AND removes those field names from `DELETED_FRONTIER_SUMMARY_FIELDS` guard
- [ ] **DC.7** B1–B6 from §3.4 are all closed with regression tests
- [ ] **DC.8** Existing `multilane_local` campaign smoke is unaffected (no regression on the r1-r50 work)
- [ ] **DC.9** `bloat-reduction-plan.md` `D1.4` entry updated to record the trigger artifact and benchmark verdict

---

## 6. Out of scope (explicit non-goals)

- Restoring `frontier_evaluator.py` cache without benchmarking evidence (see §3.5)
- Adding new engines (MOEA/D, Bayesian/Kriging surrogates) — separate plan if needed
- Changing the SSOT, DRY, SOLID, immutable-frontier-dataclass, slots=True discipline established in r1-r50
- Re-introducing the `frontier_engine_base.py` filename — that file was a JSON-state holder mislabeled as an engine ABC; the rename to `frontier_progress_state.py` stays
- Re-introducing the deleted `frontier_engine_multilane_local.py` — its lane-spec generator now lives in `frontier_scalarization.py` and stays there
- Partial-generation NSGA-III resume — see B3

---

## 7. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Revival passes A1–A4 but benchmark fails | Medium | Low | §4.3 — revert, document, leave plan parked |
| Pymoo version drift breaks the lazy-import path | Low | Medium | Pin `pymoo>=0.6,<X` in the optional extra; add CI matrix entry that exercises both pymoo-present and pymoo-absent paths |
| Benchmark fixture is gameable (NSGA-III only "wins" because of fixture bias) | Medium | High | Use a published multi-objective benchmark suite (e.g., DTLZ family from Deb-Thiele-Laumanns-Zitzler 2002) rather than an ad-hoc synthetic; document the chosen DTLZ instance and why |
| Restoring NSGA-III code re-introduces a hard runtime dep on pymoo | Low | Medium | DC.5 + the no-pymoo CI matrix entry catch this |
| The r6.2 silent-pass paths (B1–B6) are restored without their fixes | Medium | High | Each of B1–B6 has its own checkbox + regression test in DC.7; reviewer must verify each |
| Wall-time budget overrun (NSGA-III campaigns take days) | High | Low | A4 + explicit sign-off; document expected wall-time in revival commit message |

---

## 8. Suggested commit / PR structure

One PR. Title: `frontier: revive NSGA-III with engine protocol + benchmark gate`. The revival should land as a single reviewable unit, not split across multiple PRs, because the protocol abstraction, the engine restoration, and the benchmark gate are interdependent — any subset is unsafe to merge alone.

Commit body must include:
- Trigger reference (artifact path or research-direction document)
- A2 baseline result (multilane_local result that motivated revival)
- Benchmark verdict (DC.4 result)
- Wall-time budget approved by §A4
- Explicit list of B1–B6 fixes

---

## 9. References

- `docs/frontier_mode_bloat_reduction_todo_plan_2026-05-07.md` — the deletion plan; sections 4.1 (Phase 1A) and 4.2 (Phase 1B QUARANTINE) cover the original keep/drop analysis
- `b85e4f12b` — the deletion commit; revert / cherry-pick base
- Deb & Jain, "An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting Approach, Part I" (IEEE TEVC 18(4), 2014) — algorithm reference
- Deb-Thiele-Laumanns-Zitzler, "Scalable Test Problems for Evolutionary Multi-Objective Optimization" (TIK-Report 112, ETH Zürich, 2002) — DTLZ benchmark suite
- pymoo NSGA-III docs: https://pymoo.org/algorithms/moo/nsga3.html
- Giuliani, Wechsung, Cerfon, Stadler, Landreman, J. Comput. Phys. 459 (2022) 111147 — supports gradient-based ALM (the multilane_local approach), not population-based search on DOFs

---

## 10. Revision log

- 2026-05-09 r1 — initial draft after the r1-r50 bloat-reduction landing committed at `b85e4f12b`. Plan is dormant until at least one of T1–T3 is met.

---

**End of plan.** Status remains "not started" until a trigger fires.
