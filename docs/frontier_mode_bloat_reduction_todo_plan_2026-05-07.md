# Frontier Mode Bloat Reduction — TODO Plan

**Date:** 2026-05-07 (revised same day per Codex review)
**Branch:** surrogate-confinement-v2
**Status:** Proposed; awaiting NSGA3 production-evidence decision before execution
**Owner:** TBD
**Estimated effort:** ~3-5 focused days; can be split across PRs

**Revision log:**
- 2026-05-07 r1 — initial draft
- 2026-05-07 r2 — Codex review fixes: expanded NSGA3 test deletion list (Phase 1A); fixed runnable smoke command + correct summary filename (CC.3, CC.4); reframed hypervolume memoization acceptance (Phase 5.1); removed unsafe archive-metadata side-map task (Phase 6.2.3); dropped YAGNI schema-helper extraction (Phase 4.2); reworked lazy pymoo plan (Phase 5.3); split LOC done-criteria (DC.11); corrected importer counts in F1, F9
- 2026-05-07 r3 — Codex r2 review fixes: removed dropped Phase 4.2 reference from PR5 title; reworded Phase 1A.3 to "delete the optional NSGA3 summary payload validator entirely" (it's called unconditionally, not engine-gated); added PYTHONPATH/isolated-venv guidance to lazy-pymoo verification commands (Phase 5.3.1, 5.3.4) and removed destructive `pip uninstall pymoo` for local use; corrected `test_summary_validator_accepts_optional_nsga3_fields` LOC estimate from ~50 to ~127 (lines 189-315); reframed hypervolume acceptance to allow for prefix-archive sweeps in `build_frontier_hypervolume_history`
- 2026-05-07 r4 — Codex r3 review fixes: expanded Phase 1A.3 NSGA3 field list from 2 to all 6 optional fields (`frontier_generation_history`, `_path`, `frontier_engine_stats`, `frontier_evaluator_spec`, `_path`, `frontier_population_checkpoint_path`) per `frontier_contracts.py:446-489`; reframed Phase 5.1 wall-clock estimate from "~3× factor" to "up to ~3× on repeated final-archive calls; lower when prefix-history or leave-one-out calls dominate"
- 2026-05-07 r5 — Codex r4 review fix: synchronized D2.1 with Phase 1A.4 — D2.1 had stale "test_frontier_evaluator.py:111-275 / ~165 LOC" reference; now lists all 4 NSGA3-tied test blocks across 3 files totaling ~612 LOC, with cross-reference to Phase 1A.4 as canonical source

---

## 1. Context & Motivation

### 1.1 Why this plan exists
The `frontier` campaign mode under `examples/single_stage_optimization/banana_opt/frontier_*.py` (14 modules, **6,435 LOC**) plus its runner `run_single_stage_frontier_campaign.py` (**1,101 LOC**) and tests (**3,207 LOC**) accumulated through four overlapping plan iterations:

- v1 plan (2026-04-12, 407 LOC): scalarized objective using existing ALM
- v4 plan (2026-04-13, 883 LOC): campaign optimizer + Pareto archive + recommendation policies
- Global Pareto plan (2026-04-22, 493 LOC): adds NSGA-III engine
- Gradient-contract plan (2026-04-26, 952 LOC): retroactive correction for gradient/value mismatch caused by NSGA-III layering

The audit (parallel multi-agent code review, 2026-05-07) and a second independent validation (Codex review, same day) agree: the subsystem is **moderately bloated**, with ~15-20% mechanically removable LOC, and the bloat is concentrated in **structural** issues rather than line-by-line padding.

### 1.2 What this plan is NOT
- It is not a rewrite. Frontier mode delivers real, currently-used Pareto-front exploration via the `multilane_local` engine.
- It is not a deletion sweep. The support contract layer (`frontier_contracts`, `frontier_constraints`, `frontier_conditioning`, `frontier_dominance`, `frontier_recommendation`) is largely well-scoped (~4% removable) and stays.
- It is not an algorithmic redesign. The Pareto-front method, ALM inner solver, and CLI surface stay functionally equivalent.

### 1.3 Guiding principles (apply to every task)
- **KISS** — prefer one obvious path over multiple parametric ones with one caller each
- **YAGNI** — drop code paths without evidence of production use; do not build for hypothetical future engines
- **DRY** — same numeric/schema contract written exactly once
- **SSOT** — frontier code lives in `frontier_*` modules; non-frontier code does not depend on `frontier_*`
- **SOLID** — name files for what they contain (no `engine_base.py` without an engine ABC); SRP at module level
- **Functional programming** — frozen dataclasses + pure transformations + map dispatch over if-chains
- **Immutable** — `frozen=True, slots=True` for all data classes; replace mutator methods with state-returning functions
- **Memory-safe / thread-safe** — preserve existing locks; remove unnecessary mutable shared state
- **Memory-efficient** — eliminate redundant copies and recomputations
- **Performant** — memoize expensive O(N²) recomputes; avoid eager imports of optional heavy deps

---

## 2. Verified Findings (cited evidence)

The following claims have been re-validated against the working tree at HEAD. Each file:line reference resolves to the current code.

| # | Finding | Evidence | Severity |
|---|---|---|---|
| F1 | False engine layer — `frontier_engine_base.py` defines no ABC/Protocol; runner uses hard-coded `if/else` dispatch. Module has 2 importers (1 production + 1 test) — original audit's "6 importers" was symbol-count, not file-count | `frontier_engine_base.py:28` (state dataclasses, no ABC); `run_single_stage_frontier_campaign.py:38, 916-942`; `tests/geo/test_frontier_archive.py:25` | High (SOLID/SRP) |
| F2 | NSGA3 path is structurally suspicious — wired in CLI/runner/tests but no validated production artifact found | `frontier_engine_nsga3.py:33-44, 161`; `run_single_stage_frontier_campaign.py:124, 916-942, 1072-1094`; cited in `autoresearch/program_hbt_topology_surrogate_legacy_vmec.md:310` but no completed `frontier_engine: nsga3` JSON/JSONL artifact | High (YAGNI) |
| F3 | Frontier scalarization leaked into central objective module — wrong dependency direction | `single_stage_objectives.py:255-558` contains `apply_frontier_scalarization_override`, `_frontier_chebyshev_goal`, `_frontier_epsilon_penalties`, `_frontier_alm_base_total_grad` | High (SSOT) |
| F4 | `annotate_search_evaluation_finiteness` defined in frontier module is consumed by non-frontier code | defined `frontier_constraints.py:61`; consumed `single_stage_objectives.py:8` and lines 690/740/794/818/1516 | Medium (SSOT) |
| F5 | Epsilon thresholds duplicated across 3 modules | `frontier_archive.py:596-621` (post-hoc certification); `frontier_scalarization.py:633-647` (lane spec writer); `single_stage_objectives.py:539-555` (penalty enforcement) | Medium (DRY) |
| F6 | Hypervolume recomputed without memoization, called ≥3× per campaign, leave-one-out is N× | `frontier_archive.py:446-477` (`annotate_hypervolume_contributions`); also called in serialization (`frontier_archive.py:341`) and reporting (`frontier_campaign_reporting.py:303, 424`) | Medium (Performant) |
| F7 | Runtime calibration registry has 2 profiles differing by 1 integer + 1 string | `frontier_runtime_calibration.py:11-272`; profiles `reduced_fixture_v1` and `canonical_seed_v1` differ in `early_stop_patience_lanes` (2 vs 3) | Low (KISS) |
| F8 | `frontier_dominance.py` is misnamed (~85% Pareto normalization, ~15% dominance) | `frontier_dominance.py:8-72` (normalization rules); only `frontier_dominance.py:153-203` (~50 LOC) is dominance | Low (SOLID/naming) |
| F9 | `frontier_engine_multilane_local.py` is 79 LOC — 1 dataclass + 1 weight-share generator, not an engine. Has 3 importers (runner, reporting, scalarization) | `frontier_engine_multilane_local.py`; importers `run_single_stage_frontier_campaign.py:48`, `frontier_campaign_reporting.py:38`, `frontier_scalarization.py:8` | Low (SOLID/naming) |
| F10 | Hand-rolled JSON ser/de boilerplate (~150 LOC) across multiple dataclasses | `frontier_evaluator.py:70-86, 106-156, 207-235`; `frontier_engine_base.py:28-191`; `frontier_archive.py:173-205` | Low (DRY) |
| F11 | Defensive `getattr(args, ..., default)` against `argparse.Namespace` (always set by `add_argument(default=...)`) | `frontier_campaign_reporting.py:181-204, 246-255, 412-421` | Low (KISS) |
| F12 | Duplicate `--frontier-hypervolume-reference` and `runtime_defaults` resolution | `run_single_stage_frontier_campaign.py:762-771 vs 827-837`; `:874-878 vs 1035-1039` | Low (KISS) |
| F13 | `# noqa: F401` import marks unused symbol that is in fact used elsewhere | `run_single_stage_frontier_campaign.py:50` imports `generate_multilane_local_specs` (unused at runner level; actually used via `frontier_scalarization.py:714`) | Trivial |
| F14 | `pymoo` is a soft optional runtime requirement, not a declared dep | `frontier_engine_nsga3.py:33-44` (try/except ImportError); not in `pyproject.toml` / `requirements.txt` / `setup.py` | Note (correction to prior claim) |
| F15 | `_require_*` schema helpers in `frontier_contracts.py` are NOT pre-existing in artifact_contracts; they are first-occurrence | `frontier_contracts.py:491-518`; no `_require_*` in `artifact_contracts.py` / `constraint_contract.py` / etc. | Note (correction to prior claim) |
| F16 | `_load_banana_opt_module`/`EXAMPLES_ROOT` is in 1 test file, not 6; but importlib bootstrap pattern repeats across 5 files | only `test_frontier_evaluator.py:16, 25`; other 5 use small `importlib.import_module(...)` wrappers | Note (correction to prior claim) |

---

## 3. Decision Log (fill before executing Phase 1)

### Decision D1 — Keep or drop NSGA3?

**Question:** Is `--frontier-engine nsga3` exercised by any production campaign?

**Investigation TODO:**
- [ ] **D1.1** Grep all completed JSONL/JSON artifacts in `autoresearch/` for `"frontier_engine": "nsga3"`
  ```bash
  rg -l '"frontier_engine":\s*"nsga3"' /Users/suhjungdae/code/columbia/autoresearch/ 2>/dev/null
  ```
- [ ] **D1.2** Grep this repo's `tmp/` and `examples/single_stage_optimization/outputs_*` for the same string
  ```bash
  rg -l '"frontier_engine":\s*"nsga3"' tmp/ examples/single_stage_optimization/outputs_* 2>/dev/null
  ```
- [ ] **D1.3** Check lab notes / Stellaris VM run history for `--frontier-engine nsga3` invocations
- [ ] **D1.4** Ask: does anyone on the team have a campaign result that requires NSGA3 (Pareto coverage NOT reachable by `multilane_local` + ε-constraint)?

**Decision criteria:**
- [ ] D1.A — Drop NSGA3 if zero validated production artifacts exist (most likely outcome based on prior audit)
- [ ] D1.B — Keep NSGA3 only if (a) production artifact exists AND (b) there's a benchmark showing NSGA3 reaches Pareto regions multilane_local cannot

**Decision recorded here:** _TBD — fill in: DROP / QUARANTINE / KEEP_

If **DROP** → execute Phase 1A
If **QUARANTINE/KEEP** → execute Phase 1B (define real Engine Protocol + benchmark gate)

### Decision D2 — Are NSGA3-targeted tests fixtures or production validation?

- [ ] **D2.1** If D1 = DROP, plan to delete **all NSGA3-tied tests enumerated in Phase 1A.4** — 4 blocks across 3 test files totaling **~612 LOC**:
  - `tests/geo/test_frontier_evaluator.py:111-275` (~165 LOC) + helper at lines 34-35
  - `tests/geo/test_single_stage_workflow_helpers.py:4986` (~140 LOC)
  - `tests/geo/test_single_stage_workflow_helpers.py:5140` (~180 LOC)
  - `tests/geo/test_frontier_contracts.py:189-315` (~127 LOC)
  - See Phase 1A.4 (Section 4.1) for the canonical enumeration; this D2.1 entry must stay in sync with it
- [ ] **D2.2** If D1 = KEEP, plan to add a benchmark test that proves NSGA3 reaches at least one Pareto member that `multilane_local` cannot under equal evaluation budget

### Decision D3 — Preserve frontier_runtime_calibration JSON shape?

- [ ] **D3.1** Search consumed-by graph for any external tool that reads `frontier_runtime_calibration` block from a frontier summary JSON
  ```bash
  rg "frontier_runtime_calibration" --type json --type py 2>/dev/null
  ```
- [ ] **D3.2** If consumers exist outside this repo (lab notebooks, dashboards), preserve schema_version + JSON keys when collapsing the registry

---

## 4. Phase 1 — NSGA3 decision (gates everything else)

### 4.1 Phase 1A — DROP NSGA3 (preferred path)
**Acceptance criterion:** `grep -rn "nsga3\|NSGA3\|frontier_engine_nsga3" examples/ tests/` returns zero hits in source code (only string mentions in retired docs allowed).

- [ ] **1A.1** Delete `examples/single_stage_optimization/banana_opt/frontier_engine_nsga3.py` (372 LOC)
  - Pre-check: confirm no symbol from this module is imported anywhere except the runner
  ```bash
  rg "from .frontier_engine_nsga3|from banana_opt.frontier_engine_nsga3" --type py
  ```
- [ ] **1A.2** Remove NSGA3 dispatch in runner
  - File: `examples/single_stage_optimization/run_single_stage_frontier_campaign.py`
  - Lines: 52-56 (imports), 130 (CLI choices), 916-942 (engine branch), 1072-1094 (summary post-processing)
  - Replace `--frontier-engine` choices with single literal or remove flag entirely if only one mode remains
- [ ] **1A.3** Remove the optional NSGA3 summary payload validator and **all six** optional NSGA3 summary fields entirely
  - File: `examples/single_stage_optimization/banana_opt/frontier_contracts.py`
  - Function: `_validate_optional_nsga3_summary_payload` (defined line 446) — **delete the function**
  - Call site: `:416` calls it unconditionally (it's "validate optional fields if present", not gated on engine name) — **delete the call**
  - Remove **all six optional NSGA3 summary fields** the function validates from any payload contracts, JSON ser/de, runner/reporting writers that produce them, and any tests that expect them. The complete field list (per `frontier_contracts.py:446-489`):
    1. `frontier_generation_history` (list of per-generation telemetry mappings)
    2. `frontier_generation_history_path` (string path to history JSON)
    3. `frontier_engine_stats` (mapping)
    4. `frontier_evaluator_spec` (mapping with `schema_version`, `run_identity`)
    5. `frontier_evaluator_spec_path` (string path)
    6. `frontier_population_checkpoint_path` (string path)
  - Also clean up callers that *write* these fields:
    - `run_single_stage_frontier_campaign.py:1072-1094` (NSGA3 summary post-processing — already targeted by 1A.2)
    - `frontier_campaign_reporting.py` if it emits any of these (grep below)
    - Any per-engine artifact writer in `frontier_engine_nsga3.py` (already deleted in 1A.1)
  - Verify clean removal:
    ```bash
    rg -n "frontier_generation_history|frontier_engine_stats|frontier_evaluator_spec|frontier_population_checkpoint" \
       examples/single_stage_optimization/ tests/geo/
    ```
    Expected: zero hits in source (mentions in retired `docs/single_stage_frontier_global_pareto_plan_*` are fine).
- [ ] **1A.4** Delete ALL NSGA3-tied tests across the test suite
  - **Verification grep first:** `rg -n "nsga3|NSGA3" tests/geo/` — confirm full list before deletion
  - File: `tests/geo/test_frontier_evaluator.py`
    - Block: `test_nsga3_population_checkpoint_uses_final_population_arrays` (line 111, ~165 LOC)
    - Helper: `load_frontier_engine_nsga3_module` (lines 34-35)
  - File: `tests/geo/test_single_stage_workflow_helpers.py`
    - Block: `test_frontier_campaign_nsga3_records_generation_summary` (line 4986, ~140 LOC)
    - Block: `test_frontier_campaign_nsga3_resume_reuses_saved_engine_artifacts` (line 5140, ~180 LOC)
  - File: `tests/geo/test_frontier_contracts.py`
    - Block: `test_summary_validator_accepts_optional_nsga3_fields` (lines 189-315, **~127 LOC** — measured by next-method offset; previously misstated as ~50)
  - **Estimated test LOC removed in 1A.4:** ~612 (165 + 140 + 180 + 127), revised up from prior 535
- [ ] **1A.5** Update or retire `docs/single_stage_frontier_global_pareto_plan_2026-04-22.md` (mark superseded; keep as historical record)
- [ ] **1A.6** Update `docs/single_stage_frontier_gradient_contract_impl_plan_2026-04-26.md` — much of its motivation was NSGA3 gradient consistency; mark sections that no longer apply
- [ ] **1A.7** Notify `autoresearch/program_hbt_topology_surrogate_legacy_vmec.md:310` consumers; remove or update reference
- [ ] **1A.8** Remove `# noqa: F401` import of `generate_multilane_local_specs` at `run_single_stage_frontier_campaign.py:50` (duplicate; real consumer is `frontier_scalarization.py:714`)
- [ ] **1A.9** Run full test suite; expect green
  ```bash
  ./run_tests tests/geo/test_frontier_*.py
  ```
- [ ] **1A.10** Confirm runner still passes a smoke run with `multilane_local` (the only remaining engine)

**Estimated LOC removed in Phase 1A:** ~600 source + ~612 test = ~1,212 LOC (revised across r1→r2→r3 as undercounted test sites and per-block LOC were verified)

### 4.2 Phase 1B — QUARANTINE NSGA3 (only if D1 = KEEP)
**Acceptance criterion:** there exists a real `Engine` Protocol with both `multilane_local` and `nsga3` as conforming implementations; runner dispatch is via the protocol, not `if/else`; a benchmark test proves NSGA3 utility.

- [ ] **1B.1** Define `frontier_engine_protocol.py` (new file, ~50 LOC)
  ```python
  from typing import Protocol, runtime_checkable

  @runtime_checkable
  class FrontierEngine(Protocol):
      def run(self, spec: SingleStageFrontierEvaluatorSpec, budget: int, ...) -> EngineArtifacts: ...
      def load_artifacts(self, path: Path) -> EngineArtifacts | None: ...
      def name(self) -> str: ...
  ```
- [ ] **1B.2** Refactor `frontier_engine_nsga3.py` to expose a class implementing `FrontierEngine`
- [ ] **1B.3** Wrap the inline `multilane_local` execution path (currently in runner) into a class implementing `FrontierEngine`
- [ ] **1B.4** Replace runner `if/else` (lines 916-942) with a registry lookup
  ```python
  ENGINE_REGISTRY: Mapping[str, Callable[[], FrontierEngine]] = MappingProxyType({
      "multilane_local": MultilaneLocalEngine,
      "nsga3": NSGA3Engine,
  })
  engine = ENGINE_REGISTRY[args.frontier_engine]()
  ```
- [ ] **1B.5** Add `tests/geo/test_frontier_engine_benchmark.py` — must show NSGA3 reaches a Pareto member that `multilane_local` cannot under equal budget
- [ ] **1B.6** Move `pymoo` from optional try/except to a documented optional extra (e.g., `pyproject.toml` `[project.optional-dependencies]`: `nsga3 = ["pymoo>=0.6"]`)

---

## 5. Phase 2 — SSOT restoration (independent of Phase 1)

### 5.1 Move frontier scalarization back to `frontier_scalarization.py`

**Acceptance criterion:** `grep -n "frontier" examples/single_stage_optimization/banana_opt/single_stage_objectives.py` returns matches only for *consumer* references (e.g., `if args.use_frontier_scalarization:` calls into `frontier_scalarization.apply_frontier_scalarization_override(...)`).

- [ ] **2.1.1** Audit `single_stage_objectives.py:255-558` — list each function/symbol that mentions frontier
  ```bash
  rg -n "frontier" examples/single_stage_optimization/banana_opt/single_stage_objectives.py
  ```
- [ ] **2.1.2** Move `apply_frontier_scalarization_override` (line 282) to `frontier_scalarization.py`
- [ ] **2.1.3** Move `_frontier_chebyshev_goal` (line 473) to `frontier_scalarization.py` as private helper
- [ ] **2.1.4** Move `_frontier_epsilon_penalties` (line 533) to `frontier_scalarization.py` (will be re-merged in Phase 4 with archive's epsilon code)
- [ ] **2.1.5** Move `_frontier_alm_base_total_grad` (line 463) — careful: this touches gradient computation; ensure JAX function purity is preserved
- [ ] **2.1.6** Update import sites — `single_stage_objectives.py` should `from .frontier_scalarization import apply_frontier_scalarization_override` (one-line import) where it now defines the body
- [ ] **2.1.7** Verify dependency direction is clean
  ```bash
  rg -n "from .frontier|from banana_opt.frontier" examples/single_stage_optimization/banana_opt/single_stage_*.py
  ```
- [ ] **2.1.8** Run gradient-contract tests to confirm no regression
  ```bash
  ./run_tests tests/geo/test_single_stage_alm_integration.py tests/geo/test_frontier_evaluator.py
  ```

### 5.2 Move generic finiteness helper out of `frontier_constraints.py`

**Context:** `annotate_search_evaluation_finiteness` (`frontier_constraints.py:61`) is used by non-frontier code (`single_stage_objectives.py:8` and lines 690/740/794/818/1516). It is misnamed/misplaced.

- [ ] **2.2.1** Create new module `examples/single_stage_optimization/banana_opt/search_evaluation.py`
- [ ] **2.2.2** Move `annotate_search_evaluation_finiteness` + its supporting tables (`_FINITE_SCALAR_FIELDS`, `_FINITE_VECTOR_FIELDS` at `frontier_constraints.py:17-56`) to `search_evaluation.py`
- [ ] **2.2.3** Update all 6+ import sites to import from `search_evaluation` instead of `frontier_constraints`
- [ ] **2.2.4** Verify with grep that `frontier_constraints` no longer leaks into non-frontier code
  ```bash
  rg "from .frontier_constraints|from banana_opt.frontier_constraints" examples/ | grep -v frontier_
  ```
  Expected: zero hits.

**Estimated LOC moved (not deleted):** ~50 LOC. Net repo LOC change: 0. SSOT restoration: yes.

---

## 6. Phase 3 — Structural renames (low risk, high readability)

### 6.1 Rename `frontier_engine_base.py` → `frontier_progress_state.py`

**Context:** F1 — file contains zero engine abstraction; it holds JSON ser/de for `FrontierLaneContract`/`FrontierLaneRecord`/`FrontierCampaignProgress`.

- [ ] **3.1.1** `git mv examples/single_stage_optimization/banana_opt/frontier_engine_base.py examples/single_stage_optimization/banana_opt/frontier_progress_state.py`
- [ ] **3.1.2** Update all 6 importers (per audit, this is a hub module)
  ```bash
  rg -l "frontier_engine_base" examples/ tests/ | xargs sed -i '' 's/frontier_engine_base/frontier_progress_state/g'
  ```
  (verify on macOS `sed -i ''` syntax; use a portable Python script if uncertain)
- [ ] **3.1.3** Update test file `tests/geo/test_frontier_archive.py:25` reference
- [ ] **3.1.4** Run frontier test suite to confirm imports resolve
  ```bash
  ./run_tests tests/geo/test_frontier_*.py
  ```

### 6.2 Fold `frontier_engine_multilane_local.py` into `frontier_scalarization.py`

**Context:** F9 — the file is 79 LOC, 1 dataclass + 1 weight-share generator, only consumer is `frontier_scalarization.py:714`. It is not an engine.

- [ ] **3.2.1** Move `FrontierLaneSpec` dataclass and `generate_multilane_local_specs` function to top of `frontier_scalarization.py`
- [ ] **3.2.2** Update importers — sites currently importing `FrontierLaneSpec` from `frontier_engine_multilane_local` should import from `frontier_scalarization`
  ```bash
  rg "from .frontier_engine_multilane_local|from banana_opt.frontier_engine_multilane_local" examples/ tests/
  ```
- [ ] **3.2.3** Delete `frontier_engine_multilane_local.py`
- [ ] **3.2.4** Remove the redundant `# noqa: F401` import in `run_single_stage_frontier_campaign.py:50` (already covered in 1A.8 if Phase 1A executes; otherwise do here)

### 6.3 Rename `frontier_dominance.py` → `frontier_pareto_normalization.py`

**Context:** F8 — only ~50 of 373 LOC are dominance logic; the rest is Pareto normalization.

- [ ] **3.3.1** Decide whether to rename or keep — rename is honest but causes 6 importer updates and minor diff churn
- [ ] **3.3.2** If renaming: `git mv` + bulk replace; if not: add a one-line module docstring explaining the actual contents

**Phase 3 net LOC:** 0 deleted; ~3 import updates per renamed module. **Mental model improvement: significant.**

---

## 7. Phase 4 — DRY / SSOT consolidations

### 7.1 Centralize epsilon thresholds (F5)

**Acceptance criterion:** the strings `epsilon_constraint_qa_max`, `epsilon_constraint_boozer_max` (and similar threshold keys) appear exactly once in source — inside a frozen dataclass in `frontier_contracts.py`.

- [ ] **7.1.1** Add to `frontier_contracts.py`:
  ```python
  @dataclass(frozen=True, slots=True)
  class EpsilonThresholds:
      qa_max: float
      boozer_max: float

      @classmethod
      def from_rerun_contract(cls, contract: Mapping[str, object]) -> "EpsilonThresholds":
          return cls(
              qa_max=float(contract["epsilon_constraint_qa_max"]),
              boozer_max=float(contract["epsilon_constraint_boozer_max"]),
          )

      def as_payload(self) -> dict[str, float]:
          return {
              "epsilon_constraint_qa_max": self.qa_max,
              "epsilon_constraint_boozer_max": self.boozer_max,
          }
  ```
- [ ] **7.1.2** Replace 3 hand-rolled implementations:
  - `frontier_archive.py:596-621` (`_evaluate_epsilon_constraint_status`) — consume `EpsilonThresholds.from_rerun_contract`
  - `frontier_scalarization.py:633-647` — write via `EpsilonThresholds.as_payload`
  - `single_stage_objectives.py:539-555` (`_frontier_epsilon_penalties`) — consume `EpsilonThresholds.from_rerun_contract`
  - Note: if Phase 2 already moved `_frontier_epsilon_penalties` into `frontier_scalarization.py`, only 2 sites remain in this step
- [ ] **7.1.3** Add unit test: `test_epsilon_thresholds_round_trip` in `tests/geo/test_frontier_contracts.py`
- [ ] **7.1.4** Verify no string-key duplication remains
  ```bash
  rg '"epsilon_constraint_(qa|boozer)_max"' examples/single_stage_optimization/banana_opt/
  ```
  Expected: matches only inside `frontier_contracts.py`.

**Estimated LOC removed:** ~60.

### 7.2 ~~Extract shared schema helpers~~ **DROPPED — YAGNI (per Codex r2 review)**

**Original rationale:** preemptively extract `_require_*` helpers from `frontier_contracts.py:491-518` so future contract modules can import them.

**Why dropped:**
- Original audit claimed these helpers duplicate logic in `artifact_contracts.py`. That was wrong (F15) — they're first-occurrence.
- No second consumer exists today. Extracting saves ~0 LOC.
- This plan's own guiding principle (YAGNI) prohibits building for hypothetical future modules.
- Action: **leave the helpers where they are** in `frontier_contracts.py:491-518`. If a second contract module ever needs the same idiom, extract at that point.

### 7.3 Collapse parallel "best-of" extractors (audit S3.4)

**Context:** 5 implementations of lex-max sort with different gate rules across `frontier_archive.archive_best_by_metric` (archive.py:312), `_metric_sort_key` (archive.py:584), and 4 policies in `frontier_recommendation.py:210-279`.

- [ ] **7.3.1** Add a parameterized `select_best` to `frontier_recommendation.py`
  ```python
  def select_best(
      members: Sequence[FrontierArchiveMember],
      *,
      sort_priority: Sequence[tuple[str, Direction]],
      gate: Callable[[FrontierArchiveMember], bool],
      rationale: str,
  ) -> Recommendation: ...
  ```
- [ ] **7.3.2** Build a frozen `RECOMMENDATION_POLICIES: Mapping[PolicyName, Callable] = MappingProxyType({...})` registry where each value is a `partial(select_best, ...)`
- [ ] **7.3.3** Delete the 4 near-duplicate per-policy functions
- [ ] **7.3.4** Update `frontier_archive.archive_best_by_metric` to call `select_best` with empty gate

**Estimated LOC removed:** ~60. **Functional discipline: higher.**

---

## 8. Phase 5 — Performance / Memory

### 8.1 Memoize hypervolume across reporting + serialization (F6, **revised per Codex r2**)

**Context (revised):** The current `annotate_hypervolume_contributions` (`frontier_archive.py:446-477`) does 1 base call + N leave-one-out calls per pass. Each leave-one-out call has a *different* member-set, so an `lru_cache` produces zero hits within a single annotation pass. The real win is across **repeated** invocations of the same hypervolume on the same archive (annotation, then serialization at line 341, then reporting at `frontier_campaign_reporting.py:303, 424`).

**Acceptance criterion (revised per r3):**
- Within a single campaign workflow (annotate → serialize → report → hypervolume_history), the total number of underlying hypervolume computations is **no more than the number of unique (archive-member-set, reference) inputs** seen across all call sites — not 3× that count as currently happens.
- Note: `build_frontier_hypervolume_history` (`frontier_campaign_reporting.py:281`) iterates lane records building a *running prefix archive* and calls hypervolume on each prefix, so each prefix is its own unique input — the cache cannot collapse those into a single call. The win is in collapsing **repeated** identical inputs (final-archive hypervolume called from annotate + serialize + report), not in shrinking the prefix sweep.
- Verified by an instrumentation counter in a benchmark test, not by `lru_cache` cache-info hits alone.
- Optional: redesign leave-one-out using inclusion-exclusion (incremental hypervolume) to drop the O(N) leave-one-out cost — but this is algorithmic redesign, mark as separate optional task.

- [ ] **8.1.1** Identify the function `frontier_archive_hypervolume` and confirm its inputs are hashable
  - File: `frontier_archive.py` (find via `grep -n "def frontier_archive_hypervolume"`)
- [ ] **8.1.2** Wrap with `lru_cache` keyed on tuple-of-tuples of objective vectors + reference point. Confirm tuple is canonical (sorted by `member_id` or similar) so equivalent archives hash identically.
  ```python
  @functools.lru_cache(maxsize=128)
  def _hypervolume_cached(
      objective_vectors: tuple[tuple[float, ...], ...],
      reference: tuple[float, ...],
  ) -> float: ...
  ```
- [ ] **8.1.3** Refactor `annotate_hypervolume_contributions` to call the cached version
- [ ] **8.1.4** Add a benchmark test using a per-call counter (not just `cache_info`) that asserts: across one annotation pass + one serialize + one report + one history build, the number of *underlying* `_hypervolume_cached` invocations equals the number of unique input tuples encountered — typically (1 final-archive call) + (N leave-one-outs) + (M unique prefix archives from history) — and that the same final-archive computation does not repeat across annotate/serialize/report.
- [ ] **8.1.5** Verify reporting paths (`frontier_campaign_reporting.py:303, 424`) hit the cache (cache_hits ≥ 2 after annotation populates it)
- [ ] **8.1.6** (Optional) Investigate inclusion-exclusion incremental hypervolume to drop the per-pass O(N) leave-one-out cost. If pursued, becomes a separate algorithmic-redesign task with its own correctness benchmark.

**Estimated wall-clock improvement on long campaigns (revised per r4):** up to ~3× on the *repeated final-archive* hypervolume calls (annotate + serialize + report computing the same final archive). Lower in practice when prefix-history sweeps (`build_frontier_hypervolume_history`) or leave-one-out passes dominate the call count, since those calls have unique inputs and cannot be deduplicated by the cache. The realized factor depends on the ratio (repeated-final-archive calls : unique-input calls) for a given campaign size. Optional 8.1.6 (incremental hypervolume) would deliver an additional O(N) factor by collapsing leave-one-out cost.

### 8.2 Collapse runtime calibration registry (F7, with D3 guard)

**Acceptance criterion:** if D3 = preserve JSON shape, the on-disk `frontier_runtime_calibration` block in summary JSON remains identical for equivalent inputs.

- [ ] **8.2.1** Confirm D3 outcome (D3.1, D3.2 above)
- [ ] **8.2.2** Replace `FRONTIER_RUNTIME_CALIBRATION_PROFILES` registry (2 entries) with a single `FrontierRuntimeDefaults` frozen dataclass + `resolve_runtime_defaults(args)` factory
- [ ] **8.2.3** Preserve serialization shape via a `to_json_dict` method that emits the same keys profiles previously emitted (basis string, etc.)
- [ ] **8.2.4** Delete `_resolve_calibration_profile` and the profile-name CLI argument if D3 allows
- [ ] **8.2.5** Update test `test_frontier_runtime_calibration` to match new shape

**Estimated LOC removed:** ~70-90 (preserving JSON shape) or ~150 (if shape can change).

### 8.3 Lazy `pymoo` import (only if Phase 1B = KEEP NSGA3) — **revised per Codex r2**

**Context (revised):** `frontier_engine_nsga3.py` defines `class _FrontierNSGA3Problem(ElementwiseProblem)` and `class _ArchiveTrackingCallback(Callback)` at module level. These class declarations need pymoo symbols in scope at *import* time. Just moving the import statement into a function body breaks the subclass declarations.

Two viable approaches:

**Approach A — Function-local class definitions:**
- Move both classes inside `run_nsga3_frontier_campaign(...)` (and a similar function for the callback) so their definitions execute only when the function is called
- Pros: true lazy import, no module-level pymoo dependency
- Cons: classes redefined on each call (small overhead); test fixtures that import the classes directly need refactor

**Approach B — Keep current top-level optional import (`try/except ImportError`) + accept it as the lazy mechanism:**
- The try/except at `frontier_engine_nsga3.py:33-44` already returns sentinel `NSGA3 = None` etc. when pymoo absent
- The whole module import succeeds without pymoo; only call-time uses fail
- Pros: minimal change; classes remain top-level
- Cons: `import banana_opt.frontier_engine_nsga3` still attempts pymoo import (but doesn't fail) — RSS savings are smaller

- [ ] **8.3.1** Decide between Approach A and Approach B based on RSS measurement. The runner imports the engine module via `examples/single_stage_optimization` on the path, so the measurement command must set `PYTHONPATH` (or `cd` first):
  ```bash
  PYTHONPATH=examples/single_stage_optimization \
    /usr/bin/time -l python -c "import banana_opt.frontier_engine_nsga3" 2>&1 | tail -5
  ```
  Run with and without `pymoo` available in the same conda env to get RSS deltas. If cold-start cost (with pymoo present) is < 50 ms and < 30 MB RSS, prefer Approach B (smaller diff).
- [ ] **8.3.2 (A)** Move `_FrontierNSGA3Problem` and `_ArchiveTrackingCallback` class definitions inside the functions that need them; ensure no other code references the classes at module level (check tests too)
- [ ] **8.3.2 (B)** Document the existing `try/except ImportError` at module top as the intentional lazy mechanism with a comment
- [ ] **8.3.3** Document `pymoo` as optional extra in `pyproject.toml`:
  ```toml
  [project.optional-dependencies]
  nsga3 = ["pymoo>=0.6"]
  ```
- [ ] **8.3.4** Add an isolated CI job (or pytest marker) that exercises the no-pymoo path. **Do NOT run `pip uninstall pymoo` in a developer's local env.** Use a fresh venv or CI matrix entry:
  ```yaml
  # CI matrix snippet (illustrative — adapt to actual CI config)
  - name: nsga3-not-installed
    setup: pip install -e . --no-deps && pip install <minimal deps without pymoo>
    test: |
      PYTHONPATH=examples/single_stage_optimization \
        python -c "from banana_opt.frontier_engine_nsga3 import _PYMOO_IMPORT_ERROR; assert _PYMOO_IMPORT_ERROR is not None"
  ```
  Or, locally, use a throwaway env: `python -m venv /tmp/no_pymoo && /tmp/no_pymoo/bin/pip install <minimal deps> && /tmp/no_pymoo/bin/python -c "..."`

**Estimated cold-start RSS savings:** Approach A: ~150 MB on default `multilane_local` runs. Approach B: smaller (~0-30 MB depending on pymoo's lazy init). Measure before deciding.

### 8.4 Single-pass resume read

- [ ] **8.4.1** Inspect `load_resume_lane_specs` and `load_resume_manifest` in runner (lines ~635-653)
- [ ] **8.4.2** Combine into one read returning `(lane_specs, manifest)`
- [ ] **8.4.3** Verify resume tests still pass

### 8.5 Single resolution pass for runtime defaults & hypervolume reference

- [ ] **8.5.1** Refactor runner so `runtime_defaults` is resolved once (line 762-771) and reused at line 827-837 (currently re-resolved if lane count mismatches)
- [ ] **8.5.2** Refactor so `hypervolume_reference` is resolved once at lines 874-878 and reused at 1035-1039
- [ ] **8.5.3** If both resolutions genuinely depend on later state, document why with a one-line comment

---

## 9. Phase 6 — Mechanical cleanup (do last)

### 9.1 Evaluator (F10, audit findings on `frontier_evaluator.py`)

- [ ] **9.1.1** Split `build_single_stage_frontier_runtime` (lines 621-925, 305 LOC) into 3 helpers: `_resolve_runtime_inputs`, `_build_objective_bundle`, `_synthesize_spec`
- [ ] **9.1.2** Replace `constraint_violations` if-ladder (lines 535-569) with dict comprehension + 2 special-case overrides
- [ ] **9.1.3** Extract `_results_payload_from(...)` helper consumed by both valid path (lines 579-606) and invalid path (lines 1191-1206)
- [ ] **9.1.4** Inline `from_spec` classmethod alias (lines 301-309)
- [ ] **9.1.5** Drop redundant `except FrontierEvaluatorInitializationError: raise` arm (line 920); add a single comment if needed
- [ ] **9.1.6** Replace `_jsonable_value = single_stage._jsonable_value` re-export (line 1251) with direct `from ... import _jsonable_value`

### 9.2 Frozen dataclasses everywhere

- [ ] **9.2.1** Convert `FrontierCampaignManifest` (`frontier_campaign_reporting.py:68-135`) to `@dataclass(frozen=True, slots=True)`
- [ ] **9.2.2** Convert `FrontierLaneContract`, `FrontierLaneRecord`, `FrontierCampaignProgress` (in renamed `frontier_progress_state.py`) likewise
- [ ] **9.2.3** ~~Move shadow-write fields off `FrontierArchiveMember`~~ **DROPPED — UNSAFE (per Codex r2)**. The original audit verified read-vs-write only inside `frontier_archive.py`. In reality `frontier_recommendation.py:96, 131, 165, 191, 205-206, 275-276` reads `recommendation_flags` and `distance_from_seed` heavily across all 4 policies. Moving these off the dataclass would break recommendation. If a contract migration is desired in the future, scope it as a separate, dedicated task with explicit consumer rewrites — not as part of this mechanical-cleanup phase.
- [ ] **9.2.4** Run mypy or pyright if available

### 9.3 Functional dispatch

- [ ] **9.3.1** Replace `generate_frontier_lane_specs` 5-arm if-chain (`frontier_scalarization.py:700-771`) with `MappingProxyType` registry
- [ ] **9.3.2** If Phase 1B = KEEP NSGA3, replace runner engine if/else (lines 916-942) with engine registry (covered in 1B.4)

### 9.4 Hand-rolled JSON ser/de helper (F10, optional — earn before invest)

**Context:** ~150 LOC of similar `to_json_dict` / `from_json_dict` boilerplate across 3 modules. A `schema_versioned` decorator could collapse them, but the decorator itself is ~50 LOC and adds an abstraction layer. **Decide whether the benefit warrants the new abstraction.**

- [ ] **9.4.1** Decide: implement `schema_versioned` decorator, OR leave boilerplate as-is (KISS argument)
- [ ] **9.4.2** If implementing: create `_schema_helpers.versioned_dataclass(version: str, *, coerce: dict[str, Callable] | None = None)`
- [ ] **9.4.3** Migrate one dataclass at a time; each migration must keep round-trip tests green
- [ ] **9.4.4** **Halt early** if migration cost exceeds the boilerplate it removes — that's a sign the abstraction isn't paying for itself

### 9.5 Reporting cleanup

- [ ] **9.5.1** Remove defensive `getattr(args, ..., default)` blocks at `frontier_campaign_reporting.py:181-204, 246-255, 412-421` — argparse always sets these
- [ ] **9.5.2** Inline `_sanitize_lane_record_for_final_output` (lines 526-531) if it's a one-liner
- [ ] **9.5.3** Decide if `FrontierCampaignManifest` dataclass adds value over a typed dict literal; if not, replace

### 9.6 Test cleanup

- [ ] **9.6.1** Collapse 3 cache tests in `test_frontier_evaluator.py:395-536` into 1 parameterized test
- [ ] **9.6.2** Drop subprocess-based round-trip test (`test_frontier_evaluator.py:326`, ~68 LOC) — covered by simpler in-process version (`test_frontier_evaluator.py:278`)
- [ ] **9.6.3** Extract shared `importlib.import_module("banana_opt.frontier_X")` wrapper into `tests/geo/_frontier_test_helpers.py` (helps 5 of 6 frontier test files; ~30-40 LOC saved)
- [ ] **9.6.4** Audit each `patch.multiple(create=True, ...)` site for fragility — flag any test that mocks unrelated module globals
- [ ] **9.6.5** Confirm zero `xfail`, `skip`, marker-gated tests survive (the audit says there are none currently)

---

## 10. Cross-cutting: after each phase

These run after every phase commit, before opening a PR.

- [ ] **CC.1** Run frontier-relevant tests
  ```bash
  ./run_tests tests/geo/test_frontier_*.py tests/geo/test_single_stage_*.py
  ```
- [ ] **CC.2** Run ruff/lint
  ```bash
  ./run_ruff
  ```
- [ ] **CC.3** Smoke a frontier campaign run with `multilane_local`. **Required CLI args** (revised per Codex r2): `--plasma-surf-filename` and `--stage2-bs-path` are `required=True` in the parent parser at `run_single_stage_goal_mode_comparison.py:187-200`. Replace the placeholders below with actual paths from your local fixtures or lab notes.
  ```bash
  PLASMA_SURF=/path/to/wout_demo.nc           # required — VMEC wout fixture
  STAGE2_BS=/path/to/stage2/biot_savart_opt.json  # required — Stage 2 seed artifact
  OUT=tmp/frontier_smoke_phase_check_$(date +%Y%m%d_%H%M%S)

  python examples/single_stage_optimization/run_single_stage_frontier_campaign.py \
    --plasma-surf-filename "$PLASMA_SURF" \
    --stage2-bs-path "$STAGE2_BS" \
    --frontier-engine multilane_local \
    --frontier-num-lanes 3 \
    --frontier-lane-budget 10 \
    --output-root "$OUT"
  ```
- [ ] **CC.4** Diff JSON artifact shape vs prior smoke run (regression guard for D3). The default summary file is **`single_stage_frontier_campaign_summary.json`** (`frontier_campaign_reporting.py:52` defines `DEFAULT_SUMMARY_JSON`); not `summary.json`.
  ```bash
  jq -S '.frontier_runtime_calibration' "$OUT/single_stage_frontier_campaign_summary.json"
  ```
- [ ] **CC.5** Verify dependency direction
  ```bash
  rg "from .frontier_|from banana_opt.frontier_" examples/single_stage_optimization/banana_opt/single_stage_*.py
  ```
  Expected after Phase 2: zero hits, OR only consumer-style references where `single_stage_objectives` calls into `frontier_scalarization.apply_*` rather than re-defining the body.
- [ ] **CC.6** Confirm LOC delta matches plan
  ```bash
  wc -l examples/single_stage_optimization/banana_opt/frontier_*.py \
        examples/single_stage_optimization/run_single_stage_frontier_campaign.py
  ```
- [ ] **CC.7** Update STATE / lab notes if work is paused mid-phase

---

## 11. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Phase 2 (scalarization move) breaks gradient consistency | Medium | High | Run full `test_frontier_evaluator.py` and `test_single_stage_alm_integration.py` after each function move; bisect on smallest reproduction if fail |
| Phase 1A premature deletion if NSGA3 has undocumented production use | Low (per audit) | Medium | D1 investigation must complete first; if any artifact found, escalate to Phase 1B (quarantine) |
| Phase 8.2 calibration JSON shape change breaks downstream tools | Medium | Medium | D3 audit before executing; preserve `to_json_dict` shape if external consumers exist |
| Phase 6.1 rename causes wide diff churn that conflicts with concurrent branches | Low | Low | Coordinate with team; do rename in dedicated commit; merge to base before continuing |
| Hypervolume memoization (8.1) wrong cache key → silent stale results | Low | High | Cache key must be tuple-of-tuples of objective vectors (immutable, hashable); add property test asserting cache invalidates when archive changes |
| Schema-versioned dataclass refactor (9.4) introduces decorator complexity > boilerplate it removes | Medium | Low | Halt early — see 9.4.4 stop condition |
| Test deletions hide future regressions | Low | Medium | For each deleted test, confirm coverage exists in another test file; document in commit message |

---

## 12. Done criteria (final review)

A reviewer should be able to confirm all of the following with grep + the test suite.

- [ ] **DC.1** D1 decision is recorded in this file
- [ ] **DC.2** If D1 = DROP: zero hits for `nsga3` in `examples/single_stage_optimization/banana_opt/` and `tests/geo/` source code (string mentions in retired docs allowed)
- [ ] **DC.3** If D1 = QUARANTINE: a real `FrontierEngine` Protocol exists; runner dispatches via registry; benchmark test exists
- [ ] **DC.4** `single_stage_objectives.py` contains zero `_frontier_*` private functions; only consumer-style imports
- [ ] **DC.5** `annotate_search_evaluation_finiteness` lives in `search_evaluation.py` (or equivalent non-frontier module); no `frontier_constraints` import in non-frontier code
- [ ] **DC.6** `epsilon_constraint_qa_max` / `epsilon_constraint_boozer_max` string keys appear only inside `frontier_contracts.py` (one frozen dataclass)
- [ ] **DC.7** `frontier_engine_base.py` does not exist (renamed to `frontier_progress_state.py`)
- [ ] **DC.8** `frontier_engine_multilane_local.py` does not exist (folded into `frontier_scalarization.py`)
- [ ] **DC.9** `frontier_archive_hypervolume` is memoized; benchmark test demonstrates cache hits
- [ ] **DC.10** `pymoo` is either not imported (Phase 1A) or lazily imported (Phase 1B + 8.3)
- [ ] **DC.11a** Frontier module LOC (sum of `examples/single_stage_optimization/banana_opt/frontier_*.py`) < **5,500** (started at 6,435)
- [ ] **DC.11b** Runner LOC (`run_single_stage_frontier_campaign.py`) < **950** (started at 1,101)
- [ ] **DC.11c** Combined source LOC < **6,500** (started at 7,536)
- [ ] **DC.12** Frontier test LOC (`tests/geo/test_frontier_*.py`) < **2,200** (started at 3,207); plus the NSGA3 blocks deleted from `test_single_stage_workflow_helpers.py` (estimated −320 LOC there)
- [ ] **DC.13** All frontier tests green; ruff clean; smoke campaign run produces equivalent JSON shape (or shape change documented in D3)
- [ ] **DC.14** `docs/single_stage_frontier_*.md` plan files updated: superseded sections marked; current state reflects this plan's outcomes
- [ ] **DC.15** Updated audit re-run shows < 5% removable LOC (down from ~15-20%)

---

## 13. Out of scope (explicit non-goals)

- Algorithmic redesign of multi-objective optimization
- Replacing ALM with a different inner solver
- Adding new engines (e.g., MOEA/D, Bayesian/Kriging surrogates) — possible future work, separate plan
- Adding meta-level GA/LLM orchestration policies (cf. Kaptanoglu & Gil 2603.15240) — possible future work, separate plan
- Refactoring non-frontier `single_stage_*` runners (their `build_summary` / `build_parser` 4-way fork is a separate, larger problem)
- Touching `simsopt/` library code or JAX kernels

---

## 14. Suggested commit / PR structure

One PR per phase, with phase number in title. This keeps reviews tractable and makes any rollback localized.

- [ ] **PR1** `frontier: drop NSGA3 engine path` (Phase 1A) — or **PR1'** `frontier: quarantine NSGA3 behind Engine Protocol` (Phase 1B)
- [ ] **PR2** `frontier: restore SSOT — move scalarization back to frontier_scalarization.py` (Phase 2.1)
- [ ] **PR3** `frontier: extract search_evaluation helper out of frontier_constraints` (Phase 2.2)
- [ ] **PR4** `frontier: rename engine_base → progress_state; fold multilane into scalarization` (Phase 3.1, 3.2)
- [ ] **PR5** `frontier: centralize epsilon thresholds` (Phase 4.1) — Phase 4.2 dropped per r2 (YAGNI)
- [ ] **PR6** `frontier: parameterize best-of selectors` (Phase 4.3)
- [ ] **PR7** `frontier: memoize hypervolume; collapse calibration profile registry` (Phase 5.1, 5.2)
- [ ] **PR8** `frontier: lazy pymoo + single-pass resume + single-pass resolution` (Phase 5.3, 5.4, 5.5) — combine if PR1' chosen, otherwise drop the pymoo bit
- [ ] **PR9** `frontier: split build_runtime monolith; frozen dataclasses; functional dispatch` (Phase 6.1, 6.2, 6.3)
- [ ] **PR10** `frontier: reporting + test cleanup` (Phase 6.5, 6.6)
- [ ] **PR11** (optional) `frontier: schema_versioned helper` (Phase 6.4) — only if 9.4.1 decides yes

---

## 15. References

- Audit report (multi-agent, 2026-05-07) — see this conversation's record
- Codex independent validation (2026-05-07) — confirmed F1-F13; corrected F14, F15, F16
- Original plan series superseded by this work:
  - `docs/single_stage_frontier_impl_plan_2026-04-12.md` (v1)
  - `docs/single_stage_frontier_v4_requirements_and_impl_plan_2026-04-13.md` (v4)
  - `docs/single_stage_frontier_global_pareto_plan_2026-04-22.md` (NSGA3 introduction)
  - `docs/single_stage_frontier_gradient_contract_impl_plan_2026-04-26.md` (corrective)
- External literature:
  - Deb & Jain, NSGA-III Part I (IEEE TEVC 2014) — https://www.egr.msu.edu/~kdeb/papers/k2012009.pdf
  - pymoo NSGA-III docs — https://pymoo.org/algorithms/moo/nsga3.html
  - Stellarator Optimization with Constraints (arXiv 2403.11033)
  - Kaptanoglu & Gil, "AI-driven stellarator coil optimization" (arXiv 2603.15240, March 2026) — supports gradient-based ALM + meta-level GA, not NSGA-III on DOFs
  - Wechsung et al., Single-stage gradient-based stellarator coil design (JCP) — supports gradient-based approach

---

**End of plan.** When picking up this work, start at section 3 (Decision Log). Do not skip ahead to mechanical cleanup — the structural cuts unblock and shrink the mechanical work.
