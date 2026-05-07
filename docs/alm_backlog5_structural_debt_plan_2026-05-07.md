# ALM Backlog 5 Structural Refactor Plan

Date: 2026-05-07

Status: plan prepared; implementation not started in this ticket.

Scope source: Backlog 5 in `docs/alm_hardening_engineering_followup_todo_2026-05-07.md`.

## Current Tree Facts

- `examples/single_stage_optimization/alm_utils.py:2009-3437` defines `minimize_alm` as a 1,429 line function.
- `minimize_alm` currently contains 15 local functions: 13 ALM/result/history/penalty helpers plus the two SciPy inner-solve callables `inner_fun` and `alm_inner_callback`.
- `examples/single_stage_optimization/workflow_runner_common.py:119-219` defines `Stage2ArtifactConfig` as a 101 line dataclass with 54 annotated fields.
- `Stage2ArtifactConfig` currently mixes Stage 2 geometry, hardware/current controls, objective weights, ALM controls, basin hopping, finite-current wiring, target LCFS ceilings, and iota gate settings.
- Official SciPy contract checked through Context7 `/scipy/scipy`: `scipy.optimize.minimize` with `method="L-BFGS-B"` returns an `OptimizeResult` with `x`, `fun`, `nit`, `jac`, `message`, and `success`; `jac=True` is the documented combined objective-and-gradient contract.

## Objective

Refactor the structural debt without changing optimizer math, physics semantics, Stage 2 artifact identity, ALM result schema, or downstream runner behavior.

The desired end state is:

- `minimize_alm` becomes a readable orchestrator over explicit state objects and testable helper functions.
- Stage 2 configuration is represented by per-concern dataclasses while the existing public construction surface remains strict and deterministic during migration.
- The refactor has golden behavior coverage before code movement starts.
- No defensive code, broad fallbacks, optional compatibility modes, silent recovery paths, or solver policy changes are introduced.

## Non-Goals

- Do not change ALM tolerances, penalty scale policy, trust-radius policy, continuation policy, feasibility definitions, or stationarity tests.
- Do not replace SciPy L-BFGS-B or change the `jac=True` combined objective/gradient contract.
- Do not loosen validation, introduce `try/except` recovery, or add alternate solver paths.
- Do not bundle unrelated Stage 2 physics changes or downstream `autoresearch` integration changes.
- Do not widen public APIs until the existing caller matrix is covered by tests.

## Requirements

### Functional Requirements

- Preserve every key in the ALM result dictionary, including nested `constraint_metadata`, `constraint_*`, `alm_*`, `history`, and termination fields consumed by downstream code.
- Preserve callback ordering and payload shape for the live `minimize_alm`
  callback API: `inner_callback`, `accepted_callback`, `outer_state_callback`,
  and `history_callback`.
- Preserve best-feasible restore semantics on failure, penalty-cap termination, and max-outer-iteration termination.
- Preserve history truncation behavior controlled by `settings.history_max_entries`.
- Preserve strict evaluator signal contracts for `constraint_values`, `feasibility_values`, and `dual_update_values`.
- Preserve Stage 2 artifact path generation, command construction, metadata validation, seed spec generation, and iota decision-gate behavior.
- Preserve exact current failure behavior for invalid Stage 2 config values.

### Architecture Requirements

- Replace `minimize_alm` local helper closures with top-level private helpers that take explicit inputs and return explicit results.
- Use one internal ALM state carrier for mutable run state instead of many unrelated `nonlocal` variables.
- Keep hot-path helper inputs narrow: arrays, scalar settings, and explicit run state only.
- Split Stage 2 config by concern:
  - `Stage2ArtifactIOConfig`
  - `Stage2GeometryConfig`
  - `Stage2HardwareConfig`
  - `Stage2ObjectiveWeights`
  - `Stage2ConstraintPolicy`
  - `Stage2AlmControls`
  - `Stage2BasinControls`
  - `Stage2FiniteCurrentConfig`
  - `Stage2IotaConfig`
- Keep `Stage2ArtifactConfig` as a strict public facade with the current flat
  constructor signature during migration. The facade must store each field
  exactly once inside the owning frozen concern object and expose read-only flat
  properties for existing callers. Do not store duplicate flat fields on the
  facade.
- Replace raw `dataclasses.asdict(config)` and direct `dataclasses.fields`
  introspection with explicit flat projection helpers so nested concern storage
  does not change JSON payloads, artifact identity, or schema parity tests.
- Move single-concern validation to the owning concern object. Keep
  cross-concern validation in the `Stage2ArtifactConfig` composer, including
  constraint override validation and iota-mode/constraint-method consistency.
- Preserve current exception type and message for every current
  `Stage2ArtifactConfig` invalid construction path, not only paths already
  asserted by tests.

### Performance And Safety Requirements

- Do not add extra objective or constraint evaluations.
- Do not add avoidable array copies in the inner L-BFGS-B loop.
- Do not retain unbounded history or callback snapshots.
- Do not add module-level mutable state; all solver state must remain per-call and thread-safe.
- Preserve memory behavior for `history_max_entries` and existing result payloads.

### Testability Requirements

- Add golden characterization tests before moving code.
- Prefer small synthetic ALM problems over expensive Stage 2 runs for solver behavior tests.
- Cover each extracted helper through direct unit tests once it exists.
- Add Stage 2 config parity tests that compare old-vs-new artifact path, subprocess command, seed spec, and metadata payloads for representative modes.

## Execution Plan

### Phase 0: Baseline Metrics And Characterization Tests

- [ ] Add an AST-based test or script assertion that records current structural metrics:
  - `minimize_alm` line span.
  - local function count.
  - `Stage2ArtifactConfig` field count.
- [ ] Add ALM golden tests for:
  - successful convergence on a small constrained quadratic.
  - penalty-cap failure.
  - best-feasible restore on failure.
  - history truncation.
  - callback ordering and payload fields for `inner_callback`,
    `accepted_callback`, `outer_state_callback`, and `history_callback`.
  - dual-update versus penalty-increase sequencing.
  - signal shape/field contract failures.
  - full result schema keys for converged and failure-with-restore results:
    `sorted(vars(result).keys())` plus `sorted(result.alm_summary.keys())`.
- [ ] Add Stage 2 config golden tests for:
  - `resolve_stage2_artifact_path`.
  - `build_stage2_command`.
  - `build_stage2_seed_spec`.
  - `_stage2_config_constraint_layer`.
  - artifact metadata validation for finite-current and iota modes.
  - flat JSON payload projection currently produced by
    `_jsonable_stage2_config`.
  - exhaustive artifact identity parity:
    `STAGE2_ARTIFACT_PATH_FIELD_NAMES` must match the keyword-only identity
    fields accepted by `local_stage2_bs_path`, and the config projection passed
    to `local_stage2_bs_path` must contain exactly that set.

Acceptance: all characterization tests pass against the current implementation before any refactor.

### Phase 1: Extract ALM Result And History Construction

- [ ] Introduce internal private dataclasses for result-building inputs:
  - `ALMRunState`
  - `ALMFinalState`
  - `ALMHistoryEntry`
- [ ] Move result construction out of `minimize_alm`:
  - `_build_result`.
  - `_append_history_entry`.
  - `_attach_history_diagnostics`.
  - `_emit_history_snapshot`.
  - `_restore_best_feasible_on_failure`.
  - `_build_failure_result_with_optional_restore`.
- [ ] Keep result keys and numeric values identical to the characterization tests.

Acceptance: no result-schema or history-payload diff; `minimize_alm` no longer owns result formatting.

### Phase 2: Extract Penalty And Tolerance Update State Machine

- [ ] Move penalty-state evaluation and publication into explicit helpers:
  - current penalty-state evaluation.
  - post-update history refresh.
  - final penalty-state publication.
- [ ] Move `_try_penalty_increase` into a pure state-transition helper that returns the updated state and decision.
- [ ] Keep one-line reads such as current penalty argument and penalty scale as
  plain `ALMRunState` attributes or direct scalar reads; do not promote them
  to top-level helpers.
- [ ] Keep tighten-only tolerance behavior and penalty-cap behavior unchanged.

Acceptance: penalty-cap, dual-update, and no-progress tests pass without loosening assertions.

### Phase 3: Extract Inner Solve Attempt Loop

- [ ] Introduce an internal `ALMInnerAttemptRequest`.
- [ ] Introduce an internal `ALMInnerAttemptResult`.
- [ ] Move the L-BFGS-B call boundary into one helper that owns:
  - `scipy.optimize.minimize(..., method="L-BFGS-B", jac=True)`.
  - trust-radius bounds for the attempt.
  - the combined value/gradient callable.
  - early-stop callback behavior.
  - finite result selection and attempt retry decision.
- [ ] Pass all callable capture state through `ALMInnerAttemptRequest`, including
  multipliers, penalty scalar, current evaluation, feasibility/stationarity
  tolerances, and the `inner_callback` hook.
- [ ] Preserve the documented SciPy `jac=True` combined objective/gradient contract.

Acceptance: no extra objective evaluations, no callback-order drift, and no change in accepted candidate selection.

### Phase 4: Collapse `minimize_alm` Into The Orchestrator

- [ ] Reduce `minimize_alm` to the high-level sequence:
  - normalize inputs.
  - initialize per-call state.
  - run outer ALM iterations.
  - run continuation/inner attempt helper.
  - apply penalty/dual-update transition.
  - build final result.
- [ ] Remove all ALM policy local functions from `minimize_alm`.
- [ ] Remove `inner_fun` and `alm_inner_callback` from `minimize_alm`; they are
  owned by the Phase 3 inner-attempt helper.

Acceptance targets:

- `minimize_alm` line span is at most 500 lines.
- local function count is 0.
- the existing ALM validation suite and new characterization tests remain green.

### Phase 5: Split Stage 2 Artifact Configuration By Concern

- [ ] Add frozen concern dataclasses in `workflow_runner_common.py` with strict
  validation in the owning class.
- [ ] Keep `Stage2ArtifactConfig` as the public facade that accepts the current
  flat keyword constructor and composes concern dataclasses as the only storage.
- [ ] Expose read-only flat properties on `Stage2ArtifactConfig` for existing
  callers. Do not duplicate storage between facade fields and concern fields.
- [ ] Add explicit flat projection helpers for:
  - JSON payloads formerly using `asdict(config)`.
  - ALM/dataclass schema parity tests formerly using
    `fields(Stage2ArtifactConfig)`.
  - artifact-path keyword projection.
- [ ] Move helper internals to consume concern objects where it improves SSOT:
  - artifact path arguments from geometry/hardware/iota concerns.
  - command arguments from ALM, basin, finite-current, and iota concerns.
  - metadata validation from the same concern objects.
- [ ] Keep cross-concern validation in the facade composer:
  - `validate_constraint_cli_overrides` over hardware, geometry, threshold, and
    LCFS ceiling fields.
  - `stage2_iota_mode != "off"` requires `stage2_iota_target`.
  - `stage2_iota_mode == "soft"` requires positive `stage2_iota_weight`.
  - `stage2_iota_mode == "soft"` remains incompatible with
    `constraint_method == "alm"`.
  - `stage2_iota_mode == "alm"` still requires `constraint_method == "alm"`.
- [ ] Avoid temporary alternate config paths or permissive migration behavior.

Acceptance:

- Representative Stage 2 artifact paths, commands, seed specs, and metadata
  payloads remain identical.
- Exhaustive identity-field parity proves `local_stage2_bs_path`,
  `STAGE2_ARTIFACT_PATH_FIELD_NAMES`, and artifact-path config projection
  cannot drift.
- Flat JSON payloads remain identical before and after nested concern storage.
- Current invalid-construction exception types and messages are preserved.

### Phase 6: Simplification And Closeout

- [ ] Run a behavior-preserving `code-simplifier` pass on touched files only.
- [ ] Update this plan with completed checkboxes, final line spans, local function count, field ownership, and validation evidence.
- [ ] Update Backlog 5 in the engineering follow-up tracker with the closeout result.

Acceptance: the tracker no longer contains stale line/count facts.

## Validation Matrix

Run after each implementation phase that changes code:

```bash
.venv/bin/python -m py_compile \
  examples/single_stage_optimization/alm_utils.py \
  examples/single_stage_optimization/workflow_runner_common.py
.venv/bin/python -m pytest -q tests/geo/test_alm_utils.py
.venv/bin/python -m pytest -q tests/geo/test_single_stage_workflow_helpers.py
.venv/bin/python -m pytest -q tests/geo/test_single_stage_alm_integration.py
.venv/bin/python -m pytest -q tests/geo/test_single_stage_example.py \
  -k "AlmUtilsTests or build_alm_final_constraint_payload or alm_result_view_from_search_eval or stage2_main_alm_path_uses_minimize_alm or validate_resume_alm_state or current_solver_checkpoint_alm_state"
.venv/bin/python -m pytest -q tests/geo/test_constraint_contract.py tests/geo/test_banana_helper_modules.py
.venv/bin/python -m pytest -q tests/geo/test_stage2_track_b_wrappers.py
git diff --check
```

Run once before closeout:

```bash
python - <<'PY'
import ast
from pathlib import Path

checks = [
    ("examples/single_stage_optimization/alm_utils.py", "minimize_alm"),
    ("examples/single_stage_optimization/workflow_runner_common.py", "Stage2ArtifactConfig"),
]
for relpath, name in checks:
    tree = ast.parse(Path(relpath).read_text())
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            nested = [
                child
                for child in ast.walk(node)
                if isinstance(child, ast.FunctionDef) and child is not node
            ]
            fields = [
                stmt.target.id
                for stmt in getattr(node, "body", [])
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            ]
            print(
                f"{relpath}:{node.lineno}-{node.end_lineno} "
                f"{name} span={node.end_lineno - node.lineno + 1} "
                f"nested_defs={len(nested)} fields={len(fields)}"
            )
PY
```

## Commit Plan

- [ ] Commit 1: plan and baseline characterization tests.
- [ ] Commit 2: ALM result/history extraction.
- [ ] Commit 3: ALM penalty/tolerance transition extraction.
- [ ] Commit 4: ALM inner solve attempt extraction.
- [ ] Commit 5: Stage 2 config concern dataclasses and strict facade wiring.
- [ ] Commit 6: simplifier pass, tracker update, and validation evidence.

## Go/No-Go Criteria

Start implementation only after Phase 0 characterization tests are green. Stop and reassess if any phase requires changing ALM math, Stage 2 artifact identity, or SciPy optimizer semantics to make the refactor pass.
