# ALM Hardening Engineering Followup Todos

Date: 2026-05-07

Validated against: `simsopt-surrogate` HEAD `fd18380c6` (uncommitted working tree)

Status: implemented and committed in the 2026-05-07 ALM hardening closeout slice.

Relationship: extends `docs/alm_hardening_closeout_todo_plan_2026-05-07.md` with engineering-review findings raised after that tracker's implementation checklist was closed. The final six-commit list has been folded back into the closeout tracker as the historical SSOT.

## Implementation Result

- TODO 1/2 closed with Option A: `SINGLE_STAGE_ALM_CLI_FIELDS` names the single-stage / baseline-sweep defaults, while Stage 2 keeps its intentional `alm_curvature_smoothing = 0.25` default.
- TODO 3 closed with `_build_constraint_metadata_tuples(...)`, precomputed immutable metadata tuples, and no metadata attachment in the L-BFGS-B inner objective/callback hot path.
- TODO 4 validation:
  - `.venv/bin/python -m pytest -q tests/geo/test_alm_utils.py` -> `91 passed, 3 subtests passed`
  - `.venv/bin/python -m pytest -q tests/geo/test_single_stage_workflow_helpers.py` -> `120 passed`
  - `.venv/bin/python -m pytest -q tests/geo/test_single_stage_alm_integration.py` -> `73 passed`
  - `.venv/bin/python -m pytest -q tests/geo/test_single_stage_example.py -k "AlmUtilsTests or build_alm_final_constraint_payload or alm_result_view_from_search_eval or stage2_main_alm_path_uses_minimize_alm or validate_resume_alm_state or current_solver_checkpoint_alm_state"` -> `12 passed, 278 deselected`
  - `.venv/bin/python -m pytest -q tests/geo/test_constraint_contract.py tests/geo/test_banana_helper_modules.py` -> `50 passed, 3 subtests passed`
  - `git diff --check` -> pass
  - The two CLI tuple deletion gates returned zero hits.
- Final commit-only-work layout is folded into `docs/alm_hardening_closeout_todo_plan_2026-05-07.md`.

## Goal

Close the remaining engineering-principle gaps surfaced by the post-closeout review, then execute the tracker's commit-grouping pass with accurate scope. Ship a working tree where:

- The CLI defaults table is named for what it actually is (single-stage / baseline-sweep regime), not for "all ALM CLI defaults".
- The L-BFGS-B inner loop does not allocate per-call constraint-metadata copies that no inner consumer reads.
- All affected dependent files (test, runner, tracker doc) reflect the renamed surface.
- The implementation work lands as five commits per the existing tracker, plus one new perf commit for the hot-path fix.

Out of scope for this followup: pre-existing structural debt (`minimize_alm` size, `Stage2ArtifactConfig` breadth), advisory cleanups not surfaced by the review.

## Pre-Commit Must-Address

### TODO 1: Rename `ALM_CLI_FIELDS` to disambiguate regime

**Context.** `examples/single_stage_optimization/workflow_runner_common.py:42` defines `ALM_CLI_FIELDS` as the SSOT for ALM CLI metadata, but the tuple's defaults match the single-stage / baseline-sweep regime, not Stage 2:

| Field | `ALM_CLI_FIELDS` default | `Stage2ArtifactConfig.alm_curvature_smoothing` default | Stage 2 parser default |
|---|---|---|---|
| `curvature_smoothing` | `0.05` (line 56) | `0.25` (line 115) | `0.25` (`banana_coil_solver.py:526`) |

The split is **intentional** — `banana_coil_solver.py:520-525` documents that Stage 2 uses a broader softmax window. The bug is the misleading name: `ALM_CLI_FIELDS` claims to be the single source but actually encodes only one regime's defaults. A future maintainer who edits `ALM_CLI_FIELDS["curvature_smoothing"]` will silently NOT update Stage 2.

**Why it matters.** The `commit-only-work` pass will land Commit 2 (ALM CLI SSOT refactor) under the assumption that `ALM_CLI_FIELDS` is genuine SSOT. If the name stays, a future "fix the SSOT default" change becomes a covert single-stage-only change. Rename closes the drift class the SSOT rebrand was supposed to prevent.

**Tasks.**

- [x] Rename the tuple to `SINGLE_STAGE_ALM_CLI_FIELDS` in `workflow_runner_common.py:42`.
- [x] Update `alm_flag()` and `single_stage_alm_flag()` definitions if any callsite assumes the old name.
- [x] Update `append_alm_cli_flags()` body at `workflow_runner_common.py:78` to iterate the renamed tuple.
- [x] Decide on Stage 2 default surface (pick one):
  - **Option A** — keep `Stage2ArtifactConfig.alm_curvature_smoothing = 0.25` as-is; the rename alone resolves the SSOT misnomer. No derivation. Smallest diff.
  - **Option B1** — introduce `STAGE2_ALM_CLI_FIELDS` as a **full** sibling tuple (all 14 fields, with Stage 2 defaults: `curvature_smoothing=0.25`, others matching single-stage). Derive `Stage2ArtifactConfig` ALM defaults from this tuple via a factory or class-decorator. Symmetric with `SINGLE_STAGE_ALM_CLI_FIELDS`; full SSOT for both regimes.
  - **Option B2** — keep one base `ALM_CLI_FIELDS_BASE` tuple plus a `STAGE2_ALM_OVERRIDES = {"curvature_smoothing": 0.25}` mapping. Derive both per-regime tuples by overlaying. Smaller surface; explicit about what differs.
- [x] If Option B1 or B2 is chosen, also rewrite `Stage2ArtifactConfig` ALM defaults to derive from the chosen source so the `0.25` literal at `workflow_runner_common.py:115` disappears. (A diff-only tuple alone cannot derive the full Stage 2 ALM surface — Option B must cover all 14 fields, either directly or via a base+override merge.)
- [x] Add a docstring or short comment near the renamed tuple noting it covers single-stage / baseline-sweep defaults only and that Stage 2 has its own distinct curvature-smoothing default.

**Acceptance.**

- `git grep -nE "\bALM_CLI_FIELDS\b" examples/ tests/` returns zero hits (live code uses the renamed symbol only).
- `git grep -nE "\bALM_CLI_FIELDS\b" docs/alm_hardening_closeout_todo_plan_2026-05-07.md` returns zero hits (the closeout tracker is updated as part of TODO 2).
- This followup doc itself legitimately references both the old and new names in prose to describe the rename — it is intentionally **out of scope** for both gates.
- The schema-parity test continues to pass after the rename.
- The Stage 2 `0.25` curvature_smoothing default is preserved (no behavior change for Stage 2 callers).

**Suggested commit.** Folds into existing **Commit 2: ALM CLI SSOT refactor** in `docs/alm_hardening_closeout_todo_plan_2026-05-07.md:183`.

### TODO 2: Update all dependent references after the rename

**Context.** The current name has 10 dependent references across 4 files (plus the definition site at `workflow_runner_common.py:42`, which TODO 1 renames). All 10 dependents must update atomically with TODO 1.

**Tasks.**

- [x] `examples/single_stage_optimization/run_80ka_baseline_tradeoff_sweep.py:19` — import statement.
- [x] `examples/single_stage_optimization/run_80ka_baseline_tradeoff_sweep.py:140` — iteration in `parse_args()`.
- [x] `examples/single_stage_optimization/workflow_runner_common.py:78` — iteration in `append_alm_cli_flags()`.
- [x] `tests/geo/test_single_stage_workflow_helpers.py:1131` — schema-parity test reference.
- [x] `tests/geo/test_single_stage_workflow_helpers.py:1136` — schema-parity test reference.
- [x] `tests/geo/test_single_stage_workflow_helpers.py:1141` — schema-parity test length assertion.
- [x] `tests/geo/test_single_stage_workflow_helpers.py:1154` — round-trip parity test reference.
- [x] `docs/alm_hardening_closeout_todo_plan_2026-05-07.md:50` — illustrative code block in P1 SSOT section.
- [x] `docs/alm_hardening_closeout_todo_plan_2026-05-07.md:76` — Acceptance bullet text.
- [x] `docs/alm_hardening_closeout_todo_plan_2026-05-07.md:80` — Acceptance bullet text.

**Acceptance.**

- `tests/geo/test_single_stage_workflow_helpers.py::test_baseline_sweep_alm_cli_schema_matches_emitter_schema` passes.
- `tests/geo/test_single_stage_workflow_helpers.py::test_baseline_sweep_single_stage_alm_flags_round_trip_to_command` passes.
- The closeout tracker doc references the new name in all three illustrative locations.

**Suggested commit.** Same as TODO 1 — **Commit 2**.

### TODO 3: Reduce `_attach_constraint_metadata` hot-path allocation

**Context.** `examples/single_stage_optimization/alm_utils.py:2075-2081`:

```python
def _attach_constraint_metadata(evaluation: dict) -> dict:
    if diagnostic_constraint_blocks is None:
        return evaluation
    annotated = dict(evaluation)
    annotated["constraint_names"] = [str(name) for name in constraint_names]
    annotated["constraint_blocks"] = list(diagnostic_constraint_blocks)
    return annotated
```

This closure is called from:
- `inner_fun` at `alm_utils.py:2797` — every L-BFGS-B inner objective evaluation.
- `alm_inner_callback` at `alm_utils.py:2815` — every L-BFGS-B callback invocation.
- The candidate-evaluation path at `alm_utils.py:2898` — every inner attempt.
- The outer-iterate evaluation at `alm_utils.py:2620` — once per outer iteration.

For typical L-BFGS-B with `maxiter=50-200` per outer iteration × `max_outer_iterations=10` × constraint count `N≈20`, this allocates **10K-80K throwaway dicts/lists per `minimize_alm` call**.

The downstream consumers `inner_fun` (`alm_utils.py:2797-2806`) and `alm_inner_callback` (`alm_utils.py:2808-2847`) read only `total`, `grad`, `constraint_values`, `feasibility_values`, `dual_update_values`, plus the routing-state inputs — none of them read `constraint_names` or `constraint_blocks` from the evaluation dict.

The metadata is consumed at history-snapshot time on the accepted candidate via `_attach_history_diagnostics` → `_constraint_history_diagnostics_source` (which reads `evaluation["constraint_blocks"]`).

**Why it matters.** The allocation is on the L-BFGS-B inner hot path. With JAX-backed evaluators where the actual physics evaluation is fast, the metadata-attach overhead becomes a measurable share of the inner loop. Even on slower CPU paths it is gratuitous garbage that pressures the allocator. The plan called for "PERFORMANT, OPTIMIZED, MEMORY EFFICIENT" code; this is the most direct violation introduced by the hardening work.

**Tasks.**

- [x] Extract a module-level pure helper `_build_constraint_metadata_tuples(constraint_names, constraint_blocks) -> (names_tuple, blocks_tuple)` in `alm_utils.py`. Pure function, no closure capture, directly unit-testable. **Preserve the existing length-mismatch ValueError** (today at `alm_utils.py:2027-2028`, covered by the regression test `tests/geo/test_alm_utils.py::test_minimize_alm_rejects_mismatched_constraint_block_metadata` at `:280` which asserts `assertRaisesRegex(ValueError, "constraint_blocks length")`):
  ```python
  def _build_constraint_metadata_tuples(
      constraint_names: Sequence[str],
      constraint_blocks: Sequence[str] | None,
  ) -> tuple[tuple[str, ...], tuple[str, ...] | None]:
      names_tuple = tuple(str(name) for name in constraint_names)
      if constraint_blocks is None:
          return names_tuple, None
      if len(constraint_blocks) != len(constraint_names):
          raise ValueError("constraint_blocks length must match constraint_names")
      blocks_tuple = tuple(str(block) for block in constraint_blocks)
      return names_tuple, blocks_tuple
  ```
  This centralizes both the conversion and the validation (SRP) and gives the regression tests something testable that does not depend on `minimize_alm`'s closure structure. The error message is held to the existing literal so the existing regression test continues to pass without modification.
- [x] Replace the existing argument-validation block in `minimize_alm` (`alm_utils.py:2025-2029`, including the `len(constraint_blocks) != len(constraint_names)` check) with a single call into the new helper:
  ```python
  _constraint_names_tuple, _constraint_blocks_tuple = _build_constraint_metadata_tuples(
      constraint_names, constraint_blocks
  )
  ```
  Use tuples rather than lists — tuples are genuinely immutable in Python, so accidental mutation by a downstream consumer would raise. After this change, the `ValueError` for mismatched block/name lengths is raised from inside the helper, not from `minimize_alm` directly; the existing regression test still passes because it only asserts the message text and the call surface.
- [x] Confirm `tests/geo/test_alm_utils.py::test_minimize_alm_rejects_mismatched_constraint_block_metadata` (at `:280`) is still green after the helper extraction, with no test modification required. If the test ever needs adjustment because the error origin changed, leave it alone — it asserts behavior, not stack frame.
- [x] Verify that no current consumer of `evaluation["constraint_names"]` or `evaluation["constraint_blocks"]` mutates the value (calls `.append()`, item assignment, etc.) before switching to tuples. If mutation is found, fix the consumer or keep lists but route through the helper so the precompute-once invariant still holds.
- [x] Rewrite `_attach_constraint_metadata` to attach the precomputed tuples rather than rebuild per call:
  ```python
  def _attach_constraint_metadata(evaluation: dict) -> dict:
      if _constraint_blocks_tuple is None:
          return evaluation
      annotated = dict(evaluation)
      annotated["constraint_names"] = _constraint_names_tuple
      annotated["constraint_blocks"] = _constraint_blocks_tuple
      return annotated
  ```
- [x] **Stronger fix (preferred if read-site audit allows):** skip `_attach_constraint_metadata` inside `inner_fun` and `alm_inner_callback` entirely; only attach when promoting an evaluation to `current_eval` / `accepted_eval` / history. This eliminates the `dict(evaluation)` shallow copy on every inner call as well. Requires confirming no inner-loop read expects `constraint_names` / `constraint_blocks` on the evaluation dict. The grep at `alm_utils.py:2797-2847` (inner_fun + callback) and at the routing/extract helpers shows none of them read those keys today — but lock that with a follow-up audit before deleting the closure-internal call.
- [x] **Test 1 — direct helper unit test.** Cover the helper with instrumented sentinels. This is the precompute correctness gate.
  ```python
  class _CountingName:
      def __init__(self, name: str) -> None:
          self._name = name
          self.str_calls = 0
      def __str__(self) -> str:
          self.str_calls += 1
          return self._name

  class _CountingBlock:
      def __init__(self, block: str) -> None:
          self._block = block
          self.str_calls = 0
      def __str__(self) -> str:
          self.str_calls += 1
          return self._block

  def test_build_constraint_metadata_tuples_stringifies_each_input_once():
      names = [_CountingName(f"c_{i}") for i in range(3)]
      blocks = [_CountingBlock(f"g_{i}") for i in range(3)]

      names_tuple, blocks_tuple = _build_constraint_metadata_tuples(names, blocks)

      assert isinstance(names_tuple, tuple)
      assert isinstance(blocks_tuple, tuple)
      for counter in names:
          assert counter.str_calls == 1
      for counter in blocks:
          assert counter.str_calls == 1
  ```
  The helper is a pure function, so exact `== 1` is the right assertion here. There is no closure or non-hot-path consumer to inflate the count.
- [x] **Shared test fixture — force inner-evaluation count via fake `minimize`.** The naive `maxiter=5` vs `maxiter=100` comparison can false-pass if both inner solves converge early to the same iteration count. Force the L-BFGS-B inner loop to use the full budget by patching `alm_utils.minimize` with a stub that calls `inner_fun` exactly `options["maxiter"]` times before returning a non-converged result:
  ```python
  def _make_fake_minimize():
      def fake_minimize(fun, x, jac, method, bounds, callback, options):
          inner_max = int(options.get("maxiter", 1))
          x_arr = np.asarray(x, dtype=float)
          for _ in range(inner_max):
              fun(x_arr)
          return SimpleNamespace(
              x=x_arr,
              nit=inner_max,
              success=False,
              message="MAXITER",
          )
      return fake_minimize
  ```
  Both Test 2 and Test 3 below depend on this fixture so the scaling assertion observes a real inner-evaluation difference.
- [x] **Test 2 — end-to-end names scaling test.** Cover the names hot-path through observable behavior. Do **not** assert exact `str_calls == 1` because non-hot-path consumers (history diagnostics at `alm_utils.py:722`, result construction, history entry serialization) also stringify `constraint_names` per outer iteration. Assert instead that the per-name conversion count **does not scale with the inner-iteration budget**:
  ```python
  def _run_with_maxiter(maxiter: int) -> int:
      names = [_CountingName(f"c_{i}") for i in range(3)]
      with patch.object(alm_utils, "minimize", side_effect=_make_fake_minimize()):
          minimize_alm(
              x0,
              names,
              evaluate_problem,
              settings,  # max_outer_iterations fixed across both runs
              {"maxiter": maxiter, "ftol": 1.0e-12, "gtol": 1.0e-12},
          )
      return max(c.str_calls for c in names)

  def test_attach_constraint_metadata_names_does_not_scale_with_inner_iterations():
      short = _run_with_maxiter(5)
      long = _run_with_maxiter(100)
      # Buggy code: long ~ short + (100 - 5) * outer_iter_count per name.
      # Fixed code: long == short because non-hot-path callers depend only on
      # outer-iteration count, which is fixed by ALMSettings, not by maxiter.
      assert long == short, (
          f"constraint_names str() scales with inner iterations: "
          f"short(maxiter=5)={short}, long(maxiter=100)={long}"
      )
  ```
- [x] **Test 3 — end-to-end blocks scaling test (closure-path regression gate).** A future regression like `annotated["constraint_blocks"] = list(_constraint_blocks_tuple)` would pass both Test 1 (helper still produces the right tuple) and Test 2 (names attachment unchanged), but reintroduce per-call block allocation. Catch it by monkeypatching `_build_constraint_metadata_tuples` to return a counting block container, then assert iteration on that container does not scale with the inner-iteration budget:
  ```python
  class _CountingBlockContainer:
      def __init__(self, items: Sequence[str]) -> None:
          self._items = tuple(str(item) for item in items)
          self.iter_calls = 0
      def __iter__(self):
          self.iter_calls += 1
          return iter(self._items)
      def __len__(self) -> int:
          return len(self._items)
      def __getitem__(self, index):
          return self._items[index]

  def _run_blocks_with_maxiter(maxiter: int) -> int:
      counting_blocks = _CountingBlockContainer(["geometry", "current"])

      def _injecting_helper(constraint_names, constraint_blocks):
          names_tuple = tuple(str(name) for name in constraint_names)
          return names_tuple, counting_blocks

      with patch.object(
          alm_utils,
          "_build_constraint_metadata_tuples",
          side_effect=_injecting_helper,
      ), patch.object(alm_utils, "minimize", side_effect=_make_fake_minimize()):
          minimize_alm(
              x0,
              ["c_0", "c_1"],
              evaluate_problem,
              settings,
              {"maxiter": maxiter, "ftol": 1.0e-12, "gtol": 1.0e-12},
              constraint_blocks=["geometry", "current"],
          )
      return counting_blocks.iter_calls

  def test_attach_constraint_metadata_blocks_does_not_scale_with_inner_iterations():
      short = _run_blocks_with_maxiter(5)
      long = _run_blocks_with_maxiter(100)
      # Buggy attach `list(_constraint_blocks_tuple)` triggers `__iter__` per
      # inner evaluation; fixed attach passes the object by reference and only
      # non-hot-path consumers (history diagnostics, result construction) iterate.
      assert long == short, (
          f"constraint_blocks iteration scales with inner iterations: "
          f"short(maxiter=5)={short}, long(maxiter=100)={long}"
      )
  ```
  Injecting via the helper monkeypatch is the cleanest hook: it is the sole conversion point in production, so the test does not depend on `_attach_constraint_metadata`'s internals or refactor production code purely for testability.

**Acceptance.**

- A module-level pure helper `_build_constraint_metadata_tuples` exists and is the single conversion point for both names and blocks; it also enforces the existing `len(constraint_blocks) == len(constraint_names)` check with the original `"constraint_blocks length must match constraint_names"` ValueError message.
- The pre-existing regression test `tests/geo/test_alm_utils.py::test_minimize_alm_rejects_mismatched_constraint_block_metadata` continues to pass without modification.
- Constraint metadata is built once per `minimize_alm` invocation, not per inner evaluation.
- Inner-evaluation count is forced via the `_make_fake_minimize` fixture so the **scaling tests (Test 2 and Test 3)** cannot false-pass on early convergence. Test 1 is a direct unit test of the helper and does not use this fixture.
- Test 1 (direct helper unit test) asserts `str_calls == 1` for each `_CountingName` and each `_CountingBlock` after one helper call.
- Test 2 (names scaling test) asserts `max(c.str_calls)` is identical across `maxiter=5` and `maxiter=100` runs.
- Test 3 (blocks scaling test) injects a `_CountingBlockContainer` via monkeypatched helper and asserts `iter_calls` is identical across `maxiter=5` and `maxiter=100` runs.
- All three tests together close the regression class: Test 1 catches direct precompute regressions; Test 2 catches names-attach regressions including any helper-bypass; Test 3 catches blocks-attach regressions like `annotated["constraint_blocks"] = list(_constraint_blocks_tuple)` that would silently pass Tests 1 and 2.
- All existing `minimize_alm`-driven tests in `tests/geo/test_alm_utils.py` continue to pass.
- The schema-parity / round-trip tests are unaffected (this is a pure perf change).

**Suggested commit.** Net-new commit, recommended title `perf: share ALM constraint metadata across inner evaluations`. Insert as **Commit 6** in the tracker's commit grouping section.

### TODO 4: Rerun validation runner after TODOs 1-3

**Context.** The tracker's existing validation block (`docs/alm_hardening_closeout_todo_plan_2026-05-07.md:192-200`) lists the suite. After TODOs 1-3 land in the working tree but before commit grouping, rerun to confirm green.

**Tasks.**

- [x] `.venv/bin/python -m pytest -q tests/geo/test_alm_utils.py`
- [x] `.venv/bin/python -m pytest -q tests/geo/test_single_stage_workflow_helpers.py`
- [x] `.venv/bin/python -m pytest -q tests/geo/test_single_stage_alm_integration.py`
- [x] `.venv/bin/python -m pytest -q tests/geo/test_single_stage_example.py -k "AlmUtilsTests or build_alm_final_constraint_payload or alm_result_view_from_search_eval or stage2_main_alm_path_uses_minimize_alm or validate_resume_alm_state or current_solver_checkpoint_alm_state"`
- [x] `.venv/bin/python -m pytest -q tests/geo/test_constraint_contract.py tests/geo/test_banana_helper_modules.py`
- [x] `git diff --check`
- [x] CLI tuple deletion gate (live code): `git grep -nE "\bALM_CLI_FIELDS\b" examples/ tests/` returns zero hits.
- [x] CLI tuple deletion gate (closeout tracker): `git grep -nE "\bALM_CLI_FIELDS\b" docs/alm_hardening_closeout_todo_plan_2026-05-07.md` returns zero hits.

**Acceptance.**

- All pytest invocations green; counts match or exceed the prior closeout totals (`88+120+73+12+50` baseline before TODO 3 perf test added).
- Both grep gates return zero hits. (This followup doc is intentionally exempt — it discusses the rename in prose.)

## Completed Process Todos In The Tracker

The implementation checklist in `docs/alm_hardening_closeout_todo_plan_2026-05-07.md` is fully checked off, and its Commit Grouping section now records the final six committed slices.

- [x] **Commit 1**: `refactor: remove ALM off-spec escape hatches`.
- [x] **Commit 2**: `refactor: share single-stage ALM CLI fields`.
- [x] **Commit 3**: `perf: share ALM metadata and harden solver signals`.
- [x] **Commit 4**: `docs: document ALM constraint contract`.
- [x] **Commit 5**: `docs: update ALM closeout commit layout`.
- [x] **Commit 6**: `docs: close ALM engineering followup`.

## Backlog (Non-Blocking)

Items below are advisory cleanups or pre-existing structural debt. They surfaced during the engineering review but are not gating commit grouping.

### Backlog 1: Reduce Stage 2 ALM config duplication

**Context.** Three sites repeat the 14-field ALM list:
- `SINGLE_STAGE_ALM_CLI_FIELDS` at `workflow_runner_common.py:42`.
- `Stage2ArtifactConfig.alm_*` fields at `workflow_runner_common.py:100-115`.
- `resolve_stage2_artifact_path` keyword arguments at `workflow_runner_common.py:222-237`.

A new ALM field today must be added to all three sites. **No existing test catches divergence between `SINGLE_STAGE_ALM_CLI_FIELDS` and `Stage2ArtifactConfig`.** The schema-parity test at `tests/geo/test_single_stage_workflow_helpers.py:1123` only compares the baseline-sweep parser suffix set to `common.SINGLE_STAGE_ALM_CLI_FIELDS` — it never inspects `Stage2ArtifactConfig`'s `alm_*` field set or `resolve_stage2_artifact_path`'s keyword arguments. So a Stage 2 ALM field added to the single-stage tuple (or vice versa) silently goes unverified.

**Tasks.**

- [ ] Decide whether `Stage2ArtifactConfig` derives its ALM field set from a per-regime tuple (Option B from TODO 1) or stays declarative.
- [ ] If derivation is chosen, generate `Stage2ArtifactConfig` ALM fields via a class-decorator or factory that consumes the regime tuple.
- [ ] Replace the explicit `local_stage2_bs_path` keyword argument list in `resolve_stage2_artifact_path` with a `**asdict(config)` projection or similar.
- [ ] Preserve the intentional Stage 2 `curvature_smoothing=0.25` vs single-stage `0.05` split.

**Acceptance.** Adding a new ALM field touches one source location and propagates to parser, emitter, dataclass, and artifact-path resolver automatically.

### Backlog 2: Document `run_command(env=..., inherit_alm_env=...)` overlay semantics

**Context.** `workflow_runner_common.py:377-401`. The merge order `os.environ → strip ALM_ → caller env overlay` is enforced by code but not documented. A caller passing `env={"ALM_PENALTY_INIT": "5"}` while `inherit_alm_env=False` gets the explicit `ALM_PENALTY_INIT` despite the strip. Test at `test_single_stage_workflow_helpers.py:1505` documents the behavior empirically.

**Tasks.**

- [ ] Add a docstring to `run_command` describing the merge order in three lines.
- [ ] Optionally rename `env` parameter to `env_overrides` to make the overlay semantics explicit at the call site.

**Acceptance.** Reader of the function signature understands without consulting the test what `inherit_alm_env=False, env={"ALM_X": ...}` does.

### Backlog 3: Share env-building between `run_command` and `run_poincare_artifact`

**Context.** `run_poincare_artifact` at `workflow_runner_common.py:785-804` uses `env = os.environ.copy()` directly without the ALM_ strip that `run_command` applies. Today this is harmless (POINCARE doesn't read `ALM_*`), but two divergent env-build patterns is one too many for a single file.

**Tasks.**

- [ ] Extract `_build_subprocess_env(env_overrides=..., inherit_alm=...) -> dict[str, str]` as a private helper.
- [ ] Route both `run_command` and `run_poincare_artifact` through the helper.
- [ ] Confirm POINCARE smoke still works with ALM-stripped env (it should — POINCARE has no ALM dependency).

**Acceptance.** One env-build path; both subprocess launchers share semantics.

### Backlog 4: Avoid recomputing max feasibility violation in failure-result path

**Context.** `examples/single_stage_optimization/alm_utils.py:2557`:

```python
final_max_feasibility_violation=_extract_constraint_state(evaluation)[3],
```

The caller `_try_penalty_increase` at `alm_utils.py:2454` already has `penalty_update_state.max_violation`. The data is recomputed (full `_as_float_array` plus `_max_value`) on the failure path.

**Tasks.**

- [ ] Thread `max_feasibility_violation` through the `_build_failure_result_with_optional_restore` signature.
- [ ] Remove the redundant `_extract_constraint_state(evaluation)[3]` call.

**Acceptance.** No behavior change; one fewer redundant array materialization on the penalty-cap failure path.

### Backlog 5: Pre-existing structural debt

**Context.** Surfaced during review but predates this hardening work. Listed for completeness; do not bundle with the closeout commits.

- [ ] `minimize_alm` at `alm_utils.py:2009-3437` is a 1,429 line function with 15 local functions, mixing ALM orchestration, result/history construction, penalty-state transitions, best-feasible restore, and the SciPy inner-solve call boundary.
- [ ] `Stage2ArtifactConfig` at `workflow_runner_common.py:119-219` carries 54 annotated fields spanning Stage 2 geometry, hardware/current controls, objective weights, ALM controls, basin hopping, finite-current wiring, iota mode, and target LCFS ceilings.

**Plan.** See `docs/alm_backlog5_structural_debt_plan_2026-05-07.md`.

**Acceptance.** Tracked in the separate Backlog 5 refactor plan; out of scope for the ALM hardening cycle.

## Validation Runner

Same as `docs/alm_hardening_closeout_todo_plan_2026-05-07.md:192-200`, plus the new TODO 3 regression assertion. Run after TODOs 1-4 land in the working tree, again after each commit in the commit-grouping pass.

```bash
.venv/bin/python -m pytest -q tests/geo/test_alm_utils.py
.venv/bin/python -m pytest -q tests/geo/test_single_stage_workflow_helpers.py
.venv/bin/python -m pytest -q tests/geo/test_single_stage_alm_integration.py
.venv/bin/python -m pytest -q tests/geo/test_single_stage_example.py \
  -k "AlmUtilsTests or build_alm_final_constraint_payload or alm_result_view_from_search_eval or stage2_main_alm_path_uses_minimize_alm or validate_resume_alm_state or current_solver_checkpoint_alm_state"
.venv/bin/python -m pytest -q tests/geo/test_constraint_contract.py tests/geo/test_banana_helper_modules.py
git diff --check
git grep -nE "\bALM_CLI_FIELDS\b" examples/ tests/
git grep -nE "\bALM_CLI_FIELDS\b" docs/alm_hardening_closeout_todo_plan_2026-05-07.md
```

Both grep invocations must return zero hits after TODO 1 + TODO 2. This followup doc itself is exempt because it describes the rename in prose.

## Final Acceptance

- [x] TODO 1: `ALM_CLI_FIELDS` renamed; Stage 2 default split resolved via Option A.
- [x] TODO 2: All 10 dependent references updated atomically with TODO 1.
- [x] TODO 3: `_build_constraint_metadata_tuples` helper extracted (length-mismatch ValueError preserved with the original message); `_attach_constraint_metadata` consumes precomputed tuples; three regression tests added and green — Test 1 (helper unit test, exact `str_calls == 1`), Test 2 (names scaling, no growth with `maxiter`), Test 3 (blocks scaling via monkeypatched helper, no growth with `maxiter`). Tests 2 and 3 use the `_make_fake_minimize` fixture to force inner-evaluation counts; Test 1 calls the helper directly and does not need it. The pre-existing length-mismatch regression test `tests/geo/test_alm_utils.py::test_minimize_alm_rejects_mismatched_constraint_block_metadata` continues to pass without modification.
- [x] TODO 4: Validation runner returns green; both CLI tuple deletion gates (live code + closeout tracker) return zero hits.
- [x] Six-commit grouping (Commits 1-6 above) executed via a `commit-only-work` pass on the working tree.
- [x] Original tracker `docs/alm_hardening_closeout_todo_plan_2026-05-07.md` updated post-merge so its commit-grouping section reflects the final six-commit layout.
- [x] Backlog items remain below as non-blocking debt and were not bundled as new implementation scope.

## Notes

- This document is the engineering-review followup; the original plan (`docs/alm_scalar_hardening_block_penalty_removal_plan_2026-05-06.md`) and tracker (`docs/alm_hardening_closeout_todo_plan_2026-05-07.md`) remain authoritative for the implementation phase history. The final commit grouping has been merged back into the closeout tracker.
- Pre-commit work scope is small: rename + 10 reference updates + ~5 lines for the hot-path fix + extraction of the `_build_constraint_metadata_tuples` helper (preserving the existing length-mismatch ValueError) + three new regression tests (helper unit, names scaling, blocks scaling).
- After TODO 3 lands, the three-test gate (Test 1 helper unit test, Test 2 names scaling, Test 3 blocks scaling) covers the full bug class. Test 1 catches direct precompute regressions; Test 2 catches closure-level names regressions including helper bypass; Test 3 catches closure-level blocks regressions such as `annotated["constraint_blocks"] = list(_constraint_blocks_tuple)` that would silently pass Tests 1 and 2. The scaling assertions (no growth with inner-iteration budget) are robust to non-hot-path consumers that legitimately stringify `constraint_names` per outer iteration (history diagnostics at `alm_utils.py:722`, result construction). Tests 2 and 3 force the inner-evaluation count via a stub `minimize` so they cannot false-pass on early convergence.
- Grep-gate scope: `examples/` and `tests/` for live code; `docs/alm_hardening_closeout_todo_plan_2026-05-07.md` for the closeout tracker. This followup doc is exempt because it discusses the rename in prose.
