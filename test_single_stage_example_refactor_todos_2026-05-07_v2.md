# test_single_stage_example.py — Refactor Todos v2 (2026-05-07)

> **Supersedes:** `test_single_stage_example_refactor_todos_2026-05-07.md`. Six validated findings from independent review folded in. Changes: Phase 1 split per-test with explicit coverage-loss decisions, all parametrization guidance switched from `pytest.mark.parametrize` to `unittest.subTest` (file is unittest-based), Phase 2 count corrected (5, not 6), Phase 5 inventory completed (5 cases, not 4), Phase 7 helper signature tightened (no `**rest`), `ResultsEnvelopeTests` removed from out-of-scope.

---

## Context

`tests/geo/test_single_stage_example.py` (HEAD `7f5e526ef`, dirty worktree) is structurally lopsided.

**Numbers:**
- 14,056 lines in one file
- 322 test methods across 13 top-level classes
- **`SingleStageExampleTests`**: 12,453-line megaclass at lines 184–12640, owning 277 tests plus 19 helpers (avg ~43 lines/test)
- 89 commits since 2026-03-16, +15,378 / −1,577 net; current uncommitted diff: +259 / −4
- All 7 top-level test classes inherit `unittest.TestCase`
- Zero `@pytest.mark.parametrize` uses; 6 `subTest` uses (all *outside* `SingleStageExampleTests`)
- 23 `captured = {}` dict-mutation sites (functional-programming smell)

**Why it grew this way:** append-only feature work. Each new feature (e.g. `b998fddcb feat: add single-stage scipy-jax fullgraph lane`, +820 lines) bolts more methods onto `SingleStageExampleTests` instead of refactoring. The current diff continues the pattern (5 new methods).

**What this plan does:** make explicit decisions on the 3 source-grep tests (write behavioral replacements OR accept coverage loss), parametrize 5 obvious DRY clusters via `subTest` (unittest-native), extract shared fixtures for optimizer-threading and results-envelope tests, then split `SingleStageExampleTests` into ~10 focused TestCases.

**Validation runtime contract** (from `CLAUDE.md`):
```bash
conda activate jax-0.9.2
ruff check tests/geo/test_single_stage_example.py
ruff format tests/geo/test_single_stage_example.py
conda run -n jax-0.9.2 python -m pytest tests/geo/test_single_stage_example.py -v
```

After every phase: full pass must stay green. Pre-existing mypy errors are expected; only zero-regression on touched files.

**Out of scope:**
- Touching production code under `examples/single_stage_optimization/` or `src/` (except where explicitly marked in Phase 1.3.a)
- Changing `tests/geo/surface_test_helpers.py` (used by other test files)
- Refactoring `BoozerFallbackLBFGSBTests`, `SegmentDistanceTests`, `HardwareConstraintTests`, `CrossSectionNormalizationTests`, `FtolGtolDefaultTests` — these are already small and focused
- Converting unittest.TestCase classes to bare pytest classes (separate decision; see Risk #1)

---

## Phase 0 — Baseline

- [ ] **0.1** Capture pytest baseline as the regression reference
  - Command: `conda run -n jax-0.9.2 python -m pytest tests/geo/test_single_stage_example.py -v --tb=line 2>&1 | tee /tmp/baseline_test_single_stage_example.log`
  - Record: total tests collected, passed, skipped, xfailed, time
  - **Acceptance:** baseline log saved; subsequent phases must not lose passing tests beyond the explicit deletions decided in Phase 1
- [ ] **0.2** Confirm `ruff check` baseline
  - Command: `ruff check tests/geo/test_single_stage_example.py`
  - **Acceptance:** zero new ruff findings introduced; pre-existing findings (if any) recorded for later diffing

---

## Phase 1 — Source-grep test decisions (KISS, per-test handling)

The 3 source-grep tests were flagged for deletion in v1 with claimed behavioral counterparts. **Validation showed those counterparts were not equivalent for any of the 3 cases.** Decide each separately.

### 1.1 `test_cpu_boozer_surface_zero_weight_contract_uses_explicit_none_check` (line 10940)

- **What it guards:** that `src/simsopt/geo/boozersurface.py:331` uses `constraint_weight is not None` (not the truthy form `if constraint_weight:`), so that `cw=0.0` still routes to `boozer_type="ls"`
- **Coverage status:** **likely insufficient.** Real-BoozerSurface tests `tests/geo/test_boozersurface.py:707/720` route through helper `tests/geo/surface_test_helpers.py:128`, which sets `cw = None if boozer_type == 'exact' else 100.0`. The helper only exercises `{None, 100.0}`, which catches None-vs-not-None branching but **not** the zero-weight truthiness regression — under a truthy-mutation, `None → exact` and `100.0 → ls` agree with the original; only `0.0` discriminates (`is not None → "ls"`, truthy → `"exact"`). No real-BoozerSurface test in the tree passes `cw=0.0`. **Expect the mutation gate in 1.1.a to fall through to 1.1.b.**
- [ ] **1.1.a** Run a mutation gate to empirically check whether the regression *is* covered elsewhere despite the analysis above. Locally patch `boozersurface.py:331` to `if constraint_weight else "exact"`, then run `conda run -n jax-0.9.2 python -m pytest tests/geo/test_boozersurface.py -v`. If any test fails, the regression is covered → **delete `test_cpu_boozer_surface_zero_weight_contract_uses_explicit_none_check`** and revert the patch. If all tests pass, the regression is uncovered → fall through to 1.1.b.
- [ ] **1.1.b** Keep the source-grep test but rename to make purpose explicit: `test_boozersurface_constraint_weight_branch_uses_is_not_none_check_regression`. Add a docstring referencing the zero-weight contract and noting that no behavioral test passes `cw=0.0`. Do not delete.

### 1.2 `test_plotting_utils_source_divides_by_2pi` (line 13491)

- **What it guards:** that `examples/single_stage_optimization/plotting_utils.py:125-126` uses `phi_slice / (2 * np.pi)`, not `phi_slice * 2 * np.pi`
- **Coverage status:** **no behavioral coverage exists.** `test_norm_field_summary_keeps_jax_field_on_host_reporting_boundary` in the same class tests a different function (`norm_field_summary`). Deleting the source-grep test loses all coverage of this normalization.
- [ ] **1.2.a** Write a behavioral replacement in `tests/geo/test_single_stage_example.py` inside `CrossSectionNormalizationTests`:
  - Mock `surf.cross_section` to capture the phi argument it receives
  - Call `cross_section_plot` with a known `phi_slice` value
  - Assert `surf.cross_section` was called with `phi_slice / (2 * np.pi)` (use `mock.call_args.args[0]` and assert tolerance vs the expected ratio)
  - This catches the regression *and* tests behavior, not source text
- [ ] **1.2.b** After 1.2.a passes, delete the original `test_plotting_utils_source_divides_by_2pi`
- [ ] **1.2.c** **If 1.2.a is more than ~30 lines or otherwise unwieldy** (e.g. mocking matplotlib is too invasive): keep the source-grep test, do not delete. Document the decision in the test's docstring.

### 1.3 `test_source_uses_default_argument` (line 13569)

- **What it guards:** that `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:11653` uses `ftol_by_mpol.get(mpol, <default>)`, not bare `ftol_by_mpol.get(mpol)`
- **Coverage status:** **the existing "behavioral" tests (`test_ftol_gtol_have_defaults_for_all_mpol`, `test_defaults_match_dictionary_endpoints`) re-implement the lookup in the test code itself** (line 13543: `ftol = ftol_by_mpol.get(mpol, 1e-5 if mpol < 8 else 1e-10)`). They do not actually call the deployed line at 11653. If the deployed code is changed to bare `.get(mpol)`, only the source-grep test catches it.
- [ ] **1.3.a** Write a behavioral replacement: extract the lookup logic into a helper (`def _resolve_outer_tolerances(args, mpol) -> tuple[float, float]`) inside `single_stage_banana_example.py`, replace the inline expression at line ~11653 with a call to that helper, then test the helper with `mpol` values not in either dictionary (e.g. `mpol=999`). This makes the deployed lookup directly testable.
  - **Note:** this requires a **production-code change** in `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`, which v1 listed as out-of-scope. Decide explicitly: extract helper or not.
- [ ] **1.3.b** **If the production-code change is rejected**, keep the source-grep test. It is the only thing guarding that line. Rename to `test_deployed_ftol_gtol_lookup_uses_default_argument_regression`.

### 1.4 Validate

- [ ] **1.4** Run pytest after Phase 1 decisions
  - Command: `conda run -n jax-0.9.2 python -m pytest tests/geo/test_single_stage_example.py -v 2>&1 | tail -20`
  - Diff vs baseline: each deleted test accounted for in the commit message; each retained source-grep test renamed and documented; new behavioral tests counted in the +/− delta

---

## Phase 2 — Parametrize argparse boolean-flag tests via `subTest` (DRY, ~120 → ~30 lines)

7 tests at lines 4621–4730 each invoke `parse_args()` with a single `--flag` and assert `args.<attr> is True`. They test argparse, not project logic.

- [ ] **2.1** Inventory:
  - `test_parse_args_accepts_diagnose_target_lane_gradient` (line 4621)
  - `test_parse_args_accepts_diagnose_target_lane_first_line_search` (line 4639)
  - `test_parse_args_accepts_diagnose_target_lane_scaled_phase1` (line 4657)
  - `test_parse_args_accepts_record_target_lane_invalid_state_events` (line 4675)
  - `test_parse_args_accepts_diagnostic_callbacks` (line 4692) — **keep separate**: also asserts implication that `record_target_lane_invalid_state_events` is set
  - `test_parse_args_accepts_minimal_artifacts` (line 4710)
  - `test_parse_args_accepts_full_artifacts` (line 4722)
- [ ] **2.2** Replace 6 of 7 with one `subTest`-parametrized method (file uses `unittest.TestCase`, so `subTest` is the native idiom):
  ```python
  def test_parse_args_accepts_boolean_flags(self):
      module = self.load_module()
      cases = [
          ("--diagnose-target-lane-gradient",          "diagnose_target_lane_gradient"),
          ("--diagnose-target-lane-first-line-search", "diagnose_target_lane_first_line_search"),
          ("--diagnose-target-lane-scaled-phase1",     "diagnose_target_lane_scaled_phase1"),
          ("--record-target-lane-invalid-state-events", "record_target_lane_invalid_state_events"),
          ("--minimal-artifacts",                      "minimal_artifacts"),
          ("--full-artifacts",                         "full_artifacts"),
      ]
      for flag, attr in cases:
          with self.subTest(flag=flag):
              with patch.dict(os.environ, {}, clear=True), patch.object(
                  sys, "argv",
                  ["single_stage_banana_example.py", "--backend", "jax", flag],
              ):
                  args = module.parse_args()
              self.assertTrue(getattr(args, attr))
  ```
- [ ] **2.3** Validate
  - Pytest collection: **5 fewer tests** (7 originals → 1 parametrized + 1 kept separate = 2)
  - All 7 flag-attribute pairs still asserted

---

## Phase 3 — Parametrize `parse_args` defaults / lane tests via `subTest` (DRY, ~500 → ~150 lines)

~12 tests at lines 1685–1955 follow the pattern: `argv = [..., "--backend", "jax", "--optimizer-backend", X]` → `assertEqual(args.<various>, ...)`.

- [ ] **3.1** Inventory:
  - `test_parse_args_does_not_treat_optimizer_env_as_explicit_boozer_override` (1671)
  - `test_parse_args_defaults_jax_backend_to_ondevice_optimizer_lane` (1685)
  - `test_parse_args_scipy_jax_outer_defaults_boozer_to_ondevice` (1701)
  - `test_parse_args_scipy_jax_fullgraph_outer_defaults_boozer_to_ondevice` (1723)
  - `test_parse_args_rejects_scipy_jax_fullstate_outer_backend` (1745)
  - `test_parse_args_preserves_cpu_default_reference_lane` (1762)
  - `test_parse_args_defaults_boozer_algorithm_from_explicit_inner_backend` (1778)
  - `test_parse_args_accepts_boozer_limited_memory_override` (1800)
  - `test_parse_args_defaults_target_lane_sync_to_final_only` (1817)
  - `test_parse_args_defaults_target_lane_outer_maxls_to_tighter_budget` (1833)
  - `test_parse_args_benchmark_mode_uses_target_lane_trial_defaults` (1854)
  - `test_parse_args_fullgraph_preserves_cpu_boozer_newton_default` (1876, in current uncommitted diff)
  - `test_parse_args_preserves_reference_outer_maxls_default` (1895)
  - `test_parse_args_marks_explicit_initial_phase_defaults` (1917)
  - `test_parse_args_marks_explicit_stage2_bs_path` (1938)
- [ ] **3.2** Extract a tiny helper into the host class:
  ```python
  def _parse_argv(self, module, *flags):
      with patch.dict(os.environ, {}, clear=True), patch.object(
          sys, "argv", ["single_stage_banana_example.py", *flags],
      ):
          return module.parse_args()
  ```
- [ ] **3.3** Group into 2–3 `subTest`-parametrized methods grouped by assertion shape:
  - One for *outer-backend → defaults*: `(scipy-jax, scipy-jax-fullgraph, ondevice, cpu)` × tuple of `(optimizer_backend, boozer_optimizer_backend, boozer_least_squares_algorithm, ...)`
  - One for *invalid backend rejection*: each invalid value → `SystemExit`
  - Keep tests with unique semantics (`marks_explicit_*`, `benchmark_mode_uses_target_lane_trial_defaults`) as-is — different assertion shape
- [ ] **3.4** Validate

---

## Phase 4 — Parametrize `resolve_*_boozer_init_overrides_*` via `subTest` (DRY, ~200 → ~60 lines)

9 tests at lines 1059–1259 differ only in input args / asserted output keys. Same function, same harness.

- [ ] **4.1** Inventory:
  - `test_resolve_warm_start_boozer_init_overrides_is_empty_without_warm_start` (1059)
  - `test_resolve_target_lane_boozer_init_base_overrides_uses_target_lane_defaults` (1085)
  - `test_resolve_target_lane_boozer_init_base_overrides_floors_bfgs_tol` (1111)
  - `test_resolve_target_lane_boozer_init_base_overrides_is_empty_off_target_lane` (1129)
  - `test_resolve_target_lane_boozer_init_base_overrides_floors_bfgs_maxiter` (1155)
  - `test_resolve_target_lane_boozer_init_base_overrides_skips_full_memory_newton_floor_for_lbfgs` (1173)
  - `test_resolve_warm_start_boozer_init_overrides_keeps_explicit_surface_algorithm` (1191)
  - `test_resolve_warm_start_boozer_init_overrides_uses_quasi_newton_for_legacy_path` (1219)
  - `test_resolve_warm_start_boozer_init_overrides_preserves_explicit_algorithm` (1240)
- [ ] **4.2** Convert each to a `subTest` row of `(input_kwargs, expected_overrides_subset)`. Two parametrized methods (one per resolver function) is cleaner than mixing.
- [ ] **4.3** Validate

---

## Phase 5 — Parametrize `resolve_scaled_outer_phase_final_dofs_*` via `subTest` (DRY, ~120 → ~40 lines)

The full cluster is **5 cases**, not 4 (v1 missed line 2056). All call `resolve_scaled_outer_phase_final_dofs(anchor, step, 0.25, use_target_lane=True)` with the same expected `[10.5, 19.0]` result.

- [ ] **5.1** Inventory:
  - `test_resolve_scaled_outer_phase_final_dofs_target_lane_is_transfer_safe` (line **2056**) — device anchor, host step, raw inputs ← v1 missed this
  - `test_resolve_scaled_outer_phase_final_dofs_target_lane_host_anchor_device_step_is_transfer_safe` (2074)
  - `test_resolve_scaled_outer_phase_final_dofs_target_lane_host_anchor_host_step_is_transfer_safe` (2094)
  - `test_resolve_scaled_outer_phase_final_dofs_target_lane_scaled_state_host_anchor_is_transfer_safe` (2115)
  - `test_resolve_scaled_outer_phase_final_dofs_target_lane_scaled_state_host_anchor_host_step_is_transfer_safe` (2137)
- [ ] **5.2** Replace with one `subTest` parametrized method over `(anchor_placement, step_placement, use_state_wrapper, expect_jax_array)`:
  ```python
  def test_resolve_scaled_outer_phase_final_dofs_transfer_safe_grid(self):
      module = self.load_module()
      cases = [
          # (anchor_kind, step_kind, use_state_wrapper, assert_jax_array)
          ("device", "host",   False, False),  # line 2056 case
          ("host",   "device", False, False),  # line 2074
          ("host",   "host",   False, True),   # line 2094 — asserts isinstance jax.Array
          ("zero",   "device", True,  False),  # line 2115
          ("zero",   "host",   True,  True),   # line 2137 — asserts isinstance jax.Array
      ]
      for anchor_kind, step_kind, use_wrapper, assert_jax in cases:
          with self.subTest(anchor=anchor_kind, step=step_kind, wrapper=use_wrapper):
              # build anchor/step per kind, optionally wrap in ScaledOuterPhaseOptimizerState,
              # call function under transfer_guard("disallow"), assert result == [10.5, 19.0],
              # if assert_jax: assertIsInstance(final_dofs, jax.Array)
              ...
  ```
- [ ] **5.3** Validate

---

## Phase 6 — Merge `initialize_boozer_surface_threads_*` (DRY/YAGNI, ~150 → ~40 lines)

4 tests verify that kwargs pass through to `BoozerSurfaceJAX.options`. Pass-through plumbing, not logic.

- [ ] **6.1** Inventory:
  - `test_initialize_boozer_surface_threads_nondefault_optimizer_backend` (903)
  - `test_initialize_boozer_surface_limited_memory_disables_dense_linearization` (930)
  - `test_initialize_boozer_surface_threads_nondefault_least_squares_algorithm` (958)
  - `test_initialize_boozer_surface_threads_solver_budget_overrides` (985)
- [ ] **6.2** Replace with one test that passes *all* kwargs simultaneously and asserts the merged options dict:
  ```python
  def test_initialize_boozer_surface_threads_all_jax_options(self):
      module = self.load_module()
      surf_prev = FakeSurfPrev()
      fake_jax = self.build_fake_boozer_surface_jax_class(record_run_calls=False)
      with self.patch_initialize_boozer_surface_jax(module, fake_jax):
          module.initialize_boozer_surface(
              surf_prev,
              mpol=TEST_MPOL, ntor=TEST_NTOR, bs=object(),
              vol_target=TEST_VOL_TARGET, constraint_weight=1.0,
              iota=TEST_IOTA, G0=TEST_G0,
              backend="jax",
              optimizer_backend="ondevice",
              boozer_least_squares_algorithm="lm",
              boozer_limited_memory=True,
              bfgs_tol_override=3.0e-6, bfgs_maxiter_override=32,
              newton_tol_override=1.0e-7, newton_maxiter_override=9,
          )
      options = fake_jax.instances[0].options
      self.assertEqual(options["optimizer_backend"], "ondevice")
      self.assertEqual(options["least_squares_algorithm"], "lm")
      self.assertIs(options["materialize_dense_linearization"], False)
      self.assertIs(options["force_ondevice_limited_memory"], True)
      self.assertEqual(options["bfgs_tol"], 3.0e-6)
      self.assertEqual(options["bfgs_maxiter"], 32)
      self.assertEqual(options["newton_tol"], 1.0e-7)
      self.assertEqual(options["newton_maxiter"], 9)
  ```
- [ ] **6.3** Validate

---

## Phase 7 — Extract shared `Mock(spec=...)`-based fakes for `run_single_stage_optimizer_threads_*` (DRY, ~290 → ~100 lines)

7 tests at lines 7934–8412 each rebuild the same `explicit_fun`, `fake_require_target_backend_x64`, `fake_jax_minimize` skeleton with explicit kwarg signatures. The current fakes raise `TypeError` if `run_single_stage_optimizer` calls `jax_minimize(...)` with an **unexpected** kwarg — that strict-signature check is part of the test contract, not just boilerplate.

**Critical:** a bare `Mock(return_value=...)` accepts arbitrary kwargs silently and would *silently lose* this contract. v1 sketched a helper using `**rest` with the same problem. The helper must either:
- Use `Mock(spec=jax_minimize_signature_fn)` (autospec) — `Mock` raises `TypeError` on unexpected kwargs, matching the original fake's behavior, OR
- Use a bare `Mock` *and* every test asserts the **complete** kwarg keyset via `set(mock.call_args.kwargs.keys()) == EXPECTED_KEYS`.

This plan picks the autospec route — it scales without per-test bookkeeping.

- [ ] **7.1** Add a helper that returns a `Mock(spec=...)` autospecced against the real `jax_minimize` signature:
  ```python
  @staticmethod
  def _make_autospec_jax_minimize(*, return_x=None, return_nit=0, return_message="ok"):
      """Returns a Mock autospecced against optimizer_jax.jax_minimize.
      Unexpected kwargs raise TypeError, matching the original strict-signature fake."""
      from simsopt.geo import optimizer_jax  # imported here to keep top-of-file lean
      x = np.zeros(2) if return_x is None else np.asarray(return_x)
      mock = Mock(spec=optimizer_jax.jax_minimize)
      mock.return_value = types.SimpleNamespace(x=x, nit=return_nit, message=return_message)
      return mock
  ```
  - **Why `Mock(spec=fn)` and not bare `Mock()`:** the original fakes (e.g. `tests/geo/test_single_stage_example.py:7945`) declare an explicit signature `def fake_jax_minimize(fun, x0, *, method, tol, maxiter, options, value_and_grad, callback, progress_callback=None, failure_callback=None)`. If `jax_minimize` grows a new required kwarg, calling the fake without it raises `TypeError`; calling with an unexpected one *also* raises `TypeError`. A bare `Mock(return_value=...)` accepts both silently and the contract is lost. `Mock(spec=fn)` preserves both directions of the contract.
  - **Fallback if autospec is impractical** (e.g. `optimizer_jax.jax_minimize` import has side effects in this test file): use bare `Mock` and add a per-test assertion `assert set(fake_minimize.call_args.kwargs.keys()) == EXPECTED_LANE_KWARGS`. Document that decision in the helper docstring.
- [ ] **7.2** Refactor each test:
  - `test_run_single_stage_optimizer_threads_target_lane_failure_callback` (7934)
  - `test_run_single_stage_optimizer_scipy_jax_omits_private_diagnostics` (7995)
  - `test_run_single_stage_optimizer_fullgraph_uses_full_optimizer_vector` (8065)
  - `test_run_single_stage_optimizer_threads_target_lane_progress_callback` (8123)
  - `test_run_single_stage_optimizer_threads_reference_trace_contract` (8218)
  - `test_run_single_stage_optimizer_threads_target_lane_initial_step_size` (8294)
  - `test_run_single_stage_optimizer_threads_target_lane_initial_value_and_grad` (8354)
  - Pattern:
    ```python
    fake_minimize = self._make_mock_jax_minimize()
    fake_require = Mock()
    with self.patch_optimizer_jax_module(
        require_target_backend_x64=fake_require,
        jax_minimize=fake_minimize,
    ):
        result = module.run_single_stage_optimizer(...)
    self.assertEqual(fake_minimize.call_args.kwargs["method"], "lbfgs-scipy-jax")
    self.assertIs(fake_minimize.call_args.kwargs["failure_callback"], failure_callback)
    ```
- [ ] **7.3** Validate
- [ ] **7.4** **Contract regression check (both directions):** after refactor, run two local mutations to confirm the strict-signature contract survived. Revert each before continuing.
  - **Direction A — dropped kwarg:** temporarily change `run_single_stage_optimizer` to drop one kwarg from its `jax_minimize(...)` call. Affected tests should fail (because `mock.call_args.kwargs[<dropped>]` raises `KeyError`).
  - **Direction B — unexpected kwarg:** temporarily change `run_single_stage_optimizer` to pass an extra kwarg `jax_minimize(..., __probe__="x")` to its `jax_minimize(...)` call. Affected tests should fail with `TypeError` from the autospec mock (or with the explicit-keyset assertion if the fallback path was taken).
  - If only Direction A fails and Direction B silently passes, the autospec is not active — fix the helper before merging. The original `del`-block fakes caught both directions; the new helper must too.

---

## Phase 8 — Extract `build_single_stage_results_envelope` fixture (DRY, ~280 → ~120 lines)

3 tests in `ResultsEnvelopeTests` (lines 13762, 13881, 13974) each build a `SimpleNamespace` of ~16 args and call `build_single_stage_results_envelope` with **30+ kwargs**, of which only 2-3 differ per test.

> **Note:** v1 listed `ResultsEnvelopeTests` as out-of-scope, but Phase 8 touches it — the contradiction is fixed in v2: it is in scope. The class is small (3 tests) but each test is large (~90 lines), and the mismatch between "3 tests, 280 lines" justifies fixture extraction.

- [ ] **8.1** Extract `_default_envelope_kwargs(self, **overrides) -> dict` returning the merged kwargs dict
- [ ] **8.2** Extract `_default_args_namespace(self, **overrides) -> SimpleNamespace` for the args object
- [ ] **8.3** Refactor each test:
  ```python
  envelope = module.build_single_stage_results_envelope(
      **self._default_envelope_kwargs(
          # only the 2-3 keys that matter for this test's assertion
          args=self._default_args_namespace(diagnose_target_lane_scaled_phase1=True),
      )
  )
  ```
- [ ] **8.4** Validate

---

## Phase 9 — Split `SingleStageExampleTests` into focused TestCases (SRP, structural)

After Phases 1–8 the megaclass is shorter but still owns ~240 tests. Split by concern. **Move tests in groups of ~25–40, validate after each group, commit after each green pass.** Do not move all in one shot.

- [ ] **9.1** `ParseArgsTests` — ~25 `test_parse_args_*` (and the post-Phase 3 parametrized ones)
- [ ] **9.2** `BoozerSurfaceInitTests` — ~15 `test_initialize_boozer_surface_*`
- [ ] **9.3** `BoozerSurfaceWarmStartTests` — `test_resolve_warm_start_boozer_init_*`, `test_resolve_target_lane_boozer_init_*`
- [ ] **9.4** `RuntimeSpecBiotSavartTests` — `test_runtime_spec_biotsavart_*`, `test_jax_warm_start_surface_dofs_require_seed_spec_artifact`
- [ ] **9.5** `ScaledOuterProblemTests` — ~17 `test_build_scaled_outer_*`, `test_resolve_scaled_outer_*`, `test_build_target_lane_scaled_phase1_diagnosis_*`
- [ ] **9.6** `TargetLaneRetryTests` — ~12 `test_run_single_stage_target_lane_optimizer_with_retries_*`
- [ ] **9.7** `SnapshotRestoreTests` — `test_snapshot_*`, `test_snapshot_to_pytree_*`, `test_snapshot_restore_round_trip`
- [ ] **9.8** `SingleStageAdapterTests` — `test_single_stage_adapter_*`, `test_accept_step_*`, `test_objective_evaluation_trace_*`, `test_single_stage_objective_*`
- [ ] **9.9** `FailurePenaltyTests` — `test_failure_penalty_*`, `test_compute_single_stage_failure_penalty_*`, the two newly-added (current diff) reject-class tests
- [ ] **9.10** `ReportingMetricsTests` — anything touching `_make_reporting_runtime_summary`, `_make_reporting_runtime_builder`, target-restart artifact tests
- [ ] **9.11** `OptimizerThreadingTests` — `test_run_single_stage_optimizer_threads_*`, `_omits_private_diagnostics`, `_fullgraph_uses_full_optimizer_vector`, `_target_lane_requires_objective_contract`
- [ ] **9.12** Helper migration decisions (per-helper):
  - Pure `@staticmethod` with no class state → move to module-level functions
  - References `self.load_module()` → keep on a small shared base TestCase (e.g. `_SingleStageTestBase(unittest.TestCase)`) that all new classes inherit from
  - `FakeSurfaceXYZTensorFourier.instances = []` setUp pattern (line 186) → must move with any test that triggers `FakeSurfaceXYZTensorFourier`
- [ ] **9.13** Validate after each move group; commit each group separately

---

## Phase 10 — Replace `captured = {}` with `Mock.call_args` (FP, ~23 sites)

Phase 7 already converts the optimizer-threading cluster (~7 sites). This phase rolls the same pattern across the remaining ~16 sites.

- [ ] **10.1** Inventory remaining `captured = {}` sites (post-Phase 7):
  - Command: `grep -n "captured = {}" tests/geo/test_single_stage_example.py`
- [ ] **10.2** Convert in batches of ~5; validate each batch
- [ ] **10.3** **Keep `captured = {}` where:**
  - The test asserts on the *order* of multiple calls and `Mock.call_args_list` would be more verbose than the dict
  - The test needs to mutate state across multiple closure invocations (i.e. the closure does real work, not just record args)
  - SSOT/FP is a guideline, not a sledgehammer — if the dict is genuinely the cleanest representation, document it and keep it

---

## Phase 11 — Final cleanup

- [ ] **11.1** Run `ruff check` and `ruff format` over the entire file; resolve any new findings
- [ ] **11.2** Verify final stats:
  - Expected: ~9,500–10,500 lines (down from 14,056)
  - Expected: ~10–12 TestCases instead of 1 megaclass
  - Expected test count: baseline minus deletions decided in Phase 1, minus parametrize merges (each documented in commit messages), plus any new behavioral replacements from Phase 1.2.a / 1.3.a
- [ ] **11.3** Update `CLAUDE.md` "Code Review History" with a brief note on the refactor (file split, parametrization conventions adopted, source-grep test decisions)
- [ ] **11.4** Run the full validation per `CLAUDE.md`:
  ```bash
  conda run -n jax-0.9.2 python -m pytest \
    tests/test_jax_import_smoke.py \
    tests/field/test_biotsavart_jax.py \
    tests/geo/test_surface_fourier_jax.py \
    tests/geo/test_boozer_residual_jax.py \
    tests/objectives/test_integral_bdotn_jax.py \
    tests/geo/test_boozer_derivatives_jax.py \
    tests/geo/test_boozersurface_jax.py \
    tests/integration/test_jax_native_path.py \
    tests/geo/test_single_stage_example.py \
    -m "not private_optimizer_runtime" -v
  ```
- [ ] **11.5** Final commit with co-author trailer per project conventions

---

## Risk callouts

1. **`unittest.TestCase` vs pytest parametrization.** All test classes in this file inherit `unittest.TestCase`. `pytest.mark.parametrize` *technically* works on unittest methods but with sharp edges (no per-case method names in collection, surprises with `setUp`, fragile IDE discovery). **This plan uses `subTest` exclusively** for parametrization. If a future phase wants `pytest.mark.parametrize`, it must first convert the host class to bare pytest style (drop the `(unittest.TestCase)` parent, replace `self.assertX` with `assert`, replace `self.subTest` etc.) — a separate decision out of this plan's scope.

2. **`importlib`-based module loading.** `load_single_stage_example_module()` (line 71) generates a fresh module name with `uuid.uuid4().hex` per call. Splitting tests across classes preserves this; do **not** consolidate into a class-level `setUpClass` cached module — some tests intentionally monkey-patch module-level functions (e.g. `module.build_runtime_provenance = lambda **_: {"repo_sha": "deadbeef"}` at line 13764), and a shared cached module would leak that patch across tests.

3. **Source-grep tests are not always deletable.** Validation showed v1's blanket "behavioral counterparts exist" claim was wrong for at least 2 of 3 cases. Phase 1 now decides each test individually with explicit acceptance of either (a) writing a real behavioral replacement, (b) accepting coverage loss, or (c) keeping the source-grep test renamed and documented.

4. **Helper move-out.** The 19 helpers on `SingleStageExampleTests` use `self.` and `@staticmethod` interchangeably. When moving tests to new classes, decide per helper (Phase 9.12). Moving helpers prematurely without their dependent tests creates phantom imports.

5. **`subTest` failure visibility.** When merging multiple tests via `subTest`, a failure in one row reports as a single test failure with sub-case detail — but the test method name is collapsed. If CI dashboards or flaky-test trackers rely on individual method names, that tooling loses granularity. Confirm with the team before Phase 2/3/4/5 if dashboards key on names.

6. **Current uncommitted diff.** The +259/−4 diff added 5 new methods to `SingleStageExampleTests` and updated existing hardware-constraint tests. **Decide before starting Phase 0**: either (a) commit the current diff first and refactor on top (safer), or (b) rebase the new tests into the new structure as part of Phase 9 (cleaner history, riskier).

7. **No production-code changes (with one explicit exception in Phase 1.3.a).** This refactor is purely test-file restructuring, with the single explicit possible exception in Phase 1.3.a (extracting a `_resolve_outer_tolerances` helper from `single_stage_banana_example.py` so the deployed lookup becomes directly testable). That production change must be a separate, explicit decision.

8. **Parametrize regressions.** When merging multiple tests into one parametrized test, if the "duplicate" was hiding a real difference (e.g. one test passed `--backend cpu` and the other `--backend jax` and they happened to share an assertion that's actually backend-sensitive), the merge collapses real coverage. Sanity check: read each test in a cluster end-to-end before merging, not just the assertion shape.

---

## Quick-reference: line-savings estimate

| Phase | Cluster | Today | After | Net |
|---|---|---:|---:|---:|
| 1 | source-grep tests (varies) | ~30 | 0–60 | −30 to +30 |
| 2 | argparse boolean flags | ~120 | ~30 | −90 |
| 3 | parse_args defaults/lanes | ~500 | ~150 | −350 |
| 4 | resolve_*_overrides | ~200 | ~60 | −140 |
| 5 | scaled_outer transfer-safe (5 cases) | ~120 | ~40 | −80 |
| 6 | initialize_boozer_surface_threads | ~150 | ~40 | −110 |
| 7 | optimizer_threads_* (Mock) | ~290 | ~100 | −190 |
| 8 | results_envelope fixtures | ~280 | ~120 | −160 |
| 10 | captured-dict → Mock (post-7) | ~250 | ~150 | −100 |
| **Sum** | | **~1,940** | **~690–750** | **−1,190 to −1,250** |

Phase 9 (split) doesn't reduce line count materially but reduces *cognitive* size: each new TestCase is reviewable in isolation.

---

## Definition of Done

- [ ] All 11 phases checked off
- [ ] Pytest baseline (Phase 0.1) shows the same passing test set, modulo:
  - Tests deleted by explicit decision in Phase 1 (each documented in commit message)
  - Tests merged by parametrization (each merge documented)
  - New behavioral tests added in Phase 1.2.a / 1.3.a (each accounted for)
- [ ] `ruff check` clean on the file
- [ ] No production-code under `src/` modified; `examples/` modified only if Phase 1.3.a was explicitly approved
- [ ] `CLAUDE.md` updated with new file/class layout
- [ ] Each phase committed separately with a clear message; final history readable as a sequence of mechanical refactors

---

## Validation history

- **v1 → v2 (2026-05-07):** independent review flagged 6 issues; all validated against current code:
  1. Phase 1 behavioral-counterpart claims wrong for all 3 source-grep tests → Phase 1 split per-test with explicit decisions
  2. `pytest.mark.parametrize` guidance unsafe for `unittest.TestCase` file → switched to `subTest`; pytest-class conversion called out as separate scope
  3. Phase 2 count: 7 → 2 is drop of 5, not 6 → fixed
  4. Phase 5 missed line 2056 (a 5th case) → inventory expanded to 5 cases; grid description corrected
  5. Out-of-scope contradicted Phase 8 (`ResultsEnvelopeTests`) → removed from out-of-scope
  6. Phase 7 helper used `**rest` (silently swallowing new kwargs vs. original `del` contract) → switched to `Mock.call_args.kwargs` with explicit Phase 7.4 contract-regression check
- **v2 patch (2026-05-07, post-second-review):** three additional findings folded in:
  1. Phase 7 still recommended bare `Mock()`, which silently accepts unexpected kwargs and does *not* preserve the original fake's strict-signature contract → switched to `Mock(spec=jax_minimize)` autospec; corrected error-type wording from `NameError` to `TypeError`; Phase 7.4 expanded to a two-direction contract check (dropped kwarg + unexpected kwarg)
  2. Out-of-scope line 35 referenced "Phase 1.2" for the production-code exception, but the actual exception is in Phase 1.3.a → fixed
  3. Phase 1.1 overstated existing coverage: cited tests use `cw ∈ {None, 100.0}` (helper line 128), which catches None-vs-not-None branching but cannot discriminate the truthy-mutation regression that the source-grep test specifically guards (only `cw=0.0` discriminates). Rewritten to acknowledge the gap and note the mutation gate is likely to fall through to 1.1.b
