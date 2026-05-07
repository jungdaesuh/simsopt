# test_single_stage_example.py — Refactor Todos (2026-05-07)

## Context

`tests/geo/test_single_stage_example.py` (HEAD `7f5e526ef`, dirty worktree) is structurally lopsided.

**Numbers:**
- 14,056 lines in one file
- 322 test methods across 13 top-level classes
- **`SingleStageExampleTests` alone**: 12,453-line megaclass spanning lines 184–12640, owning 277 of the 322 tests plus 19 helpers (avg ~43 lines/test)
- 89 commits since 2026-03-16, +15,378 / −1,577 net; current uncommitted diff: +259 / −4
- Sibling files use `pytest.mark.parametrize` heavily; this file has only 6 `subTest` uses, all *outside* the megaclass
- 23 `captured = {}` dict-mutation patterns (functional-programming smell)

**Why it grew this way:** append-only feature work. Each new feature (e.g. `b998fddcb feat: add single-stage scipy-jax fullgraph lane`, +820 lines) bolts more methods onto `SingleStageExampleTests` instead of refactoring. The current diff continues the pattern (5 new methods, all near-duplicates of existing ones).

**What this plan does:** delete the 3 source-grep tests outright, parametrize 5 obvious DRY clusters, extract shared fixtures for the optimizer-threading and results-envelope tests, then split `SingleStageExampleTests` into ~10 focused TestCases — without losing a single behavioral assertion.

**Validation runtime contract** (from `CLAUDE.md`):
```bash
conda activate jax-0.9.2
ruff check tests/geo/test_single_stage_example.py
ruff format tests/geo/test_single_stage_example.py
conda run -n jax-0.9.2 python -m pytest tests/geo/test_single_stage_example.py -v
```

After every phase: full pass must stay green. Pre-existing mypy errors are expected; only zero-regression on touched files.

**Out of scope:**
- Touching the production code under `examples/single_stage_optimization/`
- Changing `tests/geo/surface_test_helpers.py` (used by other test files)
- Refactoring `BoozerFallbackLBFGSBTests`, `SegmentDistanceTests`, `HardwareConstraintTests`, `CrossSectionNormalizationTests`, `FtolGtolDefaultTests`, `ResultsEnvelopeTests` — these are already small and focused (delete the source-grep tests inside them; otherwise leave alone)

---

## Phase 0 — Baseline

- [ ] **0.1** Capture current pytest pass count and timing as the regression baseline
  - Command: `conda run -n jax-0.9.2 python -m pytest tests/geo/test_single_stage_example.py -v --tb=line 2>&1 | tee /tmp/baseline_test_single_stage_example.log`
  - Record: total tests collected, passed, skipped, xfailed, time
  - **Acceptance:** baseline log saved; subsequent phases must not lose passing tests
- [ ] **0.2** Confirm `ruff check` is clean on the file at HEAD before changes
  - Command: `ruff check tests/geo/test_single_stage_example.py`
  - **Acceptance:** zero new ruff findings; pre-existing findings recorded for diffing later

---

## Phase 1 — Delete the source-grep tests (KISS, ~30 lines saved)

These tests grep the project's own source files for substrings. They're brittle (renames break them without changing behavior), tautological (they don't verify behavior), and each has a behavioral counterpart elsewhere.

- [ ] **1.1** Delete `test_cpu_boozer_surface_zero_weight_contract_uses_explicit_none_check`
  - Location: `tests/geo/test_single_stage_example.py:10940`
  - Body greps `src/simsopt/geo/boozersurface.py` for the literal string `"constraint_weight is not None"`
  - Behavioral coverage already exists: `test_initialize_boozer_surface_zero_constraint_weight_keeps_least_squares_path` (line 621) exercises the actual `constraint_weight=None` branch
  - **Acceptance:** test removed; behavioral test still passes
- [ ] **1.2** Delete `test_plotting_utils_source_divides_by_2pi`
  - Location: `tests/geo/test_single_stage_example.py:13491`
  - Body greps `examples/single_stage_optimization/plotting_utils.py` for `"phi_slice / (2 * np.pi)"` vs `"phi_slice * 2 * np.pi"`
  - Class docstring says "Issue #8/#9" — regression guard for a long-fixed bug. Git history captures the fix.
  - **Acceptance:** test removed; `test_norm_field_summary_keeps_jax_field_on_host_reporting_boundary` (the behavioral test in the same class) still passes
- [ ] **1.3** Delete `test_source_uses_default_argument`
  - Location: `tests/geo/test_single_stage_example.py:13569`
  - Body greps `EXAMPLE_MODULE_PATH` for `"ftol_by_mpol.get(mpol)"` and `"gtol_by_mpol.get(mpol)"`
  - Behavioral coverage already exists in same `FtolGtolDefaultTests` class: `test_ftol_gtol_have_defaults_for_all_mpol` and `test_defaults_match_dictionary_endpoints` exercise the actual lookup behavior
  - **Acceptance:** test removed; the two behavioral siblings still pass
- [ ] **1.4** Validate
  - Command: `conda run -n jax-0.9.2 python -m pytest tests/geo/test_single_stage_example.py -v 2>&1 | tail -20`
  - Diff vs baseline: exactly **3 fewer tests collected**, zero failures, zero new errors

---

## Phase 2 — Parametrize argparse boolean-flag tests (DRY, ~120 → ~30 lines)

7 tests at lines 4621–4730 each invoke `parse_args()` with a single `--flag` and assert `args.<attr> is True`. They test argparse, not project logic.

- [ ] **2.1** Identify the full set
  - `test_parse_args_accepts_diagnose_target_lane_gradient` (line 4621)
  - `test_parse_args_accepts_diagnose_target_lane_first_line_search` (line 4639)
  - `test_parse_args_accepts_diagnose_target_lane_scaled_phase1` (line 4657)
  - `test_parse_args_accepts_record_target_lane_invalid_state_events` (line 4675)
  - `test_parse_args_accepts_diagnostic_callbacks` (line 4692) — note: also asserts `record_target_lane_invalid_state_events` is implied; keep that secondary assertion
  - `test_parse_args_accepts_minimal_artifacts` (line 4710)
  - `test_parse_args_accepts_full_artifacts` (line 4722)
- [ ] **2.2** Replace with one parametrized test using `unittest.subTest`
  - Sketch:
    ```python
    def test_parse_args_accepts_boolean_flags(self):
        module = self.load_module()
        cases = [
            ("--diagnose-target-lane-gradient",     "diagnose_target_lane_gradient"),
            ("--diagnose-target-lane-first-line-search", "diagnose_target_lane_first_line_search"),
            ("--diagnose-target-lane-scaled-phase1", "diagnose_target_lane_scaled_phase1"),
            ("--record-target-lane-invalid-state-events", "record_target_lane_invalid_state_events"),
            ("--minimal-artifacts", "minimal_artifacts"),
            ("--full-artifacts", "full_artifacts"),
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
  - Keep `test_parse_args_accepts_diagnostic_callbacks` separate (it has a non-trivial implication assertion)
- [ ] **2.3** Validate
  - Pytest must show **6 fewer tests** but identical behavior coverage; same set of CLI flags is asserted

---

## Phase 3 — Parametrize `parse_args` defaults / lane tests (DRY, ~500 → ~150 lines)

~12 tests at lines 1685–1955 follow the pattern: `argv = ["…", "--backend", "jax", "--optimizer-backend", X]` → `assertEqual(args.<various>, ...)`. Covers default, scipy-jax, scipy-jax-fullgraph, ondevice, cpu, fullstate-rejection, etc.

- [ ] **3.1** Inventory the cluster (all are inside `SingleStageExampleTests`):
  - `test_parse_args_does_not_treat_optimizer_env_as_explicit_boozer_override` (line 1671)
  - `test_parse_args_defaults_jax_backend_to_ondevice_optimizer_lane` (line 1685)
  - `test_parse_args_scipy_jax_outer_defaults_boozer_to_ondevice` (line 1701)
  - `test_parse_args_scipy_jax_fullgraph_outer_defaults_boozer_to_ondevice` (line 1723)
  - `test_parse_args_rejects_scipy_jax_fullstate_outer_backend` (line 1745)
  - `test_parse_args_preserves_cpu_default_reference_lane` (line 1762)
  - `test_parse_args_defaults_boozer_algorithm_from_explicit_inner_backend` (line 1778)
  - `test_parse_args_accepts_boozer_limited_memory_override` (line 1800)
  - `test_parse_args_defaults_target_lane_sync_to_final_only` (line 1817)
  - `test_parse_args_defaults_target_lane_outer_maxls_to_tighter_budget` (line 1833)
  - `test_parse_args_benchmark_mode_uses_target_lane_trial_defaults` (line 1854)
  - `test_parse_args_fullgraph_preserves_cpu_boozer_newton_default` (line 1876, **just added in current diff**)
  - `test_parse_args_preserves_reference_outer_maxls_default` (line 1895)
  - `test_parse_args_marks_explicit_initial_phase_defaults` (line 1917)
  - `test_parse_args_marks_explicit_stage2_bs_path` (line 1938)
- [ ] **3.2** Extract shared helper
  ```python
  def _parse_argv(self, module, *flags):
      with patch.dict(os.environ, {}, clear=True), patch.object(
          sys, "argv", ["single_stage_banana_example.py", *flags],
      ):
          return module.parse_args()
  ```
- [ ] **3.3** Group into 2–3 parametrized tests by assertion shape:
  - One for *outer-backend → defaults* combinations (scipy-jax, scipy-jax-fullgraph, ondevice, cpu): assert `(optimizer_backend, boozer_optimizer_backend, boozer_least_squares_algorithm, …)` tuple
  - One for *invalid backend rejection* (`--optimizer-backend scipy-jax-fullstate` → `SystemExit`)
  - Keep tests with unique semantics (`marks_explicit_*`, `benchmark_mode_uses_target_lane_trial_defaults`) as-is — they verify non-default flags
- [ ] **3.4** Validate

---

## Phase 4 — Parametrize `resolve_target_lane_boozer_init_base_overrides_*` (DRY, ~200 → ~60 lines)

5 tests at lines 1085–1217 differ only in input args / asserted output keys. Same function, same harness.

- [ ] **4.1** Inventory:
  - `test_resolve_target_lane_boozer_init_base_overrides_uses_target_lane_defaults` (1085)
  - `test_resolve_target_lane_boozer_init_base_overrides_floors_bfgs_tol` (1111)
  - `test_resolve_target_lane_boozer_init_base_overrides_is_empty_off_target_lane` (1129)
  - `test_resolve_target_lane_boozer_init_base_overrides_floors_bfgs_maxiter` (1155)
  - `test_resolve_target_lane_boozer_init_base_overrides_skips_full_memory_newton_floor_for_lbfgs` (1173)
  - + 4 sibling `test_resolve_warm_start_boozer_init_overrides_*` tests (1059, 1191, 1219, 1240)
- [ ] **4.2** Convert each to a parametrize row of `(input_kwargs, expected_overrides_dict_subset)` pairs
- [ ] **4.3** Validate

---

## Phase 5 — Parametrize `resolve_scaled_outer_phase_final_dofs_*_is_transfer_safe` (DRY, ~80 → ~30 lines)

A 2×2 grid of `{host, device} × {state-wrapped, raw}` with identical assertions.

- [ ] **5.1** Inventory (all at lines 2074–2158):
  - `test_resolve_scaled_outer_phase_final_dofs_target_lane_host_anchor_device_step_is_transfer_safe`
  - `test_resolve_scaled_outer_phase_final_dofs_target_lane_host_anchor_host_step_is_transfer_safe`
  - `test_resolve_scaled_outer_phase_final_dofs_target_lane_scaled_state_host_anchor_is_transfer_safe`
  - `test_resolve_scaled_outer_phase_final_dofs_target_lane_scaled_state_host_anchor_host_step_is_transfer_safe`
- [ ] **5.2** Single parametrized test with 4 rows over `(anchor_kind, step_kind, use_state_wrapper)`
- [ ] **5.3** Validate

---

## Phase 6 — Merge `initialize_boozer_surface_threads_*` (DRY/YAGNI, ~150 → ~40 lines)

4 tests verify that kwargs pass through to `BoozerSurfaceJAX.options`. This is testing pass-through plumbing, not logic.

- [ ] **6.1** Inventory:
  - `test_initialize_boozer_surface_threads_nondefault_optimizer_backend` (line 903)
  - `test_initialize_boozer_surface_limited_memory_disables_dense_linearization` (line 930)
  - `test_initialize_boozer_surface_threads_nondefault_least_squares_algorithm` (line 958)
  - `test_initialize_boozer_surface_threads_solver_budget_overrides` (line 985)
- [ ] **6.2** Replace with one test that passes *all* kwargs simultaneously and asserts the merged options dict — this verifies pass-through with one call instead of four
- [ ] **6.3** Validate

---

## Phase 7 — Extract shared fakes for `run_single_stage_optimizer_threads_*` (DRY, ~290 → ~100 lines)

5 tests at lines 7934–8222 each rebuild the same `explicit_fun`, `fake_require_target_backend_x64`, `fake_jax_minimize` skeleton with massive `del` blocks.

- [ ] **7.1** Add a helper inside `SingleStageExampleTests` (or to a future shared mixin):
  ```python
  @staticmethod
  def _make_optimizer_threading_fakes(captured):
      def fake_require(backend):
          captured["x64_backend"] = backend
      def fake_minimize(fun, x0, *, method, callback, value_and_grad,
                       progress_callback=None, failure_callback=None,
                       initial_value_and_grad=None, **rest):
          captured.update({
              "method": method,
              "x0": np.asarray(x0),
              "failure_callback": failure_callback,
              "progress_callback": progress_callback,
              "initial_value_and_grad": initial_value_and_grad,
          })
          return types.SimpleNamespace(x=np.asarray(x0), nit=0, message="ok")
      return fake_require, fake_minimize
  ```
- [ ] **7.2** Refactor each test to call the helper, then assert on `captured` keys relevant to that test
  - `test_run_single_stage_optimizer_threads_target_lane_failure_callback` (7934)
  - `test_run_single_stage_optimizer_scipy_jax_omits_private_diagnostics` (7995)
  - `test_run_single_stage_optimizer_fullgraph_uses_full_optimizer_vector` (8065)
  - `test_run_single_stage_optimizer_threads_target_lane_progress_callback` (8123)
  - `test_run_single_stage_optimizer_threads_reference_trace_contract` (8218)
  - `test_run_single_stage_optimizer_threads_target_lane_initial_step_size` (8294)
  - `test_run_single_stage_optimizer_threads_target_lane_initial_value_and_grad` (8354)
- [ ] **7.3** Validate

---

## Phase 8 — Extract `build_single_stage_results_envelope` fixture (DRY, ~280 → ~120 lines)

3 tests in `ResultsEnvelopeTests` (lines 13762, 13881, 13974) each build a `SimpleNamespace` of ~16 args and call `build_single_stage_results_envelope` with **30+ kwargs**, of which only 2-3 differ per test.

- [ ] **8.1** Extract `_default_envelope_kwargs(self, **overrides)` helper that returns the merged kwargs dict
- [ ] **8.2** Extract `_default_args_namespace(self, **overrides)` for the `SimpleNamespace`
- [ ] **8.3** Refactor each test to:
  ```python
  envelope = module.build_single_stage_results_envelope(
      **self._default_envelope_kwargs(
          # only the 2-3 keys that matter for this test
      )
  )
  ```
- [ ] **8.4** Validate

---

## Phase 9 — Split `SingleStageExampleTests` into focused TestCases (SRP, structural)

After Phases 1–8 the megaclass is shorter but still owns 240+ tests. Split by concern:

- [ ] **9.1** Move CLI tests
  - Target class: `ParseArgsTests`
  - Methods: ~25 `test_parse_args_*` (and the post-Phase 3 parametrized ones)
- [ ] **9.2** Move boozer-init tests
  - Target classes: `BoozerSurfaceInitTests` (~15 `test_initialize_boozer_surface_*`), `BoozerSurfaceWarmStartTests` (~7 `test_resolve_warm_start_boozer_init_*` + `test_resolve_target_lane_boozer_init_*`)
  - Move shared helpers (`build_fake_boozer_surface_jax_class`, `patch_initialize_boozer_surface_jax`, `initialize_boozer_surface`, `patch_surface_self_intersection_backend_unavailable`) to a shared base or module-level utility
- [ ] **9.3** Move runtime-spec tests
  - Target class: `RuntimeSpecBiotSavartTests`
  - Methods: `test_runtime_spec_biotsavart_*`, `test_jax_warm_start_surface_dofs_require_seed_spec_artifact`
- [ ] **9.4** Move scaled outer problem tests
  - Target class: `ScaledOuterProblemTests`
  - Methods: ~17 `test_build_scaled_outer_*`, `test_resolve_scaled_outer_*`, `test_build_target_lane_scaled_phase1_diagnosis_*`
- [ ] **9.5** Move target-lane retry tests
  - Target class: `TargetLaneRetryTests`
  - Methods: ~12 `test_run_single_stage_target_lane_optimizer_with_retries_*`
- [ ] **9.6** Move snapshot/restore tests
  - Target class: `SnapshotRestoreTests`
  - Methods: `test_snapshot_*`, `test_snapshot_to_pytree_*`, `test_snapshot_restore_round_trip`, `test_snapshot_records_unavailable_self_intersection_backend`
- [ ] **9.7** Move adapter / objective tests
  - Target class: `SingleStageAdapterTests`
  - Methods: `test_single_stage_adapter_*`, `test_accept_step_*`, `test_objective_evaluation_trace_*`, `test_single_stage_objective_*`
- [ ] **9.8** Move failure-penalty tests
  - Target class: `FailurePenaltyTests`
  - Methods: `test_failure_penalty_*`, `test_compute_single_stage_failure_penalty_*`, the two newly-added (in current diff) reject-class tests
- [ ] **9.9** Move reporting / artifact tests
  - Target class: `ReportingMetricsTests`
  - Methods: anything touching `_make_reporting_runtime_summary`, `_make_reporting_runtime_builder`, target-restart artifact tests
- [ ] **9.10** Move optimizer-threading tests
  - Target class: `OptimizerThreadingTests`
  - Methods: `test_run_single_stage_optimizer_threads_*`, `_omits_private_diagnostics`, `_fullgraph_uses_full_optimizer_vector`, `_target_lane_requires_objective_contract`
- [ ] **9.11** Validate after each move
  - **Important:** move tests in groups of ~25–40 at a time, validate after each group, commit after each green pass. Do not move all 240+ in one shot.

---

## Phase 10 — Replace `captured = {}` with `Mock.call_args` (FP, ~23 sites)

The captured-dict pattern is impure: tests construct closures that mutate a dict. `unittest.mock.Mock` provides the same observation purely through `mock.call_args` / `mock.call_args_list`.

- [ ] **10.1** Pick the cleanest cluster first (post-Phase 7 optimizer-threading tests) and convert one as a template
  - Before:
    ```python
    captured = {}
    def fake_minimize(...): captured["method"] = method; ...
    self.assertEqual(captured["method"], "lbfgs-scipy-jax")
    ```
  - After:
    ```python
    fake_minimize = Mock(return_value=SimpleNamespace(x=np.zeros(2), nit=0, message="ok"))
    ...
    self.assertEqual(fake_minimize.call_args.kwargs["method"], "lbfgs-scipy-jax")
    ```
- [ ] **10.2** Roll out across remaining 22 sites in batches; validate each batch
- [ ] **10.3** Note: keep `captured = {}` where the test asserts on the *order* of multiple calls and `Mock.call_args_list` would be more verbose. SSOT/FP is a guideline, not a sledgehammer.

---

## Phase 11 — Final cleanup and CLAUDE.md update

- [ ] **11.1** Run `ruff check` and `ruff format` over the entire file; resolve any new findings
- [ ] **11.2** Verify final stats:
  - Expect: ~9,500–10,500 lines (down from 14,056)
  - Expect: ~10–12 TestCases instead of 1 megaclass
  - Expect: same test count or slightly lower (lost ~3 source-grep tests + parametrized merges); zero behavioral coverage lost
- [ ] **11.3** Update `CLAUDE.md` "Code Review History" section with a brief note on the refactor (file split, parametrization conventions adopted)
- [ ] **11.4** Run full validation suite per CLAUDE.md before final commit:
  ```bash
  conda run -n jax-0.9.2 python -m pytest tests/test_jax_import_smoke.py tests/field/test_biotsavart_jax.py tests/geo/test_surface_fourier_jax.py tests/geo/test_boozer_residual_jax.py tests/objectives/test_integral_bdotn_jax.py tests/geo/test_boozer_derivatives_jax.py tests/geo/test_boozersurface_jax.py tests/integration/test_jax_native_path.py tests/geo/test_single_stage_example.py -m "not private_optimizer_runtime" -v
  ```
- [ ] **11.5** Final commit with co-author trailer per project conventions

---

## Risk callouts

1. **`importlib`-based module loading.** `load_single_stage_example_module()` (line 71) generates a fresh module name with `uuid.uuid4().hex` per call. Splitting tests across classes preserves this; do **not** consolidate into a class-level `setUpClass` cached module without verifying the example module is re-importable. Some tests *intentionally* monkey-patch module-level functions (e.g. `module.build_runtime_provenance = lambda **_: {"repo_sha": "deadbeef"}` at line 13764), and a shared cached module would leak that patch across tests.

2. **Helper move-out.** The 19 helpers on `SingleStageExampleTests` use `self.` and `@staticmethod`. When moving tests to new classes, decide per helper:
   - Pure `@staticmethod` with no class state → move to module-level functions
   - References to `self.load_module()` → keep on a small shared base TestCase
   - `FakeSurfaceXYZTensorFourier.instances = []` setUp pattern (line 186) → must move with any test that triggers `FakeSurfaceXYZTensorFourier`

3. **`subTest` vs separate methods.** `subTest` shows up nicely in pytest output but loses per-case timing isolation. For tests where one row's failure should not stop others, use `subTest`. For tests where the parametrize dimension is large or per-case timing matters, prefer `pytest.mark.parametrize` (the file already uses pytest as the runner, so this works). Be consistent within a class.

4. **Current diff pre-existing.** The current uncommitted diff (+259/−4) added 5 new methods to `SingleStageExampleTests`. **Decide before starting**: either (a) commit the current diff first and refactor on top, or (b) rebase the new tests into the new structure as part of Phase 9. Option (a) is safer; (b) is cleaner history.

5. **Parametrize regressions.** When merging multiple tests into one parametrized test, a failure in one row will be reported once; if the original tests had distinct names that other tooling (CI dashboards, flaky-test trackers) relied on, those will lose granularity. Confirm with the team before merging if dashboards depend on individual names.

6. **No production-code changes.** This refactor is purely test-file restructuring. If a parametrize merge reveals that two tests were actually testing different code paths (i.e. the "duplicate" was hiding a real difference), keep them separate and document why.

---

## Quick-reference: line-savings estimate (no behavior lost)

| Phase | Cluster | Today | After | Net |
|---|---|---:|---:|---:|
| 1 | source-grep tests | ~30 | 0 | −30 |
| 2 | argparse boolean flags | ~120 | ~30 | −90 |
| 3 | parse_args defaults/lanes | ~500 | ~150 | −350 |
| 4 | resolve_*_overrides | ~200 | ~60 | −140 |
| 5 | scaled_outer transfer-safe | ~80 | ~30 | −50 |
| 6 | initialize_boozer_surface_threads | ~150 | ~40 | −110 |
| 7 | optimizer_threads_* fakes | ~290 | ~100 | −190 |
| 8 | results_envelope fixtures | ~280 | ~120 | −160 |
| 10 | captured-dict → Mock | ~250 | ~150 | −100 |
| **Sum** | | **~1,900** | **~680** | **−1,220** |

Phase 9 (split) doesn't reduce line count materially but reduces *cognitive* size: each new TestCase is reviewable in isolation.

---

## Definition of Done

- [ ] All 11 phases checked off
- [ ] Pytest baseline (Phase 0.1) shows the same passing test set, modulo the 3 deleted source-grep tests and any parametrize-merged duplicates (each merge documented in commit messages)
- [ ] `ruff check` clean on the file
- [ ] No production-code under `src/` or `examples/` modified
- [ ] `CLAUDE.md` updated with new file/class layout
- [ ] Each phase committed separately with a clear message; final history readable as a sequence of mechanical refactors
