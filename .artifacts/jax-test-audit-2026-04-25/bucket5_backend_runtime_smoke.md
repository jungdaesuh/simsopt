# Bucket 5 audit — backend selection, JAX runtime, smoke, benchmark helpers, conftests

Audit branch: `gpu-purity-stage2-20260405` (HEAD `42b68f33d`).
Date: 2026-04-25.
Scope: `tests/test_backend.py`, `tests/test_jax_import_smoke.py`,
`tests/test_host_boundary.py`, `tests/test_biotsavart_donation_probe.py`,
`tests/test_field_cache_hot_path_benchmark.py`, `tests/test_benchmark_helpers.py`,
`tests/test_run_code_benchmark_common.py`, `tests/conftest.py`,
`tests/integration/conftest.py`, `tests/geo/conftest.py`,
`tests/subprocess/import_smoke_cases.py`, `tests/subprocess/jax_runtime_cases.py`,
`tests/subprocess/section6_fixture_probe.py`. Source modules read for
context: `src/simsopt/backend.py`, `src/simsopt/backend/__init__.py`,
`src/simsopt/backend/runtime.py`, `benchmarks/biotsavart_donation_probe.py`,
`benchmarks/field_cache_hot_path_benchmark.py`,
`benchmarks/validation_ladder_common.py`.

## 1. Per-file summary

| File | Lines | Tests | Net classification | Headline strengths | Headline gaps |
| --- | --- | --- | --- | --- | --- |
| `tests/test_backend.py` | 1830 | 83 | WELL-TIGHTENED w/ minor TAUTOLOGICAL clusters | Loads backend module from a fresh `simsopt` namespace (`_fresh_backend`) so cache state is realistic; asserts every synced env var; asserts `_validate_initialized_jax_runtime` warning vs strict-error split; ratchets XLA-flag last-write-wins semantics (lines 1608, 1624, 1682). | Module guard lifecycle is exercised by `test_backend_state_guard_sequence_*` and `test_backend_module_guard_sequence_*` which depend on test ordering — the names encode an order, but pytest does not guarantee it across `-p random` or `-x`. The “sequence_02_restores_native_cpu_defaults” pair only confirms the autouse fixture restored env, not that any consumer observed the change. |
| `tests/test_jax_import_smoke.py` | 1340 | 105 (incl. 13 parametrize ids) | WELL-TIGHTENED for the in-process AST checks; OVERLY THIN for many subprocess wrappers | Subprocess driver `_assert_python_script_passes` always runs a real `subprocess.run` with a clean env (`_BACKEND_SELECTOR_ENV_VARS` stripped) and asserts `rc == 0`. `test_audited_entrypoints_configure_runtime_before_importing_jax` and `test_*_no_private_jax_src_usage` use AST analysis to pin import ordering. | ~70 wrappers only assert the subprocess exit code — they delegate _all_ behaviour assertions to the case function. If a case stops asserting, the wrapper still passes. There is no per-test enforcement that the subprocess actually exercised JAX (e.g. no `jax in subprocess sys.modules` probe). |
| `tests/test_host_boundary.py` | 95 | 6 | WELL-TIGHTENED | `test_strict_scalar_grad_helpers_respect_transfer_guard` exercises `jax.transfer_guard("disallow")` end-to-end and asserts the host-tree result matches the analytic answer. Each helper is paired with explicit dtype assertion (`isinstance(value, np.ndarray)`, `dtype == np.float64`). | Coverage limited to 6 paths; nothing covers `host_int`/`host_float` against negative or NaN values, no test ensures `host_tree` raises (or refuses) for non-numeric leaves under `dtype=...`. |
| `tests/test_biotsavart_donation_probe.py` | 118 | 3 | DONATION/CACHING CONTRACT GAP | The probe payload is exercised end-to-end (`build_biotsavart_donation_probe_payload` for both synthetic and `real-stage2` fixtures); backend config is restored after the probe runs (`test_biotsavart_donation_probe_restores_backend_config`). | The donation contract is **not** validated. No test asserts that `donate_argnums=(0,)` actually deletes the input buffer. Buffer reuse parity is asserted via `max_abs_diff == 0.0` only because both kernels use a `_fresh_points()` host array per call — by construction the donation invariant cannot fail. Recommend a follow-up that asserts `points.is_deleted()` after the donated kernel runs. |
| `tests/test_field_cache_hot_path_benchmark.py` | 80 | 7 | TAUTOLOGICAL / WEAK ASSERTION | Tests build a compile command and assert the include flags exist; they validate `argparse` rejection paths. | Tests do **not** compile or run the benchmark. `test_format_summary_reports_speedups` feeds a hand-crafted payload and asserts that the formatter prints `15.00x`. Nothing exercises actual cache behaviour, hit/miss counts, or that the source `.cpp` peer file even compiles in this repo. |
| `tests/test_benchmark_helpers.py` | 5342 | 180 | Mix; mostly WELL-TIGHTENED for tolerance-table tests, OVERLY-MOCKED for subprocess fakes | `test_repo_pythonpath_env_*` asserts every env var that a child process must observe; `test_apply_compilation_cache_policy_*` asserts both the env state and the returned metadata. | Many tests fake `run_python_script` with `types.SimpleNamespace(returncode=0, stdout="", stderr="")` then only assert that the wrapper passes the right CLI flags — they do not exercise the actual fixture build. |
| `tests/test_run_code_benchmark_common.py` | 40 | 4 | WELL-TIGHTENED | Each test pins a specific resolver decision (default vs explicit, supported vs unsupported runtime). | Coverage limited to 4 outcomes; no test for `resolve_benchmark_backends("hybrid")` or behaviour when both backends are listed. |
| `tests/conftest.py` | 463 | n/a | WELL-TIGHTENED w/ one autouse fixture | `_guard_backend_runtime_state` snapshots all 28 backend env vars + the loaded `jax.config` state and restores both via `try/finally`. `pytest_collection_modifyitems` injects markers consistently. | The autouse fixture *invalidates* the loaded backend cache before yielding — that is by design but may surprise tests that pre-cache state in fixture setup. No regression guards if a future fixture adds a new env var to the runtime contract. |
| `tests/integration/conftest.py` | 70 | n/a | CONFTEST FOOTGUN (silent failure) | None — this conftest is critical for cross-env support. | `_patch_meta_path_finder()` walks `sys.meta_path` and **silently returns False** if no `ScikitBuildRedirectingFinder` is present; the only side effect is that none of the JAX modules are added to the editable finder. No assertion / warning. If the upstream finder is renamed, integration tests would silently re-route to the foreign package without anyone noticing. |
| `tests/geo/conftest.py` | 34 | n/a | OK | Captures `sys.modules["simsopt*"]` snapshot at conftest import time and restores in a session-scoped autouse `yield` fixture. | Snapshot taken at import time captures whatever the parent pytest worker had loaded; if `tests/geo/conftest.py` is imported after some test module has already injected a stub, that stub becomes the "clean" baseline. Bound, but fragile to import-order changes. |
| `tests/subprocess/import_smoke_cases.py` | 2270 | 0 (cases) | WELL-TIGHTENED | Each `case_*` function does its own meta-path blocking via `block_simsoptpp_imports`/`block_jax_imports` and asserts behavioural invariants (e.g. `assert "simsopt.geo.optimizer_jax_private" not in sys.modules`). The dispatcher refuses unknown cases (`exit 2`). | `case_optimizer_jax_public_reference_methods_work_without_private_package` only confirms `result.success` and that `optimizer_jax_private` did not get imported — it does not check the returned numerical value. Acceptable as smoke. |
| `tests/subprocess/jax_runtime_cases.py` | 2269 | 0 (cases) | WELL-TIGHTENED | The compile-counter handler hooks `jax.log_compiles` and asserts exactly one compile per `_run_solver`. Every case `block_until_ready`s its outputs and asserts `np.isfinite`. | The "transfer-guard" suite primarily proves that operations *don't raise* — when a future regression starts triggering implicit transfers, only the ones the case explicitly invokes will be caught. No global probe (e.g. record `jax.transfer_guard(...)` events) is asserted. |
| `tests/subprocess/section6_fixture_probe.py` | 77 | 0 (script) | WELL-TIGHTENED | Strips editable finders, optionally blocks the private optimizer, then prints a JSON payload that the parent test parses. | Coverage depends on the parent test asserting the payload — none is in this audit bucket; `test_benchmark_helpers.py` consumes it. The probe itself is fine. |

## 2. Top issues

| # | file:line | test (or fixture) | Classification | Quote | Recommended tightening |
| --- | --- | --- | --- | --- | --- |
| 1 | tests/test_biotsavart_donation_probe.py:75-95 | `test_biotsavart_donation_probe_matches_baseline`, `test_biotsavart_donation_probe_supports_real_stage2_fixture` | DONATION/CACHING CONTRACT | `assert payload["cases"]["donate_points"]["donate_argnums"] == [0]` and `assert payload["comparison"]["max_abs_diff"] == 0.0` | Donation is never enforced — `_measure_probe_case` always feeds `_fresh_points(host_points)` per call (benchmarks/biotsavart_donation_probe.py:231,237,244), so deletion never affects the next call. Add a probe that retains the donated `points` JAX array and asserts `points.is_deleted()` (or `jax.dlpack.dlpack_capsule(points)` raises) after the donated kernel returns. |
| 2 | tests/test_jax_import_smoke.py:519-526 | `test_lbfgs_ondevice_reuses_compiled_solver_across_identical_calls`, `test_bfgs_ondevice_reuses_compiled_solver_across_identical_calls` | WELL-TIGHTENED (cache-counter side) but invisible to wrapper | `_assert_ondevice_optimizer_reuses_compiled_solver(method)` only asserts subprocess `rc == 0`; the actual `handler.count == 1` enforcement lives in `_run_compile_count_case` | Have the case write a JSON payload (`{"compiles": 1}`), parse it in the wrapper, and assert the integer. Otherwise a future case that silently no-ops still passes. |
| 3 | tests/test_jax_import_smoke.py:247-300 | All `test_import_*` and `test_repo_bootstrap_*` wrappers | WEAK ASSERTION | `assert rc == 0, f"...failed:\n{err}"` (75+ instances) | All assertion logic is inside the case dispatch function. At minimum, also sample stdout for a known sentinel (`print("OK", flush=True)` in the case + `assert "OK" in out`) so the wrapper proves the case actually ran a non-trivial body, not just exited 0 from an unknown branch. |
| 4 | tests/test_field_cache_hot_path_benchmark.py:8-21 | `test_build_compile_command_includes_repo_headers` | TAUTOLOGICAL | `assert command[:5] == ["/usr/bin/c++", "-std=c++17", "-O3", "-DNDEBUG", "-w"]` | Test asserts the function returns the literal list it constructed. No execution. Either parametrize over compiler resolution (`monkeypatch shutil.which`) and assert the resolver chooses the first available, or actually invoke the compiler in a tmp_path and assert the binary exists. Currently it merely pins the format string. |
| 5 | tests/test_field_cache_hot_path_benchmark.py:23-49 | `test_format_summary_reports_speedups` | TAUTOLOGICAL | `assert "legacy/indexed speedup: 15.00x" in summary` | The test feeds `speedups: {"legacy_vs_indexed_compute": 15.0}` and then asserts the formatter prints `15.00x`. This pins the printf format and nothing else. Replace with a test that drives `format_summary` from a real measurement payload (median/mean fields filled by a recorded fixture) so any divergence between bookkeeping and output is caught. |
| 6 | tests/test_field_cache_hot_path_benchmark.py:62-66 | `test_parse_args_allows_zero_warmup` | TAUTOLOGICAL | `assert args.warmup == 0` | The test runs `parse_args(["--warmup", "0"])` and asserts `warmup == 0`. The `nonnegative_int` validator already encodes that constraint; this just round-trips an arg. Replace with an end-to-end test that runs the benchmark with `--warmup 0 --iterations 1 --samples 1` and asserts JSON keys exist. |
| 7 | tests/test_backend.py:1772-1814 | `test_backend_state_guard_sequence_01..04` | CONFTEST FOOTGUN — order dependency | `def test_backend_state_guard_sequence_01_leaves_strict_backend_override(): ... ; def test_backend_state_guard_sequence_02_restores_native_cpu_defaults(): ...` | Test ordering is enforced only by alphabetical name. Under `pytest -p random`, `pytest -x`, or sharded CI, sequence_02 may run before sequence_01 and the assertion `config.mode == "native_cpu"` becomes vacuous (just confirms the autouse fixture restored env). Replace the 4-test sequence with a single test whose body explicitly mutates → asserts → invalidates → re-asserts, or use `pytest-ordering` / `@pytest.mark.dependency`. |
| 8 | tests/test_backend.py:1816-1830 | `test_backend_module_guard_sequence_01..02` | CONFTEST FOOTGUN — module-level state | `_backend_module_guard_reloaded.update(_snapshot_backend_modules())` (module global mutated mid-test) | Same ordering issue + mutating module-level dict from inside a test. The autouse `_restore_backend_modules` fixture restores `sys.modules` between tests, but the module-level `_backend_module_guard_reloaded` dict survives across tests with no enforcement that it was populated first. Inline the contract or use a session-scoped fixture. |
| 9 | tests/integration/conftest.py:54-66 | `_patch_meta_path_finder()` autouse-on-import | CONFTEST FOOTGUN — silent miss | `for finder in sys.meta_path: if hasattr(finder, "known_source_files"): finder.known_source_files.update(new_modules); ... return True; return False` | Returns `False` silently if the editable finder is not present (e.g. when run in a pure-pip install). Existing tests appear to assume the patch took effect, but no assertion verifies it. Either assert `_patch_meta_path_finder()` returned `True` (raise otherwise — call site at line 70 ignores the return), or downgrade to a logging warning so misconfigurations surface. |
| 10 | tests/conftest.py:221-233 | `_guard_backend_runtime_state` (autouse) | WELL-TIGHTENED but undocumented invariant | `_invalidate_loaded_backend_state()` is called both before yield **and** in `finally` | The autouse fixture invalidates the cache twice. That's intentional, but a test that depends on a freshly cached value populated by a fixture earlier in the chain will see the cache cleared. Document why; add a regression test that verifies the cache is empty between tests. |
| 11 | tests/conftest.py:434-463 | `pytest_collection_modifyitems` | WELL-TIGHTENED | `if relpath_str in {"integration/test_single_stage_jax.py", ...}: item.add_marker(pytest.mark.single_stage); item.add_marker(pytest.mark.slow)` | Hard-coded path strings. If a test file is renamed and not updated here, it silently loses the `slow` / `single_stage` marker. Add a guard: warn if a known file no longer exists. |
| 12 | tests/test_jax_import_smoke.py:498-507 | `test_transfer_guard_disallow_enforces_single_stage_target_runtime_boundaries` | WELL-TIGHTENED | invokes `_JAX_SUBPROCESS_CASES_PATH` with `single-stage-target-runtime-transfer-guard`, `timeout=300` | This is a real subprocess executing a complete single-stage runtime bundle. Solid. |
| 13 | tests/test_jax_import_smoke.py:1147-1175 | `test_optimizer_jax_*_has_no_private_jax_src_usage`, `test_backend_runtime_module_has_no_private_jax_src_usage` | WELL-TIGHTENED | `_assert_no_private_jax_src_usage(_OPTIMIZER_JAX_PATH, label="...")` | Static AST analysis is the right tool here. The custom AST walker (`_find_private_jax_src_usages`) is itself unit-tested at line 234 (`test_find_private_jax_src_usages_detects_alias_attribute_access`). Solid contract. |
| 14 | tests/test_jax_import_smoke.py:406-420 | `test_audited_entrypoints_configure_runtime_before_importing_jax` | WELL-TIGHTENED | `assert min(configure_lines) < first_jax_import` | Ordering guard via AST. Strong: any future entrypoint that imports `jax` before configuration would fail. The audit list (`_ENTRYPOINT_RUNTIME_AUDIT_PATHS`) needs maintenance discipline though. Add a guard that the audit list is non-empty (defensive). |
| 15 | tests/test_jax_import_smoke.py:1243-1284 | `test_biotsavart_jax_backend_does_not_import_coil_unwrap_helper`, `test_surfaceobjectives_jax_has_no_tensor_surface_imports` | WELL-TIGHTENED | static-import AST walks for forbidden symbols | Strong; the right level for this contract. |
| 16 | tests/test_benchmark_helpers.py:1352-1399 | `test_run_python_script_streams_*` | WELL-TIGHTENED | Constructs a real child script + asserts both `capsys` and the returned `result.stdout` contents | Solid coverage of the streaming helper. |
| 17 | tests/test_benchmark_helpers.py:1182-1198 | `test_describe_compile_behavior_tracks_cache_state` | WELL-TIGHTENED | Mutates env via `monkeypatch`, calls `apply_compilation_cache_policy()` then asserts the descriptor string | Solid: drives both branches of the descriptor. |
| 18 | tests/test_benchmark_helpers.py:921-1093 | `test_apply_compilation_cache_policy_*`, `test_apply_benchmark_compilation_cache_policy_*` | WELL-TIGHTENED | Asserts both the returned metadata dict and the resulting env state | Strong: catches drift between metadata and env. |
| 19 | tests/test_benchmark_helpers.py:1110-1179 | `test_parity_ladder_tolerances_*` | WELL-TIGHTENED | Pins exact lane → tolerance values for 8 lanes | Acts as the SSOT regression for `PARITY_LADDER_TOLERANCES`. |
| 20 | tests/test_benchmark_helpers.py:166-181 | `test_resolve_configs_*` | WEAK ASSERTION — but acceptable | `assert resolve_configs(None) == DEFAULT_CONFIGS` | Round-trips the function with the same constant it returns. Acceptable as a "did the helper short-circuit on None" check. |
| 21 | tests/test_run_code_benchmark_common.py:7-9 | `_force_x64` autouse fixture | OVERLY-MOCKED | `monkeypatch.setattr(benchmark_common, "_x64_enabled", lambda: True)` | The autouse fixture stubs `_x64_enabled()` for every test in the file. That means none of these 4 tests prove the resolver behaves correctly when x64 is *not* enabled. Add a counter-test (`def test_resolver_rejects_float32_runtime`) that drops the autouse override. |
| 22 | tests/test_jax_import_smoke.py:111-120 | `_build_clean_subprocess_env` | WELL-TIGHTENED | `for name in _BACKEND_SELECTOR_ENV_VARS: env.pop(name, None); env["PYTHONPATH"] = _SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")` | Strong: every subprocess test is run with a sanitized env. List in `_BACKEND_SELECTOR_ENV_VARS` mirrors but is **not synced** with the conftest's `_BACKEND_RUNTIME_ENV_VARS` (28 vs 18 entries). Consolidate into one SSOT to prevent drift; an env var added to one and not the other could leak into a child process and silently change behaviour. |
| 23 | tests/conftest.py:29-58 | `_BACKEND_RUNTIME_ENV_VARS` constant | DRY VIOLATION (potential CONFTEST FOOTGUN) | 28 env vars enumerated; the same set is partially repeated in `tests/test_jax_import_smoke.py:_BACKEND_SELECTOR_ENV_VARS` (18 vars), `benchmarks/validation_ladder_common.py:repo_pythonpath_env`, and the runtime module itself | Single source of truth: derive both lists from `simsopt.backend.runtime` (e.g. expose a frozen tuple `BACKEND_ENV_VARS`). Today the conftest enumerates `SIMSOPT_JAX_*` env vars by hand; if a new var is added to runtime.py without updating both, tests will leak state. |
| 24 | tests/test_backend.py:1097-1132 | `test_maybe_initialize_distributed_jax_updates_sharding_device_counts` | WELL-TIGHTENED | Captures `calls` from a fake `initialize`, asserts the kwargs payload, then re-queries the tuning and asserts `device_count == 8` | Strong: end-to-end behavior assertion through the runtime cache. |
| 25 | tests/test_backend.py:1135-1196 | `test_maybe_initialize_distributed_jax_invalidates_preinit_chunk_caches` | WELL-TIGHTENED | Asserts the chunk tuning is recomputed using the new GPU index after distributed init (`pre_chunk.gpu_total_memory_mb == 24576`, `post_chunk.gpu_total_memory_mb == 8192`) | Excellent regression — proves cache invalidation actually moves the chunk policy. |
| 26 | tests/subprocess/import_smoke_cases.py:1402-1769 | `case_jax_core_specs_are_pytrees` | WELL-TIGHTENED but enormous | One subprocess case asserts pytree round-trip + JIT compatibility for ~20 spec types | Solid; this is the right level. Could be split per-spec for triage when a single one regresses. |
| 27 | tests/subprocess/jax_runtime_cases.py:179-205 | `_CompileCounter` + `_assert_run_solver_compiles_once` | WELL-TIGHTENED | `assert handler.count == 1, handler.count` | Right approach — pins compilation count, not wall time. Strong. |
| 28 | tests/subprocess/jax_runtime_cases.py:36 | module-level `_prefer_local_simsopt_source_tree()` (executes on import) | CONFTEST FOOTGUN | `_prefer_local_simsopt_source_tree()` runs at import, mutates `sys.path`, then unconditionally imports `jax`, `jnp`, `numpy`, and a long list of `simsopt.*` modules at module top | If any one of those imports raises, every subprocess test that spawns this script fails with a single root cause that masks the actual case. Wrap in try/except + structured failure print, or move imports inside `main()`. Currently any breakage in `simsopt.geo.optimizer_jax` (a parent of many cases) collapses ~50 subprocess tests with the same opaque error. |
| 29 | tests/test_benchmark_helpers.py:1616-2351 | The dozen `fake_run_python_script` monkeypatches | OVERLY-MOCKED | `def fake_run_python_script(_script_path, command, **kwargs): captured.append(command); return types.SimpleNamespace(returncode=0, stdout="", stderr="")` | These tests assert that the production code passes the right CLI flags but never run the underlying probe. That is fine for argparse-routing tests, but they shadow real subprocess coverage. Pair each one with at least one integration test that runs the full child (gated on the env where it's possible). |
| 30 | tests/test_jax_import_smoke.py:847-869 | `test_single_stage_surface_reprojection_probe_emits_structured_cpu_result` | WELL-TIGHTENED | Subprocess + JSON payload + per-stage assertion (`assert [stage["name"] for stage in payload["stages"]] == [...]`) | This is the gold standard. Replicate the JSON-payload pattern across the other compile-count tests for issue #2. |

## 3. Missing coverage

- **Donation actually deleting buffers.** No test asserts `points.is_deleted()` after `_donated_points_kernel` runs (or `jax.dlpack.dlpack_capsule(points)` raises). The `donate_argnums=(0,)` annotation could silently regress to `()` and every existing test would still pass.
- **Compile-count assertion at the wrapper level.** `_assert_ondevice_optimizer_reuses_compiled_solver` only asserts the subprocess exit code; the `handler.count == 1` invariant is invisible at the pytest layer. Surface it via a parsed JSON payload.
- **`SIMSOPT_BACKEND_MODE` round-trip with explicit JAX runtime.** No test asserts that setting `SIMSOPT_BACKEND_MODE=jax_gpu_parity` and then calling `jax.jit(...)(jnp.ones(1))` actually executes on CUDA (only that the env var resolver returns the right config). On CI hosts with CUDA available this should be a real GPU smoke; on CPU-only hosts it should be `pytest.skip`-gated.
- **`apply_jax_runtime_config()` runs successfully against a real JAX install.** Every test in `test_backend.py` uses a `types.SimpleNamespace(config=..., default_backend=...)` stub. There is no test that runs `apply_jax_runtime_config()` against the actual `jax` module and verifies `jax.config.jax_enable_x64 is True`. The conftest's `_guard_backend_runtime_state` snapshots and restores `jax.config`, so this is safe to add.
- **`tests/integration/conftest.py` patch verification.** Add `assert _patch_meta_path_finder() is True` at module load time, with a `pytest.skip` fallback when no editable finder is present.
- **No subprocess test verifies `simsoptpp` is NOT in `sys.modules` after a pure-JAX import.** The case bodies block `simsoptpp` via `block_simsoptpp_imports`, so a lazy import would raise — but no test prints `"simsoptpp" in sys.modules` to catch the *opposite* regression (a stub that silently shadows). At minimum, `case_import_pure_jax_modules` should add `assert "simsoptpp" not in sys.modules`.
- **No test for `SIMSOPT_JAX_TRANSFER_GUARD=disallow` failure mode at the boundary level.** Cases assert that operations don't raise; nothing asserts that an obvious offending operation (e.g. `jnp.asarray(np.ones(1))` after `device_put`) does raise.
- **No test for `set_backend(..., configure_runtime=True)` (the default).** Every call site in tests passes `configure_runtime=False` to avoid touching the real JAX runtime. The `True` branch (which actually invokes `apply_jax_runtime_config`) is exercised only indirectly through subprocess cases.
- **`cleanup_distributed` (no such API).** `maybe_initialize_distributed_jax` has no inverse — once initialized in-process, subsequent tests carry the state. The autouse fixture in `tests/conftest.py` invalidates the *config* cache but cannot un-initialize JAX distributed. The four `test_distributed_*` tests that go through real `jax.distributed.initialize` rely on the fact that the patched `jax` module is restored by `monkeypatch.setitem`, but a future test that runs against the real `jax.distributed.initialize` would leak.

## 4. Tightening playbook (P0 — top 8)

1. **Donation enforcement.** In `tests/test_biotsavart_donation_probe.py`, add a third test that:
   - Builds points as `points = jax.device_put(np.zeros((16, 3)))`.
   - Runs `_donated_points_kernel()(points, gammas, gammadashs, currents)`.
   - Asserts `with pytest.raises((RuntimeError, ValueError)): jnp.asarray(points)` (or `points.is_deleted()` if available on the JAX 0.9.2 array surface).
   This makes the donation contract a real invariant rather than a label.
2. **Compile-count visibility.** Modify `_run_compile_count_case` and friends in `tests/subprocess/jax_runtime_cases.py` to print a JSON payload (`{"compile_count": handler.count}`). Update `_assert_ondevice_optimizer_reuses_compiled_solver` to parse that payload and assert `payload["compile_count"] == 1`.
3. **SSOT env-var list.** Move `_BACKEND_RUNTIME_ENV_VARS` into `simsopt.backend.runtime` as a frozen tuple, then import it in `tests/conftest.py`, `tests/test_jax_import_smoke.py`, and `benchmarks/validation_ladder_common.py`. Add a unit test that fails when the list and the runtime env-handling code drift apart.
4. **Sequence-test refactor.** Convert the 4-test `test_backend_state_guard_sequence_*` and 2-test `test_backend_module_guard_sequence_*` clusters into single tests that drive the full mutation/restore lifecycle inside one body. Eliminates ordering dependence and the module-level `_backend_module_guard_reloaded` dict.
5. **Subprocess sentinel.** Each `case_*` function should print a `OK:<case_name>` sentinel as its last line; each `test_*` wrapper should `assert "OK:<case_name>" in stdout`. Cheap, prevents silent no-ops.
6. **Integration conftest assertion.** Replace silent `return False` in `tests/integration/conftest.py:_patch_meta_path_finder` with `pytest.skip("scikit-build editable finder not installed; integration tests need it")` at module load if no editable finder exists.
7. **Real-JAX `apply_jax_runtime_config` test.** Add one test in `test_backend.py` that does *not* stub `jax`: invoke `set_backend("jax_cpu_parity", configure_runtime=True)` and assert `jax.config.jax_enable_x64 is True`. The conftest already snapshots/restores `jax.config`, so this is safe.
8. **Field-cache benchmark sanity test.** Replace `test_build_compile_command_includes_repo_headers` and `test_format_summary_reports_speedups` with one test that compiles + runs the benchmark with `--iterations 1 --samples 1 --warmup 0` and parses the JSON output (skipping if no compiler is on the system). The current tests are tautologies that lock the printf format string.

## 5. Conftest leak audit

### `tests/conftest.py`
| Fixture / mutation | Scope | Cleanup | Verdict |
| --- | --- | --- | --- |
| `_guard_backend_runtime_state` (`@pytest.fixture(autouse=True)`, line 221) | function-scoped autouse | Snapshots `os.environ` for 28 backend env vars and `jax.config` (5 fields) before yield; restores both in `finally`. Also calls `_invalidate_loaded_backend_state()` before AND after. | **BOUNDED.** Strong contract. Caveat: invalidating before yield can surprise fixtures that pre-populate the cache. |
| `parity_lane` (`@pytest.fixture(params=("cpu","gpu"))`, line 379) | function-scoped | Pure parametrization, no side effects. | **BOUNDED.** |
| `pytest_collection_modifyitems` (line 434) | session-level hook | Adds markers. No env mutation. | **BOUNDED.** |
| Module-level `jax.config.update("jax_enable_x64", True)` (line 27) | conftest import | Never restored — but `_guard_backend_runtime_state` snapshots `jax_enable_x64` so per-test restoration recovers test-local mutations. The conftest-import-time mutation persists for the whole session. | **BOUNDED** for in-test isolation; **UNDOCUMENTED** as a session-wide x64 lock. Document. |

### `tests/integration/conftest.py`
| Fixture / mutation | Scope | Cleanup | Verdict |
| --- | --- | --- | --- |
| `_patch_meta_path_finder()` (line 16, called at module load on line 70) | session at conftest import | None — patch persists. | **LEAKY but bounded by intent.** The patch installs entries into the editable finder's `known_source_files` dict; never removed. Acceptable because the conftest is dedicated to integration tests, but if a non-integration test imports this module its `sys.path` semantics change. |

### `tests/geo/conftest.py`
| Fixture / mutation | Scope | Cleanup | Verdict |
| --- | --- | --- | --- |
| `_clean_simsopt_state` (module-level snapshot, line 20) | conftest import | Snapshot is captured at conftest import; restoration happens in the autouse session fixture at line 26. | **BOUNDED** but **fragile**: snapshot is taken at conftest *import* time. If `tests/geo/conftest.py` is imported after another test module has already injected stubs, those stubs become the "clean" baseline. |
| `_restore_simsopt_modules` (`@pytest.fixture(autouse=True, scope="session")`, line 25) | session autouse | `yield` then deletes any `simsopt*` module added during the session and restores the originals. | **BOUNDED.** Correct shape. |

### `tests/test_backend.py`
| Fixture / mutation | Scope | Cleanup | Verdict |
| --- | --- | --- | --- |
| `_restore_backend_modules` (`@pytest.fixture(autouse=True)`, line 47) | function autouse | Snapshots `sys.modules["simsopt"]`, `["simsopt.backend"]`, `["simsopt.backend.runtime"]`; restores in `finally`. | **BOUNDED.** |
| Module-level `_backend_module_guard_reloaded` dict (line 25) | module | Only mutated by `test_backend_module_guard_sequence_01_reloads_backend_modules`; consumed by `_02`. Never reset. | **LEAKY across test runs in same pytest session** if order is randomized. Issue #8 in the table above. |
| Many `monkeypatch` calls (`_clear_backend_env`, `_install_fake_jax`, `_set_distributed_init_env`) | per-test | All use `monkeypatch.setenv` / `monkeypatch.setitem` / `monkeypatch.delenv`. Cleanup automatic. | **BOUNDED.** |
| `monkeypatch.setattr("subprocess.run", fake_run)` in `_install_fake_nvidia_smi` (line 609) | per-test | Cleaned by monkeypatch. | **BOUNDED.** |

### `tests/test_jax_import_smoke.py`
| Fixture / mutation | Scope | Cleanup | Verdict |
| --- | --- | --- | --- |
| `_build_clean_subprocess_env` (line 111) | per-call | Pure function, returns a copy of `os.environ` with backend vars stripped. Does **not** mutate parent env. | **BOUNDED.** |
| All AST tests | n/a | Pure file reads. | **BOUNDED.** |

### `tests/test_benchmark_helpers.py`
| Fixture / mutation | Scope | Cleanup | Verdict |
| --- | --- | --- | --- |
| Many `monkeypatch.setenv`, `monkeypatch.delenv` for compilation-cache and runtime envs | per-test | Cleaned by monkeypatch. | **BOUNDED.** |
| `monkeypatch.setattr("benchmarks.validation_ladder_common.REPO_ROOT", tmp_path)` | per-test | Cleaned by monkeypatch. | **BOUNDED.** |
| `apply_compilation_cache_policy()` mutates `os.environ` and JAX config directly (called inside tests, not under a fixture) | persists until next `monkeypatch.delenv` | The autouse `_guard_backend_runtime_state` in `tests/conftest.py` cleans backend env; but `apply_compilation_cache_policy` also sets `_JAX_COMPILATION_CACHE_ENV_VAR` which **is in** `_BACKEND_RUNTIME_ENV_VARS`. | **BOUNDED via conftest.** Confirmed cross-checked. |

### `tests/test_run_code_benchmark_common.py`
| Fixture / mutation | Scope | Cleanup | Verdict |
| --- | --- | --- | --- |
| `_force_x64` (`@pytest.fixture(autouse=True)`, line 7) | function autouse | `monkeypatch.setattr` → cleaned by monkeypatch. | **BOUNDED but blinkered**: every test in the file overrides x64 detection. No test exercises the unstubbed path. See issue #21. |

### `tests/test_biotsavart_donation_probe.py`
| Fixture / mutation | Scope | Cleanup | Verdict |
| --- | --- | --- | --- |
| `set_backend("jax_gpu_fast", ..., configure_runtime=False)` (line 102) | per-test | Manually wrapped in `try/finally` with `_restore_backend_config(original)`. | **BOUNDED.** |

### `tests/subprocess/jax_runtime_cases.py`
| Mutation | Scope | Cleanup | Verdict |
| --- | --- | --- | --- |
| Module-level `_prefer_local_simsopt_source_tree()` (line 36) and unconditional top-level imports | subprocess process lifetime | Subprocess exit cleans up; no leakage to parent test. | **BOUNDED** but failure-prone (issue #28). Move imports inside `main()` to avoid mass collapse on a single import error. |
| `_configure_strict_cpu_parity_backend()` calls `simsopt_config.set_backend(...)` | subprocess process lifetime | Subprocess exit cleans up. | **BOUNDED.** |

## 6. Subprocess test audit

| Call site | Command | Assertion | Verdict |
| --- | --- | --- | --- |
| `tests/test_jax_import_smoke.py:131` (`_run_python_script` → `subprocess.run`) | `python <case_script> <case_name>` with sanitized env | Returns `(returncode, stderr)`. | Drives ~75 wrappers; assertion is `rc == 0`. **MEANINGFUL** because each case body asserts internally, but the wrapper alone cannot prove the case ran; needs sentinel (issue #5). |
| `tests/test_jax_import_smoke.py:1294, 1308, 1322, 1336` | `_run_python_script(_IMPORT_SMOKE_CASES_PATH, args=("case_field_package_import_is_lazy_with_simsoptpp",))` etc. | `assert rc == 0` | Same shape; case body asserts `simsopt.field.coil` not in `sys.modules`. **MEANINGFUL.** |
| `tests/test_jax_import_smoke.py:847-857` (`test_single_stage_surface_reprojection_probe_emits_structured_cpu_result`) | `_SINGLE_STAGE_SURFACE_REPROJECTION_PROBE_PATH` | Parses JSON output and asserts `payload["passed"] is True`, exact stage list. | **STRONG.** Gold standard. |
| `tests/test_benchmark_helpers.py:578-590, 606-621` | `run_python_script(single_stage_init_parity_module.__file__, [...])` (gated on real CUDA) | Parses payload and asserts `payload["passed"] is True`. | **STRONG.** |
| `tests/test_benchmark_helpers.py:652-672` (`_assert_benchmark_module_import_bootstraps_local_simsopt`) | `python -c "...import {module_name}; print(simsopt.__file__)"` | `assert completed.returncode == 0`; `assert str(repo_root / "src" / "simsopt" / "__init__.py") in completed.stdout.strip()` | **STRONG.** Asserts module came from this checkout, not foreign install. |
| `tests/test_benchmark_helpers.py:1352-1372` (`test_run_python_script_streams_and_captures_output`) | constructed child script + `run_python_script(..., stream_output=True)` | `assert "stdout-line" in captured.out` and `in result.stdout` | **STRONG.** End-to-end stream parity. |
| `tests/test_benchmark_helpers.py:1375-1399` (`test_run_python_script_stream_output_preserves_failure_details`) | constructed child that exits 3 | `pytest.raises(RuntimeError, match="exit code 3")` + checks captured stdout/stderr | **STRONG.** |
| `tests/test_benchmark_helpers.py:~1616, 1700, 1778, ...` (~14 sites) | `monkeypatch.setattr("...run_python_script", fake_run_python_script)` — never invokes a real subprocess | Asserts the wrapper would have called `run_python_script(...)` with the right CLI flags | **OVERLY-MOCKED.** Exercises argparse routing only; no subprocess actually runs. Needs at least one paired integration counterpart. |
| `benchmarks/field_cache_hot_path_benchmark.py:99` (`run_command`) | `cxx ...` compile + run benchmark | `tests/test_field_cache_hot_path_benchmark.py` does **not** drive this. Tests only build the command list. | **NOT TESTED.** See issue #4-#6. |
| `benchmarks/biotsavart_donation_probe.py` (subprocess invocation when run as `__main__`) | Standalone runner | `tests/test_biotsavart_donation_probe.py` calls the **library function** `build_biotsavart_donation_probe_payload` directly — no subprocess. | **IN-PROCESS.** Good for unit testing; donation contract gap remains (issue #1). |

## 7. Net assessment

The backend/runtime/conftest layer is **strong overall**: real backend modules
are loaded with realistic cache state, env vars are snapshot/restored, the
`apply_jax_runtime_config` rejection logic is exercised against every
combination of strict/non-strict and warning/error policy. Subprocess tests are
plentiful and many are gold standard (`test_single_stage_surface_reprojection_*`,
`test_run_python_script_*`).

The two structural weaknesses are:

1. **Donation invariant is unverified** (`donate_argnums=(0,)` could silently
   become `()` and every test would still pass). High-impact, easy fix.
2. **Subprocess wrapper assertions are weak** (assert `rc == 0` only). When a
   case body silently no-ops, the wrapper still passes. Add JSON payloads and
   sentinels.

A handful of TAUTOLOGICAL tests in `test_field_cache_hot_path_benchmark.py`
should be replaced with end-to-end compile/run smoke. The
`test_backend_*_sequence_*` ordering anti-pattern should be folded into
single-test mutation/restore cycles.

DONE — report at /Users/suhjungdae/code/columbia/simsopt-jax/.artifacts/jax-test-audit-2026-04-25/bucket5_backend_runtime_smoke.md
