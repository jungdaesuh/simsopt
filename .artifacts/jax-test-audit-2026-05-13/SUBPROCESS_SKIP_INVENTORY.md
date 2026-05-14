# Subprocess Skip Inventory — 2026-05-13

Generated for audit finding #11 in `TEST_QUALITY_TODOS.md`. Covers
`tests/subprocess/import_smoke_cases.py` and
`tests/subprocess/jax_runtime_cases.py`.

## Format

For each `_skip_case(...)` call site:

| File:Line | Case function | Skip reason | Precondition | Category |
| --- | --- | --- | --- | --- |

Category legend:
- **A — JAX unsupported runtime**: e.g., `private_optimizer_runtime_is_supported(jax.__version__)` returns `False`. Tracked by `PRIVATE_OPTIMIZER_JAX_VERSION` allow-list in `simsopt.geo.optimizer_jax`.
- **B — GPU absent on CPU-only host**: e.g., no JAX device with `platform == "gpu"`, or `_configure_strict_gpu_fast_backend()` returns `None`, or `next(... if device.platform == "gpu")` returns `None`.
- **C — simsoptpp absent**: requires the C++ extension but it is not built/importable.
- **D — Test fixture artifact missing**: a known input JSON/seed file is absent.
- **E — Other / not principled**: anything else; a hidden regression mask.

The `_skip_case(reason)` helper is defined in both files (raises a `SkippedCase` sentinel that `_assert_subprocess_json_sentinel` translates into `pytest.skip(reason)`). Definition lines (`import_smoke_cases.py:32` and `jax_runtime_cases.py:24`) are excluded from the inventory.

## Inventory

| File:Line | Case function | Skip reason | Precondition | Category |
| --- | --- | --- | --- | --- |
| tests/subprocess/import_smoke_cases.py:785 | `case_transfer_guard_disallow_allows_lbfgs_ondevice_quadratic_smokes` | `f"private optimizer runtime unsupported for JAX {jax.__version__}"` | `not private_optimizer_runtime_is_supported(jax.__version__)` | A |
| tests/subprocess/import_smoke_cases.py:846 | `case_transfer_guard_disallow_allows_target_minimize_structured_pytree_entry` | `f"private optimizer runtime unsupported for JAX {jax.__version__}"` | `not private_optimizer_runtime_is_supported(jax.__version__)` | A |
| tests/subprocess/import_smoke_cases.py:933 | `case_transfer_guard_disallow_allows_adam_ondevice_quadratic_smokes` | `f"private optimizer runtime unsupported for JAX {jax.__version__}"` | `not private_optimizer_runtime_is_supported(jax.__version__)` | A |
| tests/subprocess/import_smoke_cases.py:975 | `case_transfer_guard_disallow_allows_lm_ondevice_quadratic_smokes` | `f"private optimizer runtime unsupported for JAX {jax.__version__}"` | `not private_optimizer_runtime_is_supported(jax.__version__)` | A |
| tests/subprocess/import_smoke_cases.py:1052 | `case_transfer_guard_disallow_allows_ondevice_loops_with_host_closure_constants` | `f"private optimizer runtime unsupported for JAX {jax.__version__}"` | `not private_optimizer_runtime_is_supported(jax.__version__)` | A |
| tests/subprocess/import_smoke_cases.py:1089 | `case_transfer_guard_disallow_allows_gpu_ondevice_loops_with_host_constants` | `"GPU device is required"` | `next((device for device in jax.devices() if device.platform == "gpu"), None) is None` | B |
| tests/subprocess/import_smoke_cases.py:1091 | `case_transfer_guard_disallow_allows_gpu_ondevice_loops_with_host_constants` | `f"private optimizer runtime unsupported for JAX {jax.__version__}"` | `not private_optimizer_runtime_is_supported(jax.__version__)` | A |
| tests/subprocess/jax_runtime_cases.py:246 | `_run_compile_count_case` | `"strict CPU parity backend unavailable: private optimizer runtime not supported"` | `not _configure_strict_cpu_parity_backend()` (which returns `private_optimizer_runtime_is_supported(jax.__version__)`) | A |
| tests/subprocess/jax_runtime_cases.py:302 | `_run_target_compile_count_case` | `"strict CPU parity backend unavailable: private optimizer runtime not supported"` | `not _configure_strict_cpu_parity_backend()` | A |
| tests/subprocess/jax_runtime_cases.py:361 | `_run_stage2_target_compile_count_case` | `"strict CPU parity backend unavailable: private optimizer runtime not supported"` | `not _configure_strict_cpu_parity_backend()` | A |
| tests/subprocess/jax_runtime_cases.py:579 | `_run_single_stage_target_runtime_bundle_transfer_guard_case` | `"strict CPU parity backend unavailable: private optimizer runtime not supported"` | `not _configure_strict_cpu_parity_backend()` | A |
| tests/subprocess/jax_runtime_cases.py:768 | `_run_grouped_biot_savart_gpu_spec_eval_case` | `"strict GPU fast backend unavailable: no GPU device detected"` | `_configure_strict_gpu_fast_backend() is None` (no JAX device with `platform == "gpu"`) | B |
| tests/subprocess/jax_runtime_cases.py:1532 | `_run_gamma_2d_eager_host_constants_case` | `"strict GPU fast backend unavailable: no GPU device detected"` | `_configure_strict_gpu_fast_backend() is None` | B |
| tests/subprocess/jax_runtime_cases.py:1546 | `_run_closed_curve_self_intersection_summary_case` | `"strict GPU fast backend unavailable: no GPU device detected"` | `_configure_strict_gpu_fast_backend() is None` | B |
| tests/subprocess/jax_runtime_cases.py:1574 | `_run_single_stage_surface_self_intersection_case` | `"strict GPU fast backend unavailable: no GPU device detected"` | `_configure_strict_gpu_fast_backend() is None` | B |
| tests/subprocess/jax_runtime_cases.py:1666 | `_run_surface_xyztensorfourier_gamma_from_dofs_case` | `"strict GPU fast backend unavailable: no GPU device detected"` | `_configure_strict_gpu_fast_backend() is None` | B |
| tests/subprocess/jax_runtime_cases.py:2115 | `_run_stage2_target_objective_host_closure_constants_case` | `"strict GPU fast backend unavailable: no GPU device detected"` | `_configure_strict_gpu_fast_backend() is None` | B |
| tests/subprocess/jax_runtime_cases.py:2127 | `_run_stage2_target_objective_ondevice_entry_case` | `"strict GPU fast backend unavailable: no GPU device detected"` | `_configure_strict_gpu_fast_backend() is None` | B |
| tests/subprocess/jax_runtime_cases.py:2145 | `_run_grouped_biot_savart_gpu_current_arrays_case` | `"strict GPU fast backend unavailable: no GPU device detected"` | `_configure_strict_gpu_fast_backend() is None` | B |
| tests/subprocess/jax_runtime_cases.py:2169 | `_run_grouped_biot_savart_host_scalar_currents_case` | `"strict GPU fast backend unavailable: no GPU device detected"` | `_configure_strict_gpu_fast_backend() is None` | B |
| tests/subprocess/jax_runtime_cases.py:2194 | `_run_grouped_biot_savart_host_spec_vjp_case` | `"strict GPU fast backend unavailable: no GPU device detected"` | `_configure_strict_gpu_fast_backend() is None` | B |
| tests/subprocess/jax_runtime_cases.py:2219 | `_run_mutable_objective_state_case` | `"strict CPU parity backend unavailable: private optimizer runtime not supported"` | `not _configure_strict_cpu_parity_backend()` | A |
| tests/subprocess/jax_runtime_cases.py:2245 | `_run_structured_mutable_objective_state_case` | `"strict CPU parity backend unavailable: private optimizer runtime not supported"` | `not _configure_strict_cpu_parity_backend()` | A |

## Summary by category

- **A — JAX unsupported runtime**: 13 cases
- **B — GPU absent on CPU-only host**: 10 cases
- **C — simsoptpp absent**: 0 cases
- **D — Test fixture artifact missing**: 0 cases
- **E — Other / not principled**: 0 cases

Total `_skip_case(...)` call sites: 23 (excluding the two helper definitions at `import_smoke_cases.py:32` and `jax_runtime_cases.py:24`).

## Action items

No Category E cases were found. Every `_skip_case(...)` call site falls into category A (JAX private-optimizer runtime allow-list) or B (no GPU device on a CPU-only host). Both are legitimate environment-driven skips:

- Category A — gated on `simsopt.geo.optimizer_jax.private_optimizer_runtime_is_supported(jax.__version__)`. The allow-list (`PRIVATE_OPTIMIZER_JAX_VERSION`) is the SSOT; cases sit behind it because the on-device line-search internals only run on supported JAX versions.
- Category B — gated on JAX device discovery (`platform == "gpu"`). The strict-GPU-fast backend cannot be configured without a CUDA-capable device, so the body would crash with an unrelated `RuntimeError` if executed.

Both reasons describe environments where the case body cannot run, not engineering or test-design defects, so no conversion to hard failure is required by this round of the audit. No inline fix was performed.

## How to maintain

- Re-run the existing AST audit `tests/test_pytest_skip_xfail_audit.py::test_subprocess_case_audit_covers_all_repo_subprocess_case_files` after adding new subprocess cases. That audit already rejects silent `return` in `_run_*`/`case_*` bodies and forces a `_skip_case(reason)` call before returning.
- When introducing a new `_skip_case(reason)` site, place its row in the table above, classify the precondition (A/B/C/D), and confirm the reason string matches one of the standard fragments:
  - `"private optimizer runtime unsupported for JAX <version>"` (A, import-smoke style)
  - `"strict CPU parity backend unavailable: private optimizer runtime not supported"` (A, jax-runtime style)
  - `"GPU device is required"` or `"strict GPU fast backend unavailable: no GPU device detected"` (B)
  - new C/D reasons should describe the missing artifact explicitly.
- Category E skips MUST NOT be added. If a case can only be expressed as a skip on a non-environmental condition, annotate the `_skip_case(...)` call with an `# AUDIT-OK: <reason>` comment immediately above the call site so reviewers can locate the deviation, and append a row to "Action items" above.
