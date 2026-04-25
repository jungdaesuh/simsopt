# Bucket 7 Audit — Core JAX, Reductions, Curve Optimizable, Candidate Ledger, simsoptpp Compat, Stage 2 Seed Report, Accessibility

Date: 2026-04-25
Branch: gpu-purity-stage2-20260405
Auditor: Bucket-7 max-effort test-quality auditor

## 1. Per-file summary table

| File | Tests | Lines | Bottom line |
|------|-------|-------|-------------|
| `tests/core/test_jax_core_specs.py` | 13 | 460 | MOSTLY WEAK — pytree round-trip + dtype echo + happy-path discriminator coverage; 0 negative tests for spec contents (sole "rejects" is `curve_spec_kind` lookalike). The factories themselves do **no** validation beyond `int()/float()/bool()/str()` casts and a dtype-coerce, so the tests cannot tighten what isn't there. |
| `tests/core/test_reductions.py` | 5 | 99 | STRONG. `validate_reduction_mode` neg-test + 2 pairwise-tree parity tests + 2 oracle (`math.fsum`) tests with strict tier verifying it actually beats the default. The pairwise-tree tests are partially tautological (NumPy reference reimplements the same tree), but the Kahan/strict-oracle tests use a true high-precision oracle. |
| `tests/geo/test_curve_optimizable.py` | 1 (skipped) | 90 | DIFF-ONLY: 2-line refactor to use module-scope `setUp/tearDown` for `parameters['jit']` state instead of module-import-time mutation. The test class itself is `@unittest.skip`ed. Diff is correct/defensive but unreached. |
| `tests/geo/test_candidate_ledger.py` | 4 | 387 | OK. Round-trip JSON-fixture tests that pin status discrimination, status-rank ordering, summary→validation rebuild path, campaign-context threading, and unreadable-JSON tolerance. No tests of conflicting status thresholds, sort-key tie-breaks, or schema_version evolution. Module-load pattern leaks `sys.path` after first call (production code hazard, not a test bug). |
| `tests/geo/test_simsoptpp_compat.py` | 3 | 108 | TIGHT for what it covers: shape-mismatch propagation, third-derivative FD parity (`rtol=1e-6, atol=1e-6, eps=1e-6`), ABI fallback caching of `boozer_dresidual_dc`. Only covers 1 of 4 ABI keys (`KEY_BOOZER_DRESIDUAL_DC`) and does not assert `_call_modes` are reset across other tests. |
| `tests/geo/test_stage2_seed_report.py` | 4 | 167 | OK. JSON-fixture tests for status discrimination, salvageable path, sort ordering, and corrupt-results tolerance. No tests of `max_field_error` threshold, sort-key tie-breaks, or empty-scan fallback. |
| `tests/geo/test_accessibility.py` | 6 | 272 | WEAK ON PHYSICS. Tests pin "the JAX kernel is jit-cached at module scope" and "no per-instance jit attrs survive". Zero FD/oracle parity for J/dJ/ddJ values. PortSize cache invalidation is a real behavioral test (`need_to_run_code` flag transitions). |

---

## 2. Top issues (ranked)

| # | file:line | test name | classification | quote | tightening rec |
|---|-----------|-----------|----------------|-------|----------------|
| 1 | `tests/geo/test_accessibility.py:89-112` | `test_projected_enclosed_area_reuses_shared_jit_kernels` | WEAK ASSERTION | `assert accessibility_module._projected_enclosed_area_zphi_grad._cache_size() == 1` | Add a numeric FD parity check: `dJ` against central-difference of `J` at h=1e-6, rtol=1e-6. The cache-size assertion verifies code routing only. |
| 2 | `tests/geo/test_accessibility.py:115-138` | `test_directed_facing_port_reuses_shared_jit_kernels` | WEAK ASSERTION | `assert accessibility_module._upward_facing_grad._cache_size() == 1` | Same: cache-routing only. The actual `upward_facing_pure(nznorm) = sum(max(-nznorm, 0)**2)` is trivially FD-checkable; the gradient is `-2*max(-nznorm, 0)`. Add explicit J/dJ/ddJ value tests. |
| 3 | `tests/geo/test_accessibility.py:141-164` | `test_curve_in_port_penalty_reuses_shared_jit_kernels` | WEAK ASSERTION | `_curve_in_port_penalty_xy_values._cache_size() == 1` | Same — wire in FD gradient parity for `J(curves, port)` perturbed in port and curve dofs. |
| 4 | `tests/geo/test_accessibility.py:167-224` | `test_projected_curve_curve_distance_reuses_shared_jit_kernels` | WEAK ASSERTION | `assert hessian_cache_size > 0` (not even pinned to a value) | Replace with FD parity for `dJ` and finite-difference of `dJ` against `ddJ_ddport` and `ddJ_dportdcoil`. Also pin `hessian_cache_size` to its expected value rather than `> 0`. |
| 5 | `tests/geo/test_accessibility.py:227-248` | `test_projected_curve_convexity_reuses_shared_jit_kernels` | WEAK ASSERTION | `_projected_curve_convexity_zphi_value._cache_size() == 1` | Add J/dJ FD parity. The convexity penalty has well-defined sign so a sign-of-J test is also worthwhile. |
| 6 | `tests/core/test_jax_core_specs.py:183-186` | `test_make_coil_symmetry_spec_defaults_to_identity_without_rotation` | TAUTOLOGICAL | `assert symmetry.scale == 2.5; assert symmetry.has_rotation is False; np.testing.assert_array_equal(symmetry.rotmat, np.eye(3))` | Test only echoes inputs. Add a test that `apply_coil_symmetry` with `has_rotation=False` is a no-op for gamma/gammadash, and that the cache returned is precisely the identity (already partially covered by `apply_coil_symmetry` test). |
| 7 | `tests/core/test_jax_core_specs.py:198-200` | `test_curve_spec_kind_covers_all_supported_curve_variants` | WEAK ASSERTION | `assert curve_spec_kind(curve_spec) == expected_kind` | The discriminator is a chain of `isinstance` returning a string. The dict literal that drives the test is the same set of variants. Tightening: make the test parameterize against the `Literal[...]` typing definition (i.e., `assert set(samples) == set(typing.get_args(CurveSpecKind))`) so adding a variant to the type without the discriminator branch fails CI. |
| 8 | `tests/core/test_jax_core_specs.py:212-219` | `test_surface_spec_kind_covers_supported_fixed_surface_variants` | WEAK ASSERTION | same pattern as #7 | Same fix: add `Literal` cross-check; ensures every spec kind has both a constructor and a discriminator. |
| 9 | `tests/core/test_jax_core_specs.py:248-263` | `test_make_field_eval_and_fixed_surface_flux_specs_preserve_shapes_and_float64` | WEAK ASSERTION | `assert points.points.shape == (5, 3); _assert_is_float64_array(points.points); ...` | Pure shape/dtype echo. Add a test that `make_fixed_surface_flux_spec` infers `nphi/ntheta` from `normal.shape[:2]` and rejects `target.shape != normal.shape[:2]` (currently no such validation in source — file a source-side enhancement). |
| 10 | `tests/core/test_jax_core_specs.py:329-384` | `test_single_stage_runtime_spec_is_a_real_jittable_pytree` | WEAK ASSERTION + tautology | `np.testing.assert_allclose(seed_scalar(runtime), np.array(5.623))` | The 5.623 value is the sum of 3 inputs the test wrote: 0.123 + 4.5 + 1.0 = 5.623. This is a useful sanity that pytree leaves are correctly threaded through `jax.jit`, but the magic number obscures that. Replace with `np.testing.assert_allclose(seed_scalar(runtime), boozer_iota + boozer_G + surface_dofs[0])` so any future change to surface dof[0] index meaning isn't silently absorbed. |
| 11 | `tests/core/test_reductions.py:44-50` | `test_pairwise_sum_axis_matches_numpy_fixed_tree` | TAUTOLOGICAL (partial) | `_numpy_pairwise_sum_axis` reimplements `pairwise_sum_axis` exactly | Useful as a "pin the algorithm" test, but not as an oracle. Add a parallel test against `math.fsum` on flattened axis-1 data with at least 1 ill-conditioned input (e.g., `[1e16, 1.0, ..., 1.0, -1e16]`) so the test detects regressions where pairwise summation accidentally degrades to plain. |
| 12 | `tests/core/test_reductions.py:53-59` | `test_pairwise_sum_flat_matches_numpy_fixed_tree` | TAUTOLOGICAL (partial) | same as #11 | Same fix — add `math.fsum` oracle test for flat case. |
| 13 | `tests/geo/test_simsoptpp_compat.py:65-107` | `test_call_boozer_dresidual_dc_falls_back_to_alpha_only_signature` | TIGHT-SCOPE / MISSING COVERAGE | `boozer_compat._reset_call_modes()` (only resets the global state at start, not after) | The cache leaks to other tests in the same module that exercise `_call_with_abi_fallback` — currently fine because only this 1 of 4 ABI entrypoints is exercised, but add `KEY_BOOZER_RESIDUAL`, `KEY_BOOZER_RESIDUAL_DS`, `KEY_BOOZER_RESIDUAL_DS2` coverage with the same fallback semantics. Also add the success path (with_I returns successfully) — currently the test forces the fallback by raising `TypeError` inside the fake. |
| 14 | `tests/geo/test_simsoptpp_compat.py:40-62` | `test_surface_xyztensorfourier_theta_third_derivative_matches_finite_difference` | WELL-TIGHTENED | `np.testing.assert_allclose(analytical[0], finite_difference, rtol=1.0e-6, atol=1.0e-6)` with `eps=1e-6` | This is the only solid numerical test in the bucket. Tolerance/h relationship: `O(h^2)` truncation gives `~1e-12` truncation, `O(eps_machine/h)` rounding gives `~1e-10`. `1e-6` tolerance is loose by 4 orders of magnitude vs the `O(h^2)` truncation. **Tighten to `rtol=1e-9, atol=1e-9`** — the 3rd-derivative analytical formula is exact and FD should converge cleanly. |
| 15 | `tests/geo/test_simsoptpp_compat.py:26-37` | `test_mwpgp_algorithm_rejects_shape_mismatch` | WELL-TIGHTENED | `with pytest.raises(ValueError, match="shape mismatch")` | Solid — pins both error type and message substring on the simsoptpp C++ entrypoint. No suggested change. |
| 16 | `tests/geo/test_candidate_ledger.py:155-173` | `test_build_candidate_ledger_ranks_stage2_and_single_stage_candidates` | TIGHT (status-rank invariant) | `self.assertEqual(ledger["single_stage"]["best_candidate"]["status"], "research_grade")` | Pins the status-rank invariant. Add a test that `worse-run` is in `reports[1]` (i.e., that the sort actually orders by `status_rank` first), and add a tie-break test where two candidates have the same status. |
| 17 | `tests/geo/test_candidate_ledger.py:175-227` | `test_build_candidate_ledger_can_rebuild_validation_from_summary` | TIGHT-SCOPE | `self.assertEqual(ledger["single_stage"]["candidate_count"], 1)` | Verifies the rebuild path runs but only checks count and final status. Add an assertion that the rebuilt validation matches what `build_continuation_validation_report` would produce on the raw summary, since the test is the only coverage of that fallback path. |
| 18 | `tests/geo/test_candidate_ledger.py:229-326` | `test_build_candidate_ledger_threads_campaign_context_and_schedule` | OK | `self.assertEqual(best_candidate["campaign"]["run_id"], "run-001")` | Threads through several layers of context. Could add a multi-campaign test where `best_candidate` from campaign A vs campaign B yields the campaign-context attribution correctly (currently 1 campaign, 1 donor). |
| 19 | `tests/geo/test_candidate_ledger.py:328-382` | `test_build_candidate_ledger_tolerates_unreadable_stage2_results` | TIGHT | `assertTrue(any("results.json is unreadable: JSONDecodeError" in failure for failure in corrupt_report["failures"]))` | Solid — pins both classification and message substring. No suggested change. |
| 20 | `tests/geo/test_stage2_seed_report.py:70-88` | `test_evaluate_candidate_reports_research_grade_for_complete_clean_seed` | TIGHT | `self.assertEqual(report["status"], "research_grade")` | Pins the happy-path discrimination. |
| 21 | `tests/geo/test_stage2_seed_report.py:90-111` | `test_evaluate_candidate_marks_results_only_seed_as_salvageable` | TIGHT | `assertIn("restart artifacts are incomplete", report["failures"])` | Solid. Could add explicit check that legacy warning appears verbatim. |
| 22 | `tests/geo/test_stage2_seed_report.py:113-139` | `test_build_catalog_sorts_best_candidate_first` | TIGHT (sort by sort_key) | `self.assertEqual(catalog["best_candidate"]["run_dir"], str(best_run))` | Verifies sort ordering on field_error. Add: test where two candidates differ only on `coil_coil_margin` (sort_key index 2) so the secondary sort key is exercised. Currently no test verifies that index-1 (field_error) actually fires when index-0 (status_rank) ties. |
| 23 | `tests/geo/test_stage2_seed_report.py:141-163` | `test_evaluate_candidate_rejects_unreadable_results_without_crashing` | TIGHT | parallel to candidate-ledger version (#19) | Solid. |
| 24 | `tests/geo/test_accessibility.py:251-271` | `test_port_size_refreshes_cached_port_solve_on_parent_curve_mutation` | WELL-TIGHTENED | `assert objective.need_to_run_code is True; assert objective._port_area_solve is None` after mutation; `assert not np.allclose(objective._port_area_solve, initial_port_solve)` after re-solve | Genuine behavioral test of cache invalidation through `recompute_bell`. Solid — keep. |
| 25 | `tests/geo/test_curve_optimizable.py:14` | (module setup, diff vs upstream) | TIGHT (defensive) | `setUpModule, tearDownModule = make_module_jit_hooks(parameters, value=False)` | Replaces module-import-time mutation of `parameters['jit']` (which leaks across the whole pytest session) with proper module setUp/tearDown. The change is correct, but the tests inside the class are `@unittest.skip`ed, so the hooks fire but no logic depends on them. The fix is preventative — when the test is unskipped in the future, jit state will be properly contained. |

---

## 3. Missing coverage (bullets)

### Reductions

- **Associativity/commutativity property test**: pairwise sum should equal `math.fsum` to within float64 ULPs for both random and ill-conditioned inputs at multiple sizes (especially axis sizes that are NOT powers of two — current tests use 257 (just above 256) and 5×7, but no test covers `axis_size == 2**k - 1` boundary or `axis_size == 1`).
- **Empty-array path**: `pairwise_sum_axis(arr, axis=k)` with `arr.shape[k] == 0` falls through to `jnp.sum`. No test verifies the dtype of the resulting zero matches the input dtype.
- **GPU vs CPU determinism**: per project memory, GPU reduction order is a parity-acceptance concern. No test runs the same reduction on CPU and (hypothetically) on GPU and asserts bitwise equality. This is the entire point of `pairwise_sum_*` existing — guarantee bit-stable order across devices. **This is the largest coverage gap in this bucket.**
- **`scalar_square_sum` with `default="pairwise"`**: only `default="vdot"` is tested. The `pairwise` baseline branch is unreached.
- **`compensated_sum_flat` size=0 / size=1 boundaries**: not exercised.
- **`compensated_sum_flat` with negative + positive cancellations**: only tests "small positive terms after a giant value." Add a true cancellation case `[1e16, 1.0, -1e16]` where the answer is `1.0` and naïve summation gives `0.0`.

### Specs

- **No negative tests for content** (e.g., dofs.size mismatch with order, quadpoints out of [0,1), nfp <= 0). The factory functions accept anything, so adding source-side `assert/raise` AND a test pinning the rejection would be the right pair.
- **`make_coil_symmetry_spec` rotmat dimensionality**: factory calls `_as_float64_array(rotmat)` and stores it without shape check. A non-3×3 rotmat would silently reach `apply_coil_symmetry` and break in matmul. Pin a `ValueError` on `rotmat.shape != (3, 3)`.
- **`OptimizableDofMapSpec.input_mode`**: stored as `str(input_mode)`; no test verifies that `"full"`, `"partial"`, or any well-defined alphabet is accepted — and downstream code presumably branches on this string.
- **Pytree-leaf hash stability**: `treedef` equality is asserted in the round-trip but the hash of the meta_fields tuple isn't verified across two independently constructed specs with the same metadata. (Important for cache keying contract.)

### Curve optimizable

- The module is mostly skipped. When unskipped, the existing FD jacobian comparison block (lines 60-69) is itself commented out. Restoring that comparison would tighten significantly.

### Candidate ledger

- **Sort-key tie-break tests**: when two candidates have equal `status_rank`, the next sort key (field_error, then non_qs, then boozer_residual) needs explicit coverage. Currently sort orderings are tested by single-key differences only.
- **`research_usable_count` boundary**: covered count==2 case but not count==0 or count==N where N>2 with mixed statuses.
- **Schema version evolution**: `_LEDGER_SCHEMA_VERSION = 1`. No test pins the schema version into the output ledger or guards against accidental bumps.
- **`build_campaign_context_map` with no campaign summary**: covered implicitly. Not tested when `campaign_summary.json` exists but `reports` is malformed.
- **Empty `single_stage_root`**: `build_candidate_ledger` should produce `best_candidate=None`. Not tested.

### Stage 2 seed report

- **`max_field_error` threshold**: parameter accepted but never exercised in tests. Add a test that `field_error > max_field_error` produces a `failures` entry and pushes status to `salvageable`.
- **`detect_stage2_seed_artifacts`** as a public surface — not directly tested.
- **Catalog with zero candidates**: `find_stage2_run_dirs` returns empty → `best_candidate is None`, `passed=False`. Not tested.

### simsoptpp compat

- **`KEY_BOOZER_RESIDUAL`, `KEY_BOOZER_RESIDUAL_DS`, `KEY_BOOZER_RESIDUAL_DS2`**: the ABI fallback code path exists for these keys but no test covers them. Parametrize the existing test over all 4 keys.
- **Success path (with_I works)**: the test forces fallback. Add a test where the fake `func` succeeds with 9-arg signature and the cached mode becomes `"with_I"` and subsequent calls reuse it.
- **Non-fallback TypeError**: the source code only swallows `TypeError` whose `str(exc)` contains `"incompatible function arguments"`. A `TypeError` with a different message should re-raise. Add a test for that.

### Accessibility

- **Numerical FD parity for J, dJ, ddJ**: the bulk of the file is "kernel is jit-cached" routing assertions. Replace at least one routing assertion per Optimizable class with a true J/dJ FD check at small h (e.g., `h=1e-6`, `rtol=1e-6`).
- **`PortSize.explicit_solve` vs `PortSize.jax_solve`**: `solver` parameter accepts `"jax"` or `"explicit"`. Only `explicit` is exercised. The JAX solver path is unreached in tests.
- **`PortSize` direction**: `radial` and `vertical` accepted; only `vertical` tested.
- **Sign convention test for upward-facing**: `upward_facing_pure(nznorm) = sum(max(-nznorm, 0)**2)` so the penalty is zero when nznorm ≥ 0 everywhere and positive when any component points "down". A trivial sign test would catch sign-flip regressions.
- **Projection consistency**: `xy` and `zphi` projections live in parallel kernel pairs; no test verifies they agree on a curve that is symmetric under the relevant rotation.

---

## 4. Tightening playbook (P0)

1. **Add GPU/CPU bitwise reduction parity tests** (`tests/core/test_reductions.py`): when `SIMSOPT_JAX_PLATFORM=cuda`, run `pairwise_sum_flat`, `pairwise_sum_axis`, `compensated_sum_flat`, `scalar_square_sum(strict_oracle)` on CPU and CUDA with the same inputs and assert bit-identity. This is the load-bearing claim of the entire `reductions.py` module per project memory and is currently untested.
2. **Replace cache-routing assertions with FD parity** (`tests/geo/test_accessibility.py`): for each of the 5 `*_reuses_shared_jit_kernels` tests, add a J/dJ FD parity check at `h=1e-6, rtol<=1e-6`. Keep the cache-size assertions as a secondary check, not the primary.
3. **Tighten `gammadash2dash2dash2_lin` FD test** (`tests/geo/test_simsoptpp_compat.py:60`): `rtol=1e-6` is loose by 4 orders of magnitude vs the truncation/rounding optimum. Push to `rtol=1e-9, atol=1e-9`.
4. **Parametrize ABI fallback test over all 4 keys + success path + non-matching TypeError** (`tests/geo/test_simsoptpp_compat.py`): expand from 1 entrypoint × 1 path → 4 entrypoints × 3 paths.
5. **Add Literal-vs-discriminator cross-check** (`tests/core/test_jax_core_specs.py`): `assert set(samples) == set(typing.get_args(CurveSpecKind))` and same for `SurfaceSpecKind`. Catches the case where someone adds a variant to the `Literal[...]` type but forgets the `isinstance` branch in `curve_spec_kind`.
6. **Add `math.fsum` oracle test for `pairwise_sum_axis` and `pairwise_sum_flat`** (`tests/core/test_reductions.py`): replace the duplicate-NumPy-pairwise-implementation oracle with a true high-precision oracle on at least one ill-conditioned input.
7. **Add tie-break sort-key coverage** for both candidate ledger and stage 2 seed report: equal `status_rank` candidates that differ only on the next sort key.
8. **Add validation/rejection tests for spec content** (`tests/core/test_jax_core_specs.py`): `make_coil_symmetry_spec(rotmat=np.eye(2))` should raise (currently silently accepted). Either tighten the source to validate or document the looseness.

---

## 5. Import-cycle audit

The known import cycle is `simsopt.geo.curve ↔ simsopt.jax_core` via `_as_jax_float64`/`_as_runtime_float64*` helpers in `src/simsopt/geo/curve.py:64-86`. Tests that exercise lazy-import seams in this bucket:

| Test | File | What it imports | Cold-path exercised? |
|------|------|-----------------|-----------------------|
| `test_jax_core_specs.py` (all 13) | `tests/core/test_jax_core_specs.py:6-37` | `from simsopt.jax_core import (...)` at module import time | NO. The first import of `simsopt.jax_core` triggers the full transitive load of geo modules via `jax_core/specs.py:1110` which lazily imports `from ..geo.surface_fourier_jax import stellsym_scatter_indices`. By the time the test runs, both sides of the cycle are loaded. |
| `test_reductions.py` (all 5) | `tests/core/test_reductions.py:8` | `from simsopt.jax_core.reductions import (...)` | NO. `reductions.py` does NOT touch `_math_utils`, so it would in principle bypass the cycle — but the test file does not assert this; it just imports normally. Cold-state isolation would require `monkeypatch.delitem(sys.modules, ...)` for `simsopt.geo.*` and `simsopt.jax_core.*` before the import. **Not done in any test.** |
| `test_curve_optimizable.py` | `tests/geo/test_curve_optimizable.py:8` | `from simsopt.geo.curverzfourier import CurveRZFourier` | NO. Imports `curverzfourier` first which depends on `curve.py` — the lazy seam at `curve.py:67` uses `from ..jax_core._math_utils import as_jax_float64 as _distributed_as_jax_float64`, and this import happens inside the `_as_jax_float64` function body. So the lazy seam is exercised on first call to `_as_jax_float64`, but the test class is `@unittest.skip`ed and the function body is never reached. |
| `test_candidate_ledger.py` | `tests/geo/test_candidate_ledger.py` | `examples/.../candidate_ledger.py` (via importlib) | N/A. Pure JSON-shape tests; no JAX/curve cycle interaction. |
| `test_stage2_seed_report.py` | `tests/geo/test_stage2_seed_report.py` | `examples/.../stage2_seed_report.py` (via importlib) | N/A. Pure JSON-shape tests. |
| `test_simsoptpp_compat.py` | `tests/geo/test_simsoptpp_compat.py:6-7` | `simsoptpp` (via `pytest.importorskip`), `surfacexyztensorfourier`, `surfaceobjectives`, `_simsoptpp_boozer_compat` | NO. Cold-path not exercised; this test file exists to test the C++/Python ABI seam, not the JAX import cycle. |
| `test_accessibility.py` | `tests/geo/test_accessibility.py:3-14` | `simsopt.geo.accessibility`, `simsopt.geo.curve`, etc. | NO. Imports happen at module-load time. The accessibility code itself uses `from .jit import jit` which conditionally returns `jax.jit` based on `parameters["jit"]`. The cache-size assertions assume `parameters["jit"]` is True (its default). If a previous test mutated this flag (e.g., the candidate-ledger leaked sys.path issue, or test_curve_optimizable.py before the recent fix), behaviour would differ — but no test asserts the flag is in its expected state at start. |

**Verdict on cold-path coverage**: ZERO tests in this bucket exercise the cold-state lazy-import seam. Every test pre-imports both halves of the cycle by the time the seam fires. To genuinely test the lazy seam, a test would need to:

```python
import sys
for name in [k for k in sys.modules if k.startswith("simsopt")]:
    del sys.modules[name]
# Then import only one side and call the lazy helper
from simsopt.jax_core._math_utils import as_jax_float64
arr = as_jax_float64([1.0, 2.0, 3.0])
# Verify simsopt.geo.curve still loads successfully after this
```

Adding such a test belongs in a separate `tests/test_import_seam.py` file and should run with the `simsopt` cache fully cleared.

---

## 6. `test_curve_optimizable.py` diff-only audit

**Scope**: Only the 2-line diff vs `upstream_hss/master`. The unchanged upstream test (`subtest_curve_length_optimisation`) is `@unittest.skip`ed and not in audit scope.

| Diff hunk | Classification | Quote | Notes |
|-----------|----------------|-------|-------|
| `tests/geo/test_curve_optimizable.py:4` (added) | TIGHT (defensive) | `from _jit_test_state import make_module_jit_hooks` | Imports a 11-line worktree-root helper that defines a setUp/tearDown pair. Helper is at `/Users/suhjungdae/code/columbia/simsopt-jax/_jit_test_state.py`. |
| `tests/geo/test_curve_optimizable.py:14` (replaced) | TIGHT (defensive) | `setUpModule, tearDownModule = make_module_jit_hooks(parameters, value=False)` | Replaces upstream's module-import-time `parameters['jit'] = False` with proper unittest module setUp/tearDown. Correct fix — module-import-time mutation would leak across the entire pytest session. |

**Effective behavior**: When pytest collects this module, `setUpModule` sets `parameters['jit'] = False`. After the (skipped) class runs, `tearDownModule` restores the original value. Because the test class is skipped, the helper is exercised but the actual logic that depends on `jit=False` is never reached.

**Risk**: None. The change is purely preventative.

**Suggestion**: Either un-skip the test (the skip is from "MJL 2025-01-24" and may be stale) or remove the `setUpModule` machinery entirely until the test is restored. Currently the diff exists to ensure correctness of an unreached code path.

---

## Executive summary

- **Strongest tests in the bucket**: `test_simsoptpp_compat.py` third-derivative FD parity, `test_reductions.py` Kahan/oracle test, `test_accessibility.py:251` `PortSize` cache-invalidation behavioral test, candidate-ledger and seed-report unreadable-JSON tolerance tests.
- **Weakest tests in the bucket**: 5 of 6 accessibility tests assert only "the JIT cache has size 1" — zero numeric J/dJ/ddJ coverage despite easy FD checks being available.
- **Largest single coverage gap**: GPU-vs-CPU bitwise-reduction parity is the entire load-bearing claim of `reductions.py` and is untested.
- **Spec validators are weak in the source**: factories accept arbitrary inputs (no shape/range checks). Tightening tests must pair with source-side `raise ValueError(...)` additions.
- **Import-cycle cold path is not tested anywhere in this bucket** — the lazy seam in `curve.py:64-86` is never exercised in cold-import isolation.

DONE — report at /Users/suhjungdae/code/columbia/simsopt-jax/.artifacts/jax-test-audit-2026-04-25/bucket7_core_curve_misc.md
