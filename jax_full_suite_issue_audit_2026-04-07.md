# JAX Full Suite Issue Audit (2026-04-07)

This document captures the major failure families uncovered by running the full `simsopt-jax` pytest suite in the correct JAX 0.9.2 environment.

Validation run:

- Environment: `columbia-jax-0.9.2`
- Command: `conda run --no-capture-output -n columbia-jax-0.9.2 python -m pytest -q`
- Result: `1340 passed`, `499 failed`, `65 errors`, `111 skipped`, `643 subtests passed`
- Runtime: `29m13s`

## Top Clusters

- `134` unique failing nodeids in `tests/integration/test_single_stage_jax.py`
- `94` unique failing nodeids in `tests/integration/test_stage2_jax.py`
- `27` unique failing nodeids in `tests/geo/test_boozersurface_jax.py`

Largest class-level clusters from `.pytest_cache/v/cache/lastfailed`:

- `67` in `TestAdjointSolveConsistency`
- `39` in `TestStage2OptimizerContract`
- `20` in `TestTraceableObjective`
- `18` in `TestUpstreamFactoryBoozerMatrix`
- `13` in `TestMixedQuadratureParity`

## Issue Families

### 1. BoozerSurface class-identity / import contamination

Symptoms:

- `BoozerSurface` rejects a `SurfaceXYZTensorFourier` instance in `src/simsopt/geo/boozersurface.py`
- The failure reproduces when running `tests/test_benchmark_helpers.py` before `tests/geo/test_boozersurface.py`
- The same Boozer tests pass in isolation

Representative failures:

- `tests/geo/test_boozersurface.py::BoozerSurfaceTests::test_run_code`
- many fixture-setup errors in `tests/integration/test_single_stage_jax.py`

Likely root cause:

- module/class identity drift caused by direct module loading and `sys.modules` patching in tests
- strict `isinstance(...)` checks in `BoozerSurface` amplify that drift into suite-wide failures

TODO:

- [ ] Reproduce the minimal import-order contamination path and document the exact module identity mismatch
- [ ] Audit test helpers that patch `sys.modules` or use `spec_from_file_location(...)`
- [ ] Decide whether to harden `BoozerSurface` type validation or to isolate/undo the test-side module contamination
- [ ] Add an order-sensitive regression test that fails if benchmark/import helper tests poison Boozer CPU objects
- [ ] Re-run `tests/test_benchmark_helpers.py tests/geo/test_boozersurface.py -q`

### 2. Stage 2 tests require missing external equilibrium data

Symptoms:

- Stage 2 script/probe tests fail with `FileNotFoundError`
- missing file:
  `/Users/suhjungdae/code/columbia/DATABASE/EQUILIBRIA/wout_nfp22ginsburg_000_014417_iota15.nc`

Representative failures:

- `tests/integration/test_stage2_jax.py::TestStage2BananaBoundary::test_stage2_probe_reports_shared_production_banana_curve[cpu]`
- `tests/integration/test_stage2_jax.py::TestStage2OptimizerContract::*`

Likely root cause:

- test suite assumes a local Columbia data fixture outside the repo
- no repo-local fixture or skip/fallback contract exists for absent data

TODO:

- [ ] Decide whether this file is required test infrastructure or an optional local-only fixture
- [ ] If required, document the fixture bootstrap clearly in repo docs and CI contracts
- [ ] If optional, add an explicit skip or repo-local reduced fixture for Stage 2 tests
- [ ] Remove hard dependence on `/Users/suhjungdae/code/columbia/DATABASE/...` for default local test execution
- [ ] Re-run `tests/integration/test_stage2_jax.py -q`

### 3. Single-stage wrapper / helper contract regressions

Symptoms:

- helper call sites expect `(value, grad)` style contracts
- current runtime path sometimes returns scalar `0-d` JAX arrays
- monkeypatch-based tests fail because wrapper interfaces no longer match expected semantics

Representative failure:

- `tests/integration/test_single_stage_jax.py::TestBoozerResidualValue::test_value_path_matches_residual_helper_not_penalty_objective`

Likely root cause:

- recent strict-JAX cleanup changed helper return shapes or wrapper boundaries in `src/simsopt/geo/surfaceobjectives_jax.py`
- tests in `TestAdjointSolveConsistency` and `TestTraceableObjective` suggest the seam is shared, not isolated

TODO:

- [ ] Audit `_value_and_direct_coil_derivative` and the callable contract it expects from objective builders
- [ ] Check `strict_scalar_grad` / `strict_scalar_value_and_grad` integration points for value-vs-tuple drift
- [ ] Compare current wrapper contracts against the tests tightened in the April 6 commit train
- [ ] Restore consistent public helper semantics for single-stage objective wrappers
- [ ] Re-run `tests/integration/test_single_stage_jax.py -q`

### 4. CPU import-safety regressions after host-boundary cleanup

Symptoms:

- package-root CPU import smoke tests now require JAX import
- native CPU import tests fail under JAX-blocking smoke harnesses

Representative failures:

- `tests/test_jax_import_smoke.py::test_import_package_root_native_cpu_does_not_require_jax_runtime`
- `tests/test_jax_import_smoke.py::test_native_cpu_backend_selection_does_not_require_jax_runtime`
- `tests/test_jax_import_smoke.py::test_import_cpu_geo_core_entrypoints_without_jax`

Likely root cause:

- `src/simsopt/_core/jax_host_boundary.py` imports `jax` eagerly at module import time
- CPU-safe package entrypoints no longer remain JAX-optional

TODO:

- [ ] Move JAX imports in `src/simsopt/_core/jax_host_boundary.py` behind function scope where possible
- [ ] Re-validate package-root import behavior without JAX availability
- [ ] Confirm CPU geo/core entrypoints still import cleanly with blocked JAX
- [ ] Re-run `tests/test_jax_import_smoke.py -q`

### 5. Objective / derivative parity fallout beyond the main JAX seam

Symptoms:

- wide Taylor/parity failures in curve, surface, and utility objective tests
- failures are not limited to the new JAX-only files

Representative failures:

- `tests/objectives/test_utilities.py::UtilityObjectiveTesting::test_quadratic_penalty`
- `tests/objectives/test_utilities.py::UtilityObjectiveTesting::test_quadratic_penalty_hostifies_jax_scalar_objective`
- many subfailures in `tests/geo/test_curve_objectives.py`
- many subfailures in `tests/geo/test_surface_objectives.py`

Likely root cause:

- recent hostification / derivative-boundary cleanup changed when values are coerced or how derivative payloads are combined
- fallout likely passes through:
  - `src/simsopt/_core/derivative.py`
  - `src/simsopt/_core/jax_host_boundary.py`
  - `src/simsopt/objectives/utilities.py`

TODO:

- [ ] Reproduce utility-objective failures in isolation
- [ ] Check whether recent hostification changed return types expected by objective tests
- [ ] Audit derivative block coercion for effects on Taylor-test paths
- [ ] Re-run `tests/objectives/test_utilities.py -q`
- [ ] Re-run targeted `tests/geo/test_curve_objectives.py` and `tests/geo/test_surface_objectives.py` slices

### 6. Stage 2 and single-stage parity stacks remain red even in the correct env

Observation:

- The `0.9.2` env removes the earlier wrong-runtime noise, but the main JAX integration stacks are still broadly red
- This means the remaining failures are genuine integration issues, not just environment mismatch

TODO:

- [ ] Treat `columbia-jax-0.9.2` as the authoritative local env for private-optimizer / ondevice debugging
- [ ] Stop using `columbia-repro-b4815f18` for primary JAX-private suite triage
- [ ] Re-cluster after each major seam fix to confirm which failure family collapses

## Recommended Fix Order

- [ ] Fix BoozerSurface import/class-identity contamination first
- [ ] Fix CPU import-safety regressions in `jax_host_boundary.py`
- [ ] Fix Stage 2 external-fixture handling
- [ ] Fix single-stage wrapper/helper contract drift in `surfaceobjectives_jax.py`
- [ ] Re-audit objective/derivative parity fallout after the earlier seam fixes land
- [ ] Re-run full `pytest -q` in `columbia-jax-0.9.2`

## Quick Repro Commands

- `cd /Users/suhjungdae/code/columbia/simsopt-jax && conda run -n columbia-jax-0.9.2 python -m pytest -q tests/test_benchmark_helpers.py tests/geo/test_boozersurface.py::BoozerSurfaceTests::test_run_code -x`
- `cd /Users/suhjungdae/code/columbia/simsopt-jax && conda run -n columbia-jax-0.9.2 python -m pytest -q tests/integration/test_single_stage_jax.py -x`
- `cd /Users/suhjungdae/code/columbia/simsopt-jax && conda run -n columbia-jax-0.9.2 python -m pytest -q tests/integration/test_stage2_jax.py -x`
- `cd /Users/suhjungdae/code/columbia/simsopt-jax && conda run -n columbia-jax-0.9.2 python -m pytest -q tests/test_jax_import_smoke.py -x`
