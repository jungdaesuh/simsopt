# Upstream Merge Plan — simsopt-jax ← hiddenSymmetries/simsopt

**Date**: 2026-03-29
**Source**: `hiddenSymmetries/simsopt` master at `4883116f` (2026-03-24)
**Target**: `jax-port` branch at `b746e3e1` (2026-03-29)
**Common ancestor**: `539c0f98` (2025-06-09)
**Divergence**: 721 upstream commits, 273 jax-port commits
**Conflict hunks**: 30 (across 14 files)

---

## Summary

- **14 files** with merge conflicts (30 conflict hunks)
- **105 files** auto-merged (no conflict)
- **8 new files** from upstream (gained for free)
- **114 jax-port-only files** (untouched by merge)

---

## Pre-Merge

- [ ] **Fetch upstream remote into simsopt-jax**
  - `git remote add upstream_hss https://github.com/hiddenSymmetries/simsopt.git 2>/dev/null || true`
  - `git fetch upstream_hss`
  - Note: `upstream_hss` already exists in this workspace. The add command is idempotent for fresh clones.
  - Why: need the upstream refs up-to-date locally

- [ ] **Create a merge branch**
  - `git checkout -b merge-upstream-2026-03-29 jax-port`
  - Why: preserve jax-port as-is until merge is validated

- [ ] **Record a minimal pre-merge baseline on current `jax-port`**
  - Goal: capture a small, already-green reference set before changing the API surface. Do **not** add new parity tests on the old `vectorize` / Python penalty path before the merge.
  - Run only the existing targeted slices that already pass:
    - `conda run -n columbia-jax-0.9.2 python -m pytest /Users/suhjungdae/code/columbia/simsopt-jax/tests/integration/test_single_stage_jax.py -k "TestExactSolveCPUJAXParity" -v`
    - `conda run -n columbia-jax-0.9.2 python -m pytest /Users/suhjungdae/code/columbia/simsopt-jax/tests/integration/test_stage2_jax.py -k "TestStage2BananaBoundary" -v`
    - `conda run -n columbia-jax-0.9.2 python -m pytest /Users/suhjungdae/code/columbia/simsopt-jax/tests/test_benchmark_helpers.py -k "single_stage_init_parity or stage2_e2e_comparison" -v`
  - Save the command lines and outputs as the before/after comparison anchor.
  - Why: need a trusted pre-merge reference without spending time hardening soon-to-be-replaced pre-merge surfaces

---

## Conflict Resolution — 14 Files

### Critical (affects GPU/JAX functionality)

- [ ] **`src/simsopt/geo/jit.py`** — 1 conflict hunk
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/jit.py`
  - Upstream: kept `jax.config.update('jax_platform_name', 'cpu')` + changed `**args`
  - Jax-port: **removed the hardcoded CPU line** + changed to `**kwargs`
  - Resolution: **keep jax-port version** (removed CPU hardcode + `**kwargs`). This is the GPU-enabling change. Upstream's CPU hardcode would kill all GPU functionality.
  - Why it matters: **This single line is the difference between CPU-only and GPU-capable.** Upstream forces `jax_platform_name='cpu'` at import time. Jax-port removed it so `backend.py` can select the platform. Accepting upstream here would break the entire GPU port.

- [ ] **`src/simsopt/geo/boozersurface.py`** — 11 conflict hunks
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/boozersurface.py`
  - Upstream: deleted `boozer_penalty_constraints()` (Python method), removed `vectorize` param from all solvers, added `_get_residual_vector_and_jacobian()`, added `weight_inv_modB` to `minimize_boozer_penalty_constraints_ls`, added `method='manual'` option
  - Jax-port: reformatted with ruff, retained both Python and C++ methods, retained `vectorize` param
  - Resolution: **accept upstream deletions** (remove Python method, remove `vectorize`), **keep jax-port formatting**, **accept upstream new features** (manual ls, weight_inv_modB in ls)
  - Why it matters: The CPU `BoozerSurface` is the parity reference. It must match upstream exactly. The jax-port's `BoozerSurfaceJAX` is a separate class and unaffected.

- [ ] **`src/simsopt/geo/surfacerzfourier.py`** — 3 conflict hunks
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/surfacerzfourier.py`
  - Upstream: +895 lines — major expansion (flip_z, flip_phi, flip_theta, rotate_half_field_period, spectral_width, `condense_spectrum()` with JAX `@jax.jit`)
  - Jax-port: +390 lines — VJP methods for `dgamma_by_dcoeff_vjp`, `dgammadash1_by_dcoeff_vjp`, etc. (C++ additions for the JAX gradient pipeline)
  - Resolution: **accept both** — upstream new methods and jax-port VJP methods are in different code sections. The 3 conflicts are likely in shared areas (imports, class definition header).
  - Why it matters: Both additions are needed. Upstream's `condense_spectrum()` uses JAX directly. Jax-port's VJPs are used by the Boozer solver gradient path.

### Medium (shared infrastructure)

- [ ] **`src/simsopt/geo/curve.py`** — 2 conflict hunks
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/curve.py`
  - Upstream: +404 lines — `JaxCurve` class, `JaxCurveXYZFourier`, `create_equally_spaced_curves(use_jax_curve=True)`, curve VJP methods, `RotatedCurve` improvements
  - Jax-port: +927 lines — `CurveCWSFourier`, `CurveCWSFourierCPP`, Columbia-specific curve types, `RotatedCurve` changes
  - Resolution: **accept both** — different features. Conflicts likely at `RotatedCurve` and class registration areas.
  - Why it matters: Upstream's `JaxCurve` is directly useful for the GPU port — it provides JAX-native curve geometry with VJPs that `BiotSavartJAX` could leverage instead of reimplementing Fourier evaluation.

- [ ] **`src/simsopt/geo/curveobjectives.py`** — 1 conflict hunk
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/curveobjectives.py`
  - Upstream: +132 lines — refactored objectives, new force/torque terms
  - Jax-port: +433 lines — Columbia-specific curve objectives
  - Resolution: **accept both**
  - Why it matters: No direct JAX interaction, but Columbia objectives are needed for the single-stage pipeline.

- [ ] **`src/simsopt/geo/curveplanarfourier.py`** — 2 conflict hunks
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/curveplanarfourier.py`
  - Upstream: +114 lines — `JaxCurvePlanarFourier`, dof naming, docstrings
  - Jax-port: +71 lines — derivative methods
  - Resolution: **accept both**
  - Why it matters: Upstream's `JaxCurvePlanarFourier` adds JAX-native planar curves.

- [ ] **`src/simsopt/field/coil.py`** — 1 conflict hunk
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/coil.py`
  - Upstream: +514 lines — major expansion (new coil types, force objectives, `JaxCurrent` integration)
  - Jax-port: +1 -1 (trivial 1-line change)
  - Resolution: **accept upstream**, re-apply jax-port's 1-line change
  - Why it matters: Upstream's coil expansion is substantial. Jax-port barely touched this file.

### Low (imports, config, tests)

- [ ] **`src/simsopt/field/__init__.py`** — 1 conflict hunk
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/__init__.py`
  - Upstream: +2 lines (new imports)
  - Jax-port: +58 -27 (JAX module imports, try/except guards for `BiotSavartJAX`, `SquaredFluxJAX`)
  - Resolution: **keep jax-port's JAX imports**, add upstream's new imports
  - Why it matters: The jax-port's try/except import guards prevent CPU-only installs from breaking.

- [ ] **`src/simsopt/configs/__init__.py`** — 1 conflict hunk
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/configs/__init__.py`
  - Upstream: +1 -1 (import plumbing change)
  - Jax-port: +8 -1 (added configs)
  - Resolution: **accept both**
  - Note: the `get_ncsx_data` → `get_data("ncsx")` API change lives in `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/configs/zoo.py`, not `__init__.py`. See post-merge item for full caller sweep.

- [ ] **`src/simsopt/util/__init__.py`** — 1 conflict hunk
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/util/__init__.py`
  - Upstream: +2 (new util exports)
  - Jax-port: +11 -5 (JAX util exports)
  - Resolution: **merge both import lists**

- [ ] **`pyproject.toml`** — 2 conflict hunks
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/pyproject.toml`
  - Upstream: +2 -2 (dependency version bumps)
  - Jax-port: +25 -6 (JAX optional dependencies, test config)
  - Resolution: **accept upstream version bumps**, keep jax-port's JAX extras

- [ ] **`tests/field/test_biotsavart.py`** — 1 conflict hunk
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/tests/field/test_biotsavart.py`
  - Upstream: +2 -1 (trivial import change)
  - Jax-port: +3 -1 (trivial import change)
  - Resolution: trivial — merge import lines

- [ ] **`tests/geo/test_curve.py`** — 2 conflict hunks
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/tests/geo/test_curve.py`
  - Upstream: +501 lines (new JaxCurve tests, coil optimization tests)
  - Jax-port: +156 lines (CurveCWSFourier tests)
  - Resolution: **accept both** — different test classes

- [ ] **`tests/geo/test_curve_objectives.py`** — 1 conflict hunk
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/tests/geo/test_curve_objectives.py`
  - Upstream: +189 -109 (refactored objective tests)
  - Jax-port: +89 -1 (added tests)
  - Resolution: **accept upstream refactor**, re-apply jax-port additions

---

## Auto-Merged Files (105 — verify only)

- [ ] **Spot-check auto-merged files after merge**
  - `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/surfaceobjectives.py` — upstream +1 -1, jax-port +545 -228. Auto-merged but large jax-port delta; verify the M5 wrappers survived intact.
  - `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/surface.py` — upstream +23 -25, jax-port +17 -6. Both modified the base surface class.
  - `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/tracing.py` — upstream +3 -1, jax-port +9 -4.
  - `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/curvexyzfourier.py` — upstream +66 -14, jax-port -1 (trivial).
  - `/Users/suhjungdae/code/columbia/simsopt-jax/tests/field/test_particle.py` — upstream +13 -18, jax-port +15.
  - `/Users/suhjungdae/code/columbia/simsopt-jax/tests/geo/test_surface.py` — upstream +20 -7, jax-port +21.
  - Why: auto-merge can silently combine incompatible changes. These files have the highest risk.

---

## Post-Merge

- [ ] **Fix `jit.py` platform selection**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/jit.py`
  - Also: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/backend.py`
  - After resolving the conflict (keeping jax-port's removal of hardcoded CPU), verify that `backend.py`'s platform selection still works and that upstream's `JaxCurve` respects it.
  - Test: `SIMSOPT_JAX_PLATFORM=cpu python -c "from simsopt.geo.curve import JaxCurve; import jax; print(jax.default_backend())"`

- [ ] **Update `get_ncsx_data()` / `get_hsx_data()` / `get_giuliani_data()` → `get_data("ncsx")` calls**
  - Upstream changed the config API in `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/configs/zoo.py`. The old per-config functions are replaced by the generic `get_data(name)`.
  - **91 call sites across 25 files** need updating. Full list:
    - `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/configs/zoo.py` (definition site)
    - `/Users/suhjungdae/code/columbia/simsopt-jax/tests/geo/surface_test_helpers.py` (shared test fixtures)
    - `/Users/suhjungdae/code/columbia/simsopt-jax/tests/geo/test_boozersurface.py`
    - `/Users/suhjungdae/code/columbia/simsopt-jax/tests/geo/test_curve.py`
    - `/Users/suhjungdae/code/columbia/simsopt-jax/tests/geo/test_curve_objectives.py`
    - `/Users/suhjungdae/code/columbia/simsopt-jax/tests/geo/test_surface_objectives.py`
    - `/Users/suhjungdae/code/columbia/simsopt-jax/tests/geo/test_qfm.py`
    - `/Users/suhjungdae/code/columbia/simsopt-jax/tests/geo/test_finitebuild.py`
    - `/Users/suhjungdae/code/columbia/simsopt-jax/tests/geo/test_strainopt.py`
    - `/Users/suhjungdae/code/columbia/simsopt-jax/tests/geo/test_surface_rzfourier.py`
    - `/Users/suhjungdae/code/columbia/simsopt-jax/tests/field/test_biotsavart_jax.py`
    - `/Users/suhjungdae/code/columbia/simsopt-jax/tests/field/test_coil.py`
    - `/Users/suhjungdae/code/columbia/simsopt-jax/tests/field/test_coilset.py`
    - `/Users/suhjungdae/code/columbia/simsopt-jax/tests/field/test_fieldline.py`
    - `/Users/suhjungdae/code/columbia/simsopt-jax/tests/field/test_magnetic_axis_helpers.py`
    - `/Users/suhjungdae/code/columbia/simsopt-jax/tests/field/test_magneticfields.py`
    - `/Users/suhjungdae/code/columbia/simsopt-jax/tests/field/test_mpi_tracing.py`
    - `/Users/suhjungdae/code/columbia/simsopt-jax/tests/field/test_particle.py`
    - `/Users/suhjungdae/code/columbia/simsopt-jax/tests/field/test_selffieldforces.py`
    - `/Users/suhjungdae/code/columbia/simsopt-jax/examples/1_Simple/qfm.py`
    - `/Users/suhjungdae/code/columbia/simsopt-jax/examples/1_Simple/tracing_fieldlines_NCSX.py`
    - `/Users/suhjungdae/code/columbia/simsopt-jax/examples/1_Simple/tracing_particle.py`
    - `/Users/suhjungdae/code/columbia/simsopt-jax/examples/2_Intermediate/boozer.py`
    - `/Users/suhjungdae/code/columbia/simsopt-jax/examples/2_Intermediate/boozerQA.py`
    - `/Users/suhjungdae/code/columbia/simsopt-jax/examples/2_Intermediate/strain_optimization.py`

- [ ] **Update `vectorize` parameter references**
  - Upstream removed `vectorize` from all BoozerSurface solver methods. Any jax-port code passing `vectorize=True` needs the argument removed.
  - Files with actual `vectorize` references:
    - `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/boozersurface.py` — 2 method signatures (`minimize_boozer_penalty_constraints_LBFGS` at line 609, `minimize_boozer_penalty_constraints_newton` at line 724) + their dispatch logic and docstrings
    - `/Users/suhjungdae/code/columbia/simsopt-jax/tests/geo/test_boozersurface.py` — 14 code sites (parametrization loops at lines 59, 74, 234; subtest signatures at lines 83, 123, 244, 394; call sites at lines 290, 301, 417, 423)

- [ ] **Verify upstream's new `BoozerSurface` features work**
  - File: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/boozersurface.py`
  - `minimize_boozer_penalty_constraints_ls(..., method='manual')` — new manual Gauss-Newton
  - `minimize_boozer_penalty_constraints_ls(..., weight_inv_modB=True)` — new parameter
  - `_get_residual_vector_and_jacobian()` — new internal method
  - These should be exercised by upstream's new tests which come in via the merge.

- [ ] **Run post-merge validation sweep**
  - After conflict resolution and API cleanup, rerun the minimal pre-merge baseline first:
    - `conda run -n columbia-jax-0.9.2 python -m pytest /Users/suhjungdae/code/columbia/simsopt-jax/tests/integration/test_single_stage_jax.py -k "TestExactSolveCPUJAXParity" -v`
    - `conda run -n columbia-jax-0.9.2 python -m pytest /Users/suhjungdae/code/columbia/simsopt-jax/tests/integration/test_stage2_jax.py -k "TestStage2BananaBoundary" -v`
    - `conda run -n columbia-jax-0.9.2 python -m pytest /Users/suhjungdae/code/columbia/simsopt-jax/tests/test_benchmark_helpers.py -k "single_stage_init_parity or stage2_e2e_comparison" -v`
  - Then run the broader suites needed to validate the merged surface:
    - Public JAX tests (no simsoptpp): `conda run -n columbia-jax-0.9.2 python -m pytest /Users/suhjungdae/code/columbia/simsopt-jax/tests/ -m "not private_optimizer_runtime" -v`
    - Private optimizer tests: `conda run -n columbia-jax-0.9.2 python -m pytest /Users/suhjungdae/code/columbia/simsopt-jax/tests/ -m "private_optimizer_runtime" -v`
    - M2 integration: `/Users/suhjungdae/code/hbt-compare/envs/candidate-fixed/bin/python -m pytest /Users/suhjungdae/code/columbia/simsopt-jax/tests/integration/ -v`
    - Benchmark regressions: `conda run -n columbia-jax-0.9.2 python -m pytest /Users/suhjungdae/code/columbia/simsopt-jax/tests/test_benchmark_helpers.py /Users/suhjungdae/code/columbia/simsopt-jax/tests/test_hf_production_gpu_proof.py -v`

- [ ] **Verify upstream's new tests pass**
  - `/Users/suhjungdae/code/columbia/simsopt-jax/tests/geo/test_boozersurface.py` — upstream added 5 new tests (manual ls, need_to_run_code, G=None exact, non-stellsym exact)
  - `/Users/suhjungdae/code/columbia/simsopt-jax/tests/geo/test_curve.py` — upstream added JaxCurve tests
  - `/Users/suhjungdae/code/columbia/simsopt-jax/tests/geo/test_curve_objectives.py` — upstream refactored
  - `/Users/suhjungdae/code/columbia/simsopt-jax/tests/util/test_coil_optimization_helper_functions.py` — entirely new (+836 lines)

- [ ] **Evaluate `JaxCurve` integration opportunity**
  - Upstream `JaxCurve`: `/Users/suhjungdae/code/opensource/simsopt/src/simsopt/geo/curve.py:469-560`
  - Jax-port reimpl: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/biotsavart_jax_backend.py:618-685` (`_coil_arrays_in_order_from_dofs`)
  - Upstream's `JaxCurve` provides `gamma_jax`, `gammadash_jax`, `dgamma_by_dcoeff_vjp_jax` via `jax.jit`/`jax.vjp`
  - After merge, evaluate whether `BiotSavartJAX` can delegate to `JaxCurve`'s cached JIT functions instead of reimplementing Fourier evaluation
  - This is an optimization opportunity, not a merge blocker

---

## Risk Assessment

| Category | Files | Effort | Risk |
|----------|-------|--------|------|
| GPU-critical (`jit.py`) | 1 | 5 min | LOW — clear resolution (keep jax-port) |
| Boozer solver (`boozersurface.py`) | 1 | 1-2 hr | MED — 11 hunks, mostly ruff formatting vs deletion |
| Surface/curve additions | 4 | 1-2 hr | MED — both sides added features, different sections |
| Import/config plumbing | 4 | 30 min | LOW — mechanical merge of import lists |
| Tests | 3 | 30 min | LOW — both sides added non-overlapping tests |
| Auto-merge verification | 6 | 30 min | LOW — spot-check only |
| Post-merge validation | — | 1-2 hr | MED — minimal baseline rerun, broader suites, and API compatibility |

**Total estimated effort: 4-8 hours**
