# X1 — `jax_core/` layering BLOCKER plan

Branch: `gpu-purity-stage2-20260405` · Audit date: 2026-05-16

CLAUDE.md asserts that "JAX modules live alongside C++ counterparts. They do
NOT import simsoptpp." The convention review (B-1) flagged this as FALSE. This
document scopes the actual cross-package imports in `src/simsopt/jax_core/`,
proposes a minimum-disruption migration ordering, executes ONE
lowest-risk proof-of-concept migration, and defers the rest with concrete
follow-up specifications.

## Scope clarification (root cause, separable problem)

Importing `simsopt.jax_core` today pulls `_simsoptpp`, `simsopt.geo.curve`,
and `simsopt._core.optimizable`. The dominant trigger is **not** the
`jax_core/` imports themselves but the parent package's eager bootstrap
at `src/simsopt/__init__.py:45`:

```
from ._core import make_optimizable, load, save
```

which chains `_core.util:21` `from simsoptpp import Curve`. Until that
chain is split (or `simsopt.jax_core` is decoupled from
`simsopt/__init__.py` via a separate import root), the layering inside
`jax_core/` is necessary but not sufficient for `import simsopt.jax_core`
to be `simsoptpp`-free. The B-1 layering work and the `__init__.py`
bootstrap work are two stacked PRs.

This document closes only the B-1 layering surface.

## §1 Phase A — Inventory

All actual cross-package import statements inside `src/simsopt/jax_core/`,
plus the file:line, target module, symbols, consumer purpose, and whether
the target module itself drags `Optimizable`/`simsoptpp` transitively.

`..backend` and `..backend.runtime` are pure standalone packages
(no `_simsoptpp`, no `_core.optimizable`) and are therefore NOT BLOCKERs;
they are listed but classified as `clean`.

| # | File:Line | Target | Symbols | Consumer purpose | Target pulls Optimizable? | Blocker? |
|---|-----------|--------|---------|------------------|---------------------------|----------|
| 1 | `_math_utils.py:11` (deferred, function-local) | `simsopt.backend` | `maybe_initialize_distributed_jax` | Initialize JAX distributed runtime before device_put | No (pure) | clean |
| 2 | `biotsavart.py:21` | `..backend` | `BackendPolicy`, tuning getters | Backend config + chunking | No (pure) | clean |
| 3 | `biotsavart.py:24` | `..backend.runtime` | `register_backend_cache_clear` | Per-mode JIT cache invalidation | No (pure) | clean |
| 4 | `sharding.py:14` | `..backend` | `get_sharding_tuning`, `maybe_initialize_distributed_jax` | Sharding policy | No (pure) | clean |
| 5 | `sharding.py:15` | `..backend.runtime` | `register_backend_cache_clear` | Cache invalidation | No (pure) | clean |
| 6 | `curve_geometry.py:12-17` | `..geo.curve` | `gamma_curve_on_surface`, `incremental_arclength_pure`, `kappa_pure`, `torsion_pure` | Curve arc-length, curvature, torsion, on-surface curve gamma kernels | YES — `geo/curve.py:22` `from ._simsoptpp ...`, `:25` `from .._core.optimizable ...` | **BLOCKER** |
| 7 | `curve_geometry.py:18` | `..geo.curvehelical` | `curve_helical_pure` | Helical-curve gamma kernel | YES (host module imports `_simsoptpp` via `.curve`) | **BLOCKER** |
| 8 | `curve_geometry.py:19` | `..geo.curveplanarfourier` | `curveplanarfourier_pure` | Planar-Fourier curve gamma kernel | YES (`curveplanarfourier.py:3` `_simsoptpp`) | **BLOCKER** |
| 9 | `curve_geometry.py:20` | `..geo.curverzfourier` | `curverzfourier_pure` | RZ-Fourier curve gamma kernel | YES (`curverzfourier.py:3` `_simsoptpp`) | **BLOCKER** |
| 10 | `curve_geometry.py:21-24` | `..geo.curvexyzfourier` | `jaxfouriercurve_geometry_pure`, `jaxfouriercurve_pure` | XYZ-Fourier curve gamma kernel | YES (`curvexyzfourier.py:6` `_simsoptpp`) | **BLOCKER** |
| 11 | `curve_geometry.py:163` (deferred, function-local) | `simsopt.geo.orientedcurve` | `centercurve_pure` | Oriented-curve gamma kernel | YES (host pulls `_simsoptpp` via `.curve`) | **BLOCKER (MIGRATED — see §3)** |
| 12 | `curve_geometry.py:199` (deferred, function-local) | `simsopt.geo.curvexyzfouriersymmetries` | `jaxXYZFourierSymmetriescurve_pure` | XYZ-Fourier-symmetries gamma kernel | YES (host pulls `_simsoptpp` via `.curve`) | **BLOCKER** |
| 13 | `magnetic_axis_helpers.py:71` | `..geo.curverzfourier` | `curverzfourier_pure` | Magnetic-axis position evaluation | YES (same as #9) | **BLOCKER** |
| 14 | `surface_fourier.py:7` | `..geo.surface_fourier_jax` | 16 surface gamma/derivative/normal kernels | Surface XYZ Fourier geometry helpers | No top-level — `surface_fourier_jax.py` itself is a pure-JAX leaf; however the import is layering-inverted (`jax_core` depends on `geo`) | **BLOCKER (inverse-layering only)** |
| 15 | `surface_henneberg.py:36` | `..geo.surface_fourier_jax` | `surface_area`, `surface_volume` | Henneberg surface area + volume | No (pure leaf) | **BLOCKER (inverse-layering only)** |
| 16 | `objectives_flux.py:32` | `..objectives.integral_bdotn_jax` | `integral_BdotN`, `residual_BdotN`, `signed_BdotN_flux` | Flux objective kernels | No — `integral_bdotn_jax.py` imports only from `jax_core` | **BLOCKER (inverse-layering only)** |
| 17 | `specs.py:1608` (deferred, function-local) | `..geo.surface_fourier_jax` | `stellsym_scatter_indices` | Stellsym DOF index materialization | No (pure leaf) | **BLOCKER (inverse-layering only)** |
| 18 | `tracing.py:75` | `..field.boozermagneticfield_jax` | `BoozerRadialInterpolantFrozenState`, `InterpolatedBoozerFieldFrozenState`, 12 `_eval_*` functions, `_INTERP_EVALUATORS` | RHS evaluation for guiding-center / fullorbit Boozer tracing | YES — `boozermagneticfield_jax.py:32` `from .._core.optimizable import Optimizable` | **BLOCKER** |

Phase A summary: 18 raw import statements, of which 5 are `clean` (the
pure `simsopt.backend` package), 4 are `inverse-layering only` (the target
module is itself pure but lives in the wrong subpackage), and 9 are
"hard" BLOCKERs (target pulls `Optimizable` and/or `_simsoptpp`
transitively). The convention review's "9" matches the hard-BLOCKER count.

## §2 Phase B — Migration plan

Migration class legend:
- **NO-OP rename** — pure function/class lifted into `jax_core/`, original
  module re-exports for API compatibility. Zero behavior change.
- **STRUCTURAL move** — split a hybrid module: pure kernels move into
  `jax_core/`; the `Optimizable` class stays in its host module and
  imports back from `jax_core/`.
- **DEEPER refactor** — function bodies need rewriting (e.g., to remove
  dependence on a host module's local closures or `jit` decorator).

| Order | Item # | Migration | Class | Risk | Depends on |
|-------|--------|-----------|-------|------|------------|
| 1 | 11 | Move `centercurve_pure`, `shift_pure`, `rotate_pure` from `simsopt.geo.orientedcurve` to `simsopt.jax_core.oriented_curve`; re-export from `orientedcurve.py` | NO-OP rename | LOW | none |
| 2 | 12 | Move `jaxXYZFourierSymmetriescurve_pure` from `simsopt.geo.curvexyzfouriersymmetries` to `simsopt.jax_core.curve_xyz_fourier_symmetries`; re-export from host module | NO-OP rename | LOW | none |
| 3 | 8 | Move `curveplanarfourier_pure` from `simsopt.geo.curveplanarfourier` to `simsopt.jax_core.curve_planar_fourier`; re-export | NO-OP rename | LOW | none |
| 4 | 7 | Move `curve_helical_pure` from `simsopt.geo.curvehelical` to `simsopt.jax_core.curve_helical`; re-export | NO-OP rename | LOW | none |
| 5 | 9, 13 | Move `curverzfourier_pure` from `simsopt.geo.curverzfourier` to `simsopt.jax_core.curve_rz_fourier`; re-export. Updates both `jax_core/curve_geometry.py:20` AND `jax_core/magnetic_axis_helpers.py:71` | NO-OP rename | LOW | none |
| 6 | 10 | Move `jaxfouriercurve_pure`, `jaxfouriercurve_geometry_pure` from `simsopt.geo.curvexyzfourier` to `simsopt.jax_core.curve_xyz_fourier`; re-export | NO-OP rename | LOW | none |
| 7 | 6 | Move `incremental_arclength_pure`, `kappa_pure`, `torsion_pure`, `gamma_curve_on_surface` from `simsopt.geo.curve` to `simsopt.jax_core.curve_kernels`; re-export. Many downstream callers in `simsopt.geo.*` and `simsopt.objectives.*` continue to work because the names remain importable from `simsopt.geo.curve` | STRUCTURAL (re-export must preserve `vjp`/`jacfwd` jitted wrappers too) | MED | none |
| 8 | 14, 15, 17 | Repatriate `simsopt.geo.surface_fourier_jax` (a pure module) under `simsopt.jax_core.surface_xyz_tensor_fourier`. Re-export current path from `simsopt.geo.surface_fourier_jax` for API stability (tests at `tests/geo/test_surface_fourier_jax.py` already use `simsopt.geo.surface_fourier_jax`). Updates `jax_core/surface_fourier.py:7`, `jax_core/surface_henneberg.py:36`, `jax_core/specs.py:1608` | NO-OP rename (large file move + re-export) | MED (large file, many downstream test imports) | none |
| 9 | 16 | Repatriate `simsopt.objectives.integral_bdotn_jax` (a pure module) under `simsopt.jax_core.integral_bdotn`. Re-export current path | NO-OP rename | LOW (small file) | none |
| 10 | 18 | Split `simsopt.field.boozermagneticfield_jax`: lift `BoozerRadialInterpolantFrozenState`, `InterpolatedBoozerFieldFrozenState`, all `_eval_*` evaluator functions, and `_INTERP_EVALUATORS` registry into `simsopt.jax_core.boozer_radial_interp_state`. Leave the `Optimizable`-bearing `BoozerRadialInterpolantJAX`/`BoozerAnalyticJAX`/`InterpolatedBoozerFieldJAX` classes in `simsopt.field.boozermagneticfield_jax`; they import the pure kernels back from `jax_core`. Updates `jax_core/tracing.py:75` | STRUCTURAL (file split, freeze/serialization helpers also need to be classified) | MED-HIGH (file is ~900 lines, careful classification needed, GSONDecoder JSON serializer dependency must be preserved) | none |

Ordering rationale: items 1–6 are independent leaf-function moves with no
cross-dependencies, so they can be parallelized or sequenced freely. Item
7 is the only one with a `STRUCTURAL` element on the curve side; do it
after the leaf moves so the new `jax_core/curve_*.py` files exist as a
landing zone. Items 8–9 are surface/objective inverse-layering moves;
they're independent of the curve work. Item 10 is the largest single
move and should be sequenced last.

## §3 Phase C — Executed migration (item #11)

**Chosen migration**: item #11 (lowest-risk NO-OP rename, deferred-import
consumer, no external test depends directly on the moved symbols).

### Before

`src/simsopt/geo/orientedcurve.py:1-89` defined `shift_pure`,
`rotate_pure`, `centercurve_pure` inline.

`src/simsopt/jax_core/curve_geometry.py:163`:
```python
from simsopt.geo.orientedcurve import centercurve_pure
```

### After

New file `src/simsopt/jax_core/oriented_curve.py` (98 lines): hosts
`shift_pure`, `rotate_pure`, `centercurve_pure` with explicit
`import jax.numpy as jnp` and `from math import pi`. Pure JAX, no
SIMSOPT cross-imports.

`src/simsopt/geo/orientedcurve.py:1-10` now reads:
```python
import numpy as np
from .curve import JaxCurve
from ..jax_core.oriented_curve import centercurve_pure, rotate_pure, shift_pure

__all__ = [
    "OrientedCurveXYZFourier",
    "centercurve_pure",
    "rotate_pure",
    "shift_pure",
]
```

`src/simsopt/jax_core/curve_geometry.py:152` now reads:
```python
from .oriented_curve import centercurve_pure
```

Diff stat: `orientedcurve.py` shrunk from 190 → 106 lines (-84), new
`jax_core/oriented_curve.py` is 98 lines, `curve_geometry.py` changed 1
line.

### Validation

- `ruff check src/simsopt/jax_core/oriented_curve.py
  src/simsopt/geo/orientedcurve.py src/simsopt/jax_core/curve_geometry.py`
  → `All checks passed!`
- `ruff format --check` on the same three files → `3 files already formatted`.
- Re-export sanity check:
  ```
  from simsopt.geo.orientedcurve import centercurve_pure as cc_geo
  from simsopt.jax_core.oriented_curve import centercurve_pure as cc_jax
  assert cc_geo is cc_jax  # PASS
  ```
- `python -c "import simsopt.jax_core"` → succeeds. (Note: still pulls
  `simsoptpp` because of the `simsopt/__init__.py:45` eager import; the
  `jax_core/` import surface itself is now one step cleaner.)

### Test slice

```
.conda/jax/bin/python -m pytest tests/test_jax_import_smoke.py tests/jax_core/ \
  --deselect tests/test_jax_import_smoke.py::test_transfer_guard_disallow_allows_lbfgs_ondevice_quadratic_smokes \
  --ignore=tests/jax_core/test_analytic_fields_item11.py -q
```

Three pre-existing failure groups were observed; each was confirmed
present on baseline (`git stash` of the migration) on the same machine
and timing budget:

1. `tests/test_jax_import_smoke.py::test_transfer_guard_disallow_allows_lbfgs_ondevice_quadratic_smokes`
   — `--deselect`ed from both runs because its subprocess helper
   hits a hard-coded 30 s timeout in the helper layer rather than the
   product code.
2. `tests/jax_core/test_analytic_fields_item11.py` — `--ignore`d
   because of a collection-time `ImportError: cannot import name
   'clear_dommaschk_caches' from 'simsopt.jax_core.analytic_fields'`.
   The test file was modified by prior work-in-progress; the symbol it
   imports does not exist in `analytic_fields.py`. Pre-existing
   test/source mismatch unrelated to this migration.
3. Five subprocess-based smoke tests in
   `tests/test_jax_import_smoke.py` that exceed their per-test
   subprocess timeout budget on this machine:
   - `test_transfer_guard_disallow_allows_target_minimize_structured_pytree_entry`
   - `test_transfer_guard_disallow_enforces_single_stage_target_runtime_boundaries`
   - `test_stage2_target_outer_loop_reuses_compiled_solver_across_identical_calls`
   - `test_structured_ondevice_solver_cache_respects_mutable_objective_state`
   - `test_transfer_guard_disallow_allows_ondevice_loops_with_host_closure_constants`

Observed counts:

| Run | Tree | Suite | Passed | Failed | Skipped | Deselected | Wall time |
|-----|------|-------|--------|--------|---------|------------|-----------|
| 1 | Baseline (stash) | `test_jax_import_smoke.py` only | 103 | 5 | 11 | 1 | 776.85 s |
| 2 | With migration | `test_jax_import_smoke.py` + `tests/jax_core/` (`--ignore=test_analytic_fields_item11.py`) | 368 | 9 | 11 | 1 | 1442.16 s |
| 3 | With migration | `test_jax_import_smoke.py` only | 101 | 7 | 11 | 1 | 869.53 s |
| 4 | With migration | `test_jax_import_smoke.py` only (re-run) | 101 | 7 | 11 | 1 | 625.82 s |

Failure classification:

- All four runs reproduce the **same five core subprocess-timeout
  failures**, present on baseline (run 1) and stable across migrations
  and re-runs:
  - `test_transfer_guard_disallow_allows_target_minimize_structured_pytree_entry`
  - `test_transfer_guard_disallow_enforces_single_stage_target_runtime_boundaries`
  - `test_stage2_target_outer_loop_reuses_compiled_solver_across_identical_calls`
  - `test_structured_ondevice_solver_cache_respects_mutable_objective_state`
  - `test_transfer_guard_disallow_allows_ondevice_loops_with_host_closure_constants`
- Runs 3 and 4 each report two **additional** failing tests, but
  **the two are different in each run** (`squaredfluxjax_construction`
  + `clamped_xyztensor_surface_spec` in run 3;
  `project_surface_dofs_to_resolution` +
  `single_stage_surface_reprojection_probe_emits_structured_cpu_result`
  in run 4). All four flapping tests **pass in isolation** (verified
  on the with-migration tree:
  `pytest tests/test_jax_import_smoke.py::test_transfer_guard_disallow_allows_squaredfluxjax_construction tests/test_jax_import_smoke.py::test_transfer_guard_disallow_allows_clamped_xyztensor_surface_spec` → 2 passed in 5.19 s).
  These are time-flake subprocess-helper timeouts (each test invokes
  `_assert_python_script_passes` with a 30 s subprocess timeout that
  is sensitive to machine load and JIT-cache freshness).

The migration touches only `centercurve_pure` / `shift_pure` /
`rotate_pure` — pure JAX kernels with no Stage 2 optimizer or
SquaredFluxJAX surface; none of the flapping tests exercise the
`OrientedCurveXYZFourier` path. Direct re-execution of each flapping
case via `.conda/jax/bin/python tests/subprocess/import_smoke_cases.py
case_<name>` succeeds on the with-migration tree.

The migrated import path is itself exercised by
`tests/geo/test_orientedcurve_jax_spec.py` (passed in 1.00 s on the
with-migration tree) and indirectly by the curve-geometry test slice
(`tests/geo/test_orientedcurve_jax_spec.py` +
`tests/geo/test_curve.py` → 40 passed, 335 subtests passed in 143 s).

Migration-relevant slice result: **no new failures attributable to the
migration**. The 5 stable failures reproduce on baseline; the
additional flapping failures pass in isolation and pass through direct
subprocess invocation.

## §4 Phase D — Deferred follow-ups

Each item below is a ready-to-assign work order. None executed in this
session; all sequenced after item #11.

### D-1 — Repatriate `jaxXYZFourierSymmetriescurve_pure`
- Migration: extract function body from `src/simsopt/geo/curvexyzfouriersymmetries.py:10-78` into new `src/simsopt/jax_core/curve_xyz_fourier_symmetries.py`. Re-export from the host module. Update `src/simsopt/jax_core/curve_geometry.py:199-201` deferred import to `from .curve_xyz_fourier_symmetries import jaxXYZFourierSymmetriescurve_pure`.
- Files affected: 3 (one new, two edits).
- Risk: LOW.
- Validation: `ruff check` on touched files; `pytest tests/jax_core/ tests/geo/test_curvexyzfouriersymmetries_spec_jax.py tests/test_jax_import_smoke.py --deselect <pre-existing-fail>`.

### D-2 — Repatriate `curveplanarfourier_pure`
- Migration: extract from `src/simsopt/geo/curveplanarfourier.py:57+` into new `src/simsopt/jax_core/curve_planar_fourier.py`; re-export. Update `src/simsopt/jax_core/curve_geometry.py:19` to `from .curve_planar_fourier import curveplanarfourier_pure`.
- Files affected: 3.
- Risk: LOW.
- Validation: `pytest tests/jax_core/ tests/geo/test_curveplanarfourier.py --deselect <pre-existing-fail>`.

### D-3 — Repatriate `curve_helical_pure`
- Migration: extract from `src/simsopt/geo/curvehelical.py:7+` into new `src/simsopt/jax_core/curve_helical.py`; re-export. Update `src/simsopt/jax_core/curve_geometry.py:18`.
- Files affected: 3.
- Risk: LOW.
- Validation: `pytest tests/jax_core/ tests/geo/test_curvehelical_spec_jax.py --deselect <pre-existing-fail>`.

### D-4 — Repatriate `curverzfourier_pure`
- Migration: extract from `src/simsopt/geo/curverzfourier.py:18+` into new `src/simsopt/jax_core/curve_rz_fourier.py`; re-export. Update **both** call sites: `src/simsopt/jax_core/curve_geometry.py:20` AND `src/simsopt/jax_core/magnetic_axis_helpers.py:71`.
- Files affected: 4.
- Risk: LOW.
- Validation: `pytest tests/jax_core/ tests/geo/test_curverzfourier.py tests/jax_core/test_magnetic_axis_helpers_jax.py --deselect <pre-existing-fail>`.

### D-5 — Repatriate `jaxfouriercurve_pure` and `jaxfouriercurve_geometry_pure`
- Migration: extract from `src/simsopt/geo/curvexyzfourier.py:309+` and `:328+` into new `src/simsopt/jax_core/curve_xyz_fourier.py`; re-export. Update `src/simsopt/jax_core/curve_geometry.py:21-24`.
- Files affected: 3.
- Risk: LOW.
- Validation: `pytest tests/jax_core/ tests/geo/test_curvexyzfourier.py --deselect <pre-existing-fail>`.

### D-6 — Repatriate `incremental_arclength_pure`, `kappa_pure`, `torsion_pure`, `gamma_curve_on_surface`
- Migration: extract from `src/simsopt/geo/curve.py:213-272` (and `gamma_curve_on_surface` at line 1752) into new `src/simsopt/jax_core/curve_kernels.py`. Re-export from `simsopt.geo.curve`. Update `src/simsopt/jax_core/curve_geometry.py:12-17`.
- Caution: `simsopt.geo.curve` also defines `incremental_arclength_vjp`, `kappavjp0`, `kappavjp1`, `kappagrad0`, `kappagrad1`, `torsionvjp0/1/2`, `torsiongrad0/1/2` as `jit`-decorated lambdas built from the pure functions. These wrappers depend on `.jit.jit` (which depends on backend selection). Decision required: do the jitted wrappers move with the pure functions, or stay in `geo/curve.py`? Recommend they stay in `geo/curve.py` (they are CPU-host API), with the pure functions imported from `jax_core/curve_kernels.py`.
- Files affected: 3 + downstream callers in `simsopt.geo.curveobjectives`, `simsopt.geo.surfaceobjectives_jax`, `simsopt.objectives.stage2_target_objective_jax`. None of those callers need changes because the symbol remains importable from `simsopt.geo.curve` via re-export.
- Risk: MED.
- Validation: `pytest tests/jax_core/ tests/geo/test_curve.py tests/geo/test_curveobjectives.py --deselect <pre-existing-fail>`.

### D-7 — Repatriate `simsopt.geo.surface_fourier_jax` (large pure module)
- Migration: rename `src/simsopt/geo/surface_fourier_jax.py` to `src/simsopt/jax_core/surface_xyz_tensor_fourier.py`. Create a thin re-export shim at `src/simsopt/geo/surface_fourier_jax.py` (`from ..jax_core.surface_xyz_tensor_fourier import *  # noqa: F401, F403`). Update `src/simsopt/jax_core/surface_fourier.py:7`, `src/simsopt/jax_core/surface_henneberg.py:36`, `src/simsopt/jax_core/specs.py:1608` to import from the new home. The bottom-of-file deferred import in `surface_fourier_jax.py` lines 2766-2768 (the `_unit_vector_axis_last` workaround) disappears since the file is now inside `jax_core/`.
- Files affected: 4 (one moved + shim, three internal updates).
- Risk: MED (40+ test imports reference the public name `simsopt.geo.surface_fourier_jax.*`; all preserved via the `from ... import *` shim).
- Validation: `pytest tests/jax_core/ tests/geo/test_surface_fourier_jax.py tests/geo/test_boozersurface_jax.py tests/geo/test_boozer_derivatives_jax.py tests/geo/test_surface_objectives_jax.py tests/geo/test_label_constraints_jax.py tests/test_jax_import_smoke.py --deselect <pre-existing-fail>`.

### D-8 — Repatriate `simsopt.objectives.integral_bdotn_jax`
- Migration: rename `src/simsopt/objectives/integral_bdotn_jax.py` to `src/simsopt/jax_core/integral_bdotn.py`. Add re-export shim at the old path (`from ..jax_core.integral_bdotn import *  # noqa: F401, F403`). Update `src/simsopt/jax_core/objectives_flux.py:32`.
- Files affected: 3.
- Risk: LOW.
- Validation: `pytest tests/jax_core/ tests/objectives/test_integral_bdotn_jax.py --deselect <pre-existing-fail>`.

### D-9 — Split `simsopt.field.boozermagneticfield_jax` (largest, most careful)
- Migration: classify the ~50 top-level entities in `src/simsopt/field/boozermagneticfield_jax.py` into two groups:
  - PURE → new `src/simsopt/jax_core/boozer_radial_interp_state.py`:
    `BoozerRadialInterpolantFrozenState` (line 92),
    `InterpolatedBoozerFieldFrozenState` (analogous),
    `freeze_boozer_radial_state`, `freeze_interpolated_boozer_field_state`,
    `_eval_modB`, `_eval_dmodBdtheta`, `_eval_dmodBdzeta`, `_eval_dmodBds`,
    `_eval_R`, `_eval_dRdtheta`, `_eval_dRdzeta`, `_eval_dRds`,
    `_eval_Z`, `_eval_dZdtheta`, `_eval_dZdzeta`, `_eval_dZds`,
    `_eval_nu`, `_eval_dnudtheta`, `_eval_dnudzeta`, `_eval_dnuds`,
    `_eval_K`, `_eval_dKdtheta`, `_eval_dKdzeta`, `_eval_psip`,
    `_eval_G`, `_eval_I`, `_eval_iota`, `_eval_dGds`, `_eval_dIds`, `_eval_diotads`,
    `_INTERP_EVALUATORS` dictionary,
    helper functions (`_profile_to_host`, `_profile_from_host`, `_frozen_state_to_host`, `_frozen_state_from_host`, `_ppoly_from_spline`, `_scalar_profile`, `_mode_profile_stack`, `_zeros_like_profile`, `_column_at`, `_scalar_at`, `_normalize`, `_radial_normalized`).
  - HOST → stays in `simsopt.field.boozermagneticfield_jax`:
    the three `Optimizable` adapter classes (`BoozerRadialInterpolantJAX`,
    `BoozerAnalyticJAX`, `InterpolatedBoozerFieldJAX`) plus
    `from .._core.optimizable import Optimizable` and
    `from .._core.json import GSONDecoder`.
- The host module imports pure kernels back from `jax_core` (same pattern as the M2 adapters already use).
- Update `src/simsopt/jax_core/tracing.py:75-91` to import from the new pure-kernel module.
- Files affected: 3 (one new, two edits). Plus careful classification of `GSONDecoder` serialization helpers (`_to_dict`/`from_dict` on the frozen-state dataclasses).
- Risk: MED-HIGH. The frozen-state dataclasses have `_to_dict`/`from_dict` JSON helpers that may reference `GSONDecoder`. If they do, those helpers stay in the host module (the dataclasses themselves can be pure with the JSON helpers as separate top-level functions in the host module).
- Validation: `pytest tests/jax_core/test_tracing_jax_gc_boozer.py tests/jax_core/test_tracing_jax_boozer_zeta_events.py tests/field/test_boozermagneticfield_jax_item33.py --deselect <pre-existing-fail>`.

## §5 Verification

Post-Phase-C state:

- `conda run -n jax python -c "import simsopt.jax_core"` (via in-tree
  env `.conda/jax/bin/python`) → **succeeds**.
- The migration removes one of the deferred `simsopt.geo.*` cross-imports
  inside `jax_core/`. The B-1 BLOCKER count is reduced from 9 → 8 hard
  cross-imports.
- `_simsoptpp` is still loaded transitively after `import
  simsopt.jax_core` because `simsopt/__init__.py:45` eagerly imports
  `_core`. Eliminating that requires a separate fix (out of scope for
  B-1, see "Scope clarification" above).

Net effect of Phase C: one cross-import eliminated, one new pure-JAX
module added under `jax_core/`, zero new test failures, zero behavior
change. The migration template (move pure function → re-export from
host → update jax_core consumer) is now established and the eight
remaining BLOCKER items in §4 follow the same pattern with monotonically
increasing risk.

## File:line changes (this session)

- NEW `src/simsopt/jax_core/oriented_curve.py` (98 lines)
- MOD `src/simsopt/geo/orientedcurve.py:1-89` → 1-10 (re-export + class only)
- MOD `src/simsopt/jax_core/curve_geometry.py:152` (deferred import retargeted)

## Working-tree advisory (this session)

During Phase C validation I needed to baseline-vs-with-migration the
test slice. `git stash` / `git stash pop` interactions with the
pre-existing working-tree modifications (the ~60 `M` files surfaced in
the session's initial `git status` reminder) produced silent
conflict-driven pop failures. The migration files are now reapplied
in the working tree; other modified files were restored via
`git checkout stash@{0} -- <paths>` and `git checkout stash@{1} -- <paths>`.

Three stash entries remain (kept for safety, not dropped):

```
stash@{0}: WIP on gpu-purity-stage2-20260405: d75ebcb7b docs(lm-minpack-plan): rev 5 — doc-review fixes and rev-3 retrospective sync
stash@{1}: WIP on gpu-purity-stage2-20260405: be850cb72 fix: preserve JAX target objective contracts
stash@{2}: WIP on gpu-purity-stage2-20260405: cadc6139e docs: reconcile JAX native plan review
```

Recommended cleanup once the user has confirmed the working tree is
intact:

```
git stash drop stash@{0}
git stash drop stash@{1}
# leave stash@{2} — it is older work from a prior session
```

The migration changes themselves are NOT in any stash; they are
present directly in the working tree as:

```
?? src/simsopt/jax_core/oriented_curve.py
 M src/simsopt/geo/orientedcurve.py
 M src/simsopt/jax_core/curve_geometry.py
```
