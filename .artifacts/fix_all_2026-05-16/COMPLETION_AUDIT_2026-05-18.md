# Completion Audit - non-CUDA fix-all closeout

Date: 2026-05-18
Worktree: `/Users/suhjungdae/code/columbia/simsopt-jax`
HEAD: `0b2a69bf6 fix: align lbfgs-ondevice fullgraph parity`

This audit maps the original non-CUDA TODO prompt to current artifacts and
separates completed implementation work from workflow gates that still require
owner action or a post-commit rerun.

## Verdict

The goal is not fully complete.

Open non-CUDA gates:

1. Clean post-commit CPU parity matrix.
2. Stash disposition for `stash@{0..2}`.
3. Scoped commit/tag of the intended source plus report slice.

All other listed non-CUDA code-port and validation items are closed, superseded,
or future/out-of-release-scope according to the live `STATUS.md` refresh and the
current validation evidence below.

## Prompt-to-artifact checklist

| # | Requirement | Status | Evidence |
|---:|---|---|---|
| 1 | `jax_core/` layering blocker and eager `simsoptpp` cleanup | Closed for current tree | `STATUS.md` says no forbidden `jax_core` -> `geo`/`field`/`objectives` imports remain and root `_core`/`simsoptpp` bootstrap is lazy. Current diff includes `src/simsopt/__init__.py` lazy exports. |
| 2 | CLAUDE.md stale validation/docs | Closed | `CLAUDE.md` now uses `.conda/jax/bin/python`, refreshed private/public suite paths, current M5 wording, `res["primal_success"]`, and backend mode docs. |
| 3 | `integral_bdotn_jax.py` double-where NaN-gradient issue | Closed | Implementation moved to `src/simsopt/jax_core/integral_bdotn.py`; `tests/objectives/test_integral_bdotn_jax.py` adds finite-gradient and inactive-input tests. Public pure-JAX CPU suite passed with this file included. |
| 4 | BiotSavartJAX uniform `CurveXYZFourier` fast path decision | Closed | Current diff wires `_coil_set_spec_from_explicit_state` through `grouped_coil_set_spec_from_lists` for the uniform fast path and adds `test_coil_set_spec_uses_uniform_curve_xyz_fourier_fastpath`. |
| 5 | Cached-pyc `TypeError` in compile diagnostics test | Closed as non-reproduced and de-risked | `tests/test_jax_compile_diagnostics.py` no longer imports the heavy integration subprocess helper through `sys.path`; it directly tests the recorder/parser and wiring. |
| 6 | `lbfgs_ondevice_quadratic_smokes` subprocess timeout | Closed for non-CUDA local gate | Full integration suite passed. The long on-device single-stage subprocess completed in `tests/integration/` rather than timing out. |
| 7 | Inspect/drop the 3 stashes | Partially complete, owner-blocked | Stashes inspected; none has untracked payload. They contain tracked paths outside current diff and must be kept until owner approves drop or replay. |
| 8 | Run downstream `tests/integration/test_single_stage_jax.py` | Closed | `tests/integration/test_single_stage_jax.py`: 7 passed. |
| 9 | Run full non-CUDA validation suite per corrected CLAUDE.md | Closed for dirty tree | Public pure-JAX CPU: 860 passed, 114 skipped. Private optimizer runtime: 50 passed, 224 deselected. Benchmark/runtime helpers: 270 passed, 2 skipped. Full integration: 450 passed, 6 skipped. |
| 10 | Tag/commit fix-all source plus report slice | Open | No files staged; no tag points at HEAD. Working tree is too broad for blind staging. |
| 11 | Optional Track 1 LM publication write-up decision | Not required for closeout | Optional item; no release-blocking status. |
| 12 | `pm_optimization.py` double-where cleanup | Superseded/out of current release gate | `STATUS.md` medium backlog says no current medium/cleanup item remains in consolidated status. |
| 13 | `force.py` `solve_triangular` identity RHS and `static_argnums` migration | Superseded/out of current release gate | Same as #12. |
| 14 | `jax.tree_util.tree_*` to `jax.tree.*` migration | Partially included where touched, not a release blocker | Current diffs include selected migrations in Biot-Savart paths; broad tree-wide migration is not a current release gate. |
| 15 | `CircularCoil._A_impl` upstream physics review | Future/upstream review | `STATUS.md` treats it as upstream physics review, not JAX-port release scope. |
| 16 | Stale/unused cleanup | Superseded/out of current release gate | `STATUS.md` medium backlog says no current medium/cleanup item remains; stale cleanup rows are historical or future cleanup. |
| 17 | Remaining `STATUS.md` M-items except M-17 | Closed/superseded | `STATUS.md` says historical M rows are closed above, superseded by live validation, or outside JAX-port release scope. |
| 18 | S-1 unported `simsoptpp` symbols | Closed for release scope | Live S-1/S-2/S-3 re-audit found no release-scope port-completeness residual. |
| 19 | S-2 partial/missing API rows | Closed for release scope | Same as #18; remaining pure immutable helpers would be future API expansion. |
| 20 | S-3 classification/design questions | Closed for release scope | Same as #18; boundaries documented as CPU mutation, cache-oracle, non-implemented oracle, or future API. |

## Validation evidence

Commands were run with:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu
```

Recorded results:

| Gate | Result |
|---|---|
| Public pure-JAX CPU suite | 860 passed, 114 skipped |
| Private optimizer runtime suite | 50 passed, 224 deselected |
| Benchmark/runtime helper suite | 270 passed, 2 skipped |
| Full `tests/integration/` suite | 450 passed, 6 skipped |
| `tests/integration/test_single_stage_jax.py` | 7 passed |
| `tests/geo/test_curve_item05_closeout.py` | 41 passed |
| Focused curve LS CPU-boundary test | 4 passed |
| Focused source-scope closure suite | 14 passed, 6 skipped |
| Lint-module ruff check/format | passed; 4 files already formatted |

## Commit-scope status

Clear report slice:

- `STATUS.md`
- `.artifacts/fix_all_2026-05-16/*`
- `.artifacts/jax_convention_review_2026-05-16/*`

Source slice is not yet safe to stage blindly. The initially obvious seven files
depend on untracked companion modules, for example:

- `src/simsopt/objectives/integral_bdotn_jax.py` imports untracked
  `src/simsopt/jax_core/integral_bdotn.py`.
- `src/simsopt/field/biotsavart_jax_backend.py` imports untracked
  `src/simsopt/jax_core/curve_xyz_fourier.py`.
- `tests/geo/test_curve_item05_closeout.py` imports new curve wrapper exports
  routed through modified `src/simsopt/jax_core/__init__.py` and untracked
  curve-core modules.

Therefore a correct source commit must include the companion JAX-core files, or
be split after an explicit owner-approved scope decision.

Currently justified companion candidate set:

- `CLAUDE.md`
- `src/simsopt/__init__.py`
- `src/simsopt/field/biotsavart_jax_backend.py`
- `src/simsopt/jax_core/`
- `src/simsopt/_maintenance/`
- `src/simsopt/objectives/integral_bdotn_jax.py`
- `scripts/jax_where_division_lint.py`
- `tests/test_jax_where_division_lint.py`
- `tests/field/test_biotsavart_jax.py`
- `tests/field/test_biotsavart_jax_parity.py`
- `tests/geo/test_curve_item05_closeout.py`
- `tests/objectives/test_integral_bdotn_jax.py`
- `tests/test_jax_compile_diagnostics.py`

## Stash status

The three stashes remain present:

| Stash | Base | Disposition |
|---|---|---|
| `stash@{0}` | `d75ebcb7b` | Keep until owner explicitly approves dropping or replaying |
| `stash@{1}` | `be850cb72` | Keep until owner explicitly approves dropping or replaying |
| `stash@{2}` | `cadc6139e` | Keep until owner explicitly approves dropping or replaying |

## Required next actions

1. Owner chooses whether to drop or replay `stash@{0..2}`.
2. Owner confirms source commit scope, or the source slice is further reduced
   until every staged file has a proven companion set.
3. Commit/tag the approved source plus report slice.
4. Rerun or explicitly re-verify the 27-fixture CPU parity matrix against the
   clean commit/tag.
