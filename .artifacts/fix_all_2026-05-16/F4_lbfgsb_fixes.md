# F4 — L-BFGS-B private-port correctness fixes (2026-05-16)

Two surgical fixes to `src/simsopt/geo/optimizer_jax_private/_lbfgsb_scipy.py`
backed by the SciPy 1.17.1 oracle (`__lbfgsb.c`, fetched from
`https://raw.githubusercontent.com/scipy/scipy/v1.17.1/scipy/optimize/__lbfgsb.c`).

## Fix 1 — H-1: RESTART task path in 5 SciPy failure modes

### SciPy upstream reference

`__lbfgsb.c` (SciPy v1.17.1) writes the `RESTART` task and refreshes
the L-BFGS memory (`col=0, head=0, theta=1.0, iupdat=0, updatd=0`)
inside `mainlb` for five failure modes:

| # | Trigger | SciPy line | Notes |
|---|---|---|---|
| 1 | `cauchy(...)` returns `info != 0` (singular triangular) | `__lbfgsb.c:822-832` | `goto LINE222` |
| 2 | `formk(...)` returns `info != 0` (non-positive Cholesky) | `__lbfgsb.c:863-874` | `goto LINE222` |
| 3 | `cmprlb`/`subsm` `info != 0` (singular triangular) | `__lbfgsb.c:879, 885-897` | `goto LINE222` |
| 4 | `formt(...)` returns `info != 0` (non-positive Cholesky) | `__lbfgsb.c:1030-1041` | `goto LINE222` |
| 5 | `lnsrlb` failure with `col != 0` | `__lbfgsb.c:918, 939-951` | Explicit `*task = RESTART; *task_msg = NO_MSG;` then `goto LINE222` |

### Pre-existing coverage (commit `abbeb922b`)

The four post-line-search RESTART paths — cases 1, 2, 3, and 5 — were
already implemented before this fix in
`_lbfgsb_setulb_fg_start_line_search`
(`_lbfgsb_scipy.py:616-618, 719-732`) and
`_lbfgsb_setulb_subspace_line_search`
(`_lbfgsb_scipy.py:868-870, 957-970`), and in
`_lbfgsb_setulb_line_search_continue` for case 5
(`_lbfgsb_scipy.py:1103-1117`).  All four paths route through
`_lbfgsb_setulb_refreshed_memory_state` (`_lbfgsb_scipy.py:1181-1209`),
which performs the exact SciPy state reset (`isave[26]=0` head,
`isave[27]=0` col, `isave[30]=0` iupdat, `isave[34]=0` info,
`dsave[0]=1.0` theta, `lsave[3]=0` updatd) and writes
`task=RESTART, task_msg=NO_MSG`.  The outer `lbfgsb_setulb` while-loop
(`_lbfgsb_scipy.py:1534-1570`) consumes the RESTART signal and
re-enters `_lbfgsb_setulb_fg_start_line_search`, mirroring SciPy's
`goto LINE222`.

### Remaining gap before this fix

The `formt` failure (case 4) in
`_lbfgsb_setulb_new_x_next_iteration → update_branch`
(`_lbfgsb_scipy.py:1395-1428` pre-fix) refreshed memory state via
`jnp.where(refresh, ...)` but did **not** write
`task=RESTART, task_msg=NO_MSG`.  Memory was reset (col, head, theta,
iupdat, updatd) but the SciPy task signal was omitted.  Per the
`.artifacts/lbfgsb_parity_review_2026-05-16/00_SYNTHESIS.md` row 4 and
`.artifacts/jax_convention_review_2026-05-16/06_review_geo_big.md`
Finding #1.

### Applied edit (case 4)

`src/simsopt/geo/optimizer_jax_private/_lbfgsb_scipy.py:1395-1446`
(within `update_branch` inside `_lbfgsb_setulb_new_x_next_iteration`).

Before:

```python
form = lbfgsb_formt(wt, update.sy, update.ss, update.col, update.theta)
refresh = form.info != 0
next_col = jnp.where(refresh, jnp.asarray(0, dtype=jnp.int32), update.col)
next_head = jnp.where(refresh, jnp.asarray(0, dtype=jnp.int32), update.head)
next_theta = jnp.where(refresh, 1.0, update.theta)
next_iupdat = jnp.where(refresh, jnp.asarray(0, dtype=jnp.int32), iupdat)
next_updatd = ~refresh

next_wa = wa
...
workspace = state.workspace._replace(
    wa=next_wa,
    lsave=state.workspace.lsave.at[3].set(next_updatd.astype(jnp.int32)),
    isave=next_isave,
    dsave=next_dsave,
)
```

After:

```python
form = lbfgsb_formt(wt, update.sy, update.ss, update.col, update.theta)
refresh = form.info != 0
next_col = jnp.where(refresh, jnp.asarray(0, dtype=jnp.int32), update.col)
next_head = jnp.where(refresh, jnp.asarray(0, dtype=jnp.int32), update.head)
next_theta = jnp.where(refresh, 1.0, update.theta)
next_iupdat = jnp.where(refresh, jnp.asarray(0, dtype=jnp.int32), iupdat)
next_updatd = ~refresh
# SciPy 1.17.1 __lbfgsb.c:1030-1041 refreshes the L-BFGS memory and
# writes ``*task = RESTART; *task_msg = NO_MSG;`` before re-entering
# LINE222 when ``formt`` reports non-positive-definite Cholesky.
# Mirror that explicit task write so the post-refresh re-entry sees
# the same task signal that SciPy uses for case 5 (lnsrlb fail with
# ``col != 0``), which already routes through the same RESTART
# boundary in ``_lbfgsb_setulb_refreshed_memory_state``.
next_task_code = jnp.where(
    refresh,
    jnp.asarray(RESTART, dtype=jnp.int32),
    state.workspace.task[0],
)
next_task_msg = jnp.where(
    refresh,
    jnp.asarray(NO_MSG, dtype=jnp.int32),
    state.workspace.task[1],
)

next_wa = wa
...
workspace = state.workspace._replace(
    wa=next_wa,
    task=_lbfgsb_task(next_task_code, next_task_msg),
    lsave=state.workspace.lsave.at[3].set(next_updatd.astype(jnp.int32)),
    isave=next_isave,
    dsave=next_dsave,
)
```

### Path map after this fix

| # | SciPy trigger | JAX kernel | RESTART writer | State reset writer |
|---|---|---|---|---|
| 1 | cauchy info!=0 | `_lbfgsb_setulb_fg_start_line_search` (line 616-618, 727-732) | `_lbfgsb_setulb_refreshed_memory_state:1204` | `_lbfgsb_setulb_refreshed_memory_state:1196-1208` |
| 2 | formk info!=0 | `_lbfgsb_setulb_fg_start_line_search` + `_lbfgsb_setulb_subspace_line_search` (line 868-870, 965-970) | same | same |
| 3 | cmprlb/subsm info!=0 | same as #2 | same | same |
| 4 | formt info!=0 | `_lbfgsb_setulb_new_x_next_iteration → update_branch` (line 1395-1446) | `update_branch` (lines 1409-1418, 1441) | `update_branch` (lines 1397-1401, 1430-1437, 1442) |
| 5 | lnsrlb fail with col!=0 | all three line-search functions (lines 719, 957, 1104) | `_lbfgsb_setulb_restart_after_line_search:1260` via `_lbfgsb_setulb_refreshed_memory_state` | same |

### Constraints honoured

- No new constants introduced (`RESTART`/`NO_MSG` are the module-level
  scalars at `_lbfgsb_scipy.py:16, 14`).
- No Python `if`; the new task write uses `jnp.where` on the same
  `refresh = form.info != 0` predicate that already gates the memory
  refresh.
- No `try`/`except`, no defensive checks, no new helper.
- The fix sits **inside** the existing `update_branch` `jax.lax.cond`
  branch, so JAX still emits a single fused HLO for the L-BFGS update
  step and the per-iteration trace cost is unchanged.

## Fix 2 — H-2: vectorise `_lbfgsb_ddot`

`src/simsopt/geo/optimizer_jax_private/_lbfgsb_scipy.py:340-347`.

### Issue

The previous body was a Python-unrolled `for i in range(int(x.shape[0]))`
loop with a `jax.lax.cond` per element to skip exact-zero products.
Trace cost was O(n) HLO ops per call site; `_lbfgsb_ddot` is called
5+ times per L-BFGS-B iteration and twice per `lbfgsb_matupd` column,
so JIT compile time scaled linearly with `n`.  This was the audit's
H-2 / Finding #3 in
`.artifacts/jax_convention_review_2026-05-16/06_review_geo_big.md` and
item 2 of `.artifacts/lbfgsb_parity_review_2026-05-16/00_SYNTHESIS.md`.

### Byte-identity contract decision

`tests/geo/test_lbfgsb_scipy_parity.py` is the only contract under
which `_lbfgsb_ddot` is exercised end-to-end (`max_workspace_ulp =
_SETULB_REPLAY_MAX_ULP = 512`, `tests/geo/test_lbfgsb_scipy_parity.py:22`).
The replay assertions allow up to 512 ULP between the JAX state and
the SciPy oracle, so a sequential vs. tree reduction does not change
the existing pass/fail boundary.  The SciPy BLAS DDOT skip-zero
semantic — "products that are exactly zero leave the accumulator
unchanged" — is preserved by masking products to zero before a
reduction (`jnp.where(products != 0.0, products, 0.0)`).  This form
is what the audit synthesis and the convention review both
recommend, and it keeps the SciPy "explicit skip-zero" intent
visible in the code rather than collapsing to bare `jnp.sum(x * y)`.

### Applied edit

Before:

```python
def _lbfgsb_ddot(x, y) -> jax.Array:
    total = jnp.asarray(0.0, dtype=jnp.float64)
    for i in range(int(x.shape[0])):
        product = x[i] * y[i]
        # Adding exact zero is a no-op in SciPy's BLAS accumulation.
        total = jax.lax.cond(
            product != 0.0,
            lambda value: value + product,
            lambda value: value,
            total,
        )
    return total
```

After:

```python
def _lbfgsb_ddot(x, y) -> jax.Array:
    # Mirror SciPy's BLAS DDOT skip-zero accumulation in a single vectorised
    # reduction: products that are exactly zero leave the running sum
    # unchanged, so masking them to zero before the reduction preserves the
    # SciPy semantics while emitting O(1) HLO ops instead of an O(n) lax.cond
    # chain.
    products = x * y
    return jnp.sum(jnp.where(products != 0.0, products, 0.0))
```

`_lbfgsb_dnrm2` (`_lbfgsb_scipy.py:350-351`) is unchanged and still
defined as `jnp.sqrt(_lbfgsb_ddot(x, x))`, so it inherits the new
form transparently.

## Verification

`ruff check` and `ruff format` both clean on
`src/simsopt/geo/optimizer_jax_private/_lbfgsb_scipy.py` after each fix.

Tests run on the project-local `.conda/jax` interpreter
(`jax==0.10.0`, `jaxlib==0.10.0`, NumPy 2.x, Python 3.11):

- `tests/geo/test_lbfgsb_scipy_jax_kernels.py` — **52 / 52 passed**
  in 39.8 s (full kernel suite, exercises `lbfgsb_formt`,
  `lbfgsb_formk`, `lbfgsb_cmprlb`, `lbfgsb_subsm`, `lbfgsb_cauchy`,
  `lbfgsb_lnsrlb`, `lbfgsb_bmv`, `lbfgsb_dcstep`, `lbfgsb_dcsrch` —
  no `_lbfgsb_ddot` regressions, no `formt` reference regressions).
- `tests/geo/test_lbfgsb_scipy_parity.py::test_jax_setulb_initial_start_transition_matches_scipy`
  — **4 / 4 parametrised cases passed** (unconstrained, project-lower,
  project-upper, fixed-variable).
- `tests/geo/test_lbfgsb_scipy_parity.py::test_jax_setulb_line_search_restart_with_history_matches_scipy`
  — **passed** (this is the SciPy-vs-JAX RESTART replay test that
  exercises a real `col != 0` lnsrlb-restart path with full
  workspace-state equality up to ULP budget).
- `tests/geo/test_lbfgsb_scipy_parity.py::test_jax_setulb_new_x_reentry_next_line_search_matches_scipy[unconstrained]`,
  `..._reentry_projected_gradient_convergence_matches_scipy`,
  `..._reentry_relative_reduction_convergence_matches_scipy` —
  **3 / 3 passed** (full new-X reentry path, which is where the
  formt refresh would land).
- `tests/geo/test_lbfgsb_scipy_parity.py::test_jax_setulb_frozen_replay_prefix_matches_scipy[unconstrained,boxed,lower-only]`
  — **3 / 3 passed** in 12 m 47 s.  These are the SciPy-vs-JAX
  ULP-bounded full-workspace replay tests; passing them after the
  `_lbfgsb_ddot` rewrite confirms the new vectorised form stays
  within the 512-ULP budget that the parity contract permits.

No existing test exercises the SciPy formt-info!=0 path directly,
because the parity replay objectives are well-conditioned and never
trigger non-positive Cholesky in `formt`; the audit's open follow-up
is to add a singular-Hessian fixture.  All existing tests continue to
pass: when `refresh == False`, the new `next_task_code` reduces to
`state.workspace.task[0]` and the workspace is rebuilt with the same
task it already carried (i.e. `_lbfgsb_task(state.workspace.task[0],
state.workspace.task[1])` is the same `(2,) int32` array
component-by-component as the prior `state.workspace.task`).

## Files touched

- `src/simsopt/geo/optimizer_jax_private/_lbfgsb_scipy.py`
  (lines `340-347`, `1395-1446`) — both fixes.

No tests or fixtures modified.  No other files touched.

## Worktree note

During the work window an unrelated parallel agent issued
`git stash`, which moved the in-progress edits into `stash@{0}`.
The edits were re-applied to HEAD via
`git apply <(git diff stash@{0}^ stash@{0} -- src/simsopt/geo/optimizer_jax_private/_lbfgsb_scipy.py)`
without any other state-bearing interference; `ruff` clean and the
52-test kernel suite passed (18 s) after the re-application.
