# SciPy L-BFGS-B JAX Parity Implementation Plan

Date: 2026-05-15

## Goal

Implement the `lbfgs-ondevice` optimizer so its optimizer-control behavior
matches SciPy `L-BFGS-B` as closely as JAX permits. The parity target is SciPy
1.17.1, because that is the installed SciPy version in this checkout.

Bitwise parity means:

- identical optimizer state transitions when the JAX port and SciPy are fed the
  same deterministic objective/gradient values;
- identical accepted/rejected step decisions, internal `task[0]`/`task[1]`
  termination codes, counters, and work-array-derived history;
- identical public `OptimizeResult` fields where those fields are SciPy-owned.
  This is separate from internal task codes: SciPy's Python wrapper maps
  internal `_lbfgsb` task state to public `status` values `0`, `1`, or `2`;
- numeric array equality is bitwise only where the operation order is pinned.
  JAX/XLA may fuse or reorder objective, physics, and optimizer arithmetic
  kernels, so kernel-level drift must be measured separately with explicit
  tolerances after discrete optimizer-control parity is proven.

## Current Tree Facts

- SciPy dispatch for `method="L-BFGS-B"` goes through
  `scipy.optimize._minimize._minimize_lbfgsb`, then repeatedly calls the
  compiled `_lbfgsb.setulb(...)` routine.
- SciPy 1.17.1 wrapper path on this machine:
  `/opt/homebrew/Caskroom/miniforge/base/lib/python3.13/site-packages/scipy/optimize/_lbfgsb_py.py`.
- SciPy 1.17.1 raw implementation source:
  `https://raw.githubusercontent.com/scipy/scipy/v1.17.1/scipy/optimize/__lbfgsb.c`.
- SciPy 1.17.1 raw header source:
  `https://raw.githubusercontent.com/scipy/scipy/v1.17.1/scipy/optimize/__lbfgsb.h`.
- Official SciPy L-BFGS-B docs define `maxcor`, `ftol`, `gtol`, `maxfun`,
  `maxiter`, and `maxls`; they also state that `ftol = factr * eps` for the
  lower-level `fmin_l_bfgs_b` interface.
- Official JAX control-flow docs require `lax.while_loop` loop-carried values to
  keep a fixed shape and dtype across iterations. Official JAX array-update docs
  require functional `.at[...]` updates instead of in-place mutation.
- Previous target lane:
  `src/simsopt/geo/optimizer_jax_private/_lbfgs.py` cached a JAX value/grad
  kernel, then called `minimize_lbfgs_host_core(...)`.
- Current target lane:
  `src/simsopt/geo/optimizer_jax_private/_lbfgs.py` caches a JAX value/grad
  kernel, initializes the SciPy-compatible L-BFGS-B state, and runs the JAX
  `setulb/mainlb` port from `_lbfgsb_scipy.py`.
- Previous host core:
  `src/simsopt/geo/optimizer_host_lbfgs.py` is a custom unconstrained L-BFGS
  implementation with a strong-Wolfe line search and repo-specific retry/status
  behavior.
- Existing JAX line search:
  `src/simsopt/geo/optimizer_jax_private/_line_search.py` follows a
  JAX/Nocedal-Wright strong-Wolfe flow, not SciPy L-BFGS-B's `dcsrch/dcstep`
  state machine.

## Source References

- SciPy L-BFGS-B options:
  `https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html`.
- SciPy 1.17.1 Python wrapper:
  `https://raw.githubusercontent.com/scipy/scipy/v1.17.1/scipy/optimize/_lbfgsb_py.py`.
- SciPy 1.17.1 translated C core:
  `https://raw.githubusercontent.com/scipy/scipy/v1.17.1/scipy/optimize/__lbfgsb.c`.
- SciPy 1.17.1 translated C header:
  `https://raw.githubusercontent.com/scipy/scipy/v1.17.1/scipy/optimize/__lbfgsb.h`.
- SciPy 1.17.1 dedicated L-BFGS-B tests:
  `https://raw.githubusercontent.com/scipy/scipy/v1.17.1/scipy/optimize/tests/test_lbfgsb_setulb.py`
  and
  `https://raw.githubusercontent.com/scipy/scipy/v1.17.1/scipy/optimize/tests/test_lbfgsb_hessinv.py`.
- JAX `lax.while_loop` fixed-shape loop carry contract:
  `https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html`.
- JAX functional array update contract:
  `https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html`.

## TDD Rule

Implementation must proceed red-green by SciPy behavior slice:

- write or adapt the failing SciPy-oracle test first;
- implement only the smallest kernel/state-machine change needed for that test;
- keep existing `bfgs-ondevice`, SciPy/reference, Stage 2, and single-stage
  routing tests green before moving to the next slice;
- wire the public `lbfgs-ondevice` route only after the lower-level
  `setulb/mainlb` replay suite passes.

## Algorithm Gap Analysis

SciPy L-BFGS-B is not just two-loop L-BFGS plus a Wolfe search. The SciPy core
is the translated L-BFGS-B 3.0 state machine:

- `setulb`: partitions the flat work arrays and calls `mainlb`.
- `mainlb`: owns the iterative control state, task codes, tolerance checks,
  active-set transitions, and calls into the algorithm kernels.
- `active`: projects the initial point into bounds and initializes per-variable
  bound state.
- `projgr`: computes the infinity norm of the projected gradient, which is the
  `gtol`/`pgtol` convergence test.
- `cauchy`: computes the generalized Cauchy point along piecewise-linear bound
  breakpoints.
- `freev`, `formk`, `formt`, `cmprlb`, `subsm`: build and solve the compact
  limited-memory subspace problem.
- `lnsrlb`, `dcsrch`, `dcstep`: run the More-Thuente line-search state machine
  with SciPy's task/status semantics.
- `matupd`: updates `WS`, `WY`, `SS`, `SY`, `theta`, `col`, `head`, and related
  compact-memory state.

The local host core diverges materially:

- uses unconstrained two-loop recursion instead of the compact L-BFGS-B
  generalized-Cauchy/subspace flow;
- uses custom status integers instead of SciPy task/status message pairs;
- adds repo-specific stalled-step and nonfinite rejected-step behavior;
- clamps `gamma` and skips history updates using local rules;
- initializes and updates counters differently from SciPy `_lbfgsb_py.py`;
- uses `host_norm(g)` instead of projected-gradient norm semantics.

Therefore, the root fix is a SciPy `setulb/mainlb` port, not incremental tuning
of the current `minimize_lbfgs_host_core(...)`.

## Implementation Boundary

- Preserve the public method name `lbfgs-ondevice`.
- Preserve SciPy/reference lanes unchanged except for tests or explicit oracle
  helpers.
- Keep SciPy as the oracle, not as a runtime fallback for the target lane.
- Do not add a fallback mode that silently routes target execution back to
  SciPy.
- Keep all objective/gradient evaluation through the existing cached JAX
  value/grad kernel seam.
- Implement optimizer-control state as JAX data structures with static shapes.
- Match SciPy control flow around scalar reductions, while treating the
  low-level reduction arithmetic itself as device-kernel arithmetic. SciPy calls
  the BLAS linked into its wheel for `ddot_`/`dnrm2_`; the pure-JAX target must
  use JAX kernels for those operations and must not add a host-BLAS fallback.
- Treat unsupported dynamic behaviors as compile-time contract decisions, not
  try/except recovery paths.
- Model SciPy reverse communication inside the JAX loop state. The target lane
  must call the cached JAX value/grad function from compiled control flow when
  the internal task requests `FG`; it must not reintroduce a Python host
  optimizer loop.
- Keep mutable optimizer state per invocation. Do not introduce module-level
  workspaces, global trace buffers, or shared mutable state.

## Non-Goals

- Do not implement SciPy finite-difference gradient options (`eps`,
  `finite_diff_rel_step`, `workers`) for `lbfgs-ondevice`; the target lane owns
  an explicit value-and-gradient contract.
- Do not support deprecated SciPy text-output controls (`disp`, `iprint`) in the
  target lane.
- Do not keep the current two-loop strong-Wolfe core as a silent compatibility
  path for `lbfgs-ondevice`.

## Plan

### Phase 0: Pin The Oracle

- [x] Add a small source-manifest section in this plan or a companion test file
      that records the SciPy oracle version: `1.17.1`.
- [x] Record exact upstream source URLs for `_lbfgsb_py.py`, `__lbfgsb.c`, and
      `__lbfgsb.h`.
- [x] Add a local probe that confirms installed SciPy still resolves
      `_lbfgsb_py.py` and `_lbfgsb` from the expected package.
- [x] Pin public-vs-internal status semantics:
      internal `task[0]`/`task[1]` follows `_lbfgsb`; public
      `OptimizeResult.status` follows SciPy `_minimize_lbfgsb` wrapper behavior.
- [x] Decide whether to vendor translated constants/status tables into the repo
      or generate them from a checked-in snapshot. Prefer checked-in, reviewed
      constants for deterministic review.

### Phase 1: Build A SciPy Replay Oracle

- [x] Add upstream-derived tests from SciPy 1.17.1 before porting kernels:
      `test_setulb_floatround`,
      `test_gh_issue18730`,
      `test_1`,
      `test_2`,
      and `test_3` from SciPy's dedicated L-BFGS-B test files.
- [x] Create a direct reverse-communication SciPy driver around
      `_lbfgsb.setulb(...)` that records every requested `x`, `f`, and `g`.
      Do not rely on public `minimize(...)` callbacks for this; public callbacks
      do not expose SciPy's internal work arrays.
- [x] Build replay fixtures for:
      unconstrained quadratic,
      diagonal ill-conditioned quadratic,
      Rosenbrock,
      lower-bound-only case,
      upper-bound-only case,
      boxed case,
      fixed-variable case,
      finite-difference-disabled analytic-gradient case.
- [x] Preserve the exact SciPy `test_setulb_floatround` fixture as a required
      regression: seven `setulb` calls with all variables boxed in `[0, 1]`
      must never produce an out-of-bounds iterate.
- [x] Preserve the SciPy GH-18730 regression: objectives returning `float32`
      gradients must not corrupt the L-BFGS-B solve. The parity implementation
      must match SciPy's explicit float64 upcast at the optimizer-control
      boundary.
- [x] Preserve SciPy inverse-Hessian checks: `hess_inv(vector)` must match
      `hess_inv.todense()` on the scalar quartic fixture, 2-D quadratic fixture,
      and old dense implementation equivalence fixture.
- [x] Capture SciPy internal task transitions, `x`, `f`, `g`, `wa`, `iwa`,
      `task`, `ln_task`, `lsave`, `isave`, `dsave`, `nfev`, `njev`, `nit`,
      public final status, and public final message.
- [x] Add bitwise replay assertions for integer/control fields before any JAX
      port changes are accepted. Numeric arrays should be bitwise only for
      pinned operation-order fixtures; otherwise assert a documented ULP or
      absolute/relative tolerance budget tied to the exact kernel.
- [x] Add negative tests for malformed bounds and `maxls <= 0` to preserve
      SciPy error semantics at the public adapter boundary.

### Phase 2: Define JAX State Types

- [x] Add typed JAX state containers for SciPy-equivalent state:
      `x`, `l`, `u`, `nbd`, `f`, `g`, `wa`, `iwa`, `task`, `ln_task`,
      `lsave`, `isave`, `dsave`.
- [x] Keep the SciPy work-array layout:
      `2*m*n + 5*n + 11*m*m + 8*m` for `wa` and `3*n` for `iwa`.
- [x] Keep SciPy's integer task/message encoding rather than inventing a new
      status enum.
- [x] Use fixed-size JAX arrays and static `n`, `m`, `maxls`, and max iteration
      limits suitable for `lax.while_loop`.
- [x] Use functional JAX array updates (`array.at[index].set(...)`) with static
      slice sizes; dynamic values may choose indices but not slice extents.
- [x] Require JAX x64 for bitwise control tests. A test configuration without
      x64 is a failed parity setup, not a looser acceptance tier.
- [x] Add a memory budget check for planned `n`, `m`, and trace settings before
      enabling expensive parity runs; production execution must not allocate
      full optimizer traces unless explicitly requested for diagnostics.
- [x] Use `float64` for optimizer-control parity tests; treat `float32` as a
      separate later acceptance tier.

### Phase 3: Port Low-Level Kernels

- [x] Start each kernel port with a failing upstream-derived or replay-derived
      test. Do not write production kernel code before the corresponding red
      test exists.
- [x] Port `projgr` first and test projected-gradient norm bitwise against
      SciPy for all `nbd` cases.
- [x] Port `active` and verify initial projection, `iwhere`, `prjctd`,
      `cnstnd`, and `boxed`.
- [x] Port compact-memory helpers:
      `bmv`, `formt`, `formk`, `cmprlb`, and `matupd`.
- [x] Port `hpsolb`, the heap helper used by `cauchy` breakpoint ordering.
- [x] Port `cauchy` with its breakpoint ordering and heap behavior.
- [x] Port `freev` with exact active/free-set updates.
- [x] Port `subsm` with exact active/free-set updates.
- [x] Port `dcsrch` and `dcstep` before wiring the full outer loop. Do not reuse
      the existing `_line_search.py` flow for SciPy-bitwise mode.
- [x] Port the initial `lnsrlb` reverse-communication wrapper around `dcsrch`;
      full integration remains blocked on `setulb/mainlb`.
- [x] For each ported low-level function, add checked reference tests. SciPy's wheel exposes
      `_lbfgsb.setulb(...)`, but the lower-level C helpers are `static`; direct
      calls to `projgr`, `cauchy`, `subsm`, or `dcsrch` require either a
      checked-in reference translation or a tiny test-only build artifact.

### Phase 4: Port `setulb/mainlb`

- [x] Port the initial `setulb` reverse-communication entry:
      `START` / `[0, 0]` to `FG` / `FG_START` / `[3, 301]`, including
      SciPy work-array partition offsets, initial `active(...)` projection,
      saved `mainlb` locals, and fixed-shape JIT coverage.
- [x] Port the `FG_START` re-entry convergence return for
      `NORM OF PROJECTED GRADIENT <= PGTOL`, including projected-gradient
      bound semantics, saved `nfgv`/`sbgnrm`, and fixed-shape JIT coverage.
- [x] Port the initial non-converged `FG_START` re-entry through
      `cauchy`, `freev`, and `lnsrlb` to the first `FG_LNSRCH` request,
      including SciPy's saved `nseg` behavior, fixed-variable bound replay
      coverage, and fixed-shape JIT coverage.
- [x] Port the first `FG_LNSRCH` re-entry that accepts the line-search point
      and returns `NEW_X`, including saved line-search locals, projected
      gradient refresh, iteration/nfgv counters, bound edge cases, and
      fixed-shape JIT coverage.
- [x] Port the first `NEW_X` re-entry slice for projected-gradient
      convergence, relative-reduction convergence, and the nonterminal
      next-iteration path through `matupd`, `formt`, constrained and
      unconstrained `cauchy`/`freev`/`formk`/`cmprlb`/`subsm`, and the second
      `FG_LNSRCH` request. Integer and task state is exact;
      Cholesky/triangular-solve numeric work arrays use the documented kernel
      ULP budget.
- [x] Port the following `FG_LNSRCH` re-entry that accepts the second
      line-search point and returns the second raw `NEW_X`, using frozen SciPy
      reverse-communication `f,g` replay values to isolate optimizer-control
      state from live objective-kernel drift.
- [x] Add frozen SciPy `f,g` prefix replay coverage through nine
      reverse-communication events for unbounded, boxed, lower-only,
      upper-only, and fixed-variable bounds. Task, integer workspace, and
      counters are exact; trajectory and accumulated numeric workspace arrays
      use the documented 512-ULP multi-step replay budget while frozen replay
      `f,g` values remain exact.
- [x] Add full frozen SciPy `f,g` replay coverage for exact task,
      integer-workspace, logical-flag, and counter state through termination on
      unbounded, boxed, lower-only, upper-only, and fixed-variable quadratic
      traces. Numeric trajectory/workspace equality remains covered by the
      bounded prefix replay until operation-order parity is pinned further.
- [x] Implement a JAX `setulb` equivalent that accepts and returns the same
      logical state as SciPy's `setulb`; full frozen replay verifies task,
      integer workspace, logical flags, and counters.
- [x] Implement `mainlb` as a `lax.while_loop`-driven state machine.
- [x] Preserve SciPy task transitions:
      `START`, `FG`, `NEW_X`, `CONVERGENCE`, `STOP`, `WARNING`, `ERROR`,
      `ABNORMAL`.
- [x] Preserve SciPy function/gradient request behavior internally: when the
      loop state enters `FG`, the compiled JAX body evaluates the cached
      value/grad function and continues the state machine without a Python
      re-entry loop.
- [x] Preserve SciPy `factr = ftol / eps` behavior and projected-gradient
      `pgtol = gtol` behavior.
- [x] Preserve SciPy counter semantics, including deferred maxfun handling until
      a minimization iteration completes.
- [x] Preserve SciPy final `hess_inv` history extraction semantics from `wa`
      (`ws`, `wy`, and `isave[30]`).

### Phase 5: Wire Into `lbfgs-ondevice`

- [x] Replace the current target-lane call from `_lbfgs.py` to
      `minimize_lbfgs_host_core(...)` with the SciPy-compatible JAX state
      machine.
- [x] Keep the cached JAX value/grad kernel seam in `_lbfgs.py`.
- [x] Keep `initial_value_and_grad` support only if it maps exactly to SciPy's
      first `FG` request. Otherwise remove or rework it before claiming parity.
- [x] Map public `maxcor`, `ftol`, `gtol`, `maxfun`, `maxiter`, and `maxls` to
      SciPy names. Audit current `lbfgs-ondevice` defaults before changing any
      default values; a default change is a downstream regression gate, not an
      incidental port detail.
- [x] Return `OptimizeResult` fields with SciPy-compatible `success`, `status`,
      `message`, `nit`, `nfev`, `njev`, `jac`, `fun`, `x`, and `hess_inv`
      behavior.
- [x] Re-evaluate `failure_callback`, `progress_callback`, and
      `optimizer_state_trace`: keep them as observability outputs only if they
      do not perturb SciPy control flow.
- [x] Remove stale target L-BFGS-B forwarding for non-SciPy controls:
      `initial_step_size`, `maxgrad`, and target-lane `failure_callback` now
      fail closed instead of being accepted and ignored across `lbfgs-ondevice`,
      `lbfgs-scipy-jax`, and `lbfgs-scipy-jax-fullgraph`.
- [x] Keep `optimizer_state_trace` diagnostic-only and bounded. Do not make
      full-trajectory materialization part of the default production result.

### Phase 6: Test Matrix

- [x] Run the adapted SciPy 1.17.1 L-BFGS-B tests first:
      boxed-iterate float-rounding,
      float32-gradient regression,
      and inverse-Hessian matvec/dense equivalence.
- [x] Unit-test every ported kernel against checked references and full
      `_lbfgsb.setulb(...)` trace snapshots.
- [x] Run full optimizer-control replay tests with frozen `f,g` sequences and
      assert exact integer/control state equality. Numeric arrays get exact
      equality only when operation order is intentionally pinned.
- [x] Add bounded frozen prefix replay tests before claiming full replay:
      unbounded, boxed, lower-only, upper-only, and fixed-variable cases cover
      the first nine reverse-communication events with exact task/counter and
      integer-workspace equality.
- [x] Run live objective tests on CPU JAX kernels and compare against SciPy with
      strict tolerances after verifying the replay path is bitwise.
- [ ] Run GPU tests only after CPU control parity passes; GPU proof validates
      XLA/runtime behavior, not the SciPy source translation itself.
- [x] Add parity tests for bounds even if production banana paths mostly use
      unconstrained or transformed coordinates.
- [x] Add regression tests proving `lbfgs-trace` is not used as the SciPy oracle.
- [x] Add tests that prove target lane does not silently call SciPy.
- [x] Add call-site regression tests for Stage 2 and single-stage wrappers so
      existing explicit optimizer options keep flowing to `lbfgs-ondevice`.

### Phase 7: Scientific/Computation Acceptance

- [x] Fixed-state scalar objectives: SciPy CPU equals JAX CPU for value and
      gradient before optimizer comparison.
- [x] Optimizer-control replay: JAX port equals SciPy bitwise for
      integer/control state, with numeric-array equality governed by the pinned
      operation-order rule above.
- [x] JAX CPU live optimizer: same termination class and trajectory within the
      kernel-arithmetic tolerance budget.
- [ ] JAX GPU live optimizer: same termination class and trajectory within the
      GPU arithmetic tolerance budget.
- [ ] Banana/Stage 2 path: keep the trust chain explicit:
      SIMSOPT C++/SciPy -> JAX CPU -> JAX GPU -> JAX CPU/GPU agreement.

## Files Likely To Change

- `src/simsopt/geo/optimizer_jax_private/_lbfgs.py`
- `src/simsopt/geo/optimizer_jax_private/_types.py`
- `src/simsopt/geo/optimizer_jax_private/_result_converters.py`
- new file, likely
  `src/simsopt/geo/optimizer_jax_private/_lbfgsb_scipy.py`
- new test file, likely `tests/geo/test_lbfgsb_scipy_parity.py`
- targeted updates in `tests/geo/test_boozersurface_jax_private.py`
- targeted routing assertions in `tests/geo/test_boozersurface_jax.py`

## Review Gates

- [x] No broad rewrite outside the optimizer target surface.
- [x] No SciPy runtime fallback in `lbfgs-ondevice`.
- [x] No tolerance loosening to hide optimizer-control drift.
- [x] No dynamic imports.
- [x] No defensive try/except wrapper around optimizer failure.
- [x] No change to upstream CPU/reference behavior unless a test proves the
      current reference adapter is already inconsistent with SciPy.
- [x] Every status/message mapping is traceable to SciPy 1.17.1 source.
- [x] No unbounded per-iteration Python tuple growth in production optimizer
      state.
- [x] No shared mutable workspace across concurrent optimizer calls.

## Required Decisions Before Implementation

- [x] Full bound semantics are required for SciPy L-BFGS-B parity. The
      implementation may land behind small fixtures first, but final acceptance
      must cover lower-only, upper-only, boxed, fixed, and unbounded variables.
- [x] Whether to keep the current custom host core as `lbfgs-trace` diagnostic
      only, delete it after parity lands, or rename it to make the non-SciPy
      semantics explicit.
- [x] Whether GPU acceptance should require exact trajectory equality on simple
      quadratic kernels or only control-state equality under replay plus strict
      live-kernel tolerances. Decision: require exact integer/control-state
      replay equality, then strict live-kernel trajectory tolerances on GPU.
      Do not require bitwise live trajectory equality across XLA/CUDA kernels.

## Completion Audit Snapshot: 2026-05-16

### Delivered Evidence

- [x] SciPy oracle and upstream source pinning:
      `tests/geo/test_lbfgsb_scipy_parity.py` records direct `_lbfgsb.setulb`
      replay traces and SciPy wrapper-derived public status/message behavior.
- [x] JAX state-machine implementation:
      `src/simsopt/geo/optimizer_jax_private/_lbfgsb_scipy.py` contains the
      fixed-shape `setulb/mainlb` translation, SciPy task/message tables,
      bounds encoding, compact-memory kernels, More-Thuente line search, and
      inverse-Hessian history extraction.
- [x] Public `lbfgs-ondevice` route:
      `src/simsopt/geo/optimizer_jax_private/_lbfgs.py` now calls the
      SciPy-compatible JAX state machine through the cached value/gradient seam
      instead of `minimize_lbfgs_host_core(...)`.
- [x] Public result conversion:
      `src/simsopt/geo/optimizer_jax_private/_result_converters.py` maps
      task-derived SciPy public `status`, `success`, `message`, counters,
      arrays, and `LbfgsInvHessProduct` history.
- [x] CPU validation run after the simplifier pass:
      `ruff format`/`ruff check` on the touched files passed,
      scoped source mypy on the touched optimizer modules passed,
      `TestOptimizerAdapterPrivate` passed (`22 passed`),
      `TestLBFGSMethodPrivate` passed after the seeded-FG parity test addition
      (`16 passed`),
      `tests/geo/test_lbfgsb_scipy_parity.py` plus
      `tests/geo/test_lbfgsb_scipy_jax_kernels.py` passed (`135 passed`), and
      Stage 2/single-stage/Boozer call-site regressions passed (`3 passed`).
- [x] Reviewer cleanup validation after stale-contract removal:
      target L-BFGS-B methods reject `initial_step_size`, `maxgrad`, and
      `failure_callback`; single-stage and Stage 2 no longer forward target
      failure callbacks; target-lane diagnostic callbacks now cover accepted-step
      and progress hooks only; invalid-step diagnostics come from structured
      `invalid_step_log` result entries. SciPy task-backed nonfinite public
      results preserve SciPy `status`, `success`, and `message`. Validation:
      focused contract/nonfinite/downstream tests passed (`18 passed`), the
      full single-stage unit file passed (`320 passed, 20 subtests passed`),
      Stage 2 focused callback tests passed (`4 passed`), and the SciPy
      parity/kernel suite passed (`135 passed`). The full private L-BFGS method
      class was rerun after these changes (`16 passed`). A final stale-surface
      simplifier pass also passed `ruff`, focused source mypy, 17 focused
      contract/CLI tests, and `git diff --check`.
- [x] Current-tree audit refresh after the stale-surface fixes:
      `JAX_ENABLE_X64=True pytest -q tests/geo/test_lbfgsb_scipy_parity.py
      tests/geo/test_lbfgsb_scipy_jax_kernels.py` passed
      (`135 passed in 268.71s`). GPU-proof launcher/manifest contract tests
      passed locally (`67 passed, 1 skipped`), confirming the proof plumbing
      remains checked even though this host cannot execute CUDA.

### Still Open

- [x] Add explicit JAX-vs-SciPy tests for `WARNING` and `ERROR` task
      transitions in addition to the current `START`, `FG`, `NEW_X`,
      `CONVERGENCE`, `STOP`, and `ABNORMAL` evidence.
- [x] Complete the direct kernel audit for every ported helper against checked
      references or full `_lbfgsb.setulb(...)` snapshots before claiming the
      "every kernel" test-matrix item.
- [x] Commit and push the current-tree parity slice before remote GPU proof.
      The scoped parity slice was committed as
      `0a54646c1 implement scipy lbfgsb jax parity` and pushed to
      `fork/gpu-purity-stage2-20260405`. That commit contains:
      `docs/scipy_lbfgsb_jax_parity_plan_2026-05-15.md`,
      `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`,
      `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py`,
      `src/simsopt/geo/boozersurface_jax.py`,
      `src/simsopt/geo/optimizer_jax.py`,
      `src/simsopt/geo/optimizer_jax_private/_lbfgs.py`,
      `src/simsopt/geo/optimizer_jax_private/_lbfgsb_scipy.py`,
      `src/simsopt/geo/optimizer_jax_private/_result_converters.py`,
      `src/simsopt/geo/optimizer_jax_private/_types.py`,
      `tests/geo/test_boozersurface_jax.py`,
      `tests/geo/test_boozersurface_jax_private.py`,
      `tests/geo/test_lbfgsb_scipy_jax_kernels.py`,
      `tests/geo/test_lbfgsb_scipy_parity.py`,
      `tests/geo/test_single_stage_example.py`, and
      `tests/integration/test_stage2_jax.py`. A follow-up launcher provenance
      fix was committed as
      `fe0e69269 fix lightning preflight hardware provenance` and pushed to
      the same branch. The follow-up keeps Lightning preflight `hardware` as
      `["H200"]` instead of the character list produced by threading the
      scalar Lightning machine string through the HF Jobs hardware-list
      preflight helper.
- [ ] Run GPU live optimizer proof and record the accepted trajectory/control
      tolerance policy. Current local JAX devices are CPU-only:
      `jax.devices() == [CpuDevice(id=0)]`, `jax.default_backend() == "cpu"`;
      `nvidia-smi` is not installed. The no-cost Lightning launcher dry-run
      now resolves pushed HEAD
      `fe0e692698401388c3aef896f302c8d8ede0b409` on
      `gpu-purity-stage2-20260405`, emits `machine: "H200"` and
      `hardware: ["H200"]`, and constructs the remote command against the same
      SHA. This proves the launcher/preflight contract only; it is not a CUDA
      execution proof.
- [ ] Run the banana/Stage 2 scientific trust-chain proof:
      SIMSOPT C++/SciPy -> JAX CPU -> JAX GPU -> JAX CPU/GPU agreement.
- [x] Re-run the full `TestLBFGSMethodPrivate` class after the seeded-FG parity
      test addition before final closeout.
