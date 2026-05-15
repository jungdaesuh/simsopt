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
- Current target lane:
  `src/simsopt/geo/optimizer_jax_private/_lbfgs.py` caches a JAX value/grad
  kernel, then calls `minimize_lbfgs_host_core(...)`.
- Current host core:
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
- [ ] Use fixed-size JAX arrays and static `n`, `m`, `maxls`, and max iteration
      limits suitable for `lax.while_loop`.
- [x] Use functional JAX array updates (`array.at[index].set(...)`) with static
      slice sizes; dynamic values may choose indices but not slice extents.
- [x] Require JAX x64 for bitwise control tests. A test configuration without
      x64 is a failed parity setup, not a looser acceptance tier.
- [ ] Add a memory budget check for planned `n`, `m`, and trace settings before
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

- [ ] Implement a JAX `setulb` equivalent that accepts and returns the same
      logical state as SciPy's `setulb`.
- [ ] Implement `mainlb` as a `lax.while_loop`-driven state machine.
- [ ] Preserve SciPy task transitions:
      `START`, `FG`, `NEW_X`, `CONVERGENCE`, `STOP`, `WARNING`, `ERROR`,
      `ABNORMAL`.
- [ ] Preserve SciPy function/gradient request behavior internally: when the
      loop state enters `FG`, the compiled JAX body evaluates the cached
      value/grad function and continues the state machine without a Python
      re-entry loop.
- [ ] Preserve SciPy `factr = ftol / eps` behavior and projected-gradient
      `pgtol = gtol` behavior.
- [ ] Preserve SciPy counter semantics, including deferred maxfun handling until
      a minimization iteration completes.
- [ ] Preserve SciPy final `hess_inv` history extraction semantics from `wa`
      (`ws`, `wy`, and `isave[30]`).

### Phase 5: Wire Into `lbfgs-ondevice`

- [ ] Replace the current target-lane call from `_lbfgs.py` to
      `minimize_lbfgs_host_core(...)` with the SciPy-compatible JAX state
      machine.
- [ ] Keep the cached JAX value/grad kernel seam in `_lbfgs.py`.
- [ ] Keep `initial_value_and_grad` support only if it maps exactly to SciPy's
      first `FG` request. Otherwise remove or rework it before claiming parity.
- [ ] Map public `maxcor`, `ftol`, `gtol`, `maxfun`, `maxiter`, and `maxls` to
      SciPy names. Audit current `lbfgs-ondevice` defaults before changing any
      default values; a default change is a downstream regression gate, not an
      incidental port detail.
- [ ] Return `OptimizeResult` fields with SciPy-compatible `success`, `status`,
      `message`, `nit`, `nfev`, `njev`, `jac`, `fun`, `x`, and `hess_inv`
      behavior.
- [ ] Re-evaluate `failure_callback`, `progress_callback`, and
      `optimizer_state_trace`: keep them as observability outputs only if they
      do not perturb SciPy control flow.
- [ ] Keep `optimizer_state_trace` diagnostic-only and bounded. Do not make
      full-trajectory materialization part of the default production result.

### Phase 6: Test Matrix

- [ ] Run the adapted SciPy 1.17.1 L-BFGS-B tests first:
      boxed-iterate float-rounding,
      float32-gradient regression,
      and inverse-Hessian matvec/dense equivalence.
- [ ] Unit-test every ported kernel against checked references and full
      `_lbfgsb.setulb(...)` trace snapshots.
- [ ] Run full optimizer-control replay tests with frozen `f,g` sequences and
      assert exact integer/control state equality. Numeric arrays get exact
      equality only when operation order is intentionally pinned.
- [ ] Run live objective tests on CPU JAX kernels and compare against SciPy with
      strict tolerances after verifying the replay path is bitwise.
- [ ] Run GPU tests only after CPU control parity passes; GPU proof validates
      XLA/runtime behavior, not the SciPy source translation itself.
- [ ] Add parity tests for bounds even if production banana paths mostly use
      unconstrained or transformed coordinates.
- [ ] Add regression tests proving `lbfgs-trace` is not used as the SciPy oracle.
- [ ] Add tests that prove target lane does not silently call SciPy.
- [ ] Add call-site regression tests for Stage 2 and single-stage wrappers so
      existing explicit optimizer options keep flowing to `lbfgs-ondevice`.

### Phase 7: Scientific/Computation Acceptance

- [ ] Fixed-state scalar objectives: SciPy CPU equals JAX CPU for value and
      gradient before optimizer comparison.
- [ ] Optimizer-control replay: JAX port equals SciPy bitwise for
      integer/control state, with numeric-array equality governed by the pinned
      operation-order rule above.
- [ ] JAX CPU live optimizer: same termination class and trajectory within the
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

- [ ] No broad rewrite outside the optimizer target surface.
- [ ] No SciPy runtime fallback in `lbfgs-ondevice`.
- [ ] No tolerance loosening to hide optimizer-control drift.
- [ ] No dynamic imports.
- [ ] No defensive try/except wrapper around optimizer failure.
- [ ] No change to upstream CPU/reference behavior unless a test proves the
      current reference adapter is already inconsistent with SciPy.
- [ ] Every status/message mapping is traceable to SciPy 1.17.1 source.
- [ ] No unbounded per-iteration Python tuple growth in production optimizer
      state.
- [ ] No shared mutable workspace across concurrent optimizer calls.

## Required Decisions Before Implementation

- [ ] Full bound semantics are required for SciPy L-BFGS-B parity. The
      implementation may land behind small fixtures first, but final acceptance
      must cover lower-only, upper-only, boxed, fixed, and unbounded variables.
- [ ] Whether to keep the current custom host core as `lbfgs-trace` diagnostic
      only, delete it after parity lands, or rename it to make the non-SciPy
      semantics explicit.
- [ ] Whether GPU acceptance should require exact trajectory equality on simple
      quadratic kernels or only control-state equality under replay plus strict
      live-kernel tolerances.
