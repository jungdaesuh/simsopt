# simsopt-jax convention + best-practice review — unified synthesis

**Date:** 2026-05-16
**Worktree:** `/Users/suhjungdae/code/columbia/simsopt-jax`
**Branch:** `gpu-purity-stage2-20260405`
**Runtime:** JAX 0.10.0 / jaxlib 0.10.0 / Python 3.11 / NumPy 2.x
**Reviewers:** 8 parallel Opus-4.7 (max-effort) agents — 2 baseline researchers + 6 module reviewers

## Scope

Audited ~52,671 LOC of JAX-ported code across two trees:

- `src/simsopt/jax_core/*.py` — 37 files / 21,709 LOC (pure JAX kernels; should be `simsoptpp`-free)
- `src/simsopt/{field,geo,objectives,solve}/*_jax*.py` + `backend.py` — 30,962 LOC (Optimizable adapters and dispatch)
- `tests/**/test_*jax*.py` — ~100 test files

The review applied a **dual lens** to every module: (a) does it comply with simsopt's Optimizable/DOF/Derivative/parity conventions, and (b) does it comply with JAX 0.10.0 best practice. The two baselines underpinning the rubric are:

- `01_simsopt_convention_baseline.md` — 695-line convention reference derived from `_core/optimizable.py`, `_core/derivative.py`, canonical Optimizable subclasses, and `CLAUDE.md`.
- `02_jax_best_practices_baseline.md` — 924-line JBP-1 through JBP-20 reference, every rule sourced from `docs.jax.dev` (fetched 2026-05-16) and the JAX 0.10.0 changelog.

## Bottom line

The port is **structurally solid and broadly compliant**. The Optimizable contract is observed on every adapter; ancestor-invalidation tokens, recompute_bell, derivative-graph wiring, and parity-ladder tolerances all match the documented conventions. JAX-side, the team has clean discipline on pytrees, float64 enforcement, `custom_vjp` IFT adjoints, JIT closure capture, and bundle-cache invariants (`is` identity, not `id(callable)`). No correctness BLOCKERs were found at the runtime contract level — exact-lane operator-only solve, LS-lane PLU reuse, M5 IFT sign conventions, traceable cache invalidation, and failure-mode NaN-gradient propagation all hold.

**One architectural BLOCKER** is the layering violation in `jax_core/` (item B-1 below): the kernel layer is supposed to be `simsoptpp`-free per `CLAUDE.md`, but 9 cross-package imports — some at module top level — transitively pull `Optimizable` and the `_simsoptpp` shim. This is already known to the team (memory: `project_curve_jax_core_import_cycle.md`) but should be tracked explicitly.

Beyond that, the findings split into:

- **5 HIGH best-practice items** in `jax_core/` (reverse-mode-unsupported `while_loop`, an intentional axis-convention split, frozen-dataclass-with-mutable-dict, host-callback `jnp.linalg.eig`, LRU-cache backend keying);
- **4 HIGH adapter items** in `field/` (latent shape bug in `SpecBackedBiotSavartJAX`, an advertised fast path that is never engaged, `as_dict` skipping `super`, `dB_by_dX` falling through to the C++ trampoline);
- **4 HIGH solver items** in `boozersurface_jax.py`/`optimizer_jax_private/` (L-BFGS-B RESTART task path unimplemented, dense LU materialized on every successful exact solve, `_lbfgsb_ddot` Python-unrolled per-element loop, per-call re-JIT in two L-BFGS-B kernels);
- **5 HIGH M5/cache items** in `surfaceobjectives_jax.py` (CLAUDE.md M5 adapter description is out-of-date vs the implementation, no `donate_argnums` on hot custom-VJP, condition-estimator comment ambiguity, latent dead fallback to live solver, host-input rejection contract undocumented);
- **0 HIGH test items** but 3 NEEDS-WORK (sub-noise `atol=1e-14`/`1e-18` thresholds, global `np.random.seed` leakage, 5 residual re-export `is`-identity tests).

Everything else is MEDIUM/LOW polish, performance optimization, or documentation drift.

---

## A. Cross-cutting themes (read these first)

### Theme 1 — `jax_core/` is not actually `simsoptpp`-free

Nine cross-package imports — including at module top level in `curve_geometry.py`, `magnetic_axis_helpers.py`, `surface_fourier.py`, `surface_henneberg.py`, `objectives_flux.py`, and `tracing.py` — pull `simsopt.geo` and `simsopt.field` symbols, which transitively pull `Optimizable` and the `simsoptpp` C++ shim. The two specific *_jax modules with bottom-of-file deferred imports (`surface_fourier_jax.py:2763-2768`, three deferred imports in `boozer_residual_jax.py:476-510`) are documented workarounds for the cycle. CLAUDE.md asserts "the kernel layer does NOT import simsoptpp"; this is currently false.

**Severity:** BLOCKER (convention).
**Source:** `03_review_jax_core.md` §C-01.
**Why it matters:** It blocks publishing `jax_core/` as a JAX-only library; it breaks the "pure-JAX no-simsoptpp install" promise; and the lazy-import workarounds are fragile — any new top-of-file import of one of the deferred symbols re-introduces the cycle.

### Theme 2 — `while_loop` reverse-mode autodiff is not supported

Six high-traffic functions wrap their integrators in `jax.lax.while_loop`:

- `magnetic_axis_helpers.py:515` `_integrate_tangent_map`
- `tracing.py:1234` `trace_fieldline`
- `tracing.py:1778` `trace_guiding_center`
- `tracing.py:1973`/`2799` `trace_guiding_center_boozer` and full-orbit
- `tracing.py:3342` `trace_fullorbit`
- `tracing.py:858` `bracket_root_jax`

`jax.lax.while_loop` does not support reverse-mode AD without `custom_vjp` (JBP-3.3). The `on_axis_iota_rk` docstring at `magnetic_axis_helpers.py:16` explicitly advertises gradient support, which is misleading for `jax.grad`/`jax.vjp` users (forward-mode `jvp`/`jacfwd` works fine).

**Severity:** HIGH (best-practice + documentation).
**Source:** `03_review_jax_core.md` §B-01.
**Fix:** Either (a) replace `while_loop` with bounded `lax.scan` with a mask, (b) wrap with `custom_vjp` and IFT, or (c) document forward-mode-only support.

### Theme 3 — CLAUDE.md M5 adapter description is out-of-date

CLAUDE.md says: "the JAX objective wrappers use CPU surface objects (`surface.gamma()`, `label.J()`) for value computation, and JAX autodiff through `_surface_geometry_from_dofs`/`biot_savart_B` for gradient computation". The current implementation in `surfaceobjectives_jax.py` does NOT do this — `BoozerResidualJAX._compute_value_from_solved_state` (line 2388), `IotasJAX._compute_value_from_solved_state` (line 2496), `NonQuasiSymmetricRatioJAX._compute_value` (line 2673), and `MajorRadiusJAX._compute_value` (line 2557) all reconstruct surface geometry in pure JAX from `solved_state.sdofs` for value AND gradient. The numerical equivalence is preserved (both paths compute the same scalar), but the documentation gap will mislead future maintainers about the boundary contract.

**Severity:** HIGH (convention/documentation).
**Source:** `07_review_obj_solve.md` HIGH #5.
**Fix:** Update CLAUDE.md to: "wrappers use the solved-state runtime summary plus JAX-pure surface reconstruction for both value and gradient; CPU surface objects are the spec/DOF source only".

### Theme 4 — L-BFGS-B port is correct in shape but mis-handles 4 of 5 SciPy failure-recovery paths

The on-device L-BFGS-B (`optimizer_jax_private/_lbfgsb_scipy.py`) faithfully reproduces SciPy 1.17.1 Fortran L-BFGS-B (compact-form Byrd-Lu-Nocedal-Zhu 1995). Ring-buffer arithmetic, theta scaling, curvature check, dtype contract, and the predicted-reduction Wolfe line search are correct. BUT: the port omits SciPy's `task = RESTART` flag in 4 of 5 upstream failure modes (`cauchy`/`formk`/`cmprlb`/`subsm` info!=0, `lnsrlb` failure with `col != 0`). When the reference would refresh memory and continue, the JAX port re-enters the failed line-search state until `maxfun`/`maxiter` halts (livelock on ill-conditioned exact-Jacobian inputs). Confirmed independently in `.artifacts/lbfgsb_parity_review_2026-05-16/00_SYNTHESIS.md`.

**Severity:** HIGH (correctness).
**Source:** `06_review_geo_big.md` Finding #1.
**Fix:** Interpose `info != 0` gates after `lbfgsb_formk`, `lbfgsb_cmprlb`, `lbfgsb_subsm`, `lbfgsb_cauchy`; add the `col != 0` branch on `lnsrlb` failure; write `task = RESTART, task_msg = NO_MSG` in all five paths.

### Theme 5 — Test discipline is strong overall; three concrete cleanups remain

`tests/REVIEWER_ORACLE_LINT.md` discipline is enforced: no Tier-1 tautologies remain in scope; tolerance lanes use the SSOT `parity_ladder_tolerances` from `benchmarks/validation_ladder_contract.py`; the four highest-severity audit-TODO items (#1–#4) are fixed; conftest isolation (`_guard_backend_runtime_state`) prevents cross-test env-var leakage; scikit-build editable finder patch is correct and well-documented. Outstanding:

- **`test_boozersurface_jax.py:4773-4805, 7457`** — `atol=1e-14` LS outputs; below the documented cross-machine `sdofs_inf ≤ 1e-11` floor; will likely fail intermittently on alternate macOS hardware.
- **`test_stage2_jax.py:188-189`** — `_STAGE2_TARGET_SCALAR_VALUE_ATOL = 1e-18` is below float64 noise floor for any non-trivial sum.
- **`np.random.seed(...)` at 4 sites** (`test_boozersurface_jax.py:7809, 7862`, `test_sampling_jax_item22.py:311, 338`) leak global RNG.
- **5 residual re-export `is`-identity tests** carry routing docstrings now but add no parity coverage; keep or delete is a style call.

**Severity:** MEDIUM (tests).
**Source:** `08_review_tests.md`.

### Theme 6 — Alias-only ports leak the "JAX" suffix without porting anything

OpenMemory flag confirmed in this audit: `B2EnergyJAX = B2Energy` (`field/force.py:1320`) and `LpCurveForceJAX = LpCurveForce` (`field/force.py:2284`). These are public lazy exports that claim a JAX port but are identity-aliases. Out of scope for this review (in `force.py`, not `*_jax*.py`), but flagged across multiple reports.

**Severity:** LOW (deceptive API surface).
**Source:** `04_review_field.md` and `07_review_obj_solve.md` cross-references; `06_review_geo_big.md`.

---

## B. Severity-ranked unified findings

Each finding cites the source report (R03–R08) so the underlying file:line evidence is one click away.

### BLOCKER

#### B-1. `jax_core/` is not actually `simsoptpp`-free
- **Category:** simsopt-convention.
- **Source:** R03 §C-01.
- **Files:** 9 cross-package imports in `jax_core/{curve_geometry,magnetic_axis_helpers,surface_fourier,surface_henneberg,objectives_flux,tracing}.py`. Compound: `surface_fourier_jax.py:2763-2768` and `boozer_residual_jax.py:476-510` deferred imports.
- **Fix:** Migrate `*_pure` curve/surface kernels DOWN into `jax_core/`; convert any remaining cross-package callers to host-NumPy boundaries; remove the lazy-import shims.

### HIGH

#### H-1. `while_loop` reverse-mode unsupported in 6 integrators (R03 §B-01)
See Theme 2.

#### H-2. Axis-convention split between `[p, j, l]` and `[p, l, j]` (R03 §B-02)
- **Category:** simsopt-convention.
- **Source:** R03 §B-02.
- **Files:** `jax_core/analytic_pure_fields.py:331-340` (`toroidal_dB`, `toroidal_dA`, `poloidal_dB`) use `[p, l, j]` (component-first); `mirror_dB` (line 644-645) uses `[p, j, l]` (the CLAUDE.md default). Other component-first divergences in `jax_core/dipole_field.py:246-247` and `jax_core/wireframe.py:27-40`.
- **Why:** Each divergence is documented per-file and intentional (matches C++ oracle storage), but a top-down reader of CLAUDE.md will pull the wrong axes from half the kernels.
- **Fix:** Rename storage-divergent functions to `*_cpu_oracle_order` OR add an unambiguous one-line axis docstring on every dB-producing kernel.

#### H-3. `InterpolatedBoozerFieldFrozenState` is `frozen=True` but mutates `specs` dict in place (R03 §B-03)
- **Category:** simsopt-convention (IMMUTABLE) + JAX best-practice (pytree invariants).
- **Source:** R03 §B-03, R04 MEDIUM-10.
- **File:** `jax_core/interpolated_boozer_field.py:178-208` + `field/boozermagneticfield_jax.py:1473-1476`.
- **Fix:** Either drop `frozen=True` (the class isn't actually immutable) or move the lazy `specs` dict off the dataclass onto the wrapper (`InterpolatedBoozerFieldJAX._lazy_specs`).

#### H-4. `jnp.linalg.eig` on a 2x2 forces host roundtrip every call (R03 §B-04)
- **Category:** JAX best-practice (transfer guard).
- **Source:** R03 §B-04.
- **File:** `jax_core/magnetic_axis_helpers.py:598`.
- **Fix:** Use a closed-form 2x2 eigenvalue (`λ = (tr ± √(tr² − 4·det)) / 2`); for non-Hermitian inputs handle complex roots explicitly.

#### H-5. `_make_kernel` LRU cache evicts equivalent kernels (R03 §B-05)
- **Category:** JAX best-practice (compile cache).
- **Source:** R03 §B-05.
- **File:** `jax_core/biotsavart.py::_make_kernel`.
- **Fix:** Drop `jax.default_backend()` from the cache key; XLA already specializes by platform; the Python closure reuse is the only goal.

#### H-6. `SpecBackedBiotSavartJAX.x.setter` writes a free-DOF vector into `_dofs.full_x` (R04 HIGH-1)
- **Category:** simsopt-convention (Optimizable contract).
- **Source:** R04 HIGH-1.
- **File:** `field/biotsavart_jax_backend.py:496-523`.
- **Why:** Latent shape mismatch the moment ANY DOF is fixed. Today no spec-backed caller exercises the bug.
- **Fix:** Decide and document whether `BiotSavartSpec.coil_dofs` is the free vector or the full vector; route the setter through the matching accessor; add `assert coil_dofs.shape[0] == self.dof_size` at the top of `_set_coil_dofs`.

#### H-7. `BiotSavartJAX` uniform-`CurveXYZFourier` fast path never engaged (R04 HIGH-2)
- **Category:** JAX best-practice (perf) + documentation.
- **Source:** R04 HIGH-2.
- **File:** `field/biotsavart_jax_backend.py:1004-1011` (advertised), `:1092-1152` (introspection), `:1333-1397` (`_coil_arrays_in_order_from_dofs`), `:1559-1569` (`coil_set_spec`).
- **Why:** The class docstring at lines 1004-1011 promises a major perf optimization that the hot path `coil_set_spec()` bypasses.
- **Fix:** Either wire the fast path into `coil_set_spec()` so steady-state `B()`/`dB_by_dX()`/`B_vjp()` use it, or remove the dead introspection code and reflect reality in the docstring.

#### H-8. `BoozerRadialInterpolantJAX.as_dict` skips `super().as_dict()` (R04 HIGH-9)
- **Category:** simsopt-convention (serialization).
- **Source:** R04 HIGH-9.
- **File:** `field/boozermagneticfield_jax.py:913-924`.
- **Fix:** Replace the manual metadata block with `d = super().as_dict(serial_objs_dict)`; append JAX-specific keys after.

#### H-9. `InterpolatedFieldJAX.dB_by_dX` falls through to C++ trampoline (R04 HIGH-22)
- **Category:** simsopt-convention (API parity).
- **Source:** R04 HIGH-22.
- **File:** `field/interpolated_field_jax.py:21-27`.
- **Fix:** Override `dB_by_dX()` at the Python level with an explicit `RuntimeError("InterpolatedFieldJAX does not expose dB_by_dX in Cartesian coordinates. Use the source field directly, or call GradAbsB() for the physical gradient table.")`.

#### H-10. L-BFGS-B RESTART task path unimplemented in 4 of 5 failure modes (R06 #1)
See Theme 4.

#### H-11. Diagnostic dense LU computed on every successful exact-Newton solve (R06 #2)
- **Category:** JAX best-practice (perf) + simsopt-convention (exact-mode no-fallback contract).
- **Source:** R06 #2.
- **File:** `boozersurface_jax.py:5757-5762`, `:5836`.
- **Why:** `jax.scipy.linalg.lu(J)` runs whenever `jacobian_available` is true; stored as `res["PLU"]`. The runtime contract still holds (the dense LU is never read for adjoint solves), but it is wall-clock overhead and the public `dense_linear_solve_factors_available = True` is misleading.
- **Fix:** Gate the LU on a debug flag or `verbose=True`; rename or document `dense_linear_solve_factors_available` to mean "diagnostic factor present, not runtime-load-bearing".

#### H-12. `_lbfgsb_ddot` Python-unrolled `lax.cond` skip-zero loop (R06 #3)
- **Category:** JAX best-practice (trace size).
- **Source:** R06 #3.
- **File:** `optimizer_jax_private/_lbfgsb_scipy.py:340-351`.
- **Why:** O(n) HLO ops per call site, 5+ call sites per iteration, 2*n² inside `lbfgsb_matupd`. For n=500 this explodes compile time and trace size.
- **Fix:** `jnp.sum(jnp.where(x * y != 0.0, x * y, 0.0))` reproduces the skip-zero semantics with O(1) HLO emit cost. Optionally retain a "parity vs. fast" lane toggle for the byte-identity contract.

#### H-13. `_lbfgsb_initial_state_kernel` and `_lbfgsb_mainlb_kernel` re-jit per call (R06 #4)
- **Category:** JAX best-practice (compile cache).
- **Source:** R06 #4.
- **File:** `optimizer_jax_private/_lbfgs.py:93-130`.
- **Fix:** Route through `_cached_private_solver`, keyed on `(n, m, maxls, ftol, gtol)`. Existing `lru_cache(maxsize=128)` on LM/Newton-polish/exact-Newton runners is the template.

#### H-14. CLAUDE.md M5 adapter description out-of-date vs implementation (R07 #5)
See Theme 3.

#### H-15. No `donate_argnums` on hot custom-VJP scalar (R07 #2)
- **Category:** JAX best-practice (memory).
- **Source:** R07 #2.
- **File:** `surfaceobjectives_jax.py:4492` (custom-VJP scalar), `:4084` (`compiled_value_and_grad_for`).
- **Fix:** Add `donate_argnums=(0,)` so XLA can reuse the coil-dof buffer on the hot single-stage path.

#### H-16. Condition-estimator comment ambiguity in `_traceable_solve_plu_linearization` (R07 #1)
- **Category:** simsopt-convention (documentation).
- **Source:** R07 #1.
- **File:** `surfaceobjectives_jax.py:3219-3231`.
- **Why:** The comment claims "only reached for LS lane" while the function is also called by forward warm-start solves; rationale is correct (LS Hessian IS symmetric by construction) but the prose mixes "only reached for LS" with "regardless of transpose".
- **Fix:** Tighten the comment to "the LS Hessian is exactly symmetric by construction; condition number is the same for native and transposed matrices".

#### H-17. Latent dead fallback to live solver in `_traceable_solve_hessian_linearization` (R07 #4)
- **Category:** simsopt-convention (traceable adjoint contract).
- **Source:** R07 #4.
- **File:** `surfaceobjectives_jax.py:3085-3103`.
- **Why:** If `linear_solve_factors is None`, helper falls back to `_solve_hessian_least_squares_system_with_status` (LIVE solver inside jit). Currently unreachable for the LS lane (factors are always populated) but a correctness trap if a future change passes `factors=None`.
- **Fix:** Replace the live-solver fallback with `_traceable_adjoint_gradient_or_nan` (NaN-emission) so the failure mode is explicit.

#### H-18. `_traceable_runtime_entry_cache` cross-`bs_jax` lifecycle is non-obvious (R07 #3)
- **Category:** documentation.
- **Source:** R07 #3.
- **File:** `surfaceobjectives_jax.py:4194-4196`.
- **Fix:** Add a one-paragraph docstring at the cache-slot declaration documenting that the slot lives on `booz_jax` but the cache key includes `bs_jax._coil_dof_state_token`; document the invariant that `_advance_solver_generation` nulls the slot.

### MEDIUM (selected — full list in per-area reports)

- **M-1.** `boozer_residual_jacobian_composed` uses `jax.jacfwd`; `jacrev` would be cheaper for typical `n_res ≫ n_x`. (R05 M-1; `boozer_residual_jax.py:743-766`.)
- **M-2.** Divergent local `_as_jax_float64` / `_as_runtime_float64` in `surface_fourier_jax.py:100-108` strips the `reference` keyword, breaking GPU-residency preservation upheld by the canonical `jax_core._math_utils` versions. (R05 M-2.)
- **M-3.** `framedcurve_jax` missing public methods: `frame_twist`, `dframe_twist_by_dcoeff_vjp`, `rotated_frame_dcoeff_vjp`, `rotated_frame_dash_dcoeff_vjp`. Polymorphic callers AttributeError. (R05 M-3.)
- **M-4.** `LinkingNumberJAX.J()` returns `jax.Array` scalar; CPU `LinkingNumber.J()` returns Python int. Type drift across CPU/JAX boundary. (R05 M-4; `geo/curveobjectives_jax.py:365-388`.)
- **M-5.** `BoozerAnalyticJAX` and `InterpolatedBoozerFieldJAX` lack `as_dict`/`from_dict`. Parity with CPU sibling (which also lacks them) is preserved, but the JAX port has explicit `_frozen_state_to_host`/`_frozen_state_from_host` and should round-trip. (R04 MEDIUM-11.)
- **M-6.** Freeze helpers in `boozermagneticfield_jax.py:219-310` use `jnp.asarray` instead of the SSOT `_as_jax_float64`. Inconsistent with `transfer_guard("disallow")` discipline elsewhere. (R04 MEDIUM-12.)
- **M-7.** `BiotSavartJAX` does not implement `set_points_cart`/`set_points_cyl`/`B_cyl`/`GradAbsB_cyl`, blocking composition with `InterpolatedFieldJAX`. (R04 MEDIUM-4.)
- **M-8.** `_per_coil_unit_field` Python-loops over coils; bypasses coil-axis sharding on CUDA. (R04 MEDIUM-3.)
- **M-9.** `interpolated_field_jax.py` `dB_by_dX` NaN-gradient gotcha through `jnp.where`; classical "double-where" pattern needed (JBP-17.1). (R04 MEDIUM-23.)
- **M-10.** `dommaschk_jax._toroidal_baseline_B_dB` allocates fresh `ToroidalFieldJAX` per call. (R04 MEDIUM-20.)
- **M-11.** `dipole_field_jax._expand_symmetries` 90-line pure-NumPy copy of `_dipole_fields_from_symmetries` in `magneticfieldclasses.py`. (R04 LOW-19.)
- **M-12.** `wireframe_optimization_jax.bnorm_obj_matrices_jax` falls back to CPU `ext_field.B()` (R07 #7; `wireframe_optimization_jax.py:673-677`).
- **M-13.** LM lane is damped Gauss–Newton, not trust-region — `delta` is convergence-proxy only despite the "trust-region" naming. (R06 #5.)
- **M-14.** `accepted_step_callback` cannot abort L-BFGS-B (no `STOP_CALLB` write path). (R06 #6.)
- **M-15.** `_lbfgsb_scipy.py` LS `linear_solve_factors` device buffers built twice in `dense-plu-shared` callback. (R06 #8.)
- **M-16.** `_traceable_solve_hessian_linearization` exact-lane `compute_G_from_currents` cotangent asymmetry between LS (adds) and exact (doesn't); correct in practice but implicit. (R06 #7.)
- **M-17.** `SquaredFluxJAX` field-points version check uses `==` on a monotonic counter; clone+`np.array_equal` would be stricter. (R07 #6; `fluxobjective_jax.py:225, 359-365`.)
- **M-18.** `stage2_target_objective_jax._fixed_curve_penalty` Python double-loop at construction time; O(n_coil²) tracing for large fixed-coil counts. (R07 #8.)
- **M-19.** `make_traceable_objective` docstring does not state host-input rejection contract. (R07 #9.)
- **M-20.** `compute_G_from_currents` drops current signs via `jnp.abs`; document the sign contract. (R05 L-3; `label_constraints_jax.py:60-61`.)
- **M-21.** Test `_STAGE2_TARGET_SCALAR_VALUE_ATOL=1e-18` is below float64 noise floor. (R08 #1.)
- **M-22.** Tests `atol=1e-14` on LS solver outputs likely cross-machine fragile per CLAUDE.md `sdofs_inf ≤ 1e-11`. (R08 #2.)
- **M-23.** Global `np.random.seed` in 4 test sites leaks RNG state. (R08 #5.)
- **M-24.** Public-API alias-only ports: `B2EnergyJAX = B2Energy`, `LpCurveForceJAX = LpCurveForce` in `field/force.py:1320, 2284`. Deceptive `JAX` suffix.

### LOW / NIT — see per-area reports

See `03_review_jax_core.md` §B-06 to §B-09 + §A-06 + §A-07; `04_review_field.md` LOW-7/-8/-13/-14/-16/-18/-26/-27/-29 + NIT-1/-2/-15/-21/-25; `05_review_geo_small.md` L-1 to L-8; `06_review_geo_big.md` #9, #10; `07_review_obj_solve.md` #10; `08_review_tests.md` "5 residual is-identity tests" list.

---

## C. Per-module verdict matrix

| Module | LOC | Verdict | BLOCKER | HIGH | MEDIUM | Notes |
|---|---:|---|:---:|:---:|:---:|---|
| `jax_core/*` (37 files) | 21,709 | NOT READY | 1 | 5 | 5 | Layering violation; `while_loop` rev-mode; axis split; frozen-dict mutation; host-callback eig; LRU dup. |
| `field/biotsavart_jax_backend.py` | 2,037 | NEEDS WORK | 0 | 2 | 5 | Latent setter bug; advertised fast path unused; composition gaps. |
| `field/boozermagneticfield_jax.py` | 1,592 | NEEDS WORK | 0 | 1 | 3 | `as_dict` skips super; freeze helpers bypass SSOT; missing `as_dict`/`from_dict` on 2 of 3 classes. |
| `field/interpolated_field_jax.py` | 337 | NEEDS WORK | 0 | 1 | 2 | `dB_by_dX` trampoline; NaN-gradient gotcha; B_cyl cache divergence. |
| Other `field/*_jax*.py` | ~1,200 | PASS | 0 | 0 | ~6 | DRY, perf, minor parity gaps. |
| `geo/boozersurface_jax.py` | 6,104 | PASS WITH CAVEATS | 0 | 2 | 4 | Diagnostic LU on exact success; `linear_solve_factors` double-build; should be split. |
| `geo/optimizer_jax.py` + `optimizer_jax_private/*` | 5,725 | NEEDS WORK | 0 | 3 | 2 | L-BFGS-B RESTART missing; ddot unrolled; re-jit per call. |
| `geo/surfaceobjectives_jax.py` | 5,935 | PASS WITH NOTES | 0 | 5 | 3 | M5 doc drift; no donate_argnums; cache lifecycle doc gap; dead fallback. |
| `geo/surface_fourier_jax.py` | 2,768 | PASS | 0 | 0 | 2 | `_as_runtime_float64` SSOT drift; jacfwd choice ok for typical sizes. |
| Other `geo/*_jax*.py` | ~2,700 | PASS | 0 | 0 | 4 | `framedcurve_jax` missing methods; `LinkingNumberJAX` type drift; jacfwd/jacrev choice. |
| `objectives/fluxobjective_jax.py` | 433 | PASS | 0 | 0 | 1 | Best-in-class M2; minor stricter drift check available. |
| `objectives/integral_bdotn_jax.py` | 156 | PASS | 0 | 0 | 0 | Three definitions match CPU; `inf_with_nan_jvp` for `local` singular. |
| `objectives/stage2_target_objective_jax.py` | 1,292 | PASS | 0 | 0 | 1 | Construction-time O(n_coil²) for fixed-coil penalty. |
| `solve/permanent_magnet_optimization_jax.py` | 798 | PASS | 0 | 0 | 0 | Thin adapter pattern correctly applied. |
| `solve/wireframe_optimization_jax.py` | 885 | PASS WITH NOTES | 0 | 0 | 1 | CPU `ext_field.B()` fallback seam. |
| `backend.py` + `backend/runtime.py` | 1,878 | PASS | 0 | 0 | 0 | Read-at-call-time with cache; legacy env vars resolved correctly; no implicit JAX configuration at import. |
| `tests/**` (~100 files) | — | PASS | 0 | 0 | 3 | Tier-1 tautologies eliminated; 3 NEEDS-WORK items (sub-noise atol, RNG seed leakage, 5 is-identity tests). |

---

## D. Prioritized remediation list (recommended sequence)

**Tier 1 — must fix this cycle:**

1. **B-1 / Theme 1** — break the `jax_core/` → `simsopt.{geo,field,objectives}` import cycle. Migrate `*_pure` kernels into `jax_core/` permanently. Track the cycle in CLAUDE.md as a known issue until fixed.
2. **H-10 / Theme 4** — L-BFGS-B RESTART task path: add `info != 0` gates after `cauchy`/`formk`/`cmprlb`/`subsm`; add `col != 0` branch on `lnsrlb` failure; write `task = RESTART, task_msg = NO_MSG` in all 5 paths.
3. **H-14 / Theme 3** — update CLAUDE.md "M5 adapter pattern" section to match implementation: pure JAX from solved state, CPU spec/DOF as source-of-truth.
4. **H-1 / H-2 / Theme 2** — either replace `while_loop` with `scan`+mask (in 6 integrators) or document forward-mode-only support. Add unambiguous axis convention docstrings to every `dB`-producing kernel.

**Tier 2 — high-leverage fixes this quarter:**

5. **H-6** — `SpecBackedBiotSavartJAX.x.setter` shape-mismatch latent bug.
6. **H-7** — wire the `CurveXYZFourier` fast path into `coil_set_spec()` OR delete the dead introspection code.
7. **H-9** — `InterpolatedFieldJAX.dB_by_dX` explicit Python-side error.
8. **H-11** — gate diagnostic dense LU on `verbose=True`; rename `dense_linear_solve_factors_available` field on exact-lane result dict.
9. **H-12** — replace `_lbfgsb_ddot` Python-unrolled loop with `jnp.sum(jnp.where(...))`.
10. **H-13** — route L-BFGS-B kernels through `_cached_private_solver`.
11. **H-15** — add `donate_argnums=(0,)` on hot custom-VJP scalar.
12. **H-3 / H-4 / H-5** — fix `frozen=True` mutable dict; closed-form 2x2 eig; drop platform key from `_make_kernel` LRU.
13. **H-8** — `BoozerRadialInterpolantJAX.as_dict` delegate to super.

**Tier 3 — documentation, testing, polish:**

14. **H-16 / H-17 / H-18** — three documentation/contract tightenings in `surfaceobjectives_jax.py`.
15. **M-22 / M-21** — relax `atol=1e-14`/`1e-18` test thresholds OR add `pytest.mark.machine_pinned`.
16. **M-23** — migrate `np.random.seed` → `np.random.default_rng` or `parity_rng` at 4 test sites.
17. **M-24** — delete or rename `B2EnergyJAX`/`LpCurveForceJAX` alias-only ports.
18. **M-2 / M-3 / M-4** — `framedcurve_jax` API parity gap; `LinkingNumberJAX` return-type drift; `surface_fourier_jax` SSOT helpers.
19. **M-9** — apply JBP-17.1 "double-where" trick to interpolated-field NaN-gradient gotcha.
20. **All remaining LOW/NIT items** — see per-area reports.

---

## E. Cross-reference to underlying reports

| Report | LOC | Coverage |
|---|---:|---|
| `01_simsopt_convention_baseline.md` | 695 | 15 normative simsopt conventions (SC-1 to SC-15) + quick checklist. |
| `02_jax_best_practices_baseline.md` | 924 | 20 normative JAX rules (JBP-1 to JBP-20) + 95-line reviewer checklist. All citations from `docs.jax.dev` fetched on 2026-05-16. |
| `03_review_jax_core.md` | 1,066 | 37 kernel files / 21,709 LOC. 1 BLOCKER, 5 HIGH, 8 MEDIUM, 6 LOW. |
| `04_review_field.md` | 469 | 16 adapter files / 5,189 LOC. 0 BLOCKER, 4 HIGH, 11 MEDIUM, 12 LOW, 4 NIT. |
| `05_review_geo_small.md` | 633 | 8 small geo modules / 5,696 LOC. 0 BLOCKER, 0 HIGH, 4 MEDIUM, 4 LOW. |
| `06_review_geo_big.md` | 913 | `boozersurface_jax.py` + `optimizer_jax.py` + `optimizer_jax_private/*` / 16,399 LOC. 0 BLOCKER, 3 HIGH, 4 MEDIUM, 2 LOW. |
| `07_review_obj_solve.md` | 414 | M5 objective wrappers + solve adapters + backend / 10,603 LOC. 0 BLOCKER, 5 HIGH, 4 MEDIUM. |
| `08_review_tests.md` | 399 | ~100 test files. 0 BLOCKER, 0 HIGH, 3 NEEDS-WORK; 10 of 25 prior audit TODOs explicitly addressed, 6 partially. |

---

## F. Positive observations worth preserving

These are the parts of the port that *do not need to change*. They establish patterns the team should keep applying:

1. **Bundle-cache key tokenization** in `surfaceobjectives_jax.py` correctly separates structural signatures (layout, dtype, shape) from content tokens (`_traceable_solve_state_token`, `_coil_dof_state_token`). Uses `is` identity for callable signatures, NOT `id(callable)` or user-defined equality. (R06 §A.1.h, R07 §D.)
2. **`SquaredFluxJAX`** is the canonical M2 wrapper: fixed-surface JIT closure, surface-DOF fingerprint, field-points drift detection, strict native contract, ObjectiveFailure on singular gradient. (R07 verdict.)
3. **M5 IFT adjoint sign discipline** matches CPU siblings term-for-term: `BoozerResidual = direct − adj^T ∂g/∂coils`, `Iotas = − adj^T ∂g/∂coils`, `NonQuasiSymmetricRatio = direct − adj^T ∂g/∂coils`, `MajorRadius = − adj^T ∂g/∂coils`. (R07 §M5.)
4. **Result-dict schema completeness**: 14 schema definitions in `boozersurface_jax.py:267-436`; unified failure-path constructor mirrors success-path keys exactly. All `int()`/`bool()` JAX→host boundary conversions applied consistently. (R06 §A.1.c, §A.1.k.)
5. **Exact-lane operator-only runtime** correctly maintained: `_normalize_solver_options` strips `optimizer_backend` from exact path; `_build_runtime_linear_solve_callbacks` exposes only matvec/solve callbacks; the diagnostic dense LU never feeds the runtime adjoint. (R06 §A.1.d, §A.1.e.)
6. **Conftest infrastructure**: `_guard_backend_runtime_state` snapshots 24 env vars + 5 JAX config knobs; scikit-build editable finder patch is correct; `parity_lane` fixture parametrizes CPU/GPU lanes; `ensure_gpu_determinism_xla_flag` prepends `--xla_gpu_deterministic_ops` for CUDA parity. (R08 "Conftest patches audit".)
7. **Validation-ladder SSOT discipline**: nearly every parity-asserting test imports `parity_ladder_tolerances` from `benchmarks/validation_ladder_contract.py`; inline `rtol=...` literals are confined to closed-form / analytic / per-test-specific cases. (R08 "Positive notes".)
8. **No `id()` calls in cache-key construction** anywhere in the M5 wrappers or BoozerSurfaceJAX runtime; `object.__hash__` and `is` identity are used correctly per CLAUDE.md contract. (R07 §D "id() discipline".)
9. **`backend.py` / `backend/runtime.py`**: legacy `STAGE2_BACKEND`/`SIMSOPT_JAX_BACKEND` env vars are resolved; primary `SIMSOPT_BACKEND`/`SIMSOPT_JAX_PLATFORM` win; cache invalidation via `invalidate_backend_cache()` is explicit; no implicit `jax.config.update` at import time. (R07 §"Backend selection".)
10. **PRNG hygiene in `jax_core/`** is excellent: no global RNG anywhere; every random source explicitly takes a `jax.random.PRNGKey`. (R03 "Positive observations".)
11. **`custom_jvp`/`custom_vjp` discipline** sound where used; `inf_with_nan_jvp` correctly surfaces `inf` value + `nan` cotangent for `local` integral-BdotN singular branch. (R03; R07 `integral_bdotn_jax.py`.)
12. **Sharding** uses modern `NamedSharding` + `Mesh` in `jax_core/sharding.py`; no `pmap`. (R03.)

---

## G. Summary

| Metric | Count |
|---|---:|
| LOC reviewed (source) | ~52,671 |
| LOC reviewed (tests) | ~30,000 |
| Files reviewed (source) | ~70 |
| Files reviewed (tests) | ~100 |
| Reports produced | 8 (5,513 lines total) |
| BLOCKER findings | 1 |
| HIGH findings | 18 |
| MEDIUM findings | ~30 |
| LOW + NIT findings | ~40 |
| Modules with `PASS` verdict | 11 |
| Modules with `PASS WITH NOTES` | 4 |
| Modules with `NEEDS WORK` | 4 |
| Modules with `NOT READY` | 1 (`jax_core/` aggregate) |

**Net assessment:** the port is production-ready as a research lane but has one architectural BLOCKER (jax_core layering) and four discrete HIGH-correctness bugs (L-BFGS-B RESTART, `while_loop` rev-mode, `SpecBackedBiotSavartJAX` setter, `InterpolatedFieldJAX.dB_by_dX` trampoline) that should be triaged this cycle. The HIGH performance items (diagnostic dense LU on exact success, `_lbfgsb_ddot` unrolled loop, per-call re-jit) are leverage opportunities with no correctness cost. JAX best-practice discipline is strong overall; the team is correctly using `custom_vjp` for IFT, `lax.stop_gradient` on solved state, `is`-identity (not `id()`) for callable signatures, and `jax.tree_util.register_dataclass` for frozen specs. The main JAX-side gap is the absence of `donate_argnums` on hot paths and the four `while_loop`-based integrators that silently disable reverse-mode AD.

Test discipline is good and improving — Tier-1 oracle-lint tautologies are eliminated, parity-ladder tolerances are SSOT'd, and conftest isolation is robust. The three residual test-discipline items (sub-noise atol, RNG leakage, `is`-identity routing tests) are routine cleanups.

---

*End of synthesis. For specific findings with file:line evidence and concrete fix suggestions, consult the underlying numbered reports in this directory.*
